import Mathlib

namespace NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l314_31451

/-- A line in 3D space represented by a point and a direction vector. -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines intersect in 3D space. -/
def intersect (l1 l2 : Line3D) : Prop :=
  ∃ t s : ℝ, l1.point + t • l1.direction = l2.point + s • l2.direction

/-- Two lines are parallel if their direction vectors are scalar multiples of each other. -/
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Two lines are skew if they are neither intersecting nor parallel. -/
def skew (l1 l2 : Line3D) : Prop :=
  ¬(intersect l1 l2) ∧ ¬(parallel l1 l2)

/-- Theorem: If two lines in 3D space do not intersect, then they are either parallel or skew. -/
theorem non_intersecting_lines_parallel_or_skew (l1 l2 : Line3D) :
  ¬(intersect l1 l2) → parallel l1 l2 ∨ skew l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l314_31451


namespace NUMINAMATH_CALUDE_line_intersects_circle_l314_31407

/-- The line y = x + 1 intersects the circle x^2 + y^2 = 1 -/
theorem line_intersects_circle :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let circle : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = 1
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 1
  let distance_to_line : ℝ := |1| / Real.sqrt 2
  distance_to_line < radius →
  ∃ p : ℝ × ℝ, line p.1 = p.2 ∧ circle p :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l314_31407


namespace NUMINAMATH_CALUDE_bus_system_daily_passengers_l314_31424

def total_people : ℕ := 109200000
def num_weeks : ℕ := 13
def days_per_week : ℕ := 7

theorem bus_system_daily_passengers : 
  total_people / (num_weeks * days_per_week) = 1200000 := by
  sorry

end NUMINAMATH_CALUDE_bus_system_daily_passengers_l314_31424


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_for_given_polynomial_l314_31425

/-- Given a quadratic polynomial ax^2 + bx + c, 
    returns the sum of the reciprocals of its roots -/
def sum_of_reciprocal_roots (a b c : ℚ) : ℚ := -b / c

theorem sum_of_reciprocal_roots_for_given_polynomial : 
  sum_of_reciprocal_roots 7 2 6 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_for_given_polynomial_l314_31425


namespace NUMINAMATH_CALUDE_max_volume_box_l314_31487

/-- The volume function for the open-top box -/
def volume (a x : ℝ) : ℝ := x * (a - 2*x)^2

/-- The theorem stating the maximum volume and optimal cut length -/
theorem max_volume_box (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (a/2) ∧
    (∀ y ∈ Set.Ioo 0 (a/2), volume a x ≥ volume a y) ∧
    x = a/6 ∧
    volume a x = 2*a^3/27 :=
sorry

end NUMINAMATH_CALUDE_max_volume_box_l314_31487


namespace NUMINAMATH_CALUDE_intersecting_lines_length_l314_31447

/-- Given a geometric configuration with two intersecting lines AC and BD, prove that AC = 3√19 -/
theorem intersecting_lines_length (O A B C D : ℝ × ℝ) (x : ℝ) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist O A = 5 →
  dist O C = 11 →
  dist O D = 5 →
  dist O B = 6 →
  dist B D = 9 →
  x = dist A C →
  x = 3 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_length_l314_31447


namespace NUMINAMATH_CALUDE_root_implies_k_value_l314_31430

theorem root_implies_k_value (k : ℝ) : 
  (6 * ((-25 - Real.sqrt 409) / 12)^2 + 25 * ((-25 - Real.sqrt 409) / 12) + k = 0) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l314_31430


namespace NUMINAMATH_CALUDE_working_mom_time_allocation_l314_31467

theorem working_mom_time_allocation :
  let total_hours_in_day : ℝ := 24
  let work_hours : ℝ := 8
  let daughter_care_hours : ℝ := 2.25
  let household_chores_hours : ℝ := 3.25
  let total_activity_hours : ℝ := work_hours + daughter_care_hours + household_chores_hours
  let percentage_of_day : ℝ := (total_activity_hours / total_hours_in_day) * 100
  percentage_of_day = 56.25 := by
sorry

end NUMINAMATH_CALUDE_working_mom_time_allocation_l314_31467


namespace NUMINAMATH_CALUDE_sum_of_squares_not_perfect_square_l314_31477

def sum_of_squares (n : ℕ) (a : ℕ) : ℕ := 
  (2*n + 1) * a^2 + (2*n*(n+1)*(2*n+1)) / 3

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :
  ∀ a : ℕ, ¬(is_perfect_square (sum_of_squares n a)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_perfect_square_l314_31477


namespace NUMINAMATH_CALUDE_tara_bank_balance_l314_31485

/-- Calculates the balance after one year given an initial amount and an annual interest rate. -/
def balance_after_one_year (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate)

/-- Theorem stating that with an initial amount of $90 and a 10% annual interest rate, 
    the balance after one year will be $99. -/
theorem tara_bank_balance : 
  balance_after_one_year 90 0.1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_tara_bank_balance_l314_31485


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l314_31497

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.2)
  (h4 : game.targetRuns = 252) :
  requiredRunRate game = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l314_31497


namespace NUMINAMATH_CALUDE_speed_conversion_l314_31495

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 50

/-- Theorem: Converting 50 mps to kmph equals 180 kmph -/
theorem speed_conversion : speed_mps * mps_to_kmph = 180 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l314_31495


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l314_31454

theorem rain_probability_tel_aviv (p : ℝ) (n k : ℕ) (h_p : p = 1/2) (h_n : n = 6) (h_k : k = 4) :
  (n.choose k) * p^k * (1-p)^(n-k) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l314_31454


namespace NUMINAMATH_CALUDE_hydrochloric_acid_moles_l314_31406

/-- Represents the chemical reaction between Sodium bicarbonate and Hydrochloric acid -/
structure ChemicalReaction where
  sodium_bicarbonate : ℝ  -- moles of Sodium bicarbonate
  hydrochloric_acid : ℝ   -- moles of Hydrochloric acid
  sodium_chloride : ℝ     -- moles of Sodium chloride produced

/-- Theorem stating that when 1 mole of Sodium bicarbonate reacts to produce 1 mole of Sodium chloride,
    the amount of Hydrochloric acid used is also 1 mole -/
theorem hydrochloric_acid_moles (reaction : ChemicalReaction)
  (h1 : reaction.sodium_bicarbonate = 1)
  (h2 : reaction.sodium_chloride = 1) :
  reaction.hydrochloric_acid = 1 := by
  sorry


end NUMINAMATH_CALUDE_hydrochloric_acid_moles_l314_31406


namespace NUMINAMATH_CALUDE_value_of_r_l314_31452

theorem value_of_r (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 49/6 := by
sorry

end NUMINAMATH_CALUDE_value_of_r_l314_31452


namespace NUMINAMATH_CALUDE_prob_queens_or_jacks_l314_31463

/-- The probability of drawing either all three queens or at least 2 jacks from 3 cards in a standard deck -/
theorem prob_queens_or_jacks (total_cards : Nat) (num_queens : Nat) (num_jacks : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_queens = 4)
  (h3 : num_jacks = 4) : 
  (Nat.choose num_queens 3) / (Nat.choose total_cards 3) + 
  (Nat.choose num_jacks 2 * (total_cards - num_jacks) + Nat.choose num_jacks 3) / (Nat.choose total_cards 3) = 290 / 5525 := by
  sorry

end NUMINAMATH_CALUDE_prob_queens_or_jacks_l314_31463


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l314_31423

theorem divisibility_equivalence (a b : ℤ) :
  (29 ∣ 3*a + 2*b) ↔ (29 ∣ 11*a + 17*b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l314_31423


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l314_31418

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  is_pythagorean_triple 5 12 13 ∧
  ¬is_pythagorean_triple 8 12 15 ∧
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 9 40 41 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l314_31418


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l314_31436

theorem inverse_variation_problem (z w : ℝ) (k : ℝ) (h1 : z * Real.sqrt w = k) 
  (h2 : 6 * Real.sqrt 3 = k) (h3 : z = 3/2) : w = 48 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l314_31436


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l314_31492

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + a) * (x + b)) →
  (∀ x, x^2 - 19*x + 88 = (x - c) * (x - d)) →
  a + b + c + d = 32 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l314_31492


namespace NUMINAMATH_CALUDE_continuity_at_3_l314_31453

def f (x : ℝ) := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_3_l314_31453


namespace NUMINAMATH_CALUDE_triangle_third_side_count_l314_31414

theorem triangle_third_side_count : 
  let side1 : ℕ := 8
  let side2 : ℕ := 12
  let valid_third_side (x : ℕ) : Prop := 
    x + side1 > side2 ∧ 
    x + side2 > side1 ∧ 
    side1 + side2 > x
  (∃! (n : ℕ), (∀ (x : ℕ), valid_third_side x ↔ x ∈ Finset.range n) ∧ n = 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_count_l314_31414


namespace NUMINAMATH_CALUDE_abs_inequality_exponential_inequality_l314_31422

-- Problem 1
theorem abs_inequality (x : ℝ) :
  |x - 1| > 2 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
sorry

-- Problem 2
theorem exponential_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) :
  a^(1 - x) < a^(x + 1) ↔ x ∈ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_exponential_inequality_l314_31422


namespace NUMINAMATH_CALUDE_new_species_growth_pattern_l314_31484

/-- Represents the shape of population growth --/
inductive GrowthShape
  | J -- J-shaped growth
  | S -- S-shaped growth

/-- Represents the population growth pattern over time --/
structure PopulationGrowth where
  initialShape : GrowthShape
  finalShape : GrowthShape

/-- Represents a new species entering an area --/
structure NewSpecies where
  enteredArea : Bool

/-- Theorem stating the population growth pattern for a new species --/
theorem new_species_growth_pattern (species : NewSpecies) 
  (h : species.enteredArea = true) : 
  ∃ (growth : PopulationGrowth), 
    growth.initialShape = GrowthShape.J ∧ 
    growth.finalShape = GrowthShape.S :=
  sorry

end NUMINAMATH_CALUDE_new_species_growth_pattern_l314_31484


namespace NUMINAMATH_CALUDE_max_balls_in_specific_cylinder_l314_31412

/-- The maximum number of unit balls that can be placed in a cylinder -/
def max_balls_in_cylinder (cylinder_diameter : ℝ) (cylinder_height : ℝ) (ball_diameter : ℝ) : ℕ :=
  sorry

/-- Theorem: In a cylinder with diameter √2 + 1 and height 8, the maximum number of balls with diameter 1 that can be placed is 36 -/
theorem max_balls_in_specific_cylinder :
  max_balls_in_cylinder (Real.sqrt 2 + 1) 8 1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_specific_cylinder_l314_31412


namespace NUMINAMATH_CALUDE_impossibleToKnowDreamIfDiedAsleep_l314_31496

/-- Represents a person's state --/
inductive PersonState
  | Awake
  | Asleep
  | Dead

/-- Represents a dream --/
structure Dream where
  content : String

/-- Represents a person --/
structure Person where
  state : PersonState
  currentDream : Option Dream

/-- Represents the ability to share dream content --/
def canShareDream (p : Person) : Prop :=
  p.state = PersonState.Awake ∧ p.currentDream.isSome

/-- Represents the event of a person dying while asleep --/
def diedWhileAsleep (p : Person) : Prop :=
  p.state = PersonState.Dead ∧ p.currentDream.isSome

/-- Theorem: If a person died while asleep, it's impossible for others to know their exact dream --/
theorem impossibleToKnowDreamIfDiedAsleep (p : Person) :
  diedWhileAsleep p → ¬(canShareDream p) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleToKnowDreamIfDiedAsleep_l314_31496


namespace NUMINAMATH_CALUDE_chord_length_l314_31455

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) | x + 2*y - 5 + Real.sqrt 5 = 0}
  let chord := circle ∩ line
  (∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_chord_length_l314_31455


namespace NUMINAMATH_CALUDE_min_runs_ninth_game_l314_31465

def runs_5_to_8 : List Nat := [19, 15, 13, 22]

theorem min_runs_ninth_game
  (h1 : (List.sum runs_5_to_8 + x) / 8 > (List.sum runs_5_to_8 + x) / 4)
  (h2 : (List.sum runs_5_to_8 + x + y) / 9 > 17)
  (h3 : y ≥ 19)
  : ∀ z < 19, (List.sum runs_5_to_8 + x + z) / 9 ≤ 17 :=
by sorry

#check min_runs_ninth_game

end NUMINAMATH_CALUDE_min_runs_ninth_game_l314_31465


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l314_31408

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℕ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l314_31408


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l314_31413

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- The theorem stating that if points A(m-1, -3) and B(2, n) are symmetric
    with respect to the origin, then m + n = 2 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l314_31413


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_8_l314_31438

/-- Represents a bag of cards -/
def Bag := Finset Nat

/-- Creates a bag with cards numbered from 0 to 5 -/
def createBag : Bag := Finset.range 6

/-- Calculates the probability of selecting two cards with sum > 8 -/
def probSumGreaterThan8 (bag1 bag2 : Bag) : ℚ :=
  let allPairs := bag1.product bag2
  let favorablePairs := allPairs.filter (fun p => p.1 + p.2 > 8)
  favorablePairs.card / allPairs.card

/-- Main theorem: probability of sum > 8 is 1/12 -/
theorem prob_sum_greater_than_8 :
  probSumGreaterThan8 createBag createBag = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_8_l314_31438


namespace NUMINAMATH_CALUDE_partitionWays_10_l314_31499

/-- The number of ways to partition n ordered elements into 1 to n non-empty subsets,
    where the elements within each subset are contiguous. -/
def partitionWays (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.choose (n - 1) k)

/-- Theorem stating that for 10 elements, the number of partition ways is 512. -/
theorem partitionWays_10 : partitionWays 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_partitionWays_10_l314_31499


namespace NUMINAMATH_CALUDE_bear_food_per_day_l314_31404

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_eaten : ℝ := 15

/-- The number of weeks -/
def weeks : ℝ := 3

/-- The number of days in a week -/
def days_per_week : ℝ := 7

/-- Theorem: A bear eats 90 pounds of food per day -/
theorem bear_food_per_day :
  (victor_weight * victors_eaten) / (weeks * days_per_week) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bear_food_per_day_l314_31404


namespace NUMINAMATH_CALUDE_equation_solutions_l314_31464

theorem equation_solutions : 
  {x : ℝ | (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 36} = {3, 4} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l314_31464


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l314_31466

theorem infinite_solutions_iff_b_eq_neg_twelve (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l314_31466


namespace NUMINAMATH_CALUDE_parity_of_sum_of_powers_l314_31445

theorem parity_of_sum_of_powers : Even (1^1994 + 9^1994 + 8^1994 + 6^1994) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_sum_of_powers_l314_31445


namespace NUMINAMATH_CALUDE_min_value_problem_l314_31481

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 6) :
  (3 / a + 2 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 6 ∧ 3 / a₀ + 2 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l314_31481


namespace NUMINAMATH_CALUDE_base_12_5_equivalence_l314_31428

def is_valid_base_12_digit (d : ℕ) : Prop := d < 12
def is_valid_base_5_digit (d : ℕ) : Prop := d < 5

def to_base_10_from_base_12 (a b : ℕ) : ℕ := 12 * a + b
def to_base_10_from_base_5 (b a : ℕ) : ℕ := 5 * b + a

theorem base_12_5_equivalence (a b : ℕ) :
  is_valid_base_12_digit a →
  is_valid_base_12_digit b →
  is_valid_base_5_digit a →
  is_valid_base_5_digit b →
  to_base_10_from_base_12 a b = to_base_10_from_base_5 b a →
  to_base_10_from_base_12 a b = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_12_5_equivalence_l314_31428


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l314_31426

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The first line equation: x + ay - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - 1 = 0

/-- The second line equation: ax + 4y + 2 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 2 = 0

theorem parallel_lines_a_equals_two :
  ∃ a : ℝ, (∀ x y, line1 a x y ↔ line2 a x y) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l314_31426


namespace NUMINAMATH_CALUDE_columns_in_first_arrangement_l314_31449

/-- Given a group of people, prove the number of columns formed when 30 people stand in each column. -/
theorem columns_in_first_arrangement 
  (total_people : ℕ) 
  (people_per_column_second : ℕ) 
  (columns_second : ℕ) 
  (people_per_column_first : ℕ) 
  (h1 : people_per_column_second = 32) 
  (h2 : columns_second = 15) 
  (h3 : people_per_column_first = 30) 
  (h4 : total_people = people_per_column_second * columns_second) :
  total_people / people_per_column_first = 16 :=
by sorry

end NUMINAMATH_CALUDE_columns_in_first_arrangement_l314_31449


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l314_31468

/-- Given that 2 is one root of the equation 5x^2 + kx = 4, prove that -2/5 is the other root -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = 2) → 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l314_31468


namespace NUMINAMATH_CALUDE_compound_interest_rate_l314_31475

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (n : ℝ) (CI : ℝ) (r : ℝ) : 
  P = 20000 →
  t = 2 →
  n = 2 →
  CI = 1648.64 →
  (P + CI) = P * (1 + r / n) ^ (n * t) →
  r = 0.04 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l314_31475


namespace NUMINAMATH_CALUDE_app_difference_proof_l314_31421

/-- Calculates the difference between added and deleted apps -/
def appDifference (initial final added : ℕ) : ℕ :=
  added - ((initial + added) - final)

theorem app_difference_proof (initial final added : ℕ) 
  (h1 : initial = 21)
  (h2 : final = 24)
  (h3 : added = 89) :
  appDifference initial final added = 3 := by
  sorry

#eval appDifference 21 24 89

end NUMINAMATH_CALUDE_app_difference_proof_l314_31421


namespace NUMINAMATH_CALUDE_two_digit_puzzle_solution_l314_31432

theorem two_digit_puzzle_solution :
  ∃ (A B : ℕ), 
    A ≠ B ∧ 
    A ≠ 0 ∧ 
    A < 10 ∧ 
    B < 10 ∧ 
    A * B + A + B = 10 * A + B :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_puzzle_solution_l314_31432


namespace NUMINAMATH_CALUDE_line_through_circle_center_l314_31444

/-- A line intersecting a circle -/
structure LineIntersectingCircle where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The line intersects the circle at two points -/
  intersects : True
  /-- The distance between the intersection points -/
  chord_length : ℝ

/-- The theorem stating the value of k for a specific configuration -/
theorem line_through_circle_center (config : LineIntersectingCircle)
    (h1 : config.b = 2)
    (h2 : config.center = (1, 1))
    (h3 : config.radius = Real.sqrt 2)
    (h4 : config.chord_length = 2 * Real.sqrt 2) :
    config.k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l314_31444


namespace NUMINAMATH_CALUDE_quadratic_equation_special_roots_l314_31491

/-- 
Given a quadratic equation x^2 + px + q = 0 with roots D and 1-D, 
where D is the discriminant of the equation, 
prove that the only possible values for (p, q) are (-1, 0) and (-1, 3/16).
-/
theorem quadratic_equation_special_roots (p q D : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = D ∨ x = 1 - D) ∧ 
  D^2 = p^2 - 4*q →
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3/16) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_special_roots_l314_31491


namespace NUMINAMATH_CALUDE_point_relationship_l314_31462

theorem point_relationship (b y₁ y₂ : ℝ) 
  (h1 : y₁ = -(-2) + b) 
  (h2 : y₂ = -(3) + b) : 
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_point_relationship_l314_31462


namespace NUMINAMATH_CALUDE_notebooks_distributed_sang_woo_distribution_l314_31498

theorem notebooks_distributed (initial_notebooks : ℕ) (initial_pencils : ℕ) 
  (remaining_total : ℕ) : ℕ :=
  let distributed_notebooks := 
    (initial_notebooks + initial_pencils - remaining_total) / 4
  distributed_notebooks

theorem sang_woo_distribution : notebooks_distributed 12 34 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_distributed_sang_woo_distribution_l314_31498


namespace NUMINAMATH_CALUDE_median_is_165_l314_31494

/-- Represents the size categories of school uniforms -/
inductive UniformSize
| s150
| s155
| s160
| s165
| s170
| s175
| s180

/-- Represents the frequency distribution of uniform sizes -/
def uniformDistribution : List (UniformSize × Nat) :=
  [(UniformSize.s150, 1),
   (UniformSize.s155, 6),
   (UniformSize.s160, 8),
   (UniformSize.s165, 12),
   (UniformSize.s170, 5),
   (UniformSize.s175, 4),
   (UniformSize.s180, 2)]

/-- Calculates the median size of school uniforms -/
def medianUniformSize (distribution : List (UniformSize × Nat)) : UniformSize :=
  sorry

/-- Theorem stating that the median uniform size is 165cm -/
theorem median_is_165 :
  medianUniformSize uniformDistribution = UniformSize.s165 := by
  sorry

end NUMINAMATH_CALUDE_median_is_165_l314_31494


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_11_l314_31473

/-- Represents a five-digit number in the form AB,CBA --/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat := a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- Checks if the digits a, b, and c are valid for our problem --/
def valid_digits (a b c : Nat) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

theorem greatest_abcba_divisible_by_11 :
  ∃ (n : ABCBA), 
    valid_digits n.a n.b n.c ∧ 
    n.value % 11 = 0 ∧
    n.value = 96569 ∧
    (∀ (m : ABCBA), valid_digits m.a m.b m.c → m.value % 11 = 0 → m.value ≤ n.value) := by
  sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_11_l314_31473


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l314_31493

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = 2/5 + (1/3 : ℝ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = 2/5 - (1/3 : ℝ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l314_31493


namespace NUMINAMATH_CALUDE_taxi_fare_80_miles_l314_31419

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : ℝ) : ℝ :=
  sorry

theorem taxi_fare_80_miles : 
  -- Given conditions
  (taxiFare 60 = 150) →  -- 60-mile ride costs $150
  (∀ d, taxiFare d = 20 + (taxiFare d - 20) * d / 60) →  -- Flat rate of $20 and proportional charge
  -- Conclusion
  (taxiFare 80 = 193) :=
by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_80_miles_l314_31419


namespace NUMINAMATH_CALUDE_always_positive_l314_31446

theorem always_positive (x : ℝ) : (-x)^2 + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l314_31446


namespace NUMINAMATH_CALUDE_salary_change_calculation_salary_decrease_percentage_l314_31488

/-- Given an initial salary increase followed by a decrease, 
    calculate the percentage of the decrease. -/
theorem salary_change_calculation (initial_increase : ℝ) (net_increase : ℝ) : ℝ :=
  let final_factor := 1 + net_increase / 100
  let increase_factor := 1 + initial_increase / 100
  100 * (1 - final_factor / increase_factor)

/-- The percentage decrease in salary after an initial 10% increase,
    resulting in a net 1% increase, is approximately 8.18%. -/
theorem salary_decrease_percentage : 
  ∃ ε > 0, |salary_change_calculation 10 1 - 8.18| < ε :=
sorry

end NUMINAMATH_CALUDE_salary_change_calculation_salary_decrease_percentage_l314_31488


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l314_31448

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ (convexHull ℝ vertices)

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  points : Fin 3 → ℝ × ℝ
  inside : ∀ i, points i ∈ convexHull ℝ M.vertices

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem max_area_inscribed_triangle (M : ConvexPolygon) :
  ∃ (t : InscribedTriangle M), 
    (∀ i, t.points i ∈ M.vertices) ∧
    (∀ (s : InscribedTriangle M), 
      triangleArea (t.points 0) (t.points 1) (t.points 2) ≥ 
      triangleArea (s.points 0) (s.points 1) (s.points 2)) :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l314_31448


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l314_31402

theorem parametric_to_standard_equation (x y t : ℝ) :
  (∃ θ : ℝ, x = (1/2) * (Real.exp t + Real.exp (-t)) * Real.cos θ ∧
             y = (1/2) * (Real.exp t - Real.exp (-t)) * Real.sin θ) →
  x^2 * (Real.exp (2*t) - 2 + Real.exp (-2*t)) + 
  y^2 * (Real.exp (2*t) + 2 + Real.exp (-2*t)) = 
  Real.exp (6*t) - 2 * Real.exp (4*t) + Real.exp (2*t) + 
  2 * Real.exp (4*t) - 4 * Real.exp (2*t) + 2 + 
  Real.exp (2*t) - 2 * Real.exp (-2*t) + Real.exp (-4*t) :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l314_31402


namespace NUMINAMATH_CALUDE_garden_breadth_is_100_l314_31433

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_is_100 :
  ∃ (garden : RectangularGarden),
    garden.length = 250 ∧
    perimeter garden = 700 ∧
    garden.breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_is_100_l314_31433


namespace NUMINAMATH_CALUDE_shirts_purchased_l314_31405

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_cost : ℕ := 51
def num_jeans : ℕ := 2
def num_hats : ℕ := 4

theorem shirts_purchased : 
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_cost + num_jeans * jeans_cost + num_hats * hat_cost = total_cost ∧ 
    num_shirts = 3 := by
  sorry

end NUMINAMATH_CALUDE_shirts_purchased_l314_31405


namespace NUMINAMATH_CALUDE_opposite_face_is_indigo_l314_31458

-- Define the colors
inductive Color
| Orange | Blue | Yellow | Silver | Violet | Indigo

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the visibility of colors in the three views
def visibleColors (c : Cube) : List Color :=
  [c.faces 0, c.faces 1, c.faces 2, c.faces 3, c.faces 4]

-- Define the conditions
def satisfiesConditions (c : Cube) : Prop :=
  (c.faces 0 = Color.Blue) ∧  -- Blue is on top in all views
  (c.faces 1 = Color.Silver ∨ c.faces 1 = Color.Yellow) ∧  -- Right face is Silver or Yellow
  (c.faces 2 = Color.Yellow ∨ c.faces 2 = Color.Violet ∨ c.faces 2 = Color.Indigo) ∧  -- Front face changes
  (Color.Orange ∉ visibleColors c)  -- Orange is not visible

-- Theorem statement
theorem opposite_face_is_indigo (c : Cube) (h : satisfiesConditions c) :
  c.faces 5 = Color.Indigo :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_indigo_l314_31458


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l314_31415

theorem unique_quadratic_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = 2*x ↔ x = 2) → 
  (a = -2 ∧ b = 4) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l314_31415


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt_sum_power_l314_31441

theorem smallest_integer_above_sqrt_sum_power : 
  ∃ n : ℕ, n = 3742 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt_sum_power_l314_31441


namespace NUMINAMATH_CALUDE_geometric_series_product_l314_31472

theorem geometric_series_product (y : ℝ) : 
  (∑' n : ℕ, (1/3:ℝ)^n) * (∑' n : ℕ, (-1/3:ℝ)^n) = ∑' n : ℕ, (1/y:ℝ)^n → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_l314_31472


namespace NUMINAMATH_CALUDE_root_square_relation_l314_31483

/-- The polynomial h(x) = x^3 + x^2 + 2x + 8 -/
def h (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 8

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem root_square_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, j b c d x = 0 ↔ ∃ r : ℝ, h r = 0 ∧ x = r^2) →
  b = 1 ∧ c = -8 ∧ d = 32 := by
sorry

end NUMINAMATH_CALUDE_root_square_relation_l314_31483


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l314_31450

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l314_31450


namespace NUMINAMATH_CALUDE_line_segment_parameterization_sum_of_squares_l314_31460

/-- Given a line segment connecting (-4, 10) and (2, -3), parameterized by x = at + b and y = ct + d
    where -1 ≤ t ≤ 1 and t = -1 corresponds to (-4, 10), prove that a^2 + b^2 + c^2 + d^2 = 321 -/
theorem line_segment_parameterization_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, -1 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (a * (-1) + b = -4 ∧ c * (-1) + d = 10) →
  (a * 1 + b = 2 ∧ c * 1 + d = -3) →
  a^2 + b^2 + c^2 + d^2 = 321 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_sum_of_squares_l314_31460


namespace NUMINAMATH_CALUDE_binomial_30_3_l314_31489

theorem binomial_30_3 : (30 : ℕ).choose 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l314_31489


namespace NUMINAMATH_CALUDE_converse_statement_l314_31459

/-- Given that m is a real number, prove that the converse of the statement 
    "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is "If the equation x^2 + x - m = 0 has real roots, then m > 0" -/
theorem converse_statement (m : ℝ) : 
  (∃ x : ℝ, x^2 + x - m = 0) → m > 0 :=
sorry

end NUMINAMATH_CALUDE_converse_statement_l314_31459


namespace NUMINAMATH_CALUDE_train_length_l314_31482

/-- Given a train crossing a pole at a speed of 60 km/hr in 18 seconds,
    prove that the length of the train is 300 meters. -/
theorem train_length (speed : ℝ) (time_seconds : ℝ) (length : ℝ) :
  speed = 60 →
  time_seconds = 18 →
  length = speed * (time_seconds / 3600) * 1000 →
  length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l314_31482


namespace NUMINAMATH_CALUDE_usb_drive_usage_percentage_l314_31486

theorem usb_drive_usage_percentage (total_capacity : ℝ) (available_space : ℝ) 
  (h1 : total_capacity = 16) 
  (h2 : available_space = 8) : 
  (total_capacity - available_space) / total_capacity * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_usb_drive_usage_percentage_l314_31486


namespace NUMINAMATH_CALUDE_stating_probability_of_all_red_by_fourth_draw_l314_31440

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing all red balls exactly by the 4th draw -/
def prob_all_red_by_fourth_draw : ℚ := 353 / 5000

/-- 
Theorem stating that the probability of drawing all red balls exactly 
by the 4th draw is 353/5000, given the initial conditions
-/
theorem probability_of_all_red_by_fourth_draw : 
  prob_all_red_by_fourth_draw = 353 / 5000 := by sorry

end NUMINAMATH_CALUDE_stating_probability_of_all_red_by_fourth_draw_l314_31440


namespace NUMINAMATH_CALUDE_not_right_triangle_l314_31435

theorem not_right_triangle (a b c : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 4) (hc : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l314_31435


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l314_31480

theorem number_of_divisors_of_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l314_31480


namespace NUMINAMATH_CALUDE_exceptional_face_edges_multiple_of_three_l314_31479

structure Polyhedron where
  faces : Set Face
  edges : Set Edge
  edge_count : Face → ℕ
  adjacent : Face → Face → Prop
  color : Face → Bool

theorem exceptional_face_edges_multiple_of_three (P : Polyhedron) 
  (h1 : ∀ f g : Face, P.adjacent f g → P.color f ≠ P.color g)
  (h2 : ∃! f : Face, f ∈ P.faces ∧ ¬(∃ k : ℕ, P.edge_count f = 3 * k))
  (h3 : ∀ e : Edge, e ∈ P.edges → ∃! f g : Face, f ≠ g ∧ f ∈ P.faces ∧ g ∈ P.faces ∧ P.adjacent f g)
  : ∀ f : Face, f ∈ P.faces → ∃ k : ℕ, P.edge_count f = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_exceptional_face_edges_multiple_of_three_l314_31479


namespace NUMINAMATH_CALUDE_chocolate_difference_l314_31431

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption between Robert and Nickel -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l314_31431


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l314_31420

theorem rectangle_area (square_area : Real) (rectangle_breadth : Real) : Real :=
  let square_side : Real := Real.sqrt square_area
  let circle_radius : Real := square_side
  let rectangle_length : Real := circle_radius / 4
  let rectangle_area : Real := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1225 10 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l314_31420


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l314_31471

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ i : ℕ, i < count → ¬(is_prime (start + i))

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧
  (consecutive_nonprimes 90 7) ∧
  (∀ p : ℕ, p < 97 → is_prime p → ¬(∃ start : ℕ, start < p ∧ consecutive_nonprimes start 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l314_31471


namespace NUMINAMATH_CALUDE_salary_after_changes_l314_31401

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.25

theorem salary_after_changes : 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) = 2587.5 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_changes_l314_31401


namespace NUMINAMATH_CALUDE_planar_figure_division_l314_31478

/-- A planar figure with diameter 1 -/
structure PlanarFigure where
  diam : ℝ
  diam_eq_one : diam = 1

/-- The minimum diameter of n parts that a planar figure can be divided into -/
noncomputable def δ₂ (n : ℕ) (F : PlanarFigure) : ℝ := sorry

/-- Main theorem about division of planar figures -/
theorem planar_figure_division (F : PlanarFigure) : 
  (δ₂ 3 F ≤ Real.sqrt 3 / 2) ∧ 
  (δ₂ 4 F ≤ Real.sqrt 2 / 2) ∧ 
  (δ₂ 7 F ≤ 1 / 2) := by sorry

end NUMINAMATH_CALUDE_planar_figure_division_l314_31478


namespace NUMINAMATH_CALUDE_only_four_not_divide_98_l314_31476

theorem only_four_not_divide_98 :
  (∀ n ∈ ({2, 7, 14, 49} : Set Nat), 98 % n = 0) ∧ 98 % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_only_four_not_divide_98_l314_31476


namespace NUMINAMATH_CALUDE_period_start_time_l314_31434

def period_end : Nat := 17  -- 5 pm in 24-hour format
def rain_duration : Nat := 2
def no_rain_duration : Nat := 6

theorem period_start_time : 
  period_end - (rain_duration + no_rain_duration) = 9 := by
  sorry

end NUMINAMATH_CALUDE_period_start_time_l314_31434


namespace NUMINAMATH_CALUDE_cyclic_inequality_l314_31457

theorem cyclic_inequality (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q)
  (h2 : z = y^2 + p*y + q)
  (h3 : x = z^2 + p*z + q) :
  x^2*y + y^2*z + z^2*x ≥ x^2*z + y^2*x + z^2*y := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l314_31457


namespace NUMINAMATH_CALUDE_school_report_mistake_l314_31417

theorem school_report_mistake :
  ¬ ∃ (girls : ℕ), 
    let boys := girls + 373
    let total := girls + boys
    total = 3688 :=
by
  sorry

end NUMINAMATH_CALUDE_school_report_mistake_l314_31417


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l314_31416

/-- Proves that the repeating decimal 0.53207207207... is equal to 5316750/999900 -/
theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 0.53207207207 ∧ x = 5316750 / 999900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l314_31416


namespace NUMINAMATH_CALUDE_nested_square_root_evaluation_l314_31456

theorem nested_square_root_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x + Real.sqrt (x + Real.sqrt x)) = Real.sqrt (x + Real.sqrt (x + x^(1/2))) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_evaluation_l314_31456


namespace NUMINAMATH_CALUDE_remainder_theorem_l314_31400

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x + 2 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l314_31400


namespace NUMINAMATH_CALUDE_number_increased_by_45_percent_l314_31411

theorem number_increased_by_45_percent (x : ℝ) : x * 1.45 = 870 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_increased_by_45_percent_l314_31411


namespace NUMINAMATH_CALUDE_set_intersection_equality_l314_31442

def A : Set ℝ := {x | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2}

theorem set_intersection_equality (a b : ℝ) :
  A ∩ B a = C b → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l314_31442


namespace NUMINAMATH_CALUDE_x_value_l314_31461

theorem x_value (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l314_31461


namespace NUMINAMATH_CALUDE_hcl_moles_formed_l314_31403

/-- Represents the chemical reaction NH4Cl + H2O → NH4OH + HCl -/
structure ChemicalReaction where
  nh4cl_mass : ℝ
  h2o_moles : ℝ
  nh4oh_moles : ℝ
  hcl_moles : ℝ

/-- The molar mass of NH4Cl in g/mol -/
def nh4cl_molar_mass : ℝ := 53.49

/-- Theorem stating that in the given reaction, 1 mole of HCl is formed -/
theorem hcl_moles_formed (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl_mass = 53)
  (h2 : reaction.h2o_moles = 1)
  (h3 : reaction.nh4oh_moles = 1) :
  reaction.hcl_moles = 1 := by
sorry

end NUMINAMATH_CALUDE_hcl_moles_formed_l314_31403


namespace NUMINAMATH_CALUDE_functional_equation_problem_l314_31427

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
  (h1 : f 1 = 1)
  (h4 : f 4 = 7) :
  f 2022 = 4043 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l314_31427


namespace NUMINAMATH_CALUDE_continuous_function_three_times_value_l314_31439

/-- There exists a continuous function that takes every real value exactly 3 times. -/
theorem continuous_function_three_times_value :
  ∃ f : ℝ → ℝ, Continuous f ∧ ∀ y : ℝ, (∃! x₁ x₂ x₃ : ℝ, f x₁ = y ∧ f x₂ = y ∧ f x₃ = y ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_three_times_value_l314_31439


namespace NUMINAMATH_CALUDE_fraction_power_division_l314_31443

theorem fraction_power_division :
  (1 / 3 : ℚ)^4 / (1 / 5 : ℚ) = 5 / 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_division_l314_31443


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l314_31429

theorem rectangular_prism_width 
  (l h w d : ℝ) 
  (h_def : h = 2 * l)
  (l_val : l = 5)
  (diagonal : d = 17)
  (diag_eq : d^2 = l^2 + w^2 + h^2) :
  w = 2 * Real.sqrt 41 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l314_31429


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l314_31409

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | x - 1 > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l314_31409


namespace NUMINAMATH_CALUDE_eric_quarters_count_l314_31437

/-- The number of dimes Cindy tosses -/
def cindy_dimes : ℕ := 5

/-- The number of nickels Garrick throws -/
def garrick_nickels : ℕ := 8

/-- The number of pennies Ivy drops -/
def ivy_pennies : ℕ := 60

/-- The total amount in the pond in cents -/
def total_cents : ℕ := 200

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Eric flipped into the pond -/
def eric_quarters : ℕ := 2

theorem eric_quarters_count :
  eric_quarters * quarter_value = 
    total_cents - (cindy_dimes * dime_value + garrick_nickels * nickel_value + ivy_pennies * penny_value) :=
by sorry

end NUMINAMATH_CALUDE_eric_quarters_count_l314_31437


namespace NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achievable_l314_31470

theorem max_value_sqrt_product (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) = 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achievable_l314_31470


namespace NUMINAMATH_CALUDE_computer_usage_difference_l314_31490

/-- The difference in computer usage between two weeks -/
def usage_difference (last_week : ℕ) (this_week_daily : ℕ) : ℕ :=
  last_week - (this_week_daily * 7)

/-- Theorem stating the difference in computer usage -/
theorem computer_usage_difference :
  usage_difference 91 8 = 35 := by
  sorry

end NUMINAMATH_CALUDE_computer_usage_difference_l314_31490


namespace NUMINAMATH_CALUDE_alligator_journey_time_l314_31469

theorem alligator_journey_time (initial_time : ℝ) (return_time : ℝ) : 
  initial_time = 4 →
  return_time = initial_time + 2 * Real.sqrt initial_time →
  initial_time + return_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l314_31469


namespace NUMINAMATH_CALUDE_tile_difference_l314_31410

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The difference in tiles between the 11th and 10th squares -/
theorem tile_difference : tiles 11 - tiles 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_l314_31410


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l314_31474

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l314_31474
