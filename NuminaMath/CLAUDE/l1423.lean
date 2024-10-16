import Mathlib

namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1423_142319

def is_pythagorean_triple (a b c : ℚ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 5 8 12 ∧
  is_pythagorean_triple 30 40 50 ∧
  ¬ is_pythagorean_triple 9 13 15 ∧
  ¬ is_pythagorean_triple (1/6) (1/8) (1/10) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1423_142319


namespace NUMINAMATH_CALUDE_marble_217_is_red_l1423_142316

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 217th marble is red -/
theorem marble_217_is_red : marbleColor 217 = MarbleColor.Red := by
  sorry


end NUMINAMATH_CALUDE_marble_217_is_red_l1423_142316


namespace NUMINAMATH_CALUDE_f_2013_pi_third_l1423_142367

open Real

noncomputable def f₀ (x : ℝ) : ℝ := sin x - cos x

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f₀ x
  | n + 1 => deriv (f n) x

theorem f_2013_pi_third : f 2013 (π/3) = (1 + Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_f_2013_pi_third_l1423_142367


namespace NUMINAMATH_CALUDE_min_sum_squares_l1423_142300

theorem min_sum_squares (p q r s t u v w : Int) : 
  p ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  q ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  r ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  s ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  t ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  u ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  v ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  w ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
  q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
  r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
  s ≠ t → s ≠ u → s ≠ v → s ≠ w →
  t ≠ u → t ≠ v → t ≠ w →
  u ≠ v → u ≠ w →
  v ≠ w →
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1423_142300


namespace NUMINAMATH_CALUDE_number_pattern_l1423_142331

theorem number_pattern (A : ℕ) : 10 * A + 9 = A * 9 + (A + 9) := by
  sorry

end NUMINAMATH_CALUDE_number_pattern_l1423_142331


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l1423_142396

theorem first_reduction_percentage (x : ℝ) :
  (1 - x / 100) * (1 - 50 / 100) = 1 - 62.5 / 100 →
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l1423_142396


namespace NUMINAMATH_CALUDE_december_24_is_sunday_l1423_142383

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) (thanksgivingDOW : DayOfWeek) : DayOfWeek :=
  sorry

theorem december_24_is_sunday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 24)
  (h3 : dayOfWeek thanksgiving DayOfWeek.Friday = DayOfWeek.Friday) :
  dayOfWeek ⟨12, 24⟩ DayOfWeek.Friday = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_december_24_is_sunday_l1423_142383


namespace NUMINAMATH_CALUDE_pets_remaining_l1423_142363

theorem pets_remaining (initial_puppies initial_kittens sold_puppies sold_kittens : ℕ) 
  (h1 : initial_puppies = 7)
  (h2 : initial_kittens = 6)
  (h3 : sold_puppies = 2)
  (h4 : sold_kittens = 3) :
  initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 := by
sorry

end NUMINAMATH_CALUDE_pets_remaining_l1423_142363


namespace NUMINAMATH_CALUDE_line_relationships_l1423_142353

-- Define a type for lines in a plane
structure Line2D where
  -- You might represent a line by its slope and y-intercept, or by two points, etc.
  -- For this abstract representation, we'll leave the internal structure unspecified

-- Define a type for planes
structure Plane where
  -- Again, we'll leave the internal structure unspecified for this abstract representation

-- Define what it means for two lines to be non-overlapping
def non_overlapping (l1 l2 : Line2D) : Prop :=
  l1 ≠ l2

-- Define what it means for two lines to be in the same plane
def same_plane (p : Plane) (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  True  -- placeholder

-- Define parallel relationship
def parallel (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- Define intersecting relationship
def intersecting (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- The main theorem
theorem line_relationships (p : Plane) (l1 l2 : Line2D) 
  (h1 : non_overlapping l1 l2) (h2 : same_plane p l1 l2) :
  parallel l1 l2 ∨ intersecting l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l1423_142353


namespace NUMINAMATH_CALUDE_farm_work_earnings_l1423_142303

/-- Calculates the total money collected given hourly rate, hours worked, and tips. -/
def total_money_collected (hourly_rate : ℕ) (hours_worked : ℕ) (tips : ℕ) : ℕ :=
  hourly_rate * hours_worked + tips

/-- Proves that given the specified conditions, the total money collected is $240. -/
theorem farm_work_earnings : total_money_collected 10 19 50 = 240 := by
  sorry

end NUMINAMATH_CALUDE_farm_work_earnings_l1423_142303


namespace NUMINAMATH_CALUDE_tulip_fraction_l1423_142365

/-- Represents the composition of a bouquet of flowers -/
structure Bouquet where
  pink_lilies : ℝ
  red_lilies : ℝ
  pink_tulips : ℝ
  red_tulips : ℝ

/-- The fraction of tulips in a bouquet satisfying given conditions -/
theorem tulip_fraction (b : Bouquet) 
  (half_pink_lilies : b.pink_lilies = b.pink_tulips)
  (third_red_tulips : b.red_tulips = (1/3) * (b.red_lilies + b.red_tulips))
  (three_fifths_pink : b.pink_lilies + b.pink_tulips = (3/5) * (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips)) :
  (b.pink_tulips + b.red_tulips) / (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips) = 13/30 := by
  sorry

#check tulip_fraction

end NUMINAMATH_CALUDE_tulip_fraction_l1423_142365


namespace NUMINAMATH_CALUDE_multiply_123_32_125_l1423_142314

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_123_32_125_l1423_142314


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l1423_142305

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2^(sequence_a n) + 2

theorem divisibility_in_sequence (m n : ℕ) (h : m < n) :
  ∃ k : ℕ, sequence_a n = k * sequence_a m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l1423_142305


namespace NUMINAMATH_CALUDE_complex_number_location_l1423_142309

theorem complex_number_location :
  let z : ℂ := 1 / ((1 + Complex.I)^2 + 1)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1423_142309


namespace NUMINAMATH_CALUDE_discount_percentage_l1423_142377

theorem discount_percentage
  (MP : ℝ)
  (CP : ℝ)
  (SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (h2 : (SP - CP) / CP = 0.5454545454545454)
  : (MP - SP) / MP = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l1423_142377


namespace NUMINAMATH_CALUDE_pages_read_difference_l1423_142368

/-- The number of weeks required for Janet to read 2100 more pages than Belinda,
    given that Janet reads 80 pages a day and Belinda reads 30 pages a day. -/
theorem pages_read_difference (janet_daily : ℕ) (belinda_daily : ℕ) (total_difference : ℕ) :
  janet_daily = 80 →
  belinda_daily = 30 →
  total_difference = 2100 →
  (total_difference / ((janet_daily - belinda_daily) * 7) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l1423_142368


namespace NUMINAMATH_CALUDE_total_bugs_equals_63_l1423_142372

/-- The number of bugs eaten by the gecko -/
def gecko_bugs : ℕ := 12

/-- The number of bugs eaten by the lizard -/
def lizard_bugs : ℕ := gecko_bugs / 2

/-- The number of bugs eaten by the frog -/
def frog_bugs : ℕ := lizard_bugs * 3

/-- The number of bugs eaten by the toad -/
def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

/-- The total number of bugs eaten by all animals -/
def total_bugs : ℕ := gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

theorem total_bugs_equals_63 : total_bugs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_bugs_equals_63_l1423_142372


namespace NUMINAMATH_CALUDE_lowest_possible_score_l1423_142315

def exam_max_score : ℕ := 120
def num_exams : ℕ := 5
def goal_average : ℕ := 100
def current_scores : List ℕ := [90, 108, 102]

theorem lowest_possible_score :
  let total_needed : ℕ := goal_average * num_exams
  let current_total : ℕ := current_scores.sum
  let remaining_total : ℕ := total_needed - current_total
  let max_score_one_exam : ℕ := min exam_max_score remaining_total
  ∃ (lowest : ℕ), 
    lowest = remaining_total - max_score_one_exam ∧
    lowest = 80 :=
sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l1423_142315


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1423_142325

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 8*x₁ = 9 ∧ x₂^2 + 8*x₂ = 9) ∧ 
  (x₁ = -9 ∧ x₂ = 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1423_142325


namespace NUMINAMATH_CALUDE_semicircle_in_square_l1423_142328

theorem semicircle_in_square (d m n : ℝ) : 
  d > 0 →                           -- d is positive (diameter)
  8 > 0 →                           -- square side length is positive
  d ≤ 8 →                           -- semicircle fits in square
  d ≤ m - Real.sqrt n →             -- maximum value of d
  m - Real.sqrt n ≤ 8 →             -- maximum value fits in square
  (∀ x, x > 0 → x - Real.sqrt (4 * x) < m - Real.sqrt n) →  -- m - √n is indeed the maximum
  m + n = 544 := by
sorry

end NUMINAMATH_CALUDE_semicircle_in_square_l1423_142328


namespace NUMINAMATH_CALUDE_ethereum_investment_l1423_142387

theorem ethereum_investment (I : ℝ) : 
  I > 0 →
  (I * 1.25 * 1.5 = 750) →
  I = 400 := by
sorry

end NUMINAMATH_CALUDE_ethereum_investment_l1423_142387


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l1423_142332

theorem cone_volume_approximation (r h : ℝ) (π : ℝ) : 
  (1/3) * π * r^2 * h = (2/75) * (2 * π * r)^2 * h → π = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l1423_142332


namespace NUMINAMATH_CALUDE_max_value_of_f_l1423_142326

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) ∧
  f x = Real.pi / 12 + Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1423_142326


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_pow_1999_l1423_142369

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem last_three_digits_of_5_pow_1999 :
  last_three_digits (5^1999) = 125 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_pow_1999_l1423_142369


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1423_142388

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 3 →
  a + b = (Real.sqrt 3, 1) →
  ‖a - b‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1423_142388


namespace NUMINAMATH_CALUDE_floor_e_l1423_142375

theorem floor_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_e_l1423_142375


namespace NUMINAMATH_CALUDE_stone_splitting_game_winner_l1423_142364

/-- The stone-splitting game -/
def StoneSplittingGame (n : ℕ) : Prop :=
  ∃ (winner : Bool),
    (winner = true → n.Prime ∨ ∃ k, n = 2^k) ∧
    (winner = false → ¬(n.Prime ∨ ∃ k, n = 2^k))

/-- Theorem: Characterization of winning conditions in the stone-splitting game -/
theorem stone_splitting_game_winner (n : ℕ) :
  StoneSplittingGame n ↔ (n.Prime ∨ ∃ k, n = 2^k) := by sorry

end NUMINAMATH_CALUDE_stone_splitting_game_winner_l1423_142364


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1423_142337

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (60 : ℤ) = Nat.gcd 25920 213840 ∧
  ∀ (k : ℤ), k ∣ (15*x + 9) * (15*x + 15) * (15*x + 21) → k ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1423_142337


namespace NUMINAMATH_CALUDE_symmetric_points_relation_l1423_142318

/-- 
Given two points P and Q in the 2D plane, where:
- P has coordinates (m+1, 3)
- Q has coordinates (1, n-2)
- P is symmetric to Q with respect to the x-axis

This theorem proves that m-n = 1.
-/
theorem symmetric_points_relation (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (m + 1, 3) ∧ 
    Q = (1, n - 2) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) → 
  m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_relation_l1423_142318


namespace NUMINAMATH_CALUDE_furniture_production_l1423_142371

theorem furniture_production (total_wood : ℕ) (table_wood : ℕ) (chair_wood : ℕ) (tables_made : ℕ) :
  total_wood = 672 →
  table_wood = 12 →
  chair_wood = 8 →
  tables_made = 24 →
  (total_wood - tables_made * table_wood) / chair_wood = 48 :=
by sorry

end NUMINAMATH_CALUDE_furniture_production_l1423_142371


namespace NUMINAMATH_CALUDE_periodic_function_l1423_142311

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) + f (x - 1) = Real.sqrt 3 * f x

/-- The period of a function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    HasPeriod f 12 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l1423_142311


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l1423_142360

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l1423_142360


namespace NUMINAMATH_CALUDE_award_sequences_eq_sixteen_l1423_142348

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Calculates the number of possible award sequences -/
def award_sequences : ℕ := 2^num_rounds

/-- Theorem stating that the number of award sequences is 16 -/
theorem award_sequences_eq_sixteen : award_sequences = 16 := by
  sorry

end NUMINAMATH_CALUDE_award_sequences_eq_sixteen_l1423_142348


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1423_142306

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, (a - b) / c > 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1423_142306


namespace NUMINAMATH_CALUDE_distance_between_points_l1423_142321

theorem distance_between_points (x : ℝ) :
  (x - 2)^2 + (5 - 5)^2 = 5^2 → x = -3 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1423_142321


namespace NUMINAMATH_CALUDE_solution_pairs_l1423_142334

theorem solution_pairs (x y p : ℕ) (hp : Nat.Prime p) :
  x > 0 ∧ y > 0 ∧ x ≤ y ∧ (x + y) * (x * y - 1) = p * (x * y + 1) →
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (∃ q : ℕ, Nat.Prime q ∧ x = 1 ∧ y = q + 1 ∧ p = q) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l1423_142334


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1423_142398

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center (x y : ℝ) : Prop := 
  circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The theorem statement -/
theorem line_passes_through_circle_center :
  ∃ m : ℝ, ∀ x y : ℝ, circle_center x y → line_equation x y m := by sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1423_142398


namespace NUMINAMATH_CALUDE_four_dice_same_face_probability_l1423_142312

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being tossed -/
def numDice : ℕ := 4

/-- The probability of a specific outcome on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability of all dice showing the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

theorem four_dice_same_face_probability :
  allSameProbability = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_face_probability_l1423_142312


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l1423_142399

theorem quadratic_root_equivalence (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 + 2 = 0) →
  (a * (2019 - 1)^2 + b * (2019 - 1) = -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l1423_142399


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1423_142374

theorem arithmetic_calculation : 1435 + 180 / 60 * 3 - 435 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1423_142374


namespace NUMINAMATH_CALUDE_complex_fraction_real_l1423_142362

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l1423_142362


namespace NUMINAMATH_CALUDE_lines_do_not_intersect_l1423_142391

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 3), direction := (5, -8) }

/-- The second line -/
def line2 (k : ℝ) : Line2D :=
  { point := (-1, 4), direction := (2, k) }

/-- Theorem: The lines do not intersect if and only if k = -16/5 -/
theorem lines_do_not_intersect (k : ℝ) : 
  are_parallel line1 (line2 k) ↔ k = -16/5 := by
  sorry

end NUMINAMATH_CALUDE_lines_do_not_intersect_l1423_142391


namespace NUMINAMATH_CALUDE_min_area_two_rectangles_l1423_142378

/-- Given a wire of length l, cut into two pieces x and (l-x), forming two rectangles
    with length-to-width ratios of 2:1 and 3:2 respectively, the minimum value of 
    the sum of their areas is 3/104 * l^2 --/
theorem min_area_two_rectangles (l : ℝ) (h : l > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < l ∧
  (∀ (y : ℝ), 0 < y → y < l →
    x^2 / 18 + 3 * (l - x)^2 / 50 ≤ y^2 / 18 + 3 * (l - y)^2 / 50) ∧
  x^2 / 18 + 3 * (l - x)^2 / 50 = 3 * l^2 / 104 :=
sorry

end NUMINAMATH_CALUDE_min_area_two_rectangles_l1423_142378


namespace NUMINAMATH_CALUDE_inequality_range_l1423_142390

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) → 
  (m ≥ 4 ∨ m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1423_142390


namespace NUMINAMATH_CALUDE_lottery_probability_l1423_142322

/-- The number of people participating in the lottery drawing -/
def num_people : ℕ := 4

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 4

/-- The number of winning tickets -/
def winning_tickets : ℕ := 2

/-- The probability that the event ends right after the third person has finished drawing -/
def prob_end_after_third : ℚ := 1/3

theorem lottery_probability :
  (num_people = 4) →
  (total_tickets = 4) →
  (winning_tickets = 2) →
  (prob_end_after_third = 1/3) := by
  sorry

#check lottery_probability

end NUMINAMATH_CALUDE_lottery_probability_l1423_142322


namespace NUMINAMATH_CALUDE_original_combined_cost_l1423_142349

/-- Represents the original prices of items --/
structure OriginalPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Represents the discounted prices of items --/
structure DiscountedPrices where
  dress : ℝ
  shoes : ℝ
  handbag : ℝ
  necklace : ℝ

/-- Calculates the total savings before the coupon --/
def totalSavings (original : OriginalPrices) (discounted : DiscountedPrices) : ℝ :=
  (original.dress - discounted.dress) +
  (original.shoes - discounted.shoes) +
  (original.handbag - discounted.handbag) +
  (original.necklace - discounted.necklace)

/-- Calculates the total discounted price before the coupon --/
def totalDiscountedPrice (discounted : DiscountedPrices) : ℝ :=
  discounted.dress + discounted.shoes + discounted.handbag + discounted.necklace

/-- The main theorem --/
theorem original_combined_cost (original : OriginalPrices) (discounted : DiscountedPrices)
  (h1 : discounted.dress = original.dress / 2 - 10)
  (h2 : discounted.shoes = original.shoes * 0.85)
  (h3 : discounted.handbag = original.handbag - 30)
  (h4 : discounted.necklace = original.necklace)
  (h5 : discounted.necklace ≤ original.dress)
  (h6 : totalSavings original discounted = 120)
  (h7 : totalDiscountedPrice discounted * 0.9 = totalDiscountedPrice discounted - 120) :
  original.dress + original.shoes + original.handbag + original.necklace = 1200 := by
  sorry


end NUMINAMATH_CALUDE_original_combined_cost_l1423_142349


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l1423_142351

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem minimum_value_implies_m (m : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f x m ≥ -3) ∧ (∃ x : ℝ, x ≥ 2 ∧ f x m = -3) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l1423_142351


namespace NUMINAMATH_CALUDE_school_arrival_time_l1423_142352

/-- Calculates how early (in minutes) a boy arrives at school on the second day given the following conditions:
  * The distance between home and school is 2.5 km
  * On the first day, he travels at 5 km/hr and arrives 5 minutes late
  * On the second day, he travels at 10 km/hr and arrives early
-/
theorem school_arrival_time (distance : ℝ) (speed1 speed2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 ∧ 
  speed1 = 5 ∧ 
  speed2 = 10 ∧ 
  late_time = 5 → 
  (distance / speed1 * 60 - late_time) - (distance / speed2 * 60) = 10 := by
  sorry

#check school_arrival_time

end NUMINAMATH_CALUDE_school_arrival_time_l1423_142352


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1423_142310

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 2 = 0}

-- Define the condition that x ∈ A is necessary but not sufficient for x ∈ B
def necessary_not_sufficient (a : ℝ) : Prop :=
  B a ⊆ A ∧ B a ≠ A

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ a ∈ ({-1, 0, 1} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1423_142310


namespace NUMINAMATH_CALUDE_bryans_precious_stones_l1423_142359

theorem bryans_precious_stones (price_per_stone total_amount : ℕ) 
  (h1 : price_per_stone = 1785)
  (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryans_precious_stones_l1423_142359


namespace NUMINAMATH_CALUDE_cab_driver_first_day_income_l1423_142308

def cab_driver_income (day2 day3 day4 day5 : ℕ) (average : ℕ) : Prop :=
  ∃ day1 : ℕ,
    day2 = 250 ∧
    day3 = 450 ∧
    day4 = 400 ∧
    day5 = 800 ∧
    average = 500 ∧
    (day1 + day2 + day3 + day4 + day5) / 5 = average ∧
    day1 = 600

theorem cab_driver_first_day_income :
  ∀ day2 day3 day4 day5 average : ℕ,
    cab_driver_income day2 day3 day4 day5 average →
    ∃ day1 : ℕ, day1 = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_first_day_income_l1423_142308


namespace NUMINAMATH_CALUDE_hayes_laundry_loads_l1423_142397

/-- The number of detergent pods in a pack -/
def pods_per_pack : ℕ := 39

/-- The number of packs Hayes needs for a full year -/
def packs_per_year : ℕ := 4

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of loads of laundry Hayes does in a week -/
def loads_per_week : ℕ := (pods_per_pack * packs_per_year) / weeks_per_year

theorem hayes_laundry_loads : loads_per_week = 3 := by sorry

end NUMINAMATH_CALUDE_hayes_laundry_loads_l1423_142397


namespace NUMINAMATH_CALUDE_power_product_three_six_l1423_142341

theorem power_product_three_six : (3^5 * 6^5 : ℕ) = 34012224 := by
  sorry

end NUMINAMATH_CALUDE_power_product_three_six_l1423_142341


namespace NUMINAMATH_CALUDE_exists_unique_t_l1423_142344

-- Define the function f
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem exists_unique_t : ∃! t : ℝ,
  (f (-6) = 2 ∧ f 2 = -6) ∧
  (∀ x k : ℝ, k > 0 → f (x + k) < f x) ∧
  (∀ x : ℝ, |f (x - t) + 2| < 4 ↔ -4 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_exists_unique_t_l1423_142344


namespace NUMINAMATH_CALUDE_total_eggs_needed_l1423_142345

def eggs_from_andrew : ℕ := 155
def eggs_to_buy : ℕ := 67

theorem total_eggs_needed : 
  eggs_from_andrew + eggs_to_buy = 222 := by sorry

end NUMINAMATH_CALUDE_total_eggs_needed_l1423_142345


namespace NUMINAMATH_CALUDE_smallest_m_value_l1423_142382

def count_quadruplets (m : ℕ) : ℕ :=
  sorry

theorem smallest_m_value :
  ∃ (m : ℕ),
    (count_quadruplets m = 125000) ∧
    (∀ (a b c d : ℕ), (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) →
                      (count_quadruplets m = 125000)) ∧
    (∀ (m' : ℕ), m' < m →
      (count_quadruplets m' ≠ 125000 ∨
       ∃ (a b c d : ℕ), Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                         Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m' ∧
                         count_quadruplets m' ≠ 125000)) ∧
    m = 9450000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_value_l1423_142382


namespace NUMINAMATH_CALUDE_impossible_to_reach_target_l1423_142327

/-- Represents the configuration of matchsticks on a square's vertices -/
structure SquareConfig where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ

/-- Calculates S for a given configuration -/
def S (config : SquareConfig) : ℤ :=
  config.a₁ - config.a₂ + config.a₃ - config.a₄

/-- Represents a valid move in the matchstick game -/
inductive Move
  | move_a₁ (k : ℕ)
  | move_a₂ (k : ℕ)
  | move_a₃ (k : ℕ)
  | move_a₄ (k : ℕ)

/-- Applies a move to a configuration -/
def apply_move (config : SquareConfig) (move : Move) : SquareConfig :=
  match move with
  | Move.move_a₁ k => ⟨config.a₁ - k, config.a₂ + k, config.a₃, config.a₄ + k⟩
  | Move.move_a₂ k => ⟨config.a₁ + k, config.a₂ - k, config.a₃ + k, config.a₄⟩
  | Move.move_a₃ k => ⟨config.a₁, config.a₂ + k, config.a₃ - k, config.a₄ + k⟩
  | Move.move_a₄ k => ⟨config.a₁ + k, config.a₂, config.a₃ + k, config.a₄ - k⟩

/-- The main theorem stating the impossibility of reaching the target configuration -/
theorem impossible_to_reach_target :
  ∀ (moves : List Move),
  let start_config := ⟨1, 0, 0, 0⟩
  let end_config := List.foldl apply_move start_config moves
  end_config ≠ ⟨1, 9, 8, 9⟩ := by
  sorry

/-- Lemma: S mod 3 is invariant under moves -/
lemma S_mod_3_invariant (config : SquareConfig) (move : Move) :
  (S config) % 3 = (S (apply_move config move)) % 3 := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_reach_target_l1423_142327


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1423_142357

/-- Given a geometric sequence of real numbers {a_n}, prove that if the sum of the first three terms is 2
    and the sum of the 4th, 5th, and 6th terms is 16, then the sum of the 7th, 8th, and 9th terms is 128. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 4 + a 5 + a 6 = 16) : a 7 + a 8 + a 9 = 128 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1423_142357


namespace NUMINAMATH_CALUDE_school_election_votes_l1423_142336

theorem school_election_votes (eliot_votes shaun_votes other_votes : ℕ) : 
  eliot_votes = 2 * shaun_votes →
  shaun_votes = 5 * other_votes →
  eliot_votes = 160 →
  other_votes = 16 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l1423_142336


namespace NUMINAMATH_CALUDE_circle_properties_l1423_142307

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The circle contains the point (2,0)
  circle_equation 2 0 ∧
  -- The line contains the point (2,0)
  line_equation 2 0 ∧
  -- The circle is tangent to the line at (2,0)
  ∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_equation x y ∧ line_equation x y →
      x = 2 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1423_142307


namespace NUMINAMATH_CALUDE_second_expression_proof_l1423_142323

theorem second_expression_proof (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 74) (h2 : a = 28) : x = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_proof_l1423_142323


namespace NUMINAMATH_CALUDE_smallest_a_is_eight_l1423_142340

/-- A function that represents the expression x^4 + a^2 + x^2 --/
def f (a x : ℤ) : ℤ := x^4 + a^2 + x^2

/-- A predicate that checks if a number is composite --/
def is_composite (n : ℤ) : Prop := ∃ (p q : ℤ), p ≠ 1 ∧ q ≠ 1 ∧ n = p * q

theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬is_composite (f a x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_eight_l1423_142340


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1423_142386

-- Define the constants
def regular_rate : ℝ := 12
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℝ := 63.62

-- Define the function to calculate total compensation
def total_compensation : ℝ :=
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings

-- Theorem statement
theorem bus_driver_compensation :
  total_compensation = 976.02 := by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1423_142386


namespace NUMINAMATH_CALUDE_problem_stack_total_l1423_142358

/-- Represents a stack of logs -/
structure LogStack where
  topRow : ℕ
  bottomRow : ℕ

/-- Calculates the total number of logs in a stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let n := stack.bottomRow - stack.topRow + 1
  n * (stack.topRow + stack.bottomRow) / 2

/-- The specific log stack described in the problem -/
def problemStack : LogStack := { topRow := 5, bottomRow := 15 }

/-- Theorem stating that the total number of logs in the problem stack is 110 -/
theorem problem_stack_total : totalLogs problemStack = 110 := by
  sorry

end NUMINAMATH_CALUDE_problem_stack_total_l1423_142358


namespace NUMINAMATH_CALUDE_books_from_first_shop_l1423_142338

/-- 
Proves that the number of books bought from the first shop is 65, given:
- Total cost of books from first shop is 1150
- 50 books were bought from the second shop for 920
- The average price per book is 18
-/
theorem books_from_first_shop : 
  ∀ (x : ℕ), 
  (1150 + 920 : ℚ) / (x + 50 : ℚ) = 18 → x = 65 := by
sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l1423_142338


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1423_142389

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 200) * ((Real.sqrt 180 / Real.sqrt 72) - (Real.sqrt 224 / Real.sqrt 56)) = Real.sqrt 10 - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1423_142389


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_nine_l1423_142380

theorem smallest_k_divisible_by_nine (k : ℕ) : k = 2024 ↔ 
  k > 2019 ∧ 
  (∀ m : ℕ, m > 2019 ∧ m < k → ¬(9 ∣ (m * (m + 1) / 2))) ∧ 
  (9 ∣ (k * (k + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_nine_l1423_142380


namespace NUMINAMATH_CALUDE_mean_temperature_l1423_142333

def temperatures : List ℝ := [79, 81, 83, 85, 84, 86, 88, 87, 85, 84]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 84.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1423_142333


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1423_142366

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and the length of the conjugate axis is twice that of the transverse axis (b = 2a),
    prove that its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_axis : b = 2 * a) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1423_142366


namespace NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1423_142346

/-- Given a right triangle XYZ with XY = 8 and YZ = 17, 
    the sum of the areas of squares constructed on sides YZ and XZ is 514. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 8^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 17^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →
  17^2 + ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 514 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1423_142346


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1423_142393

def f (a x : ℝ) := a * x^2 + x + 1

theorem quadratic_function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: Maximum value in the interval [-4, -2]
  (∀ x ∈ Set.Icc (-4) (-2), f a x ≤ (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  (∃ x ∈ Set.Icc (-4) (-2), f a x = (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  -- Part 2: Maximum value of a given root conditions
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 → f a x₂ = 0 → x₁ ≠ x₂ → x₁ / x₂ ∈ Set.Icc (1/10) 10 → a ≤ 1/4) ∧
  (∃ x₁ x₂ : ℝ, f (1/4) x₁ = 0 ∧ f (1/4) x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ / x₂ ∈ Set.Icc (1/10) 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1423_142393


namespace NUMINAMATH_CALUDE_quadratic_increasing_iff_m_gt_one_l1423_142394

/-- A quadratic function of the form y = x^2 + (m-3)x + m + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-3)*x + m + 1

/-- The derivative of the quadratic function with respect to x -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (m-3)

theorem quadratic_increasing_iff_m_gt_one (m : ℝ) :
  (∀ x > 1, ∀ h > 0, quadratic_function m (x + h) > quadratic_function m x) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_iff_m_gt_one_l1423_142394


namespace NUMINAMATH_CALUDE_clock_right_angle_time_l1423_142347

/-- The time (in minutes) between two consecutive instances of the clock hands forming a right angle after 7 PM -/
def time_between_right_angles : ℚ := 360 / 11

/-- The angle (in degrees) that the minute hand moves relative to the hour hand between two consecutive right angle formations -/
def relative_angle_change : ℚ := 180

theorem clock_right_angle_time :
  time_between_right_angles = 360 / 11 :=
by sorry

end NUMINAMATH_CALUDE_clock_right_angle_time_l1423_142347


namespace NUMINAMATH_CALUDE_intersection_distance_l1423_142335

-- Define the line C₁
def C₁ (x y : ℝ) : Prop := y - 2*x + 1 = 0

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1423_142335


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1423_142376

-- Problem 1
theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 := by sorry

-- Problem 2
theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 := by sorry

-- Problem 3
theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 := by sorry

-- Problem 4
theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1423_142376


namespace NUMINAMATH_CALUDE_card_difference_l1423_142356

/-- Given a total of 500 cards divided in the ratio of 11:9, prove that the difference between the larger share and the smaller share is 50 cards. -/
theorem card_difference (total : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (h1 : total = 500) (h2 : ratio_a = 11) (h3 : ratio_b = 9) : 
  (total * ratio_a) / (ratio_a + ratio_b) - (total * ratio_b) / (ratio_a + ratio_b) = 50 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l1423_142356


namespace NUMINAMATH_CALUDE_circle_through_pole_equation_l1423_142320

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * c.radius * Real.cos (p.θ - c.center.θ)

theorem circle_through_pole_equation 
  (c : PolarCircle) 
  (h1 : c.center = PolarPoint.mk (Real.sqrt 2) 0) 
  (h2 : c.radius = Real.sqrt 2) :
  ∀ (p : PolarPoint), circleEquation c p ↔ p.r = 2 * Real.sqrt 2 * Real.cos p.θ :=
by sorry

end NUMINAMATH_CALUDE_circle_through_pole_equation_l1423_142320


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1423_142373

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsScored := game.firstPartRunRate * game.firstPartOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 45)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.5)
  (h4 : game.targetRuns = 350) :
  requiredRunRate game = 9 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1423_142373


namespace NUMINAMATH_CALUDE_house_tower_difference_l1423_142301

/-- Represents the number of blocks Randy used for different purposes -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ

/-- Theorem stating the difference in blocks used for house and tower -/
theorem house_tower_difference (randy : BlockCounts)
  (h1 : randy.total = 90)
  (h2 : randy.house = 89)
  (h3 : randy.tower = 63) :
  randy.house - randy.tower = 26 := by
  sorry

end NUMINAMATH_CALUDE_house_tower_difference_l1423_142301


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l1423_142313

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is on the x-axis -/
def Point.onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Check if a point is on the y-axis -/
def Point.onYAxis (p : Point) : Prop :=
  p.x = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem line_equation_through_midpoint (m p q : Point) (l : Line) :
  m = Point.mk 1 (-2) →
  p.onXAxis →
  q.onYAxis →
  m.isMidpointOf p q →
  p.onLine l →
  q.onLine l →
  m.onLine l →
  l = Line.mk 2 (-1) (-4) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l1423_142313


namespace NUMINAMATH_CALUDE_union_determines_m_l1423_142342

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {2, 3}

theorem union_determines_m (m : ℝ) (h : A m ∪ B = {1, 2, 3}) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_m_l1423_142342


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1423_142392

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) (h5 : n ≥ 2) :
  a / (b + c)^(1/n : ℝ) + b / (c + a)^(1/n : ℝ) + c / (a + b)^(1/n : ℝ) ≥ 3 / 2^(1/n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1423_142392


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1423_142329

theorem rational_equation_solution (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x^2 - 3*x - 4) / (x - 4) = 3*x + k → x = (1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1423_142329


namespace NUMINAMATH_CALUDE_ludwig_earnings_l1423_142302

/-- Calculates the weekly earnings of a worker with given work schedule and daily salary. -/
def weeklyEarnings (totalDays : ℕ) (halfDays : ℕ) (dailySalary : ℚ) : ℚ :=
  let fullDays := totalDays - halfDays
  fullDays * dailySalary + halfDays * (dailySalary / 2)

/-- Theorem stating that under the given conditions, the weekly earnings are $55. -/
theorem ludwig_earnings :
  weeklyEarnings 7 3 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ludwig_earnings_l1423_142302


namespace NUMINAMATH_CALUDE_expression_evaluation_l1423_142304

theorem expression_evaluation (a b : ℝ) (ha : a = 6) (hb : b = 2) :
  3 / (a + b) + a^2 = 291 / 8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1423_142304


namespace NUMINAMATH_CALUDE_pool_time_ratio_l1423_142379

/-- The ratio of George's time to Elaine's time in the pool --/
def time_ratio (jerry_time elaine_time george_time : ℚ) : ℚ × ℚ :=
  (george_time, elaine_time)

theorem pool_time_ratio :
  ∀ (jerry_time elaine_time george_time total_time : ℚ),
    jerry_time = 3 →
    elaine_time = 2 * jerry_time →
    total_time = 11 →
    total_time = jerry_time + elaine_time + george_time →
    time_ratio jerry_time elaine_time george_time = (1, 3) := by
  sorry

#check pool_time_ratio

end NUMINAMATH_CALUDE_pool_time_ratio_l1423_142379


namespace NUMINAMATH_CALUDE_incorrect_calculation_D_l1423_142384

theorem incorrect_calculation_D :
  (∀ x : ℝ, x * 0 = 0) ∧
  (∀ x y : ℝ, y ≠ 0 → x / y = x * (1 / y)) ∧
  (∀ x y : ℝ, x * (-y) = -(x * y)) →
  ¬(1 / 3 / (-1) = 3 * (-1)) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_calculation_D_l1423_142384


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l1423_142354

/-- The distance in miles Wanda walks to school one way -/
def distance_to_school : ℝ := 0.5

/-- The number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks after the given number of weeks -/
def total_distance : ℝ :=
  distance_to_school * 2 * round_trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance :
  total_distance = 40 := by sorry

end NUMINAMATH_CALUDE_wanda_walking_distance_l1423_142354


namespace NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1423_142361

theorem truth_values_of_p_and_q (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_truth_values_of_p_and_q_l1423_142361


namespace NUMINAMATH_CALUDE_usual_time_to_school_l1423_142370

theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 →
  usual_time > 0 →
  (5/4 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l1423_142370


namespace NUMINAMATH_CALUDE_problem_solved_probability_l1423_142330

theorem problem_solved_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 2/3) 
  (h2 : prob_B = 3/4) 
  : prob_A + prob_B - prob_A * prob_B = 11/12 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solved_probability_l1423_142330


namespace NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l1423_142355

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l1423_142355


namespace NUMINAMATH_CALUDE_sum_three_numbers_l1423_142324

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 60 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  7 * c = N → 
  N = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l1423_142324


namespace NUMINAMATH_CALUDE_adjusted_work_schedule_earnings_l1423_142317

/-- Proves that the adjusted work schedule results in the same total earnings --/
theorem adjusted_work_schedule_earnings (initial_hours_per_week : ℝ) 
  (initial_weeks : ℕ) (missed_weeks : ℕ) (total_earnings : ℝ) 
  (adjusted_hours_per_week : ℝ) :
  initial_hours_per_week = 25 →
  initial_weeks = 15 →
  missed_weeks = 3 →
  total_earnings = 3750 →
  adjusted_hours_per_week = 31.25 →
  (initial_weeks - missed_weeks : ℝ) * adjusted_hours_per_week = initial_weeks * initial_hours_per_week :=
by sorry

end NUMINAMATH_CALUDE_adjusted_work_schedule_earnings_l1423_142317


namespace NUMINAMATH_CALUDE_brownie_cutting_l1423_142381

theorem brownie_cutting (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) : 
  pan_length * pan_width - (pan_length * pan_width / (piece_length * piece_width)) * (piece_length * piece_width) = 0 :=
by sorry

end NUMINAMATH_CALUDE_brownie_cutting_l1423_142381


namespace NUMINAMATH_CALUDE_base_eight_4372_equals_2298_l1423_142339

def base_eight_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_4372_equals_2298 :
  base_eight_to_decimal [2, 7, 3, 4] = 2298 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_4372_equals_2298_l1423_142339


namespace NUMINAMATH_CALUDE_sum_of_algebra_values_l1423_142350

-- Define the function that assigns numeric values to letters based on their position
def letterValue (position : ℕ) : ℤ :=
  match position % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 0 => -3
  | _ => 0  -- This case should never occur due to the modulo operation

-- Define the positions of letters in "ALGEBRA"
def algebraPositions : List ℕ := [1, 12, 7, 5, 2, 18, 1]

-- Theorem statement
theorem sum_of_algebra_values :
  (algebraPositions.map letterValue).sum = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_algebra_values_l1423_142350


namespace NUMINAMATH_CALUDE_markers_given_l1423_142395

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end NUMINAMATH_CALUDE_markers_given_l1423_142395


namespace NUMINAMATH_CALUDE_age_ratio_l1423_142343

theorem age_ratio (sum_ages : ℕ) (your_age : ℕ) : 
  sum_ages = 40 → your_age = 10 → (sum_ages - your_age) / your_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1423_142343


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l1423_142385

theorem sqrt_difference_equals_five_sixths :
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l1423_142385
