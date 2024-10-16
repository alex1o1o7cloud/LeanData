import Mathlib

namespace NUMINAMATH_CALUDE_late_train_speed_l310_31027

/-- Proves that given a journey of 15 km, if a train traveling at 100 kmph reaches the destination on time,
    and a train traveling at speed v kmph reaches the destination 15 minutes late, then v = 37.5 kmph. -/
theorem late_train_speed (journey_length : ℝ) (on_time_speed : ℝ) (late_time_diff : ℝ) (v : ℝ) :
  journey_length = 15 →
  on_time_speed = 100 →
  late_time_diff = 0.25 →
  journey_length / on_time_speed + late_time_diff = journey_length / v →
  v = 37.5 := by
  sorry

#check late_train_speed

end NUMINAMATH_CALUDE_late_train_speed_l310_31027


namespace NUMINAMATH_CALUDE_maple_to_pine_ratio_l310_31002

theorem maple_to_pine_ratio (total_trees : ℕ) (oaks : ℕ) (firs : ℕ) (palms : ℕ) 
  (h1 : total_trees = 150)
  (h2 : oaks = 20)
  (h3 : firs = 35)
  (h4 : palms = 25)
  (h5 : ∃ (maple pine : ℕ), total_trees = oaks + firs + palms + maple + pine ∧ maple = 2 * pine) :
  ∃ (m p : ℕ), m / p = 2 ∧ m > 0 ∧ p > 0 := by
  sorry

end NUMINAMATH_CALUDE_maple_to_pine_ratio_l310_31002


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l310_31030

theorem arithmetic_progression_implies_equal_numbers 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_arithmetic : (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2 = (a + b) / 2) : 
  a = b := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l310_31030


namespace NUMINAMATH_CALUDE_base6_divisible_by_13_l310_31081

def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

theorem base6_divisible_by_13 (d : Nat) :
  d ≤ 5 → (base6_to_base10 d % 13 = 0 ↔ d = 5) := by
  sorry

end NUMINAMATH_CALUDE_base6_divisible_by_13_l310_31081


namespace NUMINAMATH_CALUDE_committee_meeting_pencils_committee_meeting_pencils_correct_l310_31003

/-- The number of pencils brought to a committee meeting -/
theorem committee_meeting_pencils : ℕ :=
  let associate_prof : ℕ := 2  -- number of associate professors
  let assistant_prof : ℕ := 7  -- number of assistant professors
  let total_people : ℕ := 9    -- total number of people present
  let total_charts : ℕ := 16   -- total number of charts brought

  -- Each associate professor brings 2 pencils and 1 chart
  -- Each assistant professor brings 1 pencil and 2 charts
  -- The total number of pencils is what we want to prove
  have h1 : associate_prof + assistant_prof = total_people := by sorry
  have h2 : associate_prof + 2 * assistant_prof = total_charts := by sorry
  
  11

theorem committee_meeting_pencils_correct : committee_meeting_pencils = 11 := by sorry

end NUMINAMATH_CALUDE_committee_meeting_pencils_committee_meeting_pencils_correct_l310_31003


namespace NUMINAMATH_CALUDE_weekly_caloric_deficit_l310_31028

/-- Jonathan's daily caloric intake on regular days -/
def regularDailyIntake : ℕ := 2500

/-- Jonathan's extra caloric intake on Saturday -/
def saturdayExtraIntake : ℕ := 1000

/-- Jonathan's daily caloric burn -/
def dailyCaloriesBurned : ℕ := 3000

/-- Number of days in a week -/
def daysInWeek : ℕ := 7

/-- Number of regular intake days in a week -/
def regularIntakeDays : ℕ := 6

theorem weekly_caloric_deficit :
  (daysInWeek * dailyCaloriesBurned) - 
  (regularIntakeDays * regularDailyIntake + (regularDailyIntake + saturdayExtraIntake)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_caloric_deficit_l310_31028


namespace NUMINAMATH_CALUDE_base_conversion_1234_to_base_4_l310_31084

theorem base_conversion_1234_to_base_4 :
  (3 * 4^4 + 4 * 4^3 + 1 * 4^2 + 0 * 4^1 + 2 * 4^0) = 1234 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1234_to_base_4_l310_31084


namespace NUMINAMATH_CALUDE_average_of_numbers_l310_31011

theorem average_of_numbers (x : ℝ) : 
  ((x + 5) + 14 + x + 5) / 4 = 9 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_average_of_numbers_l310_31011


namespace NUMINAMATH_CALUDE_mod_twelve_five_eleven_l310_31035

theorem mod_twelve_five_eleven (m : ℕ) : 
  12^5 ≡ m [ZMOD 11] → 0 ≤ m → m < 11 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_five_eleven_l310_31035


namespace NUMINAMATH_CALUDE_points_difference_l310_31054

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of matches in a round-robin tournament -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_matches num_teams * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := total_matches num_teams * 2 * tie_points

/-- The theorem stating the difference between maximum and minimum total points -/
theorem points_difference :
  max_total_points - min_total_points = 30 := by sorry

end NUMINAMATH_CALUDE_points_difference_l310_31054


namespace NUMINAMATH_CALUDE_more_valid_placements_diff_intersections_l310_31068

/-- Represents the number of radial streets in city N -/
def radial_streets : ℕ := 7

/-- Represents the number of parallel streets in city N -/
def parallel_streets : ℕ := 7

/-- Total number of intersections in the city -/
def total_intersections : ℕ := radial_streets * parallel_streets

/-- Calculates the number of valid store placements when stores must not be at the same intersection -/
def valid_placements_diff_intersections : ℕ := total_intersections * (total_intersections - 1)

/-- Calculates the number of valid store placements when stores must not be on the same street -/
def valid_placements_diff_streets : ℕ := 
  valid_placements_diff_intersections - 2 * (radial_streets * (total_intersections - radial_streets))

/-- Theorem stating that the condition of different intersections allows more valid placements -/
theorem more_valid_placements_diff_intersections : 
  valid_placements_diff_intersections > valid_placements_diff_streets :=
sorry

end NUMINAMATH_CALUDE_more_valid_placements_diff_intersections_l310_31068


namespace NUMINAMATH_CALUDE_round_to_scientific_notation_l310_31006

/-- Rounds a real number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Converts a real number to scientific notation (a * 10^b form) -/
def toScientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem round_to_scientific_notation :
  let x := -29800000
  let sigFigs := 3
  let (a, b) := toScientificNotation (roundToSigFigs x sigFigs)
  a = -2.98 ∧ b = 7 := by sorry

end NUMINAMATH_CALUDE_round_to_scientific_notation_l310_31006


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l310_31099

theorem last_two_digits_sum (n : ℕ) : (9^23 + 11^23) % 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l310_31099


namespace NUMINAMATH_CALUDE_no_natural_solution_equation_l310_31032

theorem no_natural_solution_equation : ∀ x y : ℕ, 2^x + 21^x ≠ y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_equation_l310_31032


namespace NUMINAMATH_CALUDE_tower_painting_ways_level_painting_ways_l310_31034

/-- Represents the number of ways to paint a single level of the tower -/
def paint_level : ℕ := 96

/-- Represents the number of colors available for painting -/
def num_colors : ℕ := 3

/-- Represents the number of levels in the tower -/
def num_levels : ℕ := 3

/-- Theorem stating the number of ways to paint the entire tower -/
theorem tower_painting_ways :
  num_colors * paint_level * paint_level = 27648 :=
by sorry

/-- Theorem stating the number of ways to paint a single level -/
theorem level_painting_ways :
  paint_level = 96 :=
by sorry

end NUMINAMATH_CALUDE_tower_painting_ways_level_painting_ways_l310_31034


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l310_31075

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- The first line: ax + 2y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 1 = 0

/-- The second line: x + y + 4 = 0 -/
def line2 (x y : ℝ) : Prop :=
  x + y + 4 = 0

theorem lines_parallel_iff (a : ℝ) :
  parallel a 2 1 1 1 4 ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l310_31075


namespace NUMINAMATH_CALUDE_equilateral_side_length_l310_31014

/-- Given a diagram with an equilateral triangle and a right-angled triangle,
    where the right-angled triangle has a side length of 6 and both triangles
    have a 45-degree angle, the side length y of the equilateral triangle is 6√2. -/
theorem equilateral_side_length (y : ℝ) : y = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_side_length_l310_31014


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l310_31048

/-- A polyhedron formed by a unit square base and four points above its vertices -/
structure UnitSquarePolyhedron where
  -- Heights of the points above the unit square vertices
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The volume of the UnitSquarePolyhedron -/
def volume (p : UnitSquarePolyhedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific polyhedron is 4.5 -/
theorem specific_polyhedron_volume :
  ∃ (p : UnitSquarePolyhedron),
    p.h1 = 3 ∧ p.h2 = 4 ∧ p.h3 = 6 ∧ p.h4 = 5 ∧
    volume p = 4.5 :=
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l310_31048


namespace NUMINAMATH_CALUDE_painted_faces_count_l310_31064

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Calculates the number of smaller cubes with at least two painted faces -/
def cubes_with_two_or_more_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : 
  cubes_with_two_or_more_painted_faces c = 32 := by sorry

end NUMINAMATH_CALUDE_painted_faces_count_l310_31064


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l310_31042

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt ((x + 3)^2) = |x + 3| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l310_31042


namespace NUMINAMATH_CALUDE_cupcake_cost_split_l310_31049

theorem cupcake_cost_split (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  cost_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * cost_per_cupcake / (num_people : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_cost_split_l310_31049


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l310_31079

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_3 : a 3 = 4) (h_6 : a 6 = 1/2) : a 4 + a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l310_31079


namespace NUMINAMATH_CALUDE_unique_a_for_set_equality_l310_31026

theorem unique_a_for_set_equality : ∃! a : ℝ, ({1, 3, a^2} ∪ {1, a+2} : Set ℝ) = {1, 3, a^2} := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_set_equality_l310_31026


namespace NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l310_31010

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (∀ x, HasDerivAt (f a) ((3 * a * x^2) + 8 * x + 3) x) →
  HasDerivAt (f a) 2 1 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l310_31010


namespace NUMINAMATH_CALUDE_aron_dusting_days_l310_31009

/-- Represents the cleaning schedule and durations for Aron -/
structure CleaningSchedule where
  vacuumingTimePerDay : ℕ
  vacuumingDaysPerWeek : ℕ
  dustingTimePerDay : ℕ
  totalCleaningTimePerWeek : ℕ

/-- Calculates the number of days Aron spends dusting per week -/
def dustingDaysPerWeek (schedule : CleaningSchedule) : ℕ :=
  let totalVacuumingTime := schedule.vacuumingTimePerDay * schedule.vacuumingDaysPerWeek
  let totalDustingTime := schedule.totalCleaningTimePerWeek - totalVacuumingTime
  totalDustingTime / schedule.dustingTimePerDay

/-- Theorem stating that Aron spends 2 days a week dusting -/
theorem aron_dusting_days (schedule : CleaningSchedule)
    (h1 : schedule.vacuumingTimePerDay = 30)
    (h2 : schedule.vacuumingDaysPerWeek = 3)
    (h3 : schedule.dustingTimePerDay = 20)
    (h4 : schedule.totalCleaningTimePerWeek = 130) :
    dustingDaysPerWeek schedule = 2 := by
  sorry

#eval dustingDaysPerWeek {
  vacuumingTimePerDay := 30,
  vacuumingDaysPerWeek := 3,
  dustingTimePerDay := 20,
  totalCleaningTimePerWeek := 130
}

end NUMINAMATH_CALUDE_aron_dusting_days_l310_31009


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l310_31094

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- A one-digit number is a natural number less than 10. -/
def isOneDigit (n : ℕ) : Prop := n < 10

/-- A two-digit number is a natural number greater than or equal to 10 and less than 100. -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The two smallest one-digit primes are 2 and 3. -/
axiom smallest_one_digit_primes : ∀ n : ℕ, isPrime n → isOneDigit n → n = 2 ∨ n = 3

/-- The smallest two-digit prime is 11. -/
axiom smallest_two_digit_prime : ∀ n : ℕ, isPrime n → isTwoDigit n → n ≥ 11

theorem product_of_smallest_primes : 
  ∃ p q r : ℕ, 
    isPrime p ∧ isOneDigit p ∧
    isPrime q ∧ isOneDigit q ∧
    isPrime r ∧ isTwoDigit r ∧
    p * q * r = 66 :=
sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l310_31094


namespace NUMINAMATH_CALUDE_perpendicular_vectors_exist_minimum_dot_product_l310_31023

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![5, 1]
def OB : Fin 2 → ℝ := ![1, 7]
def OC : Fin 2 → ℝ := ![4, 2]

/-- Vector OM as a function of t -/
def OM (t : ℝ) : Fin 2 → ℝ := fun i => t * OC i

/-- Vector MA as a function of t -/
def MA (t : ℝ) : Fin 2 → ℝ := fun i => OA i - OM t i

/-- Vector MB as a function of t -/
def MB (t : ℝ) : Fin 2 → ℝ := fun i => OB i - OM t i

/-- Dot product of MA and MB -/
def MA_dot_MB (t : ℝ) : ℝ := (MA t 0) * (MB t 0) + (MA t 1) * (MB t 1)

theorem perpendicular_vectors_exist :
  ∃ t : ℝ, MA_dot_MB t = 0 ∧ t = (5 + Real.sqrt 10) / 5 ∨ t = (5 - Real.sqrt 10) / 5 := by
  sorry

theorem minimum_dot_product :
  ∃ t : ℝ, ∀ s : ℝ, MA_dot_MB t ≤ MA_dot_MB s ∧ OM t = OC := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_exist_minimum_dot_product_l310_31023


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l310_31039

/-- A moving circle passes through a fixed point F(2, 0) and is tangent to the line x = -2.
    The trajectory of its center C is a parabola. -/
theorem trajectory_of_moving_circle_center (C : ℝ × ℝ) : 
  (∃ (r : ℝ), (C.1 - 2)^2 + C.2^2 = r^2 ∧ (C.1 + 2)^2 + C.2^2 = r^2) →
  C.2^2 = 8 * C.1 := by
  sorry


end NUMINAMATH_CALUDE_trajectory_of_moving_circle_center_l310_31039


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l310_31088

def A : Set ℝ := {x | x^2 - 1 = 0}
def B : Set ℝ := {-1, 2, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l310_31088


namespace NUMINAMATH_CALUDE_sticker_count_after_loss_l310_31056

/-- Given a number of stickers per page, an initial number of pages, and a number of lost pages,
    calculate the total number of remaining stickers. -/
def remaining_stickers (stickers_per_page : ℕ) (initial_pages : ℕ) (lost_pages : ℕ) : ℕ :=
  (initial_pages - lost_pages) * stickers_per_page

theorem sticker_count_after_loss :
  remaining_stickers 20 12 1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_after_loss_l310_31056


namespace NUMINAMATH_CALUDE_total_clothes_donated_l310_31022

/-- Proves that the total number of clothes donated is 87 given the specified conditions --/
theorem total_clothes_donated (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 12 →
  pants = 5 * shirts →
  shorts = pants / 4 →
  shirts + pants + shorts = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_donated_l310_31022


namespace NUMINAMATH_CALUDE_ratio_of_shares_l310_31066

theorem ratio_of_shares (total amount_c : ℕ) (h1 : total = 2000) (h2 : amount_c = 1600) :
  (total - amount_c) / amount_c = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_shares_l310_31066


namespace NUMINAMATH_CALUDE_monotonicity_condition_equiv_a_range_l310_31037

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

/-- Theorem stating the equivalence between the monotonicity condition and the range of a -/
theorem monotonicity_condition_equiv_a_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ a ∈ Set.Icc (-3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_equiv_a_range_l310_31037


namespace NUMINAMATH_CALUDE_susie_savings_account_l310_31018

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem susie_savings_account :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let years : ℕ := 21
  let result := compound_interest principal rate years
  ∃ ε > 0, |result - 8017.84| < ε :=
sorry

end NUMINAMATH_CALUDE_susie_savings_account_l310_31018


namespace NUMINAMATH_CALUDE_fish_disappeared_l310_31060

def original_goldfish : ℕ := 7
def original_catfish : ℕ := 12
def original_guppies : ℕ := 8
def original_angelfish : ℕ := 5
def current_total : ℕ := 27

theorem fish_disappeared : 
  original_goldfish + original_catfish + original_guppies + original_angelfish - current_total = 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_disappeared_l310_31060


namespace NUMINAMATH_CALUDE_statement_equivalence_l310_31016

/-- Represents the property of being happy -/
def happy : Prop := sorry

/-- Represents the property of possessing the food item -/
def possess : Prop := sorry

/-- The statement "Happy people all possess it" -/
def original_statement : Prop := happy → possess

/-- The statement "People who do not possess it are unhappy" -/
def equivalent_statement : Prop := ¬possess → ¬happy

/-- Theorem stating that the original statement is logically equivalent to the equivalent statement -/
theorem statement_equivalence : original_statement ↔ equivalent_statement :=
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l310_31016


namespace NUMINAMATH_CALUDE_base8_of_215_l310_31086

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base8_of_215 : toBase8 215 = 327 := by sorry

end NUMINAMATH_CALUDE_base8_of_215_l310_31086


namespace NUMINAMATH_CALUDE_probability_marked_standard_deck_l310_31065

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (total_ranks : ℕ)
  (total_suits : ℕ)
  (marked_ranks : ℕ)

/-- A standard deck with 52 cards, 13 ranks, 4 suits, and 4 marked ranks -/
def standard_deck : Deck :=
  { total_cards := 52,
    total_ranks := 13,
    total_suits := 4,
    marked_ranks := 4 }

/-- The probability of drawing a card with a special symbol -/
def probability_marked (d : Deck) : ℚ :=
  (d.marked_ranks * d.total_suits) / d.total_cards

/-- Theorem: The probability of drawing a card with a special symbol from a standard deck is 4/13 -/
theorem probability_marked_standard_deck :
  probability_marked standard_deck = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_marked_standard_deck_l310_31065


namespace NUMINAMATH_CALUDE_reciprocal_opposite_theorem_l310_31000

theorem reciprocal_opposite_theorem (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (c + d)^2 - a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_theorem_l310_31000


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l310_31062

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_multiple : 
  (∀ k : ℕ, is_three_digit k ∧ 3 ∣ k ∧ 7 ∣ k ∧ 11 ∣ k → 231 ≤ k) ∧ 
  is_three_digit 231 ∧ 3 ∣ 231 ∧ 7 ∣ 231 ∧ 11 ∣ 231 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l310_31062


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l310_31036

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l310_31036


namespace NUMINAMATH_CALUDE_solution_set_l310_31076

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := x^(lg x) = x^3 / 100

-- Theorem statement
theorem solution_set : 
  {x : ℝ | equation x} = {10, 100} :=
sorry

end NUMINAMATH_CALUDE_solution_set_l310_31076


namespace NUMINAMATH_CALUDE_total_distance_is_963_l310_31008

/-- The total combined distance of objects thrown by Bill, Ted, and Alice -/
def total_distance (ted_sticks ted_rocks : ℕ) 
  (bill_stick_dist bill_rock_dist : ℝ) : ℝ :=
  let bill_sticks := ted_sticks - 6
  let alice_sticks := ted_sticks / 2
  let bill_rocks := ted_rocks / 2
  let alice_rocks := bill_rocks * 3
  let ted_stick_dist := bill_stick_dist * 1.5
  let alice_stick_dist := bill_stick_dist * 2
  let ted_rock_dist := bill_rock_dist * 1.25
  let alice_rock_dist := bill_rock_dist * 3
  (bill_sticks : ℝ) * bill_stick_dist +
  (ted_sticks : ℝ) * ted_stick_dist +
  (alice_sticks : ℝ) * alice_stick_dist +
  (bill_rocks : ℝ) * bill_rock_dist +
  (ted_rocks : ℝ) * ted_rock_dist +
  (alice_rocks : ℝ) * alice_rock_dist

/-- Theorem stating the total distance is 963 meters given the problem conditions -/
theorem total_distance_is_963 :
  total_distance 12 18 8 6 = 963 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_963_l310_31008


namespace NUMINAMATH_CALUDE_line_proof_l310_31031

-- Define the lines
def line1 (x y : ℝ) : Prop := 4 * x + 2 * y + 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 2 * y + 9 = 0
def line3 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def result_line (x y : ℝ) : Prop := 4 * x - 2 * y + 11 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular
      ((4 : ℝ) / 2) -- Slope of result_line
      ((-1 : ℝ) / 2) -- Slope of line3
  := by sorry

end NUMINAMATH_CALUDE_line_proof_l310_31031


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l310_31095

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem otimes_inequality_implies_a_range :
  (∀ x ∈ Set.Icc 1 2, otimes (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l310_31095


namespace NUMINAMATH_CALUDE_units_digit_problem_l310_31007

theorem units_digit_problem : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 10 = 3) ∧ 
  ((35^87 + x^53) % 10 = 8) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l310_31007


namespace NUMINAMATH_CALUDE_cistern_length_is_nine_l310_31071

/-- Represents a rectangular cistern with water -/
structure WaterCistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  totalWetArea : ℝ

/-- Calculates the wet surface area of a cistern -/
def wetSurfaceArea (c : WaterCistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern with given parameters is 9 meters -/
theorem cistern_length_is_nine :
  ∃ (c : WaterCistern),
    c.width = 4 ∧
    c.depth = 1.25 ∧
    c.totalWetArea = 68.5 ∧
    wetSurfaceArea c = c.totalWetArea ∧
    c.length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_is_nine_l310_31071


namespace NUMINAMATH_CALUDE_alien_gems_count_l310_31005

/-- Converts a number from base 6 to base 10 --/
def base6To10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- The number of gems the alien has --/
def alienGems : ℕ := base6To10 2 5 6

theorem alien_gems_count : alienGems = 108 := by
  sorry

end NUMINAMATH_CALUDE_alien_gems_count_l310_31005


namespace NUMINAMATH_CALUDE_not_in_third_quadrant_l310_31063

def linear_function (x : ℝ) : ℝ := -2 * x + 5

theorem not_in_third_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_not_in_third_quadrant_l310_31063


namespace NUMINAMATH_CALUDE_cosine_tangent_ratio_equals_two_l310_31083

theorem cosine_tangent_ratio_equals_two : 
  (Real.cos (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.cos (50 * π / 180)) = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_tangent_ratio_equals_two_l310_31083


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l310_31029

def total_players : ℕ := 16
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def starters : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_starters :
  choose (total_players - num_triplets - num_twins) (starters - 1 - 1) * num_triplets * num_twins = 2772 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l310_31029


namespace NUMINAMATH_CALUDE_connie_gave_juan_marbles_l310_31001

/-- The number of marbles Connie gave to Juan -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Connie gave 183 marbles to Juan -/
theorem connie_gave_juan_marbles : marbles_given 776 593 = 183 := by
  sorry

end NUMINAMATH_CALUDE_connie_gave_juan_marbles_l310_31001


namespace NUMINAMATH_CALUDE_juggler_path_radius_l310_31020

/-- The equation of the path described by the juggler's balls -/
def path_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 2*x + 4*y

/-- The radius of the path described by the juggler's balls -/
def path_radius : ℝ := 0

/-- Theorem stating that the radius of the path is 0 -/
theorem juggler_path_radius :
  ∀ x y : ℝ, path_equation x y → (x - 1)^2 + (y - 2)^2 = path_radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_juggler_path_radius_l310_31020


namespace NUMINAMATH_CALUDE_cup_stacking_l310_31015

theorem cup_stacking (a₁ a₂ a₄ a₅ : ℕ) (h1 : a₁ = 17) (h2 : a₂ = 21) (h4 : a₄ = 29) (h5 : a₅ = 33)
  (h_pattern : ∃ d : ℕ, a₂ = a₁ + d ∧ a₄ = a₂ + 2*d ∧ a₅ = a₄ + d) :
  ∃ a₃ : ℕ, a₃ = 25 ∧ a₃ = a₂ + (a₂ - a₁) := by
  sorry

end NUMINAMATH_CALUDE_cup_stacking_l310_31015


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l310_31050

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S T : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (quad : Quadrilateral) : Prop :=
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d quad.P quad.T = 5 ∧
  d quad.T quad.R = 7 ∧
  d quad.Q quad.T = 6 ∧
  d quad.T quad.S = 2 ∧
  d quad.P quad.Q = 5

-- Theorem statement
theorem quadrilateral_diagonal_length 
  (quad : Quadrilateral) 
  (h : satisfies_conditions quad) : 
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d quad.P quad.S = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l310_31050


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l310_31019

theorem same_remainder_divisor : ∃ (r : ℕ), 
  1108 % 23 = r ∧ 
  1453 % 23 = r ∧ 
  1844 % 23 = r ∧ 
  2281 % 23 = r :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l310_31019


namespace NUMINAMATH_CALUDE_probability_all_red_fourth_draw_correct_l310_31043

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing all red balls exactly after the 4th draw -/
def probability_all_red_fourth_draw : ℝ := 0.0434

/-- Theorem stating the probability of drawing all red balls exactly after the 4th draw -/
theorem probability_all_red_fourth_draw_correct :
  probability_all_red_fourth_draw = 
    (initial_red_balls / total_balls) * 
    ((initial_white_balls + 1) / total_balls) * 
    (initial_red_balls / total_balls) * 
    (1 / (initial_white_balls + 1)) := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_fourth_draw_correct_l310_31043


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l310_31021

/-- Given a survey of households and their soap brand preferences, 
    prove the ratio of households using only brand B to those using both brands. -/
theorem soap_brand_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_w : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_w = 60)
  (h4 : both = 40)
  : (total - neither - only_w - both) / both = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l310_31021


namespace NUMINAMATH_CALUDE_sum_of_roots_l310_31089

theorem sum_of_roots (h b x₁ x₂ : ℝ) (hx : x₁ ≠ x₂) 
  (eq₁ : 4 * x₁^2 - h * x₁ = b) (eq₂ : 4 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l310_31089


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l310_31004

theorem min_cut_length_for_non_triangle (a b c : ℕ) (ha : a = 9) (hb : b = 16) (hc : c = 18) :
  ∃ (x : ℕ), x = 8 ∧
  (∀ (y : ℕ), y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l310_31004


namespace NUMINAMATH_CALUDE_average_income_proof_l310_31046

def family_size : ℕ := 4

def income_1 : ℕ := 8000
def income_2 : ℕ := 15000
def income_3 : ℕ := 6000
def income_4 : ℕ := 11000

def total_income : ℕ := income_1 + income_2 + income_3 + income_4

theorem average_income_proof :
  total_income / family_size = 10000 := by
  sorry

end NUMINAMATH_CALUDE_average_income_proof_l310_31046


namespace NUMINAMATH_CALUDE_parallel_line_length_l310_31053

/-- A triangle with a base of 20 inches and height of 10 inches, 
    divided into four equal areas by two parallel lines -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  baseParallel : ℝ
  base_eq : base = 20
  height_eq : height = 10
  equal_areas : baseParallel > 0 ∧ baseParallel < base

theorem parallel_line_length (t : DividedTriangle) : t.baseParallel = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l310_31053


namespace NUMINAMATH_CALUDE_unique_solution_l310_31038

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x^2 - 5 * x + 6 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l310_31038


namespace NUMINAMATH_CALUDE_evaluate_expression_l310_31055

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l310_31055


namespace NUMINAMATH_CALUDE_rationalize_denominator_l310_31052

theorem rationalize_denominator (x : ℝ) :
  x > 0 → (45 * Real.sqrt 3) / Real.sqrt x = 3 * Real.sqrt 15 ↔ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l310_31052


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l310_31013

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2*x + 3 - x^2 > 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l310_31013


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l310_31070

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, x = 1 → k^2 * x^2 - 6*k*x + 8 ≥ 0) → 
  (k ≥ 4 ∨ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l310_31070


namespace NUMINAMATH_CALUDE_max_area_right_triangle_in_semicircle_l310_31098

theorem max_area_right_triangle_in_semicircle :
  let R : ℝ := 1/2
  let semicircle := {(x, y) : ℝ × ℝ | x^2 + y^2 = R^2 ∧ y ≥ 0}
  let triangle := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 2*R ∧ 0 ≤ y ∧ (x, y) ∈ semicircle}
  let area (x : ℝ) : ℝ := x * Real.sqrt (R^2 - x^2) / 2
  ∃ (x : ℝ), x ∈ Set.Icc 0 (2*R) ∧ 
    (∀ (y : ℝ), y ∈ Set.Icc 0 (2*R) → area y ≤ area x) ∧
    area x = 3 * Real.sqrt 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_in_semicircle_l310_31098


namespace NUMINAMATH_CALUDE_tom_fishing_probability_l310_31097

-- Define the weather conditions
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

-- Define the probability of Tom going fishing for each weather condition
def fishing_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.7
  | Weather.Rainy => 0.3
  | Weather.Cloudy => 0.5

-- Define the probability of each weather condition
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.3
  | Weather.Rainy => 0.5
  | Weather.Cloudy => 0.2

-- Theorem stating the probability of Tom going fishing
theorem tom_fishing_probability :
  (fishing_prob Weather.Sunny * weather_prob Weather.Sunny +
   fishing_prob Weather.Rainy * weather_prob Weather.Rainy +
   fishing_prob Weather.Cloudy * weather_prob Weather.Cloudy) = 0.46 := by
  sorry


end NUMINAMATH_CALUDE_tom_fishing_probability_l310_31097


namespace NUMINAMATH_CALUDE_smallest_values_for_equation_l310_31033

theorem smallest_values_for_equation (a b c : ℤ) 
  (ha : a > 2) (hb : b < 10) (hc : c ≥ 0) 
  (heq : 32 = a + 2*b + 3*c) : 
  a = 4 ∧ b = 8 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_values_for_equation_l310_31033


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_l310_31090

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

theorem sum_of_consecutive_odd_primes (p q : ℕ) (h : ConsecutivePrimes p q) (hp_odd : Odd p) (hq_odd : Odd q) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ p + q = 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_l310_31090


namespace NUMINAMATH_CALUDE_workshop_workers_workshop_workers_proof_l310_31067

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers : ℕ :=
  let avg_salary : ℚ := 8000
  let num_technicians : ℕ := 7
  let avg_salary_technicians : ℚ := 18000
  let avg_salary_others : ℚ := 6000
  42

/-- Proof that the total number of workers is 42 -/
theorem workshop_workers_proof : workshop_workers = 42 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_workshop_workers_proof_l310_31067


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l310_31091

theorem choose_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l310_31091


namespace NUMINAMATH_CALUDE_stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l310_31073

/-- Given that Jake has 3 fewer peaches and 4 more apples than Steven, and Steven has 19 apples,
    prove that the difference between Steven's apples and peaches is 19 - P,
    where P is the number of peaches Steven has. -/
theorem stevens_apple_peach_difference (P : ℕ) : ℕ → Prop :=
  let steven_apples : ℕ := 19
  let steven_peaches : ℕ := P
  let jake_peaches : ℕ := P - 3
  let jake_apples : ℕ := steven_apples + 4
  λ _ => steven_apples - steven_peaches = 19 - P

/-- Proof of the theorem -/
theorem prove_stevens_apple_peach_difference (P : ℕ) :
  stevens_apple_peach_difference P P :=
by
  sorry

end NUMINAMATH_CALUDE_stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l310_31073


namespace NUMINAMATH_CALUDE_jills_race_time_l310_31059

/-- Proves that Jill's race time is 32 seconds -/
theorem jills_race_time (jack_first_half : ℕ) (jack_second_half : ℕ) (time_difference : ℕ) :
  jack_first_half = 19 →
  jack_second_half = 6 →
  time_difference = 7 →
  jack_first_half + jack_second_half + time_difference = 32 :=
by sorry

end NUMINAMATH_CALUDE_jills_race_time_l310_31059


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l310_31045

theorem fraction_equality_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → (D * x - 13) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l310_31045


namespace NUMINAMATH_CALUDE_original_bales_count_l310_31012

theorem original_bales_count (bales_stacked bales_now : ℕ) 
  (h1 : bales_stacked = 26)
  (h2 : bales_now = 54) :
  bales_now - bales_stacked = 28 := by
  sorry

end NUMINAMATH_CALUDE_original_bales_count_l310_31012


namespace NUMINAMATH_CALUDE_minimum_value_at_one_l310_31074

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (a + 1) * x^2 - (a^2 + 3*a - 3) * x

theorem minimum_value_at_one (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_at_one_l310_31074


namespace NUMINAMATH_CALUDE_find_b_l310_31044

-- Define the real number √3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Define the equation (1 + √3)^5 = a + b√3
def equation (a b : ℚ) : Prop := (1 + sqrt3) ^ 5 = a + b * sqrt3

-- Theorem statement
theorem find_b : ∃ (a b : ℚ), equation a b ∧ b = 44 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l310_31044


namespace NUMINAMATH_CALUDE_countable_planar_graph_coloring_l310_31093

-- Define a type for colors
inductive Color
| blue
| red
| green

-- Define a type for graphs
structure Graph (α : Type) where
  vertices : Set α
  edges : Set (α × α)

-- Define what it means for a graph to be planar
def isPlanar {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a graph to be countable
def isCountable {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a cycle to be odd
def isOddCycle {α : Type} (G : Graph α) (cycle : List α) : Prop := sorry

-- Define what it means for a coloring to be valid (no odd monochromatic cycles)
def isValidColoring {α : Type} (G : Graph α) (coloring : α → Color) : Prop :=
  ∀ cycle, isOddCycle G cycle → ∃ v ∈ cycle, ∃ w ∈ cycle, coloring v ≠ coloring w

-- The main theorem
theorem countable_planar_graph_coloring 
  {α : Type} (G : Graph α) 
  (h_planar : isPlanar G) 
  (h_countable : isCountable G) 
  (h_finite : ∀ (H : Graph α), isPlanar H → (Finite H.vertices) → 
    ∃ coloring : α → Color, isValidColoring H coloring) :
  ∃ coloring : α → Color, isValidColoring G coloring := by
  sorry

end NUMINAMATH_CALUDE_countable_planar_graph_coloring_l310_31093


namespace NUMINAMATH_CALUDE_intersection_circle_radius_squared_l310_31047

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at four points. 
    All four points lie on a circle. This theorem proves that the radius squared 
    of this circle is 1/4. -/
theorem intersection_circle_radius_squared (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 6 = (y + 1)^2) → 
  ((x - 3/2)^2 + (y + 3/2)^2 = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_intersection_circle_radius_squared_l310_31047


namespace NUMINAMATH_CALUDE_big_sale_commission_l310_31096

theorem big_sale_commission 
  (sales_before : ℕ)
  (total_sales : ℕ)
  (avg_increase : ℚ)
  (new_avg : ℚ) :
  sales_before = 5 →
  total_sales = 6 →
  avg_increase = 150 →
  new_avg = 250 →
  (total_sales * new_avg - sales_before * (new_avg - avg_increase)) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_big_sale_commission_l310_31096


namespace NUMINAMATH_CALUDE_condition_relationship_l310_31041

theorem condition_relationship :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l310_31041


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l310_31069

/-- Given a hyperbola with equation x²/a - y²/2 = 1 and one asymptote 2x - y = 0, 
    prove that a = 1/2 -/
theorem hyperbola_asymptote (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a - y^2 / 2 = 1 → (2*x - y = 0 ∨ 2*x + y = 0)) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l310_31069


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l310_31061

/-- Definition of an ellipse with given center and axis lengths -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Theorem: Sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h1 : e.center = (-3, 4)) 
  (h2 : e.semi_major_axis = 7) 
  (h3 : e.semi_minor_axis = 2) : 
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

/-- Main theorem to be proved -/
theorem specific_ellipse_sum : 
  ∃ (e : Ellipse), e.center = (-3, 4) ∧ e.semi_major_axis = 7 ∧ e.semi_minor_axis = 2 ∧
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l310_31061


namespace NUMINAMATH_CALUDE_jade_handled_83_transactions_l310_31072

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
def cal_transactions : ℕ := anthony_transactions * 2 / 3
def jade_transactions : ℕ := cal_transactions + 17

-- Theorem to prove
theorem jade_handled_83_transactions : jade_transactions = 83 := by
  sorry

end NUMINAMATH_CALUDE_jade_handled_83_transactions_l310_31072


namespace NUMINAMATH_CALUDE_imon_entanglement_reduction_l310_31092

/-- Represents a graph of imons and their entanglements -/
structure ImonGraph where
  vertices : Set ℕ
  edges : Set (ℕ × ℕ)

/-- Operation 1: Remove a vertex with odd degree -/
def removeOddDegreeVertex (G : ImonGraph) (v : ℕ) : ImonGraph :=
  sorry

/-- Operation 2: Duplicate the graph and connect each vertex to its duplicate -/
def duplicateGraph (G : ImonGraph) : ImonGraph :=
  sorry

/-- Predicate to check if a graph has no edges -/
def hasNoEdges (G : ImonGraph) : Prop :=
  G.edges = ∅

/-- Main theorem: There exists a sequence of operations to reduce any ImonGraph to one with no edges -/
theorem imon_entanglement_reduction (G : ImonGraph) :
  ∃ (seq : List (ImonGraph → ImonGraph)), hasNoEdges (seq.foldl (λ g f => f g) G) :=
  sorry

end NUMINAMATH_CALUDE_imon_entanglement_reduction_l310_31092


namespace NUMINAMATH_CALUDE_marks_departure_time_correct_l310_31087

-- Define the problem parameters
def robs_normal_time : ℝ := 1
def robs_additional_time : ℝ := 0.5
def marks_normal_time_factor : ℝ := 3
def marks_time_reduction : ℝ := 0.2
def time_zone_difference : ℝ := 2
def robs_departure_time : ℝ := 11

-- Define the function to calculate Mark's departure time
def calculate_marks_departure_time : ℝ :=
  let robs_travel_time := robs_normal_time + robs_additional_time
  let marks_normal_time := marks_normal_time_factor * robs_normal_time
  let marks_travel_time := marks_normal_time * (1 - marks_time_reduction)
  let robs_arrival_time := robs_departure_time + robs_travel_time
  let marks_arrival_time := robs_arrival_time + time_zone_difference
  marks_arrival_time - marks_travel_time

-- Theorem statement
theorem marks_departure_time_correct :
  calculate_marks_departure_time = 11 + 36 / 60 :=
sorry

end NUMINAMATH_CALUDE_marks_departure_time_correct_l310_31087


namespace NUMINAMATH_CALUDE_kittens_sold_count_l310_31077

/-- Represents the pet store's sales scenario -/
structure PetStoreSales where
  kitten_price : ℕ
  puppy_price : ℕ
  total_revenue : ℕ
  puppy_count : ℕ

/-- Calculates the number of kittens sold -/
def kittens_sold (s : PetStoreSales) : ℕ :=
  (s.total_revenue - s.puppy_price * s.puppy_count) / s.kitten_price

/-- Theorem stating the number of kittens sold -/
theorem kittens_sold_count (s : PetStoreSales) 
  (h1 : s.kitten_price = 6)
  (h2 : s.puppy_price = 5)
  (h3 : s.total_revenue = 17)
  (h4 : s.puppy_count = 1) :
  kittens_sold s = 2 := by
  sorry

end NUMINAMATH_CALUDE_kittens_sold_count_l310_31077


namespace NUMINAMATH_CALUDE_point_s_y_coordinate_l310_31051

/-- Given two points R(-3, 4) and S(5, y) in a coordinate system, 
    prove that y = 8 when the slope of the line through R and S is 1/2 -/
theorem point_s_y_coordinate (y : ℝ) : 
  let R : ℝ × ℝ := (-3, 4)
  let S : ℝ × ℝ := (5, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = 1/2 → y = 8 := by
sorry

end NUMINAMATH_CALUDE_point_s_y_coordinate_l310_31051


namespace NUMINAMATH_CALUDE_fraction_addition_l310_31017

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l310_31017


namespace NUMINAMATH_CALUDE_min_both_like_problem_l310_31057

def min_both_like (total surveyed beethoven_fans chopin_fans both_and_vivaldi : ℕ) : ℕ :=
  max (beethoven_fans + chopin_fans - total) both_and_vivaldi

theorem min_both_like_problem :
  let total := 200
  let beethoven_fans := 150
  let chopin_fans := 120
  let both_and_vivaldi := 80
  min_both_like total beethoven_fans chopin_fans both_and_vivaldi = 80 := by
sorry

end NUMINAMATH_CALUDE_min_both_like_problem_l310_31057


namespace NUMINAMATH_CALUDE_replaced_person_weight_l310_31082

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l310_31082


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_1_2_l310_31058

/-- Equation of motion for an object -/
def s (t : ℝ) : ℝ := 2 * (1 - t^2)

/-- Instantaneous velocity at time t -/
def v (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 : v 1.2 = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_1_2_l310_31058


namespace NUMINAMATH_CALUDE_track_circumference_l310_31025

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B has traveled when they first meet -/
  first_meeting : ℝ
  /-- The distance A is shy of completing a full lap at the second meeting -/
  second_meeting : ℝ

/-- Theorem stating that under the given conditions, the track's circumference is 720 yards -/
theorem track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting = 150)
  (h2 : track.second_meeting = 90)
  (h3 : track.circumference > 0) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end NUMINAMATH_CALUDE_track_circumference_l310_31025


namespace NUMINAMATH_CALUDE_factors_of_180_l310_31080

/-- The number of positive factors of 180 is 18 -/
theorem factors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_180_l310_31080


namespace NUMINAMATH_CALUDE_log_equation_solution_l310_31085

theorem log_equation_solution (k c p : ℝ) (h : k > 0) (hp : p > 0) :
  Real.log k^2 / Real.log 10 = c - 2 * Real.log p / Real.log 10 →
  k = 10^c / p := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l310_31085


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l310_31078

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l310_31078


namespace NUMINAMATH_CALUDE_inequality_proof_l310_31024

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l310_31024


namespace NUMINAMATH_CALUDE_point_on_circle_l310_31040

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle (P Q : ℝ × ℝ) :
  P = (1, 0) →
  unit_circle P.1 P.2 →
  unit_circle Q.1 Q.2 →
  arc_length (4 * π / 3) = abs (Real.arccos P.1 - Real.arccos Q.1) →
  Q = (-1/2, Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l310_31040
