import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1411_141189

/-- A rectangular prism with given face areas has a volume of 24 cubic centimeters. -/
theorem rectangular_prism_volume (w h d : ℝ) 
  (front_area : w * h = 12)
  (side_area : d * h = 6)
  (top_area : d * w = 8) :
  w * h * d = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1411_141189


namespace NUMINAMATH_CALUDE_power_of_729_two_thirds_l1411_141158

theorem power_of_729_two_thirds : (729 : ℝ) ^ (2/3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_729_two_thirds_l1411_141158


namespace NUMINAMATH_CALUDE_divisor_problem_l1411_141115

theorem divisor_problem (x d : ℕ) (h1 : x % d = 5) (h2 : (x + 13) % 41 = 18) : d = 41 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1411_141115


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1411_141100

theorem geometric_sequence_sum (a₁ a₂ a₃ a₆ a₇ a₈ : ℚ) :
  a₁ = 4096 →
  a₂ = 1024 →
  a₃ = 256 →
  a₆ = 4 →
  a₇ = 1 →
  a₈ = 1/4 →
  ∃ r : ℚ, r ≠ 0 ∧
    (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = a₁ * (a₂ / a₁)^(n-1)) →
    a₄ + a₅ = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1411_141100


namespace NUMINAMATH_CALUDE_sum_37_29_base5_l1411_141165

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_37_29_base5 :
  addBase5 (toBase5 37) (toBase5 29) = [2, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_sum_37_29_base5_l1411_141165


namespace NUMINAMATH_CALUDE_third_bounce_height_l1411_141176

/-- Given an initial height and a bounce ratio, calculates the height of the nth bounce -/
def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

/-- Converts meters to centimeters -/
def meters_to_cm (meters : ℝ) : ℝ :=
  meters * 100

theorem third_bounce_height :
  let initial_height : ℝ := 12.8
  let bounce_ratio : ℝ := 1/4
  let third_bounce_m := bounce_height initial_height bounce_ratio 3
  meters_to_cm third_bounce_m = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_bounce_height_l1411_141176


namespace NUMINAMATH_CALUDE_zhuoma_combinations_l1411_141113

/-- The number of different styles of backpacks -/
def num_backpack_styles : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_case_styles : ℕ := 2

/-- The number of different combinations of backpack and pencil case styles -/
def num_combinations : ℕ := num_backpack_styles * num_pencil_case_styles

theorem zhuoma_combinations :
  num_combinations = 4 :=
by sorry

end NUMINAMATH_CALUDE_zhuoma_combinations_l1411_141113


namespace NUMINAMATH_CALUDE_april_flower_sale_l1411_141156

/-- April's flower sale problem -/
theorem april_flower_sale (initial_roses : ℕ) (remaining_roses : ℕ) (price_per_rose : ℕ) :
  initial_roses = 13 →
  remaining_roses = 4 →
  price_per_rose = 4 →
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
sorry

end NUMINAMATH_CALUDE_april_flower_sale_l1411_141156


namespace NUMINAMATH_CALUDE_maximum_mark_calculation_maximum_mark_is_500_l1411_141125

theorem maximum_mark_calculation (passing_threshold : ℝ) (student_score : ℕ) (failure_margin : ℕ) : ℝ :=
  let passing_mark : ℕ := student_score + failure_margin
  let maximum_mark : ℝ := passing_mark / passing_threshold
  maximum_mark

theorem maximum_mark_is_500 :
  maximum_mark_calculation 0.33 125 40 = 500 := by
  sorry

end NUMINAMATH_CALUDE_maximum_mark_calculation_maximum_mark_is_500_l1411_141125


namespace NUMINAMATH_CALUDE_ceiling_sqrt_900_l1411_141142

theorem ceiling_sqrt_900 : ⌈Real.sqrt 900⌉ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_900_l1411_141142


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1411_141124

theorem arithmetic_calculations :
  ((-5 + 8 - 2 : ℚ) = 1) ∧
  ((-3 * (5/6) / (-1/4) : ℚ) = 10) ∧
  ((-3/17 + (-3.75) + (-14/17) + 3 * (3/4) : ℚ) = -1) ∧
  ((-1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 : ℚ) = -5/6) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1411_141124


namespace NUMINAMATH_CALUDE_min_value_expression_l1411_141192

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) ≥ 12 ∧
  (Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) = 12 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1411_141192


namespace NUMINAMATH_CALUDE_periodic_function_l1411_141154

/-- A function f is periodic with period 2c if it satisfies the given functional equation. -/
theorem periodic_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f (x + c) = 2 / (1 + f x) - 1) →
  (∀ x, f (x + 2*c) = f x) :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_l1411_141154


namespace NUMINAMATH_CALUDE_mrs_hilt_total_miles_l1411_141172

/-- Mrs. Hilt's fitness schedule for a week --/
structure FitnessSchedule where
  monday_run : ℕ
  monday_swim : ℕ
  wednesday_run : ℕ
  wednesday_bike : ℕ
  friday_run : ℕ
  friday_swim : ℕ
  friday_bike : ℕ
  sunday_bike : ℕ

/-- Calculate the total miles for a given fitness schedule --/
def total_miles (schedule : FitnessSchedule) : ℕ :=
  schedule.monday_run + schedule.monday_swim +
  schedule.wednesday_run + schedule.wednesday_bike +
  schedule.friday_run + schedule.friday_swim + schedule.friday_bike +
  schedule.sunday_bike

/-- Mrs. Hilt's actual fitness schedule --/
def mrs_hilt_schedule : FitnessSchedule := {
  monday_run := 3
  monday_swim := 1
  wednesday_run := 2
  wednesday_bike := 6
  friday_run := 7
  friday_swim := 2
  friday_bike := 3
  sunday_bike := 10
}

/-- Theorem: Mrs. Hilt's total miles for the week is 34 --/
theorem mrs_hilt_total_miles :
  total_miles mrs_hilt_schedule = 34 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_miles_l1411_141172


namespace NUMINAMATH_CALUDE_airline_seats_per_row_l1411_141135

/-- Proves that the number of seats in each row is 7 for an airline company with given conditions. -/
theorem airline_seats_per_row :
  let num_airplanes : ℕ := 5
  let rows_per_airplane : ℕ := 20
  let flights_per_airplane_per_day : ℕ := 2
  let total_passengers_per_day : ℕ := 1400
  let seats_per_row : ℕ := total_passengers_per_day / (num_airplanes * flights_per_airplane_per_day * rows_per_airplane)
  seats_per_row = 7 := by sorry

end NUMINAMATH_CALUDE_airline_seats_per_row_l1411_141135


namespace NUMINAMATH_CALUDE_max_removable_correct_l1411_141149

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  yellow : Nat
  red : Nat
  black : Nat

/-- Checks if the remaining marbles satisfy the condition -/
def satisfiesCondition (bag : MarbleBag) : Prop :=
  (bag.yellow ≥ 4 ∧ (bag.red ≥ 3 ∨ bag.black ≥ 3)) ∨
  (bag.red ≥ 4 ∧ (bag.yellow ≥ 3 ∨ bag.black ≥ 3)) ∨
  (bag.black ≥ 4 ∧ (bag.yellow ≥ 3 ∨ bag.red ≥ 3))

/-- The initial bag of marbles -/
def initialBag : MarbleBag := ⟨8, 7, 5⟩

/-- The maximum number of marbles that can be removed -/
def maxRemovable : Nat := 7

theorem max_removable_correct :
  (∀ (removed : MarbleBag), 
    removed.yellow + removed.red + removed.black ≤ maxRemovable →
    satisfiesCondition ⟨initialBag.yellow - removed.yellow, 
                        initialBag.red - removed.red, 
                        initialBag.black - removed.black⟩) ∧
  (∃ (removed : MarbleBag), 
    removed.yellow + removed.red + removed.black = maxRemovable + 1 ∧
    ¬satisfiesCondition ⟨initialBag.yellow - removed.yellow, 
                         initialBag.red - removed.red, 
                         initialBag.black - removed.black⟩) :=
by sorry

end NUMINAMATH_CALUDE_max_removable_correct_l1411_141149


namespace NUMINAMATH_CALUDE_calculate_expression_l1411_141155

theorem calculate_expression : 500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1411_141155


namespace NUMINAMATH_CALUDE_more_white_boxes_than_red_l1411_141133

theorem more_white_boxes_than_red (balls_per_box : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h1 : balls_per_box = 6)
  (h2 : white_balls = 30)
  (h3 : red_balls = 18) :
  white_balls / balls_per_box - red_balls / balls_per_box = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_more_white_boxes_than_red_l1411_141133


namespace NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l1411_141195

/-- Rounds a natural number to the nearest thousand -/
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  ((n + 500) / 1000) * 1000

/-- Converts a natural number to ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem rounding_317500_equals_31_8_ten_thousand :
  to_ten_thousands (round_to_nearest_thousand 317500) = 31.8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l1411_141195


namespace NUMINAMATH_CALUDE_min_side_triangle_l1411_141183

theorem min_side_triangle (a b c : ℝ) (A B C : ℝ) : 
  a + b = 2 → C = 2 * π / 3 → c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_side_triangle_l1411_141183


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l1411_141129

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  total_visits : Nat
  unique_shoppers : Nat
  two_store_visitors : Nat
  stores_in_town : Nat

/-- Calculates the maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : Nat :=
  let remaining_visits := scenario.total_visits - 2 * scenario.two_store_visitors
  let remaining_shoppers := scenario.unique_shoppers - scenario.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited_is_four (scenario : ShoppingScenario) : 
  scenario.total_visits = 23 →
  scenario.unique_shoppers = 12 →
  scenario.two_store_visitors = 8 →
  scenario.stores_in_town = 8 →
  max_stores_visited scenario = 4 := by
  sorry

#eval max_stores_visited ⟨23, 12, 8, 8⟩

end NUMINAMATH_CALUDE_max_stores_visited_is_four_l1411_141129


namespace NUMINAMATH_CALUDE_unit_distance_preservation_implies_all_distance_preservation_l1411_141198

/-- A function that maps points on a plane to other points on the same plane -/
def PlaneMap (Plane : Type*) := Plane → Plane

/-- Distance function between two points on a plane -/
def distance (Plane : Type*) := Plane → Plane → ℝ

/-- A function preserves unit distances if the distance between the images of any two points
    that are one unit apart is also one unit -/
def preserves_unit_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d P Q = 1 → d (f P) (f Q) = 1

/-- A function preserves all distances if the distance between the images of any two points
    is equal to the distance between the original points -/
def preserves_all_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d (f P) (f Q) = d P Q

/-- Main theorem: if a plane map preserves unit distances, it preserves all distances -/
theorem unit_distance_preservation_implies_all_distance_preservation
  (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :
  preserves_unit_distances Plane f d → preserves_all_distances Plane f d :=
by
  sorry

end NUMINAMATH_CALUDE_unit_distance_preservation_implies_all_distance_preservation_l1411_141198


namespace NUMINAMATH_CALUDE_min_value_theorem_l1411_141137

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1411_141137


namespace NUMINAMATH_CALUDE_fill_time_with_both_pumps_l1411_141148

-- Define the fill rates for the old and new pumps
def old_pump_rate : ℚ := 1 / 600
def new_pump_rate : ℚ := 1 / 200

-- Define the combined fill rate
def combined_rate : ℚ := old_pump_rate + new_pump_rate

-- Theorem to prove
theorem fill_time_with_both_pumps :
  (1 : ℚ) / combined_rate = 150 := by sorry

end NUMINAMATH_CALUDE_fill_time_with_both_pumps_l1411_141148


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1411_141153

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1411_141153


namespace NUMINAMATH_CALUDE_debate_team_count_l1411_141157

/-- The number of girls in the debate club -/
def num_girls : ℕ := 4

/-- The number of boys in the debate club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for each team -/
def girls_per_team : ℕ := 3

/-- The number of boys to be chosen for each team -/
def boys_per_team : ℕ := 3

/-- Theorem stating the total number of possible debate teams -/
theorem debate_team_count : 
  (Nat.choose num_girls girls_per_team) * (Nat.choose num_boys boys_per_team) = 80 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_count_l1411_141157


namespace NUMINAMATH_CALUDE_talia_father_age_l1411_141101

-- Define Talia's current age
def talia_age : ℕ := 20 - 7

-- Define Talia's mom's current age
def mom_age : ℕ := 3 * talia_age

-- Define Talia's father's current age
def father_age : ℕ := mom_age - 3

-- Theorem statement
theorem talia_father_age : father_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_talia_father_age_l1411_141101


namespace NUMINAMATH_CALUDE_square_sum_diff_l1411_141179

theorem square_sum_diff (a b : ℝ) 
  (h1 : (a + b)^2 = 8) 
  (h2 : (a - b)^2 = 12) : 
  a^2 + b^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_diff_l1411_141179


namespace NUMINAMATH_CALUDE_fraction_equality_l1411_141160

theorem fraction_equality (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1411_141160


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1411_141171

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) : 
  square_side = 40 →
  triangle_base = 30 →
  triangle_height = 30 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 700 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1411_141171


namespace NUMINAMATH_CALUDE_retirement_savings_l1411_141193

/-- Calculates the final amount using simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the final amount after 15 years is 1,640,000 rubles -/
theorem retirement_savings : 
  let principal : ℝ := 800000
  let rate : ℝ := 0.07
  let time : ℝ := 15
  simpleInterest principal rate time = 1640000 := by
  sorry

end NUMINAMATH_CALUDE_retirement_savings_l1411_141193


namespace NUMINAMATH_CALUDE_alphazian_lost_words_l1411_141130

/-- The number of letters in the Alphazian alphabet -/
def alphabet_size : ℕ := 128

/-- The number of forbidden letters -/
def forbidden_letters : ℕ := 2

/-- The maximum word length in Alphazia -/
def max_word_length : ℕ := 2

/-- Calculates the number of lost words due to letter prohibition in Alphazia -/
def lost_words : ℕ :=
  forbidden_letters + (alphabet_size * forbidden_letters)

theorem alphazian_lost_words :
  lost_words = 258 := by sorry

end NUMINAMATH_CALUDE_alphazian_lost_words_l1411_141130


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l1411_141162

/-- Represents a trapezoid with side lengths a, b, c, and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the possible areas of a trapezoid. -/
def possibleAreas (t : Trapezoid) : Set ℝ :=
  sorry

/-- Checks if a number is not divisible by the square of any prime. -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  sorry

/-- The main theorem about the trapezoid areas. -/
theorem trapezoid_area_sum (t : Trapezoid) 
    (h1 : t.a = 4 ∧ t.b = 6 ∧ t.c = 8 ∧ t.d = 10) :
    ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
      (∀ A ∈ possibleAreas t, ∃ k, A = k * (r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃)) ∧
      notDivisibleBySquareOfPrime n₁ ∧
      notDivisibleBySquareOfPrime n₂ ∧
      ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l1411_141162


namespace NUMINAMATH_CALUDE_fraction_simplification_l1411_141128

theorem fraction_simplification : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1411_141128


namespace NUMINAMATH_CALUDE_triangle_properties_l1411_141126

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B = t.b * Real.cos t.A)
  (h2 : t.b = 6)
  (h3 : t.c = 2 * t.a) :
  t.B = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1411_141126


namespace NUMINAMATH_CALUDE_keith_bought_four_digimon_packs_l1411_141103

/-- The number of Digimon card packs Keith bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of each Digimon card pack in dollars -/
def digimon_pack_cost : ℝ := 4.45

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent in dollars -/
def total_spent : ℝ := 23.86

/-- Theorem stating that Keith bought 4 packs of Digimon cards -/
theorem keith_bought_four_digimon_packs :
  (num_digimon_packs : ℝ) * digimon_pack_cost + baseball_deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_keith_bought_four_digimon_packs_l1411_141103


namespace NUMINAMATH_CALUDE_special_function_inequality_l1411_141151

/-- A function f: ℝ → ℝ satisfying f(x) + f''(x) > 1 for all x -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (deriv^[2] f) x > 1

/-- Theorem stating the relationship between f(2) - 1 and e^(f(3) - 1) -/
theorem special_function_inequality (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 2 - 1 < Real.exp (f 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l1411_141151


namespace NUMINAMATH_CALUDE_f_max_value_l1411_141161

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

/-- The maximum value of f(x) is 22 -/
theorem f_max_value : ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1411_141161


namespace NUMINAMATH_CALUDE_next_but_one_perfect_square_l1411_141177

theorem next_but_one_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m > x ∧ m < n ∧ ∃ k : ℕ, m = k^2) ∧ n = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_next_but_one_perfect_square_l1411_141177


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_power_l1411_141150

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites
    and their y-coordinates are equal. -/
def symmetricYAxis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetry_implies_sum_power (a b : ℝ) :
  symmetricYAxis (a, -2) (-1, b) → (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_power_l1411_141150


namespace NUMINAMATH_CALUDE_tan_theta_value_l1411_141184

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ)) : 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1411_141184


namespace NUMINAMATH_CALUDE_abs_greater_necessary_not_sufficient_l1411_141170

theorem abs_greater_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > b → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_abs_greater_necessary_not_sufficient_l1411_141170


namespace NUMINAMATH_CALUDE_real_equal_roots_iff_k_values_l1411_141118

/-- The quadratic equation in question -/
def equation (k x : ℝ) : ℝ := 3 * x^2 - 2 * k * x + 3 * x + 12

/-- Condition for real and equal roots -/
def has_real_equal_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, equation k x = 0 ∧ 
  ∀ y : ℝ, equation k y = 0 → y = x

/-- Theorem stating the values of k for which the equation has real and equal roots -/
theorem real_equal_roots_iff_k_values :
  ∀ k : ℝ, has_real_equal_roots k ↔ (k = -9/2 ∨ k = 15/2) :=
sorry

end NUMINAMATH_CALUDE_real_equal_roots_iff_k_values_l1411_141118


namespace NUMINAMATH_CALUDE_chess_piece_probability_l1411_141186

/-- The probability of drawing a red piece first and a green piece second from a bag of chess pieces -/
theorem chess_piece_probability (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 32) 
  (h2 : red = 16) 
  (h3 : green = 16) 
  (h4 : red + green = total) : 
  (red / total) * (green / (total - 1)) = 8 / 31 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_probability_l1411_141186


namespace NUMINAMATH_CALUDE_cleaning_payment_l1411_141104

theorem cleaning_payment (rate : ℚ) (rooms : ℚ) : 
  rate = 12 / 3 → rooms = 9 / 4 → rate * rooms = 9 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l1411_141104


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1411_141190

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1411_141190


namespace NUMINAMATH_CALUDE_right_triangle_ratio_square_l1411_141108

theorem right_triangle_ratio_square (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : c^2 = a^2 + b^2) (h5 : a / b = b / c) : (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_square_l1411_141108


namespace NUMINAMATH_CALUDE_test_questions_l1411_141167

theorem test_questions (score : ℕ) (correct : ℕ) (incorrect : ℕ) :
  score = correct - 2 * incorrect →
  score = 73 →
  correct = 91 →
  correct + incorrect = 100 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_l1411_141167


namespace NUMINAMATH_CALUDE_ice_cream_sales_l1411_141199

theorem ice_cream_sales (tuesday_sales : ℕ) (wednesday_sales : ℕ) : 
  wednesday_sales = 2 * tuesday_sales →
  tuesday_sales + wednesday_sales = 36000 →
  tuesday_sales = 12000 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l1411_141199


namespace NUMINAMATH_CALUDE_equation_solution_l1411_141197

theorem equation_solution : ∃ x : ℝ, 0.6 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1411_141197


namespace NUMINAMATH_CALUDE_pink_tie_probability_l1411_141174

-- Define the number of ties of each color
def black_ties : ℕ := 5
def gold_ties : ℕ := 7
def pink_ties : ℕ := 8

-- Define the total number of ties
def total_ties : ℕ := black_ties + gold_ties + pink_ties

-- Define the probability of choosing a pink tie
def prob_pink_tie : ℚ := pink_ties / total_ties

-- Theorem statement
theorem pink_tie_probability :
  prob_pink_tie = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_pink_tie_probability_l1411_141174


namespace NUMINAMATH_CALUDE_working_mom_time_allocation_l1411_141111

theorem working_mom_time_allocation :
  let total_hours_in_day : ℝ := 24
  let work_hours : ℝ := 8
  let daughter_care_hours : ℝ := 2.25
  let household_chores_hours : ℝ := 3.25
  let total_activity_hours : ℝ := work_hours + daughter_care_hours + household_chores_hours
  let percentage_of_day : ℝ := (total_activity_hours / total_hours_in_day) * 100
  percentage_of_day = 56.25 := by
sorry

end NUMINAMATH_CALUDE_working_mom_time_allocation_l1411_141111


namespace NUMINAMATH_CALUDE_exists_counterexample_l1411_141122

/-- A binary operation on a set S satisfying a * (b * a) = b for all a, b in S -/
class SpecialOperation (S : Type) where
  op : S → S → S
  property : ∀ (a b : S), op a (op b a) = b

/-- Theorem stating that there exist elements a and b in S such that (a*b)*a ≠ a -/
theorem exists_counterexample {S : Type} [SpecialOperation S] [Inhabited S] [Nontrivial S] :
  ∃ (a b : S), (SpecialOperation.op (SpecialOperation.op a b) a) ≠ a := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_l1411_141122


namespace NUMINAMATH_CALUDE_sandys_sum_attempt_l1411_141164

/-- Sandy's sum attempt problem -/
theorem sandys_sum_attempt :
  ∀ (correct_marks incorrect_marks total_marks correct_sums : ℕ),
    correct_marks = 3 →
    incorrect_marks = 2 →
    total_marks = 45 →
    correct_sums = 21 →
    ∃ (total_sums : ℕ),
      total_sums = correct_sums + (total_marks - correct_marks * correct_sums) / incorrect_marks ∧
      total_sums = 30 :=
by sorry

end NUMINAMATH_CALUDE_sandys_sum_attempt_l1411_141164


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1411_141120

/-- A geometric sequence with a_m = 3 and a_{m+6} = 24 -/
def GeometricSequence (a : ℕ → ℝ) (m : ℕ) : Prop :=
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧ 
  a m = 3 ∧ 
  a (m + 6) = 24

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h : GeometricSequence a m) : 
  a (m + 18) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1411_141120


namespace NUMINAMATH_CALUDE_sara_bird_count_l1411_141168

/-- The number of dozens of birds Sara saw -/
def dozens_of_birds : ℕ := 8

/-- The number of birds in one dozen -/
def birds_per_dozen : ℕ := 12

/-- The total number of birds Sara saw -/
def total_birds : ℕ := dozens_of_birds * birds_per_dozen

theorem sara_bird_count : total_birds = 96 := by
  sorry

end NUMINAMATH_CALUDE_sara_bird_count_l1411_141168


namespace NUMINAMATH_CALUDE_problem_solution_l1411_141139

theorem problem_solution (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_eq : x - 10 = M)
  (y_eq : y + 10 = M)
  (z_eq : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1411_141139


namespace NUMINAMATH_CALUDE_even_function_f_2_l1411_141114

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x + 3)

-- State the theorem
theorem even_function_f_2 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : f a 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_even_function_f_2_l1411_141114


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1411_141112

/-- Given that 2 is one root of the equation 5x^2 + kx = 4, prove that -2/5 is the other root -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = 2) → 
  (∃ x : ℝ, 5 * x^2 + k * x = 4 ∧ x = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1411_141112


namespace NUMINAMATH_CALUDE_pressure_functions_exist_l1411_141140

/-- Represents the gas pressure in a vessel as a function of time. -/
def PressureFunction := ℝ → ℝ

/-- Represents the parameters of the gas system. -/
structure GasSystem where
  V₁ : ℝ  -- Volume of vessel 1
  V₂ : ℝ  -- Volume of vessel 2
  P₁ : ℝ  -- Initial pressure in vessel 1
  P₂ : ℝ  -- Initial pressure in vessel 2
  a  : ℝ  -- Flow rate coefficient
  b  : ℝ  -- Pressure change coefficient

/-- Defines the conditions for valid pressure functions in the gas system. -/
def ValidPressureFunctions (sys : GasSystem) (p₁ p₂ : PressureFunction) : Prop :=
  -- Initial conditions
  p₁ 0 = sys.P₁ ∧ p₂ 0 = sys.P₂ ∧
  -- Conservation of mass
  ∀ t, sys.V₁ * p₁ t + sys.V₂ * p₂ t = sys.V₁ * sys.P₁ + sys.V₂ * sys.P₂ ∧
  -- Differential equations
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = -sys.b * sys.V₁ * (deriv p₁ t) ∧
  ∀ t, sys.a * (p₁ t ^ 2 - p₂ t ^ 2) = sys.b * sys.V₂ * (deriv p₂ t)

/-- Theorem stating the existence of valid pressure functions for a given gas system. -/
theorem pressure_functions_exist (sys : GasSystem) :
  ∃ (p₁ p₂ : PressureFunction), ValidPressureFunctions sys p₁ p₂ := by
  sorry


end NUMINAMATH_CALUDE_pressure_functions_exist_l1411_141140


namespace NUMINAMATH_CALUDE_greatest_integer_side_length_l1411_141105

theorem greatest_integer_side_length (area : ℝ) (h : area < 150) :
  ∃ (s : ℕ), s * s ≤ area ∧ ∀ (t : ℕ), t * t ≤ area → t ≤ s ∧ s = 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_side_length_l1411_141105


namespace NUMINAMATH_CALUDE_A_ends_with_14_zeros_l1411_141173

theorem A_ends_with_14_zeros :
  let A := 2^7 * (7^14 + 1) + 2^6 * 7^11 * 10^2 + 2^6 * 7^7 * 10^4 + 2^4 * 7^3 * 10^6
  A = 10^14 := by sorry

end NUMINAMATH_CALUDE_A_ends_with_14_zeros_l1411_141173


namespace NUMINAMATH_CALUDE_coloring_book_problem_l1411_141106

theorem coloring_book_problem (book1 : Nat) (book2 : Nat) (colored : Nat) : 
  book1 = 23 → book2 = 32 → colored = 44 → book1 + book2 - colored = 11 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l1411_141106


namespace NUMINAMATH_CALUDE_system_solution_l1411_141144

theorem system_solution (u v w : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0)
  (eq1 : 3 / (u * v) + 15 / (v * w) = 2)
  (eq2 : 15 / (v * w) + 5 / (w * u) = 2)
  (eq3 : 5 / (w * u) + 3 / (u * v) = 2) :
  (u = 1 ∧ v = 3 ∧ w = 5) ∨ (u = -1 ∧ v = -3 ∧ w = -5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1411_141144


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_6_and_9_l1411_141182

/-- A function that checks if a natural number consists only of digits 6 and 9 -/
def only_six_and_nine (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 9

/-- The largest number consisting of only 6 and 9 digits that is divisible by 18 -/
def m : ℕ := 969696

theorem largest_multiple_of_18_with_6_and_9 :
  (∀ k : ℕ, k > m → ¬(only_six_and_nine k ∧ 18 ∣ k)) ∧
  only_six_and_nine m ∧
  18 ∣ m ∧
  m / 18 = 53872 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_6_and_9_l1411_141182


namespace NUMINAMATH_CALUDE_ski_price_calculation_l1411_141191

theorem ski_price_calculation (initial_price : ℝ) 
  (morning_discount : ℝ) (noon_increase : ℝ) (afternoon_discount : ℝ) : 
  initial_price = 200 →
  morning_discount = 0.4 →
  noon_increase = 0.25 →
  afternoon_discount = 0.2 →
  (initial_price * (1 - morning_discount) * (1 + noon_increase) * (1 - afternoon_discount)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ski_price_calculation_l1411_141191


namespace NUMINAMATH_CALUDE_optimal_arrangement_l1411_141159

/-- Represents the arrangement of workers in a factory --/
structure WorkerArrangement where
  total_workers : ℕ
  type_a_workers : ℕ
  type_b_workers : ℕ
  type_a_production : ℕ
  type_b_production : ℕ
  set_a_units : ℕ
  set_b_units : ℕ

/-- Checks if the arrangement produces exact sets --/
def produces_exact_sets (arrangement : WorkerArrangement) : Prop :=
  arrangement.total_workers = arrangement.type_a_workers + arrangement.type_b_workers ∧
  arrangement.type_a_workers * arrangement.type_a_production / arrangement.set_a_units =
  arrangement.type_b_workers * arrangement.type_b_production / arrangement.set_b_units

/-- Theorem stating that the given arrangement produces exact sets --/
theorem optimal_arrangement :
  produces_exact_sets {
    total_workers := 104,
    type_a_workers := 72,
    type_b_workers := 32,
    type_a_production := 8,
    type_b_production := 12,
    set_a_units := 3,
    set_b_units := 2
  } := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l1411_141159


namespace NUMINAMATH_CALUDE_circle_radius_l1411_141188

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = r^2) →
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2) →
  ∃ r : ℝ, r = 3 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1411_141188


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_necessary_not_sufficient_condition_l1411_141119

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 1)*(x - m - 1) > 0}

theorem intersection_when_m_zero :
  A ∩ B 0 = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m < -2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_necessary_not_sufficient_condition_l1411_141119


namespace NUMINAMATH_CALUDE_charity_donation_l1411_141102

/-- The number of pennies collected by Cassandra -/
def cassandra_pennies : ℕ := 5000

/-- The difference in pennies collected between Cassandra and James -/
def difference : ℕ := 276

/-- The number of pennies collected by James -/
def james_pennies : ℕ := cassandra_pennies - difference

/-- The total number of pennies donated to charity -/
def total_donated : ℕ := cassandra_pennies + james_pennies

theorem charity_donation :
  total_donated = 9724 :=
sorry

end NUMINAMATH_CALUDE_charity_donation_l1411_141102


namespace NUMINAMATH_CALUDE_gcd_lcm_triples_count_l1411_141181

theorem gcd_lcm_triples_count : 
  (Finset.filter 
    (fun (triple : ℕ × ℕ × ℕ) => 
      Nat.gcd (Nat.gcd triple.1 triple.2.1) triple.2.2 = 15 ∧ 
      Nat.lcm (Nat.lcm triple.1 triple.2.1) triple.2.2 = 3^15 * 5^18)
    (Finset.product (Finset.range (3^15 * 5^18 + 1)) 
      (Finset.product (Finset.range (3^15 * 5^18 + 1)) (Finset.range (3^15 * 5^18 + 1))))).card = 8568 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_triples_count_l1411_141181


namespace NUMINAMATH_CALUDE_volleyball_team_lineup_count_l1411_141138

def volleyball_team_size : ℕ := 14
def starting_lineup_size : ℕ := 6
def triplet_size : ℕ := 3

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem volleyball_team_lineup_count :
  choose volleyball_team_size starting_lineup_size -
  choose (volleyball_team_size - triplet_size) (starting_lineup_size - triplet_size) = 2838 :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_lineup_count_l1411_141138


namespace NUMINAMATH_CALUDE_crayon_boxes_l1411_141121

theorem crayon_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (boxes_needed : ℕ) : 
  total_crayons = 80 → 
  crayons_per_box = 8 → 
  boxes_needed = total_crayons / crayons_per_box →
  boxes_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_crayon_boxes_l1411_141121


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1411_141110

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1411_141110


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l1411_141136

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l1411_141136


namespace NUMINAMATH_CALUDE_marriage_age_proof_l1411_141178

/-- The average age of a husband and wife at the time of their marriage -/
def average_age_at_marriage : ℝ := 23

/-- The number of years passed since the marriage -/
def years_passed : ℕ := 5

/-- The age of the child -/
def child_age : ℕ := 1

/-- The current average age of the family -/
def current_family_average_age : ℝ := 19

/-- The number of people in the family -/
def family_size : ℕ := 3

theorem marriage_age_proof :
  average_age_at_marriage = 23 :=
by
  sorry

#check marriage_age_proof

end NUMINAMATH_CALUDE_marriage_age_proof_l1411_141178


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1411_141169

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 2*α - 2005 = 0) → 
  (β^2 + 2*β - 2005 = 0) → 
  (α^2 + 3*α + β = 2003) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1411_141169


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l1411_141117

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The GDP value in ten thousand yuan -/
def gdp : ℝ := 84300000

/-- The scientific notation representation of the GDP -/
def gdp_scientific : ScientificNotation := {
  coefficient := 8.43,
  exponent := 7,
  h1 := by sorry
}

/-- Theorem stating that the GDP in scientific notation is correct -/
theorem gdp_scientific_notation_correct : 
  gdp = gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent := by sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_correct_l1411_141117


namespace NUMINAMATH_CALUDE_investment_growth_l1411_141107

/-- The initial investment amount that grows to $563.35 after 5 years at 12% annual interest rate compounded yearly. -/
def initial_investment : ℝ := 319.77

/-- The final amount after 5 years of investment. -/
def final_amount : ℝ := 563.35

/-- The annual interest rate as a decimal. -/
def interest_rate : ℝ := 0.12

/-- The number of years the money is invested. -/
def years : ℕ := 5

/-- Theorem stating that the initial investment grows to the final amount after the specified time and interest rate. -/
theorem investment_growth :
  final_amount = initial_investment * (1 + interest_rate) ^ years := by
  sorry

#eval initial_investment

end NUMINAMATH_CALUDE_investment_growth_l1411_141107


namespace NUMINAMATH_CALUDE_prize_money_problem_l1411_141166

/-- The prize money problem -/
theorem prize_money_problem (total_students : Nat) (team_members : Nat) (member_prize : Nat) (extra_prize : Nat) :
  total_students = 10 →
  team_members = 9 →
  member_prize = 200 →
  extra_prize = 90 →
  ∃ (captain_prize : Nat),
    captain_prize = extra_prize + (captain_prize + team_members * member_prize) / total_students ∧
    captain_prize = 300 := by
  sorry

end NUMINAMATH_CALUDE_prize_money_problem_l1411_141166


namespace NUMINAMATH_CALUDE_bacteria_population_after_nine_days_l1411_141132

/-- Represents the population of bacteria after a given number of 3-day periods -/
def bacteriaPopulation (initialCount : ℕ) (periods : ℕ) : ℕ :=
  initialCount * (3 ^ periods)

/-- Theorem stating that the bacteria population after 9 days (3 periods) is 36 -/
theorem bacteria_population_after_nine_days :
  bacteriaPopulation 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_population_after_nine_days_l1411_141132


namespace NUMINAMATH_CALUDE_cookie_is_circle_with_radius_nine_l1411_141146

/-- The cookie's boundary equation -/
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 28 = 6*x + 20*y

/-- The circle equation with center (3, 10) and radius 9 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 10)^2 = 81

/-- Theorem stating that the cookie boundary is equivalent to a circle with radius 9 -/
theorem cookie_is_circle_with_radius_nine :
  ∀ x y : ℝ, cookie_boundary x y ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_cookie_is_circle_with_radius_nine_l1411_141146


namespace NUMINAMATH_CALUDE_expression_value_l1411_141143

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1411_141143


namespace NUMINAMATH_CALUDE_construct_3x3x3_cube_l1411_141147

/-- Represents a 3D piece with given dimensions -/
structure Piece where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the collection of pieces available for construction -/
structure PieceSet where
  large_pieces : List Piece
  small_pieces : List Piece

/-- Represents a 3D cube -/
structure Cube where
  side_length : ℕ

/-- Checks if a set of pieces can construct the given cube -/
def can_construct_cube (pieces : PieceSet) (cube : Cube) : Prop :=
  -- The actual implementation would involve complex logic to check if the pieces can form the cube
  sorry

/-- The main theorem stating that the given set of pieces can construct a 3x3x3 cube -/
theorem construct_3x3x3_cube : 
  let pieces : PieceSet := {
    large_pieces := List.replicate 6 { length := 1, width := 2, height := 2 },
    small_pieces := List.replicate 3 { length := 1, width := 1, height := 1 }
  }
  let target_cube : Cube := { side_length := 3 }
  can_construct_cube pieces target_cube := by
  sorry


end NUMINAMATH_CALUDE_construct_3x3x3_cube_l1411_141147


namespace NUMINAMATH_CALUDE_total_gum_pieces_l1411_141163

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l1411_141163


namespace NUMINAMATH_CALUDE_fraction_inequality_l1411_141123

theorem fraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1411_141123


namespace NUMINAMATH_CALUDE_condition_for_squared_inequality_l1411_141175

theorem condition_for_squared_inequality (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_squared_inequality_l1411_141175


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1411_141196

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x) ∧
  (∀ x, f (-x + 5) = f (x - 3)) ∧
  (∃! x, f x = x)

/-- The domain and range conditions for the quadratic function -/
def DomainRangeCondition (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x, f x ∈ Set.Icc (3*m) (3*n) ↔ x ∈ Set.Icc m n)

theorem quadratic_function_theorem :
  ∀ a b : ℝ, ∀ f : ℝ → ℝ,
  QuadraticFunction a b f →
  ∃ m n : ℝ,
    (∀ x, f x = -1/2 * x^2 + x) ∧
    m = -4 ∧ n = 0 ∧
    DomainRangeCondition f m n :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1411_141196


namespace NUMINAMATH_CALUDE_company_ratio_is_9_47_l1411_141131

/-- Represents the ratio of managers to non-managers in a company -/
structure ManagerRatio where
  managers : ℕ
  non_managers : ℕ

/-- The company's policy for manager to non-manager ratio -/
axiom company_ratio : ManagerRatio

/-- The ratio is constant across all departments -/
axiom ratio_constant (dept1 dept2 : ManagerRatio) : 
  dept1.managers * dept2.non_managers = dept1.non_managers * dept2.managers

/-- In a department with 9 managers, the maximum number of non-managers is 47 -/
axiom max_non_managers : ∃ (dept : ManagerRatio), dept.managers = 9 ∧ dept.non_managers = 47

/-- The company ratio is equal to 9:47 -/
theorem company_ratio_is_9_47 : company_ratio.managers = 9 ∧ company_ratio.non_managers = 47 := by
  sorry

end NUMINAMATH_CALUDE_company_ratio_is_9_47_l1411_141131


namespace NUMINAMATH_CALUDE_sixDigitPermutations_eq_60_l1411_141194

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 9 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 9 is equal to 60 -/
theorem sixDigitPermutations_eq_60 : sixDigitPermutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutations_eq_60_l1411_141194


namespace NUMINAMATH_CALUDE_oneBlack_twoWhite_mutually_exclusive_not_contradictory_l1411_141180

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def oneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def twoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- The theorem stating that oneBlack and twoWhite are mutually exclusive but not contradictory -/
theorem oneBlack_twoWhite_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(oneBlack outcome ∧ twoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, ¬oneBlack outcome ∧ ¬twoWhite outcome) :=
sorry

end NUMINAMATH_CALUDE_oneBlack_twoWhite_mutually_exclusive_not_contradictory_l1411_141180


namespace NUMINAMATH_CALUDE_length_width_difference_l1411_141152

/-- The length of the basketball court in meters -/
def court_length : ℝ := 31

/-- The width of the basketball court in meters -/
def court_width : ℝ := 17

/-- The perimeter of the basketball court in meters -/
def court_perimeter : ℝ := 96

theorem length_width_difference : court_length - court_width = 14 := by
  sorry

end NUMINAMATH_CALUDE_length_width_difference_l1411_141152


namespace NUMINAMATH_CALUDE_peach_multiple_l1411_141185

theorem peach_multiple (martine_peaches benjy_peaches gabrielle_peaches m : ℕ) : 
  martine_peaches = m * benjy_peaches + 6 →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_peach_multiple_l1411_141185


namespace NUMINAMATH_CALUDE_jacksons_grade_l1411_141141

/-- Calculates a student's grade based on study time and grade increase rate -/
def calculate_grade (video_game_hours : ℝ) (study_time_ratio : ℝ) (grade_increase_rate : ℝ) : ℝ :=
  video_game_hours * study_time_ratio * grade_increase_rate

/-- Proves that Jackson's grade is 45 points given the problem conditions -/
theorem jacksons_grade :
  let video_game_hours : ℝ := 9
  let study_time_ratio : ℝ := 1/3
  let grade_increase_rate : ℝ := 15
  calculate_grade video_game_hours study_time_ratio grade_increase_rate = 45 := by
  sorry


end NUMINAMATH_CALUDE_jacksons_grade_l1411_141141


namespace NUMINAMATH_CALUDE_no_non_zero_integer_solution_l1411_141127

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_integer_solution_l1411_141127


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l1411_141116

theorem theater_ticket_sales
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l1411_141116


namespace NUMINAMATH_CALUDE_negative_inequality_l1411_141134

theorem negative_inequality (a b : ℝ) (h : a > b) : -2 - a < -2 - b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l1411_141134


namespace NUMINAMATH_CALUDE_inequality_proof_l1411_141187

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (c/a)*(8*b+c) + (d/b)*(8*c+d) + (a/c)*(8*d+a) + (b/d)*(8*a+b) ≥ 9*(a+b+c+d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1411_141187


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1411_141145

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 6*y - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1411_141145


namespace NUMINAMATH_CALUDE_gcd_of_44_33_55_l1411_141109

/-- The greatest common divisor of 44, 33, and 55 is 11. -/
theorem gcd_of_44_33_55 : Nat.gcd 44 (Nat.gcd 33 55) = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_44_33_55_l1411_141109
