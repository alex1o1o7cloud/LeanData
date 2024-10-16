import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l2955_295519

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l2955_295519


namespace NUMINAMATH_CALUDE_volume_of_CO2_released_l2955_295565

/-- The volume of CO₂ gas released in a chemical reaction --/
theorem volume_of_CO2_released (n : ℝ) (Vₘ : ℝ) (h1 : n = 2.4) (h2 : Vₘ = 22.4) :
  n * Vₘ = 53.76 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_CO2_released_l2955_295565


namespace NUMINAMATH_CALUDE_negation_equivalence_l2955_295544

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2955_295544


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2955_295577

/-- The function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2955_295577


namespace NUMINAMATH_CALUDE_double_mean_value_function_range_l2955_295573

/-- A function f is a double mean value function on [a,b] if there exist
    two distinct points x₁ and x₂ in (a,b) such that
    f''(x₁) = f''(x₂) = (f(b) - f(a)) / (b - a) -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
  (deriv^[2] f) x₁ = (f b - f a) / (b - a) ∧
  (deriv^[2] f) x₂ = (f b - f a) / (b - a)

/-- The main theorem -/
theorem double_mean_value_function_range :
  ∀ m : ℝ, is_double_mean_value_function (fun x ↦ x^3 - 6/5 * x^2) 0 m →
  3/5 < m ∧ m ≤ 6/5 := by sorry

end NUMINAMATH_CALUDE_double_mean_value_function_range_l2955_295573


namespace NUMINAMATH_CALUDE_part1_part2_l2955_295587

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) : 
  (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2955_295587


namespace NUMINAMATH_CALUDE_tangent_circles_radii_product_l2955_295551

/-- A circle tangent to both x and y axes with center (a, a) and radius a -/
def TangentCircle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - a)^2 = a^2}

/-- The condition that the circle passes through (3, 4) -/
def PassesThroughPoint (a : ℝ) : Prop :=
  (3 - a)^2 + (4 - a)^2 = a^2

theorem tangent_circles_radii_product :
  ∃ r₁ r₂ : ℝ,
    (PassesThroughPoint r₁ ∧ PassesThroughPoint r₂) ∧
    (r₁ ≠ r₂) ∧
    (r₁ * r₂ = 25) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_product_l2955_295551


namespace NUMINAMATH_CALUDE_faster_train_speed_l2955_295575

/-- Calculates the speed of a faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 86 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l2955_295575


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2955_295585

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/5 is 5/4 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/5
  let S : ℝ := ∑' n, a * r^n
  S = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2955_295585


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l2955_295512

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6 : ℚ) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l2955_295512


namespace NUMINAMATH_CALUDE_product_of_r_values_l2955_295578

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x) = (r₁ - x) / 8 ↔ 1 / (3 * x) = (r₂ - x) / 8)) ∧ 
  (∀ r : ℝ, (∃! x : ℝ, x ≠ 0 ∧ 1 / (3 * x) = (r - x) / 8) → (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -32/3 :=
sorry

end NUMINAMATH_CALUDE_product_of_r_values_l2955_295578


namespace NUMINAMATH_CALUDE_product_of_xy_l2955_295530

theorem product_of_xy (x y : ℝ) (h : 3 * (2 * x * y + 9) = 51) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_xy_l2955_295530


namespace NUMINAMATH_CALUDE_number_of_valid_passwords_l2955_295516

/-- The number of digits in the password -/
def password_length : ℕ := 5

/-- The range of possible digits -/
def digit_range : ℕ := 10

/-- The number of passwords starting with the forbidden sequence -/
def forbidden_passwords : ℕ := 10

/-- Calculates the number of valid passwords -/
def valid_passwords : ℕ := digit_range ^ password_length - forbidden_passwords

/-- Theorem stating the number of valid passwords -/
theorem number_of_valid_passwords : valid_passwords = 99990 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_passwords_l2955_295516


namespace NUMINAMATH_CALUDE_dish_washing_time_l2955_295539

theorem dish_washing_time (dawn_time andy_time : ℕ) : 
  andy_time = 2 * dawn_time + 6 →
  andy_time = 46 →
  dawn_time = 20 := by
sorry

end NUMINAMATH_CALUDE_dish_washing_time_l2955_295539


namespace NUMINAMATH_CALUDE_series_sum_l2955_295543

/-- The sum of a specific infinite series given positive real numbers a and b where a > 3b -/
theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > 3 * b) : 
  let series_term (n : ℕ) := 1 / (((3 * n - 6) * a - (n^2 - 5*n + 6) * b) * ((3 * n - 3) * a - (n^2 - 4*n + 3) * b))
  ∑' n, series_term n = 1 / (b * (a - b)) := by
sorry

end NUMINAMATH_CALUDE_series_sum_l2955_295543


namespace NUMINAMATH_CALUDE_no_roots_implies_not_integer_l2955_295503

theorem no_roots_implies_not_integer (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) :
  ¬ ∃ n : ℤ, 20*(b - a) = n := by
  sorry

end NUMINAMATH_CALUDE_no_roots_implies_not_integer_l2955_295503


namespace NUMINAMATH_CALUDE_town_distance_bounds_l2955_295589

/-- Given two towns A and B that are 8 km apart, and towns B and C that are 10 km apart,
    prove that the distance between towns A and C is at least 2 km and at most 18 km. -/
theorem town_distance_bounds (A B C : ℝ × ℝ) : 
  dist A B = 8 → dist B C = 10 → 2 ≤ dist A C ∧ dist A C ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_town_distance_bounds_l2955_295589


namespace NUMINAMATH_CALUDE_scientific_notation_152300_l2955_295568

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_152300 :
  toScientificNotation 152300 = ScientificNotation.mk 1.523 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_152300_l2955_295568


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l2955_295515

theorem sum_of_real_solutions (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), 
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₁)) = x₁) ∧
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₂)) = x₂) ∧
    (x₁ + x₂ = Real.sqrt (3 * a + Real.sqrt (2 * a)) + Real.sqrt (3 * a - Real.sqrt (2 * a))) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l2955_295515


namespace NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2955_295562

theorem average_after_removing_two_numbers
  (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) (final_avg : ℚ)
  (h1 : n = 50)
  (h2 : initial_avg = 38)
  (h3 : removed1 = 45)
  (h4 : removed2 = 55)
  (h5 : final_avg = 37.5) :
  initial_avg * n - (removed1 + removed2) = final_avg * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_average_after_removing_two_numbers_l2955_295562


namespace NUMINAMATH_CALUDE_intersection_size_l2955_295596

/-- Given a finite universe U and two subsets A and B, 
    this theorem calculates the size of their intersection. -/
theorem intersection_size 
  (U A B : Finset ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : Finset.card U = 215)
  (h4 : Finset.card A = 170)
  (h5 : Finset.card B = 142)
  (h6 : Finset.card (U \ (A ∪ B)) = 38) :
  Finset.card (A ∩ B) = 135 := by
sorry

end NUMINAMATH_CALUDE_intersection_size_l2955_295596


namespace NUMINAMATH_CALUDE_aaron_cards_proof_l2955_295527

def aaron_final_cards (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial_aaron + found - lost - given

theorem aaron_cards_proof (initial_arthur : ℕ) (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ)
  (h1 : initial_arthur = 6)
  (h2 : initial_aaron = 5)
  (h3 : found = 62)
  (h4 : lost = 15)
  (h5 : given = 28) :
  aaron_final_cards initial_aaron found lost given = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_aaron_cards_proof_l2955_295527


namespace NUMINAMATH_CALUDE_number_problem_l2955_295541

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2955_295541


namespace NUMINAMATH_CALUDE_expected_participants_l2955_295511

/-- The expected number of participants in a school clean-up event after three years,
    given an initial number of participants and an annual increase rate. -/
theorem expected_participants (initial : ℕ) (increase_rate : ℚ) :
  initial = 800 →
  increase_rate = 1/2 →
  (initial * (1 + increase_rate)^3 : ℚ) = 2700 := by
  sorry

end NUMINAMATH_CALUDE_expected_participants_l2955_295511


namespace NUMINAMATH_CALUDE_minimum_dimes_for_scarf_l2955_295571

/-- The cost of the scarf in cents -/
def scarf_cost : ℕ := 4285

/-- The amount of money Chloe has without dimes, in cents -/
def initial_money : ℕ := 4000 + 100 + 50

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed to buy the scarf -/
def min_dimes_needed : ℕ := 14

theorem minimum_dimes_for_scarf :
  min_dimes_needed = (scarf_cost - initial_money + dime_value - 1) / dime_value :=
by sorry

end NUMINAMATH_CALUDE_minimum_dimes_for_scarf_l2955_295571


namespace NUMINAMATH_CALUDE_flowers_per_vase_is_nine_l2955_295524

/-- The number of carnations -/
def carnations : ℕ := 4

/-- The number of roses -/
def roses : ℕ := 23

/-- The total number of vases needed -/
def vases : ℕ := 3

/-- The total number of flowers -/
def total_flowers : ℕ := carnations + roses

/-- The number of flowers one vase can hold -/
def flowers_per_vase : ℕ := total_flowers / vases

theorem flowers_per_vase_is_nine : flowers_per_vase = 9 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_vase_is_nine_l2955_295524


namespace NUMINAMATH_CALUDE_unique_x_divisible_by_15_l2955_295572

def is_valid_x (x : ℕ) : Prop :=
  x < 10 ∧ (∃ n : ℕ, x * 1000 + 200 + x * 10 + 3 = 15 * n)

theorem unique_x_divisible_by_15 : ∃! x : ℕ, is_valid_x x :=
  sorry

end NUMINAMATH_CALUDE_unique_x_divisible_by_15_l2955_295572


namespace NUMINAMATH_CALUDE_population_ratio_l2955_295525

-- Define populations as real numbers
variable (P_A P_B P_C P_D P_E P_F : ℝ)

-- Define the relationships between city populations
def population_relations : Prop :=
  (P_A = 8 * P_B) ∧
  (P_B = 5 * P_C) ∧
  (P_D = 3 * P_C) ∧
  (P_D = P_E / 2) ∧
  (P_F = P_A / 4)

-- Theorem to prove
theorem population_ratio (h : population_relations P_A P_B P_C P_D P_E P_F) :
  P_E / P_B = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l2955_295525


namespace NUMINAMATH_CALUDE_find_divisor_l2955_295532

theorem find_divisor (x : ℝ) (y : ℝ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 4) / y = 5) : 
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2955_295532


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2955_295554

/-- Given a three-digit number abc where a, b, and c are non-zero digits,
    prove that abc = 425 if the sum of the other five three-digit numbers
    formed by rearranging a, b, c is 2017. -/
theorem three_digit_number_problem (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) +
  (100 * a + 10 * c + b) +
  (100 * b + 10 * a + c) +
  (100 * b + 10 * c + a) +
  (100 * c + 10 * a + b) = 2017 →
  100 * a + 10 * b + c = 425 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2955_295554


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l2955_295597

/-- The slope of the line tangent to a circle at the point (8, 3) is -1, 
    given that the center of the circle is at (1, -4). -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (1, -4) → point = (8, 3) → 
  (point.1 - center.1) * (point.2 - center.2) = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l2955_295597


namespace NUMINAMATH_CALUDE_expression_value_l2955_295534

theorem expression_value : ∀ x : ℝ, x = 2 → 3 * x^2 - 4 * x + 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2955_295534


namespace NUMINAMATH_CALUDE_fruit_salad_cherries_l2955_295552

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Checks if the fruit salad satisfies the given conditions --/
def isValidFruitSalad (fs : FruitSalad) : Prop :=
  fs.blueberries + fs.raspberries + fs.grapes + fs.cherries = 280 ∧
  fs.raspberries = 2 * fs.blueberries ∧
  fs.grapes = 3 * fs.cherries ∧
  fs.cherries = 4 * fs.raspberries

/-- Theorem stating that a valid fruit salad has 64 cherries --/
theorem fruit_salad_cherries (fs : FruitSalad) :
  isValidFruitSalad fs → fs.cherries = 64 := by
  sorry

#check fruit_salad_cherries

end NUMINAMATH_CALUDE_fruit_salad_cherries_l2955_295552


namespace NUMINAMATH_CALUDE_hike_attendance_l2955_295507

/-- The number of cars used for the hike -/
def num_cars : ℕ := 3

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 2

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The total number of people who went on the hike -/
def total_people : ℕ := num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van

theorem hike_attendance : total_people = 58 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l2955_295507


namespace NUMINAMATH_CALUDE_initial_leaves_count_l2955_295574

/-- The number of leaves Mikey had initially -/
def initial_leaves : ℕ := sorry

/-- The number of leaves that blew away -/
def blown_leaves : ℕ := 244

/-- The number of leaves left -/
def remaining_leaves : ℕ := 112

/-- Theorem stating that the initial number of leaves is 356 -/
theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end NUMINAMATH_CALUDE_initial_leaves_count_l2955_295574


namespace NUMINAMATH_CALUDE_expression_simplification_l2955_295555

theorem expression_simplification :
  (12 - 2 * Real.sqrt 35 + Real.sqrt 14 + Real.sqrt 10) / (Real.sqrt 7 - Real.sqrt 5 + Real.sqrt 2) = 2 * Real.sqrt 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2955_295555


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l2955_295542

/-- Given two parabolas C₁ and C₂ with specific properties, prove that C₂ passes through a fixed point. -/
theorem parabola_intersection_fixed_point 
  (C₁_vertex : ℝ × ℝ) 
  (C₁_focus : ℝ × ℝ)
  (a b : ℝ) :
  let C₁_vertex_x := Real.sqrt 2 - 1
  let C₁_vertex_y := 1
  let C₁_focus_x := Real.sqrt 2 - 3/4
  let C₁_focus_y := 1
  let C₂_eq (x y : ℝ) := y^2 - a*y + x + 2*b = 0
  let fixed_point := (Real.sqrt 2 - 1/2, 1)
  C₁_vertex = (C₁_vertex_x, C₁_vertex_y) →
  C₁_focus = (C₁_focus_x, C₁_focus_y) →
  (∃ (x₀ y₀ : ℝ), 
    (y₀^2 - 2*y₀ - x₀ + Real.sqrt 2 = 0) ∧ 
    (C₂_eq x₀ y₀) ∧ 
    ((2*y₀ - 2) * (2*y₀ - a) = -1)) →
  C₂_eq fixed_point.1 fixed_point.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_point_l2955_295542


namespace NUMINAMATH_CALUDE_four_solutions_l2955_295536

def is_solution (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (4 : ℚ) / m + (2 : ℚ) / n = 1

def solution_count : ℕ := 4

theorem four_solutions :
  ∃ (S : Finset (ℕ × ℕ)), S.card = solution_count ∧
    (∀ (p : ℕ × ℕ), p ∈ S ↔ is_solution p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l2955_295536


namespace NUMINAMATH_CALUDE_parabola_translation_l2955_295508

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola --/
def original : Parabola := { a := -2, h := -2, k := 3 }

/-- The translated parabola --/
def translated : Parabola := { a := -2, h := 1, k := -1 }

/-- The translation that moves the original parabola to the translated parabola --/
def translation : Translation := { dx := 3, dy := -4 }

theorem parabola_translation : 
  ∀ (x y : ℝ), 
  (y = -2 * (x - translated.h)^2 + translated.k) ↔ 
  (y + translation.dy = -2 * ((x - translation.dx) - original.h)^2 + original.k) :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2955_295508


namespace NUMINAMATH_CALUDE_special_function_value_at_one_l2955_295550

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y, and f(2) = 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ f 2 = 4

/-- Theorem: If f is a special function, then f(1) = 2 -/
theorem special_function_value_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_at_one_l2955_295550


namespace NUMINAMATH_CALUDE_prize_location_questions_l2955_295598

/-- Represents the three doors in the game show. -/
inductive Door
| left
| center
| right

/-- Represents the host's response to a question. -/
inductive Response
| yes
| no

/-- The maximum number of lies the host can tell. -/
def max_lies : ℕ := 10

/-- The function that determines the minimum number of questions needed to locate the prize. -/
def min_questions_to_locate_prize (doors : List Door) (max_lies : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 32 questions are needed to locate the prize with certainty. -/
theorem prize_location_questions (doors : List Door) (h1 : doors.length = 3) :
  min_questions_to_locate_prize doors max_lies = 32 :=
sorry

end NUMINAMATH_CALUDE_prize_location_questions_l2955_295598


namespace NUMINAMATH_CALUDE_f_range_on_domain_l2955_295580

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_range_on_domain :
  ∃ (min max : ℝ), min = -1 ∧ max = 8 ∧
  (∀ x ∈ domain, min ≤ f x ∧ f x ≤ max) ∧
  (∃ x₁ ∈ domain, f x₁ = min) ∧
  (∃ x₂ ∈ domain, f x₂ = max) :=
sorry

end NUMINAMATH_CALUDE_f_range_on_domain_l2955_295580


namespace NUMINAMATH_CALUDE_custom_operation_equality_l2955_295549

/-- Custom operation $ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l2955_295549


namespace NUMINAMATH_CALUDE_cube_volumes_from_surface_area_l2955_295563

theorem cube_volumes_from_surface_area :
  ∀ a b c : ℕ,
  (6 * (a^2 + b^2 + c^2) = 564) →
  (a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volumes_from_surface_area_l2955_295563


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2955_295566

/-- The asymptotes of the hyperbola x²/4 - y² = 1 are y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → 
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2955_295566


namespace NUMINAMATH_CALUDE_jack_vacation_budget_l2955_295593

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents the amount of money Jack has saved in base 8 -/
def jack_savings : ℕ := 3777

/-- Represents the cost of the airline ticket in base 10 -/
def ticket_cost : ℕ := 1200

/-- Calculates the remaining money after buying the ticket -/
def remaining_money : ℕ := base8_to_base10 jack_savings - ticket_cost

theorem jack_vacation_budget :
  remaining_money = 847 := by sorry

end NUMINAMATH_CALUDE_jack_vacation_budget_l2955_295593


namespace NUMINAMATH_CALUDE_six_digit_difference_l2955_295560

/-- Function f for 6-digit numbers -/
def f (n : ℕ) : ℕ :=
  let u := n / 100000 % 10
  let v := n / 10000 % 10
  let w := n / 1000 % 10
  let x := n / 100 % 10
  let y := n / 10 % 10
  let z := n % 10
  2^u * 3^v * 5^w * 7^x * 11^y * 13^z

/-- Theorem: If f(abcdef) = 13 * f(ghijkl), then abcdef - ghijkl = 1 -/
theorem six_digit_difference (abcdef ghijkl : ℕ) 
  (h1 : 100000 ≤ abcdef ∧ abcdef < 1000000)
  (h2 : 100000 ≤ ghijkl ∧ ghijkl < 1000000)
  (h3 : f abcdef = 13 * f ghijkl) : 
  abcdef - ghijkl = 1 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_difference_l2955_295560


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l2955_295557

/-- The distance from Shinyoung's house to the midpoint of the path to school -/
theorem distance_to_midpoint (house_to_office village_to_school : ℕ) : 
  house_to_office = 1700 →
  village_to_school = 900 →
  (house_to_office + village_to_school) / 2 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l2955_295557


namespace NUMINAMATH_CALUDE_largest_divisor_of_a_pow_25_minus_a_l2955_295520

theorem largest_divisor_of_a_pow_25_minus_a : 
  ∃ (n : ℕ), n = 2730 ∧ 
  (∀ (a : ℤ), (a^25 - a) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (a : ℤ), (a^25 - a) % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_a_pow_25_minus_a_l2955_295520


namespace NUMINAMATH_CALUDE_subtracted_number_proof_l2955_295559

theorem subtracted_number_proof (initial_number : ℝ) (subtracted_number : ℝ) : 
  initial_number = 22.142857142857142 →
  ((initial_number + 5) * 7) / 5 - subtracted_number = 33 →
  subtracted_number = 5 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_proof_l2955_295559


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l2955_295553

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l2955_295553


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2955_295540

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : 
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2955_295540


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l2955_295529

/-- The ratio of the volume of a regular octahedron formed by joining the centers of adjoining faces
    of a cube to the volume of the cube, when the cube has a side length of 2 units. -/
theorem octahedron_cube_volume_ratio : 
  let cube_side : ℝ := 2
  let cube_volume : ℝ := cube_side ^ 3
  let octahedron_side : ℝ := Real.sqrt 2
  let octahedron_volume : ℝ := (octahedron_side ^ 3 * Real.sqrt 2) / 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry


end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l2955_295529


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2955_295591

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 5

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2955_295591


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2955_295548

theorem price_reduction_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 3000)
  (h2 : current_price = 2400) :
  (original_price - current_price) / original_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2955_295548


namespace NUMINAMATH_CALUDE_tennis_players_count_l2955_295518

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : neither = 5)
  (h4 : both = 3) :
  ∃ tennis : ℕ, tennis = 18 ∧ 
  tennis = total - neither - (badminton - both) := by
  sorry

end NUMINAMATH_CALUDE_tennis_players_count_l2955_295518


namespace NUMINAMATH_CALUDE_postage_cost_correct_l2955_295582

-- Define the postage pricing structure
def base_rate : ℚ := 50 / 100
def additional_rate : ℚ := 15 / 100
def weight_increment : ℚ := 1 / 2
def package_weight : ℚ := 28 / 10
def cost_cap : ℚ := 130 / 100

-- Calculate the postage cost
def postage_cost : ℚ :=
  base_rate + additional_rate * (Int.ceil ((package_weight - 1) / weight_increment))

-- Theorem to prove
theorem postage_cost_correct : 
  postage_cost = 110 / 100 ∧ postage_cost ≤ cost_cap := by
  sorry

end NUMINAMATH_CALUDE_postage_cost_correct_l2955_295582


namespace NUMINAMATH_CALUDE_max_sin_sum_l2955_295570

theorem max_sin_sum (α β θ : Real) : 
  α + β = 2 * Real.pi / 3 →
  α > 0 →
  β > 0 →
  (∀ x y, x + y = 2 * Real.pi / 3 → x > 0 → y > 0 → 
    Real.sin α + 2 * Real.sin β ≥ Real.sin x + 2 * Real.sin y) →
  α = θ →
  Real.cos θ = Real.sqrt 21 / 7 := by
sorry

end NUMINAMATH_CALUDE_max_sin_sum_l2955_295570


namespace NUMINAMATH_CALUDE_license_plate_count_l2955_295517

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := num_letters^3 * num_digits * num_even_digits

theorem license_plate_count :
  total_license_plates = 878800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2955_295517


namespace NUMINAMATH_CALUDE_min_value_expression_l2955_295506

open Real

theorem min_value_expression :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 - 12 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (3 - Real.sqrt 2) * Real.sin x + 1) *
    (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2955_295506


namespace NUMINAMATH_CALUDE_van_capacity_l2955_295595

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 22 → adults = 2 → vans = 3 → (students + adults) / vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l2955_295595


namespace NUMINAMATH_CALUDE_basketball_court_fits_l2955_295576

theorem basketball_court_fits (total_area : ℝ) (court_area : ℝ) (length_width_ratio : ℝ) (space_width : ℝ) :
  total_area = 1100 ∧ 
  court_area = 540 ∧ 
  length_width_ratio = 5/3 ∧
  space_width = 1 →
  ∃ (width : ℝ), 
    width > 0 ∧
    length_width_ratio * width * width = court_area ∧
    (length_width_ratio * width + 2 * space_width) * (width + 2 * space_width) ≤ total_area :=
by sorry

#check basketball_court_fits

end NUMINAMATH_CALUDE_basketball_court_fits_l2955_295576


namespace NUMINAMATH_CALUDE_percentage_of_difference_l2955_295510

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P * (x - y) = 0.3 * (x + y) →
  y = (1/3) * x →
  P = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l2955_295510


namespace NUMINAMATH_CALUDE_max_a_value_l2955_295545

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 1 + a * Real.cos x ≥ 2/3 * Real.sin (π/2 + 2*x)) → 
  a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2955_295545


namespace NUMINAMATH_CALUDE_total_bus_ride_distance_l2955_295505

theorem total_bus_ride_distance :
  let vince_ride : ℚ := 5/8
  let zachary_ride : ℚ := 1/2
  let alice_ride : ℚ := 17/20
  let rebecca_ride : ℚ := 2/5
  vince_ride + zachary_ride + alice_ride + rebecca_ride = 19/8
  := by sorry

end NUMINAMATH_CALUDE_total_bus_ride_distance_l2955_295505


namespace NUMINAMATH_CALUDE_grid_toothpick_count_l2955_295537

/-- Calculates the number of toothpicks in a grid with a missing center block -/
def toothpick_count (length width missing_size : ℕ) : ℕ :=
  let vertical := (length + 1) * width - missing_size * missing_size
  let horizontal := (width + 1) * length - missing_size * missing_size
  vertical + horizontal

/-- Theorem stating the correct number of toothpicks for the given grid -/
theorem grid_toothpick_count :
  toothpick_count 30 20 2 = 1242 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpick_count_l2955_295537


namespace NUMINAMATH_CALUDE_height_of_cone_l2955_295533

/-- Theorem: Height of a cone with specific volume and vertex angle -/
theorem height_of_cone (V : ℝ) (angle : ℝ) (h : ℝ) :
  V = 16384 * Real.pi ∧ angle = 90 →
  h = (49152 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_height_of_cone_l2955_295533


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l2955_295567

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates a sequence of selected numbers for systematic sampling -/
def generate_sequence (s : SystematicSampling) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if a sequence is valid for the given systematic sampling -/
def is_valid_sequence (s : SystematicSampling) (seq : List Nat) : Prop :=
  seq.length = s.sample_size ∧
  seq.all (· ≤ s.total_students) ∧
  seq = generate_sequence s

theorem correct_systematic_sampling :
  let s : SystematicSampling := {
    total_students := 60,
    sample_size := 5,
    start := 6,
    interval := 12
  }
  is_valid_sequence s [6, 18, 30, 42, 54] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sampling_l2955_295567


namespace NUMINAMATH_CALUDE_range_of_a_given_negative_root_l2955_295583

/-- Given that the equation 5^x = (a+3)/(5-a) has a negative root, 
    prove that the range of values for a is -3 < a < 1 -/
theorem range_of_a_given_negative_root (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3)/(5-a)) → -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_negative_root_l2955_295583


namespace NUMINAMATH_CALUDE_prob_art_second_given_pe_first_l2955_295501

def total_courses : ℕ := 6
def pe_courses : ℕ := 4
def art_courses : ℕ := 2

def prob_pe_first : ℚ := pe_courses / total_courses
def prob_art_second : ℚ := art_courses / (total_courses - 1)

theorem prob_art_second_given_pe_first :
  (prob_pe_first * prob_art_second) / prob_pe_first = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_prob_art_second_given_pe_first_l2955_295501


namespace NUMINAMATH_CALUDE_sum_is_composite_l2955_295546

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (k m : ℕ+), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2955_295546


namespace NUMINAMATH_CALUDE_symmetric_absolute_value_function_l2955_295500

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_absolute_value_function (a : ℝ) :
  IsSymmetricAbout (fun x ↦ |x + 2*a| - 1) 1 → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_absolute_value_function_l2955_295500


namespace NUMINAMATH_CALUDE_sally_quarters_count_l2955_295513

/-- Given that Sally had 760 quarters initially and received 418 more quarters,
    prove that she now has 1178 quarters in total. -/
theorem sally_quarters_count (initial : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : initial = 760)
    (h2 : additional = 418)
    (h3 : total = initial + additional) :
  total = 1178 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_count_l2955_295513


namespace NUMINAMATH_CALUDE_card_value_decrease_l2955_295579

theorem card_value_decrease (initial_value : ℝ) (h : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.1)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.19 := by
sorry

end NUMINAMATH_CALUDE_card_value_decrease_l2955_295579


namespace NUMINAMATH_CALUDE_larger_number_ratio_l2955_295556

theorem larger_number_ratio (a b : ℕ+) (k : ℚ) (s : ℤ) 
  (h1 : (a : ℚ) / (b : ℚ) = k)
  (h2 : k < 1)
  (h3 : (a : ℤ) + (b : ℤ) = s) :
  max a b = |s| / (1 + k) :=
sorry

end NUMINAMATH_CALUDE_larger_number_ratio_l2955_295556


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2955_295538

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (4, 2)

-- Define a predicate for points on the circumcircle of triangle ABP
def on_circumcircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  ∀ A B : ℝ × ℝ,
  given_circle A.1 A.2 →
  given_circle B.1 B.2 →
  (∃ t : ℝ, A = (4 * t / (t^2 + 1), 2 * t^2 / (t^2 + 1))) →
  (∃ s : ℝ, B = (4 * s / (s^2 + 1), 2 * s^2 / (s^2 + 1))) →
  on_circumcircle A.1 A.2 ∧ on_circumcircle B.1 B.2 ∧ on_circumcircle point_P.1 point_P.2 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l2955_295538


namespace NUMINAMATH_CALUDE_minimum_value_x_plus_reciprocal_l2955_295523

theorem minimum_value_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ x₀ > 1, x₀ + 1 / (x₀ - 1) = 3 :=
sorry


end NUMINAMATH_CALUDE_minimum_value_x_plus_reciprocal_l2955_295523


namespace NUMINAMATH_CALUDE_tomatoes_needed_fried_green_tomatoes_l2955_295521

theorem tomatoes_needed (slices_per_tomato : ℕ) (slices_per_meal : ℕ) (people : ℕ) : ℕ :=
  let total_slices := slices_per_meal * people
  total_slices / slices_per_tomato

theorem fried_green_tomatoes :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_needed_fried_green_tomatoes_l2955_295521


namespace NUMINAMATH_CALUDE_karens_round_trip_distance_l2955_295569

/-- The total distance Karen covers for a round trip to the library -/
def total_distance (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  2 * (shelves * books_per_shelf)

/-- Proof that Karen's round trip distance is 3200 miles -/
theorem karens_round_trip_distance :
  total_distance 4 400 = 3200 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_karens_round_trip_distance_l2955_295569


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l2955_295547

/-- The number of trees in the coconut grove that yield 60 nuts per year -/
def trees_60 (x : ℝ) : ℝ := x + 3

/-- The number of trees in the coconut grove that yield 120 nuts per year -/
def trees_120 (x : ℝ) : ℝ := x

/-- The number of trees in the coconut grove that yield 180 nuts per year -/
def trees_180 (x : ℝ) : ℝ := x - 3

/-- The total number of trees in the coconut grove -/
def total_trees (x : ℝ) : ℝ := trees_60 x + trees_120 x + trees_180 x

/-- The total number of nuts produced by all trees in the coconut grove -/
def total_nuts (x : ℝ) : ℝ := 60 * trees_60 x + 120 * trees_120 x + 180 * trees_180 x

/-- The average yield per tree per year -/
def average_yield : ℝ := 100

theorem coconut_grove_problem :
  ∃ x : ℝ, total_nuts x = average_yield * total_trees x ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l2955_295547


namespace NUMINAMATH_CALUDE_distance_between_points_l2955_295522

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -1)
  let p2 : ℝ × ℝ := (7, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 74 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l2955_295522


namespace NUMINAMATH_CALUDE_laptop_price_l2955_295599

theorem laptop_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_l2955_295599


namespace NUMINAMATH_CALUDE_card_shuffle_bound_l2955_295528

theorem card_shuffle_bound (n : ℕ) (hn : n > 0) : 
  Nat.totient (2 * n - 1) ≤ 2 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_card_shuffle_bound_l2955_295528


namespace NUMINAMATH_CALUDE_alloy_b_ratio_l2955_295558

/-- Represents the composition of an alloy -/
structure Alloy where
  total_weight : ℝ
  tin_weight : ℝ
  lead_weight : ℝ
  copper_weight : ℝ

/-- The ratio of two components in an alloy -/
def ratio (a b : ℝ) : ℝ × ℝ := (a, b)

theorem alloy_b_ratio (alloy_a alloy_b : Alloy) (mixed_alloy : Alloy) :
  alloy_a.total_weight = 120 →
  alloy_b.total_weight = 180 →
  ratio alloy_a.lead_weight alloy_a.tin_weight = (2, 3) →
  mixed_alloy.tin_weight = 139.5 →
  mixed_alloy.total_weight = alloy_a.total_weight + alloy_b.total_weight →
  mixed_alloy.tin_weight = alloy_a.tin_weight + alloy_b.tin_weight →
  ratio alloy_b.tin_weight alloy_b.copper_weight = (3, 5) := by
  sorry

end NUMINAMATH_CALUDE_alloy_b_ratio_l2955_295558


namespace NUMINAMATH_CALUDE_box_of_books_l2955_295592

theorem box_of_books (box_weight : ℕ) (book_weight : ℕ) (h1 : box_weight = 42) (h2 : book_weight = 3) :
  box_weight / book_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_box_of_books_l2955_295592


namespace NUMINAMATH_CALUDE_doraemon_dorayakis_l2955_295584

/-- Represents the possible moves in rock-paper-scissors game -/
inductive Move
| Rock
| Scissors

/-- Represents the outcome of a single round -/
inductive Outcome
| Win
| Lose
| Tie

/-- Calculates the outcome of a round given two moves -/
def roundOutcome (move1 move2 : Move) : Outcome :=
  match move1, move2 with
  | Move.Rock, Move.Scissors => Outcome.Win
  | Move.Scissors, Move.Rock => Outcome.Lose
  | _, _ => Outcome.Tie

/-- Calculates the number of dorayakis received based on the outcome -/
def dorayakisForOutcome (outcome : Outcome) : Nat :=
  match outcome with
  | Outcome.Win => 2
  | Outcome.Lose => 0
  | Outcome.Tie => 1

/-- Represents a player's strategy -/
structure Strategy where
  move : Nat → Move

/-- Doraemon's strategy of always playing Rock -/
def doraemonStrategy : Strategy :=
  { move := λ _ => Move.Rock }

/-- Nobita's strategy of playing Scissors once every 10 rounds, Rock otherwise -/
def nobitaStrategy : Strategy :=
  { move := λ round => if round % 10 == 0 then Move.Scissors else Move.Rock }

/-- Calculates the total dorayakis received by a player over multiple rounds -/
def totalDorayakis (playerStrategy opponentStrategy : Strategy) (rounds : Nat) : Nat :=
  (List.range rounds).foldl (λ acc round =>
    acc + dorayakisForOutcome (roundOutcome (playerStrategy.move round) (opponentStrategy.move round))
  ) 0

theorem doraemon_dorayakis :
  totalDorayakis doraemonStrategy nobitaStrategy 20 = 10 ∧
  totalDorayakis nobitaStrategy doraemonStrategy 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_doraemon_dorayakis_l2955_295584


namespace NUMINAMATH_CALUDE_new_average_after_drop_l2955_295581

/-- Theorem: New average after student drops class -/
theorem new_average_after_drop (n : ℕ) (old_avg : ℚ) (drop_score : ℚ) :
  n = 16 →
  old_avg = 62.5 →
  drop_score = 70 →
  (n : ℚ) * old_avg - drop_score = ((n - 1) : ℚ) * 62 :=
by sorry

end NUMINAMATH_CALUDE_new_average_after_drop_l2955_295581


namespace NUMINAMATH_CALUDE_triangle_part1_triangle_part2_l2955_295502

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Part 1
theorem triangle_part1 (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  ((A = π / 3 ∧ C = 5 * π / 12 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
   (A = 2 * π / 3 ∧ C = π / 12 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
sorry

-- Part 2
theorem triangle_part2 (h1 : Real.cos B / Real.cos C = -b / (2 * a + c)) 
                       (h2 : b = Real.sqrt 13) (h3 : a + c = 4) :
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_part1_triangle_part2_l2955_295502


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2955_295514

theorem system_of_equations_solution : 
  let x : ℚ := -29/2
  let y : ℚ := -71/2
  (7 * x - 3 * y = 5) ∧ (y - 3 * x = 8) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2955_295514


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2955_295509

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 8]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * w i)

-- Theorem statement
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (a x) (b x) → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2955_295509


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l2955_295504

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem stating that the total amount spent is equal to the sum of individual game costs -/
theorem total_spent_equals_sum_of_games :
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l2955_295504


namespace NUMINAMATH_CALUDE_equation_solution_l2955_295531

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → x ≠ -6 → 
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2955_295531


namespace NUMINAMATH_CALUDE_first_season_episodes_l2955_295590

/-- The number of seasons in the TV show -/
def num_seasons : ℕ := 5

/-- The cost per episode for the first season in dollars -/
def first_season_cost : ℕ := 100000

/-- The cost per episode for seasons after the first in dollars -/
def other_season_cost : ℕ := 2 * first_season_cost

/-- The increase factor for the number of episodes in each season after the first -/
def episode_increase_factor : ℚ := 3/2

/-- The number of episodes in the last season -/
def last_season_episodes : ℕ := 24

/-- The total cost to produce all episodes in dollars -/
def total_cost : ℕ := 16800000

/-- Calculate the total cost of all seasons given the number of episodes in the first season -/
def calculate_total_cost (first_season_episodes : ℕ) : ℚ :=
  let first_season := first_season_cost * first_season_episodes
  let second_season := other_season_cost * (episode_increase_factor * first_season_episodes)
  let third_season := other_season_cost * (episode_increase_factor^2 * first_season_episodes)
  let fourth_season := other_season_cost * (episode_increase_factor^3 * first_season_episodes)
  let fifth_season := other_season_cost * last_season_episodes
  first_season + second_season + third_season + fourth_season + fifth_season

/-- Theorem stating that the number of episodes in the first season is 8 -/
theorem first_season_episodes : ∃ (x : ℕ), x = 8 ∧ calculate_total_cost x = total_cost := by
  sorry

end NUMINAMATH_CALUDE_first_season_episodes_l2955_295590


namespace NUMINAMATH_CALUDE_interesting_coeffs_of_product_l2955_295588

/-- A real number is interesting if it can be expressed as a + b√2 where a and b are integers -/
def interesting (r : ℝ) : Prop :=
  ∃ (a b : ℤ), r = a + b * Real.sqrt 2

/-- A polynomial with interesting coefficients -/
def interesting_poly (p : Polynomial ℝ) : Prop :=
  ∀ i, interesting (p.coeff i)

/-- The main theorem -/
theorem interesting_coeffs_of_product
  (A B Q : Polynomial ℝ)
  (hA : interesting_poly A)
  (hB : interesting_poly B)
  (hB_const : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  interesting_poly Q :=
sorry

end NUMINAMATH_CALUDE_interesting_coeffs_of_product_l2955_295588


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2955_295535

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2955_295535


namespace NUMINAMATH_CALUDE_solve_equation_l2955_295564

theorem solve_equation : ∃ x : ℝ, 2.25 * x = 45 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2955_295564


namespace NUMINAMATH_CALUDE_laundromat_cost_l2955_295526

def service_fee : ℝ := 3
def first_hour_cost : ℝ := 10
def additional_hour_cost : ℝ := 15
def usage_time : ℝ := 2.75
def discount_rate : ℝ := 0.1

def calculate_cost : ℝ :=
  let base_cost := first_hour_cost + (usage_time - 1) * additional_hour_cost
  let total_cost := base_cost + service_fee
  let discount := total_cost * discount_rate
  total_cost - discount

theorem laundromat_cost :
  calculate_cost = 35.32 := by sorry

end NUMINAMATH_CALUDE_laundromat_cost_l2955_295526


namespace NUMINAMATH_CALUDE_shirt_sweater_cost_l2955_295586

/-- The total cost of a shirt and a sweater given their price relationship -/
theorem shirt_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  shirt_price = sweater_price - 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end NUMINAMATH_CALUDE_shirt_sweater_cost_l2955_295586


namespace NUMINAMATH_CALUDE_stating_selling_price_is_43_l2955_295561

/-- Represents the selling price of an article when the loss is equal to the profit. -/
def selling_price_equal_loss_profit (cost_price : ℕ) (profit_price : ℕ) : ℕ :=
  cost_price * 2 - profit_price

/-- 
Theorem stating that the selling price of an article is 43 when the loss is equal to the profit,
given that the cost price is 50 and the profit obtained by selling for 57 is the same as the loss
obtained by selling for the unknown price.
-/
theorem selling_price_is_43 :
  selling_price_equal_loss_profit 50 57 = 43 := by
  sorry

#eval selling_price_equal_loss_profit 50 57

end NUMINAMATH_CALUDE_stating_selling_price_is_43_l2955_295561


namespace NUMINAMATH_CALUDE_amount_added_l2955_295594

theorem amount_added (N A : ℝ) : 
  N = 1.375 → 
  0.6667 * N + A = 1.6667 → 
  A = 0.750025 := by
sorry

end NUMINAMATH_CALUDE_amount_added_l2955_295594
