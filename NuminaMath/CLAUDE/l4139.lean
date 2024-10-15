import Mathlib

namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l4139_413911

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) : B a ⊆ A → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l4139_413911


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l4139_413926

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  (is_pythagorean_triple 6 8 10) ∧
  ¬(is_pythagorean_triple 6 7 10) ∧
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 8) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l4139_413926


namespace NUMINAMATH_CALUDE_juice_boxes_calculation_l4139_413924

/-- Calculates the total number of juice boxes needed for a school year. -/
def total_juice_boxes (num_children : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  num_children * days_per_week * weeks_per_year

/-- Proves that the total number of juice boxes needed for the given conditions is 375. -/
theorem juice_boxes_calculation :
  let num_children : ℕ := 3
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 25
  total_juice_boxes num_children days_per_week weeks_per_year = 375 := by
  sorry


end NUMINAMATH_CALUDE_juice_boxes_calculation_l4139_413924


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4139_413904

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x, x^2 - 1 > 0 ∧ ¬(x - 1 > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4139_413904


namespace NUMINAMATH_CALUDE_shaded_percentage_7x7_grid_l4139_413914

/-- The percentage of shaded squares in a 7x7 grid with 20 shaded squares -/
theorem shaded_percentage_7x7_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 7 * 7 →
  shaded_squares = 20 →
  (shaded_squares : Real) / total_squares * 100 = 20 / 49 * 100 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_7x7_grid_l4139_413914


namespace NUMINAMATH_CALUDE_suit_price_problem_l4139_413913

theorem suit_price_problem (P : ℝ) : 
  (0.7 * (1.3 * P) = 182) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_problem_l4139_413913


namespace NUMINAMATH_CALUDE_sector_area_l4139_413927

theorem sector_area (circumference : ℝ) (central_angle : ℝ) : 
  circumference = 16 * Real.pi → 
  central_angle = Real.pi / 4 → 
  (central_angle / (2 * Real.pi)) * ((circumference^2) / (4 * Real.pi)) = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l4139_413927


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l4139_413932

theorem greatest_perimeter_of_special_triangle : 
  let is_valid_triangle (x : ℕ) := x + 3*x > 15 ∧ x + 15 > 3*x ∧ 3*x + 15 > x
  let perimeter (x : ℕ) := x + 3*x + 15
  ∀ x : ℕ, is_valid_triangle x → perimeter x ≤ 43 ∧ ∃ y : ℕ, is_valid_triangle y ∧ perimeter y = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l4139_413932


namespace NUMINAMATH_CALUDE_count_valid_a_l4139_413909

theorem count_valid_a : ∃! n : ℕ, n > 0 ∧ 
  (∃ a_set : Finset ℕ, 
    (∀ a ∈ a_set, a > 0 ∧ 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27) ∧
    (∀ a : ℕ, a > 0 → 3 ∣ a → a ∣ 18 → a ∣ 27 → a ∈ a_set) ∧
    Finset.card a_set = n) :=
by sorry

end NUMINAMATH_CALUDE_count_valid_a_l4139_413909


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l4139_413916

theorem greatest_prime_factor_of_expression :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l4139_413916


namespace NUMINAMATH_CALUDE_donny_gas_change_l4139_413902

/-- Calculates the change Donny receives after filling up his truck's gas tank. -/
theorem donny_gas_change (tank_capacity : ℕ) (initial_fuel : ℕ) (fuel_cost : ℕ) (payment : ℕ) : 
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check donny_gas_change

end NUMINAMATH_CALUDE_donny_gas_change_l4139_413902


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_constant_is_zero_l4139_413919

/-- The conversion factor from Celsius to Fahrenheit -/
def celsius_to_fahrenheit_factor : ℚ := 9 / 5

/-- The change in Fahrenheit temperature -/
def fahrenheit_change : ℚ := 26

/-- The change in Celsius temperature -/
def celsius_change : ℚ := 14.444444444444445

/-- The constant in the Celsius to Fahrenheit conversion formula -/
def celsius_to_fahrenheit_constant : ℚ := 0

theorem celsius_to_fahrenheit_constant_is_zero :
  celsius_to_fahrenheit_constant = 0 := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_constant_is_zero_l4139_413919


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l4139_413917

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the condition for the set of points
def SatisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z - i) + Complex.abs (z + i) = 3

-- Define the set of points satisfying the condition
def PointSet : Set ℂ :=
  {z : ℂ | SatisfiesCondition z}

-- Theorem statement
theorem trajectory_is_ellipse :
  ∃ (a b : ℝ) (center : ℂ), 
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    PointSet = {z : ℂ | (z.re - center.re)^2 / a^2 + (z.im - center.im)^2 / b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l4139_413917


namespace NUMINAMATH_CALUDE_haj_daily_cost_l4139_413907

/-- The daily operation cost for Mr. Haj's grocery store -/
def daily_cost : ℝ → Prop := λ T => 
  -- 2/5 of total cost is for salary
  let salary := (2/5) * T
  -- Remaining after salary
  let remaining_after_salary := T - salary
  -- 1/4 of remaining after salary is for delivery
  let delivery := (1/4) * remaining_after_salary
  -- Amount for orders
  let orders := 1800
  -- Total cost equals sum of salary, delivery, and orders
  T = salary + delivery + orders

/-- Theorem stating the daily operation cost for Mr. Haj's grocery store -/
theorem haj_daily_cost : ∃ T : ℝ, daily_cost T ∧ T = 8000 := by
  sorry

end NUMINAMATH_CALUDE_haj_daily_cost_l4139_413907


namespace NUMINAMATH_CALUDE_complex_square_sum_of_squares_l4139_413928

theorem complex_square_sum_of_squares (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑a + Complex.I * ↑b)^2 = (3 : ℂ) + Complex.I * 4 →
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_complex_square_sum_of_squares_l4139_413928


namespace NUMINAMATH_CALUDE_kolya_mistake_l4139_413925

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens < 10
  h_ones : ones < 10

/-- Represents a four-digit number of the form effe -/
structure FourDigitNumberEFFE where
  e : Nat
  f : Nat
  h_e : e < 10
  h_f : f < 10

/-- Function to check if a two-digit number is divisible by 11 -/
def isDivisibleBy11 (n : TwoDigitNumber) : Prop :=
  (n.tens - n.ones) % 11 = 0

/-- The main theorem -/
theorem kolya_mistake
  (ab cd : TwoDigitNumber)
  (effe : FourDigitNumberEFFE)
  (h_mult : ab.tens * 10 + ab.ones * cd.tens * 10 + cd.ones = effe.e * 1000 + effe.f * 100 + effe.f * 10 + effe.e)
  (h_distinct : ab.tens ≠ ab.ones ∧ cd.tens ≠ cd.ones ∧ ab.tens ≠ cd.tens ∧ ab.tens ≠ cd.ones ∧ ab.ones ≠ cd.tens ∧ ab.ones ≠ cd.ones)
  : isDivisibleBy11 ab ∨ isDivisibleBy11 cd :=
sorry

end NUMINAMATH_CALUDE_kolya_mistake_l4139_413925


namespace NUMINAMATH_CALUDE_count_squares_l4139_413923

/-- The number of groups of squares in the figure -/
def num_groups : ℕ := 5

/-- The number of squares in each group -/
def squares_per_group : ℕ := 5

/-- The total number of squares in the figure -/
def total_squares : ℕ := num_groups * squares_per_group

theorem count_squares : total_squares = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_l4139_413923


namespace NUMINAMATH_CALUDE_transportation_budget_theorem_l4139_413918

def total_budget : ℝ := 1200000

def known_percentages : List ℝ := [39, 27, 14, 9, 5, 3.5]

def transportation_percentage : ℝ := 100 - (known_percentages.sum)

theorem transportation_budget_theorem :
  (transportation_percentage = 2.5) ∧
  (transportation_percentage / 100 * 360 = 9) ∧
  (transportation_percentage / 100 * 360 * π / 180 = π / 20) ∧
  (transportation_percentage / 100 * total_budget = 30000) :=
by sorry

end NUMINAMATH_CALUDE_transportation_budget_theorem_l4139_413918


namespace NUMINAMATH_CALUDE_fraction_integer_condition_l4139_413901

theorem fraction_integer_condition (n : ℤ) : 
  (↑(n + 1) / ↑(2 * n - 1) : ℚ).isInt ↔ n = 2 ∨ n = 1 ∨ n = 0 ∨ n = -1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_condition_l4139_413901


namespace NUMINAMATH_CALUDE_nested_sqrt_range_l4139_413900

theorem nested_sqrt_range :
  ∃ y : ℝ, y = Real.sqrt (4 + y) ∧ 2 ≤ y ∧ y < 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_range_l4139_413900


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l4139_413905

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the circle -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- Length of the chord on y-axis -/
def chord_length : ℝ := 8

theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 4)^2 + y^2 = r^2) ∧ 
  (∃ (a : ℝ), a^2 + x^2 = (chord_length/2)^2) →
  trajectory_equation x y := by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l4139_413905


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l4139_413922

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x : ℝ | x^2 ≥ 4}

theorem set_intersection_theorem :
  P ∩ (Set.univ \ Q) = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l4139_413922


namespace NUMINAMATH_CALUDE_student_council_max_profit_l4139_413912

/-- Calculate the maximum amount of money the student council can make from selling erasers --/
theorem student_council_max_profit (
  boxes : ℕ)
  (erasers_per_box : ℕ)
  (price_per_eraser : ℚ)
  (bulk_discount_rate : ℚ)
  (bulk_purchase_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : boxes = 48)
  (h2 : erasers_per_box = 24)
  (h3 : price_per_eraser = 3/4)
  (h4 : bulk_discount_rate = 1/10)
  (h5 : bulk_purchase_threshold = 10)
  (h6 : sales_tax_rate = 3/50)
  : ∃ (max_profit : ℚ), max_profit = 82426/100 :=
by
  sorry

end NUMINAMATH_CALUDE_student_council_max_profit_l4139_413912


namespace NUMINAMATH_CALUDE_smallest_upper_bound_D_l4139_413933

def D (n : ℕ+) : ℚ := 5 - (2 * n.val + 5 : ℚ) / 2^n.val

theorem smallest_upper_bound_D :
  ∃ t : ℕ, (∀ n : ℕ+, D n < t) ∧ (∀ s : ℕ, s < t → ∃ m : ℕ+, D m ≥ s) :=
  sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_D_l4139_413933


namespace NUMINAMATH_CALUDE_tensor_equation_solution_l4139_413930

/-- Custom binary operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_equation_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_tensor_equation_solution_l4139_413930


namespace NUMINAMATH_CALUDE_function_equivalence_l4139_413921

theorem function_equivalence : ∀ x : ℝ, (3 * x)^3 = x := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l4139_413921


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l4139_413903

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (l m : Line) (α : Plane) 
  (h1 : l ≠ m) 
  (h2 : perp l α) 
  (h3 : perp m α) : 
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l4139_413903


namespace NUMINAMATH_CALUDE_problem_solution_l4139_413908

theorem problem_solution (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : m + n = 105 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4139_413908


namespace NUMINAMATH_CALUDE_cubic_double_root_abs_ab_l4139_413929

/-- Given a cubic polynomial with a double root and an integer third root, prove |ab| = 3360 -/
theorem cubic_double_root_abs_ab (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, (x - r)^2 * (x - s) = x^3 + a*x^2 + b*x + 16*a)) →
  |a * b| = 3360 :=
by sorry

end NUMINAMATH_CALUDE_cubic_double_root_abs_ab_l4139_413929


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l4139_413915

theorem salt_solution_mixture (x : ℝ) : 
  (0.20 * x + 0.60 * 40 = 0.40 * (x + 40)) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l4139_413915


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4139_413931

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (48 - b) + c / (72 - c) = 9) :
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4139_413931


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l4139_413920

theorem vector_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l4139_413920


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4139_413934

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 9) : 
  a^2 + b^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l4139_413934


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l4139_413906

/-- Given a train and platform with known dimensions, calculate the time to cross the platform. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_cross_pole : ℝ) 
  (train_length_positive : 0 < train_length)
  (platform_length_positive : 0 < platform_length)
  (time_to_cross_pole_positive : 0 < time_to_cross_pole) :
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 
    (train_length + platform_length) * time_to_cross_pole / train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l4139_413906


namespace NUMINAMATH_CALUDE_water_tank_capacity_l4139_413910

theorem water_tank_capacity : ∀ (c : ℝ), c > 0 →
  (1 / 3 : ℝ) * c + 5 = (1 / 2 : ℝ) * c → c = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l4139_413910
