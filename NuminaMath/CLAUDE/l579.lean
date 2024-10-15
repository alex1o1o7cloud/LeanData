import Mathlib

namespace NUMINAMATH_CALUDE_adam_students_in_ten_years_l579_57914

theorem adam_students_in_ten_years : 
  let students_per_year : ℕ := 50
  let first_year_students : ℕ := 40
  let total_years : ℕ := 10
  (total_years - 1) * students_per_year + first_year_students = 490 := by
  sorry

end NUMINAMATH_CALUDE_adam_students_in_ten_years_l579_57914


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l579_57934

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (60/23, 50/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8*x - 5*y = 10

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 6*x + 2*y = 20

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l579_57934


namespace NUMINAMATH_CALUDE_rectangle_area_below_line_l579_57900

/-- Given a rectangle bounded by y = 2a, y = -b, x = -2c, and x = d, 
    where a, b, c, and d are positive real numbers, and a line y = x + a 
    intersecting the rectangle, this theorem states the area of the 
    rectangle below the line. -/
theorem rectangle_area_below_line 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let rectangle_area := (2*a + b) * (d + 2*c)
  let triangle_area := (1/2) * (d + 2*c + b + a) * (a + b + 2*c)
  rectangle_area - triangle_area = 
    (2*a + b) * (d + 2*c) - (1/2) * (d + 2*c + b + a) * (a + b + 2*c) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_below_line_l579_57900


namespace NUMINAMATH_CALUDE_ellipse_condition_l579_57975

/-- If the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2 ↔
      (x^2 / a^2) + ((y + 1)^2 / b^2) = 1) →
  m > 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l579_57975


namespace NUMINAMATH_CALUDE_statue_original_cost_l579_57979

theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 660)
  (h2 : profit_percentage = 20) :
  let original_cost := selling_price / (1 + profit_percentage / 100)
  original_cost = 550 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l579_57979


namespace NUMINAMATH_CALUDE_resistance_change_l579_57949

/-- Represents the change in resistance when a switch is closed in a circuit with three resistors. -/
theorem resistance_change (R₁ R₂ R₃ : ℝ) (h₁ : R₁ = 1) (h₂ : R₂ = 2) (h₃ : R₃ = 4) :
  ∃ (ε : ℝ), abs (R₁ + (R₂ * R₃) / (R₂ + R₃) - R₁ + 0.67) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_resistance_change_l579_57949


namespace NUMINAMATH_CALUDE_quadcycle_count_l579_57937

theorem quadcycle_count (b t q : ℕ) : 
  b + t + q = 10 →
  2*b + 3*t + 4*q = 29 →
  q = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadcycle_count_l579_57937


namespace NUMINAMATH_CALUDE_parallel_lines_not_always_equal_l579_57973

-- Define a line in a plane
structure Line :=
  (extends_infinitely : Bool)
  (can_be_measured : Bool)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.extends_infinitely ∧ l2.extends_infinitely ∧ ¬l1.can_be_measured ∧ ¬l2.can_be_measured

-- Theorem: Two parallel lines are not always equal
theorem parallel_lines_not_always_equal :
  ∃ l1 l2 : Line, parallel l1 l2 ∧ l1 ≠ l2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_not_always_equal_l579_57973


namespace NUMINAMATH_CALUDE_tens_digit_of_2020_pow_2021_minus_2022_l579_57925

theorem tens_digit_of_2020_pow_2021_minus_2022 : ∃ n : ℕ, 
  (2020^2021 - 2022) % 100 = 70 + n ∧ n < 10 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2020_pow_2021_minus_2022_l579_57925


namespace NUMINAMATH_CALUDE_prime_even_intersection_l579_57919

def isPrime (n : ℕ) : Prop := sorry

def isEven (n : ℕ) : Prop := sorry

def P : Set ℕ := {n | isPrime n}
def Q : Set ℕ := {n | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l579_57919


namespace NUMINAMATH_CALUDE_divisor_problem_l579_57929

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 199 →
  quotient = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l579_57929


namespace NUMINAMATH_CALUDE_quadratic_x_axis_intersection_l579_57984

theorem quadratic_x_axis_intersection (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_x_axis_intersection_l579_57984


namespace NUMINAMATH_CALUDE_sales_volume_estimate_l579_57991

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -5 * x + 150

-- Define the theorem
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation 10 - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_sales_volume_estimate_l579_57991


namespace NUMINAMATH_CALUDE_bd_length_is_six_l579_57947

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem bd_length_is_six (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 6 →
  length ABCD.B ABCD.C = 11 →
  length ABCD.C ABCD.D = 6 →
  length ABCD.D ABCD.A = 8 →
  ∃ n : ℕ, length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 6 := by
  sorry

end NUMINAMATH_CALUDE_bd_length_is_six_l579_57947


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l579_57943

theorem product_mod_seventeen : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l579_57943


namespace NUMINAMATH_CALUDE_room_area_from_carpet_l579_57955

/-- Given a rectangular carpet covering 30% of a room's floor area, 
    if the carpet measures 4 feet by 9 feet, 
    then the total floor area of the room is 120 square feet. -/
theorem room_area_from_carpet (carpet_length carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) (total_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_coverage_percent = 30 →
  carpet_length * carpet_width / total_area = carpet_coverage_percent / 100 →
  total_area = 120 :=
by sorry

end NUMINAMATH_CALUDE_room_area_from_carpet_l579_57955


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l579_57906

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l579_57906


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l579_57902

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_line_is_correct :
  let (x₀, y₀) := point
  (∀ x, tangent_line x (f x)) ∧
  (tangent_line x₀ y₀) ∧
  (∀ x, x ≠ x₀ → ¬(tangent_line x (f x) ∧ tangent_line x₀ y₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l579_57902


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l579_57921

/-- Given plane vectors a and b satisfying certain conditions, prove that the magnitude of their sum is √21. -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  ‖a‖ = 2 → 
  ‖b‖ = 3 → 
  a - b = (Real.sqrt 2, Real.sqrt 3) →
  ‖a + b‖ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l579_57921


namespace NUMINAMATH_CALUDE_weight_relationships_l579_57939

/-- Given the weights of Brenda, Mel, and Tom, prove their relationships and specific weights. -/
theorem weight_relationships (B M T : ℕ) : 
  B = 3 * M + 10 →  -- Brenda weighs 10 pounds more than 3 times Mel's weight
  T = 2 * M →       -- Tom weighs twice as much as Mel
  2 * T = B →       -- Tom weighs half as much as Brenda
  B = 220 →         -- Brenda weighs 220 pounds
  M = 70 ∧ T = 140  -- Prove that Mel weighs 70 pounds and Tom weighs 140 pounds
:= by sorry

end NUMINAMATH_CALUDE_weight_relationships_l579_57939


namespace NUMINAMATH_CALUDE_kim_coffee_time_l579_57923

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  payroll_update_time_per_employee : ℕ
  number_of_employees : ℕ
  total_time : ℕ

/-- Theorem stating that Kim spends 5 minutes making coffee -/
theorem kim_coffee_time (routine : MorningRoutine)
  (h1 : routine.status_update_time_per_employee = 2)
  (h2 : routine.payroll_update_time_per_employee = 3)
  (h3 : routine.number_of_employees = 9)
  (h4 : routine.total_time = 50)
  (h5 : routine.total_time = routine.coffee_time +
    routine.number_of_employees * (routine.status_update_time_per_employee +
    routine.payroll_update_time_per_employee)) :
  routine.coffee_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_kim_coffee_time_l579_57923


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_point_symmetric_line_wrt_line_l579_57964

-- Define the original line l
def l (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line to be reflected
def line_to_reflect (x y : ℝ) : Prop := x - y - 2 = 0

-- Statement for the first part of the problem
theorem symmetric_line_wrt_point :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), l x' y' → 
      (x + x') / 2 = M.1 ∧ (y + y') / 2 = M.2) →
    y = a * x + b ↔ y = 2 * x - 9 :=
sorry

-- Statement for the second part of the problem
theorem symmetric_line_wrt_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), line_to_reflect x' y' → 
      ∃ (x'' y'' : ℝ), l ((x + x'') / 2) ((y + y'') / 2) ∧
      (y'' - y) / (x'' - x) = -1 / (2 : ℝ)) →
    a * x + b * y + c = 0 ↔ 7 * x - y + 16 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_point_symmetric_line_wrt_line_l579_57964


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_three_l579_57982

/-- A quadratic function f(x) = x^2 - 2ax + a - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 3

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -1/4),
    then a ≤ -3 -/
theorem monotone_decreasing_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → x < -1/4 → f a x > f a y) → a ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_neg_three_l579_57982


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l579_57966

theorem smallest_digit_for_divisibility_by_nine :
  ∀ d : Nat, d ≤ 9 →
    (526000 + d * 1000 + 45) % 9 = 0 →
    d ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l579_57966


namespace NUMINAMATH_CALUDE_total_cost_calculation_l579_57915

/-- Calculates the total cost of a medical visit given the insurance coverage percentage and out-of-pocket cost -/
theorem total_cost_calculation (insurance_coverage_percent : ℝ) (out_of_pocket_cost : ℝ) : 
  insurance_coverage_percent = 80 → 
  out_of_pocket_cost = 60 → 
  (100 - insurance_coverage_percent) / 100 * (out_of_pocket_cost / ((100 - insurance_coverage_percent) / 100)) = 300 := by
sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l579_57915


namespace NUMINAMATH_CALUDE_system_solutions_l579_57911

theorem system_solutions (x y z : ℚ) : 
  ((x + 1) * (3 - 4 * y) = (6 * x + 1) * (3 - 2 * y) ∧
   (4 * x - 1) * (z + 1) = (x + 1) * (z - 1) ∧
   (3 - y) * (z - 2) = (1 - 3 * y) * (z - 6)) ↔
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   (x = 10/19 ∧ y = 25/7 ∧ z = 25/4)) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l579_57911


namespace NUMINAMATH_CALUDE_least_n_for_adjacent_probability_l579_57908

def adjacent_probability (n : ℕ) : ℚ :=
  (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1))

theorem least_n_for_adjacent_probability : 
  (∀ k < 90, adjacent_probability k ≥ 1 / 2015) ∧ 
  adjacent_probability 90 < 1 / 2015 :=
sorry

end NUMINAMATH_CALUDE_least_n_for_adjacent_probability_l579_57908


namespace NUMINAMATH_CALUDE_roots_sequence_property_l579_57988

/-- Given x₁ and x₂ are roots of x² - 6x + 1 = 0, prove that for all natural numbers n,
    aₙ = x₁ⁿ + x₂ⁿ is an integer and not a multiple of 5. -/
theorem roots_sequence_property (x₁ x₂ : ℝ) (h : x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, (x₁^n + x₂^n = k) ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_roots_sequence_property_l579_57988


namespace NUMINAMATH_CALUDE_complement_intersection_eq_set_l579_57926

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_eq_set : (Aᶜ ∩ Bᶜ) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_set_l579_57926


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l579_57916

/-- Represents the speed of a swimmer in various conditions -/
structure SwimmerSpeed where
  downstream : ℝ
  upstream : ℝ
  stillWater : ℝ

/-- Theorem stating that given the downstream and upstream speeds, 
    we can determine the speed in still water -/
theorem swimmer_speed_in_still_water 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_distance : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 72) 
  (h2 : downstream_time = 4) 
  (h3 : upstream_distance = 36) 
  (h4 : upstream_time = 6) :
  ∃ (s : SwimmerSpeed), 
    s.downstream = downstream_distance / downstream_time ∧
    s.upstream = upstream_distance / upstream_time ∧
    s.stillWater = 12 := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l579_57916


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l579_57970

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

/-- The y-intercept of the parabola -/
def d : ℝ := parabola 0

/-- The x-intercepts of the parabola -/
noncomputable def e : ℝ := (9 + Real.sqrt 33) / 6
noncomputable def f : ℝ := (9 - Real.sqrt 33) / 6

/-- Theorem stating that the sum of the y-intercept and x-intercepts is 7 -/
theorem parabola_intercepts_sum : d + e + f = 7 := by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l579_57970


namespace NUMINAMATH_CALUDE_squared_sum_product_l579_57905

theorem squared_sum_product (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) :
  a^2 * b + a * b^2 = 108 := by sorry

end NUMINAMATH_CALUDE_squared_sum_product_l579_57905


namespace NUMINAMATH_CALUDE_donation_proof_l579_57917

/-- The amount donated to Animal Preservation Park -/
def animal_park_donation : ℝ := sorry

/-- The amount donated to Treetown National Park and The Forest Reserve combined -/
def combined_donation : ℝ := animal_park_donation + 140

/-- The total donation to all three parks -/
def total_donation : ℝ := 1000

theorem donation_proof : combined_donation = 570 := by
  sorry

end NUMINAMATH_CALUDE_donation_proof_l579_57917


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l579_57994

/-- The function f(x) = a^(x-2016) + 1 has a fixed point at (2016, 2) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = a^(x - 2016) + 1) ∧ f 2016 = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l579_57994


namespace NUMINAMATH_CALUDE_dice_surface_area_l579_57944

/-- The surface area of a cube with edge length 20 centimeters is 2400 square centimeters. -/
theorem dice_surface_area :
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length ^ 2
  surface_area = 2400 := by sorry

end NUMINAMATH_CALUDE_dice_surface_area_l579_57944


namespace NUMINAMATH_CALUDE_coloring_books_shelves_l579_57946

/-- Calculates the number of shelves needed to display remaining coloring books --/
def shelves_needed (initial_stock : ℕ) (sold : ℕ) (donated : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((initial_stock - sold - donated) + books_per_shelf - 1) / books_per_shelf

/-- Theorem stating that given the problem conditions, 6 shelves are needed --/
theorem coloring_books_shelves :
  shelves_needed 150 55 30 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_shelves_l579_57946


namespace NUMINAMATH_CALUDE_pyramid_volume_with_conditions_l579_57930

/-- The volume of a right pyramid with a hexagonal base -/
noncomputable def pyramidVolume (totalSurfaceArea : ℝ) (triangularFaceRatio : ℝ) : ℝ :=
  let hexagonalBaseArea := totalSurfaceArea / 3
  let sideLength := Real.sqrt (320 / (3 * Real.sqrt 3))
  let triangularHeight := 160 / sideLength
  let pyramidHeight := Real.sqrt (triangularHeight^2 - (sideLength / 2)^2)
  (1 / 3) * hexagonalBaseArea * pyramidHeight

/-- Theorem: The volume of the pyramid with given conditions -/
theorem pyramid_volume_with_conditions :
  ∃ (V : ℝ), pyramidVolume 720 (1/3) = V :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_with_conditions_l579_57930


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l579_57901

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  other_angle : ℝ

/-- A function that attempts to construct an isosceles triangle from given angles -/
noncomputable def construct_isosceles_triangle (ba oa : ℝ) : Option IsoscelesTriangle := sorry

/-- Theorem stating that an isosceles triangle is not uniquely determined by a base angle and another angle -/
theorem isosceles_triangle_not_unique :
  ∃ (ba₁ oa₁ ba₂ oa₂ : ℝ),
    ba₁ = ba₂ ∧
    oa₁ = oa₂ ∧
    construct_isosceles_triangle ba₁ oa₁ ≠ construct_isosceles_triangle ba₂ oa₂ :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l579_57901


namespace NUMINAMATH_CALUDE_margies_driving_distance_l579_57938

/-- Proves that Margie can drive 400 miles with $50 worth of gas -/
theorem margies_driving_distance 
  (car_efficiency : ℝ) 
  (gas_price : ℝ) 
  (gas_budget : ℝ) 
  (h1 : car_efficiency = 40) 
  (h2 : gas_price = 5) 
  (h3 : gas_budget = 50) : 
  (gas_budget / gas_price) * car_efficiency = 400 := by
sorry

end NUMINAMATH_CALUDE_margies_driving_distance_l579_57938


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l579_57927

/-- Calculates the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of digits of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Converts a positive integer to its binary representation -/
def to_binary (n : ℕ) : List ℕ := sorry

/-- Calculates the cost of transmitting a number using Option 1 -/
def cost_option1 (n : ℕ) : ℕ :=
  sum_of_digits n + num_digits n

/-- Calculates the cost of transmitting a number using Option 2 -/
def cost_option2 (n : ℕ) : ℕ :=
  let binary := to_binary n
  (binary.filter (· = 1)).length + (binary.filter (· = 0)).length + binary.length

/-- Checks if the costs are equal for both options -/
def costs_equal (n : ℕ) : Prop :=
  cost_option1 n = cost_option2 n

theorem largest_equal_cost_number :
  ∀ n : ℕ, n < 2000 → n > 1539 → ¬(costs_equal n) := by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l579_57927


namespace NUMINAMATH_CALUDE_fraction_equality_l579_57924

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (2 * x + 8 * y) / (8 * x - 2 * y) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l579_57924


namespace NUMINAMATH_CALUDE_apple_picking_contest_l579_57992

/-- The number of apples picked by Marin -/
def marin_apples : ℕ := 9

/-- The number of apples picked by Donald -/
def donald_apples : ℕ := 11

/-- The number of apples picked by Ana -/
def ana_apples : ℕ := 2 * (marin_apples + donald_apples)

/-- The total number of apples picked by all three participants -/
def total_apples : ℕ := marin_apples + donald_apples + ana_apples

theorem apple_picking_contest :
  total_apples = 60 := by sorry

end NUMINAMATH_CALUDE_apple_picking_contest_l579_57992


namespace NUMINAMATH_CALUDE_flower_producing_plants_l579_57990

theorem flower_producing_plants 
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate flower_production_rate : ℚ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 3/5)
  (h4 : sunflower_germination_rate = 4/5)
  (h5 : flower_production_rate = 4/5) :
  ⌊(daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate⌋ = 28 :=
by sorry

end NUMINAMATH_CALUDE_flower_producing_plants_l579_57990


namespace NUMINAMATH_CALUDE_max_value_rational_function_l579_57972

theorem max_value_rational_function : 
  ∃ (n : ℕ), n = 97 ∧ 
  (∀ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) ≤ n) ∧
  (∀ m : ℕ, m > n → ∃ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) < m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l579_57972


namespace NUMINAMATH_CALUDE_cereal_eating_time_l579_57942

def fat_rate : ℚ := 1 / 25
def thin_rate : ℚ := 1 / 35
def medium_rate : ℚ := 1 / 28
def total_cereal : ℚ := 5

def combined_rate : ℚ := fat_rate + thin_rate + medium_rate

def time_taken : ℚ := total_cereal / combined_rate

theorem cereal_eating_time : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ time_taken - 48 < ε ∧ 48 - time_taken < ε :=
sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l579_57942


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l579_57956

theorem painted_cube_theorem (n : ℕ) (h1 : n > 4) :
  (2 * (n - 2) = n^2 - 2*n + 1) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l579_57956


namespace NUMINAMATH_CALUDE_voting_stabilizes_l579_57931

/-- Represents the state of votes in a circular arrangement -/
def VoteState := Vector Bool 25

/-- Represents the next state of votes based on the current state -/
def nextState (current : VoteState) : VoteState :=
  Vector.ofFn (fun i =>
    let prev := current.get ((i - 1 + 25) % 25)
    let next := current.get ((i + 1) % 25)
    let curr := current.get i
    if prev = next then curr else !curr)

/-- Theorem stating that the voting pattern will eventually stabilize -/
theorem voting_stabilizes : ∃ (n : ℕ), ∀ (initial : VoteState),
  ∃ (k : ℕ), k ≤ n ∧ nextState^[k] initial = nextState^[k+1] initial :=
sorry


end NUMINAMATH_CALUDE_voting_stabilizes_l579_57931


namespace NUMINAMATH_CALUDE_stacys_farm_chickens_l579_57996

theorem stacys_farm_chickens :
  ∀ (total_animals sick_animals piglets goats : ℕ),
    piglets = 40 →
    goats = 34 →
    sick_animals = 50 →
    2 * sick_animals = total_animals →
    total_animals = piglets + goats + (total_animals - piglets - goats) →
    total_animals - piglets - goats = 26 := by
  sorry

end NUMINAMATH_CALUDE_stacys_farm_chickens_l579_57996


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l579_57951

theorem quadratic_equation_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 9 = (x + 3)^2) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l579_57951


namespace NUMINAMATH_CALUDE_father_son_age_relation_l579_57976

/-- Proves the number of years it takes for a father to be twice as old as his son -/
theorem father_son_age_relation (father_age : ℕ) (son_age : ℕ) (years : ℕ) : 
  father_age = 45 →
  father_age = 3 * son_age →
  father_age + years = 2 * (son_age + years) →
  years = 15 := by
sorry

end NUMINAMATH_CALUDE_father_son_age_relation_l579_57976


namespace NUMINAMATH_CALUDE_valid_numbers_l579_57997

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  x + y + z = (10 * x + y) - (10 * y + z)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {209, 428, 647, 866, 214, 433, 652, 871} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l579_57997


namespace NUMINAMATH_CALUDE_alpha_more_advantageous_regular_l579_57959

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the number of visits in a year for regular attendance -/
def regularAttendanceVisits : ℕ := 96

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ := club.monthlyFee * 12

/-- Calculates the cost per visit for regular attendance -/
def costPerVisitRegular (club : FitnessClub) : ℚ :=
  (yearlyCost club : ℚ) / regularAttendanceVisits

/-- Alpha and Beta fitness clubs -/
def alpha : FitnessClub := ⟨"Alpha", 999⟩
def beta : FitnessClub := ⟨"Beta", 1299⟩

/-- Theorem stating that Alpha is more advantageous for regular attendance -/
theorem alpha_more_advantageous_regular : 
  costPerVisitRegular alpha < costPerVisitRegular beta := by
  sorry

end NUMINAMATH_CALUDE_alpha_more_advantageous_regular_l579_57959


namespace NUMINAMATH_CALUDE_class_division_transfer_l579_57987

/-- 
Given a class divided into two groups with 26 and 22 people respectively,
prove that the number of people transferred (x) from the second group to the first
satisfies the equation x + 26 = 3(22 - x) when the first group becomes
three times the size of the second group after the transfer.
-/
theorem class_division_transfer (x : ℤ) : x + 26 = 3 * (22 - x) ↔ 
  (26 + x = 3 * (22 - x) ∧ 
   26 + x > 0 ∧
   22 - x > 0) := by
  sorry

#check class_division_transfer

end NUMINAMATH_CALUDE_class_division_transfer_l579_57987


namespace NUMINAMATH_CALUDE_muffin_price_theorem_l579_57936

/-- Promotional sale: Buy three muffins at regular price, get fourth muffin free -/
def promotional_sale (regular_price : ℝ) : ℝ := 3 * regular_price

/-- The total amount John paid for four muffins -/
def total_paid : ℝ := 15

theorem muffin_price_theorem :
  ∃ (regular_price : ℝ), promotional_sale regular_price = total_paid ∧ regular_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_theorem_l579_57936


namespace NUMINAMATH_CALUDE_horner_v3_value_l579_57907

/-- Horner's Method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁶ - 5x⁵ + 6x⁴ + x² + 0.3x + 2 -/
def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, -5, 6, 0, 1, 0.3, 2]

/-- Theorem: v₃ = -40 when evaluating f(-2) using Horner's Method -/
theorem horner_v3_value :
  let x := -2
  let v₀ := 1
  let v₁ := v₀ * x + f_coeffs[1]!
  let v₂ := v₁ * x + f_coeffs[2]!
  let v₃ := v₂ * x + f_coeffs[3]!
  v₃ = -40 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l579_57907


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l579_57969

/-- Given a geometric sequence where a₁ = 3 and a₃ = 1/9, prove that a₆ = 1/81 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℚ) (h1 : a 1 = 3) (h3 : a 3 = 1/9) :
  a 6 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l579_57969


namespace NUMINAMATH_CALUDE_complex_distance_to_origin_l579_57971

theorem complex_distance_to_origin : 
  let z : ℂ := (I^2016 - 2*I^2014) / (2 - I)^2
  Complex.abs z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_to_origin_l579_57971


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l579_57958

theorem angle_terminal_side_value (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  (∃ (x y : ℝ), x = -4 * m ∧ y = 3 * m ∧ 
    x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
    y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l579_57958


namespace NUMINAMATH_CALUDE_recurrence_implies_general_formula_l579_57910

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (n * a n - 2 * a (n + 1)) / a (n + 1) = n

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 / (n * (n + 1))

theorem recurrence_implies_general_formula (a : ℕ → ℝ) :
  RecurrenceSequence a → GeneralFormula a := by
  sorry

end NUMINAMATH_CALUDE_recurrence_implies_general_formula_l579_57910


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l579_57948

theorem complex_fraction_calculation : 
  let expr1 := (5 / 8 * 3 / 7 + 1 / 4 * 2 / 6) - (2 / 3 * 1 / 4 - 1 / 5 * 4 / 9)
  let expr2 := 7 / 9 * 2 / 5 * 1 / 2 * 5040 + 1 / 3 * 3 / 8 * 9 / 11 * 4230
  (expr1 * expr2 : ℚ) = 336 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l579_57948


namespace NUMINAMATH_CALUDE_retailer_profit_percent_l579_57985

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is approximately 18.58%. -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 225 28 300
  (result > 18.58 - ε) ∧ (result < 18.58 + ε) :=
sorry

end NUMINAMATH_CALUDE_retailer_profit_percent_l579_57985


namespace NUMINAMATH_CALUDE_number_puzzle_l579_57920

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 15) = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l579_57920


namespace NUMINAMATH_CALUDE_monthly_income_p_l579_57999

def average_income (x y : ℕ) := (x + y) / 2

theorem monthly_income_p (p q r : ℕ) 
  (h1 : average_income p q = 5050)
  (h2 : average_income q r = 6250)
  (h3 : average_income p r = 5200) :
  p = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_p_l579_57999


namespace NUMINAMATH_CALUDE_mall_price_change_loss_l579_57912

theorem mall_price_change_loss : ∀ (a b : ℝ),
  a * (1.2 : ℝ)^2 = 23.04 →
  b * (0.8 : ℝ)^2 = 23.04 →
  (a + b) - 2 * 23.04 = 5.92 := by
sorry

end NUMINAMATH_CALUDE_mall_price_change_loss_l579_57912


namespace NUMINAMATH_CALUDE_machine_doesnt_require_repair_l579_57989

/-- Represents a weighing machine --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  all_deviations_bounded : Prop
  standard_deviation_bounded : Prop

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  ¬m.all_deviations_bounded ∨
  ¬m.standard_deviation_bounded

/-- Theorem stating that the machine does not require repair --/
theorem machine_doesnt_require_repair (m : WeighingMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.all_deviations_bounded)
  (h4 : m.standard_deviation_bounded) :
  ¬(requires_repair m) :=
sorry

end NUMINAMATH_CALUDE_machine_doesnt_require_repair_l579_57989


namespace NUMINAMATH_CALUDE_present_worth_calculation_present_worth_approximation_l579_57961

/-- Calculates the present worth of an investment given specific interest rates and banker's gain --/
theorem present_worth_calculation (banker_gain : ℝ) : ∃ P : ℝ,
  P * (1.05 * 1.1025 * 1.1255 - 1) = banker_gain :=
by
  sorry

/-- Verifies that the calculated present worth is approximately 114.94 --/
theorem present_worth_approximation (P : ℝ) 
  (h : P * (1.05 * 1.1025 * 1.1255 - 1) = 36) : 
  114.9 < P ∧ P < 115 :=
by
  sorry

end NUMINAMATH_CALUDE_present_worth_calculation_present_worth_approximation_l579_57961


namespace NUMINAMATH_CALUDE_mowing_time_calculation_l579_57977

/-- Calculates the time required to mow a rectangular lawn -/
def mowing_time (length width swath overlap speed : ℚ) : ℚ :=
  let effective_swath := (swath - overlap) / 12
  let strips := width / effective_swath
  let total_distance := strips * length
  total_distance / speed

/-- Theorem stating the time required to mow the lawn under given conditions -/
theorem mowing_time_calculation :
  mowing_time 100 180 (30/12) (6/12) 4000 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_mowing_time_calculation_l579_57977


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l579_57904

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of point symmetry with respect to a line --/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line_of_symmetry x₀ y₀ ∧
    (x₀ - x₁ = x₂ - x₀) ∧ (y₀ - y₁ = y₂ - y₀)

/-- Theorem: The point (2, -2) is symmetric to (-1, 1) with respect to the line x-y-1=0 --/
theorem symmetric_point_theorem : symmetric_points (-1) 1 2 (-2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l579_57904


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l579_57968

/-- Given a line segment with midpoint (3, -1) and one endpoint (7, 3), 
    prove that the other endpoint is (-1, -5). -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -1))
  (h_endpoint1 : endpoint1 = (7, 3)) :
  ∃ endpoint2 : ℝ × ℝ, 
    endpoint2 = (-1, -5) ∧ 
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l579_57968


namespace NUMINAMATH_CALUDE_log_sum_equality_l579_57953

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + 8^(2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l579_57953


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l579_57980

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 3 + 145 / 999 ∧ x = 3142 / 999 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l579_57980


namespace NUMINAMATH_CALUDE_product_closest_to_315_l579_57962

def product : ℝ := 3.57 * 9.052 * (6.18 + 3.821)

def options : List ℝ := [200, 300, 315, 400, 500]

theorem product_closest_to_315 :
  ∀ x ∈ options, |product - 315| ≤ |product - x| :=
sorry

end NUMINAMATH_CALUDE_product_closest_to_315_l579_57962


namespace NUMINAMATH_CALUDE_area_of_triangle_PMF_l579_57928

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ
  focus : ℝ × ℝ

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The foot of the perpendicular from a point to the directrix -/
def footOfPerpendicular (p : Parabola) (point : PointOnParabola p) : ℝ × ℝ :=
  (p.directrix, point.y)

/-- The theorem stating the area of the triangle PMF -/
theorem area_of_triangle_PMF (p : Parabola) (P : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  p.directrix = -1 →
  p.focus = (1, 0) →
  (P.x - p.directrix)^2 + P.y^2 = 5^2 →
  let M := footOfPerpendicular p P
  let F := p.focus
  let area := (1/2) * |P.y| * 5
  area = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PMF_l579_57928


namespace NUMINAMATH_CALUDE_circle_chords_count_l579_57952

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of points needed to form a chord -/
def points_per_chord : ℕ := 2

theorem circle_chords_count :
  choose num_points points_per_chord = 45 := by sorry

end NUMINAMATH_CALUDE_circle_chords_count_l579_57952


namespace NUMINAMATH_CALUDE_prop_equivalence_l579_57983

theorem prop_equivalence (p q : Prop) : (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_prop_equivalence_l579_57983


namespace NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l579_57957

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (α β γ : Plane) (m n : Line) :
  α ≠ β → α ≠ γ → β ≠ γ →  -- Three different planes
  intersect α γ = m →      -- α ∩ γ = m
  intersect β γ = n →      -- β ∩ γ = n
  parallel_planes α β →    -- If α ∥ β
  parallel_lines m n :=    -- Then m ∥ n
by sorry

end NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l579_57957


namespace NUMINAMATH_CALUDE_registration_methods_count_l579_57913

/-- The number of subjects available for registration -/
def num_subjects : ℕ := 4

/-- The number of students registering -/
def num_students : ℕ := 3

/-- The number of different registration methods -/
def registration_methods : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of registration methods is 64 -/
theorem registration_methods_count : registration_methods = 64 := by sorry

end NUMINAMATH_CALUDE_registration_methods_count_l579_57913


namespace NUMINAMATH_CALUDE_chocolate_bar_ratio_l579_57940

theorem chocolate_bar_ratio (total pieces : ℕ) (michael paige mandy : ℕ) : 
  total = 60 →
  paige = (total - michael) / 2 →
  mandy = 15 →
  total = michael + paige + mandy →
  michael / total = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_ratio_l579_57940


namespace NUMINAMATH_CALUDE_fully_filled_boxes_l579_57974

theorem fully_filled_boxes (total_cards : ℕ) (max_per_box : ℕ) (h1 : total_cards = 94) (h2 : max_per_box = 8) :
  (total_cards / max_per_box : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_fully_filled_boxes_l579_57974


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l579_57950

theorem stock_price_uniqueness (n : Nat) (k l : Nat) (h_n : 0 < n ∧ n < 100) :
  (1 + n / 100 : ℚ) ^ k * (1 - n / 100 : ℚ) ^ l ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l579_57950


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l579_57935

theorem remainder_sum_powers_mod_5 :
  (Nat.pow 9 7 + Nat.pow 4 5 + Nat.pow 3 9) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l579_57935


namespace NUMINAMATH_CALUDE_canteen_distance_l579_57903

theorem canteen_distance (a b c : ℝ) (h1 : a = 360) (h2 : c = 800) (h3 : a^2 + b^2 = c^2) :
  b / 2 = 438.6 := by sorry

end NUMINAMATH_CALUDE_canteen_distance_l579_57903


namespace NUMINAMATH_CALUDE_slope_of_line_parallel_lines_coefficient_l579_57993

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a / b) * x - c / b :=
sorry

theorem parallel_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = 0) → a = -6 :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_parallel_lines_coefficient_l579_57993


namespace NUMINAMATH_CALUDE_only_KC2H3O2_turns_pink_l579_57986

-- Define the set of solutes
inductive Solute
| NaCl
| KC2H3O2
| LiBr
| NH4NO3

-- Define a function to determine if a solution is basic
def isBasic (s : Solute) : Prop :=
  match s with
  | Solute.KC2H3O2 => True
  | _ => False

-- Define a function to check if phenolphthalein turns pink
def turnsPink (s : Solute) : Prop := isBasic s

-- Theorem statement
theorem only_KC2H3O2_turns_pink :
  ∀ s : Solute, turnsPink s ↔ s = Solute.KC2H3O2 :=
by sorry

end NUMINAMATH_CALUDE_only_KC2H3O2_turns_pink_l579_57986


namespace NUMINAMATH_CALUDE_complex_equation_solution_l579_57922

theorem complex_equation_solution :
  ∃ (z : ℂ), 3 - 2 * Complex.I * z = 5 + 3 * Complex.I * z ∧ z = (2 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l579_57922


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l579_57965

/-- The perimeter of an equilateral triangle with side length 5 cm is 15 cm. -/
theorem equilateral_triangle_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 5 →
  perimeter = 3 * side_length →
  perimeter = 15 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l579_57965


namespace NUMINAMATH_CALUDE_factorization_theorem_l579_57941

/-- The polynomial to be factored -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 9 - 64*x^4

/-- The first factor of the factorization -/
def f1 (x : ℝ) : ℝ := -8*x^2 + x + 3

/-- The second factor of the factorization -/
def f2 (x : ℝ) : ℝ := 8*x^2 + x + 3

/-- Theorem stating that p(x) is equal to the product of f1(x) and f2(x) for all real x -/
theorem factorization_theorem : ∀ x : ℝ, p x = f1 x * f2 x := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l579_57941


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l579_57960

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = (3 + Real.sqrt 14) / 2 + Complex.I * Real.sqrt 14 / 7 ∧
  z₂ = (3 - Real.sqrt 14) / 2 - Complex.I * Real.sqrt 14 / 7 ∧
  z₁^2 - 3*z₁ + 2 = 3 - 2*Complex.I ∧
  z₂^2 - 3*z₂ + 2 = 3 - 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l579_57960


namespace NUMINAMATH_CALUDE_both_p_and_q_false_l579_57933

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := (∀ a b : ℝ, a*b > 4 → (a > 2 ∧ b > 2)) ∧ 
                ¬(∀ a b : ℝ, (a > 2 ∧ b > 2) → a*b > 4)

-- Theorem stating that both p and q are false
theorem both_p_and_q_false : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_both_p_and_q_false_l579_57933


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l579_57967

def saturday_sales : ℕ := 60
def total_sales : ℕ := 150

def sunday_sales : ℕ := total_sales - saturday_sales

def percentage_increase : ℚ := (sunday_sales - saturday_sales : ℚ) / saturday_sales * 100

theorem sales_increase_percentage :
  sunday_sales > saturday_sales →
  percentage_increase = 50 := by
  sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l579_57967


namespace NUMINAMATH_CALUDE_x_value_l579_57932

theorem x_value (x y : ℝ) 
  (h1 : x - y = 8)
  (h2 : x + y = 16)
  (h3 : x * y = 48) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l579_57932


namespace NUMINAMATH_CALUDE_pencils_per_box_l579_57963

theorem pencils_per_box (num_boxes : ℕ) (pencils_given : ℕ) (pencils_left : ℕ) :
  num_boxes = 2 ∧ pencils_given = 15 ∧ pencils_left = 9 →
  (pencils_given + pencils_left) / num_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_box_l579_57963


namespace NUMINAMATH_CALUDE_john_hired_twenty_lessons_l579_57954

/-- Given the cost of a piano, the original price of a lesson, the discount percentage,
    and the total cost, calculate the number of lessons hired. -/
def number_of_lessons (piano_cost lesson_price discount_percent total_cost : ℚ) : ℚ :=
  let discounted_price := lesson_price * (1 - discount_percent / 100)
  let lesson_cost := total_cost - piano_cost
  lesson_cost / discounted_price

/-- Prove that given the specified costs and discount, John hired 20 lessons. -/
theorem john_hired_twenty_lessons :
  number_of_lessons 500 40 25 1100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_hired_twenty_lessons_l579_57954


namespace NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l579_57995

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := sorry

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := sorry

/-- Calculates the total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  shellsInJarA 0 = 50 →
  shellsInJarB 0 = 30 →
  (∀ k : ℕ, shellsInJarA (k + 1) = shellsInJarA k + 20) →
  (∀ k : ℕ, shellsInJarB (k + 1) = shellsInJarB k * 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarA (k + 1) = shellsInJarA k / 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarB (k + 1) = shellsInJarB k / 2) →
  totalShells 6 = 97 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l579_57995


namespace NUMINAMATH_CALUDE_largest_five_digit_distinct_odd_number_l579_57998

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_are_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / 10^i) % 10 ≠ (n / 10^j) % 10

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → is_odd_digit ((n / 10^i) % 10)

theorem largest_five_digit_distinct_odd_number :
  ∀ n : ℕ, is_five_digit_number n → digits_are_distinct n → all_digits_odd n →
    n ≤ 97531 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_distinct_odd_number_l579_57998


namespace NUMINAMATH_CALUDE_library_theorem_l579_57909

def library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ) : ℕ :=
  let books_remaining := total_books - 
    (day1_students + day2_students + day3_students) * books_per_student
  books_remaining / books_per_student

theorem library_theorem : 
  library_problem 120 5 4 5 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_library_theorem_l579_57909


namespace NUMINAMATH_CALUDE_inequality_solution_l579_57945

theorem inequality_solution (y : ℝ) : 
  (7/30 : ℝ) + |y - 3/10| < 11/30 ↔ 1/6 < y ∧ y < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l579_57945


namespace NUMINAMATH_CALUDE_anna_chargers_l579_57981

theorem anna_chargers (phone_chargers : ℕ) (laptop_chargers : ℕ) : 
  laptop_chargers = 5 * phone_chargers →
  phone_chargers + laptop_chargers = 24 →
  phone_chargers = 4 := by
sorry

end NUMINAMATH_CALUDE_anna_chargers_l579_57981


namespace NUMINAMATH_CALUDE_sammy_remaining_problems_l579_57978

theorem sammy_remaining_problems (total : ℕ) (fractions decimals multiplication division : ℕ)
  (completed_fractions completed_decimals completed_multiplication completed_division : ℕ)
  (h1 : total = 115)
  (h2 : fractions = 35)
  (h3 : decimals = 40)
  (h4 : multiplication = 20)
  (h5 : division = 20)
  (h6 : completed_fractions = 11)
  (h7 : completed_decimals = 17)
  (h8 : completed_multiplication = 9)
  (h9 : completed_division = 5)
  (h10 : total = fractions + decimals + multiplication + division) :
  total - (completed_fractions + completed_decimals + completed_multiplication + completed_division) = 73 := by
sorry

end NUMINAMATH_CALUDE_sammy_remaining_problems_l579_57978


namespace NUMINAMATH_CALUDE_intersection_equals_B_implies_a_is_one_l579_57918

def A : Set ℝ := {-1, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1}

theorem intersection_equals_B_implies_a_is_one (a : ℝ) : A ∩ B a = B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_B_implies_a_is_one_l579_57918
