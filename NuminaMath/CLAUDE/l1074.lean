import Mathlib

namespace NUMINAMATH_CALUDE_catman_do_whiskers_l1074_107422

theorem catman_do_whiskers (princess_puff_whiskers : ℕ) (catman_do_whiskers : ℕ) : 
  princess_puff_whiskers = 14 →
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 →
  catman_do_whiskers = 22 := by
  sorry

end NUMINAMATH_CALUDE_catman_do_whiskers_l1074_107422


namespace NUMINAMATH_CALUDE_pi_sqrt3_minus_cos30_squared_l1074_107498

theorem pi_sqrt3_minus_cos30_squared :
  (π / (Real.sqrt 3 - 1)) ^ 0 - (Real.cos (π / 6)) ^ 2 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pi_sqrt3_minus_cos30_squared_l1074_107498


namespace NUMINAMATH_CALUDE_duty_arrangements_eq_180_l1074_107446

/-- The number of different duty arrangements for 3 staff members over 5 days -/
def duty_arrangements (num_staff : ℕ) (num_days : ℕ) (max_days_per_staff : ℕ) : ℕ :=
  -- Number of ways to choose the person working only one day
  num_staff *
  -- Number of ways to permute the duties
  (Nat.factorial num_days / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)) *
  -- Number of ways to assign the two-day duties to the remaining two staff members
  Nat.factorial 2

/-- Theorem stating that the number of duty arrangements for the given conditions is 180 -/
theorem duty_arrangements_eq_180 :
  duty_arrangements 3 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_eq_180_l1074_107446


namespace NUMINAMATH_CALUDE_star_four_six_l1074_107474

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem star_four_six : star 4 6 = 100 := by
  sorry

end NUMINAMATH_CALUDE_star_four_six_l1074_107474


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1074_107479

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 7).sum (λ k => Nat.choose 6 k * 2^(6 - k) * if k = 3 then 1 else 0) = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1074_107479


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1074_107409

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1074_107409


namespace NUMINAMATH_CALUDE_distance_X_to_CD_l1074_107441

/-- Square with side length 2s and quarter-circle arcs -/
structure SquareWithArcs (s : ℝ) :=
  (A B C D : ℝ × ℝ)
  (X : ℝ × ℝ)
  (h_square : A = (0, 0) ∧ B = (2*s, 0) ∧ C = (2*s, 2*s) ∧ D = (0, 2*s))
  (h_arc_A : (X.1 - A.1)^2 + (X.2 - A.2)^2 = (2*s)^2)
  (h_arc_B : (X.1 - B.1)^2 + (X.2 - B.2)^2 = (2*s)^2)
  (h_X_inside : 0 < X.1 ∧ X.1 < 2*s ∧ 0 < X.2 ∧ X.2 < 2*s)

/-- The distance from X to side CD in a SquareWithArcs is 2s(2 - √3) -/
theorem distance_X_to_CD (s : ℝ) (sq : SquareWithArcs s) :
  2*s - sq.X.2 = 2*s*(2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_distance_X_to_CD_l1074_107441


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l1074_107428

/-- Given a compound where 9 moles weigh 8100 grams, prove that its molecular weight is 900 grams/mole. -/
theorem molecular_weight_proof (compound : Type) 
  (moles : ℕ) (total_weight : ℝ) (molecular_weight : ℝ) 
  (h1 : moles = 9) 
  (h2 : total_weight = 8100) 
  (h3 : total_weight = moles * molecular_weight) : 
  molecular_weight = 900 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l1074_107428


namespace NUMINAMATH_CALUDE_special_function_properties_l1074_107476

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ a b : ℝ, f (a * b) = a * f b + b * f a) ∧
  f (1 / 2) = 1

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  f (1 / 4) = 1 ∧
  f (1 / 8) = 3 / 4 ∧
  f (1 / 16) = 1 / 2 ∧
  ∀ n : ℕ, n > 0 → f (2 ^ (-n : ℝ)) = n * (1 / 2) ^ (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l1074_107476


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l1074_107487

theorem units_digit_47_power_47 : (47^47) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l1074_107487


namespace NUMINAMATH_CALUDE_sequence_2023rd_term_l1074_107406

theorem sequence_2023rd_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a n / 2) - (1 / (2 * a (n + 1))) = a (n + 1) - (1 / a n)) :
  a 2023 = 1 ∨ a 2023 = (1 / 2) ^ 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2023rd_term_l1074_107406


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1074_107453

theorem complex_magnitude_equation (n : ℝ) (hn : n > 0) :
  Complex.abs (3 + n * Complex.I) = 3 * Real.sqrt 10 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1074_107453


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l1074_107486

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(53*b)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l1074_107486


namespace NUMINAMATH_CALUDE_intersection_line_of_two_circles_l1074_107403

/-- Given two circles with equations x^2 + y^2 + 4x - 4y - 1 = 0 and x^2 + y^2 + 2x - 13 = 0,
    the line passing through their intersection points has the equation x - 2y + 6 = 0 -/
theorem intersection_line_of_two_circles (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
  (x - 2*y + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_two_circles_l1074_107403


namespace NUMINAMATH_CALUDE_complex_product_real_l1074_107461

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + b * Complex.I
  (z₁ * z₂).im = 0 → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1074_107461


namespace NUMINAMATH_CALUDE_exactly_two_out_of_three_germinate_l1074_107434

def seed_germination_probability : ℚ := 3/5

def exactly_two_out_of_three_probability : ℚ :=
  3 * seed_germination_probability^2 * (1 - seed_germination_probability)

theorem exactly_two_out_of_three_germinate :
  exactly_two_out_of_three_probability = 54/125 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_three_germinate_l1074_107434


namespace NUMINAMATH_CALUDE_root_expression_value_l1074_107470

theorem root_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) :
  (a - 1)^2 + a*(a - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l1074_107470


namespace NUMINAMATH_CALUDE_expression_equality_l1074_107444

theorem expression_equality : 4 + 3/10 + 9/1000 = 4.309 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1074_107444


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1074_107454

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → 
  (b^2 + 8*b + 4 = 0) → 
  (a/b + b/a = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1074_107454


namespace NUMINAMATH_CALUDE_two_year_increase_l1074_107439

/-- 
Given an initial amount that increases by 1/8th of itself each year, 
this theorem proves that after two years, the amount will be as calculated.
-/
theorem two_year_increase (initial_amount : ℝ) : 
  initial_amount = 70400 → 
  (initial_amount * (9/8) * (9/8) : ℝ) = 89070 := by
  sorry

end NUMINAMATH_CALUDE_two_year_increase_l1074_107439


namespace NUMINAMATH_CALUDE_box_volume_increase_l1074_107440

/-- Theorem about the volume of a rectangular box after increasing dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4500)
  (surface_area : 2 * (l * w + l * h + w * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 216) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1074_107440


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l1074_107412

theorem max_value_of_trigonometric_expression :
  let y : ℝ → ℝ := λ x => Real.tan (x + 5 * Real.pi / 6) - Real.tan (x + Real.pi / 3) + Real.sin (x + Real.pi / 3)
  let max_value := (4 + Real.sqrt 3) / (2 * Real.sqrt 3)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x ≤ max_value ∧
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x₀ = max_value := by
sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l1074_107412


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l1074_107426

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let win_amount : ℚ := 4
  let lose_amount : ℚ := 9
  let expected_value := p_heads * win_amount - p_tails * lose_amount
  expected_value = -1/3 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l1074_107426


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l1074_107494

/-- 
Given a restaurant that:
- Made some hamburgers and 4 hot dogs
- Served 3 hamburgers
- Had 6 hamburgers left over

Prove that the initial number of hamburgers was 9.
-/
theorem restaurant_hamburgers (served : ℕ) (leftover : ℕ) : 
  served = 3 → leftover = 6 → served + leftover = 9 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l1074_107494


namespace NUMINAMATH_CALUDE_complex_calculation_l1074_107451

theorem complex_calculation (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  3*a - 4*b = 7 - 25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l1074_107451


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_four_l1074_107471

/-- If the sum of two monomials -3a^m*b^2 and (1/2)a^2*b^n is still a monomial, then mn = 4 -/
theorem monomial_sum_implies_mn_four (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), -3 * a^m * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) →
  m * n = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_four_l1074_107471


namespace NUMINAMATH_CALUDE_james_hives_l1074_107485

theorem james_hives (honey_per_hive : ℝ) (jar_capacity : ℝ) (jars_to_buy : ℕ) :
  honey_per_hive = 20 →
  jar_capacity = 0.5 →
  jars_to_buy = 100 →
  (honey_per_hive * (jars_to_buy : ℝ) * jar_capacity) / honey_per_hive = 5 :=
by sorry

end NUMINAMATH_CALUDE_james_hives_l1074_107485


namespace NUMINAMATH_CALUDE_latticePoindsInsideTriangleABO_l1074_107431

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0, 30)
def B : ℤ × ℤ := (20, 10)
def O : ℤ × ℤ := (0, 0)

-- Define a function to calculate the area of a triangle
def triangleArea (p1 p2 p3 : ℤ × ℤ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define Pick's theorem
def picksTheorem (S : ℚ) (N L : ℤ) : Prop :=
  S = N + L / 2 - 1

-- State the theorem
theorem latticePoindsInsideTriangleABO :
  ∃ (N L : ℤ),
    picksTheorem (triangleArea A B O) N L ∧
    L = 60 ∧
    N = 271 :=
  sorry

end NUMINAMATH_CALUDE_latticePoindsInsideTriangleABO_l1074_107431


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1074_107464

def parabola1 (x y : ℝ) : Prop := y = 3 * x^2 - 4 * x + 2
def parabola2 (x y : ℝ) : Prop := y = x^3 - 2 * x^2 + x + 2

def intersection_points : Set (ℝ × ℝ) :=
  {(0, 2),
   ((5 + Real.sqrt 5) / 2, 3 * ((5 + Real.sqrt 5) / 2)^2 - 4 * ((5 + Real.sqrt 5) / 2) + 2),
   ((5 - Real.sqrt 5) / 2, 3 * ((5 - Real.sqrt 5) / 2)^2 - 4 * ((5 - Real.sqrt 5) / 2) + 2)}

theorem parabolas_intersection :
  ∀ x y : ℝ, (parabola1 x y ∧ parabola2 x y) ↔ (x, y) ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1074_107464


namespace NUMINAMATH_CALUDE_triple_f_of_3_l1074_107492

def f (x : ℝ) : ℝ := 7 * x - 3

theorem triple_f_of_3 : f (f (f 3)) = 858 := by sorry

end NUMINAMATH_CALUDE_triple_f_of_3_l1074_107492


namespace NUMINAMATH_CALUDE_function_symmetry_l1074_107483

/-- A function f : ℝ → ℝ is symmetric with respect to the point (a, b) if f(x) + f(2a - x) = 2b for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

/-- The function property given in the problem -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-x)

theorem function_symmetry (f : ℝ → ℝ) (h : FunctionProperty f) :
  SymmetricAboutPoint f 1 0 := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_l1074_107483


namespace NUMINAMATH_CALUDE_kyro_debt_payment_percentage_l1074_107456

/-- Proves that Kyro paid 80% of her debt to Fernanda given the problem conditions -/
theorem kyro_debt_payment_percentage (aryan_debt : ℝ) (kyro_debt : ℝ) 
  (aryan_payment_percentage : ℝ) (initial_savings : ℝ) (final_savings : ℝ) :
  aryan_debt = 1200 →
  aryan_debt = 2 * kyro_debt →
  aryan_payment_percentage = 0.6 →
  initial_savings = 300 →
  final_savings = 1500 →
  (kyro_debt - (final_savings - initial_savings - aryan_payment_percentage * aryan_debt)) / kyro_debt = 0.2 := by
  sorry

#check kyro_debt_payment_percentage

end NUMINAMATH_CALUDE_kyro_debt_payment_percentage_l1074_107456


namespace NUMINAMATH_CALUDE_base7_product_digit_sum_l1074_107465

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list --/
def sumDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem base7_product_digit_sum :
  let a := base7ToDecimal 34
  let b := base7ToDecimal 52
  let product := a * b
  let base9Product := decimalToBase9 product
  sumDigits base9Product = 10 := by sorry

end NUMINAMATH_CALUDE_base7_product_digit_sum_l1074_107465


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1074_107481

theorem binomial_expansion_constant_term (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose n 4) →
  (Nat.choose n (n / 2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1074_107481


namespace NUMINAMATH_CALUDE_expression_equals_36_l1074_107417

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l1074_107417


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1074_107480

/-- The speed of a boat in still water, given stream speed and downstream travel data -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) : 
  stream_speed = 4 →
  downstream_distance = 112 →
  downstream_time = 4 →
  (downstream_distance / downstream_time) - stream_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1074_107480


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1074_107462

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1074_107462


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l1074_107489

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of extra topping options -/
def extra_toppings : ℕ := 3

/-- The number of topping combinations (including no extra toppings) -/
def topping_combinations : ℕ := 2^extra_toppings

/-- The total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count : total_varieties = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l1074_107489


namespace NUMINAMATH_CALUDE_nonzero_real_solution_cube_equation_l1074_107429

theorem nonzero_real_solution_cube_equation (y : ℝ) (h1 : y ≠ 0) (h2 : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_cube_equation_l1074_107429


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1074_107402

theorem quadratic_equation_proof (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →
  (p^2 - 2*p + m - 1 = 0) →
  ((p^2 - 2*p + 3) * (m + 4) = 7) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1074_107402


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1074_107435

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1074_107435


namespace NUMINAMATH_CALUDE_sum_equals_one_after_removal_l1074_107469

theorem sum_equals_one_after_removal : 
  let original_sum := (1/2 : ℚ) + 1/4 + 1/6 + 1/8 + 1/10 + 1/12
  let remaining_sum := (1/2 : ℚ) + 1/4 + 1/6 + 1/12
  remaining_sum = 1 := by sorry

end NUMINAMATH_CALUDE_sum_equals_one_after_removal_l1074_107469


namespace NUMINAMATH_CALUDE_circle_properties_l1074_107475

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 - 4*x + 3*y

-- Define the center and radius
def center : ℝ × ℝ := (4, -3)
def radius : ℝ := 5

theorem circle_properties :
  (circle_equation O.1 O.2 = 0) ∧
  (circle_equation M.1 M.2 = 0) ∧
  (circle_equation N.1 N.2 = 0) ∧
  (∀ (x y : ℝ), circle_equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1074_107475


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1074_107404

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 30 → -- One acute angle is 30 degrees
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180 degrees, and one angle is right (90 degrees)
  β = 60 := by -- The other acute angle is 60 degrees
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1074_107404


namespace NUMINAMATH_CALUDE_units_digit_of_sum_power_problem_solution_l1074_107424

theorem units_digit_of_sum_power (a b n : ℕ) : 
  (a + b) % 10 = 1 → ((a + b)^n) % 10 = 1 :=
by
  sorry

theorem problem_solution : ((5619 + 2272)^124) % 10 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_power_problem_solution_l1074_107424


namespace NUMINAMATH_CALUDE_square_difference_equality_l1074_107491

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1074_107491


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1074_107443

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  -- Side lengths
  a : ℝ  -- Length of the side opposite to the 30° angle
  b : ℝ  -- Length of the side opposite to the 60° angle
  c : ℝ  -- Length of the hypotenuse (opposite to the 90° angle)
  -- Properties of a 30-60-90 triangle
  h1 : a = c / 2
  h2 : b = a * Real.sqrt 3

/-- Theorem: In a 30-60-90 triangle with side length opposite to 60° angle equal to 12, 
    the length of the hypotenuse is 8√3 -/
theorem hypotenuse_length (t : Triangle30_60_90) (h : t.b = 12) : t.c = 8 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_l1074_107443


namespace NUMINAMATH_CALUDE_littleTwelve_game_count_l1074_107400

/-- Represents a basketball conference with two divisions -/
structure BasketballConference where
  teamsPerDivision : ℕ
  inDivisionGames : ℕ
  crossDivisionGames : ℕ

/-- Calculates the total number of games in the conference -/
def totalGames (conf : BasketballConference) : ℕ :=
  2 * (conf.teamsPerDivision.choose 2 * conf.inDivisionGames) + 
  conf.teamsPerDivision * conf.teamsPerDivision * conf.crossDivisionGames

/-- The Little Twelve Basketball Conference -/
def littleTwelve : BasketballConference := {
  teamsPerDivision := 6
  inDivisionGames := 2
  crossDivisionGames := 1
}

theorem littleTwelve_game_count : totalGames littleTwelve = 96 := by
  sorry

end NUMINAMATH_CALUDE_littleTwelve_game_count_l1074_107400


namespace NUMINAMATH_CALUDE_bakery_customers_l1074_107468

theorem bakery_customers (total_pastries : ℕ) (regular_customers : ℕ) (pastry_difference : ℕ) :
  total_pastries = 392 →
  regular_customers = 28 →
  pastry_difference = 6 →
  ∃ (actual_customers : ℕ),
    actual_customers * (total_pastries / regular_customers - pastry_difference) = total_pastries ∧
    actual_customers = 49 := by
  sorry

end NUMINAMATH_CALUDE_bakery_customers_l1074_107468


namespace NUMINAMATH_CALUDE_candy_distribution_l1074_107408

/-- Given 200 candies distributed among A, B, and C, where A has more than twice as many candies as B,
    and B has more than three times as many candies as C, prove that the minimum number of candies A
    can have is 121, and the maximum number of candies C can have is 19. -/
theorem candy_distribution (a b c : ℕ) : 
  a + b + c = 200 →
  a > 2 * b →
  b > 3 * c →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → a' ≥ a) →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → c' ≤ c) →
  a = 121 ∧ c = 19 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1074_107408


namespace NUMINAMATH_CALUDE_catch_up_time_is_correct_l1074_107472

/-- The time (in minutes) for the minute hand to catch up with the hour hand after 8:00 --/
def catch_up_time : ℚ :=
  let minute_hand_speed : ℚ := 6
  let hour_hand_speed : ℚ := 1/2
  let initial_hour_hand_position : ℚ := 240
  (initial_hour_hand_position / (minute_hand_speed - hour_hand_speed))

theorem catch_up_time_is_correct : catch_up_time = 43 + 7/11 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_time_is_correct_l1074_107472


namespace NUMINAMATH_CALUDE_largest_digit_sum_l1074_107405

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) :
  isDigit a → isDigit b → isDigit c →
  isPrime y →
  0 ≤ y ∧ y ≤ 7 →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  a + b + c ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l1074_107405


namespace NUMINAMATH_CALUDE_joe_cookies_sold_l1074_107407

/-- The number of cookies Joe sold -/
def cookies : ℕ := sorry

/-- The cost to make each cookie in dollars -/
def cost : ℚ := 1

/-- The markup percentage -/
def markup : ℚ := 20 / 100

/-- The selling price of each cookie -/
def selling_price : ℚ := cost * (1 + markup)

/-- The total revenue in dollars -/
def revenue : ℚ := 60

theorem joe_cookies_sold :
  cookies = 50 ∧
  selling_price * cookies = revenue :=
sorry

end NUMINAMATH_CALUDE_joe_cookies_sold_l1074_107407


namespace NUMINAMATH_CALUDE_smallest_intersection_percentage_l1074_107436

theorem smallest_intersection_percentage (S J : ℝ) : 
  S = 90 → J = 80 → 
  ∃ (I : ℝ), I ≥ 70 ∧ I ≤ S ∧ I ≤ J ∧ 
  ∀ (I' : ℝ), I' ≤ S ∧ I' ≤ J → I' ≤ I := by
  sorry

end NUMINAMATH_CALUDE_smallest_intersection_percentage_l1074_107436


namespace NUMINAMATH_CALUDE_correct_num_clowns_l1074_107478

/-- The number of clowns attending a carousel --/
def num_clowns : ℕ := 4

/-- The number of children attending the carousel --/
def num_children : ℕ := 30

/-- The total number of candies initially --/
def total_candies : ℕ := 700

/-- The number of candies given to each person --/
def candies_per_person : ℕ := 20

/-- The number of candies left after distribution --/
def candies_left : ℕ := 20

/-- Theorem stating that the number of clowns is correct given the conditions --/
theorem correct_num_clowns :
  num_clowns * candies_per_person + num_children * candies_per_person + candies_left = total_candies :=
by sorry

end NUMINAMATH_CALUDE_correct_num_clowns_l1074_107478


namespace NUMINAMATH_CALUDE_f_equals_g_l1074_107449

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := (x^3)^(1/3)

-- Statement to prove
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l1074_107449


namespace NUMINAMATH_CALUDE_sqrt_sum_abs_equal_six_l1074_107421

theorem sqrt_sum_abs_equal_six :
  Real.sqrt 2 + Real.sqrt 16 + |Real.sqrt 2 - 2| = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_abs_equal_six_l1074_107421


namespace NUMINAMATH_CALUDE_parabola_directrix_l1074_107463

/-- A parabola with equation y^2 = 2px and focus on the line 2x + 3y - 4 = 0 has directrix x = -2 -/
theorem parabola_directrix (p : ℝ) : 
  ∃ (f : ℝ × ℝ), 
    (∀ (x y : ℝ), y^2 = 2*p*x ↔ ((x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2)) ∧ 
    (2*f.1 + 3*f.2 - 4 = 0) → 
    (f.1 = 2 ∧ f.2 = 0 ∧ ∀ (x : ℝ), x = -2 ↔ x = f.1 - p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1074_107463


namespace NUMINAMATH_CALUDE_simplify_fraction_l1074_107447

theorem simplify_fraction (a b : ℝ) (h : a + b ≠ 0) :
  ((a - b) / (a + 2*b)) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) - 2 = -a / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1074_107447


namespace NUMINAMATH_CALUDE_angle_FAH_is_45_degrees_l1074_107442

/-- Given a unit square ABCD with EF parallel to AB, GH parallel to BC, BF = 1/4, and BF + DH = FH, 
    the measure of angle FAH is 45 degrees. -/
theorem angle_FAH_is_45_degrees (A B C D E F G H : ℝ × ℝ) : 
  -- Unit square ABCD
  A = (0, 1) ∧ B = (0, 0) ∧ C = (1, 0) ∧ D = (1, 1) →
  -- EF is parallel to AB
  (E.2 - F.2) / (E.1 - F.1) = (A.2 - B.2) / (A.1 - B.1) →
  -- GH is parallel to BC
  (G.2 - H.2) / (G.1 - H.1) = (B.2 - C.2) / (B.1 - C.1) →
  -- BF = 1/4
  F = (1/4, 0) →
  -- BF + DH = FH
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) + 
  Real.sqrt ((D.1 - H.1)^2 + (D.2 - H.2)^2) = 
  Real.sqrt ((F.1 - H.1)^2 + (F.2 - H.2)^2) →
  -- Angle FAH is 45 degrees
  Real.arctan (((A.2 - F.2) / (A.1 - F.1) - (A.2 - H.2) / (A.1 - H.1)) / 
    (1 + (A.2 - F.2) / (A.1 - F.1) * (A.2 - H.2) / (A.1 - H.1))) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_FAH_is_45_degrees_l1074_107442


namespace NUMINAMATH_CALUDE_inequality_proof_l1074_107415

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1074_107415


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1074_107452

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 2 + a 4 = 5) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 2^(4 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1074_107452


namespace NUMINAMATH_CALUDE_factorial_division_l1074_107493

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1074_107493


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1074_107416

theorem purely_imaginary_complex_number (m : ℝ) :
  (2 * m^2 - 3 * m - 2 : ℂ) + (6 * m^2 + 5 * m + 1 : ℂ) * Complex.I = Complex.I * ((6 * m^2 + 5 * m + 1 : ℝ) : ℂ) →
  m = -1 ∨ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1074_107416


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l1074_107414

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents a coloring of the chessboard -/
def Coloring := Square → Bool

/-- Checks if three squares form a trimino -/
def isTrimino (s1 s2 s3 : Square) : Prop := sorry

/-- Counts the number of red squares in a coloring -/
def countRedSquares (c : Coloring) : Nat := sorry

/-- Checks if a coloring has no red trimino -/
def hasNoRedTrimino (c : Coloring) : Prop := sorry

/-- Checks if every trimino in a coloring has at least one red square -/
def everyTriminoHasRed (c : Coloring) : Prop := sorry

theorem chessboard_coloring_theorem :
  (∃ c : Coloring, hasNoRedTrimino c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, hasNoRedTrimino c → countRedSquares c ≤ 32) ∧
  (∃ c : Coloring, everyTriminoHasRed c ∧ countRedSquares c = 32) ∧
  (∀ c : Coloring, everyTriminoHasRed c → countRedSquares c ≥ 32) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l1074_107414


namespace NUMINAMATH_CALUDE_number_multiplication_problem_l1074_107401

theorem number_multiplication_problem :
  ∃ x : ℝ, x * 4 * 25 = 812 ∧ x = 8.12 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_problem_l1074_107401


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1074_107430

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1074_107430


namespace NUMINAMATH_CALUDE_mark_donation_shelters_l1074_107410

/-- The number of shelters Mark donates soup to -/
def num_shelters (people_per_shelter : ℕ) (cans_per_person : ℕ) (total_cans : ℕ) : ℕ :=
  total_cans / (people_per_shelter * cans_per_person)

theorem mark_donation_shelters :
  num_shelters 30 10 1800 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_donation_shelters_l1074_107410


namespace NUMINAMATH_CALUDE_profit_calculation_l1074_107477

/-- The number of pencils John needs to sell to make a profit of $120 -/
def pencils_to_sell : ℕ := 1200

/-- The cost of buying 5 pencils in dollars -/
def buy_cost : ℚ := 7

/-- The number of pencils John buys at the given cost -/
def buy_quantity : ℕ := 5

/-- The selling price of 4 pencils in dollars -/
def sell_price : ℚ := 6

/-- The number of pencils John sells at the given price -/
def sell_quantity : ℕ := 4

/-- The desired profit in dollars -/
def target_profit : ℚ := 120

/-- Theorem stating that the number of pencils John needs to sell to make a profit of $120 is correct -/
theorem profit_calculation (p : ℕ) (h : p = pencils_to_sell) :
  (p : ℚ) * (sell_price / sell_quantity - buy_cost / buy_quantity) = target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1074_107477


namespace NUMINAMATH_CALUDE_bus_passengers_problem_l1074_107499

/-- Given a bus with an initial number of passengers and a number of passengers who got off,
    calculate the number of passengers remaining on the bus. -/
def passengers_remaining (initial : ℕ) (got_off : ℕ) : ℕ :=
  initial - got_off

/-- Theorem stating that given 90 initial passengers and 47 passengers who got off,
    the number of remaining passengers is 43. -/
theorem bus_passengers_problem :
  passengers_remaining 90 47 = 43 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_problem_l1074_107499


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l1074_107460

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def is_least_with_property (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + 1 = b ∧ b ≤ 20 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ a ∧ k ≠ b → is_divisible n k) ∧
  (∀ m : ℕ, m < n → ¬∃ (c d : ℕ), c + 1 = d ∧ d ≤ 20 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ c ∧ k ≠ d → is_divisible m k))

theorem least_number_with_divisibility_property :
  is_least_with_property 12252240 := by sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l1074_107460


namespace NUMINAMATH_CALUDE_sin_inequality_l1074_107458

theorem sin_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  Real.sin (2 * α) + 2 / Real.sin (2 * α) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l1074_107458


namespace NUMINAMATH_CALUDE_student_number_problem_l1074_107455

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1074_107455


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l1074_107496

/-- The ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- The chord type -/
structure Chord where
  p1 : Point
  p2 : Point

/-- The theorem statement -/
theorem ellipse_chord_theorem (e : Ellipse) (c : Chord) (F1 F2 : Point) :
  e.a = 5 →
  e.b = 4 →
  F1.x = -3 →
  F1.y = 0 →
  F2.x = 3 →
  F2.y = 0 →
  (c.p1.x^2 / 25 + c.p1.y^2 / 16 = 1) →
  (c.p2.x^2 / 25 + c.p2.y^2 / 16 = 1) →
  (c.p1.x - F1.x) * (c.p2.y - F1.y) = (c.p1.y - F1.y) * (c.p2.x - F1.x) →
  (Real.pi = 2 * Real.pi * (Real.sqrt (5 * 5 / 36))) →
  |c.p1.y - c.p2.y| = 5/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l1074_107496


namespace NUMINAMATH_CALUDE_function_range_complement_l1074_107437

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 2

theorem function_range_complement :
  {k : ℝ | ∀ x, f x ≠ k} = Set.Iio (-3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_complement_l1074_107437


namespace NUMINAMATH_CALUDE_shaded_semicircle_perimeter_l1074_107482

/-- The perimeter of a shaded region in a semicircle -/
theorem shaded_semicircle_perimeter (r : ℝ) (h : r = 2) :
  let arc_length := π * r / 2
  let radii_length := 2 * r
  arc_length + radii_length = π + 4 := by
  sorry


end NUMINAMATH_CALUDE_shaded_semicircle_perimeter_l1074_107482


namespace NUMINAMATH_CALUDE_power_mod_seven_l1074_107413

theorem power_mod_seven : 3^1995 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l1074_107413


namespace NUMINAMATH_CALUDE_computer_price_increase_l1074_107433

theorem computer_price_increase (c : ℝ) : 
  c + c * 0.3 = 351 → c + 351 = 621 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1074_107433


namespace NUMINAMATH_CALUDE_no_prime_pairs_with_integer_ratios_l1074_107411

theorem no_prime_pairs_with_integer_ratios : 
  ¬ ∃ (x y : ℕ), Prime x ∧ Prime y ∧ y < x ∧ x ≤ 200 ∧ 
  (x / y : ℚ).isInt ∧ ((x + 1) / (y + 1) : ℚ).isInt := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pairs_with_integer_ratios_l1074_107411


namespace NUMINAMATH_CALUDE_area_difference_l1074_107484

/-- A right isosceles triangle with base length 1 -/
structure RightIsoscelesTriangle where
  base : ℝ
  base_eq_one : base = 1

/-- Configuration of two identical squares in the triangle (Figure 2) -/
structure SquareConfig2 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = 1 / 4

/-- Configuration of two identical squares in the triangle (Figure 3) -/
structure SquareConfig3 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = Real.sqrt 2 / 6

/-- Total area of squares in Configuration 2 -/
def totalArea2 (t : RightIsoscelesTriangle) (c : SquareConfig2 t) : ℝ :=
  2 * c.side_length ^ 2

/-- Total area of squares in Configuration 3 -/
def totalArea3 (t : RightIsoscelesTriangle) (c : SquareConfig3 t) : ℝ :=
  2 * c.side_length ^ 2

/-- The main theorem stating the difference in areas -/
theorem area_difference (t : RightIsoscelesTriangle) 
  (c2 : SquareConfig2 t) (c3 : SquareConfig3 t) : 
  totalArea2 t c2 - totalArea3 t c3 = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_l1074_107484


namespace NUMINAMATH_CALUDE_simplify_expression_l1074_107427

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1074_107427


namespace NUMINAMATH_CALUDE_max_area_enclosure_l1074_107467

/-- Represents a rectangular enclosure with length and width. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is 500 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 500

/-- The length of the enclosure is at least 100 feet. -/
def minLengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def minWidthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- Theorem stating that the maximum area of the enclosure satisfying all constraints is 15625 square feet. -/
theorem max_area_enclosure :
  ∃ (e : Enclosure),
    perimeterConstraint e ∧
    minLengthConstraint e ∧
    minWidthConstraint e ∧
    (∀ (e' : Enclosure),
      perimeterConstraint e' ∧
      minLengthConstraint e' ∧
      minWidthConstraint e' →
      area e' ≤ area e) ∧
    area e = 15625 :=
  sorry

end NUMINAMATH_CALUDE_max_area_enclosure_l1074_107467


namespace NUMINAMATH_CALUDE_largest_number_with_123_l1074_107438

theorem largest_number_with_123 :
  let a := 321
  let b := 21^3
  let c := 3^21
  let d := 2^31
  (c > a) ∧ (c > b) ∧ (c > d) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_123_l1074_107438


namespace NUMINAMATH_CALUDE_average_cookies_l1074_107459

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 125 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_cookies_l1074_107459


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1074_107473

/-- Given two similar triangles PQR and STU, prove that if PQ = 12, QR = 10, and ST = 18, then TU = 15 -/
theorem similar_triangles_side_length 
  (PQ QR ST TU : ℝ) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ PQ = k * ST ∧ QR = k * TU) 
  (h_PQ : PQ = 12) 
  (h_QR : QR = 10) 
  (h_ST : ST = 18) : 
  TU = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1074_107473


namespace NUMINAMATH_CALUDE_remaining_jellybeans_l1074_107490

/-- Calculates the number of jelly beans remaining in a container after distribution --/
def jellybeans_remaining (initial_count : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_beans : ℕ) : ℕ :=
  initial_count - (first_group * 2 * last_group_beans + last_group * last_group_beans)

/-- Theorem stating the number of jelly beans remaining in the container --/
theorem remaining_jellybeans : 
  jellybeans_remaining 8000 10 6 4 400 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_remaining_jellybeans_l1074_107490


namespace NUMINAMATH_CALUDE_puppies_difference_l1074_107448

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := 2 * second_week

/-- The total number of puppies Yuri has after four weeks -/
def total_puppies : ℕ := 74

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := total_puppies - (first_week + second_week + third_week)

theorem puppies_difference : fourth_week - first_week = 10 := by
  sorry

end NUMINAMATH_CALUDE_puppies_difference_l1074_107448


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1074_107450

theorem geometric_sequence_second_term :
  ∀ (a : ℕ+) (r : ℕ+),
    a = 5 →
    a * r^4 = 1280 →
    a * r = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1074_107450


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1074_107418

theorem min_value_expression (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  (x₁^2 + x₂^2) / (x₁ - x₂) ≥ 16 :=
by sorry

theorem min_value_achieved (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  ∃ x₁' x₂' : ℝ, x₁' + x₂' = 16 ∧ x₁' > x₂' ∧ (x₁'^2 + x₂'^2) / (x₁' - x₂') = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l1074_107418


namespace NUMINAMATH_CALUDE_unique_six_digit_square_l1074_107420

/-- Checks if a number has all digits different --/
def has_different_digits (n : Nat) : Bool :=
  sorry

/-- Checks if digits in a number are in ascending order --/
def digits_ascending (n : Nat) : Bool :=
  sorry

/-- The unique six-digit perfect square with ascending, different digits --/
theorem unique_six_digit_square : 
  ∃! n : Nat, 
    100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
    has_different_digits n ∧ 
    digits_ascending n ∧ 
    ∃ m : Nat, n = m^2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_square_l1074_107420


namespace NUMINAMATH_CALUDE_square_rectangle_ratio_is_5_28_l1074_107497

/-- The number of horizontal or vertical lines on the checkerboard -/
def num_lines : ℕ := 8

/-- The size of the checkerboard -/
def board_size : ℕ := 7

/-- The number of rectangles on the checkerboard -/
def num_rectangles : ℕ := (num_lines.choose 2) ^ 2

/-- The number of squares on the checkerboard -/
def num_squares : ℕ := board_size * (board_size + 1) * (2 * board_size + 1) / 6

/-- The ratio of squares to rectangles -/
def square_rectangle_ratio : ℚ := num_squares / num_rectangles

theorem square_rectangle_ratio_is_5_28 : square_rectangle_ratio = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_ratio_is_5_28_l1074_107497


namespace NUMINAMATH_CALUDE_problem_statement_l1074_107419

theorem problem_statement (x y z w : ℝ) 
  (eq1 : 2^x + y = 7)
  (eq2 : 2^8 = y + x)
  (eq3 : z = Real.sin (x - y))
  (eq4 : w = 3 * (y + z)) :
  ∃ (result : ℝ), (x + y + z + w) / 4 = result := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1074_107419


namespace NUMINAMATH_CALUDE_ellipse_sum_property_l1074_107432

/-- Represents an ellipse with its properties -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length
  θ : ℝ  -- rotation angle in radians

/-- Theorem: For a specific ellipse, the sum of its center coordinates and axis lengths is 11 -/
theorem ellipse_sum_property : 
  ∀ (e : Ellipse), 
  e.h = -2 ∧ e.k = 3 ∧ e.a = 6 ∧ e.b = 4 ∧ e.θ = π/4 → 
  e.h + e.k + e.a + e.b = 11 := by
sorry

end NUMINAMATH_CALUDE_ellipse_sum_property_l1074_107432


namespace NUMINAMATH_CALUDE_inequality_proof_l1074_107466

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧ 
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1074_107466


namespace NUMINAMATH_CALUDE_frustum_volume_l1074_107457

/-- The volume of a frustum formed by cutting a square pyramid parallel to its base -/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 16 →
  altitude = 10 →
  small_base_edge = 8 →
  small_altitude = 5 →
  let original_volume := (1 / 3) * base_edge^2 * altitude
  let small_volume := (1 / 3) * small_base_edge^2 * small_altitude
  original_volume - small_volume = 2240 / 3 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l1074_107457


namespace NUMINAMATH_CALUDE_person_a_higher_probability_l1074_107423

/-- Represents the space station simulation programming challenge. -/
structure Challenge where
  total_questions : Nat
  questions_per_participant : Nat
  passing_threshold : Nat
  person_a_correct_questions : Nat
  person_b_success_probability : Real

/-- Calculates the probability of passing the challenge given the number of correct programs. -/
def probability_of_passing (c : Challenge) (correct_programs : Nat) : Real :=
  if correct_programs ≥ c.passing_threshold then 1 else 0

/-- Calculates the probability of person B passing the challenge. -/
def person_b_passing_probability (c : Challenge) : Real :=
  sorry

/-- Calculates the probability of person A passing the challenge. -/
def person_a_passing_probability (c : Challenge) : Real :=
  sorry

/-- The main theorem stating that person A has a higher probability of passing the challenge. -/
theorem person_a_higher_probability (c : Challenge) 
  (h1 : c.total_questions = 10)
  (h2 : c.questions_per_participant = 3)
  (h3 : c.passing_threshold = 2)
  (h4 : c.person_a_correct_questions = 6)
  (h5 : c.person_b_success_probability = 0.6) :
  person_a_passing_probability c > person_b_passing_probability c :=
sorry

end NUMINAMATH_CALUDE_person_a_higher_probability_l1074_107423


namespace NUMINAMATH_CALUDE_intersection_M_P_l1074_107425

-- Define the sets M and P
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}
def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_M_P : M ∩ P = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_P_l1074_107425


namespace NUMINAMATH_CALUDE_michaels_pets_percentage_l1074_107445

theorem michaels_pets_percentage (total_pets : ℕ) (cat_percentage : ℚ) (num_bunnies : ℕ) :
  total_pets = 36 →
  cat_percentage = 1/2 →
  num_bunnies = 9 →
  (total_pets : ℚ) * (1 - cat_percentage) - num_bunnies = (total_pets : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_michaels_pets_percentage_l1074_107445


namespace NUMINAMATH_CALUDE_at_least_one_multiple_of_11_l1074_107488

def base_n_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem at_least_one_multiple_of_11 :
  ∃ n : Nat, 2 ≤ n ∧ n ≤ 101 ∧ 
  (base_n_to_decimal [3, 4, 5, 7, 6, 2] n) % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_multiple_of_11_l1074_107488


namespace NUMINAMATH_CALUDE_will_chocolate_boxes_l1074_107495

theorem will_chocolate_boxes :
  ∀ (boxes_given : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ),
    boxes_given = 3 →
    pieces_per_box = 4 →
    pieces_left = 16 →
    (boxes_given * pieces_per_box + pieces_left) / pieces_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_boxes_l1074_107495
