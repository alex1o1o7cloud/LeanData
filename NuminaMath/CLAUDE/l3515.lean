import Mathlib

namespace NUMINAMATH_CALUDE_erasers_bought_l3515_351525

theorem erasers_bought (initial_erasers final_erasers : ℝ) (h1 : initial_erasers = 95.0) (h2 : final_erasers = 137) : 
  final_erasers - initial_erasers = 42 := by
  sorry

end NUMINAMATH_CALUDE_erasers_bought_l3515_351525


namespace NUMINAMATH_CALUDE_a_less_than_neg_one_sufficient_not_necessary_l3515_351514

theorem a_less_than_neg_one_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x < -1 → x + 1/x < -2) ∧
  (∃ y : ℝ, y ≥ -1 ∧ y + 1/y < -2) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_neg_one_sufficient_not_necessary_l3515_351514


namespace NUMINAMATH_CALUDE_infinite_binary_sequences_and_powerset_cardinality_l3515_351561

/-- The type of infinite binary sequences -/
def InfiniteBinarySequence := ℕ → Fin 2

/-- The cardinality of the continuum -/
def ContinuumCardinality := Cardinal.mk (Set ℝ)

theorem infinite_binary_sequences_and_powerset_cardinality :
  (Cardinal.mk (Set InfiniteBinarySequence) = ContinuumCardinality) ∧
  (Cardinal.mk (Set (Set ℕ)) = ContinuumCardinality) := by
  sorry

end NUMINAMATH_CALUDE_infinite_binary_sequences_and_powerset_cardinality_l3515_351561


namespace NUMINAMATH_CALUDE_problem_statement_l3515_351536

theorem problem_statement (g : ℕ) (r₁ r₂ : ℕ) : 
  g = 29 →
  r₂ = 11 →
  g > 0 →
  ∃ k₁ k₂ : ℕ, 1255 = g * k₁ + r₁ ∧ 1490 = g * k₂ + r₂ →
  (∀ d : ℕ, d > g → ¬(∃ m₁ m₂ : ℕ, 1255 = d * m₁ + r₁ ∧ 1490 = d * m₂ + r₂)) →
  r₁ = 8 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3515_351536


namespace NUMINAMATH_CALUDE_function_inequality_implies_range_l3515_351522

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem function_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f (x^2 + 2) + f (-2*a*x) ≥ 0) →
  a ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_range_l3515_351522


namespace NUMINAMATH_CALUDE_range_of_x_minus_y_l3515_351504

theorem range_of_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) :
  ∃ a b : ℝ, a = -4 ∧ b = 2 ∧ a < x - y ∧ x - y < b :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_y_l3515_351504


namespace NUMINAMATH_CALUDE_original_profit_percentage_l3515_351559

def original_selling_price : ℝ := 550

def new_selling_price (original_purchase_price : ℝ) : ℝ :=
  original_purchase_price * 0.9 * 1.3

theorem original_profit_percentage : 
  ∃ (original_purchase_price : ℝ) (original_profit_percentage : ℝ),
    original_purchase_price * (1 + original_profit_percentage / 100) = original_selling_price ∧
    new_selling_price original_purchase_price = original_selling_price + 35 ∧
    original_profit_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l3515_351559


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3515_351552

/-- Proves the general term of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h3 : ∃ m : ℝ, ∀ n : ℕ, Real.sqrt (8 * S n + 2 * n) = m + (n - 1) * d) :
  ∀ n : ℕ, a n = 4 * n - 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3515_351552


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_twice_exterior_l3515_351578

theorem polygon_sides_when_interior_twice_exterior :
  ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 2 * 360) →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_twice_exterior_l3515_351578


namespace NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l3515_351563

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l3515_351563


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3515_351540

theorem sum_of_decimals : (1 : ℚ) + 0.101 + 0.011 + 0.001 = 1.113 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3515_351540


namespace NUMINAMATH_CALUDE_equation_solutions_range_l3515_351512

theorem equation_solutions_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) →
  m ∈ Set.Icc 0 8 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_range_l3515_351512


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3515_351544

/-- The equation (x-y)^2 = 3(x^2 - y^2) represents a hyperbola -/
theorem equation_represents_hyperbola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b^2 - 4*a*c > 0 ∧
  ∀ (x y : ℝ), (x - y)^2 = 3*(x^2 - y^2) ↔ a*x^2 + b*x*y + c*y^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3515_351544


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l3515_351545

/-- Calculates the discount percentage for a shirt given its cost price, profit margin, and sale price. -/
theorem shirt_discount_percentage
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (sale_price : ℝ)
  (h1 : cost_price = 20)
  (h2 : profit_margin = 0.3)
  (h3 : sale_price = 13) :
  (cost_price * (1 + profit_margin) - sale_price) / (cost_price * (1 + profit_margin)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_percentage_l3515_351545


namespace NUMINAMATH_CALUDE_largest_package_size_l3515_351533

theorem largest_package_size (alex bella carlos : ℕ) 
  (h_alex : alex = 36)
  (h_bella : bella = 48)
  (h_carlos : carlos = 60) :
  Nat.gcd alex (Nat.gcd bella carlos) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l3515_351533


namespace NUMINAMATH_CALUDE_elenas_earnings_l3515_351521

/-- Calculates the total earnings given an hourly wage and number of hours worked -/
def totalEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) : ℚ :=
  hourlyWage * hoursWorked

/-- Proves that Elena's earnings for 4 hours at $13.25 per hour is $53.00 -/
theorem elenas_earnings :
  totalEarnings (13.25 : ℚ) (4 : ℚ) = (53 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_elenas_earnings_l3515_351521


namespace NUMINAMATH_CALUDE_floor_product_l3515_351571

theorem floor_product : ⌊(21.7 : ℝ)⌋ * ⌊(-21.7 : ℝ)⌋ = -462 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_l3515_351571


namespace NUMINAMATH_CALUDE_circle_center_l3515_351523

/-- The center of the circle with equation x^2 + y^2 + 4x - 6y + 9 = 0 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 6*y + 9 = 0) ↔ ((x + 2)^2 + (y - 3)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_l3515_351523


namespace NUMINAMATH_CALUDE_g_negative_three_l3515_351555

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

-- State the theorem
theorem g_negative_three (d e f : ℝ) : g d e f 3 = -9 → g d e f (-3) = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_l3515_351555


namespace NUMINAMATH_CALUDE_total_study_time_is_135_l3515_351586

def math_time : ℕ := 60

def geography_time : ℕ := math_time / 2

def science_time : ℕ := (math_time + geography_time) / 2

def total_study_time : ℕ := math_time + geography_time + science_time

theorem total_study_time_is_135 : total_study_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_study_time_is_135_l3515_351586


namespace NUMINAMATH_CALUDE_power_sum_is_integer_l3515_351590

theorem power_sum_is_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_is_integer_l3515_351590


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l3515_351553

/-- Calculates the time required to pump water out of a flooded basement. -/
theorem basement_water_pump_time
  (basement_length : ℝ)
  (basement_width : ℝ)
  (water_depth_inches : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (cubic_foot_to_gallon : ℝ)
  (h1 : basement_length = 30)
  (h2 : basement_width = 40)
  (h3 : water_depth_inches = 24)
  (h4 : num_pumps = 4)
  (h5 : pump_rate = 10)
  (h6 : cubic_foot_to_gallon = 7.5) :
  (basement_length * basement_width * (water_depth_inches / 12) * cubic_foot_to_gallon) /
  (num_pumps * pump_rate) = 450 := by
  sorry

#check basement_water_pump_time

end NUMINAMATH_CALUDE_basement_water_pump_time_l3515_351553


namespace NUMINAMATH_CALUDE_apple_ratio_proof_l3515_351516

theorem apple_ratio_proof (red_apples green_apples : ℕ) : 
  red_apples = 32 →
  red_apples + green_apples = 44 →
  (red_apples : ℚ) / green_apples = 8 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_proof_l3515_351516


namespace NUMINAMATH_CALUDE_triangle_properties_l3515_351517

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom side_a : a = 4
axiom side_c : c = Real.sqrt 13
axiom sin_relation : Real.sin A = 4 * Real.sin B

-- State the theorem
theorem triangle_properties : b = 1 ∧ C = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3515_351517


namespace NUMINAMATH_CALUDE_cylinder_volume_scale_l3515_351576

/-- Given a cylinder with volume V, radius r, and height h, 
    if the radius is tripled and the height is quadrupled, 
    then the new volume V' is 36 times the original volume V. -/
theorem cylinder_volume_scale (V r h : ℝ) (h1 : V = π * r^2 * h) : 
  let V' := π * (3*r)^2 * (4*h)
  V' = 36 * V := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_scale_l3515_351576


namespace NUMINAMATH_CALUDE_sophomore_count_l3515_351565

theorem sophomore_count (total_students : ℕ) 
  (junior_percent : ℚ) (senior_percent : ℚ) (sophomore_percent : ℚ) :
  total_students = 45 →
  junior_percent = 1/5 →
  senior_percent = 3/20 →
  sophomore_percent = 1/10 →
  ∃ (juniors seniors sophomores : ℕ),
    juniors + seniors + sophomores = total_students ∧
    (junior_percent : ℚ) * juniors = (senior_percent : ℚ) * seniors ∧
    (senior_percent : ℚ) * seniors = (sophomore_percent : ℚ) * sophomores ∧
    sophomores = 21 :=
by sorry

end NUMINAMATH_CALUDE_sophomore_count_l3515_351565


namespace NUMINAMATH_CALUDE_larger_number_problem_l3515_351575

theorem larger_number_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3515_351575


namespace NUMINAMATH_CALUDE_closest_integer_to_largest_root_squared_l3515_351591

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 - 2*x + 3

-- State the theorem
theorem closest_integer_to_largest_root_squared : 
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
    (a > b ∧ a > c) ∧
    (abs (a^2 - 67) < 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_largest_root_squared_l3515_351591


namespace NUMINAMATH_CALUDE_choose_4_from_10_l3515_351593

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l3515_351593


namespace NUMINAMATH_CALUDE_root_properties_l3515_351502

theorem root_properties (m n : ℤ) (x₁ x₂ : ℝ) : 
  Odd m → Odd n → x₁^2 + m*x₁ + n = 0 → x₂^2 + m*x₂ + n = 0 → x₁ ≠ x₂ →
  ¬(∃ k : ℤ, x₁ = k) ∧ ¬(∃ k : ℤ, x₂ = k) := by
sorry

end NUMINAMATH_CALUDE_root_properties_l3515_351502


namespace NUMINAMATH_CALUDE_problem_solution_l3515_351520

theorem problem_solution (A B : ℝ) : 
  (A^2 = 0.012345678987654321 * (List.sum (List.range 9) + List.sum (List.reverse (List.range 9)))) →
  (B^2 = 0.012345679012345679) →
  9 * (10^9 : ℝ) * (1 - |A|) * B = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3515_351520


namespace NUMINAMATH_CALUDE_student_calculation_correct_result_problem_statement_l3515_351549

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem student_calculation (c d : ℕ) : ℚ :=
  74 * (1 + (c : ℚ) / 10 + (d : ℚ) / 100) + 3

theorem correct_result (c d : ℕ) : ℚ :=
  74 * repeating_decimal c d + 3

theorem problem_statement (c d : ℕ) : 
  correct_result c d - student_calculation c d = 1.2 → c = 1 ∧ d = 6 :=
sorry

end NUMINAMATH_CALUDE_student_calculation_correct_result_problem_statement_l3515_351549


namespace NUMINAMATH_CALUDE_square_side_length_l3515_351573

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3515_351573


namespace NUMINAMATH_CALUDE_equation_solution_l3515_351528

theorem equation_solution : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 5) * (x - 2)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3515_351528


namespace NUMINAMATH_CALUDE_remainder_polynomial_division_l3515_351596

theorem remainder_polynomial_division (z : ℂ) : 
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 - 1 = (z^3 - 1) * (Q z) + R z) ∧ 
    (∃ (a b c : ℂ), ∀ z, R z = a*z^2 + b*z + c) ∧
    R z = z^2 + z - 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_polynomial_division_l3515_351596


namespace NUMINAMATH_CALUDE_percentage_problem_l3515_351567

theorem percentage_problem (x : ℝ) : 
  (15 / 100 * 40 = x / 100 * 16 + 2) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3515_351567


namespace NUMINAMATH_CALUDE_number_of_boys_l3515_351550

theorem number_of_boys (num_vans : ℕ) (students_per_van : ℕ) (num_girls : ℕ) : 
  num_vans = 5 → students_per_van = 28 → num_girls = 80 → 
  num_vans * students_per_van - num_girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3515_351550


namespace NUMINAMATH_CALUDE_cycle_price_problem_l3515_351511

/-- Given a cycle sold at a 25% loss for Rs. 2100, prove that the original price was Rs. 2800. -/
theorem cycle_price_problem (selling_price : ℝ) (loss_percentage : ℝ) 
    (h1 : selling_price = 2100)
    (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 2800 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_problem_l3515_351511


namespace NUMINAMATH_CALUDE_sale_price_increase_l3515_351515

theorem sale_price_increase (regular_price : ℝ) (regular_price_positive : regular_price > 0) : 
  let sale_price := regular_price * (1 - 0.2)
  let price_increase := regular_price - sale_price
  let percent_increase := (price_increase / sale_price) * 100
  percent_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_sale_price_increase_l3515_351515


namespace NUMINAMATH_CALUDE_marble_problem_l3515_351599

theorem marble_problem (total : ℝ) (red blue yellow purple white : ℝ) : 
  red + blue + yellow + purple + white = total ∧
  red = 0.25 * total ∧
  blue = 0.15 * total ∧
  yellow = 0.20 * total ∧
  purple = 0.05 * total ∧
  white = 50 ∧
  total = 143 →
  blue + (red / 3) = 33 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3515_351599


namespace NUMINAMATH_CALUDE_two_over_x_is_inverse_proportion_l3515_351568

/-- A function f is an inverse proportion function if there exists a constant k such that f(x) = k/x for all non-zero x. -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 2/x is an inverse proportion function. -/
theorem two_over_x_is_inverse_proportion :
  is_inverse_proportion (λ x : ℝ => 2 / x) := by
  sorry


end NUMINAMATH_CALUDE_two_over_x_is_inverse_proportion_l3515_351568


namespace NUMINAMATH_CALUDE_right_triangle_parity_l3515_351518

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_parity_l3515_351518


namespace NUMINAMATH_CALUDE_hannah_total_spending_l3515_351569

def hannah_fair_spending (initial_amount : ℝ) (ride_percent : ℝ) (game_percent : ℝ)
  (dessert_cost : ℝ) (cotton_candy_cost : ℝ) (hotdog_cost : ℝ) (keychain_cost : ℝ) : ℝ :=
  (initial_amount * ride_percent) + (initial_amount * game_percent) +
  dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost

theorem hannah_total_spending :
  hannah_fair_spending 80 0.35 0.25 7 4 5 6 = 70 := by
  sorry

end NUMINAMATH_CALUDE_hannah_total_spending_l3515_351569


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3515_351547

/-- Given a circle with area 225π cm², its diameter is 30 cm. -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3515_351547


namespace NUMINAMATH_CALUDE_circle_ratio_l3515_351531

theorem circle_ratio (a b r R : ℝ) (hr : r > 0) (hR : R > r) (h_area : π * R^2 = (a + b) / b * (π * R^2 - π * r^2)) : 
  R / r = Real.sqrt ((a + b) / a) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l3515_351531


namespace NUMINAMATH_CALUDE_like_terms_imply_exponent_relation_l3515_351537

/-- Given that -25a^(2m)b and 7a^4b^(3-n) are like terms, prove that 2m - n = 2 -/
theorem like_terms_imply_exponent_relation (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ), -25 * a^(2*m) * b = k * (7 * a^4 * b^(3-n))) : 
  2 * m - n = 2 :=
sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponent_relation_l3515_351537


namespace NUMINAMATH_CALUDE_ryan_english_time_l3515_351572

/-- The time Ryan spends on learning English, given the total time spent on learning
    English and Chinese, and the time spent on learning Chinese. -/
def time_learning_english (total_time : ℝ) (chinese_time : ℝ) : ℝ :=
  total_time - chinese_time

/-- Theorem stating that Ryan spends 2 hours learning English -/
theorem ryan_english_time :
  time_learning_english 3 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_time_l3515_351572


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3515_351500

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  ((4 * (Real.cos (12 * π / 180))^2 - 2) * Real.sin (12 * π / 180)) = 
  -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3515_351500


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l3515_351527

theorem height_on_hypotenuse (a b h : ℝ) : 
  a = 3 → b = 6 → a^2 + b^2 = (a*b/h)^2 → h = (6 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l3515_351527


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3515_351519

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 5 / 13 → 
  Nat.gcd a b = 19 → 
  Nat.lcm a b = 1235 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3515_351519


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_l3515_351558

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_l3515_351558


namespace NUMINAMATH_CALUDE_rhombus_diagonal_roots_l3515_351506

theorem rhombus_diagonal_roots (m : ℝ) : 
  let side_length : ℝ := 5
  let diagonal_equation (x : ℝ) := x^2 + (2*m - 1)*x + m^2 + 3
  (∃ (OA OB : ℝ), 
    OA^2 + OB^2 = side_length^2 ∧ 
    diagonal_equation OA = 0 ∧ 
    diagonal_equation OB = 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_roots_l3515_351506


namespace NUMINAMATH_CALUDE_negative_product_inequality_l3515_351551

theorem negative_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_inequality_l3515_351551


namespace NUMINAMATH_CALUDE_percentage_equivalence_l3515_351534

theorem percentage_equivalence : ∀ x : ℚ,
  (60 / 100) * 600 = (x / 100) * 720 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l3515_351534


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l3515_351564

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h1 : x + 3 / x = y + 3 / y) (h2 : x + y = 4) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l3515_351564


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l3515_351509

theorem range_of_a_for_quadratic_inequality 
  (h : ∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) :
  a ∈ Set.Ioi (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l3515_351509


namespace NUMINAMATH_CALUDE_seans_apples_l3515_351508

/-- Sean's apple problem -/
theorem seans_apples (initial_apples final_apples susans_apples : ℕ) :
  final_apples = initial_apples + susans_apples →
  susans_apples = 8 →
  final_apples = 17 →
  initial_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_seans_apples_l3515_351508


namespace NUMINAMATH_CALUDE_range_of_alpha_plus_three_beta_l3515_351513

theorem range_of_alpha_plus_three_beta 
  (h1 : ∀ α β : ℝ, -1 ≤ α + β ∧ α + β ≤ 1 → 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) :
  ∀ α β : ℝ, (-1 ≤ α + β ∧ α + β ≤ 1) → (1 ≤ α + 2*β ∧ α + 2*β ≤ 3) → 
  (1 ≤ α + 3*β ∧ α + 3*β ≤ 7) := by
sorry

end NUMINAMATH_CALUDE_range_of_alpha_plus_three_beta_l3515_351513


namespace NUMINAMATH_CALUDE_profit_difference_l3515_351579

def original_profit_percentage : ℝ := 0.1
def new_purchase_discount : ℝ := 0.1
def new_profit_percentage : ℝ := 0.3
def original_selling_price : ℝ := 1099.999999999999

theorem profit_difference :
  let original_purchase_price := original_selling_price / (1 + original_profit_percentage)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_discount)
  let new_selling_price := new_purchase_price * (1 + new_profit_percentage)
  new_selling_price - original_selling_price = 70 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l3515_351579


namespace NUMINAMATH_CALUDE_odd_function_property_positive_x_property_negative_x_property_l3515_351542

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x + Real.log x else x - Real.log (-x)

-- State the theorem
theorem odd_function_property (x : ℝ) : f (-x) = -f x := by sorry

-- State the positive x property
theorem positive_x_property (x : ℝ) (h : x > 0) : f x = x + Real.log x := by sorry

-- State the negative x property
theorem negative_x_property (x : ℝ) (h : x < 0) : f x = x - Real.log (-x) := by sorry

end NUMINAMATH_CALUDE_odd_function_property_positive_x_property_negative_x_property_l3515_351542


namespace NUMINAMATH_CALUDE_max_perimeter_after_cut_l3515_351577

theorem max_perimeter_after_cut (original_length original_width cut_length cut_width : ℝ) 
  (h1 : original_length = 20)
  (h2 : original_width = 16)
  (h3 : cut_length = 8)
  (h4 : cut_width = 4)
  (h5 : cut_length ≤ original_length ∧ cut_width ≤ original_width) :
  ∃ (remaining_perimeter : ℝ), 
    remaining_perimeter ≤ 2 * (original_length + original_width) + 2 * min cut_length cut_width ∧
    remaining_perimeter = 88 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_after_cut_l3515_351577


namespace NUMINAMATH_CALUDE_jonathan_exercise_distance_l3515_351554

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  total_time : ℝ

/-- Theorem stating that if Jonathan travels the same distance each day and 
    spends a total of 6 hours exercising in a week, given his speeds on different days, 
    he travels 6 miles on each exercise day. -/
theorem jonathan_exercise_distance (routine : ExerciseRoutine) 
  (h1 : routine.monday_speed = 2)
  (h2 : routine.wednesday_speed = 3)
  (h3 : routine.friday_speed = 6)
  (h4 : routine.total_time = 6)
  (h5 : ∃ d : ℝ, d > 0 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time) :
  ∃ d : ℝ, d = 6 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time := by
  sorry

end NUMINAMATH_CALUDE_jonathan_exercise_distance_l3515_351554


namespace NUMINAMATH_CALUDE_sum_of_remainders_is_93_l3515_351592

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a = b + 2 ∧ b = c + 1 ∧ c = d + 1 ∧ d = e + 1 ∧
    0 ≤ e ∧ e < 10 ∧ 2 ≤ a ∧ a ≤ 6

def valid_numbers : List ℕ :=
  [23456, 34567, 45678, 56789, 67890]

theorem sum_of_remainders_is_93 :
  (valid_numbers.map (· % 43)).sum = 93 :=
sorry

end NUMINAMATH_CALUDE_sum_of_remainders_is_93_l3515_351592


namespace NUMINAMATH_CALUDE_linear_function_triangle_area_l3515_351594

theorem linear_function_triangle_area (k : ℝ) : 
  (1/2 * 3 * |3/k| = 24) → (k = 3/16 ∨ k = -3/16) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_triangle_area_l3515_351594


namespace NUMINAMATH_CALUDE_remainder_2673_base12_div_9_l3515_351595

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2673 --/
def base12_2673 : List Nat := [2, 6, 7, 3]

theorem remainder_2673_base12_div_9 :
  (base12ToDecimal base12_2673) % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_2673_base12_div_9_l3515_351595


namespace NUMINAMATH_CALUDE_ceiling_abs_negative_l3515_351526

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_ceiling_abs_negative_l3515_351526


namespace NUMINAMATH_CALUDE_mechanic_average_earning_l3515_351588

/-- The average earning of a mechanic for a week, given specific conditions -/
theorem mechanic_average_earning
  (first_four_avg : ℝ)
  (last_four_avg : ℝ)
  (fourth_day_earning : ℝ)
  (h1 : first_four_avg = 25)
  (h2 : last_four_avg = 22)
  (h3 : fourth_day_earning = 20) :
  (4 * first_four_avg + 4 * last_four_avg - fourth_day_earning) / 7 = 24 := by
  sorry

#check mechanic_average_earning

end NUMINAMATH_CALUDE_mechanic_average_earning_l3515_351588


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l3515_351574

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l3515_351574


namespace NUMINAMATH_CALUDE_rogers_books_l3515_351501

/-- Given that Roger reads a certain number of books per week and takes a specific number of weeks to finish a series, calculate the total number of books in the series. -/
theorem rogers_books (books_per_week : ℕ) (weeks_to_finish : ℕ) : books_per_week = 6 → weeks_to_finish = 5 → books_per_week * weeks_to_finish = 30 := by
  sorry

#check rogers_books

end NUMINAMATH_CALUDE_rogers_books_l3515_351501


namespace NUMINAMATH_CALUDE_profit_after_five_days_days_for_ten_thousand_profit_l3515_351541

/-- Profit calculation function -/
def profit (x : ℝ) : ℝ :=
  (50 + 2*x) * (700 - 15*x) - 700 * 40 - 50 * x

/-- Theorem for profit after 5 days -/
theorem profit_after_five_days : profit 5 = 9250 := by sorry

/-- Theorem for days to store for 10,000 yuan profit -/
theorem days_for_ten_thousand_profit :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ profit x = 10000 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_profit_after_five_days_days_for_ten_thousand_profit_l3515_351541


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3515_351570

theorem quadratic_inequality_solution (p q : ℝ) :
  (∀ x, (1/p) * x^2 + q * x + p > 0 ↔ 2 < x ∧ x < 4) →
  p = -2 * Real.sqrt 2 ∧ q = (3/2) * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3515_351570


namespace NUMINAMATH_CALUDE_initial_kibble_amount_l3515_351524

/-- The amount of kibble Luna is supposed to eat daily -/
def daily_kibble : ℕ := 2

/-- The amount of kibble Mary gave Luna in the morning -/
def mary_morning : ℕ := 1

/-- The amount of kibble Mary gave Luna in the evening -/
def mary_evening : ℕ := 1

/-- The amount of kibble Frank gave Luna in the afternoon -/
def frank_afternoon : ℕ := 1

/-- The amount of kibble remaining in the bag the next morning -/
def remaining_kibble : ℕ := 7

/-- The theorem stating the initial amount of kibble in the bag -/
theorem initial_kibble_amount : 
  mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon + remaining_kibble = 12 := by
  sorry

#check initial_kibble_amount

end NUMINAMATH_CALUDE_initial_kibble_amount_l3515_351524


namespace NUMINAMATH_CALUDE_odd_function_g_l3515_351529

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f in terms of g -/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem odd_function_g (g : ℝ → ℝ) :
  IsOdd (f g) → (f g 1 = 1) → g = fun x ↦ x^5 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_g_l3515_351529


namespace NUMINAMATH_CALUDE_number_properties_l3515_351543

theorem number_properties :
  (∃! x : ℝ, -x = x) ∧
  (∀ x : ℝ, x ≠ 0 → (1 / x = x ↔ x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, x < -1 → 1 / x > x) ∧
  (∀ y : ℝ, y > 1 → 1 / y < y) ∧
  (∃ n : ℕ, ∀ m : ℕ, n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l3515_351543


namespace NUMINAMATH_CALUDE_barons_claim_impossible_l3515_351589

/-- Represents the number of games played by each participant -/
def GameDistribution := List ℕ

/-- A chess tournament with the given rules -/
structure ChessTournament where
  participants : ℕ
  initialGamesPerParticipant : ℕ
  claimedDistribution : GameDistribution

/-- Checks if a game distribution is valid for the given tournament rules -/
def isValidDistribution (t : ChessTournament) (d : GameDistribution) : Prop :=
  d.length = t.participants ∧
  d.sum = t.participants * t.initialGamesPerParticipant + 2 * (d.sum / 2 - t.participants * t.initialGamesPerParticipant / 2)

/-- The specific tournament described in the problem -/
def baronsTournament : ChessTournament where
  participants := 8
  initialGamesPerParticipant := 7
  claimedDistribution := [11, 11, 10, 8, 8, 8, 7, 7]

/-- Theorem stating that the Baron's claim is impossible -/
theorem barons_claim_impossible :
  ¬ isValidDistribution baronsTournament baronsTournament.claimedDistribution :=
sorry

end NUMINAMATH_CALUDE_barons_claim_impossible_l3515_351589


namespace NUMINAMATH_CALUDE_vector_relations_l3515_351582

-- Define points A, B, C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

-- Define vectors a, b, c
def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

-- Define points M and N
def M : ℝ × ℝ := (C.1 + 3 * c.1, C.2 + 3 * c.2)
def N : ℝ × ℝ := (C.1 - 2 * b.1, C.2 - 2 * b.2)

theorem vector_relations :
  (3 * a.1 + b.1 - 3 * c.1, 3 * a.2 + b.2 - 3 * c.2) = (6, -42) ∧
  a = (-b.1 - c.1, -b.2 - c.2) ∧
  M = (0, 20) ∧
  N = (9, 2) ∧
  (N.1 - M.1, N.2 - M.2) = (9, -18) := by sorry

end NUMINAMATH_CALUDE_vector_relations_l3515_351582


namespace NUMINAMATH_CALUDE_tv_show_episodes_l3515_351510

/-- Given a TV show with the following properties:
  - There were 9 seasons before a new season was announced
  - The last (10th) season has 4 more episodes than the others
  - Each episode is 0.5 hours long
  - It takes 112 hours to watch all episodes after the last season finishes
  This theorem proves that each season (except the last) has 22 episodes. -/
theorem tv_show_episodes :
  let seasons_before : ℕ := 9
  let extra_episodes_last_season : ℕ := 4
  let episode_length : ℚ := 1/2
  let total_watch_time : ℕ := 112
  let episodes_per_season : ℕ := (2 * total_watch_time - 2 * extra_episodes_last_season) / (2 * seasons_before + 2)
  episodes_per_season = 22 := by
sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l3515_351510


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3515_351598

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) → (x = 2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3515_351598


namespace NUMINAMATH_CALUDE_work_completion_time_l3515_351539

/-- Given that P persons can complete a work in 24 days, 
    prove that 2P persons can complete half of the work in 6 days. -/
theorem work_completion_time 
  (P : ℕ) -- number of persons
  (full_work : ℝ) -- amount of full work
  (h1 : P > 0) -- assumption that there's at least one person
  (h2 : full_work > 0) -- assumption that there's some work to be done
  (h3 : P * 24 * full_work = P * 24 * full_work) -- work completion condition
  : (2 * P) * 6 * (full_work / 2) = P * 24 * full_work := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3515_351539


namespace NUMINAMATH_CALUDE_prop1_prop2_prop3_false_prop4_quadratic_equation_properties_l3515_351507

-- Define a quadratic equation
def QuadraticEquation (a b c : ℤ) := {x : ℚ | a * x^2 + b * x + c = 0}

-- Define the discriminant
def Discriminant (a b c : ℤ) : ℤ := b^2 - 4*a*c

-- Define a perfect square
def IsPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

-- Proposition 1
theorem prop1 (a b c : ℤ) (ha : a ≠ 0) :
  IsPerfectSquare (Discriminant a b c) → ∃ x : ℚ, x ∈ QuadraticEquation a b c :=
sorry

-- Proposition 2
theorem prop2 (a b c : ℤ) (ha : a ≠ 0) :
  (∃ x : ℚ, x ∈ QuadraticEquation a b c) → IsPerfectSquare (Discriminant a b c) :=
sorry

-- Proposition 3 (counterexample)
theorem prop3_false : ∃ a b c : ℚ, a ≠ 0 ∧ ∃ x : ℚ, a * x^2 + b * x + c = 0 :=
sorry

-- Proposition 4
theorem prop4 (a b c : ℤ) (ha : a ≠ 0) (haodd : Odd a) (hbodd : Odd b) (hcodd : Odd c) :
  ¬∃ x : ℚ, x ∈ QuadraticEquation a b c :=
sorry

-- Main theorem combining all propositions
theorem quadratic_equation_properties :
  (∀ a b c : ℤ, a ≠ 0 → (IsPerfectSquare (Discriminant a b c) ↔ ∃ x : ℚ, x ∈ QuadraticEquation a b c)) ∧
  (∃ a b c : ℚ, a ≠ 0 ∧ ∃ x : ℚ, a * x^2 + b * x + c = 0) ∧
  (∀ a b c : ℤ, a ≠ 0 → Odd a → Odd b → Odd c → ¬∃ x : ℚ, x ∈ QuadraticEquation a b c) :=
sorry

end NUMINAMATH_CALUDE_prop1_prop2_prop3_false_prop4_quadratic_equation_properties_l3515_351507


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3515_351535

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 18

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} =
  {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3515_351535


namespace NUMINAMATH_CALUDE_total_cost_is_122_4_l3515_351503

/-- Calculates the total cost of Zoe's app usage over a year -/
def total_cost (initial_app_cost monthly_fee annual_discount in_game_cost upgrade_cost membership_discount : ℝ) : ℝ :=
  let first_two_months := 2 * monthly_fee
  let annual_plan_cost := (12 * monthly_fee) * (1 - annual_discount)
  let discounted_in_game := in_game_cost * (1 - membership_discount)
  let discounted_upgrade := upgrade_cost * (1 - membership_discount)
  initial_app_cost + first_two_months + annual_plan_cost + discounted_in_game + discounted_upgrade

/-- Theorem stating that the total cost is $122.4 given the specified conditions -/
theorem total_cost_is_122_4 :
  total_cost 5 8 0.15 10 12 0.1 = 122.4 := by
  sorry

#eval total_cost 5 8 0.15 10 12 0.1

end NUMINAMATH_CALUDE_total_cost_is_122_4_l3515_351503


namespace NUMINAMATH_CALUDE_equal_color_squares_count_l3515_351566

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid with a specific pattern of black cells -/
def Grid : Matrix (Fin 5) (Fin 5) Cell := sorry

/-- Checks if a sub-square has an equal number of black and white cells -/
def has_equal_colors (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of sub-squares with equal black and white cells -/
def count_equal_color_squares (g : Matrix (Fin 5) (Fin 5) Cell) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem equal_color_squares_count :
  count_equal_color_squares Grid = 16 :=
sorry

end NUMINAMATH_CALUDE_equal_color_squares_count_l3515_351566


namespace NUMINAMATH_CALUDE_final_row_ordered_l3515_351587

variable (m n : ℕ)
variable (C : ℕ → ℕ → ℕ)

-- C[i][j] represents the card number at row i and column j
axiom row_ordered : ∀ i j k, j < k → C i j < C i k
axiom col_ordered : ∀ i j k, i < k → C i j < C k j

theorem final_row_ordered :
  ∀ i j k, j < k → C i j < C i k :=
sorry

end NUMINAMATH_CALUDE_final_row_ordered_l3515_351587


namespace NUMINAMATH_CALUDE_height_study_concepts_l3515_351585

/-- Represents a student in the study -/
structure Student where
  height : ℝ

/-- Represents the statistical study of student heights -/
structure HeightStudy where
  allStudents : Finset Student
  sampledStudents : Finset Student
  h_sampled_subset : sampledStudents ⊆ allStudents

/-- Main theorem about the statistical concepts in the height study -/
theorem height_study_concepts (study : HeightStudy) 
  (h_total : study.allStudents.card = 480)
  (h_sampled : study.sampledStudents.card = 80) :
  (∃ (population : Finset Student), population = study.allStudents) ∧
  (∃ (sample_size : ℕ), sample_size = study.sampledStudents.card) ∧
  (∃ (sample : Finset Student), sample = study.sampledStudents) ∧
  (∃ (individual : Student), individual ∈ study.allStudents) :=
sorry

end NUMINAMATH_CALUDE_height_study_concepts_l3515_351585


namespace NUMINAMATH_CALUDE_harvest_duration_l3515_351538

theorem harvest_duration (total_earnings : ℕ) (weekly_earnings : ℕ) (h1 : total_earnings = 133) (h2 : weekly_earnings = 7) :
  total_earnings / weekly_earnings = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l3515_351538


namespace NUMINAMATH_CALUDE_factor_expression_l3515_351546

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3515_351546


namespace NUMINAMATH_CALUDE_sequence_functions_l3515_351583

/-- Arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Geometric sequence -/
def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- Theorem: The n-th term of an arithmetic sequence is a linear function of n,
    and the n-th term of a geometric sequence is an exponential function of n -/
theorem sequence_functions (a₁ d r : ℝ) (n : ℕ) :
  (∃ m b : ℝ, arithmetic_sequence a₁ d n = m * n + b) ∧
  (∃ c base : ℝ, geometric_sequence a₁ r n = c * base^n) :=
sorry

end NUMINAMATH_CALUDE_sequence_functions_l3515_351583


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l3515_351532

/-- Protein content of food items and consumption amounts -/
def collagen_protein : ℕ := 9
def protein_powder_protein : ℕ := 21
def steak_protein : ℕ := 56
def yogurt_protein : ℕ := 15
def almonds_protein : ℕ := 12

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 2
def steak_count : ℕ := 1
def yogurt_servings : ℕ := 1
def almonds_cups : ℕ := 1

/-- Total protein consumed by Arnold -/
def total_protein : ℕ :=
  collagen_protein * collagen_scoops +
  protein_powder_protein * protein_powder_scoops +
  steak_protein * steak_count +
  yogurt_protein * yogurt_servings +
  almonds_protein * almonds_cups

/-- Theorem stating that the total protein consumed is 134 grams -/
theorem arnold_protein_consumption : total_protein = 134 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l3515_351532


namespace NUMINAMATH_CALUDE_hramps_are_frafs_and_grups_l3515_351580

-- Define the sets
variable (Erogs Frafs Grups Hramps : Set α)

-- Define the conditions
variable (h1 : Erogs ⊆ Frafs)
variable (h2 : Grups ⊆ Frafs)
variable (h3 : Hramps ⊆ Erogs)
variable (h4 : Hramps ⊆ Grups)
variable (h5 : ∃ x, x ∈ Frafs ∧ x ∈ Grups)

-- Theorem to prove
theorem hramps_are_frafs_and_grups :
  Hramps ⊆ Frafs ∧ Hramps ⊆ Grups :=
sorry

end NUMINAMATH_CALUDE_hramps_are_frafs_and_grups_l3515_351580


namespace NUMINAMATH_CALUDE_total_crayons_count_l3515_351562

/-- The number of children -/
def num_children : ℕ := 7

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 8

/-- The total number of crayons -/
def total_crayons : ℕ := num_children * crayons_per_child

theorem total_crayons_count : total_crayons = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_count_l3515_351562


namespace NUMINAMATH_CALUDE_spanish_test_average_score_l3515_351597

theorem spanish_test_average_score (marco_score margaret_score average_score : ℝ) : 
  marco_score = 0.9 * average_score →
  margaret_score = marco_score + 5 →
  margaret_score = 86 →
  average_score = 90 := by
sorry

end NUMINAMATH_CALUDE_spanish_test_average_score_l3515_351597


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3515_351560

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3515_351560


namespace NUMINAMATH_CALUDE_compute_expression_l3515_351505

theorem compute_expression : 8 * (1/3)^4 = 8/81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3515_351505


namespace NUMINAMATH_CALUDE_distance_to_reflection_l3515_351584

/-- Given a point F with coordinates (-5, 3), prove that the distance between F
    and its reflection over the y-axis is 10. -/
theorem distance_to_reflection (F : ℝ × ℝ) : 
  F = (-5, 3) → ‖F - (5, 3)‖ = 10 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_l3515_351584


namespace NUMINAMATH_CALUDE_day_after_53_friday_is_tuesday_l3515_351530

/-- The days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

theorem day_after_53_friday_is_tuesday :
  dayAfter DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_day_after_53_friday_is_tuesday_l3515_351530


namespace NUMINAMATH_CALUDE_expression_value_l3515_351581

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - 2*z = 0)
  (eq2 : x + 3*y - 28*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y*z) / (y^2 + z^2) = 280/37 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3515_351581


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l3515_351556

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
    n > 2 → 
    (n - 2) * (n - 1) * n = 15 * (3 * n - 3) → 
    (n - 2)^3 + (n - 1)^3 + n^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l3515_351556


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3515_351557

/-- Given two lines that are parallel, prove that the value of 'a' is 1/2 -/
theorem parallel_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (2 * a * y - 1 = 0 ↔ y = 1 / (2 * a))) →
  (∀ x y : ℝ, ((3 * a - 1) * x + y - 1 = 0 ↔ y = -(3 * a - 1) * x + 1)) →
  (∀ x y : ℝ, 2 * a * y - 1 = 0 → (3 * a - 1) * x + y - 1 = 0 → x = 0) →
  a = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_a_value_l3515_351557


namespace NUMINAMATH_CALUDE_dodecagon_area_times_hundred_l3515_351548

/-- The area of a regular dodecagon inscribed in a unit circle -/
def dodecagonArea : ℝ := 3

/-- 100 times the area of a regular dodecagon inscribed in a unit circle -/
def hundredTimesDodecagonArea : ℝ := 100 * dodecagonArea

theorem dodecagon_area_times_hundred : hundredTimesDodecagonArea = 300 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_area_times_hundred_l3515_351548
