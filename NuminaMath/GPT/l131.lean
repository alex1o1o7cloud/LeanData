import Mathlib

namespace inequality_inequation_l131_13145

theorem inequality_inequation (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x + y + z = 1) :
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 :=
by
  sorry

end inequality_inequation_l131_13145


namespace radius_of_tangent_circle_l131_13186

def is_tangent_coor_axes_and_leg (r : ℝ) : Prop :=
  -- Circle with radius r is tangent to coordinate axes and one leg of the triangle
  ∃ O B C : ℝ × ℝ, 
  -- Conditions: centers and tangency
  O = (r, r) ∧ 
  B = (0, 2) ∧ 
  C = (2, 0) ∧ 
  r = 1

theorem radius_of_tangent_circle :
  ∀ r : ℝ, is_tangent_coor_axes_and_leg r → r = 1 :=
by
  sorry

end radius_of_tangent_circle_l131_13186


namespace inequality_holds_l131_13110

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) :=
sorry

end inequality_holds_l131_13110


namespace area_ratio_of_regular_polygons_l131_13198

noncomputable def area_ratio (r : ℝ) : ℝ :=
  let A6 := (3 * Real.sqrt 3 / 2) * r^2
  let s8 := r * Real.sqrt (2 - Real.sqrt 2)
  let A8 := 2 * (1 + Real.sqrt 2) * (s8 ^ 2)
  A8 / A6

theorem area_ratio_of_regular_polygons (r : ℝ) :
  area_ratio r = 4 * (1 + Real.sqrt 2) * (2 - Real.sqrt 2) / (3 * Real.sqrt 3) :=
  sorry

end area_ratio_of_regular_polygons_l131_13198


namespace eggs_divided_l131_13164

theorem eggs_divided (boxes : ℝ) (eggs_per_box : ℝ) (total_eggs : ℝ) :
  boxes = 2.0 → eggs_per_box = 1.5 → total_eggs = boxes * eggs_per_box → total_eggs = 3.0 :=
by
  intros
  sorry

end eggs_divided_l131_13164


namespace heaviest_person_is_42_27_l131_13142

-- Define the main parameters using the conditions
def heaviest_person_weight (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real) (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) : Real :=
  let h := P 2 + 7.7
  h

-- State the theorem
theorem heaviest_person_is_42_27 (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real)
  (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) :
  heaviest_person_weight M P Q H L S = 42.27 :=
sorry

end heaviest_person_is_42_27_l131_13142


namespace remainder_div_741147_6_l131_13152

theorem remainder_div_741147_6 : 741147 % 6 = 3 :=
by
  sorry

end remainder_div_741147_6_l131_13152


namespace medicine_dosage_l131_13103

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l131_13103


namespace train_speed_is_180_kmh_l131_13195

-- Defining the conditions
def train_length : ℕ := 1500  -- meters
def platform_length : ℕ := 1500  -- meters
def crossing_time : ℕ := 1  -- minute

-- Function to compute the speed in km/hr
def speed_in_km_per_hr (length : ℕ) (time : ℕ) : ℕ :=
  let distance := length + length
  let speed_m_per_min := distance / time
  let speed_km_per_hr := speed_m_per_min * 60 / 1000
  speed_km_per_hr

-- The main theorem we need to prove
theorem train_speed_is_180_kmh :
  speed_in_km_per_hr train_length crossing_time = 180 :=
by
  sorry

end train_speed_is_180_kmh_l131_13195


namespace size_of_can_of_concentrate_l131_13134

theorem size_of_can_of_concentrate
  (can_to_water_ratio : ℕ := 1 + 3)
  (servings_needed : ℕ := 320)
  (serving_size : ℕ := 6)
  (total_volume : ℕ := servings_needed * serving_size) :
  ∃ C : ℕ, C = total_volume / can_to_water_ratio :=
by
  sorry

end size_of_can_of_concentrate_l131_13134


namespace area_of_defined_region_eq_14_point_4_l131_13135

def defined_region (x y : ℝ) : Prop :=
  |5 * x - 20| + |3 * y + 9| ≤ 6

def region_area : ℝ :=
  14.4

theorem area_of_defined_region_eq_14_point_4 :
  (∃ (x y : ℝ), defined_region x y) → region_area = 14.4 :=
by
  sorry

end area_of_defined_region_eq_14_point_4_l131_13135


namespace product_eq_1519000000_div_6561_l131_13193

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end product_eq_1519000000_div_6561_l131_13193


namespace total_teachers_in_all_departments_is_637_l131_13166

noncomputable def total_teachers : ℕ :=
  let major_departments := 9
  let minor_departments := 8
  let teachers_per_major := 45
  let teachers_per_minor := 29
  (major_departments * teachers_per_major) + (minor_departments * teachers_per_minor)

theorem total_teachers_in_all_departments_is_637 : total_teachers = 637 := 
  by
  sorry

end total_teachers_in_all_departments_is_637_l131_13166


namespace line_through_intersection_of_circles_l131_13107

theorem line_through_intersection_of_circles 
  (x y : ℝ)
  (C1 : x^2 + y^2 = 10)
  (C2 : (x-1)^2 + (y-3)^2 = 20) :
  x + 3 * y = 0 :=
sorry

end line_through_intersection_of_circles_l131_13107


namespace remainder_777_777_mod_13_l131_13126

theorem remainder_777_777_mod_13 : (777^777) % 13 = 1 := by
  sorry

end remainder_777_777_mod_13_l131_13126


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l131_13157

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l131_13157


namespace point_A_in_QuadrantIII_l131_13127

-- Define the Cartesian Point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point being in Quadrant III
def inQuadrantIII (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Given point A
def A : Point := { x := -1, y := -2 }

-- The theorem stating that point A lies in Quadrant III
theorem point_A_in_QuadrantIII : inQuadrantIII A :=
  by
    sorry

end point_A_in_QuadrantIII_l131_13127


namespace det_dilation_matrix_l131_13177

section DilationMatrixProof

def E : Matrix (Fin 3) (Fin 3) ℝ := !![5, 0, 0; 0, 5, 0; 0, 0, 5]

theorem det_dilation_matrix :
  Matrix.det E = 125 :=
by {
  sorry
}

end DilationMatrixProof

end det_dilation_matrix_l131_13177


namespace simplify_polynomial_subtraction_l131_13187

/--
  Given the polynomials (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) and (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5),
  prove that their difference simplifies to x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3.
-/
theorem simplify_polynomial_subtraction  (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) = x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 :=
sorry

end simplify_polynomial_subtraction_l131_13187


namespace Dan_reaches_Cate_in_25_seconds_l131_13104

theorem Dan_reaches_Cate_in_25_seconds
  (d : ℝ) (v_d : ℝ) (v_c : ℝ)
  (h1 : d = 50)
  (h2 : v_d = 8)
  (h3 : v_c = 6) :
  (d / (v_d - v_c) = 25) :=
by
  sorry

end Dan_reaches_Cate_in_25_seconds_l131_13104


namespace a_pow_m_minus_a_pow_n_divisible_by_30_l131_13149

theorem a_pow_m_minus_a_pow_n_divisible_by_30
  (a m n k : ℕ)
  (h_n_ge_two : n ≥ 2)
  (h_m_gt_n : m > n)
  (h_m_n_diff : m = n + 4 * k) :
  30 ∣ (a ^ m - a ^ n) :=
sorry

end a_pow_m_minus_a_pow_n_divisible_by_30_l131_13149


namespace cubic_identity_l131_13191

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l131_13191


namespace rectangle_new_area_l131_13181

theorem rectangle_new_area (original_area : ℝ) (new_length_factor : ℝ) (new_width_factor : ℝ) 
  (h1 : original_area = 560) (h2 : new_length_factor = 1.2) (h3 : new_width_factor = 0.85) : 
  new_length_factor * new_width_factor * original_area = 571 := 
by 
  sorry

end rectangle_new_area_l131_13181


namespace hannahs_trip_cost_l131_13113

noncomputable def calculate_gas_cost (initial_odometer final_odometer : ℕ) (fuel_economy_mpg : ℚ) (cost_per_gallon : ℚ) : ℚ :=
  let distance := final_odometer - initial_odometer
  let fuel_used := distance / fuel_economy_mpg
  fuel_used * cost_per_gallon

theorem hannahs_trip_cost :
  calculate_gas_cost 36102 36131 32 (385 / 100) = 276 / 100 :=
by
  sorry

end hannahs_trip_cost_l131_13113


namespace final_number_after_increase_l131_13143

-- Define the original number and the percentage increase
def original_number : ℕ := 70
def increase_percentage : ℝ := 0.50

-- Define the function to calculate the final number after the increase
def final_number : ℝ := original_number * (1 + increase_percentage)

-- The proof statement that the final number is 105
theorem final_number_after_increase : final_number = 105 :=
by
  sorry

end final_number_after_increase_l131_13143


namespace infinite_geometric_series_sum_l131_13119

theorem infinite_geometric_series_sum :
  let a := (4 : ℚ) / 3
  let r := -(9 : ℚ) / 16
  (a / (1 - r)) = (64 : ℚ) / 75 :=
by
  sorry

end infinite_geometric_series_sum_l131_13119


namespace investment_ratio_same_period_l131_13137

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end investment_ratio_same_period_l131_13137


namespace complex_subtraction_l131_13125

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 1 + I

theorem complex_subtraction : z1 - z2 = 2 + 3 * I := by
  sorry

end complex_subtraction_l131_13125


namespace fractions_order_l131_13100

theorem fractions_order:
  (20 / 15) < (25 / 18) ∧ (25 / 18) < (23 / 16) ∧ (23 / 16) < (21 / 14) :=
by
  sorry

end fractions_order_l131_13100


namespace division_of_203_by_single_digit_l131_13185

theorem division_of_203_by_single_digit (d : ℕ) (h : 1 ≤ d ∧ d < 10) : 
  ∃ q : ℕ, q = 203 / d ∧ (10 ≤ q ∧ q < 100 ∨ 100 ≤ q ∧ q < 1000) := 
by
  sorry

end division_of_203_by_single_digit_l131_13185


namespace grandfather_7_times_older_after_8_years_l131_13155

theorem grandfather_7_times_older_after_8_years :
  ∃ x : ℕ, ∀ (g_age ng_age : ℕ), 50 < g_age ∧ g_age < 90 ∧ g_age = 31 * ng_age → g_age + x = 7 * (ng_age + x) → x = 8 :=
by
  sorry

end grandfather_7_times_older_after_8_years_l131_13155


namespace mean_transformation_l131_13122

theorem mean_transformation (x1 x2 x3 x4 : ℝ)
                            (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4)
                            (s2 : ℝ)
                            (h_var : s2 = (1 / 4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
                            (x1 + 2 + x2 + 2 + x3 + 2 + x4 + 2) / 4 = 4 :=
by
  sorry

end mean_transformation_l131_13122


namespace log_relationship_l131_13154

noncomputable def log_m (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem log_relationship (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  log_m m 0.3 > log_m m 0.5 :=
by
  sorry

end log_relationship_l131_13154


namespace large_box_count_l131_13180

variable (x y : ℕ)

theorem large_box_count (h₁ : x + y = 21) (h₂ : 120 * x + 80 * y = 2000) : x = 8 := by
  sorry

end large_box_count_l131_13180


namespace trapezoid_PR_length_l131_13161

noncomputable def PR_length (PQ RS QS PR : ℝ) (angle_QSP angle_SRP : ℝ) : Prop :=
  PQ < RS ∧ 
  QS = 2 ∧ 
  angle_QSP = 30 ∧ 
  angle_SRP = 60 ∧ 
  RS / PQ = 7 / 3 ∧ 
  PR = 8 / 3

theorem trapezoid_PR_length (PQ RS QS PR : ℝ) 
  (angle_QSP angle_SRP : ℝ) 
  (h1 : PQ < RS) 
  (h2 : QS = 2) 
  (h3 : angle_QSP = 30) 
  (h4 : angle_SRP = 60) 
  (h5 : RS / PQ = 7 / 3) :
  PR = 8 / 3 := 
by
  sorry

end trapezoid_PR_length_l131_13161


namespace consecutive_numbers_sum_39_l131_13146

theorem consecutive_numbers_sum_39 (n : ℕ) (hn : n + (n + 1) = 39) : n + 1 = 20 :=
sorry

end consecutive_numbers_sum_39_l131_13146


namespace car_speed_in_first_hour_l131_13118

theorem car_speed_in_first_hour (x : ℝ) 
  (second_hour_speed : ℝ := 40)
  (average_speed : ℝ := 60)
  (h : (x + second_hour_speed) / 2 = average_speed) :
  x = 80 := 
by
  -- Additional steps needed to solve this theorem
  sorry

end car_speed_in_first_hour_l131_13118


namespace solve_quadratic_eq_l131_13109

theorem solve_quadratic_eq : ∃ (a b : ℕ), a = 145 ∧ b = 7 ∧ a + b = 152 ∧ 
  ∀ x, x = Real.sqrt a - b → x^2 + 14 * x = 96 :=
by 
  use 145, 7
  simp
  sorry

end solve_quadratic_eq_l131_13109


namespace maximum_elephants_l131_13132

theorem maximum_elephants (e_1 e_2 : ℕ) :
  (∃ e_1 e_2 : ℕ, 28 * e_1 + 37 * e_2 = 1036 ∧ (∀ k, 28 * e_1 + 37 * e_2 = k → k ≤ 1036 )) → 
  28 * e_1 + 37 * e_2 = 1036 :=
sorry

end maximum_elephants_l131_13132


namespace fraction_replaced_l131_13150

theorem fraction_replaced :
  ∃ x : ℚ, (0.60 * (1 - x) + 0.25 * x = 0.35) ∧ x = 5 / 7 := by
    sorry

end fraction_replaced_l131_13150


namespace work_completion_l131_13151

theorem work_completion (a b : Type) (work_done_together work_done_by_a work_done_by_b : ℝ) 
  (h1 : work_done_together = 1 / 12) 
  (h2 : work_done_by_a = 1 / 20) 
  (h3 : work_done_by_b = work_done_together - work_done_by_a) : 
  work_done_by_b = 1 / 30 :=
by
  sorry

end work_completion_l131_13151


namespace problem1_solution_l131_13159

theorem problem1_solution : ∀ x : ℝ, x^2 - 6 * x + 9 = (5 - 2 * x)^2 → (x = 8/3 ∨ x = 2) :=
sorry

end problem1_solution_l131_13159


namespace license_plate_difference_l131_13163

theorem license_plate_difference :
  (26^3 * 10^4) - (26^4 * 10^3) = -281216000 :=
by
  sorry

end license_plate_difference_l131_13163


namespace wealth_ratio_l131_13120

theorem wealth_ratio (W P : ℝ) (hW_pos : 0 < W) (hP_pos : 0 < P) :
  let wX := 0.54 * W / (0.40 * P)
  let wY := 0.30 * W / (0.20 * P)
  wX / wY = 0.9 := 
by
  sorry

end wealth_ratio_l131_13120


namespace polynomial_remainder_l131_13173

theorem polynomial_remainder (x : ℝ) :
  (x^4 + 3 * x^2 - 4) % (x^2 + 2) = x^2 - 4 :=
sorry

end polynomial_remainder_l131_13173


namespace lisa_likes_only_last_digit_zero_l131_13133

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8

def is_divisible_by_5_and_2 (n : ℕ) : Prop :=
  is_divisible_by_5 n ∧ is_divisible_by_2 n

theorem lisa_likes_only_last_digit_zero : ∀ n, is_divisible_by_5_and_2 n → n % 10 = 0 :=
by
  sorry

end lisa_likes_only_last_digit_zero_l131_13133


namespace probability_three_heads_in_a_row_l131_13188

theorem probability_three_heads_in_a_row (h : ℝ) (p_head : h = 1/2) (ind_flips : ∀ (n : ℕ), true) : 
  (1/2 * 1/2 * 1/2 = 1/8) :=
by
  sorry

end probability_three_heads_in_a_row_l131_13188


namespace average_M_possibilities_l131_13170

theorem average_M_possibilities (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
    (12 = (8 + 15 + M) / 3) ∨ (15 = (8 + 15 + M) / 3) :=
  sorry

end average_M_possibilities_l131_13170


namespace housewife_saving_l131_13115

theorem housewife_saving :
  let total_money := 450
  let groceries_fraction := 3 / 5
  let household_items_fraction := 1 / 6
  let personal_care_items_fraction := 1 / 10
  let groceries_expense := groceries_fraction * total_money
  let household_items_expense := household_items_fraction * total_money
  let personal_care_items_expense := personal_care_items_fraction * total_money
  let total_expense := groceries_expense + household_items_expense + personal_care_items_expense
  total_money - total_expense = 60 :=
by
  sorry

end housewife_saving_l131_13115


namespace second_puppy_weight_l131_13116

variables (p1 p2 c1 c2 : ℝ)

-- Conditions from the problem statement
axiom h1 : p1 + p2 + c1 + c2 = 36
axiom h2 : p1 + c2 = 3 * c1
axiom h3 : p1 + c1 = c2
axiom h4 : p2 = 1.5 * p1

-- The question to prove: how much does the second puppy weigh
theorem second_puppy_weight : p2 = 108 / 11 :=
by sorry

end second_puppy_weight_l131_13116


namespace function_intersection_at_most_one_l131_13129

theorem function_intersection_at_most_one (f : ℝ → ℝ) (a : ℝ) :
  ∃! b, f b = a := sorry

end function_intersection_at_most_one_l131_13129


namespace tangent_line_parallel_l131_13131

theorem tangent_line_parallel (x y : ℝ) (h_parab : y = 2 * x^2) (h_parallel : ∃ (m b : ℝ), 4 * x - y + b = 0) : 
    (∃ b, 4 * x - y - b = 0) := 
by
  sorry

end tangent_line_parallel_l131_13131


namespace sum_mod_13_l131_13165

theorem sum_mod_13 :
  (9023 % 13 = 5) → 
  (9024 % 13 = 6) → 
  (9025 % 13 = 7) → 
  (9026 % 13 = 8) → 
  ((9023 + 9024 + 9025 + 9026) % 13 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_mod_13_l131_13165


namespace together_work_days_l131_13121

theorem together_work_days (A B C : ℕ) (nine_days : A = 9) (eighteen_days : B = 18) (twelve_days : C = 12) :
  (1 / A + 1 / B + 1 / C) = 1 / 4 :=
by
  sorry

end together_work_days_l131_13121


namespace max_marks_400_l131_13108

theorem max_marks_400 {M : ℝ} (h : 0.45 * M = 150 + 30) : M = 400 := 
by
  sorry

end max_marks_400_l131_13108


namespace quiz_competition_l131_13184

theorem quiz_competition (x : ℕ) :
  (10 * x - 4 * (20 - x) ≥ 88) ↔ (x ≥ 12) :=
by 
  sorry

end quiz_competition_l131_13184


namespace inequality_system_solution_l131_13112

theorem inequality_system_solution (x : ℝ) : 
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 :=
by
  sorry

end inequality_system_solution_l131_13112


namespace find_k_l131_13156

theorem find_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 :=
sorry

end find_k_l131_13156


namespace smallest_angle_measure_in_triangle_l131_13175

theorem smallest_angle_measure_in_triangle (a b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : c > 2 * Real.sqrt 2) :
  ∃ x : ℝ, x = 140 ∧ C < x :=
sorry

end smallest_angle_measure_in_triangle_l131_13175


namespace proof_seq_l131_13114

open Nat

-- Definition of sequence {a_n}
def seq_a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * seq_a n

-- Definition of sum S_n of sequence {b_n}
def sum_S : ℕ → ℕ
| 0 => 0
| n + 1 => sum_S n + (2^n)

-- Definition of sequence {b_n}
def seq_b : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * seq_b n

-- Definition of sequence {c_n}
def seq_c (n : ℕ) : ℕ := seq_b n * log 3 (seq_a n) -- Note: log base 3

-- Sum of first n terms of {c_n}
def sum_T : ℕ → ℕ
| 0 => 0
| n + 1 => sum_T n + seq_c n

-- Proof statement
theorem proof_seq (n : ℕ) :
  (seq_a n = 3 ^ n) ∧
  (2 * seq_b n - 1 = sum_S 0 * sum_S n) ∧
  (sum_T n = (n - 2) * 2 ^ (n + 2)) :=
sorry

end proof_seq_l131_13114


namespace extreme_value_at_one_l131_13197

noncomputable def f (x : ℝ) (a : ℝ) := (x^2 + a) / (x + 1)

theorem extreme_value_at_one (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y-1) < δ → abs (f y a - f 1 a) < ε)) →
  a = 3 :=
by
  sorry

end extreme_value_at_one_l131_13197


namespace min_value_of_a_plus_b_minus_c_l131_13174

open Real

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ) :
  (∀ (x y : ℝ), 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) →
  (∃ c_min, c_min = 2 ∧ ∀ c', c' = a + b - c → c' ≥ c_min) :=
by
  sorry

end min_value_of_a_plus_b_minus_c_l131_13174


namespace accessories_per_doll_l131_13192

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end accessories_per_doll_l131_13192


namespace middle_number_is_9_l131_13162

-- Define the problem conditions
variable (x y z : ℕ)

-- Lean proof statement
theorem middle_number_is_9 
  (h1 : x + y = 16)
  (h2 : x + z = 21)
  (h3 : y + z = 23)
  (h4 : x < y)
  (h5 : y < z) : y = 9 :=
by
  sorry

end middle_number_is_9_l131_13162


namespace team_combination_count_l131_13105

theorem team_combination_count (n k : ℕ) (hn : n = 7) (hk : k = 4) :
  ∃ m, m = Nat.choose n k ∧ m = 35 :=
by
  sorry

end team_combination_count_l131_13105


namespace find_m_l131_13189

variables (AB AC AD : ℝ × ℝ)
variables (m : ℝ)

-- Definitions of vectors
def vector_AB : ℝ × ℝ := (-1, 2)
def vector_AC : ℝ × ℝ := (2, 3)
def vector_AD (m : ℝ) : ℝ × ℝ := (m, -3)

-- Conditions
def collinear (B C D : ℝ × ℝ) : Prop := ∃ k : ℝ, B = k • C ∨ C = k • D ∨ D = k • B

-- Main statement to prove
theorem find_m (h1 : vector_AB = (-1, 2))
               (h2 : vector_AC = (2, 3))
               (h3 : vector_AD m = (m, -3))
               (h4 : collinear vector_AB vector_AC (vector_AD m)) :
  m = -16 :=
sorry

end find_m_l131_13189


namespace proper_sets_exist_l131_13144

def proper_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, (1 ≤ w ∧ w ≤ 500) → ∃ (used_weights : List ℕ), (used_weights ⊆ weights) ∧ (used_weights.sum = w ∧ ∀ (alternative_weights : List ℕ), (alternative_weights ⊆ weights ∧ alternative_weights.sum = w) → used_weights = alternative_weights)

theorem proper_sets_exist (weights : List ℕ) :
  (weights.sum = 500) → 
  ∃ (sets : List (List ℕ)), sets.length = 3 ∧ (∀ s ∈ sets, proper_set s) :=
by
  sorry

end proper_sets_exist_l131_13144


namespace range_of_m_l131_13199

variable {m x : ℝ}

theorem range_of_m (h : ∀ x, -1 < x ∧ x < 4 ↔ x > 2 * m ^ 2 - 3) : m ∈ [-1, 1] :=
sorry

end range_of_m_l131_13199


namespace chairperson_and_committee_ways_l131_13194

-- Definitions based on conditions
def total_people : ℕ := 10
def ways_to_choose_chairperson : ℕ := total_people
def ways_to_choose_committee (remaining_people : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose remaining_people committee_size

-- The resulting theorem
theorem chairperson_and_committee_ways :
  ways_to_choose_chairperson * ways_to_choose_committee (total_people - 1) 3 = 840 :=
by
  sorry

end chairperson_and_committee_ways_l131_13194


namespace binary101_to_decimal_l131_13136

theorem binary101_to_decimal :
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  binary_101 = 5 := 
by
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show binary_101 = 5
  sorry

end binary101_to_decimal_l131_13136


namespace john_guests_count_l131_13176

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def additional_fractional_guests : ℝ := 0.60
def total_cost_when_wife_gets_her_way : ℕ := 50000

theorem john_guests_count (G : ℕ) :
  venue_cost + cost_per_guest * (1 + additional_fractional_guests) * G = 
  total_cost_when_wife_gets_her_way →
  G = 50 :=
by
  sorry

end john_guests_count_l131_13176


namespace measurement_units_correct_l131_13148

structure Measurement (A : Type) where
  value : A
  unit : String

def height_of_desk : Measurement ℕ := ⟨70, "centimeters"⟩
def weight_of_apple : Measurement ℕ := ⟨240, "grams"⟩
def duration_of_soccer_game : Measurement ℕ := ⟨90, "minutes"⟩
def dad_daily_work_duration : Measurement ℕ := ⟨8, "hours"⟩

theorem measurement_units_correct :
  height_of_desk.unit = "centimeters" ∧
  weight_of_apple.unit = "grams" ∧
  duration_of_soccer_game.unit = "minutes" ∧
  dad_daily_work_duration.unit = "hours" :=
by
  sorry

end measurement_units_correct_l131_13148


namespace car_speed_problem_l131_13130

theorem car_speed_problem (x : ℝ) (h1 : ∀ x, x + 30 / 2 = 65) : x = 100 :=
by
  sorry

end car_speed_problem_l131_13130


namespace truck_travel_distance_l131_13123

def original_distance : ℝ := 300
def original_gas : ℝ := 10
def increased_efficiency_percent : ℝ := 1.10
def new_gas : ℝ := 15

theorem truck_travel_distance :
  let original_efficiency := original_distance / original_gas;
  let new_efficiency := original_efficiency * increased_efficiency_percent;
  let distance := new_gas * new_efficiency;
  distance = 495 :=
by
  sorry

end truck_travel_distance_l131_13123


namespace profit_margin_increase_l131_13158

theorem profit_margin_increase (CP : ℝ) (SP : ℝ) (NSP : ℝ) (initial_margin : ℝ) (desired_margin : ℝ) :
  initial_margin = 0.25 → desired_margin = 0.40 → SP = (1 + initial_margin) * CP → NSP = (1 + desired_margin) * CP →
  ((NSP - SP) / SP) * 100 = 12 := 
by 
  intros h1 h2 h3 h4
  sorry

end profit_margin_increase_l131_13158


namespace calc_result_neg2xy2_pow3_l131_13117

theorem calc_result_neg2xy2_pow3 (x y : ℝ) : 
  (-2 * x * y^2)^3 = -8 * x^3 * y^6 := 
by 
  sorry

end calc_result_neg2xy2_pow3_l131_13117


namespace smallest_four_digit_number_divisible_by_smallest_primes_l131_13147

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l131_13147


namespace book_price_is_correct_l131_13111

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l131_13111


namespace find_omega_l131_13141

noncomputable def f (x : ℝ) (ω φ : ℝ) := Real.sin (ω * x + φ)

theorem find_omega (ω φ : ℝ) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
  (h_symm : ∀ x : ℝ, f (3 * π / 4 + x) ω φ = f (3 * π / 4 - x) ω φ)
  (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ω φ ≤ f x2 ω φ) :
  ω = 2 / 3 ∨ ω = 2 :=
sorry

end find_omega_l131_13141


namespace tan_add_l131_13138

theorem tan_add (α β : ℝ) (h1 : Real.tan (α - π / 6) = 3 / 7) (h2 : Real.tan (π / 6 + β) = 2 / 5) : Real.tan (α + β) = 1 := by
  sorry

end tan_add_l131_13138


namespace unknown_number_value_l131_13169

theorem unknown_number_value (x n : ℝ) (h1 : 0.75 / x = n / 8) (h2 : x = 2) : n = 3 :=
by
  sorry

end unknown_number_value_l131_13169


namespace initial_plan_days_l131_13190

-- Define the given conditions in Lean
variables (D : ℕ) -- Initial planned days for completing the job
variables (P : ℕ) -- Number of people initially hired
variables (Q : ℕ) -- Number of people fired
variables (W1 : ℚ) -- Portion of the work done before firing people
variables (D1 : ℕ) -- Days taken to complete W1 portion of work
variables (W2 : ℚ) -- Remaining portion of the work done after firing people
variables (D2 : ℕ) -- Days taken to complete W2 portion of work

-- Conditions from the problem
axiom h1 : P = 10
axiom h2 : Q = 2
axiom h3 : W1 = 1 / 4
axiom h4 : D1 = 20
axiom h5 : W2 = 3 / 4
axiom h6 : D2 = 75

-- The main theorem that proves the total initially planned days were 80
theorem initial_plan_days : D = 80 :=
sorry

end initial_plan_days_l131_13190


namespace post_office_mail_in_six_months_l131_13124

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end post_office_mail_in_six_months_l131_13124


namespace compute_diameter_of_garden_roller_l131_13183

noncomputable def diameter_of_garden_roller (length : ℝ) (area_per_revolution : ℝ) (pi : ℝ) :=
  let radius := (area_per_revolution / (2 * pi * length))
  2 * radius

theorem compute_diameter_of_garden_roller :
  diameter_of_garden_roller 3 (66 / 5) (22 / 7) = 1.4 := by
  sorry

end compute_diameter_of_garden_roller_l131_13183


namespace algebraic_expression_value_l131_13196

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 23 - 1) : x^2 + 2 * x + 2 = 24 :=
by
  -- Start of the proof
  sorry -- Proof is omitted as per instructions

end algebraic_expression_value_l131_13196


namespace arithmetic_geometric_sequence_l131_13102

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 3 * 2 = a 1 + 2 * d)
  (h4 : a 4 = a 1 + 3 * d)
  (h5 : a 8 = a 1 + 7 * d)
  (h_geo : (a 1 + 3 * d) ^ 2 = (a 1 + 2 * d) * (a 1 + 7 * d))
  (h_sum : S 4 = (a 1 * 4) + (d * (4 * 3 / 2))) :
  a 1 * d < 0 ∧ d * S 4 < 0 :=
by sorry

end arithmetic_geometric_sequence_l131_13102


namespace tank_empty_time_l131_13179

def tank_capacity : ℝ := 6480
def leak_time : ℝ := 6
def inlet_rate_per_minute : ℝ := 4.5
def inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

theorem tank_empty_time : tank_capacity / (tank_capacity / leak_time - inlet_rate_per_hour) = 8 := 
by
  sorry

end tank_empty_time_l131_13179


namespace find_other_root_l131_13139

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l131_13139


namespace opposite_of_2023_is_minus_2023_l131_13171

def opposite (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_2023_is_minus_2023 : opposite 2023 (-2023) :=
by
  sorry

end opposite_of_2023_is_minus_2023_l131_13171


namespace average_of_original_set_l131_13182

theorem average_of_original_set
  (A : ℝ)
  (n : ℕ)
  (B : ℝ)
  (h1 : n = 7)
  (h2 : B = 5 * A)
  (h3 : B / n = 100)
  : A = 20 :=
by
  sorry

end average_of_original_set_l131_13182


namespace distance_between_circle_centers_l131_13140

-- Define the given side lengths of the triangle
def DE : ℝ := 12
def DF : ℝ := 15
def EF : ℝ := 9

-- Define the problem and assertion
theorem distance_between_circle_centers :
  ∃ d : ℝ, d = 12 * Real.sqrt 13 :=
sorry

end distance_between_circle_centers_l131_13140


namespace speed_boat_upstream_l131_13153

-- Define the conditions provided in the problem
def V_b : ℝ := 8.5  -- Speed of the boat in still water (in km/hr)
def V_downstream : ℝ := 13 -- Speed of the boat downstream (in km/hr)
def V_s : ℝ := V_downstream - V_b  -- Speed of the stream (in km/hr), derived from V_downstream and V_b
def V_upstream (V_b : ℝ) (V_s : ℝ) : ℝ := V_b - V_s  -- Speed of the boat upstream (in km/hr)

-- Statement to prove: the speed of the boat upstream is 4 km/hr
theorem speed_boat_upstream :
  V_upstream V_b V_s = 4 :=
by
  -- This line is for illustration, replace with an actual proof
  sorry

end speed_boat_upstream_l131_13153


namespace diff_of_cubes_is_sum_of_squares_l131_13128

theorem diff_of_cubes_is_sum_of_squares (n : ℤ) : 
  (n+2)^3 - n^3 = n^2 + (n+2)^2 + (2*n+2)^2 := 
by sorry

end diff_of_cubes_is_sum_of_squares_l131_13128


namespace average_weight_of_eight_boys_l131_13172

theorem average_weight_of_eight_boys :
  let avg16 := 50.25
  let avg24 := 48.55
  let total_weight_16 := 16 * avg16
  let total_weight_all := 24 * avg24
  let W := (total_weight_all - total_weight_16) / 8
  W = 45.15 :=
by
  sorry

end average_weight_of_eight_boys_l131_13172


namespace expression_value_l131_13106

-- Define the given condition as an assumption
variable (x : ℝ)
variable (h : 2 * x^2 + 3 * x - 1 = 7)

-- Define the target expression and the required result
theorem expression_value :
  4 * x^2 + 6 * x + 9 = 25 :=
by
  sorry

end expression_value_l131_13106


namespace smallest_m_plus_n_l131_13160

theorem smallest_m_plus_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 3 * m^3 = 5 * n^5) : m + n = 720 :=
by
  sorry

end smallest_m_plus_n_l131_13160


namespace unknown_rate_of_blankets_l131_13168

theorem unknown_rate_of_blankets (x : ℝ) :
  2 * 100 + 5 * 150 + 2 * x = 9 * 150 → x = 200 :=
by
  sorry

end unknown_rate_of_blankets_l131_13168


namespace john_children_l131_13101

def total_notebooks (john_notebooks : ℕ) (wife_notebooks : ℕ) (children : ℕ) := 
  2 * children + 5 * children

theorem john_children (c : ℕ) (h : total_notebooks 2 5 c = 21) :
  c = 3 :=
sorry

end john_children_l131_13101


namespace cos_eq_cos_of_n_l131_13167

theorem cos_eq_cos_of_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (283 * Real.pi / 180)) : n = 77 :=
by sorry

end cos_eq_cos_of_n_l131_13167


namespace solve_system_of_equations_l131_13178

theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : 2 * x + y = 21) : x + y = 9 := by
  sorry

end solve_system_of_equations_l131_13178
