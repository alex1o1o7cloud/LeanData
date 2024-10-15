import Mathlib

namespace NUMINAMATH_GPT_purchase_price_of_grinder_l380_38037

theorem purchase_price_of_grinder (G : ℝ) (H : 0.95 * G + 8800 - (G + 8000) = 50) : G = 15000 := 
sorry

end NUMINAMATH_GPT_purchase_price_of_grinder_l380_38037


namespace NUMINAMATH_GPT_two_pi_irrational_l380_38053

-- Assuming \(\pi\) is irrational as is commonly accepted
def irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem two_pi_irrational : irrational (2 * Real.pi) := 
by 
  sorry

end NUMINAMATH_GPT_two_pi_irrational_l380_38053


namespace NUMINAMATH_GPT_max_selection_no_five_times_l380_38033

theorem max_selection_no_five_times (S : Finset ℕ) (hS : S = Finset.Icc 1 2014) :
  ∃ n, n = 1665 ∧ 
  ∀ (a b : ℕ), a ∈ S → b ∈ S → (a = 5 * b ∨ b = 5 * a) → false :=
sorry

end NUMINAMATH_GPT_max_selection_no_five_times_l380_38033


namespace NUMINAMATH_GPT_joe_average_score_l380_38048

theorem joe_average_score (A B C : ℕ) (lowest_score : ℕ) (final_average : ℕ) :
  lowest_score = 45 ∧ final_average = 65 ∧ (A + B + C) / 3 = final_average →
  (A + B + C + lowest_score) / 4 = 60 := by
  sorry

end NUMINAMATH_GPT_joe_average_score_l380_38048


namespace NUMINAMATH_GPT_correct_equation_solves_time_l380_38074

noncomputable def solve_time_before_stop (t : ℝ) : Prop :=
  let total_trip_time := 4 -- total trip time in hours including stop
  let stop_time := 0.5 -- stop time in hours
  let total_distance := 180 -- total distance in km
  let speed_before_stop := 60 -- speed before stop in km/h
  let speed_after_stop := 80 -- speed after stop in km/h
  let time_after_stop := total_trip_time - stop_time - t -- time after the stop in hours
  speed_before_stop * t + speed_after_stop * time_after_stop = total_distance -- distance equation

-- The theorem states that the equation is valid for solving t
theorem correct_equation_solves_time :
  solve_time_before_stop t = (60 * t + 80 * (7/2 - t) = 180) :=
sorry -- proof not required

end NUMINAMATH_GPT_correct_equation_solves_time_l380_38074


namespace NUMINAMATH_GPT_y_work_time_l380_38091

noncomputable def total_work := 1 

noncomputable def work_rate_x := 1 / 40
noncomputable def work_x_in_8_days := 8 * work_rate_x
noncomputable def remaining_work := total_work - work_x_in_8_days

noncomputable def work_rate_y := remaining_work / 36

theorem y_work_time :
  (1 / work_rate_y) = 45 :=
by
  sorry

end NUMINAMATH_GPT_y_work_time_l380_38091


namespace NUMINAMATH_GPT_box_dimensions_sum_l380_38039

theorem box_dimensions_sum (A B C : ℝ)
  (h1 : A * B = 18)
  (h2 : A * C = 32)
  (h3 : B * C = 50) :
  A + B + C = 57.28 := 
sorry

end NUMINAMATH_GPT_box_dimensions_sum_l380_38039


namespace NUMINAMATH_GPT_gray_region_area_l380_38042

noncomputable def area_of_gray_region (C_center D_center : ℝ × ℝ) (C_radius D_radius : ℝ) :=
  let rect_area := 35
  let semicircle_C_area := (25 * Real.pi) / 2
  let quarter_circle_D_area := (16 * Real.pi) / 4
  rect_area - semicircle_C_area - quarter_circle_D_area

theorem gray_region_area :
  area_of_gray_region (5, 5) (12, 5) 5 4 = 35 - 16.5 * Real.pi :=
by
  simp [area_of_gray_region]
  sorry

end NUMINAMATH_GPT_gray_region_area_l380_38042


namespace NUMINAMATH_GPT_prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l380_38095

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + (deriv g x) = 10
axiom f_cond2 : ∀ x : ℝ, f x - (deriv g (4 - x)) = 10
axiom g_even : ∀ x : ℝ, g x = g (-x)

theorem prove_f_2_eq_10 : f 2 = 10 := sorry
theorem prove_f_4_eq_10 : f 4 = 10 := sorry
theorem prove_f'_neg1_eq_f'_neg3 : deriv f (-1) = deriv f (-3) := sorry
theorem prove_f'_2023_ne_0 : deriv f 2023 ≠ 0 := sorry

end NUMINAMATH_GPT_prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l380_38095


namespace NUMINAMATH_GPT_number_of_juniors_l380_38049

variables (J S x : ℕ)

theorem number_of_juniors (h1 : (2 / 5 : ℚ) * J = x)
                          (h2 : (1 / 4 : ℚ) * S = x)
                          (h3 : J + S = 30) :
  J = 11 :=
sorry

end NUMINAMATH_GPT_number_of_juniors_l380_38049


namespace NUMINAMATH_GPT_annual_percentage_increase_20_l380_38000

variable (P0 P1 : ℕ) (r : ℚ)

-- Population initial condition
def initial_population : Prop := P0 = 10000

-- Population after 1 year condition
def population_after_one_year : Prop := P1 = 12000

-- Define the annual percentage increase formula
def percentage_increase (P0 P1 : ℕ) : ℚ := ((P1 - P0 : ℚ) / P0) * 100

-- State the theorem
theorem annual_percentage_increase_20
  (h1 : initial_population P0)
  (h2 : population_after_one_year P1) :
  percentage_increase P0 P1 = 20 := by
  sorry

end NUMINAMATH_GPT_annual_percentage_increase_20_l380_38000


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l380_38081

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), 
  (a = 3 ∧ b = 6 ∧ (c = 6 ∨ c = 3)) ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  (a + b + c = 15) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l380_38081


namespace NUMINAMATH_GPT_remainder_2027_div_28_l380_38035

theorem remainder_2027_div_28 : 2027 % 28 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2027_div_28_l380_38035


namespace NUMINAMATH_GPT_empty_can_mass_l380_38084

-- Define the mass of the full can
def full_can_mass : ℕ := 35

-- Define the mass of the can with half the milk
def half_can_mass : ℕ := 18

-- The theorem stating the mass of the empty can
theorem empty_can_mass : full_can_mass - (2 * (full_can_mass - half_can_mass)) = 1 := by
  sorry

end NUMINAMATH_GPT_empty_can_mass_l380_38084


namespace NUMINAMATH_GPT_victor_score_l380_38030

-- Definitions based on the conditions
def max_marks : ℕ := 300
def percentage : ℕ := 80

-- Statement to be proved
theorem victor_score : (percentage * max_marks) / 100 = 240 := by
  sorry

end NUMINAMATH_GPT_victor_score_l380_38030


namespace NUMINAMATH_GPT_even_function_implies_a_zero_l380_38013

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, (x^2 - |x + a|) = (x^2 - |x - a|)) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_function_implies_a_zero_l380_38013


namespace NUMINAMATH_GPT_harry_morning_routine_l380_38065

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end NUMINAMATH_GPT_harry_morning_routine_l380_38065


namespace NUMINAMATH_GPT_g_six_g_seven_l380_38026

noncomputable def g : ℝ → ℝ :=
sorry

axiom additivity : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_three : g 3 = 4

theorem g_six : g 6 = 8 :=
by {
  -- proof steps to be added by the prover
  sorry
}

theorem g_seven : g 7 = 28 / 3 :=
by {
  -- proof steps to be added by the prover
  sorry
}

end NUMINAMATH_GPT_g_six_g_seven_l380_38026


namespace NUMINAMATH_GPT_number_of_people_in_family_l380_38044

-- Define the conditions
def planned_spending : ℝ := 15
def savings_percentage : ℝ := 0.40
def cost_per_orange : ℝ := 1.5

-- Define the proof target: the number of people in the family
theorem number_of_people_in_family : 
  planned_spending * savings_percentage / cost_per_orange = 4 := 
by
  -- sorry to skip the proof; this is for statement only
  sorry

end NUMINAMATH_GPT_number_of_people_in_family_l380_38044


namespace NUMINAMATH_GPT_find_x4_l380_38086

theorem find_x4 (x_1 x_2 : ℝ) (h1 : 0 < x_1) (h2 : x_1 < x_2) 
  (P : (ℝ × ℝ)) (Q : (ℝ × ℝ)) (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (R : (ℝ × ℝ)) (x_4 : ℝ) :
  R = ((x_1 + x_2) / 2, (Real.log x_1 + Real.log x_2) / 2) →
  Real.log x_4 = (Real.log x_1 + Real.log x_2) / 2 →
  x_4 = Real.sqrt 1000 :=
by 
  intro hR hT
  sorry

end NUMINAMATH_GPT_find_x4_l380_38086


namespace NUMINAMATH_GPT_find_certain_number_l380_38009

theorem find_certain_number (x : ℝ) (h : 0.80 * x = (4 / 5 * 20) + 16) : x = 40 :=
by sorry

end NUMINAMATH_GPT_find_certain_number_l380_38009


namespace NUMINAMATH_GPT_gcd_204_85_l380_38071

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end NUMINAMATH_GPT_gcd_204_85_l380_38071


namespace NUMINAMATH_GPT_john_school_year_hours_l380_38056

theorem john_school_year_hours (summer_earnings : ℝ) (summer_hours_per_week : ℝ) (summer_weeks : ℝ) (target_school_earnings : ℝ) (school_weeks : ℝ) :
  summer_earnings = 4000 → summer_hours_per_week = 40 → summer_weeks = 8 → target_school_earnings = 5000 → school_weeks = 25 →
  (target_school_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_weeks) = 16 :=
by
  sorry

end NUMINAMATH_GPT_john_school_year_hours_l380_38056


namespace NUMINAMATH_GPT_solution_set_of_inequality_l380_38087

theorem solution_set_of_inequality
  (a b : ℝ)
  (h1 : a < 0) 
  (h2 : b / a = 1) :
  { x : ℝ | (x - 1) * (a * x + b) < 0 } = { x : ℝ | x < -1 } ∪ {x : ℝ | 1 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l380_38087


namespace NUMINAMATH_GPT_average_marks_of_class_l380_38001

theorem average_marks_of_class :
  (∀ (students total_students: ℕ) (marks95 marks0: ℕ) (avg_remaining: ℕ),
    total_students = 25 →
    students = 3 →
    marks95 = 95 →
    students = 5 →
    marks0 = 0 →
    (total_students - students - students) = 17 →
    avg_remaining = 45 →
    ((students * marks95 + students * marks0 + (total_students - students - students) * avg_remaining) / total_students) = 42)
:= sorry

end NUMINAMATH_GPT_average_marks_of_class_l380_38001


namespace NUMINAMATH_GPT_rational_root_k_values_l380_38099

theorem rational_root_k_values (k : ℤ) :
  (∃ x : ℚ, x^2017 - x^2016 + x^2 + k * x + 1 = 0) ↔ (k = 0 ∨ k = -2) :=
by
  sorry

end NUMINAMATH_GPT_rational_root_k_values_l380_38099


namespace NUMINAMATH_GPT_value_of_expression_l380_38012

theorem value_of_expression {a b : ℝ} (h1 : 2 * a^2 + 6 * a - 14 = 0) (h2 : 2 * b^2 + 6 * b - 14 = 0) :
  (2 * a - 3) * (4 * b - 6) = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l380_38012


namespace NUMINAMATH_GPT_smallest_initial_number_sum_of_digits_l380_38036

theorem smallest_initial_number_sum_of_digits : ∃ (N : ℕ), 
  (0 ≤ N ∧ N < 1000) ∧ 
  ∃ (k : ℕ), 16 * N + 700 + 50 * k < 1000 ∧ 
  (N = 16) ∧ 
  (Nat.digits 10 N).sum = 7 := 
by
  sorry

end NUMINAMATH_GPT_smallest_initial_number_sum_of_digits_l380_38036


namespace NUMINAMATH_GPT_find_m_value_l380_38045

theorem find_m_value (m a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h1 : (x + m)^9 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + 
  a_8 * (x + 1)^8 + a_9 * (x + 1)^9)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 = 3^9) :
  m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l380_38045


namespace NUMINAMATH_GPT_cabbage_price_is_4_02_l380_38019

noncomputable def price_of_cabbage (broccoli_price_per_pound: ℝ) (broccoli_pounds: ℝ) 
                                    (orange_price_each: ℝ) (oranges: ℝ) 
                                    (bacon_price_per_pound: ℝ) (bacon_pounds: ℝ) 
                                    (chicken_price_per_pound: ℝ) (chicken_pounds: ℝ) 
                                    (budget_percentage_for_meat: ℝ) 
                                    (meat_price: ℝ) : ℝ := 
  let broccoli_total := broccoli_pounds * broccoli_price_per_pound
  let oranges_total := oranges * orange_price_each
  let bacon_total := bacon_pounds * bacon_price_per_pound
  let chicken_total := chicken_pounds * chicken_price_per_pound
  let subtotal := broccoli_total + oranges_total + bacon_total + chicken_total
  let total_budget := meat_price / budget_percentage_for_meat
  total_budget - subtotal

theorem cabbage_price_is_4_02 : 
  price_of_cabbage 4 3 0.75 3 3 1 3 2 0.33 9 = 4.02 := 
by 
  sorry

end NUMINAMATH_GPT_cabbage_price_is_4_02_l380_38019


namespace NUMINAMATH_GPT_smallest_x_value_l380_38067

open Real

theorem smallest_x_value (x : ℝ) 
  (h : x * abs x = 3 * x + 2) : 
  x = -2 ∨ (∀ y, y * abs y = 3 * y + 2 → y ≥ -2) := sorry

end NUMINAMATH_GPT_smallest_x_value_l380_38067


namespace NUMINAMATH_GPT_min_value_a_l380_38089

theorem min_value_a (a : ℝ) : (∀ x : ℝ, a < x → 2 * x + 2 / (x - a) ≥ 7) → a ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_l380_38089


namespace NUMINAMATH_GPT_train_speed_l380_38066

theorem train_speed (distance_meters : ℕ) (time_seconds : ℕ) 
  (h_distance : distance_meters = 150) (h_time : time_seconds = 20) : 
  distance_meters / 1000 / (time_seconds / 3600) = 27 :=
by 
  have h1 : distance_meters = 150 := h_distance
  have h2 : time_seconds = 20 := h_time
  -- other intermediate steps would go here, but are omitted
  -- for now, we assume the final calculation is:
  sorry

end NUMINAMATH_GPT_train_speed_l380_38066


namespace NUMINAMATH_GPT_olivia_wallet_final_amount_l380_38051

variable (initial_money : ℕ) (money_added : ℕ) (money_spent : ℕ)

theorem olivia_wallet_final_amount
  (h1 : initial_money = 100)
  (h2 : money_added = 148)
  (h3 : money_spent = 89) :
  initial_money + money_added - money_spent = 159 :=
  by 
    sorry

end NUMINAMATH_GPT_olivia_wallet_final_amount_l380_38051


namespace NUMINAMATH_GPT_cross_section_area_l380_38076

-- Definitions for the conditions stated in the problem
def frustum_height : ℝ := 6
def upper_base_side : ℝ := 4
def lower_base_side : ℝ := 8

-- The main statement to be proved
theorem cross_section_area :
  (exists (cross_section_area : ℝ),
    cross_section_area = 16 * Real.sqrt 6) :=
sorry

end NUMINAMATH_GPT_cross_section_area_l380_38076


namespace NUMINAMATH_GPT_no_square_from_vertices_of_equilateral_triangles_l380_38077

-- Definitions
def equilateral_triangle_grid (p : ℝ × ℝ) : Prop := 
  ∃ k l : ℤ, p.1 = k * (1 / 2) ∧ p.2 = l * (Real.sqrt 3 / 2)

def form_square_by_vertices (A B C D : ℝ × ℝ) : Prop := 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 ∧ 
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = (D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2 ∧ 
  (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2
  
-- Problem Statement
theorem no_square_from_vertices_of_equilateral_triangles :
  ¬ ∃ (A B C D : ℝ × ℝ), 
    equilateral_triangle_grid A ∧ 
    equilateral_triangle_grid B ∧ 
    equilateral_triangle_grid C ∧ 
    equilateral_triangle_grid D ∧ 
    form_square_by_vertices A B C D :=
by
  sorry

end NUMINAMATH_GPT_no_square_from_vertices_of_equilateral_triangles_l380_38077


namespace NUMINAMATH_GPT_height_of_tree_l380_38021

-- Definitions based on conditions
def net_gain (hop: ℕ) (slip: ℕ) : ℕ := hop - slip

def total_distance (hours: ℕ) (net_gain: ℕ) (final_hop: ℕ) : ℕ :=
  hours * net_gain + final_hop

-- Conditions
def hop : ℕ := 3
def slip : ℕ := 2
def time : ℕ := 20

-- Deriving the net gain per hour
#eval net_gain hop slip  -- Evaluates to 1

-- Final height proof problem
theorem height_of_tree : total_distance 19 (net_gain hop slip) hop = 22 := by
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_height_of_tree_l380_38021


namespace NUMINAMATH_GPT_matrix_multiplication_correct_l380_38020

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 3, -1], ![1, -2, 5], ![0, 6, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 4], ![3, 2, -1], ![0, 4, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![11, 2, 7], ![-5, 16, -4], ![18, 16, -8]]

theorem matrix_multiplication_correct :
  A * B = C :=
by
  sorry

end NUMINAMATH_GPT_matrix_multiplication_correct_l380_38020


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l380_38011

noncomputable def f (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) := a ^ (-x^2 + 3 * x + 2)

theorem monotonic_increasing_interval (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x1 x2 : ℝ, (3 / 2 < x1 ∧ x1 < x2) → f a h x1 < f a h x2 :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l380_38011


namespace NUMINAMATH_GPT_shelby_gold_stars_l380_38088

theorem shelby_gold_stars (stars_yesterday stars_today : ℕ) (h1 : stars_yesterday = 4) (h2 : stars_today = 3) :
  stars_yesterday + stars_today = 7 := 
by
  sorry

end NUMINAMATH_GPT_shelby_gold_stars_l380_38088


namespace NUMINAMATH_GPT_option_C_incorrect_l380_38082

def p (x y : ℝ) : ℝ := x^3 - 3 * x^2 * y + 3 * x * y^2 - y^3

theorem option_C_incorrect (x y : ℝ) : 
  ((x^3 - 3 * x^2 * y) - (3 * x * y^2 + y^3)) ≠ p x y := by
  sorry

end NUMINAMATH_GPT_option_C_incorrect_l380_38082


namespace NUMINAMATH_GPT_div_expr_l380_38097

namespace Proof

theorem div_expr (x : ℝ) (h : x = 3.242 * 10) : x / 100 = 0.3242 := by
  sorry

end Proof

end NUMINAMATH_GPT_div_expr_l380_38097


namespace NUMINAMATH_GPT_isosceles_triangle_l380_38079

-- Given: sides a, b, c of a triangle satisfying a specific condition
-- To Prove: the triangle is isosceles (has at least two equal sides)

theorem isosceles_triangle (a b c : ℝ)
  (h : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l380_38079


namespace NUMINAMATH_GPT_find_angle_B_l380_38018

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l380_38018


namespace NUMINAMATH_GPT_smallest_number_of_coins_l380_38093

theorem smallest_number_of_coins : ∃ (n : ℕ), 
  n ≡ 2 [MOD 5] ∧ 
  n ≡ 1 [MOD 4] ∧ 
  n ≡ 0 [MOD 3] ∧ 
  n = 57 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l380_38093


namespace NUMINAMATH_GPT_john_weekly_earnings_increase_l380_38068

theorem john_weekly_earnings_increase (original_earnings new_earnings : ℕ) 
  (h₀ : original_earnings = 60) 
  (h₁ : new_earnings = 72) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_john_weekly_earnings_increase_l380_38068


namespace NUMINAMATH_GPT_burglar_goods_value_l380_38058

theorem burglar_goods_value (V : ℝ) (S : ℝ) (S_increased : ℝ) (S_total : ℝ) (h1 : S = V / 5000) (h2 : S_increased = 1.25 * S) (h3 : S_total = S_increased + 2) (h4 : S_total = 12) : V = 40000 := by
  sorry

end NUMINAMATH_GPT_burglar_goods_value_l380_38058


namespace NUMINAMATH_GPT_Vanya_two_digit_number_l380_38024

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_Vanya_two_digit_number_l380_38024


namespace NUMINAMATH_GPT_total_money_spent_l380_38062

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end NUMINAMATH_GPT_total_money_spent_l380_38062


namespace NUMINAMATH_GPT_set_of_a_l380_38055

theorem set_of_a (a : ℝ) :
  (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → -- Set A contains elements
  (a ≠ 0 ∧ a ^ 2 - 4 * a = 0) →           -- Conditions a ≠ 0 and Δ = 0
  a = 4 := 
sorry

end NUMINAMATH_GPT_set_of_a_l380_38055


namespace NUMINAMATH_GPT_arithmetic_mean_of_a_and_b_is_sqrt3_l380_38022

theorem arithmetic_mean_of_a_and_b_is_sqrt3 :
  let a := (Real.sqrt 3 + Real.sqrt 2)
  let b := (Real.sqrt 3 - Real.sqrt 2)
  (a + b) / 2 = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_a_and_b_is_sqrt3_l380_38022


namespace NUMINAMATH_GPT_max_value_of_g_l380_38072

def g (n : ℕ) : ℕ :=
  if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ m, ∀ n, g n ≤ m ∧ (∃ k, g k = m) :=
by
  use 29
  sorry

end NUMINAMATH_GPT_max_value_of_g_l380_38072


namespace NUMINAMATH_GPT_length_of_field_l380_38004

-- Define the problem conditions
variables (width length : ℕ)
  (pond_area field_area : ℕ)
  (h1 : length = 2 * width)
  (h2 : pond_area = 64)
  (h3 : pond_area = field_area / 8)

-- Define the proof problem
theorem length_of_field : length = 32 :=
by
  -- We'll provide the proof later
  sorry

end NUMINAMATH_GPT_length_of_field_l380_38004


namespace NUMINAMATH_GPT_inscribed_rectangle_l380_38085

theorem inscribed_rectangle (b h : ℝ) : ∃ x : ℝ, 
  (∃ q : ℝ, x = q / 2) → 
  ∃ x : ℝ, 
    (∃ q : ℝ, q = 2 * x ∧ x = h * q / (2 * h + b)) :=
sorry

end NUMINAMATH_GPT_inscribed_rectangle_l380_38085


namespace NUMINAMATH_GPT_ratio_second_to_first_l380_38057

-- Define the given conditions and variables
variables 
  (total_water : ℕ := 1200)
  (neighborhood1_usage : ℕ := 150)
  (neighborhood4_usage : ℕ := 350)
  (x : ℕ) -- water usage by second neighborhood

-- Define the usage by third neighborhood in terms of the second neighborhood usage
def neighborhood3_usage := x + 100

-- Define remaining water usage after substracting neighborhood 4 usage
def remaining_water := total_water - neighborhood4_usage

-- The sum of water used by neighborhoods
def total_usage_neighborhoods := neighborhood1_usage + neighborhood3_usage x + x

theorem ratio_second_to_first (h : total_usage_neighborhoods x = remaining_water) :
  (x : ℚ) / neighborhood1_usage = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_second_to_first_l380_38057


namespace NUMINAMATH_GPT_number_of_members_after_four_years_l380_38034

theorem number_of_members_after_four_years (b : ℕ → ℕ) (initial_condition : b 0 = 21) 
    (yearly_update : ∀ k, b (k + 1) = 4 * b k - 9) : 
    b 4 = 4611 :=
    sorry

end NUMINAMATH_GPT_number_of_members_after_four_years_l380_38034


namespace NUMINAMATH_GPT_rowing_distance_l380_38069

theorem rowing_distance
  (v_still : ℝ) (v_current : ℝ) (time : ℝ)
  (h1 : v_still = 15) (h2 : v_current = 3) (h3 : time = 17.998560115190784) :
  (v_still + v_current) * 1000 / 3600 * time = 89.99280057595392 :=
by
  rw [h1, h2, h3] -- Apply the given conditions
  -- This will reduce to proving (15 + 3) * 1000 / 3600 * 17.998560115190784 = 89.99280057595392
  sorry

end NUMINAMATH_GPT_rowing_distance_l380_38069


namespace NUMINAMATH_GPT_song_book_cost_correct_l380_38003

noncomputable def cost_of_trumpet : ℝ := 145.16
noncomputable def total_spent : ℝ := 151.00
noncomputable def cost_of_song_book : ℝ := total_spent - cost_of_trumpet

theorem song_book_cost_correct : cost_of_song_book = 5.84 :=
  by
    sorry

end NUMINAMATH_GPT_song_book_cost_correct_l380_38003


namespace NUMINAMATH_GPT_min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l380_38025

variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b)

theorem min_ab : 16 ≤ a * b :=
sorry

theorem min_a_b : 9 ≤ a + b :=
sorry

theorem max_two_a_one_b : 2 > (2 / a + 1 / b) :=
sorry

theorem min_one_a_sq_four_b_sq : 1 / 5 ≤ (1 / a^2 + 4 / b^2) :=
sorry

end NUMINAMATH_GPT_min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l380_38025


namespace NUMINAMATH_GPT_Oscar_height_correct_l380_38006

-- Definitions of the given conditions
def Tobias_height : ℕ := 184
def avg_height : ℕ := 178

def heights_valid (Victor Peter Oscar Tobias : ℕ) : Prop :=
  Tobias = 184 ∧ (Tobias + Victor + Peter + Oscar) / 4 = 178 ∧ 
  Victor = Tobias + (Tobias - Peter) ∧ 
  Oscar = Peter - (Tobias - Peter)

theorem Oscar_height_correct :
  ∃ (k : ℕ), ∀ (Victor Peter Oscar : ℕ), heights_valid Victor Peter Oscar Tobias_height →
  Oscar = 160 :=
by
  sorry

end NUMINAMATH_GPT_Oscar_height_correct_l380_38006


namespace NUMINAMATH_GPT_pictures_at_dolphin_show_l380_38075

def taken_before : Int := 28
def total_pictures_taken : Int := 44

theorem pictures_at_dolphin_show : total_pictures_taken - taken_before = 16 := by
  -- solution proof goes here
  sorry

end NUMINAMATH_GPT_pictures_at_dolphin_show_l380_38075


namespace NUMINAMATH_GPT_parabola_directrix_l380_38098

theorem parabola_directrix (x y : ℝ) (h : y = x^2) : 4 * y + 1 = 0 := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l380_38098


namespace NUMINAMATH_GPT_repeating_decimal_fraction_form_l380_38064

noncomputable def repeating_decimal_rational := 2.71717171

theorem repeating_decimal_fraction_form : 
  repeating_decimal_rational = 269 / 99 ∧ (269 + 99 = 368) := 
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_form_l380_38064


namespace NUMINAMATH_GPT_min_value_ratio_l380_38094

variable {α : Type*} [LinearOrderedField α]

theorem min_value_ratio (a : ℕ → α) (h1 : a 7 = a 6 + 2 * a 5) (h2 : ∃ m n : ℕ, a m * a n = 8 * a 1^2) :
  ∃ m n : ℕ, (1 / m + 4 / n = 11 / 6) :=
by
  sorry

end NUMINAMATH_GPT_min_value_ratio_l380_38094


namespace NUMINAMATH_GPT_trapezoid_median_l380_38023

theorem trapezoid_median 
  (h : ℝ)
  (triangle_base : ℝ := 24)
  (trapezoid_base1 : ℝ := 15)
  (trapezoid_base2 : ℝ := 33)
  (triangle_area_eq_trapezoid_area : (1 / 2) * triangle_base * h = ((trapezoid_base1 + trapezoid_base2) / 2) * h)
  : (trapezoid_base1 + trapezoid_base2) / 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_median_l380_38023


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l380_38050

theorem geometric_series_common_ratio (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) 
  (h₃ : S = a / (1 - r)) : r = 21 / 25 := 
sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l380_38050


namespace NUMINAMATH_GPT_div_problem_l380_38083

theorem div_problem : 150 / (6 / 3) = 75 := by
  sorry

end NUMINAMATH_GPT_div_problem_l380_38083


namespace NUMINAMATH_GPT_length_of_BC_l380_38059

noncomputable def perimeter (a b c : ℝ) := a + b + c
noncomputable def area (b c : ℝ) (A : ℝ) := 0.5 * b * c * (Real.sin A)

theorem length_of_BC
  (a b c : ℝ)
  (h_perimeter : perimeter a b c = 20)
  (h_area : area b c (Real.pi / 3) = 10 * Real.sqrt 3) :
  a = 7 :=
by
  sorry

end NUMINAMATH_GPT_length_of_BC_l380_38059


namespace NUMINAMATH_GPT_shark_sightings_in_cape_may_l380_38029

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_shark_sightings_in_cape_may_l380_38029


namespace NUMINAMATH_GPT_number_of_boys_l380_38043

theorem number_of_boys (total_students girls : ℕ) (h1 : total_students = 13) (h2 : girls = 6) :
  total_students - girls = 7 :=
by 
  -- We'll skip the proof as instructed
  sorry

end NUMINAMATH_GPT_number_of_boys_l380_38043


namespace NUMINAMATH_GPT_Angie_necessities_amount_l380_38070

noncomputable def Angie_salary : ℕ := 80
noncomputable def Angie_left_over : ℕ := 18
noncomputable def Angie_taxes : ℕ := 20
noncomputable def Angie_expenses : ℕ := Angie_salary - Angie_left_over
noncomputable def Angie_necessities : ℕ := Angie_expenses - Angie_taxes

theorem Angie_necessities_amount :
  Angie_necessities = 42 :=
by
  unfold Angie_necessities
  unfold Angie_expenses
  sorry

end NUMINAMATH_GPT_Angie_necessities_amount_l380_38070


namespace NUMINAMATH_GPT_dealership_sales_l380_38078

theorem dealership_sales :
  (∀ (n : ℕ), 3 * n ≤ 36 → 5 * n ≤ x) →
  (36 / 3) * 5 = 60 :=
by
  sorry

end NUMINAMATH_GPT_dealership_sales_l380_38078


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l380_38027

theorem necessary_and_sufficient_condition {a b : ℝ} :
  (a > b) ↔ (a^3 > b^3) := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l380_38027


namespace NUMINAMATH_GPT_find_diff_eq_l380_38010

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
∃ (C1 C2 : ℝ), ∀ x : ℝ, y x = C1 * x + C2

theorem find_diff_eq (y : ℝ → ℝ) (C1 C2 : ℝ) (h : ∀ x : ℝ, y x = C1 * x + C2) :
  ∀ x : ℝ, (deriv (deriv y)) x = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_diff_eq_l380_38010


namespace NUMINAMATH_GPT_integral_eq_exp_integral_eq_one_l380_38040

noncomputable
def y1 (τ : ℝ) (t : ℝ) (y : ℝ → ℝ) : Prop :=
  y τ = ∫ x in (0 : ℝ)..t, y x + 1

theorem integral_eq_exp (y : ℝ → ℝ) : 
  (∀ τ t, y1 τ t y) ↔ (∀ t, y t = Real.exp t) := 
  sorry

noncomputable
def y2 (t : ℝ) (y : ℝ → ℝ) : Prop :=
  ∫ x in (0 : ℝ)..t, y x * Real.sin (t - x) = 1 - Real.cos t

theorem integral_eq_one (y : ℝ → ℝ) : 
  (∀ t, y2 t y) ↔ (∀ t, y t = 1) :=
  sorry

end NUMINAMATH_GPT_integral_eq_exp_integral_eq_one_l380_38040


namespace NUMINAMATH_GPT_remaining_unit_area_l380_38032

theorem remaining_unit_area
    (total_units : ℕ)
    (total_area : ℕ)
    (num_12x6_units : ℕ)
    (length_12x6_unit : ℕ)
    (width_12x6_unit : ℕ)
    (remaining_units_area : ℕ)
    (num_remaining_units : ℕ)
    (remaining_unit_area : ℕ) :
  total_units = 72 →
  total_area = 8640 →
  num_12x6_units = 30 →
  length_12x6_unit = 12 →
  width_12x6_unit = 6 →
  remaining_units_area = total_area - (num_12x6_units * length_12x6_unit * width_12x6_unit) →
  num_remaining_units = total_units - num_12x6_units →
  remaining_unit_area = remaining_units_area / num_remaining_units →
  remaining_unit_area = 154 :=
by
  intros h_total_units h_total_area h_num_12x6_units h_length_12x6_unit h_width_12x6_unit h_remaining_units_area h_num_remaining_units h_remaining_unit_area
  sorry

end NUMINAMATH_GPT_remaining_unit_area_l380_38032


namespace NUMINAMATH_GPT_prob_select_math_books_l380_38038

theorem prob_select_math_books :
  let total_books := 5
  let math_books := 3
  let total_ways_select_2 := Nat.choose total_books 2
  let ways_select_2_math := Nat.choose math_books 2
  let probability := (ways_select_2_math : ℚ) / total_ways_select_2
  probability = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_prob_select_math_books_l380_38038


namespace NUMINAMATH_GPT_bleaching_takes_3_hours_l380_38052

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end NUMINAMATH_GPT_bleaching_takes_3_hours_l380_38052


namespace NUMINAMATH_GPT_avg_b_c_is_45_l380_38080

-- Define the weights of a, b, and c
variables (a b c : ℝ)

-- Conditions given in the problem
def avg_a_b_c (a b c : ℝ) := (a + b + c) / 3 = 45
def avg_a_b (a b : ℝ) := (a + b) / 2 = 40
def weight_b (b : ℝ) := b = 35

-- Theorem statement
theorem avg_b_c_is_45 (a b c : ℝ) (h1 : avg_a_b_c a b c) (h2 : avg_a_b a b) (h3 : weight_b b) :
  (b + c) / 2 = 45 := by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_avg_b_c_is_45_l380_38080


namespace NUMINAMATH_GPT_Jeff_has_20_trucks_l380_38016

theorem Jeff_has_20_trucks
  (T C : ℕ)
  (h1 : C = 2 * T)
  (h2 : T + C = 60) :
  T = 20 :=
sorry

end NUMINAMATH_GPT_Jeff_has_20_trucks_l380_38016


namespace NUMINAMATH_GPT_calculate_expression_l380_38073

theorem calculate_expression :
  |(-1 : ℝ)| + Real.sqrt 9 - (1 - Real.sqrt 3)^0 - (1/2)^(-1 : ℝ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l380_38073


namespace NUMINAMATH_GPT_find_b1_over_b2_l380_38096

variable {a b k a1 a2 b1 b2 : ℝ}

-- Assuming a is inversely proportional to b
def inversely_proportional (a b : ℝ) (k : ℝ) : Prop :=
  a * b = k

-- Define that a_1 and a_2 are nonzero and their ratio is 3/4
def a1_a2_ratio (a1 a2 : ℝ) (ratio : ℝ) : Prop :=
  a1 / a2 = ratio

-- Define that b_1 and b_2 are nonzero
def nonzero (x : ℝ) : Prop :=
  x ≠ 0

theorem find_b1_over_b2 (a1 a2 b1 b2 : ℝ) (h1 : inversely_proportional a b k)
  (h2 : a1_a2_ratio a1 a2 (3 / 4))
  (h3 : nonzero a1) (h4 : nonzero a2) (h5 : nonzero b1) (h6 : nonzero b2) :
  b1 / b2 = 4 / 3 := 
sorry

end NUMINAMATH_GPT_find_b1_over_b2_l380_38096


namespace NUMINAMATH_GPT_lisa_phone_spending_l380_38090

variable (cost_phone : ℕ) (cost_contract_per_month : ℕ) (case_percentage : ℕ) (headphones_ratio : ℕ)

/-- Given the cost of the phone, the monthly contract cost, 
    the percentage cost of the case, and ratio cost of headphones,
    prove that the total spending in the first year is correct.
-/ 
theorem lisa_phone_spending 
    (h_cost_phone : cost_phone = 1000) 
    (h_cost_contract_per_month : cost_contract_per_month = 200) 
    (h_case_percentage : case_percentage = 20)
    (h_headphones_ratio : headphones_ratio = 2) :
    cost_phone + (cost_phone * case_percentage / 100) + 
    ((cost_phone * case_percentage / 100) / headphones_ratio) + 
    (cost_contract_per_month * 12) = 3700 :=
by
  sorry

end NUMINAMATH_GPT_lisa_phone_spending_l380_38090


namespace NUMINAMATH_GPT_quadratic_eq_two_distinct_real_roots_l380_38017

theorem quadratic_eq_two_distinct_real_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2 * x₁ + a = 0) ∧ (x₂^2 - 2 * x₂ + a = 0)) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_two_distinct_real_roots_l380_38017


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l380_38007

theorem solve_arithmetic_sequence (y : ℝ) (h1 : y ^ 2 = (4 + 25) / 2) (h2 : y > 0) :
  y = Real.sqrt 14.5 :=
sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l380_38007


namespace NUMINAMATH_GPT_cut_out_area_l380_38005

theorem cut_out_area (x : ℝ) (h1 : x * (x - 10) = 1575) : 10 * x - 10 * 10 = 450 := by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_cut_out_area_l380_38005


namespace NUMINAMATH_GPT_sum_of_first_six_primes_l380_38014

theorem sum_of_first_six_primes : (2 + 3 + 5 + 7 + 11 + 13) = 41 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_six_primes_l380_38014


namespace NUMINAMATH_GPT_possible_values_of_a_l380_38047

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a / x

theorem possible_values_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l380_38047


namespace NUMINAMATH_GPT_range_of_a_l380_38028

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → 0 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ x₀, -2 ≤ x₀ ∧ x₀ ≤ 2 ∧ (a * x₀ - 1 = f x)) →
  a ∈ Set.Iic (-5/2) ∪ Set.Ici (5/2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l380_38028


namespace NUMINAMATH_GPT_min_value_z_l380_38092

theorem min_value_z (x y z : ℤ) (h1 : x + y + z = 100) (h2 : x < y) (h3 : y < 2 * z) : z ≥ 21 :=
sorry

end NUMINAMATH_GPT_min_value_z_l380_38092


namespace NUMINAMATH_GPT_LaurynCompanyEmployees_l380_38060

noncomputable def LaurynTotalEmployees (men women total : ℕ) : Prop :=
  men = 80 ∧ women = men + 20 ∧ total = men + women

theorem LaurynCompanyEmployees : ∃ total, ∀ men women, LaurynTotalEmployees men women total → total = 180 :=
by 
  sorry

end NUMINAMATH_GPT_LaurynCompanyEmployees_l380_38060


namespace NUMINAMATH_GPT_transform_equation_to_square_form_l380_38015

theorem transform_equation_to_square_form : 
  ∀ x : ℝ, (x^2 - 6 * x = 0) → ∃ m n : ℝ, (x + m) ^ 2 = n ∧ m = -3 ∧ n = 9 := 
sorry

end NUMINAMATH_GPT_transform_equation_to_square_form_l380_38015


namespace NUMINAMATH_GPT_delta_value_l380_38061

theorem delta_value (Δ : ℤ) : 5 * (-3) = Δ - 3 → Δ = -12 :=
by
  sorry

end NUMINAMATH_GPT_delta_value_l380_38061


namespace NUMINAMATH_GPT_housewife_more_oil_l380_38041

theorem housewife_more_oil 
    (reduction_percent : ℝ := 10)
    (reduced_price : ℝ := 16)
    (budget : ℝ := 800)
    (approx_answer : ℝ := 5.01) :
    let P := reduced_price / (1 - reduction_percent / 100)
    let Q_original := budget / P
    let Q_reduced := budget / reduced_price
    let delta_Q := Q_reduced - Q_original
    abs (delta_Q - approx_answer) < 0.02 := 
by
  -- Let the goal be irrelevant to the proof because the proof isn't provided
  sorry

end NUMINAMATH_GPT_housewife_more_oil_l380_38041


namespace NUMINAMATH_GPT_divisor_greater_2016_l380_38031

theorem divisor_greater_2016 (d : ℕ) (h : 2016 / d = 0) : d > 2016 :=
sorry

end NUMINAMATH_GPT_divisor_greater_2016_l380_38031


namespace NUMINAMATH_GPT_bacterium_descendants_l380_38002

theorem bacterium_descendants (n a : ℕ) (h : a ≤ n / 2) :
  ∃ k, a ≤ k ∧ k ≤ 2 * a - 1 := 
sorry

end NUMINAMATH_GPT_bacterium_descendants_l380_38002


namespace NUMINAMATH_GPT_boards_per_package_calculation_l380_38063

-- Defining the conditions
def total_boards : ℕ := 154
def num_packages : ℕ := 52

-- Defining the division of total_boards by num_packages within rationals
def boards_per_package : ℚ := total_boards / num_packages

-- Prove that the boards per package is mathematically equal to the total boards divided by the number of packages
theorem boards_per_package_calculation :
  boards_per_package = 154 / 52 := by
  sorry

end NUMINAMATH_GPT_boards_per_package_calculation_l380_38063


namespace NUMINAMATH_GPT_value_of_f_8_minus_f_4_l380_38046

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_8_minus_f_4 :
  -- Conditions
  (∀ x, f (-x) = -f x) ∧              -- odd function
  (∀ x, f (x + 5) = f x) ∧            -- period of 5
  (f 1 = 1) ∧                         -- f(1) = 1
  (f 2 = 3) →                         -- f(2) = 3
  -- Goal
  f 8 - f 4 = -2 :=
sorry

end NUMINAMATH_GPT_value_of_f_8_minus_f_4_l380_38046


namespace NUMINAMATH_GPT_outfits_count_l380_38054

theorem outfits_count (shirts ties pants belts : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 5) (h_pants : pants = 4) (h_belts : belts = 2) : 
  (shirts * pants * (ties + 1) * (belts + 1 + 1) = 504) :=
by
  rw [h_shirts, h_ties, h_pants, h_belts]
  sorry

end NUMINAMATH_GPT_outfits_count_l380_38054


namespace NUMINAMATH_GPT_bryden_payment_l380_38008

theorem bryden_payment :
  (let face_value := 0.25
   let quarters := 6
   let collector_multiplier := 16
   let discount := 0.10
   let initial_payment := collector_multiplier * (quarters * face_value)
   let final_payment := initial_payment - (initial_payment * discount)
   final_payment = 21.6) :=
by
  sorry

end NUMINAMATH_GPT_bryden_payment_l380_38008
