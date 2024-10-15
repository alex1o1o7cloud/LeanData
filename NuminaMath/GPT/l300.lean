import Mathlib

namespace NUMINAMATH_GPT_betty_wallet_l300_30003

theorem betty_wallet :
  let wallet_cost := 125.75
  let initial_amount := wallet_cost / 2
  let parents_contribution := 45.25
  let grandparents_contribution := 2 * parents_contribution
  let brothers_contribution := 3/4 * grandparents_contribution
  let aunts_contribution := 1/2 * brothers_contribution
  let total_amount := initial_amount + parents_contribution + grandparents_contribution + brothers_contribution + aunts_contribution
  total_amount - wallet_cost = 174.6875 :=
by
  sorry

end NUMINAMATH_GPT_betty_wallet_l300_30003


namespace NUMINAMATH_GPT_inequality_sum_l300_30086

theorem inequality_sum {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) : a + c > b + d :=
by
  sorry

end NUMINAMATH_GPT_inequality_sum_l300_30086


namespace NUMINAMATH_GPT_like_terms_set_l300_30081

theorem like_terms_set (a b : ℕ) (x y : ℝ) : 
  (¬ (a = b)) ∧
  ((-2 * x^3 * y^3 = y^3 * x^3)) ∧ 
  (¬ (1 * x * y = 2 * x * y^3)) ∧ 
  (¬ (-6 = x)) :=
by
  sorry

end NUMINAMATH_GPT_like_terms_set_l300_30081


namespace NUMINAMATH_GPT_seven_fifths_of_fraction_l300_30032

theorem seven_fifths_of_fraction :
  (7 / 5) * (-18 / 4) = -63 / 10 :=
by
  sorry

end NUMINAMATH_GPT_seven_fifths_of_fraction_l300_30032


namespace NUMINAMATH_GPT_cos_double_angle_l300_30025

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) : Real.cos (2 * θ) = -1/3 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l300_30025


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l300_30024

theorem geometric_sequence_third_term (a₁ a₄ : ℕ) (r : ℕ) (h₁ : a₁ = 4) (h₂ : a₄ = 256) (h₃ : a₄ = a₁ * r^3) : a₁ * r^2 = 64 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l300_30024


namespace NUMINAMATH_GPT_speed_of_stream_l300_30028

-- Define the problem conditions
def downstream_distance := 100 -- distance in km
def downstream_time := 8 -- time in hours
def upstream_distance := 75 -- distance in km
def upstream_time := 15 -- time in hours

-- Define the constants
def total_distance (B S : ℝ) := downstream_distance = (B + S) * downstream_time
def total_time (B S : ℝ) := upstream_distance = (B - S) * upstream_time

-- Stating the main theorem to be proved
theorem speed_of_stream (B S : ℝ) (h1 : total_distance B S) (h2 : total_time B S) : S = 3.75 := by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l300_30028


namespace NUMINAMATH_GPT_seq_20_l300_30022

noncomputable def seq (n : ℕ) : ℝ := 
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 1/2
  else sorry -- The actual function definition based on the recurrence relation is omitted for brevity

lemma seq_recurrence (n : ℕ) (hn : n ≥ 1) :
  2 / seq (n + 1) = (seq n + seq (n + 2)) / (seq n * seq (n + 2)) :=
sorry

theorem seq_20 : seq 20 = 1/20 :=
sorry

end NUMINAMATH_GPT_seq_20_l300_30022


namespace NUMINAMATH_GPT_container_volumes_l300_30040

variable (a : ℕ)

theorem container_volumes (h₁ : a = 18) :
  a^3 = 5832 ∧ (a - 4)^3 = 2744 ∧ (a - 6)^3 = 1728 :=
by {
  sorry
}

end NUMINAMATH_GPT_container_volumes_l300_30040


namespace NUMINAMATH_GPT_quadratic_completing_square_l300_30037

theorem quadratic_completing_square
  (a : ℤ) (b : ℤ) (c : ℤ)
  (h1 : a > 0)
  (h2 : 64 * a^2 * x^2 - 96 * x - 48 = 64 * x^2 - 96 * x - 48)
  (h3 : (a * x + b)^2 = c) :
  a + b + c = 86 :=
sorry

end NUMINAMATH_GPT_quadratic_completing_square_l300_30037


namespace NUMINAMATH_GPT_math_problem_l300_30019

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end NUMINAMATH_GPT_math_problem_l300_30019


namespace NUMINAMATH_GPT_P_investment_time_l300_30090

noncomputable def investment_in_months 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop)
  (time_Q : ℕ)
  (time_P : ℕ)
  (x : ℕ) : Prop :=
  investment_ratio_PQ 7 5 ∧ 
  profit_ratio_PQ 7 9 ∧ 
  time_Q = 9 ∧ 
  (7 * time_P) / (5 * time_Q) = 7 / 9

theorem P_investment_time 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop) 
  (x : ℕ) : Prop :=
  ∀ (t : ℕ), investment_in_months investment_ratio_PQ profit_ratio_PQ 9 t x → t = 5

end NUMINAMATH_GPT_P_investment_time_l300_30090


namespace NUMINAMATH_GPT_diesel_train_slower_l300_30076

theorem diesel_train_slower
    (t_cattle_speed : ℕ)
    (t_cattle_early_hours : ℕ)
    (t_diesel_hours : ℕ)
    (total_distance : ℕ)
    (diesel_speed : ℕ) :
  t_cattle_speed = 56 →
  t_cattle_early_hours = 6 →
  t_diesel_hours = 12 →
  total_distance = 1284 →
  diesel_speed = 23 →
  t_cattle_speed - diesel_speed = 33 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_diesel_train_slower_l300_30076


namespace NUMINAMATH_GPT_gcd_gx_x_is_210_l300_30018

-- Define the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

-- The main proof problem
theorem gcd_gx_x_is_210 (x : ℕ) (hx : is_multiple_of 17280 x) :
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)) x = 210 :=
by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_is_210_l300_30018


namespace NUMINAMATH_GPT_total_bill_is_95_l300_30067

noncomputable def total_bill := 28 + 8 + 10 + 6 + 14 + 11 + 12 + 6

theorem total_bill_is_95 : total_bill = 95 := by
  sorry

end NUMINAMATH_GPT_total_bill_is_95_l300_30067


namespace NUMINAMATH_GPT_initial_plank_count_l300_30091

def Bedroom := 8
def LivingRoom := 20
def Kitchen := 11
def DiningRoom := 13
def Hallway := 4
def GuestBedroom := Bedroom - 2
def Study := GuestBedroom + 3
def BedroomReplacements := 3
def LivingRoomReplacements := 2
def StudyReplacements := 1
def LeftoverPlanks := 7

def TotalPlanksUsed := 
  (Bedroom + BedroomReplacements) +
  (LivingRoom + LivingRoomReplacements) +
  (Kitchen) +
  (DiningRoom) +
  (GuestBedroom + BedroomReplacements) +
  (Hallway * 2) +
  (Study + StudyReplacements)

theorem initial_plank_count : 
  TotalPlanksUsed + LeftoverPlanks = 91 := 
by
  sorry

end NUMINAMATH_GPT_initial_plank_count_l300_30091


namespace NUMINAMATH_GPT_find_xyz_values_l300_30041

theorem find_xyz_values (x y z : ℝ) (h₁ : x + y + z = Real.pi) (h₂ : x ≥ 0) (h₃ : y ≥ 0) (h₄ : z ≥ 0) :
    (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = Real.pi) ∨
    (x = Real.pi / 6 ∧ y = Real.pi / 3 ∧ z = Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_find_xyz_values_l300_30041


namespace NUMINAMATH_GPT_mona_biked_monday_l300_30074

-- Define the constants and conditions
def distance_biked_weekly : ℕ := 30
def distance_biked_wednesday : ℕ := 12
def speed_flat_road : ℕ := 15
def speed_reduction_percentage : ℕ := 20

-- Define the main problem and conditions in Lean
theorem mona_biked_monday (M : ℕ)
  (h1 : 2 * M + distance_biked_wednesday + M = distance_biked_weekly)  -- total distance biked in the week
  (h2 : 2 * M * (100 - speed_reduction_percentage) / 100 / 15 = 2 * M / 12)  -- speed reduction effect
  : M = 6 :=
sorry 

end NUMINAMATH_GPT_mona_biked_monday_l300_30074


namespace NUMINAMATH_GPT_min_children_l300_30061

theorem min_children (x : ℕ) : 
  (4 * x + 28 - 5 * (x - 1) < 5) ∧ (4 * x + 28 - 5 * (x - 1) ≥ 2) → (x = 29) :=
by
  sorry

end NUMINAMATH_GPT_min_children_l300_30061


namespace NUMINAMATH_GPT_tulips_sum_l300_30073

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end NUMINAMATH_GPT_tulips_sum_l300_30073


namespace NUMINAMATH_GPT_problem_statement_l300_30043

theorem problem_statement (x y : ℝ) (hx : x - y = 3) (hxy : x = 4 ∧ y = 1) : 2 * (x - y) = 6 * y :=
by
  rcases hxy with ⟨hx', hy'⟩
  rw [hx', hy']
  sorry

end NUMINAMATH_GPT_problem_statement_l300_30043


namespace NUMINAMATH_GPT_net_gain_difference_l300_30075

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end NUMINAMATH_GPT_net_gain_difference_l300_30075


namespace NUMINAMATH_GPT_completion_days_together_l300_30054

-- Definitions based on given conditions
variable (W : ℝ) -- Total work
variable (A : ℝ) -- Work done by A in one day
variable (B : ℝ) -- Work done by B in one day

-- Condition 1: A alone completes the work in 20 days
def work_done_by_A := A = W / 20

-- Condition 2: A and B working with B half a day complete the work in 15 days
def work_done_by_A_and_half_B := A + (1 / 2) * B = W / 15

-- Prove: A and B together will complete the work in 60 / 7 days if B works full time
theorem completion_days_together (h1 : work_done_by_A W A) (h2 : work_done_by_A_and_half_B W A B) :
  ∃ D : ℝ, D = 60 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_completion_days_together_l300_30054


namespace NUMINAMATH_GPT_sampling_probabilities_equal_l300_30093

variables (total_items first_grade_items second_grade_items equal_grade_items substandard_items : ℕ)
variables (p_1 p_2 p_3 : ℚ)

-- Conditions given in the problem
def conditions := 
  total_items = 160 ∧ 
  first_grade_items = 48 ∧ 
  second_grade_items = 64 ∧ 
  equal_grade_items = 3 ∧ 
  substandard_items = 1 ∧ 
  p_1 = 1 / 8 ∧ 
  p_2 = 1 / 8 ∧ 
  p_3 = 1 / 8

-- The theorem to be proved
theorem sampling_probabilities_equal (h : conditions total_items first_grade_items second_grade_items equal_grade_items substandard_items p_1 p_2 p_3) :
  p_1 = p_2 ∧ p_2 = p_3 :=
sorry

end NUMINAMATH_GPT_sampling_probabilities_equal_l300_30093


namespace NUMINAMATH_GPT_polynomial_simplification_l300_30017

theorem polynomial_simplification (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l300_30017


namespace NUMINAMATH_GPT_simplify_fraction_l300_30080

theorem simplify_fraction (k : ℝ) : 
  (∃ a b : ℝ, (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a = 1 ∧ b = 3 ∧ (a / b) = 1/3) := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l300_30080


namespace NUMINAMATH_GPT_not_perfect_square_7_pow_2025_all_others_perfect_squares_l300_30050

theorem not_perfect_square_7_pow_2025 :
  ¬ (∃ x : ℕ, x^2 = 7^2025) :=
sorry

theorem all_others_perfect_squares :
  (∃ x : ℕ, x^2 = 6^2024) ∧
  (∃ x : ℕ, x^2 = 8^2026) ∧
  (∃ x : ℕ, x^2 = 9^2027) ∧
  (∃ x : ℕ, x^2 = 10^2028) :=
sorry

end NUMINAMATH_GPT_not_perfect_square_7_pow_2025_all_others_perfect_squares_l300_30050


namespace NUMINAMATH_GPT_total_money_shared_l300_30004

theorem total_money_shared (ratio_jonah ratio_kira ratio_liam kira_share : ℕ)
  (h_ratio : ratio_jonah = 2) (h_ratio2 : ratio_kira = 3) (h_ratio3 : ratio_liam = 8)
  (h_kira : kira_share = 45) :
  (ratio_jonah * (kira_share / ratio_kira) + kira_share + ratio_liam * (kira_share / ratio_kira)) = 195 := 
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l300_30004


namespace NUMINAMATH_GPT_total_employees_with_advanced_degrees_l300_30053

theorem total_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (num_females : ℕ) 
  (num_males_college_only : ℕ) 
  (num_females_advanced_degrees : ℕ)
  (h1 : total_employees = 180)
  (h2 : num_females = 110)
  (h3 : num_males_college_only = 35)
  (h4 : num_females_advanced_degrees = 55) :
  ∃ num_employees_advanced_degrees : ℕ, num_employees_advanced_degrees = 90 :=
by
  have num_males := total_employees - num_females
  have num_males_advanced_degrees := num_males - num_males_college_only
  have num_employees_advanced_degrees := num_males_advanced_degrees + num_females_advanced_degrees
  use num_employees_advanced_degrees
  sorry

end NUMINAMATH_GPT_total_employees_with_advanced_degrees_l300_30053


namespace NUMINAMATH_GPT_payroll_amount_l300_30055

theorem payroll_amount (P : ℝ) 
  (h1 : P > 500000) 
  (h2 : 0.004 * (P - 500000) - 1000 = 600) :
  P = 900000 :=
by
  sorry

end NUMINAMATH_GPT_payroll_amount_l300_30055


namespace NUMINAMATH_GPT_skating_rink_visitors_by_noon_l300_30029

-- Defining the initial conditions
def initial_visitors : ℕ := 264
def visitors_left : ℕ := 134
def visitors_arrived : ℕ := 150

-- Theorem to prove the number of people at the skating rink by noon
theorem skating_rink_visitors_by_noon : initial_visitors - visitors_left + visitors_arrived = 280 := 
by 
  sorry

end NUMINAMATH_GPT_skating_rink_visitors_by_noon_l300_30029


namespace NUMINAMATH_GPT_ratio_of_ages_l300_30058

-- Define the conditions and the main proof goal
theorem ratio_of_ages (R J : ℕ) (Tim_age : ℕ) (h1 : Tim_age = 5) (h2 : J = R + 2) (h3 : J = Tim_age + 12) :
  R / Tim_age = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l300_30058


namespace NUMINAMATH_GPT_find_b_l300_30047

theorem find_b (c b : ℤ) (h : ∃ k : ℤ, (x^2 - x - 1) * (c * x - 3) = c * x^3 + b * x^2 + 3) : b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l300_30047


namespace NUMINAMATH_GPT_Billy_age_l300_30060

-- Defining the ages of Billy, Joe, and Sam
variable (B J S : ℕ)

-- Conditions given in the problem
axiom Billy_twice_Joe : B = 2 * J
axiom sum_BJ_three_times_S : B + J = 3 * S
axiom Sam_age : S = 27

-- Statement to prove
theorem Billy_age : B = 54 :=
by
  sorry

end NUMINAMATH_GPT_Billy_age_l300_30060


namespace NUMINAMATH_GPT_no_positive_a_for_inequality_l300_30082

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_GPT_no_positive_a_for_inequality_l300_30082


namespace NUMINAMATH_GPT_vector_satisfies_condition_l300_30009

def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 5 + 2 * t)
def line_m (s : ℝ) : ℝ × ℝ := (1 + 2 * s, 3 + 2 * s)

variable (A B P : ℝ × ℝ)

def vector_BA (B A : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vector_v : ℝ × ℝ := (1, -1)

theorem vector_satisfies_condition : 
  2 * vector_v.1 - vector_v.2 = 3 := by
  sorry

end NUMINAMATH_GPT_vector_satisfies_condition_l300_30009


namespace NUMINAMATH_GPT_solve_for_x_and_calculate_l300_30096

theorem solve_for_x_and_calculate (x y : ℚ) 
  (h1 : 102 * x - 5 * y = 25) 
  (h2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 :=
by 
  -- These proof steps would solve the problem and validate the theorem
  sorry

end NUMINAMATH_GPT_solve_for_x_and_calculate_l300_30096


namespace NUMINAMATH_GPT_equilateral_triangle_of_angle_and_side_sequences_l300_30087

variable {A B C a b c : ℝ}

theorem equilateral_triangle_of_angle_and_side_sequences
  (H_angles_arithmetic : 2 * B = A + C)
  (H_sum_angles : A + B + C = Real.pi)
  (H_sides_geometric : b^2 = a * c) :
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3 ∧ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_of_angle_and_side_sequences_l300_30087


namespace NUMINAMATH_GPT_square_nonneg_l300_30052

theorem square_nonneg (x h k : ℝ) (h_eq: (x + h)^2 = k) : k ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_square_nonneg_l300_30052


namespace NUMINAMATH_GPT_dryer_less_than_washing_machine_by_30_l300_30070

-- Definitions based on conditions
def washing_machine_price : ℝ := 100
def discount_rate : ℝ := 0.10
def total_paid_after_discount : ℝ := 153

-- The equation for price of the dryer
def original_dryer_price (D : ℝ) : Prop :=
  washing_machine_price + D - discount_rate * (washing_machine_price + D) = total_paid_after_discount

-- The statement we need to prove
theorem dryer_less_than_washing_machine_by_30 (D : ℝ) (h : original_dryer_price D) :
  washing_machine_price - D = 30 :=
by 
  sorry

end NUMINAMATH_GPT_dryer_less_than_washing_machine_by_30_l300_30070


namespace NUMINAMATH_GPT_spencer_total_jumps_l300_30012

noncomputable def jumps_per_minute : ℕ := 4
noncomputable def minutes_per_session : ℕ := 10
noncomputable def sessions_per_day : ℕ := 2
noncomputable def days : ℕ := 5

theorem spencer_total_jumps : 
  (jumps_per_minute * minutes_per_session) * (sessions_per_day * days) = 400 :=
by
  sorry

end NUMINAMATH_GPT_spencer_total_jumps_l300_30012


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l300_30071

theorem geometric_progression_common_ratio (r : ℝ) (a : ℝ) (h_pos : 0 < a)
    (h_geom_prog : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) :
    r^3 + r^2 + r - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l300_30071


namespace NUMINAMATH_GPT_product_of_four_consecutive_is_perfect_square_l300_30031

theorem product_of_four_consecutive_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_is_perfect_square_l300_30031


namespace NUMINAMATH_GPT_value_of_r_minus_p_l300_30001

variable (p q r : ℝ)

-- The conditions given as hypotheses
def arithmetic_mean_pq := (p + q) / 2 = 10
def arithmetic_mean_qr := (q + r) / 2 = 25

-- The goal is to prove that r - p = 30
theorem value_of_r_minus_p (h1: arithmetic_mean_pq p q) (h2: arithmetic_mean_qr q r) :
  r - p = 30 := by
  sorry

end NUMINAMATH_GPT_value_of_r_minus_p_l300_30001


namespace NUMINAMATH_GPT_slope_of_parallel_line_l300_30062

theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) : 
  (5 * x - 3 * y = 12) → m = 5 / 3 → (∃ b : ℝ, y = (5 / 3) * x + b) :=
by
  intro h_eqn h_slope
  use -4 / 3
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l300_30062


namespace NUMINAMATH_GPT_fourth_number_in_sequence_l300_30016

noncomputable def fifth_number_in_sequence : ℕ := 78
noncomputable def increment : ℕ := 11
noncomputable def final_number_in_sequence : ℕ := 89

theorem fourth_number_in_sequence : (fifth_number_in_sequence - increment) = 67 := by
  sorry

end NUMINAMATH_GPT_fourth_number_in_sequence_l300_30016


namespace NUMINAMATH_GPT_solution_l300_30045

def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^5 + p*x^3 + q*x - 8

theorem solution (p q : ℝ) (h : f (-2) p q = 10) : f 2 p q = -26 := by
  sorry

end NUMINAMATH_GPT_solution_l300_30045


namespace NUMINAMATH_GPT_regression_line_is_y_eq_x_plus_1_l300_30092

def Point : Type := ℝ × ℝ

def A : Point := (1, 2)
def B : Point := (2, 3)
def C : Point := (3, 4)
def D : Point := (4, 5)

def points : List Point := [A, B, C, D]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.foldr (fun x acc => x + acc) 0) / lst.length

noncomputable def regression_line (pts : List Point) : ℝ → ℝ :=
  let xs := pts.map Prod.fst
  let ys := pts.map Prod.snd
  fun x : ℝ => x + 1

theorem regression_line_is_y_eq_x_plus_1 :
  regression_line points = fun x => x + 1 := sorry

end NUMINAMATH_GPT_regression_line_is_y_eq_x_plus_1_l300_30092


namespace NUMINAMATH_GPT_percentage_error_l300_30069

theorem percentage_error (e : ℝ) : (1 + e / 100)^2 = 1.1025 → e = 5.125 := 
by sorry

end NUMINAMATH_GPT_percentage_error_l300_30069


namespace NUMINAMATH_GPT_trigo_identity_l300_30026

variable (α : ℝ)

theorem trigo_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (Real.pi / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_trigo_identity_l300_30026


namespace NUMINAMATH_GPT_garden_width_l300_30068

theorem garden_width (w l : ℝ) (h_length : l = 3 * w) (h_area : l * w = 675) : w = 15 :=
by
  sorry

end NUMINAMATH_GPT_garden_width_l300_30068


namespace NUMINAMATH_GPT_john_vegetables_used_l300_30011

noncomputable def pounds_of_beef_bought : ℕ := 4
noncomputable def pounds_of_beef_used : ℕ := pounds_of_beef_bought - 1
noncomputable def pounds_of_vegetables_used : ℕ := 2 * pounds_of_beef_used

theorem john_vegetables_used : pounds_of_vegetables_used = 6 :=
by
  -- the proof can be provided here later
  sorry

end NUMINAMATH_GPT_john_vegetables_used_l300_30011


namespace NUMINAMATH_GPT_divide_shape_into_equal_parts_l300_30036

-- Definitions and conditions
structure Shape where
  has_vertical_symmetry : Bool
  -- Other properties of the shape can be added as necessary

def vertical_line_divides_equally (s : Shape) : Prop :=
  s.has_vertical_symmetry

-- Theorem statement
theorem divide_shape_into_equal_parts (s : Shape) (h : s.has_vertical_symmetry = true) :
  vertical_line_divides_equally s :=
by
  -- Begin proof
  sorry

end NUMINAMATH_GPT_divide_shape_into_equal_parts_l300_30036


namespace NUMINAMATH_GPT_dice_arithmetic_progression_l300_30008

theorem dice_arithmetic_progression :
  let valid_combinations := [
     (1, 1, 1), (1, 3, 2), (1, 5, 3), 
     (2, 4, 3), (2, 6, 4), (3, 3, 3),
     (3, 5, 4), (4, 6, 5), (5, 5, 5)
  ]
  (valid_combinations.length : ℚ) / (6^3 : ℚ) = 1 / 24 :=
  sorry

end NUMINAMATH_GPT_dice_arithmetic_progression_l300_30008


namespace NUMINAMATH_GPT_no_nat_solutions_m2_eq_n2_plus_2014_l300_30033

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end NUMINAMATH_GPT_no_nat_solutions_m2_eq_n2_plus_2014_l300_30033


namespace NUMINAMATH_GPT_compute_mod_expression_l300_30007

theorem compute_mod_expression :
  (3 * (1 / 7) + 9 * (1 / 13)) % 72 = 18 := sorry

end NUMINAMATH_GPT_compute_mod_expression_l300_30007


namespace NUMINAMATH_GPT_unique_solution_l300_30049

noncomputable def check_triplet (a b c : ℕ) : Prop :=
  5^a + 3^b - 2^c = 32

theorem unique_solution : ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ check_triplet a b c ↔ (a = 2 ∧ b = 2 ∧ c = 1) :=
  by sorry

end NUMINAMATH_GPT_unique_solution_l300_30049


namespace NUMINAMATH_GPT_radius_of_circle_l300_30072

theorem radius_of_circle :
  ∃ r : ℝ, ∀ x : ℝ, (x^2 + r = x) ↔ (r = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l300_30072


namespace NUMINAMATH_GPT_polygon_diagonals_150_sides_l300_30000

-- Define the function to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The theorem to state what we want to prove
theorem polygon_diagonals_150_sides : num_diagonals 150 = 11025 :=
by sorry

end NUMINAMATH_GPT_polygon_diagonals_150_sides_l300_30000


namespace NUMINAMATH_GPT_average_income_family_l300_30089

theorem average_income_family (income1 income2 income3 income4 : ℕ) 
  (h1 : income1 = 8000) (h2 : income2 = 15000) (h3 : income3 = 6000) (h4 : income4 = 11000) :
  (income1 + income2 + income3 + income4) / 4 = 10000 := by
  sorry

end NUMINAMATH_GPT_average_income_family_l300_30089


namespace NUMINAMATH_GPT_tan_periodic_mod_l300_30063

theorem tan_periodic_mod (m : ℤ) (h1 : -180 < m) (h2 : m < 180) : 
  (m : ℤ) = 10 := by
  sorry

end NUMINAMATH_GPT_tan_periodic_mod_l300_30063


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l300_30021

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 - 8 * x + c > 0) ↔ (0 < c ∧ c < 16) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l300_30021


namespace NUMINAMATH_GPT_number_of_trees_in_yard_l300_30098

theorem number_of_trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 360 ∧ tree_distance = 12 → 
  (yard_length / tree_distance + 1 = 31) :=
by
  intros yard_length tree_distance h
  have h1 : yard_length = 360 := h.1
  have h2 : tree_distance = 12 := h.2
  sorry

end NUMINAMATH_GPT_number_of_trees_in_yard_l300_30098


namespace NUMINAMATH_GPT_mode_of_scores_is_85_l300_30059

-- Define the scores based on the given stem-and-leaf plot
def scores : List ℕ := [50, 55, 55, 62, 62, 68, 70, 71, 75, 79, 81, 81, 83, 85, 85, 85, 92, 96, 96, 98, 100, 100]

-- Define a function to compute the mode
def mode (s : List ℕ) : ℕ :=
  s.foldl (λ acc x => if s.count x > s.count acc then x else acc) 0

-- The theorem to prove that the mode of the scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_mode_of_scores_is_85_l300_30059


namespace NUMINAMATH_GPT_rojo_speed_l300_30038

theorem rojo_speed (R : ℝ) 
  (H : 32 = (R + 3) * 4) : R = 5 :=
sorry

end NUMINAMATH_GPT_rojo_speed_l300_30038


namespace NUMINAMATH_GPT_Ma_Xiaohu_speed_l300_30095

theorem Ma_Xiaohu_speed
  (distance_home_school : ℕ := 1800)
  (distance_to_school : ℕ := 1600)
  (father_speed_factor : ℕ := 2)
  (time_difference : ℕ := 10)
  (x : ℕ)
  (hx : distance_home_school - distance_to_school = 200)
  (hspeed : father_speed_factor * x = 2 * x)
  :
  (distance_to_school / x) - (distance_to_school / (2 * x)) = time_difference ↔ x = 80 :=
by
  sorry

end NUMINAMATH_GPT_Ma_Xiaohu_speed_l300_30095


namespace NUMINAMATH_GPT_coin_flip_sequences_count_l300_30015

noncomputable def num_sequences_with_given_occurrences : ℕ :=
  sorry

theorem coin_flip_sequences_count : num_sequences_with_given_occurrences = 560 :=
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_count_l300_30015


namespace NUMINAMATH_GPT_work_rate_solution_l300_30034

theorem work_rate_solution (y : ℕ) (hy : y > 0) : 
  ∃ z : ℕ, z = (y^2 + 3 * y) / (2 * y + 3) :=
by
  sorry

end NUMINAMATH_GPT_work_rate_solution_l300_30034


namespace NUMINAMATH_GPT_A_rotated_l300_30065

-- Define initial coordinates of point A
def A_initial : ℝ × ℝ := (1, 2)

-- Define the transformation for a 180-degree clockwise rotation around the origin
def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- The Lean statement to prove the coordinates after the rotation
theorem A_rotated : rotate_180_deg A_initial = (-1, -2) :=
by
  sorry

end NUMINAMATH_GPT_A_rotated_l300_30065


namespace NUMINAMATH_GPT_ratio_aerobics_to_weight_training_l300_30088

def time_spent_exercising : ℕ := 250
def time_spent_aerobics : ℕ := 150
def time_spent_weight_training : ℕ := 100

theorem ratio_aerobics_to_weight_training :
    (time_spent_aerobics / gcd time_spent_aerobics time_spent_weight_training) = 3 ∧
    (time_spent_weight_training / gcd time_spent_aerobics time_spent_weight_training) = 2 :=
by
    sorry

end NUMINAMATH_GPT_ratio_aerobics_to_weight_training_l300_30088


namespace NUMINAMATH_GPT_soybean_cornmeal_proof_l300_30078

theorem soybean_cornmeal_proof :
  ∃ (x y : ℝ), 
    (0.14 * x + 0.07 * y = 0.13 * 280) ∧
    (x + y = 280) ∧
    (x = 240) ∧
    (y = 40) :=
by
  sorry

end NUMINAMATH_GPT_soybean_cornmeal_proof_l300_30078


namespace NUMINAMATH_GPT_tan_sum_l300_30044

theorem tan_sum (θ : ℝ) (h : Real.sin (2 * θ) = 2 / 3) : Real.tan θ + 1 / Real.tan θ = 3 := sorry

end NUMINAMATH_GPT_tan_sum_l300_30044


namespace NUMINAMATH_GPT_num_real_a_satisfy_union_l300_30010

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end NUMINAMATH_GPT_num_real_a_satisfy_union_l300_30010


namespace NUMINAMATH_GPT_part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l300_30014

def set_A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def set_B (m : ℝ) : Set ℝ := {x | x < m}

-- Problem 1
theorem part1_A_complement_B_intersection_eq (m : ℝ) (h : m = 3) :
  set_A ∩ {x | x >= 3} = {x | 3 <= x ∧ x < 4} :=
sorry

-- Problem 2
theorem part2_m_le_neg2 (m : ℝ) (h : set_A ∩ set_B m = ∅) :
  m <= -2 :=
sorry

-- Problem 3
theorem part3_m_ge_4 (m : ℝ) (h : set_A ∩ set_B m = set_A) :
  m >= 4 :=
sorry

end NUMINAMATH_GPT_part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l300_30014


namespace NUMINAMATH_GPT_four_x_plus_y_greater_than_four_z_l300_30066

theorem four_x_plus_y_greater_than_four_z
  (x y z : ℝ)
  (h1 : y > 2 * z)
  (h2 : 2 * z > 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) > 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z)
  : 4 * x + y > 4 * z := 
by
  sorry

end NUMINAMATH_GPT_four_x_plus_y_greater_than_four_z_l300_30066


namespace NUMINAMATH_GPT_street_sweeper_routes_l300_30046

def num_routes (A B C : Type) :=
  -- Conditions: Starts from point A, 
  -- travels through all streets exactly once, 
  -- and returns to point A.
  -- Correct Answer: Total routes = 12
  2 * 6 = 12

theorem street_sweeper_routes (A B C : Type) : num_routes A B C := by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_street_sweeper_routes_l300_30046


namespace NUMINAMATH_GPT_probability_two_red_two_blue_one_green_l300_30002

def total_ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_choose_red (r : ℕ) : ℕ := total_ways_to_choose 4 r
def ways_to_choose_blue (b : ℕ) : ℕ := total_ways_to_choose 3 b
def ways_to_choose_green (g : ℕ) : ℕ := total_ways_to_choose 2 g

def successful_outcomes (r b g : ℕ) : ℕ :=
  ways_to_choose_red r * ways_to_choose_blue b * ways_to_choose_green g

def total_outcomes : ℕ := total_ways_to_choose 9 5

def probability_of_selection (r b g : ℕ) : ℚ :=
  (successful_outcomes r b g : ℚ) / (total_outcomes : ℚ)

theorem probability_two_red_two_blue_one_green :
  probability_of_selection 2 2 1 = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_probability_two_red_two_blue_one_green_l300_30002


namespace NUMINAMATH_GPT_sector_area_l300_30079

/-- The area of a sector with a central angle of 72 degrees and a radius of 20 cm is 80π cm². -/
theorem sector_area (radius : ℝ) (angle : ℝ) (h_angle_deg : angle = 72) (h_radius : radius = 20) :
  (angle / 360) * π * radius^2 = 80 * π :=
by sorry

end NUMINAMATH_GPT_sector_area_l300_30079


namespace NUMINAMATH_GPT_sequence_contains_30_l300_30027

theorem sequence_contains_30 :
  ∃ n : ℕ, n * (n + 1) = 30 :=
sorry

end NUMINAMATH_GPT_sequence_contains_30_l300_30027


namespace NUMINAMATH_GPT_sequence_divisible_by_three_l300_30056

-- Define the conditions
variable (k : ℕ) (h_pos_k : k > 0)
variable (a : ℕ → ℤ)
variable (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n)

-- Define the proof goal
theorem sequence_divisible_by_three (k : ℕ) (h_pos_k : k > 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n) : (k - 2) % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sequence_divisible_by_three_l300_30056


namespace NUMINAMATH_GPT_purely_imaginary_z_eq_a2_iff_a2_l300_30064

theorem purely_imaginary_z_eq_a2_iff_a2 (a : Real) : 
(∃ (b : Real), a^2 - a - 2 = 0 ∧ a + 1 ≠ 0) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_z_eq_a2_iff_a2_l300_30064


namespace NUMINAMATH_GPT_monotone_increasing_range_of_a_l300_30013

noncomputable def f (a x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_increasing_range_of_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Icc (-1 / 3 : ℝ) (1 / 3 : ℝ)) :=
sorry

end NUMINAMATH_GPT_monotone_increasing_range_of_a_l300_30013


namespace NUMINAMATH_GPT_four_digit_number_difference_l300_30023

theorem four_digit_number_difference
    (digits : List ℕ) (h_digits : digits = [2, 0, 1, 3, 1, 2, 2, 1, 0, 8, 4, 0])
    (max_val : ℕ) (h_max_val : max_val = 3840)
    (min_val : ℕ) (h_min_val : min_val = 1040) :
    max_val - min_val = 2800 :=
by
    sorry

end NUMINAMATH_GPT_four_digit_number_difference_l300_30023


namespace NUMINAMATH_GPT_length_of_train_is_correct_l300_30048

noncomputable def speed_kmh := 30 
noncomputable def time_s := 9 
noncomputable def speed_ms := (speed_kmh * 1000) / 3600 
noncomputable def length_of_train := speed_ms * time_s

theorem length_of_train_is_correct : length_of_train = 75 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_train_is_correct_l300_30048


namespace NUMINAMATH_GPT_greatest_positive_integer_x_l300_30084

theorem greatest_positive_integer_x : ∃ (x : ℕ), (x > 0) ∧ (∀ y : ℕ, y > 0 → (y^3 < 20 * y → y ≤ 4)) ∧ (x^3 < 20 * x) ∧ ∀ z : ℕ, (z > 0) → (z^3 < 20 * z → x ≥ z)  :=
sorry

end NUMINAMATH_GPT_greatest_positive_integer_x_l300_30084


namespace NUMINAMATH_GPT_pilot_weeks_l300_30077

-- Given conditions
def milesTuesday : ℕ := 1134
def milesThursday : ℕ := 1475
def totalMiles : ℕ := 7827

-- Calculate total miles flown in one week
def milesPerWeek : ℕ := milesTuesday + milesThursday

-- Define the proof problem statement
theorem pilot_weeks (w : ℕ) (h : w * milesPerWeek = totalMiles) : w = 3 :=
by
  -- Here we would provide the proof, but we leave it with a placeholder
  sorry

end NUMINAMATH_GPT_pilot_weeks_l300_30077


namespace NUMINAMATH_GPT_conic_not_parabola_l300_30051

def conic_equation (m x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

theorem conic_not_parabola (m : ℝ) :
  ¬ (∃ (x y : ℝ), conic_equation m x y ∧ ∃ (a b c d e f : ℝ), m * x^2 + (m + 1) * y^2 = a * x^2 + b * xy + c * y^2 + d * x + e * y + f ∧ (a = 0 ∨ c = 0) ∧ (b ≠ 0 ∨ a ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0)) :=  
sorry

end NUMINAMATH_GPT_conic_not_parabola_l300_30051


namespace NUMINAMATH_GPT_slope_of_line_l300_30035

theorem slope_of_line (θ : ℝ) (h_cosθ : (Real.cos θ) = 4/5) : (Real.sin θ) / (Real.cos θ) = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l300_30035


namespace NUMINAMATH_GPT_average_speed_of_car_l300_30083

theorem average_speed_of_car : 
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  total_distance / total_time = 55 := 
by
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  show total_distance / total_time = 55
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l300_30083


namespace NUMINAMATH_GPT_jason_advertising_cost_l300_30005

def magazine_length : ℕ := 9
def magazine_width : ℕ := 12
def cost_per_square_inch : ℕ := 8
def half (x : ℕ) := x / 2
def area (L W : ℕ) := L * W
def total_cost (a c : ℕ) := a * c

theorem jason_advertising_cost :
  total_cost (half (area magazine_length magazine_width)) cost_per_square_inch = 432 := by
  sorry

end NUMINAMATH_GPT_jason_advertising_cost_l300_30005


namespace NUMINAMATH_GPT_value_of_a_plus_b_l300_30042

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem value_of_a_plus_b (a b : ℝ) (h1 : 3 * a + b = 4) (h2 : a + b + 1 = 3) : a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l300_30042


namespace NUMINAMATH_GPT_age_of_B_l300_30030

theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 37) : B = 14 :=
by sorry

end NUMINAMATH_GPT_age_of_B_l300_30030


namespace NUMINAMATH_GPT_cos_sin_identity_l300_30006

theorem cos_sin_identity (x : ℝ) (h : Real.cos (x - Real.pi / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * Real.pi / 3) + Real.sin (Real.pi / 3 - x) ^ 2 = 5 / 3 :=
sorry

end NUMINAMATH_GPT_cos_sin_identity_l300_30006


namespace NUMINAMATH_GPT_stephanie_running_time_l300_30057

theorem stephanie_running_time
  (Speed : ℝ) (Distance : ℝ) (Time : ℝ)
  (h1 : Speed = 5)
  (h2 : Distance = 15)
  (h3 : Time = Distance / Speed) :
  Time = 3 :=
sorry

end NUMINAMATH_GPT_stephanie_running_time_l300_30057


namespace NUMINAMATH_GPT_yoonseok_handshakes_l300_30099

-- Conditions
def totalFriends : ℕ := 12
def yoonseok := "Yoonseok"
def adjacentFriends (i : ℕ) : Prop := i = 1 ∨ i = (totalFriends - 1)

-- Problem Statement
theorem yoonseok_handshakes : 
  ∀ (totalFriends : ℕ) (adjacentFriends : ℕ → Prop), 
    totalFriends = 12 → 
    (∀ i, adjacentFriends i ↔ i = 1 ∨ i = (totalFriends - 1)) → 
    (totalFriends - 1 - 2 = 9) := by
  intros totalFriends adjacentFriends hTotal hAdjacent
  have hSub : totalFriends - 1 - 2 = 9 := by sorry
  exact hSub

end NUMINAMATH_GPT_yoonseok_handshakes_l300_30099


namespace NUMINAMATH_GPT_age_ratio_l300_30094

theorem age_ratio (R D : ℕ) (hR : R + 4 = 32) (hD : D = 21) : R / D = 4 / 3 := 
by sorry

end NUMINAMATH_GPT_age_ratio_l300_30094


namespace NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l300_30097

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_equilateral_triangle_l300_30097


namespace NUMINAMATH_GPT_find_number_l300_30085

theorem find_number (x : ℝ) : ((x - 50) / 4) * 3 + 28 = 73 → x = 110 := 
  by 
  sorry

end NUMINAMATH_GPT_find_number_l300_30085


namespace NUMINAMATH_GPT_melanie_gave_mother_l300_30020

theorem melanie_gave_mother {initial_dimes dad_dimes final_dimes dimes_given : ℕ}
  (h₁ : initial_dimes = 7)
  (h₂ : dad_dimes = 8)
  (h₃ : final_dimes = 11)
  (h₄ : initial_dimes + dad_dimes - dimes_given = final_dimes) :
  dimes_given = 4 :=
by 
  sorry

end NUMINAMATH_GPT_melanie_gave_mother_l300_30020


namespace NUMINAMATH_GPT_max_product_decomposition_l300_30039

theorem max_product_decomposition : ∃ x y : ℝ, x + y = 100 ∧ x * y = 50 * 50 := by
  sorry

end NUMINAMATH_GPT_max_product_decomposition_l300_30039
