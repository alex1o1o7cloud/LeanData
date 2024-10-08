import Mathlib

namespace power_of_two_l228_228432

theorem power_of_two (n : ℕ) (h : 2^n = 32 * (1 / 2) ^ 2) : n = 3 :=
by {
  sorry
}

end power_of_two_l228_228432


namespace flowers_per_bug_l228_228202

theorem flowers_per_bug (bugs : ℝ) (flowers : ℝ) (h_bugs : bugs = 2.0) (h_flowers : flowers = 3.0) :
  flowers / bugs = 1.5 :=
by
  sorry

end flowers_per_bug_l228_228202


namespace Billy_age_l228_228965

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

end Billy_age_l228_228965


namespace total_value_of_item_l228_228113

variable (V : ℝ) -- Total value of the item

def import_tax (V : ℝ) := 0.07 * (V - 1000) -- Definition of import tax

theorem total_value_of_item
  (htax_paid : import_tax V = 112.70) :
  V = 2610 := 
by
  sorry

end total_value_of_item_l228_228113


namespace problem_statement_l228_228881

theorem problem_statement (x y : ℝ) (hx : x - y = 3) (hxy : x = 4 ∧ y = 1) : 2 * (x - y) = 6 * y :=
by
  rcases hxy with ⟨hx', hy'⟩
  rw [hx', hy']
  sorry

end problem_statement_l228_228881


namespace find_b_l228_228376

theorem find_b (b : ℕ) (h1 : 40 < b) (h2 : b < 120) 
    (h3 : b % 4 = 3) (h4 : b % 5 = 3) (h5 : b % 6 = 3) : 
    b = 63 := by
  sorry

end find_b_l228_228376


namespace find_k_l228_228155

theorem find_k (k : ℚ) : 
  ((3, -8) ≠ (k, 20)) ∧ 
  (∃ m, (4 * m = -3) ∧ (20 - (-8) = m * (k - 3))) → 
  k = -103/3 := 
by
  sorry

end find_k_l228_228155


namespace exists_i_for_inequality_l228_228657

theorem exists_i_for_inequality (n : ℕ) (x : ℕ → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) :=
by
  sorry

end exists_i_for_inequality_l228_228657


namespace sampling_probabilities_equal_l228_228944

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

end sampling_probabilities_equal_l228_228944


namespace solution_mix_percentage_l228_228639

theorem solution_mix_percentage
  (x y z : ℝ)
  (hx1 : x + y + z = 100)
  (hx2 : 0.40 * x + 0.50 * y + 0.30 * z = 46)
  (hx3 : z = 100 - x - y) :
  x = 40 ∧ y = 60 ∧ z = 0 :=
by
  sorry

end solution_mix_percentage_l228_228639


namespace find_three_digit_perfect_square_l228_228576

noncomputable def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n % 100) / 10) * (n % 10)

theorem find_three_digit_perfect_square :
  ∃ (n H : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (n = H * H) ∧ (digit_product n = H - 1) :=
by {
  sorry
}

end find_three_digit_perfect_square_l228_228576


namespace cashier_can_satisfy_request_l228_228695

theorem cashier_can_satisfy_request (k : ℕ) (h : k > 8) : ∃ m n : ℕ, k = 3 * m + 5 * n :=
sorry

end cashier_can_satisfy_request_l228_228695


namespace sufficient_not_necessary_a_eq_one_l228_228762

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x + f (-x) = 0

theorem sufficient_not_necessary_a_eq_one 
  (a : ℝ) 
  (h₁ : a = 1) 
  : is_odd_function (f a) := sorry

end sufficient_not_necessary_a_eq_one_l228_228762


namespace angle_at_630_is_15_degrees_l228_228375

-- Definitions for positions of hour and minute hands at 6:30 p.m.
def angle_per_hour : ℝ := 30
def minute_hand_position_630 : ℝ := 180
def hour_hand_position_630 : ℝ := 195

-- The angle between the hour hand and minute hand at 6:30 p.m.
def angle_between_hands_630 : ℝ := |hour_hand_position_630 - minute_hand_position_630|

-- Statement to prove
theorem angle_at_630_is_15_degrees :
  angle_between_hands_630 = 15 := by
  sorry

end angle_at_630_is_15_degrees_l228_228375


namespace unique_solution_l228_228915

noncomputable def check_triplet (a b c : ℕ) : Prop :=
  5^a + 3^b - 2^c = 32

theorem unique_solution : ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ check_triplet a b c ↔ (a = 2 ∧ b = 2 ∧ c = 1) :=
  by sorry

end unique_solution_l228_228915


namespace initial_overs_played_l228_228164

-- Define the conditions
def initial_run_rate : ℝ := 6.2
def remaining_overs : ℝ := 40
def remaining_run_rate : ℝ := 5.5
def target_runs : ℝ := 282

-- Define what we seek to prove
theorem initial_overs_played :
  ∃ x : ℝ, (6.2 * x) + (5.5 * 40) = 282 ∧ x = 10 :=
by
  sorry

end initial_overs_played_l228_228164


namespace arithmetic_sequence_sum_l228_228275

variable (S : ℕ → ℕ) -- Define a function S that gives the sum of the first n terms.
variable (n : ℕ)     -- Define a natural number n.

-- Conditions based on the problem statement
axiom h1 : S n = 3
axiom h2 : S (2 * n) = 10

-- The theorem we need to prove
theorem arithmetic_sequence_sum : S (3 * n) = 21 :=
by
  sorry

end arithmetic_sequence_sum_l228_228275


namespace radius_of_circle_l228_228952

theorem radius_of_circle :
  ∃ r : ℝ, ∀ x : ℝ, (x^2 + r = x) ↔ (r = 1 / 4) :=
by
  sorry

end radius_of_circle_l228_228952


namespace xiaoyu_money_left_l228_228019

def box_prices (x y z : ℝ) : Prop :=
  2 * x + 5 * y = z + 3 ∧ 5 * x + 2 * y = z - 3

noncomputable def money_left (x y z : ℝ) : ℝ :=
  z - 7 * x
  
theorem xiaoyu_money_left (x y z : ℝ) (hx : box_prices x y z) :
  money_left x y z = 7 := by
  sorry

end xiaoyu_money_left_l228_228019


namespace second_set_parallel_lines_l228_228293

theorem second_set_parallel_lines (n : ℕ) (h1 : 5 * (n - 1) = 420) : n = 85 :=
by sorry

end second_set_parallel_lines_l228_228293


namespace triangle_area_16_l228_228450

theorem triangle_area_16 : 
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 8)
  let base := (B.1 - A.1)
  let height := (C.2 - A.2)
  (base * height) / 2 = 16 := by
  sorry

end triangle_area_16_l228_228450


namespace football_outcomes_l228_228047

theorem football_outcomes : 
  ∃ (W D L : ℕ), (3 * W + D = 19) ∧ (W + D + L = 14) ∧ 
  ((W = 3 ∧ D = 10 ∧ L = 1) ∨ 
   (W = 4 ∧ D = 7 ∧ L = 3) ∨ 
   (W = 5 ∧ D = 4 ∧ L = 5) ∨ 
   (W = 6 ∧ D = 1 ∧ L = 7)) ∧
  (∀ W' D' L' : ℕ, (3 * W' + D' = 19) → (W' + D' + L' = 14) → 
    (W' = 3 ∧ D' = 10 ∧ L' = 1) ∨ 
    (W' = 4 ∧ D' = 7 ∧ L' = 3) ∨ 
    (W' = 5 ∧ D' = 4 ∧ L' = 5) ∨ 
    (W' = 6 ∧ D' = 1 ∧ L' = 7)) := 
sorry

end football_outcomes_l228_228047


namespace third_quadrant_angle_bisector_l228_228303

theorem third_quadrant_angle_bisector
  (a b : ℝ)
  (hA : A = (-4,a))
  (hB : B = (-2,b))
  (h_lineA : a = -4)
  (h_lineB : b = -2)
  : a + b + a * b = 2 :=
by
  sorry

end third_quadrant_angle_bisector_l228_228303


namespace pictures_on_front_l228_228017

-- Conditions
variable (total_pictures : ℕ)
variable (pictures_on_back : ℕ)

-- Proof obligation
theorem pictures_on_front (h1 : total_pictures = 15) (h2 : pictures_on_back = 9) : total_pictures - pictures_on_back = 6 :=
sorry

end pictures_on_front_l228_228017


namespace value_of_f_neg2_l228_228779

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem value_of_f_neg2 (a b c : ℝ) (h1 : f a b c 5 + f a b c (-5) = 6) (h2 : f a b c 2 = 8) :
  f a b c (-2) = -2 := by
  sorry

end value_of_f_neg2_l228_228779


namespace divided_number_l228_228756

theorem divided_number (x y : ℕ) (h1 : 7 * x + 5 * y = 146) (h2 : y = 11) : x + y = 24 :=
sorry

end divided_number_l228_228756


namespace max_value_frac_l228_228247

theorem max_value_frac (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  ∃ z, z = (x + y) / x ∧ z ≤ 2 / 3 := by
  sorry

end max_value_frac_l228_228247


namespace product_of_binomials_l228_228224

theorem product_of_binomials :
  (2*x^2 + 3*x - 4) * (x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 :=
by {
  sorry
}

end product_of_binomials_l228_228224


namespace unique_not_in_range_of_g_l228_228200

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g (p q r s : ℝ) (hps_qr_zero : p * s + q * r = 0) 
  (hpr_rs_zero : p * r + r * s = 0) (hg3 : g p q r s 3 = 3) 
  (hg81 : g p q r s 81 = 81) (h_involution : ∀ x ≠ (-s / r), g p q r s (g p q r s x) = x) :
  ∀ x : ℝ, x ≠ 42 :=
sorry

end unique_not_in_range_of_g_l228_228200


namespace original_cost_of_statue_l228_228488

theorem original_cost_of_statue (sale_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : sale_price = 620) 
  (h2 : profit_percent = 0.25) 
  (h3 : sale_price = (1 + profit_percent) * original_cost) : 
  original_cost = 496 :=
by
  sorry

end original_cost_of_statue_l228_228488


namespace min_value_2x_plus_y_l228_228377

theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 ∧ (∀ y : ℝ, |y| ≤ 2 - x → x ≥ -1 → 2 * x + y ≥ -5) ∧ (2 * x + y = -5) :=
by
  sorry

end min_value_2x_plus_y_l228_228377


namespace find_original_rabbits_l228_228864

theorem find_original_rabbits (R S : ℕ) (h1 : R + S = 50)
  (h2 : 4 * R + 8 * S = 2 * R + 16 * S) :
  R = 40 :=
sorry

end find_original_rabbits_l228_228864


namespace perfect_square_pattern_l228_228734

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l228_228734


namespace pyramid_side_length_l228_228420

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l228_228420


namespace equation_of_tangent_circle_l228_228612

/-- Lean Statement for the circle problem -/
theorem equation_of_tangent_circle (center_C : ℝ × ℝ)
    (h1 : ∃ x, center_C = (x, 0) ∧ x - 0 + 1 = 0)
    (circle_tangent : ∃ r, ((2 - (center_C.1))^2 + (3 - (center_C.2))^2 = (2 * Real.sqrt 2) + r)) :
    ∃ r, (x + 1)^2 + y^2 = r^2 := 
sorry

end equation_of_tangent_circle_l228_228612


namespace solve_w_from_system_of_equations_l228_228628

open Real

variables (w x y z : ℝ)

theorem solve_w_from_system_of_equations
  (h1 : 2 * w + x + y + z = 1)
  (h2 : w + 2 * x + y + z = 2)
  (h3 : w + x + 2 * y + z = 2)
  (h4 : w + x + y + 2 * z = 1) :
  w = -1 / 5 :=
by
  sorry

end solve_w_from_system_of_equations_l228_228628


namespace identify_irrational_number_l228_228967

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end identify_irrational_number_l228_228967


namespace largest_lcm_l228_228360

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end largest_lcm_l228_228360


namespace trig_expression_identity_l228_228518

theorem trig_expression_identity (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) : 
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 :=
by
  sorry

end trig_expression_identity_l228_228518


namespace net_gain_difference_l228_228961

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

end net_gain_difference_l228_228961


namespace range_of_a_l228_228439

theorem range_of_a (a : ℝ) : (forall x : ℝ, (a-3) * x > 1 → x < 1 / (a-3)) → a < 3 :=
by
  sorry

end range_of_a_l228_228439


namespace geometric_sequence_ratio_l228_228195
-- Lean 4 Code

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ n

theorem geometric_sequence_ratio
  (a : ℝ)
  (q : ℝ)
  (h_pos : a > 0)
  (h_q_neq_1 : q ≠ 1)
  (h_arith_seq : 2 * a_n a q 4 = a_n a q 2 + a_n a q 5)
  : (a_n a q 2 + a_n a q 3) / (a_n a q 3 + a_n a q 4) = (Real.sqrt 5 - 1) / 2 :=
by {
  sorry
}

end geometric_sequence_ratio_l228_228195


namespace trig_identity_l228_228150

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
    Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := 
by 
  sorry

end trig_identity_l228_228150


namespace speed_of_current_l228_228048

theorem speed_of_current (d : ℝ) (c : ℝ) : 
  ∀ (h1 : ∀ (t : ℝ), d = (30 - c) * (40 / 60)) (h2 : ∀ (t : ℝ), d = (30 + c) * (25 / 60)), 
  c = 90 / 13 := by
  sorry

end speed_of_current_l228_228048


namespace Lucy_retirement_month_l228_228027

theorem Lucy_retirement_month (start_month : ℕ) (duration : ℕ) (March : ℕ) (May : ℕ) : 
  (start_month = March) ∧ (duration = 3) → (start_month + duration - 1 = May) :=
by
  intro h
  have h_start_month := h.1
  have h_duration := h.2
  sorry

end Lucy_retirement_month_l228_228027


namespace arccos_cos_10_l228_228365

theorem arccos_cos_10 : Real.arccos (Real.cos 10) = 2 := by
  sorry

end arccos_cos_10_l228_228365


namespace skating_rink_visitors_by_noon_l228_228884

-- Defining the initial conditions
def initial_visitors : ℕ := 264
def visitors_left : ℕ := 134
def visitors_arrived : ℕ := 150

-- Theorem to prove the number of people at the skating rink by noon
theorem skating_rink_visitors_by_noon : initial_visitors - visitors_left + visitors_arrived = 280 := 
by 
  sorry

end skating_rink_visitors_by_noon_l228_228884


namespace divisibility_by_2880_l228_228327

theorem divisibility_by_2880 (n : ℕ) : 
  (∃ t u : ℕ, (n = 16 * t - 2 ∨ n = 16 * t + 2 ∨ n = 8 * u - 1 ∨ n = 8 * u + 1) ∧ ¬(n % 3 = 0) ∧ ¬(n % 5 = 0)) ↔
  2880 ∣ (n^2 - 4) * (n^2 - 1) * (n^2 + 3) :=
sorry

end divisibility_by_2880_l228_228327


namespace boat_speed_24_l228_228234

def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  let speed_downstream := x + 3
  let time := 1 / 4 -- 15 minutes in hours
  let distance := 6.75
  let equation := distance = speed_downstream * time
  equation ∧ x = 24

theorem boat_speed_24 (x : ℝ) (rate_of_current : ℝ) (time_minutes : ℝ) (distance_traveled : ℝ) 
  (h1 : rate_of_current = 3) (h2 : time_minutes = 15) (h3 : distance_traveled = 6.75) : speed_of_boat_in_still_water 24 := 
by
  -- Convert time in minutes to hours
  have time_in_hours : ℝ := time_minutes / 60
  -- Effective downstream speed
  have effective_speed := 24 + rate_of_current
  -- The equation to be satisfied
  have equation := distance_traveled = effective_speed * time_in_hours
  -- Simplify and solve
  sorry

end boat_speed_24_l228_228234


namespace girl_boy_lineup_probability_l228_228308

theorem girl_boy_lineup_probability :
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  (valid_configurations : ℚ) / total_configurations = 0.058 :=
by
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  have h : (valid_configurations : ℚ) / total_configurations = 0.058 := sorry
  exact h

end girl_boy_lineup_probability_l228_228308


namespace Ma_Xiaohu_speed_l228_228920

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

end Ma_Xiaohu_speed_l228_228920


namespace nylon_needed_for_one_dog_collor_l228_228621

-- Define the conditions as given in the problem
def nylon_for_dog (x : ℝ) : ℝ := x
def nylon_for_cat : ℝ := 10
def total_nylon_used (x : ℝ) : ℝ := 9 * (nylon_for_dog x) + 3 * (nylon_for_cat)

-- Prove the required statement under the given conditions
theorem nylon_needed_for_one_dog_collor : total_nylon_used 18 = 192 :=
by
  -- adding the proof step using sorry as required
  sorry

end nylon_needed_for_one_dog_collor_l228_228621


namespace principle_calculation_l228_228146

noncomputable def calculate_principal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principle_calculation :
  calculate_principal 1456 0.05 2.4 = 1300 :=
by
  sorry

end principle_calculation_l228_228146


namespace Sam_distance_l228_228343

theorem Sam_distance (miles_Marguerite: ℝ) (hours_Marguerite: ℕ) (hours_Sam: ℕ) (speed_factor: ℝ) 
  (h1: miles_Marguerite = 150) 
  (h2: hours_Marguerite = 3) 
  (h3: hours_Sam = 4)
  (h4: speed_factor = 1.2) :
  let average_speed_Marguerite := miles_Marguerite / hours_Marguerite
  let average_speed_Sam := speed_factor * average_speed_Marguerite
  let distance_Sam := average_speed_Sam * hours_Sam
  distance_Sam = 240 := 
by 
  sorry

end Sam_distance_l228_228343


namespace cost_of_flight_XY_l228_228093

theorem cost_of_flight_XY :
  let d_XY : ℕ := 4800
  let booking_fee : ℕ := 150
  let cost_per_km : ℚ := 0.12
  ∃ cost : ℚ, cost = d_XY * cost_per_km + booking_fee ∧ cost = 726 := 
by
  sorry

end cost_of_flight_XY_l228_228093


namespace polygon_diagonals_150_sides_l228_228855

-- Define the function to calculate the number of diagonals in a polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The theorem to state what we want to prove
theorem polygon_diagonals_150_sides : num_diagonals 150 = 11025 :=
by sorry

end polygon_diagonals_150_sides_l228_228855


namespace probability_two_red_two_green_l228_228914

theorem probability_two_red_two_green (total_red total_blue total_green : ℕ)
  (total_marbles total_selected : ℕ) (probability : ℚ)
  (h_total_marbles: total_marbles = total_red + total_blue + total_green)
  (h_total_selected: total_selected = 4)
  (h_red_selected: 2 ≤ total_red)
  (h_green_selected: 2 ≤ total_green)
  (h_total_selected_le: total_selected ≤ total_marbles)
  (h_probability: probability = (Nat.choose total_red 2 * Nat.choose total_green 2) / (Nat.choose total_marbles total_selected))
  (h_total_red: total_red = 12)
  (h_total_blue: total_blue = 8)
  (h_total_green: total_green = 5):
  probability = 2 / 39 :=
by
  sorry

end probability_two_red_two_green_l228_228914


namespace find_m_l228_228179

theorem find_m (S : ℕ → ℝ) (m : ℕ) (h1 : S m = -2) (h2 : S (m+1) = 0) (h3 : S (m+2) = 3) : m = 4 :=
by
  sorry

end find_m_l228_228179


namespace flour_needed_l228_228064

theorem flour_needed (sugar flour : ℕ) (h1 : sugar = 50) (h2 : sugar / 10 = flour) : flour = 5 :=
by
  sorry

end flour_needed_l228_228064


namespace determine_k_l228_228349

-- Definitions of the vectors a and b.
variables (a b : ℝ)

-- Noncomputable definition of the scalar k.
noncomputable def k_value : ℝ :=
  (2 : ℚ) / 7

-- Definition of line through vectors a and b as a parametric equation.
def line_through (a b : ℝ) (t : ℝ) : ℝ :=
  a + t * (b - a)

-- Hypothesis: The vector k * a + (5/7) * b is on the line passing through a and b.
def vector_on_line (a b : ℝ) (k : ℝ) : Prop :=
  ∃ t : ℝ, k * a + (5/7) * b = line_through a b t

-- Proof that k must be 2/7 for the vector to be on the line.
theorem determine_k (a b : ℝ) : vector_on_line a b k_value :=
by sorry

end determine_k_l228_228349


namespace exterior_angle_parallel_lines_l228_228448

theorem exterior_angle_parallel_lines
  (k l : Prop) 
  (triangle_has_angles : ∃ (a b c : ℝ), a = 40 ∧ b = 40 ∧ c = 100 ∧ a + b + c = 180)
  (exterior_angle_eq : ∀ (y : ℝ), y = 180 - 100) :
  ∃ (x : ℝ), x = 80 :=
by
  sorry

end exterior_angle_parallel_lines_l228_228448


namespace Tod_speed_is_25_mph_l228_228193

-- Definitions of the conditions
def miles_north : ℕ := 55
def miles_west : ℕ := 95
def hours_driven : ℕ := 6

-- The total distance travelled
def total_distance : ℕ := miles_north + miles_west

-- The speed calculation, dividing total distance by hours driven
def speed : ℕ := total_distance / hours_driven

-- The theorem to prove
theorem Tod_speed_is_25_mph : speed = 25 :=
by
  -- Proof of the theorem will be filled here, but for now using sorry
  sorry

end Tod_speed_is_25_mph_l228_228193


namespace remaining_speed_l228_228215

theorem remaining_speed (D : ℝ) (V : ℝ) 
  (h1 : 0.35 * D / 35 + 0.65 * D / V = D / 50) : V = 32.5 :=
by sorry

end remaining_speed_l228_228215


namespace multiplicative_inverse_185_mod_341_l228_228089

theorem multiplicative_inverse_185_mod_341 :
  ∃ (b: ℕ), b ≡ 74466 [MOD 341] ∧ 185 * b ≡ 1 [MOD 341] :=
sorry

end multiplicative_inverse_185_mod_341_l228_228089


namespace password_probability_l228_228655

def is_prime_single_digit : Fin 10 → Prop
| 2 | 3 | 5 | 7 => true
| _ => false

def is_vowel : Char → Prop
| 'A' | 'E' | 'I' | 'O' | 'U' => true
| _ => false

def is_positive_even_single_digit : Fin 9 → Prop
| 2 | 4 | 6 | 8 => true
| _ => false

def prime_probability : ℚ := 4 / 10
def vowel_probability : ℚ := 5 / 26
def even_pos_digit_probability : ℚ := 4 / 9

theorem password_probability :
  prime_probability * vowel_probability * even_pos_digit_probability = 8 / 117 := by
  sorry

end password_probability_l228_228655


namespace dara_employment_wait_time_l228_228088

theorem dara_employment_wait_time :
  ∀ (min_age current_jane_age years_later half_age_factor : ℕ), 
  min_age = 25 → 
  current_jane_age = 28 → 
  years_later = 6 → 
  half_age_factor = 2 →
  (min_age - (current_jane_age + years_later) / half_age_factor - years_later) = 14 :=
by
  intros min_age current_jane_age years_later half_age_factor 
  intros h_min_age h_current_jane_age h_years_later h_half_age_factor
  sorry

end dara_employment_wait_time_l228_228088


namespace called_back_students_l228_228277

/-- Given the number of girls, boys, and students who didn't make the cut,
    this theorem proves the number of students who got called back. -/
theorem called_back_students (girls boys didnt_make_the_cut : ℕ)
    (h_girls : girls = 39)
    (h_boys : boys = 4)
    (h_didnt_make_the_cut : didnt_make_the_cut = 17) :
    girls + boys - didnt_make_the_cut = 26 := by
  sorry

end called_back_students_l228_228277


namespace point_on_y_axis_m_value_l228_228835

theorem point_on_y_axis_m_value (m : ℝ) (h : 6 - 2 * m = 0) : m = 3 := by
  sorry

end point_on_y_axis_m_value_l228_228835


namespace parallel_lines_a_value_l228_228014

theorem parallel_lines_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
      (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l228_228014


namespace bryson_new_shoes_l228_228445

-- Define the conditions as variables and constant values
def pairs_of_shoes : ℕ := 2 -- Number of pairs Bryson bought
def shoes_per_pair : ℕ := 2 -- Number of shoes per pair

-- Define the theorem to prove the question == answer
theorem bryson_new_shoes : pairs_of_shoes * shoes_per_pair = 4 :=
by
  sorry -- Proof placeholder

end bryson_new_shoes_l228_228445


namespace octagon_area_inscribed_in_square_l228_228991

noncomputable def side_length_of_square (perimeter : ℝ) : ℝ :=
  perimeter / 4

noncomputable def trisected_segment_length (side_length : ℝ) : ℝ :=
  side_length / 3

noncomputable def area_of_removed_triangle (segment_length : ℝ) : ℝ :=
  (segment_length * segment_length) / 2

noncomputable def total_area_removed_by_triangles (area_of_triangle : ℝ) : ℝ :=
  4 * area_of_triangle

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_of_octagon (area_of_square : ℝ) (total_area_removed : ℝ) : ℝ :=
  area_of_square - total_area_removed

theorem octagon_area_inscribed_in_square (perimeter : ℝ) (H : perimeter = 144) :
  area_of_octagon (area_of_square (side_length_of_square perimeter))
    (total_area_removed_by_triangles (area_of_removed_triangle (trisected_segment_length (side_length_of_square perimeter))))
  = 1008 :=
by
  rw [H]
  -- Intermediate steps would contain calculations for side_length_of_square, trisected_segment_length, area_of_removed_triangle, total_area_removed_by_triangles, and area_of_square based on the given perimeter.
  sorry

end octagon_area_inscribed_in_square_l228_228991


namespace ratio_aerobics_to_weight_training_l228_228951

def time_spent_exercising : ℕ := 250
def time_spent_aerobics : ℕ := 150
def time_spent_weight_training : ℕ := 100

theorem ratio_aerobics_to_weight_training :
    (time_spent_aerobics / gcd time_spent_aerobics time_spent_weight_training) = 3 ∧
    (time_spent_weight_training / gcd time_spent_aerobics time_spent_weight_training) = 2 :=
by
    sorry

end ratio_aerobics_to_weight_training_l228_228951


namespace regression_line_is_y_eq_x_plus_1_l228_228963

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

end regression_line_is_y_eq_x_plus_1_l228_228963


namespace city_raised_money_for_charity_l228_228760

-- Definitions based on conditions from part a)
def price_regular_duck : ℝ := 3.0
def price_large_duck : ℝ := 5.0
def number_regular_ducks_sold : ℕ := 221
def number_large_ducks_sold : ℕ := 185

-- Definition to represent the main theorem: Total money raised
noncomputable def total_money_raised : ℝ :=
  price_regular_duck * number_regular_ducks_sold + price_large_duck * number_large_ducks_sold

-- Theorem to prove that the total money raised is $1588.00
theorem city_raised_money_for_charity : total_money_raised = 1588.0 := by
  sorry

end city_raised_money_for_charity_l228_228760


namespace Kyle_fish_count_l228_228197

def Carla_fish := 8
def Total_fish := 36
def Kyle_fish := (Total_fish - Carla_fish) / 2

theorem Kyle_fish_count : Kyle_fish = 14 :=
by
  -- This proof will be provided later
  sorry

end Kyle_fish_count_l228_228197


namespace loan_payment_difference_l228_228604

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + P * r * t

noncomputable def loan1_payment (P : ℝ) (r : ℝ) (n : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  let A1 := compounded_amount P r n t1
  let one_third_payment := A1 / 3
  let remaining := A1 - one_third_payment
  one_third_payment + compounded_amount remaining r n t2

noncomputable def loan2_payment (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  simple_interest_amount P r t

noncomputable def positive_difference (x y : ℝ) : ℝ :=
  if x > y then x - y else y - x

theorem loan_payment_difference: 
  ∀ P : ℝ, ∀ r1 r2 : ℝ, ∀ n : ℝ, ∀ t1 t2 : ℝ,
  P = 12000 → r1 = 0.08 → r2 = 0.09 → n = 12 → t1 = 7 → t2 = 8 →
  positive_difference 
    (loan2_payment P r2 (t1 + t2)) 
    (loan1_payment P r1 n t1 t2) = 2335 := 
by
  intros
  sorry

end loan_payment_difference_l228_228604


namespace matrix_addition_correct_l228_228767

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![![-6, 2], ![7, -10]]
def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, -1], ![7, -5]]

theorem matrix_addition_correct : matrixA + matrixB = matrixC := by
  sorry

end matrix_addition_correct_l228_228767


namespace zoey_finishes_20th_book_on_wednesday_l228_228989

theorem zoey_finishes_20th_book_on_wednesday :
  let days_spent := (20 * 21) / 2
  (days_spent % 7) = 0 → 
  (start_day : ℕ) → start_day = 3 → ((start_day + days_spent) % 7) = 3 :=
by
  sorry

end zoey_finishes_20th_book_on_wednesday_l228_228989


namespace right_triangle_area_l228_228973

theorem right_triangle_area (a b c : ℕ) (habc : a = 3 ∧ b = 4 ∧ c = 5) : 
  (a * a + b * b = c * c) → 
  1 / 2 * (a * b) = 6 :=
by
  sorry

end right_triangle_area_l228_228973


namespace determine_asymptotes_l228_228716

noncomputable def hyperbola_eccentricity_asymptote_relation (a b : ℝ) (e : ℝ) (k : ℝ) :=
  a > 0 ∧ b > 0 ∧ (e = Real.sqrt 2 * |k|) ∧ (k = b / a)

theorem determine_asymptotes (a b : ℝ) (h : hyperbola_eccentricity_asymptote_relation a b (Real.sqrt (a^2 + b^2) / a) (b / a)) :
  true := sorry

end determine_asymptotes_l228_228716


namespace Eithan_savings_account_l228_228717

variable (initial_amount wife_firstson_share firstson_remaining firstson_secondson_share 
          secondson_remaining secondson_thirdson_share thirdson_remaining 
          charity_donation remaining_after_charity tax_rate final_remaining : ℝ)

theorem Eithan_savings_account:
  initial_amount = 5000 →
  wife_firstson_share = initial_amount * (2/5) →
  firstson_remaining = initial_amount - wife_firstson_share →
  firstson_secondson_share = firstson_remaining * (3/10) →
  secondson_remaining = firstson_remaining - firstson_secondson_share →
  thirdson_remaining = secondson_remaining * (1-0.30) →
  charity_donation = 200 →
  remaining_after_charity = thirdson_remaining - charity_donation →
  tax_rate = 0.05 →
  final_remaining = remaining_after_charity * (1 - tax_rate) →
  final_remaining = 927.2 := 
  by
    intros
    sorry

end Eithan_savings_account_l228_228717


namespace find_a_over_b_l228_228999

variable (x y z a b : ℝ)
variable (h₁ : 4 * x - 2 * y + z = a)
variable (h₂ : 6 * y - 12 * x - 3 * z = b)
variable (h₃ : b ≠ 0)

theorem find_a_over_b : a / b = -1 / 3 :=
by
  sorry

end find_a_over_b_l228_228999


namespace linear_combination_solution_l228_228036

theorem linear_combination_solution :
  ∃ a b c : ℚ, 
    a • (⟨1, -2, 3⟩ : ℚ × ℚ × ℚ) + b • (⟨4, 1, -1⟩ : ℚ × ℚ × ℚ) + c • (⟨-3, 2, 1⟩ : ℚ × ℚ × ℚ) = ⟨0, 1, 4⟩ ∧
    a = -491/342 ∧
    b = 233/342 ∧
    c = 49/38 :=
by
  sorry

end linear_combination_solution_l228_228036


namespace find_a_from_conditions_l228_228361

theorem find_a_from_conditions
  (a b : ℤ)
  (h₁ : 2584 * a + 1597 * b = 0)
  (h₂ : 1597 * a + 987 * b = -1) :
  a = 1597 :=
by sorry

end find_a_from_conditions_l228_228361


namespace school_adding_seats_l228_228790

theorem school_adding_seats (row_seats : ℕ) (seat_cost : ℕ) (discount_rate : ℝ) (total_cost : ℕ) (n : ℕ) 
                         (total_seats : ℕ) (discounted_seat_cost : ℕ)
                         (total_groups : ℕ) (rows : ℕ) :
  row_seats = 8 →
  seat_cost = 30 →
  discount_rate = 0.10 →
  total_cost = 1080 →
  discounted_seat_cost = seat_cost * (1 - discount_rate) →
  total_seats = total_cost / discounted_seat_cost →
  total_groups = total_seats / 10 →
  rows = total_seats / row_seats →
  rows = 5 :=
by
  intros hrowseats hseatcost hdiscountrate htotalcost hdiscountedseatcost htotalseats htotalgroups hrows
  sorry

end school_adding_seats_l228_228790


namespace candles_used_l228_228387

theorem candles_used (starting_candles used_candles remaining_candles : ℕ) (h1 : starting_candles = 44) (h2 : remaining_candles = 12) : used_candles = 32 :=
by
  sorry

end candles_used_l228_228387


namespace million_to_scientific_notation_l228_228724

theorem million_to_scientific_notation (population_henan : ℝ) (h : population_henan = 98.83 * 10^6) :
  population_henan = 9.883 * 10^7 :=
by sorry

end million_to_scientific_notation_l228_228724


namespace calc_result_l228_228622

theorem calc_result : 75 * 1313 - 25 * 1313 = 65750 := 
by 
  sorry

end calc_result_l228_228622


namespace quadratic_m_value_l228_228157

theorem quadratic_m_value (m : ℤ) (hm1 : |m| = 2) (hm2 : m ≠ 2) : m = -2 :=
sorry

end quadratic_m_value_l228_228157


namespace part_a_part_b_l228_228551

theorem part_a (x y : ℕ) (h : x^3 + 5 * y = y^3 + 5 * x) : x = y :=
sorry

theorem part_b : ∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (x^3 + 5 * y = y^3 + 5 * x) :=
sorry

end part_a_part_b_l228_228551


namespace minimum_workers_needed_l228_228799

theorem minimum_workers_needed 
  (total_days : ℕ)
  (completed_days : ℕ)
  (initial_workers : ℕ)
  (fraction_completed : ℚ)
  (remaining_fraction : ℚ)
  (remaining_days : ℕ)
  (rate_completed_per_day : ℚ)
  (required_rate_per_day : ℚ)
  (equal_productivity : Prop) 
  : initial_workers = 10 :=
by
  -- Definitions
  let total_days := 40
  let completed_days := 10
  let initial_workers := 10
  let fraction_completed := 1 / 4
  let remaining_fraction := 1 - fraction_completed
  let remaining_days := total_days - completed_days
  let rate_completed_per_day := fraction_completed / completed_days
  let required_rate_per_day := remaining_fraction / remaining_days
  let equal_productivity := true

  -- Sorry is used to skip the proof
  sorry

end minimum_workers_needed_l228_228799


namespace rhinos_horn_segment_area_l228_228487

theorem rhinos_horn_segment_area :
  let full_circle_area (r : ℝ) := π * r^2
  let quarter_circle_area (r : ℝ) := (1 / 4) * full_circle_area r
  let half_circle_area (r : ℝ) := (1 / 2) * full_circle_area r
  let larger_quarter_circle_area := quarter_circle_area 4
  let smaller_half_circle_area := half_circle_area 2
  let rhinos_horn_segment_area := larger_quarter_circle_area - smaller_half_circle_area
  rhinos_horn_segment_area = 2 * π := 
by sorry 

end rhinos_horn_segment_area_l228_228487


namespace balance_scale_weights_part_a_balance_scale_weights_part_b_l228_228056

-- Part (a)
theorem balance_scale_weights_part_a (w : List ℕ) (h : w = List.range (90 + 1) \ List.range 1) :
  ¬ ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

-- Part (b)
theorem balance_scale_weights_part_b (w : List ℕ) (h : w = List.range (99 + 1) \ List.range 1) :
  ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

end balance_scale_weights_part_a_balance_scale_weights_part_b_l228_228056


namespace change_calculation_l228_228061

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple = 4.25) := by
  sorry

end change_calculation_l228_228061


namespace rectangle_area_difference_196_l228_228475

noncomputable def max_min_area_difference (P : ℕ) (A_max A_min : ℕ) : Prop :=
  ( ∃ l w : ℕ, 2 * l + 2 * w = P ∧ A_max = l * w ) ∧
  ( ∃ l' w' : ℕ, 2 * l' + 2 * w' = P ∧ A_min = l' * w' ) ∧
  (A_max - A_min = 196)

theorem rectangle_area_difference_196 : max_min_area_difference 60 225 29 :=
by
  sorry

end rectangle_area_difference_196_l228_228475


namespace pencils_ratio_l228_228574

theorem pencils_ratio 
  (cindi_pencils : ℕ := 60)
  (marcia_mul_cindi : ℕ := 2)
  (total_pencils : ℕ := 480)
  (marcia_pencils : ℕ := marcia_mul_cindi * cindi_pencils) 
  (donna_pencils : ℕ := total_pencils - marcia_pencils) :
  donna_pencils / marcia_pencils = 3 := by
  sorry

end pencils_ratio_l228_228574


namespace greatest_possible_selling_price_l228_228594

variable (products : ℕ)
variable (average_price : ℝ)
variable (min_price : ℝ)
variable (less_than_1000_products : ℕ)

theorem greatest_possible_selling_price
  (h1 : products = 20)
  (h2 : average_price = 1200)
  (h3 : min_price = 400)
  (h4 : less_than_1000_products = 10) :
  ∃ max_price, max_price = 11000 := 
by
  sorry

end greatest_possible_selling_price_l228_228594


namespace sqrt_meaningful_range_l228_228315

theorem sqrt_meaningful_range {x : ℝ} (h : x - 1 ≥ 0) : x ≥ 1 :=
sorry

end sqrt_meaningful_range_l228_228315


namespace cos_B_in_triangle_l228_228796

theorem cos_B_in_triangle (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = Real.pi) : 
  Real.cos B = 1 / 2 :=
sorry

end cos_B_in_triangle_l228_228796


namespace cookies_per_tray_l228_228749

def num_trays : ℕ := 4
def num_packs : ℕ := 8
def cookies_per_pack : ℕ := 12
def total_cookies : ℕ := num_packs * cookies_per_pack

theorem cookies_per_tray : total_cookies / num_trays = 24 := by
  sorry

end cookies_per_tray_l228_228749


namespace probability_at_most_one_red_light_l228_228355

def probability_of_no_red_light (p : ℚ) (n : ℕ) : ℚ := (1 - p) ^ n

def probability_of_exactly_one_red_light (p : ℚ) (n : ℕ) : ℚ :=
  (n.choose 1) * p ^ 1 * (1 - p) ^ (n - 1)

theorem probability_at_most_one_red_light (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 4) :
  probability_of_no_red_light p n + probability_of_exactly_one_red_light p n = 16 / 27 :=
by
  rw [h.1, h.2]
  sorry

end probability_at_most_one_red_light_l228_228355


namespace non_degenerate_ellipse_condition_l228_228497

theorem non_degenerate_ellipse_condition (x y k a : ℝ) :
  (3 * x^2 + 9 * y^2 - 12 * x + 27 * y = k) ∧
  (∃ h : ℝ, 3 * (x - h)^2 + 9 * (y + 3/2)^2 = k + 129/4) ∧
  (k > a) ↔ (a = -129 / 4) :=
by
  sorry

end non_degenerate_ellipse_condition_l228_228497


namespace stephanie_running_time_l228_228927

theorem stephanie_running_time
  (Speed : ℝ) (Distance : ℝ) (Time : ℝ)
  (h1 : Speed = 5)
  (h2 : Distance = 15)
  (h3 : Time = Distance / Speed) :
  Time = 3 :=
sorry

end stephanie_running_time_l228_228927


namespace calculate_3_to_5_mul_7_to_5_l228_228458

theorem calculate_3_to_5_mul_7_to_5 : 3^5 * 7^5 = 4084101 :=
by {
  -- Sorry is added to skip the proof; assuming the proof is done following standard arithmetic calculations
  sorry
}

end calculate_3_to_5_mul_7_to_5_l228_228458


namespace compute_P_part_l228_228782

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l228_228782


namespace find_ABC_l228_228381

theorem find_ABC (A B C : ℝ) (h : ∀ n : ℕ, n > 0 → 2 * n^3 + 3 * n^2 = A * (n * (n - 1) * (n - 2)) / 6 + B * (n * (n - 1)) / 2 + C * n) :
  A = 12 ∧ B = 18 ∧ C = 5 :=
by {
  sorry
}

end find_ABC_l228_228381


namespace parallel_vectors_sufficiency_l228_228273

noncomputable def parallel_vectors_sufficiency_problem (a b : ℝ × ℝ) (x : ℝ) : Prop :=
a = (1, x) ∧ b = (x, 4) →
(x = 2 → ∃ k : ℝ, k • a = b) ∧ (∃ k : ℝ, k • a = b → x = 2 ∨ x = -2)

theorem parallel_vectors_sufficiency (x : ℝ) :
  parallel_vectors_sufficiency_problem (1, x) (x, 4) x :=
sorry

end parallel_vectors_sufficiency_l228_228273


namespace ratio_of_jars_to_pots_l228_228820

theorem ratio_of_jars_to_pots 
  (jars : ℕ)
  (pots : ℕ)
  (k : ℕ)
  (marbles_total : ℕ)
  (h1 : jars = 16)
  (h2 : jars = k * pots)
  (h3 : ∀ j, j = 5)
  (h4 : ∀ p, p = 15)
  (h5 : marbles_total = 200) :
  (jars / pots = 2) :=
by
  sorry

end ratio_of_jars_to_pots_l228_228820


namespace quadruple_nested_function_l228_228642

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem quadruple_nested_function (k : ℕ) (h : k = 1) : a (a (a (a (k)))) = 458329 :=
by
  rw [h]
  sorry

end quadruple_nested_function_l228_228642


namespace tan_y_l228_228196

theorem tan_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hy : 0 < y ∧ y < π / 2)
  (hsiny : Real.sin y = 2 * a * b / (a^2 + b^2)) :
  Real.tan y = 2 * a * b / (a^2 - b^2) :=
sorry

end tan_y_l228_228196


namespace square_area_rational_l228_228686

-- Define the condition: the side length of the square is a rational number.
def is_rational (x : ℚ) : Prop := true

-- Define the theorem to be proved: If the side length of a square is rational, then its area is rational.
theorem square_area_rational (s : ℚ) (h : is_rational s) : is_rational (s * s) := 
sorry

end square_area_rational_l228_228686


namespace cube_volume_given_surface_area_l228_228391

theorem cube_volume_given_surface_area (SA : ℝ) (a V : ℝ) (h : SA = 864) (h1 : 6 * a^2 = SA) (h2 : V = a^3) : 
  V = 1728 := 
by 
  sorry

end cube_volume_given_surface_area_l228_228391


namespace number_of_m_values_l228_228324

theorem number_of_m_values (m : ℕ) (h1 : 4 * m > 11) (h2 : m < 12) : 
  11 - 3 + 1 = 9 := 
sorry

end number_of_m_values_l228_228324


namespace cell_phone_height_l228_228726

theorem cell_phone_height (width perimeter : ℕ) (h1 : width = 9) (h2 : perimeter = 46) : 
  ∃ length : ℕ, length = 14 ∧ perimeter = 2 * (width + length) :=
by
  sorry

end cell_phone_height_l228_228726


namespace negation_proof_l228_228359

theorem negation_proof :
  (¬ ∀ x : ℝ, x < 0 → 1 - x > Real.exp x) ↔ (∃ x_0 : ℝ, x_0 < 0 ∧ 1 - x_0 ≤ Real.exp x_0) :=
by
  sorry

end negation_proof_l228_228359


namespace smallest_unrepresentable_integer_l228_228205

theorem smallest_unrepresentable_integer :
  ∃ n : ℕ, (∀ a b c d : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → 
  n ≠ (2^a - 2^b) / (2^c - 2^d)) ∧ n = 11 :=
by
  sorry

end smallest_unrepresentable_integer_l228_228205


namespace paint_price_max_boxes_paint_A_l228_228031

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l228_228031


namespace equilateral_triangle_of_angle_and_side_sequences_l228_228916

variable {A B C a b c : ℝ}

theorem equilateral_triangle_of_angle_and_side_sequences
  (H_angles_arithmetic : 2 * B = A + C)
  (H_sum_angles : A + B + C = Real.pi)
  (H_sides_geometric : b^2 = a * c) :
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3 ∧ a = b ∧ b = c :=
by
  sorry

end equilateral_triangle_of_angle_and_side_sequences_l228_228916


namespace binom_18_6_eq_13260_l228_228159

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l228_228159


namespace rower_trip_time_to_Big_Rock_l228_228172

noncomputable def row_trip_time (rowing_speed_in_still_water : ℝ) (river_speed : ℝ) (distance_to_destination : ℝ) : ℝ :=
  let speed_upstream := rowing_speed_in_still_water - river_speed
  let speed_downstream := rowing_speed_in_still_water + river_speed
  let time_upstream := distance_to_destination / speed_upstream
  let time_downstream := distance_to_destination / speed_downstream
  time_upstream + time_downstream

theorem rower_trip_time_to_Big_Rock :
  row_trip_time 7 2 3.2142857142857144 = 1 :=
by
  sorry

end rower_trip_time_to_Big_Rock_l228_228172


namespace value_of_playstation_l228_228004

theorem value_of_playstation (V : ℝ) (H1 : 700 + 200 = 900) (H2 : V - 0.2 * V = 0.8 * V) (H3 : 0.8 * V = 900 - 580) : V = 400 :=
by
  sorry

end value_of_playstation_l228_228004


namespace tan_sum_formula_eq_l228_228058

theorem tan_sum_formula_eq {θ : ℝ} (h1 : ∃θ, θ ∈ Set.Ico 0 (2 * Real.pi) 
  ∧ ∃P, P = (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4)) 
  ∧ θ = (3 * Real.pi / 4)) : 
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := 
sorry

end tan_sum_formula_eq_l228_228058


namespace tangency_point_l228_228516

def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 18
def parabola2 (y : ℝ) : ℝ := y^2 + 60 * y + 910

theorem tangency_point (x y : ℝ) (h1 : y = parabola1 x) (h2 : x = parabola2 y) :
  x = -9 / 2 ∧ y = -59 / 2 :=
by
  sorry

end tangency_point_l228_228516


namespace percentage_error_l228_228931

theorem percentage_error (e : ℝ) : (1 + e / 100)^2 = 1.1025 → e = 5.125 := 
by sorry

end percentage_error_l228_228931


namespace domain_range_of_g_l228_228255

variable (f : ℝ → ℝ)
variable (dom_f : Set.Icc 1 3)
variable (rng_f : Set.Icc 0 1)
variable (g : ℝ → ℝ)
variable (g_eq : ∀ x, g x = 2 - f (x - 1))

theorem domain_range_of_g :
  (Set.Icc 2 4) = { x | ∃ y, x = y ∧ g y = (g y) } ∧ Set.Icc 1 2 = { z | ∃ w, z = g w} :=
  sorry

end domain_range_of_g_l228_228255


namespace polynomial_sum_l228_228568

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_sum : ∃ a b c d : ℝ, 
  (g a b c d (-3 * Complex.I) = 0) ∧
  (g a b c d (1 + Complex.I) = 0) ∧
  (g a b c d (3 * Complex.I) = 0) ∧
  (g a b c d (1 - Complex.I) = 0) ∧ 
  (a + b + c + d = 9) := by
  sorry

end polynomial_sum_l228_228568


namespace sum_of_x_and_y_l228_228498

theorem sum_of_x_and_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
(hx15 : x < 15) (hy15 : y < 15) (h : x + y + x * y = 119) : x + y = 21 ∨ x + y = 20 := 
by
  sorry

end sum_of_x_and_y_l228_228498


namespace apples_more_than_grapes_l228_228013

theorem apples_more_than_grapes 
  (total_weight : ℕ) (weight_ratio_apples : ℕ) (weight_ratio_peaches : ℕ) (weight_ratio_grapes : ℕ) : 
  weight_ratio_apples = 12 → 
  weight_ratio_peaches = 8 → 
  weight_ratio_grapes = 7 → 
  total_weight = 54 →
  ((12 * total_weight / (12 + 8 + 7)) - (7 * total_weight / (12 + 8 + 7))) = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end apples_more_than_grapes_l228_228013


namespace werewolf_knight_is_A_l228_228810

structure Person :=
  (isKnight : Prop)
  (isLiar : Prop)
  (isWerewolf : Prop)

variables (A B C : Person)

-- A's statement: "At least one of us is a liar."
def statementA (A B C : Person) : Prop := A.isLiar ∨ B.isLiar ∨ C.isLiar

-- B's statement: "C is a knight."
def statementB (C : Person) : Prop := C.isKnight

theorem werewolf_knight_is_A (A B C : Person) 
  (hA : statementA A B C)
  (hB : statementB C)
  (hWerewolfKnight : ∃ x : Person, x.isWerewolf ∧ x.isKnight ∧ ¬ (A ≠ x ∧ B ≠ x ∧ C ≠ x))
  : A.isWerewolf ∧ A.isKnight :=
sorry

end werewolf_knight_is_A_l228_228810


namespace find_a_b_l228_228301

theorem find_a_b
  (f : ℝ → ℝ) (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_f : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (h_eq : ∀ x, f x - f a = (x - b) * (x - a)^2) :
  a = -2 ∧ b = 1 :=
by
  sorry

end find_a_b_l228_228301


namespace solution_l228_228950

def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^5 + p*x^3 + q*x - 8

theorem solution (p q : ℝ) (h : f (-2) p q = 10) : f 2 p q = -26 := by
  sorry

end solution_l228_228950


namespace find_xyz_values_l228_228868

theorem find_xyz_values (x y z : ℝ) (h₁ : x + y + z = Real.pi) (h₂ : x ≥ 0) (h₃ : y ≥ 0) (h₄ : z ≥ 0) :
    (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = Real.pi) ∨
    (x = Real.pi / 6 ∧ y = Real.pi / 3 ∧ z = Real.pi / 2) :=
sorry

end find_xyz_values_l228_228868


namespace ratio_is_1_to_3_l228_228662

-- Definitions based on the conditions
def washed_on_wednesday : ℕ := 6
def washed_on_thursday : ℕ := 2 * washed_on_wednesday
def washed_on_friday : ℕ := washed_on_thursday / 2
def total_washed : ℕ := 26
def washed_on_saturday : ℕ := total_washed - washed_on_wednesday - washed_on_thursday - washed_on_friday

-- The ratio calculation
def ratio_saturday_to_wednesday : ℚ := washed_on_saturday / washed_on_wednesday

-- The theorem to prove
theorem ratio_is_1_to_3 : ratio_saturday_to_wednesday = 1 / 3 :=
by
  -- Insert proof here
  sorry

end ratio_is_1_to_3_l228_228662


namespace cooking_dishes_time_l228_228039

def total_awake_time : ℝ := 16
def work_time : ℝ := 8
def gym_time : ℝ := 2
def bath_time : ℝ := 0.5
def homework_bedtime_time : ℝ := 1
def packing_lunches_time : ℝ := 0.5
def cleaning_time : ℝ := 0.5
def shower_leisure_time : ℝ := 2
def total_allocated_time : ℝ := work_time + gym_time + bath_time + homework_bedtime_time + packing_lunches_time + cleaning_time + shower_leisure_time

theorem cooking_dishes_time : total_awake_time - total_allocated_time = 1.5 := by
  sorry

end cooking_dishes_time_l228_228039


namespace complex_pow_six_eq_eight_i_l228_228079

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l228_228079


namespace distance_EC_l228_228212

-- Define the points and given distances as conditions
structure Points :=
  (A B C D E : Type)

-- Distances between points
variables {Points : Type}
variables (dAB dBC dCD dDE dEA dEC : ℝ)
variables [Nonempty Points]

-- Specify conditions: distances in kilometers
def distances_given (dAB dBC dCD dDE dEA : ℝ) : Prop :=
  dAB = 30 ∧ dBC = 80 ∧ dCD = 236 ∧ dDE = 86 ∧ dEA = 40

-- Main theorem: prove that the distance from E to C is 63.4 km
theorem distance_EC (h : distances_given 30 80 236 86 40) : dEC = 63.4 :=
sorry

end distance_EC_l228_228212


namespace arithmetic_sequence_a11_l228_228057

theorem arithmetic_sequence_a11 (a : ℕ → ℤ) (h_arithmetic : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_a3 : a 3 = 4) (h_a5 : a 5 = 8) : a 11 = 12 :=
by
  sorry

end arithmetic_sequence_a11_l228_228057


namespace factorize_expression_l228_228065

variable (a b c : ℝ)

theorem factorize_expression : 
  (a - 2 * b) * (a - 2 * b - 4) + 4 - c ^ 2 = ((a - 2 * b) - 2 + c) * ((a - 2 * b) - 2 - c) := 
by
  sorry

end factorize_expression_l228_228065


namespace rectangular_eq_of_C_slope_of_l_l228_228985

noncomputable section

/-- Parametric equations for curve C -/
def parametric_eq (θ : ℝ) : ℝ × ℝ :=
⟨4 * Real.cos θ, 3 * Real.sin θ⟩

/-- Question 1: Prove that the rectangular coordinate equation of curve C is (x^2)/16 + (y^2)/9 = 1. -/
theorem rectangular_eq_of_C (x y θ : ℝ) (h₁ : x = 4 * Real.cos θ) (h₂ : y = 3 * Real.sin θ) : 
  x^2 / 16 + y^2 / 9 = 1 := 
sorry

/-- Line passing through point M(2, 2) with parametric equations -/
def line_through_M (t α : ℝ) : ℝ × ℝ :=
⟨2 + t * Real.cos α, 2 + t * Real.sin α⟩ 

/-- Question 2: Prove that the slope of line l passing M(2, 2) which intersects curve C at points A and B is -9/16 -/
theorem slope_of_l (t₁ t₂ α : ℝ) (t₁_t₂_sum_zero : (9 * Real.sin α + 36 * Real.cos α) = 0) :
  Real.tan α = -9 / 16 :=
sorry

end rectangular_eq_of_C_slope_of_l_l228_228985


namespace original_price_of_shoes_l228_228480

theorem original_price_of_shoes (x : ℝ) (h : 1/4 * x = 18) : x = 72 := by
  sorry

end original_price_of_shoes_l228_228480


namespace minimum_m_l228_228393

theorem minimum_m (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 24 * m = n ^ 4) : m ≥ 54 :=
sorry

end minimum_m_l228_228393


namespace largest_k_for_positive_root_l228_228237

theorem largest_k_for_positive_root : ∃ k : ℤ, k = 1 ∧ ∀ k' : ℤ, (k' > 1) → ¬ (∃ x > 0, 3 * x * (2 * k' * x - 5) - 2 * x^2 + 8 = 0) :=
by
  sorry

end largest_k_for_positive_root_l228_228237


namespace fraction_equality_l228_228545

theorem fraction_equality : (18 / (5 * 107 + 3) = 18 / 538) := 
by
  -- Proof skipped
  sorry

end fraction_equality_l228_228545


namespace density_of_second_part_l228_228455

theorem density_of_second_part (V m : ℝ) (h1 : ∀ V m : ℝ, V_1 = 0.3 * V) 
  (h2 : ∀ V m : ℝ, m_1 = 0.6 * m) 
  (rho1 : ρ₁ = 7800) : 
  ∃ ρ₂, ρ₂ = 2229 :=
by sorry

end density_of_second_part_l228_228455


namespace age_ratio_l228_228919

theorem age_ratio (R D : ℕ) (hR : R + 4 = 32) (hD : D = 21) : R / D = 4 / 3 := 
by sorry

end age_ratio_l228_228919


namespace total_houses_is_160_l228_228117

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l228_228117


namespace forest_coverage_2009_min_annual_growth_rate_l228_228804

variables (a : ℝ)

-- Conditions
def initially_forest_coverage (a : ℝ) := a
def annual_natural_growth_rate := 0.02

-- Questions reformulated:
-- Part 1: Prove the forest coverage at the end of 2009
theorem forest_coverage_2009 : (∃ a : ℝ, (y : ℝ) = a * (1 + 0.02)^5 ∧ y = 1.104 * a) :=
by sorry

-- Part 2: Prove the minimum annual average growth rate by 2014
theorem min_annual_growth_rate : (∀ p : ℝ, (a : ℝ) * (1 + p)^10 ≥ 2 * a → p ≥ 0.072) :=
by sorry

end forest_coverage_2009_min_annual_growth_rate_l228_228804


namespace second_polygon_sides_l228_228509

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l228_228509


namespace total_runs_opponents_correct_l228_228644

-- Define the scoring conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def lost_games_scores : List ℕ := [3, 5, 7, 9, 11, 13]
def won_games_scores : List ℕ := [2, 4, 6, 8, 10, 12]

-- Define the total runs scored by opponents in lost games
def total_runs_lost_games : ℕ := (lost_games_scores.map (λ x => x + 1)).sum

-- Define the total runs scored by opponents in won games
def total_runs_won_games : ℕ := (won_games_scores.map (λ x => x / 2)).sum

-- Total runs scored by opponents (given)
def total_runs_opponents : ℕ := total_runs_lost_games + total_runs_won_games

-- The theorem to prove
theorem total_runs_opponents_correct : total_runs_opponents = 75 := by
  -- Proof goes here
  sorry

end total_runs_opponents_correct_l228_228644


namespace initial_money_proof_l228_228510

-- Definition: Dan's initial money, the money spent, and the money left.
def initial_money : ℝ := sorry
def spent_money : ℝ := 1.0
def left_money : ℝ := 2.0

-- Theorem: Prove that Dan's initial money is the sum of the money spent and the money left.
theorem initial_money_proof : initial_money = spent_money + left_money :=
sorry

end initial_money_proof_l228_228510


namespace expand_polynomial_l228_228483

variable {x y z : ℝ}

theorem expand_polynomial : (x + 10 * z + 5) * (2 * y + 15) = 2 * x * y + 20 * y * z + 15 * x + 10 * y + 150 * z + 75 :=
  sorry

end expand_polynomial_l228_228483


namespace value_of_expression_l228_228390

theorem value_of_expression (x : ℝ) (h : x^2 + x + 1 = 8) : 4 * x^2 + 4 * x + 9 = 37 :=
by
  sorry

end value_of_expression_l228_228390


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l228_228436

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l228_228436


namespace seven_fifths_of_fraction_l228_228840

theorem seven_fifths_of_fraction :
  (7 / 5) * (-18 / 4) = -63 / 10 :=
by
  sorry

end seven_fifths_of_fraction_l228_228840


namespace compute_pow_l228_228174

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l228_228174


namespace chapters_per_day_l228_228876

theorem chapters_per_day (chapters : ℕ) (total_days : ℕ) : ℝ :=
  let chapters := 2
  let total_days := 664
  chapters / total_days

example : chapters_per_day 2 664 = 2 / 664 := by sorry

end chapters_per_day_l228_228876


namespace cans_in_third_bin_l228_228109

noncomputable def num_cans_in_bin (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | 4 => 11
  | 5 => 16
  | _ => sorry

theorem cans_in_third_bin :
  num_cans_in_bin 3 = 7 :=
sorry

end cans_in_third_bin_l228_228109


namespace slope_of_parallel_line_l228_228917

theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) : 
  (5 * x - 3 * y = 12) → m = 5 / 3 → (∃ b : ℝ, y = (5 / 3) * x + b) :=
by
  intro h_eqn h_slope
  use -4 / 3
  sorry

end slope_of_parallel_line_l228_228917


namespace lines_intersect_at_single_point_l228_228347

def line1 (a b x y: ℝ) := a * x + 2 * b * y + 3 * (a + b + 1) = 0
def line2 (a b x y: ℝ) := b * x + 2 * (a + b + 1) * y + 3 * a = 0
def line3 (a b x y: ℝ) := (a + b + 1) * x + 2 * a * y + 3 * b = 0

theorem lines_intersect_at_single_point (a b : ℝ) :
  (∃ x y : ℝ, line1 a b x y ∧ line2 a b x y ∧ line3 a b x y) ↔ a + b = -1/2 :=
by
  sorry

end lines_intersect_at_single_point_l228_228347


namespace new_percentage_of_water_l228_228112

noncomputable def initial_weight : ℝ := 100
noncomputable def initial_percentage_water : ℝ := 99 / 100
noncomputable def initial_weight_water : ℝ := initial_weight * initial_percentage_water
noncomputable def initial_weight_non_water : ℝ := initial_weight - initial_weight_water
noncomputable def new_weight : ℝ := 25

theorem new_percentage_of_water :
  ((new_weight - initial_weight_non_water) / new_weight) * 100 = 96 :=
by
  sorry

end new_percentage_of_water_l228_228112


namespace orthocentric_tetrahedron_edge_tangent_iff_l228_228166

structure Tetrahedron :=
(V : Type*)
(a b c d e f : V)
(is_orthocentric : Prop)
(has_edge_tangent_sphere : Prop)
(face_equilateral : Prop)
(edges_converging_equal : Prop)

variable (T : Tetrahedron)

noncomputable def edge_tangent_iff_equilateral_edges_converging_equal : Prop :=
T.has_edge_tangent_sphere ↔ (T.face_equilateral ∧ T.edges_converging_equal)

-- Now create the theorem statement
theorem orthocentric_tetrahedron_edge_tangent_iff :
  T.is_orthocentric →
  (∀ a d b e c f p r : ℝ, 
    a + d = b + e ∧ b + e = c + f ∧ a^2 + d^2 = b^2 + e^2 ∧ b^2 + e^2 = c^2 + f^2 ) → 
    edge_tangent_iff_equilateral_edges_converging_equal T := 
by
  intros
  unfold edge_tangent_iff_equilateral_edges_converging_equal
  sorry

end orthocentric_tetrahedron_edge_tangent_iff_l228_228166


namespace carol_total_points_l228_228438

-- Define the conditions for Carol's game points.
def first_round_points := 17
def second_round_points := 6
def last_round_points := -16

-- Prove that the total points at the end of the game are 7.
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l228_228438


namespace num_real_a_satisfy_union_l228_228867

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_l228_228867


namespace volume_of_rectangular_prism_l228_228469

theorem volume_of_rectangular_prism
  (l w h : ℝ)
  (Hlw : l * w = 10)
  (Hwh : w * h = 15)
  (Hlh : l * h = 6) : l * w * h = 30 := 
by
  sorry

end volume_of_rectangular_prism_l228_228469


namespace equal_divide_remaining_amount_all_girls_l228_228148

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l228_228148


namespace stacy_days_to_complete_paper_l228_228396

def total_pages : ℕ := 66
def pages_per_day : ℕ := 11

theorem stacy_days_to_complete_paper :
  total_pages / pages_per_day = 6 := by
  sorry

end stacy_days_to_complete_paper_l228_228396


namespace geometric_sequence_common_ratio_l228_228754

theorem geometric_sequence_common_ratio (a_1 q : ℝ) (hne1 : q ≠ 1)
  (h : (a_1 * (1 - q^4) / (1 - q)) = 5 * (a_1 * (1 - q^2) / (1 - q))) :
  q = -1 ∨ q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l228_228754


namespace gcd_gx_x_is_210_l228_228887

-- Define the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

-- The main proof problem
theorem gcd_gx_x_is_210 (x : ℕ) (hx : is_multiple_of 17280 x) :
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)) x = 210 :=
by
  sorry

end gcd_gx_x_is_210_l228_228887


namespace no_positive_a_for_inequality_l228_228943

theorem no_positive_a_for_inequality (a : ℝ) (h : 0 < a) : 
  ¬ ∀ x : ℝ, |Real.cos x| + |Real.cos (a * x)| > Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_for_inequality_l228_228943


namespace correct_answers_count_l228_228091

theorem correct_answers_count
  (c w : ℕ)
  (h1 : 4 * c - 2 * w = 420)
  (h2 : c + w = 150) : 
  c = 120 :=
sorry

end correct_answers_count_l228_228091


namespace trapezoid_ratio_l228_228706

theorem trapezoid_ratio (A B C D M N K : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup K]
  (CM MD CN NA AD BC : ℝ)
  (h1 : CM / MD = 4 / 3)
  (h2 : CN / NA = 4 / 3) 
  : AD / BC = 7 / 12 :=
by
  sorry

end trapezoid_ratio_l228_228706


namespace seventy_seventh_digit_is_three_l228_228427

-- Define the sequence of digits from the numbers 60 to 1 in decreasing order.
def sequence_of_digits : List Nat :=
  (List.range' 1 60).reverse.bind (fun n => n.digits 10)

-- Define a function to get the nth digit from the list.
def digit_at_position (n : Nat) : Option Nat :=
  sequence_of_digits.get? (n - 1)

-- The statement to prove
theorem seventy_seventh_digit_is_three : digit_at_position 77 = some 3 :=
sorry

end seventy_seventh_digit_is_three_l228_228427


namespace range_of_a_l228_228209

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) :
  (0 < a ∧ 4 - 4 * a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l228_228209


namespace four_digit_integer_l228_228502

theorem four_digit_integer (a b c d : ℕ) 
(h1: a + b + c + d = 14) (h2: b + c = 9) (h3: a - d = 1)
(h4: (a - b + c - d) % 11 = 0) : 1000 * a + 100 * b + 10 * c + d = 3542 :=
by
  sorry

end four_digit_integer_l228_228502


namespace arithmetic_sequence_property_l228_228446

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_property
  (a : ℕ → α) (h1 : a 1 + a 8 = 9) (h4 : a 4 = 3) : a 5 = 6 :=
by
  sorry

end arithmetic_sequence_property_l228_228446


namespace no_distinct_nat_numbers_eq_l228_228001

theorem no_distinct_nat_numbers_eq (x y z t : ℕ) (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t) 
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) : x ^ x + y ^ y ≠ z ^ z + t ^ t := 
by 
  sorry

end no_distinct_nat_numbers_eq_l228_228001


namespace base_329_digits_even_l228_228408

noncomputable def base_of_four_digit_even_final : ℕ := 5

theorem base_329_digits_even (b : ℕ) (h1 : b^3 ≤ 329) (h2 : 329 < b^4)
  (h3 : ∀ d, 329 % b = d → d % 2 = 0) : b = base_of_four_digit_even_final :=
by sorry

end base_329_digits_even_l228_228408


namespace smallest_k_for_divisibility_l228_228766

theorem smallest_k_for_divisibility : (∃ k : ℕ, ∀ z : ℂ, z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^k - 1 ∧ (∀ m : ℕ, m < k → ∃ z : ℂ, ¬(z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^m - 1))) ↔ k = 14 := sorry

end smallest_k_for_divisibility_l228_228766


namespace max_choir_members_l228_228323

theorem max_choir_members : 
  ∃ (m : ℕ), 
    (∃ k : ℕ, m = k^2 + 11) ∧ 
    (∃ n : ℕ, m = n * (n + 5)) ∧ 
    (∀ m' : ℕ, 
      ((∃ k' : ℕ, m' = k' * k' + 11) ∧ 
       (∃ n' : ℕ, m' = n' * (n' + 5))) → 
      m' ≤ 266) ∧ 
    m = 266 :=
by sorry

end max_choir_members_l228_228323


namespace find_radius_of_third_circle_l228_228053

noncomputable def radius_of_third_circle_equals_shaded_region (r1 r2 r3 : ℝ) : Prop :=
  let area_large := Real.pi * (r2 ^ 2)
  let area_small := Real.pi * (r1 ^ 2)
  let area_shaded := area_large - area_small
  let area_third_circle := Real.pi * (r3 ^ 2)
  area_shaded = area_third_circle

theorem find_radius_of_third_circle (r1 r2 : ℝ) (r1_eq : r1 = 17) (r2_eq : r2 = 27) : ∃ r3 : ℝ, r3 = 10 * Real.sqrt 11 ∧ radius_of_third_circle_equals_shaded_region r1 r2 r3 := 
by
  sorry

end find_radius_of_third_circle_l228_228053


namespace functional_equation_solution_l228_228407

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x + f y) + f (y + f x) = 2 * f (x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro f h
  sorry

end functional_equation_solution_l228_228407


namespace gcd_282_470_l228_228824

theorem gcd_282_470 : Int.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l228_228824


namespace most_probable_standard_parts_in_batch_l228_228763

theorem most_probable_standard_parts_in_batch :
  let q := 0.075
  let p := 1 - q
  let n := 39
  ∃ k₀ : ℤ, 36 ≤ k₀ ∧ k₀ ≤ 37 := 
by
  sorry

end most_probable_standard_parts_in_batch_l228_228763


namespace remainder_when_divided_by_8_l228_228344

theorem remainder_when_divided_by_8 (k : ℤ) : ((63 * k + 25) % 8) = 1 := 
by sorry

end remainder_when_divided_by_8_l228_228344


namespace like_terms_set_l228_228942

theorem like_terms_set (a b : ℕ) (x y : ℝ) : 
  (¬ (a = b)) ∧
  ((-2 * x^3 * y^3 = y^3 * x^3)) ∧ 
  (¬ (1 * x * y = 2 * x * y^3)) ∧ 
  (¬ (-6 = x)) :=
by
  sorry

end like_terms_set_l228_228942


namespace prob_teamB_wins_first_game_l228_228136
-- Import the necessary library

-- Define the conditions and the question in a Lean theorem statement
theorem prob_teamB_wins_first_game :
  (∀ (win_A win_B : ℕ), win_A < 4 ∧ win_B = 4) →
  (∀ (team_wins_game : ℕ → Prop), (team_wins_game 2 = false) ∧ (team_wins_game 3 = true)) →
  (∀ (team_wins_series : Prop), team_wins_series = (win_B ≥ 4 ∧ win_A < 4)) →
  (∀ (game_outcome_distribution : ℕ → ℕ → ℕ → ℕ → ℚ), game_outcome_distribution 4 4 2 2 = 1 / 2) →
  (∀ (first_game_outcome : Prop), first_game_outcome = true) →
  true :=
sorry

end prob_teamB_wins_first_game_l228_228136


namespace quadratic_inequality_solution_l228_228544

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 10 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 5} :=
by
  sorry

end quadratic_inequality_solution_l228_228544


namespace pens_distribution_l228_228511

theorem pens_distribution (friends : ℕ) (pens : ℕ) (at_least_one : ℕ) 
  (h1 : friends = 4) (h2 : pens = 10) (h3 : at_least_one = 1) 
  (h4 : ∀ f : ℕ, f < friends → at_least_one ≤ f) :
  ∃ ways : ℕ, ways = 142 := 
sorry

end pens_distribution_l228_228511


namespace combined_population_port_perry_lazy_harbor_l228_228186

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l228_228186


namespace sin_240_eq_neg_sqrt3_div_2_l228_228345

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l228_228345


namespace intersection_of_A_and_B_l228_228751

-- Definitions based on conditions
def A : Set ℝ := { x | x + 2 = 0 }
def B : Set ℝ := { x | x^2 - 4 = 0 }

-- Theorem statement proving the question == answer given conditions
theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by 
  sorry

end intersection_of_A_and_B_l228_228751


namespace theo_needs_84_eggs_l228_228836

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end theo_needs_84_eggs_l228_228836


namespace increase_75_by_150_percent_l228_228163

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l228_228163


namespace f_8_plus_f_9_l228_228801

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_even_transformed : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 1

theorem f_8_plus_f_9 : f 8 + f 9 = 1 :=
sorry

end f_8_plus_f_9_l228_228801


namespace fraction_subtraction_l228_228598

theorem fraction_subtraction (x : ℝ) : (8000 * x - (0.05 / 100 * 8000) = 796) → x = 0.1 :=
by
  sorry

end fraction_subtraction_l228_228598


namespace prove_Φ_eq_8_l228_228358

-- Define the structure of the problem.
def condition (Φ : ℕ) : Prop := 504 / Φ = 40 + 3 * Φ

-- Define the main proof question.
theorem prove_Φ_eq_8 (Φ : ℕ) (h : condition Φ) : Φ = 8 := 
sorry

end prove_Φ_eq_8_l228_228358


namespace equation_no_solution_at_5_l228_228134

theorem equation_no_solution_at_5 :
  ∀ (some_expr : ℝ), ¬(1 / (5 + 5) + some_expr = 1 / (5 - 5)) :=
by
  intro some_expr
  sorry

end equation_no_solution_at_5_l228_228134


namespace third_number_in_first_set_l228_228038

theorem third_number_in_first_set (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end third_number_in_first_set_l228_228038


namespace solve_for_x_and_calculate_l228_228928

theorem solve_for_x_and_calculate (x y : ℚ) 
  (h1 : 102 * x - 5 * y = 25) 
  (h2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 :=
by 
  -- These proof steps would solve the problem and validate the theorem
  sorry

end solve_for_x_and_calculate_l228_228928


namespace cost_of_pet_snake_l228_228380

theorem cost_of_pet_snake (original_amount : ℕ) (amount_left : ℕ) (cost : ℕ) 
  (h1 : original_amount = 73) (h2 : amount_left = 18) : cost = 55 :=
by
  sorry

end cost_of_pet_snake_l228_228380


namespace Mitya_age_l228_228080

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l228_228080


namespace behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l228_228424

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 5 * x ^ 2 + 4

theorem behavior_of_g_as_x_approaches_infinity_and_negative_infinity :
  (∀ ε > 0, ∃ M > 0, ∀ x > M, g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x < -N, g x > ε) :=
by
  sorry

end behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l228_228424


namespace geometric_seq_sum_four_and_five_l228_228207

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) := a₁ * q^(n-1)

theorem geometric_seq_sum_four_and_five :
  (∀ n, geom_seq a₁ q n > 0) →
  geom_seq a₁ q 3 = 4 →
  geom_seq a₁ q 6 = 1 / 2 →
  geom_seq a₁ q 4 + geom_seq a₁ q 5 = 3 :=
by
  sorry

end geometric_seq_sum_four_and_five_l228_228207


namespace average_ABC_is_three_l228_228648
-- Import the entirety of the Mathlib library

-- Define the required conditions and the theorem to be proved
theorem average_ABC_is_three (A B C : ℝ) 
    (h1 : 2012 * C - 4024 * A = 8048) 
    (h2 : 2012 * B + 6036 * A = 10010) : 
    (A + B + C) / 3 = 3 := 
by
  sorry

end average_ABC_is_three_l228_228648


namespace gifts_receiving_ribbon_l228_228798

def total_ribbon := 18
def ribbon_per_gift := 2
def remaining_ribbon := 6

theorem gifts_receiving_ribbon : (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 := by
  sorry

end gifts_receiving_ribbon_l228_228798


namespace solve_equation_l228_228435

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l228_228435


namespace B_investment_l228_228227

theorem B_investment (A : ℝ) (t_B : ℝ) (profit_ratio : ℝ) (B_investment_result : ℝ) : 
  A = 27000 → t_B = 4.5 → profit_ratio = 2 → B_investment_result = 36000 :=
by
  intro hA htB hpR
  sorry

end B_investment_l228_228227


namespace books_written_l228_228181

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l228_228181


namespace find_x_satisfying_inequality_l228_228609

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end find_x_satisfying_inequality_l228_228609


namespace find_f3_l228_228526

noncomputable def f : ℝ → ℝ := sorry

theorem find_f3 (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : f 3 = -11 :=
sorry

end find_f3_l228_228526


namespace price_reduction_for_desired_profit_l228_228044

def profit_per_piece (x : ℝ) : ℝ := 40 - x
def pieces_sold_per_day (x : ℝ) : ℝ := 20 + 2 * x

theorem price_reduction_for_desired_profit (x : ℝ) :
  (profit_per_piece x) * (pieces_sold_per_day x) = 1200 ↔ (x = 10 ∨ x = 20) := by
  sorry

end price_reduction_for_desired_profit_l228_228044


namespace sum_of_squares_due_to_regression_eq_72_l228_228750

theorem sum_of_squares_due_to_regression_eq_72
    (total_squared_deviations : ℝ)
    (correlation_coefficient : ℝ)
    (h1 : total_squared_deviations = 120)
    (h2 : correlation_coefficient = 0.6)
    : total_squared_deviations * correlation_coefficient^2 = 72 :=
by
  -- Proof goes here
  sorry

end sum_of_squares_due_to_regression_eq_72_l228_228750


namespace exp_values_l228_228532

variable {a x y : ℝ}

theorem exp_values (hx : a^x = 3) (hy : a^y = 2) :
  a^(x - y) = 3 / 2 ∧ a^(2 * x + y) = 18 :=
by
  sorry

end exp_values_l228_228532


namespace trigonometric_identity_l228_228493

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) : 
  2 * Real.sin α * Real.cos α - (Real.cos α)^2 = -1 := 
by
  sorry

end trigonometric_identity_l228_228493


namespace minimum_tetrahedra_partition_l228_228180

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end minimum_tetrahedra_partition_l228_228180


namespace fixed_point_of_function_l228_228257

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ x : ℝ, (x = 1 → a^(x-1) = 1) :=
by
  sorry

end fixed_point_of_function_l228_228257


namespace find_ratio_l228_228241

noncomputable def ratio_CN_AN (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) : Prop :=
  CN / AN = 5 / 24

theorem find_ratio (BM MC BK AB CN AN : ℝ) (h1 : BM / MC = 4 / 5) (h2 : BK / AB = 1 / 5) (h3 : BM + MC = BC) (h4 : BK = BK) (h5 : BK + AB = 6 * BK) : 
  ratio_CN_AN BM MC BK AB CN AN h1 h2 :=
by
  sorry

end find_ratio_l228_228241


namespace find_a_l228_228908

noncomputable def f (a x : ℝ) : ℝ := 2^x / (2^x + a * x)

variables (a p q : ℝ)

theorem find_a
  (h1 : f a p = 6 / 5)
  (h2 : f a q = -1 / 5)
  (h3 : 2^(p + q) = 16 * p * q)
  (h4 : a > 0) :
  a = 4 :=
  sorry

end find_a_l228_228908


namespace inequality_solution_set_l228_228787

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0) :
  ∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ b * x^2 - 5 * x + a > 0 :=
sorry

end inequality_solution_set_l228_228787


namespace negation_equivalence_l228_228646

variable (U : Type) (S R : U → Prop)

-- Original statement: All students of this university are non-residents, i.e., ∀ x, S(x) → ¬ R(x)
def original_statement : Prop := ∀ x, S x → ¬ R x

-- Negation of the original statement: ∃ x, S(x) ∧ R(x)
def negated_statement : Prop := ∃ x, S x ∧ R x

-- Lean statement to prove that the negation of the original statement is equivalent to some students are residents
theorem negation_equivalence : ¬ original_statement U S R = negated_statement U S R :=
sorry

end negation_equivalence_l228_228646


namespace number_of_machines_l228_228807

theorem number_of_machines (X : ℕ)
  (h1 : 20 = (10 : ℝ) * X * 0.4) :
  X = 5 := sorry

end number_of_machines_l228_228807


namespace maximum_p_value_l228_228826

noncomputable def max_p_value (a b c : ℝ) : ℝ :=
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1)

theorem maximum_p_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a + c = b) :
  ∃ p_max, p_max = 10 / 3 ∧ ∀ p, p = max_p_value a b c → p ≤ p_max :=
sorry

end maximum_p_value_l228_228826


namespace range_of_x_for_acute_angle_l228_228517

theorem range_of_x_for_acute_angle (x : ℝ) (h₁ : (x, 2*x) ≠ (0, 0)) (h₂ : (x+1, x+3) ≠ (0, 0)) (h₃ : (3*x^2 + 7*x > 0)) : 
  x < -7/3 ∨ (0 < x ∧ x < 1) ∨ x > 1 :=
by {
  -- This theorem asserts the given range of x given the dot product solution.
  sorry
}

end range_of_x_for_acute_angle_l228_228517


namespace find_a_plus_b_l228_228806

theorem find_a_plus_b (a b : ℝ) (h₁ : ∀ x, x - b < 0 → x < b) 
  (h₂ : ∀ x, x + a > 0 → x > -a) 
  (h₃ : ∀ x, 2 < x ∧ x < 3 → -a < x ∧ x < b) : 
  a + b = 1 :=
by
  sorry

end find_a_plus_b_l228_228806


namespace speed_first_hour_l228_228382

theorem speed_first_hour (x : ℝ) :
  (∃ x, (x + 45) / 2 = 65) → x = 85 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  sorry

end speed_first_hour_l228_228382


namespace solve_for_k_l228_228394

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l228_228394


namespace SarahCansYesterday_l228_228765

variable (S : ℕ)
variable (LaraYesterday : ℕ := S + 30)
variable (SarahToday : ℕ := 40)
variable (LaraToday : ℕ := 70)
variable (YesterdayTotal : ℕ := LaraYesterday + S)
variable (TodayTotal : ℕ := SarahToday + LaraToday)

theorem SarahCansYesterday : 
  TodayTotal + 20 = YesterdayTotal -> 
  S = 50 :=
by
  sorry

end SarahCansYesterday_l228_228765


namespace tiffany_bags_l228_228817

/-!
## Problem Statement
Tiffany was collecting cans for recycling. On Monday she had some bags of cans. 
She found 3 bags of cans on the next day and 7 bags of cans the day after that. 
She had altogether 20 bags of cans. Prove that the number of bags of cans she had on Monday is 10.
-/

theorem tiffany_bags (M : ℕ) (h1 : M + 3 + 7 = 20) : M = 10 :=
by {
  sorry
}

end tiffany_bags_l228_228817


namespace real_number_x_equal_2_l228_228637

theorem real_number_x_equal_2 (x : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2 * i) * (x + i) = 4 - 3 * i → x = 2 :=
by
  sorry

end real_number_x_equal_2_l228_228637


namespace taller_tree_height_l228_228618

/-- The top of one tree is 20 feet higher than the top of another tree.
    The heights of the two trees are in the ratio 2:3.
    The shorter tree is 40 feet tall.
    Show that the height of the taller tree is 60 feet. -/
theorem taller_tree_height 
  (shorter_tree_height : ℕ) 
  (height_difference : ℕ)
  (height_ratio_num : ℕ)
  (height_ratio_denom : ℕ)
  (H1 : shorter_tree_height = 40)
  (H2 : height_difference = 20)
  (H3 : height_ratio_num = 2)
  (H4 : height_ratio_denom = 3)
  : ∃ taller_tree_height : ℕ, taller_tree_height = 60 :=
by
  sorry

end taller_tree_height_l228_228618


namespace polynomial_remainder_l228_228879

theorem polynomial_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (a b : ℝ),
    (x^150 = (x^2 - 5*x + 6) * Q x + (a*x + b)) ∧
    (2 * a + b = 2^150) ∧
    (3 * a + b = 3^150) ∧ 
    (a = 3^150 - 2^150) ∧ 
    (b = 2^150 - 2 * 3^150 + 2 * 2^150) := sorry

end polynomial_remainder_l228_228879


namespace functional_equation_implies_identity_l228_228616

theorem functional_equation_implies_identity 
  (f : ℝ → ℝ) 
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → 
    f ((x + y) / 2) + f ((2 * x * y) / (x + y)) = f x + f y) 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  : 2 * f (Real.sqrt (x * y)) = f x + f y := sorry

end functional_equation_implies_identity_l228_228616


namespace range_of_m_l228_228373

noncomputable def quadratic_function : Type := ℝ → ℝ

variable (f : quadratic_function)

axiom quadratic : ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x-2)^2 + b
axiom symmetry : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom condition1 : f 0 = 3
axiom condition2 : f 2 = 1
axiom max_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), f x ≤ 3
axiom min_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x

theorem range_of_m : ∀ m : ℝ, (∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x ∧ f x ≤ 3) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro m
  intro h
  sorry

end range_of_m_l228_228373


namespace original_oil_weight_is_75_l228_228678

def initial_oil_weight (original : ℝ) : Prop :=
  let first_remaining := original / 2
  let second_remaining := first_remaining * (4 / 5)
  second_remaining = 30

theorem original_oil_weight_is_75 : ∃ (original : ℝ), initial_oil_weight original ∧ original = 75 :=
by
  use 75
  unfold initial_oil_weight
  sorry

end original_oil_weight_is_75_l228_228678


namespace number_division_l228_228364

theorem number_division (x : ℤ) (h : x - 17 = 55) : x / 9 = 8 :=
by 
  sorry

end number_division_l228_228364


namespace product_of_abcd_l228_228748

theorem product_of_abcd :
  ∃ (a b c d : ℚ), 
    3 * a + 4 * b + 6 * c + 8 * d = 42 ∧ 
    4 * (d + c) = b ∧ 
    4 * b + 2 * c = a ∧ 
    c - 2 = d ∧ 
    a * b * c * d = (367 * 76 * 93 * -55) / (37^2 * 74^2) :=
sorry

end product_of_abcd_l228_228748


namespace cuboid_volume_l228_228788

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 14) (h_height : height = 13) : base_area * height = 182 := by
  sorry

end cuboid_volume_l228_228788


namespace work_rate_solution_l228_228848

theorem work_rate_solution (y : ℕ) (hy : y > 0) : 
  ∃ z : ℕ, z = (y^2 + 3 * y) / (2 * y + 3) :=
by
  sorry

end work_rate_solution_l228_228848


namespace range_of_m_for_roots_greater_than_2_l228_228020

theorem range_of_m_for_roots_greater_than_2 :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + (m-2)*x + 5 - m = 0 → x > 2) ↔ (-5 < m ∧ m ≤ -4) :=
  sorry

end range_of_m_for_roots_greater_than_2_l228_228020


namespace find_line_eqn_from_bisected_chord_l228_228052

noncomputable def line_eqn_from_bisected_chord (x y : ℝ) : Prop :=
  2 * x + y - 3 = 0

theorem find_line_eqn_from_bisected_chord (
  A B : ℝ × ℝ) 
  (hA :  (A.1^2) / 2 + (A.2^2) / 4 = 1)
  (hB :  (B.1^2) / 2 + (B.2^2) / 4 = 1)
  (h_mid : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  line_eqn_from_bisected_chord 1 1 :=
by 
  sorry

end find_line_eqn_from_bisected_chord_l228_228052


namespace LCM_quotient_l228_228693

-- Define M as the least common multiple of integers from 12 to 25
def LCM_12_25 : ℕ := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 
                       (Nat.lcm 12 13) 14) 15) 16) 17) (Nat.lcm (Nat.lcm 
                       (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 18 19) 20) 21) 22) 23) 24)

-- Define N as the least common multiple of LCM_12_25, 36, 38, 40, 42, 44, 45
def N : ℕ := Nat.lcm LCM_12_25 (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 36 38) 40) 42) (Nat.lcm 44 45))

-- Prove that N / LCM_12_25 = 1
theorem LCM_quotient : N / LCM_12_25 = 1 := by
    sorry

end LCM_quotient_l228_228693


namespace albert_wins_strategy_l228_228173

theorem albert_wins_strategy (n : ℕ) (h : n = 1999) : 
  ∃ strategy : (ℕ → ℕ), (∀ tokens : ℕ, tokens = n → tokens > 1 → 
  (∃ next_tokens : ℕ, next_tokens < tokens ∧ next_tokens ≥ 1 ∧ next_tokens ≥ tokens / 2) → 
  (∃ k, tokens = 2^k + 1) → strategy n = true) :=
sorry

end albert_wins_strategy_l228_228173


namespace coins_division_remainder_l228_228489

theorem coins_division_remainder
  (n : ℕ)
  (h1 : n % 6 = 4)
  (h2 : n % 5 = 3)
  (h3 : n = 28) :
  n % 7 = 0 :=
by
  sorry

end coins_division_remainder_l228_228489


namespace initial_mean_correctness_l228_228060

variable (M : ℝ)

theorem initial_mean_correctness (h1 : 50 * M + 20 = 50 * 36.5) : M = 36.1 :=
by 
  sorry

end initial_mean_correctness_l228_228060


namespace quadrilateral_area_l228_228298

theorem quadrilateral_area (EF FG EH HG : ℕ) (hEFH : EF * EF + FG * FG = 25)
(hEHG : EH * EH + HG * HG = 25) (h_distinct : EF ≠ EH ∧ FG ≠ HG) 
(h_greater_one : EF > 1 ∧ FG > 1 ∧ EH > 1 ∧ HG > 1) :
  (EF * FG) / 2 + (EH * HG) / 2 = 12 := 
sorry

end quadrilateral_area_l228_228298


namespace vector_satisfies_condition_l228_228874

def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 5 + 2 * t)
def line_m (s : ℝ) : ℝ × ℝ := (1 + 2 * s, 3 + 2 * s)

variable (A B P : ℝ × ℝ)

def vector_BA (B A : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vector_v : ℝ × ℝ := (1, -1)

theorem vector_satisfies_condition : 
  2 * vector_v.1 - vector_v.2 = 3 := by
  sorry

end vector_satisfies_condition_l228_228874


namespace num_divisors_not_divisible_by_2_of_360_l228_228414

def is_divisor (n d : ℕ) : Prop := d ∣ n

def is_prime (p : ℕ) : Prop := Nat.Prime p

noncomputable def prime_factors (n : ℕ) : List ℕ := sorry -- To be implemented if needed

def count_divisors_not_divisible_by_2 (n : ℕ) : ℕ :=
  let factors : List ℕ := prime_factors 360
  let a := 0
  let b_choices := [0, 1, 2]
  let c_choices := [0, 1]
  (b_choices.length) * (c_choices.length)

theorem num_divisors_not_divisible_by_2_of_360 :
  count_divisors_not_divisible_by_2 360 = 6 :=
by sorry

end num_divisors_not_divisible_by_2_of_360_l228_228414


namespace cars_sold_on_second_day_l228_228583

theorem cars_sold_on_second_day (x : ℕ) 
  (h1 : 14 + x + 27 = 57) : x = 16 :=
by 
  sorry

end cars_sold_on_second_day_l228_228583


namespace ratio_volumes_l228_228169

theorem ratio_volumes (hA rA hB rB : ℝ) (hA_def : hA = 30) (rA_def : rA = 15) (hB_def : hB = rA) (rB_def : rB = 2 * hA) :
    (1 / 3 * Real.pi * rA^2 * hA) / (1 / 3 * Real.pi * rB^2 * hB) = 1 / 24 :=
by
  -- skipping the proof
  sorry

end ratio_volumes_l228_228169


namespace simplify_decimal_l228_228160

theorem simplify_decimal : (3416 / 1000 : ℚ) = 427 / 125 := by
  sorry

end simplify_decimal_l228_228160


namespace algebraic_identity_neg_exponents_l228_228464

theorem algebraic_identity_neg_exponents (x y z : ℂ) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end algebraic_identity_neg_exponents_l228_228464


namespace f_12_eq_12_l228_228982

noncomputable def f : ℕ → ℤ := sorry

axiom f_int (n : ℕ) (hn : 0 < n) : ∃ k : ℤ, f n = k
axiom f_2 : f 2 = 2
axiom f_mul (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n
axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m > n → f m > f n

theorem f_12_eq_12 : f 12 = 12 := sorry

end f_12_eq_12_l228_228982


namespace hannah_strawberries_l228_228429

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end hannah_strawberries_l228_228429


namespace find_n_l228_228419

noncomputable def r1 : ℚ := 6 / 15
noncomputable def S1 : ℚ := 15 / (1 - r1)
noncomputable def r2 (n : ℚ) : ℚ := (6 + n) / 15
noncomputable def S2 (n : ℚ) : ℚ := 15 / (1 - r2 n)

theorem find_n : ∃ (n : ℚ), S2 n = 3 * S1 ∧ n = 6 :=
by
  use 6
  sorry

end find_n_l228_228419


namespace equilateral_triangle_of_condition_l228_228328

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + 2 * b^2 + c^2 - 2 * b * (a + c) = 0) : a = b ∧ b = c :=
by
  /- Proof goes here -/
  sorry

end equilateral_triangle_of_condition_l228_228328


namespace total_balloons_l228_228682

-- Define the conditions
def joan_balloons : ℕ := 9
def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2

-- The statement we want to prove
theorem total_balloons : joan_balloons + sally_balloons + jessica_balloons = 16 :=
by
  sorry

end total_balloons_l228_228682


namespace part1_part2_l228_228573

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l228_228573


namespace total_time_to_fill_tank_with_leak_l228_228677

theorem total_time_to_fill_tank_with_leak
  (C : ℝ) -- Capacity of the tank
  (rate1 : ℝ := C / 20) -- Rate of pipe 1 filling the tank
  (rate2 : ℝ := C / 30) -- Rate of pipe 2 filling the tank
  (combined_rate : ℝ := rate1 + rate2) -- Combined rate of both pipes
  (effective_rate : ℝ := (2 / 3) * combined_rate) -- Effective rate considering the leak
  : (C / effective_rate = 18) :=
by
  -- The proof would go here but is removed per the instructions.
  sorry

end total_time_to_fill_tank_with_leak_l228_228677


namespace max_blue_points_l228_228783

-- We define the number of spheres and the categorization of red and green spheres
def number_of_spheres : ℕ := 2016

-- Definition of the number of red spheres
def red_spheres (r : ℕ) : Prop := r <= number_of_spheres

-- Definition of the number of green spheres as the complement of red spheres
def green_spheres (r : ℕ) : ℕ := number_of_spheres - r

-- Definition of the number of blue points as the intersection of red and green spheres
def blue_points (r : ℕ) : ℕ := r * green_spheres r

-- Theorem: Given the conditions, the maximum number of blue points is 1016064
theorem max_blue_points : ∃ r : ℕ, red_spheres r ∧ blue_points r = 1016064 := by
  sorry

end max_blue_points_l228_228783


namespace stockholm_uppsala_distance_l228_228333

variable (map_distance : ℝ) (scale_factor : ℝ)

def actual_distance (d : ℝ) (s : ℝ) : ℝ := d * s

theorem stockholm_uppsala_distance :
  actual_distance 65 20 = 1300 := by
  sorry

end stockholm_uppsala_distance_l228_228333


namespace tan_add_pi_over_4_l228_228597

theorem tan_add_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_add_pi_over_4_l228_228597


namespace bd_squared_l228_228547

theorem bd_squared (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 9) : 
  (b - d) ^ 2 = 4 := 
sorry

end bd_squared_l228_228547


namespace seat_arrangement_l228_228707

theorem seat_arrangement :
  ∃ (arrangement : Fin 7 → String), 
  (arrangement 6 = "Diane") ∧
  (∃ (i j : Fin 7), i < j ∧ arrangement i = "Carla" ∧ arrangement j = "Adam" ∧ j = (i + 1)) ∧
  (∃ (i j k : Fin 7), i < j ∧ j < k ∧ arrangement i = "Brian" ∧ arrangement j = "Ellie" ∧ (k - i) ≥ 3) ∧
  arrangement 3 = "Carla" := 
sorry

end seat_arrangement_l228_228707


namespace am_gm_inequality_l228_228491

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) :=
sorry

end am_gm_inequality_l228_228491


namespace number_of_items_in_U_l228_228680

theorem number_of_items_in_U (U A B : Finset ℕ)
  (hB : B.card = 41)
  (not_A_nor_B : U.card - A.card - B.card + (A ∩ B).card = 59)
  (hAB : (A ∩ B).card = 23)
  (hA : A.card = 116) :
  U.card = 193 :=
by sorry

end number_of_items_in_U_l228_228680


namespace prob_top_three_cards_all_hearts_l228_228971

-- Define the total numbers of cards and suits
def total_cards := 52
def hearts_count := 13

-- Define the probability calculation as per the problem statement
def prob_top_three_hearts : ℚ :=
  (13 * 12 * 11 : ℚ) / (52 * 51 * 50 : ℚ)

-- The theorem states that the probability of the top three cards being all hearts is 11/850
theorem prob_top_three_cards_all_hearts : prob_top_three_hearts = 11 / 850 := by
  -- The proof details are not required, just stating the structure
  sorry

end prob_top_three_cards_all_hearts_l228_228971


namespace locus_of_P_is_ellipse_l228_228703

-- Definitions and conditions
def circle_A (x y : ℝ) : Prop := (x + 3) ^ 2 + y ^ 2 = 100
def fixed_point_B : ℝ × ℝ := (3, 0)
def circle_P_passes_through_B (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3) ^ 2 + center.2 ^ 2 = radius ^ 2
def circle_P_tangent_to_A_internally (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 + 3) ^ 2 + center.2 ^ 2 = (10 - radius) ^ 2

-- Statement of the problem to prove in Lean
theorem locus_of_P_is_ellipse :
  ∃ (foci_A B : ℝ × ℝ) (a b : ℝ), (foci_A = (-3, 0)) ∧ (foci_B = (3, 0)) ∧ (a = 5) ∧ (b = 4) ∧ 
  (∀ (x y : ℝ), (∃ (P : ℝ × ℝ) (radius : ℝ), circle_P_passes_through_B P radius ∧ circle_P_tangent_to_A_internally P radius ∧ P = (x, y)) ↔ 
  (x ^ 2) / 25 + (y ^ 2) / 16 = 1)
:=
sorry

end locus_of_P_is_ellipse_l228_228703


namespace speed_of_stream_l228_228883

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

end speed_of_stream_l228_228883


namespace probability_compare_l228_228890

-- Conditions
def v : ℝ := 0.1
def n : ℕ := 998

-- Binomial distribution formula
noncomputable def binom_prob (n k : ℕ) (v : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (v ^ k) * ((1 - v) ^ (n - k))

-- Theorem to prove
theorem probability_compare :
  binom_prob n 99 v > binom_prob n 100 v :=
by
  sorry

end probability_compare_l228_228890


namespace unique_real_solution_N_l228_228789

theorem unique_real_solution_N (N : ℝ) :
  (∃! (x y : ℝ), 2 * x^2 + 4 * x * y + 7 * y^2 - 12 * x - 2 * y + N = 0) ↔ N = 23 :=
by
  sorry

end unique_real_solution_N_l228_228789


namespace shaded_percentage_of_large_square_l228_228515

theorem shaded_percentage_of_large_square
  (side_length_small_square : ℕ)
  (side_length_large_square : ℕ)
  (total_border_squares : ℕ)
  (shaded_border_squares : ℕ)
  (central_region_shaded_fraction : ℚ)
  (total_area_large_square : ℚ)
  (shaded_area_border_squares : ℚ)
  (shaded_area_central_region : ℚ) :
  side_length_small_square = 1 →
  side_length_large_square = 5 →
  total_border_squares = 16 →
  shaded_border_squares = 8 →
  central_region_shaded_fraction = 3 / 4 →
  total_area_large_square = 25 →
  shaded_area_border_squares = 8 →
  shaded_area_central_region = (3 / 4) * 9 →
  (shaded_area_border_squares + shaded_area_central_region) / total_area_large_square = 0.59 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end shaded_percentage_of_large_square_l228_228515


namespace roots_absolute_value_l228_228831

noncomputable def quadratic_roots_property (p : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 ≠ r2 ∧
  r1 + r2 = -p ∧
  r1 * r2 = 16 ∧
  ∃ r : ℝ, r = r1 ∨ r = r2 ∧ abs r > 4

theorem roots_absolute_value (p : ℝ) (r1 r2 : ℝ) :
  quadratic_roots_property p r1 r2 → ∃ r : ℝ, (r = r1 ∨ r = r2) ∧ abs r > 4 :=
sorry

end roots_absolute_value_l228_228831


namespace total_bill_is_95_l228_228960

noncomputable def total_bill := 28 + 8 + 10 + 6 + 14 + 11 + 12 + 6

theorem total_bill_is_95 : total_bill = 95 := by
  sorry

end total_bill_is_95_l228_228960


namespace friends_meeting_probability_l228_228555

noncomputable def n_value (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2) : ℝ :=
  d - e * Real.sqrt f

theorem friends_meeting_probability (n : ℝ) (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2)
  (H : n = n_value d e f h1 h2 h3) : d + e + f = 92 :=
  by
  sorry

end friends_meeting_probability_l228_228555


namespace perimeter_of_regular_polygon_l228_228007

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : n = 3) (h2 : side_length = 5) (h3 : exterior_angle = 120) : 
  n * side_length = 15 :=
by
  sorry

end perimeter_of_regular_polygon_l228_228007


namespace find_range_of_a_l228_228405

def have_real_roots (a : ℝ) : Prop := a^2 - 16 ≥ 0

def is_increasing_on_interval (a : ℝ) : Prop := a ≥ -12

theorem find_range_of_a (a : ℝ) : ((have_real_roots a ∨ is_increasing_on_interval a) ∧ ¬(have_real_roots a ∧ is_increasing_on_interval a)) → (a < -12 ∨ (-4 < a ∧ a < 4)) :=
by 
  sorry

end find_range_of_a_l228_228405


namespace necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l228_228231

-- Proof Problem 1
theorem necessary_condition_for_q_implies_m_in_range (m : ℝ) (h1 : 0 < m) :
  (∀ x : ℝ, 2 - m ≤ x ∧ x ≤ 2 + m → -2 ≤ x ∧ x ≤ 6) →
  0 < m ∧ m ≤ 4 :=
by
  sorry

-- Proof Problem 2
theorem neg_p_or_neg_q_false_implies_x_in_range (m : ℝ) (x : ℝ)
  (h2 : m = 2)
  (h3 : (x + 2) * (x - 6) ≤ 0)
  (h4 : 2 - m ≤ x ∧ x ≤ 2 + m)
  (h5 : ¬ ((x + 2) * (x - 6) > 0 ∨ x < 2 - m ∨ x > 2 + m)) :
  0 ≤ x ∧ x ≤ 4 :=
by
  sorry

end necessary_condition_for_q_implies_m_in_range_neg_p_or_neg_q_false_implies_x_in_range_l228_228231


namespace current_prices_l228_228659

theorem current_prices (initial_ram_price initial_ssd_price : ℝ) 
  (ram_increase_1 ram_decrease_1 ram_decrease_2 : ℝ) 
  (ssd_increase_1 ssd_decrease_1 ssd_decrease_2 : ℝ) 
  (initial_ram : initial_ram_price = 50) 
  (initial_ssd : initial_ssd_price = 100) 
  (ram_increase_factor : ram_increase_1 = 0.30 * initial_ram_price) 
  (ram_decrease_factor_1 : ram_decrease_1 = 0.15 * (initial_ram_price + ram_increase_1)) 
  (ram_decrease_factor_2 : ram_decrease_2 = 0.20 * ((initial_ram_price + ram_increase_1) - ram_decrease_1)) 
  (ssd_increase_factor : ssd_increase_1 = 0.10 * initial_ssd_price) 
  (ssd_decrease_factor_1 : ssd_decrease_1 = 0.05 * (initial_ssd_price + ssd_increase_1)) 
  (ssd_decrease_factor_2 : ssd_decrease_2 = 0.12 * ((initial_ssd_price + ssd_increase_1) - ssd_decrease_1)) 
  : 
  ((initial_ram_price + ram_increase_1 - ram_decrease_1 - ram_decrease_2) = 44.20) ∧ 
  ((initial_ssd_price + ssd_increase_1 - ssd_decrease_1 - ssd_decrease_2) = 91.96) := 
by
  sorry

end current_prices_l228_228659


namespace perimeter_of_equilateral_triangle_l228_228929

theorem perimeter_of_equilateral_triangle (s : ℝ) 
  (h1 : (s ^ 2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_equilateral_triangle_l228_228929


namespace concert_cost_l228_228090

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l228_228090


namespace theta_third_quadrant_l228_228737

theorem theta_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  π < θ ∧ θ < 3 * π / 2 :=
by 
  sorry

end theta_third_quadrant_l228_228737


namespace part_a_part_b_l228_228617

namespace ShaltaevBoltaev

variables {s b : ℕ}

-- Condition: 175s > 125b
def condition1 (s b : ℕ) : Prop := 175 * s > 125 * b

-- Condition: 175s < 126b
def condition2 (s b : ℕ) : Prop := 175 * s < 126 * b

-- Prove that 3s + b > 80
theorem part_a (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 80 := sorry

-- Prove that 3s + b > 100
theorem part_b (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 100 := sorry

end ShaltaevBoltaev

end part_a_part_b_l228_228617


namespace Carl_avg_gift_bags_l228_228777

theorem Carl_avg_gift_bags :
  ∀ (known expected extravagant remaining : ℕ), 
  known = 50 →
  expected = 40 →
  extravagant = 10 →
  remaining = 60 →
  (known + expected) - extravagant - remaining = 30 := by
  intros
  sorry

end Carl_avg_gift_bags_l228_228777


namespace probability_of_gui_field_in_za_field_l228_228768

noncomputable def area_gui_field (base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * base * height

noncomputable def area_za_field (small_base large_base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (small_base + large_base) * height

theorem probability_of_gui_field_in_za_field :
  let b1 := 10
  let b2 := 20
  let h1 := 10
  let base_gui := 8
  let height_gui := 5
  let za_area := area_za_field b1 b2 h1
  let gui_area := area_gui_field base_gui height_gui
  (gui_area / za_area) = (2 / 15 : ℚ) := by
    sorry

end probability_of_gui_field_in_za_field_l228_228768


namespace dress_hem_length_in_feet_l228_228310

def stitch_length_in_inches : ℚ := 1 / 4
def stitches_per_minute : ℕ := 24
def time_in_minutes : ℕ := 6

theorem dress_hem_length_in_feet :
  (stitch_length_in_inches * (stitches_per_minute * time_in_minutes)) / 12 = 3 :=
by
  sorry

end dress_hem_length_in_feet_l228_228310


namespace perpendicular_lines_foot_l228_228143

variables (a b c : ℝ)

theorem perpendicular_lines_foot (h1 : a * -2/20 = -1)
  (h2_foot_l1 : a * 1 + 4 * c - 2 = 0)
  (h3_foot_l2 : 2 * 1 - 5 * c + b = 0) :
  a + b + c = -4 :=
sorry

end perpendicular_lines_foot_l228_228143


namespace compute_sum_of_squares_l228_228258

noncomputable def polynomial_roots (p q r : ℂ) : Prop := 
  (p^3 - 15 * p^2 + 22 * p - 8 = 0) ∧ 
  (q^3 - 15 * q^2 + 22 * q - 8 = 0) ∧ 
  (r^3 - 15 * r^2 + 22 * r - 8 = 0) 

theorem compute_sum_of_squares (p q r : ℂ) (h : polynomial_roots p q r) :
  (p + q) ^ 2 + (q + r) ^ 2 + (r + p) ^ 2 = 406 := 
sorry

end compute_sum_of_squares_l228_228258


namespace total_animals_l228_228641

theorem total_animals (H C2 C1 : ℕ) (humps_eq : 2 * C2 + C1 = 200) (horses_eq : H = C2) :
  H + C2 + C1 = 200 :=
by
  /- Proof steps are not required -/
  sorry

end total_animals_l228_228641


namespace least_positive_multiple_of_17_gt_500_l228_228250

theorem least_positive_multiple_of_17_gt_500 : ∃ n: ℕ, n > 500 ∧ n % 17 = 0 ∧ n = 510 := by
  sorry

end least_positive_multiple_of_17_gt_500_l228_228250


namespace units_digit_of_1583_pow_1246_l228_228225

theorem units_digit_of_1583_pow_1246 : 
  (1583^1246) % 10 = 9 := 
sorry

end units_digit_of_1583_pow_1246_l228_228225


namespace perpendicular_line_plane_implies_perpendicular_lines_l228_228705

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Assume inclusion of lines in planes, parallelism, and perpendicularity properties.
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)

-- Given definitions based on the conditions
variable (is_perpendicular : perpendicular m α)
variable (is_subset : subset n α)

-- Prove that m is perpendicular to n
theorem perpendicular_line_plane_implies_perpendicular_lines
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end perpendicular_line_plane_implies_perpendicular_lines_l228_228705


namespace find_number_l228_228103

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l228_228103


namespace least_tiles_required_l228_228229

def floor_length : ℕ := 5000
def floor_breadth : ℕ := 1125
def gcd_floor : ℕ := Nat.gcd floor_length floor_breadth
def tile_area : ℕ := gcd_floor ^ 2
def floor_area : ℕ := floor_length * floor_breadth
def tiles_count : ℕ := floor_area / tile_area

theorem least_tiles_required : tiles_count = 360 :=
by
  sorry

end least_tiles_required_l228_228229


namespace not_power_of_two_l228_228184

theorem not_power_of_two (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ¬ ∃ k : ℕ, (36 * m + n) * (m + 36 * n) = 2 ^ k :=
sorry

end not_power_of_two_l228_228184


namespace pool_filled_in_48_minutes_with_both_valves_open_l228_228251

def rate_first_valve_fills_pool_in_2_hours (V1 : ℚ) : Prop :=
  V1 * 120 = 12000

def rate_second_valve_50_more_than_first (V1 V2 : ℚ) : Prop :=
  V2 = V1 + 50

def pool_capacity : ℚ := 12000

def combined_rate (V1 V2 combinedRate : ℚ) : Prop :=
  combinedRate = V1 + V2

def time_to_fill_pool_with_both_valves_open (combinedRate time : ℚ) : Prop :=
  time = pool_capacity / combinedRate

theorem pool_filled_in_48_minutes_with_both_valves_open
  (V1 V2 combinedRate time : ℚ) :
  rate_first_valve_fills_pool_in_2_hours V1 →
  rate_second_valve_50_more_than_first V1 V2 →
  combined_rate V1 V2 combinedRate →
  time_to_fill_pool_with_both_valves_open combinedRate time →
  time = 48 :=
by
  intros
  sorry

end pool_filled_in_48_minutes_with_both_valves_open_l228_228251


namespace similar_triangles_proportionalities_l228_228653

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end similar_triangles_proportionalities_l228_228653


namespace solution_set_of_inequality_l228_228459

theorem solution_set_of_inequality (x : ℝ) : (x - 1 ≤ (1 + x) / 3) → (x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l228_228459


namespace base_conversion_min_sum_l228_228305

theorem base_conversion_min_sum (a b : ℕ) (h1 : 3 * a + 6 = 6 * b + 3) (h2 : 6 < a) (h3 : 6 < b) : a + b = 20 :=
sorry

end base_conversion_min_sum_l228_228305


namespace Carmen_average_speed_l228_228596

/-- Carmen participates in a two-part cycling race. In the first part, she covers 24 miles in 3 hours.
    In the second part, due to fatigue, her speed decreases, and she takes 4 hours to cover 16 miles.
    Calculate Carmen's average speed for the entire race. -/
theorem Carmen_average_speed :
  let distance1 := 24 -- miles in the first part
  let time1 := 3 -- hours in the first part
  let distance2 := 16 -- miles in the second part
  let time2 := 4 -- hours in the second part
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 40 / 7 :=
by
  sorry

end Carmen_average_speed_l228_228596


namespace four_digit_number_difference_l228_228844

theorem four_digit_number_difference
    (digits : List ℕ) (h_digits : digits = [2, 0, 1, 3, 1, 2, 2, 1, 0, 8, 4, 0])
    (max_val : ℕ) (h_max_val : max_val = 3840)
    (min_val : ℕ) (h_min_val : min_val = 1040) :
    max_val - min_val = 2800 :=
by
    sorry

end four_digit_number_difference_l228_228844


namespace hexagon_perimeter_eq_4_sqrt_3_over_3_l228_228546

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_eq_4_sqrt_3_over_3 :
  ∀ (s : ℝ), (∃ s, (3 * Real.sqrt 3 / 2) * s^2 = s) → hexagon_perimeter s = 4 * Real.sqrt 3 / 3 :=
by
  simp
  sorry

end hexagon_perimeter_eq_4_sqrt_3_over_3_l228_228546


namespace rectangle_area_from_diagonal_l228_228757

theorem rectangle_area_from_diagonal (x : ℝ) (w : ℝ) (h_lw : 3 * w = 3 * w) (h_diag : x^2 = 10 * w^2) : 
    (3 * w^2 = (3 / 10) * x^2) :=
by 
sorry

end rectangle_area_from_diagonal_l228_228757


namespace solve_real_equation_l228_228513

theorem solve_real_equation (x : ℝ) (h : x^4 + (3 - x)^4 = 82) : x = 2.5 ∨ x = 0.5 :=
sorry

end solve_real_equation_l228_228513


namespace geometric_sequence_properties_l228_228975

theorem geometric_sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  (∀ n, a n = 2^(n - 1)) ∧ (S 6 = 63) := 
by 
  sorry

end geometric_sequence_properties_l228_228975


namespace total_pokemon_cards_l228_228228

def initial_cards : Nat := 27
def received_cards : Nat := 41
def lost_cards : Nat := 20

theorem total_pokemon_cards : initial_cards + received_cards - lost_cards = 48 := by
  sorry

end total_pokemon_cards_l228_228228


namespace min_point_transformed_graph_l228_228378

noncomputable def original_eq (x : ℝ) : ℝ := 2 * |x| - 4

noncomputable def translated_eq (x : ℝ) : ℝ := 2 * |x - 3| - 8

theorem min_point_transformed_graph : translated_eq 3 = -8 :=
by
  -- Solution steps would go here
  sorry

end min_point_transformed_graph_l228_228378


namespace point_quadrant_I_or_IV_l228_228023

def is_point_on_line (x y : ℝ) : Prop := 4 * x + 3 * y = 12
def is_equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

def point_in_quadrant_I (x y : ℝ) : Prop := (x > 0 ∧ y > 0)
def point_in_quadrant_IV (x y : ℝ) : Prop := (x > 0 ∧ y < 0)

theorem point_quadrant_I_or_IV (x y : ℝ) 
  (h1 : is_point_on_line x y) 
  (h2 : is_equidistant_from_axes x y) :
  point_in_quadrant_I x y ∨ point_in_quadrant_IV x y :=
sorry

end point_quadrant_I_or_IV_l228_228023


namespace binomial_12_11_eq_12_l228_228700

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l228_228700


namespace cards_difference_l228_228601

theorem cards_difference
  (H : ℕ)
  (F : ℕ)
  (B : ℕ)
  (hH : H = 200)
  (hF : F = 4 * H)
  (hTotal : B + F + H = 1750) :
  F - B = 50 :=
by
  sorry

end cards_difference_l228_228601


namespace percentage_students_enrolled_in_bio_l228_228369

-- Problem statement
theorem percentage_students_enrolled_in_bio (total_students : ℕ) (students_not_in_bio : ℕ) 
    (h1 : total_students = 880) (h2 : students_not_in_bio = 462) : 
    ((total_students - students_not_in_bio : ℚ) / total_students) * 100 = 47.5 := by 
  -- Proof is omitted
  sorry

end percentage_students_enrolled_in_bio_l228_228369


namespace model_y_completion_time_l228_228262

theorem model_y_completion_time :
  ∀ (T : ℝ), (∃ k ≥ 0, k = 20) →
  (∀ (task_completed_x_per_minute : ℝ), task_completed_x_per_minute = 1 / 60) →
  (∀ (task_completed_y_per_minute : ℝ), task_completed_y_per_minute = 1 / T) →
  (20 * (1 / 60) + 20 * (1 / T) = 1) →
  T = 30 :=
by
  sorry

end model_y_completion_time_l228_228262


namespace range_of_m_l228_228434

noncomputable def is_quadratic (m : ℝ) : Prop := (m^2 - 4) ≠ 0

theorem range_of_m (m : ℝ) : is_quadratic m → m ≠ 2 ∧ m ≠ -2 :=
by sorry

end range_of_m_l228_228434


namespace total_distance_between_alice_bob_l228_228221

-- Define the constants for Alice's and Bob's speeds and the time duration in terms of conditions.
def alice_speed := 1 / 12  -- miles per minute
def bob_speed := 3 / 20    -- miles per minute
def time_duration := 120   -- minutes

-- Statement: Prove that the total distance between Alice and Bob after 2 hours is 28 miles.
theorem total_distance_between_alice_bob : (alice_speed * time_duration) + (bob_speed * time_duration) = 28 :=
by
  sorry

end total_distance_between_alice_bob_l228_228221


namespace team_A_has_more_uniform_heights_l228_228812

-- Definitions of the conditions
def avg_height_team_A : ℝ := 1.65
def avg_height_team_B : ℝ := 1.65

def variance_team_A : ℝ := 1.5
def variance_team_B : ℝ := 2.4

-- Theorem stating the problem solution
theorem team_A_has_more_uniform_heights :
  variance_team_A < variance_team_B :=
by
  -- Proof omitted
  sorry

end team_A_has_more_uniform_heights_l228_228812


namespace solution_l228_228144

noncomputable def x : ℕ := 13

theorem solution : (3 * x) - (36 - x) = 16 := by
  sorry

end solution_l228_228144


namespace probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l228_228476

noncomputable def diameter := 19 -- mm
noncomputable def side_length := 50 -- mm, side length of each square
noncomputable def total_area := side_length^2 -- 2500 mm^2 for each square
noncomputable def coin_radius := diameter / 2 -- 9.5 mm

theorem probability_completely_inside_square : 
  (side_length - 2 * coin_radius)^2 / total_area = 961 / 2500 :=
by sorry

theorem probability_partial_one_edge :
  4 * ((side_length - 2 * coin_radius) * coin_radius) / total_area = 1178 / 2500 :=
by sorry

theorem probability_partial_two_edges_not_vertex :
  (4 * ((diameter)^2 - (coin_radius^2 * Real.pi / 4))) / total_area = (4 * 290.12) / 2500 :=
by sorry

theorem probability_vertex :
  4 * (coin_radius^2 * Real.pi / 4) / total_area = 4 * 70.88 / 2500 :=
by sorry

end probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l228_228476


namespace largest_room_width_l228_228152

theorem largest_room_width (w : ℕ) :
  (w * 30 - 15 * 8 = 1230) → (w = 45) :=
by
  intro h
  sorry

end largest_room_width_l228_228152


namespace initial_plank_count_l228_228956

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

end initial_plank_count_l228_228956


namespace circle_equation_passing_through_P_l228_228770

-- Define the problem conditions
def P : ℝ × ℝ := (3, 1)
def l₁ (x y : ℝ) := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) := x + 2 * y - 7 = 0

-- The main theorem statement
theorem circle_equation_passing_through_P :
  ∃ (α β : ℝ), 
    ((α = 4 ∧ β = -1) ∨ (α = 4 / 5 ∧ β = 3 / 5)) ∧ 
    ((x - α)^2 + (y - β)^2 = 5) :=
  sorry

end circle_equation_passing_through_P_l228_228770


namespace last_passenger_seats_probability_l228_228614

theorem last_passenger_seats_probability (n : ℕ) (hn : n > 0) :
  ∀ (P : ℝ), P = 1 / 2 :=
by
  sorry

end last_passenger_seats_probability_l228_228614


namespace n_power_four_plus_sixtyfour_power_n_composite_l228_228645

theorem n_power_four_plus_sixtyfour_power_n_composite (n : ℕ) : ∃ m k, m * k = n^4 + 64^n ∧ m > 1 ∧ k > 1 :=
by
  sorry

end n_power_four_plus_sixtyfour_power_n_composite_l228_228645


namespace regular_seven_gon_l228_228620

theorem regular_seven_gon 
    (A : Fin 7 → ℝ × ℝ)
    (cong_diagonals_1 : ∀ (i : Fin 7), dist (A i) (A ((i + 2) % 7)) = dist (A 0) (A 2))
    (cong_diagonals_2 : ∀ (i : Fin 7), dist (A i) (A ((i + 3) % 7)) = dist (A 0) (A 3))
    : ∀ (i j : Fin 7), dist (A i) (A ((i + 1) % 7)) = dist (A j) (A ((j + 1) % 7)) :=
by sorry

end regular_seven_gon_l228_228620


namespace sample_size_is_100_l228_228384

-- Define the number of students selected for the sample.
def num_students_sampled : ℕ := 100

-- The statement that the sample size is equal to the number of students sampled.
theorem sample_size_is_100 : num_students_sampled = 100 := 
by {
  -- Proof goes here
  sorry
}

end sample_size_is_100_l228_228384


namespace find_pictures_museum_l228_228198

-- Define the given conditions
def pictures_zoo : Nat := 24
def pictures_deleted : Nat := 14
def pictures_remaining : Nat := 22

-- Define the target: the number of pictures taken at the museum
def pictures_museum : Nat := 12

-- State the goal to be proved
theorem find_pictures_museum :
  pictures_zoo + pictures_museum - pictures_deleted = pictures_remaining :=
sorry

end find_pictures_museum_l228_228198


namespace sara_caught_five_trout_l228_228665

theorem sara_caught_five_trout (S M : ℕ) (h1 : M = 2 * S) (h2 : M = 10) : S = 5 :=
by
  sorry

end sara_caught_five_trout_l228_228665


namespace tan_function_constants_l228_228066

theorem tan_function_constants (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_period : b ≠ 0 ∧ ∃ k : ℤ, b * (3 / 2) = k * π) 
(h_pass : a * Real.tan (b * (π / 4)) = 3) : a * b = 2 * Real.sqrt 3 :=
by 
  sorry

end tan_function_constants_l228_228066


namespace negation_of_p_l228_228395

-- Define the original predicate
def p (x₀ : ℝ) : Prop := x₀^2 > 1

-- Define the negation of the predicate
def not_p : Prop := ∀ x : ℝ, x^2 ≤ 1

-- Prove the negation of the proposition
theorem negation_of_p : (∃ x₀ : ℝ, p x₀) ↔ not_p := by
  sorry

end negation_of_p_l228_228395


namespace soybean_cornmeal_proof_l228_228940

theorem soybean_cornmeal_proof :
  ∃ (x y : ℝ), 
    (0.14 * x + 0.07 * y = 0.13 * 280) ∧
    (x + y = 280) ∧
    (x = 240) ∧
    (y = 40) :=
by
  sorry

end soybean_cornmeal_proof_l228_228940


namespace distance_from_pole_l228_228337

-- Define the structure for polar coordinates.
structure PolarCoordinates where
  r : ℝ
  θ : ℝ

-- Define point A with its polar coordinates.
def A : PolarCoordinates := { r := 3, θ := -4 }

-- State the problem to prove that the distance |OA| is 3.
theorem distance_from_pole (A : PolarCoordinates) : A.r = 3 :=
by {
  sorry
}

end distance_from_pole_l228_228337


namespace polygon_sides_l228_228094

theorem polygon_sides (n : ℕ) (f : ℕ) (h1 : f = n * (n - 3) / 2) (h2 : 2 * n = f) : n = 7 :=
  by
  sorry

end polygon_sides_l228_228094


namespace stickers_distribution_l228_228320

theorem stickers_distribution : 
  (10 + 5 - 1).choose (5 - 1) = 1001 := 
by
  sorry

end stickers_distribution_l228_228320


namespace rank_of_matrix_A_is_2_l228_228423

def matrix_A : Matrix (Fin 4) (Fin 5) ℚ :=
  ![![3, -1, 1, 2, -8],
    ![7, -1, 2, 1, -12],
    ![11, -1, 3, 0, -16],
    ![10, -2, 3, 3, -20]]

theorem rank_of_matrix_A_is_2 : Matrix.rank matrix_A = 2 := by
  sorry

end rank_of_matrix_A_is_2_l228_228423


namespace ceil_sqrt_169_eq_13_l228_228418

theorem ceil_sqrt_169_eq_13 : Int.ceil (Real.sqrt 169) = 13 := by
  sorry

end ceil_sqrt_169_eq_13_l228_228418


namespace Lena_stops_in_X_l228_228449

def circumference : ℕ := 60
def distance_run : ℕ := 7920
def starting_point : String := "T"
def quarter_stops : String := "X"

theorem Lena_stops_in_X :
  (distance_run / circumference) * circumference + (distance_run % circumference) = distance_run →
  distance_run % circumference = 0 →
  (distance_run % circumference = 0 → starting_point = quarter_stops) →
  quarter_stops = "X" :=
sorry

end Lena_stops_in_X_l228_228449


namespace max_price_reduction_l228_228692

theorem max_price_reduction (CP SP : ℝ) (profit_margin : ℝ) (max_reduction : ℝ) :
  CP = 1000 ∧ SP = 1500 ∧ profit_margin = 0.05 → SP - max_reduction = CP * (1 + profit_margin) → max_reduction = 450 :=
by {
  sorry
}

end max_price_reduction_l228_228692


namespace sin_alpha_value_l228_228531

open Real

theorem sin_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 4) = 1 / 3) :
  sin α = (4 - sqrt 2) / 6 :=
sorry

end sin_alpha_value_l228_228531


namespace compute_sin_product_l228_228600

theorem compute_sin_product : 
  (1 - Real.sin (Real.pi / 12)) *
  (1 - Real.sin (5 * Real.pi / 12)) *
  (1 - Real.sin (7 * Real.pi / 12)) *
  (1 - Real.sin (11 * Real.pi / 12)) = 
  (1 / 16) :=
by
  sorry

end compute_sin_product_l228_228600


namespace purchase_price_of_article_l228_228730

theorem purchase_price_of_article (P M : ℝ) (h1 : M = 55) (h2 : M = 0.30 * P + 12) : P = 143.33 :=
  sorry

end purchase_price_of_article_l228_228730


namespace bullet_train_speed_l228_228279

theorem bullet_train_speed 
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_cross : ℝ)
  (combined_length : ℝ)
  (time_cross_hours : ℝ)
  (relative_speed : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 270 → 
  length_train2 = 230.04 →
  speed_train2 = 80 →
  time_cross = 9 →
  combined_length = (length_train1 + length_train2) / 1000 →
  time_cross_hours = time_cross / 3600 →
  relative_speed = combined_length / time_cross_hours →
  relative_speed = speed_train1 + speed_train2 →
  speed_train1 = 120.016 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bullet_train_speed_l228_228279


namespace square_nonneg_l228_228924

theorem square_nonneg (x h k : ℝ) (h_eq: (x + h)^2 = k) : k ≥ 0 := 
by 
  sorry

end square_nonneg_l228_228924


namespace expand_product_l228_228579

theorem expand_product (x : ℝ) (hx : x ≠ 0) : (3 / 7) * (7 / x - 5 * x ^ 3) = 3 / x - (15 / 7) * x ^ 3 :=
by
  sorry

end expand_product_l228_228579


namespace evaluate_expression_l228_228085

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 8 = 43046724 := 
by
  sorry

end evaluate_expression_l228_228085


namespace percentage_died_by_bombardment_l228_228018

theorem percentage_died_by_bombardment (P_initial : ℝ) (P_remaining : ℝ) (died_percentage : ℝ) (fear_percentage : ℝ) :
  P_initial = 3161 → P_remaining = 2553 → fear_percentage = 0.15 → 
  P_initial - (died_percentage/100) * P_initial - fear_percentage * (P_initial - (died_percentage/100) * P_initial) = P_remaining → 
  abs (died_percentage - 4.98) < 0.01 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_died_by_bombardment_l228_228018


namespace rate_of_fencing_is_4_90_l228_228481

noncomputable def rate_of_fencing_per_meter : ℝ :=
  let area_hectares := 13.86
  let cost := 6466.70
  let area_m2 := area_hectares * 10000
  let radius := Real.sqrt (area_m2 / Real.pi)
  let circumference := 2 * Real.pi * radius
  cost / circumference

theorem rate_of_fencing_is_4_90 :
  rate_of_fencing_per_meter = 4.90 := sorry

end rate_of_fencing_is_4_90_l228_228481


namespace cube_inscribed_circumscribed_volume_ratio_l228_228482

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio_l228_228482


namespace range_of_x_in_function_l228_228514

theorem range_of_x_in_function (x : ℝ) : (y = 1/(x + 3) → x ≠ -3) :=
sorry

end range_of_x_in_function_l228_228514


namespace min_pq_value_l228_228022

theorem min_pq_value : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 98 * p = q ^ 3 ∧ (∀ p' q' : ℕ, p' > 0 ∧ q' > 0 ∧ 98 * p' = q' ^ 3 → p' + q' ≥ p + q) ∧ p + q = 42 :=
sorry

end min_pq_value_l228_228022


namespace remainder_of_large_product_mod_17_l228_228072

theorem remainder_of_large_product_mod_17 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end remainder_of_large_product_mod_17_l228_228072


namespace evaluate_expression_l228_228259

theorem evaluate_expression :
  let a := 2020
  let b := 2016
  (2^a + 2^b) / (2^a - 2^b) = 17 / 15 :=
by
  sorry

end evaluate_expression_l228_228259


namespace largest_non_representable_as_sum_of_composites_l228_228816

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l228_228816


namespace radius_formula_l228_228630

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  let angle := 42 * Real.pi / 180 -- converting 42 degrees to radians
  let R := a / (Real.sqrt 3)
  let h := R * Real.tan angle
  Real.sqrt ((R * R) + (h * h))

theorem radius_formula (a : ℝ) : radius_of_circumscribed_sphere a = (a * Real.sqrt 3) / 3 :=
by
  sorry

end radius_formula_l228_228630


namespace solution_part1_solution_part2_l228_228274

variable (f : ℝ → ℝ) (a x m : ℝ)

def problem_statement :=
  (∀ x : ℝ, f x = abs (x - a)) ∧
  (∀ x : ℝ, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5)

theorem solution_part1 (x : ℝ) (h : problem_statement f a) : a = 2 :=
by
  sorry

theorem solution_part2 (x : ℝ) (h : problem_statement f a) :
  (∀ x : ℝ, f x + f (x + 5) ≥ m) → m ≤ 5 :=
by
  sorry

end solution_part1_solution_part2_l228_228274


namespace find_u_l228_228133

theorem find_u (u : ℝ) : (∃ x : ℝ, x = ( -15 - Real.sqrt 145 ) / 8 ∧ 4 * x^2 + 15 * x + u = 0) ↔ u = 5 := by
  sorry

end find_u_l228_228133


namespace students_enjoy_both_music_and_sports_l228_228744

theorem students_enjoy_both_music_and_sports :
  ∀ (T M S N B : ℕ), T = 55 → M = 35 → S = 45 → N = 4 → B = M + S - (T - N) → B = 29 :=
by
  intros T M S N B hT hM hS hN hB
  rw [hT, hM, hS, hN] at hB
  exact hB

end students_enjoy_both_music_and_sports_l228_228744


namespace find_n_for_arithmetic_sequence_l228_228304

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence_l228_228304


namespace minimum_value_x_2y_l228_228522

theorem minimum_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) : x + 2 * y = 8 :=
sorry

end minimum_value_x_2y_l228_228522


namespace tom_books_l228_228453

-- Definitions based on the conditions
def joan_books : ℕ := 10
def total_books : ℕ := 48

-- The theorem statement: Proving that Tom has 38 books
theorem tom_books : (total_books - joan_books) = 38 := by
  -- Here we would normally provide a proof, but we use sorry to skip this.
  sorry

end tom_books_l228_228453


namespace reduction_amount_is_250_l228_228698

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250_l228_228698


namespace consistent_scale_l228_228162

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale_l228_228162


namespace parabola_equation_hyperbola_equation_l228_228316

-- Part 1: Prove the standard equation of the parabola given the directrix.
theorem parabola_equation (x y : ℝ) : x = -2 → y^2 = 8 * x := 
by
  -- Here we will include proof steps based on given conditions
  sorry

-- Part 2: Prove the standard equation of the hyperbola given center at origin, focus on the x-axis,
-- the given asymptotes, and its real axis length.
theorem hyperbola_equation (x y a b : ℝ) : 
  a = 1 → b = 2 → y = 2 * x ∨ y = -2 * x → x^2 - (y^2 / 4) = 1 :=
by
  -- Here we will include proof steps based on given conditions
  sorry

end parabola_equation_hyperbola_equation_l228_228316


namespace football_cost_l228_228296

-- Definitions derived from conditions
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52
def total_spent : ℝ := 20.52

-- The statement to prove the cost of the football
theorem football_cost :
  ∃ (football_cost : ℝ), football_cost = total_spent - marbles_cost - baseball_cost :=
sorry

end football_cost_l228_228296


namespace find_youngest_age_l228_228388

noncomputable def youngest_child_age 
  (meal_cost_mother : ℝ) 
  (meal_cost_per_year : ℝ) 
  (total_bill : ℝ) 
  (triplets_count : ℕ) := 
  {y : ℝ // 
    (∃ t : ℝ, 
      meal_cost_mother + meal_cost_per_year * (triplets_count * t + y) = total_bill ∧ y = 2 ∨ y = 5)}

theorem find_youngest_age : 
  youngest_child_age 3.75 0.50 12.25 3 := 
sorry

end find_youngest_age_l228_228388


namespace gumball_draw_probability_l228_228581

def prob_blue := 2 / 3
def prob_two_blue := (16 / 36)
def prob_pink := 1 - prob_blue

theorem gumball_draw_probability
    (h1 : prob_two_blue = prob_blue * prob_blue)
    (h2 : prob_blue + prob_pink = 1) :
    prob_pink = 1 / 3 := 
by
  sorry

end gumball_draw_probability_l228_228581


namespace common_root_exists_l228_228688

theorem common_root_exists :
  ∃ x, (3 * x^4 + 13 * x^3 + 20 * x^2 + 17 * x + 7 = 0) ∧ (3 * x^4 + x^3 - 8 * x^2 + 11 * x - 7 = 0) → x = -7 / 3 := 
by
  sorry

end common_root_exists_l228_228688


namespace geometric_seq_problem_l228_228533

-- Definitions to capture the geometric sequence and the known condition
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ)

-- Given the condition a_1 * a_8^3 * a_15 = 243
axiom geom_seq_condition : a 1 * (a 8)^3 * a 15 = 243

theorem geometric_seq_problem 
  (h : is_geometric_sequence a) : (a 9)^3 / (a 11) = 9 :=
sorry

end geometric_seq_problem_l228_228533


namespace triangular_number_difference_30_28_l228_228049

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_number_difference_30_28 : triangular_number 30 - triangular_number 28 = 59 := 
by
  sorry

end triangular_number_difference_30_28_l228_228049


namespace am_gm_inequality_l228_228210

theorem am_gm_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_inequality_l228_228210


namespace problem_l228_228552

def is_acute_angle (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_first_quadrant (θ: ℝ) : Prop := θ > 0 ∧ θ < 90
def in_second_quadrant (θ: ℝ) : Prop := θ > 90 ∧ θ < 180

def cond1 (θ: ℝ) : Prop := θ < 90 → is_acute_angle θ
def cond2 (θ: ℝ) : Prop := in_first_quadrant θ → θ ≥ 0
def cond3 (θ: ℝ) : Prop := is_acute_angle θ → in_first_quadrant θ
def cond4 (θ θ': ℝ) : Prop := in_second_quadrant θ → in_first_quadrant θ' → θ > θ'

theorem problem :
  (¬ ∃ θ, cond1 θ) ∧ (¬ ∃ θ, cond2 θ) ∧ (∃ θ, cond3 θ) ∧ (¬ ∃ θ θ', cond4 θ θ') →
  (number_of_correct_propositions = 1) :=
  by
    sorry

end problem_l228_228552


namespace solve_for_a_l228_228272

theorem solve_for_a (x a : ℝ) (hx_pos : 0 < x) (hx_sqrt1 : x = (a+1)^2) (hx_sqrt2 : x = (a-3)^2) : a = 1 :=
by
  sorry

end solve_for_a_l228_228272


namespace part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l228_228897

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

end part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l228_228897


namespace solve_for_x_l228_228002

theorem solve_for_x (x : ℝ) (h : 24 - 6 = 3 * x + 3) : x = 5 := by
  sorry

end solve_for_x_l228_228002


namespace geometric_sequence_third_term_l228_228845

theorem geometric_sequence_third_term (a₁ a₄ : ℕ) (r : ℕ) (h₁ : a₁ = 4) (h₂ : a₄ = 256) (h₃ : a₄ = a₁ * r^3) : a₁ * r^2 = 64 := 
by
  sorry

end geometric_sequence_third_term_l228_228845


namespace zebra_crossing_distance_l228_228701

theorem zebra_crossing_distance
  (boulevard_width : ℝ)
  (distance_along_stripes : ℝ)
  (stripe_length : ℝ)
  (distance_between_stripes : ℝ) :
  boulevard_width = 60 →
  distance_along_stripes = 22 →
  stripe_length = 65 →
  distance_between_stripes = (60 * 22) / 65 →
  distance_between_stripes = 20.31 :=
by
  intros h1 h2 h3 h4
  sorry

end zebra_crossing_distance_l228_228701


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l228_228689

theorem solve_quadratic_1 (x : Real) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5) :=
by
  sorry

theorem solve_quadratic_2 (x : Real) : (x - 1)^2 = 2 * (x - 1) ↔ (x = 1 ∨ x = 3) :=
by
  sorry

theorem solve_quadratic_3 (x : Real) : (x + 1)^2 = 4 * x^2 ↔ (x = 1 ∨ x = -1 / 3) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l228_228689


namespace rectangle_perimeter_l228_228775

theorem rectangle_perimeter 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (relatively_prime : Nat.gcd (a_4 + a_7 + a_9) (a_2 + a_8 + a_6) = 1)
  (h1 : a_1 + a_2 = a_4)
  (h2 : a_1 + a_4 = a_5)
  (h3 : a_4 + a_5 = a_7)
  (h4 : a_5 + a_7 = a_9)
  (h5 : a_2 + a_4 + a_7 = a_8)
  (h6 : a_2 + a_8 = a_6)
  (h7 : a_1 + a_5 + a_9 = a_3)
  (h8 : a_3 + a_6 = a_8 + a_7) :
  2 * ((a_4 + a_7 + a_9) + (a_2 + a_8 + a_6)) = 164 := 
sorry -- proof omitted

end rectangle_perimeter_l228_228775


namespace garden_width_l228_228930

theorem garden_width (w l : ℝ) (h_length : l = 3 * w) (h_area : l * w = 675) : w = 15 :=
by
  sorry

end garden_width_l228_228930


namespace new_volume_of_cylinder_l228_228786

theorem new_volume_of_cylinder
  (r h : ℝ) -- original radius and height
  (V : ℝ) -- original volume
  (h_volume : V = π * r^2 * h) -- volume formula for the original cylinder
  (new_radius : ℝ := 3 * r) -- new radius is three times the original radius
  (new_volume : ℝ) -- new volume to be determined
  (h_original_volume : V = 10) -- original volume equals 10 cubic feet
  : new_volume = 9 * V := -- new volume should be 9 times the original volume
by
  sorry

end new_volume_of_cylinder_l228_228786


namespace ellipse_eccentricity_equilateral_triangle_l228_228242

theorem ellipse_eccentricity_equilateral_triangle
  (c a : ℝ) (h : c / a = 1 / 2) : eccentricity = 1 / 2 :=
by
  -- Proof goes here, we add sorry to skip proof content
  sorry

end ellipse_eccentricity_equilateral_triangle_l228_228242


namespace arithmetic_sequence_problem_l228_228118

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Use the given specific conditions
theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 2 * a 3 = 21) : 
  a 1 * a 4 = -11 :=
sorry

end arithmetic_sequence_problem_l228_228118


namespace certain_number_approximation_l228_228290

theorem certain_number_approximation (h1 : 2994 / 14.5 = 177) (h2 : 29.94 / x = 17.7) : x = 2.57455 := by
  sorry

end certain_number_approximation_l228_228290


namespace age_ratio_l228_228869

theorem age_ratio 
    (a m s : ℕ) 
    (h1 : m = 60) 
    (h2 : m = 3 * a) 
    (h3 : s = 40) : 
    (m + a) / s = 2 :=
by
    sorry

end age_ratio_l228_228869


namespace price_per_working_game_eq_six_l228_228713

-- Define the total number of video games
def total_games : Nat := 10

-- Define the number of non-working video games
def non_working_games : Nat := 8

-- Define the total income from selling working games
def total_earning : Nat := 12

-- Calculate the number of working video games
def working_games : Nat := total_games - non_working_games

-- Define the expected price per working game
def expected_price_per_game : Nat := 6

-- Theorem statement: Prove that the price per working game is $6
theorem price_per_working_game_eq_six :
  total_earning / working_games = expected_price_per_game :=
by sorry

end price_per_working_game_eq_six_l228_228713


namespace ratio_of_edges_l228_228097

noncomputable def cube_volume (edge : ℝ) : ℝ := edge^3

theorem ratio_of_edges 
  {a b : ℝ} 
  (h : cube_volume a / cube_volume b = 27) : 
  a / b = 3 :=
by
  sorry

end ratio_of_edges_l228_228097


namespace percentage_w_less_x_l228_228859

theorem percentage_w_less_x 
    (z : ℝ) 
    (y : ℝ) 
    (x : ℝ) 
    (w : ℝ) 
    (hy : y = 1.20 * z)
    (hx : x = 1.20 * y)
    (hw : w = 1.152 * z) 
    : (x - w) / x * 100 = 20 :=
by
  sorry

end percentage_w_less_x_l228_228859


namespace unique_k_largest_n_l228_228575

theorem unique_k_largest_n :
  ∃! k : ℤ, ∃ n : ℕ, (n > 0) ∧ (5 / 18 < n / (n + k) ∧ n / (n + k) < 9 / 17) ∧ (n = 1) :=
by
  sorry

end unique_k_largest_n_l228_228575


namespace sum_not_prime_30_l228_228741

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_not_prime_30 (p1 p2 : ℕ) (hp1 : is_prime p1) (hp2 : is_prime p2) (h : p1 + p2 = 30) : false :=
sorry

end sum_not_prime_30_l228_228741


namespace find_costs_of_A_and_B_find_price_reduction_l228_228278

-- Definitions for part 1
def cost_of_type_A_and_B (x y : ℕ) : Prop :=
  (5 * x + 3 * y = 450) ∧ (10 * x + 8 * y = 1000)

-- Part 1: Prove that x and y satisfy the cost conditions
theorem find_costs_of_A_and_B (x y : ℕ) (hx : 5 * x + 3 * y = 450) (hy : 10 * x + 8 * y = 1000) : 
  x = 60 ∧ y = 50 :=
sorry

-- Definitions for part 2
def daily_profit_condition (m : ℕ) : Prop :=
  (100 + 20 * m > 200) ∧ ((80 - m) * (100 + 20 * m) + 7000 = 10000)

-- Part 2: Prove that the price reduction m meets the profit condition
theorem find_price_reduction (m : ℕ) (hm : 100 + 20 * m > 200) (hp : (80 - m) * (100 + 20 * m) + 7000 = 10000) : 
  m = 10 :=
sorry

end find_costs_of_A_and_B_find_price_reduction_l228_228278


namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l228_228285

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b_l228_228285


namespace simplify_expression_l228_228073

theorem simplify_expression (m n : ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3^(m * n / (m - n))) : 
  ((
    (x^(2 / m) - 9 * x^(2 / n)) *
    ((x^(1 - m))^(1 / m) - 3 * (x^(1 - n))^(1 / n))
  ) / (
    (x^(1 / m) + 3 * x^(1 / n))^2 - 12 * x^((m + n) / (m * n))
  ) = (x^(1 / m) + 3 * x^(1 / n)) / x) := 
sorry

end simplify_expression_l228_228073


namespace find_a_l228_228354

noncomputable def triangle_side (a b c : ℝ) (A : ℝ) (area : ℝ) : ℝ :=
if b + c = 2 * Real.sqrt 3 ∧ A = Real.pi / 3 ∧ area = Real.sqrt 3 / 2 then
  Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
else 0

theorem find_a (b c : ℝ) (h1 : b + c = 2 * Real.sqrt 3) (h2 : Real.cos (Real.pi / 3) = 1 / 2) (area : ℝ)
  (h3 : area = Real.sqrt 3 / 2)
  (a := triangle_side (Real.sqrt 6) b c (Real.pi / 3) (Real.sqrt 3 / 2)) :
  a = Real.sqrt 6 :=
sorry

end find_a_l228_228354


namespace number_of_trees_in_yard_l228_228921

theorem number_of_trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 360 ∧ tree_distance = 12 → 
  (yard_length / tree_distance + 1 = 31) :=
by
  intros yard_length tree_distance h
  have h1 : yard_length = 360 := h.1
  have h2 : tree_distance = 12 := h.2
  sorry

end number_of_trees_in_yard_l228_228921


namespace problem_statement_l228_228508

def star (x y : Nat) : Nat :=
  match x, y with
  | 1, 1 => 4 | 1, 2 => 3 | 1, 3 => 2 | 1, 4 => 1
  | 2, 1 => 1 | 2, 2 => 4 | 2, 3 => 3 | 2, 4 => 2
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 3
  | 4, 1 => 3 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4
  | _, _ => 0  -- This line handles unexpected inputs.

theorem problem_statement : star (star 3 2) (star 2 1) = 4 := by
  sorry

end problem_statement_l228_228508


namespace quadratic_inequality_solution_l228_228847

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 - 8 * x + c > 0) ↔ (0 < c ∧ c < 16) := 
sorry

end quadratic_inequality_solution_l228_228847


namespace number_divisible_by_33_l228_228478

theorem number_divisible_by_33 (x y : ℕ) 
  (h1 : (x + y) % 3 = 2) 
  (h2 : (y - x) % 11 = 8) : 
  (27850 + 1000 * x + y) % 33 = 0 := 
sorry

end number_divisible_by_33_l228_228478


namespace flower_count_l228_228363

theorem flower_count (roses carnations : ℕ) (h₁ : roses = 5) (h₂ : carnations = 5) : roses + carnations = 10 :=
by
  sorry

end flower_count_l228_228363


namespace arithmetic_expression_value_l228_228034

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l228_228034


namespace min_expr_l228_228681

theorem min_expr (a b c d : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) (hd : Odd d) (a_pos: 0 < a) (b_pos: 0 < b) (c_pos: 0 < c) (d_pos: 0 < d)
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) = 34 := 
sorry

end min_expr_l228_228681


namespace james_music_BPM_l228_228968

theorem james_music_BPM 
  (hours_per_day : ℕ)
  (beats_per_week : ℕ)
  (days_per_week : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_per_day : ℕ)
  (total_minutes_per_week : ℕ)
  (BPM : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : beats_per_week = 168000)
  (h3 : days_per_week = 7)
  (h4 : minutes_per_hour = 60)
  (h5 : minutes_per_day = hours_per_day * minutes_per_hour)
  (h6 : total_minutes_per_week = minutes_per_day * days_per_week)
  (h7 : BPM = beats_per_week / total_minutes_per_week)
  : BPM = 200 :=
sorry

end james_music_BPM_l228_228968


namespace find_f_at_4_l228_228307

def f (n : ℕ) : ℕ := sorry -- We define the function f.

theorem find_f_at_4 : (∀ x : ℕ, f (2 * x) = 3 * x^2 + 1) → f 4 = 13 :=
by
  sorry

end find_f_at_4_l228_228307


namespace find_b_l228_228966

theorem find_b (c b : ℤ) (h : ∃ k : ℤ, (x^2 - x - 1) * (c * x - 3) = c * x^3 + b * x^2 + 3) : b = -6 :=
by
  sorry

end find_b_l228_228966


namespace min_children_l228_228937

theorem min_children (x : ℕ) : 
  (4 * x + 28 - 5 * (x - 1) < 5) ∧ (4 * x + 28 - 5 * (x - 1) ≥ 2) → (x = 29) :=
by
  sorry

end min_children_l228_228937


namespace poles_needed_to_enclose_plot_l228_228122

-- Defining the lengths of the sides
def side1 : ℕ := 15
def side2 : ℕ := 22
def side3 : ℕ := 40
def side4 : ℕ := 30
def side5 : ℕ := 18

-- Defining the distance between poles
def dist_first_three_sides : ℕ := 4
def dist_last_two_sides : ℕ := 5

-- Defining the function to calculate required poles for a side
def calculate_poles (length : ℕ) (distance : ℕ) : ℕ :=
  (length / distance) + 1

-- Total poles needed before adjustment
def total_poles_before_adjustment : ℕ :=
  calculate_poles side1 dist_first_three_sides +
  calculate_poles side2 dist_first_three_sides +
  calculate_poles side3 dist_first_three_sides +
  calculate_poles side4 dist_last_two_sides +
  calculate_poles side5 dist_last_two_sides

-- Adjustment for shared poles at corners
def total_poles : ℕ :=
  total_poles_before_adjustment - 5

-- The theorem to prove
theorem poles_needed_to_enclose_plot : total_poles = 29 := by
  sorry

end poles_needed_to_enclose_plot_l228_228122


namespace gold_copper_alloy_ratio_l228_228069

theorem gold_copper_alloy_ratio 
  (water : ℝ) 
  (G : ℝ) 
  (C : ℝ) 
  (H1 : G = 10 * water)
  (H2 : C = 6 * water)
  (H3 : 10 * G + 6 * C = 8 * (G + C)) : 
  G / C = 1 :=
by
  sorry

end gold_copper_alloy_ratio_l228_228069


namespace triangle_height_l228_228352

theorem triangle_height (s h : ℝ) 
  (area_square : s^2 = s * s) 
  (area_triangle : 1/2 * s * h = s^2) 
  (areas_equal : s^2 = s^2) : 
  h = 2 * s := 
sorry

end triangle_height_l228_228352


namespace Sarah_substitution_l228_228610

theorem Sarah_substitution :
  ∀ (f g h i j : ℤ), 
    f = 2 → g = 4 → h = 5 → i = 10 →
    (f - (g - (h * (i - j))) = 48 - 5 * j) →
    (f - g - h * i - j = -52 - j) →
    j = 25 :=
by
  intros f g h i j hfg hi hhi hmf hCm hRn
  sorry

end Sarah_substitution_l228_228610


namespace dryer_less_than_washing_machine_by_30_l228_228933

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

end dryer_less_than_washing_machine_by_30_l228_228933


namespace probability_two_red_two_blue_one_green_l228_228860

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

end probability_two_red_two_blue_one_green_l228_228860


namespace min_students_green_eyes_backpack_no_glasses_l228_228083

theorem min_students_green_eyes_backpack_no_glasses
  (S G B Gl : ℕ)
  (h_S : S = 25)
  (h_G : G = 15)
  (h_B : B = 18)
  (h_Gl : Gl = 6)
  : ∃ x, x ≥ 8 ∧ x + Gl ≤ S ∧ x ≤ min G B :=
sorry

end min_students_green_eyes_backpack_no_glasses_l228_228083


namespace no_b_satisfies_l228_228634

theorem no_b_satisfies (b : ℝ) : ¬ (2 * 1 - b * (-2) + 1 ≤ 0 ∧ 2 * (-1) - b * 2 + 1 ≤ 0) :=
by
  sorry

end no_b_satisfies_l228_228634


namespace count_three_digit_numbers_divisible_by_13_l228_228479

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l228_228479


namespace min_value_expression_l228_228139

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 48) :
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2 ≥ 144 :=
sorry

end min_value_expression_l228_228139


namespace product_is_correct_l228_228773

-- Define the variables and conditions
variables {a b c d : ℚ}

-- State the conditions
def conditions (a b c d : ℚ) :=
  3 * a + 2 * b + 4 * c + 6 * d = 36 ∧
  4 * (d + c) = b ∧
  4 * b + 2 * c = a ∧
  c - 2 = d

-- The theorem statement
theorem product_is_correct (a b c d : ℚ) (h : conditions a b c d) :
  a * b * c * d = -315 / 32 :=
sorry

end product_is_correct_l228_228773


namespace red_balls_count_l228_228468

theorem red_balls_count (R W : ℕ) (h1 : R / W = 4 / 5) (h2 : W = 20) : R = 16 := sorry

end red_balls_count_l228_228468


namespace age_of_B_l228_228875

theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 37) : B = 14 :=
by sorry

end age_of_B_l228_228875


namespace total_possible_rankings_l228_228664

-- Define the players
inductive Player
| P | Q | R | S

-- Define the tournament results
inductive Result
| win | lose

-- Define Saturday's match outcomes
structure SaturdayOutcome :=
(P_vs_Q: Result)
(R_vs_S: Result)

-- Function to compute the number of possible tournament ranking sequences
noncomputable def countTournamentSequences : Nat :=
  let saturdayOutcomes: List SaturdayOutcome :=
    [ {P_vs_Q := Result.win, R_vs_S := Result.win}
    , {P_vs_Q := Result.win, R_vs_S := Result.lose}
    , {P_vs_Q := Result.lose, R_vs_S := Result.win}
    , {P_vs_Q := Result.lose, R_vs_S := Result.lose}
    ]
  let sundayPermutations (outcome : SaturdayOutcome) : Nat :=
    2 * 2  -- 2 permutations for 1st and 2nd places * 2 permutations for 3rd and 4th places per each outcome
  saturdayOutcomes.foldl (fun acc outcome => acc + sundayPermutations outcome) 0

-- Define the theorem to prove the total number of permutations
theorem total_possible_rankings : countTournamentSequences = 8 :=
by
  -- Proof steps here (proof omitted)
  sorry

end total_possible_rankings_l228_228664


namespace longest_side_of_triangle_l228_228733

variable (x y : ℝ)

def side1 := 10
def side2 := 2*y + 3
def side3 := 3*x + 2

theorem longest_side_of_triangle
  (h_perimeter : side1 + side2 + side3 = 45)
  (h_side2_pos : side2 > 0)
  (h_side3_pos : side3 > 0) :
  side3 = 32 :=
sorry

end longest_side_of_triangle_l228_228733


namespace solve_inequalities_l228_228643

theorem solve_inequalities :
  {x : ℝ // 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 8} = {x : ℝ // 3 < x ∧ x < 4} :=
sorry

end solve_inequalities_l228_228643


namespace min_value_of_expression_min_value_achieved_l228_228367

noncomputable def f (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_of_expression : ∀ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6452.25 :=
by sorry

end min_value_of_expression_min_value_achieved_l228_228367


namespace problem_solution_l228_228588

noncomputable def a (n : ℕ) : ℕ := 2 * n - 3

noncomputable def b (n : ℕ) : ℕ := 2 ^ n

noncomputable def c (n : ℕ) : ℕ := a n * b n

noncomputable def sum_c (n : ℕ) : ℕ :=
  (2 * n - 5) * 2 ^ (n + 1) + 10

theorem problem_solution :
  ∀ n : ℕ, n > 0 →
  (S_n = 2 * (b n - 1)) ∧
  (a 2 = b 1 - 1) ∧
  (a 5 = b 3 - 1)
  →
  (∀ n, a n = 2 * n - 3) ∧
  (∀ n, b n = 2 ^ n) ∧
  (sum_c n = (2 * n - 5) * 2 ^ (n + 1) + 10) :=
by
  intros n hn h
  sorry


end problem_solution_l228_228588


namespace quadratic_eq_real_roots_l228_228153

theorem quadratic_eq_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9 / 8 :=
by
  sorry

end quadratic_eq_real_roots_l228_228153


namespace solution_to_system_l228_228541

def system_of_equations (x y : ℝ) : Prop := (x^2 - 9 * y^2 = 36) ∧ (3 * x + y = 6)

theorem solution_to_system : 
  {p : ℝ × ℝ | system_of_equations p.1 p.2} = { (12 / 5, -6 / 5), (3, -3) } := 
by sorry

end solution_to_system_l228_228541


namespace space_needed_between_apple_trees_l228_228182

-- Definitions based on conditions
def apple_tree_width : ℕ := 10
def peach_tree_width : ℕ := 12
def space_between_peach_trees : ℕ := 15
def total_space : ℕ := 71
def number_of_apple_trees : ℕ := 2
def number_of_peach_trees : ℕ := 2

-- Lean 4 theorem statement
theorem space_needed_between_apple_trees :
  (total_space 
   - (number_of_peach_trees * peach_tree_width + space_between_peach_trees))
  - (number_of_apple_trees * apple_tree_width) 
  = 12 := by
  sorry

end space_needed_between_apple_trees_l228_228182


namespace average_age_decrease_l228_228986

theorem average_age_decrease (N T : ℕ) (h₁ : (T : ℝ) / N - 3 = (T - 30 : ℝ) / N) : N = 10 :=
sorry

end average_age_decrease_l228_228986


namespace find_minimum_value_l228_228050

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + a * |x - 1| + 1

-- The statement of the proof problem
theorem find_minimum_value (a : ℝ) (h : a ≥ 0) :
  (a = 0 → ∀ x, f x a ≥ 1 ∧ ∃ x, f x a = 1) ∧
  ((0 < a ∧ a < 2) → ∀ x, f x a ≥ -a^2 / 4 + a + 1 ∧ ∃ x, f x a = -a^2 / 4 + a + 1) ∧
  (a ≥ 2 → ∀ x, f x a ≥ 2 ∧ ∃ x, f x a = 2) := 
by
  sorry

end find_minimum_value_l228_228050


namespace adult_tickets_l228_228535

theorem adult_tickets (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : A = 40 :=
by {
  -- Proof omitted
  sorry
}

end adult_tickets_l228_228535


namespace complement_inter_section_l228_228095

-- Define the sets M and N
def M : Set ℝ := { x | x^2 - 2*x - 3 >= 0 }
def N : Set ℝ := { x | abs (x - 2) <= 1 }

-- Define the complement of M in ℝ
def compl_M : Set ℝ := { x | -1 < x ∧ x < 3 }

-- Define the expected result set
def expected_set : Set ℝ := { x | 1 <= x ∧ x < 3 }

-- State the theorem to prove
theorem complement_inter_section : compl_M ∩ N = expected_set := by
  sorry

end complement_inter_section_l228_228095


namespace alfred_saving_goal_l228_228578

theorem alfred_saving_goal (leftover : ℝ) (monthly_saving : ℝ) (months : ℕ) :
  leftover = 100 → monthly_saving = 75 → months = 12 → leftover + monthly_saving * months = 1000 :=
by
  sorry

end alfred_saving_goal_l228_228578


namespace fred_initial_sheets_l228_228319

theorem fred_initial_sheets (X : ℕ) (h1 : X + 307 - 156 = 363) : X = 212 :=
by
  sorry

end fred_initial_sheets_l228_228319


namespace find_EQ_l228_228353

open Real

noncomputable def Trapezoid_EFGH (EF FG GH HE EQ QF : ℝ) : Prop :=
  EF = 110 ∧
  FG = 60 ∧
  GH = 23 ∧
  HE = 75 ∧
  EQ + QF = EF ∧
  EQ = 250 / 3

theorem find_EQ (EF FG GH HE EQ QF : ℝ) (h : Trapezoid_EFGH EF FG GH HE EQ QF) :
  EQ = 250 / 3 :=
by
  sorry

end find_EQ_l228_228353


namespace george_collected_50_marbles_l228_228523

theorem george_collected_50_marbles (w y g r total : ℕ)
  (hw : w = total / 2)
  (hy : y = 12)
  (hg : g = y / 2)
  (hr : r = 7)
  (htotal : total = w + y + g + r) :
  total = 50 := by
  sorry

end george_collected_50_marbles_l228_228523


namespace truth_probability_l228_228289

variables (P_A P_B P_AB : ℝ)

theorem truth_probability (h1 : P_B = 0.60) (h2 : P_AB = 0.48) : P_A = 0.80 :=
by
  have h3 : P_AB = P_A * P_B := sorry  -- Placeholder for the rule: P(A and B) = P(A) * P(B)
  rw [h2, h1] at h3
  sorry

end truth_probability_l228_228289


namespace solution_value_of_a_l228_228477

noncomputable def verify_a (a : ℚ) (A : Set ℚ) : Prop :=
  A = {a - 2, 2 * a^2 + 5 * a, 12} ∧ -3 ∈ A

theorem solution_value_of_a (a : ℚ) (A : Set ℚ) (h : verify_a a A) : a = -3 / 2 := by
  sorry

end solution_value_of_a_l228_228477


namespace share_of_b_l228_228070

theorem share_of_b (x : ℝ) (h : 3300 / ((7/2) * x) = 2 / 7) :  
   let total_profit := 3300
   let B_share := (x / ((7/2) * x)) * total_profit
   B_share = 942.86 :=
by sorry

end share_of_b_l228_228070


namespace last_four_digits_of_5_pow_9000_l228_228505

theorem last_four_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 1250]) : 
  5^9000 ≡ 1 [MOD 1250] :=
sorry

end last_four_digits_of_5_pow_9000_l228_228505


namespace increasing_function_range_a_l228_228671

theorem increasing_function_range_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = if x > 1 then a^x else (4 - a/2)*x + 2) ∧
  (∀ x y, x < y → f x ≤ f y) →
  4 ≤ a ∧ a < 8 :=
by
  sorry

end increasing_function_range_a_l228_228671


namespace point_B_coordinates_l228_228549

-- Defining the vector a
def vec_a : ℝ × ℝ := (1, 0)

-- Defining the point A
def A : ℝ × ℝ := (4, 4)

-- Definition of the line y = 2x
def on_line (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Defining a vector as being parallel to another vector
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean statement for the proof
theorem point_B_coordinates (B : ℝ × ℝ) (h1 : on_line B) (h2 : parallel (B.1 - 4, B.2 - 4) vec_a) :
  B = (2, 4) :=
sorry

end point_B_coordinates_l228_228549


namespace solve_for_x_l228_228254

theorem solve_for_x (x : ℝ) (h : 5 / (4 + 1 / x) = 1) : x = 1 :=
by
  sorry

end solve_for_x_l228_228254


namespace books_sold_in_february_l228_228784

theorem books_sold_in_february (F : ℕ) 
  (h_avg : (15 + F + 17) / 3 = 16): 
  F = 16 := 
by 
  sorry

end books_sold_in_february_l228_228784


namespace min_max_sum_eq_one_l228_228006

theorem min_max_sum_eq_one 
  (x : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_eq_one : (x 1 + x 2 + x 3 + x 4 + x 5) = 1) :
  (min (max (x 1 + x 2) (max (x 2 + x 3) (max (x 3 + x 4) (x 4 + x 5)))) = (1 / 3)) :=
by
  sorry

end min_max_sum_eq_one_l228_228006


namespace monotone_increasing_range_of_a_l228_228901

noncomputable def f (a x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_increasing_range_of_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Icc (-1 / 3 : ℝ) (1 / 3 : ℝ)) :=
sorry

end monotone_increasing_range_of_a_l228_228901


namespace gunny_bag_can_hold_packets_l228_228675

theorem gunny_bag_can_hold_packets :
  let ton_to_kg := 1000
  let max_capacity_tons := 13
  let pound_to_kg := 0.453592
  let ounce_to_g := 28.3495
  let kilo_to_g := 1000
  let wheat_packet_pounds := 16
  let wheat_packet_ounces := 4
  let max_capacity_kg := max_capacity_tons * ton_to_kg
  let wheat_packet_kg := wheat_packet_pounds * pound_to_kg + (wheat_packet_ounces * ounce_to_g) / kilo_to_g
  max_capacity_kg / wheat_packet_kg >= 1763 := 
by
  sorry

end gunny_bag_can_hold_packets_l228_228675


namespace proof_of_expression_l228_228147

theorem proof_of_expression (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 :=
by {
  sorry
}

end proof_of_expression_l228_228147


namespace thirty_percent_of_x_l228_228218

noncomputable def x : ℝ := 160 / 0.40

theorem thirty_percent_of_x (h : 0.40 * x = 160) : 0.30 * x = 120 :=
sorry

end thirty_percent_of_x_l228_228218


namespace average_of_first_12_l228_228832

theorem average_of_first_12 (avg25 : ℝ) (avg12 : ℝ) (avg_last12 : ℝ) (result_13th : ℝ) : 
  (avg25 = 18) → (avg_last12 = 17) → (result_13th = 78) → 
  25 * avg25 = (12 * avg12) + result_13th + (12 * avg_last12) → avg12 = 14 :=
by 
  sorry

end average_of_first_12_l228_228832


namespace inequality_sum_l228_228936

theorem inequality_sum {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) : a + c > b + d :=
by
  sorry

end inequality_sum_l228_228936


namespace phones_left_is_7500_l228_228086

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_l228_228086


namespace distance_to_water_source_l228_228110

theorem distance_to_water_source (d : ℝ) :
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 5)) → 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_water_source_l228_228110


namespace pandas_bamboo_consumption_l228_228026

/-- Given:
  1. An adult panda can eat 138 pounds of bamboo each day.
  2. A baby panda can eat 50 pounds of bamboo a day.
Prove: the total pounds of bamboo eaten by both pandas in a week is 1316 pounds. -/
theorem pandas_bamboo_consumption :
  let adult_daily_bamboo := 138
  let baby_daily_bamboo := 50
  let days_in_week := 7
  (adult_daily_bamboo * days_in_week) + (baby_daily_bamboo * days_in_week) = 1316 := by
  sorry

end pandas_bamboo_consumption_l228_228026


namespace scientific_notation_equivalent_l228_228534

theorem scientific_notation_equivalent : ∃ a n, (3120000 : ℝ) = a * 10^n ∧ a = 3.12 ∧ n = 6 :=
by
  exists 3.12
  exists 6
  sorry

end scientific_notation_equivalent_l228_228534


namespace payment_is_variable_l228_228138

variable (x y : ℕ)

def price_of_pen : ℕ := 3

theorem payment_is_variable (x y : ℕ) (h : y = price_of_pen * x) : 
  (price_of_pen = 3) ∧ (∃ n : ℕ, y = 3 * n) :=
by 
  sorry

end payment_is_variable_l228_228138


namespace tshirts_equation_l228_228715

theorem tshirts_equation (x : ℝ) 
    (hx : x > 0)
    (march_cost : ℝ := 120000)
    (april_cost : ℝ := 187500)
    (april_increase : ℝ := 1.4)
    (cost_increase : ℝ := 5) :
    120000 / x + 5 = 187500 / (1.4 * x) :=
by 
  sorry

end tshirts_equation_l228_228715


namespace minimum_value_f_minimum_value_abc_l228_228504

noncomputable def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 3)

theorem minimum_value_f : ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f x ≥ m := 
by
  let m := 1
  existsi m
  sorry

theorem minimum_value_abc (a b c : ℝ) (h : a + 2 * b + 3 * c = 1) : ∃ n : ℝ, n = 1/14 ∧ a^2 + b^2 + c^2 ≥ n :=
by
  let n := 1 / 14
  existsi n
  sorry

end minimum_value_f_minimum_value_abc_l228_228504


namespace adam_apples_l228_228096

theorem adam_apples (x : ℕ) 
  (h1 : 15 + 75 * x = 240) : x = 3 :=
sorry

end adam_apples_l228_228096


namespace best_fit_model_l228_228368

-- Define the coefficients of determination for each model
noncomputable def R2_Model1 : ℝ := 0.75
noncomputable def R2_Model2 : ℝ := 0.90
noncomputable def R2_Model3 : ℝ := 0.45
noncomputable def R2_Model4 : ℝ := 0.65

-- State the theorem 
theorem best_fit_model : 
  R2_Model2 ≥ R2_Model1 ∧ 
  R2_Model2 ≥ R2_Model3 ∧ 
  R2_Model2 ≥ R2_Model4 :=
by
  sorry

end best_fit_model_l228_228368


namespace find_angle_C_l228_228286

-- Definitions based on conditions
variables (α β γ : ℝ) -- Angles of the triangle

-- Condition: Angles between the altitude and the angle bisector at vertices A and B are equal
-- This implies α = β
def angles_equal (α β : ℝ) : Prop :=
  α = β

-- Condition: Sum of the angles in a triangle is 180 degrees
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Condition: Angle at vertex C is greater than angles at vertices A and B
def c_greater_than_a_and_b (α γ : ℝ) : Prop :=
  γ > α

-- The proof problem: Prove γ = 120 degrees given the conditions
theorem find_angle_C (α β γ : ℝ) (h1 : angles_equal α β) (h2 : angles_sum_to_180 α β γ) (h3 : c_greater_than_a_and_b α γ) : γ = 120 :=
by
  sorry

end find_angle_C_l228_228286


namespace complex_num_z_imaginary_square_l228_228442

theorem complex_num_z_imaginary_square (z : ℂ) (h1 : z.im ≠ 0) (h2 : z.re = 0) (h3 : ((z + 1) ^ 2).re = 0) :
  z = Complex.I ∨ z = -Complex.I :=
by
  sorry

end complex_num_z_imaginary_square_l228_228442


namespace number_of_positive_divisors_of_60_l228_228470

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l228_228470


namespace problem_statement_l228_228652

theorem problem_statement (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 :=
by sorry

end problem_statement_l228_228652


namespace sn_geq_mnplus1_l228_228311

namespace Polysticks

def n_stick (n : ℕ) : Type := sorry -- formalize the definition of n-stick
def n_mino (n : ℕ) : Type := sorry -- formalize the definition of n-mino

def S (n : ℕ) : ℕ := sorry -- define the number of n-sticks
def M (n : ℕ) : ℕ := sorry -- define the number of n-minos

theorem sn_geq_mnplus1 (n : ℕ) : S n ≥ M (n+1) := sorry

end Polysticks

end sn_geq_mnplus1_l228_228311


namespace arrange_books_l228_228055

noncomputable def numberOfArrangements : Nat :=
  4 * 3 * 6 * (Nat.factorial 9)

theorem arrange_books :
  numberOfArrangements = 26210880 := by
  sorry

end arrange_books_l228_228055


namespace solve_for_z_l228_228292

variable (x y z : ℝ)

theorem solve_for_z (h : 1 / x - 1 / y = 1 / z) : z = x * y / (y - x) := 
sorry

end solve_for_z_l228_228292


namespace theta_in_third_quadrant_l228_228357

-- Define the mathematical conditions
variable (θ : ℝ)
axiom cos_theta_neg : Real.cos θ < 0
axiom cos_minus_sin_eq_sqrt : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)

-- Prove that θ is in the third quadrant
theorem theta_in_third_quadrant : 
  (∀ θ : ℝ, Real.cos θ < 0 → Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) → 
    Real.sin θ < 0 ∧ Real.cos θ < 0) :=
by sorry

end theta_in_third_quadrant_l228_228357


namespace lcm_48_147_l228_228722

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := sorry

end lcm_48_147_l228_228722


namespace problem1_problem2_l228_228593

def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 7
def S (x : ℝ) (k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

theorem problem1 (k : ℝ) : (∀ x, S x k → A x) → k ≤ 4 :=
by
  sorry

theorem problem2 (k : ℝ) : (∀ x, ¬(A x ∧ S x k)) → k < 2 ∨ k > 6 :=
by
  sorry

end problem1_problem2_l228_228593


namespace exterior_angle_of_regular_pentagon_l228_228078

theorem exterior_angle_of_regular_pentagon : 
  (360 / 5) = 72 := by
  sorry

end exterior_angle_of_regular_pentagon_l228_228078


namespace find_f_l228_228540

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : ∀ x : ℝ, f x = x + 1 :=
by
  sorry

end find_f_l228_228540


namespace computation_of_expression_l228_228430

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l228_228430


namespace volume_of_fuel_A_l228_228340

variables (V_A V_B : ℝ)

def condition1 := V_A + V_B = 212
def condition2 := 0.12 * V_A + 0.16 * V_B = 30

theorem volume_of_fuel_A :
  condition1 V_A V_B → condition2 V_A V_B → V_A = 98 :=
by
  intros h1 h2
  sorry

end volume_of_fuel_A_l228_228340


namespace perimeter_of_triangle_LMN_l228_228619

variable (K L M N : Type)
variables [MetricSpace K]
variables [MetricSpace L]
variables [MetricSpace M]
variables [MetricSpace N]
variables (KL LN MN : ℝ)
variables (perimeter_LMN : ℝ)

-- Given conditions
axiom KL_eq_24 : KL = 24
axiom LN_eq_24 : LN = 24
axiom MN_eq_9  : MN = 9

-- Prove the perimeter is 57
theorem perimeter_of_triangle_LMN : perimeter_LMN = KL + LN + MN → perimeter_LMN = 57 :=
by sorry

end perimeter_of_triangle_LMN_l228_228619


namespace max_halls_visitable_max_triangles_in_chain_l228_228833

-- Definition of the problem conditions
def castle_side_length : ℝ := 100
def num_halls : ℕ := 100
def hall_side_length : ℝ := 10
def max_visitable_halls : ℕ := 91

-- Theorem statements
theorem max_halls_visitable (S : ℝ) (n : ℕ) (H : ℝ) :
  S = 100 ∧ n = 100 ∧ H = 10 → max_visitable_halls = 91 :=
by sorry

-- Definitions for subdividing an equilateral triangle and the chain of triangles
def side_divisions (k : ℕ) : ℕ := k
def total_smaller_triangles (k : ℕ) : ℕ := k^2
def max_chain_length (k : ℕ) : ℕ := k^2 - k + 1

-- Theorem statements
theorem max_triangles_in_chain (k : ℕ) :
  max_chain_length k = k^2 - k + 1 :=
by sorry

end max_halls_visitable_max_triangles_in_chain_l228_228833


namespace problem_solution_A_problem_solution_C_l228_228051

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l228_228051


namespace tan_diff_eq_sqrt_three_l228_228276

open Real

theorem tan_diff_eq_sqrt_three (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : cos α * cos β = 1 / 6) (h5 : sin α * sin β = 1 / 3) : 
  tan (β - α) = sqrt 3 := by
  sorry

end tan_diff_eq_sqrt_three_l228_228276


namespace sum_of_specific_terms_in_arithmetic_sequence_l228_228054

theorem sum_of_specific_terms_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_S11 : S 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end sum_of_specific_terms_in_arithmetic_sequence_l228_228054


namespace problem_l228_228582

theorem problem (p q : ℝ) (h : 5 * p^2 - 20 * p + 15 = 0 ∧ 5 * q^2 - 20 * q + 15 = 0) : (p * q - 3)^2 = 0 := 
sorry

end problem_l228_228582


namespace liquid_flow_problem_l228_228484

variables (x y z : ℝ)

theorem liquid_flow_problem 
    (h1 : 1/x + 1/y + 1/z = 1/6) 
    (h2 : y = 0.75 * x) 
    (h3 : z = y + 10) : 
    x = 56/3 ∧ y = 14 ∧ z = 24 :=
sorry

end liquid_flow_problem_l228_228484


namespace fourth_number_in_sequence_l228_228885

noncomputable def fifth_number_in_sequence : ℕ := 78
noncomputable def increment : ℕ := 11
noncomputable def final_number_in_sequence : ℕ := 89

theorem fourth_number_in_sequence : (fifth_number_in_sequence - increment) = 67 := by
  sorry

end fourth_number_in_sequence_l228_228885


namespace round_trip_time_l228_228329

theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : 
  boat_speed = 8 → stream_speed = 2 → distance = 210 → 
  ((distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed))) = 56 :=
by
  intros hb hs hd
  sorry

end round_trip_time_l228_228329


namespace payroll_amount_l228_228912

theorem payroll_amount (P : ℝ) 
  (h1 : P > 500000) 
  (h2 : 0.004 * (P - 500000) - 1000 = 600) :
  P = 900000 :=
by
  sorry

end payroll_amount_l228_228912


namespace ratio_of_ages_l228_228953

-- Define the conditions and the main proof goal
theorem ratio_of_ages (R J : ℕ) (Tim_age : ℕ) (h1 : Tim_age = 5) (h2 : J = R + 2) (h3 : J = Tim_age + 12) :
  R / Tim_age = 3 := 
by
  sorry

end ratio_of_ages_l228_228953


namespace total_students_appeared_l228_228154

variable (T : ℝ) -- total number of students

def fraction_failed := 0.65
def num_failed := 546

theorem total_students_appeared :
  0.65 * T = 546 → T = 840 :=
by
  intro h
  sorry

end total_students_appeared_l228_228154


namespace hyperbola_m_value_l228_228267

noncomputable def m_value : ℝ := 2 * (Real.sqrt 2 - 1)

theorem hyperbola_m_value (a : ℝ) (m : ℝ) (AF_2 AF_1 BF_2 BF_1 : ℝ)
  (h1 : a = 1)
  (h2 : AF_2 = m)
  (h3 : AF_1 = 2 + AF_2)
  (h4 : AF_1 = m + BF_2)
  (h5 : BF_2 = 2)
  (h6 : BF_1 = 4)
  (h7 : BF_1 = Real.sqrt 2 * AF_1) :
  m = m_value :=
by
  sorry

end hyperbola_m_value_l228_228267


namespace factorize_difference_of_squares_l228_228236

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) :=
sorry

end factorize_difference_of_squares_l228_228236


namespace two_numbers_product_l228_228781

theorem two_numbers_product (x y : ℕ) 
  (h1 : x + y = 90) 
  (h2 : x - y = 10) : x * y = 2000 :=
by
  sorry

end two_numbers_product_l228_228781


namespace extreme_points_inequality_l228_228987

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / 2 * x ^ 2 - x

theorem extreme_points_inequality 
  (a : ℝ)
  (ha : 0 < a ∧ a < 1)
  (alpha beta : ℝ)
  (h_eq_alpha : alpha = -Real.sqrt (1 - a))
  (h_eq_beta : beta = Real.sqrt (1 - a))
  (h_order : alpha < beta) :
  (f a beta / alpha) < (1 / 2) :=
sorry

end extreme_points_inequality_l228_228987


namespace preparation_start_month_l228_228033

variable (ExamMonth : ℕ)
def start_month (ExamMonth : ℕ) : ℕ :=
  (ExamMonth - 5) % 12

theorem preparation_start_month :
  ∀ (ExamMonth : ℕ), start_month ExamMonth = (ExamMonth - 5) % 12 :=
by
  sorry

end preparation_start_month_l228_228033


namespace find_principal_sum_l228_228613

theorem find_principal_sum (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) 
  (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : P = 8925 := 
by
  sorry

end find_principal_sum_l228_228613


namespace min_weighings_to_find_counterfeit_l228_228433

-- Definition of the problem conditions.
def coin_is_genuine (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins m = coins (Fin.mk 0 sorry)

def counterfit_coin_is_lighter (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins n < coins m

-- The theorem statement
theorem min_weighings_to_find_counterfeit :
  (∀ coins : Fin 10 → ℝ, ∃ n : Fin 10, coin_is_genuine coins n ∧ counterfit_coin_is_lighter coins n → ∃ min_weighings : ℕ, min_weighings = 3) :=
by {
  sorry
}

end min_weighings_to_find_counterfeit_l228_228433


namespace simplify_expression_solve_inequality_system_l228_228577

-- Problem 1
theorem simplify_expression (m n : ℝ) (h1 : 3 * m - 2 * n ≠ 0) (h2 : 3 * m + 2 * n ≠ 0) (h3 : 9 * m ^ 2 - 4 * n ^ 2 ≠ 0) :
  ((1 / (3 * m - 2 * n) - 1 / (3 * m + 2 * n)) / (m * n / (9 * m ^ 2 - 4 * n ^ 2))) = (4 / m) :=
sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) (h1 : 3 * x + 10 > 5 * x - 2 * (5 - x)) (h2 : (x + 3) / 5 > 1 - x) :
  1 / 3 < x ∧ x < 5 :=
sorry

end simplify_expression_solve_inequality_system_l228_228577


namespace students_in_class_l228_228467

variable (G B : ℕ)

def total_plants (G B : ℕ) : ℕ := 3 * G + B / 3

theorem students_in_class (h1 : total_plants G B = 24) (h2 : B / 3 = 6) : G + B = 24 :=
by
  sorry

end students_in_class_l228_228467


namespace cos_alpha_value_l228_228040

-- Definitions for conditions and theorem statement

def condition_1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def condition_2 (α : ℝ) : Prop :=
  Real.cos (Real.pi / 3 + α) = 1 / 3

theorem cos_alpha_value (α : ℝ) (h1 : condition_1 α) (h2 : condition_2 α) :
  Real.cos α = (1 + 2 * Real.sqrt 6) / 6 := sorry

end cos_alpha_value_l228_228040


namespace A_rotated_l228_228958

-- Define initial coordinates of point A
def A_initial : ℝ × ℝ := (1, 2)

-- Define the transformation for a 180-degree clockwise rotation around the origin
def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- The Lean statement to prove the coordinates after the rotation
theorem A_rotated : rotate_180_deg A_initial = (-1, -2) :=
by
  sorry

end A_rotated_l228_228958


namespace not_perfect_square_7_pow_2025_all_others_perfect_squares_l228_228939

theorem not_perfect_square_7_pow_2025 :
  ¬ (∃ x : ℕ, x^2 = 7^2025) :=
sorry

theorem all_others_perfect_squares :
  (∃ x : ℕ, x^2 = 6^2024) ∧
  (∃ x : ℕ, x^2 = 8^2026) ∧
  (∃ x : ℕ, x^2 = 9^2027) ∧
  (∃ x : ℕ, x^2 = 10^2028) :=
sorry

end not_perfect_square_7_pow_2025_all_others_perfect_squares_l228_228939


namespace four_times_remaining_marbles_l228_228076

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l228_228076


namespace eggs_collection_l228_228137

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l228_228137


namespace cat_mouse_positions_after_247_moves_l228_228168

-- Definitions for Positions:
inductive Position
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| TopMiddle
| RightMiddle
| BottomMiddle
| LeftMiddle

open Position

-- Function to calculate position of the cat
def cat_position (n : ℕ) : Position :=
  match n % 4 with
  | 0 => TopLeft
  | 1 => TopRight
  | 2 => BottomRight
  | 3 => BottomLeft
  | _ => TopLeft   -- This case is impossible since n % 4 is in {0, 1, 2, 3}

-- Function to calculate position of the mouse
def mouse_position (n : ℕ) : Position :=
  match n % 8 with
  | 0 => TopMiddle
  | 1 => TopRight
  | 2 => RightMiddle
  | 3 => BottomRight
  | 4 => BottomMiddle
  | 5 => BottomLeft
  | 6 => LeftMiddle
  | 7 => TopLeft
  | _ => TopMiddle -- This case is impossible since n % 8 is in {0, 1, .., 7}

-- Target theorem
theorem cat_mouse_positions_after_247_moves :
  cat_position 247 = BottomRight ∧ mouse_position 247 = LeftMiddle :=
by
  sorry

end cat_mouse_positions_after_247_moves_l228_228168


namespace complement_intersection_eq_l228_228372

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

-- Definition of complement of A in U
def complement_U_A : Set ℕ := U \ A

-- The main statement to prove
theorem complement_intersection_eq :
  (complement_U_A ∩ B) = {1, 3, 7} :=
by sorry

end complement_intersection_eq_l228_228372


namespace quadratic_root_range_quadratic_product_of_roots_l228_228238

-- Problem (1): Prove the range of m.
theorem quadratic_root_range (m : ℝ) :
  (∀ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 → x1 ≠ x2) ↔ m < 1 := 
sorry

-- Problem (2): Prove the existence of m such that x1 * x2 = 0.
theorem quadratic_product_of_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + 2 * (m - 1) * x + m^2 - 1 = 0 ∧ x1 * x2 = 0) ↔ m = -1 := 
sorry

end quadratic_root_range_quadratic_product_of_roots_l228_228238


namespace total_votes_400_l228_228461

theorem total_votes_400 
    (V : ℝ)
    (h1 : ∃ (c1_votes c2_votes : ℝ), c1_votes = 0.70 * V ∧ c2_votes = 0.30 * V)
    (h2 : ∃ (majority : ℝ), majority = 160)
    (h3 : ∀ (c1_votes c2_votes majority : ℝ), c1_votes - c2_votes = majority) : V = 400 :=
by 
  sorry

end total_votes_400_l228_228461


namespace no_nat_solutions_m2_eq_n2_plus_2014_l228_228852

theorem no_nat_solutions_m2_eq_n2_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by
  sorry

end no_nat_solutions_m2_eq_n2_plus_2014_l228_228852


namespace tangent_ellipse_hyperbola_l228_228739

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ↔ x^2 - n * (y - 1)^2 = 4) →
  n = 9 / 5 :=
by sorry

end tangent_ellipse_hyperbola_l228_228739


namespace boys_from_pine_l228_228530

/-- 
Given the following conditions:
1. There are 150 students at the camp.
2. There are 90 boys at the camp.
3. There are 60 girls at the camp.
4. There are 70 students from Maple High School.
5. There are 80 students from Pine High School.
6. There are 20 girls from Oak High School.
7. There are 30 girls from Maple High School.

Prove that the number of boys from Pine High School is 70.
--/
theorem boys_from_pine (total_students boys girls maple_high pine_high oak_girls maple_girls : ℕ)
  (H1 : total_students = 150)
  (H2 : boys = 90)
  (H3 : girls = 60)
  (H4 : maple_high = 70)
  (H5 : pine_high = 80)
  (H6 : oak_girls = 20)
  (H7 : maple_girls = 30) : 
  ∃ pine_boys : ℕ, pine_boys = 70 :=
by
  -- Proof goes here
  sorry

end boys_from_pine_l228_228530


namespace find_principal_amount_l228_228119

-- Definitions of the conditions
def rate_of_interest : ℝ := 0.20
def time_period : ℕ := 2
def interest_difference : ℝ := 144

-- Definitions for Simple Interest (SI) and Compound Interest (CI)
def simple_interest (P : ℝ) : ℝ := P * rate_of_interest * time_period
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_of_interest)^time_period - P

-- Statement to prove the principal amount given the conditions
theorem find_principal_amount (P : ℝ) : 
    compound_interest P - simple_interest P = interest_difference → P = 3600 := by
    sorry

end find_principal_amount_l228_228119


namespace trivia_team_members_l228_228348

theorem trivia_team_members (x : ℕ) (h : 3 * (x - 6) = 27) : x = 15 := 
by
  sorry

end trivia_team_members_l228_228348


namespace sum_of_fractions_l228_228374

-- Definitions (Conditions)
def frac1 : ℚ := 5 / 13
def frac2 : ℚ := 9 / 11

-- Theorem (Equivalent Proof Problem)
theorem sum_of_fractions : frac1 + frac2 = 172 / 143 := 
by
  -- Proof skipped
  sorry

end sum_of_fractions_l228_228374


namespace gcf_360_270_lcm_360_270_l228_228428

def prime_factors_360 := [(2, 3), (3, 2), (5, 1)]
def prime_factors_270 := [(2, 1), (3, 3), (5, 1)]

def GCF (a b: ℕ) : ℕ := 2^1 * 3^2 * 5^1
def LCM (a b: ℕ) : ℕ := 2^3 * 3^3 * 5^1

-- Theorem: The GCF of 360 and 270 is 90
theorem gcf_360_270 : GCF 360 270 = 90 := by
  sorry

-- Theorem: The LCM of 360 and 270 is 1080
theorem lcm_360_270 : LCM 360 270 = 1080 := by
  sorry

end gcf_360_270_lcm_360_270_l228_228428


namespace candy_bar_split_l228_228745
noncomputable def split (total: ℝ) (people: ℝ): ℝ := total / people

theorem candy_bar_split: split 5.0 3.0 = 1.67 :=
by
  sorry

end candy_bar_split_l228_228745


namespace general_term_formula_l228_228605

theorem general_term_formula (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → (n+1) * a (n+1) - n * a n^2 + (n+1) * a n * a (n+1) - n * a n = 0) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by
  sorry

end general_term_formula_l228_228605


namespace beth_sold_l228_228564

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l228_228564


namespace inequality_solution_l228_228977

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 9 / (x + 6) ≥ 2) ↔ (x ∈ Set.Ico (-6 : ℝ) (-3) ∪ Set.Ioc (-2) 3) := 
sorry

end inequality_solution_l228_228977


namespace evaluate_expression_l228_228379

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)
variable (h4 : ∀ x, g (g_inv x) = x)
variable (h5 : ∀ x, g_inv (g x) = x)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 :=
by
  -- The proof is omitted
  sorry

end evaluate_expression_l228_228379


namespace probability_of_second_ball_white_is_correct_l228_228009

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct_l228_228009


namespace task_pages_l228_228599

theorem task_pages (A B T : ℕ) (hB : B = A + 5) (hTogether : (A + B) * 18 = T)
  (hAlone : A * 60 = T) : T = 225 :=
by
  sorry

end task_pages_l228_228599


namespace jelly_bean_problem_l228_228082

theorem jelly_bean_problem 
  (x y : ℕ) 
  (h1 : x + y = 1200) 
  (h2 : x = 3 * y - 400) :
  x = 800 := 
sorry

end jelly_bean_problem_l228_228082


namespace solution_set_of_inequality_l228_228140

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Problem conditions
theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x, x < 0 → f x = x + 2) :
  { x : ℝ | 2 * f x - 1 < 0 } = { x : ℝ | x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) } :=
by
  sorry

end solution_set_of_inequality_l228_228140


namespace orthogonal_trajectories_angle_at_origin_l228_228562

theorem orthogonal_trajectories_angle_at_origin (x y : ℝ) (a : ℝ) :
  ((x + 2 * y) ^ 2 = a * (x + y)) →
  (∃ φ : ℝ, φ = π / 4) :=
by
  sorry

end orthogonal_trajectories_angle_at_origin_l228_228562


namespace total_dinners_sold_203_l228_228404

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l228_228404


namespace simplify_polynomial_l228_228409

theorem simplify_polynomial (x : ℝ) :
  (5 - 5 * x - 10 * x^2 + 10 + 15 * x - 20 * x^2 - 10 + 20 * x + 30 * x^2) = 5 + 30 * x :=
  by sorry

end simplify_polynomial_l228_228409


namespace ratio_swordfish_to_pufferfish_l228_228673

theorem ratio_swordfish_to_pufferfish (P S : ℕ) (n : ℕ) 
  (hP : P = 15)
  (hTotal : S + P = 90)
  (hRelation : S = n * P) : 
  (S : ℚ) / (P : ℚ) = 5 := 
by 
  sorry

end ratio_swordfish_to_pufferfish_l228_228673


namespace equivalent_fraction_l228_228785

theorem equivalent_fraction :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) :=
by 
  sorry

end equivalent_fraction_l228_228785


namespace circle_center_radius_sum_correct_l228_228570

noncomputable def circle_center_radius_sum (eq : String) : ℝ :=
  if h : eq = "x^2 + 8x - 2y^2 - 6y = -6" then
    let c : ℝ := -4
    let d : ℝ := -3 / 2
    let s : ℝ := Real.sqrt (47 / 4)
    c + d + s
  else 0

theorem circle_center_radius_sum_correct :
  circle_center_radius_sum "x^2 + 8x - 2y^2 - 6y = -6" = (-11 + Real.sqrt 47) / 2 :=
by
  -- proof omitted
  sorry

end circle_center_radius_sum_correct_l228_228570


namespace minimum_time_to_finish_route_l228_228177

-- Step (a): Defining conditions and necessary terms
def points : Nat := 12
def segments_between_points : ℕ := 17
def time_per_segment : ℕ := 10 -- in minutes
def total_time_in_minutes : ℕ := segments_between_points * time_per_segment -- Total time in minutes

-- Step (c): Proving the question == answer given conditions
theorem minimum_time_to_finish_route (K : ℕ) : K = 4 :=
by
  have time_in_hours : ℕ := total_time_in_minutes / 60
  have minimum_time : ℕ := 4
  sorry -- proof needed

end minimum_time_to_finish_route_l228_228177


namespace shifted_parabola_eq_l228_228970

def initial_parabola (x : ℝ) : ℝ := 5 * x^2

def shifted_parabola (x : ℝ) : ℝ := 5 * (x + 2)^2 + 3

theorem shifted_parabola_eq :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  intro x
  sorry

end shifted_parabola_eq_l228_228970


namespace triangle_side_relationship_l228_228979

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem triangle_side_relationship 
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = 40 * Real.pi / 180)
  (hβ : β = 60 * Real.pi / 180)
  (hγ : γ = 80 * Real.pi / 180)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * (a + b + c) = b * (b + c) :=
sorry

end triangle_side_relationship_l228_228979


namespace arithmetic_sequence_value_l228_228035

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_value (h : a 3 + a 5 + a 11 + a 13 = 80) : a 8 = 20 :=
sorry

end arithmetic_sequence_value_l228_228035


namespace speed_of_stream_l228_228283

theorem speed_of_stream (v : ℝ) 
    (h1 : ∀ (v : ℝ), v ≠ 0 → (80 / (36 + v) = 40 / (36 - v))) : 
    v = 12 := 
by 
    sorry

end speed_of_stream_l228_228283


namespace initial_number_of_students_l228_228312

theorem initial_number_of_students (W : ℝ) (n : ℕ) (new_student_weight avg_weight1 avg_weight2 : ℝ)
  (h1 : avg_weight1 = 15)
  (h2 : new_student_weight = 13)
  (h3 : avg_weight2 = 14.9)
  (h4 : W = n * avg_weight1)
  (h5 : W + new_student_weight = (n + 1) * avg_weight2) : n = 19 := 
by
  sorry

end initial_number_of_students_l228_228312


namespace subtract_two_percent_is_multiplying_l228_228636

theorem subtract_two_percent_is_multiplying (a : ℝ) : (a - 0.02 * a) = 0.98 * a := by
  sorry

end subtract_two_percent_is_multiplying_l228_228636


namespace young_li_age_l228_228244

theorem young_li_age (x : ℝ) (old_li_age : ℝ) 
  (h1 : old_li_age = 2.5 * x)  
  (h2 : old_li_age + 10 = 2 * (x + 10)) : 
  x = 20 := 
by
  sorry

end young_li_age_l228_228244


namespace problem_1_l228_228626

theorem problem_1 : -9 + 5 - (-12) + (-3) = 5 :=
by {
  -- Proof goes here
  sorry
}

end problem_1_l228_228626


namespace total_sum_vowels_l228_228743

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end total_sum_vowels_l228_228743


namespace scientific_notation_36000_l228_228067

theorem scientific_notation_36000 : 36000 = 3.6 * (10^4) := 
by 
  -- Skipping the proof by adding sorry
  sorry

end scientific_notation_36000_l228_228067


namespace exponent_multiplication_identity_l228_228827

theorem exponent_multiplication_identity : 2^4 * 3^2 * 5^2 * 7 = 6300 := sorry

end exponent_multiplication_identity_l228_228827


namespace percentage_water_fresh_fruit_l228_228536

-- Definitions of the conditions
def weight_dried_fruit : ℝ := 12
def water_content_dried_fruit : ℝ := 0.15
def weight_fresh_fruit : ℝ := 101.99999999999999

-- Derived definitions based on the conditions
def weight_non_water_dried_fruit : ℝ := weight_dried_fruit - (water_content_dried_fruit * weight_dried_fruit)
def weight_non_water_fresh_fruit : ℝ := weight_non_water_dried_fruit
def weight_water_fresh_fruit : ℝ := weight_fresh_fruit - weight_non_water_fresh_fruit

-- Proof statement
theorem percentage_water_fresh_fruit :
  (weight_water_fresh_fruit / weight_fresh_fruit) * 100 = 90 :=
sorry

end percentage_water_fresh_fruit_l228_228536


namespace percentage_for_overnight_stays_l228_228793

noncomputable def total_bill : ℝ := 5000
noncomputable def medication_percentage : ℝ := 0.50
noncomputable def food_cost : ℝ := 175
noncomputable def ambulance_cost : ℝ := 1700

theorem percentage_for_overnight_stays :
  let medication_cost := medication_percentage * total_bill
  let remaining_bill := total_bill - medication_cost
  let cost_for_overnight_stays := remaining_bill - food_cost - ambulance_cost
  (cost_for_overnight_stays / remaining_bill) * 100 = 25 :=
by
  sorry

end percentage_for_overnight_stays_l228_228793


namespace average_speed_palindrome_l228_228747

open Nat

theorem average_speed_palindrome :
  ∀ (initial final : ℕ) (time : ℕ), (initial = 12321) →
    (final = 12421) →
    (time = 3) →
    (∃ speed : ℚ, speed = (final - initial) / time ∧ speed = 33.33) :=
by
  intros initial final time h_initial h_final h_time
  sorry

end average_speed_palindrome_l228_228747


namespace no_such_m_exists_l228_228776

theorem no_such_m_exists : ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0 :=
sorry

end no_such_m_exists_l228_228776


namespace year_2024_AD_representation_l228_228539

def year_representation (y: Int) : Int :=
  if y > 0 then y else -y

theorem year_2024_AD_representation : year_representation 2024 = 2024 :=
by sorry

end year_2024_AD_representation_l228_228539


namespace allocation_ways_l228_228206

/-- Defining the number of different balls and boxes -/
def num_balls : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem asserting the number of ways to place the balls into the boxes -/
theorem allocation_ways : (num_boxes ^ num_balls) = 81 := by
  sorry

end allocation_ways_l228_228206


namespace hex_product_l228_228346

def hex_to_dec (h : Char) : Nat :=
  match h with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | c   => c.toNat - '0'.toNat

noncomputable def dec_to_hex (n : Nat) : String :=
  let q := n / 16
  let r := n % 16
  let r_hex := if r < 10 then Char.ofNat (r + '0'.toNat) else Char.ofNat (r - 10 + 'A'.toNat)
  (if q > 0 then toString q else "") ++ Char.toString r_hex

theorem hex_product :
  dec_to_hex (hex_to_dec 'A' * hex_to_dec 'B') = "6E" :=
by
  sorry

end hex_product_l228_228346


namespace reduced_price_l228_228318

open Real

noncomputable def original_price : ℝ := 33.33

variables (P R: ℝ) (Q : ℝ)

theorem reduced_price
  (h1 : R = 0.75 * P)
  (h2 : P * 500 / P = 500)
  (h3 : 0.75 * P * (Q + 5) = 500)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  -- The proof will be provided here
  sorry

end reduced_price_l228_228318


namespace amount_earned_from_each_family_l228_228905

theorem amount_earned_from_each_family
  (goal : ℕ) (earn_from_fifteen_families : ℕ) (additional_needed : ℕ) (three_families : ℕ) 
  (earn_from_three_families_total : ℕ) (per_family_earn : ℕ) :
  goal = 150 →
  earn_from_fifteen_families = 75 →
  additional_needed = 45 →
  three_families = 3 →
  earn_from_three_families_total = (goal - additional_needed) - earn_from_fifteen_families →
  per_family_earn = earn_from_three_families_total / three_families →
  per_family_earn = 10 :=
by
  sorry

end amount_earned_from_each_family_l228_228905


namespace even_function_value_of_a_l228_228802

theorem even_function_value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x * (Real.exp x + a * Real.exp (-x))) (h_even : ∀ x : ℝ, f x = f (-x)) : a = -1 := 
by
  sorry

end even_function_value_of_a_l228_228802


namespace swimmer_upstream_distance_l228_228506

theorem swimmer_upstream_distance (v : ℝ) (c : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
                                   (downstream_speed : ℝ) (upstream_time : ℝ) : 
  c = 4.5 →
  downstream_distance = 55 →
  downstream_time = 5 →
  downstream_speed = downstream_distance / downstream_time →
  v + c = downstream_speed →
  upstream_time = 5 →
  (v - c) * upstream_time = 10 := 
by
  intro h_c
  intro h_downstream_distance
  intro h_downstream_time
  intro h_downstream_speed
  intro h_effective_downstream
  intro h_upstream_time
  sorry

end swimmer_upstream_distance_l228_228506


namespace yoonseok_handshakes_l228_228913

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

end yoonseok_handshakes_l228_228913


namespace nested_inverse_value_l228_228774

def f (x : ℝ) : ℝ := 5 * x + 6

noncomputable def f_inv (y : ℝ) : ℝ := (y - 6) / 5

theorem nested_inverse_value :
  f_inv (f_inv 16) = -4/5 :=
by
  sorry

end nested_inverse_value_l228_228774


namespace domain_sqrt_function_l228_228654

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

theorem domain_sqrt_function (a : ℝ) :
  quadratic_nonneg_for_all_x a ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end domain_sqrt_function_l228_228654


namespace percentage_markup_l228_228124

theorem percentage_markup (P : ℝ) : 
  (∀ (n : ℕ) (cost price total_earned : ℝ),
    n = 50 →
    cost = 1 →
    price = 1 + P / 100 →
    total_earned = 60 →
    n * price = total_earned) →
  P = 20 :=
by
  intro h
  have h₁ := h 50 1 (1 + P / 100) 60 rfl rfl rfl rfl
  sorry  -- Placeholder for proof steps

end percentage_markup_l228_228124


namespace lap_time_improvement_l228_228815

theorem lap_time_improvement (initial_laps : ℕ) (initial_time : ℕ) (current_laps : ℕ) (current_time : ℕ)
  (h1 : initial_laps = 15) (h2 : initial_time = 45) (h3 : current_laps = 18) (h4 : current_time = 42) :
  (45 / 15 - 42 / 18 : ℚ) = 2 / 3 :=
by
  sorry

end lap_time_improvement_l228_228815


namespace total_spent_is_correct_l228_228211

def cost_of_lunch : ℝ := 60.50
def tip_percentage : ℝ := 0.20
def tip_amount : ℝ := cost_of_lunch * tip_percentage
def total_amount_spent : ℝ := cost_of_lunch + tip_amount

theorem total_spent_is_correct : total_amount_spent = 72.60 := by
  -- placeholder for the proof
  sorry

end total_spent_is_correct_l228_228211


namespace root_range_of_f_eq_zero_solution_set_of_f_le_zero_l228_228297

variable (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + (2 * m + 1) * x + 2

theorem root_range_of_f_eq_zero (h : ∃ r1 r2 : ℝ, r1 > 1 ∧ r2 < 1 ∧ f r1 = 0 ∧ f r2 = 0) : -1 < m ∧ m < 0 :=
sorry

theorem solution_set_of_f_le_zero : 
  (m = 0 -> ∀ x, f x ≤ 0 ↔ x ≤ - 2) ∧
  (m < 0 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) ∧
  (0 < m ∧ m < 1/2 -> ∀ x, f x ≤ 0 ↔ - (1/m) ≤ x ∧ x ≤ - 2) ∧
  (m = 1/2 -> ∀ x, f x ≤ 0 ↔ x = - 2) ∧
  (m > 1/2 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) :=
sorry

end root_range_of_f_eq_zero_solution_set_of_f_le_zero_l228_228297


namespace quadratic_same_roots_abs_l228_228219

theorem quadratic_same_roots_abs (d e : ℤ) : 
  (∀ x : ℤ, |x - 8| = 3 ↔ x = 11 ∨ x = 5) →
  (∀ x : ℤ, x^2 + d * x + e = 0 ↔ x = 11 ∨ x = 5) →
  (d, e) = (-16, 55) :=
by
  intro h₁ h₂
  have h₃ : ∀ x : ℤ, x^2 - 16 * x + 55 = 0 ↔ x = 11 ∨ x = 5 := sorry
  sorry

end quadratic_same_roots_abs_l228_228219


namespace necessary_and_sufficient_condition_l228_228350

theorem necessary_and_sufficient_condition (x : ℝ) : (0 < (1 / x) ∧ (1 / x) < 1) ↔ (1 < x) := sorry

end necessary_and_sufficient_condition_l228_228350


namespace smallest_value_is_nine_l228_228854

noncomputable def smallest_possible_value (a b c d : ℝ) : ℝ :=
  (⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ : ℝ)

theorem smallest_value_is_nine {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_possible_value a b c d = 9 :=
sorry

end smallest_value_is_nine_l228_228854


namespace dan_remaining_money_l228_228656

noncomputable def calculate_remaining_money (initial_amount : ℕ) : ℕ :=
  let candy_bars_qty := 5
  let candy_bar_price := 125
  let candy_bars_discount := 10
  let gum_qty := 3
  let gum_price := 80
  let soda_qty := 4
  let soda_price := 240
  let chips_qty := 2
  let chip_price := 350
  let chips_discount := 15
  let low_tax := 7
  let high_tax := 12

  let total_candy_bars_cost := candy_bars_qty * candy_bar_price
  let discounted_candy_bars_cost := total_candy_bars_cost * (100 - candy_bars_discount) / 100

  let total_gum_cost := gum_qty * gum_price

  let total_soda_cost := soda_qty * soda_price

  let total_chips_cost := chips_qty * chip_price
  let discounted_chips_cost := total_chips_cost * (100 - chips_discount) / 100

  let candy_bars_tax := discounted_candy_bars_cost * low_tax / 100
  let gum_tax := total_gum_cost * low_tax / 100

  let soda_tax := total_soda_cost * high_tax / 100
  let chips_tax := discounted_chips_cost * high_tax / 100

  let total_candy_bars_with_tax := discounted_candy_bars_cost + candy_bars_tax
  let total_gum_with_tax := total_gum_cost + gum_tax
  let total_soda_with_tax := total_soda_cost + soda_tax
  let total_chips_with_tax := discounted_chips_cost + chips_tax

  let total_cost := total_candy_bars_with_tax + total_gum_with_tax + total_soda_with_tax + total_chips_with_tax

  initial_amount - total_cost

theorem dan_remaining_money : 
  calculate_remaining_money 10000 = 7399 :=
sorry

end dan_remaining_money_l228_228656


namespace find_multiplier_l228_228126

theorem find_multiplier (x : ℕ) (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 :=
sorry

end find_multiplier_l228_228126


namespace length_of_rectangle_l228_228553

theorem length_of_rectangle (l : ℝ) (s : ℝ) 
  (perimeter_square : 4 * s = 160) 
  (area_relation : s^2 = 5 * (l * 10)) : 
  l = 32 :=
by
  sorry

end length_of_rectangle_l228_228553


namespace sets_equal_l228_228447

-- Definitions of sets M and N
def M := { u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

-- Theorem statement asserting M = N
theorem sets_equal : M = N :=
by sorry

end sets_equal_l228_228447


namespace positive_difference_of_numbers_l228_228421

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l228_228421


namespace black_pens_count_l228_228287

variable (T B : ℕ)
variable (h1 : (3/10:ℚ) * T = 12)
variable (h2 : (1/5:ℚ) * T = B)

theorem black_pens_count (h1 : (3/10:ℚ) * T = 12) (h2 : (1/5:ℚ) * T = B) : B = 8 := by
  sorry

end black_pens_count_l228_228287


namespace find_m_if_divisible_by_11_l228_228559

theorem find_m_if_divisible_by_11 : ∃ m : ℕ, m < 10 ∧ (734000000 + m*100000 + 8527) % 11 = 0 ↔ m = 6 :=
by {
    sorry
}

end find_m_if_divisible_by_11_l228_228559


namespace derivatives_at_zero_l228_228167

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem derivatives_at_zero :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv (deriv f) 0 = -4 ∧
  deriv (deriv (deriv f)) 0 = 0 ∧
  deriv (deriv (deriv (deriv f))) 0 = 16 :=
by
  sorry

end derivatives_at_zero_l228_228167


namespace first_course_cost_l228_228362

theorem first_course_cost (x : ℝ) (h1 : 60 - (x + (x + 5) + 0.25 * (x + 5)) = 20) : x = 15 :=
by sorry

end first_course_cost_l228_228362


namespace roots_magnitudes_less_than_one_l228_228208

theorem roots_magnitudes_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + A * r + B = 0))
  (h2 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + C * r + D = 0)) :
  ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + (1 / 2 * (A + C)) * r + (1 / 2 * (B + D)) = 0) :=
by
  sorry

end roots_magnitudes_less_than_one_l228_228208


namespace find_A_l228_228201

def is_valid_A (A : ℕ) : Prop :=
  A = 1 ∨ A = 2 ∨ A = 4 ∨ A = 7 ∨ A = 9

def number (A : ℕ) : ℕ :=
  3 * 100000 + 0 * 10000 + 5 * 1000 + 2 * 100 + 0 * 10 + A

theorem find_A (A : ℕ) (h_valid_A : is_valid_A A) : A = 1 ↔ Nat.Prime (number A) :=
by
  sorry

end find_A_l228_228201


namespace find_m_l228_228625

def vector (α : Type*) := α × α

def a : vector ℤ := (1, -2)
def b : vector ℤ := (3, 0)

def two_a_plus_b (a b : vector ℤ) : vector ℤ := (2 * a.1 + b.1, 2 * a.2 + b.2)
def m_a_minus_b (m : ℤ) (a b : vector ℤ) : vector ℤ := (m * a.1 - b.1, m * a.2 - b.2)

def parallel (v w : vector ℤ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_m : parallel (two_a_plus_b a b) (m_a_minus_b (-2) a b) :=
by
  sorry -- proof placeholder

end find_m_l228_228625


namespace xiao_pang_xiao_ya_books_l228_228558

theorem xiao_pang_xiao_ya_books : 
  ∀ (x y : ℕ), 
    (x + 2 * x = 66) → 
    (y + y / 3 = 92) → 
    (2 * x = 2 * x) → 
    (y = 3 * (y / 3)) → 
    ((22 + 69) - (2 * 22 + 69 / 3) = 24) :=
by
  intros x y h1 h2 h3 h4
  sorry

end xiao_pang_xiao_ya_books_l228_228558


namespace sum_of_reciprocal_squares_of_roots_l228_228317

theorem sum_of_reciprocal_squares_of_roots (a b c : ℝ) 
    (h_roots : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c) :
    a + b + c = 6 ∧ ab + bc + ca = 11 ∧ abc = 6 → 
    (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 := 
by
  sorry

end sum_of_reciprocal_squares_of_roots_l228_228317


namespace nurses_count_l228_228606

theorem nurses_count (total : ℕ) (ratio_doc : ℕ) (ratio_nurse : ℕ) (nurses : ℕ) : 
  total = 200 → 
  ratio_doc = 4 → 
  ratio_nurse = 6 → 
  nurses = (ratio_nurse * total / (ratio_doc + ratio_nurse)) → 
  nurses = 120 := 
by 
  intros h_total h_ratio_doc h_ratio_nurse h_calc
  rw [h_total, h_ratio_doc, h_ratio_nurse] at h_calc
  simp at h_calc
  exact h_calc

end nurses_count_l228_228606


namespace find_first_number_l228_228188

theorem find_first_number (x : ℕ) (h1 : x + 35 = 62) : x = 27 := by
  sorry

end find_first_number_l228_228188


namespace infinite_solutions_a_value_l228_228882

theorem infinite_solutions_a_value (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) ↔ a = 5 := 
by 
  sorry

end infinite_solutions_a_value_l228_228882


namespace no_solution_inequality_l228_228660

theorem no_solution_inequality (m x : ℝ) (h1 : x - 2 * m < 0) (h2 : x + m > 2) : m ≤ 2 / 3 :=
  sorry

end no_solution_inequality_l228_228660


namespace divide_shape_into_equal_parts_l228_228850

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

end divide_shape_into_equal_parts_l228_228850


namespace average_income_family_l228_228923

theorem average_income_family (income1 income2 income3 income4 : ℕ) 
  (h1 : income1 = 8000) (h2 : income2 = 15000) (h3 : income3 = 6000) (h4 : income4 = 11000) :
  (income1 + income2 + income3 + income4) / 4 = 10000 := by
  sorry

end average_income_family_l228_228923


namespace school_raised_amount_correct_l228_228087

def school_fundraising : Prop :=
  let mrsJohnson := 2300
  let mrsSutton := mrsJohnson / 2
  let missRollin := mrsSutton * 8
  let topThreeTotal := missRollin * 3
  let mrEdward := missRollin * 0.75
  let msAndrea := mrEdward * 1.5
  let totalRaised := mrsJohnson + mrsSutton + missRollin + mrEdward + msAndrea
  let adminFee := totalRaised * 0.02
  let maintenanceExpense := totalRaised * 0.05
  let totalDeductions := adminFee + maintenanceExpense
  let finalAmount := totalRaised - totalDeductions
  finalAmount = 28737

theorem school_raised_amount_correct : school_fundraising := 
by 
  sorry

end school_raised_amount_correct_l228_228087


namespace eight_pow_15_div_sixtyfour_pow_6_l228_228650

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l228_228650


namespace sum_fifth_to_seventh_terms_arith_seq_l228_228666

theorem sum_fifth_to_seventh_terms_arith_seq (a d : ℤ)
  (h1 : a + 7 * d = 16) (h2 : a + 8 * d = 22) (h3 : a + 9 * d = 28) :
  (a + 4 * d) + (a + 5 * d) + (a + 6 * d) = 12 :=
by
  sorry

end sum_fifth_to_seventh_terms_arith_seq_l228_228666


namespace P_investment_time_l228_228955

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

end P_investment_time_l228_228955


namespace pastries_made_l228_228332

theorem pastries_made (P cakes_sold pastries_sold extra_pastries : ℕ)
  (h1 : cakes_sold = 78)
  (h2 : pastries_sold = 154)
  (h3 : extra_pastries = 76)
  (h4 : pastries_sold = cakes_sold + extra_pastries) :
  P = 154 := sorry

end pastries_made_l228_228332


namespace find_b6b8_l228_228752

-- Define sequences {a_n} (arithmetic sequence) and {b_n} (geometric sequence)
variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Given conditions
axiom h1 : ∀ n m : ℕ, a m = a n + (m - n) * (a (n + 1) - a n) -- Arithmetic sequence property
axiom h2 : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0
axiom h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1 -- Geometric sequence property
axiom h4 : b 7 = a 7
axiom h5 : ∀ n : ℕ, b n > 0                 -- Assuming b_n has positive terms
axiom h6 : ∀ n : ℕ, a n > 0                 -- Positive terms in sequence a_n

-- Proof objective
theorem find_b6b8 : b 6 * b 8 = 16 :=
by sorry

end find_b6b8_l228_228752


namespace geometric_progression_common_ratio_l228_228935

theorem geometric_progression_common_ratio (r : ℝ) (a : ℝ) (h_pos : 0 < a)
    (h_geom_prog : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) :
    r^3 + r^2 + r - 1 = 0 :=
by
  sorry

end geometric_progression_common_ratio_l228_228935


namespace percentage_reduction_is_58_perc_l228_228203

-- Define the conditions
def initial_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.7 * P
def increased_price (P : ℝ) : ℝ := 1.2 * (discount_price P)
def clearance_price (P : ℝ) : ℝ := 0.5 * (increased_price P)

-- The statement of the proof problem
theorem percentage_reduction_is_58_perc (P : ℝ) (h : P > 0) :
  (1 - (clearance_price P / initial_price P)) * 100 = 58 :=
by
  -- Proof omitted
  sorry

end percentage_reduction_is_58_perc_l228_228203


namespace manufacturer_l228_228161

-- Let x be the manufacturer's suggested retail price
variable (x : ℝ)

-- Regular discount range from 10% to 30%
def regular_discount (d : ℝ) : Prop := d >= 0.10 ∧ d <= 0.30

-- Additional discount during sale 
def additional_discount : ℝ := 0.20

-- The final discounted price is $16.80
def final_price (x : ℝ) : Prop := ∃ d, regular_discount d ∧ 0.80 * ((1 - d) * x) = 16.80

theorem manufacturer's_suggested_retail_price :
  final_price x → x = 30 := by
  sorry

end manufacturer_l228_228161


namespace lucy_picked_more_l228_228165

variable (Mary Peter Lucy : ℕ)
variable (Mary_amt Peter_amt Lucy_amt : ℕ)

-- Conditions
def mary_amount : Mary_amt = 12 := sorry
def twice_as_peter : Mary_amt = 2 * Peter_amt := sorry
def total_picked : Mary_amt + Peter_amt + Lucy_amt = 26 := sorry

-- Statement to Prove
theorem lucy_picked_more (h1: Mary_amt = 12) (h2: Mary_amt = 2 * Peter_amt) (h3: Mary_amt + Peter_amt + Lucy_amt = 26) :
  Lucy_amt - Peter_amt = 2 := 
sorry

end lucy_picked_more_l228_228165


namespace remainder_when_divided_by_x_minus_2_l228_228129

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 10

-- State the theorem about the remainder when f(x) is divided by x-2
theorem remainder_when_divided_by_x_minus_2 : f 2 = 30 := by
  -- This is where the proof would go, but we use sorry to skip the proof.
  sorry

end remainder_when_divided_by_x_minus_2_l228_228129


namespace evaluate_f_at_points_l228_228560

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l228_228560


namespace linear_eq_solution_l228_228896

theorem linear_eq_solution (m : ℤ) (x : ℝ) (h1 : |m| = 1) (h2 : 1 - m ≠ 0) : x = -1/2 :=
by
  sorry

end linear_eq_solution_l228_228896


namespace rat_to_chihuahua_ratio_is_six_to_one_l228_228116

noncomputable def chihuahuas_thought_to_be : ℕ := 70
noncomputable def actual_rats : ℕ := 60

theorem rat_to_chihuahua_ratio_is_six_to_one
    (h : chihuahuas_thought_to_be - actual_rats = 10) :
    actual_rats / (chihuahuas_thought_to_be - actual_rats) = 6 :=
by
  sorry

end rat_to_chihuahua_ratio_is_six_to_one_l228_228116


namespace coin_flip_sequences_count_l228_228853

noncomputable def num_sequences_with_given_occurrences : ℕ :=
  sorry

theorem coin_flip_sequences_count : num_sequences_with_given_occurrences = 560 :=
  sorry

end coin_flip_sequences_count_l228_228853


namespace evaluate_f_l228_228217

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 4 * x

theorem evaluate_f (h : f 3 - f (-3) = 672) : True :=
by
  sorry

end evaluate_f_l228_228217


namespace rojo_speed_l228_228838

theorem rojo_speed (R : ℝ) 
  (H : 32 = (R + 3) * 4) : R = 5 :=
sorry

end rojo_speed_l228_228838


namespace sequence_contains_30_l228_228858

theorem sequence_contains_30 :
  ∃ n : ℕ, n * (n + 1) = 30 :=
sorry

end sequence_contains_30_l228_228858


namespace central_angle_correct_l228_228571

-- Define arc length, radius, and central angle
variables (l r α : ℝ)

-- Given conditions
def arc_length := 3
def radius := 2

-- Theorem to prove
theorem central_angle_correct : (l = arc_length) → (r = radius) → (l = r * α) → α = 3 / 2 :=
by
  intros h1 h2 h3
  sorry

end central_angle_correct_l228_228571


namespace greatest_positive_integer_x_l228_228947

theorem greatest_positive_integer_x : ∃ (x : ℕ), (x > 0) ∧ (∀ y : ℕ, y > 0 → (y^3 < 20 * y → y ≤ 4)) ∧ (x^3 < 20 * x) ∧ ∀ z : ℕ, (z > 0) → (z^3 < 20 * z → x ≥ z)  :=
sorry

end greatest_positive_integer_x_l228_228947


namespace mona_biked_monday_l228_228949

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

end mona_biked_monday_l228_228949


namespace part1_part2_l228_228863

-- Define the universal set R
def R := ℝ

-- Define set A
def A (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0

-- Define set B parameterized by a
def B (x a : ℝ) : Prop := (x - (a + 5)) / (x - a) > 0

-- Prove (1): A ∩ B when a = -2
theorem part1 : { x : ℝ | A x } ∩ { x : ℝ | B x (-2) } = { x : ℝ | 3 < x ∧ x ≤ 4 } :=
by
  sorry

-- Prove (2): The range of a such that A ⊆ B
theorem part2 : { a : ℝ | ∀ x, A x → B x a } = { a : ℝ | a < -6 ∨ a > 4 } :=
by
  sorry

end part1_part2_l228_228863


namespace problem_statement_l228_228216

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

def n : ℕ := sorry  -- n is the number of possible values of g(3)
def s : ℝ := sorry  -- s is the sum of all possible values of g(3)

theorem problem_statement : n * s = 0 := sorry

end problem_statement_l228_228216


namespace log2_sufficient_not_necessary_l228_228012

noncomputable def baseTwoLog (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (baseTwoLog a > baseTwoLog b) ↔ (a > b) :=
sorry

end log2_sufficient_not_necessary_l228_228012


namespace range_of_quadratic_function_l228_228465

variable (x : ℝ)
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem range_of_quadratic_function :
  (∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = quadratic_function x) ↔ (1 ≤ y ∧ y ≤ 5)) :=
by
  sorry

end range_of_quadratic_function_l228_228465


namespace trig_identity_proof_l228_228721

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end trig_identity_proof_l228_228721


namespace average_pages_per_book_l228_228994

-- Conditions
def book_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def number_of_books : ℕ := 6

-- Given these conditions, we need to prove the average number of pages per book is 160.
theorem average_pages_per_book (book_thickness_in_inches : ℕ) (pages_per_inch : ℕ) (number_of_books : ℕ)
  (h1 : book_thickness_in_inches = 12)
  (h2 : pages_per_inch = 80)
  (h3 : number_of_books = 6) :
  (book_thickness_in_inches * pages_per_inch) / number_of_books = 160 := by
  sorry

end average_pages_per_book_l228_228994


namespace nails_sum_is_correct_l228_228125

-- Define the fractions for sizes 2d, 3d, 5d, and 8d
def fraction_2d : ℚ := 1 / 6
def fraction_3d : ℚ := 2 / 15
def fraction_5d : ℚ := 1 / 10
def fraction_8d : ℚ := 1 / 8

-- Define the expected answer
def expected_fraction : ℚ := 21 / 40

-- The theorem to prove
theorem nails_sum_is_correct : fraction_2d + fraction_3d + fraction_5d + fraction_8d = expected_fraction :=
by
  -- The proof is not required as per the instructions
  sorry

end nails_sum_is_correct_l228_228125


namespace suzy_total_jumps_in_two_days_l228_228462

-- Definitions based on the conditions in the problem
def yesterdays_jumps : ℕ := 247
def additional_jumps_today : ℕ := 131
def todays_jumps : ℕ := yesterdays_jumps + additional_jumps_today

-- Lean statement of the proof problem
theorem suzy_total_jumps_in_two_days : yesterdays_jumps + todays_jumps = 625 := by
  sorry

end suzy_total_jumps_in_two_days_l228_228462


namespace eval_otimes_l228_228156

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem eval_otimes : otimes 4 2 = 18 :=
by
  sorry

end eval_otimes_l228_228156


namespace abs_inequality_solution_l228_228135

theorem abs_inequality_solution (x : ℝ) : 
  (|2 * x + 1| > 3) ↔ (x > 1 ∨ x < -2) :=
sorry

end abs_inequality_solution_l228_228135


namespace player_A_prize_received_event_A_not_low_probability_l228_228271

-- Condition Definitions
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3
def a : ℚ := 243

-- Part 1: Player A's Prize
theorem player_A_prize_received :
  (a * (p * p + 3 * p * (1 - p) * p + 3 * (1 - p) * p * p + (1 - p) * (1 - p) * p * p)) = 216 := sorry

-- Part 2: Probability of Event A with Low Probability Conditions
def low_probability_event (prob : ℚ) : Prop := prob < 0.05

-- Probability that player B wins the entire prize
def event_A_probability (p : ℚ) : ℚ :=
  (1 - p) ^ 3 + 3 * p * (1 - p) ^ 3

theorem event_A_not_low_probability (p : ℚ) (hp : p ≥ 3 / 4) :
  ¬ low_probability_event (event_A_probability p) := sorry

end player_A_prize_received_event_A_not_low_probability_l228_228271


namespace total_people_waiting_l228_228321

theorem total_people_waiting 
  (initial_first_line : ℕ := 7)
  (left_first_line : ℕ := 4)
  (joined_first_line : ℕ := 8)
  (initial_second_line : ℕ := 12)
  (left_second_line : ℕ := 3)
  (joined_second_line : ℕ := 10)
  (initial_third_line : ℕ := 15)
  (left_third_line : ℕ := 5)
  (joined_third_line : ℕ := 7) :
  (initial_first_line - left_first_line + joined_first_line) +
  (initial_second_line - left_second_line + joined_second_line) +
  (initial_third_line - left_third_line + joined_third_line) = 47 :=
by
  sorry

end total_people_waiting_l228_228321


namespace line_equation_is_correct_l228_228800

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

theorem line_equation_is_correct (x y t : ℝ)
  (h1: x = 3 * t + 6)
  (h2: y = 5 * t - 7) :
  y = (5 / 3) * x - 17 :=
sorry

end line_equation_is_correct_l228_228800


namespace calculate_x_one_minus_f_l228_228392

noncomputable def x := (2 + Real.sqrt 3) ^ 500
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem calculate_x_one_minus_f : x * (1 - f) = 1 := by
  sorry

end calculate_x_one_minus_f_l228_228392


namespace prime_p_equals_2_l228_228728

theorem prime_p_equals_2 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs: Nat.Prime s)
  (h_sum : p + q + r = 2 * s) (h_order : 1 < p ∧ p < q ∧ q < r) : p = 2 :=
sorry

end prime_p_equals_2_l228_228728


namespace line_circle_no_intersect_l228_228142

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l228_228142


namespace allison_uploads_480_hours_in_june_l228_228524

noncomputable def allison_upload_total_hours : Nat :=
  let before_june_16 := 10 * 15
  let from_june_16_to_23 := 15 * 8
  let from_june_24_to_end := 30 * 7
  before_june_16 + from_june_16_to_23 + from_june_24_to_end

theorem allison_uploads_480_hours_in_june :
  allison_upload_total_hours = 480 := by
  sorry

end allison_uploads_480_hours_in_june_l228_228524


namespace trapezoid_area_eq_15_l228_228171

theorem trapezoid_area_eq_15 :
  let line1 := fun (x : ℝ) => 2 * x
  let line2 := fun (x : ℝ) => 8
  let line3 := fun (x : ℝ) => 2
  let y_axis := fun (y : ℝ) => 0
  let intersection_points := [
    (4, 8),   -- Intersection of line1 and line2
    (1, 2),   -- Intersection of line1 and line3
    (0, 8),   -- Intersection of y_axis and line2
    (0, 2)    -- Intersection of y_axis and line3
  ]
  let base1 := (4 - 0 : ℝ)  -- Length of top base 
  let base2 := (1 - 0 : ℝ)  -- Length of bottom base
  let height := (8 - 2 : ℝ) -- Vertical distance between line2 and line3
  (0.5 * (base1 + base2) * height = 15.0) := by
  sorry

end trapezoid_area_eq_15_l228_228171


namespace students_sampled_from_second_grade_l228_228399

def arithmetic_sequence (a d : ℕ) : Prop :=
  3 * a - d = 1200

def stratified_sampling (total students second_grade : ℕ) : ℕ :=
  (second_grade * students) / total

theorem students_sampled_from_second_grade 
  (total students : ℕ)
  (h1 : total = 1200)
  (h2 : students = 48)
  (a d : ℕ)
  (h3 : arithmetic_sequence a d)
: stratified_sampling total students a = 16 :=
by
  rw [h1, h2]
  sorry

end students_sampled_from_second_grade_l228_228399


namespace fewest_printers_l228_228704

/-!
# Fewest Printers Purchase Problem
Given two types of computer printers costing $350 and $200 per unit, respectively,
given that the company wants to spend equal amounts on both types of printers.
Prove that the fewest number of printers the company can purchase is 11.
-/

theorem fewest_printers (p1 p2 : ℕ) (h1 : p1 = 350) (h2 : p2 = 200) :
  ∃ n1 n2 : ℕ, p1 * n1 = p2 * n2 ∧ n1 + n2 = 11 := 
sorry

end fewest_printers_l228_228704


namespace exponential_increasing_l228_228736

theorem exponential_increasing (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 :=
by
  sorry

end exponential_increasing_l228_228736


namespace calculate_expression_l228_228128

theorem calculate_expression : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := 
by 
  /-
  In Lean, we typically perform arithmetic simplifications step by step;
  however, for the purpose of this example, only stating the goal:
  -/
  sorry

end calculate_expression_l228_228128


namespace min_distance_from_curve_to_focus_l228_228690

noncomputable def minDistanceToFocus (x y θ : ℝ) : ℝ :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  a - c

theorem min_distance_from_curve_to_focus :
  ∀ θ : ℝ, minDistanceToFocus (2 * Real.cos θ) (3 * Real.sin θ) θ = 3 - Real.sqrt 5 :=
by
  sorry

end min_distance_from_curve_to_focus_l228_228690


namespace cos_double_angle_l228_228856

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) : Real.cos (2 * θ) = -1/3 := by
  sorry

end cos_double_angle_l228_228856


namespace product_of_solutions_eq_zero_l228_228383

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l228_228383


namespace min_f_l228_228569

noncomputable def f (x y z : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem min_f (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end min_f_l228_228569


namespace sin_c_eq_tan_b_find_side_length_c_l228_228759

-- (1) Prove that sinC = tanB
theorem sin_c_eq_tan_b {a b c : ℝ} {C : ℝ} (h1 : a / b = 1 + Real.cos C) : 
  Real.sin C = Real.tan B := by
  sorry

-- (2) If given conditions, find the value of c
theorem find_side_length_c {a b c : ℝ} {B C : ℝ} 
  (h1 : Real.cos B = 2 * Real.sqrt 7 / 7)
  (h2 : 0 < C ∧ C < Real.pi / 2)
  (h3 : 1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) 
  : c = Real.sqrt 7 := by
  sorry

end sin_c_eq_tan_b_find_side_length_c_l228_228759


namespace ratio_of_students_to_dishes_l228_228683

theorem ratio_of_students_to_dishes (m n : ℕ) 
  (h_students : n > 0)
  (h_dishes : ∃ dishes : Finset ℕ, dishes.card = 100)
  (h_each_student_tastes_10 : ∀ student : Finset ℕ, student.card = 10) 
  (h_pairs_taste_by_m_students : ∀ {d1 d2 : ℕ} (hd1 : d1 ∈ Finset.range 100) (hd2 : d2 ∈ Finset.range 100), m = 10) 
  : n / m = 110 := by
  sorry

end ratio_of_students_to_dishes_l228_228683


namespace cos_sin_identity_l228_228880

theorem cos_sin_identity (x : ℝ) (h : Real.cos (x - Real.pi / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * Real.pi / 3) + Real.sin (Real.pi / 3 - x) ^ 2 = 5 / 3 :=
sorry

end cos_sin_identity_l228_228880


namespace initial_birds_count_l228_228857

theorem initial_birds_count (current_total_birds birds_joined initial_birds : ℕ) 
  (h1 : current_total_birds = 6) 
  (h2 : birds_joined = 4) : 
  initial_birds = current_total_birds - birds_joined → 
  initial_birds = 2 :=
by 
  intro h3
  rw [h1, h2] at h3
  exact h3

end initial_birds_count_l228_228857


namespace Debby_spent_on_yoyo_l228_228607

theorem Debby_spent_on_yoyo 
  (hat_tickets stuffed_animal_tickets total_tickets : ℕ) 
  (h1 : hat_tickets = 2) 
  (h2 : stuffed_animal_tickets = 10) 
  (h3 : total_tickets = 14) 
  : ∃ yoyo_tickets : ℕ, hat_tickets + stuffed_animal_tickets + yoyo_tickets = total_tickets ∧ yoyo_tickets = 2 := 
by 
  sorry

end Debby_spent_on_yoyo_l228_228607


namespace attendance_difference_is_85_l228_228204

def saturday_attendance : ℕ := 80
def monday_attendance : ℕ := saturday_attendance - 20
def wednesday_attendance : ℕ := monday_attendance + 50
def friday_attendance : ℕ := saturday_attendance + monday_attendance
def thursday_attendance : ℕ := 45
def expected_audience : ℕ := 350

def total_attendance : ℕ := 
  saturday_attendance + 
  monday_attendance + 
  wednesday_attendance + 
  friday_attendance + 
  thursday_attendance

def more_people_attended_than_expected : ℕ :=
  total_attendance - expected_audience

theorem attendance_difference_is_85 : more_people_attended_than_expected = 85 := 
by
  unfold more_people_attended_than_expected
  unfold total_attendance
  unfold saturday_attendance
  unfold monday_attendance
  unfold wednesday_attendance
  unfold friday_attendance
  unfold thursday_attendance
  unfold expected_audience
  exact sorry

end attendance_difference_is_85_l228_228204


namespace lunch_break_duration_l228_228996

theorem lunch_break_duration :
  ∃ L : ℝ, 
    ∀ (p h : ℝ),
      (9 - L) * (p + h) = 0.4 ∧
      (7 - L) * h = 0.3 ∧
      (12 - L) * p = 0.3 →
      L = 0.5 := by
  sorry

end lunch_break_duration_l228_228996


namespace females_on_police_force_l228_228520

theorem females_on_police_force (H : ∀ (total_female_officers total_officers_on_duty female_officers_on_duty : ℕ), 
  total_officers_on_duty = 500 ∧ female_officers_on_duty = total_officers_on_duty / 2 ∧ female_officers_on_duty = total_female_officers / 4) :
  ∃ total_female_officers : ℕ, total_female_officers = 1000 := 
by {
  sorry
}

end females_on_police_force_l228_228520


namespace aston_found_pages_l228_228185

-- Given conditions
def pages_per_comic := 25
def initial_untorn_comics := 5
def total_comics_now := 11

-- The number of pages Aston found on the floor
theorem aston_found_pages :
  (total_comics_now - initial_untorn_comics) * pages_per_comic = 150 := 
by
  sorry

end aston_found_pages_l228_228185


namespace line_segment_length_is_0_7_l228_228697

def isLineSegment (length : ℝ) (finite : Bool) : Prop :=
  finite = true ∧ length = 0.7

theorem line_segment_length_is_0_7 : isLineSegment 0.7 true :=
by
  sorry

end line_segment_length_is_0_7_l228_228697


namespace impossible_to_equalize_numbers_l228_228261

theorem impossible_to_equalize_numbers (nums : Fin 6 → ℤ) :
  ¬ (∃ n : ℤ, ∀ i : Fin 6, nums i = n) :=
sorry

end impossible_to_equalize_numbers_l228_228261


namespace possible_values_a_l228_228984

def A : Set ℝ := {-1, 2}
def B (a : ℝ) : Set ℝ := {x | a * x^2 = 2 ∧ a ≥ 0}

def whale_swallowing (S T : Set ℝ) : Prop :=
S ⊆ T ∨ T ⊆ S

def moth_eating (S T : Set ℝ) : Prop :=
(∃ x, x ∈ S ∧ x ∈ T) ∧ ¬(S ⊆ T) ∧ ¬(T ⊆ S)

def valid_a (a : ℝ) : Prop :=
whale_swallowing A (B a) ∨ moth_eating A (B a)

theorem possible_values_a :
  {a : ℝ | valid_a a} = {0, 1/2, 2} :=
sorry

end possible_values_a_l228_228984


namespace cost_of_five_dozen_apples_l228_228288

theorem cost_of_five_dozen_apples 
  (cost_four_dozen : ℝ) 
  (cost_one_dozen : ℝ) 
  (cost_five_dozen : ℝ) 
  (h1 : cost_four_dozen = 31.20) 
  (h2 : cost_one_dozen = cost_four_dozen / 4) 
  (h3 : cost_five_dozen = 5 * cost_one_dozen)
  : cost_five_dozen = 39.00 :=
sorry

end cost_of_five_dozen_apples_l228_228288


namespace solve_for_x_l228_228667

-- We state the problem as a theorem.
theorem solve_for_x (y x : ℚ) : 
  (x - 60) / 3 = (4 - 3 * x) / 6 + y → x = (124 + 6 * y) / 5 :=
by
  -- The actual proof part is skipped with sorry.
  sorry

end solve_for_x_l228_228667


namespace triangle_angles_l228_228460

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : B = 120)
  (h3 : (∃D, A = D ∧ (A + A + C = 180 ∨ A + C + C = 180)) ∨ (∃E, C = E ∧ (B + 15 + 45 = 180 ∨ B + 15 + 15 = 180))) :
  (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
sorry

end triangle_angles_l228_228460


namespace pyramid_can_be_oblique_l228_228341

-- Define what it means for the pyramid to have a regular triangular base.
def regular_triangular_base (pyramid : Type) : Prop := sorry

-- Define what it means for each lateral face to be an isosceles triangle.
def isosceles_lateral_faces (pyramid : Type) : Prop := sorry

-- Define what it means for a pyramid to be oblique.
def can_be_oblique (pyramid : Type) : Prop := sorry

-- Defining pyramid as a type.
variable (pyramid : Type)

-- The theorem stating the problem's conclusion.
theorem pyramid_can_be_oblique 
  (h1 : regular_triangular_base pyramid) 
  (h2 : isosceles_lateral_faces pyramid) : 
  can_be_oblique pyramid :=
sorry

end pyramid_can_be_oblique_l228_228341


namespace triple_supplementary_angle_l228_228809

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l228_228809


namespace avg_remaining_two_l228_228042

theorem avg_remaining_two (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 8) (h2 : (a + b + c) / 3 = 4) :
  (d + e) / 2 = 14 := by
  sorry

end avg_remaining_two_l228_228042


namespace janet_percentage_of_snowballs_l228_228995

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l228_228995


namespace students_in_classes_saved_money_strategy_class7_1_l228_228542

-- Part (1): Prove the number of students in each class
theorem students_in_classes (x : ℕ) (h1 : 40 < x) (h2 : x < 50) 
  (h3 : 105 - x > 50) (h4 : 15 * x + 12 * (105 - x) = 1401) : x = 47 ∧ (105 - x) = 58 := by
  sorry

-- Part (2): Prove the amount saved by purchasing tickets together
theorem saved_money(amt_per_ticket : ℕ → ℕ) 
  (h1 : amt_per_ticket 105 = 1401) 
  (h2 : ∀n, n > 100 → amt_per_ticket n = 1050) : amt_per_ticket 105 - 1050 = 351 := by
  sorry

-- Part (3): Strategy to save money for class 7 (1)
theorem strategy_class7_1 (students_1 : ℕ) (h1 : students_1 = 47) 
  (cost_15 : students_1 * 15 = 705) 
  (cost_51 : 51 * 12 = 612) : 705 - 612 = 93 := by
  sorry

end students_in_classes_saved_money_strategy_class7_1_l228_228542


namespace opposite_of_x_is_positive_l228_228284

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive_l228_228284


namespace width_of_crate_l228_228366

theorem width_of_crate
  (r : ℝ) (h : ℝ) (w : ℝ)
  (h_crate : h = 6 ∨ h = 10 ∨ w = 6 ∨ w = 10)
  (r_tank : r = 4)
  (height_longest_crate : h > w)
  (maximize_volume : ∃ d : ℝ, d = 2 * r ∧ w = d) :
  w = 8 := 
sorry

end width_of_crate_l228_228366


namespace sophia_read_more_pages_l228_228718

variable (total_pages : ℝ) (finished_fraction : ℝ)
variable (pages_read : ℝ) (pages_left : ℝ) (pages_more : ℝ)

theorem sophia_read_more_pages :
  total_pages = 269.99999999999994 ∧
  finished_fraction = 2/3 ∧
  pages_read = finished_fraction * total_pages ∧
  pages_left = total_pages - pages_read →
  pages_more = pages_read - pages_left →
  pages_more = 90 := 
by
  intro h
  sorry

end sophia_read_more_pages_l228_228718


namespace final_position_total_distance_l228_228565

-- Define the movements as a list
def movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

-- Prove that the final position of the turtle is 5 meters north of the starting point
theorem final_position (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum movements = 5 :=
by
  rw [h]
  sorry

-- Prove that the total distance crawled by the turtle is 47 meters
theorem total_distance (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum (List.map Int.natAbs movements) = 47 :=
by
  rw [h]
  sorry

end final_position_total_distance_l228_228565


namespace time_to_chop_an_onion_is_4_minutes_l228_228803

noncomputable def time_to_chop_pepper := 3
noncomputable def time_to_grate_cheese_per_omelet := 1
noncomputable def time_to_cook_omelet := 5
noncomputable def peppers_needed := 4
noncomputable def onions_needed := 2
noncomputable def omelets_needed := 5
noncomputable def total_time := 50

theorem time_to_chop_an_onion_is_4_minutes : 
  (total_time - (peppers_needed * time_to_chop_pepper + omelets_needed * time_to_grate_cheese_per_omelet + omelets_needed * time_to_cook_omelet)) / onions_needed = 4 := by sorry

end time_to_chop_an_onion_is_4_minutes_l228_228803


namespace sum_of_roots_eq_p_l228_228602

variable (p q : ℝ)
variable (hq : q = p^2 - 1)

theorem sum_of_roots_eq_p (h : q = p^2 - 1) : 
  let r1 := p
  let r2 := q
  r1 + r2 = p := 
sorry

end sum_of_roots_eq_p_l228_228602


namespace find_a_l228_228500

noncomputable def f (x a : ℝ) : ℝ := x / (x^2 + a)

theorem find_a (a : ℝ) (h_positive : a > 0) (h_max : ∀ x, x ∈ Set.Ici 1 → f x a ≤ f 1 a) :
  a = Real.sqrt 3 - 1 := by
  sorry

end find_a_l228_228500


namespace sequence_divisible_by_three_l228_228926

-- Define the conditions
variable (k : ℕ) (h_pos_k : k > 0)
variable (a : ℕ → ℤ)
variable (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n)

-- Define the proof goal
theorem sequence_divisible_by_three (k : ℕ) (h_pos_k : k > 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n) : (k - 2) % 3 = 0 :=
by
  sorry

end sequence_divisible_by_three_l228_228926


namespace conic_not_parabola_l228_228945

def conic_equation (m x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

theorem conic_not_parabola (m : ℝ) :
  ¬ (∃ (x y : ℝ), conic_equation m x y ∧ ∃ (a b c d e f : ℝ), m * x^2 + (m + 1) * y^2 = a * x^2 + b * xy + c * y^2 + d * x + e * y + f ∧ (a = 0 ∨ c = 0) ∧ (b ≠ 0 ∨ a ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0)) :=  
sorry

end conic_not_parabola_l228_228945


namespace arithmetic_sequence_common_difference_l228_228063

theorem arithmetic_sequence_common_difference 
    (a_2 : ℕ → ℕ) (S_4 : ℕ) (a_n : ℕ → ℕ → ℕ) (S_n : ℕ → ℕ → ℕ → ℕ)
    (h1 : a_2 2 = 3) (h2 : S_4 = 16) 
    (h3 : ∀ n a_1 d, a_n a_1 n = a_1 + (n-1)*d)
    (h4 : ∀ n a_1 d, S_n n a_1 d = n / 2 * (2*a_1 + (n-1)*d)) : ∃ d, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l228_228063


namespace max_value_of_expression_l228_228398

theorem max_value_of_expression (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 / 4 + 9 * y₁^2 / 4 = 1) 
  (h₂ : x₂^2 / 4 + 9 * y₂^2 / 4 = 1) 
  (h₃ : x₁ * x₂ + 9 * y₁ * y₂ = -2) :
  (|2 * x₁ + 3 * y₁ - 3| + |2 * x₂ + 3 * y₂ - 3|) ≤ 6 + 2 * Real.sqrt 5 :=
sorry

end max_value_of_expression_l228_228398


namespace circle_iff_m_gt_neg_1_over_2_l228_228629

noncomputable def represents_circle (m: ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + x + y - m = 0) → m > -1/2

theorem circle_iff_m_gt_neg_1_over_2 (m : ℝ) : represents_circle m ↔ m > -1/2 := by
  sorry

end circle_iff_m_gt_neg_1_over_2_l228_228629


namespace find_colored_copies_l228_228714

variable (cost_c cost_w total_copies total_cost : ℝ)
variable (colored_copies white_copies : ℝ)

def colored_copies_condition (cost_c cost_w total_copies total_cost : ℝ) :=
  ∃ (colored_copies white_copies : ℝ),
    colored_copies + white_copies = total_copies ∧
    cost_c * colored_copies + cost_w * white_copies = total_cost

theorem find_colored_copies :
  colored_copies_condition 0.10 0.05 400 22.50 → 
  ∃ (c : ℝ), c = 50 :=
by 
  sorry

end find_colored_copies_l228_228714


namespace find_abcde_l228_228245

theorem find_abcde (N : ℕ) (a b c d e f : ℕ) (h : a ≠ 0) 
(h1 : N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
(h2 : (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) :
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437 :=
by sorry

end find_abcde_l228_228245


namespace dan_initial_amount_l228_228909

variables (initial_amount spent_amount remaining_amount : ℝ)

theorem dan_initial_amount (h1 : spent_amount = 1) (h2 : remaining_amount = 2) : initial_amount = spent_amount + remaining_amount := by
  sorry

end dan_initial_amount_l228_228909


namespace functional_equation_solution_l228_228248

theorem functional_equation_solution (f : ℚ → ℕ) :
  (∀ (x y : ℚ) (hx : 0 < x) (hy : 0 < y),
    f (x * y) * Nat.gcd (f x * f y) (f (x⁻¹) * f (y⁻¹)) = (x * y) * f (x⁻¹) * f (y⁻¹))
  → (∀ (x : ℚ) (hx : 0 < x), f x = x.num) :=
sorry

end functional_equation_solution_l228_228248


namespace diesel_train_slower_l228_228962

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

end diesel_train_slower_l228_228962


namespace simplify_expression_l228_228611

theorem simplify_expression : 
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2)) - 
  (Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2 / 3))) = -Real.sqrt 2 :=
by
  sorry

end simplify_expression_l228_228611


namespace total_employees_with_advanced_degrees_l228_228964

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

end total_employees_with_advanced_degrees_l228_228964


namespace negation_of_proposition_l228_228586

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l228_228586


namespace rectangle_area_l228_228192

theorem rectangle_area (l w : ℝ) (h₁ : (2 * l + 2 * w) = 46) (h₂ : (l^2 + w^2) = 289) : l * w = 120 :=
by
  sorry

end rectangle_area_l228_228192


namespace tan_sum_l228_228895

theorem tan_sum (θ : ℝ) (h : Real.sin (2 * θ) = 2 / 3) : Real.tan θ + 1 / Real.tan θ = 3 := sorry

end tan_sum_l228_228895


namespace solution_m_value_l228_228114

theorem solution_m_value (m : ℝ) : 
  (m^2 - 5*m + 4 > 0) ∧ (m^2 - 2*m = 0) ↔ m = 0 :=
by
  sorry

end solution_m_value_l228_228114


namespace interest_rate_calculation_l228_228888

-- Define the problem conditions and proof statement in Lean
theorem interest_rate_calculation 
  (P : ℝ) (r : ℝ) (T : ℝ) (CI SI diff : ℝ) 
  (principal_condition : P = 6000.000000000128)
  (time_condition : T = 2)
  (diff_condition : diff = 15)
  (CI_formula : CI = P * (1 + r)^T - P)
  (SI_formula : SI = P * r * T)
  (difference_condition : CI - SI = diff) : 
  r = 0.05 := 
by 
  sorry

end interest_rate_calculation_l228_228888


namespace phantom_needs_more_money_l228_228253

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l228_228253


namespace always_real_roots_range_of_b_analytical_expression_parabola_l228_228974

-- Define the quadratic equation with parameter m
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (5 * m - 1) * x + 4 * m - 4

-- Part 1: Prove the equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 := 
sorry

-- Part 2: Find the range of b such that the line intersects the parabola at two distinct points
theorem range_of_b (b : ℝ) : 
  (∀ m : ℝ, m = 1 → (b > -25/4 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = (x1 + b) ∧ quadratic_eq m x2 = (x2 + b)))) :=
sorry

-- Part 3: Find the analytical expressions of the parabolas given the distance condition
theorem analytical_expression_parabola (m : ℝ) : 
  (∀ x1 x2 : ℝ, (|x1 - x2| = 2 → quadratic_eq m x1 = 0 → quadratic_eq m x2 = 0) → 
  (m = -1 ∨ m = -1/5) → 
  ((quadratic_eq (-1) x = -x^2 + 6*x - 8) ∨ (quadratic_eq (-1/5) x = -1/5*x^2 + 2*x - 24/5))) :=
sorry

end always_real_roots_range_of_b_analytical_expression_parabola_l228_228974


namespace white_ball_probability_l228_228892

theorem white_ball_probability :
  ∀ (n : ℕ), (2/(n+2) = 2/5) → (n = 3) → (n/(n+2) = 3/5) :=
by
  sorry

end white_ball_probability_l228_228892


namespace marble_ratio_l228_228608

theorem marble_ratio 
  (K A M : ℕ) 
  (M_has_5_times_as_many_as_K : M = 5 * K)
  (M_has_85_marbles : M = 85)
  (M_has_63_more_than_A : M = A + 63)
  (A_needs_12_more : A + 12 = 34) :
  34 / 17 = 2 := 
by 
  sorry

end marble_ratio_l228_228608


namespace sodas_total_l228_228183

def morning_sodas : ℕ := 77
def afternoon_sodas : ℕ := 19
def total_sodas : ℕ := morning_sodas + afternoon_sodas

theorem sodas_total :
  total_sodas = 96 :=
by
  sorry

end sodas_total_l228_228183


namespace fraction_to_decimal_l228_228735

theorem fraction_to_decimal :
  (45 : ℚ) / (5 ^ 3) = 0.360 :=
by
  sorry

end fraction_to_decimal_l228_228735


namespace jenny_coins_value_l228_228627

theorem jenny_coins_value (n d : ℕ) (h1 : d = 30 - n) (h2 : 150 + 5 * n = 300 - 5 * n + 120) :
  (300 - 5 * n : ℚ) / 100 = 1.65 := 
by
  sorry

end jenny_coins_value_l228_228627


namespace beau_age_today_l228_228712

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l228_228712


namespace no_valid_positive_x_l228_228306

theorem no_valid_positive_x
  (π : Real)
  (R H x : Real)
  (hR : R = 5)
  (hH : H = 10)
  (hx_pos : x > 0) :
  ¬π * (R + x) ^ 2 * H = π * R ^ 2 * (H + x) :=
by
  sorry

end no_valid_positive_x_l228_228306


namespace sprint_team_total_miles_l228_228132

-- Define the number of people and miles per person as constants
def numberOfPeople : ℕ := 250
def milesPerPerson : ℝ := 7.5

-- Assertion to prove the total miles
def totalMilesRun : ℝ := numberOfPeople * milesPerPerson

-- Proof statement
theorem sprint_team_total_miles : totalMilesRun = 1875 := 
by 
  -- Proof to be filled in
  sorry

end sprint_team_total_miles_l228_228132


namespace height_comparison_of_cylinder_and_rectangular_solid_l228_228983

theorem height_comparison_of_cylinder_and_rectangular_solid
  (V : ℝ) (A : ℝ) (h_cylinder : ℝ) (h_rectangular_solid : ℝ)
  (equal_volume : V = V)
  (equal_base_areas : A = A)
  (height_cylinder_eq : h_cylinder = V / A)
  (height_rectangular_solid_eq : h_rectangular_solid = V / A)
  : ¬ (h_cylinder > h_rectangular_solid) :=
by {
  sorry
}

end height_comparison_of_cylinder_and_rectangular_solid_l228_228983


namespace find_t_l228_228676

theorem find_t (t : ℝ) : 
  (∃ (m b : ℝ), (∀ x y, (y = m * x + b) → ((x = 1 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 5 ∧ y = 19))) ∧ (28 = 28 * m + b) ∧ (t = 28 * m + b)) → 
  t = 88 :=
by
  sorry

end find_t_l228_228676


namespace inequality_proof_l228_228738

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := 
sorry

end inequality_proof_l228_228738


namespace remainder_of_n_plus_4500_l228_228998

theorem remainder_of_n_plus_4500 (n : ℕ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := 
by
  sorry

end remainder_of_n_plus_4500_l228_228998


namespace current_value_l228_228246

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l228_228246


namespace max_x2_plus_2xy_plus_3y2_l228_228907

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end max_x2_plus_2xy_plus_3y2_l228_228907


namespace train_length_proof_l228_228780

-- Definitions for conditions
def jogger_speed_kmh : ℕ := 9
def train_speed_kmh : ℕ := 45
def initial_distance_ahead_m : ℕ := 280
def time_to_pass_s : ℕ := 40

-- Conversion factors
def km_per_hr_to_m_per_s (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

-- Converted speeds
def jogger_speed_m_per_s : ℕ := km_per_hr_to_m_per_s jogger_speed_kmh
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s train_speed_kmh

-- Relative speed
def relative_speed_m_per_s : ℕ := train_speed_m_per_s - jogger_speed_m_per_s

-- Distance covered relative to the jogger
def distance_covered_relative_m : ℕ := relative_speed_m_per_s * time_to_pass_s

-- Length of the train
def length_of_train_m : ℕ := distance_covered_relative_m + initial_distance_ahead_m

-- Theorem to prove 
theorem train_length_proof : length_of_train_m = 680 := 
by
   sorry

end train_length_proof_l228_228780


namespace machine_minutes_worked_l228_228336

theorem machine_minutes_worked {x : ℕ} 
  (h_rate : ∀ y : ℕ, 6 * y = number_of_shirts_machine_makes_yesterday)
  (h_today : 14 = number_of_shirts_machine_makes_today)
  (h_total : number_of_shirts_machine_makes_yesterday + number_of_shirts_machine_makes_today = 156) : 
  x = 23 :=
by
  sorry

end machine_minutes_worked_l228_228336


namespace gcd_lcm_sum_l228_228457

theorem gcd_lcm_sum :
  Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := 
by
  sorry

end gcd_lcm_sum_l228_228457


namespace minutes_between_bathroom_visits_l228_228280

-- Definition of the conditions
def movie_duration_hours : ℝ := 2.5
def bathroom_uses : ℕ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement for the proof
theorem minutes_between_bathroom_visits :
  let total_movie_minutes := movie_duration_hours * minutes_per_hour
  let intervals := bathroom_uses + 1
  total_movie_minutes / intervals = 37.5 :=
by
  sorry

end minutes_between_bathroom_visits_l228_228280


namespace brian_spent_on_kiwis_l228_228906

theorem brian_spent_on_kiwis :
  ∀ (cost_per_dozen_apples : ℝ)
    (cost_for_24_apples : ℝ)
    (initial_money : ℝ)
    (subway_fare_one_way : ℝ)
    (total_remaining : ℝ)
    (kiwis_spent : ℝ)
    (bananas_spent : ℝ),
  cost_per_dozen_apples = 14 →
  cost_for_24_apples = 2 * cost_per_dozen_apples →
  initial_money = 50 →
  subway_fare_one_way = 3.5 →
  total_remaining = initial_money - 2 * subway_fare_one_way - cost_for_24_apples →
  total_remaining = 15 →
  bananas_spent = kiwis_spent / 2 →
  kiwis_spent + bananas_spent = total_remaining →
  kiwis_spent = 10 :=
by
  -- Sorry means we are skipping the proof
  sorry

end brian_spent_on_kiwis_l228_228906


namespace Eddie_number_divisibility_l228_228557

theorem Eddie_number_divisibility (n: ℕ) (h₁: n = 40) (h₂: n % 5 = 0): n % 2 = 0 := 
by
  sorry

end Eddie_number_divisibility_l228_228557


namespace retail_price_l228_228822

theorem retail_price (W M : ℝ) (hW : W = 20) (hM : M = 80) : W + (M / 100) * W = 36 := by
  sorry

end retail_price_l228_228822


namespace minimum_value_of_fractions_l228_228778

theorem minimum_value_of_fractions (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) : 
  ∃ a b, (0 < a) ∧ (0 < b) ∧ (1 / a + 1 / b = 1) ∧ (∃ t, ∀ x y, (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / y = 1) -> t = (1 / (x - 1) + 4 / (y - 1))) := 
sorry

end minimum_value_of_fractions_l228_228778


namespace curve_C_is_circle_l228_228302

noncomputable def curve_C_equation (a : ℝ) : Prop := ∀ x y : ℝ, a * (x^2) + a * (y^2) - 2 * a^2 * x - 4 * y = 0

theorem curve_C_is_circle
  (a : ℝ)
  (ha : a ≠ 0)
  (h_line_intersects : ∃ M N : ℝ × ℝ, (M.2 = -2 * M.1 + 4) ∧ (N.2 = -2 * N.1 + 4) ∧ (M.1^2 + M.2^2 = N.1^2 + N.2^2) ∧ M ≠ N)
  :
  (curve_C_equation 2) ∧ (∀ x y, x^2 + y^2 - 4*x - 2*y = 0) :=
sorry -- Proof is to be provided

end curve_C_is_circle_l228_228302


namespace tulips_sum_l228_228922

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l228_228922


namespace sum_of_tangent_points_l228_228141

noncomputable def f (x : ℝ) : ℝ := 
  max (max (-7 * x - 19) (3 * x - 1)) (5 * x + 3)

theorem sum_of_tangent_points :
  ∃ x4 x5 x6 : ℝ, 
  (∃ q : ℝ → ℝ, 
    (∀ x, q x = f x ∨ (q x - (-7 * x - 19)) = b * (x - x4)^2
    ∨ (q x - (3 * x - 1)) = b * (x - x5)^2 
    ∨ (q x - (5 * x + 3)) = b * (x - x6)^2)) ∧
  x4 + x5 + x6 = -3.2 :=
sorry

end sum_of_tangent_points_l228_228141


namespace similar_triangles_perimeter_l228_228426

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end similar_triangles_perimeter_l228_228426


namespace days_per_book_l228_228889

theorem days_per_book (total_books : ℕ) (total_days : ℕ)
  (h1 : total_books = 41)
  (h2 : total_days = 492) :
  total_days / total_books = 12 :=
by
  -- proof goes here
  sorry

end days_per_book_l228_228889


namespace purely_imaginary_z_eq_a2_iff_a2_l228_228948

theorem purely_imaginary_z_eq_a2_iff_a2 (a : Real) : 
(∃ (b : Real), a^2 - a - 2 = 0 ∧ a + 1 ≠ 0) → a = 2 :=
by
  sorry

end purely_imaginary_z_eq_a2_iff_a2_l228_228948


namespace sum_of_solutions_eq_3_l228_228862

theorem sum_of_solutions_eq_3 (x y : ℝ) (h1 : x * y = 1) (h2 : x + y = 3) :
  x + y = 3 := sorry

end sum_of_solutions_eq_3_l228_228862


namespace solution_exists_l228_228727

noncomputable def equation (x : ℝ) := 
  (x^2 - 5 * x + 4) / (x - 1) + (2 * x^2 + 7 * x - 4) / (2 * x - 1)

theorem solution_exists : equation 2 = 4 := by
  sorry

end solution_exists_l228_228727


namespace inequality_equality_condition_l228_228615

theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_equality_condition_l228_228615


namespace fraction_of_70cm_ropes_l228_228993

theorem fraction_of_70cm_ropes (R : ℕ) (avg_all : ℚ) (avg_70 : ℚ) (avg_85 : ℚ) (total_len : R * avg_all = 480) 
  (total_ropes : R = 6) : 
  ∃ f : ℚ, f = 1 / 3 ∧ f * R * avg_70 + (R - f * R) * avg_85 = R * avg_all :=
by
  sorry

end fraction_of_70cm_ropes_l228_228993


namespace Isabel_earning_l228_228264

-- Define the number of bead necklaces sold
def bead_necklaces : ℕ := 3

-- Define the number of gem stone necklaces sold
def gemstone_necklaces : ℕ := 3

-- Define the cost of each necklace
def cost_per_necklace : ℕ := 6

-- Calculate the total number of necklaces sold
def total_necklaces : ℕ := bead_necklaces + gemstone_necklaces

-- Calculate the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings is 36 dollars
theorem Isabel_earning : total_earnings = 36 := by
  sorry

end Isabel_earning_l228_228264


namespace right_triangle_area_eq_8_over_3_l228_228755

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end right_triangle_area_eq_8_over_3_l228_228755


namespace find_a_not_perfect_square_l228_228175

theorem find_a_not_perfect_square :
  {a : ℕ | ∀ n : ℕ, n > 0 → ¬(∃ k : ℕ, n * (n + a) = k * k)} = {1, 2, 4} :=
sorry

end find_a_not_perfect_square_l228_228175


namespace min_convex_number_l228_228699

noncomputable def minimum_convex_sets (A B C : ℝ × ℝ) : ℕ :=
  if A ≠ B ∧ B ≠ C ∧ C ≠ A then 3 else 4

theorem min_convex_number (A B C : ℝ × ℝ) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  minimum_convex_sets A B C = 3 :=
by 
  sorry

end min_convex_number_l228_228699


namespace polynomial_simplification_l228_228886

theorem polynomial_simplification (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := 
by 
  sorry

end polynomial_simplification_l228_228886


namespace polynomial_remainder_l228_228101

theorem polynomial_remainder (x : ℝ) : 
  (x - 1)^100 + (x - 2)^200 = (x^2 - 3 * x + 2) * (some_q : ℝ) + 1 :=
sorry

end polynomial_remainder_l228_228101


namespace price_of_other_frisbees_proof_l228_228632

noncomputable def price_of_other_frisbees (P : ℝ) : Prop :=
  ∃ x : ℝ, x + (60 - x) = 60 ∧ x ≥ 0 ∧ P * x + 4 * (60 - x) = 204 ∧ (60 - x) ≥ 24

theorem price_of_other_frisbees_proof : price_of_other_frisbees 3 :=
by
  sorry

end price_of_other_frisbees_proof_l228_228632


namespace correct_calculation_l228_228270

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := 
by sorry

end correct_calculation_l228_228270


namespace first_day_exceeds_200_l228_228525

def bacteria_count (n : ℕ) : ℕ := 4 * 3^n

def exceeds_200 (n : ℕ) : Prop := bacteria_count n > 200

theorem first_day_exceeds_200 : ∃ n, exceeds_200 n ∧ ∀ m < n, ¬ exceeds_200 m :=
by sorry

end first_day_exceeds_200_l228_228525


namespace original_employee_count_l228_228003

theorem original_employee_count (employees_operations : ℝ) 
                                (employees_sales : ℝ) 
                                (employees_finance : ℝ) 
                                (employees_hr : ℝ) 
                                (employees_it : ℝ) 
                                (h1 : employees_operations / 0.82 = 192)
                                (h2 : employees_sales / 0.75 = 135)
                                (h3 : employees_finance / 0.85 = 123)
                                (h4 : employees_hr / 0.88 = 66)
                                (h5 : employees_it / 0.90 = 90) : 
                                employees_operations + employees_sales + employees_finance + employees_hr + employees_it = 734 :=
sorry

end original_employee_count_l228_228003


namespace reasoning_common_sense_l228_228443

theorem reasoning_common_sense :
  (∀ P Q: Prop, names_not_correct → P → ¬Q → affairs_not_successful → ¬Q)
  ∧ (∀ R S: Prop, affairs_not_successful → R → ¬S → rites_not_flourish → ¬S)
  ∧ (∀ T U: Prop, rites_not_flourish → T → ¬U → punishments_not_executed_properly → ¬U)
  ∧ (∀ V W: Prop, punishments_not_executed_properly → V → ¬W → people_nowhere_hands_feet → ¬W)
  → reasoning_is_common_sense :=
by sorry

end reasoning_common_sense_l228_228443


namespace range_F_l228_228170

-- Define the function and its critical points
def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem range_F : ∀ y : ℝ, y ∈ Set.range F ↔ -4 ≤ y := by
  sorry

end range_F_l228_228170


namespace arccos_cos_9_eq_2_717_l228_228028

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end arccos_cos_9_eq_2_717_l228_228028


namespace multiplicative_inverse_modulo_l228_228997

noncomputable def A := 123456
noncomputable def B := 153846
noncomputable def N := 500000

theorem multiplicative_inverse_modulo :
  (A * B * N) % 1000000 = 1 % 1000000 :=
by
  sorry

end multiplicative_inverse_modulo_l228_228997


namespace distinct_three_digit_numbers_count_l228_228098

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l228_228098


namespace street_sweeper_routes_l228_228959

def num_routes (A B C : Type) :=
  -- Conditions: Starts from point A, 
  -- travels through all streets exactly once, 
  -- and returns to point A.
  -- Correct Answer: Total routes = 12
  2 * 6 = 12

theorem street_sweeper_routes (A B C : Type) : num_routes A B C := by
  -- The proof is omitted as per instructions
  sorry

end street_sweeper_routes_l228_228959


namespace left_square_side_length_l228_228746

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l228_228746


namespace bricklayer_wall_l228_228668

/-- 
A bricklayer lays a certain number of meters of wall per day and works for a certain number of days.
Given the daily work rate and the number of days worked, this proof shows that the total meters of 
wall laid equals the product of the daily work rate and the number of days.
-/
theorem bricklayer_wall (daily_rate : ℕ) (days_worked : ℕ) (total_meters : ℕ) 
  (h1 : daily_rate = 8) (h2 : days_worked = 15) : total_meters = 120 :=
by {
  sorry
}

end bricklayer_wall_l228_228668


namespace polynomial_solution_l228_228808

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x, (x + 2019) * (P.eval x) = x * (P.eval (x + 1))) :
  ∃ C : ℝ, P = Polynomial.C C * Polynomial.X * (Polynomial.X + 2018) :=
sorry

end polynomial_solution_l228_228808


namespace car_catches_up_in_6_hours_l228_228123

-- Conditions
def speed_truck := 40 -- km/h
def speed_car_initial := 50 -- km/h
def speed_car_increment := 5 -- km/h
def distance_between := 135 -- km

-- Solution: car catches up in 6 hours
theorem car_catches_up_in_6_hours : 
  ∃ n : ℕ, n = 6 ∧ (n * speed_truck + distance_between) ≤ (n * speed_car_initial + (n * (n - 1) / 2 * speed_car_increment)) := 
by
  sorry

end car_catches_up_in_6_hours_l228_228123


namespace cost_of_scissor_l228_228249

noncomputable def scissor_cost (initial_money: ℕ) (scissors: ℕ) (eraser_count: ℕ) (eraser_cost: ℕ) (remaining_money: ℕ) :=
  (initial_money - remaining_money - (eraser_count * eraser_cost)) / scissors

theorem cost_of_scissor : scissor_cost 100 8 10 4 20 = 5 := 
by 
  sorry 

end cost_of_scissor_l228_228249


namespace initial_soccer_balls_l228_228711

theorem initial_soccer_balls {x : ℕ} (h : x + 18 = 24) : x = 6 := 
sorry

end initial_soccer_balls_l228_228711


namespace ring_toss_total_earnings_l228_228256

noncomputable def daily_earnings : ℕ := 144
noncomputable def number_of_days : ℕ := 22
noncomputable def total_earnings : ℕ := daily_earnings * number_of_days

theorem ring_toss_total_earnings :
  total_earnings = 3168 := by
  sorry

end ring_toss_total_earnings_l228_228256


namespace number_of_participants_l228_228527

theorem number_of_participants (total_gloves : ℕ) (gloves_per_participant : ℕ)
  (h : total_gloves = 126) (h' : gloves_per_participant = 2) : 
  (total_gloves / gloves_per_participant = 63) :=
by
  sorry

end number_of_participants_l228_228527


namespace slope_of_line_l228_228849

theorem slope_of_line (θ : ℝ) (h_cosθ : (Real.cos θ) = 4/5) : (Real.sin θ) / (Real.cos θ) = 3/4 :=
by
  sorry

end slope_of_line_l228_228849


namespace difference_of_roots_l228_228309

theorem difference_of_roots (r1 r2 : ℝ) 
    (h_eq : ∀ x : ℝ, x^2 - 9 * x + 4 = 0 ↔ x = r1 ∨ x = r2) : 
    abs (r1 - r2) = Real.sqrt 65 := 
sorry

end difference_of_roots_l228_228309


namespace max_abs_asin_b_l228_228045

theorem max_abs_asin_b (a b c : ℝ) (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) :
  ∃ M : ℝ, (∀ x : ℝ, |a * Real.sin x + b| ≤ M) ∧ M = 2 :=
sorry

end max_abs_asin_b_l228_228045


namespace profit_percentage_is_25_percent_l228_228567

noncomputable def costPrice : ℝ := 47.50
noncomputable def markedPrice : ℝ := 64.54
noncomputable def discountRate : ℝ := 0.08

noncomputable def discountAmount : ℝ := discountRate * markedPrice
noncomputable def sellingPrice : ℝ := markedPrice - discountAmount
noncomputable def profit : ℝ := sellingPrice - costPrice
noncomputable def profitPercentage : ℝ := (profit / costPrice) * 100

theorem profit_percentage_is_25_percent :
  profitPercentage = 25 := by
  sorry

end profit_percentage_is_25_percent_l228_228567


namespace complex_division_l228_228325

noncomputable def imagine_unit : ℂ := Complex.I

theorem complex_division :
  (Complex.mk (-3) 1) / (Complex.mk 1 (-1)) = (Complex.mk (-2) 1) :=
by
sorry

end complex_division_l228_228325


namespace burpees_percentage_contribution_l228_228548

theorem burpees_percentage_contribution :
  let total_time : ℝ := 20
  let jumping_jacks : ℝ := 30
  let pushups : ℝ := 22
  let situps : ℝ := 45
  let burpees : ℝ := 15
  let lunges : ℝ := 25

  let jumping_jacks_rate := jumping_jacks / total_time
  let pushups_rate := pushups / total_time
  let situps_rate := situps / total_time
  let burpees_rate := burpees / total_time
  let lunges_rate := lunges / total_time

  let total_rate := jumping_jacks_rate + pushups_rate + situps_rate + burpees_rate + lunges_rate

  (burpees_rate / total_rate) * 100 = 10.95 :=
by
  sorry

end burpees_percentage_contribution_l228_228548


namespace value_of_r_minus_p_l228_228865

variable (p q r : ℝ)

-- The conditions given as hypotheses
def arithmetic_mean_pq := (p + q) / 2 = 10
def arithmetic_mean_qr := (q + r) / 2 = 25

-- The goal is to prove that r - p = 30
theorem value_of_r_minus_p (h1: arithmetic_mean_pq p q) (h2: arithmetic_mean_qr q r) :
  r - p = 30 := by
  sorry

end value_of_r_minus_p_l228_228865


namespace committee_vote_change_l228_228587

-- Let x be the number of votes for the resolution initially.
-- Let y be the number of votes against the resolution initially.
-- The total number of voters is 500: x + y = 500.
-- The initial margin by which the resolution was defeated: y - x = m.
-- In the re-vote, the resolution passed with a margin three times the initial margin: x' - y' = 3m.
-- The number of votes for the re-vote was 13/12 of the votes against initially: x' = 13/12 * y.
-- The total number of voters remains 500 in the re-vote: x' + y' = 500.

theorem committee_vote_change (x y x' y' m : ℕ)
  (h1 : x + y = 500)
  (h2 : y - x = m)
  (h3 : x' - y' = 3 * m)
  (h4 : x' = 13 * y / 12)
  (h5 : x' + y' = 500) : x' - x = 40 := 
  by
  sorry

end committee_vote_change_l228_228587


namespace radiator_initial_fluid_l228_228871

theorem radiator_initial_fluid (x : ℝ)
  (h1 : (0.10 * x - 0.10 * 2.2857 + 0.80 * 2.2857) = 0.50 * x) :
  x = 4 :=
sorry

end radiator_initial_fluid_l228_228871


namespace math_problem_l228_228877

theorem math_problem (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -5) : x + y = -16/9 := 
sorry

end math_problem_l228_228877


namespace random_event_is_option_D_l228_228538

-- Definitions based on conditions
def rains_without_clouds : Prop := false
def like_charges_repel : Prop := true
def seeds_germinate_without_moisture : Prop := false
def draw_card_get_1 : Prop := true

-- Proof statement
theorem random_event_is_option_D : 
  (¬ rains_without_clouds ∧ like_charges_repel ∧ ¬ seeds_germinate_without_moisture ∧ draw_card_get_1) →
  (draw_card_get_1 = true) :=
by sorry

end random_event_is_option_D_l228_228538


namespace find_ck_l228_228591

theorem find_ck 
  (d r : ℕ)                -- d : common difference, r : common ratio
  (k : ℕ)                  -- k : integer such that certain conditions hold
  (hn2 : (k-2) > 0)        -- ensure (k-2) > 0
  (hk1 : (k+1) > 0)        -- ensure (k+1) > 0
  (h1 : 1 + (k-3) * d + r^(k-3) = 120) -- c_{k-1} = 120
  (h2 : 1 + k * d + r^k = 1200) -- c_{k+1} = 1200
  : (1 + (k-1) * d + r^(k-1)) = 263 := -- c_k = 263
sorry

end find_ck_l228_228591


namespace probability_nan_kai_l228_228647

theorem probability_nan_kai :
  let total_outcomes := Nat.choose 6 4
  let successful_outcomes := Nat.choose 4 4
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 1 / 15 :=
by
  sorry

end probability_nan_kai_l228_228647


namespace carpet_rate_l228_228485

theorem carpet_rate (length breadth cost area: ℝ) (h₁ : length = 13) (h₂ : breadth = 9) (h₃ : cost = 1872) (h₄ : area = length * breadth) :
  cost / area = 16 := by
  sorry

end carpet_rate_l228_228485


namespace simplify_fraction_l228_228941

theorem simplify_fraction (k : ℝ) : 
  (∃ a b : ℝ, (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a = 1 ∧ b = 3 ∧ (a / b) = 1/3) := by
  sorry

end simplify_fraction_l228_228941


namespace part1_part2_l228_228330

open Set

variable (a : ℝ)

def real_universe := @univ ℝ

def set_A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def set_B : Set ℝ := {x | 2 < x ∧ x < 10}
def set_C (a : ℝ) : Set ℝ := {x | x ≤ a}

noncomputable def complement_A := (real_universe \ set_A)

theorem part1 : (complement_A ∩ set_B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } :=
by sorry

theorem part2 : set_A ⊆ set_C a → a > 7 :=
by sorry

end part1_part2_l228_228330


namespace find_function_p_t_additional_hours_l228_228222

variable (p0 : ℝ) (t k : ℝ)

-- Given condition: initial concentration decreased by 1/5 after one hour
axiom filtration_condition_1 : (p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)))
axiom filtration_condition_2 : (p0 * ((4 : ℝ) / 5) = p0 * (Real.exp (-k)))

-- Problem 1: Find the function p(t)
theorem find_function_p_t : ∃ k, ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)) := by
  sorry

-- Problem 2: Find the additional hours of filtration needed
theorem additional_hours (h : ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t))) :
  ∀ t, p0 * ((4 : ℝ) / 5) ^ t ≤ (p0 / 1000) → t ≥ 30 := by
  sorry

end find_function_p_t_additional_hours_l228_228222


namespace quadratic_equation_in_x_l228_228585

theorem quadratic_equation_in_x (m : ℤ) (h1 : abs m = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
sorry

end quadratic_equation_in_x_l228_228585


namespace total_baseball_fans_l228_228005

theorem total_baseball_fans (Y M B : ℕ)
  (h1 : Y = 3 / 2 * M)
  (h2 : M = 88)
  (h3 : B = 5 / 4 * M) :
  Y + M + B = 330 :=
by
  sorry

end total_baseball_fans_l228_228005


namespace ternary_1021_to_decimal_l228_228674

-- Define the function to convert a ternary string to decimal
def ternary_to_decimal (n : String) : Nat :=
  n.foldr (fun c acc => acc * 3 + (c.toNat - '0'.toNat)) 0

-- The statement to prove
theorem ternary_1021_to_decimal : ternary_to_decimal "1021" = 34 := by
  sorry

end ternary_1021_to_decimal_l228_228674


namespace probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l228_228976

noncomputable def germination_rate : ℝ := 0.9
noncomputable def non_germination_rate : ℝ := 1 - germination_rate
noncomputable def strong_seedling_rate : ℝ := 0.6
noncomputable def non_strong_seedling_rate : ℝ := 1 - strong_seedling_rate

theorem probability_two_seeds_missing_seedlings :
  (non_germination_rate ^ 2) = 0.01 := sorry

theorem probability_two_seeds_no_strong_seedlings :
  (non_strong_seedling_rate ^ 2) = 0.16 := sorry

theorem probability_three_seeds_having_seedlings :
  (1 - non_germination_rate ^ 3) = 0.999 := sorry

theorem probability_three_seeds_having_strong_seedlings :
  (1 - non_strong_seedling_rate ^ 3) = 0.936 := sorry

end probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l228_228976


namespace find_james_number_l228_228474

theorem find_james_number (x : ℝ) 
  (h1 : 3 * (3 * x + 10) = 141) : 
  x = 12.33 :=
by 
  sorry

end find_james_number_l228_228474


namespace intersection_eq_l228_228878

def A := {x : ℝ | |x| = x}
def B := {x : ℝ | x^2 + x ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | 0 ≤ x} := by
  sorry

end intersection_eq_l228_228878


namespace domain_of_h_l228_228194

def domain_f : Set ℝ := {x | -10 ≤ x ∧ x ≤ 3}

def h_dom := {x | -3 * x ∈ domain_f}

theorem domain_of_h :
  h_dom = {x | x ≥ 10 / 3} :=
by
  sorry

end domain_of_h_l228_228194


namespace katrina_cookies_left_l228_228413

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l228_228413


namespace total_candidates_l228_228220

def average_marks_all_candidates : ℕ := 35
def average_marks_passed_candidates : ℕ := 39
def average_marks_failed_candidates : ℕ := 15
def passed_candidates : ℕ := 100

theorem total_candidates (T : ℕ) (F : ℕ) 
  (h1 : 35 * T = 39 * passed_candidates + 15 * F)
  (h2 : T = passed_candidates + F) : T = 120 := 
  sorry

end total_candidates_l228_228220


namespace ratio_of_a_to_b_l228_228389

-- Given conditions
variables {a b x : ℝ}
-- a and b are positive real numbers distinct from 1
variables (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1)
-- Given equation involving logarithms
variables (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2)

-- Prove that the ratio of a to b is a^(sqrt(7/5))
theorem ratio_of_a_to_b (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1) (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2) :
  b = a ^ Real.sqrt (7 / 5) :=
sorry

end ratio_of_a_to_b_l228_228389


namespace trigo_identity_l228_228903

variable (α : ℝ)

theorem trigo_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (Real.pi / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end trigo_identity_l228_228903


namespace determine_cards_per_friend_l228_228563

theorem determine_cards_per_friend (n_cards : ℕ) (n_friends : ℕ) (h : n_cards = 12) : n_friends > 0 → (n_cards / n_friends) = (12 / n_friends) :=
by
  sorry

end determine_cards_per_friend_l228_228563


namespace container_volumes_l228_228842

variable (a : ℕ)

theorem container_volumes (h₁ : a = 18) :
  a^3 = 5832 ∧ (a - 4)^3 = 2744 ∧ (a - 6)^3 = 1728 :=
by {
  sorry
}

end container_volumes_l228_228842


namespace find_integer_l228_228772

theorem find_integer (x : ℕ) (h : (4 * x) ^ 2 - 3 * x = 1764) : x = 18 := 
by 
  sorry

end find_integer_l228_228772


namespace multiplication_decomposition_l228_228990

theorem multiplication_decomposition :
  100 * 3 = 100 + 100 + 100 :=
sorry

end multiplication_decomposition_l228_228990


namespace max_levels_passed_prob_pass_three_levels_l228_228099

-- Define the conditions of the game
def max_roll (n : ℕ) : ℕ := 6 * n
def pass_condition (n : ℕ) : ℕ := 2^n

-- Problem 1: Prove the maximum number of levels a person can pass
theorem max_levels_passed : ∃ n : ℕ, (∀ m : ℕ, m > n → max_roll m ≤ pass_condition m) ∧ (∀ m : ℕ, m ≤ n → max_roll m > pass_condition m) :=
by sorry

-- Define the probabilities for passing each level
def prob_pass_level_1 : ℚ := 4 / 6
def prob_pass_level_2 : ℚ := 30 / 36
def prob_pass_level_3 : ℚ := 160 / 216

-- Problem 2: Prove the probability of passing the first three levels consecutively
theorem prob_pass_three_levels : prob_pass_level_1 * prob_pass_level_2 * prob_pass_level_3 = 100 / 243 :=
by sorry

end max_levels_passed_prob_pass_three_levels_l228_228099


namespace trig_identity_example_l228_228062

theorem trig_identity_example (α : Real) (h : Real.cos α = 3 / 5) : Real.cos (2 * α) + Real.sin α ^ 2 = 9 / 25 := by
  sorry

end trig_identity_example_l228_228062


namespace derivative_of_sin_squared_is_sin_2x_l228_228339

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared_is_sin_2x : 
  ∀ x : ℝ, deriv f x = sin (2 * x) :=
by
  sorry

end derivative_of_sin_squared_is_sin_2x_l228_228339


namespace tunnel_length_correct_l228_228813

noncomputable def length_of_tunnel
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time_s := crossing_time_min * 60
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem tunnel_length_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  -- The proof will be filled later.
  sorry

end tunnel_length_correct_l228_228813


namespace trig_identity_l228_228411

open Real

theorem trig_identity (α : ℝ) (h : tan α = -1/2) : 1 - sin (2 * α) = 9/5 := 
  sorry

end trig_identity_l228_228411


namespace car_dealership_sales_l228_228401

theorem car_dealership_sales (trucks_ratio suvs_ratio trucks_expected suvs_expected : ℕ)
  (h_ratio : trucks_ratio = 5 ∧ suvs_ratio = 8)
  (h_expected : trucks_expected = 35 ∧ suvs_expected = 56) :
  (trucks_ratio : ℚ) / suvs_ratio = (trucks_expected : ℚ) / suvs_expected :=
by
  sorry

end car_dealership_sales_l228_228401


namespace problem1_problem2_problem3_problem4_l228_228725

-- Define predicate conditions and solutions in Lean 4 for each problem

theorem problem1 (x : ℝ) :
  -2 * x^2 + 3 * x + 9 > 0 ↔ (-3 / 2 < x ∧ x < 3) := by
  sorry

theorem problem2 (x : ℝ) :
  (8 - x) / (5 + x) > 1 ↔ (-5 < x ∧ x ≤ 3 / 2) := by
  sorry

theorem problem3 (x : ℝ) :
  ¬ (-x^2 + 2 * x - 3 > 0) ↔ True := by
  sorry

theorem problem4 (x : ℝ) :
  x^2 - 14 * x + 50 > 0 ↔ True := by
  sorry

end problem1_problem2_problem3_problem4_l228_228725


namespace remaining_volume_l228_228342

-- Given
variables (a d : ℚ) 
-- Define the volumes of sections as arithmetic sequence terms
def volume (n : ℕ) := a + n*d

-- Define total volume of bottom three sections
def bottomThreeVolume := volume a 0 + volume a d + volume a (2 * d) = 4

-- Define total volume of top four sections
def topFourVolume := volume a (5 * d) + volume a (6 * d) + volume a (7 * d) + volume a (8 * d) = 3

-- Define the volumes of the two middle sections
def middleTwoVolume := volume a (3 * d) + volume a (4 * d) = 2 + 3 / 22

-- Prove that the total volume of the remaining two sections is 2 3/22
theorem remaining_volume : bottomThreeVolume a d ∧ topFourVolume a d → middleTwoVolume a d :=
sorry  -- Placeholder for the actual proof

end remaining_volume_l228_228342


namespace coeffs_sum_eq_40_l228_228769

theorem coeffs_sum_eq_40 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (2 * x - 1) ^ 5 = a_0 * x ^ 5 + a_1 * x ^ 4 + a_2 * x ^ 3 + a_3 * x ^ 2 + a_4 * x + a_5) :
  a_2 + a_3 = 40 :=
sorry

end coeffs_sum_eq_40_l228_228769


namespace fraction_difference_l228_228025

theorem fraction_difference :
  (↑(1+4+7) / ↑(2+5+8)) - (↑(2+5+8) / ↑(1+4+7)) = - (9 / 20) :=
by
  sorry

end fraction_difference_l228_228025


namespace inequality_proof_l228_228010

theorem inequality_proof (a b x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l228_228010


namespace smallest_sum_arith_geo_sequence_l228_228043

theorem smallest_sum_arith_geo_sequence :
  ∃ (X Y Z W : ℕ),
    X < Y ∧ Y < Z ∧ Z < W ∧
    (2 * Y = X + Z) ∧
    (Y ^ 2 = Z * X) ∧
    (Z / Y = 7 / 4) ∧
    (X + Y + Z + W = 97) :=
by
  sorry

end smallest_sum_arith_geo_sequence_l228_228043


namespace relationship_abcd_l228_228000

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end relationship_abcd_l228_228000


namespace Cassini_l228_228356

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

-- State Cassini's Identity theorem
theorem Cassini (n : ℕ) : Fibonacci (n + 1) * Fibonacci (n - 1) - (Fibonacci n) ^ 2 = (-1) ^ n := 
by sorry

end Cassini_l228_228356


namespace police_emergency_number_prime_divisor_l228_228720

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l228_228720


namespace sum_base6_l228_228015

theorem sum_base6 (a b c : ℕ) 
  (ha : a = 1 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 1 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hc : c = 1 * 6^1 + 5 * 6^0) :
  a + b + c = 2 * 6^3 + 2 * 6^2 + 0 * 6^1 + 3 * 6^0 :=
by 
  sorry

end sum_base6_l228_228015


namespace find_divisor_of_115_l228_228190

theorem find_divisor_of_115 (x : ℤ) (N : ℤ)
  (hN : N = 115)
  (h1 : N % 38 = 1)
  (h2 : N % x = 1) :
  x = 57 :=
by
  sorry

end find_divisor_of_115_l228_228190


namespace initial_honey_amount_l228_228451

variable (H : ℝ)

theorem initial_honey_amount :
  (0.70 * 0.60 * 0.50) * H = 315 → H = 1500 :=
by
  sorry

end initial_honey_amount_l228_228451


namespace dice_arithmetic_progression_l228_228873

theorem dice_arithmetic_progression :
  let valid_combinations := [
     (1, 1, 1), (1, 3, 2), (1, 5, 3), 
     (2, 4, 3), (2, 6, 4), (3, 3, 3),
     (3, 5, 4), (4, 6, 5), (5, 5, 5)
  ]
  (valid_combinations.length : ℚ) / (6^3 : ℚ) = 1 / 24 :=
  sorry

end dice_arithmetic_progression_l228_228873


namespace triangle_side_length_l228_228371

theorem triangle_side_length 
  (side1 : ℕ) (side2 : ℕ) (side3 : ℕ) (P : ℕ)
  (h_side1 : side1 = 5)
  (h_side3 : side3 = 30)
  (h_P : P = 55) :
  side1 + side2 + side3 = P → side2 = 20 :=
by
  intros h
  sorry 

end triangle_side_length_l228_228371


namespace R_l228_228240

variable (a d n : ℕ)

def arith_sum (k : ℕ) : ℕ :=
  k * (a + (k - 1) * d / 2)

def s1 := arith_sum n
def s2 := arith_sum (3 * n)
def s3 := arith_sum (5 * n)
def s4 := arith_sum (7 * n)

def R' := s4 - s3 - s2

theorem R'_depends_on_d_n : 
  R' = 2 * d * n^2 := 
by 
  sorry

end R_l228_228240


namespace find_Y_l228_228100

theorem find_Y (Y : ℝ) (h : (100 + Y / 90) * 90 = 9020) : Y = 20 := 
by 
  sorry

end find_Y_l228_228100


namespace cricket_team_captain_age_l228_228898

theorem cricket_team_captain_age
    (C W : ℕ)
    (h1 : W = C + 3)
    (h2 : (23 * 11) = (22 * 9) + C + W)
    : C = 26 :=
by
    sorry

end cricket_team_captain_age_l228_228898


namespace ratio_of_wire_lengths_l228_228300

theorem ratio_of_wire_lengths (b_pieces : ℕ) (b_piece_length : ℕ)
  (c_piece_length : ℕ) (cubes_volume : ℕ) :
  b_pieces = 12 →
  b_piece_length = 8 →
  c_piece_length = 2 →
  cubes_volume = (b_piece_length ^ 3) →
  b_pieces * b_piece_length * cubes_volume
    / (cubes_volume * (12 * c_piece_length)) = (1 / 128) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_wire_lengths_l228_228300


namespace CarmenBrushLengthIsCorrect_l228_228472

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end CarmenBrushLengthIsCorrect_l228_228472


namespace hyperbola_foci_y_axis_l228_228670

theorem hyperbola_foci_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → (1/a < 0 ∧ 1/b > 0)) : a < 0 ∧ b > 0 :=
by
  sorry

end hyperbola_foci_y_axis_l228_228670


namespace jelly_beans_in_jar_X_l228_228233

theorem jelly_beans_in_jar_X : 
  ∀ (X Y : ℕ), (X + Y = 1200) → (X = 3 * Y - 400) → X = 800 :=
by
  sorry

end jelly_beans_in_jar_X_l228_228233


namespace fraction_simplification_l228_228454

-- Define the numerator and denominator based on given conditions
def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

-- Lean theorem that encapsulates the problem
theorem fraction_simplification : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_simplification_l228_228454


namespace general_term_a_n_sum_of_b_n_l228_228322

-- Proof Problem 1: General term of sequence {a_n}
theorem general_term_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4) 
    (h3 : ∀ n ≥ 2, a (n+1) - a n = 2) : 
    ∀ n, a n = 2 * n :=
by
  sorry

-- Proof Problem 2: Sum of the first n terms of sequence {b_n}
theorem sum_of_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
    (h : ∀ n, (1 / (a n ^ 2 - 1) : ℝ) + b n = 2^n) :
    T n = 2^(n+1) - n / (2*n + 1) :=
by
  sorry

end general_term_a_n_sum_of_b_n_l228_228322


namespace students_taking_only_science_l228_228631

theorem students_taking_only_science (total_students : ℕ) (students_science : ℕ) (students_math : ℕ)
  (h1 : total_students = 120) (h2 : students_science = 80) (h3 : students_math = 75) :
  (students_science - (students_science + students_math - total_students)) = 45 :=
by
  sorry

end students_taking_only_science_l228_228631


namespace fraction_of_widgets_second_shift_l228_228492

theorem fraction_of_widgets_second_shift (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let first_shift_widgets := x * y
  let second_shift_widgets := (2 / 3) * x * (4 / 3) * y
  let total_widgets := first_shift_widgets + second_shift_widgets
  let fraction_second_shift := second_shift_widgets / total_widgets
  fraction_second_shift = 8 / 17 :=
by
  sorry

end fraction_of_widgets_second_shift_l228_228492


namespace robert_monthly_expenses_l228_228981

def robert_basic_salary : ℝ := 1250
def robert_sales : ℝ := 23600
def first_tier_limit : ℝ := 10000
def second_tier_limit : ℝ := 20000
def first_tier_rate : ℝ := 0.10
def second_tier_rate : ℝ := 0.12
def third_tier_rate : ℝ := 0.15
def savings_rate : ℝ := 0.20

def first_tier_commission : ℝ :=
  first_tier_limit * first_tier_rate

def second_tier_commission : ℝ :=
  (second_tier_limit - first_tier_limit) * second_tier_rate

def third_tier_commission : ℝ :=
  (robert_sales - second_tier_limit) * third_tier_rate

def total_commission : ℝ :=
  first_tier_commission + second_tier_commission + third_tier_commission

def total_earnings : ℝ :=
  robert_basic_salary + total_commission

def savings : ℝ :=
  total_earnings * savings_rate

def monthly_expenses : ℝ :=
  total_earnings - savings

theorem robert_monthly_expenses :
  monthly_expenses = 3192 := by
  sorry

end robert_monthly_expenses_l228_228981


namespace fraction_goldfish_preference_l228_228108

theorem fraction_goldfish_preference
  (students_per_class : ℕ)
  (students_prefer_golfish_miss_johnson : ℕ)
  (students_prefer_golfish_ms_henderson : ℕ)
  (students_prefer_goldfish_total : ℕ)
  (miss_johnson_fraction : ℚ)
  (ms_henderson_fraction : ℚ)
  (total_students_prefer_goldfish_feldstein : ℕ)
  (feldstein_fraction : ℚ) :
  miss_johnson_fraction = 1/6 ∧
  ms_henderson_fraction = 1/5 ∧
  students_per_class = 30 ∧
  students_prefer_golfish_miss_johnson = miss_johnson_fraction * students_per_class ∧
  students_prefer_golfish_ms_henderson = ms_henderson_fraction * students_per_class ∧
  students_prefer_goldfish_total = 31 ∧
  students_prefer_goldfish_total = students_prefer_golfish_miss_johnson + students_prefer_golfish_ms_henderson + total_students_prefer_goldfish_feldstein ∧
  feldstein_fraction * students_per_class = total_students_prefer_goldfish_feldstein
  →
  feldstein_fraction = 2 / 3 :=
by 
  sorry

end fraction_goldfish_preference_l228_228108


namespace nancy_total_spending_l228_228723

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l228_228723


namespace sqrt_sum_of_fractions_l228_228021

theorem sqrt_sum_of_fractions :
  (Real.sqrt ((25 / 36) + (16 / 9)) = (Real.sqrt 89) / 6) :=
by
  sorry

end sqrt_sum_of_fractions_l228_228021


namespace parallel_vectors_solution_l228_228213

noncomputable def vector_a : (ℝ × ℝ) := (1, 2)
noncomputable def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -4)

def vectors_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_solution (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = -2 :=
sorry

end parallel_vectors_solution_l228_228213


namespace least_integer_value_abs_l228_228071

theorem least_integer_value_abs (x : ℤ) : 
  (∃ x : ℤ, (abs (3 * x + 5) ≤ 20) ∧ (∀ y : ℤ, (abs (3 * y + 5) ≤ 20) → x ≤ y)) ↔ x = -8 :=
by
  sorry

end least_integer_value_abs_l228_228071


namespace count_difference_l228_228732

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end count_difference_l228_228732


namespace initial_typists_count_l228_228431

theorem initial_typists_count 
  (typists_rate : ℕ → ℕ)
  (letters_in_20min : ℕ)
  (total_typists : ℕ)
  (letters_in_1hour : ℕ)
  (initial_typists : ℕ) 
  (h1 : letters_in_20min = 38)
  (h2 : letters_in_1hour = 171)
  (h3 : total_typists = 30)
  (h4 : ∀ t, 3 * (typists_rate t) = letters_in_1hour / total_typists)
  (h5 : ∀ t, typists_rate t = letters_in_20min / t) 
  : initial_typists = 20 := 
sorry

end initial_typists_count_l228_228431


namespace length_of_train_is_correct_l228_228932

noncomputable def speed_kmh := 30 
noncomputable def time_s := 9 
noncomputable def speed_ms := (speed_kmh * 1000) / 3600 
noncomputable def length_of_train := speed_ms * time_s

theorem length_of_train_is_correct : length_of_train = 75 := 
by 
  sorry

end length_of_train_is_correct_l228_228932


namespace problem_1_problem_2_l228_228239

-- (1) Conditions and proof statement
theorem problem_1 (x y m : ℝ) (P : ℝ × ℝ) (k : ℝ) :
  (x, y) = (1, 2) → m = 1 →
  ((x - 1)^2 + (y - 2)^2 = 4) →
  P = (3, -1) →
  (l : ℝ → ℝ → Prop) →
  (∀ x y, l x y ↔ x = 3 ∨ (5 * x + 12 * y - 3 = 0)) →
  l 3 (-1) →
  l (x + k * (3 - x)) (y-1) := sorry

-- (2) Conditions and proof statement
theorem problem_2 (x y m : ℝ) (line : ℝ → ℝ) :
  (x - 1)^2 + (y - 2)^2 = 5 - m →
  m < 5 →
  (2 * (5 - m - 20) ^ (1/2) = 2 * (5) ^ (1/2)) →
  m = -20 := sorry

end problem_1_problem_2_l228_228239


namespace log_eqn_l228_228649

theorem log_eqn (a b : ℝ) (h1 : a = (Real.log 400 / Real.log 16))
                          (h2 : b = Real.log 20 / Real.log 2) : a = (1/2) * b :=
sorry

end log_eqn_l228_228649


namespace range_of_p_l228_228041

def A := {x : ℝ | x^2 - x - 2 > 0}
def B := {x : ℝ | (3 / x) - 1 ≥ 0}
def intersection := {x : ℝ | x ∈ A ∧ x ∈ B}
def C (p : ℝ) := {x : ℝ | 2 * x + p ≤ 0}

theorem range_of_p (p : ℝ) : (∀ x : ℝ, x ∈ intersection → x ∈ C p) → p < -6 := by
  sorry

end range_of_p_l228_228041


namespace vertex_of_parabola_find_shift_m_l228_228791

-- Problem 1: Vertex of the given parabola
theorem vertex_of_parabola : 
  ∃ x y: ℝ, (y = 2 * x^2 + 4 * x - 6) ∧ (x, y) = (-1, -8) := 
by
  -- Proof goes here
  sorry

-- Problem 2: Finding the shift m
theorem find_shift_m (m : ℝ) (h : m > 0) : 
  (∀ x (hx : (x = (x + m)) ∧ (2 * x^2 + 4 * x - 6 = 0)), x = 1 ∨ x = -3) ∧ 
  ((-3 + m) = 0) → m = 3 :=
by
  -- Proof goes here
  sorry

end vertex_of_parabola_find_shift_m_l228_228791


namespace figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l228_228214

-- Given conditions:
def initial_squares : ℕ := 3
def initial_perimeter : ℕ := 8
def squares_per_step : ℕ := 2
def perimeter_per_step : ℕ := 4

-- Statement proving Figure 8 has 17 squares
theorem figure8_squares : 3 + 2 * (8 - 1) = 17 := by sorry

-- Statement proving Figure 12 has a perimeter of 52 cm
theorem figure12_perimeter : 8 + 4 * (12 - 1) = 52 := by sorry

-- Statement proving no positive integer C yields perimeter of 38 cm
theorem no_figure_C : ¬∃ C : ℕ, 8 + 4 * (C - 1) = 38 := by sorry
  
-- Statement proving closest D giving the ratio for perimeter between Figure 29 and Figure D
theorem figure29_figureD_ratio : (8 + 4 * (29 - 1)) * 11 = 4 * (8 + 4 * (81 - 1)) := by sorry

end figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l228_228214


namespace determine_x_l228_228452

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l228_228452


namespace combined_bus_rides_length_l228_228495

theorem combined_bus_rides_length :
  let v := 0.62
  let z := 0.5
  let a := 0.72
  v + z + a = 1.84 :=
by
  let v := 0.62
  let z := 0.5
  let a := 0.72
  show v + z + a = 1.84
  sorry

end combined_bus_rides_length_l228_228495


namespace tan_periodic_mod_l228_228918

theorem tan_periodic_mod (m : ℤ) (h1 : -180 < m) (h2 : m < 180) : 
  (m : ℤ) = 10 := by
  sorry

end tan_periodic_mod_l228_228918


namespace tuition_fee_l228_228151

theorem tuition_fee (R T : ℝ) (h1 : T + R = 2584) (h2 : T = R + 704) : T = 1644 := by sorry

end tuition_fee_l228_228151


namespace rectangle_ratio_of_semicircles_l228_228731

theorem rectangle_ratio_of_semicircles (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h : a * b = π * b^2) : a / b = π := by
  sorry

end rectangle_ratio_of_semicircles_l228_228731


namespace completed_shape_perimeter_602_l228_228077

noncomputable def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

noncomputable def total_perimeter_no_overlap (n : ℕ) (length width : ℝ) : ℝ :=
  n * rectangle_perimeter length width

noncomputable def total_reduction (n : ℕ) (overlap : ℝ) : ℝ :=
  (n - 1) * overlap

noncomputable def overall_perimeter (n : ℕ) (length width overlap : ℝ) : ℝ :=
  total_perimeter_no_overlap n length width - total_reduction n overlap

theorem completed_shape_perimeter_602 :
  overall_perimeter 100 3 1 2 = 602 :=
by
  sorry

end completed_shape_perimeter_602_l228_228077


namespace marys_income_percent_of_juans_income_l228_228490

variables (M T J : ℝ)

theorem marys_income_percent_of_juans_income (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marys_income_percent_of_juans_income_l228_228490


namespace prob1_prob2_odd_prob2_monotonic_prob3_l228_228385

variable (a : ℝ) (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, f (log a x) = a / (a^2 - 1) * (x - 1 / x))
variable (ha : 0 < a ∧ a < 1)

-- Problem 1: Prove the expression for f(x)
theorem prob1 (x : ℝ) : f x = a / (a^2 - 1) * (a^x - a^(-x)) := sorry

-- Problem 2: Prove oddness and monotonicity of f(x)
theorem prob2_odd : ∀ x, f (-x) = -f x := sorry
theorem prob2_monotonic : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ < f x₂) := sorry

-- Problem 3: Determine the range of k
theorem prob3 (k : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → f (3 * t^2 - 1) + f (4 * t - k) > 0) → (k < 6) := sorry

end prob1_prob2_odd_prob2_monotonic_prob3_l228_228385


namespace smallest_norm_l228_228708

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end smallest_norm_l228_228708


namespace sum_of_squares_l228_228115

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_l228_228115


namespace completion_days_together_l228_228911

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

end completion_days_together_l228_228911


namespace remaining_standby_time_l228_228403

variable (fully_charged_standby : ℝ) (fully_charged_gaming : ℝ)
variable (standby_time : ℝ) (gaming_time : ℝ)

theorem remaining_standby_time
  (h1 : fully_charged_standby = 10)
  (h2 : fully_charged_gaming = 2)
  (h3 : standby_time = 4)
  (h4 : gaming_time = 1.5) :
  (10 - ((standby_time * (1 / fully_charged_standby)) + (gaming_time * (1 / fully_charged_gaming)))) * 10 = 1 :=
by
  sorry

end remaining_standby_time_l228_228403


namespace value_of_a_plus_b_l228_228841

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem value_of_a_plus_b (a b : ℝ) (h1 : 3 * a + b = 4) (h2 : a + b + 1 = 3) : a + b = 2 :=
by
  sorry

end value_of_a_plus_b_l228_228841


namespace dice_sum_not_possible_l228_228386

theorem dice_sum_not_possible (a b c d : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) 
(h₃ : 1 ≤ c ∧ c ≤ 6) (h₄ : 1 ≤ d ∧ d ≤ 6) (h_product : a * b * c * d = 216) : 
(a + b + c + d ≠ 15) ∧ (a + b + c + d ≠ 16) ∧ (a + b + c + d ≠ 18) :=
sorry

end dice_sum_not_possible_l228_228386


namespace john_vegetables_used_l228_228899

noncomputable def pounds_of_beef_bought : ℕ := 4
noncomputable def pounds_of_beef_used : ℕ := pounds_of_beef_bought - 1
noncomputable def pounds_of_vegetables_used : ℕ := 2 * pounds_of_beef_used

theorem john_vegetables_used : pounds_of_vegetables_used = 6 :=
by
  -- the proof can be provided here later
  sorry

end john_vegetables_used_l228_228899


namespace volleyball_ranking_l228_228106

-- Define type for place
inductive Place where
  | first : Place
  | second : Place
  | third : Place

-- Define type for teams
inductive Team where
  | A : Team
  | B : Team
  | C : Team

open Place Team

-- Given conditions as hypotheses
def LiMing_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p first A ∨ p third A) ∧ (p first B ∨ p third B) ∧ 
  ¬ (p first A ∧ p third A) ∧ ¬ (p first B ∧ p third B)

def ZhangHua_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p third A ∨ p first C) ∧ (p third A ∨ p first A) ∧ 
  ¬ (p third A ∧ p first A) ∧ ¬ (p first C ∧ p third C)

def WangQiang_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p second C ∨ p third B) ∧ (p second C ∨ p third C) ∧ 
  ¬ (p second C ∧ p third C) ∧ ¬ (p third B ∧ p second B)

-- Final proof problem
theorem volleyball_ranking (p : Place → Team → Prop) :
    (LiMing_prediction_half_correct p) →
    (ZhangHua_prediction_half_correct p) →
    (WangQiang_prediction_half_correct p) →
    p first C ∧ p second A ∧ p third B :=
  by
    sorry

end volleyball_ranking_l228_228106


namespace seashells_problem_l228_228107

theorem seashells_problem
  (F : ℕ)
  (h : (150 - F) / 2 = 55) :
  F = 40 :=
  sorry

end seashells_problem_l228_228107


namespace tan_alpha_20_l228_228295

theorem tan_alpha_20 (α : ℝ) 
  (h : Real.tan (α + 80 * Real.pi / 180) = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (α + 20 * Real.pi / 180) = Real.sqrt 3 / 7 := 
sorry

end tan_alpha_20_l228_228295


namespace mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l228_228081

def card_is_heart (c : ℕ) := c ≥ 1 ∧ c ≤ 13

def card_is_diamond (c : ℕ) := c ≥ 14 ∧ c ≤ 26

def card_is_red (c : ℕ) := c ≥ 1 ∧ c ≤ 26

def card_is_black (c : ℕ) := c ≥ 27 ∧ c ≤ 52

def card_is_face_234610 (c : ℕ) := c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6 ∨ c = 10

def card_is_face_2345678910 (c : ℕ) :=
  c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10

def card_is_face_AKQJ (c : ℕ) :=
  c = 1 ∨ c = 11 ∨ c = 12 ∨ c = 13

def card_is_ace_king_queen_jack (c : ℕ) := c = 1 ∨ (c ≥ 11 ∧ c ≤ 13)

theorem mutually_exclusive_pair2 : ∀ c : ℕ, card_is_red c ≠ card_is_black c := by
  sorry

theorem complementary_pair2 : ∀ c : ℕ, card_is_red c ∨ card_is_black c := by
  sorry

theorem mutually_exclusive_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ≠ card_is_ace_king_queen_jack c := by
  sorry

theorem complementary_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ∨ card_is_ace_king_queen_jack c := by
  sorry

end mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l228_228081


namespace Carol_max_chance_l228_228281

-- Definitions of the conditions
def Alice_random_choice (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def Bob_random_choice (b : ℝ) : Prop := 0.4 ≤ b ∧ b ≤ 0.6
def Carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Statement that Carol maximizes her chances by picking 0.5
theorem Carol_max_chance : ∃ c : ℝ, (∀ a b : ℝ, Alice_random_choice a → Bob_random_choice b → Carol_wins a b c) ∧ c = 0.5 := 
sorry

end Carol_max_chance_l228_228281


namespace percentage_of_fair_haired_employees_who_are_women_l228_228633

variable (E : ℝ) -- Total number of employees
variable (h1 : 0.1 * E = women_with_fair_hair_E) -- 10% of employees are women with fair hair
variable (h2 : 0.25 * E = fair_haired_employees_E) -- 25% of employees have fair hair

theorem percentage_of_fair_haired_employees_who_are_women :
  (women_with_fair_hair_E / fair_haired_employees_E) * 100 = 40 :=
by
  sorry

end percentage_of_fair_haired_employees_who_are_women_l228_228633


namespace no_such_two_digit_number_exists_l228_228554

theorem no_such_two_digit_number_exists :
  ¬ ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
                 (10 * x + y = 2 * (x^2 + y^2) + 6) ∧
                 (10 * x + y = 4 * (x * y) + 6) := by
  -- We need to prove that no two-digit number satisfies
  -- both conditions.
  sorry

end no_such_two_digit_number_exists_l228_228554


namespace exists_strictly_increasing_sequence_l228_228819

theorem exists_strictly_increasing_sequence 
  (N : ℕ) : 
  (∃ (t : ℕ), t^2 ≤ N ∧ N < t^2 + t) →
  (∃ (s : ℕ → ℕ), (∀ n : ℕ, s n < s (n + 1)) ∧ 
   (∃ k : ℕ, ∀ n : ℕ, s (n + 1) - s n = k) ∧
   (∀ n : ℕ, s (s n) - s (s (n - 1)) ≤ N 
      ∧ N < s (1 + s n) - s (s (n - 1)))) :=
by
  sorry

end exists_strictly_increasing_sequence_l228_228819


namespace term_in_sequence_l228_228635

   theorem term_in_sequence (n : ℕ) (h1 : 1 ≤ n) (h2 : 6 * n + 1 = 2005) : n = 334 :=
   by
     sorry
   
end term_in_sequence_l228_228635


namespace arithmetic_sequence_a15_value_l228_228397

variables {a : ℕ → ℤ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15_value
  (h1 : is_arithmetic_sequence a)
  (h2 : a 3 + a 13 = 20)
  (h3 : a 2 = -2) : a 15 = 24 :=
by sorry

end arithmetic_sequence_a15_value_l228_228397


namespace volume_tetrahedron_l228_228893

def A1 := 4^2
def A2 := 3^2
def h := 1

theorem volume_tetrahedron:
  (h / 3 * (A1 + A2 + Real.sqrt (A1 * A2))) = 37 / 3 := by
  sorry

end volume_tetrahedron_l228_228893


namespace total_pounds_of_peppers_l228_228416

-- Definitions and conditions
def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335

-- Theorem statement
theorem total_pounds_of_peppers : green_peppers + red_peppers = 5.666666666666667 :=
by
  sorry

end total_pounds_of_peppers_l228_228416


namespace quadratic_equal_real_roots_l228_228104

theorem quadratic_equal_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + m = 1 ∧ 
                              (∀ y : ℝ, y ≠ x → y^2 - 4 * y + m ≠ 1)) : m = 5 :=
by sorry

end quadratic_equal_real_roots_l228_228104


namespace inequality_d_l228_228685

-- We define the polynomial f with integer coefficients
variable (f : ℤ → ℤ)

-- The function for f^k iteration
def iter (f: ℤ → ℤ) : ℕ → ℤ → ℤ
| 0, x => x
| (n + 1), x => f (iter f n x)

-- Definition of d(a, k) based on the problem statement
def d (a : ℤ) (k : ℕ) : ℝ := |(iter f k a : ℤ) - a|

-- Given condition that d(a, k) is positive
axiom d_pos (a : ℤ) (k : ℕ) : 0 < d f a k

-- The statement to be proved
theorem inequality_d (a : ℤ) (k : ℕ) : d f a k ≥ ↑k / 3 := by
  sorry

end inequality_d_l228_228685


namespace probability_of_25_cents_heads_l228_228592

/-- 
Considering the flipping of five specific coins: a penny, a nickel, a dime,
a quarter, and a half dollar, prove that the probability of getting at least
25 cents worth of heads is 3 / 4.
-/
theorem probability_of_25_cents_heads :
  let total_outcomes := 2^5
  let successful_outcomes_1 := 2^4
  let successful_outcomes_2 := 2^3
  let successful_outcomes := successful_outcomes_1 + successful_outcomes_2
  (successful_outcomes / total_outcomes : ℚ) = 3 / 4 :=
by
  sorry

end probability_of_25_cents_heads_l228_228592


namespace proof_problem_l228_228406

noncomputable def a : ℝ := 0.85 * 250
noncomputable def b : ℝ := 0.75 * 180
noncomputable def c : ℝ := 0.90 * 320

theorem proof_problem :
  (a - b = 77.5) ∧ (77.5 < c) :=
by
  sorry

end proof_problem_l228_228406


namespace union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l228_228029

open Set

variable (a : ℝ)

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := univ

theorem union_A_B :
  A ∪ B = {x | 1 ≤ x ∧ x ≤ 8} := by
  sorry

theorem compl_A_inter_B :
  (U \ A) ∩ B = {x | 1 ≤ x ∧ x < 2} := by
  sorry

theorem intersection_A_C_not_empty :
  (A ∩ C a ≠ ∅) → a < 8 := by
  sorry

end union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l228_228029


namespace steve_final_amount_l228_228466

def initial_deposit : ℝ := 100
def interest_years_1_to_3 : ℝ := 0.10
def interest_years_4_to_5 : ℝ := 0.08
def annual_deposit_years_1_to_2 : ℝ := 10
def annual_deposit_years_3_to_5 : ℝ := 15

def total_after_one_year (initial : ℝ) (annual : ℝ) (interest : ℝ) : ℝ :=
  initial * (1 + interest) + annual

def steve_saving_after_five_years : ℝ :=
  let year1 := total_after_one_year initial_deposit annual_deposit_years_1_to_2 interest_years_1_to_3
  let year2 := total_after_one_year year1 annual_deposit_years_1_to_2 interest_years_1_to_3
  let year3 := total_after_one_year year2 annual_deposit_years_3_to_5 interest_years_1_to_3
  let year4 := total_after_one_year year3 annual_deposit_years_3_to_5 interest_years_4_to_5
  let year5 := total_after_one_year year4 annual_deposit_years_3_to_5 interest_years_4_to_5
  year5

theorem steve_final_amount :
  steve_saving_after_five_years = 230.88768 := by
  sorry

end steve_final_amount_l228_228466


namespace price_36kg_apples_l228_228074

-- Definitions based on given conditions
def cost_per_kg_first_30 (l : ℕ) (n₁ : ℕ) (total₁ : ℕ) : Prop :=
  n₁ = 10 ∧ l = total₁ / n₁

def total_cost_33kg (l q : ℕ) (total₂ : ℕ) : Prop :=
  30 * l + 3 * q = total₂

-- Question to prove
def total_cost_36kg (l q : ℕ) (cost_36 : ℕ) : Prop :=
  30 * l + 6 * q = cost_36

theorem price_36kg_apples (l q cost_36 : ℕ) :
  (cost_per_kg_first_30 l 10 200) →
  (total_cost_33kg l q 663) →
  cost_36 = 726 :=
by
  intros h₁ h₂
  sorry

end price_36kg_apples_l228_228074


namespace full_tank_capacity_l228_228590

theorem full_tank_capacity (speed : ℝ) (gas_usage_per_mile : ℝ) (time : ℝ) (gas_used_fraction : ℝ) (distance_per_tank : ℝ) (gallons_used : ℝ)
  (h1 : speed = 50)
  (h2 : gas_usage_per_mile = 1 / 30)
  (h3 : time = 5)
  (h4 : gas_used_fraction = 0.8333333333333334)
  (h5 : distance_per_tank = speed * time)
  (h6 : gallons_used = distance_per_tank * gas_usage_per_mile)
  (h7 : gallons_used = 0.8333333333333334 * 10) :
  distance_per_tank / 30 / 0.8333333333333334 = 10 :=
by sorry

end full_tank_capacity_l228_228590


namespace same_different_color_ways_equal_l228_228543

-- Definitions based on conditions in the problem
def num_black : ℕ := 15
def num_white : ℕ := 10

def same_color_ways : ℕ :=
  Nat.choose num_black 2 + Nat.choose num_white 2

def different_color_ways : ℕ :=
  num_black * num_white

-- The proof statement
theorem same_different_color_ways_equal : same_color_ways = different_color_ways :=
by
  sorry

end same_different_color_ways_equal_l228_228543


namespace seq_20_l228_228861

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

end seq_20_l228_228861


namespace taxes_are_135_l228_228313

def gross_pay : ℕ := 450
def net_pay : ℕ := 315
def taxes_paid (G N: ℕ) : ℕ := G - N

theorem taxes_are_135 : taxes_paid gross_pay net_pay = 135 := by
  sorry

end taxes_are_135_l228_228313


namespace smallest_palindrome_not_five_digit_l228_228121

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l228_228121


namespace abigail_fence_building_l228_228837

theorem abigail_fence_building :
  ∀ (initial_fences : Nat) (time_per_fence : Nat) (hours_building : Nat) (minutes_per_hour : Nat),
    initial_fences = 10 →
    time_per_fence = 30 →
    hours_building = 8 →
    minutes_per_hour = 60 →
    initial_fences + (minutes_per_hour / time_per_fence) * hours_building = 26 :=
by
  intros initial_fences time_per_fence hours_building minutes_per_hour
  sorry

end abigail_fence_building_l228_228837


namespace baba_yaga_powder_problem_l228_228696

theorem baba_yaga_powder_problem (A B d : ℝ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 := 
sorry

end baba_yaga_powder_problem_l228_228696


namespace range_of_k_l228_228314

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 2 then 2 / x else (x - 1)^3

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - k = 0 ∧ f x2 - k = 0) ↔ (0 < k ∧ k < 1) := sorry

end range_of_k_l228_228314


namespace monotonic_decreasing_interval_l228_228825

noncomputable def y (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem monotonic_decreasing_interval :
  {x : ℝ | (∃ y', y' = 3 * x^2 - 3 ∧ y' < 0)} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end monotonic_decreasing_interval_l228_228825


namespace roots_positive_range_no_negative_roots_opposite_signs_range_l228_228494

theorem roots_positive_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → (6 < m ∧ m ≤ 8 ∨ m ≥ 24) :=
sorry

theorem no_negative_roots (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → ¬ (∀ α β, (α < 0 ∧ β < 0)) :=
sorry

theorem opposite_signs_range (m : ℝ) : (8 * x^2 - m * x + (m - 6) = 0) → m < 6 :=
sorry

end roots_positive_range_no_negative_roots_opposite_signs_range_l228_228494


namespace tagged_fish_ratio_l228_228603

theorem tagged_fish_ratio (tagged_first_catch : ℕ) 
(tagged_second_catch : ℕ) (total_second_catch : ℕ) 
(h1 : tagged_first_catch = 30) (h2 : tagged_second_catch = 2) 
(h3 : total_second_catch = 50) : tagged_second_catch / total_second_catch = 1 / 25 :=
by
  sorry

end tagged_fish_ratio_l228_228603


namespace problem_l228_228969

noncomputable def f : ℝ → ℝ := sorry

theorem problem (f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
                (h : ∀ x : ℝ, 0 < x → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3 / 2 :=
sorry

end problem_l228_228969


namespace largest_int_with_remainder_l228_228691

theorem largest_int_with_remainder (k : ℤ) (h₁ : k < 95) (h₂ : k % 7 = 5) : k = 94 := by
sorry

end largest_int_with_remainder_l228_228691


namespace domain_of_f_l228_228891

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / (Real.sqrt (x - 7))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = f x } = {x : ℝ | x > 7} :=
by
  sorry

end domain_of_f_l228_228891


namespace arithmetic_geometric_seq_l228_228811

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l228_228811


namespace find_side_difference_l228_228507

def triangle_ABC : Type := ℝ
def angle_B := 20
def angle_C := 40
def length_AD := 2

theorem find_side_difference (ABC : triangle_ABC) (B : ℝ) (C : ℝ) (AD : ℝ) (BC AB : ℝ) :
  B = angle_B → C = angle_C → AD = length_AD → BC - AB = 2 :=
by 
  sorry

end find_side_difference_l228_228507


namespace range_of_a_l228_228823

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 5^x = a + 3) → a > -3 :=
by
  sorry

end range_of_a_l228_228823


namespace children_being_catered_l228_228797

-- Define the total meal units available
def meal_units_for_adults : ℕ := 70
def meal_units_for_children : ℕ := 90
def meals_eaten_by_adults : ℕ := 14
def remaining_meal_units : ℕ := meal_units_for_adults - meals_eaten_by_adults

theorem children_being_catered :
  (remaining_meal_units * meal_units_for_children) / meal_units_for_adults = 72 := by
{
  sorry
}

end children_being_catered_l228_228797


namespace evaluate_expression_l228_228370

theorem evaluate_expression :
  -(12 * 2) - (3 * 2) + ((-18 / 3) * -4) = -6 := 
by
  sorry

end evaluate_expression_l228_228370


namespace average_star_rating_l228_228761

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l228_228761


namespace eq_triangle_perimeter_l228_228499

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l228_228499


namespace number_of_people_who_purchased_only_book_A_l228_228130

-- Define the conditions and the problem
theorem number_of_people_who_purchased_only_book_A 
    (total_A : ℕ) (total_B : ℕ) (both_AB : ℕ) (only_B : ℕ) :
    (total_A = 2 * total_B) → 
    (both_AB = 500) → 
    (both_AB = 2 * only_B) → 
    (total_B = only_B + both_AB) → 
    (total_A - both_AB = 1000) :=
by
  sorry

end number_of_people_who_purchased_only_book_A_l228_228130


namespace diff_between_roots_l228_228719

theorem diff_between_roots (p : ℝ) (r s : ℝ)
  (h_eq : ∀ x : ℝ, x^2 - (p+1)*x + (p^2 + 2*p - 3)/4 = 0 → x = r ∨ x = s)
  (h_ge : r ≥ s) :
  r - s = Real.sqrt (2*p + 1 - p^2) := by
  sorry

end diff_between_roots_l228_228719


namespace average_speed_of_car_l228_228946

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

end average_speed_of_car_l228_228946


namespace total_trees_after_planting_l228_228566

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l228_228566


namespace coordinates_of_point_l228_228623

theorem coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, 3)) : (x, y) = (-2, 3) :=
by
  exact h

end coordinates_of_point_l228_228623


namespace initial_number_of_students_l228_228075

theorem initial_number_of_students (S : ℕ) (h : S + 6 = 37) : S = 31 :=
sorry

end initial_number_of_students_l228_228075


namespace registered_voters_democrats_l228_228764

variables (D R : ℝ)

theorem registered_voters_democrats :
  (D + R = 100) →
  (0.80 * D + 0.30 * R = 65) →
  D = 70 :=
by
  intros h1 h2
  sorry

end registered_voters_democrats_l228_228764


namespace inequality_holds_l228_228988

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, a*x^2 + 2*a*x - 2 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_holds_l228_228988


namespace quadratic_completing_square_l228_228866

theorem quadratic_completing_square
  (a : ℤ) (b : ℤ) (c : ℤ)
  (h1 : a > 0)
  (h2 : 64 * a^2 * x^2 - 96 * x - 48 = 64 * x^2 - 96 * x - 48)
  (h3 : (a * x + b)^2 = c) :
  a + b + c = 86 :=
sorry

end quadratic_completing_square_l228_228866


namespace find_ab_cd_l228_228230

variables (a b c d : ℝ)

def special_eq (x : ℝ) := 
  (min (20 * x + 19) (19 * x + 20) = (a * x + b) - abs (c * x + d))

theorem find_ab_cd (h : ∀ x : ℝ, special_eq a b c d x) :
  a * b + c * d = 380 := 
sorry

end find_ab_cd_l228_228230


namespace stone_travel_distance_l228_228814

/-- Define the radii --/
def radius_fountain := 15
def radius_stone := 3

/-- Prove the distance the stone needs to travel along the fountain's edge --/
theorem stone_travel_distance :
  let circumference_fountain := 2 * Real.pi * ↑radius_fountain
  let circumference_stone := 2 * Real.pi * ↑radius_stone
  let distance_traveled := circumference_stone
  distance_traveled = 6 * Real.pi := by
  -- Placeholder for proof, based on conditions given
  sorry

end stone_travel_distance_l228_228814


namespace alpha_eq_beta_l228_228638

variable {α β : ℝ}

theorem alpha_eq_beta
  (h_alpha : 0 < α ∧ α < (π / 2))
  (h_beta : 0 < β ∧ β < (π / 2))
  (h_sin : Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β)) :
  α = β :=
by
  sorry

end alpha_eq_beta_l228_228638


namespace betty_wallet_l228_228851

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

end betty_wallet_l228_228851


namespace yellow_tint_percentage_l228_228102

theorem yellow_tint_percentage {V₀ V₁ V_t red_pct yellow_pct : ℝ} 
  (hV₀ : V₀ = 40)
  (hRed : red_pct = 0.20)
  (hYellow : yellow_pct = 0.25)
  (hAdd : V₁ = 10) :
  (yellow_pct * V₀ + V₁) / (V₀ + V₁) = 0.40 :=
by
  sorry

end yellow_tint_percentage_l228_228102


namespace betty_total_stones_l228_228471

def stones_per_bracelet : ℕ := 14
def number_of_bracelets : ℕ := 10
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_total_stones : total_stones = 140 := by
  sorry

end betty_total_stones_l228_228471


namespace problem1_problem2_l228_228232

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (m^2 - 4) / (4 + 4 * m + m^2) / ((m - 2) / (2 * m - 2)) * ((m + 2) / (m - 1)) = 2 := 
sorry

end problem1_problem2_l228_228232


namespace employee_salary_l228_228351

theorem employee_salary (A B : ℝ) (h1 : A + B = 560) (h2 : A = 1.5 * B) : B = 224 :=
by
  sorry

end employee_salary_l228_228351


namespace total_length_of_figure_2_segments_l228_228105

-- Definitions based on conditions
def rectangle_length : ℕ := 10
def rectangle_breadth : ℕ := 6
def square_side : ℕ := 4
def interior_segment : ℕ := rectangle_breadth / 2

-- Summing up the lengths of segments in Figure 2
def total_length_of_segments : ℕ :=
  square_side + 2 * rectangle_length + interior_segment

-- Mathematical proof problem statement
theorem total_length_of_figure_2_segments :
  total_length_of_segments = 27 :=
sorry

end total_length_of_figure_2_segments_l228_228105


namespace correct_operation_l228_228589

theorem correct_operation (a : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  ((-4 * a^3)^2 = 16 * a^6) ∧ 
  (a^6 / a^6 ≠ 0) ∧ 
  ((a - 1)^2 ≠ a^2 - 1) := by
  sorry

end correct_operation_l228_228589


namespace article_filling_correct_l228_228294

-- definitions based on conditions provided
def Gottlieb_Daimler := "Gottlieb Daimler was a German engineer."
def Invented_Car := "Daimler is normally believed to have invented the car."

-- Statement we want to prove
theorem article_filling_correct : 
  (Gottlieb_Daimler = "Gottlieb Daimler was a German engineer.") ∧ 
  (Invented_Car = "Daimler is normally believed to have invented the car.") →
  ("Gottlieb Daimler, a German engineer, is normally believed to have invented the car." = 
   "Gottlieb Daimler, a German engineer, is normally believed to have invented the car.") :=
by
  sorry

end article_filling_correct_l228_228294


namespace lineup_count_l228_228417

-- Define five distinct people
inductive Person 
| youngest : Person 
| oldest : Person 
| person1 : Person 
| person2 : Person 
| person3 : Person 

-- Define the total number of people
def numberOfPeople : ℕ := 5

-- Define a function to calculate the number of ways to line up five people with constraints
def lineupWays : ℕ := 3 * 4 * 3 * 2 * 1

-- State the theorem
theorem lineup_count (h₁ : numberOfPeople = 5) (h₂ : ¬ ∃ (p : Person), p = Person.youngest ∨ p = Person.oldest → p = Person.youngest) :
  lineupWays = 72 :=
by
  sorry

end lineup_count_l228_228417


namespace mode_of_scores_is_85_l228_228938

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

end mode_of_scores_is_85_l228_228938


namespace four_x_plus_y_greater_than_four_z_l228_228925

theorem four_x_plus_y_greater_than_four_z
  (x y z : ℝ)
  (h1 : y > 2 * z)
  (h2 : 2 * z > 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) > 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z)
  : 4 * x + y > 4 * z := 
by
  sorry

end four_x_plus_y_greater_than_four_z_l228_228925


namespace compute_mod_expression_l228_228872

theorem compute_mod_expression :
  (3 * (1 / 7) + 9 * (1 / 13)) % 72 = 18 := sorry

end compute_mod_expression_l228_228872


namespace comb_eq_l228_228199

theorem comb_eq {n : ℕ} (h : Nat.choose 18 n = Nat.choose 18 2) : n = 2 ∨ n = 16 :=
by
  sorry

end comb_eq_l228_228199


namespace average_sale_over_six_months_l228_228131

theorem average_sale_over_six_months : 
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  average_sale = 3500 :=
by
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  show average_sale = 3500
  sorry

end average_sale_over_six_months_l228_228131


namespace pluto_orbit_scientific_notation_l228_228282

theorem pluto_orbit_scientific_notation : 5900000000 = 5.9 * 10^9 := by
  sorry

end pluto_orbit_scientific_notation_l228_228282


namespace jaymee_older_than_twice_shara_l228_228235

-- Given conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 22

-- Theorem to prove how many years older Jaymee is than twice Shara's age
theorem jaymee_older_than_twice_shara : jaymee_age - 2 * shara_age = 2 := by
  sorry

end jaymee_older_than_twice_shara_l228_228235


namespace debby_photos_of_friends_l228_228335

theorem debby_photos_of_friends (F : ℕ) (h1 : 23 + F = 86) : F = 63 := by
  -- Proof steps will go here
  sorry

end debby_photos_of_friends_l228_228335


namespace remainder_91_pow_91_mod_100_l228_228331

theorem remainder_91_pow_91_mod_100 : Nat.mod (91 ^ 91) 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l228_228331


namespace solve_quadratic_inequality_l228_228537

theorem solve_quadratic_inequality (a : ℝ) (x : ℝ) :
  (x^2 - a * x + a - 1 ≤ 0) ↔
  (a < 2 ∧ a - 1 ≤ x ∧ x ≤ 1) ∨
  (a = 2 ∧ x = 1) ∨
  (a > 2 ∧ 1 ≤ x ∧ x ≤ a - 1) := 
by
  sorry

end solve_quadratic_inequality_l228_228537


namespace train_passes_bridge_in_20_seconds_l228_228902

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 90

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_passes_bridge_in_20_seconds :
  travel_time = 20 := by
  sorry

end train_passes_bridge_in_20_seconds_l228_228902


namespace min_a_plus_b_l228_228572

open Real

theorem min_a_plus_b (a b : ℕ) (h_a_pos : a > 1) (h_ab : ∃ a b, (a^2 * b - 1) / (a * b^2) = 1 / 2024) :
  a + b = 228 :=
sorry

end min_a_plus_b_l228_228572


namespace initial_dogwood_trees_in_park_l228_228503

def num_added_trees := 5 + 4
def final_num_trees := 16
def initial_num_trees (x : ℕ) := x

theorem initial_dogwood_trees_in_park (x : ℕ) 
  (h1 : num_added_trees = 9) 
  (h2 : final_num_trees = 16) : 
  initial_num_trees x + num_added_trees = final_num_trees → 
  x = 7 := 
by 
  intro h3
  rw [initial_num_trees, num_added_trees] at h3
  linarith

end initial_dogwood_trees_in_park_l228_228503


namespace spencer_total_jumps_l228_228900

noncomputable def jumps_per_minute : ℕ := 4
noncomputable def minutes_per_session : ℕ := 10
noncomputable def sessions_per_day : ℕ := 2
noncomputable def days : ℕ := 5

theorem spencer_total_jumps : 
  (jumps_per_minute * minutes_per_session) * (sessions_per_day * days) = 400 :=
by
  sorry

end spencer_total_jumps_l228_228900


namespace intersection_M_N_l228_228795

def M : Set ℝ :=
  {x | |x| ≤ 2}

def N : Set ℝ :=
  {x | Real.exp x ≥ 1}

theorem intersection_M_N :
  (M ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l228_228795


namespace focus_of_parabola_l228_228334

-- Definitions for the problem
def parabola_eq (x y : ℝ) : Prop := y = 2 * x^2

def general_parabola_form (x y h k p : ℝ) : Prop :=
  4 * p * (y - k) = (x - h)^2

def vertex_origin (h k : ℝ) : Prop := h = 0 ∧ k = 0

-- Lean statement asserting that the focus of the given parabola is (0, 1/8)
theorem focus_of_parabola : ∃ p : ℝ, parabola_eq x y → general_parabola_form x y 0 0 p ∧ p = 1/8 := by
  sorry

end focus_of_parabola_l228_228334


namespace mn_eq_one_l228_228742

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end mn_eq_one_l228_228742


namespace sector_area_l228_228957

/-- The area of a sector with a central angle of 72 degrees and a radius of 20 cm is 80π cm². -/
theorem sector_area (radius : ℝ) (angle : ℝ) (h_angle_deg : angle = 72) (h_radius : radius = 20) :
  (angle / 360) * π * radius^2 = 80 * π :=
by sorry

end sector_area_l228_228957


namespace inequality_solution_l228_228501

theorem inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 7 → a * x^2 + b * x + c > 0) →
  (∀ x : ℝ, (x < -1/7 ∨ x > 1/4) ↔ c * x^2 - b * x + a > 0) :=
by
  sorry

end inequality_solution_l228_228501


namespace gcd_pow_minus_one_l228_228226

theorem gcd_pow_minus_one (n m : ℕ) (hn : n = 1030) (hm : m = 1040) :
  Nat.gcd (2^n - 1) (2^m - 1) = 1023 := 
by
  sorry

end gcd_pow_minus_one_l228_228226


namespace coin_arrangements_l228_228291

/-- We define the conditions for Robert's coin arrangement problem. -/
def gold_coins := 5
def silver_coins := 5
def total_coins := gold_coins + silver_coins

/-- We define the number of ways to arrange 5 gold coins and 5 silver coins in 10 positions,
using the binomial coefficient. -/
def arrangements_colors : ℕ := Nat.choose total_coins gold_coins

/-- We define the number of possible configurations for the orientation of the coins
such that no two adjacent coins are face to face. -/
def arrangements_orientation : ℕ := 11

/-- The total number of distinguishable arrangements of the coins. -/
def total_arrangements : ℕ := arrangements_colors * arrangements_orientation

theorem coin_arrangements : total_arrangements = 2772 := by
  -- The proof is omitted.
  sorry

end coin_arrangements_l228_228291


namespace sum_of_arithmetic_sequence_l228_228910

def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_sum_f : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by 
  sorry

end sum_of_arithmetic_sequence_l228_228910


namespace inequality_solution_l228_228818

theorem inequality_solution (x : ℝ) :
  4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 → x ∈ Set.Ioc (5 / 2 : ℝ) (20 / 7 : ℝ) := by
  sorry

end inequality_solution_l228_228818


namespace average_speed_of_planes_l228_228059

-- Definitions for the conditions
def num_passengers_plane1 : ℕ := 50
def num_passengers_plane2 : ℕ := 60
def num_passengers_plane3 : ℕ := 40
def base_speed : ℕ := 600
def speed_reduction_per_passenger : ℕ := 2

-- Calculate speeds of each plane according to given conditions
def speed_plane1 := base_speed - num_passengers_plane1 * speed_reduction_per_passenger
def speed_plane2 := base_speed - num_passengers_plane2 * speed_reduction_per_passenger
def speed_plane3 := base_speed - num_passengers_plane3 * speed_reduction_per_passenger

-- Calculate the total speed and average speed
def total_speed := speed_plane1 + speed_plane2 + speed_plane3
def average_speed := total_speed / 3

-- The theorem to prove the average speed is 500 MPH
theorem average_speed_of_planes : average_speed = 500 := by
  sorry

end average_speed_of_planes_l228_228059


namespace jason_advertising_cost_l228_228894

def magazine_length : ℕ := 9
def magazine_width : ℕ := 12
def cost_per_square_inch : ℕ := 8
def half (x : ℕ) := x / 2
def area (L W : ℕ) := L * W
def total_cost (a c : ℕ) := a * c

theorem jason_advertising_cost :
  total_cost (half (area magazine_length magazine_width)) cost_per_square_inch = 432 := by
  sorry

end jason_advertising_cost_l228_228894


namespace two_lines_parallel_same_plane_l228_228669

-- Defining the types for lines and planes
variable (Line : Type) (Plane : Type)

-- Defining the relationships similar to the mathematical conditions
variable (parallel_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Defining the non-overlapping relationships between lines (assuming these relations are mutually exclusive)
axiom parallel_or_intersect_or_skew : ∀ (a b: Line), 
  (parallel a b ∨ intersect a b ∨ skew a b)

-- The statement we want to prove
theorem two_lines_parallel_same_plane (a b: Line) (α: Plane) :
  parallel_to_plane a α → parallel_to_plane b α → (parallel a b ∨ intersect a b ∨ skew a b) :=
by
  intro ha hb
  apply parallel_or_intersect_or_skew

end two_lines_parallel_same_plane_l228_228669


namespace expand_expression_l228_228663

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l228_228663


namespace cubicsum_eq_neg36_l228_228191

noncomputable def roots (p q r : ℝ) := 
  ∃ l : ℝ, (p^3 - 12) / p = l ∧ (q^3 - 12) / q = l ∧ (r^3 - 12) / r = l

theorem cubicsum_eq_neg36 {p q r : ℝ} (h : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hl : roots p q r) :
  p^3 + q^3 + r^3 = -36 :=
sorry

end cubicsum_eq_neg36_l228_228191


namespace vertex_of_quadratic1_vertex_of_quadratic2_l228_228410

theorem vertex_of_quadratic1 :
  ∃ x y : ℝ, 
  (∀ x', 2 * x'^2 - 4 * x' - 1 = 2 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = -3) :=
by sorry

theorem vertex_of_quadratic2 :
  ∃ x y : ℝ, 
  (∀ x', -3 * x'^2 + 6 * x' - 2 = -3 * (x' - x)^2 + y) ∧ 
  (x = 1 ∧ y = 1) :=
by sorry

end vertex_of_quadratic1_vertex_of_quadratic2_l228_228410


namespace inequality_proof_l228_228145

-- Conditions: a > b and c > d
variables {a b c d : ℝ}

-- The main statement to prove: d - a < c - b with given conditions
theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := 
sorry

end inequality_proof_l228_228145


namespace generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l228_228068

-- Define the number five as 4, as we are using five 4s
def four := 4

-- Now prove that each number from 1 to 22 can be generated using the conditions
theorem generate_1 : 1 = (4 / 4) * (4 / 4) := sorry
theorem generate_2 : 2 = (4 / 4) + (4 / 4) := sorry
theorem generate_3 : 3 = ((4 + 4 + 4) / 4) - (4 / 4) := sorry
theorem generate_4 : 4 = 4 * (4 - 4) + 4 := sorry
theorem generate_5 : 5 = 4 + (4 / 4) := sorry
theorem generate_6 : 6 = 4 + 4 - (4 / 4) := sorry
theorem generate_7 : 7 = 4 + 4 - (4 / 4) := sorry
theorem generate_8 : 8 = 4 + 4 := sorry
theorem generate_9 : 9 = 4 + 4 + (4 / 4) := sorry
theorem generate_10 : 10 = 4 * (2 + 4 / 4) := sorry
theorem generate_11 : 11 = 4 * (3 - 1 / 4) := sorry
theorem generate_12 : 12 = 4 + 4 + 4 := sorry
theorem generate_13 : 13 = (4 * 4) - (4 / 4) - 4 := sorry
theorem generate_14 : 14 = 4 * (4 - 1 / 4) := sorry
theorem generate_15 : 15 = 4 * 4 - (4 / 4) - 1 := sorry
theorem generate_16 : 16 = 4 * (4 - (4 - 4) / 4) := sorry
theorem generate_17 : 17 = 4 * (4 + 4 / 4) := sorry
theorem generate_18 : 18 = 4 * 4 + 4 - 4 / 4 := sorry
theorem generate_19 : 19 = 4 + 4 + 4 + 4 + 3 := sorry
theorem generate_20 : 20 = 4 + 4 + 4 + 4 + 4 := sorry
theorem generate_21 : 21 = 4 * 4 + (4 - 1) / 4 := sorry
theorem generate_22 : 22 = (4 * 4 + 4) / 4 := sorry

end generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l228_228068


namespace melanie_gave_mother_l228_228846

theorem melanie_gave_mother {initial_dimes dad_dimes final_dimes dimes_given : ℕ}
  (h₁ : initial_dimes = 7)
  (h₂ : dad_dimes = 8)
  (h₃ : final_dimes = 11)
  (h₄ : initial_dimes + dad_dimes - dimes_given = final_dimes) :
  dimes_given = 4 :=
by 
  sorry

end melanie_gave_mother_l228_228846


namespace factor_problem_l228_228519

theorem factor_problem (x y m : ℝ) (h : (1 - 2 * x + y) ∣ (4 * x * y - 4 * x^2 - y^2 - m)) :
  m = -1 :=
by
  sorry

end factor_problem_l228_228519


namespace min_area_triangle_ABC_l228_228821

def point (α : Type*) := (α × α)

def area_of_triangle (A B C : point ℤ) : ℚ :=
  (1/2 : ℚ) * abs (36 * (C.snd) - 15 * (C.fst))

theorem min_area_triangle_ABC :
  ∃ (C : point ℤ), area_of_triangle (0, 0) (36, 15) C = 3 / 2 :=
by
  sorry

end min_area_triangle_ABC_l228_228821


namespace min_value_expression_l228_228187

theorem min_value_expression : ∃ x y : ℝ, (xy-2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_expression_l228_228187


namespace solve_for_x_l228_228580

theorem solve_for_x (x y : ℤ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 :=
by
  sorry

end solve_for_x_l228_228580


namespace shaded_area_in_octagon_l228_228661

theorem shaded_area_in_octagon (s r : ℝ) (h_s : s = 4) (h_r : r = s / 2) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_semicircles := 8 * (π * r^2 / 2)
  area_octagon - area_semicircles = 32 * (1 + Real.sqrt 2) - 16 * π := by
  sorry

end shaded_area_in_octagon_l228_228661


namespace option_b_not_valid_l228_228149

theorem option_b_not_valid (a b c d : ℝ) (h_arith_seq : b - a = d ∧ c - b = d ∧ d ≠ 0) : 
  a^3 * b + b^3 * c + c^3 * a < a^4 + b^4 + c^4 :=
by sorry

end option_b_not_valid_l228_228149


namespace sequence_b_10_eq_110_l228_228299

theorem sequence_b_10_eq_110 :
  (∃ (b : ℕ → ℕ), b 1 = 2 ∧ (∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) ∧ b 10 = 110) :=
sorry

end sequence_b_10_eq_110_l228_228299


namespace parallelepiped_length_l228_228758

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l228_228758


namespace intervals_of_monotonicity_range_of_values_for_a_l228_228684

noncomputable def f (a x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, a ≤ -1 → deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioc 0 (1 + a), -1 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Set.Ioi (1 + a), -1 < a → deriv (f a) x > 0) :=
sorry

theorem range_of_values_for_a (a : ℝ) (e : ℝ) (h : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 1 e, f a x ≤ 0) → (a ≤ -2 ∨ a ≥ (e^2 + 1) / (e - 1)) :=
sorry

end intervals_of_monotonicity_range_of_values_for_a_l228_228684


namespace arun_age_l228_228561

theorem arun_age (A G M : ℕ) (h1 : (A - 6) / 18 = G) (h2 : G = M - 2) (h3 : M = 5) : A = 60 :=
by
  sorry

end arun_age_l228_228561


namespace equation1_no_solution_equation2_solution_l228_228702

/-- Prove that the equation (4-x)/(x-3) + 1/(3-x) = 1 has no solution. -/
theorem equation1_no_solution (x : ℝ) : x ≠ 3 → ¬ (4 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by intro hx; sorry

/-- Prove that the equation (x+1)/(x-1) - 6/(x^2-1) = 1 has solution x = 2. -/
theorem equation2_solution (x : ℝ) : x = 2 ↔ (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1 :=
by sorry

end equation1_no_solution_equation2_solution_l228_228702


namespace find_number_l228_228954

theorem find_number (x : ℝ) : ((x - 50) / 4) * 3 + 28 = 73 → x = 110 := 
  by 
  sorry

end find_number_l228_228954


namespace alpha_parallel_to_beta_l228_228584

variables (a b : ℝ → ℝ → ℝ) (α β : ℝ → ℝ)

-- Definitions based on conditions
def are_distinct_lines : a ≠ b := sorry
def are_distinct_planes : α ≠ β := sorry

def line_parallel_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define parallel relation
def line_perpendicular_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define perpendicular relation
def planes_parallel (p1 p2 : ℝ → ℝ) : Prop := sorry -- Define planes being parallel

-- Given as conditions
axiom a_perpendicular_to_alpha : line_perpendicular_to_plane a α
axiom b_perpendicular_to_beta : line_perpendicular_to_plane b β
axiom a_parallel_to_b : a = b

-- The proposition to prove
theorem alpha_parallel_to_beta : planes_parallel α β :=
by {
  -- Placeholder for the logic provided through the previous solution steps.
  sorry
}

end alpha_parallel_to_beta_l228_228584


namespace rectangle_area_l228_228008

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area_l228_228008


namespace union_A_B_complement_union_l228_228011

-- Define \( U \), \( A \), and \( B \)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

-- Define complement in the universe \( U \)
def complement_U (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- Statements to prove
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
  sorry

theorem complement_union : complement_U A ∪ complement_U B = {x | x < 3 ∨ x ≥ 5} :=
  sorry

end union_A_B_complement_union_l228_228011


namespace equilateral_triangle_sum_l228_228024

noncomputable def equilateral_triangle (a b c : Complex) (s : ℝ) : Prop :=
  Complex.abs (a - b) = s ∧ Complex.abs (b - c) = s ∧ Complex.abs (c - a) = s

theorem equilateral_triangle_sum (a b c : Complex):
  equilateral_triangle a b c 18 →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  intros h_triangle h_sum
  sorry

end equilateral_triangle_sum_l228_228024


namespace moles_of_CaCl2_l228_228092

theorem moles_of_CaCl2 (HCl moles_of_HCl : ℕ) (CaCO3 moles_of_CaCO3 : ℕ) 
  (reaction : (CaCO3 = 1) → (HCl = 2) → (moles_of_HCl = 6) → (moles_of_CaCO3 = 3)) :
  ∃ moles_of_CaCl2 : ℕ, moles_of_CaCl2 = 3 :=
by
  sorry

end moles_of_CaCl2_l228_228092


namespace proof_time_lent_to_C_l228_228189

theorem proof_time_lent_to_C :
  let P_B := 5000
  let R := 0.1
  let T_B := 2
  let Total_Interest := 2200
  let P_C := 3000
  let I_B := P_B * R * T_B
  let I_C := Total_Interest - I_B
  let T_C := I_C / (P_C * R)
  T_C = 4 :=
by
  sorry

end proof_time_lent_to_C_l228_228189


namespace cos_double_angle_l228_228512

open Real

theorem cos_double_angle (α β : ℝ) 
    (h1 : sin α = 2 * sin β) 
    (h2 : tan α = 3 * tan β) :
  cos (2 * α) = -1 / 4 ∨ cos (2 * α) = 1 := 
sorry

end cos_double_angle_l228_228512


namespace min_sum_geometric_sequence_l228_228338

noncomputable def sequence_min_value (a : ℕ → ℝ) : ℝ :=
  a 4 + a 3 - 2 * a 2 - 2 * a 1

theorem min_sum_geometric_sequence (a : ℕ → ℝ)
  (h : sequence_min_value a = 6) :
  a 5 + a 6 = 48 := 
by
  sorry

end min_sum_geometric_sequence_l228_228338


namespace division_result_l228_228437

theorem division_result : (5 * 6 + 4) / 8 = 4.25 :=
by
  sorry

end division_result_l228_228437


namespace binomial_standard_deviation_l228_228679

noncomputable def standard_deviation_binomial (n : ℕ) (p : ℝ) : ℝ :=
  Real.sqrt (n * p * (1 - p))

theorem binomial_standard_deviation (n : ℕ) (p : ℝ) (hn : 0 ≤ n) (hp : 0 ≤ p) (hp1: p ≤ 1) :
  standard_deviation_binomial n p = Real.sqrt (n * p * (1 - p)) :=
by
  sorry

end binomial_standard_deviation_l228_228679


namespace minimum_stamps_to_make_47_cents_l228_228595

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end minimum_stamps_to_make_47_cents_l228_228595


namespace anand_present_age_l228_228265

theorem anand_present_age (A B : ℕ) 
  (h1 : B = A + 10)
  (h2 : A - 10 = (B - 10) / 3) :
  A = 15 :=
sorry

end anand_present_age_l228_228265


namespace units_digit_expression_l228_228223

theorem units_digit_expression :
  ((2 * 21 * 2019 + 2^5) - 4^3) % 10 = 6 := 
sorry

end units_digit_expression_l228_228223


namespace solution_set_of_inequality_l228_228326

theorem solution_set_of_inequality (a x : ℝ) (h : 1 < a) :
  (x - a) * (x - (1 / a)) > 0 ↔ x < 1 / a ∨ x > a :=
by
  sorry

end solution_set_of_inequality_l228_228326


namespace find_X_l228_228243

def spadesuit (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

theorem find_X (X : ℝ) (h : spadesuit X 5 = 23) : X = 7.75 :=
by sorry

end find_X_l228_228243


namespace cos_double_angle_l228_228828

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 := 
sorry

end cos_double_angle_l228_228828


namespace quadratic_expression_value_l228_228440

theorem quadratic_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 * x2 = 2) (hx : x1^2 - 4 * x1 + 2 = 0) :
  x1^2 - 4 * x1 + 2 * x1 * x2 = 2 :=
sorry

end quadratic_expression_value_l228_228440


namespace perimeter_area_ratio_le_8_l228_228473

/-- Let \( S \) be a shape in the plane obtained as a union of finitely many unit squares.
    The perimeter of a single unit square is 4 and its area is 1.
    Prove that the ratio of the perimeter \( P \) and the area \( A \) of \( S \)
    is at most 8, i.e., \(\frac{P}{A} \leq 8\). -/
theorem perimeter_area_ratio_le_8
  (S : Set (ℝ × ℝ)) 
  (unit_square : ∀ (x y : ℝ), (x, y) ∈ S → (x + 1, y + 1) ∈ S ∧ (x + 1, y) ∈ S ∧ (x, y + 1) ∈ S ∧ (x, y) ∈ S)
  (P A : ℝ)
  (unit_square_perimeter : ∀ (x y : ℝ), (x, y) ∈ S → P = 4)
  (unit_square_area : ∀ (x y : ℝ), (x, y) ∈ S → A = 1) :
  P / A ≤ 8 :=
sorry

end perimeter_area_ratio_le_8_l228_228473


namespace number_of_raccoons_l228_228651

/-- Jason pepper-sprays some raccoons and 6 times as many squirrels. 
Given that he pepper-sprays a total of 84 animals, the number of raccoons he pepper-sprays is 12. -/
theorem number_of_raccoons (R : Nat) (h1 : 84 = R + 6 * R) : R = 12 :=
by
  sorry

end number_of_raccoons_l228_228651


namespace min_value_f_max_value_bac_l228_228252

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| - |x - 1|

theorem min_value_f : ∃ k : ℝ, (∀ x : ℝ, f x ≥ k) ∧ k = -2 := 
by
  sorry

theorem max_value_bac (a b c : ℝ) 
  (h1 : a^2 + c^2 + b^2 / 2 = 2) : 
  ∃ m : ℝ, (∀ a b c : ℝ, a^2 + c^2 + b^2 / 2 = 2 → b * (a + c) ≤ m) ∧ m = 2 := 
by
  sorry

end min_value_f_max_value_bac_l228_228252


namespace units_digit_expression_l228_228176

lemma units_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 := sorry

lemma units_digit_5_pow_2024 : (5 ^ 2024) % 10 = 5 := sorry

lemma units_digit_11_pow_2025 : (11 ^ 2025) % 10 = 1 := sorry

theorem units_digit_expression : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by 
  have h1 := units_digit_2_pow_2023
  have h2 := units_digit_5_pow_2024
  have h3 := units_digit_11_pow_2025
  sorry

end units_digit_expression_l228_228176


namespace prod_is_96_l228_228046

noncomputable def prod_of_nums (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : ℝ := x * y

theorem prod_is_96 (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : prod_of_nums x y h1 h2 = 96 :=
by
  sorry

end prod_is_96_l228_228046


namespace directrix_of_parabola_l228_228084

theorem directrix_of_parabola (x y : ℝ) (h : y = (1/4) * x^2) : y = -1 :=
sorry

end directrix_of_parabola_l228_228084


namespace product_of_two_numbers_l228_228111

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 :=
by
  sorry

end product_of_two_numbers_l228_228111


namespace not_covered_by_homothetic_polygons_l228_228422

structure Polygon :=
  (vertices : Set (ℝ × ℝ))

def homothetic (M : Polygon) (k : ℝ) (O : ℝ × ℝ) : Polygon :=
  {
    vertices := {p | ∃ (q : ℝ × ℝ) (hq : q ∈ M.vertices), p = (O.1 + k * (q.1 - O.1), O.2 + k * (q.2 - O.2))}
  }

theorem not_covered_by_homothetic_polygons (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1)
  (O1 O2 : ℝ × ℝ) :
  ¬ (∀ p ∈ M.vertices, p ∈ (homothetic M k O1).vertices ∨ p ∈ (homothetic M k O2).vertices) := by
  sorry

end not_covered_by_homothetic_polygons_l228_228422


namespace xy_range_l228_228412

theorem xy_range (x y : ℝ) (h1 : y = 3 * (⌊x⌋) + 2) (h2 : y = 4 * (⌊x - 3⌋) + 6) (h3 : (⌊x⌋ : ℝ) ≠ x) :
  34 < x + y ∧ x + y < 35 := 
by 
  sorry

end xy_range_l228_228412


namespace min_x2_plus_y2_l228_228127

theorem min_x2_plus_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_x2_plus_y2_l228_228127


namespace expected_attempts_for_10_suitcases_l228_228753

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (1 / 2) * (n * (n + 1) / 2) + (n / 2) - (Real.log n + 0.577)

theorem expected_attempts_for_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 1 :=
by
  sorry

end expected_attempts_for_10_suitcases_l228_228753


namespace total_money_shared_l228_228904

theorem total_money_shared (ratio_jonah ratio_kira ratio_liam kira_share : ℕ)
  (h_ratio : ratio_jonah = 2) (h_ratio2 : ratio_kira = 3) (h_ratio3 : ratio_liam = 8)
  (h_kira : kira_share = 45) :
  (ratio_jonah * (kira_share / ratio_kira) + kira_share + ratio_liam * (kira_share / ratio_kira)) = 195 := 
by
  sorry

end total_money_shared_l228_228904


namespace sophie_buys_six_doughnuts_l228_228415

variable (num_doughnuts : ℕ)

theorem sophie_buys_six_doughnuts 
  (h1 : 5 * 2 = 10)
  (h2 : 4 * 2 = 8)
  (h3 : 15 * 0.60 = 9)
  (h4 : 10 + 8 + 9 = 27)
  (h5 : 33 - 27 = 6)
  (h6 : num_doughnuts * 1 = 6) :
  num_doughnuts = 6 := 
  by
    sorry

end sophie_buys_six_doughnuts_l228_228415


namespace not_exist_three_numbers_l228_228037

theorem not_exist_three_numbers :
  ¬ ∃ (a b c : ℝ),
  (b^2 - 4 * a * c > 0 ∧ (-b / a > 0) ∧ (c / a > 0)) ∧
  (b^2 - 4 * a * c > 0 ∧ (-b / a < 0) ∧ (c / a > 0)) :=
by
  sorry

end not_exist_three_numbers_l228_228037


namespace triangle_inequality_l228_228486

variables {a b c : ℝ}

def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (h : sides_of_triangle a b c) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l228_228486


namespace binom_15_13_eq_105_l228_228456

theorem binom_15_13_eq_105 : Nat.choose 15 13 = 105 := by
  sorry

end binom_15_13_eq_105_l228_228456


namespace total_parallelepipeds_l228_228263

theorem total_parallelepipeds (m n k : ℕ) : 
  ∃ (num : ℕ), num == (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8 :=
  sorry

end total_parallelepipeds_l228_228263


namespace distance_between_foci_of_ellipse_l228_228260

theorem distance_between_foci_of_ellipse :
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  2 * c = 4 * Real.sqrt 15 :=
by
  let center : (ℝ × ℝ) := (8, 2)
  let a : ℝ := 16 / 2 -- half the length of the major axis
  let b : ℝ := 4 / 2  -- half the length of the minor axis
  let c : ℝ := Real.sqrt (a^2 - b^2) -- distance from the center to each focus
  show 2 * c = 4 * Real.sqrt 15
  sorry

end distance_between_foci_of_ellipse_l228_228260


namespace pilot_weeks_l228_228934

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

end pilot_weeks_l228_228934


namespace solve_for_k_in_quadratic_l228_228709

theorem solve_for_k_in_quadratic :
  ∃ k : ℝ, (∀ x1 x2 : ℝ,
    x1 + x2 = 3 ∧
    x1 * x2 + 2 * x1 + 2 * x2 = 1 ∧
    (x1^2 - 3*x1 + k = 0) ∧ (x2^2 - 3*x2 + k = 0)) →
  k = -5 :=
sorry

end solve_for_k_in_quadratic_l228_228709


namespace max_points_for_top_teams_l228_228672

-- Definitions based on the problem conditions
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def points_for_loss : ℕ := 0
def number_of_teams : ℕ := 8
def number_of_games_between_each_pair : ℕ := 2
def total_games : ℕ := (number_of_teams * (number_of_teams - 1) / 2) * number_of_games_between_each_pair
def total_points_in_tournament : ℕ := total_games * points_for_win
def top_teams : ℕ := 4

-- Theorem stating the correct answer
theorem max_points_for_top_teams : (total_points_in_tournament / number_of_teams = 33) :=
sorry

end max_points_for_top_teams_l228_228672


namespace number_of_people_third_day_l228_228016

variable (X : ℕ)
variable (total : ℕ := 246)
variable (first_day : ℕ := 79)
variable (second_day_third_day_diff : ℕ := 47)

theorem number_of_people_third_day :
  (first_day + (X + second_day_third_day_diff) + X = total) → 
  X = 60 := by
  sorry

end number_of_people_third_day_l228_228016


namespace dog_total_bones_l228_228550

-- Define the number of original bones and dug up bones as constants
def original_bones : ℕ := 493
def dug_up_bones : ℕ := 367

-- Define the total bones the dog has now
def total_bones : ℕ := original_bones + dug_up_bones

-- State and prove the theorem
theorem dog_total_bones : total_bones = 860 := by
  -- placeholder for the proof
  sorry

end dog_total_bones_l228_228550


namespace calculate_expression_l228_228687

theorem calculate_expression : 
  ∀ (x y z : ℤ), x = 2 → y = -3 → z = 7 → (x^2 + y^2 + z^2 - 2 * x * y) = 74 :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end calculate_expression_l228_228687


namespace store_shelves_l228_228178

theorem store_shelves (initial_books sold_books books_per_shelf : ℕ) 
    (h_initial: initial_books = 27)
    (h_sold: sold_books = 6)
    (h_per_shelf: books_per_shelf = 7) :
    (initial_books - sold_books) / books_per_shelf = 3 := by
  sorry

end store_shelves_l228_228178


namespace calculate_neg_pow_mul_l228_228978

theorem calculate_neg_pow_mul (a : ℝ) : -a^4 * a^3 = -a^7 := by
  sorry

end calculate_neg_pow_mul_l228_228978


namespace last_two_digits_of_sum_l228_228032

noncomputable def last_two_digits_sum_factorials : ℕ :=
  let fac : List ℕ := List.map (fun n => Nat.factorial (n * 3)) [1, 2, 3, 4, 5, 6, 7]
  fac.foldl (fun acc x => (acc + x) % 100) 0

theorem last_two_digits_of_sum : last_two_digits_sum_factorials = 6 :=
by
  sorry

end last_two_digits_of_sum_l228_228032


namespace quadratic_one_positive_root_l228_228870

theorem quadratic_one_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y ∈ {t | t^2 - a * t + a - 2 = 0} → y = x)) → a ≤ 2 :=
by
  sorry

end quadratic_one_positive_root_l228_228870


namespace product_of_primes_l228_228266

theorem product_of_primes :
  let p1 := 11
  let p2 := 13
  let p3 := 997
  p1 * p2 * p3 = 142571 :=
by
  sorry

end product_of_primes_l228_228266


namespace smallest_value_of_c_l228_228992

theorem smallest_value_of_c :
  ∃ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c ∧ (∀ d : ℚ, (3 * d + 4) * (d - 2) = 9 * d → c ≤ d) ∧ c = -8 / 3 := 
sorry

end smallest_value_of_c_l228_228992


namespace probability_A_not_losing_l228_228834

variable (P_A_wins : ℝ)
variable (P_draw : ℝ)
variable (P_A_not_losing : ℝ)

theorem probability_A_not_losing 
  (h1 : P_A_wins = 0.3) 
  (h2 : P_draw = 0.5) 
  (h3 : P_A_not_losing = P_A_wins + P_draw) :
  P_A_not_losing = 0.8 :=
sorry

end probability_A_not_losing_l228_228834


namespace kolya_made_mistake_l228_228640

theorem kolya_made_mistake (ab cd effe : ℕ)
  (h_eq : ab * cd = effe)
  (h_eff_div_11 : effe % 11 = 0)
  (h_ab_cd_not_div_11 : ab % 11 ≠ 0 ∧ cd % 11 ≠ 0) :
  false :=
by
  -- Note: This is where the proof would go, but we are illustrating the statement only.
  sorry

end kolya_made_mistake_l228_228640


namespace no_natural_number_n_exists_l228_228158

theorem no_natural_number_n_exists :
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end no_natural_number_n_exists_l228_228158


namespace num_square_free_odds_l228_228496

noncomputable def is_square_free (m : ℕ) : Prop :=
  ∀ n : ℕ, n^2 ∣ m → n = 1

noncomputable def count_square_free_odds : ℕ :=
  (199 - 1) / 2 - (11 + 4 + 2 + 1 + 1 + 1)

theorem num_square_free_odds : count_square_free_odds = 79 := by
  sorry

end num_square_free_odds_l228_228496


namespace product_of_four_consecutive_is_perfect_square_l228_228843

theorem product_of_four_consecutive_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
by
  sorry

end product_of_four_consecutive_is_perfect_square_l228_228843


namespace final_value_A_is_5_l228_228444

/-
Problem: Given a 3x3 grid of numbers and a series of operations that add or subtract 1 to two adjacent cells simultaneously, prove that the number in position A in the table on the right is 5.
Conditions:
1. The initial grid is:
   \[
   \begin{array}{ccc}
   a & b & c \\
   d & e & f \\
   g & h & i \\
   \end{array}
   \]
2. Each operation involves adding or subtracting 1 from two adjacent cells.
3. The sum of all numbers in the grid remains unchanged.
-/

def table_operations (a b c d e f g h i : ℤ) : ℤ :=
-- A is determined based on the given problem and conditions
  5

theorem final_value_A_is_5 (a b c d e f g h i : ℤ) : 
  table_operations a b c d e f g h i = 5 :=
sorry

end final_value_A_is_5_l228_228444


namespace total_travel_time_l228_228771

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l228_228771


namespace divisibility_properties_l228_228972

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬(a + b ∣ a^(2*k) + b^(2*k)) ∧ ¬(a - b ∣ a^(2*k) + b^(2*k))) ∧ 
  ((a + b ∣ a^(2*k) - b^(2*k)) ∧ (a - b ∣ a^(2*k) - b^(2*k))) ∧ 
  (a + b ∣ a^(2*k + 1) + b^(2*k + 1)) ∧ 
  (a - b ∣ a^(2*k + 1) - b^(2*k + 1)) := 
by sorry

end divisibility_properties_l228_228972


namespace rhombus_longer_diagonal_l228_228792

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l228_228792


namespace probability_of_choosing_red_base_l228_228030

theorem probability_of_choosing_red_base (A B : Prop) (C D : Prop) : 
  let red_bases := 2
  let total_bases := 4
  let probability := red_bases / total_bases
  probability = 1 / 2 := 
by
  sorry

end probability_of_choosing_red_base_l228_228030


namespace range_of_a_l228_228980

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), (0 < x1 ∧ x1 < x2 ∧ x2 < 1) → (a * x2 - x2^3) - (a * x1 - x1^3) > x2 - x1) : a ≥ 4 :=
sorry


end range_of_a_l228_228980


namespace replace_asterisks_l228_228400

theorem replace_asterisks (x : ℕ) (h : (x / 20) * (x / 180) = 1) : x = 60 := by
  sorry

end replace_asterisks_l228_228400


namespace savings_percentage_l228_228794

theorem savings_percentage (I S : ℝ) (h1 : I > 0) (h2 : S > 0) (h3 : S ≤ I) 
  (h4 : 1.25 * I - 2 * S + I - S = 2 * (I - S)) :
  (S / I) * 100 = 25 :=
by
  sorry

end savings_percentage_l228_228794


namespace solve_for_x_l228_228425

theorem solve_for_x (x : ℚ) (h : (x + 2) / (x - 3) = (x - 4) / (x + 5)) : x = 1 / 7 :=
sorry

end solve_for_x_l228_228425


namespace total_notebooks_l228_228829

-- Define the problem conditions
theorem total_notebooks (x : ℕ) (hx : x*x + 20 = (x+1)*(x+1) - 9) : x*x + 20 = 216 :=
by
  have h1 : x*x + 20 = 216 := sorry
  exact h1

end total_notebooks_l228_228829


namespace extra_interest_is_correct_l228_228268

def principal : ℝ := 5000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

def interest1 : ℝ := simple_interest principal rate1 time
def interest2 : ℝ := simple_interest principal rate2 time

def extra_interest : ℝ := interest1 - interest2

theorem extra_interest_is_correct : extra_interest = 600 := by
  sorry

end extra_interest_is_correct_l228_228268


namespace squares_difference_l228_228694

theorem squares_difference (a b : ℕ) (h₁ : a = 601) (h₂ : b = 597) : a^2 - b^2 = 4792 := by
  rw [h₁, h₂]
  -- insert actual proof here
  sorry

end squares_difference_l228_228694


namespace product_of_integers_whose_cubes_sum_to_189_l228_228120

theorem product_of_integers_whose_cubes_sum_to_189 :
  ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  sorry

end product_of_integers_whose_cubes_sum_to_189_l228_228120


namespace reduction_of_cycle_l228_228740

noncomputable def firstReductionPercentage (P : ℝ) (x : ℝ) : Prop :=
  P * (1 - (x / 100)) * 0.8 = 0.6 * P

theorem reduction_of_cycle (P x : ℝ) (hP : 0 < P) : firstReductionPercentage P x → x = 25 :=
by
  intros h
  unfold firstReductionPercentage at h
  sorry

end reduction_of_cycle_l228_228740


namespace smallest_w_factor_l228_228830

theorem smallest_w_factor:
  ∃ w : ℕ, (∃ n : ℕ, n = 936 * w ∧ 
              2 ^ 5 ∣ n ∧ 
              3 ^ 3 ∣ n ∧ 
              14 ^ 2 ∣ n) ∧ 
              w = 1764 :=
sorry

end smallest_w_factor_l228_228830


namespace rain_probability_at_most_3_days_l228_228624

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end rain_probability_at_most_3_days_l228_228624


namespace max_product_decomposition_l228_228839

theorem max_product_decomposition : ∃ x y : ℝ, x + y = 100 ∧ x * y = 50 * 50 := by
  sorry

end max_product_decomposition_l228_228839


namespace total_clients_correct_l228_228463

-- Define the number of each type of cars and total cars
def num_cars : ℕ := 12
def num_sedans : ℕ := 4
def num_coupes : ℕ := 4
def num_suvs : ℕ := 4

-- Define the number of selections per car and total selections required
def selections_per_car : ℕ := 3

-- Define the number of clients per type of car
def num_clients_who_like_sedans : ℕ := (num_sedans * selections_per_car) / 2
def num_clients_who_like_coupes : ℕ := (num_coupes * selections_per_car) / 2
def num_clients_who_like_suvs : ℕ := (num_suvs * selections_per_car) / 2

-- Compute total number of clients
def total_clients : ℕ := num_clients_who_like_sedans + num_clients_who_like_coupes + num_clients_who_like_suvs

-- Prove that the total number of clients is 18
theorem total_clients_correct : total_clients = 18 := by
  sorry

end total_clients_correct_l228_228463


namespace monomial_sum_exponents_l228_228710

theorem monomial_sum_exponents (m n : ℕ) (h₁ : m - 1 = 2) (h₂ : n = 2) : m^n = 9 := 
by
  sorry

end monomial_sum_exponents_l228_228710


namespace larger_number_of_hcf_lcm_l228_228729

theorem larger_number_of_hcf_lcm (hcf : ℕ) (a b : ℕ) (f1 f2 : ℕ) 
  (hcf_condition : hcf = 20) 
  (factors_condition : f1 = 21 ∧ f2 = 23) 
  (lcm_condition : Nat.lcm a b = hcf * f1 * f2):
  max a b = 460 := 
  sorry

end larger_number_of_hcf_lcm_l228_228729


namespace repeating_decimal_to_fraction_l228_228441

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end repeating_decimal_to_fraction_l228_228441


namespace inverse_proportional_ratios_l228_228529

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l228_228529


namespace circle_equation_and_shortest_chord_l228_228658

-- Definitions based on given conditions
def point_P : ℝ × ℝ := (4, -1)
def line_l1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line_l2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- The circle should be such that it intersects line l1 at point P and its center lies on line l2
theorem circle_equation_and_shortest_chord 
  (C : ℝ × ℝ) (r : ℝ) (hC_l2 : line_l2 C.1 C.2)
  (h_intersect : ∃ (k : ℝ), point_P.1 = (C.1 + k * (C.1 - point_P.1)) ∧ point_P.2 = (C.2 + k * (C.2 - point_P.2))) :
  -- Proving (1): Equation of the circle
  ((C.1 = 3) ∧ (C.2 = 5) ∧ r^2 = 37) ∧
  -- Proving (2): Length of the shortest chord through the origin is 2 * sqrt(3)
  (2 * Real.sqrt 3 = 2 * Real.sqrt (r^2 - ((C.1^2 + C.2^2) - (2 * C.1 * 0 + 2 * C.2 * 0)))) :=
by
  sorry

end circle_equation_and_shortest_chord_l228_228658


namespace find_a_b_max_min_values_l228_228521

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 + 2 * a * x + b

theorem find_a_b (a b : ℝ) :
  f' (-3) a b = 0 ∧ f (-3) a b = 9 → a = 1 ∧ b = -3 :=
  by sorry

theorem max_min_values (a b : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x a b ≥ -5 / 3 ∧ f x a b ≤ 9 :=
  by sorry

end find_a_b_max_min_values_l228_228521


namespace triangle_longest_side_l228_228269

theorem triangle_longest_side (y : ℝ) (h₁ : 8 + (y + 5) + (3 * y + 2) = 45) : 
  ∃ s1 s2 s3, s1 = 8 ∧ s2 = y + 5 ∧ s3 = 3 * y + 2 ∧ (s1 + s2 + s3 = 45) ∧ (s3 = 24.5) := 
by
  sorry

end triangle_longest_side_l228_228269


namespace distance_between_points_l228_228805

noncomputable def distance (x1 y1 x2 y2 : ℝ) := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : 
  distance (-3) (1/2) 4 (-7) = Real.sqrt 105.25 := 
by 
  sorry

end distance_between_points_l228_228805


namespace f_is_odd_and_increasing_l228_228528

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_is_odd_and_increasing_l228_228528


namespace sara_picked_6_pears_l228_228402

def total_pears : ℕ := 11
def tim_pears : ℕ := 5
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_6_pears : sara_pears = 6 := by
  sorry

end sara_picked_6_pears_l228_228402


namespace ratio_pentagon_rectangle_l228_228556

theorem ratio_pentagon_rectangle (s_p w : ℝ) (H_pentagon : 5 * s_p = 60) (H_rectangle : 6 * w = 80) : s_p / w = 9 / 10 :=
by
  sorry

end ratio_pentagon_rectangle_l228_228556
