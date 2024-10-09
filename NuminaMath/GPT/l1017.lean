import Mathlib

namespace g_g_g_g_3_eq_101_l1017_101723

def g (m : ℕ) : ℕ :=
  if m < 5 then m^2 + 1 else 2 * m + 3

theorem g_g_g_g_3_eq_101 : g (g (g (g 3))) = 101 :=
  by {
    -- the proof goes here
    sorry
  }

end g_g_g_g_3_eq_101_l1017_101723


namespace teairra_shirts_l1017_101770

theorem teairra_shirts (S : ℕ) (pants_total : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ)
  (pants_total_eq : pants_total = 24)
  (plaid_shirts_eq : plaid_shirts = 3)
  (purple_pants_eq : purple_pants = 5)
  (neither_plaid_nor_purple_eq : neither_plaid_nor_purple = 21) :
  (S - plaid_shirts + (pants_total - purple_pants) = neither_plaid_nor_purple) → S = 5 :=
by
  sorry

end teairra_shirts_l1017_101770


namespace total_students_correct_l1017_101747

noncomputable def num_roman_numerals : ℕ := 7
noncomputable def sketches_per_numeral : ℕ := 5
noncomputable def total_students : ℕ := 35

theorem total_students_correct : num_roman_numerals * sketches_per_numeral = total_students := by
  sorry

end total_students_correct_l1017_101747


namespace largest_angle_l1017_101705

theorem largest_angle (y : ℝ) (h : 40 + 70 + y = 180) : y = 70 :=
by
  sorry

end largest_angle_l1017_101705


namespace goods_train_speed_is_52_l1017_101782

def man_train_speed : ℕ := 60 -- speed of the man's train in km/h
def goods_train_length : ℕ := 280 -- length of the goods train in meters
def time_to_pass : ℕ := 9 -- time for the goods train to pass the man in seconds
def relative_speed_kmph : ℕ := (goods_train_length * 3600) / (time_to_pass * 1000) -- relative speed in km/h, calculated as (0.28 km / (9/3600) h)
def goods_train_speed : ℕ := relative_speed_kmph - man_train_speed -- speed of the goods train in km/h

theorem goods_train_speed_is_52 : goods_train_speed = 52 := by
  sorry

end goods_train_speed_is_52_l1017_101782


namespace fraction_multiplication_subtraction_l1017_101744

theorem fraction_multiplication_subtraction :
  (3 + 1 / 117) * (4 + 1 / 119) - (2 - 1 / 117) * (6 - 1 / 119) - (5 / 119) = 10 / 117 :=
by
  sorry

end fraction_multiplication_subtraction_l1017_101744


namespace eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l1017_101746

theorem eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256 :
  11^2 + 2 * 11 * 5 + 5^2 = 256 := by
  sorry

end eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l1017_101746


namespace percentage_increase_of_cube_surface_area_l1017_101706

-- Basic setup definitions and conditions
variable (a : ℝ)

-- Step 1: Initial surface area
def initial_surface_area : ℝ := 6 * a^2

-- Step 2: New edge length after 50% growth
def new_edge_length : ℝ := 1.5 * a

-- Step 3: New surface area after edge growth
def new_surface_area : ℝ := 6 * (new_edge_length a)^2

-- Step 4: Surface area after scaling by 1.5
def scaled_surface_area : ℝ := new_surface_area a * (1.5)^2

-- Prove the percentage increase
theorem percentage_increase_of_cube_surface_area :
  (scaled_surface_area a - initial_surface_area a) / initial_surface_area a * 100 = 406.25 := by
  sorry

end percentage_increase_of_cube_surface_area_l1017_101706


namespace total_investment_with_interest_l1017_101750

theorem total_investment_with_interest
  (total_investment : ℝ)
  (amount_at_3_percent : ℝ)
  (interest_rate_3 : ℝ)
  (interest_rate_5 : ℝ)
  (remaining_amount : ℝ)
  (interest_3 : ℝ)
  (interest_5 : ℝ) :
  total_investment = 1000 →
  amount_at_3_percent = 199.99999999999983 →
  interest_rate_3 = 0.03 →
  interest_rate_5 = 0.05 →
  remaining_amount = total_investment - amount_at_3_percent →
  interest_3 = amount_at_3_percent * interest_rate_3 →
  interest_5 = remaining_amount * interest_rate_5 →
  total_investment + interest_3 + remaining_amount + interest_5 = 1046 :=
by
  intros H1 H2 H3 H4 H5 H6 H7
  sorry

end total_investment_with_interest_l1017_101750


namespace triangle_side_c_l1017_101779

noncomputable def angle_B_eq_2A (A B : ℝ) := B = 2 * A
noncomputable def side_a_eq_1 (a : ℝ) := a = 1
noncomputable def side_b_eq_sqrt3 (b : ℝ) := b = Real.sqrt 3

noncomputable def find_side_c (A B C a b c : ℝ) :=
  angle_B_eq_2A A B ∧
  side_a_eq_1 a ∧
  side_b_eq_sqrt3 b →
  c = 2

theorem triangle_side_c (A B C a b c : ℝ) : find_side_c A B C a b c :=
by sorry

end triangle_side_c_l1017_101779


namespace solution_set_inequality_l1017_101757

theorem solution_set_inequality (x : ℝ) : 
  (2 < 1 / (x - 1) ∧ 1 / (x - 1) < 3) ↔ (4 / 3 < x ∧ x < 3 / 2) := 
by
  sorry

end solution_set_inequality_l1017_101757


namespace find_f1_plus_g1_l1017_101715

-- Definition of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Definition of g being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

-- Statement of the proof problem
theorem find_f1_plus_g1 
  (f g : ℝ → ℝ) 
  (hf : is_even_function f) 
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 2 :=
sorry

end find_f1_plus_g1_l1017_101715


namespace inequality_transformation_l1017_101791

theorem inequality_transformation (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = 2 * x + 3) (h2 : a > 0) (h3 : b > 0) :
  (∀ x, |f x + 5| < a → |x + 3| < b) ↔ b ≤ a / 2 :=
sorry

end inequality_transformation_l1017_101791


namespace solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l1017_101763

-- For the first polynomial equation
theorem solve_cubic_eq_a (x : ℝ) : x^3 - 3 * x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

-- For the second polynomial equation
theorem solve_cubic_eq_b (x : ℝ) : x^3 - 19 * x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
by sorry

-- For the third polynomial equation
theorem solve_cubic_eq_c (x : ℝ) : x^3 + 4 * x^2 + 6 * x + 4 = 0 ↔ x = -2 :=
by sorry

end solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l1017_101763


namespace count_three_digit_multiples_of_35_l1017_101730

theorem count_three_digit_multiples_of_35 : 
  ∃ n : ℕ, n = 26 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → (x % 35 = 0 → x = 35 * (3 + ((x / 35) - 3))) := 
sorry

end count_three_digit_multiples_of_35_l1017_101730


namespace min_value_x2_y2_z2_l1017_101720

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  4 ≤ x^2 + y^2 + z^2 :=
sorry

end min_value_x2_y2_z2_l1017_101720


namespace simplify_expression_l1017_101702

theorem simplify_expression (a b : ℝ) :
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 :=
by
  sorry

end simplify_expression_l1017_101702


namespace find_slope_l1017_101752

theorem find_slope (m : ℝ) : 
    (∀ x : ℝ, (2, 13) = (x, 5 * x + 3)) → 
    (∀ x : ℝ, (2, 13) = (x, m * x + 1)) → 
    m = 6 :=
by 
  intros hP hQ
  have h_inter_p := hP 2
  have h_inter_q := hQ 2
  simp at h_inter_p h_inter_q
  have : 13 = 5 * 2 + 3 := h_inter_p
  have : 13 = m * 2 + 1 := h_inter_q
  linarith

end find_slope_l1017_101752


namespace alpha_proportional_l1017_101714

theorem alpha_proportional (alpha beta gamma : ℝ) (h1 : ∀ β γ, (β = 15 ∧ γ = 3) → α = 5)
    (h2 : beta = 30) (h3 : gamma = 6) : alpha = 2.5 :=
sorry

end alpha_proportional_l1017_101714


namespace wire_cut_ratio_l1017_101783

-- Define lengths a and b
variable (a b : ℝ)

-- Define perimeter equal condition
axiom perimeter_eq : 4 * (a / 4) = 6 * (b / 6)

-- The statement to prove
theorem wire_cut_ratio (h : 4 * (a / 4) = 6 * (b / 6)) : a / b = 1 :=
by
  sorry

end wire_cut_ratio_l1017_101783


namespace octagon_ratio_l1017_101754

theorem octagon_ratio (total_area : ℝ) (area_below_PQ : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) (XQ QY : ℝ) :
  total_area = 10 ∧
  area_below_PQ = 5 ∧
  triangle_base = 5 ∧
  triangle_height = 8 / 5 ∧
  area_below_PQ = 1 + (1 / 2) * triangle_base * triangle_height ∧
  XQ + QY = triangle_base ∧
  (1 / 2) * (XQ + QY) * triangle_height = 5
  → (XQ / QY) = 2 / 3 := 
sorry

end octagon_ratio_l1017_101754


namespace find_certain_number_l1017_101768

-- Define the conditions
variable (m : ℕ)
variable (h_lcm : Nat.lcm 24 m = 48)
variable (h_gcd : Nat.gcd 24 m = 8)

-- State the theorem to prove
theorem find_certain_number (h_lcm : Nat.lcm 24 m = 48) (h_gcd : Nat.gcd 24 m = 8) : m = 16 :=
sorry

end find_certain_number_l1017_101768


namespace probability_event_comparison_l1017_101743

theorem probability_event_comparison (m n : ℕ) :
  let P_A := (2 * m * n) / (m + n)^2
  let P_B := (m^2 + n^2) / (m + n)^2
  P_A ≤ P_B ∧ (P_A = P_B ↔ m = n) :=
by
  sorry

end probability_event_comparison_l1017_101743


namespace anyas_hair_loss_l1017_101777

theorem anyas_hair_loss (H : ℝ) 
  (washes_hair_loss : H > 0) 
  (brushes_hair_loss : H / 2 > 0) 
  (grows_back : ∃ h : ℝ, h = 49 ∧ H + H / 2 + 1 = h) :
  H = 32 :=
by
  sorry

end anyas_hair_loss_l1017_101777


namespace circle_n_gon_area_ineq_l1017_101707

variable {n : ℕ} {S S1 S2 : ℝ}

theorem circle_n_gon_area_ineq (h1 : S1 > 0) (h2 : S > 0) (h3 : S2 > 0) : 
  S * S = S1 * S2 := 
sorry

end circle_n_gon_area_ineq_l1017_101707


namespace find_a_and_b_l1017_101789

theorem find_a_and_b (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {2, 3}) 
  (hB : B = {x | x^2 + a * x + b = 0}) 
  (h_intersection : A ∩ B = {2}) 
  (h_union : A ∪ B = A) : 
  (a + b = 0) ∨ (a + b = 1) := 
sorry

end find_a_and_b_l1017_101789


namespace g_three_eighths_l1017_101722

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l1017_101722


namespace red_candies_difference_l1017_101741

def jar1_ratio_red : ℕ := 7
def jar1_ratio_yellow : ℕ := 3
def jar2_ratio_red : ℕ := 5
def jar2_ratio_yellow : ℕ := 4
def total_yellow : ℕ := 108

theorem red_candies_difference :
  ∀ (x y : ℚ), jar1_ratio_yellow * x + jar2_ratio_yellow * y = total_yellow ∧ jar1_ratio_red + jar1_ratio_yellow = jar2_ratio_red + jar2_ratio_yellow → 10 * x = 9 * y → 7 * x - 5 * y = 21 := 
by sorry

end red_candies_difference_l1017_101741


namespace quadratic_completing_the_square_q_l1017_101736

theorem quadratic_completing_the_square_q (x p q : ℝ) (h : 4 * x^2 + 8 * x - 468 = 0) :
  (∃ p, (x + p)^2 = q) → q = 116 := sorry

end quadratic_completing_the_square_q_l1017_101736


namespace algebraic_inequality_solution_l1017_101748

theorem algebraic_inequality_solution (x : ℝ) : (1 + 2 * x ≤ 8 + 3 * x) → (x ≥ -7) :=
by
  sorry

end algebraic_inequality_solution_l1017_101748


namespace andrew_total_donation_l1017_101726

/-
Problem statement:
Andrew started donating 7k to an organization on his 11th birthday. Yesterday, Andrew turned 29.
Verify that the total amount Andrew has donated is 126k.
-/

theorem andrew_total_donation 
  (annual_donation : ℕ := 7000) 
  (start_age : ℕ := 11) 
  (current_age : ℕ := 29) 
  (years_donating : ℕ := current_age - start_age) 
  (total_donated : ℕ := annual_donation * years_donating) :
  total_donated = 126000 := 
by 
  sorry

end andrew_total_donation_l1017_101726


namespace new_students_weights_correct_l1017_101799

-- Definitions of the initial conditions
def initial_student_count : ℕ := 29
def initial_avg_weight : ℚ := 28
def total_initial_weight := initial_student_count * initial_avg_weight
def new_student_counts : List ℕ := [30, 31, 32, 33]
def new_avg_weights : List ℚ := [27.2, 27.8, 27.6, 28]

-- Weights of the four new students
def W1 : ℚ := 4
def W2 : ℚ := 45.8
def W3 : ℚ := 21.4
def W4 : ℚ := 40.8

-- The proof statement
theorem new_students_weights_correct :
  total_initial_weight = 812 ∧
  W1 = 4 ∧
  W2 = 45.8 ∧
  W3 = 21.4 ∧
  W4 = 40.8 ∧
  (total_initial_weight + W1) = 816 ∧
  (total_initial_weight + W1) / new_student_counts.head! = new_avg_weights.head! ∧
  (total_initial_weight + W1 + W2) = 861.8 ∧
  (total_initial_weight + W1 + W2) / new_student_counts.tail.head! = new_avg_weights.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3) = 883.2 ∧
  (total_initial_weight + W1 + W2 + W3) / new_student_counts.tail.tail.head! = new_avg_weights.tail.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3 + W4) = 924 ∧
  (total_initial_weight + W1 + W2 + W3 + W4) / new_student_counts.tail.tail.tail.head! = new_avg_weights.tail.tail.tail.head! :=
by
  sorry

end new_students_weights_correct_l1017_101799


namespace find_f_prime_at_1_l1017_101792

variable (f : ℝ → ℝ)

-- Initial condition
variable (h : ∀ x, f x = x^2 + deriv f 2 * (Real.log x - x))

-- The goal is to prove that f'(1) = 2
theorem find_f_prime_at_1 : deriv f 1 = 2 :=
by
  sorry

end find_f_prime_at_1_l1017_101792


namespace abs_neg_three_l1017_101758

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l1017_101758


namespace gas_pipe_probability_l1017_101773

-- Define the conditions as Lean hypotheses
theorem gas_pipe_probability (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : x + y ≤ 100) (h25x : 25 ≤ x) (h25y : 25 ≤ y)
    (h100xy : 75 ≥ x + y) :
  ∃ (p : ℝ), p = 1/16 :=
by
  sorry

end gas_pipe_probability_l1017_101773


namespace two_perfect_squares_not_two_perfect_cubes_l1017_101760

-- Define the initial conditions as Lean assertions
def isSumOfTwoPerfectSquares (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^2

def isSumOfTwoPerfectCubes (n : ℕ) := ∃ a b : ℕ, n = a^3 + b^3

-- Lean 4 statement to show 2005^2005 is a sum of two perfect squares
theorem two_perfect_squares :
  isSumOfTwoPerfectSquares (2005^2005) :=
sorry

-- Lean 4 statement to show 2005^2005 is not a sum of two perfect cubes
theorem not_two_perfect_cubes :
  ¬ isSumOfTwoPerfectCubes (2005^2005) :=
sorry

end two_perfect_squares_not_two_perfect_cubes_l1017_101760


namespace part1_part2_part3_l1017_101756

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def set_C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

theorem part1 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∪ set_B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (∅ ⊂ (set_A a ∩ set_B)) ∧ (set_A a ∩ set_C = ∅) → a = -2 :=
by
  sorry

theorem part3 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∩ set_C) ∧ (set_A a ∩ set_B ≠ ∅) → a = -3 :=
by
  sorry

end part1_part2_part3_l1017_101756


namespace cubic_inches_in_two_cubic_feet_l1017_101703

theorem cubic_inches_in_two_cubic_feet :
  (12 ^ 3) * 2 = 3456 := by
  sorry

end cubic_inches_in_two_cubic_feet_l1017_101703


namespace train_pass_time_l1017_101718

theorem train_pass_time
  (v : ℝ) (l_tunnel l_train : ℝ) (h_v : v = 75) (h_l_tunnel : l_tunnel = 3.5) (h_l_train : l_train = 0.25) :
  (l_tunnel + l_train) / v * 60 = 3 :=
by 
  -- Placeholder for the proof
  sorry

end train_pass_time_l1017_101718


namespace total_time_to_4864_and_back_l1017_101731

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance : ℝ)
variable (Sboat : speed_boat = 14) (Sstream : speed_stream = 1.2) (Dist : distance = 4864)

theorem total_time_to_4864_and_back :
  let speed_downstream := speed_boat + speed_stream
  let speed_upstream := speed_boat - speed_stream
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_time := time_downstream + time_upstream
  total_time = 700 :=
by
  sorry

end total_time_to_4864_and_back_l1017_101731


namespace polar_bear_daily_fish_intake_l1017_101764

theorem polar_bear_daily_fish_intake : 
  (0.2 + 0.4 = 0.6) := by
  sorry

end polar_bear_daily_fish_intake_l1017_101764


namespace xy_sum_value_l1017_101755

theorem xy_sum_value (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
sorry

end xy_sum_value_l1017_101755


namespace fred_money_last_week_l1017_101733

theorem fred_money_last_week (F_current F_earned F_last_week : ℕ) 
  (h_current : F_current = 86)
  (h_earned : F_earned = 63)
  (h_last_week : F_last_week = 23) :
  F_current - F_earned = F_last_week := 
by
  sorry

end fred_money_last_week_l1017_101733


namespace polygon_sides_l1017_101709

theorem polygon_sides :
  ∃ (n : ℕ), (n * (n - 3) / 2) = n + 33 ∧ n = 11 :=
by
  sorry

end polygon_sides_l1017_101709


namespace tanC_over_tanA_plus_tanC_over_tanB_l1017_101738

theorem tanC_over_tanA_plus_tanC_over_tanB {a b c : ℝ} (A B C : ℝ) (h : a / b + b / a = 6 * Real.cos C) (acute_triangle : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
sorry -- Proof not required

end tanC_over_tanA_plus_tanC_over_tanB_l1017_101738


namespace triangle_perimeter_l1017_101775

theorem triangle_perimeter
  (a : ℝ) (a_gt_5 : a > 5)
  (ellipse : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 25 = 1)
  (dist_foci : 8 = 2 * 4) :
  4 * Real.sqrt (41) = 4 * Real.sqrt (41) := by
sorry

end triangle_perimeter_l1017_101775


namespace binom_coeffs_not_coprime_l1017_101700

open Nat

theorem binom_coeffs_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) : 
  Nat.gcd (Nat.choose n k) (Nat.choose n m) > 1 := 
sorry

end binom_coeffs_not_coprime_l1017_101700


namespace unknown_number_lcm_hcf_l1017_101794

theorem unknown_number_lcm_hcf (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 192) 
  (hcf_ab : Nat.gcd a b = 16) 
  (known_number : a = 64) :
  b = 48 :=
by
  sorry -- Proof is omitted as per instruction

end unknown_number_lcm_hcf_l1017_101794


namespace find_a_and_c_range_of_m_l1017_101708

theorem find_a_and_c (a c : ℝ) 
  (h : ∀ x, 1 < x ∧ x < 3 ↔ ax^2 + x + c > 0) 
  : a = -1/4 ∧ c = -3/4 := 
sorry

theorem range_of_m (m : ℝ) 
  (h : ∀ x, (-1/4)*x^2 + 2*x - 3 > 0 → x + m > 0) 
  : m ≥ -2 :=
sorry

end find_a_and_c_range_of_m_l1017_101708


namespace sofia_initial_floor_l1017_101769

theorem sofia_initial_floor (x : ℤ) (h1 : x + 7 - 6 + 5 = 20) : x = 14 := 
sorry

end sofia_initial_floor_l1017_101769


namespace lcm_of_numbers_l1017_101793

theorem lcm_of_numbers (a b c d : ℕ) (h1 : a = 8) (h2 : b = 24) (h3 : c = 36) (h4 : d = 54) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 216 := 
by 
  sorry

end lcm_of_numbers_l1017_101793


namespace freds_average_book_cost_l1017_101795

theorem freds_average_book_cost :
  ∀ (initial_amount spent_amount num_books remaining_amount avg_cost : ℕ),
    initial_amount = 236 →
    remaining_amount = 14 →
    num_books = 6 →
    spent_amount = initial_amount - remaining_amount →
    avg_cost = spent_amount / num_books →
    avg_cost = 37 :=
by
  intros initial_amount spent_amount num_books remaining_amount avg_cost h_init h_rem h_books h_spent h_avg
  sorry

end freds_average_book_cost_l1017_101795


namespace problem_inequality_l1017_101727

theorem problem_inequality (a : ℝ) (h_pos : 0 < a) : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a :=
by 
  sorry

end problem_inequality_l1017_101727


namespace find_older_friend_age_l1017_101787

theorem find_older_friend_age (A B C : ℕ) 
  (h1 : A - B = 2) 
  (h2 : A - C = 5) 
  (h3 : A + B + C = 110) : 
  A = 39 := 
by 
  sorry

end find_older_friend_age_l1017_101787


namespace rectangle_width_l1017_101785

-- The Lean statement only with given conditions and the final proof goal
theorem rectangle_width (w l : ℕ) (P : ℕ) (h1 : l = w - 3) (h2 : P = 2 * w + 2 * l) (h3 : P = 54) :
  w = 15 :=
by
  sorry

end rectangle_width_l1017_101785


namespace range_of_f_minus_2_l1017_101778

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_minus_2 (a b : ℝ) (h1 : 1 ≤ f (-1) a b) (h2 : f (-1) a b ≤ 2) (h3 : 2 ≤ f 1 a b) (h4 : f 1 a b ≤ 4) :
  6 ≤ f (-2) a b ∧ f (-2) a b ≤ 10 :=
sorry

end range_of_f_minus_2_l1017_101778


namespace geometric_series_3000_terms_sum_l1017_101784

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end geometric_series_3000_terms_sum_l1017_101784


namespace count_four_digit_numbers_l1017_101749

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l1017_101749


namespace fourth_square_area_l1017_101759

theorem fourth_square_area (AB BC CD AC x : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_AC1 : AC^2 = AB^2 + BC^2) 
  (h_AC2 : AC^2 = CD^2 + x^2) :
  x^2 = 10 :=
by
  sorry

end fourth_square_area_l1017_101759


namespace solve_system_for_x_l1017_101725

theorem solve_system_for_x :
  ∃ x y : ℝ, (2 * x + y = 4) ∧ (x + 2 * y = 5) ∧ (x = 1) :=
by
  sorry

end solve_system_for_x_l1017_101725


namespace trajectory_equation_no_such_point_l1017_101796

-- Conditions for (I): The ratio of the distances is given
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let M := (1, 0)
  let N := (4, 0)
  2 * Real.sqrt ((P.1 - M.1)^2 + P.2^2) = Real.sqrt ((P.1 - N.1)^2 + P.2^2)

-- Proof of (I): Find the trajectory equation of point P
theorem trajectory_equation : 
  ∀ P : ℝ × ℝ, ratio_condition P → P.1^2 + P.2^2 = 4 :=
by
  sorry

-- Conditions for (II): Given points A, B, C
def points_condition (P : ℝ × ℝ) : Prop :=
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (-4, 2)
  (P.1 + 2)^2 + (P.2 + 2)^2 + 
  (P.1 + 2)^2 + (P.2 - 6)^2 + 
  (P.1 + 4)^2 + (P.2 - 2)^2 = 36

-- Proof of (II): Determine the non-existence of point P
theorem no_such_point (P : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 4 → ¬ points_condition P :=
by
  sorry

end trajectory_equation_no_such_point_l1017_101796


namespace jellybean_ratio_l1017_101716

theorem jellybean_ratio (jellybeans_large: ℕ) (large_glasses: ℕ) (small_glasses: ℕ) (total_jellybeans: ℕ) (jellybeans_per_large: ℕ) (jellybeans_per_small: ℕ)
  (h1 : jellybeans_large = 50)
  (h2 : large_glasses = 5)
  (h3 : small_glasses = 3)
  (h4 : total_jellybeans = 325)
  (h5 : jellybeans_per_large = jellybeans_large * large_glasses)
  (h6 : jellybeans_per_small * small_glasses = total_jellybeans - jellybeans_per_large)
  : jellybeans_per_small = 25 ∧ jellybeans_per_small / jellybeans_large = 1 / 2 :=
by
  sorry

end jellybean_ratio_l1017_101716


namespace selling_price_per_machine_l1017_101735

theorem selling_price_per_machine (parts_cost patent_cost : ℕ) (num_machines : ℕ) 
  (hc1 : parts_cost = 3600) (hc2 : patent_cost = 4500) (hc3 : num_machines = 45) :
  (parts_cost + patent_cost) / num_machines = 180 :=
by
  sorry

end selling_price_per_machine_l1017_101735


namespace factor_of_polynomial_l1017_101762

def polynomial (x : ℝ) : ℝ := x^4 - 4*x^2 + 16
def q1 (x : ℝ) : ℝ := x^2 + 4
def q2 (x : ℝ) : ℝ := x - 2
def q3 (x : ℝ) : ℝ := x^2 - 4*x + 4
def q4 (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem factor_of_polynomial : (∃ (f g : ℝ → ℝ), polynomial x = f x * g x) ∧ (q4 = f ∨ q4 = g) := by sorry

end factor_of_polynomial_l1017_101762


namespace scientific_notation_140000000_l1017_101790

theorem scientific_notation_140000000 :
  140000000 = 1.4 * 10^8 := 
sorry

end scientific_notation_140000000_l1017_101790


namespace race_distance_l1017_101737

theorem race_distance (a b c : ℝ) (d : ℝ) 
  (h1 : d / a = (d - 15) / b)
  (h2 : d / b = (d - 30) / c)
  (h3 : d / a = (d - 40) / c) : 
  d = 90 :=
by sorry

end race_distance_l1017_101737


namespace mod_remainder_l1017_101797

theorem mod_remainder :
  ((85^70 + 19^32)^16) % 21 = 16 := by
  -- Given conditions
  have h1 : 85^70 % 21 = 1 := sorry
  have h2 : 19^32 % 21 = 4 := sorry
  -- Conclusion
  sorry

end mod_remainder_l1017_101797


namespace value_of_g_13_l1017_101710

def g (n : ℕ) : ℕ := n^2 + 2 * n + 23

theorem value_of_g_13 : g 13 = 218 :=
by 
  sorry

end value_of_g_13_l1017_101710


namespace find_angle_B_max_value_a_squared_plus_c_squared_l1017_101742

variable {A B C : ℝ} -- Angles A, B, C in radians
variable {a b c : ℝ} -- Sides opposite to these angles

-- Problem 1
theorem find_angle_B (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : B = Real.pi / 3 :=
by
  sorry -- Proof is not needed

-- Problem 2
theorem max_value_a_squared_plus_c_squared (h : b = Real.sqrt 3)
  (h' : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : (a^2 + c^2) ≤ 6 :=
by
  sorry -- Proof is not needed

end find_angle_B_max_value_a_squared_plus_c_squared_l1017_101742


namespace rectangle_area_l1017_101713

-- Declare the given conditions
def circle_radius : ℝ := 5
def rectangle_width : ℝ := 2 * circle_radius
def length_to_width_ratio : ℝ := 2

-- Given that the length to width ratio is 2:1, calculate the length
def rectangle_length : ℝ := length_to_width_ratio * rectangle_width

-- Define the statement we need to prove
theorem rectangle_area :
  rectangle_length * rectangle_width = 200 :=
by
  sorry

end rectangle_area_l1017_101713


namespace original_number_l1017_101776

theorem original_number (n : ℚ) (h : (3 * (n + 3) - 2) / 3 = 10) : n = 23 / 3 := 
sorry

end original_number_l1017_101776


namespace b_over_c_equals_1_l1017_101719

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end b_over_c_equals_1_l1017_101719


namespace mass_percentage_B_in_H3BO3_l1017_101740

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_B : ℝ := 10.81
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_H3BO3 : ℝ := 3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

theorem mass_percentage_B_in_H3BO3 : (atomic_mass_B / molar_mass_H3BO3) * 100 = 17.48 :=
by
  sorry

end mass_percentage_B_in_H3BO3_l1017_101740


namespace Maria_needs_more_l1017_101765

def num_mechanics : Nat := 20
def num_thermodynamics : Nat := 50
def num_optics : Nat := 30
def total_questions : Nat := num_mechanics + num_thermodynamics + num_optics

def correct_mechanics : Nat := (80 * num_mechanics) / 100
def correct_thermodynamics : Nat := (50 * num_thermodynamics) / 100
def correct_optics : Nat := (70 * num_optics) / 100
def correct_total : Nat := correct_mechanics + correct_thermodynamics + correct_optics

def correct_for_passing : Nat := (65 * total_questions) / 100
def additional_needed : Nat := correct_for_passing - correct_total

theorem Maria_needs_more:
  additional_needed = 3 := by
  sorry

end Maria_needs_more_l1017_101765


namespace find_n_l1017_101745

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : ∃ k : ℤ, 721 = n + 360 * k): n = 1 :=
sorry

end find_n_l1017_101745


namespace find_y_coordinate_l1017_101701

-- Define points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the property that a point P satisfies PA + PD = PB + PC = 10
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PD := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PA + PD = 10 ∧ PB + PC = 10

-- Lean statement to prove the y-coordinate of P that satisfies the condition
theorem find_y_coordinate :
  ∃ (P : ℝ × ℝ), satisfies_condition P ∧ ∃ (a b c d : ℕ), a = 0 ∧ b = 1 ∧ c = 21 ∧ d = 3 ∧ P.2 = (14 + Real.sqrt 21) / 3 ∧ a + b + c + d = 25 :=
by
  sorry

end find_y_coordinate_l1017_101701


namespace combined_stripes_is_22_l1017_101761

-- Definition of stripes per shoe for each person based on the conditions
def stripes_per_shoe_Olga : ℕ := 3
def stripes_per_shoe_Rick : ℕ := stripes_per_shoe_Olga - 1
def stripes_per_shoe_Hortense : ℕ := stripes_per_shoe_Olga * 2

-- The total combined number of stripes on all shoes for Olga, Rick, and Hortense
def total_stripes : ℕ := 2 * (stripes_per_shoe_Olga + stripes_per_shoe_Rick + stripes_per_shoe_Hortense)

-- The statement to prove that the total number of stripes on all their shoes is 22
theorem combined_stripes_is_22 : total_stripes = 22 :=
by
  sorry

end combined_stripes_is_22_l1017_101761


namespace product_of_roots_of_cubic_l1017_101739

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end product_of_roots_of_cubic_l1017_101739


namespace loss_percentage_is_13_l1017_101717

def cost_price : ℕ := 1500
def selling_price : ℕ := 1305
def loss : ℕ := cost_price - selling_price
def loss_percentage : ℚ := (loss : ℚ) / cost_price * 100

theorem loss_percentage_is_13 :
  loss_percentage = 13 := 
by
  sorry

end loss_percentage_is_13_l1017_101717


namespace total_books_l1017_101732

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end total_books_l1017_101732


namespace period_start_time_l1017_101798

theorem period_start_time (end_time : ℕ) (rained_hours : ℕ) (not_rained_hours : ℕ) (total_hours : ℕ) (start_time : ℕ) 
  (h1 : end_time = 17) -- 5 pm as 17 in 24-hour format 
  (h2 : rained_hours = 2)
  (h3 : not_rained_hours = 6)
  (h4 : total_hours = rained_hours + not_rained_hours)
  (h5 : total_hours = 8)
  (h6 : start_time = end_time - total_hours)
  : start_time = 9 :=
sorry

end period_start_time_l1017_101798


namespace fatima_total_donation_l1017_101771

theorem fatima_total_donation :
  let cloth1 := 100
  let cloth1_piece1 := 0.40 * cloth1
  let cloth1_piece2 := 0.30 * cloth1
  let cloth1_piece3 := 0.30 * cloth1
  let donation1 := cloth1_piece2 + cloth1_piece3

  let cloth2 := 65
  let cloth2_piece1 := 0.55 * cloth2
  let cloth2_piece2 := 0.45 * cloth2
  let donation2 := cloth2_piece2

  let cloth3 := 48
  let cloth3_piece1 := 0.60 * cloth3
  let cloth3_piece2 := 0.40 * cloth3
  let donation3 := cloth3_piece2

  donation1 + donation2 + donation3 = 108.45 :=
by
  sorry

end fatima_total_donation_l1017_101771


namespace trig_expression_tangent_l1017_101780

theorem trig_expression_tangent (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 :=
sorry

end trig_expression_tangent_l1017_101780


namespace simplify_expression_eq_69_l1017_101712

theorem simplify_expression_eq_69 : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end simplify_expression_eq_69_l1017_101712


namespace find_x_values_l1017_101751

theorem find_x_values (x : ℝ) :
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ - (12 / 7 : ℝ) ≤ x ∧ x < - (3 / 4 : ℝ) :=
by
  sorry

end find_x_values_l1017_101751


namespace find_X_value_l1017_101774

theorem find_X_value (X : ℝ) : 
  (1.5 * ((3.6 * 0.48 * 2.5) / (0.12 * X * 0.5)) = 1200.0000000000002) → 
  X = 0.225 :=
by
  sorry

end find_X_value_l1017_101774


namespace half_sum_of_squares_of_even_or_odd_l1017_101729

theorem half_sum_of_squares_of_even_or_odd (n1 n2 : ℤ) (a b : ℤ) :
  (n1 % 2 = 0 ∧ n2 % 2 = 0 ∧ n1 = 2*a ∧ n2 = 2*b ∨
   n1 % 2 = 1 ∧ n2 % 2 = 1 ∧ n1 = 2*a + 1 ∧ n2 = 2*b + 1) →
  ∃ x y : ℤ, (n1^2 + n2^2) / 2 = x^2 + y^2 :=
by
  intro h
  sorry

end half_sum_of_squares_of_even_or_odd_l1017_101729


namespace coplanar_vertices_sum_even_l1017_101753

theorem coplanar_vertices_sum_even (a b c d e f g h : ℤ) :
  (∃ (a b c d : ℤ), true ∧ (a + b + c + d) % 2 = 0) :=
sorry

end coplanar_vertices_sum_even_l1017_101753


namespace fractions_product_l1017_101772

theorem fractions_product :
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end fractions_product_l1017_101772


namespace integer_solution_pairs_l1017_101788

theorem integer_solution_pairs (a b : ℕ) (h_pos : a > 0 ∧ b > 0):
  (∃ k : ℕ, k > 0 ∧ a^2 = k * (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, l > 0 ∧ ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
sorry

end integer_solution_pairs_l1017_101788


namespace integral_solutions_count_l1017_101711

theorem integral_solutions_count (m : ℕ) (h : m > 0) :
  ∃ S : Finset (ℕ × ℕ), S.card = m ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + p.2^2 + 2 * p.1 * p.2 - m * p.1 - m * p.2 - m - 1 = 0) := 
sorry

end integral_solutions_count_l1017_101711


namespace number_of_participants_l1017_101728

theorem number_of_participants (n : ℕ) (hn : n = 862) 
    (h_lower : 575 ≤ n * 2 / 3) 
    (h_upper : n * 7 / 9 ≤ 670) : 
    ∃ p, (575 ≤ p) ∧ (p ≤ 670) ∧ (p % 11 = 0) ∧ ((p - 575) / 11 + 1 = 8) :=
by
  sorry

end number_of_participants_l1017_101728


namespace feb1_is_wednesday_l1017_101734

-- Define the days of the week as a data type
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define a function that models the backward count for days of the week from a given day
def days_backward (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => start
  | 1 => match start with
         | Sunday => Saturday
         | Monday => Sunday
         | Tuesday => Monday
         | Wednesday => Tuesday
         | Thursday => Wednesday
         | Friday => Thursday
         | Saturday => Friday
  | 2 => match start with
         | Sunday => Friday
         | Monday => Saturday
         | Tuesday => Sunday
         | Wednesday => Monday
         | Thursday => Tuesday
         | Friday => Wednesday
         | Saturday => Thursday
  | 3 => match start with
         | Sunday => Thursday
         | Monday => Friday
         | Tuesday => Saturday
         | Wednesday => Sunday
         | Thursday => Monday
         | Friday => Tuesday
         | Saturday => Wednesday
  | 4 => match start with
         | Sunday => Wednesday
         | Monday => Thursday
         | Tuesday => Friday
         | Wednesday => Saturday
         | Thursday => Sunday
         | Friday => Monday
         | Saturday => Tuesday
  | 5 => match start with
         | Sunday => Tuesday
         | Monday => Wednesday
         | Tuesday => Thursday
         | Wednesday => Friday
         | Thursday => Saturday
         | Friday => Sunday
         | Saturday => Monday
  | 6 => match start with
         | Sunday => Monday
         | Monday => Tuesday
         | Tuesday => Wednesday
         | Wednesday => Thursday
         | Thursday => Friday
         | Friday => Saturday
         | Saturday => Sunday
  | _ => start  -- This case is unreachable because days % 7 is always between 0 and 6

-- Proof statement: given February 28 is a Tuesday, prove that February 1 is a Wednesday
theorem feb1_is_wednesday (h : days_backward Tuesday 27 = Wednesday) : True :=
by
  sorry

end feb1_is_wednesday_l1017_101734


namespace problem_statement_l1017_101781

noncomputable def expr : ℝ :=
  (1 - Real.sqrt 5)^0 + abs (-Real.sqrt 2) - 2 * Real.cos (Real.pi / 4) + (1 / 4 : ℝ)⁻¹

theorem problem_statement : expr = 5 := by
  sorry

end problem_statement_l1017_101781


namespace largest_three_digit_product_l1017_101721

theorem largest_three_digit_product : 
    ∃ (n : ℕ), 
    (n = 336) ∧ 
    (n > 99 ∧ n < 1000) ∧ 
    (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = x * y * (5 * x + 2 * y) ∧ 
        ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ k * m = (5 * x + 2 * y)) :=
by
  sorry

end largest_three_digit_product_l1017_101721


namespace wire_not_used_is_20_l1017_101767

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l1017_101767


namespace difference_nickels_is_8q_minus_20_l1017_101786

variable (q : ℤ)

-- Define the number of quarters for Alice and Bob
def alice_quarters : ℤ := 7 * q - 3
def bob_quarters : ℤ := 3 * q + 7

-- Define the worth of a quarter in nickels
def quarter_to_nickels (quarters : ℤ) : ℤ := 2 * quarters

-- Define the difference in quarters
def difference_quarters : ℤ := alice_quarters q - bob_quarters q

-- Define the difference in their amount of money in nickels
def difference_nickels (q : ℤ) : ℤ := quarter_to_nickels (difference_quarters q)

theorem difference_nickels_is_8q_minus_20 : difference_nickels q = 8 * q - 20 := by
  sorry

end difference_nickels_is_8q_minus_20_l1017_101786


namespace min_value_fractions_l1017_101766

theorem min_value_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2 ≤ (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) :=
by sorry

end min_value_fractions_l1017_101766


namespace f_g_2_eq_256_l1017_101704

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem f_g_2_eq_256 : f (g 2) = 256 := by
  sorry

end f_g_2_eq_256_l1017_101704


namespace solve_problem_for_m_n_l1017_101724

theorem solve_problem_for_m_n (m n : ℕ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m * (n + m) = n * (n - m)) :
  ((∃ h : ℕ, m = (2 * h + 1) * h ∧ n = (2 * h + 1) * (h + 1)) ∨ 
   (∃ h : ℕ, h > 0 ∧ m = 2 * h * (4 * h^2 - 1) ∧ n = 2 * h * (4 * h^2 + 1))) := 
sorry

end solve_problem_for_m_n_l1017_101724
