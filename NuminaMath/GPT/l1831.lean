import Mathlib

namespace derivative_evaluation_at_pi_over_3_l1831_183195

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x) + Real.tan x

theorem derivative_evaluation_at_pi_over_3 :
  deriv f (Real.pi / 3) = 3 :=
sorry

end derivative_evaluation_at_pi_over_3_l1831_183195


namespace proof_problem_l1831_183125

variable (α β : ℝ)

def interval_αβ : Prop := 
  α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ 
  β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

def condition : Prop := α * Real.sin α - β * Real.sin β > 0

theorem proof_problem (h1 : interval_αβ α β) (h2 : condition α β) : α ^ 2 > β ^ 2 := 
sorry

end proof_problem_l1831_183125


namespace carl_max_value_carry_l1831_183193

variables (rock_weight_3_pound : ℕ := 3) (rock_value_3_pound : ℕ := 9)
          (rock_weight_6_pound : ℕ := 6) (rock_value_6_pound : ℕ := 20)
          (rock_weight_2_pound : ℕ := 2) (rock_value_2_pound : ℕ := 5)
          (weight_limit : ℕ := 20)
          (max_six_pound_rocks : ℕ := 2)

noncomputable def max_value_carry : ℕ :=
  max (2 * rock_value_6_pound + 2 * rock_value_3_pound) 
      (4 * rock_value_3_pound + 4 * rock_value_2_pound)

theorem carl_max_value_carry : max_value_carry = 58 :=
by sorry

end carl_max_value_carry_l1831_183193


namespace min_value_of_a_l1831_183184

-- Defining the properties of the function f
variable {f : ℝ → ℝ}
variable (even_f : ∀ x, f x = f (-x))
variable (mono_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Necessary condition involving f and a
variable {a : ℝ}
variable (a_condition : f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1)

-- Main statement proving that the minimum value of a is 1/2
theorem min_value_of_a : a = 1/2 :=
sorry

end min_value_of_a_l1831_183184


namespace carla_games_won_l1831_183121

theorem carla_games_won (F C : ℕ) (h1 : F + C = 30) (h2 : F = C / 2) : C = 20 :=
by
  sorry

end carla_games_won_l1831_183121


namespace find_number_l1831_183181

theorem find_number (x : ℕ) (h : x / 46 - 27 = 46) : x = 3358 :=
by
  sorry

end find_number_l1831_183181


namespace total_people_hired_l1831_183147

theorem total_people_hired (H L : ℕ) (hL : L = 1) (payroll : ℕ) (hPayroll : 129 * H + 82 * L = 3952) : H + L = 31 := by
  sorry

end total_people_hired_l1831_183147


namespace ratio_of_larger_to_smaller_l1831_183119

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l1831_183119


namespace exists_unique_root_in_interval_l1831_183112

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_unique_root_in_interval : 
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
sorry

end exists_unique_root_in_interval_l1831_183112


namespace max_profit_l1831_183187

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l1831_183187


namespace museum_ticket_cost_l1831_183199

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l1831_183199


namespace area_of_curvilinear_trapezoid_steps_l1831_183145

theorem area_of_curvilinear_trapezoid_steps (steps : List String) :
  (steps = ["division", "approximation", "summation", "taking the limit"]) :=
sorry

end area_of_curvilinear_trapezoid_steps_l1831_183145


namespace john_remaining_money_l1831_183118

theorem john_remaining_money (q : ℝ) : 
  let drink_cost := 5 * q
  let medium_pizza_cost := 3 * 2 * q
  let large_pizza_cost := 2 * 3 * q
  let dessert_cost := 4 * (1 / 2) * q
  let total_cost := drink_cost + medium_pizza_cost + large_pizza_cost + dessert_cost
  let initial_money := 60
  initial_money - total_cost = 60 - 19 * q :=
by
  sorry

end john_remaining_money_l1831_183118


namespace age_difference_28_l1831_183146

variable (li_lin_age_father_sum li_lin_age_future father_age_future : ℕ)

theorem age_difference_28 
    (h1 : li_lin_age_father_sum = 50)
    (h2 : ∀ x, li_lin_age_future = x → father_age_future = 3 * x - 2)
    (h3 : li_lin_age_future + 4 = li_lin_age_father_sum + 8 - (father_age_future + 4))
    : li_lin_age_father_sum - li_lin_age_future = 28 :=
sorry

end age_difference_28_l1831_183146


namespace abigail_monthly_saving_l1831_183109

-- Definitions based on the conditions
def total_saving := 48000
def months_in_year := 12

-- The statement to be proved
theorem abigail_monthly_saving : total_saving / months_in_year = 4000 :=
by sorry

end abigail_monthly_saving_l1831_183109


namespace rate_per_sqm_is_correct_l1831_183123

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l1831_183123


namespace find_k_l1831_183198

theorem find_k 
  (k : ℝ)
  (p_eq : ∀ x : ℝ, (4 * x + 3 = k * x - 9) → (x = -3 → (k = 0)))
: k = 0 :=
by sorry

end find_k_l1831_183198


namespace curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l1831_183103

-- Definitions for the conditions
def curve_C_polar (ρ θ : ℝ) := ρ = 4 * Real.sin θ
def line_l_parametric (x y t : ℝ) := 
  x = (Real.sqrt 3 / 2) * t ∧ 
  y = 1 + (1 / 2) * t

-- Theorem statements
theorem curve_C_cartesian_eq : ∀ x y : ℝ,
  (∃ (ρ θ : ℝ), curve_C_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

theorem line_l_general_eq : ∀ x y t : ℝ,
  line_l_parametric x y t →
  x - (Real.sqrt 3) * y + Real.sqrt 3 = 0 :=
by sorry

theorem max_area_triangle_PAB : ∀ (P A B : ℝ × ℝ),
  (∃ (θ : ℝ), P = ⟨2 * Real.cos θ, 2 + 2 * Real.sin θ⟩ ∧
   (∃ t : ℝ, line_l_parametric A.1 A.2 t) ∧
   (∃ t' : ℝ, line_l_parametric B.1 B.2 t') ∧
   A ≠ B) →
  (1/2) * Real.sqrt 13 * (2 + Real.sqrt 3 / 2) = (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
by sorry

end curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l1831_183103


namespace trigonometric_identity_l1831_183185

variable (α : ℝ)

theorem trigonometric_identity :
  4.9 * (Real.sin (7 * Real.pi / 8 - 2 * α))^2 - (Real.sin (9 * Real.pi / 8 - 2 * α))^2 = 
  Real.sin (4 * α) / Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l1831_183185


namespace radius_increase_l1831_183158

theorem radius_increase (C1 C2 : ℝ) (π : ℝ) (hC1 : C1 = 40) (hC2 : C2 = 50) (hπ : π > 0) : 
  (C2 - C1) / (2 * π) = 5 / π := 
sorry

end radius_increase_l1831_183158


namespace second_team_pieces_l1831_183129

-- Definitions for the conditions
def total_pieces_required : ℕ := 500
def pieces_first_team : ℕ := 189
def pieces_third_team : ℕ := 180

-- The number of pieces the second team made
def pieces_second_team : ℕ := total_pieces_required - (pieces_first_team + pieces_third_team)

-- The theorem we are proving
theorem second_team_pieces : pieces_second_team = 131 := by
  unfold pieces_second_team
  norm_num
  sorry

end second_team_pieces_l1831_183129


namespace basketball_team_win_rate_l1831_183151

theorem basketball_team_win_rate (won_first : ℕ) (total : ℕ) (remaining : ℕ)
    (desired_rate : ℚ) (x : ℕ) (H_won : won_first = 30) (H_total : total = 100)
    (H_remaining : remaining = 55) (H_desired : desired_rate = 13/20) :
    (30 + x) / 100 = 13 / 20 ↔ x = 35 := by
    sorry

end basketball_team_win_rate_l1831_183151


namespace floor_sqrt_225_l1831_183139

theorem floor_sqrt_225 : Int.floor (Real.sqrt 225) = 15 := by
  sorry

end floor_sqrt_225_l1831_183139


namespace find_values_of_real_numbers_l1831_183189

theorem find_values_of_real_numbers (x y : ℝ)
  (h : 2 * x - 1 + (y + 1) * Complex.I = x - y - (x + y) * Complex.I) :
  x = 3 ∧ y = -2 :=
sorry

end find_values_of_real_numbers_l1831_183189


namespace intersection_M_N_l1831_183157

def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | x > 1} :=
by
  sorry

end intersection_M_N_l1831_183157


namespace average_speed_l1831_183173

theorem average_speed
  (distance1 : ℝ)
  (time1 : ℝ)
  (distance2 : ℝ)
  (time2 : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (average_speed : ℝ)
  (h1 : distance1 = 90)
  (h2 : time1 = 1)
  (h3 : distance2 = 50)
  (h4 : time2 = 1)
  (h5 : total_distance = distance1 + distance2)
  (h6 : total_time = time1 + time2)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 70 := 
sorry

end average_speed_l1831_183173


namespace min_abs_val_of_36_power_minus_5_power_l1831_183122

theorem min_abs_val_of_36_power_minus_5_power :
  ∃ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| = 11 := sorry

end min_abs_val_of_36_power_minus_5_power_l1831_183122


namespace length_of_BC_l1831_183138

theorem length_of_BC (b : ℝ) (h : b ^ 4 = 125) : 2 * b = 10 :=
sorry

end length_of_BC_l1831_183138


namespace num_three_digit_numbers_no_repeat_l1831_183156

theorem num_three_digit_numbers_no_repeat (digits : Finset ℕ) (h : digits = {1, 2, 3, 4}) :
  (digits.card = 4) →
  ∀ d1 d2 d3, d1 ∈ digits → d2 ∈ digits → d3 ∈ digits →
  d1 ≠ d2 → d1 ≠ d3 → d2 ≠ d3 → 
  3 * 2 * 1 * digits.card = 24 :=
by
  sorry

end num_three_digit_numbers_no_repeat_l1831_183156


namespace correct_transformation_l1831_183142

theorem correct_transformation (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
sorry

end correct_transformation_l1831_183142


namespace next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l1831_183192

-- Problem: Next number after 48 in the sequence
theorem next_number_after_48 (x : ℕ) (h₁ : x % 3 = 0) (h₂ : (x + 1) = 64) : x = 63 := sorry

-- Problem: Eighth number in the sequence
theorem eighth_number_in_sequence (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 8) : n = 168 := sorry

-- Problem: 2013th number in the sequence
theorem two_thousand_thirteenth_number (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 2013) : n = 9120399 := sorry

end next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l1831_183192


namespace Abby_in_seat_3_l1831_183176

variables (P : Type) [Inhabited P]
variables (Abby Bret Carl Dana : P)
variables (seat : P → ℕ)

-- Conditions from the problem:
-- Bret is actually sitting in seat #2.
axiom Bret_in_seat_2 : seat Bret = 2

-- False statement 1: Dana is next to Bret.
axiom false_statement_1 : ¬ (seat Dana = 1 ∨ seat Dana = 3)

-- False statement 2: Carl is sitting between Dana and Bret.
axiom false_statement_2 : ¬ (seat Carl = 1)

-- The final translated proof problem:
theorem Abby_in_seat_3 : seat Abby = 3 :=
sorry

end Abby_in_seat_3_l1831_183176


namespace cost_of_6_bottle_caps_l1831_183111

-- Define the cost of each bottle cap
def cost_per_bottle_cap : ℕ := 2

-- Define how many bottle caps we are buying
def number_of_bottle_caps : ℕ := 6

-- Define the total cost of the bottle caps
def total_cost : ℕ := 12

-- The proof statement to prove that the total cost is as expected
theorem cost_of_6_bottle_caps :
  cost_per_bottle_cap * number_of_bottle_caps = total_cost :=
by
  sorry

end cost_of_6_bottle_caps_l1831_183111


namespace age_of_new_person_l1831_183183

theorem age_of_new_person (avg_age : ℝ) (x : ℝ) 
  (h1 : 10 * avg_age - (10 * (avg_age - 3)) = 42 - x) : 
  x = 12 := 
by
  sorry

end age_of_new_person_l1831_183183


namespace find_positive_real_solutions_l1831_183137

theorem find_positive_real_solutions (x : ℝ) (h1 : 0 < x) 
(h2 : 3 / 5 * (2 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
    x = (40 + Real.sqrt 1636) / 2 ∨ x = (-20 + Real.sqrt 388) / 2 := by
  sorry

end find_positive_real_solutions_l1831_183137


namespace sin_four_alpha_l1831_183166

theorem sin_four_alpha (α : ℝ) (h1 : Real.sin (2 * α) = -4 / 5) (h2 : -Real.pi / 4 < α ∧ α < Real.pi / 4) :
  Real.sin (4 * α) = -24 / 25 :=
sorry

end sin_four_alpha_l1831_183166


namespace intersection_eq_T_l1831_183143

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l1831_183143


namespace goods_train_speed_l1831_183153

def speed_of_goods_train (length_in_meters : ℕ) (time_in_seconds : ℕ) (speed_of_man_train_kmph : ℕ) : ℕ :=
  let length_in_km := length_in_meters / 1000
  let time_in_hours := time_in_seconds / 3600
  let relative_speed_kmph := (length_in_km * 3600) / time_in_hours
  relative_speed_kmph - speed_of_man_train_kmph

theorem goods_train_speed :
  speed_of_goods_train 280 9 50 = 62 := by
  sorry

end goods_train_speed_l1831_183153


namespace find_CB_l1831_183106

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)

-- Given condition
-- D divides AB in the ratio 1:3 such that CA = a and CD = b

def D_divides_AB (A B D : V) : Prop := ∃ (k : ℝ), k = 1 / 4 ∧ A + k • (B - A) = D

theorem find_CB (CA CD : V) (A B D : V) (h1 : CA = A) (h2 : CD = B)
  (h3 : D_divides_AB A B D) : (B - A) = -3 • CA + 4 • CD :=
sorry

end find_CB_l1831_183106


namespace theater_tickets_l1831_183178

theorem theater_tickets (O B P : ℕ) (h1 : O + B + P = 550) 
  (h2 : 15 * O + 10 * B + 25 * P = 9750) (h3: P = 5 * O) (h4 : O ≥ 50) : 
  B - O = 179 :=
by
  sorry

end theater_tickets_l1831_183178


namespace total_candy_given_l1831_183124

def candy_given_total (a b c : ℕ) : ℕ := a + b + c

def first_10_friends_candy (n : ℕ) := 10 * n

def next_7_friends_candy (n : ℕ) := 7 * (2 * n)

def remaining_friends_candy := 50

theorem total_candy_given (n : ℕ) (h1 : first_10_friends_candy 12 = 120)
  (h2 : next_7_friends_candy 12 = 168) (h3 : remaining_friends_candy = 50) :
  candy_given_total 120 168 50 = 338 := by
  sorry

end total_candy_given_l1831_183124


namespace right_triangle_side_81_exists_arithmetic_progression_l1831_183136

theorem right_triangle_side_81_exists_arithmetic_progression :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a - d)^2 + a^2 = (a + d)^2 ∧ (3*d = 81 ∨ 4*d = 81 ∨ 5*d = 81) :=
sorry

end right_triangle_side_81_exists_arithmetic_progression_l1831_183136


namespace four_digit_numbers_count_l1831_183161

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l1831_183161


namespace b_profit_l1831_183117

noncomputable def profit_share (x t : ℝ) : ℝ :=
  let total_profit := 31500
  let a_investment := 3 * x
  let a_period := 2 * t
  let b_investment := x
  let b_period := t
  let profit_ratio_a := a_investment * a_period
  let profit_ratio_b := b_investment * b_period
  let total_ratio := profit_ratio_a + profit_ratio_b
  let b_share := profit_ratio_b / total_ratio
  b_share * total_profit

theorem b_profit (x t : ℝ) : profit_share x t = 4500 :=
by
  sorry

end b_profit_l1831_183117


namespace batman_game_cost_l1831_183108

theorem batman_game_cost (total_spent superman_cost : ℝ) 
  (H1 : total_spent = 18.66) (H2 : superman_cost = 5.06) :
  total_spent - superman_cost = 13.60 :=
by
  sorry

end batman_game_cost_l1831_183108


namespace solve_for_y_l1831_183102

theorem solve_for_y 
  (x y : ℝ) 
  (h1 : 2 * x - 3 * y = 9) 
  (h2 : x + y = 8) : 
  y = 1.4 := 
sorry

end solve_for_y_l1831_183102


namespace fraction_students_say_dislike_actually_like_l1831_183107

theorem fraction_students_say_dislike_actually_like (total_students : ℕ) (like_dancing_fraction : ℚ) 
  (like_dancing_say_dislike_fraction : ℚ) (dislike_dancing_say_dislike_fraction : ℚ) : 
  (∃ frac : ℚ, frac = 40.7 / 100) :=
by
  let total_students := (200 : ℕ)
  let like_dancing_fraction := (70 / 100 : ℚ)
  let like_dancing_say_dislike_fraction := (25 / 100 : ℚ)
  let dislike_dancing_say_dislike_fraction := (85 / 100 : ℚ)
  
  let total_like_dancing := total_students * like_dancing_fraction
  let total_dislike_dancing :=  total_students * (1 - like_dancing_fraction)
  let like_dancing_say_dislike := total_like_dancing * like_dancing_say_dislike_fraction
  let dislike_dancing_say_dislike := total_dislike_dancing * dislike_dancing_say_dislike_fraction
  let total_say_dislike := like_dancing_say_dislike + dislike_dancing_say_dislike
  let fraction_say_dislike_actually_like := like_dancing_say_dislike / total_say_dislike
  
  existsi fraction_say_dislike_actually_like
  sorry

end fraction_students_say_dislike_actually_like_l1831_183107


namespace convex_polygon_with_arith_prog_angles_l1831_183100

theorem convex_polygon_with_arith_prog_angles 
  (n : ℕ) 
  (angles : Fin n → ℝ)
  (is_convex : ∀ i, angles i < 180)
  (arithmetic_progression : ∃ a d, d = 3 ∧ ∀ i, angles i = a + i * d)
  (largest_angle : ∃ i, angles i = 150)
  : n = 24 :=
sorry

end convex_polygon_with_arith_prog_angles_l1831_183100


namespace rachel_pizza_eaten_l1831_183186

theorem rachel_pizza_eaten (pizza_total : ℕ) (pizza_bella : ℕ) (pizza_rachel : ℕ) :
  pizza_total = pizza_bella + pizza_rachel → pizza_bella = 354 → pizza_total = 952 → pizza_rachel = 598 :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  sorry

end rachel_pizza_eaten_l1831_183186


namespace union_condition_implies_l1831_183162

-- Define set A as per the given condition
def setA : Set ℝ := { x | x * (x - 1) ≤ 0 }

-- Define set B as per the given condition with parameter a
def setB (a : ℝ) : Set ℝ := { x | Real.log x ≤ a }

-- Given condition A ∪ B = A, we need to prove that a ≤ 0
theorem union_condition_implies (a : ℝ) (h : setA ∪ setB a = setA) : a ≤ 0 := 
by
  sorry

end union_condition_implies_l1831_183162


namespace find_f_x_l1831_183190

def tan : ℝ → ℝ := sorry  -- tan function placeholder
def cos : ℝ → ℝ := sorry  -- cos function placeholder
def sin : ℝ → ℝ := sorry  -- sin function placeholder

axiom conditions : 
  tan 45 = 1 ∧
  cos 60 = 2 ∧
  sin 90 = 3 ∧
  cos 180 = 4 ∧
  sin 270 = 5

theorem find_f_x :
  ∃ f x, (f x = 6) ∧ 
  (f = tan ∧ x = 360) := 
sorry

end find_f_x_l1831_183190


namespace equilibrium_possible_l1831_183148

theorem equilibrium_possible (n : ℕ) : (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 4 * k + 3 = n) ↔
  (∃ S1 S2 : Finset ℕ, S1 ∪ S2 = Finset.range (n+1) ∧
                     S1 ∩ S2 = ∅ ∧
                     S1.sum id = S2.sum id) := 
sorry

end equilibrium_possible_l1831_183148


namespace parabola_triangle_areas_l1831_183149

-- Define necessary points and expressions
variables (x1 y1 x2 y2 x3 y3 : ℝ)
variables (m n : ℝ)
def parabola_eq (x y : ℝ) := y ^ 2 = 4 * x
def median_line (m n x y : ℝ) := m * x + n * y - m = 0
def areas_sum_sq (S1 S2 S3 : ℝ) := S1 ^ 2 + S2 ^ 2 + S3 ^ 2 = 3

-- Main statement
theorem parabola_triangle_areas :
  (parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ parabola_eq x3 y3) →
  (m ≠ 0) →
  (median_line m n 1 0) →
  (x1 + x2 + x3 = 3) →
  ∃ S1 S2 S3 : ℝ, areas_sum_sq S1 S2 S3 :=
by sorry

end parabola_triangle_areas_l1831_183149


namespace set_union_complement_l1831_183170

-- Definitions based on provided problem statement
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}
def CRQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- The theorem to prove
theorem set_union_complement : P ∪ CRQ = {x | -2 < x ∧ x ≤ 3} :=
by
  -- Skip the proof
  sorry

end set_union_complement_l1831_183170


namespace product_probability_correct_l1831_183154

/-- Define probabilities for spins of Paco and Dani --/
def prob_paco := 1 / 5
def prob_dani := 1 / 15

/-- Define the probability that the product of spins is less than 30 --/
def prob_product_less_than_30 : ℚ :=
  (2 / 5) + (1 / 5) * (9 / 15) + (1 / 5) * (7 / 15) + (1 / 5) * (5 / 15)

theorem product_probability_correct : prob_product_less_than_30 = 17 / 25 :=
by sorry

end product_probability_correct_l1831_183154


namespace total_earthworms_in_box_l1831_183164

-- Definitions of the conditions
def applesPaidByOkeydokey := 5
def applesPaidByArtichokey := 7
def earthwormsReceivedByOkeydokey := 25
def ratio := earthwormsReceivedByOkeydokey / applesPaidByOkeydokey -- which should be 5

-- Theorem statement proving the total number of earthworms in the box
theorem total_earthworms_in_box :
  (applesPaidByOkeydokey + applesPaidByArtichokey) * ratio = 60 :=
by
  sorry

end total_earthworms_in_box_l1831_183164


namespace sum_of_numbers_l1831_183160

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a - 2)^2 - (b - 2)^2 = 18): 
  a + b = -2 := 
by 
  sorry

end sum_of_numbers_l1831_183160


namespace baseball_weight_l1831_183159

theorem baseball_weight
  (weight_total : ℝ)
  (weight_soccer_ball : ℝ)
  (n_soccer_balls : ℕ)
  (n_baseballs : ℕ)
  (total_weight : ℝ)
  (B : ℝ) :
  n_soccer_balls * weight_soccer_ball + n_baseballs * B = total_weight →
  n_soccer_balls = 9 →
  weight_soccer_ball = 0.8 →
  n_baseballs = 7 →
  total_weight = 10.98 →
  B = 0.54 := sorry

end baseball_weight_l1831_183159


namespace MN_intersection_correct_l1831_183131

-- Define the sets M and N
def setM : Set ℝ := {y | ∃ x ∈ (Set.univ : Set ℝ), y = x^2 + 2*x - 3}
def setN : Set ℝ := {x | |x - 2| ≤ 3}

-- Reformulated sets
def setM_reformulated : Set ℝ := {y | y ≥ -4}
def setN_reformulated : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The intersection set
def MN_intersection : Set ℝ := {y | -1 ≤ y ∧ y ≤ 5}

-- The theorem stating the intersection of M and N equals MN_intersection
theorem MN_intersection_correct :
  {y | ∃ x ∈ setN_reformulated, y = x^2 + 2*x - 3} = MN_intersection :=
sorry  -- Proof not required as per instruction

end MN_intersection_correct_l1831_183131


namespace average_speed_round_trip_l1831_183105

theorem average_speed_round_trip :
  ∀ (D : ℝ), 
  D > 0 → 
  let upstream_speed := 6 
  let downstream_speed := 5 
  (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 :=
by
  intro D hD
  let upstream_speed := 6
  let downstream_speed := 5
  have h : (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 := sorry
  exact h

end average_speed_round_trip_l1831_183105


namespace find_smallest_n_l1831_183144

theorem find_smallest_n 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (h1 : a_n 1 + a_n 2 = 9 / 2)
    (h2 : S_n 4 = 45 / 8)
    (h3 : ∀ n, S_n n = (1 / 2) * n * (a_n 1 + a_n n)) :
    ∃ n : ℕ, a_n n < 1 / 10 ∧ ∀ m : ℕ, m < n → a_n m ≥ 1 / 10 := 
sorry

end find_smallest_n_l1831_183144


namespace max_boxes_fit_l1831_183172

theorem max_boxes_fit 
  (L_large W_large H_large : ℕ) 
  (L_small W_small H_small : ℕ) 
  (h1 : L_large = 12) 
  (h2 : W_large = 14) 
  (h3 : H_large = 16) 
  (h4 : L_small = 3) 
  (h5 : W_small = 7) 
  (h6 : H_small = 2) 
  : ((L_large * W_large * H_large) / (L_small * W_small * H_small) = 64) :=
by
  sorry

end max_boxes_fit_l1831_183172


namespace train_speed_kph_l1831_183133

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l1831_183133


namespace power_of_two_plus_one_div_by_power_of_three_l1831_183130

theorem power_of_two_plus_one_div_by_power_of_three (n : ℕ) : 3^(n + 1) ∣ (2^(3^n) + 1) :=
sorry

end power_of_two_plus_one_div_by_power_of_three_l1831_183130


namespace sequence_1001st_term_l1831_183191

theorem sequence_1001st_term (a b : ℤ) (h1 : b = 2 * a - 3) : 
  ∃ n : ℤ, n = 1001 → (a + 1000 * (20 * a - 30)) = 30003 := 
by 
  sorry

end sequence_1001st_term_l1831_183191


namespace hyperbola_constants_l1831_183114

theorem hyperbola_constants (h k a c b : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 2 ∧ c = 5 ∧ b = Real.sqrt 21 → 
  h + k + a + b = 0 + Real.sqrt 21 :=
by
  intro hka
  sorry

end hyperbola_constants_l1831_183114


namespace solvability_condition_l1831_183180

def is_solvable (p : ℕ) [Fact (Nat.Prime p)] :=
  ∃ α : ℤ, α * (α - 1) + 3 ≡ 0 [ZMOD p] ↔ ∃ β : ℤ, β * (β - 1) + 25 ≡ 0 [ZMOD p]

theorem solvability_condition (p : ℕ) [Fact (Nat.Prime p)] : 
  is_solvable p :=
sorry

end solvability_condition_l1831_183180


namespace mabel_tomatoes_l1831_183116

theorem mabel_tomatoes (x : ℕ)
  (plant_1_bore : ℕ)
  (plant_2_bore : ℕ := x + 4)
  (total_first_two_plants : ℕ := x + plant_2_bore)
  (plant_3_bore : ℕ := 3 * total_first_two_plants)
  (plant_4_bore : ℕ := 3 * total_first_two_plants)
  (total_tomatoes : ℕ)
  (h1 : total_first_two_plants = 2 * x + 4)
  (h2 : plant_3_bore = 3 * (2 * x + 4))
  (h3 : plant_4_bore = 3 * (2 * x + 4))
  (h4 : total_tomatoes = x + plant_2_bore + plant_3_bore + plant_4_bore)
  (h5 : total_tomatoes = 140) :
   x = 8 :=
by
  sorry

end mabel_tomatoes_l1831_183116


namespace carl_personal_owe_l1831_183152

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l1831_183152


namespace range_of_a_l1831_183175

def proposition_P (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_Q (a : ℝ) := 5 - 2*a > 1

theorem range_of_a :
  (∃! (p : Prop), (p = proposition_P a ∨ p = proposition_Q a) ∧ p) →
  a ∈ Set.Iic (-2) :=
by
  sorry

end range_of_a_l1831_183175


namespace more_students_than_rabbits_l1831_183168

theorem more_students_than_rabbits :
  let number_of_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let total_students := students_per_classroom * number_of_classrooms
  let total_rabbits := rabbits_per_classroom * number_of_classrooms
  total_students - total_rabbits = 95 := by
  sorry

end more_students_than_rabbits_l1831_183168


namespace cost_of_iced_coffee_for_2_weeks_l1831_183167

def cost_to_last_for_2_weeks (servings_per_bottle servings_per_day price_per_bottle duration_in_days : ℕ) : ℕ :=
  let total_servings_needed := servings_per_day * duration_in_days
  let bottles_needed := total_servings_needed / servings_per_bottle
  bottles_needed * price_per_bottle

theorem cost_of_iced_coffee_for_2_weeks :
  cost_to_last_for_2_weeks 6 3 3 14 = 21 :=
by
  sorry

end cost_of_iced_coffee_for_2_weeks_l1831_183167


namespace solve_quadratic_inequality_l1831_183174

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end solve_quadratic_inequality_l1831_183174


namespace problem_a_b_c_d_l1831_183169

open Real

/-- The main theorem to be proved -/
theorem problem_a_b_c_d
  (a b c d : ℝ)
  (hab : 0 < a) (hcd : 0 < c) (hab' : 0 < b) (hcd' : 0 < d)
  (h1 : a > c) (h2 : b < d)
  (h3 : a + sqrt b ≥ c + sqrt d)
  (h4 : sqrt a + b ≤ sqrt c + d) :
  a + b + c + d > 1 :=
by
  sorry

end problem_a_b_c_d_l1831_183169


namespace num_outfits_l1831_183101

def num_shirts := 6
def num_ties := 4
def num_pants := 3
def outfits : ℕ := num_shirts * num_pants * (num_ties + 1)

theorem num_outfits: outfits = 90 :=
by 
  -- sorry will be removed when proof is provided
  sorry

end num_outfits_l1831_183101


namespace comprehensive_survey_l1831_183128

def suitable_for_census (s: String) : Prop := 
  s = "Surveying the heights of all classmates in the class"

theorem comprehensive_survey : suitable_for_census "Surveying the heights of all classmates in the class" :=
by
  sorry

end comprehensive_survey_l1831_183128


namespace quadratic_eq_with_roots_l1831_183140

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l1831_183140


namespace line_equation_under_transformation_l1831_183132

noncomputable def T1_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

noncomputable def T2_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 3]
]

noncomputable def NM_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -2],
  ![3, 0]
]

theorem line_equation_under_transformation :
  ∀ x y : ℝ, (∃ x' y' : ℝ, NM_matrix.mulVec ![x, y] = ![x', y'] ∧ x' = y') → 3 * x + 2 * y = 0 :=
by sorry

end line_equation_under_transformation_l1831_183132


namespace calculate_expression_l1831_183194

theorem calculate_expression : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end calculate_expression_l1831_183194


namespace second_number_is_915_l1831_183155

theorem second_number_is_915 :
  ∃ (n1 n2 n3 n4 n5 n6 : ℤ), 
    n1 = 3 ∧ 
    n2 = 915 ∧ 
    n3 = 138 ∧ 
    n4 = 1917 ∧ 
    n5 = 2114 ∧ 
    ∃ x: ℤ, 
      (n1 + n2 + n3 + n4 + n5 + x) / 6 = 12 ∧ 
      n2 = 915 :=
by 
  sorry

end second_number_is_915_l1831_183155


namespace sanctuary_feeding_ways_l1831_183141

/-- A sanctuary houses six different pairs of animals, each pair consisting of a male and female.
  The caretaker must feed the animals alternately by gender, meaning no two animals of the same gender 
  can be fed consecutively. Given the additional constraint that the male giraffe cannot be fed 
  immediately before the female giraffe and that the feeding starts with the male lion, 
  there are exactly 7200 valid ways to complete the feeding. -/
theorem sanctuary_feeding_ways : 
  ∃ ways : ℕ, ways = 7200 :=
by sorry

end sanctuary_feeding_ways_l1831_183141


namespace count_sums_of_two_cubes_lt_400_l1831_183113

theorem count_sums_of_two_cubes_lt_400 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, ∃ a b, 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 7 ∧ n = a^3 + b^3 ∧ (Odd a ∨ Odd b) ∧ n < 400) ∧
    s.card = 15 :=
by 
  sorry

end count_sums_of_two_cubes_lt_400_l1831_183113


namespace circle_diameter_l1831_183182

open Real

theorem circle_diameter (r_D : ℝ) (r_C : ℝ) (h_D : r_D = 10) (h_ratio: (π * (r_D ^ 2 - r_C ^ 2)) / (π * r_C ^ 2) = 4) : 2 * r_C = 4 * sqrt 5 :=
by sorry

end circle_diameter_l1831_183182


namespace each_person_towel_day_l1831_183135

def total_people (families : ℕ) (members_per_family : ℕ) : ℕ :=
  families * members_per_family

def total_towels (loads : ℕ) (towels_per_load : ℕ) : ℕ :=
  loads * towels_per_load

def towels_per_day (total_towels : ℕ) (days : ℕ) : ℕ :=
  total_towels / days

def towels_per_person_per_day (towels_per_day : ℕ) (total_people : ℕ) : ℕ :=
  towels_per_day / total_people

theorem each_person_towel_day
  (families : ℕ) (members_per_family : ℕ) (days : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h_family : families = 3) (h_members : members_per_family = 4) (h_days : days = 7)
  (h_loads : loads = 6) (h_towels_per_load : towels_per_load = 14) :
  towels_per_person_per_day (towels_per_day (total_towels loads towels_per_load) days) (total_people families members_per_family) = 1 :=
by {
  -- Import necessary assumptions
  sorry
}

end each_person_towel_day_l1831_183135


namespace rounds_on_sunday_l1831_183104

theorem rounds_on_sunday (round_time total_time saturday_rounds : ℕ) (h1 : round_time = 30)
(h2 : total_time = 780) (h3 : saturday_rounds = 11) : 
(total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry

end rounds_on_sunday_l1831_183104


namespace total_students_correct_l1831_183197

-- Define the number of students who play football, cricket, both and neither.
def play_football : ℕ := 325
def play_cricket : ℕ := 175
def play_both : ℕ := 90
def play_neither : ℕ := 50

-- Define the total number of students
def total_students : ℕ := play_football + play_cricket - play_both + play_neither

-- Prove that the total number of students is 460 given the conditions
theorem total_students_correct : total_students = 460 := by
  sorry

end total_students_correct_l1831_183197


namespace sum_of_possible_values_of_z_l1831_183179

theorem sum_of_possible_values_of_z (x y z : ℂ) 
  (h₁ : z^2 + 5 * x = 10 * z)
  (h₂ : y^2 + 5 * z = 10 * y)
  (h₃ : x^2 + 5 * y = 10 * x) :
  z = 0 ∨ z = 9 / 5 := by
  sorry

end sum_of_possible_values_of_z_l1831_183179


namespace percentage_increase_l1831_183134

noncomputable def percentMoreThan (a b : ℕ) : ℕ :=
  ((a - b) * 100) / b

theorem percentage_increase (x y z : ℕ) (h1 : z = 300) (h2 : x = 5 * y / 4) (h3 : x + y + z = 1110) :
  percentMoreThan y z = 20 := by
  sorry

end percentage_increase_l1831_183134


namespace trees_in_yard_l1831_183165

theorem trees_in_yard (L d : ℕ) (hL : L = 250) (hd : d = 5) : 
  (L / d + 1) = 51 := by
  sorry

end trees_in_yard_l1831_183165


namespace probability_of_rain_l1831_183171

-- Define the conditions in Lean
variables (x : ℝ) -- probability of rain

-- Known condition: taking an umbrella 20% of the time
def takes_umbrella : Prop := 0.2 = x + ((1 - x) * x)

-- The desired problem statement
theorem probability_of_rain : takes_umbrella x → x = 1 / 9 :=
by
  -- placeholder for the proof
  intro h
  sorry

end probability_of_rain_l1831_183171


namespace perimeter_gt_sixteen_l1831_183120

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l1831_183120


namespace exists_good_set_l1831_183126

variable (M : Set ℕ) [DecidableEq M] [Fintype M]
variable (f : Finset ℕ → ℕ)

theorem exists_good_set :
  ∃ T : Finset ℕ, T.card = 10 ∧ (∀ k ∈ T, f (T.erase k) ≠ k) := by
  sorry

end exists_good_set_l1831_183126


namespace prove_ordered_pair_l1831_183188

-- Definition of the problem
def satisfies_equation1 (x y : ℚ) : Prop :=
  3 * x - 4 * y = -7

def satisfies_equation2 (x y : ℚ) : Prop :=
  7 * x - 3 * y = 5

-- Definition of the correct answer
def correct_answer (x y : ℚ) : Prop :=
  x = -133 / 57 ∧ y = 64 / 19

-- Main theorem to prove
theorem prove_ordered_pair :
  correct_answer (-133 / 57) (64 / 19) :=
by
  unfold correct_answer
  constructor
  { sorry }
  { sorry }

end prove_ordered_pair_l1831_183188


namespace tony_rollercoasters_l1831_183150

theorem tony_rollercoasters :
  let s1 := 50 -- speed of the first rollercoaster
  let s2 := 62 -- speed of the second rollercoaster
  let s3 := 73 -- speed of the third rollercoaster
  let s4 := 70 -- speed of the fourth rollercoaster
  let s5 := 40 -- speed of the fifth rollercoaster
  let avg_speed := 59 -- Tony's average speed during the day
  let total_speed := s1 + s2 + s3 + s4 + s5
  total_speed / avg_speed = 5 := sorry

end tony_rollercoasters_l1831_183150


namespace sum_of_roots_l1831_183163

-- sum of roots of first polynomial
def S1 : ℚ := -(-6 / 3)

-- sum of roots of second polynomial
def S2 : ℚ := -(8 / 4)

-- proof statement
theorem sum_of_roots : S1 + S2 = 0 :=
by
  -- placeholders
  sorry

end sum_of_roots_l1831_183163


namespace person_is_not_sane_l1831_183127

-- Definitions
def Person : Type := sorry
def sane : Person → Prop := sorry
def human : Person → Prop := sorry
def vampire : Person → Prop := sorry
def declares (p : Person) (s : String) : Prop := sorry

-- Conditions
axiom transylvanian_declares_vampire (p : Person) : declares p "I am a vampire"
axiom sane_human_never_claims_vampire (p : Person) : sane p → human p → ¬ declares p "I am a vampire"
axiom sane_vampire_never_admits_vampire (p : Person) : sane p → vampire p → ¬ declares p "I am a vampire"
axiom insane_human_might_claim_vampire (p : Person) : ¬ sane p → human p → declares p "I am a vampire"
axiom insane_vampire_might_admit_vampire (p : Person) : ¬ sane p → vampire p → declares p "I am a vampire"

-- Proof statement
theorem person_is_not_sane (p : Person) : declares p "I am a vampire" → ¬ sane p :=
by
  intros h
  sorry

end person_is_not_sane_l1831_183127


namespace find_tangent_c_l1831_183110

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → (c = 1) :=
by
  intros h
  sorry

end find_tangent_c_l1831_183110


namespace pow_div_l1831_183196

theorem pow_div (a : ℝ) : (-a) ^ 6 / a ^ 3 = a ^ 3 := by
  sorry

end pow_div_l1831_183196


namespace horizontal_asymptote_exists_x_intercepts_are_roots_l1831_183115

noncomputable def given_function (x : ℝ) : ℝ :=
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_exists :
  ∃ L : ℝ, ∀ x : ℝ, (∃ M : ℝ, M > 0 ∧ (∀ x > M, abs (given_function x - L) < 1)) ∧ L = 0 := 
sorry

theorem x_intercepts_are_roots :
  ∀ y, y = 0 ↔ ∃ x : ℝ, x ≠ 0 ∧ 15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5 = 0 :=
sorry

end horizontal_asymptote_exists_x_intercepts_are_roots_l1831_183115


namespace parallelogram_angle_B_l1831_183177

theorem parallelogram_angle_B (A C B D : ℝ) (h₁ : A + C = 110) (h₂ : A = C) : B = 125 :=
by sorry

end parallelogram_angle_B_l1831_183177
