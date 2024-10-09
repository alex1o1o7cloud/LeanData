import Mathlib

namespace abs_value_solution_l448_44875

theorem abs_value_solution (a : ℝ) : |-a| = |-5.333| → (a = 5.333 ∨ a = -5.333) :=
by
  sorry

end abs_value_solution_l448_44875


namespace trivia_team_total_score_l448_44894

theorem trivia_team_total_score 
  (scores : List ℕ)
  (present_members : List ℕ)
  (H_score : scores = [4, 6, 2, 8, 3, 5, 10, 3, 7])
  (H_present : present_members = scores) :
  List.sum present_members = 48 := 
by
  sorry

end trivia_team_total_score_l448_44894


namespace next_working_day_together_l448_44862

theorem next_working_day_together : 
  let greta_days := 5
  let henry_days := 3
  let linda_days := 9
  let sam_days := 8
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm (Nat.lcm greta_days henry_days) linda_days) sam_days ∧ n = 360 :=
by
  sorry

end next_working_day_together_l448_44862


namespace number_of_marks_for_passing_l448_44842

theorem number_of_marks_for_passing (T P : ℝ) 
  (h1 : 0.40 * T = P - 40) 
  (h2 : 0.60 * T = P + 20) 
  (h3 : 0.45 * T = P - 10) :
  P = 160 :=
by
  sorry

end number_of_marks_for_passing_l448_44842


namespace left_square_side_length_l448_44818

theorem left_square_side_length (x : ℕ) (h1 : x + (x + 17) + (x + 11) = 52) : x = 8 :=
sorry

end left_square_side_length_l448_44818


namespace speed_of_stream_l448_44841

theorem speed_of_stream (downstream_speed upstream_speed : ℕ) (h1 : downstream_speed = 12) (h2 : upstream_speed = 8) : 
  (downstream_speed - upstream_speed) / 2 = 2 :=
by
  sorry

end speed_of_stream_l448_44841


namespace expected_profit_calculation_l448_44830

theorem expected_profit_calculation:
  let odd1 := 1.28
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let initial_bet := 5.00
  let total_payout := initial_bet * (odd1 * odd2 * odd3 * odd4)
  let expected_profit := total_payout - initial_bet
  expected_profit = 212.822 := by
  sorry

end expected_profit_calculation_l448_44830


namespace sum_distances_l448_44826

noncomputable def lengthAB : ℝ := 2
noncomputable def lengthA'B' : ℝ := 5
noncomputable def midpointAB : ℝ := lengthAB / 2
noncomputable def midpointA'B' : ℝ := lengthA'B' / 2
noncomputable def distancePtoD : ℝ := 0.5
noncomputable def proportionality_constant : ℝ := lengthA'B' / lengthAB

theorem sum_distances : distancePtoD + (proportionality_constant * distancePtoD) = 1.75 := by
  sorry

end sum_distances_l448_44826


namespace problem1_problem2_l448_44864

theorem problem1 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := sorry

theorem problem2 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (A + C) / (2 * B + A) = 9 / 5 := sorry

end problem1_problem2_l448_44864


namespace find_number_l448_44883

theorem find_number (x : ℝ) (h : 45 - 3 * x = 12) : x = 11 :=
sorry

end find_number_l448_44883


namespace gcd_6Tn_nplus1_l448_44897

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_l448_44897


namespace geometric_sequence_a7_eq_64_l448_44882

open Nat

theorem geometric_sequence_a7_eq_64 (a : ℕ → ℕ) (h1 : a 1 = 1) (hrec : ∀ n : ℕ, a (n + 1) = 2 * a n) : a 7 = 64 := by
  sorry

end geometric_sequence_a7_eq_64_l448_44882


namespace least_number_to_subtract_l448_44803

theorem least_number_to_subtract (x : ℕ) (h : 509 - x = 45 * n) : ∃ x, (509 - x) % 9 = 0 ∧ (509 - x) % 15 = 0 ∧ x = 14 := by
  sorry

end least_number_to_subtract_l448_44803


namespace max_points_of_intersection_l448_44898

-- Define the lines and their properties
variable (L : Fin 150 → Prop)

-- Condition: L_5n are parallel to each other
def parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k

-- Condition: L_{5n-1} pass through a given point B
def passing_through_B (n : ℕ) :=
  ∃ k, n = 5 * k + 1

-- Condition: L_{5n-2} are parallel to another line not parallel to those in parallel_group
def other_parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k + 3

-- Total number of points of intersection of pairs of lines from the complete set
theorem max_points_of_intersection (L : Fin 150 → Prop)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_parallel_group : ∀ i j : Fin 150, parallel_group i → parallel_group j → L i = L j)
  (h_through_B : ∀ i j : Fin 150, passing_through_B i → passing_through_B j → L i = L j)
  (h_other_parallel_group : ∀ i j : Fin 150, other_parallel_group i → other_parallel_group j → L i = L j)
  : ∃ P, P = 8071 := 
sorry

end max_points_of_intersection_l448_44898


namespace number_of_perpendicular_points_on_ellipse_l448_44877

theorem number_of_perpendicular_points_on_ellipse :
  ∃ (P : ℝ × ℝ), (P ∈ {P : ℝ × ℝ | (P.1^2 / 8) + (P.2^2 / 4) = 1})
  ∧ (∀ (F1 F2 : ℝ × ℝ), F1 ≠ F2 → ∀ (P : ℝ × ℝ), ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) = 0) :=
sorry

end number_of_perpendicular_points_on_ellipse_l448_44877


namespace squirrel_acorns_initial_stash_l448_44808

theorem squirrel_acorns_initial_stash (A : ℕ) 
  (h1 : 3 * (A / 3 - 60) = 30) : A = 210 := 
sorry

end squirrel_acorns_initial_stash_l448_44808


namespace area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l448_44831

-- Defining the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point P of line1 and line2
def P : ℝ × ℝ := (-2, 2)

-- Perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Line l, passing through P and perpendicular to perpendicular_line
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intercepts of line_l with axes
def x_intercept : ℝ := -1
def y_intercept : ℝ := -2

-- Verifying area of the triangle formed by the intercepts
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

#check line1
#check line2
#check P
#check perpendicular_line
#check line_l
#check x_intercept
#check y_intercept
#check area_of_triangle

theorem area_of_triangle_formed_by_line_l_and_axes :
  ∀ (x : ℝ) (y : ℝ),
    line_l x 0 → line_l 0 y →
    area_of_triangle (abs x) (abs y) = 1 :=
by
  intros x y hx hy
  sorry

theorem equation_of_line_l :
  ∀ (x y : ℝ),
    (line1 x y ∧ line2 x y) →
    (perpendicular_line x y) →
    line_l x y :=
by
  intros x y h1 h2
  sorry

end area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l448_44831


namespace range_of_a_l448_44833

noncomputable def f (a x : ℝ) : ℝ :=
if h : a ≤ x ∧ x < 0 then -((1/2)^x)
else if h' : 0 ≤ x ∧ x ≤ 4 then -(x^2) + 2*x
else 0

theorem range_of_a (a : ℝ) (h : ∀ x, f a x ∈ Set.Icc (-8 : ℝ) (1 : ℝ)) : 
  a ∈ Set.Ico (-3 : ℝ) 0 :=
sorry

end range_of_a_l448_44833


namespace equation_of_line_through_A_parallel_to_given_line_l448_44891

theorem equation_of_line_through_A_parallel_to_given_line :
  ∃ c : ℝ, 
    (∀ x y : ℝ, 2 * x - y + c = 0 ↔ ∃ a b : ℝ, a = -1 ∧ b = 0 ∧ 2 * a - b + 1 = 0) :=
sorry

end equation_of_line_through_A_parallel_to_given_line_l448_44891


namespace bus_driver_total_earnings_l448_44802

noncomputable def regular_rate : ℝ := 20
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours : ℝ := 45.714285714285715
noncomputable def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
noncomputable def overtime_hours : ℝ := total_hours - regular_hours
noncomputable def regular_pay : ℝ := regular_rate * regular_hours
noncomputable def overtime_pay : ℝ := overtime_rate * overtime_hours
noncomputable def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_total_earnings :
  total_compensation = 1000 :=
by
  sorry

end bus_driver_total_earnings_l448_44802


namespace find_interest_rate_l448_44886

-- Defining the conditions
def P : ℝ := 5000
def A : ℝ := 5302.98
def t : ℝ := 1.5
def n : ℕ := 2

-- Statement of the problem in Lean 4
theorem find_interest_rate (P A t : ℝ) (n : ℕ) (hP : P = 5000) (hA : A = 5302.98) (ht : t = 1.5) (hn : n = 2) : 
  ∃ r : ℝ, r * 100 = 3.96 :=
sorry

end find_interest_rate_l448_44886


namespace range_of_m_l448_44870

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + 4 = 0) → x > 1) ↔ (2 ≤ m ∧ m < 5/2) := sorry

end range_of_m_l448_44870


namespace problem_l448_44885

open Real

noncomputable def f (ω a x : ℝ) := (1 / 2) * (sin (ω * x) + a * cos (ω * x))

theorem problem (a : ℝ) 
  (hω_range : 0 < ω ∧ ω ≤ 1)
  (h_f_sym1 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h_f_sym2 : ∀ x, f ω a (x - π) = f ω a (x + π))
  (x1 x2 : ℝ) 
  (h_x_in_interval1 : -π/3 < x1 ∧ x1 < 5*π/3)
  (h_x_in_interval2 : -π/3 < x2 ∧ x2 < 5*π/3)
  (h_distinct : x1 ≠ x2)
  (h_f_neg_half1 : f ω a x1 = -1/2)
  (h_f_neg_half2 : f ω a x2 = -1/2) :
  (f 1 (sqrt 3) x = sin (x + π/3)) ∧ (x1 + x2 = 7*π/3) :=
by
  sorry

end problem_l448_44885


namespace distance_travelled_l448_44816

theorem distance_travelled
  (d : ℝ)                   -- distance in kilometers
  (train_speed : ℝ)         -- train speed in km/h
  (ship_speed : ℝ)          -- ship speed in km/h
  (time_difference : ℝ)     -- time difference in hours
  (h1 : train_speed = 48)
  (h2 : ship_speed = 60)
  (h3 : time_difference = 2) :
  d = 480 := 
by
  sorry

end distance_travelled_l448_44816


namespace Nikolai_faster_than_Gennady_l448_44857

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end Nikolai_faster_than_Gennady_l448_44857


namespace smallest_x_satisfies_equation_l448_44858

theorem smallest_x_satisfies_equation : 
  ∀ x : ℚ, 7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45) → x = -7 / 5 :=
by {
  sorry
}

end smallest_x_satisfies_equation_l448_44858


namespace find_difference_of_segments_l448_44819

theorem find_difference_of_segments 
  (a b c d x y : ℝ)
  (h1 : a + b = 70)
  (h2 : b + c = 90)
  (h3 : c + d = 130)
  (h4 : a + d = 110)
  (hx_y_sum : x + y = 130)
  (hx_c : x = c)
  (hy_d : y = d) : 
  |x - y| = 13 :=
sorry

end find_difference_of_segments_l448_44819


namespace shuttlecock_weight_probability_l448_44869

variable (p_lt_4_8 : ℝ) -- Probability that its weight is less than 4.8 g
variable (p_le_4_85 : ℝ) -- Probability that its weight is not greater than 4.85 g

theorem shuttlecock_weight_probability (h1 : p_lt_4_8 = 0.3) (h2 : p_le_4_85 = 0.32) :
  p_le_4_85 - p_lt_4_8 = 0.02 :=
by
  sorry

end shuttlecock_weight_probability_l448_44869


namespace johns_number_is_1500_l448_44881

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end johns_number_is_1500_l448_44881


namespace length_of_first_platform_l448_44800

noncomputable def speed (distance time : ℕ) :=
  distance / time

theorem length_of_first_platform 
  (L : ℕ) (train_length : ℕ) (time1 time2 : ℕ) (platform2_length : ℕ) (speed : ℕ) 
  (H1 : L + train_length = speed * time1) 
  (H2 : platform2_length + train_length = speed * time2) 
  (train_length_eq : train_length = 30) 
  (time1_eq : time1 = 12) 
  (time2_eq : time2 = 15) 
  (platform2_length_eq : platform2_length = 120) 
  (speed_eq : speed = 10) : L = 90 :=
by
  sorry

end length_of_first_platform_l448_44800


namespace spinner_win_sector_area_l448_44872

open Real

theorem spinner_win_sector_area (r : ℝ) (P : ℝ)
  (h_r : r = 8) (h_P : P = 3 / 7) : 
  ∃ A : ℝ, A = 192 * π / 7 :=
by
  sorry

end spinner_win_sector_area_l448_44872


namespace original_number_l448_44806

theorem original_number (x : ℝ) (h : 1.2 * x = 1080) : x = 900 := by
  sorry

end original_number_l448_44806


namespace candle_height_half_after_9_hours_l448_44823

-- Define the initial heights and burn rates
def initial_height_first : ℝ := 12
def burn_rate_first : ℝ := 2
def initial_height_second : ℝ := 15
def burn_rate_second : ℝ := 3

-- Define the height functions after t hours
def height_first (t : ℝ) : ℝ := initial_height_first - burn_rate_first * t
def height_second (t : ℝ) : ℝ := initial_height_second - burn_rate_second * t

-- Prove that at t = 9, the height of the first candle is half the height of the second candle
theorem candle_height_half_after_9_hours : height_first 9 = 0.5 * height_second 9 := by
  sorry

end candle_height_half_after_9_hours_l448_44823


namespace radius_base_circle_of_cone_l448_44860

theorem radius_base_circle_of_cone 
  (θ : ℝ) (R : ℝ) (arc_length : ℝ) (r : ℝ)
  (h1 : θ = 120) 
  (h2 : R = 9)
  (h3 : arc_length = (θ / 360) * 2 * Real.pi * R)
  (h4 : 2 * Real.pi * r = arc_length)
  : r = 3 := 
sorry

end radius_base_circle_of_cone_l448_44860


namespace solve_diophantine_l448_44871

theorem solve_diophantine : ∀ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ (x^3 - y^3 = x * y + 61) → (x, y) = (6, 5) :=
by
  intros x y h
  sorry

end solve_diophantine_l448_44871


namespace total_daisies_sold_l448_44887

-- Conditions Definitions
def first_day_sales : ℕ := 45
def second_day_sales : ℕ := first_day_sales + 20
def third_day_sales : ℕ := 2 * second_day_sales - 10
def fourth_day_sales : ℕ := 120

-- Question: Prove that the total sales over the four days is 350.
theorem total_daisies_sold :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = 350 := by
  sorry

end total_daisies_sold_l448_44887


namespace pastries_left_to_take_home_l448_44853

def initial_cupcakes : ℕ := 7
def initial_cookies : ℕ := 5
def pastries_sold : ℕ := 4

theorem pastries_left_to_take_home :
  initial_cupcakes + initial_cookies - pastries_sold = 8 := by
  sorry

end pastries_left_to_take_home_l448_44853


namespace tommy_total_balloons_l448_44846

-- Define the conditions from part (a)
def original_balloons : Nat := 26
def additional_balloons : Nat := 34

-- Define the proof problem from part (c)
theorem tommy_total_balloons : original_balloons + additional_balloons = 60 := by
  -- Skip the actual proof
  sorry

end tommy_total_balloons_l448_44846


namespace find_m_from_expansion_l448_44835

theorem find_m_from_expansion (m n : ℤ) (h : (x : ℝ) → (x + 3) * (x + n) = x^2 + m * x - 21) : m = -4 :=
by
  sorry

end find_m_from_expansion_l448_44835


namespace complex_number_purely_imaginary_l448_44821

variable {m : ℝ}

theorem complex_number_purely_imaginary (h1 : 2 * m^2 + m - 1 = 0) (h2 : -m^2 - 3 * m - 2 ≠ 0) : m = 1/2 := by
  sorry

end complex_number_purely_imaginary_l448_44821


namespace base_subtraction_l448_44807

-- Define the base 8 number 765432_8 and its conversion to base 10
def base8Number : ℕ := 7 * (8^5) + 6 * (8^4) + 5 * (8^3) + 4 * (8^2) + 3 * (8^1) + 2 * (8^0)

-- Define the base 9 number 543210_9 and its conversion to base 10
def base9Number : ℕ := 5 * (9^5) + 4 * (9^4) + 3 * (9^3) + 2 * (9^2) + 1 * (9^1) + 0 * (9^0)

-- Lean 4 statement for the proof problem
theorem base_subtraction : (base8Number : ℤ) - (base9Number : ℤ) = -67053 := by
    sorry

end base_subtraction_l448_44807


namespace number_of_cherry_pie_days_l448_44815

theorem number_of_cherry_pie_days (A C : ℕ) (h1 : A + C = 7) (h2 : 12 * A = 12 * C + 12) : C = 3 :=
sorry

end number_of_cherry_pie_days_l448_44815


namespace distance_between_points_l448_44847

open Real

theorem distance_between_points :
  let P := (1, 3)
  let Q := (-5, 7)
  dist P Q = 2 * sqrt 13 :=
by
  let P := (1, 3)
  let Q := (-5, 7)
  sorry

end distance_between_points_l448_44847


namespace find_omega_value_l448_44896

theorem find_omega_value (ω : ℝ) (h : ω > 0) (h_dist : (1/2) * (2 * π / ω) = π / 6) : ω = 6 :=
by
  sorry

end find_omega_value_l448_44896


namespace walter_percent_of_dollar_l448_44873

theorem walter_percent_of_dollar
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (penny_value : Nat := 1)
  (nickel_value : Nat := 5)
  (dime_value : Nat := 10)
  (dollar_value : Nat := 100)
  (total_value := pennies * penny_value + nickels * nickel_value + dimes * dime_value) :
  pennies = 2 ∧ nickels = 3 ∧ dimes = 2 →
  (total_value * 100) / dollar_value = 37 :=
by
  sorry

end walter_percent_of_dollar_l448_44873


namespace num_girls_in_school_l448_44865

noncomputable def total_students : ℕ := 1600
noncomputable def sample_students : ℕ := 200
noncomputable def girls_less_than_boys_in_sample : ℕ := 10

-- Equations from conditions
def boys_in_sample (B G : ℕ) : Prop := G = B - girls_less_than_boys_in_sample
def sample_size (B G : ℕ) : Prop := B + G = sample_students

-- Proportion condition
def proportional_condition (G G_total : ℕ) : Prop := G * total_students = G_total * sample_students

-- Total number of girls in the school
def total_girls_in_school (G_total : ℕ) : Prop := G_total = 760

theorem num_girls_in_school :
  ∃ B G G_total : ℕ, boys_in_sample B G ∧ sample_size B G ∧ proportional_condition G G_total ∧ total_girls_in_school G_total :=
sorry

end num_girls_in_school_l448_44865


namespace g_difference_l448_44892

-- Define the function g(n)
def g (n : ℤ) : ℚ := (1/2 : ℚ) * n^2 * (n + 3)

-- State the theorem
theorem g_difference (s : ℤ) : g s - g (s - 1) = (1/2 : ℚ) * (3 * s - 2) := by
  sorry

end g_difference_l448_44892


namespace cauliflower_sales_l448_44804

namespace WeeklyMarket

def broccoliPrice := 3
def totalEarnings := 520
def broccolisSold := 19

def carrotPrice := 2
def spinachPrice := 4
def spinachWeight := 8 -- This is derived from solving $4S = 2S + $16 

def broccoliEarnings := broccolisSold * broccoliPrice
def carrotEarnings := spinachWeight * carrotPrice -- This is twice copied

def spinachEarnings : ℕ := spinachWeight * spinachPrice
def tomatoEarnings := broccoliEarnings + spinachEarnings

def otherEarnings : ℕ := broccoliEarnings + carrotEarnings + spinachEarnings + tomatoEarnings

def cauliflowerEarnings : ℕ := totalEarnings - otherEarnings -- This directly from subtraction of earnings

theorem cauliflower_sales : cauliflowerEarnings = 310 :=
  by
    -- only the statement part, no actual proof needed
    sorry

end WeeklyMarket

end cauliflower_sales_l448_44804


namespace smallest_odd_prime_factor_l448_44838

theorem smallest_odd_prime_factor (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (2023 ^ 8 + 1) % p = 0 ↔ p = 17 := 
by
  sorry

end smallest_odd_prime_factor_l448_44838


namespace inequality_solutions_l448_44850

theorem inequality_solutions (a : ℚ) :
  (∀ x : ℕ, 0 < x ∧ x ≤ 3 → 3 * (x - 1) < 2 * (x + a) - 5) →
  (∃ x : ℕ, 0 < x ∧ x = 4 → ¬ (3 * (x - 1) < 2 * (x + a) - 5)) →
  (5 / 2 < a ∧ a ≤ 3) :=
sorry

end inequality_solutions_l448_44850


namespace number_of_girls_l448_44856

theorem number_of_girls
  (total_boys : ℕ)
  (total_boys_eq : total_boys = 10)
  (fraction_girls_reading : ℚ)
  (fraction_girls_reading_eq : fraction_girls_reading = 5/6)
  (fraction_boys_reading : ℚ)
  (fraction_boys_reading_eq : fraction_boys_reading = 4/5)
  (total_not_reading : ℕ)
  (total_not_reading_eq : total_not_reading = 4)
  (G : ℝ)
  (remaining_girls_reading : (1 - fraction_girls_reading) * G = 2)
  (remaining_boys_not_reading : (1 - fraction_boys_reading) * total_boys = 2)
  (remaining_total_not_reading : 2 + 2 = total_not_reading)
  : G = 12 :=
by
  sorry

end number_of_girls_l448_44856


namespace mass_percentage_of_Ba_l448_44839

theorem mass_percentage_of_Ba {BaX : Type} {molar_mass_Ba : ℝ} {compound_mass : ℝ} {mass_Ba : ℝ}:
  molar_mass_Ba = 137.33 ∧ 
  compound_mass = 100 ∧
  mass_Ba = 66.18 →
  (mass_Ba / compound_mass * 100) = 66.18 :=
by
  sorry

end mass_percentage_of_Ba_l448_44839


namespace first_term_geometric_l448_44810

-- Definition: geometric sequence properties
variables (a r : ℚ) -- sequence terms are rational numbers
variables (n : ℕ)

-- Conditions: fifth and sixth terms of a geometric sequence
def fifth_term_geometric (a r : ℚ) : ℚ := a * r^4
def sixth_term_geometric (a r : ℚ) : ℚ := a * r^5

-- Proof: given conditions
theorem first_term_geometric (a r : ℚ) (h1 : fifth_term_geometric a r = 48) 
  (h2 : sixth_term_geometric a r = 72) : a = 768 / 81 :=
by {
  sorry
}

end first_term_geometric_l448_44810


namespace angle_half_second_quadrant_l448_44867

theorem angle_half_second_quadrant (α : ℝ) (k : ℤ) :
  (π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
  (∃ m : ℤ, (π / 4 + m * π < α / 2 ∧ α / 2 < π / 2 + m * π)) ∨ 
  (∃ n : ℤ, (5 * π / 4 + n * π < α / 2 ∧ α / 2 < 3 * π / 2 + n * π)) :=
by
  sorry

end angle_half_second_quadrant_l448_44867


namespace evaluate_expression_l448_44851

-- Define the expression as given in the problem
def expr1 : ℤ := |9 - 8 * (3 - 12)|
def expr2 : ℤ := |5 - 11|

-- Define the mathematical equivalence
theorem evaluate_expression : (expr1 - expr2) = 75 := by
  sorry

end evaluate_expression_l448_44851


namespace arithmetic_seq_a12_l448_44825

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (a1 d : ℝ) 
  (h_arith : arithmetic_seq a a1 d)
  (h7_and_9 : a 7 + a 9 = 16)
  (h4 : a 4 = 1) :
  a 12 = 15 :=
by
  sorry

end arithmetic_seq_a12_l448_44825


namespace insurance_covers_80_percent_l448_44813

-- Definitions from the problem conditions
def cost_per_aid : ℕ := 2500
def num_aids : ℕ := 2
def johns_payment : ℕ := 1000

-- Total cost of hearing aids
def total_cost : ℕ := cost_per_aid * num_aids

-- Insurance payment
def insurance_payment : ℕ := total_cost - johns_payment

-- The theorem to prove
theorem insurance_covers_80_percent :
  (insurance_payment * 100 / total_cost) = 80 :=
by
  sorry

end insurance_covers_80_percent_l448_44813


namespace boat_trip_duration_l448_44840

noncomputable def boat_trip_time (B P : ℝ) : Prop :=
  (P = 4 * B) ∧ (B + P = 10)

theorem boat_trip_duration (B P : ℝ) (h : boat_trip_time B P) : B = 2 :=
by
  cases h with
  | intro hP hTotal =>
    sorry

end boat_trip_duration_l448_44840


namespace trajectory_of_M_l448_44824

theorem trajectory_of_M (M : ℝ × ℝ) (h : (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2)) :
  (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2) :=
by
  sorry

end trajectory_of_M_l448_44824


namespace problem_a_problem_c_l448_44843

variable {a b : ℝ}

theorem problem_a (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : ab ≤ 1 / 8 :=
by
  sorry

theorem problem_c (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : 1 / a + 2 / b ≥ 9 :=
by
  sorry

end problem_a_problem_c_l448_44843


namespace extreme_value_and_inequality_l448_44817

theorem extreme_value_and_inequality
  (f : ℝ → ℝ)
  (a c : ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_extreme : f 1 = -2)
  (h_f_def : ∀ x : ℝ, f x = a * x^3 + c * x)
  (h_a_c : a = 1 ∧ c = -3) :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, 1 < x → deriv f x > 0) ∧
  f (-1) = 2 ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 → |f x₁ - f x₂| < 4) :=
by sorry

end extreme_value_and_inequality_l448_44817


namespace find_digit_l448_44845

theorem find_digit (p q r : ℕ) (hq : p ≠ q) (hr : p ≠ r) (hq' : q ≠ r) 
    (hp_pos : 0 < p ∧ p < 10)
    (hq_pos : 0 < q ∧ q < 10)
    (hr_pos : 0 < r ∧ r < 10)
    (h1 : 10 * p + q = 17)
    (h2 : 10 * p + r = 13)
    (h3 : p + q + r = 11) : 
    q = 7 :=
sorry

end find_digit_l448_44845


namespace man_speed_against_current_proof_l448_44811

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end man_speed_against_current_proof_l448_44811


namespace required_more_visits_l448_44836

-- Define the conditions
def n := 395
def m := 2
def v1 := 135
def v2 := 112
def v3 := 97

-- Define the target statement
theorem required_more_visits : (n * m) - (v1 + v2 + v3) = 446 := by
  sorry

end required_more_visits_l448_44836


namespace inequality_proof_l448_44889

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l448_44889


namespace find_common_difference_l448_44888

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

-- defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ + n * d

-- condition for the sum of even indexed terms
def sum_even_terms (a : ℕ → ℝ) : ℝ := a 2 + a 4 + a 6 + a 8 + a 10

-- condition for the sum of odd indexed terms
def sum_odd_terms (a : ℕ → ℝ) : ℝ := a 1 + a 3 + a 5 + a 7 + a 9

-- main theorem to prove
theorem find_common_difference
  (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h_even_sum : sum_even_terms a = 30)
  (h_odd_sum : sum_odd_terms a = 25) :
  d = 1 := by
  sorry

end find_common_difference_l448_44888


namespace tailor_cut_skirt_l448_44895

theorem tailor_cut_skirt (cut_pants cut_skirt : ℝ) (h1 : cut_pants = 0.5) (h2 : cut_skirt = cut_pants + 0.25) : cut_skirt = 0.75 :=
by
  sorry

end tailor_cut_skirt_l448_44895


namespace correct_product_l448_44876

theorem correct_product (a b : ℕ) (a' : ℕ) (h1 : a' = (a % 10) * 10 + (a / 10)) 
  (h2 : a' * b = 143) (h3 : 10 ≤ a ∧ a < 100):
  a * b = 341 :=
sorry

end correct_product_l448_44876


namespace union_complement_eq_set_l448_44884

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end union_complement_eq_set_l448_44884


namespace smallest_sum_of_inverses_l448_44820

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end smallest_sum_of_inverses_l448_44820


namespace articles_for_z_men_l448_44874

-- The necessary conditions and given values
def articles_produced (men hours days : ℕ) := men * hours * days

theorem articles_for_z_men (x z : ℕ) (H : articles_produced x x x = x^2) :
  articles_produced z z z = z^3 / x := by
  sorry

end articles_for_z_men_l448_44874


namespace find_ab_l448_44861

theorem find_ab (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by
  sorry

end find_ab_l448_44861


namespace john_weight_end_l448_44880

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l448_44880


namespace niko_total_profit_l448_44828

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l448_44828


namespace arithmetic_seq_a7_value_l448_44832

theorem arithmetic_seq_a7_value {a : ℕ → ℝ} (h_positive : ∀ n, 0 < a n)
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_eq : 3 * a 6 - (a 7) ^ 2 + 3 * a 8 = 0) : a 7 = 6 :=
  sorry

end arithmetic_seq_a7_value_l448_44832


namespace no_all_perfect_squares_l448_44855

theorem no_all_perfect_squares (x : ℤ) 
  (h1 : ∃ a : ℤ, 2 * x - 1 = a^2) 
  (h2 : ∃ b : ℤ, 5 * x - 1 = b^2) 
  (h3 : ∃ c : ℤ, 13 * x - 1 = c^2) : 
  False :=
sorry

end no_all_perfect_squares_l448_44855


namespace find_value_l448_44878

theorem find_value (x : ℝ) (f₁ f₂ : ℝ) (p : ℝ) (y₁ y₂ : ℝ) 
  (h1 : x * f₁ = (p * x) * y₁)
  (h2 : x * f₂ = (p * x) * y₂)
  (hf₁ : f₁ = 1 / 3)
  (hx : x = 4)
  (hy₁ : y₁ = 8)
  (hf₂ : f₂ = 1 / 8):
  y₂ = 3 := by
sorry

end find_value_l448_44878


namespace darry_steps_l448_44854

theorem darry_steps (f_steps : ℕ) (f_times : ℕ) (s_steps : ℕ) (s_times : ℕ) (no_other_steps : ℕ)
  (hf : f_steps = 11)
  (hf_times : f_times = 10)
  (hs : s_steps = 6)
  (hs_times : s_times = 7)
  (h_no_other : no_other_steps = 0) :
  (f_steps * f_times + s_steps * s_times + no_other_steps = 152) :=
by
  sorry

end darry_steps_l448_44854


namespace total_weight_of_lifts_l448_44837

theorem total_weight_of_lifts 
  (F S : ℕ)
  (h1 : F = 400)
  (h2 : 2 * F = S + 300) :
  F + S = 900 :=
by
  sorry

end total_weight_of_lifts_l448_44837


namespace regular_tetrahedron_fourth_vertex_l448_44868

theorem regular_tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ), 
    ((x, y, z) = (0, 0, 6) ∨ (x, y, z) = (0, 0, -6)) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 6) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 5) ^ 2 + (y - 0) ^ 2 + (z - 6) ^ 2 = 36) := 
by
  sorry

end regular_tetrahedron_fourth_vertex_l448_44868


namespace multiplier_condition_l448_44801

theorem multiplier_condition (a b : ℚ) (h : a * b ≤ b) : (b ≥ 0 ∧ a ≤ 1) ∨ (b ≤ 0 ∧ a ≥ 1) :=
by 
  sorry

end multiplier_condition_l448_44801


namespace function_decreasing_range_l448_44852

theorem function_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1) ≤ (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1)) ↔ (0 ≤ a ∧ a ≤ 1 / 3) :=
sorry

end function_decreasing_range_l448_44852


namespace oil_cylinder_capacity_l448_44893

theorem oil_cylinder_capacity
  (C : ℚ) -- total capacity of the cylinder, given as a rational number
  (h1 : 3 / 4 * C + 4 = 4 / 5 * C) -- equation representing the condition of initial and final amounts of oil in the cylinder
  : C = 80 := -- desired result showing the total capacity

sorry

end oil_cylinder_capacity_l448_44893


namespace arithmetic_sequence_value_l448_44814

theorem arithmetic_sequence_value (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_cond : a 3 + a 9 = 15 - a 6) : a 6 = 5 :=
sorry

end arithmetic_sequence_value_l448_44814


namespace smallest_set_of_circular_handshakes_l448_44829

def circular_handshake_smallest_set (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n / 2 else (n / 2) + 1

theorem smallest_set_of_circular_handshakes :
  circular_handshake_smallest_set 36 = 18 :=
by
  sorry

end smallest_set_of_circular_handshakes_l448_44829


namespace total_meals_sold_l448_44899

-- Definitions based on the conditions
def ratio_kids_adult := 2 / 1
def kids_meals := 8

-- The proof problem statement
theorem total_meals_sold : (∃ adults_meals : ℕ, 2 * adults_meals = kids_meals) → (kids_meals + 4 = 12) := 
by 
  sorry

end total_meals_sold_l448_44899


namespace hyperbola_proof_l448_44848

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end hyperbola_proof_l448_44848


namespace percentage_increase_correct_l448_44849

def bookstore_earnings : ℕ := 60
def tutoring_earnings : ℕ := 40
def new_bookstore_earnings : ℕ := 100
def additional_tutoring_fee : ℕ := 15
def old_total_earnings : ℕ := bookstore_earnings + tutoring_earnings
def new_total_earnings : ℕ := new_bookstore_earnings + (tutoring_earnings + additional_tutoring_fee)
def overall_percentage_increase : ℚ := (((new_total_earnings - old_total_earnings : ℚ) / old_total_earnings) * 100)

theorem percentage_increase_correct :
  overall_percentage_increase = 55 := sorry

end percentage_increase_correct_l448_44849


namespace scatter_plot_convention_l448_44827

def explanatory_variable := "x-axis"
def predictor_variable := "y-axis"

theorem scatter_plot_convention :
  explanatory_variable = "x-axis" ∧ predictor_variable = "y-axis" :=
by sorry

end scatter_plot_convention_l448_44827


namespace odd_n_divisibility_l448_44834

theorem odd_n_divisibility (n : ℤ) : (∃ a : ℤ, n ∣ 4 * a^2 - 1) ↔ (n % 2 ≠ 0) :=
by
  sorry

end odd_n_divisibility_l448_44834


namespace line_passes_through_point_l448_44863

theorem line_passes_through_point :
  ∀ (m : ℝ), (∃ y : ℝ, y - 2 = m * (-1) + m) :=
by
  intros m
  use 2
  sorry

end line_passes_through_point_l448_44863


namespace find_ff_half_l448_44879

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x + 1 else -x + 3

theorem find_ff_half : f (f (1 / 2)) = 3 / 2 := 
by 
  sorry

end find_ff_half_l448_44879


namespace largest_k_inequality_l448_44866

theorem largest_k_inequality :
  ∃ k : ℝ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 3 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a * b - b * c - c * a)) ∧ k = 5 :=
sorry

end largest_k_inequality_l448_44866


namespace division_by_n_minus_1_squared_l448_44822

theorem division_by_n_minus_1_squared (n : ℕ) (h : n > 2) : (n ^ (n - 1) - 1) % ((n - 1) ^ 2) = 0 :=
sorry

end division_by_n_minus_1_squared_l448_44822


namespace reservoir_capacity_l448_44812

-- Definitions based on the conditions
def storm_deposit : ℚ := 120 * 10^9
def final_full_percentage : ℚ := 0.85
def initial_full_percentage : ℚ := 0.55
variable (C : ℚ) -- total capacity of the reservoir in gallons

-- The statement we want to prove
theorem reservoir_capacity :
  final_full_percentage * C - initial_full_percentage * C = storm_deposit →
  C = 400 * 10^9
:= by
  sorry

end reservoir_capacity_l448_44812


namespace find_divisor_l448_44844

-- Definitions from the condition
def original_number : ℕ := 724946
def least_number_subtracted : ℕ := 6
def remaining_number : ℕ := original_number - least_number_subtracted

theorem find_divisor (h1 : remaining_number % least_number_subtracted = 0) :
  Nat.gcd original_number least_number_subtracted = 2 :=
sorry

end find_divisor_l448_44844


namespace exists_positive_integers_seq_l448_44805

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.sum

def prod_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.prod

theorem exists_positive_integers_seq (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n.succ → ℕ),
    (∀ i : Fin n, sum_of_digits (a i) < sum_of_digits (a i.succ)) ∧
    (∀ i : Fin n, sum_of_digits (a i) = prod_of_digits (a i.succ)) ∧
    (∀ i : Fin n, 0 < (a i)) :=
by
  sorry

end exists_positive_integers_seq_l448_44805


namespace problem_inequality_l448_44890

theorem problem_inequality 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m + n = 1) : 
  (m + 1 / m) * (n + 1 / n) ≥ 25 / 4 := 
sorry

end problem_inequality_l448_44890


namespace max_profit_at_80_l448_44859

-- Definitions based on conditions
def cost_price : ℝ := 40
def functional_relationship (x : ℝ) : ℝ := -x + 140
def profit (x : ℝ) : ℝ := (x - cost_price) * functional_relationship x

-- Statement to prove that maximum profit is achieved at x = 80
theorem max_profit_at_80 : (40 ≤ 80) → (80 ≤ 80) → profit 80 = 2400 := by
  sorry

end max_profit_at_80_l448_44859


namespace y_decreases_as_x_less_than_4_l448_44809

theorem y_decreases_as_x_less_than_4 (x : ℝ) : (x < 4) → ((x - 4)^2 + 3 < (4 - 4)^2 + 3) :=
by
  sorry

end y_decreases_as_x_less_than_4_l448_44809
