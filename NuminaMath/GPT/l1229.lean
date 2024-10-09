import Mathlib

namespace solve_inequality_l1229_122928

theorem solve_inequality (x : ℝ) : 2 * x^2 + 8 * x ≤ -6 ↔ -3 ≤ x ∧ x ≤ -1 :=
by
  sorry

end solve_inequality_l1229_122928


namespace quadratic_inequality_solution_l1229_122992

-- Given a quadratic inequality, prove the solution set in interval notation.
theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x ^ 2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 :=
sorry

end quadratic_inequality_solution_l1229_122992


namespace soccer_players_l1229_122906

/-- 
If the total number of socks in the washing machine is 16,
and each player wears a pair of socks (2 socks per player), 
then the number of players is 8. 
-/
theorem soccer_players (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) : total_socks / socks_per_player = 8 :=
by
  -- Proof goes here
  sorry

end soccer_players_l1229_122906


namespace cyclic_quadrilateral_ptolemy_l1229_122987

theorem cyclic_quadrilateral_ptolemy 
  (a b c d : ℝ) 
  (h : a + b + c + d = Real.pi) :
  Real.sin (a + b) * Real.sin (b + c) = Real.sin a * Real.sin c + Real.sin b * Real.sin d :=
by
  sorry

end cyclic_quadrilateral_ptolemy_l1229_122987


namespace trapezoid_other_base_possible_lengths_l1229_122920

-- Definition of the trapezoid problem.
structure Trapezoid where
  height : ℕ
  leg1 : ℕ
  leg2 : ℕ
  base1 : ℕ

-- The given conditions
def trapezoid_data : Trapezoid :=
{ height := 12, leg1 := 20, leg2 := 15, base1 := 42 }

-- The proof problem in Lean 4 statement
theorem trapezoid_other_base_possible_lengths (t : Trapezoid) :
  t = trapezoid_data → (∃ b : ℕ, (b = 17 ∨ b = 35)) :=
by
  intro h_data_eq
  sorry

end trapezoid_other_base_possible_lengths_l1229_122920


namespace translate_sin_eq_cos_l1229_122911

theorem translate_sin_eq_cos (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  (∀ x, Real.cos (x - Real.pi / 6) = Real.sin (x + φ)) → φ = Real.pi / 3 :=
by
  sorry

end translate_sin_eq_cos_l1229_122911


namespace card_worth_l1229_122941

theorem card_worth (value_per_card : ℕ) (num_cards_traded : ℕ) (profit : ℕ) (value_traded : ℕ) (worth_received : ℕ) :
  value_per_card = 8 →
  num_cards_traded = 2 →
  profit = 5 →
  value_traded = num_cards_traded * value_per_card →
  worth_received = value_traded + profit →
  worth_received = 21 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end card_worth_l1229_122941


namespace inverse_proportion_l1229_122908

theorem inverse_proportion (α β k : ℝ) (h1 : α * β = k) (h2 : α = 5) (h3 : β = 10) : (α = 25 / 2) → (β = 4) := by sorry

end inverse_proportion_l1229_122908


namespace total_cookies_eaten_l1229_122982

-- Definitions of the cookies eaten
def charlie_cookies := 15
def father_cookies := 10
def mother_cookies := 5

-- The theorem to prove the total number of cookies eaten
theorem total_cookies_eaten : charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end total_cookies_eaten_l1229_122982


namespace algebraic_expression_for_A_l1229_122965

variable {x y A : ℝ}

theorem algebraic_expression_for_A
  (h : (3 * x + 2 * y) ^ 2 = (3 * x - 2 * y) ^ 2 + A) :
  A = 24 * x * y :=
sorry

end algebraic_expression_for_A_l1229_122965


namespace perpendicular_lines_slope_l1229_122915

theorem perpendicular_lines_slope (a : ℝ)
  (h : (a * (a + 2)) = -1) : a = -1 :=
sorry

end perpendicular_lines_slope_l1229_122915


namespace possible_measure_of_angle_AOC_l1229_122963

-- Given conditions
def angle_AOB : ℝ := 120
def OC_bisects_angle_AOB (x : ℝ) : Prop := x = 60
def OD_bisects_angle_AOB_and_OC_bisects_angle (x y : ℝ) : Prop :=
  (y = 60 ∧ (x = 30 ∨ x = 90))

-- Theorem statement
theorem possible_measure_of_angle_AOC (angle_AOC : ℝ) :
  (OC_bisects_angle_AOB angle_AOC ∨ 
  (OD_bisects_angle_AOB_and_OC_bisects_angle angle_AOC 60)) →
  (angle_AOC = 30 ∨ angle_AOC = 60 ∨ angle_AOC = 90) :=
by
  sorry

end possible_measure_of_angle_AOC_l1229_122963


namespace meaningful_sqrt_range_l1229_122901

theorem meaningful_sqrt_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end meaningful_sqrt_range_l1229_122901


namespace number_of_organizations_in_foundation_l1229_122958

def company_raised : ℕ := 2500
def donation_percentage : ℕ := 80
def each_organization_receives : ℕ := 250
def total_donated : ℕ := (donation_percentage * company_raised) / 100

theorem number_of_organizations_in_foundation : total_donated / each_organization_receives = 8 :=
by
  sorry

end number_of_organizations_in_foundation_l1229_122958


namespace quadratic_inequality_solution_l1229_122969

theorem quadratic_inequality_solution (x : ℝ) :
    -15 * x^2 + 10 * x + 5 > 0 ↔ (-1 / 3 : ℝ) < x ∧ x < 1 :=
by
  sorry

end quadratic_inequality_solution_l1229_122969


namespace nancy_crayons_l1229_122907

theorem nancy_crayons (p c t : ℕ) (h1 : p = 41) (h2 : c = 15) (h3 : t = p * c) : t = 615 :=
by
  sorry

end nancy_crayons_l1229_122907


namespace angle_A_eq_pi_over_3_perimeter_eq_24_l1229_122940

namespace TriangleProof

-- We introduce the basic setup for the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition
axiom condition : 2 * b = 2 * a * Real.cos C + c

-- Part 1: Prove angle A is π/3
theorem angle_A_eq_pi_over_3 (h : 2 * b = 2 * a * Real.cos C + c) :
  A = Real.pi / 3 :=
sorry

-- Part 2: Given a = 10 and the area is 8√3, prove perimeter is 24
theorem perimeter_eq_24 (a_eq_10 : a = 10) (area_eq_8sqrt3 : 8 * Real.sqrt 3 = (1 / 2) * b * c * Real.sin A) :
  a + b + c = 24 :=
sorry

end TriangleProof

end angle_A_eq_pi_over_3_perimeter_eq_24_l1229_122940


namespace students_attend_Purum_Elementary_School_l1229_122972
open Nat

theorem students_attend_Purum_Elementary_School (P N : ℕ) 
  (h1 : P + N = 41) (h2 : P = N + 3) : P = 22 :=
sorry

end students_attend_Purum_Elementary_School_l1229_122972


namespace prob_l1229_122980

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + 1 / (2 + 1 / x))

theorem prob (x1 x2 x3 : ℝ) (h1 : x1 = 0) 
  (h2 : 2 + 1 / x2 = 0) 
  (h3 : 2 + 1 / (2 + 1 / x3) = 0) : 
  x1 + x2 + x3 = -9 / 10 := 
sorry

end prob_l1229_122980


namespace polynomial_decomposition_l1229_122978

-- Define the given polynomial
def P (x y z : ℝ) : ℝ := x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2

-- Define the target decomposition
def Q (x y z : ℝ) : ℝ := (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2

theorem polynomial_decomposition (x y z : ℝ) : P x y z = Q x y z :=
  sorry

end polynomial_decomposition_l1229_122978


namespace magic_king_total_episodes_l1229_122968

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l1229_122968


namespace Bryan_has_more_skittles_l1229_122914

-- Definitions for conditions
def Bryan_skittles : ℕ := 50
def Ben_mms : ℕ := 20

-- Main statement to be proven
theorem Bryan_has_more_skittles : Bryan_skittles > Ben_mms ∧ Bryan_skittles - Ben_mms = 30 :=
by
  sorry

end Bryan_has_more_skittles_l1229_122914


namespace find_coordinates_of_P_l1229_122930

-- Define the problem conditions
def P (m : ℤ) := (2 * m + 4, m - 1)
def A := (2, -4)
def line_l (y : ℤ) := y = -4
def P_on_line_l (m : ℤ) := line_l (m - 1)

theorem find_coordinates_of_P (m : ℤ) (h : P_on_line_l m) : P m = (-2, -4) := 
  by sorry

end find_coordinates_of_P_l1229_122930


namespace min_time_to_one_ball_l1229_122947

-- Define the problem in Lean
theorem min_time_to_one_ball (n : ℕ) (h : n = 99) : 
  ∃ T : ℕ, T = 98 ∧ ∀ t < T, ∃ ball_count : ℕ, ball_count > 1 :=
by
  -- Since we are not providing the proof, we use "sorry"
  sorry

end min_time_to_one_ball_l1229_122947


namespace union_of_sets_l1229_122962

def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }

theorem union_of_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x : ℝ | 2 < x ∧ x < 10 }) :=
by
  sorry

end union_of_sets_l1229_122962


namespace not_q_true_l1229_122921

theorem not_q_true (p q : Prop) (hp : p = true) (hq : q = false) : ¬q = true :=
by
  sorry

end not_q_true_l1229_122921


namespace factor_expression_l1229_122951

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_expression_l1229_122951


namespace train_crosses_tunnel_in_45_sec_l1229_122910

/-- Given the length of the train, the length of the platform, the length of the tunnel, 
and the time taken to cross the platform, prove the time taken for the train to cross the tunnel is 45 seconds. -/
theorem train_crosses_tunnel_in_45_sec (l_train : ℕ) (l_platform : ℕ) (t_platform : ℕ) (l_tunnel : ℕ)
  (h_train_length : l_train = 330)
  (h_platform_length : l_platform = 180)
  (h_time_platform : t_platform = 15)
  (h_tunnel_length : l_tunnel = 1200) :
  (l_train + l_tunnel) / ((l_train + l_platform) / t_platform) = 45 :=
by
  -- placeholder for the actual proof
  sorry

end train_crosses_tunnel_in_45_sec_l1229_122910


namespace find_C_coordinates_l1229_122932

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 11, y := 9 }
def B : Point := { x := 2, y := -3 }
def D : Point := { x := -1, y := 3 }

-- Define the isosceles property
def is_isosceles (A B C : Point) : Prop :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2) = Real.sqrt ((A.x - C.x) ^ 2 + (A.y - C.y) ^ 2)

-- Define the midpoint property
def is_midpoint (D B C : Point) : Prop :=
  D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2

theorem find_C_coordinates (C : Point)
  (h_iso : is_isosceles A B C)
  (h_mid : is_midpoint D B C) :
  C = { x := -4, y := 9 } := 
  sorry

end find_C_coordinates_l1229_122932


namespace min_socks_no_conditions_l1229_122974

theorem min_socks_no_conditions (m n : Nat) (h : (m * (m - 1) = 2 * (m + n) * (m + n - 1))) : 
  m + n ≥ 4 := sorry

end min_socks_no_conditions_l1229_122974


namespace inequality_proof_l1229_122929

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d :=
  sorry

end inequality_proof_l1229_122929


namespace distinguishable_triangles_count_l1229_122955

def count_distinguishable_triangles (colors : ℕ) : ℕ :=
  let corner_cases := colors + (colors * (colors - 1)) + (colors * (colors - 1) * (colors - 2) / 6)
  let edge_cases := colors * colors
  let center_cases := colors
  corner_cases * edge_cases * center_cases

theorem distinguishable_triangles_count :
  count_distinguishable_triangles 8 = 61440 :=
by
  unfold count_distinguishable_triangles
  -- corner_cases = 8 + 8 * 7 + (8 * 7 * 6) / 6 = 120
  -- edge_cases = 8 * 8 = 64
  -- center_cases = 8
  -- Total = 120 * 64 * 8 = 61440
  sorry

end distinguishable_triangles_count_l1229_122955


namespace smallest_value_a_plus_b_l1229_122953

theorem smallest_value_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 3^7 * 5^3 = a^b) : a + b = 3376 :=
sorry

end smallest_value_a_plus_b_l1229_122953


namespace distinct_9_pointed_stars_l1229_122912

-- Define a function to count the distinct n-pointed stars for a given n
def count_distinct_stars (n : ℕ) : ℕ :=
  -- Functionality to count distinct stars will be implemented here
  sorry

-- Theorem stating the number of distinct 9-pointed stars
theorem distinct_9_pointed_stars : count_distinct_stars 9 = 2 :=
  sorry

end distinct_9_pointed_stars_l1229_122912


namespace marbles_left_mrs_hilt_marbles_left_l1229_122933

-- Define the initial number of marbles
def initial_marbles : ℕ := 38

-- Define the number of marbles lost
def marbles_lost : ℕ := 15

-- Define the number of marbles given away
def marbles_given_away : ℕ := 6

-- Define the number of marbles found
def marbles_found : ℕ := 8

-- Use these definitions to calculate the total number of marbles left
theorem marbles_left : ℕ :=
  initial_marbles - marbles_lost - marbles_given_away + marbles_found

-- Prove that total number of marbles left is 25
theorem mrs_hilt_marbles_left : marbles_left = 25 := by 
  sorry

end marbles_left_mrs_hilt_marbles_left_l1229_122933


namespace range_of_alpha_l1229_122937

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 5 * x

theorem range_of_alpha (α : ℝ) (h₀ : -1 < α) (h₁ : α < 1) (h₂ : f (1 - α) + f (1 - α^2) < 0) : 1 < α ∧ α < Real.sqrt 2 := by
  sorry

end range_of_alpha_l1229_122937


namespace total_worth_of_travelers_checks_l1229_122946

theorem total_worth_of_travelers_checks (x y : ℕ) (h1 : x + y = 30) (h2 : 50 * (x - 18) + 100 * y = 900) : 
  50 * x + 100 * y = 1800 := 
by
  sorry

end total_worth_of_travelers_checks_l1229_122946


namespace tank_capacity_l1229_122996

theorem tank_capacity (one_third_full : ℚ) (added_water : ℚ) (capacity : ℚ) 
  (h1 : one_third_full = 1 / 3) 
  (h2 : 2 * one_third_full * capacity = 16) 
  (h3 : added_water = 16) 
  : capacity = 24 := 
by
  sorry

end tank_capacity_l1229_122996


namespace minimum_value_l1229_122944

theorem minimum_value(a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (2 / a + 3 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_l1229_122944


namespace curlers_count_l1229_122939

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l1229_122939


namespace not_sufficient_nor_necessary_l1229_122986

theorem not_sufficient_nor_necessary (a b : ℝ) :
  ¬((a^2 > b^2) → (a > b)) ∧ ¬((a > b) → (a^2 > b^2)) :=
by
  sorry

end not_sufficient_nor_necessary_l1229_122986


namespace cat_catches_mouse_l1229_122988

-- Define the distances
def AB := 200
def BC := 140
def CD := 20

-- Define the speeds (in meters per minute)
def mouse_speed := 60
def cat_speed := 80

-- Define the total distances the mouse and cat travel
def mouse_total_distance := 320 -- The mouse path is along a zigzag route initially specified in the problem
def cat_total_distance := AB + BC + CD -- 360 meters as calculated

-- Define the times they take to reach point D
def mouse_time := mouse_total_distance / mouse_speed -- 5.33 minutes
def cat_time := cat_total_distance / cat_speed -- 4.5 minutes

-- Proof problem statement
theorem cat_catches_mouse : cat_time < mouse_time := 
by
  sorry

end cat_catches_mouse_l1229_122988


namespace combined_selling_price_correct_l1229_122984

def cost_A : ℕ := 500
def cost_B : ℕ := 800
def cost_C : ℕ := 1200
def profit_A : ℕ := 25
def profit_B : ℕ := 30
def profit_C : ℕ := 20

def selling_price (cost profit_percentage : ℕ) : ℕ :=
  cost + (profit_percentage * cost / 100)

def combined_selling_price : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

theorem combined_selling_price_correct : combined_selling_price = 3105 := by
  sorry

end combined_selling_price_correct_l1229_122984


namespace range_of_m_l1229_122957

noncomputable def f (x m : ℝ) := (1/2) * x^2 + m * x + Real.log x

noncomputable def f_prime (x m : ℝ) := x + 1/x + m

theorem range_of_m (x0 m : ℝ) 
  (h1 : (1/2) ≤ x0 ∧ x0 ≤ 3) 
  (unique_x0 : ∀ y, f_prime y m = 0 → y = x0) 
  (cond1 : f_prime (1/2) m < 0) 
  (cond2 : f_prime 3 m ≥ 0) 
  : -10 / 3 ≤ m ∧ m < -5 / 2 :=
sorry

end range_of_m_l1229_122957


namespace petya_can_write_divisible_by_2019_l1229_122970

open Nat

theorem petya_can_write_divisible_by_2019 (M : ℕ) (h : ∃ k : ℕ, M = (10^k - 1) / 9) : ∃ N : ℕ, (N = (10^M - 1) / 9) ∧ 2019 ∣ N :=
by
  sorry

end petya_can_write_divisible_by_2019_l1229_122970


namespace trains_crossing_time_l1229_122973

-- Definitions based on conditions
def train_length : ℕ := 120
def time_train1_cross_pole : ℕ := 10
def time_train2_cross_pole : ℕ := 15

-- Question reformulated as a proof goal
theorem trains_crossing_time :
  let v1 := train_length / time_train1_cross_pole  -- Speed of train 1
  let v2 := train_length / time_train2_cross_pole  -- Speed of train 2
  let relative_speed := v1 + v2                    -- Relative speed in opposite directions
  let total_distance := train_length + train_length -- Sum of both trains' lengths
  let time_to_cross := total_distance / relative_speed -- Time to cross each other
  time_to_cross = 12 := 
by
  -- The proof here is stated, but not needed in this task
  -- All necessary computation steps
  sorry

end trains_crossing_time_l1229_122973


namespace shaded_area_percentage_is_correct_l1229_122938

noncomputable def total_area_of_square : ℕ := 49

noncomputable def area_of_first_shaded_region : ℕ := 2^2

noncomputable def area_of_second_shaded_region : ℕ := 25 - 9

noncomputable def area_of_third_shaded_region : ℕ := 49 - 36

noncomputable def total_shaded_area : ℕ :=
  area_of_first_shaded_region + area_of_second_shaded_region + area_of_third_shaded_region

noncomputable def percent_shaded_area : ℚ :=
  (total_shaded_area : ℚ) / total_area_of_square * 100

theorem shaded_area_percentage_is_correct :
  percent_shaded_area = 67.35 := by
sorry

end shaded_area_percentage_is_correct_l1229_122938


namespace max_remainder_l1229_122918

theorem max_remainder (y : ℕ) : 
  ∃ q r : ℕ, y = 11 * q + r ∧ r < 11 ∧ r = 10 := by sorry

end max_remainder_l1229_122918


namespace division_by_power_of_ten_l1229_122950

theorem division_by_power_of_ten (a b : ℕ) (h_a : a = 10^7) (h_b : b = 5 * 10^4) : a / b = 200 := by
  sorry

end division_by_power_of_ten_l1229_122950


namespace trajectory_eqn_of_point_Q_l1229_122931

theorem trajectory_eqn_of_point_Q 
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (A : ℝ × ℝ := (-2, 0))
  (B : ℝ × ℝ := (2, 0))
  (l : ℝ := 10 / 3) 
  (hP_on_l : P.1 = l)
  (hQ_on_AP : (Q.2 * -4) = Q.1 * (P.2 - 0) - (P.2 * -4))
  (hBP_perp_BQ : (Q.2 * 4) = -Q.1 * ((3 * P.2) / 4 - 2))
: (Q.1^2 / 4) + Q.2^2 = 1 :=
sorry

end trajectory_eqn_of_point_Q_l1229_122931


namespace fewer_pushups_l1229_122971

theorem fewer_pushups (sets: ℕ) (pushups_per_set : ℕ) (total_pushups : ℕ) 
  (h1 : sets = 3) (h2 : pushups_per_set = 15) (h3 : total_pushups = 40) :
  sets * pushups_per_set - total_pushups = 5 :=
by
  sorry

end fewer_pushups_l1229_122971


namespace math_problem_l1229_122990

theorem math_problem (x : ℤ) :
  let a := 1990 * x + 1989
  let b := 1990 * x + 1990
  let c := 1990 * x + 1991
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end math_problem_l1229_122990


namespace triangle_is_obtuse_l1229_122997

theorem triangle_is_obtuse
  (α : ℝ)
  (h1 : α > 0 ∧ α < π)
  (h2 : Real.sin α + Real.cos α = 2 / 3) :
  ∃ β γ, β > 0 ∧ β < π ∧ γ > 0 ∧ γ < π ∧ β + γ + α = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end triangle_is_obtuse_l1229_122997


namespace circle_radius_triple_area_l1229_122926

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end circle_radius_triple_area_l1229_122926


namespace solve_quartic_equation_l1229_122954

theorem solve_quartic_equation (a b c : ℤ) (x : ℤ) : 
  x^4 + a * x^2 + b * x + c = 0 :=
sorry

end solve_quartic_equation_l1229_122954


namespace total_expenditure_correct_l1229_122977

def length : ℝ := 50
def width : ℝ := 30
def cost_per_square_meter : ℝ := 100

def area (L W : ℝ) : ℝ := L * W
def total_expenditure (A C : ℝ) : ℝ := A * C

theorem total_expenditure_correct :
  total_expenditure (area length width) cost_per_square_meter = 150000 := by
  sorry

end total_expenditure_correct_l1229_122977


namespace trig_values_same_terminal_side_l1229_122999

-- Statement: The trigonometric function values of angles with the same terminal side are equal.
theorem trig_values_same_terminal_side (θ₁ θ₂ : ℝ) (h : ∃ k : ℤ, θ₂ = θ₁ + 2 * k * π) :
  (∀ f : ℝ -> ℝ, f θ₁ = f θ₂) :=
by
  sorry

end trig_values_same_terminal_side_l1229_122999


namespace sin_2012_equals_neg_sin_32_l1229_122922

theorem sin_2012_equals_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = -Real.sin (32 * Real.pi / 180) := by
  sorry

end sin_2012_equals_neg_sin_32_l1229_122922


namespace minimum_time_for_xiang_qing_fried_eggs_l1229_122981

-- Define the time taken for each individual step
def wash_scallions_time : ℕ := 1
def beat_eggs_time : ℕ := 1 / 2
def mix_egg_scallions_time : ℕ := 1
def wash_pan_time : ℕ := 1 / 2
def heat_pan_time : ℕ := 1 / 2
def heat_oil_time : ℕ := 1 / 2
def cook_dish_time : ℕ := 2

-- Define the total minimum time required
def minimum_time : ℕ := 5

-- The main theorem stating that the minimum time required is 5 minutes
theorem minimum_time_for_xiang_qing_fried_eggs :
  wash_scallions_time + beat_eggs_time + mix_egg_scallions_time + wash_pan_time + heat_pan_time + heat_oil_time + cook_dish_time = minimum_time := 
by sorry

end minimum_time_for_xiang_qing_fried_eggs_l1229_122981


namespace solve_for_x_l1229_122975

theorem solve_for_x (x y : ℕ) (h₁ : 9 ^ y = 3 ^ x) (h₂ : y = 6) : x = 12 :=
by
  sorry

end solve_for_x_l1229_122975


namespace wine_consumption_correct_l1229_122905

-- Definitions based on conditions
def drank_after_first_pound : ℚ := 1
def drank_after_second_pound : ℚ := 1
def drank_after_third_pound : ℚ := 1 / 2
def drank_after_fourth_pound : ℚ := 1 / 4
def drank_after_fifth_pound : ℚ := 1 / 8
def drank_after_sixth_pound : ℚ := 1 / 16

-- Total wine consumption
def total_wine_consumption : ℚ :=
  drank_after_first_pound + drank_after_second_pound +
  drank_after_third_pound + drank_after_fourth_pound +
  drank_after_fifth_pound + drank_after_sixth_pound

-- Theorem statement
theorem wine_consumption_correct :
  total_wine_consumption = 47 / 16 :=
by
  sorry

end wine_consumption_correct_l1229_122905


namespace prime_range_for_integer_roots_l1229_122903

theorem prime_range_for_integer_roots (p : ℕ) (h_prime : Prime p) 
  (h_int_roots : ∃ (a b : ℤ), a + b = -p ∧ a * b = -300 * p) : 
  1 < p ∧ p ≤ 11 :=
sorry

end prime_range_for_integer_roots_l1229_122903


namespace rectangle_perimeter_l1229_122994

-- Definitions based on the conditions
def length : ℕ := 15
def width : ℕ := 8

-- Definition of the perimeter function
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Statement of the theorem we need to prove
theorem rectangle_perimeter : perimeter length width = 46 := by
  sorry

end rectangle_perimeter_l1229_122994


namespace hcf_of_36_and_x_is_12_l1229_122925

theorem hcf_of_36_and_x_is_12 (x : ℕ) (h : Nat.gcd 36 x = 12) : x = 48 :=
sorry

end hcf_of_36_and_x_is_12_l1229_122925


namespace sum_of_first_half_of_numbers_l1229_122904

theorem sum_of_first_half_of_numbers 
  (avg_total : ℝ) 
  (total_count : ℕ) 
  (avg_second_half : ℝ) 
  (sum_total : ℝ)
  (sum_second_half : ℝ)
  (sum_first_half : ℝ) 
  (h1 : total_count = 8)
  (h2 : avg_total = 43.1)
  (h3 : avg_second_half = 46.6)
  (h4 : sum_total = avg_total * total_count)
  (h5 : sum_second_half = 4 * avg_second_half)
  (h6 : sum_first_half = sum_total - sum_second_half)
  :
  sum_first_half = 158.4 := 
sorry

end sum_of_first_half_of_numbers_l1229_122904


namespace find_a_20_l1229_122913

-- Arithmetic sequence definition and known conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

-- Conditions
def condition1 : Prop := a 1 + a 2 + a 3 = 6
def condition2 : Prop := a 5 = 8

-- The main statement to prove
theorem find_a_20 (h_arith : arithmetic_sequence a) (h_cond1 : condition1 a) (h_cond2 : condition2 a) : 
  a 20 = 38 := by
  sorry

end find_a_20_l1229_122913


namespace find_a_b_sum_l1229_122934

theorem find_a_b_sum (a b : ℕ) (h : a^2 - b^4 = 2009) : a + b = 47 :=
sorry

end find_a_b_sum_l1229_122934


namespace smallest_four_digit_multiple_of_17_is_1013_l1229_122917

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l1229_122917


namespace inequality_solution_l1229_122945

theorem inequality_solution (m : ℝ) (h : m < -1) :
  (if m = -3 then
    {x : ℝ | x > 1} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if -3 < m ∧ m < -1 then
    ({x : ℝ | x < m / (m + 3)} ∪ {x : ℝ | x > 1}) =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else if m < -3 then
    {x : ℝ | 1 < x ∧ x < m / (m + 3)} =
    {x : ℝ | ((m + 3) * x^2 - (2 * m + 3) * x + m > 0)}
  else
    False) :=
by
  sorry

end inequality_solution_l1229_122945


namespace fractions_sum_l1229_122923

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end fractions_sum_l1229_122923


namespace basketball_weight_l1229_122983

variable {b c : ℝ}

theorem basketball_weight (h1 : 8 * b = 4 * c) (h2 : 3 * c = 120) : b = 20 :=
by
  -- Proof omitted
  sorry

end basketball_weight_l1229_122983


namespace find_k_min_value_quadratic_zero_l1229_122993

theorem find_k_min_value_quadratic_zero (x y k : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 10 * x - 6 * y + 9 = 0) ↔ k = 1 :=
by
  sorry

end find_k_min_value_quadratic_zero_l1229_122993


namespace cube_div_identity_l1229_122979

theorem cube_div_identity (a b : ℕ) (h1 : a = 6) (h2 : b = 3) : 
  (a^3 - b^3) / (a^2 + a * b + b^2) = 3 :=
by {
  sorry
}

end cube_div_identity_l1229_122979


namespace find_h_l1229_122949

def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

theorem find_h : ∃ a h k, (h = -3 / 2) ∧ (f x = a * (x - h)^2 + k) :=
by
  -- Proof steps would go here
  sorry

end find_h_l1229_122949


namespace initial_men_count_l1229_122935

theorem initial_men_count (x : ℕ) (h : x * 25 = 15 * 60) : x = 36 :=
by
  sorry

end initial_men_count_l1229_122935


namespace minimum_value_l1229_122967

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem minimum_value (a b c d : ℝ) (h1 : a < (2 / 3) * b) 
  (h2 : ∀ x, 3 * a * x^2 + 2 * b * x + c ≥ 0) : 
  ∃ (x : ℝ), ∀ c, 2 * b - 3 * a ≠ 0 → (c = (b^2 / 3 / a)) → (c / (2 * b - 3 * a) ≥ 1) :=
by
  sorry

end minimum_value_l1229_122967


namespace find_y_arithmetic_mean_l1229_122989

theorem find_y_arithmetic_mean (y : ℝ) 
  (h : (8 + 15 + 20 + 7 + y + 9) / 6 = 12) : 
  y = 13 :=
sorry

end find_y_arithmetic_mean_l1229_122989


namespace problem1_problem2_l1229_122900

-- We define a point P(x, y) on the circle x^2 + y^2 = 2y.
variables {x y a : ℝ}

-- Condition for the point P to be on the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

-- Definition for 2x + y range
def range_2x_plus_y (x y : ℝ) : Prop := - Real.sqrt 5 + 1 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 5 + 1

-- Definition for the range of a given x + y + a ≥ 0
def range_a (x y a : ℝ) : Prop := x + y + a ≥ 0 → a ≥ Real.sqrt 2 - 1

-- Main statements to prove
theorem problem1 (hx : on_circle x y) : range_2x_plus_y x y := sorry

theorem problem2 (hx : on_circle x y) (h : ∀ θ, x = Real.cos θ ∧ y = 1 + Real.sin θ) : range_a x y a := sorry

end problem1_problem2_l1229_122900


namespace factor_x_squared_minus_169_l1229_122916

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end factor_x_squared_minus_169_l1229_122916


namespace total_tickets_l1229_122961

theorem total_tickets (A C total_tickets total_cost : ℕ) 
  (adult_ticket_cost : ℕ := 8) (child_ticket_cost : ℕ := 5) 
  (total_cost_paid : ℕ := 201) (child_tickets_count : ℕ := 21) 
  (ticket_cost_eqn : 8 * A + 5 * 21 = 201) 
  (adult_tickets_count : A = total_cost_paid - (child_ticket_cost * child_tickets_count) / adult_ticket_cost) :
  total_tickets = A + child_tickets_count :=
sorry

end total_tickets_l1229_122961


namespace find_constants_and_intervals_l1229_122991

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x^2 - 2 * x
def f' (x : ℝ) (a b : ℝ) := 3 * a * x^2 + 2 * b * x - 2

theorem find_constants_and_intervals :
  (f' (1 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (f' (-2 : ℝ) (1/3 : ℝ) (1/2 : ℝ) = 0) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f' x (1/3 : ℝ) (1/2 : ℝ) < 0 ↔ -2 < x ∧ x < 1) :=
by {
  sorry
}

end find_constants_and_intervals_l1229_122991


namespace sqrt_floor_square_l1229_122902

theorem sqrt_floor_square (h1 : 7 < Real.sqrt 50) (h2 : Real.sqrt 50 < 8) :
  Int.floor (Real.sqrt 50) ^ 2 = 49 := by
  sorry

end sqrt_floor_square_l1229_122902


namespace price_reduction_l1229_122985

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end price_reduction_l1229_122985


namespace linda_savings_l1229_122976

theorem linda_savings (S : ℝ) (h1 : 1 / 4 * S = 150) : S = 600 :=
sorry

end linda_savings_l1229_122976


namespace number_of_sheep_l1229_122966

-- Define the conditions as given in the problem
variables (S H : ℕ)
axiom ratio_condition : S * 7 = H * 3
axiom food_condition : H * 230 = 12880

-- The theorem to prove
theorem number_of_sheep : S = 24 :=
by sorry

end number_of_sheep_l1229_122966


namespace determine_p_in_terms_of_q_l1229_122943

variable {p q : ℝ}

-- Given the condition in the problem
def log_condition (p q : ℝ) : Prop :=
  Real.log p + 2 * Real.log q = Real.log (2 * p + q)

-- The goal is to prove that under this condition, the following holds
theorem determine_p_in_terms_of_q (h : log_condition p q) :
  p = q / (q^2 - 2) :=
sorry

end determine_p_in_terms_of_q_l1229_122943


namespace value_of_x_y_l1229_122995

noncomputable def real_ln : ℝ → ℝ := sorry

theorem value_of_x_y (x y : ℝ) (h : 3 * x - y ≤ real_ln (x + 2 * y - 3) + real_ln (2 * x - 3 * y + 5)) :
  x + y = 16 / 7 :=
sorry

end value_of_x_y_l1229_122995


namespace ratio_of_A_to_B_l1229_122924

-- Definitions of the conditions.
def amount_A : ℕ := 200
def total_amount : ℕ := 600
def amount_B : ℕ := total_amount - amount_A

-- The proof statement.
theorem ratio_of_A_to_B :
  amount_A / amount_B = 1 / 2 := 
sorry

end ratio_of_A_to_B_l1229_122924


namespace white_ring_weight_l1229_122927

def weight_of_orange_ring : ℝ := 0.08
def weight_of_purple_ring : ℝ := 0.33
def total_weight_of_rings : ℝ := 0.83

def weight_of_white_ring (total : ℝ) (orange : ℝ) (purple : ℝ) : ℝ :=
  total - (orange + purple)

theorem white_ring_weight :
  weight_of_white_ring total_weight_of_rings weight_of_orange_ring weight_of_purple_ring = 0.42 :=
by
  sorry

end white_ring_weight_l1229_122927


namespace simplify_expression_l1229_122960

theorem simplify_expression (x y : ℤ) :
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y :=
by sorry

end simplify_expression_l1229_122960


namespace divide_subtract_result_l1229_122952

theorem divide_subtract_result (x : ℕ) (h : (x - 26) / 2 = 37) : 48 - (x / 4) = 23 := 
by
  sorry

end divide_subtract_result_l1229_122952


namespace PS_length_correct_l1229_122909

variable {Triangle : Type}

noncomputable def PR := 15

noncomputable def PS_length (PS SR : ℝ) (PR : ℝ) : Prop :=
  PS + SR = PR ∧ (PS / SR) = (3 / 4)

theorem PS_length_correct : 
  ∃ PS SR : ℝ, PS_length PS SR PR ∧ PS = (45 / 7) :=
sorry

end PS_length_correct_l1229_122909


namespace coins_remainder_l1229_122919

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l1229_122919


namespace range_of_2a_minus_b_l1229_122956

variable (a b : ℝ)
variable (h1 : -2 < a ∧ a < 2)
variable (h2 : 2 < b ∧ b < 3)

theorem range_of_2a_minus_b (a b : ℝ) (h1 : -2 < a ∧ a < 2) (h2 : 2 < b ∧ b < 3) :
  -7 < 2 * a - b ∧ 2 * a - b < 2 := sorry

end range_of_2a_minus_b_l1229_122956


namespace problem1_problem2_problem3_problem4_l1229_122998

-- Problem 1
theorem problem1 : (- (3 : ℝ) / 7) + (1 / 5) + (2 / 7) + (- (6 / 5)) = - (8 / 7) :=
by
  sorry

-- Problem 2
theorem problem2 : -(-1) + 3^2 / (1 - 4) * 2 = -5 :=
by
  sorry

-- Problem 3
theorem problem3 :  (-(1 / 6))^2 / ((1 / 2 - 1 / 3)^2) / (abs (-6))^2 = 1 / 36 :=
by
  sorry

-- Problem 4
theorem problem4 : (-1) ^ 1000 - 2.45 * 8 + 2.55 * (-8) = -39 :=
by
  sorry

end problem1_problem2_problem3_problem4_l1229_122998


namespace distance_from_edge_l1229_122964

theorem distance_from_edge (wall_width picture_width x : ℕ) (h_wall : wall_width = 24) (h_picture : picture_width = 4) (h_centered : x + picture_width + x = wall_width) : x = 10 := by
  -- Proof is omitted
  sorry

end distance_from_edge_l1229_122964


namespace number_of_ways_to_place_balls_l1229_122942

theorem number_of_ways_to_place_balls : 
  let balls := 3 
  let boxes := 4 
  (boxes^balls = 64) :=
by
  sorry

end number_of_ways_to_place_balls_l1229_122942


namespace smallest_positive_value_of_a_minus_b_l1229_122959

theorem smallest_positive_value_of_a_minus_b :
  ∃ (a b : ℤ), 17 * a + 6 * b = 13 ∧ a - b = 17 :=
by
  sorry

end smallest_positive_value_of_a_minus_b_l1229_122959


namespace inv_prop_func_point_l1229_122948

theorem inv_prop_func_point {k : ℝ} :
  (∃ y x : ℝ, y = k / x ∧ (x = 2 ∧ y = -1)) → k = -2 :=
by
  intro h
  -- Proof would go here
  sorry

end inv_prop_func_point_l1229_122948


namespace monthly_salary_l1229_122936

theorem monthly_salary (S : ℝ) (E : ℝ) 
  (h1 : S - 1.20 * E = 220)
  (h2 : E = 0.80 * S) :
  S = 5500 :=
by
  sorry

end monthly_salary_l1229_122936
