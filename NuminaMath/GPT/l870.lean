import Mathlib

namespace lions_deers_15_minutes_l870_87081

theorem lions_deers_15_minutes :
  ∀ (n : ℕ), (15 * n = 15 * 15 → n = 15 → ∀ t, t = 15) := by
  sorry

end lions_deers_15_minutes_l870_87081


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_l870_87065

theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ) (h_d_nonzero : d ≠ 0)
  (h_arith : ∀ n, a_n = a_n 0 + n * d)
  (h_S9 : S 9 = 90)
  (h_geom : ∃ (a1 a2 a4 : ℕ), a2^2 = a1 * a4)
  (h_common_diff : d = a_n 1 - a_n 0)
  : ∀ n, a_n = 2 * n  := 
sorry

theorem sum_of_first_n_terms
  (b_n : ℕ → ℕ)
  (T : ℕ → ℕ)
  (a_n : ℕ → ℕ) 
  (h_b_def : ∀ n, b_n = 1 / (a_n n * a_n (n+1)))
  (h_a_form : ∀ n, a_n = 2 * n)
  : ∀ n, T n = n / (4 * n + 4) :=
sorry

end arithmetic_sequence_general_term_sum_of_first_n_terms_l870_87065


namespace slower_train_speed_l870_87055

theorem slower_train_speed
  (v : ℝ)  -- The speed of the slower train
  (faster_train_speed : ℝ := 46)  -- The speed of the faster train
  (train_length : ℝ := 37.5)  -- The length of each train in meters
  (time_to_pass : ℝ := 27)  -- Time taken to pass in seconds
  (kms_to_ms : ℝ := 1000 / 3600)  -- Conversion factor from km/hr to m/s
  (relative_distance : ℝ := 2 * train_length)  -- Distance covered when passing

  (h : relative_distance = (faster_train_speed - v) * kms_to_ms * time_to_pass) :
  v = 36 :=
by
  -- The proof should be placed here
  sorry

end slower_train_speed_l870_87055


namespace total_games_is_seven_l870_87077

def total_football_games (games_missed : ℕ) (games_attended : ℕ) : ℕ :=
  games_missed + games_attended

theorem total_games_is_seven : total_football_games 4 3 = 7 := 
by
  sorry

end total_games_is_seven_l870_87077


namespace probability_of_three_faces_painted_l870_87039

def total_cubes : Nat := 27
def corner_cubes_painted (total : Nat) : Nat := 8
def probability_of_corner_cube (corner : Nat) (total : Nat) : Rat := corner / total

theorem probability_of_three_faces_painted :
    probability_of_corner_cube (corner_cubes_painted total_cubes) total_cubes = 8 / 27 := 
by 
  sorry

end probability_of_three_faces_painted_l870_87039


namespace least_addition_l870_87056

theorem least_addition (a b n : ℕ) (h_a : Nat.Prime a) (h_b : Nat.Prime b) (h_a_val : a = 23) (h_b_val : b = 29) (h_n : n = 1056) :
  ∃ m : ℕ, (m + n) % (a * b) = 0 ∧ m = 278 :=
by
  sorry

end least_addition_l870_87056


namespace z_share_in_profit_l870_87016

noncomputable def investment_share (investment : ℕ) (months : ℕ) : ℕ := investment * months

noncomputable def profit_share (profit : ℕ) (share : ℚ) : ℚ := (profit : ℚ) * share

theorem z_share_in_profit 
  (investment_X : ℕ := 36000)
  (investment_Y : ℕ := 42000)
  (investment_Z : ℕ := 48000)
  (months_X : ℕ := 12)
  (months_Y : ℕ := 12)
  (months_Z : ℕ := 8)
  (total_profit : ℕ := 14300) :
  profit_share total_profit (investment_share investment_Z months_Z / 
            (investment_share investment_X months_X + 
             investment_share investment_Y months_Y + 
             investment_share investment_Z months_Z)) = 2600 := 
by
  sorry

end z_share_in_profit_l870_87016


namespace arithmetic_sequence_properties_l870_87059

noncomputable def arithmeticSeq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ d : ℕ) (n : ℕ) (h1 : d = 2)
  (h2 : (a₁ + d)^2 = a₁ * (a₁ + 3 * d)) :
  (a₁ = 2) ∧ (∃ S, S = (n * (2 * a₁ + (n - 1) * d)) / 2 ∧ S = n^2 + n) :=
by 
  sorry

end arithmetic_sequence_properties_l870_87059


namespace men_left_hostel_l870_87086

-- Definitions based on the conditions given
def initialMen : ℕ := 250
def initialDays : ℕ := 28
def remainingDays : ℕ := 35

-- The theorem we need to prove
theorem men_left_hostel (x : ℕ) (h : initialMen * initialDays = (initialMen - x) * remainingDays) : x = 50 :=
by
  sorry

end men_left_hostel_l870_87086


namespace find_m_value_l870_87045

theorem find_m_value 
  (h : ∀ x y m : ℝ, 2*x + y + m = 0 → (1 : ℝ)*x + (-2 : ℝ)*y + 0 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ m : ℝ, m = 0 :=
sorry

end find_m_value_l870_87045


namespace cube_sum_eq_2702_l870_87063

noncomputable def x : ℝ := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
noncomputable def y : ℝ := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)

theorem cube_sum_eq_2702 : x^3 + y^3 = 2702 :=
by
  sorry

end cube_sum_eq_2702_l870_87063


namespace isosceles_triangle_relation_range_l870_87089

-- Definitions of the problem conditions and goal
variables (x y : ℝ)

-- Given conditions
def isosceles_triangle (x y : ℝ) :=
  x + x + y = 10

-- Prove the relationship and range 
theorem isosceles_triangle_relation_range (h : isosceles_triangle x y) :
  y = 10 - 2 * x ∧ (5 / 2 < x ∧ x < 5) :=
  sorry

end isosceles_triangle_relation_range_l870_87089


namespace length_of_AB_l870_87072

theorem length_of_AB
  (AP PB AQ QB : ℝ) 
  (h_ratioP : 5 * AP = 3 * PB)
  (h_ratioQ : 3 * AQ = 2 * QB)
  (h_PQ : AQ = AP + 3 ∧ QB = PB - 3)
  (h_PQ_length : AQ - AP = 3)
  : AP + PB = 120 :=
by {
  sorry
}

end length_of_AB_l870_87072


namespace net_cut_square_l870_87041

-- Define the dimensions of the parallelepiped
structure Parallelepiped :=
  (length width height : ℕ)
  (length_eq : length = 2)
  (width_eq : width = 1)
  (height_eq : height = 1)

-- Define the net of the parallelepiped
structure NetConfig :=
  (total_squares : ℕ)
  (cut_squares : ℕ)
  (remaining_squares : ℕ)
  (cut_positions : Fin 5) -- Five possible cut positions

-- The remaining net has 9 squares after cutting one square
theorem net_cut_square (p : Parallelepiped) : 
  ∃ net : NetConfig, net.total_squares = 10 ∧ net.cut_squares = 1 ∧ net.remaining_squares = 9 ∧ net.cut_positions = 5 := 
sorry

end net_cut_square_l870_87041


namespace problem_l870_87057

variables {a b c d : ℝ}

theorem problem (h1 : c + d = 14 * a) (h2 : c * d = 15 * b) (h3 : a + b = 14 * c) (h4 : a * b = 15 * d) (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) :
  a + b + c + d = 3150 := sorry

end problem_l870_87057


namespace pizza_cost_per_slice_correct_l870_87028

noncomputable def pizza_cost_per_slice : ℝ :=
  let base_pizza_cost := 10.00
  let first_topping_cost := 2.00
  let next_two_toppings_cost := 2.00
  let remaining_toppings_cost := 2.00
  let total_cost := base_pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  total_cost / 8

theorem pizza_cost_per_slice_correct :
  pizza_cost_per_slice = 2.00 :=
by
  unfold pizza_cost_per_slice
  sorry

end pizza_cost_per_slice_correct_l870_87028


namespace intersection_of_sets_l870_87025

def set_M : Set ℝ := { x | x >= 2 }
def set_N : Set ℝ := { x | -1 <= x ∧ x <= 3 }
def set_intersection : Set ℝ := { x | 2 <= x ∧ x <= 3 }

theorem intersection_of_sets : (set_M ∩ set_N) = set_intersection := by
  sorry

end intersection_of_sets_l870_87025


namespace total_weight_of_rings_l870_87069

-- Define the weights of the rings
def weight_orange : Real := 0.08
def weight_purple : Real := 0.33
def weight_white : Real := 0.42
def weight_blue : Real := 0.59
def weight_red : Real := 0.24
def weight_green : Real := 0.16

-- Define the total weight of the rings
def total_weight : Real :=
  weight_orange + weight_purple + weight_white + weight_blue + weight_red + weight_green

-- The task is to prove that the total weight equals 1.82
theorem total_weight_of_rings : total_weight = 1.82 := 
  by
    sorry

end total_weight_of_rings_l870_87069


namespace savings_wednesday_l870_87000

variable (m t s w : ℕ)

theorem savings_wednesday :
  m = 15 → t = 28 → s = 28 → 2 * s = 56 → 
  m + t + w = 56 → w = 13 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end savings_wednesday_l870_87000


namespace basketball_weight_l870_87021

-- Definitions based on the given conditions
variables (b c : ℕ) -- weights of basketball and bicycle in pounds

-- Condition 1: Nine basketballs weigh the same as six bicycles
axiom condition1 : 9 * b = 6 * c

-- Condition 2: Four bicycles weigh a total of 120 pounds
axiom condition2 : 4 * c = 120

-- The proof statement we need to prove
theorem basketball_weight : b = 20 :=
by
  sorry

end basketball_weight_l870_87021


namespace megatek_manufacturing_percentage_l870_87015

theorem megatek_manufacturing_percentage (angle_manufacturing : ℝ) (full_circle : ℝ) 
  (h1 : angle_manufacturing = 162) (h2 : full_circle = 360) :
  (angle_manufacturing / full_circle) * 100 = 45 :=
by
  sorry

end megatek_manufacturing_percentage_l870_87015


namespace isabella_stops_l870_87085

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_stops (P : ℕ → ℚ) (h : ∀ n, P n = 1 / (n * (n + 1))) : 
  ∃ n : ℕ, n = 55 ∧ P n < 1 / 3000 :=
by {
  sorry
}

end isabella_stops_l870_87085


namespace tan_150_degree_is_correct_l870_87094

noncomputable def tan_150_degree_is_negative_sqrt_3_div_3 : Prop :=
  let theta := Real.pi * 150 / 180
  let ref_angle := Real.pi * 30 / 180
  let cos_150 := -Real.cos ref_angle
  let sin_150 := Real.sin ref_angle
  Real.tan theta = -Real.sqrt 3 / 3

theorem tan_150_degree_is_correct :
  tan_150_degree_is_negative_sqrt_3_div_3 :=
by
  sorry

end tan_150_degree_is_correct_l870_87094


namespace solve_modified_system_l870_87043

theorem solve_modified_system (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : 4 * a1 + 6 * b1 = c1) 
  (h2 : 4 * a2 + 6 * b2 = c2) :
  (4 * a1 * 5 + 3 * b1 * 10 = 5 * c1) ∧ (4 * a2 * 5 + 3 * b2 * 10 = 5 * c2) :=
by
  sorry

end solve_modified_system_l870_87043


namespace assignment_increment_l870_87090

theorem assignment_increment (M : ℤ) : (M = M + 3) → false :=
by
  sorry

end assignment_increment_l870_87090


namespace max_intersections_l870_87082

theorem max_intersections (n1 n2 k : ℕ) 
  (h1 : n1 ≤ n2)
  (h2 : k ≤ n1) : 
  ∃ max_intersections : ℕ, 
  max_intersections = k * n2 :=
by
  sorry

end max_intersections_l870_87082


namespace sum_youngest_oldest_l870_87040

variables {a1 a2 a3 a4 a5 : ℕ}

def mean_age (x y z u v : ℕ) : ℕ := (x + y + z + u + v) / 5
def median_age (x y z u v : ℕ) : ℕ := z

theorem sum_youngest_oldest
  (h_mean: mean_age a1 a2 a3 a4 a5 = 10) 
  (h_median: median_age a1 a2 a3 a4 a5 = 7)
  (h_sorted: a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) :
  a1 + a5 = 23 :=
sorry

end sum_youngest_oldest_l870_87040


namespace ratio_a_div_8_to_b_div_7_l870_87047

theorem ratio_a_div_8_to_b_div_7 (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 8) / (b / 7) = 1 :=
sorry

end ratio_a_div_8_to_b_div_7_l870_87047


namespace simplify_expression_l870_87022

theorem simplify_expression : 
  (Real.sqrt 2 * 2^(1/2) * 2) + (18 / 3 * 2) - (8^(1/2) * 4) = 16 - 8 * Real.sqrt 2 :=
by 
  sorry  -- proof omitted

end simplify_expression_l870_87022


namespace parallelogram_angle_l870_87050

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end parallelogram_angle_l870_87050


namespace value_of_x_l870_87051

theorem value_of_x (x : ℝ) (h : 4 * x + 5 * x + x + 2 * x = 360) : x = 30 := 
by
  sorry

end value_of_x_l870_87051


namespace moles_of_NaCl_formed_l870_87003

-- Define the conditions
def moles_NaOH : ℕ := 3
def moles_HCl : ℕ := 3

-- Define the balanced chemical equation as a relation
def reaction (NaOH HCl NaCl H2O : ℕ) : Prop :=
  NaOH = HCl ∧ HCl = NaCl ∧ H2O = NaCl

-- Define the proof problem
theorem moles_of_NaCl_formed :
  ∀ (NaOH HCl NaCl H2O : ℕ), NaOH = 3 → HCl = 3 → reaction NaOH HCl NaCl H2O → NaCl = 3 :=
by
  intros NaOH HCl NaCl H2O hNa hHCl hReaction
  sorry

end moles_of_NaCl_formed_l870_87003


namespace hexagon_rotation_angle_l870_87080

theorem hexagon_rotation_angle (θ : ℕ) : θ = 90 → ¬ ∃ k, k * 60 = θ ∨ θ = 360 :=
by
  sorry

end hexagon_rotation_angle_l870_87080


namespace bird_count_l870_87035

theorem bird_count (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : parakeets_per_cage = 7) 
  (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) : 
  total_birds = 72 := 
  by
  sorry

end bird_count_l870_87035


namespace no_solution_inequality_l870_87079

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 :=
by
  intro h
  sorry

end no_solution_inequality_l870_87079


namespace allocation_schemes_count_l870_87017

open BigOperators -- For working with big operator notations
open Finset -- For working with finite sets
open Nat -- For natural number operations

-- Define the number of students and dormitories
def num_students : ℕ := 7
def num_dormitories : ℕ := 2

-- Define the constraint for minimum students in each dormitory
def min_students_in_dormitory : ℕ := 2

-- Compute the number of ways to allocate students given the conditions
noncomputable def number_of_allocation_schemes : ℕ :=
  (Nat.choose num_students 3) * (Nat.choose 4 2) + (Nat.choose num_students 2) * (Nat.choose 5 2)

-- The theorem stating the total number of allocation schemes
theorem allocation_schemes_count :
  number_of_allocation_schemes = 112 :=
  by sorry

end allocation_schemes_count_l870_87017


namespace solve_xy_eq_yx_l870_87001

theorem solve_xy_eq_yx (x y : ℕ) (hxy : x ≠ y) : x^y = y^x ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_xy_eq_yx_l870_87001


namespace max_value_ab_l870_87006

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l870_87006


namespace isosceles_triangle_perimeter_l870_87042

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6 ∨ a = 7) (h₂ : b = 6 ∨ b = 7) (h₃ : a ≠ b) :
  (2 * a + b = 19) ∨ (2 * b + a = 20) :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_perimeter_l870_87042


namespace complement_supplement_measure_l870_87053

theorem complement_supplement_measure (x : ℝ) (h : 180 - x = 3 * (90 - x)) : 
  (180 - x = 135) ∧ (90 - x = 45) :=
by {
  sorry
}

end complement_supplement_measure_l870_87053


namespace sqrt_calculation_l870_87008

theorem sqrt_calculation : Real.sqrt (36 * Real.sqrt 16) = 12 := 
by
  sorry

end sqrt_calculation_l870_87008


namespace problem_statement_l870_87087

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f (x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom initial_condition : f 2 = 3

theorem problem_statement : f 2006 + f 2007 = 3 :=
by
  sorry

end problem_statement_l870_87087


namespace fraction_sum_l870_87062

theorem fraction_sum :
  (1 / 4 : ℚ) + (2 / 9) + (3 / 6) = 35 / 36 := 
sorry

end fraction_sum_l870_87062


namespace minimum_value_of_expression_l870_87019

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_eq : x + y + z = 5) :
  (9 / x + 4 / y + 25 / z) ≥ 20 :=
sorry

end minimum_value_of_expression_l870_87019


namespace necessary_condition_inequality_l870_87044

theorem necessary_condition_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 := 
sorry

end necessary_condition_inequality_l870_87044


namespace problem1_problem2_l870_87083

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- First problem: A ∩ B
theorem problem1 (a : ℝ) (ha : a = 4) : A a ∩ B = {x | 6 < x ∧ x ≤ 7} :=
by sorry

-- Second problem: A ∪ B = B
theorem problem2 (a : ℝ) : (A a ∪ B = B) ↔ (a < -4 ∨ a > 5) :=
by sorry

end problem1_problem2_l870_87083


namespace set_M_listed_correctly_l870_87026

theorem set_M_listed_correctly :
  {a : ℕ+ | ∃ (n : ℤ), 4 = n * (1 - a)} = {2, 3, 4} := by
sorry

end set_M_listed_correctly_l870_87026


namespace fraction_identity_l870_87067

theorem fraction_identity :
  ( (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) = (432 / 1105) ) :=
by
  sorry

end fraction_identity_l870_87067


namespace solve_system_of_equations_l870_87011

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - y = 2 ∧ 3 * x + y = 4 ∧ x = 1.5 ∧ y = -0.5 :=
by
  sorry

end solve_system_of_equations_l870_87011


namespace third_red_yellow_flash_is_60_l870_87029

-- Define the flashing intervals for red, yellow, and green lights
def red_interval : Nat := 3
def yellow_interval : Nat := 4
def green_interval : Nat := 8

-- Define the function for finding the time of the third occurrence of only red and yellow lights flashing together
def third_red_yellow_flash : Nat :=
  let lcm_red_yellow := Nat.lcm red_interval yellow_interval
  let times := (List.range (100)).filter (fun t => t % lcm_red_yellow = 0 ∧ t % green_interval ≠ 0)
  times[2] -- Getting the third occurrence

-- Prove that the third occurrence time is 60 seconds
theorem third_red_yellow_flash_is_60 :
  third_red_yellow_flash = 60 :=
  by
    -- Proof goes here
    sorry

end third_red_yellow_flash_is_60_l870_87029


namespace distance_travelled_l870_87024

def speed : ℕ := 3 -- speed in feet per second
def time : ℕ := 3600 -- time in seconds (1 hour)

theorem distance_travelled : speed * time = 10800 := by
  sorry

end distance_travelled_l870_87024


namespace real_y_values_l870_87032

theorem real_y_values (x : ℝ) :
  (∃ y : ℝ, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23 / 9 ∨ x ≥ 5 / 3) :=
by
  sorry

end real_y_values_l870_87032


namespace min_colors_shapes_l870_87070

def representable_centers (C S : Nat) : Nat :=
  C + (C * (C - 1)) / 2 + S + S * (S - 1)

theorem min_colors_shapes (C S : Nat) :
  ∀ (C S : Nat), (C + (C * (C - 1)) / 2 + S + S * (S - 1)) ≥ 12 → (C, S) = (3, 3) :=
sorry

end min_colors_shapes_l870_87070


namespace three_circles_area_less_than_total_radius_squared_l870_87046

theorem three_circles_area_less_than_total_radius_squared
    (x y z R : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : R > 0)
    (descartes_theorem : ( (1/x + 1/y + 1/z - 1/R)^2 = 2 * ( (1/x)^2 + (1/y)^2 + (1/z)^2 + (1/R)^2 ) )) :
    x^2 + y^2 + z^2 < 4 * R^2 := 
sorry

end three_circles_area_less_than_total_radius_squared_l870_87046


namespace sum_of_areas_is_correct_l870_87052

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct_l870_87052


namespace find_omega_l870_87076

theorem find_omega 
  (w : ℝ) 
  (h₁ : 0 < w)
  (h₂ : (π / w) = (π / 2)) : w = 2 :=
by
  sorry

end find_omega_l870_87076


namespace find_six_digit_number_l870_87096

theorem find_six_digit_number (a b c d e f : ℕ) (N : ℕ) :
  a = 1 ∧ f = 7 ∧
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
  (f - 1) * 10^5 + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e = 5 * N →
  N = 142857 :=
by
  sorry

end find_six_digit_number_l870_87096


namespace students_taking_neither_l870_87018

-- Defining given conditions as Lean definitions
def total_students : ℕ := 70
def students_math : ℕ := 42
def students_physics : ℕ := 35
def students_chemistry : ℕ := 25
def students_math_physics : ℕ := 18
def students_math_chemistry : ℕ := 10
def students_physics_chemistry : ℕ := 8
def students_all_three : ℕ := 5

-- Define the problem to prove
theorem students_taking_neither : total_students
  - (students_math - students_math_physics - students_math_chemistry + students_all_three
    + students_physics - students_math_physics - students_physics_chemistry + students_all_three
    + students_chemistry - students_math_chemistry - students_physics_chemistry + students_all_three
    + students_math_physics - students_all_three
    + students_math_chemistry - students_all_three
    + students_physics_chemistry - students_all_three
    + students_all_three) = 0 := by
  sorry

end students_taking_neither_l870_87018


namespace values_of_a_and_b_solution_set_inequality_l870_87064

-- Part (I)
theorem values_of_a_and_b (a b : ℝ) (h : ∀ x, -1 < x ∧ x < 1 → x^2 - a * x - x + b < 0) :
  a = -1 ∧ b = -1 := sorry

-- Part (II)
theorem solution_set_inequality (a : ℝ) (h : a = b) :
  (∀ x, x^2 - a * x - x + a < 0 → (x = 1 → false) 
      ∧ (0 < 1 - a → (x = 1 → false))
      ∧ (1 < - a → (x = 1 → false))) := sorry

end values_of_a_and_b_solution_set_inequality_l870_87064


namespace square_nonneg_of_nonneg_l870_87012

theorem square_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x^2 :=
sorry

end square_nonneg_of_nonneg_l870_87012


namespace corn_cobs_each_row_l870_87031

theorem corn_cobs_each_row (x : ℕ) 
  (h1 : 13 * x + 16 * x = 116) : 
  x = 4 :=
by sorry

end corn_cobs_each_row_l870_87031


namespace special_op_2_4_5_l870_87093

def special_op (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem special_op_2_4_5 : special_op 2 4 5 = -24 := by
  sorry

end special_op_2_4_5_l870_87093


namespace polynomial_root_reciprocal_square_sum_l870_87058

theorem polynomial_root_reciprocal_square_sum :
  ∀ (a b c : ℝ), (a + b + c = 6) → (a * b + b * c + c * a = 11) → (a * b * c = 6) →
  (1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 = 49 / 36) :=
by
  intros a b c h_sum h_prod_sum h_prod
  sorry

end polynomial_root_reciprocal_square_sum_l870_87058


namespace games_played_by_player_3_l870_87098

theorem games_played_by_player_3 (games_1 games_2 : ℕ) (rotation_system : ℕ) :
  games_1 = 10 → games_2 = 21 →
  rotation_system = (games_2 - games_1) →
  rotation_system = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end games_played_by_player_3_l870_87098


namespace ab_diff_2023_l870_87037

theorem ab_diff_2023 (a b : ℝ) 
  (h : a^2 + b^2 - 4 * a - 6 * b + 13 = 0) : (a - b) ^ 2023 = -1 :=
sorry

end ab_diff_2023_l870_87037


namespace Mona_bikes_30_miles_each_week_l870_87066

theorem Mona_bikes_30_miles_each_week :
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  total_distance = 30 := by
  let monday_distance := 6
  let wednesday_distance := 12
  let saturday_distance := 2 * monday_distance
  let total_distance := monday_distance + wednesday_distance + saturday_distance
  show total_distance = 30
  sorry

end Mona_bikes_30_miles_each_week_l870_87066


namespace find_m_l870_87009

open Set Real

noncomputable def setA : Set ℝ := {x | x < 2}
noncomputable def setB : Set ℝ := {x | x > 4}
noncomputable def setC (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

theorem find_m (m : ℝ) : setC m ⊆ (setA ∪ setB) → m < 3 :=
by
  sorry

end find_m_l870_87009


namespace angle_complement_supplement_l870_87005

theorem angle_complement_supplement (x : ℝ) (h : 90 - x = 3 / 4 * (180 - x)) : x = 180 :=
by
  sorry

end angle_complement_supplement_l870_87005


namespace brochures_multiple_of_6_l870_87095

theorem brochures_multiple_of_6 (n : ℕ) (P : ℕ) (B : ℕ) 
  (hP : P = 12) (hn : n = 6) : ∃ k : ℕ, B = 6 * k := 
sorry

end brochures_multiple_of_6_l870_87095


namespace distance_to_canada_l870_87038

theorem distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) (driving_time : ℝ) (distance : ℝ) :
  speed = 60 ∧ total_time = 7 ∧ stop_time = 1 ∧ driving_time = total_time - stop_time ∧
  distance = speed * driving_time → distance = 360 :=
by
  sorry

end distance_to_canada_l870_87038


namespace erica_blank_question_count_l870_87014

variable {C W B : ℕ}

theorem erica_blank_question_count
  (h1 : C + W + B = 20)
  (h2 : 7 * C - 4 * W = 100) :
  B = 1 :=
by
  sorry

end erica_blank_question_count_l870_87014


namespace total_points_l870_87091

theorem total_points (paul_points cousin_points : ℕ) 
  (h_paul : paul_points = 3103) 
  (h_cousin : cousin_points = 2713) : 
  paul_points + cousin_points = 5816 := by
sorry

end total_points_l870_87091


namespace area_of_circular_flower_bed_l870_87027

theorem area_of_circular_flower_bed (C : ℝ) (hC : C = 62.8) : ∃ (A : ℝ), A = 314 :=
by
  sorry

end area_of_circular_flower_bed_l870_87027


namespace proof_l870_87007

-- Define the equation and its conditions
def equation (x m : ℤ) : Prop := (3 * x - 1) / 2 + m = 3

-- Part 1: Prove that for m = 5, the corresponding x must be 1
def part1 : Prop :=
  ∃ x : ℤ, equation x 5 ∧ x = 1

-- Part 2: Prove that if the equation has a positive integer solution, the positive integer m must be 2
def part2 : Prop :=
  ∃ m x : ℤ, m > 0 ∧ x > 0 ∧ equation x m ∧ m = 2

theorem proof : part1 ∧ part2 :=
  by
    sorry

end proof_l870_87007


namespace time_difference_l870_87004

-- Define the conditions
def time_to_nile_delta : Nat := 4
def number_of_alligators : Nat := 7
def combined_walking_time : Nat := 46

-- Define the mathematical statement we want to prove
theorem time_difference (x : Nat) :
  4 + 7 * (time_to_nile_delta + x) = combined_walking_time → x = 2 :=
by
  sorry

end time_difference_l870_87004


namespace parallel_lines_determine_plane_l870_87033

def determine_plane_by_parallel_lines := 
  let condition_4 := true -- Two parallel lines
  condition_4 = true

theorem parallel_lines_determine_plane : determine_plane_by_parallel_lines = true :=
by 
  sorry

end parallel_lines_determine_plane_l870_87033


namespace water_polo_team_selection_l870_87048

theorem water_polo_team_selection :
  let total_players := 20
  let team_size := 9
  let goalies := 2
  let remaining_players := total_players - goalies
  let combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  combination total_players goalies * combination remaining_players (team_size - goalies) = 6046560 :=
by
  -- Definitions and calculations to be filled here.
  sorry

end water_polo_team_selection_l870_87048


namespace total_number_of_students_is_40_l870_87060

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l870_87060


namespace find_y_when_x_is_4_l870_87020

def inverse_proportional (x y : ℝ) : Prop :=
  ∃ C : ℝ, x * y = C

theorem find_y_when_x_is_4 :
  ∀ x y : ℝ,
  inverse_proportional x y →
  (x + y = 20) →
  (x - y = 4) →
  (∃ y, y = 24 ∧ x = 4) :=
by
  sorry

end find_y_when_x_is_4_l870_87020


namespace number_of_rectangles_l870_87073

open Real Set

-- Given points A, B, C, D on a line L and a length k
variables {A B C D : ℝ} (L : Set ℝ) (k : ℝ)

-- The points are distinct and ordered on the line
axiom h1 : A ≠ B ∧ B ≠ C ∧ C ≠ D
axiom h2 : A < B ∧ B < C ∧ C < D

-- We need to show there are two rectangles with certain properties
theorem number_of_rectangles : 
  (∃ (rect1 rect2 : Set ℝ), 
    rect1 ≠ rect2 ∧ 
    (∃ (a1 b1 c1 d1 : ℝ), rect1 = {a1, b1, c1, d1} ∧ 
      a1 < b1 ∧ b1 < c1 ∧ c1 < d1 ∧ 
      (d1 - c1 = k ∨ c1 - b1 = k)) ∧ 
    (∃ (a2 b2 c2 d2 : ℝ), rect2 = {a2, b2, c2, d2} ∧ 
      a2 < b2 ∧ b2 < c2 ∧ c2 < d2 ∧ 
      (d2 - c2 = k ∨ c2 - b2 = k))
  ) :=
sorry

end number_of_rectangles_l870_87073


namespace matrix_pow_2018_l870_87061

open Matrix

-- Define the specific matrix
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![1, 1]]

-- Formalize the statement
theorem matrix_pow_2018 : A ^ 2018 = ![![1, 0], ![2018, 1]] :=
  sorry

end matrix_pow_2018_l870_87061


namespace find_T_value_l870_87054

theorem find_T_value (x y : ℤ) (R : ℤ) (h : R = 30) (h2 : (R / 2) * x * y = 21 * x + 20 * y - 13) :
    x = 3 ∧ y = 2 → x * y = 6 := by
  sorry

end find_T_value_l870_87054


namespace lockers_count_l870_87034

theorem lockers_count 
(TotalCost : ℝ) 
(first_cents : ℝ) 
(additional_cents : ℝ) 
(locker_start : ℕ) 
(locker_end : ℕ) : 
  TotalCost = 155.94 
  → first_cents = 0 
  → additional_cents = 0.03 
  → locker_start = 2 
  → locker_end = 1825 := 
by
  -- Declare the number of lockers as a variable and use it to construct the proof
  let num_lockers := locker_end - locker_start + 1
  -- The cost for labeling can be calculated and matched with TotalCost
  sorry

end lockers_count_l870_87034


namespace sum_of_squares_and_product_l870_87002

theorem sum_of_squares_and_product (x y : ℕ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y = Real.sqrt 202 := 
by
  sorry

end sum_of_squares_and_product_l870_87002


namespace bond_value_after_8_years_l870_87068

theorem bond_value_after_8_years :
  ∀ (P A r t : ℝ), P = 240 → r = 0.0833333333333332 → t = 8 →
  (A = P * (1 + r * t)) → A = 400 :=
by
  sorry

end bond_value_after_8_years_l870_87068


namespace largest_number_value_l870_87036

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l870_87036


namespace find_m_value_l870_87078

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l870_87078


namespace area_of_triangle_l870_87075

-- Definitions
variables {A B C : Type}
variables {i j k : ℕ}
variables (AB AC : ℝ)
variables (s t : ℝ)
variables (sinA : ℝ) (cosA : ℝ)

-- Conditions 
axiom sin_A : sinA = 4 / 5
axiom dot_product : s * t * cosA = 6

-- The problem theorem
theorem area_of_triangle : (1 / 2) * s * t * sinA = 4 :=
by
  sorry

end area_of_triangle_l870_87075


namespace denominator_of_expression_l870_87010

theorem denominator_of_expression (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end denominator_of_expression_l870_87010


namespace combined_total_l870_87013

variable (Jane Jean : ℕ)

theorem combined_total (h1 : Jean = 3 * Jane) (h2 : Jean = 57) : Jane + Jean = 76 := by
  sorry

end combined_total_l870_87013


namespace regular_polygon_sides_l870_87074

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l870_87074


namespace total_tickets_sold_l870_87097

theorem total_tickets_sold (A C : ℕ) (total_revenue : ℝ) (cost_adult cost_child : ℝ) :
  (cost_adult = 6.00) →
  (cost_child = 4.50) →
  (total_revenue = 2100.00) →
  (C = 200) →
  (cost_adult * ↑A + cost_child * ↑C = total_revenue) →
  A + C = 400 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof omitted
  sorry

end total_tickets_sold_l870_87097


namespace gangster_avoid_police_l870_87023

variable (a v : ℝ)
variable (house_side_length streets_distance neighbouring_distance police_interval : ℝ)
variable (police_speed gangster_speed_to_avoid_police : ℝ)

-- Given conditions
axiom house_properties : house_side_length = a ∧ neighbouring_distance = 2 * a
axiom streets_properties : streets_distance = 3 * a
axiom police_properties : police_interval = 9 * a ∧ police_speed = v

-- Correct answer in terms of Lean
theorem gangster_avoid_police :
  gangster_speed_to_avoid_police = 2 * v ∨ gangster_speed_to_avoid_police = v / 2 :=
by
  sorry

end gangster_avoid_police_l870_87023


namespace maximize_profit_l870_87099
-- Importing the entire necessary library

-- Definitions and conditions
def cost_price : ℕ := 40
def minimum_selling_price : ℕ := 44
def maximum_profit_margin : ℕ := 30
def sales_at_minimum_price : ℕ := 300
def price_increase_effect : ℕ := 10
def max_profit_price := 52
def max_profit := 2640

-- Function relationship between y and x
def sales_volume (x : ℕ) : ℕ := 300 - 10 * (x - 44)

-- Range of x
def valid_price (x : ℕ) : Prop := 44 ≤ x ∧ x ≤ 52

-- Statement of the problem
theorem maximize_profit (x : ℕ) (hx : valid_price x) : 
  sales_volume x = 300 - 10 * (x - 44) ∧
  44 ≤ x ∧ x ≤ 52 ∧
  x = 52 → 
  (x - cost_price) * (sales_volume x) = max_profit :=
sorry

end maximize_profit_l870_87099


namespace unique_solution_exists_l870_87071

def f (x y z : ℕ) : ℕ := (x + y - 2) * (x + y - 1) / 2 - z

theorem unique_solution_exists :
  ∀ (a b c d : ℕ), f a b c = 1993 ∧ f c d a = 1993 → (a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42) :=
by
  intros a b c d h
  sorry

end unique_solution_exists_l870_87071


namespace find_remainder_l870_87092

def dividend : ℕ := 997
def divisor : ℕ := 23
def quotient : ℕ := 43

theorem find_remainder : ∃ r : ℕ, dividend = (divisor * quotient) + r ∧ r = 8 :=
by
  sorry

end find_remainder_l870_87092


namespace solve_inequality_l870_87084

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if h : a > -1 then { x : ℝ | -1 < x ∧ x < a }
  else if h : a < -1 then { x : ℝ | a < x ∧ x < -1 }
  else ∅

theorem solve_inequality (x a : ℝ) :
  (x^2 + (1 - a)*x - a < 0) ↔ (
    (a > -1 → x ∈ { x : ℝ | -1 < x ∧ x < a }) ∧
    (a < -1 → x ∈ { x : ℝ | a < x ∧ x < -1 }) ∧
    (a = -1 → False)
  ) :=
sorry

end solve_inequality_l870_87084


namespace arithmetic_sequence_k_value_l870_87088

theorem arithmetic_sequence_k_value (a1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a1 = 1)
  (h2 : d = 2)
  (h3 : ∀ k : ℕ, S (k+2) - S k = 24) : k = 5 := 
sorry

end arithmetic_sequence_k_value_l870_87088


namespace exponential_fixed_point_l870_87030

theorem exponential_fixed_point (a : ℝ) (hx₁ : a > 0) (hx₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a ^ x) } := by
  sorry 

end exponential_fixed_point_l870_87030


namespace pathway_bricks_total_is_280_l870_87049

def total_bricks (n : ℕ) : ℕ :=
  let odd_bricks := 2 * (1 + 1 + ((n / 2) - 1) * 2)
  let even_bricks := 4 * (1 + 2 + (n / 2 - 1) * 2)
  odd_bricks + even_bricks
   
theorem pathway_bricks_total_is_280 (n : ℕ) (h : total_bricks n = 280) : n = 10 :=
sorry

end pathway_bricks_total_is_280_l870_87049
