import Mathlib

namespace sam_distance_traveled_l636_63680

-- Variables definition
variables (distance_marguerite : ℝ) (time_marguerite : ℝ) (time_sam : ℝ)

-- Given conditions
def marguerite_conditions : Prop :=
  distance_marguerite = 150 ∧
  time_marguerite = 3 ∧
  time_sam = 4

-- Statement to prove
theorem sam_distance_traveled (h : marguerite_conditions distance_marguerite time_marguerite time_sam) : 
  distance_marguerite / time_marguerite * time_sam = 200 :=
sorry

end sam_distance_traveled_l636_63680


namespace abc_eq_zero_l636_63633

variable (a b c : ℝ) (n : ℕ)

theorem abc_eq_zero
  (h1 : a^n + b^n = c^n)
  (h2 : a^(n+1) + b^(n+1) = c^(n+1))
  (h3 : a^(n+2) + b^(n+2) = c^(n+2)) :
  a * b * c = 0 :=
sorry

end abc_eq_zero_l636_63633


namespace total_legs_proof_l636_63621

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l636_63621


namespace unused_streetlights_remain_l636_63627

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l636_63627


namespace project_contribution_l636_63677

theorem project_contribution (total_cost : ℝ) (num_participants : ℝ) (expected_contribution : ℝ) 
  (h1 : total_cost = 25 * 10^9) 
  (h2 : num_participants = 300 * 10^6) 
  (h3 : expected_contribution = 83) : 
  total_cost / num_participants = expected_contribution := 
by 
  sorry

end project_contribution_l636_63677


namespace gas_pressure_inversely_proportional_l636_63647

theorem gas_pressure_inversely_proportional
  (p v k : ℝ)
  (v_i v_f : ℝ)
  (p_i p_f : ℝ)
  (h1 : v_i = 3.5)
  (h2 : p_i = 8)
  (h3 : v_f = 7)
  (h4 : p * v = k)
  (h5 : p_i * v_i = k)
  (h6 : p_f * v_f = k) : p_f = 4 := by
  sorry

end gas_pressure_inversely_proportional_l636_63647


namespace number_of_digits_if_million_place_l636_63670

theorem number_of_digits_if_million_place (n : ℕ) (h : n = 1000000) : 7 = 7 := by
  sorry

end number_of_digits_if_million_place_l636_63670


namespace rainfall_in_2011_l636_63643

-- Define the parameters
def avg_rainfall_2010 : ℝ := 37.2
def increase_from_2010_to_2011 : ℝ := 1.8
def months_in_a_year : ℕ := 12

-- Define the total rainfall in 2011
def total_rainfall_2011 : ℝ := 468

-- Prove that the total rainfall in Driptown in 2011 is 468 mm
theorem rainfall_in_2011 :
  avg_rainfall_2010 + increase_from_2010_to_2011 = 39.0 → 
  12 * (avg_rainfall_2010 + increase_from_2010_to_2011) = total_rainfall_2011 :=
by sorry

end rainfall_in_2011_l636_63643


namespace mean_score_all_students_l636_63619

theorem mean_score_all_students
  (M A E : ℝ) (m a e : ℝ)
  (hM : M = 78)
  (hA : A = 68)
  (hE : E = 82)
  (h_ratio_ma : m / a = 4 / 5)
  (h_ratio_mae : (m + a) / e = 9 / 2)
  : (M * m + A * a + E * e) / (m + a + e) = 74.4 := by
  sorry

end mean_score_all_students_l636_63619


namespace value_of_S_l636_63668

-- Defining the condition as an assumption
def one_third_one_eighth_S (S : ℝ) : Prop :=
  (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120

-- The statement we need to prove
theorem value_of_S (S : ℝ) (h : one_third_one_eighth_S S) : S = 120 :=
by
  sorry

end value_of_S_l636_63668


namespace current_average_age_of_seven_persons_l636_63615

theorem current_average_age_of_seven_persons (T : ℕ)
  (h1 : T + 12 = 6 * 43)
  (h2 : 69 = 69)
  : (T + 69) / 7 = 45 := by
  sorry

end current_average_age_of_seven_persons_l636_63615


namespace cube_tetrahedron_volume_ratio_l636_63673

theorem cube_tetrahedron_volume_ratio :
  let s := 2
  let v1 := (0, 0, 0)
  let v2 := (2, 2, 0)
  let v3 := (2, 0, 2)
  let v4 := (0, 2, 2)
  let a := Real.sqrt 8 -- Side length of the tetrahedron
  let volume_tetra := (a^3 * Real.sqrt 2) / 12
  let volume_cube := s^3
  volume_cube / volume_tetra = 6 * Real.sqrt 2 := 
by
  -- Proof content skipped
  intros
  sorry

end cube_tetrahedron_volume_ratio_l636_63673


namespace big_joe_height_is_8_l636_63629

variable (Pepe_height Frank_height Larry_height Ben_height BigJoe_height : ℝ)

axiom Pepe_height_def : Pepe_height = 4.5
axiom Frank_height_def : Frank_height = Pepe_height + 0.5
axiom Larry_height_def : Larry_height = Frank_height + 1
axiom Ben_height_def : Ben_height = Larry_height + 1
axiom BigJoe_height_def : BigJoe_height = Ben_height + 1

theorem big_joe_height_is_8 :
  BigJoe_height = 8 :=
sorry

end big_joe_height_is_8_l636_63629


namespace cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l636_63690

theorem cannot_represent_1986_as_sum_of_squares_of_6_odd_integers
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 % 2 = 1) 
  (h2 : a2 % 2 = 1) 
  (h3 : a3 % 2 = 1) 
  (h4 : a4 % 2 = 1) 
  (h5 : a5 % 2 = 1) 
  (h6 : a6 % 2 = 1) : 
  ¬ (1986 = a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2) := 
by 
  sorry

end cannot_represent_1986_as_sum_of_squares_of_6_odd_integers_l636_63690


namespace sum_of_roots_eq_three_l636_63693

-- Definitions of the polynomials
def poly1 (x : ℝ) : ℝ := 3 * x^3 + 3 * x^2 - 9 * x + 27
def poly2 (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + 5

-- Theorem stating the sum of the roots of the given equation is 3
theorem sum_of_roots_eq_three : 
  (∀ a b c d e f g h i : ℝ, 
    (poly1 a = 0) → (poly1 b = 0) → (poly1 c = 0) → 
    (poly2 d = 0) → (poly2 e = 0) → (poly2 f = 0) →
    a + b + c + d + e + f = 3) := 
by
  sorry

end sum_of_roots_eq_three_l636_63693


namespace tangent_line_find_a_l636_63694

theorem tangent_line_find_a (a : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ x, f x = x + 1/x - a * Real.log x)
  (h_tangent : ∀ x, tangent x = x + 1)
  (h_deriv : deriv f x₀ = deriv tangent x₀)
  (h_eq : f x₀ = tangent x₀) :
  a = -1 :=
sorry

end tangent_line_find_a_l636_63694


namespace part1_l636_63675

theorem part1 (a : ℤ) (h : a = -2) : 
  ((a^2 + a) / (a^2 - 3 * a)) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2 / 3 := by
  rw [h]
  sorry

end part1_l636_63675


namespace cost_per_person_is_125_l636_63603

-- Defining the conditions
def totalCost : ℤ := 25000000000
def peopleSharing : ℤ := 200000000

-- Define the expected cost per person based on the conditions
def costPerPerson : ℤ := totalCost / peopleSharing

-- Proving that the cost per person is 125 dollars.
theorem cost_per_person_is_125 : costPerPerson = 125 := by
  sorry

end cost_per_person_is_125_l636_63603


namespace person_B_correct_probability_l636_63635

-- Define probabilities
def P_A_correct : ℝ := 0.4
def P_A_incorrect : ℝ := 1 - P_A_correct
def P_B_correct_if_A_incorrect : ℝ := 0.5
def P_B_correct : ℝ := P_A_incorrect * P_B_correct_if_A_incorrect

-- Theorem statement
theorem person_B_correct_probability : P_B_correct = 0.3 :=
by
  -- Problem conditions implicitly used in definitions
  sorry

end person_B_correct_probability_l636_63635


namespace largest_number_in_sequence_is_48_l636_63642

theorem largest_number_in_sequence_is_48 
    (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ) 
    (h1 : 0 < a_1) 
    (h2 : a_1 < a_2 ∧ a_2 < a_3 ∧ a_3 < a_4 ∧ a_4 < a_5 ∧ a_5 < a_6)
    (h3 : ∃ k_1 k_2 k_3 k_4 k_5 : ℕ, k_1 > 1 ∧ k_2 > 1 ∧ k_3 > 1 ∧ k_4 > 1 ∧ k_5 > 1 ∧ 
          a_2 = k_1 * a_1 ∧ a_3 = k_2 * a_2 ∧ a_4 = k_3 * a_3 ∧ a_5 = k_4 * a_4 ∧ a_6 = k_5 * a_5)
    (h4 : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 79) 
    : a_6 = 48 := 
by 
    sorry

end largest_number_in_sequence_is_48_l636_63642


namespace intersection_set_l636_63656

def M : Set ℤ := {1, 2, 3, 5, 7}
def N : Set ℤ := {x | ∃ k ∈ M, x = 2 * k - 1}
def I : Set ℤ := {1, 3, 5}

theorem intersection_set :
  M ∩ N = I :=
by sorry

end intersection_set_l636_63656


namespace false_statement_about_circles_l636_63671

variable (P Q : Type) [MetricSpace P] [MetricSpace Q]
variable (p q : ℝ)
variable (dist_PQ : ℝ)

theorem false_statement_about_circles 
  (hA : p - q = dist_PQ → false)
  (hB : p + q = dist_PQ → false)
  (hC : p + q < dist_PQ → false)
  (hD : p - q < dist_PQ → false) : 
  false :=
by sorry

end false_statement_about_circles_l636_63671


namespace fraction_simplification_l636_63631

theorem fraction_simplification :
  (1722 ^ 2 - 1715 ^ 2) / (1729 ^ 2 - 1708 ^ 2) = 1 / 3 := by
  sorry

end fraction_simplification_l636_63631


namespace max_rectangles_in_triangle_l636_63676

theorem max_rectangles_in_triangle : 
  (∃ (n : ℕ), n = 192 ∧ 
  ∀ (i j : ℕ), i + j < 7 → ∀ (a b : ℕ), a ≤ 6 - i ∧ b ≤ 6 - j → 
  ∃ (rectangles : ℕ), rectangles = (6 - i) * (6 - j)) :=
sorry

end max_rectangles_in_triangle_l636_63676


namespace range_of_m_l636_63605

open Real

theorem range_of_m 
    (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (m : ℝ)
    (h : m * (a + 1/a) / sqrt 2 > 1) : 
    m ≥ sqrt 2 / 2 :=
sorry

end range_of_m_l636_63605


namespace degrees_for_cherry_pie_l636_63639

theorem degrees_for_cherry_pie
  (n c a b : ℕ)
  (hc : c = 15)
  (ha : a = 10)
  (hb : b = 9)
  (hn : n = 48)
  (half_remaining_cherry : (n - (c + a + b)) / 2 = 7) :
  (7 / 48 : ℚ) * 360 = 52.5 := 
by sorry

end degrees_for_cherry_pie_l636_63639


namespace hyperbola_eccentricity_range_l636_63602

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_B: ∃ A B : ℝ, x = -c ∧ |AF| = b^2 / a ∧ |CF| = a + c) :
  e > 2 :=
by
  sorry

end hyperbola_eccentricity_range_l636_63602


namespace stratified_sampling_l636_63653

theorem stratified_sampling 
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (H_male_students : male_students = 40)
  (H_female_students : female_students = 30)
  (H_sample_size : sample_size = 7)
  (H_stratified_sample : sample_size = male_students_drawn + female_students_drawn) :
  male_students_drawn = 4 ∧ female_students_drawn = 3  :=
sorry

end stratified_sampling_l636_63653


namespace compute_expression_l636_63628

-- Lean 4 statement for the mathematic equivalence proof problem
theorem compute_expression:
  (1004^2 - 996^2 - 1002^2 + 998^2) = 8000 := by
  sorry

end compute_expression_l636_63628


namespace solve_for_x_l636_63697

theorem solve_for_x : (∃ x : ℝ, ((10 - 2 * x) ^ 2 = 4 * x ^ 2 + 16) ∧ x = 2.1) :=
by
  sorry

end solve_for_x_l636_63697


namespace walking_speed_l636_63688

theorem walking_speed (x : ℝ) (h1 : 20 / x = 40 / (x + 5)) : x + 5 = 10 :=
  by
  sorry

end walking_speed_l636_63688


namespace aaron_weekly_earnings_l636_63679

def minutes_worked_monday : ℕ := 90
def minutes_worked_tuesday : ℕ := 40
def minutes_worked_wednesday : ℕ := 135
def minutes_worked_thursday : ℕ := 45
def minutes_worked_friday : ℕ := 60
def minutes_worked_saturday1 : ℕ := 90
def minutes_worked_saturday2 : ℕ := 75
def hourly_rate : ℕ := 4

def total_minutes_worked : ℕ :=
  minutes_worked_monday + 
  minutes_worked_tuesday + 
  minutes_worked_wednesday +
  minutes_worked_thursday + 
  minutes_worked_friday +
  minutes_worked_saturday1 + 
  minutes_worked_saturday2

def total_hours_worked : ℕ := total_minutes_worked / 60

def total_earnings : ℕ := total_hours_worked * hourly_rate

theorem aaron_weekly_earnings : total_earnings = 36 := by 
  sorry -- The proof is omitted.

end aaron_weekly_earnings_l636_63679


namespace base_five_to_ten_3214_l636_63624

theorem base_five_to_ten_3214 : (3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 4 * 5^0) = 434 := by
  sorry

end base_five_to_ten_3214_l636_63624


namespace oranges_now_is_50_l636_63689

def initial_fruits : ℕ := 150
def remaining_fruits : ℕ := initial_fruits / 2
def num_limes (L : ℕ) (O : ℕ) : Prop := O = 2 * L
def total_remaining_fruits (L : ℕ) (O : ℕ) : Prop := O + L = remaining_fruits

theorem oranges_now_is_50 : ∃ O L : ℕ, num_limes L O ∧ total_remaining_fruits L O ∧ O = 50 := by
  sorry

end oranges_now_is_50_l636_63689


namespace cycling_distance_l636_63638

-- Define the conditions
def cycling_time : ℕ := 40  -- Total cycling time in minutes
def time_per_interval : ℕ := 10  -- Time per interval in minutes
def distance_per_interval : ℕ := 2  -- Distance per interval in miles

-- Proof statement
theorem cycling_distance : (cycling_time / time_per_interval) * distance_per_interval = 8 := by
  sorry

end cycling_distance_l636_63638


namespace calculate_expression_l636_63672

theorem calculate_expression : ∀ x y : ℝ, x = 7 → y = 3 → (x - y) ^ 2 * (x + y) = 160 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end calculate_expression_l636_63672


namespace rebus_solution_l636_63616

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l636_63616


namespace montoya_food_budget_l636_63608

theorem montoya_food_budget (g t e : ℝ) (h1 : g = 0.6) (h2 : t = 0.8) : e = 0.2 :=
by sorry

end montoya_food_budget_l636_63608


namespace p_sufficient_but_not_necessary_for_q_l636_63625

variable (x : ℝ) (p q : Prop)

def p_condition : Prop := 0 < x ∧ x < 1
def q_condition : Prop := x^2 < 2 * x

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p_condition x → q_condition x) ∧
  ¬ (∀ x : ℝ, q_condition x → p_condition x) := by
  sorry

end p_sufficient_but_not_necessary_for_q_l636_63625


namespace markup_constant_relationship_l636_63651

variable (C S : ℝ) (k : ℝ)
variable (fractional_markup : k * S = 0.25 * C)
variable (relation : S = C + k * S)

theorem markup_constant_relationship (fractional_markup : k * S = 0.25 * C) (relation : S = C + k * S) :
  k = 1 / 5 :=
by
  sorry

end markup_constant_relationship_l636_63651


namespace smallest_number_diminished_by_35_l636_63622

def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

def conditions : List ℕ := [5, 10, 15, 20, 25, 30, 35]

def lcm_conditions := lcm_list conditions

theorem smallest_number_diminished_by_35 :
  ∃ n, n - 35 = lcm_conditions :=
sorry

end smallest_number_diminished_by_35_l636_63622


namespace convert_to_scientific_notation_l636_63695

theorem convert_to_scientific_notation :
  (1670000000 : ℝ) = 1.67 * 10 ^ 9 := 
by
  sorry

end convert_to_scientific_notation_l636_63695


namespace problem1_problem2_l636_63634

open Classical

theorem problem1 (x : ℝ) : -x^2 + 4 * x - 4 < 0 ↔ x ≠ 2 :=
sorry

theorem problem2 (x : ℝ) : (1 - x) / (x - 5) > 0 ↔ 1 < x ∧ x < 5 :=
sorry

end problem1_problem2_l636_63634


namespace range_function_l636_63652

open Real

noncomputable def function_to_prove (x : ℝ) (a : ℕ) : ℝ := x + 2 * a / x

theorem range_function (a : ℕ) (h1 : a^2 - a < 2) (h2 : a ≠ 0) : 
  Set.range (function_to_prove · a) = {y : ℝ | y ≤ -2 * sqrt 2} ∪ {y : ℝ | y ≥ 2 * sqrt 2} :=
by
  sorry

end range_function_l636_63652


namespace largest_multiple_of_three_l636_63655

theorem largest_multiple_of_three (n : ℕ) (h : 3 * n + (3 * n + 3) + (3 * n + 6) = 117) : 3 * n + 6 = 42 :=
by
  sorry

end largest_multiple_of_three_l636_63655


namespace number_of_knights_l636_63685

/--
On the island of Liars and Knights, a circular arrangement is called correct if everyone standing in the circle
can say that among his two neighbors there is a representative of his tribe. One day, 2019 natives formed a correct
arrangement in a circle. A liar approached them and said: "Now together we can also form a correct arrangement in a circle."
Prove that the number of knights in the initial arrangement is 1346.
-/
theorem number_of_knights : 
  ∀ (K L : ℕ), 
    (K + L = 2019) → 
    (K ≥ 2 * L) → 
    (K ≤ 2 * L + 1) → 
  K = 1346 :=
by
  intros K L h1 h2 h3
  sorry

end number_of_knights_l636_63685


namespace filtration_concentration_l636_63600

-- Variables and conditions used in the problem
variable (P P0 : ℝ) (k t : ℝ)
variable (h1 : P = P0 * Real.exp (-k * t))
variable (h2 : Real.exp (-2 * k) = 0.8)

-- Main statement: Prove the concentration after 5 hours is approximately 57% of the original
theorem filtration_concentration :
  (P0 * Real.exp (-5 * k)) / P0 = 0.57 :=
by sorry

end filtration_concentration_l636_63600


namespace other_number_is_300_l636_63611

theorem other_number_is_300 (A B : ℕ) (h1 : A = 231) (h2 : lcm A B = 2310) (h3 : gcd A B = 30) : B = 300 := by
  sorry

end other_number_is_300_l636_63611


namespace statement_I_l636_63669

section Problem
variable (g : ℝ → ℝ)

-- Conditions
def cond1 : Prop := ∀ x : ℝ, g x > 0
def cond2 : Prop := ∀ a b : ℝ, g a * g b = g (a + 2 * b)

-- Statement I to be proved
theorem statement_I (h1 : cond1 g) (h2 : cond2 g) : g 0 = 1 :=
by
  -- Proof is omitted
  sorry
end Problem

end statement_I_l636_63669


namespace intersection_P_Q_l636_63623

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | x^2 < 4}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := 
  sorry

end intersection_P_Q_l636_63623


namespace team_total_points_l636_63649

-- Definition of Wade's average points per game
def wade_avg_points_per_game := 20

-- Definition of teammates' average points per game
def teammates_avg_points_per_game := 40

-- Definition of the number of games
def number_of_games := 5

-- The total points calculation problem
theorem team_total_points 
  (Wade_avg : wade_avg_points_per_game = 20)
  (Teammates_avg : teammates_avg_points_per_game = 40)
  (Games : number_of_games = 5) :
  5 * wade_avg_points_per_game + 5 * teammates_avg_points_per_game = 300 := 
by 
  -- The proof is omitted and marked as sorry
  sorry

end team_total_points_l636_63649


namespace letter_ratio_l636_63665

theorem letter_ratio (G B M : ℕ) (h1 : G = B + 10) 
                     (h2 : B = 40) 
                     (h3 : G + B + M = 270) : 
                     M / (G + B) = 2 := 
by 
  sorry

end letter_ratio_l636_63665


namespace line_through_origin_in_quadrants_l636_63618

theorem line_through_origin_in_quadrants (A B C : ℝ) :
  (-A * x - B * y + C = 0) ∧ (0 = 0) ∧ (exists x y, 0 < x * y) →
  (C = 0) ∧ (A * B < 0) :=
sorry

end line_through_origin_in_quadrants_l636_63618


namespace sugar_price_difference_l636_63681

theorem sugar_price_difference (a b : ℝ) (h : (3 / 5 * a + 2 / 5 * b) - (2 / 5 * a + 3 / 5 * b) = 1.32) :
  a - b = 6.6 :=
by
  sorry

end sugar_price_difference_l636_63681


namespace arithmetic_seq_common_diff_l636_63661

theorem arithmetic_seq_common_diff
  (a₃ a₇ S₁₀ : ℤ)
  (h₁ : a₃ + a₇ = 16)
  (h₂ : S₁₀ = 85)
  (a₃_eq : ∃ a₁ d : ℤ, a₃ = a₁ + 2 * d)
  (a₇_eq : ∃ a₁ d : ℤ, a₇ = a₁ + 6 * d)
  (S₁₀_eq : ∃ a₁ d : ℤ, S₁₀ = 10 * a₁ + 45 * d) :
  ∃ d : ℤ, d = 1 :=
by
  sorry

end arithmetic_seq_common_diff_l636_63661


namespace find_f_neg_one_l636_63636

theorem find_f_neg_one (f h : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
    (h2 : ∀ x, h x = f x - 9) (h3 : h 1 = 2) : f (-1) = -11 := 
by
  sorry

end find_f_neg_one_l636_63636


namespace simplify_expression_l636_63692

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 :=
by
  sorry

end simplify_expression_l636_63692


namespace pump_B_rate_l636_63641

noncomputable def rate_A := 1 / 2
noncomputable def rate_C := 1 / 6

theorem pump_B_rate :
  ∃ B : ℝ, (rate_A + B - rate_C = 4 / 3) ∧ (B = 1) := by
  sorry

end pump_B_rate_l636_63641


namespace trapezoid_area_l636_63646

-- Definitions based on conditions
def CL_div_LD (CL LD : ℝ) : Prop := CL / LD = 1 / 4

-- The main statement we want to prove
theorem trapezoid_area (BC CD : ℝ) (h1 : BC = 9) (h2 : CD = 30) (CL LD : ℝ) (h3 : CL_div_LD CL LD) : 
  1/2 * (BC + AD) * 24 = 972 :=
sorry

end trapezoid_area_l636_63646


namespace sally_last_10_shots_made_l636_63613

def sally_initial_shots : ℕ := 30
def sally_initial_success_rate : ℝ := 0.60
def sally_additional_shots : ℕ := 10
def sally_final_success_rate : ℝ := 0.65

theorem sally_last_10_shots_made (x : ℕ) 
  (h1 : sally_initial_success_rate * sally_initial_shots = 18)
  (h2 : sally_final_success_rate * (sally_initial_shots + sally_additional_shots) = 26) :
  x = 8 :=
by
  sorry

end sally_last_10_shots_made_l636_63613


namespace a_b_finish_job_in_15_days_l636_63667

theorem a_b_finish_job_in_15_days (A B C : ℝ) 
  (h1 : A + B + C = 1 / 5)
  (h2 : C = 1 / 7.5) : 
  (1 / (A + B)) = 15 :=
by
  sorry

end a_b_finish_job_in_15_days_l636_63667


namespace y_relation_l636_63606

noncomputable def f (x : ℝ) : ℝ := -2 * x + 5

theorem y_relation (x1 y1 y2 y3 : ℝ) (h1 : y1 = f x1) (h2 : y2 = f (x1 - 2)) (h3 : y3 = f (x1 + 3)) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end y_relation_l636_63606


namespace cos_105_sub_alpha_l636_63644

variable (α : ℝ)

-- Condition
def condition : Prop := Real.cos (75 * Real.pi / 180 + α) = 1 / 2

-- Statement
theorem cos_105_sub_alpha (h : condition α) : Real.cos (105 * Real.pi / 180 - α) = -1 / 2 :=
by
  sorry

end cos_105_sub_alpha_l636_63644


namespace correct_equation_l636_63648

theorem correct_equation (x : ℝ) : 3 * x + 20 = 4 * x - 25 :=
by sorry

end correct_equation_l636_63648


namespace at_least_one_not_less_than_two_l636_63699

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → false := 
sorry

end at_least_one_not_less_than_two_l636_63699


namespace decimal_division_l636_63686

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end decimal_division_l636_63686


namespace triangle_problems_l636_63696

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {S : ℝ}

theorem triangle_problems
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13)
  (h3 : b + c = 5) :
  (A = π / 3) ∧ (S = Real.sqrt 3) :=
by
  sorry

end triangle_problems_l636_63696


namespace team_win_percentage_remaining_l636_63601

theorem team_win_percentage_remaining (won_first_30: ℝ) (total_games: ℝ) (total_wins: ℝ)
  (h1: won_first_30 = 0.40 * 30)
  (h2: total_games = 120)
  (h3: total_wins = 0.70 * total_games) :
  (total_wins - won_first_30) / (total_games - 30) * 100 = 80 :=
by
  sorry


end team_win_percentage_remaining_l636_63601


namespace largest_4digit_div_by_35_l636_63632

theorem largest_4digit_div_by_35 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (35 ∣ n) ∧ (∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (35 ∣ m) → m ≤ n) ∧ n = 9985 :=
by
  sorry

end largest_4digit_div_by_35_l636_63632


namespace sequence_bound_l636_63658

theorem sequence_bound (n : ℕ) (a : ℝ) (a_seq : ℕ → ℝ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq n = a) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k < n - 1 → a_seq (k + 1) ≤ (a_seq k + a_seq (k + 2)) / 2) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a_seq k ≤ a := 
by
  sorry

end sequence_bound_l636_63658


namespace correct_statements_count_l636_63687

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count_l636_63687


namespace part1_solution_set_part2_minimum_value_l636_63654

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

theorem part1_solution_set (x : ℝ) :
  (f x ≥ -1) ↔ (2 / 3 ≤ x ∧ x ≤ 6) := sorry

variables {a b c : ℝ}
theorem part2_minimum_value (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = 6) :
  (1 / (2 * a + b) + 1 / (2 * a + c) ≥ 2 / 3) := 
sorry

end part1_solution_set_part2_minimum_value_l636_63654


namespace transformed_curve_l636_63650

theorem transformed_curve (x y : ℝ) :
  (∃ (x1 y1 : ℝ), x1 = 3*x ∧ y1 = 2*y ∧ (x1^2 / 9 + y1^2 / 4 = 1)) →
  x^2 + y^2 = 1 :=
by
  sorry

end transformed_curve_l636_63650


namespace fourth_derivative_at_0_l636_63674

noncomputable def f : ℝ → ℝ := sorry

axiom f_at_0 : f 0 = 1
axiom f_prime_at_0 : deriv f 0 = 2
axiom f_double_prime : ∀ t, deriv (deriv f) t = 4 * deriv f t - 3 * f t + 1

-- We want to prove that the fourth derivative of f at 0 equals 54
theorem fourth_derivative_at_0 : deriv (deriv (deriv (deriv f))) 0 = 54 :=
sorry

end fourth_derivative_at_0_l636_63674


namespace number_of_members_l636_63609

variable (n : ℕ)

-- Conditions
def each_member_contributes_n_cents : Prop := n * n = 64736

-- Theorem that relates to the number of members being 254
theorem number_of_members (h : each_member_contributes_n_cents n) : n = 254 :=
sorry

end number_of_members_l636_63609


namespace length_of_bridge_l636_63666

theorem length_of_bridge (length_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (total_distance : ℝ) (bridge_length : ℝ) :
  length_train = 160 →
  speed_kmh = 45 →
  time_sec = 30 →
  speed_ms = 45 * (1000 / 3600) →
  total_distance = speed_ms * time_sec →
  bridge_length = total_distance - length_train →
  bridge_length = 215 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end length_of_bridge_l636_63666


namespace exponent_problem_l636_63662

theorem exponent_problem : (-1 : ℝ)^2003 / (-1 : ℝ)^2004 = -1 := by
  sorry

end exponent_problem_l636_63662


namespace liquid_X_percentage_correct_l636_63626

noncomputable def percent_liquid_X_in_solution_A := 0.8 / 100
noncomputable def percent_liquid_X_in_solution_B := 1.8 / 100

noncomputable def weight_solution_A := 400.0
noncomputable def weight_solution_B := 700.0

noncomputable def weight_liquid_X_in_A := percent_liquid_X_in_solution_A * weight_solution_A
noncomputable def weight_liquid_X_in_B := percent_liquid_X_in_solution_B * weight_solution_B

noncomputable def total_weight_solution := weight_solution_A + weight_solution_B
noncomputable def total_weight_liquid_X := weight_liquid_X_in_A + weight_liquid_X_in_B

noncomputable def percent_liquid_X_in_mixed_solution := (total_weight_liquid_X / total_weight_solution) * 100

theorem liquid_X_percentage_correct :
  percent_liquid_X_in_mixed_solution = 1.44 :=
by
  sorry

end liquid_X_percentage_correct_l636_63626


namespace domain_f_2x_l636_63684

-- Given conditions as definitions
def domain_f_x_minus_1 (x : ℝ) := 3 < x ∧ x ≤ 7

-- The main theorem statement that needs a proof
theorem domain_f_2x : (∀ x : ℝ, domain_f_x_minus_1 (x-1) → (1 < x ∧ x ≤ 3)) :=
by
  -- Proof steps will be here, however, as requested, they are omitted.
  sorry

end domain_f_2x_l636_63684


namespace triangle_ratio_condition_l636_63698

theorem triangle_ratio_condition (a b c : ℝ) (A B C : ℝ) (h1 : b * Real.cos C + c * Real.cos B = 2 * b)
  (h2 : a = b * Real.sin A / Real.sin B)
  (h3 : b = a * Real.sin B / Real.sin A)
  (h4 : c = a * Real.sin C / Real.sin A)
  (h5 : ∀ x, Real.sin (B + C) = Real.sin x): 
  b / a = 1 / 2 :=
by
  sorry

end triangle_ratio_condition_l636_63698


namespace gcd_of_18_and_30_l636_63691

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l636_63691


namespace burger_cost_l636_63620

theorem burger_cost :
  ∃ b s f : ℕ, 4 * b + 2 * s + 3 * f = 480 ∧ 3 * b + s + 2 * f = 360 ∧ b = 80 :=
by
  sorry

end burger_cost_l636_63620


namespace production_rate_equation_l636_63617

theorem production_rate_equation (x : ℝ) (h1 : ∀ t : ℝ, t = 600 / (x + 8)) (h2 : ∀ t : ℝ, t = 400 / x) : 
  600/(x + 8) = 400/x :=
by
  sorry

end production_rate_equation_l636_63617


namespace room_length_l636_63630

/-- Define the conditions -/
def width : ℝ := 3.75
def cost_paving : ℝ := 6187.5
def cost_per_sqm : ℝ := 300

/-- Prove that the length of the room is 5.5 meters -/
theorem room_length : 
  (cost_paving / cost_per_sqm) / width = 5.5 :=
by
  sorry

end room_length_l636_63630


namespace determine_digits_l636_63645

def digit (n : Nat) : Prop := n < 10

theorem determine_digits :
  ∃ (A B C D : Nat), digit A ∧ digit B ∧ digit C ∧ digit D ∧
    (1000 * A + 100 * B + 10 * B + B) ^ 2 = 10000 * A + 1000 * C + 100 * D + 10 * B + B ∧
    (1000 * C + 100 * D + 10 * D + D) ^ 3 = 10000 * A + 1000 * C + 100 * B + 10 * D + D ∧
    A = 9 ∧ B = 6 ∧ C = 2 ∧ D = 1 := 
by
  sorry

end determine_digits_l636_63645


namespace find_multiple_of_hats_l636_63637

/-
   Given:
   - Fire chief Simpson has 15 hats.
   - Policeman O'Brien now has 34 hats.
   - Before he lost one, Policeman O'Brien had 5 more hats than a certain multiple of Fire chief Simpson's hats.
   Prove:
   The multiple of Fire chief Simpson's hats that Policeman O'Brien had before he lost one is 2.
-/

theorem find_multiple_of_hats :
  ∃ x : ℕ, 34 + 1 = 5 + 15 * x ∧ x = 2 :=
by
  sorry

end find_multiple_of_hats_l636_63637


namespace range_of_b_l636_63683

theorem range_of_b (a b : ℝ) (h1 : a ≠ 0) (h2 : a * b^2 > a) (h3 : a > a * b) : b < -1 :=
sorry

end range_of_b_l636_63683


namespace degree_measure_supplement_complement_l636_63607

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end degree_measure_supplement_complement_l636_63607


namespace train_length_is_150_l636_63610

noncomputable def train_length (v_km_hr : ℝ) (t_sec : ℝ) : ℝ :=
  let v_m_s := v_km_hr * (5 / 18)
  v_m_s * t_sec

theorem train_length_is_150 :
  train_length 122 4.425875438161669 = 150 :=
by
  -- It follows directly from the given conditions and known conversion factor
  -- The actual proof steps would involve arithmetic simplifications.
  sorry

end train_length_is_150_l636_63610


namespace minimum_value_expression_l636_63678

theorem minimum_value_expression {a : ℝ} (h₀ : 1 < a) (h₁ : a < 4) : 
  (∃ m : ℝ, (∀ x : ℝ, 1 < x ∧ x < 4 → m ≤ (x / (4 - x) + 1 / (x - 1))) ∧ m = 2) :=
sorry

end minimum_value_expression_l636_63678


namespace nina_money_proof_l636_63657

def total_money_nina_has (W M : ℝ) : Prop :=
  (10 * W = M) ∧ (14 * (W - 1.75) = M)

theorem nina_money_proof (W M : ℝ) (h : total_money_nina_has W M) : M = 61.25 :=
by 
  sorry

end nina_money_proof_l636_63657


namespace expressions_inequivalence_l636_63659

theorem expressions_inequivalence (x : ℝ) (h : x > 0) :
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (x + 1) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (x + 1) ^ (2 * x + 2)) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (0.5 * x + x) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (2 * x + 2) ^ (2 * x + 2)) := by
  sorry

end expressions_inequivalence_l636_63659


namespace length_PC_in_rectangle_l636_63682

theorem length_PC_in_rectangle (PA PB PD: ℝ) (P_inside: True) 
(h1: PA = 5) (h2: PB = 7) (h3: PD = 3) : PC = Real.sqrt 65 := 
sorry

end length_PC_in_rectangle_l636_63682


namespace arrangement_count_l636_63614

theorem arrangement_count (students : Fin 6) (teacher : Bool) :
  (teacher = true) ∧
  ∀ (A B : Fin 6), 
    A ≠ 0 ∧ B ≠ 5 →
    A ≠ B →
    (Sorry) = 960 := sorry

end arrangement_count_l636_63614


namespace second_rooster_weight_l636_63612

theorem second_rooster_weight (cost_per_kg : ℝ) (weight_1 : ℝ) (total_earnings : ℝ) (weight_2 : ℝ) :
  cost_per_kg = 0.5 →
  weight_1 = 30 →
  total_earnings = 35 →
  total_earnings = weight_1 * cost_per_kg + weight_2 * cost_per_kg →
  weight_2 = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end second_rooster_weight_l636_63612


namespace total_residents_l636_63640

open Set

/-- 
In a village, there are 912 residents who speak Bashkir, 
653 residents who speak Russian, 
and 435 residents who speak both languages.
Prove the total number of residents in the village is 1130.
-/
theorem total_residents (A B : Finset ℕ) (nA nB nAB : ℕ)
  (hA : nA = 912)
  (hB : nB = 653)
  (hAB : nAB = 435) :
  nA + nB - nAB = 1130 := by
  sorry

end total_residents_l636_63640


namespace quadratic_equation_m_condition_l636_63604

theorem quadratic_equation_m_condition (m : ℝ) :
  (m + 1 ≠ 0) ↔ (m ≠ -1) :=
by sorry

end quadratic_equation_m_condition_l636_63604


namespace sum_of_coefficients_l636_63663

theorem sum_of_coefficients (a b : ℝ)
  (h1 : 15 * a^4 * b^2 = 135)
  (h2 : 6 * a^5 * b = -18) :
  (a + b)^6 = 64 := by
  sorry

end sum_of_coefficients_l636_63663


namespace repair_cost_l636_63664

theorem repair_cost
  (R : ℝ) -- R is the cost to repair the used shoes
  (new_shoes_cost : ℝ := 30) -- New shoes cost $30.00
  (new_shoes_lifetime : ℝ := 2) -- New shoes last for 2 years
  (percentage_increase : ℝ := 42.857142857142854) 
  (h1 : new_shoes_cost / new_shoes_lifetime = R + (percentage_increase / 100) * R) :
  R = 10.50 :=
by
  sorry

end repair_cost_l636_63664


namespace find_original_cost_of_chips_l636_63660

def original_cost_chips (discount amount_spent : ℝ) : ℝ :=
  discount + amount_spent

theorem find_original_cost_of_chips :
  original_cost_chips 17 18 = 35 := by
  sorry

end find_original_cost_of_chips_l636_63660
