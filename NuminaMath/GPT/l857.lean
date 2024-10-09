import Mathlib

namespace communication_scenarios_l857_85726

theorem communication_scenarios
  (nA : ℕ) (nB : ℕ) (hA : nA = 10) (hB : nB = 20) : 
  (∃ scenarios : ℕ, scenarios = 2 ^ (nA * nB)) :=
by
  use 2 ^ (10 * 20)
  sorry

end communication_scenarios_l857_85726


namespace monotonic_intervals_range_of_values_l857_85778

-- Part (1): Monotonic intervals of the function
theorem monotonic_intervals (a : ℝ) (h_a : a = 0) :
  (∀ x, 0 < x ∧ x < 1 → (1 + Real.log x) / x > 0) ∧ (∀ x, 1 < x → (1 + Real.log x) / x < 0) :=
by
  sorry

-- Part (2): Range of values for \(a\)
theorem range_of_values (a : ℝ) (h_f : ∀ x, 0 < x → (1 + Real.log x) / x - a ≤ 0) : 
  1 ≤ a :=
by
  sorry

end monotonic_intervals_range_of_values_l857_85778


namespace heesu_received_most_sweets_l857_85798

theorem heesu_received_most_sweets
  (total_sweets : ℕ)
  (minsus_sweets : ℕ)
  (jaeyoungs_sweets : ℕ)
  (heesus_sweets : ℕ)
  (h_total : total_sweets = 30)
  (h_minsu : minsus_sweets = 12)
  (h_jaeyoung : jaeyoungs_sweets = 3)
  (h_heesu : heesus_sweets = 15) :
  heesus_sweets = max minsus_sweets (max jaeyoungs_sweets heesus_sweets) :=
by sorry

end heesu_received_most_sweets_l857_85798


namespace compare_logs_l857_85788

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.logb 2 3
noncomputable def c : ℝ := Real.logb 5 8

theorem compare_logs : a < c ∧ c < b := by
  sorry

end compare_logs_l857_85788


namespace roots_range_of_a_l857_85729

theorem roots_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + (a - 2)*|x - 3| + 9 - 2*a = 0) ↔ a > 0 ∨ a = -2 :=
sorry

end roots_range_of_a_l857_85729


namespace min_S_n_at_24_l857_85709

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n : ℤ) * (2 * n - 48)

theorem min_S_n_at_24 : (∀ n : ℕ, n > 0 → S_n n ≥ S_n 24) ∧ S_n 24 < S_n 25 :=
by 
  sorry

end min_S_n_at_24_l857_85709


namespace quadratic_equation_general_form_l857_85713

theorem quadratic_equation_general_form (x : ℝ) (h : 4 * x = x^2 - 8) : x^2 - 4 * x - 8 = 0 :=
sorry

end quadratic_equation_general_form_l857_85713


namespace range_of_a_l857_85737

noncomputable def f (x a : ℝ) : ℝ := (Real.sqrt x) / (x^3 - 3 * x + a)

theorem range_of_a (a : ℝ) :
    (∀ x, 0 ≤ x → x^3 - 3 * x + a ≠ 0) ↔ 2 < a := 
by 
  sorry

end range_of_a_l857_85737


namespace max_days_for_same_shift_l857_85760

open BigOperators

-- We define the given conditions
def nurses : ℕ := 15
def shifts_per_day : ℕ := 24 / 8
noncomputable def total_pairs : ℕ := (nurses.choose 2)

-- The main statement to prove
theorem max_days_for_same_shift : 
  35 = total_pairs / shifts_per_day := by
  sorry

end max_days_for_same_shift_l857_85760


namespace gcd_m_l857_85738

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by
  sorry

end gcd_m_l857_85738


namespace quadratic_two_distinct_real_roots_l857_85708

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  2 * k ≠ 0 → (8 * k + 1)^2 - 64 * k^2 > 0 → k > -1 / 16 ∧ k ≠ 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l857_85708


namespace vertex_of_parabola_l857_85771

theorem vertex_of_parabola (c d : ℝ) :
  (∀ x, -2 * x^2 + c * x + d ≤ 0 ↔ x ≥ -7 / 2) →
  ∃ k, k = (-7 / 2 : ℝ) ∧ y = -2 * (x + 7 / 2)^2 + 0 := 
sorry

end vertex_of_parabola_l857_85771


namespace smallest_of_three_l857_85733

noncomputable def A : ℕ := 38 + 18
noncomputable def B : ℕ := A - 26
noncomputable def C : ℕ := B / 3

theorem smallest_of_three : C < A ∧ C < B := by
  sorry

end smallest_of_three_l857_85733


namespace value_of_x_plus_y_l857_85730

theorem value_of_x_plus_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l857_85730


namespace veranda_area_correct_l857_85727

noncomputable def area_veranda (length_room : ℝ) (width_room : ℝ) (width_veranda : ℝ) (radius_obstacle : ℝ) : ℝ :=
  let total_length := length_room + 2 * width_veranda
  let total_width := width_room + 2 * width_veranda
  let area_total := total_length * total_width
  let area_room := length_room * width_room
  let area_circle := Real.pi * radius_obstacle^2
  area_total - area_room - area_circle

theorem veranda_area_correct :
  area_veranda 18 12 2 3 = 107.726 :=
by sorry

end veranda_area_correct_l857_85727


namespace minimum_value_a_l857_85770

theorem minimum_value_a (a : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 2| ≤ a) → a ≥ 3 :=
by 
  sorry

end minimum_value_a_l857_85770


namespace problem_sum_150_consecutive_integers_l857_85794

theorem problem_sum_150_consecutive_integers : 
  ∃ k : ℕ, 150 * k + 11325 = 5310375 :=
sorry

end problem_sum_150_consecutive_integers_l857_85794


namespace arrange_magnitudes_l857_85751

theorem arrange_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y := by
  sorry

end arrange_magnitudes_l857_85751


namespace canoe_speed_downstream_l857_85745

theorem canoe_speed_downstream (V_upstream V_s V_c V_downstream : ℝ) 
    (h1 : V_upstream = 6) 
    (h2 : V_s = 2) 
    (h3 : V_upstream = V_c - V_s) 
    (h4 : V_downstream = V_c + V_s) : 
  V_downstream = 10 := 
by 
  sorry

end canoe_speed_downstream_l857_85745


namespace geometric_progression_solution_l857_85731

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end geometric_progression_solution_l857_85731


namespace property_P_difference_l857_85780

noncomputable def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    6 * 2^(n / 2) - n - 5 
  else 
    4 * 2^((n + 1) / 2) - n - 5

theorem property_P_difference : f 9 - f 8 = 31 := by
  sorry

end property_P_difference_l857_85780


namespace molecular_weight_compound_l857_85767

-- Definitions of atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Definitions of the number of atoms in the compound
def num_Cu : ℝ := 1
def num_C : ℝ := 1
def num_O : ℝ := 3

-- The molecular weight of the compound
def molecular_weight : ℝ := (num_Cu * atomic_weight_Cu) + (num_C * atomic_weight_C) + (num_O * atomic_weight_O)

-- Statement to prove
theorem molecular_weight_compound : molecular_weight = 123.554 := by
  sorry

end molecular_weight_compound_l857_85767


namespace sum_of_z_values_l857_85768

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_z_values (z : ℝ) : 
  (f (4 * z) = 13) → (∃ z1 z2 : ℝ, z1 = 1/8 ∧ z2 = -1/4 ∧ z1 + z2 = -1/8) :=
sorry

end sum_of_z_values_l857_85768


namespace identify_vanya_l857_85753

structure Twin :=
(name : String)
(truth_teller : Bool)

def is_vanya_truth_teller (twin : Twin) (vanya vitya : Twin) : Prop :=
  twin = vanya ∧ twin.truth_teller ∨ twin = vitya ∧ ¬twin.truth_teller

theorem identify_vanya
  (vanya vitya : Twin)
  (h_vanya : vanya.name = "Vanya")
  (h_vitya : vitya.name = "Vitya")
  (h_one_truth : ∃ t : Twin, t = vanya ∨ t = vitya ∧ (t.truth_teller = true ∨ t.truth_teller = false))
  (h_one_lie : ∀ t : Twin, t = vanya ∨ t = vitya → ¬(t.truth_teller = true ∧ t = vitya) ∧ ¬(t.truth_teller = false ∧ t = vanya)) :
  ∀ twin : Twin, twin = vanya ∨ twin = vitya →
  (is_vanya_truth_teller twin vanya vitya ↔ (twin = vanya ∧ twin.truth_teller = true)) :=
by
  sorry

end identify_vanya_l857_85753


namespace find_ab_l857_85781

noncomputable def validate_ab : Prop :=
  let n : ℕ := 8
  let a : ℕ := n^2 - 1
  let b : ℕ := n
  a = 63 ∧ b = 8

theorem find_ab : validate_ab :=
by
  sorry

end find_ab_l857_85781


namespace number_of_distinguishable_arrangements_l857_85761

-- Define the conditions
def num_blue_tiles : Nat := 1
def num_red_tiles : Nat := 2
def num_green_tiles : Nat := 3
def num_yellow_tiles : Nat := 2
def total_tiles : Nat := num_blue_tiles + num_red_tiles + num_green_tiles + num_yellow_tiles

-- The goal is to prove the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  (Nat.factorial total_tiles) / ((Nat.factorial num_green_tiles) * 
                                (Nat.factorial num_red_tiles) * 
                                (Nat.factorial num_yellow_tiles) * 
                                (Nat.factorial num_blue_tiles)) = 1680 := by
  sorry

end number_of_distinguishable_arrangements_l857_85761


namespace determine_identity_l857_85725

-- Define the types for human and vampire
inductive Being
| human
| vampire

-- Define the responses for sanity questions
def claims_sanity (b : Being) : Prop :=
  match b with
  | Being.human   => true
  | Being.vampire => false

-- Proof statement: Given that a human always claims sanity and a vampire always claims insanity,
-- asking "Are you sane?" will determine their identity. 
theorem determine_identity (b : Being) (h : b = Being.human ↔ claims_sanity b = true) : 
  ((claims_sanity b = true) → b = Being.human) ∧ ((claims_sanity b = false) → b = Being.vampire) :=
sorry

end determine_identity_l857_85725


namespace number_of_partners_l857_85790

def total_profit : ℝ := 80000
def majority_owner_share := 0.25 * total_profit
def remaining_profit := total_profit - majority_owner_share
def partner_share := 0.25 * remaining_profit
def combined_share := majority_owner_share + 2 * partner_share

theorem number_of_partners : combined_share = 50000 → remaining_profit / partner_share = 4 := by
  intro h1
  have h_majority : majority_owner_share = 0.25 * total_profit := by sorry
  have h_remaining : remaining_profit = total_profit - majority_owner_share := by sorry
  have h_partner : partner_share = 0.25 * remaining_profit := by sorry
  have h_combined : combined_share = majority_owner_share + 2 * partner_share := by sorry
  calc
    remaining_profit / partner_share = _ := by sorry
    4 = 4 := by sorry

end number_of_partners_l857_85790


namespace range_of_2a_minus_b_l857_85724

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2 * a^2 - a * b - b^2 - 4 = 0) :
  (2 * a - b) ∈ (Set.Ici (8 / 3)) :=
sorry

end range_of_2a_minus_b_l857_85724


namespace cellphone_surveys_l857_85715

theorem cellphone_surveys
  (regular_rate : ℕ)
  (total_surveys : ℕ)
  (higher_rate_multiplier : ℕ)
  (total_earnings : ℕ)
  (higher_rate_bonus : ℕ)
  (x : ℕ) :
  regular_rate = 10 → total_surveys = 100 →
  higher_rate_multiplier = 130 → total_earnings = 1180 →
  higher_rate_bonus = 3 → (10 * (100 - x) + 13 * x = 1180) →
  x = 60 :=
by
  sorry

end cellphone_surveys_l857_85715


namespace beth_friends_l857_85734

theorem beth_friends (F : ℝ) (h1 : 4 / F + 6 = 6.4) : F = 10 :=
by
  sorry

end beth_friends_l857_85734


namespace find_number_l857_85717

theorem find_number (x : ℝ) (h : 20 * (x / 5) = 40) : x = 10 :=
by
  sorry

end find_number_l857_85717


namespace division_of_decimals_l857_85752

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end division_of_decimals_l857_85752


namespace owen_initial_turtles_l857_85775

variables (O J : ℕ)

-- Conditions
def johanna_turtles := J = O - 5
def owen_final_turtles := 2 * O + J / 2 = 50

-- Theorem statement
theorem owen_initial_turtles (h1 : johanna_turtles O J) (h2 : owen_final_turtles O J) : O = 21 :=
sorry

end owen_initial_turtles_l857_85775


namespace can_form_triangle_l857_85718

theorem can_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  (a = 7 ∧ b = 12 ∧ c = 17) → True :=
by
  sorry

end can_form_triangle_l857_85718


namespace outlet_pipe_emptying_time_l857_85777

theorem outlet_pipe_emptying_time :
  let rate1 := 1 / 18
  let rate2 := 1 / 20
  let fill_time := 0.08333333333333333
  ∃ x : ℝ, (rate1 + rate2 - 1 / x = 1 / fill_time) → x = 45 :=
by
  intro rate1 rate2 fill_time
  use 45
  intro h
  sorry

end outlet_pipe_emptying_time_l857_85777


namespace cryptarithmetic_proof_l857_85720

theorem cryptarithmetic_proof (A B C D : ℕ) 
  (h1 : A * B = 6) 
  (h2 : C = 2) 
  (h3 : A + B + D = 13) 
  (h4 : A + B + C = D) : 
  D = 6 :=
by
  sorry

end cryptarithmetic_proof_l857_85720


namespace zeros_of_quadratic_l857_85748

theorem zeros_of_quadratic : ∃ x : ℝ, x^2 - x - 2 = 0 -> (x = -1 ∨ x = 2) :=
by
  sorry

end zeros_of_quadratic_l857_85748


namespace successful_combinations_l857_85765

def herbs := 4
def gems := 6
def incompatible_combinations := 3

theorem successful_combinations : herbs * gems - incompatible_combinations = 21 := by
  sorry

end successful_combinations_l857_85765


namespace hundredth_odd_positive_integer_l857_85702

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l857_85702


namespace edward_toy_cars_l857_85743

def initial_amount : ℝ := 17.80
def cost_per_car : ℝ := 0.95
def cost_of_race_track : ℝ := 6.00
def remaining_amount : ℝ := 8.00

theorem edward_toy_cars : ∃ (n : ℕ), initial_amount - remaining_amount = n * cost_per_car + cost_of_race_track ∧ n = 4 := by
  sorry

end edward_toy_cars_l857_85743


namespace point_in_third_quadrant_l857_85785

theorem point_in_third_quadrant (m : ℝ) : 
  (-1 < 0 ∧ -2 + m < 0) ↔ (m < 2) :=
by 
  sorry

end point_in_third_quadrant_l857_85785


namespace initial_weight_cucumbers_l857_85736

theorem initial_weight_cucumbers (W : ℝ) (h1 : 0.99 * W + 0.01 * W = W) 
                                  (h2 : W = (50 - 0.98 * 50 + 0.01 * W))
                                  (h3 : 50 > 0) : W = 100 := 
sorry

end initial_weight_cucumbers_l857_85736


namespace triangle_intersect_sum_l857_85710

theorem triangle_intersect_sum (P Q R S T U : ℝ × ℝ) :
  P = (0, 8) →
  Q = (0, 0) →
  R = (10, 0) →
  S = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  T = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  ∃ U : ℝ × ℝ, 
    (U.1 = (P.1 + ((T.2 - P.2) / (T.1 - P.1)) * (U.1 - P.1)) ∧
     U.2 = (R.2 + ((S.2 - R.2) / (S.1 - R.1)) * (U.1 - R.1))) ∧
    (U.1 + U.2) = 6 :=
by
  sorry

end triangle_intersect_sum_l857_85710


namespace arithmetic_geometric_sequence_l857_85744

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_d : d ≠ 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d)
  (h_geo : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)) :
  (S 4 - S 2) / (S 5 - S 3) = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l857_85744


namespace problem_statement_l857_85703

theorem problem_statement (P : ℝ) (h : P = 1 / (Real.log 11 / Real.log 2) + 1 / (Real.log 11 / Real.log 3) + 1 / (Real.log 11 / Real.log 4) + 1 / (Real.log 11 / Real.log 5)) : 1 < P ∧ P < 2 := 
sorry

end problem_statement_l857_85703


namespace identify_worst_player_l857_85784

-- Define the participants
inductive Participant
| father
| sister
| son
| daughter

open Participant

-- Conditions
def participants : List Participant :=
  [father, sister, son, daughter]

def twins (p1 p2 : Participant) : Prop := 
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def not_same_sex (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def older_by_one_year (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father)

-- Question: who is the worst player?
def worst_player : Participant := sister

-- Proof statement
theorem identify_worst_player
  (h_twins : ∃ p1 p2, twins p1 p2)
  (h_not_same_sex : ∀ p1 p2, twins p1 p2 → not_same_sex p1 p2)
  (h_age_diff : ∀ p1 p2, twins p1 p2 → older_by_one_year p1 p2) :
  worst_player = sister :=
sorry

end identify_worst_player_l857_85784


namespace Jessie_final_weight_l857_85721

variable (initial_weight : ℝ) (loss_first_week : ℝ) (loss_rate_second_week : ℝ)
variable (loss_second_week : ℝ) (total_loss : ℝ) (final_weight : ℝ)

def Jessie_weight_loss_problem : Prop :=
  initial_weight = 92 ∧
  loss_first_week = 5 ∧
  loss_rate_second_week = 1.3 ∧
  loss_second_week = loss_rate_second_week * loss_first_week ∧
  total_loss = loss_first_week + loss_second_week ∧
  final_weight = initial_weight - total_loss ∧
  final_weight = 80.5

theorem Jessie_final_weight : Jessie_weight_loss_problem initial_weight loss_first_week loss_rate_second_week loss_second_week total_loss final_weight :=
by
  sorry

end Jessie_final_weight_l857_85721


namespace bricks_needed_for_wall_l857_85786

noncomputable def number_of_bricks_needed
    (brick_length : ℕ)
    (brick_width : ℕ)
    (brick_height : ℕ)
    (wall_length_m : ℕ)
    (wall_height_m : ℕ)
    (wall_thickness_cm : ℕ) : ℕ :=
  let wall_length_cm := wall_length_m * 100
  let wall_height_cm := wall_height_m * 100
  let wall_volume := wall_length_cm * wall_height_cm * wall_thickness_cm
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume -- This rounds up to the nearest whole number.

theorem bricks_needed_for_wall : number_of_bricks_needed 5 11 6 8 6 2 = 2910 :=
sorry

end bricks_needed_for_wall_l857_85786


namespace graph_passes_through_quadrants_l857_85759

theorem graph_passes_through_quadrants :
  ∀ x : ℝ, (4 * x + 2 > 0 → (x > 0)) ∨ (4 * x + 2 > 0 → (x < 0)) ∨ (4 * x + 2 < 0 → (x < 0)) :=
by
  intro x
  sorry

end graph_passes_through_quadrants_l857_85759


namespace Kaleb_got_rid_of_7_shirts_l857_85773

theorem Kaleb_got_rid_of_7_shirts (initial_shirts : ℕ) (remaining_shirts : ℕ) 
    (h1 : initial_shirts = 17) (h2 : remaining_shirts = 10) : initial_shirts - remaining_shirts = 7 := 
by
  sorry

end Kaleb_got_rid_of_7_shirts_l857_85773


namespace complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l857_85793

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of a set
def C_A : Set ℕ := U \ A
def C_B : Set ℕ := U \ B

-- Questions rephrased as theorem statements
theorem complement_of_A_in_U : C_A = {2, 4, 5} := by sorry
theorem intersection_of_A_and_B : A ∩ B = ∅ := by sorry
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by sorry
theorem union_of_complements_of_A_and_B : C_A ∪ C_B = U := by sorry

end complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l857_85793


namespace train_speed_l857_85787

theorem train_speed (distance time : ℝ) (h₁ : distance = 240) (h₂ : time = 4) : 
  ((distance / time) * 3.6) = 216 := 
by 
  rw [h₁, h₂] 
  sorry

end train_speed_l857_85787


namespace sum_in_Q_l857_85776

open Set

def is_set_P (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_set_Q (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k - 1
def is_set_M (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k + 1

variables (a b : ℤ)

theorem sum_in_Q (ha : is_set_P a) (hb : is_set_Q b) : is_set_Q (a + b) := 
sorry

end sum_in_Q_l857_85776


namespace closest_point_on_line_y_eq_3x_plus_2_l857_85722

theorem closest_point_on_line_y_eq_3x_plus_2 (x y : ℝ) :
  ∃ (p : ℝ × ℝ), p = (-1 / 2, 1 / 2) ∧ y = 3 * x + 2 ∧ p = (x, y) :=
by
-- We skip the proof steps and provide the statement only
sorry

end closest_point_on_line_y_eq_3x_plus_2_l857_85722


namespace cake_icing_l857_85749

/-- Define the cake conditions -/
structure Cake :=
  (dimension : ℕ)
  (small_cube_dimension : ℕ)
  (total_cubes : ℕ)
  (iced_faces : ℕ)

/-- Define the main theorem to prove the number of smaller cubes with icing on exactly two sides -/
theorem cake_icing (c : Cake) : 
  c.dimension = 5 ∧ c.small_cube_dimension = 1 ∧ c.total_cubes = 125 ∧ c.iced_faces = 4 →
  ∃ n, n = 20 :=
by
  sorry

end cake_icing_l857_85749


namespace replacement_paint_intensity_l857_85795

theorem replacement_paint_intensity 
  (P_original : ℝ) (P_new : ℝ) (f : ℝ) (I : ℝ) :
  P_original = 50 →
  P_new = 45 →
  f = 0.2 →
  0.8 * P_original + f * I = P_new →
  I = 25 :=
by
  intros
  sorry

end replacement_paint_intensity_l857_85795


namespace constant_speed_l857_85754

open Real

def total_trip_time := 50.0
def total_distance := 2790.0
def break_interval := 5.0
def break_duration := 0.5
def hotel_search_time := 0.5

theorem constant_speed :
  let number_of_breaks := total_trip_time / break_interval
  let total_break_time := number_of_breaks * break_duration
  let actual_driving_time := total_trip_time - total_break_time - hotel_search_time
  let constant_speed := total_distance / actual_driving_time
  constant_speed = 62.7 :=
by
  -- Provide proof here
  sorry

end constant_speed_l857_85754


namespace no_prime_solution_in_2_to_7_l857_85700

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_solution_in_2_to_7 : ∀ p : ℕ, is_prime p ∧ 2 ≤ p ∧ p ≤ 7 → (2 * p^3 - p^2 - 15 * p + 22) ≠ 0 :=
by
  intros p hp
  have h := hp.left
  sorry

end no_prime_solution_in_2_to_7_l857_85700


namespace pages_same_units_digit_l857_85706

theorem pages_same_units_digit (n : ℕ) (H : n = 63) : 
  ∃ (count : ℕ), count = 13 ∧ ∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → 
  (((x % 10) = ((n + 1 - x) % 10)) → (x = 2 ∨ x = 7 ∨ x = 12 ∨ x = 17 ∨ x = 22 ∨ x = 27 ∨ x = 32 ∨ x = 37 ∨ x = 42 ∨ x = 47 ∨ x = 52 ∨ x = 57 ∨ x = 62)) :=
by
  sorry

end pages_same_units_digit_l857_85706


namespace ratio_of_juniors_to_seniors_l857_85792

theorem ratio_of_juniors_to_seniors (j s : ℕ) (h : (1 / 3) * j = (2 / 3) * s) : j / s = 2 :=
by
  sorry

end ratio_of_juniors_to_seniors_l857_85792


namespace arun_gokul_age_subtract_l857_85716

theorem arun_gokul_age_subtract:
  ∃ x : ℕ, (60 - x) / 18 = 3 → x = 6 :=
sorry

end arun_gokul_age_subtract_l857_85716


namespace sum_q_p_evaluation_l857_85714

def p (x : Int) : Int := x^2 - 3
def q (x : Int) : Int := x - 2

def T : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

noncomputable def f (x : Int) : Int := q (p x)

noncomputable def sum_f_T : Int := List.sum (List.map f T)

theorem sum_q_p_evaluation :
  sum_f_T = 15 :=
by
  sorry

end sum_q_p_evaluation_l857_85714


namespace problem_statement_l857_85711

theorem problem_statement : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end problem_statement_l857_85711


namespace dividend_divisor_quotient_l857_85789

theorem dividend_divisor_quotient (x y z : ℕ) 
  (h1 : x = 6 * y) 
  (h2 : y = 6 * z) 
  (h3 : x = y * z) : 
  x = 216 ∧ y = 36 ∧ z = 6 := 
by
  sorry

end dividend_divisor_quotient_l857_85789


namespace taxi_trip_distance_l857_85750

theorem taxi_trip_distance
  (initial_fee : ℝ)
  (per_segment_charge : ℝ)
  (segment_distance : ℝ)
  (total_charge : ℝ)
  (segments_traveled : ℝ)
  (total_miles : ℝ) :
  initial_fee = 2.25 →
  per_segment_charge = 0.3 →
  segment_distance = 2/5 →
  total_charge = 4.95 →
  total_miles = segments_traveled * segment_distance →
  segments_traveled = (total_charge - initial_fee) / per_segment_charge →
  total_miles = 3.6 :=
by
  intros h_initial_fee h_per_segment_charge h_segment_distance h_total_charge h_total_miles h_segments_traveled
  sorry

end taxi_trip_distance_l857_85750


namespace functional_equation_unique_solution_l857_85774

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + f b + f (f c) = 0 → f a ^ 3 + b * f b ^ 2 + c ^ 2 * f c = 3 * a * b * c) →
  (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry

end functional_equation_unique_solution_l857_85774


namespace time_to_cross_man_l857_85712

-- Definitions based on the conditions
def speed_faster_train_kmph := 72 -- km per hour
def speed_slower_train_kmph := 36 -- km per hour
def length_faster_train_m := 200 -- meters

-- Convert speeds from km/h to m/s
def speed_faster_train_mps := speed_faster_train_kmph * 1000 / 3600 -- meters per second
def speed_slower_train_mps := speed_slower_train_kmph * 1000 / 3600 -- meters per second

-- Relative speed calculation
def relative_speed_mps := speed_faster_train_mps - speed_slower_train_mps -- meters per second

-- Prove the time to cross the man in the slower train
theorem time_to_cross_man : length_faster_train_m / relative_speed_mps = 20 := by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_man_l857_85712


namespace divisibility_by_24_l857_85739

theorem divisibility_by_24 (n : ℤ) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) :=
sorry

end divisibility_by_24_l857_85739


namespace ellipse_has_correct_equation_l857_85705

noncomputable def ellipse_Equation (a b : ℝ) (eccentricity : ℝ) (triangle_perimeter : ℝ) : Prop :=
  let c := a * eccentricity
  (a > b) ∧ (b > 0) ∧ (eccentricity = (Real.sqrt 3) / 3) ∧ (triangle_perimeter = 4 * (Real.sqrt 3)) ∧
  (a = Real.sqrt 3) ∧ (b^2 = a^2 - c^2) ∧
  (c = 1) ∧
  (b = Real.sqrt 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ ((x^2 / 3) + (y^2 / 2) = 1))

theorem ellipse_has_correct_equation : ellipse_Equation (Real.sqrt 3) (Real.sqrt 2) ((Real.sqrt 3) / 3) (4 * (Real.sqrt 3)) := 
sorry

end ellipse_has_correct_equation_l857_85705


namespace stock_and_bond_value_relation_l857_85762

-- Definitions for conditions
def more_valuable_shares : ℕ := 14
def less_valuable_shares : ℕ := 26
def face_value_bond : ℝ := 1000
def coupon_rate_bond : ℝ := 0.06
def discount_rate_bond : ℝ := 0.03
def total_assets_value : ℝ := 2106

-- Lean statement for the proof problem
theorem stock_and_bond_value_relation (x y : ℝ) 
    (h1 : face_value_bond * (1 - discount_rate_bond) = 970)
    (h2 : 27 * x + y = total_assets_value) :
    y = 2106 - 27 * x :=
by
  sorry

end stock_and_bond_value_relation_l857_85762


namespace harry_bought_l857_85732

-- Definitions based on the conditions
def initial_bottles := 35
def jason_bought := 5
def final_bottles := 24

-- Theorem stating the number of bottles Harry bought
theorem harry_bought :
  (initial_bottles - jason_bought) - final_bottles = 6 :=
by
  sorry

end harry_bought_l857_85732


namespace rectangle_length_l857_85758

theorem rectangle_length (side_length_square : ℝ) (width_rectangle : ℝ) (area_equal : ℝ) 
  (square_area : side_length_square * side_length_square = area_equal) 
  (rectangle_area : width_rectangle * (width_rectangle * length) = area_equal) : 
  length = 24 :=
by 
  sorry

end rectangle_length_l857_85758


namespace expected_audience_l857_85766

theorem expected_audience (Sat Mon Wed Fri : ℕ) (extra_people expected_total : ℕ)
  (h1 : Sat = 80)
  (h2 : Mon = 80 - 20)
  (h3 : Wed = Mon + 50)
  (h4 : Fri = Sat + Mon)
  (h5 : extra_people = 40)
  (h6 : expected_total = Sat + Mon + Wed + Fri - extra_people) :
  expected_total = 350 := 
sorry

end expected_audience_l857_85766


namespace a_leq_neg4_l857_85728

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def neg_p (a x : ℝ) : Prop := ¬(p a x)
def neg_q (x : ℝ) : Prop := ¬(q x)

theorem a_leq_neg4 (a : ℝ) (h_neg_p : ∀ x, neg_p a x → neg_q x) (h_a_neg : a < 0) :
  a ≤ -4 :=
sorry

end a_leq_neg4_l857_85728


namespace inverse_variation_l857_85742

variable (a b : ℝ)

theorem inverse_variation (h_ab : a * b = 400) :
  (b = 0.25 ∧ a = 1600) ∨ (b = 1.0 ∧ a = 400) :=
  sorry

end inverse_variation_l857_85742


namespace avg_salary_difference_l857_85707

theorem avg_salary_difference (factory_payroll : ℕ) (factory_workers : ℕ) (office_payroll : ℕ) (office_workers : ℕ)
  (h1 : factory_payroll = 30000) (h2 : factory_workers = 15)
  (h3 : office_payroll = 75000) (h4 : office_workers = 30) :
  (office_payroll / office_workers) - (factory_payroll / factory_workers) = 500 := by
  sorry

end avg_salary_difference_l857_85707


namespace max_tan_beta_l857_85701

theorem max_tan_beta (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2) 
  (h : α + β ≠ π / 2) (h_sin_cos : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β ≤ Real.sqrt 3 / 3 :=
sorry

end max_tan_beta_l857_85701


namespace exists_natural_number_n_l857_85769

theorem exists_natural_number_n (t : ℕ) (ht : t > 0) :
  ∃ n : ℕ, n > 1 ∧ Nat.gcd n t = 1 ∧ ∀ k : ℕ, k > 0 → ∃ m : ℕ, m > 1 → n^k + t ≠ m^m :=
by
  sorry

end exists_natural_number_n_l857_85769


namespace parameterization_solution_l857_85747

/-- Proof problem statement:
  Given the line equation y = 3x - 11 and its parameterization representation,
  the ordered pair (s, h) that satisfies both conditions is (3, 15).
-/
theorem parameterization_solution : ∃ s h : ℝ, 
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (s, -2) + t • (5, h)) ∧ y = 3 * x - 11) → 
  (s = 3 ∧ h = 15) :=
by
  -- introduce s and h 
  use 3
  use 15
  -- skip the proof
  sorry

end parameterization_solution_l857_85747


namespace tangent_lines_from_point_to_circle_l857_85735

theorem tangent_lines_from_point_to_circle : 
  ∀ (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ), 
  P = (2, 3) → C = (1, 1) → r = 1 → 
  (∃ k : ℝ, ((3 : ℝ) * P.1 - (4 : ℝ) * P.2 + 6 = 0) ∨ (P.1 = 2)) :=
by
  intros P C r hP hC hr
  sorry

end tangent_lines_from_point_to_circle_l857_85735


namespace two_people_same_birthday_l857_85797

noncomputable def population : ℕ := 6000000000

noncomputable def max_age_seconds : ℕ := 150 * 366 * 24 * 60 * 60

theorem two_people_same_birthday :
  ∃ (a b : ℕ) (ha : a < population) (hb : b < population) (hab : a ≠ b),
  (∃ (t : ℕ) (ht_a : t < max_age_seconds) (ht_b : t < max_age_seconds), true) :=
by
  sorry

end two_people_same_birthday_l857_85797


namespace range_of_k_l857_85755

theorem range_of_k (k : ℝ) : 
  (∃ a b : ℝ, x^2 + ky^2 = 2 ∧ a^2 = 2/k ∧ b^2 = 2 ∧ a > b) → 0 < k ∧ k < 1 :=
by {
  sorry
}

end range_of_k_l857_85755


namespace inequality_solution_range_l857_85723

theorem inequality_solution_range (x : ℝ) : (x^2 + 3*x - 10 < 0) ↔ (-5 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_range_l857_85723


namespace case_a_case_b_case_c_l857_85704

-- Definitions of game manageable
inductive Player
| First
| Second

def sum_of_dimensions (m n : Nat) : Nat := m + n

def is_winning_position (m n : Nat) : Player :=
  if sum_of_dimensions m n % 2 = 1 then Player.First else Player.Second

-- Theorem statements for the given grid sizes
theorem case_a : is_winning_position 9 10 = Player.First := 
  sorry

theorem case_b : is_winning_position 10 12 = Player.Second := 
  sorry

theorem case_c : is_winning_position 9 11 = Player.Second := 
  sorry

end case_a_case_b_case_c_l857_85704


namespace find_AC_l857_85782

theorem find_AC (AB DC AD : ℕ) (hAB : AB = 13) (hDC : DC = 20) (hAD : AD = 5) : 
  AC = 24.2 := 
sorry

end find_AC_l857_85782


namespace four_lines_set_l857_85741

-- Define the ⬩ operation
def clubsuit (a b : ℝ) := a^3 * b - a * b^3

-- Define the main theorem
theorem four_lines_set (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (y = 0 ∨ x = 0 ∨ y = x ∨ y = -x) :=
by sorry

end four_lines_set_l857_85741


namespace largest_divisor_of_consecutive_odd_integers_l857_85791

theorem largest_divisor_of_consecutive_odd_integers :
  ∀ (x : ℤ), (∃ (d : ℤ) (m : ℤ), d = 48 ∧ (x * (x + 2) * (x + 4) * (x + 6)) = d * m) :=
by 
-- We assert that for any integer x, 48 always divides the product of
-- four consecutive odd integers starting from x
sorry

end largest_divisor_of_consecutive_odd_integers_l857_85791


namespace pictures_deleted_l857_85746

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 15) 
  (h2 : museum_pics = 18) 
  (h3 : remaining_pics = 2) : 
  zoo_pics + museum_pics - remaining_pics = 31 :=
by 
  sorry

end pictures_deleted_l857_85746


namespace difference_is_correct_l857_85796

-- Define the digits
def digits : List ℕ := [9, 2, 1, 5]

-- Define the largest number that can be formed by these digits
def largestNumber : ℕ :=
  1000 * 9 + 100 * 5 + 10 * 2 + 1 * 1

-- Define the smallest number that can be formed by these digits
def smallestNumber : ℕ :=
  1000 * 1 + 100 * 2 + 10 * 5 + 1 * 9

-- Define the correct difference
def difference : ℕ :=
  largestNumber - smallestNumber

-- Theorem statement
theorem difference_is_correct : difference = 8262 :=
by
  sorry

end difference_is_correct_l857_85796


namespace weight_measurement_l857_85772

theorem weight_measurement :
  ∀ (w : Set ℕ), w = {1, 3, 9, 27} → (∀ n ∈ w, ∃ k, k = n ∧ k ∈ w) →
  ∃ (num_sets : ℕ), num_sets = 41 := by
  intros w hw hcomb
  sorry

end weight_measurement_l857_85772


namespace no_positive_integer_makes_sum_prime_l857_85763

theorem no_positive_integer_makes_sum_prime : ¬ ∃ n : ℕ, 0 < n ∧ Prime (4^n + n^4) :=
by
  sorry

end no_positive_integer_makes_sum_prime_l857_85763


namespace solve_for_k_l857_85783

theorem solve_for_k (k : ℝ) (h : 2 * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : k = 65 := 
by
  sorry

end solve_for_k_l857_85783


namespace translate_parabola_l857_85799

theorem translate_parabola (x y : ℝ) :
  (y = 2 * x^2 + 3) →
  (∃ x y, y = 2 * (x - 3)^2 + 5) :=
sorry

end translate_parabola_l857_85799


namespace shortest_time_between_ships_l857_85757

theorem shortest_time_between_ships 
  (AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) (angle_ABA' : ℝ) : (AB = 10) → (speed_A = 4) → (speed_B = 6) → (angle_ABA' = 60) →
  ∃ t : ℝ, (t = 150/7 / 60) :=
by
  intro hAB hSpeedA hSpeedB hAngle
  sorry

end shortest_time_between_ships_l857_85757


namespace Nick_total_money_l857_85756

variable (nickels : Nat) (dimes : Nat) (quarters : Nat)
variable (value_nickel : Nat := 5) (value_dime : Nat := 10) (value_quarter : Nat := 25)

def total_value (nickels dimes quarters : Nat) : Nat :=
  nickels * value_nickel + dimes * value_dime + quarters * value_quarter

theorem Nick_total_money :
  total_value 6 2 1 = 75 := by
  sorry

end Nick_total_money_l857_85756


namespace rowing_upstream_distance_l857_85719

theorem rowing_upstream_distance 
  (b s t d1 d2 : ℝ)
  (h1 : s = 7)
  (h2 : d1 = 72)
  (h3 : t = 3)
  (h4 : d1 = (b + s) * t) :
  d2 = (b - s) * t → d2 = 30 :=
by 
  intros h5
  sorry

end rowing_upstream_distance_l857_85719


namespace least_comic_books_l857_85764

theorem least_comic_books (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 1) : n = 17 :=
sorry

end least_comic_books_l857_85764


namespace total_number_of_girls_is_13_l857_85779

def number_of_girls (n : ℕ) (B : ℕ) : Prop :=
  ∃ A : ℕ, (A = B - 5) ∧ (A = B + 8)

theorem total_number_of_girls_is_13 (n : ℕ) (B : ℕ) :
  number_of_girls n B → n = 13 :=
by
  intro h
  sorry

end total_number_of_girls_is_13_l857_85779


namespace determine_f4_l857_85740

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem determine_f4 (f : ℝ → ℝ) (h_odd : odd_function f) (h_f_neg : ∀ x, x < 0 → f x = x * (2 - x)) : f 4 = 24 :=
by
  sorry

end determine_f4_l857_85740
