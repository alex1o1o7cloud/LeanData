import Mathlib

namespace solve_for_q_l1662_166294

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l1662_166294


namespace base_n_representation_l1662_166293

theorem base_n_representation (n : ℕ) (b : ℕ) (h₀ : 8 < n) (h₁ : ∃ b, (n : ℤ)^2 - (n+8) * (n : ℤ) + b = 0) : 
  b = 8 * n :=
by
  sorry

end base_n_representation_l1662_166293


namespace holes_in_compartment_l1662_166231

theorem holes_in_compartment :
  ∀ (rect : Type) (holes : ℕ) (compartments : ℕ),
  compartments = 9 →
  holes = 20 →
  (∃ (compartment : rect ) (n : ℕ), n ≥ 3) :=
by
  intros rect holes compartments h_compartments h_holes
  sorry

end holes_in_compartment_l1662_166231


namespace minimum_value_Q_l1662_166243

theorem minimum_value_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 47 := 
  sorry

end minimum_value_Q_l1662_166243


namespace range_of_b_min_value_a_add_b_min_value_ab_l1662_166206

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l1662_166206


namespace inequality_1_inequality_2_inequality_3_inequality_4_l1662_166259

-- Definition for the first problem
theorem inequality_1 (x : ℝ) : |2 * x - 1| < 15 ↔ (-7 < x ∧ x < 8) := by
  sorry
  
-- Definition for the second problem
theorem inequality_2 (x : ℝ) : x^2 + 6 * x - 16 < 0 ↔ (-8 < x ∧ x < 2) := by
  sorry

-- Definition for the third problem
theorem inequality_3 (x : ℝ) : |2 * x + 1| > 13 ↔ (x < -7 ∨ x > 6) := by
  sorry

-- Definition for the fourth problem
theorem inequality_4 (x : ℝ) : x^2 - 2 * x > 0 ↔ (x < 0 ∨ x > 2) := by
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l1662_166259


namespace present_age_of_younger_l1662_166237

-- Definition based on conditions
variable (y e : ℕ)
variable (h1 : e = y + 20)
variable (h2 : e - 8 = 5 * (y - 8))

-- Statement to be proven
theorem present_age_of_younger (y e: ℕ) (h1: e = y + 20) (h2: e - 8 = 5 * (y - 8)) : y = 13 := 
by 
  sorry

end present_age_of_younger_l1662_166237


namespace probability_red_or_white_is_7_over_10_l1662_166271

/-
A bag consists of 20 marbles, of which 6 are blue, 9 are red, and the remainder are white.
If Lisa is to select a marble from the bag at random, prove that the probability that the
marble will be red or white is 7/10.
-/
def num_marbles : ℕ := 20
def num_blue : ℕ := 6
def num_red : ℕ := 9
def num_white : ℕ := num_marbles - (num_blue + num_red)

def probability_red_or_white : ℚ :=
  (num_red + num_white) / num_marbles

theorem probability_red_or_white_is_7_over_10 :
  probability_red_or_white = 7 / 10 := 
sorry

end probability_red_or_white_is_7_over_10_l1662_166271


namespace inequality_solution_l1662_166212

theorem inequality_solution (x : ℝ) : (5 < x ∧ x ≤ 6) ↔ (x-3)/(x-5) ≥ 3 :=
by
  sorry

end inequality_solution_l1662_166212


namespace eq_op_op_op_92_l1662_166284

noncomputable def opN (N : ℝ) : ℝ := 0.75 * N + 2

theorem eq_op_op_op_92 : opN (opN (opN 92)) = 43.4375 :=
by
  sorry

end eq_op_op_op_92_l1662_166284


namespace log_expression_value_l1662_166255

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_expression_value :
  log_base 3 32 * log_base 4 9 - log_base 2 (3/4) + log_base 2 6 = 8 := 
by 
  sorry

end log_expression_value_l1662_166255


namespace defective_probability_bayesian_probabilities_l1662_166218

noncomputable def output_proportion_A : ℝ := 0.25
noncomputable def output_proportion_B : ℝ := 0.35
noncomputable def output_proportion_C : ℝ := 0.40

noncomputable def defect_rate_A : ℝ := 0.05
noncomputable def defect_rate_B : ℝ := 0.04
noncomputable def defect_rate_C : ℝ := 0.02

noncomputable def probability_defective : ℝ :=
  output_proportion_A * defect_rate_A +
  output_proportion_B * defect_rate_B +
  output_proportion_C * defect_rate_C 

theorem defective_probability :
  probability_defective = 0.0345 := 
  by sorry

noncomputable def P_A_given_defective : ℝ :=
  (output_proportion_A * defect_rate_A) / probability_defective

noncomputable def P_B_given_defective : ℝ :=
  (output_proportion_B * defect_rate_B) / probability_defective

noncomputable def P_C_given_defective : ℝ :=
  (output_proportion_C * defect_rate_C) / probability_defective

theorem bayesian_probabilities :
  P_A_given_defective = 25 / 69 ∧
  P_B_given_defective = 28 / 69 ∧
  P_C_given_defective = 16 / 69 :=
  by sorry

end defective_probability_bayesian_probabilities_l1662_166218


namespace alpha_beta_diff_l1662_166220

theorem alpha_beta_diff 
  (α β : ℝ)
  (h1 : α + β = 17)
  (h2 : α * β = 70) : |α - β| = 3 :=
by
  sorry

end alpha_beta_diff_l1662_166220


namespace length_of_AB_l1662_166269

-- Definitions based on given conditions:
variables (AB BC CD DE AE AC : ℕ)
variables (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21)

-- The theorem stating the length of AB given the conditions.
theorem length_of_AB (AB BC CD DE AE AC : ℕ)
  (h1 : BC = 3 * CD) (h2 : DE = 8) (h3 : AC = 11) (h4 : AE = 21) : AB = 5 := by
  sorry

end length_of_AB_l1662_166269


namespace my_age_is_five_times_son_age_l1662_166275

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l1662_166275


namespace seq_1000_eq_2098_l1662_166209

-- Define the sequence a_n
def seq (n : ℕ) : ℤ := sorry

-- Initial conditions
axiom a1 : seq 1 = 100
axiom a2 : seq 2 = 101

-- Recurrence relation condition
axiom recurrence_relation : ∀ n : ℕ, 1 ≤ n → seq n + seq (n+1) + seq (n+2) = 2 * ↑n + 3

-- Main theorem to prove
theorem seq_1000_eq_2098 : seq 1000 = 2098 :=
by {
  sorry
}

end seq_1000_eq_2098_l1662_166209


namespace ferry_journey_time_difference_l1662_166277

/-
  Problem statement:
  Prove that the journey of ferry Q is 1 hour longer than the journey of ferry P,
  given the following conditions:
  1. Ferry P travels for 3 hours at 6 kilometers per hour.
  2. Ferry Q takes a route that is two times longer than ferry P.
  3. Ferry P is slower than ferry Q by 3 kilometers per hour.
-/

theorem ferry_journey_time_difference :
  let speed_P := 6
  let time_P := 3
  let distance_P := speed_P * time_P
  let distance_Q := 2 * distance_P
  let speed_diff := 3
  let speed_Q := speed_P + speed_diff
  let time_Q := distance_Q / speed_Q
  time_Q - time_P = 1 :=
by
  sorry

end ferry_journey_time_difference_l1662_166277


namespace magnitude_of_angle_A_range_of_b_plus_c_l1662_166251

--- Definitions for the conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition a / (sqrt 3 * cos A) = c / sin C
axiom condition1 : a / (Real.sqrt 3 * Real.cos A) = c / Real.sin C

-- Given a = 6
axiom condition2 : a = 6

-- Conditions for sides b and c being positive
axiom condition3 : b > 0
axiom condition4 : c > 0
-- Condition for triangle inequality
axiom condition5 : b + c > a

-- Part (I) Find the magnitude of angle A
theorem magnitude_of_angle_A : A = Real.pi / 3 :=
by
  sorry

-- Part (II) Determine the range of values for b + c given a = 6
theorem range_of_b_plus_c : 6 < b + c ∧ b + c ≤ 12 :=
by
  sorry

end magnitude_of_angle_A_range_of_b_plus_c_l1662_166251


namespace sum_of_fractions_l1662_166241

theorem sum_of_fractions : 
  (7 / 8 + 3 / 4) = (13 / 8) :=
by
  sorry

end sum_of_fractions_l1662_166241


namespace sin_neg_three_pi_over_four_l1662_166228

theorem sin_neg_three_pi_over_four : Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_three_pi_over_four_l1662_166228


namespace final_price_l1662_166203

def initial_price : ℝ := 200
def discount_morning : ℝ := 0.40
def increase_noon : ℝ := 0.25
def discount_afternoon : ℝ := 0.20

theorem final_price : 
  let price_after_morning := initial_price * (1 - discount_morning)
  let price_after_noon := price_after_morning * (1 + increase_noon)
  let final_price := price_after_noon * (1 - discount_afternoon)
  final_price = 120 := 
by
  sorry

end final_price_l1662_166203


namespace sonika_initial_deposit_l1662_166242

variable (P R : ℝ)

theorem sonika_initial_deposit :
  (P + (P * R * 3) / 100 = 9200) → (P + (P * (R + 2.5) * 3) / 100 = 9800) → P = 8000 :=
by
  intros h1 h2
  sorry

end sonika_initial_deposit_l1662_166242


namespace simple_interest_principal_l1662_166227

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end simple_interest_principal_l1662_166227


namespace find_linear_in_two_variables_l1662_166286

def is_linear_in_two_variables (eq : String) : Bool :=
  eq = "x=y+1"

theorem find_linear_in_two_variables :
  (is_linear_in_two_variables "4xy=2" = false) ∧
  (is_linear_in_two_variables "1-x=7" = false) ∧
  (is_linear_in_two_variables "x^2+2y=-2" = false) ∧
  (is_linear_in_two_variables "x=y+1" = true) :=
by
  sorry

end find_linear_in_two_variables_l1662_166286


namespace general_term_sequence_l1662_166205

/--
Given the sequence a : ℕ → ℝ such that a 0 = 1/2,
a 1 = 1/4,
a 2 = -1/8,
a 3 = 1/16,
and we observe that
a n = (-(1/2))^n,
prove that this formula holds for all n : ℕ.
-/
theorem general_term_sequence (a : ℕ → ℝ) :
  (∀ n, a n = (-(1/2))^n) :=
sorry

end general_term_sequence_l1662_166205


namespace combined_weight_l1662_166264

-- Define the conditions
variables (Ron_weight Roger_weight Rodney_weight : ℕ)

-- Define the conditions as Lean propositions
def conditions : Prop :=
  Rodney_weight = 2 * Roger_weight ∧ 
  Roger_weight = 4 * Ron_weight - 7 ∧ 
  Rodney_weight = 146

-- Define the proof goal
def proof_goal : Prop :=
  Rodney_weight + Roger_weight + Ron_weight = 239

theorem combined_weight (Ron_weight Roger_weight Rodney_weight : ℕ) (h : conditions Ron_weight Roger_weight Rodney_weight) : 
  proof_goal Ron_weight Roger_weight Rodney_weight :=
sorry

end combined_weight_l1662_166264


namespace prove_sums_l1662_166214

-- Given conditions
def condition1 (a b : ℤ) : Prop := ∀ x : ℝ, (x + a) * (x + b) = x^2 + 9 * x + 14
def condition2 (b c : ℤ) : Prop := ∀ x : ℝ, (x + b) * (x - c) = x^2 + 7 * x - 30

-- We need to prove that a + b + c = 15
theorem prove_sums (a b c : ℤ) (h1: condition1 a b) (h2: condition2 b c) : a + b + c = 15 := 
sorry

end prove_sums_l1662_166214


namespace sum_of_remainders_l1662_166225

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := 
by 
  sorry

end sum_of_remainders_l1662_166225


namespace total_population_l1662_166276

-- Define the conditions
variables (T G Td Lb : ℝ)

-- Given conditions and the result
def conditions : Prop :=
  G = 1 / 2 * T ∧
  Td = 0.60 * G ∧
  Lb = 16000 ∧
  T = Td + G + Lb

-- Problem statement: Prove that the total population T is 80000
theorem total_population (h : conditions T G Td Lb) : T = 80000 :=
by
  sorry

end total_population_l1662_166276


namespace radius_area_tripled_l1662_166261

theorem radius_area_tripled (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
by {
  sorry
}

end radius_area_tripled_l1662_166261


namespace c_minus_a_equals_90_l1662_166281

variable (a b c : ℝ)

def average_a_b (a b : ℝ) : Prop := (a + b) / 2 = 45
def average_b_c (b c : ℝ) : Prop := (b + c) / 2 = 90

theorem c_minus_a_equals_90
  (h1 : average_a_b a b)
  (h2 : average_b_c b c) :
  c - a = 90 :=
  sorry

end c_minus_a_equals_90_l1662_166281


namespace find_perimeter_correct_l1662_166256

noncomputable def find_perimeter (L W : ℝ) (x : ℝ) :=
  L * W = (L + 6) * (W - 2) ∧
  L * W = (L - 12) * (W + 6) ∧
  x = 2 * (L + W)

theorem find_perimeter_correct : ∀ (L W : ℝ), L * W = (L + 6) * (W - 2) → 
                                      L * W = (L - 12) * (W + 6) → 
                                      2 * (L + W) = 132 :=
sorry

end find_perimeter_correct_l1662_166256


namespace income_of_sixth_member_l1662_166274

def income_member1 : ℝ := 11000
def income_member2 : ℝ := 15000
def income_member3 : ℝ := 10000
def income_member4 : ℝ := 9000
def income_member5 : ℝ := 13000
def number_of_members : ℕ := 6
def average_income : ℝ := 12000
def total_income_of_five_members := income_member1 + income_member2 + income_member3 + income_member4 + income_member5

theorem income_of_sixth_member :
  6 * average_income - total_income_of_five_members = 14000 := by
  sorry

end income_of_sixth_member_l1662_166274


namespace range_of_k_l1662_166211

theorem range_of_k {x y k : ℝ} :
  (∀ x y, 2 * x - y ≤ 1 ∧ x + y ≥ 2 ∧ y - x ≤ 2) →
  (z = k * x + 2 * y) →
  (∀ (x y : ℝ), z = k * x + 2 * y → (x = 1) ∧ (y = 1)) →
  -4 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l1662_166211


namespace missing_digit_is_4_l1662_166238

theorem missing_digit_is_4 (x : ℕ) (hx : 7385 = 7380 + x + 5)
  (hdiv : (7 + 3 + 8 + x + 5) % 9 = 0) : x = 4 :=
by
  sorry

end missing_digit_is_4_l1662_166238


namespace total_number_of_items_in_base10_l1662_166210

theorem total_number_of_items_in_base10 : 
  let clay_tablets := (2 * 5^0 + 3 * 5^1 + 4 * 5^2 + 1 * 5^3)
  let bronze_sculptures := (1 * 5^0 + 4 * 5^1 + 0 * 5^2 + 2 * 5^3)
  let stone_carvings := (2 * 5^0 + 3 * 5^1 + 2 * 5^2)
  let total_items := clay_tablets + bronze_sculptures + stone_carvings
  total_items = 580 := by
  sorry

end total_number_of_items_in_base10_l1662_166210


namespace min_trips_calculation_l1662_166240

noncomputable def min_trips (total_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  ⌈total_weight / truck_capacity⌉₊

theorem min_trips_calculation : min_trips 18.5 3.9 = 5 :=
by
  -- Proof goes here
  sorry

end min_trips_calculation_l1662_166240


namespace alyssa_games_next_year_l1662_166229

/-- Alyssa went to 11 games this year -/
def games_this_year : ℕ := 11

/-- Alyssa went to 13 games last year -/
def games_last_year : ℕ := 13

/-- Alyssa will go to a total of 39 games -/
def total_games : ℕ := 39

/-- Alyssa plans to go to 15 games next year -/
theorem alyssa_games_next_year : 
  games_this_year + games_last_year <= total_games ∧
  total_games - (games_this_year + games_last_year) = 15 := by {
  sorry
}

end alyssa_games_next_year_l1662_166229


namespace gcd_4536_13440_216_l1662_166224

def gcd_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_4536_13440_216 : gcd_of_three_numbers 4536 13440 216 = 216 :=
by
  sorry

end gcd_4536_13440_216_l1662_166224


namespace complex_quadrant_l1662_166217

open Complex

theorem complex_quadrant :
  let z := (1 - I) * (3 + I)
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l1662_166217


namespace monitor_height_l1662_166246

theorem monitor_height (width circumference : ℕ) (h_width : width = 12) (h_circumference : circumference = 38) :
  2 * (width + 7) = circumference :=
by
  sorry

end monitor_height_l1662_166246


namespace female_cows_percentage_l1662_166208

theorem female_cows_percentage (TotalCows PregnantFemaleCows : Nat) (PregnantPercentage : ℚ)
    (h1 : TotalCows = 44)
    (h2 : PregnantFemaleCows = 11)
    (h3 : PregnantPercentage = 0.50) :
    (PregnantFemaleCows / PregnantPercentage / TotalCows) * 100 = 50 := 
sorry

end female_cows_percentage_l1662_166208


namespace quadrilateral_choices_l1662_166230

theorem quadrilateral_choices :
  let available_rods : List ℕ := (List.range' 1 41).diff [5, 12, 20]
  let valid_rods := available_rods.filter (λ x => 4 ≤ x ∧ x ≤ 36)
  valid_rods.length = 30 := sorry

end quadrilateral_choices_l1662_166230


namespace kite_ratio_equality_l1662_166270

-- Definitions for points, lines, and conditions in the geometric setup
variables {Point : Type*} [MetricSpace Point]

-- Assuming A, B, C, D, P, E, F, G, H, I, J are points
variable (A B C D P E F G H I J : Point)

-- Conditions based on the problem
variables (AB_eq_AD : dist A B = dist A D)
          (BC_eq_CD : dist B C = dist C D)
          (on_BD : P ∈ line B D)
          (line_PE_inter_AD : E ∈ line P E ∧ E ∈ line A D)
          (line_PF_inter_BC : F ∈ line P F ∧ F ∈ line B C)
          (line_PG_inter_AB : G ∈ line P G ∧ G ∈ line A B)
          (line_PH_inter_CD : H ∈ line P H ∧ H ∈ line C D)
          (GF_inter_BD_at_I : I ∈ line G F ∧ I ∈ line B D)
          (EH_inter_BD_at_J : J ∈ line E H ∧ J ∈ line B D)

-- The statement to prove
theorem kite_ratio_equality :
  dist P I / dist P B = dist P J / dist P D := sorry

end kite_ratio_equality_l1662_166270


namespace rich_knight_l1662_166207

-- Definitions for the problem
inductive Status
| knight  -- Always tells the truth
| knave   -- Always lies

def tells_truth (s : Status) : Prop := 
  s = Status.knight

def lies (s : Status) : Prop := 
  s = Status.knave

def not_poor (s : Status) : Prop := 
  s = Status.knight ∨ s = Status.knave -- Knights can either be poor or wealthy

def wealthy (s : Status) : Prop :=
  s = Status.knight

-- Statement to be proven
theorem rich_knight (s : Status) (h_truth : tells_truth s) (h_not_poor : not_poor s) : wealthy s :=
by
  sorry

end rich_knight_l1662_166207


namespace max_value_of_function_l1662_166200

noncomputable def function_y (x : ℝ) : ℝ := x + Real.sin x

theorem max_value_of_function : 
  ∀ (a b : ℝ), a = 0 → b = Real.pi → 
  (∀ x : ℝ, x ∈ Set.Icc a b → x + Real.sin x ≤ Real.pi) :=
by
  intros a b ha hb x hx
  sorry

end max_value_of_function_l1662_166200


namespace circle_diameter_l1662_166249

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l1662_166249


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l1662_166239

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l1662_166239


namespace original_price_of_RAM_l1662_166213

variables (P : ℝ)

-- Conditions extracted from the problem statement
def priceAfterFire (P : ℝ) : ℝ := 1.30 * P
def priceAfterDecrease (P : ℝ) : ℝ := 1.04 * P

-- The given current price
axiom current_price : priceAfterDecrease P = 52

-- Theorem to prove the original price P
theorem original_price_of_RAM : P = 50 :=
sorry

end original_price_of_RAM_l1662_166213


namespace soda_cans_ratio_l1662_166265

theorem soda_cans_ratio
  (initial_cans : ℕ := 22)
  (cans_taken : ℕ := 6)
  (final_cans : ℕ := 24)
  (x : ℚ := 1 / 2)
  (cans_left : ℕ := 16)
  (cans_bought : ℕ := 16 * 1 / 2) :
  (cans_bought / cans_left : ℚ) = 1 / 2 :=
sorry

end soda_cans_ratio_l1662_166265


namespace find_parallelepiped_dimensions_l1662_166226

theorem find_parallelepiped_dimensions :
  ∃ (x y z : ℕ),
    (x * y * z = 2 * (x * y + y * z + z * x)) ∧
    (x = 6 ∧ y = 6 ∧ z = 6 ∨
     x = 5 ∧ y = 5 ∧ z = 10 ∨
     x = 4 ∧ y = 8 ∧ z = 8 ∨
     x = 3 ∧ y = 12 ∧ z = 12 ∨
     x = 3 ∧ y = 7 ∧ z = 42 ∨
     x = 3 ∧ y = 8 ∧ z = 24 ∨
     x = 3 ∧ y = 9 ∧ z = 18 ∨
     x = 3 ∧ y = 10 ∧ z = 15 ∨
     x = 4 ∧ y = 5 ∧ z = 20 ∨
     x = 4 ∧ y = 6 ∧ z = 12) :=
by
  sorry

end find_parallelepiped_dimensions_l1662_166226


namespace fish_distribution_l1662_166254

theorem fish_distribution 
  (fish_caught : ℕ)
  (eyes_per_fish : ℕ := 2)
  (total_eyes : ℕ := 24)
  (people : ℕ := 3)
  (eyes_eaten_by_dog : ℕ := 2)
  (eyes_eaten_by_oomyapeck : ℕ := 22)
  (oomyapeck_total_eyes : eyes_eaten_by_oomyapeck + eyes_eaten_by_dog = total_eyes)
  (fish_per_person := fish_caught / people)
  (fish_eyes_relation : total_eyes = eyes_per_fish * fish_caught) :
  fish_per_person = 4 := by
  sorry

end fish_distribution_l1662_166254


namespace c_is_perfect_square_or_not_even_c_cannot_be_even_l1662_166216

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem c_is_perfect_square_or_not_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_odd : c % 2 = 1) : is_perfect_square c :=
sorry

theorem c_cannot_be_even 
  (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b))
  (h_even : c % 2 = 0) : false :=
sorry

end c_is_perfect_square_or_not_even_c_cannot_be_even_l1662_166216


namespace range_of_f_l1662_166287

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ y, (∃ x, x ∈ Set.Icc (-1:ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc 0 (Real.pi^4 / 8) :=
sorry

end range_of_f_l1662_166287


namespace rectangle_side_length_l1662_166258

theorem rectangle_side_length (x : ℝ) (h1 : 0 < x) (h2 : 2 * (x + 6) = 40) : x = 14 :=
by
  sorry

end rectangle_side_length_l1662_166258


namespace bob_speed_lt_40_l1662_166296

theorem bob_speed_lt_40 (v_b v_a : ℝ) (h1 : v_a > 45) (h2 : 180 / v_a < 180 / v_b - 0.5) :
  v_b < 40 :=
by
  -- Variables and constants
  let distance := 180
  let min_speed_alice := 45
  -- Conditions
  have h_distance := distance
  have h_min_speed_alice := min_speed_alice
  have h_time_alice := (distance : ℝ) / v_a
  have h_time_bob := (distance : ℝ) / v_b
  -- Given conditions inequalities
  have ineq := h2
  have alice_min_speed := h1
  -- Now apply these facts and derived inequalities to prove bob_speed_lt_40
  sorry

end bob_speed_lt_40_l1662_166296


namespace composite_has_at_least_three_divisors_l1662_166248

def is_composite (n : ℕ) : Prop := ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_divisors (n : ℕ) (h : is_composite n) : ∃ a b c, a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c :=
sorry

end composite_has_at_least_three_divisors_l1662_166248


namespace quadratic_has_two_distinct_real_roots_l1662_166233

/-- The quadratic equation x^2 + 2x - 3 = 0 has two distinct real roots. -/
theorem quadratic_has_two_distinct_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ ^ 2 + 2 * x₁ - 3 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 3 = 0) := by
sorry

end quadratic_has_two_distinct_real_roots_l1662_166233


namespace tan_alpha_sin_double_angle_l1662_166253

theorem tan_alpha_sin_double_angle (α : ℝ) (h : Real.tan α = 3/4) : Real.sin (2 * α) = 24/25 :=
by
  sorry

end tan_alpha_sin_double_angle_l1662_166253


namespace sum_of_roots_l1662_166222

theorem sum_of_roots (a β : ℝ) 
  (h1 : a^2 - 2 * a = 1) 
  (h2 : β^2 - 2 * β - 1 = 0) 
  (hne : a ≠ β) 
  : a + β = 2 := 
sorry

end sum_of_roots_l1662_166222


namespace license_plate_increase_l1662_166247

def old_license_plates : ℕ := 26 * (10^5)

def new_license_plates : ℕ := 26^2 * (10^4)

theorem license_plate_increase :
  (new_license_plates / old_license_plates : ℝ) = 2.6 := by
  sorry

end license_plate_increase_l1662_166247


namespace max_not_sum_S_l1662_166272

def S : Set ℕ := {n | ∃ k : ℕ, n = 10^k + 1000}

theorem max_not_sum_S : ∀ x : ℕ, (∀ y ∈ S, ∃ m : ℕ, x ≠ m * y) ↔ x = 34999 := by
  sorry

end max_not_sum_S_l1662_166272


namespace trays_from_second_table_l1662_166267

def trays_per_trip : ℕ := 4
def trips : ℕ := 9
def trays_from_first_table : ℕ := 20

theorem trays_from_second_table :
  trays_per_trip * trips - trays_from_first_table = 16 :=
by
  sorry

end trays_from_second_table_l1662_166267


namespace action_figure_value_l1662_166235

theorem action_figure_value (
    V1 V2 V3 V4 : ℝ
) : 5 * 15 = 75 ∧ 
    V1 - 5 + V2 - 5 + V3 - 5 + V4 - 5 + (20 - 5) = 55 ∧
    V1 + V2 + V3 + V4 + 20 = 80 → 
    ∀ i, i = 15 := by
    sorry

end action_figure_value_l1662_166235


namespace multiplicative_inverse_l1662_166244

def A : ℕ := 123456
def B : ℕ := 162738
def N : ℕ := 503339
def modulo : ℕ := 1000000

theorem multiplicative_inverse :
  (A * B * N) % modulo = 1 :=
by
  -- placeholder for proof
  sorry

end multiplicative_inverse_l1662_166244


namespace four_cells_different_colors_l1662_166215

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l1662_166215


namespace karl_total_income_correct_l1662_166232

noncomputable def price_of_tshirt : ℝ := 5
noncomputable def price_of_pants : ℝ := 4
noncomputable def price_of_skirt : ℝ := 6
noncomputable def price_of_refurbished_tshirt : ℝ := price_of_tshirt / 2

noncomputable def discount_for_skirts (n : ℕ) : ℝ := (n / 2) * 2 * price_of_skirt * 0.10
noncomputable def discount_for_tshirts (n : ℕ) : ℝ := (n / 5) * 5 * price_of_tshirt * 0.20
noncomputable def discount_for_pants (n : ℕ) : ℝ := 0 -- accounted for in quantity

noncomputable def sales_tax (amount : ℝ) : ℝ := amount * 0.08

noncomputable def total_income : ℝ := 
  let tshirt_income := 8 * price_of_tshirt + 7 * price_of_refurbished_tshirt - discount_for_tshirts 15
  let pants_income := 6 * price_of_pants - discount_for_pants 6
  let skirts_income := 12 * price_of_skirt - discount_for_skirts 12
  let income_before_tax := tshirt_income + pants_income + skirts_income
  income_before_tax + sales_tax income_before_tax

theorem karl_total_income_correct : total_income = 141.80 :=
by
  sorry

end karl_total_income_correct_l1662_166232


namespace team_a_completion_rate_l1662_166263

theorem team_a_completion_rate :
  ∃ x : ℝ, (9000 / x - 9000 / (1.5 * x) = 15) ∧ x = 200 :=
by {
  sorry
}

end team_a_completion_rate_l1662_166263


namespace sought_line_eq_l1662_166202

-- Definitions used in the conditions
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def line_perpendicular (x y : ℝ) : Prop := x + y = 0
def center_of_circle : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem sought_line_eq (x y : ℝ) :
  (circle_eq x y ∧ line_perpendicular x y ∧ (x, y) = center_of_circle) →
  (x + y + 1 = 0) :=
by
  sorry

end sought_line_eq_l1662_166202


namespace servant_leaves_after_nine_months_l1662_166278

-- Definitions based on conditions
def yearly_salary : ℕ := 90 + 90
def monthly_salary : ℕ := yearly_salary / 12
def amount_received : ℕ := 45 + 90

-- The theorem to prove
theorem servant_leaves_after_nine_months :
    amount_received / monthly_salary = 9 :=
by
  -- Using the provided conditions, we establish the equality we need.
  sorry

end servant_leaves_after_nine_months_l1662_166278


namespace domain_function_1_domain_function_2_domain_function_3_l1662_166234

-- Define the conditions and the required domain equivalence in Lean 4
-- Problem (1)
theorem domain_function_1 (x : ℝ): x + 2 ≠ 0 ∧ x + 5 ≥ 0 ↔ x ≥ -5 ∧ x ≠ -2 := 
sorry

-- Problem (2)
theorem domain_function_2 (x : ℝ): x^2 - 4 ≥ 0 ∧ 4 - x^2 ≥ 0 ∧ x^2 - 9 ≠ 0 ↔ (x = 2 ∨ x = -2) :=
sorry

-- Problem (3)
theorem domain_function_3 (x : ℝ): x - 5 ≥ 0 ∧ |x| ≠ 7 ↔ x ≥ 5 ∧ x ≠ 7 :=
sorry

end domain_function_1_domain_function_2_domain_function_3_l1662_166234


namespace total_amount_due_is_correct_l1662_166288

-- Define the initial conditions
def initial_amount : ℝ := 350
def first_year_interest_rate : ℝ := 0.03
def second_and_third_years_interest_rate : ℝ := 0.05

-- Define the total amount calculation after three years.
def total_amount_after_three_years (P : ℝ) (r1 : ℝ) (r2 : ℝ) : ℝ :=
  let first_year_amount := P * (1 + r1)
  let second_year_amount := first_year_amount * (1 + r2)
  let third_year_amount := second_year_amount * (1 + r2)
  third_year_amount

theorem total_amount_due_is_correct : 
  total_amount_after_three_years initial_amount first_year_interest_rate second_and_third_years_interest_rate = 397.45 :=
by
  sorry

end total_amount_due_is_correct_l1662_166288


namespace range_of_a_product_greater_than_one_l1662_166219

namespace ProofProblem

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + x^2 - a * x + 2

variables {x1 x2 a : ℝ}

-- Conditions
axiom f_has_two_distinct_zeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2

-- Goal 1: Prove the range of a
theorem range_of_a : a ∈ Set.Ioi 3 := sorry  -- Formal expression for (3, +∞) in Lean

-- Goal 2: Prove x1 * x2 > 1 given that a is in the correct range
theorem product_greater_than_one (ha : a ∈ Set.Ioi 3) : x1 * x2 > 1 := sorry

end ProofProblem

end range_of_a_product_greater_than_one_l1662_166219


namespace m_range_for_circle_l1662_166285

def is_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m - 3) * x + 2 * y + 5 = 0

theorem m_range_for_circle (m : ℝ) :
  (∀ x y : ℝ, is_circle x y m) → ((m > 5) ∨ (m < 1)) :=
by 
  sorry -- Proof not required

end m_range_for_circle_l1662_166285


namespace initial_discount_l1662_166292

theorem initial_discount (P D : ℝ) 
  (h1 : P - 71.4 = 5.25)
  (h2 : P * (1 - D) * 1.25 = 71.4) : 
  D = 0.255 :=
by {
  sorry
}

end initial_discount_l1662_166292


namespace unique_9_tuple_satisfying_condition_l1662_166280

theorem unique_9_tuple_satisfying_condition :
  ∃! (a : Fin 9 → ℕ), 
    (∀ i j k : Fin 9, i < j ∧ j < k →
      ∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100) :=
sorry

end unique_9_tuple_satisfying_condition_l1662_166280


namespace negation_prop_l1662_166282

variable {U : Type} (A B : Set U)
variable (x : U)

theorem negation_prop (h : x ∈ A ∩ B) : (x ∉ A ∩ B) → (x ∉ A ∧ x ∉ B) :=
sorry

end negation_prop_l1662_166282


namespace blue_water_bottles_initial_count_l1662_166289

theorem blue_water_bottles_initial_count
    (red : ℕ) (black : ℕ) (taken_out : ℕ) (left : ℕ) (initial_blue : ℕ) :
    red = 2 →
    black = 3 →
    taken_out = 5 →
    left = 4 →
    initial_blue + red + black = taken_out + left →
    initial_blue = 4 := by
  intros
  sorry

end blue_water_bottles_initial_count_l1662_166289


namespace sum_two_primes_eq_91_prod_is_178_l1662_166201

theorem sum_two_primes_eq_91_prod_is_178
  (p1 p2 : ℕ) 
  (hp1 : p1.Prime) 
  (hp2 : p2.Prime) 
  (h_sum : p1 + p2 = 91) :
  p1 * p2 = 178 := 
sorry

end sum_two_primes_eq_91_prod_is_178_l1662_166201


namespace remainder_67pow67_add_67_div_68_l1662_166279

-- Lean statement starting with the question and conditions translated to Lean

theorem remainder_67pow67_add_67_div_68 : 
  (67 ^ 67 + 67) % 68 = 66 := 
by
  -- Condition: 67 ≡ -1 mod 68
  have h : 67 % 68 = -1 % 68 := by norm_num
  sorry

end remainder_67pow67_add_67_div_68_l1662_166279


namespace sector_central_angle_l1662_166250

theorem sector_central_angle 
  (R : ℝ) (P : ℝ) (θ : ℝ) (π : ℝ) (L : ℝ)
  (h1 : P = 83) 
  (h2 : R = 14)
  (h3 : P = 2 * R + L)
  (h4 : L = θ * R)
  (degree_conversion : θ * (180 / π) = 225) : 
  θ * (180 / π) = 225 :=
by sorry

end sector_central_angle_l1662_166250


namespace rectangle_y_value_l1662_166299

theorem rectangle_y_value 
  (y : ℝ)
  (A : (0, 0) = E ∧ (0, 5) = F ∧ (y, 5) = G ∧ (y, 0) = H)
  (area : 5 * y = 35)
  (y_pos : y > 0) :
  y = 7 :=
sorry

end rectangle_y_value_l1662_166299


namespace twenty_million_in_scientific_notation_l1662_166236

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l1662_166236


namespace largest_K_inequality_l1662_166295

noncomputable def largest_K : ℝ := 18

theorem largest_K_inequality (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
(h_cond : a * b + b * c + c * a = a * b * c) :
( (a^a * (b^2 + c^2)) / ((a^a - 1)^2) + (b^b * (c^2 + a^2)) / ((b^b - 1)^2) + (c^c * (a^2 + b^2)) / ((c^c - 1)^2) )
≥ largest_K * ((a + b + c) / (a * b * c - 1)) ^ 2 :=
sorry

end largest_K_inequality_l1662_166295


namespace evaluate_f_difference_l1662_166266

def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

theorem evaluate_f_difference : f 5 - f (-5) = 800 := by
  sorry

end evaluate_f_difference_l1662_166266


namespace correct_factorization_l1662_166260

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l1662_166260


namespace fibonacci_series_sum_l1662_166291

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n-1) + fibonacci (n-2)

noncomputable def sum_fibonacci_fraction : ℚ :=
  ∑' (n : ℕ), (fibonacci n : ℚ) / (5^n : ℚ)

theorem fibonacci_series_sum : sum_fibonacci_fraction = 5 / 19 := by
  sorry

end fibonacci_series_sum_l1662_166291


namespace arithmetic_sequence_sum_l1662_166290

theorem arithmetic_sequence_sum :
  3 * (75 + 77 + 79 + 81 + 83) = 1185 := by
  sorry

end arithmetic_sequence_sum_l1662_166290


namespace solve_for_x_l1662_166257

-- Define the operation *
def op (a b : ℝ) : ℝ := 2 * a - b

-- The theorem statement
theorem solve_for_x :
  (∃ x : ℝ, op x (op 1 3) = 2) ∧ (∀ x, op x -1 = 2)
  → x = 1/2 := by
  sorry

end solve_for_x_l1662_166257


namespace big_al_ate_40_bananas_on_june_7_l1662_166297

-- Given conditions
def bananas_eaten_on_day (initial_bananas : ℕ) (day : ℕ) : ℕ :=
  initial_bananas + 4 * (day - 1)

def total_bananas_eaten (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 1 +
  bananas_eaten_on_day initial_bananas 2 +
  bananas_eaten_on_day initial_bananas 3 +
  bananas_eaten_on_day initial_bananas 4 +
  bananas_eaten_on_day initial_bananas 5 +
  bananas_eaten_on_day initial_bananas 6 +
  bananas_eaten_on_day initial_bananas 7

noncomputable def final_bananas_on_june_7 (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 7

-- Theorem to be proved
theorem big_al_ate_40_bananas_on_june_7 :
  ∃ initial_bananas, total_bananas_eaten initial_bananas = 196 ∧ final_bananas_on_june_7 initial_bananas = 40 :=
sorry

end big_al_ate_40_bananas_on_june_7_l1662_166297


namespace average_speed_l1662_166262

theorem average_speed
    (distance1 distance2 : ℕ)
    (time1 time2 : ℕ)
    (h1 : distance1 = 100)
    (h2 : distance2 = 80)
    (h3 : time1 = 1)
    (h4 : time2 = 1) :
    (distance1 + distance2) / (time1 + time2) = 90 :=
by
  sorry

end average_speed_l1662_166262


namespace sum_of_integers_ending_in_2_between_100_and_500_l1662_166204

theorem sum_of_integers_ending_in_2_between_100_and_500 :
  let s : List ℤ := List.range' 102 400 10
  let sum_of_s := s.sum
  sum_of_s = 11880 :=
by
  sorry

end sum_of_integers_ending_in_2_between_100_and_500_l1662_166204


namespace find_min_values_l1662_166221

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 - 2 * x * y + 6 * y^2 - 14 * x - 6 * y + 72

theorem find_min_values :
  (∀x y : ℝ, f x y ≥ f (15 / 2) (1 / 2)) ∧ f (15 / 2) (1 / 2) = 22.5 :=
by
  sorry

end find_min_values_l1662_166221


namespace acute_angle_sine_diff_l1662_166245

theorem acute_angle_sine_diff (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : Real.sin α = (Real.sqrt 5) / 5) (h₃ : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 :=
sorry

end acute_angle_sine_diff_l1662_166245


namespace tenth_term_geometric_sequence_l1662_166298

theorem tenth_term_geometric_sequence :
  let a := (8 : ℚ)
  let r := (-2 / 3 : ℚ)
  a * r^9 = -4096 / 19683 :=
by
  sorry

end tenth_term_geometric_sequence_l1662_166298


namespace greatest_value_of_b_l1662_166283

theorem greatest_value_of_b (b : ℝ) : -b^2 + 8 * b - 15 ≥ 0 → b ≤ 5 := sorry

end greatest_value_of_b_l1662_166283


namespace edges_parallel_to_axes_l1662_166223

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end edges_parallel_to_axes_l1662_166223


namespace total_amount_spent_l1662_166252

theorem total_amount_spent (avg_price_goat : ℕ) (num_goats : ℕ) (avg_price_cow : ℕ) (num_cows : ℕ) (total_spent : ℕ) 
  (h1 : avg_price_goat = 70) (h2 : num_goats = 10) (h3 : avg_price_cow = 400) (h4 : num_cows = 2) :
  total_spent = 1500 :=
by
  have cost_goats := avg_price_goat * num_goats
  have cost_cows := avg_price_cow * num_cows
  have total := cost_goats + cost_cows
  sorry

end total_amount_spent_l1662_166252


namespace binary_101_is_5_l1662_166268

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal : List Nat → Nat :=
  List.foldl (λ acc x => acc * 2 + x) 0

-- Convert the binary number 101₂ (which is [1, 0, 1] in list form) to decimal
theorem binary_101_is_5 : binary_to_decimal [1, 0, 1] = 5 := 
by 
  sorry

end binary_101_is_5_l1662_166268


namespace simplify_expr1_simplify_expr2_l1662_166273

-- Defining the necessary variables as real numbers for the proof
variables (x y : ℝ)

-- Prove the first expression simplification
theorem simplify_expr1 : 
  (x + 2 * y) * (x - 2 * y) - x * (x + 3 * y) = -4 * y^2 - 3 * x * y :=
  sorry

-- Prove the second expression simplification
theorem simplify_expr2 : 
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4 * x + 4) / (x + 1)) = (x + 2) / (x - 2) :=
  sorry

end simplify_expr1_simplify_expr2_l1662_166273
