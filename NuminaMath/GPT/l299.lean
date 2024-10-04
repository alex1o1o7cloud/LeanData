import Mathlib

namespace ab_bc_ca_abc_inequality_l299_299034

open Real

theorem ab_bc_ca_abc_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 + a * b * c = 4 →
    0 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 2 :=
by
  intro a b c
  intro h
  sorry

end ab_bc_ca_abc_inequality_l299_299034


namespace nilpotent_matrix_squared_zero_l299_299164

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l299_299164


namespace perimeter_of_triangle_ABC_l299_299427

-- Define the focal points and their radius
def radius : ℝ := 2

-- Define the distances between centers of the tangent circles
def center_distance : ℝ := 2 * radius

-- Define the lengths of the sides of the triangle ABC based on the problem constraints
def AB : ℝ := 2 * radius + 2 * center_distance
def BC : ℝ := 2 * radius + center_distance
def CA : ℝ := 2 * radius + center_distance

-- Define the perimeter calculation
def perimeter : ℝ := AB + BC + CA

-- Theorem stating the actual perimeter of the triangle ABC
theorem perimeter_of_triangle_ABC : perimeter = 28 := by
  sorry

end perimeter_of_triangle_ABC_l299_299427


namespace inequality_example_l299_299947

variable {a b c : ℝ} -- Declare a, b, c as real numbers

theorem inequality_example
  (ha : 0 < a)  -- Condition: a is positive
  (hb : 0 < b)  -- Condition: b is positive
  (hc : 0 < c) :  -- Condition: c is positive
  (ab * (a + b) + ac * (a + c) + bc * (b + c)) / (abc) ≥ 6 := 
sorry  -- Proof is skipped

end inequality_example_l299_299947


namespace stratified_sampling_city_B_l299_299220

theorem stratified_sampling_city_B (sales_points_A : ℕ) (sales_points_B : ℕ) (sales_points_C : ℕ) (total_sales_points : ℕ) (sample_size : ℕ)
(h_total : total_sales_points = 450)
(h_sample : sample_size = 90)
(h_sales_points_A : sales_points_A = 180)
(h_sales_points_B : sales_points_B = 150)
(h_sales_points_C : sales_points_C = 120) :
  (sample_size * sales_points_B / total_sales_points) = 30 := 
by
  sorry

end stratified_sampling_city_B_l299_299220


namespace xy_plus_one_ge_four_l299_299576

theorem xy_plus_one_ge_four {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x + 1) * (y + 1) >= 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
by
  sorry

end xy_plus_one_ge_four_l299_299576


namespace fraction_division_l299_299074

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l299_299074


namespace rate_of_simple_interest_l299_299841

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end rate_of_simple_interest_l299_299841


namespace fraction_division_l299_299347

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299347


namespace smallest_consecutive_integer_l299_299761

theorem smallest_consecutive_integer (n : ℤ) (h : 7 * n + 21 = 112) : n = 13 :=
sorry

end smallest_consecutive_integer_l299_299761


namespace supplementary_angles_difference_l299_299600
-- Import necessary libraries

-- Define the conditions
def are_supplementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

def ratio_7_2 (θ₁ θ₂ : ℝ) : Prop := θ₁ / θ₂ = 7 / 2

-- State the theorem
theorem supplementary_angles_difference (θ₁ θ₂ : ℝ) 
  (h_supp : are_supplementary θ₁ θ₂) 
  (h_ratio : ratio_7_2 θ₁ θ₂) :
  |θ₁ - θ₂| = 100 :=
by
  sorry

end supplementary_angles_difference_l299_299600


namespace calculate_expression_l299_299449

theorem calculate_expression : 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 :=
by 
  sorry

end calculate_expression_l299_299449


namespace find_s_l299_299871

theorem find_s 
  (a b c x s z : ℕ)
  (h1 : a + b = x)
  (h2 : x + c = s)
  (h3 : s + a = z)
  (h4 : b + c + z = 16) : 
  s = 8 := 
sorry

end find_s_l299_299871


namespace Greg_gold_amount_l299_299695

noncomputable def gold_amounts (G K : ℕ) : Prop :=
  G = K / 4 ∧ G + K = 100

theorem Greg_gold_amount (G K : ℕ) (h : gold_amounts G K) : G = 20 := 
by
  sorry

end Greg_gold_amount_l299_299695


namespace problem_I_problem_II_l299_299552

noncomputable def f (x a : ℝ) : ℝ := 2 / x + a * Real.log x

theorem problem_I (a : ℝ) (h : a > 0) (h' : (2:ℝ) = (1 / (4 / a)) * (a^2) / 8):
  ∃ x : ℝ, f x a = f (1 / 2) a := sorry

theorem problem_II (a : ℝ) (h : a > 0) (h' : ∃ x : ℝ, f x a < 2) :
  (True : Prop) := sorry

end problem_I_problem_II_l299_299552


namespace xn_plus_inv_xn_l299_299914

theorem xn_plus_inv_xn (θ : ℝ) (x : ℝ) (n : ℕ) (h₀ : 0 < θ) (h₁ : θ < π / 2)
  (h₂ : x + 1 / x = -2 * Real.sin θ) (hn_pos : 0 < n) :
  x ^ n + x⁻¹ ^ n = -2 * Real.sin (n * θ) := by
  sorry

end xn_plus_inv_xn_l299_299914


namespace residue_of_neg_1237_mod_37_l299_299671

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l299_299671


namespace match_scheduling_ways_l299_299236

def different_ways_to_schedule_match (num_players : Nat) (num_rounds : Nat) : Nat :=
  (num_rounds.factorial * num_rounds.factorial)

theorem match_scheduling_ways : different_ways_to_schedule_match 4 4 = 576 :=
by
  sorry

end match_scheduling_ways_l299_299236


namespace complex_expr_evaluation_l299_299874

def complex_expr : ℤ :=
  2 * (3 * (2 * (3 * (2 * (3 * (2 + 1) * 2) + 2) * 2) + 2) * 2) + 2

theorem complex_expr_evaluation : complex_expr = 5498 := by
  sorry

end complex_expr_evaluation_l299_299874


namespace find_m_from_inequality_l299_299135

theorem find_m_from_inequality :
  (∀ x, x^2 - (m+2)*x > 0 ↔ (x < 0 ∨ x > 2)) → m = 0 :=
by
  sorry

end find_m_from_inequality_l299_299135


namespace find_x_l299_299495

theorem find_x : ∃ x : ℕ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end find_x_l299_299495


namespace relationship_between_D_and_A_l299_299915

variables (A B C D : Prop)

def sufficient_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬ (Q → P)
def necessary_not_sufficient (P Q : Prop) : Prop := (Q → P) ∧ ¬ (P → Q)
def necessary_and_sufficient (P Q : Prop) : Prop := (P ↔ Q)

-- Conditions
axiom h1 : sufficient_not_necessary A B
axiom h2 : necessary_not_sufficient C B
axiom h3 : necessary_and_sufficient D C

-- Proof Goal
theorem relationship_between_D_and_A : necessary_not_sufficient D A :=
by
  sorry

end relationship_between_D_and_A_l299_299915


namespace red_section_not_damaged_l299_299662

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l299_299662


namespace find_other_person_weight_l299_299464

theorem find_other_person_weight
    (initial_avg_weight : ℕ)
    (final_avg_weight : ℕ)
    (initial_group_size : ℕ)
    (new_person_weight : ℕ)
    (final_group_size : ℕ)
    (initial_total_weight : ℕ)
    (final_total_weight : ℕ)
    (new_total_weight : ℕ)
    (other_person_weight : ℕ) :
  initial_avg_weight = 48 →
  final_avg_weight = 51 →
  initial_group_size = 23 →
  final_group_size = 25 →
  new_person_weight = 93 →
  initial_total_weight = initial_group_size * initial_avg_weight →
  final_total_weight = final_group_size * final_avg_weight →
  new_total_weight = initial_total_weight + new_person_weight + other_person_weight →
  final_total_weight = new_total_weight →
  other_person_weight = 78 :=
by
  sorry

end find_other_person_weight_l299_299464


namespace probability_at_least_one_male_l299_299446

theorem probability_at_least_one_male :
  let total_students := 5
  let males := 3
  let females := 2
  let choose_2_students := (choose total_students 2)
  let choose_2_females := (choose females 2)
  1 - (choose_2_females / choose_2_students) = 9 / 10 :=
by {
  sorry
}

end probability_at_least_one_male_l299_299446


namespace min_value_of_sequence_l299_299459

theorem min_value_of_sequence 
  (a : ℤ) 
  (a_sequence : ℕ → ℤ) 
  (h₀ : a_sequence 0 = a)
  (h_rec : ∀ n, a_sequence (n + 1) = 2 * a_sequence n - n ^ 2)
  (h_pos : ∀ n, a_sequence n > 0) :
  ∃ k, a_sequence k = 3 := 
sorry

end min_value_of_sequence_l299_299459


namespace number_of_perfect_squares_criteria_l299_299553

noncomputable def number_of_multiples_of_40_squares_lt_4e6 : ℕ :=
  let upper_limit := 2000
  let multiple := 40
  let largest_multiple := upper_limit - (upper_limit % multiple)
  largest_multiple / multiple

theorem number_of_perfect_squares_criteria :
  number_of_multiples_of_40_squares_lt_4e6 = 49 :=
sorry

end number_of_perfect_squares_criteria_l299_299553


namespace smallest_t_for_sine_polar_circle_l299_299401

theorem smallest_t_for_sine_polar_circle :
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → ∃ r : ℝ, r = Real.sin θ) ∧
           (∀ θ : ℝ, (θ = t) → ∃ r : ℝ, r = 0) ∧
           (∀ t' : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t') → ∃ r : ℝ, r = Real.sin θ) →
                       (∀ θ : ℝ, (θ = t') → ∃ r : ℝ, r = 0) → t' ≥ t) :=
by
  sorry

end smallest_t_for_sine_polar_circle_l299_299401


namespace johns_trip_distance_is_160_l299_299717

noncomputable def total_distance (y : ℕ) : Prop :=
  y / 2 + 40 + y / 4 = y

theorem johns_trip_distance_is_160 : ∃ y : ℕ, total_distance y ∧ y = 160 :=
by
  use 160
  unfold total_distance
  sorry

end johns_trip_distance_is_160_l299_299717


namespace Tony_can_add_4_pairs_of_underwear_l299_299205

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l299_299205


namespace usual_time_to_bus_stop_l299_299088

theorem usual_time_to_bus_stop
  (T : ℕ) (S : ℕ)
  (h : S * T = (4/5 * S) * (T + 9)) :
  T = 36 :=
by
  sorry

end usual_time_to_bus_stop_l299_299088


namespace type1_pieces_count_l299_299102

theorem type1_pieces_count (n : ℕ) (pieces : ℕ → ℕ)  (nonNegative : ∀ i, pieces i ≥ 0) :
  pieces 1 ≥ 4 * n - 1 :=
sorry

end type1_pieces_count_l299_299102


namespace angle_BAO_eq_angle_CAH_l299_299436

noncomputable def is_triangle (A B C : Type) : Prop := sorry
noncomputable def orthocenter (A B C H : Type) : Prop := sorry
noncomputable def circumcenter (A B C O : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem angle_BAO_eq_angle_CAH (A B C H O : Type) 
  (hABC : is_triangle A B C)
  (hH : orthocenter A B C H)
  (hO : circumcenter A B C O):
  angle B A O = angle C A H := 
  sorry

end angle_BAO_eq_angle_CAH_l299_299436


namespace percentage_decrease_l299_299325

theorem percentage_decrease (x : ℝ) 
  (h1 : 400 * (1 - x / 100) * 1.40 = 476) : 
  x = 15 := 
by 
  sorry

end percentage_decrease_l299_299325


namespace find_a5_plus_a7_l299_299917

variable {a : ℕ → ℕ}

-- Assume a is a geometric sequence with common ratio q and first term a1.
def geometric_sequence (a : ℕ → ℕ) (a_1 : ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a_1 * q ^ n

-- Given conditions of the problem:
def conditions (a : ℕ → ℕ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

-- The objective is to prove a_5 + a_7 = 160
theorem find_a5_plus_a7 (a : ℕ → ℕ) (a_1 q : ℕ) (h_geo : geometric_sequence a a_1 q) (h_cond : conditions a) : a 5 + a 7 = 160 :=
  sorry

end find_a5_plus_a7_l299_299917


namespace count_integers_satisfying_inequality_l299_299906

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l299_299906


namespace value_of_a_l299_299271

theorem value_of_a (a : ℝ) (h : (2 : ℝ)^a = (1 / 2 : ℝ)) : a = -1 := 
sorry

end value_of_a_l299_299271


namespace valid_votes_per_candidate_l299_299430

theorem valid_votes_per_candidate (total_votes : ℕ) (invalid_percentage valid_percentage_A valid_percentage_B : ℚ) 
                                  (A_votes B_votes C_votes valid_votes : ℕ) :
  total_votes = 1250000 →
  invalid_percentage = 20 →
  valid_percentage_A = 45 →
  valid_percentage_B = 35 →
  valid_votes = total_votes * (1 - invalid_percentage / 100) →
  A_votes = valid_votes * (valid_percentage_A / 100) →
  B_votes = valid_votes * (valid_percentage_B / 100) →
  C_votes = valid_votes - A_votes - B_votes →
  valid_votes = 1000000 ∧ A_votes = 450000 ∧ B_votes = 350000 ∧ C_votes = 200000 :=
by {
  sorry
}

end valid_votes_per_candidate_l299_299430


namespace students_present_each_day_l299_299017
open BigOperators

namespace Absenteeism

def absenteeism_rate : ℕ → ℝ 
| 0 => 14
| n+1 => absenteeism_rate n + 2

def present_rate (n : ℕ) : ℝ := 100 - absenteeism_rate n

theorem students_present_each_day :
  present_rate 0 = 86 ∧
  present_rate 1 = 84 ∧
  present_rate 2 = 82 ∧
  present_rate 3 = 80 ∧
  present_rate 4 = 78 := 
by
  -- Placeholder for the proof steps
  sorry

end Absenteeism

end students_present_each_day_l299_299017


namespace infinite_series_sum_l299_299651

theorem infinite_series_sum :
  (∑' n : Nat, (4 * n + 1) / ((4 * n - 1)^2 * (4 * n + 3)^2)) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l299_299651


namespace find_mass_plate_l299_299881

-- Define the region D in Cartesian coordinates
def regionD (x y : ℝ) : Prop :=
  1 ≤ (x^2 / 16 + y^2) ∧ (x^2 / 16 + y^2) ≤ 3 ∧ y ≥ x / 4 ∧ x ≥ 0

-- Define the surface density μ
def surfaceDensity (x y : ℝ) : ℝ :=
  x / (y^5)

-- Define the integral for mass
def mass (m : ℝ) : Prop :=
  m = ∫ x in 0..(2*√3), ∫ y in (x / 4)..(sqrt(3 - x^2 / 16)), surfaceDensity x y

-- Prove that the mass is equal to 4
theorem find_mass_plate : ∃ (m : ℝ), mass m ∧ m = 4 :=
by
  sorry

end find_mass_plate_l299_299881


namespace cycle_selling_price_l299_299095

theorem cycle_selling_price
  (cost_price : ℝ)
  (selling_price : ℝ)
  (percentage_gain : ℝ)
  (h_cost_price : cost_price = 1000)
  (h_percentage_gain : percentage_gain = 8) :
  selling_price = cost_price + (percentage_gain / 100) * cost_price :=
by
  sorry

end cycle_selling_price_l299_299095


namespace count_integers_satisfy_inequality_l299_299904

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l299_299904


namespace usual_time_to_catch_bus_l299_299216

variable {S T T' D : ℝ}

theorem usual_time_to_catch_bus (h1 : D = S * T)
  (h2 : D = (4 / 5) * S * T')
  (h3 : T' = T + 4) : T = 16 := by
  sorry

end usual_time_to_catch_bus_l299_299216


namespace red_section_not_damaged_l299_299663

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l299_299663


namespace ratio_eq_neg_1009_l299_299108

theorem ratio_eq_neg_1009 (p q : ℝ) (h : (1 / p + 1 / q) / (1 / p - 1 / q) = 1009) : (p + q) / (p - q) = -1009 := 
by 
  sorry

end ratio_eq_neg_1009_l299_299108


namespace largest_angle_is_75_l299_299191

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l299_299191


namespace proof_by_contradiction_example_l299_299617

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l299_299617


namespace speed_conversion_l299_299522

theorem speed_conversion (s : ℝ) (h1 : s = 1 / 3) : s * 3.6 = 1.2 := by
  -- Proof follows from the conditions given
  sorry

end speed_conversion_l299_299522


namespace compare_abc_l299_299299

noncomputable def a : ℝ := Real.exp (Real.sqrt Real.pi)
noncomputable def b : ℝ := Real.sqrt Real.pi + 1
noncomputable def c : ℝ := (Real.log Real.pi) / Real.exp 1 + 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l299_299299


namespace partitions_equiv_l299_299940

-- Definition of partitions into distinct integers
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Definition of partitions into odd integers
def b (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Theorem stating that the number of partitions into distinct integers equals the number of partitions into odd integers
theorem partitions_equiv (n : ℕ) : a n = b n :=
sorry

end partitions_equiv_l299_299940


namespace total_money_is_2800_l299_299517

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l299_299517


namespace lcm_18_24_l299_299773

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299773


namespace lcm_18_24_eq_72_l299_299786

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299786


namespace matrix_exponentiation_l299_299525

theorem matrix_exponentiation (a n : ℕ) (M : Matrix (Fin 3) (Fin 3) ℕ) (N : Matrix (Fin 3) (Fin 3) ℕ) :
  (M^n = N) →
  M = ![
    ![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]
  ] →
  N = ![
    ![1, 27, 3060],
    ![0, 1, 45],
    ![0, 0, 1]
  ] →
  a + n = 289 :=
by
  intros h1 h2 h3
  sorry

end matrix_exponentiation_l299_299525


namespace fraction_division_l299_299077

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l299_299077


namespace incorrect_inequality_l299_299423

variable (a b : ℝ)

theorem incorrect_inequality (h : a > b) : ¬ (-2 * a > -2 * b) :=
by sorry

end incorrect_inequality_l299_299423


namespace minimum_number_of_tiles_l299_299995

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l299_299995


namespace adult_ticket_cost_l299_299594

-- Definitions based on the conditions
def num_adults : ℕ := 10
def num_children : ℕ := 11
def total_bill : ℝ := 124
def child_ticket_cost : ℝ := 4

-- The proof which determines the cost of one adult ticket
theorem adult_ticket_cost : ∃ (A : ℝ), A * num_adults = total_bill - (num_children * child_ticket_cost) ∧ A = 8 := 
by
  sorry

end adult_ticket_cost_l299_299594


namespace lcm_18_24_l299_299794

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299794


namespace tank_filling_time_with_leaks_l299_299636

theorem tank_filling_time_with_leaks (pump_time : ℝ) (leak1_time : ℝ) (leak2_time : ℝ) (leak3_time : ℝ) (fill_time : ℝ)
  (h1 : pump_time = 2)
  (h2 : fill_time = 3)
  (h3 : leak1_time = 6)
  (h4 : leak2_time = 8)
  (h5 : leak3_time = 12) :
  fill_time = 8 := 
sorry

end tank_filling_time_with_leaks_l299_299636


namespace car_cleaning_ratio_l299_299582

theorem car_cleaning_ratio
    (outside_cleaning_time : ℕ)
    (total_cleaning_time : ℕ)
    (h1 : outside_cleaning_time = 80)
    (h2 : total_cleaning_time = 100) :
    (total_cleaning_time - outside_cleaning_time) / outside_cleaning_time = 1 / 4 :=
by
  sorry

end car_cleaning_ratio_l299_299582


namespace hyperbola_sum_l299_299428

noncomputable def h : ℝ := -3
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 50
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem hyperbola_sum :
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end hyperbola_sum_l299_299428


namespace time_to_write_all_rearrangements_in_hours_l299_299444

/-- Michael's name length is 7 (number of unique letters) -/
def name_length : Nat := 7

/-- Michael can write 10 rearrangements per minute -/
def write_rate : Nat := 10

/-- Number of rearrangements of Michael's name -/
def num_rearrangements : Nat := (name_length.factorial)

theorem time_to_write_all_rearrangements_in_hours :
  (num_rearrangements / write_rate : ℚ) / 60 = 8.4 := by
  sorry

end time_to_write_all_rearrangements_in_hours_l299_299444


namespace lcm_18_24_l299_299799

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299799


namespace find_a4_plus_a6_l299_299895

variable {a : ℕ → ℝ}

-- Geometric sequence definition
def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions for the problem
axiom seq_geometric : is_geometric_seq a
axiom seq_positive : ∀ n : ℕ, n > 0 → a n > 0
axiom given_equation : a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81

-- The problem to prove
theorem find_a4_plus_a6 : a 4 + a 6 = 9 :=
sorry

end find_a4_plus_a6_l299_299895


namespace A_takes_4_hours_l299_299988

variables (A B C : ℝ)

-- Given conditions
axiom h1 : 1 / B + 1 / C = 1 / 2
axiom h2 : 1 / A + 1 / C = 1 / 2
axiom h3 : B = 4

-- What we need to prove: A = 4
theorem A_takes_4_hours :
  A = 4 := by
  sorry

end A_takes_4_hours_l299_299988


namespace lcm_18_24_eq_72_l299_299779

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299779


namespace sum_remainder_l299_299148

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l299_299148


namespace molecular_weight_BaSO4_l299_299354

-- Definitions for atomic weights of elements.
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_S : ℝ := 32.07
def atomic_weight_O : ℝ := 16.00

-- Defining the number of atoms in BaSO4
def num_Ba : ℕ := 1
def num_S : ℕ := 1
def num_O : ℕ := 4

-- Statement to be proved
theorem molecular_weight_BaSO4 :
  (num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O) = 233.40 := 
by
  sorry

end molecular_weight_BaSO4_l299_299354


namespace age_ratio_in_4_years_l299_299091

-- Definitions based on conditions
def Age6YearsAgoVimal := 12
def Age6YearsAgoSaroj := 10
def CurrentAgeSaroj := 16
def CurrentAgeVimal := Age6YearsAgoVimal + 6

-- Lean statement to prove the problem
theorem age_ratio_in_4_years (x : ℕ) 
  (h_ratio : (CurrentAgeVimal + x) / (CurrentAgeSaroj + x) = 11 / 10) :
  x = 4 := 
sorry

end age_ratio_in_4_years_l299_299091


namespace probability_of_drawing_red_ball_l299_299924

theorem probability_of_drawing_red_ball (total_balls red_balls white_balls: ℕ) 
    (h1 : total_balls = 5) 
    (h2 : red_balls = 2) 
    (h3 : white_balls = 3) : 
    (red_balls : ℚ) / total_balls = 2 / 5 := 
by 
    sorry

end probability_of_drawing_red_ball_l299_299924


namespace fraction_division_l299_299075

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l299_299075


namespace symmetric_point_x_axis_l299_299889

theorem symmetric_point_x_axis (P Q : ℝ × ℝ) (hP : P = (-1, 2)) (hQ : Q = (P.1, -P.2)) : Q = (-1, -2) :=
sorry

end symmetric_point_x_axis_l299_299889


namespace rahul_deepak_age_ratio_l299_299469

-- Define the conditions
variables (R D : ℕ)
axiom deepak_age : D = 33
axiom rahul_future_age : R + 6 = 50

-- Define the theorem to prove the ratio
theorem rahul_deepak_age_ratio : R / D = 4 / 3 :=
by
  -- Placeholder for proof
  sorry

end rahul_deepak_age_ratio_l299_299469


namespace correct_transformation_l299_299084

theorem correct_transformation (x : ℝ) (h : 3 * x - 7 = 2 * x) : 3 * x - 2 * x = 7 :=
sorry

end correct_transformation_l299_299084


namespace natasha_average_speed_climbing_l299_299983

theorem natasha_average_speed_climbing :
  ∀ D : ℝ,
    (total_time = 3 + 2) →
    (total_distance = 2 * D) →
    (average_speed = total_distance / total_time) →
    (average_speed = 3) →
    (D = 7.5) →
    (climb_speed = D / 3) →
    (climb_speed = 2.5) :=
by
  intros D total_time_eq total_distance_eq average_speed_eq average_speed_is_3 D_is_7_5 climb_speed_eq
  sorry

end natasha_average_speed_climbing_l299_299983


namespace sum_lent_250_l299_299374

theorem sum_lent_250 (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (hR : R = 4) (hT : T = 8) (hSI1 : SI = P - 170) 
  (hSI2 : SI = (P * R * T) / 100) : 
  P = 250 := 
by 
  sorry

end sum_lent_250_l299_299374


namespace average_weight_of_three_l299_299451

theorem average_weight_of_three
  (rachel_weight jimmy_weight adam_weight : ℝ)
  (h1 : rachel_weight = 75)
  (h2 : jimmy_weight = rachel_weight + 6)
  (h3 : adam_weight = rachel_weight - 15) :
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by
  sorry

end average_weight_of_three_l299_299451


namespace range_of_m_l299_299682

-- Define the constants used in the problem
def a : ℝ := 0.8
def b : ℝ := 1.2

-- Define the logarithmic inequality problem
theorem range_of_m (m : ℝ) : (a^(b^m) < b^(a^m)) → m < 0 := sorry

end range_of_m_l299_299682


namespace odd_sum_pairs_count_l299_299544

open Finset

-- Define the set of numbers from 1 to 10
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define what it means for a sum to be odd
def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

-- Define the set of pairs whose sum is odd
def odd_sum_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S ×ˢ S -- Cartesian product of S with itself
  |> filter (λ pair, is_odd_sum pair.1 pair.2)

-- The theorem we need to prove: There are 25 such pairs
theorem odd_sum_pairs_count : (odd_sum_pairs S).card = 25 :=
by sorry

end odd_sum_pairs_count_l299_299544


namespace find_star_value_l299_299364

theorem find_star_value (x : ℤ) :
  45 - (28 - (37 - (15 - x))) = 58 ↔ x = 19 :=
  by
    sorry

end find_star_value_l299_299364


namespace daffodil_stamps_count_l299_299959

theorem daffodil_stamps_count (r d : ℕ) (h1 : r = 2) (h2 : r = d) : d = 2 := by
  sorry

end daffodil_stamps_count_l299_299959


namespace lcm_18_24_eq_72_l299_299769

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299769


namespace hyperbola_equations_l299_299001

def eq1 (x y : ℝ) : Prop := x^2 - 4 * y^2 = (5 + Real.sqrt 6)^2
def eq2 (x y : ℝ) : Prop := 4 * y^2 - x^2 = 4

theorem hyperbola_equations 
  (x y : ℝ)
  (hx1 : x - 2 * y = 0)
  (hx2 : x + 2 * y = 0)
  (dist : Real.sqrt ((x - 5)^2 + y^2) = Real.sqrt 6) :
  eq1 x y ∧ eq2 x y := 
sorry

end hyperbola_equations_l299_299001


namespace range_of_a_opposite_sides_l299_299967

theorem range_of_a_opposite_sides {a : ℝ} (h : (0 + 0 - a) * (1 + 1 - a) < 0) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_opposite_sides_l299_299967


namespace number_of_possible_A2_eq_one_l299_299162

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l299_299162


namespace shortest_remaining_side_length_l299_299638

noncomputable def triangle_has_right_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem shortest_remaining_side_length {a b : ℝ} (ha : a = 5) (hb : b = 12) (h_right_angle : ∃ c, triangle_has_right_angle a b c) :
  ∃ c, c = 5 :=
by 
  sorry

end shortest_remaining_side_length_l299_299638


namespace exists_consecutive_numbers_divisible_by_3_5_7_l299_299069

theorem exists_consecutive_numbers_divisible_by_3_5_7 :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 200 ∧
    a % 3 = 0 ∧ (a + 1) % 5 = 0 ∧ (a + 2) % 7 = 0 :=
by
  sorry

end exists_consecutive_numbers_divisible_by_3_5_7_l299_299069


namespace problem_solution_l299_299684

theorem problem_solution {a b : ℝ} (h : a * b + b^2 = 12) : (a + b)^2 - (a + b) * (a - b) = 24 :=
by sorry

end problem_solution_l299_299684


namespace five_numbers_property_l299_299821

theorem five_numbers_property :
  let S := {1680, 1692, 1694, 1695, 1696} in
  ∀ (a b ∈ S), a > b → a % (a - b) = 0 := by
  sorry

end five_numbers_property_l299_299821


namespace parallel_vectors_x_value_l299_299010

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_vectors_x_value (x : ℝ) :
  ∀ k : ℝ,
  k ≠ 0 ∧ k * 1 = -2 ∧ k * -2 = x →
  x = 4 :=
by
  intros k hk
  have hk1 : k * 1 = -2 := hk.2.1
  have hk2 : k * -2 = x := hk.2.2
  -- Proceed from here to the calculations according to the steps in b):
  sorry

end parallel_vectors_x_value_l299_299010


namespace total_balloons_sam_and_dan_l299_299175

noncomputable def sam_initial_balloons : ℝ := 46.0
noncomputable def balloons_given_to_fred : ℝ := 10.0
noncomputable def dan_balloons : ℝ := 16.0

theorem total_balloons_sam_and_dan :
  (sam_initial_balloons - balloons_given_to_fred) + dan_balloons = 52.0 := 
by 
  sorry

end total_balloons_sam_and_dan_l299_299175


namespace rita_bought_four_pounds_l299_299952

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l299_299952


namespace veranda_area_correct_l299_299323

-- Definitions of the room dimensions and veranda width
def room_length : ℝ := 18
def room_width : ℝ := 12
def veranda_width : ℝ := 2

-- Definition of the total length including veranda
def total_length : ℝ := room_length + 2 * veranda_width

-- Definition of the total width including veranda
def total_width : ℝ := room_width + 2 * veranda_width

-- Definition of the area of the entire space (room plus veranda)
def area_entire_space : ℝ := total_length * total_width

-- Definition of the area of the room
def area_room : ℝ := room_length * room_width

-- Definition of the area of the veranda
def area_veranda : ℝ := area_entire_space - area_room

-- Theorem statement to prove the area of the veranda
theorem veranda_area_correct : area_veranda = 136 := 
by
  sorry

end veranda_area_correct_l299_299323


namespace max_underwear_pairs_l299_299203

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l299_299203


namespace correct_conclusions_l299_299059

variable (a b c m : ℝ)
variable (y1 y2 : ℝ)

-- Conditions: 
-- Parabola y = ax^2 + bx + c, intersects x-axis at (-3,0) and (1,0)
-- a < 0
-- Points P(m-2, y1) and Q(m, y2) are on the parabola, y1 < y2

def parabola_intersects_x_axis_at_A_B : Prop :=
  ∀ x : ℝ, x = -3 ∨ x = 1 → a * x^2 + b * x + c = 0

def concavity_and_roots : Prop :=
  a < 0 ∧ b = 2 * a ∧ c = -3 * a

def conclusion_1 : Prop :=
  a * b * c < 0

def conclusion_2 : Prop :=
  b^2 - 4 * a * c > 0

def conclusion_3 : Prop :=
  3 * b + 2 * c = 0

def conclusion_4 : Prop :=
  y1 < y2 → m ≤ -1

-- Correct conclusions given the parabola properties
theorem correct_conclusions :
  concavity_and_roots a b c →
  parabola_intersects_x_axis_at_A_B a b c →
  conclusion_1 a b c ∨ conclusion_2 a b c ∨ conclusion_3 a b c ∨ conclusion_4 a b c :=
sorry

end correct_conclusions_l299_299059


namespace B_squared_ge_AC_l299_299047

variable {a b c A B C : ℝ}

theorem B_squared_ge_AC
  (h1 : b^2 < a * c)
  (h2 : a * C - 2 * b * B + c * A = 0) :
  B^2 ≥ A * C := 
sorry

end B_squared_ge_AC_l299_299047


namespace popsicles_eaten_l299_299443

theorem popsicles_eaten (total_minutes : ℕ) (minutes_per_popsicle : ℕ) (h : total_minutes = 405) (k : minutes_per_popsicle = 12) :
  (total_minutes / minutes_per_popsicle) = 33 :=
by
  sorry

end popsicles_eaten_l299_299443


namespace problem_one_problem_two_l299_299897

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a + Real.log x) / x

theorem problem_one (a : ℝ) 
  (h1 : ∀ e : ℝ, e > 0 → Deriv f e = -1/e^2)
  (h2 : (e:ℝ) > 0 → ∃ x : ℝ, f(x) is_extremum_in (0, 1)) : 0 < a ∧ a < 1 := 
sorry

theorem problem_two (x : ℝ) (h1 : x > 1) (a : ℝ) 
  (h2 : f x a / (Real.exp 1 + 1) > 2 * Real.exp (x - 1) / ((x + 1) * (x * Real.exp x + 1))) := 
sorry

end problem_one_problem_two_l299_299897


namespace q_zero_iff_arithmetic_l299_299896

-- Definitions of the terms and conditions
variables (A B q : ℝ) (hA : A ≠ 0)
def Sn (n : ℕ) : ℝ := A * n^2 + B * n + q
def arithmetic_sequence (an : ℕ → ℝ) : Prop := ∃ d a1, ∀ n, an n = a1 + n * d

-- The proof statement we need to show
theorem q_zero_iff_arithmetic (an : ℕ → ℝ) :
  (q = 0) ↔ (∃ a1 d, ∀ n, Sn A B 0 n = (d / 2) * n^2 + (a1 - d / 2) * n) :=
sorry

end q_zero_iff_arithmetic_l299_299896


namespace problem_part1_problem_part2_l299_299169

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b * x / Real.log x) - (a * x)
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ :=
  (b * (Real.log x - 1) / (Real.log x)^2) - a

theorem problem_part1 (a b : ℝ) :
  (f' (Real.exp 2) a b = -(3/4)) ∧ (f (Real.exp 2) a b = -(1/2) * (Real.exp 2)) →
  a = 1 ∧ b = 1 :=
sorry

theorem problem_part2 (a : ℝ) :
  (∃ x1 x2, x1 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f x1 a 1 ≤ f' x2 a 1 + a) →
  a ≥ (1/2 - 1/(4 * Real.exp 2)) :=
sorry

end problem_part1_problem_part2_l299_299169


namespace find_a9_l299_299129

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a9 (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a4a6 : a 4 + a 6 = 8) : 
  a 9 = 5 :=
sorry

end find_a9_l299_299129


namespace fixed_point_chord_through_plane_l299_299126

theorem fixed_point_chord_through_plane
  {S : Type*} [normed_group S]
  (circle_S : circle S)
  (tangent_point_N : S)
  (tangent_line_l : line S)
  (diameter_NM : line S)
  (fixed_point_A : S)
  (circle_through_A : circle S)
  (intersecting_points_C_D : S)
  (P Q : S) :
  (tangent_point_N ∈ circle_S) ∧ (tangent_line_l ∈ tangent_point_N) ∧ 
  (diameter_NM ∈ circle_S) ∧ (fixed_point_A ∈ diameter_NM) ∧ 
  (circle_through_A ∈ fixed_point_A) ∧ (circle_through_A ∈ tangent_line_l) ∧ 
  (intersecting_points_C_D ∈ intersect circle_through_A tangent_line_l) ∧ 
  (P ∈ circle_through_A) ∧ (Q ∈ circle_through_A) →
  ∃ K : S, K ∈ diameter_NM ∧ chord_through (P, Q) K := sorry

end fixed_point_chord_through_plane_l299_299126


namespace woman_weaves_amount_on_20th_day_l299_299020

theorem woman_weaves_amount_on_20th_day
  (a d : ℚ)
  (a2 : a + d = 17) -- second-day weaving in inches
  (S15 : 15 * a + 105 * d = 720) -- total for the first 15 days in inches
  : a + 19 * d = 108 := -- weaving on the twentieth day in inches (9 feet)
by
  sorry

end woman_weaves_amount_on_20th_day_l299_299020


namespace simplify_expression_l299_299317

variable (x : ℝ)

theorem simplify_expression : 2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := 
  sorry

end simplify_expression_l299_299317


namespace remainder_when_divided_by_x_minus_2_l299_299210

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 2 * x^2 + 11 * x - 6

theorem remainder_when_divided_by_x_minus_2 :
  (f 2) = 16 := by
  sorry

end remainder_when_divided_by_x_minus_2_l299_299210


namespace smallest_angle_of_quadrilateral_l299_299057

theorem smallest_angle_of_quadrilateral 
  (x : ℝ) 
  (h1 : x + 2 * x + 3 * x + 4 * x = 360) : 
  x = 36 :=
by
  sorry

end smallest_angle_of_quadrilateral_l299_299057


namespace lcm_18_24_l299_299805

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299805


namespace red_section_no_damage_probability_l299_299656

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l299_299656


namespace distribution_schemes_36_l299_299377

def num_distribution_schemes (total_students english_excellent computer_skills : ℕ) : ℕ :=
  if total_students = 8 ∧ english_excellent = 2 ∧ computer_skills = 3 then 36 else 0

theorem distribution_schemes_36 :
  num_distribution_schemes 8 2 3 = 36 :=
by
 sorry

end distribution_schemes_36_l299_299377


namespace mike_salary_calculation_l299_299681

theorem mike_salary_calculation
  (F : ℝ) (M : ℝ) (new_M : ℝ) (x : ℝ)
  (F_eq : F = 1000)
  (M_eq : M = x * F)
  (increase_eq : new_M = 1.40 * M)
  (new_M_val : new_M = 15400) :
  M = 11000 ∧ x = 11 :=
by
  sorry

end mike_salary_calculation_l299_299681


namespace radius_of_inscribed_circle_l299_299410

variable (p q r : ℝ)

theorem radius_of_inscribed_circle (hp : p > 0) (hq : q > 0) (area_eq : q^2 = r * p) : r = q^2 / p :=
by
  sorry

end radius_of_inscribed_circle_l299_299410


namespace variance_Y_eq_15_l299_299004

-- Definitions for the problem

-- Condition 1: X follows a binomial distribution B(5, 1/4)
def binom_dist : ProbabilityTheory.Discrete.val (prob = ProbabilityTheory.Hypergeometric(0, 1, 2, 3)) := sorry

-- Condition 2: Y is defined as 4 * X - 3
def Y (X : ℝ) := 4 * X - 3

-- The proof goal: V(Y) = 15
theorem variance_Y_eq_15 (X : ℝ) [hx : ProbabilityTheory.Binomial (5, 1/4)] : 
  let Y := 4 * X - 3 in
  ProbabilityTheory.Variance Y = 15 := by
sorry

end variance_Y_eq_15_l299_299004


namespace find_local_value_of_7_in_difference_l299_299209

-- Define the local value of 3 in the number 28943712.
def local_value_of_3_in_28943712 : Nat := 30000

-- Define the property that the local value of 7 in a number Y is 7000.
def local_value_of_7 (Y : Nat) : Prop := (Y / 1000 % 10) = 7

-- Define the unknown number X and its difference with local value of 3 in 28943712.
variable (X : Nat)

-- Assumption: The difference between X and local_value_of_3_in_28943712 results in a number whose local value of 7 is 7000.
axiom difference_condition : local_value_of_7 (X - local_value_of_3_in_28943712)

-- The proof problem statement to be solved.
theorem find_local_value_of_7_in_difference : local_value_of_7 (X - local_value_of_3_in_28943712) = true :=
by
  -- Proof is omitted.
  sorry

end find_local_value_of_7_in_difference_l299_299209


namespace total_pens_l299_299460

theorem total_pens (black_pens blue_pens : ℕ) (h1 : black_pens = 4) (h2 : blue_pens = 4) : black_pens + blue_pens = 8 :=
by
  sorry

end total_pens_l299_299460


namespace slope_of_line_l299_299614

theorem slope_of_line (x y : ℝ) (h : 2 * y = -3 * x + 6) : (∃ m b : ℝ, y = m * x + b) ∧  (m = -3 / 2) :=
by 
  sorry

end slope_of_line_l299_299614


namespace rectangle_diagonal_length_l299_299064

theorem rectangle_diagonal_length (l : ℝ) (L W d : ℝ) 
  (h_ratio : L = 5 * l ∧ W = 2 * l)
  (h_perimeter : 2 * (L + W) = 100) :
  d = (5 * Real.sqrt 290) / 7 :=
by
  sorry

end rectangle_diagonal_length_l299_299064


namespace increased_percentage_l299_299485

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l299_299485


namespace geometric_sequence_x_value_l299_299399

theorem geometric_sequence_x_value (x : ℝ) (r : ℝ) 
  (h1 : 12 * r = x) 
  (h2 : x * r = 2 / 3) 
  (h3 : 0 < x) :
  x = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_x_value_l299_299399


namespace arithmetic_progression_pairs_count_l299_299398

theorem arithmetic_progression_pairs_count (x y : ℝ) 
  (h1 : x = (15 + y) / 2)
  (h2 : x + x * y = 2 * y) : 
  (∃ x1 y1, x1 = (15 + y1) / 2 ∧ x1 + x1 * y1 = 2 * y1 ∧ x1 = (9 + 3 * Real.sqrt 7) / 2 ∧ y1 = -6 + 3 * Real.sqrt 7) ∨ 
  (∃ x2 y2, x2 = (15 + y2) / 2 ∧ x2 + x2 * y2 = 2 * y2 ∧ x2 = (9 - 3 * Real.sqrt 7) / 2 ∧ y2 = -6 - 3 * Real.sqrt 7) := 
sorry

end arithmetic_progression_pairs_count_l299_299398


namespace solve_quadratic_eq_l299_299604

theorem solve_quadratic_eq : (x : ℝ) → (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_quadratic_eq_l299_299604


namespace incorrect_statement_d_l299_299491

-- Definitions based on the problem's conditions
def is_acute (θ : ℝ) := 0 < θ ∧ θ < 90

def is_complementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 90

def is_supplementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 180

-- Statement D from the problem
def statement_d (θ : ℝ) := is_acute θ → ∀ θc, is_complementary θ θc → θ > θc

-- The theorem we want to prove
theorem incorrect_statement_d : ¬(∀ θ : ℝ, statement_d θ) := 
by sorry

end incorrect_statement_d_l299_299491


namespace value_of_expression_when_x_is_2_l299_299979

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l299_299979


namespace triangle_longest_side_l299_299184

theorem triangle_longest_side 
  (x : ℝ)
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) :
  2 * x + 1 = 17 := by
  sorry

end triangle_longest_side_l299_299184


namespace how_many_integers_satisfy_l299_299903

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l299_299903


namespace max_number_of_cubes_l299_299815

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end max_number_of_cubes_l299_299815


namespace total_spent_l299_299289

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l299_299289


namespace max_cubes_in_box_l299_299816

theorem max_cubes_in_box :
  let volume_of_cube := 27 -- volume of each small cube in cubic centimetres
  let dimensions_of_box := (8, 9, 12) -- dimensions of the box in centimetres
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3 -- volume of the box
  volume_of_box / volume_of_cube = 32 := 
by
  let volume_of_cube := 27
  let dimensions_of_box := (8, 9, 12)
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3
  show volume_of_box / volume_of_cube = 32
  sorry

end max_cubes_in_box_l299_299816


namespace raisins_in_other_boxes_l299_299393

theorem raisins_in_other_boxes (total_raisins : ℕ) (raisins_box1 : ℕ) (raisins_box2 : ℕ) (other_boxes : ℕ) (num_other_boxes : ℕ) :
  total_raisins = 437 →
  raisins_box1 = 72 →
  raisins_box2 = 74 →
  num_other_boxes = 3 →
  other_boxes = (total_raisins - raisins_box1 - raisins_box2) / num_other_boxes →
  other_boxes = 97 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end raisins_in_other_boxes_l299_299393


namespace coat_price_reduction_l299_299066

theorem coat_price_reduction 
  (original_price : ℝ) 
  (reduction_percentage : ℝ)
  (h_original_price : original_price = 500)
  (h_reduction_percentage : reduction_percentage = 60) :
  original_price * (reduction_percentage / 100) = 300 :=
by 
  sorry

end coat_price_reduction_l299_299066


namespace S_on_circumcircle_circle_S_through_C_and_I_l299_299588

open EuclideanGeometry

variables {A B C S I : Point}
variables (triangle_ABC : Triangle A B C)

-- Assume that S is the intersection of the angle bisector of ∠BAC and the perpendicular bisector of segment BC
axiom h₁ : IsIntersection S (AngleBisector (angle BAC A B C)) (PerpendicularBisector (segment B C))

-- Assume that the circle centered at S passes through B
axiom h₂ : CircleCenteredAt S (distance S B)

-- Prove that S lies on the circumcircle of triangle ABC
theorem S_on_circumcircle :
  On (Circumcircle triangle_ABC) S :=
sorry

-- Prove that the circle centered at S passing through B also passes through C and the incenter I of triangle ABC
theorem circle_S_through_C_and_I :
  (distance S C) = (distance S B) ∧ (distance S I) = (distance S B) :=
sorry

end S_on_circumcircle_circle_S_through_C_and_I_l299_299588


namespace cost_per_pizza_l299_299737

theorem cost_per_pizza (total_amount : ℝ) (num_pizzas : ℕ) (H : total_amount = 24) (H1 : num_pizzas = 3) : 
  (total_amount / num_pizzas) = 8 := 
by 
  sorry

end cost_per_pizza_l299_299737


namespace brownies_count_l299_299046

variable (total_people : Nat) (pieces_per_person : Nat) (cookies : Nat) (candy : Nat) (brownies : Nat)

def total_dessert_needed : Nat := total_people * pieces_per_person

def total_pieces_have : Nat := cookies + candy

def total_brownies_needed : Nat := total_dessert_needed total_people pieces_per_person - total_pieces_have cookies candy

theorem brownies_count (h1 : total_people = 7)
                       (h2 : pieces_per_person = 18)
                       (h3 : cookies = 42)
                       (h4 : candy = 63) :
                       total_brownies_needed total_people pieces_per_person cookies candy = 21 :=
by
  rw [h1, h2, h3, h4]
  sorry

end brownies_count_l299_299046


namespace paper_fold_length_l299_299635

theorem paper_fold_length (length_orig : ℝ) (h : length_orig = 12) : length_orig / 2 = 6 :=
by
  rw [h]
  norm_num

end paper_fold_length_l299_299635


namespace problem_solution_l299_299035

theorem problem_solution :
  ∀ (a b c d : ℝ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    (a^2 = 7 ∨ a^2 = 8) →
    (b^2 = 7 ∨ b^2 = 8) →
    (c^2 = 7 ∨ c^2 = 8) →
    (d^2 = 7 ∨ d^2 = 8) →
    a^2 + b^2 + c^2 + d^2 = 30 :=
by sorry

end problem_solution_l299_299035


namespace gardener_works_days_l299_299645

theorem gardener_works_days :
  let rose_bushes := 20
  let cost_per_rose_bush := 150
  let gardener_hourly_wage := 30
  let gardener_hours_per_day := 5
  let soil_volume := 100
  let cost_per_soil := 5
  let total_project_cost := 4100
  let total_gardening_days := 4
  (rose_bushes * cost_per_rose_bush + soil_volume * cost_per_soil + total_gardening_days * gardener_hours_per_day * gardener_hourly_wage = total_project_cost) →
  total_gardening_days = 4 :=
by
  intros
  sorry

end gardener_works_days_l299_299645


namespace blue_balls_in_JarB_l299_299332

-- Defining the conditions
def ratio_white_blue (white blue : ℕ) : Prop := white / gcd white blue = 5 ∧ blue / gcd white blue = 3

def white_balls_in_B := 15

-- Proof statement
theorem blue_balls_in_JarB :
  ∃ (blue : ℕ), ratio_white_blue 15 blue ∧ blue = 9 :=
by {
  -- Proof outline (not required, thus just using sorry)
  sorry
}


end blue_balls_in_JarB_l299_299332


namespace find_a_b_l299_299247

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem find_a_b (a b : ℝ) (x : ℝ) (h : 5 * (log a x) ^ 2 + 2 * (log b x) ^ 2 = (10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) + (Real.log x) ^ 2) :
  b = a ^ (2 / (5 + Real.sqrt 17)) ∨ b = a ^ (2 / (5 - Real.sqrt 17)) :=
sorry

end find_a_b_l299_299247


namespace triangle_non_existence_triangle_existence_l299_299828

-- Definition of the triangle inequality theorem for a triangle with given sides.
def triangle_exists (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_non_existence (h : ¬ triangle_exists 2 3 7) : true := by
  sorry

theorem triangle_existence (h : triangle_exists 5 5 5) : true := by
  sorry

end triangle_non_existence_triangle_existence_l299_299828


namespace marble_problem_l299_299331

theorem marble_problem (R B : ℝ) 
  (h1 : R + B = 6000) 
  (h2 : (R + B) - |R - B| = 4800) 
  (h3 : B > R) : B = 3600 :=
sorry

end marble_problem_l299_299331


namespace monotonic_intervals_when_a_zero_range_of_a_if_f_positive_l299_299257

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x + a * x

theorem monotonic_intervals_when_a_zero :
    ∃ I1 I2 : Set ℝ, (I1 = { x | 0 < x ∧ x < 1 }) ∧ (I2 = { x | 1 < x }) ∧
        (∀ x y ∈ I1, x < y → f x 0 > f y 0) ∧ (∀ x y ∈ I2, x < y → f x 0 < f y 0) :=
by
  sorry

theorem range_of_a_if_f_positive (a : ℝ) :
    (∀ x > 0, (x - 1) * Real.log x + a * x > 0) → a > 0 :=
by
  sorry

end monotonic_intervals_when_a_zero_range_of_a_if_f_positive_l299_299257


namespace sin_value_l299_299886

theorem sin_value (α : ℝ) (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (α + Real.pi / 6) = -3 / 5) : 
  Real.sin (2 * α + Real.pi / 12) = -17 * Real.sqrt 2 / 50 := 
sorry

end sin_value_l299_299886


namespace combinations_count_l299_299168

theorem combinations_count:
  let valid_a (a: ℕ) := a < 1000 ∧ a % 29 = 7
  let valid_b (b: ℕ) := b < 1000 ∧ b % 47 = 22
  let valid_c (c: ℕ) (a b: ℕ) := c < 1000 ∧ c = (a + b) % 23 
  ∃ (a b c: ℕ), valid_a a ∧ valid_b b ∧ valid_c c a b :=
  sorry

end combinations_count_l299_299168


namespace lcm_18_24_eq_72_l299_299784

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299784


namespace find_value_of_fraction_of_x_six_l299_299521

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log b)

theorem find_value_of_fraction_of_x_six (x : ℝ) (h : log_base (10 * x) 10 + log_base (100 * x ^ 2) 10 = -1) : 
    1 / x ^ 6 = 31622.7766 :=
by
  sorry

end find_value_of_fraction_of_x_six_l299_299521


namespace arithmetic_geom_sequence_a2_l299_299888

theorem arithmetic_geom_sequence_a2 :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n+1) = a n + 2) →  -- Arithmetic sequence with common difference of 2
    a 1 * a 4 = a 3 ^ 2 →  -- Geometric sequence property for a_1, a_3, a_4
    a 2 = -6 :=             -- The value of a_2
by
  intros a h_arith h_geom
  sorry

end arithmetic_geom_sequence_a2_l299_299888


namespace count_three_digit_multiples_13_and_5_l299_299011

theorem count_three_digit_multiples_13_and_5 : 
  ∃ count : ℕ, count = 14 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 65 = 0) → 
  (∃ k : ℕ, n = k * 65 ∧ 2 ≤ k ∧ k ≤ 15) → count = 14 :=
by
  sorry

end count_three_digit_multiples_13_and_5_l299_299011


namespace probability_not_miss_is_correct_l299_299757

-- Define the probability that Peter will miss his morning train
def p_miss : ℚ := 5 / 12

-- Define the probability that Peter does not miss his morning train
def p_not_miss : ℚ := 1 - p_miss

-- The theorem to prove
theorem probability_not_miss_is_correct : p_not_miss = 7 / 12 :=
by
  -- Proof omitted
  sorry

end probability_not_miss_is_correct_l299_299757


namespace division_of_fractions_l299_299353

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299353


namespace lcm_18_24_eq_72_l299_299782

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299782


namespace lcm_18_24_l299_299775

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299775


namespace total_difference_is_correct_l299_299044

-- Define the harvest rates
def valencia_weekday_ripe := 90
def valencia_weekday_unripe := 38
def navel_weekday_ripe := 125
def navel_weekday_unripe := 65
def blood_weekday_ripe := 60
def blood_weekday_unripe := 42

def valencia_weekend_ripe := 75
def valencia_weekend_unripe := 33
def navel_weekend_ripe := 100
def navel_weekend_unripe := 57
def blood_weekend_ripe := 45
def blood_weekend_unripe := 36

-- Define the number of weekdays and weekend days
def weekdays := 5
def weekend_days := 2

-- Calculate the total harvests
def total_valencia_ripe := valencia_weekday_ripe * weekdays + valencia_weekend_ripe * weekend_days
def total_valencia_unripe := valencia_weekday_unripe * weekdays + valencia_weekend_unripe * weekend_days
def total_navel_ripe := navel_weekday_ripe * weekdays + navel_weekend_ripe * weekend_days
def total_navel_unripe := navel_weekday_unripe * weekdays + navel_weekend_unripe * weekend_days
def total_blood_ripe := blood_weekday_ripe * weekdays + blood_weekend_ripe * weekend_days
def total_blood_unripe := blood_weekday_unripe * weekdays + blood_weekend_unripe * weekend_days

-- Calculate the total differences
def valencia_difference := total_valencia_ripe - total_valencia_unripe
def navel_difference := total_navel_ripe - total_navel_unripe
def blood_difference := total_blood_ripe - total_blood_unripe

-- Define the total difference
def total_difference := valencia_difference + navel_difference + blood_difference

-- Theorem statement
theorem total_difference_is_correct :
  total_difference = 838 := by
  sorry

end total_difference_is_correct_l299_299044


namespace parallel_lines_slope_l299_299153

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end parallel_lines_slope_l299_299153


namespace sum_in_base_b_l299_299729

-- Definitions needed to articulate the problem
def base_b_value (n : ℕ) (b : ℕ) : ℕ :=
  match n with
  | 12 => b + 2
  | 15 => b + 5
  | 16 => b + 6
  | 3146 => 3 * b^3 + 1 * b^2 + 4 * b + 6
  | _  => 0

def s_in_base_b (b : ℕ) : ℕ :=
  base_b_value 12 b + base_b_value 15 b + base_b_value 16 b

theorem sum_in_base_b (b : ℕ) (h : (base_b_value 12 b) * (base_b_value 15 b) * (base_b_value 16 b) = base_b_value 3146 b) :
  s_in_base_b b = 44 := by
  sorry

end sum_in_base_b_l299_299729


namespace find_vector_at_t4_l299_299371

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end find_vector_at_t4_l299_299371


namespace tyler_meal_combinations_is_720_l299_299611

-- Required imports for permutations and combinations
open Nat
open BigOperators

-- Assumptions based on the problem conditions
def meat_options  := 4
def veg_options := 4
def dessert_options := 5
def bread_options := 3

-- Using combinations and permutations for calculations
def comb(n k : ℕ) := Nat.choose n k
def perm(n k : ℕ) := n.factorial / (n - k).factorial

-- Number of ways to choose meals
def meal_combinations : ℕ :=
  meat_options * (comb veg_options 2) * dessert_options * (perm bread_options 2)

theorem tyler_meal_combinations_is_720 : meal_combinations = 720 := by
  -- We provide proof later; for now, put sorry to skip
  sorry

end tyler_meal_combinations_is_720_l299_299611


namespace range_of_a_l299_299138

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x | x^2 ≤ 1} ∪ {a} ↔ x ∈ {x | x^2 ≤ 1}) → (-1 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end range_of_a_l299_299138


namespace number_of_uncool_parents_l299_299426

variable (total_students cool_dads cool_moms cool_both : ℕ)

theorem number_of_uncool_parents (h1 : total_students = 40)
                                  (h2 : cool_dads = 18)
                                  (h3 : cool_moms = 22)
                                  (h4 : cool_both = 10) :
    total_students - (cool_dads + cool_moms - cool_both) = 10 := by
  sorry

end number_of_uncool_parents_l299_299426


namespace length_of_first_platform_l299_299998

theorem length_of_first_platform 
  (train_length : ℕ) (first_time : ℕ) (second_platform_length : ℕ) (second_time : ℕ)
  (speed_first : ℕ) (speed_second : ℕ) :
  train_length = 230 → 
  first_time = 15 → 
  second_platform_length = 250 → 
  second_time = 20 → 
  speed_first = (train_length + L) / first_time →
  speed_second = (train_length + second_platform_length) / second_time →
  speed_first = speed_second →
  (L : ℕ) = 130 :=
by
  sorry

end length_of_first_platform_l299_299998


namespace g_of_36_l299_299300

theorem g_of_36 (g : ℕ → ℕ)
  (h1 : ∀ n, g (n + 1) > g n)
  (h2 : ∀ m n, g (m * n) = g m * g n)
  (h3 : ∀ m n, m ≠ n ∧ m ^ n = n ^ m → (g m = n ∨ g n = m))
  (h4 : ∀ n, g (n ^ 2) = g n * n) :
  g 36 = 36 :=
  sorry

end g_of_36_l299_299300


namespace area_of_rectangle_l299_299322

-- Define the conditions
variable {S1 S2 S3 S4 : ℝ} -- side lengths of the four squares

-- The conditions:
-- 1. Four non-overlapping squares
-- 2. The area of the shaded square is 4 square inches
def conditions (S1 S2 S3 S4 : ℝ) : Prop :=
    S1^2 = 4 -- Given that one of the squares has an area of 4 square inches

-- The proof problem:
theorem area_of_rectangle (S1 S2 S3 S4 : ℝ) (h1 : 2 * S1 = S2) (h2 : 2 * S2 = S3) (h3 : conditions S1 S2 S3 S4) : 
    S1^2 + S2^2 + S3^2 = 24 :=
by
  sorry

end area_of_rectangle_l299_299322


namespace largest_square_side_length_l299_299838

theorem largest_square_side_length (smallest_square_side next_square_side : ℕ) (h1 : smallest_square_side = 1) 
(h2 : next_square_side = smallest_square_side + 6) :
  ∃ x : ℕ, x = 7 :=
by
  existsi 7
  sorry

end largest_square_side_length_l299_299838


namespace probability_no_success_l299_299659

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l299_299659


namespace joshua_skittles_l299_299292

theorem joshua_skittles (eggs : ℝ) (skittles_per_friend : ℝ) (friends : ℝ) (h1 : eggs = 6.0) (h2 : skittles_per_friend = 40.0) (h3 : friends = 5.0) : skittles_per_friend * friends = 200.0 := 
by 
  sorry

end joshua_skittles_l299_299292


namespace pure_gala_trees_l299_299493

theorem pure_gala_trees (T F G : ℝ) (h1 : F + 0.10 * T = 221)
  (h2 : F = 0.75 * T) : G = T - F - 0.10 * T := 
by 
  -- We define G and show it equals 39
  have eq : T = F / 0.75 := by sorry
  have G_eq : G = T - F - 0.10 * T := by sorry 
  exact G_eq

end pure_gala_trees_l299_299493


namespace quadratic_to_completed_square_l299_299652

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 2

-- Define the completed square form of the function.
def completed_square_form (x : ℝ) : ℝ := (x + 1)^2 - 3

-- The theorem statement that needs to be proven.
theorem quadratic_to_completed_square :
  ∀ x : ℝ, quadratic_function x = completed_square_form x :=
by sorry

end quadratic_to_completed_square_l299_299652


namespace driving_time_constraint_l299_299609

variable (x y z : ℝ)

theorem driving_time_constraint (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  3 + (60 / x) + (90 / y) + (200 / z) ≤ 10 :=
sorry

end driving_time_constraint_l299_299609


namespace scatter_plot_variable_placement_l299_299083

theorem scatter_plot_variable_placement
  (forecast explanatory : Type)
  (scatter_plot : explanatory → forecast → Prop) : 
  ∀ (x : explanatory) (y : forecast), scatter_plot x y → (True -> True) := 
by
  intros x y h
  sorry

end scatter_plot_variable_placement_l299_299083


namespace f_of_1_eq_zero_l299_299597

-- Conditions
variables (f : ℝ → ℝ)
-- f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
-- f is a periodic function with a period of 2
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem f_of_1_eq_zero {f : ℝ → ℝ} (h1 : odd_function f) (h2 : periodic_function f) : f 1 = 0 :=
by { sorry }

end f_of_1_eq_zero_l299_299597


namespace oak_trees_remaining_l299_299762

-- Variables representing the initial number of oak trees and the number of cut down trees.
variables (initial_trees cut_down_trees remaining_trees : ℕ)

-- Conditions of the problem.
def initial_trees_condition : initial_trees = 9 := sorry
def cut_down_trees_condition : cut_down_trees = 2 := sorry

-- Theorem representing the proof problem.
theorem oak_trees_remaining (h1 : initial_trees = 9) (h2 : cut_down_trees = 2) :
  remaining_trees = initial_trees - cut_down_trees :=
sorry

end oak_trees_remaining_l299_299762


namespace percentage_increase_on_sale_l299_299819

theorem percentage_increase_on_sale (P S : ℝ) (hP : P ≠ 0) (hS : S ≠ 0)
  (h_price_reduction : (0.8 : ℝ) * P * S * (1 + (X / 100)) = 1.44 * P * S) :
  X = 80 := by
  sorry

end percentage_increase_on_sale_l299_299819


namespace expression_evaluation_l299_299402

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l299_299402


namespace cone_from_sector_radius_l299_299361

theorem cone_from_sector_radius (r : ℝ) (slant_height : ℝ) : 
  (r = 9) ∧ (slant_height = 12) ↔ 
  (∃ (sector_angle : ℝ) (sector_radius : ℝ), 
    sector_angle = 270 ∧ sector_radius = 12 ∧ 
    slant_height = sector_radius ∧ 
    (2 * π * r = sector_angle / 360 * 2 * π * sector_radius)) :=
by
  sorry

end cone_from_sector_radius_l299_299361


namespace train_speed_conversion_l299_299103

theorem train_speed_conversion (s_mps : ℝ) (h : s_mps = 30.002399999999998) : 
  s_mps * 3.6 = 108.01 :=
by
  sorry

end train_speed_conversion_l299_299103


namespace lcm_18_24_l299_299814
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299814


namespace paws_on_ground_are_correct_l299_299392

-- Problem statement
def num_paws_on_ground (total_dogs : ℕ) (half_on_all_fours : ℕ) (paws_on_all_fours : ℕ) (half_on_two_legs : ℕ) (paws_on_two_legs : ℕ) : ℕ :=
  half_on_all_fours * paws_on_all_fours + half_on_two_legs * paws_on_two_legs

theorem paws_on_ground_are_correct :
  let total_dogs := 12
  let half_on_all_fours := 6
  let half_on_two_legs := 6
  let paws_on_all_fours := 4
  let paws_on_two_legs := 2
  num_paws_on_ground total_dogs half_on_all_fours paws_on_all_fours half_on_two_legs paws_on_two_legs = 36 :=
by sorry

end paws_on_ground_are_correct_l299_299392


namespace ratio_of_y_to_x_l299_299706

theorem ratio_of_y_to_x (c x y : ℝ) (hx : x = 0.90 * c) (hy : y = 1.20 * c) :
  y / x = 4 / 3 := 
sorry

end ratio_of_y_to_x_l299_299706


namespace how_many_integers_satisfy_l299_299902

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l299_299902


namespace barbara_typing_time_l299_299859

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l299_299859


namespace cos_periodicity_even_function_property_l299_299880

theorem cos_periodicity_even_function_property (n : ℤ) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) (h_range : -180 ≤ n ∧ n ≤ 180) : n = 43 :=
by
  sorry

end cos_periodicity_even_function_property_l299_299880


namespace original_number_is_0_2_l299_299310

theorem original_number_is_0_2 :
  ∃ x : ℝ, (1 / (1 / x - 1) - 1 = -0.75) ∧ x = 0.2 :=
by
  sorry

end original_number_is_0_2_l299_299310


namespace subset_fourth_power_l299_299575

theorem subset_fourth_power (M : Finset ℕ) (hM1 : M.card = 1985) 
  (hM2 : ∀ n ∈ M, ∀ p : ℕ, p.prime → (p ∣ n → p ≤ 26)) :
  ∃ A : Finset ℕ, A ⊆ M ∧ A.card = 4 ∧ ∃ k : ℕ, ∏ a in A, a = k ^ 4 :=
begin
  sorry
end

end subset_fourth_power_l299_299575


namespace range_of_a_for_inequality_l299_299272

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a > 0) → a > 1 :=
by
  sorry

end range_of_a_for_inequality_l299_299272


namespace crushing_load_l299_299243

theorem crushing_load (T H C : ℝ) (L : ℝ) 
  (h1 : T = 5) (h2 : H = 10) (h3 : C = 3)
  (h4 : L = C * 25 * T^4 / H^2) : 
  L = 468.75 :=
by
  sorry

end crushing_load_l299_299243


namespace time_spent_moving_l299_299458

noncomputable def time_per_trip_filling : ℝ := 15
noncomputable def time_per_trip_driving : ℝ := 30
noncomputable def time_per_trip_unloading : ℝ := 20
noncomputable def number_of_trips : ℕ := 10

theorem time_spent_moving :
  10.83 = (time_per_trip_filling + time_per_trip_driving + time_per_trip_unloading) * number_of_trips / 60 :=
by
  sorry

end time_spent_moving_l299_299458


namespace luke_number_of_rounds_l299_299442

variable (points_per_round total_points : ℕ)

theorem luke_number_of_rounds 
  (h1 : points_per_round = 3)
  (h2 : total_points = 78) : 
  total_points / points_per_round = 26 := 
by 
  sorry

end luke_number_of_rounds_l299_299442


namespace secondTrain_speed_l299_299072

/-
Conditions:
1. Two trains start from A and B and travel towards each other.
2. The distance between them is 1100 km.
3. At the time of their meeting, one train has traveled 100 km more than the other.
4. The first train's speed is 50 kmph.
-/

-- Let v be the speed of the second train
def secondTrainSpeed (v : ℝ) : Prop :=
  ∃ d : ℝ, 
    d > 0 ∧
    v > 0 ∧
    (d + (d - 100) = 1100) ∧
    ((d / 50) = ((d - 100) / v))

-- Here is the main theorem translating the problem statement:
theorem secondTrain_speed :
  secondTrainSpeed (250 / 6) :=
by
  sorry

end secondTrain_speed_l299_299072


namespace greatest_integer_leq_fraction_l299_299474

theorem greatest_integer_leq_fraction (N D : ℝ) (hN : N = 4^103 + 3^103 + 2^103) (hD : D = 4^100 + 3^100 + 2^100) :
  ⌊N / D⌋ = 64 :=
by
  sorry

end greatest_integer_leq_fraction_l299_299474


namespace gibi_percentage_is_59_l299_299923

-- Define the conditions
def max_score := 700
def avg_score := 490
def jigi_percent := 55
def mike_percent := 99
def lizzy_percent := 67

def jigi_score := (jigi_percent * max_score) / 100
def mike_score := (mike_percent * max_score) / 100
def lizzy_score := (lizzy_percent * max_score) / 100

def total_score := 4 * avg_score
def gibi_score := total_score - (jigi_score + mike_score + lizzy_score)

def gibi_percent := (gibi_score * 100) / max_score

-- The proof goal
theorem gibi_percentage_is_59 : gibi_percent = 59 := by
  sorry

end gibi_percentage_is_59_l299_299923


namespace simplify_div_expr_l299_299177

theorem simplify_div_expr (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
sorry

end simplify_div_expr_l299_299177


namespace space_between_trees_l299_299154

theorem space_between_trees (n_trees : ℕ) (tree_space : ℕ) (total_length : ℕ) (spaces_between_trees : ℕ) (result_space : ℕ) 
  (h1 : n_trees = 8)
  (h2 : tree_space = 1)
  (h3 : total_length = 148)
  (h4 : spaces_between_trees = n_trees - 1)
  (h5 : result_space = (total_length - n_trees * tree_space) / spaces_between_trees) : 
  result_space = 20 := 
by sorry

end space_between_trees_l299_299154


namespace circular_seat_coloring_l299_299591

def count_colorings (n : ℕ) : ℕ :=
  sorry

theorem circular_seat_coloring :
  count_colorings 6 = 66 :=
by
  sorry

end circular_seat_coloring_l299_299591


namespace john_total_spent_l299_299291

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l299_299291


namespace find_am_2n_l299_299145

-- Definition of the conditions
variables {a : ℝ} {m n : ℝ}
axiom am_eq_5 : a ^ m = 5
axiom an_eq_2 : a ^ n = 2

-- The statement we want to prove
theorem find_am_2n : a ^ (m - 2 * n) = 5 / 4 :=
by {
  sorry
}

end find_am_2n_l299_299145


namespace green_pairs_count_l299_299281

variable (blueShirtedStudents : Nat)
variable (yellowShirtedStudents : Nat)
variable (greenShirtedStudents : Nat)
variable (totalStudents : Nat)
variable (totalPairs : Nat)
variable (blueBluePairs : Nat)

def green_green_pairs (blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs : Nat) : Nat := 
  greenShirtedStudents / 2

theorem green_pairs_count
  (h1 : blueShirtedStudents = 70)
  (h2 : yellowShirtedStudents = 80)
  (h3 : greenShirtedStudents = 50)
  (h4 : totalStudents = 200)
  (h5 : totalPairs = 100)
  (h6 : blueBluePairs = 30) : 
  green_green_pairs blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs = 25 := by
  sorry

end green_pairs_count_l299_299281


namespace chess_tournament_l299_299068

theorem chess_tournament (n k : ℕ) (S : ℕ) (m : ℕ) 
  (h1 : S ≤ k * n) 
  (h2 : S ≥ m * n) 
  : m ≤ k := 
by 
  sorry

end chess_tournament_l299_299068


namespace fewest_tiles_to_cover_region_l299_299997

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l299_299997


namespace max_fraction_l299_299754

theorem max_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1 / 3) : (a, b) = (25, 76) :=
sorry

end max_fraction_l299_299754


namespace division_of_fractions_l299_299352

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299352


namespace find_rs_l299_299249

theorem find_rs :
  ∃ r s : ℝ, ∀ x : ℝ, 8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 8 * (x - r) ^ 2 * (x - s) * (x - 1) :=
sorry

end find_rs_l299_299249


namespace paving_cost_correct_l299_299984

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 400
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost_correct :
  cost = 8250 := by
  sorry

end paving_cost_correct_l299_299984


namespace solution_l299_299002

axiom f : ℝ → ℝ

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → y ≤ 0 → f x > f y

def main_problem : Prop :=
  even_function f ∧ decreasing_function f ∧ f (-2) = 0 → ∀ x, f x < 0 ↔ x > -2 ∧ x < 2

theorem solution : main_problem :=
by
  sorry

end solution_l299_299002


namespace michaels_brother_money_end_l299_299732

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l299_299732


namespace division_of_fractions_l299_299349

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299349


namespace john_new_cards_l299_299934

def cards_per_page : ℕ := 3
def old_cards : ℕ := 16
def pages_used : ℕ := 8

theorem john_new_cards : (pages_used * cards_per_page) - old_cards = 8 := by
  sorry

end john_new_cards_l299_299934


namespace red_section_not_damaged_l299_299665

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l299_299665


namespace unique_solution_l299_299877

def is_prime (n : ℕ) : Prop := Nat.Prime n

def eq_triple (m p q : ℕ) : Prop :=
  2 ^ m * p ^ 2 + 1 = q ^ 5

theorem unique_solution (m p q : ℕ) (h1 : m > 0) (h2 : is_prime p) (h3 : is_prime q) :
  eq_triple m p q ↔ (m, p, q) = (1, 11, 3) := by
  sorry

end unique_solution_l299_299877


namespace paired_divisors_prime_properties_l299_299573

theorem paired_divisors_prime_properties (n : ℕ) (h : n > 0) (h_pairing : ∃ (pairing : (ℕ × ℕ) → Prop), 
  (∀ d1 d2 : ℕ, 
    pairing (d1, d2) → d1 * d2 = n ∧ Prime (d1 + d2))) : 
  (∀ (d1 d2 : ℕ), d1 ≠ d2 → d1 + d2 ≠ d3 + d4) ∧ (∀ p : ℕ, Prime p → ¬ p ∣ n) :=
by
  sorry

end paired_divisors_prime_properties_l299_299573


namespace find_income_separator_l299_299396

-- Define the income and tax parameters
def income : ℝ := 60000
def total_tax : ℝ := 8000
def rate1 : ℝ := 0.10
def rate2 : ℝ := 0.20

-- Define the function for total tax calculation
def tax (I : ℝ) : ℝ := rate1 * I + rate2 * (income - I)

theorem find_income_separator (I : ℝ) (h: tax I = total_tax) : I = 40000 :=
by sorry

end find_income_separator_l299_299396


namespace michaels_brother_final_amount_l299_299734

theorem michaels_brother_final_amount :
  ∀ (michael_money michael_brother_initial michael_give_half candy_cost money_left : ℕ),
  michael_money = 42 →
  michael_brother_initial = 17 →
  michael_give_half = michael_money / 2 →
  let michael_brother_total := michael_brother_initial + michael_give_half in
  candy_cost = 3 →
  money_left = michael_brother_total - candy_cost →
  money_left = 35 :=
by
  intros michael_money michael_brother_initial michael_give_half candy_cost money_left
  intros h1 h2 h3 michael_brother_total h4 h5
  sorry

end michaels_brother_final_amount_l299_299734


namespace tax_rate_l299_299230

noncomputable def payroll_tax : Float := 300000
noncomputable def tax_paid : Float := 200
noncomputable def tax_threshold : Float := 200000

theorem tax_rate (tax_rate : Float) : 
  (payroll_tax - tax_threshold) * tax_rate = tax_paid → tax_rate = 0.002 := 
by
  sorry

end tax_rate_l299_299230


namespace lcm_18_24_l299_299813
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299813


namespace function_has_two_zeros_l299_299006

/-- 
Given the function y = x + 1/(2x) + t has two zeros under the condition t > 0,
prove that the range of the real number t is (-∞, -√2).
-/
theorem function_has_two_zeros (t : ℝ) (ht : t > 0) : t < -Real.sqrt 2 :=
sorry

end function_has_two_zeros_l299_299006


namespace simplify_evaluate_expression_l299_299178

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * (a * b^2) - 2 = -2 :=
by
  sorry

end simplify_evaluate_expression_l299_299178


namespace count_integer_points_l299_299195

-- Define the conditions: the parabola P with focus at (0,0) and passing through (6,4) and (-6,-4)
def parabola (P : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y : ℝ, P (x, y) ↔ y = a*x^2 + b) ∧ 
  P (6, 4) ∧ P (-6, -4)

-- Define the main theorem to be proved: the count of integer points satisfying the inequality
theorem count_integer_points (P : ℝ × ℝ → Prop) (hP : parabola P) :
  ∃ n : ℕ, n = 45 ∧ ∀ (x y : ℤ), P (x, y) → |6 * x + 4 * y| ≤ 1200 :=
sorry

end count_integer_points_l299_299195


namespace linda_winning_probability_l299_299566

noncomputable def probability_linda_wins : ℝ :=
  (1 / 16 : ℝ) / (1 - (1 / 32 : ℝ))

theorem linda_winning_probability :
  probability_linda_wins = 2 / 31 :=
sorry

end linda_winning_probability_l299_299566


namespace part_a_part_b_part_c_l299_299562

def quadradois (n : ℕ) : Prop :=
  ∃ (S1 S2 : ℕ), S1 ≠ S2 ∧ (S1 * S1 + S2 * S2 ≤ S1 * S1 + S2 * S2 + (n - 2))

theorem part_a : quadradois 6 := 
sorry

theorem part_b : quadradois 2015 := 
sorry

theorem part_c : ∀ (n : ℕ), n > 5 → quadradois n := 
sorry

end part_a_part_b_part_c_l299_299562


namespace contradiction_proof_l299_299616

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l299_299616


namespace barbara_typing_time_l299_299852

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l299_299852


namespace find_a_b_tangent_line_at_zero_l299_299133

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3
noncomputable def f' (a b x : ℝ) : ℝ := 2 * a * x + b

theorem find_a_b :
  ∃ a b : ℝ, (a ≠ 0) ∧ (∀ x, f' a b x = 2 * x - 8) := 
sorry

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x + x^2 - 8 * x + 3
noncomputable def g' (x : ℝ) : ℝ := Real.exp x * Real.sin x + Real.exp x * Real.cos x + 2 * x - 8

theorem tangent_line_at_zero :
  g' 0 = -7 ∧ g 0 = 3 ∧ (∀ y, y = 3 + (-7) * x) := 
sorry

end find_a_b_tangent_line_at_zero_l299_299133


namespace correct_result_l299_299267

theorem correct_result (x : ℕ) (h : x + 65 = 125) : x + 95 = 155 :=
sorry

end correct_result_l299_299267


namespace smallest_value_at_x_5_l299_299537

-- Define the variable x
def x : ℕ := 5

-- Define each expression
def exprA := 8 / x
def exprB := 8 / (x + 2)
def exprC := 8 / (x - 2)
def exprD := x / 8
def exprE := (x + 2) / 8

-- The goal is to prove that exprD yields the smallest value
theorem smallest_value_at_x_5 : exprD = min (min (min exprA exprB) (min exprC exprE)) :=
sorry

end smallest_value_at_x_5_l299_299537


namespace jacket_cost_is_30_l299_299174

-- Let's define the given conditions
def num_dresses := 5
def cost_per_dress := 20 -- dollars
def num_pants := 3
def cost_per_pant := 12 -- dollars
def num_jackets := 4
def transport_cost := 5 -- dollars
def initial_amount := 400 -- dollars
def remaining_amount := 139 -- dollars

-- Define the cost per jacket
def cost_per_jacket := 30 -- dollars

-- Final theorem statement to be proved
theorem jacket_cost_is_30:
  num_dresses * cost_per_dress + num_pants * cost_per_pant + num_jackets * cost_per_jacket + transport_cost = initial_amount - remaining_amount :=
sorry

end jacket_cost_is_30_l299_299174


namespace expression_evaluation_l299_299403

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l299_299403


namespace total_spent_l299_299288

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l299_299288


namespace min_value_at_constraints_l299_299000

open Classical

noncomputable def min_value (x y : ℝ) : ℝ := (x^2 + y^2 + x) / (x * y)

def constraints (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x + 2 * y = 1

theorem min_value_at_constraints : 
∃ (x y : ℝ), constraints x y ∧ min_value x y = 2 * Real.sqrt 2 + 2 :=
by
  sorry

end min_value_at_constraints_l299_299000


namespace part1_solution_part2_solution_l299_299724

-- Definitions for propositions p and q
def p (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- The actual Lean 4 statements
theorem part1_solution (x : ℝ) (m : ℝ) (hm : m = 1) (hp : p m x) (hq : q x) : 2 ≤ x ∧ x < 3 := by
  sorry

theorem part2_solution (m : ℝ) (hm : m > 0) (hsuff : ∀ x, q x → p m x) : (4 / 3) < m ∧ m < 2 := by
  sorry

end part1_solution_part2_solution_l299_299724


namespace tom_total_spent_on_video_games_l299_299070

-- Conditions
def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

-- Statement to be proved
theorem tom_total_spent_on_video_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end tom_total_spent_on_video_games_l299_299070


namespace positive_integer_x_l299_299500

theorem positive_integer_x (x : ℕ) (hx : 15 * x = x^2 + 56) : x = 8 := by
  sorry

end positive_integer_x_l299_299500


namespace sum_reciprocals_square_l299_299327

theorem sum_reciprocals_square (x y : ℕ) (h : x * y = 11) : (1 : ℚ) / (↑x ^ 2) + (1 : ℚ) / (↑y ^ 2) = 122 / 121 :=
by
  sorry

end sum_reciprocals_square_l299_299327


namespace lcm_18_24_l299_299791

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299791


namespace inverse_of_f_at_2_l299_299598

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inverse_of_f_at_2 : ∀ x, x ≥ 0 → f x = 2 → x = Real.sqrt 3 :=
by
  intro x hx heq
  sorry

end inverse_of_f_at_2_l299_299598


namespace interest_rate_simple_and_compound_l299_299512

theorem interest_rate_simple_and_compound (P T: ℝ) (SI CI R: ℝ) 
  (simple_interest_eq: SI = (P * R * T) / 100)
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (hP : P = 3000) (hT : T = 2) (hSI : SI = 300) (hCI : CI = 307.50) :
  R = 5 :=
by
  sorry

end interest_rate_simple_and_compound_l299_299512


namespace ratio_a5_b5_l299_299564

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : ℕ → ℚ) : ℕ → ℚ
| 0 := 0
| (n + 1) := sum_arithmetic_seq a n + a (n + 1)

theorem ratio_a5_b5
  {a b : ℕ → ℚ}
  {d_a d_b : ℚ}
  (ha : arithmetic_seq a d_a)
  (hb : arithmetic_seq b d_b)
  (h : ∀ n, (sum_arithmetic_seq a n) / (sum_arithmetic_seq b n) = (7 * n + 5) / (n + 3))
  : (a 5) / (b 5) = 17 / 3 :=
sorry

end ratio_a5_b5_l299_299564


namespace episodes_lost_per_season_l299_299653

theorem episodes_lost_per_season (s1 s2 : ℕ) (e : ℕ) (remaining : ℕ) (total_seasons : ℕ) (total_episodes_before : ℕ) (total_episodes_lost : ℕ)
  (h1 : s1 = 12) (h2 : s2 = 14) (h3 : e = 16) (h4 : remaining = 364) 
  (h5 : total_seasons = s1 + s2) (h6 : total_episodes_before = s1 * e + s2 * e) 
  (h7 : total_episodes_lost = total_episodes_before - remaining) :
  total_episodes_lost / total_seasons = 2 := by
  sorry

end episodes_lost_per_season_l299_299653


namespace coin_heads_probability_l299_299067

theorem coin_heads_probability
    (prob_tails : ℚ := 1/2)
    (prob_specific_sequence : ℚ := 0.0625)
    (flips : ℕ := 4)
    (ht : prob_tails = 1 / 2)
    (hs : prob_specific_sequence = (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) 
    : ∀ (p_heads : ℚ), p_heads = 1 - prob_tails := by
  sorry

end coin_heads_probability_l299_299067


namespace problem_1_problem_2_l299_299252

open Real

-- Part 1
theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) :=
sorry

-- Part 2
theorem problem_2 : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 4 ∧ (4 / (a * b) + a / b = (1 + sqrt 5) / 2) :=
sorry

end problem_1_problem_2_l299_299252


namespace box_volume_l299_299356

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l299_299356


namespace color_blocks_probability_at_least_one_box_match_l299_299238

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end color_blocks_probability_at_least_one_box_match_l299_299238


namespace lcm_18_24_l299_299778

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299778


namespace find_ab_l299_299033

theorem find_ab (a b q r : ℕ) (h : a > 0) (h2 : b > 0) (h3 : (a^2 + b^2) / (a + b) = q) (h4 : (a^2 + b^2) % (a + b) = r) (h5 : q^2 + r = 2010) : a * b = 1643 :=
sorry

end find_ab_l299_299033


namespace radius_of_circle_is_ten_l299_299497

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

theorem radius_of_circle_is_ten :
  radius_of_circle 20 = 10 :=
by
  unfold radius_of_circle
  sorry

end radius_of_circle_is_ten_l299_299497


namespace unique_intersection_point_l299_299032

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point : ∃ a : ℝ, f a = a ∧ f a = -1 ∧ f a = f⁻¹ a :=
by 
  sorry

end unique_intersection_point_l299_299032


namespace find_m_l299_299416

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (m : ℝ) (hS : ∀ n, S n = m * 2^(n-1) - 3) 
               (ha1 : a 1 = S 1) (han : ∀ n > 1, a n = S n - S (n - 1)) 
               (ratio : ∀ n > 1, a (n+1) / a n = 1/2): 
  m = 6 := 
sorry

end find_m_l299_299416


namespace triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l299_299920

theorem triangle_a_eq_5_over_3
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : b = Real.sqrt 5 * Real.sin B) :
  a = 5 / 3 := sorry

theorem triangle_b_plus_c_eq_4
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : a = Real.sqrt 6)
  (h3 : 1 / 2 * b * c * Real.sin A = Real.sqrt 5 / 2) :
  b + c = 4 := sorry

end triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l299_299920


namespace probability_no_success_l299_299658

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l299_299658


namespace popsicle_total_l299_299927

def popsicle_count (g c b : Nat) : Nat :=
  g + c + b

theorem popsicle_total : 
  let g := 2
  let c := 13
  let b := 2
  popsicle_count g c b = 17 := by
  sorry

end popsicle_total_l299_299927


namespace Karen_sold_boxes_l299_299029

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end Karen_sold_boxes_l299_299029


namespace sum_of_arithmetic_sequence_l299_299720

theorem sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) 
  (h1 : ∀ n, S n = n * a₁ + (n - 1) * n / 2 * d)
  (h2 : S 1 / S 4 = 1 / 10) :
  S 3 / S 5 = 2 / 5 := 
sorry

end sum_of_arithmetic_sequence_l299_299720


namespace downstream_speed_is_28_l299_299836

-- Define the speed of the man in still water
def speed_in_still_water : ℝ := 24

-- Define the speed of the man rowing upstream
def speed_upstream : ℝ := 20

-- Define the speed of the stream
def speed_stream : ℝ := speed_in_still_water - speed_upstream

-- Define the speed of the man rowing downstream
def speed_downstream : ℝ := speed_in_still_water + speed_stream

-- The main theorem stating that the speed of the man rowing downstream is 28 kmph
theorem downstream_speed_is_28 : speed_downstream = 28 := by
  sorry

end downstream_speed_is_28_l299_299836


namespace sets_equal_l299_299234

def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, abs (-(Real.sqrt 3))}

theorem sets_equal : A = B :=
by 
  sorry

end sets_equal_l299_299234


namespace opposite_of_pi_is_neg_pi_l299_299964

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l299_299964


namespace box_volume_l299_299358

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l299_299358


namespace increased_colored_area_l299_299041

theorem increased_colored_area
  (P : ℝ) -- Perimeter of the original convex pentagon
  (s : ℝ) -- Distance from the points colored originally
  : 
  s * P + π * s^2 = 23.14 :=
by
  sorry

end increased_colored_area_l299_299041


namespace greatest_x_l299_299766

theorem greatest_x (x : ℕ) (h_pos : 0 < x) (h_ineq : (x^6) / (x^3) < 18) : x = 2 :=
by sorry

end greatest_x_l299_299766


namespace difference_of_squares_l299_299274

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := 
sorry

end difference_of_squares_l299_299274


namespace Zoe_siblings_l299_299318

structure Child where
  eyeColor : String
  hairColor : String
  height : String

def Emma : Child := { eyeColor := "Green", hairColor := "Red", height := "Tall" }
def Zoe : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Short" }
def Liam : Child := { eyeColor := "Green", hairColor := "Brown", height := "Short" }
def Noah : Child := { eyeColor := "Gray", hairColor := "Red", height := "Tall" }
def Mia : Child := { eyeColor := "Green", hairColor := "Red", height := "Short" }
def Lucas : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Tall" }

def sibling (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

theorem Zoe_siblings : sibling Zoe Noah ∧ sibling Zoe Lucas ∧ ∃ x, sibling Noah x ∧ sibling Lucas x :=
by
  sorry

end Zoe_siblings_l299_299318


namespace total_students_high_school_l299_299991

theorem total_students_high_school (s10 s11 s12 total_students sample: ℕ ) 
  (h1 : s10 = 600) 
  (h2 : sample = 45) 
  (h3 : s11 = 20) 
  (h4 : s12 = 10) 
  (h5 : sample = s10 + s11 + s12) : 
  total_students = 1800 :=
by 
  sorry

end total_students_high_school_l299_299991


namespace find_two_digit_divisors_l299_299510

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_remainder (a b r : ℕ) : Prop := a = b * (a / b) + r

theorem find_two_digit_divisors (n : ℕ) (h1 : is_two_digit n) (h2 : has_remainder 723 n 30) :
  n = 33 ∨ n = 63 ∨ n = 77 ∨ n = 99 :=
sorry

end find_two_digit_divisors_l299_299510


namespace common_roots_cubic_polynomials_l299_299246

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end common_roots_cubic_polynomials_l299_299246


namespace smallest_x_value_l299_299477

theorem smallest_x_value (x : ℤ) (h : 3 * x^2 - 4 < 20) : x = -2 :=
sorry

end smallest_x_value_l299_299477


namespace count_integers_n_satisfying_inequality_l299_299901

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l299_299901


namespace not_in_second_column_l299_299039

theorem not_in_second_column : ¬∃ (n : ℕ), (1 ≤ n ∧ n ≤ 400) ∧ 3 * n + 1 = 131 :=
by sorry

end not_in_second_column_l299_299039


namespace diagonals_of_square_equal_proof_l299_299005

-- Let us define the conditions
def square (s : Type) : Prop := True -- Placeholder for the actual definition of square
def parallelogram (p : Type) : Prop := True -- Placeholder for the actual definition of parallelogram
def diagonals_equal (q : Type) : Prop := True -- Placeholder for the property that diagonals are equal

-- Given conditions
axiom square_is_parallelogram {s : Type} (h1 : square s) : parallelogram s
axiom diagonals_of_parallelogram_equal {p : Type} (h2 : parallelogram p) : diagonals_equal p
axiom diagonals_of_square_equal {s : Type} (h3 : square s) : diagonals_equal s

-- Proof statement
theorem diagonals_of_square_equal_proof (s : Type) (h1 : square s) : diagonals_equal s :=
by
  apply diagonals_of_square_equal h1

end diagonals_of_square_equal_proof_l299_299005


namespace find_natural_solution_l299_299673

theorem find_natural_solution (x y : ℕ) (h : y^6 + 2 * y^3 - y^2 + 1 = x^3) : x = 1 ∧ y = 0 :=
by
  sorry

end find_natural_solution_l299_299673


namespace ship_selection_and_arrangement_l299_299454

def numSelectionsAndArrangements (n m k : ℕ) : ℕ :=
  (Nat.choose n k - Nat.choose m k) * Nat.factorial k

theorem ship_selection_and_arrangement :
  numSelectionsAndArrangements 8 6 3 = 216 := by
  sorry

end ship_selection_and_arrangement_l299_299454


namespace exponent_equation_l299_299701

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l299_299701


namespace cos_B_value_l299_299710

theorem cos_B_value (A B C a b c : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) :
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end cos_B_value_l299_299710


namespace simplify_powers_l299_299740

-- Defining the multiplicative rule for powers
def power_mul (x : ℕ) (a b : ℕ) : ℕ := x^(a+b)

-- Proving that x^5 * x^6 = x^11
theorem simplify_powers (x : ℕ) : x^5 * x^6 = x^11 :=
by
  change x^5 * x^6 = x^(5 + 6)
  sorry

end simplify_powers_l299_299740


namespace lcm_18_24_l299_299800

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299800


namespace find_wrong_observation_value_l299_299599

-- Defining the given conditions
def original_mean : ℝ := 36
def corrected_mean : ℝ := 36.5
def num_observations : ℕ := 50
def correct_value : ℝ := 30

-- Defining the given sums based on means
def original_sum : ℝ := num_observations * original_mean
def corrected_sum : ℝ := num_observations * corrected_mean

-- The wrong value can be calculated based on the difference
def wrong_value : ℝ := correct_value + (corrected_sum - original_sum)

-- The theorem to prove
theorem find_wrong_observation_value (h : original_sum = 1800) (h' : corrected_sum = 1825) :
  wrong_value = 55 :=
sorry

end find_wrong_observation_value_l299_299599


namespace circumscribed_center_on_Ox_axis_l299_299465

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end circumscribed_center_on_Ox_axis_l299_299465


namespace sum_of_three_digit_numbers_l299_299731

theorem sum_of_three_digit_numbers (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  222 * (a + b + c) ≠ 2021 := 
sorry

end sum_of_three_digit_numbers_l299_299731


namespace cats_count_l299_299098

-- Definitions based on conditions
def heads_eqn (H C : ℕ) : Prop := H + C = 15
def legs_eqn (H C : ℕ) : Prop := 2 * H + 4 * C = 44

-- The main proof problem
theorem cats_count (H C : ℕ) (h1 : heads_eqn H C) (h2 : legs_eqn H C) : C = 7 :=
by
  sorry

end cats_count_l299_299098


namespace find_ordered_pairs_l299_299674

theorem find_ordered_pairs (x y : ℝ) :
  x^2 * y = 3 ∧ x + x * y = 4 → (x, y) = (1, 3) ∨ (x, y) = (3, 1 / 3) :=
sorry

end find_ordered_pairs_l299_299674


namespace homework_duration_equation_l299_299073

-- Define the initial and final durations and the rate of decrease
def initial_duration : ℝ := 100
def final_duration : ℝ := 70
def rate_of_decrease (x : ℝ) : ℝ := x

-- Statement of the proof problem
theorem homework_duration_equation (x : ℝ) :
  initial_duration * (1 - rate_of_decrease x) ^ 2 = final_duration :=
sorry

end homework_duration_equation_l299_299073


namespace probability_no_success_l299_299661

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l299_299661


namespace oranges_to_put_back_l299_299621

theorem oranges_to_put_back
  (p_A p_O : ℕ)
  (A O : ℕ)
  (total_fruits : ℕ)
  (initial_avg_price new_avg_price : ℕ)
  (x : ℕ)
  (h1 : p_A = 40)
  (h2 : p_O = 60)
  (h3 : total_fruits = 15)
  (h4 : initial_avg_price = 48)
  (h5 : new_avg_price = 45)
  (h6 : A + O = total_fruits)
  (h7 : (p_A * A + p_O * O) / total_fruits = initial_avg_price)
  (h8 : (720 - 60 * x) / (15 - x) = 45) :
  x = 3 :=
by
  sorry

end oranges_to_put_back_l299_299621


namespace smallest_single_discount_more_advantageous_l299_299250

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  (∀ n, 0 < n -> (1 - (n:ℝ)/100) < 0.64 ∧ (1 - (n:ℝ)/100) < 0.658503 ∧ (1 - (n:ℝ)/100) < 0.63) → 
  n = 38 := 
sorry

end smallest_single_discount_more_advantageous_l299_299250


namespace adult_ticket_cost_l299_299473

variables (x : ℝ)

-- Conditions
def total_tickets := 510
def senior_tickets := 327
def senior_ticket_cost := 15
def total_receipts := 8748

-- Calculation based on the conditions
def adult_tickets := total_tickets - senior_tickets
def senior_receipts := senior_tickets * senior_ticket_cost
def adult_receipts := total_receipts - senior_receipts

-- Define the problem as an assertion to prove
theorem adult_ticket_cost :
  adult_receipts / adult_tickets = 21 := by
  -- Proof steps will go here, but for now, we'll use sorry.
  sorry

end adult_ticket_cost_l299_299473


namespace karen_boxes_l299_299030

theorem karen_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by {
  rw [h_cases, h_boxes_per_case],
  norm_num,
  sorry
}

end karen_boxes_l299_299030


namespace find_4_oplus_2_l299_299936

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l299_299936


namespace profit_per_meter_correct_l299_299842

noncomputable def total_selling_price := 6788
noncomputable def num_meters := 78
noncomputable def cost_price_per_meter := 58.02564102564102
noncomputable def total_cost_price := 4526 -- rounded total
noncomputable def total_profit := 2262 -- calculated total profit
noncomputable def profit_per_meter := 29

theorem profit_per_meter_correct :
  (total_selling_price - total_cost_price) / num_meters = profit_per_meter :=
by
  sorry

end profit_per_meter_correct_l299_299842


namespace total_apples_collected_l299_299387

-- Definitions based on conditions
def number_of_green_apples : ℕ := 124
def number_of_red_apples : ℕ := 3 * number_of_green_apples

-- Proof statement
theorem total_apples_collected : number_of_red_apples + number_of_green_apples = 496 := by
  sorry

end total_apples_collected_l299_299387


namespace find_angle_degree_l299_299116

theorem find_angle_degree (x : ℝ) (h1 : 90 - x = (2 / 5) * (180 - x)) : x = 30 :=
sorry

end find_angle_degree_l299_299116


namespace rita_bought_four_pounds_l299_299951

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l299_299951


namespace water_tank_capacity_l299_299082

theorem water_tank_capacity
  (tank_capacity : ℝ)
  (h : 0.30 * tank_capacity = 0.90 * tank_capacity - 54) :
  tank_capacity = 90 :=
by
  -- proof goes here
  sorry

end water_tank_capacity_l299_299082


namespace seconds_in_minutes_l299_299263

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l299_299263


namespace mostSuitableForComprehensiveSurvey_l299_299362

-- Definitions of conditions
def optionA := "Understanding the sleep time of middle school students nationwide"
def optionB := "Understanding the water quality of a river"
def optionC := "Surveying the vision of all classmates"
def optionD := "Surveying the number of fish in a pond"

-- Define the notion of being the most suitable option for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : String) := option = optionC

-- The theorem statement
theorem mostSuitableForComprehensiveSurvey : isSuitableForComprehensiveSurvey optionC := by
  -- This is the Lean 4 statement where we accept the hypotheses
  -- and conclude the theorem. Proof is omitted with "sorry".
  sorry

end mostSuitableForComprehensiveSurvey_l299_299362


namespace max_k_inequality_l299_299405

theorem max_k_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∀ k ≤ 2, ( ( (b - c) ^ 2 * (b + c) / a ) + 
             ( (c - a) ^ 2 * (c + a) / b ) + 
             ( (a - b) ^ 2 * (a + b) / c ) 
             ≥ k * ( a^2 + b^2 + c^2 - a*b - b*c - c*a ) ) :=
by
  sorry

end max_k_inequality_l299_299405


namespace not_minimum_on_l299_299134

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x ^ 2 - m * x

theorem not_minimum_on (m : ℝ) : 
  ¬ (∃ x ∈ Set.Icc 1 2, f x m = Real.exp 2 - 2 * m ∧ 
  ∀ y ∈ Set.Icc 1 2, f y m ≥ f x m) :=
sorry

end not_minimum_on_l299_299134


namespace savanna_total_animals_l299_299452

def num_lions_safari := 100
def num_snakes_safari := num_lions_safari / 2
def num_giraffes_safari := num_snakes_safari - 10
def num_elephants_safari := num_lions_safari / 4

def num_lions_savanna := num_lions_safari * 2
def num_snakes_savanna := num_snakes_safari * 3
def num_giraffes_savanna := num_giraffes_safari + 20
def num_elephants_savanna := num_elephants_safari * 5
def num_zebras_savanna := (num_lions_savanna + num_snakes_savanna) / 2

def total_animals_savanna := 
  num_lions_savanna 
  + num_snakes_savanna 
  + num_giraffes_savanna 
  + num_elephants_savanna 
  + num_zebras_savanna

open Nat
theorem savanna_total_animals : total_animals_savanna = 710 := by
  sorry

end savanna_total_animals_l299_299452


namespace whiskers_ratio_l299_299542

/-- Four cats live in the old grey house at the end of the road. Their names are Puffy, Scruffy, Buffy, and Juniper.
Puffy has three times more whiskers than Juniper, but a certain ratio as many as Scruffy. Buffy has the same number of whiskers
as the average number of whiskers on the three other cats. Prove that the ratio of Puffy's whiskers to Scruffy's whiskers is 1:2
given Juniper has 12 whiskers and Buffy has 40 whiskers. -/
theorem whiskers_ratio (J B P S : ℕ) (hJ : J = 12) (hB : B = 40) (hP : P = 3 * J) (hAvg : B = (P + S + J) / 3) :
  P / gcd P S = 1 ∧ S / gcd P S = 2 := by
  sorry

end whiskers_ratio_l299_299542


namespace correct_option_given_inequality_l299_299012

theorem correct_option_given_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
sorry

end correct_option_given_inequality_l299_299012


namespace permutations_mississippi_correct_l299_299526

def permutations_mississippi : ℕ :=
  Nat.factorial 11 / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2)

theorem permutations_mississippi_correct : permutations_mississippi = 34650 :=
  by {
    unfold permutations_mississippi,
    -- Calculation steps omitted
    sorry
  }

end permutations_mississippi_correct_l299_299526


namespace combined_width_approximately_8_l299_299971

noncomputable def C1 := 352 / 7
noncomputable def C2 := 528 / 7
noncomputable def C3 := 704 / 7

noncomputable def r1 := C1 / (2 * Real.pi)
noncomputable def r2 := C2 / (2 * Real.pi)
noncomputable def r3 := C3 / (2 * Real.pi)

noncomputable def W1 := r2 - r1
noncomputable def W2 := r3 - r2

noncomputable def combined_width := W1 + W2

theorem combined_width_approximately_8 :
  |combined_width - 8| < 1 :=
by
  sorry

end combined_width_approximately_8_l299_299971


namespace lcm_18_24_l299_299803

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299803


namespace conditional_expectation_inequality_l299_299437

open ProbabilityTheory

variables {Ω : Type*} [MeasureSpace Ω]
variable (X Y Z : Ω → ℝ)
variable [Measurable X] [Measurable Y] [Measurable Z]
variable (hXY_ind : IndepFun X Y) (h2X_fin : E[X^2] < ∞) (h2Y_fin : E[Y^2] < ∞)
variable (h2Z_fin : E[Z^2] < ∞) (hZ_zero_mean : E[Z] = 0)

theorem conditional_expectation_inequality
  (h_assumptions : IndepFun X Y ∧ E[X^2] < ∞ ∧ E[Y^2] < ∞ ∧ E[Z^2] < ∞ ∧ E[Z] = 0) :
  E[(∥ conditionalExpectations Z X ∥^2] + E[(∥ conditionalExpectations Z Y ∥^2] ≤ E[Z^2] := 
sorry

end conditional_expectation_inequality_l299_299437


namespace meaningful_sqrt_range_l299_299708

theorem meaningful_sqrt_range (x : ℝ) (h : 0 ≤ x + 3) : -3 ≤ x :=
by sorry

end meaningful_sqrt_range_l299_299708


namespace seconds_in_minutes_l299_299264

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end seconds_in_minutes_l299_299264


namespace equation_one_solution_equation_two_solution_l299_299744

theorem equation_one_solution (x : ℝ) (h : 7 * x - 20 = 2 * (3 - 3 * x)) : x = 2 :=
by {
  sorry
}

theorem equation_two_solution (x : ℝ) (h : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1) : x = -1 :=
by {
  sorry
}

end equation_one_solution_equation_two_solution_l299_299744


namespace solve_for_x_l299_299050

theorem solve_for_x : (1 / 3 - 1 / 4) * 2 = 1 / 6 :=
by
  -- Sorry is used to skip the proof; the proof steps are not included.
  sorry

end solve_for_x_l299_299050


namespace gold_distribution_l299_299928

theorem gold_distribution :
  ∃ (d : ℚ), 
    (4 * (a1: ℚ) + 6 * d = 3) ∧ 
    (3 * (a1: ℚ) + 24 * d = 4) ∧
    d = 7 / 78 :=
by {
  sorry
}

end gold_distribution_l299_299928


namespace solve_equations_l299_299954

theorem solve_equations :
  (∀ x : ℝ, x^2 - 2 * x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 0 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equations_l299_299954


namespace find_4_oplus_2_l299_299937

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l299_299937


namespace apples_total_l299_299160

theorem apples_total (initial_apples : ℕ) (additional_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 56 → 
  additional_apples = 49 → 
  total_apples = initial_apples + additional_apples → 
  total_apples = 105 :=
by 
  intros h_initial h_additional h_total 
  rw [h_initial, h_additional] at h_total 
  exact h_total

end apples_total_l299_299160


namespace triangle_area_ratio_l299_299071

theorem triangle_area_ratio (a b c a' b' c' r : ℝ)
    (h1 : a^2 + b^2 = c^2)
    (h2 : a'^2 + b'^2 = c'^2)
    (h3 : r = c' / 2)
    (S : ℝ := (1/2) * a * b)
    (S' : ℝ := (1/2) * a' * b') :
    S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_ratio_l299_299071


namespace carlos_earnings_l299_299431

theorem carlos_earnings (h1 : ∃ w, 18 * w = w * 18) (h2 : ∃ w, 30 * w = w * 30) (h3 : ∀ w, 30 * w - 18 * w = 54) : 
  ∃ w, 18 * w + 30 * w = 216 := 
sorry

end carlos_earnings_l299_299431


namespace find_constant_A_l299_299955

theorem find_constant_A :
  ∀ (x : ℝ)
  (A B C D : ℝ),
      (
        (1 : ℝ) / (x^4 - 20 * x^3 + 147 * x^2 - 490 * x + 588) = 
        (A / (x + 3)) + (B / (x - 4)) + (C / ((x - 4)^2)) + (D / (x - 7))
      ) →
      A = - (1 / 490) := 
by 
  intro x A B C D h
  sorry

end find_constant_A_l299_299955


namespace maximum_b_n_T_l299_299687

/-- Given a sequence {a_n} defined recursively and b_n = a_n / n.
   We need to prove that for all n in positive natural numbers,
   b_n is greater than or equal to T, and the maximum such T is 3. -/
theorem maximum_b_n_T (T : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  (∀ n, n ≥ 1 → b n = a n / n) →
  (∀ n, n ≥ 1 → b n ≥ T) →
  T ≤ 3 :=
by
  sorry

end maximum_b_n_T_l299_299687


namespace count_integers_satisfy_inequality_l299_299905

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l299_299905


namespace craft_store_pricing_maximize_daily_profit_l299_299489

theorem craft_store_pricing (profit_per_item marked_price cost_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₂ : 8 * 0.85 * marked_price + 12 * (marked_price - 35) = 20 * cost_price)
  : cost_price = 155 ∧ marked_price = 200 := 
sorry

theorem maximize_daily_profit (profit_per_item cost_price marked_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₃ : ∀ p : ℝ, (100 + 4 * (200 - p)) * (p - cost_price) ≤ 4900)
  : p = 190 ∧ daily_profit = 4900 :=
sorry

end craft_store_pricing_maximize_daily_profit_l299_299489


namespace total_logs_in_stack_l299_299378

/-- The total number of logs in a stack where the top row has 5 logs,
each succeeding row has one more log than the one above,
and the bottom row has 15 logs. -/
theorem total_logs_in_stack :
  let a := 5               -- first term (logs in the top row)
  let l := 15              -- last term (logs in the bottom row)
  let n := l - a + 1       -- number of terms (rows)
  let S := n / 2 * (a + l) -- sum of the arithmetic series
  S = 110 := sorry

end total_logs_in_stack_l299_299378


namespace count_integers_in_interval_l299_299908

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l299_299908


namespace opposite_of_pi_l299_299965

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l299_299965


namespace children_playing_both_sports_l299_299712

variable (total_children : ℕ) (T : ℕ) (S : ℕ) (N : ℕ)

theorem children_playing_both_sports 
  (h1 : total_children = 38) 
  (h2 : T = 19) 
  (h3 : S = 21) 
  (h4 : N = 10) : 
  (T + S) - (total_children - N) = 12 := 
by
  sorry

end children_playing_both_sports_l299_299712


namespace club_men_count_l299_299990

theorem club_men_count (M W : ℕ) (h1 : M + W = 30) (h2 : M + (W / 3 : ℕ) = 20) : M = 15 := by
  -- proof omitted
  sorry

end club_men_count_l299_299990


namespace fraction_division_l299_299343

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299343


namespace mr_c_gain_1000_l299_299172

-- Define the initial conditions
def initial_mr_c_cash := 15000
def initial_mr_c_house := 12000
def initial_mrs_d_cash := 16000

-- Define the changes in the house value
def house_value_appreciated := 13000
def house_value_depreciated := 11000

-- Define the cash changes after transactions
def mr_c_cash_after_first_sale := initial_mr_c_cash + house_value_appreciated
def mrs_d_cash_after_first_sale := initial_mrs_d_cash - house_value_appreciated
def mrs_d_cash_after_second_sale := mrs_d_cash_after_first_sale + house_value_depreciated
def mr_c_cash_after_second_sale := mr_c_cash_after_first_sale - house_value_depreciated

-- Define the final net worth for Mr. C
def final_mr_c_cash := mr_c_cash_after_second_sale
def final_mr_c_house := house_value_depreciated
def final_mr_c_net_worth := final_mr_c_cash + final_mr_c_house
def initial_mr_c_net_worth := initial_mr_c_cash + initial_mr_c_house

-- Statement to prove
theorem mr_c_gain_1000 : final_mr_c_net_worth = initial_mr_c_net_worth + 1000 := by
  sorry

end mr_c_gain_1000_l299_299172


namespace find_abc_l299_299675

theorem find_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : a^3 + b^3 + c^3 = 2001 → (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) := 
sorry

end find_abc_l299_299675


namespace trigonometric_identity_l299_299142

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) : 
  (1 / Real.cos (2 * α)) + Real.tan (2 * α) = 2012 := 
by
  -- This will be the proof body which we omit with sorry
  sorry

end trigonometric_identity_l299_299142


namespace fraction_division_l299_299076

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l299_299076


namespace lcm_18_24_eq_72_l299_299781

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299781


namespace rancher_cows_l299_299375

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end rancher_cows_l299_299375


namespace total_time_six_laps_l299_299742

-- Defining the constants and conditions
def total_distance : Nat := 500
def speed_part1 : Nat := 3
def distance_part1 : Nat := 150
def speed_part2 : Nat := 6
def distance_part2 : Nat := total_distance - distance_part1
def laps : Nat := 6

-- Calculating the times based on conditions
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2
def time_per_lap := time_part1 + time_part2
def total_time := laps * time_per_lap

-- The goal is to prove the total time is 10 minutes and 48 seconds (648 seconds)
theorem total_time_six_laps : total_time = 648 :=
-- proof would go here
sorry

end total_time_six_laps_l299_299742


namespace cute_pairs_count_l299_299471

def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

def max_cute_pairs : Prop :=
  ∀ (ages : Finset ℕ), 
  (∀ x ∈ ages, 1 ≤ x ∧ x ≤ 100) →
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ pair ∈ pairs, is_cute_pair pair.1 pair.2) ∧
    (∀ x ∈ pairs, ∀ y ∈ pairs, x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) ∧
    pairs.card = 43)

theorem cute_pairs_count : max_cute_pairs := 
sorry

end cute_pairs_count_l299_299471


namespace lassis_from_mangoes_l299_299866

def ratio (lassis mangoes : ℕ) : Prop := lassis = 11 * mangoes / 2

theorem lassis_from_mangoes (mangoes : ℕ) (h : mangoes = 10) : ratio 55 mangoes :=
by
  rw [h]
  unfold ratio
  sorry

end lassis_from_mangoes_l299_299866


namespace custom_op_evaluation_l299_299939

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l299_299939


namespace minimize_expression_l299_299130

theorem minimize_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
(h4 : x^2 + y^2 + z^2 = 1) : 
  z = Real.sqrt 2 - 1 :=
sorry

end minimize_expression_l299_299130


namespace find_m_l299_299913

theorem find_m (x n m : ℝ) (h : (x + n)^2 = x^2 + 4*x + m) : m = 4 :=
sorry

end find_m_l299_299913


namespace page_sum_incorrect_l299_299324

theorem page_sum_incorrect (sheets : List (Nat × Nat)) (h_sheets_len : sheets.length = 25)
  (h_consecutive : ∀ (a b : Nat), (a, b) ∈ sheets → (b = a + 1 ∨ a = b + 1))
  (h_sum_eq_2020 : (sheets.map (λ p => p.1 + p.2)).sum = 2020) : False :=
by
  sorry

end page_sum_incorrect_l299_299324


namespace solution_to_first_equation_solution_to_second_equation_l299_299179

theorem solution_to_first_equation (x : ℝ) : 
  x^2 - 6 * x + 1 = 0 ↔ x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2 :=
by sorry

theorem solution_to_second_equation (x : ℝ) : 
  (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
by sorry

end solution_to_first_equation_solution_to_second_equation_l299_299179


namespace ratio_Rachel_Sara_l299_299311

-- Define Sara's spending
def Sara_shoes_spending : ℝ := 50
def Sara_dress_spending : ℝ := 200

-- Define Rachel's budget
def Rachel_budget : ℝ := 500

-- Calculate Sara's total spending
def Sara_total_spending : ℝ := Sara_shoes_spending + Sara_dress_spending

-- Define the theorem to prove the ratio
theorem ratio_Rachel_Sara : (Rachel_budget / Sara_total_spending) = 2 := by
  -- Proof is omitted (you would fill in the proof here)
  sorry

end ratio_Rachel_Sara_l299_299311


namespace ratio_preference_l299_299043

-- Definitions based on conditions
def total_respondents : ℕ := 180
def preferred_brand_x : ℕ := 150
def preferred_brand_y : ℕ := total_respondents - preferred_brand_x

-- Theorem statement to prove the ratio of preferences
theorem ratio_preference : preferred_brand_x / preferred_brand_y = 5 := by
  sorry

end ratio_preference_l299_299043


namespace brownies_pieces_count_l299_299037

-- Definitions of the conditions
def pan_length : ℕ := 24
def pan_width : ℕ := 15
def pan_area : ℕ := pan_length * pan_width -- pan_area = 360

def piece_length : ℕ := 3
def piece_width : ℕ := 2
def piece_area : ℕ := piece_length * piece_width -- piece_area = 6

-- Definition of the question and proving the expected answer
theorem brownies_pieces_count : (pan_area / piece_area) = 60 := by
  sorry

end brownies_pieces_count_l299_299037


namespace simplify_polynomial_l299_299741

variable (x : ℝ)

theorem simplify_polynomial :
  (6*x^10 + 8*x^9 + 3*x^7) + (2*x^12 + 3*x^10 + x^9 + 5*x^7 + 4*x^4 + 7*x + 6) =
  2*x^12 + 9*x^10 + 9*x^9 + 8*x^7 + 4*x^4 + 7*x + 6 :=
by
  sorry

end simplify_polynomial_l299_299741


namespace g_of_f_eq_l299_299723

def f (A B x : ℝ) : ℝ := A * x^2 - B^2
def g (B x : ℝ) : ℝ := B * x + B^2

theorem g_of_f_eq (A B : ℝ) (hB : B ≠ 0) : 
  g B (f A B 1) = B * A - B^3 + B^2 := 
by
  sorry

end g_of_f_eq_l299_299723


namespace find_d_l299_299414

-- Define the proportional condition
def in_proportion (a b c d : ℕ) : Prop := a * d = b * c

-- Given values as parameters
variables {a b c d : ℕ}

-- Theorem to be proven
theorem find_d (h : in_proportion a b c d) (ha : a = 1) (hb : b = 2) (hc : c = 3) : d = 6 :=
sorry

end find_d_l299_299414


namespace partition_exists_min_n_in_A_l299_299499

-- Definition of subsets and their algebraic properties
variable (A B C : Set ℕ)

-- The Initial conditions
axiom A_squared_eq_A : ∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)
axiom B_squared_eq_C : ∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)
axiom C_squared_eq_B : ∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)
axiom AB_eq_B : ∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)
axiom AC_eq_C : ∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)
axiom BC_eq_A : ∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)

-- Statement for the partition existence with given conditions
theorem partition_exists :
  ∃ A B C : Set ℕ, (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
               (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
               (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
               (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
               (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
               (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) :=
sorry

-- Statement for the minimum n in A such that n and n+1 are both in A is at most 77
theorem min_n_in_A :
  ∀ A B C : Set ℕ,
    (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
    (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
    (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
    (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
    (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
    (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) →
    ∃ n : ℕ, (n ∈ A) ∧ (n + 1 ∈ A) ∧ n ≤ 77 :=
sorry

end partition_exists_min_n_in_A_l299_299499


namespace chessboard_disk_cover_l299_299629

noncomputable def chessboardCoveredSquares : ℕ :=
  let D : ℝ := 1 -- assuming D is a positive real number; actual value irrelevant as it gets cancelled in the comparison
  let grid_size : ℕ := 8
  let total_squares : ℕ := grid_size * grid_size
  let boundary_squares : ℕ := 28 -- pre-calculated in the insides steps
  let interior_squares : ℕ := total_squares - boundary_squares
  let non_covered_corners : ℕ := 4
  interior_squares - non_covered_corners

theorem chessboard_disk_cover : chessboardCoveredSquares = 32 := sorry

end chessboard_disk_cover_l299_299629


namespace arithmetic_sequence_l299_299297

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l299_299297


namespace irrational_number_line_representation_l299_299214

theorem irrational_number_line_representation :
  ∀ (x : ℝ), ¬ (∃ r s : ℚ, x = r / s ∧ r ≠ 0 ∧ s ≠ 0) → ∃ p : ℝ, x = p := 
by
  sorry

end irrational_number_line_representation_l299_299214


namespace min_value_g_geq_6_min_value_g_eq_6_l299_299680

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l299_299680


namespace find_third_side_of_triangle_l299_299472

noncomputable def area_triangle_given_sides_angle {a b c : ℝ} (A : ℝ) : Prop :=
  A = 1/2 * a * b * Real.sin c

noncomputable def cosine_law_third_side {a b c : ℝ} (cosα : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosα

theorem find_third_side_of_triangle (a b : ℝ) (Area : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_b : b = 3) (h_Area : Area = 3) :
  ∃ c : ℝ, (c = Real.sqrt 5 ∨ c = Real.sqrt 29) :=
by
  sorry

end find_third_side_of_triangle_l299_299472


namespace simplify_expression_l299_299334

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l299_299334


namespace red_section_no_damage_probability_l299_299655

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l299_299655


namespace f_five_eq_three_f_three_x_inv_f_243_l299_299013

-- Define the function f satisfying the given conditions.
def f (x : ℕ) : ℕ :=
  if x = 5 then 3
  else if x = 15 then 9
  else if x = 45 then 27
  else if x = 135 then 81
  else if x = 405 then 243
  else 0

-- Define the condition f(5) = 3
theorem f_five_eq_three : f 5 = 3 := rfl

-- Define the condition f(3x) = 3f(x) for all x
theorem f_three_x (x : ℕ) : f (3 * x) = 3 * f x :=
sorry

-- Prove that f⁻¹(243) = 405.
theorem inv_f_243 : f (405) = 243 :=
by sorry

-- Concluding the proof statement using the concluded theorems.
example : f (405) = 243 :=
by apply inv_f_243

end f_five_eq_three_f_three_x_inv_f_243_l299_299013


namespace emily_purchased_9_wall_prints_l299_299873

/-
  Given the following conditions:
  - cost_of_each_pair_of_curtains = 30
  - num_of_pairs_of_curtains = 2
  - installation_cost = 50
  - cost_of_each_wall_print = 15
  - total_order_cost = 245

  Prove that Emily purchased 9 wall prints
-/
noncomputable def num_wall_prints_purchased 
  (cost_of_each_pair_of_curtains : ℝ) 
  (num_of_pairs_of_curtains : ℝ) 
  (installation_cost : ℝ) 
  (cost_of_each_wall_print : ℝ) 
  (total_order_cost : ℝ) 
  : ℝ :=
  (total_order_cost - (num_of_pairs_of_curtains * cost_of_each_pair_of_curtains + installation_cost)) / cost_of_each_wall_print

theorem emily_purchased_9_wall_prints
  (cost_of_each_pair_of_curtains : ℝ := 30) 
  (num_of_pairs_of_curtains : ℝ := 2) 
  (installation_cost : ℝ := 50) 
  (cost_of_each_wall_print : ℝ := 15) 
  (total_order_cost : ℝ := 245) :
  num_wall_prints_purchased cost_of_each_pair_of_curtains num_of_pairs_of_curtains installation_cost cost_of_each_wall_print total_order_cost = 9 :=
sorry

end emily_purchased_9_wall_prints_l299_299873


namespace isosceles_triangle_largest_angle_l299_299283

theorem isosceles_triangle_largest_angle (α : ℝ) (β : ℝ)
  (h1 : 0 < α) (h2 : α = 30) (h3 : β = 30):
  ∃ γ : ℝ, γ = 180 - 2 * α ∧ γ = 120 := by
  sorry

end isosceles_triangle_largest_angle_l299_299283


namespace proportion_equiv_l299_299501

theorem proportion_equiv (X : ℕ) (h : 8 / 4 = X / 240) : X = 480 :=
by
  sorry

end proportion_equiv_l299_299501


namespace cubic_roots_sum_cube_l299_299036

theorem cubic_roots_sum_cube (a b c : ℂ) (h : ∀x : ℂ, (x=a ∨ x=b ∨ x=c) → (x^3 - 2*x^2 + 3*x - 4 = 0)) : a^3 + b^3 + c^3 = 2 :=
sorry

end cubic_roots_sum_cube_l299_299036


namespace fraction_multiplication_validity_l299_299541

theorem fraction_multiplication_validity (a b m x : ℝ) (hb : b ≠ 0) : 
  (x ≠ m) ↔ (b * (x - m) ≠ 0) :=
by
  sorry

end fraction_multiplication_validity_l299_299541


namespace evaluate_expression_at_x_eq_2_l299_299977

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l299_299977


namespace triangle_angleC_l299_299565

noncomputable def angleC_possible_values
  (a b : ℝ) (A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hA : A = Real.pi / 4) : Prop :=
  ∃ C : ℝ, (C = Real.pi * 5 / 12 ∨ C = Real.pi / 12)

theorem triangle_angleC
  (a b : ℝ) (A C : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hA : A = Real.pi / 4) 
  (hC: C = Real.pi * 5 / 12 ∨ C = Real.pi / 12) :
  angleC_possible_values a b A :=
begin
  sorry
end

end triangle_angleC_l299_299565


namespace square_area_eq_36_l299_299101

theorem square_area_eq_36 (A_triangle : ℝ) (P_triangle : ℝ) 
  (h1 : A_triangle = 16 * Real.sqrt 3)
  (h2 : P_triangle = 3 * (Real.sqrt (16 * 4 * Real.sqrt 3)))
  (h3 : ∀ a, 4 * a = P_triangle) : 
  a^2 = 36 :=
by sorry

end square_area_eq_36_l299_299101


namespace Darren_paints_432_feet_l299_299666

theorem Darren_paints_432_feet (t : ℝ) (h : t = 792) (paint_ratio : ℝ) 
  (h_ratio : paint_ratio = 1.20) : 
  let d := t / (1 + paint_ratio)
  let D := d * paint_ratio
  D = 432 :=
by
  sorry

end Darren_paints_432_feet_l299_299666


namespace accurate_scale_l299_299222

-- Definitions for the weights on each scale
variables (a b c d e x : ℝ)

-- Given conditions
def condition1 := c = b - 0.3
def condition2 := d = c - 0.1
def condition3 := e = a - 0.1
def condition4 := c = e - 0.1
def condition5 := 5 * x = a + b + c + d + e

-- Proof statement
theorem accurate_scale 
  (h1 : c = b - 0.3)
  (h2 : d = c - 0.1)
  (h3 : e = a - 0.1)
  (h4 : c = e - 0.1)
  (h5 : 5 * x = a + b + c + d + e) : e = x :=
by
  sorry

end accurate_scale_l299_299222


namespace reciprocal_of_neg2019_l299_299328

theorem reciprocal_of_neg2019 : (1 / -2019) = - (1 / 2019) := 
by
  sorry

end reciprocal_of_neg2019_l299_299328


namespace cindys_correct_result_l299_299024

-- Explicitly stating the conditions as definitions
def incorrect_operation_result := 260
def x := (incorrect_operation_result / 5) - 7

theorem cindys_correct_result : 5 * x + 7 = 232 :=
by
  -- Placeholder for the proof
  sorry

end cindys_correct_result_l299_299024


namespace check_number_of_correct_statements_l299_299607

def statement_1 (α : Type) [LinearOrder α] (seq : ℕ → α) (formula : ℕ → α) : Prop :=
∀ n, seq n = formula n ∧ formula = λ n, seq n

def statement_2 (a : ℕ → ℚ) : Prop :=
∀ n, a n = (n+1)/(n+2)

def statement_3 (a : ℕ → ℚ) : Prop :=
∀ x : ℚ, ¬ ∃ n: ℕ, a n = x ∧ a n = a (n+1)

def statement_4 (a b : ℕ → ℚ): Prop :=
∀ n, a n = (-1) ^ n ∧ b n = (-1) ^ (n+1)

theorem check_number_of_correct_statements :
  (statement_1 real (λ n, (n:ℚ) / ((n + 1):ℚ)) → false) ∧
  (statement_2 (λ n, (n:ℚ) / ((n + 1):ℚ)) → false) ∧
  statement_3 (λ n, (n:ℚ) / ((n + 1):ℚ)) ∧
  (statement_4 (λ n, (1:ℚ) * (-1)^n) (λ n, (1:ℚ) * (-1)^(n+1)) → false) →
  true
:= sorry

end check_number_of_correct_statements_l299_299607


namespace chef_used_apples_l299_299833

theorem chef_used_apples (initial_apples remaining_apples used_apples : ℕ) 
  (h1 : initial_apples = 40) 
  (h2 : remaining_apples = 39) 
  (h3 : used_apples = initial_apples - remaining_apples) : 
  used_apples = 1 := 
  sorry

end chef_used_apples_l299_299833


namespace sector_area_is_4_l299_299415

/-- Given a sector of a circle with perimeter 8 and central angle 2 radians,
    the area of the sector is 4. -/
theorem sector_area_is_4 (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l / r = 2) : 
    (1 / 2) * l * r = 4 :=
sorry

end sector_area_is_4_l299_299415


namespace B_investment_time_l299_299999

theorem B_investment_time (x : ℝ) (m : ℝ) :
  let A_share := x * 12
  let B_share := 2 * x * (12 - m)
  let C_share := 3 * x * 4
  let total_gain := 18600
  let A_gain := 6200
  let ratio := A_gain / total_gain
  ratio = 1 / 3 →
  A_share = 1 / 3 * (A_share + B_share + C_share) →
  m = 6 := by
sorry

end B_investment_time_l299_299999


namespace bananas_oranges_equiv_l299_299390

def bananas_apples_equiv (x y : ℕ) : Prop :=
  4 * x = 3 * y

def apples_oranges_equiv (w z : ℕ) : Prop :=
  9 * w = 5 * z

theorem bananas_oranges_equiv (x y w z : ℕ) (h1 : bananas_apples_equiv x y) (h2 : apples_oranges_equiv y z) :
  bananas_apples_equiv 24 18 ∧ apples_oranges_equiv 18 10 :=
by sorry

end bananas_oranges_equiv_l299_299390


namespace seonho_original_money_l299_299176

variable (X : ℝ)
variable (spent_snacks : ℝ := (1/4) * X)
variable (remaining_after_snacks : ℝ := X - spent_snacks)
variable (spent_food : ℝ := (2/3) * remaining_after_snacks)
variable (final_remaining : ℝ := remaining_after_snacks - spent_food)

theorem seonho_original_money :
  final_remaining = 2500 -> X = 10000 := by
  -- Proof goes here
  sorry

end seonho_original_money_l299_299176


namespace John_surveyed_total_people_l299_299716

theorem John_surveyed_total_people :
  ∃ P D : ℝ, 
  0 ≤ P ∧ 
  D = 0.868 * P ∧ 
  21 = 0.457 * D ∧ 
  P = 53 :=
by
  sorry

end John_surveyed_total_people_l299_299716


namespace subset_implies_range_l299_299256

open Set

-- Definitions based on the problem statement
def A : Set ℝ := { x : ℝ | x < 5 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x < a }

-- Theorem statement
theorem subset_implies_range (a : ℝ) (h : A ⊆ B a) : a ≥ 5 :=
sorry

end subset_implies_range_l299_299256


namespace max_blocks_fit_in_box_l299_299612

def box_dimensions : ℕ × ℕ × ℕ := (4, 6, 2)
def block_dimensions : ℕ × ℕ × ℕ := (3, 2, 1)
def block_volume := 6
def box_volume := 48

theorem max_blocks_fit_in_box (box_dimensions : ℕ × ℕ × ℕ)
    (block_dimensions : ℕ × ℕ × ℕ) : 
  (box_volume / block_volume = 8) := 
by
  sorry

end max_blocks_fit_in_box_l299_299612


namespace bacon_percentage_l299_299863

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l299_299863


namespace certain_number_value_l299_299151

theorem certain_number_value
  (t b c x : ℝ)
  (h1 : (t + b + c + x + 15) / 5 = 12)
  (h2 : (t + b + c + 29) / 4 = 15) :
  x = 14 :=
by 
  sorry

end certain_number_value_l299_299151


namespace min_val_m_l299_299438

theorem min_val_m (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h : 24 * m = n ^ 4) : m = 54 :=
sorry

end min_val_m_l299_299438


namespace ryan_lamps_probability_l299_299049

theorem ryan_lamps_probability :
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_ways_to_arrange := Nat.choose total_lamps red_lamps
  let total_ways_to_turn_on := Nat.choose total_lamps 4
  let remaining_blue := blue_lamps - 1 -- Due to leftmost lamp being blue and off
  let remaining_red := red_lamps - 1 -- Due to rightmost lamp being red and on
  let remaining_red_after_middle := remaining_red - 1 -- Due to middle lamp being red and off
  let remaining_lamps := remaining_blue + remaining_red_after_middle
  let ways_to_assign_remaining_red := Nat.choose remaining_lamps remaining_red_after_middle
  let ways_to_turn_on_remaining_lamps := Nat.choose remaining_lamps 2
  let favorable_ways := ways_to_assign_remaining_red * ways_to_turn_on_remaining_lamps
  let total_possibilities := total_ways_to_arrange * total_ways_to_turn_on
  favorable_ways / total_possibilities = (10 / 490) := by
  sorry

end ryan_lamps_probability_l299_299049


namespace union_sets_l299_299128

-- Given sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 4, 5} := by
  sorry

end union_sets_l299_299128


namespace age_difference_l299_299835

def A := 10
def B := 8
def C := B / 2
def total_age (A B C : ℕ) : Prop := A + B + C = 22

theorem age_difference (A B C : ℕ) (hB : B = 8) (hC : B = 2 * C) (h_total : total_age A B C) : A - B = 2 := by
  sorry

end age_difference_l299_299835


namespace number_of_integers_satisfying_ineq_l299_299911

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l299_299911


namespace barbara_typing_time_l299_299851

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l299_299851


namespace michaels_brother_final_amount_l299_299735

theorem michaels_brother_final_amount :
  ∀ (michael_money michael_brother_initial michael_give_half candy_cost money_left : ℕ),
  michael_money = 42 →
  michael_brother_initial = 17 →
  michael_give_half = michael_money / 2 →
  let michael_brother_total := michael_brother_initial + michael_give_half in
  candy_cost = 3 →
  money_left = michael_brother_total - candy_cost →
  money_left = 35 :=
by
  intros michael_money michael_brother_initial michael_give_half candy_cost money_left
  intros h1 h2 h3 michael_brother_total h4 h5
  sorry

end michaels_brother_final_amount_l299_299735


namespace quadratic_root_in_interval_l299_299181

variable (a b c : ℝ)

theorem quadratic_root_in_interval 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end quadratic_root_in_interval_l299_299181


namespace g_min_value_l299_299677

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l299_299677


namespace inequality_solution_l299_299829

theorem inequality_solution (a b : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x < 2) → (ax^2 + bx + 2 > 0)) →
  a + b = 1 :=
by
  sorry

end inequality_solution_l299_299829


namespace intersection_of_sets_l299_299269

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets : (setA ∩ { x | 1 - x^2 ∈ setB }) = Set.Icc (-1) 1 :=
by
  sorry

end intersection_of_sets_l299_299269


namespace find_polynomials_l299_299878

-- Define our polynomial P(x)
def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, (x-1) * P.eval (x+1) - (x+2) * P.eval x = 0

-- State the theorem
theorem find_polynomials (P : Polynomial ℝ) :
  polynomial_condition P ↔ ∃ a : ℝ, P = Polynomial.C a * (Polynomial.X^3 - Polynomial.X) :=
by
  sorry

end find_polynomials_l299_299878


namespace find_n_l299_299704

theorem find_n (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  unfold pow at h
  sorry

end find_n_l299_299704


namespace find_n_from_exponent_equation_l299_299700

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l299_299700


namespace barbara_typing_time_l299_299858

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l299_299858


namespace math_proof_problem_l299_299137

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem math_proof_problem :
  (complement_R A ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end math_proof_problem_l299_299137


namespace percentage_increase_l299_299592

theorem percentage_increase (A B : ℝ) (y : ℝ) (h : A > B) (h1 : B > 0) (h2 : C = A + B) (h3 : C = (1 + y / 100) * B) : y = 100 * (A / B) := 
sorry

end percentage_increase_l299_299592


namespace emily_age_l299_299385

theorem emily_age (A B C D E : ℕ) (h1 : A = B - 4) (h2 : B = C + 5) (h3 : D = C + 2) (h4 : E = A + D - B) (h5 : B = 20) : E = 13 :=
by sorry

end emily_age_l299_299385


namespace volleyball_team_selection_l299_299445

/-- The volleyball team has 16 players including 4 specific quadruplets. The task is to choose 6 starters with exactly 1 quadruplet. -/
theorem volleyball_team_selection : 
  let players_with_quadruplets := 16
  let quadruplets := 4
  let starters := 6
  (∃ (quadruplet_starters: Finset (Fin 16)) (team: Finset (Fin 16)),
    quadruplet_starters.card = 1 ∧ team.card = 5 ∧ quadruplet_starters ∩ team = ∅ ∧
    quadruplet_starters ∪ team ⊆ (Finset.range 16) ∧ quadruplet_starters ∪ team).card = starters 
→  4 * (choose 12 5) = 3168 := 
by
  sorry

end volleyball_team_selection_l299_299445


namespace find_purchase_price_minimum_number_of_speed_skating_shoes_l299_299989

/-
A certain school in Zhangjiakou City is preparing to purchase speed skating shoes and figure skating shoes to promote ice and snow activities on campus.

If they buy 30 pairs of speed skating shoes and 20 pairs of figure skating shoes, the total cost is $8500.
If they buy 40 pairs of speed skating shoes and 10 pairs of figure skating shoes, the total cost is $8000.
The school purchases a total of 50 pairs of both types of ice skates, and the total cost does not exceed $8900.
-/

def price_system (x y : ℝ) : Prop :=
  30 * x + 20 * y = 8500 ∧ 40 * x + 10 * y = 8000

def minimum_speed_skating_shoes (x y m : ℕ) : Prop :=
  150 * m + 200 * (50 - m) ≤ 8900

theorem find_purchase_price :
  ∃ x y : ℝ, price_system x y ∧ x = 150 ∧ y = 200 :=
by
  /- Proof goes here -/
  sorry

theorem minimum_number_of_speed_skating_shoes :
  ∃ m, minimum_speed_skating_shoes 150 200 m ∧ m = 22 :=
by
  /- Proof goes here -/
  sorry

end find_purchase_price_minimum_number_of_speed_skating_shoes_l299_299989


namespace probability_red_or_green_l299_299987

variable (P_brown P_purple P_green P_red P_yellow : ℝ)

def conditions : Prop :=
  P_brown = 0.3 ∧
  P_brown = 3 * P_purple ∧
  P_green = P_purple ∧
  P_red = P_yellow ∧
  P_brown + P_purple + P_green + P_red + P_yellow = 1

theorem probability_red_or_green (h : conditions P_brown P_purple P_green P_red P_yellow) :
  P_red + P_green = 0.35 :=
by
  sorry

end probability_red_or_green_l299_299987


namespace rectangle_perimeter_is_22_l299_299509

-- Definition of sides of the triangle DEF
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Helper function to compute the area of a right triangle
def triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Ensure the triangle is a right triangle and calculate its area
def area_of_triangle : ℕ :=
  if (side1 * side1 + side2 * side2 = hypotenuse * hypotenuse) then
    triangle_area side1 side2
  else
    0

-- Definition of rectangle's width and equation to find its perimeter
def width : ℕ := 5
def rectangle_length : ℕ := area_of_triangle / width
def perimeter_of_rectangle : ℕ := 2 * (width + rectangle_length)

theorem rectangle_perimeter_is_22 : perimeter_of_rectangle = 22 :=
by
  -- Proof content goes here
  sorry

end rectangle_perimeter_is_22_l299_299509


namespace mean_difference_is_882_l299_299747

variable (S : ℤ) (N : ℤ) (S_N_correct : N = 1000)

def actual_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 98000) / N

def incorrect_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 980000) / N

theorem mean_difference_is_882 
  (S : ℤ) 
  (N : ℤ) 
  (S_N_correct : N = 1000) 
  (S_in_range : 8200 ≤ S) 
  (S_actual : S + 98000 ≤ 980000) :
  incorrect_mean S N - actual_mean S N = 882 := 
by
  /- Proof steps would go here -/
  sorry

end mean_difference_is_882_l299_299747


namespace sum_of_d_and_e_l299_299758

theorem sum_of_d_and_e (d e : ℤ) : 
  (∃ d e : ℤ, ∀ x : ℝ, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  sorry

end sum_of_d_and_e_l299_299758


namespace greatest_possible_x_l299_299974

theorem greatest_possible_x (x : ℕ) (h : x^3 < 15) : x ≤ 2 := by
  sorry

end greatest_possible_x_l299_299974


namespace find_principal_l299_299596

theorem find_principal (CI SI : ℝ) (hCI : CI = 11730) (hSI : SI = 10200)
  (P R : ℝ)
  (hSI_form : SI = P * R * 2 / 100)
  (hCI_form : CI = P * (1 + R / 100)^2 - P) :
  P = 34000 := by
  sorry

end find_principal_l299_299596


namespace larger_number_l299_299750

theorem larger_number (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (hcf_eq : hcf = 23) (fact1_eq : factor1 = 13) (fact2_eq : factor2 = 14) : 
  max (hcf * factor1) (hcf * factor2) = 322 := 
by
  sorry

end larger_number_l299_299750


namespace work_completion_l299_299620

theorem work_completion (a b : ℕ) (h1 : a + b = 5) (h2 : a = 10) : b = 10 := by
  sorry

end work_completion_l299_299620


namespace solve_farm_l299_299280

def farm_problem (P H L T : ℕ) : Prop :=
  L = 4 * P + 2 * H ∧
  T = P + H ∧
  L = 3 * T + 36 →
  P = H + 36

-- Theorem statement
theorem solve_farm : ∃ P H L T : ℕ, farm_problem P H L T :=
by sorry

end solve_farm_l299_299280


namespace electricity_average_l299_299834

-- Define the daily electricity consumptions
def electricity_consumptions : List ℕ := [110, 101, 121, 119, 114]

-- Define the function to calculate the average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Formalize the proof problem
theorem electricity_average :
  average electricity_consumptions = 113 :=
  sorry

end electricity_average_l299_299834


namespace nilpotent_matrix_squared_zero_l299_299165

variable {R : Type*} [Field R]
variable (A : Matrix (Fin 2) (Fin 2) R)

theorem nilpotent_matrix_squared_zero (h : A^4 = 0) : A^2 = 0 := 
sorry

end nilpotent_matrix_squared_zero_l299_299165


namespace find_f_3_l299_299944

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : 
  (∀ (x : ℝ), x ≠ 0 → 27 * f (-x) / x - x^2 * f (1 / x) = - 2 * x^2) →
  f 3 = 2 :=
sorry

end find_f_3_l299_299944


namespace find_number_l299_299217

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 13) : x = 6.5 :=
by
  sorry

end find_number_l299_299217


namespace calculate_value_l299_299646

theorem calculate_value : 15 * (1 / 3) + 45 * (2 / 3) = 35 := 
by
simp -- We use simp to simplify the expression
sorry -- We put sorry as we are skipping the full proof

end calculate_value_l299_299646


namespace ellipse_area_calc_l299_299711

noncomputable def ellipse_area (a b : ℝ) : ℝ :=
  real.pi * a * b

theorem ellipse_area_calc :
  let center := (5 : ℝ, 2 : ℝ)
  let semi_major_axis := 10
  let point_on_ellipse := (13 : ℝ, 6 : ℝ)
  let b := 20 / 3
  in ellipse_area semi_major_axis b = (200 * real.pi) / 3 :=
by
  let center := (5 : ℝ, 2 : ℝ)
  let semi_major_axis := 10
  let point_on_ellipse := (13 : ℝ, 6 : ℝ)
  let b := 20 / 3
  have h : ellipse_area semi_major_axis b = (200 * real.pi) / 3, from sorry
  exact h

end ellipse_area_calc_l299_299711


namespace expand_product_l299_299111

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 :=
by
  sorry

end expand_product_l299_299111


namespace box_volume_l299_299359

theorem box_volume
  (l w h : ℝ)
  (A1 : l * w = 36)
  (A2 : w * h = 18)
  (A3 : l * h = 8) :
  l * w * h = 102 := 
sorry

end box_volume_l299_299359


namespace largest_angle_in_ratio_triangle_l299_299190

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l299_299190


namespace problem_equivalent_l299_299253

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem problem_equivalent (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by
  sorry

end problem_equivalent_l299_299253


namespace even_function_a_eq_neg1_l299_299577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * (Real.exp x + a * Real.exp (-x))

/-- Given that the function f(x) = x(e^x + a e^{-x}) is an even function, prove that a = -1. -/
theorem even_function_a_eq_neg1 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
sorry

end even_function_a_eq_neg1_l299_299577


namespace smallest_n_for_divisibility_l299_299727

theorem smallest_n_for_divisibility (a₁ a₂ : ℕ) (n : ℕ) (h₁ : a₁ = 5 / 8) (h₂ : a₂ = 25) :
  (∃ n : ℕ, n ≥ 1 ∧ (a₁ * (40 ^ (n - 1)) % 2000000 = 0)) → (n = 7) :=
by
  sorry

end smallest_n_for_divisibility_l299_299727


namespace second_ball_red_probability_l299_299511

-- Definitions based on given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def first_ball_is_red : Prop := true

-- The probability that the second ball drawn is red given the first ball drawn is red
def prob_second_red_given_first_red : ℚ :=
  (red_balls - 1) / (total_balls - 1)

theorem second_ball_red_probability :
  first_ball_is_red → prob_second_red_given_first_red = 5 / 9 :=
by
  intro _
  -- proof goes here
  sorry

end second_ball_red_probability_l299_299511


namespace regular_pentagon_l299_299726

-- Definition for a Convex Pentagon with equal sides
structure ConvexPentagon (A B C D E : Type) :=
(equal_sides : ∀ (a b : Type), a ≠ b → a.length = b.length)
(convex : ∀ (a A' : Type), (a + A') = 180)
(order_of_angles : (∀ {a b c d e : Type}, a.angle ≥ b.angle) ∧ (b.angle ≥ c.angle) ∧ (c.angle ≥ d.angle) ∧ (d.angle ≥ e.angle))

theorem regular_pentagon (ABCDE : ConvexPentagon) : 
  ∃ (A B C D E : Type), A.angle = B.angle ∧ B.angle = C.angle ∧ C.angle = D.angle ∧ D.angle = E.angle :=
sorry

end regular_pentagon_l299_299726


namespace power_inequality_l299_299173

theorem power_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : abs x < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end power_inequality_l299_299173


namespace sixth_term_sequence_l299_299872

theorem sixth_term_sequence (a : ℕ → ℕ) (h₁ : a 0 = 3) (h₂ : ∀ n, a (n + 1) = (a n)^2) : 
  a 5 = 1853020188851841 := 
by {
  sorry
}

end sixth_term_sequence_l299_299872


namespace total_selling_price_l299_299633

theorem total_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (cost3 : ℝ) 
  (profit_percent1 : ℝ) (profit_percent2 : ℝ) (profit_percent3 : ℝ) :
  cost1 = 600 → cost2 = 450 → cost3 = 750 →
  profit_percent1 = 0.08 → profit_percent2 = 0.10 → profit_percent3 = 0.15 →
  (cost1 * (1 + profit_percent1) + cost2 * (1 + profit_percent2) + cost3 * (1 + profit_percent3)) = 2005.50 :=
by
  intros h1 h2 h3 p1 p2 p3
  simp [h1, h2, h3, p1, p2, p3]
  sorry

end total_selling_price_l299_299633


namespace gcd_360_504_l299_299183

theorem gcd_360_504 : Int.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l299_299183


namespace cone_ratio_approx_l299_299685

noncomputable def cone_min_surface_area_ratio (V : ℝ) : ℝ :=
  let r := Real.cbrt (3 * V / Real.pi) in
  let h := 3 * Real.cbrt (V / Real.pi) in
  h / r

theorem cone_ratio_approx (V : ℝ) : 
  cone_min_surface_area_ratio V = 2.08 :=
sorry

end cone_ratio_approx_l299_299685


namespace evaluate_expression_at_x_eq_2_l299_299976

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l299_299976


namespace pool_capacity_l299_299090

theorem pool_capacity:
  (∃ (V1 V2 : ℝ) (t : ℝ), 
    (V1 = t / 120) ∧ 
    (V2 = V1 + 50) ∧ 
    (V1 + V2 = t / 48) ∧ 
    t = 12000) := 
by 
  sorry

end pool_capacity_l299_299090


namespace fibonacci_problem_l299_299957

theorem fibonacci_problem 
  (F : ℕ → ℕ)
  (h1 : F 1 = 1)
  (h2 : F 2 = 1)
  (h3 : ∀ n ≥ 3, F n = F (n - 1) + F (n - 2))
  (a b c : ℕ)
  (h4 : F c = 2 * F b - F a)
  (h5 : F c - F a = F a)
  (h6 : a + c = 1700) :
  a = 849 := 
sorry

end fibonacci_problem_l299_299957


namespace fraction_division_l299_299079

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l299_299079


namespace number_of_diagonal_intersections_of_convex_n_gon_l299_299534

theorem number_of_diagonal_intersections_of_convex_n_gon (n : ℕ) (h : 4 ≤ n) :
  (∀ P : Π m, m = n ↔ m ≥ 4, ∃ i : ℕ, i = n * (n - 1) * (n - 2) * (n - 3) / 24) := 
by
  sorry

end number_of_diagonal_intersections_of_convex_n_gon_l299_299534


namespace total_words_read_l299_299140

/-- Proof Problem Statement:
  Given the following conditions:
  - Henri has 8 hours to watch movies and read.
  - He watches one movie for 3.5 hours.
  - He watches another movie for 1.5 hours.
  - He watches two more movies with durations of 1.25 hours and 0.75 hours, respectively.
  - He reads for the remaining time after watching movies.
  - For the first 30 minutes of reading, he reads at a speed of 12 words per minute.
  - For the following 20 minutes, his reading speed decreases to 8 words per minute.
  - In the last remaining minutes, his reading speed increases to 15 words per minute.
  Prove that the total number of words Henri reads during his free time is 670.
--/
theorem total_words_read : 8 * 60 - (7 * 60) = 60 ∧
  (30 * 12) + (20 * 8) + ((60 - 30 - 20) * 15) = 670 :=
by
  sorry

end total_words_read_l299_299140


namespace rita_bought_4_pounds_l299_299950

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l299_299950


namespace find_n_l299_299703

theorem find_n (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  unfold pow at h
  sorry

end find_n_l299_299703


namespace arrangements_with_AB_together_l299_299406

theorem arrangements_with_AB_together (students : Finset α) (A B : α) (hA : A ∈ students) (hB : B ∈ students) (h_students : students.card = 5) : 
  ∃ n, n = 48 :=
sorry

end arrangements_with_AB_together_l299_299406


namespace lcm_18_24_l299_299796

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299796


namespace total_money_l299_299518

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l299_299518


namespace amount_b_l299_299823

-- Definitions of the conditions
variables (a b : ℚ) 

def condition1 : Prop := a + b = 1210
def condition2 : Prop := (2 / 3) * a = (1 / 2) * b

-- The theorem to prove
theorem amount_b (h₁ : condition1 a b) (h₂ : condition2 a b) : b = 691.43 :=
sorry

end amount_b_l299_299823


namespace sum_of_three_pentagons_l299_299386

variable (x y : ℚ)

axiom eq1 : 3 * x + 2 * y = 27
axiom eq2 : 2 * x + 3 * y = 25

theorem sum_of_three_pentagons : 3 * y = 63 / 5 := 
by {
  sorry -- No need to provide proof steps
}

end sum_of_three_pentagons_l299_299386


namespace find_b_minus_c_l299_299408

noncomputable def a (n : ℕ) : ℝ :=
  if h : n > 1 then 1 / Real.log 1009 * Real.log n else 0

noncomputable def b : ℝ :=
  a 2 + a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = -Real.logb 1009 1938 := by
  sorry

end find_b_minus_c_l299_299408


namespace triangle_perimeter_l299_299432

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 3) (h2 : b = 3) 
    (h3 : c^2 = a * Real.cos B + b * Real.cos A) : 
    a + b + c = 7 :=
by 
  sorry

end triangle_perimeter_l299_299432


namespace total_games_played_l299_299502

-- Definition of the conditions
def teams : Nat := 10
def games_per_pair : Nat := 4

-- Statement of the problem
theorem total_games_played (teams games_per_pair : Nat) : 
  teams = 10 → 
  games_per_pair = 4 → 
  ∃ total_games, total_games = 180 :=
by
  intro h1 h2
  sorry

end total_games_played_l299_299502


namespace not_enough_funds_to_buy_two_books_l299_299973

def storybook_cost : ℝ := 25.5
def sufficient_funds (amount : ℝ) : Prop := amount >= 50

theorem not_enough_funds_to_buy_two_books : ¬ sufficient_funds (2 * storybook_cost) :=
by
  sorry

end not_enough_funds_to_buy_two_books_l299_299973


namespace fraction_division_l299_299346

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299346


namespace max_value_fraction_l299_299892

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∀ z, z = (x / (2 * x + y) + y / (x + 2 * y)) → z ≤ (2 / 3) :=
by
  sorry

end max_value_fraction_l299_299892


namespace value_of_expression_l299_299546

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2005 = -2004 :=
by
  sorry

end value_of_expression_l299_299546


namespace total_votes_l299_299429

theorem total_votes (V : ℝ) (win_percentage : ℝ) (majority : ℝ) (lose_percentage : ℝ)
  (h1 : win_percentage = 0.75) (h2 : lose_percentage = 0.25) (h3 : majority = 420) :
  V = 840 :=
by
  sorry

end total_votes_l299_299429


namespace sum_of_p_for_circumcenter_on_Ox_l299_299466

/--
Given the quadratic equation: y = 2^p * x^2 + 5 * p * x - 2^(p^2)
and the triangle ABC formed by the intersections with the axes,
find the sum of all values of the parameter p for which the center of the
circle circumscribing the triangle ABC lies on the Ox axis.
-/
theorem sum_of_p_for_circumcenter_on_Ox {p : ℝ} :
  ∑ p ∈ {p | ∃ (x1 x2 : ℝ), 
        (2^p * x1^2 + 5*p*x1 - 2^(p^2) = 0) ∧ 
        (2^p * x2^2 + 5*p*x2 - 2^(p^2) = 0) ∧ 
        ∃ (C : ℝ × ℝ), 
          C = (0, -2^(p^2)) ∧ 
          ((-2^(p^2) / x1) * (-2^(p^2) / x2) = -1)},
    p = -1 :=
begin
  sorry
end

end sum_of_p_for_circumcenter_on_Ox_l299_299466


namespace lcm_18_24_l299_299812
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299812


namespace count_scalene_triangles_natural_sides_l299_299847

theorem count_scalene_triangles_natural_sides :
  let scalene_triangles := { S : ℕ × ℕ × ℕ | 
    let a := S.1, 
        b := S.2.1, 
        c := S.2.2 in 
    a < b ∧ b < c ∧ 
    a + c = 2 * b ∧ 
    a + b + c ≤ 30 ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 } in
  scalene_triangles.finite.count = 20 :=
by 
  sorry

end count_scalene_triangles_natural_sides_l299_299847


namespace parabola_conditions_l299_299061

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l299_299061


namespace total_number_of_coins_l299_299831

theorem total_number_of_coins (num_5c : Nat) (num_10c : Nat) (h1 : num_5c = 16) (h2 : num_10c = 16) : num_5c + num_10c = 32 := by
  sorry

end total_number_of_coins_l299_299831


namespace greatest_ribbon_length_l299_299640

-- Define lengths of ribbons
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

-- Condition ensures gcd and prime check
def gcd_is_prime (n : ℕ) : Prop :=
  ∃ d : ℕ, (∀ l ∈ ribbon_lengths, d ∣ l) ∧ Prime d ∧ n = d

-- Prove the greatest length that can make the ribbon pieces, with no ribbon left over, is 2
theorem greatest_ribbon_length : ∃ d, gcd_is_prime d ∧ ∀ m, gcd_is_prime m → m ≤ 2 := 
sorry

end greatest_ribbon_length_l299_299640


namespace largest_angle_in_ratio_3_4_5_l299_299186

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l299_299186


namespace lcm_18_24_l299_299807

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299807


namespace cube_distance_l299_299094

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end cube_distance_l299_299094


namespace net_wealth_after_transactions_l299_299306

-- Define initial values and transactions
def initial_cash_A : ℕ := 15000
def initial_cash_B : ℕ := 20000
def initial_house_value : ℕ := 15000
def first_transaction_price : ℕ := 20000
def depreciation_rate : ℝ := 0.15

-- Post-depreciation house value
def depreciated_house_value : ℝ := initial_house_value * (1 - depreciation_rate)

-- Final amounts after transactions
def final_cash_A : ℝ := (initial_cash_A + first_transaction_price) - depreciated_house_value
def final_cash_B : ℝ := depreciated_house_value

-- Net changes in wealth
def net_change_wealth_A : ℝ := final_cash_A + depreciated_house_value - (initial_cash_A + initial_house_value)
def net_change_wealth_B : ℝ := final_cash_B - initial_cash_B

-- Our proof goal
theorem net_wealth_after_transactions :
  net_change_wealth_A = 5000 ∧ net_change_wealth_B = -7250 :=
by
  sorry

end net_wealth_after_transactions_l299_299306


namespace jeffrey_steps_l299_299487

theorem jeffrey_steps (distance : ℕ) (forward_steps : ℕ) (backward_steps : ℕ) (effective_distance : ℤ) 
  (total_actual_steps : ℤ) (h1 : forward_steps = 3) (h2 : backward_steps = 2)
  (h3 : effective_distance = 1) 
  (h4 : total_distance : effective_distance * distance) :
  total_actual_steps = 330 :=
  by sorry

end jeffrey_steps_l299_299487


namespace probability_at_least_one_solves_l299_299139

theorem probability_at_least_one_solves :
  ∀ (P : Type → Prop) [ProbabilityTheory P],
  let A := 0.6
  let B := 0.7 in
  independent_events A B →
  1 - ((1 - A) * (1 - B)) = 0.88 :=
by
  intros P _ A B independence
  sorry

end probability_at_least_one_solves_l299_299139


namespace mona_cookie_count_l299_299736

theorem mona_cookie_count {M : ℕ} (h1 : (M - 5) + (M - 5 + 10) + M = 60) : M = 20 :=
by
  sorry

end mona_cookie_count_l299_299736


namespace find_a_l299_299698

theorem find_a (a : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 + a * i) * i = -3 + i) : a = 3 :=
by
  sorry

end find_a_l299_299698


namespace smallest_number_divisible_l299_299215

theorem smallest_number_divisible (n : ℕ) :
  (n + 2) % 12 = 0 ∧ 
  (n + 2) % 30 = 0 ∧ 
  (n + 2) % 48 = 0 ∧ 
  (n + 2) % 74 = 0 ∧ 
  (n + 2) % 100 = 0 ↔ 
  n = 44398 :=
by sorry

end smallest_number_divisible_l299_299215


namespace residue_of_neg_1237_mod_37_l299_299670

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l299_299670


namespace simplify_log_expression_l299_299313

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l299_299313


namespace arrangement_count_is_12_l299_299105

-- The problem: Arranging the letters a, a, b, b, c, c in a 3x2 grid with no repeats in any row or column
def letters : List Char := ['a', 'a', 'b', 'b', 'c', 'c']

-- Definition of a valid 3x2 arrangement
def valid_arrangement (grid : List (List Char)) : Prop :=
  grid.length = 3 ∧
  ∀ row, row ∈ grid → row.length = 2 ∧ row.nodup ∧ -- Rows must have 2 different elements
  ∀ i, i < 2 → (list_erase_nth grid i).nodup      -- Columns must have different elements

-- Define the total number of valid arrangements
def count_valid_arrangements : Nat :=
  (List.permutations letters).count (λ p, valid_arrangement (List.chunk 2 p))

theorem arrangement_count_is_12 : count_valid_arrangements = 12 :=
by
  sorry

end arrangement_count_is_12_l299_299105


namespace price_decrease_percentage_l299_299425

theorem price_decrease_percentage (P₀ P₁ P₂ : ℝ) (x : ℝ) :
  P₀ = 1 → P₁ = P₀ * 1.25 → P₂ = P₁ * (1 - x / 100) → P₂ = 1 → x = 20 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end price_decrease_percentage_l299_299425


namespace complete_square_example_l299_299818

theorem complete_square_example :
  ∃ c : ℝ, ∃ d : ℝ, (∀ x : ℝ, x^2 + 12 * x + 4 = (x + c)^2 - d) ∧ d = 32 := by
  sorry

end complete_square_example_l299_299818


namespace kindergarten_library_models_l299_299958

theorem kindergarten_library_models
  (paid : ℕ)
  (reduced_price : ℕ)
  (models_total_gt_5 : ℕ)
  (bought : ℕ) 
  (condition : paid = 570 ∧ reduced_price = 95 ∧ models_total_gt_5 > 5 ∧ bought = 3 * (2 : ℕ)) :
  exists x : ℕ, bought / 3 = x ∧ x = 2 :=
by
  sorry

end kindergarten_library_models_l299_299958


namespace count_valid_triangles_l299_299848

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end count_valid_triangles_l299_299848


namespace negation_universal_proposition_l299_299194

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by sorry

end negation_universal_proposition_l299_299194


namespace car_speed_first_hour_l299_299606

theorem car_speed_first_hour (speed1 speed2 avg_speed : ℕ) (h1 : speed2 = 70) (h2 : avg_speed = 95) :
  (2 * avg_speed) = speed1 + speed2 → speed1 = 120 :=
by
  sorry

end car_speed_first_hour_l299_299606


namespace slope_of_line_inclination_l299_299156

theorem slope_of_line_inclination (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 180) 
  (h3 : Real.tan (α * Real.pi / 180) = Real.sqrt 3 / 3) : α = 30 :=
by
  sorry

end slope_of_line_inclination_l299_299156


namespace expression_equals_two_l299_299316

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l299_299316


namespace line_eq_x_1_parallel_y_axis_l299_299182

theorem line_eq_x_1_parallel_y_axis (P : ℝ × ℝ) (hP : P = (1, 0)) (h_parallel : ∀ y : ℝ, (1, y) = P ∨ P = (1, y)) :
  ∃ x : ℝ, (∀ y : ℝ, P = (x, y)) → x = 1 := 
by 
  sorry

end line_eq_x_1_parallel_y_axis_l299_299182


namespace regular_hexagon_perimeter_is_30_l299_299583

-- Define a regular hexagon with each side length 5 cm
def regular_hexagon_side_length : ℝ := 5

-- Define the perimeter of a regular hexagon
def regular_hexagon_perimeter (side_length : ℝ) : ℝ := 6 * side_length

-- State the theorem about the perimeter of a regular hexagon with side length 5 cm
theorem regular_hexagon_perimeter_is_30 : regular_hexagon_perimeter regular_hexagon_side_length = 30 := 
by 
  sorry

end regular_hexagon_perimeter_is_30_l299_299583


namespace set_equality_l299_299693

def P : Set ℝ := { x | x^2 = 1 }

theorem set_equality : P = {-1, 1} :=
by
  sorry

end set_equality_l299_299693


namespace lcm_18_24_l299_299776

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299776


namespace largest_rectangle_area_l299_299229

theorem largest_rectangle_area (l w : ℕ) (hl : l > 0) (hw : w > 0) (hperimeter : 2 * l + 2 * w = 42)
  (harea_diff : ∃ (l1 w1 l2 w2 : ℕ), l1 > 0 ∧ w1 > 0 ∧ l2 > 0 ∧ w2 > 0 ∧ 2 * l1 + 2 * w1 = 42 
  ∧ 2 * l2 + 2 * w2 = 42 ∧ (l1 * w1) - (l2 * w2) = 90) : (l * w ≤ 110) :=
sorry

end largest_rectangle_area_l299_299229


namespace fraction_equivalence_l299_299875

theorem fraction_equivalence :
  ( (3 / 7 + 2 / 3) / (5 / 11 + 3 / 8) ) = (119 / 90) :=
by
  sorry

end fraction_equivalence_l299_299875


namespace minimum_value_l299_299125

/-- 
Given \(a > 0\), \(b > 0\), and \(a + 2b = 1\),
prove that the minimum value of \(\frac{2}{a} + \frac{1}{b}\) is 8.
-/
theorem minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) : 
  (∀ a b : ℝ, (a > 0) → (b > 0) → (a + 2 * b = 1) → (∃ c : ℝ, c = 8 ∧ ∀ x y : ℝ, (x = a) → (y = b) → (c ≤ (2 / x) + (1 / y)))) :=
sorry

end minimum_value_l299_299125


namespace sum_of_geometric_progression_l299_299131

theorem sum_of_geometric_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (a1 a3 : ℝ) (h1 : a1 + a3 = 5) (h2 : a1 * a3 = 4)
  (h3 : a 1 = a1) (h4 : a 3 = a3)
  (h5 : ∀ k, a (k + 1) > a k)  -- Sequence is increasing
  (h6 : S n = a 1 * ((1 - (2:ℝ) ^ n) / (1 - 2)))
  (h7 : n = 6) :
  S 6 = 63 :=
sorry

end sum_of_geometric_progression_l299_299131


namespace max_possible_x_plus_y_l299_299009

theorem max_possible_x_plus_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x * y - (x + y) = Nat.gcd x y + Nat.lcm x y) :
  x + y ≤ 10 := sorry

end max_possible_x_plus_y_l299_299009


namespace roger_total_miles_l299_299048

def morning_miles : ℕ := 2
def evening_multiplicative_factor : ℕ := 5
def evening_miles := evening_multiplicative_factor * morning_miles
def third_session_subtract : ℕ := 1
def third_session_miles := (2 * morning_miles) - third_session_subtract
def total_miles := morning_miles + evening_miles + third_session_miles

theorem roger_total_miles : total_miles = 15 := by
  sorry

end roger_total_miles_l299_299048


namespace Morgan_first_SAT_score_l299_299171

variable (S : ℝ) -- Morgan's first SAT score
variable (improved_score : ℝ := 1100) -- Improved score on second attempt
variable (improvement_rate : ℝ := 0.10) -- Improvement rate

theorem Morgan_first_SAT_score:
  improved_score = S * (1 + improvement_rate) → S = 1000 := 
by 
  sorry

end Morgan_first_SAT_score_l299_299171


namespace value_of_a_l299_299916
noncomputable def find_a (a b c : ℝ) : ℝ :=
if 2 * b = a + c ∧ (a * c) * (b * c) = ((a * b) ^ 2) ∧ a + b + c = 6 then a else 0

theorem value_of_a (a b c : ℝ) :
  (2 * b = a + c) ∧ ((a * c) * (b * c) = (a * b) ^ 2) ∧ (a + b + c = 6) ∧ (a ≠ c) ∧ (a ≠ b) ∧ (b ≠ c) → a = 4 :=
by sorry

end value_of_a_l299_299916


namespace bianca_birthday_money_l299_299120

/-- Define the number of friends Bianca has -/
def number_of_friends : ℕ := 5

/-- Define the amount of dollars each friend gave -/
def dollars_per_friend : ℕ := 6

/-- The total amount of dollars Bianca received -/
def total_dollars_received : ℕ := number_of_friends * dollars_per_friend

/-- Prove that the total amount of dollars Bianca received is 30 -/
theorem bianca_birthday_money : total_dollars_received = 30 :=
by
  sorry

end bianca_birthday_money_l299_299120


namespace parity_of_f_l299_299301

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ :=
  x * (x - 2) * (x - 1) * x * (x + 1) * (x + 2)

theorem parity_of_f :
  is_even_function f ∧ ¬ (∃ g : ℝ → ℝ, g = f ∧ (∀ x : ℝ, g (-x) = -g x)) :=
by
  sorry

end parity_of_f_l299_299301


namespace xy_sum_l299_299948

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 20) : x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by
  sorry

end xy_sum_l299_299948


namespace farmer_total_cows_l299_299221

theorem farmer_total_cows (cows : ℕ) 
  (h1 : 1 / 3 + 1 / 6 + 1 / 8 = 5 / 8) 
  (h2 : (3 / 8) * cows = 15) : 
  cows = 40 := by
  -- Given conditions:
  -- h1: The first three sons receive a total of 5/8 of the cows.
  -- h2: The fourth son receives 3/8 of the cows, which is 15 cows.
  sorry

end farmer_total_cows_l299_299221


namespace vertex_y_coordinate_l299_299110

theorem vertex_y_coordinate (x : ℝ) : 
    let a := -6
    let b := 24
    let c := -7
    ∃ k : ℝ, k = 17 ∧ ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x - 2)^2 + k) := 
by 
  sorry

end vertex_y_coordinate_l299_299110


namespace smallest_perimeter_of_triangle_with_consecutive_odd_integers_l299_299817

theorem smallest_perimeter_of_triangle_with_consecutive_odd_integers :
  ∃ (a b c : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
  (a < b) ∧ (b < c) ∧ (c = a + 4) ∧
  (a + b > c) ∧ (b + c > a) ∧ (a + c > b) ∧ 
  (a + b + c = 15) :=
by
  sorry

end smallest_perimeter_of_triangle_with_consecutive_odd_integers_l299_299817


namespace option_A_option_B_option_D_l299_299890

-- Given real numbers a, b, c such that a > b > 1 and c > 0,
-- prove the following inequalities.
variables {a b c : ℝ}

-- Assume the conditions
axiom H1 : a > b
axiom H2 : b > 1
axiom H3 : c > 0

-- Statements to prove
theorem option_A (H1: a > b) (H2: b > 1) (H3: c > 0) : a^2 - bc > b^2 - ac := sorry
theorem option_B (H1: a > b) (H2: b > 1) : a^3 > b^2 := sorry
theorem option_D (H1: a > b) (H2: b > 1) : a + (1/a) > b + (1/b) := sorry
  
end option_A_option_B_option_D_l299_299890


namespace probability_of_ge_four_is_one_eighth_l299_299206

noncomputable def probability_ge_four : ℝ :=
sorry

theorem probability_of_ge_four_is_one_eighth :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) →
  (probability_ge_four = 1 / 8) :=
sorry

end probability_of_ge_four_is_one_eighth_l299_299206


namespace find_sports_package_channels_l299_299719

-- Defining the conditions
def initial_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def reduce_package_by : ℕ := 10
def supreme_sports_package : ℕ := 7
def final_channels : ℕ := 147

-- Defining the situation before the final step
def channels_after_reduction := initial_channels - channels_taken_away + channels_replaced - reduce_package_by
def channels_after_supreme := channels_after_reduction + supreme_sports_package

-- Prove the original sports package added 8 channels
theorem find_sports_package_channels : ∀ sports_package_added : ℕ,
  sports_package_added + channels_after_supreme = final_channels → sports_package_added = 8 :=
by
  intro sports_package_added
  intro h
  sorry

end find_sports_package_channels_l299_299719


namespace maximum_area_rectangle_l299_299755

-- Define the conditions
def length (x : ℝ) := x
def width (x : ℝ) := 2 * x
def perimeter (x : ℝ) := 2 * (length x + width x)

-- The proof statement
theorem maximum_area_rectangle (h : perimeter x = 40) : 2 * (length x) * (width x) = 800 / 9 :=
by
  sorry

end maximum_area_rectangle_l299_299755


namespace abs_ac_bd_leq_one_l299_299528

theorem abs_ac_bd_leq_one {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : |a * c + b * d| ≤ 1 :=
by
  sorry

end abs_ac_bd_leq_one_l299_299528


namespace area_hexagon_STUVWX_l299_299021

noncomputable def area_of_hexagon (area_PQR : ℕ) (small_area : ℕ) : ℕ := 
  area_PQR - (3 * small_area)

theorem area_hexagon_STUVWX : 
  let area_PQR := 45
  let small_area := 1 
  ∃ area_hexagon, area_hexagon = 42 := 
by
  let area_PQR := 45
  let small_area := 1
  let area_hexagon := area_of_hexagon area_PQR small_area
  use area_hexagon
  sorry

end area_hexagon_STUVWX_l299_299021


namespace fewest_tiles_to_cover_region_l299_299996

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l299_299996


namespace rhombus_area_l299_299404

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : 
  (d1 * d2) / 2 = 160 := by
sorry

end rhombus_area_l299_299404


namespace value_of_x_l299_299470

theorem value_of_x (w : ℝ) (hw : w = 90) (z : ℝ) (hz : z = 2 / 3 * w) (y : ℝ) (hy : y = 1 / 4 * z) (x : ℝ) (hx : x = 1 / 2 * y) : x = 7.5 :=
by
  -- Proof skipped; conclusion derived from conditions
  sorry

end value_of_x_l299_299470


namespace find_m_of_quad_roots_l299_299593

theorem find_m_of_quad_roots
  (a b : ℝ) (m : ℝ)
  (ha : a = 5)
  (hb : b = -4)
  (h_roots : ∀ x : ℂ, (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ x = (2 - Complex.I * Real.sqrt 143) / 5) →
                     (a * x^2 + b * x + m = 0)) :
  m = 7.95 :=
by
  -- Proof goes here
  sorry

end find_m_of_quad_roots_l299_299593


namespace simplify_expr_l299_299561

-- Define the condition
def y : ℕ := 77

-- Define the expression and the expected result
def expr := (7 * y + 77) / 77

-- The theorem statement
theorem simplify_expr : expr = 8 :=
by
  sorry

end simplify_expr_l299_299561


namespace minute_hand_distance_traveled_l299_299601

noncomputable def radius : ℝ := 8
noncomputable def minutes_in_one_revolution : ℝ := 60
noncomputable def total_minutes : ℝ := 45

theorem minute_hand_distance_traveled :
  (total_minutes / minutes_in_one_revolution) * (2 * Real.pi * radius) = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_traveled_l299_299601


namespace exponent_problem_l299_299412

variable {a m n : ℝ}

theorem exponent_problem (h1 : a^m = 2) (h2 : a^n = 3) : a^(3*m + 2*n) = 72 := 
  sorry

end exponent_problem_l299_299412


namespace integer_for_finitely_many_n_l299_299456

theorem integer_for_finitely_many_n (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, N < n → ¬ ∃ k : ℤ, (a + 1 / 2) ^ n + (b + 1 / 2) ^ n = k := 
sorry

end integer_for_finitely_many_n_l299_299456


namespace cistern_wet_surface_area_l299_299219

def cistern_length : ℝ := 4
def cistern_width : ℝ := 8
def water_depth : ℝ := 1.25

def area_bottom (l w : ℝ) : ℝ := l * w
def area_pair1 (l h : ℝ) : ℝ := 2 * (l * h)
def area_pair2 (w h : ℝ) : ℝ := 2 * (w * h)
def total_wet_surface_area (l w h : ℝ) : ℝ := area_bottom l w + area_pair1 l h + area_pair2 w h

theorem cistern_wet_surface_area : total_wet_surface_area cistern_length cistern_width water_depth = 62 := 
by 
  sorry

end cistern_wet_surface_area_l299_299219


namespace correct_conclusions_l299_299060

variable (a b c m : ℝ)
variable (y1 y2 : ℝ)

-- Conditions: 
-- Parabola y = ax^2 + bx + c, intersects x-axis at (-3,0) and (1,0)
-- a < 0
-- Points P(m-2, y1) and Q(m, y2) are on the parabola, y1 < y2

def parabola_intersects_x_axis_at_A_B : Prop :=
  ∀ x : ℝ, x = -3 ∨ x = 1 → a * x^2 + b * x + c = 0

def concavity_and_roots : Prop :=
  a < 0 ∧ b = 2 * a ∧ c = -3 * a

def conclusion_1 : Prop :=
  a * b * c < 0

def conclusion_2 : Prop :=
  b^2 - 4 * a * c > 0

def conclusion_3 : Prop :=
  3 * b + 2 * c = 0

def conclusion_4 : Prop :=
  y1 < y2 → m ≤ -1

-- Correct conclusions given the parabola properties
theorem correct_conclusions :
  concavity_and_roots a b c →
  parabola_intersects_x_axis_at_A_B a b c →
  conclusion_1 a b c ∨ conclusion_2 a b c ∨ conclusion_3 a b c ∨ conclusion_4 a b c :=
sorry

end correct_conclusions_l299_299060


namespace inequality_solution_l299_299198

theorem inequality_solution (x : ℝ) : 
  (x-20) / (x+16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 := by
  sorry

end inequality_solution_l299_299198


namespace division_of_fractions_l299_299340

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299340


namespace approximate_number_of_fish_in_pond_l299_299921

theorem approximate_number_of_fish_in_pond :
  (∃ N : ℕ, 
  (∃ tagged1 tagged2 : ℕ, tagged1 = 50 ∧ tagged2 = 10) ∧
  (∃ caught1 caught2 : ℕ, caught1 = 50 ∧ caught2 = 50) ∧
  ((tagged2 : ℝ) / caught2 = (tagged1 : ℝ) / (N : ℝ)) ∧
  N = 250) :=
sorry

end approximate_number_of_fish_in_pond_l299_299921


namespace equilateral_triangle_perimeter_l299_299893

theorem equilateral_triangle_perimeter (p_ADC : ℝ) (h_ratio : ∀ s1 s2 : ℝ, s1 / s2 = 1 / 2) :
  p_ADC = 9 + 3 * Real.sqrt 3 → (3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3) :=
by
  intro h
  have h1 : 3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3 := sorry
  exact h1

end equilateral_triangle_perimeter_l299_299893


namespace triangle_angle_D_l299_299023

theorem triangle_angle_D (F E D : ℝ) (hF : F = 15) (hE : E = 3 * F) (h_triangle : D + E + F = 180) : D = 120 := by
  sorry

end triangle_angle_D_l299_299023


namespace lcm_18_24_eq_72_l299_299788

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299788


namespace correct_choice_l299_299641

def PropA : Prop := ∀ x : ℝ, x^2 + 3 < 0
def PropB : Prop := ∀ x : ℕ, x^2 ≥ 1
def PropC : Prop := ∃ x : ℤ, x^5 < 1
def PropD : Prop := ∃ x : ℚ, x^2 = 3

theorem correct_choice : ¬PropA ∧ ¬PropB ∧ PropC ∧ ¬PropD := by
  sorry

end correct_choice_l299_299641


namespace total_money_l299_299519

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l299_299519


namespace gcd_factorials_l299_299532

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := by
  sorry

end gcd_factorials_l299_299532


namespace expression_expansion_l299_299244

noncomputable def expand_expression : Polynomial ℤ :=
 -2 * (5 * Polynomial.X^3 - 7 * Polynomial.X^2 + Polynomial.X - 4)

theorem expression_expansion :
  expand_expression = -10 * Polynomial.X^3 + 14 * Polynomial.X^2 - 2 * Polynomial.X + 8 :=
by
  sorry

end expression_expansion_l299_299244


namespace lcm_18_24_l299_299801

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299801


namespace increase_by_percentage_l299_299480

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l299_299480


namespace residue_of_neg_1237_mod_37_l299_299669

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l299_299669


namespace fraction_division_l299_299078

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l299_299078


namespace fraction_to_terminating_decimal_l299_299112

theorem fraction_to_terminating_decimal :
  (45 : ℚ) / 64 = (703125 : ℚ) / 1000000 := by
  sorry

end fraction_to_terminating_decimal_l299_299112


namespace arithmetic_sequence_ratio_l299_299899

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Given two arithmetic sequences a_n and b_n, their sums of the first n terms are S_n and T_n respectively.
-- Given that S_n / T_n = (2n + 2) / (n + 3).
-- Prove that a_10 / b_10 = 20 / 11.

theorem arithmetic_sequence_ratio (h : ∀ n, S_n n / T_n n = (2 * n + 2) / (n + 3)) : (a_n 10) / (b_n 10) = 20 / 11 := 
by
  sorry

end arithmetic_sequence_ratio_l299_299899


namespace harmonic_mean_pairs_count_l299_299538

open Nat

theorem harmonic_mean_pairs_count :
  ∃! n : ℕ, (∀ x y : ℕ, x < y ∧ x > 0 ∧ y > 0 ∧ (2 * x * y) / (x + y) = 4^15 → n = 29) :=
sorry

end harmonic_mean_pairs_count_l299_299538


namespace williams_tips_fraction_l299_299089

theorem williams_tips_fraction
  (A : ℝ) -- average tips for months other than August
  (h : ∀ A, A > 0) -- assuming some positivity constraint for non-degenerate mean
  (h_august : A ≠ 0) -- assuming average can’t be zero
  (august_tips : ℝ := 10 * A)
  (other_months_tips : ℝ := 6 * A)
  (total_tips : ℝ := 16 * A) :
  (august_tips / total_tips) = (5 / 8) := 
sorry

end williams_tips_fraction_l299_299089


namespace range_of_x_l299_299260

open Set

noncomputable def M (x : ℝ) : Set ℝ := {x^2, 1}

theorem range_of_x (x : ℝ) (hx : M x) : x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end range_of_x_l299_299260


namespace solve_for_x_l299_299141

theorem solve_for_x (x y : ℝ) (h1 : 2 * x - 3 * y = 18) (h2 : x + 2 * y = 8) : x = 60 / 7 := sorry

end solve_for_x_l299_299141


namespace inequality_true_l299_299143

-- Define the conditions
variables (a b : ℝ) (h : a < b) (hb_neg : b < 0)

-- State the theorem to be proved
theorem inequality_true (ha : a < b) (hb : b < 0) : (|a| / |b| > 1) :=
sorry

end inequality_true_l299_299143


namespace max_value_x_plus_2y_l299_299418

theorem max_value_x_plus_2y (x y : ℝ) (h : |x| + |y| ≤ 1) : x + 2 * y ≤ 2 :=
sorry

end max_value_x_plus_2y_l299_299418


namespace inequality_proof_l299_299827

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
    (h_sum : a + b + c + d = 8) :
    (a^3 / (a^2 + b + c) + b^3 / (b^2 + c + d) + c^3 / (c^2 + d + a) + d^3 / (d^2 + a + b)) ≥ 4 :=
by
  sorry

end inequality_proof_l299_299827


namespace expand_product_l299_299529

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10 * y + 21 := by
  sorry

end expand_product_l299_299529


namespace bruce_three_times_son_in_six_years_l299_299864

-- Define the current ages of Bruce and his son
def bruce_age : ℕ := 36
def son_age : ℕ := 8

-- Define the statement to be proved
theorem bruce_three_times_son_in_six_years :
  ∃ (x : ℕ), x = 6 ∧ ∀ t, (t = x) → (bruce_age + t = 3 * (son_age + t)) :=
by
  sorry

end bruce_three_times_son_in_six_years_l299_299864


namespace lcm_18_24_l299_299802

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299802


namespace g_recursion_relation_l299_299303

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2)^n

theorem g_recursion_relation (n : ℕ) : g (n + 1) - 2 * g n + g (n - 1) = 0 :=
  sorry

end g_recursion_relation_l299_299303


namespace find_m_l299_299109

noncomputable def s : ℕ → ℚ
| 1        := 2
| (n + 1) := if (n + 1) % 3 = 0 then 1 + s ((n + 1) / 3) else 1 / s n

theorem find_m (m : ℕ) (h : s m = 34 / 81) : m = 82 :=
by
  sorry

end find_m_l299_299109


namespace dormouse_stole_flour_l299_299492

-- Define the suspects
inductive Suspect 
| MarchHare 
| MadHatter 
| Dormouse 

open Suspect 

-- Condition 1: Only one of three suspects stole the flour
def only_one_thief (s : Suspect) : Prop := 
  s = MarchHare ∨ s = MadHatter ∨ s = Dormouse

-- Condition 2: Only the person who stole the flour gave a truthful testimony
def truthful (thief : Suspect) (testimony : Suspect → Prop) : Prop :=
  testimony thief

-- Condition 3: The March Hare testified that the Mad Hatter stole the flour
def marchHare_testimony (s : Suspect) : Prop := 
  s = MadHatter

-- The theorem to prove: Dormouse stole the flour
theorem dormouse_stole_flour : 
  ∃ thief : Suspect, only_one_thief thief ∧ 
    (∀ s : Suspect, (s = thief ↔ truthful s marchHare_testimony) → thief = Dormouse) :=
by
  sorry

end dormouse_stole_flour_l299_299492


namespace increase_by_percentage_l299_299479

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l299_299479


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l299_299556

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l299_299556


namespace number_of_geese_l299_299042

theorem number_of_geese (A x n k : ℝ) 
  (h1 : A = k * x * n)
  (h2 : A = (k + 20) * x * (n - 75))
  (h3 : A = (k - 15) * x * (n + 100)) 
  : n = 300 :=
sorry

end number_of_geese_l299_299042


namespace kaleb_earnings_and_boxes_l299_299570

-- Conditions
def initial_games : ℕ := 76
def games_sold : ℕ := 46
def price_15_dollar : ℕ := 20
def price_10_dollar : ℕ := 15
def price_8_dollar : ℕ := 11
def games_per_box : ℕ := 5

-- Definitions and proof problem
theorem kaleb_earnings_and_boxes (initial_games games_sold price_15_dollar price_10_dollar price_8_dollar games_per_box : ℕ) :
  let earnings := (price_15_dollar * 15) + (price_10_dollar * 10) + (price_8_dollar * 8)
  let remaining_games := initial_games - games_sold
  let boxes_needed := remaining_games / games_per_box
  earnings = 538 ∧ boxes_needed = 6 :=
by
  sorry

end kaleb_earnings_and_boxes_l299_299570


namespace total_number_of_animals_l299_299752

-- Define the problem conditions
def number_of_cats : ℕ := 645
def number_of_dogs : ℕ := 567

-- State the theorem to be proved
theorem total_number_of_animals : number_of_cats + number_of_dogs = 1212 := by
  sorry

end total_number_of_animals_l299_299752


namespace train_cross_duration_l299_299157

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 162
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_duration :
  time_to_cross_pole = 250 / (162 * (1000 / 3600)) :=
by
  -- The detailed proof is omitted as per instructions
  sorry

end train_cross_duration_l299_299157


namespace find_k_l299_299136

noncomputable def f (k : ℤ) (x : ℝ) := (k^2 + k - 1) * x^(k^2 - 3 * k)

-- The conditions in the problem
variables (k : ℤ) (x : ℝ)
axiom sym_y_axis : ∀ (x : ℝ), f k (-x) = f k x
axiom decreasing_on_positive : ∀ x1 x2, 0 < x1 → x1 < x2 → f k x1 > f k x2

-- The proof problem statement
theorem find_k : k = 1 :=
sorry

end find_k_l299_299136


namespace power_of_m_divisible_by_33_l299_299563

theorem power_of_m_divisible_by_33 (m : ℕ) (h : m > 0) (k : ℕ) (h_pow : (m ^ k) % 33 = 0) :
  ∃ n, n > 0 ∧ 11 ∣ m ^ n :=
by
  sorry

end power_of_m_divisible_by_33_l299_299563


namespace lcm_18_24_l299_299777

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299777


namespace volume_region_l299_299118

noncomputable def f (x y z w : ℝ) : ℝ :=
  |x + y + z + w| + |x + y + z - w| + |x + y - z + w| + |x - y + z + w| + |-x + y + z + w|

theorem volume_region (S : Set (ℝ × ℝ × ℝ × ℝ)) :
  S = {p | let (x, y, z, w) := p in f x y z w ≤ 6} →
  let vol := MeasureTheory.volume.measure_univ.to_real in
  vol = (2 : ℝ) / 3 :=
by
  sorry

end volume_region_l299_299118


namespace value_of_f_sum_l299_299413

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h_odd : ∀ x, f (-x) = -f x) : Prop
axiom period_9 (h_period : ∀ x, f (x + 9) = f x) : Prop
axiom f_one (h_f1 : f 1 = 5) : Prop

theorem value_of_f_sum (h_odd : ∀ x, f (-x) = -f x)
                       (h_period : ∀ x, f (x + 9) = f x)
                       (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 :=
sorry

end value_of_f_sum_l299_299413


namespace number_of_games_is_15_l299_299822

-- Definition of the given conditions
def total_points : ℕ := 345
def avg_points_per_game : ℕ := 4 + 10 + 9
def number_of_games (total_points : ℕ) (avg_points_per_game : ℕ) := total_points / avg_points_per_game

-- The theorem stating the proof problem
theorem number_of_games_is_15 : number_of_games total_points avg_points_per_game = 15 :=
by
  -- Skipping the proof as only the statement is required
  sorry

end number_of_games_is_15_l299_299822


namespace reduction_for_same_profit_cannot_reach_460_profit_l299_299554

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l299_299554


namespace problem_statement_l299_299722

noncomputable def f (x : ℝ) := 3 * x ^ 5 + 4 * x ^ 4 - 5 * x ^ 3 + 2 * x ^ 2 + x + 6
noncomputable def d (x : ℝ) := x ^ 3 + 2 * x ^ 2 - x - 3
noncomputable def q (x : ℝ) := 3 * x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) := 19 * x ^ 2 - 11 * x - 57

theorem problem_statement : (f 1 = q 1 * d 1 + r 1) ∧ q 1 + r 1 = -47 := by
  sorry

end problem_statement_l299_299722


namespace peter_needs_5000_for_vacation_l299_299448

variable (currentSavings : ℕ) (monthlySaving : ℕ) (months : ℕ)

-- Conditions
def peterSavings := currentSavings
def monthlySavings := monthlySaving
def savingDuration := months

-- Goal
def vacationFundsRequired (currentSavings monthlySaving months : ℕ) : ℕ :=
  currentSavings + (monthlySaving * months)

theorem peter_needs_5000_for_vacation
  (h1 : currentSavings = 2900)
  (h2 : monthlySaving = 700)
  (h3 : months = 3) :
  vacationFundsRequired currentSavings monthlySaving months = 5000 := by
  sorry

end peter_needs_5000_for_vacation_l299_299448


namespace fraction_of_income_from_tips_l299_299382

variable (S T I : ℝ)

/- Definition of the conditions -/
def tips_condition : Prop := T = (3 / 4) * S
def income_condition : Prop := I = S + T

/- The proof problem statement, asserting the desired result -/
theorem fraction_of_income_from_tips (h1 : tips_condition S T) (h2 : income_condition S T I) : T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l299_299382


namespace typing_time_l299_299855

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l299_299855


namespace total_steps_needed_l299_299488

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end total_steps_needed_l299_299488


namespace find_triples_l299_299530

theorem find_triples (x y z : ℕ) :
  (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) ↔ (x = 1 ∧ y = 2 ∧ z = 1) :=
sorry

end find_triples_l299_299530


namespace lcm_18_24_eq_72_l299_299770

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299770


namespace max_value_of_function_l299_299613

open Real 

theorem max_value_of_function : ∀ x : ℝ, 
  cos (2 * x) + 6 * cos (π / 2 - x) ≤ 5 ∧ 
  ∃ x' : ℝ, cos (2 * x') + 6 * cos (π / 2 - x') = 5 :=
by 
  sorry

end max_value_of_function_l299_299613


namespace lcm_18_24_eq_72_l299_299768

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299768


namespace lcm_18_24_l299_299809
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299809


namespace basketball_three_point_shots_l299_299065

theorem basketball_three_point_shots (t h f : ℕ) 
  (h1 : 2 * t = 6 * h)
  (h2 : f = h - 4)
  (h3: 2 * t + 3 * h + f = 76)
  (h4: t + h + f = 40) : h = 8 :=
sorry

end basketball_three_point_shots_l299_299065


namespace distance_between_stations_l299_299333

theorem distance_between_stations
  (time_start_train1 time_meet time_start_train2 : ℕ) -- time in hours (7 a.m., 11 a.m., 8 a.m.)
  (speed_train1 speed_train2 : ℕ) -- speed in kmph (20 kmph, 25 kmph)
  (distance_covered_train1 distance_covered_train2 : ℕ)
  (total_distance : ℕ) :
  time_start_train1 = 7 ∧ time_meet = 11 ∧ time_start_train2 = 8 ∧ speed_train1 = 20 ∧ speed_train2 = 25 ∧
  distance_covered_train1 = (time_meet - time_start_train1) * speed_train1 ∧
  distance_covered_train2 = (time_meet - time_start_train2) * speed_train2 ∧
  total_distance = distance_covered_train1 + distance_covered_train2 →
  total_distance = 155 := by
{
  sorry
}

end distance_between_stations_l299_299333


namespace boat_speed_determination_l299_299605

theorem boat_speed_determination :
  ∃ x : ℝ, 
    (∀ u d : ℝ, u = 170 / (x + 6) ∧ d = 170 / (x - 6))
    ∧ (u + d = 68)
    ∧ (x = 9) := 
by
  sorry

end boat_speed_determination_l299_299605


namespace inequality_holds_for_any_x_l299_299434

theorem inequality_holds_for_any_x (n : ℕ) (a : Fin n → ℝ) (h₀ : n > 0) 
  (h₁ : ∀ i j, i ≤ j → a i ≤ a j) (h₂ : (Finset.sum (Finset.range n) (λ i, (i + 1) * (a ⟨i, Nat.lt_of_lt_of_le (Nat.lt_succ_self i) (Nat.lt_of_succ_lt_succ h₀).le⟩))) = 0) :
  ∀ x : ℝ, 0 ≤ (Finset.sum (Finset.range n) (λ i, (a ⟨i, Nat.lt_of_lt_of_le (Nat.lt_succ_self i) (Nat.lt_of_succ_lt_succ h₀).le⟩) * ⌊(i + 1) * x⌋)) :=
begin
  sorry
end

end inequality_holds_for_any_x_l299_299434


namespace fraction_product_l299_299080

theorem fraction_product :
  (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 :=
by sorry

end fraction_product_l299_299080


namespace sum_first_10_terms_l299_299121

def arithmetic_sequence (a d : Int) (n : Int) : Int :=
  a + (n - 1) * d

def arithmetic_sum (a d : Int) (n : Int) : Int :=
  (n : Int) * a + (n * (n - 1) / 2) * d

theorem sum_first_10_terms  
  (a d : Int)
  (h1 : (a + 3 * d)^2 = (a + 2 * d) * (a + 6 * d))
  (h2 : arithmetic_sum a d 8 = 32)
  : arithmetic_sum a d 10 = 60 :=
sorry

end sum_first_10_terms_l299_299121


namespace number_of_students_l299_299455

/--
Statement: Several students are seated around a circular table. 
Each person takes one piece from a bag containing 120 pieces of candy 
before passing it to the next. Chris starts with the bag, takes one piece 
and also ends up with the last piece. Prove that the number of students
at the table could be 7 or 17.
-/
theorem number_of_students (n : Nat) (h : 120 > 0) :
  (∃ k, 119 = k * n ∧ n ≥ 1) → (n = 7 ∨ n = 17) :=
by
  sorry

end number_of_students_l299_299455


namespace lcm_18_24_eq_72_l299_299767

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299767


namespace smallest_sum_of_digits_l299_299200

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_sum_of_digits (N : ℕ) (hN_pos : 0 < N) 
  (h : sum_of_digits N = 3 * sum_of_digits (N + 1)) :
  sum_of_digits N = 12 :=
by {
  sorry
}

end smallest_sum_of_digits_l299_299200


namespace division_of_fractions_l299_299336

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299336


namespace rita_bought_4_pounds_l299_299949

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l299_299949


namespace polar_coordinates_of_point_l299_299523

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ) (r θ : ℝ), x = -1 ∧ y = 1 ∧ r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi
  → r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := 
by
  intros x y r θ h
  sorry

end polar_coordinates_of_point_l299_299523


namespace correct_statement_l299_299213

variable (P Q : Prop)
variable (hP : P)
variable (hQ : Q)

theorem correct_statement :
  (P ∧ Q) :=
by
  exact ⟨hP, hQ⟩

end correct_statement_l299_299213


namespace only_A_can_form_triangle_l299_299212

/--
Prove that from the given sets of lengths, only the set {5cm, 8cm, 12cm} can form a valid triangle.

Given:
- A: 5 cm, 8 cm, 12 cm
- B: 2 cm, 3 cm, 6 cm
- C: 3 cm, 3 cm, 6 cm
- D: 4 cm, 7 cm, 11 cm

We need to show that only Set A satisfies the triangle inequality theorem.
-/
theorem only_A_can_form_triangle :
  (∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 12 → a + b > c ∧ a + c > b ∧ b + c > a) ∧
  (∀ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 3 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 4 ∧ b = 7 ∧ c = 11 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry -- Proof to be provided

end only_A_can_form_triangle_l299_299212


namespace find_f_minus_2_l299_299942

namespace MathProof

def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 5

theorem find_f_minus_2 (a b c : ℝ) (h : f a b c 2 = 3) : f a b c (-2) = -13 := 
by
  sorry

end MathProof

end find_f_minus_2_l299_299942


namespace quadratic_real_roots_iff_l299_299254

theorem quadratic_real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 :=
by
  -- Proof is omitted, we only need the statement
  sorry

end quadratic_real_roots_iff_l299_299254


namespace four_consecutive_integers_plus_one_is_square_l299_299738

theorem four_consecutive_integers_plus_one_is_square (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n ^ 2 + n - 1) ^ 2 := 
by 
  sorry

end four_consecutive_integers_plus_one_is_square_l299_299738


namespace not_characteristic_of_algorithm_l299_299820

def characteristic_of_algorithm (c : String) : Prop :=
  c = "Abstraction" ∨ c = "Precision" ∨ c = "Finiteness"

theorem not_characteristic_of_algorithm : 
  ¬ characteristic_of_algorithm "Uniqueness" :=
by
  sorry

end not_characteristic_of_algorithm_l299_299820


namespace closest_ratio_adults_children_l299_299595

theorem closest_ratio_adults_children (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 2) (h3 : c ≥ 2) : 
  (a : ℚ) / (c : ℚ) = 1 :=
  sorry

end closest_ratio_adults_children_l299_299595


namespace division_of_fractions_l299_299348

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299348


namespace find_cost_price_l299_299972

theorem find_cost_price (C : ℝ) (h1 : 0.88 * C + 1500 = 1.12 * C) : C = 6250 := 
by
  sorry

end find_cost_price_l299_299972


namespace largest_angle_in_ratio_3_4_5_l299_299187

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l299_299187


namespace lcm_18_24_l299_299810
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299810


namespace no_such_integers_l299_299259

def p (x : ℤ) : ℤ := x^2 + x - 70

theorem no_such_integers : ¬ (∃ m n : ℤ, 0 < m ∧ m < n ∧ n ∣ p m ∧ (n + 1) ∣ p (m + 1)) :=
by
  sorry

end no_such_integers_l299_299259


namespace car_average_speed_l299_299087

-- Definitions based on conditions
def distance_first_hour : ℤ := 100
def distance_second_hour : ℤ := 60
def time_first_hour : ℤ := 1
def time_second_hour : ℤ := 1

-- Total distance and time calculations
def total_distance : ℤ := distance_first_hour + distance_second_hour
def total_time : ℤ := time_first_hour + time_second_hour

-- The average speed of the car
def average_speed : ℤ := total_distance / total_time

-- Proof statement
theorem car_average_speed : average_speed = 80 := by
  sorry

end car_average_speed_l299_299087


namespace find_e_l299_299239

-- Definitions of the problem conditions
def Q (x : ℝ) (f d e : ℝ) := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) :
  (∀ x : ℝ, Q x f d e = 3 * x^3 + d * x^2 + e * x + f) →
  (f = 9) →
  ((∃ p q r : ℝ, p + q + r = - d / 3 ∧ p * q * r = - f / 3
    ∧ 1 / (p + q + r) = -3
    ∧ 3 + d + e + f = p * q * r) →
    e = -16) :=
by
  intros hQ hf hroots
  sorry

end find_e_l299_299239


namespace smallest_n_for_cookies_l299_299714

theorem smallest_n_for_cookies :
  ∃ n : ℕ, 15 * n - 1 % 11 = 0 ∧ (∀ m : ℕ, 15 * m - 1 % 11 = 0 → n ≤ m) :=
sorry

end smallest_n_for_cookies_l299_299714


namespace complex_number_sum_l299_299692

noncomputable def x : ℝ := 3 / 5
noncomputable def y : ℝ := -3 / 5

theorem complex_number_sum :
  (x + y) = -2 / 5 := 
by
  sorry

end complex_number_sum_l299_299692


namespace rhombus_area_l299_299667

noncomputable def sqrt125 : ℝ := Real.sqrt 125

theorem rhombus_area 
  (p q : ℝ) 
  (h1 : p < q) 
  (h2 : p + 8 = q) 
  (h3 : ∀ a b : ℝ, a^2 + b^2 = 125 ↔ 2*a = p ∧ 2*b = q) : 
  p*q/2 = 60.5 :=
by
  sorry

end rhombus_area_l299_299667


namespace find_m_l299_299898

-- Definitions based on conditions
def Point (α : Type) := α × α

def A : Point ℝ := (2, -3)
def B : Point ℝ := (4, 3)
def C (m : ℝ) : Point ℝ := (5, m)

-- The collinearity condition
def collinear (p1 p2 p3 : Point ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- The proof problem
theorem find_m (m : ℝ) : collinear A B (C m) → m = 6 :=
by
  sorry

end find_m_l299_299898


namespace reduction_for_same_profit_cannot_reach_460_profit_l299_299555

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l299_299555


namespace cone_volume_proof_l299_299536

noncomputable def slant_height := 21
noncomputable def horizontal_semi_axis := 10
noncomputable def vertical_semi_axis := 12
noncomputable def equivalent_radius :=
  Real.sqrt (horizontal_semi_axis * vertical_semi_axis)
noncomputable def cone_height :=
  Real.sqrt (slant_height ^ 2 - equivalent_radius ^ 2)

noncomputable def cone_volume :=
  (1 / 3) * Real.pi * horizontal_semi_axis * vertical_semi_axis * cone_height

theorem cone_volume_proof :
  cone_volume = 2250.24 * Real.pi := sorry

end cone_volume_proof_l299_299536


namespace monotone_f_range_a_l299_299054

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

theorem monotone_f_range_a (a : ℝ) :
  (∀ (x y : ℝ), x <= y → f a x >= f a y) →
  1 / 2 <= a ∧ a <= 5 / 8 :=
sorry

end monotone_f_range_a_l299_299054


namespace Jackson_game_time_l299_299930

/-- Jackson's grade increases by 15 points for every hour he spends studying, 
    and his grade is 45 points, prove that he spends 9 hours playing video 
    games when he spends 3 hours studying and 1/3 of his study time on 
    playing video games. -/
theorem Jackson_game_time (S G : ℕ) (h1 : 15 * S = 45) (h2 : G = 3 * S) : G = 9 :=
by
  sorry

end Jackson_game_time_l299_299930


namespace solution_exists_l299_299248

theorem solution_exists (x : ℝ) :
  (|x - 10| + |x - 14| = |2 * x - 24|) ↔ (x = 12) :=
by
  sorry

end solution_exists_l299_299248


namespace common_difference_arithmetic_sequence_l299_299688

theorem common_difference_arithmetic_sequence :
  ∃ d : ℝ, (d ≠ 0) ∧ (∀ (n : ℕ), a_n = 1 + (n-1) * d) ∧ ((1 + 2 * d)^2 = 1 * (1 + 8 * d)) → d = 1 :=
by
  sorry

end common_difference_arithmetic_sequence_l299_299688


namespace fraction_of_salary_on_rent_l299_299225

theorem fraction_of_salary_on_rent
  (S : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (remaining_amount : ℝ) (approx_salary : ℝ)
  (food_fraction_eq : food_fraction = 1 / 5)
  (clothes_fraction_eq : clothes_fraction = 3 / 5)
  (remaining_amount_eq : remaining_amount = 19000)
  (approx_salary_eq : approx_salary = 190000) :
  ∃ (H : ℝ), H = 1 / 10 :=
by
  sorry

end fraction_of_salary_on_rent_l299_299225


namespace find_max_z_l299_299918

theorem find_max_z :
  ∃ (x y : ℝ), abs x + abs y ≤ 4 ∧ 2 * x + y ≤ 4 ∧ (2 * x - y) = (20 / 3) :=
by
  sorry

end find_max_z_l299_299918


namespace lcm_18_24_l299_299804

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299804


namespace lcm_18_24_l299_299793

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299793


namespace each_persons_share_l299_299985

def total_bill : ℝ := 211.00
def number_of_people : ℕ := 5
def tip_rate : ℝ := 0.15

theorem each_persons_share :
  (total_bill * (1 + tip_rate)) / number_of_people = 48.53 := 
by sorry

end each_persons_share_l299_299985


namespace triangle_to_initial_position_l299_299844

-- Definitions for triangle vertices
structure Point where
  x : Int
  y : Int

def p1 : Point := { x := 0, y := 0 }
def p2 : Point := { x := 6, y := 0 }
def p3 : Point := { x := 0, y := 4 }

-- Definitions for transformations
def rotate90 (p : Point) : Point := { x := -p.y, y := p.x }
def rotate180 (p : Point) : Point := { x := -p.x, y := -p.y }
def rotate270 (p : Point) : Point := { x := p.y, y := -p.x }
def reflect_y_eq_x (p : Point) : Point := { x := p.y, y := p.x }
def reflect_y_eq_neg_x (p : Point) : Point := { x := -p.y, y := -p.x }

-- Definitions for combination of transformations
-- This part defines how to combine transformations, e.g., as a sequence of three transformations.
def transform (fs : List (Point → Point)) (p : Point) : Point :=
  fs.foldl (fun acc f => f acc) p

-- The total number of valid sequences that return the triangle to its original position
def valid_sequences_count : Int := 6

-- Lean 4 statement
theorem triangle_to_initial_position : valid_sequences_count = 6 := by
  sorry

end triangle_to_initial_position_l299_299844


namespace CatsFavoriteNumber_l299_299107

theorem CatsFavoriteNumber :
  ∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧ 
    (∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n = p1 * p2 * p3) ∧ 
    (∀ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      n ≠ a ∧ n ≠ b ∧ n ≠ c ∧ n ≠ d ∧
      a + b - c = d ∨ b + c - d = a ∨ c + d - a = b ∨ d + a - b = c →
      (a = 30 ∧ b = 42 ∧ c = 66 ∧ d = 78)) ∧
    (n = 70) := by
  sorry

end CatsFavoriteNumber_l299_299107


namespace triangle_ABC_angles_l299_299287

theorem triangle_ABC_angles (A B C D M : Point)
  (h1: foot_of_altitude D A B C)
  (h2: midpoint M B C)
  (h3: ∠BAD = ∠DAM ∧ ∠DAM = ∠MAC) :
  is_triangle_with_angles A B C 90 60 30 :=
sorry

end triangle_ABC_angles_l299_299287


namespace probability_after_6_passes_l299_299608

noncomputable section

-- We define people
inductive Person
| A | B | C

-- Probability that person A has the ball after n passes
def P : ℕ → Person → ℚ
| 0, Person.A => 1
| 0, _ => 0
| n+1, Person.A => (P n Person.B + P n Person.C) / 2
| n+1, Person.B => (P n Person.A + P n Person.C) / 2
| n+1, Person.C => (P n Person.A + P n Person.B) / 2

theorem probability_after_6_passes :
  P 6 Person.A = 11 / 32 := by
  sorry

end probability_after_6_passes_l299_299608


namespace remainder_expression_l299_299981

theorem remainder_expression (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) : 
  (x + 3 * u * y) % y = v := 
by
  sorry

end remainder_expression_l299_299981


namespace red_section_no_damage_probability_l299_299657

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l299_299657


namespace quadratic_equation_solution_l299_299152

-- Define the problem statement and the conditions: the equation being quadratic.
theorem quadratic_equation_solution (m : ℤ) :
  (∃ (a : ℤ), a ≠ 0 ∧ (a*x^2 - x - 2 = 0)) →
  m = -1 :=
by
  sorry

end quadratic_equation_solution_l299_299152


namespace solve_for_a_l299_299560

theorem solve_for_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_eq_exponents : a ^ b = b ^ a) (h_b_equals_3a : b = 3 * a) : a = Real.sqrt 3 :=
sorry

end solve_for_a_l299_299560


namespace constant_sum_l299_299760

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem constant_sum (a1 d : ℝ) (h : 3 * arithmetic_sequence a1 d 8 = k) :
  ∃ k : ℝ, sum_arithmetic_sequence a1 d 15 = k :=
sorry

end constant_sum_l299_299760


namespace lcm_18_24_eq_72_l299_299789

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299789


namespace arithmetic_sequence_l299_299298

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l299_299298


namespace bacon_percentage_l299_299862

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l299_299862


namespace zero_is_multiple_of_all_primes_l299_299360

theorem zero_is_multiple_of_all_primes :
  ∀ (x : ℕ), (∀ p : ℕ, Prime p → ∃ n : ℕ, x = n * p) ↔ x = 0 := by
sorry

end zero_is_multiple_of_all_primes_l299_299360


namespace circumscribed_sphere_radius_is_3_l299_299547

noncomputable def radius_of_circumscribed_sphere (SA SB SC : ℝ) : ℝ :=
  let space_diagonal := Real.sqrt (SA^2 + SB^2 + SC^2)
  space_diagonal / 2

theorem circumscribed_sphere_radius_is_3 : radius_of_circumscribed_sphere 2 4 4 = 3 :=
by
  unfold radius_of_circumscribed_sphere
  simp
  apply sorry

end circumscribed_sphere_radius_is_3_l299_299547


namespace division_of_fractions_l299_299351

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299351


namespace opposite_of_pi_l299_299966

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l299_299966


namespace cylindrical_to_rectangular_conversion_l299_299868

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 6 (5 * Real.pi / 4) (-3) = (-3 * Real.sqrt 2, -3 * Real.sqrt 2, -3) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l299_299868


namespace saturday_earnings_l299_299093

variable (S : ℝ)
variable (totalEarnings : ℝ := 5182.50)
variable (difference : ℝ := 142.50)

theorem saturday_earnings : 
  S + (S - difference) = totalEarnings → S = 2662.50 := 
by 
  intro h 
  sorry

end saturday_earnings_l299_299093


namespace evaluate_expression_at_x_eq_2_l299_299975

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end evaluate_expression_at_x_eq_2_l299_299975


namespace number_of_articles_l299_299270

variables (C S N : ℝ)
noncomputable def gain : ℝ := 3 / 7

-- Cost price of 50 articles is equal to the selling price of N articles
axiom cost_price_eq_selling_price : 50 * C = N * S

-- Selling price is cost price plus gain percentage
axiom selling_price_with_gain : S = C * (1 + gain)

-- Goal: Prove that N = 35
theorem number_of_articles (h1 : 50 * C = N * C * (10 / 7)) : N = 35 := by
  sorry

end number_of_articles_l299_299270


namespace percentage_of_x_is_2x_minus_y_l299_299276

variable (x y : ℝ)
variable (h1 : x / y = 4)
variable (h2 : y ≠ 0)

theorem percentage_of_x_is_2x_minus_y :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end percentage_of_x_is_2x_minus_y_l299_299276


namespace total_shapes_proof_l299_299746

def stars := 50
def stripes := 13

def circles : ℕ := (stars / 2) - 3
def squares : ℕ := (2 * stripes) + 6
def triangles : ℕ := (stars - stripes) * 2
def diamonds : ℕ := (stars + stripes) / 4

def total_shapes : ℕ := circles + squares + triangles + diamonds

theorem total_shapes_proof : total_shapes = 143 := by
  sorry

end total_shapes_proof_l299_299746


namespace division_of_fractions_l299_299337

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299337


namespace greater_number_is_84_l299_299624

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) : x = 84 :=
sorry

end greater_number_is_84_l299_299624


namespace intersecting_lines_solution_l299_299961

theorem intersecting_lines_solution (a b : ℝ) :
  (∃ (a b : ℝ), 
    ((a^2 + 1) * 2 - 2 * b * (-3) = 4) ∧ 
    ((1 - a) * 2 + b * (-3) = 9)) →
  (a, b) = (4, -5) ∨ (a, b) = (-2, -1) :=
by
  sorry

end intersecting_lines_solution_l299_299961


namespace exotic_meat_original_price_l299_299224

theorem exotic_meat_original_price (y : ℝ) :
  (0.75 * (y / 4) = 4.5) → y = 96 :=
by
  intro h
  sorry

end exotic_meat_original_price_l299_299224


namespace sqrt_sum_equality_l299_299355

open Real

theorem sqrt_sum_equality :
  (sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8) :=
sorry

end sqrt_sum_equality_l299_299355


namespace expression_equals_two_l299_299315

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l299_299315


namespace sum_of_inscribed_angles_l299_299218

-- Define the circle and its division into arcs.
def circle_division (O : Type) (total_arcs : ℕ) := total_arcs = 16

-- Define the inscribed angles x and y.
def inscribed_angle (O : Type) (arc_subtended : ℕ) := arc_subtended

-- Define the conditions for angles x and y subtending 3 and 5 arcs respectively.
def angle_x := inscribed_angle ℝ 3
def angle_y := inscribed_angle ℝ 5

-- Theorem stating the sum of the inscribed angles x and y.
theorem sum_of_inscribed_angles 
  (O : Type)
  (total_arcs : ℕ)
  (h1 : circle_division O total_arcs)
  (h2 : inscribed_angle O angle_x = 3)
  (h3 : inscribed_angle O angle_y = 5) :
  33.75 + 56.25 = 90 :=
by
  sorry

end sum_of_inscribed_angles_l299_299218


namespace eval_expression_l299_299887

theorem eval_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) : (x + 1) / (x - 1) = 1 + Real.sqrt 2 := 
by
  sorry

end eval_expression_l299_299887


namespace simple_interest_rate_l299_299840

theorem simple_interest_rate :
  ∀ (P : ℝ) (R : ℝ),
    (9 / 6) * P = P + (P * R * 12) / 100 →
    R = 100 / 24 :=
by {
  intros P R H,
  sorry
}

end simple_interest_rate_l299_299840


namespace length_of_other_parallel_side_l299_299114

theorem length_of_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : 323 = 1/2 * (20 + b) * 17) :
  b = 18 :=
sorry

end length_of_other_parallel_side_l299_299114


namespace find_x_value_l299_299926

noncomputable def x_value (x y z : ℝ) : Prop :=
  (26 = (z + x) / 2) ∧
  (z = 52 - x) ∧
  (52 - x = (26 + y) / 2) ∧
  (y = 78 - 2 * x) ∧
  (78 - 2 * x = (8 + (52 - x)) / 2) ∧
  (x = 32)

theorem find_x_value : ∃ x y z : ℝ, x_value x y z :=
by
  use 32  -- x
  use 14  -- y derived from 78 - 2x where x = 32 leads to y = 14
  use 20  -- z derived from 52 - x where x = 32 leads to z = 20
  unfold x_value
  simp
  sorry

end find_x_value_l299_299926


namespace increase_75_by_150_percent_l299_299482

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l299_299482


namespace number_of_fifth_graders_l299_299745

-- Define the conditions given in the problem.
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Derived definitions with the help of the conditions.
def total_seats : ℕ := buses * seats_per_bus
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_sixth_and_seventh_graders : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := total_sixth_and_seventh_graders + total_chaperones
def seats_for_fifth_graders : ℕ := total_seats - seats_taken

-- The final statement to prove the number of fifth graders.
theorem number_of_fifth_graders : seats_for_fifth_graders = 109 :=
by
  sorry

end number_of_fifth_graders_l299_299745


namespace insulin_pills_per_day_l299_299262

def conditions (I B A : ℕ) : Prop := 
  B = 3 ∧ A = 2 * B ∧ 7 * (I + B + A) = 77

theorem insulin_pills_per_day : ∃ (I : ℕ), ∀ (B A : ℕ), conditions I B A → I = 2 := by
  sorry

end insulin_pills_per_day_l299_299262


namespace cube_edge_length_surface_area_equals_volume_l299_299117

theorem cube_edge_length_surface_area_equals_volume (a : ℝ) (h : 6 * a ^ 2 = a ^ 3) : a = 6 := 
by {
  sorry
}

end cube_edge_length_surface_area_equals_volume_l299_299117


namespace race_course_length_l299_299086

variable (v_A v_B d : ℝ)

theorem race_course_length (h1 : v_A = 4 * v_B) (h2 : (d - 60) / v_B = d / v_A) : d = 80 := by
  sorry

end race_course_length_l299_299086


namespace stratified_sampling_number_of_grade12_students_in_sample_l299_299100

theorem stratified_sampling_number_of_grade12_students_in_sample 
  (total_students : ℕ)
  (students_grade10 : ℕ)
  (students_grade11_minus_grade12 : ℕ)
  (sampled_students_grade10 : ℕ)
  (total_students_eq : total_students = 1290)
  (students_grade10_eq : students_grade10 = 480)
  (students_grade11_minus_grade12_eq : students_grade11_minus_grade12 = 30)
  (sampled_students_grade10_eq : sampled_students_grade10 = 96) :
  ∃ n : ℕ, n = 78 :=
by
  -- Proof would go here, but we are skipping with "sorry"
  sorry

end stratified_sampling_number_of_grade12_students_in_sample_l299_299100


namespace trees_distance_l299_299922

theorem trees_distance (num_trees : ℕ) (yard_length : ℕ) (trees_at_end : Prop) (tree_count : num_trees = 26) (yard_size : yard_length = 800) : 
  (yard_length / (num_trees - 1)) = 32 := 
by
  sorry

end trees_distance_l299_299922


namespace unique_three_digit_base_g_l299_299450

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end unique_three_digit_base_g_l299_299450


namespace shape_at_22_l299_299063

-- Define the pattern
def pattern : List String := ["triangle", "square", "diamond", "diamond", "circle"]

-- Function to get the nth shape in the repeated pattern sequence
def getShape (n : Nat) : String :=
  pattern.get! (n % pattern.length)

-- Statement to prove
theorem shape_at_22 : getShape 21 = "square" :=
by
  sorry

end shape_at_22_l299_299063


namespace erasers_per_friend_l299_299527

variable (erasers friends : ℕ)

theorem erasers_per_friend (h1 : erasers = 3840) (h2 : friends = 48) :
  erasers / friends = 80 :=
by sorry

end erasers_per_friend_l299_299527


namespace lcm_18_24_eq_72_l299_299780

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299780


namespace cos_double_angle_l299_299549

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4 / 5) : Real.cos (2 * α) = 7 / 25 := 
by
  sorry

end cos_double_angle_l299_299549


namespace negation_equivalence_l299_299753

variables (x : ℝ)

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

def has_rational_square (x : ℝ) : Prop := ∃ (q : ℚ), ↑q * ↑q = x * x

def proposition := ∃ (x : ℝ), is_irrational x ∧ has_rational_square x

theorem negation_equivalence :
  (¬ proposition) ↔ ∀ (x : ℝ), is_irrational x → ¬ has_rational_square x :=
by sorry

end negation_equivalence_l299_299753


namespace original_number_is_80_l299_299825

variable (e : ℝ)

def increased_value := 1.125 * e
def decreased_value := 0.75 * e
def difference_condition := increased_value e - decreased_value e = 30

theorem original_number_is_80 (h : difference_condition e) : e = 80 :=
sorry

end original_number_is_80_l299_299825


namespace gcd_gx_x_l299_299891

theorem gcd_gx_x (x : ℕ) (h : 2520 ∣ x) : 
  Nat.gcd ((4*x + 5) * (5*x + 2) * (11*x + 8) * (3*x + 7)) x = 280 := 
sorry

end gcd_gx_x_l299_299891


namespace partA_partB_partC_l299_299326
noncomputable section

def n : ℕ := 100
def p : ℝ := 0.8
def q : ℝ := 1 - p

def binomial_prob (k1 k2 : ℕ) : ℝ := sorry

theorem partA : binomial_prob 70 85 = 0.8882 := sorry
theorem partB : binomial_prob 70 100 = 0.9938 := sorry
theorem partC : binomial_prob 0 69 = 0.0062 := sorry

end partA_partB_partC_l299_299326


namespace return_time_is_2_hours_l299_299031

noncomputable def distance_home_city_hall := 6
noncomputable def speed_to_city_hall := 3 -- km/h
noncomputable def additional_distance_return := 2 -- km
noncomputable def speed_return := 4 -- km/h
noncomputable def total_trip_time := 4 -- hours

theorem return_time_is_2_hours :
  (distance_home_city_hall + additional_distance_return) / speed_return = 2 :=
by
  sorry

end return_time_is_2_hours_l299_299031


namespace first_investment_percentage_l299_299277

variable (P : ℝ)
variable (x : ℝ := 1400)  -- investment amount in the first investment
variable (y : ℝ := 600)   -- investment amount at 8 percent
variable (income_difference : ℝ := 92)
variable (total_investment : ℝ := 2000)
variable (rate_8_percent : ℝ := 0.08)
variable (exceed_by : ℝ := 92)

theorem first_investment_percentage :
  P * x - rate_8_percent * y = exceed_by →
  total_investment = x + y →
  P = 0.10 :=
by
  -- Solution steps can be filled here if needed
  sorry

end first_investment_percentage_l299_299277


namespace smallest_value_l299_299144

theorem smallest_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (h1 : a = 2 * b) (h2 : b = 2 * c) (h3 : 4 * c = a) :
    (Int.floor ((a + b : ℚ) / c) + Int.floor ((b + c : ℚ) / a) + Int.floor ((c + a : ℚ) / b)) = 8 := 
sorry

end smallest_value_l299_299144


namespace solution_set_of_inequality_l299_299690

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  { x : ℝ | |f (x - 2)| > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end solution_set_of_inequality_l299_299690


namespace original_length_in_meters_l299_299586

-- Conditions
def erased_length : ℝ := 10 -- 10 cm
def remaining_length : ℝ := 90 -- 90 cm

-- Question: What is the original length of the line in meters?
theorem original_length_in_meters : (remaining_length + erased_length) / 100 = 1 := 
by 
  -- The proof is omitted
  sorry

end original_length_in_meters_l299_299586


namespace diff_between_roots_l299_299531

theorem diff_between_roots (p : ℝ) (r s : ℝ)
  (h_eq : ∀ x : ℝ, x^2 - (p+1)*x + (p^2 + 2*p - 3)/4 = 0 → x = r ∨ x = s)
  (h_ge : r ≥ s) :
  r - s = Real.sqrt (2*p + 1 - p^2) := by
  sorry

end diff_between_roots_l299_299531


namespace minimum_number_of_tiles_l299_299994

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l299_299994


namespace liquid_X_percentage_in_B_l299_299580

noncomputable def percentage_of_solution_B (X_A : ℝ) (w_A w_B total_X : ℝ) : ℝ :=
  let X_B := (total_X - (w_A * (X_A / 100))) / w_B 
  X_B * 100

theorem liquid_X_percentage_in_B :
  percentage_of_solution_B 0.8 500 700 19.92 = 2.274 := by
  sorry

end liquid_X_percentage_in_B_l299_299580


namespace count_solutions_cos2x_plus_3sin2x_eq_1_l299_299912

open Real

theorem count_solutions_cos2x_plus_3sin2x_eq_1 :
  ∀ x : ℝ, (-10 < x ∧ x < 45 → cos x ^ 2 + 3 * sin x ^ 2 = 1) → 
  ∃! n : ℕ, n = 18 := 
by
  intro x hEq
  sorry

end count_solutions_cos2x_plus_3sin2x_eq_1_l299_299912


namespace parabola_conditions_l299_299062

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l299_299062


namespace curve_B_is_not_good_l299_299686

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end curve_B_is_not_good_l299_299686


namespace find_a_c_l299_299969

theorem find_a_c (a c : ℝ) (h1 : a + c = 35) (h2 : a < c)
  (h3 : ∀ x : ℝ, a * x^2 + 30 * x + c = 0 → ∃! x, a * x^2 + 30 * x + c = 0) :
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by
  sorry

end find_a_c_l299_299969


namespace arithmetic_expression_equality_l299_299515

theorem arithmetic_expression_equality :
  ( ( (4 + 6 + 5) * 2 ) / 4 - ( (3 * 2) / 4 ) ) = 6 :=
by sorry

end arithmetic_expression_equality_l299_299515


namespace bug_probability_nine_moves_l299_299367

noncomputable def bug_cube_probability (moves : ℕ) : ℚ := sorry

/-- 
The probability that after exactly 9 moves, a bug starting at one vertex of a cube 
and moving randomly along the edges will have visited every vertex exactly once and 
revisited one vertex once more. 
-/
theorem bug_probability_nine_moves : bug_cube_probability 9 = 16 / 6561 := by
  sorry

end bug_probability_nine_moves_l299_299367


namespace red_section_no_damage_probability_l299_299654

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l299_299654


namespace pure_imaginary_solution_l299_299167

theorem pure_imaginary_solution (a : ℝ) (ha : a + 5 * Complex.I / (1 - 2 * Complex.I) = a + (1 : ℂ) * Complex.I) :
  a = 2 :=
by
  sorry

end pure_imaginary_solution_l299_299167


namespace smallest_solution_is_neg_sqrt_13_l299_299400

noncomputable def smallest_solution (x : ℝ) : Prop :=
  x^4 - 26 * x^2 + 169 = 0 ∧ ∀ y : ℝ, y^4 - 26 * y^2 + 169 = 0 → x ≤ y

theorem smallest_solution_is_neg_sqrt_13 :
  smallest_solution (-Real.sqrt 13) :=
by
  sorry

end smallest_solution_is_neg_sqrt_13_l299_299400


namespace no_ratio_p_squared_l299_299166

theorem no_ratio_p_squared {p : ℕ} (hp : Nat.Prime p) :
  ∀ l n m : ℕ, 1 ≤ l → (∃ k : ℕ, k = p^l) → ((2 * (n*(n+1)) = (m*(m+1))*p^(2*l)) → false) := 
sorry

end no_ratio_p_squared_l299_299166


namespace value_of_m_l299_299008

theorem value_of_m (m : ℝ) :
  let A := {2, 3}
  let B := {x : ℝ | m * x - 6 = 0}
  (B ⊆ A) → (m = 0 ∨ m = 2 ∨ m = 3) :=
by
  intros A B h
  sorry

end value_of_m_l299_299008


namespace non_differentiable_counter_example_continuous_implies_differentiable_l299_299625

noncomputable def example_counter_example_f (x : ℝ) : ℝ :=
  if x ∈ Set.uprod (Set.range (Set.univ : Set ℚ)) then 1 else 0

noncomputable def example_g (x : ℝ) : ℝ := 0

def example_a_n (n : ℕ) : ℝ := 1 / (n : ℝ)

theorem non_differentiable_counter_example :
  let f := example_counter_example_f
  let g := example_g
  let a_n := example_a_n
  (∀ x : ℝ, g' x = lim (n → ∞) (f (x + a_n n) - f x) / (a_n n)) →
  ¬ (differentiable f) :=
sorry

theorem continuous_implies_differentiable {f : ℝ → ℝ} {g : ℝ → ℝ}
  (a_n : ℕ → ℝ) (h_a_n : ∀ n, a_n n > 0) (h_a_n_zero : lim (n → ∞) a_n n = 0) :
  (∀ x : ℝ, differentiable g ∧
    (∀ x : ℝ, g' x = lim (n → ∞) (f (x + a_n n) - f x) / (a_n n))) →
  continuous f →
  differentiable f :=
sorry

end non_differentiable_counter_example_continuous_implies_differentiable_l299_299625


namespace greatest_possible_d_l299_299837

noncomputable def point_2d_units_away_origin (d : ℝ) : Prop :=
  2 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d + 5)^2)

theorem greatest_possible_d : 
  ∃ d : ℝ, point_2d_units_away_origin d ∧ d = (5 + Real.sqrt 244) / 3 :=
sorry

end greatest_possible_d_l299_299837


namespace fireworks_display_l299_299381

def num_digits_year : ℕ := 4
def fireworks_per_digit : ℕ := 6
def regular_letters_phrase : ℕ := 12
def fireworks_per_regular_letter : ℕ := 5

def fireworks_H : ℕ := 8
def fireworks_E : ℕ := 7
def fireworks_L : ℕ := 6
def fireworks_O : ℕ := 9

def num_boxes : ℕ := 100
def fireworks_per_box : ℕ := 10

def total_fireworks : ℕ :=
  (num_digits_year * fireworks_per_digit) +
  (regular_letters_phrase * fireworks_per_regular_letter) +
  (fireworks_H + fireworks_E + 2 * fireworks_L + fireworks_O) + 
  (num_boxes * fireworks_per_box)

theorem fireworks_display : total_fireworks = 1120 := by
  sorry

end fireworks_display_l299_299381


namespace simplify_log_expression_l299_299314

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l299_299314


namespace find_arith_seq_common_diff_l299_299691

-- Let a_n be the nth term of the arithmetic sequence and S_n be the sum of the first n terms
variable {a : ℕ → ℝ} -- arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of first n terms of the sequence

-- Given conditions in the problem
axiom sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3)
axiom sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2)
axiom condition1 : ((S 4) / 12) - ((S 3) / 9) = 1

-- Prove that the common difference d is 6
theorem find_arith_seq_common_diff (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3))
  (sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2))
  (condition1 : (S 4) / 12 - (S 3) / 9 = 1) : 
  d = 6 := 
sorry

end find_arith_seq_common_diff_l299_299691


namespace lcm_18_24_l299_299795

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299795


namespace expression_value_l299_299211

-- Proving the value of the expression using the factorial and sum formulas
theorem expression_value :
  (Nat.factorial 10) / (10 * 11 / 2) = 66069 := 
sorry

end expression_value_l299_299211


namespace first_percentage_increase_l299_299602

theorem first_percentage_increase (x : ℝ) :
  (1 + x / 100) * 1.4 = 1.82 → x = 30 := 
by 
  intro h
  -- start your proof here
  sorry

end first_percentage_increase_l299_299602


namespace cost_price_l299_299849

theorem cost_price (MP : ℝ) (SP : ℝ) (C : ℝ) 
  (h1 : MP = 87.5) 
  (h2 : SP = 0.95 * MP) 
  (h3 : SP = 1.25 * C) : 
  C = 66.5 := 
by
  sorry

end cost_price_l299_299849


namespace sqrt_144_times_3_squared_l299_299520

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end sqrt_144_times_3_squared_l299_299520


namespace contradiction_proof_l299_299615

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l299_299615


namespace calculation_result_l299_299208

theorem calculation_result :
  15 * (1 / 3 + 1 / 4 + 1 / 6)⁻¹ = 20 := by
  sorry

end calculation_result_l299_299208


namespace bird_problem_l299_299201

theorem bird_problem (B : ℕ) (h : (2 / 15) * B = 60) : B = 450 ∧ (2 / 15) * B = 60 :=
by
  sorry

end bird_problem_l299_299201


namespace number_of_paths_3x3_l299_299870

-- Definition of the problem conditions
def grid_moves (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- Lean statement for the proof problem
theorem number_of_paths_3x3 : grid_moves 3 3 = 20 := by
  sorry

end number_of_paths_3x3_l299_299870


namespace exponent_equation_l299_299702

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l299_299702


namespace quadratic_roots_sum_product_l299_299275

theorem quadratic_roots_sum_product : 
  ∃ x1 x2 : ℝ, (x1^2 - 2*x1 - 4 = 0) ∧ (x2^2 - 2*x2 - 4 = 0) ∧ 
  (x1 ≠ x2) ∧ (x1 + x2 + x1 * x2 = -2) :=
sorry

end quadratic_roots_sum_product_l299_299275


namespace average_apples_per_hour_l299_299307

theorem average_apples_per_hour (total_apples : ℝ) (total_hours : ℝ) (h1 : total_apples = 5.0) (h2 : total_hours = 3.0) : total_apples / total_hours = 1.67 :=
  sorry

end average_apples_per_hour_l299_299307


namespace max_value_of_s_l299_299574

theorem max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 10)
  (h2 : p * q + p * r + p * s + q * r + q * s + r * s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 :=
sorry

end max_value_of_s_l299_299574


namespace parabola_b_value_l299_299196

variable {q : ℝ}

theorem parabola_b_value (a b c : ℝ) (h_a : a = -3 / q)
  (h_eq : ∀ x : ℝ, (a * x^2 + b * x + c) = a * (x - q)^2 + q)
  (h_intercept : (a * 0^2 + b * 0 + c) = -2 * q)
  (h_q_nonzero : q ≠ 0) :
  b = 6 / q := 
sorry

end parabola_b_value_l299_299196


namespace Nina_second_distance_l299_299946

theorem Nina_second_distance 
  (total_distance : ℝ) 
  (first_run : ℝ) 
  (second_same_run : ℝ)
  (run_twice : first_run = 0.08 ∧ second_same_run = 0.08)
  (total : total_distance = 0.83)
  : (total_distance - (first_run + second_same_run)) = 0.67 := by
  sorry

end Nina_second_distance_l299_299946


namespace inscribed_circle_radius_in_sector_l299_299953

theorem inscribed_circle_radius_in_sector
  (radius : ℝ)
  (sector_fraction : ℝ)
  (r : ℝ) :
  radius = 4 →
  sector_fraction = 1/3 →
  r = 2 * Real.sqrt 3 - 2 →
  true := by
sorry

end inscribed_circle_radius_in_sector_l299_299953


namespace Jamie_needs_to_climb_40_rungs_l299_299932

-- Define the conditions
def height_of_new_tree : ℕ := 20
def rungs_climbed_previous : ℕ := 12
def height_of_previous_tree : ℕ := 6
def rungs_per_foot := rungs_climbed_previous / height_of_previous_tree

-- Define the theorem
theorem Jamie_needs_to_climb_40_rungs :
  height_of_new_tree * rungs_per_foot = 40 :=
by
  -- Proof placeholder
  sorry

end Jamie_needs_to_climb_40_rungs_l299_299932


namespace parabola_y_range_l299_299935

theorem parabola_y_range
  (x y : ℝ)
  (M_on_C : x^2 = 8 * y)
  (F : ℝ × ℝ)
  (F_focus : F = (0, 2))
  (circle_intersects_directrix : F.2 + y > 4) :
  y > 2 :=
by
  sorry

end parabola_y_range_l299_299935


namespace sum_of_babies_ages_in_five_years_l299_299051

-- Given Definitions
def lioness_age := 12
def hyena_age := lioness_age / 2
def lioness_baby_age := lioness_age / 2
def hyena_baby_age := hyena_age / 2

-- The declaration of the statement to be proven
theorem sum_of_babies_ages_in_five_years : (lioness_baby_age + 5) + (hyena_baby_age + 5) = 19 :=
by 
  sorry 

end sum_of_babies_ages_in_five_years_l299_299051


namespace find_number_l299_299865

theorem find_number (x : ℕ) : x * 9999 = 4691130840 → x = 469200 :=
by
  intros h
  sorry

end find_number_l299_299865


namespace arithmetic_seq_problem_l299_299925

variable {a : Nat → ℝ}  -- a_n represents the value at index n
variable {d : ℝ} -- The common difference in the arithmetic sequence

-- Define the general term of the arithmetic sequence
def arithmeticSeq (a : Nat → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a n = a1 + n * d

-- The main proof problem
theorem arithmetic_seq_problem
  (a1 : ℝ)
  (d : ℝ)
  (a : Nat → ℝ)
  (h_arithmetic: arithmeticSeq a a1 d)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - (1 / 2) * a 8 = 8 := 
  by
  sorry

end arithmetic_seq_problem_l299_299925


namespace custom_op_evaluation_l299_299938

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l299_299938


namespace max_value_of_expression_l299_299498

theorem max_value_of_expression (x y : ℝ) 
  (h : Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + (Real.sqrt (y * (1 - x)) / Real.sqrt 7)) :
  x + 7 * y ≤ 57 / 8 :=
sorry

end max_value_of_expression_l299_299498


namespace jihye_wallet_total_l299_299279

-- Declare the amounts
def notes_amount : Nat := 2 * 1000
def coins_amount : Nat := 560

-- Theorem statement asserting the total amount
theorem jihye_wallet_total : notes_amount + coins_amount = 2560 := by
  sorry

end jihye_wallet_total_l299_299279


namespace probability_all_6_numbers_appear_l299_299476

noncomputable def probability_all_numbers_appear_at_least_once (n m : ℕ) (k : Fin n) : ℝ := 
  let dies := pmf.pure (λ (_ : Fin m) => fin.choose (Fin 6))
  let events := List.map (λ i => pmf.mass dies {ω | ω i = k}) (List.range m)
  1 - ∑ s in Finset.powersetFin (Finset.univ : Finset (Fin 6)), (-1) ^ (Finset.card s + 1) * ∏ i in s, events i

theorem probability_all_6_numbers_appear :
  probability_all_numbers_appear_at_least_once 10 10 6 = 0.2718 := 
sorry

end probability_all_6_numbers_appear_l299_299476


namespace largest_angle_in_ratio_triangle_l299_299188

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l299_299188


namespace multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l299_299956

variable (a b c : ℕ)

-- Define the conditions as hypotheses
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k, n = 3 * k
def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k
def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

-- Hypotheses
axiom ha : is_multiple_of_3 a
axiom hb : is_multiple_of_12 b
axiom hc : is_multiple_of_9 c

-- Statements to be proved
theorem multiple_of_3_b : is_multiple_of_3 b := sorry
theorem multiple_of_3_a_minus_b : is_multiple_of_3 (a - b) := sorry
theorem multiple_of_3_a_minus_c : is_multiple_of_3 (a - c) := sorry
theorem multiple_of_3_c_minus_b : is_multiple_of_3 (c - b) := sorry

end multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l299_299956


namespace unique_fraction_representation_l299_299389

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  ∃! (x y : ℕ), (x ≠ y) ∧ (2 * x * y = p * (x + y)) :=
by
  sorry

end unique_fraction_representation_l299_299389


namespace solve_x_1_solve_x_2_solve_x_3_l299_299535

-- Proof 1: Given 356 * x = 2492, prove that x = 7
theorem solve_x_1 (x : ℕ) (h : 356 * x = 2492) : x = 7 :=
sorry

-- Proof 2: Given x / 39 = 235, prove that x = 9165
theorem solve_x_2 (x : ℕ) (h : x / 39 = 235) : x = 9165 :=
sorry

-- Proof 3: Given 1908 - x = 529, prove that x = 1379
theorem solve_x_3 (x : ℕ) (h : 1908 - x = 529) : x = 1379 :=
sorry

end solve_x_1_solve_x_2_solve_x_3_l299_299535


namespace lcm_18_24_l299_299792

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l299_299792


namespace find_x_value_l299_299986

-- Definitions based on the conditions
def varies_inversely_as_square (k : ℝ) (x y : ℝ) : Prop := x = k / y^2

def given_condition (k : ℝ) : Prop := 1 = k / 3^2

-- The main proof problem to solve
theorem find_x_value (k : ℝ) (y : ℝ) (h1 : varies_inversely_as_square k 1 3) (h2 : y = 9) : 
  varies_inversely_as_square k (1/9) y :=
sorry

end find_x_value_l299_299986


namespace opposite_numbers_A_l299_299619

theorem opposite_numbers_A :
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1
  
  (A1 = -A2 ∧ A2 = 1) ∧ ¬(B1 = -B2) ∧ ¬(C1 = -C2) ∧ ¬(D1 = -D2)
:= by
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1

  sorry

end opposite_numbers_A_l299_299619


namespace find_a_l299_299694

theorem find_a (a : ℝ) : 
  let A := {1, 2, 3}
  let B := {x : ℝ | x^2 - (a + 1) * x + a = 0}
  A ∪ B = A → a = 1 ∨ a = 2 ∨ a = 3 :=
by
  intros
  sorry

end find_a_l299_299694


namespace green_pill_cost_l299_299384

theorem green_pill_cost : 
  ∃ (g p : ℝ), (14 * (g + p) = 546) ∧ (g = p + 1) ∧ (g = 20) :=
by
  let g := 20
  let p := g - 1
  have h1 : 14 * (g + p) = 546 := sorry -- 14 days total cost
  have h2 : g = p + 1 := by simp [p] -- cost difference
  exact ⟨g, p, h1, h2, rfl⟩

end green_pill_cost_l299_299384


namespace circle_radius_zero_l299_299883

theorem circle_radius_zero :
  ∀ (x y : ℝ),
    (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) →
    ((x - 1)^2 + (y - 2)^2 = 0) → 
    0 = 0 :=
by
  intros x y h_eq h_circle
  sorry

end circle_radius_zero_l299_299883


namespace decorations_count_l299_299524

/-
Danai is decorating her house for Halloween. She puts 12 plastic skulls all around the house.
She has 4 broomsticks, 1 for each side of the front and back doors to the house.
She puts up 12 spiderwebs around various areas of the house.
Danai puts twice as many pumpkins around the house as she put spiderwebs.
She also places a large cauldron on the dining room table.
Danai has the budget left to buy 20 more decorations and has 10 left to put up.
-/

def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def cauldron := 1
def budget_remaining := 20
def undecorated_items := 10

def initial_decorations := plastic_skulls + broomsticks + spiderwebs + pumpkins + cauldron
def additional_decorations := budget_remaining + undecorated_items
def total_decorations := initial_decorations + additional_decorations

theorem decorations_count : total_decorations = 83 := by
  /- Detailed proof steps -/
  sorry

end decorations_count_l299_299524


namespace triangle_ABC_is_acute_l299_299929

theorem triangle_ABC_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h1: a^2 + b^2 >= c^2) (h2: b^2 + c^2 >= a^2) (h3: c^2 + a^2 >= b^2)
  (h4: (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11)
  (h5: (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

end triangle_ABC_is_acute_l299_299929


namespace find_interest_rate_l299_299630

-- Definitions from the conditions
def principal : ℕ := 1050
def time_period : ℕ := 6
def interest : ℕ := 378  -- Interest calculated as Rs. 1050 - Rs. 672

-- Correct Answer
def interest_rate : ℕ := 6

-- Lean 4 statement of the proof problem
theorem find_interest_rate (P : ℕ) (t : ℕ) (I : ℕ) 
    (hP : P = principal) (ht : t = time_period) (hI : I = interest) : 
    (I * 100) / (P * t) = interest_rate :=
by {
    sorry
}

end find_interest_rate_l299_299630


namespace necessary_but_not_sufficient_for_gt_l299_299545

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_gt : a > b → a > b - 1 :=
by sorry

end necessary_but_not_sufficient_for_gt_l299_299545


namespace find_x_l299_299626

theorem find_x (x : ℝ) : (3 / 4 * 1 / 2 * 2 / 5) * x = 765.0000000000001 → x = 5100.000000000001 :=
by
  intro h
  sorry

end find_x_l299_299626


namespace harmonic_mean_is_54_div_11_l299_299007

-- Define lengths of sides
def a : ℕ := 3
def b : ℕ := 6
def c : ℕ := 9

-- Define the harmonic mean calculation function
def harmonic_mean (x y z : ℕ) : ℚ :=
  let reciprocals_sum : ℚ := (1 / x + 1 / y + 1 / z)
  let average_reciprocal : ℚ := reciprocals_sum / 3
  1 / average_reciprocal

-- Prove that the harmonic mean of the given lengths is 54/11
theorem harmonic_mean_is_54_div_11 : harmonic_mean a b c = 54 / 11 := by
  sorry

end harmonic_mean_is_54_div_11_l299_299007


namespace find_an_l299_299295

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l299_299295


namespace average_speed_round_trip_l299_299380

-- Define average speed calculation for round trip

open Real

theorem average_speed_round_trip (S : ℝ) (hS : S > 0) :
  let t1 := S / 6
  let t2 := S / 4
  let total_distance := 2 * S
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 4.8 :=
  by
    sorry

end average_speed_round_trip_l299_299380


namespace geometric_series_common_ratio_l299_299115

theorem geometric_series_common_ratio (a r : ℝ) (n : ℕ) 
(h1 : a = 7 / 3) 
(h2 : r = 49 / 21)
(h3 : r = 343 / 147):
  r = 7 / 3 :=
by
  sorry

end geometric_series_common_ratio_l299_299115


namespace trains_meet_480_km_away_l299_299494

-- Define the conditions
def bombay_express_speed : ℕ := 60 -- speed in km/h
def rajdhani_express_speed : ℕ := 80 -- speed in km/h
def bombay_express_start_time : ℕ := 1430 -- 14:30 in 24-hour format
def rajdhani_express_start_time : ℕ := 1630 -- 16:30 in 24-hour format

-- Define the function to calculate the meeting point distance
noncomputable def meeting_distance (bombay_speed rajdhani_speed : ℕ) (bombay_start rajdhani_start : ℕ) : ℕ :=
  let t := 6 -- time taken for Rajdhani to catch up in hours, derived from the solution
  rajdhani_speed * t

-- The statement we need to prove:
theorem trains_meet_480_km_away :
  meeting_distance bombay_express_speed rajdhani_express_speed bombay_express_start_time rajdhani_express_start_time = 480 := by
  sorry

end trains_meet_480_km_away_l299_299494


namespace odd_primes_mod_32_l299_299571

-- Define the set of odd primes less than 2^5
def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Define the product of all elements in the list
def N : ℕ := odd_primes_less_than_32.foldl (·*·) 1

-- State the theorem
theorem odd_primes_mod_32 :
  N % 32 = 9 :=
sorry

end odd_primes_mod_32_l299_299571


namespace division_of_fractions_l299_299350

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299350


namespace fraction_division_l299_299345

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299345


namespace man_was_absent_for_days_l299_299824

theorem man_was_absent_for_days
  (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 10 * x - 2 * y = 216) :
  y = 7 :=
by
  sorry

end man_was_absent_for_days_l299_299824


namespace dog_roaming_area_comparison_l299_299305

theorem dog_roaming_area_comparison :
  let r := 10
  let a1 := (1/2) * Real.pi * r^2
  let a2 := (3/4) * Real.pi * r^2 - (1/4) * Real.pi * 6^2 
  a2 > a1 ∧ a2 - a1 = 16 * Real.pi :=
by
  sorry

end dog_roaming_area_comparison_l299_299305


namespace wang_hao_not_last_l299_299391

-- Define the total number of ways to select and arrange 3 players out of 6
def ways_total : ℕ := Nat.factorial 6 / Nat.factorial (6 - 3)

-- Define the number of ways in which Wang Hao is the last player
def ways_wang_last : ℕ := Nat.factorial 5 / Nat.factorial (5 - 2)

-- Proof statement
theorem wang_hao_not_last : ways_total - ways_wang_last = 100 :=
by sorry

end wang_hao_not_last_l299_299391


namespace correct_conclusions_l299_299885

noncomputable def f1 (x : ℝ) : ℝ := 2^x - 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem correct_conclusions :
  ((∀ x, 0 < x ∧ x < 1 → f4 x > f1 x ∧ f4 x > f2 x ∧ f4 x > f3 x) ∧
  (∀ x, x > 1 → f4 x < f1 x ∧ f4 x < f2 x ∧ f4 x < f3 x)) ∧
  (∀ x, ¬(f3 x > f1 x ∧ f3 x > f2 x ∧ f3 x > f4 x) ∧
        ¬(f3 x < f1 x ∧ f3 x < f2 x ∧ f3 x < f4 x)) ∧
  (∃ x, x > 0 ∧ ∀ y, y > x → f1 y > f2 y ∧ f1 y > f3 y ∧ f1 y > f4 y) := by
  sorry

end correct_conclusions_l299_299885


namespace proof_problem_l299_299756

-- Necessary types and noncomputable definitions
noncomputable def a_seq : ℕ → ℕ := sorry
noncomputable def b_seq : ℕ → ℕ := sorry

-- The conditions in the problem are used as assumptions
axiom partition : ∀ (n : ℕ), n > 0 → a_seq n < a_seq (n + 1)
axiom b_def : ∀ (n : ℕ), n > 0 → b_seq n = a_seq n + n

-- The mathematical equivalent proof problem stated
theorem proof_problem (n : ℕ) (hn : n > 0) : a_seq n + b_seq n = a_seq (b_seq n) :=
sorry

end proof_problem_l299_299756


namespace find_third_number_l299_299830

theorem find_third_number (x : ℝ) : 3 + 33 + x + 3.33 = 369.63 → x = 330.30 :=
by
  intros h
  sorry

end find_third_number_l299_299830


namespace lcm_18_24_l299_299811
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l299_299811


namespace dice_sum_to_11_l299_299207

/-- Define the conditions for the outcomes of the dice rolls -/
def valid_outcomes (x : Fin 5 → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ 6) ∧ (x 0 + x 1 + x 2 + x 3 + x 4 = 11)

/-- Prove that there are exactly 205 ways to achieve a sum of 11 with five different colored dice -/
theorem dice_sum_to_11 : 
  (∃ (s : Finset (Fin 5 → ℕ)), (∀ x ∈ s, valid_outcomes x) ∧ s.card = 205) :=
  by
    sorry

end dice_sum_to_11_l299_299207


namespace interest_rate_proof_l299_299504
noncomputable def interest_rate_B (P : ℝ) (rA : ℝ) (t : ℝ) (gain_B : ℝ) : ℝ := 
  (P * rA * t + gain_B) / (P * t)

theorem interest_rate_proof
  (P : ℝ := 3500)
  (rA : ℝ := 0.10)
  (t : ℝ := 3)
  (gain_B : ℝ := 210) :
  interest_rate_B P rA t gain_B = 0.12 :=
sorry

end interest_rate_proof_l299_299504


namespace puppies_left_l299_299233

theorem puppies_left (initial_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : initial_puppies = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_puppies = initial_puppies - given_away) : 
  remaining_puppies = 5 :=
  by
  sorry

end puppies_left_l299_299233


namespace count_integers_n_satisfying_inequality_l299_299900

theorem count_integers_n_satisfying_inequality :
  {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset.card = 7 :=
by sorry

end count_integers_n_satisfying_inequality_l299_299900


namespace lcm_18_24_eq_72_l299_299783

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l299_299783


namespace division_of_fractions_l299_299339

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299339


namespace sum_remainder_l299_299147

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l299_299147


namespace number_of_stacks_l299_299581

theorem number_of_stacks (total_coins stacks coins_per_stack : ℕ) (h1 : coins_per_stack = 3) (h2 : total_coins = 15) (h3 : total_coins = stacks * coins_per_stack) : stacks = 5 :=
by
  sorry

end number_of_stacks_l299_299581


namespace copper_needed_l299_299992

theorem copper_needed (T : ℝ) (lead_percentage : ℝ) (lead_weight : ℝ) (copper_percentage : ℝ) 
  (h_lead_percentage : lead_percentage = 0.25)
  (h_lead_weight : lead_weight = 5)
  (h_copper_percentage : copper_percentage = 0.60)
  (h_total_weight : T = lead_weight / lead_percentage) :
  copper_percentage * T = 12 := 
by
  sorry

end copper_needed_l299_299992


namespace total_money_is_2800_l299_299516

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l299_299516


namespace number_of_fours_is_even_l299_299235

theorem number_of_fours_is_even 
  (x y z : ℕ) 
  (h1 : x + y + z = 80) 
  (h2 : 3 * x + 4 * y + 5 * z = 276) : 
  Even y :=
by
  sorry

end number_of_fours_is_even_l299_299235


namespace radius_inscribed_sphere_quadrilateral_pyramid_l299_299329

noncomputable def radius_of_inscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt 5 - 1) / 4

theorem radius_inscribed_sphere_quadrilateral_pyramid (a : ℝ) :
  r = radius_of_inscribed_sphere a :=
by
  -- problem conditions:
  -- side of the base a
  -- height a
  -- result: r = a * (Real.sqrt 5 - 1) / 4
  sorry

end radius_inscribed_sphere_quadrilateral_pyramid_l299_299329


namespace angle_complement_supplement_l299_299397

theorem angle_complement_supplement (θ : ℝ) (h1 : 90 - θ = (1/3) * (180 - θ)) : θ = 45 :=
by
  sorry

end angle_complement_supplement_l299_299397


namespace line_through_point_parallel_to_given_l299_299321

open Real

theorem line_through_point_parallel_to_given (x y : ℝ) :
  (∃ (m : ℝ), (y - 0 = m * (x - 1)) ∧ x - 2*y - 1 = 0) ↔
  (x = 1 ∧ y = 0 ∧ ∃ l, x - 2*y - l = 0) :=
by sorry

end line_through_point_parallel_to_given_l299_299321


namespace computer_table_cost_price_l299_299622

theorem computer_table_cost_price (CP SP : ℝ) (h1 : SP = CP * (124 / 100)) (h2 : SP = 8091) :
  CP = 6525 :=
by
  sorry

end computer_table_cost_price_l299_299622


namespace frequency_of_middle_rectangle_l299_299709

theorem frequency_of_middle_rectangle
    (n : ℕ)
    (A : ℕ)
    (h1 : A + (n - 1) * A = 160) :
    A = 32 :=
by
  sorry

end frequency_of_middle_rectangle_l299_299709


namespace triangles_hyperbola_parallel_l299_299845

variable (a b c a1 b1 c1 : ℝ)

-- Defining the property that all vertices lie on the hyperbola y = 1/x
def on_hyperbola (x : ℝ) (y : ℝ) : Prop := y = 1 / x

-- Defining the parallelism condition for line segments
def parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem triangles_hyperbola_parallel
  (H1A : on_hyperbola a (1 / a))
  (H1B : on_hyperbola b (1 / b))
  (H1C : on_hyperbola c (1 / c))
  (H2A : on_hyperbola a1 (1 / a1))
  (H2B : on_hyperbola b1 (1 / b1))
  (H2C : on_hyperbola c1 (1 / c1))
  (H_AB_parallel_A1B1 : parallel ((b - a) / (a * b * (a - b))) ((b1 - a1) / (a1 * b1 * (a1 - b1))))
  (H_BC_parallel_B1C1 : parallel ((c - b) / (b * c * (b - c))) ((c1 - b1) / (b1 * c1 * (b1 - c1)))) :
  parallel ((c1 - a) / (a * c1 * (a - c1))) ((c - a1) / (a1 * c * (a1 - c))) :=
sorry

end triangles_hyperbola_parallel_l299_299845


namespace max_value_g_l299_299468

def g : ℕ → ℕ
| n => if n < 7 then n + 8 else g (n - 3)

theorem max_value_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 14 := by
  sorry

end max_value_g_l299_299468


namespace total_students_l299_299490

theorem total_students (students_in_front : ℕ) (position_from_back : ℕ) : 
  students_in_front = 6 ∧ position_from_back = 5 → 
  students_in_front + 1 + (position_from_back - 1) = 11 :=
by
  sorry

end total_students_l299_299490


namespace arithmetic_sequence_k_value_l299_299435

theorem arithmetic_sequence_k_value (a_1 d : ℕ) (h1 : a_1 = 1) (h2 : d = 2) (k : ℕ) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) (h_condition : S (k + 2) - S k = 24) : k = 5 :=
by {
  sorry
}

end arithmetic_sequence_k_value_l299_299435


namespace solve_quadratic_inequality_l299_299457

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end solve_quadratic_inequality_l299_299457


namespace negation_of_p_l299_299304

def proposition_p (n : ℕ) : Prop := 3^n ≥ n + 1

theorem negation_of_p : (∃ n0 : ℕ, 3^n0 < n0^2 + 1) :=
  by sorry

end negation_of_p_l299_299304


namespace karsyn_total_payment_l299_299159

def initial_price : ℝ := 600
def discount_rate : ℝ := 0.20
def phone_case_cost : ℝ := 25
def screen_protector_cost : ℝ := 15
def store_discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.035

noncomputable def total_payment : ℝ :=
  let discounted_price := discount_rate * initial_price
  let total_cost := discounted_price + phone_case_cost + screen_protector_cost
  let store_discount := store_discount_rate * total_cost
  let discounted_total := total_cost - store_discount
  let tax := sales_tax_rate * discounted_total
  discounted_total + tax

theorem karsyn_total_payment : total_payment = 157.32 := by
  sorry

end karsyn_total_payment_l299_299159


namespace lcm_18_24_l299_299798

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299798


namespace count_integers_satisfying_inequality_l299_299907

theorem count_integers_satisfying_inequality : 
  ∃ (s : Finset ℤ), (∀ n ∈ s, (n - 3) * (n + 5) < 0) ∧ s.card = 7 :=
begin
  sorry
end

end count_integers_satisfying_inequality_l299_299907


namespace roots_of_quadratic_l299_299149

theorem roots_of_quadratic (a b : ℝ) (h : a ≠ 0) (h1 : a + b = 0) :
  ∀ x, (a * x^2 + b * x = 0) → (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_quadratic_l299_299149


namespace simplify_and_evaluate_expression_l299_299590

-- Definitions of the variables and their values
def x : ℤ := -2
def y : ℚ := 1 / 2

-- Theorem statement
theorem simplify_and_evaluate_expression : 
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 
  (1 : ℚ) / 2 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_expression_l299_299590


namespace negation_of_universal_prop_l299_299683

variable (a : ℝ)

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, 0 < x → Real.log x = a) ↔ (∃ x : ℝ, 0 < x ∧ Real.log x ≠ a) :=
by
  sorry

end negation_of_universal_prop_l299_299683


namespace lcm_18_24_eq_72_l299_299772

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299772


namespace objects_meet_probability_l299_299309

open Classical
open Finset
open Probability

noncomputable theory

def count_paths (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose n k

def total_paths (n : ℕ) : ℕ :=
  2 ^ n

def meeting_probability (n : ℕ) : ℝ :=
  (finset.sum (range (7)) (λ i, (count_paths n i : ℝ))) / (total_paths n * total_paths n)

theorem objects_meet_probability :
  meeting_probability 9 = 0.162 :=
by
  have : meeting_probability 9 =
    (Nat.choose 9 0 * Nat.choose 9 0 + Nat.choose 9 1 * Nat.choose 9 1 + Nat.choose 9 2 * Nat.choose 9 2 +
    Nat.choose 9 3 * Nat.choose 9 3 + Nat.choose 9 4 * Nat.choose 9 4 + Nat.choose 9 5 * Nat.choose 9 5 + 
    Nat.choose 9 6 * Nat.choose 9 6) / (2 ^ 18) := rfl
  calc
    meeting_probability 9 = (1 + 81 + 1296 + 6561 + 11664 + 9025 + 2916) / 262144 : by simp [this]
    ... = 42544 / 262144 : by norm_num
    ... = 0.162 : by norm_num

end objects_meet_probability_l299_299309


namespace final_price_calculation_l299_299237

theorem final_price_calculation 
  (ticket_price : ℝ)
  (initial_discount : ℝ)
  (additional_discount : ℝ)
  (sales_tax : ℝ)
  (final_price : ℝ) 
  (h1 : ticket_price = 200) 
  (h2 : initial_discount = 0.25) 
  (h3 : additional_discount = 0.15) 
  (h4 : sales_tax = 0.07)
  (h5 : final_price = (ticket_price * (1 - initial_discount)) * (1 - additional_discount) * (1 + sales_tax)):
  final_price = 136.43 :=
by
  sorry

end final_price_calculation_l299_299237


namespace imo_inequality_l299_299439

variable {a b c : ℝ}

theorem imo_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : (a + b) * (b + c) * (c + a) = 1) :
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ (1 / 2) := 
sorry

end imo_inequality_l299_299439


namespace g_min_value_l299_299678

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l299_299678


namespace total_votes_is_120_l299_299016

-- Define the conditions
def Fiona_votes : ℕ := 48
def fraction_of_votes : ℚ := 2 / 5

-- The proof goal
theorem total_votes_is_120 (V : ℕ) (h : Fiona_votes = fraction_of_votes * V) : V = 120 :=
by
  sorry

end total_votes_is_120_l299_299016


namespace lcm_18_24_l299_299797

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l299_299797


namespace selena_left_with_l299_299587

/-- Selena got a tip of $99 and spent money on various foods whose individual costs are provided. 
Prove that she will be left with $38. -/
theorem selena_left_with : 
  let tip := 99
  let steak_cost := 24
  let num_steaks := 2
  let burger_cost := 3.5
  let num_burgers := 2
  let ice_cream_cost := 2
  let num_ice_cream := 3
  let total_spent := (steak_cost * num_steaks) + (burger_cost * num_burgers) + (ice_cream_cost * num_ice_cream)
  tip - total_spent = 38 := 
by 
  sorry

end selena_left_with_l299_299587


namespace lcm_18_24_l299_299808

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299808


namespace lcm_18_24_l299_299774

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l299_299774


namespace expenditure_representation_correct_l299_299014

-- Define the representation of income
def income_representation (income : ℝ) : ℝ :=
  income

-- Define the representation of expenditure
def expenditure_representation (expenditure : ℝ) : ℝ :=
  -expenditure

-- Condition: an income of 10.5 yuan is represented as +10.5 yuan.
-- We need to prove: an expenditure of 6 yuan is represented as -6 yuan.
theorem expenditure_representation_correct (h : income_representation 10.5 = 10.5) : 
  expenditure_representation 6 = -6 :=
by
  sorry

end expenditure_representation_correct_l299_299014


namespace find_an_l299_299296

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l299_299296


namespace not_all_polynomials_sum_of_cubes_l299_299649

theorem not_all_polynomials_sum_of_cubes :
  ¬ ∀ P : Polynomial ℤ, ∃ Q : Polynomial ℤ, P = Q^3 + Q^3 + Q^3 :=
by
  sorry

end not_all_polynomials_sum_of_cubes_l299_299649


namespace product_of_consecutive_numbers_with_25_is_perfect_square_l299_299568

theorem product_of_consecutive_numbers_with_25_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n * (n + 1)) + 25 = k^2 := 
by
  -- Proof body omitted
  sorry

end product_of_consecutive_numbers_with_25_is_perfect_square_l299_299568


namespace range_of_a_l299_299707

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ Real.exp 1) ↔ ∀ x : ℝ, 0 < x → a * Real.log (a * x) ≤ Real.exp x := 
by 
  sorry

end range_of_a_l299_299707


namespace range_of_a_l299_299440

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + a > 0

def proposition_q (a : ℝ) : Prop :=
  a - 1 > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l299_299440


namespace price_increase_decrease_l299_299379

theorem price_increase_decrease (P : ℝ) (h : 0.84 * P = P * (1 - (x / 100)^2)) : x = 40 := by
  sorry

end price_increase_decrease_l299_299379


namespace determine_F_value_l299_299092

theorem determine_F_value (D E F : ℕ) (h1 : (9 + 6 + D + 1 + E + 8 + 2) % 3 = 0) (h2 : (5 + 4 + E + D + 2 + 1 + F) % 3 = 0) : 
  F = 2 := 
by
  sorry

end determine_F_value_l299_299092


namespace annual_increase_in_living_space_l299_299514

-- Definitions based on conditions
def population_2000 : ℕ := 200000
def living_space_2000_per_person : ℝ := 8
def target_living_space_2004_per_person : ℝ := 10
def annual_growth_rate : ℝ := 0.01
def years : ℕ := 4

-- Goal stated as a theorem
theorem annual_increase_in_living_space :
  let final_population := population_2000 * (1 + annual_growth_rate)^years
  let total_living_space_2004 := target_living_space_2004_per_person * final_population
  let initial_living_space := living_space_2000_per_person * population_2000
  let total_additional_space := total_living_space_2004 - initial_living_space
  let average_annual_increase := total_additional_space / years
  average_annual_increase = 120500.0 :=
sorry

end annual_increase_in_living_space_l299_299514


namespace find_dividing_line_l299_299223

/--
A line passing through point P(1,1) divides the circular region \{(x, y) \mid x^2 + y^2 \leq 4\} into two parts,
making the difference in area between these two parts the largest. Prove that the equation of this line is x + y - 2 = 0.
-/
theorem find_dividing_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ (A B C : ℝ), A * 1 + B * 1 + C = 0 ∧
                 (∀ x y, x^2 + y^2 ≤ 4 → A * x + B * y + C = 0 → (x + y - 2) = 0) :=
sorry

end find_dividing_line_l299_299223


namespace divisible_by_91_l299_299751

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 202020
  | _ => -- Define the sequence here, ensuring it constructs the number properly with inserted '2's
    sorry -- this might be a more complex function to define

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n :=
  sorry

end divisible_by_91_l299_299751


namespace original_planned_production_l299_299369

theorem original_planned_production (x : ℝ) (hx1 : x ≠ 0) (hx2 : 210 / x - 210 / (1.5 * x) = 5) : x = 14 :=
by sorry

end original_planned_production_l299_299369


namespace lcm_18_24_l299_299806

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l299_299806


namespace union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l299_299123

open Set

variables (U : Set ℝ) (A B : Set ℝ) (a : ℝ)

def A_def : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def comp_U_A : Set ℝ := { x | x < 1 ∨ x > 4 }

theorem union_A_B_at_a_3 (h : a = 3) :
  A_def ∪ B_def 3 = { x | 1 ≤ x ∧ x ≤ 5 } :=
sorry

theorem inter_B_compl_A_at_a_3 (h : a = 3) :
  B_def 3 ∩ comp_U_A = { x | 4 < x ∧ x ≤ 5 } :=
sorry

theorem B_subset_A_imp_a_range (h : B_def a ⊆ A_def) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l299_299123


namespace jennifer_book_spending_l299_299028

variable (initial_total : ℕ)
variable (spent_sandwich : ℚ)
variable (spent_museum : ℚ)
variable (money_left : ℕ)

theorem jennifer_book_spending :
  initial_total = 90 → 
  spent_sandwich = 1/5 * 90 → 
  spent_museum = 1/6 * 90 → 
  money_left = 12 →
  (initial_total - money_left - (spent_sandwich + spent_museum)) / initial_total = 1/2 :=
by
  intros h_initial_total h_spent_sandwich h_spent_museum h_money_left
  sorry

end jennifer_book_spending_l299_299028


namespace ratio_nephews_l299_299231

variable (N : ℕ) -- The number of nephews Alden has now.
variable (Alden_had_50 : Prop := 50 = 50)
variable (Vihaan_more_60 : Prop := Vihaan = N + 60)
variable (Together_260 : Prop := N + (N + 60) = 260)

theorem ratio_nephews (N : ℕ) 
  (H1 : Alden_had_50)
  (H2 : Vihaan_more_60)
  (H3 : Together_260) :
  50 / N = 1 / 2 :=
by
  sorry

end ratio_nephews_l299_299231


namespace exp_values_l299_299146

variable {a x y : ℝ}

theorem exp_values (hx : a^x = 3) (hy : a^y = 2) :
  a^(x - y) = 3 / 2 ∧ a^(2 * x + y) = 18 :=
by
  sorry

end exp_values_l299_299146


namespace largest_negative_integer_solution_l299_299676

theorem largest_negative_integer_solution :
  ∃ x : ℤ, x < 0 ∧ 50 * x + 14 % 24 = 10 % 24 ∧ ∀ y : ℤ, (y < 0 ∧ y % 12 = 10 % 12 → y ≤ x) :=
by
  sorry

end largest_negative_integer_solution_l299_299676


namespace employee_price_l299_299363

theorem employee_price (wholesale_cost retail_markup employee_discount : ℝ) 
    (h₁ : wholesale_cost = 200) 
    (h₂ : retail_markup = 0.20) 
    (h₃ : employee_discount = 0.25) : 
    (wholesale_cost * (1 + retail_markup)) * (1 - employee_discount) = 180 := 
by
  sorry

end employee_price_l299_299363


namespace height_of_smaller_cone_removed_l299_299096

noncomputable def frustum_area_lower_base : ℝ := 196 * Real.pi
noncomputable def frustum_area_upper_base : ℝ := 16 * Real.pi
def frustum_height : ℝ := 30

theorem height_of_smaller_cone_removed (r1 r2 H : ℝ)
  (h1 : r1 = Real.sqrt (frustum_area_lower_base / Real.pi))
  (h2 : r2 = Real.sqrt (frustum_area_upper_base / Real.pi))
  (h3 : r2 / r1 = 2 / 7)
  (h4 : frustum_height = (5 / 7) * H) :
  H - frustum_height = 12 :=
by 
  sorry

end height_of_smaller_cone_removed_l299_299096


namespace smallest_period_of_f_l299_299884

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 1

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

end smallest_period_of_f_l299_299884


namespace ratio_of_P_Q_l299_299056

theorem ratio_of_P_Q (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4 * x) = (x^2 + x + 15) / (x^3 + x^2 - 20 * x)) :
  Q / P = -45 / 2 :=
by
  sorry

end ratio_of_P_Q_l299_299056


namespace problem_statement_l299_299572

noncomputable def AB2_AC2_BC2_eq_4 (l m n k : ℝ) : Prop :=
  let D := (l+k, 0, 0)
  let E := (0, m+k, 0)
  let F := (0, 0, n+k)
  let AB_sq := 4 * (n+k)^2
  let AC_sq := 4 * (m+k)^2
  let BC_sq := 4 * (l+k)^2
  AB_sq + AC_sq + BC_sq = 4 * ((l+k)^2 + (m+k)^2 + (n+k)^2)

theorem problem_statement (l m n k : ℝ) : 
  AB2_AC2_BC2_eq_4 l m n k :=
by
  sorry

end problem_statement_l299_299572


namespace seconds_in_12_5_minutes_l299_299266

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l299_299266


namespace range_of_m_l299_299015

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x^2 + 2 * x - m - 1 = 0) → m ≥ -2 := 
by
  sorry

end range_of_m_l299_299015


namespace bacon_calories_percentage_l299_299860

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l299_299860


namespace correlation_snoring_heart_disease_l299_299019

theorem correlation_snoring_heart_disease {patients : ℕ} (correlation_confidence : ℝ) :
  (correlation_confidence > 99 / 100) ∧ 
  (patients = 100) →
  ∃ (snoring_patients : ℕ → Prop), 
    (∀ p, snoring_patients p → p < patients) ∧ 
    (∃ p, ¬ snoring_patients p) :=
by
  sorry

end correlation_snoring_heart_disease_l299_299019


namespace arithmetic_sequence_sum_l299_299894

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (h_arith : arithmetic_sequence a)
    (h_a2 : a 2 = 3)
    (h_a1_a6 : a 1 + a 6 = 12) : a 7 + a 8 + a 9 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l299_299894


namespace zero_and_one_positions_l299_299372

theorem zero_and_one_positions (a : ℝ) :
    (0 = (a + (-a)) / 2) ∧ (1 = ((a + (-a)) / 2 + 1)) :=
by
  sorry

end zero_and_one_positions_l299_299372


namespace sum_and_product_of_roots_l299_299876

theorem sum_and_product_of_roots (a b : ℝ) (h1 : a * a * a - 4 * a * a - a + 4 = 0)
  (h2 : b * b * b - 4 * b * b - b + 4 = 0) :
  a + b + a * b = -1 :=
sorry

end sum_and_product_of_roots_l299_299876


namespace smallest_n_lil_wayne_rain_l299_299038

noncomputable def probability_rain (n : ℕ) : ℝ := 
  1 / 2 - 1 / 2^(n + 1)

theorem smallest_n_lil_wayne_rain :
  ∃ n : ℕ, probability_rain n > 0.499 ∧ (∀ m : ℕ, m < n → probability_rain m ≤ 0.499) ∧ n = 9 := 
by
  sorry

end smallest_n_lil_wayne_rain_l299_299038


namespace integer_pairs_satisfying_equation_and_nonnegative_product_l299_299533

theorem integer_pairs_satisfying_equation_and_nonnegative_product :
  ∃ (pairs : List (ℤ × ℤ)), 
    (∀ p ∈ pairs, p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3) ∧ 
    pairs.length = 35 :=
by sorry

end integer_pairs_satisfying_equation_and_nonnegative_product_l299_299533


namespace proof_by_contradiction_example_l299_299618

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l299_299618


namespace speed_of_stream_l299_299623

theorem speed_of_stream
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time_eq_upstream_time : downstream_distance / (boat_speed + v) = upstream_distance / (boat_speed - v)) :
  v = 8 :=
by
  let v := 8
  sorry

end speed_of_stream_l299_299623


namespace original_circle_area_l299_299388

theorem original_circle_area (A : ℝ) (h1 : ∃ sector_area : ℝ, sector_area = 5) (h2 : A / 64 = 5) : A = 320 := 
by sorry

end original_circle_area_l299_299388


namespace john_loses_probability_eq_3_over_5_l299_299968

-- Definitions used directly from the conditions in a)
def probability_win := 2 / 5
def probability_lose := 1 - probability_win

-- The theorem statement
theorem john_loses_probability_eq_3_over_5 : 
  probability_lose = 3 / 5 := 
by
  sorry -- proof is to be filled in later

end john_loses_probability_eq_3_over_5_l299_299968


namespace simplify_fraction_l299_299739

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by
  sorry

end simplify_fraction_l299_299739


namespace earphone_cost_correct_l299_299373

-- Given conditions
def mean_expenditure : ℕ := 500

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thu : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300

def pen_cost : ℕ := 30
def notebook_cost : ℕ := 50

-- Goal: cost of the earphone
def total_expenditure_week : ℕ := 7 * mean_expenditure
def expenditure_6days : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thu + expenditure_sat + expenditure_sun
def expenditure_fri : ℕ := total_expenditure_week - expenditure_6days
def expenditure_fri_items : ℕ := pen_cost + notebook_cost
def earphone_cost : ℕ := expenditure_fri - expenditure_fri_items

theorem earphone_cost_correct :
  earphone_cost = 620 :=
by
  sorry

end earphone_cost_correct_l299_299373


namespace find_k_l299_299365

def equation (k : ℝ) (x : ℝ) : Prop := 2 * x^2 + 3 * x - k = 0

theorem find_k (k : ℝ) (h : equation k 7) : k = 119 :=
by
  sorry

end find_k_l299_299365


namespace multiplication_modulo_l299_299319

theorem multiplication_modulo :
  ∃ n : ℕ, (253 * 649 ≡ n [MOD 100]) ∧ (0 ≤ n) ∧ (n < 100) ∧ (n = 97) := 
by
  sorry

end multiplication_modulo_l299_299319


namespace solution_interval_l299_299441

-- Define the differentiable function f over the interval (-∞, 0)
variable {f : ℝ → ℝ}
variable (hf : ∀ x < 0, HasDerivAt f (f' x) x)
variable (hx_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2)

-- Proof statement to show the solution interval
theorem solution_interval :
  {x : ℝ | (x + 2018)^2 * f (x + 2018) - 4 * f (-2) > 0} = {x | x < -2020} :=
sorry

end solution_interval_l299_299441


namespace fraction_division_l299_299342

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299342


namespace bob_more_than_alice_l299_299232

-- Definitions for conditions
def initial_investment_alice : ℕ := 10000
def initial_investment_bob : ℕ := 10000
def multiple_alice : ℕ := 3
def multiple_bob : ℕ := 7

-- Derived conditions based on the investment multiples
def final_amount_alice : ℕ := initial_investment_alice * multiple_alice
def final_amount_bob : ℕ := initial_investment_bob * multiple_bob

-- Statement of the problem
theorem bob_more_than_alice : final_amount_bob - final_amount_alice = 40000 :=
by
  -- Proof to be filled in
  sorry

end bob_more_than_alice_l299_299232


namespace prob_no_adjacent_same_roll_l299_299407

-- Definition of the problem conditions
def num_people : ℕ := 5
def num_sides_die : ℕ := 6

-- Probability that no two adjacent people roll the same number
def prob_no_adj_same : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6)

-- Proof statement
theorem prob_no_adjacent_same_roll : prob_no_adj_same = 625 / 1296 := by
  sorry

end prob_no_adjacent_same_roll_l299_299407


namespace find_original_price_l299_299227

-- Definitions for the conditions mentioned in the problem
variables {P : ℝ} -- Original price per gallon in dollars

-- Proof statement assuming the given conditions
theorem find_original_price 
  (h1 : ∃ P : ℝ, P > 0) -- There exists a positive price per gallon in dollars
  (h2 : (250 / (0.9 * P)) = (250 / P + 5)) -- After a 10% price reduction, 5 gallons more can be bought for $250
  : P = 25 / 4.5 := -- The solution states the original price per gallon is approximately $5.56
by
  sorry -- Proof omitted

end find_original_price_l299_299227


namespace division_of_fractions_l299_299338

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299338


namespace largest_angle_in_ratio_3_4_5_l299_299185

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l299_299185


namespace pentagon_segment_condition_l299_299506

-- Define the problem context and hypothesis
variable (a b c d e : ℝ)

theorem pentagon_segment_condition 
  (h₁ : a + b + c + d + e = 3)
  (h₂ : a ≤ b)
  (h₃ : b ≤ c)
  (h₄ : c ≤ d)
  (h₅ : d ≤ e) : 
  a < 3 / 2 ∧ b < 3 / 2 ∧ c < 3 / 2 ∧ d < 3 / 2 ∧ e < 3 / 2 := 
sorry

end pentagon_segment_condition_l299_299506


namespace seating_arrangements_zero_l299_299639

def valid_seating_arrangements (fixed_person : String) (persons : List String) : Nat :=
  let arrangements := 
    persons.permutations.filter (λ perm => 
      -- Alice refuses to sit next to Bob or Carla
      not ((perm[0] = "Alice" ∧ (perm[1] = "Bob" ∨ perm[1] = "Carla")) 
           ∨ (perm[1] = "Alice" ∧ (perm[0] = "Bob" ∨ perm[0] = "Carla")))
      -- Derek refuses to sit next to Eric
      ∧ not ((perm[2] = "Derek" ∧ perm[3] = "Eric") 
             ∨ (perm[3] = "Derek" ∧ perm[2] = "Eric"))
      -- Carla refuses to sit next to Derek
      ∧ not ((perm[1] = "Carla" ∧ perm[2] = "Derek") 
             ∨ (perm[2] = "Carla" ∧ perm[1] = "Derek")))
  arrangements.length

theorem seating_arrangements_zero :
  valid_seating_arrangements "Alice" ["Bob", "Carla", "Derek", "Eric"] = 0 :=
by {
  sorry
}

end seating_arrangements_zero_l299_299639


namespace find_n_from_exponent_equation_l299_299699

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l299_299699


namespace reinforcement_size_l299_299097

theorem reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) (days_remaining : ℕ) (reinforcement : ℕ) : 
  initial_men = 150 → initial_days = 31 → days_before_reinforcement = 16 → days_remaining = 5 → (150 * 15) = (150 + reinforcement) * 5 → reinforcement = 300 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end reinforcement_size_l299_299097


namespace solve_arcsin_eq_l299_299743

open Real

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (sin x) = (3 * x) / 4

theorem solve_arcsin_eq(x : ℝ) (h : problem_statement x) (h_range: - (2 * π) / 3 ≤ x ∧ x ≤ (2 * π) / 3) : x = 0 :=
sorry

end solve_arcsin_eq_l299_299743


namespace training_weeks_l299_299158

variable (adoption_fee training_per_week cert_cost insurance_coverage out_of_pocket : ℕ)
variable (x : ℕ)

def adoption_fee_value : ℕ := 150
def training_per_week_cost : ℕ := 250
def certification_cost_value : ℕ := 3000
def insurance_coverage_percentage : ℕ := 90
def total_out_of_pocket : ℕ := 3450

theorem training_weeks :
  adoption_fee = adoption_fee_value →
  training_per_week = training_per_week_cost →
  cert_cost = certification_cost_value →
  insurance_coverage = insurance_coverage_percentage →
  out_of_pocket = total_out_of_pocket →
  (out_of_pocket = adoption_fee + training_per_week * x + (cert_cost * (100 - insurance_coverage)) / 100) →
  x = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end training_weeks_l299_299158


namespace number_of_distinct_intersection_points_l299_299241

theorem number_of_distinct_intersection_points :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
  let line := {p : ℝ × ℝ | p.1 = 4}
  let intersection_points := circle ∩ line
  ∃! p : ℝ × ℝ, p ∈ intersection_points :=
by
  sorry

end number_of_distinct_intersection_points_l299_299241


namespace barbara_typing_time_l299_299853

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end barbara_typing_time_l299_299853


namespace division_of_fractions_l299_299341

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l299_299341


namespace increase_75_by_150_percent_l299_299481

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l299_299481


namespace total_amount_due_is_correct_l299_299643

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

end total_amount_due_is_correct_l299_299643


namespace smallest_natural_number_l299_299567

theorem smallest_natural_number :
  ∃ N : ℕ, ∃ f : ℕ → ℕ → ℕ, 
  f (f (f 9 8 - f 7 6) 5 + 4 - f 3 2) 1 = N ∧
  N = 1 := 
by sorry

end smallest_natural_number_l299_299567


namespace no_sum_of_squares_of_rationals_l299_299589

theorem no_sum_of_squares_of_rationals (p q r s : ℕ) (hq : q ≠ 0) (hs : s ≠ 0)
    (hpq : Nat.gcd p q = 1) (hrs : Nat.gcd r s = 1) :
    (↑p / q : ℚ) ^ 2 + (↑r / s : ℚ) ^ 2 ≠ 168 := by 
    sorry

end no_sum_of_squares_of_rationals_l299_299589


namespace Nancy_shelved_biographies_l299_299040

def NancyBooks.shelved_books_from_top : Nat := 12 + 8 + 4 -- history + romance + poetry
def NancyBooks.total_books_on_cart : Nat := 46
def NancyBooks.bottom_books_after_top_shelved : Nat := 46 - 24
def NancyBooks.mystery_books_on_bottom : Nat := NancyBooks.bottom_books_after_top_shelved / 2
def NancyBooks.western_novels_on_bottom : Nat := 5
def NancyBooks.biographies : Nat := NancyBooks.bottom_books_after_top_shelved - NancyBooks.mystery_books_on_bottom - NancyBooks.western_novels_on_bottom

theorem Nancy_shelved_biographies : NancyBooks.biographies = 6 := by
  sorry

end Nancy_shelved_biographies_l299_299040


namespace probability_difference_l299_299368

-- Definitions for probabilities
def P_plane : ℚ := 7 / 10
def P_train : ℚ := 3 / 10
def P_on_time_plane : ℚ := 8 / 10
def P_on_time_train : ℚ := 9 / 10

-- Events definitions
def P_arrive_on_time : ℚ := (7 / 10) * (8 / 10) + (3 / 10) * (9 / 10)
def P_plane_and_on_time : ℚ := (7 / 10) * (8 / 10)
def P_train_and_on_time : ℚ := (3 / 10) * (9 / 10)
def P_conditional_plane_given_on_time : ℚ := P_plane_and_on_time / P_arrive_on_time
def P_conditional_train_given_on_time : ℚ := P_train_and_on_time / P_arrive_on_time

theorem probability_difference :
  P_conditional_plane_given_on_time - P_conditional_train_given_on_time = 29 / 83 :=
by sorry

end probability_difference_l299_299368


namespace maximum_value_of_a_l299_299548

theorem maximum_value_of_a
  (a b c d : ℝ)
  (h1 : b + c + d = 3 - a)
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) :
  a ≤ 2 := by
  sorry

end maximum_value_of_a_l299_299548


namespace sachin_is_younger_than_rahul_by_18_years_l299_299312

-- Definitions based on conditions
def sachin_age : ℕ := 63
def ratio_of_ages : ℚ := 7 / 9

-- Assertion that based on the given conditions, Sachin is 18 years younger than Rahul
theorem sachin_is_younger_than_rahul_by_18_years (R : ℕ) (h1 : (sachin_age : ℚ) / R = ratio_of_ages) : R - sachin_age = 18 :=
by
  sorry

end sachin_is_younger_than_rahul_by_18_years_l299_299312


namespace toothpicks_150th_stage_l299_299960

-- Define the arithmetic sequence parameters
def first_term : ℕ := 4
def common_difference : ℕ := 4

-- Define the term number we are interested in
def stage_number : ℕ := 150

-- The total number of toothpicks in the nth stage of an arithmetic sequence
def num_toothpicks (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

-- Theorem stating the number of toothpicks in the 150th stage
theorem toothpicks_150th_stage : num_toothpicks first_term common_difference stage_number = 600 :=
by
  sorry

end toothpicks_150th_stage_l299_299960


namespace sum_ages_is_13_l299_299293

-- Define the variables for the ages
variables (a b c : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  a * b * c = 72 ∧ a < b ∧ c < b

-- State the theorem to be proved
theorem sum_ages_is_13 (h : conditions a b c) : a + b + c = 13 :=
sorry

end sum_ages_is_13_l299_299293


namespace parallelogram_not_symmetrical_l299_299085

-- Define the shapes
inductive Shape
| circle
| rectangle
| isosceles_trapezoid
| parallelogram

-- Define what it means for a shape to be symmetrical
def is_symmetrical (s: Shape) : Prop :=
  match s with
  | Shape.circle => True
  | Shape.rectangle => True
  | Shape.isosceles_trapezoid => True
  | Shape.parallelogram => False -- The condition we're interested in proving

-- The main theorem stating the problem
theorem parallelogram_not_symmetrical : is_symmetrical Shape.parallelogram = False :=
by
  sorry

end parallelogram_not_symmetrical_l299_299085


namespace problem_1_problem_2_l299_299728

-- First Problem
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x : ℝ, f x - 2 * |x - 7| ≤ 0) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → a ≥ -12 :=
by
  intros
  sorry

-- Second Problem
theorem problem_2 (f : ℝ → ℝ) (a m : ℝ) (h1 : a = 1) 
  (h2 : ∀ x : ℝ, f x + |x + 7| ≥ m) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → m ≤ 7 :=
by
  intros
  sorry

end problem_1_problem_2_l299_299728


namespace seconds_in_12_5_minutes_l299_299265

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l299_299265


namespace arc_length_120_degrees_l299_299003

theorem arc_length_120_degrees (π : ℝ) : 
  let R := π
  let n := 120
  (n * π * R) / 180 = (2 * π^2) / 3 := 
by
  let R := π
  let n := 120
  sorry

end arc_length_120_degrees_l299_299003


namespace joe_new_average_l299_299715

def joe_tests_average (a b c d : ℝ) : Prop :=
  ((a + b + c + d) / 4 = 35) ∧ (min a (min b (min c d)) = 20)

theorem joe_new_average (a b c d : ℝ) (h : joe_tests_average a b c d) :
  ((a + b + c + d - min a (min b (min c d))) / 3 = 40) :=
sorry

end joe_new_average_l299_299715


namespace largest_angle_is_75_l299_299193

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l299_299193


namespace terry_daily_driving_time_l299_299461

theorem terry_daily_driving_time 
  (d1: ℝ) (s1: ℝ)
  (d2: ℝ) (s2: ℝ)
  (d3: ℝ) (s3: ℝ)
  (h1 : d1 = 15) (h2 : s1 = 30)
  (h3 : d2 = 35) (h4 : s2 = 50)
  (h5 : d3 = 10) (h6 : s3 = 40) : 
  2 * ((d1 / s1) + (d2 / s2) + (d3 / s3)) = 2.9 := 
by
  sorry

end terry_daily_driving_time_l299_299461


namespace wolf_and_nobel_prize_laureates_l299_299627

-- Definitions from the conditions
def num_total_scientists : ℕ := 50
def num_wolf_prize_laureates : ℕ := 31
def num_nobel_prize_laureates : ℕ := 29
def num_no_wolf_prize_and_yes_nobel := 3 -- N_W = N_W'
def num_without_wolf_or_nobel : ℕ := num_total_scientists - num_wolf_prize_laureates - 11 -- Derived from N_W' 

-- The statement to be proved
theorem wolf_and_nobel_prize_laureates :
  ∃ W_N, W_N = num_nobel_prize_laureates - (19 - 3) ∧ W_N = 18 :=
  by
    sorry

end wolf_and_nobel_prize_laureates_l299_299627


namespace travel_time_l299_299993

theorem travel_time (v : ℝ) (d : ℝ) (t : ℝ) (hv : v = 65) (hd : d = 195) : t = 3 :=
by
  sorry

end travel_time_l299_299993


namespace number_of_zeros_f_l299_299058

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2 * x + 5

theorem number_of_zeros_f : 
  (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ 0 < a ∧ 0 < b ∧ a ≠ b) ∧ ∀ c, f c = 0 → c = a ∨ c = b :=
by
  sorry

end number_of_zeros_f_l299_299058


namespace correct_average_weight_l299_299748

-- Definitions
def initial_average_weight : ℝ := 58.4
def number_of_boys : ℕ := 20
def misread_weight_initial : ℝ := 56
def misread_weight_correct : ℝ := 68

-- Correct average weight
theorem correct_average_weight : 
  let initial_total_weight := initial_average_weight * (number_of_boys : ℝ)
  let difference := misread_weight_correct - misread_weight_initial
  let correct_total_weight := initial_total_weight + difference
  let correct_average_weight := correct_total_weight / (number_of_boys : ℝ)
  correct_average_weight = 59 :=
by
  -- Insert the proof steps if needed
  sorry

end correct_average_weight_l299_299748


namespace dad_caught_more_trouts_l299_299648

-- Definitions based on conditions
def caleb_trouts : ℕ := 2
def dad_trouts : ℕ := 3 * caleb_trouts

-- The proof problem: proving dad caught 4 more trouts than Caleb
theorem dad_caught_more_trouts : dad_trouts = caleb_trouts + 4 :=
by
  sorry

end dad_caught_more_trouts_l299_299648


namespace sophia_book_problem_l299_299180

/-
Prove that the total length of the book P is 270 pages, and verify the number of pages read by Sophia
on the 4th and 5th days (50 and 40 pages respectively), given the following conditions:
1. Sophia finished 2/3 of the book in the first three days.
2. She calculated that she finished 90 more pages than she has yet to read.
3. She plans to finish the entire book within 5 days.
4. She will read 10 fewer pages each day from the 4th day until she finishes.
-/

theorem sophia_book_problem
  (P : ℕ)
  (h1 : (2/3 : ℝ) * P = P - (90 + (1/3 : ℝ) * P))
  (h2 : P = 3 * 90)
  (remaining_pages : ℕ := P / 3)
  (h3 : remaining_pages = 90)
  (pages_day4 : ℕ)
  (pages_day5 : ℕ := pages_day4 - 10)
  (h4 : pages_day4 + pages_day4 - 10 = 90)
  (h5 : 2 * pages_day4 - 10 = 90)
  (h6 : 2 * pages_day4 = 100)
  (h7 : pages_day4 = 50) :
  P = 270 ∧ pages_day4 = 50 ∧ pages_day5 = 40 := 
by {
  sorry -- Proof is skipped
}

end sophia_book_problem_l299_299180


namespace calc_f_xh_min_f_x_l299_299424

def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

theorem calc_f_xh_min_f_x (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := 
by
  sorry

end calc_f_xh_min_f_x_l299_299424


namespace fraction_division_l299_299344

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l299_299344


namespace Tony_can_add_4_pairs_of_underwear_l299_299204

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l299_299204


namespace opposite_of_pi_is_neg_pi_l299_299963

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l299_299963


namespace problem_l299_299366

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem problem : f 5 + f (-5) = 0 := by
  sorry

end problem_l299_299366


namespace count_integers_in_interval_l299_299909

theorem count_integers_in_interval :
  {n : ℤ | -5 < n ∧ n < 3}.finite ∧ {n : ℤ | -5 < n ∧ n < 3}.to_finset.card = 7 := by
sorry

end count_integers_in_interval_l299_299909


namespace no_eight_roots_for_nested_quadratics_l299_299550

theorem no_eight_roots_for_nested_quadratics
  (f g h : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e k : ℝ, ∀ x, g x = d * x^2 + e * x + k)
  (hh : ∃ p q r : ℝ, ∀ x, h x = p * x^2 + q * x + r)
  (hroots : ∀ x, f (g (h x)) = 0 → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8)) :
  false :=
by
  sorry

end no_eight_roots_for_nested_quadratics_l299_299550


namespace point_on_same_side_as_l299_299982

def f (x y : ℝ) : ℝ := 2 * x - y + 1

theorem point_on_same_side_as (x1 y1 : ℝ) (h : f 1 2 > 0) : f 1 0 > 0 := sorry

end point_on_same_side_as_l299_299982


namespace increased_percentage_l299_299484

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l299_299484


namespace aaron_guesses_correctly_l299_299026

noncomputable def P_H : ℝ := 2 / 3
noncomputable def P_T : ℝ := 1 / 3
noncomputable def P_G_H : ℝ := 2 / 3
noncomputable def P_G_T : ℝ := 1 / 3

noncomputable def p : ℝ := P_H * P_G_H + P_T * P_G_T

theorem aaron_guesses_correctly :
  9000 * p = 5000 :=
by
  sorry

end aaron_guesses_correctly_l299_299026


namespace abs_sub_abs_eq_six_l299_299268

theorem abs_sub_abs_eq_six
  (a b : ℝ)
  (h₁ : |a| = 4)
  (h₂ : |b| = 2)
  (h₃ : a * b < 0) :
  |a - b| = 6 :=
sorry

end abs_sub_abs_eq_six_l299_299268


namespace cost_of_8_dozen_oranges_l299_299513

noncomputable def cost_per_dozen (cost_5_dozen : ℝ) : ℝ :=
  cost_5_dozen / 5

noncomputable def cost_8_dozen (cost_5_dozen : ℝ) : ℝ :=
  8 * cost_per_dozen cost_5_dozen

theorem cost_of_8_dozen_oranges (cost_5_dozen : ℝ) (h : cost_5_dozen = 39) : cost_8_dozen cost_5_dozen = 62.4 :=
by
  sorry

end cost_of_8_dozen_oranges_l299_299513


namespace all_iterated_quadratic_eq_have_integer_roots_l299_299052

noncomputable def initial_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -p ∧ x1 * x2 = q

noncomputable def iterated_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∀ i : ℕ, i ≤ 9 → ∃ x1 x2 : ℤ, x1 + x2 = -(p + i) ∧ x1 * x2 = (q + i)

theorem all_iterated_quadratic_eq_have_integer_roots :
  ∃ p q : ℤ, initial_quadratic_eq_has_integer_roots p q ∧ iterated_quadratic_eq_has_integer_roots p q :=
sorry

end all_iterated_quadratic_eq_have_integer_roots_l299_299052


namespace percentage_water_in_fresh_grapes_is_65_l299_299543

noncomputable def percentage_water_in_fresh_grapes 
  (weight_fresh : ℝ) (weight_dried : ℝ) (percentage_water_dried : ℝ) : ℝ :=
  100 - ((weight_dried / weight_fresh) - percentage_water_dried / 100 * weight_dried / weight_fresh) * 100

theorem percentage_water_in_fresh_grapes_is_65 :
  percentage_water_in_fresh_grapes 400 155.56 10 = 65 := 
by
  sorry

end percentage_water_in_fresh_grapes_is_65_l299_299543


namespace subsets_with_mean_equal_5_l299_299697

open Finset

noncomputable def original_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem subsets_with_mean_equal_5 : 
  (card (filter (λ t, (t ∈ (original_set.subsets.filter (λ s, s.card = 2))) ∧ 
    (original_set.sum - t.sum) / ( original_set.card - 2) = 5) (original_set.subsets))) = 4 := 
sorry

end subsets_with_mean_equal_5_l299_299697


namespace find_a_l299_299725

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem find_a (a : ℝ) (h : {x | x^2 - 3 * x + 2 = 0} ∩ {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} = {2}) :
  a = -3 ∨ a = -1 :=
by
  sorry

end find_a_l299_299725


namespace charley_pencils_final_count_l299_299650

def charley_initial_pencils := 50
def lost_pencils_while_moving := 8
def misplaced_fraction_first_week := 1 / 3
def lost_fraction_second_week := 1 / 4

theorem charley_pencils_final_count:
  let initial := charley_initial_pencils
  let after_moving := initial - lost_pencils_while_moving
  let misplaced_first_week := misplaced_fraction_first_week * after_moving
  let remaining_after_first_week := after_moving - misplaced_first_week
  let lost_second_week := lost_fraction_second_week * remaining_after_first_week
  let final_pencils := remaining_after_first_week - lost_second_week
  final_pencils = 21 := 
sorry

end charley_pencils_final_count_l299_299650


namespace concentration_after_removing_water_l299_299081

theorem concentration_after_removing_water :
  ∀ (initial_volume : ℝ) (initial_percentage : ℝ) (water_removed : ℝ),
  initial_volume = 18 →
  initial_percentage = 0.4 →
  water_removed = 6 →
  (initial_percentage * initial_volume) / (initial_volume - water_removed) * 100 = 60 :=
by
  intros initial_volume initial_percentage water_removed h1 h2 h3
  rw [h1, h2, h3]
  sorry

end concentration_after_removing_water_l299_299081


namespace grasshopper_jump_distance_l299_299055

theorem grasshopper_jump_distance (g f m : ℕ)
    (h1 : f = g + 32)
    (h2 : m = f - 26)
    (h3 : m = 31) : g = 25 :=
by
  sorry

end grasshopper_jump_distance_l299_299055


namespace library_visitors_total_l299_299933

theorem library_visitors_total
  (visitors_monday : ℕ)
  (visitors_tuesday : ℕ)
  (average_visitors_remaining_days : ℕ)
  (remaining_days : ℕ)
  (total_visitors : ℕ)
  (hmonday : visitors_monday = 50)
  (htuesday : visitors_tuesday = 2 * visitors_monday)
  (haverage : average_visitors_remaining_days = 20)
  (hremaining_days : remaining_days = 5)
  (htotal : total_visitors =
    visitors_monday + visitors_tuesday + remaining_days * average_visitors_remaining_days) :
  total_visitors = 250 :=
by
  -- here goes the proof, marked as sorry for now
  sorry

end library_visitors_total_l299_299933


namespace convex_hexagon_possibilities_l299_299420

noncomputable def hexagon_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6]

theorem convex_hexagon_possibilities : 
  ∃ (hexagons : List (List ℕ)), 
    (∀ h ∈ hexagons, 
      (h.length = 6) ∧ 
      (∀ a ∈ h, a ∈ hexagon_side_lengths)) ∧ 
      (hexagons.length = 3) := 
sorry

end convex_hexagon_possibilities_l299_299420


namespace line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l299_299286

section BarycentricCoordinates

variables {A1 A2 A3 A4 : Type} 

def barycentric_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 + x3 + x4 = 1

theorem line_A1_A2_condition (x1 x2 x3 x4 : ℝ) : 
  barycentric_condition x1 x2 x3 x4 → (x3 = 0 ∧ x4 = 0) ↔ (x1 + x2 = 1) :=
by
  sorry

theorem plane_A1_A2_A3_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x4 = 0) ↔ (x1 + x2 + x3 = 1) :=
by
  sorry

theorem plane_through_A3_A4_parallel_to_A1_A2_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x1 = -x2 ∧ x3 + x4 = 1) ↔ (x1 + x2 + x3 + x4 = 1) :=
by
  sorry

end BarycentricCoordinates

end line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l299_299286


namespace larger_number_l299_299467

theorem larger_number (L S : ℕ) (h1 : L - S = 1345) (h2 : L = 6 * S + 15) : L = 1611 :=
by
  sorry

end larger_number_l299_299467


namespace fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l299_299539

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Condition 1: a = 1, b = 5; the fixed points are x = -1 or x = -4
theorem fixed_points_a_one_b_five : 
  ∀ x : ℝ, is_fixed_point (f 1 5) x ↔ x = -1 ∨ x = -4 := by
  -- Proof goes here
  sorry

-- Condition 2: For any real b, f(x) always having two distinct fixed points implies 0 < a < 1
theorem range_of_a_two_distinct_fixed_points : 
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) ↔ 0 < a ∧ a < 1 := by
  -- Proof goes here
  sorry

end fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l299_299539


namespace julia_picnic_meals_l299_299718

theorem julia_picnic_meals :
  let sandwiches := 4
  let salads := 5
  let choose_salads := Nat.choose salads 3
  let drinks := 3
  sandwiches * choose_salads * drinks = 120 := by
    let sandwiches := 4
    let salads := 5
    let choose_salads := Nat.choose salads 3
    let drinks := 3
    have h1 : sandwiches * choose_salads * drinks = 4 * 10 * 3 := by
      simp [Nat.choose, sandwiches, salads, drinks]
    have h2 : 4 * 10 * 3 = 120 := rfl
    exact eq.trans h1 h2

end julia_picnic_meals_l299_299718


namespace x_equals_y_squared_plus_2y_minus_1_l299_299558

theorem x_equals_y_squared_plus_2y_minus_1 (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 :=
sorry

end x_equals_y_squared_plus_2y_minus_1_l299_299558


namespace spending_limit_l299_299585

variable (n b total_spent limit: ℕ)

theorem spending_limit (hne: n = 34) (hbe: b = n + 5) (hts: total_spent = n + b) (hlo: total_spent = limit + 3) : limit = 70 := by
  sorry

end spending_limit_l299_299585


namespace necessary_but_not_sufficient_condition_l299_299551

-- Conditions from the problem
def p (x : ℝ) : Prop := (x - 1) * (x - 3) ≤ 0
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Theorem statement to prove the correct answer
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → (¬p x → ¬q x) ∧ (q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
by
  -- Solution is implied by given answer in problem description
  sorry

end necessary_but_not_sufficient_condition_l299_299551


namespace hyperbola_eccentricity_range_l299_299127

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 < (Real.sqrt (a^2 + b^2)) / a) ∧ ((Real.sqrt (a^2 + b^2)) / a < (2 * Real.sqrt 3) / 3) :=
sorry

end hyperbola_eccentricity_range_l299_299127


namespace range_sin_cos_two_x_is_minus2_to_9_over_8_l299_299197

noncomputable def range_of_function : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.sin x + Real.cos (2 * x) }

theorem range_sin_cos_two_x_is_minus2_to_9_over_8 :
  range_of_function = Set.Icc (-2) (9 / 8) := 
by
  sorry

end range_sin_cos_two_x_is_minus2_to_9_over_8_l299_299197


namespace john_total_spent_l299_299290

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l299_299290


namespace base_5_conversion_correct_l299_299867

def base_5_to_base_10 : ℕ := 2 * 5^2 + 4 * 5^1 + 2 * 5^0

theorem base_5_conversion_correct : base_5_to_base_10 = 72 :=
by {
  -- Proof (not required in the problem statement)
  sorry
}

end base_5_conversion_correct_l299_299867


namespace productivity_increase_l299_299970

/-- 
The original workday is 8 hours. 
During the first 6 hours, productivity is at the planned level (1 unit/hour). 
For the next 2 hours, productivity falls by 25% (0.75 units/hour). 
The workday is extended by 1 hour (now 9 hours). 
During the first 6 hours of the extended shift, productivity remains at the planned level (1 unit/hour). 
For the remaining 3 hours of the extended shift, productivity falls by 30% (0.7 units/hour). 
Prove that the overall productivity for the shift increased by 8% as a result of extending the workday.
-/
theorem productivity_increase
  (planned_productivity : ℝ)
  (initial_work_hours : ℝ)
  (initial_productivity_drop : ℝ)
  (extended_work_hours : ℝ)
  (extended_productivity_drop : ℝ)
  (initial_total_work : ℝ)
  (extended_total_work : ℝ)
  (percentage_increase : ℝ) :
  planned_productivity = 1 →
  initial_work_hours = 8 →
  initial_productivity_drop = 0.25 →
  extended_work_hours = 9 →
  extended_productivity_drop = 0.30 →
  initial_total_work = 7.5 →
  extended_total_work = 8.1 →
  percentage_increase = 8 →
  ((extended_total_work - initial_total_work) / initial_total_work * 100) = percentage_increase :=
sorry

end productivity_increase_l299_299970


namespace MrBensonPaidCorrectAmount_l299_299370

-- Definitions based on the conditions
def generalAdmissionTicketPrice : ℤ := 40
def VIPTicketPrice : ℤ := 60
def premiumTicketPrice : ℤ := 80

def generalAdmissionTicketsBought : ℤ := 10
def VIPTicketsBought : ℤ := 3
def premiumTicketsBought : ℤ := 2

def generalAdmissionExcessThreshold : ℤ := 8
def VIPExcessThreshold : ℤ := 2
def premiumExcessThreshold : ℤ := 1

def generalAdmissionDiscountPercentage : ℤ := 3
def VIPDiscountPercentage : ℤ := 7
def premiumDiscountPercentage : ℤ := 10

-- Function to calculate the cost without discounts
def costWithoutDiscount : ℤ :=
  (generalAdmissionTicketsBought * generalAdmissionTicketPrice) +
  (VIPTicketsBought * VIPTicketPrice) +
  (premiumTicketsBought * premiumTicketPrice)

-- Function to calculate the total discount
def totalDiscount : ℤ :=
  let generalAdmissionDiscount := if generalAdmissionTicketsBought > generalAdmissionExcessThreshold then 
    (generalAdmissionTicketsBought - generalAdmissionExcessThreshold) * generalAdmissionTicketPrice * generalAdmissionDiscountPercentage / 100 else 0
  let VIPDiscount := if VIPTicketsBought > VIPExcessThreshold then 
    (VIPTicketsBought - VIPExcessThreshold) * VIPTicketPrice * VIPDiscountPercentage / 100 else 0
  let premiumDiscount := if premiumTicketsBought > premiumExcessThreshold then 
    (premiumTicketsBought - premiumExcessThreshold) * premiumTicketPrice * premiumDiscountPercentage / 100 else 0
  generalAdmissionDiscount + VIPDiscount + premiumDiscount

-- Function to calculate the total cost after discounts
def totalCostAfterDiscount : ℤ := costWithoutDiscount - totalDiscount

-- Proof statement
theorem MrBensonPaidCorrectAmount :
  totalCostAfterDiscount = 723 :=
by
  sorry

end MrBensonPaidCorrectAmount_l299_299370


namespace divisibility_by_2880_l299_299409

theorem divisibility_by_2880 (n : ℕ) : 
  (∃ t u : ℕ, (n = 16 * t - 2 ∨ n = 16 * t + 2 ∨ n = 8 * u - 1 ∨ n = 8 * u + 1) ∧ ¬(n % 3 = 0) ∧ ¬(n % 5 = 0)) ↔
  2880 ∣ (n^2 - 4) * (n^2 - 1) * (n^2 + 3) :=
sorry

end divisibility_by_2880_l299_299409


namespace value_of_expression_when_x_is_2_l299_299980

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l299_299980


namespace find_x_perpendicular_l299_299419

/-- Given vectors a = ⟨-1, 2⟩ and b = ⟨1, x⟩, if a is perpendicular to (a + 2 * b),
    then x = -3/4. -/
theorem find_x_perpendicular
  (x : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (1, x))
  (h : (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0)) :
  x = -3 / 4 :=
sorry

end find_x_perpendicular_l299_299419


namespace jason_cuts_lawns_l299_299027

theorem jason_cuts_lawns 
  (time_per_lawn: ℕ)
  (total_cutting_time_hours: ℕ)
  (total_cutting_time_minutes: ℕ)
  (total_yards_cut: ℕ) : 
  time_per_lawn = 30 → 
  total_cutting_time_hours = 8 → 
  total_cutting_time_minutes = total_cutting_time_hours * 60 → 
  total_yards_cut = total_cutting_time_minutes / time_per_lawn → 
  total_yards_cut = 16 :=
by
  intros
  sorry

end jason_cuts_lawns_l299_299027


namespace sum_of_transformed_numbers_l299_299330

variables (a b x k S : ℝ)

-- Define the condition that a + b = S
def sum_condition : Prop := a + b = S

-- Define the function that represents the final sum after transformations
def final_sum (a b x k : ℝ) : ℝ :=
  k * (a + x) + k * (b + x)

-- The theorem statement to prove
theorem sum_of_transformed_numbers (h : sum_condition a b S) : 
  final_sum a b x k = k * S + 2 * k * x :=
by
  sorry

end sum_of_transformed_numbers_l299_299330


namespace james_coffee_weekdays_l299_299931

theorem james_coffee_weekdays :
  ∃ (c d : ℕ) (k : ℤ), (c + d = 5) ∧ 
                      (3 * c + 2 * d + 10 = k / 3) ∧ 
                      (k % 3 = 0) ∧ 
                      c = 2 :=
by 
  sorry

end james_coffee_weekdays_l299_299931


namespace complement_of_A_in_U_l299_299578

-- Define the universal set U and the subset A
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 2, 5, 7}

-- Define the complement of A with respect to U
def complementU_A : Set Nat := {x ∈ U | x ∉ A}

-- Prove the complement of A in U is {3, 4, 6}
theorem complement_of_A_in_U :
  complementU_A = {3, 4, 6} :=
by
  sorry

end complement_of_A_in_U_l299_299578


namespace eval_expression_l299_299603

theorem eval_expression :
  ((-2 : ℤ) ^ 3 : ℝ) ^ (1/3 : ℝ) - (-1 : ℤ) ^ 0 = -3 := by
  sorry

end eval_expression_l299_299603


namespace area_of_rectangle_l299_299839

theorem area_of_rectangle (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 90) : w * l = 379.6875 :=
by
  sorry

end area_of_rectangle_l299_299839


namespace solve_star_op_eq_l299_299869

def star_op (a b : ℕ) : ℕ :=
  if a < b then b * b else b * b * b

theorem solve_star_op_eq :
  ∃ x : ℕ, 5 * star_op 5 x = 64 ∧ (x = 4 ∨ x = 8) :=
sorry

end solve_star_op_eq_l299_299869


namespace alternating_sum_l299_299647

theorem alternating_sum : 
  (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19 + 21 - 23 + 25 - 27 + 29 - 31 + 33 - 35 + 37 - 39 + 41 = 21) :=
by
  sorry

end alternating_sum_l299_299647


namespace last_digit_largest_prime_l299_299053

-- Definition and conditions
def largest_known_prime : ℕ := 2^216091 - 1

-- The statement of the problem we want to prove
theorem last_digit_largest_prime : (largest_known_prime % 10) = 7 := by
  sorry

end last_digit_largest_prime_l299_299053


namespace trig_expression_value_l299_299124

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 + 4 * Real.sin α * Real.cos α - 9 * Real.cos α ^ 2 = 21 / 10 :=
by
  sorry

end trig_expression_value_l299_299124


namespace distance_between_trains_l299_299610

def speed_train1 : ℝ := 11 -- Speed of the first train in mph
def speed_train2 : ℝ := 31 -- Speed of the second train in mph
def time_travelled : ℝ := 8 -- Time in hours

theorem distance_between_trains : 
  (speed_train2 * time_travelled) - (speed_train1 * time_travelled) = 160 := by
  sorry

end distance_between_trains_l299_299610


namespace min_value_fraction_l299_299882

theorem min_value_fraction {x : ℝ} (h : x > 8) : 
    ∃ c : ℝ, (∀ y : ℝ, y = (x^2) / ((x - 8)^2) → c ≤ y) ∧ c = 1 := 
sorry

end min_value_fraction_l299_299882


namespace merchant_profit_l299_299496

theorem merchant_profit (C S : ℝ) (h: 20 * C = 15 * S) : 
  (S - C) / C * 100 = 33.33 := by
sorry

end merchant_profit_l299_299496


namespace incorrect_correlation_coefficient_range_l299_299642

noncomputable def regression_analysis_conditions 
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) : Prop :=
  non_deterministic_relationship ∧
  correlation_coefficient_range ∧
  perfect_correlation ∧
  correlation_coefficient_sign

theorem incorrect_correlation_coefficient_range
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) :
  regression_analysis_conditions 
    non_deterministic_relationship 
    correlation_coefficient_range 
    perfect_correlation 
    correlation_coefficient_sign →
  ¬ correlation_coefficient_range :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end incorrect_correlation_coefficient_range_l299_299642


namespace probability_at_least_one_passes_l299_299689

theorem probability_at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 :=
by sorry

end probability_at_least_one_passes_l299_299689


namespace same_probability_of_selection_l299_299099

variable (students : Type) [Fintype students]

variable (team1_sample team2_sample : Finset students)

variable (n : ℕ) (h1 : Fintype.card team1_sample = n) (h2 : Fintype.card team2_sample = n)

variable (reasonable_sampling : (x : students) → (x ∈ team1_sample) = (x ∈ team2_sample))

theorem same_probability_of_selection (students : Type) [Fintype students] (team1_sample team2_sample : Finset students) 
  (n : ℕ) (h1 : Fintype.card team1_sample = n) (h2 : Fintype.card team2_sample = n)
  (reasonable_sampling : (x : students) → (x ∈ team1_sample) = (x ∈ team2_sample)) :
  ∀ (x : students), (x ∈ team1_sample) = (x ∈ team2_sample) :=
begin
  sorry
end

end same_probability_of_selection_l299_299099


namespace monotonicity_of_f_range_of_a_l299_299132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) - a * x

theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
  (∀ x < 0, f a x ≥ f a (x + 1)) ∧ (∀ x > 0, f a x ≤ f a (x + 1)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.sin x - Real.cos x + 2 - a * x) ↔ a ∈ Set.Ici 1 :=
sorry

end monotonicity_of_f_range_of_a_l299_299132


namespace probability_no_success_l299_299660

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l299_299660


namespace length_of_top_side_l299_299637

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end length_of_top_side_l299_299637


namespace difference_of_scores_l299_299763

variable {x y : ℝ}

theorem difference_of_scores (h : x / y = 4) : x - y = 3 * y := by
  sorry

end difference_of_scores_l299_299763


namespace amount_paid_correct_l299_299569

def initial_debt : ℕ := 100
def hourly_wage : ℕ := 15
def hours_worked : ℕ := 4
def amount_paid_before_work : ℕ := initial_debt - (hourly_wage * hours_worked)

theorem amount_paid_correct : amount_paid_before_work = 40 := by
  sorry

end amount_paid_correct_l299_299569


namespace number_of_rectangles_l299_299150

theorem number_of_rectangles (m n : ℕ) (h1 : m = 8) (h2 : n = 10) : (m - 1) * (n - 1) = 63 := by
  sorry

end number_of_rectangles_l299_299150


namespace find_side_a_l299_299278

noncomputable def maximum_area (A b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ (b + 2 * c = 8) ∧ 
  ((1 / 2) * b * c * Real.sin (2 * Real.pi / 3) = (Real.sqrt 3 / 2) * c * (4 - c) ∧ 
   (∀ (c' : ℝ), (Real.sqrt 3 / 2) * c' * (4 - c') ≤ 2 * Real.sqrt 3) ∧ 
   c = 2)

theorem find_side_a (A b c a : ℝ) (h : maximum_area A b c) :
  a = 2 * Real.sqrt 7 := 
by
  sorry

end find_side_a_l299_299278


namespace max_a_plus_b_l299_299199

noncomputable def max_temperature_diff (a b : ℝ) (t : ℝ) : ℝ :=
  a * Real.sin t + b * Real.cos t

theorem max_a_plus_b 
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : ∃ T : ℝ → ℝ, ∀ t : ℝ, T t = max_temperature_diff a b t)
  (temp_diff : ∀ T : ℝ → ℝ, ∃ Tmax Tmin : ℝ, Tmax - Tmin = 10) :
  a + b ≤ 5 * Real.sqrt 2 :=
sorry

end max_a_plus_b_l299_299199


namespace problem_solution_l299_299447

noncomputable def time_without_distraction : ℝ :=
  let rate_A := 1 / 10
  let rate_B := 0.75 * rate_A
  let rate_C := 0.5 * rate_A
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

noncomputable def time_with_distraction : ℝ :=
  let rate_A := 0.9 * (1 / 10)
  let rate_B := 0.9 * (0.75 * (1 / 10))
  let rate_C := 0.9 * (0.5 * (1 / 10))
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

theorem problem_solution :
  time_without_distraction = 40 / 9 ∧
  time_with_distraction = 44.44 / 9 := by
  sorry

end problem_solution_l299_299447


namespace ratio_of_distances_l299_299634

theorem ratio_of_distances (d_5 d_4 : ℝ) (h1 : d_5 + d_4 ≤ 26.67) (h2 : d_5 / 5 + d_4 / 4 = 6) : 
  d_5 / (d_5 + d_4) = 1 / 2 :=
sorry

end ratio_of_distances_l299_299634


namespace min_trials_correct_l299_299475

noncomputable def minimum_trials (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) : ℕ :=
  Nat.floor ((Real.log (1 - α)) / (Real.log (1 - p))) + 1

-- The theorem to prove the correctness of minimum_trials
theorem min_trials_correct (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) :
  ∃ n : ℕ, minimum_trials α p hα hp = n ∧ (1 - (1 - p)^n ≥ α) :=
by
  sorry

end min_trials_correct_l299_299475


namespace monthly_food_expense_l299_299242

-- Definitions based on the given conditions
def E : ℕ := 6000
def R : ℕ := 640
def EW : ℕ := E / 4
def I : ℕ := E / 5
def L : ℕ := 2280

-- Define the monthly food expense F
def F : ℕ := E - (R + EW + I) - L

-- The theorem stating that the monthly food expense is 380
theorem monthly_food_expense : F = 380 := 
by
  -- proof goes here
  sorry

end monthly_food_expense_l299_299242


namespace find_f_2008_l299_299941

noncomputable def f (x : ℝ) : ℝ := Real.cos x

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
match n with
| 0     => f
| (n+1) => (deriv (f_n n))

theorem find_f_2008 (x : ℝ) : (f_n 2008) x = Real.cos x := by
  sorry

end find_f_2008_l299_299941


namespace complement_intersection_l299_299730

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) : (U \ (M ∩ N)) = {1, 4} := 
by
  sorry

end complement_intersection_l299_299730


namespace sweets_leftover_candies_l299_299672

theorem sweets_leftover_candies (n : ℕ) (h : n % 8 = 5) : (3 * n) % 8 = 7 :=
sorry

end sweets_leftover_candies_l299_299672


namespace commutative_op_l299_299943

variable {S : Type} (op : S → S → S)

-- Conditions
axiom cond1 : ∀ (a b : S), op a (op a b) = b
axiom cond2 : ∀ (a b : S), op (op a b) b = a

-- Proof problem statement
theorem commutative_op : ∀ (a b : S), op a b = op b a :=
by
  intros a b
  sorry

end commutative_op_l299_299943


namespace sum_first_n_geometric_terms_l299_299161

theorem sum_first_n_geometric_terms (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 2 = 2) (h2 : S 6 = 4) :
  S 4 = 1 + Real.sqrt 5 :=
by
  sorry

end sum_first_n_geometric_terms_l299_299161


namespace paving_cost_correct_l299_299826

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_m : ℝ := 300
def area (length : ℝ) (width : ℝ) : ℝ := length * width
def cost (area : ℝ) (rate : ℝ) : ℝ := area * rate

theorem paving_cost_correct :
  cost (area length width) rate_per_sq_m = 6187.50 :=
by
  sorry

end paving_cost_correct_l299_299826


namespace lcm_18_24_eq_72_l299_299787

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299787


namespace parabola_transformation_l299_299022

-- Defining the original parabola
def original_parabola (x : ℝ) : ℝ :=
  3 * x^2

-- Condition: Transformation 1 -> Translation 4 units to the right
def translated_right_parabola (x : ℝ) : ℝ :=
  original_parabola (x - 4)

-- Condition: Transformation 2 -> Translation 1 unit upwards
def translated_up_parabola (x : ℝ) : ℝ :=
  translated_right_parabola x + 1

-- Statement that needs to be proved
theorem parabola_transformation :
  ∀ x : ℝ, translated_up_parabola x = 3 * (x - 4)^2 + 1 :=
by
  intros x
  sorry

end parabola_transformation_l299_299022


namespace lcm_18_24_eq_72_l299_299771

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299771


namespace sum_of_a5_a6_l299_299018

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 + a 2 = 1 ∧ a 3 + a 4 = 4 ∧ q^2 = 4

theorem sum_of_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) (h_cond : geometric_conditions a q) :
  a 5 + a 6 = 16 :=
sorry

end sum_of_a5_a6_l299_299018


namespace hyperbola_find_a_b_l299_299258

def hyperbola_conditions (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (∃ e : ℝ, e = 2) ∧ (∃ c : ℝ, c = 4)

theorem hyperbola_find_a_b (a b : ℝ) : hyperbola_conditions a b → a = 2 ∧ b = 2 * Real.sqrt 3 := 
sorry

end hyperbola_find_a_b_l299_299258


namespace remainder_3a_plus_b_l299_299170

theorem remainder_3a_plus_b (p q : ℤ) (a b : ℤ)
  (h1 : a = 98 * p + 92)
  (h2 : b = 147 * q + 135) :
  ((3 * a + b) % 49) = 19 := by
sorry

end remainder_3a_plus_b_l299_299170


namespace marbles_left_l299_299308

-- Definitions and conditions
def marbles_initial : ℕ := 38
def marbles_lost : ℕ := 15

-- Statement of the problem
theorem marbles_left : marbles_initial - marbles_lost = 23 := by
  sorry

end marbles_left_l299_299308


namespace sugar_solution_l299_299832

theorem sugar_solution (V x : ℝ) (h1 : V > 0) (h2 : 0.1 * (V - x) + 0.5 * x = 0.2 * V) : x / V = 1 / 4 :=
by sorry

end sugar_solution_l299_299832


namespace weight_of_A_l299_299463

theorem weight_of_A (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : (B + C + D + E) / 4 = 79) 
  (h4 : E = D + 7): 
  A = 79 := by
  have h5 : A + B + C = 252 := by
    linarith [h1]
  have h6 : A + B + C + D = 320 := by
    linarith [h2]
  have h7 : B + C + D + E = 316 := by
    linarith [h3]
  have hD : D = 68 := by
    linarith [h5, h6]
  have hE : E = 75 := by
    linarith [hD, h4]
  have hBC : B + C = 252 - A := by
    linarith [h5]
  have : 252 - A + 68 + 75 = 316 := by
    linarith [h7, hBC, hD, hE]
  linarith

end weight_of_A_l299_299463


namespace tangent_of_curve_at_point_l299_299749

def curve (x : ℝ) : ℝ := x^3 - 4 * x

def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

theorem tangent_of_curve_at_point : 
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ tangent_line x y) :=
sorry

end tangent_of_curve_at_point_l299_299749


namespace carla_total_marbles_l299_299395

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem carla_total_marbles : initial_marbles + bought_marbles = 321.0 := by
  sorry

end carla_total_marbles_l299_299395


namespace age_of_15th_person_l299_299462

variable (avg_age_20 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ) (A : ℕ)
variable (num_20 : ℕ) (num_5 : ℕ) (num_9 : ℕ)

theorem age_of_15th_person (h1 : avg_age_20 = 15) (h2 : avg_age_5 = 14) (h3 : avg_age_9 = 16)
  (h4 : num_20 = 20) (h5 : num_5 = 5) (h6 : num_9 = 9) :
  (num_20 * avg_age_20) = (num_5 * avg_age_5) + (num_9 * avg_age_9) + A → A = 86 :=
by
  sorry

end age_of_15th_person_l299_299462


namespace value_of_expression_when_x_is_2_l299_299978

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l299_299978


namespace range_of_m_l299_299540

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (mx-1)*(x-2) > 0 ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_of_m_l299_299540


namespace Vikki_take_home_pay_is_correct_l299_299764

noncomputable def Vikki_take_home_pay : ℝ :=
  let hours_worked : ℝ := 42
  let hourly_pay_rate : ℝ := 12
  let gross_earnings : ℝ := hours_worked * hourly_pay_rate

  let fed_tax_first_300 : ℝ := 300 * 0.15
  let amount_over_300 : ℝ := gross_earnings - 300
  let fed_tax_excess : ℝ := amount_over_300 * 0.22
  let total_federal_tax : ℝ := fed_tax_first_300 + fed_tax_excess

  let state_tax : ℝ := gross_earnings * 0.07
  let retirement_contribution : ℝ := gross_earnings * 0.06
  let insurance_cover : ℝ := gross_earnings * 0.03
  let union_dues : ℝ := 5

  let total_deductions : ℝ := total_federal_tax + state_tax + retirement_contribution + insurance_cover + union_dues
  let take_home_pay : ℝ := gross_earnings - total_deductions
  take_home_pay

theorem Vikki_take_home_pay_is_correct : Vikki_take_home_pay = 328.48 :=
by
  sorry

end Vikki_take_home_pay_is_correct_l299_299764


namespace increase_75_by_150_percent_l299_299483

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l299_299483


namespace michaels_brother_money_end_l299_299733

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l299_299733


namespace number_of_integers_with_three_divisors_l299_299962

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end number_of_integers_with_three_divisors_l299_299962


namespace paint_for_smaller_statues_l299_299508

open Real

theorem paint_for_smaller_statues :
  ∀ (paint_needed : ℝ) (height_big_statue height_small_statue : ℝ) (num_small_statues : ℝ),
  height_big_statue = 10 → height_small_statue = 2 → paint_needed = 5 → num_small_statues = 200 →
  (paint_needed / (height_big_statue / height_small_statue) ^ 2) * num_small_statues = 40 :=
by
  intros paint_needed height_big_statue height_small_statue num_small_statues
  intros h_big_height h_small_height h_paint_needed h_num_small
  rw [h_big_height, h_small_height, h_paint_needed, h_num_small]
  sorry

end paint_for_smaller_statues_l299_299508


namespace base7_addition_l299_299113

theorem base7_addition (X Y : ℕ) (h1 : X + 5 = 9) (h2 : Y + 2 = 4) : X + Y = 6 :=
by
  sorry

end base7_addition_l299_299113


namespace arithmetic_sequence_sum_l299_299284

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h₁ : ∀ n, a (n + 1) = a n + d)
    (h₂ : a 3 + a 5 + a 7 + a 9 + a 11 = 20) : a 1 + a 13 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l299_299284


namespace min_value_g_geq_6_min_value_g_eq_6_l299_299679

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l299_299679


namespace cara_optimal_reroll_two_dice_probability_l299_299394

def probability_reroll_two_dice : ℚ :=
  -- Probability derived from Cara's optimal reroll decisions
  5 / 27

theorem cara_optimal_reroll_two_dice_probability :
  cara_probability_optimal_reroll_two_dice = 5 / 27 := by sorry

end cara_optimal_reroll_two_dice_probability_l299_299394


namespace probability_same_color_l299_299850

open_locale big_operators

-- Definitions for the problem's conditions
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles
def number_of_draws : ℕ := 4

-- Definitions for the probabilities
def P_all_red : ℚ := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) *
                      ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

def P_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) *
                       ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

def P_all_blue : ℚ := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) *
                      ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

def P_same_color : ℚ := P_all_red + P_all_white + P_all_blue

-- Theorem statement to prove the probability of drawing four marbles of the same color
theorem probability_same_color :
  P_same_color = 55 / 3060 :=
sorry

end probability_same_color_l299_299850


namespace basketball_games_count_l299_299945

noncomputable def tokens_per_game : ℕ := 3
noncomputable def total_tokens : ℕ := 18
noncomputable def air_hockey_games : ℕ := 2
noncomputable def air_hockey_tokens := air_hockey_games * tokens_per_game
noncomputable def remaining_tokens := total_tokens - air_hockey_tokens

theorem basketball_games_count :
  (remaining_tokens / tokens_per_game) = 4 := by
  sorry

end basketball_games_count_l299_299945


namespace increase_by_percentage_l299_299478

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l299_299478


namespace largest_angle_in_ratio_triangle_l299_299189

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l299_299189


namespace find_side_a_from_triangle_conditions_l299_299919

-- Define the variables.
variables (A : ℝ) (b : ℝ) (area : ℝ) (a : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  A = 60 * Real.pi / 180 ∧ -- Convert 60 degrees to radians for Lean.
  b = 1 ∧
  area = Real.sqrt 3

-- The theorem we want to prove.
theorem find_side_a_from_triangle_conditions :
  conditions A b area → a = Real.sqrt 13 :=
by
  sorry

end find_side_a_from_triangle_conditions_l299_299919


namespace simple_interest_rate_l299_299273

theorem simple_interest_rate (P : ℝ) (increase_time : ℝ) (increase_amount : ℝ) 
(hP : P = 2000) (h_increase_time : increase_time = 4) (h_increase_amount : increase_amount = 40) :
  ∃ R : ℝ, (2000 * R / 100 * (increase_time + 4) - 2000 * R / 100 * increase_time = increase_amount) ∧ (R = 0.5) := 
by
  sorry

end simple_interest_rate_l299_299273


namespace cos_of_tan_l299_299251

/-- Given a triangle ABC with angle A such that tan(A) = -5/12, prove cos(A) = -12/13. -/
theorem cos_of_tan (A : ℝ) (h : Real.tan A = -5 / 12) : Real.cos A = -12 / 13 := by
  sorry

end cos_of_tan_l299_299251


namespace average_coins_per_day_l299_299104

theorem average_coins_per_day :
  let a := 10
  let d := 10
  let n := 7
  let extra := 20
  let total_coins := a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d + extra)
  total_coins = 300 →
  total_coins / n = 300 / 7 :=
by
  sorry

end average_coins_per_day_l299_299104


namespace find_special_four_digit_square_l299_299879

theorem find_special_four_digit_square :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    n = 8281 ∧
    a = c ∧
    b + 1 = d ∧
    n = (91 : ℕ) ^ 2 :=
by
  sorry

end find_special_four_digit_square_l299_299879


namespace percentage_of_discount_l299_299507

variable (C : ℝ) -- Cost Price of the Book

-- Conditions
axiom profit_with_discount (C : ℝ) : ∃ S_d : ℝ, S_d = C * 1.235
axiom profit_without_discount (C : ℝ) : ∃ S_nd : ℝ, S_nd = C * 2.30

-- Theorem to prove
theorem percentage_of_discount (C : ℝ) : 
  ∃ discount_percentage : ℝ, discount_percentage = 46.304 := by
  sorry

end percentage_of_discount_l299_299507


namespace barbara_typing_time_l299_299857

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end barbara_typing_time_l299_299857


namespace max_min_P_l299_299294

theorem max_min_P (a b c : ℝ) (h : |a + b| + |b + c| + |c + a| = 8) :
  (a^2 + b^2 + c^2 = 48) ∨ (a^2 + b^2 + c^2 = 16 / 3) :=
sorry

end max_min_P_l299_299294


namespace desired_percentage_total_annual_income_l299_299106

variable (investment1 : ℝ)
variable (investment2 : ℝ)
variable (rate1 : ℝ)
variable (rate2 : ℝ)

theorem desired_percentage_total_annual_income (h1 : investment1 = 2000)
  (h2 : rate1 = 0.05)
  (h3 : investment2 = 1000-1e-13)
  (h4 : rate2 = 0.08):
  ((investment1 * rate1 + investment2 * rate2) / (investment1 + investment2) * 100) = 6 := by
  sorry

end desired_percentage_total_annual_income_l299_299106


namespace tan_alpha_eq_one_then_expr_value_l299_299559

theorem tan_alpha_eq_one_then_expr_value (α : ℝ) (h : Real.tan α = 1) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 2 / 3 :=
by
  sorry

end tan_alpha_eq_one_then_expr_value_l299_299559


namespace polynomial_product_c_l299_299721

theorem polynomial_product_c (b c : ℝ) (h1 : b = 2 * c - 1) (h2 : (x^2 + b * x + c) = 0 → (∃ r : ℝ, x = r)) :
  c = 1 / 2 :=
sorry

end polynomial_product_c_l299_299721


namespace simplify_expression_l299_299335

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l299_299335


namespace trajectory_equation_max_area_triangle_l299_299713

-- Definition of trajectory Γ
def trajectory (x y : ℝ) : Prop :=
  y ≠ 0 ∧ (x ^ 2 / 4 + y ^ 2 = 1)

-- Statement of the first question
theorem trajectory_equation (x y : ℝ) (h : trajectory x y) : 
  x^2 / 4 + y^2 = 1 := 
  by 
    have h1 : y ≠ 0 := h.left
    have h2 : x^2 / 4 + y^2 = 1 := h.right
    exact h2

-- Statement of the second question
theorem max_area_triangle (b : ℝ) (h : 5 - b^2 > 0) : 
  ∃ (S : ℝ), S = 2/5 * (b + 3) * √(5 - b^2) ∧ 
  S ≤ 16/5 :=
  by 
    use 2/5 * (b + 3) * √(5 - b^2)
    split
    case a => rfl
    case b => sorry

end trajectory_equation_max_area_triangle_l299_299713


namespace least_number_l299_299433

noncomputable def permutations := list.perm

noncomputable def alpha (n : ℕ) : list ℕ := list.range n
noncomputable def beta (n : ℕ) : list ℕ := alpha n
noncomputable def gamma (n : ℕ) : list ℕ := alpha n
noncomputable def delta (n : ℕ) : list ℕ := list.reverse (alpha n)

theorem least_number (n : ℕ) (h : n ≥ 2) :
  (∃ (α β γ δ : list ℕ), α.permutations ∧ β.permutations ∧ γ.permutations ∧ δ.permutations ∧ 
    list.sum (list.zip_with (*) α β) = (list.sum (list.zip_with (*) γ δ) * 19 / 10))
  → n = 28 :=
sorry

end least_number_l299_299433


namespace number_of_cows_l299_299376

-- Definitions
variable (H C : ℕ)
variable h1 : C = 5 * H
variable h2 : C + H = 168

-- Proof Statement
theorem number_of_cows : C = 140 :=
by
  -- each line needs to be filled according
  sorry

end number_of_cows_l299_299376


namespace algebraic_expression_value_l299_299705

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2 * x - 1 = 0) : x^3 - x^2 - 3 * x + 2 = 3 := 
by
  sorry

end algebraic_expression_value_l299_299705


namespace find_kg_of_mangoes_l299_299261

-- Define the conditions
def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 965
def cost_of_mangoes (m : ℕ) : ℕ := 45 * m

-- Formalize the proof problem
theorem find_kg_of_mangoes (m : ℕ) :
  cost_of_grapes + cost_of_mangoes m = total_amount_paid → m = 9 :=
by
  intros h
  sorry

end find_kg_of_mangoes_l299_299261


namespace typing_time_l299_299856

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l299_299856


namespace trajectory_of_complex_point_l299_299320

open Complex Topology

theorem trajectory_of_complex_point (z : ℂ) (hz : ‖z‖ ≤ 1) : 
  {w : ℂ | ‖w‖ ≤ 1} = {w : ℂ | w.re * w.re + w.im * w.im ≤ 1} :=
sorry

end trajectory_of_complex_point_l299_299320


namespace red_section_not_damaged_l299_299664

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l299_299664


namespace residue_of_neg_1237_mod_37_l299_299668

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l299_299668


namespace lcm_18_24_eq_72_l299_299790

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299790


namespace probability_increasing_l299_299119

noncomputable def p (x : ℝ) : ℝ := sorry -- This is a placeholder for the actual function definition.

theorem probability_increasing (E : set ℝ) (hE1 : [0,1] ⊆ E) (hE2 : E ⊆ [0, +∞]) (hE3 : is_compact E) : 
  ∀ x y : ℝ, -1 ≤ x ∧ x < 0 ∧ -1 ≤ y ∧ y < 0 ∧ x < y → p x ≤ p y := 
sorry

end probability_increasing_l299_299119


namespace find_complement_l299_299417

-- Define predicate for a specific universal set U and set A
def universal_set (a : ℤ) (x : ℤ) : Prop :=
  x = a^2 - 2 ∨ x = 2 ∨ x = 1

def set_A (a : ℤ) (x : ℤ) : Prop :=
  x = a ∨ x = 1

-- Define complement of A with respect to U
def complement_U_A (a : ℤ) (x : ℤ) : Prop :=
  universal_set a x ∧ ¬ set_A a x

-- Main theorem statement
theorem find_complement (a : ℤ) (h : a ≠ 2) : { x | complement_U_A a x } = {2} :=
by
  sorry

end find_complement_l299_299417


namespace chemical_reaction_l299_299240

def reaction_balanced (koh nh4i ki nh3 h2o : ℕ) : Prop :=
  koh = nh4i ∧ nh4i = ki ∧ ki = nh3 ∧ nh3 = h2o

theorem chemical_reaction
  (KOH NH4I : ℕ)
  (h1 : KOH = 3)
  (h2 : NH4I = 3)
  (balanced : reaction_balanced KOH NH4I 3 3 3) :
  (∃ (NH3 KI H2O : ℕ),
    NH3 = 3 ∧ KI = 3 ∧ H2O = 3 ∧ 
    NH3 = NH4I - NH4I ∧
    KI = KOH - KOH ∧
    H2O = KOH - KOH) ∧
  (KOH = NH4I) := 
by sorry

end chemical_reaction_l299_299240


namespace box_volume_l299_299357

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l299_299357


namespace sum_of_first_3n_terms_l299_299282

variable {S : ℕ → ℝ}
variable {n : ℕ}
variable {a b : ℝ}

def arithmetic_sum (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, S (m + 1) = S m + (d * (m + 1))

theorem sum_of_first_3n_terms (h1 : S n = a) (h2 : S (2 * n) = b) 
  (h3 : arithmetic_sum S) : S (3 * n) = 3 * b - 2 * a :=
by
  sorry

end sum_of_first_3n_terms_l299_299282


namespace range_of_a_l299_299122

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 :=
by
  sorry

end range_of_a_l299_299122


namespace forty_percent_of_number_l299_299045

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 20) : 0.40 * N = 240 :=
by
  sorry

end forty_percent_of_number_l299_299045


namespace probability_of_pink_tie_l299_299628

theorem probability_of_pink_tie 
  (black_ties gold_ties pink_ties : ℕ) 
  (h_black : black_ties = 5) 
  (h_gold : gold_ties = 7) 
  (h_pink : pink_ties = 8) 
  (h_total : (5 + 7 + 8) = (black_ties + gold_ties + pink_ties)) 
  : (pink_ties : ℚ) / (black_ties + gold_ties + pink_ties) = 2 / 5 := 
by 
  sorry

end probability_of_pink_tie_l299_299628


namespace combined_score_is_75_l299_299644

variable (score1 : ℕ) (total1 : ℕ)
variable (score2 : ℕ) (total2 : ℕ)
variable (score3 : ℕ) (total3 : ℕ)

-- Conditions: Antonette's scores and the number of problems in each test
def Antonette_scores : Prop :=
  score1 = 60 * total1 / 100 ∧ total1 = 15 ∧
  score2 = 85 * total2 / 100 ∧ total2 = 20 ∧
  score3 = 75 * total3 / 100 ∧ total3 = 25

-- Theorem to prove the combined score is 75% (45 out of 60) rounded to the nearest percent
theorem combined_score_is_75
  (h : Antonette_scores score1 total1 score2 total2 score3 total3) :
  100 * (score1 + score2 + score3) / (total1 + total2 + total3) = 75 :=
by sorry

end combined_score_is_75_l299_299644


namespace increased_percentage_l299_299486

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l299_299486


namespace sandy_friend_puppies_l299_299453

theorem sandy_friend_puppies (original_puppies friend_puppies final_puppies : ℕ)
    (h1 : original_puppies = 8) (h2 : final_puppies = 12) :
    friend_puppies = final_puppies - original_puppies := by
    sorry

end sandy_friend_puppies_l299_299453


namespace typing_time_l299_299854

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l299_299854


namespace allan_balloons_l299_299846

def initial_balloons : ℕ := 5
def additional_balloons : ℕ := 3
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem allan_balloons :
  total_balloons = 8 :=
sorry

end allan_balloons_l299_299846


namespace max_underwear_pairs_l299_299202

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l299_299202


namespace green_pill_cost_l299_299383

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end green_pill_cost_l299_299383


namespace train_pass_bridge_time_l299_299843

noncomputable def totalDistance (trainLength bridgeLength : ℕ) : ℕ :=
  trainLength + bridgeLength

noncomputable def speedInMPerSecond (speedInKmPerHour : ℕ) : ℝ :=
  (speedInKmPerHour * 1000) / 3600

noncomputable def timeToPass (totalDistance : ℕ) (speedInMPerSecond : ℝ) : ℝ :=
  totalDistance / speedInMPerSecond

theorem train_pass_bridge_time
  (trainLength : ℕ) (bridgeLength : ℕ) (speedInKmPerHour : ℕ)
  (h_train : trainLength = 300)
  (h_bridge : bridgeLength = 115)
  (h_speed : speedInKmPerHour = 35) :
  timeToPass (totalDistance trainLength bridgeLength) (speedInMPerSecond speedInKmPerHour) = 42.7 :=
by
  sorry

end train_pass_bridge_time_l299_299843


namespace largest_angle_is_75_l299_299192

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l299_299192


namespace frac_x_y_value_l299_299302

theorem frac_x_y_value (x y : ℝ) (h1 : 3 < (2 * x - y) / (x + 2 * y))
(h2 : (2 * x - y) / (x + 2 * y) < 7) (h3 : ∃ (t : ℤ), x = t * y) : x / y = -4 := by
  sorry

end frac_x_y_value_l299_299302


namespace one_thirds_in_fraction_l299_299696

theorem one_thirds_in_fraction : (11 / 5) / (1 / 3) = 33 / 5 := by
  sorry

end one_thirds_in_fraction_l299_299696


namespace find_integer_n_l299_299245

theorem find_integer_n (n : ℤ) (hn : -150 < n ∧ n < 150) : (n = 80 ∨ n = -100) ↔ (Real.tan (n * Real.pi / 180) = Real.tan (1340 * Real.pi / 180)) :=
by 
  sorry

end find_integer_n_l299_299245


namespace problem_proof_l299_299422

def P : Set ℝ := {x | x ≤ 3}

theorem problem_proof : {-1} ⊆ P := 
sorry

end problem_proof_l299_299422


namespace tank_full_capacity_l299_299631

-- Define the conditions
def gas_tank_initially_full_fraction : ℚ := 4 / 5
def gas_tank_after_usage_fraction : ℚ := 1 / 3
def used_gallons : ℚ := 18

-- Define the statement that translates to "How many gallons does this tank hold when it is full?"
theorem tank_full_capacity (x : ℚ) : 
  gas_tank_initially_full_fraction * x - gas_tank_after_usage_fraction * x = used_gallons → 
  x = 270 / 7 :=
sorry

end tank_full_capacity_l299_299631


namespace regular_train_pass_time_l299_299632

-- Define the lengths of the trains
def high_speed_train_length : ℕ := 400
def regular_train_length : ℕ := 600

-- Define the observation time for the passenger on the high-speed train
def observation_time : ℕ := 3

-- Define the problem to find the time x for the regular train passenger
theorem regular_train_pass_time :
  ∃ (x : ℕ), (regular_train_length / observation_time) * x = high_speed_train_length :=
by 
  sorry

end regular_train_pass_time_l299_299632


namespace bacon_calories_percentage_l299_299861

theorem bacon_calories_percentage (total_calories : ℕ) (bacon_strip_calories : ℕ) (num_strips : ℕ)
    (h1 : total_calories = 1250) (h2 : bacon_strip_calories = 125) (h3 : num_strips = 2) :
    (bacon_strip_calories * num_strips * 100) / total_calories = 20 := by
  sorry

end bacon_calories_percentage_l299_299861


namespace base_five_to_decimal_l299_299765

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2 => 2 * 5^0
  | 3 => 3 * 5^1
  | 1 => 1 * 5^2
  | _ => 0

theorem base_five_to_decimal : base5_to_base10 2 + base5_to_base10 3 + base5_to_base10 1 = 42 :=
by sorry

end base_five_to_decimal_l299_299765


namespace cos_a3_value_l299_299155

theorem cos_a3_value (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 1 + a 3 + a 5 = Real.pi) : 
  Real.cos (a 3) = 1/2 := 
by 
  sorry

end cos_a3_value_l299_299155


namespace rectangle_area_l299_299285

theorem rectangle_area (a b c d : ℝ) 
  (ha : a = 4) 
  (hb : b = 4) 
  (hc : c = 4) 
  (hd : d = 1) :
  ∃ E F G H : ℝ,
    (E = 0 ∧ F = 3 ∧ G = 4 ∧ H = 0) →
    (a + b + c + d) = 10 :=
by
  intros
  sorry

end rectangle_area_l299_299285


namespace right_triangle_of_ratio_and_right_angle_l299_299759

-- Define the sides and the right angle condition based on the problem conditions
variable (x : ℝ) (hx : 0 < x)

-- Variables for the sides in the given ratio
def a := 3 * x
def b := 4 * x
def c := 5 * x

-- The proposition we need to prove
theorem right_triangle_of_ratio_and_right_angle (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by sorry  -- Proof not required as per instructions

end right_triangle_of_ratio_and_right_angle_l299_299759


namespace walk_to_lake_park_restaurant_is_zero_l299_299025

noncomputable def time_to_hidden_lake : ℕ := 15
noncomputable def time_to_return_from_hidden_lake : ℕ := 7
noncomputable def total_walk_time_dante : ℕ := 22

theorem walk_to_lake_park_restaurant_is_zero :
  ∃ (x : ℕ), (2 * x + time_to_hidden_lake + time_to_return_from_hidden_lake = total_walk_time_dante) → x = 0 :=
by
  use 0
  intros
  sorry

end walk_to_lake_park_restaurant_is_zero_l299_299025


namespace midpoint_coordinates_l299_299255

theorem midpoint_coordinates (A B M : ℝ × ℝ) (hx : A = (2, -4)) (hy : B = (-6, 2)) (hm : M = (-2, -1)) :
  let (x1, y1) := A
  let (x2, y2) := B
  M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
  sorry

end midpoint_coordinates_l299_299255


namespace problem_prime_square_plus_two_l299_299421

theorem problem_prime_square_plus_two (P : ℕ) (hP_prime : Prime P) (hP2_plus_2_prime : Prime (P^2 + 2)) : P^4 + 1921 = 2002 :=
by
  sorry

end problem_prime_square_plus_two_l299_299421


namespace grain_output_l299_299584

-- Define the condition regarding grain output.
def premier_goal (x : ℝ) : Prop :=
  x > 1.3

-- The mathematical statement that needs to be proved, given the condition.
theorem grain_output (x : ℝ) (h : premier_goal x) : x > 1.3 :=
by
  sorry

end grain_output_l299_299584


namespace minimum_value_a_plus_2b_l299_299411

theorem minimum_value_a_plus_2b {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) : a + 2 * b = 9 :=
by sorry

end minimum_value_a_plus_2b_l299_299411


namespace basketball_team_initial_games_l299_299503

theorem basketball_team_initial_games (G W : ℝ) 
  (h1 : W = 0.70 * G) 
  (h2 : W + 2 = 0.60 * (G + 10)) : 
  G = 40 :=
by
  sorry

end basketball_team_initial_games_l299_299503


namespace lattice_points_condition_l299_299579

/-- A lattice point is a point on the plane with integer coordinates. -/
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

/-- A triangle in the plane with three vertices and at least two lattice points inside. -/
structure Triangle :=
  (A B C : LatticePoint)
  (lattice_points_inside : List LatticePoint)
  (lattice_points_nonempty : lattice_points_inside.length ≥ 2)

noncomputable def exists_lattice_points (T : Triangle) : Prop :=
∃ (X Y : LatticePoint) (hX : X ∈ T.lattice_points_inside) (hY : Y ∈ T.lattice_points_inside), 
  ((∃ (V : LatticePoint), V = T.A ∨ V = T.B ∨ V = T.C ∧ ∃ (k : ℤ), (k : ℝ) * (Y.x - X.x) = (V.x - X.x) ∧ (k : ℝ) * (Y.y - X.y) = (V.y - X.y)) ∨
  (∃ (l m n : ℝ), l * (Y.x - X.x) = m * (T.A.x - T.B.x) ∧ l * (Y.y - X.y) = m * (T.A.y - T.B.y) ∨ l * (Y.x - X.x) = n * (T.B.x - T.C.x) ∧ l * (Y.y - X.y) = n * (T.B.y - T.C.y) ∨ l * (Y.x - X.x) = m * (T.C.x - T.A.x) ∧ l * (Y.y - X.y) = m * (T.C.y - T.A.y)))

theorem lattice_points_condition (T : Triangle) : exists_lattice_points T :=
sorry

end lattice_points_condition_l299_299579


namespace tetrahedron_volume_ratio_l299_299228

theorem tetrahedron_volume_ratio
  (a b : ℝ)
  (larger_tetrahedron : a = 6)
  (smaller_tetrahedron : b = a / 2) :
  (b^3 / a^3) = 1 / 8 := 
by 
  sorry

end tetrahedron_volume_ratio_l299_299228


namespace quadrilateral_condition_l299_299505

variable (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a + b + c + d = 2) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ a + b + c > 1 :=
by
  sorry

end quadrilateral_condition_l299_299505


namespace number_of_integers_satisfying_ineq_l299_299910

theorem number_of_integers_satisfying_ineq : 
  (finset.card {n : ℤ | (n - 3) * (n + 5) < 0}.to_finset) = 7 := 
sorry

end number_of_integers_satisfying_ineq_l299_299910


namespace lcm_18_24_eq_72_l299_299785

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l299_299785


namespace number_of_possible_A2_eq_one_l299_299163

noncomputable def unique_possible_A2 (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  (A^4 = 0) → (A^2 = 0)

theorem number_of_possible_A2_eq_one (A : Matrix (Fin 2) (Fin 2) ℝ) :
  unique_possible_A2 A :=
by 
  sorry

end number_of_possible_A2_eq_one_l299_299163


namespace percentage_error_l299_299226

theorem percentage_error (x : ℝ) (hx : x ≠ 0) :
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 :=
by
  sorry

end percentage_error_l299_299226


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l299_299557

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l299_299557
