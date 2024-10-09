import Mathlib

namespace find_amount_l2179_217903

def total_amount (A : ℝ) : Prop :=
  A / 20 = A / 25 + 100

theorem find_amount 
  (A : ℝ) 
  (h : total_amount A) : 
  A = 10000 := 
  sorry

end find_amount_l2179_217903


namespace amount_after_two_years_l2179_217920

theorem amount_after_two_years (P : ℝ) (r1 r2 : ℝ) : 
  P = 64000 → 
  r1 = 0.12 → 
  r2 = 0.15 → 
  (P + P * r1) + (P + P * r1) * r2 = 82432 := by
  sorry

end amount_after_two_years_l2179_217920


namespace savings_correct_l2179_217963

noncomputable def savings (income expenditure : ℕ) : ℕ :=
income - expenditure

theorem savings_correct (I E : ℕ) (h_ratio :  I / E = 10 / 4) (h_income : I = 19000) :
  savings I E = 11400 :=
sorry

end savings_correct_l2179_217963


namespace cricket_team_members_l2179_217944

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (team_avg_age : ℕ) 
  (remaining_avg_age : ℕ) 
  (h1 : captain_age = 26)
  (h2 : wicket_keeper_age = 29)
  (h3 : team_avg_age = 23)
  (h4 : remaining_avg_age = 22) 
  (h5 : team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age) : 
  n = 11 := 
sorry

end cricket_team_members_l2179_217944


namespace restocked_bags_correct_l2179_217934

def initial_stock := 55
def sold_bags := 23
def final_stock := 164

theorem restocked_bags_correct :
  (final_stock - (initial_stock - sold_bags)) = 132 :=
by
  -- The proof would go here, but we use sorry to skip it.
  sorry

end restocked_bags_correct_l2179_217934


namespace exists_centrally_symmetric_inscribed_convex_hexagon_l2179_217972

-- Definition of a convex polygon with vertices
def convex_polygon (W : Type) : Prop := sorry

-- Definition of the unit area condition
def has_unit_area (W : Type) : Prop := sorry

-- Definition of being centrally symmetric
def is_centrally_symmetric (V : Type) : Prop := sorry

-- Definition of being inscribed
def is_inscribed_polygon (V W : Type) : Prop := sorry

-- Definition of a convex hexagon
def convex_hexagon (V : Type) : Prop := sorry

-- Main theorem statement
theorem exists_centrally_symmetric_inscribed_convex_hexagon (W : Type) 
  (hW_convex : convex_polygon W) (hW_area : has_unit_area W) : 
  ∃ V : Type, convex_hexagon V ∧ is_centrally_symmetric V ∧ is_inscribed_polygon V W ∧ sorry :=
  sorry

end exists_centrally_symmetric_inscribed_convex_hexagon_l2179_217972


namespace base_of_exponential_function_l2179_217962

theorem base_of_exponential_function (a : ℝ) (h : ∀ x : ℝ, y = a^x) :
  (a > 1 ∧ (a - 1 / a = 1)) ∨ (0 < a ∧ a < 1 ∧ (1 / a - a = 1)) → 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end base_of_exponential_function_l2179_217962


namespace product_469160_9999_l2179_217960

theorem product_469160_9999 :
  469160 * 9999 = 4690696840 :=
by
  sorry

end product_469160_9999_l2179_217960


namespace expression_value_l2179_217919

theorem expression_value : 4 * (8 - 2) ^ 2 - 6 = 138 :=
by
  sorry

end expression_value_l2179_217919


namespace radius_of_inscribed_semicircle_in_isosceles_triangle_l2179_217970

theorem radius_of_inscribed_semicircle_in_isosceles_triangle
    (BC : ℝ) (h : ℝ) (r : ℝ)
    (H_eq : BC = 24)
    (H_height : h = 18)
    (H_area : 0.5 * BC * h = 0.5 * 24 * 18) :
    r = 18 / π := by
    sorry

end radius_of_inscribed_semicircle_in_isosceles_triangle_l2179_217970


namespace pick_three_cards_in_order_l2179_217957

theorem pick_three_cards_in_order (deck_size : ℕ) (first_card_ways : ℕ) (second_card_ways : ℕ) (third_card_ways : ℕ) 
  (total_combinations : ℕ) (h1 : deck_size = 52) (h2 : first_card_ways = 52) 
  (h3 : second_card_ways = 51) (h4 : third_card_ways = 50) (h5 : total_combinations = first_card_ways * second_card_ways * third_card_ways) : 
  total_combinations = 132600 := 
by 
  sorry

end pick_three_cards_in_order_l2179_217957


namespace grade_assignment_ways_l2179_217902

theorem grade_assignment_ways : (4 ^ 12) = 16777216 := by
  sorry

end grade_assignment_ways_l2179_217902


namespace arithmetic_sequence_sum_cubes_l2179_217927

theorem arithmetic_sequence_sum_cubes (x : ℤ) (k : ℕ) (h : ∀ i, 0 <= i ∧ i <= k → (x + 2 * i : ℤ)^3 =
  -1331) (hk : k > 3) : k = 6 :=
sorry

end arithmetic_sequence_sum_cubes_l2179_217927


namespace solve_inequality_l2179_217930

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l2179_217930


namespace kids_with_red_hair_l2179_217925

theorem kids_with_red_hair (total_kids : ℕ) (ratio_red ratio_blonde ratio_black : ℕ) 
  (h_ratio : ratio_red + ratio_blonde + ratio_black = 16) (h_total : total_kids = 48) :
  (total_kids / (ratio_red + ratio_blonde + ratio_black)) * ratio_red = 9 :=
by
  sorry

end kids_with_red_hair_l2179_217925


namespace sin_cos_cos_sin_unique_pair_exists_uniq_l2179_217961

noncomputable def theta (x : ℝ) : ℝ := Real.sin (Real.cos x) - x

theorem sin_cos_cos_sin_unique_pair_exists_uniq (h : 0 < c ∧ c < (1/2) * Real.pi ∧ 0 < d ∧ d < (1/2) * Real.pi) :
  (∃! (c d : ℝ), Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d ∧ c < d) :=
sorry

end sin_cos_cos_sin_unique_pair_exists_uniq_l2179_217961


namespace percentage_calculation_l2179_217923

theorem percentage_calculation (amount : ℝ) (percentage : ℝ) (res : ℝ) :
  amount = 400 → percentage = 0.25 → res = amount * percentage → res = 100 := by
  intro h_amount h_percentage h_res
  rw [h_amount, h_percentage] at h_res
  norm_num at h_res
  exact h_res

end percentage_calculation_l2179_217923


namespace fraction_of_7000_l2179_217994

theorem fraction_of_7000 (x : ℝ) 
  (h1 : (1 / 10 / 100) * 7000 = 7) 
  (h2 : x * 7000 - 7 = 700) : 
  x = 0.101 :=
by
  sorry

end fraction_of_7000_l2179_217994


namespace turner_total_tickets_l2179_217929

-- Definition of conditions
def days := 3
def rollercoaster_rides_per_day := 3
def catapult_rides_per_day := 2
def ferris_wheel_rides_per_day := 1

def rollercoaster_ticket_cost := 4
def catapult_ticket_cost := 4
def ferris_wheel_ticket_cost := 1

-- Proof statement
theorem turner_total_tickets : 
  days * (rollercoaster_rides_per_day * rollercoaster_ticket_cost 
  + catapult_rides_per_day * catapult_ticket_cost 
  + ferris_wheel_rides_per_day * ferris_wheel_ticket_cost) 
  = 63 := 
by
  sorry

end turner_total_tickets_l2179_217929


namespace students_with_both_pets_l2179_217952

theorem students_with_both_pets
  (D C : Finset ℕ)
  (h_union : (D ∪ C).card = 48)
  (h_D : D.card = 30)
  (h_C : C.card = 34) :
  (D ∩ C).card = 16 :=
by sorry

end students_with_both_pets_l2179_217952


namespace valid_b_values_count_l2179_217987

theorem valid_b_values_count : 
  (∃! b : ℤ, ∃ x1 x2 x3 : ℤ, 
    (∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧ 
    (20 ≤ b^2 ∧ b^2 < 29)) :=
sorry

end valid_b_values_count_l2179_217987


namespace probability_in_dark_l2179_217959

theorem probability_in_dark (rev_per_min : ℕ) (given_prob : ℝ) (h1 : rev_per_min = 3) (h2 : given_prob = 0.25) :
  given_prob = 0.25 :=
by
  sorry

end probability_in_dark_l2179_217959


namespace complement_A_eq_interval_l2179_217964

-- Define the universal set U as the set of all real numbers.
def U : Set ℝ := Set.univ

-- Define the set A using the condition x^2 - 2x - 3 > 0.
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U.
def A_complement : Set ℝ := { x | -1 <= x ∧ x <= 3 }

theorem complement_A_eq_interval : A_complement = { x | -1 <= x ∧ x <= 3 } :=
by
  sorry

end complement_A_eq_interval_l2179_217964


namespace fourth_square_area_l2179_217933

theorem fourth_square_area (PQ QR RS QS : ℝ)
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 49)
  (h3 : RS^2 = 64) :
  QS^2 = 138 :=
by
  sorry

end fourth_square_area_l2179_217933


namespace equation_pattern_l2179_217985
open Nat

theorem equation_pattern (n : ℕ) (h_pos : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end equation_pattern_l2179_217985


namespace smallest_consecutive_divisible_by_17_l2179_217912

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_consecutive_divisible_by_17 :
  ∃ (n m : ℕ), 
    (m = n + 1) ∧
    sum_digits n % 17 = 0 ∧ 
    sum_digits m % 17 = 0 ∧ 
    n = 8899 ∧ 
    m = 8900 := 
by
  sorry

end smallest_consecutive_divisible_by_17_l2179_217912


namespace max_ways_to_ascend_and_descend_l2179_217996

theorem max_ways_to_ascend_and_descend :
  let east := 2
  let west := 3
  let south := 4
  let north := 1
  let ascend_descend_ways (ascend: ℕ) (n_1 n_2 n_3: ℕ) := ascend * (n_1 + n_2 + n_3)
  (ascend_descend_ways south east west north > ascend_descend_ways east west south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways west east south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways north east west south) := sorry

end max_ways_to_ascend_and_descend_l2179_217996


namespace incorrect_statement_S9_lt_S10_l2179_217956

variable {a : ℕ → ℝ} -- Sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms
variable {d : ℝ}     -- Common difference

-- Arithmetic sequence definition
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0 + n * (n-1) * d / 2)

-- Given conditions
variable 
  (arith_seq : arithmetic_sequence a d)
  (sum_terms : sum_of_first_n_terms a S)
  (H1 : S 9 < S 8)
  (H2 : S 8 = S 7)

-- Prove the statement
theorem incorrect_statement_S9_lt_S10 : 
  ¬ (S 9 < S 10) := 
sorry

end incorrect_statement_S9_lt_S10_l2179_217956


namespace exponentiation_problem_l2179_217922

theorem exponentiation_problem (a b : ℤ) (h : 3 ^ a * 9 ^ b = (1 / 3 : ℚ)) : a + 2 * b = -1 :=
sorry

end exponentiation_problem_l2179_217922


namespace sum_of_products_nonpos_l2179_217945

theorem sum_of_products_nonpos (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 :=
sorry

end sum_of_products_nonpos_l2179_217945


namespace seokgi_share_is_67_l2179_217951

-- The total length of the wire
def length_of_wire := 150

-- Seokgi's share is 16 cm shorter than Yeseul's share
def is_shorter_by (Y S : ℕ) := S = Y - 16

-- The sum of Yeseul's and Seokgi's shares equals the total length
def total_share (Y S : ℕ) := Y + S = length_of_wire

-- Prove that Seokgi's share is 67 cm
theorem seokgi_share_is_67 (Y S : ℕ) (h1 : is_shorter_by Y S) (h2 : total_share Y S) : 
  S = 67 :=
sorry

end seokgi_share_is_67_l2179_217951


namespace find_m_l2179_217905

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (4 - m) / (m - 2) = 2 * m) : 
  m = (3 + Real.sqrt 41) / 4 := by
  sorry

end find_m_l2179_217905


namespace complex_modulus_to_real_l2179_217915

theorem complex_modulus_to_real (a : ℝ) (h : (a + 1)^2 + (1 - a)^2 = 10) : a = 2 ∨ a = -2 :=
sorry

end complex_modulus_to_real_l2179_217915


namespace compute_expression_l2179_217947

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l2179_217947


namespace find_PF2_l2179_217932

-- Statement of the problem

def hyperbola_1 (x y: ℝ) := (x^2 / 16) - (y^2 / 20) = 1

theorem find_PF2 (x y PF1 PF2: ℝ) (a : ℝ)
    (h_hyperbola : hyperbola_1 x y)
    (h_a : a = 4) 
    (h_dist_PF1 : PF1 = 9) :
    abs (PF1 - PF2) = 2 * a → PF2 = 17 :=
by
  intro h1
  sorry

end find_PF2_l2179_217932


namespace investment_principal_l2179_217980

theorem investment_principal (A r : ℝ) (n t : ℕ) (P : ℝ) : 
  r = 0.07 → n = 4 → t = 5 → A = 60000 → 
  A = P * (1 + r / n)^(n * t) →
  P = 42409 :=
by
  sorry

end investment_principal_l2179_217980


namespace evaluate_expression_l2179_217950

theorem evaluate_expression : 
  (-2 : ℤ)^2004 + 3 * (-2)^2003 = (-2)^2003 :=
by
  sorry

end evaluate_expression_l2179_217950


namespace power_modulus_l2179_217954

theorem power_modulus (n : ℕ) : (2 : ℕ) ^ 345 % 5 = 2 :=
by sorry

end power_modulus_l2179_217954


namespace find_number_l2179_217909

theorem find_number (n : ℕ) (h1 : n % 5 = 0) (h2 : 70 ≤ n ∧ n ≤ 90) (h3 : Nat.Prime n) : n = 85 := 
sorry

end find_number_l2179_217909


namespace traders_gain_percentage_l2179_217931

theorem traders_gain_percentage (C : ℝ) (h : 0 < C) : 
  let cost_of_100_pens := 100 * C
  let gain := 40 * C
  let selling_price := cost_of_100_pens + gain
  let gain_percentage := (gain / cost_of_100_pens) * 100
  gain_percentage = 40 := by
  sorry

end traders_gain_percentage_l2179_217931


namespace boy_scouts_percentage_l2179_217997

variable (S B G : ℝ)

-- Conditions
-- Given B + G = S
axiom condition1 : B + G = S

-- Given 0.75B + 0.625G = 0.7S
axiom condition2 : 0.75 * B + 0.625 * G = 0.7 * S

-- Goal
theorem boy_scouts_percentage : B / S = 0.6 :=
by sorry

end boy_scouts_percentage_l2179_217997


namespace maximum_value_squared_l2179_217979

theorem maximum_value_squared (a b : ℝ) (h₁ : 0 < b) (h₂ : b ≤ a) :
  (∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
  (a / b)^2 ≤ 4 / 3 := 
sorry

end maximum_value_squared_l2179_217979


namespace smallest_b_undefined_inverse_l2179_217938

theorem smallest_b_undefined_inverse (b : ℕ) (h1 : Nat.gcd b 84 > 1) (h2 : Nat.gcd b 90 > 1) : b = 6 :=
sorry

end smallest_b_undefined_inverse_l2179_217938


namespace molecular_weight_of_Aluminium_hydroxide_l2179_217941

-- Given conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Definition of molecular weight of Aluminium hydroxide
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

-- Proof statement
theorem molecular_weight_of_Aluminium_hydroxide : molecular_weight_Al_OH_3 = 78.01 :=
  by sorry

end molecular_weight_of_Aluminium_hydroxide_l2179_217941


namespace hyperbola_center_l2179_217921

theorem hyperbola_center : ∃ c : ℝ × ℝ, c = (3, 5) ∧
  ∀ x y : ℝ, 9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 891 = 0 → (c.1 = 3 ∧ c.2 = 5) :=
by
  use (3, 5)
  sorry

end hyperbola_center_l2179_217921


namespace gallons_per_cubic_foot_l2179_217953

theorem gallons_per_cubic_foot (mix_per_pound : ℝ) (capacity_cubic_feet : ℕ) (weight_per_gallon : ℝ)
    (price_per_tbs : ℝ) (total_cost : ℝ) (total_gallons : ℝ) :
  mix_per_pound = 1.5 →
  capacity_cubic_feet = 6 →
  weight_per_gallon = 8 →
  price_per_tbs = 0.5 →
  total_cost = 270 →
  total_gallons = total_cost / (price_per_tbs * mix_per_pound * weight_per_gallon) →
  total_gallons / capacity_cubic_feet = 7.5 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h2, h6]
  sorry

end gallons_per_cubic_foot_l2179_217953


namespace integer_division_condition_l2179_217971

theorem integer_division_condition (n : ℕ) (h1 : n > 1): (∃ k : ℕ, 2^n + 1 = k * n^2) → n = 3 :=
by sorry

end integer_division_condition_l2179_217971


namespace initial_water_in_hole_l2179_217926

theorem initial_water_in_hole (total_needed additional_needed initial : ℕ) (h1 : total_needed = 823) (h2 : additional_needed = 147) :
  initial = total_needed - additional_needed :=
by
  sorry

end initial_water_in_hole_l2179_217926


namespace maria_remaining_towels_l2179_217986

-- Define the number of green towels Maria bought
def greenTowels : ℕ := 58

-- Define the number of white towels Maria bought
def whiteTowels : ℕ := 43

-- Define the total number of towels Maria bought
def totalTowels : ℕ := greenTowels + whiteTowels

-- Define the number of towels Maria gave to her mother
def towelsGiven : ℕ := 87

-- Define the resulting number of towels Maria has
def remainingTowels : ℕ := totalTowels - towelsGiven

-- Prove that the remaining number of towels is 14
theorem maria_remaining_towels : remainingTowels = 14 :=
by
  sorry

end maria_remaining_towels_l2179_217986


namespace minute_hand_rotation_l2179_217918

theorem minute_hand_rotation :
  (10 / 60) * (2 * Real.pi) = (- Real.pi / 3) :=
by
  sorry

end minute_hand_rotation_l2179_217918


namespace triangle_inequality_l2179_217990

theorem triangle_inequality (a b c m_A : ℝ)
  (h1 : 2*m_A ≤ b + c)
  (h2 : a^2 + (2*m_A)^2 = (b^2) + (c^2)) :
  a^2 + 4*m_A^2 ≤ (b + c)^2 :=
by {
  sorry
}

end triangle_inequality_l2179_217990


namespace eval_log32_4_l2179_217973

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end eval_log32_4_l2179_217973


namespace cost_of_one_shirt_l2179_217966

theorem cost_of_one_shirt (J S K : ℕ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) (h3 : 3 * J + 2 * S + K = 90) : S = 15 :=
by
  sorry

end cost_of_one_shirt_l2179_217966


namespace senior_tickets_count_l2179_217984

-- Define variables and problem conditions
variables (A S : ℕ)

-- Total number of tickets equation
def total_tickets (A S : ℕ) : Prop := A + S = 510

-- Total receipts equation
def total_receipts (A S : ℕ) : Prop := 21 * A + 15 * S = 8748

-- Prove that the number of senior citizen tickets S is 327
theorem senior_tickets_count (A S : ℕ) (h1 : total_tickets A S) (h2 : total_receipts A S) : S = 327 :=
sorry

end senior_tickets_count_l2179_217984


namespace cost_of_insulation_l2179_217943

def rectangular_tank_dimension_l : ℕ := 6
def rectangular_tank_dimension_w : ℕ := 3
def rectangular_tank_dimension_h : ℕ := 2
def total_cost : ℕ := 1440

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def cost_per_square_foot (total_cost surface_area : ℕ) : ℕ := total_cost / surface_area

theorem cost_of_insulation : 
  cost_per_square_foot total_cost (surface_area rectangular_tank_dimension_l rectangular_tank_dimension_w rectangular_tank_dimension_h) = 20 :=
by
  sorry

end cost_of_insulation_l2179_217943


namespace min_sum_of_dimensions_l2179_217948

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 2310) :
  a + b + c = 42 :=
sorry

end min_sum_of_dimensions_l2179_217948


namespace solution_set_of_inequality_l2179_217993

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) : 
  ((2 * x) / (x - 2) ≤ 1) ↔ (-2 ≤ x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l2179_217993


namespace base_satisfying_eq_l2179_217937

theorem base_satisfying_eq : ∃ a : ℕ, (11 < a) ∧ (293 * a^2 + 9 * a + 3 + (4 * a^2 + 6 * a + 8) = 7 * a^2 + 3 * a + 11) ∧ (a = 12) :=
by
  sorry

end base_satisfying_eq_l2179_217937


namespace probability_of_defective_product_is_0_032_l2179_217983

-- Defining the events and their probabilities
def P_H1 : ℝ := 0.30
def P_H2 : ℝ := 0.25
def P_H3 : ℝ := 0.45

-- Defining the probabilities of defects given each production line
def P_A_given_H1 : ℝ := 0.03
def P_A_given_H2 : ℝ := 0.02
def P_A_given_H3 : ℝ := 0.04

-- Summing up the total probabilities
def P_A : ℝ :=
  P_H1 * P_A_given_H1 +
  P_H2 * P_A_given_H2 +
  P_H3 * P_A_given_H3

-- The statement to be proven
theorem probability_of_defective_product_is_0_032 :
  P_A = 0.032 :=
by
  -- Proof would go here
  sorry

end probability_of_defective_product_is_0_032_l2179_217983


namespace total_toothpicks_480_l2179_217916

/- Define the number of toothpicks per side -/
def toothpicks_per_side : ℕ := 15

/- Define the number of horizontal lines in the grid -/
def horizontal_lines (sides : ℕ) : ℕ := sides + 1

/- Define the number of vertical lines in the grid -/
def vertical_lines (sides : ℕ) : ℕ := sides + 1

/- Define the total number of toothpicks used -/
def total_toothpicks (sides : ℕ) : ℕ :=
  (horizontal_lines sides * toothpicks_per_side) + (vertical_lines sides * toothpicks_per_side)

/- Theorem statement: Prove that for a grid with 15 toothpicks per side, the total number of toothpicks is 480 -/
theorem total_toothpicks_480 : total_toothpicks 15 = 480 :=
  sorry

end total_toothpicks_480_l2179_217916


namespace joan_spent_on_jacket_l2179_217936

def total_spent : ℝ := 42.33
def shorts_spent : ℝ := 15.00
def shirt_spent : ℝ := 12.51
def jacket_spent : ℝ := 14.82

theorem joan_spent_on_jacket :
  total_spent - shorts_spent - shirt_spent = jacket_spent :=
by
  sorry

end joan_spent_on_jacket_l2179_217936


namespace gcd_12_20_l2179_217965

theorem gcd_12_20 : Nat.gcd 12 20 = 4 := by
  sorry

end gcd_12_20_l2179_217965


namespace tommy_number_of_nickels_l2179_217977

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l2179_217977


namespace quadratic_has_real_root_l2179_217906

theorem quadratic_has_real_root (a b : ℝ) : ¬ (∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 := 
by
  sorry

end quadratic_has_real_root_l2179_217906


namespace isosceles_triangle_angle_condition_l2179_217982

theorem isosceles_triangle_angle_condition (A B C : ℝ) (h_iso : A = B) (h_angle_eq : A = 2 * C ∨ C = 2 * A) :
    (A = 45 ∨ A = 72) ∧ (B = 45 ∨ B = 72) :=
by
  -- Given isosceles triangle properties.
  sorry

end isosceles_triangle_angle_condition_l2179_217982


namespace solve_system_of_inequalities_l2179_217917

theorem solve_system_of_inequalities 
  (x : ℝ) 
  (h1 : x - 3 * (x - 2) ≥ 4)
  (h2 : (1 + 2 * x) / 3 > x - 1) : 
  x ≤ 1 := 
sorry

end solve_system_of_inequalities_l2179_217917


namespace common_points_intervals_l2179_217907

noncomputable def h (x : ℝ) : ℝ := (2 * Real.log x) / x

theorem common_points_intervals (a : ℝ) (h₀ : 1 < a) : 
  (∀ f g : ℝ → ℝ, (f x = a ^ x) → (g x = x ^ 2) → 
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ f x₃ = g x₃) → 
  a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end common_points_intervals_l2179_217907


namespace sin_double_angle_shifted_l2179_217974

theorem sin_double_angle_shifted (θ : ℝ) (h : Real.cos (θ + Real.pi) = - 1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = - 7 / 9 :=
by
  sorry

end sin_double_angle_shifted_l2179_217974


namespace sum_2016_eq_1008_l2179_217913

-- Define the arithmetic sequence {a_n} and the sum of the first n terms S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
variable (h_arith_seq : ∀ n m, a (n+1) - a n = a (m+1) - a m)
variable (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)

-- Additional conditions from the problem
variable (h_vector : a 4 + a 2013 = 1)

-- Goal: Prove that the sum of the first 2016 terms equals 1008
theorem sum_2016_eq_1008 : S 2016 = 1008 := by
  sorry

end sum_2016_eq_1008_l2179_217913


namespace brianna_marbles_lost_l2179_217981

theorem brianna_marbles_lost
  (total_marbles : ℕ)
  (remaining_marbles : ℕ)
  (L : ℕ)
  (gave_away : ℕ)
  (dog_ate : ℚ)
  (h1 : total_marbles = 24)
  (h2 : remaining_marbles = 10)
  (h3 : gave_away = 2 * L)
  (h4 : dog_ate = L / 2)
  (h5 : total_marbles - remaining_marbles = L + gave_away + dog_ate) : L = 4 := 
by
  sorry

end brianna_marbles_lost_l2179_217981


namespace factor_polynomial_l2179_217924

theorem factor_polynomial (x y : ℝ) : 
  x^4 + 4 * y^4 = (x^2 - 2 * x * y + 2 * y^2) * (x^2 + 2 * x * y + 2 * y^2) :=
by
  sorry

end factor_polynomial_l2179_217924


namespace luke_payments_difference_l2179_217978

theorem luke_payments_difference :
  let principal := 12000
  let rate := 0.08
  let years := 10
  let n_quarterly := 4
  let n_annually := 1
  let quarterly_rate := rate / n_quarterly
  let annually_rate := rate / n_annually
  let balance_plan1_5years := principal * (1 + quarterly_rate)^(n_quarterly * 5)
  let payment_plan1_5years := balance_plan1_5years / 3
  let remaining_balance_plan1_5years := balance_plan1_5years - payment_plan1_5years
  let final_balance_plan1_10years := remaining_balance_plan1_5years * (1 + quarterly_rate)^(n_quarterly * 5)
  let total_payment_plan1 := payment_plan1_5years + final_balance_plan1_10years
  let final_balance_plan2_10years := principal * (1 + annually_rate)^years
  (total_payment_plan1 - final_balance_plan2_10years).abs = 1022 :=
by
  sorry

end luke_payments_difference_l2179_217978


namespace solution_set_of_inequality_l2179_217935

variable (a b x : ℝ)
variable (h1 : ∀ x, ax + b > 0 ↔ 1 < x)

theorem solution_set_of_inequality : ∀ x, (ax + b) * (x - 2) < 0 ↔ (1 < x ∧ x < 2) :=
by sorry

end solution_set_of_inequality_l2179_217935


namespace find_counterfeit_coin_l2179_217901

-- Define the context of the problem
variables (coins : Fin 6 → ℝ) -- six coins represented as a function from Fin 6 to their weights
          (is_counterfeit : Fin 6 → Prop) -- a predicate indicating if the coin is counterfeit
          (real_weight : ℝ) -- the unknown weight of a real coin

-- Existence assertion for the counterfeit coin
axiom exists_counterfeit : ∃ x, is_counterfeit x

-- Define the total weights of coins 1&2 and 3&4
def weight_1_2 := coins 0 + coins 1
def weight_3_4 := coins 2 + coins 3

-- Statement of the problem
theorem find_counterfeit_coin :
  (weight_1_2 = weight_3_4 → (is_counterfeit 4 ∨ is_counterfeit 5)) ∧ 
  (weight_1_2 ≠ weight_3_4 → (is_counterfeit 0 ∨ is_counterfeit 1 ∨ is_counterfeit 2 ∨ is_counterfeit 3)) :=
sorry

end find_counterfeit_coin_l2179_217901


namespace find_r_in_parallelogram_l2179_217968

theorem find_r_in_parallelogram 
  (θ : ℝ) 
  (r : ℝ)
  (CAB DBA DBC ACB AOB : ℝ)
  (h1 : CAB = 3 * DBA)
  (h2 : DBC = 2 * DBA)
  (h3 : ACB = r * (t * AOB))
  (h4 : t = 4 / 3)
  (h5 : AOB = 180 - 2 * DBA)
  (h6 : ACB = 180 - 4 * DBA) :
  r = 1 / 3 :=
by
  sorry

end find_r_in_parallelogram_l2179_217968


namespace sqrt_mul_sqrt_eq_six_l2179_217992

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l2179_217992


namespace action_figures_per_shelf_l2179_217955

/-- Mike has 64 action figures he wants to display. If each shelf 
    in his room can hold a certain number of figures and he needs 8 
    shelves, prove that each shelf can hold 8 figures. -/
theorem action_figures_per_shelf :
  (64 / 8) = 8 :=
by
  sorry

end action_figures_per_shelf_l2179_217955


namespace total_pages_in_book_l2179_217939

theorem total_pages_in_book (x : ℕ) : 
  (x - (x / 6 + 8) - ((5 * x / 6 - 8) / 5 + 10) - ((4 * x / 6 - 18) / 4 + 12) = 72) → 
  x = 195 :=
by
  sorry

end total_pages_in_book_l2179_217939


namespace largest_possible_number_of_red_socks_l2179_217946

noncomputable def max_red_socks (t : ℕ) (r : ℕ) : Prop :=
  t ≤ 1991 ∧
  ((r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = 1 / 2) ∧
  ∀ r', r' ≤ 990 → (t ≤ 1991 ∧
    ((r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = 1 / 2) → r ≤ r')

theorem largest_possible_number_of_red_socks :
  ∃ t r, max_red_socks t r ∧ r = 990 :=
by
  sorry

end largest_possible_number_of_red_socks_l2179_217946


namespace total_cards_in_stack_l2179_217988

theorem total_cards_in_stack (n : ℕ) (H1: 252 ≤ 2 * n) (H2 : (2 * n) % 2 = 0)
                             (H3 : ∀ k : ℕ, k ≤ 2 * n → (if k % 2 = 0 then k / 2 else (k + 1) / 2) * 2 = k) :
  2 * n = 504 := sorry

end total_cards_in_stack_l2179_217988


namespace hexagon_coloring_count_l2179_217991

-- Defining the conditions
def has7Colors : Type := Fin 7

-- Hexagon vertices
inductive Vertex
| A | B | C | D | E | F

-- Adjacent vertices
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => true
| Vertex.B, Vertex.C => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.E => true
| Vertex.E, Vertex.F => true
| Vertex.F, Vertex.A => true
| _, _ => false

-- Non-adjacent vertices (diagonals)
def non_adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.D => true
| Vertex.B, Vertex.E => true
| Vertex.C, Vertex.E => true
| Vertex.C, Vertex.F => true
| Vertex.D, Vertex.F => true
| Vertex.E, Vertex.A => true
| Vertex.F, Vertex.A => true
| Vertex.F, Vertex.B => true
| _, _ => false

-- Coloring function
def valid_coloring (coloring : Vertex → has7Colors) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2, non_adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2 v3, adjacent v1 v2 → adjacent v2 v3 → adjacent v1 v3 → coloring v1 ≠ coloring v3)

noncomputable def count_valid_colorings : Nat :=
  -- This is a placeholder for the count function
  sorry

theorem hexagon_coloring_count : count_valid_colorings = 21000 := 
  sorry

end hexagon_coloring_count_l2179_217991


namespace gain_percent_is_correct_l2179_217989

theorem gain_percent_is_correct :
  let CP : ℝ := 450
  let SP : ℝ := 520
  let gain : ℝ := SP - CP
  let gain_percent : ℝ := (gain / CP) * 100
  gain_percent = 15.56 :=
by
  sorry

end gain_percent_is_correct_l2179_217989


namespace petya_digits_l2179_217998

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l2179_217998


namespace marshmallow_ratio_l2179_217914

theorem marshmallow_ratio:
  (∀ h m b, 
    h = 8 ∧ 
    m = 3 * h ∧ 
    h + m + b = 44
  ) → (1 / 2 = b / m) :=
by
sorry

end marshmallow_ratio_l2179_217914


namespace total_amount_l2179_217928

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l2179_217928


namespace find_m_b_l2179_217940

noncomputable def line_equation (x y : ℝ) :=
  (⟨-1, 4⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩ : ℝ × ℝ) = 0

theorem find_m_b : ∃ m b : ℝ, (∀ (x y : ℝ), line_equation x y → y = m * x + b) ∧ m = 1 / 4 ∧ b = -23 / 4 :=
by
  sorry

end find_m_b_l2179_217940


namespace max_product_of_two_integers_with_sum_2004_l2179_217942

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l2179_217942


namespace number_of_sets_l2179_217995

theorem number_of_sets (M : Set ℕ) : 
  {1, 2} ⊆ M → M ⊆ {1, 2, 3, 4} → ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_sets_l2179_217995


namespace largest_possible_b_l2179_217900

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
sorry

end largest_possible_b_l2179_217900


namespace watch_loss_percentage_l2179_217904

noncomputable def loss_percentage (CP SP_gain : ℝ) : ℝ :=
  100 * (CP - SP_gain) / CP

theorem watch_loss_percentage (CP : ℝ) (SP_gain : ℝ) :
  (SP_gain = CP + 0.04 * CP) →
  (CP = 700) →
  (CP - (SP_gain - 140) = CP * (16 / 100)) :=
by
  intros h_SP_gain h_CP
  rw [h_SP_gain, h_CP]
  simp
  sorry

end watch_loss_percentage_l2179_217904


namespace h_two_n_mul_h_2024_l2179_217910

variable {h : ℕ → ℝ}
variable {k : ℝ}
variable (n : ℕ) (k_ne_zero : k ≠ 0)

-- Condition 1: h(m + n) = h(m) * h(n)
axiom h_add_mul (m n : ℕ) : h (m + n) = h m * h n

-- Condition 2: h(2) = k
axiom h_two : h 2 = k

theorem h_two_n_mul_h_2024 : h (2 * n) * h 2024 = k^(n + 1012) := 
  sorry

end h_two_n_mul_h_2024_l2179_217910


namespace storks_more_than_birds_l2179_217958

def birds := 4
def initial_storks := 3
def additional_storks := 6

theorem storks_more_than_birds :
  (initial_storks + additional_storks) - birds = 5 := 
by
  sorry

end storks_more_than_birds_l2179_217958


namespace fixed_monthly_fee_l2179_217949

theorem fixed_monthly_fee (x y z : ℝ) 
  (h1 : x + y = 18.50) 
  (h2 : x + y + 3 * z = 23.45) : 
  x = 7.42 := 
by 
  sorry

end fixed_monthly_fee_l2179_217949


namespace disinfectant_usage_l2179_217967

theorem disinfectant_usage (x : ℝ) (hx1 : 0 < x) (hx2 : 120 / x / 2 = 120 / (x + 4)) : x = 4 :=
by
  sorry

end disinfectant_usage_l2179_217967


namespace customerPaidPercentGreater_l2179_217999

-- Definitions for the conditions
def costOfManufacture (C : ℝ) : ℝ := C
def designerPrice (C : ℝ) : ℝ := C * 1.40
def retailerTaxedPrice (C : ℝ) : ℝ := (C * 1.40) * 1.05
def customerInitialPrice (C : ℝ) : ℝ := ((C * 1.40) * 1.05) * 1.10
def customerFinalPrice (C : ℝ) : ℝ := (((C * 1.40) * 1.05) * 1.10) * 0.90

-- The theorem statement
theorem customerPaidPercentGreater (C : ℝ) (hC : 0 < C) : 
    (customerFinalPrice C - costOfManufacture C) / costOfManufacture C * 100 = 45.53 := by 
  sorry

end customerPaidPercentGreater_l2179_217999


namespace angle_quadrant_l2179_217911

theorem angle_quadrant 
  (θ : Real) 
  (h1 : Real.cos θ > 0) 
  (h2 : Real.sin (2 * θ) < 0) : 
  3 * π / 2 < θ ∧ θ < 2 * π := 
by
  sorry

end angle_quadrant_l2179_217911


namespace right_triangle_distance_midpoint_l2179_217908

noncomputable def distance_from_F_to_midpoint_DE
  (D E F : ℝ × ℝ)
  (right_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C) 
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)
  : ℝ :=
  if hD : (D.1 - E.1)^2 + (D.2 - E.2)^2 = DE^2 then
    if hF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = DF^2 then
      if hDE : DE = 15 then
        (15 / 2) --distance from F to midpoint of DE
      else
        0 -- This will never be executed since DE = 15 is a given condition
    else
      0 -- This will never be executed since DF = 9 is a given condition
  else
    0 -- This will never be executed since EF = 12 is a given condition

theorem right_triangle_distance_midpoint
  (D E F : ℝ × ℝ)
  (h_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C)
  (hDE : (D.1 - E.1)^2 + (D.2 - E.2)^2 = 15^2)
  (hDF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9^2)
  (hEF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 12^2) :
  distance_from_F_to_midpoint_DE D E F h_triangle 15 9 12 = 7.5 :=
by sorry

end right_triangle_distance_midpoint_l2179_217908


namespace percent_runs_by_running_eq_18_75_l2179_217976

/-
Define required conditions.
-/
def total_runs : ℕ := 224
def boundaries_runs : ℕ := 9 * 4
def sixes_runs : ℕ := 8 * 6
def twos_runs : ℕ := 12 * 2
def threes_runs : ℕ := 4 * 3
def byes_runs : ℕ := 6 * 1
def running_runs : ℕ := twos_runs + threes_runs + byes_runs

/-
Define the proof problem to show that the percentage of the total score made by running between the wickets is 18.75%.
-/
theorem percent_runs_by_running_eq_18_75 : (running_runs : ℚ) / total_runs * 100 = 18.75 := by
  sorry

end percent_runs_by_running_eq_18_75_l2179_217976


namespace server_processes_21600000_requests_l2179_217975

theorem server_processes_21600000_requests :
  (15000 * 1440 = 21600000) :=
by
  -- Calculations and step-by-step proof
  sorry

end server_processes_21600000_requests_l2179_217975


namespace count_3_digit_multiples_of_13_l2179_217969

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l2179_217969
