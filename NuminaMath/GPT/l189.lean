import Mathlib

namespace intersection_is_correct_l189_189201

def M : Set ℤ := {x | x^2 + 3 * x + 2 > 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_is_correct : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_is_correct_l189_189201


namespace steve_speed_back_l189_189364

open Real

noncomputable def steves_speed_on_way_back : ℝ := 15

theorem steve_speed_back
  (distance_to_work : ℝ)
  (traffic_time_to_work : ℝ)
  (traffic_time_back : ℝ)
  (total_time : ℝ)
  (speed_ratio : ℝ) :
  distance_to_work = 30 →
  traffic_time_to_work = 30 →
  traffic_time_back = 15 →
  total_time = 405 →
  speed_ratio = 2 →
  steves_speed_on_way_back = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end steve_speed_back_l189_189364


namespace speed_equivalence_l189_189872

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end speed_equivalence_l189_189872


namespace total_sodas_bought_l189_189210

-- Condition 1: Number of sodas they drank
def sodas_drank : ℕ := 3

-- Condition 2: Number of extra sodas Robin had
def sodas_extras : ℕ := 8

-- Mathematical equivalence we want to prove: Total number of sodas bought by Robin
theorem total_sodas_bought : sodas_drank + sodas_extras = 11 := by
  sorry

end total_sodas_bought_l189_189210


namespace simplify_and_evaluate_expression_l189_189499

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression_l189_189499


namespace minimum_soldiers_to_add_l189_189622

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189622


namespace number_of_diagonals_25_sides_l189_189888

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l189_189888


namespace max_gcd_lcm_eq_10_l189_189066

open Nat -- Opening the namespace for natural numbers

theorem max_gcd_lcm_eq_10
  (a b c : ℕ) 
  (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) :
  gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_eq_10_l189_189066


namespace Isaiah_types_more_l189_189971

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l189_189971


namespace range_of_x_l189_189436

theorem range_of_x (x : ℝ) : (|x + 1| + |x - 1| = 2) → (-1 ≤ x ∧ x ≤ 1) :=
by
  intro h
  sorry

end range_of_x_l189_189436


namespace supplement_of_supplement_l189_189711

def supplement (angle : ℝ) : ℝ :=
  180 - angle

theorem supplement_of_supplement (θ : ℝ) (h : θ = 35) : supplement (supplement θ) = 35 := by
  -- It is enough to state the theorem; the proof is not required as per the instruction.
  sorry

end supplement_of_supplement_l189_189711


namespace tan_subtraction_l189_189946

theorem tan_subtraction (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 :=
by
  sorry

end tan_subtraction_l189_189946


namespace parabola_vertex_l189_189503

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end parabola_vertex_l189_189503


namespace total_blue_balloons_l189_189196

def Joan_balloons : Nat := 9
def Sally_balloons : Nat := 5
def Jessica_balloons : Nat := 2

theorem total_blue_balloons : Joan_balloons + Sally_balloons + Jessica_balloons = 16 :=
by
  sorry

end total_blue_balloons_l189_189196


namespace probability_at_least_6_heads_in_10_flips_l189_189250

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l189_189250


namespace ones_digit_of_p_is_3_l189_189566

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l189_189566


namespace quadratic_roots_real_or_imaginary_l189_189774

theorem quadratic_roots_real_or_imaginary (a b c d: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) 
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
∃ (A B C: ℝ), (A = a ∨ A = b ∨ A = c ∨ A = d) ∧ (B = a ∨ B = b ∨ B = c ∨ B = d) ∧ (C = a ∨ C = b ∨ C = c ∨ C = d) ∧ 
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ 
((1 - 4*B*C ≥ 0 ∧ 1 - 4*C*A ≥ 0 ∧ 1 - 4*A*B ≥ 0) ∨ (1 - 4*B*C < 0 ∧ 1 - 4*C*A < 0 ∧ 1 - 4*A*B < 0)) :=
by
  sorry

end quadratic_roots_real_or_imaginary_l189_189774


namespace pay_nineteen_rubles_l189_189411

/-- 
Given a purchase cost of 19 rubles, a customer with only three-ruble bills, 
and a cashier with only five-ruble bills, both having 15 bills each,
prove that it is possible for the customer to pay exactly 19 rubles.
-/
theorem pay_nineteen_rubles (purchase_cost : ℕ) (customer_bills cashier_bills : ℕ) 
  (customer_denomination cashier_denomination : ℕ) (customer_count cashier_count : ℕ) :
  purchase_cost = 19 →
  customer_denomination = 3 →
  cashier_denomination = 5 →
  customer_count = 15 →
  cashier_count = 15 →
  (∃ m n : ℕ, m * customer_denomination - n * cashier_denomination = purchase_cost 
  ∧ m ≤ customer_count ∧ n ≤ cashier_count) :=
by
  intros
  sorry

end pay_nineteen_rubles_l189_189411


namespace intersect_x_axis_unique_l189_189379

theorem intersect_x_axis_unique (a : ℝ) : (∀ x, (ax^2 + (3 - a) * x + 1) = 0 → x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersect_x_axis_unique_l189_189379


namespace intersection_of_lines_l189_189397

theorem intersection_of_lines :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 2 * y = -7 * x - 2 ∧ x = -18 / 17 ∧ y = 46 / 17 :=
by
  sorry

end intersection_of_lines_l189_189397


namespace find_real_num_l189_189803

noncomputable def com_num (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

theorem find_real_num (a : ℝ) : (∃ b : ℝ, com_num a = b * Complex.I) → a = -6 :=
by
  sorry

end find_real_num_l189_189803


namespace younger_son_age_after_30_years_l189_189686

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l189_189686


namespace smallest_prime_dividing_4_pow_11_plus_6_pow_13_l189_189717

-- Definition of the problem
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem smallest_prime_dividing_4_pow_11_plus_6_pow_13 :
  ∃ p : ℕ, is_prime p ∧ p ∣ (4^11 + 6^13) ∧ ∀ q : ℕ, is_prime q ∧ q ∣ (4^11 + 6^13) → p ≤ q :=
by {
  sorry
}

end smallest_prime_dividing_4_pow_11_plus_6_pow_13_l189_189717


namespace GlobalConnect_more_cost_effective_if_x_300_l189_189117

def GlobalConnectCost (x : ℕ) : ℝ := 50 + 0.4 * x
def QuickConnectCost (x : ℕ) : ℝ := 0.6 * x

theorem GlobalConnect_more_cost_effective_if_x_300 : 
  GlobalConnectCost 300 < QuickConnectCost 300 :=
by
  sorry

end GlobalConnect_more_cost_effective_if_x_300_l189_189117


namespace convex_polygon_diagonals_l189_189885

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l189_189885


namespace probability_not_snowing_l189_189990

variable (P_snowing : ℚ)
variable (h : P_snowing = 2/5)

theorem probability_not_snowing (P_not_snowing : ℚ) : 
  P_not_snowing = 3 / 5 :=
by 
  -- sorry to skip the proof
  sorry

end probability_not_snowing_l189_189990


namespace inverse_function_value_l189_189035

def g (x : ℝ) : ℝ := 4 * x ^ 3 - 5

theorem inverse_function_value (x : ℝ) : g x = -1 ↔ x = 1 :=
by
  sorry

end inverse_function_value_l189_189035


namespace quadratic_equation_divisible_by_x_minus_one_l189_189439

theorem quadratic_equation_divisible_by_x_minus_one (a b c : ℝ) (h1 : (x - 1) ∣ (a * x * x + b * x + c)) (h2 : c = 2) :
  (a = 1 ∧ b = -3 ∧ c = 2) → a * x * x + b * x + c = x^2 - 3 * x + 2 :=
by
  sorry

end quadratic_equation_divisible_by_x_minus_one_l189_189439


namespace percent_of_a_is_4b_l189_189980

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : (4 * b) / a = 20 / 9 :=
by sorry

end percent_of_a_is_4b_l189_189980


namespace factories_checked_by_second_group_l189_189425

theorem factories_checked_by_second_group 
(T : ℕ) (G1 : ℕ) (R : ℕ) 
(hT : T = 169) 
(hG1 : G1 = 69) 
(hR : R = 48) : 
T - (G1 + R) = 52 :=
by {
  sorry
}

end factories_checked_by_second_group_l189_189425


namespace tomatoes_left_l189_189392

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l189_189392


namespace cyclic_cosine_inequality_l189_189349

theorem cyclic_cosine_inequality
  (α β γ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (hγ : 0 ≤ γ ∧ γ ≤ π / 2)
  (cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4 ∧
    (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4
      ≤ (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) :=
by 
  sorry

end cyclic_cosine_inequality_l189_189349


namespace inequality_solution_set_l189_189853

theorem inequality_solution_set (x : ℝ) : (x^2 ≥ 4) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l189_189853


namespace hammer_nail_cost_l189_189003

variable (h n : ℝ)

theorem hammer_nail_cost (h n : ℝ)
    (h1 : 4 * h + 5 * n = 10.45)
    (h2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 :=
sorry

end hammer_nail_cost_l189_189003


namespace average_remaining_two_l189_189105

theorem average_remaining_two (a b c d e : ℝ) 
  (h1 : (a + b + c + d + e) / 5 = 12) 
  (h2 : (a + b + c) / 3 = 4) : 
  (d + e) / 2 = 24 :=
by 
  sorry

end average_remaining_two_l189_189105


namespace arithmetic_sequence_a5_l189_189051

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : 
  a 5 = 7 :=
by
  -- proof to be filled later
  sorry

end arithmetic_sequence_a5_l189_189051


namespace max_value_m_l189_189166

theorem max_value_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2 * x - 1 < m) → m ≤ 5 :=
by
  sorry

end max_value_m_l189_189166


namespace solution_set_of_inequality_l189_189697

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l189_189697


namespace find_S17_l189_189028

-- Definitions based on the conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (a1 : ℝ) (d : ℝ)

-- Conditions from the problem restated in Lean
axiom arithmetic_sequence : ∀ n, a n = a1 + (n - 1) * d
axiom sum_of_n_terms : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)
axiom arithmetic_subseq : 2 * a 7 = a 5 + 3

-- Theorem to prove
theorem find_S17 : S 17 = 51 :=
by sorry

end find_S17_l189_189028


namespace complement_of_A_is_negatives_l189_189999

theorem complement_of_A_is_negatives :
  let U := Set.univ (α := ℝ)
  let A := {x : ℝ | x ≥ 0}
  (U \ A) = {x : ℝ | x < 0} :=
by
  sorry

end complement_of_A_is_negatives_l189_189999


namespace initially_calculated_average_weight_l189_189842

theorem initially_calculated_average_weight 
  (A : ℚ)
  (h1 : ∀ sum_weight_corr : ℚ, sum_weight_corr = 20 * 58.65)
  (h2 : ∀ misread_weight_corr : ℚ, misread_weight_corr = 56)
  (h3 : ∀ correct_weight_corr : ℚ, correct_weight_corr = 61)
  (h4 : (20 * A + (correct_weight_corr - misread_weight_corr)) = 20 * 58.65) :
  A = 58.4 := 
sorry

end initially_calculated_average_weight_l189_189842


namespace coordinates_B_l189_189449

theorem coordinates_B (A B : ℝ × ℝ) (distance : ℝ) (A_coords : A = (-1, 3)) 
  (AB_parallel_x : A.snd = B.snd) (AB_distance : abs (A.fst - B.fst) = distance) :
  (B = (-6, 3) ∨ B = (4, 3)) :=
by
  sorry

end coordinates_B_l189_189449


namespace sum_of_gcd_and_lcm_l189_189863

-- Definitions of gcd and lcm for the conditions
def gcd_of_42_and_56 : ℕ := Nat.gcd 42 56
def lcm_of_24_and_18 : ℕ := Nat.lcm 24 18

-- Lean statement that the sum of the gcd and lcm is 86
theorem sum_of_gcd_and_lcm : gcd_of_42_and_56 + lcm_of_24_and_18 = 86 := by
  sorry

end sum_of_gcd_and_lcm_l189_189863


namespace volume_of_snow_l189_189409

theorem volume_of_snow (L W H : ℝ) (hL : L = 30) (hW : W = 3) (hH : H = 0.75) :
  L * W * H = 67.5 := by
  sorry

end volume_of_snow_l189_189409


namespace curve_symmetry_l189_189235

-- Define the curve as a predicate
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0

-- Define the point symmetry condition for a line
def is_symmetric_about_line (curve : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve x y → line x y

-- Define the line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Main theorem stating the curve is symmetrical about the line x + y = 0
theorem curve_symmetry : is_symmetric_about_line curve line_x_plus_y_eq_0 := 
sorry

end curve_symmetry_l189_189235


namespace fractional_part_tiled_l189_189285

def room_length : ℕ := 12
def room_width : ℕ := 20
def number_of_tiles : ℕ := 40
def tile_area : ℕ := 1

theorem fractional_part_tiled :
  (number_of_tiles * tile_area : ℚ) / (room_length * room_width) = 1 / 6 :=
by
  sorry

end fractional_part_tiled_l189_189285


namespace range_of_log2_sin_squared_l189_189142

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def sin_squared_log_range (x : ℝ) : ℝ :=
  log2 ((Real.sin x) ^ 2)

theorem range_of_log2_sin_squared (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  ∃ y, y = sin_squared_log_range x ∧ y ≤ 0 :=
by
  sorry

end range_of_log2_sin_squared_l189_189142


namespace tires_sale_price_l189_189907

variable (n : ℕ)
variable (t p_original p_sale : ℝ)

theorem tires_sale_price
  (h₁ : n = 4)
  (h₂ : t = 36)
  (h₃ : p_original = 84)
  (h₄ : p_sale = p_original - t / n) :
  p_sale = 75 := by
  sorry

end tires_sale_price_l189_189907


namespace bus_people_next_pickup_point_l189_189112

theorem bus_people_next_pickup_point (bus_capacity : ℕ) (fraction_first_pickup : ℚ) (cannot_board : ℕ)
  (h1 : bus_capacity = 80)
  (h2 : fraction_first_pickup = 3 / 5)
  (h3 : cannot_board = 18) : 
  ∃ people_next_pickup : ℕ, people_next_pickup = 50 :=
by
  sorry

end bus_people_next_pickup_point_l189_189112


namespace union_of_setA_and_setB_l189_189484

def setA : Set ℕ := {1, 2, 4}
def setB : Set ℕ := {2, 6}

theorem union_of_setA_and_setB :
  setA ∪ setB = {1, 2, 4, 6} :=
by sorry

end union_of_setA_and_setB_l189_189484


namespace two_pow_n_minus_one_divisible_by_seven_iff_l189_189239

theorem two_pow_n_minus_one_divisible_by_seven_iff (n : ℕ) (h : n > 0) :
  (2^n - 1) % 7 = 0 ↔ n % 3 = 0 :=
sorry

end two_pow_n_minus_one_divisible_by_seven_iff_l189_189239


namespace marshmallow_total_l189_189929

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end marshmallow_total_l189_189929


namespace eventually_periodic_sequence_l189_189058

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n m : ℕ, 0 < n → 0 < m → a (n + 2 * m) ∣ (a n + a (n + m)))
  : ∃ N d : ℕ, 0 < N ∧ 0 < d ∧ ∀ n > N, a n = a (n + d) :=
sorry

end eventually_periodic_sequence_l189_189058


namespace Isaiah_types_more_l189_189970

theorem Isaiah_types_more (Micah_rate Isaiah_rate : ℕ) (h_Micah : Micah_rate = 20) (h_Isaiah : Isaiah_rate = 40) :
  (Isaiah_rate * 60 - Micah_rate * 60) = 1200 :=
by
  -- Here we assume we need to prove this theorem
  sorry

end Isaiah_types_more_l189_189970


namespace total_wrappers_l189_189130

theorem total_wrappers (a m : ℕ) (ha : a = 34) (hm : m = 15) : a + m = 49 :=
by
  sorry

end total_wrappers_l189_189130


namespace urn_probability_four_each_l189_189882

def number_of_sequences := Nat.choose 6 3

def probability_of_sequence := (1/3) * (1/2) * (3/5) * (1/2) * (4/7) * (5/8)

def total_probability := number_of_sequences * probability_of_sequence

theorem urn_probability_four_each :
  total_probability = 5 / 14 := by
  -- proof goes here
  sorry

end urn_probability_four_each_l189_189882


namespace jacob_twice_as_old_l189_189814

theorem jacob_twice_as_old (x : ℕ) : 18 + x = 2 * (9 + x) → x = 0 := by
  intro h
  linarith

end jacob_twice_as_old_l189_189814


namespace solve_students_in_fifth_grade_class_l189_189829

noncomputable def number_of_students_in_each_fifth_grade_class 
    (third_grade_classes : ℕ) 
    (third_grade_students_per_class : ℕ)
    (fourth_grade_classes : ℕ) 
    (fourth_grade_students_per_class : ℕ) 
    (fifth_grade_classes : ℕ)
    (total_lunch_cost : ℝ)
    (hamburger_cost : ℝ)
    (carrot_cost : ℝ)
    (cookie_cost : ℝ) : ℝ :=
  
  let total_students_third := third_grade_classes * third_grade_students_per_class
  let total_students_fourth := fourth_grade_classes * fourth_grade_students_per_class
  let lunch_cost_per_student := hamburger_cost + carrot_cost + cookie_cost
  let total_students := total_students_third + total_students_fourth
  let total_cost_third_fourth := total_students * lunch_cost_per_student
  let total_cost_fifth := total_lunch_cost - total_cost_third_fourth
  let fifth_grade_students := total_cost_fifth / lunch_cost_per_student
  let students_per_fifth_class := fifth_grade_students / fifth_grade_classes
  students_per_fifth_class

theorem solve_students_in_fifth_grade_class : 
    number_of_students_in_each_fifth_grade_class 5 30 4 28 4 1036 2.10 0.50 0.20 = 27 := 
by 
  sorry

end solve_students_in_fifth_grade_class_l189_189829


namespace eggs_per_week_is_84_l189_189473

-- Define the number of pens
def number_of_pens : Nat := 4

-- Define the number of emus per pen
def emus_per_pen : Nat := 6

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Define the number of eggs per female emu per day
def eggs_per_female_emu_per_day : Nat := 1

-- Calculate the total number of emus
def total_emus : Nat := number_of_pens * emus_per_pen

-- Calculate the number of female emus
def female_emus : Nat := total_emus / 2

-- Calculate the number of eggs per day
def eggs_per_day : Nat := female_emus * eggs_per_female_emu_per_day

-- Calculate the number of eggs per week
def eggs_per_week : Nat := eggs_per_day * days_in_week

-- The theorem to prove
theorem eggs_per_week_is_84 : eggs_per_week = 84 := by
  sorry

end eggs_per_week_is_84_l189_189473


namespace maximize_profit_l189_189731

variable {k : ℝ} (hk : k > 0)
variable {x : ℝ} (hx : 0 < x ∧ x < 0.06)

def deposit_volume (x : ℝ) : ℝ := k * x
def interest_paid (x : ℝ) : ℝ := k * x ^ 2
def profit (x : ℝ) : ℝ := (0.06 * k^2 * x) - (k * x^2)

theorem maximize_profit : 0.03 = x :=
by
  sorry

end maximize_profit_l189_189731


namespace younger_son_age_in_30_years_l189_189690

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l189_189690


namespace total_wet_surface_area_is_62_l189_189540

-- Define the dimensions of the cistern
def length_cistern : ℝ := 8
def width_cistern : ℝ := 4
def depth_water : ℝ := 1.25

-- Define the calculation of the wet surface area
def bottom_surface_area : ℝ := length_cistern * width_cistern
def longer_side_surface_area : ℝ := length_cistern * depth_water * 2
def shorter_end_surface_area : ℝ := width_cistern * depth_water * 2

-- Sum up all wet surface areas
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_surface_area + shorter_end_surface_area

-- The theorem stating that the total wet surface area is 62 m²
theorem total_wet_surface_area_is_62 : total_wet_surface_area = 62 := by
  sorry

end total_wet_surface_area_is_62_l189_189540


namespace university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l189_189362

-- Part 1
def probability_A_exactly_one_subject : ℚ :=
  3 * (1/2) * (1/2)^2

def probability_B_exactly_one_subject (m : ℚ) : ℚ :=
  (1/6) * (2/5)^2 + (5/6) * (3/5) * (2/5) * 2

theorem university_A_pass_one_subject : probability_A_exactly_one_subject = 3/8 :=
sorry

theorem university_B_pass_one_subject_when_m_3_5 : probability_B_exactly_one_subject (3/5) = 32/75 :=
sorry

-- Part 2
def expected_A : ℚ :=
  3 * (1/2)

def expected_B (m : ℚ) : ℚ :=
  ((17 - 7 * m) / 30) + (2 * (3 + 14 * m) / 30) + (3 * m / 10)

theorem preferred_range_of_m : 0 < m ∧ m < 11/15 → expected_A > expected_B m :=
sorry

end university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l189_189362


namespace lines_are_parallel_and_not_coincident_l189_189515

theorem lines_are_parallel_and_not_coincident (a : ℝ) :
  (a * (a - 1) - 3 * 2 = 0) ∧ (3 * (a - 7) - a * 3 * a ≠ 0) ↔ a = 3 :=
by
  sorry

end lines_are_parallel_and_not_coincident_l189_189515


namespace range_of_a_l189_189441

variable {x a : ℝ}

def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := |x| > a

theorem range_of_a (h : ¬p x → ¬q x a) : a ≤ 1 :=
sorry

end range_of_a_l189_189441


namespace negation_proof_l189_189694

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 2 * x)) ↔ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proof_l189_189694


namespace string_cheese_packages_l189_189476

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end string_cheese_packages_l189_189476


namespace sufficient_but_not_necessary_condition_ellipse_l189_189992

theorem sufficient_but_not_necessary_condition_ellipse (a : ℝ) :
  (a^2 > 1 → ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1)) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → (a^2 > 1 ∨ 0 < a^2 ∧ a^2 < 1)) → ¬ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_ellipse_l189_189992


namespace largest_of_seven_consecutive_odd_numbers_l189_189373

theorem largest_of_seven_consecutive_odd_numbers (a b c d e f g : ℤ) 
  (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) 
  (h5: e % 2 = 1) (h6: f % 2 = 1) (h7: g % 2 = 1)
  (h8 : a + b + c + d + e + f + g = 105)
  (h9 : b = a + 2) (h10 : c = a + 4) (h11 : d = a + 6)
  (h12 : e = a + 8) (h13 : f = a + 10) (h14 : g = a + 12) :
  g = 21 :=
by 
  sorry

end largest_of_seven_consecutive_odd_numbers_l189_189373


namespace xy_sum_l189_189799

variable (x y : ℚ)

theorem xy_sum : (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 := by
  intros h1 h2
  sorry

end xy_sum_l189_189799


namespace needed_correct_to_pass_l189_189271

def total_questions : Nat := 120
def genetics_questions : Nat := 20
def ecology_questions : Nat := 50
def evolution_questions : Nat := 50

def correct_genetics : Nat := (60 * genetics_questions) / 100
def correct_ecology : Nat := (50 * ecology_questions) / 100
def correct_evolution : Nat := (70 * evolution_questions) / 100
def total_correct : Nat := correct_genetics + correct_ecology + correct_evolution

def passing_rate : Nat := 65
def passing_score : Nat := (passing_rate * total_questions) / 100

theorem needed_correct_to_pass : (passing_score - total_correct) = 6 := 
by
  sorry

end needed_correct_to_pass_l189_189271


namespace find_recip_sum_of_shifted_roots_l189_189352

noncomputable def reciprocal_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) : ℝ :=
  1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2)

theorem find_recip_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) :
  reciprocal_sum_of_shifted_roots α β γ hαβγ = -19 / 14 :=
  sorry

end find_recip_sum_of_shifted_roots_l189_189352


namespace remove_candies_even_distribution_l189_189371

theorem remove_candies_even_distribution (candies friends : ℕ) (h_candies : candies = 30) (h_friends : friends = 4) :
  ∃ k, candies - k % friends = 0 ∧ k = 2 :=
by
  sorry

end remove_candies_even_distribution_l189_189371


namespace find_4a_3b_l189_189891

noncomputable def g (x : ℝ) : ℝ := 4 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := g x + 2

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_4a_3b (a b : ℝ) (h_inv : ∀ x : ℝ, f (f_inv x) a b = x) : 4 * a + 3 * b = 4 :=
by
  -- Proof skipped for now
  sorry

end find_4a_3b_l189_189891


namespace average_rate_of_change_interval_l189_189234

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

theorem average_rate_of_change_interval (f : ℝ → ℝ) (x₀ x₁ : ℝ) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  sorry

end average_rate_of_change_interval_l189_189234


namespace max_expr_value_l189_189695

theorem max_expr_value (a b c d : ℝ) (h_a : -8.5 ≤ a ∧ a ≤ 8.5)
                       (h_b : -8.5 ≤ b ∧ b ≤ 8.5)
                       (h_c : -8.5 ≤ c ∧ c ≤ 8.5)
                       (h_d : -8.5 ≤ d ∧ d ≤ 8.5) :
                       a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 306 :=
sorry

end max_expr_value_l189_189695


namespace ones_digit_of_prime_p_l189_189570

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l189_189570


namespace probability_exactly_four_even_out_of_eight_l189_189015

theorem probability_exactly_four_even_out_of_eight :
  (nat.choose 8 4 * (2/3)^4 * (1/3)^4) = (1120 / 6561) := sorry

end probability_exactly_four_even_out_of_eight_l189_189015


namespace price_of_computer_and_desk_l189_189704

theorem price_of_computer_and_desk (x y : ℕ) 
  (h1 : 10 * x + 200 * y = 90000)
  (h2 : 12 * x + 120 * y = 90000) : 
  x = 6000 ∧ y = 150 :=
by
  sorry

end price_of_computer_and_desk_l189_189704


namespace sequence_problem_l189_189316

theorem sequence_problem (a : ℕ → ℝ) (pos_terms : ∀ n, a n > 0)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, (a n + 1) * a (n + 2) = 1)
  (h2 : a 2 = a 6) :
  a 11 + a 12 = (11 / 18) + ((Real.sqrt 5 - 1) / 2) := by
  sorry

end sequence_problem_l189_189316


namespace packages_needed_l189_189475

/-- Kelly puts string cheeses in her kids' lunches 5 days per week. Her oldest wants 2 every day and her youngest will only eat 1.
The packages come with 30 string cheeses per pack. Prove that Kelly will need 2 packages of string cheese to fill her kids' lunches for 4 weeks. -/
theorem packages_needed (days_per_week : ℕ) (oldest_per_day : ℕ) (youngest_per_day : ℕ) (package_size : ℕ) (weeks : ℕ) :
  days_per_week = 5 →
  oldest_per_day = 2 →
  youngest_per_day = 1 →
  package_size = 30 →
  weeks = 4 →
  (2 * days_per_week + 1 * days_per_week) * weeks / package_size = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end packages_needed_l189_189475


namespace side_length_is_prime_l189_189520

-- Define the integer side length of the square
variable (a : ℕ)

-- Define the conditions
def impossible_rectangle (m n : ℕ) : Prop :=
  m * n = a^2 ∧ m ≠ 1 ∧ n ≠ 1

-- Declare the theorem to be proved
theorem side_length_is_prime (h : ∀ m n : ℕ, impossible_rectangle a m n → false) : Nat.Prime a := sorry

end side_length_is_prime_l189_189520


namespace candy_bar_reduction_l189_189734

variable (W P x : ℝ)
noncomputable def percent_reduction := (x / W) * 100

theorem candy_bar_reduction (h_weight_reduced : W > 0) 
                            (h_price_same : P > 0) 
                            (h_price_increase : P / (W - x) = (5 / 3) * (P / W)) :
    percent_reduction W x = 40 := 
sorry

end candy_bar_reduction_l189_189734


namespace fraction_numerator_greater_than_denominator_l189_189517

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5 / 3 → (8 / 11 < x ∧ x < 5 / 3) ∨ (5 / 3 < x ∧ x ≤ 3) ↔ (8 * x - 3 > 5 - 3 * x) := by
  sorry

end fraction_numerator_greater_than_denominator_l189_189517


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l189_189253

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l189_189253


namespace arithmetic_sequence_term_l189_189650

theorem arithmetic_sequence_term {a : ℕ → ℤ} 
  (h1 : a 4 = -4) 
  (h2 : a 8 = 4) : 
  a 12 = 12 := 
by 
  sorry

end arithmetic_sequence_term_l189_189650


namespace ab_bd_ratio_l189_189150

-- Definitions based on the conditions
variables {A B C D : ℝ}
variables (h1 : A / B = 1 / 2) (h2 : B / C = 8 / 5)

-- Math equivalence proving AB/BD = 4/13 based on given conditions
theorem ab_bd_ratio
  (h1 : A / B = 1 / 2)
  (h2 : B / C = 8 / 5) :
  A / (B + C) = 4 / 13 :=
by
  sorry

end ab_bd_ratio_l189_189150


namespace digits_right_of_decimal_l189_189037

theorem digits_right_of_decimal : 
  ∃ n : ℕ, (3^6 : ℚ) / ((6^4 : ℚ) * 625) = 9 * 10^(-4 : ℤ) ∧ n = 4 := 
by 
  sorry

end digits_right_of_decimal_l189_189037


namespace derivative_of_f_l189_189983

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) : (deriv f x) = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by
  -- This statement skips the proof details
  sorry

end derivative_of_f_l189_189983


namespace total_daisies_l189_189340

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l189_189340


namespace probability_of_both_gender_selection_l189_189880

noncomputable def probability_both_gender_selected : ℚ :=
  let total_ways := (Nat.choose 8 5) in
  let male_ways := (Nat.choose 5 5) in
  let prob_only_males := male_ways / total_ways in
  1 - prob_only_males

theorem probability_of_both_gender_selection :
  probability_both_gender_selected = 55 / 56 := 
sorry

end probability_of_both_gender_selection_l189_189880


namespace expand_and_simplify_l189_189286

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5 * x^2 + 4 * x - 20 := 
sorry

end expand_and_simplify_l189_189286


namespace frank_peanuts_average_l189_189767

theorem frank_peanuts_average :
  let one_dollar := 7 * 1
  let five_dollar := 4 * 5
  let ten_dollar := 2 * 10
  let twenty_dollar := 1 * 20
  let total_money := one_dollar + five_dollar + ten_dollar + twenty_dollar
  let change := 4
  let money_spent := total_money - change
  let cost_per_pound := 3
  let total_pounds := money_spent / cost_per_pound
  let days := 7
  let average_per_day := total_pounds / days
  average_per_day = 3 :=
by
  sorry

end frank_peanuts_average_l189_189767


namespace count_multiples_4_6_10_less_300_l189_189591

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l189_189591


namespace jackson_money_l189_189658

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l189_189658


namespace find_divisor_l189_189146

theorem find_divisor (n x : ℕ) (hx : x ≠ 11) (hn : n = 386) 
  (h1 : ∃ k : ℤ, n = k * x + 1) (h2 : ∀ m : ℤ, n = 11 * m + 1 → n = 386) : x = 5 :=
  sorry

end find_divisor_l189_189146


namespace arithmetic_prog_sum_bound_l189_189243

noncomputable def Sn (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_prog_sum_bound (n : ℕ) (a1 an : ℝ) (d : ℝ) (h_d_neg : d < 0) 
  (ha_n : an = a1 + (n - 1) * d) :
  n * an < Sn n a1 d ∧ Sn n a1 d < n * a1 :=
by 
  sorry

end arithmetic_prog_sum_bound_l189_189243


namespace number_divisible_by_33_l189_189706

theorem number_divisible_by_33 (x y : ℕ) 
  (h1 : (x + y) % 3 = 2) 
  (h2 : (y - x) % 11 = 8) : 
  (27850 + 1000 * x + y) % 33 = 0 := 
sorry

end number_divisible_by_33_l189_189706


namespace quadratic_equation_solution_l189_189149

theorem quadratic_equation_solution (m : ℝ) (h : m ≠ 1) : 
  (m^2 - 3 * m + 2 = 0) → m = 2 :=
by
  sorry

end quadratic_equation_solution_l189_189149


namespace poly_a_c_sum_l189_189483

theorem poly_a_c_sum {a b c d : ℝ} (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + a * x + b)
  (hg : ∀ x, g x = x^2 + c * x + d)
  (hv_f_root_g : g (-a / 2) = 0)
  (hv_g_root_f : f (-c / 2) = 0)
  (f_min : ∀ x, f x ≥ -25)
  (g_min : ∀ x, g x ≥ -25)
  (f_g_intersect : f 50 = -25 ∧ g 50 = -25) : a + c = -101 :=
by
  sorry

end poly_a_c_sum_l189_189483


namespace heart_and_face_card_probability_l189_189393

noncomputable def probability_heart_face_card : ℚ :=
  -- step-1: Calculate respective probabilities and sum them as in the given solution
  let P_ace_of_hearts_first := (1 / 52) * (11 / 51)
  let P_heart_not_ace_first := (12 / 52) * (12 / 51)
  P_ace_of_hearts_first + P_heart_not_ace_first

theorem heart_and_face_card_probability :
  probability_heart_face_card = 5 / 86 :=
begin
  sorry
end

end heart_and_face_card_probability_l189_189393


namespace zero_of_f_inequality_l189_189109

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - Real.log (x^3 + 1)

variable (a b c x : ℝ)
variable (h : 0 < a ∧ a < b ∧ b < c)
variable (hx : f x = 0)
variable (h₀ : f a * f b * f c < 0)

theorem zero_of_f_inequality :
  ¬ (x > c) :=
by 
  sorry

end zero_of_f_inequality_l189_189109


namespace candy_distribution_l189_189896

-- Definition of the problem
def emily_candies : ℕ := 30
def friends : ℕ := 4

-- Lean statement to prove
theorem candy_distribution : emily_candies % friends = 2 :=
by sorry

end candy_distribution_l189_189896


namespace discount_store_purchase_l189_189730

theorem discount_store_purchase (n x y : ℕ) (hn : 2 * n + (x + y) = 2 * n) 
(h1 : 8 * x + 9 * y = 172) (hx : 0 ≤ x) (hy : 0 ≤ y): 
x = 8 ∧ y = 12 :=
sorry

end discount_store_purchase_l189_189730


namespace train_waiting_probability_l189_189742

-- Conditions
def trains_per_hour : ℕ := 1
def total_minutes : ℕ := 60
def wait_time : ℕ := 10

-- Proposition
theorem train_waiting_probability : 
  (wait_time : ℝ) / (total_minutes / trains_per_hour) = 1 / 6 :=
by
  -- Here we assume the proof proceeds correctly
  sorry

end train_waiting_probability_l189_189742


namespace determine_angle_G_l189_189806

theorem determine_angle_G 
  (C D E F G : ℝ)
  (hC : C = 120) 
  (h_linear_pair : C + D = 180)
  (hE : E = 50) 
  (hF : F = D) 
  (h_triangle_sum : E + F + G = 180) :
  G = 70 := 
sorry

end determine_angle_G_l189_189806


namespace circle_radius_square_l189_189408

-- Definition of the problem setup
variables {EF GH ER RF GS SH R S : ℝ}

-- Given conditions
def condition1 : ER = 23 := by sorry
def condition2 : RF = 23 := by sorry
def condition3 : GS = 31 := by sorry
def condition4 : SH = 15 := by sorry

-- Circle radius to be proven
def radius_squared : ℝ := 706

-- Lean 4 theorem statement
theorem circle_radius_square (h1 : ER = 23) (h2 : RF = 23) (h3 : GS = 31) (h4 : SH = 15) :
  (r : ℝ) ^ 2 = 706 := sorry

end circle_radius_square_l189_189408


namespace prove_inequalities_l189_189770

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logb 5 2
noncomputable def z := Real.exp (-1 / 2)

theorem prove_inequalities : y < z ∧ z < x := by
  unfold x y z
  sorry

end prove_inequalities_l189_189770


namespace candy_per_day_eq_eight_l189_189764

def candy_received_from_neighbors : ℝ := 11.0
def candy_received_from_sister : ℝ := 5.0
def days_candy_lasted : ℝ := 2.0

theorem candy_per_day_eq_eight :
  (candy_received_from_neighbors + candy_received_from_sister) / days_candy_lasted = 8.0 :=
by
  sorry

end candy_per_day_eq_eight_l189_189764


namespace y1_lt_y2_l189_189918

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l189_189918


namespace minimum_soldiers_to_add_l189_189626

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189626


namespace product_of_rational_solutions_eq_twelve_l189_189858

theorem product_of_rational_solutions_eq_twelve :
  ∃ c1 c2 : ℕ, (c1 > 0) ∧ (c2 > 0) ∧ 
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c1 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c1 = d^2) ∧
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c2 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c2 = d^2) ∧
               c1 * c2 = 12 := sorry

end product_of_rational_solutions_eq_twelve_l189_189858


namespace solve_equation_l189_189795

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 :=
sorry

end solve_equation_l189_189795


namespace function_is_monotonic_and_odd_l189_189444

   variable (a : ℝ) (x : ℝ)

   noncomputable def f : ℝ := (a^x - a^(-x))

   theorem function_is_monotonic_and_odd (h1 : a > 0) (h2 : a ≠ 1) : 
     (∀ x : ℝ, f (-x) = -f (x)) ∧ ((a > 1 → ∀ x y : ℝ, x < y → f x < f y) ∧ (0 < a ∧ a < 1 → ∀ x y : ℝ, x < y → f x > f y)) :=
   by
         sorry
   
end function_is_monotonic_and_odd_l189_189444


namespace problem_pm_sqrt5_sin_tan_l189_189156

theorem problem_pm_sqrt5_sin_tan
  (m : ℝ)
  (h_m_nonzero : m ≠ 0)
  (cos_alpha : ℝ)
  (h_cos_alpha : cos_alpha = (Real.sqrt 2 * m) / 4)
  (P : ℝ × ℝ)
  (h_P : P = (m, -Real.sqrt 3))
  (r : ℝ)
  (h_r : r = Real.sqrt (3 + m^2)) :
    (∃ m, m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
    (∃ sin_alpha tan_alpha,
      (sin_alpha = - Real.sqrt 6 / 4 ∧ tan_alpha = -Real.sqrt 15 / 5)) :=
by
  sorry

end problem_pm_sqrt5_sin_tan_l189_189156


namespace exp_neg_eq_l189_189600

theorem exp_neg_eq (θ φ : ℝ) (h : Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 2 : ℂ) + (1 / 3 : ℂ) * Complex.I) :
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 2 : ℂ) - (1 / 3 : ℂ) * Complex.I :=
by sorry

end exp_neg_eq_l189_189600


namespace cost_per_dozen_l189_189664

theorem cost_per_dozen (total_cost : ℝ) (total_rolls dozens : ℝ) (cost_per_dozen : ℝ) (h₁ : total_cost = 15) (h₂ : total_rolls = 36) (h₃ : dozens = total_rolls / 12) (h₄ : cost_per_dozen = total_cost / dozens) : cost_per_dozen = 5 :=
by
  sorry

end cost_per_dozen_l189_189664


namespace system_of_equations_l189_189138

theorem system_of_equations (x y z : ℝ) (h1 : 4 * x - 6 * y - 2 * z = 0) (h2 : 2 * x + 6 * y - 28 * z = 0) (hz : z ≠ 0) :
  (x^2 - 6 * x * y) / (y^2 + 4 * z^2) = -5 :=
by
  sorry

end system_of_equations_l189_189138


namespace total_profit_is_2560_l189_189215

noncomputable def basicWashPrice : ℕ := 5
noncomputable def deluxeWashPrice : ℕ := 10
noncomputable def premiumWashPrice : ℕ := 15

noncomputable def basicCarsWeekday : ℕ := 50
noncomputable def deluxeCarsWeekday : ℕ := 40
noncomputable def premiumCarsWeekday : ℕ := 20

noncomputable def employeeADailyWage : ℕ := 110
noncomputable def employeeBDailyWage : ℕ := 90
noncomputable def employeeCDailyWage : ℕ := 100
noncomputable def employeeDDailyWage : ℕ := 80

noncomputable def operatingExpenseWeekday : ℕ := 200

noncomputable def totalProfit : ℕ := 
  let revenueWeekday := (basicCarsWeekday * basicWashPrice) + 
                        (deluxeCarsWeekday * deluxeWashPrice) + 
                        (premiumCarsWeekday * premiumWashPrice)
  let totalRevenue := revenueWeekday * 5
  let wageA := employeeADailyWage * 5
  let wageB := employeeBDailyWage * 2
  let wageC := employeeCDailyWage * 3
  let wageD := employeeDDailyWage * 2
  let totalWages := wageA + wageB + wageC + wageD
  let totalOperatingExpenses := operatingExpenseWeekday * 5
  totalRevenue - (totalWages + totalOperatingExpenses)

theorem total_profit_is_2560 : totalProfit = 2560 := by
  sorry

end total_profit_is_2560_l189_189215


namespace woman_working_days_l189_189532

-- Define the conditions
def man_work_rate := 1 / 6
def boy_work_rate := 1 / 18
def combined_work_rate := 1 / 4

-- Question statement in Lean 4
theorem woman_working_days :
  ∃ W : ℚ, (man_work_rate + W + boy_work_rate = combined_work_rate) ∧ (1 / W = 1296) :=
sorry

end woman_working_days_l189_189532


namespace problem1_problem2_problem3_l189_189029

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x - 4) = -f x)
variable (h_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ 2 → x ≤ y → y ≤ 2 → f x ≤ f y)

-- Problem statements
theorem problem1 : f 2012 = 0 := sorry

theorem problem2 : ∀ x, f (4 - x) = -f (4 + x) := sorry

theorem problem3 : f (-25) < f 80 ∧ f 80 < f 11 := sorry

end problem1_problem2_problem3_l189_189029


namespace contingency_fund_allocation_l189_189940

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l189_189940


namespace min_max_values_of_f_l189_189514

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x >= - (3 * Real.pi / 2)) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = - (3 * Real.pi / 2)) ∧
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f x <= Real.pi / 2 + 2) ∧
  (∃ x ∈ set.Icc 0 (2 * Real.pi), f x = Real.pi / 2 + 2) :=
by {
  -- Proof omitted
  sorry
}

end min_max_values_of_f_l189_189514


namespace Cary_height_is_72_l189_189550

variable (Cary_height Bill_height Jan_height : ℕ)

-- Conditions
axiom Bill_height_is_half_Cary_height : Bill_height = Cary_height / 2
axiom Jan_height_is_6_inches_taller_than_Bill : Jan_height = Bill_height + 6
axiom Jan_height_is_42 : Jan_height = 42

-- Theorem statement
theorem Cary_height_is_72 : Cary_height = 72 := 
by
  sorry

end Cary_height_is_72_l189_189550


namespace combined_loss_l189_189488

variable (initial : ℕ) (donation : ℕ) (prize : ℕ) (final : ℕ) (lottery_winning : ℕ) (X : ℕ)

theorem combined_loss (h1 : initial = 10) (h2 : donation = 4) (h3 : prize = 90) 
                      (h4 : final = 94) (h5 : lottery_winning = 65) :
                      (initial - donation + prize - X + lottery_winning = final) ↔ (X = 67) :=
by
  -- proof steps will go here
  sorry

end combined_loss_l189_189488


namespace single_shot_decrease_l189_189535

theorem single_shot_decrease (S : ℝ) (r1 r2 r3 : ℝ) (h1 : r1 = 0.05) (h2 : r2 = 0.10) (h3 : r3 = 0.15) :
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100 = 27.325 := 
by
  sorry

end single_shot_decrease_l189_189535


namespace arithmetic_expression_count_l189_189026

theorem arithmetic_expression_count (f : ℕ → ℤ) 
  (h1 : f 1 = 9)
  (h2 : f 2 = 99)
  (h_recur : ∀ n ≥ 2, f n = 9 * (f (n - 1)) + 36 * (f (n - 2))) :
  ∀ n, f n = (7 / 10 : ℚ) * 12^n - (1 / 5 : ℚ) * (-3)^n := sorry

end arithmetic_expression_count_l189_189026


namespace responses_needed_l189_189332

-- Define the given conditions
def rate : ℝ := 0.80
def num_mailed : ℕ := 375

-- Statement to prove
theorem responses_needed :
  rate * num_mailed = 300 := by
  sorry

end responses_needed_l189_189332


namespace minimum_soldiers_to_add_l189_189605

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189605


namespace compute_value_of_expression_l189_189827

theorem compute_value_of_expression (p q : ℝ) (hpq : 3 * p^2 - 5 * p - 8 = 0) (hq : 3 * q^2 - 5 * q - 8 = 0) (hneq : p ≠ q) :
  3 * (p^2 - q^2) / (p - q) = 5 :=
by
  have hpq_sum : p + q = 5 / 3 := sorry
  exact sorry

end compute_value_of_expression_l189_189827


namespace length_diff_width_8m_l189_189724

variables (L W : ℝ)

theorem length_diff_width_8m (h1: W = (1/2) * L) (h2: L * W = 128) : L - W = 8 :=
by sorry

end length_diff_width_8m_l189_189724


namespace max_sector_area_l189_189319

theorem max_sector_area (r θ : ℝ) (S : ℝ) (h_perimeter : 2 * r + θ * r = 16)
  (h_max_area : S = 1 / 2 * θ * r^2) :
  r = 4 ∧ θ = 2 ∧ S = 16 := by
  -- sorry, the proof is expected to go here
  sorry

end max_sector_area_l189_189319


namespace spaceship_initial_people_count_l189_189148

/-- For every 100 additional people that board a spaceship, its speed is halved.
     The speed of the spaceship with a certain number of people on board is 500 km per hour.
     The speed of the spaceship when there are 400 people on board is 125 km/hr.
     Prove that the number of people on board when the spaceship was moving at 500 km/hr is 200. -/
theorem spaceship_initial_people_count (speed : ℕ → ℕ) (n : ℕ) :
  (∀ k, speed (k + 100) = speed k / 2) →
  speed n = 500 →
  speed 400 = 125 →
  n = 200 :=
by
  intro half_speed speed_500 speed_400
  sorry

end spaceship_initial_people_count_l189_189148


namespace tangent_ellipse_hyperbola_l189_189083

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ↔ x^2 - n * (y - 1)^2 = 4) →
  n = 9 / 5 :=
by sorry

end tangent_ellipse_hyperbola_l189_189083


namespace euro_operation_example_l189_189722

def euro_operation (x y : ℕ) : ℕ := 3 * x * y

theorem euro_operation_example : euro_operation 3 (euro_operation 4 5) = 540 :=
by sorry

end euro_operation_example_l189_189722


namespace minimum_soldiers_to_add_l189_189603

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189603


namespace integer_solution_inequality_l189_189038

theorem integer_solution_inequality (x : ℤ) : ((x - 1)^2 ≤ 4) → ([-1, 0, 1, 2, 3].count x = 5) :=
by
  sorry

end integer_solution_inequality_l189_189038


namespace find_phi_symmetric_l189_189909

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.sqrt 3 * (Real.cos (2 * x)))

theorem find_phi_symmetric : ∃ φ : ℝ, (φ = Real.pi / 12) ∧ ∀ x : ℝ, f (-x + φ) = f (x + φ) := 
sorry

end find_phi_symmetric_l189_189909


namespace percentage_of_number_l189_189947

theorem percentage_of_number (X P : ℝ) (h1 : 0.20 * X = 80) (h2 : (P / 100) * X = 160) : P = 40 := by
  sorry

end percentage_of_number_l189_189947


namespace inequality_solution_l189_189760

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by
  sorry

end inequality_solution_l189_189760


namespace daisy_dog_toys_l189_189363

-- Given conditions
def dog_toys_monday : ℕ := 5
def dog_toys_tuesday_left : ℕ := 3
def dog_toys_tuesday_bought : ℕ := 3
def dog_toys_wednesday_all_found : ℕ := 13

-- The question we need to answer
def dog_toys_bought_wednesday : ℕ := 7

-- Statement to prove
theorem daisy_dog_toys :
  (dog_toys_monday - dog_toys_tuesday_left + dog_toys_tuesday_left + dog_toys_tuesday_bought + dog_toys_bought_wednesday = dog_toys_wednesday_all_found) :=
sorry

end daisy_dog_toys_l189_189363


namespace max_OA_div_OB_value_l189_189052

open Real

noncomputable def parametric_curve_C (α : ℝ) : ℝ × ℝ := (1 + cos α, sin α)
noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ := (1 - t, 3 + t)

def polar_ray_m (θ β : ℝ) : Prop := θ = β ∧ ∀ ρ, ρ > 0

def ρ₁ (β : ℝ) : ℝ := 2 * cos β
def ρ₂ (β : ℝ) : ℝ := 2 * sqrt 2 / (sin (β + π/4))

def max_OA_div_OB : ℝ := (sqrt 2 + 1)/4

theorem max_OA_div_OB_value (β : ℝ) (hβ1 : β ∈ Ioo (-π / 4) (π / 4)) : 
  ∃ (A B : ℝ × ℝ), polar_ray_m A.1 β ∧ polar_ray_m B.1 β ∧ 
  (parametric_curve_C β = A) ∧ (parametric_line_l β = B) ∧
  (abs (1 / ρ₂ β)) = max_OA_div_OB :=
sorry

end max_OA_div_OB_value_l189_189052


namespace pass_probability_is_two_thirds_l189_189185

noncomputable def hypergeometric_distribution (m n k : ℕ) (x : ℕ) : ℚ :=
(nat.choose m x * nat.choose n (k - x)) / nat.choose (m + n) k

def pass_probability : ℚ :=
  hypergeometric_distribution 6 4 3 2 + hypergeometric_distribution 6 4 3 3

theorem pass_probability_is_two_thirds :
  pass_probability = 2 / 3 :=
by
  sorry

end pass_probability_is_two_thirds_l189_189185


namespace range_of_a_l189_189162

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → (a / x₁ + x₁ * Real.log x₁ ≥ x₂^3 - x₂^2 - 3)) →
  a ∈ Set.Ici 1 :=
by
  sorry

end range_of_a_l189_189162


namespace gcd_in_range_l189_189507

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l189_189507


namespace unique_y_star_l189_189422

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_y_star :
  ∃! y : ℝ, star 4 y = 20 :=
by 
  sorry

end unique_y_star_l189_189422


namespace most_likely_outcome_l189_189308

-- Defining the conditions
def equally_likely (n : ℕ) (k : ℕ) := (Nat.choose n k) * (1 / 2)^n

-- Defining the problem statement
theorem most_likely_outcome :
  (equally_likely 5 3 = 5 / 16 ∧ equally_likely 5 2 = 5 / 16) :=
sorry

end most_likely_outcome_l189_189308


namespace equal_share_candy_l189_189595

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l189_189595


namespace painting_cost_in_cny_l189_189974

theorem painting_cost_in_cny (usd_to_nad : ℝ) (usd_to_cny : ℝ) (painting_cost_nad : ℝ) :
  usd_to_nad = 8 → usd_to_cny = 7 → painting_cost_nad = 160 →
  painting_cost_nad / usd_to_nad * usd_to_cny = 140 :=
by
  intros
  sorry

end painting_cost_in_cny_l189_189974


namespace no_1968_classes_l189_189471

theorem no_1968_classes :
  ∀ (classes : Finset (Finset ℕ)), 
    (∀ n ∈ classes, n.nonempty) →
    classes.card = 1968 →
    (∀ (m n : ℕ), (∃ p q : ℕ, (m = p*100 + q) ∨ (m = p*10 + q)) → 
    (∃ c ∈ classes, n ∈ c ∧ m ∈ c)) →
  False :=
begin
  intros classes nonempty_classes card_1968 transform_preservation,
  -- Since we don't need to prove it, we leave it as sorry
  sorry
end

end no_1968_classes_l189_189471


namespace minimum_distinct_numbers_l189_189491

theorem minimum_distinct_numbers (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j : ℕ, 1 ≤ i ∧ i < 2006 ∧ 1 ≤ j ∧ j < 2006 ∧ i ≠ j → a i / a (i + 1) ≠ a j / a (j + 1)) :
  ∃ (n : ℕ), n = 46 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2006 ∧ 1 ≤ j ∧ j ≤ i ∧ (a i = a j → i = j) :=
sorry

end minimum_distinct_numbers_l189_189491


namespace meaningful_iff_x_ne_1_l189_189598

theorem meaningful_iff_x_ne_1 (x : ℝ) : (x - 1) ≠ 0 ↔ (x ≠ 1) :=
by 
  sorry

end meaningful_iff_x_ne_1_l189_189598


namespace gross_profit_percentage_l189_189851

theorem gross_profit_percentage (sales_price gross_profit cost : ℝ) 
  (h1 : sales_price = 81) 
  (h2 : gross_profit = 51) 
  (h3 : cost = sales_price - gross_profit) : 
  (gross_profit / cost) * 100 = 170 := 
by
  simp [h1, h2, h3]
  sorry

end gross_profit_percentage_l189_189851


namespace quadratic_function_correct_options_c_d_l189_189338

theorem quadratic_function_correct_options_c_d
  (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : -b / (2 * a) = 2)
  (h3 : c > -1) :
  true :=
by {
  sorry
}

end quadratic_function_correct_options_c_d_l189_189338


namespace value_of_a_minus_b_l189_189646

theorem value_of_a_minus_b (a b : ℤ) 
  (h₁ : |a| = 7) 
  (h₂ : |b| = 5) 
  (h₃ : a < b) : 
  a - b = -12 ∨ a - b = -2 := 
sorry

end value_of_a_minus_b_l189_189646


namespace minimum_soldiers_to_add_l189_189630

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189630


namespace last_digit_product_3_2001_7_2002_13_2003_l189_189846

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_product_3_2001_7_2002_13_2003 :
  last_digit (3^2001 * 7^2002 * 13^2003) = 9 :=
by
  sorry

end last_digit_product_3_2001_7_2002_13_2003_l189_189846


namespace player_avg_increase_l189_189744

theorem player_avg_increase
  (matches_played : ℕ)
  (initial_avg : ℕ)
  (next_match_runs : ℕ)
  (total_runs : ℕ)
  (new_total_runs : ℕ)
  (new_avg : ℕ)
  (desired_avg_increase : ℕ) :
  matches_played = 10 ∧ initial_avg = 32 ∧ next_match_runs = 76 ∧ total_runs = 320 ∧ 
  new_total_runs = 396 ∧ new_avg = 32 + desired_avg_increase ∧ 
  11 * new_avg = new_total_runs → desired_avg_increase = 4 := 
by
  sorry

end player_avg_increase_l189_189744


namespace length_of_box_l189_189497

theorem length_of_box (rate : ℕ) (width : ℕ) (depth : ℕ) (time : ℕ) (volume : ℕ) (length : ℕ) :
  rate = 4 →
  width = 6 →
  depth = 2 →
  time = 21 →
  volume = rate * time →
  length = volume / (width * depth) →
  length = 7 :=
by
  intros
  sorry

end length_of_box_l189_189497


namespace g_eval_l189_189354

-- Define the function g
def g (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := (2 * a + b) / (c - a)

-- Theorem to prove g(2, 4, -1) = -8 / 3
theorem g_eval :
  g 2 4 (-1) = -8 / 3 := 
by
  sorry

end g_eval_l189_189354


namespace contingency_fund_correct_l189_189938

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l189_189938


namespace remainder_of_poly_div_l189_189861

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_div_l189_189861


namespace problem_statement_l189_189172

/-- For any positive integer n, given θ ∈ (0, π) and x ∈ ℂ such that 
x + 1/x = 2√2 cos θ - sin θ, it follows that x^n + 1/x^n = 2 cos (n α). -/
theorem problem_statement (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (x : ℂ) (hx : x + 1/x = 2 * (2:ℝ).sqrt * θ.cos - θ.sin)
  (n : ℕ) (hn : 0 < n) : x^n + x⁻¹^n = 2 * θ.cos * n := 
  sorry

end problem_statement_l189_189172


namespace right_triangle_third_side_l189_189951

theorem right_triangle_third_side (a b : ℝ) (h : a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2)
  (h1 : a = 3 ∧ b = 5 ∨ a = 5 ∧ b = 3) : c = 4 ∨ c = Real.sqrt 34 :=
sorry

end right_triangle_third_side_l189_189951


namespace find_smallest_number_l189_189699

theorem find_smallest_number (x y z : ℝ) 
  (h1 : x + y + z = 150) 
  (h2 : y = 3 * x + 10) 
  (h3 : z = x^2 - 5) 
  : x = 10.21 :=
sorry

end find_smallest_number_l189_189699


namespace f_sum_zero_l189_189381

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Define hypotheses based on the problem's conditions
axiom f_cube (x : ℝ) : f (x ^ 3) = (f x) ^ 3
axiom f_inj (x1 x2 : ℝ) (h : x1 ≠ x2) : f x1 ≠ f x2

-- State the proof problem
theorem f_sum_zero : f 0 + f 1 + f (-1) = 0 :=
sorry

end f_sum_zero_l189_189381


namespace parabola_focus_distance_l189_189776

noncomputable def PF (x₁ : ℝ) : ℝ := x₁ + 1
noncomputable def QF (x₂ : ℝ) : ℝ := x₂ + 1

theorem parabola_focus_distance 
  (x₁ x₂ : ℝ) (h₁ : x₂ = 3 * x₁ + 2) : 
  QF x₂ / PF x₁ = 3 :=
by
  sorry

end parabola_focus_distance_l189_189776


namespace closest_approx_w_l189_189423

noncomputable def w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)

theorem closest_approx_w : |w + 296.073| < 0.001 :=
by
  sorry

end closest_approx_w_l189_189423


namespace six_player_round_robin_matches_l189_189752

theorem six_player_round_robin_matches : 
  ∀ (n : ℕ), n = 6 → ((n * (n - 1)) / 2) = 15 := by 
  intros n hn 
  rw hn 
  -- now we should have (6 * 5) / 2 = 15, but we will leave this to sorry
  sorry

end six_player_round_robin_matches_l189_189752


namespace geom_seq_arith_form_l189_189189

theorem geom_seq_arith_form (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : (a 1, (1 / 2) * a 3, 2 * a 2) ∈ SetOf p q r where p + r = 2 * q) :
  (a 6 + a 8 + a 10) / (a 7 + a 9 + a 11) = Real.sqrt 2 - 1 :=
by
  sorry

end geom_seq_arith_form_l189_189189


namespace inverse_proportion_inequality_l189_189914

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l189_189914


namespace committee_selection_l189_189545

theorem committee_selection:
  let total_candidates := 15
  let former_members := 6
  let positions := 4
  let total_combinations := Nat.choose total_candidates positions
  let non_former_candidates := total_candidates - former_members
  let no_former_combinations := Nat.choose non_former_candidates positions
  total_combinations - no_former_combinations = 1239 :=
by
  let total_candidates := 15
  let former_members := 6
  let positions := 4
  let total_combinations := Nat.choose total_candidates positions
  let non_former_candidates := total_candidates - former_members
  let no_former_combinations := Nat.choose non_former_candidates positions
  sorry

end committee_selection_l189_189545


namespace man_l189_189261

theorem man's_speed_against_the_current (vm vc : ℝ) 
(h1: vm + vc = 15) 
(h2: vm - vc = 10) : 
vm - vc = 10 := 
by 
  exact h2

end man_l189_189261


namespace larger_solution_quadratic_l189_189290

theorem larger_solution_quadratic :
  (∃ a b : ℝ, a ≠ b ∧ (a = 9) ∧ (b = -2) ∧
              (∀ x : ℝ, x^2 - 7 * x - 18 = 0 → (x = a ∨ x = b))) →
  9 = max a b :=
by
  sorry

end larger_solution_quadratic_l189_189290


namespace second_batch_jelly_beans_weight_l189_189477

theorem second_batch_jelly_beans_weight (J : ℝ) (h1 : 2 * 3 + J > 0) (h2 : (6 + J) * 2 = 16) : J = 2 :=
sorry

end second_batch_jelly_beans_weight_l189_189477


namespace smallest_X_l189_189963

theorem smallest_X (T : ℕ) (hT_digits : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) (hX_int : ∃ (X : ℕ), T = 20 * X) : ∃ T, ∀ X, X = T / 20 → X = 55 :=
by
  sorry

end smallest_X_l189_189963


namespace greatest_integer_value_l189_189529

theorem greatest_integer_value (x : ℤ) (h : ∃ x : ℤ, x = 29 ∧ ∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*(x+6) + 26)) :
  (∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*k + 26) → x = 29) :=
by
  sorry

end greatest_integer_value_l189_189529


namespace parabola_from_hyperbola_l189_189926

noncomputable def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

noncomputable def parabola_equation_1 (x y : ℝ) : Prop := y^2 = -24 * x

noncomputable def parabola_equation_2 (x y : ℝ) : Prop := y^2 = 24 * x

theorem parabola_from_hyperbola :
  (∃ x y : ℝ, hyperbola_equation x y) →
  (∃ x y : ℝ, parabola_equation_1 x y ∨ parabola_equation_2 x y) :=
by
  intro h
  -- proof is omitted
  sorry

end parabola_from_hyperbola_l189_189926


namespace infinite_primes_of_the_year_2022_l189_189273

theorem infinite_primes_of_the_year_2022 :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p % 2 = 1 ∧ p ^ 2022 ∣ n ^ 2022 + 2022 :=
sorry

end infinite_primes_of_the_year_2022_l189_189273


namespace find_range_of_a_l189_189445

variable {f : ℝ → ℝ}
noncomputable def domain_f : Set ℝ := {x | 7 ≤ x ∧ x < 15}
noncomputable def domain_f_2x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}
noncomputable def A_or_B_eq_r (a : ℝ) : Prop := domain_f_2x_plus_1 ∪ B a = Set.univ

theorem find_range_of_a (a : ℝ) : 
  A_or_B_eq_r a → 3 ≤ a ∧ a < 6 := 
sorry

end find_range_of_a_l189_189445


namespace permutations_red_l189_189789

theorem permutations_red (n : ℕ) (h : (3 * n)! / ((n!) * (n!) * (n!)) = 6) : n = 1 :=
by
  sorry

end permutations_red_l189_189789


namespace pre_bought_ticket_price_l189_189527

variable (P : ℕ)

theorem pre_bought_ticket_price :
  (20 * P = 6000 - 2900) → P = 155 :=
by
  intro h
  sorry

end pre_bought_ticket_price_l189_189527


namespace number_of_solutions_sine_exponential_l189_189020

theorem number_of_solutions_sine_exponential :
  let f := λ x => Real.sin x
  let g := λ x => (1 / 3) ^ x
  ∃ n, n = 150 ∧ ∀ k ∈ Set.Icc (0 : ℝ) (150 * Real.pi), f k = g k → (k : ℝ) ∈ {n : ℝ | n ∈ Set.Icc (0 : ℝ) (150 * Real.pi)} :=
sorry

end number_of_solutions_sine_exponential_l189_189020


namespace count_positive_integers_divisible_by_4_6_10_less_than_300_l189_189589

-- The problem states the following conditions
def is_divisible_by (m n : ℕ) : Prop := m % n = 0
def less_than_300 (n : ℕ) : Prop := n < 300

-- We want to prove the number of positive integers less than 300 that are divisible by 4, 6, and 10
theorem count_positive_integers_divisible_by_4_6_10_less_than_300 :
  (Finset.card (Finset.filter 
    (λ n, is_divisible_by n 4 ∧ is_divisible_by n 6 ∧ is_divisible_by n 10 ∧ less_than_300 n)
    ((Finset.range 300).filter (λ n, n ≠ 0)))) = 4 :=
by
  sorry

end count_positive_integers_divisible_by_4_6_10_less_than_300_l189_189589


namespace parabola_vertex_l189_189504

theorem parabola_vertex : ∃ h k : ℝ, (∀ x : ℝ, 2 * (x - h)^2 + k = 2 * (x - 5)^2 + 3) ∧ h = 5 ∧ k = 3 :=
by {
  use 5,
  use 3,
  split,
  { intro x,
    simp },
  exact ⟨rfl, rfl⟩,
}

end parabola_vertex_l189_189504


namespace ring_tower_height_l189_189122

theorem ring_tower_height : 
  let thickness := 2
  let smallest_outside_diameter := 10
  let largest_outside_diameter := 30
  let num_rings := (largest_outside_diameter - smallest_outside_diameter) / thickness + 1
  let total_distance := num_rings * thickness + smallest_outside_diameter - thickness
  total_distance = 200 :=
by {
  let thickness := 2,
  let smallest_outside_diameter := 10,
  let largest_outside_diameter := 30,
  let num_rings := (largest_outside_diameter - smallest_outside_diameter) / thickness + 1,
  let total_distance := num_rings * thickness + smallest_outside_diameter - thickness,
  have h : total_distance = 200 := sorry,
  exact h,
}

end ring_tower_height_l189_189122


namespace new_person_weight_l189_189723

theorem new_person_weight 
    (W : ℝ) -- total weight of original 8 people
    (x : ℝ) -- weight of the new person
    (increase_by : ℝ) -- average weight increases by 2.5 kg
    (replaced_weight : ℝ) -- weight of the replaced person (55 kg)
    (h1 : increase_by = 2.5)
    (h2 : replaced_weight = 55)
    (h3 : x = replaced_weight + (8 * increase_by)) : x = 75 := 
by
  sorry

end new_person_weight_l189_189723


namespace extreme_values_l189_189779

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 3

theorem extreme_values (a b : ℝ) : 
  (f a b (-1) = 10) ∧ (f a b 2 = -17) →
  (6 * (-1)^2 + 2 * a * (-1) + b = 0) ∧ (6 * 2^2 + 2 * (a * 2) + b = 0) →
  a = -3 ∧ b = -12 :=
by 
  sorry

end extreme_values_l189_189779


namespace find_equation_for_second_machine_l189_189873

theorem find_equation_for_second_machine (x : ℝ) : 
  (1 / 6) + (1 / x) = 1 / 3 ↔ (x = 6) := 
by 
  sorry

end find_equation_for_second_machine_l189_189873


namespace total_daisies_l189_189341

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l189_189341


namespace sum_first_12_terms_l189_189453

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 ^ (n - 1) else 2 * n - 1

def S (n : ℕ) : ℕ := 
  (Finset.range n).sum a

theorem sum_first_12_terms : S 12 = 1443 :=
by
  sorry

end sum_first_12_terms_l189_189453


namespace minimum_soldiers_to_add_l189_189606

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189606


namespace num_ways_to_select_ascend_triple_l189_189151

theorem num_ways_to_select_ascend_triple :
  (∑ a1 in finset.Ico 1 12, ∑ a2 in finset.Ico (a1 + 3) 15, finset.card (finset.Ico (a2 + 3) 15)) = 120 := by
  sorry

end num_ways_to_select_ascend_triple_l189_189151


namespace part2_l189_189586

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - x

theorem part2 (a : ℝ) (h : a > 0) (x : ℝ) : f a x < (a - 1) * Real.log a + a^2 := 
  sorry

end part2_l189_189586


namespace students_failed_in_english_l189_189467

variable (H : ℝ) (E : ℝ) (B : ℝ) (P : ℝ)

theorem students_failed_in_english
  (hH : H = 34 / 100) 
  (hB : B = 22 / 100)
  (hP : P = 44 / 100)
  (hIE : (1 - P) = H + E - B) :
  E = 44 / 100 := 
sorry

end students_failed_in_english_l189_189467


namespace find_p_q_sum_p_plus_q_l189_189890

noncomputable def probability_third_six : ℚ :=
  have fair_die_prob_two_sixes := (1 / 6) * (1 / 6)
  have biased_die_prob_two_sixes := (2 / 3) * (2 / 3)
  have total_prob_two_sixes := (1 / 2) * fair_die_prob_two_sixes + (1 / 2) * biased_die_prob_two_sixes
  have prob_fair_given_two_sixes := fair_die_prob_two_sixes / total_prob_two_sixes
  have prob_biased_given_two_sixes := biased_die_prob_two_sixes / total_prob_two_sixes
  let prob_third_six :=
    prob_fair_given_two_sixes * (1 / 6) +
    prob_biased_given_two_sixes * (2 / 3)
  prob_third_six

theorem find_p_q_sum : 
  probability_third_six = 65 / 102 :=
by sorry

theorem p_plus_q : 
  65 + 102 = 167 :=
by sorry

end find_p_q_sum_p_plus_q_l189_189890


namespace school_anniversary_problem_l189_189679

theorem school_anniversary_problem
    (total_cost : ℕ)
    (cost_commemorative_albums cost_bone_china_cups : ℕ)
    (num_commemorative_albums num_bone_china_cups : ℕ)
    (price_commemorative_album price_bone_china_cup : ℕ)
    (H1 : total_cost = 312000)
    (H2 : cost_commemorative_albums + cost_bone_china_cups = total_cost)
    (H3 : cost_commemorative_albums = 3 * cost_bone_china_cups)
    (H4 : price_commemorative_album = 3 / 2 * price_bone_china_cup)
    (H5 : num_bone_china_cups = 4 * num_commemorative_albums + 1600) :
    (cost_commemorative_albums = 72000 ∧ cost_bone_china_cups = 240000) ∧
    (price_commemorative_album = 45 ∧ price_bone_china_cup = 30) :=
by
  sorry

end school_anniversary_problem_l189_189679


namespace eric_boxes_l189_189016

def numberOfBoxes (totalPencils : Nat) (pencilsPerBox : Nat) : Nat :=
  totalPencils / pencilsPerBox

theorem eric_boxes :
  numberOfBoxes 27 9 = 3 := by
  sorry

end eric_boxes_l189_189016


namespace intersection_A_B_l189_189062

-- Conditions
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

-- Proof of the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l189_189062


namespace total_daisies_l189_189345

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l189_189345


namespace width_of_box_is_correct_l189_189848

noncomputable def length_of_box : ℝ := 62
noncomputable def height_lowered : ℝ := 0.5
noncomputable def volume_removed_in_gallons : ℝ := 5812.5
noncomputable def gallons_to_cubic_feet : ℝ := 1 / 7.48052

theorem width_of_box_is_correct :
  let volume_removed_in_cubic_feet := volume_removed_in_gallons * gallons_to_cubic_feet
  let area_of_base := length_of_box * W
  let needed_volume := area_of_base * height_lowered
  volume_removed_in_cubic_feet = needed_volume →
  W = 25.057 :=
by
  sorry

end width_of_box_is_correct_l189_189848


namespace inverse_proportion_inequality_l189_189915

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l189_189915


namespace smallest_number_of_people_l189_189099

open Nat

theorem smallest_number_of_people (x : ℕ) :
  (∃ x, x % 18 = 0 ∧ x % 50 = 0 ∧
  (∀ y, y % 18 = 0 ∧ y % 50 = 0 → x ≤ y)) → x = 450 :=
by
  sorry

end smallest_number_of_people_l189_189099


namespace y1_lt_y2_l189_189917

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l189_189917


namespace gcd_consecutive_odd_product_l189_189892

theorem gcd_consecutive_odd_product (n : ℕ) (hn : n % 2 = 0 ∧ n > 0) : 
  Nat.gcd ((n+1)*(n+3)*(n+7)*(n+9)) 15 = 15 := 
sorry

end gcd_consecutive_odd_product_l189_189892


namespace min_max_f_on_interval_l189_189512

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_f_on_interval :
  ∃ min max, min = - (3 * Real.pi) / 2 ∧ max = (Real.pi / 2) + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min ∧ f x ≤ max) :=
sorry

end min_max_f_on_interval_l189_189512


namespace number_of_elements_in_set_l189_189684

-- We define the conditions in terms of Lean definitions.
variable (n : ℕ) (S : ℕ)

-- Define the initial wrong average condition
def wrong_avg_condition : Prop := (S + 26) / n = 18

-- Define the corrected average condition
def correct_avg_condition : Prop := (S + 36) / n = 19

-- The main theorem to be proved
theorem number_of_elements_in_set (h1 : wrong_avg_condition n S) (h2 : correct_avg_condition n S) : n = 10 := 
sorry

end number_of_elements_in_set_l189_189684


namespace max_value_of_f_l189_189908

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_value_of_f:
  ∃ x ∈ Set.Ioo 0 Real.pi, ∀ y ∈ Set.Ioo 0 Real.pi, f y ≤ 2 ∧ f x = 2 ∧ x = Real.pi / 6 :=
by sorry

end max_value_of_f_l189_189908


namespace maximum_expr_value_l189_189652

theorem maximum_expr_value :
  ∃ (x y e f : ℕ), (e = 4 ∧ x = 3 ∧ y = 2 ∧ f = 0) ∧
  (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4) ∧
  (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ∧
  (y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4) ∧
  (f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4) ∧
  (e ≠ x ∧ e ≠ y ∧ e ≠ f ∧ x ≠ y ∧ x ≠ f ∧ y ≠ f) ∧
  (e * x^y - f = 36) :=
by
  sorry

end maximum_expr_value_l189_189652


namespace correct_articles_l189_189018

-- Definitions based on conditions provided in the problem
def sentence := "Traveling in ____ outer space is quite ____ exciting experience."
def first_blank_article := "no article"
def second_blank_article := "an"

-- Statement of the proof problem
theorem correct_articles : 
  (first_blank_article = "no article" ∧ second_blank_article = "an") :=
by
  sorry

end correct_articles_l189_189018


namespace find_ab_l189_189176

-- Define the conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 3
hypothesis h2 : a^2 + b^2 = 29

-- State the theorem
theorem find_ab : a * b = 10 :=
by
  sorry

end find_ab_l189_189176


namespace final_share_approx_equal_l189_189386

noncomputable def total_bill : ℝ := 211.0
noncomputable def number_of_people : ℝ := 6.0
noncomputable def tip_percentage : ℝ := 0.15
noncomputable def tip_amount : ℝ := tip_percentage * total_bill
noncomputable def total_amount : ℝ := total_bill + tip_amount
noncomputable def each_person_share : ℝ := total_amount / number_of_people

theorem final_share_approx_equal :
  abs (each_person_share - 40.44) < 0.01 :=
by
  sorry

end final_share_approx_equal_l189_189386


namespace valentines_left_l189_189671

theorem valentines_left (initial_valentines given_away : ℕ) (h_initial : initial_valentines = 30) (h_given : given_away = 8) :
  initial_valentines - given_away = 22 :=
by {
  sorry
}

end valentines_left_l189_189671


namespace necklace_ratio_l189_189454

variable {J Q H : ℕ}

theorem necklace_ratio (h1 : H = J + 5) (h2 : H = 25) (h3 : H = Q + 15) : Q / J = 1 / 2 := by
  sorry

end necklace_ratio_l189_189454


namespace unique_bounded_sequence_exists_l189_189910

variable (a : ℝ) (n : ℕ) (hn_pos : n > 0)

theorem unique_bounded_sequence_exists :
  ∃! (x : ℕ → ℝ), (x 0 = 0) ∧ (x (n+1) = 0) ∧
                   (∀ i, 1 ≤ i ∧ i ≤ n → (1/2) * (x (i+1) + x (i-1)) = x i + x i ^ 3 - a ^ 3) ∧
                   (∀ i, i ≤ n + 1 → |x i| ≤ |a|) := by
  sorry

end unique_bounded_sequence_exists_l189_189910


namespace min_soldiers_needed_l189_189613

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189613


namespace race_distance_l189_189047

theorem race_distance (dA dB dC : ℝ) (h1 : dA = 1000) (h2 : dB = 900) (h3 : dB = 800) (h4 : dC = 700) (d : ℝ) (h5 : d = dA + 127.5) :
  d = 600 :=
sorry

end race_distance_l189_189047


namespace common_ratio_of_geometric_sequence_l189_189854

theorem common_ratio_of_geometric_sequence (S : ℕ → ℝ) (a_1 a_2 : ℝ) (q : ℝ)
  (h1 : S 3 = a_1 * (1 + q + q^2))
  (h2 : 2 * S 3 = 2 * a_1 + a_2) : 
  q = -1/2 := 
sorry

end common_ratio_of_geometric_sequence_l189_189854


namespace negation_if_then_l189_189988

theorem negation_if_then (x : ℝ) : ¬ (x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) :=
by 
  sorry

end negation_if_then_l189_189988


namespace sin_cos_solution_set_l189_189904
open Real

theorem sin_cos_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * π + (-1)^k * (π / 6) - (π / 3)} =
  {x : ℝ | sin x + sqrt 3 * cos x = 1} :=
by sorry

end sin_cos_solution_set_l189_189904


namespace kennedy_distance_to_school_l189_189959

def miles_per_gallon : ℕ := 19
def initial_gallons : ℕ := 2
def distance_softball_park : ℕ := 6
def distance_burger_restaurant : ℕ := 2
def distance_friends_house : ℕ := 4
def distance_home : ℕ := 11

def total_distance_possible : ℕ := miles_per_gallon * initial_gallons
def distance_after_school : ℕ := distance_softball_park + distance_burger_restaurant + distance_friends_house + distance_home
def distance_to_school : ℕ := total_distance_possible - distance_after_school

theorem kennedy_distance_to_school :
  distance_to_school = 15 :=
by
  sorry

end kennedy_distance_to_school_l189_189959


namespace susan_books_l189_189355

theorem susan_books (S : ℕ) (h1 : S + 4 * S = 3000) : S = 600 :=
by 
  sorry

end susan_books_l189_189355


namespace total_seats_taken_l189_189089

def students_per_bus : ℝ := 14.0
def number_of_buses : ℝ := 2.0

theorem total_seats_taken :
  students_per_bus * number_of_buses = 28.0 :=
by
  sorry

end total_seats_taken_l189_189089


namespace decrease_hours_worked_l189_189118

theorem decrease_hours_worked (initial_hourly_wage : ℝ) (initial_hours_worked : ℝ) :
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  initial_hours_worked > 0 → 
  initial_hourly_wage > 0 → 
  new_hours_worked < initial_hours_worked :=
by
  intros initial_hours_worked_pos initial_hourly_wage_pos
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  sorry

end decrease_hours_worked_l189_189118


namespace number_of_zeros_of_f_l189_189516

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin (2 * x)

theorem number_of_zeros_of_f : (∃ l : List ℝ, (∀ x ∈ l, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧ l.length = 4) := 
by
  sorry

end number_of_zeros_of_f_l189_189516


namespace octagon_perimeter_correct_l189_189716

def octagon_perimeter (n : ℕ) (side_length : ℝ) : ℝ :=
  n * side_length

theorem octagon_perimeter_correct :
  octagon_perimeter 8 3 = 24 :=
by
  sorry

end octagon_perimeter_correct_l189_189716


namespace lcm_36_100_eq_900_l189_189296

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l189_189296


namespace find_s_t_l189_189682

noncomputable def problem_constants (a b c : ℝ) : Prop :=
  (a^3 + 3 * a^2 + 4 * a - 11 = 0) ∧
  (b^3 + 3 * b^2 + 4 * b - 11 = 0) ∧
  (c^3 + 3 * c^2 + 4 * c - 11 = 0)

theorem find_s_t (a b c s t : ℝ) (h1 : problem_constants a b c) (h2 : (a + b) * (b + c) * (c + a) = -t)
  (h3 : (a + b) * (b + c) + (b + c) * (c + a) + (c + a) * (a + b) = s) :
s = 8 ∧ t = 23 :=
sorry

end find_s_t_l189_189682


namespace period1_period2_multiple_l189_189080

theorem period1_period2_multiple
  (students_period1 : ℕ)
  (students_period2 : ℕ)
  (h_students_period1 : students_period1 = 11)
  (h_students_period2 : students_period2 = 8)
  (M : ℕ)
  (h_condition : students_period1 = M * students_period2 - 5) :
  M = 2 :=
by
  sorry

end period1_period2_multiple_l189_189080


namespace math_problem_l189_189110

noncomputable def a : ℝ := (0.96)^3 
noncomputable def b : ℝ := (0.1)^3 
noncomputable def c : ℝ := (0.96)^2 
noncomputable def d : ℝ := (0.1)^2 

theorem math_problem : a - b / c + 0.096 + d = 0.989651 := 
by 
  -- skip proof 
  sorry

end math_problem_l189_189110


namespace vector_perpendicular_to_a_l189_189544

theorem vector_perpendicular_to_a :
  let a := (4, 3)
  let b := (3, -4)
  a.1 * b.1 + a.2 * b.2 = 0 := by
  let a := (4, 3)
  let b := (3, -4)
  sorry

end vector_perpendicular_to_a_l189_189544


namespace solve_686_l189_189500

theorem solve_686 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 686 := 
by
  sorry

end solve_686_l189_189500


namespace locomotive_distance_l189_189740

theorem locomotive_distance 
  (speed_train : ℝ) (speed_sound : ℝ) (time_diff : ℝ)
  (h_train : speed_train = 20) 
  (h_sound : speed_sound = 340) 
  (h_time : time_diff = 4) : 
  ∃ x : ℝ, x = 85 := 
by 
  sorry

end locomotive_distance_l189_189740


namespace cycling_problem_l189_189245

theorem cycling_problem (x : ℝ) (h₀ : x > 0) :
  30 / x - 30 / (x + 3) = 2 / 3 :=
sorry

end cycling_problem_l189_189245


namespace total_daisies_l189_189339

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l189_189339


namespace sin_cos_identity_l189_189171

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := 
  sorry

end sin_cos_identity_l189_189171


namespace find_n_l189_189828

-- Define x and y
def x : ℕ := 3
def y : ℕ := 1

-- Define n based on the given expression.
def n : ℕ := x - y^(x - (y + 1))

-- State the theorem
theorem find_n : n = 2 := by
  sorry

end find_n_l189_189828


namespace students_water_count_l189_189466

-- Define the given conditions
def pct_students_juice (total_students : ℕ) : ℕ := 70 * total_students / 100
def pct_students_water (total_students : ℕ) : ℕ := 30 * total_students / 100
def students_juice (total_students : ℕ) : Prop := pct_students_juice total_students = 140

-- Define the proposition that needs to be proven
theorem students_water_count (total_students : ℕ) (h1 : students_juice total_students) : 
  pct_students_water total_students = 60 := 
by
  sorry


end students_water_count_l189_189466


namespace B_pow_five_l189_189825

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![2, 3], ![4, 6]]
  
theorem B_pow_five : 
  B^5 = (4096 : ℝ) • B + (0 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end B_pow_five_l189_189825


namespace prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l189_189095

-- Define events and their probabilities.
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8

-- Given P(A and B) = P(A) * P(B)
def prob_AB : ℝ := prob_A * prob_B

-- Statements to prove
theorem prob_both_hit : prob_AB = 0.64 :=
by
  -- P(A and B) = 0.8 * 0.8 = 0.64
  exact sorry

theorem prob_exactly_one_hit : (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B) = 0.32 :=
by
  -- P(A and not B) + P(not A and B) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32
  exact sorry

theorem prob_at_least_one_hit : (1 - (1 - prob_A) * (1 - prob_B)) = 0.96 :=
by
  -- 1 - P(not A and not B) = 1 - 0.04 = 0.96
  exact sorry

end prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l189_189095


namespace sin_sq_sub_cos_sq_l189_189925

-- Given condition
variable {α : ℝ}
variable (h : Real.sin α = Real.sqrt 5 / 5)

-- Proof goal
theorem sin_sq_sub_cos_sq (h : Real.sin α = Real.sqrt 5 / 5) : Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 := sorry

end sin_sq_sub_cos_sq_l189_189925


namespace lcm_36_100_is_900_l189_189300

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l189_189300


namespace ones_digit_of_prime_sequence_l189_189567

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l189_189567


namespace contingency_fund_amount_l189_189945

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l189_189945


namespace maximum_n_l189_189157

def arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d

def is_positive_first_term (a : ℕ → ℤ) : Prop :=
  a 0 > 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n-1))) / 2

def roots_of_equation (a1006 a1007 : ℤ) : Prop :=
  a1006 * a1007 = -2011 ∧ a1006 + a1007 = 2012

theorem maximum_n (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence_max_n a S 1007)
  (h2 : is_positive_first_term a)
  (h3 : sum_of_first_n_terms a S)
  (h4 : ∃ a1006 a1007, roots_of_equation a1006 a1007 ∧ a 1006 = a1006 ∧ a 1007 = a1007) :
  ∃ n, S n > 0 → n ≤ 1007 := 
sorry

end maximum_n_l189_189157


namespace find_value_of_expression_l189_189457

theorem find_value_of_expression (a b : ℝ) (h : a + 2 * b - 1 = 0) : 3 * a + 6 * b = 3 :=
by
  sorry

end find_value_of_expression_l189_189457


namespace lcm_of_36_and_100_l189_189292

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l189_189292


namespace square_side_length_l189_189791

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l189_189791


namespace total_cans_in_display_l189_189087

theorem total_cans_in_display :
  ∃ n S : ℕ,
  let a1 := 30,
      d := -4,
      an := 1 
  in 
  (an = a1 + (n-1) * d)
  ∧ (S = n * (a1 + an) / 2)
  ∧ (S = 128) :=
sorry

end total_cans_in_display_l189_189087


namespace complement_of_angle_l189_189378

theorem complement_of_angle (x : ℝ) (h1 : 3 * x + 10 = 90 - x) : 3 * x + 10 = 70 :=
by
  sorry

end complement_of_angle_l189_189378


namespace largest_among_abcd_l189_189460

theorem largest_among_abcd (a b c d k : ℤ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = k + 3 ∧
  a = k + 1 ∧
  b = k - 2 ∧
  d = k - 4 ∧
  c > a ∧
  c > b ∧
  c > d :=
by
  sorry

end largest_among_abcd_l189_189460


namespace max_value_of_quadratic_l189_189137

theorem max_value_of_quadratic :
  ∀ (x : ℝ), ∃ y : ℝ, y = -3 * x^2 + 18 ∧
  (∀ x' : ℝ, -3 * x'^2 + 18 ≤ y) := by
  sorry

end max_value_of_quadratic_l189_189137


namespace amara_remaining_clothes_l189_189417

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l189_189417


namespace vertex_of_parabola_l189_189522

def f (x : ℝ) : ℝ := 2 - (2*x + 1)^2

theorem vertex_of_parabola :
  (∀ x : ℝ, f x ≤ 2) ∧ (f (-1/2) = 2) :=
by
  sorry

end vertex_of_parabola_l189_189522


namespace share_difference_3600_l189_189881

theorem share_difference_3600 (x : ℕ) (p q r : ℕ) (h1 : p = 3 * x) (h2 : q = 7 * x) (h3 : r = 12 * x) (h4 : r - q = 4500) : q - p = 3600 := by
  sorry

end share_difference_3600_l189_189881


namespace kendra_shirts_for_two_weeks_l189_189821

def school_days := 5
def after_school_club_days := 3
def one_week_shirts := school_days + after_school_club_days + 1 (Saturday) + 2 (Sunday)
def two_weeks_shirts := one_week_shirts * 2

theorem kendra_shirts_for_two_weeks : two_weeks_shirts = 22 :=
by
  -- Prove the theorem
  sorry

end kendra_shirts_for_two_weeks_l189_189821


namespace ab_value_l189_189177

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end ab_value_l189_189177


namespace total_daisies_l189_189342

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l189_189342


namespace quadratic_root_and_a_value_l189_189645

theorem quadratic_root_and_a_value (a : ℝ) (h1 : (a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) (h2 : a + 3 ≠ 0) : a = 3 :=
by
  sorry

end quadratic_root_and_a_value_l189_189645


namespace product_of_abc_l189_189092

variable (a b c m : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 200
def condition2 : Prop := 8 * a = m
def condition3 : Prop := m = b - 10
def condition4 : Prop := m = c + 10

-- The theorem to prove
theorem product_of_abc :
  a + b + c = 200 ∧ 8 * a = m ∧ m = b - 10 ∧ m = c + 10 →
  a * b * c = 505860000 / 4913 :=
by
  sorry

end product_of_abc_l189_189092


namespace isaiah_types_more_words_than_micah_l189_189973

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l189_189973


namespace min_soldiers_to_add_l189_189615

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189615


namespace prob_one_mistake_eq_l189_189222

-- Define the probability of making a mistake on a single question
def prob_mistake : ℝ := 0.1

-- Define the probability of answering correctly on a single question
def prob_correct : ℝ := 1 - prob_mistake

-- Define the probability of answering all three questions correctly
def three_correct : ℝ := prob_correct ^ 3

-- Define the probability of making at least one mistake in three questions
def prob_at_least_one_mistake := 1 - three_correct

-- The theorem states that the above probability is equal to 1 - 0.9^3
theorem prob_one_mistake_eq :
  prob_at_least_one_mistake = 1 - (0.9 ^ 3) :=
by
  sorry

end prob_one_mistake_eq_l189_189222


namespace inner_circle_radius_l189_189962

theorem inner_circle_radius :
  ∃ (r : ℝ) (a b c d : ℕ), 
    (r = (-78 + 70 * Real.sqrt 3) / 26) ∧ 
    (a = 78) ∧ 
    (b = 70) ∧ 
    (c = 3) ∧ 
    (d = 26) ∧ 
    (Nat.gcd a d = 1) ∧ 
    (a + b + c + d = 177) := 
sorry

end inner_circle_radius_l189_189962


namespace find_B_squared_l189_189762

def f (x : ℝ) : ℝ := sqrt 27 + 100 / (sqrt 27 + 100 / (sqrt 27 + 100 * x / (x + 1)))

theorem find_B_squared :
  let B := (abs ((- sqrt 27 + sqrt (27 + 40000)) / 200)
           + abs ((- sqrt 27 - sqrt (27 + 40000)) / 200)) in
  B^2 = 4.0027 :=
by
  sorry

end find_B_squared_l189_189762


namespace box_third_dimension_l189_189875

theorem box_third_dimension (num_cubes : ℕ) (cube_volume box_vol : ℝ) (dim1 dim2 h : ℝ) (h_num_cubes : num_cubes = 24) (h_cube_volume : cube_volume = 27) (h_dim1 : dim1 = 9) (h_dim2 : dim2 = 12) (h_box_vol : box_vol = num_cubes * cube_volume) :
  box_vol = dim1 * dim2 * h → h = 6 := 
by
  sorry

end box_third_dimension_l189_189875


namespace int_as_sum_of_squares_l189_189675

theorem int_as_sum_of_squares (n : ℤ) : ∃ a b c : ℤ, n = a^2 + b^2 - c^2 :=
sorry

end int_as_sum_of_squares_l189_189675


namespace lcm_36_100_l189_189305

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l189_189305


namespace last_three_digits_of_5_pow_9000_l189_189104

theorem last_three_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 800]) : 5^9000 ≡ 1 [MOD 800] :=
by
  -- The proof is omitted here according to the instruction
  sorry

end last_three_digits_of_5_pow_9000_l189_189104


namespace pure_imaginary_denom_rationalization_l189_189061

theorem pure_imaginary_denom_rationalization (a : ℝ) : 
  (∃ b : ℝ, 1 - a * Complex.I * Complex.I = b * Complex.I) → a = 0 :=
by
  sorry

end pure_imaginary_denom_rationalization_l189_189061


namespace optimal_room_rate_to_maximize_income_l189_189259

noncomputable def max_income (x : ℝ) : ℝ := x * (300 - 0.5 * (x - 200))

theorem optimal_room_rate_to_maximize_income :
  ∀ x, 200 ≤ x → x ≤ 800 → max_income x ≤ max_income 400 :=
by
  sorry

end optimal_room_rate_to_maximize_income_l189_189259


namespace kendra_shirts_needed_l189_189822

def shirts_needed_per_week (school_days after_school_club_days saturday_shirts sunday_church_shirt sunday_rest_of_day_shirt : ℕ) : ℕ :=
  school_days + after_school_club_days + saturday_shirts + sunday_church_shirt + sunday_rest_of_day_shirt

def shirts_needed (weeks shirts_per_week : ℕ) : ℕ :=
  weeks * shirts_per_week

theorem kendra_shirts_needed : shirts_needed 2 (
  shirts_needed_per_week 5 3 1 1 1
) = 22 :=
by
  simp [shirts_needed, shirts_needed_per_week]
  rfl

end kendra_shirts_needed_l189_189822


namespace equal_share_candy_l189_189596

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l189_189596


namespace solve_log_equation_l189_189991

noncomputable def solution_to_log_equation : ℝ :=
  2

theorem solve_log_equation (x : ℝ) :
  log 2 (4^x + 4) = x + log 2 (2^(x + 1) - 3) ↔ x = 2 := 
by {
  sorry
}

end solve_log_equation_l189_189991


namespace system1_solution_system2_solution_l189_189214

-- Problem 1
theorem system1_solution (x y : ℝ) (h1 : 3 * x - 2 * y = 6) (h2 : 2 * x + 3 * y = 17) : 
  x = 4 ∧ y = 3 :=
by {
  sorry
}

-- Problem 2
theorem system2_solution (x y : ℝ) (h1 : x + 4 * y = 14) 
  (h2 : (x - 3) / 4 - (y - 3) / 3 = 1 / 12) : 
  x = 3 ∧ y = 11 / 4 :=
by {
  sorry
}

end system1_solution_system2_solution_l189_189214


namespace max_gcd_lcm_condition_l189_189068

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l189_189068


namespace proof_theorem_l189_189849

noncomputable def proof_problem (a b c : ℝ) := 
  (2 * b = a + c) ∧ 
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) ∧ 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)

theorem proof_theorem (a b c : ℝ) (h : proof_problem a b c) :
  (a = b ∧ b = c) ∨ 
  (∃ (x : ℝ), x ≠ 0 ∧ a = -4 * x ∧ b = -x ∧ c = 2 * x) :=
by
  sorry

end proof_theorem_l189_189849


namespace molecular_weight_compound_l189_189398

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

end molecular_weight_compound_l189_189398


namespace lcm_of_36_and_100_l189_189291

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l189_189291


namespace compute_expr_l189_189135

theorem compute_expr {x : ℝ} (h : x = 5) : (x^6 - 2 * x^3 + 1) / (x^3 - 1) = 124 :=
by
  sorry

end compute_expr_l189_189135


namespace distinct_triples_l189_189288

theorem distinct_triples (a b c : ℕ) (h₁: 2 * a - 1 = k₁ * b) (h₂: 2 * b - 1 = k₂ * c) (h₃: 2 * c - 1 = k₃ * a) :
  (a, b, c) = (7, 13, 25) ∨ (a, b, c) = (13, 25, 7) ∨ (a, b, c) = (25, 7, 13) := sorry

end distinct_triples_l189_189288


namespace circle_area_l189_189758

theorem circle_area (x y : ℝ) : 
  x^2 + y^2 - 18 * x + 8 * y = -72 → 
  ∃ r : ℝ, r = 5 ∧ π * r ^ 2 = 25 * π := 
by
  sorry

end circle_area_l189_189758


namespace james_tylenol_daily_intake_l189_189956

def tylenol_per_tablet : ℕ := 375
def tablets_per_dose : ℕ := 2
def hours_per_dose : ℕ := 6
def hours_per_day : ℕ := 24

theorem james_tylenol_daily_intake :
  (hours_per_day / hours_per_dose) * (tablets_per_dose * tylenol_per_tablet) = 3000 := by
  sorry

end james_tylenol_daily_intake_l189_189956


namespace ratio_of_refurb_to_new_tshirt_l189_189474

def cost_of_new_tshirt : ℤ := 5
def cost_of_pants : ℤ := 4
def cost_of_skirt : ℤ := 6

-- Total income from selling two new T-shirts, one pair of pants, four skirts, and six refurbished T-shirts is $53.
def total_income : ℤ := 53

-- Total income from selling new items.
def income_from_new_items : ℤ :=
  2 * cost_of_new_tshirt + cost_of_pants + 4 * cost_of_skirt

-- Income from refurbished T-shirts.
def income_from_refurb_tshirts : ℤ :=
  total_income - income_from_new_items

-- Number of refurbished T-shirts sold.
def num_refurb_tshirts_sold : ℤ := 6

-- Price of one refurbished T-shirt.
def cost_of_refurb_tshirt : ℤ :=
  income_from_refurb_tshirts / num_refurb_tshirts_sold

-- Prove the ratio of the price of a refurbished T-shirt to a new T-shirt is 0.5
theorem ratio_of_refurb_to_new_tshirt :
  (cost_of_refurb_tshirt : ℚ) / cost_of_new_tshirt = 0.5 := 
sorry

end ratio_of_refurb_to_new_tshirt_l189_189474


namespace Lance_workdays_per_week_l189_189894

theorem Lance_workdays_per_week (weekly_hours hourly_wage daily_earnings : ℕ) 
  (h1 : weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63) :
  weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end Lance_workdays_per_week_l189_189894


namespace contingency_fund_amount_l189_189944

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l189_189944


namespace three_digit_cubes_divisible_by_4_l189_189170

-- Let's define the conditions in Lean
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Let's combine these conditions to define the target predicate in Lean
def is_target_number (n : ℕ) : Prop := is_three_digit n ∧ is_perfect_cube n ∧ is_divisible_by_4 n

-- The statement to be proven: that there is only one such number
theorem three_digit_cubes_divisible_by_4 : 
  (∃! n, is_target_number n) :=
sorry

end three_digit_cubes_divisible_by_4_l189_189170


namespace least_possible_sections_l189_189264

theorem least_possible_sections (A C N : ℕ) (h1 : 7 * A = 11 * C) (h2 : N = A + C) : N = 18 :=
sorry

end least_possible_sections_l189_189264


namespace spoiled_apples_l189_189538

theorem spoiled_apples (S G : ℕ) (h1 : S + G = 8) (h2 : (G * (G - 1)) / 2 = 21) : S = 1 :=
by
  sorry

end spoiled_apples_l189_189538


namespace find_p_over_q_at_0_l189_189280

noncomputable def p (x : ℝ) := 3 * (x - 4) * (x - 1)
noncomputable def q (x : ℝ) := (x + 3) * (x - 1) * (x - 4)

theorem find_p_over_q_at_0 : (p 0) / (q 0) = 1 := 
by
  sorry

end find_p_over_q_at_0_l189_189280


namespace surface_area_of_sphere_l189_189743

-- Define the conditions from the problem.

variables (r R : ℝ) -- r is the radius of the cross-section, R is the radius of the sphere.
variables (π : ℝ := Real.pi) -- Define π using the real pi constant.
variables (h_dist : 1 = 1) -- Distance from the plane to the center is 1 unit.
variables (h_area_cross_section : π = π * r^2) -- Area of the cross-section is π.

-- State to prove the surface area of the sphere is 8π.
theorem surface_area_of_sphere :
    ∃ (R : ℝ), (R^2 = 2) → (4 * π * R^ 2 = 8 * π) := sorry

end surface_area_of_sphere_l189_189743


namespace opposite_of_one_sixth_l189_189519

theorem opposite_of_one_sixth : (-(1 / 6) : ℚ) = -1 / 6 := 
by
  sorry

end opposite_of_one_sixth_l189_189519


namespace area_of_DEF_l189_189542

variable (t4_area t5_area t6_area : ℝ) (a_DEF : ℝ)

def similar_triangles_area := (t4_area = 1) ∧ (t5_area = 16) ∧ (t6_area = 36)

theorem area_of_DEF 
  (h : similar_triangles_area t4_area t5_area t6_area) :
  a_DEF = 121 := sorry

end area_of_DEF_l189_189542


namespace PQRS_value_l189_189031

theorem PQRS_value :
  let P := (Real.sqrt 2011 + Real.sqrt 2010)
  let Q := (-Real.sqrt 2011 - Real.sqrt 2010)
  let R := (Real.sqrt 2011 - Real.sqrt 2010)
  let S := (Real.sqrt 2010 - Real.sqrt 2011)
  P * Q * R * S = -1 :=
by
  sorry

end PQRS_value_l189_189031


namespace minimum_value_l189_189922

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 4) : 
  ∃ (x : ℝ), (x = 2) ∧ (∀ (p q : ℝ), q > 0 → p > 0 → 2 * p + q = 4 → x ≤ (1 / p + 2 / q)) := 
sorry

end minimum_value_l189_189922


namespace brass_selling_price_l189_189989

noncomputable def copper_price : ℝ := 0.65
noncomputable def zinc_price : ℝ := 0.30
noncomputable def total_weight_brass : ℝ := 70
noncomputable def weight_copper : ℝ := 30
noncomputable def weight_zinc := total_weight_brass - weight_copper
noncomputable def cost_copper := weight_copper * copper_price
noncomputable def cost_zinc := weight_zinc * zinc_price
noncomputable def total_cost := cost_copper + cost_zinc
noncomputable def selling_price_per_pound := total_cost / total_weight_brass

theorem brass_selling_price :
  selling_price_per_pound = 0.45 :=
by
  sorry

end brass_selling_price_l189_189989


namespace intersection_of_M_and_N_l189_189326

open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by 
  sorry

end intersection_of_M_and_N_l189_189326


namespace arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l189_189281

open Real

theorem arctan_sum_lt_pi_div_two_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  arctan x + arctan y < (π / 2) ↔ x * y < 1 :=
sorry

theorem arctan_sum_lt_pi_iff (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  arctan x + arctan y + arctan z < π ↔ x * y * z < x + y + z :=
sorry

end arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l189_189281


namespace average_xyz_l189_189968

theorem average_xyz (x y z : ℝ) 
  (h1 : 2003 * z - 4006 * x = 1002) 
  (h2 : 2003 * y + 6009 * x = 4004) : (x + y + z) / 3 = 5 / 6 :=
by
  sorry

end average_xyz_l189_189968


namespace pentagon_edges_same_color_l189_189874

theorem pentagon_edges_same_color
  (A B : Fin 5 → Fin 5)
  (C : (Fin 5 → Fin 5) × (Fin 5 → Fin 5) → Bool)
  (condition : ∀ (i j : Fin 5), ∀ (k l m : Fin 5), (C (i, j) = C (k, l) → C (i, j) ≠ C (k, m))) :
  (∀ (x : Fin 5), C (A x, A ((x + 1) % 5)) = C (B x, B ((x + 1) % 5))) :=
by
sorry

end pentagon_edges_same_color_l189_189874


namespace speed_of_man_in_still_water_l189_189260

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 18) (h2 : v_m - v_s = 13) : v_m = 15.5 :=
by {
  -- Proof is not required as per the instructions
  sorry
}

end speed_of_man_in_still_water_l189_189260


namespace principal_amount_l189_189506

theorem principal_amount (P : ℝ) (h : (P * 0.1236) - (P * 0.12) = 36) : P = 10000 := 
sorry

end principal_amount_l189_189506


namespace range_of_a_l189_189178

theorem range_of_a (a : ℝ) : (4 - a < 0) → (a > 4) :=
by
  intros h
  sorry

end range_of_a_l189_189178


namespace max_gcd_lcm_l189_189071

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l189_189071


namespace distribution_plans_l189_189558

theorem distribution_plans :
  let classes := range 3,
    teachers := range 5 in
  (∃ (class_assignments : teachers → classes → Prop),
    (∀ t, ∃ c, class_assignments t c) ∧ -- every teacher is assigned to a class
    (∀ c, 1 ≤ (Finset.card (Finset.filter (class_assignments · c) teachers)) ∧ -- each class has at least 1 teacher
        (Finset.card (Finset.filter (class_assignments · c) teachers) ≤ 2))) →
  ((Finset.card (Finset.filter (λ c, Finset.card (Finset.filter (class_assignments · c) teachers) = 2) classes) = 1) ∧ -- one class has exactly 2 teachers
   ∀ c, (1 ≤ Finset.card (Finset.filter (class_assignments · c) teachers) ∧
        Finset.card (Finset.filter (class_assignments · c) teachers) ≤ 2)) →
  ∃ N, N = 30 := 
sorry

end distribution_plans_l189_189558


namespace contingency_fund_correct_l189_189939

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l189_189939


namespace sum_of_cubes_l189_189681

theorem sum_of_cubes (p q r : ℝ) (h1 : p + q + r = 7) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 :=
by
  sorry

end sum_of_cubes_l189_189681


namespace elgin_money_l189_189419

theorem elgin_money {A B C D E : ℤ} 
  (h1 : |A - B| = 19) 
  (h2 : |B - C| = 9) 
  (h3 : |C - D| = 5) 
  (h4 : |D - E| = 4) 
  (h5 : |E - A| = 11) 
  (h6 : A + B + C + D + E = 60) : 
  E = 10 := 
sorry

end elgin_money_l189_189419


namespace algebra_expression_value_l189_189154

theorem algebra_expression_value (x y : ℝ) (h : x = 2 * y + 1) : x^2 - 4 * x * y + 4 * y^2 = 1 := 
by 
  sorry

end algebra_expression_value_l189_189154


namespace base_eight_seventeen_five_is_one_two_five_l189_189395

def base_eight_to_base_ten (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_seventeen_five_is_one_two_five :
  base_eight_to_base_ten 175 = 125 :=
by
  sorry

end base_eight_seventeen_five_is_one_two_five_l189_189395


namespace sum_geq_three_implies_one_geq_two_l189_189707

theorem sum_geq_three_implies_one_geq_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by { sorry }

end sum_geq_three_implies_one_geq_two_l189_189707


namespace minimum_soldiers_to_add_l189_189631

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189631


namespace not_p_and_pq_false_not_necessarily_p_or_q_l189_189333

theorem not_p_and_pq_false_not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) := by
  sorry

end not_p_and_pq_false_not_necessarily_p_or_q_l189_189333


namespace find_x_l189_189331

theorem find_x (x : ℕ) (h : 1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) : x = 100 :=
by {
  sorry
}

end find_x_l189_189331


namespace solution_set_f_x_le_5_l189_189032

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 + Real.log x / Real.log 2 else x^2 - x - 1

theorem solution_set_f_x_le_5 : {x : ℝ | f x ≤ 5} = Set.Icc (-2 : ℝ) 4 := by
  sorry

end solution_set_f_x_le_5_l189_189032


namespace find_pots_l189_189469

def num_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def total_cost := 390

theorem find_pots (x y : ℕ) (h1 : x + y = num_pots) (h2 : cost_green_lily * x + cost_spider_plant * y = total_cost) :
  x = 38 ∧ y = 8 :=
by
  sorry

end find_pots_l189_189469


namespace sum_of_ages_l189_189394

def Tyler_age : ℕ := 5

def Clay_age (T C : ℕ) : Prop :=
  T = 3 * C + 1

theorem sum_of_ages (C : ℕ) (h : Clay_age Tyler_age C) :
  Tyler_age + C = 6 :=
sorry

end sum_of_ages_l189_189394


namespace harmonic_mean_lcm_gcd_sum_l189_189376

theorem harmonic_mean_lcm_gcd_sum {m n : ℕ} (h_lcm : Nat.lcm m n = 210) (h_gcd : Nat.gcd m n = 6) (h_sum : m + n = 72) :
  (1 / (m : ℚ) + 1 / (n : ℚ)) = 2 / 35 := 
sorry

end harmonic_mean_lcm_gcd_sum_l189_189376


namespace soldiers_to_add_l189_189632

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189632


namespace number_of_Cl_atoms_l189_189114

def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

def H_atoms : ℕ := 1
def O_atoms : ℕ := 2
def total_molecular_weight : ℝ := 68

theorem number_of_Cl_atoms :
  (total_molecular_weight - (H_atoms * atomic_weight_H + O_atoms * atomic_weight_O)) / atomic_weight_Cl = 1 :=
by
  -- proof to show this holds
  sorry

end number_of_Cl_atoms_l189_189114


namespace berries_from_fourth_bush_l189_189356

def number_of_berries (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 5 => 19
  | _ => sorry  -- Assume the given pattern

theorem berries_from_fourth_bush : number_of_berries 4 = 12 :=
by sorry

end berries_from_fourth_bush_l189_189356


namespace sequence_bounded_l189_189912

theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_dep : ∀ k n m l, k + n = m + l → (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ m M : ℝ, ∀ n, m ≤ a n ∧ a n ≤ M :=
sorry

end sequence_bounded_l189_189912


namespace girls_attending_picnic_l189_189455

theorem girls_attending_picnic (g b : ℕ) (h1 : g + b = 1200) (h2 : (2 * g) / 3 + b / 2 = 730) : (2 * g) / 3 = 520 :=
by
  -- The proof steps would go here.
  sorry

end girls_attending_picnic_l189_189455


namespace general_form_of_aₙ_find_m_T_l189_189090

-- Definitions

def has_property_aₙ (a : ℕ → ℕ ) : Prop :=
  ∀ n : ℕ, 0 < n → (a 1 + ∑ i in Finset.range n, (1 / (i + 1)) * a (i + 2)) = a n + 1

def initial_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1) * a (n - i)

def b (S: ℕ → ℕ) (n: ℕ) : ℕ :=
  1 / (3 * S n)

def T (b: ℕ → ℕ) (n: ℕ) : ℕ :=
  ∑ i in Finset.range n, b i

-- Proof goals

theorem general_form_of_aₙ (a : ℕ → ℕ) (n : ℕ) (h₁ : has_property_aₙ a) (h₂ : initial_condition a) : ∀ n, a n = n := 
  sorry

theorem find_m_T (a : ℕ → ℕ) (S: ℕ → ℕ) (b: ℕ → ℕ) (T : ℕ → ℕ) (h_a : ∀ n, a n = n) : ∃ m, ∀ n, T n < m / 2024 :=
  sorry

end general_form_of_aₙ_find_m_T_l189_189090


namespace flea_jump_no_lava_l189_189197

theorem flea_jump_no_lava
  (A B F : ℕ)
  (n : ℕ) 
  (h_posA : 0 < A)
  (h_posB : 0 < B)
  (h_AB : A < B)
  (h_2A : B < 2 * A)
  (h_ineq1 : A * (n + 1) ≤ B - A * n)
  (h_ineq2 : B - A < A * n) :
  ∃ (F : ℕ), F = (n - 1) * A + B := sorry

end flea_jump_no_lava_l189_189197


namespace first_customer_bought_5_l189_189957

variables 
  (x : ℕ) -- Number of boxes the first customer bought
  (x2 : ℕ) -- Number of boxes the second customer bought
  (x3 : ℕ) -- Number of boxes the third customer bought
  (x4 : ℕ) -- Number of boxes the fourth customer bought
  (x5 : ℕ) -- Number of boxes the fifth customer bought

def goal : ℕ := 150
def remaining_boxes : ℕ := 75
def sold_boxes := x + x2 + x3 + x4 + x5

axiom second_customer (hx2 : x2 = 4 * x) : True
axiom third_customer (hx3 : x3 = (x2 / 2)) : True
axiom fourth_customer (hx4 : x4 = 3 * x3) : True
axiom fifth_customer (hx5 : x5 = 10) : True
axiom sales_goal (hgoal : sold_boxes = goal - remaining_boxes) : True

theorem first_customer_bought_5 (hx2 : x2 = 4 * x) 
                                (hx3 : x3 = (x2 / 2)) 
                                (hx4 : x4 = 3 * x3) 
                                (hx5 : x5 = 10) 
                                (hgoal : sold_boxes = goal - remaining_boxes) : 
                                x = 5 :=
by
  -- Here, we would perform the proof steps
  sorry

end first_customer_bought_5_l189_189957


namespace Q_subset_P_l189_189582

-- Definitions
def P : Set ℝ := {x : ℝ | x ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Statement to prove
theorem Q_subset_P : Q ⊆ P :=
sorry

end Q_subset_P_l189_189582


namespace jacobs_hourly_wage_l189_189815

theorem jacobs_hourly_wage (jake_total_earnings : ℕ) (jake_days : ℕ) (hours_per_day : ℕ) (jake_thrice_jacob : ℕ) 
    (h_total_jake : jake_total_earnings = 720) 
    (h_jake_days : jake_days = 5) 
    (h_hours_per_day : hours_per_day = 8)
    (h_jake_thrice_jacob : jake_thrice_jacob = 3) 
    (jacob_hourly_wage : ℕ) :
  jacob_hourly_wage = 6 := 
by
  sorry

end jacobs_hourly_wage_l189_189815


namespace inscribed_circle_radius_of_rhombus_l189_189399

theorem inscribed_circle_radius_of_rhombus (d1 d2 : ℝ) (a r : ℝ) : 
  d1 = 15 → d2 = 24 → a = Real.sqrt ((15 / 2)^2 + (24 / 2)^2) → 
  (d1 * d2) / 2 = 2 * a * r → 
  r = 60.07 / 13 :=
by
  intros h1 h2 h3 h4
  sorry

end inscribed_circle_radius_of_rhombus_l189_189399


namespace num_perfect_square_factors_1800_l189_189931

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end num_perfect_square_factors_1800_l189_189931


namespace Amelia_sell_JetBars_l189_189750

theorem Amelia_sell_JetBars (M : ℕ) (h : 2 * M - 16 = 74) : M = 45 := by
  sorry

end Amelia_sell_JetBars_l189_189750


namespace soldiers_to_add_l189_189633

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189633


namespace smallest_number_is_51_l189_189997

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l189_189997


namespace number_of_weeks_in_a_single_harvest_season_l189_189485

-- Define constants based on conditions
def weeklyEarnings : ℕ := 1357
def totalHarvestSeasons : ℕ := 73
def totalEarnings : ℕ := 22090603

-- Prove the number of weeks in a single harvest season
theorem number_of_weeks_in_a_single_harvest_season :
  (totalEarnings / weeklyEarnings) / totalHarvestSeasons = 223 := 
  by
    sorry

end number_of_weeks_in_a_single_harvest_season_l189_189485


namespace probability_of_6_consecutive_heads_l189_189258

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l189_189258


namespace couples_at_prom_l189_189277

theorem couples_at_prom (total_students attending_alone attending_with_partners couples : ℕ) 
  (h1 : total_students = 123) 
  (h2 : attending_alone = 3) 
  (h3 : attending_with_partners = total_students - attending_alone) 
  (h4 : couples = attending_with_partners / 2) : 
  couples = 60 := 
by 
  sorry

end couples_at_prom_l189_189277


namespace eq_of_op_star_l189_189452

theorem eq_of_op_star (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^b^2)^n = a^(bn)^2 ↔ n = 1 := by
sorry

end eq_of_op_star_l189_189452


namespace min_a_4_l189_189769

theorem min_a_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 9 * x + y = x * y) : 
  4 * x + y ≥ 25 :=
sorry

end min_a_4_l189_189769


namespace log_one_plus_two_x_lt_two_x_l189_189175
open Real

theorem log_one_plus_two_x_lt_two_x {x : ℝ} (hx : x > 0) : log (1 + 2 * x) < 2 * x :=
sorry

end log_one_plus_two_x_lt_two_x_l189_189175


namespace find_x_l189_189183

-- Define conditions
def simple_interest (x y : ℝ) : Prop :=
  x * y * 2 / 100 = 800

def compound_interest (x y : ℝ) : Prop :=
  x * ((1 + y / 100)^2 - 1) = 820

-- Prove x = 8000 given the conditions
theorem find_x (x y : ℝ) (h1 : simple_interest x y) (h2 : compound_interest x y) : x = 8000 :=
  sorry

end find_x_l189_189183


namespace min_max_values_l189_189511

noncomputable def f (x : ℝ) := cos x + (x + 1) * sin x + 1

theorem min_max_values :
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≥ - (3 * π) / 2) ∧
  (∀ x ∈ Icc (0 : ℝ) (2 * π), f x ≤ (π / 2 + 2)) :=
sorry

end min_max_values_l189_189511


namespace probability_at_least_6_heads_in_10_flips_l189_189252

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l189_189252


namespace fraction_value_condition_l189_189314

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l189_189314


namespace square_side_length_l189_189793

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l189_189793


namespace passing_time_for_platform_l189_189729

def train_length : ℕ := 1100
def time_to_cross_tree : ℕ := 110
def platform_length : ℕ := 700
def speed := train_length / time_to_cross_tree
def combined_length := train_length + platform_length

theorem passing_time_for_platform : 
  let speed := train_length / time_to_cross_tree
  let combined_length := train_length + platform_length
  combined_length / speed = 180 :=
by
  sorry

end passing_time_for_platform_l189_189729


namespace probability_same_color_l189_189958

-- Definitions according to conditions
def total_socks : ℕ := 24
def blue_pairs : ℕ := 7
def green_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_blue_socks : ℕ := blue_pairs * 2
def total_green_socks : ℕ := green_pairs * 2
def total_red_socks : ℕ := red_pairs * 2

-- Probability calculations
def probability_blue : ℚ := (total_blue_socks * (total_blue_socks - 1)) / (total_socks * (total_socks - 1))
def probability_green : ℚ := (total_green_socks * (total_green_socks - 1)) / (total_socks * (total_socks - 1))
def probability_red : ℚ := (total_red_socks * (total_red_socks - 1)) / (total_socks * (total_socks - 1))

def total_probability : ℚ := probability_blue + probability_green + probability_red

theorem probability_same_color : total_probability = 28 / 69 :=
by
  sorry

end probability_same_color_l189_189958


namespace general_formula_l189_189198

noncomputable def a : ℕ → ℕ
| 0       => 5
| (n + 1) => 2 * a n + 3

theorem general_formula : ∀ n, a n = 2 ^ (n + 2) - 3 :=
by
  sorry

end general_formula_l189_189198


namespace simplify_cube_root_18_24_30_l189_189838

noncomputable def cube_root_simplification (a b c : ℕ) : ℕ :=
  let sum_cubes := a^3 + b^3 + c^3
  36

theorem simplify_cube_root_18_24_30 : 
  cube_root_simplification 18 24 30 = 36 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_cube_root_18_24_30_l189_189838


namespace probability_of_6_consecutive_heads_l189_189256

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l189_189256


namespace frequency_of_group_samples_l189_189745

-- Conditions
def sample_capacity : ℕ := 32
def group_frequency : ℝ := 0.125

-- Theorem statement
theorem frequency_of_group_samples : group_frequency * sample_capacity = 4 :=
by sorry

end frequency_of_group_samples_l189_189745


namespace vector_dot_product_proof_l189_189169

variable (a b : ℝ × ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vector_dot_product_proof
  (h1 : a = (1, -3))
  (h2 : b = (3, 7)) :
  dot_product a b = -18 :=
by 
  sorry

end vector_dot_product_proof_l189_189169


namespace susan_coins_value_l189_189217

-- Define the conditions as Lean functions and statements.
def total_coins (n d : ℕ) := n + d = 30
def value_if_swapped (n : ℕ) := 10 * n + 5 * (30 - n)
def value_original (n : ℕ) := 5 * n + 10 * (30 - n)
def conditions (n : ℕ) := value_if_swapped n = value_original n + 90

-- The proof statement
theorem susan_coins_value (n d : ℕ) (h1 : total_coins n d) (h2 : conditions n) : 5 * n + 10 * d = 180 := by
  sorry

end susan_coins_value_l189_189217


namespace find_f6_l189_189986

-- Define the function f
variable {f : ℝ → ℝ}
-- The function satisfies f(x + y) = f(x) + f(y) for all real numbers x and y
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
-- f(4) = 6
axiom f_of_4 : f 4 = 6

theorem find_f6 : f 6 = 9 :=
by
    sorry

end find_f6_l189_189986


namespace range_of_f_l189_189463

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) * Real.exp x

theorem range_of_f'_over_f (f : ℝ → ℝ)
  (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x)
  (h2 : f 0 = 1)
  (x : ℝ) (hx : 0 < x) :
  1 < (deriv f x) / (f x) ∧ (deriv f x) / (f x) ≤ 2 :=
by
  sorry

end range_of_f_l189_189463


namespace ratio_of_chocolate_to_regular_milk_l189_189406

def total_cartons : Nat := 24
def regular_milk_cartons : Nat := 3
def chocolate_milk_cartons : Nat := total_cartons - regular_milk_cartons

theorem ratio_of_chocolate_to_regular_milk (h1 : total_cartons = 24) (h2 : regular_milk_cartons = 3) :
  chocolate_milk_cartons / regular_milk_cartons = 7 :=
by 
  -- Skipping proof with sorry
  sorry

end ratio_of_chocolate_to_regular_milk_l189_189406


namespace simplify_fraction_l189_189676

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := 
by 
  sorry

end simplify_fraction_l189_189676


namespace matrix_property_l189_189823

-- Given conditions
variables (a b : ℝ)
variables (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Assume b > a^2 and properties of matrix A
theorem matrix_property (h1 : b > a^2) (h2 : Matrix.trace A = 2 * a) (h3 : Matrix.det A = b) :
  Matrix.det (A * A - 2 * a • A + b • (1 : Matrix (Fin 2) (Fin 2) ℝ)) = 0 :=
sorry

end matrix_property_l189_189823


namespace swimming_lane_length_l189_189486

theorem swimming_lane_length (round_trips : ℕ) (total_distance : ℕ) (lane_length : ℕ) 
  (h1 : round_trips = 4) (h2 : total_distance = 800) 
  (h3 : total_distance = lane_length * (round_trips * 2)) : 
  lane_length = 100 := 
by
  sorry

end swimming_lane_length_l189_189486


namespace average_distance_one_hour_l189_189224

theorem average_distance_one_hour (d : ℝ) (t : ℝ) (h1 : d = 100) (h2 : t = 5 / 4) : (d / t) = 80 :=
by
  sorry

end average_distance_one_hour_l189_189224


namespace find_m_value_l189_189216

theorem find_m_value (m : ℚ) :
  ∀ x, (3 : ℚ) * x^2 - 7 * x + m = 0 ↔ discriminant (3 : ℚ) (-7 : ℚ) m = 0 → m = 49 / 12 :=
by {
  intros,
  sorry
}

end find_m_value_l189_189216


namespace cream_ratio_l189_189672

variable (servings : ℕ) (fat_per_serving : ℕ) (fat_per_cup : ℕ)
variable (h_servings : servings = 4) (h_fat_per_serving : fat_per_serving = 11) (h_fat_per_cup : fat_per_cup = 88)

theorem cream_ratio (total_fat : ℕ) (h_total_fat : total_fat = fat_per_serving * servings) :
  (total_fat : ℚ) / fat_per_cup = 1 / 2 :=
by
  sorry

end cream_ratio_l189_189672


namespace find_f_neg2_l189_189380

theorem find_f_neg2 (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^5 + a*x^3 + x^2 + b*x + 2) (h₂ : f 2 = 3) : f (-2) = 9 :=
by
  sorry

end find_f_neg2_l189_189380


namespace triangle_probability_is_9_over_35_l189_189498

-- Define the stick lengths
def stick_lengths : List ℕ := [2, 3, 5, 7, 11, 13, 17]

-- Define the function that checks the triangle inequality for three lengths
def can_form_triangle (a b c : ℕ) : Bool :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Get all combinations of three stick lengths
def all_combinations : List (ℕ × ℕ × ℕ) :=
  List.combinations 3 stick_lengths |>.map (λ l => (l[0], l[1], l[2]))

-- Get the valid combinations that can form a triangle
def valid_combinations : List (ℕ × ℕ × ℕ) :=
  all_combinations.filter (λ ⟨a, b, c⟩ => can_form_triangle a b c)

-- Calculate the probability as a rational number
def triangle_probability : ℚ :=
  valid_combinations.length / all_combinations.length

-- The theorem to prove
theorem triangle_probability_is_9_over_35 : triangle_probability = 9 / 35 :=
  sorry

end triangle_probability_is_9_over_35_l189_189498


namespace cheaper_store_difference_in_cents_l189_189125

/-- Given the following conditions:
1. Best Deals offers \$12 off the list price of \$52.99.
2. Market Value offers 20% off the list price of \$52.99.
 -/
theorem cheaper_store_difference_in_cents :
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  best_deals_price < market_value_price →
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  difference_in_cents = 140 := by
  intro h
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  sorry

end cheaper_store_difference_in_cents_l189_189125


namespace car_distance_l189_189733

variable (T_initial : ℕ) (T_new : ℕ) (S : ℕ) (D : ℕ)

noncomputable def calculate_distance (T_initial T_new S : ℕ) : ℕ :=
  S * T_new

theorem car_distance :
  T_initial = 6 →
  T_new = (3 / 2) * T_initial →
  S = 16 →
  D = calculate_distance T_initial T_new S →
  D = 144 :=
by
  sorry

end car_distance_l189_189733


namespace inequality_proof_l189_189027

theorem inequality_proof
  (a b c A α : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α)
  (h_sum : a + b + c = A)
  (h_A : A ≤ 1) :
  (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ≥ 3 * (3 / A - A / 3) ^ α :=
by
  sorry

end inequality_proof_l189_189027


namespace find_q_l189_189056

theorem find_q (a b m p q : ℚ) 
  (h1 : ∀ x, x^2 - m * x + 3 = (x - a) * (x - b)) 
  (h2 : a * b = 3) 
  (h3 : (x^2 - p * x + q) = (x - (a + 1/b)) * (x - (b + 1/a))) : 
  q = 16 / 3 := 
by sorry

end find_q_l189_189056


namespace gcd_of_given_lengths_l189_189282

def gcd_of_lengths_is_eight : Prop :=
  let lengths := [48, 64, 80, 120]
  ∃ d, d = 8 ∧ (∀ n ∈ lengths, d ∣ n)

theorem gcd_of_given_lengths : gcd_of_lengths_is_eight := 
  sorry

end gcd_of_given_lengths_l189_189282


namespace max_value_of_f_on_S_l189_189496

noncomputable def S : Set ℝ := { x | x^4 - 13 * x^2 + 36 ≤ 0 }
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_f_on_S : ∃ x ∈ S, ∀ y ∈ S, f y ≤ f x ∧ f x = 18 :=
by
  sorry

end max_value_of_f_on_S_l189_189496


namespace smallest_number_is_51_l189_189996

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l189_189996


namespace probability_both_selected_l189_189860

theorem probability_both_selected (P_C : ℚ) (P_B : ℚ) (hC : P_C = 4/5) (hB : P_B = 3/5) : 
  ((4/5) * (3/5)) = (12/25) := by
  sorry

end probability_both_selected_l189_189860


namespace inequality_proof_l189_189200

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
by
  sorry

end inequality_proof_l189_189200


namespace fill_tank_with_two_pipes_l189_189525

def Pipe (Rate : Type) := Rate

theorem fill_tank_with_two_pipes
  (capacity : ℝ)
  (three_pipes_fill_time : ℝ)
  (h1 : three_pipes_fill_time = 12)
  (pipe_rate : ℝ)
  (h2 : pipe_rate = capacity / 36) :
  2 * pipe_rate * 18 = capacity := 
by 
  sorry

end fill_tank_with_two_pipes_l189_189525


namespace total_theme_parks_l189_189472

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end total_theme_parks_l189_189472


namespace roots_are_distinct_and_negative_l189_189442

theorem roots_are_distinct_and_negative : 
  (∀ x : ℝ, x^2 + m * x + 1 = 0 → ∃! (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2) ↔ m > 2 :=
by
  sorry

end roots_are_distinct_and_negative_l189_189442


namespace derrick_has_34_pictures_l189_189208

-- Assume Ralph has 26 pictures of wild animals
def ralph_pictures : ℕ := 26

-- Derrick has 8 more pictures than Ralph
def derrick_pictures : ℕ := ralph_pictures + 8

-- Prove that Derrick has 34 pictures of wild animals
theorem derrick_has_34_pictures : derrick_pictures = 34 := by
  sorry

end derrick_has_34_pictures_l189_189208


namespace license_plate_count_l189_189930

-- Define the number of letters and digits
def num_letters := 26
def num_digits := 10
def num_odd_digits := 5  -- (1, 3, 5, 7, 9)
def num_even_digits := 5  -- (0, 2, 4, 6, 8)

-- Calculate the number of possible license plates
theorem license_plate_count : 
  (num_letters ^ 3) * ((num_even_digits * num_odd_digits * num_digits) * 3) = 13182000 :=
by sorry

end license_plate_count_l189_189930


namespace bailey_credit_cards_l189_189420

theorem bailey_credit_cards (dog_treats : ℕ) (chew_toys : ℕ) (rawhide_bones : ℕ) (items_per_charge : ℕ) (total_items : ℕ) (credit_cards : ℕ)
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : rawhide_bones = 10)
  (h4 : items_per_charge = 5)
  (h5 : total_items = dog_treats + chew_toys + rawhide_bones)
  (h6 : credit_cards = total_items / items_per_charge) :
  credit_cards = 4 :=
by
  sorry

end bailey_credit_cards_l189_189420


namespace younger_son_age_in_30_years_l189_189691

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end younger_son_age_in_30_years_l189_189691


namespace num_perfect_square_factors_of_1800_l189_189932

theorem num_perfect_square_factors_of_1800 : 
  ∃ n : ℕ, n = 8 ∧ ∀ m : ℕ, m ∣ 1800 → (∃ k : ℕ, m = k^2) ↔ m ∈ {d | d ∣ 1800 ∧ is_square d} := 
sorry

end num_perfect_square_factors_of_1800_l189_189932


namespace fraction_of_triangle_area_l189_189366

open Real

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) / 2)

def A : point := (2, 0)
def B : point := (8, 12)
def C : point := (14, 0)

def X : point := (6, 0)
def Y : point := (8, 4)
def Z : point := (10, 0)

theorem fraction_of_triangle_area :
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  sorry

end fraction_of_triangle_area_l189_189366


namespace original_intensity_45_percent_l189_189079

variable (I : ℝ) -- Intensity of the original red paint in percentage.

-- Conditions
variable (h1 : 25 * 0.25 + 0.75 * I = 40) -- Given conditions about the intensities and the new solution.
variable (h2 : ∀ I : ℝ, 0.75 * I + 25 * 0.25 = 40) -- Rewriting the given condition to look specifically for I.

theorem original_intensity_45_percent (I : ℝ) (h1 : 25 * 0.25 + 0.75 * I = 40) : I = 45 := by
  -- We only need the statement. Proof is not required.
  sorry

end original_intensity_45_percent_l189_189079


namespace cats_owners_percentage_l189_189648

noncomputable def percentage_of_students_owning_cats (total_students : ℕ) (cats_owners : ℕ) : ℚ :=
  (cats_owners : ℚ) / (total_students : ℚ) * 100

theorem cats_owners_percentage (total_students : ℕ) (cats_owners : ℕ)
  (dogs_owners : ℕ) (birds_owners : ℕ)
  (h_total_students : total_students = 400)
  (h_cats_owners : cats_owners = 80)
  (h_dogs_owners : dogs_owners = 120)
  (h_birds_owners : birds_owners = 40) :
  percentage_of_students_owning_cats total_students cats_owners = 20 :=
by {
  -- We state the proof but leave it as sorry so it's an incomplete placeholder.
  sorry
}

end cats_owners_percentage_l189_189648


namespace min_max_values_on_interval_l189_189513

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l189_189513


namespace personBCatchesPersonAAtB_l189_189072

-- Definitions based on the given problem's conditions
def personADepartsTime : ℕ := 8 * 60  -- Person A departs at 8:00 AM, given in minutes
def personBDepartsTime : ℕ := 9 * 60  -- Person B departs at 9:00 AM, given in minutes
def catchUpTime : ℕ := 11 * 60        -- Persons meet at 11:00 AM, given in minutes
def returnMultiplier : ℕ := 2         -- Person B returns at double the speed
def chaseMultiplier : ℕ := 2          -- After returning, Person B doubles their speed again

-- Exact question we want to prove
def meetAtBTime : ℕ := 12 * 60 + 48   -- Time when Person B catches up with Person A at point B

-- Statement to be proven
theorem personBCatchesPersonAAtB :
  ∀ (VA VB : ℕ) (x : ℕ),
    VA = 2 * x ∧ VB = 3 * x →
    ∃ t : ℕ, t = meetAtBTime := by
  sorry

end personBCatchesPersonAAtB_l189_189072


namespace range_of_a_l189_189780

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x
noncomputable def k (x : ℝ) : ℝ := (Real.log x + x) / x^2

theorem range_of_a (a : ℝ) (h_zero : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l189_189780


namespace phi_value_l189_189034

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h : |φ| < π / 2) :
  (∀ x : ℝ, f (x + π / 3) φ = f (-(x + π / 3)) φ) → φ = -(π / 6) :=
by
  intro h'
  sorry

end phi_value_l189_189034


namespace minimum_value_of_f_is_15_l189_189935

noncomputable def f (x : ℝ) : ℝ := 9 * x + (1 / (x - 1))

theorem minimum_value_of_f_is_15 (h : ∀ x, x > 1) : ∃ x, x > 1 ∧ f x = 15 :=
by sorry

end minimum_value_of_f_is_15_l189_189935


namespace equivalent_octal_to_decimal_l189_189265

def octal_to_decimal (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n+1 => (n % 10) + 8 * octal_to_decimal (n / 10)

theorem equivalent_octal_to_decimal : octal_to_decimal 753 = 491 :=
by
  sorry

end equivalent_octal_to_decimal_l189_189265


namespace gcd_1734_816_1343_l189_189086

theorem gcd_1734_816_1343 : Int.gcd (Int.gcd 1734 816) 1343 = 17 :=
by
  sorry

end gcd_1734_816_1343_l189_189086


namespace contingency_fund_amount_l189_189943

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l189_189943


namespace greatest_possible_y_l189_189680

theorem greatest_possible_y (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 :=
sorry

end greatest_possible_y_l189_189680


namespace at_least_6_heads_in_10_flips_l189_189247

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l189_189247


namespace soybean_cornmeal_proof_l189_189502

theorem soybean_cornmeal_proof :
  ∃ (x y : ℝ), 
    (0.14 * x + 0.07 * y = 0.13 * 280) ∧
    (x + y = 280) ∧
    (x = 240) ∧
    (y = 40) :=
by
  sorry

end soybean_cornmeal_proof_l189_189502


namespace simplify_expression_l189_189677

theorem simplify_expression (x y : ℝ) : 
    3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := 
by 
    sorry

end simplify_expression_l189_189677


namespace Olivia_hours_worked_on_Monday_l189_189361

/-- Olivia works on multiple days in a week with given wages per hour and total income -/
theorem Olivia_hours_worked_on_Monday 
  (M : ℕ)  -- Hours worked on Monday
  (rate_per_hour : ℕ := 9) -- Olivia’s earning rate per hour
  (hours_Wednesday : ℕ := 3)  -- Hours worked on Wednesday
  (hours_Friday : ℕ := 6)  -- Hours worked on Friday
  (total_income : ℕ := 117)  -- Total income earned this week
  (hours_total : ℕ := hours_Wednesday + hours_Friday + M)
  (income_calc : ℕ := rate_per_hour * hours_total) :
  -- Prove that the hours worked on Monday is 4 given the conditions
  income_calc = total_income → M = 4 :=
by
  sorry

end Olivia_hours_worked_on_Monday_l189_189361


namespace problem1_problem2_l189_189578

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l189_189578


namespace min_soldiers_to_add_l189_189618

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189618


namespace fraction_of_area_l189_189368

noncomputable section

open Real

-- Definitions of points A, B, C, X, Y, and Z with their given coordinates
def A := (2, 0) : ℝ × ℝ
def B := (8, 12) : ℝ × ℝ
def C := (14, 0) : ℝ × ℝ

def X := (6, 0) : ℝ × ℝ
def Y := (8, 4) : ℝ × ℝ
def Z := (10, 0) : ℝ × ℝ

-- Definition of the area of a triangle given vertices
def area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := 
  abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₂.2 - p₁.2) * (p₃.1 - p₁.1)) / 2

-- Areas of triangles ABC and XYZ
def area_ABC := area A B C
def area_XYZ := area X Y Z

-- The Lean statement
theorem fraction_of_area : (area_XYZ / area_ABC) = 1 / 9 := by
  sorry

end fraction_of_area_l189_189368


namespace problem_statement_l189_189936

theorem problem_statement (a b c : ℤ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 3) 
  (h3 : |c| = 6) 
  (h4 : |a + b| = - (a + b)) 
  (h5 : |a + c| = a + c) : 
  a - b + c = -2 ∨ a - b + c = 4 :=
sorry

end problem_statement_l189_189936


namespace find_ellipse_parameters_l189_189824

noncomputable def ellipse_centers_and_axes (F1 F2 : ℝ × ℝ) (d : ℝ) (tangent_slope : ℝ) :=
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  let a := d / 2
  let c := (Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  (h, k, a, b)

theorem find_ellipse_parameters :
  let F1 := (-1, 1)
  let F2 := (5, 1)
  let d := 10
  let tangent_at_x_axis_slope := 1
  let (h, k, a, b) := ellipse_centers_and_axes F1 F2 d tangent_at_x_axis_slope
  h + k + a + b = 12 :=
by
  sorry

end find_ellipse_parameters_l189_189824


namespace kekai_garage_sale_l189_189819

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end kekai_garage_sale_l189_189819


namespace range_of_a_l189_189786

theorem range_of_a (a : ℝ) : 
  (∀ x, (x > 2 ∨ x < -1) → ¬(x^2 + 4 * x + a < 0)) → a ≥ 3 :=
by
  sorry

end range_of_a_l189_189786


namespace no_sol_for_frac_eq_l189_189976

theorem no_sol_for_frac_eq (x y : ℕ) (h : x > 1) : ¬ (y^5 + 1 = (x^7 - 1) / (x - 1)) :=
sorry

end no_sol_for_frac_eq_l189_189976


namespace range_of_m_l189_189181

noncomputable def quadratic_inequality_solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem range_of_m :
  { m : ℝ | quadratic_inequality_solution_set_is_R m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
by
  sorry

end range_of_m_l189_189181


namespace cookies_yesterday_l189_189794

theorem cookies_yesterday (cookies_today : ℕ) (difference : ℕ)
  (h1 : cookies_today = 140)
  (h2 : difference = 30) :
  cookies_today - difference = 110 :=
by
  sorry

end cookies_yesterday_l189_189794


namespace island_of_misfortune_l189_189727

def statement (n : ℕ) (knight : ℕ → Prop) (liar : ℕ → Prop) : Prop :=
  ∀ k : ℕ, k < n → (
    if k = 0 then ∀ m : ℕ, (m % 2 = 1) ↔ liar m
    else if k = 1 then ∀ m : ℕ, (m % 3 = 1) ↔ liar m
    else ∀ m : ℕ, (m % (k + 1) = 1) ↔ liar m
  )

theorem island_of_misfortune :
  ∃ n : ℕ, n >= 2 ∧ statement n knight liar
:= sorry

end island_of_misfortune_l189_189727


namespace person_walks_distance_l189_189869

theorem person_walks_distance {D t : ℝ} (h1 : 5 * t = D) (h2 : 10 * t = D + 20) : D = 20 :=
by
  sorry

end person_walks_distance_l189_189869


namespace red_grapes_more_than_three_times_green_l189_189805

-- Definitions from conditions
variables (G R B : ℕ)
def condition1 := R = 3 * G + (R - 3 * G)
def condition2 := B = G - 5
def condition3 := R + G + B = 102
def condition4 := R = 67

-- The proof problem
theorem red_grapes_more_than_three_times_green : (R = 67) ∧ (R + G + (G - 5) = 102) ∧ (R = 3 * G + (R - 3 * G)) → R - 3 * G = 7 :=
by sorry

end red_grapes_more_than_three_times_green_l189_189805


namespace abs_m_minus_1_greater_eq_abs_m_minus_1_l189_189430

theorem abs_m_minus_1_greater_eq_abs_m_minus_1 (m : ℝ) : |m - 1| ≥ |m| - 1 := 
sorry

end abs_m_minus_1_greater_eq_abs_m_minus_1_l189_189430


namespace tomatoes_left_l189_189391

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l189_189391


namespace solution_set_of_inequality_l189_189560

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l189_189560


namespace max_gcd_lcm_condition_l189_189069

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_condition_l189_189069


namespace contingency_fund_correct_l189_189937

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l189_189937


namespace arithmetic_series_sum_l189_189136

def first_term (k : ℕ) : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def number_of_terms (k : ℕ) : ℕ := 2 * k + 3
def nth_term (k n : ℕ) : ℕ := (first_term k) + (n - 1) * common_difference
def sum_of_terms (k : ℕ) : ℕ :=
  let n := number_of_terms k
  let a := first_term k
  let l := nth_term k n
  n * (a + l) / 2

theorem arithmetic_series_sum (k : ℕ) : sum_of_terms k = 2 * k^3 + 7 * k^2 + 10 * k + 6 :=
sorry

end arithmetic_series_sum_l189_189136


namespace cylindrical_to_rectangular_l189_189421

theorem cylindrical_to_rectangular (r θ z : ℝ) (h₁ : r = 10) (h₂ : θ = Real.pi / 6) (h₃ : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5 * Real.sqrt 3, 5, 2) := 
by
  sorry

end cylindrical_to_rectangular_l189_189421


namespace problem_statement_l189_189011

noncomputable def calculateValue (n : ℕ) : ℕ :=
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n)

theorem problem_statement : calculateValue 10 = 466 := by
  sorry

end problem_statement_l189_189011


namespace sum_three_digit_integers_from_200_to_900_l189_189100

theorem sum_three_digit_integers_from_200_to_900 : 
  let a := 200
  let l := 900
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 385550 := by
    let a := 200
    let l := 900
    let d := 1
    let n := (l - a) / d + 1
    let S := n / 2 * (a + l)
    sorry

end sum_three_digit_integers_from_200_to_900_l189_189100


namespace two_pow_gt_twice_n_plus_one_l189_189097

theorem two_pow_gt_twice_n_plus_one (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
sorry

end two_pow_gt_twice_n_plus_one_l189_189097


namespace negation_proposition_equivalence_l189_189221

theorem negation_proposition_equivalence :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_proposition_equivalence_l189_189221


namespace ratio_of_boys_to_girls_l189_189414

theorem ratio_of_boys_to_girls (boys : ℕ) (students : ℕ) (h1 : boys = 42) (h2 : students = 48) : (boys : ℚ) / (students - boys : ℚ) = 7 / 1 := 
by
  sorry

end ratio_of_boys_to_girls_l189_189414


namespace younger_son_age_after_30_years_l189_189687

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end younger_son_age_after_30_years_l189_189687


namespace distribute_coins_l189_189048

/-- The number of ways to distribute 25 identical coins among 4 schoolchildren -/
theorem distribute_coins :
  (Nat.choose 28 3) = 3276 :=
by
  sorry

end distribute_coins_l189_189048


namespace scientific_notation_123000_l189_189649

theorem scientific_notation_123000 : (123000 : ℝ) = 1.23 * 10^5 := by
  sorry

end scientific_notation_123000_l189_189649


namespace function_passes_through_one_one_l189_189022

noncomputable def f (a x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_through_one_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 := 
by
  sorry

end function_passes_through_one_one_l189_189022


namespace max_perimeter_isosceles_triangle_l189_189365

/-- Out of all triangles with the same base and the same angle at the vertex, 
    the triangle with the largest perimeter is isosceles -/
theorem max_perimeter_isosceles_triangle {α β γ : ℝ} (b : ℝ) (B : ℝ) (A C : ℝ) 
  (hB : 0 < B ∧ B < π) (hβ : α + C = B) (h1 : A = β) (h2 : γ = β) :
  α = γ := sorry

end max_perimeter_isosceles_triangle_l189_189365


namespace monotonically_increasing_sequence_l189_189883

theorem monotonically_increasing_sequence (k : ℝ) : (∀ n : ℕ+, n^2 + k * n < (n + 1)^2 + k * (n + 1)) ↔ k > -3 := by
  sorry

end monotonically_increasing_sequence_l189_189883


namespace log_base_243_l189_189009

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_base_243 : log_base 3 243 = 5 := by
  -- this is the statement, proof is omitted
  sorry

end log_base_243_l189_189009


namespace find_x_l189_189537

-- Define the conditions as hypotheses
def problem_statement (x : ℤ) : Prop :=
  (3 * x > 30) ∧ (x ≥ 10) ∧ (x > 5) ∧ 
  (x = 9)

-- Define the theorem statement
theorem find_x : ∃ x : ℤ, problem_statement x :=
by
  -- Sorry to skip proof as instructed
  sorry

end find_x_l189_189537


namespace cube_sum_is_integer_l189_189673

theorem cube_sum_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^3 + 1/a^3 = m :=
sorry

end cube_sum_is_integer_l189_189673


namespace total_cost_proof_l189_189081

-- Define the cost of items
def cost_of_1kg_of_mango (M : ℚ) : Prop := sorry
def cost_of_1kg_of_rice (R : ℚ) : Prop := sorry
def cost_of_1kg_of_flour (F : ℚ) : Prop := F = 23

-- Condition 1: cost of some kg of mangos is equal to the cost of 24 kg of rice
def condition1 (M R : ℚ) (x : ℚ) : Prop := M * x = R * 24

-- Condition 2: cost of 6 kg of flour equals to the cost of 2 kg of rice
def condition2 (R : ℚ) : Prop := 23 * 6 = R * 2

-- Final proof problem
theorem total_cost_proof (M R F : ℚ) (x : ℚ) 
  (h1: condition1 M R x) 
  (h2: condition2 R) 
  (h3: cost_of_1kg_of_flour F) :
  4 * (69 * 24 / x) + 3 * R + 5 * 23 = 1978 :=
sorry

end total_cost_proof_l189_189081


namespace number_divisible_by_k_cube_l189_189802

theorem number_divisible_by_k_cube (k : ℕ) (h : k = 42) : ∃ n, (k^3) % n = 0 ∧ n = 74088 := by
  sorry

end number_divisible_by_k_cube_l189_189802


namespace pencils_to_sell_for_desired_profit_l189_189002

/-- Definitions based on the conditions provided in the problem. -/
def total_pencils : ℕ := 2000
def cost_per_pencil : ℝ := 0.20
def sell_price_per_pencil : ℝ := 0.40
def desired_profit : ℝ := 160
def total_cost : ℝ := total_pencils * cost_per_pencil

/-- The theorem considers all the conditions and asks to prove the number of pencils to sell -/
theorem pencils_to_sell_for_desired_profit : 
  (desired_profit + total_cost) / sell_price_per_pencil = 1400 :=
by 
  sorry

end pencils_to_sell_for_desired_profit_l189_189002


namespace minimum_soldiers_to_add_l189_189627

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189627


namespace compute_a_plus_b_l189_189480

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l189_189480


namespace proof_problem_l189_189961

open Nat

noncomputable def has_at_least_three_distinct_prime_divisors (n : ℕ) (a : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ : ℕ, p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ p₁ ∣ a ∧ p₂ ∣ a ∧ p₃ ∣ a

theorem proof_problem (p : ℕ) (hp : Prime p) (h : 2^(p - 1) ≡ 1 [MOD p^2]) (n : ℕ) (hn : n ∈ ℕ) :
  has_at_least_three_distinct_prime_divisors n ((p - 1) * (factorial p + 2^n)) :=
sorry

end proof_problem_l189_189961


namespace overall_average_tickets_sold_l189_189407

variable {M : ℕ} -- number of male members
variable {F : ℕ} -- number of female members
variable (male_to_female_ratio : M * 2 = F) -- 1:2 ratio
variable (average_female : ℕ) (average_male : ℕ) -- average tickets sold by female/male members
variable (total_tickets_female : F * average_female = 70 * F) -- Total tickets sold by female members
variable (total_tickets_male : M * average_male = 58 * M) -- Total tickets sold by male members

-- The overall average number of raffle tickets sold per member is 66.
theorem overall_average_tickets_sold 
  (h1 : 70 * F + 58 * M = 198 * M) -- total tickets sold
  (h2 : M + F = 3 * M) -- total number of members
  : (70 * F + 58 * M) / (M + F) = 66 := by
  sorry

end overall_average_tickets_sold_l189_189407


namespace overtime_pay_correct_l189_189274

theorem overtime_pay_correct
  (overlap_slow : ℝ := 69) -- Slow clock minute-hand overlap in minutes
  (overlap_normal : ℝ := 12 * 60 / 11) -- Normal clock minute-hand overlap in minutes
  (hours_worked : ℝ := 8) -- The normal working hours a worker believes working
  (hourly_wage : ℝ := 4) -- The normal hourly wage
  (overtime_rate : ℝ := 1.5) -- Overtime pay rate
  (expected_overtime_pay : ℝ := 2.60) -- The expected overtime pay
  
  : hours_worked * (overlap_slow / overlap_normal) * hourly_wage * (overtime_rate - 1) = expected_overtime_pay :=
by
  sorry

end overtime_pay_correct_l189_189274


namespace quadratic_cubic_inequalities_l189_189670

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x : ℝ) : ℝ := -x ^ 3 + 5 * x - 3

variable (x : ℝ)

theorem quadratic_cubic_inequalities (h : 0 < x) : 
  (f x ≥ 2 * x - 1) ∧ (g x ≤ 2 * x - 1) := 
sorry

end quadratic_cubic_inequalities_l189_189670


namespace girls_in_school_play_l189_189524

theorem girls_in_school_play (G : ℕ) (boys : ℕ) (total_parents : ℕ)
  (h1 : boys = 8) (h2 : total_parents = 28) (h3 : 2 * boys + 2 * G = total_parents) : 
  G = 6 :=
sorry

end girls_in_school_play_l189_189524


namespace find_divisor_l189_189046

theorem find_divisor (q r D : ℕ) (hq : q = 120) (hr : r = 333) (hD : 55053 = D * q + r) : D = 456 :=
by
  sorry

end find_divisor_l189_189046


namespace K_time_for_distance_l189_189241

theorem K_time_for_distance (s : ℝ) (hs : s > 0) :
  (let K_time := 45 / s
   let M_speed := s - 1 / 2
   let M_time := 45 / M_speed
   K_time = M_time - 3 / 4) -> K_time = 45 / s := 
by
  sorry

end K_time_for_distance_l189_189241


namespace sum_sequence_l189_189855

noncomputable def sum_first_n_minus_1_terms (n : ℕ) : ℕ :=
  (2^n - n - 1)

theorem sum_sequence (n : ℕ) : 
  sum_first_n_minus_1_terms n = (2^n - n - 1) :=
by
  sorry 

end sum_sequence_l189_189855


namespace solve_for_x_l189_189334

theorem solve_for_x (x : ℝ) (h : 3 - (1 / (2 - x)) = (1 / (2 - x))) : x = 4 / 3 := 
by {
  sorry
}

end solve_for_x_l189_189334


namespace remainder_x1001_mod_poly_l189_189903

noncomputable def remainder_poly_div (n k : ℕ) (f g : Polynomial ℚ) : Polynomial ℚ :=
  Polynomial.modByMonic f g

theorem remainder_x1001_mod_poly :
  remainder_poly_div 1001 3 (Polynomial.X ^ 1001) (Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X + 1) = Polynomial.X ^ 2 :=
by
  sorry

end remainder_x1001_mod_poly_l189_189903


namespace find_expression_value_l189_189492

variable (x y z : ℚ)
variable (h1 : x - y + 2 * z = 1)
variable (h2 : x + y + 4 * z = 3)

theorem find_expression_value : x + 2 * y + 5 * z = 4 := 
by {
  sorry
}

end find_expression_value_l189_189492


namespace fraction_of_areas_l189_189367

/-- Points A, B, C, X, Y, Z coordinates definitions --/
structure Point :=
(x : ℝ)
(y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

/-- Area of a triangle given base and height --/
def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Area of triangle ABC --/
def Area_ABC := area_triangle (C.x - A.x) B.y

/-- Area of triangle XYZ --/
def Area_XYZ := area_triangle (Z.x - X.x) Y.y

theorem fraction_of_areas : Area_XYZ / Area_ABC = 1 / 9 := by
  sorry

end fraction_of_areas_l189_189367


namespace polynomial_subtraction_simplify_l189_189076

open Polynomial

noncomputable def p : Polynomial ℚ := 3 * X^2 + 9 * X - 5
noncomputable def q : Polynomial ℚ := 2 * X^2 + 3 * X - 10
noncomputable def result : Polynomial ℚ := X^2 + 6 * X + 5

theorem polynomial_subtraction_simplify : 
  p - q = result :=
by
  sorry

end polynomial_subtraction_simplify_l189_189076


namespace at_least_two_equal_elements_l189_189953

open Function

theorem at_least_two_equal_elements :
  ∀ (k : Fin 10 → Fin 10),
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j) → False :=
by
  intros k h
  sorry

end at_least_two_equal_elements_l189_189953


namespace soil_bags_needed_l189_189747

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end soil_bags_needed_l189_189747


namespace cracked_to_broken_eggs_ratio_l189_189357

theorem cracked_to_broken_eggs_ratio (total_eggs : ℕ) (broken_eggs : ℕ) (P C : ℕ)
  (h1 : total_eggs = 24)
  (h2 : broken_eggs = 3)
  (h3 : P - C = 9)
  (h4 : P + C = 21) :
  (C : ℚ) / (broken_eggs : ℚ) = 2 :=
by
  sorry

end cracked_to_broken_eggs_ratio_l189_189357


namespace radish_patch_size_l189_189410

theorem radish_patch_size (R P : ℕ) (h1 : P = 2 * R) (h2 : P / 6 = 5) : R = 15 := by
  sorry

end radish_patch_size_l189_189410


namespace contingency_fund_allocation_l189_189942

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l189_189942


namespace point_in_fourth_quadrant_l189_189924

noncomputable def a : ℤ := 2

theorem point_in_fourth_quadrant (x y : ℤ) (h1 : x = a - 1) (h2 : y = a - 3) (h3 : x > 0) (h4 : y < 0) : a = 2 := by
  sorry

end point_in_fourth_quadrant_l189_189924


namespace problem_part1_problem_part2_l189_189012

-- Statement part (1)
theorem problem_part1 : ( (2 / 3) - (1 / 4) - (1 / 6) ) * 24 = 6 :=
sorry

-- Statement part (2)
theorem problem_part2 : (-2)^3 + (-9 + (-3)^2 * (1 / 3)) = -14 :=
sorry

end problem_part1_problem_part2_l189_189012


namespace measure_of_angle_C_l189_189131

variable (C D : ℕ)
variable (h1 : C + D = 180)
variable (h2 : C = 5 * D)

theorem measure_of_angle_C : C = 150 :=
by
  sorry

end measure_of_angle_C_l189_189131


namespace at_least_one_not_less_than_two_l189_189840

theorem at_least_one_not_less_than_two
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a >= 2 ∨ b >= 2 ∨ c >= 2 := 
sorry

end at_least_one_not_less_than_two_l189_189840


namespace calculate_p_p1_neg1_p_neg5_neg2_l189_189059

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then
    x + y
  else if x < 0 ∧ y < 0 then
    x - 2 * y
  else
    3 * x + y

theorem calculate_p_p1_neg1_p_neg5_neg2 :
  p (p 1 (-1)) (p (-5) (-2)) = 5 :=
by
  sorry

end calculate_p_p1_neg1_p_neg5_neg2_l189_189059


namespace range_of_a_l189_189165

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.exp x - 1) - Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, 0 < x0 ∧ f (g x0) a > f x0 a) ↔ 1 < a := sorry

end range_of_a_l189_189165


namespace elena_novel_pages_l189_189559

theorem elena_novel_pages
  (days_vacation : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_three_days : ℕ)
  (pages_last_day : ℕ)
  (h1 : days_vacation = 6)
  (h2 : pages_first_two_days = 2 * 42)
  (h3 : pages_next_three_days = 3 * 35)
  (h4 : pages_last_day = 15) :
  pages_first_two_days + pages_next_three_days + pages_last_day = 204 := by
  sorry

end elena_novel_pages_l189_189559


namespace find_r_l189_189653

theorem find_r 
  (r RB QC : ℝ)
  (angleA : ℝ)
  (h0 : RB = 6)
  (h1 : QC = 4)
  (h2 : angleA = 90) :
  (r + 6) ^ 2 + (r + 4) ^ 2 = 10 ^ 2 → r = 2 := 
by 
  sorry

end find_r_l189_189653


namespace ab_value_l189_189404

theorem ab_value (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 33) : a * b = 18 := 
by
  sorry

end ab_value_l189_189404


namespace amara_clothing_remaining_l189_189418

theorem amara_clothing_remaining :
  ∀ (initial donation_one donation_factor discard : ℕ),
    initial = 100 →
    donation_one = 5 →
    donation_factor = 3 →
    discard = 15 →
    let total_donated := donation_one + (donation_factor * donation_one) in
    let remaining_after_donation := initial - total_donated in
    let final_remaining := remaining_after_donation - discard in
    final_remaining = 65 := 
by
  sorry

end amara_clothing_remaining_l189_189418


namespace cos_4_arccos_l189_189899

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end cos_4_arccos_l189_189899


namespace solve_for_x_l189_189557

theorem solve_for_x (x : ℝ) (hx : x^(1/10) * (x^(3/2))^(1/10) = 3) : x = 9 :=
sorry

end solve_for_x_l189_189557


namespace quadratic_roots_integer_sum_eq_198_l189_189451

theorem quadratic_roots_integer_sum_eq_198 (x p q x1 x2 : ℤ) 
  (h_eqn : x^2 + p * x + q = 0)
  (h_roots : (x - x1) * (x - x2) = 0)
  (h_pq_sum : p + q = 198) :
  (x1 = 2 ∧ x2 = 200) ∨ (x1 = 0 ∧ x2 = -198) :=
sorry

end quadratic_roots_integer_sum_eq_198_l189_189451


namespace shared_candy_equally_l189_189593

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l189_189593


namespace surface_area_of_T_is_630_l189_189351

noncomputable def s : ℕ := 582
noncomputable def t : ℕ := 42
noncomputable def u : ℕ := 6

theorem surface_area_of_T_is_630 : s + t + u = 630 :=
by
  sorry

end surface_area_of_T_is_630_l189_189351


namespace fraction_value_l189_189312

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l189_189312


namespace number_of_roosters_l189_189227

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l189_189227


namespace total_yield_l189_189656

theorem total_yield (x y z : ℝ)
  (h1 : 0.4 * z + 0.2 * x = 1)
  (h2 : 0.1 * y - 0.1 * z = -0.5)
  (h3 : 0.1 * x + 0.2 * y = 4) :
  x + y + z = 15 :=
sorry

end total_yield_l189_189656


namespace opposite_sides_line_range_a_l189_189584

theorem opposite_sides_line_range_a (a : ℝ) :
  (3 * 2 - 2 * 1 + a) * (3 * -1 - 2 * 3 + a) < 0 → -4 < a ∧ a < 9 := by
  sorry

end opposite_sides_line_range_a_l189_189584


namespace soldiers_to_add_l189_189634

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189634


namespace gcd_of_two_powers_l189_189712

-- Define the expressions
def two_pow_1015_minus_1 : ℤ := 2^1015 - 1
def two_pow_1024_minus_1 : ℤ := 2^1024 - 1

-- Define the gcd function and the target value
noncomputable def gcd_expr : ℤ := Int.gcd (2^1015 - 1) (2^1024 - 1)
def target : ℤ := 511

-- The statement we want to prove
theorem gcd_of_two_powers : gcd_expr = target := by 
  sorry

end gcd_of_two_powers_l189_189712


namespace product_calculation_l189_189553

theorem product_calculation :
  12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end product_calculation_l189_189553


namespace pollen_allergy_expected_count_l189_189834

theorem pollen_allergy_expected_count : 
  ∀ (sample_size : ℕ) (pollen_allergy_ratio : ℚ), 
  pollen_allergy_ratio = 1/4 ∧ sample_size = 400 → sample_size * pollen_allergy_ratio = 100 :=
  by 
    intros
    sorry

end pollen_allergy_expected_count_l189_189834


namespace johnson_class_more_students_l189_189358

theorem johnson_class_more_students
  (finley_class_students : ℕ)
  (johnson_class_students : ℕ)
  (h_finley : finley_class_students = 24)
  (h_johnson : johnson_class_students = 22) :
  johnson_class_students - finley_class_students / 2 = 10 :=
  sorry

end johnson_class_more_students_l189_189358


namespace acceptable_N_value_l189_189771

noncomputable def discriminant_game (N : ℕ) : Prop :=
  ∀ (a : ℕ -> ℕ) (n : ℕ), ∃ (β : ℕ), ∀ s₁ s₂ : ℕ → ℕ, 
  (∑ k in fin_range n + 1, a k * β ^ k) = (∑ k in fin_range n + 1, s₁ k * β ^ k) → a = s₁ 

theorem acceptable_N_value : discriminant_game N → N = 1 :=
by
  sorry

end acceptable_N_value_l189_189771


namespace total_drink_volume_l189_189531

theorem total_drink_volume (oj wj gj : ℕ) (hoj : oj = 25) (hwj : wj = 40) (hgj : gj = 70) : (gj * 100) / (100 - oj - wj) = 200 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_drink_volume_l189_189531


namespace bret_total_spend_l189_189276

/-- Bret and his team are working late along with another team of 4 co-workers.
He decides to order dinner for everyone. -/

def team_A : ℕ := 4 -- Bret’s team
def team_B : ℕ := 4 -- Other team

def main_meal_cost : ℕ := 12
def team_A_appetizers_cost : ℕ := 2 * 6  -- Two appetizers at $6 each
def team_B_appetizers_cost : ℕ := 3 * 8  -- Three appetizers at $8 each
def sharing_plates_cost : ℕ := 4 * 10    -- Four sharing plates at $10 each

def tip_percentage : ℝ := 0.20           -- Tip is 20%
def rush_order_fee : ℕ := 5              -- Rush order fee is $5
def sales_tax : ℝ := 0.07                -- Local sales tax is 7%

def total_cost_without_tip_and_tax : ℕ :=
  team_A * main_meal_cost + team_B * main_meal_cost + team_A_appetizers_cost +
  team_B_appetizers_cost + sharing_plates_cost

def total_cost_with_tip : ℝ :=
  total_cost_without_tip_and_tax + 
  (tip_percentage * total_cost_without_tip_and_tax)

def total_cost_before_tax : ℝ :=
  total_cost_with_tip + rush_order_fee

def final_total_cost : ℝ :=
  total_cost_before_tax + (sales_tax * total_cost_with_tip)


theorem bret_total_spend : final_total_cost = 225.85 := by
  sorry

end bret_total_spend_l189_189276


namespace gcd_n_cube_plus_27_n_plus_3_l189_189147

theorem gcd_n_cube_plus_27_n_plus_3 (n : ℕ) (h : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
sorry

end gcd_n_cube_plus_27_n_plus_3_l189_189147


namespace solution_for_a_l189_189174

theorem solution_for_a (x : ℝ) (a : ℝ) (h : 2 * x - a = 0) (hx : x = 1) : a = 2 := by
  rw [hx] at h
  linarith


end solution_for_a_l189_189174


namespace tangent_line_eq_at_0_max_min_values_l189_189927

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_eq_at_0 : ∀ x : ℝ, x = 0 → f x = 1 :=
by
  sorry

theorem max_min_values : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 0 ≥ f x) ∧ (f (Real.pi / 2) = -Real.pi / 2) :=
by
  sorry

end tangent_line_eq_at_0_max_min_values_l189_189927


namespace roger_toys_l189_189494

theorem roger_toys (initial_money spent_money toy_cost remaining_money toys : ℕ) 
  (h1 : initial_money = 63) 
  (h2 : spent_money = 48) 
  (h3 : toy_cost = 3) 
  (h4 : remaining_money = initial_money - spent_money) 
  (h5 : toys = remaining_money / toy_cost) : 
  toys = 5 := 
by 
  sorry

end roger_toys_l189_189494


namespace train_cross_time_l189_189721

-- Definitions from the conditions
def length_of_train : ℤ := 600
def speed_of_man_kmh : ℤ := 2
def speed_of_train_kmh : ℤ := 56

-- Conversion factors and speed conversion
def kmh_to_mph_factor : ℤ := 1000 / 3600 -- 1 km/hr = 0.27778 m/s approximately

def speed_of_man_ms : ℤ := speed_of_man_kmh * kmh_to_mph_factor -- Convert speed of man to m/s
def speed_of_train_ms : ℤ := speed_of_train_kmh * kmh_to_mph_factor -- Convert speed of train to m/s

-- Calculating relative speed
def relative_speed_ms : ℤ := speed_of_train_ms - speed_of_man_ms

-- Calculating the time taken to cross
def time_to_cross : ℤ := length_of_train / relative_speed_ms 

-- The theorem to prove
theorem train_cross_time : time_to_cross = 40 := 
by sorry

end train_cross_time_l189_189721


namespace min_soldiers_needed_l189_189611

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189611


namespace find_smallest_number_l189_189995

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l189_189995


namespace eval_fraction_product_l189_189225

theorem eval_fraction_product :
  ((1 + (1 / 3)) * (1 + (1 / 4)) = (5 / 3)) :=
by
  sorry

end eval_fraction_product_l189_189225


namespace smallest_largest_sum_l189_189665

theorem smallest_largest_sum (a b c : ℝ) (m M : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : m = (1/3))
  (h4 : M = 1) :
  (m + M) = 4 / 3 := by
sorry

end smallest_largest_sum_l189_189665


namespace total_daisies_l189_189346

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l189_189346


namespace triangle_inequality_l189_189327

theorem triangle_inequality (a b c R r : ℝ) 
  (habc : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area1 : a * b * c = 4 * R * S)
  (h_area2 : S = r * (a + b + c) / 2) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := 
sorry

end triangle_inequality_l189_189327


namespace find_number_l189_189098

theorem find_number
  (P : ℝ) (R : ℝ) (hP : P = 0.0002) (hR : R = 2.4712) :
  (12356 * P = R) := by
  sorry

end find_number_l189_189098


namespace parallelogram_base_length_l189_189405

theorem parallelogram_base_length (b : ℝ) (A : ℝ) (h : ℝ)
  (H1 : A = 288) 
  (H2 : h = 2 * b) 
  (H3 : A = b * h) : 
  b = 12 := 
by 
  sorry

end parallelogram_base_length_l189_189405


namespace roots_polynomial_l189_189666

theorem roots_polynomial (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + c * a = 19) (h3 : a * b * c = 8) : 
  (1 + a) * (1 + b) * (1 + c) = 46 :=
by
  sorry

end roots_polynomial_l189_189666


namespace sufficient_not_necessary_condition_l189_189317

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h1 : c < b) (h2 : b < a) :
  (ac < 0 → ab > ac) ∧ (ab > ac → ac < 0) → false :=
sorry

end sufficient_not_necessary_condition_l189_189317


namespace max_observing_relations_lemma_l189_189700

/-- There are 24 robots on a plane, each with a 70-degree field of view. -/
def robots : ℕ := 24

/-- Definition of field of view for each robot. -/
def field_of_view : ℝ := 70

/-- Maximum number of observing relations. Observing is a one-sided relation. -/
def max_observing_relations := 468

/-- Theorem: The maximum number of observing relations among 24 robots,
each with a 70-degree field of view, is 468. -/
theorem max_observing_relations_lemma : max_observing_relations = 468 :=
by
  sorry

end max_observing_relations_lemma_l189_189700


namespace lcm_36_100_eq_900_l189_189295

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l189_189295


namespace triangle_area_multiplication_factor_l189_189337

theorem triangle_area_multiplication_factor
  (a b : ℝ) (θ : ℝ) :
  let A := (a * b * Real.sin θ) / 2 in
  let A' := (3 * a * b * Real.sin (θ + 15 * Real.pi / 180)) / 2 in
  (A' / A) = 3 * (Real.sin (θ + 15 * Real.pi / 180) / Real.sin θ) :=
by
  sorry

end triangle_area_multiplication_factor_l189_189337


namespace arctan_sum_l189_189551

theorem arctan_sum : 
  let x := (3 : ℝ) / 7
  let y := 7 / 3
  x * y = 1 → (Real.arctan x + Real.arctan y = Real.pi / 2) :=
by
  intros x y h
  -- Proof goes here
  sorry

end arctan_sum_l189_189551


namespace soldiers_to_add_l189_189635

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189635


namespace people_on_bus_now_l189_189459

variable (x : ℕ)

def original_people_on_bus : ℕ := 38
def people_got_on_bus (x : ℕ) : ℕ := x
def people_left_bus (x : ℕ) : ℕ := x + 9

theorem people_on_bus_now (x : ℕ) : original_people_on_bus - people_left_bus x + people_got_on_bus x = 29 := 
by
  sorry

end people_on_bus_now_l189_189459


namespace value_of_g_at_x_minus_5_l189_189797

-- Definition of the function g
def g (x : ℝ) : ℝ := -3

-- The theorem we need to prove
theorem value_of_g_at_x_minus_5 (x : ℝ) : g (x - 5) = -3 := by
  sorry

end value_of_g_at_x_minus_5_l189_189797


namespace width_of_vessel_is_5_l189_189737

open Real

noncomputable def width_of_vessel : ℝ :=
  let edge := 5
  let rise := 2.5
  let base_length := 10
  let volume_cube := edge ^ 3
  let volume_displaced := volume_cube
  let width := volume_displaced / (base_length * rise)
  width

theorem width_of_vessel_is_5 :
  width_of_vessel = 5 := by
    sorry

end width_of_vessel_is_5_l189_189737


namespace lcm_36_100_is_900_l189_189301

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l189_189301


namespace tetrahedron_edge_length_l189_189864

theorem tetrahedron_edge_length (a : ℝ) (V : ℝ) 
  (h₀ : V = 0.11785113019775793) 
  (h₁ : V = (Real.sqrt 2 / 12) * a^3) : a = 1 := by
  sorry

end tetrahedron_edge_length_l189_189864


namespace total_pizzas_served_l189_189001

def lunch_pizzas : ℚ := 12.5
def dinner_pizzas : ℚ := 8.25

theorem total_pizzas_served : lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end total_pizzas_served_l189_189001


namespace arithmetic_expression_equality_l189_189010

theorem arithmetic_expression_equality :
  15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := 
by 
  sorry

end arithmetic_expression_equality_l189_189010


namespace triangle_probability_l189_189832

noncomputable def stick_lengths : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def valid_triangles (lst: List ℕ) : ℕ :=
  -- Define a function that counts valid triangle combinations
  -- where the sum of the two smaller sides is greater than the third side.
  (lst.combination 3).count (λ triplet,
    let sorted_triplet := tripletSorted triplet in
    sorted_triplet.head! + sorted_triplet.tail.head! > sorted_triplet.last!)

noncomputable def total_combinations (lst: List ℕ) : ℕ :=
  lst.combination 3 |>.length

theorem triangle_probability :
  let valid_count := valid_triangles stick_lengths in
  let total_count := total_combinations stick_lengths in
  (valid_count, total_count) = (25, 84) ∧ (valid_count : ℚ) / total_count = 25 / 84 :=
by
  sorry

end triangle_probability_l189_189832


namespace length_of_larger_sheet_l189_189377

theorem length_of_larger_sheet : 
  ∃ L : ℝ, 2 * (L * 11) = 2 * (5.5 * 11) + 100 ∧ L = 10 :=
by
  sorry

end length_of_larger_sheet_l189_189377


namespace gcf_180_240_300_l189_189713

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end gcf_180_240_300_l189_189713


namespace mary_pays_fifteen_l189_189203

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_per_5_fruits : ℕ := 1

def apples_bought : ℕ := 5
def oranges_bought : ℕ := 3
def bananas_bought : ℕ := 2

def total_cost_before_discount : ℕ :=
  apples_bought * apple_cost +
  oranges_bought * orange_cost +
  bananas_bought * banana_cost

def total_fruits : ℕ :=
  apples_bought + oranges_bought + bananas_bought

def total_discount : ℕ :=
  (total_fruits / 5) * discount_per_5_fruits

def final_amount_to_pay : ℕ :=
  total_cost_before_discount - total_discount

theorem mary_pays_fifteen : final_amount_to_pay = 15 := by
  sorry

end mary_pays_fifteen_l189_189203


namespace directrix_of_parabola_l189_189763

-- Define the equation of the parabola and what we need to prove
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 6

-- Theorem stating the directrix of the given parabola
theorem directrix_of_parabola :
  ∀ x : ℝ, y = parabola_equation x → y = 47 / 8 := 
by
  sorry

end directrix_of_parabola_l189_189763


namespace compare_slopes_l189_189021

noncomputable def f (p q r x : ℝ) := x^3 + p * x^2 + q * x + r

noncomputable def s (p q x : ℝ) := 3 * x^2 + 2 * p * x + q

theorem compare_slopes (p q r a b c : ℝ) (hb : b ≠ 0) (ha : a ≠ c) 
  (hfa : f p q r a = 0) (hfc : f p q r c = 0) : a > c → s p q a > s p q c := 
by
  sorry

end compare_slopes_l189_189021


namespace quadratic_passes_through_neg3_n_l189_189844

-- Definition of the quadratic function with given conditions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
axiom max_at_neg2 : ∀ x, quadratic a b c x ≤ 8
axiom value_at_neg2 : quadratic a b c (-2) = 8
axiom passes_through_1_4 : quadratic a b c 1 = 4

-- Statement to prove
theorem quadratic_passes_through_neg3_n : quadratic a b c (-3) = 68 / 9 :=
sorry

end quadratic_passes_through_neg3_n_l189_189844


namespace coordinates_of_A_l189_189809

-- Definition of the distance function for any point (x, y)
def distance_to_x_axis (x y : ℝ) : ℝ := abs y
def distance_to_y_axis (x y : ℝ) : ℝ := abs x

-- Point A's coordinates
def point_is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The main theorem to prove
theorem coordinates_of_A :
  ∃ (x y : ℝ), 
  point_is_in_fourth_quadrant x y ∧ 
  distance_to_x_axis x y = 3 ∧ 
  distance_to_y_axis x y = 6 ∧ 
  (x, y) = (6, -3) :=
by 
  sorry

end coordinates_of_A_l189_189809


namespace MN_eq_l189_189555

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}
def operation (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

theorem MN_eq : operation M N = {1, 4} :=
sorry

end MN_eq_l189_189555


namespace find_positive_integer_M_l189_189232

theorem find_positive_integer_M (M : ℕ) (h : 36^2 * 81^2 = 18^2 * M^2) : M = 162 := by
  sorry

end find_positive_integer_M_l189_189232


namespace tan_of_geometric_sequence_is_negative_sqrt_3_l189_189163

variable {a : ℕ → ℝ} 

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q, m + n = p + q → a m * a n = a p * a q

theorem tan_of_geometric_sequence_is_negative_sqrt_3 
  (hgeo : is_geometric_sequence a)
  (hcond : a 2 * a 3 * a 4 = - a 7 ^ 2 ∧ a 7 ^ 2 = 64) :
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = - Real.sqrt 3 :=
sorry

end tan_of_geometric_sequence_is_negative_sqrt_3_l189_189163


namespace leastCookies_l189_189830

theorem leastCookies (b : ℕ) :
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) →
  b = 179 :=
by
  sorry

end leastCookies_l189_189830


namespace acuteAnglesSum_l189_189778

theorem acuteAnglesSum (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2) (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end acuteAnglesSum_l189_189778


namespace age_of_sisters_l189_189094

theorem age_of_sisters (a b : ℕ) (h1 : 10 * a - 9 * b = 89) 
  (h2 : 10 = 10) : a = 17 ∧ b = 9 :=
by sorry

end age_of_sisters_l189_189094


namespace vacant_seats_l189_189045

open Nat

-- Define the conditions as Lean definitions
def num_tables : Nat := 5
def seats_per_table : Nat := 8
def occupied_tables : Nat := 2
def people_per_occupied_table : Nat := 3
def unusable_tables : Nat := 1

-- Calculate usable tables
def usable_tables : Nat := num_tables - unusable_tables

-- Calculate total occupied people
def total_occupied_people : Nat := occupied_tables * people_per_occupied_table

-- Calculate total seats for occupied tables
def total_seats_occupied_tables : Nat := occupied_tables * seats_per_table

-- Calculate vacant seats in occupied tables
def vacant_seats_occupied_tables : Nat := total_seats_occupied_tables - total_occupied_people

-- Calculate completely unoccupied tables
def unoccupied_tables : Nat := usable_tables - occupied_tables

-- Calculate total seats for unoccupied tables
def total_seats_unoccupied_tables : Nat := unoccupied_tables * seats_per_table

-- Calculate total vacant seats
def total_vacant_seats : Nat := vacant_seats_occupied_tables + total_seats_unoccupied_tables

-- Theorem statement to prove
theorem vacant_seats : total_vacant_seats = 26 := by
  sorry

end vacant_seats_l189_189045


namespace minimum_soldiers_to_add_l189_189640

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189640


namespace brinley_animal_count_l189_189895

def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 12 * leopards
def cheetahs : ℕ := snakes / 3  -- rounding down implicitly considered
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem brinley_animal_count : total_animals = 673 :=
by
  -- Mathematical proof would go here.
  sorry

end brinley_animal_count_l189_189895


namespace moles_C2H6_for_HCl_l189_189307

theorem moles_C2H6_for_HCl 
  (form_HCl : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : ℕ) : 
  (6 * (reaction * moles_Cl2)) = form_HCl * (6 * reaction) :=
by
  -- The necessary proof steps will go here
  sorry

end moles_C2H6_for_HCl_l189_189307


namespace inverse_proportion_inequality_l189_189921

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l189_189921


namespace initial_amounts_l189_189703

theorem initial_amounts (x y z : ℕ) (h1 : x + y + z = 24)
  (h2 : z = 24 - x - y)
  (h3 : x - (y + z) = 8)
  (h4 : y - (x + z) = 12) :
  x = 13 ∧ y = 7 ∧ z = 4 :=
by
  sorry

end initial_amounts_l189_189703


namespace sarah_jamie_julien_ratio_l189_189662

theorem sarah_jamie_julien_ratio (S J : ℕ) (R : ℝ) :
  -- Conditions
  (J = S + 20) ∧
  (S = R * 50) ∧
  (7 * (J + S + 50) = 1890) ∧
  -- Prove the ratio
  R = 2 := by
  sorry

end sarah_jamie_julien_ratio_l189_189662


namespace nuts_to_raisins_ratio_l189_189754

/-- 
Given that Chris mixed 3 pounds of raisins with 4 pounds of nuts 
and the total cost of the raisins was 0.15789473684210525 of the total cost of the mixture, 
prove that the ratio of the cost of a pound of nuts to the cost of a pound of raisins is 4:1. 
-/
theorem nuts_to_raisins_ratio (R N : ℝ)
    (h1 : 3 * R = 0.15789473684210525 * (3 * R + 4 * N)) :
    N / R = 4 :=
sorry  -- proof skipped

end nuts_to_raisins_ratio_l189_189754


namespace min_value_a_b_c_l189_189158

theorem min_value_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : 9 * a + 4 * b = a * b * c) :
  a + b + c = 10 := sorry

end min_value_a_b_c_l189_189158


namespace problem_solution_l189_189190

-- Define the parametric equations for C1 and C2
def parametric_C1 (t : ℝ) : ℝ × ℝ := (1 - (real.sqrt 2) / 2 * t, 3 + (real.sqrt 2) / 2 * t)
def parametric_C2 (φ : ℝ) : ℝ × ℝ := (1 + real.cos φ, real.sin φ)

-- Define the polar equations
def polar_equation_C1 (ρ θ : ℝ) : Prop := ρ * (real.cos θ + real.sin θ) = 4
def polar_equation_C2 (ρ θ : ℝ) : Prop := ρ = 2 * real.cos θ

-- Define the ray intersection conditions and the resulting maximum value
def max_intersection_value (α : ℝ) : Prop :=
  -real.pi / 4 < α ∧ α < real.pi / 2 →
  (∃ ρ1 ρ2 : ℝ, ρ1 = 4 / (real.cos α + real.sin α) ∧ ρ2 = 2 * real.cos α ∧
    ρ2 / ρ1 = (real.sqrt 2 + 1) / 4)

-- Final theorem statement
theorem problem_solution :
  (∀ t : ℝ, ∃ ρ θ : ℝ, polar_equation_C1 ρ θ) →
  (∀ φ : ℝ, ∃ ρ θ : ℝ, polar_equation_C2 ρ θ) →
  (∀ α : ℝ, max_intersection_value α) :=
by sorry

end problem_solution_l189_189190


namespace problem_1_problem_2_l189_189782

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem problem_1 (x : ℝ) : f x + x^2 - 4 > 0 ↔ (x > 2 ∨ x < -1) := sorry

theorem problem_2 {m : ℝ} (h : m > 3) : ∃ x : ℝ, f x < g x m := sorry

end problem_1_problem_2_l189_189782


namespace number_of_diagonals_25_sides_l189_189887

theorem number_of_diagonals_25_sides (n : ℕ) (h : n = 25) : 
    (n * (n - 3)) / 2 = 275 := by
  sorry

end number_of_diagonals_25_sides_l189_189887


namespace gina_minutes_of_netflix_l189_189433

-- Define the conditions given in the problem
def gina_chooses_three_times_as_often (g s : ℕ) : Prop :=
  g = 3 * s

def total_shows_watched (g s : ℕ) : Prop :=
  g + s = 24

def duration_per_show : ℕ := 50

-- The theorem that encapsulates the problem statement and the correct answer
theorem gina_minutes_of_netflix (g s : ℕ) (h1 : gina_chooses_three_times_as_often g s) 
    (h2 : total_shows_watched g s) :
    g * duration_per_show = 900 :=
by
  sorry

end gina_minutes_of_netflix_l189_189433


namespace candle_problem_l189_189705

theorem candle_problem :
  ∃ x : ℚ,
    (1 - x / 6 = 3 * (1 - x / 5)) ∧
    x = 60 / 13 :=
by
  -- let initial_height_first_candle be 1
  -- let rate_first_burns be 1 / 6
  -- let initial_height_second_candle be 1
  -- let rate_second_burns be 1 / 5
  -- We want to prove:
  -- 1 - x / 6 = 3 * (1 - x / 5) ∧ x = 60 / 13
  sorry

end candle_problem_l189_189705


namespace no_sol_for_eq_xn_minus_yn_eq_2k_l189_189960

theorem no_sol_for_eq_xn_minus_yn_eq_2k (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_n : n > 2) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^n - y^n = 2^k := 
sorry

end no_sol_for_eq_xn_minus_yn_eq_2k_l189_189960


namespace disqualified_team_participants_l189_189654

theorem disqualified_team_participants
  (initial_teams : ℕ) (initial_avg : ℕ) (final_teams : ℕ) (final_avg : ℕ)
  (total_initial : ℕ) (total_final : ℕ) :
  initial_teams = 9 →
  initial_avg = 7 →
  final_teams = 8 →
  final_avg = 6 →
  total_initial = initial_teams * initial_avg →
  total_final = final_teams * final_avg →
  total_initial - total_final = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end disqualified_team_participants_l189_189654


namespace trigonometric_identity_proof_l189_189060

noncomputable def a : ℝ := -35 / 6 * Real.pi

theorem trigonometric_identity_proof :
  (2 * Real.sin (Real.pi + a) * Real.cos (Real.pi - a) - Real.cos (Real.pi + a)) / 
  (1 + Real.sin a ^ 2 + Real.sin (Real.pi - a) - Real.cos (Real.pi + a) ^ 2) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_proof_l189_189060


namespace circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l189_189321

-- Problem (1)
theorem circle_a_lt_8 (x y a : ℝ) (h : x^2 + y^2 - 4*x - 4*y + a = 0) : 
  a < 8 :=
by
  sorry

-- Problem (2)
theorem tangent_lines (a : ℝ) (h : a = -17) : 
  ∃ (k : ℝ), k * 7 - 6 - 7 * k = 0 ∧
  ((39 * k + 80 * (-7) - 207 = 0) ∨ (k = 7)) :=
by
  sorry

-- Problem (3)
theorem perpendicular_circle_intersection (x1 x2 y1 y2 a : ℝ) 
  (h1: 2 * x1 - y1 - 3 = 0) 
  (h2: 2 * x2 - y2 - 3 = 0) 
  (h3: x1 * x2 + y1 * y2 = 0) 
  (hpoly : 5 * x1 * x2 - 6 * (x1 + x2) + 9 = 0): 
  a = -6 / 5 :=
by
  sorry

end circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l189_189321


namespace degrees_to_radians_conversion_l189_189140

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end degrees_to_radians_conversion_l189_189140


namespace share_of_C_l189_189495

theorem share_of_C (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 578) : 
  C = 408 :=
by
  -- Proof goes here
  sorry

end share_of_C_l189_189495


namespace increasing_function_when_a_eq_2_range_of_a_for_solution_set_l189_189323

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x - a * (x - 1) / (x + 1)

theorem increasing_function_when_a_eq_2 :
  ∀ ⦃x⦄, x > 0 → (f 2 x - f 2 1) * (x - 1) > 0 := sorry

theorem range_of_a_for_solution_set :
  ∀ ⦃a x⦄, f a x ≥ 0 ↔ (x ≥ 1) → a ≤ 1 := sorry

end increasing_function_when_a_eq_2_range_of_a_for_solution_set_l189_189323


namespace probability_two_girls_l189_189647

theorem probability_two_girls (total_students girls boys : ℕ) (htotal : total_students = 6) (hg : girls = 4) (hb : boys = 2) :
  (Nat.choose girls 2 / Nat.choose total_students 2 : ℝ) = 2 / 5 := by
  sorry

end probability_two_girls_l189_189647


namespace total_daisies_l189_189343

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l189_189343


namespace minimum_soldiers_to_add_l189_189625

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189625


namespace minimum_soldiers_to_add_l189_189642

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189642


namespace rhombus_perimeter_area_l189_189505

theorem rhombus_perimeter_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (right_angle : ∀ (x : ℝ), x = d1 / 2 ∧ x = d2 / 2 → x * x + x * x = (d1 / 2)^2 + (d2 / 2)^2) : 
  ∃ (P A : ℝ), P = 52 ∧ A = 120 :=
by
  sorry

end rhombus_perimeter_area_l189_189505


namespace equivalent_proof_problem_l189_189464

-- Define the conditions as Lean 4 definitions
variable (x₁ x₂ : ℝ)

-- The conditions given in the problem
def condition1 : Prop := x₁ * Real.logb 2 x₁ = 1008
def condition2 : Prop := x₂ * 2^x₂ = 1008

-- The problem to be proved
theorem equivalent_proof_problem (hx₁ : condition1 x₁) (hx₂ : condition2 x₂) : 
  x₁ * x₂ = 1008 := 
sorry

end equivalent_proof_problem_l189_189464


namespace students_not_receiving_A_l189_189807

theorem students_not_receiving_A (total_students : ℕ) (students_A_physics : ℕ) (students_A_chemistry : ℕ) (students_A_both : ℕ) (h_total : total_students = 40) (h_A_physics : students_A_physics = 10) (h_A_chemistry : students_A_chemistry = 18) (h_A_both : students_A_both = 6) : (total_students - ((students_A_physics + students_A_chemistry) - students_A_both)) = 18 := 
by
  sorry

end students_not_receiving_A_l189_189807


namespace approx_ineq_l189_189289

noncomputable def approx (x : ℝ) : ℝ := 1 + 6 * (-0.002 : ℝ)

theorem approx_ineq (x : ℝ) (h : x = 0.998) : 
  abs ((x^6) - approx x) < 0.001 :=
by
  sorry

end approx_ineq_l189_189289


namespace price_of_first_shirt_l189_189543

theorem price_of_first_shirt
  (price1 price2 price3 : ℕ)
  (total_shirts : ℕ)
  (min_avg_price_of_remaining : ℕ)
  (total_avg_price_of_all : ℕ)
  (prices_of_first_3 : price1 = 100 ∧ price2 = 90 ∧ price3 = 82)
  (condition1 : total_shirts = 10)
  (condition2 : min_avg_price_of_remaining = 104)
  (condition3 : total_avg_price_of_all > 100) :
  price1 = 100 :=
by
  sorry

end price_of_first_shirt_l189_189543


namespace amara_clothing_remaining_l189_189416

theorem amara_clothing_remaining :
  (initial_clothing - donated_first - donated_second - thrown_away = remaining_clothing) :=
by
  let initial_clothing := 100
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining_clothing := 65
  sorry

end amara_clothing_remaining_l189_189416


namespace problem_l189_189073

theorem problem (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := by
  sorry

end problem_l189_189073


namespace part_a_no_solutions_part_a_infinite_solutions_l189_189867

theorem part_a_no_solutions (a : ℝ) (x y : ℝ) : 
    a = -1 → ¬(∃ x y : ℝ, a * x + y = a^2 ∧ x + a * y = 1) :=
sorry

theorem part_a_infinite_solutions (a : ℝ) (x y : ℝ) : 
    a = 1 → ∃ x : ℝ, ∃ y : ℝ, a * x + y = a^2 ∧ x + a * y = 1 :=
sorry

end part_a_no_solutions_part_a_infinite_solutions_l189_189867


namespace find_flights_of_stairs_l189_189831

def t_flight : ℕ := 11
def t_bomb : ℕ := 72
def t_spent : ℕ := 165
def t_diffuse : ℕ := 17

def total_time_running : ℕ := t_spent + (t_bomb - t_diffuse)
def flights_of_stairs (t_run: ℕ) (time_per_flight: ℕ) : ℕ := t_run / time_per_flight

theorem find_flights_of_stairs :
  flights_of_stairs total_time_running t_flight = 20 :=
by
  sorry

end find_flights_of_stairs_l189_189831


namespace larger_number_l189_189856

theorem larger_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 4) : x = 17 :=
by
sorry

end larger_number_l189_189856


namespace purchase_price_mobile_l189_189370

-- Definitions of the given conditions
def purchase_price_refrigerator : ℝ := 15000
def loss_percent_refrigerator : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10
def overall_profit : ℝ := 50

-- Defining the statement to prove
theorem purchase_price_mobile (P : ℝ)
  (h1 : purchase_price_refrigerator = 15000)
  (h2 : loss_percent_refrigerator = 0.05)
  (h3 : profit_percent_mobile = 0.10)
  (h4 : overall_profit = 50) :
  (15000 * (1 - 0.05) + P * (1 + 0.10)) - (15000 + P) = 50 → P = 8000 :=
by {
  -- Proof is omitted
  sorry
}

end purchase_price_mobile_l189_189370


namespace eval_g_at_2_l189_189173

def g (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem eval_g_at_2 : g 2 = 3 :=
by {
  -- This is the place for proof steps, currently it is filled with sorry.
  sorry
}

end eval_g_at_2_l189_189173


namespace print_shop_Y_charge_l189_189431

variable (y : ℝ)
variable (X_charge_per_copy : ℝ := 1.25)
variable (Y_charge_for_60_color_copies : ℝ := 60 * y)
variable (X_charge_for_60_color_copies : ℝ := 60 * X_charge_per_copy)
variable (additional_cost_at_Y : ℝ := 90)

theorem print_shop_Y_charge :
  Y_charge_for_60_color_copies = X_charge_for_60_color_copies + additional_cost_at_Y → y = 2.75 := by
  sorry

end print_shop_Y_charge_l189_189431


namespace eval_expression_l189_189017

theorem eval_expression : (825 * 825) - (824 * 826) = 1 := by
  sorry

end eval_expression_l189_189017


namespace total_fruits_sum_l189_189132

theorem total_fruits_sum (Mike_oranges Matt_apples Mark_bananas Mary_grapes : ℕ)
  (hMike : Mike_oranges = 3)
  (hMatt : Matt_apples = 2 * Mike_oranges)
  (hMark : Mark_bananas = Mike_oranges + Matt_apples)
  (hMary : Mary_grapes = Mike_oranges + Matt_apples + Mark_bananas + 5) :
  Mike_oranges + Matt_apples + Mark_bananas + Mary_grapes = 41 :=
by
  sorry

end total_fruits_sum_l189_189132


namespace lcm_36_100_l189_189303

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l189_189303


namespace boys_more_than_girls_l189_189800

theorem boys_more_than_girls
  (x y a b : ℕ)
  (h1 : x > y)
  (h2 : x * a + y * b = x * b + y * a - 1) :
  x = y + 1 :=
sorry

end boys_more_than_girls_l189_189800


namespace segment_association_l189_189826

theorem segment_association (x y : ℝ) 
  (h1 : ∃ (D : ℝ), ∀ (P : ℝ), abs (P - D) ≤ 5) 
  (h2 : ∃ (D' : ℝ), ∀ (P' : ℝ), abs (P' - D') ≤ 9)
  (h3 : 3 * x - 2 * y = 6) : 
  x + y = 12 := 
by sorry

end segment_association_l189_189826


namespace prove_sum_l189_189868

theorem prove_sum (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := by
  sorry

end prove_sum_l189_189868


namespace average_increase_by_3_l189_189111

def initial_average_before_inning_17 (A : ℝ) : Prop :=
  16 * A + 85 = 17 * 37

theorem average_increase_by_3 (A : ℝ) (h : initial_average_before_inning_17 A) :
  37 - A = 3 :=
by
  sorry

end average_increase_by_3_l189_189111


namespace problem_statement_l189_189574

theorem problem_statement (a b : ℝ) (h1 : 1 / a + 1 / b = Real.sqrt 5) (h2 : a ≠ b) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 :=
by
  sorry

end problem_statement_l189_189574


namespace opposite_number_of_sqrt_of_9_is_neg3_l189_189518

theorem opposite_number_of_sqrt_of_9_is_neg3 :
  - (Real.sqrt 9) = -3 :=
by
  -- The proof is omitted as required.
  sorry

end opposite_number_of_sqrt_of_9_is_neg3_l189_189518


namespace least_clock_equivalent_l189_189833

def clock_equivalent (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + 12 * k = b

theorem least_clock_equivalent (h : ℕ) (hh : h > 3) (hq : clock_equivalent h (h * h)) :
  h = 4 :=
by
  sorry

end least_clock_equivalent_l189_189833


namespace eval_f_at_4_l189_189458

def f (x : ℕ) : ℕ := 5 * x + 2

theorem eval_f_at_4 : f 4 = 22 :=
by
  sorry

end eval_f_at_4_l189_189458


namespace find_XY_square_l189_189964

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end find_XY_square_l189_189964


namespace polyhedron_faces_l189_189115

theorem polyhedron_faces (V E F T P t p : ℕ)
  (hF : F = 20)
  (hFaces : t + p = 20)
  (hTriangles : t = 2 * p)
  (hVertex : T = 2 ∧ P = 2)
  (hEdges : E = (3 * t + 5 * p) / 2)
  (hEuler : V - E + F = 2) :
  100 * P + 10 * T + V = 238 :=
by
  sorry

end polyhedron_faces_l189_189115


namespace sum_of_inserted_numbers_l189_189470

theorem sum_of_inserted_numbers (x y : ℝ) (h1 : x^2 = 2 * y) (h2 : 2 * y = x + 20) :
  x + y = 4 ∨ x + y = 17.5 :=
sorry

end sum_of_inserted_numbers_l189_189470


namespace shooting_prob_l189_189415

theorem shooting_prob (p : ℝ) (h₁ : (1 / 3) * (1 / 2) * (1 - p) + (1 / 3) * (1 / 2) * p + (2 / 3) * (1 / 2) * p = 7 / 18) :
  p = 2 / 3 :=
sorry

end shooting_prob_l189_189415


namespace correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l189_189401

theorem correct_operation_B (a : ℝ) : a^3 / a = a^2 := 
by sorry

theorem incorrect_operation_A (a : ℝ) : a^2 + a^5 ≠ a^7 := 
by sorry

theorem incorrect_operation_C (a : ℝ) : (3 * a^2)^2 ≠ 6 * a^4 := 
by sorry

theorem incorrect_operation_D (a b : ℝ) : (a - b)^2 ≠ a^2 - b^2 := 
by sorry

end correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l189_189401


namespace minimum_soldiers_to_add_l189_189623

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189623


namespace max_value_sqrt_sum_l189_189668

open Real

noncomputable def max_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) : ℝ :=
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) :
  max_sqrt_sum x y z h1 h2 h3 h_sum ≤ 3 * sqrt 8 :=
sorry

end max_value_sqrt_sum_l189_189668


namespace A_minus_3B_A_minus_3B_independent_of_y_l189_189153

variables (x y : ℝ)
def A : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B : ℝ := x^2 - 2*x - y + x*y - 5

theorem A_minus_3B (x y : ℝ) : A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

theorem A_minus_3B_independent_of_y (x : ℝ) (hyp : ∀ y : ℝ, A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15) :
  5 - 7*x = 0 → x = 5 / 7 :=
by
  sorry

end A_minus_3B_A_minus_3B_independent_of_y_l189_189153


namespace gina_netflix_time_l189_189432

theorem gina_netflix_time (sister_shows : ℕ) (show_length : ℕ) (ratio : ℕ) (sister_ratio : ℕ) :
sister_shows = 24 →
show_length = 50 →
ratio = 3 →
sister_ratio = 1 →
(ratio * sister_shows * show_length = 3600) :=
begin
  intros hs hl hr hsr,
  rw hs,
  rw hl,
  rw hr,
  rw hsr,
  norm_num,
  sorry
end

end gina_netflix_time_l189_189432


namespace harry_started_with_79_l189_189329

-- Definitions using the conditions
def harry_initial_apples (x : ℕ) : Prop :=
  (x + 5 = 84)

-- Theorem statement proving the initial number of apples Harry started with
theorem harry_started_with_79 : ∃ x : ℕ, harry_initial_apples x ∧ x = 79 :=
by
  sorry

end harry_started_with_79_l189_189329


namespace isaiah_types_more_words_than_micah_l189_189972

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l189_189972


namespace divisibility_by_n_l189_189845

variable (a b c : ℤ) (n : ℕ)

theorem divisibility_by_n
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2 * n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * ↑n := 
sorry

end divisibility_by_n_l189_189845


namespace total_digits_l189_189683

theorem total_digits (n S S6 S4 : ℕ) 
  (h1 : S = 80 * n)
  (h2 : S6 = 6 * 58)
  (h3 : S4 = 4 * 113)
  (h4 : S = S6 + S4) : 
  n = 10 :=
by 
  sorry

end total_digits_l189_189683


namespace scientific_notation_of_number_l189_189811

theorem scientific_notation_of_number :
  (0.0000000033 : ℝ) = 3.3 * 10^(-9) :=
sorry

end scientific_notation_of_number_l189_189811


namespace domain_tan_2x_plus_pi_over_3_l189_189219

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def domain_tan_transformed : Set ℝ :=
  {x | ∃ k : ℤ, x = k * (Real.pi / 2) + Real.pi / 12}

theorem domain_tan_2x_plus_pi_over_3 :
  (∀ x, ¬ (x ∈ domain_tan)) ↔ (∀ x, ¬ (x ∈ domain_tan_transformed)) :=
by
  sorry

end domain_tan_2x_plus_pi_over_3_l189_189219


namespace find_b_l189_189546

def oscillation_period (a b c d : ℝ) (oscillations : ℝ) : Prop :=
  oscillations = 5 * (2 * Real.pi) / b

theorem find_b
  (a b c d : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0)
  (osc_complexity: oscillation_period a b c d 5):
  b = 5 := by
  sorry

end find_b_l189_189546


namespace roots_sum_l189_189481

theorem roots_sum (a b : ℝ) 
  (h1 : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
          (roots_of (λ x, x^3 - 8 * x^2 + a * x - b) = {r1, r2, r3}) ) :
  a + b = 31 :=
sorry

end roots_sum_l189_189481


namespace fraction_division_l189_189231

theorem fraction_division :
  (3 / 4) / (5 / 6) = 9 / 10 :=
by {
  -- We skip the proof as per the instructions
  sorry
}

end fraction_division_l189_189231


namespace sum_of_midpoints_l189_189993

variable (a b c : ℝ)

def sum_of_vertices := a + b + c

theorem sum_of_midpoints (h : sum_of_vertices a b c = 15) :
  (a + b)/2 + (a + c)/2 + (b + c)/2 = 15 :=
by
  sorry

end sum_of_midpoints_l189_189993


namespace correct_inequality_relation_l189_189585

theorem correct_inequality_relation :
  ¬(∀ (a b c : ℝ), a > b ↔ a * (c^2) > b * (c^2)) ∧
  ¬(∀ (a b : ℝ), a > b → (1/a) < (1/b)) ∧
  ¬(∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∀ (a b c : ℝ), a > b ∧ b > 1 ∧ c < 0 → a^c < b^c) := sorry

end correct_inequality_relation_l189_189585


namespace minimum_soldiers_to_add_l189_189639

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189639


namespace harriet_ran_48_miles_l189_189025

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end harriet_ran_48_miles_l189_189025


namespace find_coordinates_l189_189928

def pointA : ℝ × ℝ := (2, -4)
def pointB : ℝ × ℝ := (0, 6)
def pointC : ℝ × ℝ := (-8, 10)

def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_coordinates :
  scalar_mult (1/2) (vector pointA pointC) - 
  scalar_mult (1/4) (vector pointB pointC) = (-3, 6) :=
by
  sorry

end find_coordinates_l189_189928


namespace sum_x_coordinates_midpoints_l189_189223

theorem sum_x_coordinates_midpoints (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_x_coordinates_midpoints_l189_189223


namespace greatest_matching_pairs_left_l189_189534

-- Define the initial number of pairs and lost individual shoes
def initial_pairs : ℕ := 26
def lost_ind_shoes : ℕ := 9

-- The statement to be proved
theorem greatest_matching_pairs_left : 
  (initial_pairs * 2 - lost_ind_shoes) / 2 + (initial_pairs - (initial_pairs * 2 - lost_ind_shoes) / 2) / 1 = 17 := 
by 
  sorry

end greatest_matching_pairs_left_l189_189534


namespace rectangle_to_square_y_l189_189412

theorem rectangle_to_square_y (y : ℝ) (a b : ℝ) (s : ℝ) (h1 : a = 7) (h2 : b = 21)
  (h3 : s^2 = a * b) (h4 : y = s / 2) : y = 7 * Real.sqrt 3 / 2 :=
by
  -- proof skipped
  sorry

end rectangle_to_square_y_l189_189412


namespace train_pass_jogger_in_40_seconds_l189_189739

noncomputable def time_to_pass_jogger (jogger_speed_kmh : ℝ) (train_speed_kmh : ℝ) (initial_distance_m : ℝ) (train_length_m : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - jogger_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)  -- Conversion from km/hr to m/s
  let total_distance_m := initial_distance_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_pass_jogger_in_40_seconds :
  time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_pass_jogger_in_40_seconds_l189_189739


namespace no_positive_ints_cube_l189_189039

theorem no_positive_ints_cube (n : ℕ) : ¬ ∃ y : ℕ, 3 * n^2 + 3 * n + 7 = y^3 := 
sorry

end no_positive_ints_cube_l189_189039


namespace apple_pie_theorem_l189_189054

theorem apple_pie_theorem (total_apples : ℕ) (not_ripe_apples : ℕ) (apples_per_pie : ℕ) (total_ripe_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 34)
  (h2 : not_ripe_apples = 6)
  (h3 : apples_per_pie = 4)
  (h4 : total_ripe_apples = total_apples - not_ripe_apples)
  (h5 : number_of_pies = total_ripe_apples / apples_per_pie) :
  number_of_pies = 7 :=
  by
  have h6 : total_apples - not_ripe_apples = 28 := by rw [h1, h2]; norm_num
  have h7 : total_ripe_apples = 28 := by rw [h4, h6]
  have h8 : 28 / apples_per_pie = 7 := by rw [h3]; norm_num
  rw [h7, h5, h8]
  sorry

end apple_pie_theorem_l189_189054


namespace soldiers_to_add_l189_189636

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189636


namespace evaluate_expression_l189_189897

-- Define the terms a and b
def a : ℕ := 2023
def b : ℕ := 2024

-- The given expression
def expression : ℤ := (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b)

-- The theorem to prove
theorem evaluate_expression : expression = ↑a := 
by sorry

end evaluate_expression_l189_189897


namespace binom_eq_one_binom_320_l189_189013

theorem binom_eq_one (n : ℕ) : (n.choose n) = 1 :=
  by sorry

theorem binom_320 : Nat.choose 320 320 = 1 :=
  by exact binom_eq_one 320

end binom_eq_one_binom_320_l189_189013


namespace triangle_split_points_l189_189812

noncomputable def smallest_n_for_split (AB BC CA : ℕ) : ℕ := 
  if AB = 13 ∧ BC = 14 ∧ CA = 15 then 27 else sorry

theorem triangle_split_points (AB BC CA : ℕ) (h : AB = 13 ∧ BC = 14 ∧ CA = 15) :
  smallest_n_for_split AB BC CA = 27 :=
by
  cases h with | intro h1 h23 => sorry

-- Assertions for the explicit values provided in the conditions
example : smallest_n_for_split 13 14 15 = 27 :=
  triangle_split_points 13 14 15 ⟨rfl, rfl, rfl⟩

end triangle_split_points_l189_189812


namespace count_divisible_by_4_6_10_l189_189590

theorem count_divisible_by_4_6_10 :
  (card {n : ℕ | n < 300 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0}) = 4 :=
by 
  sorry

end count_divisible_by_4_6_10_l189_189590


namespace karl_savings_l189_189818

noncomputable def cost_per_notebook : ℝ := 3.75
noncomputable def notebooks_bought : ℕ := 8
noncomputable def discount_rate : ℝ := 0.25
noncomputable def original_total_cost : ℝ := notebooks_bought * cost_per_notebook
noncomputable def discount_per_notebook : ℝ := cost_per_notebook * discount_rate
noncomputable def discounted_price_per_notebook : ℝ := cost_per_notebook - discount_per_notebook
noncomputable def discounted_total_cost : ℝ := notebooks_bought * discounted_price_per_notebook
noncomputable def total_savings : ℝ := original_total_cost - discounted_total_cost

theorem karl_savings : total_savings = 7.50 := by 
  sorry

end karl_savings_l189_189818


namespace find_positive_integers_l189_189143

theorem find_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (1 / m + 1 / n - 1 / (m * n) = 2 / 5) ↔ 
  (m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4) :=
by sorry

end find_positive_integers_l189_189143


namespace roots_of_cubic_l189_189230

-- Define the cubic equation having roots 3 and -2
def cubic_eq (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The proof problem statement
theorem roots_of_cubic (a b c d : ℝ) (h₁ : a ≠ 0)
  (h₂ : cubic_eq a b c d 3)
  (h₃ : cubic_eq a b c d (-2)) : 
  (b + c) / a = -7 := 
sorry

end roots_of_cubic_l189_189230


namespace part_I_min_value_part_II_a_range_l189_189325

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) - abs (x + 3)

theorem part_I_min_value (x : ℝ) : f x 1 ≥ -7 / 2 :=
by sorry 

theorem part_II_a_range (x a : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 3) (hf : f x a ≤ 4) : -4 ≤ a ∧ a ≤ 7 :=
by sorry

end part_I_min_value_part_II_a_range_l189_189325


namespace arithmetic_sequence_properties_l189_189773

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h2 : d ≠ 0)
  (h3 : ∀ n, S n ≤ S 8) :
  d < 0 ∧ S 17 ≤ 0 := 
sorry

end arithmetic_sequence_properties_l189_189773


namespace range_of_m_l189_189182

theorem range_of_m 
  (m : ℝ)
  (h1 : ∀ (x : ℤ), x < m → 7 - 2 * x ≤ 1)
  (h2 : ∃ k : ℕ, set_of (λ i : ℤ, 3 ≤ i ∧ i < m).card = k ∧ k = 4) :
  6 < m ∧ m ≤ 7 :=
sorry

end range_of_m_l189_189182


namespace ratio_of_ages_l189_189233

noncomputable def ratio_4th_to_3rd (age1 age2 age3 age4 age5 : ℕ) : ℚ :=
  age4 / age3

theorem ratio_of_ages
  (age1 age2 age3 age4 age5 : ℕ)
  (h1 : (age1 + age5) / 2 = 18)
  (h2 : age1 = 10)
  (h3 : age2 = age1 - 2)
  (h4 : age3 = age2 + 4)
  (h5 : age4 = age3 / 2)
  (h6 : age5 = age4 + 20) :
  ratio_4th_to_3rd age1 age2 age3 age4 age5 = 1 / 2 :=
by
  sorry

end ratio_of_ages_l189_189233


namespace proof_expression_equals_60_times_10_power_1501_l189_189549

noncomputable def expression_equals_60_times_10_power_1501 : Prop :=
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501

theorem proof_expression_equals_60_times_10_power_1501 :
  expression_equals_60_times_10_power_1501 :=
by 
  sorry

end proof_expression_equals_60_times_10_power_1501_l189_189549


namespace lcm_of_36_and_100_l189_189294

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l189_189294


namespace gcd_1443_999_l189_189693

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end gcd_1443_999_l189_189693


namespace contrapositive_negation_l189_189456

-- Define the main condition of the problem
def statement_p (x y : ℝ) : Prop :=
  (x - 1) * (y + 2) = 0 → (x = 1 ∨ y = -2)

-- Prove the contrapositive of statement_p
theorem contrapositive (x y : ℝ) : 
  (x ≠ 1 ∧ y ≠ -2) → ¬ ((x - 1) * (y + 2) = 0) :=
by 
  sorry

-- Prove the negation of statement_p
theorem negation (x y : ℝ) : 
  ((x - 1) * (y + 2) = 0) → ¬ (x = 1 ∨ y = -2) :=
by 
  sorry

end contrapositive_negation_l189_189456


namespace james_birthday_stickers_l189_189108

def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

def birthday_stickers (s_initial s_final : ℕ) : ℕ := s_final - s_initial

theorem james_birthday_stickers :
  birthday_stickers initial_stickers final_stickers = 22 := by
  sorry

end james_birthday_stickers_l189_189108


namespace division_remainder_l189_189429

-- let f(r) = r^15 + r + 1
def f (r : ℝ) : ℝ := r^15 + r + 1

-- let g(r) = r^2 - 1
def g (r : ℝ) : ℝ := r^2 - 1

-- remainder polynomial b(r)
def b (r : ℝ) : ℝ := r + 1

-- Lean statement to prove that polynomial division of f(r) by g(r) 
-- yields the remainder b(r)
theorem division_remainder (r : ℝ) : (f r) % (g r) = b r :=
  sorry

end division_remainder_l189_189429


namespace ellipse_equation_l189_189985

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end ellipse_equation_l189_189985


namespace basketball_shots_l189_189044

theorem basketball_shots (total_points total_3pt_shots: ℕ) 
  (h1: total_points = 26) 
  (h2: total_3pt_shots = 4) 
  (h3: ∀ points_from_3pt_shots, points_from_3pt_shots = 3 * total_3pt_shots) :
  let points_from_3pt_shots := 3 * total_3pt_shots
  let points_from_2pt_shots := total_points - points_from_3pt_shots
  let total_2pt_shots := points_from_2pt_shots / 2
  total_2pt_shots + total_3pt_shots = 11 :=
by
  sorry

end basketball_shots_l189_189044


namespace planet_not_observed_l189_189489

theorem planet_not_observed (k : ℕ) (d : Fin (2*k+1) → Fin (2*k+1) → ℝ) 
  (h_d : ∀ i j : Fin (2*k+1), i ≠ j → d i i = 0 ∧ d i j ≠ d i i) 
  (h_astronomer : ∀ i : Fin (2*k+1), ∃ j : Fin (2*k+1), j ≠ i ∧ ∀ k : Fin (2*k+1), k ≠ i → d i j < d i k) : 
  ∃ i : Fin (2*k+1), ∀ j : Fin (2*k+1), i ≠ j → ∃ l : Fin (2*k+1), (j ≠ l ∧ d l i < d l j) → false :=
  sorry

end planet_not_observed_l189_189489


namespace pirate_coins_l189_189541

def coins_remain (k : ℕ) (x : ℕ) : ℕ :=
  if k = 0 then x else coins_remain (k - 1) x * (15 - k) / 15

theorem pirate_coins (x : ℕ) :
  (∀ k < 15, (k + 1) * coins_remain k x % 15 = 0) → 
  coins_remain 14 x = 8442 :=
sorry

end pirate_coins_l189_189541


namespace max_min_x2_sub_xy_add_y2_l189_189669

/-- Given a point \((x, y)\) on the curve defined by \( |5x + y| + |5x - y| = 20 \), prove that the maximum value of \(x^2 - xy + y^2\) is 124 and the minimum value is 3. -/
theorem max_min_x2_sub_xy_add_y2 (x y : ℝ) (h : abs (5 * x + y) + abs (5 * x - y) = 20) :
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_sub_xy_add_y2_l189_189669


namespace minimum_gb_for_cheaper_plan_l189_189126

theorem minimum_gb_for_cheaper_plan : ∃ g : ℕ, (g ≥ 778) ∧ 
  (∀ g' < 778, 3000 + (if g' ≤ 500 then 8 * g' else 8 * 500 + 6 * (g' - 500)) ≥ 15 * g') ∧ 
  3000 + (if g ≤ 500 then 8 * g else 8 * 500 + 6 * (g - 500)) < 15 * g :=
by
  sorry

end minimum_gb_for_cheaper_plan_l189_189126


namespace result_when_j_divided_by_26_l189_189801

noncomputable def j := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 10 11) (Nat.lcm 12 13)) (Nat.lcm 14 15))

theorem result_when_j_divided_by_26 : j / 26 = 2310 := by 
  sorry

end result_when_j_divided_by_26_l189_189801


namespace common_ratio_q_l189_189577

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => geom_seq a q n * q

def sum_geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => sum_geom_seq a q n + geom_seq a q (n + 1)

theorem common_ratio_q (a q : α) (hq : 0 < q) (h_inc : ∀ n, geom_seq a q n < geom_seq a q (n + 1))
  (h1 : geom_seq a q 1 = 2)
  (h2 : sum_geom_seq a q 2 = 7) :
  q = 2 :=
sorry

end common_ratio_q_l189_189577


namespace trajectory_equation_necessary_not_sufficient_l189_189384

theorem trajectory_equation_necessary_not_sufficient :
  ∀ (x y : ℝ), (|x| = |y|) → (y = |x|) ↔ (necessary_not_sufficient) :=
by
  sorry

end trajectory_equation_necessary_not_sufficient_l189_189384


namespace smaller_angle_in_parallelogram_l189_189804

theorem smaller_angle_in_parallelogram 
  (opposite_angles : ∀ A B C D : ℝ, A = C ∧ B = D)
  (adjacent_angles_supplementary : ∀ A B : ℝ, A + B = π)
  (angle_diff : ∀ A B : ℝ, B = A + π/9) :
  ∃ θ : ℝ, θ = 4 * π / 9 :=
by
  sorry

end smaller_angle_in_parallelogram_l189_189804


namespace quadratic_roots_l189_189696

theorem quadratic_roots {a : ℝ} :
  (4 < a ∧ a < 6) ∨ (a > 12) → 
  (∃ x1 x2 : ℝ, x1 = a + Real.sqrt (18 * (a - 4)) ∧ x2 = a - Real.sqrt (18 * (a - 4)) ∧ x1 > 0 ∧ x2 > 0) :=
by sorry

end quadratic_roots_l189_189696


namespace used_car_percentage_l189_189708

-- Define the variables and conditions
variables (used_car_price original_car_price : ℕ) (h_used_car_price : used_car_price = 15000) (h_original_price : original_car_price = 37500)

-- Define the statement to prove the percentage
theorem used_car_percentage (h : used_car_price / original_car_price * 100 = 40) : true :=
sorry

end used_car_percentage_l189_189708


namespace min_soldiers_needed_l189_189609

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189609


namespace no_sphinx_tiling_l189_189889

def equilateral_triangle_tiling_problem (side_length : ℕ) (pointing_up : ℕ) (pointing_down : ℕ) : Prop :=
  let total_triangles := side_length * side_length
  pointing_up + pointing_down = total_triangles ∧ 
  total_triangles = 36 ∧
  pointing_down = 1 + 2 + 3 + 4 + 5 ∧
  pointing_up = 1 + 2 + 3 + 4 + 5 + 6 ∧
  (pointing_up % 2 = 1) ∧
  (pointing_down % 2 = 1) ∧
  (2 * pointing_up + 4 * pointing_down ≠ total_triangles ∧ 4 * pointing_up + 2 * pointing_down ≠ total_triangles)

theorem no_sphinx_tiling : ¬equilateral_triangle_tiling_problem 6 21 15 :=
by
  sorry

end no_sphinx_tiling_l189_189889


namespace tile_count_difference_l189_189205

theorem tile_count_difference :
  let red_initial := 15
  let yellow_initial := 10
  let yellow_added := 18
  let yellow_total := yellow_initial + yellow_added
  let red_total := red_initial
  yellow_total - red_total = 13 :=
by
  sorry

end tile_count_difference_l189_189205


namespace lcm_36_100_eq_900_l189_189297

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l189_189297


namespace robotics_club_students_l189_189360

theorem robotics_club_students
  (total_students : ℕ)
  (cs_students : ℕ)
  (electronics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 50)
  (h3 : electronics_students = 35)
  (h4 : both_students = 25) :
  total_students - (cs_students - both_students + electronics_students - both_students + both_students) = 20 :=
by
  sorry

end robotics_club_students_l189_189360


namespace counterexample_to_proposition_l189_189103

theorem counterexample_to_proposition (a b : ℝ) (ha : a = 1) (hb : b = -1) :
  a > b ∧ ¬ (1 / a < 1 / b) :=
by
  sorry

end counterexample_to_proposition_l189_189103


namespace colorings_equivalence_l189_189810

-- Define the problem setup
structure ProblemSetup where
  n : ℕ  -- Number of disks (8)
  blue : ℕ  -- Number of blue disks (3)
  red : ℕ  -- Number of red disks (3)
  green : ℕ  -- Number of green disks (2)
  rotations : ℕ  -- Number of rotations (4: 90°, 180°, 270°, 360°)
  reflections : ℕ  -- Number of reflections (8: 4 through vertices and 4 through midpoints)

def number_of_colorings (setup : ProblemSetup) : ℕ :=
  sorry -- This represents the complex implementation details

def correct_answer : ℕ := 43

theorem colorings_equivalence : ∀ (setup : ProblemSetup),
  setup.n = 8 → setup.blue = 3 → setup.red = 3 → setup.green = 2 → setup.rotations = 4 → setup.reflections = 8 →
  number_of_colorings setup = correct_answer :=
by
  intros setup h1 h2 h3 h4 h5 h6
  sorry

end colorings_equivalence_l189_189810


namespace total_molecular_weight_of_products_l189_189283

/-- Problem Statement: Determine the total molecular weight of the products formed when
    8 moles of Copper(II) carbonate (CuCO3) react with 6 moles of Diphosphorus pentoxide (P4O10)
    to form Copper(II) phosphate (Cu3(PO4)2) and Carbon dioxide (CO2). -/
theorem total_molecular_weight_of_products 
  (moles_CuCO3 : ℕ) 
  (moles_P4O10 : ℕ)
  (atomic_weight_Cu : ℝ := 63.55)
  (atomic_weight_P : ℝ := 30.97)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_C : ℝ := 12.01)
  (molecular_weight_CuCO3 : ℝ := atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O)
  (molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O)
  (molecular_weight_Cu3PO4_2 : ℝ := (3 * atomic_weight_Cu) + (2 * atomic_weight_P) + (8 * atomic_weight_O))
  (moles_Cu3PO4_2_formed : ℝ := (8 : ℝ) / 3)
  (moles_CO2_formed : ℝ := 8)
  (total_molecular_weight_Cu3PO4_2 : ℝ := moles_Cu3PO4_2_formed * molecular_weight_Cu3PO4_2)
  (total_molecular_weight_CO2 : ℝ := moles_CO2_formed * molecular_weight_CO2) : 
  (total_molecular_weight_Cu3PO4_2 + total_molecular_weight_CO2) = 1368.45 := by
  sorry

end total_molecular_weight_of_products_l189_189283


namespace sum_of_products_l189_189091

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  ab + bc + ac = 72 := 
by 
  sorry

end sum_of_products_l189_189091


namespace shared_candy_equally_l189_189594

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l189_189594


namespace r_at_5_l189_189667

def r (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 - 1

theorem r_at_5 :
  r 5 = 48 := by
  sorry

end r_at_5_l189_189667


namespace percentage_change_difference_l189_189950

theorem percentage_change_difference (total_students : ℕ) (initial_enjoy : ℕ) (initial_not_enjoy : ℕ) (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  total_students = 100 →
  initial_enjoy = 40 →
  initial_not_enjoy = 60 →
  final_enjoy = 80 →
  final_not_enjoy = 20 →
  (40 ≤ y ∧ y ≤ 80) ∧ (40 - 40 = 0) ∧ (80 - 40 = 40) ∧ (80 - 40 = 40) :=
by
  sorry

end percentage_change_difference_l189_189950


namespace min_value_expression_l189_189315

theorem min_value_expression (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) :
  xy + (1 / xy) = 17 / 4 :=
sorry

end min_value_expression_l189_189315


namespace angle_B_value_l189_189192

theorem angle_B_value (a b c B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
    B = (Real.pi / 3) ∨ B = (2 * Real.pi / 3) :=
by
    sorry

end angle_B_value_l189_189192


namespace theta_solutions_count_l189_189040

theorem theta_solutions_count :
  (∃ (count : ℕ), count = 4 ∧ ∀ θ, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 1 - 4 * Real.sin θ + 5 * Real.cos (2 * θ) = 0 ↔ count = 4) :=
sorry

end theta_solutions_count_l189_189040


namespace first_term_geometric_progression_l189_189998

theorem first_term_geometric_progression (a r : ℝ) 
  (h1 : a / (1 - r) = 6)
  (h2 : a + a * r = 9 / 2) :
  a = 3 ∨ a = 9 := 
sorry -- Proof omitted

end first_term_geometric_progression_l189_189998


namespace repeating_decimal_to_fraction_l189_189709

theorem repeating_decimal_to_fraction : 
  (x : ℝ) (h : x = 0.4 + 36 / (10^1 + 10^2 + 10^3 + ...)) : x = 24 / 55 :=
sorry

end repeating_decimal_to_fraction_l189_189709


namespace sequence_formula_l189_189191

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (a (n - 1)) + 2^(n-1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
    a n = 2^n - 1 := 
sorry

end sequence_formula_l189_189191


namespace min_soldiers_needed_l189_189612

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189612


namespace leon_older_than_aivo_in_months_l189_189563

theorem leon_older_than_aivo_in_months
    (jolyn therese aivo leon : ℕ)
    (h1 : jolyn = therese + 2)
    (h2 : therese = aivo + 5)
    (h3 : jolyn = leon + 5) :
    leon = aivo + 2 := 
sorry

end leon_older_than_aivo_in_months_l189_189563


namespace factorization_example_l189_189521

theorem factorization_example (x : ℝ) : (x^2 - 4 * x + 4) = (x - 2)^2 :=
by sorry

end factorization_example_l189_189521


namespace lcm_36_100_is_900_l189_189302

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l189_189302


namespace range_of_a_for_inequality_l189_189784

noncomputable def has_solution_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (x^2 + a*x - 2 < 0)

theorem range_of_a_for_inequality : ∀ a : ℝ, has_solution_in_interval a ↔ a < 1 :=
by sorry

end range_of_a_for_inequality_l189_189784


namespace total_daisies_l189_189344

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l189_189344


namespace smallest_n_l189_189862

theorem smallest_n (n : ℕ) (h : 23 * n ≡ 789 [MOD 11]) : n = 9 :=
sorry

end smallest_n_l189_189862


namespace largest_element_in_A_inter_B_l189_189785

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2023 }
def B : Set ℕ := { n | ∃ k : ℤ, n = 3 * k + 2 ∧ n > 0 }

theorem largest_element_in_A_inter_B : ∃ x ∈ (A ∩ B), ∀ y ∈ (A ∩ B), y ≤ x ∧ x = 2021 := by
  sorry

end largest_element_in_A_inter_B_l189_189785


namespace find_angle_beta_l189_189427

theorem find_angle_beta (α β : ℝ)
  (h1 : (π / 2) < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19)
  (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := 
sorry

end find_angle_beta_l189_189427


namespace simplify_expression_l189_189075

variables (x y : ℝ)

theorem simplify_expression :
  (3 * x)^4 + (4 * x) * (x^3) + (5 * y)^2 = 85 * x^4 + 25 * y^2 :=
by
  sorry

end simplify_expression_l189_189075


namespace admission_price_for_adults_l189_189005

-- Constants and assumptions
def children_ticket_price : ℕ := 25
def total_persons : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80

-- Definitions based on the conditions
def adults_attended : ℕ := total_persons - children_attended
def total_amount_from_children : ℕ := children_attended * children_ticket_price
def total_amount_from_adults (A : ℕ) : ℕ := total_collected_cents - total_amount_from_children
def adult_ticket_price := (total_collected_cents - total_amount_from_children) / adults_attended

-- Theorem statement to be proved
theorem admission_price_for_adults : adult_ticket_price = 60 := by
  sorry

end admission_price_for_adults_l189_189005


namespace not_monotonic_in_interval_l189_189085

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x - 5

theorem not_monotonic_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f a x ≠ (1/3) * x^3 - x^2 + a * x - 5) → a ≥ 1 ∨ a ≤ -3 :=
sorry

end not_monotonic_in_interval_l189_189085


namespace stock_index_approximation_l189_189246

noncomputable def stock_index_after_days (initial_index : ℝ) (daily_increase : ℝ) (days : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ (days - 1)

theorem stock_index_approximation :
  let initial_index := 2
  let daily_increase := 0.02
  let days := 100
  abs (stock_index_after_days initial_index daily_increase days - 2.041) < 0.001 :=
by
  sorry

end stock_index_approximation_l189_189246


namespace middle_part_l189_189933

theorem middle_part (x : ℝ) (h : 2 * x + (2 / 3) * x + (2 / 9) * x = 120) : 
  (2 / 3) * x = 27.6 :=
by
  -- Assuming the given conditions
  sorry

end middle_part_l189_189933


namespace unique_solution_positive_integers_l189_189287

theorem unique_solution_positive_integers :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ ∃ k m : ℤ, a^3 + 6 * a * b + 1 = k^3 ∧ b^3 + 6 * a * b + 1 = m^3) → (a = 1 ∧ b = 1) :=
by
  -- Proof goes here
  sorry

end unique_solution_positive_integers_l189_189287


namespace eval_oplus_otimes_l189_189775

-- Define the operations ⊕ and ⊗
def my_oplus (a b : ℕ) := a + b + 1
def my_otimes (a b : ℕ) := a * b - 1

-- Statement of the proof problem
theorem eval_oplus_otimes : my_oplus (my_oplus 5 7) (my_otimes 2 4) = 21 :=
by
  sorry

end eval_oplus_otimes_l189_189775


namespace final_value_A_eq_B_pow_N_l189_189528

-- Definitions of conditions
def compute_A (A B : ℕ) (N : ℕ) : ℕ :=
    if N ≤ 0 then 
        1 
    else 
        let rec compute_loop (A' B' N' : ℕ) : ℕ :=
            if N' = 0 then A' 
            else 
                let B'' := B' * B'
                let N'' := N' / 2
                let A'' := if N' % 2 = 1 then A' * B' else A'
                compute_loop A'' B'' N'' 
        compute_loop A B N

-- Theorem statement
theorem final_value_A_eq_B_pow_N (A B N : ℕ) : compute_A A B N = B ^ N :=
    sorry

end final_value_A_eq_B_pow_N_l189_189528


namespace billion_to_scientific_l189_189124
noncomputable def scientific_notation_of_billion (n : ℝ) : ℝ := n * 10^9
theorem billion_to_scientific (a : ℝ) : scientific_notation_of_billion a = 1.48056 * 10^11 :=
by sorry

end billion_to_scientific_l189_189124


namespace simplify_and_evaluate_expression_l189_189839

theorem simplify_and_evaluate_expression (m n : ℤ) (h_m : m = -1) (h_n : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l189_189839


namespace rational_reciprocal_pow_2014_l189_189119

theorem rational_reciprocal_pow_2014 (a : ℚ) (h : a = 1 / a) : a ^ 2014 = 1 := by
  sorry

end rational_reciprocal_pow_2014_l189_189119


namespace prove_nabla_squared_l189_189385

theorem prove_nabla_squared:
  ∃ (odot nabla : ℕ), odot < 20 ∧ nabla < 20 ∧ odot ≠ nabla ∧
  (nabla * nabla * odot = nabla) ∧ (nabla * nabla = 64) :=
by
  sorry

end prove_nabla_squared_l189_189385


namespace valid_k_range_l189_189783

noncomputable def fx (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + k * x + k + 3

theorem valid_k_range:
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → fx k x ≥ 0) ↔ (k ≥ -3 / 13) :=
by
  sorry

end valid_k_range_l189_189783


namespace min_soldiers_to_add_l189_189619

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189619


namespace hemisphere_containers_needed_l189_189270

theorem hemisphere_containers_needed 
  (total_volume : ℕ) (volume_per_hemisphere : ℕ) 
  (h₁ : total_volume = 11780) 
  (h₂ : volume_per_hemisphere = 4) : 
  total_volume / volume_per_hemisphere = 2945 := 
by
  sorry

end hemisphere_containers_needed_l189_189270


namespace problem_l189_189220

def f (x a : ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem problem (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : f 1 a = 4 :=
sorry

end problem_l189_189220


namespace johns_minutes_billed_l189_189906

theorem johns_minutes_billed 
  (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 5) (h2 : cost_per_minute = 0.25) (h3 : total_bill = 12.02) :
  ⌊(total_bill - monthly_fee) / cost_per_minute⌋ = 28 :=
by
  sorry

end johns_minutes_billed_l189_189906


namespace min_soldiers_to_add_l189_189617

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189617


namespace max_integer_is_twelve_l189_189702

theorem max_integer_is_twelve
  (a b c d e : ℕ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : (a + b + c + d + e) / 5 = 9)
  (h6 : ((a - 9)^2 + (b - 9)^2 + (c - 9)^2 + (d - 9)^2 + (e - 9)^2) / 5 = 4) :
  e = 12 := sorry

end max_integer_is_twelve_l189_189702


namespace prob_green_is_correct_l189_189757

-- Define the probability of picking any container
def prob_pick_container : ℚ := 1 / 4

-- Define the probability of drawing a green ball from each container
def prob_green_A : ℚ := 6 / 10
def prob_green_B : ℚ := 3 / 10
def prob_green_C : ℚ := 3 / 10
def prob_green_D : ℚ := 5 / 10

-- Define the individual probabilities for a green ball, accounting for container selection
def prob_green_given_A : ℚ := prob_pick_container * prob_green_A
def prob_green_given_B : ℚ := prob_pick_container * prob_green_B
def prob_green_given_C : ℚ := prob_pick_container * prob_green_C
def prob_green_given_D : ℚ := prob_pick_container * prob_green_D

-- Calculate the total probability of selecting a green ball
def prob_green_total : ℚ := prob_green_given_A + prob_green_given_B + prob_green_given_C + prob_green_given_D

-- Theorem statement: The probability of selecting a green ball is 17/40
theorem prob_green_is_correct : prob_green_total = 17 / 40 :=
by
  -- Proof will be provided here.
  sorry

end prob_green_is_correct_l189_189757


namespace find_angle_phi_l189_189000

-- Definitions for the conditions given in the problem
def folded_paper_angle (φ : ℝ) : Prop := 0 < φ ∧ φ < 90

def angle_XOY := 144

-- The main statement to be proven
theorem find_angle_phi (φ : ℝ) (h1 : folded_paper_angle φ) : φ = 81 :=
sorry

end find_angle_phi_l189_189000


namespace maximize_angle_l189_189050

structure Point where
  x : ℝ
  y : ℝ

def A (a : ℝ) : Point := ⟨0, a⟩
def B (b : ℝ) : Point := ⟨0, b⟩

theorem maximize_angle
  (a b : ℝ)
  (h : a > b)
  (h₁ : b > 0)
  : ∃ (C : Point), C = ⟨Real.sqrt (a * b), 0⟩ :=
sorry

end maximize_angle_l189_189050


namespace ones_digit_of_p_is_3_l189_189564

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l189_189564


namespace shorter_diagonal_of_rhombus_l189_189984

variable (d s : ℝ)  -- d for shorter diagonal, s for the side length of the rhombus

theorem shorter_diagonal_of_rhombus 
  (h1 : ∀ (s : ℝ), s = 39)
  (h2 : ∀ (a b : ℝ), a^2 + b^2 = s^2)
  (h3 : ∀ (d a : ℝ), (d / 2)^2 + a^2 = 39^2)
  (h4 : 72 / 2 = 36)
  : d = 30 := 
by 
  sorry

end shorter_diagonal_of_rhombus_l189_189984


namespace find_lunch_break_duration_l189_189674

def lunch_break_duration : ℝ → ℝ → ℝ → ℝ
  | s, a, L => L

theorem find_lunch_break_duration (s a L : ℝ) :
  (8 - L) * (s + a) = 0.6 ∧ (6.4 - L) * a = 0.28 ∧ (9.6 - L) * s = 0.12 →
  lunch_break_duration s a L = 1 :=
  by
    sorry

end find_lunch_break_duration_l189_189674


namespace exists_prime_and_cube_root_l189_189836

theorem exists_prime_and_cube_root (n : ℕ) (hn : 0 < n) :
  ∃ (p m : ℕ), p.Prime ∧ p % 6 = 5 ∧ ¬p ∣ n ∧ n ≡ m^3 [MOD p] :=
sorry

end exists_prime_and_cube_root_l189_189836


namespace rectangle_area_is_correct_l189_189736

noncomputable def inscribed_rectangle_area (r : ℝ) (l_to_w_ratio : ℝ) : ℝ :=
  let width := 2 * r
  let length := l_to_w_ratio * width
  length * width

theorem rectangle_area_is_correct :
  inscribed_rectangle_area 7 3 = 588 :=
  by
    -- The proof goes here
    sorry

end rectangle_area_is_correct_l189_189736


namespace angle_B_value_l189_189320

noncomputable def degree_a (A : ℝ) : Prop := A = 30 ∨ A = 60

noncomputable def degree_b (A B : ℝ) : Prop := B = 3 * A - 60

theorem angle_B_value (A B : ℝ) 
  (h1 : B = 3 * A - 60)
  (h2 : A = 30 ∨ A = 60) :
  B = 30 ∨ B = 120 :=
by
  sorry

end angle_B_value_l189_189320


namespace usual_time_is_60_l189_189725

variable (S T T' D : ℝ)

-- Defining the conditions
axiom condition1 : T' = T + 12
axiom condition2 : D = S * T
axiom condition3 : D = (5 / 6) * S * T'

-- The theorem to prove
theorem usual_time_is_60 (S T T' D : ℝ) 
  (h1 : T' = T + 12)
  (h2 : D = S * T)
  (h3 : D = (5 / 6) * S * T') : T = 60 := 
sorry

end usual_time_is_60_l189_189725


namespace gray_eyed_brunettes_l189_189335

-- Given conditions
def total_students : ℕ := 60
def brunettes : ℕ := 35
def green_eyed_blondes : ℕ := 20
def gray_eyed_total : ℕ := 25

-- Conclude that the number of gray-eyed brunettes is 20
theorem gray_eyed_brunettes :
    (gray_eyed_total - (total_students - brunettes - green_eyed_blondes)) = 20 := by
    sorry

end gray_eyed_brunettes_l189_189335


namespace delta_four_equal_zero_l189_189765

-- Define the sequence u_n
def u (n : ℕ) : ℤ := n^3 + n

-- Define the ∆ operator
def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0   => u
  | k+1 => delta1 (delta k u)

-- The theorem statement
theorem delta_four_equal_zero (n : ℕ) : delta 4 u n = 0 :=
by sorry

end delta_four_equal_zero_l189_189765


namespace three_digit_numbers_l189_189019

theorem three_digit_numbers (n : ℕ) (a b c : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n = 100 * a + 10 * b + c)
  (h3 : b^2 = a * c)
  (h4 : (10 * b + c) % 4 = 0) :
  n = 124 ∨ n = 248 ∨ n = 444 ∨ n = 964 ∨ n = 888 :=
sorry

end three_digit_numbers_l189_189019


namespace total_books_l189_189978

def sam_books : ℕ := 110
def joan_books : ℕ := 102

theorem total_books : sam_books + joan_books = 212 := by
  sorry

end total_books_l189_189978


namespace evaluate_expression_l189_189211

variable (x y : ℚ)

theorem evaluate_expression 
  (hx : x = 2) 
  (hy : y = -1 / 5) : 
  (2 * x - 3)^2 - (x + 2 * y) * (x - 2 * y) - 3 * y^2 + 3 = 1 / 25 :=
by
  sorry

end evaluate_expression_l189_189211


namespace geom_seq_problem_l189_189186

variable {a : ℕ → ℝ}  -- positive geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a n = a 0 * r^n

theorem geom_seq_problem
  (h_geom : geom_seq a)
  (cond : a 0 * a 4 + 2 * a 2 * a 4 + a 2 * a 6 = 25) :
  a 2 + a 4 = 5 :=
sorry

end geom_seq_problem_l189_189186


namespace vehicle_value_last_year_l189_189857

variable (v_this_year v_last_year : ℝ)

theorem vehicle_value_last_year:
  v_this_year = 16000 ∧ v_this_year = 0.8 * v_last_year → v_last_year = 20000 :=
by
  -- Proof steps can be added here, but replaced with sorry as per instructions.
  sorry

end vehicle_value_last_year_l189_189857


namespace probability_integer_solution_l189_189180

theorem probability_integer_solution (a : ℤ) (h₀ : a ≥ 0) (h_max : a ≤ 6) :
    let x := (a + 4) / 2 in
    ∃ s : finset ℤ, s = {0, 1, 2, 3, 4, 5, 6} ∧
    let valid_a := s.filter (λ a, ∃ x : ℤ, x = (a + 4) / 2) in
    valid_a.card / s.card = 3 / 7 :=
by
  sorry

end probability_integer_solution_l189_189180


namespace rooster_count_l189_189228

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l189_189228


namespace harry_did_not_get_an_A_l189_189284

theorem harry_did_not_get_an_A
  (emily_Imp_frank : Prop)
  (frank_Imp_gina : Prop)
  (gina_Imp_harry : Prop)
  (exactly_one_did_not_get_an_A : ¬ (emily_Imp_frank ∧ frank_Imp_gina ∧ gina_Imp_harry)) :
  ¬ harry_Imp_gina :=
  sorry

end harry_did_not_get_an_A_l189_189284


namespace find_ab_l189_189443

noncomputable def validate_ab : Prop :=
  let n : ℕ := 8
  let a : ℕ := n^2 - 1
  let b : ℕ := n
  a = 63 ∧ b = 8

theorem find_ab : validate_ab :=
by
  sorry

end find_ab_l189_189443


namespace function_increasing_interval_l189_189014

theorem function_increasing_interval :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi),
  (2 * Real.sin ((Real.pi / 6) - 2 * x) : ℝ)
  ≤ 2 * Real.sin ((Real.pi / 6) - 2 * x + 1)) ↔ (x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
sorry

end function_increasing_interval_l189_189014


namespace bargain_range_l189_189955

theorem bargain_range (cost_price lowest_cp highest_cp : ℝ)
  (h_lowest : lowest_cp = 50)
  (h_highest : highest_cp = 200 / 3)
  (h_marked_at : cost_price = 100)
  (h_lowest_markup : lowest_cp * 2 = cost_price)
  (h_highest_markup : highest_cp * 1.5 = cost_price)
  (profit_margin : ∀ (cp : ℝ), (cp * 1.2 ≥ cp)) : 
  (60 ≤ cost_price * 1.2 ∧ cost_price * 1.2 ≤ 80) :=
by
  sorry

end bargain_range_l189_189955


namespace distinct_cubes_meet_condition_l189_189155

theorem distinct_cubes_meet_condition :
  ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + b + c + d + e + f = 60) ∧
    ∃ (k : ℕ), 
        ((a = k) ∧ (b = k) ∧ (c = k) ∧ (d = k) ∧ (e = k) ∧ (f = k)) ∧
        -- Number of distinct ways
        (∃ (num_ways : ℕ), num_ways = 84) :=
sorry

end distinct_cubes_meet_condition_l189_189155


namespace probability_of_C_l189_189876

def region_prob_A := (1 : ℚ) / 4
def region_prob_B := (1 : ℚ) / 3
def region_prob_D := (1 : ℚ) / 6

theorem probability_of_C :
  (region_prob_A + region_prob_B + region_prob_D + (1 : ℚ) / 4) = 1 :=
by
  sorry

end probability_of_C_l189_189876


namespace probability_at_least_6_heads_in_10_flips_l189_189251

theorem probability_at_least_6_heads_in_10_flips : 
  let total_outcomes := 1024 in 
  let favorable_outcomes := 15 in 
  (favorable_outcomes / total_outcomes : ℚ) = 15 / 1024 :=
by
  sorry

end probability_at_least_6_heads_in_10_flips_l189_189251


namespace lines_through_point_l189_189866

theorem lines_through_point (k : ℝ) : ∀ x y : ℝ, (y = k * (x - 1)) ↔ (x = 1 ∧ y = 0) ∨ (x ≠ 1 ∧ y / (x - 1) = k) :=
by
  sorry

end lines_through_point_l189_189866


namespace xyz_value_l189_189965

noncomputable def find_xyz (x y z : ℝ) 
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) : ℝ :=
  if (x * y * z = 31 / 3) then 31 / 3 else 0  -- This should hold with the given conditions

theorem xyz_value (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) :
  find_xyz x y z h₁ h₂ h₃ = 31 / 3 :=
by 
  sorry  -- The proof should demonstrate that find_xyz equals 31 / 3 given the conditions

end xyz_value_l189_189965


namespace sarah_correct_answer_percentage_l189_189372

theorem sarah_correct_answer_percentage
  (q1 q2 q3 : ℕ)   -- Number of questions in the first, second, and third tests.
  (p1 p2 p3 : ℕ → ℝ)   -- Percentages of questions Sarah got right in the first, second, and third tests.
  (m : ℕ)   -- Number of calculation mistakes:
  (h_q1 : q1 = 30) (h_q2 : q2 = 20) (h_q3 : q3 = 50)
  (h_p1 : p1 q1 = 0.85) (h_p2 : p2 q2 = 0.75) (h_p3 : p3 q3 = 0.90)
  (h_m : m = 3) :
  ∃ pct_correct : ℝ, pct_correct = 83 :=
by
  sorry

end sarah_correct_answer_percentage_l189_189372


namespace total_cost_of_trip_l189_189202

def totalDistance (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def gallonsUsed (distance miles_per_gallon : ℕ) : ℕ :=
  distance / miles_per_gallon

def totalCost (gallons : ℕ) (cost_per_gallon : ℕ) : ℕ :=
  gallons * cost_per_gallon

theorem total_cost_of_trip :
  (totalDistance 10 6 5 9 = 30) →
  (gallonsUsed 30 15 = 2) →
  totalCost 2 35 = 700 :=
by
  sorry

end total_cost_of_trip_l189_189202


namespace y1_lt_y2_l189_189916

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end y1_lt_y2_l189_189916


namespace identify_triangle_centers_l189_189966

variable (P : Fin 7 → Type)
variable (I O H L G N K : Type)
variable (P1 P2 P3 P4 P5 P6 P7 : Type)
variable (cond : (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H))

theorem identify_triangle_centers :
  (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H) :=
by sorry

end identify_triangle_centers_l189_189966


namespace proposition_false_at_6_l189_189263

variable (P : ℕ → Prop)

theorem proposition_false_at_6 (h1 : ∀ k : ℕ, 0 < k → P k → P (k + 1)) (h2 : ¬P 7): ¬P 6 :=
by
  sorry

end proposition_false_at_6_l189_189263


namespace ones_digit_of_prime_p_l189_189572

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l189_189572


namespace apples_pie_calculation_l189_189053

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end apples_pie_calculation_l189_189053


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l189_189254

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l189_189254


namespace ones_digit_of_prime_sequence_l189_189569

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l189_189569


namespace incorrect_expression_l189_189101

theorem incorrect_expression : ¬ (5 = (Real.sqrt (-5))^2) :=
by
  sorry

end incorrect_expression_l189_189101


namespace fraction_value_condition_l189_189313

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end fraction_value_condition_l189_189313


namespace fraction_of_area_l189_189369

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let base := (C.1 - A.1).abs
  let height := B.2
  (base * height) / 2

theorem fraction_of_area {A B C X Y Z: (ℝ × ℝ)}
  (hA : A = (2, 0)) (hB : B = (8, 12)) (hC : C = (14, 0))
  (hX : X = (6, 0)) (hY : Y = (8, 4)) (hZ : Z = (10, 0)):
  (area_of_triangle X Y Z) / (area_of_triangle A B C) = 1 / 9 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end fraction_of_area_l189_189369


namespace poly_expansion_l189_189424

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion_l189_189424


namespace exists_number_between_70_and_80_with_gcd_10_l189_189510

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l189_189510


namespace perpendicular_condition_l189_189437

def line := Type
def plane := Type

variables {α : plane} {a b : line}

-- Conditions: define parallelism and perpendicularity
def parallel (a : line) (α : plane) : Prop := sorry
def perpendicular (a : line) (α : plane) : Prop := sorry
def perpendicular_lines (a b : line) : Prop := sorry

-- Given Hypotheses
variable (h1 : parallel a α)
variable (h2 : perpendicular b α)

-- Statement to prove
theorem perpendicular_condition (h1 : parallel a α) (h2 : perpendicular b α) :
  (perpendicular_lines b a) ∧ (¬ (perpendicular_lines b a → perpendicular b α)) := 
sorry

end perpendicular_condition_l189_189437


namespace total_daisies_l189_189347

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l189_189347


namespace minimum_soldiers_to_add_l189_189628

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189628


namespace sum_of_decimals_l189_189879

theorem sum_of_decimals :
  5.467 + 2.349 + 3.785 = 11.751 :=
sorry

end sum_of_decimals_l189_189879


namespace drawing_blue_ball_probability_l189_189468

noncomputable def probability_of_blue_ball : ℚ :=
  let total_balls := 10
  let blue_balls := 6
  blue_balls / total_balls

theorem drawing_blue_ball_probability :
  probability_of_blue_ball = 3 / 5 :=
by
  sorry -- Proof is omitted as per instructions.

end drawing_blue_ball_probability_l189_189468


namespace dave_apps_left_l189_189141

def initial_apps : ℕ := 24
def initial_files : ℕ := 9
def files_left : ℕ := 5
def apps_left (files_left: ℕ) : ℕ := files_left + 7

theorem dave_apps_left :
  apps_left files_left = 12 :=
by
  sorry

end dave_apps_left_l189_189141


namespace total_people_veg_l189_189184

-- Definitions based on the conditions
def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 6

-- The statement we need to prove
theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 19 :=
by
  sorry

end total_people_veg_l189_189184


namespace dividend_from_tonys_stock_l189_189536

theorem dividend_from_tonys_stock (investment price_per_share total_income : ℝ) 
  (h1 : investment = 3200) (h2 : price_per_share = 85) (h3 : total_income = 250) : 
  (total_income / (investment / price_per_share)) = 6.76 :=
by 
  sorry

end dividend_from_tonys_stock_l189_189536


namespace curve_is_parabola_l189_189900

theorem curve_is_parabola (r θ : ℝ) : (r = 1 / (1 - Real.cos θ)) ↔ ∃ x y : ℝ, y^2 = 2 * x + 1 :=
by 
  sorry

end curve_is_parabola_l189_189900


namespace number_of_violas_l189_189735

theorem number_of_violas (V : ℕ) 
  (cellos : ℕ := 800) 
  (pairs : ℕ := 70) 
  (probability : ℝ := 0.00014583333333333335) 
  (h : probability = pairs / (cellos * V)) : V = 600 :=
by
  sorry

end number_of_violas_l189_189735


namespace algebraic_expression_value_l189_189330

theorem algebraic_expression_value
  (a b x y : ℤ)
  (h1 : x = a)
  (h2 : y = b)
  (h3 : x - 2 * y = 7) :
  -a + 2 * b + 1 = -6 :=
by
  -- the proof steps are omitted as instructed
  sorry

end algebraic_expression_value_l189_189330


namespace system_of_equations_solution_l189_189501

theorem system_of_equations_solution (x y z : ℝ) :
  (x * y + x * z = 8 - x^2) →
  (x * y + y * z = 12 - y^2) →
  (y * z + z * x = -4 - z^2) →
  (x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1) :=
by
  sorry

end system_of_equations_solution_l189_189501


namespace distance_between_red_lights_l189_189212

def position_of_nth_red (n : ℕ) : ℕ :=
  7 * (n - 1) / 3 + n

def in_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_red_lights :
  in_feet ((position_of_nth_red 30 - position_of_nth_red 5) * 8) = 41 :=
by
  sorry

end distance_between_red_lights_l189_189212


namespace correct_statement_C_l189_189562

-- Define the function
def linear_function (x : ℝ) : ℝ := -3 * x + 1

-- Define the condition for statement C
def statement_C (x : ℝ) : Prop := x > 1 / 3 → linear_function x < 0

-- The theorem to be proved
theorem correct_statement_C : ∀ x : ℝ, statement_C x := by
  sorry

end correct_statement_C_l189_189562


namespace family_gathering_l189_189336

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end family_gathering_l189_189336


namespace smallest_number_divisible_by_conditions_l189_189106

theorem smallest_number_divisible_by_conditions:
  ∃ n : ℕ, (∀ d ∈ [8, 12, 22, 24], d ∣ (n - 12)) ∧ (n = 252) :=
by
  sorry

end smallest_number_divisible_by_conditions_l189_189106


namespace area_of_trapezoid_l189_189207

-- Define the parameters as given in the problem
def PQ : ℝ := 40
def RS : ℝ := 25
def h : ℝ := 10
def PR : ℝ := 20

-- Assert the quadrilateral is a trapezoid with bases PQ and RS parallel
def isTrapezoid (PQ RS : ℝ) (h : ℝ) (PR : ℝ) : Prop := true -- this is just a placeholder to state that it's a trapezoid

-- The main statement for the area of the trapezoid
theorem area_of_trapezoid (h : ℝ) (PQ RS : ℝ) (h : ℝ) (PR : ℝ) (is_trapezoid : isTrapezoid PQ RS h PR) : (1/2) * (PQ + RS) * h = 325 :=
by
  sorry

end area_of_trapezoid_l189_189207


namespace find_digits_l189_189024

-- Define the digits range
def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

-- Define the five-digit numbers
def num_abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 111 * c
def num_abbbb (a b : ℕ) : ℕ := 10000 * a + 1111 * b

-- Problem statement
theorem find_digits (a b c : ℕ) (h_da : is_digit a) (h_db : is_digit b) (h_dc : is_digit c) :
  (num_abccc a b c) + 1 = (num_abbbb a b) ↔
  (a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0) :=
sorry

end find_digits_l189_189024


namespace valentina_burger_length_l189_189859

-- Definitions and conditions
def share : ℕ := 6
def total_length (share : ℕ) : ℕ := 2 * share

-- Proof statement
theorem valentina_burger_length : total_length share = 12 := by
  sorry

end valentina_burger_length_l189_189859


namespace angle_half_in_first_quadrant_l189_189777

theorem angle_half_in_first_quadrant (α : ℝ) (hα : 90 < α ∧ α < 180) : 0 < α / 2 ∧ α / 2 < 90 := 
sorry

end angle_half_in_first_quadrant_l189_189777


namespace minimum_soldiers_to_add_l189_189604

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189604


namespace trigonometric_identity_example_l189_189128

theorem trigonometric_identity_example :
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_example_l189_189128


namespace first_day_is_wednesday_l189_189981

theorem first_day_is_wednesday (day22_wednesday : ∀ n, n = 22 → (n = 22 → "Wednesday" = "Wednesday")) :
  ∀ n, n = 1 → (n = 1 → "Wednesday" = "Wednesday") :=
by
  sorry

end first_day_is_wednesday_l189_189981


namespace production_days_l189_189023

-- Definitions of the conditions
variables (n : ℕ) (P : ℕ)
variable (H1 : P = n * 50)
variable (H2 : (P + 60) / (n + 1) = 55)

-- Theorem to prove that n = 1 given the conditions
theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 60) / (n + 1) = 55) : n = 1 :=
by
  sorry

end production_days_l189_189023


namespace probability_of_red_ball_l189_189954

noncomputable def total_balls : Nat := 4 + 2
noncomputable def red_balls : Nat := 2

theorem probability_of_red_ball :
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
sorry

end probability_of_red_ball_l189_189954


namespace cos_pi_minus_alpha_l189_189435

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 7) : Real.cos (Real.pi - α) = - (1 / 7) := by
  sorry

end cos_pi_minus_alpha_l189_189435


namespace negative_remainder_l189_189599

theorem negative_remainder (a : ℤ) (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end negative_remainder_l189_189599


namespace find_k_l189_189309

theorem find_k (k : ℚ) : (∀ x y : ℚ, (x, y) = (2, 1) → 3 * k * x - k = -4 * y - 2) → k = -(6 / 5) :=
by
  intro h
  have key := h 2 1 rfl
  have : 3 * k * 2 - k = -4 * 1 - 2 := key
  linarith

end find_k_l189_189309


namespace product_of_odd_primes_mod_sixteen_l189_189055

-- Define the set of odd primes less than 16
def odd_primes_less_than_sixteen : List ℕ := [3, 5, 7, 11, 13]

-- Define the product of all odd primes less than 16
def N : ℕ := odd_primes_less_than_sixteen.foldl (· * ·) 1

-- Proposition to prove: N ≡ 7 (mod 16)
theorem product_of_odd_primes_mod_sixteen :
  (N % 16) = 7 :=
  sorry

end product_of_odd_primes_mod_sixteen_l189_189055


namespace triangle_ABC_problem_l189_189583

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ := a + b + c

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = 3) 
  (h2 : B = π / 3) 
  (area : ℝ)
  (h3 : (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3) :

  perimeter_of_triangle a b c = 18 ∧ 
  Real.sin (2 * A) = 39 * Real.sqrt 3 / 98 := 
by 
  sorry

end triangle_ABC_problem_l189_189583


namespace no_real_roots_quadratic_l189_189949

theorem no_real_roots_quadratic (k : ℝ) : 
  ∀ (x : ℝ), k * x^2 - 2 * x + 1 / 2 ≠ 0 → k > 2 :=
by 
  intro x h
  have h1 : (-2)^2 - 4 * k * (1/2) < 0 := sorry
  have h2 : 4 - 2 * k < 0 := sorry
  have h3 : 2 < k := sorry
  exact h3

end no_real_roots_quadratic_l189_189949


namespace work_completion_days_l189_189871

open Real

theorem work_completion_days (days_A : ℝ) (days_B : ℝ) (amount_total : ℝ) (amount_C : ℝ) :
  days_A = 6 ∧ days_B = 8 ∧ amount_total = 5000 ∧ amount_C = 625.0000000000002 →
  (1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1)) = 5 / 12 →
  1 / ((1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1))) = 2.4 :=
  sorry

end work_completion_days_l189_189871


namespace minimum_soldiers_to_add_l189_189602

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189602


namespace repeating_decimal_to_fraction_l189_189710

theorem repeating_decimal_to_fraction : ∃ (r : ℚ), r = 0.4 + 0.0036 * (1/(1 - 0.01)) ∧ r = 42 / 55 :=
by
  sorry

end repeating_decimal_to_fraction_l189_189710


namespace john_average_score_change_l189_189816

/-- Given John's scores on his biology exams, calculate the change in his average score after the fourth exam. -/
theorem john_average_score_change :
  let first_three_scores := [84, 88, 95]
  let fourth_score := 92
  let first_average := (84 + 88 + 95) / 3
  let new_average := (84 + 88 + 95 + 92) / 4
  new_average - first_average = 0.75 :=
by
  sorry

end john_average_score_change_l189_189816


namespace correct_subtraction_result_l189_189530

theorem correct_subtraction_result (n : ℕ) (h : 40 / n = 5) : 20 - n = 12 := by
sorry

end correct_subtraction_result_l189_189530


namespace incorrect_independence_test_conclusion_l189_189102

-- Definitions for each condition
def independence_test_principle_of_small_probability (A : Prop) : Prop :=
A  -- Statement A: The independence test is based on the principle of small probability.

def independence_test_conclusion_variability (C : Prop) : Prop :=
C  -- Statement C: Different samples may lead to different conclusions in the independence test.

def independence_test_not_the_only_method (D : Prop) : Prop :=
D  -- Statement D: The independence test is not the only method to determine whether two categorical variables are related.

-- Incorrect statement B
def independence_test_conclusion_always_correct (B : Prop) : Prop :=
B  -- Statement B: The conclusion drawn from the independence test is always correct.

-- Prove that statement B is incorrect given conditions A, C, and D
theorem incorrect_independence_test_conclusion (A B C D : Prop) 
  (hA : independence_test_principle_of_small_probability A)
  (hC : independence_test_conclusion_variability C)
  (hD : independence_test_not_the_only_method D) :
  ¬ independence_test_conclusion_always_correct B :=
sorry

end incorrect_independence_test_conclusion_l189_189102


namespace digit_number_is_203_l189_189741

theorem digit_number_is_203 {A B C : ℕ} (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) :
  100 * A + 10 * B + C = 203 :=
by
  sorry

end digit_number_is_203_l189_189741


namespace soldiers_to_add_l189_189637

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l189_189637


namespace object_distance_traveled_l189_189462

theorem object_distance_traveled
  (t : ℕ) (v_mph : ℝ) (mile_to_feet : ℕ)
  (h_t : t = 2)
  (h_v : v_mph = 68.18181818181819)
  (h_mile : mile_to_feet = 5280) :
  ∃ d : ℝ, d = 200 :=
by {
  sorry
}

end object_distance_traveled_l189_189462


namespace parabola_solution_l189_189279

noncomputable def parabola_coefficients (a b c : ℝ) : Prop :=
  (6 : ℝ) = a * (5 : ℝ)^2 + b * (5 : ℝ) + c ∧
  0 = a * (3 : ℝ)^2 + b * (3 : ℝ) + c

theorem parabola_solution :
  ∃ (a b c : ℝ), parabola_coefficients a b c ∧ (a + b + c = 6) :=
by {
  -- definitions and constraints based on problem conditions
  sorry
}

end parabola_solution_l189_189279


namespace integral_2x_plus_3_squared_l189_189901

open Real

-- Define the function to be integrated
def f (x : ℝ) := (2 * x + 3) ^ 2

-- State the theorem for the indefinite integral
theorem integral_2x_plus_3_squared :
  ∃ C : ℝ, ∫ x, f x = (1 / 6) * (2 * x + 3) ^ 3 + C :=
by
  sorry

end integral_2x_plus_3_squared_l189_189901


namespace jason_borrowed_amount_l189_189348

theorem jason_borrowed_amount (hours cycles value_per_cycle remaining_hrs remaining_value total_value: ℕ) : 
  hours = 39 → cycles = (hours / 7) → value_per_cycle = 28 → remaining_hrs = (hours % 7) →
  remaining_value = (1 + 2 + 3 + 4) →
  total_value = (cycles * value_per_cycle + remaining_value) →
  total_value = 150 := 
by {
  sorry
}

end jason_borrowed_amount_l189_189348


namespace subtraction_base_8_correct_l189_189884

def sub_in_base_8 (a b : Nat) : Nat := sorry

theorem subtraction_base_8_correct : sub_in_base_8 (sub_in_base_8 0o123 0o51) 0o15 = 0o25 :=
sorry

end subtraction_base_8_correct_l189_189884


namespace avg_tickets_male_l189_189238

theorem avg_tickets_male (M F : ℕ) (w : ℕ) 
  (h1 : M / F = 1 / 2) 
  (h2 : (M + F) * 66 = M * w + F * 70) 
  : w = 58 := 
sorry

end avg_tickets_male_l189_189238


namespace farmer_field_l189_189116

theorem farmer_field (m : ℤ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 :=
by
  sorry

end farmer_field_l189_189116


namespace smallest_multiplier_to_perfect_square_l189_189269

-- Definitions for the conditions
def y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6

-- The theorem statement itself
theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, (∀ m : ℕ, (y * m = k) → (∃ n : ℕ, (k * y) = n^2)) :=
by
  let y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6
  let smallest_k := 70
  have h : y = 2^33 * 3^20 * 5^3 * 7^5 := by sorry
  use smallest_k
  intros m hm
  use (2^17 * 3^10 * 5 * 7)
  sorry

end smallest_multiplier_to_perfect_square_l189_189269


namespace batsman_average_increase_l189_189732

theorem batsman_average_increase (A : ℝ) (X : ℝ) (runs_11th_inning : ℝ) (average_11th_inning : ℝ) 
  (h_runs_11th_inning : runs_11th_inning = 85) 
  (h_average_11th_inning : average_11th_inning = 35) 
  (h_eq : (10 * A + runs_11th_inning) / 11 = average_11th_inning) :
  X = 5 := 
by 
  sorry

end batsman_average_increase_l189_189732


namespace Jackson_money_is_125_l189_189660

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l189_189660


namespace max_m_n_squared_l189_189759

theorem max_m_n_squared (m n : ℤ) 
  (hmn : 1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981)
  (h_eq : (n^2 - m*n - m^2)^2 = 1) : 
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_squared_l189_189759


namespace min_value_of_expression_l189_189911

theorem min_value_of_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n n = (4/3) * (a_n n - 1)) :
  ∃ (n : ℕ), (4^(n - 2) + 1) * (16 / a_n n + 1) = 4 :=
by
  sorry

end min_value_of_expression_l189_189911


namespace find_C_work_rate_l189_189237

-- Conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 6

-- Combined work rate of A and B
def AB_work_rate := A_work_rate + B_work_rate

-- Total work rate when C is assisting, completing in 2 days
def total_work_rate_of_ABC := 1 / 2

theorem find_C_work_rate : ∃ c : ℕ, (AB_work_rate + 1 / c = total_work_rate_of_ABC) ∧ c = 12 :=
by
  -- To complete the proof, we solve the equation for c
  sorry

end find_C_work_rate_l189_189237


namespace trajectory_of_P_l189_189030

open Real

-- Definitions of points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Definition of the condition on moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F2 - dist P F1) = 4

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1 ∧ x ≤ -2

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, satisfies_condition P → ∃ x y : ℝ, P = (x, y) ∧ hyperbola_equation x y :=
by
  sorry

end trajectory_of_P_l189_189030


namespace increase_in_average_commission_l189_189063

theorem increase_in_average_commission :
  ∀ (new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 : ℕ),
    new_avg = 400 → 
    n1 = 6 → 
    n2 = n1 - 1 → 
    big_sale = 1300 →
    total_earnings = new_avg * n1 →
    commission = total_earnings - big_sale →
    old_avg = commission / n2 →
    new_avg - old_avg = 180 :=
by 
  intros new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end increase_in_average_commission_l189_189063


namespace solve_x_in_equation_l189_189678

theorem solve_x_in_equation (a b x : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : x ≠ 0) : 
  (b ≠ 0 ∧ (1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) → x = a^2 - b^2) ∧ 
  (b = 0 ∧ a ≠ 0 ∧ (1 / a + a / x = 1 / a + a / x) → x ≠ 0) := 
by
  sorry

end solve_x_in_equation_l189_189678


namespace f_2014_odd_f_2014_not_even_l189_189756

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 1 / x
| (n + 1), x => 1 / (x + f n x)

theorem f_2014_odd :
  ∀ x : ℝ, f 2014 x = - f 2014 (-x) :=
sorry

theorem f_2014_not_even :
  ∃ x : ℝ, f 2014 x ≠ f 2014 (-x) :=
sorry

end f_2014_odd_f_2014_not_even_l189_189756


namespace inverse_proportion_inequality_l189_189920

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l189_189920


namespace boys_playing_both_sports_l189_189533

theorem boys_playing_both_sports : 
  ∀ (total boys basketball football neither both : ℕ), 
  total = 22 → boys = 22 → basketball = 13 → football = 15 → neither = 3 → 
  boys = basketball + football - both + neither → 
  both = 9 :=
by
  intros total boys basketball football neither both
  intros h_total h_boys h_basketball h_football h_neither h_formula
  sorry

end boys_playing_both_sports_l189_189533


namespace abigail_written_words_l189_189878

theorem abigail_written_words (total_words : ℕ) (typing_rate_per_hour : ℕ) (remaining_minutes : ℕ) :
  total_words = 1000 →
  typing_rate_per_hour = 600 →
  remaining_minutes = 80 →
  ∃ words_written : ℕ, words_written = 200 :=
by
  assume h_total_words ht_rate_per_hour h_remaining_minutes
  sorry

end abigail_written_words_l189_189878


namespace evaluate_g_5_times_l189_189353

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x + 2 else 3 * x + 1

theorem evaluate_g_5_times : g (g (g (g (g 1)))) = 12 := by
  sorry


end evaluate_g_5_times_l189_189353


namespace min_height_regular_quadrilateral_pyramid_l189_189107

theorem min_height_regular_quadrilateral_pyramid (r : ℝ) (a : ℝ) (h : 2 * r < a / 2) : 
  ∃ x : ℝ, (0 < x) ∧ (∃ V : ℝ, ∀ x' : ℝ, V = (a^2 * x) / 3 ∧ (∀ x' ≠ x, V < (a^2 * x') / 3)) ∧ x = (r * (5 + Real.sqrt 17)) / 2 :=
sorry

end min_height_regular_quadrilateral_pyramid_l189_189107


namespace notebook_pen_cost_correct_l189_189082

noncomputable def notebook_pen_cost : Prop :=
  ∃ (x y : ℝ), 
  3 * x + 2 * y = 7.40 ∧ 
  2 * x + 5 * y = 9.75 ∧ 
  (x + 3 * y) = 5.53

theorem notebook_pen_cost_correct : notebook_pen_cost :=
sorry

end notebook_pen_cost_correct_l189_189082


namespace polar_coordinates_intersection_points_l189_189049

/-- Convert given parametric equations and circle equation to polar form,
and prove the intersection points in polar coordinates --/
theorem polar_coordinates_intersection_points :
  let line_parametric (t : ℝ) := (2 - real.sqrt 3 * t, t)
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let line_cartesian (x y : ℝ) := x + real.sqrt 3 * y - 2 = 0
  let polar_line (rho theta : ℝ) := rho * real.cos (theta - real.pi / 3) = 1
  let polar_circle (rho : ℝ) := rho = 2 in
  ∃ theta_1 theta_2,
    theta_1 ∈ [0, 2 * real.pi) ∧ theta_2 ∈ [0, 2 * real.pi) ∧
    ((2, theta_1) ∨ (2, theta_2)) ∧
    (polar_line 2 theta_1 ∧ polar_line 2 theta_2) :=
  sorry

end polar_coordinates_intersection_points_l189_189049


namespace minimum_soldiers_to_add_l189_189643

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189643


namespace part1_part2_l189_189580

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l189_189580


namespace term_in_AP_is_zero_l189_189808

theorem term_in_AP_is_zero (a d : ℤ) 
  (h : (a + 4 * d) + (a + 20 * d) = (a + 7 * d) + (a + 14 * d) + (a + 12 * d)) :
  a + (-9) * d = 0 :=
by
  sorry

end term_in_AP_is_zero_l189_189808


namespace possible_measures_of_angle_X_l189_189088

theorem possible_measures_of_angle_X :
  ∃ (n : ℕ), n = 17 ∧ ∀ (X Y : ℕ), 
    (X > 0) → 
    (Y > 0) → 
    (∃ k : ℕ, k ≥ 1 ∧ X = k * Y) → 
    X + Y = 180 → 
    ∃ d : ℕ, d ∈ {d | d ∣ 180 } ∧ d ≥ 2 :=
by
  sorry

end possible_measures_of_angle_X_l189_189088


namespace exists_number_between_70_and_80_with_gcd_10_l189_189509

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l189_189509


namespace seeds_per_packet_l189_189278

theorem seeds_per_packet (total_seedlings packets : ℕ) (h1 : total_seedlings = 420) (h2 : packets = 60) : total_seedlings / packets = 7 :=
by 
  sorry

end seeds_per_packet_l189_189278


namespace apples_left_l189_189113

theorem apples_left (initial_apples : ℕ) (difference_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 46) 
  (h2 : difference_apples = 32) 
  (h3 : final_apples = initial_apples - difference_apples) : 
  final_apples = 14 := 
by
  rw [h1, h2] at h3
  exact h3

end apples_left_l189_189113


namespace part1_part2_l189_189581

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l189_189581


namespace sector_area_l189_189841

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 40) : (θ / 360) * π * r^2 = 16 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_l189_189841


namespace smallest_cut_length_l189_189746

theorem smallest_cut_length (x : ℕ) (h₁ : 9 ≥ x) (h₂ : 12 ≥ x) (h₃ : 15 ≥ x)
  (h₄ : x ≥ 6) (h₅ : x ≥ 12) (h₆ : x ≥ 18) : x = 6 :=
by
  sorry

end smallest_cut_length_l189_189746


namespace last_card_in_box_l189_189093

-- Define the zigzag pattern
def card_position (n : Nat) : Nat :=
  let cycle_pos := n % 12
  if cycle_pos = 0 then
    12
  else
    cycle_pos

def box_for_card (pos : Nat) : Nat :=
  if pos ≤ 7 then
    pos
  else
    14 - pos

theorem last_card_in_box : box_for_card (card_position 2015) = 3 := by
  sorry

end last_card_in_box_l189_189093


namespace lcm_36_100_l189_189306

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l189_189306


namespace max_pens_min_pens_l189_189719

def pen_prices : List ℕ := [2, 3, 4]
def total_money : ℕ := 31

/-- Given the conditions of the problem, prove the maximum number of pens -/
theorem max_pens  (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 14 := by
  sorry

/-- Given the conditions of the problem, prove the minimum number of pens -/
theorem min_pens (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 9 := by
  sorry

end max_pens_min_pens_l189_189719


namespace truck_driver_gas_l189_189268

variables (miles_per_gallon distance_to_station gallons_to_add gallons_in_tank total_gallons_needed : ℕ)
variables (current_gas_in_tank : ℕ)
variables (h1 : miles_per_gallon = 3)
variables (h2 : distance_to_station = 90)
variables (h3 : gallons_to_add = 18)

theorem truck_driver_gas :
  current_gas_in_tank = 12 :=
by
  -- Prove that the truck driver already has 12 gallons of gas in his tank,
  -- given the conditions provided.
  sorry

end truck_driver_gas_l189_189268


namespace find_m_l189_189168

def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) : vector_perpendicular (3, 1) (m, -3) → m = 1 :=
by
  sorry

end find_m_l189_189168


namespace tangent_line_equation_at_point_l189_189084

theorem tangent_line_equation_at_point 
  (x y : ℝ) (h_curve : y = x^3 - 2 * x) (h_point : (x, y) = (1, -1)) : 
  (x - y - 2 = 0) := 
sorry

end tangent_line_equation_at_point_l189_189084


namespace sqrt_seven_l189_189698

theorem sqrt_seven (x : ℝ) : x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end sqrt_seven_l189_189698


namespace tomatoes_left_l189_189388

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l189_189388


namespace proof_problem_l189_189033

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem proof_problem : f (Real.log 2) + f (Real.log (1 / 2)) = 6 := 
by 
  sorry

end proof_problem_l189_189033


namespace total_dots_not_visible_l189_189766

theorem total_dots_not_visible
    (num_dice : ℕ)
    (dots_per_die : ℕ)
    (visible_faces : ℕ → ℕ)
    (visible_faces_count : ℕ)
    (total_dots : ℕ)
    (dots_visible : ℕ) :
    num_dice = 4 →
    dots_per_die = 21 →
    visible_faces 0 = 1 →
    visible_faces 1 = 2 →
    visible_faces 2 = 2 →
    visible_faces 3 = 3 →
    visible_faces 4 = 4 →
    visible_faces 5 = 5 →
    visible_faces 6 = 6 →
    visible_faces 7 = 6 →
    visible_faces_count = 8 →
    total_dots = num_dice * dots_per_die →
    dots_visible = visible_faces 0 + visible_faces 1 + visible_faces 2 + visible_faces 3 + visible_faces 4 + visible_faces 5 + visible_faces 6 + visible_faces 7 →
    total_dots - dots_visible = 55 := by
  sorry

end total_dots_not_visible_l189_189766


namespace minimum_soldiers_to_add_l189_189641

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189641


namespace smallest_white_erasers_l189_189127

def total_erasers (n : ℕ) (pink : ℕ) (orange : ℕ) (purple : ℕ) (white : ℕ) : Prop :=
  pink = n / 5 ∧ orange = n / 6 ∧ purple = 10 ∧ white = n - (pink + orange + purple)

theorem smallest_white_erasers : ∃ n : ℕ, ∃ pink : ℕ, ∃ orange : ℕ, ∃ purple : ℕ, ∃ white : ℕ,
  total_erasers n pink orange purple white ∧ white = 9 := sorry

end smallest_white_erasers_l189_189127


namespace greatest_mondays_in_45_days_l189_189396

-- Define the days in a week
def days_in_week : ℕ := 7

-- Define the total days being considered
def total_days : ℕ := 45

-- Calculate the complete weeks in the total days
def complete_weeks : ℕ := total_days / days_in_week

-- Calculate the extra days
def extra_days : ℕ := total_days % days_in_week

-- Define that the period starts on Monday (condition)
def starts_on_monday : Bool := true

-- Prove that the greatest number of Mondays in the first 45 days is 7
theorem greatest_mondays_in_45_days (h1 : days_in_week = 7) (h2 : total_days = 45) (h3 : starts_on_monday = true) : 
  (complete_weeks + if starts_on_monday && extra_days >= 1 then 1 else 0) = 7 := 
by
  sorry

end greatest_mondays_in_45_days_l189_189396


namespace cos_equiv_l189_189902

theorem cos_equiv (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) (hcos : Real.cos (n * Real.pi / 180) = Real.cos (1018 * Real.pi / 180)) : n = 62 := 
sorry

end cos_equiv_l189_189902


namespace fuel_tank_initial_capacity_l189_189817

variables (fuel_consumption : ℕ) (journey_distance remaining_fuel initial_fuel : ℕ)

-- Define conditions
def fuel_consumption_rate := 12      -- liters per 100 km
def journey := 275                  -- km
def remaining := 14                 -- liters
def fuel_converted := (fuel_consumption_rate * journey) / 100

-- Define the proposition to be proved
theorem fuel_tank_initial_capacity :
  initial_fuel = fuel_converted + remaining :=
sorry

end fuel_tank_initial_capacity_l189_189817


namespace find_number_and_n_l189_189870

def original_number (x y z n : ℕ) : Prop := 
  n = 2 ∧ 100 * x + 10 * y + z = 178

theorem find_number_and_n (x y z n : ℕ) :
  (∀ x y z n, original_number x y z n) ↔ (n = 2 ∧ 100 * x + 10 * y + z = 178) := 
sorry

end find_number_and_n_l189_189870


namespace fraction_value_l189_189311

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end fraction_value_l189_189311


namespace ones_digit_of_p_is_3_l189_189565

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l189_189565


namespace chess_group_unique_pairings_l189_189701

theorem chess_group_unique_pairings:
  ∀ (players games : ℕ), players = 50 → games = 1225 →
  (∃ (games_per_pair : ℕ), games_per_pair = 1 ∧ (∀ p: ℕ, p < players → (players - 1) * games_per_pair = games)) :=
by
  sorry

end chess_group_unique_pairings_l189_189701


namespace probability_of_non_defective_is_0_92_l189_189262

-- Definitions of given conditions
def P_GradeB := 0.05
def P_GradeC := 0.03
def P_Defective := P_GradeB + P_GradeC
def P_GradeA := 1 - P_Defective

-- The theorem we need to prove
theorem probability_of_non_defective_is_0_92 : P_GradeA = 0.92 :=
by 
  unfold P_GradeA P_Defective P_GradeB P_GradeC
  -- This step just simplifies the definitions to show the desired equality
  calc
    1 - (0.05 + 0.03) = 1 - 0.08   : by sorry
                      ... = 0.92   : by sorry

end probability_of_non_defective_is_0_92_l189_189262


namespace gcf_of_180_240_300_l189_189714

def prime_factors (n : ℕ) : ℕ → Prop :=
λ p, p ^ (nat.factorization n p)

def gcf (n1 n2 n3 : ℕ) : ℕ :=
nat.gcd n1 (nat.gcd n2 n3)

theorem gcf_of_180_240_300 : gcf 180 240 300 = 60 := by
  sorry

end gcf_of_180_240_300_l189_189714


namespace bags_needed_l189_189748

-- Define the dimensions of one raised bed
def length_of_bed := 8
def width_of_bed := 4
def height_of_bed := 1

-- Calculate the volume of one raised bed
def volume_of_one_bed := length_of_bed * width_of_bed * height_of_bed

-- Define the number of beds
def number_of_beds := 2

-- Calculate the total volume needed for both beds
def total_volume := number_of_beds * volume_of_one_bed

-- Define the volume of soil in one bag
def volume_per_bag := 4

-- Calculate the number of bags needed
def number_of_bags := total_volume / volume_per_bag

-- Prove that the number of bags needed is 16
theorem bags_needed : number_of_bags = 16 := by
  show number_of_bags = 16 from sorry

end bags_needed_l189_189748


namespace cooking_ways_l189_189065

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end cooking_ways_l189_189065


namespace other_team_members_points_l189_189655

theorem other_team_members_points :
  ∃ (x : ℕ), ∃ (y : ℕ), (y ≤ 9 * 3) ∧ (x = y + 18 + x / 3 + x / 5) ∧ y = 24 :=
by
  sorry

end other_team_members_points_l189_189655


namespace find_k_l189_189403

theorem find_k (k : ℕ) : (1/2)^18 * (1/81)^k = (1/18)^18 → k = 9 :=
by
  intro h
  sorry

end find_k_l189_189403


namespace at_least_six_heads_in_10_flips_is_129_over_1024_l189_189255

def fair_coin_flip (n : ℕ) (prob_heads prob_tails : ℚ) : Prop :=
  (prob_heads = 1/2 ∧ prob_tails = 1/2)

noncomputable def at_least_six_consecutive_heads_probability (n : ℕ) : ℚ :=
  if n = 10 then 129 / 1024 else 0  -- this is specific to 10 flips and should be defined based on actual calculation for different n
  
theorem at_least_six_heads_in_10_flips_is_129_over_1024 :
  fair_coin_flip 10 (1/2) (1/2) →
  at_least_six_consecutive_heads_probability 10 = 129 / 1024 :=
by
  intros
  sorry

end at_least_six_heads_in_10_flips_is_129_over_1024_l189_189255


namespace distribute_balls_l189_189592

open Nat

theorem distribute_balls : 
  (∑ (c : ℕ × ℕ × ℕ) in {x : ℕ × ℕ × ℕ | x.1 + x.2 + x.3 = 6 ∧ x.1 ≤ 2 ∧ x.2 ≤ 2 ∧ x.3 ≤ 2}.to_finset, 
    if c.1 = 2 ∧ c.2 = 2 ∧ c.3 = 2 then 
      choose 6 2 * choose 4 2 * choose 2 2 / 6
    else if c.1 = 3 ∧ c.2 = 3 ∧ c.3 = 0 then
      choose 6 3 * choose 3 3 / 2
    else if c.1 = 4 ∧ c.2 = 2 ∧ c.3 = 0 then
      choose 6 4 * choose 2 2
    else if c.1 = 3 ∧ c.2 = 2 ∧ c.3 = 1 then
      choose 6 3 * choose 3 2 * choose 1 1
    else 0) = 100 := by
  sorry

end distribute_balls_l189_189592


namespace base_b_prime_digits_l189_189240

theorem base_b_prime_digits (b' : ℕ) (h1 : b'^4 ≤ 216) (h2 : 216 < b'^5) : b' = 3 :=
by {
  sorry
}

end base_b_prime_digits_l189_189240


namespace incorrect_proposition_example_l189_189718

theorem incorrect_proposition_example (p q : Prop) (h : ¬ (p ∧ q)) : ¬ (¬p ∧ ¬q) :=
by
  sorry

end incorrect_proposition_example_l189_189718


namespace correct_word_for_blank_l189_189204

theorem correct_word_for_blank :
  (∀ (word : String), word = "that" ↔ word = "whoever" ∨ word = "someone" ∨ word = "that" ∨ word = "any") :=
by
  sorry

end correct_word_for_blank_l189_189204


namespace lcm_36_100_l189_189304

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l189_189304


namespace S_div_T_is_one_half_l189_189478

def T (x y z : ℝ) := x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1

def supports (a b c x y z : ℝ) := 
  (x >= a ∧ y >= b ∧ z < c) ∨ 
  (x >= a ∧ z >= c ∧ y < b) ∨ 
  (y >= b ∧ z >= c ∧ x < a)

def S (x y z : ℝ) := T x y z ∧ supports (1/4) (1/4) (1/2) x y z

theorem S_div_T_is_one_half :
  let area_T := 1 -- Normalizing since area of T is in fact √3 / 2 but we care about ratios
  let area_S := 1/2 * area_T -- Given by the problem solution
  area_S / area_T = 1/2 := 
sorry

end S_div_T_is_one_half_l189_189478


namespace joe_max_money_l189_189843

noncomputable def max_guaranteed_money (initial_money : ℕ) (max_bet : ℕ) (num_bets : ℕ) : ℕ :=
  if initial_money = 100 ∧ max_bet = 17 ∧ num_bets = 5 then 98 else 0

theorem joe_max_money : max_guaranteed_money 100 17 5 = 98 := by
  sorry

end joe_max_money_l189_189843


namespace water_dispenser_capacity_l189_189692

theorem water_dispenser_capacity :
  ∀ (x : ℝ), (0.25 * x = 60) → x = 240 :=
by
  intros x h
  sorry

end water_dispenser_capacity_l189_189692


namespace apples_remaining_in_each_basket_l189_189195

-- Definition of conditions
def total_apples : ℕ := 128
def number_of_baskets : ℕ := 8
def apples_taken_per_basket : ℕ := 7

-- Definition of the problem
theorem apples_remaining_in_each_basket :
  (total_apples / number_of_baskets) - apples_taken_per_basket = 9 := 
by 
  sorry

end apples_remaining_in_each_basket_l189_189195


namespace number_of_subsets_l189_189597

-- Define the set
def my_set : Set ℕ := {1, 2, 3}

-- Theorem statement
theorem number_of_subsets : Finset.card (Finset.powerset {1, 2, 3}) = 8 :=
by
  sorry

end number_of_subsets_l189_189597


namespace intersection_A_B_l189_189328

def A (x : ℝ) : Prop := 0 < x ∧ x < 2
def B (x : ℝ) : Prop := -1 < x ∧ x < 1
def C (x : ℝ) : Prop := 0 < x ∧ x < 1

theorem intersection_A_B : ∀ x, A x ∧ B x ↔ C x := by
  sorry

end intersection_A_B_l189_189328


namespace sin_4theta_l189_189796

theorem sin_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5) :
  Real.sin (4 * θ) = (144 * Real.sqrt 7) / 625 := by
  sorry

end sin_4theta_l189_189796


namespace number_of_ordered_triples_modulo_1000000_l189_189350

def p : ℕ := 2017
def N : ℕ := sorry -- N is the number of ordered triples (a, b, c)

theorem number_of_ordered_triples_modulo_1000000 (N : ℕ) (h : ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ p * (p - 1) ∧ 1 ≤ b ∧ b ≤ p * (p - 1) ∧ a^b - b^a = p * c → true) : 
  N % 1000000 = 2016 :=
sorry

end number_of_ordered_triples_modulo_1000000_l189_189350


namespace find_smallest_number_l189_189994

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l189_189994


namespace election_proof_l189_189187

noncomputable def election_problem : Prop :=
  ∃ (V : ℝ) (votesA votesB votesC : ℝ),
  (votesA = 0.35 * V) ∧
  (votesB = votesA + 1800) ∧
  (votesC = 0.5 * votesA) ∧
  (V = votesA + votesB + votesC) ∧
  (V = 14400) ∧
  ((votesA / V) * 100 = 35) ∧
  ((votesB / V) * 100 = 47.5) ∧
  ((votesC / V) * 100 = 17.5)

theorem election_proof : election_problem := sorry

end election_proof_l189_189187


namespace scientific_notation_of_2102000_l189_189275

theorem scientific_notation_of_2102000 : ∃ (x : ℝ) (n : ℤ), 2102000 = x * 10 ^ n ∧ x = 2.102 ∧ n = 6 :=
by
  sorry

end scientific_notation_of_2102000_l189_189275


namespace function_monotone_increasing_l189_189145

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - log x

theorem function_monotone_increasing : ∀ x, 1 ≤ x → (0 < x) → (1 / 2) * x^2 - log x = f x → (∀ y, 1 ≤ y → (0 < y) → (f y ≤ f x)) :=
sorry

end function_monotone_increasing_l189_189145


namespace square_side_length_l189_189790

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l189_189790


namespace lcm_36_100_is_900_l189_189299

def prime_factors_36 : ℕ → Prop := 
  λ n, n = 36 → (2^2 * 3^2)

def prime_factors_100 : ℕ → Prop := 
  λ n, n = 100 → (2^2 * 5^2)

def lcm_36_100 := lcm 36 100

theorem lcm_36_100_is_900 : lcm_36_100 = 900 :=
by {
  sorry,
}

end lcm_36_100_is_900_l189_189299


namespace tomatoes_left_l189_189387

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l189_189387


namespace work_ratio_l189_189601

theorem work_ratio (M B : ℝ) 
  (h1 : 5 * (12 * M + 16 * B) = 1)
  (h2 : 4 * (13 * M + 24 * B) = 1) : 
  M / B = 2 := 
  sorry

end work_ratio_l189_189601


namespace product_of_102_and_27_l189_189975

theorem product_of_102_and_27 : 102 * 27 = 2754 :=
by
  sorry

end product_of_102_and_27_l189_189975


namespace allocation_methods_count_l189_189413

/-- The number of ways to allocate 24 quotas to 3 venues such that:
1. Each venue gets at least one quota.
2. Each venue gets a different number of quotas.
is equal to 222. -/
theorem allocation_methods_count : 
  ∃ n : ℕ, n = 222 ∧ 
  ∃ (a b c: ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a + b + c = 24 ∧ 
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c := 
sorry

end allocation_methods_count_l189_189413


namespace compute_a_plus_b_l189_189479

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l189_189479


namespace fraction_irreducible_l189_189374

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by {
    sorry
}

end fraction_irreducible_l189_189374


namespace quadratic_equation_exists_l189_189236

theorem quadratic_equation_exists : 
  ∃ a b c : ℝ, (a ≠ 0) ∧ (a + b + c = 0) ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) :=
by {
  let a := 1 : ℝ,
  let b := -2 : ℝ,
  let c := 1 : ℝ,
  use [a, b, c],
  split,
  {
    -- Proof for a ≠ 0
    exact one_ne_zero,
  },
  split,
  {
    -- Proof for a + b + c = 0
    calc
      a + b + c = 1 + -2 + 1 : by sorry
             ... = 0 : by sorry,
  },
  {
    -- Proof that the resultant quadratic equation is ax^2 + bx + c = 0
    intro x,
    calc
      a * x^2 + b * x + c = (1 : ℝ) * x^2 + (-2 : ℝ) * x + (1 : ℝ) : by sorry
                         ... = x^2 - 2 * x + 1 : by sorry,
  },
}

end quadratic_equation_exists_l189_189236


namespace A_inter_B_l189_189159

open Set Real

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { y | ∃ x, y = exp x }

theorem A_inter_B :
  A ∩ B = { z | 0 < z ∧ z < 3 } :=
by
  sorry

end A_inter_B_l189_189159


namespace abs_diff_of_slopes_l189_189588

theorem abs_diff_of_slopes (k1 k2 b : ℝ) (h : k1 * k2 < 0) (area_cond : (1 / 2) * 3 * |k1 - k2| * 3 = 9) :
  |k1 - k2| = 2 :=
by
  sorry

end abs_diff_of_slopes_l189_189588


namespace cubic_eq_root_nature_l189_189554

-- Definitions based on the problem statement
def cubic_eq (x : ℝ) : Prop := x^3 + 3 * x^2 - 4 * x - 12 = 0

-- The main theorem statement
theorem cubic_eq_root_nature :
  (∃ p n₁ n₂ : ℝ, cubic_eq p ∧ cubic_eq n₁ ∧ cubic_eq n₂ ∧ p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧ p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂) :=
sorry

end cubic_eq_root_nature_l189_189554


namespace ratio_m_n_is_3_over_19_l189_189461

def is_linear (a b : ℚ → ℚ) := ∀ x y, a x = 1 ∧ b y = 1

def constants_eq (m n : ℚ) : Prop := 
  3 * m + 5 * n + 9 = 1 ∧ 4 * m - 2 * n - 1 = 1

theorem ratio_m_n_is_3_over_19 (m n : ℚ) (h : constants_eq m n) :
  m / n = 3 / 19 :=
sorry

end ratio_m_n_is_3_over_19_l189_189461


namespace part1_part2_l189_189310

variable (a b : ℝ)

-- Part (1)
theorem part1 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h : a ≠ b) :
  A + B > 0 := sorry

-- Part (2)
theorem part2 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h: a * b = 1) : 
  A - B = -4 := sorry

end part1_part2_l189_189310


namespace roots_difference_one_l189_189893

theorem roots_difference_one (p : ℝ) :
  (∃ (x y : ℝ), (x^3 - 7 * x + p = 0) ∧ (y^3 - 7 * y + p = 0) ∧ (x - y = 1)) ↔ (p = 6 ∨ p = -6) :=
sorry

end roots_difference_one_l189_189893


namespace binomial_sum_l189_189552

theorem binomial_sum (n k : ℕ) (h : n = 10) (hk : k = 3) :
  Nat.choose n k + Nat.choose n (n - k) = 240 :=
by
  -- placeholder for actual proof
  sorry

end binomial_sum_l189_189552


namespace rooster_count_l189_189229

theorem rooster_count (total_chickens hens roosters : ℕ) 
  (h1 : total_chickens = roosters + hens)
  (h2 : roosters = 2 * hens)
  (h3 : total_chickens = 9000) 
  : roosters = 6000 := 
by
  sorry

end rooster_count_l189_189229


namespace number_of_roosters_l189_189226

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l189_189226


namespace minimum_soldiers_to_add_l189_189607

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  (N + 82) % 84 = 0 :=
by
  sorry

end minimum_soldiers_to_add_l189_189607


namespace minimum_soldiers_to_add_l189_189624

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189624


namespace log_product_identity_l189_189133

noncomputable def log {a b : ℝ} (ha : 1 < a) (hb : 0 < b) : ℝ := Real.log b / Real.log a

theorem log_product_identity : 
  log (by norm_num : (1 : ℝ) < 2) (by norm_num : (0 : ℝ) < 9) * 
  log (by norm_num : (1 : ℝ) < 3) (by norm_num : (0 : ℝ) < 8) = 6 :=
sorry

end log_product_identity_l189_189133


namespace sum_of_fractions_l189_189548

theorem sum_of_fractions : 
  (1/12 + 2/12 + 3/12 + 4/12 + 5/12 + 6/12 + 7/12 + 8/12 + 9/12 + 65/12 + 3/4) = 119 / 12 :=
by
  sorry

end sum_of_fractions_l189_189548


namespace perimeter_of_triangle_LMN_l189_189813

variable (K L M N : Type)
variables [MetricSpace K]
variables [MetricSpace L]
variables [MetricSpace M]
variables [MetricSpace N]
variables (KL LN MN : ℝ)
variables (perimeter_LMN : ℝ)

-- Given conditions
axiom KL_eq_24 : KL = 24
axiom LN_eq_24 : LN = 24
axiom MN_eq_9  : MN = 9

-- Prove the perimeter is 57
theorem perimeter_of_triangle_LMN : perimeter_LMN = KL + LN + MN → perimeter_LMN = 57 :=
by sorry

end perimeter_of_triangle_LMN_l189_189813


namespace determine_F_l189_189194

theorem determine_F (A H S M F : ℕ) (ha : 0 < A) (hh : 0 < H) (hs : 0 < S) (hm : 0 < M) (hf : 0 < F):
  (A * x + H * y = z) →
  (S * x + M * y = z) →
  (F * x = z) →
  (H > A) →
  (A ≠ H) →
  (S ≠ M) →
  (F ≠ A) →
  (F ≠ H) →
  (F ≠ S) →
  (F ≠ M) →
  x = z / F →
  y = ((F - A) / H * z) / z →
  F = (A * F - S * H) / (M - H) := sorry

end determine_F_l189_189194


namespace total_score_is_248_l189_189788

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℕ := (geography_score + math_score + english_score) / 3

theorem total_score_is_248 : geography_score + math_score + english_score + history_score = 248 := by
  -- proofs go here
  sorry

end total_score_is_248_l189_189788


namespace part1_part2_l189_189164

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Question 1: Prove that f(x) ≥ 3/4
theorem part1 (x a : ℝ) : f x a ≥ 3 / 4 := 
sorry

-- Question 2: Given f(4) < 13, find the range of a
theorem part2 (a : ℝ) (h : f 4 a < 13) : -2 < a ∧ a < 3 := 
sorry

end part1_part2_l189_189164


namespace total_spending_in_CAD_proof_l189_189006

-- Define Jayda's spending
def Jayda_spending_stall1 : ℤ := 400
def Jayda_spending_stall2 : ℤ := 120
def Jayda_spending_stall3 : ℤ := 250

-- Define the factor by which Aitana spends more
def Aitana_factor : ℚ := 2 / 5

-- Define the sales tax rate
def sales_tax_rate : ℚ := 0.10

-- Define the exchange rate from USD to CAD
def exchange_rate : ℚ := 1.25

-- Calculate Jayda's total spending in USD before tax
def Jayda_total_spending : ℤ := Jayda_spending_stall1 + Jayda_spending_stall2 + Jayda_spending_stall3

-- Calculate Aitana's spending at each stall
def Aitana_spending_stall1 : ℚ := Jayda_spending_stall1 + (Aitana_factor * Jayda_spending_stall1)
def Aitana_spending_stall2 : ℚ := Jayda_spending_stall2 + (Aitana_factor * Jayda_spending_stall2)
def Aitana_spending_stall3 : ℚ := Jayda_spending_stall3 + (Aitana_factor * Jayda_spending_stall3)

-- Calculate Aitana's total spending in USD before tax
def Aitana_total_spending : ℚ := Aitana_spending_stall1 + Aitana_spending_stall2 + Aitana_spending_stall3

-- Calculate the combined total spending in USD before tax
def combined_total_spending_before_tax : ℚ := Jayda_total_spending + Aitana_total_spending

-- Calculate the sales tax amount
def sales_tax : ℚ := sales_tax_rate * combined_total_spending_before_tax

-- Calculate the total spending including sales tax
def total_spending_including_tax : ℚ := combined_total_spending_before_tax + sales_tax

-- Convert the total spending to Canadian dollars
def total_spending_in_CAD : ℚ := total_spending_including_tax * exchange_rate

-- The theorem to be proven
theorem total_spending_in_CAD_proof : total_spending_in_CAD = 2541 := sorry

end total_spending_in_CAD_proof_l189_189006


namespace triangle_third_side_max_length_l189_189096

theorem triangle_third_side_max_length (a b : ℕ) (ha : a = 5) (hb : b = 11) : ∃ (c : ℕ), c = 15 ∧ (a + c > b ∧ b + c > a ∧ a + b > c) :=
by 
  sorry

end triangle_third_side_max_length_l189_189096


namespace number_of_matches_is_85_l189_189523

open Nat

/-- This definition calculates combinations of n taken k at a time. -/
def binom (n k : ℕ) : ℕ := n.choose k

/-- The calculation of total number of matches in the entire tournament. -/
def total_matches (groups teams_per_group : ℕ) : ℕ :=
  let matches_per_group := binom teams_per_group 2
  let total_matches_first_round := groups * matches_per_group
  let matches_final_round := binom groups 2
  total_matches_first_round + matches_final_round

/-- Theorem proving the total number of matches played is 85, given 5 groups with 6 teams each. -/
theorem number_of_matches_is_85 : total_matches 5 6 = 85 :=
  by
  sorry

end number_of_matches_is_85_l189_189523


namespace determine_k_l189_189576

theorem determine_k (S : ℕ → ℝ) (k : ℝ)
  (hSn : ∀ n, S n = k + 2 * (1 / 3)^n)
  (a1 : ℝ := S 1)
  (a2 : ℝ := S 2 - S 1)
  (a3 : ℝ := S 3 - S 2)
  (geom_property : a2^2 = a1 * a3) :
  k = -2 := 
by
  sorry

end determine_k_l189_189576


namespace find_dividend_l189_189428

-- Define the given conditions
def quotient : ℝ := 0.0012000000000000001
def divisor : ℝ := 17

-- State the problem: Prove that the dividend is the product of the quotient and the divisor
theorem find_dividend (q : ℝ) (d : ℝ) (hq : q = 0.0012000000000000001) (hd : d = 17) : 
  q * d = 0.0204000000000000027 :=
sorry

end find_dividend_l189_189428


namespace inverse_proportion_inequality_l189_189913

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l189_189913


namespace pipe_B_fill_time_l189_189121

variable (A B C : ℝ)
variable (fill_time : ℝ := 16)
variable (total_tank : ℝ := 1)

-- Conditions
axiom condition1 : A + B + C = (1 / fill_time)
axiom condition2 : A = 2 * B
axiom condition3 : B = 2 * C

-- Prove that B alone will take 56 hours to fill the tank
theorem pipe_B_fill_time : B = (1 / 56) :=
by sorry

end pipe_B_fill_time_l189_189121


namespace probability_at_least_50_cents_l189_189539

-- Define the types for coins
inductive Coin
| penny
| nickel
| dime

open Coin

-- Given conditions
def box : List Coin := List.replicate 2 penny ++ List.replicate 4 nickel ++ List.replicate 6 dime
def num_coins := 6

-- Define the value of each coin
def value (c : Coin) : ℕ :=
  match c with
  | penny   => 1
  | nickel  => 5
  | dime    => 10

-- Function to calculate the total value of a list of coins
def total_value (coins : List Coin) : ℕ :=
  coins.map value |>.sum

-- Total number of ways to draw 6 coins out of 12 coins
def total_outcomes := nat.choose 12 6

-- Number of successful outcomes
def successful_outcomes := 
  ((List.replicate 1 penny ++ List.replicate 5 dime) :: 
   (List.replicate 2 nickel ++ List.replicate 4 dime) ::
   (List.replicate 1 nickel ++ List.replicate 5 dime) ::
   [List.replicate 6 dime])
  .count (λ coins, total_value coins >= 50)

-- Calculate the probability
def probability := (successful_outcomes.toRat / total_outcomes.toRat)

-- Prove the probability is as expected
theorem probability_at_least_50_cents : probability = 127 / 924 := by
  sorry

end probability_at_least_50_cents_l189_189539


namespace value_of_expression_l189_189905

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = 3) (h3 : z = 4) :
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 :=
by
  sorry

end value_of_expression_l189_189905


namespace simplify_expression_l189_189078

noncomputable def algebraic_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) : ℚ :=
(1 - 3 / (a + 2)) / ((a^2 - 2 * a + 1) / (a^2 - 4))

theorem simplify_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) :
  algebraic_expression a h1 h2 h3 = (a - 2) / (a - 1) :=
by
  sorry

end simplify_expression_l189_189078


namespace am_gm_inequality_l189_189057

theorem am_gm_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by 
  sorry

end am_gm_inequality_l189_189057


namespace evaluate_expression_l189_189934

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 :=
by {
    sorry
}

end evaluate_expression_l189_189934


namespace part1_l189_189242

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
sorry

end part1_l189_189242


namespace cost_prices_l189_189738

theorem cost_prices (C_t C_c C_b : ℝ)
  (h1 : 2 * C_t = 1000)
  (h2 : 1.75 * C_c = 1750)
  (h3 : 0.75 * C_b = 1500) :
  C_t = 500 ∧ C_c = 1000 ∧ C_b = 2000 :=
by
  sorry

end cost_prices_l189_189738


namespace solve_combination_eq_l189_189375

theorem solve_combination_eq (x : ℕ) (h : x ≥ 3) : 
  (Nat.choose x 3 + Nat.choose x 2 = 12 * (x - 1)) ↔ (x = 9) := 
by
  sorry

end solve_combination_eq_l189_189375


namespace hyperbola_parabola_focus_l189_189587

open Classical

theorem hyperbola_parabola_focus :
  ∃ a : ℝ, (a > 0) ∧ (∃ c > 0, (c = 2) ∧ (a^2 + 3 = c^2)) → a = 1 :=
sorry

end hyperbola_parabola_focus_l189_189587


namespace contingency_fund_allocation_l189_189941

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l189_189941


namespace probability_of_6_consecutive_heads_l189_189257

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l189_189257


namespace ball_box_distribution_l189_189041

theorem ball_box_distribution :
  ∃ (distinct_ways : ℕ), distinct_ways = 7 :=
by
  let num_balls := 5
  let num_boxes := 4
  sorry

end ball_box_distribution_l189_189041


namespace simplify_expression_l189_189077

variable {x : ℝ}

theorem simplify_expression (h1 : x ≠ -1) (h2 : x ≠ 1) :
  ( 
    ( ((x + 1)^3 * (x^2 - x + 1)^3) / (x^3 + 1)^3 )^3 *
    ( ((x - 1)^3 * (x^2 + x + 1)^3) / (x^3 - 1)^3 )^3 
  ) = 1 := by
  sorry

end simplify_expression_l189_189077


namespace convert_neg300_degrees_to_radians_l189_189139

/-- Definition to convert degrees to radians -/
def degrees_to_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

/-- Problem statement: Converting -300 degrees to radians should equal -5/3 times pi -/
theorem convert_neg300_degrees_to_radians :
  degrees_to_radians (-300) = - (5/3) * Real.pi :=
by
  sorry

end convert_neg300_degrees_to_radians_l189_189139


namespace smallest_group_size_l189_189490

theorem smallest_group_size (n : ℕ) (k : ℕ) (hk : k > 2) (h1 : n % 2 = 0) (h2 : n % k = 0) :
  n = 6 :=
sorry

end smallest_group_size_l189_189490


namespace ravi_refrigerator_purchase_price_l189_189209

theorem ravi_refrigerator_purchase_price (purchase_price_mobile : ℝ) (sold_mobile : ℝ)
  (profit : ℝ) (loss : ℝ) (overall_profit : ℝ)
  (H1 : purchase_price_mobile = 8000)
  (H2 : loss = 0.04)
  (H3 : profit = 0.10)
  (H4 : overall_profit = 200) :
  ∃ R : ℝ, 0.96 * R + sold_mobile = R + purchase_price_mobile + overall_profit ∧ R = 15000 :=
by
  use 15000
  sorry

end ravi_refrigerator_purchase_price_l189_189209


namespace digits_difference_l189_189685

/-- Given a two-digit number represented as 10X + Y and the number obtained by interchanging its digits as 10Y + X,
    if the difference between the original number and the interchanged number is 81, 
    then the difference between the tens digit X and the units digit Y is 9. -/
theorem digits_difference (X Y : ℕ) (h : (10 * X + Y) - (10 * Y + X) = 81) : X - Y = 9 :=
by
  sorry

end digits_difference_l189_189685


namespace gcd_in_range_l189_189508

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l189_189508


namespace cos_4_arccos_2_5_l189_189898

noncomputable def arccos_2_5 : ℝ := Real.arccos (2/5)

theorem cos_4_arccos_2_5 : Real.cos (4 * arccos_2_5) = -47 / 625 :=
by
  -- Define x = arccos 2/5
  let x := arccos_2_5
  -- Declare the assumption cos x = 2/5
  have h_cos_x : Real.cos x = 2 / 5 := Real.cos_arccos (by norm_num : 2 / 5 ∈ Set.Icc (-1 : ℝ) 1)
  -- sorry to skip the proof
  sorry

end cos_4_arccos_2_5_l189_189898


namespace computation_of_sqrt_expr_l189_189755

theorem computation_of_sqrt_expr : 
  (Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549) := 
by
  sorry

end computation_of_sqrt_expr_l189_189755


namespace decimal_sum_sqrt_l189_189837

theorem decimal_sum_sqrt (a b : ℝ) (h₁ : a = Real.sqrt 5 - 2) (h₂ : b = Real.sqrt 13 - 3) : 
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
  sorry

end decimal_sum_sqrt_l189_189837


namespace sum_of_digits_ABCED_l189_189382

theorem sum_of_digits_ABCED {A B C D E : ℕ} (hABCED : 3 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) = 111111) :
  A + B + C + D + E = 20 := 
by
  sorry

end sum_of_digits_ABCED_l189_189382


namespace parabola_min_value_roots_l189_189987

-- Lean definition encapsulating the problem conditions and conclusion
theorem parabola_min_value_roots (a b c : ℝ) 
  (h1 : ∀ x, (a * x^2 + b * x + c) ≥ 36)
  (hvc : (b^2 - 4 * a * c) = 0)
  (hx1 : (a * (-3)^2 + b * (-3) + c) = 0)
  (hx2 : (a * (5)^2 + b * 5 + c) = 0)
  : a + b + c = 36 := by
  sorry

end parabola_min_value_roots_l189_189987


namespace max_gcd_lcm_l189_189070

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l189_189070


namespace number_of_cyclic_sets_l189_189952

open Nat

-- Define the conditions
def team_played_each_other_once (n : ℕ) : Prop :=
  n = 25

def won_and_lost_each_game (wins losses : ℕ) : Prop :=
  wins = 12 ∧ losses = 12

def total_sets_of_three_teams (n : ℕ) : ℕ :=
  (choose n 3)

def sets_where_one_team_dominates (n dominance : ℕ) : ℕ :=
  n * dominance

def cyclic_sets_count (total_sets dominated_sets : ℕ) : ℕ :=
  total_sets - dominated_sets

-- Define the final statement
theorem number_of_cyclic_sets :
  ∀ (n wins losses : ℕ),
  team_played_each_other_once n → won_and_lost_each_game wins losses →
  let total := total_sets_of_three_teams n in
  let dominance := choose wins 2 in
  let dominated := sets_where_one_team_dominates n dominance in
  cyclic_sets_count total dominated = 650 :=
by
  intros n wins losses n_cond wl_cond
  have n_eq : n = 25 := n_cond
  have wins_eq : wins = 12 := wl_cond.1
  have losses_eq : losses = 12 := wl_cond.2
  simp only [n_eq, wins_eq, losses_eq,
             total_sets_of_three_teams, sets_where_one_team_dominates,
             choose, choose_eq_fact_div_fact]
  sorry

end number_of_cyclic_sets_l189_189952


namespace safe_trip_possible_l189_189152

-- Define the time intervals and eruption cycles
def total_round_trip_time := 16
def trail_time := 8
def crater1_cycle := 18
def crater2_cycle := 10
def crater1_erupt := 1
def crater1_quiet := 17
def crater2_erupt := 1
def crater2_quiet := 9

-- Ivan wants to safely reach the summit and return
theorem safe_trip_possible : ∃ t, 
  -- t is a valid start time where both craters are quiet
  ((t % crater1_cycle) ≥ crater1_erupt ∧ (t % crater2_cycle) ≥ crater2_erupt) ∧
  -- t + total_round_trip_time is also safe for both craters
  (((t + total_round_trip_time) % crater1_cycle) ≥ crater1_erupt ∧ ((t + total_round_trip_time) % crater2_cycle) ≥ crater2_erupt) :=
sorry

end safe_trip_possible_l189_189152


namespace no_such_number_exists_l189_189193

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

/-- Define the number N as a sequence of digits a_n a_{n-1} ... a_0 -/
def number (a b : ℕ) (n : ℕ) : ℕ := a * 10^n + b

theorem no_such_number_exists :
  ¬ ∃ (N a_n b : ℕ) (n : ℕ), is_digit a_n ∧ a_n ≠ 0 ∧ b < 10^n ∧
    N = number a_n b n ∧
    b = N / 57 :=
sorry

end no_such_number_exists_l189_189193


namespace geometric_sequence_first_term_l189_189561

variable (a y z : ℕ)
variable (r : ℕ)
variable (h₁ : 16 = a * r^2)
variable (h₂ : 128 = a * r^4)

theorem geometric_sequence_first_term 
  (h₃ : r = 2) : a = 4 :=
by
  sorry

end geometric_sequence_first_term_l189_189561


namespace find_some_number_l189_189798

theorem find_some_number (x : ℤ) (h : 45 - (28 - (x - (15 - 20))) = 59) : x = 37 :=
by
  sorry

end find_some_number_l189_189798


namespace retailer_discount_problem_l189_189129

theorem retailer_discount_problem
  (CP MP SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (0.65 * CP))
  (h3 : SP = CP + (0.2375 * CP)) :
  (MP - SP) / MP * 100 = 25 :=
by
  sorry

end retailer_discount_problem_l189_189129


namespace Sierra_Crest_Trail_Length_l189_189663

theorem Sierra_Crest_Trail_Length (a b c d e : ℕ) 
(h1 : a + b + c = 36) 
(h2 : b + d = 30) 
(h3 : d + e = 38) 
(h4 : a + d = 32) : 
a + b + c + d + e = 74 := by
  sorry

end Sierra_Crest_Trail_Length_l189_189663


namespace james_total_payment_l189_189661

noncomputable def first_pair_cost : ℝ := 40
noncomputable def second_pair_cost : ℝ := 60
noncomputable def discount_applied_to : ℝ := min first_pair_cost second_pair_cost
noncomputable def discount_amount := discount_applied_to / 2
noncomputable def total_before_extra_discount := first_pair_cost + (second_pair_cost - discount_amount)
noncomputable def extra_discount := total_before_extra_discount / 4
noncomputable def final_amount := total_before_extra_discount - extra_discount

theorem james_total_payment : final_amount = 60 := by
  sorry

end james_total_payment_l189_189661


namespace minimum_soldiers_to_add_l189_189620

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189620


namespace transformed_function_zero_l189_189877

-- Definitions based on conditions
def f : ℝ → ℝ → ℝ := sorry  -- Assume this is the given function f(x, y)

-- Transformed function according to symmetry and reflections
def transformed_f (x y : ℝ) : Prop := f (y + 2) (x - 2) = 0

-- Lean statement to be proved
theorem transformed_function_zero (x y : ℝ) : transformed_f x y := sorry

end transformed_function_zero_l189_189877


namespace sum_of_square_and_divisor_not_square_l189_189835

theorem sum_of_square_and_divisor_not_square {A B : ℕ} (hA : A ≠ 0) (hA_square : ∃ k : ℕ, A = k * k) (hB_divisor : B ∣ A) : ¬ (∃ m : ℕ, A + B = m * m) := by
  -- Proof is omitted
  sorry

end sum_of_square_and_divisor_not_square_l189_189835


namespace average_of_remaining_primes_l189_189644

theorem average_of_remaining_primes (avg30: ℕ) (avg15: ℕ) (h1 : avg30 = 110) (h2 : avg15 = 95) : 
  ((30 * avg30 - 15 * avg15) / 15) = 125 := 
by
  -- Proof
  sorry

end average_of_remaining_primes_l189_189644


namespace expr_D_is_diff_of_squares_l189_189007

-- Definitions for the expressions
def expr_A (a b : ℤ) : ℤ := (a + 2 * b) * (-a - 2 * b)
def expr_B (m n : ℤ) : ℤ := (2 * m - 3 * n) * (3 * n - 2 * m)
def expr_C (x y : ℤ) : ℤ := (2 * x - 3 * y) * (3 * x + 2 * y)
def expr_D (a b : ℤ) : ℤ := (a - b) * (-b - a)

-- Theorem stating that Expression D can be calculated using the difference of squares formula
theorem expr_D_is_diff_of_squares (a b : ℤ) : expr_D a b = a^2 - b^2 :=
by sorry

end expr_D_is_diff_of_squares_l189_189007


namespace statement_B_l189_189359

variable (Student : Type)
variable (nora : Student)
variable (correctly_answered_all_math_questions : Student → Prop)
variable (received_at_least_B : Student → Prop)

theorem statement_B :
  (∀ s : Student, correctly_answered_all_math_questions s → received_at_least_B s) →
  (¬ received_at_least_B nora → ∃ q : Student, ¬ correctly_answered_all_math_questions q) :=
by
  intros h hn
  sorry

end statement_B_l189_189359


namespace circle_tangent_to_parabola_directrix_l189_189318

theorem circle_tangent_to_parabola_directrix (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 1/4 = 0 → y^2 = 4 * x → x = -1) → m = 3/4 :=
by
  sorry

end circle_tangent_to_parabola_directrix_l189_189318


namespace sum_of_first_n_terms_l189_189440

theorem sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + 2 * a 2 = 3)
  (h2 : ∀ n, a (n + 1) = a n + 2) :
  ∀ n, S n = n * (n - 4 / 3) := 
sorry

end sum_of_first_n_terms_l189_189440


namespace min_soldiers_to_add_l189_189616

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189616


namespace pyramid_volume_l189_189573

noncomputable def volume_pyramid (a b : ℝ) : ℝ :=
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2))

theorem pyramid_volume (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 < 4 * b^2) :
  volume_pyramid a b =
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2)) :=
sorry

end pyramid_volume_l189_189573


namespace cost_of_notebook_l189_189465

theorem cost_of_notebook (num_students : ℕ) (more_than_half_bought : ℕ) (num_notebooks : ℕ) 
                         (cost_per_notebook : ℕ) (total_cost : ℕ) 
                         (half_students : more_than_half_bought > 18) 
                         (more_than_one_notebook : num_notebooks > 1) 
                         (cost_gt_notebooks : cost_per_notebook > num_notebooks) 
                         (calc_total_cost : more_than_half_bought * cost_per_notebook * num_notebooks = 2310) :
  cost_per_notebook = 11 := 
sorry

end cost_of_notebook_l189_189465


namespace solve_fraction_equation_l189_189979

theorem solve_fraction_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_equation_l189_189979


namespace younger_son_age_30_years_later_eq_60_l189_189688

variable (age_diff : ℕ) (elder_age : ℕ) (younger_age_30_years_later : ℕ)

-- Conditions
axiom h1 : age_diff = 10
axiom h2 : elder_age = 40

-- Definition of younger son's current age
def younger_age : ℕ := elder_age - age_diff

-- Definition of younger son's age 30 years from now
def younger_age_future : ℕ := younger_age + 30

-- Proving the required statement
theorem younger_son_age_30_years_later_eq_60 (h_age_diff : age_diff = 10) (h_elder_age : elder_age = 40) :
  younger_age_future elder_age age_diff = 60 :=
by
  unfold younger_age
  unfold younger_age_future
  rw [h_age_diff, h_elder_age]
  sorry

end younger_son_age_30_years_later_eq_60_l189_189688


namespace largest_lcm_value_is_60_l189_189715

-- Define the conditions
def lcm_values : List ℕ := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 9, Nat.lcm 15 12, Nat.lcm 15 10, Nat.lcm 15 15]

-- State the proof problem
theorem largest_lcm_value_is_60 : lcm_values.maximum = some 60 :=
by
  repeat { sorry }

end largest_lcm_value_is_60_l189_189715


namespace neither_necessary_nor_sufficient_l189_189969

-- defining polynomial inequalities
def inequality_1 (a1 b1 c1 x : ℝ) : Prop := a1 * x^2 + b1 * x + c1 > 0
def inequality_2 (a2 b2 c2 x : ℝ) : Prop := a2 * x^2 + b2 * x + c2 > 0

-- defining proposition P and proposition Q
def P (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := ∀ x : ℝ, inequality_1 a1 b1 c1 x ↔ inequality_2 a2 b2 c2 x
def Q (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

-- prove that Q is neither a necessary nor sufficient condition for P
theorem neither_necessary_nor_sufficient {a1 b1 c1 a2 b2 c2 : ℝ} : ¬(Q a1 b1 c1 a2 b2 c2 ↔ P a1 b1 c1 a2 b2 c2) := 
sorry

end neither_necessary_nor_sufficient_l189_189969


namespace remainder_7_pow_253_mod_12_l189_189400

theorem remainder_7_pow_253_mod_12 : (7 ^ 253) % 12 = 7 := by
  sorry

end remainder_7_pow_253_mod_12_l189_189400


namespace smallest_number_increased_by_nine_divisible_by_8_11_24_l189_189726

theorem smallest_number_increased_by_nine_divisible_by_8_11_24 :
  ∃ x : ℕ, (x + 9) % 8 = 0 ∧ (x + 9) % 11 = 0 ∧ (x + 9) % 24 = 0 ∧ x = 255 :=
by
  sorry

end smallest_number_increased_by_nine_divisible_by_8_11_24_l189_189726


namespace other_root_and_m_l189_189948

-- Definitions for the conditions
def quadratic_eq (m : ℝ) := ∀ x : ℝ, x^2 + 2 * x + m = 0
def root (x : ℝ) (m : ℝ) := x^2 + 2 * x + m = 0

-- Theorem statement
theorem other_root_and_m (m : ℝ) (h : root 2 m) : ∃ t : ℝ, (2 + t = -2) ∧ (2 * t = m) ∧ t = -4 ∧ m = -8 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end other_root_and_m_l189_189948


namespace fresh_fruit_sold_l189_189213

-- Define the conditions
def total_fruit_sold : ℕ := 9792
def frozen_fruit_sold : ℕ := 3513

-- Define what we need to prove
theorem fresh_fruit_sold : (total_fruit_sold - frozen_fruit_sold = 6279) := by
  sorry

end fresh_fruit_sold_l189_189213


namespace jackson_money_l189_189657

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l189_189657


namespace range_of_m_l189_189781

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + |x - 1| ≥ (m + 2) * x - 1) ↔ (-3 - 2 * Real.sqrt 2) ≤ m ∧ m ≤ 0 := 
sorry

end range_of_m_l189_189781


namespace part1_part2_l189_189322

open Real

noncomputable def f (x a : ℝ) : ℝ := 45 * abs (x - a) + 45 * abs (x - 5)

theorem part1 (a : ℝ) :
    (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≤ 2 ∨ a ≥ 8) :=
sorry

theorem part2 (a : ℝ) (ha : a = 2) :
    ∀ (x : ℝ), (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + Real.sqrt 3) :=
sorry

end part1_part2_l189_189322


namespace pretzels_count_l189_189064

-- Define the number of pretzels
def pretzels : ℕ := 64

-- Given conditions
def goldfish (P : ℕ) : ℕ := 4 * P
def suckers : ℕ := 32
def kids : ℕ := 16
def items_per_kid : ℕ := 22
def total_items (P : ℕ) : ℕ := P + goldfish P + suckers

-- The theorem to prove
theorem pretzels_count : total_items pretzels = kids * items_per_kid := by
  sorry

end pretzels_count_l189_189064


namespace roots_sum_l189_189482

theorem roots_sum (a b : ℝ) 
  (h1 : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
          (roots_of (λ x, x^3 - 8 * x^2 + a * x - b) = {r1, r2, r3}) ) :
  a + b = 31 :=
sorry

end roots_sum_l189_189482


namespace solution_set_of_inequality_l189_189161

variable (a b x : ℝ)
variable (h1 : a < 0)

theorem solution_set_of_inequality (h : a * x + b < 0) : x > -b / a :=
sorry

end solution_set_of_inequality_l189_189161


namespace irreducible_fraction_eq_l189_189206

theorem irreducible_fraction_eq (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.gcd p q = 1) (h4 : q % 2 = 1) :
  ∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2 ^ k - 1) :=
by
  sorry

end irreducible_fraction_eq_l189_189206


namespace unique_solution_l189_189144

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  x > 0 ∧ (x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18)

theorem unique_solution :
  ∀ x : ℝ, satisfies_condition x ↔ x = 6 :=
by
  intro x
  unfold satisfies_condition
  sorry

end unique_solution_l189_189144


namespace lcm_36_100_eq_900_l189_189298

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l189_189298


namespace amy_remaining_money_l189_189865

-- Define initial amount and purchases
def initial_amount : ℝ := 15
def stuffed_toy_cost : ℝ := 2
def hot_dog_cost : ℝ := 3.5
def candy_apple_cost : ℝ := 1.5
def discount_rate : ℝ := 0.5

-- Define the discounted hot_dog_cost
def discounted_hot_dog_cost := hot_dog_cost * discount_rate

-- Define the total spent
def total_spent := stuffed_toy_cost + discounted_hot_dog_cost + candy_apple_cost

-- Define the remaining amount
def remaining_amount := initial_amount - total_spent

theorem amy_remaining_money : remaining_amount = 9.75 := by
  sorry

end amy_remaining_money_l189_189865


namespace european_postcards_cost_l189_189728

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end european_postcards_cost_l189_189728


namespace sabrina_fraction_books_second_month_l189_189977

theorem sabrina_fraction_books_second_month (total_books : ℕ) (pages_per_book : ℕ) (books_first_month : ℕ) (pages_total_read : ℕ)
  (h_total_books : total_books = 14)
  (h_pages_per_book : pages_per_book = 200)
  (h_books_first_month : books_first_month = 4)
  (h_pages_total_read : pages_total_read = 1000) :
  let total_pages := total_books * pages_per_book
  let pages_first_month := books_first_month * pages_per_book
  let pages_remaining := total_pages - pages_first_month
  let books_remaining := total_books - books_first_month
  let pages_read_first_month := total_pages - pages_total_read
  let pages_read_second_month := pages_read_first_month - pages_first_month
  let books_second_month := pages_read_second_month / pages_per_book
  let fraction_books := books_second_month / books_remaining
  fraction_books = 1 / 2 :=
by
  sorry

end sabrina_fraction_books_second_month_l189_189977


namespace smallest_c_for_polynomial_l189_189850

theorem smallest_c_for_polynomial :
  ∃ r1 r2 r3 : ℕ, (r1 * r2 * r3 = 2310) ∧ (r1 + r2 + r3 = 52) := sorry

end smallest_c_for_polynomial_l189_189850


namespace find_origin_coordinates_l189_189179

variable (x y : ℝ)

def original_eq (x y : ℝ) := x^2 - y^2 - 2*x - 2*y - 1 = 0

def transformed_eq (x' y' : ℝ) := x'^2 - y'^2 = 1

theorem find_origin_coordinates (x y : ℝ) :
  original_eq (x - 1) (y + 1) ↔ transformed_eq x y :=
by
  sorry

end find_origin_coordinates_l189_189179


namespace find_K_find_t_l189_189042

-- Proof Problem for G9.2
theorem find_K (x : ℚ) (K : ℚ) (h1 : x = 1.9898989) (h2 : x - 1 = K / 99) : K = 98 :=
sorry

-- Proof Problem for G9.3
theorem find_t (p q r t : ℚ)
  (h_avg1 : (p + q + r) / 3 = 18)
  (h_avg2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : t = 20 :=
sorry

end find_K_find_t_l189_189042


namespace min_soldiers_needed_l189_189608

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189608


namespace even_function_f_l189_189448

-- Problem statement:
-- Given that f is an even function and that for x < 0, f(x) = x^2 - 1/x,
-- prove that f(1) = 2.

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 1/x else 0

theorem even_function_f {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_neg_def : ∀ x, x < 0 → f x = x^2 - 1/x) : f 1 = 2 :=
by
  -- Proof body (to be completed)
  sorry

end even_function_f_l189_189448


namespace symmetric_line_b_value_l189_189447

theorem symmetric_line_b_value (b : ℝ) : 
  (∃ l1 l2 : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), l1 (x, y) ↔ y = -2 * x + b) ∧ 
    (∃ p2 : ℝ × ℝ, p2 = (1, 6) ∧ l2 p2) ∧
    l2 (-1, 6) ∧ 
    (∀ (x y : ℝ), l1 (x, y) ↔ l2 (-x, y))) →
  b = 4 := 
by
  sorry

end symmetric_line_b_value_l189_189447


namespace square_side_length_l189_189792

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l189_189792


namespace exists_ab_negated_l189_189383

theorem exists_ab_negated :
  ¬ (∀ a b : ℝ, (a + b = 0 → a^2 + b^2 = 0)) ↔ 
  ∃ a b : ℝ, (a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by
  sorry

end exists_ab_negated_l189_189383


namespace ellipse_standard_eq_l189_189446

theorem ellipse_standard_eq
  (e : ℝ) (a b : ℝ) (h1 : e = 1 / 2) (h2 : 2 * a = 4) (h3 : b^2 = a^2 - (a * e)^2)
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ↔
    ( ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ) :=
by
  sorry

end ellipse_standard_eq_l189_189446


namespace range_m_plus_2n_l189_189324

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def m_value (t : ℝ) : ℝ := 1 / t + 1 / (t ^ 2)

noncomputable def n_value (t : ℝ) : ℝ := Real.log t - 2 / t - 1

noncomputable def g (x : ℝ) : ℝ := (1 / (x ^ 2)) + 2 * Real.log x - (3 / x) - 2

theorem range_m_plus_2n :
  ∀ m n : ℝ, (∃ t > 0, m = m_value t ∧ n = n_value t) →
  (m + 2 * n) ∈ Set.Ici (-2 * Real.log 2 - 4) := by
  sorry

end range_m_plus_2n_l189_189324


namespace quadratic_has_real_root_for_any_t_l189_189036

theorem quadratic_has_real_root_for_any_t (s : ℝ) :
  (∀ t : ℝ, ∃ x : ℝ, s * x^2 + t * x + s - 1 = 0) ↔ (0 < s ∧ s ≤ 1) :=
by
  sorry

end quadratic_has_real_root_for_any_t_l189_189036


namespace convex_polygon_diagonals_l189_189886

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l189_189886


namespace smallest_product_of_two_distinct_primes_greater_than_50_l189_189761

theorem smallest_product_of_two_distinct_primes_greater_than_50 : 
  ∃ (p q : ℕ), p > 50 ∧ q > 50 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 3127 :=
by 
  sorry

end smallest_product_of_two_distinct_primes_greater_than_50_l189_189761


namespace total_marks_l189_189266

-- Variables and conditions
variables (M C P : ℕ)
variable (h1 : C = P + 20)
variable (h2 : (M + C) / 2 = 40)

-- Theorem statement
theorem total_marks (M C P : ℕ) (h1 : C = P + 20) (h2 : (M + C) / 2 = 40) : M + P = 60 :=
sorry

end total_marks_l189_189266


namespace polynomial_solution_l189_189199

theorem polynomial_solution (f : ℝ → ℝ) (x : ℝ) (h : f (x^2 + 2) = x^4 + 6 * x^2 + 4) : 
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  sorry

end polynomial_solution_l189_189199


namespace tomatoes_left_l189_189390

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction_eaten : ℚ) :
  initial_tomatoes = 21 ∧ birds = 2 ∧ fraction_eaten = 1/3 ->
  initial_tomatoes - initial_tomatoes * fraction_eaten = 14 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  rw [Nat.cast_sub 21 7 _, Nat.cast_mul, Nat.cast_div]; norm_num -- Converting to rational arithmetic and proving directly
  exact le_of_lt_nat (div_lt_self (zero_lt_nat 21) (zero_lt_nat 3))

end tomatoes_left_l189_189390


namespace ones_digit_of_prime_p_l189_189571

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l189_189571


namespace blocks_to_store_l189_189751

theorem blocks_to_store
  (T : ℕ) (S : ℕ)
  (hT : T = 25)
  (h_total_walk : S + 6 + 8 = T) :
  S = 11 :=
by
  sorry

end blocks_to_store_l189_189751


namespace minimum_soldiers_to_add_l189_189629

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : 
  ∃ k : ℕ, 84 * k + 2 - N = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189629


namespace perimeter_of_figure_l189_189651

variable (x y : ℝ)
variable (lengths : Set ℝ)
variable (perpendicular_adjacent : Prop)
variable (area : ℝ)

-- Conditions
def condition_1 : Prop := ∀ l ∈ lengths, l = x ∨ l = y
def condition_2 : Prop := perpendicular_adjacent
def condition_3 : Prop := area = 252
def condition_4 : Prop := x = 2 * y

-- Problem statement
theorem perimeter_of_figure
  (h1 : condition_1 x y lengths)
  (h2 : condition_2 perpendicular_adjacent)
  (h3 : condition_3 area)
  (h4 : condition_4 x y) :
  ∃ perimeter : ℝ, perimeter = 96 := by
  sorry

end perimeter_of_figure_l189_189651


namespace infinitely_many_n_squared_plus_one_no_special_divisor_l189_189244

theorem infinitely_many_n_squared_plus_one_no_special_divisor :
  ∃ (f : ℕ → ℕ), (∀ n, f n ≠ 0) ∧ ∀ n, ∀ k, f n^2 + 1 ≠ k^2 + 1 ∨ k^2 + 1 = 1 :=
by
  sorry

end infinitely_many_n_squared_plus_one_no_special_divisor_l189_189244


namespace minimum_soldiers_to_add_l189_189638

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189638


namespace find_four_digit_number_l189_189426

def digits_sum (n : ℕ) : ℕ := (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
def digits_product (n : ℕ) : ℕ := (n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)

theorem find_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (digits_sum n) * (digits_product n) = 3990 :=
by
  -- The proof is omitted as instructed.
  sorry

end find_four_digit_number_l189_189426


namespace min_value_of_a2_plus_b2_l189_189556

theorem min_value_of_a2_plus_b2 
  (a b : ℝ) 
  (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4 := 
sorry

end min_value_of_a2_plus_b2_l189_189556


namespace max_gcd_lcm_eq_10_l189_189067

open Nat -- Opening the namespace for natural numbers

theorem max_gcd_lcm_eq_10
  (a b c : ℕ) 
  (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) :
  gcd (lcm a b) c ≤ 10 := sorry

end max_gcd_lcm_eq_10_l189_189067


namespace minimum_soldiers_to_add_l189_189621

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) : ∃ (add : ℕ), add = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l189_189621


namespace root_of_equation_value_l189_189923

theorem root_of_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2 * m^2 - 4 * m + 5 = 11 := 
by
  sorry

end root_of_equation_value_l189_189923


namespace inverse_proportion_inequality_l189_189919

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end inverse_proportion_inequality_l189_189919


namespace tan_alpha_plus_pi_over_4_l189_189768

theorem tan_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_l189_189768


namespace regular_rate_survey_l189_189004

theorem regular_rate_survey (R : ℝ) 
  (total_surveys : ℕ := 50)
  (rate_increase : ℝ := 0.30)
  (cellphone_surveys : ℕ := 35)
  (total_earnings : ℝ := 605) :
  35 * (1.30 * R) + 15 * R = 605 → R = 10 :=
by
  sorry

end regular_rate_survey_l189_189004


namespace volume_ratio_of_cubes_l189_189043

theorem volume_ratio_of_cubes (s2 : ℝ) : 
  let s1 := s2 * (Real.sqrt 3)
  let V1 := s1^3
  let V2 := s2^3
  V1 / V2 = 3 * (Real.sqrt 3) :=
by
  admit -- si



end volume_ratio_of_cubes_l189_189043


namespace husband_age_l189_189123

theorem husband_age (a b : ℕ) (w_age h_age : ℕ) (ha : a > 0) (hb : b > 0) 
  (hw_age : w_age = 10 * a + b) 
  (hh_age : h_age = 10 * b + a) 
  (h_older : h_age > w_age)
  (h_difference : 9 * (b - a) = a + b) :
  h_age = 54 :=
by
  sorry

end husband_age_l189_189123


namespace breadth_of_rectangular_plot_l189_189847

variable (b l : ℕ)

def length_eq_thrice_breadth (b : ℕ) : ℕ := 3 * b

def area_of_rectangle_eq_2700 (b l : ℕ) : Prop := l * b = 2700

theorem breadth_of_rectangular_plot (h1 : l = 3 * b) (h2 : l * b = 2700) : b = 30 :=
by
  sorry

end breadth_of_rectangular_plot_l189_189847


namespace kekai_remaining_money_l189_189820

-- Definitions based on given conditions
def shirts_sold := 5
def price_per_shirt := 1
def pants_sold := 5
def price_per_pant := 3
def half_fraction := 1 / 2 : ℝ

-- Proving that Kekai's remaining money is $10
theorem kekai_remaining_money : 
  let earnings_from_shirts := shirts_sold * price_per_shirt in
  let earnings_from_pants := pants_sold * price_per_pant in
  let total_earnings := earnings_from_shirts + earnings_from_pants in
  let money_given_to_parents := total_earnings * half_fraction in
  let remaining_money := total_earnings - money_given_to_parents in
  remaining_money = 10 :=
by
  sorry

end kekai_remaining_money_l189_189820


namespace ones_digit_of_prime_sequence_l189_189568

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l189_189568


namespace dot_product_of_vectors_l189_189167

theorem dot_product_of_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  sorry

end dot_product_of_vectors_l189_189167


namespace binom_divisibility_by_prime_l189_189402

-- Given definitions
variable (p k : ℕ) (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2)

-- Main theorem statement
theorem binom_divisibility_by_prime
  (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2) :
  Nat.choose (p - k + 1) k - Nat.choose (p - k - 1) (k - 2) ≡ 0 [MOD p] :=
sorry

end binom_divisibility_by_prime_l189_189402


namespace tomatoes_left_l189_189389

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction_eaten : ℚ) :
  initial_tomatoes = 21 ∧ birds = 2 ∧ fraction_eaten = 1/3 ->
  initial_tomatoes - initial_tomatoes * fraction_eaten = 14 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  norm_num
  rw [Nat.cast_sub 21 7 _, Nat.cast_mul, Nat.cast_div]; norm_num -- Converting to rational arithmetic and proving directly
  exact le_of_lt_nat (div_lt_self (zero_lt_nat 21) (zero_lt_nat 3))

end tomatoes_left_l189_189389


namespace younger_son_age_30_years_later_eq_60_l189_189689

variable (age_diff : ℕ) (elder_age : ℕ) (younger_age_30_years_later : ℕ)

-- Conditions
axiom h1 : age_diff = 10
axiom h2 : elder_age = 40

-- Definition of younger son's current age
def younger_age : ℕ := elder_age - age_diff

-- Definition of younger son's age 30 years from now
def younger_age_future : ℕ := younger_age + 30

-- Proving the required statement
theorem younger_son_age_30_years_later_eq_60 (h_age_diff : age_diff = 10) (h_elder_age : elder_age = 40) :
  younger_age_future elder_age age_diff = 60 :=
by
  unfold younger_age
  unfold younger_age_future
  rw [h_age_diff, h_elder_age]
  sorry

end younger_son_age_30_years_later_eq_60_l189_189689


namespace problem1_problem2_l189_189753

theorem problem1 :
  (2 / 3) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3) * Real.sqrt 27 = - (4 / 3) * Real.sqrt 6 :=
sorry

theorem problem2 :
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 :=
sorry

end problem1_problem2_l189_189753


namespace at_least_6_heads_in_10_flips_l189_189248

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l189_189248


namespace geometric_sequence_common_ratio_l189_189188

variables {a_n : ℕ → ℝ} {S_n q : ℝ}

axiom a1_eq : a_n 1 = 2
axiom an_eq : ∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0
axiom Sn_eq : ∀ n, a_n n = -64 → S_n = -42 → q = -2

theorem geometric_sequence_common_ratio (q : ℝ) :
  (∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0) →
  a_n 1 = 2 →
  (∀ n, a_n n = -64 → S_n = -42 → q = -2) :=
by intros _ _ _; sorry

end geometric_sequence_common_ratio_l189_189188


namespace trig_expression_value_l189_189160

open Real

theorem trig_expression_value (x : ℝ) (h : tan (π - x) = -2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
sorry

end trig_expression_value_l189_189160


namespace complement_U_P_l189_189434

def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

theorem complement_U_P :
  (U \ P) = Set.Ici (1 / 2) := 
by
  sorry

end complement_U_P_l189_189434


namespace distance_vancouver_calgary_l189_189218

theorem distance_vancouver_calgary : 
  ∀ (map_distance : ℝ) (scale : ℝ) (terrain_factor : ℝ), 
    map_distance = 12 →
    scale = 35 →
    terrain_factor = 1.1 →
    map_distance * scale * terrain_factor = 462 := by
  intros map_distance scale terrain_factor 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end distance_vancouver_calgary_l189_189218


namespace problem1_problem2_l189_189579

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l189_189579


namespace Jackson_money_is_125_l189_189659

-- Definitions of given conditions
def Williams_money : ℕ := sorry
def Jackson_money : ℕ := 5 * Williams_money

-- Given condition: together they have $150
def total_money_condition : Prop := 
  Jackson_money + Williams_money = 150

-- Proof statement
theorem Jackson_money_is_125 
  (h1 : total_money_condition) : 
  Jackson_money = 125 := 
by
  sorry

end Jackson_money_is_125_l189_189659


namespace min_soldiers_to_add_l189_189614

theorem min_soldiers_to_add (N : ℕ) (k m : ℕ) (h1 : N = 7 * k + 2) (h2 : N = 12 * m + 2) :
  let add := lcm 7 12 - 2 in add = 82 :=
by
  -- Define N to satisfy the given conditions
  let N := 7 * 12 + 2
  let add := 84 - 2
  have h3 : add = 82 := by simp
  exact h3
  sorry

end min_soldiers_to_add_l189_189614


namespace min_spend_for_free_delivery_l189_189272

theorem min_spend_for_free_delivery : 
  let chicken_price := 1.5 * 6.00
  let lettuce_price := 3.00
  let tomato_price := 2.50
  let sweet_potato_price := 4 * 0.75
  let broccoli_price := 2 * 2.00
  let brussel_sprouts_price := 2.50
  let current_total := chicken_price + lettuce_price + tomato_price + sweet_potato_price + broccoli_price + brussel_sprouts_price
  let additional_needed := 11.00 
  let minimum_spend := current_total + additional_needed
  minimum_spend = 35.00 :=
by
  sorry

end min_spend_for_free_delivery_l189_189272


namespace at_least_6_heads_in_10_flips_l189_189249

def coin_flip : Type := bool

def is_heads (x : coin_flip) : Prop := x = tt

def num_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i : ℕ, i + n ≤ l.length ∧ l.drop i.take n = list.replicate n tt

def prob_at_least_n_consecutive_heads (l : list coin_flip) (n : ℕ) : Prop :=
  ∃ i ≤ l.length - n + 1, list.replicate n tt = l.drop (i - 1).take n

noncomputable def at_least_6_heads_in_10_flips_prob : ℚ :=
  (129:ℚ) / (1024:ℚ)

theorem at_least_6_heads_in_10_flips :
  prob_at_least_n_consecutive_heads (list.replicate 10 coin_flip) 6 = at_least_6_heads_in_10_flips_prob :=
by
  sorry

end at_least_6_heads_in_10_flips_l189_189249


namespace expected_value_of_winnings_l189_189487

theorem expected_value_of_winnings :
  let primes := [2, 3, 5, 7]
  let composites := [4, 6, 8]
  let p_prime := 4/8
  let p_composite := 3/8
  let p_one := 1/8
  let winnings_primes := 2*2 + 2*3 + 2*5 + 2*7
  let loss_composite := -1
  let loss_one := -3
  let E := p_prime * winnings_primes + p_composite * loss_composite + p_one * loss_one
  E = 16.25 :=
by
  let primes := [2, 3, 5, 7]
  let composites := [4, 6, 8]
  let p_prime := 4/8
  let p_composite := 3/8
  let p_one := 1/8
  let winnings_primes := 2*2 + 2*3 + 2*5 + 2*7
  let loss_composite := -1
  let loss_one := -3
  let E := p_prime * winnings_primes + p_composite * loss_composite + p_one * loss_one
  sorry

end expected_value_of_winnings_l189_189487


namespace find_gain_percent_l189_189720

theorem find_gain_percent (CP SP : ℝ) (h1 : CP = 20) (h2 : SP = 25) : 100 * ((SP - CP) / CP) = 25 := by
  sorry

end find_gain_percent_l189_189720


namespace max_value_of_x_plus_y_l189_189438

theorem max_value_of_x_plus_y (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : x + y ≤ 5 :=
sorry

end max_value_of_x_plus_y_l189_189438


namespace total_apples_and_pears_l189_189120

theorem total_apples_and_pears (x y : ℤ) 
  (h1 : x = 3 * (y / 2 + 1)) 
  (h2 : x = 5 * (y / 4 - 3)) : 
  x + y = 39 :=
sorry

end total_apples_and_pears_l189_189120


namespace tracy_feeds_dogs_times_per_day_l189_189526

theorem tracy_feeds_dogs_times_per_day : 
  let cups_per_meal_per_dog := 1.5
  let dogs := 2
  let total_pounds_per_day := 4
  let cups_per_pound := 2.25
  (total_pounds_per_day * cups_per_pound) / (dogs * cups_per_meal_per_dog) = 3 :=
by
  sorry

end tracy_feeds_dogs_times_per_day_l189_189526


namespace lcm_of_36_and_100_l189_189293

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l189_189293


namespace find_x_coordinate_l189_189772

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x ∧ x > 0 

noncomputable def is_twice_distance (x : ℝ) : Prop :=
  let focus_x : ℝ := 3 / 2
  let d1 := x + focus_x
  let d2 := x
  d1 = 2 * d2

theorem find_x_coordinate (x y : ℝ) :
  point_on_parabola x y →
  is_twice_distance x →
  x = 3 / 2 :=
by
  intros
  sorry

end find_x_coordinate_l189_189772


namespace train_time_to_pass_platform_l189_189267

-- Definitions as per the conditions
def length_of_train : ℕ := 720 -- Length of train in meters
def speed_of_train_kmh : ℕ := 72 -- Speed of train in km/hr
def length_of_platform : ℕ := 280 -- Length of platform in meters

-- Conversion factor and utility functions
def kmh_to_ms (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

def time_to_pass (distance speed_ms : ℕ) : ℕ :=
  distance / speed_ms

-- Main statement to be proven
theorem train_time_to_pass_platform :
  time_to_pass (total_distance length_of_train length_of_platform) (kmh_to_ms speed_of_train_kmh) = 50 :=
by
  sorry

end train_time_to_pass_platform_l189_189267


namespace kicks_before_break_l189_189493

def total_kicks : ℕ := 98
def kicks_after_break : ℕ := 36
def kicks_needed_to_goal : ℕ := 19

theorem kicks_before_break :
  total_kicks - (kicks_after_break + kicks_needed_to_goal) = 43 := 
by
  -- proof wanted
  sorry

end kicks_before_break_l189_189493


namespace problem_a_add_b_eq_five_l189_189852

variable {a b : ℝ}

theorem problem_a_add_b_eq_five
  (h1 : ∀ x, -2 < x ∧ x < 3 → ax^2 + x + b > 0)
  (h2 : a < 0) :
  a + b = 5 :=
sorry

end problem_a_add_b_eq_five_l189_189852


namespace min_soldiers_needed_l189_189610

theorem min_soldiers_needed (N : ℕ) (k : ℕ) (m : ℕ) : 
  (N ≡ 2 [MOD 7]) → (N ≡ 2 [MOD 12]) → (N = 2) → (84 - N = 82) :=
by
  sorry

end min_soldiers_needed_l189_189610


namespace sum_of_first_17_terms_arithmetic_sequence_l189_189450

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem sum_of_first_17_terms_arithmetic_sequence
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 3 + a 9 + a 15 = 9) :
  sum_of_first_n_terms a 17 = 51 :=
sorry

end sum_of_first_17_terms_arithmetic_sequence_l189_189450


namespace savannah_rolls_l189_189074

-- Definitions and conditions
def total_gifts := 12
def gifts_per_roll_1 := 3
def gifts_per_roll_2 := 5
def gifts_per_roll_3 := 4

-- Prove the number of rolls
theorem savannah_rolls :
  gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3 = total_gifts →
  3 + 5 + 4 = 12 →
  3 = total_gifts / (gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3) :=
by
  intros h1 h2
  sorry

end savannah_rolls_l189_189074


namespace carla_initial_marbles_l189_189134

theorem carla_initial_marbles (total_marbles : ℕ) (bought_marbles : ℕ) (initial_marbles : ℕ) 
  (h1 : total_marbles = 187) (h2 : bought_marbles = 134) (h3 : total_marbles = initial_marbles + bought_marbles) : 
  initial_marbles = 53 := 
sorry

end carla_initial_marbles_l189_189134


namespace bathing_suits_per_model_l189_189982

def models : ℕ := 6
def evening_wear_sets_per_model : ℕ := 3
def time_per_trip_minutes : ℕ := 2
def total_show_time_minutes : ℕ := 60

theorem bathing_suits_per_model : (total_show_time_minutes - (models * evening_wear_sets_per_model * time_per_trip_minutes)) / (time_per_trip_minutes * models) = 2 :=
by
  sorry

end bathing_suits_per_model_l189_189982


namespace value_of_x_plus_y_l189_189575

theorem value_of_x_plus_y (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
by
  sorry

end value_of_x_plus_y_l189_189575


namespace right_angle_triangle_sets_l189_189008

def is_right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angle_triangle_sets :
  ¬ is_right_angle_triangle (2 / 3) 2 (5 / 4) :=
by {
  sorry
}

end right_angle_triangle_sets_l189_189008


namespace probability_at_least_one_multiple_of_4_is_correct_l189_189749

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 100
  let multiples_of_4 := 25
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let p_non_multiple := (non_multiples_of_4 : ℚ) / total_numbers
  let p_both_non_multiples := p_non_multiple^2
  let p_at_least_one_multiple := 1 - p_both_non_multiples
  p_at_least_one_multiple

theorem probability_at_least_one_multiple_of_4_is_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_is_correct_l189_189749


namespace find_m_l189_189787

open Real

-- Definitions based on problem conditions
def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

-- The dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Prove the final statement using given conditions
theorem find_m (m : ℝ) (h1 : dot_product (vector_a m) vector_b + dot_product vector_b vector_b = 0) :
  m = 8 :=
sorry

end find_m_l189_189787


namespace problem1_problem2_problem3_problem4_l189_189547

-- Proving the given mathematical equalities

theorem problem1 (x : ℝ) : 
  (x^4)^3 + (x^3)^4 - 2 * x^4 * x^8 = 0 := 
  sorry

theorem problem2 (x y : ℝ) : 
  (-2 * x^2 * y^3)^2 * (x * y)^3 = 4 * x^7 * y^9 := 
  sorry

theorem problem3 (a : ℝ) : 
  (-2 * a)^6 - (-3 * a^3)^2 + (-(2 * a)^2)^3 = -9 * a^6 := 
  sorry

theorem problem4 : 
  abs (- 1/8) + (Real.pi - 3)^0 + (- 1/2)^3 - (1/3)^(-2) = 8/9 := 
  sorry

end problem1_problem2_problem3_problem4_l189_189547


namespace tetrahedron_BC_squared_l189_189967

theorem tetrahedron_BC_squared (AB AC BC R r : ℝ) 
  (h1 : AB = 1) 
  (h2 : AC = 1) 
  (h3 : 1 ≤ BC) 
  (h4 : R = 4 * r) 
  (concentric : AB = AC ∧ R > 0 ∧ r > 0) :
  BC^2 = 1 + Real.sqrt (7 / 15) := 
by 
sorry

end tetrahedron_BC_squared_l189_189967
