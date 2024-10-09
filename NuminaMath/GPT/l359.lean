import Mathlib

namespace all_positive_integers_in_A_l359_35958

variable (A : Set ℕ)

-- Conditions
def has_at_least_three_elements : Prop :=
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c

def all_divisors_in_set : Prop :=
  ∀ m : ℕ, m ∈ A → (∀ d : ℕ, d ∣ m → d ∈ A)

def  bc_plus_one_in_set : Prop :=
  ∀ b c : ℕ, 1 < b → b < c → b ∈ A → c ∈ A → 1 + b * c ∈ A

-- Theorem statement
theorem all_positive_integers_in_A
  (h1 : has_at_least_three_elements A)
  (h2 : all_divisors_in_set A)
  (h3 : bc_plus_one_in_set A) : ∀ n : ℕ, n > 0 → n ∈ A := 
by
  -- proof steps would go here
  sorry

end all_positive_integers_in_A_l359_35958


namespace factor_polynomial_l359_35957

theorem factor_polynomial :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x ^ 2 =
  (3 * x ^ 2 + 59 * x + 231) * (3 * x ^ 2 + 53 * x + 231) := by
  sorry

end factor_polynomial_l359_35957


namespace petya_wins_prize_probability_atleast_one_wins_probability_l359_35976

/-- Petya and 9 other people each roll a fair six-sided die. 
    A player wins a prize if they roll a number that nobody else rolls more than once.-/
theorem petya_wins_prize_probability : (5 / 6) ^ 9 = 0.194 :=
sorry

/-- The probability that at least one player gets a prize in the game where Petya and
    9 others roll a fair six-sided die is 0.919. -/
theorem atleast_one_wins_probability : 1 - (1 / 6) ^ 9 = 0.919 :=
sorry

end petya_wins_prize_probability_atleast_one_wins_probability_l359_35976


namespace Jorge_Giuliana_cakes_l359_35912

theorem Jorge_Giuliana_cakes (C : ℕ) :
  (2 * 7 + 2 * C + 2 * 30 = 110) → (C = 18) :=
by
  sorry

end Jorge_Giuliana_cakes_l359_35912


namespace find_x_l359_35974

theorem find_x : ∃ x : ℤ, x + 3 * 10 = 33 → x = 3 := by
  sorry

end find_x_l359_35974


namespace exists_m_square_between_l359_35968

theorem exists_m_square_between (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, a < m^2 ∧ m^2 < d := 
sorry

end exists_m_square_between_l359_35968


namespace speed_of_other_person_l359_35964

-- Definitions related to the problem conditions
def pooja_speed : ℝ := 3  -- Pooja's speed in km/hr
def time : ℝ := 4  -- Time in hours
def distance : ℝ := 20  -- Distance between them after 4 hours in km

-- Define the unknown speed S as a parameter to be solved
variable (S : ℝ)

-- Define the relative speed when moving in opposite directions
def relative_speed (S : ℝ) : ℝ := S + pooja_speed

-- Create a theorem to encapsulate the problem and to be proved
theorem speed_of_other_person 
  (h : distance = relative_speed S * time) : S = 2 := 
  sorry

end speed_of_other_person_l359_35964


namespace library_visitor_ratio_l359_35973

theorem library_visitor_ratio (T : ℕ) (h1 : 50 + T + 20 * 4 = 250) : T / 50 = 2 :=
by
  sorry

end library_visitor_ratio_l359_35973


namespace sum_of_edges_l359_35936

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l359_35936


namespace smallest_n_multiple_of_7_l359_35913

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end smallest_n_multiple_of_7_l359_35913


namespace celsius_to_fahrenheit_l359_35982

theorem celsius_to_fahrenheit (C F : ℤ) (h1 : C = 50) (h2 : C = 5 / 9 * (F - 32)) : F = 122 :=
by
  sorry

end celsius_to_fahrenheit_l359_35982


namespace negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l359_35901

open Real

theorem negation_of_exists_sin_gt_one_equiv_forall_sin_le_one :
  (¬ (∃ x : ℝ, sin x > 1)) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l359_35901


namespace digits_solution_l359_35949

noncomputable def validate_reverse_multiplication
  (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
  (E * 10000 + D * 1000 + C * 100 + B * 10 + A)

theorem digits_solution :
  validate_reverse_multiplication 2 1 9 7 8 :=
by
  sorry

end digits_solution_l359_35949


namespace cubed_ge_sqrt_ab_squared_l359_35987

theorem cubed_ge_sqrt_ab_squared (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^3 + b^3 ≥ (ab)^(1/2) * (a^2 + b^2) :=
sorry

end cubed_ge_sqrt_ab_squared_l359_35987


namespace first_programmer_loses_l359_35993

noncomputable def programSequence : List ℕ :=
  List.range 1999 |>.map (fun i => 2^i)

def validMove (sequence : List ℕ) (move : List ℕ) : Prop :=
  move.length = 5 ∧ move.all (λ i => i < sequence.length ∧ sequence.get! i > 0)

def applyMove (sequence : List ℕ) (move : List ℕ) : List ℕ :=
  move.foldl
    (λ seq i => seq.set i (seq.get! i - 1))
    sequence

def totalWeight (sequence : List ℕ) : ℕ :=
  sequence.foldl (· + ·) 0

theorem first_programmer_loses : ∀ seq moves,
  seq = programSequence →
  (∀ move, validMove seq move → False) →
  applyMove seq moves = seq →
  totalWeight seq = 2^1999 - 1 :=
by
  intro seq moves h_seq h_valid_move h_apply_move
  sorry

end first_programmer_loses_l359_35993


namespace amy_spent_32_l359_35900

theorem amy_spent_32 (x: ℝ) (h1: 0.15 * x + 1.6 * x + x = 55) : 1.6 * x = 32 :=
by
  sorry

end amy_spent_32_l359_35900


namespace min_value_2a_plus_b_l359_35945

theorem min_value_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (1/a) + (2/b) = 1): 2 * a + b = 8 :=
sorry

end min_value_2a_plus_b_l359_35945


namespace evaluate_complex_expression_l359_35988

noncomputable def expression := 
  Complex.mk (-1) (Real.sqrt 3) / 2

noncomputable def conjugate_expression := 
  Complex.mk (-1) (-Real.sqrt 3) / 2

theorem evaluate_complex_expression :
  (expression ^ 12 + conjugate_expression ^ 12) = 2 := by
  sorry

end evaluate_complex_expression_l359_35988


namespace quadratic_symmetry_l359_35994

def quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

theorem quadratic_symmetry (b c : ℝ) :
  let f := quadratic b c
  (f 2) < (f 1) ∧ (f 1) < (f 4) :=
by
  sorry

end quadratic_symmetry_l359_35994


namespace conic_section_eccentricities_cubic_l359_35931

theorem conic_section_eccentricities_cubic : 
  ∃ (e1 e2 e3 : ℝ), 
    (e1 = 1) ∧ 
    (0 < e2 ∧ e2 < 1) ∧ 
    (e3 > 1) ∧ 
    2 * e1^3 - 7 * e1^2 + 7 * e1 - 2 = 0 ∧
    2 * e2^3 - 7 * e2^2 + 7 * e2 - 2 = 0 ∧
    2 * e3^3 - 7 * e3^2 + 7 * e3 - 2 = 0 := 
by
  sorry

end conic_section_eccentricities_cubic_l359_35931


namespace adam_more_apples_than_combined_l359_35906

def adam_apples : Nat := 10
def jackie_apples : Nat := 2
def michael_apples : Nat := 5

theorem adam_more_apples_than_combined : 
  adam_apples - (jackie_apples + michael_apples) = 3 :=
by
  sorry

end adam_more_apples_than_combined_l359_35906


namespace cosine_identity_l359_35942

variable (α : ℝ)

theorem cosine_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) : 
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cosine_identity_l359_35942


namespace solve_for_m_l359_35943

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end solve_for_m_l359_35943


namespace radius_of_third_circle_l359_35940

open Real

theorem radius_of_third_circle (r : ℝ) :
  let r_large := 40
  let r_small := 25
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  let region_area := area_large - area_small
  let half_region_area := region_area / 2
  let third_circle_area := π * r^2
  (third_circle_area = half_region_area) -> r = 15 * sqrt 13 :=
by
  sorry

end radius_of_third_circle_l359_35940


namespace part_1_part_2_part_3_l359_35902

def whiteHorseNumber (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem part_1 : 
  whiteHorseNumber (-2) (-4) 1 = -5/3 :=
by sorry

theorem part_2 : 
  max (whiteHorseNumber (-2) (-4) 1) (max (whiteHorseNumber (-2) 1 (-4)) 
  (max (whiteHorseNumber (-4) (-2) 1) (max (whiteHorseNumber (-4) 1 (-2)) 
  (max (whiteHorseNumber 1 (-4) (-2)) (whiteHorseNumber 1 (-2) (-4)) )))) = 2/3 :=
by sorry

theorem part_3 (x : ℚ) (h : ∃a b c : ℚ, a = -1 ∧ b = 6 ∧ c = x ∧ whiteHorseNumber a b c = 2) : 
  x = -7 ∨ x = 8 :=
by sorry

end part_1_part_2_part_3_l359_35902


namespace distance_of_ladder_to_building_l359_35916

theorem distance_of_ladder_to_building :
  ∀ (c a b : ℕ), c = 25 ∧ a = 20 ∧ (a^2 + b^2 = c^2) → b = 15 :=
by
  intros c a b h
  rcases h with ⟨hc, ha, hpyth⟩
  have h1 : c = 25 := hc
  have h2 : a = 20 := ha
  have h3 : a^2 + b^2 = c^2 := hpyth
  sorry

end distance_of_ladder_to_building_l359_35916


namespace inequality_holds_for_all_reals_l359_35911

theorem inequality_holds_for_all_reals (x : ℝ) : 
  7 / 20 + |3 * x - 2 / 5| ≥ 1 / 4 :=
sorry

end inequality_holds_for_all_reals_l359_35911


namespace find_all_a_l359_35997

def digit_sum_base_4038 (n : ℕ) : ℕ :=
  n.digits 4038 |>.sum

def is_good (n : ℕ) : Prop :=
  2019 ∣ digit_sum_base_4038 n

def is_bad (n : ℕ) : Prop :=
  ¬ is_good n

def satisfies_condition (seq : ℕ → ℕ) (a : ℝ) : Prop :=
  (∀ n, seq n ≤ a * n) ∧ ∀ n, seq n = seq (n + 1) + 1

theorem find_all_a (a : ℝ) (h1 : 1 ≤ a) :
  (∀ seq, (∀ n m, n ≠ m → seq n ≠ seq m) → satisfies_condition seq a →
    ∃ n_infinitely, is_bad (seq n_infinitely)) ↔ a < 2019 := sorry

end find_all_a_l359_35997


namespace find_a_l359_35977

theorem find_a (a b c d : ℕ) (h1 : 2 * a + 2 = b) (h2 : 2 * b + 2 = c) (h3 : 2 * c + 2 = d) (h4 : 2 * d + 2 = 62) : a = 2 :=
by
  sorry

end find_a_l359_35977


namespace inequality_any_k_l359_35950

theorem inequality_any_k (x y z : ℝ) (k : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) : 
  x ^ (-k : ℤ) + y ^ (-k : ℤ) + z ^ (-k : ℤ) ≥ x ^ k + y ^ k + z ^ k :=
sorry

end inequality_any_k_l359_35950


namespace white_water_addition_l359_35915

theorem white_water_addition :
  ∃ (W H I T E A R : ℕ), 
  W ≠ H ∧ W ≠ I ∧ W ≠ T ∧ W ≠ E ∧ W ≠ A ∧ W ≠ R ∧
  H ≠ I ∧ H ≠ T ∧ H ≠ E ∧ H ≠ A ∧ H ≠ R ∧
  I ≠ T ∧ I ≠ E ∧ I ≠ A ∧ I ≠ R ∧
  T ≠ E ∧ T ≠ A ∧ T ≠ R ∧
  E ≠ A ∧ E ≠ R ∧
  A ≠ R ∧
  W = 8 ∧ I = 6 ∧ P = 1 ∧ C = 9 ∧ N = 0 ∧
  (10000 * W + 1000 * H + 100 * I + 10 * T + E) + 
  (10000 * W + 1000 * A + 100 * T + 10 * E + R) = 169069 :=
by 
  sorry

end white_water_addition_l359_35915


namespace find_number_l359_35933

theorem find_number (x : ℝ) (h : (((x + 1.4) / 3 - 0.7) * 9 = 5.4)) : x = 2.5 :=
by 
  sorry

end find_number_l359_35933


namespace equal_or_equal_exponents_l359_35907

theorem equal_or_equal_exponents
  (a b c p q r : ℕ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h1 : a^p + b^q + c^r = a^q + b^r + c^p)
  (h2 : a^q + b^r + c^p = a^r + b^p + c^q) :
  a = b ∧ b = c ∧ c = a ∨ p = q ∧ q = r ∧ r = p :=
  sorry

end equal_or_equal_exponents_l359_35907


namespace tangent_length_from_A_to_circle_l359_35934

noncomputable def point_A_polar : (ℝ × ℝ) := (6, Real.pi)
noncomputable def circle_eq_polar (θ : ℝ) : ℝ := -4 * Real.cos θ

theorem tangent_length_from_A_to_circle : 
  ∃ (length : ℝ), length = 2 * Real.sqrt 3 ∧ 
  (∃ (ρ θ : ℝ), point_A_polar = (6, Real.pi) ∧ ρ = circle_eq_polar θ) :=
sorry

end tangent_length_from_A_to_circle_l359_35934


namespace minimum_value_proof_l359_35979

variables {A B C : ℝ}
variable (triangle_ABC : 
  ∀ {A B C : ℝ}, 
  (A > 0 ∧ A < π / 2) ∧ 
  (B > 0 ∧ B < π / 2) ∧ 
  (C > 0 ∧ C < π / 2))

noncomputable def minimum_value (A B C : ℝ) :=
  3 * (Real.tan B) * (Real.tan C) + 
  2 * (Real.tan A) * (Real.tan C) + 
  1 * (Real.tan A) * (Real.tan B)

theorem minimum_value_proof (h : 
  ∀ (A B C : ℝ), 
  (1 / (Real.tan A * Real.tan B)) + 
  (1 / (Real.tan B * Real.tan C)) + 
  (1 / (Real.tan C * Real.tan A)) = 1) 
  : minimum_value A B C = 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_proof_l359_35979


namespace evaluate_expression_l359_35946

theorem evaluate_expression :
  (42 / (9 - 3 * 2)) * 4 = 56 :=
by
  sorry

end evaluate_expression_l359_35946


namespace minimum_value_f_l359_35937

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l359_35937


namespace side_length_of_square_l359_35923

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l359_35923


namespace min_value_of_f_in_interval_l359_35905

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem min_value_of_f_in_interval (k : ℝ) :
  (f 1 k = -k ∧ k ≤ 2) ∨ 
  (∃ k', k' = 2 ∧ f (k'/2) k = - (k'^2) / 4 - 1 ∧ 2 < k ∧ k < 8) ∨ 
  (f 4 k = 15 - 4 * k ∧ k ≥ 8) :=
by sorry

end min_value_of_f_in_interval_l359_35905


namespace same_curve_option_B_l359_35961

theorem same_curve_option_B : 
  (∀ x y : ℝ, |y| = |x| ↔ y = x ∨ y = -x) ∧ (∀ x y : ℝ, y^2 = x^2 ↔ y = x ∨ y = -x) :=
by
  sorry

end same_curve_option_B_l359_35961


namespace janice_class_girls_l359_35999

theorem janice_class_girls : ∃ (g b : ℕ), (3 * b = 4 * g) ∧ (g + b + 2 = 32) ∧ (g = 13) := by
  sorry

end janice_class_girls_l359_35999


namespace number_of_terms_in_arithmetic_sequence_l359_35926

theorem number_of_terms_in_arithmetic_sequence 
  (a d n l : ℤ) (h1 : a = 7) (h2 : d = 2) (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : n = 70 := 
by sorry

end number_of_terms_in_arithmetic_sequence_l359_35926


namespace min_value_ge_8_min_value_8_at_20_l359_35922

noncomputable def min_value (x : ℝ) (h : x > 4) : ℝ := (x + 12) / Real.sqrt (x - 4)

theorem min_value_ge_8 (x : ℝ) (h : x > 4) : min_value x h ≥ 8 := sorry

theorem min_value_8_at_20 : min_value 20 (by norm_num) = 8 := sorry

end min_value_ge_8_min_value_8_at_20_l359_35922


namespace molecular_weight_of_compound_l359_35980

noncomputable def molecularWeight (Ca_wt : ℝ) (O_wt : ℝ) (H_wt : ℝ) (nCa : ℕ) (nO : ℕ) (nH : ℕ) : ℝ :=
  (nCa * Ca_wt) + (nO * O_wt) + (nH * H_wt)

theorem molecular_weight_of_compound :
  molecularWeight 40.08 15.999 1.008 1 2 2 = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l359_35980


namespace product_lcm_gcd_eq_128_l359_35972

theorem product_lcm_gcd_eq_128 : (Int.gcd 8 16) * (Int.lcm 8 16) = 128 :=
by
  sorry

end product_lcm_gcd_eq_128_l359_35972


namespace binary_digit_sum_property_l359_35967

def binary_digit_sum (n : Nat) : Nat :=
  n.digits 2 |>.foldr (· + ·) 0

theorem binary_digit_sum_property (k : Nat) (h_pos : 0 < k) :
  (Finset.range (2^k)).sum (λ n => binary_digit_sum (n + 1)) = 2^(k - 1) * k + 1 := 
sorry

end binary_digit_sum_property_l359_35967


namespace calculate_expression_l359_35935

variable {a : ℝ}

theorem calculate_expression (h₁ : a ≠ 0) (h₂ : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := 
sorry

end calculate_expression_l359_35935


namespace maria_sandwich_count_l359_35930

open Nat

noncomputable def numberOfSandwiches (meat_choices cheese_choices topping_choices : Nat) :=
  (choose meat_choices 2) * (choose cheese_choices 2) * (choose topping_choices 2)

theorem maria_sandwich_count : numberOfSandwiches 12 11 8 = 101640 := by
  sorry

end maria_sandwich_count_l359_35930


namespace max_abs_sum_value_l359_35991

noncomputable def max_abs_sum (x y : ℝ) : ℝ := |x| + |y|

theorem max_abs_sum_value (x y : ℝ) (h : x^2 + y^2 = 4) : max_abs_sum x y ≤ 2 * Real.sqrt 2 :=
by {
  sorry
}

end max_abs_sum_value_l359_35991


namespace N_subset_M_l359_35966

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x - 2 = 0}

theorem N_subset_M : N ⊆ M := sorry

end N_subset_M_l359_35966


namespace volume_of_new_pyramid_is_108_l359_35986

noncomputable def volume_of_cut_pyramid : ℝ :=
  let base_edge_length := 12 * Real.sqrt 2
  let slant_edge_length := 15
  let cut_height := 4.5
  -- Calculate the height of the original pyramid using Pythagorean theorem
  let original_height := Real.sqrt (slant_edge_length^2 - (base_edge_length/2 * Real.sqrt 2)^2)
  -- Calculate the remaining height of the smaller pyramid
  let remaining_height := original_height - cut_height
  -- Calculate the scale factor
  let scale_factor := remaining_height / original_height
  -- New base edge length
  let new_base_edge_length := base_edge_length * scale_factor
  -- New base area
  let new_base_area := (new_base_edge_length)^2
  -- Volume of the new pyramid
  (1 / 3) * new_base_area * remaining_height

-- Define the statement to prove
theorem volume_of_new_pyramid_is_108 :
  volume_of_cut_pyramid = 108 :=
by
  sorry

end volume_of_new_pyramid_is_108_l359_35986


namespace abs_diff_inequality_l359_35944

theorem abs_diff_inequality (a b c h : ℝ) (hab : |a - c| < h) (hbc : |b - c| < h) : |a - b| < 2 * h := 
by
  sorry

end abs_diff_inequality_l359_35944


namespace motorist_spent_on_petrol_l359_35910

def original_price_per_gallon : ℝ := 5.56
def reduction_percentage : ℝ := 0.10
def new_price_per_gallon := original_price_per_gallon - (0.10 * original_price_per_gallon)
def gallons_more_after_reduction : ℝ := 5

theorem motorist_spent_on_petrol (X : ℝ) 
  (h1 : new_price_per_gallon = original_price_per_gallon - (reduction_percentage * original_price_per_gallon))
  (h2 : (X / new_price_per_gallon) - (X / original_price_per_gallon) = gallons_more_after_reduction) :
  X = 250.22 :=
by
  sorry

end motorist_spent_on_petrol_l359_35910


namespace find_m_positive_root_l359_35941

theorem find_m_positive_root :
  (∃ x > 0, (x - 4) / (x - 3) - m - 4 = m / (3 - x)) → m = 1 :=
by
  sorry

end find_m_positive_root_l359_35941


namespace square_reciprocal_sum_integer_l359_35951

theorem square_reciprocal_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^2 + 1/a^2 = m := by
  sorry

end square_reciprocal_sum_integer_l359_35951


namespace smallest_k_divides_l359_35953

-- Given Problem: z^{12} + z^{11} + z^8 + z^7 + z^6 + z^3 + 1 divides z^k - 1
theorem smallest_k_divides (
  k : ℕ
) : (∀ z : ℂ, (z ^ 12 + z ^ 11 + z ^ 8 + z ^ 7 + z ^ 6 + z ^ 3 + 1) ∣ (z ^ k - 1) ↔ k = 182) :=
sorry

end smallest_k_divides_l359_35953


namespace lottery_probability_l359_35954

theorem lottery_probability (x_1 x_2 x_3 x_4 : ℝ) (p : ℝ) (h0 : 0 < p ∧ p < 1) : 
  x_1 = p * x_3 → 
  x_2 = p * x_4 + (1 - p) * x_1 → 
  x_3 = p + (1 - p) * x_2 → 
  x_4 = p + (1 - p) * x_3 → 
  x_2 = 0.19 :=
by
  sorry

end lottery_probability_l359_35954


namespace value_of_f_3x_minus_7_l359_35939

def f (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_f_3x_minus_7 (x : ℝ) : f (3 * x - 7) = 9 * x - 16 :=
by
  -- Proof goes here
  sorry

end value_of_f_3x_minus_7_l359_35939


namespace min_value_expr_l359_35917

theorem min_value_expr (a : ℝ) (h₁ : 0 < a) (h₂ : a < 3) : 
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → x < 3 → (1/x + 9/(3 - x)) ≥ m) ∧ m = 16 / 3 :=
sorry

end min_value_expr_l359_35917


namespace expected_participants_in_2005_l359_35904

open Nat

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 1.2
def num_years : ℕ := 5
def expected_participants_2005 : ℚ := 1244

theorem expected_participants_in_2005 :
  (initial_participants : ℚ) * annual_increase_rate ^ num_years = expected_participants_2005 := by
  sorry

end expected_participants_in_2005_l359_35904


namespace exponent_form_l359_35927

theorem exponent_form (x : ℕ) (k : ℕ) : (3^x) % 10 = 7 ↔ x = 4 * k + 3 :=
by
  sorry

end exponent_form_l359_35927


namespace obtuse_triangle_of_sin_cos_sum_l359_35920

theorem obtuse_triangle_of_sin_cos_sum
  (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h_eq : Real.sin A + Real.cos A = 12 / 25) :
  π / 2 < A ∧ A < π :=
sorry

end obtuse_triangle_of_sin_cos_sum_l359_35920


namespace geometric_sequence_seventh_term_l359_35956

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end geometric_sequence_seventh_term_l359_35956


namespace total_students_in_class_l359_35995

theorem total_students_in_class (R S : ℕ)
  (h1 : 2 + 12 * 1 + 12 * 2 + 3 * R = S * 2)
  (h2 : S = 2 + 12 + 12 + R) :
  S = 42 :=
by
  sorry

end total_students_in_class_l359_35995


namespace shaded_region_area_l359_35960

noncomputable def side_length := 1 -- Length of each side of the squares, in cm.

-- Conditions
def top_square_center_above_edge : Prop := 
  ∀ square1 square2 square3 : ℝ, square3 = (square1 + square2) / 2

-- Question: Area of the shaded region
def area_of_shaded_region := 1 -- area in cm^2

-- Lean 4 Statement
theorem shaded_region_area :
  top_square_center_above_edge → area_of_shaded_region = 1 := 
by
  sorry

end shaded_region_area_l359_35960


namespace determinant_sum_is_34_l359_35955

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![5, -2],
  ![3, 4]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 3],
  ![-1, 2]
]

-- Prove the determinant of the sum of A and B is 34
theorem determinant_sum_is_34 : Matrix.det (A + B) = 34 := by
  sorry

end determinant_sum_is_34_l359_35955


namespace simplify_expression_l359_35948

theorem simplify_expression (a : ℤ) (h_range : -3 < a ∧ a ≤ 0) (h_notzero : a ≠ 0) (h_notone : a ≠ 1 ∧ a ≠ -1) :
  (a - (2 * a - 1) / a) / (1 / a - a) = -3 :=
by
  have h_eq : (a - (2 * a - 1) / a) / (1 / a - a) = (1 - a) / (1 + a) :=
    sorry
  have h_a_neg_two : a = -2 :=
    sorry
  rw [h_eq, h_a_neg_two]
  sorry


end simplify_expression_l359_35948


namespace largest_integer_n_neg_l359_35962

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l359_35962


namespace max_chips_can_be_removed_l359_35989

theorem max_chips_can_be_removed (initial_chips : (Fin 10) × (Fin 10) → ℕ) 
  (condition : ∀ i j, initial_chips (i, j) = 1) : 
    ∃ removed_chips : ℕ, removed_chips = 90 :=
by
  sorry

end max_chips_can_be_removed_l359_35989


namespace union_of_M_and_N_l359_35971

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def compl_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_of_M_and_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} :=
sorry

end union_of_M_and_N_l359_35971


namespace min_value_of_f_l359_35929

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_f :
  ∀ (x : ℝ), -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≥ f (-Real.pi / 2) :=
by
  intro x hx
  -- conditions are given, statement declared, but proof is not provided
  sorry

end min_value_of_f_l359_35929


namespace group_B_same_order_l359_35996

-- Definitions for the expressions in each group
def expr_A1 := 2 * 9 / 3
def expr_A2 := 2 + 9 * 3

def expr_B1 := 36 - 9 + 5
def expr_B2 := 36 / 6 * 5

def expr_C1 := 56 / 7 * 5
def expr_C2 := 56 + 7 * 5

-- Theorem stating that Group B expressions have the same order of operations
theorem group_B_same_order : (expr_B1 = expr_B2) := 
  sorry

end group_B_same_order_l359_35996


namespace part1_part2_l359_35928

-- Definitions and conditions
variables {A B C a b c : ℝ}
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A)) -- Given condition

-- Part (1): If A = 2B, then find C
theorem part1 (h2 : A = 2 * B) : C = (5 / 8) * π := by
  sorry

-- Part (2): Prove that 2a² = b² + c²
theorem part2 : 2 * a^2 = b^2 + c^2 := by
  sorry

end part1_part2_l359_35928


namespace number_of_distinct_configurations_l359_35918

-- Definitions of the problem conditions
structure CubeConfig where
  white_cubes : Finset (Fin 8)
  blue_cubes : Finset (Fin 8)
  condition_1 : white_cubes.card = 5
  condition_2 : blue_cubes.card = 3
  condition_3 : ∀ x ∈ white_cubes, x ∉ blue_cubes

def distinctConfigCount (configs : Finset CubeConfig) : ℕ :=
  (configs.filter (λ config => 
    config.white_cubes.card = 5 ∧
    config.blue_cubes.card = 3 ∧
    (∀ x ∈ config.white_cubes, x ∉ config.blue_cubes)
  )).card

-- Theorem stating the correct number of distinct configurations
theorem number_of_distinct_configurations : distinctConfigCount ∅ = 5 := 
  sorry

end number_of_distinct_configurations_l359_35918


namespace largest_common_value_less_than_1000_l359_35914

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 5 + 8 * m) ∧ 
            (∀ b : ℕ, (b < 1000 ∧ (∃ n : ℤ, b = 4 + 5 * n) ∧ (∃ m : ℤ, b = 5 + 8 * m)) → b ≤ a) :=
sorry

end largest_common_value_less_than_1000_l359_35914


namespace find_a_squared_plus_b_squared_l359_35921

theorem find_a_squared_plus_b_squared 
  (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 104) : 
  a^2 + b^2 = 1392 := 
by 
  sorry

end find_a_squared_plus_b_squared_l359_35921


namespace bobby_shoes_cost_l359_35975

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end bobby_shoes_cost_l359_35975


namespace polynomial_coeff_sum_neg_33_l359_35984

theorem polynomial_coeff_sum_neg_33
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 - 3 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -33 :=
by sorry

end polynomial_coeff_sum_neg_33_l359_35984


namespace triangle_area_is_96_l359_35981

/-- Given a square with side length 8 and an overlapping area that is both three-quarters
    of the area of the square and half of the area of a triangle, prove the triangle's area is 96. -/
theorem triangle_area_is_96 (a : ℕ) (area_of_square : ℕ) (overlapping_area : ℕ) (area_of_triangle : ℕ) 
  (h1 : a = 8) 
  (h2 : area_of_square = a * a) 
  (h3 : overlapping_area = (3 * area_of_square) / 4) 
  (h4 : overlapping_area = area_of_triangle / 2) : 
  area_of_triangle = 96 := 
by 
  sorry

end triangle_area_is_96_l359_35981


namespace car_speed_first_hour_l359_35932

theorem car_speed_first_hour 
  (x : ℝ)  -- Speed of the car in the first hour.
  (s2 : ℝ)  -- Speed of the car in the second hour is fixed at 40 km/h.
  (avg_speed : ℝ)  -- Average speed over two hours is 65 km/h.
  (h1 : s2 = 40)  -- speed in the second hour is 40 km/h.
  (h2 : avg_speed = 65)  -- average speed is 65 km/h
  (h3 : avg_speed = (x + s2) / 2)  -- definition of average speed
  : x = 90 := 
  sorry

end car_speed_first_hour_l359_35932


namespace number_of_ways_to_put_7_balls_in_2_boxes_l359_35919

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l359_35919


namespace marbles_total_l359_35908

theorem marbles_total (yellow blue red total : ℕ)
  (hy : yellow = 5)
  (h_ratio : blue / red = 3 / 4)
  (h_red : red = yellow + 3)
  (h_total : total = yellow + blue + red) : total = 19 :=
by
  sorry

end marbles_total_l359_35908


namespace research_development_success_l359_35998

theorem research_development_success 
  (P_A : ℝ)  -- probability of Team A successfully developing a product
  (P_B : ℝ)  -- probability of Team B successfully developing a product
  (independent : Bool)  -- independence condition (dummy for clarity)
  (h1 : P_A = 2/3)
  (h2 : P_B = 3/5) 
  (h3 : independent = true) :
  (1 - (1 - P_A) * (1 - P_B) = 13/15) :=
by
  sorry

end research_development_success_l359_35998


namespace solve_phi_eq_l359_35925

noncomputable def φ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat := (1 - Real.sqrt 5) / 2
noncomputable def F : ℕ → ℤ
| n =>
  if n = 0 then 0
  else if n = 1 then 1
  else F (n - 1) + F (n - 2)

theorem solve_phi_eq (n : ℕ) :
  ∃ x y : ℤ, x * φ ^ (n + 1) + y * φ^n = 1 ∧ 
    x = (-1 : ℤ)^(n+1) * F n ∧ y = (-1 : ℤ)^n * F (n + 1) := by
  sorry

end solve_phi_eq_l359_35925


namespace stratified_sampling_2nd_year_students_l359_35924

theorem stratified_sampling_2nd_year_students
  (students_1st_year : ℕ) (students_2nd_year : ℕ) (students_3rd_year : ℕ) (total_sample_size : ℕ) :
  students_1st_year = 1000 ∧ students_2nd_year = 800 ∧ students_3rd_year = 700 ∧ total_sample_size = 100 →
  (students_2nd_year * total_sample_size / (students_1st_year + students_2nd_year + students_3rd_year) = 32) :=
by
  intro h
  sorry

end stratified_sampling_2nd_year_students_l359_35924


namespace real_numbers_satisfy_relation_l359_35959

theorem real_numbers_satisfy_relation (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end real_numbers_satisfy_relation_l359_35959


namespace find_x_eq_728_l359_35909

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l359_35909


namespace annual_feeding_cost_is_correct_l359_35947

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l359_35947


namespace at_most_n_maximum_distance_pairs_l359_35938

theorem at_most_n_maximum_distance_pairs (n : ℕ) (h : n > 2) 
(points : Fin n → ℝ × ℝ) :
  ∃ (maxDistPairs : Finset (Fin n × Fin n)), (maxDistPairs.card ≤ n) ∧ 
  ∀ (p1 p2 : Fin n), (p1, p2) ∈ maxDistPairs → 
  (∀ (q1 q2 : Fin n), dist (points q1) (points q2) ≤ dist (points p1) (points p2)) :=
sorry

end at_most_n_maximum_distance_pairs_l359_35938


namespace system_of_equations_solution_system_of_inequalities_solution_l359_35978

theorem system_of_equations_solution (x y : ℝ) :
  (3 * x - 4 * y = 1) → (5 * x + 2 * y = 6) → 
  x = 1 ∧ y = 0.5 := by
  sorry

theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x + 6 > 0) → (x - 2 < -x) → 
  -2 < x ∧ x < 1 := by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l359_35978


namespace total_peaches_l359_35983

theorem total_peaches (num_baskets num_red num_green : ℕ)
    (h1 : num_baskets = 11)
    (h2 : num_red = 10)
    (h3 : num_green = 18) : (num_red + num_green) * num_baskets = 308 := by
  sorry

end total_peaches_l359_35983


namespace rotate_D_90_clockwise_l359_35970

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℤ) : Point ℤ :=
  ⟨p.y, -p.x⟩

def D : Point ℤ := ⟨-3, 2⟩
def E : Point ℤ := ⟨0, 5⟩
def F : Point ℤ := ⟨0, 2⟩

theorem rotate_D_90_clockwise :
  rotate_90_clockwise D = Point.mk 2 (-3) :=
by
  sorry

end rotate_D_90_clockwise_l359_35970


namespace area_of_triangle_l359_35992

theorem area_of_triangle (h : ℝ) (a : ℝ) (b : ℝ) (hypotenuse : h = 13) (side_a : a = 5) (right_triangle : a^2 + b^2 = h^2) : 
  ∃ (area : ℝ), area = 30 := 
by
  sorry

end area_of_triangle_l359_35992


namespace cricket_run_rate_l359_35985

theorem cricket_run_rate (r : ℝ) (o₁ T o₂ : ℕ) (r₁ : ℝ) (Rₜ : ℝ) : 
  r = 4.8 ∧ o₁ = 10 ∧ T = 282 ∧ o₂ = 40 ∧ r₁ = (T - r * o₁) / o₂ → Rₜ = 5.85 := 
by 
  intros h
  sorry

end cricket_run_rate_l359_35985


namespace winning_vote_majority_l359_35969

theorem winning_vote_majority (h1 : 0.70 * 900 = 630)
                             (h2 : 0.30 * 900 = 270) :
  630 - 270 = 360 :=
by
  sorry

end winning_vote_majority_l359_35969


namespace afternoon_sales_l359_35952

variable (x y : ℕ)

theorem afternoon_sales (hx : y = 2 * x) (hy : x + y = 390) : y = 260 := by
  sorry

end afternoon_sales_l359_35952


namespace complement_intersection_l359_35963

noncomputable def U : Set Real := Set.univ
noncomputable def M : Set Real := { x : Real | Real.log x < 0 }
noncomputable def N : Set Real := { x : Real | (1 / 2) ^ x ≥ Real.sqrt (1 / 2) }

theorem complement_intersection (U M N : Set Real) : 
  (Set.compl M ∩ N) = Set.Iic 0 :=
by
  sorry

end complement_intersection_l359_35963


namespace average_cost_is_thirteen_l359_35903

noncomputable def averageCostPerPen (pensCost shippingCost : ℝ) (totalPens : ℕ) : ℕ :=
  Nat.ceil ((pensCost + shippingCost) * 100 / totalPens)

theorem average_cost_is_thirteen :
  averageCostPerPen 29.85 8.10 300 = 13 :=
by
  sorry

end average_cost_is_thirteen_l359_35903


namespace canal_depth_l359_35965

-- Define the problem parameters
def top_width : ℝ := 6
def bottom_width : ℝ := 4
def cross_section_area : ℝ := 10290

-- Define the theorem to prove the depth of the canal
theorem canal_depth :
  (1 / 2) * (top_width + bottom_width) * h = cross_section_area → h = 2058 :=
by sorry

end canal_depth_l359_35965


namespace Dawn_commissioned_paintings_l359_35990

theorem Dawn_commissioned_paintings (time_per_painting : ℕ) (total_earnings : ℕ) (earnings_per_hour : ℕ) 
  (h1 : time_per_painting = 2) 
  (h2 : total_earnings = 3600) 
  (h3 : earnings_per_hour = 150) : 
  (total_earnings / (time_per_painting * earnings_per_hour) = 12) :=
by 
  sorry

end Dawn_commissioned_paintings_l359_35990
