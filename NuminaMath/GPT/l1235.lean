import Mathlib

namespace NUMINAMATH_GPT_ratio_volumes_l1235_123597

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_volumes_l1235_123597


namespace NUMINAMATH_GPT_total_boxes_stacked_l1235_123595

/-- Definitions used in conditions --/
def box_width : ℕ := 1
def box_length : ℕ := 1
def land_width : ℕ := 44
def land_length : ℕ := 35
def first_day_layers : ℕ := 7
def second_day_layers : ℕ := 3

/-- Theorem stating the number of boxes stacked in two days --/
theorem total_boxes_stacked : first_day_layers * (land_width * land_length) + second_day_layers * (land_width * land_length) = 15400 := by
  sorry

end NUMINAMATH_GPT_total_boxes_stacked_l1235_123595


namespace NUMINAMATH_GPT_correct_description_of_sperm_l1235_123545

def sperm_carries_almost_no_cytoplasm (sperm : Type) : Prop := sorry

theorem correct_description_of_sperm : sperm_carries_almost_no_cytoplasm sperm := 
sorry

end NUMINAMATH_GPT_correct_description_of_sperm_l1235_123545


namespace NUMINAMATH_GPT_negation_of_square_zero_l1235_123539

variable {m : ℝ}

def is_positive (m : ℝ) : Prop := m > 0
def square_is_zero (m : ℝ) : Prop := m^2 = 0

theorem negation_of_square_zero (h : ∀ m, is_positive m → square_is_zero m) :
  ∀ m, ¬ is_positive m → ¬ square_is_zero m := 
sorry

end NUMINAMATH_GPT_negation_of_square_zero_l1235_123539


namespace NUMINAMATH_GPT_sum_binomial_coefficients_l1235_123528

theorem sum_binomial_coefficients (a b : ℕ) (h1 : a = 2^3) (h2 : b = (2 + 1)^3) : a + b = 35 :=
by
  sorry

end NUMINAMATH_GPT_sum_binomial_coefficients_l1235_123528


namespace NUMINAMATH_GPT_sum_of_common_ratios_of_sequences_l1235_123544

def arithmetico_geometric_sequence (a b c : ℕ → ℝ) (r : ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n + d ∧ b (n + 1) = r * b n + d

theorem sum_of_common_ratios_of_sequences {m n : ℝ}
    {a1 a2 a3 b1 b2 b3 : ℝ}
    (p q : ℝ)
    (h_a1 : a1 = m)
    (h_a2 : a2 = m * p + 5)
    (h_a3 : a3 = m * p^2 + 5 * p + 5)
    (h_b1 : b1 = n)
    (h_b2 : b2 = n * q + 5)
    (h_b3 : b3 = n * q^2 + 5 * q + 5)
    (h_cond : a3 - b3 = 3 * (a2 - b2)) :
    p + q = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_of_sequences_l1235_123544


namespace NUMINAMATH_GPT_f_eq_2x_pow_5_l1235_123526

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_eq_2x_pow_5 (x : ℝ) : f x = (2*x)^5 :=
by
  sorry

end NUMINAMATH_GPT_f_eq_2x_pow_5_l1235_123526


namespace NUMINAMATH_GPT_total_pepper_weight_l1235_123598

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end NUMINAMATH_GPT_total_pepper_weight_l1235_123598


namespace NUMINAMATH_GPT_domain_of_f_l1235_123596

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f : {x : ℝ | (x > -1) ∧ (x ≠ 0) ∧ (x ∈ [-3, 3])} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1235_123596


namespace NUMINAMATH_GPT_book_cost_in_cny_l1235_123566

-- Conditions
def usd_to_nad : ℝ := 7      -- One US dollar to Namibian dollar
def usd_to_cny : ℝ := 6      -- One US dollar to Chinese yuan
def book_cost_nad : ℝ := 168 -- Cost of the book in Namibian dollars

-- Statement to prove
theorem book_cost_in_cny : book_cost_nad * (usd_to_cny / usd_to_nad) = 144 :=
sorry

end NUMINAMATH_GPT_book_cost_in_cny_l1235_123566


namespace NUMINAMATH_GPT_sum_S_17_33_50_l1235_123522

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then - (n / 2)
  else (n / 2) + 1

theorem sum_S_17_33_50 : (S 17) + (S 33) + (S 50) = 1 := by
  sorry

end NUMINAMATH_GPT_sum_S_17_33_50_l1235_123522


namespace NUMINAMATH_GPT_triangle_side_ratio_range_l1235_123588

theorem triangle_side_ratio_range (A B C a b c : ℝ) (h1 : A + 4 * B = 180) (h2 : C = 3 * B) (h3 : 0 < B ∧ B < 45) 
  (h4 : a / b = Real.sin (4 * B) / Real.sin B) : 
  1 < a / b ∧ a / b < 3 := 
sorry

end NUMINAMATH_GPT_triangle_side_ratio_range_l1235_123588


namespace NUMINAMATH_GPT_wire_cut_l1235_123550

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end NUMINAMATH_GPT_wire_cut_l1235_123550


namespace NUMINAMATH_GPT_origin_not_in_A_point_M_in_A_l1235_123564

def set_A : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ x + 2 * y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2 * x + y - 5 ≤ 0}

theorem origin_not_in_A : (0, 0) ∉ set_A := by
  sorry

theorem point_M_in_A : (1, 1) ∈ set_A := by
  sorry

end NUMINAMATH_GPT_origin_not_in_A_point_M_in_A_l1235_123564


namespace NUMINAMATH_GPT_abs_ineq_l1235_123537

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end NUMINAMATH_GPT_abs_ineq_l1235_123537


namespace NUMINAMATH_GPT_primes_or_prime_squares_l1235_123541

theorem primes_or_prime_squares (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d, d ∣ n → d > 1 → (d - 1) ∣ (n - 1)) : 
  (∃ p, Nat.Prime p ∧ (n = p ∨ n = p * p)) :=
by
  sorry

end NUMINAMATH_GPT_primes_or_prime_squares_l1235_123541


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l1235_123589

theorem cannot_form_right_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l1235_123589


namespace NUMINAMATH_GPT_units_digit_m_squared_plus_3_to_m_l1235_123575

theorem units_digit_m_squared_plus_3_to_m (m : ℕ) (h : m = 2021^2 + 3^2021) : (m^2 + 3^m) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_m_squared_plus_3_to_m_l1235_123575


namespace NUMINAMATH_GPT_crayons_total_l1235_123540

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end NUMINAMATH_GPT_crayons_total_l1235_123540


namespace NUMINAMATH_GPT_negation_of_statement_l1235_123587

theorem negation_of_statement (h: ∀ x : ℝ, |x| + x^2 ≥ 0) :
  ¬ (∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_statement_l1235_123587


namespace NUMINAMATH_GPT_arithmetic_sequence_thm_l1235_123580

theorem arithmetic_sequence_thm
  (a : ℕ → ℝ)
  (h1 : a 1 + a 4 + a 7 = 48)
  (h2 : a 2 + a 5 + a 8 = 40)
  (d : ℝ)
  (h3 : ∀ n, a (n + 1) = a n + d) :
  a 3 + a 6 + a 9 = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_thm_l1235_123580


namespace NUMINAMATH_GPT_boat_speed_l1235_123521

theorem boat_speed (v : ℝ) : 
  let rate_current := 7
  let distance := 35.93
  let time := 44 / 60
  (v + rate_current) * time = distance → v = 42 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_boat_speed_l1235_123521


namespace NUMINAMATH_GPT_determine_xyz_l1235_123579

theorem determine_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 23 / 3 := 
by { sorry }

end NUMINAMATH_GPT_determine_xyz_l1235_123579


namespace NUMINAMATH_GPT_final_number_independent_of_order_l1235_123533

theorem final_number_independent_of_order 
  (p q r : ℕ) : 
  ∃ k : ℕ, 
    (p % 2 ≠ 0 ∨ q % 2 ≠ 0 ∨ r % 2 ≠ 0) ∧ 
    (∀ (p' q' r' : ℕ), 
       p' + q' + r' = p + q + r → 
       p' % 2 = p % 2 ∧ q' % 2 = q % 2 ∧ r' % 2 = r % 2 → 
       (p' = 1 ∧ q' = 0 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 1 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 0 ∧ r' = 1) → 
       k = p ∨ k = q ∨ k = r) := 
sorry

end NUMINAMATH_GPT_final_number_independent_of_order_l1235_123533


namespace NUMINAMATH_GPT_cappuccino_cost_l1235_123584

theorem cappuccino_cost 
  (total_order_cost drip_price espresso_price latte_price syrup_price cold_brew_price total_other_cost : ℝ)
  (h1 : total_order_cost = 25)
  (h2 : drip_price = 2 * 2.25)
  (h3 : espresso_price = 3.50)
  (h4 : latte_price = 2 * 4.00)
  (h5 : syrup_price = 0.50)
  (h6 : cold_brew_price = 2 * 2.50)
  (h7 : total_other_cost = drip_price + espresso_price + latte_price + syrup_price + cold_brew_price) :
  total_order_cost - total_other_cost = 3.50 := 
by
  sorry

end NUMINAMATH_GPT_cappuccino_cost_l1235_123584


namespace NUMINAMATH_GPT_average_speed_l1235_123599

theorem average_speed (initial final time : ℕ) (h_initial : initial = 2002) (h_final : final = 2332) (h_time : time = 11) : 
  (final - initial) / time = 30 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1235_123599


namespace NUMINAMATH_GPT_scientific_notation_of_graphene_l1235_123548

theorem scientific_notation_of_graphene :
  0.00000000034 = 3.4 * 10^(-10) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_graphene_l1235_123548


namespace NUMINAMATH_GPT_most_likely_wins_l1235_123507

theorem most_likely_wins {N : ℕ} (h : N > 0) :
  let p := 1 / 2
  let n := 2 * N
  let E := n * p
  E = N := 
by
  sorry

end NUMINAMATH_GPT_most_likely_wins_l1235_123507


namespace NUMINAMATH_GPT_coeff_x2_in_PQ_is_correct_l1235_123505

variable (c : ℝ)

def P (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2 - 3 * x + 1
def Q (x : ℝ) : ℝ := 3 * x^3 + c * x^2 - 8 * x - 5

def coeff_x2 (x : ℝ) : ℝ := -20 - 2 * c

theorem coeff_x2_in_PQ_is_correct :
  (4 : ℝ) * (-5) + (-3) * c + c = -20 - 2 * c := by
  sorry

end NUMINAMATH_GPT_coeff_x2_in_PQ_is_correct_l1235_123505


namespace NUMINAMATH_GPT_students_with_no_preference_l1235_123555

def total_students : ℕ := 210
def prefer_mac : ℕ := 60
def equally_prefer_both (x : ℕ) : ℕ := x / 3

def no_preference_students : ℕ :=
  total_students - (prefer_mac + equally_prefer_both prefer_mac)

theorem students_with_no_preference :
  no_preference_students = 130 :=
by
  sorry

end NUMINAMATH_GPT_students_with_no_preference_l1235_123555


namespace NUMINAMATH_GPT_matrix_operation_value_l1235_123538

theorem matrix_operation_value : 
  let p := 4 
  let q := 5
  let r := 2
  let s := 3 
  (p * s - q * r) = 2 :=
by
  sorry

end NUMINAMATH_GPT_matrix_operation_value_l1235_123538


namespace NUMINAMATH_GPT_find_sum_of_money_l1235_123515

theorem find_sum_of_money (P : ℝ) (H1 : P * 0.18 * 2 - P * 0.12 * 2 = 840) : P = 7000 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_money_l1235_123515


namespace NUMINAMATH_GPT_isosceles_triangle_l1235_123510

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1235_123510


namespace NUMINAMATH_GPT_green_peaches_sum_l1235_123549

theorem green_peaches_sum (G1 G2 G3 : ℕ) : 
  (4 + G1) + (4 + G2) + (3 + G3) = 20 → G1 + G2 + G3 = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_green_peaches_sum_l1235_123549


namespace NUMINAMATH_GPT_servings_in_box_l1235_123554

theorem servings_in_box (total_cereal : ℕ) (serving_size : ℕ) (total_cereal_eq : total_cereal = 18) (serving_size_eq : serving_size = 2) :
  total_cereal / serving_size = 9 :=
by
  sorry

end NUMINAMATH_GPT_servings_in_box_l1235_123554


namespace NUMINAMATH_GPT_cone_volume_divided_by_pi_l1235_123562

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end NUMINAMATH_GPT_cone_volume_divided_by_pi_l1235_123562


namespace NUMINAMATH_GPT_roots_of_unity_cubic_l1235_123525

noncomputable def countRootsOfUnityCubic (c d e : ℤ) : ℕ := sorry

theorem roots_of_unity_cubic :
  ∃ (z : ℂ) (n : ℕ), (z^n = 1) ∧ (∃ (c d e : ℤ), z^3 + c * z^2 + d * z + e = 0)
  ∧ countRootsOfUnityCubic c d e = 12 :=
sorry

end NUMINAMATH_GPT_roots_of_unity_cubic_l1235_123525


namespace NUMINAMATH_GPT_fish_ratio_l1235_123590

theorem fish_ratio (B T S Bo : ℕ) 
  (hBilly : B = 10) 
  (hTonyBilly : T = 3 * B) 
  (hSarahTony : S = T + 5) 
  (hBobbySarah : Bo = 2 * S) 
  (hTotalFish : Bo + S + T + B = 145) : 
  T / B = 3 :=
by sorry

end NUMINAMATH_GPT_fish_ratio_l1235_123590


namespace NUMINAMATH_GPT_min_value_ineq_l1235_123572

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) : 
  (∀ z : ℝ, z = (4 / x + 1 / y) → z ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_min_value_ineq_l1235_123572


namespace NUMINAMATH_GPT_calculate_original_lemon_price_l1235_123593

variable (p_lemon_old p_lemon_new p_grape_old p_grape_new : ℝ)
variable (num_lemons num_grapes revenue : ℝ)

theorem calculate_original_lemon_price :
  ∀ (L : ℝ),
  -- conditions
  p_lemon_old = L ∧
  p_lemon_new = L + 4 ∧
  p_grape_old = 7 ∧
  p_grape_new = 9 ∧
  num_lemons = 80 ∧
  num_grapes = 140 ∧
  revenue = 2220 ->
  -- proof that the original price is 8
  p_lemon_old = 8 :=
by
  intros L h
  have h1 : p_lemon_new = L + 4 := h.2.1
  have h2 : p_grape_old = 7 := h.2.2.1
  have h3 : p_grape_new = 9 := h.2.2.2.1
  have h4 : num_lemons = 80 := h.2.2.2.2.1
  have h5 : num_grapes = 140 := h.2.2.2.2.2.1
  have h6 : revenue = 2220 := h.2.2.2.2.2.2
  sorry

end NUMINAMATH_GPT_calculate_original_lemon_price_l1235_123593


namespace NUMINAMATH_GPT_max_ab_bc_cd_l1235_123565

theorem max_ab_bc_cd {a b c d : ℝ} (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_sum : a + b + c + d = 200) (h_a : a = 2 * d) : 
  ab + bc + cd ≤ 14166.67 :=
sorry

end NUMINAMATH_GPT_max_ab_bc_cd_l1235_123565


namespace NUMINAMATH_GPT_fifth_term_in_geometric_sequence_l1235_123509

variable (y : ℝ)

def geometric_sequence : ℕ → ℝ
| 0       => 3
| (n + 1) => geometric_sequence n * (3 * y)

theorem fifth_term_in_geometric_sequence (y : ℝ) : 
  geometric_sequence y 4 = 243 * y^4 :=
sorry

end NUMINAMATH_GPT_fifth_term_in_geometric_sequence_l1235_123509


namespace NUMINAMATH_GPT_find_sets_l1235_123556

theorem find_sets (a b c d : ℕ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d)
  (h₅ : (abcd - 1) % ((a-1) * (b-1) * (c-1) * (d-1)) = 0) :
  (a = 3 ∧ b = 5 ∧ c = 17 ∧ d = 255) ∨ (a = 2 ∧ b = 4 ∧ c = 10 ∧ d = 80) :=
by
  sorry

end NUMINAMATH_GPT_find_sets_l1235_123556


namespace NUMINAMATH_GPT_solve_quadratic_completing_square_l1235_123500

theorem solve_quadratic_completing_square (x : ℝ) :
  (2 * x^2 - 4 * x - 1 = 0) ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_completing_square_l1235_123500


namespace NUMINAMATH_GPT_linear_equation_m_not_eq_4_l1235_123514

theorem linear_equation_m_not_eq_4 (m x y : ℝ) :
  (m * x + 3 * y = 4 * x - 1) → m ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_m_not_eq_4_l1235_123514


namespace NUMINAMATH_GPT_carpet_size_l1235_123583

def length := 5
def width := 2
def area := length * width

theorem carpet_size : area = 10 := by
  sorry

end NUMINAMATH_GPT_carpet_size_l1235_123583


namespace NUMINAMATH_GPT_sum_of_D_coordinates_l1235_123524

noncomputable def sum_of_coordinates_of_D (D : ℝ × ℝ) (M C : ℝ × ℝ) : ℝ :=
  D.1 + D.2

theorem sum_of_D_coordinates (D M C : ℝ × ℝ) (H_M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) 
                             (H_M_value : M = (5, 9)) (H_C_value : C = (11, 5)) : 
                             sum_of_coordinates_of_D D M C = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_D_coordinates_l1235_123524


namespace NUMINAMATH_GPT_girls_joined_l1235_123527

theorem girls_joined (initial_girls : ℕ) (boys : ℕ) (girls_more_than_boys : ℕ) (G : ℕ) :
  initial_girls = 632 →
  boys = 410 →
  girls_more_than_boys = 687 →
  initial_girls + G = boys + girls_more_than_boys →
  G = 465 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end NUMINAMATH_GPT_girls_joined_l1235_123527


namespace NUMINAMATH_GPT_x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l1235_123592

theorem x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one {x : ℝ} (h : x + 1 / x = 2) : x^12 = 1 :=
by
  -- The proof will go here, but it is omitted.
  sorry

end NUMINAMATH_GPT_x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l1235_123592


namespace NUMINAMATH_GPT_k_eq_1_l1235_123519

theorem k_eq_1 
  (n m k : ℕ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hk : k > 0) 
  (h : (n - 1) * n * (n + 1) = m^k) : 
  k = 1 := 
sorry

end NUMINAMATH_GPT_k_eq_1_l1235_123519


namespace NUMINAMATH_GPT_least_number_of_roots_l1235_123542

variable (g : ℝ → ℝ) -- Declare the function g with domain ℝ and codomain ℝ

-- Define the conditions as assumptions.
variable (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
variable (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
variable (h3 : g 0 = 0)

-- State the theorem to prove the necessary number of roots.
theorem least_number_of_roots : ∀ a b : ℝ, a ≤ -2000 ∧ b ≥ 2000 → ∃ n ≥ 668, ∃ x : ℝ, g x = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  -- To be filled in with the logic to prove the theorem.
  sorry

end NUMINAMATH_GPT_least_number_of_roots_l1235_123542


namespace NUMINAMATH_GPT_dividend_calculation_l1235_123557

theorem dividend_calculation 
  (D : ℝ) (Q : ℕ) (R : ℕ) 
  (hD : D = 164.98876404494382)
  (hQ : Q = 89)
  (hR : R = 14) :
  ⌈D * Q + R⌉ = 14698 :=
sorry

end NUMINAMATH_GPT_dividend_calculation_l1235_123557


namespace NUMINAMATH_GPT_consecutive_even_number_difference_l1235_123529

theorem consecutive_even_number_difference (x : ℤ) (h : x^2 - (x - 2)^2 = 2012) : x = 504 :=
sorry

end NUMINAMATH_GPT_consecutive_even_number_difference_l1235_123529


namespace NUMINAMATH_GPT_point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l1235_123582

theorem point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb
  (x1 x2 : ℝ) : 
  (x1 * x2 / 4 = -1) ↔ ((x1 / 2) * (x2 / 2) = -1) :=
by sorry

end NUMINAMATH_GPT_point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l1235_123582


namespace NUMINAMATH_GPT_length_of_uncovered_side_l1235_123591

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end NUMINAMATH_GPT_length_of_uncovered_side_l1235_123591


namespace NUMINAMATH_GPT_monotonicity_and_range_of_a_l1235_123586

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem monotonicity_and_range_of_a (a : ℝ) (t : ℝ) (ht : t ≥ 1) :
  (∀ x, x > 0 → f x a ≥ f t a - 3) → a ≤ 2 := 
sorry

end NUMINAMATH_GPT_monotonicity_and_range_of_a_l1235_123586


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1235_123508

theorem hyperbola_eccentricity (a b : ℝ) (h : ∃ P : ℝ × ℝ, ∃ A : ℝ × ℝ, ∃ F : ℝ × ℝ, 
  (∃ c : ℝ, F = (c, 0) ∧ A = (-a, 0) ∧ P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
  (F.fst - P.fst) ^ 2 + P.snd ^ 2 = (F.fst + a) ^ 2 ∧ (F.fst - A.fst) ^ 2 + (F.snd - A.snd) ^ 2 = (F.fst + a) ^ 2 ∧ 
  (P.snd = F.snd) ∧ (abs (F.fst - A.fst) = abs (F.fst - P.fst)))) : 
∃ e : ℝ, e = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1235_123508


namespace NUMINAMATH_GPT_inequality_proof_l1235_123530

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1235_123530


namespace NUMINAMATH_GPT_x_minus_y_options_l1235_123502

theorem x_minus_y_options (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) :
  (x - y ≠ 2013) ∧ (x - y ≠ 2014) ∧ (x - y ≠ 2015) ∧ (x - y ≠ 2016) := 
sorry

end NUMINAMATH_GPT_x_minus_y_options_l1235_123502


namespace NUMINAMATH_GPT_value_of_m_l1235_123523

theorem value_of_m (m : ℝ) (h₁ : m^2 - 9 * m + 19 = 1) (h₂ : 2 * m^2 - 7 * m - 9 ≤ 0) : m = 3 :=
sorry

end NUMINAMATH_GPT_value_of_m_l1235_123523


namespace NUMINAMATH_GPT_second_coloring_book_pictures_l1235_123501

theorem second_coloring_book_pictures (P1 P2 P_colored P_left : ℕ) (h1 : P1 = 23) (h2 : P_colored = 44) (h3 : P_left = 11) (h4 : P1 + P2 = P_colored + P_left) :
  P2 = 32 :=
by
  rw [h1, h2, h3] at h4
  linarith

end NUMINAMATH_GPT_second_coloring_book_pictures_l1235_123501


namespace NUMINAMATH_GPT_evaluate_expression_l1235_123531

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1235_123531


namespace NUMINAMATH_GPT_combined_area_percentage_l1235_123503

theorem combined_area_percentage (D_S : ℝ) (D_R : ℝ) (D_T : ℝ) (A_S A_R A_T : ℝ)
  (h1 : D_R = 0.20 * D_S)
  (h2 : D_T = 0.40 * D_R)
  (h3 : A_R = Real.pi * (D_R / 2) ^ 2)
  (h4 : A_T = Real.pi * (D_T / 2) ^ 2)
  (h5 : A_S = Real.pi * (D_S / 2) ^ 2) :
  ((A_R + A_T) / A_S) * 100 = 4.64 := by
  sorry

end NUMINAMATH_GPT_combined_area_percentage_l1235_123503


namespace NUMINAMATH_GPT_percentage_temporary_workers_l1235_123513

-- Definitions based on the given conditions
def total_workers : ℕ := 100
def percentage_technicians : ℝ := 0.9
def percentage_non_technicians : ℝ := 0.1
def percentage_permanent_technicians : ℝ := 0.9
def percentage_permanent_non_technicians : ℝ := 0.1

-- Statement to prove that the percentage of temporary workers is 18%
theorem percentage_temporary_workers :
  100 * (1 - (percentage_permanent_technicians * percentage_technicians +
              percentage_permanent_non_technicians * percentage_non_technicians)) = 18 :=
by sorry

end NUMINAMATH_GPT_percentage_temporary_workers_l1235_123513


namespace NUMINAMATH_GPT_find_seventh_value_l1235_123563

theorem find_seventh_value (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
  (h₁ : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (h₂ : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (h₃ : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 :=
sorry

end NUMINAMATH_GPT_find_seventh_value_l1235_123563


namespace NUMINAMATH_GPT_max_triangle_area_l1235_123511

theorem max_triangle_area (a b c : ℝ) (h1 : b + c = 8) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) :
  (a - b + c) * (a + b - c) ≤ 64 / 17 :=
by sorry

end NUMINAMATH_GPT_max_triangle_area_l1235_123511


namespace NUMINAMATH_GPT_number_of_blobs_of_glue_is_96_l1235_123568

def pyramid_blobs_of_glue : Nat :=
  let layer1 := 4 * (4 - 1) * 2
  let layer2 := 3 * (3 - 1) * 2
  let layer3 := 2 * (2 - 1) * 2
  let between1_and_2 := 3 * 3 * 4
  let between2_and_3 := 2 * 2 * 4
  let between3_and_4 := 4
  layer1 + layer2 + layer3 + between1_and_2 + between2_and_3 + between3_and_4

theorem number_of_blobs_of_glue_is_96 :
  pyramid_blobs_of_glue = 96 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blobs_of_glue_is_96_l1235_123568


namespace NUMINAMATH_GPT_find_a5_l1235_123559

-- Define the geometric sequence and the given conditions
def geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Define the conditions for our problem
def conditions (a : ℕ → ℝ) :=
  geom_sequence a 2 ∧ (∀ n, 0 < a n) ∧ a 3 * a 11 = 16

-- Our goal is to prove that a_5 = 1
theorem find_a5 (a : ℕ → ℝ) (h : conditions a) : a 5 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a5_l1235_123559


namespace NUMINAMATH_GPT_no_infinite_arithmetic_progression_l1235_123577

open Classical

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem no_infinite_arithmetic_progression
  (f_strict_inc : ∀ x y : R, 0 < x ∧ 0 < y → x < y → f x < f y)
  (f_convex : ∀ x y : R, 0 < x ∧ 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ∀ a : ℕ → R, (∀ n : ℕ, a n = f n) → ¬(∃ d : R, ∀ k : ℕ, a (k + 1) - a k = d) :=
sorry

end NUMINAMATH_GPT_no_infinite_arithmetic_progression_l1235_123577


namespace NUMINAMATH_GPT_trains_clear_each_other_in_12_seconds_l1235_123581

noncomputable def length_train1 : ℕ := 137
noncomputable def length_train2 : ℕ := 163
noncomputable def speed_train1_kmph : ℕ := 42
noncomputable def speed_train2_kmph : ℕ := 48

noncomputable def kmph_to_mps (v : ℕ) : ℚ := v * (5 / 18)
noncomputable def total_distance : ℕ := length_train1 + length_train2
noncomputable def relative_speed_kmph : ℕ := speed_train1_kmph + speed_train2_kmph
noncomputable def relative_speed_mps : ℚ := kmph_to_mps relative_speed_kmph

theorem trains_clear_each_other_in_12_seconds :
  (total_distance : ℚ) / relative_speed_mps = 12 := by
  sorry

end NUMINAMATH_GPT_trains_clear_each_other_in_12_seconds_l1235_123581


namespace NUMINAMATH_GPT_trigonometric_identity_l1235_123534

theorem trigonometric_identity (α β : ℝ) : 
  ((Real.tan α + Real.tan β) / Real.tan (α + β)) 
  + ((Real.tan α - Real.tan β) / Real.tan (α - β)) 
  + 2 * (Real.tan α) ^ 2 
 = 2 / (Real.cos α) ^ 2 :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1235_123534


namespace NUMINAMATH_GPT_beta_max_success_ratio_l1235_123571

-- Define Beta's score conditions
variables (a b c d : ℕ)
def beta_score_conditions :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
  (a * 25 < b * 9) ∧
  (c * 25 < d * 17) ∧
  (b + d = 600)

-- Define Beta's success ratio
def beta_success_ratio :=
  (a + c) / 600

theorem beta_max_success_ratio :
  beta_score_conditions a b c d →
  beta_success_ratio a c ≤ 407 / 600 :=
sorry

end NUMINAMATH_GPT_beta_max_success_ratio_l1235_123571


namespace NUMINAMATH_GPT_vacation_books_pair_count_l1235_123532

/-- 
Given three distinct mystery novels, three distinct fantasy novels, and three distinct biographies,
we want to prove that the number of possible pairs of books of different genres is 27.
-/

theorem vacation_books_pair_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let total_books := mystery_books + fantasy_books + biography_books
  let pairs := (total_books * (total_books - 3)) / 2
  pairs = 27 := 
by
  sorry

end NUMINAMATH_GPT_vacation_books_pair_count_l1235_123532


namespace NUMINAMATH_GPT_simplify_expression_l1235_123551

variables (a b : ℝ)

theorem simplify_expression (h₁ : a = 2) (h₂ : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1235_123551


namespace NUMINAMATH_GPT_eval_expression_l1235_123535

theorem eval_expression : 8 - (6 / (4 - 2)) = 5 := 
sorry

end NUMINAMATH_GPT_eval_expression_l1235_123535


namespace NUMINAMATH_GPT_how_many_fewer_runs_did_E_score_l1235_123574

-- Define the conditions
variables (a b c d e : ℕ)
variable (h1 : 5 * 36 = 180)
variable (h2 : d = e + 5)
variable (h3 : e = 20)
variable (h4 : b = d + e)
variable (h5 : b + c = 107)
variable (h6 : a + b + c + d + e = 180)

-- Specification to be proved
theorem how_many_fewer_runs_did_E_score :
  a - e = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_how_many_fewer_runs_did_E_score_l1235_123574


namespace NUMINAMATH_GPT_largest_of_decimals_l1235_123516

theorem largest_of_decimals :
  let a := 0.993
  let b := 0.9899
  let c := 0.990
  let d := 0.989
  let e := 0.9909
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  sorry

end NUMINAMATH_GPT_largest_of_decimals_l1235_123516


namespace NUMINAMATH_GPT_additional_charge_per_2_5_mile_l1235_123552

theorem additional_charge_per_2_5_mile (x : ℝ) : 
  (∀ (total_charge distance charge_per_segment initial_fee : ℝ),
    total_charge = 5.65 →
    initial_fee = 2.5 →
    distance = 3.6 →
    charge_per_segment = (3.6 / (2/5)) →
    total_charge = initial_fee + charge_per_segment * x → 
    x = 0.35) :=
by
  intros total_charge distance charge_per_segment initial_fee
  intros h_total_charge h_initial_fee h_distance h_charge_per_segment h_eq
  sorry

end NUMINAMATH_GPT_additional_charge_per_2_5_mile_l1235_123552


namespace NUMINAMATH_GPT_problem_statement_l1235_123569

def f (x : Int) : Int :=
  if x > 6 then x^2 - 4
  else if -6 <= x && x <= 6 then 3*x + 2
  else 5

def adjusted_f (x : Int) : Int :=
  let fx := f x
  if x % 3 == 0 then fx + 5 else fx

theorem problem_statement : 
  adjusted_f (-8) + adjusted_f 0 + adjusted_f 9 = 94 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1235_123569


namespace NUMINAMATH_GPT_jellybean_removal_l1235_123520

theorem jellybean_removal 
    (initial_count : ℕ) 
    (first_removal : ℕ) 
    (added_back : ℕ) 
    (final_count : ℕ)
    (initial_count_eq : initial_count = 37)
    (first_removal_eq : first_removal = 15)
    (added_back_eq : added_back = 5)
    (final_count_eq : final_count = 23) :
    (initial_count - first_removal + added_back - final_count) = 4 :=
by 
    sorry

end NUMINAMATH_GPT_jellybean_removal_l1235_123520


namespace NUMINAMATH_GPT_gcd_of_factorials_l1235_123578

-- Define factorials
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define 7!
def seven_factorial : ℕ := factorial 7

-- Define (11! / 4!)
def eleven_div_four_factorial : ℕ := factorial 11 / factorial 4

-- GCD function based on prime factorization (though a direct gcd function also exists, we follow the steps)
def prime_factorization_gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Proof statement
theorem gcd_of_factorials : prime_factorization_gcd seven_factorial eleven_div_four_factorial = 5040 := by
  sorry

end NUMINAMATH_GPT_gcd_of_factorials_l1235_123578


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1235_123543

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
  ∃ d, (∀ n, a (n + 1) - a n = d) ∧ d = 9 / 4 := 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1235_123543


namespace NUMINAMATH_GPT_probability_of_winning_at_least_10_rubles_l1235_123570

-- Definitions based on conditions
def total_tickets : ℕ := 100
def win_20_rubles_tickets : ℕ := 5
def win_15_rubles_tickets : ℕ := 10
def win_10_rubles_tickets : ℕ := 15
def win_2_rubles_tickets : ℕ := 25
def win_nothing_tickets : ℕ := total_tickets - (win_20_rubles_tickets + win_15_rubles_tickets + win_10_rubles_tickets + win_2_rubles_tickets)

-- Probability calculations
def prob_win_20_rubles : ℚ := win_20_rubles_tickets / total_tickets
def prob_win_15_rubles : ℚ := win_15_rubles_tickets / total_tickets
def prob_win_10_rubles : ℚ := win_10_rubles_tickets / total_tickets

-- Prove the probability of winning at least 10 rubles
theorem probability_of_winning_at_least_10_rubles : 
  prob_win_20_rubles + prob_win_15_rubles + prob_win_10_rubles = 0.30 := by
  sorry

end NUMINAMATH_GPT_probability_of_winning_at_least_10_rubles_l1235_123570


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l1235_123558

theorem quadratic_inequality_solution_range (a : ℝ) :
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + 1 / 4 ≤ 0) ↔ 0 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l1235_123558


namespace NUMINAMATH_GPT_man_is_older_by_l1235_123517

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end NUMINAMATH_GPT_man_is_older_by_l1235_123517


namespace NUMINAMATH_GPT_star_polygon_internal_angles_sum_l1235_123536

-- Define the core aspects of the problem using type defintions and axioms.
def n_star_polygon_total_internal_angle_sum (n : ℕ) : ℝ :=
  180 * (n - 4)

theorem star_polygon_internal_angles_sum (n : ℕ) (h : n ≥ 6) :
  n_star_polygon_total_internal_angle_sum n = 180 * (n - 4) :=
by
  -- This step would involve the formal proof using Lean
  sorry

end NUMINAMATH_GPT_star_polygon_internal_angles_sum_l1235_123536


namespace NUMINAMATH_GPT_solve_root_equation_l1235_123585

noncomputable def sqrt4 (x : ℝ) : ℝ := x^(1/4)

theorem solve_root_equation (x : ℝ) :
  sqrt4 (43 - 2 * x) + sqrt4 (39 + 2 * x) = 4 ↔ x = 21 ∨ x = -13.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_root_equation_l1235_123585


namespace NUMINAMATH_GPT_hyperbola_same_foci_l1235_123547

-- Define the conditions for the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := (x^2 / 12) + (y^2 / 4) = 1
def hyperbola (x y m : ℝ) : Prop := (x^2 / m) - y^2 = 1

-- Statement to be proved in Lean 4
theorem hyperbola_same_foci : ∃ m : ℝ, ∀ x y : ℝ, ellipse x y → hyperbola x y m :=
by
  have a_squared := 12
  have b_squared := 4
  have c_squared := a_squared - b_squared
  have c := Real.sqrt c_squared
  have c_value : c = 2 * Real.sqrt 2 := by sorry
  let m := c^2 - 1
  exact ⟨m, by sorry⟩

end NUMINAMATH_GPT_hyperbola_same_foci_l1235_123547


namespace NUMINAMATH_GPT_borrowed_sheets_l1235_123560

theorem borrowed_sheets (sheets borrowed: ℕ) (average_page : ℝ) 
  (total_pages : ℕ := 80) (pages_per_sheet : ℕ := 2) (total_sheets : ℕ := 40) 
  (h1 : borrowed ≤ total_sheets)
  (h2 : sheets = total_sheets - borrowed)
  (h3 : average_page = 26) : borrowed = 17 :=
sorry 

end NUMINAMATH_GPT_borrowed_sheets_l1235_123560


namespace NUMINAMATH_GPT_group_division_l1235_123518

theorem group_division (total_students groups_per_group : ℕ) (h1 : total_students = 30) (h2 : groups_per_group = 5) : 
  (total_students / groups_per_group) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_group_division_l1235_123518


namespace NUMINAMATH_GPT_crayons_given_correct_l1235_123546

def crayons_lost : ℕ := 161
def additional_crayons : ℕ := 410
def crayons_given (lost : ℕ) (additional : ℕ) : ℕ := lost + additional

theorem crayons_given_correct : crayons_given crayons_lost additional_crayons = 571 :=
by
  sorry

end NUMINAMATH_GPT_crayons_given_correct_l1235_123546


namespace NUMINAMATH_GPT_find_abc_l1235_123576

theorem find_abc (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a, b, c) = (3, 5, 15) ∨ (a, b, c) = (2, 4, 8) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l1235_123576


namespace NUMINAMATH_GPT_sandy_books_from_first_shop_l1235_123567

theorem sandy_books_from_first_shop 
  (cost_first_shop : ℕ)
  (books_second_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (total_cost : ℕ)
  (total_books : ℕ)
  (num_books_first_shop : ℕ) :
  cost_first_shop = 1480 →
  books_second_shop = 55 →
  cost_second_shop = 920 →
  average_price = 20 →
  total_cost = cost_first_shop + cost_second_shop →
  total_books = total_cost / average_price →
  num_books_first_shop + books_second_shop = total_books →
  num_books_first_shop = 65 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_sandy_books_from_first_shop_l1235_123567


namespace NUMINAMATH_GPT_geometric_sum_S6_l1235_123561

open Real

-- Define a geometric sequence
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a * n else a * (1 - q ^ n) / (1 - q)

-- Given conditions
variables (a q : ℝ) (n : ℕ)
variable (S3 : ℝ)
variable (q : ℝ) (h_q : q = 2)
variable (h_S3 : S3 = 7)

theorem geometric_sum_S6 :
  sum_geometric a 2 6 = 63 :=
  by
    sorry

end NUMINAMATH_GPT_geometric_sum_S6_l1235_123561


namespace NUMINAMATH_GPT_arman_age_in_years_l1235_123573

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end NUMINAMATH_GPT_arman_age_in_years_l1235_123573


namespace NUMINAMATH_GPT_jar_size_is_half_gallon_l1235_123506

theorem jar_size_is_half_gallon : 
  ∃ (x : ℝ), (48 = 3 * 16) ∧ (16 + 16 * x + 16 * 0.25 = 28) ∧ x = 0.5 :=
by
  -- Implementation goes here
  sorry

end NUMINAMATH_GPT_jar_size_is_half_gallon_l1235_123506


namespace NUMINAMATH_GPT_sum_remainder_l1235_123594

theorem sum_remainder (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 13) 
  (h3 : c % 15 = 9) :
  (a + b + c) % 15 = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_remainder_l1235_123594


namespace NUMINAMATH_GPT_jim_needs_more_miles_l1235_123504

-- Define the conditions
def totalMiles : ℕ := 1200
def drivenMiles : ℕ := 923

-- Define the question and the correct answer
def remainingMiles : ℕ := totalMiles - drivenMiles

-- The theorem statement
theorem jim_needs_more_miles : remainingMiles = 277 :=
by
  -- This will contain the proof which is to be done later
  sorry

end NUMINAMATH_GPT_jim_needs_more_miles_l1235_123504


namespace NUMINAMATH_GPT_find_number_l1235_123512

theorem find_number (x : ℝ) (h : 0.45 * x = 162) : x = 360 :=
sorry

end NUMINAMATH_GPT_find_number_l1235_123512


namespace NUMINAMATH_GPT_cos_17pi_over_4_eq_sqrt2_over_2_l1235_123553

theorem cos_17pi_over_4_eq_sqrt2_over_2 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_17pi_over_4_eq_sqrt2_over_2_l1235_123553
