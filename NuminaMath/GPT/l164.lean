import Mathlib

namespace exists_geometric_arithmetic_progressions_l164_164636

theorem exists_geometric_arithmetic_progressions (n : ℕ) (hn : n > 3) :
  ∃ (x y : ℕ → ℕ),
  (∀ m < n, x (m + 1) = (1 + ε)^m ∧ y (m + 1) = (1 + (m + 1) * ε - δ)) ∧
  ∀ m < n, x m < y m ∧ y m < x (m + 1) :=
by
  sorry

end exists_geometric_arithmetic_progressions_l164_164636


namespace cost_of_apples_l164_164376

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164376


namespace binomial_16_12_l164_164426

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l164_164426


namespace sin_cos_identity_l164_164759

theorem sin_cos_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = 2 / 3) : Real.sin θ - Real.cos θ = (Real.sqrt 14) / 3 ∨ Real.sin θ - Real.cos θ = -(Real.sqrt 14) / 3 :=
by
  sorry

end sin_cos_identity_l164_164759


namespace determine_A_l164_164663

open Real

theorem determine_A (A B C : ℝ)
  (h_decomposition : ∀ x, x ≠ 4 ∧ x ≠ -2 -> (x + 2) / (x^3 - 9 * x^2 + 14 * x + 24) = A / (x - 4) + B / (x - 3) + C / (x + 2)^2)
  (h_factorization : ∀ x, (x^3 - 9 * x^2 + 14 * x + 24) = (x - 4) * (x - 3) * (x + 2)^2) :
  A = 1 / 6 := 
sorry

end determine_A_l164_164663


namespace probability_at_least_5_heads_l164_164710

def fair_coin_probability_at_least_5_heads : ℚ :=
  (Nat.choose 7 5 + Nat.choose 7 6 + Nat.choose 7 7) / 2^7

theorem probability_at_least_5_heads :
  fair_coin_probability_at_least_5_heads = 29 / 128 := 
  by
    sorry

end probability_at_least_5_heads_l164_164710


namespace dad_steps_90_l164_164112

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l164_164112


namespace set_aside_bars_each_day_l164_164756

-- Definitions for the conditions
def total_bars : Int := 20
def bars_traded : Int := 3
def bars_per_sister : Int := 5
def number_of_sisters : Int := 2
def days_in_week : Int := 7

-- Our goal is to prove that Greg set aside 1 bar per day
theorem set_aside_bars_each_day
  (h1 : 20 - 3 = 17)
  (h2 : 5 * 2 = 10)
  (h3 : 17 - 10 = 7)
  (h4 : 7 / 7 = 1) :
  (total_bars - bars_traded - (bars_per_sister * number_of_sisters)) / days_in_week = 1 := by
  sorry

end set_aside_bars_each_day_l164_164756


namespace activity_popularity_order_l164_164284

-- Definitions for the fractions representing activity popularity
def dodgeball_popularity : Rat := 9 / 24
def magic_show_popularity : Rat := 4 / 12
def singing_contest_popularity : Rat := 1 / 3

-- Theorem stating the order of activities based on popularity
theorem activity_popularity_order :
  dodgeball_popularity > magic_show_popularity ∧ magic_show_popularity = singing_contest_popularity :=
by 
  sorry

end activity_popularity_order_l164_164284


namespace solve_quadratic_eq1_solve_quadratic_eq2_l164_164030

theorem solve_quadratic_eq1 (x : ℝ) :
  x^2 - 4 * x + 3 = 0 ↔ (x = 3 ∨ x = 1) :=
sorry

theorem solve_quadratic_eq2 (x : ℝ) :
  x^2 - x - 3 = 0 ↔ (x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) :=
sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l164_164030


namespace train_length_l164_164556

theorem train_length (speed_first_train speed_second_train : ℝ) (length_second_train : ℝ) (cross_time : ℝ) (L1 : ℝ) : 
  speed_first_train = 100 ∧ 
  speed_second_train = 60 ∧ 
  length_second_train = 300 ∧ 
  cross_time = 18 → 
  L1 = 420 :=
by
  sorry

end train_length_l164_164556


namespace determine_a_l164_164904

theorem determine_a 
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x - 2| < 3 ↔ - 5 / 3 < x ∧ x < 1 / 3) : 
  a = -3 := by 
  sorry

end determine_a_l164_164904


namespace Kerry_age_l164_164012

theorem Kerry_age :
  (let cost_per_box := 2.5 in
   let total_cost := 5 in
   let number_of_boxes := total_cost / cost_per_box in
   let candles_per_box := 12 in
   let total_candles := number_of_boxes * candles_per_box in
   let number_of_cakes := 3 in
   let kerry_age := total_candles / number_of_cakes in
   kerry_age = 8) :=
by
  sorry

end Kerry_age_l164_164012


namespace no_base_6_digit_divisible_by_7_l164_164300

theorem no_base_6_digit_divisible_by_7 :
  ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (652 + 42 * d)) :=
by
  intros d hd
  sorry

end no_base_6_digit_divisible_by_7_l164_164300


namespace unitD_questionnaires_l164_164953

theorem unitD_questionnaires :
  ∀ (numA numB numC numD total_drawn : ℕ),
  (2 * numB = numA + numC) →  -- arithmetic sequence condition for B
  (2 * numC = numB + numD) →  -- arithmetic sequence condition for C
  (numA + numB + numC + numD = 1000) →  -- total number condition
  (total_drawn = 150) →  -- total drawn condition
  (numB = 30) →  -- unit B condition
  (total_drawn = (30 - d) + 30 + (30 + d) + (30 + 2 * d)) →
  (d = 15) →
  30 + 2 * d = 60 :=
by
  sorry

end unitD_questionnaires_l164_164953


namespace determine_h_l164_164292

theorem determine_h (x : ℝ) : 
  ∃ h : ℝ → ℝ, (4*x^4 + 11*x^3 + h x = 10*x^3 - x^2 + 4*x - 7) ↔ (h x = -4*x^4 - x^3 - x^2 + 4*x - 7) :=
by
  sorry

end determine_h_l164_164292


namespace wishing_well_probability_l164_164558

theorem wishing_well_probability :
  ∃ m n : ℕ, Nat.Coprime m n ∧ 
    (∀ y ∈ Finset.range 11, 
      ∃ a b ∈ Finset.range 10, 11 * (a - y) = a * (b - y)) ∧ 
    m + n = 111 :=
by
  sorry

end wishing_well_probability_l164_164558


namespace gas_pressure_inversely_proportional_l164_164285

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

end gas_pressure_inversely_proportional_l164_164285


namespace lcm_gcd_48_180_l164_164160

theorem lcm_gcd_48_180 :
  Nat.lcm 48 180 = 720 ∧ Nat.gcd 48 180 = 12 :=
by
  sorry

end lcm_gcd_48_180_l164_164160


namespace fifty_percent_of_number_l164_164763

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  0.6 * x = 42

-- Define the statement we need to prove
theorem fifty_percent_of_number (x : ℝ) (h : given_condition x) : 0.5 * x = 35 := by
  sorry

end fifty_percent_of_number_l164_164763


namespace magical_stack_card_count_l164_164039

theorem magical_stack_card_count :
  ∃ n, n = 157 + 78 ∧ 2 * n = 470 :=
by
  let n := 235
  use n
  have h1: n = 157 + 78 := by sorry
  have h2: 2 * n = 470 := by sorry
  exact ⟨h1, h2⟩

end magical_stack_card_count_l164_164039


namespace k_value_for_inequality_l164_164599

theorem k_value_for_inequality :
    (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d)) ∧
    (∀ k : ℝ, (∀ a b c d : ℝ, a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 → a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) → k = 3/4) :=
sorry

end k_value_for_inequality_l164_164599


namespace average_speed_with_stoppages_l164_164154

theorem average_speed_with_stoppages
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (moving_time_per_hour : ℝ)
  (total_distance_moved : ℝ)
  (total_time_with_stoppages : ℝ) :
  avg_speed_without_stoppages = 60 → 
  stoppage_time_per_hour = 45 / 60 →
  moving_time_per_hour = 15 / 60 →
  total_distance_moved = avg_speed_without_stoppages * moving_time_per_hour →
  total_time_with_stoppages = 1 →
  (total_distance_moved / total_time_with_stoppages) = 15 :=
by
  intros
  sorry

end average_speed_with_stoppages_l164_164154


namespace integer_parts_are_divisible_by_17_l164_164940

-- Define that a is the greatest positive root of the given polynomial
def is_greatest_positive_root (a : ℝ) : Prop :=
  (∀ x : ℝ, x^3 - 3 * x^2 + 1 = 0 → x ≤ a) ∧ a > 0 ∧ (a^3 - 3 * a^2 + 1 = 0)

-- Define the main theorem to prove
theorem integer_parts_are_divisible_by_17 (a : ℝ)
  (h_root : is_greatest_positive_root a) :
  (⌊a ^ 1788⌋ % 17 = 0) ∧ (⌊a ^ 1988⌋ % 17 = 0) := 
sorry

end integer_parts_are_divisible_by_17_l164_164940


namespace area_of_triangle_ABC_l164_164794

-- Axiom statements representing the conditions
axiom medians_perpendicular (A B C D E G : Type) : Prop
axiom median_ad_length (A D : Type) : Prop
axiom median_be_length (B E : Type) : Prop

-- Main theorem statement
theorem area_of_triangle_ABC
  (A B C D E G : Type)
  (h1 : medians_perpendicular A B C D E G)
  (h2 : median_ad_length A D) -- AD = 18
  (h3 : median_be_length B E) -- BE = 24
  : ∃ (area : ℝ), area = 576 :=
sorry

end area_of_triangle_ABC_l164_164794


namespace problem_equiv_l164_164720

theorem problem_equiv :
  ((2001 * 2021 + 100) * (1991 * 2031 + 400)) / (2011^4) = 1 :=
by
  sorry

end problem_equiv_l164_164720


namespace unique_position_all_sequences_one_l164_164534

-- Define the main theorem
theorem unique_position_all_sequences_one (n : ℕ) (sequences : Fin (2^(n-1)) → Fin n → Bool) :
  (∀ a b c : Fin (2^(n-1)), ∃ p : Fin n, sequences a p = true ∧ sequences b p = true ∧ sequences c p = true) →
  ∃! p : Fin n, ∀ i : Fin (2^(n-1)), sequences i p = true :=
by
  sorry

end unique_position_all_sequences_one_l164_164534


namespace grant_total_earnings_l164_164911

theorem grant_total_earnings:
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30
  let glove_discount_rate := 0.20
  let cleats_pair_count := 2
  let cleats_price_per_pair := 10
  let glove_discount := baseball_glove * glove_discount_rate
  let glove_selling_price := baseball_glove - glove_discount
  let cleats_total := cleats_pair_count * cleats_price_per_pair
  let total_earnings := baseball_cards + baseball_bat + glove_selling_price + cleats_total
  in total_earnings = 79 :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end grant_total_earnings_l164_164911


namespace one_of_them_is_mistaken_l164_164676

theorem one_of_them_is_mistaken
  (k n x y : ℕ) 
  (hYakov: k * (x + 5) = n * y)
  (hYuri: k * x = n * (y - 3)) :
  False :=
by
  sorry

end one_of_them_is_mistaken_l164_164676


namespace quadratic_residue_l164_164491

theorem quadratic_residue (a : ℤ) (p : ℕ) (hp : p > 2) (ha_nonzero : a ≠ 0) :
  (∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ 1 [ZMOD p]) ∧
  (¬ ∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ -1 [ZMOD p]) :=
sorry

end quadratic_residue_l164_164491


namespace number_of_m_l164_164188

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end number_of_m_l164_164188


namespace find_A2_A7_l164_164788

theorem find_A2_A7 (A : ℕ → ℝ) (hA1A11 : A 11 - A 1 = 56)
  (hAiAi2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → A (i+2) - A i ≤ 12)
  (hAjAj3 : ∀ j, 1 ≤ j ∧ j ≤ 8 → A (j+3) - A j ≥ 17) : 
  A 7 - A 2 = 29 :=
by
  sorry

end find_A2_A7_l164_164788


namespace systematic_sampling_l164_164378

theorem systematic_sampling (N : ℕ) (k : ℕ) (interval : ℕ) (seq : List ℕ) : 
  N = 70 → k = 7 → interval = 10 → 
  seq = [3, 13, 23, 33, 43, 53, 63] := 
by 
  intros hN hk hInt;
  sorry

end systematic_sampling_l164_164378


namespace intersection_is_2_l164_164610

-- Define the sets A and B
def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { -1, 0, 2 }

-- Define the complement of A
def A_complement : Set ℝ := { x | x ≥ 1 }

-- Define the intersection of the complement of A and B
def intersection : Set ℝ := A_complement ∩ B

-- Prove that the intersection is {2}
theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l164_164610


namespace find_number_l164_164077

theorem find_number (x : ℕ) (h : x + 5 * 8 = 340) : x = 300 :=
sorry

end find_number_l164_164077


namespace find_sticker_price_l164_164182

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l164_164182


namespace price_of_child_ticket_l164_164090

theorem price_of_child_ticket (C : ℝ) 
  (adult_ticket_price : ℝ := 8) 
  (total_tickets_sold : ℕ := 34) 
  (adult_tickets_sold : ℕ := 12) 
  (total_revenue : ℝ := 236) 
  (h1 : 12 * adult_ticket_price + (34 - 12) * C = total_revenue) :
  C = 6.36 :=
by
  sorry

end price_of_child_ticket_l164_164090


namespace max_value_expression_l164_164320

theorem max_value_expression (θ : ℝ) : 
  2 ≤ 5 + 3 * Real.sin θ ∧ 5 + 3 * Real.sin θ ≤ 8 → 
  (∃ θ, (14 / (5 + 3 * Real.sin θ)) = 7) := 
sorry

end max_value_expression_l164_164320


namespace quadratic_function_properties_l164_164363

/-- The graph of the quadratic function y = x^2 - 4x - 1 opens upwards,
    and the vertex is at (2, -5). -/
theorem quadratic_function_properties :
  ∀ (x : ℝ), let y := x^2 - 4*x - 1 in
  (∃ (h k : ℝ), y = (x - h)^2 + k ∧ h = 2 ∧ k = -5) ∧ (∃ a : ℝ, a > 0 ∧ (y = a * (x - 2)^2 - 5)) :=
by
  sorry

end quadratic_function_properties_l164_164363


namespace tailor_charges_30_per_hour_l164_164935

noncomputable def tailor_hourly_rate (shirts pants : ℕ) (shirt_hours pant_hours total_cost : ℝ) :=
  total_cost / (shirts * shirt_hours + pants * pant_hours)

theorem tailor_charges_30_per_hour :
  tailor_hourly_rate 10 12 1.5 3 1530 = 30 := by
  sorry

end tailor_charges_30_per_hour_l164_164935


namespace simplify_expression_l164_164295

variable (a b : ℝ)

theorem simplify_expression :
  (a^3 - b^3) / (a * b) - (ab - b^2) / (ab - a^3) = (a^2 + ab + b^2) / b :=
by
  sorry

end simplify_expression_l164_164295


namespace Grant_made_total_l164_164913

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l164_164913


namespace diameter_of_tripled_volume_sphere_l164_164360

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l164_164360


namespace sin_60_proof_l164_164578

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l164_164578


namespace find_other_root_l164_164801

theorem find_other_root (z : ℂ) (z_squared : z^2 = -91 + 104 * I) (root1 : z = 7 + 10 * I) : z = -7 - 10 * I :=
by
  sorry

end find_other_root_l164_164801


namespace price_of_each_toy_l164_164951

variables (T : ℝ)

-- Given conditions
def total_cost (T : ℝ) : ℝ := 3 * T + 2 * 5 + 5 * 6

theorem price_of_each_toy :
  total_cost T = 70 → T = 10 :=
sorry

end price_of_each_toy_l164_164951


namespace solve_equation_l164_164592

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) (h : a^2 = b * (b + 7)) : 
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by sorry

end solve_equation_l164_164592


namespace first_discount_is_10_l164_164242

def list_price : ℝ := 70
def final_price : ℝ := 59.85
def second_discount : ℝ := 0.05

theorem first_discount_is_10 :
  ∃ (x : ℝ), list_price * (1 - x/100) * (1 - second_discount) = final_price ∧ x = 10 :=
by
  sorry

end first_discount_is_10_l164_164242


namespace roots_numerically_equal_but_opposite_signs_l164_164758

noncomputable def value_of_m (a b c : ℝ) : ℝ := (a - b) / (a + b)

theorem roots_numerically_equal_but_opposite_signs
  (a b c m : ℝ)
  (h : ∀ x : ℝ, (a ≠ 0 ∧ a + b ≠ 0) ∧ (x^2 - b*x = (ax - c) * (m - 1) / (m + 1))) 
  (root_condition : ∃ x₁ x₂ : ℝ, x₁ = -x₂ ∧ x₁ * x₂ != 0) :
  m = value_of_m a b c :=
by
  sorry

end roots_numerically_equal_but_opposite_signs_l164_164758


namespace pencil_length_l164_164402

theorem pencil_length
  (R P L : ℕ)
  (h1 : P = R + 3)
  (h2 : P = L - 2)
  (h3 : R + P + L = 29) :
  L = 12 :=
by
  sorry

end pencil_length_l164_164402


namespace number_of_distinct_integers_from_special_fractions_sums_l164_164722

def is_special (a b : ℕ) : Prop := a + b = 15

def special_fractions : List ℚ :=
  (List.range 14).map (λ k => (k+1 : ℚ) / (15 - (k+1)))

def valid_sums (f g : ℚ) : Proposition :=
  (f + g).denom = 1

theorem number_of_distinct_integers_from_special_fractions_sums :
  (special_fractions.product special_fractions).filter (λ p => valid_sums p.1 p.2) .map (λ p => (p.1 + p.2).nat).erase_dup.length = 9 :=
sorry

end number_of_distinct_integers_from_special_fractions_sums_l164_164722


namespace translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l164_164848

def f_translation : ℝ → ℝ :=
  fun x => (x - 1)^2 - 2

def f_quad (a x : ℝ) : ℝ :=
  x^2 - 2*a*x - 1

theorem translated_quadratic :
  ∀ x, f_translation x = (x - 1)^2 - 2 :=
by
  intro x
  simp [f_translation]

theorem range_of_translated_quadratic :
  ∀ x, 0 ≤ x ∧ x ≤ 4 → -2 ≤ f_translation x ∧ f_translation x ≤ 7 :=
by
  sorry

theorem min_value_on_interval :
  ∀ a, 
    (a ≤ 0 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a x ≥ -1)) ∧
    (0 < a ∧ a < 2 → f_quad a a = -a^2 - 1) ∧
    (a ≥ 2 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a 2 = -4*a + 3)) :=
by
  sorry

end translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l164_164848


namespace minimum_possible_n_l164_164274

theorem minimum_possible_n (n p : ℕ) (h1: p > 0) (h2: 15 * n - 45 = 105) : n = 10 :=
sorry

end minimum_possible_n_l164_164274


namespace jesters_on_stilts_count_l164_164406

theorem jesters_on_stilts_count :
  ∃ j e : ℕ, 3 * j + 4 * e = 50 ∧ j + e = 18 ∧ j = 22 :=
by 
  sorry

end jesters_on_stilts_count_l164_164406


namespace hexagon_bc_de_eq_14_l164_164480

theorem hexagon_bc_de_eq_14
  (α β γ δ ε ζ : ℝ)
  (angle_cond : α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)
  (AB BC CD DE EF FA : ℝ)
  (sum_AB_BC : AB + BC = 11)
  (diff_FA_CD : FA - CD = 3)
  : BC + DE = 14 := sorry

end hexagon_bc_de_eq_14_l164_164480


namespace sum_of_turning_angles_l164_164627

variable (radius distance : ℝ) (C : ℝ)

theorem sum_of_turning_angles (H1 : radius = 10) (H2 : distance = 30000) (H3 : C = 2 * radius * Real.pi) :
  (distance / C) * 2 * Real.pi ≥ 2998 :=
by
  sorry

end sum_of_turning_angles_l164_164627


namespace three_digit_divisible_by_8_l164_164992

theorem three_digit_divisible_by_8 : ∃ n : ℕ, n / 100 = 5 ∧ n % 10 = 3 ∧ n % 8 = 0 :=
by
  use 533
  sorry

end three_digit_divisible_by_8_l164_164992


namespace polynomial_coeff_divisible_by_5_l164_164783

theorem polynomial_coeff_divisible_by_5
  (a b c : ℤ)
  (h : ∀ k : ℤ, (a * k^2 + b * k + c) % 5 = 0) :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 :=
by
  sorry

end polynomial_coeff_divisible_by_5_l164_164783


namespace angle_measure_is_zero_l164_164888

-- Definitions corresponding to conditions
variable (x : ℝ)

def complement (x : ℝ) := 90 - x
def supplement (x : ℝ) := 180 - x

-- Final proof statement
theorem angle_measure_is_zero (h : complement x = (1 / 2) * supplement x) : x = 0 :=
  sorry

end angle_measure_is_zero_l164_164888


namespace solve_for_x_l164_164404

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end solve_for_x_l164_164404


namespace outlet_pipe_emptying_time_l164_164381

noncomputable def fill_rate_pipe1 : ℝ := 1 / 18
noncomputable def fill_rate_pipe2 : ℝ := 1 / 30
noncomputable def empty_rate_outlet_pipe (x : ℝ) : ℝ := 1 / x
noncomputable def combined_rate (x : ℝ) : ℝ := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_outlet_pipe x
noncomputable def total_fill_time : ℝ := 0.06666666666666665

theorem outlet_pipe_emptying_time : ∃ x : ℝ, combined_rate x = 1 / total_fill_time ∧ x = 45 :=
by
  sorry

end outlet_pipe_emptying_time_l164_164381


namespace find_min_value_l164_164900

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (3 / a) - (4 / b) + (5 / c)

theorem find_min_value (a b c : ℝ) (h1 : c > 0) (h2 : 4 * a^2 - 2 * a * b + 4 * b^2 = c) (h3 : ∀ x y : ℝ, |2 * a + b| ≥ |2 * x + y|) :
  minValue a b c = -2 :=
sorry

end find_min_value_l164_164900


namespace fraction_value_l164_164693

theorem fraction_value : (2020 / (20 * 20 : ℝ)) = 5.05 := by
  sorry

end fraction_value_l164_164693


namespace sum_first_11_terms_l164_164917

-- Define the arithmetic sequence and sum formula
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions given
variables (a1 d : ℤ)
axiom condition : (a1 + d) + (a1 + 9 * d) = 4

-- Proof statement
theorem sum_first_11_terms : sum_arithmetic_sequence a1 d 11 = 22 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_first_11_terms_l164_164917


namespace cages_needed_l164_164269

theorem cages_needed (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 13) (h2 : sold_puppies = 7) (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := 
by
  sorry

end cages_needed_l164_164269


namespace find_s_l164_164212

def f (x s : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x + s

theorem find_s (s : ℝ) : f (-1) s = 0 → s = 9 :=
by
  sorry

end find_s_l164_164212


namespace nearest_integer_to_power_l164_164829

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l164_164829


namespace max_n_for_polynomial_l164_164934

theorem max_n_for_polynomial (P : Polynomial ℤ) (hdeg : P.degree = 2022) :
  ∃ n ≤ 2022, ∀ {a : Fin n → ℤ}, 
    (∀ i, P.eval (a i) = i) ↔ n = 2022 :=
by sorry

end max_n_for_polynomial_l164_164934


namespace polar_bear_daily_salmon_consumption_l164_164729

/-- Polar bear's fish consumption conditions and daily salmon amount calculation -/
theorem polar_bear_daily_salmon_consumption (h1: ℝ) (h2: ℝ) : 
  (h1 = 0.2) → (h2 = 0.6) → (h2 - h1 = 0.4) :=
by
  sorry

end polar_bear_daily_salmon_consumption_l164_164729


namespace increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l164_164905

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m * Real.log x

-- Part (1): Prove m >= 2 is the range for which f(x) is increasing
theorem increasing_f_iff_m_ge_two (m : ℝ) : (∀ x > 0, (2 * x - 4 + m / x) ≥ 0) ↔ m ≥ 2 := sorry

-- Part (2): Prove the given inequality for m = 3
theorem inequality_when_m_equals_three (x : ℝ) (h : x > 0) : (1 / 9) * x ^ 3 - (f x 3) > 2 := sorry

end increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l164_164905


namespace minimum_cuts_for_48_rectangles_l164_164950

theorem minimum_cuts_for_48_rectangles : 
  ∃ n : ℕ, n = 6 ∧ (∀ m < 6, 2 ^ m < 48) ∧ 2 ^ n ≥ 48 :=
by
  sorry

end minimum_cuts_for_48_rectangles_l164_164950


namespace infinite_series_value_l164_164417

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l164_164417


namespace weather_conclusion_l164_164871

variables (T C : ℝ) (visitors : ℕ)

def condition1 : Prop :=
  (T ≥ 75.0 ∧ C < 10) → visitors > 100

def condition2 : Prop :=
  visitors ≤ 100

theorem weather_conclusion (h1 : condition1 T C visitors) (h2 : condition2 visitors) : 
  T < 75.0 ∨ C ≥ 10 :=
by 
  sorry

end weather_conclusion_l164_164871


namespace dad_steps_90_l164_164111

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l164_164111


namespace tan_sum_l164_164471

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 96 / 65)
  (h2 : Real.cos x + Real.cos y = 72 / 65) :
  Real.tan x + Real.tan y = 507 / 112 := 
sorry

end tan_sum_l164_164471


namespace students_standing_count_l164_164596

def students_seated : ℕ := 300
def teachers_seated : ℕ := 30
def total_attendees : ℕ := 355

theorem students_standing_count : total_attendees - (students_seated + teachers_seated) = 25 :=
by
  sorry

end students_standing_count_l164_164596


namespace perimeter_of_8_sided_figure_l164_164685

theorem perimeter_of_8_sided_figure (n : ℕ) (len : ℕ) (h1 : n = 8) (h2 : len = 2) :
  n * len = 16 := by
  sorry

end perimeter_of_8_sided_figure_l164_164685


namespace expected_value_of_a_squared_l164_164067

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l164_164067


namespace probability_ephraim_keiko_l164_164938

-- Define the probability that Ephraim gets a certain number of heads tossing two pennies
def prob_heads_ephraim (n : Nat) : ℚ :=
  if n = 2 then 1 / 4
  else if n = 1 then 1 / 2
  else if n = 0 then 1 / 4
  else 0

-- Define the probability that Keiko gets a certain number of heads tossing one penny
def prob_heads_keiko (n : Nat) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 0 then 1 / 2
  else 0

-- Define the probability that Ephraim and Keiko get the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads_ephraim 0 * prob_heads_keiko 0) + (prob_heads_ephraim 1 * prob_heads_keiko 1) + (prob_heads_ephraim 2 * prob_heads_keiko 2)

-- The statement that requires proof
theorem probability_ephraim_keiko : prob_same_heads = 3 / 8 := 
  sorry

end probability_ephraim_keiko_l164_164938


namespace length_AC_l164_164927
open Real

-- Define the conditions and required proof
theorem length_AC (AB DC AD : ℝ) (h1 : AB = 17) (h2 : DC = 25) (h3 : AD = 8) : 
  abs (sqrt ((AD + DC - AD)^2 + (DC - sqrt (AB^2 - AD^2))^2) - 33.6) < 0.1 := 
  by
  -- The proof is omitted for brevity
  sorry

end length_AC_l164_164927


namespace dad_steps_l164_164118

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l164_164118


namespace nancy_deleted_files_correct_l164_164646

-- Variables and conditions
def nancy_original_files : Nat := 43
def files_per_folder : Nat := 6
def number_of_folders : Nat := 2

-- Definition of the number of files that were deleted
def nancy_files_deleted : Nat :=
  nancy_original_files - (files_per_folder * number_of_folders)

-- Theorem to prove
theorem nancy_deleted_files_correct :
  nancy_files_deleted = 31 :=
by
  sorry

end nancy_deleted_files_correct_l164_164646


namespace sum_of_squares_of_roots_l164_164108

theorem sum_of_squares_of_roots :
  (∃ x1 x2 : ℝ, 5 * x1^2 - 3 * x1 - 11 = 0 ∧ 5 * x2^2 - 3 * x2 - 11 = 0 ∧ x1 ≠ x2) →
  (x1 + x2 = 3 / 5 ∧ x1 * x2 = -11 / 5) →
  (x1^2 + x2^2 = 119 / 25) :=
by intro h1 h2; sorry

end sum_of_squares_of_roots_l164_164108


namespace zero_points_product_l164_164907

noncomputable def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a) - (1 / 2) ^ x

theorem zero_points_product (a x1 x2 : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hx1_zero : f a x1 = 0) (hx2_zero : f a x2 = 0) : 0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end zero_points_product_l164_164907


namespace probability_of_infinite_events_l164_164642

open MeasureTheory ProbabilityTheory

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (A : ℕ → Set Ω)

theorem probability_of_infinite_events (h : ProbMeasure.prob_eventually_inf A > 0) :
  ∃ (n_k : ℕ → ℕ), ∀ K : ℕ, ProbMeasure.prob (⋂ i in (finset.range K), A (n_k i)) > 0 :=
sorry

end probability_of_infinite_events_l164_164642


namespace sqrt_36_eq_6_l164_164674

theorem sqrt_36_eq_6 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_36_eq_6_l164_164674


namespace trips_per_student_l164_164206

theorem trips_per_student
  (num_students : ℕ := 5)
  (chairs_per_trip : ℕ := 5)
  (total_chairs : ℕ := 250)
  (T : ℕ) :
  num_students * chairs_per_trip * T = total_chairs → T = 10 :=
by
  intro h
  sorry

end trips_per_student_l164_164206


namespace rachel_found_boxes_l164_164026

theorem rachel_found_boxes (pieces_per_box total_pieces B : ℕ) 
  (h1 : pieces_per_box = 7) 
  (h2 : total_pieces = 49) 
  (h3 : B = total_pieces / pieces_per_box) : B = 7 := 
by 
  sorry

end rachel_found_boxes_l164_164026


namespace find_angle_A_find_value_of_c_l164_164932

variable {a b c A B C : ℝ}

-- Define the specific conditions as Lean 'variables' and 'axioms'
-- Condition: In triangle ABC, the sides opposite to angles A, B and C are a, b, and c respectively.
axiom triangle_ABC_sides : b = 2 * (a * Real.cos B - c)

-- Part (1): Prove the value of angle A
theorem find_angle_A (h : b = 2 * (a * Real.cos B - c)) : A = (2 * Real.pi) / 3 :=
by
  sorry

-- Condition: a * cos C = sqrt 3 and b = 1
axiom cos_C_value : a * Real.cos C = Real.sqrt 3
axiom b_value : b = 1

-- Part (2): Prove the value of c
theorem find_value_of_c (h1 : a * Real.cos C = Real.sqrt 3) (h2 : b = 1) : c = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end find_angle_A_find_value_of_c_l164_164932


namespace even_function_b_eq_zero_l164_164761

theorem even_function_b_eq_zero (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x) = (x^2 - b * x)) → b = 0 :=
by sorry

end even_function_b_eq_zero_l164_164761


namespace instantaneous_velocity_at_t_2_l164_164818

theorem instantaneous_velocity_at_t_2 
  (t : ℝ) (x1 y1 x2 y2: ℝ) : 
  (t = 2) → 
  (x1 = 0) → (y1 = 4) → 
  (x2 = 12) → (y2 = -2) → 
  ((y2 - y1) / (x2 - x1) = -1 / 2) := 
by 
  intros ht hx1 hy1 hx2 hy2
  sorry

end instantaneous_velocity_at_t_2_l164_164818


namespace dad_steps_l164_164125

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l164_164125


namespace selected_six_numbers_have_two_correct_statements_l164_164447

def selection := {n : ℕ // 1 ≤ n ∧ n ≤ 11}

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_multiple (a b : ℕ) : Prop := a ≠ b ∧ (b % a = 0 ∨ a % b = 0)

def is_double_multiple (a b : ℕ) : Prop := a ≠ b ∧ (2 * a = b ∨ 2 * b = a)

theorem selected_six_numbers_have_two_correct_statements (s : Finset selection) (h : s.card = 6) :
  ∃ n1 n2 : selection, is_coprime n1.1 n2.1 ∧ ∃ n1 n2 : selection, is_double_multiple n1.1 n2.1 :=
by
  -- The detailed proof is omitted.
  sorry

end selected_six_numbers_have_two_correct_statements_l164_164447


namespace commission_percentage_l164_164893

def commission_rate (amount: ℕ) : ℚ :=
  if amount <= 500 then
    0.20 * amount
  else
    0.20 * 500 + 0.50 * (amount - 500)

theorem commission_percentage (total_sale : ℕ) (h : total_sale = 800) :
  (commission_rate total_sale) / total_sale * 100 = 31.25 :=
by
  sorry

end commission_percentage_l164_164893


namespace expected_value_a_squared_is_correct_l164_164070

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l164_164070


namespace cathy_total_money_l164_164101

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l164_164101


namespace surface_area_of_box_l164_164494

variable {l w h : ℝ}

def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * h + w * h + l * w)

theorem surface_area_of_box (l w h : ℝ) : surfaceArea l w h = 2 * (l * h + w * h + l * w) :=
by
  sorry

end surface_area_of_box_l164_164494


namespace percentage_increase_sides_l164_164191

theorem percentage_increase_sides (P : ℝ) :
  (1 + P/100) ^ 2 = 1.3225 → P = 15 := 
by
  sorry

end percentage_increase_sides_l164_164191


namespace pipe_B_time_l164_164078

theorem pipe_B_time (C : ℝ) (T : ℝ) 
    (h1 : 2 / 3 * C + C / 3 = C)
    (h2 : C / 36 + C / (3 * T) = C / 14.4) 
    (h3 : T > 0) : 
    T = 8 := 
sorry

end pipe_B_time_l164_164078


namespace remaining_milk_correct_l164_164819

def arranged_milk : ℝ := 21.52
def sold_milk : ℝ := 12.64
def remaining_milk (total : ℝ) (sold : ℝ) : ℝ := total - sold

theorem remaining_milk_correct :
  remaining_milk arranged_milk sold_milk = 8.88 :=
by
  sorry

end remaining_milk_correct_l164_164819


namespace first_number_in_proportion_is_correct_l164_164185

-- Define the proportion condition
def proportion_condition (a x : ℝ) : Prop := a / x = 5 / 11

-- Define the given known value for x
def x_value : ℝ := 1.65

-- Define the correct answer for a
def correct_a : ℝ := 0.75

-- The theorem to prove
theorem first_number_in_proportion_is_correct :
  ∀ a : ℝ, proportion_condition a x_value → a = correct_a := by
  sorry

end first_number_in_proportion_is_correct_l164_164185


namespace problem_statement_l164_164207

-- Define y as the sum of the given terms
def y : ℤ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

-- The theorem to prove that y is a multiple of 8, 16, 32, and 64
theorem problem_statement : 
  (8 ∣ y) ∧ (16 ∣ y) ∧ (32 ∣ y) ∧ (64 ∣ y) :=
by sorry

end problem_statement_l164_164207


namespace expected_value_a_squared_norm_bound_l164_164065

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l164_164065


namespace problem_statement_l164_164656

-- Define S(n) as the given series
def S (n : ℕ) : ℤ := Finset.sum (Finset.range (n + 1)) (λ m, (-1 : ℤ)^m * Nat.choose n m)

-- Statement of the problem in Lean 4
theorem problem_statement : 1990 * (∑ m in Finset.range 1991, (-1 : ℤ)^m / (1990 - m) * Nat.choose 1990 m) + 1 = 0 := by
  sorry

end problem_statement_l164_164656


namespace probability_prime_multiple_of_11_l164_164500

/-- Given 60 cards numbered from 1 to 60, we are to find the probability of selecting a card where 
    the number on the card is both prime and a multiple of 11. -/
theorem probability_prime_multiple_of_11 : 
  let n := 60 in
  let cards := finset.range (n + 1) in
  let prime_multiple_of_11 := {k ∈ cards | nat.prime k ∧ (11 ∣ k)} in
  (prime_multiple_of_11.card : ℚ) / n = 1 / 60 :=
by
  sorry

end probability_prime_multiple_of_11_l164_164500


namespace denomination_is_100_l164_164496

-- Define the initial conditions
def num_bills : ℕ := 8
def total_savings : ℕ := 800

-- Define the denomination of the bills
def denomination_bills (num_bills : ℕ) (total_savings : ℕ) : ℕ := 
  total_savings / num_bills

-- The theorem stating the denomination is $100
theorem denomination_is_100 :
  denomination_bills num_bills total_savings = 100 := by
  sorry

end denomination_is_100_l164_164496


namespace dad_steps_l164_164123

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l164_164123


namespace Jorge_is_24_years_younger_l164_164782

-- Define the conditions
def Jorge_age_2005 := 16
def Simon_age_2010 := 45

-- Prove that Jorge is 24 years younger than Simon
theorem Jorge_is_24_years_younger :
  (Simon_age_2010 - (Jorge_age_2005 + 5) = 24) :=
by
  sorry

end Jorge_is_24_years_younger_l164_164782


namespace find_n_l164_164774

/-- In the expansion of (1 + 3x)^n, where n is a positive integer and n >= 6, 
    if the coefficients of x^5 and x^6 are equal, then n is 7. -/
theorem find_n (n : ℕ) (h₀ : 0 < n) (h₁ : 6 ≤ n)
  (h₂ : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : 
  n = 7 := 
sorry

end find_n_l164_164774


namespace sin_60_proof_l164_164579

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l164_164579


namespace smallest_positive_z_l164_164230

open Real

-- Definitions for the conditions
def sin_zero_condition (x : ℝ) : Prop := sin x = 0
def sin_half_condition (x z : ℝ) : Prop := sin (x + z) = 1 / 2

-- Theorem for the proof objective
theorem smallest_positive_z (x z : ℝ) (hx : sin_zero_condition x) (hz : sin_half_condition x z) : z = π / 6 := 
sorry

end smallest_positive_z_l164_164230


namespace number_of_possible_values_r_l164_164817

noncomputable def is_closest_approx (r : ℝ) : Prop :=
  (r >= 0.2857) ∧ (r < 0.2858)

theorem number_of_possible_values_r : 
  ∃ n : ℕ, (∀ r : ℝ, is_closest_approx r ↔ r = 0.2857 ∨ r = 0.2858 ∨ r = 0.2859) ∧ n = 3 :=
by
  sorry

end number_of_possible_values_r_l164_164817


namespace fraction_meaningful_l164_164256

theorem fraction_meaningful (x : ℝ) : (∃ y, y = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_l164_164256


namespace custom_star_calc_l164_164588

-- defining the custom operation "*"
def custom_star (a b : ℤ) : ℤ :=
  a * b - (b-1) * b

-- providing the theorem statement
theorem custom_star_calc : custom_star 2 (-3) = -18 :=
  sorry

end custom_star_calc_l164_164588


namespace prove_trig_values_l164_164304

/-- Given angles A and B, where both are acute angles,
  and their sine values are known,
  we aim to prove the cosine of (A + B) and the measure
  of angle C in triangle ABC. -/
theorem prove_trig_values (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (sin_A_eq : Real.sin A = (Real.sqrt 5) / 5)
  (sin_B_eq : Real.sin B = (Real.sqrt 10) / 10) :
  Real.cos (A + B) = (Real.sqrt 2) / 2 ∧ (π - (A + B)) = 3 * π / 4 := by
sorry

end prove_trig_values_l164_164304


namespace find_line_equation_l164_164440

theorem find_line_equation : 
  ∃ (m : ℝ), (∀ (x y : ℝ), (2 * x + y - 5 = 0) → (m = -2)) → 
  ∀ (x₀ y₀ : ℝ), (x₀ = -2) ∧ (y₀ = 3) → 
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * x₀ + b * y₀ + c = 0) ∧ (a = 1 ∧ b = -2 ∧ c = 8) := 
by
  sorry

end find_line_equation_l164_164440


namespace Marcus_pretzels_l164_164512

theorem Marcus_pretzels (John_pretzels : ℕ) (Marcus_more_than_John : ℕ) (h1 : John_pretzels = 28) (h2 : Marcus_more_than_John = 12) : Marcus_more_than_John + John_pretzels = 40 :=
by
  sorry

end Marcus_pretzels_l164_164512


namespace cost_of_apples_l164_164371

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164371


namespace billy_points_l164_164566

theorem billy_points (B : ℤ) (h : B - 9 = 2) : B = 11 := 
by 
  sorry

end billy_points_l164_164566


namespace upstream_speed_l164_164860

theorem upstream_speed (Vm Vdownstream Vupstream Vs : ℝ) 
  (h1 : Vm = 50) 
  (h2 : Vdownstream = 55) 
  (h3 : Vdownstream = Vm + Vs) 
  (h4 : Vupstream = Vm - Vs) : 
  Vupstream = 45 :=
by
  sorry

end upstream_speed_l164_164860


namespace number_of_students_in_chemistry_class_l164_164096

variables (students : Finset ℕ) (n : ℕ)
  (x y z cb cp bp c b : ℕ)
  (students_in_total : students.card = 120)
  (chem_bio : cb = 35)
  (bio_phys : bp = 15)
  (chem_phys : cp = 10)
  (total_equation : 120 = x + y + z + cb + bp + cp)
  (chem_equation : c = y + cb + cp)
  (bio_equation : b = x + cb + bp)
  (chem_bio_relation : 4 * b = c)
  (no_all_three_classes : true)

theorem number_of_students_in_chemistry_class : c = 153 :=
  sorry

end number_of_students_in_chemistry_class_l164_164096


namespace arrangement_count_SUCCESS_l164_164877

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l164_164877


namespace problem1_l164_164063

theorem problem1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : a^2 + b^2 = 4 ∧ a * b = 1 := 
by
  sorry

end problem1_l164_164063


namespace milk_in_jugs_l164_164487

theorem milk_in_jugs (x y : ℝ) (h1 : x + y = 70) (h2 : y + 0.125 * x = 0.875 * x) :
  x = 40 ∧ y = 30 := 
sorry

end milk_in_jugs_l164_164487


namespace problem_statement_l164_164955
noncomputable def not_divisible (n : ℕ) : Prop := ∃ k : ℕ, (5^n - 3^n) = (2^n + 65) * k
theorem problem_statement (n : ℕ) (h : 0 < n) : ¬ not_divisible n := sorry

end problem_statement_l164_164955


namespace range_of_a_l164_164328

open Real

theorem range_of_a (k a : ℝ) : 
  (∀ k : ℝ, ∀ x y : ℝ, k * x - y - k + 2 = 0 → x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-7 : ℝ) (-2) ∪ Set.Ioi 1) := 
sorry

end range_of_a_l164_164328


namespace possible_values_l164_164939

noncomputable def matrixN (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem possible_values (x y z : ℂ) (h1 : (matrixN x y z)^3 = 1)
  (h2 : x * y * z = 1) : x^3 + y^3 + z^3 = 4 ∨ x^3 + y^3 + z^3 = -2 :=
  sorry

end possible_values_l164_164939


namespace sum_first_n_terms_arithmetic_sequence_l164_164237

/-- Define the arithmetic sequence with common difference d and a given term a₄. -/
def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

/-- Define the sum of the first n terms of an arithmetic sequence. -/
def sum_of_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * ((2 * a₁ + (n - 1) * d) / 2)

theorem sum_first_n_terms_arithmetic_sequence :
  ∀ n : ℕ, 
  ∀ a₁ : ℤ, 
  (∀ d, d = 2 → (∀ a₁, (a₁ + 3 * d = 8) → sum_of_arithmetic_sequence a₁ d n = (n : ℤ) * ((n : ℤ) + 1))) :=
by
  intros n a₁ d hd h₁
  sorry

end sum_first_n_terms_arithmetic_sequence_l164_164237


namespace expression_evaluation_l164_164383

theorem expression_evaluation :
  (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 :=
by
  sorry

end expression_evaluation_l164_164383


namespace g_one_fourth_l164_164815

noncomputable def g : ℝ → ℝ := sorry

theorem g_one_fourth :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧  -- g(x) is defined for 0 ≤ x ≤ 1
  g 0 = 0 ∧                                    -- g(0) = 0
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧ -- g is non-decreasing
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧ -- symmetric property
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)   -- scaling property
  → g (1/4) = 1/2 :=
sorry

end g_one_fourth_l164_164815


namespace final_price_of_purchases_l164_164434

theorem final_price_of_purchases :
  let electronic_discount := 0.20
  let clothing_discount := 0.15
  let bundle_discount := 10
  let voucher_threshold := 200
  let voucher_value := 20
  let voucher_limit := 2
  let delivery_charge := 15
  let tax_rate := 0.08

  let electronic_original_price := 150
  let clothing_original_price := 80
  let num_clothing := 2

  -- Calculate discounts
  let electronic_discount_amount := electronic_original_price * electronic_discount
  let electronic_discount_price := electronic_original_price - electronic_discount_amount
  let clothing_discount_amount := clothing_original_price * clothing_discount
  let clothing_discount_price := clothing_original_price - clothing_discount_amount

  -- Sum of discounted clothing items
  let total_clothing_discount_price := clothing_discount_price * num_clothing

  -- Calculate bundle discount
  let total_before_bundle_discount := electronic_discount_price + total_clothing_discount_price
  let total_after_bundle_discount := total_before_bundle_discount - bundle_discount

  -- Calculate vouchers
  let num_vouchers := if total_after_bundle_discount >= voucher_threshold * 2 then voucher_limit else 
                      if total_after_bundle_discount >= voucher_threshold then 1 else 0
  let total_voucher_amount := num_vouchers * voucher_value
  let total_after_voucher_discount := total_after_bundle_discount - total_voucher_amount

  -- Add delivery charge
  let total_before_tax := total_after_voucher_discount + delivery_charge

  -- Calculate tax
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount

  final_price = 260.28 :=
by
  -- the actual proof will be included here
  sorry

end final_price_of_purchases_l164_164434


namespace a1_greater_than_floor_2n_over_3_l164_164060

theorem a1_greater_than_floor_2n_over_3
  (n : ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ i j : ℕ, i < j → i ≤ n ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j : ℕ, i ≠ j → i ≤ n ∧ j ≤ n → lcm (a i) (a j) > 2 * n)
  (h_max : ∀ i : ℕ, i ≤ n → a i ≤ 2 * n) :
  a 1 > (2 * n) / 3 :=
by
  sorry

end a1_greater_than_floor_2n_over_3_l164_164060


namespace cathy_total_money_l164_164098

theorem cathy_total_money : 
  let initial := 12 
  let dad_contribution := 25 
  let mom_contribution := 2 * dad_contribution 
  let total_money := initial + dad_contribution + mom_contribution 
  in total_money = 87 :=
by
  let initial := 12
  let dad_contribution := 25
  let mom_contribution := 2 * dad_contribution
  let total_money := initial + dad_contribution + mom_contribution
  show total_money = 87
  sorry

end cathy_total_money_l164_164098


namespace avg_of_first_5_multiples_of_5_l164_164533

theorem avg_of_first_5_multiples_of_5 : (5 + 10 + 15 + 20 + 25) / 5 = 15 := 
by {
  sorry
}

end avg_of_first_5_multiples_of_5_l164_164533


namespace compound_interest_eq_440_l164_164995

-- Define the conditions
variables (P R T SI CI : ℝ)
variables (H_SI : SI = P * R * T / 100)
variables (H_R : R = 20)
variables (H_T : T = 2)
variables (H_given : SI = 400)
variables (H_question : CI = P * (1 + R / 100)^T - P)

-- Define the goal to prove
theorem compound_interest_eq_440 : CI = 440 :=
by
  -- Conditions and the result should be proved here, but we'll use sorry to skip the proof step.
  sorry

end compound_interest_eq_440_l164_164995


namespace change_received_l164_164082

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l164_164082


namespace pen_more_expensive_than_two_notebooks_l164_164864

variable (T R C : ℝ)

-- Conditions
axiom cond1 : T + R + C = 120
axiom cond2 : 5 * T + 2 * R + 3 * C = 350

-- Theorem statement
theorem pen_more_expensive_than_two_notebooks :
  R > 2 * T :=
by
  -- omit the actual proof, but check statement correctness
  sorry

end pen_more_expensive_than_two_notebooks_l164_164864


namespace circle_count_2012_l164_164089

/-
The pattern is defined as follows: 
○●, ○○●, ○○○●, ○○○○●, …
We need to prove that the number of ● in the first 2012 circles is 61.
-/

-- Define the pattern sequence
def circlePattern (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Total number of circles in the first k segments:
def totalCircles (k : ℕ) : ℕ :=
  k * (k + 1) / 2 + k

theorem circle_count_2012 : 
  ∃ (n : ℕ), totalCircles n ≤ 2012 ∧ 2012 < totalCircles (n + 1) ∧ n = 61 :=
by
  sorry

end circle_count_2012_l164_164089


namespace good_numbers_not_good_number_l164_164621

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n),
    ∀ k : Fin n, is_perfect_square (k + 1 + a k + 1)

theorem good_numbers :
  ∀ n : ℕ, 
    n ∈ {13, 15, 17, 19} ↔ is_good_number n :=
begin
  intro n,
  cases n,
  apply Iff.intro,
  { intro h,
    repeat
      { cases h } <|>
      { cases h } <|>
      { cases h with _ h },
    all_goals {exact true_intro} },
  
  { intro h,
    cases h with a ha,
    sorry }
end

theorem not_good_number :
  ¬ is_good_number 11 :=
begin
  intro h,
  cases h with a ha,
  sorry
end

end good_numbers_not_good_number_l164_164621


namespace problem_m_value_l164_164322

noncomputable def find_m (m : ℝ) : Prop :=
  let a : ℝ := real.sqrt (10 - m)
  let b : ℝ := real.sqrt (m - 2)
  (2 * real.sqrt (a^2 - b^2) = 4) ∧ (10 - m > m - 2) ∧ (m - 2 > 0) ∧ (10 - m > 0)

theorem problem_m_value (m : ℝ) : find_m m → m = 4 := by
  sorry

end problem_m_value_l164_164322


namespace cows_in_herd_l164_164543

theorem cows_in_herd (n : ℕ) (h1 : n / 3 + n / 6 + n / 7 < n) (h2 : 15 = n * 5 / 14) : n = 42 :=
sorry

end cows_in_herd_l164_164543


namespace Cathy_total_money_l164_164102

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l164_164102


namespace Jordana_current_age_is_80_l164_164937

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end Jordana_current_age_is_80_l164_164937


namespace binom_16_12_eq_1820_l164_164421

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l164_164421


namespace c_ge_one_l164_164789

theorem c_ge_one (a b : ℕ) (c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (a + 1) / (b + c) = b / a) : c ≥ 1 := 
sorry

end c_ge_one_l164_164789


namespace part1_part2_l164_164340

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem part1 (a x : ℝ) :
  (a < 1 ∧ f x a < 0 ↔ a < x ∧ x < 1) ∧
  (a = 1 ∧ ¬(f x a < 0)) ∧
  (a > 1 ∧ f x a < 0 ↔ 1 < x ∧ x < a) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 < x → f x a ≥ -1) → a ≤ 3 :=
sorry

end part1_part2_l164_164340


namespace odd_square_divisors_l164_164595

theorem odd_square_divisors (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (f g : ℕ), (f > g) ∧ (∀ d, d ∣ (n * n) → d % 4 = 1 ↔ (0 < f)) ∧ (∀ d, d ∣ (n * n) → d % 4 = 3 ↔ (0 < g)) :=
by
  sorry

end odd_square_divisors_l164_164595


namespace sum_of_coordinates_l164_164507

noncomputable def endpoint_x (x : ℤ) := (-3 + x) / 2 = 2
noncomputable def endpoint_y (y : ℤ) := (-15 + y) / 2 = -5

theorem sum_of_coordinates : ∃ x y : ℤ, endpoint_x x ∧ endpoint_y y ∧ x + y = 12 :=
by
  sorry

end sum_of_coordinates_l164_164507


namespace proof_problem_l164_164462

noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 1) + x - 1

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : g x1 = 0)  -- x1 is the root of the equation
  (h2 : 2 * x2 - 1 = 0)  -- x2 is the zero point of f(x) = 2x - 1
  : |x1 - x2| ≤ 1/4 :=
sorry

end proof_problem_l164_164462


namespace M_intersection_N_eq_M_l164_164466

def is_element_of_M (y : ℝ) : Prop := ∃ x : ℝ, y = 2^x
def is_element_of_N (y : ℝ) : Prop := ∃ x : ℝ, y = x^2

theorem M_intersection_N_eq_M : {y | is_element_of_M y} ∩ {y | is_element_of_N y} = {y | is_element_of_M y} :=
by
  sorry

end M_intersection_N_eq_M_l164_164466


namespace disease_cases_1975_l164_164625

theorem disease_cases_1975 (cases_1950 cases_2000 : ℕ) (cases_1950_eq : cases_1950 = 500000)
  (cases_2000_eq : cases_2000 = 1000) (linear_decrease : ∀ t : ℕ, 1950 ≤ t ∧ t ≤ 2000 →
  ∃ k : ℕ, cases_1950 - (k * (t - 1950)) = cases_2000) : 
  ∃ cases_1975 : ℕ, cases_1975 = 250500 := 
by
  -- Setting up known values
  let decrease_duration := 2000 - 1950
  let total_decrease := cases_1950 - cases_2000
  let annual_decrease := total_decrease / decrease_duration
  let years_from_1950_to_1975 := 1975 - 1950
  let decline_by_1975 := annual_decrease * years_from_1950_to_1975
  let cases_1975 := cases_1950 - decline_by_1975
  -- Returning the desired value
  use cases_1975
  sorry

end disease_cases_1975_l164_164625


namespace cost_pants_shirt_l164_164220

variable (P S C : ℝ)

theorem cost_pants_shirt (h1 : P + C = 244) (h2 : C = 5 * S) (h3 : C = 180) : P + S = 100 := by
  sorry

end cost_pants_shirt_l164_164220


namespace smallest_positive_integer_l164_164523

theorem smallest_positive_integer (N : ℕ) :
  (N % 2 = 1) ∧
  (N % 3 = 2) ∧
  (N % 4 = 3) ∧
  (N % 5 = 4) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ↔ 
  N = 2519 := by {
  sorry
}

end smallest_positive_integer_l164_164523


namespace find_angle_ACB_l164_164358

variable (A B C D E F : Type) [HasCos A] [HasCos B] [HasCos C] [HasCos D]

def is_midpoint (p1 p2 : Type) [HasNorm p1] [HasNorm p2] (m : Type) :=
  dist m p1 = dist m p2

variables (AB CD EF : Type) [HasNorm AB] [HasNorm CD] [HasNorm EF]

variable (cos_phi_AB_CD : ℝ)
variable (AB_length : ℝ)
variable (CD_length : ℝ)
variable (EF_length : ℝ)

noncomputable def angle_ACB : ℝ :=
  real.arccos (5 / 8)

theorem find_angle_ACB
  (h1 : cos_phi_AB_CD = (√35)/10)
  (h2 : 2 * √5 = AB_length)
  (h3 : 2 * √7 = CD_length)
  (h4 : EF_length = √13)
  (h5 : is_midpoint A B E)
  (h6 : is_midpoint C D F)
  (h7 : ∀ (u : Type) [HasNorm u], dist E F = EF_length → dist u E = dist u F → dist A B = AB_length → dist C D = CD_length → orthogonal E F u) :
  angle_ACB = real.arccos (5 / 8) := by sorry

end find_angle_ACB_l164_164358


namespace dad_steps_l164_164115

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l164_164115


namespace dot_product_not_sufficient_nor_necessary_for_parallel_l164_164210

open Real

-- Definitions for plane vectors \overrightarrow{a} and \overrightarrow{b}
variables (a b : ℝ × ℝ)

-- Dot product definition for two plane vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Parallelism condition for plane vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k • v2) ∨ v2 = (k • v1)

-- Statement to be proved
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ¬ (∀ a b : ℝ × ℝ, (dot_product a b > 0) ↔ (parallel a b)) :=
sorry

end dot_product_not_sufficient_nor_necessary_for_parallel_l164_164210


namespace arrangement_count_SUCCESS_l164_164875

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l164_164875


namespace expected_value_is_correct_l164_164093

-- Define the monetary outcomes associated with each side
def monetaryOutcome (X : String) : ℚ :=
  if X = "A" then 2 else 
  if X = "B" then -4 else 
  if X = "C" then 6 else 
  0

-- Define the probabilities associated with each side
def probability (X : String) : ℚ :=
  if X = "A" then 1/3 else 
  if X = "B" then 1/2 else 
  if X = "C" then 1/6 else 
  0

-- Compute the expected value
def expectedMonetaryOutcome : ℚ := (probability "A" * monetaryOutcome "A") 
                                + (probability "B" * monetaryOutcome "B") 
                                + (probability "C" * monetaryOutcome "C")

theorem expected_value_is_correct : 
  expectedMonetaryOutcome = -2/3 := by
  sorry

end expected_value_is_correct_l164_164093


namespace parabola_standard_equations_l164_164458

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l164_164458


namespace solve_equation_simplify_expression_l164_164411

-- Problem (1)
theorem solve_equation : ∀ x : ℝ, x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by
  sorry

-- Problem (2)
theorem simplify_expression : ∀ a b : ℝ, a ≠ b → (a ≠ 0 ∧ b ≠ 0) →
  (3 * a ^ 2 - 3 * b ^ 2) / (a ^ 2 * b + a * b ^ 2) /
  (1 - (a ^ 2 + b ^ 2) / (2 * a * b)) = -6 / (a - b) := by
  sorry

end solve_equation_simplify_expression_l164_164411


namespace cupcakes_frosted_in_10_minutes_l164_164567

theorem cupcakes_frosted_in_10_minutes (r1 r2 time : ℝ) (cagney_rate lacey_rate : r1 = 1 / 15 ∧ r2 = 1 / 25)
  (time_in_seconds : time = 600) :
  (1 / ((1 / r1) + (1 / r2)) * time) = 64 := by
  sorry

end cupcakes_frosted_in_10_minutes_l164_164567


namespace equal_shipments_by_truck_l164_164517

theorem equal_shipments_by_truck (T : ℕ) (hT1 : 120 % T = 0) (hT2 : T ≠ 5) : T = 2 :=
by
  sorry

end equal_shipments_by_truck_l164_164517


namespace fraction_filled_l164_164866

-- Definitions for the given conditions
variables (x C : ℝ) (h₁ : 20 * x / 3 = 25 * C / 5) 

-- The goal is to show that x / C = 3 / 4
theorem fraction_filled (h₁ : 20 * x / 3 = 25 * C / 5) : x / C = 3 / 4 :=
by sorry

end fraction_filled_l164_164866


namespace impossible_trailing_zeros_l164_164273

theorem impossible_trailing_zeros (n : ℕ) : ¬ ∃ (k : ℝ), k = 123.75999999999999 ∧ k = ∑ i in (List.range (n + 1)).filter (λ x, 5 ^ x ≤ n), n / 5 ^ i := 
by
  sorry

end impossible_trailing_zeros_l164_164273


namespace seven_thirteenths_of_3940_percent_25000_l164_164964

noncomputable def seven_thirteenths (x : ℝ) : ℝ := (7 / 13) * x

noncomputable def percent (part whole : ℝ) : ℝ := (part / whole) * 100

theorem seven_thirteenths_of_3940_percent_25000 :
  percent (seven_thirteenths 3940) 25000 = 8.484 :=
by
  sorry

end seven_thirteenths_of_3940_percent_25000_l164_164964


namespace units_digit_sum_l164_164691

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum :
  units_digit (24^3 + 17^3) = 7 :=
by
  sorry

end units_digit_sum_l164_164691


namespace inscribed_square_side_length_l164_164276

theorem inscribed_square_side_length (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  ∃ x : ℝ, x = 24 / 7 :=
by
  sorry

end inscribed_square_side_length_l164_164276


namespace min_band_members_exists_l164_164075

theorem min_band_members_exists (n : ℕ) :
  (∃ n, (∃ k : ℕ, n = 9 * k) ∧ (∃ m : ℕ, n = 10 * m) ∧ (∃ p : ℕ, n = 11 * p)) → n = 990 :=
by
  sorry

end min_band_members_exists_l164_164075


namespace trigonometric_inequality_l164_164344

noncomputable def a : Real := (1/2) * Real.cos (8 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * Real.pi / 180)
noncomputable def b : Real := (2 * Real.tan (14 * Real.pi / 180)) / (1 - (Real.tan (14 * Real.pi / 180))^2)
noncomputable def c : Real := Real.sqrt ((1 - Real.cos (48 * Real.pi / 180)) / 2)

theorem trigonometric_inequality :
  a < c ∧ c < b := by
  sorry

end trigonometric_inequality_l164_164344


namespace complex_number_corresponding_to_OB_l164_164483

theorem complex_number_corresponding_to_OB :
  let OA : ℂ := 6 + 5 * Complex.I
  let AB : ℂ := 4 + 5 * Complex.I
  OB = OA + AB -> OB = 10 + 10 * Complex.I := by
  sorry

end complex_number_corresponding_to_OB_l164_164483


namespace committee_probability_l164_164814

variable (B G : ℕ) -- Number of boys and girls
variable (n : ℕ) -- Size of the committee 
variable (N : ℕ) -- Total number of members

theorem committee_probability 
  (h_B : B = 12) 
  (h_G : G = 8) 
  (h_N : N = 20) 
  (h_n : n = 4) : 
  (1 - (Nat.choose 12 4 + Nat.choose 8 4) / Nat.choose 20 4) = 4280 / 4845 := 
by sorry

end committee_probability_l164_164814


namespace odd_function_value_l164_164605

def f (a x : ℝ) : ℝ := -x^3 + (a-2)*x^2 + x

-- Test that f(x) is an odd function:
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value (a : ℝ) (h : is_odd_function (f a)) : f a a = -6 :=
by
  sorry

end odd_function_value_l164_164605


namespace problem_l164_164969

theorem problem
  (a b : ℚ)
  (h1 : 3 * a + 5 * b = 47)
  (h2 : 7 * a + 2 * b = 52)
  : a + b = 35 / 3 :=
sorry

end problem_l164_164969


namespace diamonds_count_l164_164092

-- Definitions based on the conditions given in the problem
def totalGems : Nat := 5155
def rubies : Nat := 5110
def diamonds (total rubies : Nat) : Nat := total - rubies

-- Statement of the proof problem
theorem diamonds_count : diamonds totalGems rubies = 45 := by
  sorry

end diamonds_count_l164_164092


namespace volume_frustum_l164_164278

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

theorem volume_frustum (original_base_edge original_height small_base_edge small_height : ℝ)
  (h_orig : original_base_edge = 10) (h_orig_height : original_height = 10)
  (h_small : small_base_edge = 5) (h_small_height : small_height = 5) :
  volume_pyramid original_base_edge original_height - volume_pyramid small_base_edge small_height
  = 875 / 3 := by
    simp [volume_pyramid, h_orig, h_orig_height, h_small, h_small_height]
    sorry

end volume_frustum_l164_164278


namespace strawberries_left_correct_l164_164982

-- Define the initial and given away amounts in kilograms and grams
def initial_strawberries_kg : Int := 3
def initial_strawberries_g : Int := 300
def given_strawberries_kg : Int := 1
def given_strawberries_g : Int := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : Int) : Int := kg * 1000

-- Calculate the total strawberries initially and given away in grams
def total_initial_strawberries_g : Int :=
  (kg_to_g initial_strawberries_kg) + initial_strawberries_g

def total_given_strawberries_g : Int :=
  (kg_to_g given_strawberries_kg) + given_strawberries_g

-- The amount of strawberries left after giving some away
def strawberries_left : Int :=
  total_initial_strawberries_g - total_given_strawberries_g

-- The statement to prove
theorem strawberries_left_correct :
  strawberries_left = 1400 :=
by
  sorry

end strawberries_left_correct_l164_164982


namespace equation_solutions_exist_l164_164497

theorem equation_solutions_exist (d x y : ℤ) (hx : Odd x) (hy : Odd y)
  (hxy : x^2 - d * y^2 = -4) : ∃ X Y : ℕ, X^2 - d * Y^2 = -1 :=
by
  sorry  -- Proof is omitted as per the instructions

end equation_solutions_exist_l164_164497


namespace geometric_sequence_eighth_term_l164_164823

theorem geometric_sequence_eighth_term (a r : ℝ) (h₀ : a = 27) (h₁ : r = 1/3) :
  a * r^7 = 1/81 :=
by
  rw [h₀, h₁]
  sorry

end geometric_sequence_eighth_term_l164_164823


namespace max_value_expr_l164_164297

theorem max_value_expr (x y : ℝ) : (2 * x + 3 * y + 4) / (Real.sqrt (x^4 + y^2 + 1)) ≤ Real.sqrt 29 := sorry

end max_value_expr_l164_164297


namespace triangle_inequality_right_triangle_l164_164004

theorem triangle_inequality_right_triangle
  (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a + b) / Real.sqrt 2 ≤ c :=
by sorry

end triangle_inequality_right_triangle_l164_164004


namespace age_difference_l164_164330

variable (A B : ℕ)

-- Given conditions
def B_is_95 : Prop := B = 95
def A_after_30_years : Prop := A + 30 = 2 * (B - 30)

-- Theorem to prove
theorem age_difference (h1 : B_is_95 B) (h2 : A_after_30_years A B) : A - B = 5 := 
by
  sorry

end age_difference_l164_164330


namespace range_of_x_l164_164765

theorem range_of_x (x : ℝ) : (x ≠ 2) ↔ ∃ y : ℝ, y = 5 / (x - 2) :=
begin
  sorry
end

end range_of_x_l164_164765


namespace correct_statements_l164_164612

-- Define the conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := 2 < a ∧ a < 3
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := a > 1

-- Prove the statements
theorem correct_statements (a : ℝ) : (p a → q a) ∧ (r a → q a) :=
by {
    sorry
}

end correct_statements_l164_164612


namespace right_triangle_set_l164_164263

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end right_triangle_set_l164_164263


namespace seq_general_formula_l164_164740

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

theorem seq_general_formula {a : ℕ → ℝ} (h1 : a 1 = 1) (h2 : seq a) :
  ∀ n, a n = 1 / 2 ^ (n - 1) :=
by
  sorry

end seq_general_formula_l164_164740


namespace suitable_k_first_third_quadrants_l164_164921

theorem suitable_k_first_third_quadrants (k : ℝ) : 
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
by
  sorry

end suitable_k_first_third_quadrants_l164_164921


namespace students_with_both_pets_l164_164153

theorem students_with_both_pets
  (D C : Finset ℕ)
  (h_union : (D ∪ C).card = 48)
  (h_D : D.card = 30)
  (h_C : C.card = 34) :
  (D ∩ C).card = 16 :=
by sorry

end students_with_both_pets_l164_164153


namespace sin_60_eq_sqrt_three_div_two_l164_164574

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l164_164574


namespace earnings_difference_l164_164150

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end earnings_difference_l164_164150


namespace percent_uni_no_job_choice_l164_164196

variable (P_ND_JC P_JC P_UD P_U_NJC P_NJC : ℝ)
variable (h1 : P_ND_JC = 0.18)
variable (h2 : P_JC = 0.40)
variable (h3 : P_UD = 0.37)

theorem percent_uni_no_job_choice :
  (P_UD - (P_JC - P_ND_JC)) / (1 - P_JC) = 0.25 :=
by
  sorry

end percent_uni_no_job_choice_l164_164196


namespace regular_polygon_property_l164_164498

variables {n : ℕ}
variables {r : ℝ} -- r is the radius of the circumscribed circle
variables {t_2n : ℝ} -- t_2n is the area of the 2n-gon
variables {k_n : ℝ} -- k_n is the perimeter of the n-gon

theorem regular_polygon_property
  (h1 : t_2n = (n * k_n * r) / 2)
  (h2 : k_n = n * a_n) :
  (t_2n / r^2) = (k_n / (2 * r)) :=
by sorry

end regular_polygon_property_l164_164498


namespace max_earnings_mary_l164_164792

def wage_rate : ℝ := 8
def first_hours : ℕ := 20
def max_hours : ℕ := 80
def regular_tip_rate : ℝ := 2
def overtime_rate_increase : ℝ := 1.25
def overtime_tip_rate : ℝ := 3
def overtime_bonus_threshold : ℕ := 5
def overtime_bonus_amount : ℝ := 20

noncomputable def total_earnings (hours : ℕ) : ℝ :=
  let regular_hours := min hours first_hours
  let overtime_hours := if hours > first_hours then hours - first_hours else 0
  let overtime_blocks := overtime_hours / overtime_bonus_threshold
  let regular_earnings := regular_hours * (wage_rate + regular_tip_rate)
  let overtime_earnings := overtime_hours * (wage_rate * overtime_rate_increase + overtime_tip_rate)
  let bonuses := (overtime_blocks) * overtime_bonus_amount
  regular_earnings + overtime_earnings + bonuses

theorem max_earnings_mary : total_earnings max_hours = 1220 := by
  sorry

end max_earnings_mary_l164_164792


namespace range_of_m_l164_164916

theorem range_of_m (m : ℝ) (h : (2 - m) * (|m| - 3) < 0) : (-3 < m ∧ m < 2) ∨ (m > 3) :=
sorry

end range_of_m_l164_164916


namespace value_of_f5_f_neg5_l164_164449

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 - a * x^3 + b * x + 2

-- Given conditions
variable (a b : ℝ)
axiom h1 : f (-5) a b = 3

-- The proposition to prove
theorem value_of_f5_f_neg5 : f 5 a b + f (-5) a b = 4 :=
by
  -- Include the result of the proof
  sorry

end value_of_f5_f_neg5_l164_164449


namespace square_distance_between_intersections_l164_164987

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end square_distance_between_intersections_l164_164987


namespace camille_birds_count_l164_164288

theorem camille_birds_count : 
  let cardinals := 3 in
  let robins := 4 * cardinals in
  let blue_jays := 2 * cardinals in
  let sparrows := 3 * cardinals + 1 in
  cardinals + robins + blue_jays + sparrows = 31 :=
by 
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  show cardinals + robins + blue_jays + sparrows = 31
  calc 
    cardinals + robins + blue_jays + sparrows = 3 + (4 * 3) + (2 * 3) + (3 * 3 + 1) : by rfl
    ... = 3 + 12 + 6 + 10 : by rfl
    ... = 31 : by rfl

end camille_birds_count_l164_164288


namespace angle_A_30_side_b_sqrt2_l164_164479

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the dot product of vectors AB and AC is 2√3 times the area S, 
    then angle A equals 30 degrees --/
theorem angle_A_30 {a b c S : ℝ} (h : (a * b * Real.sqrt 3 * c * Real.sin (π / 6)) = 2 * Real.sqrt 3 * S) : 
  A = π / 6 :=
sorry

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the tangent of angles A, B, C are in the ratio 1:2:3 and c equals 1, 
    then side b equals √2 --/
theorem side_b_sqrt2 {A B C : ℝ} (a b c : ℝ) (h_tan_ratio : Real.tan A / Real.tan B = 1 / 2 ∧ Real.tan B / Real.tan C = 2 / 3)
  (h_c : c = 1) : b = Real.sqrt 2 :=
sorry

end angle_A_30_side_b_sqrt2_l164_164479


namespace find_m_value_l164_164317

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2 * Real.sqrt 2, 2)
noncomputable def b : ℝ × ℝ := (0, 2)
noncomputable def c (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 2)

theorem find_m_value (h : (a.1 + 2 * b.1) * (m) + (a.2 + 2 * b.2) * (Real.sqrt 2) = 0) : m = -3 :=
by
  sorry

end find_m_value_l164_164317


namespace vacation_days_l164_164073

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def towels_per_day_per_person : ℕ := 1
def washer_capacity : ℕ := 14
def num_loads : ℕ := 6

def total_people : ℕ := num_families * people_per_family
def towels_per_day : ℕ := total_people * towels_per_day_per_person
def total_towels : ℕ := num_loads * washer_capacity

def days_at_vacation_rental := total_towels / towels_per_day

theorem vacation_days : days_at_vacation_rental = 7 := by
  sorry

end vacation_days_l164_164073


namespace tangent_line_eq_l164_164159

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

theorem tangent_line_eq {x y : ℝ} (hx : x = 1) (hy : y = 2) (H : circle_eq x y) :
  y = 2 :=
by
  sorry

end tangent_line_eq_l164_164159


namespace number_of_subsets_l164_164064

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end number_of_subsets_l164_164064


namespace find_value_l164_164903

def set_condition (s : Set ℕ) : Prop := s = {0, 1, 2}

def one_relationship_correct (a b c : ℕ) : Prop :=
  (a ≠ 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b = 2 ∧ c = 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0)
  ∨ (a ≠ 2 ∧ b = 0 ∧ c ≠ 0)

theorem find_value (a b c : ℕ) (h1 : set_condition {a, b, c}) (h2 : one_relationship_correct a b c) :
  100 * c + 10 * b + a = 102 :=
sorry

end find_value_l164_164903


namespace polynomial_equality_l164_164184

theorem polynomial_equality :
  (3 * x + 1) ^ 4 = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e →
  a - b + c - d + e = 16 :=
by
  intro h
  sorry

end polynomial_equality_l164_164184


namespace verify_incorrect_option_l164_164339

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end verify_incorrect_option_l164_164339


namespace cats_on_edges_l164_164695

variables {W1 W2 B1 B2 : ℕ}  -- representing positions of cats on a line

def distance_from_white_to_black_sum_1 (a1 a2 : ℕ) : Prop := a1 + a2 = 4
def distance_from_white_to_black_sum_2 (b1 b2 : ℕ) : Prop := b1 + b2 = 8
def distance_from_black_to_white_sum_1 (b1 a1 : ℕ) : Prop := b1 + a1 = 9
def distance_from_black_to_white_sum_2 (b2 a2 : ℕ) : Prop := b2 + a2 = 3

theorem cats_on_edges
  (a1 a2 b1 b2 : ℕ)
  (h1 : distance_from_white_to_black_sum_1 a1 a2)
  (h2 : distance_from_white_to_black_sum_2 b1 b2)
  (h3 : distance_from_black_to_white_sum_1 b1 a1)
  (h4 : distance_from_black_to_white_sum_2 b2 a2) :
  (a1 = 2) ∧ (a2 = 2) ∧ (b1 = 7) ∧ (b2 = 1) ∧ (W1 = min W1 W2) ∧ (B2 = max B1 B2) :=
sorry

end cats_on_edges_l164_164695


namespace prime_m_l164_164591

theorem prime_m (m : ℕ) (hm : m ≥ 2) :
  (∀ n : ℕ, (m / 3 ≤ n) → (n ≤ m / 2) → (n ∣ Nat.choose n (m - 2 * n))) → Nat.Prime m :=
by
  intro h
  sorry

end prime_m_l164_164591


namespace time_morning_is_one_l164_164694

variable (D : ℝ)  -- Define D as the distance between the two points.

def morning_speed := 20 -- Morning speed (km/h)
def afternoon_speed := 10 -- Afternoon speed (km/h)
def time_difference := 1 -- Time difference (hour)

-- Proving that the morning time t_m is equal to 1 hour
theorem time_morning_is_one (t_m t_a : ℝ) 
  (h1 : t_m - t_a = time_difference) 
  (h2 : D = morning_speed * t_m) 
  (h3 : D = afternoon_speed * t_a) : 
  t_m = 1 := 
by
  sorry

end time_morning_is_one_l164_164694


namespace inscribed_circle_radius_l164_164521

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaUsingHeron (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  let K := areaUsingHeron a b c
  K / s

theorem inscribed_circle_radius : inscribedCircleRadius 26 18 20 = Real.sqrt 31 :=
  sorry

end inscribed_circle_radius_l164_164521


namespace square_division_l164_164655

theorem square_division (n : ℕ) (h : n ≥ 6) : ∃ squares : ℕ, squares = n ∧ can_divide_into_squares(squares) := 
sorry

end square_division_l164_164655


namespace dad_steps_are_90_l164_164143

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l164_164143


namespace retail_price_increase_l164_164551

theorem retail_price_increase (R W : ℝ) (h1 : 0.80 * R = 1.44000000000000014 * W)
  : ((R - W) / W) * 100 = 80 :=
by 
  sorry

end retail_price_increase_l164_164551


namespace question_l164_164968

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 7

theorem question (x : ℝ) (n : ℕ) (h1 : 2 < x ∧ x < 3) (h2 : f x = 0) : n = 2 := by
  sorry

end question_l164_164968


namespace smallest_integer_l164_164680

theorem smallest_integer {x y z : ℕ} (h1 : 2*y = x) (h2 : 3*y = z) (h3 : x + y + z = 60) : y = 6 :=
by
  sorry

end smallest_integer_l164_164680


namespace large_beaker_multiple_small_beaker_l164_164554

variables (S L : ℝ) (k : ℝ)

theorem large_beaker_multiple_small_beaker 
  (h1 : Small_beaker = S)
  (h2 : Large_beaker = k * S)
  (h3 : Salt_water_in_small = S/2)
  (h4 : Fresh_water_in_large = (Large_beaker) / 5)
  (h5 : (Salt_water_in_small + Fresh_water_in_large = 0.3 * (Large_beaker))) :
  k = 5 :=
sorry

end large_beaker_multiple_small_beaker_l164_164554


namespace off_the_rack_suit_cost_l164_164007

theorem off_the_rack_suit_cost (x : ℝ)
  (h1 : ∀ y, y = 3 * x + 200)
  (h2 : ∀ y, x + y = 1400) :
  x = 300 :=
by
  sorry

end off_the_rack_suit_cost_l164_164007


namespace ratio_of_democrats_l164_164675

theorem ratio_of_democrats (F M : ℕ) (h1 : F + M = 750) (h2 : (1/2 : ℚ) * F = 125) (h3 : (1/4 : ℚ) * M = 125) :
  (125 + 125 : ℚ) / 750 = 1 / 3 := by
  sorry

end ratio_of_democrats_l164_164675


namespace tens_digit_of_13_pow_2021_l164_164432

theorem tens_digit_of_13_pow_2021 :
  let p := 2021
  let base := 13
  let mod_val := 100
  let digit := (base^p % mod_val) / 10
  digit = 1 := by
  sorry

end tens_digit_of_13_pow_2021_l164_164432


namespace problem_solution_l164_164177

open Set

theorem problem_solution
    (a b : ℝ)
    (ineq : ∀ x : ℝ, 1 < x ∧ x < b → a * x^2 - 3 * x + 2 < 0)
    (f : ℝ → ℝ := λ x => (2 * a + b) * x - 1 / ((a - b) * (x - 1))) :
    a = 1 ∧ b = 2 ∧ (∀ x, 1 < x ∧ x < b → f x ≥ 8 ∧ (f x = 8 ↔ x = 3 / 2)) :=
by
  sorry

end problem_solution_l164_164177


namespace part1_part2_l164_164908

variables (m : ℝ)

def p (m : ℝ) : Prop := 2^m > Real.sqrt 2
def q (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ (x^2 - 2*x + m^2 = 0) ∧ (y^2 - 2*y + m^2 = 0)

theorem part1 :
  (p m ∧ q m) → (1 / 2 < m ∧ m < 1) :=
sorry

theorem part2 :
  ((p m ∨ q m) ∧ ¬ (p m ∧ q m)) → 
  (m ∈ Set.Ioc (-1 : ℝ) (1 / 2) ∪ Set.Ici (1 : ℝ)) :=
sorry

end part1_part2_l164_164908


namespace find_n_l164_164169

-- Definitions based on the given conditions
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- The mathematically equivalent proof problem statement:
theorem find_n (n : ℕ) (p : ℝ) (h1 : binomial_expectation n p = 6) (h2 : binomial_variance n p = 3) : n = 12 :=
sorry

end find_n_l164_164169


namespace find_k_and_general_term_l164_164748

noncomputable def sum_of_first_n_terms (n k : ℝ) : ℝ :=
  -n^2 + (10 + k) * n + (k - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  -2 * n + 12

theorem find_k_and_general_term :
  (∀ n k : ℝ, sum_of_first_n_terms n k = sum_of_first_n_terms n (1 : ℝ)) ∧
  (∀ n : ℕ, ∃ an : ℝ, an = general_term n) :=
by
  sorry

end find_k_and_general_term_l164_164748


namespace binom_coeff_div_prime_l164_164892

open Nat

theorem binom_coeff_div_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p :=
by
  sorry

end binom_coeff_div_prime_l164_164892


namespace final_result_is_four_l164_164865

theorem final_result_is_four (x : ℕ) (h1 : x = 208) (y : ℕ) (h2 : y = x / 2) (z : ℕ) (h3 : z = y - 100) : z = 4 :=
by {
  sorry
}

end final_result_is_four_l164_164865


namespace count_valid_m_l164_164189

theorem count_valid_m : 
    ∃ m : ℤ, ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b) ∧
    (x^2 - m * x + (m + 2006) = 0) ∧
    (5 = {m : ℤ | ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b)}.card) :=
sorry

end count_valid_m_l164_164189


namespace joe_bath_shop_bottles_l164_164781

theorem joe_bath_shop_bottles (b : ℕ) (n : ℕ) (m : ℕ) 
    (h1 : 5 * n = b * m)
    (h2 : 5 * n = 95)
    (h3 : b * m = 95)
    (h4 : b ≠ 1)
    (h5 : b ≠ 95): 
    b = 19 := 
by 
    sorry

end joe_bath_shop_bottles_l164_164781


namespace range_of_x_l164_164766

theorem range_of_x (x : ℝ) : (x ≠ 2) ↔ ∃ y : ℝ, y = 5 / (x - 2) :=
begin
  sorry
end

end range_of_x_l164_164766


namespace find_a_value_l164_164345

noncomputable def prob_sum_equals_one (a : ℝ) : Prop :=
  a * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_a_value (a : ℝ) (h : prob_sum_equals_one a) : a = 16/15 :=
sorry

end find_a_value_l164_164345


namespace perfect_square_trinomial_l164_164611

noncomputable def p (k : ℝ) (x : ℝ) : ℝ :=
  4 * x^2 + 2 * k * x + 9

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, p k x = (2 * x + b)^2) → (k = 6 ∨ k = -6) :=
by 
  intro h
  sorry

end perfect_square_trinomial_l164_164611


namespace minimum_value_of_x_plus_y_l164_164307

theorem minimum_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (x - 1) * (y - 1) = 1) : x + y = 4 :=
sorry

end minimum_value_of_x_plus_y_l164_164307


namespace mork_tax_rate_l164_164795

theorem mork_tax_rate (M R : ℝ) (h1 : 0.15 = 0.15) (h2 : 4 * M = Mindy_income) (h3 : (R / 100 * M + 0.15 * 4 * M) = 0.21 * 5 * M):
  R = 45 :=
sorry

end mork_tax_rate_l164_164795


namespace smallest_number_of_students_l164_164870

theorem smallest_number_of_students (n9 n7 n8 : ℕ) (h7 : 9 * n7 = 7 * n9) (h8 : 5 * n8 = 9 * n9) :
  n9 + n7 + n8 = 134 :=
by
  -- Skipping proof with sorry
  sorry

end smallest_number_of_students_l164_164870


namespace min_value_objective_l164_164790

variable (x y : ℝ)

def constraints : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

def objective (x y : ℝ) : ℝ := y - 2 * x

theorem min_value_objective :
  constraints x y → ∃ x y, objective x y = -7 :=
by
  sorry

end min_value_objective_l164_164790


namespace cat_finishes_food_on_next_monday_l164_164227

noncomputable def cat_food_consumption_per_day : ℚ := (1 / 4) + (1 / 6)

theorem cat_finishes_food_on_next_monday :
  ∃ n : ℕ, n = 8 ∧ (n * cat_food_consumption_per_day > 8) := sorry

end cat_finishes_food_on_next_monday_l164_164227


namespace intersection_with_y_axis_l164_164559

-- Define the original linear function
def original_function (x : ℝ) : ℝ := -2 * x + 3

-- Define the function after moving it up by 2 units
def moved_up_function (x : ℝ) : ℝ := original_function x + 2

-- State the theorem to prove the intersection with the y-axis
theorem intersection_with_y_axis : moved_up_function 0 = 5 :=
by
  sorry

end intersection_with_y_axis_l164_164559


namespace desiree_age_l164_164151

theorem desiree_age (D C G Gr : ℕ) 
  (h1 : D = 2 * C)
  (h2 : D + 30 = (2 * (C + 30)) / 3 + 14)
  (h3 : G = D + C)
  (h4 : G + 20 = 3 * (D - C))
  (h5 : Gr = (D + 10) * (C + 10) / 2) : 
  D = 6 := 
sorry

end desiree_age_l164_164151


namespace g_g_g_g_3_eq_101_l164_164291

def g (m : ℕ) : ℕ :=
  if m < 5 then m^2 + 1 else 2 * m + 3

theorem g_g_g_g_3_eq_101 : g (g (g (g 3))) = 101 :=
  by {
    -- the proof goes here
    sorry
  }

end g_g_g_g_3_eq_101_l164_164291


namespace orthodiagonal_quadrilateral_l164_164336

-- Define the quadrilateral sides and their relationships
variables (AB BC CD DA : ℝ)
variables (h1 : AB = 20) (h2 : BC = 70) (h3 : CD = 90)
theorem orthodiagonal_quadrilateral : AB^2 + CD^2 = BC^2 + DA^2 → DA = 60 :=
by
  sorry

end orthodiagonal_quadrilateral_l164_164336


namespace increment_in_radius_l164_164834

theorem increment_in_radius (C1 C2 : ℝ) (hC1 : C1 = 50) (hC2 : C2 = 60) : 
  ((C2 / (2 * Real.pi)) - (C1 / (2 * Real.pi)) = (5 / Real.pi)) :=
by
  sorry

end increment_in_radius_l164_164834


namespace product_of_base8_digits_of_8654_l164_164686

theorem product_of_base8_digits_of_8654 : 
  let base10 := 8654
  let base8_rep := [2, 0, 7, 1, 6] -- Representing 8654(10) to 20716(8)
  (base8_rep.prod = 0) :=
  sorry

end product_of_base8_digits_of_8654_l164_164686


namespace find_number_x_l164_164272

theorem find_number_x (x : ℝ) (h : 2500 - x / 20.04 = 2450) : x = 1002 :=
by
  -- Proof can be written here, but skipped by using sorry
  sorry

end find_number_x_l164_164272


namespace sequence_general_term_and_sum_sum_tn_bound_l164_164750

theorem sequence_general_term_and_sum (c : ℝ) (h₁ : c = 1) 
  (f : ℕ → ℝ) (hf : ∀ x, f x = (1 / 3) ^ x) :
  (∀ n, a_n = -2 / 3 ^ n) ∧ (∀ n, b_n = 2 * n - 1) :=
by {
  sorry
}

theorem sum_tn_bound (h₂ : ∀ n > 0, T_n = (1 / 2) * (1 - 1 / (2 * n + 1))) :
  ∃ n, T_n > 1005 / 2014 ∧ n = 252 :=
by {
  sorry
}

end sequence_general_term_and_sum_sum_tn_bound_l164_164750


namespace longer_side_length_l164_164540

-- Define the relevant entities: radius, area of the circle, and rectangle conditions.
noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := Real.pi * radius^2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm.
theorem longer_side_length : ∃ (l : ℝ), (area_rectangle = l * shorter_side) → (l = 9 * Real.pi) :=
by
  sorry

end longer_side_length_l164_164540


namespace part_one_part_two_l164_164270

def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Question (1)
theorem part_one (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
sorry

-- Question (2)
theorem part_two (a : ℝ) (h : a ≥ 1) : ∀ (y : ℝ), (∃ x : ℝ, f x a = y) ↔ (∃ b : ℝ, y = b + 2 ∧ b ≥ a) := 
sorry

end part_one_part_two_l164_164270


namespace speed_with_current_l164_164861

-- Define the constants
def speed_of_current : ℝ := 2.5
def speed_against_current : ℝ := 20

-- Define the man's speed in still water
axiom speed_in_still_water : ℝ
axiom speed_against_current_eq : speed_in_still_water - speed_of_current = speed_against_current

-- The statement we need to prove
theorem speed_with_current : speed_in_still_water + speed_of_current = 25 := sorry

end speed_with_current_l164_164861


namespace river_and_building_geometry_l164_164034

open Real

theorem river_and_building_geometry (x y : ℝ) :
  (tan 60 * x = y) ∧ (tan 30 * (x + 30) = y) → x = 15 ∧ y = 15 * sqrt 3 :=
by
  sorry

end river_and_building_geometry_l164_164034


namespace one_positive_zero_l164_164906

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - 1

theorem one_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ f x a = 0 :=
sorry

end one_positive_zero_l164_164906


namespace xy_difference_l164_164918

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by {
    sorry
}

end xy_difference_l164_164918


namespace cars_per_day_l164_164696

noncomputable def paul_rate : ℝ := 2
noncomputable def jack_rate : ℝ := 3
noncomputable def paul_jack_rate : ℝ := paul_rate + jack_rate
noncomputable def hours_per_day : ℝ := 8
noncomputable def total_cars : ℝ := paul_jack_rate * hours_per_day

theorem cars_per_day : total_cars = 40 := by
  sorry

end cars_per_day_l164_164696


namespace least_multiple_36_sum_digits_l164_164259

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_multiple_36_sum_digits :
  ∃ n : ℕ, n = 36 ∧ (36 ∣ n) ∧ (9 ∣ digit_sum n) ∧ (∀ m : ℕ, (36 ∣ m) ∧ (9 ∣ digit_sum m) → 36 ≤ m) :=
by sorry

end least_multiple_36_sum_digits_l164_164259


namespace alpha_beta_sum_equal_two_l164_164314

theorem alpha_beta_sum_equal_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0) 
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := 
sorry

end alpha_beta_sum_equal_two_l164_164314


namespace simplify_expression_l164_164807

theorem simplify_expression (x y : ℝ) :  3 * x + 5 * x + 7 * x + 2 * y = 15 * x + 2 * y := 
by 
  sorry

end simplify_expression_l164_164807


namespace scooter_gain_percent_l164_164659

def initial_cost : ℝ := 900
def first_repair_cost : ℝ := 150
def second_repair_cost : ℝ := 75
def third_repair_cost : ℝ := 225
def selling_price : ℝ := 1800

theorem scooter_gain_percent :
  let total_cost := initial_cost + first_repair_cost + second_repair_cost + third_repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 33.33 :=
by
  sorry

end scooter_gain_percent_l164_164659


namespace joseph_total_distance_l164_164203

-- Distance Joseph runs on Monday
def d1 : ℕ := 900

-- Increment each day
def increment : ℕ := 200

-- Adjust distance calculation
def d2 := d1 + increment
def d3 := d2 + increment

-- Total distance calculation
def total_distance := d1 + d2 + d3

-- Prove that the total distance is 3300 meters
theorem joseph_total_distance : total_distance = 3300 :=
by sorry

end joseph_total_distance_l164_164203


namespace math_problem_l164_164472

variables {R : Type*} [Ring R] (x y z : R)

theorem math_problem (h : x * y + y * z + z * x = 0) : 
  3 * x * y * z + x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0 :=
by 
  sorry

end math_problem_l164_164472


namespace jean_spots_l164_164779

theorem jean_spots (total_spots upper_torso_spots back_hindspots sides_spots : ℕ)
  (h1 : upper_torso_spots = 30)
  (h2 : total_spots = 2 * upper_torso_spots)
  (h3 : back_hindspots = total_spots / 3)
  (h4 : sides_spots = total_spots - upper_torso_spots - back_hindspots) :
  sides_spots = 10 :=
by
  sorry

end jean_spots_l164_164779


namespace dad_steps_l164_164145

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l164_164145


namespace exists_n0_find_N_l164_164753

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Definition of the sequence {a_n}
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = f (a n)

-- Problem (1): Existence of n0
theorem exists_n0 (a : ℕ → ℝ) (h_seq : seq a) (h_a1 : a 1 = 3) : 
  ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n :=
  sorry

-- Problem (2): Smallest N
theorem find_N (a : ℕ → ℝ) (h_seq : seq a) (m : ℕ) (h_m : m > 1) 
  (h_a1 : 1 + 1 / (m : ℝ) < a 1 ∧ a 1 < m / (m - 1)) : 
  ∃ N : ℕ, ∀ n ≥ N, 0 < a n ∧ a n < 1 :=
  sorry

end exists_n0_find_N_l164_164753


namespace min_elements_of_B_l164_164755

def A (k : ℝ) : Set ℝ :=
if k < 0 then {x | (k / 4 + 9 / (4 * k) + 3) < x ∧ x < 11 / 2}
else if k = 0 then {x | x < 11 / 2}
else if 0 < k ∧ k < 1 ∨ k > 9 then {x | x < 11 / 2 ∨ x > k / 4 + 9 / (4 * k) + 3}
else if 1 ≤ k ∧ k ≤ 9 then {x | x < k / 4 + 9 / (4 * k) + 3 ∨ x > 11 / 2}
else ∅

def B (k : ℝ) : Set ℤ := {x : ℤ | ↑x ∈ A k}

theorem min_elements_of_B (k : ℝ) (hk : k < 0) : 
  B k = {2, 3, 4, 5} :=
sorry

end min_elements_of_B_l164_164755


namespace linear_function_no_first_quadrant_l164_164505

theorem linear_function_no_first_quadrant : 
  ¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = -3 * x - 2 := by
  sorry

end linear_function_no_first_quadrant_l164_164505


namespace problem1_arithmetic_sequence_problem2_geometric_sequence_l164_164847

-- Problem (1)
variable (S : Nat → Int)
variable (a : Nat → Int)

axiom S10_eq_50 : S 10 = 50
axiom S20_eq_300 : S 20 = 300
axiom S_def : (∀ n : Nat, n > 0 → S n = n * a 1 + (n * (n-1) / 2) * (a 2 - a 1))

theorem problem1_arithmetic_sequence (n : Nat) : a n = 2 * n - 6 := sorry

-- Problem (2)
variable (a : Nat → Int)

axiom S3_eq_a2_plus_10a1 : S 3 = a 2 + 10 * a 1
axiom a5_eq_81 : a 5 = 81
axiom positive_terms : ∀ n, a n > 0

theorem problem2_geometric_sequence (n : Nat) : S n = (3 ^ n - 1) / 2 := sorry

end problem1_arithmetic_sequence_problem2_geometric_sequence_l164_164847


namespace dad_steps_90_l164_164114

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l164_164114


namespace problem1_problem2_l164_164571

theorem problem1 : -7 + 13 - 6 + 20 = 20 := 
by
  sorry

theorem problem2 : -2^3 + (2 - 3) - 2 * (-1)^2023 = -7 := 
by
  sorry

end problem1_problem2_l164_164571


namespace average_calls_per_day_l164_164009

theorem average_calls_per_day :
  let calls := [35, 46, 27, 61, 31] in
  (calls.sum / (calls.length : ℝ)) = 40 :=
by
  sorry

end average_calls_per_day_l164_164009


namespace dad_steps_l164_164139

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l164_164139


namespace jennifer_sister_age_l164_164936

-- Define the conditions
def in_ten_years_jennifer_age (current_age_j : ℕ) : ℕ := current_age_j + 10
def in_ten_years_jordana_age (current_age_j current_age_jo : ℕ) : ℕ := current_age_jo + 10
def jennifer_will_be_30 := ∀ (current_age_j : ℕ), in_ten_years_jennifer_age current_age_j = 30
def jordana_will_be_three_times_jennifer := ∀ (current_age_jo current_age_j : ℕ), 
  in_ten_years_jordana_age current_age_j current_age_jo = 3 * in_ten_years_jennifer_age current_age_j

-- Prove that Jordana is currently 80 years old given the conditions
theorem jennifer_sister_age (current_age_jo current_age_j : ℕ) 
  (H1 : jennifer_will_be_30 current_age_j) 
  (H2 : jordana_will_be_three_times_jennifer current_age_jo current_age_j) : 
  current_age_jo = 80 :=
by
  sorry

end jennifer_sister_age_l164_164936


namespace find_W_l164_164552

noncomputable def volume_of_space (r_sphere r_cylinder h_cylinder : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h_cylinder
  let V_cone := (1 / 3) * Real.pi * r_cylinder^2 * h_cylinder
  V_sphere - V_cylinder - V_cone

theorem find_W : volume_of_space 6 4 10 = (224 / 3) * Real.pi := by
  sorry

end find_W_l164_164552


namespace perpendicular_vectors_x_eq_5_l164_164218

def vector_a (x : ℝ) : ℝ × ℝ := (2, x + 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_eq_5 (x : ℝ)
  (h : dot_product (vector_a x) (vector_b x) = 0) :
  x = 5 :=
sorry

end perpendicular_vectors_x_eq_5_l164_164218


namespace solve_system_eqns_l164_164032

theorem solve_system_eqns (x y z a : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2)
  (h3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := 
by
  sorry

end solve_system_eqns_l164_164032


namespace range_of_m_l164_164504

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h_even : ∀ x, f x = f (-x)) 
 (h_decreasing : ∀ {x y}, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x)
 (h_condition : ∀ x, 1 ≤ x → x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (Real.log x + 3 - 2 * m * x)) :
  m ∈ Set.Icc (1 / (2 * Real.exp 1)) ((Real.log 3 + 6) / 6) :=
sorry

end range_of_m_l164_164504


namespace problem_part1_problem_part2_problem_part3_l164_164179

open Set

noncomputable def U := ℝ
noncomputable def A := { x : ℝ | x < -4 ∨ x > 1 }
noncomputable def B := { x : ℝ | -3 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem_part1 :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 3 } := by sorry

theorem problem_part2 :
  compl A ∪ compl B = { x : ℝ | x ≤ 1 ∨ x > 3 } := by sorry

theorem problem_part3 (k : ℝ) :
  { x : ℝ | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1 } ⊆ A → k > 1 := by sorry

end problem_part1_problem_part2_problem_part3_l164_164179


namespace exists_k_l164_164337

-- Define P as a non-constant homogeneous polynomial with real coefficients
def homogeneous_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ (a b : ℝ), P (a * a) (b * b) = (a * a) ^ n * (b * b) ^ n

-- Define the main problem
theorem exists_k (P : ℝ → ℝ → ℝ) (hP : ∃ n : ℕ, homogeneous_polynomial n P)
  (h : ∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) :
  ∃ k : ℕ, ∀ x y : ℝ, P x y = (x^2 + y^2) ^ k :=
sorry

end exists_k_l164_164337


namespace tens_digit_3_pow_2016_eq_2_l164_164690

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem tens_digit_3_pow_2016_eq_2 : tens_digit (3 ^ 2016) = 2 := by
  sorry

end tens_digit_3_pow_2016_eq_2_l164_164690


namespace third_quadrant_angles_l164_164509

theorem third_quadrant_angles :
  {α : ℝ | ∃ k : ℤ, π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π} =
  {α | π < α ∧ α < 3 * π / 2} :=
sorry

end third_quadrant_angles_l164_164509


namespace cottage_cheese_quantity_l164_164896

theorem cottage_cheese_quantity (x : ℝ) 
    (milk_fat : ℝ := 0.05) 
    (curd_fat : ℝ := 0.155) 
    (whey_fat : ℝ := 0.005) 
    (milk_mass : ℝ := 1) 
    (h : (curd_fat * x + whey_fat * (milk_mass - x)) = milk_fat * milk_mass) : 
    x = 0.3 :=
    sorry

end cottage_cheese_quantity_l164_164896


namespace sin_60_eq_sqrt3_div_2_l164_164576

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l164_164576


namespace nearest_int_to_expr_l164_164827

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l164_164827


namespace trains_pass_each_other_l164_164989

noncomputable def time_to_pass (speed1 speed2 distance : ℕ) : ℚ :=
  (distance : ℚ) / ((speed1 + speed2) : ℚ) * 60

theorem trains_pass_each_other :
  time_to_pass 60 80 100 = 42.86 := sorry

end trains_pass_each_other_l164_164989


namespace pair_a_n_uniq_l164_164293

theorem pair_a_n_uniq (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_eq : 3^n = a^2 - 16) : a = 5 ∧ n = 2 := 
by 
  sorry

end pair_a_n_uniq_l164_164293


namespace pond_field_area_ratio_l164_164816

theorem pond_field_area_ratio
  (l : ℝ) (w : ℝ) (A_field : ℝ) (A_pond : ℝ)
  (h1 : l = 2 * w)
  (h2 : l = 16)
  (h3 : A_field = l * w)
  (h4 : A_pond = 8 * 8) :
  A_pond / A_field = 1 / 2 :=
by
  sorry

end pond_field_area_ratio_l164_164816


namespace range_of_f_l164_164898

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → f x ∈ Set.Icc (-5) 3) ∧
  (∀ y : ℝ, y ∈ Set.Icc (-5) 3 → ∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) :=
by
  sorry

end range_of_f_l164_164898


namespace sin_60_eq_sqrt3_div_2_l164_164581

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l164_164581


namespace cesaro_sum_51_term_sequence_l164_164891

noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let T := List.scanl (· + ·) 0 B
  T.drop 1 |>.sum / B.length

theorem cesaro_sum_51_term_sequence (B : List ℝ) (h_length : B.length = 49)
  (h_cesaro_sum_49 : cesaro_sum B = 500) :
  cesaro_sum (B ++ [0, 0]) = 1441.18 :=
by
  sorry

end cesaro_sum_51_term_sequence_l164_164891


namespace minimum_score_4th_quarter_l164_164257

theorem minimum_score_4th_quarter (q1 q2 q3 : ℕ) (q4 : ℕ) :
  q1 = 85 → q2 = 80 → q3 = 90 →
  (q1 + q2 + q3 + q4) / 4 ≥ 85 →
  q4 ≥ 85 :=
by intros hq1 hq2 hq3 h_avg
   sorry

end minimum_score_4th_quarter_l164_164257


namespace intersection_point_l164_164652

theorem intersection_point (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
  ∃ x y, (y = a * x^2 + b * x + c) ∧ (y = a * x^2 - b * x + c + d) ∧ x = d / (2 * b) ∧ y = a * (d / (2 * b))^2 + (d / 2) + c :=
by
  sorry

end intersection_point_l164_164652


namespace distance_sum_l164_164167

-- Definitions and conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  ( (Real.sqrt 2 / 2) * t, 3 + (Real.sqrt 2 / 2) * t )

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def curve_C (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 2 * Real.sin θ

def cartesian_curve_C (x y : ℝ) : Prop :=
  x^2 = 2 * y

def cartesian_line (x y : ℝ) : Prop := 
  x - y + 3 = 0

def point_P : ℝ × ℝ := (0, 3)

-- Main theorem to prove
theorem distance_sum (t1 t2 : ℝ) (h_t : t1^2 - 2 * Real.sqrt 2 * t1 - 12 = 0 ∧ t2^2 - 2 * Real.sqrt 2 * t2 - 12 = 0) :
  abs (t1 - t2) = 2 * Real.sqrt 14 :=
by 
  sorry

end distance_sum_l164_164167


namespace C_finishes_job_in_days_l164_164838

theorem C_finishes_job_in_days :
  ∀ (A B C : ℚ),
    (A + B = 1 / 15) →
    (A + B + C = 1 / 3) →
    1 / C = 3.75 :=
by
  intros A B C hab habc
  sorry

end C_finishes_job_in_days_l164_164838


namespace T_n_correct_l164_164773

def a_n (n : ℕ) : ℤ := 2 * n - 5

def b_n (n : ℕ) : ℤ := 2^n

def C_n (n : ℕ) : ℤ := |a_n n| * b_n n

def T_n : ℕ → ℤ
| 1     => 6
| 2     => 10
| n     => if n >= 3 then 34 + (2 * n - 7) * 2^(n + 1) else 0  -- safeguard for invalid n

theorem T_n_correct (n : ℕ) (hyp : n ≥ 1) : 
  T_n n = 
  if n = 1 then 6 
  else if n = 2 then 10 
  else if n ≥ 3 then 34 + (2 * n - 7) * 2^(n + 1) 
  else 0 := 
by 
sorry

end T_n_correct_l164_164773


namespace correct_answer_l164_164681

def total_contestants : Nat := 56
def selected_contestants : Nat := 14

theorem correct_answer :
  (total_contestants = 56) →
  (selected_contestants = 14) →
  (selected_contestants = 14) :=
by
  intro h_total h_selected
  exact h_selected

end correct_answer_l164_164681


namespace exists_constant_a_l164_164746

theorem exists_constant_a (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : (m : ℝ) / n < Real.sqrt 7) :
  ∃ (a : ℝ), a > 1 ∧ (7 - (m^2 : ℝ) / (n^2 : ℝ) ≥ a / (n^2 : ℝ)) ∧ a = 3 :=
by
  sorry

end exists_constant_a_l164_164746


namespace smallest_four_digit_palindrome_divisible_by_6_l164_164689

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

theorem smallest_four_digit_palindrome_divisible_by_6 : 
  ∃ (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ is_divisible_by_6 n ∧ 
  ∀ (m : ℕ), is_four_digit m ∧ is_palindrome m ∧ is_divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_palindrome_divisible_by_6_l164_164689


namespace sue_charge_per_dog_l164_164104

def amount_saved_christian : ℝ := 5
def amount_saved_sue : ℝ := 7
def charge_per_yard : ℝ := 5
def yards_mowed_christian : ℝ := 4
def total_cost_perfume : ℝ := 50
def additional_amount_needed : ℝ := 6
def dogs_walked_sue : ℝ := 6

theorem sue_charge_per_dog :
  (amount_saved_christian + (charge_per_yard * yards_mowed_christian) + amount_saved_sue + (dogs_walked_sue * x) + additional_amount_needed = total_cost_perfume) → x = 2 :=
by
  sorry

end sue_charge_per_dog_l164_164104


namespace inequality_x2_8_over_xy_y2_l164_164351

open Real

theorem inequality_x2_8_over_xy_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^2 + 8 / (x * y) + y^2 ≥ 8 := 
sorry

end inequality_x2_8_over_xy_y2_l164_164351


namespace crayon_count_l164_164408

theorem crayon_count (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) : initial_crayons - eaten_crayons = 10 := 
by 
  sorry

end crayon_count_l164_164408


namespace compute_b1c1_b2c2_b3c3_l164_164211

theorem compute_b1c1_b2c2_b3c3 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -1 :=
by
  sorry

end compute_b1c1_b2c2_b3c3_l164_164211


namespace total_amount_is_correct_l164_164708

-- Given conditions
def original_price : ℝ := 200
def discount_rate: ℝ := 0.25
def coupon_value: ℝ := 10
def tax_rate: ℝ := 0.05

-- Define the price calculations
def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
def price_after_coupon (p : ℝ) (c : ℝ) : ℝ := p - c
def final_price_with_tax (p : ℝ) (t : ℝ) : ℝ := p * (1 + t)

-- Goal: Prove the final amount the customer pays
theorem total_amount_is_correct : final_price_with_tax (price_after_coupon (discounted_price original_price discount_rate) coupon_value) tax_rate = 147 := by
  sorry

end total_amount_is_correct_l164_164708


namespace sum_of_digits_of_B_is_7_l164_164639

theorem sum_of_digits_of_B_is_7 : 
  let A := 16 ^ 16
  let sum_digits (n : ℕ) : ℕ := n.digits 10 |>.sum
  let S := sum_digits
  let B := S (S A)
  sum_digits B = 7 :=
sorry

end sum_of_digits_of_B_is_7_l164_164639


namespace black_cards_remaining_proof_l164_164851

def initial_black_cards := 26
def black_cards_taken_out := 4
def black_cards_remaining := initial_black_cards - black_cards_taken_out

theorem black_cards_remaining_proof : black_cards_remaining = 22 := 
by sorry

end black_cards_remaining_proof_l164_164851


namespace part1_answer1_part1_answer2_part2_answer1_part2_answer2_l164_164338

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem part1_answer1 : A ∩ C = {3, 4, 5, 6, 7} :=
by
  sorry

theorem part1_answer2 : A \ B = {5, 6, 7, 8, 9, 10} :=
by
  sorry

theorem part2_answer1 : A \ (B ∪ C) = {8, 9, 10} :=
by 
  sorry

theorem part2_answer2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :=
by 
  sorry

end part1_answer1_part1_answer2_part2_answer1_part2_answer2_l164_164338


namespace students_selected_l164_164250

-- Define the number of boys and girls
def boys : ℕ := 13
def girls : ℕ := 10

-- Define the combination function as it is useful for calculations
def combination (n k : ℕ) := nat.choose n k

-- Define the condition for the number of ways to select 1 girl and 2 boys
def ways_to_select : ℕ := 780

-- Define the correct answer that needs to be proven
def selected_students : ℕ := 1 + 2

-- Theorem statement
theorem students_selected (h : combination girls 1 * combination boys 2 = ways_to_select) : selected_students = 3 :=
by sorry

end students_selected_l164_164250


namespace nearest_integer_to_power_l164_164830

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l164_164830


namespace dad_steps_are_90_l164_164140

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l164_164140


namespace find_number_ge_40_l164_164852

theorem find_number_ge_40 (x : ℝ) : 0.90 * x > 0.80 * 30 + 12 → x > 40 :=
by sorry

end find_number_ge_40_l164_164852


namespace line_intersects_xaxis_at_l164_164719

theorem line_intersects_xaxis_at (x y : ℝ) 
  (h : 4 * y - 5 * x = 15) 
  (hy : y = 0) : (x, y) = (-3, 0) :=
by
  sorry

end line_intersects_xaxis_at_l164_164719


namespace dad_steps_l164_164136

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l164_164136


namespace toll_for_18_wheel_truck_l164_164980

-- Definitions based on conditions
def num_axles (total_wheels : ℕ) (wheels_front_axle : ℕ) (wheels_per_other_axle : ℕ) : ℕ :=
  1 + (total_wheels - wheels_front_axle) / wheels_per_other_axle

def toll (x : ℕ) : ℝ :=
  0.50 + 0.50 * (x - 2)

-- The problem statement to prove
theorem toll_for_18_wheel_truck : toll (num_axles 18 2 4) = 2.00 := by
  sorry

end toll_for_18_wheel_truck_l164_164980


namespace fabian_cards_l164_164439

theorem fabian_cards : ∃ (g y b r : ℕ),
  (g > 0 ∧ g < 10) ∧ (y > 0 ∧ y < 10) ∧ (b > 0 ∧ b < 10) ∧ (r > 0 ∧ r < 10) ∧
  (g * y = g) ∧
  (b = r) ∧
  (b * r = 10 * g + y) ∧ 
  (g = 8) ∧
  (y = 1) ∧
  (b = 9) ∧
  (r = 9) :=
by
  sorry

end fabian_cards_l164_164439


namespace max_distance_from_center_of_square_l164_164713

theorem max_distance_from_center_of_square :
  let A := (0, 0)
  let B := (1, 0)
  let C := (1, 1)
  let D := (0, 1)
  let O := (0.5, 0.5)
  ∃ P : ℝ × ℝ, 
  (let u := dist P A
   let v := dist P B
   let w := dist P C
   u^2 + v^2 + w^2 = 2)
  → dist O P = (1 + 2 * Real.sqrt 2) / (3 * Real.sqrt 2) :=
by sorry

end max_distance_from_center_of_square_l164_164713


namespace camille_saw_31_birds_l164_164287

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end camille_saw_31_birds_l164_164287


namespace kids_meals_sold_l164_164405

theorem kids_meals_sold (x y : ℕ) (h1 : x / y = 2) (h2 : x + y = 12) : x = 8 :=
by
  sorry

end kids_meals_sold_l164_164405


namespace dad_steps_90_l164_164113

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l164_164113


namespace simplify_and_evaluate_l164_164966

theorem simplify_and_evaluate :
  ∀ (a : ℚ), a = 3 → ((a - 1) / (a + 2) / ((a ^ 2 - 2 * a) / (a ^ 2 - 4)) - (a + 1) / a) = -2 / 3 :=
by
  intros a ha
  have : a = 3 := ha
  sorry

end simplify_and_evaluate_l164_164966


namespace quadratic_function_inequality_l164_164725

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem quadratic_function_inequality (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic_function a b c x = quadratic_function a b c (2 - x)) :
  ∀ x : ℝ, quadratic_function a b c (2 ^ x) < quadratic_function a b c (3 ^ x) :=
by
  sorry

end quadratic_function_inequality_l164_164725


namespace shaded_area_l164_164931

-- Definitions and conditions from the problem
def Square1Side := 4 -- in inches
def Square2Side := 12 -- in inches
def Triangle_DGF_similar_to_Triangle_AHF : Prop := (4 / 12) = (3 / 16)

theorem shaded_area
  (h1 : Square1Side = 4)
  (h2 : Square2Side = 12)
  (h3 : Triangle_DGF_similar_to_Triangle_AHF) :
  ∃ shaded_area : ℕ, shaded_area = 10 :=
by
  -- Calculation steps here
  sorry

end shaded_area_l164_164931


namespace percentage_reduction_in_price_l164_164400

-- Definitions for the conditions in the problem
def reduced_price_per_kg : ℕ := 30
def extra_oil_obtained_kg : ℕ := 10
def total_money_spent : ℕ := 1500

-- Definition of the original price per kg of oil
def original_price_per_kg : ℕ := 75

-- Statement to prove the percentage reduction
theorem percentage_reduction_in_price : 
  (original_price_per_kg - reduced_price_per_kg) * 100 / original_price_per_kg = 60 := by
  sorry

end percentage_reduction_in_price_l164_164400


namespace teacher_engineer_ratio_l164_164356

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers and e is the number of engineers.
  (h1 : (40 * t + 55 * e) / (t + e) = 45)
  : t = 2 * e :=
by
  sorry

end teacher_engineer_ratio_l164_164356


namespace dad_steps_l164_164146

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l164_164146


namespace tigers_wins_l164_164971

def totalGames : ℕ := 56
def losses : ℕ := 12
def ties : ℕ := losses / 2

theorem tigers_wins : totalGames - losses - ties = 38 := by
  sorry

end tigers_wins_l164_164971


namespace min_knights_l164_164799

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end min_knights_l164_164799


namespace solve_cubic_eq_l164_164692

theorem solve_cubic_eq (x : ℝ) : (8 - x)^3 = x^3 → x = 8 :=
by
  sorry

end solve_cubic_eq_l164_164692


namespace equivalent_proof_problem_l164_164386

def option_A : ℚ := 14 / 10
def option_B : ℚ := 1 + 2 / 5
def option_C : ℚ := 1 + 6 / 15
def option_D : ℚ := 1 + 3 / 8
def option_E : ℚ := 1 + 28 / 20
def target : ℚ := 7 / 5

theorem equivalent_proof_problem : option_D ≠ target :=
by {
  sorry
}

end equivalent_proof_problem_l164_164386


namespace count_triples_l164_164803

open Finset

variable (n : ℕ) (s : Finset (Fin n))

-- Definitions of the subsets A, B, C
variables {A B C : Finset (Fin n)}

-- Conditions
def condition1 : Prop := A ∩ B ∩ C = ∅
def condition2 : Prop := A ∩ B ≠ ∅
def condition3 : Prop := C ∩ B ≠ ∅

-- The theorem statement
theorem count_triples (n : ℕ) :
  ∃ (A B C : Finset (Fin n)),
    condition1 ∧ condition2 ∧ condition3 ∧
    (7 ^ n - 2 * 6 ^ n + 5 ^ n) :=
  sorry

end count_triples_l164_164803


namespace min_value_x_y_l164_164762

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 / (x + 1) + 1 / (y + 1) = 1) :
  x + y ≥ 14 :=
sorry

end min_value_x_y_l164_164762


namespace bill_new_profit_percentage_l164_164286

theorem bill_new_profit_percentage 
  (original_SP : ℝ)
  (profit_percent : ℝ)
  (increment : ℝ)
  (CP : ℝ)
  (CP_new : ℝ)
  (SP_new : ℝ)
  (Profit_new : ℝ)
  (new_profit_percent : ℝ) :
  original_SP = 439.99999999999966 →
  profit_percent = 0.10 →
  increment = 28 →
  CP = original_SP / (1 + profit_percent) →
  CP_new = CP * (1 - profit_percent) →
  SP_new = original_SP + increment →
  Profit_new = SP_new - CP_new →
  new_profit_percent = (Profit_new / CP_new) * 100 →
  new_profit_percent = 30 :=
by
  -- sorry to skip the proof
  sorry

end bill_new_profit_percentage_l164_164286


namespace problem_1_problem_2_l164_164170

-- Define propositions
def prop_p (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - m) + y^2 / m = 1)

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

def prop_s (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

-- Problems
theorem problem_1 (m : ℝ) (h : prop_s m) : m < 0 ∨ m ≥ 1 := 
  sorry

theorem problem_2 {m : ℝ} (h1 : prop_p m ∨ prop_q m) (h2 : ¬ prop_q m) : 1 ≤ m ∧ m < 2 :=
  sorry

end problem_1_problem_2_l164_164170


namespace total_amount_to_be_divided_l164_164869

theorem total_amount_to_be_divided
  (k m x : ℕ)
  (h1 : 18 * k = x)
  (h2 : 20 * m = x)
  (h3 : 13 * m = 11 * k + 1400) :
  x = 36000 := 
sorry

end total_amount_to_be_divided_l164_164869


namespace perimeter_of_one_of_the_rectangles_l164_164775

noncomputable def perimeter_of_rectangle (z w : ℕ) : ℕ :=
  2 * z

theorem perimeter_of_one_of_the_rectangles (z w : ℕ) :
  ∃ P, P = perimeter_of_rectangle z w :=
by
  use 2 * z
  sorry

end perimeter_of_one_of_the_rectangles_l164_164775


namespace arithmetic_expression_eval_l164_164721

theorem arithmetic_expression_eval :
  -1 ^ 4 + (4 - ((3 / 8 + 1 / 6 - 3 / 4) * 24)) / 5 = 0.8 := by
  sorry

end arithmetic_expression_eval_l164_164721


namespace colored_shirts_count_l164_164926

theorem colored_shirts_count (n : ℕ) (h1 : 6 = 6) (h2 : (1 / (n : ℝ)) ^ 6 = 1 / 120) : n = 2 := 
sorry

end colored_shirts_count_l164_164926


namespace margaret_mean_score_l164_164443

theorem margaret_mean_score : 
  let all_scores_sum := 832
  let cyprian_scores_count := 5
  let margaret_scores_count := 4
  let cyprian_mean_score := 92
  let cyprian_scores_sum := cyprian_scores_count * cyprian_mean_score
  (all_scores_sum - cyprian_scores_sum) / margaret_scores_count = 93 := by
  sorry

end margaret_mean_score_l164_164443


namespace sin_60_eq_sqrt_three_div_two_l164_164573

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_60_eq_sqrt_three_div_two_l164_164573


namespace sin_60_eq_sqrt3_div_2_l164_164582

theorem sin_60_eq_sqrt3_div_2 :
  ∃ (Q : ℝ × ℝ), dist Q (1, 0) = 1 ∧ angle (1, 0) Q = real.pi / 3 ∧ Q.2 = real.sqrt 3 / 2 := sorry

end sin_60_eq_sqrt3_div_2_l164_164582


namespace dollar_eval_l164_164589

def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

theorem dollar_eval (x : ℝ) : dollar (x^3 + x) (x - x^3) = 16 * x^8 :=
by
  sorry

end dollar_eval_l164_164589


namespace smallest_value_c_plus_d_l164_164014

noncomputable def problem1 (c d : ℝ) : Prop :=
c > 0 ∧ d > 0 ∧ (c^2 ≥ 12 * d) ∧ ((3 * d)^2 ≥ 4 * c)

theorem smallest_value_c_plus_d : ∃ c d : ℝ, problem1 c d ∧ c + d = 4 / Real.sqrt 3 + 4 / 9 :=
sorry

end smallest_value_c_plus_d_l164_164014


namespace simplify_expression_l164_164162

theorem simplify_expression :
  ((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4) = 43 / 4 := 
sorry

end simplify_expression_l164_164162


namespace division_scaling_l164_164058

theorem division_scaling (h : 204 / 12.75 = 16) : 2.04 / 1.275 = 16 :=
sorry

end division_scaling_l164_164058


namespace strawberry_cost_l164_164095

theorem strawberry_cost (price_per_basket : ℝ) (num_baskets : ℕ) (total_cost : ℝ)
  (h1 : price_per_basket = 16.50) (h2 : num_baskets = 4) : total_cost = 66.00 :=
by
  sorry

end strawberry_cost_l164_164095


namespace p_q_r_inequality_l164_164213

theorem p_q_r_inequality (p q r : ℝ) (h₁ : ∀ x, (x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) ↔ (x - p) * (x - q) ≤ 0) (h₂ : p < q) : p + 2 * q + 3 * r = 1 :=
by
  sorry

end p_q_r_inequality_l164_164213


namespace angle_C_is_30_degrees_l164_164486

theorem angle_C_is_30_degrees
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (A_rad: 0 ≤ A ∧ A ≤ Real.pi)
  (B_rad: 0 ≤ B ∧ B ≤ Real.pi)
  (C_rad : 0 ≤ C ∧ C ≤ Real.pi)
  (triangle_condition: A + B + C = Real.pi) :
  C = Real.pi / 6 :=
sorry

end angle_C_is_30_degrees_l164_164486


namespace shorter_piece_length_l164_164074

theorem shorter_piece_length (x : ℕ) (h1 : 177 = x + 2*x) : x = 59 :=
by sorry

end shorter_piece_length_l164_164074


namespace difference_of_squares_144_l164_164572

theorem difference_of_squares_144 (n : ℕ) (h : 3 * n + 3 < 150) : (n + 2)^2 - n^2 = 144 :=
by
  -- Given the conditions, we need to show this holds.
  sorry

end difference_of_squares_144_l164_164572


namespace sin_60_eq_sqrt3_div_2_l164_164575

theorem sin_60_eq_sqrt3_div_2 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  -- proof skipped
  sorry

end sin_60_eq_sqrt3_div_2_l164_164575


namespace B_completes_work_in_12_hours_l164_164266

theorem B_completes_work_in_12_hours:
  let A := 1 / 4
  let C := (1 / 2) - A
  let B := (1 / 3) - C
  (1 / B) = 12 :=
by
  -- placeholder for the proof
  sorry

end B_completes_work_in_12_hours_l164_164266


namespace inscribed_circle_radius_A_B_D_l164_164044

theorem inscribed_circle_radius_A_B_D (AB CD: ℝ) (angleA acuteAngleD: Prop)
  (M N: Type) (MN: ℝ) (area_trapezoid: ℝ)
  (radius: ℝ) : 
  AB = 2 ∧ CD = 3 ∧ angleA ∧ acuteAngleD ∧ MN = 4 ∧ area_trapezoid = (26 * Real.sqrt 2) / 3 
  → radius = (16 * Real.sqrt 2) / (15 + Real.sqrt 129) :=
by
  intro h
  sorry

end inscribed_circle_radius_A_B_D_l164_164044


namespace range_of_function_l164_164430

theorem range_of_function : 
  ∀ y : ℝ, 
  (∃ x : ℝ, y = x^2 + 1) ↔ (y ≥ 1) :=
by
  sorry

end range_of_function_l164_164430


namespace infinite_series_sum_l164_164414

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l164_164414


namespace robert_reading_books_l164_164957

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l164_164957


namespace line_equation_midpoint_ellipse_l164_164172

theorem line_equation_midpoint_ellipse (x1 y1 x2 y2 : ℝ) 
  (h_midpoint_x : x1 + x2 = 4) (h_midpoint_y : y1 + y2 = 2)
  (h_ellipse_x1_y1 : (x1^2) / 12 + (y1^2) / 4 = 1) (h_ellipse_x2_y2 : (x2^2) / 12 + (y2^2) / 4 = 1) :
  2 * (x1 - x2) + 3 * (y1 - y2) = 0 :=
sorry

end line_equation_midpoint_ellipse_l164_164172


namespace intersection_points_l164_164524

-- Define the four line equations
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℝ) : Prop := 5 * x - 15 * y = 15

-- State the theorem for intersection points
theorem intersection_points : 
  (line1 (18/11) (13/11) ∧ line2 (18/11) (13/11)) ∧ 
  (line2 (21/11) (8/11) ∧ line3 (21/11) (8/11)) :=
by
  sorry

end intersection_points_l164_164524


namespace no_intersection_l164_164776

def M := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def N (a : ℝ) := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = a }

theorem no_intersection (a : ℝ) : M ∩ (N a) = ∅ ↔ a ∈ (Set.Ioo (2-Real.sqrt 2) (2+Real.sqrt 2)) := 
by 
  sorry

end no_intersection_l164_164776


namespace particle_probability_l164_164275

theorem particle_probability 
  (P : ℕ → ℝ) (n : ℕ)
  (h1 : P 0 = 1)
  (h2 : P 1 = 2 / 3)
  (h3 : ∀ n ≥ 3, P n = 2 / 3 * P (n-1) + 1 / 3 * P (n-2)) :
  P n = 2 / 3 + 1 / 12 * (1 - (-1 / 3)^(n-1)) := 
sorry

end particle_probability_l164_164275


namespace cos_pi_minus_alpha_l164_164164

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π - α) = - (1 / 3) :=
by
  sorry

end cos_pi_minus_alpha_l164_164164


namespace range_of_m_l164_164325

-- Define the conditions
theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, (m-1) * x^2 + 2 * x + 1 = 0 → 
     (m-1 ≠ 0) ∧ 
     (4 - 4 * (m - 1) > 0)) ↔ 
    (m < 2 ∧ m ≠ 1) :=
sorry

end range_of_m_l164_164325


namespace R_geq_2r_l164_164965

variable {a b c : ℝ} (s : ℝ) (Δ : ℝ) (R r : ℝ)

-- Assuming conditions from the problem
def circumradius (a b c Δ : ℝ) : ℝ := (a * b * c) / (4 * Δ)
def inradius (Δ s : ℝ) : ℝ := Δ / s
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem R_geq_2r (h1 : R = circumradius a b c Δ)
                (h2 : r = inradius Δ s)
                (h3 : s = semi_perimeter a b c) :
  R ≥ 2 * r :=
by {
  -- Proof would be provided here
  sorry
}

end R_geq_2r_l164_164965


namespace students_count_l164_164972

theorem students_count (n : ℕ) (avg_age_n_students : ℕ) (sum_age_7_students1 : ℕ) (sum_age_7_students2 : ℕ) (last_student_age : ℕ) :
  avg_age_n_students = 15 →
  sum_age_7_students1 = 7 * 14 →
  sum_age_7_students2 = 7 * 16 →
  last_student_age = 15 →
  (sum_age_7_students1 + sum_age_7_students2 + last_student_age = avg_age_n_students * n) →
  n = 15 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end students_count_l164_164972


namespace sqrt_c_is_202_l164_164635

theorem sqrt_c_is_202 (a b c : ℝ) (h1 : a + b = -2020) (h2 : a * b = c) (h3 : a / b + b / a = 98) : 
  Real.sqrt c = 202 :=
by
  sorry

end sqrt_c_is_202_l164_164635


namespace expected_number_of_threes_l164_164258

noncomputable def expected_threes_on_two_8sided_dice : ℚ :=
  ∑ i in finset.range 3, (1/8 : ℚ) ^ i * (7/8) ^ (2 - i) * (nat.choose 2 i)

theorem expected_number_of_threes (dice : nat) : 
  dice = 2 ∧ ∀ k, 1 <= k ∧ k <= 8 → (1/8 : ℚ) = (1 : ℚ) / 8 :=
begin
   have H : expected_threes_on_two_8sided_dice = 1 / 4, by sorry,
   exact H
end

end expected_number_of_threes_l164_164258


namespace household_peak_consumption_l164_164924

theorem household_peak_consumption
  (p_orig p_peak p_offpeak : ℝ)
  (consumption : ℝ)
  (monthly_savings : ℝ)
  (x : ℝ)
  (h_orig : p_orig = 0.52)
  (h_peak : p_peak = 0.55)
  (h_offpeak : p_offpeak = 0.35)
  (h_consumption : consumption = 200)
  (h_savings : monthly_savings = 0.10) :
  (p_orig - p_peak) * x + (p_orig - p_offpeak) * (consumption - x) ≥ p_orig * consumption * monthly_savings → x ≤ 118 :=
sorry

end household_peak_consumption_l164_164924


namespace dad_steps_l164_164147

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l164_164147


namespace train_length_l164_164555

variable (L : ℝ) -- The length of the train

def length_of_platform : ℝ := 250 -- The length of the platform

def time_to_cross_platform : ℝ := 33 -- Time to cross the platform in seconds

def time_to_cross_pole : ℝ := 18 -- Time to cross the signal pole in seconds

-- The speed of the train is constant whether it crosses the platform or the signal pole.
-- Therefore, we equate the expressions for speed and solve for L.
theorem train_length (h1 : time_to_cross_platform * L = time_to_cross_pole * (L + length_of_platform)) :
  L = 300 :=
by
  -- Proof will be here
  sorry

end train_length_l164_164555


namespace min_omega_symmetry_l164_164463

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem min_omega_symmetry :
  ∃ (omega : ℝ), omega > 0 ∧ 
  (∀ x : ℝ, Real.cos (omega * (x - π / 12)) = Real.cos (omega * (2 * (π / 4) - x) - omega * π / 12) ) ∧ 
  (∀ ω_, ω_ > 0 → 
  (∀ x : ℝ, Real.cos (ω_ * (x - π / 12)) = Real.cos (ω_ * (2 * (π / 4) - x) - ω_ * π / 12) → 
  omega ≤ ω_)) ∧ omega = 6 :=
sorry

end min_omega_symmetry_l164_164463


namespace sum_of_side_lengths_l164_164510

theorem sum_of_side_lengths (A B C : ℕ) (hA : A = 10) (h_nat_B : B > 0) (h_nat_C : C > 0)
(h_eq_area : B^2 + C^2 = A^2) : B + C = 14 :=
sorry

end sum_of_side_lengths_l164_164510


namespace Anne_wander_time_l164_164473

theorem Anne_wander_time (distance speed : ℝ) (h1 : distance = 3.0) (h2 : speed = 2.0) : distance / speed = 1.5 := by
  -- Given conditions
  sorry

end Anne_wander_time_l164_164473


namespace ara_height_l164_164562

theorem ara_height (shea_height_now : ℝ) (shea_growth_percent : ℝ) (ara_growth_fraction : ℝ)
    (height_now : shea_height_now = 75) (growth_percent : shea_growth_percent = 0.25) 
    (growth_fraction : ara_growth_fraction = (2/3)) : 
    ∃ ara_height_now : ℝ, ara_height_now = 70 := by
  sorry

end ara_height_l164_164562


namespace discriminant_of_quadratic_l164_164733

-- Define the quadratic equation coefficients
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

-- Define the discriminant function
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- State the theorem
theorem discriminant_of_quadratic :
  discriminant a b c = 81 / 4 :=
by
  -- We provide the result of the computation directly
  sorry

end discriminant_of_quadratic_l164_164733


namespace find_number_l164_164850

theorem find_number {x : ℝ} (h : (1/3) * x = 130.00000000000003) : x = 390 := 
sorry

end find_number_l164_164850


namespace simplify_and_evaluate_l164_164354

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 2) (h₃ : x ≠ -2) :
  ((x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2 * x + 1))) = x + 1 ∧ ((x = 1) → (x + 1 = 2)) :=
by
  sorry

end simplify_and_evaluate_l164_164354


namespace number_line_steps_l164_164647

theorem number_line_steps (total_steps : ℕ) (total_distance : ℕ) (steps_taken : ℕ) (result_distance : ℕ) 
  (h1 : total_distance = 36) (h2 : total_steps = 9) (h3 : steps_taken = 6) : 
  result_distance = (steps_taken * (total_distance / total_steps)) → result_distance = 24 :=
by
  intros H
  sorry

end number_line_steps_l164_164647


namespace books_read_l164_164962

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l164_164962


namespace sum_exterior_angles_triangle_and_dodecagon_l164_164280

-- Definitions derived from conditions
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

-- Proof problem statement
theorem sum_exterior_angles_triangle_and_dodecagon :
  is_polygon 3 ∧ is_polygon 12 → sum_exterior_angles 3 + sum_exterior_angles 12 = 720 :=
by
  sorry

end sum_exterior_angles_triangle_and_dodecagon_l164_164280


namespace sum_of_denominators_of_fractions_l164_164513

theorem sum_of_denominators_of_fractions {a b : ℕ} (ha : 3 * a / 5 * b + 2 * a / 9 * b + 4 * a / 15 * b = 28 / 45) (gcd_ab : Nat.gcd a b = 1) :
  5 * b + 9 * b + 15 * b = 203 := sorry

end sum_of_denominators_of_fractions_l164_164513


namespace solve_cubed_root_equation_l164_164808

theorem solve_cubed_root_equation :
  (∃ x : ℚ, (5 - 2 / x) ^ (1 / 3) = -3) ↔ x = 1 / 16 := 
by
  sorry

end solve_cubed_root_equation_l164_164808


namespace computer_sale_price_percent_l164_164706

theorem computer_sale_price_percent (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) :
  original_price = 500 ∧ discount1 = 0.25 ∧ discount2 = 0.10 ∧ discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price * 100 = 64.13 :=
by
  intro h
  sorry

end computer_sale_price_percent_l164_164706


namespace find_Finley_age_l164_164027

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end find_Finley_age_l164_164027


namespace dad_steps_l164_164126

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l164_164126


namespace proportion_equal_l164_164531

theorem proportion_equal (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 :=
by
  sorry

end proportion_equal_l164_164531


namespace value_of_g_neg2_l164_164787

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem value_of_g_neg2 : g (-2) = -3 := 
by sorry

end value_of_g_neg2_l164_164787


namespace infinitely_many_k_numbers_unique_k_4_l164_164736

theorem infinitely_many_k_numbers_unique_k_4 :
  ∀ k : ℕ, (∃ n : ℕ, (∃ r : ℕ, n = r * (r + k)) ∧ (∃ m : ℕ, n = m^2 - k)
          ∧ ∀ N : ℕ, ∃ r : ℕ, ∃ m : ℕ, N < r ∧ (r * (r + k) = m^2 - k)) ↔ k = 4 :=
by
  sorry

end infinitely_many_k_numbers_unique_k_4_l164_164736


namespace calculate_loss_percentage_l164_164548

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end calculate_loss_percentage_l164_164548


namespace percentage_less_than_l164_164478

theorem percentage_less_than (x y : ℝ) (P : ℝ) (h1 : y = 1.6667 * x) (h2 : x = (1 - P / 100) * y) : P = 66.67 :=
sorry

end percentage_less_than_l164_164478


namespace silvia_order_total_cost_l164_164029

theorem silvia_order_total_cost :
  let quiche_price : ℝ := 15
  let croissant_price : ℝ := 3
  let biscuit_price : ℝ := 2
  let quiche_count : ℝ := 2
  let croissant_count : ℝ := 6
  let biscuit_count : ℝ := 6
  let discount_rate : ℝ := 0.10
  let pre_discount_total : ℝ := (quiche_price * quiche_count) + (croissant_price * croissant_count) + (biscuit_price * biscuit_count)
  let discount_amount : ℝ := pre_discount_total * discount_rate
  let post_discount_total : ℝ := pre_discount_total - discount_amount
  pre_discount_total > 50 → post_discount_total = 54 :=
by
  sorry

end silvia_order_total_cost_l164_164029


namespace find_value_of_y_l164_164901

theorem find_value_of_y (x y : ℚ) 
  (h1 : x = 51) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 63000) : 
  y = 8 / 17 := 
by 
  sorry

end find_value_of_y_l164_164901


namespace not_possible_to_cover_l164_164168

namespace CubeCovering

-- Defining the cube and its properties
def cube_side_length : ℕ := 4
def face_area := cube_side_length * cube_side_length
def total_faces : ℕ := 6
def faces_to_cover : ℕ := 3

-- Defining the paper strips and their properties
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def strip_area := strip_length * strip_width
def num_strips : ℕ := 16

-- Calculate the total area to cover
def total_area_to_cover := faces_to_cover * face_area
def total_area_strips := num_strips * strip_area

-- Statement: Prove that it is not possible to cover the three faces
theorem not_possible_to_cover : total_area_to_cover = 48 → total_area_strips = 48 → false := by
  intro h1 h2
  sorry

end CubeCovering

end not_possible_to_cover_l164_164168


namespace dad_steps_l164_164121

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l164_164121


namespace evaluate_expression_l164_164832

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 8 = 43046724 := 
by
  sorry

end evaluate_expression_l164_164832


namespace jonessa_take_home_pay_l164_164233

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l164_164233


namespace percentage_difference_l164_164267

theorem percentage_difference : (0.5 * 56) - (0.3 * 50) = 13 := by
  sorry

end percentage_difference_l164_164267


namespace complex_purely_imaginary_l164_164920

theorem complex_purely_imaginary (x : ℝ) :
  (x^2 - 1 = 0) → (x - 1 ≠ 0) → x = -1 :=
by
  intro h1 h2
  sorry

end complex_purely_imaginary_l164_164920


namespace determine_C_plus_D_l164_164198

theorem determine_C_plus_D (A B C D : ℕ) 
  (hA : A ≠ 0) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D → 
  C + D = 5 :=
by
    sorry

end determine_C_plus_D_l164_164198


namespace find_m_l164_164318

def vec (α : Type*) := (α × α)
def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) :
  let a : vec ℝ := (1, 3)
  let b : vec ℝ := (-2, m)
  let c : vec ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  dot_product a c = 0 → m = -1 :=
by
  sorry

end find_m_l164_164318


namespace sticker_price_l164_164181

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l164_164181


namespace find_set_A_l164_164468

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5})
variable (h1 : (U \ A) ∩ B = {0, 4})
variable (h2 : (U \ A) ∩ (U \ B) = {3, 5})

theorem find_set_A :
  A = {1, 2} :=
by
  sorry

end find_set_A_l164_164468


namespace find_first_odd_number_l164_164348

theorem find_first_odd_number (x : ℤ)
  (h : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) : x = 7 :=
by
  sorry

end find_first_odd_number_l164_164348


namespace dad_steps_l164_164137

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l164_164137


namespace sum_squares_inequality_l164_164249

theorem sum_squares_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
(h_sum : x + y + z = 3) : 
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := 
by 
  sorry

end sum_squares_inequality_l164_164249


namespace square_division_l164_164654

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end square_division_l164_164654


namespace determine_CD_l164_164930

theorem determine_CD (AB : ℝ) (BD : ℝ) (BC : ℝ) (CD : ℝ) (Angle_ADB : ℝ)
  (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30)
  (h2 : Angle_ADB = 90)
  (h3 : sin_A = 4/5)
  (h4 : sin_C = 1/5)
  (h5 : BD = sin_A * AB)
  (h6 : BC = BD / sin_C) :
  CD = 24 * Real.sqrt 23 := by
  sorry

end determine_CD_l164_164930


namespace model_N_completion_time_l164_164857

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end model_N_completion_time_l164_164857


namespace whole_number_M_l164_164277

theorem whole_number_M (M : ℤ) (hM : 9 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) : M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end whole_number_M_l164_164277


namespace triangle_cosine_condition_l164_164452

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Definitions according to the problem conditions
def law_of_sines (a b : ℝ) (A B : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B

theorem triangle_cosine_condition (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : law_of_sines a b A B)
  (h1 : a > b) : Real.cos (2 * A) < Real.cos (2 * B) ↔ a > b :=
by
  sorry

end triangle_cosine_condition_l164_164452


namespace triangle_side_length_l164_164171

theorem triangle_side_length {A B C : Type*} 
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end triangle_side_length_l164_164171


namespace christen_potatoes_peeled_l164_164469

-- Define the initial conditions and setup
def initial_potatoes := 50
def homer_rate := 4
def christen_rate := 6
def time_homer_alone := 5
def combined_rate := homer_rate + christen_rate

-- Calculate the number of potatoes peeled by Homer alone in the first 5 minutes
def potatoes_peeled_by_homer_alone := time_homer_alone * homer_rate

-- Calculate the remaining potatoes after Homer peeled alone
def remaining_potatoes := initial_potatoes - potatoes_peeled_by_homer_alone

-- Calculate the time taken for Homer and Christen to peel the remaining potatoes together
def time_to_finish_together := remaining_potatoes / combined_rate

-- Calculate the number of potatoes peeled by Christen during the shared work period
def potatoes_peeled_by_christen := christen_rate * time_to_finish_together

-- The final theorem we need to prove
theorem christen_potatoes_peeled : potatoes_peeled_by_christen = 18 := by
  sorry

end christen_potatoes_peeled_l164_164469


namespace binomial_sixteen_twelve_eq_l164_164419

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l164_164419


namespace part_I_part_II_l164_164614

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

def monotonicity_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

theorem part_I (a : ℝ) : 
  monotonicity_in_interval (f a) 0 (Real.log 3) = monotonicity_in_interval (F a) 0 (Real.log 3) ↔ a ≤ -3 :=
sorry

theorem part_II (a : ℝ) (ha : a ∈ Set.Iic (-1 / Real.exp 2)) : 
  (∃ x, x > 0 ∧ g a x = M) → M ≥ 0 :=
sorry

end part_I_part_II_l164_164614


namespace west_movement_is_negative_seven_l164_164649

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l164_164649


namespace fraction_sum_l164_164602

theorem fraction_sum : (3 / 8) + (9 / 14) = (57 / 56) := by
  sorry

end fraction_sum_l164_164602


namespace molly_age_l164_164697

theorem molly_age
  (S M : ℕ)
  (h1 : S / M = 4 / 3)
  (h2 : S + 6 = 30) :
  M = 18 :=
sorry

end molly_age_l164_164697


namespace similarity_coordinates_l164_164482

theorem similarity_coordinates {B B1 : ℝ × ℝ} 
  (h₁ : ∃ (k : ℝ), k = 2 ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → x₁ = x / k ∨ x₁ = x / -k) ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → y₁ = y / k ∨ y₁ = y / -k))
  (h₂ : B = (-4, -2)) :
  B1 = (-2, -1) ∨ B1 = (2, 1) :=
sorry

end similarity_coordinates_l164_164482


namespace sum_of_angles_x_y_l164_164539

theorem sum_of_angles_x_y :
  let num_arcs := 15
  let angle_per_arc := 360 / num_arcs
  let central_angle_x := 3 * angle_per_arc
  let central_angle_y := 5 * angle_per_arc
  let inscribed_angle (central_angle : ℝ) := central_angle / 2
  let angle_x := inscribed_angle central_angle_x
  let angle_y := inscribed_angle central_angle_y
  angle_x + angle_y = 96 := 
  sorry

end sum_of_angles_x_y_l164_164539


namespace calculate_expression_l164_164570

theorem calculate_expression : 
  -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5 / 2 :=
by
  sorry

end calculate_expression_l164_164570


namespace value_of_x_squared_plus_y_squared_l164_164923

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : 
  x^2 + y^2 = 4 :=
  sorry

end value_of_x_squared_plus_y_squared_l164_164923


namespace largest_angle_l164_164333

theorem largest_angle (y : ℝ) (h : 40 + 70 + y = 180) : y = 70 :=
by
  sorry

end largest_angle_l164_164333


namespace Maddie_spent_on_tshirts_l164_164948

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l164_164948


namespace fraction_sum_l164_164700

theorem fraction_sum :
  (1 / 3 : ℚ) + (1 / 2 : ℚ) - (5 / 6 : ℚ) + (1 / 5 : ℚ) + (1 / 4 : ℚ) - (9 / 20 : ℚ) - (2 / 15 : ℚ) = -2 / 15 :=
by {
  sorry
}

end fraction_sum_l164_164700


namespace simplify_sum_l164_164687

theorem simplify_sum : 
  (-1: ℤ)^(2010) + (-1: ℤ)^(2011) + (1: ℤ)^(2012) + (-1: ℤ)^(2013) = -2 := by
  sorry

end simplify_sum_l164_164687


namespace cost_of_apples_l164_164369

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164369


namespace geometric_common_ratio_l164_164747

theorem geometric_common_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (a₁ * (1 - q ^ 3)) / (1 - q) / ((a₁ * (1 - q ^ 2)) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  -- Proof omitted
  sorry

end geometric_common_ratio_l164_164747


namespace arrangement_count_SUCCESS_l164_164876

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l164_164876


namespace meaningful_expression_range_l164_164769

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l164_164769


namespace value_of_a_8_l164_164909

-- Definitions of the sequence and sum of first n terms
def sum_first_terms (S : ℕ → ℕ) := ∀ n : ℕ, n > 0 → S n = n^2

-- Definition of the term a_n
def a_n (S : ℕ → ℕ) (n : ℕ) := S n - S (n - 1)

-- The theorem we want to prove: a_8 = 15
theorem value_of_a_8 (S : ℕ → ℕ) (h_sum : sum_first_terms S) : a_n S 8 = 15 :=
by
  sorry

end value_of_a_8_l164_164909


namespace tim_income_percent_less_than_juan_l164_164643

theorem tim_income_percent_less_than_juan (T M J : ℝ) (h1 : M = 1.5 * T) (h2 : M = 0.9 * J) :
  (J - T) / J = 0.4 :=
by
  sorry

end tim_income_percent_less_than_juan_l164_164643


namespace sum_D_E_F_l164_164362

theorem sum_D_E_F (D E F : ℤ) (h : ∀ x, x^3 + D * x^2 + E * x + F = (x + 3) * x * (x - 4)) : 
  D + E + F = -13 :=
by
  sorry

end sum_D_E_F_l164_164362


namespace west_move_7m_l164_164650

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end west_move_7m_l164_164650


namespace complement_M_in_U_l164_164616

-- Define the universal set U and set M
def U : Finset ℕ := {4, 5, 6, 8, 9}
def M : Finset ℕ := {5, 6, 8}

-- Define the complement of M in U
def complement (U M : Finset ℕ) : Finset ℕ := U \ M

-- Prove that the complement of M in U is {4, 9}
theorem complement_M_in_U : complement U M = {4, 9} := by
  sorry

end complement_M_in_U_l164_164616


namespace fraction_identity_l164_164474

theorem fraction_identity (a b : ℝ) (h : a ≠ b) (h₁ : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end fraction_identity_l164_164474


namespace find_m_value_l164_164323

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end find_m_value_l164_164323


namespace part_I_part_II_l164_164175

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 4)

-- Part I
theorem part_I : f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2 :=
by
  sorry

-- Part II
theorem part_II (x : ℝ) (h : f x = Real.sqrt 2 / 3) : Real.sin (2 * x) = 5 / 9 :=
by
  sorry

end part_I_part_II_l164_164175


namespace machines_working_together_l164_164678

theorem machines_working_together (x : ℝ) :
  (∀ P Q R : ℝ, P = x + 4 ∧ Q = x + 2 ∧ R = 2 * x + 2 ∧ (1 / P + 1 / Q + 1 / R = 1 / x)) ↔ (x = 2 / 3) :=
by
  sorry

end machines_working_together_l164_164678


namespace inequality_solution_l164_164662

theorem inequality_solution (x : ℝ) : 
  (x + 10) / (x^2 + 2 * x + 5) ≥ 0 ↔ x ∈ Set.Ici (-10) :=
sorry

end inequality_solution_l164_164662


namespace sequence_converges_l164_164451

theorem sequence_converges (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
    ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(a n)^ (1/n) - l| < ε :=
by
  sorry

end sequence_converges_l164_164451


namespace relationship_between_f_l164_164942

variable (f : ℝ → ℝ)

-- Conditions
variables (x₁ x₂ : ℝ) (h₁ : f (x + 1) = f (-x + 1)) (h₂ : ∀ x, (x - 1) * f' x < 0)
variables (h₃ : x₁ < x₂) (h₄ : x₁ + x₂ > 2)

-- The statement to prove
theorem relationship_between_f (hf_diff : Differentiable ℝ f) : f x₁ > f x₂ := sorry

end relationship_between_f_l164_164942


namespace dad_steps_l164_164130

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l164_164130


namespace remain_when_divided_l164_164899

open Nat

def sum_of_squares_primes_mod_3 (ps : List ℕ) : ℕ :=
  (ps.map (λ p, p^2)).sum % 3

theorem remain_when_divided (ps : List ℕ) (h₀ : ps.length = 99) 
  (h₁ : ∀ p ∈ ps, Prime p) 
  (h₂ : ∀ p₁ p₂ ∈ ps, p₁ ≠ p₂ → p₁ ≠ p₂) :
  sum_of_squares_primes_mod_3 ps = 0 ∨ sum_of_squares_primes_mod_3 ps = 2 := by {
  sorry
}

end remain_when_divided_l164_164899


namespace Abby_has_17_quarters_l164_164717

theorem Abby_has_17_quarters (q n : ℕ) (h1 : q + n = 23) (h2 : 25 * q + 5 * n = 455) : q = 17 :=
sorry

end Abby_has_17_quarters_l164_164717


namespace intersection_eq_l164_164615

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x - 1) * (x + 2) < 0}

-- Define the intended intersection result
def C : Set ℤ := {-1, 0}

-- The theorem to prove
theorem intersection_eq : A ∩ {x | (x - 1) * (x + 2) < 0} = C := by
  sorry

end intersection_eq_l164_164615


namespace sin_minus_cos_eq_neg_sqrt2_div3_l164_164897

theorem sin_minus_cos_eq_neg_sqrt2_div3
  (θ : ℝ)
  (h1 : sin θ + cos θ = 4 / 3)
  (h2 : 0 < θ ∧ θ < π / 4) :
  sin θ - cos θ = -real.sqrt 2 / 3 := 
sorry

end sin_minus_cos_eq_neg_sqrt2_div3_l164_164897


namespace dartboard_odd_sum_probability_l164_164798

theorem dartboard_odd_sum_probability :
  let innerR := 4
  let outerR := 8
  let inner_points := [3, 1, 1]
  let outer_points := [2, 3, 3]
  let total_area := π * outerR^2
  let inner_area := π * innerR^2
  let outer_area := total_area - inner_area
  let each_inner_area := inner_area / 3
  let each_outer_area := outer_area / 3
  let odd_area := 2 * each_inner_area + 2 * each_outer_area
  let even_area := each_inner_area + each_outer_area
  let P_odd := odd_area / total_area
  let P_even := even_area / total_area
  let odd_sum_prob := 2 * (P_odd * P_even)
  odd_sum_prob = 4 / 9 := by
    sorry

end dartboard_odd_sum_probability_l164_164798


namespace average_rate_of_change_is_7_l164_164234

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define the proof problem
theorem average_rate_of_change_is_7 : 
  ((f b - f a) / (b - a)) = 7 :=
by 
  -- The proof would go here
  sorry

end average_rate_of_change_is_7_l164_164234


namespace grant_earnings_l164_164914

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l164_164914


namespace calculator_change_problem_l164_164085

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l164_164085


namespace cos_double_angle_zero_l164_164745

theorem cos_double_angle_zero (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = Real.cos (Real.pi / 6 + α)) : Real.cos (2 * α) = 0 := 
sorry

end cos_double_angle_zero_l164_164745


namespace find_solutions_l164_164887

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 14*x - 8)) = 0

theorem find_solutions : {x : ℝ | equation x} = {2, -4, 1, -8} :=
  by
  sorry

end find_solutions_l164_164887


namespace polygon_perimeter_eq_21_l164_164549

-- Definitions and conditions from the given problem
def rectangle_side_a := 6
def rectangle_side_b := 4
def triangle_hypotenuse := 5

-- The combined polygon perimeter proof statement
theorem polygon_perimeter_eq_21 :
  let rectangle_perimeter := 2 * (rectangle_side_a + rectangle_side_b)
  let adjusted_perimeter := rectangle_perimeter - rectangle_side_b + triangle_hypotenuse
  adjusted_perimeter = 21 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end polygon_perimeter_eq_21_l164_164549


namespace mr_blue_carrots_l164_164644

theorem mr_blue_carrots :
  let steps_length := 3 -- length of each step in feet
  let garden_length_steps := 25 -- length of garden in steps
  let garden_width_steps := 35 -- width of garden in steps
  let length_feet := garden_length_steps * steps_length -- length of garden in feet
  let width_feet := garden_width_steps * steps_length -- width of garden in feet
  let area_feet2 := length_feet * width_feet -- area of garden in square feet
  let yield_rate := 3 / 4 -- yield rate of carrots in pounds per square foot
  let expected_yield := area_feet2 * yield_rate -- expected yield in pounds
  expected_yield = 5906.25
:= by
  sorry

end mr_blue_carrots_l164_164644


namespace Cathy_total_money_l164_164103

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l164_164103


namespace sum_of_four_digit_multiples_of_5_l164_164052

theorem sum_of_four_digit_multiples_of_5 :
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  S = 9895500 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end sum_of_four_digit_multiples_of_5_l164_164052


namespace quadratic_inequality_l164_164956

theorem quadratic_inequality 
  (a b c : ℝ) 
  (h₁ : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1)
  (x : ℝ) 
  (hx : |x| ≤ 1) : 
  |c * x^2 + b * x + a| ≤ 2 := 
sorry

end quadratic_inequality_l164_164956


namespace segment_length_after_reflection_l164_164682

structure Point :=
(x : ℝ)
(y : ℝ)

def reflect_over_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

def distance (p1 p2 : Point) : ℝ :=
abs (p1.y - p2.y)

theorem segment_length_after_reflection :
  let C : Point := {x := -3, y := 2}
  let C' : Point := reflect_over_x_axis C
  distance C C' = 4 :=
by
  sorry

end segment_length_after_reflection_l164_164682


namespace line_tangent_to_ellipse_l164_164324

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1 → m^2 = 35/9) := 
sorry

end line_tangent_to_ellipse_l164_164324


namespace trip_to_museum_l164_164976

theorem trip_to_museum (x y z w : ℕ) 
  (h2 : y = 2 * x) 
  (h3 : z = 2 * x - 6) 
  (h4 : w = x + 9) 
  (htotal : x + y + z + w = 75) : 
  x = 12 := 
by 
  sorry

end trip_to_museum_l164_164976


namespace correct_average_l164_164392

theorem correct_average (avg: ℕ) (n: ℕ) (incorrect: ℕ) (correct: ℕ) 
  (h_avg : avg = 16) (h_n : n = 10) (h_incorrect : incorrect = 25) (h_correct : correct = 35) :
  (avg * n + (correct - incorrect)) / n = 17 := 
by
  sorry

end correct_average_l164_164392


namespace trapezoid_is_proposition_l164_164388

-- Define what it means to be a proposition
def is_proposition (s : String) : Prop := ∃ b : Bool, (s = "A trapezoid is a quadrilateral" ∨ s = "Construct line AB" ∨ s = "x is an integer" ∨ s = "Will it snow today?") ∧ 
  (b → s = "A trapezoid is a quadrilateral") 

-- Main proof statement
theorem trapezoid_is_proposition : is_proposition "A trapezoid is a quadrilateral" :=
  sorry

end trapezoid_is_proposition_l164_164388


namespace find_right_triangle_sides_l164_164247

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_condition (a b c : ℕ) : Prop :=
  a * b = 3 * (a + b + c)

theorem find_right_triangle_sides :
  ∃ (a b c : ℕ),
    is_right_triangle a b c ∧ area_condition a b c ∧
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
sorry

end find_right_triangle_sides_l164_164247


namespace robert_reading_books_l164_164958

theorem robert_reading_books (pages_per_hour : ℕ) (pages_per_book : ℕ) (total_hours : ℕ) 
  (h1 : pages_per_hour = 120) (h2 : pages_per_book = 360) (h3 : total_hours = 8) : 
  (total_hours / (pages_per_book / pages_per_hour) : ℝ).toInt = 2 :=
by
  sorry

end robert_reading_books_l164_164958


namespace fair_coin_expected_value_1000_l164_164624

open ProbabilityTheory

noncomputable def coin_flip_expected_value (n : ℕ) : ℝ :=
  ∑ k in finset.range (n+1), (n*(n-1))/4

theorem fair_coin_expected_value_1000 :
  coin_flip_expected_value 1000 = 249750 := by
  sorry

end fair_coin_expected_value_1000_l164_164624


namespace students_passed_both_l164_164481

noncomputable def F_H : ℝ := 32
noncomputable def F_E : ℝ := 56
noncomputable def F_HE : ℝ := 12
noncomputable def total_percentage : ℝ := 100

theorem students_passed_both : (total_percentage - (F_H + F_E - F_HE)) = 24 := by
  sorry

end students_passed_both_l164_164481


namespace value_is_sqrt_5_over_3_l164_164216

noncomputable def findValue (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) : ℝ :=
  (x + y) / (x - y)

theorem value_is_sqrt_5_over_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) :
  findValue x y h1 h2 h3 = Real.sqrt (5 / 3) :=
sorry

end value_is_sqrt_5_over_3_l164_164216


namespace profit_percentage_is_50_l164_164658

noncomputable def cost_of_machine := 11000
noncomputable def repair_cost := 5000
noncomputable def transportation_charges := 1000
noncomputable def selling_price := 25500

noncomputable def total_cost := cost_of_machine + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost) * 100

theorem profit_percentage_is_50 : profit_percentage = 50 := by
  sorry

end profit_percentage_is_50_l164_164658


namespace findC_coordinates_l164_164929

-- Points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Defining points A, B, and stating that point C lies on the positive x-axis
def A : Point := {x := -4, y := -2}
def B : Point := {x := 0, y := -2}
def C (cx : ℝ) : Point := {x := cx, y := 0}

-- The condition that the triangle OBC is similar to triangle ABO
def isSimilar (A B O : Point) (C : Point) : Prop :=
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let OB := (B.x - O.x)^2 + (B.y - O.y)^2
  let OC := (C.x - O.x)^2 + (C.y - O.y)^2
  AB / OB = OB / OC

theorem findC_coordinates :
  ∃ (cx : ℝ), (C cx = {x := 1, y := 0} ∨ C cx = {x := 4, y := 0}) ∧
  isSimilar A B {x := 0, y := 0} (C cx) :=
by
  sorry

end findC_coordinates_l164_164929


namespace dad_steps_are_90_l164_164141

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l164_164141


namespace units_digit_of_quotient_l164_164152

theorem units_digit_of_quotient : 
  let n := 1993
  let term1 := 4 ^ n
  let term2 := 6 ^ n
  (term1 + term2) % 5 = 0 →
  let quotient := (term1 + term2) / 5
  (quotient % 10 = 0) := 
by 
  sorry

end units_digit_of_quotient_l164_164152


namespace trig_eq_solution_l164_164355

open Real

theorem trig_eq_solution (x : ℝ) :
    (∃ k : ℤ, x = -arccos ((sqrt 13 - 1) / 4) + 2 * k * π) ∨ 
    (∃ k : ℤ, x = -arccos ((1 - sqrt 13) / 4) + 2 * k * π) ↔ 
    (cos 5 * x - cos 7 * x) / (sin 4 * x + sin 2 * x) = 2 * abs (sin 2 * x) := by
  sorry

end trig_eq_solution_l164_164355


namespace cylinder_volume_relation_l164_164587

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end cylinder_volume_relation_l164_164587


namespace housewife_oil_cost_l164_164867

theorem housewife_oil_cost (P R M : ℝ) (hR : R = 45) (hReduction : (P - R) = (15 / 100) * P)
  (hMoreOil : M / P = M / R + 4) : M = 150.61 := 
by
  sorry

end housewife_oil_cost_l164_164867


namespace class_heights_mode_median_l164_164226

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℕ := sorry

theorem class_heights_mode_median 
  (A : List ℕ) -- Heights of students from Class A
  (B : List ℕ) -- Heights of students from Class B
  (hA : A = [170, 170, 169, 171, 171, 171])
  (hB : B = [168, 170, 170, 172, 169, 170]) :
  mode A = 171 ∧ median B = 170 := sorry

end class_heights_mode_median_l164_164226


namespace number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l164_164003

-- Let's define the conditions.
def balls_in_first_pocket : Nat := 2
def balls_in_second_pocket : Nat := 4
def balls_in_third_pocket : Nat := 5

-- Proof for the first question
theorem number_of_ways_to_take_one_ball_from_pockets : 
  balls_in_first_pocket + balls_in_second_pocket + balls_in_third_pocket = 11 := 
by
  sorry

-- Proof for the second question
theorem number_of_ways_to_take_one_ball_each_from_pockets : 
  balls_in_first_pocket * balls_in_second_pocket * balls_in_third_pocket = 40 := 
by
  sorry

end number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l164_164003


namespace triangle_height_l164_164813

theorem triangle_height (area base height : ℝ) (h1 : area = 500) (h2 : base = 50) (h3 : area = (1 / 2) * base * height) : height = 20 :=
sorry

end triangle_height_l164_164813


namespace grant_total_earnings_l164_164910

theorem grant_total_earnings:
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := 30
  let glove_discount_rate := 0.20
  let cleats_pair_count := 2
  let cleats_price_per_pair := 10
  let glove_discount := baseball_glove * glove_discount_rate
  let glove_selling_price := baseball_glove - glove_discount
  let cleats_total := cleats_pair_count * cleats_price_per_pair
  let total_earnings := baseball_cards + baseball_bat + glove_selling_price + cleats_total
  in total_earnings = 79 :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end grant_total_earnings_l164_164910


namespace union_M_N_eq_l164_164963

open Set

-- Define M according to the condition x^2 < 15 for x in ℕ
def M : Set ℕ := {x | x^2 < 15}

-- Define N according to the correct answer
def N : Set ℕ := {x | 0 < x ∧ x < 5}

-- Prove that M ∪ N = {x | 0 ≤ x ∧ x < 5}
theorem union_M_N_eq : M ∪ N = {x : ℕ | 0 ≤ x ∧ x < 5} :=
sorry

end union_M_N_eq_l164_164963


namespace acute_triangle_on_perpendicular_lines_l164_164316

theorem acute_triangle_on_perpendicular_lines :
  ∀ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) →
  ∃ (x y z : ℝ), (x^2 = (b^2 + c^2 - a^2) / 2) ∧ (y^2 = (a^2 + c^2 - b^2) / 2) ∧ (z^2 = (a^2 + b^2 - c^2) / 2) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
by
  sorry

end acute_triangle_on_perpendicular_lines_l164_164316


namespace width_of_rectangle_l164_164335

-- Define the side length of the square and the length of the rectangle.
def side_length_square : ℝ := 12
def length_rectangle : ℝ := 18

-- Calculate the perimeter of the square.
def perimeter_square : ℝ := 4 * side_length_square

-- This definition represents the perimeter of the rectangle made from the same wire.
def perimeter_rectangle : ℝ := perimeter_square

-- Show that the width of the rectangle is 6 cm.
theorem width_of_rectangle : ∃ W : ℝ, 2 * (length_rectangle + W) = perimeter_rectangle ∧ W = 6 :=
by
  use 6
  simp [length_rectangle, perimeter_rectangle, side_length_square]
  norm_num
  sorry

end width_of_rectangle_l164_164335


namespace event_distance_l164_164410

noncomputable def distance_to_event (cost_per_mile : ℝ) (days : ℕ) (rides_per_day : ℕ) (total_cost : ℝ) : ℝ :=
  total_cost / (days * rides_per_day * cost_per_mile)

theorem event_distance 
  (cost_per_mile : ℝ)
  (days : ℕ)
  (rides_per_day : ℕ)
  (total_cost : ℝ)
  (h1 : cost_per_mile = 2.5)
  (h2 : days = 7)
  (h3 : rides_per_day = 2)
  (h4 : total_cost = 7000) : 
  distance_to_event cost_per_mile days rides_per_day total_cost = 200 :=
by {
  sorry
}

end event_distance_l164_164410


namespace lines_intersect_at_point_l164_164593

theorem lines_intersect_at_point :
  ∃ (x y : ℝ), (3 * x + 4 * y + 7 = 0) ∧ (x - 2 * y - 1 = 0) ∧ (x = -1) ∧ (y = -1) :=
by
  sorry

end lines_intersect_at_point_l164_164593


namespace sin_60_proof_l164_164580

noncomputable def sin_60_eq_sqrt3_div_2 : Prop :=
  Real.sin (π / 3) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq_sqrt3_div_2 :=
sorry

end sin_60_proof_l164_164580


namespace three_room_partition_l164_164399

open Finset

noncomputable def possible_partition (G : Type) [Fintype G] (knows : G → G → Prop) {h : ∀ a b, knows a b → knows b a}
  (no_four_chain : ∀ a b c d, ¬(knows a b ∧ knows b c ∧ knows c d)): Prop :=
  ∃ (rooms : G → Fin 3), ∀ a b, rooms a = rooms b → ¬knows a b

-- Proof is omitted.
theorem three_room_partition (G : Type) [Fintype G] (knows : G → G → Prop) {h : ∀ a b, knows a b → knows b a}
  (no_four_chain : ∀ a b c d, ¬(knows a b ∧ knows b c ∧ knows c d)) : possible_partition G knows :=
sorry

end three_room_partition_l164_164399


namespace dad_steps_l164_164135

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l164_164135


namespace absent_children_count_l164_164021

theorem absent_children_count : ∀ (total_children present_children absent_children bananas : ℕ), 
  total_children = 260 → 
  bananas = 2 * total_children → 
  bananas = 4 * present_children → 
  present_children + absent_children = total_children →
  absent_children = 130 :=
by
  intros total_children present_children absent_children bananas h1 h2 h3 h4
  sorry

end absent_children_count_l164_164021


namespace find_largest_square_area_l164_164094

def area_of_largest_square (XY YZ XZ : ℝ) (sum_of_areas : ℝ) (right_angle : Prop) : Prop :=
  sum_of_areas = XY^2 + YZ^2 + XZ^2 + 4 * YZ^2 ∧  -- sum of areas condition
  right_angle ∧                                    -- right angle condition
  XZ^2 = XY^2 + YZ^2 ∧                             -- Pythagorean theorem
  sum_of_areas = 650 ∧                             -- total area condition
  XY = YZ                                          -- assumption for simplified solving.

theorem find_largest_square_area (XY YZ XZ : ℝ) (sum_of_areas : ℝ):
  area_of_largest_square XY YZ XZ sum_of_areas (90 = 90) → 2 * XY^2 + 5 * YZ^2 = 650 → XZ^2 = 216.67 :=
sorry

end find_largest_square_area_l164_164094


namespace overlapping_segments_length_l164_164366

theorem overlapping_segments_length 
    (total_length : ℝ) 
    (actual_distance : ℝ) 
    (num_overlaps : ℕ) 
    (h1 : total_length = 98) 
    (h2 : actual_distance = 83)
    (h3 : num_overlaps = 6) :
    (total_length - actual_distance) / num_overlaps = 2.5 :=
by
  sorry

end overlapping_segments_length_l164_164366


namespace inverse_proposition_l164_164240

theorem inverse_proposition (a : ℝ) :
  (a > 1 → a > 0) → (a > 0 → a > 1) :=
by 
  intros h1 h2
  sorry

end inverse_proposition_l164_164240


namespace final_price_lower_than_budget_l164_164495

theorem final_price_lower_than_budget :
  let budget := 1500
  let T := 750 -- budget equally split for TV
  let S := 750 -- budget equally split for Sound System
  let TV_price_with_discount := (T - 150) * 0.80
  let SoundSystem_price_with_discount := S * 0.85
  let combined_price_before_tax := TV_price_with_discount + SoundSystem_price_with_discount
  let final_price_with_tax := combined_price_before_tax * 1.08
  budget - final_price_with_tax = 293.10 :=
by
  sorry

end final_price_lower_than_budget_l164_164495


namespace horse_catches_up_l164_164858

-- Definitions based on given conditions
def dog_speed := 20 -- derived from 5 steps * 4 meters
def horse_speed := 21 -- derived from 3 steps * 7 meters
def initial_distance := 30 -- dog has already run 30 meters

-- Statement to be proved
theorem horse_catches_up (d h : ℕ) (time : ℕ) :
  d = dog_speed → h = horse_speed →
  initial_distance = 30 →
  h * time = initial_distance + dog_speed * time →
  time = 600 / (h - d) ∧ h * time - initial_distance = 600 :=
by
  intros
  -- Proof placeholders
  sorry  -- Omit the actual proof steps

end horse_catches_up_l164_164858


namespace distance_after_12_seconds_time_to_travel_380_meters_l164_164668

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end distance_after_12_seconds_time_to_travel_380_meters_l164_164668


namespace cube_volume_increase_l164_164833

theorem cube_volume_increase (s : ℝ) (surface_area : ℝ) 
  (h1 : surface_area = 6 * s^2) (h2 : surface_area = 864) : 
  (1.5 * s)^3 = 5832 :=
by
  sorry

end cube_volume_increase_l164_164833


namespace calculate_distance_to_friend_l164_164780

noncomputable def distance_to_friend (d t : ℝ) : Prop :=
  (d = 45 * (t + 1)) ∧ (d = 45 + 65 * (t - 0.75))

theorem calculate_distance_to_friend : ∃ d t: ℝ, distance_to_friend d t ∧ d = 155 :=
by
  exists 155
  exists 2.4375
  sorry

end calculate_distance_to_friend_l164_164780


namespace cost_apples_l164_164374

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l164_164374


namespace initial_number_of_men_l164_164357

theorem initial_number_of_men (M A : ℕ) 
  (h1 : ((M * A) - 22 + 42 = M * (A + 2))) : M = 10 :=
by
  sorry

end initial_number_of_men_l164_164357


namespace find_m_l164_164315

noncomputable def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
noncomputable def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m (m : ℝ) : B m ⊆ A m ↔ m = 1 ∨ m = 3 :=
by
  sorry

end find_m_l164_164315


namespace correct_adjacent_book_left_l164_164986

-- Define the parameters
variable (prices : ℕ → ℕ)
variable (n : ℕ)
variable (step : ℕ)

-- Given conditions
axiom h1 : n = 31
axiom h2 : step = 2
axiom h3 : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step
axiom h4 : prices 30 = prices 15 + prices 14

-- We need to show that the adjacent book referred to is at the left of the middle book.
theorem correct_adjacent_book_left (h : n = 31) (prices_step : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step) : prices 30 = prices 15 + prices 14 := by
  sorry

end correct_adjacent_book_left_l164_164986


namespace book_pages_total_l164_164202

theorem book_pages_total
  (days_in_week : ℕ)
  (daily_read_times : ℕ)
  (pages_per_time : ℕ)
  (additional_pages_per_day : ℕ)
  (num_days : days_in_week = 7)
  (times_per_day : daily_read_times = 3)
  (pages_each_time : pages_per_time = 6)
  (extra_pages : additional_pages_per_day = 2) :
  daily_read_times * pages_per_time + additional_pages_per_day * days_in_week = 140 := 
sorry

end book_pages_total_l164_164202


namespace find_x_to_print_800_leaflets_in_3_minutes_l164_164550

theorem find_x_to_print_800_leaflets_in_3_minutes (x : ℝ) :
  (800 / 12 + 800 / x = 800 / 3) → (1 / 12 + 1 / x = 1 / 3) :=
by
  intro h
  have h1 : 800 / 12 = 200 / 3 := by norm_num
  have h2 : 800 / 3 = 800 / 3 := by norm_num
  sorry

end find_x_to_print_800_leaflets_in_3_minutes_l164_164550


namespace solution_set_inequality_l164_164673

theorem solution_set_inequality (x : ℝ) (h1 : x < -3) (h2 : x < 2) : x < -3 :=
by
  exact h1

end solution_set_inequality_l164_164673


namespace meaningful_expression_range_l164_164768

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l164_164768


namespace harry_worked_41_hours_l164_164884

def james_earnings (x : ℝ) : ℝ :=
  (40 * x) + (7 * 2 * x)

def harry_earnings (x : ℝ) (h : ℝ) : ℝ :=
  (24 * x) + (11 * 1.5 * x) + (2 * h * x)

def harry_hours_worked (h : ℝ) : ℝ :=
  24 + 11 + h

theorem harry_worked_41_hours (x : ℝ) (h : ℝ) 
  (james_worked : james_earnings x = 54 * x)
  (harry_paid_same : harry_earnings x h = james_earnings x) :
  harry_hours_worked h = 41 :=
by
  -- sorry is used to skip the proof steps
  sorry

end harry_worked_41_hours_l164_164884


namespace binomial_coefficient_of_x_l164_164236

-- definition of binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem we want to prove
theorem binomial_coefficient_of_x {x : ℕ} (h : x ≠ 0) :
  let term (k : ℕ) := binom 5 k * (1/2)^k * x^(10-3*k)
  ∃ k, 10 - 3 * k = 1 ∧ term k = (5 / 4) * x :=
begin
  sorry
end

end binomial_coefficient_of_x_l164_164236


namespace total_percent_decrease_l164_164076

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end total_percent_decrease_l164_164076


namespace factorize_x_squared_minus_nine_l164_164155

theorem factorize_x_squared_minus_nine : ∀ (x : ℝ), x^2 - 9 = (x - 3) * (x + 3) :=
by
  intro x
  exact sorry

end factorize_x_squared_minus_nine_l164_164155


namespace angle_A_value_l164_164488

/-- 
In triangle ABC, the sides opposite to angles A, B, C are a, b, and c respectively.
Given:
  - C = π / 3,
  - b = √6,
  - c = 3,
Prove that A = 5π / 12.
-/
theorem angle_A_value (a b c : ℝ) (A B C : ℝ) (hC : C = Real.pi / 3) (hb : b = Real.sqrt 6) (hc : c = 3) :
  A = 5 * Real.pi / 12 :=
sorry

end angle_A_value_l164_164488


namespace mod_residue_l164_164569

theorem mod_residue : (250 * 15 - 337 * 5 + 22) % 13 = 7 := by
  sorry

end mod_residue_l164_164569


namespace jamal_green_marbles_l164_164630

theorem jamal_green_marbles
  (Y B K T : ℕ)
  (hY : Y = 12)
  (hB : B = 10)
  (hK : K = 1)
  (h_total : 1 / T = 1 / 28) :
  T - (Y + B + K) = 5 :=
by
  -- sorry, proof goes here
  sorry

end jamal_green_marbles_l164_164630


namespace hcf_of_two_numbers_l164_164988

theorem hcf_of_two_numbers (H L P : ℕ) (h1 : L = 160) (h2 : P = 2560) (h3 : H * L = P) : H = 16 :=
by
  sorry

end hcf_of_two_numbers_l164_164988


namespace proof_problem_l164_164757

theorem proof_problem (a b c d x : ℝ)
  (h1 : c = 6 * d)
  (h2 : 2 * a = 1 / (-b))
  (h3 : abs x = 9) :
  (2 * a * b - 6 * d + c - x / 3 = -4) ∨ (2 * a * b - 6 * d + c - x / 3 = 2) :=
by
  sorry

end proof_problem_l164_164757


namespace molly_gift_cost_l164_164199

noncomputable def cost_per_package : ℕ := 5
noncomputable def num_parents : ℕ := 2
noncomputable def num_brothers : ℕ := 3
noncomputable def num_sisters_in_law : ℕ := num_brothers -- each brother is married
noncomputable def num_children_per_brother : ℕ := 2
noncomputable def num_nieces_nephews : ℕ := num_brothers * num_children_per_brother
noncomputable def total_relatives : ℕ := num_parents + num_brothers + num_sisters_in_law + num_nieces_nephews

theorem molly_gift_cost : (total_relatives * cost_per_package) = 70 := by
  sorry

end molly_gift_cost_l164_164199


namespace total_amount_spent_l164_164519

def price_per_deck (n : ℕ) : ℝ :=
if n <= 3 then 8 else if n <= 6 then 7 else 6

def promotion_price (price : ℝ) : ℝ :=
price * 0.5

def total_cost (decks_victor decks_friend : ℕ) : ℝ :=
let cost_victor :=
  if decks_victor % 2 = 0 then
    let pairs := decks_victor / 2
    price_per_deck decks_victor * pairs + promotion_price (price_per_deck decks_victor) * pairs
  else sorry
let cost_friend :=
  if decks_friend = 2 then
    price_per_deck decks_friend + promotion_price (price_per_deck decks_friend)
  else sorry
cost_victor + cost_friend

theorem total_amount_spent : total_cost 6 2 = 43.5 := sorry

end total_amount_spent_l164_164519


namespace pages_per_side_is_4_l164_164631

-- Define the conditions
def num_books := 2
def pages_per_book := 600
def sheets_used := 150
def sides_per_sheet := 2

-- Define the total number of pages and sides
def total_pages := num_books * pages_per_book
def total_sides := sheets_used * sides_per_sheet

-- Prove the number of pages per side is 4
theorem pages_per_side_is_4 : total_pages / total_sides = 4 := by
  sorry

end pages_per_side_is_4_l164_164631


namespace cost_of_apples_l164_164377

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164377


namespace line_through_point_and_isosceles_triangle_l164_164158

def is_line_eq (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def is_isosceles_right_triangle_with_axes (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0

theorem line_through_point_and_isosceles_triangle (a b c : ℝ) (hx : ℝ) (hy : ℝ) :
  is_line_eq a b c hx hy ∧ is_isosceles_right_triangle_with_axes a b → 
  ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 1 ∧ b = -1 ∧ c = -1)) :=
by
  sorry

end line_through_point_and_isosceles_triangle_l164_164158


namespace sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l164_164045

theorem sqrt_sqrt_of_81_eq_pm3_and_cube_root_self (x : ℝ) : 
  (∃ y : ℝ, y^2 = 81 ∧ (x^2 = y → x = 3 ∨ x = -3)) ∧ (∀ z : ℝ, z^3 = z → (z = 1 ∨ z = -1 ∨ z = 0)) := by
  sorry

end sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l164_164045


namespace rewrite_neg_multiplication_as_exponent_l164_164353

theorem rewrite_neg_multiplication_as_exponent :
  -2 * 2 * 2 * 2 = - (2^4) :=
by
  sorry

end rewrite_neg_multiplication_as_exponent_l164_164353


namespace compare_neg_fractions_l164_164105

theorem compare_neg_fractions : (-5/4 : ℚ) > (-4/3 : ℚ) := 
sorry

end compare_neg_fractions_l164_164105


namespace number_of_BMWs_sold_l164_164398

theorem number_of_BMWs_sold (total_cars : ℕ) (ford_percentage nissan_percentage volkswagen_percentage : ℝ) 
    (h1 : total_cars = 300)
    (h2 : ford_percentage = 0.2)
    (h3 : nissan_percentage = 0.25)
    (h4 : volkswagen_percentage = 0.1) :
    ∃ (bmw_percentage : ℝ) (bmw_cars : ℕ), bmw_percentage = 0.45 ∧ bmw_cars = 135 :=
by 
    sorry

end number_of_BMWs_sold_l164_164398


namespace circle_equations_centered_on_line_y_equals_2x_l164_164538

noncomputable def circle_equation : set (set (ℝ × ℝ)) :=
  {circle | ∃ a R, 
    circle = {p | (p.1 - a)^2 + (p.2 - 2 * a)^2 = R^2} ∧
    (a - 2)^2 + (2 * a)^2 = R^2 ∧
    a^2 + (2 * a - 4)^2 = R^2}

theorem circle_equations_centered_on_line_y_equals_2x :
  circle_equation =
    {{p | (p.1 - 1)^2 + (p.2 - 2)^2 = 5} ∨ {p | (p.1 + 1)^2 + (p.2 + 2)^2 = 5}} :=
  sorry

end circle_equations_centered_on_line_y_equals_2x_l164_164538


namespace distance_between_closest_points_correct_l164_164412

noncomputable def circle_1_center : ℝ × ℝ := (3, 3)
noncomputable def circle_2_center : ℝ × ℝ := (20, 12)
noncomputable def circle_1_radius : ℝ := circle_1_center.2
noncomputable def circle_2_radius : ℝ := circle_2_center.2
noncomputable def distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (12 - 3)^2)
noncomputable def distance_between_closest_points : ℝ := distance_between_centers - (circle_1_radius + circle_2_radius)

theorem distance_between_closest_points_correct :
  distance_between_closest_points = Real.sqrt 370 - 15 :=
sorry

end distance_between_closest_points_correct_l164_164412


namespace number_of_blue_balloons_l164_164097

def total_balloons : ℕ := 37
def red_balloons : ℕ := 14
def green_balloons : ℕ := 10

theorem number_of_blue_balloons : (total_balloons - red_balloons - green_balloons) = 13 := 
by
  -- Placeholder for the proof
  sorry

end number_of_blue_balloons_l164_164097


namespace prove_zero_function_l164_164598

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ x y : ℝ, f (x ^ 333 + y) = f (x ^ 2018 + 2 * y) + f (x ^ 42)

theorem prove_zero_function : ∀ x : ℝ, f x = 0 :=
by
  sorry

end prove_zero_function_l164_164598


namespace lottery_probability_correct_l164_164243

/-- The binomial coefficient function -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of matching MegaBall and WinnerBalls in the lottery -/
noncomputable def lottery_probability : ℚ :=
  let megaBall_prob := (1 : ℚ) / 30
  let winnerBalls_prob := (1 : ℚ) / binom 45 6
  megaBall_prob * winnerBalls_prob

theorem lottery_probability_correct : lottery_probability = (1 : ℚ) / 244351800 := by
  sorry

end lottery_probability_correct_l164_164243


namespace wait_time_probability_l164_164397

theorem wait_time_probability
  (P_B1_8_00 : ℚ)
  (P_B1_8_20 : ℚ)
  (P_B1_8_40 : ℚ)
  (P_B2_9_00 : ℚ)
  (P_B2_9_20 : ℚ)
  (P_B2_9_40 : ℚ)
  (h_independent : true)
  (h_employee_arrival : true)
  (h_P_B1 : P_B1_8_00 = 1/4 ∧ P_B1_8_20 = 1/2 ∧ P_B1_8_40 = 1/4)
  (h_P_B2 : P_B2_9_00 = 1/4 ∧ P_B2_9_20 = 1/2 ∧ P_B2_9_40 = 1/4) :
  (P_B1_8_00 * P_B2_9_20 + P_B1_8_00 * P_B2_9_40 = 3/16) :=
sorry

end wait_time_probability_l164_164397


namespace dad_steps_l164_164149

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l164_164149


namespace gummy_vitamins_cost_l164_164011

def bottle_discounted_price (P D_s : ℝ) : ℝ :=
  P * (1 - D_s)

def normal_purchase_discounted_price (discounted_price D_n : ℝ) : ℝ :=
  discounted_price * (1 - D_n)

def bulk_purchase_discounted_price (discounted_price D_b : ℝ) : ℝ :=
  discounted_price * (1 - D_b)

def total_cost (normal_bottles bulk_bottles normal_price bulk_price : ℝ) : ℝ :=
  (normal_bottles * normal_price) + (bulk_bottles * bulk_price)

def apply_coupons (total_cost N_c C : ℝ) : ℝ :=
  total_cost - (N_c * C)

theorem gummy_vitamins_cost 
  (P N_c C D_s D_n D_b : ℝ) 
  (normal_bottles bulk_bottles : ℕ) :
  bottle_discounted_price P D_s = 12.45 → 
  normal_purchase_discounted_price 12.45 D_n = 11.33 → 
  bulk_purchase_discounted_price 12.45 D_b = 11.83 → 
  total_cost 4 3 11.33 11.83 = 80.81 → 
  apply_coupons 80.81 N_c C = 70.81 :=
sorry

end gummy_vitamins_cost_l164_164011


namespace students_in_high_school_l164_164856

-- Definitions from conditions
def H (L: ℝ) : ℝ := 4 * L
def middleSchoolStudents : ℝ := 300
def combinedStudents (H: ℝ) (L: ℝ) : ℝ := H + L
def combinedIsSevenTimesMiddle (H: ℝ) (L: ℝ) : Prop := combinedStudents H L = 7 * middleSchoolStudents

-- The main goal to prove
theorem students_in_high_school (L H: ℝ) (h1: H = 4 * L) (h2: combinedIsSevenTimesMiddle H L) : H = 1680 := by
  sorry

end students_in_high_school_l164_164856


namespace tan_7pi_over_6_l164_164535

noncomputable def tan_periodic (θ : ℝ) : Prop :=
  ∀ k : ℤ, Real.tan (θ + k * Real.pi) = Real.tan θ

theorem tan_7pi_over_6 : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 :=
by
  sorry

end tan_7pi_over_6_l164_164535


namespace number_of_5_dollar_bills_l164_164705

theorem number_of_5_dollar_bills (x y : ℝ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
sorry

end number_of_5_dollar_bills_l164_164705


namespace soaked_part_solution_l164_164529

theorem soaked_part_solution 
  (a b : ℝ) (c : ℝ) 
  (h : c * (2/3) * a * b = 2 * a^2 * b^3 + (1/3) * a^3 * b^2) :
  c = 3 * a * b^2 + (1/2) * a^2 * b :=
by
  sorry

end soaked_part_solution_l164_164529


namespace problem1_problem2_l164_164607

variable (x a : ℝ)

def P := x^2 - 5*a*x + 4*a^2 < 0
def Q := (x^2 - 2*x - 8 <= 0) ∧ (x^2 + 3*x - 10 > 0)

theorem problem1 (h : 1 = a) (hP : P x a) (hQ : Q x) : 2 < x ∧ x ≤ 4 :=
sorry

theorem problem2 (h1 : ∀ x, ¬P x a → ¬Q x) (h2 : ∃ x, P x a ∧ ¬Q x) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l164_164607


namespace cost_of_apples_l164_164370

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164370


namespace average_calls_per_day_l164_164010

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end average_calls_per_day_l164_164010


namespace meaningful_expression_range_l164_164770

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l164_164770


namespace coin_flip_sequences_count_l164_164079

theorem coin_flip_sequences_count : 
  let total_flips := 10;
  let heads_fixed := 2;
  (2 : ℕ) ^ (total_flips - heads_fixed) = 256 := 
by 
  sorry

end coin_flip_sequences_count_l164_164079


namespace residue_of_11_pow_2048_mod_19_l164_164051

theorem residue_of_11_pow_2048_mod_19 :
  (11 ^ 2048) % 19 = 16 := 
by
  sorry

end residue_of_11_pow_2048_mod_19_l164_164051


namespace sum_of_digits_of_fraction_is_nine_l164_164727

theorem sum_of_digits_of_fraction_is_nine : 
  ∃ (x y : Nat), (4 / 11 : ℚ) = x / 10 + y / 100 + x / 1000 + y / 10000 + (x + y) / 100000 -- and other terms
  ∧ x + y = 9 := 
sorry

end sum_of_digits_of_fraction_is_nine_l164_164727


namespace mode_is_37_median_is_36_l164_164002

namespace ProofProblem

def data_set : List ℕ := [34, 35, 36, 34, 36, 37, 37, 36, 37, 37]

def mode (l : List ℕ) : ℕ := sorry -- Implementing a mode function

def median (l : List ℕ) : ℕ := sorry -- Implementing a median function

theorem mode_is_37 : mode data_set = 37 := 
  by 
    sorry -- Proof of mode

theorem median_is_36 : median data_set = 36 := 
  by
    sorry -- Proof of median

end ProofProblem

end mode_is_37_median_is_36_l164_164002


namespace problem1_problem2_l164_164062

-- Problem 1: Prove that the given expression evaluates to the correct answer
theorem problem1 :
  2 * Real.sin (Real.pi / 6) - (2015 - Real.pi)^0 + abs (1 - Real.tan (Real.pi / 3)) = abs (1 - Real.sqrt 3) :=
sorry

-- Problem 2: Prove that the solutions to the given equation are correct
theorem problem2 (x : ℝ) :
  (x-2)^2 = 3 * (x-2) → x = 2 ∨ x = 5 :=
sorry

end problem1_problem2_l164_164062


namespace negation_of_universal_proposition_l164_164364

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 3 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2 * x + 3 < 0 := 
sorry

end negation_of_universal_proposition_l164_164364


namespace circle_area_l164_164568

-- Define the conditions of the problem
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- State the proof problem
theorem circle_area : (∀ (x y : ℝ), circle_equation x y) → (∀ r : ℝ, r = 2 → π * r^2 = 4 * π) :=
by
  sorry

end circle_area_l164_164568


namespace max_value_condition_min_value_condition_l164_164846

theorem max_value_condition (x : ℝ) (h : x < 0) : (x^2 + x + 1) / x ≤ -1 :=
sorry

theorem min_value_condition (x : ℝ) (h : x > -1) : ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end max_value_condition_min_value_condition_l164_164846


namespace fn_prime_factor_bound_l164_164215

theorem fn_prime_factor_bound (n : ℕ) (h : n ≥ 3) : 
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^(2^n) + 1)) ∧ p > 2^(n+2) * (n+1) :=
sorry

end fn_prime_factor_bound_l164_164215


namespace tangent_parallel_to_line_at_point_l164_164979

theorem tangent_parallel_to_line_at_point (P0 : ℝ × ℝ) 
  (curve : ℝ → ℝ) (line_slope : ℝ) : 
  curve = (fun x => x^3 + x - 2) ∧ line_slope = 4 ∧
  (∃ x0, P0 = (x0, curve x0) ∧ 3*x0^2 + 1 = line_slope) → 
  P0 = (1, 0) :=
by 
  sorry

end tangent_parallel_to_line_at_point_l164_164979


namespace triangle_count_from_10_points_l164_164036

/--
Given 10 distinct points on the circumference of a circle,
the number of different triangles such that no two triangles share the same side.
-/
theorem triangle_count_from_10_points : (Nat.choose 10 3) = 120 := 
sorry

end triangle_count_from_10_points_l164_164036


namespace parabola_standard_equations_l164_164457

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l164_164457


namespace proportion_third_number_l164_164619

theorem proportion_third_number
  (x : ℝ) (y : ℝ)
  (h1 : 0.60 * 4 = x * y)
  (h2 : x = 0.39999999999999997) :
  y = 6 :=
by
  sorry

end proportion_third_number_l164_164619


namespace infinite_series_sum_l164_164415

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l164_164415


namespace find_f_neg_two_l164_164811

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : 3 * f (1 / x) + (2 * f x) / x = x ^ 2

theorem find_f_neg_two : f (-2) = 67 / 20 :=
by
  sorry

end find_f_neg_two_l164_164811


namespace distance_from_ground_at_speed_25_is_137_5_l164_164403
noncomputable section

-- Define the initial conditions and givens
def buildingHeight : ℝ := 200
def speedProportionalityConstant : ℝ := 10
def distanceProportionalityConstant : ℝ := 10

-- Define the speed function and distance function
def speed (t : ℝ) : ℝ := speedProportionalityConstant * t
def distance (t : ℝ) : ℝ := distanceProportionalityConstant * (t * t)

-- Define the specific time when speed is 25 m/sec
def timeWhenSpeedIs25 : ℝ := 25 / speedProportionalityConstant

-- Define the distance traveled at this specific time
def distanceTraveledAtTime : ℝ := distance timeWhenSpeedIs25

-- Calculate the distance from the ground
def distanceFromGroundAtSpeed25 : ℝ := buildingHeight - distanceTraveledAtTime

-- State the theorem
theorem distance_from_ground_at_speed_25_is_137_5 :
  distanceFromGroundAtSpeed25 = 137.5 :=
sorry

end distance_from_ground_at_speed_25_is_137_5_l164_164403


namespace polygon_sides_l164_164245

theorem polygon_sides (n : Nat) (h : (360 : ℝ) / (180 * (n - 2)) = 2 / 9) : n = 11 :=
by
  sorry

end polygon_sides_l164_164245


namespace melissa_coupe_sale_l164_164347

theorem melissa_coupe_sale :
  ∃ x : ℝ, (0.02 * x + 0.02 * 2 * x = 1800) ∧ x = 30000 :=
by
  sorry

end melissa_coupe_sale_l164_164347


namespace number_of_polynomials_l164_164666

-- Define conditions
def is_positive_integer (n : ℤ) : Prop :=
  5 * 151 * n > 0

-- Define the main theorem
theorem number_of_polynomials (n : ℤ) (h : is_positive_integer n) : 
  ∃ k : ℤ, k = ⌊n / 2⌋ + 1 :=
by
  sorry

end number_of_polynomials_l164_164666


namespace g_is_odd_l164_164489

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) - (1 / 2)

theorem g_is_odd (x : ℝ) : g (-x) = -g x :=
by sorry

end g_is_odd_l164_164489


namespace solutions_shifted_quadratic_l164_164311

theorem solutions_shifted_quadratic (a h k : ℝ) (x1 x2: ℝ)
  (h1 : a * (-1 - h)^2 + k = 0)
  (h2 : a * (3 - h)^2 + k = 0) :
  a * (0 - (h + 1))^2 + k = 0 ∧ a * (4 - (h + 1))^2 + k = 0 :=
by
  sorry

end solutions_shifted_quadratic_l164_164311


namespace money_sister_gave_l164_164020

theorem money_sister_gave (months_saved : ℕ) (savings_per_month : ℕ) (total_paid : ℕ) 
  (h1 : months_saved = 3) 
  (h2 : savings_per_month = 70) 
  (h3 : total_paid = 260) : 
  (total_paid - (months_saved * savings_per_month) = 50) :=
by {
  sorry
}

end money_sister_gave_l164_164020


namespace find_valid_pairs_l164_164886

def satisfies_condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a ^ 2017 + b) % (a * b) = 0

theorem find_valid_pairs : 
  ∀ (a b : ℕ), satisfies_condition a b → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := 
by
  sorry

end find_valid_pairs_l164_164886


namespace expected_value_squared_minimum_vector_norm_l164_164071

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l164_164071


namespace find_third_number_l164_164985

theorem find_third_number (x y : ℕ) (h1 : x = 3)
  (h2 : (x + 1) / (x + 5) = (x + 5) / (x + y)) : y = 13 :=
by
  sorry

end find_third_number_l164_164985


namespace smallest_angle_in_triangle_l164_164332

theorem smallest_angle_in_triangle (x : ℝ) 
  (h_ratio : 4 * x < 5 * x ∧ 5 * x < 9 * x) 
  (h_sum : 4 * x + 5 * x + 9 * x = 180) : 
  4 * x = 40 :=
by
  sorry

end smallest_angle_in_triangle_l164_164332


namespace rhombus_diagonal_l164_164380

theorem rhombus_diagonal (d1 d2 : ℝ) (area_tri : ℝ) (h1 : d1 = 15) (h2 : area_tri = 75) :
  (d1 * d2) / 2 = 2 * area_tri → d2 = 20 :=
by
  sorry

end rhombus_diagonal_l164_164380


namespace slices_with_both_toppings_l164_164853

theorem slices_with_both_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (all_with_topping : total_slices = 15 ∧ pepperoni_slices = 8 ∧ mushroom_slices = 12 ∧ ∀ i, i < 15 → (i < 8 ∨ i < 12)) :
  ∃ n, (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices ∧ n = 5 :=
by
  sorry

end slices_with_both_toppings_l164_164853


namespace family_vacation_rain_days_l164_164839

theorem family_vacation_rain_days (r_m r_a : ℕ) 
(h_rain_days : r_m + r_a = 13)
(clear_mornings : r_a = 11)
(clear_afternoons : r_m = 12) : 
r_m + r_a = 23 := 
by 
  sorry

end family_vacation_rain_days_l164_164839


namespace point_outside_circle_l164_164310

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) : a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l164_164310


namespace normal_dist_symmetry_l164_164477

noncomputable theory

open ProbabilityTheory MeasureTheory

variables {X : Type} [MeasureSpace X]

def X_dist : ProbabilityMassFunction X := 
  classical.arbitrary (ProbabilityMassFunction X)

axiom hX : X_dist.hasLeftIntegral (set.univ X)
  (ProbabilityDensityFunction.normal X_dist 1 2)

/-- Given a random variable X that follows a normal distribution N(1,4),
and probabilities P(0 < X < 3) = m and P(-1 < X < 2) = n,
we prove that m = n. -/
theorem normal_dist_symmetry (X : ℝ) (hX : NormalDist X 1 2)
    (hm : P (0 < X ∧ X < 3) = m) (hn : P (-1 < X ∧ X < 2) = n) : m = n := 
sorry

end normal_dist_symmetry_l164_164477


namespace tangent_line_at_zero_max_min_values_in_interval_l164_164751

noncomputable
def f (x : ℝ) := (Real.sin x) / (Real.exp x) - x

theorem tangent_line_at_zero :
  let f'(x : ℝ) := (Real.cos x - Real.sin x) / (Real.exp x) - 1 in
  f'(0) = 0 ∧ f 0 = 0 → ∀ (x : ℝ), (0, f 0) = (0, 0) → (f x = 0) :=
by
  intros
  sorry

theorem max_min_values_in_interval :
  let f'(x : ℝ) := (Real.cos x - Real.sin x) / (Real.exp x) - 1 in
  ∀ x ∈ Icc 0 π, 
  (∀ x ∈ (Ioo 0 π), f'(x) < 0) →
  f 0 = 0 ∧ f π = -π →
  (∀ x, f x ≤ f 0 ∧ f x ≥ f π) :=
by
  intros
  sorry

end tangent_line_at_zero_max_min_values_in_interval_l164_164751


namespace value_of_a_plus_b_2023_l164_164467

theorem value_of_a_plus_b_2023 
    (x y a b : ℤ)
    (h1 : 4*x + 3*y = 11)
    (h2 : 2*x - y = 3)
    (h3 : a*x + b*y = -2)
    (h4 : b*x - a*y = 6)
    (hx : x = 2)
    (hy : y = 1) :
    (a + b) ^ 2023 = 0 := 
sorry

end value_of_a_plus_b_2023_l164_164467


namespace total_sticks_used_l164_164393

-- Definitions based on the conditions
def hexagons : Nat := 800
def sticks_for_first_hexagon : Nat := 6
def sticks_per_additional_hexagon : Nat := 5

-- The theorem to prove
theorem total_sticks_used :
  sticks_for_first_hexagon + (hexagons - 1) * sticks_per_additional_hexagon = 4001 := by
  sorry

end total_sticks_used_l164_164393


namespace range_of_a_l164_164943

theorem range_of_a (x a : ℝ) (p : 0 < x ∧ x < 1)
  (q : (x - a) * (x - (a + 2)) ≤ 0) (h : ∀ x, (0 < x ∧ x < 1) → (x - a) * (x - (a + 2)) ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l164_164943


namespace system1_solution_system2_solution_l164_164031

theorem system1_solution : 
  ∃ (x y : ℤ), 2 * x + 3 * y = -1 ∧ y = 4 * x - 5 ∧ x = 1 ∧ y = -1 := by 
    sorry

theorem system2_solution : 
  ∃ (x y : ℤ), 3 * x + 2 * y = 20 ∧ 4 * x - 5 * y = 19 ∧ x = 6 ∧ y = 1 := by 
    sorry

end system1_solution_system2_solution_l164_164031


namespace num_distinct_integers_written_as_sums_of_special_fractions_l164_164723

theorem num_distinct_integers_written_as_sums_of_special_fractions :
  let special_fraction (a b : ℕ) := a + b = 15
  ∃ n : ℕ, n = 11 ∧ 
    ∀ i j : ℕ, special_fraction i j → 
      ∃ k : ℕ, 
        is_sum_of_special_fractions i j k → k < 29 := sorry

def is_sum_of_special_fractions (i j k : ℕ) : Prop := -- Custom definition to define sum of special fractions.
  -- details to be filled in as necessary
  sorry

end num_distinct_integers_written_as_sums_of_special_fractions_l164_164723


namespace problem1_problem2_problem3_problem4_l164_164872

-- Problem (1)
theorem problem1 : (-8 - 6 + 24) = 10 :=
by sorry

-- Problem (2)
theorem problem2 : (-48 / 6 + -21 * (-1 / 3)) = -1 :=
by sorry

-- Problem (3)
theorem problem3 : ((1 / 8 - 1 / 3 + 1 / 4) * -24) = -1 :=
by sorry

-- Problem (4)
theorem problem4 : (-1^4 - (1 + 0.5) * (1 / 3) * (1 - (-2)^2)) = 0.5 :=
by sorry

end problem1_problem2_problem3_problem4_l164_164872


namespace largest_number_divisible_by_48_is_9984_l164_164385

def largest_divisible_by_48 (n : ℕ) := ∀ m ≥ n, m % 48 = 0 → m ≤ 9999

theorem largest_number_divisible_by_48_is_9984 :
  largest_divisible_by_48 9984 ∧ 9999 / 10^3 = 9 ∧ 48 ∣ 9984 ∧ 9984 < 10000 :=
by
  sorry

end largest_number_divisible_by_48_is_9984_l164_164385


namespace shop_owner_percentage_profit_l164_164088

theorem shop_owner_percentage_profit
  (cp : ℝ)  -- cost price of 1 kg
  (cheat_buy : ℝ) -- cheat percentage when buying
  (cheat_sell : ℝ) -- cheat percentage when selling
  (h_cp : cp = 100) -- cost price is $100
  (h_cheat_buy : cheat_buy = 15) -- cheat by 15% when buying
  (h_cheat_sell : cheat_sell = 20) -- cheat by 20% when selling
  :
  let weight_bought := 1 + (cheat_buy / 100)
  let weight_sold := 1 - (cheat_sell / 100)
  let real_selling_price_per_kg := cp / weight_sold
  let total_selling_price := weight_bought * real_selling_price_per_kg
  let profit := total_selling_price - cp
  let percentage_profit := (profit / cp) * 100
  percentage_profit = 43.75 := 
by
  sorry

end shop_owner_percentage_profit_l164_164088


namespace circle_center_and_radius_l164_164739

-- Define the given conditions
variable (a : ℝ) (h : a^2 = a + 2 ∧ a ≠ 0)

-- Define the equation
noncomputable def circle_equation (x y : ℝ) : ℝ := a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

-- Lean definition to represent the problem
theorem circle_center_and_radius :
  (∃a : ℝ, a ≠ 0 ∧ a^2 = a + 2 ∧
    (∃x y : ℝ, circle_equation a x y = 0) ∧
    ((a = -1) → ((∃x y : ℝ, (x + 2)^2 + (y + 4)^2 = 25) ∧
                 (center_x = -2) ∧ (center_y = -4) ∧ (radius = 5)))) :=
by
  sorry

end circle_center_and_radius_l164_164739


namespace Okeydokey_should_receive_25_earthworms_l164_164059

def applesOkeydokey : ℕ := 5
def applesArtichokey : ℕ := 7
def totalEarthworms : ℕ := 60
def totalApples : ℕ := applesOkeydokey + applesArtichokey
def okeydokeyProportion : ℚ := applesOkeydokey / totalApples
def okeydokeyEarthworms : ℚ := okeydokeyProportion * totalEarthworms

theorem Okeydokey_should_receive_25_earthworms : okeydokeyEarthworms = 25 := by
  sorry

end Okeydokey_should_receive_25_earthworms_l164_164059


namespace smallest_two_digit_number_l164_164837

theorem smallest_two_digit_number :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧
            n % 12 = 0 ∧
            n % 5 = 4 ∧
            ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ m % 12 = 0 ∧ m % 5 = 4 → n ≤ m :=
  by {
  -- proof shows the mathematical statement is true
  sorry
}

end smallest_two_digit_number_l164_164837


namespace question1_question2_l164_164061

-- Define required symbols and parameters
variables {x : ℝ} {b c : ℝ}

-- Statement 1: Proving b + c given the conditions on the inequality
theorem question1 (h : ∀ x, -1 < x ∧ x < 3 → 5*x^2 - b*x + c < 0) : b + c = -25 := sorry

-- Statement 2: Proving the solution set for the given inequality
theorem question2 (h : ∀ x, (2 * x - 5) / (x + 4) ≥ 0 → (x ≥ 5 / 2 ∨ x < -4)) : 
  {x | (2 * x - 5) / (x + 4) ≥ 0} = {x | x ≥ 5/2 ∨ x < -4} := sorry

end question1_question2_l164_164061


namespace vacation_cost_correct_l164_164633

namespace VacationCost

-- Define constants based on conditions
def starting_charge_per_dog : ℝ := 2
def charge_per_block : ℝ := 1.25
def number_of_dogs : ℕ := 20
def total_blocks : ℕ := 128
def family_members : ℕ := 5

-- Define total earnings from walking dogs
def total_earnings : ℝ :=
  (number_of_dogs * starting_charge_per_dog) + (total_blocks * charge_per_block)

-- Define the total cost of the vacation
noncomputable def total_cost_of_vacation : ℝ :=
  total_earnings / family_members * family_members

-- Proof statement: The total cost of the vacation is $200
theorem vacation_cost_correct : total_cost_of_vacation = 200 := by
  sorry

end VacationCost

end vacation_cost_correct_l164_164633


namespace stripe_width_l164_164238

theorem stripe_width (x : ℝ) (h : 60 * x - x^2 = 400) : x = 30 - 5 * Real.sqrt 5 := 
  sorry

end stripe_width_l164_164238


namespace ratio_karen_beatrice_l164_164205

noncomputable def karen_crayons : ℕ := 128
noncomputable def judah_crayons : ℕ := 8
noncomputable def gilbert_crayons : ℕ := 4 * judah_crayons
noncomputable def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem ratio_karen_beatrice :
  karen_crayons / beatrice_crayons = 2 := by
sorry

end ratio_karen_beatrice_l164_164205


namespace nearest_int_to_expr_l164_164828

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l164_164828


namespace number_of_pairs_101_l164_164000

theorem number_of_pairs_101 :
  (∃ n : ℕ, (∀ a b : ℕ, (a > 0) → (b > 0) → (a + b = 101) → (b > a) → (n = 50))) :=
sorry

end number_of_pairs_101_l164_164000


namespace download_time_l164_164361

def first_segment_size : ℝ := 30
def first_segment_rate : ℝ := 5
def second_segment_size : ℝ := 40
def second_segment_rate1 : ℝ := 10
def second_segment_rate2 : ℝ := 2
def third_segment_size : ℝ := 20
def third_segment_rate1 : ℝ := 8
def third_segment_rate2 : ℝ := 4

theorem download_time :
  let time_first := first_segment_size / first_segment_rate
  let time_second := (10 / second_segment_rate1) + (10 / second_segment_rate2) + (10 / second_segment_rate1) + (10 / second_segment_rate2)
  let time_third := (10 / third_segment_rate1) + (10 / third_segment_rate2)
  time_first + time_second + time_third = 21.75 :=
by
  sorry

end download_time_l164_164361


namespace kevin_leaves_l164_164634

theorem kevin_leaves (n : ℕ) (h : n > 1) : ∃ k : ℕ, n = k^3 ∧ n^2 = k^6 ∧ n = 8 := by
  sorry

end kevin_leaves_l164_164634


namespace set_intersection_l164_164744

noncomputable def SetA : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
noncomputable def SetB : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem set_intersection :
  SetA ∩ SetB = {x | 2 < x ∧ x < 3} := by
  sorry

end set_intersection_l164_164744


namespace total_goals_in_5_matches_l164_164530

theorem total_goals_in_5_matches 
  (x : ℝ) 
  (h1 : 4 * x + 3 = 5 * (x + 0.2)) 
  : 4 * x + 3 = 11 :=
by
  -- The proof is omitted here
  sorry

end total_goals_in_5_matches_l164_164530


namespace inequality_proof_l164_164025

noncomputable def inequality_holds (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop :=
  (a ^ 2) / (b - 1) + (b ^ 2) / (a - 1) ≥ 8

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  inequality_holds a b ha hb :=
sorry

end inequality_proof_l164_164025


namespace power_mod_l164_164054

theorem power_mod (h : 5 ^ 200 ≡ 1 [MOD 1000]) : 5 ^ 6000 ≡ 1 [MOD 1000] :=
by
  sorry

end power_mod_l164_164054


namespace solution_xy_l164_164157

noncomputable def find_xy (x y : ℚ) : Prop :=
  (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1 / 3

theorem solution_xy :
  find_xy (10 + 1 / 3) (10 + 2 / 3) :=
by
  sorry

end solution_xy_l164_164157


namespace standard_equation_of_parabola_l164_164460

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l164_164460


namespace find_function_l164_164156

theorem find_function (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y - 2023) →
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 :=
by
  intros h
  sorry

end find_function_l164_164156


namespace dad_steps_l164_164131

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l164_164131


namespace meaningful_expression_range_l164_164767

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l164_164767


namespace unique_zero_of_function_l164_164309

theorem unique_zero_of_function (a : ℝ) :
  (∃! x : ℝ, e^(abs x) + 2 * a - 1 = 0) ↔ a = 0 := 
by 
  sorry

end unique_zero_of_function_l164_164309


namespace amrita_bakes_cake_next_thursday_l164_164560

theorem amrita_bakes_cake_next_thursday (n m : ℕ) (h1 : n = 5) (h2 : m = 7) : Nat.lcm n m = 35 :=
by
  -- Proof goes here
  sorry

end amrita_bakes_cake_next_thursday_l164_164560


namespace difference_between_max_and_min_area_l164_164223

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

noncomputable def min_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

theorem difference_between_max_and_min_area :
  ∃ (l_max l_min w_max w_min : ℕ),
    2 * l_max + 2 * w_max = 60 ∧
    2 * l_min + 2 * w_min = 60 ∧
    (l_max * w_max - l_min * w_min = 196) :=
by
  sorry

end difference_between_max_and_min_area_l164_164223


namespace find_x_value_l164_164506

noncomputable def x_value := 92

open BigOperators

section

variables (data : list ℝ) (x : ℝ)

theorem find_x_value :
  (mean data = x) ∧ (median data = x) ∧ (mode data = x) ∧ (frequency x data ≥ 2) ↔ x = 92 :=
begin
  sorry
end

end

end find_x_value_l164_164506


namespace nearest_integer_to_expression_correct_l164_164825

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l164_164825


namespace circle1_correct_circle2_correct_l164_164994

noncomputable def circle1_eq (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*x - 6*y - 12

noncomputable def circle2_eq (x y : ℝ) : ℝ :=
  36*x^2 + 36*y^2 - 24*x + 72*y + 31

theorem circle1_correct (x y : ℝ) :
  ((x + 2)^2 + (y - 3)^2 = 25) ↔ (circle1_eq x y = 0) :=
sorry

theorem circle2_correct (x y : ℝ) :
  (36 * ((x - 1/3)^2 + (y + 1)^2) = 9) ↔ (circle2_eq x y = 0) :=
sorry

end circle1_correct_circle2_correct_l164_164994


namespace binomial_sixteen_twelve_eq_l164_164420

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l164_164420


namespace dad_steps_are_90_l164_164144

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l164_164144


namespace distance_between_lines_l164_164742

-- Define lines l1 and l2
def line_l1 (x y : ℝ) := x + y + 1 = 0
def line_l2 (x y : ℝ) := 2 * x + 2 * y + 3 = 0

-- Proof statement for the distance between parallel lines
theorem distance_between_lines :
  let a := 1
  let b := 1
  let c1 := 1
  let c2 := 3 / 2
  let distance := |c2 - c1| / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 2 / 4 :=
by
  sorry

end distance_between_lines_l164_164742


namespace number_of_shelves_l164_164081

theorem number_of_shelves (a d S : ℕ) (h1 : a = 3) (h2 : d = 3) (h3 : S = 225) : 
  ∃ n : ℕ, (S = n * (2 * a + (n - 1) * d) / 2) ∧ (n = 15) := 
by {
  sorry
}

end number_of_shelves_l164_164081


namespace loss_percentage_is_correct_l164_164547

noncomputable def watch_loss_percentage (SP_loss SP_profit : ℕ) (profit_percentage : ℕ) : ℕ :=
  let CP := SP_profit / (1 + profit_percentage / 100) in
  let loss := CP - SP_loss in
  loss * 100 / CP

theorem loss_percentage_is_correct :
  watch_loss_percentage 1140 1260 5 = 5 :=
by
  sorry

end loss_percentage_is_correct_l164_164547


namespace driving_distance_l164_164563

theorem driving_distance:
  ∀ a b: ℕ, (a + b = 500 ∧ a ≥ 150 ∧ b ≥ 150) → 
  (⌊Real.sqrt (a^2 + b^2)⌋ = 380) :=
by
  intro a b
  intro h
  sorry

end driving_distance_l164_164563


namespace square_of_1037_l164_164290

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := 
by {
  -- Proof omitted
  sorry
}

end square_of_1037_l164_164290


namespace intersection_eq_l164_164641

open Set

def setA : Set ℤ := {x | x ≥ -4}
def setB : Set ℤ := {x | x ≤ 3}

theorem intersection_eq : (setA ∩ setB) = {x | -4 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l164_164641


namespace expected_sum_of_rolls_l164_164542

def fair_die : ℙ (Sum int → int) := 
sorry

noncomputable def expected_rolls_to_2010 (die: ℙ (Sum int → int)) : ℝ :=
sorry

theorem expected_sum_of_rolls :
  expected_rolls_to_2010(fair_die) = 574.5238095 :=
sorry

end expected_sum_of_rolls_l164_164542


namespace no_six_coins_sum_70_cents_l164_164967

theorem no_six_coins_sum_70_cents :
  ¬ ∃ (p n d q : ℕ), p + n + d + q = 6 ∧ p + 5 * n + 10 * d + 25 * q = 70 :=
by
  sorry

end no_six_coins_sum_70_cents_l164_164967


namespace dad_steps_l164_164138

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l164_164138


namespace smoothie_combinations_l164_164715

theorem smoothie_combinations :
  let flavors := 5
  let supplements := 8
  (flavors * Nat.choose supplements 3) = 280 :=
by
  sorry

end smoothie_combinations_l164_164715


namespace find_m_l164_164319

theorem find_m (m : ℤ) (a := (3, m)) (b := (1, -2)) (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) : m = -1 :=
sorry

end find_m_l164_164319


namespace ms_walker_drives_24_miles_each_way_l164_164645

theorem ms_walker_drives_24_miles_each_way
  (D : ℝ)
  (H1 : 1 / 60 * D + 1 / 40 * D = 1) :
  D = 24 := 
sorry

end ms_walker_drives_24_miles_each_way_l164_164645


namespace unique_intersection_l164_164001

theorem unique_intersection (a : ℝ) (h : 2 * a = -1) :
  ∃! x, 2 * a = abs (x - a) - 1 :=
by
  sorry

end unique_intersection_l164_164001


namespace lowest_possible_sale_price_percentage_l164_164716

noncomputable def list_price : ℝ := 80
noncomputable def max_initial_discount_percent : ℝ := 0.5
noncomputable def summer_sale_discount_percent : ℝ := 0.2
noncomputable def membership_discount_percent : ℝ := 0.1
noncomputable def coupon_discount_percent : ℝ := 0.05

theorem lowest_possible_sale_price_percentage :
  let max_initial_discount := max_initial_discount_percent * list_price
  let summer_sale_discount := summer_sale_discount_percent * list_price
  let membership_discount := membership_discount_percent * list_price
  let coupon_discount := coupon_discount_percent * list_price
  let lowest_sale_price := list_price * (1 - max_initial_discount_percent) - summer_sale_discount - membership_discount - coupon_discount
  (lowest_sale_price / list_price) * 100 = 15 :=
by
  sorry

end lowest_possible_sale_price_percentage_l164_164716


namespace assumption_for_contradiction_l164_164350

theorem assumption_for_contradiction (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (h : 5 ∣ a * b) : 
  ¬ (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) := 
sorry

end assumption_for_contradiction_l164_164350


namespace max_value_expression_l164_164743

variable (a b : ℝ)

theorem max_value_expression (h : a^2 + b^2 = 3 + a * b) : 
  ∃ a b : ℝ, (2 * a - 3 * b)^2 + (a + 2 * b) * (a - 2 * b) = 22 :=
by
  -- This is a placeholder for the actual proof
  sorry

end max_value_expression_l164_164743


namespace circles_intersect_l164_164294

noncomputable def positional_relationship (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : String :=
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  if radius1 + radius2 > d ∧ d > abs (radius1 - radius2) then "Intersecting"
  else if radius1 + radius2 = d then "Externally tangent"
  else if abs (radius1 - radius2) = d then "Internally tangent"
  else "Separate"

theorem circles_intersect :
  positional_relationship (0, 1) (1, 2) 1 2 = "Intersecting" :=
by
  sorry

end circles_intersect_l164_164294


namespace Maddie_spent_on_tshirts_l164_164949

theorem Maddie_spent_on_tshirts
  (packs_white : ℕ)
  (count_white_per_pack : ℕ)
  (packs_blue : ℕ)
  (count_blue_per_pack : ℕ)
  (cost_per_tshirt : ℕ)
  (h_white : packs_white = 2)
  (h_count_w : count_white_per_pack = 5)
  (h_blue : packs_blue = 4)
  (h_count_b : count_blue_per_pack = 3)
  (h_cost : cost_per_tshirt = 3) :
  (packs_white * count_white_per_pack + packs_blue * count_blue_per_pack) * cost_per_tshirt = 66 := by
  sorry

end Maddie_spent_on_tshirts_l164_164949


namespace female_students_group_together_l164_164983

theorem female_students_group_together (male female : ℕ) (grouped_females : Bool) 
  (h1 : male = 5) (h2 : female = 3) (h3 : grouped_females = true) : 
  ∃ total_arrangements : ℕ, total_arrangements = 720 :=
by
  have h4 : 5! = 120 := by
    norm_num
  have h5 : 3! = 6 := by
    norm_num
  use 5! * 3!
  rw [h4, h5]
  norm_num
  exact rfl

end female_students_group_together_l164_164983


namespace binomial_sixteen_twelve_eq_l164_164418

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l164_164418


namespace largest_integer_y_l164_164824

theorem largest_integer_y (y : ℤ) : (y / 4 + 3 / 7 : ℝ) < 9 / 4 → y ≤ 7 := by
  intros h
  sorry -- Proof needed

end largest_integer_y_l164_164824


namespace geometric_sequence_ratio_l164_164327

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = 2 * a n - 2

theorem geometric_sequence_ratio (h : ∀ n, condition_1 a S n) : (a 8 / a 6 = 4) :=
sorry

end geometric_sequence_ratio_l164_164327


namespace direct_variation_exponent_l164_164919

variable {X Y Z : Type}

theorem direct_variation_exponent (k j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^3) : 
  ∃ m : ℝ, x = m * z^12 :=
by
  sorry

end direct_variation_exponent_l164_164919


namespace side_increase_percentage_l164_164246

theorem side_increase_percentage (s : ℝ) (p : ℝ) 
  (h : (s^2) * (1.5625) = (s * (1 + p / 100))^2) : p = 25 := 
sorry

end side_increase_percentage_l164_164246


namespace alien_run_time_l164_164677

variable (v_r v_f : ℝ) -- velocities in km/h
variable (T_r T_f : ℝ) -- time in hours
variable (D_r D_f : ℝ) -- distances in kilometers

theorem alien_run_time :
  v_r = 15 ∧ v_f = 10 ∧ (T_f = T_r + 0.5) ∧ (D_r = D_f) ∧ (D_r = v_r * T_r) ∧ (D_f = v_f * T_f) → T_f = 1.5 :=
by
  intros h
  rcases h with ⟨_, _, _, _, _, _⟩
  -- proof goes here
  sorry

end alien_run_time_l164_164677


namespace monthly_interest_payment_l164_164718

theorem monthly_interest_payment (P : ℝ) (R : ℝ) (monthly_payment : ℝ)
  (hP : P = 28800) (hR : R = 0.09) : 
  monthly_payment = (P * R) / 12 :=
by
  sorry

end monthly_interest_payment_l164_164718


namespace cost_apples_l164_164373

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l164_164373


namespace first_term_of_sequence_l164_164741

theorem first_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end first_term_of_sequence_l164_164741


namespace incorrect_statement_d_l164_164526

theorem incorrect_statement_d :
  (¬(abs 2 = -2)) :=
by sorry

end incorrect_statement_d_l164_164526


namespace appetizer_cost_per_person_l164_164603

def chip_cost : ℝ := 3 * 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def total_cost : ℝ := chip_cost + creme_fraiche_cost + caviar_cost
def number_people : ℝ := 3
def cost_per_person : ℝ := total_cost / number_people

theorem appetizer_cost_per_person : cost_per_person = 27.00 := 
by
  -- proof would go here
  sorry

end appetizer_cost_per_person_l164_164603


namespace D_neither_sufficient_nor_necessary_for_A_l164_164231

theorem D_neither_sufficient_nor_necessary_for_A 
  (A B C D : Prop) 
  (h1 : A → B) 
  (h2 : ¬(B → A)) 
  (h3 : B ↔ C) 
  (h4 : C → D) 
  (h5 : ¬(D → C)) 
  :
  ¬(D → A) ∧ ¬(A → D) :=
by 
  sorry

end D_neither_sufficient_nor_necessary_for_A_l164_164231


namespace train_length_l164_164391

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length_meters : ℕ) : speed_kmph = 90 → time_sec = 4 → length_meters = 100 :=
by
  intros h₁ h₂
  have speed_mps : ℕ := speed_kmph * 1000 / 3600
  have speed_mps_val : speed_mps = 25 := sorry
  have distance : ℕ := speed_mps * time_sec
  have distance_val : distance = 100 := sorry
  exact sorry

end train_length_l164_164391


namespace water_purification_problem_l164_164844

variable (x : ℝ) (h : x > 0)

theorem water_purification_problem
  (h1 : ∀ (p : ℝ), p = 2400)
  (h2 : ∀ (eff : ℝ), eff = 1.2)
  (h3 : ∀ (time_saved : ℝ), time_saved = 40) :
  (2400 * 1.2 / x) - (2400 / x) = 40 := by
  sorry

end water_purification_problem_l164_164844


namespace fraction_of_number_is_three_quarters_l164_164395

theorem fraction_of_number_is_three_quarters 
  (f : ℚ) 
  (h1 : 76 ≠ 0) 
  (h2 : f * 76 = 76 - 19) : 
  f = 3 / 4 :=
by
  sorry

end fraction_of_number_is_three_quarters_l164_164395


namespace find_b_for_perpendicular_lines_l164_164670

theorem find_b_for_perpendicular_lines:
  (∃ b : ℝ, ∀ (x y : ℝ), (3 * x + y - 5 = 0) ∧ (b * x + y + 2 = 0) → b = -1/3) :=
by
  sorry

end find_b_for_perpendicular_lines_l164_164670


namespace numbers_not_coprime_l164_164005

theorem numbers_not_coprime (b : ℕ) (h : b = 2013^2013 + 2) : Int.gcd ((b^3 + 1 : ℤ)) ((b^2 + 2 : ℤ)) ≠ 1 := 
sorry

end numbers_not_coprime_l164_164005


namespace no_half_dimension_cuboid_l164_164806

theorem no_half_dimension_cuboid
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) :
  ¬ (a' * b' * c' = (1 / 2) * a * b * c ∧ 2 * (a' * b' + b' * c' + c' * a') = a * b + b * c + c * a) :=
by
  sorry

end no_half_dimension_cuboid_l164_164806


namespace find_x_l164_164441

theorem find_x (x : ℚ) : x * 9999 = 724827405 → x = 72492.75 :=
by
  sorry

end find_x_l164_164441


namespace arc_PQ_circumference_l164_164209

-- Definitions based on the identified conditions
def radius : ℝ := 24
def angle_PRQ : ℝ := 90

-- The theorem to prove based on the question and correct answer
theorem arc_PQ_circumference : 
  angle_PRQ = 90 → 
  ∃ arc_length : ℝ, arc_length = (2 * Real.pi * radius) / 4 ∧ arc_length = 12 * Real.pi :=
by
  sorry

end arc_PQ_circumference_l164_164209


namespace cell_count_at_end_of_days_l164_164855

-- Defining the conditions
def initial_cells : ℕ := 2
def split_ratio : ℕ := 3
def days : ℕ := 9
def cycle_days : ℕ := 3

-- The main statement to be proved
theorem cell_count_at_end_of_days :
  (initial_cells * split_ratio^((days / cycle_days) - 1)) = 18 :=
by
  sorry

end cell_count_at_end_of_days_l164_164855


namespace find_expression_value_l164_164303

theorem find_expression_value (x : ℝ) : 
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  have h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := sorry
  exact h

end find_expression_value_l164_164303


namespace combined_sleep_hours_l164_164724

theorem combined_sleep_hours :
  let connor_sleep_hours := 6
  let luke_sleep_hours := connor_sleep_hours + 2
  let emma_sleep_hours := connor_sleep_hours - 1
  let ava_sleep_hours :=
    2 * 5 + 
    2 * (5 + 1) + 
    2 * (5 + 2) + 
    (5 + 3)
  let puppy_sleep_hours := 2 * luke_sleep_hours
  let cat_sleep_hours := 4 + 7
  7 * connor_sleep_hours +
  7 * luke_sleep_hours +
  7 * emma_sleep_hours +
  ava_sleep_hours +
  7 * puppy_sleep_hours +
  7 * cat_sleep_hours = 366 :=
by
  sorry

end combined_sleep_hours_l164_164724


namespace maddie_spent_in_all_l164_164947

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l164_164947


namespace cost_apples_l164_164372

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l164_164372


namespace roots_quadratic_eq_value_l164_164786

theorem roots_quadratic_eq_value (d e : ℝ) (h : 3 * d^2 + 4 * d - 7 = 0) (h' : 3 * e^2 + 4 * e - 7 = 0) : 
  (d - 2) * (e - 2) = 13 / 3 := 
by
  sorry

end roots_quadratic_eq_value_l164_164786


namespace D_72_value_l164_164941

-- Define D(n) as described
def D (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual function definition

-- Theorem statement
theorem D_72_value : D 72 = 97 :=
by sorry

end D_72_value_l164_164941


namespace initial_birds_l164_164033

theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
sorry

end initial_birds_l164_164033


namespace inequality_proof_l164_164017

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 :=
by
  sorry

end inequality_proof_l164_164017


namespace dad_steps_l164_164124

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l164_164124


namespace largest_integral_x_l164_164734

theorem largest_integral_x (x : ℤ) : 
  (1 / 4 : ℝ) < (x / 7) ∧ (x / 7) < (7 / 11 : ℝ) → x ≤ 4 := 
  sorry

end largest_integral_x_l164_164734


namespace dad_steps_l164_164120

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l164_164120


namespace success_permutations_correct_l164_164878

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l164_164878


namespace minimum_value_l164_164944

theorem minimum_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ x : ℝ, x = 4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ∧ x ≥ 4 * Real.sqrt 3 :=
by sorry

end minimum_value_l164_164944


namespace new_milk_water_ratio_l164_164546

theorem new_milk_water_ratio
  (original_milk : ℚ)
  (original_water : ℚ)
  (added_water : ℚ)
  (h_ratio : original_milk / original_water = 2 / 1)
  (h_milk_qty : original_milk = 45)
  (h_added_water : added_water = 10) :
  original_milk / (original_water + added_water) = 18 / 13 :=
by
  sorry

end new_milk_water_ratio_l164_164546


namespace min_heaviest_weight_l164_164821

theorem min_heaviest_weight : 
  ∃ (w : ℕ), ∀ (weights : Fin 8 → ℕ),
    (∀ i j, i ≠ j → weights i ≠ weights j) ∧
    (∀ (a b c d : Fin 8),
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      (weights a + weights b) ≠ (weights c + weights d) ∧ 
      max (max (weights a) (weights b)) (max (weights c) (weights d)) >= w) 
  → w = 34 := 
by
  sorry

end min_heaviest_weight_l164_164821


namespace complex_purely_imaginary_m_value_l164_164622

theorem complex_purely_imaginary_m_value (m : ℝ) :
  (m^2 - 1 = 0) ∧ (m + 1 ≠ 0) → m = 1 :=
by
  sorry

end complex_purely_imaginary_m_value_l164_164622


namespace expected_value_of_a_squared_l164_164068

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l164_164068


namespace binomial_16_12_l164_164425

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l164_164425


namespace polygon_sides_l164_164714

theorem polygon_sides (n : ℕ) (h : n * (n - 3) / 2 = 20) : n = 8 :=
by
  sorry

end polygon_sides_l164_164714


namespace allocation_schemes_l164_164975

theorem allocation_schemes (A B C : Type)
  (P : A -> B -> Prop)
  [finite A] [finite B]
  (n_people : fintype.card A = 3)
  (n_communities : fintype.card B = 7)
  (max_people_in_community : ∀ b : B, (∑ a : A, if P a b then 1 else 0) <= 2):
  ∃ (allocation_schemes : nat), allocation_schemes = 336 := 
sorry

end allocation_schemes_l164_164975


namespace convert_base8_to_base7_l164_164584

theorem convert_base8_to_base7 (n : ℕ) : n = 536 → (num_to_base7 536) = 1054 :=
by
  sorry

def num_to_base10 (n : ℕ) : ℕ :=
  let d2 := (n / 100) % 10 * 8^2
  let d1 := (n / 10) % 10 * 8^1
  let d0 := (n / 1) % 10 * 8^0
  d2 + d1 + d0

def num_to_base7_aux (n : ℕ) (acc : ℕ) (pos : ℕ) : ℕ :=
  if n = 0 then acc
  else
    let q := n / 7
    let r := n % 7
    num_to_base7_aux q ((r * 10^pos) + acc) (pos + 1)

def num_to_base7 (n : ℕ) : ℕ :=
  num_to_base7_aux (num_to_base10 n) 0 0

end convert_base8_to_base7_l164_164584


namespace avg_speed_between_B_and_C_l164_164862

noncomputable def avg_speed_from_B_to_C : ℕ := 20

theorem avg_speed_between_B_and_C
    (A_to_B_dist : ℕ := 120)
    (A_to_B_time : ℕ := 4)
    (B_to_C_dist : ℕ := 120) -- three-thirds of A_to_B_dist
    (C_to_D_dist : ℕ := 60) -- half of B_to_C_dist
    (C_to_D_time : ℕ := 2)
    (total_avg_speed : ℕ := 25)
    : avg_speed_from_B_to_C = 20 := 
  sorry

end avg_speed_between_B_and_C_l164_164862


namespace maddie_spent_in_all_l164_164946

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l164_164946


namespace tom_weekly_fluid_intake_l164_164296

-- Definitions based on the conditions.
def soda_cans_per_day : ℕ := 5
def ounces_per_can : ℕ := 12
def water_ounces_per_day : ℕ := 64
def days_per_week : ℕ := 7

-- The mathematical proof problem statement.
theorem tom_weekly_fluid_intake :
  (soda_cans_per_day * ounces_per_can + water_ounces_per_day) * days_per_week = 868 := 
by
  sorry

end tom_weekly_fluid_intake_l164_164296


namespace minimum_knights_l164_164800

def T_shirtNumber := Fin 80 -- Represent T-shirt numbers from 1 to 80
def Islander := {i // i < 80} -- Each islander is associated with a T-shirt number

-- Knight and Liar definitions
def is_knight (i : Islander) : Prop := sorry -- Definition to be refined
def is_liar (i : Islander) : Prop := sorry -- Definition to be refined

-- Statements that islanders can make
def statement1 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 > i.1 ∧ sorry

def statement2 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 < i.1 ∧ sorry

-- Problem statement: Proving the minimum number of knights
theorem minimum_knights (kn : Fin 80 → bool) :
  (∀ i, if kn i then is_knight i else is_liar i) →
  (∀ i, is_knight i → (statement1 i ∨ statement2 i)) →
  (∀ i, is_liar i → ¬(statement1 i ∨ statement2 i)) →
  (∃ (k_cnt : ℕ), k_cnt = 70 ∧ sorry) :=
sorry

end minimum_knights_l164_164800


namespace worker_savings_fraction_l164_164055

theorem worker_savings_fraction (P : ℝ) (F : ℝ) (h1 : P > 0) (h2 : 12 * F * P = 5 * (1 - F) * P) : F = 5 / 17 :=
by
  sorry

end worker_savings_fraction_l164_164055


namespace dad_steps_90_l164_164110

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l164_164110


namespace smallest_palindrome_divisible_by_6_l164_164688

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end smallest_palindrome_divisible_by_6_l164_164688


namespace difference_of_squares_division_l164_164261

theorem difference_of_squares_division :
  let a := 121
  let b := 112
  (a^2 - b^2) / 3 = 699 :=
by
  sorry

end difference_of_squares_division_l164_164261


namespace total_hours_worked_l164_164561

def hours_day1 : ℝ := 2.5
def increment_day2 : ℝ := 0.5
def hours_day2 : ℝ := hours_day1 + increment_day2
def hours_day3 : ℝ := 3.75

theorem total_hours_worked :
  hours_day1 + hours_day2 + hours_day3 = 9.25 :=
sorry

end total_hours_worked_l164_164561


namespace time_to_overflow_equals_correct_answer_l164_164954

-- Definitions based on conditions
def pipeA_fill_time : ℚ := 32
def pipeB_fill_time : ℚ := pipeA_fill_time / 5

-- Derived rates from the conditions
def pipeA_rate : ℚ := 1 / pipeA_fill_time
def pipeB_rate : ℚ := 1 / pipeB_fill_time
def combined_rate : ℚ := pipeA_rate + pipeB_rate

-- The time to overflow when both pipes are filling the tank simultaneously
def time_to_overflow : ℚ := 1 / combined_rate

-- The statement we are going to prove
theorem time_to_overflow_equals_correct_answer : time_to_overflow = 16 / 3 :=
by sorry

end time_to_overflow_equals_correct_answer_l164_164954


namespace Finley_age_proof_l164_164028

variable (Jill_age : ℕ) (Roger_age : ℕ) (Finley_age : ℕ)

-- Condition 1: Jill is 20 years old now
def Jill_current_age : Jill_age = 20 := rfl

-- Condition 2: Roger's age is 5 more than twice Jill's age
def Roger_age_relation : Roger_age = 2 * Jill_age + 5 := sorry

-- Condition 3: In 15 years, their age difference will be 30 years less than Finley's age
def Finley_age_relation (Jill_future_age Roger_future_age Finley_future_age : ℕ) : 
  Jill_future_age = Jill_age + 15 ∧ 
  Roger_future_age = Roger_age + 15 ∧ 
  Finley_future_age = Finley_age + 15 ∧ 
  (Roger_future_age - Jill_future_age = Finley_future_age - 30) := 
  sorry

-- Theorem: Find Finley's current age
theorem Finley_age_proof : Finley_age = 40 :=
by
  -- Assume all conditions mentioned above
  let Jill_age := 20
  let Roger_age := 2 * Jill_age + 5
  let Jill_future_age := Jill_age + 15
  let Roger_future_age := Roger_age + 15
  let Finley_future_age := Finley_age + 15

  -- Using the relation for Finley’s age in the future
  have h1 : Roger_future_age - Jill_future_age = Finley_future_age - 30 := sorry

  -- Calculate Jill's, Roger's, and Finley's future ages and show the math relation
  have h2 : Jill_future_age = 35 := by simp [Jill_age]
  have h3 : Roger_future_age = 60 := by simp [Jill_age, Roger_age]
  have h4 : Roger_future_age - Jill_future_age = 25 := by simp [h2, h3]

  -- Find Finley's future age
  have h5 : Finley_future_age = 55 := by linarith [h1, h4]

  -- Determine Finley's current age
  have h6 : Finley_age = 40 := by simp [h5]

  exact h6

end Finley_age_proof_l164_164028


namespace points_on_circle_l164_164091

theorem points_on_circle (n : ℕ) (h1 : ∃ (k : ℕ), k = (35 - 7) ∧ n = 2 * k) : n = 56 :=
sorry

end points_on_circle_l164_164091


namespace prob_xi_eq_12_l164_164331

noncomputable def prob_of_draws (total_draws red_draws : ℕ) (prob_red prob_white : ℚ) : ℚ :=
    (Nat.choose (total_draws - 1) (red_draws - 1)) * (prob_red ^ (red_draws - 1)) * (prob_white ^ (total_draws - red_draws)) * prob_red

theorem prob_xi_eq_12 :
    prob_of_draws 12 10 (3 / 8) (5 / 8) = 
    (Nat.choose 11 9) * (3 / 8)^9 * (5 / 8)^2 * (3 / 8) :=
by sorry

end prob_xi_eq_12_l164_164331


namespace ellipse_condition_l164_164321

theorem ellipse_condition (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m) = 1) →
  (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_l164_164321


namespace coo_coo_count_correct_l164_164407

theorem coo_coo_count_correct :
  let monday_coos := 89
  let tuesday_coos := 179
  let wednesday_coos := 21
  let total_coos := monday_coos + tuesday_coos + wednesday_coos
  total_coos = 289 :=
by
  sorry

end coo_coo_count_correct_l164_164407


namespace partial_fraction_series_sum_l164_164106

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end partial_fraction_series_sum_l164_164106


namespace expected_value_a_squared_is_correct_l164_164069

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l164_164069


namespace at_least_one_travels_l164_164435

-- Define the probabilities of A and B traveling
def P_A := 1 / 3
def P_B := 1 / 4

-- Define the probability that person A does not travel
def P_not_A := 1 - P_A

-- Define the probability that person B does not travel
def P_not_B := 1 - P_B

-- Define the probability that neither person A nor person B travels
def P_neither := P_not_A * P_not_B

-- Define the probability that at least one of them travels
def P_at_least_one := 1 - P_neither

theorem at_least_one_travels : P_at_least_one = 1 / 2 := by
  sorry

end at_least_one_travels_l164_164435


namespace triangle_inradius_l164_164244

theorem triangle_inradius (A p r : ℝ) 
    (h1 : p = 35) 
    (h2 : A = 78.75) 
    (h3 : A = (r * p) / 2) : 
    r = 4.5 :=
sorry

end triangle_inradius_l164_164244


namespace area_of_polygon_DEFG_l164_164933

-- Given conditions
def isosceles_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 ∧ AC = 2 ∧ BC = 1

def square (side : ℝ) : ℝ :=
  side * side

def constructed_square_areas_equal (AB AC : ℝ) (D E F G : Type) : Prop :=
  square AB = square AC ∧ square AB = 4 ∧ square AC = 4

-- Question to prove
theorem area_of_polygon_DEFG (A B C D E F G : Type) (AB AC BC : ℝ) 
  (h1 : isosceles_triangle A B C AB AC BC) 
  (h2 : constructed_square_areas_equal AB AC D E F G) : 
  square AB + square AC = 8 :=
by
  sorry

end area_of_polygon_DEFG_l164_164933


namespace width_of_rectangle_l164_164664

-- Define the problem constants and parameters
variable (L W : ℝ)

-- State the main theorem about the width
theorem width_of_rectangle (h₁ : L * W = 50) (h₂ : L + W = 15) : W = 5 :=
sorry

end width_of_rectangle_l164_164664


namespace compute_expression_l164_164528

-- Lean 4 statement for the mathematic equivalence proof problem
theorem compute_expression:
  (1004^2 - 996^2 - 1002^2 + 998^2) = 8000 := by
  sorry

end compute_expression_l164_164528


namespace value_of_x_minus_y_squared_l164_164166

theorem value_of_x_minus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : (x - y) ^ 2 = 1 :=
by
  sorry

end value_of_x_minus_y_squared_l164_164166


namespace pq_logic_l164_164190

theorem pq_logic (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by
  sorry

end pq_logic_l164_164190


namespace distance_at_40_kmph_l164_164394

theorem distance_at_40_kmph (x y : ℕ) 
  (h1 : x + y = 250) 
  (h2 : x / 40 + y / 60 = 6) : 
  x = 220 :=
by
  sorry

end distance_at_40_kmph_l164_164394


namespace inequality_solution_min_value_of_a2_b2_c2_min_achieved_l164_164464

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

theorem inequality_solution :
  ∀ x : ℝ, (f x ≥ 3) ↔ (x ≤ -1 ∨ x ≥ 1) :=
by sorry

theorem min_value_of_a2_b2_c2 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  a^2 + b^2 + c^2 ≥ 3/7 :=
by sorry

theorem min_achieved (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  (2*a = b) ∧ (b = c/2) ∧ (a^2 + b^2 + c^2 = 3/7) :=
by sorry

end inequality_solution_min_value_of_a2_b2_c2_min_achieved_l164_164464


namespace james_pays_660_for_bed_and_frame_l164_164201

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end james_pays_660_for_bed_and_frame_l164_164201


namespace average_percentage_of_10_students_l164_164701

theorem average_percentage_of_10_students 
  (avg_15_students : ℕ := 80)
  (n_15_students : ℕ := 15)
  (total_students : ℕ := 25)
  (overall_avg : ℕ := 84) : 
  ∃ (x : ℕ), ((n_15_students * avg_15_students + 10 * x) / total_students = overall_avg) → x = 90 := 
sorry

end average_percentage_of_10_students_l164_164701


namespace sum_of_consecutive_page_numbers_l164_164508

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20250) : n + (n + 1) = 285 := 
sorry

end sum_of_consecutive_page_numbers_l164_164508


namespace root_proof_l164_164161

noncomputable def p : ℝ := (-5 + Real.sqrt 21) / 2
noncomputable def q : ℝ := (-5 - Real.sqrt 21) / 2

theorem root_proof :
  (∃ (p q : ℝ), (∀ x : ℝ, x^3 + 6 * x^2 + 6 * x + 1 = 0 → (x = p ∨ x = q ∨ x = -1)) ∧ 
                 ((p = (-5 + Real.sqrt 21) / 2) ∧ (q = (-5 - Real.sqrt 21) / 2))) →
  (p / q + q / p = 23) :=
by
  sorry

end root_proof_l164_164161


namespace intersection_A_B_l164_164609

def A : Set ℝ := { x | |x - 1| < 2 }
def B : Set ℝ := { x | Real.log x / Real.log 2 ≤ 1 }

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := 
sorry

end intersection_A_B_l164_164609


namespace sum_of_D_coordinates_l164_164802

noncomputable def sum_of_coordinates_of_D (D : ℝ × ℝ) (M C : ℝ × ℝ) : ℝ :=
  D.1 + D.2

theorem sum_of_D_coordinates (D M C : ℝ × ℝ) (H_M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) 
                             (H_M_value : M = (5, 9)) (H_C_value : C = (11, 5)) : 
                             sum_of_coordinates_of_D D M C = 12 :=
sorry

end sum_of_D_coordinates_l164_164802


namespace determine_value_of_product_l164_164493

theorem determine_value_of_product (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := 
by 
  sorry

end determine_value_of_product_l164_164493


namespace find_m_l164_164667

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) = (m^2 - m - 5) * x^(m - 1) ∧ 
  (m^2 - m - 5) * (m - 1) * x^(m - 2) > 0) → m = 3 :=
by
  sorry

end find_m_l164_164667


namespace correct_option_is_D_l164_164302

noncomputable def expression1 (a b : ℝ) : Prop := a + b > 2 * b^2
noncomputable def expression2 (a b : ℝ) : Prop := a^5 + b^5 > a^3 * b^2 + a^2 * b^3
noncomputable def expression3 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * (a - b - 1)
noncomputable def expression4 (a b : ℝ) : Prop := (b / a) + (a / b) > 2

theorem correct_option_is_D (a b : ℝ) (h : a ≠ b) : 
  (expression3 a b ∧ ¬expression1 a b ∧ ¬expression2 a b ∧ ¬expression4 a b) :=
by
  sorry

end correct_option_is_D_l164_164302


namespace cathy_total_money_l164_164100

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l164_164100


namespace quadratic_roots_l164_164820

theorem quadratic_roots (x : ℝ) : (x ^ 2 - 3 = 0) → (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  intro h
  sorry

end quadratic_roots_l164_164820


namespace quadratic_to_square_form_l164_164109

theorem quadratic_to_square_form (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) :=
sorry

end quadratic_to_square_form_l164_164109


namespace find_k_value_l164_164456

theorem find_k_value (x k : ℝ) (h : x = -3) (h_eq : k * (x - 2) - 4 = k - 2 * x) : k = -5/3 := by
  sorry

end find_k_value_l164_164456


namespace min_value_2_l164_164608

noncomputable def min_value (a b : ℝ) : ℝ :=
  1 / a + 1 / (b + 1)

theorem min_value_2 {a b : ℝ} (h1 : a > 0) (h2 : b > -1) (h3 : a + b = 1) : min_value a b = 2 :=
by
  sorry

end min_value_2_l164_164608


namespace power_of_i_l164_164981

theorem power_of_i : (Complex.I ^ 2018) = -1 := by
  sorry

end power_of_i_l164_164981


namespace expected_value_gt_median_l164_164707

variable {a b : ℝ} (f : ℝ → ℝ) [measure_space ℝ] (X : ℝ → ℝ) [is_probability_measure X]

-- Assume the conditions of the problem
-- f is the probability density function of the random variable X
-- f(x) = 0 for x < a and x >= b
-- f is continuous, positive, and monotonically decreasing on [a, b)

def pdf_support (x : ℝ) : Prop := (a ≤ x) ∧ (x < b)
def pdf_zero_outside (x : ℝ) : Prop := (x < a) ∨ (x ≥ b) → f x = 0
def pdf_positive (x : ℝ) : Prop := pdf_support x → 0 < f x
def pdf_monotonic (x₁ x₂ : ℝ) : Prop := (pdf_support x₁ ∧ pdf_support x₂ ∧ x₁ < x₂) → f x₁ ≥ f x₂

theorem expected_value_gt_median
  (median_zero : ∫ x in (set_of pdf_support), f x dX = 0.5) 
  (h_zero_outside : ∀ x, pdf_zero_outside x) 
  (h_continuous : continuous_on f (set_of pdf_support))
  (h_positive : ∀ x, pdf_positive x)
  (h_monotonic : ∀ x₁ x₂, pdf_monotonic x₁ x₂) :
  ∫ x, x * f x dX > 0 := 
by
  sorry

end expected_value_gt_median_l164_164707


namespace sticker_price_l164_164180

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l164_164180


namespace kerry_age_l164_164013

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end kerry_age_l164_164013


namespace sum_of_cubes_of_roots_l164_164492

theorem sum_of_cubes_of_roots (P : Polynomial ℝ)
  (hP : P = Polynomial.C (-1) + Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X) 
  (x1 x2 x3 : ℝ) 
  (hr : P.eval x1 = 0 ∧ P.eval x2 = 0 ∧ P.eval x3 = 0) :
  x1^3 + x2^3 + x3^3 = 3 := 
sorry

end sum_of_cubes_of_roots_l164_164492


namespace number_of_solutions_l164_164613

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x < 150 ∧ x > 0 → (x + 25) % 47 = 80 % 47 → x ∈ {8 + 47*k | k : ℕ ∧ k < 4}) ∧ n = 4 :=
by
  sorry

end number_of_solutions_l164_164613


namespace probability_divisible_by_3_l164_164225

theorem probability_divisible_by_3 (a b c : ℕ) (h : a ∈ Finset.range 2008 ∧ b ∈ Finset.range 2008 ∧ c ∈ Finset.range 2008) :
  (∃ p : ℚ, p = 1265/2007 ∧ (abc + ac + a) % 3 = 0) :=
sorry

end probability_divisible_by_3_l164_164225


namespace change_sum_equals_108_l164_164793

theorem change_sum_equals_108 :
  ∃ (amounts : List ℕ), (∀ a ∈ amounts, a < 100 ∧ ((a % 25 = 4) ∨ (a % 5 = 4))) ∧
    amounts.sum = 108 := 
by
  sorry

end change_sum_equals_108_l164_164793


namespace binom_16_12_eq_1820_l164_164423

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l164_164423


namespace tripod_height_l164_164557

-- Define the conditions of the problem
structure Tripod where
  leg_length : ℝ
  angle_equal : Bool
  top_height : ℝ
  broken_length : ℝ

def m : ℕ := 27
def n : ℕ := 10

noncomputable def h : ℝ := m / Real.sqrt n

theorem tripod_height :
  ∀ (t : Tripod),
  t.leg_length = 6 →
  t.angle_equal = true →
  t.top_height = 3 →
  t.broken_length = 2 →
  (h = m / Real.sqrt n) →
  (⌊m + Real.sqrt n⌋ = 30) :=
by
  intros
  sorry

end tripod_height_l164_164557


namespace real_roots_exist_for_all_real_K_l164_164894

theorem real_roots_exist_for_all_real_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x-1) * (x-2) * (x-3) :=
by
  sorry

end real_roots_exist_for_all_real_K_l164_164894


namespace dad_steps_l164_164132

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l164_164132


namespace lower_limit_of_arun_weight_l164_164282

-- Given conditions for Arun's weight
variables (W : ℝ)
variables (avg_val : ℝ)

-- Define the conditions
def arun_weight_condition_1 := W < 72
def arun_weight_condition_2 := 60 < W ∧ W < 70
def arun_weight_condition_3 := W ≤ 67
def arun_weight_avg := avg_val = 66

-- The math proof problem statement
theorem lower_limit_of_arun_weight 
  (h1: arun_weight_condition_1 W) 
  (h2: arun_weight_condition_2 W) 
  (h3: arun_weight_condition_3 W) 
  (h4: arun_weight_avg avg_val) :
  ∃ (lower_limit : ℝ), lower_limit = 65 :=
sorry

end lower_limit_of_arun_weight_l164_164282


namespace concert_ticket_to_motorcycle_ratio_l164_164006

theorem concert_ticket_to_motorcycle_ratio (initial_amount spend_motorcycle remaining_amount : ℕ)
  (h_initial : initial_amount = 5000)
  (h_spend_motorcycle : spend_motorcycle = 2800)
  (amount_left := initial_amount - spend_motorcycle)
  (h_remaining : remaining_amount = 825)
  (h_amount_left : ∃ C : ℕ, amount_left - C - (1/4 : ℚ) * (amount_left - C) = remaining_amount) :
  ∃ C : ℕ, (C / amount_left) = (1 / 2 : ℚ) := sorry

end concert_ticket_to_motorcycle_ratio_l164_164006


namespace sum_of_faces_l164_164437

variable (a d b c e f : ℕ)
variable (pos_a : a > 0) (pos_d : d > 0) (pos_b : b > 0) (pos_c : c > 0) 
variable (pos_e : e > 0) (pos_f : f > 0)
variable (h : a * b * e + a * b * f + a * c * e + a * c * f + d * b * e + d * b * f + d * c * e + d * c * f = 1176)

theorem sum_of_faces : a + d + b + c + e + f = 33 := by
  sorry

end sum_of_faces_l164_164437


namespace employed_females_percentage_l164_164777

-- Definitions of the conditions
def employment_rate : ℝ := 0.60
def male_employment_rate : ℝ := 0.15

-- The theorem to prove
theorem employed_females_percentage : employment_rate - male_employment_rate = 0.45 := by
  sorry

end employed_females_percentage_l164_164777


namespace problem_p_s_difference_l164_164024

def P : ℤ := 12 - (3 * 4)
def S : ℤ := (12 - 3) * 4

theorem problem_p_s_difference : P - S = -36 := by
  sorry

end problem_p_s_difference_l164_164024


namespace percent_of_100_is_30_l164_164702

theorem percent_of_100_is_30 : (30 / 100) * 100 = 30 := 
by
  sorry

end percent_of_100_is_30_l164_164702


namespace solution_set_inequality_l164_164672

theorem solution_set_inequality (x : ℝ) : (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := 
sorry

end solution_set_inequality_l164_164672


namespace correct_comprehensive_survey_l164_164527

-- Definitions for the types of surveys.
inductive Survey
| A : Survey
| B : Survey
| C : Survey
| D : Survey

-- Function that identifies the survey suitable for a comprehensive survey.
def is_comprehensive_survey (s : Survey) : Prop :=
  match s with
  | Survey.A => False            -- A is for sampling, not comprehensive
  | Survey.B => False            -- B is for sampling, not comprehensive
  | Survey.C => False            -- C is for sampling, not comprehensive
  | Survey.D => True             -- D is suitable for comprehensive survey

-- The theorem to prove that D is the correct answer.
theorem correct_comprehensive_survey : is_comprehensive_survey Survey.D = True := by
  sorry

end correct_comprehensive_survey_l164_164527


namespace sin_60_proof_l164_164577

noncomputable def sin_60_eq : Prop :=
  sin (60 * real.pi / 180) = real.sqrt 3 / 2

theorem sin_60_proof : sin_60_eq :=
sorry

end sin_60_proof_l164_164577


namespace greatest_value_of_sum_l164_164796

variable (a b c : ℕ)

theorem greatest_value_of_sum
    (h1 : 2022 < a)
    (h2 : 2022 < b)
    (h3 : 2022 < c)
    (h4 : ∃ k1 : ℕ, a + b = k1 * (c - 2022))
    (h5 : ∃ k2 : ℕ, a + c = k2 * (b - 2022))
    (h6 : ∃ k3 : ℕ, b + c = k3 * (a - 2022)) :
    a + b + c = 2022 * 85 := 
  sorry

end greatest_value_of_sum_l164_164796


namespace Tn_lt_Sn_div_2_l164_164784

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n

noncomputable def S (n : ℕ) : ℝ := 
  (3 / 2) * (1 - (1 / 3)^n)

noncomputable def T (n : ℕ) : ℝ := 
  (3 / 4) * (1 - (1 / 3)^n) - (n / 2) * (1 / 3)^(n + 1)

theorem Tn_lt_Sn_div_2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l164_164784


namespace rectangle_area_and_perimeter_l164_164087

-- Given conditions as definitions
def length : ℕ := 5
def width : ℕ := 3

-- Proof problems
theorem rectangle_area_and_perimeter :
  (length * width = 15) ∧ (2 * (length + width) = 16) :=
by
  sorry

end rectangle_area_and_perimeter_l164_164087


namespace Sara_sister_notebooks_l164_164499

theorem Sara_sister_notebooks :
  let initial_notebooks := 4 
  let ordered_notebooks := (3 / 2) * initial_notebooks -- 150% more notebooks
  let notebooks_after_order := initial_notebooks + ordered_notebooks
  let notebooks_after_loss := notebooks_after_order - 2 -- lost 2 notebooks
  let sold_notebooks := (1 / 4) * notebooks_after_loss -- sold 25% of remaining notebooks
  let notebooks_after_sales := notebooks_after_loss - sold_notebooks
  let notebooks_after_giveaway := notebooks_after_sales - 3 -- gave away 3 notebooks
  notebooks_after_giveaway = 3 := 
by {
  sorry
}

end Sara_sister_notebooks_l164_164499


namespace logan_average_speed_l164_164970

theorem logan_average_speed 
  (tamika_hours : ℕ)
  (tamika_speed : ℕ)
  (logan_hours : ℕ)
  (tamika_distance : ℕ)
  (logan_distance : ℕ)
  (distance_diff : ℕ)
  (diff_condition : tamika_distance = logan_distance + distance_diff) :
  tamika_hours = 8 →
  tamika_speed = 45 →
  logan_hours = 5 →
  tamika_distance = tamika_speed * tamika_hours →
  distance_diff = 85 →
  logan_distance / logan_hours = 55 :=
by
  sorry

end logan_average_speed_l164_164970


namespace dad_steps_l164_164117

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l164_164117


namespace arrangement_count_is_43200_l164_164251

noncomputable def number_of_arrangements : Nat :=
  let number_of_boys := 6
  let number_of_girls := 3
  let boys_arrangements := Nat.factorial number_of_boys
  let spaces := number_of_boys - 1
  let girls_arrangements := Nat.factorial (spaces) / Nat.factorial (spaces - number_of_girls)
  boys_arrangements * girls_arrangements

theorem arrangement_count_is_43200 :
  number_of_arrangements = 43200 := by
  sorry

end arrangement_count_is_43200_l164_164251


namespace solve_equation_l164_164749

-- Define the equation as a function of y
def equation (y : ℝ) : ℝ :=
  y^4 - 20 * y + 1

-- State the theorem that y = -1 satisfies the equation.
theorem solve_equation : equation (-1) = 22 := 
  sorry

end solve_equation_l164_164749


namespace milkman_total_profit_l164_164545

-- Declare the conditions
def initialMilk : ℕ := 50
def initialWater : ℕ := 15
def firstMixtureMilk : ℕ := 30
def firstMixtureWater : ℕ := 8
def remainingMilk : ℕ := initialMilk - firstMixtureMilk
def secondMixtureMilk : ℕ := remainingMilk
def secondMixtureWater : ℕ := 7
def costOfMilkPerLiter : ℕ := 20
def sellingPriceFirstMixturePerLiter : ℕ := 17
def sellingPriceSecondMixturePerLiter : ℕ := 15
def totalCostOfMilk := (firstMixtureMilk + secondMixtureMilk) * costOfMilkPerLiter
def totalRevenueFirstMixture := (firstMixtureMilk + firstMixtureWater) * sellingPriceFirstMixturePerLiter
def totalRevenueSecondMixture := (secondMixtureMilk + secondMixtureWater) * sellingPriceSecondMixturePerLiter
def totalRevenue := totalRevenueFirstMixture + totalRevenueSecondMixture
def totalProfit := totalRevenue - totalCostOfMilk

-- Proof statement
theorem milkman_total_profit : totalProfit = 51 := by
  sorry

end milkman_total_profit_l164_164545


namespace number_of_tiles_l164_164553

open Real

noncomputable def room_length : ℝ := 10
noncomputable def room_width : ℝ := 15
noncomputable def tile_length : ℝ := 5 / 12
noncomputable def tile_width : ℝ := 2 / 3

theorem number_of_tiles :
  (room_length * room_width) / (tile_length * tile_width) = 540 := by
  sorry

end number_of_tiles_l164_164553


namespace gcd_expression_l164_164299

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression_l164_164299


namespace standard_equation_of_parabola_l164_164459

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l164_164459


namespace slices_per_person_is_correct_l164_164048

-- Conditions
def slices_per_tomato : Nat := 8
def total_tomatoes : Nat := 20
def people_for_meal : Nat := 8

-- Calculate number of slices for a single person
def slices_needed_for_single_person (slices_per_tomato : Nat) (total_tomatoes : Nat) (people_for_meal : Nat) : Nat :=
  (slices_per_tomato * total_tomatoes) / people_for_meal

-- The statement to be proved
theorem slices_per_person_is_correct : slices_needed_for_single_person slices_per_tomato total_tomatoes people_for_meal = 20 :=
by
  sorry

end slices_per_person_is_correct_l164_164048


namespace division_problem_l164_164384

theorem division_problem : 0.05 / 0.0025 = 20 := 
sorry

end division_problem_l164_164384


namespace apples_in_third_basket_l164_164984

theorem apples_in_third_basket (total_apples : ℕ) (x : ℕ) (y : ℕ) 
    (h_total : total_apples = 2014)
    (h_second_basket : 49 + x = total_apples - 2 * y - x - y)
    (h_first_basket : total_apples - 2 * y - x + y = 2 * y)
    : x + y = 655 :=
by
    sorry

end apples_in_third_basket_l164_164984


namespace child_l164_164863

noncomputable def child's_ticket_cost : ℕ :=
  let adult_ticket_price := 7
  let total_tickets := 900
  let total_revenue := 5100
  let childs_tickets_sold := 400
  let adult_tickets_sold := total_tickets - childs_tickets_sold
  let total_adult_revenue := adult_tickets_sold * adult_ticket_price
  let total_child_revenue := total_revenue - total_adult_revenue
  let child's_ticket_price := total_child_revenue / childs_tickets_sold
  child's_ticket_price

theorem child's_ticket_cost_is_4 : child's_ticket_cost = 4 :=
by
  have adult_ticket_price := 7
  have total_tickets := 900
  have total_revenue := 5100
  have childs_tickets_sold := 400
  have adult_tickets_sold := total_tickets - childs_tickets_sold
  have total_adult_revenue := adult_tickets_sold * adult_ticket_price
  have total_child_revenue := total_revenue - total_adult_revenue
  have child's_ticket_price := total_child_revenue / childs_tickets_sold
  show child's_ticket_cost = 4
  sorry

end child_l164_164863


namespace total_weight_of_peppers_l164_164617

theorem total_weight_of_peppers
  (green_peppers : ℝ) 
  (red_peppers : ℝ)
  (h_green : green_peppers = 0.33)
  (h_red : red_peppers = 0.33) :
  green_peppers + red_peppers = 0.66 := 
by
  sorry

end total_weight_of_peppers_l164_164617


namespace count_success_permutations_l164_164881

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l164_164881


namespace bus_problem_l164_164329

-- Define the participants in 2005
def participants_2005 (k : ℕ) : ℕ := 27 * k + 19

-- Define the participants in 2006
def participants_2006 (k : ℕ) : ℕ := participants_2005 k + 53

-- Define the total number of buses needed in 2006
def buses_needed_2006 (k : ℕ) : ℕ := (participants_2006 k) / 27 + if (participants_2006 k) % 27 = 0 then 0 else 1

-- Define the total number of buses needed in 2005
def buses_needed_2005 (k : ℕ) : ℕ := k + 1

-- Define the additional buses needed in 2006 compared to 2005
def additional_buses_2006 (k : ℕ) := buses_needed_2006 k - buses_needed_2005 k

-- Define the number of people in the incomplete bus in 2006
def people_in_incomplete_bus_2006 (k : ℕ) := (participants_2006 k) % 27

-- The proof statement to be proved
theorem bus_problem (k : ℕ) : additional_buses_2006 k = 2 ∧ people_in_incomplete_bus_2006 k = 9 := by
  sorry

end bus_problem_l164_164329


namespace rational_solutions_k_l164_164895

theorem rational_solutions_k (k : ℕ) (h : k > 0) : (∃ x : ℚ, 2 * (k : ℚ) * x^2 + 36 * x + 3 * (k : ℚ) = 0) → k = 6 :=
by
  -- proof to be written
  sorry

end rational_solutions_k_l164_164895


namespace ellipse_parameters_sum_l164_164431

def ellipse_sum (h k a b : ℝ) : ℝ :=
  h + k + a + b

theorem ellipse_parameters_sum :
  let h := 5
  let k := -3
  let a := 7
  let b := 4
  ellipse_sum h k a b = 13 := by
  sorry

end ellipse_parameters_sum_l164_164431


namespace dad_steps_l164_164122

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l164_164122


namespace minimum_value_of_f_l164_164890

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem minimum_value_of_f : ∀ x : ℝ, 0 ≤ x → f x ≥ f 0 :=
by
  intro x hx
  unfold f
  admit

end minimum_value_of_f_l164_164890


namespace west_move_7m_l164_164651

-- Definitions and conditions
def east_move (distance : Int) : Int := distance -- Moving east
def west_move (distance : Int) : Int := -distance -- Moving west is represented as negative

-- Problem: Prove that moving west by 7m is denoted by -7m given the conditions.
theorem west_move_7m : west_move 7 = -7 :=
by
  -- Proof will be handled here normally, but it's omitted as per instruction
  sorry

end west_move_7m_l164_164651


namespace condition_necessity_not_sufficiency_l164_164993

theorem condition_necessity_not_sufficiency (a : ℝ) : 
  (2 / a < 1 → a^2 > 4) ∧ ¬(2 / a < 1 ↔ a^2 > 4) :=
by {
  sorry
}

end condition_necessity_not_sufficiency_l164_164993


namespace xyz_inequality_l164_164041

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 :=
by
  sorry

end xyz_inequality_l164_164041


namespace original_time_to_cover_distance_l164_164390

theorem original_time_to_cover_distance (S : ℝ) (T : ℝ) (D : ℝ) :
  (0.8 * S) * (T + 10 / 60) = S * T → T = 2 / 3 :=
  by sorry

end original_time_to_cover_distance_l164_164390


namespace sqrt_221_between_15_and_16_l164_164730

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end sqrt_221_between_15_and_16_l164_164730


namespace expected_value_squared_minimum_vector_norm_l164_164072

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l164_164072


namespace num_marbles_removed_l164_164536

theorem num_marbles_removed (total_marbles red_marbles : ℕ) (prob_neither_red : ℚ) 
  (h₁ : total_marbles = 84) (h₂ : red_marbles = 12) (h₃ : prob_neither_red = 36 / 49) : 
  total_marbles - red_marbles = 2 :=
by
  sorry

end num_marbles_removed_l164_164536


namespace jessa_needs_470_cupcakes_l164_164334

def total_cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_afterschool_club : ℕ) : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) + pe_class_students + (afterschool_clubs * students_per_afterschool_club)

theorem jessa_needs_470_cupcakes :
  total_cupcakes_needed 8 40 80 2 35 = 470 :=
by
  sorry

end jessa_needs_470_cupcakes_l164_164334


namespace Grant_made_total_l164_164912

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l164_164912


namespace cost_of_apples_l164_164375

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l164_164375


namespace brad_amount_l164_164268

-- Definitions for the conditions
def total_amount (j d b : ℚ) := j + d + b = 68
def josh_twice_brad (j b : ℚ) := j = 2 * b
def josh_three_fourths_doug (j d : ℚ) := j = (3 / 4) * d

-- The theorem we want to prove
theorem brad_amount : ∃ (b : ℚ), (∃ (j d : ℚ), total_amount j d b ∧ josh_twice_brad j b ∧ josh_three_fourths_doug j d) ∧ b = 12 :=
sorry

end brad_amount_l164_164268


namespace ratio_of_number_to_ten_l164_164401

theorem ratio_of_number_to_ten (n : ℕ) (h : n = 200) : n / 10 = 20 :=
by
  sorry

end ratio_of_number_to_ten_l164_164401


namespace cubic_eq_real_roots_roots_product_eq_neg_nine_l164_164174

theorem cubic_eq_real_roots :
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 + (3 / 2) * (1 - a) * x1^2 - 3 * a * x1 + b = 0 ∧
    x2^3 + (3 / 2) * (1 - a) * x2^2 - 3 * a * x2 + b = 0 ∧
    x3^3 + (3 / 2) * (1 - a) * x3^2 - 3 * a * x3 + b = 0) :=
sorry

theorem roots_product_eq_neg_nine :
  let a := 1
  let b := 1
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 - 3 * x1 + 1 = 0 ∧
    x2^3 - 3 * x2 + 1 = 0 ∧
    x3^3 - 3 * x3 + 1 = 0 ∧
    (x1^2 - 2 - x2) * (x2^2 - 2 - x3) * (x3^2 - 2 - x1) = -9) :=
sorry

end cubic_eq_real_roots_roots_product_eq_neg_nine_l164_164174


namespace herder_bulls_l164_164699

theorem herder_bulls (total_bulls : ℕ) (herder_fraction : ℚ) (claims : total_bulls = 70) (fraction_claim : herder_fraction = (2/3) * (1/3)) : herder_fraction * (total_bulls : ℚ) = 315 :=
by sorry

end herder_bulls_l164_164699


namespace integer_squared_equals_product_l164_164470

theorem integer_squared_equals_product : 
  3^8 * 3^12 * 2^5 * 2^10 = 1889568^2 :=
by
  sorry

end integer_squared_equals_product_l164_164470


namespace area_of_circle_l164_164037

theorem area_of_circle (C : ℝ) (hC : C = 36 * Real.pi) : 
  ∃ k : ℝ, (∃ r : ℝ, r = 18 ∧ k = r^2 ∧ (pi * r^2 = k * pi)) ∧ k = 324 :=
by
  sorry

end area_of_circle_l164_164037


namespace binomial_16_12_l164_164424

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l164_164424


namespace problem_statement_l164_164043

theorem problem_statement (n : ℕ) (p : ℕ) (k : ℕ) (x : ℕ → ℕ)
  (h1 : n ≥ 3)
  (h2 : ∀ i, 1 ≤ i → i ≤ n → x i < 2 * x 1)
  (h3 : ∀ i j, 1 ≤ i → i < j → j ≤ n → x i < x j)
  (hp : p.Prime)
  (hk : 0 < k)
  (hP : ∀ (P : ℕ), P = ∏ i in Finset.range n.succ \ {0}, x i → P % p^k = 0) :
  (∏ i in Finset.range n.succ \ {0}, x i) / p^k ≥ Nat.factorial n :=
sorry

end problem_statement_l164_164043


namespace odd_n_divisibility_l164_164731

theorem odd_n_divisibility (n : ℤ) : (∃ a : ℤ, n ∣ 4 * a^2 - 1) ↔ (n % 2 ≠ 0) :=
by
  sorry

end odd_n_divisibility_l164_164731


namespace average_weight_l164_164235

variable (A B C : ℝ) 

theorem average_weight (h1 : (A + B) / 2 = 48) (h2 : (B + C) / 2 = 42) (h3 : B = 51) :
  (A + B + C) / 3 = 43 := by
  sorry

end average_weight_l164_164235


namespace maximum_omega_l164_164313

noncomputable def f (omega varphi : ℝ) (x : ℝ) : ℝ :=
  Real.cos (omega * x + varphi)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem maximum_omega (omega varphi : ℝ)
    (h0 : omega > 0)
    (h1 : 0 < varphi ∧ varphi < π)
    (h2 : is_odd_function (f omega varphi))
    (h3 : is_monotonically_decreasing (f omega varphi) (-π/3) (π/6)) :
  omega ≤ 3/2 :=
sorry

end maximum_omega_l164_164313


namespace algebraic_identity_l164_164764

theorem algebraic_identity (theta : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < theta) (h2 : theta < π) (h3 : x + x⁻¹ = 2 * Real.cos theta) : 
  x^n + (x⁻¹)^n = 2 * Real.cos (n * theta) :=
by
  sorry

end algebraic_identity_l164_164764


namespace quadratic_inequality_solution_l164_164429

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 5*x + 6 > 0) ↔ (x < -3 ∨ x > -2) :=
  by
    sorry

end quadratic_inequality_solution_l164_164429


namespace cost_price_of_watch_l164_164841

variable (CP SP1 SP2 : ℝ)

theorem cost_price_of_watch (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.04 * CP)
  (h3 : SP2 = SP1 + 200) : CP = 10000 / 7 := 
by
  sorry

end cost_price_of_watch_l164_164841


namespace seating_arrangement_l164_164436

noncomputable def number_of_ways_to_seat_six_people (people : Finset ℕ) : ℕ :=
let n := people.card in
if h : n = 8 then
  let k := 6 in
  Nat.choose n (n - k) * (Nat.factorial k / k)
else
  0

theorem seating_arrangement (people : Finset ℕ) (h : people.card = 8) : 
  number_of_ways_to_seat_six_people people = 3360 := by
sorry

end seating_arrangement_l164_164436


namespace jonessa_take_home_pay_l164_164232

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l164_164232


namespace total_selling_price_l164_164080

theorem total_selling_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
    (h1 : original_price = 80) (h2 : discount_rate = 0.25) (h3 : tax_rate = 0.10) :
  let discount_amt := original_price * discount_rate
  let sale_price := original_price - discount_amt
  let tax_amt := sale_price * tax_rate
  let total_price := sale_price + tax_amt
  total_price = 66 := by
  sorry

end total_selling_price_l164_164080


namespace y_is_multiple_of_3_and_6_l164_164637

-- Define y as a sum of given numbers
def y : ℕ := 48 + 72 + 144 + 216 + 432 + 648 + 2592

theorem y_is_multiple_of_3_and_6 :
  (y % 3 = 0) ∧ (y % 6 = 0) :=
by
  -- Proof would go here, but we will end with sorry
  sorry

end y_is_multiple_of_3_and_6_l164_164637


namespace investment_worth_l164_164219

theorem investment_worth {x : ℝ} (x_pos : 0 < x) :
  ∀ (initial_investment final_value : ℝ) (years : ℕ),
  (initial_investment * 3^years = final_value) → 
  initial_investment = 1500 → final_value = 13500 → 
  8 = x → years = 2 →
  years * (112 / x) = 28 := 
by
  sorry

end investment_worth_l164_164219


namespace room_breadth_l164_164600

theorem room_breadth (length height diagonal : ℕ) (h_length : length = 12) (h_height : height = 9) (h_diagonal : diagonal = 17) : 
  ∃ breadth : ℕ, breadth = 8 :=
by
  -- Using the three-dimensional Pythagorean theorem:
  -- d² = length² + breadth² + height²
  -- 17² = 12² + b² + 9²
  -- 289 = 144 + b² + 81
  -- 289 = 225 + b²
  -- b² = 289 - 225
  -- b² = 64
  -- Taking the square root of both sides, we find:
  -- b = √64
  -- b = 8
  let b := 8
  existsi b
  -- This is a skip step, where we assert the breadth equals 8
  sorry

end room_breadth_l164_164600


namespace find_a_b_k_l164_164974

noncomputable def a (k : ℕ) : ℕ := if h : k = 9 then 243 else sorry
noncomputable def b (k : ℕ) : ℕ := if h : k = 9 then 3 else sorry

theorem find_a_b_k (a b k : ℕ) (hb : b = 3) (ha : a = 243) (hk : k = 9)
  (h1 : a * b = k^3) (h2 : a / b = k^2) (h3 : 100 ≤ a * b ∧ a * b < 1000) :
  a = 243 ∧ b = 3 ∧ k = 9 :=
by 
  sorry

end find_a_b_k_l164_164974


namespace possible_value_of_n_l164_164503

open Nat

def coefficient_is_rational (n r : ℕ) : Prop :=
  (n - r) % 2 = 0 ∧ r % 3 = 0

theorem possible_value_of_n :
  ∃ n : ℕ, n > 0 ∧ (∀ r : ℕ, r ≤ n → coefficient_is_rational n r) ↔ n = 9 :=
sorry

end possible_value_of_n_l164_164503


namespace problem_statement_l164_164326

noncomputable def a : ℚ := 18 / 11
noncomputable def c : ℚ := -30 / 11

theorem problem_statement (a b c : ℚ) (h1 : b / a = 4)
    (h2 : b = 18 - 7 * a) (h3 : c = 2 * a - 6):
    a = 18 / 11 ∧ c = -30 / 11 :=
by
  sorry

end problem_statement_l164_164326


namespace max_value_expression_l164_164737

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (M : ℝ), M = (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ∧ M = 1/24 := 
sorry

end max_value_expression_l164_164737


namespace dad_steps_l164_164129

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l164_164129


namespace prob_point_in_region_l164_164086

theorem prob_point_in_region :
  let rect_area := 18
  let intersect_area := 15 / 2
  let probability := intersect_area / rect_area
  probability = 5 / 12 :=
by
  sorry

end prob_point_in_region_l164_164086


namespace estimate_passed_students_l164_164446

-- Definitions for the given conditions
def total_papers_in_city : ℕ := 5000
def papers_selected : ℕ := 400
def papers_passed : ℕ := 360

-- The theorem stating the problem in Lean
theorem estimate_passed_students : 
    (5000:ℕ) * ((360:ℕ) / (400:ℕ)) = (4500:ℕ) :=
by
  -- Providing a trivial sorry to skip the proof.
  sorry

end estimate_passed_students_l164_164446


namespace find_number_l164_164996

theorem find_number (N : ℝ)
  (h1 : 5 / 6 * N = 5 / 16 * N + 250) :
  N = 480 :=
sorry

end find_number_l164_164996


namespace peach_trees_count_l164_164252

theorem peach_trees_count : ∀ (almond_trees: ℕ), almond_trees = 300 → 2 * almond_trees - 30 = 570 :=
by
  intros
  sorry

end peach_trees_count_l164_164252


namespace appetizer_cost_per_person_l164_164604

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end appetizer_cost_per_person_l164_164604


namespace number_of_students_l164_164738

theorem number_of_students 
    (N : ℕ) 
    (h_percentage_5 : 28 * N % 100 = 0)
    (h_percentage_4 : 35 * N % 100 = 0)
    (h_percentage_3 : 25 * N % 100 = 0)
    (h_percentage_2 : 12 * N % 100 = 0)
    (h_class_limit : N ≤ 4 * 30) 
    (h_num_classes : 4 * 30 < 120)
    : N = 100 := 
by 
  sorry

end number_of_students_l164_164738


namespace students_in_class_l164_164193

theorem students_in_class
  (S : ℕ)
  (h1 : S / 3 * 4 / 3 = 12) :
  S = 36 := 
sorry

end students_in_class_l164_164193


namespace geometric_sequence_sum_l164_164173

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℕ := λ k => 2^k) 
  (S : ℕ → ℕ := λ k => (1 - 2^k) / (1 - 2)) :
  S (n + 1) = 2 * a n - 1 :=
by
  sorry

end geometric_sequence_sum_l164_164173


namespace gerbils_left_l164_164712

theorem gerbils_left (initial count sold : ℕ) (h_initial : count = 85) (h_sold : sold = 69) : 
  count - sold = 16 := 
by 
  sorry

end gerbils_left_l164_164712


namespace probability_non_defective_l164_164192

theorem probability_non_defective (total_pens defective_pens : ℕ) (h_total : total_pens = 12) (h_defective : defective_pens = 6) :
  let non_defective_pens := total_pens - defective_pens in
  let probability_first_non_defective := non_defective_pens / total_pens in
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1) in
  probability_first_non_defective * probability_second_non_defective = 5 / 22 :=
by
  sorry

end probability_non_defective_l164_164192


namespace max_value_expression_le_380_l164_164973

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_le_380 (a b c d : ℝ)
  (ha : -9.5 ≤ a ∧ a ≤ 9.5)
  (hb : -9.5 ≤ b ∧ b ≤ 9.5)
  (hc : -9.5 ≤ c ∧ c ≤ 9.5)
  (hd : -9.5 ≤ d ∧ d ≤ 9.5) :
  max_value_expression a b c d ≤ 380 :=
sorry

end max_value_expression_le_380_l164_164973


namespace solve_equation_l164_164809

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
sorry

end solve_equation_l164_164809


namespace find_x_2y_3z_l164_164902

theorem find_x_2y_3z (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (h1 : x ≤ y) (h2 : y ≤ z) (h3 : x + y + z = 12) (h4 : x * y + y * z + z * x = 41) :
  x + 2 * y + 3 * z = 29 :=
by
  sorry

end find_x_2y_3z_l164_164902


namespace complete_the_square_result_l164_164053

-- Define the equation
def initial_eq (x : ℝ) : Prop := x^2 + 4 * x + 3 = 0

-- State the theorem based on the condition and required to prove the question equals the answer
theorem complete_the_square_result (x : ℝ) : initial_eq x → (x + 2) ^ 2 = 1 := 
by
  intro h
  -- Proof is to be skipped
  sorry

end complete_the_square_result_l164_164053


namespace least_positive_integer_to_multiple_of_4_l164_164991

theorem least_positive_integer_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ ((563 + n) % 4 = 0) ∧ n = 1 := 
by
  sorry

end least_positive_integer_to_multiple_of_4_l164_164991


namespace distance_between_foci_of_hyperbola_l164_164665

-- Define the asymptotes as lines
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 7

-- Define the condition that the hyperbola passes through the point (4, 5)
def passes_through (x y : ℝ) : Prop := (x, y) = (4, 5)

-- Statement to prove
theorem distance_between_foci_of_hyperbola : 
  (asymptote1 4 = 5) ∧ (asymptote2 4 = 5) ∧ passes_through 4 5 → 
  (∀ a b c : ℝ, a^2 = 9 ∧ b^2 = 9/4 ∧ c^2 = a^2 + b^2 → 2 * c = 3 * Real.sqrt 5) := 
by
  intro h
  sorry

end distance_between_foci_of_hyperbola_l164_164665


namespace Robert_books_count_l164_164960

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l164_164960


namespace fruit_shop_problem_l164_164628

variable (x y z : ℝ)

theorem fruit_shop_problem
  (h1 : x + 4 * y + 2 * z = 27.2)
  (h2 : 2 * x + 6 * y + 2 * z = 32.4) :
  x + 2 * y = 5.2 :=
by
  sorry

end fruit_shop_problem_l164_164628


namespace polynomial_form_l164_164732

def is_even_poly (P : ℝ → ℝ) : Prop := 
  ∀ x, P x = P (-x)

theorem polynomial_form (P : ℝ → ℝ) (hP : ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) : 
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x ^ 4 + b * x ^ 2 := 
  sorry

end polynomial_form_l164_164732


namespace at_least_one_even_difference_l164_164214

-- Statement of the problem in Lean 4
theorem at_least_one_even_difference 
  (a b : Fin (2 * n + 1) → ℤ) 
  (hperm : ∃ σ : Equiv.Perm (Fin (2 * n + 1)), ∀ k, a k = (b ∘ σ) k) : 
  ∃ k, (a k - b k) % 2 = 0 := 
sorry

end at_least_one_even_difference_l164_164214


namespace max_piece_length_total_pieces_l164_164822

-- Definitions based on the problem's conditions
def length1 : ℕ := 42
def length2 : ℕ := 63
def gcd_length : ℕ := Nat.gcd length1 length2

-- Theorem statements based on the realized correct answers
theorem max_piece_length (h1 : length1 = 42) (h2 : length2 = 63) :
  gcd_length = 21 := by
  sorry

theorem total_pieces (h1 : length1 = 42) (h2 : length2 = 63) :
  (length1 / gcd_length) + (length2 / gcd_length) = 5 := by
  sorry

end max_piece_length_total_pieces_l164_164822


namespace digits_sum_is_23_l164_164204

/-
Juan chooses a five-digit positive integer.
Maria erases the ones digit and gets a four-digit number.
The sum of this four-digit number and the original five-digit number is 52,713.
What can the sum of the five digits of the original number be?
-/

theorem digits_sum_is_23 (x y : ℕ) (h1 : 1000 ≤ x) (h2 : x ≤ 9999) (h3 : y ≤ 9) (h4 : 11 * x + y = 52713) :
  (x / 1000) + (x / 100 % 10) + (x / 10 % 10) + (x % 10) + y = 23 :=
by {
  sorry -- Proof goes here.
}

end digits_sum_is_23_l164_164204


namespace files_remaining_l164_164998

theorem files_remaining 
(h_music_files : ℕ := 16) 
(h_video_files : ℕ := 48) 
(h_files_deleted : ℕ := 30) :
(h_music_files + h_video_files - h_files_deleted = 34) := 
by sorry

end files_remaining_l164_164998


namespace james_final_payment_l164_164200

variable (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ)

def final_cost (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ) : ℚ :=
let bed_cost := bed_frame_cost * bed_cost_multiplier in
let total_cost := bed_cost + bed_frame_cost in
let discount := total_cost * discount_rate in
total_cost - discount

theorem james_final_payment : final_cost 75 10 (20 / 100) = 660 := by
  unfold final_cost
  -- Step: Compute bed cost
  have bed_cost : ℕ := 75 * 10 by norm_num
  -- Step: Compute total cost
  have total_cost : ℕ := bed_cost + 75 by norm_num
  -- Step: Compute discount
  have discount : ℚ := total_cost * (20 / 100) by norm_num
  -- Step: Compute final cost
  have final_payment : ℚ := total_cost - discount by norm_num
  -- Assertion
  exact final_payment

#eval james_final_payment

end james_final_payment_l164_164200


namespace min_value_N_l164_164785

theorem min_value_N (a b c d e f : ℤ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < e) (h₅ : 0 < f)
  (h_sum : a + b + c + d + e + f = 4020) :
  ∃ N : ℤ, N = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ N = 805 :=
by
  sorry

end min_value_N_l164_164785


namespace find_f_2000_l164_164040

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ (x y : ℝ), f (x + y) = f (x * y)
axiom f_property2 : f (-1/2) = -1/2

theorem find_f_2000 : f 2000 = -1/2 := 
sorry

end find_f_2000_l164_164040


namespace paving_cost_l164_164241

theorem paving_cost (length width rate : ℝ) (h_length : length = 8) (h_width : width = 4.75) (h_rate : rate = 900) :
  length * width * rate = 34200 :=
by
  rw [h_length, h_width, h_rate]
  norm_num

end paving_cost_l164_164241


namespace part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l164_164229

noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def W (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

theorem part_1_relationship (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 40) :
  W x = -10 * x^2 + 500 * x - 4000 := by
  sorry

theorem part_2_solution (x : ℝ) (h₀ : W x = 1250) :
  x = 15 ∨ x = 35 := by
  sorry

theorem part_2_preferred (x : ℝ) (h₀ : W x = 1250) (h₁ : y 15 ≥ y 35) :
  x = 15 := by
  sorry

theorem part_3_max_W (x : ℝ) (h₀ : 28 ≤ x) (h₁ : x ≤ 35) :
  W x ≤ 2160 := by
  sorry

theorem part_3_max_at_28 :
  W 28 = 2160 := by
  sorry

end part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l164_164229


namespace distribute_items_among_people_l164_164197

theorem distribute_items_among_people :
  (Nat.choose (10 + 3 - 1) 3) = 220 := 
by sorry

end distribute_items_among_people_l164_164197


namespace average_speed_lila_l164_164791

-- Definitions
def distance1 : ℝ := 50 -- miles
def speed1 : ℝ := 20 -- miles per hour
def distance2 : ℝ := 20 -- miles
def speed2 : ℝ := 40 -- miles per hour
def break_time : ℝ := 0.5 -- hours

-- Question to prove: Lila's average speed for the entire ride is 20 miles per hour
theorem average_speed_lila (d1 d2 s1 s2 bt : ℝ) 
  (h1 : d1 = distance1) (h2 : s1 = speed1) (h3 : d2 = distance2) (h4 : s2 = speed2) (h5 : bt = break_time) :
  (d1 + d2) / (d1 / s1 + d2 / s2 + bt) = 20 :=
by
  sorry

end average_speed_lila_l164_164791


namespace range_of_y_l164_164450

theorem range_of_y :
  ∀ (y x : ℝ), x = 4 - y → (-2 ≤ x ∧ x ≤ -1) → (5 ≤ y ∧ y ≤ 6) :=
by
  intros y x h1 h2
  sorry

end range_of_y_l164_164450


namespace final_laptop_price_l164_164859

theorem final_laptop_price :
  let original_price := 1000.00
  let first_discounted_price := original_price * (1 - 0.10)
  let second_discounted_price := first_discounted_price * (1 - 0.25)
  let recycling_fee := second_discounted_price * 0.05
  let final_price := second_discounted_price + recycling_fee
  final_price = 708.75 :=
by
  sorry

end final_laptop_price_l164_164859


namespace max_soccer_balls_l164_164812

theorem max_soccer_balls (bought_balls : ℕ) (total_cost : ℕ) (available_money : ℕ) (unit_cost : ℕ)
    (h1 : bought_balls = 6) (h2 : total_cost = 168) (h3 : available_money = 500)
    (h4 : unit_cost = total_cost / bought_balls) :
    (available_money / unit_cost) = 17 := 
by
  sorry

end max_soccer_balls_l164_164812


namespace boat_upstream_time_l164_164778

theorem boat_upstream_time (v t : ℝ) (d c : ℝ) 
  (h1 : d = 24) (h2 : c = 1) (h3 : 4 * (v + c) = d) 
  (h4 : d / (v - c) = t) : t = 6 :=
by
  sorry

end boat_upstream_time_l164_164778


namespace integral_evaluation_l164_164885

noncomputable def integral_value : Real :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - (x - 1)^2) - x)

theorem integral_evaluation :
  integral_value = (Real.pi / 4) - 1 / 2 :=
by
  sorry

end integral_evaluation_l164_164885


namespace minimum_z_value_l164_164306

theorem minimum_z_value (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : x^2 + y^2 ≥ 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_z_value_l164_164306


namespace pell_eq_unique_fund_sol_l164_164217

theorem pell_eq_unique_fund_sol (x y x_0 y_0 : ℕ) 
  (h1 : x_0^2 - 2003 * y_0^2 = 1) 
  (h2 : ∀ x y, x > 0 ∧ y > 0 → x^2 - 2003 * y^2 = 1 → ∃ n : ℕ, x + Real.sqrt 2003 * y = (x_0 + Real.sqrt 2003 * y_0)^n)
  (hx_pos : x > 0) 
  (hy_pos : y > 0)
  (h_sol : x^2 - 2003 * y^2 = 1) 
  (hprime : ∀ p : ℕ, Prime p → p ∣ x → p ∣ x_0)
  : x = x_0 ∧ y = y_0 :=
sorry

end pell_eq_unique_fund_sol_l164_164217


namespace expansion_l164_164597

variable (x : ℝ)

noncomputable def expr : ℝ := (3 / 4) * (8 / (x^2) + 5 * x - 6)

theorem expansion :
  expr x = (6 / (x^2)) + (15 * x / 4) - 4.5 :=
by
  sorry

end expansion_l164_164597


namespace determine_xy_l164_164671

noncomputable section

open Real

def op_defined (ab xy : ℝ × ℝ) : ℝ × ℝ :=
  (ab.1 * xy.1 + ab.2 * xy.2, ab.1 * xy.2 + ab.2 * xy.1)

theorem determine_xy (x y : ℝ) :
  (∀ (a b : ℝ), op_defined (a, b) (x, y) = (a, b)) → (x = 1 ∧ y = 0) :=
by
  sorry

end determine_xy_l164_164671


namespace find_parabola_constant_l164_164038

theorem find_parabola_constant (a b c : ℝ) (h_vertex : ∀ y, (4:ℝ) = -5 / 4 * y * y + 5 / 2 * y + c)
  (h_point : (-1:ℝ) = -5 / 4 * (3:ℝ) ^ 2 + 5 / 2 * (3:ℝ) + c ) :
  c = 11 / 4 :=
sorry

end find_parabola_constant_l164_164038


namespace inequality_S_sum_l164_164016

open Finset BigOperators

noncomputable def T (n : ℕ) : ℝ := (n * (n + 1)) / 2

noncomputable def S (n : ℕ) : ℝ := ∑ k in range (n + 1), 1 / T k

theorem inequality_S_sum :
  (∑ k in range 1996, 1 / S k) > 1001 :=
sorry

end inequality_S_sum_l164_164016


namespace binom_16_12_eq_1820_l164_164422

theorem binom_16_12_eq_1820 : Nat.choose 16 12 = 1820 :=
by
  sorry

end binom_16_12_eq_1820_l164_164422


namespace find_k_l164_164308

theorem find_k (x y k : ℝ) (h1 : x = 1) (h2 : y = 4) (h3 : k * x + y = 3) : k = -1 :=
by
  sorry

end find_k_l164_164308


namespace number_of_n_with_odd_tens_digit_in_square_l164_164444

def ends_in_3_or_7 (n : ℕ) : Prop :=
  n % 10 = 3 ∨ n % 10 = 7

def tens_digit_odd (n : ℕ) : Prop :=
  ((n * n / 10) % 10) % 2 = 1

theorem number_of_n_with_odd_tens_digit_in_square :
  ∀ n ∈ {n : ℕ | n ≤ 50 ∧ ends_in_3_or_7 n}, ¬tens_digit_odd n :=
by 
  sorry

end number_of_n_with_odd_tens_digit_in_square_l164_164444


namespace original_card_count_l164_164709

theorem original_card_count
  (r b : ℕ)
  (initial_prob_red : (r : ℚ) / (r + b) = 2 / 5)
  (prob_red_after_adding_black : (r : ℚ) / (r + (b + 6)) = 1 / 3) :
  r + b = 30 := sorry

end original_card_count_l164_164709


namespace dad_steps_l164_164134

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l164_164134


namespace count_success_permutations_l164_164883

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l164_164883


namespace max_students_distribute_eq_pens_pencils_l164_164698

theorem max_students_distribute_eq_pens_pencils (n_pens n_pencils n : ℕ) (h_pens : n_pens = 890) (h_pencils : n_pencils = 630) :
  (∀ k : ℕ, k > n → (n_pens % k ≠ 0 ∨ n_pencils % k ≠ 0)) → (n = Nat.gcd n_pens n_pencils) := by
  sorry

end max_students_distribute_eq_pens_pencils_l164_164698


namespace exponential_function_fixed_point_l164_164239

theorem exponential_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end exponential_function_fixed_point_l164_164239


namespace find_added_number_l164_164711

def original_number : ℕ := 5
def doubled : ℕ := 2 * original_number
def resultant (added : ℕ) : ℕ := 3 * (doubled + added)
def final_result : ℕ := 57

theorem find_added_number (added : ℕ) (h : resultant added = final_result) : added = 9 :=
sorry

end find_added_number_l164_164711


namespace dad_steps_l164_164119

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l164_164119


namespace solve_for_m_l164_164476

noncomputable def has_positive_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (m / (x - 3) - 1 / (3 - x) = 2)

theorem solve_for_m (m : ℝ) : has_positive_root m → m = -1 :=
sorry

end solve_for_m_l164_164476


namespace area_under_pressure_l164_164475

theorem area_under_pressure (F : ℝ) (S : ℝ) (p : ℝ) (hF : F = 100) (hp : p > 1000) (hpressure : p = F / S) :
  S < 0.1 :=
by
  sorry

end area_under_pressure_l164_164475


namespace find_x_l164_164015

-- Let \( x \) be a real number such that 
-- \( x = 2 \left( \frac{1}{x} \cdot (-x) \right) - 5 \).
-- Prove \( x = -7 \).

theorem find_x (x : ℝ) (h : x = 2 * (1 / x * (-x)) - 5) : x = -7 :=
by
  sorry

end find_x_l164_164015


namespace min_value_b_over_a_l164_164752

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (Real.exp 1 - a) * x - b

theorem min_value_b_over_a 
  (a b : ℝ)
  (h_cond : ∀ x > 0, f x a b ≤ 0)
  (h_b : b = -1 - Real.log (a - Real.exp 1)) 
  (h_a_gt_e : a > Real.exp 1) :
  ∃ (x : ℝ), x = 2 * Real.exp 1 ∧ (b / a) = - (1 / Real.exp 1) := 
sorry

end min_value_b_over_a_l164_164752


namespace zeros_of_geometric_sequence_quadratic_l164_164760

theorem zeros_of_geometric_sequence_quadratic (a b c : ℝ) (h_geometric : b^2 = a * c) (h_pos : a * c > 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := by
sorry

end zeros_of_geometric_sequence_quadratic_l164_164760


namespace prob_c_not_adjacent_to_a_or_b_l164_164047

-- Definitions for the conditions
def num_students : ℕ := 7
def a_and_b_together : Prop := true
def c_on_edge : Prop := true

-- Main theorem: probability c not adjacent to a or b under given conditions
theorem prob_c_not_adjacent_to_a_or_b
  (h1 : a_and_b_together)
  (h2 : c_on_edge) :
  ∃ (p : ℚ), p = 0.8 := by
  sorry

end prob_c_not_adjacent_to_a_or_b_l164_164047


namespace nearest_integer_to_expression_correct_l164_164826

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l164_164826


namespace hyperbola_condition_l164_164176

theorem hyperbola_condition (a : ℝ) (h : a > 0)
  (e : ℝ) (h_e : e = Real.sqrt (1 + 4 / (a^2))) :
  (e > Real.sqrt 2) ↔ (0 < a ∧ a < 1) := 
sorry

end hyperbola_condition_l164_164176


namespace parabola_translation_vertex_l164_164379

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end parabola_translation_vertex_l164_164379


namespace no_solution_15x_29y_43z_t2_l164_164843

theorem no_solution_15x_29y_43z_t2 (x y z t : ℕ) : ¬ (15 ^ x + 29 ^ y + 43 ^ z = t ^ 2) :=
by {
  -- We'll insert the necessary conditions for the proof here
  sorry -- proof goes here
}

end no_solution_15x_29y_43z_t2_l164_164843


namespace find_c_l164_164889

noncomputable def f (c x : ℝ) : ℝ :=
  c * x^3 + 17 * x^2 - 4 * c * x + 45

theorem find_c (h : f c (-5) = 0) : c = 94 / 21 :=
by sorry

end find_c_l164_164889


namespace ratio_first_term_to_common_difference_l164_164835

theorem ratio_first_term_to_common_difference
  (a d : ℝ)
  (S_n : ℕ → ℝ)
  (hS_n : ∀ n, S_n n = (n / 2) * (2 * a + (n - 1) * d))
  (h : S_n 15 = 3 * S_n 10) :
  a / d = -2 :=
by
  sorry

end ratio_first_term_to_common_difference_l164_164835


namespace simplify_trig_expression_l164_164660

theorem simplify_trig_expression : 
  ∀ (deg : ℝ), deg = 1 :=
by
  let sin := Real.sin
  let cos := Real.cos
  have h : ∀ (x : ℝ), cos (-x) = cos x := by sorry
  have h10_20_eq := sin 10 + sin 20
  have h_cos10_20_eq := cos 10 + cos 20
  have h1 := h10_20_eq
  have h2 := h_cos10_20_eq
  have main_eq := (sin 10 + sin 20) / (cos 10 + cos 20)
  have main_term := (2 * sin (15) * cos (-5)) / (2 * cos (15) * cos (-5))
  have simpl := main_term = (sin 15 / cos 15)
  have final := simpl = Real.tan (15)
  show final = Real.tan (15)

end simplify_trig_expression_l164_164660


namespace largest_fraction_l164_164387

theorem largest_fraction :
  let fA : ℚ := 2 / 5
  let fB : ℚ := 3 / 7
  let fC : ℚ := 4 / 9
  let fD : ℚ := 7 / 15
  let fE : ℚ := 9 / 20
  let fF : ℚ := 11 / 25
  in fD > fA ∧ fD > fB ∧ fD > fC ∧ fD > fE ∧ fD > fF :=
by
  -- Definitions of each fraction
  let fA : ℚ := 2 / 5
  let fB : ℚ := 3 / 7
  let fC : ℚ := 4 / 9
  let fD : ℚ := 7 / 15
  let fE : ℚ := 9 / 20
  let fF : ℚ := 11 / 25
  -- Proof requirement
  sorry

end largest_fraction_l164_164387


namespace simplify_condition_l164_164228

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) - Real.sqrt (-1 - x)

theorem simplify_condition (x : ℝ) (h1 : 1 + x ≥ 0) (h2 : -1 - x ≥ 0) : simplify_expression x = 0 :=
by
  rw [simplify_expression]
  sorry

end simplify_condition_l164_164228


namespace g_sqrt_45_l164_164640

noncomputable def g (x : ℝ) : ℝ :=
if x % 1 = 0 then 7 * x + 6 else ⌊x⌋ + 7

theorem g_sqrt_45 : g (Real.sqrt 45) = 13 := by
  sorry

end g_sqrt_45_l164_164640


namespace range_of_a_l164_164187

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → -3 * x^2 + a ≤ 0) ↔ a ≤ 3 := by
  sorry

end range_of_a_l164_164187


namespace third_car_year_l164_164679

theorem third_car_year (y1 y2 y3 : ℕ) (h1 : y1 = 1970) (h2 : y2 = y1 + 10) (h3 : y3 = y2 + 20) : y3 = 2000 :=
by
  sorry

end third_car_year_l164_164679


namespace solution_is_111_l164_164502

-- Define the system of equations
def system_of_equations (x y z : ℝ) :=
  (x^2 + 7 * y + 2 = 2 * z + 4 * Real.sqrt (7 * x - 3)) ∧
  (y^2 + 7 * z + 2 = 2 * x + 4 * Real.sqrt (7 * y - 3)) ∧
  (z^2 + 7 * x + 2 = 2 * y + 4 * Real.sqrt (7 * z - 3))

-- Prove that x = 1, y = 1, z = 1 is a solution to the system of equations
theorem solution_is_111 : system_of_equations 1 1 1 :=
by
  sorry

end solution_is_111_l164_164502


namespace multiple_of_9_is_multiple_of_3_l164_164999

theorem multiple_of_9_is_multiple_of_3 (n : ℤ) (h : ∃ k : ℤ, n = 9 * k) : ∃ m : ℤ, n = 3 * m :=
by
  sorry

end multiple_of_9_is_multiple_of_3_l164_164999


namespace solve_dance_circles_problem_l164_164195

open Finset

-- Definitions based on the conditions provided in the problem
def children := (univ : Finset (Fin 5)) -- There's a set of 5 children, each distinct

noncomputable def numWaysToDivideIntoDanceCircles : ℕ :=
  let S := fun (n k : ℕ) => StirlingSecondKind.partition n k in
  let partitions := S 5 2 in
  let circles_permutations := fun k => factorial (k - 1) * factorial (5 - k - 1) in
  let totalConfigurations := 
    (univ.ssubsets.filter (fun s => s.card ≠ 0)).sum (λ s, 
      let k := s.card in
      (choose 5 k) * circles_permutations k
    ) in
  totalConfigurations / 2

-- Theorem to state the correctness of the counted ways as 50
theorem solve_dance_circles_problem :
  numWaysToDivideIntoDanceCircles = 50 := by
  sorry

end solve_dance_circles_problem_l164_164195


namespace geometric_series_expr_l164_164413

theorem geometric_series_expr :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * 
  (1 + 4 * (1 + 4)))))))))) + 100 = 5592504 := 
sorry

end geometric_series_expr_l164_164413


namespace stratified_sampling_yogurt_adult_milk_powder_sum_l164_164194

theorem stratified_sampling_yogurt_adult_milk_powder_sum :
  let liquid_milk_brands := 40
  let yogurt_brands := 10
  let infant_formula_brands := 30
  let adult_milk_powder_brands := 20
  let total_brands := liquid_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sample_size := 20
  let yogurt_sample := sample_size * yogurt_brands / total_brands
  let adult_milk_powder_sample := sample_size * adult_milk_powder_brands / total_brands
  yogurt_sample + adult_milk_powder_sample = 6 :=
by
  sorry

end stratified_sampling_yogurt_adult_milk_powder_sum_l164_164194


namespace work_completion_time_l164_164255

-- Define work rates for workers p, q, and r
def work_rate_p : ℚ := 1 / 12
def work_rate_q : ℚ := 1 / 9
def work_rate_r : ℚ := 1 / 18

-- Define time they work in respective phases
def time_p : ℚ := 2
def time_pq : ℚ := 3

-- Define the total time taken to complete the work
def total_time : ℚ := 6

-- Prove that the total time to complete the work is 6 days
theorem work_completion_time :
  (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq + (1 - (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq)) / (work_rate_p + work_rate_q + work_rate_r)) = total_time :=
by sorry

end work_completion_time_l164_164255


namespace success_permutations_correct_l164_164879

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l164_164879


namespace next_ten_winners_each_receive_160_l164_164997

def total_prize_money : ℕ := 2400

def first_winner_amount : ℕ := total_prize_money / 3

def remaining_amount : ℕ := total_prize_money - first_winner_amount

def each_of_ten_winners_receive : ℕ := remaining_amount / 10

theorem next_ten_winners_each_receive_160 : each_of_ten_winners_receive = 160 := by
  sorry

end next_ten_winners_each_receive_160_l164_164997


namespace largest_divisor_36_l164_164448

open Nat

noncomputable def f (n : ℕ) : ℕ := (2 * n + 7) * 3^(n + 9)

theorem largest_divisor_36 (m : ℕ) : (∀ n : ℕ, 0 < n → m ∣ f n) ↔ m = 36 :=
by
  -- Left part: assume hypothesis and prove m = 36 (proof omitted)
  -- Right part: assume m = 36 and prove hypothesis (proof omitted)
  sorry

end largest_divisor_36_l164_164448


namespace maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l164_164754

def f (x : ℝ) (t : ℝ) : ℝ := abs (2 * x - 1) - abs (t * x + 3)

theorem maximum_value_when_t_is_2 :
  ∃ x : ℝ, (f x 2) ≤ 4 ∧ ∀ y : ℝ, (f y 2) ≤ (f x 2) := sorry

theorem solve_for_t_when_maximum_value_is_2 :
  ∃ t : ℝ, t > 0 ∧ (∀ x : ℝ, (f x t) ≤ 2 ∧ (∃ y : ℝ, (f y t) = 2)) → t = 6 := sorry

end maximum_value_when_t_is_2_solve_for_t_when_maximum_value_is_2_l164_164754


namespace cathy_total_money_l164_164099

theorem cathy_total_money : 
  let initial := 12 
  let dad_contribution := 25 
  let mom_contribution := 2 * dad_contribution 
  let total_money := initial + dad_contribution + mom_contribution 
  in total_money = 87 :=
by
  let initial := 12
  let dad_contribution := 25
  let mom_contribution := 2 * dad_contribution
  let total_money := initial + dad_contribution + mom_contribution
  show total_money = 87
  sorry

end cathy_total_money_l164_164099


namespace c_horses_months_l164_164842

theorem c_horses_months (cost_total Rs_a Rs_b num_horses_a num_months_a num_horses_b num_months_b num_horses_c amount_paid_b : ℕ) (x : ℕ) 
  (h1 : cost_total = 841) 
  (h2 : Rs_a = 12 * 8)
  (h3 : Rs_b = 16 * 9)
  (h4 : amount_paid_b = 348)
  (h5 : 96 * (amount_paid_b / Rs_b) + (18 * x) * (amount_paid_b / Rs_b) = cost_total - amount_paid_b) :
  x = 11 :=
sorry

end c_horses_months_l164_164842


namespace largest_number_l164_164544

theorem largest_number (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29) :
  d = 21 := 
sorry

end largest_number_l164_164544


namespace max_ab_l164_164165

theorem max_ab (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 3 ≤ a + b ∧ a + b ≤ 4) : ab ≤ 15 / 4 :=
sorry

end max_ab_l164_164165


namespace percentage_y_more_than_z_l164_164772

theorem percentage_y_more_than_z :
  ∀ (P y x k : ℕ),
    P = 200 →
    740 = x + y + P →
    x = (5 / 4) * y →
    y = P * (1 + k / 100) →
    k = 20 :=
by
  sorry

end percentage_y_more_than_z_l164_164772


namespace dad_steps_l164_164148

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l164_164148


namespace smallest_n_mod_equiv_l164_164522

open Int

theorem smallest_n_mod_equiv : ∃ n : ℕ, 0 < n ∧ 29 * n ≡ 5678 [MOD 11] ∧ ∀ m : ℕ, (0 < m ∧ 29 * m ≡ 5678 [MOD 11]) → n ≤ m := by
  use 9
  split
  · exact Nat.succ_pos' _
  split
  · calc 
      29 * 9 ≡ 261 [MOD 11] := by norm_num
      5678 ≡ 2 [MOD 11] := by norm_num
      _ ≡ 2 [MOD 11] := by ring
  · intro m hm
    obtain ⟨_, hm⟩ := hm
    have : ∃ k : ℤ, 29 * m = 5678 + 11 * k := by 
      cases hm with x hx
      use x
      norm_cast at hx
    sorry

end smallest_n_mod_equiv_l164_164522


namespace infinite_series_value_l164_164416

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l164_164416


namespace greatest_groups_of_stuffed_animals_l164_164428

def stuffed_animals_grouping : Prop :=
  let cats := 26
  let dogs := 14
  let bears := 18
  let giraffes := 22
  gcd (gcd (gcd cats dogs) bears) giraffes = 2

theorem greatest_groups_of_stuffed_animals : stuffed_animals_grouping :=
by sorry

end greatest_groups_of_stuffed_animals_l164_164428


namespace eight_sharp_two_equals_six_thousand_l164_164874

def new_operation (a b : ℕ) : ℕ :=
  (a + b) ^ 3 * (a - b)

theorem eight_sharp_two_equals_six_thousand : new_operation 8 2 = 6000 := 
  by
    sorry

end eight_sharp_two_equals_six_thousand_l164_164874


namespace quadratic_intersect_x_axis_l164_164178

theorem quadratic_intersect_x_axis (a : ℝ) : (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) :=
by
  sorry

end quadratic_intersect_x_axis_l164_164178


namespace triangle_area_proof_l164_164518

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end triangle_area_proof_l164_164518


namespace ones_digit_542_mul_3_is_6_l164_164042

/--
Given that the ones (units) digit of 542 is 2, prove that the ones digit of 542 multiplied by 3 is 6.
-/
theorem ones_digit_542_mul_3_is_6 (h: ∃ n : ℕ, 542 = 10 * n + 2) : (542 * 3) % 10 = 6 := 
by
  sorry

end ones_digit_542_mul_3_is_6_l164_164042


namespace average_weight_of_remaining_boys_l164_164046

theorem average_weight_of_remaining_boys (avg_weight_16: ℝ) (avg_weight_total: ℝ) (weight_16: ℝ) (total_boys: ℝ) (avg_weight_8: ℝ) : 
  (avg_weight_16 = 50.25) → (avg_weight_total = 48.55) → (weight_16 = 16 * avg_weight_16) → (total_boys = 24) → 
  (total_weight = total_boys * avg_weight_total) → (weight_16 + 8 * avg_weight_8 = total_weight) → avg_weight_8 = 45.15 :=
by
  intros h_avg_weight_16 h_avg_weight_total h_weight_16 h_total_boys h_total_weight h_equation
  sorry

end average_weight_of_remaining_boys_l164_164046


namespace problem1_l164_164271

theorem problem1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := sorry

end problem1_l164_164271


namespace calculator_change_problem_l164_164084

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l164_164084


namespace tangent_line_through_point_l164_164978

noncomputable def y (x : ℝ) : ℝ := x^3 - 2 * x
noncomputable def tangent_slope (x0 : ℝ) : ℝ := 3 * x0^2 - 2

theorem tangent_line_through_point : 
  ∃ (x0 : ℝ) (y0 : ℝ), y0 = y x0 ∧ 
  (let k := (y0 + 1) / (x0 - 1) in
   (k = tangent_slope x0) ∧ 
   ((x0 = 1 ∧ k = 1 ∧ (x - y = 2)) ∨ 
    (x0 = -1/2 ∧ k = 5/4 ∧ (5 * x + 4 * y = 1)))) :=
by
  sorry

end tangent_line_through_point_l164_164978


namespace usb_drive_total_capacity_l164_164797

-- Define the conditions as α = total capacity, β = busy space (50%), γ = available space (50%)
variable (α : ℕ) -- Total capacity of the USB drive in gigabytes
variable (β γ : ℕ) -- Busy space and available space in gigabytes
variable (h1 : β = α / 2) -- 50% of total capacity is busy
variable (h2 : γ = 8)  -- 8 gigabytes are still available

-- Define the problem as a theorem that these conditions imply the total capacity
theorem usb_drive_total_capacity (h : γ = α / 2) : α = 16 :=
by
  -- defer the proof
  sorry

end usb_drive_total_capacity_l164_164797


namespace input_language_is_input_l164_164525

def is_print_statement (statement : String) : Prop := 
  statement = "PRINT"

def is_input_statement (statement : String) : Prop := 
  statement = "INPUT"

def is_conditional_statement (statement : String) : Prop := 
  statement = "IF"

theorem input_language_is_input :
  is_input_statement "INPUT" := 
by
  -- Here we need to show "INPUT" is an input statement
  sorry

end input_language_is_input_l164_164525


namespace dad_steps_l164_164128

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l164_164128


namespace cos_diff_expression_eq_half_l164_164433

theorem cos_diff_expression_eq_half :
  (Real.cos (Real.pi * 24 / 180) * Real.cos (Real.pi * 36 / 180) -
   Real.cos (Real.pi * 66 / 180) * Real.cos (Real.pi * 54 / 180)) = 1 / 2 := by
sorry

end cos_diff_expression_eq_half_l164_164433


namespace west_movement_is_negative_seven_l164_164648

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end west_movement_is_negative_seven_l164_164648


namespace cat_catches_mouse_l164_164845

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

end cat_catches_mouse_l164_164845


namespace geometric_sequence_S4_l164_164485

/-
In the geometric sequence {a_n}, S_2 = 7, S_6 = 91. Prove that S_4 = 28.
-/

theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 7) 
  (h_S6 : S 6 = 91) :
  S 4 = 28 := 
sorry

end geometric_sequence_S4_l164_164485


namespace count_heads_at_night_l164_164990

variables (J T D : ℕ)

theorem count_heads_at_night (h1 : 2 * J + 4 * T + 2 * D = 56) : J + T + D = 14 :=
by
  -- Skip the proof
  sorry

end count_heads_at_night_l164_164990


namespace change_received_l164_164083

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l164_164083


namespace problem1_problem2_l164_164018

namespace MathProof

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem problem1 (m : ℝ) :
  (∀ x, 0 < x → f x m > 0) → -2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5 :=
sorry

theorem problem2 (m : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ f x m = 0) → -2 < m ∧ m < 0 :=
sorry

end MathProof

end problem1_problem2_l164_164018


namespace log_sum_property_l164_164312

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def f_inv (a : ℝ) (y : ℝ) := a ^ y

theorem log_sum_property (a : ℝ) (h1 : f_inv a 2 = 9) (h2 : f a 9 = 2) : f a 9 + f a 6 = 1 :=
by
  sorry

end log_sum_property_l164_164312


namespace min_even_number_for_2015_moves_l164_164952

theorem min_even_number_for_2015_moves (N : ℕ) (hN : N ≥ 2) :
  ∃ k : ℕ, N = 2 ^ k ∧ 2 ^ k ≥ 2 ∧ k ≥ 4030 :=
sorry

end min_even_number_for_2015_moves_l164_164952


namespace monthly_income_of_B_l164_164265

variable (x y : ℝ)

-- Monthly incomes in the ratio 5:6
axiom income_ratio (A_income B_income : ℝ) : A_income = 5 * x ∧ B_income = 6 * x

-- Monthly expenditures in the ratio 3:4
axiom expenditure_ratio (A_expenditure B_expenditure : ℝ) : A_expenditure = 3 * y ∧ B_expenditure = 4 * y

-- Savings of A and B
axiom savings_A (A_income A_expenditure : ℝ) : 1800 = A_income - A_expenditure
axiom savings_B (B_income B_expenditure : ℝ) : 1600 = B_income - B_expenditure

-- The theorem to prove
theorem monthly_income_of_B (B_income : ℝ) (x y : ℝ) 
  (h1 : A_income = 5 * x)
  (h2 : B_income = 6 * x)
  (h3: A_expenditure = 3 * y)
  (h4: B_expenditure = 4 * y)
  (h5 : 1800 = 5 * x - 3 * y)
  (h6 : 1600 = 6 * x - 4 * y)
  : B_income = 7200 := by
  sorry

end monthly_income_of_B_l164_164265


namespace second_coloring_book_pictures_l164_164657

-- Let P1 be the number of pictures in the first coloring book.
def P1 := 23

-- Let P2 be the number of pictures in the second coloring book.
variable (P2 : Nat)

-- Let colored_pics be the number of pictures Rachel colored.
def colored_pics := 44

-- Let remaining_pics be the number of pictures Rachel still has to color.
def remaining_pics := 11

-- Total number of pictures in both coloring books.
def total_pics := colored_pics + remaining_pics

theorem second_coloring_book_pictures :
  P2 = total_pics - P1 :=
by
  -- We need to prove that P2 = 32.
  sorry

end second_coloring_book_pictures_l164_164657


namespace expression_never_prime_l164_164728

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (n : ℕ) (h : is_prime n) : ¬is_prime (n^2 + 75) :=
sorry

end expression_never_prime_l164_164728


namespace steves_earning_l164_164810

variable (pounds_picked : ℕ → ℕ) -- pounds picked on day i: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday

def payment_per_pound : ℕ := 2

def total_money_made : ℕ :=
  (pounds_picked 0 * payment_per_pound) + 
  (pounds_picked 1 * payment_per_pound) + 
  (pounds_picked 2 * payment_per_pound) + 
  (pounds_picked 3 * payment_per_pound)

theorem steves_earning 
  (h0 : pounds_picked 0 = 8)
  (h1 : pounds_picked 1 = 3 * pounds_picked 0)
  (h2 : pounds_picked 2 = 0)
  (h3 : pounds_picked 3 = 18) : 
  total_money_made pounds_picked = 100 := by
  sorry

end steves_earning_l164_164810


namespace part1_part2_l164_164453

-- Definition of sets A and B
def A (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem (Ⅰ)
theorem part1 (a : ℝ) : (A a ∩ B = ∅ ∧ A a ∪ B = Set.univ) → a = 2 :=
by
  sorry

-- Theorem (Ⅱ)
theorem part2 (a : ℝ) : (A a ⊆ B ∧ A a ≠ ∅) → (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l164_164453


namespace number_of_real_roots_l164_164661

theorem number_of_real_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b^2 + 1 = 0) :
  (c > 0 → ∃ x1 x2 x3 : ℝ, 
    (x1 = b * Real.sqrt c ∨ x1 = -b * Real.sqrt c ∨ x1 = -c / b) ∧
    (x2 = b * Real.sqrt c ∨ x2 = -b * Real.sqrt c ∨ x2 = -c / b) ∧
    (x3 = b * Real.sqrt c ∨ x3 = -b * Real.sqrt c ∨ x3 = -c / b)) ∧
  (c < 0 → ∃ x1 : ℝ, x1 = -c / b) :=
by
  sorry

end number_of_real_roots_l164_164661


namespace pq_r_sum_l164_164341

theorem pq_r_sum (p q r : ℝ) (h1 : p^3 - 18 * p^2 + 27 * p - 72 = 0) 
                 (h2 : 27 * q^3 - 243 * q^2 + 729 * q - 972 = 0)
                 (h3 : 3 * r = 9) : p + q + r = 18 :=
by
  sorry

end pq_r_sum_l164_164341


namespace f_difference_l164_164343

noncomputable def f (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 3) / 12 * ((1 + Real.sqrt 3) / 2)^n + 
  (6 - 4 * Real.sqrt 3) / 12 * ((1 - Real.sqrt 3) / 2)^n

theorem f_difference (n : ℕ) : f (n + 1) - f n = (Real.sqrt 3 - 3) / 4 * f n :=
  sorry

end f_difference_l164_164343


namespace Robert_books_count_l164_164959

theorem Robert_books_count (reading_speed : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) :
    reading_speed = 120 ∧ pages_per_book = 360 ∧ hours_available = 8 →
    ∀ books_readable : ℕ, books_readable = 2 :=
by
  intros h
  cases h with rs pb_and_ha
  cases pb_and_ha with pb ha
  rw [rs, pb, ha]
  -- Here we would write the proof, but we just use "sorry" to skip it.
  sorry

end Robert_books_count_l164_164959


namespace paula_remaining_money_l164_164349

-- Define the given conditions
def given_amount : ℕ := 109
def cost_shirt : ℕ := 11
def number_shirts : ℕ := 2
def cost_pants : ℕ := 13

-- Calculate total spending
def total_spent : ℕ := (cost_shirt * number_shirts) + cost_pants

-- Define the remaining amount Paula has
def remaining_amount : ℕ := given_amount - total_spent

-- State the theorem
theorem paula_remaining_money : remaining_amount = 74 := by
  -- Proof goes here
  sorry

end paula_remaining_money_l164_164349


namespace maximize_xyplusxzplusyzplusy2_l164_164342

theorem maximize_xyplusxzplusyzplusy2 (x y z : ℝ) (h1 : x + 2 * y + z = 7) (h2 : y ≥ 0) :
  xy + xz + yz + y^2 ≤ 10.5 :=
sorry

end maximize_xyplusxzplusyzplusy2_l164_164342


namespace dad_steps_l164_164133

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l164_164133


namespace unique_three_digit_numbers_count_l164_164684

theorem unique_three_digit_numbers_count :
  ∃ l : List Nat, (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    l = [230, 203, 302, 320] ∧ l.length = 4 := 
by
  sorry

end unique_three_digit_numbers_count_l164_164684


namespace quadratic_complete_square_l164_164922

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 2 * x + 3 = (x - 1)^2 + 2) := 
by
  intro x
  sorry

end quadratic_complete_square_l164_164922


namespace daily_sale_correct_l164_164023

-- Define the original and additional amounts in kilograms
def original_rice := 4 * 1000 -- 4 tons converted to kilograms
def additional_rice := 4000 -- kilograms
def total_rice := original_rice + additional_rice -- total amount of rice in kilograms
def days := 4 -- days to sell all the rice

-- Statement to prove: The amount to be sold each day
def daily_sale_amount := 2000 -- kilograms per day

theorem daily_sale_correct : total_rice / days = daily_sale_amount :=
by 
  -- This is a placeholder for the proof
  sorry

end daily_sale_correct_l164_164023


namespace find_C_l164_164868

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : 
  C = 10 := 
by
  sorry

end find_C_l164_164868


namespace compute_series_sum_l164_164107

theorem compute_series_sum :
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 0.834 :=
by
  sorry

end compute_series_sum_l164_164107


namespace n_div_p_eq_27_l164_164365

theorem n_div_p_eq_27 (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0)
    (h4 : ∃ r1 r2 : ℝ, r1 * r2 = m ∧ r1 + r2 = -p ∧ (3 * r1) * (3 * r2) = n ∧ 3 * (r1 + r2) = -m)
    : n / p = 27 := sorry

end n_div_p_eq_27_l164_164365


namespace sequence_a_100_l164_164186

theorem sequence_a_100 : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902) :=
by
  sorry

end sequence_a_100_l164_164186


namespace students_who_like_both_channels_l164_164396

theorem students_who_like_both_channels (total_students : ℕ) 
    (sports_channel : ℕ) (arts_channel : ℕ) (neither_channel : ℕ)
    (h_total : total_students = 100) (h_sports : sports_channel = 68) 
    (h_arts : arts_channel = 55) (h_neither : neither_channel = 3) :
    ∃ x, (x = 26) :=
by
  have h_at_least_one := total_students - neither_channel
  have h_A_union_B := sports_channel + arts_channel - h_at_least_one
  use h_A_union_B
  sorry

end students_who_like_both_channels_l164_164396


namespace find_constant_N_l164_164586

variables (r h V_A V_B : ℝ)

theorem find_constant_N 
  (h_eq_r : h = r) 
  (r_eq_h : r = h) 
  (vol_relation : V_A = 3 * V_B) 
  (vol_A : V_A = π * r^2 * h) 
  (vol_B : V_B = π * h^2 * r) : 
 ∃ N : ℝ, V_A = N * π * h^3 ∧ N = 9 := 
by 
  use 9
  split
  sorry  -- Proof that V_A = 9 * π * h^3 goes here
  exact eq.refl 9  -- This confirms N = 9 without further proof.


end find_constant_N_l164_164586


namespace work_time_l164_164445

-- Definitions and conditions
variables (A B C D h : ℝ)
variable (h_def : ℝ := 1 / (1 / A + 1 / B + 1 / D))

-- Conditions
axiom cond1 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (A - 8)
axiom cond2 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (B - 2)
axiom cond3 : 1 / A + 1 / B + 1 / C + 1 / D = 3 / C
axiom cond4 : 1 / A + 1 / B + 1 / D = 2 / C

-- The statement to prove
theorem work_time : h_def = 16 / 11 := by
  sorry

end work_time_l164_164445


namespace largest_difference_l164_164208

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
by
  sorry  -- Proof is omitted as per instructions.

end largest_difference_l164_164208


namespace Shiela_stars_per_bottle_l164_164805

theorem Shiela_stars_per_bottle (total_stars : ℕ) (total_classmates : ℕ) (h1 : total_stars = 45) (h2 : total_classmates = 9) :
  total_stars / total_classmates = 5 := 
by 
  sorry

end Shiela_stars_per_bottle_l164_164805


namespace minimum_value_f_on_interval_l164_164735

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^3 / (Real.sin x) + (Real.sin x)^3 / (Real.cos x)

theorem minimum_value_f_on_interval : ∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x = 1 ∧ ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≥ 1 :=
by sorry

end minimum_value_f_on_interval_l164_164735


namespace dad_steps_l164_164116

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l164_164116


namespace square_adjacent_to_multiple_of_5_l164_164352

theorem square_adjacent_to_multiple_of_5 (n : ℤ) (h : n % 5 ≠ 0) : (∃ k : ℤ, n^2 = 5 * k + 1) ∨ (∃ k : ℤ, n^2 = 5 * k - 1) := 
by
  sorry

end square_adjacent_to_multiple_of_5_l164_164352


namespace miles_on_first_day_l164_164221

variable (x : ℝ)

/-- The distance traveled on the first day is x miles. -/
noncomputable def second_day_distance := (3/4) * x

/-- The distance traveled on the second day is (3/4)x miles. -/
noncomputable def third_day_distance := (1/2) * (x + second_day_distance x)

theorem miles_on_first_day
    (total_distance : x + second_day_distance x + third_day_distance x = 525)
    : x = 200 :=
sorry

end miles_on_first_day_l164_164221


namespace equation_1_solve_equation_2_solve_l164_164501

-- The first equation
theorem equation_1_solve (x : ℝ) (h : 4 * (x - 2) = 2 * x) : x = 4 :=
by
  sorry

-- The second equation
theorem equation_2_solve (x : ℝ) (h : (x + 1) / 4 = 1 - (1 - x) / 3) : x = -5 :=
by
  sorry

end equation_1_solve_equation_2_solve_l164_164501


namespace problem_statement_l164_164606

noncomputable def m (α : ℝ) : ℝ := - (Real.sqrt 2) / 4

noncomputable def tan_alpha (α : ℝ) : ℝ := 2 * Real.sqrt 2

theorem problem_statement (α : ℝ) (P : (ℝ × ℝ)) (h1 : P = (m α, 1)) (h2 : Real.cos α = - 1 / 3) :
  (P.1 = - (Real.sqrt 2) / 4) ∧ (Real.tan α = 2 * Real.sqrt 2) :=
by
  sorry

end problem_statement_l164_164606


namespace Dans_placed_scissors_l164_164253

theorem Dans_placed_scissors (initial_scissors placed_scissors total_scissors : ℕ) 
  (h1 : initial_scissors = 39) 
  (h2 : total_scissors = initial_scissors + placed_scissors) 
  (h3 : total_scissors = 52) : placed_scissors = 13 := 
by 
  sorry

end Dans_placed_scissors_l164_164253


namespace h_evaluation_l164_164945

variables {a b c : ℝ}

-- Definitions and conditions
def p (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c
def h (x : ℝ) : ℝ := sorry -- Definition of h(x) in terms of the roots of p(x)

theorem h_evaluation (ha : a < b) (hb : b < c) : h 2 = (2 + 2 * a + 3 * b + c) / (c^2) :=
sorry

end h_evaluation_l164_164945


namespace perimeter_of_ABFCDE_l164_164427

theorem perimeter_of_ABFCDE {side : ℝ} (h : side = 12) : 
  ∃ perimeter : ℝ, perimeter = 84 :=
by
  sorry

end perimeter_of_ABFCDE_l164_164427


namespace tabitha_honey_days_l164_164035

noncomputable def days_of_honey (cups_per_day servings_per_cup total_servings : ℕ) : ℕ :=
  total_servings / (cups_per_day * servings_per_cup)

theorem tabitha_honey_days :
  let cups_per_day := 3
  let servings_per_cup := 1
  let ounces_container := 16
  let servings_per_ounce := 6
  let total_servings := ounces_container * servings_per_ounce
  days_of_honey cups_per_day servings_per_cup total_servings = 32 :=
by
  sorry

end tabitha_honey_days_l164_164035


namespace new_kite_area_l164_164163

def original_base := 7
def original_height := 6
def scale_factor := 2
def side_length := 2

def new_base := original_base * scale_factor
def new_height := original_height * scale_factor
def half_new_height := new_height / 2

def area_triangle := (1 / 2 : ℚ) * new_base * half_new_height
def total_area := 2 * area_triangle

theorem new_kite_area : total_area = 84 := by
  sorry

end new_kite_area_l164_164163


namespace values_only_solution_l164_164594

variables (m n : ℝ) (x a b c : ℝ)

noncomputable def equation := (x + m)^3 - (x + n)^3 = (m + n)^3

theorem values_only_solution (hm : m ≠ 0) (hn : n ≠ 0) (hne : m ≠ n)
  (hx : x = a * m + b * n + c) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end values_only_solution_l164_164594


namespace decreasing_function_iff_a_range_l164_164623

noncomputable def f (a x : ℝ) : ℝ := (1 - 2 * a) ^ x

theorem decreasing_function_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 0 < a ∧ a < 1/2 :=
by
  sorry

end decreasing_function_iff_a_range_l164_164623


namespace problem_conditions_equation_right_triangle_vertex_coordinates_l164_164461

theorem problem_conditions_equation : 
  ∃ (a b c : ℝ), a = -1 ∧ b = -2 ∧ c = 3 ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - (-(x + 1))^2 + 4) ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - x^2 - 2 * x + 3)
:= sorry

theorem right_triangle_vertex_coordinates :
  ∀ x y : ℝ, x = -1 ∧ 
  (y = -2 ∨ y = 4 ∨ y = (3 + (17:ℝ).sqrt) / 2 ∨ y = (3 - (17:ℝ).sqrt) / 2)
  ∧ 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (-3, 0)
  let C : ℝ × ℝ := (0, 3)
  let P : ℝ × ℝ := (x, y)
  let BC : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let PB : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let PC : ℝ := (P.1 - C.1)^2 + (P.2 - C.2)^2
  (BC + PB = PC ∨ BC + PC = PB ∨ PB + PC = BC)
:= sorry

end problem_conditions_equation_right_triangle_vertex_coordinates_l164_164461


namespace plum_purchase_l164_164520

theorem plum_purchase
    (x : ℕ)
    (h1 : ∃ x, 5 * (6 * (4 * x) / 5) - 6 * ((5 * x) / 6) = -30) :
    2 * x = 60 := sorry

end plum_purchase_l164_164520


namespace books_read_l164_164961

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l164_164961


namespace red_paint_quarts_l164_164301

theorem red_paint_quarts (r g w : ℕ) (ratio_rw : r * 5 = w * 4) (w_quarts : w = 15) : r = 12 :=
by 
  -- We provide the skeleton of the proof here: the detailed steps are skipped (as instructed).
  sorry

end red_paint_quarts_l164_164301


namespace smallest_d_for_divisibility_by_9_l164_164298

theorem smallest_d_for_divisibility_by_9 : ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (437003 + d * 100) % 9 = 0 ∧ ∀ d', 0 ≤ d' ∧ d' < d → ((437003 + d' * 100) % 9 ≠ 0) :=
by
  sorry

end smallest_d_for_divisibility_by_9_l164_164298


namespace grant_earnings_l164_164915

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l164_164915


namespace percent_of_x_is_y_l164_164057

variable (x y : ℝ)

theorem percent_of_x_is_y (h : 0.20 * (x - y) = 0.15 * (x + y)) : (y / x) * 100 = 100 / 7 :=
by
  sorry

end percent_of_x_is_y_l164_164057


namespace cos_value_l164_164454

theorem cos_value (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (2 * π / 3 - α) = 1 / 3 :=
by
  sorry

end cos_value_l164_164454


namespace combined_weight_is_18442_l164_164490

noncomputable def combined_weight_proof : ℝ :=
  let elephant_weight_tons := 3
  let donkey_weight_percentage := 0.1
  let giraffe_weight_tons := 1.5
  let hippopotamus_weight_kg := 4000
  let elephant_food_oz := 16
  let donkey_food_lbs := 5
  let giraffe_food_kg := 3
  let hippopotamus_food_g := 5000

  let ton_to_pounds := 2000
  let kg_to_pounds := 2.20462
  let oz_to_pounds := 1 / 16
  let g_to_pounds := 0.00220462

  let elephant_weight_pounds := elephant_weight_tons * ton_to_pounds
  let donkey_weight_pounds := (1 - donkey_weight_percentage) * elephant_weight_pounds
  let giraffe_weight_pounds := giraffe_weight_tons * ton_to_pounds
  let hippopotamus_weight_pounds := hippopotamus_weight_kg * kg_to_pounds

  let elephant_food_pounds := elephant_food_oz * oz_to_pounds
  let giraffe_food_pounds := giraffe_food_kg * kg_to_pounds
  let hippopotamus_food_pounds := hippopotamus_food_g * g_to_pounds

  elephant_weight_pounds + donkey_weight_pounds + giraffe_weight_pounds + hippopotamus_weight_pounds +
  elephant_food_pounds + donkey_food_lbs + giraffe_food_pounds + hippopotamus_food_pounds

theorem combined_weight_is_18442 : combined_weight_proof = 18442 := by
  sorry

end combined_weight_is_18442_l164_164490


namespace bookcase_length_in_inches_l164_164222

theorem bookcase_length_in_inches (feet_length : ℕ) (inches_per_foot : ℕ) (h1 : feet_length = 4) (h2 : inches_per_foot = 12) : (feet_length * inches_per_foot) = 48 :=
by
  sorry

end bookcase_length_in_inches_l164_164222


namespace sum_largest_smallest_5_6_7_l164_164601

/--
Given the digits 5, 6, and 7, if we form all possible three-digit numbers using each digit exactly once, 
then the sum of the largest and smallest of these numbers is 1332.
-/
theorem sum_largest_smallest_5_6_7 : 
  let d1 := 5
  let d2 := 6
  let d3 := 7
  let smallest := 100 * d1 + 10 * d2 + d3
  let largest := 100 * d3 + 10 * d2 + d1
  smallest + largest = 1332 := 
by
  sorry

end sum_largest_smallest_5_6_7_l164_164601


namespace find_number_l164_164442

noncomputable def calc1 : Float := 0.47 * 1442
noncomputable def calc2 : Float := 0.36 * 1412
noncomputable def diff : Float := calc1 - calc2

theorem find_number :
  ∃ (n : Float), (diff + n = 6) :=
sorry

end find_number_l164_164442


namespace sum_first_3n_terms_is_36_l164_164977

-- Definitions and conditions
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2
def sum_first_2n_terms (a d : ℤ) (n : ℕ) : ℤ := 2 * n * (2 * a + (2 * n - 1) * d) / 2
def sum_first_3n_terms (a d : ℤ) (n : ℕ) : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2

axiom h1 : ∀ (a d : ℤ) (n : ℕ), sum_first_n_terms a d n = 48
axiom h2 : ∀ (a d : ℤ) (n : ℕ), sum_first_2n_terms a d n = 60

theorem sum_first_3n_terms_is_36 (a d : ℤ) (n : ℕ) : sum_first_3n_terms a d n = 36 := by
  sorry

end sum_first_3n_terms_is_36_l164_164977


namespace find_y_l164_164618

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end find_y_l164_164618


namespace meeting_time_eqn_l164_164683

-- Mathematical definitions derived from conditions:
def distance := 270 -- Cities A and B are 270 kilometers apart.
def speed_fast_train := 120 -- Speed of the fast train is 120 km/h.
def speed_slow_train := 75 -- Speed of the slow train is 75 km/h.
def time_head_start := 1 -- Slow train departs 1 hour before the fast train.

-- Let x be the number of hours it takes for the two trains to meet after the fast train departs
def x : Real := sorry

-- Proving the equation representing the situation:
theorem meeting_time_eqn : 75 * 1 + (120 + 75) * x = 270 :=
by
  sorry

end meeting_time_eqn_l164_164683


namespace find_number_of_As_l164_164484

variables (M L S : ℕ)

def number_of_As (M L S : ℕ) : Prop :=
  M + L = 23 ∧ S + M = 18 ∧ S + L = 15

theorem find_number_of_As (M L S : ℕ) (h : number_of_As M L S) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end find_number_of_As_l164_164484


namespace problem_180_l164_164281

variables (P Q : Prop)

theorem problem_180 (h : P → Q) : ¬ (P ∨ ¬Q) :=
sorry

end problem_180_l164_164281


namespace inscribed_circle_radius_eq_l164_164726

noncomputable def inscribedCircleRadius :=
  let AB := 6
  let AC := 7
  let BC := 8
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  r

theorem inscribed_circle_radius_eq :
  inscribedCircleRadius = Real.sqrt 413.4375 / 10.5 := by
  sorry

end inscribed_circle_radius_eq_l164_164726


namespace total_kayaks_built_l164_164409

/-- Geometric sequence sum definition -/
def geom_sum (a r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a
  else a * (r ^ n - 1) / (r - 1)

/-- Problem statement: Prove that the total number of kayaks built by the end of June is 726 -/
theorem total_kayaks_built : geom_sum 6 3 5 = 726 :=
  sorry

end total_kayaks_built_l164_164409


namespace Winnie_the_Pooh_guarantee_kilogram_l164_164928

noncomputable def guarantee_minimum_honey : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 + a2 + a3 + a4 + a5 = 3 ∧
    min (min (a1 + a2) (a2 + a3)) (min (a3 + a4) (a4 + a5)) ≥ 1

theorem Winnie_the_Pooh_guarantee_kilogram :
  guarantee_minimum_honey :=
sorry

end Winnie_the_Pooh_guarantee_kilogram_l164_164928


namespace candy_count_l164_164514

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l164_164514


namespace order_y1_y2_y3_l164_164465

-- Defining the parabolic function and the points A, B, C
def parabola (a x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

-- Points A, B, C
def y1 (a : ℝ) : ℝ := parabola a (-1)
def y2 (a : ℝ) : ℝ := parabola a 2
def y3 (a : ℝ) : ℝ := parabola a 4

-- Assumption: a > 0
variables (a : ℝ) (h : a > 0)

-- The theorem to prove
theorem order_y1_y2_y3 : 
  y2 a < y1 a ∧ y1 a < y3 a :=
sorry

end order_y1_y2_y3_l164_164465


namespace success_permutations_correct_l164_164880

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l164_164880


namespace yellow_balloons_ratio_l164_164254

theorem yellow_balloons_ratio 
  (total_balloons : ℕ) 
  (colors : ℕ) 
  (yellow_balloons_taken : ℕ) 
  (h_total_balloons : total_balloons = 672)
  (h_colors : colors = 4)
  (h_yellow_balloons_taken : yellow_balloons_taken = 84) :
  yellow_balloons_taken / (total_balloons / colors) = 1 / 2 :=
sorry

end yellow_balloons_ratio_l164_164254


namespace triangle_ABC_no_common_factor_l164_164367

theorem triangle_ABC_no_common_factor (a b c : ℕ) (h_coprime: Nat.gcd (Nat.gcd a b) c = 1)
  (h_angleB_eq_2angleC : True) (h_b_lt_600 : b < 600) : False :=
by
  sorry

end triangle_ABC_no_common_factor_l164_164367


namespace sally_needs_8_napkins_l164_164804

theorem sally_needs_8_napkins :
  let tablecloth_length := 102
  let tablecloth_width := 54
  let napkin_length := 6
  let napkin_width := 7
  let total_material_needed := 5844
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let material_needed_for_napkins := total_material_needed - tablecloth_area
  let number_of_napkins := material_needed_for_napkins / napkin_area
  number_of_napkins = 8 :=
by
  sorry

end sally_needs_8_napkins_l164_164804


namespace time_taken_to_cross_platform_l164_164873

noncomputable def length_of_train : ℝ := 100 -- in meters
noncomputable def speed_of_train_km_hr : ℝ := 60 -- in km/hr
noncomputable def length_of_platform : ℝ := 150 -- in meters

noncomputable def speed_of_train_m_s := speed_of_train_km_hr * (1000 / 3600) -- converting km/hr to m/s
noncomputable def total_distance := length_of_train + length_of_platform
noncomputable def time_taken := total_distance / speed_of_train_m_s

theorem time_taken_to_cross_platform : abs (time_taken - 15) < 0.1 :=
by
  sorry

end time_taken_to_cross_platform_l164_164873


namespace smallest_integer_larger_than_expression_l164_164260

theorem smallest_integer_larger_than_expression :
  ∃ n : ℤ, n = 248 ∧ (↑n > ((Real.sqrt 5 + Real.sqrt 3) ^ 4 : ℝ)) :=
by
  sorry

end smallest_integer_larger_than_expression_l164_164260


namespace probability_either_but_not_both_l164_164516

open Classical

def event_chile := Ω → Prop
def event_madagascar := Ω → Prop

variable {Ω : Type}

axiom P_chile : ProbabilityTheory ℙ event_chile
axiom P_madagascar : ProbabilityTheory ℙ event_madagascar

axiom P_chile_given : ℙ[event_chile] = 0.5
axiom P_madagascar_given : ℙ[event_madagascar] = 0.5

theorem probability_either_but_not_both :
  ℙ[event_chile ∧ ¬event_madagascar] + ℙ[¬event_chile ∧ event_madagascar] = 0.5 :=
  sorry

end probability_either_but_not_both_l164_164516


namespace cirrus_to_cumulus_is_four_l164_164368

noncomputable def cirrus_to_cumulus_ratio (Ci Cu Cb : ℕ) : ℕ :=
  Ci / Cu

theorem cirrus_to_cumulus_is_four :
  ∀ (Ci Cu Cb : ℕ), (Cb = 3) → (Cu = 12 * Cb) → (Ci = 144) → cirrus_to_cumulus_ratio Ci Cu Cb = 4 :=
by
  intros Ci Cu Cb hCb hCu hCi
  sorry

end cirrus_to_cumulus_is_four_l164_164368


namespace right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l164_164262

theorem right_triangle_set_A :
  let a := 1
      b := 2
      c := Real.sqrt 5
  in a^2 + b^2 = c^2 := 
sorry

theorem not_right_triangle_set_B :
  let a := 6
      b := 8
      c := 9
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_C :
  let a := Real.sqrt 3
      b := Real.sqrt 2
      c := 5
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_D :
  let a := 3^2
      b := 4^2
      c := 5^2
  in a^2 + b^2 ≠ c^2 :=
sorry

end right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l164_164262


namespace equation_solution_unique_l164_164248

theorem equation_solution_unique (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
    (2 / (x - 3) = 3 / x ↔ x = 9) :=
by
  sorry

end equation_solution_unique_l164_164248


namespace prob_a_wins_match_l164_164382

-- Define the probability of A winning a single game
def prob_win_a_single_game : ℚ := 1 / 3

-- Define the probability of A winning two consecutive games
def prob_win_a_two_consec_games : ℚ := prob_win_a_single_game * prob_win_a_single_game

-- Define the probability of A winning two games with one loss in between
def prob_win_a_two_wins_one_loss_first : ℚ := prob_win_a_single_game * (1 - prob_win_a_single_game) * prob_win_a_single_game
def prob_win_a_two_wins_one_loss_second : ℚ := (1 - prob_win_a_single_game) * prob_win_a_single_game * prob_win_a_single_game

-- Define the total probability of A winning the match
def prob_a_winning_match : ℚ := prob_win_a_two_consec_games + prob_win_a_two_wins_one_loss_first + prob_win_a_two_wins_one_loss_second

-- The theorem to be proved
theorem prob_a_wins_match : prob_a_winning_match = 7 / 27 :=
by sorry

end prob_a_wins_match_l164_164382


namespace expected_value_a_squared_norm_bound_l164_164066

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l164_164066


namespace remainder_333_pow_333_mod_11_l164_164831

theorem remainder_333_pow_333_mod_11 : (333 ^ 333) % 11 = 5 := by
  sorry

end remainder_333_pow_333_mod_11_l164_164831


namespace luke_total_points_l164_164019

-- Definitions based on conditions
def points_per_round : ℕ := 3
def rounds_played : ℕ := 26

-- Theorem stating the question and correct answer
theorem luke_total_points : points_per_round * rounds_played = 78 := 
by 
  sorry

end luke_total_points_l164_164019


namespace subtraction_result_l164_164050

theorem subtraction_result: (3.75 - 1.4 = 2.35) :=
by
  sorry

end subtraction_result_l164_164050


namespace savings_percentage_first_year_l164_164389

noncomputable def savings_percentage (I S : ℝ) : ℝ := (S / I) * 100

theorem savings_percentage_first_year (I S : ℝ) (h1 : S = 0.20 * I) :
  savings_percentage I S = 20 :=
by
  unfold savings_percentage
  rw [h1]
  field_simp
  norm_num
  sorry

end savings_percentage_first_year_l164_164389


namespace count_success_permutations_l164_164882

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l164_164882


namespace middle_person_distance_l164_164703

noncomputable def Al_position (t : ℝ) : ℝ := 6 * t
noncomputable def Bob_position (t : ℝ) : ℝ := 10 * t - 12
noncomputable def Cy_position (t : ℝ) : ℝ := 8 * t - 32

theorem middle_person_distance (t : ℝ) (h₁ : t ≥ 0) (h₂ : t ≥ 2) (h₃ : t ≥ 4) :
  (Al_position t = 52) ∨ (Bob_position t = 52) ∨ (Cy_position t = 52) :=
sorry

end middle_person_distance_l164_164703


namespace bacteria_colony_first_day_exceeds_100_l164_164925

theorem bacteria_colony_first_day_exceeds_100 :
  ∃ n : ℕ, 3 * 2^n > 100 ∧ (∀ m < n, 3 * 2^m ≤ 100) :=
sorry

end bacteria_colony_first_day_exceeds_100_l164_164925


namespace convert_base8_to_base7_l164_164583

theorem convert_base8_to_base7 : (536%8).toBase 7 = 1010%7 :=
by
  sorry

end convert_base8_to_base7_l164_164583


namespace determine_number_of_20_pound_boxes_l164_164629

variable (numBoxes : ℕ) (avgWeight : ℕ) (x : ℕ) (y : ℕ)

theorem determine_number_of_20_pound_boxes 
  (h1 : numBoxes = 30) 
  (h2 : avgWeight = 18) 
  (h3 : x + y = 30) 
  (h4 : 10 * x + 20 * y = 540) : 
  y = 24 :=
  by
  sorry

end determine_number_of_20_pound_boxes_l164_164629


namespace average_after_12th_inning_revised_average_not_out_l164_164537

theorem average_after_12th_inning (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) : (A + 2) = 70 :=
by
  -- Calculation steps are skipped
  sorry

theorem revised_average_not_out (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) (H_not_out : 11 * A + 92 = 840) :
  (11 * A + 92) / 9 = 93.33 :=
by
  -- Calculation steps are skipped
  sorry

end average_after_12th_inning_revised_average_not_out_l164_164537


namespace pizza_slices_l164_164854

theorem pizza_slices (total_slices slices_with_pepperoni slices_with_mushrooms : ℕ) (h1 : total_slices = 15)
  (h2 : slices_with_pepperoni = 8) (h3 : slices_with_mushrooms = 12)
  (h4 : ∀ slice, slice < total_slices → (slice ∈ {x | x < slices_with_pepperoni} ∨ slice ∈ {x | x < slices_with_mushrooms})) :
  ∃ n : ℕ, (slices_with_pepperoni - n) + (slices_with_mushrooms - n) + n = total_slices ∧ n = 5 :=
by simp [h1, h2, h3]; use 5; linarith; sorry

end pizza_slices_l164_164854


namespace joe_left_pocket_initial_l164_164532

-- Definitions from conditions
def total_money : ℕ := 200
def initial_left_pocket (L : ℕ) : ℕ := L
def initial_right_pocket (R : ℕ) : ℕ := R
def transfer_one_fourth (L : ℕ) : ℕ := L - L / 4
def add_to_right (R : ℕ) (L : ℕ) : ℕ := R + L / 4
def transfer_20 (L : ℕ) : ℕ := transfer_one_fourth L - 20
def add_20_to_right (R : ℕ) (L : ℕ) : ℕ := add_to_right R L + 20

-- Statement to prove
theorem joe_left_pocket_initial (L R : ℕ) (h₁ : L + R = total_money) 
  (h₂ : transfer_20 L = add_20_to_right R L) : 
  initial_left_pocket L = 160 :=
by
  sorry

end joe_left_pocket_initial_l164_164532


namespace distinct_positive_integers_solution_l164_164590

theorem distinct_positive_integers_solution (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : 1 / x + 1 / y = 2 / 7) : (x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4) :=
by
  sorry -- proof to be filled in.

end distinct_positive_integers_solution_l164_164590


namespace initial_books_l164_164224

theorem initial_books (added_books : ℝ) (books_per_shelf : ℝ) (shelves : ℝ) 
  (total_books : ℝ) : total_books = shelves * books_per_shelf → 
  shelves = 14 → books_per_shelf = 4.0 → added_books = 10.0 → 
  total_books - added_books = 46.0 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_books_l164_164224


namespace sufficient_but_not_necessary_condition_l164_164305

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, |x + 1| + |x - 1| ≥ m
def proposition_q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - 2 * m * x₀ + m^2 + m - 3 = 0

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (proposition_p m → proposition_q m) ∧ ¬ (proposition_q m → proposition_p m) :=
sorry

end sufficient_but_not_necessary_condition_l164_164305


namespace ball_reaches_height_less_than_2_after_6_bounces_l164_164704

theorem ball_reaches_height_less_than_2_after_6_bounces :
  ∃ (k : ℕ), 16 * (2/3) ^ k < 2 ∧ ∀ (m : ℕ), m < k → 16 * (2/3) ^ m ≥ 2 :=
by
  sorry

end ball_reaches_height_less_than_2_after_6_bounces_l164_164704


namespace find_sticker_price_l164_164183

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l164_164183


namespace dad_steps_are_90_l164_164142

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l164_164142


namespace car_tank_capacity_is_12_gallons_l164_164049

noncomputable def truck_tank_capacity : ℕ := 20
noncomputable def truck_tank_half_full : ℕ := truck_tank_capacity / 2
noncomputable def car_tank_third_full (car_tank_capacity : ℕ) : ℕ := car_tank_capacity / 3
noncomputable def total_gallons_added : ℕ := 18

theorem car_tank_capacity_is_12_gallons (car_tank_capacity : ℕ) 
    (h1 : truck_tank_half_full + (car_tank_third_full car_tank_capacity) + 18 = truck_tank_capacity + car_tank_capacity) 
    (h2 : total_gallons_added = 18) : car_tank_capacity = 12 := 
by
  sorry

end car_tank_capacity_is_12_gallons_l164_164049


namespace mark_more_than_kate_l164_164653

variables {K P M : ℕ}

-- Conditions
def total_hours (P K M : ℕ) : Prop := P + K + M = 189
def pat_as_kate (P K : ℕ) : Prop := P = 2 * K
def pat_as_mark (P M : ℕ) : Prop := P = M / 3

-- Statement
theorem mark_more_than_kate (K P M : ℕ) (h1 : total_hours P K M)
  (h2 : pat_as_kate P K) (h3 : pat_as_mark P M) : M - K = 105 :=
by {
  sorry
}

end mark_more_than_kate_l164_164653


namespace profit_difference_l164_164541

theorem profit_difference
  (p1 p2 : ℝ)
  (h1 : p1 > p2)
  (h2 : p1 + p2 = 3635000)
  (h3 : p2 = 442500) :
  p1 - p2 = 2750000 :=
by 
  sorry

end profit_difference_l164_164541


namespace percentage_problem_l164_164620

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by 
  sorry

end percentage_problem_l164_164620


namespace solution_set_of_x_x_plus_2_lt_3_l164_164511

theorem solution_set_of_x_x_plus_2_lt_3 :
  {x : ℝ | x*(x + 2) < 3} = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_x_x_plus_2_lt_3_l164_164511


namespace first_pair_weight_l164_164289

variable (total_weight : ℕ) (second_pair_weight : ℕ) (third_pair_weight : ℕ)

theorem first_pair_weight (h : total_weight = 32) (h_second : second_pair_weight = 5) (h_third : third_pair_weight = 8) : 
    total_weight - 2 * (second_pair_weight + third_pair_weight) = 6 :=
by
  sorry

end first_pair_weight_l164_164289


namespace convert_536_oct_to_base7_l164_164585

def octal_to_decimal (n : ℕ) : ℕ :=
  n % 10 + (n / 10 % 10) * 8 + (n / 100 % 10) * 64

def decimal_to_base7 (n : ℕ) : ℕ :=
  n % 7 + (n / 7 % 7) * 10 + (n / 49 % 7) * 100 + (n / 343 % 7) * 1000

theorem convert_536_oct_to_base7 : 
  decimal_to_base7 (octal_to_decimal 536) = 1010 :=
by
  sorry

end convert_536_oct_to_base7_l164_164585


namespace symmetry_probability_is_two_fifths_l164_164283

-- Define the grid and center point P
def P := (5, 5 : ℕ × ℕ)

-- Define symmetry axes and count favorable points excluding the center
def count_points_on_symmetry_axes : ℕ :=
  let vert_points := 8
  let horiz_points := 8
  let leading_diag_points := 8
  let anti_diag_points := 8
  vert_points + horiz_points + leading_diag_points + anti_diag_points

-- Total number of possible points excluding the center
def total_possible_points := 80

-- Calculate probability
def symmetry_probability : ℚ :=
  count_points_on_symmetry_axes / total_possible_points

-- Final theorem statement
theorem symmetry_probability_is_two_fifths : symmetry_probability = 2 / 5 := by sorry

end symmetry_probability_is_two_fifths_l164_164283


namespace necessary_conditions_l164_164455

theorem necessary_conditions (a b c d e : ℝ) (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) :
  a = c ∨ a + b + c + d + e = 0 :=
by
  sorry

end necessary_conditions_l164_164455


namespace mary_initial_amount_l164_164279

theorem mary_initial_amount (current_amount pie_cost mary_after_pie : ℕ) 
  (h1 : pie_cost = 6) 
  (h2 : mary_after_pie = 52) :
  current_amount = pie_cost + mary_after_pie → 
  current_amount = 58 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end mary_initial_amount_l164_164279


namespace change_is_4_25_l164_164008

-- Define the conditions
def apple_cost : ℝ := 0.75
def amount_paid : ℝ := 5.00

-- State the theorem
theorem change_is_4_25 : amount_paid - apple_cost = 4.25 :=
by
  sorry

end change_is_4_25_l164_164008


namespace income_growth_l164_164626

theorem income_growth (x : ℝ) : 12000 * (1 + x)^2 = 14520 :=
sorry

end income_growth_l164_164626


namespace compute_a_plus_b_l164_164359

-- Define the volume formula for a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

-- Given conditions
def radius_small_sphere : ℝ := 6
def volume_small_sphere := volume_of_sphere radius_small_sphere
def volume_large_sphere := 3 * volume_small_sphere

-- Radius of the larger sphere
def radius_large_sphere := (volume_large_sphere * 3 / (4 * π))^(1/3)
def diameter_large_sphere := 2 * radius_large_sphere

-- Express diameter in the form a*root(3, b)
def a : ℕ := 12
def b : ℕ := 3

-- The mathematically equivalent proof problem
theorem compute_a_plus_b : (a + b) = 15 := by
  sorry

end compute_a_plus_b_l164_164359


namespace probability_of_ace_ten_king_l164_164849

noncomputable def probability_first_ace_second_ten_third_king : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem probability_of_ace_ten_king :
  probability_first_ace_second_ten_third_king = 2/16575 :=
by
  sorry

end probability_of_ace_ten_king_l164_164849


namespace log_equation_solution_l164_164264

theorem log_equation_solution {x : ℝ} (hx : x > 0) (hx1 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by 
  sorry

end log_equation_solution_l164_164264


namespace barry_wand_trick_l164_164022

theorem barry_wand_trick (n : ℕ) (h : (n + 3 : ℝ) / 3 = 50) : n = 147 := by
  sorry

end barry_wand_trick_l164_164022


namespace money_last_weeks_l164_164346

theorem money_last_weeks (mowing_earning : ℕ) (weeding_earning : ℕ) (spending_per_week : ℕ) 
  (total_amount : ℕ) (weeks : ℕ) :
  mowing_earning = 9 →
  weeding_earning = 18 →
  spending_per_week = 3 →
  total_amount = mowing_earning + weeding_earning →
  weeks = total_amount / spending_per_week →
  weeks = 9 :=
by
  intros
  sorry

end money_last_weeks_l164_164346


namespace zs_share_in_profit_l164_164836

noncomputable def calculateProfitShare (x_investment y_investment z_investment z_months total_profit : ℚ) : ℚ :=
  let x_invest_months := x_investment * 12
  let y_invest_months := y_investment * 12
  let z_invest_months := z_investment * z_months
  let total_invest_months := x_invest_months + y_invest_months + z_invest_months
  let z_share := z_invest_months / total_invest_months
  total_profit * z_share

theorem zs_share_in_profit :
  calculateProfitShare 36000 42000 48000 8 14190 = 2580 :=
by
  sorry

end zs_share_in_profit_l164_164836


namespace distance_after_12_sec_time_to_travel_380_meters_l164_164669

-- Define the function expressing the distance s in terms of the travel time t
def distance (t : ℝ) : ℝ := 9 * t + (1 / 2) * t^2

-- Proof problem 1: Distance traveled after 12 seconds
theorem distance_after_12_sec : distance 12 = 180 := 
sorry

-- Proof problem 2: Time to travel 380 meters
theorem time_to_travel_380_meters (t : ℝ) (h : distance t = 380) : t = 20 := 
sorry

end distance_after_12_sec_time_to_travel_380_meters_l164_164669


namespace dad_steps_l164_164127

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l164_164127


namespace max_cos_a_correct_l164_164638

noncomputable def max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) : ℝ :=
  Real.sqrt 3 - 1

theorem max_cos_a_correct (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  max_cos_a a b h = Real.sqrt 3 - 1 :=
sorry

end max_cos_a_correct_l164_164638


namespace original_price_of_trouser_l164_164632

theorem original_price_of_trouser (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 30) (h2 : discount = 0.70) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l164_164632


namespace CarriesJellybeanCount_l164_164565

-- Definitions based on conditions in part a)
def BertBoxJellybeans : ℕ := 150
def BertBoxVolume : ℕ := 6
def CarriesBoxVolume : ℕ := 3 * 2 * 4 * BertBoxVolume -- (3 * height, 2 * width, 4 * length)

-- Theorem statement in Lean based on part c)
theorem CarriesJellybeanCount : (CarriesBoxVolume / BertBoxVolume) * BertBoxJellybeans = 3600 := by 
  sorry

end CarriesJellybeanCount_l164_164565


namespace point_below_line_range_l164_164771

theorem point_below_line_range (t : ℝ) : (2 * (-2) - 3 * t + 6 > 0) → t < (2 / 3) :=
by {
  sorry
}

end point_below_line_range_l164_164771


namespace eval_expression_l164_164438

def base8_to_base10 (n : Nat) : Nat :=
  2 * 8^2 + 4 * 8^1 + 5 * 8^0

def base4_to_base10 (n : Nat) : Nat :=
  1 * 4^1 + 5 * 4^0

def base5_to_base10 (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 2 * 5^0

def base6_to_base10 (n : Nat) : Nat :=
  3 * 6^1 + 2 * 6^0

theorem eval_expression : 
  base8_to_base10 245 / base4_to_base10 15 - base5_to_base10 232 / base6_to_base10 32 = 15 :=
by sorry

end eval_expression_l164_164438


namespace find_distance_l164_164840

-- Definitions based on conditions
def speed : ℝ := 75 -- in km/hr
def time : ℝ := 4 -- in hr

-- Statement to be proved
theorem find_distance : speed * time = 300 := by
  sorry

end find_distance_l164_164840


namespace owen_wins_with_n_bullseyes_l164_164564

-- Define the parameters and conditions
def initial_score_lead : ℕ := 60
def total_shots : ℕ := 120
def bullseye_points : ℕ := 9
def minimum_points_per_shot : ℕ := 3
def max_points_per_shot : ℕ := 9
def n : ℕ := 111

-- Define the condition for Owen's winning requirement
theorem owen_wins_with_n_bullseyes :
  6 * 111 + 360 > 1020 :=
by
  sorry

end owen_wins_with_n_bullseyes_l164_164564


namespace candy_count_l164_164515

variables (S M L : ℕ)

theorem candy_count :
  S + M + L = 110 ∧ S + L = 100 ∧ L = S + 20 → S = 40 ∧ M = 10 ∧ L = 60 :=
by
  intros h
  sorry

end candy_count_l164_164515


namespace time_to_cover_length_l164_164056

def escalator_rate : ℝ := 12 -- rate of the escalator in feet per second
def person_rate : ℝ := 8 -- rate of the person in feet per second
def escalator_length : ℝ := 160 -- length of the escalator in feet

theorem time_to_cover_length : escalator_length / (escalator_rate + person_rate) = 8 := by
  sorry

end time_to_cover_length_l164_164056
