import Mathlib

namespace A_lent_5000_to_B_l970_97030

noncomputable def principalAmountB
    (P_C : ℝ)
    (r : ℝ)
    (total_interest : ℝ)
    (P_B : ℝ) : Prop :=
  let I_B := P_B * r * 2
  let I_C := P_C * r * 4
  I_B + I_C = total_interest

theorem A_lent_5000_to_B :
  principalAmountB 3000 0.10 2200 5000 :=
by
  sorry

end A_lent_5000_to_B_l970_97030


namespace horse_tile_system_l970_97072

theorem horse_tile_system (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + (1 / 3 : ℚ) * y = 100) : 
  ∃ (x y : ℕ), (x + y = 100) ∧ (3 * x + (1 / 3 : ℚ) * y = 100) :=
by sorry

end horse_tile_system_l970_97072


namespace apples_left_proof_l970_97029

def apples_left (mike_apples : Float) (nancy_apples : Float) (keith_apples_eaten : Float): Float :=
  mike_apples + nancy_apples - keith_apples_eaten

theorem apples_left_proof :
  apples_left 7.0 3.0 6.0 = 4.0 :=
by
  unfold apples_left
  norm_num
  sorry

end apples_left_proof_l970_97029


namespace area_change_l970_97043

variable (p k : ℝ)
variable {N : ℝ}

theorem area_change (hN : N = 1/2 * (p * p)) (q : ℝ) (hq : q = k * p) :
  q = k * p -> (1/2 * (q * q) = k^2 * N) :=
by
  intros
  sorry

end area_change_l970_97043


namespace side_length_a_cosine_A_l970_97044

variable (A B C : Real)
variable (a b c : Real)
variable (triangle_inequality : a + b + c = 10)
variable (sine_equation : Real.sin B + Real.sin C = 4 * Real.sin A)
variable (bc_product : b * c = 16)

theorem side_length_a :
  a = 2 :=
  sorry

theorem cosine_A :
  b + c = 8 → 
  a = 2 → 
  b * c = 16 →
  Real.cos A = 7 / 8 :=
  sorry

end side_length_a_cosine_A_l970_97044


namespace geometric_mean_of_1_and_9_is_pm3_l970_97020

theorem geometric_mean_of_1_and_9_is_pm3 (a b c : ℝ) (h₀ : a = 1) (h₁ : b = 9) (h₂ : c^2 = a * b) : c = 3 ∨ c = -3 := by
  sorry

end geometric_mean_of_1_and_9_is_pm3_l970_97020


namespace relationship_between_areas_l970_97032

-- Assume necessary context and setup
variables (A B C C₁ C₂ : ℝ)
variables (a b c : ℝ) (h : a^2 + b^2 = c^2)

-- Define the conditions
def right_triangle := a = 8 ∧ b = 15 ∧ c = 17
def circumscribed_circle (d : ℝ) := d = 17
def areas_relation (A B C₁ C₂ : ℝ) := (C₁ < C₂) ∧ (A + B = C₁ + C₂)

-- Problem statement in Lean 4
theorem relationship_between_areas (ht : right_triangle 8 15 17) (hc : circumscribed_circle 17) :
  areas_relation A B C₁ C₂ :=
by sorry

end relationship_between_areas_l970_97032


namespace tan_theta_minus_pi_four_l970_97016

theorem tan_theta_minus_pi_four (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) (h3 : Real.sin θ = -3/5) :
  Real.tan (θ - π / 4) = -1 / 7 :=
sorry

end tan_theta_minus_pi_four_l970_97016


namespace correct_propositions_l970_97077

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem correct_propositions :
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) ∧
  (Real.sqrt 2 = f (Real.pi / 24)) ∧
  (f (-1) ≠ f 1) ∧
  (∀ x, Real.pi / 24 ≤ x ∧ x ≤ 13 * Real.pi / 24 -> (f (x + 1e-6) < f x)) ∧
  (∀ x, (Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) = f x)
  := by
    sorry

end correct_propositions_l970_97077


namespace garden_width_l970_97066

theorem garden_width (w : ℕ) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end garden_width_l970_97066


namespace larger_number_l970_97059

theorem larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end larger_number_l970_97059


namespace solve_for_S_l970_97018

theorem solve_for_S (S : ℝ) (h : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120) : S = 120 :=
sorry

end solve_for_S_l970_97018


namespace isosceles_triangle_perimeter_l970_97011

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 3) (h2 : b = 1) :
  (a = 3 ∧ b = 1) ∧ (a + b > b ∨ b + b > a) → a + a + b = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l970_97011


namespace a_minus_b_value_l970_97084

theorem a_minus_b_value (a b c : ℝ) (x : ℝ) 
    (h1 : (2 * x - 3) ^ 2 = a * x ^ 2 + b * x + c)
    (h2 : x = 0 → c = 9)
    (h3 : x = 1 → a + b + c = 1)
    (h4 : x = -1 → (2 * (-1) - 3) ^ 2 = a * (-1) ^ 2 + b * (-1) + c) : 
    a - b = 16 :=
by  
  sorry

end a_minus_b_value_l970_97084


namespace sum_of_seven_digits_l970_97036

theorem sum_of_seven_digits : 
  ∃ (digits : Finset ℕ), 
    digits.card = 7 ∧ 
    digits ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    ∃ (a b c d e f g : ℕ), 
      a + b + c = 25 ∧ 
      d + e + f + g = 17 ∧ 
      digits = {a, b, c, d, e, f, g} ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
      c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
      d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
      e ≠ f ∧ e ≠ g ∧
      f ≠ g ∧
      (a + b + c + d + e + f + g = 33) := sorry

end sum_of_seven_digits_l970_97036


namespace ellipse_range_of_k_l970_97033

theorem ellipse_range_of_k (k : ℝ) :
  (4 - k > 0) → (k - 1 > 0) → (4 - k ≠ k - 1) → (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_of_k_l970_97033


namespace monotonic_increasing_condition_l970_97083

open Real

noncomputable def f (x : ℝ) (l a : ℝ) : ℝ := x^2 - x + l + a * log x

theorem monotonic_increasing_condition (l a : ℝ) (x : ℝ) (hx : x > 0) 
  (h : ∀ x, x > 0 → deriv (f l a) x ≥ 0) : 
  a > 1 / 8 :=
by
  sorry

end monotonic_increasing_condition_l970_97083


namespace kiana_and_her_siblings_age_sum_l970_97048

theorem kiana_and_her_siblings_age_sum :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 256 ∧ a + b + c = 38 :=
by
sorry

end kiana_and_her_siblings_age_sum_l970_97048


namespace count_obtuse_triangle_values_k_l970_97096

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  if a ≥ b ∧ a ≥ c then a * a > b * b + c * c 
  else if b ≥ a ∧ b ≥ c then b * b > a * a + c * c
  else c * c > a * a + b * b

theorem count_obtuse_triangle_values_k :
  ∃! (k : ℕ), is_triangle 8 18 k ∧ is_obtuse_triangle 8 18 k :=
sorry

end count_obtuse_triangle_values_k_l970_97096


namespace frank_money_left_l970_97013

theorem frank_money_left (initial_money : ℝ) (spent_groceries : ℝ) (spent_magazine : ℝ) :
  initial_money = 600 →
  spent_groceries = (1/5) * initial_money →
  spent_magazine = (1/4) * (initial_money - spent_groceries) →
  initial_money - spent_groceries - spent_magazine = 360 := 
by
  intro h1 h2 h3
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  sorry

end frank_money_left_l970_97013


namespace murtha_pebbles_after_20_days_l970_97093

/- Define the sequence function for the pebbles collected each day -/
def pebbles_collected_day (n : ℕ) : ℕ :=
  if (n = 0) then 0 else 1 + pebbles_collected_day (n - 1)

/- Define the total pebbles collected by the nth day -/
def total_pebbles_collected (n : ℕ) : ℕ :=
  (n * (pebbles_collected_day n)) / 2

/- Define the total pebbles given away by the nth day -/
def pebbles_given_away (n : ℕ) : ℕ :=
  (n / 5) * 3

/- Define the net total of pebbles Murtha has on the nth day -/
def pebbles_net (n : ℕ) : ℕ :=
  total_pebbles_collected (n + 1) - pebbles_given_away (n + 1)

/- The main theorem about the pebbles Murtha has after the 20th day -/
theorem murtha_pebbles_after_20_days : pebbles_net 19 = 218 := 
  by sorry

end murtha_pebbles_after_20_days_l970_97093


namespace faster_speed_l970_97040

theorem faster_speed (v : ℝ) :
  (∀ t : ℝ, (40 / 10 = t) ∧ (60 / v = t)) → v = 15 :=
by
  sorry

end faster_speed_l970_97040


namespace no_pairs_of_a_and_d_l970_97022

theorem no_pairs_of_a_and_d :
  ∀ (a d : ℝ), (∀ (x y: ℝ), 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0) -> False :=
by 
  sorry

end no_pairs_of_a_and_d_l970_97022


namespace johns_percentage_increase_l970_97008

theorem johns_percentage_increase (original_amount new_amount : ℕ) (h₀ : original_amount = 30) (h₁ : new_amount = 40) :
  (new_amount - original_amount) * 100 / original_amount = 33 :=
by
  sorry

end johns_percentage_increase_l970_97008


namespace problem_correctness_l970_97005

theorem problem_correctness
  (correlation_A : ℝ)
  (correlation_B : ℝ)
  (chi_squared : ℝ)
  (P_chi_squared_5_024 : ℝ)
  (P_chi_squared_6_635 : ℝ)
  (P_X_leq_2 : ℝ)
  (P_X_lt_0 : ℝ) :
  correlation_A = 0.66 →
  correlation_B = -0.85 →
  chi_squared = 6.352 →
  P_chi_squared_5_024 = 0.025 →
  P_chi_squared_6_635 = 0.01 →
  P_X_leq_2 = 0.68 →
  P_X_lt_0 = 0.32 →
  (abs correlation_B > abs correlation_A) ∧
  (1 - P_chi_squared_5_024 < 0.99) ∧
  (P_X_lt_0 = 1 - P_X_leq_2) ∧
  (false) := sorry

end problem_correctness_l970_97005


namespace mary_score_is_95_l970_97091

theorem mary_score_is_95
  (s c w : ℕ)
  (h1 : s > 90)
  (h2 : s = 35 + 5 * c - w)
  (h3 : c + w = 30)
  (h4 : ∀ c' w', s = 35 + 5 * c' - w' → c + w = c' + w' → (c', w') = (c, w)) :
  s = 95 :=
by
  sorry

end mary_score_is_95_l970_97091


namespace reduced_price_is_3_84_l970_97055

noncomputable def reduced_price_per_dozen (original_price : ℝ) (bananas_for_40 : ℕ) : ℝ := 
  let reduced_price := 0.6 * original_price
  let total_bananas := bananas_for_40 + 50
  let price_per_banana := 40 / total_bananas
  12 * price_per_banana

theorem reduced_price_is_3_84 
  (original_price : ℝ) 
  (bananas_for_40 : ℕ) 
  (h₁ : 40 = bananas_for_40 * original_price) 
  (h₂ : bananas_for_40 = 75) 
    : reduced_price_per_dozen original_price bananas_for_40 = 3.84 :=
sorry

end reduced_price_is_3_84_l970_97055


namespace math_problem_l970_97041

variable (x : ℕ)
variable (h : x + 7 = 27)

theorem math_problem : (x = 20) ∧ (((x / 5) + 5) * 7 = 63) :=
by
  have h1 : x = 20 := by {
    -- x can be solved here using the condition, but we use sorry to skip computation.
    sorry
  }
  have h2 : (((x / 5) + 5) * 7 = 63) := by {
    -- The second part result can be computed using the derived x value, but we use sorry to skip computation.
    sorry
  }
  exact ⟨h1, h2⟩

end math_problem_l970_97041


namespace least_common_multiple_of_812_and_3214_is_correct_l970_97067

def lcm_812_3214 : ℕ :=
  Nat.lcm 812 3214

theorem least_common_multiple_of_812_and_3214_is_correct :
  lcm_812_3214 = 1304124 := by
  sorry

end least_common_multiple_of_812_and_3214_is_correct_l970_97067


namespace sum_of_squares_of_real_solutions_l970_97089

theorem sum_of_squares_of_real_solutions :
  (∀ x : ℝ, |x^2 - 3 * x + 1 / 400| = 1 / 400)
  → ((0^2 : ℝ) + 3^2 + (9 - 1 / 100) = 999 / 100) := sorry

end sum_of_squares_of_real_solutions_l970_97089


namespace goldfish_equal_number_after_n_months_l970_97042

theorem goldfish_equal_number_after_n_months :
  ∃ (n : ℕ), 2 * 4^n = 162 * 3^n ∧ n = 6 :=
by
  sorry

end goldfish_equal_number_after_n_months_l970_97042


namespace simplify_expression_l970_97064

variable (a b c x : ℝ)

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def p (x a b c : ℝ) : ℝ :=
  (x - a)^3/(a - b)*(a - c) + a*x +
  (x - b)^3/(b - a)*(b - c) + b*x +
  (x - c)^3/(c - a)*(c - b) + c*x

theorem simplify_expression (h : distinct a b c) :
  p x a b c = a + b + c + 3*x + 1 := by
  sorry

end simplify_expression_l970_97064


namespace fermat_large_prime_solution_l970_97079

theorem fermat_large_prime_solution (n : ℕ) (hn : n > 0) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (x y z : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^n + y^n ≡ z^n [ZMOD p]) :=
sorry

end fermat_large_prime_solution_l970_97079


namespace modulus_sum_complex_l970_97002

theorem modulus_sum_complex :
  let z1 : Complex := Complex.mk 3 (-8)
  let z2 : Complex := Complex.mk 4 6
  Complex.abs (z1 + z2) = Real.sqrt 53 := by
  sorry

end modulus_sum_complex_l970_97002


namespace difference_in_areas_l970_97097

def S1 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 2 + Real.log (x + y) / Real.log 2

def S2 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 3 + Real.log (x + y) / Real.log 2

theorem difference_in_areas : 
  let area_S1 := π * 1 ^ 2
  let area_S2 := π * (Real.sqrt 13) ^ 2
  area_S2 - area_S1 = 12 * π :=
by
  sorry

end difference_in_areas_l970_97097


namespace min_m_for_four_elements_l970_97028

open Set

theorem min_m_for_four_elements (n : ℕ) (hn : n ≥ 2) :
  ∃ m, m = 2 * n + 2 ∧ 
  (∀ (S : Finset ℕ), S.card = m → 
    (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a = b + c + d)) :=
by
  sorry

end min_m_for_four_elements_l970_97028


namespace weight_ratio_l970_97069

variable (J : ℕ) (T : ℕ) (L : ℕ) (S : ℕ)

theorem weight_ratio (h_jake_weight : J = 152) (h_total_weight : J + S = 212) (h_weight_loss : L = 32) :
    (J - L) / (T - J) = 2 :=
by
  sorry

end weight_ratio_l970_97069


namespace least_number_subtracted_divisible_l970_97052

theorem least_number_subtracted_divisible (n : ℕ) (divisor : ℕ) (rem : ℕ) :
  n = 427398 → divisor = 15 → n % divisor = rem → rem = 3 → ∃ k : ℕ, n - k = 427395 :=
by
  intros
  use 3
  sorry

end least_number_subtracted_divisible_l970_97052


namespace square_center_sum_l970_97056

noncomputable def sum_of_center_coordinates (A B C D : ℝ × ℝ) : ℝ :=
  let center : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  center.1 + center.2

theorem square_center_sum
  (A B C D : ℝ × ℝ)
  (h1 : 9 = A.1) (h2 : 0 = A.2)
  (h3 : 4 = B.1) (h4 : 0 = B.2)
  (h5 : 0 = C.1) (h6 : 3 = C.2)
  (h7: A.1 < B.1) (h8: A.2 < C.2) :
  sum_of_center_coordinates A B C D = 8 := 
by
  sorry

end square_center_sum_l970_97056


namespace num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l970_97003

theorem num_three_digit_numbers_divisible_by_5_and_6_with_digit_6 : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (6 ∈ n.digits 10)) ∧ S.card = 6 :=
by
  sorry

end num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l970_97003


namespace trajectory_of_M_l970_97098

-- Define the conditions: P moves on the circle, and Q is fixed
variable (P Q M : ℝ × ℝ)
variable (P_moves_on_circle : P.1^2 + P.2^2 = 1)
variable (Q_fixed : Q = (3, 0))
variable (M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))

-- Theorem statement
theorem trajectory_of_M :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_of_M_l970_97098


namespace even_factors_count_l970_97074

theorem even_factors_count (n : ℕ) (h : n = 2^4 * 3^2 * 5 * 7) : 
  ∃ k : ℕ, k = 48 ∧ ∃ a b c d : ℕ, 
  1 ≤ a ∧ a ≤ 4 ∧
  0 ≤ b ∧ b ≤ 2 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  0 ≤ d ∧ d ≤ 1 ∧
  k = (4 - 1 + 1) * (2 + 1) * (1 + 1) * (1 + 1) := by
  sorry

end even_factors_count_l970_97074


namespace john_multiple_is_correct_l970_97065

noncomputable def compute_multiple (cost_per_computer : ℝ) 
                                   (num_computers : ℕ)
                                   (rent : ℝ)
                                   (non_rent_expenses : ℝ)
                                   (profit : ℝ) : ℝ :=
  let total_revenue := (num_computers : ℝ) * cost_per_computer
  let total_expenses := (num_computers : ℝ) * 800 + rent + non_rent_expenses
  let x := (total_expenses + profit) / total_revenue
  x

theorem john_multiple_is_correct :
  compute_multiple 800 60 5000 3000 11200 = 1.4 := by
  sorry

end john_multiple_is_correct_l970_97065


namespace jenny_eggs_per_basket_l970_97039

theorem jenny_eggs_per_basket :
  ∃ n, (30 % n = 0 ∧ 42 % n = 0 ∧ 18 % n = 0 ∧ n >= 6) → n = 6 :=
by
  sorry

end jenny_eggs_per_basket_l970_97039


namespace initial_persons_count_l970_97021

open Real

def average_weight_increase (n : ℕ) (increase_per_person : ℝ) : ℝ :=
  increase_per_person * n

def weight_difference (new_weight old_weight : ℝ) : ℝ :=
  new_weight - old_weight

theorem initial_persons_count :
  ∀ (n : ℕ),
  average_weight_increase n 2.5 = weight_difference 95 75 → n = 8 :=
by
  intro n h
  sorry

end initial_persons_count_l970_97021


namespace profit_percent_l970_97075

theorem profit_percent (CP SP : ℤ) (h : CP/SP = 2/3) : (SP - CP) * 100 / CP = 50 := 
by
  sorry

end profit_percent_l970_97075


namespace arrange_COMMUNICATION_l970_97078

theorem arrange_COMMUNICATION : 
  let n := 12
  let o_count := 2
  let i_count := 2
  let n_count := 2
  let m_count := 2
  let total_repeats := o_count * i_count * n_count * m_count
  n.factorial / (o_count.factorial * i_count.factorial * n_count.factorial * m_count.factorial) = 29937600 :=
by sorry

end arrange_COMMUNICATION_l970_97078


namespace sum_of_divisors_5_cubed_l970_97073

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end sum_of_divisors_5_cubed_l970_97073


namespace total_shells_l970_97061

theorem total_shells :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let ed_scallop_shells := 3
  let jacob_more_shells := 2
  let marissa_limpet_shells := 5
  let marissa_oyster_shells := 6
  let marissa_conch_shells := 3
  let marissa_scallop_shells := 1
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + ed_scallop_shells
  let jacob_shells := ed_shells + jacob_more_shells
  let marissa_shells := marissa_limpet_shells + marissa_oyster_shells + marissa_conch_shells + marissa_scallop_shells
  let shells_at_beach := ed_shells + jacob_shells + marissa_shells
  let total_shells := shells_at_beach + initial_shells
  total_shells = 51 := by
  sorry

end total_shells_l970_97061


namespace solve_quadratic_eq_l970_97006

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 → x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l970_97006


namespace triangles_not_necessarily_congruent_l970_97068

-- Define the triangles and their properties
structure Triangle :=
  (A B C : ℝ)

-- Define angles and measures for heights and medians
def angle (t : Triangle) : ℝ := sorry
def height_from (t : Triangle) (v : ℝ) : ℝ := sorry
def median_from (t : Triangle) (v : ℝ) : ℝ := sorry

theorem triangles_not_necessarily_congruent
  (T₁ T₂ : Triangle)
  (h_angle : angle T₁ = angle T₂)
  (h_height : height_from T₁ T₁.B = height_from T₂ T₂.B)
  (h_median : median_from T₁ T₁.C = median_from T₂ T₂.C) :
  ¬ (T₁ = T₂) := 
sorry

end triangles_not_necessarily_congruent_l970_97068


namespace scientific_notation_of_1_5_million_l970_97045

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end scientific_notation_of_1_5_million_l970_97045


namespace smallest_constant_c_l970_97025

def satisfies_conditions (f : ℝ → ℝ) :=
  ∀ ⦃x : ℝ⦄, (0 ≤ x ∧ x ≤ 1) → (f x ≥ 0 ∧ (x = 1 → f 1 = 1) ∧
  (∀ y, 0 ≤ y → y ≤ 1 → x + y ≤ 1 → f x + f y ≤ f (x + y)))

theorem smallest_constant_c :
  ∀ {f : ℝ → ℝ},
  satisfies_conditions f →
  ∃ c : ℝ, (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x) ∧
  (∀ c', c' < 2 → ∃ x, 0 ≤ x → x ≤ 1 ∧ f x > c' * x) :=
by sorry

end smallest_constant_c_l970_97025


namespace circle_equation_value_l970_97063

theorem circle_equation_value (a : ℝ) :
  (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 → False) → a = -1 :=
by
  intros h
  sorry

end circle_equation_value_l970_97063


namespace option_a_option_b_option_c_option_d_l970_97012

open Real

theorem option_a (x : ℝ) (h1 : 0 < x) (h2 : x < π) : x > sin x :=
sorry

theorem option_b (x : ℝ) (h : 0 < x) : ¬ (1 - (1 / x) > log x) :=
sorry

theorem option_c (x : ℝ) : (x + 1) * exp x >= -1 / (exp 2) :=
sorry

theorem option_d : ¬ (∀ x : ℝ, x^2 > - (1 / x)) :=
sorry

end option_a_option_b_option_c_option_d_l970_97012


namespace unique_peg_placement_l970_97004

theorem unique_peg_placement :
  ∃! f : Fin 6 → Fin 6 → Option (Fin 6), ∀ i j k, 
    (∃ c, f i k = some c) →
    (∃ c, f j k = some c) →
    i = j ∧ match f i j with
    | some c => f j k ≠ some c
    | none => True :=
  sorry

end unique_peg_placement_l970_97004


namespace coin_probability_l970_97046

theorem coin_probability :
  let value_quarters : ℚ := 15.00
  let value_nickels : ℚ := 15.00
  let value_dimes : ℚ := 10.00
  let value_pennies : ℚ := 5.00
  let number_quarters := value_quarters / 0.25
  let number_nickels := value_nickels / 0.05
  let number_dimes := value_dimes / 0.10
  let number_pennies := value_pennies / 0.01
  let total_coins := number_quarters + number_nickels + number_dimes + number_pennies
  let probability := (number_quarters + number_dimes) / total_coins
  probability = (1 / 6) := by 
sorry

end coin_probability_l970_97046


namespace total_price_of_25_shirts_l970_97024

theorem total_price_of_25_shirts (S W : ℝ) (H1 : W = S + 4) (H2 : 75 * W = 1500) : 
  25 * S = 400 :=
by
  -- Proof would go here
  sorry

end total_price_of_25_shirts_l970_97024


namespace poly_not_33_l970_97076

theorem poly_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by sorry

end poly_not_33_l970_97076


namespace more_radishes_correct_l970_97099

def total_radishes : ℕ := 88
def radishes_first_basket : ℕ := 37

def more_radishes_in_second_basket := total_radishes - radishes_first_basket - radishes_first_basket

theorem more_radishes_correct : more_radishes_in_second_basket = 14 :=
by
  sorry

end more_radishes_correct_l970_97099


namespace binom_12_6_l970_97035

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l970_97035


namespace arithmetic_sequence_properties_l970_97009

theorem arithmetic_sequence_properties
    (n s1 s2 s3 : ℝ)
    (h1 : s1 = 8)
    (h2 : s2 = 50)
    (h3 : s3 = 134)
    (h4 : n = 8) :
    n^2 * s3 - 3 * n * s1 * s2 + 2 * s1^2 = 0 := 
by {
  sorry
}

end arithmetic_sequence_properties_l970_97009


namespace perfect_square_trinomial_m_l970_97092

theorem perfect_square_trinomial_m (m : ℝ) :
  (∀ x : ℝ, ∃ b : ℝ, x^2 + 2 * (m - 3) * x + 16 = (1 * x + b)^2) → (m = 7 ∨ m = -1) :=
by 
  intro h
  sorry

end perfect_square_trinomial_m_l970_97092


namespace preferred_dividend_rate_l970_97054

noncomputable def dividend_rate_on_preferred_shares
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℕ)
  (semi_annual_dividend_common : ℚ)
  (total_annual_dividend : ℚ)
  (dividend_rate_preferred : ℚ) : Prop :=
  preferred_shares * par_value * (dividend_rate_preferred / 100) +
  2 * (common_shares * par_value * (semi_annual_dividend_common / 100)) =
  total_annual_dividend

theorem preferred_dividend_rate
  (h1 : 1200 = 1200)
  (h2 : 3000 = 3000)
  (h3 : 50 = 50)
  (h4 : 3.5 = 3.5)
  (h5 : 16500 = 16500) :
  dividend_rate_on_preferred_shares 1200 3000 50 3.5 16500 10 :=
by sorry

end preferred_dividend_rate_l970_97054


namespace smallest_m_4_and_n_229_l970_97000

def satisfies_condition (m n : ℕ) : Prop :=
  19 * m + 8 * n = 1908

def is_smallest_m (m n : ℕ) : Prop :=
  ∀ m' n', satisfies_condition m' n' → m' > 0 → n' > 0 → m ≤ m'

theorem smallest_m_4_and_n_229 : ∃ (m n : ℕ), satisfies_condition m n ∧ is_smallest_m m n ∧ m = 4 ∧ n = 229 :=
by
  sorry

end smallest_m_4_and_n_229_l970_97000


namespace inequality_l970_97086

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 3) :
  1 / (4 - a^2) + 1 / (4 - b^2) + 1 / (4 - c^2) ≤ 9 / (a + b + c)^2 :=
by
  sorry

end inequality_l970_97086


namespace James_distance_ridden_l970_97014

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l970_97014


namespace max_small_packages_l970_97031

theorem max_small_packages (L S : ℝ) (W : ℝ) (h1 : W = 12 * L) (h2 : W = 20 * S) :
  (∃ n_smalls, n_smalls = 5 ∧ W - 9 * L = n_smalls * S) :=
by
  sorry

end max_small_packages_l970_97031


namespace total_weekly_sleep_correct_l970_97057

-- Definition of the weekly sleep time for cougar, zebra, and lion
def cougar_sleep_even_days : Nat := 4
def cougar_sleep_odd_days : Nat := 6
def zebra_sleep_even_days := (cougar_sleep_even_days + 2)
def zebra_sleep_odd_days := (cougar_sleep_odd_days + 2)
def lion_sleep_even_days := (zebra_sleep_even_days - 3)
def lion_sleep_odd_days := (cougar_sleep_odd_days + 1)

def total_weekly_sleep_time : Nat :=
  (4 * cougar_sleep_odd_days + 3 * cougar_sleep_even_days) + -- Cougar's total sleep in a week
  (4 * zebra_sleep_odd_days + 3 * zebra_sleep_even_days) + -- Zebra's total sleep in a week
  (4 * lion_sleep_odd_days + 3 * lion_sleep_even_days) -- Lion's total sleep in a week

theorem total_weekly_sleep_correct : total_weekly_sleep_time = 123 := 
by
  -- Total for the week according to given conditions
  sorry -- Proof is omitted, only the statement is required

end total_weekly_sleep_correct_l970_97057


namespace martin_total_distance_l970_97095

noncomputable def calculate_distance_traveled : ℕ :=
  let segment1 := 70 * 3 -- 210 km
  let segment2 := 80 * 4 -- 320 km
  let segment3 := 65 * 3 -- 195 km
  let segment4 := 50 * 2 -- 100 km
  let segment5 := 90 * 4 -- 360 km
  segment1 + segment2 + segment3 + segment4 + segment5

theorem martin_total_distance : calculate_distance_traveled = 1185 :=
by
  sorry

end martin_total_distance_l970_97095


namespace intersection_M_N_l970_97058

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l970_97058


namespace number_of_belts_l970_97051

def ties := 34
def black_shirts := 63
def white_shirts := 42

def jeans := (2 / 3 : ℚ) * (black_shirts + white_shirts)
def scarves (B : ℚ) := (1 / 2 : ℚ) * (ties + B)

theorem number_of_belts (B : ℚ) : jeans = scarves B + 33 → B = 40 := by
  -- This theorem states the required proof but leaves the proof itself as a placeholder.
  -- The proof would involve solving equations algebraically as shown in the solution steps.
  sorry

end number_of_belts_l970_97051


namespace correct_operation_l970_97007

theorem correct_operation : 
  (a^2 + a^2 = 2 * a^2) = false ∧ 
  ((-3 * a * b^2)^2 = -6 * a^2 * b^4) = false ∧ 
  (a^6 / (-a)^2 = a^4) = true ∧ 
  ((a - b)^2 = a^2 - b^2) = false :=
sorry

end correct_operation_l970_97007


namespace mathematician_daily_questions_l970_97070

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end mathematician_daily_questions_l970_97070


namespace flour_needed_l970_97082

theorem flour_needed (cookies : ℕ) (flour : ℕ) (k : ℕ) (f_whole_wheat f_all_purpose : ℕ) 
  (h : cookies = 45) (h1 : flour = 3) (h2 : k = 90) (h3 : (k / 2) = 45) 
  (h4 : f_all_purpose = (flour * (k / cookies)) / 2) 
  (h5 : f_whole_wheat = (flour * (k / cookies)) / 2) : 
  f_all_purpose = 3 ∧ f_whole_wheat = 3 := 
by
  sorry

end flour_needed_l970_97082


namespace arithmetic_sequence_sum_ratio_l970_97019

theorem arithmetic_sequence_sum_ratio 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (a : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (a_n_def : ∀ n, a_n n = a + (n - 1) * d) 
  (S_n_def : ∀ n, S_n n = n * (2 * a + (n - 1) * d) / 2) 
  (h : 3 * (a + 4 * d) = 5 * (a + 2 * d)) : 
  S_n 5 / S_n 3 = 5 / 2 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l970_97019


namespace range_of_expression_l970_97034

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  - π / 6 < 2 * α - β / 2 ∧ 2 * α - β / 2 < π :=
sorry

end range_of_expression_l970_97034


namespace fixed_point_is_one_three_l970_97090

noncomputable def fixed_point_of_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : ℝ × ℝ :=
  (1, 3)

theorem fixed_point_is_one_three {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  fixed_point_of_function a h_pos h_ne_one = (1, 3) :=
  sorry

end fixed_point_is_one_three_l970_97090


namespace radius_of_circumscribed_circle_l970_97027

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l970_97027


namespace elephant_weight_equivalence_l970_97050

variable (y : ℝ)
variable (porter_weight : ℝ := 120)
variable (blocks_1 : ℝ := 20)
variable (blocks_2 : ℝ := 21)
variable (porters_1 : ℝ := 3)
variable (porters_2 : ℝ := 1)

theorem elephant_weight_equivalence :
  (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 := 
sorry

end elephant_weight_equivalence_l970_97050


namespace sixth_year_fee_l970_97081

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l970_97081


namespace max_min_sum_eq_two_l970_97038

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 + Real.sqrt 2 * Real.sin (x + Real.pi / 4)) / (2 * x ^ 2 + Real.cos x)

theorem max_min_sum_eq_two (a b : ℝ) (h_max : ∀ x, f x ≤ a) (h_min : ∀ x, b ≤ f x) (h_max_val : ∃ x, f x = a) (h_min_val : ∃ x, f x = b) :
  a + b = 2 := 
sorry

end max_min_sum_eq_two_l970_97038


namespace max_value_expression_l970_97071

noncomputable def a (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def b (φ : ℝ) : ℝ := 3 * Real.sin φ

theorem max_value_expression (φ θ : ℝ) : 
  ∃ c : ℝ, c = 3 * Real.cos (θ - φ) ∧ c ≤ 3 := by
  sorry

end max_value_expression_l970_97071


namespace work_problem_l970_97026

/-- 
  Suppose A can complete a work in \( x \) days alone, 
  B can complete the work in 20 days,
  and together they work for 7 days, leaving a fraction of 0.18333333333333335 of the work unfinished.
  Prove that \( x = 15 \).
 -/
theorem work_problem (x : ℝ) : 
  (∀ (B : ℝ), B = 20 → (∀ (f : ℝ), f = 0.18333333333333335 → (7 * (1 / x + 1 / B) = 1 - f)) → x = 15) := 
sorry

end work_problem_l970_97026


namespace perfect_square_divisors_count_l970_97087

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : Nat := factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 *
                                   factorial 6 * factorial 7 * factorial 8 * factorial 9 * factorial 10

def count_perfect_square_divisors (n : Nat) : Nat := sorry -- This would involve the correct function implementation.

theorem perfect_square_divisors_count :
  count_perfect_square_divisors product_of_factorials = 2160 :=
sorry

end perfect_square_divisors_count_l970_97087


namespace cubic_sum_l970_97062

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l970_97062


namespace school_total_payment_l970_97053

def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def chaperones_per_class : ℕ := 5
def student_fee : ℝ := 5.50
def adult_fee : ℝ := 6.50

def total_students : ℕ := num_classes * students_per_class
def total_adults : ℕ := num_classes * chaperones_per_class

def total_student_cost : ℝ := total_students * student_fee
def total_adult_cost : ℝ := total_adults * adult_fee

def total_cost : ℝ := total_student_cost + total_adult_cost

theorem school_total_payment : total_cost = 1010.0 := by
  sorry

end school_total_payment_l970_97053


namespace quadratic_real_roots_l970_97094

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_l970_97094


namespace floor_exponents_eq_l970_97015

theorem floor_exponents_eq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_inf_k : ∃ᶠ k in at_top, ∃ (k : ℕ), ⌊a ^ k⌋ + ⌊b ^ k⌋ = ⌊a⌋ ^ k + ⌊b⌋ ^ k) :
  ⌊a ^ 2014⌋ + ⌊b ^ 2014⌋ = ⌊a⌋ ^ 2014 + ⌊b⌋ ^ 2014 := by
  sorry

end floor_exponents_eq_l970_97015


namespace find_number_l970_97017

theorem find_number (x : ℝ) (h : 0.40 * x = 130 + 190) : x = 800 :=
by {
  -- The proof will go here
  sorry
}

end find_number_l970_97017


namespace length_of_EF_l970_97085

theorem length_of_EF (AB BC : ℝ) (DE DF : ℝ) (Area_ABC : ℝ) (Area_DEF : ℝ) (EF : ℝ) 
  (h₁ : AB = 10) (h₂ : BC = 15) (h₃ : DE = DF) (h₄ : Area_DEF = (1/3) * Area_ABC) 
  (h₅ : Area_ABC = AB * BC) (h₆ : Area_DEF = (1/2) * (DE * DF)) : 
  EF = 10 * Real.sqrt 2 := 
by 
  sorry

end length_of_EF_l970_97085


namespace alicia_bought_more_markers_l970_97047

theorem alicia_bought_more_markers (price_per_marker : ℝ) (n_h : ℝ) (n_a : ℝ) (m : ℝ) 
    (h_hector : n_h * price_per_marker = 2.76) 
    (h_alicia : n_a * price_per_marker = 4.07)
    (h_diff : n_a - n_h = m) : 
  m = 13 :=
sorry

end alicia_bought_more_markers_l970_97047


namespace largest_multiple_of_9_less_than_110_l970_97001

theorem largest_multiple_of_9_less_than_110 : ∃ x, (x < 110 ∧ x % 9 = 0 ∧ ∀ y, (y < 110 ∧ y % 9 = 0) → y ≤ x) ∧ x = 108 :=
by
  sorry

end largest_multiple_of_9_less_than_110_l970_97001


namespace curve_is_circle_l970_97010

theorem curve_is_circle (r θ : ℝ) (h : r = 3 * Real.sin θ) : 
  ∃ c : ℝ × ℝ, c = (0, 3 / 2) ∧ ∀ p : ℝ × ℝ, ∃ R : ℝ, R = 3 / 2 ∧ 
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = R^2 :=
sorry

end curve_is_circle_l970_97010


namespace hyperbola_asymptote_eqn_l970_97088

theorem hyperbola_asymptote_eqn :
  ∀ (x y : ℝ),
  (y ^ 2 / 4 - x ^ 2 = 1) → (y = 2 * x ∨ y = -2 * x) := by
sorry

end hyperbola_asymptote_eqn_l970_97088


namespace sum_of_natural_numbers_l970_97049

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  2^(n-1)

theorem sum_of_natural_numbers (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k = number_of_ways n :=
by
  use 2^(n-1)
  sorry

end sum_of_natural_numbers_l970_97049


namespace odd_and_even_inter_empty_l970_97080

-- Define the set of odd numbers
def odd_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define the set of even numbers
def even_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- The theorem stating that the intersection of odd numbers and even numbers is empty
theorem odd_and_even_inter_empty : odd_numbers ∩ even_numbers = ∅ :=
by
  -- placeholder for the proof
  sorry

end odd_and_even_inter_empty_l970_97080


namespace estate_area_is_correct_l970_97060

noncomputable def actual_area_of_estate (length_in_inches : ℕ) (width_in_inches : ℕ) (scale : ℕ) : ℕ :=
  let actual_length := length_in_inches * scale
  let actual_width := width_in_inches * scale
  actual_length * actual_width

theorem estate_area_is_correct :
  actual_area_of_estate 9 6 350 = 6615000 := by
  -- Here, we would provide the proof steps, but for this exercise, we use sorry.
  sorry

end estate_area_is_correct_l970_97060


namespace divisibility_problem_l970_97023

theorem divisibility_problem
  (h1 : 5^3 ∣ 1978^100 - 1)
  (h2 : 10^4 ∣ 3^500 - 1)
  (h3 : 2003 ∣ 2^286 - 1) :
  2^4 * 5^7 * 2003 ∣ (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) :=
by sorry

end divisibility_problem_l970_97023


namespace max_A_l970_97037

noncomputable def A (x y : ℝ) : ℝ :=
  x^4 * y + x * y^4 + x^3 * y + x * y^3 + x^2 * y + x * y^2

theorem max_A (x y : ℝ) (h : x + y = 1) : A x y ≤ 7 / 16 :=
sorry

end max_A_l970_97037
