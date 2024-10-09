import Mathlib

namespace total_number_of_coins_is_15_l1678_167832

theorem total_number_of_coins_is_15 (x : ℕ) (h : 1*x + 5*x + 10*x + 25*x + 50*x = 273) : 5 * x = 15 :=
by {
  -- Proof omitted
  sorry
}

end total_number_of_coins_is_15_l1678_167832


namespace biology_books_needed_l1678_167897

-- Define the problem in Lean
theorem biology_books_needed
  (B P Q R F Z₁ Z₂ : ℕ)
  (b p : ℝ)
  (H1 : B ≠ P)
  (H2 : B ≠ Q)
  (H3 : B ≠ R)
  (H4 : B ≠ F)
  (H5 : P ≠ Q)
  (H6 : P ≠ R)
  (H7 : P ≠ F)
  (H8 : Q ≠ R)
  (H9 : Q ≠ F)
  (H10 : R ≠ F)
  (H11 : 0 < B ∧ 0 < P ∧ 0 < Q ∧ 0 < R ∧ 0 < F)
  (H12 : Bb + Pp = Z₁)
  (H13 : Qb + Rp = Z₂)
  (H14 : Fb = Z₁)
  (H15 : Z₂ < Z₁) :
  F = (Q - B) / (P - R) :=
by
  sorry  -- Proof to be provided

end biology_books_needed_l1678_167897


namespace determine_n_l1678_167816

noncomputable def P : ℤ → ℤ := sorry

theorem determine_n (n : ℕ) (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P x = 2 ∨ P x = 1 ∨ P x = 0)
  (h0 : ∀ k : ℕ, k ≤ n → P (3 * k) = 2)
  (h1 : ∀ k : ℕ, k < n → P (3 * k + 1) = 1)
  (h2 : ∀ k : ℕ, k < n → P (3 * k + 2) = 0)
  (h_f : P (3 * n + 1) = 730) :
  n = 4 := 
sorry

end determine_n_l1678_167816


namespace solve_for_m_l1678_167815

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end solve_for_m_l1678_167815


namespace inequality_transfers_l1678_167846

variables (a b c d : ℝ)

theorem inequality_transfers (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_transfers_l1678_167846


namespace isabel_earnings_l1678_167836

theorem isabel_earnings :
  ∀ (bead_necklaces gem_necklaces cost_per_necklace : ℕ),
    bead_necklaces = 3 →
    gem_necklaces = 3 →
    cost_per_necklace = 6 →
    (bead_necklaces + gem_necklaces) * cost_per_necklace = 36 := by
sorry

end isabel_earnings_l1678_167836


namespace sqrt_expression_evaluation_l1678_167855

theorem sqrt_expression_evaluation (sqrt48 : Real) (sqrt1div3 : Real) 
  (h1 : sqrt48 = 4 * Real.sqrt 3) (h2 : sqrt1div3 = Real.sqrt (1 / 3)) :
  (-1 / 2) * sqrt48 * sqrt1div3 = -2 :=
by 
  rw [h1, h2]
  -- Continue with the simplification steps, however
  sorry

end sqrt_expression_evaluation_l1678_167855


namespace area_of_square_with_perimeter_32_l1678_167821

theorem area_of_square_with_perimeter_32 :
  ∀ (s : ℝ), 4 * s = 32 → s * s = 64 :=
by
  intros s h
  sorry

end area_of_square_with_perimeter_32_l1678_167821


namespace polygon_perimeter_greater_than_2_l1678_167879

-- Definition of the conditions
variable (polygon : Set (ℝ × ℝ))
variable (A B : ℝ × ℝ)
variable (P : ℝ)

axiom point_in_polygon (p : ℝ × ℝ) : p ∈ polygon
axiom A_in_polygon : A ∈ polygon
axiom B_in_polygon : B ∈ polygon
axiom path_length_condition (γ : ℝ → ℝ × ℝ) (γ_in_polygon : ∀ t, γ t ∈ polygon) (hA : γ 0 = A) (hB : γ 1 = B) : ∀ t₁ t₂, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ 1 → dist (γ t₁) (γ t₂) > 1

-- Statement to prove
theorem polygon_perimeter_greater_than_2 : P > 2 :=
sorry

end polygon_perimeter_greater_than_2_l1678_167879


namespace range_of_a_l1678_167834

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end range_of_a_l1678_167834


namespace cubic_polynomial_range_l1678_167833

-- Define the conditions and the goal in Lean
theorem cubic_polynomial_range :
  ∀ x : ℝ, (x^2 - 5 * x + 6 < 0) → (41 < x^3 + 5 * x^2 + 6 * x + 1) ∧ (x^3 + 5 * x^2 + 6 * x + 1 < 91) :=
by
  intros x hx
  have h1 : 2 < x := sorry
  have h2 : x < 3 := sorry
  have h3 : (x^3 + 5 * x^2 + 6 * x + 1) > 41 := sorry
  have h4 : (x^3 + 5 * x^2 + 6 * x + 1) < 91 := sorry
  exact ⟨h3, h4⟩ 

end cubic_polynomial_range_l1678_167833


namespace no_2018_zero_on_curve_l1678_167873

theorem no_2018_zero_on_curve (a c d : ℝ) (hac : a * c > 0) : ¬∃(d : ℝ), (2018 : ℝ) ^ 2 * a + 2018 * c + d = 0 := 
by {
  sorry
}

end no_2018_zero_on_curve_l1678_167873


namespace algebra_expression_never_zero_l1678_167870

theorem algebra_expression_never_zero (x : ℝ) : (1 : ℝ) / (x - 1) ≠ 0 :=
sorry

end algebra_expression_never_zero_l1678_167870


namespace a_equals_2t_squared_l1678_167863

theorem a_equals_2t_squared {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4 * a = b^2) :
  ∃ t : ℕ, 0 < t ∧ a = 2 * t^2 :=
sorry

end a_equals_2t_squared_l1678_167863


namespace quadratic_roots_squared_sum_l1678_167817

theorem quadratic_roots_squared_sum (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 - 2 * n - 1 = 0) : m^2 + n^2 = 6 :=
sorry

end quadratic_roots_squared_sum_l1678_167817


namespace monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l1678_167881

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the problem stating that when a <= 0, f(x) is increasing on (0, +∞)
theorem monotonicity_increasing_when_a_nonpos (a : ℝ) (h : a ≤ 0) :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Define the problem stating that when a > 0, f(x) is increasing on (0, 1/a) and decreasing on (1/a, +∞)
theorem monotonicity_increasing_decreasing_when_a_pos (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x < (1 / a) → y < (1 / a) → f a x < f a y) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (1 / a) < x → (1 / a) < y → f a y < f a x) :=
sorry

-- Define the problem for the range of a such that f(x) < 0 for all x in (0, +∞)
theorem range_of_a_for_f_less_than_zero (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x < 0) ↔ a ∈ Set.Ioi (1 / Real.exp 1) :=
sorry

end monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l1678_167881


namespace height_radius_ratio_l1678_167859

variables (R H V : ℝ) (π : ℝ) (A : ℝ)

-- Given conditions
def volume_condition : Prop := π * R^2 * H = V / 2
def surface_area : ℝ := 2 * π * R^2 + 2 * π * R * H

-- Statement to prove
theorem height_radius_ratio (h_volume : volume_condition R H V π) :
  H / R = 2 := 
sorry

end height_radius_ratio_l1678_167859


namespace neither_sufficient_nor_necessary_l1678_167876

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (h1 : 0 < a * b ∧ a * b < 1) : ¬ (b < 1 / a) ∨ ¬ (1 / a < b) := by
  sorry

end neither_sufficient_nor_necessary_l1678_167876


namespace linear_function_quadrants_l1678_167866

theorem linear_function_quadrants (k b : ℝ) :
  (∀ x, (0 < x → 0 < k * x + b) ∧ (x < 0 → 0 < k * x + b) ∧ (x < 0 → k * x + b < 0)) →
  k > 0 ∧ b > 0 :=
by
  sorry

end linear_function_quadrants_l1678_167866


namespace fifth_power_ends_with_same_digit_l1678_167851

theorem fifth_power_ends_with_same_digit (a : ℕ) : a^5 % 10 = a % 10 :=
by sorry

end fifth_power_ends_with_same_digit_l1678_167851


namespace ratio_of_red_to_blue_marbles_l1678_167871

theorem ratio_of_red_to_blue_marbles:
  ∀ (R B : ℕ), 
    R + B = 30 →
    2 * (20 - B) = 10 →
    B = 15 → 
    R = 15 →
    R / B = 1 :=
by intros R B h₁ h₂ h₃ h₄
   sorry

end ratio_of_red_to_blue_marbles_l1678_167871


namespace initial_walnut_trees_l1678_167812

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end initial_walnut_trees_l1678_167812


namespace construct_all_naturals_starting_from_4_l1678_167857

-- Define the operations f, g, h
def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n  -- h is only meaningful if n is even

-- Main theorem: prove that starting from 4, every natural number can be constructed
theorem construct_all_naturals_starting_from_4 :
  ∀ (n : ℕ), ∃ (k : ℕ), (f^[k] 4 = n ∨ g^[k] 4 = n ∨ h^[k] 4 = n) :=
by sorry


end construct_all_naturals_starting_from_4_l1678_167857


namespace min_calls_correct_l1678_167826

-- Define a function that calculates the minimum number of calls given n people
def min_calls (n : ℕ) : ℕ :=
  2 * n - 2

-- Theorem to prove that min_calls(n) given the conditions is equal to 2n - 2
theorem min_calls_correct (n : ℕ) (h : n ≥ 2) : min_calls n = 2 * n - 2 :=
by
  sorry

end min_calls_correct_l1678_167826


namespace verify_extrema_l1678_167845

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema_l1678_167845


namespace price_of_72_cans_is_18_36_l1678_167856

def regular_price_per_can : ℝ := 0.30
def discount_percent : ℝ := 0.15
def number_of_cans : ℝ := 72

def discounted_price_per_can : ℝ := regular_price_per_can - (discount_percent * regular_price_per_can)
def total_price (num_cans : ℝ) : ℝ := num_cans * discounted_price_per_can

theorem price_of_72_cans_is_18_36 :
  total_price number_of_cans = 18.36 :=
by
  /- Proof details omitted -/
  sorry

end price_of_72_cans_is_18_36_l1678_167856


namespace arith_geo_mean_extended_arith_geo_mean_l1678_167802
noncomputable section

open Real

-- Definition for Problem 1
def arith_geo_mean_inequality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : Prop :=
  (a + b) / 2 ≥ Real.sqrt (a * b)

-- Theorem for Problem 1
theorem arith_geo_mean (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : arith_geo_mean_inequality a b h1 h2 :=
  sorry

-- Definition for Problem 2
def extended_arith_geo_mean_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c

-- Theorem for Problem 2
theorem extended_arith_geo_mean (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : extended_arith_geo_mean_inequality a b c h1 h2 h3 :=
  sorry

end arith_geo_mean_extended_arith_geo_mean_l1678_167802


namespace man_speed_l1678_167892

theorem man_speed (time_in_minutes : ℕ) (distance_in_km : ℕ) 
  (h_time : time_in_minutes = 30) 
  (h_distance : distance_in_km = 5) : 
  (distance_in_km : ℝ) / (time_in_minutes / 60 : ℝ) = 10 :=
by 
  sorry

end man_speed_l1678_167892


namespace additional_distance_sam_runs_more_than_sarah_l1678_167824

theorem additional_distance_sam_runs_more_than_sarah
  (street_width : ℝ) (block_side_length : ℝ)
  (h1 : street_width = 30) (h2 : block_side_length = 500) :
  let P_Sarah := 4 * block_side_length
  let P_Sam := 4 * (block_side_length + 2 * street_width)
  P_Sam - P_Sarah = 240 :=
by
  sorry

end additional_distance_sam_runs_more_than_sarah_l1678_167824


namespace range_of_a_for_zero_l1678_167804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_zero (a : ℝ) : a ≤ 2 * Real.log 2 - 2 → ∃ x : ℝ, f a x = 0 := by
  sorry

end range_of_a_for_zero_l1678_167804


namespace imaginary_part_of_z_l1678_167838

namespace ComplexNumberProof

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number
def z : ℂ := i^2 * (1 + i)

-- Prove the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
    -- Proof goes here
    sorry

end ComplexNumberProof

end imaginary_part_of_z_l1678_167838


namespace cyclist_average_speed_l1678_167830

noncomputable def total_distance : ℝ := 10 + 5 + 15 + 20 + 30
noncomputable def time_first_segment : ℝ := 10 / 12
noncomputable def time_second_segment : ℝ := 5 / 6
noncomputable def time_third_segment : ℝ := 15 / 16
noncomputable def time_fourth_segment : ℝ := 20 / 14
noncomputable def time_fifth_segment : ℝ := 30 / 20

noncomputable def total_time : ℝ := time_first_segment + time_second_segment + time_third_segment + time_fourth_segment + time_fifth_segment

noncomputable def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed = 12.93 := by
  sorry

end cyclist_average_speed_l1678_167830


namespace total_cost_ice_cream_l1678_167850

noncomputable def price_Chocolate : ℝ := 2.50
noncomputable def price_Vanilla : ℝ := 2.00
noncomputable def price_Strawberry : ℝ := 2.25
noncomputable def price_Mint : ℝ := 2.20
noncomputable def price_WaffleCone : ℝ := 1.50
noncomputable def price_ChocolateChips : ℝ := 1.00
noncomputable def price_Fudge : ℝ := 1.25
noncomputable def price_WhippedCream : ℝ := 0.75

def scoops_Pierre : ℕ := 3  -- 2 scoops Chocolate + 1 scoop Mint
def scoops_Mother : ℕ := 4  -- 2 scoops Vanilla + 1 scoop Strawberry + 1 scoop Mint

noncomputable def price_Pierre_BeforeOffer : ℝ :=
  2 * price_Chocolate + price_Mint + price_WaffleCone + price_ChocolateChips

noncomputable def free_Pierre : ℝ := price_Mint -- Mint is the cheapest among Pierre's choices

noncomputable def price_Pierre_AfterOffer : ℝ := price_Pierre_BeforeOffer - free_Pierre

noncomputable def price_Mother_BeforeOffer : ℝ :=
  2 * price_Vanilla + price_Strawberry + price_Mint + price_WaffleCone + price_Fudge + price_WhippedCream

noncomputable def free_Mother : ℝ := price_Vanilla -- Vanilla is the cheapest among Mother's choices

noncomputable def price_Mother_AfterOffer : ℝ := price_Mother_BeforeOffer - free_Mother

noncomputable def total_BeforeDiscount : ℝ := price_Pierre_AfterOffer + price_Mother_AfterOffer

noncomputable def discount_Amount : ℝ := total_BeforeDiscount * 0.15

noncomputable def total_AfterDiscount : ℝ := total_BeforeDiscount - discount_Amount

theorem total_cost_ice_cream : total_AfterDiscount = 14.83 := by
  sorry


end total_cost_ice_cream_l1678_167850


namespace final_weight_is_sixteen_l1678_167875

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l1678_167875


namespace find_value_l1678_167839

theorem find_value (x y : ℝ) (h : x - 2 * y = 1) : 3 - 4 * y + 2 * x = 5 := sorry

end find_value_l1678_167839


namespace problem_statement_l1678_167865

theorem problem_statement (a : ℝ) (h : a^2 - 2 * a + 1 = 0) : 4 * a - 2 * a^2 + 2 = 4 := 
sorry

end problem_statement_l1678_167865


namespace problem_f_of_3_l1678_167800

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem problem_f_of_3 : f (f 3) = 10 := by
  sorry

end problem_f_of_3_l1678_167800


namespace find_constants_l1678_167822

theorem find_constants : 
  ∃ (a b : ℝ), a • (⟨1, 4⟩ : ℝ × ℝ) + b • (⟨3, -2⟩ : ℝ × ℝ) = (⟨5, 6⟩ : ℝ × ℝ) ∧ a = 2 ∧ b = 1 :=
by 
  sorry

end find_constants_l1678_167822


namespace range_of_q_eq_eight_inf_l1678_167819

noncomputable def q (x : ℝ) : ℝ := (x^2 + 2)^3

theorem range_of_q_eq_eight_inf (x : ℝ) : 0 ≤ x → ∃ y, y = q x ∧ 8 ≤ y := sorry

end range_of_q_eq_eight_inf_l1678_167819


namespace polynomial_roots_l1678_167858

noncomputable def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem polynomial_roots:
  ∃ (a b c d : ℝ), a = -3 ∧ b = -1 ∧ c = -1 ∧ d = 2 ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) :=
sorry

end polynomial_roots_l1678_167858


namespace next_equalities_from_conditions_l1678_167848

-- Definitions of the equality conditions
def eq1 : Prop := 3^2 + 4^2 = 5^2
def eq2 : Prop := 10^2 + 11^2 + 12^2 = 13^2 + 14^2
def eq3 : Prop := 21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2
def eq4 : Prop := 36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2

-- The next equalities we want to prove
def eq5 : Prop := 55^2 + 56^2 + 57^2 + 58^2 + 59^2 + 60^2 = 61^2 + 62^2 + 63^2 + 64^2 + 65^2
def eq6 : Prop := 78^2 + 79^2 + 80^2 + 81^2 + 82^2 + 83^2 + 84^2 = 85^2 + 86^2 + 87^2 + 88^2 + 89^2 + 90^2

theorem next_equalities_from_conditions : eq1 → eq2 → eq3 → eq4 → (eq5 ∧ eq6) :=
by
  sorry

end next_equalities_from_conditions_l1678_167848


namespace largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l1678_167854

-- Define the sequence and its cyclic property
def cyclicSequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 4) = 1000 * (seq n % 10) + 100 * (seq (n + 1) % 10) + 10 * (seq (n + 2) % 10) + (seq (n + 3) % 10)

-- Define the property of T being the sum of the sequence
def sumOfSequence (seq : ℕ → ℕ) (T : ℕ) : Prop :=
  T = seq 0 + seq 1 + seq 2 + seq 3

-- Define the statement that T is always divisible by 101
theorem largest_prime_divisor_of_sum_of_cyclic_sequence_is_101
  (seq : ℕ → ℕ) (T : ℕ)
  (h1 : cyclicSequence seq)
  (h2 : sumOfSequence seq T) :
  (101 ∣ T) := 
sorry

end largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l1678_167854


namespace abs_inequality_l1678_167829

theorem abs_inequality (x y : ℝ) (h1 : |x| < 2) (h2 : |y| < 2) : |4 - x * y| > 2 * |x - y| :=
by
  sorry

end abs_inequality_l1678_167829


namespace find_first_day_speed_l1678_167828

theorem find_first_day_speed (t : ℝ) (d : ℝ) (v : ℝ) (h1 : d = 2.5) 
  (h2 : v * (t - 7/60) = d) (h3 : 10 * (t - 8/60) = d) : v = 9.375 :=
by {
  -- Proof omitted for brevity
  sorry
}

end find_first_day_speed_l1678_167828


namespace quadratic_has_equal_roots_l1678_167835

theorem quadratic_has_equal_roots (b : ℝ) (h : ∃ x : ℝ, b*x^2 + 2*b*x + 4 = 0 ∧ b*x^2 + 2*b*x + 4 = 0) :
  b = 4 :=
sorry

end quadratic_has_equal_roots_l1678_167835


namespace probability_of_earning_exactly_2300_in_3_spins_l1678_167840

-- Definitions of the conditions
def spinner_sections : List ℕ := [0, 1000, 200, 7000, 300]
def equal_area_sections : Prop := true  -- Each section has the same area, simple condition

-- Proving the probability of earning exactly $2300 in three spins
theorem probability_of_earning_exactly_2300_in_3_spins :
  ∃ p : ℚ, p = 3 / 125 := sorry

end probability_of_earning_exactly_2300_in_3_spins_l1678_167840


namespace reading_homework_is_4_l1678_167867

-- Defining the conditions.
variables (R : ℕ)  -- Number of pages of reading homework
variables (M : ℕ)  -- Number of pages of math homework

-- Rachel has 7 pages of math homework.
def math_homework_equals_7 : Prop := M = 7

-- Rachel has 3 more pages of math homework than reading homework.
def math_minus_reads_is_3 : Prop := M = R + 3

-- Prove the number of pages of reading homework is 4.
theorem reading_homework_is_4 (M R : ℕ) 
  (h1 : math_homework_equals_7 M) -- M = 7
  (h2 : math_minus_reads_is_3 M R) -- M = R + 3
  : R = 4 :=
sorry

end reading_homework_is_4_l1678_167867


namespace part_a_part_b_part_c_part_d_l1678_167877

open Nat

theorem part_a (y z : ℕ) (hy : 0 < y) (hz : 0 < z) : 
  (1 = 1 / y + 1 / z) ↔ (y = 2 ∧ z = 1) := 
by 
  sorry

theorem part_b (y z : ℕ) (hy : y ≥ 2) (hz : 0 < z) : 
  (1 / 2 + 1 / y = 1 / 2 + 1 / z) ↔ (y = z ∧ y ≥ 2) ∨ (y = 1 ∧ z = 1) := 
by 
  sorry 

theorem part_c (y z : ℕ) (hy : y ≥ 3) (hz : 0 < z) : 
  (1 / 3 + 1 / y = 1 / 2 + 1 / z) ↔ 
    (y = 3 ∧ z = 6) ∨ 
    (y = 4 ∧ z = 12) ∨ 
    (y = 5 ∧ z = 30) ∨ 
    (y = 2 ∧ z = 3) := 
by 
  sorry 

theorem part_d (x y : ℕ) (hx : x ≥ 4) (hy : y ≥ 4) : 
  ¬(1 / x + 1 / y = 1 / 2 + 1 / z) := 
by 
  sorry

end part_a_part_b_part_c_part_d_l1678_167877


namespace sum_first_five_terms_eq_ninety_three_l1678_167884

variable (a : ℕ → ℕ)

-- Definitions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

variables (a1 : ℕ) (a2 : ℕ) (a4 : ℕ)
variables (S : ℕ → ℕ)

-- Conditions
axiom a1_value : a1 = 3
axiom a2a4_value : a2 * a4 = 144

-- Question: Prove S_5 = 93
theorem sum_first_five_terms_eq_ninety_three
    (h1 : geometric_sequence a)
    (h2 : a 1 = a1)
    (h3 : a 2 = a2)
    (h4 : a 4 = a4)
    (Sn_def : S 5 = (a1 * (1 - (2:ℕ)^5)) / (1 - 2)) :
  S 5 = 93 :=
sorry

end sum_first_five_terms_eq_ninety_three_l1678_167884


namespace reduced_price_per_kg_l1678_167837

theorem reduced_price_per_kg {P R : ℝ} (H1 : R = 0.75 * P) (H2 : 1100 = 1100 / P * P) (H3 : 1100 = (1100 / P + 5) * R) : R = 55 :=
by sorry

end reduced_price_per_kg_l1678_167837


namespace train_crossing_time_is_correct_l1678_167805

-- Define the constant values
def train_length : ℝ := 350        -- Train length in meters
def train_speed : ℝ := 20          -- Train speed in m/s
def crossing_time : ℝ := 17.5      -- Time to cross the signal post in seconds

-- Proving the relationship that the time taken for the train to cross the signal post is as calculated
theorem train_crossing_time_is_correct : (train_length / train_speed) = crossing_time :=
by
  sorry

end train_crossing_time_is_correct_l1678_167805


namespace parabola_equation_l1678_167878

-- Define the given conditions
def vertex : ℝ × ℝ := (3, 5)
def point_on_parabola : ℝ × ℝ := (4, 2)

-- Prove that the equation is as specified
theorem parabola_equation :
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x y : ℝ, (y = a * x^2 + b * x + c) ↔
     (y = -3 * x^2 + 18 * x - 22) ∧ (vertex.snd = -3 * (vertex.fst - 3)^2 + 5) ∧
     (point_on_parabola.snd = a * point_on_parabola.fst^2 + b * point_on_parabola.fst + c)) := 
sorry

end parabola_equation_l1678_167878


namespace geometric_prog_105_l1678_167831

theorem geometric_prog_105 {a q : ℝ} 
  (h_sum : a + a * q + a * q^2 = 105) 
  (h_arith : a * q - a = (a * q^2 - 15) - a * q) :
  (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 0.5) :=
by
  sorry

end geometric_prog_105_l1678_167831


namespace sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l1678_167888

-- Definitions based on conditions
def standard_flavor_to_water_ratio := 1 / 30
def standard_flavor_to_corn_syrup_ratio := 1 / 12
def sport_water_amount := 60
def sport_corn_syrup_amount := 4
def sport_flavor_to_water_ratio := 1 / 60
def sport_flavor_amount := 1 -- derived from sport_water_amount * sport_flavor_to_water_ratio

-- The main theorem to prove
theorem sport_flavoring_to_corn_syrup_ratio_is_three_times_standard :
  1 / 4 = 3 * (1 / 12) :=
by
  sorry

end sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l1678_167888


namespace pies_from_apples_l1678_167872

theorem pies_from_apples (total_apples : ℕ) (percent_handout : ℝ) (apples_per_pie : ℕ) 
  (h_total : total_apples = 800) (h_percent : percent_handout = 0.65) (h_per_pie : apples_per_pie = 15) : 
  (total_apples * (1 - percent_handout)) / apples_per_pie = 18 := 
by 
  sorry

end pies_from_apples_l1678_167872


namespace pages_to_read_tomorrow_l1678_167880

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l1678_167880


namespace sum_of_four_powers_l1678_167813

theorem sum_of_four_powers (a : ℕ) : 4 * a^3 = 500 :=
by
  rw [Nat.pow_succ, Nat.pow_succ]
  sorry

end sum_of_four_powers_l1678_167813


namespace eight_points_on_circle_l1678_167808

theorem eight_points_on_circle
  (R : ℝ) (hR : R > 0)
  (points : Fin 8 → (ℝ × ℝ))
  (hpoints : ∀ i : Fin 8, (points i).1 ^ 2 + (points i).2 ^ 2 ≤ R ^ 2) :
  ∃ (i j : Fin 8), i ≠ j ∧ (dist (points i) (points j) < R) :=
sorry

end eight_points_on_circle_l1678_167808


namespace coat_price_reduction_l1678_167861

theorem coat_price_reduction :
  let orig_price := 500
  let first_discount := 0.15 * orig_price
  let price_after_first := orig_price - first_discount
  let second_discount := 0.10 * price_after_first
  let price_after_second := price_after_first - second_discount
  let tax := 0.07 * price_after_second
  let price_with_tax := price_after_second + tax
  let final_price := price_with_tax - 200
  let reduction_amount := orig_price - final_price
  let percent_reduction := (reduction_amount / orig_price) * 100
  percent_reduction = 58.145 :=
by
  sorry

end coat_price_reduction_l1678_167861


namespace sufficient_but_not_necessary_l1678_167874

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end sufficient_but_not_necessary_l1678_167874


namespace total_highlighters_is_49_l1678_167849

-- Define the number of highlighters of each color
def pink_highlighters : Nat := 15
def yellow_highlighters : Nat := 12
def blue_highlighters : Nat := 9
def green_highlighters : Nat := 7
def purple_highlighters : Nat := 6

-- Define the total number of highlighters
def total_highlighters : Nat := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters

-- Statement that the total number of highlighters should be 49
theorem total_highlighters_is_49 : total_highlighters = 49 := by
  sorry

end total_highlighters_is_49_l1678_167849


namespace ratio_spaghetti_to_manicotti_l1678_167889

-- Definitions of the given conditions
def total_students : ℕ := 800
def spaghetti_preferred : ℕ := 320
def manicotti_preferred : ℕ := 160

-- The theorem statement
theorem ratio_spaghetti_to_manicotti : spaghetti_preferred / manicotti_preferred = 2 :=
by sorry

end ratio_spaghetti_to_manicotti_l1678_167889


namespace solution1_solution2_l1678_167853

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l1678_167853


namespace appropriate_mass_units_l1678_167885

def unit_of_mass_basket_of_eggs : String :=
  if 5 = 5 then "kilograms" else "unknown"

def unit_of_mass_honeybee : String :=
  if 5 = 5 then "grams" else "unknown"

def unit_of_mass_tank : String :=
  if 6 = 6 then "tons" else "unknown"

theorem appropriate_mass_units :
  unit_of_mass_basket_of_eggs = "kilograms" ∧
  unit_of_mass_honeybee = "grams" ∧
  unit_of_mass_tank = "tons" :=
by {
  -- skip the proof
  sorry
}

end appropriate_mass_units_l1678_167885


namespace children_on_bus_l1678_167823

theorem children_on_bus (initial_children additional_children total_children : ℕ)
  (h1 : initial_children = 64)
  (h2 : additional_children = 14)
  (h3 : total_children = initial_children + additional_children) :
  total_children = 78 :=
by
  rw [h1, h2] at h3
  exact h3

end children_on_bus_l1678_167823


namespace find_other_integer_l1678_167882

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 140) (h2 : x = 20 ∨ y = 20) : x = 20 ∧ y = 20 :=
by
  sorry

end find_other_integer_l1678_167882


namespace triangle_sum_of_squares_not_right_l1678_167895

noncomputable def is_right_triangle (a b c : ℝ) : Prop := 
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem triangle_sum_of_squares_not_right
  (a b r : ℝ) :
  a^2 + b^2 = (2 * r)^2 → ¬ ∃ (c : ℝ), is_right_triangle a b c := 
sorry

end triangle_sum_of_squares_not_right_l1678_167895


namespace saffron_milk_caps_and_milk_caps_in_basket_l1678_167894

structure MushroomBasket :=
  (total : ℕ)
  (saffronMilkCapCount : ℕ)
  (milkCapCount : ℕ)
  (TotalMushrooms : total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < milkCapCount)

theorem saffron_milk_caps_and_milk_caps_in_basket
  (basket : MushroomBasket)
  (TotalMushrooms : basket.total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < basket.saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < basket.milkCapCount) :
  basket.saffronMilkCapCount = 19 ∧ basket.milkCapCount = 11 :=
sorry

end saffron_milk_caps_and_milk_caps_in_basket_l1678_167894


namespace maximize_area_of_sector_l1678_167862

noncomputable def area_of_sector (x y : ℝ) : ℝ := (1 / 2) * x * y

theorem maximize_area_of_sector : 
  ∃ x y : ℝ, 2 * x + y = 20 ∧ (∀ (x : ℝ), x > 0 → 
  (∀ (y : ℝ), y > 0 → 2 * x + y = 20 → area_of_sector x y ≤ area_of_sector 5 (20 - 2 * 5))) ∧ x = 5 :=
by
  sorry

end maximize_area_of_sector_l1678_167862


namespace cost_of_eight_books_l1678_167868

theorem cost_of_eight_books (x : ℝ) (h : 2 * x = 34) : 8 * x = 136 :=
by
  sorry

end cost_of_eight_books_l1678_167868


namespace tan_sum_example_l1678_167827

theorem tan_sum_example :
  let t1 := Real.tan (17 * Real.pi / 180)
  let t2 := Real.tan (43 * Real.pi / 180)
  t1 + t2 + Real.sqrt 3 * t1 * t2 = Real.sqrt 3 := sorry

end tan_sum_example_l1678_167827


namespace oliver_final_amount_is_54_04_l1678_167810

noncomputable def final_amount : ℝ :=
  let initial := 33
  let feb_spent := 0.15 * initial
  let after_feb := initial - feb_spent
  let march_added := 32
  let after_march := after_feb + march_added
  let march_spent := 0.10 * after_march
  after_march - march_spent

theorem oliver_final_amount_is_54_04 : final_amount = 54.04 := by
  sorry

end oliver_final_amount_is_54_04_l1678_167810


namespace sin_difference_identity_l1678_167818

theorem sin_difference_identity 
  (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 1 / 3) : 
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := 
  sorry

end sin_difference_identity_l1678_167818


namespace range_of_a_l1678_167852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) (h4 : ∀ x, g a x ≤ 0 → ¬(x < 0 ∧ g a x > 0)) :
  2 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l1678_167852


namespace rectangle_area_l1678_167887

def length : ℝ := 15
def width : ℝ := 0.9 * length
def area : ℝ := length * width

theorem rectangle_area : area = 202.5 := by
  sorry

end rectangle_area_l1678_167887


namespace div_c_a_l1678_167891

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l1678_167891


namespace smallest_n_for_gcd_l1678_167820

theorem smallest_n_for_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 4) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 4) > 1 → n ≤ m) → n = 38 :=
by
  sorry

end smallest_n_for_gcd_l1678_167820


namespace max_min_product_xy_theorem_l1678_167886

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l1678_167886


namespace simplify_fraction_l1678_167843

theorem simplify_fraction (x y z : ℕ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4 / 3 :=
by
  sorry

end simplify_fraction_l1678_167843


namespace reflected_ray_equation_l1678_167803

-- Definitions for the given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Problem statement: proving equation of the reflected ray
theorem reflected_ray_equation : 
  ∀ x y : ℝ, incident_line x = y ∧ reflection_line x = y → x - 2*y - 1 = 0 :=
by
  sorry

end reflected_ray_equation_l1678_167803


namespace decode_CLUE_is_8671_l1678_167883

def BEST_OF_LUCK_code : List (Char × Nat) :=
  [('B', 0), ('E', 1), ('S', 2), ('T', 3), ('O', 4), ('F', 5),
   ('L', 6), ('U', 7), ('C', 8), ('K', 9)]

def decode (code : List (Char × Nat)) (word : String) : Option Nat :=
  word.toList.mapM (λ c => List.lookup c code) >>= (λ digits => 
  Option.some (Nat.ofDigits 10 digits))

theorem decode_CLUE_is_8671 :
  decode BEST_OF_LUCK_code "CLUE" = some 8671 :=
by
  -- Proof omitted
  sorry

end decode_CLUE_is_8671_l1678_167883


namespace pens_bought_l1678_167898

-- Define the given conditions
def num_notebooks : ℕ := 10
def cost_per_pen : ℕ := 2
def total_paid : ℕ := 30
def cost_per_notebook : ℕ := 0  -- Assumption that notebooks are free

-- Converted condition that 10N + 2P = 30 and N = 0
def equation (N P : ℕ) : Prop := (10 * N + 2 * P = total_paid)

-- Statement to prove that if notebooks are free, 15 pens were bought
theorem pens_bought (N : ℕ) (P : ℕ) (hN : N = cost_per_notebook) (h : equation N P) : P = 15 :=
by sorry

end pens_bought_l1678_167898


namespace tea_mixture_ratio_l1678_167807

theorem tea_mixture_ratio
    (x y : ℝ)
    (h₁ : 62 * x + 72 * y = 64.5 * (x + y)) :
    x / y = 3 := by
  sorry

end tea_mixture_ratio_l1678_167807


namespace age_hence_l1678_167893

theorem age_hence (A x : ℕ) (h1 : A = 50)
  (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 :=
by sorry

end age_hence_l1678_167893


namespace G_at_8_l1678_167869

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_8 :
  (G 4 = 8) →
  (∀ x : ℝ, (x^2 + 3 * x + 2 ≠ 0) →
    G (2 * x) / G (x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)) →
  G 8 = 112 / 3 :=
by
  intros h1 h2
  sorry

end G_at_8_l1678_167869


namespace simplification_of_fractional_equation_l1678_167841

theorem simplification_of_fractional_equation (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) -> (x - 4 * (3 - x) = -6) :=
by
  sorry

end simplification_of_fractional_equation_l1678_167841


namespace pencil_cost_l1678_167806

theorem pencil_cost 
  (x y : ℚ)
  (h1 : 3 * x + 2 * y = 165)
  (h2 : 4 * x + 7 * y = 303) :
  y = 19.155 := 
by
  sorry

end pencil_cost_l1678_167806


namespace smallest_n_with_314_in_decimal_l1678_167860

theorem smallest_n_with_314_in_decimal {m n : ℕ} (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n) 
  (h_contains_314 : ∃ k : ℕ, (10^k * m) % n == 314) : n = 315 :=
sorry

end smallest_n_with_314_in_decimal_l1678_167860


namespace gross_profit_without_discount_l1678_167844

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end gross_profit_without_discount_l1678_167844


namespace digit_is_4_l1678_167899

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem digit_is_4 (d : ℕ) (hd0 : is_even d) (hd1 : is_divisible_by_3 (14 + d)) : d = 4 :=
  sorry

end digit_is_4_l1678_167899


namespace not_solution_B_l1678_167814

theorem not_solution_B : ¬ (1 + 6 = 5) := by
  sorry

end not_solution_B_l1678_167814


namespace product_of_roots_eq_neg_125_over_4_l1678_167811

theorem product_of_roots_eq_neg_125_over_4 :
  (∀ x y : ℝ, (24 * x^2 + 60 * x - 750 = 0 ∧ 24 * y^2 + 60 * y - 750 = 0 ∧ x ≠ y) → x * y = -125 / 4) :=
by
  intro x y h
  sorry

end product_of_roots_eq_neg_125_over_4_l1678_167811


namespace product_of_first_two_terms_l1678_167809

theorem product_of_first_two_terms (a_7 : ℕ) (d : ℕ) (a_7_eq : a_7 = 17) (d_eq : d = 2) :
  let a_1 := a_7 - 6 * d
  let a_2 := a_1 + d
  a_1 * a_2 = 35 :=
by
  sorry

end product_of_first_two_terms_l1678_167809


namespace slope_of_perpendicular_line_l1678_167847

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end slope_of_perpendicular_line_l1678_167847


namespace problem_statement_l1678_167801

variable {f : ℝ → ℝ}

-- Condition 1: f(x) has domain ℝ (implicitly given by the type signature ωf)
-- Condition 2: f is decreasing on the interval (6, +∞)
def is_decreasing_on_6_infty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 6 < x → x < y → f x > f y

-- Condition 3: y = f(x + 6) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) = f (-x - 6)

-- The statement to prove
theorem problem_statement (h_decrease : is_decreasing_on_6_infty f) (h_even_shift : is_even_shifted f) : f 5 > f 8 :=
sorry

end problem_statement_l1678_167801


namespace sum_of_numbers_l1678_167825

theorem sum_of_numbers : 3 + 33 + 333 + 33.3 = 402.3 :=
  by
    sorry

end sum_of_numbers_l1678_167825


namespace evaluate_fraction_l1678_167890

theorem evaluate_fraction (a b : ℤ) (h1 : a = 5) (h2 : b = -2) : (5 : ℝ) / (a + b) = 5 / 3 :=
by
  sorry

end evaluate_fraction_l1678_167890


namespace math_majors_consecutive_probability_l1678_167896

def twelve_people := 12
def math_majors := 5
def physics_majors := 4
def biology_majors := 3

def total_ways := Nat.choose twelve_people math_majors

-- Computes the probability that all five math majors sit in consecutive seats
theorem math_majors_consecutive_probability :
  (12 : ℕ) / (Nat.choose twelve_people math_majors) = 1 / 66 := by
  sorry

end math_majors_consecutive_probability_l1678_167896


namespace jacob_ate_five_pies_l1678_167864

theorem jacob_ate_five_pies (weight_hot_dog weight_burger weight_pie noah_burgers mason_hotdogs_total_weight : ℕ)
    (H1 : weight_hot_dog = 2)
    (H2 : weight_burger = 5)
    (H3 : weight_pie = 10)
    (H4 : noah_burgers = 8)
    (H5 : mason_hotdogs_total_weight = 30)
    (H6 : ∀ x, 3 * x = (mason_hotdogs_total_weight / weight_hot_dog)) :
    (∃ y, y = (mason_hotdogs_total_weight / weight_hot_dog / 3) ∧ y = 5) :=
by
  sorry

end jacob_ate_five_pies_l1678_167864


namespace range_of_a_l1678_167842

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) : -4 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l1678_167842
