import Mathlib

namespace NUMINAMATH_GPT_scientific_notation_of_1_5_million_l2339_233966

theorem scientific_notation_of_1_5_million : 
    (1.5 * 10^6 = 1500000) :=
by
    sorry

end NUMINAMATH_GPT_scientific_notation_of_1_5_million_l2339_233966


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2339_233910

namespace BoatSpeed

variables (V_b V_s : ℝ)

def condition1 : Prop := V_b + V_s = 15
def condition2 : Prop := V_b - V_s = 5

theorem boat_speed_in_still_water (h1 : condition1 V_b V_s) (h2 : condition2 V_b V_s) : V_b = 10 :=
by
  sorry

end BoatSpeed

end NUMINAMATH_GPT_boat_speed_in_still_water_l2339_233910


namespace NUMINAMATH_GPT_odd_and_even_inter_empty_l2339_233959

-- Define the set of odd numbers
def odd_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define the set of even numbers
def even_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- The theorem stating that the intersection of odd numbers and even numbers is empty
theorem odd_and_even_inter_empty : odd_numbers ∩ even_numbers = ∅ :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_odd_and_even_inter_empty_l2339_233959


namespace NUMINAMATH_GPT_total_price_of_25_shirts_l2339_233903

theorem total_price_of_25_shirts (S W : ℝ) (H1 : W = S + 4) (H2 : 75 * W = 1500) : 
  25 * S = 400 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_total_price_of_25_shirts_l2339_233903


namespace NUMINAMATH_GPT_tan_theta_minus_pi_four_l2339_233927

theorem tan_theta_minus_pi_four (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) (h3 : Real.sin θ = -3/5) :
  Real.tan (θ - π / 4) = -1 / 7 :=
sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_four_l2339_233927


namespace NUMINAMATH_GPT_min_m_for_four_elements_l2339_233918

open Set

theorem min_m_for_four_elements (n : ℕ) (hn : n ≥ 2) :
  ∃ m, m = 2 * n + 2 ∧ 
  (∀ (S : Finset ℕ), S.card = m → 
    (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a = b + c + d)) :=
by
  sorry

end NUMINAMATH_GPT_min_m_for_four_elements_l2339_233918


namespace NUMINAMATH_GPT_fermat_large_prime_solution_l2339_233945

theorem fermat_large_prime_solution (n : ℕ) (hn : n > 0) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (x y z : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^n + y^n ≡ z^n [ZMOD p]) :=
sorry

end NUMINAMATH_GPT_fermat_large_prime_solution_l2339_233945


namespace NUMINAMATH_GPT_max_small_packages_l2339_233923

theorem max_small_packages (L S : ℝ) (W : ℝ) (h1 : W = 12 * L) (h2 : W = 20 * S) :
  (∃ n_smalls, n_smalls = 5 ∧ W - 9 * L = n_smalls * S) :=
by
  sorry

end NUMINAMATH_GPT_max_small_packages_l2339_233923


namespace NUMINAMATH_GPT_jenny_eggs_per_basket_l2339_233982

theorem jenny_eggs_per_basket :
  ∃ n, (30 % n = 0 ∧ 42 % n = 0 ∧ 18 % n = 0 ∧ n >= 6) → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_jenny_eggs_per_basket_l2339_233982


namespace NUMINAMATH_GPT_inequality_l2339_233975

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 3) :
  1 / (4 - a^2) + 1 / (4 - b^2) + 1 / (4 - c^2) ≤ 9 / (a + b + c)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_l2339_233975


namespace NUMINAMATH_GPT_least_common_multiple_of_812_and_3214_is_correct_l2339_233942

def lcm_812_3214 : ℕ :=
  Nat.lcm 812 3214

theorem least_common_multiple_of_812_and_3214_is_correct :
  lcm_812_3214 = 1304124 := by
  sorry

end NUMINAMATH_GPT_least_common_multiple_of_812_and_3214_is_correct_l2339_233942


namespace NUMINAMATH_GPT_weight_ratio_l2339_233961

variable (J : ℕ) (T : ℕ) (L : ℕ) (S : ℕ)

theorem weight_ratio (h_jake_weight : J = 152) (h_total_weight : J + S = 212) (h_weight_loss : L = 32) :
    (J - L) / (T - J) = 2 :=
by
  sorry

end NUMINAMATH_GPT_weight_ratio_l2339_233961


namespace NUMINAMATH_GPT_radius_of_circumscribed_circle_l2339_233901

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circumscribed_circle_l2339_233901


namespace NUMINAMATH_GPT_goldfish_equal_number_after_n_months_l2339_233941

theorem goldfish_equal_number_after_n_months :
  ∃ (n : ℕ), 2 * 4^n = 162 * 3^n ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_equal_number_after_n_months_l2339_233941


namespace NUMINAMATH_GPT_razorback_tshirt_profit_l2339_233999

theorem razorback_tshirt_profit
  (total_tshirts_sold : ℕ)
  (tshirts_sold_arkansas_game : ℕ)
  (money_made_arkansas_game : ℕ) :
  total_tshirts_sold = 163 →
  tshirts_sold_arkansas_game = 89 →
  money_made_arkansas_game = 8722 →
  money_made_arkansas_game / tshirts_sold_arkansas_game = 98 :=
by 
  intros _ _ _
  sorry

end NUMINAMATH_GPT_razorback_tshirt_profit_l2339_233999


namespace NUMINAMATH_GPT_preferred_dividend_rate_l2339_233944

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

end NUMINAMATH_GPT_preferred_dividend_rate_l2339_233944


namespace NUMINAMATH_GPT_john_multiple_is_correct_l2339_233978

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

end NUMINAMATH_GPT_john_multiple_is_correct_l2339_233978


namespace NUMINAMATH_GPT_triangles_not_necessarily_congruent_l2339_233968

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

end NUMINAMATH_GPT_triangles_not_necessarily_congruent_l2339_233968


namespace NUMINAMATH_GPT_right_triangle_median_l2339_233920

noncomputable def median_to_hypotenuse_length (a b : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (a^2 + b^2)
  hypotenuse / 2

theorem right_triangle_median
  (a b : ℝ) (h_a : a = 3) (h_b : b = 4) :
  median_to_hypotenuse_length a b = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_median_l2339_233920


namespace NUMINAMATH_GPT_area_change_l2339_233972

variable (p k : ℝ)
variable {N : ℝ}

theorem area_change (hN : N = 1/2 * (p * p)) (q : ℝ) (hq : q = k * p) :
  q = k * p -> (1/2 * (q * q) = k^2 * N) :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_change_l2339_233972


namespace NUMINAMATH_GPT_determine_g_l2339_233998

variable {R : Type*} [CommRing R]

theorem determine_g (g : R → R) (x : R) :
  (4 * x^5 + 3 * x^3 - 2 * x + 1 + g x = 7 * x^3 - 5 * x^2 + 4 * x - 3) →
  g x = -4 * x^5 + 4 * x^3 - 5 * x^2 + 6 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_g_l2339_233998


namespace NUMINAMATH_GPT_max_min_sum_eq_two_l2339_233952

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 + Real.sqrt 2 * Real.sin (x + Real.pi / 4)) / (2 * x ^ 2 + Real.cos x)

theorem max_min_sum_eq_two (a b : ℝ) (h_max : ∀ x, f x ≤ a) (h_min : ∀ x, b ≤ f x) (h_max_val : ∃ x, f x = a) (h_min_val : ∃ x, f x = b) :
  a + b = 2 := 
sorry

end NUMINAMATH_GPT_max_min_sum_eq_two_l2339_233952


namespace NUMINAMATH_GPT_prime_pair_solution_l2339_233915

-- Steps a) and b) are incorporated into this Lean statement
theorem prime_pair_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q ∣ 3^p + 3^q ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) :=
sorry

end NUMINAMATH_GPT_prime_pair_solution_l2339_233915


namespace NUMINAMATH_GPT_larger_number_l2339_233938

theorem larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end NUMINAMATH_GPT_larger_number_l2339_233938


namespace NUMINAMATH_GPT_even_factors_count_l2339_233954

theorem even_factors_count (n : ℕ) (h : n = 2^4 * 3^2 * 5 * 7) : 
  ∃ k : ℕ, k = 48 ∧ ∃ a b c d : ℕ, 
  1 ≤ a ∧ a ≤ 4 ∧
  0 ≤ b ∧ b ≤ 2 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  0 ≤ d ∧ d ≤ 1 ∧
  k = (4 - 1 + 1) * (2 + 1) * (1 + 1) * (1 + 1) := by
  sorry

end NUMINAMATH_GPT_even_factors_count_l2339_233954


namespace NUMINAMATH_GPT_monotonicity_f_range_of_b_l2339_233933

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def p (a b : ℝ) (x : ℝ) : Prop := f a x ≤ 2 * b
def q (b : ℝ) : Prop := ∀ x, (x = -3 → (x^2 + (2*b + 1)*x - b - 1) > 0) ∧ 
                           (x = -2 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 0 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 1 → (x^2 + (2*b + 1)*x - b - 1) > 0)

theorem monotonicity_f (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : ∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 := by
  sorry

theorem range_of_b (b : ℝ) (hp_or : ∃ x, p a b x ∨ q b) (hp_and : ∀ x, ¬(p a b x ∧ q b)) :
    (1/5 < b ∧ b < 1/2) ∨ (b ≥ 5/7) := by
    sorry

end NUMINAMATH_GPT_monotonicity_f_range_of_b_l2339_233933


namespace NUMINAMATH_GPT_fixed_point_is_one_three_l2339_233939

noncomputable def fixed_point_of_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : ℝ × ℝ :=
  (1, 3)

theorem fixed_point_is_one_three {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  fixed_point_of_function a h_pos h_ne_one = (1, 3) :=
  sorry

end NUMINAMATH_GPT_fixed_point_is_one_three_l2339_233939


namespace NUMINAMATH_GPT_coin_probability_l2339_233963

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

end NUMINAMATH_GPT_coin_probability_l2339_233963


namespace NUMINAMATH_GPT_tenth_day_is_monday_l2339_233926

theorem tenth_day_is_monday (runs_20_mins : ∀ d ∈ [1, 7], d = 1 ∨ d = 6 ∨ d = 7 → True)
                            (total_minutes : 5 * 60 = 300)
                            (first_day_is_saturday : 1 = 6) :
   (10 % 7 = 3) :=
by
  sorry

end NUMINAMATH_GPT_tenth_day_is_monday_l2339_233926


namespace NUMINAMATH_GPT_find_number_l2339_233928

theorem find_number (x : ℝ) (h : 0.40 * x = 130 + 190) : x = 800 :=
by {
  -- The proof will go here
  sorry
}

end NUMINAMATH_GPT_find_number_l2339_233928


namespace NUMINAMATH_GPT_correct_propositions_l2339_233964

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem correct_propositions :
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * x - Real.pi / 12)) ∧
  (Real.sqrt 2 = f (Real.pi / 24)) ∧
  (f (-1) ≠ f 1) ∧
  (∀ x, Real.pi / 24 ≤ x ∧ x ≤ 13 * Real.pi / 24 -> (f (x + 1e-6) < f x)) ∧
  (∀ x, (Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) = f x)
  := by
    sorry

end NUMINAMATH_GPT_correct_propositions_l2339_233964


namespace NUMINAMATH_GPT_sum_of_natural_numbers_l2339_233987

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  2^(n-1)

theorem sum_of_natural_numbers (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k = number_of_ways n :=
by
  use 2^(n-1)
  sorry

end NUMINAMATH_GPT_sum_of_natural_numbers_l2339_233987


namespace NUMINAMATH_GPT_prime_factor_of_sum_l2339_233997

theorem prime_factor_of_sum (n : ℤ) : ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ (2 * n + 1 + 2 * n + 3 + 2 * n + 5 + 2 * n + 7) % p = 0 :=
by
  sorry

end NUMINAMATH_GPT_prime_factor_of_sum_l2339_233997


namespace NUMINAMATH_GPT_circle_equation_value_l2339_233947

theorem circle_equation_value (a : ℝ) :
  (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 → False) → a = -1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_circle_equation_value_l2339_233947


namespace NUMINAMATH_GPT_painting_house_cost_l2339_233921

theorem painting_house_cost 
  (judson_contrib : ℕ := 500)
  (kenny_contrib : ℕ := judson_contrib + (judson_contrib * 20) / 100)
  (camilo_contrib : ℕ := kenny_contrib + 200) :
  judson_contrib + kenny_contrib + camilo_contrib = 1900 :=
by
  sorry

end NUMINAMATH_GPT_painting_house_cost_l2339_233921


namespace NUMINAMATH_GPT_sum_of_squares_of_real_solutions_l2339_233955

theorem sum_of_squares_of_real_solutions :
  (∀ x : ℝ, |x^2 - 3 * x + 1 / 400| = 1 / 400)
  → ((0^2 : ℝ) + 3^2 + (9 - 1 / 100) = 999 / 100) := sorry

end NUMINAMATH_GPT_sum_of_squares_of_real_solutions_l2339_233955


namespace NUMINAMATH_GPT_mathematician_daily_questions_l2339_233962

theorem mathematician_daily_questions :
  (518 + 476) / 7 = 142 := by
  sorry

end NUMINAMATH_GPT_mathematician_daily_questions_l2339_233962


namespace NUMINAMATH_GPT_math_problem_l2339_233967

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

end NUMINAMATH_GPT_math_problem_l2339_233967


namespace NUMINAMATH_GPT_monotonic_increasing_condition_l2339_233992

open Real

noncomputable def f (x : ℝ) (l a : ℝ) : ℝ := x^2 - x + l + a * log x

theorem monotonic_increasing_condition (l a : ℝ) (x : ℝ) (hx : x > 0) 
  (h : ∀ x, x > 0 → deriv (f l a) x ≥ 0) : 
  a > 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_condition_l2339_233992


namespace NUMINAMATH_GPT_sum_of_seven_digits_l2339_233937

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

end NUMINAMATH_GPT_sum_of_seven_digits_l2339_233937


namespace NUMINAMATH_GPT_num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l2339_233924

theorem num_three_digit_numbers_divisible_by_5_and_6_with_digit_6 : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (6 ∈ n.digits 10)) ∧ S.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l2339_233924


namespace NUMINAMATH_GPT_geometric_series_sum_l2339_233935

theorem geometric_series_sum (a r : ℝ) 
  (h1 : a * (1 - r / (1 - r)) = 18) 
  (h2 : a * (r / (1 - r)) = 8) : r = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_geometric_series_sum_l2339_233935


namespace NUMINAMATH_GPT_factorize_expression_l2339_233995

theorem factorize_expression (m n : ℝ) :
  2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2339_233995


namespace NUMINAMATH_GPT_sixth_year_fee_l2339_233960

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end NUMINAMATH_GPT_sixth_year_fee_l2339_233960


namespace NUMINAMATH_GPT_simplify_expression_l2339_233948

variable (a b c x : ℝ)

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def p (x a b c : ℝ) : ℝ :=
  (x - a)^3/(a - b)*(a - c) + a*x +
  (x - b)^3/(b - a)*(b - c) + b*x +
  (x - c)^3/(c - a)*(c - b) + c*x

theorem simplify_expression (h : distinct a b c) :
  p x a b c = a + b + c + 3*x + 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2339_233948


namespace NUMINAMATH_GPT_solve_for_S_l2339_233929

theorem solve_for_S (S : ℝ) (h : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120) : S = 120 :=
sorry

end NUMINAMATH_GPT_solve_for_S_l2339_233929


namespace NUMINAMATH_GPT_relationship_between_areas_l2339_233916

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

end NUMINAMATH_GPT_relationship_between_areas_l2339_233916


namespace NUMINAMATH_GPT_geometric_sequence_sum_is_120_l2339_233996

noncomputable def sum_first_four_geometric_seq (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4

theorem geometric_sequence_sum_is_120 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_pos_geometric : 0 < q ∧ q < 1)
  (h_a3_a5 : a 3 + a 5 = 20)
  (h_a3_a5_product : a 3 * a 5 = 64) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) :
  sum_first_four_geometric_seq a q = 120 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_is_120_l2339_233996


namespace NUMINAMATH_GPT_colorings_10x10_board_l2339_233911

def colorings_count (n : ℕ) : ℕ := 2^11 - 2

theorem colorings_10x10_board : colorings_count 10 = 2046 :=
by
  sorry

end NUMINAMATH_GPT_colorings_10x10_board_l2339_233911


namespace NUMINAMATH_GPT_school_total_payment_l2339_233990

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

end NUMINAMATH_GPT_school_total_payment_l2339_233990


namespace NUMINAMATH_GPT_circle_equation_l2339_233934

theorem circle_equation : 
  ∃ (b : ℝ), (∀ (x y : ℝ), (x^2 + (y - b)^2 = 1 ↔ (x = 1 ∧ y = 2) → b = 2)) :=
sorry

end NUMINAMATH_GPT_circle_equation_l2339_233934


namespace NUMINAMATH_GPT_perfect_square_divisors_count_l2339_233979

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def product_of_factorials : Nat := factorial 1 * factorial 2 * factorial 3 * factorial 4 * factorial 5 *
                                   factorial 6 * factorial 7 * factorial 8 * factorial 9 * factorial 10

def count_perfect_square_divisors (n : Nat) : Nat := sorry -- This would involve the correct function implementation.

theorem perfect_square_divisors_count :
  count_perfect_square_divisors product_of_factorials = 2160 :=
sorry

end NUMINAMATH_GPT_perfect_square_divisors_count_l2339_233979


namespace NUMINAMATH_GPT_max_A_l2339_233951

noncomputable def A (x y : ℝ) : ℝ :=
  x^4 * y + x * y^4 + x^3 * y + x * y^3 + x^2 * y + x * y^2

theorem max_A (x y : ℝ) (h : x + y = 1) : A x y ≤ 7 / 16 :=
sorry

end NUMINAMATH_GPT_max_A_l2339_233951


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2339_233906

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

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2339_233906


namespace NUMINAMATH_GPT_johns_percentage_increase_l2339_233917

theorem johns_percentage_increase (original_amount new_amount : ℕ) (h₀ : original_amount = 30) (h₁ : new_amount = 40) :
  (new_amount - original_amount) * 100 / original_amount = 33 :=
by
  sorry

end NUMINAMATH_GPT_johns_percentage_increase_l2339_233917


namespace NUMINAMATH_GPT_alicia_bought_more_markers_l2339_233971

theorem alicia_bought_more_markers (price_per_marker : ℝ) (n_h : ℝ) (n_a : ℝ) (m : ℝ) 
    (h_hector : n_h * price_per_marker = 2.76) 
    (h_alicia : n_a * price_per_marker = 4.07)
    (h_diff : n_a - n_h = m) : 
  m = 13 :=
sorry

end NUMINAMATH_GPT_alicia_bought_more_markers_l2339_233971


namespace NUMINAMATH_GPT_floor_exponents_eq_l2339_233904

theorem floor_exponents_eq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_inf_k : ∃ᶠ k in at_top, ∃ (k : ℕ), ⌊a ^ k⌋ + ⌊b ^ k⌋ = ⌊a⌋ ^ k + ⌊b⌋ ^ k) :
  ⌊a ^ 2014⌋ + ⌊b ^ 2014⌋ = ⌊a⌋ ^ 2014 + ⌊b⌋ ^ 2014 := by
  sorry

end NUMINAMATH_GPT_floor_exponents_eq_l2339_233904


namespace NUMINAMATH_GPT_divisibility_problem_l2339_233931

theorem divisibility_problem
  (h1 : 5^3 ∣ 1978^100 - 1)
  (h2 : 10^4 ∣ 3^500 - 1)
  (h3 : 2003 ∣ 2^286 - 1) :
  2^4 * 5^7 * 2003 ∣ (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) :=
by sorry

end NUMINAMATH_GPT_divisibility_problem_l2339_233931


namespace NUMINAMATH_GPT_reduced_price_is_3_84_l2339_233969

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

end NUMINAMATH_GPT_reduced_price_is_3_84_l2339_233969


namespace NUMINAMATH_GPT_binom_12_6_l2339_233946

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end NUMINAMATH_GPT_binom_12_6_l2339_233946


namespace NUMINAMATH_GPT_arrange_COMMUNICATION_l2339_233956

theorem arrange_COMMUNICATION : 
  let n := 12
  let o_count := 2
  let i_count := 2
  let n_count := 2
  let m_count := 2
  let total_repeats := o_count * i_count * n_count * m_count
  n.factorial / (o_count.factorial * i_count.factorial * n_count.factorial * m_count.factorial) = 29937600 :=
by sorry

end NUMINAMATH_GPT_arrange_COMMUNICATION_l2339_233956


namespace NUMINAMATH_GPT_geometric_mean_of_1_and_9_is_pm3_l2339_233908

theorem geometric_mean_of_1_and_9_is_pm3 (a b c : ℝ) (h₀ : a = 1) (h₁ : b = 9) (h₂ : c^2 = a * b) : c = 3 ∨ c = -3 := by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_1_and_9_is_pm3_l2339_233908


namespace NUMINAMATH_GPT_garden_width_l2339_233950

theorem garden_width (w : ℕ) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end NUMINAMATH_GPT_garden_width_l2339_233950


namespace NUMINAMATH_GPT_apples_left_proof_l2339_233919

def apples_left (mike_apples : Float) (nancy_apples : Float) (keith_apples_eaten : Float): Float :=
  mike_apples + nancy_apples - keith_apples_eaten

theorem apples_left_proof :
  apples_left 7.0 3.0 6.0 = 4.0 :=
by
  unfold apples_left
  norm_num
  sorry

end NUMINAMATH_GPT_apples_left_proof_l2339_233919


namespace NUMINAMATH_GPT_range_of_a_l2339_233913

variable (a : ℝ)
variable (x : ℝ)

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ∀ x, otimes (x - a) (x + a) < 1) : - 1 / 2 < a ∧ a < 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2339_233913


namespace NUMINAMATH_GPT_poly_not_33_l2339_233936

theorem poly_not_33 (x y : ℤ) : x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by sorry

end NUMINAMATH_GPT_poly_not_33_l2339_233936


namespace NUMINAMATH_GPT_total_handshakes_l2339_233993

section Handshakes

-- Define the total number of players
def total_players : ℕ := 4 + 6

-- Define the number of players in 2 and 3 player teams
def num_2player_teams : ℕ := 2
def num_3player_teams : ℕ := 2

-- Define the number of players per 2 player team and 3 player team
def players_per_2player_team : ℕ := 2
def players_per_3player_team : ℕ := 3

-- Define the total number of players in 2 player teams and in 3 player teams
def total_2player_team_players : ℕ := num_2player_teams * players_per_2player_team
def total_3player_team_players : ℕ := num_3player_teams * players_per_3player_team

-- Calculate handshakes
def handshakes (total_2player : ℕ) (total_3player : ℕ) : ℕ :=
  let h1 := total_2player * (total_players - players_per_2player_team) / 2
  let h2 := total_3player * (total_players - players_per_3player_team) / 2
  h1 + h2

-- Prove the total number of handshakes
theorem total_handshakes : handshakes total_2player_team_players total_3player_team_players = 37 :=
by
  have h1 := total_2player_team_players * (total_players - players_per_2player_team) / 2
  have h2 := total_3player_team_players * (total_players - players_per_3player_team) / 2
  have h_total := h1 + h2
  sorry

end Handshakes

end NUMINAMATH_GPT_total_handshakes_l2339_233993


namespace NUMINAMATH_GPT_square_center_sum_l2339_233980

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

end NUMINAMATH_GPT_square_center_sum_l2339_233980


namespace NUMINAMATH_GPT_simplify_polynomial_simplify_expression_l2339_233925

-- Problem 1:
theorem simplify_polynomial (x : ℝ) : 
  2 * x^3 - 4 * x^2 - 3 * x - 2 * x^2 - x^3 + 5 * x - 7 = x^3 - 6 * x^2 + 2 * x - 7 := 
by
  sorry

-- Problem 2:
theorem simplify_expression (m n : ℝ) (A B : ℝ) (hA : A = 2 * m^2 - m * n) (hB : B = m^2 + 2 * m * n - 5) : 
  4 * A - 2 * B = 6 * m^2 - 8 * m * n + 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_simplify_expression_l2339_233925


namespace NUMINAMATH_GPT_no_pairs_of_a_and_d_l2339_233900

theorem no_pairs_of_a_and_d :
  ∀ (a d : ℝ), (∀ (x y: ℝ), 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0) -> False :=
by 
  sorry

end NUMINAMATH_GPT_no_pairs_of_a_and_d_l2339_233900


namespace NUMINAMATH_GPT_elephant_weight_equivalence_l2339_233985

variable (y : ℝ)
variable (porter_weight : ℝ := 120)
variable (blocks_1 : ℝ := 20)
variable (blocks_2 : ℝ := 21)
variable (porters_1 : ℝ := 3)
variable (porters_2 : ℝ := 1)

theorem elephant_weight_equivalence :
  (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 := 
sorry

end NUMINAMATH_GPT_elephant_weight_equivalence_l2339_233985


namespace NUMINAMATH_GPT_modulus_sum_complex_l2339_233905

theorem modulus_sum_complex :
  let z1 : Complex := Complex.mk 3 (-8)
  let z2 : Complex := Complex.mk 4 6
  Complex.abs (z1 + z2) = Real.sqrt 53 := by
  sorry

end NUMINAMATH_GPT_modulus_sum_complex_l2339_233905


namespace NUMINAMATH_GPT_flour_needed_l2339_233991

theorem flour_needed (cookies : ℕ) (flour : ℕ) (k : ℕ) (f_whole_wheat f_all_purpose : ℕ) 
  (h : cookies = 45) (h1 : flour = 3) (h2 : k = 90) (h3 : (k / 2) = 45) 
  (h4 : f_all_purpose = (flour * (k / cookies)) / 2) 
  (h5 : f_whole_wheat = (flour * (k / cookies)) / 2) : 
  f_all_purpose = 3 ∧ f_whole_wheat = 3 := 
by
  sorry

end NUMINAMATH_GPT_flour_needed_l2339_233991


namespace NUMINAMATH_GPT_intersection_M_N_l2339_233957

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2339_233957


namespace NUMINAMATH_GPT_James_distance_ridden_l2339_233912

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end NUMINAMATH_GPT_James_distance_ridden_l2339_233912


namespace NUMINAMATH_GPT_pascal_triangle_21st_number_l2339_233922

theorem pascal_triangle_21st_number 
: (Nat.choose 22 2) = 231 :=
by 
  sorry

end NUMINAMATH_GPT_pascal_triangle_21st_number_l2339_233922


namespace NUMINAMATH_GPT_profit_percent_l2339_233949

theorem profit_percent (CP SP : ℤ) (h : CP/SP = 2/3) : (SP - CP) * 100 / CP = 50 := 
by
  sorry

end NUMINAMATH_GPT_profit_percent_l2339_233949


namespace NUMINAMATH_GPT_horse_tile_system_l2339_233940

theorem horse_tile_system (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + (1 / 3 : ℚ) * y = 100) : 
  ∃ (x y : ℕ), (x + y = 100) ∧ (3 * x + (1 / 3 : ℚ) * y = 100) :=
by sorry

end NUMINAMATH_GPT_horse_tile_system_l2339_233940


namespace NUMINAMATH_GPT_kiana_and_her_siblings_age_sum_l2339_233986

theorem kiana_and_her_siblings_age_sum :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 256 ∧ a + b + c = 38 :=
by
sorry

end NUMINAMATH_GPT_kiana_and_her_siblings_age_sum_l2339_233986


namespace NUMINAMATH_GPT_mary_score_is_95_l2339_233977

theorem mary_score_is_95
  (s c w : ℕ)
  (h1 : s > 90)
  (h2 : s = 35 + 5 * c - w)
  (h3 : c + w = 30)
  (h4 : ∀ c' w', s = 35 + 5 * c' - w' → c + w = c' + w' → (c', w') = (c, w)) :
  s = 95 :=
by
  sorry

end NUMINAMATH_GPT_mary_score_is_95_l2339_233977


namespace NUMINAMATH_GPT_sum_of_divisors_5_cubed_l2339_233953

theorem sum_of_divisors_5_cubed :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c = 5^3) ∧ (a = 1) ∧ (b = 5) ∧ (c = 25) ∧ (a + b + c = 31) :=
sorry

end NUMINAMATH_GPT_sum_of_divisors_5_cubed_l2339_233953


namespace NUMINAMATH_GPT_work_problem_l2339_233909

/-- 
  Suppose A can complete a work in \( x \) days alone, 
  B can complete the work in 20 days,
  and together they work for 7 days, leaving a fraction of 0.18333333333333335 of the work unfinished.
  Prove that \( x = 15 \).
 -/
theorem work_problem (x : ℝ) : 
  (∀ (B : ℝ), B = 20 → (∀ (f : ℝ), f = 0.18333333333333335 → (7 * (1 / x + 1 / B) = 1 - f)) → x = 15) := 
sorry

end NUMINAMATH_GPT_work_problem_l2339_233909


namespace NUMINAMATH_GPT_faster_speed_l2339_233983

theorem faster_speed (v : ℝ) :
  (∀ t : ℝ, (40 / 10 = t) ∧ (60 / v = t)) → v = 15 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l2339_233983


namespace NUMINAMATH_GPT_pyramid_property_l2339_233994

-- Define the areas of the faces of the right-angled triangular pyramid.
variables (S_ABC S_ACD S_ADB S_BCD : ℝ)

-- Define the condition that the areas correspond to a right-angled triangular pyramid.
def right_angled_triangular_pyramid (S_ABC S_ACD S_ADB S_BCD : ℝ) : Prop :=
  S_BCD^2 = S_ABC^2 + S_ACD^2 + S_ADB^2

-- State the theorem to be proven.
theorem pyramid_property : right_angled_triangular_pyramid S_ABC S_ACD S_ADB S_BCD :=
sorry

end NUMINAMATH_GPT_pyramid_property_l2339_233994


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l2339_233907

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

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l2339_233907


namespace NUMINAMATH_GPT_number_of_belts_l2339_233988

def ties := 34
def black_shirts := 63
def white_shirts := 42

def jeans := (2 / 3 : ℚ) * (black_shirts + white_shirts)
def scarves (B : ℚ) := (1 / 2 : ℚ) * (ties + B)

theorem number_of_belts (B : ℚ) : jeans = scarves B + 33 → B = 40 := by
  -- This theorem states the required proof but leaves the proof itself as a placeholder.
  -- The proof would involve solving equations algebraically as shown in the solution steps.
  sorry

end NUMINAMATH_GPT_number_of_belts_l2339_233988


namespace NUMINAMATH_GPT_side_length_a_cosine_A_l2339_233965

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

end NUMINAMATH_GPT_side_length_a_cosine_A_l2339_233965


namespace NUMINAMATH_GPT_total_shells_l2339_233958

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

end NUMINAMATH_GPT_total_shells_l2339_233958


namespace NUMINAMATH_GPT_length_of_EF_l2339_233974

theorem length_of_EF (AB BC : ℝ) (DE DF : ℝ) (Area_ABC : ℝ) (Area_DEF : ℝ) (EF : ℝ) 
  (h₁ : AB = 10) (h₂ : BC = 15) (h₃ : DE = DF) (h₄ : Area_DEF = (1/3) * Area_ABC) 
  (h₅ : Area_ABC = AB * BC) (h₆ : Area_DEF = (1/2) * (DE * DF)) : 
  EF = 10 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_EF_l2339_233974


namespace NUMINAMATH_GPT_initial_persons_count_l2339_233914

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

end NUMINAMATH_GPT_initial_persons_count_l2339_233914


namespace NUMINAMATH_GPT_cubic_sum_l2339_233976

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_GPT_cubic_sum_l2339_233976


namespace NUMINAMATH_GPT_estate_area_is_correct_l2339_233970

noncomputable def actual_area_of_estate (length_in_inches : ℕ) (width_in_inches : ℕ) (scale : ℕ) : ℕ :=
  let actual_length := length_in_inches * scale
  let actual_width := width_in_inches * scale
  actual_length * actual_width

theorem estate_area_is_correct :
  actual_area_of_estate 9 6 350 = 6615000 := by
  -- Here, we would provide the proof steps, but for this exercise, we use sorry.
  sorry

end NUMINAMATH_GPT_estate_area_is_correct_l2339_233970


namespace NUMINAMATH_GPT_hyperbola_asymptote_eqn_l2339_233984

theorem hyperbola_asymptote_eqn :
  ∀ (x y : ℝ),
  (y ^ 2 / 4 - x ^ 2 = 1) → (y = 2 * x ∨ y = -2 * x) := by
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_eqn_l2339_233984


namespace NUMINAMATH_GPT_a_minus_b_value_l2339_233973

theorem a_minus_b_value (a b c : ℝ) (x : ℝ) 
    (h1 : (2 * x - 3) ^ 2 = a * x ^ 2 + b * x + c)
    (h2 : x = 0 → c = 9)
    (h3 : x = 1 → a + b + c = 1)
    (h4 : x = -1 → (2 * (-1) - 3) ^ 2 = a * (-1) ^ 2 + b * (-1) + c) : 
    a - b = 16 :=
by  
  sorry

end NUMINAMATH_GPT_a_minus_b_value_l2339_233973


namespace NUMINAMATH_GPT_total_weekly_sleep_correct_l2339_233981

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

end NUMINAMATH_GPT_total_weekly_sleep_correct_l2339_233981


namespace NUMINAMATH_GPT_max_value_expression_l2339_233943

noncomputable def a (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def b (φ : ℝ) : ℝ := 3 * Real.sin φ

theorem max_value_expression (φ θ : ℝ) : 
  ∃ c : ℝ, c = 3 * Real.cos (θ - φ) ∧ c ≤ 3 := by
  sorry

end NUMINAMATH_GPT_max_value_expression_l2339_233943


namespace NUMINAMATH_GPT_A_lent_5000_to_B_l2339_233902

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

end NUMINAMATH_GPT_A_lent_5000_to_B_l2339_233902


namespace NUMINAMATH_GPT_james_total_spent_l2339_233930

noncomputable def total_cost : ℝ :=
  let milk_price := 3.0
  let bananas_price := 2.0
  let bread_price := 1.5
  let cereal_price := 4.0
  let milk_tax := 0.20
  let bananas_tax := 0.15
  let bread_tax := 0.10
  let cereal_tax := 0.25
  let milk_total := milk_price * (1 + milk_tax)
  let bananas_total := bananas_price * (1 + bananas_tax)
  let bread_total := bread_price * (1 + bread_tax)
  let cereal_total := cereal_price * (1 + cereal_tax)
  milk_total + bananas_total + bread_total + cereal_total

theorem james_total_spent : total_cost = 12.55 :=
  sorry

end NUMINAMATH_GPT_james_total_spent_l2339_233930


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2339_233932

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 3) (h2 : b = 1) :
  (a = 3 ∧ b = 1) ∧ (a + b > b ∨ b + b > a) → a + a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2339_233932


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_l2339_233989

theorem least_number_subtracted_divisible (n : ℕ) (divisor : ℕ) (rem : ℕ) :
  n = 427398 → divisor = 15 → n % divisor = rem → rem = 3 → ∃ k : ℕ, n - k = 427395 :=
by
  intros
  use 3
  sorry

end NUMINAMATH_GPT_least_number_subtracted_divisible_l2339_233989
