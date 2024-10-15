import Mathlib

namespace NUMINAMATH_GPT_simplify_exponentiation_l1087_108703

theorem simplify_exponentiation (x : ℕ) :
  (x^5 * x^3)^2 = x^16 := 
by {
  sorry -- proof will go here
}

end NUMINAMATH_GPT_simplify_exponentiation_l1087_108703


namespace NUMINAMATH_GPT_Olivia_score_l1087_108793

theorem Olivia_score 
  (n : ℕ) (m : ℕ) (average20 : ℕ) (average21 : ℕ)
  (h_n : n = 20) (h_m : m = 21) (h_avg20 : average20 = 85) (h_avg21 : average21 = 86)
  : ∃ (scoreOlivia : ℕ), scoreOlivia = m * average21 - n * average20 :=
by
  sorry

end NUMINAMATH_GPT_Olivia_score_l1087_108793


namespace NUMINAMATH_GPT_Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l1087_108716

-- Define \( S_n \) following the given conditions
def S (n : ℕ) : ℕ :=
  let a := 2^n + 1 -- first term
  let b := 2^(n+1) - 1 -- last term
  let m := b - a + 1 -- number of terms
  (m * (a + b)) / 2 -- sum of the arithmetic series

-- The first part: Prove that \( S_n \) is divisible by 3 for all positive integers \( n \)
theorem Sn_divisible_by_3 (n : ℕ) (hn : 0 < n) : 3 ∣ S n := sorry

-- The second part: Prove that \( S_n \) is divisible by 9 if and only if \( n \) is even
theorem Sn_divisible_by_9_iff_even (n : ℕ) (hn : 0 < n) : 9 ∣ S n ↔ Even n := sorry

end NUMINAMATH_GPT_Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l1087_108716


namespace NUMINAMATH_GPT_isla_capsules_days_l1087_108772

theorem isla_capsules_days (days_in_july : ℕ) (days_forgot : ℕ) (known_days_in_july : days_in_july = 31) (known_days_forgot : days_forgot = 2) : days_in_july - days_forgot = 29 := 
by
  -- Placeholder for proof, not required in the response.
  sorry

end NUMINAMATH_GPT_isla_capsules_days_l1087_108772


namespace NUMINAMATH_GPT_solve_equation_l1087_108733

noncomputable def equation (x : ℝ) : ℝ :=
(13 * x - x^2) / (x + 1) * (x + (13 - x) / (x + 1))

theorem solve_equation :
  equation 1 = 42 ∧ equation 6 = 42 ∧ equation (3 + Real.sqrt 2) = 42 ∧ equation (3 - Real.sqrt 2) = 42 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1087_108733


namespace NUMINAMATH_GPT_total_built_up_area_l1087_108782

theorem total_built_up_area
    (A1 A2 A3 A4 : ℕ)
    (hA1 : A1 = 480)
    (hA2 : A2 = 560)
    (hA3 : A3 = 200)
    (hA4 : A4 = 440)
    (total_plot_area : ℕ)
    (hplots : total_plot_area = 4 * (480 + 560 + 200 + 440) / 4)
    : 800 = total_plot_area - (A1 + A2 + A3 + A4) :=
by
  -- This is where the solution will be filled in
  sorry

end NUMINAMATH_GPT_total_built_up_area_l1087_108782


namespace NUMINAMATH_GPT_missing_digit_B_divisible_by_3_l1087_108794

theorem missing_digit_B_divisible_by_3 (B : ℕ) (h1 : (2 * 10 + 8 + B) % 3 = 0) :
  B = 2 :=
sorry

end NUMINAMATH_GPT_missing_digit_B_divisible_by_3_l1087_108794


namespace NUMINAMATH_GPT_school_children_equation_l1087_108798

theorem school_children_equation
  (C B : ℕ)
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 := by
  sorry

end NUMINAMATH_GPT_school_children_equation_l1087_108798


namespace NUMINAMATH_GPT_reciprocal_of_neg_one_fifth_l1087_108763

theorem reciprocal_of_neg_one_fifth : (-(1 / 5) : ℚ)⁻¹ = -5 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_one_fifth_l1087_108763


namespace NUMINAMATH_GPT_find_number_of_students_l1087_108730

theorem find_number_of_students (N : ℕ) (h1 : T = 80 * N) (h2 : (T - 350) / (N - 5) = 90) 
: N = 10 := 
by 
  -- Proof steps would go here. Omitted as per the instruction.
  sorry

end NUMINAMATH_GPT_find_number_of_students_l1087_108730


namespace NUMINAMATH_GPT_aiyanna_more_cookies_than_alyssa_l1087_108709

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end NUMINAMATH_GPT_aiyanna_more_cookies_than_alyssa_l1087_108709


namespace NUMINAMATH_GPT_fraction_n_m_l1087_108704

noncomputable def a (k : ℝ) := 2*k + 1
noncomputable def b (k : ℝ) := 3*k + 2
noncomputable def c (k : ℝ) := 3 - 4*k
noncomputable def S (k : ℝ) := a k + 2*(b k) + 3*(c k)

theorem fraction_n_m : 
  (∀ (k : ℝ), -1/2 ≤ k ∧ k ≤ 3/4 → (S (3/4) = 11 ∧ S (-1/2) = 16)) → 
  11/16 = 11 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_n_m_l1087_108704


namespace NUMINAMATH_GPT_how_many_years_older_l1087_108755

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_how_many_years_older_l1087_108755


namespace NUMINAMATH_GPT_no_solution_for_equation_l1087_108717

theorem no_solution_for_equation (x : ℝ) (hx : x ≠ -1) :
  (5 * x + 2) / (x^2 + x) ≠ 3 / (x + 1) := 
sorry

end NUMINAMATH_GPT_no_solution_for_equation_l1087_108717


namespace NUMINAMATH_GPT_find_counterfeit_l1087_108799

-- Definitions based on the conditions
structure Coin :=
(weight : ℝ)
(is_genuine : Bool)

def is_counterfeit (coins : List Coin) : Prop :=
  ∃ (c : Coin) (h : c ∈ coins), ¬c.is_genuine

def weigh (c1 c2 : Coin) : ℝ := c1.weight - c2.weight

def identify_counterfeit (coins : List Coin) : Prop :=
  ∀ (a b c d : Coin), 
    coins = [a, b, c, d] →
    (¬a.is_genuine ∨ ¬b.is_genuine ∨ ¬c.is_genuine ∨ ¬d.is_genuine) →
    (weigh a b = 0 ∧ weigh c d ≠ 0 ∨ weigh a c = 0 ∧ weigh b d ≠ 0 ∨ weigh a d = 0 ∧ weigh b c ≠ 0) →
    (∃ (fake_coin : Coin), fake_coin ∈ coins ∧ ¬fake_coin.is_genuine)

-- Proof statement
theorem find_counterfeit (coins : List Coin) :
  (∃ (c : Coin), c ∈ coins ∧ ¬c.is_genuine) →
  identify_counterfeit coins :=
by
  sorry

end NUMINAMATH_GPT_find_counterfeit_l1087_108799


namespace NUMINAMATH_GPT_current_speed_l1087_108778

theorem current_speed (r w : ℝ) 
  (h1 : 21 / (r + w) + 3 = 21 / (r - w))
  (h2 : 21 / (1.5 * r + w) + 0.75 = 21 / (1.5 * r - w)) 
  : w = 9.8 :=
by
  sorry

end NUMINAMATH_GPT_current_speed_l1087_108778


namespace NUMINAMATH_GPT_arithmetic_to_geometric_l1087_108795

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ)
  (h_arithmetic : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_d_nonzero : d ≠ 0):
  ((a2^2 = a1 * a3 ∨ a2^2 = a1 * a4 ∨ a3^2 = a1 * a4 ∨ a3^2 = a2 * a4) → (a1 / d = 1 ∨ a1 / d = -4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_to_geometric_l1087_108795


namespace NUMINAMATH_GPT_sin_half_angle_product_lt_quarter_l1087_108718

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h : A + B + C = 180) :
    Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := 
    sorry

end NUMINAMATH_GPT_sin_half_angle_product_lt_quarter_l1087_108718


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1087_108791

def M : Set ℝ := { x | x ≤ 0 }
def N : Set ℝ := { -2, 0, 1 }

theorem intersection_of_M_and_N : M ∩ N = { -2, 0 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1087_108791


namespace NUMINAMATH_GPT_number_of_squares_in_H_l1087_108780

-- Define the set H
def H : Set (ℤ × ℤ) :=
{ p | 2 ≤ abs p.1 ∧ abs p.1 ≤ 10 ∧ 2 ≤ abs p.2 ∧ abs p.2 ≤ 10 }

-- State the problem
theorem number_of_squares_in_H : 
  (∃ S : Finset (ℤ × ℤ), S.card = 20 ∧ 
    ∀ square ∈ S, 
      (∃ a b c d : ℤ × ℤ, 
        a ∈ H ∧ b ∈ H ∧ c ∈ H ∧ d ∈ H ∧ 
        (∃ s : ℤ, s ≥ 8 ∧ 
          (a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = d.1 ∧ d.2 = a.2 ∧ 
           abs (a.1 - c.1) = s ∧ abs (a.2 - d.2) = s)))) :=
sorry

end NUMINAMATH_GPT_number_of_squares_in_H_l1087_108780


namespace NUMINAMATH_GPT_sum_first_10_terms_l1087_108707

noncomputable def a (n : ℕ) := 1 / (4 * (n + 1) ^ 2 - 1)

theorem sum_first_10_terms : (Finset.range 10).sum a = 10 / 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_l1087_108707


namespace NUMINAMATH_GPT_professional_pay_per_hour_l1087_108737

def professionals : ℕ := 2
def hours_per_day : ℕ := 6
def days : ℕ := 7
def total_cost : ℕ := 1260

theorem professional_pay_per_hour :
  (total_cost / (professionals * hours_per_day * days) = 15) :=
by
  sorry

end NUMINAMATH_GPT_professional_pay_per_hour_l1087_108737


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1087_108775

theorem quadratic_two_distinct_real_roots (k : ℝ) : ∃ x : ℝ, x^2 + 2 * x - k = 0 ∧ 
  (∀ x1 x2: ℝ, x1 ≠ x2 → x1^2 + 2 * x1 - k = 0 ∧ x2^2 + 2 * x2 - k = 0) ↔ k > -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1087_108775


namespace NUMINAMATH_GPT_josie_leftover_amount_l1087_108797

-- Define constants and conditions
def initial_amount : ℝ := 20.00
def milk_price : ℝ := 4.00
def bread_price : ℝ := 3.50
def detergent_price : ℝ := 10.25
def bananas_price_per_pound : ℝ := 0.75
def bananas_weight : ℝ := 2.0
def detergent_coupon : ℝ := 1.25
def milk_discount_rate : ℝ := 0.5

-- Define the total cost before any discounts
def total_cost_before_discounts : ℝ := 
  milk_price + bread_price + detergent_price + (bananas_weight * bananas_price_per_pound)

-- Define the discounted prices
def milk_discounted_price : ℝ := milk_price * milk_discount_rate
def detergent_discounted_price : ℝ := detergent_price - detergent_coupon

-- Define the total cost after discounts
def total_cost_after_discounts : ℝ := 
  milk_discounted_price + bread_price + detergent_discounted_price + 
  (bananas_weight * bananas_price_per_pound)

-- Prove the amount left over
theorem josie_leftover_amount : initial_amount - total_cost_after_discounts = 4.00 := by
  simp [total_cost_before_discounts, milk_discounted_price, detergent_discounted_price,
    total_cost_after_discounts, initial_amount, milk_price, bread_price, detergent_price,
    bananas_price_per_pound, bananas_weight, detergent_coupon, milk_discount_rate]
  sorry

end NUMINAMATH_GPT_josie_leftover_amount_l1087_108797


namespace NUMINAMATH_GPT_darij_grinberg_inequality_l1087_108796

theorem darij_grinberg_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a + b + c ≤ (bc / (b + c)) + (ca / (c + a)) + (ab / (a + b)) + (1 / 2 * ((bc / a) + (ca / b) + (ab / c))) := 
by sorry

end NUMINAMATH_GPT_darij_grinberg_inequality_l1087_108796


namespace NUMINAMATH_GPT_chairs_left_l1087_108725

-- Conditions
def red_chairs : Nat := 4
def yellow_chairs : Nat := 2 * red_chairs
def blue_chairs : Nat := yellow_chairs - 2
def lisa_borrows : Nat := 3

-- Theorem
theorem chairs_left (chairs_left : Nat) : chairs_left = red_chairs + yellow_chairs + blue_chairs - lisa_borrows :=
by
  sorry

end NUMINAMATH_GPT_chairs_left_l1087_108725


namespace NUMINAMATH_GPT_circle_equation_center_xaxis_radius_2_l1087_108721

theorem circle_equation_center_xaxis_radius_2 (a x y : ℝ) :
  (0:ℝ) < 2 ∧ (a - 1)^2 + 2^2 = 4 -> (x - 1)^2 + y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_center_xaxis_radius_2_l1087_108721


namespace NUMINAMATH_GPT_find_a3_plus_a5_l1087_108759

variable (a : ℕ → ℝ)
variable (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n)
variable (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25)

theorem find_a3_plus_a5 (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n) (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_plus_a5_l1087_108759


namespace NUMINAMATH_GPT_range_of_a_range_of_m_l1087_108714

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x < |1 - 2 * a|) ↔ a ∈ (Set.Iic (-3/2) ∪ Set.Ici (5/2)) := by sorry

theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 2 * Real.sqrt 6 * t + f m = 0) ↔ m ∈ (Set.Icc (-1) 2) := by sorry

end NUMINAMATH_GPT_range_of_a_range_of_m_l1087_108714


namespace NUMINAMATH_GPT_min_value_proven_l1087_108739

open Real

noncomputable def min_value (x y : ℝ) (h1 : log x + log y = 1) : Prop :=
  2 * x + 5 * y ≥ 20 ∧ (2 * x + 5 * y = 20 ↔ 2 * x = 5 * y ∧ x * y = 10)

theorem min_value_proven (x y : ℝ) (h1 : log x + log y = 1) :
  min_value x y h1 :=
sorry

end NUMINAMATH_GPT_min_value_proven_l1087_108739


namespace NUMINAMATH_GPT_largest_number_by_replacement_l1087_108701

theorem largest_number_by_replacement 
  (n : ℝ) (n_1 n_3 n_6 n_8 : ℝ)
  (h : n = -0.3168)
  (h1 : n_1 = -0.3468)
  (h3 : n_3 = -0.4168)
  (h6 : n_6 = -0.3148)
  (h8 : n_8 = -0.3164)
  : n_6 > n_1 ∧ n_6 > n_3 ∧ n_6 > n_8 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_largest_number_by_replacement_l1087_108701


namespace NUMINAMATH_GPT_movie_replay_count_l1087_108784

def movie_length_hours : ℝ := 1.5
def advertisement_length_minutes : ℝ := 20
def theater_operating_hours : ℝ := 11

theorem movie_replay_count :
  let movie_length_minutes := movie_length_hours * 60
  let total_showing_time_minutes := movie_length_minutes + advertisement_length_minutes
  let operating_time_minutes := theater_operating_hours * 60
  (operating_time_minutes / total_showing_time_minutes) = 6 :=
by
  sorry

end NUMINAMATH_GPT_movie_replay_count_l1087_108784


namespace NUMINAMATH_GPT_geometric_progression_condition_l1087_108708

theorem geometric_progression_condition {b : ℕ → ℝ} (b1_ne_b2 : b 1 ≠ b 2) (h : ∀ n, b (n + 2) = b n / b (n + 1)) :
  (∀ n, b (n+1) / b n = b 2 / b 1) ↔ b 1 = b 2^3 := sorry

end NUMINAMATH_GPT_geometric_progression_condition_l1087_108708


namespace NUMINAMATH_GPT_percentage_increase_sale_l1087_108712

theorem percentage_increase_sale (P S : ℝ) (hP : P > 0) (hS : S > 0) 
  (h1 : ∀ P S : ℝ, 0.7 * P * S * (1 + X / 100) = 1.26 * P * S) : 
  X = 80 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_sale_l1087_108712


namespace NUMINAMATH_GPT_sum_of_coefficients_l1087_108745

theorem sum_of_coefficients :
  (∃ a b c d e : ℤ, 512 * x ^ 3 + 27 = a * x * (c * x ^ 2 + d * x + e) + b * (c * x ^ 2 + d * x + e)) →
  (a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9) →
  a + b + c + d + e = 60 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1087_108745


namespace NUMINAMATH_GPT_incorrect_score_modulo_l1087_108770

theorem incorrect_score_modulo (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9) : 
  ∃ remainder : ℕ, remainder = (90 * a + 9 * b + c) % 9 ∧ 0 ≤ remainder ∧ remainder ≤ 9 := 
by
  sorry

end NUMINAMATH_GPT_incorrect_score_modulo_l1087_108770


namespace NUMINAMATH_GPT_cannot_fold_patternD_to_cube_l1087_108711

def patternA : Prop :=
  -- 5 squares arranged in a cross shape
  let squares := 5
  let shape  := "cross"
  squares = 5 ∧ shape = "cross"

def patternB : Prop :=
  -- 4 squares in a straight line
  let squares := 4
  let shape  := "line"
  squares = 4 ∧ shape = "line"

def patternC : Prop :=
  -- 3 squares in an L shape, and 2 squares attached to one end of the L making a T shape
  let squares := 5
  let shape  := "T"
  squares = 5 ∧ shape = "T"

def patternD : Prop :=
  -- 6 squares in a "+" shape with one extra square
  let squares := 7
  let shape  := "plus"
  squares = 7 ∧ shape = "plus"

theorem cannot_fold_patternD_to_cube :
  patternD → ¬ (patternA ∨ patternB ∨ patternC) :=
by
  sorry

end NUMINAMATH_GPT_cannot_fold_patternD_to_cube_l1087_108711


namespace NUMINAMATH_GPT_prove_trigonometric_identities_l1087_108768

variable {α : ℝ}

theorem prove_trigonometric_identities
  (h1 : 0 < α ∧ α < π)
  (h2 : Real.cos α = -3/5) :
  Real.tan α = -4/3 ∧
  (Real.cos (2 * α) - Real.cos (π / 2 + α) = 13/25) := 
by
  sorry

end NUMINAMATH_GPT_prove_trigonometric_identities_l1087_108768


namespace NUMINAMATH_GPT_box_third_dimension_length_l1087_108700

noncomputable def box_height (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  let total_volume := num_cubes * cube_volume
  total_volume / (length * width)

theorem box_third_dimension_length (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ)
  (h_num_cubes : num_cubes = 24)
  (h_cube_volume : cube_volume = 27)
  (h_length : length = 8)
  (h_width : width = 12) :
  box_height num_cubes cube_volume length width = 6.75 :=
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_box_third_dimension_length_l1087_108700


namespace NUMINAMATH_GPT_range_of_m_l1087_108787

theorem range_of_m (m x : ℝ) (h1 : (3 * x) / (x - 1) = m / (x - 1) + 2) (h2 : x ≥ 0) (h3 : x ≠ 1) : 
  m ≥ 2 ∧ m ≠ 3 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1087_108787


namespace NUMINAMATH_GPT_translation_correct_l1087_108747

-- Define the points in the Cartesian coordinate system
structure Point where
  x : ℤ
  y : ℤ

-- Given points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 2 }

-- Translated point A' (A₁)
def A₁ : Point := { x := 2, y := -1 }

-- Define the translation applied to a point
def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Calculate the translation vector from A to A'
def translationVector : Point :=
  { x := A₁.x - A.x, y := A₁.y - A.y }

-- Define the expected point B' (B₁)
def B₁ : Point := { x := 4, y := 1 }

-- Theorem statement
theorem translation_correct :
  translate B translationVector = B₁ :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_translation_correct_l1087_108747


namespace NUMINAMATH_GPT_inscribed_circle_radius_in_quarter_circle_l1087_108740

theorem inscribed_circle_radius_in_quarter_circle (R r : ℝ) (hR : R = 4) :
  (r + r * Real.sqrt 2 = R) ↔ r = 4 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_in_quarter_circle_l1087_108740


namespace NUMINAMATH_GPT_inequality_solution_l1087_108788

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1/4) ∧ (x - 2 > 0) → x > 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l1087_108788


namespace NUMINAMATH_GPT_consecutive_page_sum_l1087_108790

theorem consecutive_page_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_GPT_consecutive_page_sum_l1087_108790


namespace NUMINAMATH_GPT_combined_meows_l1087_108786

theorem combined_meows (first_cat_freq second_cat_freq third_cat_freq : ℕ) 
  (time : ℕ) 
  (h1 : first_cat_freq = 3)
  (h2 : second_cat_freq = 2 * first_cat_freq)
  (h3 : third_cat_freq = second_cat_freq / 3)
  (h4 : time = 5) : 
  first_cat_freq * time + second_cat_freq * time + third_cat_freq * time = 55 := 
by
  sorry

end NUMINAMATH_GPT_combined_meows_l1087_108786


namespace NUMINAMATH_GPT_max_chain_length_in_subdivided_triangle_l1087_108756

-- Define an equilateral triangle subdivision
structure EquilateralTriangleSubdivided (n : ℕ) :=
(n_squares : ℕ)
(n_squares_eq : n_squares = n^2)

-- Define the problem's chain concept
def maximum_chain_length (n : ℕ) : ℕ :=
n^2 - n + 1

-- Main statement
theorem max_chain_length_in_subdivided_triangle
  (n : ℕ) (triangle : EquilateralTriangleSubdivided n) :
  maximum_chain_length n = n^2 - n + 1 :=
by sorry

end NUMINAMATH_GPT_max_chain_length_in_subdivided_triangle_l1087_108756


namespace NUMINAMATH_GPT_rectangle_area_pairs_l1087_108760

theorem rectangle_area_pairs :
  { p : ℕ × ℕ | p.1 * p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0 } = { (1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1) } :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_pairs_l1087_108760


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1087_108742

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 17 :=
by
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  show sum_of_coordinates = 17
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1087_108742


namespace NUMINAMATH_GPT_total_amount_received_l1087_108706
noncomputable section

variables (B : ℕ) (H1 : (1 / 3 : ℝ) * B = 50)
theorem total_amount_received (H2 : (2 / 3 : ℝ) * B = 100) (H3 : ∀ (x : ℕ), x = 5): 
  100 * 5 = 500 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_received_l1087_108706


namespace NUMINAMATH_GPT_drum_capacity_ratio_l1087_108743

variable {C_X C_Y : ℝ}

theorem drum_capacity_ratio (h1 : C_X / 2 + C_Y / 2 = 3 * C_Y / 4) : C_Y / C_X = 2 :=
by
  have h2: C_X / 2 = C_Y / 4 := by
    sorry
  have h3: C_X = C_Y / 2 := by
    sorry
  rw [h3]
  have h4: C_Y / (C_Y / 2) = 2 := by
    sorry
  exact h4

end NUMINAMATH_GPT_drum_capacity_ratio_l1087_108743


namespace NUMINAMATH_GPT_negation_proposition_l1087_108785

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1087_108785


namespace NUMINAMATH_GPT_max_volume_rectangular_frame_l1087_108773

theorem max_volume_rectangular_frame (L W H : ℝ) (h1 : 2 * W = L) (h2 : 4 * (L + W) + 4 * H = 18) :
  volume = (2 * 1 * 1.5 : ℝ) := 
sorry

end NUMINAMATH_GPT_max_volume_rectangular_frame_l1087_108773


namespace NUMINAMATH_GPT_gg1_eq_13_l1087_108781

def g (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1
else if n < 6 then 2 * n + 3
else 4 * n - 2

theorem gg1_eq_13 : g (g (g 1)) = 13 :=
by
  sorry

end NUMINAMATH_GPT_gg1_eq_13_l1087_108781


namespace NUMINAMATH_GPT_paul_coins_difference_l1087_108776

/-- Paul owes Paula 145 cents and has a pocket full of 10-cent coins, 
20-cent coins, and 50-cent coins. Prove that the difference between 
the largest and smallest number of coins he can use to pay her is 9. -/
theorem paul_coins_difference :
  ∃ min_coins max_coins : ℕ, 
    (min_coins = 5 ∧ max_coins = 14) ∧ (max_coins - min_coins = 9) :=
by
  sorry

end NUMINAMATH_GPT_paul_coins_difference_l1087_108776


namespace NUMINAMATH_GPT_gcd_m_n_l1087_108744

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l1087_108744


namespace NUMINAMATH_GPT_artist_painting_time_l1087_108765

theorem artist_painting_time (hours_per_week : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → weeks = 4 → total_paintings = 40 →
  ((hours_per_week * weeks) / total_paintings) = 3 := by
  intros h_hours h_weeks h_paintings
  sorry

end NUMINAMATH_GPT_artist_painting_time_l1087_108765


namespace NUMINAMATH_GPT_quadratic_transformation_l1087_108741

theorem quadratic_transformation (a b c : ℝ) (h : a * x^2 + b * x + c = 5 * (x + 2)^2 - 7) :
  ∃ (n m g : ℝ), 2 * a * x^2 + 2 * b * x + 2 * c = n * (x - g)^2 + m ∧ g = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_transformation_l1087_108741


namespace NUMINAMATH_GPT_log_m_n_iff_m_minus_1_n_minus_1_l1087_108719

theorem log_m_n_iff_m_minus_1_n_minus_1 (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) (h3 : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) :=
sorry

end NUMINAMATH_GPT_log_m_n_iff_m_minus_1_n_minus_1_l1087_108719


namespace NUMINAMATH_GPT_probability_of_different_groups_is_correct_l1087_108757

-- Define the number of total members and groups
def num_groups : ℕ := 6
def members_per_group : ℕ := 3
def total_members : ℕ := num_groups * members_per_group

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 3 people from different groups
noncomputable def probability_different_groups : ℚ :=
  binom num_groups 3 / binom total_members 3

-- State the theorem we want to prove
theorem probability_of_different_groups_is_correct :
  probability_different_groups = 5 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_different_groups_is_correct_l1087_108757


namespace NUMINAMATH_GPT_correct_calculation_is_D_l1087_108749

theorem correct_calculation_is_D 
  (a b x : ℝ) :
  ¬ (5 * a + 2 * b = 7 * a * b) ∧
  ¬ (x ^ 2 - 3 * x ^ 2 = -2) ∧
  ¬ (7 * a - b + (7 * a + b) = 0) ∧
  (4 * a - (-7 * a) = 11 * a) :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_is_D_l1087_108749


namespace NUMINAMATH_GPT_quadratic_function_a_value_l1087_108732

theorem quadratic_function_a_value (a : ℝ) (h₁ : a ≠ 1) :
  (∀ x : ℝ, ∃ c₀ c₁ c₂ : ℝ, (a-1) * x^(a^2 + 1) + 2 * x + 3 = c₂ * x^2 + c₁ * x + c₀) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_a_value_l1087_108732


namespace NUMINAMATH_GPT_find_a_range_l1087_108720

theorem find_a_range (a : ℝ) (x : ℝ) (h1 : a * x < 6) (h2 : (3 * x - 6 * a) / 2 > a / 3 - 1) :
  a ≤ -3 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_range_l1087_108720


namespace NUMINAMATH_GPT_range_of_a_l1087_108792

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a x : ℝ) := a * Real.sqrt x
noncomputable def f' (x₀ : ℝ) := Real.exp x₀
noncomputable def g' (a t : ℝ) := a / (2 * Real.sqrt t)

theorem range_of_a (a : ℝ) (x₀ t : ℝ) (hx₀ : x₀ = 1 - t) (ht_pos : t > 0)
  (h1 : f x₀ = Real.exp x₀)
  (h2 : g a t = a * Real.sqrt t)
  (h3 : f x₀ = g' a t)
  (h4 : (Real.exp x₀ - a * Real.sqrt t) / (x₀ - t) = Real.exp x₀) :
    0 < a ∧ a ≤ Real.sqrt (2 * Real.exp 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1087_108792


namespace NUMINAMATH_GPT_minimum_value_of_f_l1087_108736

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∃ a > 2, (∀ x > 2, f x ≥ f a) ∧ a = 3 := by
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1087_108736


namespace NUMINAMATH_GPT_find_x_in_sequence_l1087_108764

theorem find_x_in_sequence
  (x d1 d2 : ℤ)
  (h1 : d1 = x - 1370)
  (h2 : d2 = 1070 - x)
  (h3 : -180 - 1070 = -1250)
  (h4 : -6430 - (-180) = -6250)
  (h5 : d2 - d1 = 5000) :
  x = 3720 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_find_x_in_sequence_l1087_108764


namespace NUMINAMATH_GPT_fg_minus_gf_l1087_108728

-- Definitions provided by the conditions
def f (x : ℝ) : ℝ := 4 * x + 8
def g (x : ℝ) : ℝ := 2 * x - 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -17 := 
  sorry

end NUMINAMATH_GPT_fg_minus_gf_l1087_108728


namespace NUMINAMATH_GPT_binomial_coefficient_sum_l1087_108753

theorem binomial_coefficient_sum :
  Nat.choose 10 3 + Nat.choose 10 2 = 165 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_sum_l1087_108753


namespace NUMINAMATH_GPT_longest_third_side_of_triangle_l1087_108750

theorem longest_third_side_of_triangle {a b : ℕ} (ha : a = 8) (hb : b = 9) : 
  ∃ c : ℕ, 1 < c ∧ c < 17 ∧ ∀ (d : ℕ), (1 < d ∧ d < 17) → d ≤ c :=
by
  sorry

end NUMINAMATH_GPT_longest_third_side_of_triangle_l1087_108750


namespace NUMINAMATH_GPT_train_speed_proof_l1087_108702

noncomputable def train_speed (L : ℕ) (t : ℝ) (v_m : ℝ) : ℝ :=
  let v_m_m_s := v_m * (1000 / 3600)
  let v_rel := L / t
  v_rel + v_m_m_s

theorem train_speed_proof
  (L : ℕ)
  (t : ℝ)
  (v_m : ℝ)
  (hL : L = 900)
  (ht : t = 53.99568034557235)
  (hv_m : v_m = 3)
  : train_speed L t v_m = 63.0036 :=
  by sorry

end NUMINAMATH_GPT_train_speed_proof_l1087_108702


namespace NUMINAMATH_GPT_tom_average_score_increase_l1087_108738

def initial_scores : List ℕ := [72, 78, 81]
def fourth_exam_score : ℕ := 90

theorem tom_average_score_increase :
  let initial_avg := (initial_scores.sum : ℚ) / (initial_scores.length : ℚ)
  let total_score_after_fourth := initial_scores.sum + fourth_exam_score
  let new_avg := (total_score_after_fourth : ℚ) / (initial_scores.length + 1 : ℚ)
  new_avg - initial_avg = 3.25 := by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tom_average_score_increase_l1087_108738


namespace NUMINAMATH_GPT_problem_statement_l1087_108762

theorem problem_statement (m n : ℝ) (h : m + n = 1 / 2 * m * n) : (m - 2) * (n - 2) = 4 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1087_108762


namespace NUMINAMATH_GPT_recurrence_relation_l1087_108758

def u (n : ℕ) : ℕ := sorry

theorem recurrence_relation (n : ℕ) : 
  u (n + 1) = (n + 1) * u n - (n * (n - 1)) / 2 * u (n - 2) :=
sorry

end NUMINAMATH_GPT_recurrence_relation_l1087_108758


namespace NUMINAMATH_GPT_coords_P_origin_l1087_108769

variable (x y : Int)
def point_P := (-5, 3)

theorem coords_P_origin : point_P = (-5, 3) := 
by 
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_coords_P_origin_l1087_108769


namespace NUMINAMATH_GPT_xy_divides_x2_plus_2y_minus_1_l1087_108761

theorem xy_divides_x2_plus_2y_minus_1 (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2 * y - 1) ↔ (∃ t : ℕ, t > 0 ∧ ((x = 1 ∧ y = t) ∨ (x = 2 * t - 1 ∧ y = t)
  ∨ (x = 3 ∧ y = 8) ∨ (x = 5 ∧ y = 8))) :=
by
  sorry

end NUMINAMATH_GPT_xy_divides_x2_plus_2y_minus_1_l1087_108761


namespace NUMINAMATH_GPT_no_real_solutions_for_equation_l1087_108766

theorem no_real_solutions_for_equation :
  ¬ (∃ x : ℝ, (2 * x - 3 * x + 7)^2 + 2 = -|2 * x|) :=
by 
-- proof will go here
sorry

end NUMINAMATH_GPT_no_real_solutions_for_equation_l1087_108766


namespace NUMINAMATH_GPT_mod_add_5000_l1087_108783

theorem mod_add_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 :=
sorry

end NUMINAMATH_GPT_mod_add_5000_l1087_108783


namespace NUMINAMATH_GPT_solve_for_x_l1087_108752

theorem solve_for_x (x : ℝ) (h₁: 0.45 * x = 0.15 * (1 + x)) : x = 0.5 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1087_108752


namespace NUMINAMATH_GPT_area_increase_is_50_l1087_108726

def length := 13
def width := 10
def length_new := length + 2
def width_new := width + 2
def area_original := length * width
def area_new := length_new * width_new
def area_increase := area_new - area_original

theorem area_increase_is_50 : area_increase = 50 :=
by
  -- Here we will include the steps to prove the theorem if required
  sorry

end NUMINAMATH_GPT_area_increase_is_50_l1087_108726


namespace NUMINAMATH_GPT_simplify_div_l1087_108710

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end NUMINAMATH_GPT_simplify_div_l1087_108710


namespace NUMINAMATH_GPT_total_marbles_l1087_108748

theorem total_marbles
  (R B Y : ℕ)  -- Red, Blue, and Yellow marbles as natural numbers
  (h_ratio : 2 * (R + B + Y) = 9 * Y)  -- The ratio condition translated
  (h_yellow : Y = 36)  -- The number of yellow marbles condition
  : R + B + Y = 81 :=  -- Statement that the total number of marbles is 81
sorry

end NUMINAMATH_GPT_total_marbles_l1087_108748


namespace NUMINAMATH_GPT_area_of_garden_l1087_108731

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end NUMINAMATH_GPT_area_of_garden_l1087_108731


namespace NUMINAMATH_GPT_find_fourth_number_l1087_108767

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_find_fourth_number_l1087_108767


namespace NUMINAMATH_GPT_regular_pay_limit_l1087_108734

theorem regular_pay_limit (x : ℝ) : 3 * x + 6 * 13 = 198 → x = 40 :=
by
  intro h
  -- proof skipped
  sorry

end NUMINAMATH_GPT_regular_pay_limit_l1087_108734


namespace NUMINAMATH_GPT_todd_money_after_repay_l1087_108722

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_todd_money_after_repay_l1087_108722


namespace NUMINAMATH_GPT_range_of_inclination_angle_l1087_108705

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end NUMINAMATH_GPT_range_of_inclination_angle_l1087_108705


namespace NUMINAMATH_GPT_num_ordered_pairs_l1087_108724

theorem num_ordered_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x * y = 4410) : 
  ∃ (n : ℕ), n = 36 :=
sorry

end NUMINAMATH_GPT_num_ordered_pairs_l1087_108724


namespace NUMINAMATH_GPT_travel_cost_from_B_to_C_l1087_108715

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def travel_cost_by_air (distance : ℝ) (booking_fee : ℝ) (per_km_cost : ℝ) : ℝ :=
  booking_fee + (distance * per_km_cost)

theorem travel_cost_from_B_to_C :
  let AC := 4000
  let AB := 4500
  let BC := Real.sqrt (AB^2 - AC^2)
  let booking_fee := 120
  let per_km_cost := 0.12
  travel_cost_by_air BC booking_fee per_km_cost = 367.39 := by
  sorry

end NUMINAMATH_GPT_travel_cost_from_B_to_C_l1087_108715


namespace NUMINAMATH_GPT_range_of_a_l1087_108779

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1087_108779


namespace NUMINAMATH_GPT_estimate_students_less_than_2_hours_probability_one_male_one_female_l1087_108723

-- Definitions from the conditions
def total_students_surveyed : ℕ := 40
def total_grade_ninth_students : ℕ := 400
def freq_0_1 : ℕ := 8
def freq_1_2 : ℕ := 20
def freq_2_3 : ℕ := 7
def freq_3_4 : ℕ := 5
def male_students_at_least_3_hours : ℕ := 2
def female_students_at_least_3_hours : ℕ := 3

-- Question 1 proof statement
theorem estimate_students_less_than_2_hours :
  total_grade_ninth_students * (freq_0_1 + freq_1_2) / total_students_surveyed = 280 :=
by sorry

-- Question 2 proof statement
theorem probability_one_male_one_female :
  (male_students_at_least_3_hours * female_students_at_least_3_hours) / (Nat.choose 5 2) = (3 / 5) :=
by sorry

end NUMINAMATH_GPT_estimate_students_less_than_2_hours_probability_one_male_one_female_l1087_108723


namespace NUMINAMATH_GPT_product_of_two_smaller_numbers_is_85_l1087_108774

theorem product_of_two_smaller_numbers_is_85
  (A B C : ℝ)
  (h1 : B = 10)
  (h2 : C - B = B - A)
  (h3 : B * C = 115) :
  A * B = 85 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_smaller_numbers_is_85_l1087_108774


namespace NUMINAMATH_GPT_fedya_deposit_l1087_108789

theorem fedya_deposit (n : ℕ) (h1 : n < 30) (h2 : 847 * 100 % (100 - n) = 0) : 
  (847 * 100 / (100 - n) = 1100) :=
by
  sorry

end NUMINAMATH_GPT_fedya_deposit_l1087_108789


namespace NUMINAMATH_GPT_find_m_and_union_A_B_l1087_108751

variable (m : ℝ)
noncomputable def A := ({3, 4, m^2 - 3 * m - 1} : Set ℝ)
noncomputable def B := ({2 * m, -3} : Set ℝ)

theorem find_m_and_union_A_B (h : A m ∩ B m = ({-3} : Set ℝ)) :
  m = 1 ∧ A m ∪ B m = ({-3, 2, 3, 4} : Set ℝ) :=
sorry

end NUMINAMATH_GPT_find_m_and_union_A_B_l1087_108751


namespace NUMINAMATH_GPT_value_of_f_at_112_5_l1087_108713

noncomputable def f : ℝ → ℝ := sorry

lemma f_even_func (x : ℝ) : f x = f (-x) := sorry
lemma f_func_eq (x : ℝ) : f x + f (x + 1) = 4 := sorry
lemma f_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x + 12 := sorry

theorem value_of_f_at_112_5 : f 112.5 = 2 := sorry

end NUMINAMATH_GPT_value_of_f_at_112_5_l1087_108713


namespace NUMINAMATH_GPT_lending_rate_is_7_percent_l1087_108735

-- Conditions
def principal : ℝ := 5000
def borrowing_rate : ℝ := 0.04  -- 4% p.a. simple interest
def time : ℕ := 2  -- 2 years
def gain_per_year : ℝ := 150

-- Proof of the final statement
theorem lending_rate_is_7_percent :
  let borrowing_interest := principal * borrowing_rate * time / 100
  let interest_per_year := borrowing_interest / 2
  let total_interest_earned_per_year := interest_per_year + gain_per_year
  (total_interest_earned_per_year * 100) / principal = 7 :=
by
  sorry

end NUMINAMATH_GPT_lending_rate_is_7_percent_l1087_108735


namespace NUMINAMATH_GPT_cube_side_length_l1087_108727

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = (6 * n^3) / 3) : n = 3 :=
sorry

end NUMINAMATH_GPT_cube_side_length_l1087_108727


namespace NUMINAMATH_GPT_num_teams_is_seventeen_l1087_108746

-- Each team faces all other teams 10 times and there are 1360 games in total.
def total_teams (n : ℕ) : Prop := 1360 = (n * (n - 1) * 10) / 2

theorem num_teams_is_seventeen : ∃ n : ℕ, total_teams n ∧ n = 17 := 
by 
  sorry

end NUMINAMATH_GPT_num_teams_is_seventeen_l1087_108746


namespace NUMINAMATH_GPT_sum_of_prime_factors_77_l1087_108754

theorem sum_of_prime_factors_77 : (7 + 11 = 18) := by
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_77_l1087_108754


namespace NUMINAMATH_GPT_ratio_perimeters_not_integer_l1087_108771

theorem ratio_perimeters_not_integer
  (a k l : ℤ) (h_a_pos : a > 0) (h_k_pos : k > 0) (h_l_pos : l > 0)
  (h_area : a^2 = k * l) :
  ¬ ∃ n : ℤ, n = (k + l) / (2 * a) :=
by
  sorry

end NUMINAMATH_GPT_ratio_perimeters_not_integer_l1087_108771


namespace NUMINAMATH_GPT_correct_option_l1087_108777

theorem correct_option (a b : ℝ) : (ab) ^ 2 = a ^ 2 * b ^ 2 :=
by sorry

end NUMINAMATH_GPT_correct_option_l1087_108777


namespace NUMINAMATH_GPT_jodi_walks_days_l1087_108729

section
variables {d : ℕ} -- d is the number of days Jodi walks per week

theorem jodi_walks_days (h : 1 * d + 2 * d + 3 * d + 4 * d = 60) : d = 6 := by
  sorry

end

end NUMINAMATH_GPT_jodi_walks_days_l1087_108729
