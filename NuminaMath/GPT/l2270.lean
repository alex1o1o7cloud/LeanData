import Mathlib

namespace annual_income_is_32000_l2270_227030

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l2270_227030


namespace simplify_and_evaluate_expression_l2270_227014

-- Define a and b with given values
def a := 1 / 2
def b := 1 / 3

-- Define the expression
def expr := 5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b)

-- State the theorem
theorem simplify_and_evaluate_expression : expr = 2 / 3 := 
by
  -- Proof can be inserted here
  sorry

end simplify_and_evaluate_expression_l2270_227014


namespace mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l2270_227025

-- Definitions for molar masses used in calculations
def molar_mass_Ca := 40.08
def molar_mass_O := 16.00
def molar_mass_H := 1.01
def molar_mass_Na := 22.99
def molar_mass_C := 12.01
def molar_mass_K := 39.10
def molar_mass_S := 32.07

-- Molar masses of the compounds
def molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_Na2CO3 := 2 * molar_mass_Na + molar_mass_C + 3 * molar_mass_O
def molar_mass_K2SO4 := 2 * molar_mass_K + molar_mass_S + 4 * molar_mass_O

-- Mass of O in each compound
def mass_O_CaOH2 := 2 * molar_mass_O
def mass_O_Na2CO3 := 3 * molar_mass_O
def mass_O_K2SO4 := 4 * molar_mass_O

-- Mass percentages of O in each compound
def mass_percent_O_CaOH2 := (mass_O_CaOH2 / molar_mass_CaOH2) * 100
def mass_percent_O_Na2CO3 := (mass_O_Na2CO3 / molar_mass_Na2CO3) * 100
def mass_percent_O_K2SO4 := (mass_O_K2SO4 / molar_mass_K2SO4) * 100

theorem mass_percent_O_CaOH2_is_correct :
  mass_percent_O_CaOH2 = 43.19 := by sorry

theorem mass_percent_O_Na2CO3_is_correct :
  mass_percent_O_Na2CO3 = 45.29 := by sorry

theorem mass_percent_O_K2SO4_is_correct :
  mass_percent_O_K2SO4 = 36.73 := by sorry

end mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l2270_227025


namespace least_possible_integer_discussed_l2270_227060
open Nat

theorem least_possible_integer_discussed (N : ℕ) (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → k ≠ 8 ∧ k ≠ 9 → k ∣ N) : N = 2329089562800 :=
sorry

end least_possible_integer_discussed_l2270_227060


namespace find_k_l2270_227050

theorem find_k :
  ∀ (k : ℤ),
    (∃ a1 a2 a3 : ℤ,
        a1 = 49 + k ∧
        a2 = 225 + k ∧
        a3 = 484 + k ∧
        2 * a2 = a1 + a3) →
    k = 324 :=
by
  sorry

end find_k_l2270_227050


namespace total_wood_needed_l2270_227046

theorem total_wood_needed : 
      (4 * 4 + 4 * (4 * 5)) + 
      (10 * 6 + 10 * (6 - 3)) + 
      (8 * 5.5) + 
      (6 * (5.5 * 2) + 6 * (5.5 * 1.5)) = 345.5 := 
by 
  sorry

end total_wood_needed_l2270_227046


namespace total_distance_collinear_centers_l2270_227041

theorem total_distance_collinear_centers (r1 r2 r3 : ℝ) (d12 d13 d23 : ℝ) 
  (h1 : r1 = 6) 
  (h2 : r2 = 14) 
  (h3 : d12 = r1 + r2) 
  (h4 : d13 = r3 - r1) 
  (h5 : d23 = r3 - r2) :
  d13 = d12 + r1 := by
  -- proof follows here
  sorry

end total_distance_collinear_centers_l2270_227041


namespace equation_has_at_least_two_distinct_roots_l2270_227080

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l2270_227080


namespace piles_can_be_combined_l2270_227024

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l2270_227024


namespace find_c_l2270_227033

theorem find_c (c : ℝ) : (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + x - c < 0) → c = 2 :=
by
  intros h
  -- Sorry to skip the proof
  sorry

end find_c_l2270_227033


namespace max_n_base_10_l2270_227031

theorem max_n_base_10:
  ∃ (A B C n: ℕ), (A < 5 ∧ B < 5 ∧ C < 5) ∧
                 (n = 25 * A + 5 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧ 
                 (∀ (A' B' C' n': ℕ), 
                 (A' < 5 ∧ B' < 5 ∧ C' < 5) ∧ (n' = 25 * A' + 5 * B' + C') ∧ 
                 (n' = 81 * C' + 9 * B' + A') → n' ≤ n) →
  n = 111 :=
by {
    sorry
}

end max_n_base_10_l2270_227031


namespace range_of_m_l2270_227011

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x + m) * (2 - x) < 1) ↔ (-4 < m ∧ m < 0) :=
sorry

end range_of_m_l2270_227011


namespace total_spending_is_140_l2270_227009

-- Define definitions for each day's spending based on the conditions.
def monday_spending : ℕ := 6
def tuesday_spending : ℕ := 2 * monday_spending
def wednesday_spending : ℕ := 2 * (monday_spending + tuesday_spending)
def thursday_spending : ℕ := (monday_spending + tuesday_spending + wednesday_spending) / 3
def friday_spending : ℕ := thursday_spending - 4
def saturday_spending : ℕ := friday_spending + (friday_spending / 2)
def sunday_spending : ℕ := tuesday_spending + saturday_spending

-- The total spending for the week.
def total_spending : ℕ := 
  monday_spending + 
  tuesday_spending + 
  wednesday_spending + 
  thursday_spending + 
  friday_spending + 
  saturday_spending + 
  sunday_spending

-- The theorem to prove that the total spending is $140.
theorem total_spending_is_140 : total_spending = 140 := 
  by {
    -- Due to the problem's requirement, we skip the proof steps.
    sorry
  }

end total_spending_is_140_l2270_227009


namespace jaden_toy_cars_problem_l2270_227026

theorem jaden_toy_cars_problem :
  let initial := 14
  let bought := 28
  let birthday := 12
  let to_vinnie := 3
  let left := 43
  let total := initial + bought + birthday
  let after_vinnie := total - to_vinnie
  (after_vinnie - left = 8) :=
by
  sorry

end jaden_toy_cars_problem_l2270_227026


namespace total_spending_l2270_227070

-- Conditions
def pop_spending : ℕ := 15
def crackle_spending : ℕ := 3 * pop_spending
def snap_spending : ℕ := 2 * crackle_spending

-- Theorem stating the total spending
theorem total_spending : snap_spending + crackle_spending + pop_spending = 150 :=
by
  sorry

end total_spending_l2270_227070


namespace part1_part2_l2270_227066

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 6

-- Part (I)
theorem part1 (a : ℝ) (h : a = 5) : ∀ x : ℝ, f x 5 < 0 ↔ -3 < x ∧ x < -2 := by
  sorry

-- Part (II)
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by
  sorry

end part1_part2_l2270_227066


namespace K_3_15_10_eq_151_30_l2270_227055

def K (a b c : ℕ) : ℚ := (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a

theorem K_3_15_10_eq_151_30 : K 3 15 10 = 151 / 30 := 
by
  sorry

end K_3_15_10_eq_151_30_l2270_227055


namespace three_digit_numbers_l2270_227015

theorem three_digit_numbers (N : ℕ) (a b c : ℕ) 
  (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : b ≤ 9 ∧ c ≤ 9)
  (h4 : a - b + c % 11 = 0)
  (h5 : N % 11 = 0)
  (h6 : N = 11 * (a^2 + b^2 + c^2)) :
  N = 550 ∨ N = 803 :=
  sorry

end three_digit_numbers_l2270_227015


namespace ff_of_10_eq_2_l2270_227007

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then x^2 + 1 else Real.log x

theorem ff_of_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end ff_of_10_eq_2_l2270_227007


namespace max_square_test_plots_l2270_227048

theorem max_square_test_plots (h_field_dims : (24 : ℝ) = 24 ∧ (52 : ℝ) = 52)
    (h_total_fencing : 1994 = 1994)
    (h_partitioning : ∀ (n : ℤ), n % 6 = 0 → n ≤ 19 → 
      (104 * n - 76 ≤ 1994) → (n / 6 * 13)^2 = 702) :
    ∃ n : ℤ, (n / 6 * 13)^2 = 702 := sorry

end max_square_test_plots_l2270_227048


namespace part1_part2_l2270_227036

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≤ 0) : a ≥ 1 / Real.exp 1 :=
  sorry

noncomputable def g (x b : ℝ) : ℝ := Real.log x + 1/2 * x^2 - (b + 1) * x

theorem part2 (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ 3/2) (h2 : x1 < x2) (hx3 : g x1 b - g x2 b ≥ k) : k ≤ 15/8 - 2 * Real.log 2 :=
  sorry

end part1_part2_l2270_227036


namespace emily_subtracts_99_l2270_227013

theorem emily_subtracts_99 (a b : ℕ) : (a = 50) → (b = 1) → (49^2 = 50^2 - 99) :=
by
  sorry

end emily_subtracts_99_l2270_227013


namespace gcd_689_1021_l2270_227099

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 :=
by sorry

end gcd_689_1021_l2270_227099


namespace sum_of_powers_eight_l2270_227021

variable {a b : ℝ}

theorem sum_of_powers_eight :
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^8 + b^8 = 47 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  -- Proof to be filled in
  sorry

end sum_of_powers_eight_l2270_227021


namespace perimeter_of_triangle_hyperbola_l2270_227019

theorem perimeter_of_triangle_hyperbola (x y : ℝ) (F1 F2 A B : ℝ) :
  (x^2 / 16) - (y^2 / 9) = 1 →
  |A - F2| - |A - F1| = 8 →
  |B - F2| - |B - F1| = 8 →
  |B - A| = 5 →
  |A - F2| + |B - F2| + |B - A| = 26 :=
by
  sorry

end perimeter_of_triangle_hyperbola_l2270_227019


namespace quadratic_factor_n_l2270_227035

theorem quadratic_factor_n (n : ℤ) (h : ∃ m : ℤ, (x + 5) * (x + m) = x^2 + 7 * x + n) : n = 10 :=
sorry

end quadratic_factor_n_l2270_227035


namespace maria_green_beans_l2270_227002

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l2270_227002


namespace alternating_intersections_l2270_227084

theorem alternating_intersections (n : ℕ)
  (roads : Fin n → ℝ → ℝ) -- Roads are functions from reals to reals
  (h_straight : ∀ (i : Fin n), ∃ (a b : ℝ), ∀ x, roads i x = a * x + b) 
  (h_intersect : ∀ (i j : Fin n), i ≠ j → ∃ x, roads i x = roads j x)
  (h_two_roads : ∀ (x y : ℝ), ∃! (i j : Fin n), i ≠ j ∧ roads i x = roads j y) :
  ∃ (design : ∀ (i : Fin n), ℝ → Prop), 
  -- ensuring alternation, road 'i' alternates crossings with other roads 
  (∀ (i : Fin n) (x y : ℝ), 
    roads i x = roads i y → (design i x ↔ ¬design i y)) := sorry

end alternating_intersections_l2270_227084


namespace arithmetic_sequence_k_value_l2270_227078

theorem arithmetic_sequence_k_value (a : ℕ → ℤ) (S: ℕ → ℤ)
    (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
    (h2 : S 11 = S 4)
    (h3 : a 1 = 1)
    (h4 : ∃ k, a k + a 4 = 0) :
    ∃ k, k = 12 :=
by 
  sorry

end arithmetic_sequence_k_value_l2270_227078


namespace percentage_of_useful_items_l2270_227072

theorem percentage_of_useful_items
  (junk_percentage : ℚ)
  (useful_items junk_items total_items : ℕ)
  (h1 : junk_percentage = 0.70)
  (h2 : useful_items = 8)
  (h3 : junk_items = 28)
  (h4 : junk_percentage * total_items = junk_items) :
  (useful_items : ℚ) / (total_items : ℚ) * 100 = 20 :=
sorry

end percentage_of_useful_items_l2270_227072


namespace return_trip_time_l2270_227034

variable {d p w : ℝ} -- Distance, plane's speed in calm air, wind speed

theorem return_trip_time (h1 : d = 75 * (p - w)) 
                         (h2 : d / (p + w) = d / p - 10) :
                         (d / (p + w) = 15 ∨ d / (p + w) = 50) :=
sorry

end return_trip_time_l2270_227034


namespace decimal_zeros_l2270_227054

theorem decimal_zeros (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ (n : ℕ), n = 2 ∧ (∃ d : ℚ, d = 5 / 2520 ∧ ↑d = 0.004) :=
by
  -- We assume the factorization of 2520 is correct
  have h_fact := h
  -- We need to prove there are exactly 2 zeros between the decimal point and the first non-zero digit
  sorry

end decimal_zeros_l2270_227054


namespace allowance_amount_l2270_227005

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end allowance_amount_l2270_227005


namespace geometric_solution_l2270_227076

theorem geometric_solution (x y : ℝ) (h : x^2 + 2 * y^2 - 10 * x + 12 * y + 43 = 0) : x = 5 ∧ y = -3 := 
  by sorry

end geometric_solution_l2270_227076


namespace circles_tangent_l2270_227083

theorem circles_tangent
  (rA rB rC rD rF : ℝ) (rE : ℚ) (m n : ℕ)
  (m_n_rel_prime : Int.gcd m n = 1)
  (rA_pos : 0 < rA) (rB_pos : 0 < rB)
  (rC_pos : 0 < rC) (rD_pos : 0 < rD)
  (rF_pos : 0 < rF)
  (inscribed_triangle_in_A : True)  -- Triangle T is inscribed in circle A
  (B_tangent_A : True)  -- Circle B is internally tangent to circle A
  (C_tangent_A : True)  -- Circle C is internally tangent to circle A
  (D_tangent_A : True)  -- Circle D is internally tangent to circle A
  (B_externally_tangent_E : True)  -- Circle B is externally tangent to circle E
  (C_externally_tangent_E : True)  -- Circle C is externally tangent to circle E
  (D_externally_tangent_E : True)  -- Circle D is externally tangent to circle E
  (F_tangent_A : True)  -- Circle F is internally tangent to circle A at midpoint of side opposite to B's tangency
  (F_externally_tangent_E : True)  -- Circle F is externally tangent to circle E
  (rA_eq : rA = 12) (rB_eq : rB = 5)
  (rC_eq : rC = 3) (rD_eq : rD = 2)
  (rF_eq : rF = 1)
  (rE_eq : rE = m / n)
  : m + n = 23 :=
by
  sorry

end circles_tangent_l2270_227083


namespace sqrt_9_eq_pm3_l2270_227027

theorem sqrt_9_eq_pm3 : ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_9_eq_pm3_l2270_227027


namespace solve_triangle_l2270_227006

open Real

noncomputable def triangle_sides_angles (a b c A B C : ℝ) : Prop :=
  b^2 - (2 * (sqrt 3 / 3) * b * c * sin A) + c^2 = a^2

theorem solve_triangle 
  (b c : ℝ) (hb : b = 2) (hc : c = 3)
  (h : triangle_sides_angles a b c A B C) : 
  (A = π / 3) ∧ 
  (a = sqrt 7) ∧ 
  (sin (2 * B - A) = 3 * sqrt 3 / 14) := 
by
  sorry

end solve_triangle_l2270_227006


namespace right_triangle_hypotenuse_equals_area_l2270_227086

/-- Given a right triangle where the hypotenuse is equal to the area, 
    show that the scaling factor x satisfies the equation. -/
theorem right_triangle_hypotenuse_equals_area 
  (m n x : ℝ) (h_hyp: (m^2 + n^2) * x = mn * (m^2 - n^2) * x^2) :
  x = (m^2 + n^2) / (mn * (m^2 - n^2)) := 
by
  sorry

end right_triangle_hypotenuse_equals_area_l2270_227086


namespace sequence_general_formula_l2270_227042

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1)) ∧ a n = 1 / (3 * n - 2) :=
by sorry

end sequence_general_formula_l2270_227042


namespace graph_transformation_point_l2270_227096

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end graph_transformation_point_l2270_227096


namespace lateral_surface_area_of_cylinder_l2270_227040

theorem lateral_surface_area_of_cylinder :
  (∀ (side_length : ℕ), side_length = 10 → 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 100 * Real.pi) :=
by
  sorry

end lateral_surface_area_of_cylinder_l2270_227040


namespace find_angle_l2270_227016

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l2270_227016


namespace team_score_is_correct_l2270_227010

-- Definitions based on given conditions
def connor_score : ℕ := 2
def amy_score : ℕ := connor_score + 4
def jason_score : ℕ := 2 * amy_score
def combined_score : ℕ := connor_score + amy_score + jason_score
def emily_score : ℕ := 3 * combined_score
def team_score : ℕ := connor_score + amy_score + jason_score + emily_score

-- Theorem stating team_score should be 80
theorem team_score_is_correct : team_score = 80 := by
  sorry

end team_score_is_correct_l2270_227010


namespace not_characteristic_of_algorithm_l2270_227028

def characteristic_of_algorithm (c : String) : Prop :=
  c = "Abstraction" ∨ c = "Precision" ∨ c = "Finiteness"

theorem not_characteristic_of_algorithm : 
  ¬ characteristic_of_algorithm "Uniqueness" :=
by
  sorry

end not_characteristic_of_algorithm_l2270_227028


namespace even_quadruple_composition_l2270_227088

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end even_quadruple_composition_l2270_227088


namespace max_possible_b_l2270_227077

theorem max_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
by sorry

end max_possible_b_l2270_227077


namespace contingency_fund_amount_l2270_227085

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l2270_227085


namespace option_C_correct_l2270_227074

theorem option_C_correct {a : ℝ} : a^2 * a^3 = a^5 := by
  -- Proof to be filled
  sorry

end option_C_correct_l2270_227074


namespace period_of_repeating_decimal_l2270_227065

def is_100_digit_number_with_98_sevens (a : ℕ) : Prop :=
  ∃ (n : ℕ), n = 10^98 ∧ a = 1776 + 1777 * n

theorem period_of_repeating_decimal (a : ℕ) (h : is_100_digit_number_with_98_sevens a) : 
  (1:ℚ) / a == 1 / 99 := 
  sorry

end period_of_repeating_decimal_l2270_227065


namespace words_lost_equal_137_l2270_227004

-- Definitions based on conditions
def letters_in_oz : ℕ := 68
def forbidden_letter_index : ℕ := 7

def words_lost_due_to_forbidden_letter : ℕ :=
  let one_letter_words_lost : ℕ := 1
  let two_letter_words_lost : ℕ := 2 * (letters_in_oz - 1)
  one_letter_words_lost + two_letter_words_lost

-- Theorem stating that the words lost due to prohibition is 137
theorem words_lost_equal_137 :
  words_lost_due_to_forbidden_letter = 137 :=
sorry

end words_lost_equal_137_l2270_227004


namespace gunny_bag_capacity_l2270_227039

def pounds_per_ton : ℝ := 2500
def ounces_per_pound : ℝ := 16
def packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

theorem gunny_bag_capacity :
  (packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound) / pounds_per_ton) = 13 := 
by
  sorry

end gunny_bag_capacity_l2270_227039


namespace Ravi_Prakash_finish_together_l2270_227061

-- Definitions based on conditions
def Ravi_time := 24
def Prakash_time := 40

-- Main theorem statement
theorem Ravi_Prakash_finish_together :
  (1 / Ravi_time + 1 / Prakash_time) = 1 / 15 :=
by
  sorry

end Ravi_Prakash_finish_together_l2270_227061


namespace domain_f_a_5_abs_inequality_ab_l2270_227018

-- Definition for the domain of f(x) when a=5
def domain_of_f_a_5 (x : ℝ) : Prop := |x + 1| + |x + 2| - 5 ≥ 0

-- The theorem to find the domain A of the function f(x) when a=5.
theorem domain_f_a_5 (x : ℝ) : domain_of_f_a_5 x ↔ (x ≤ -4 ∨ x ≥ 1) :=
by
  sorry

-- Theorem to prove the inequality for a, b ∈ (-1, 1)
theorem abs_inequality_ab (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| :=
by
  sorry

end domain_f_a_5_abs_inequality_ab_l2270_227018


namespace jenna_reading_pages_l2270_227091

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end jenna_reading_pages_l2270_227091


namespace oatmeal_cookies_divisible_by_6_l2270_227037

theorem oatmeal_cookies_divisible_by_6 (O : ℕ) (h1 : 48 % 6 = 0) (h2 : O % 6 = 0) :
    ∃ x : ℕ, O = 6 * x :=
by sorry

end oatmeal_cookies_divisible_by_6_l2270_227037


namespace find_n_for_sine_equality_l2270_227052

theorem find_n_for_sine_equality : 
  ∃ (n: ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (670 * Real.pi / 180) ∧ n = -50 := by
  sorry

end find_n_for_sine_equality_l2270_227052


namespace value_of_x_l2270_227038

theorem value_of_x (x : ℝ) (h : x = 52 * (1 + 20 / 100)) : x = 62.4 :=
by sorry

end value_of_x_l2270_227038


namespace min_value_fraction_expression_l2270_227075

theorem min_value_fraction_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := 
by
  sorry

end min_value_fraction_expression_l2270_227075


namespace find_f91_plus_fm91_l2270_227094

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

theorem find_f91_plus_fm91 (a b c : ℝ) (h : f 91 a b c = 1) : f 91 a b c + f (-91) a b c = 2 := by
  sorry

end find_f91_plus_fm91_l2270_227094


namespace find_d_l2270_227079

theorem find_d (a b c d : ℤ) (h_poly : ∃ s1 s2 s3 s4 : ℤ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧ 
  ( ∀ x, (Polynomial.eval x (Polynomial.C d + Polynomial.X * Polynomial.C c + Polynomial.X^2 * Polynomial.C b + Polynomial.X^3 * Polynomial.C a + Polynomial.X^4)) =
    (x + s1) * (x + s2) * (x + s3) * (x + s4) ) ) 
  (h_sum : a + b + c + d = 2013) : d = 0 :=
by
  sorry

end find_d_l2270_227079


namespace ratio_of_cubes_l2270_227069

/-- A cubical block of metal weighs 7 pounds. Another cube of the same metal, with sides of a certain ratio longer, weighs 56 pounds. Prove that the ratio of the side length of the second cube to the first cube is 2:1. --/
theorem ratio_of_cubes (s r : ℝ) (weight1 weight2 : ℝ)
  (h1 : weight1 = 7) (h2 : weight2 = 56)
  (h_vol1 : weight1 = s^3)
  (h_vol2 : weight2 = (r * s)^3) :
  r = 2 := 
sorry

end ratio_of_cubes_l2270_227069


namespace same_type_l2270_227090

variable (X Y : Prop) 

-- Definition of witnesses A and B based on their statements
def witness_A (A : Prop) := A ↔ (X → Y)
def witness_B (B : Prop) := B ↔ (¬X ∨ Y)

-- Proposition stating that A and B must be of the same type
theorem same_type (A B : Prop) (HA : witness_A X Y A) (HB : witness_B X Y B) : 
  (A = B) := 
sorry

end same_type_l2270_227090


namespace find_n_divisible_by_highest_power_of_2_l2270_227053

def a_n (n : ℕ) : ℕ :=
  10^n * 999 + 488

theorem find_n_divisible_by_highest_power_of_2:
  ∀ n : ℕ, (n > 0) → (a_n n = 10^n * 999 + 488) → (∃ k : ℕ, 2^(k + 9) ∣ a_n 6) := sorry

end find_n_divisible_by_highest_power_of_2_l2270_227053


namespace four_point_questions_l2270_227032

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 :=
sorry

end four_point_questions_l2270_227032


namespace largest_value_l2270_227068

def expr_A : ℕ := 3 + 1 + 0 + 5
def expr_B : ℕ := 3 * 1 + 0 + 5
def expr_C : ℕ := 3 + 1 * 0 + 5
def expr_D : ℕ := 3 * 1 + 0 * 5
def expr_E : ℕ := 3 * 1 + 0 * 5 * 3

theorem largest_value :
  expr_A > expr_B ∧
  expr_A > expr_C ∧
  expr_A > expr_D ∧
  expr_A > expr_E :=
by
  sorry

end largest_value_l2270_227068


namespace two_cos_45_eq_sqrt_two_l2270_227087

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l2270_227087


namespace line_through_point_equal_intercepts_l2270_227092

theorem line_through_point_equal_intercepts (x y a b : ℝ) :
  ∀ (x y : ℝ), 
    (x - 1) = a → 
    (y - 2) = b →
    (a = -1 ∨ a = 2) → 
    ((x + y - 3 = 0) ∨ (2 * x - y = 0)) := by
  sorry

end line_through_point_equal_intercepts_l2270_227092


namespace sequence_negation_l2270_227003

theorem sequence_negation (x : ℕ → ℝ) (x1_pos : x 1 > 0) (x1_neq1 : x 1 ≠ 1)
  (rec_seq : ∀ n : ℕ, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∃ n : ℕ, x n ≤ x (n + 1) :=
sorry

end sequence_negation_l2270_227003


namespace relationship_between_y_l2270_227081

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l2270_227081


namespace lattice_points_in_region_l2270_227022

theorem lattice_points_in_region : ∃ n : ℕ, n = 1 ∧ ∀ p : ℤ × ℤ, 
  (p.snd = abs p.fst ∨ p.snd = -(p.fst ^ 3) + 6 * (p.fst)) → n = 1 :=
by
  sorry

end lattice_points_in_region_l2270_227022


namespace equation_1_solution_equation_2_solution_l2270_227097

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end equation_1_solution_equation_2_solution_l2270_227097


namespace maximum_monthly_profit_l2270_227089

-- Let's set up our conditions

def selling_price := 25
def monthly_profit := 120
def cost_price := 20
def selling_price_threshold := 32
def relationship (x n : ℝ) := -10 * x + n

-- Define the value of n
def value_of_n : ℝ := 370

-- Profit function
def profit_function (x n : ℝ) : ℝ := (x - cost_price) * (relationship x n)

-- Define the condition for maximum profit where the selling price should be higher than 32
def max_profit_condition (n : ℝ) (x : ℝ) := x > selling_price_threshold

-- Define what the maximum profit should be
def max_profit := 160

-- The main theorem to be proven
theorem maximum_monthly_profit :
  (relationship selling_price value_of_n = monthly_profit) →
  max_profit_condition value_of_n 32 →
  profit_function 32 value_of_n = max_profit :=
by sorry

end maximum_monthly_profit_l2270_227089


namespace bike_distance_from_rest_l2270_227029

variable (u : ℝ) (a : ℝ) (t : ℝ)

theorem bike_distance_from_rest (h1 : u = 0) (h2 : a = 0.5) (h3 : t = 8) : 
  (1 / 2 * a * t^2 = 16) :=
by
  sorry

end bike_distance_from_rest_l2270_227029


namespace book_costs_and_scenarios_l2270_227044

theorem book_costs_and_scenarios :
  (∃ (x y : ℕ), x + 3 * y = 180 ∧ 3 * x + y = 140 ∧ 
    (x = 30) ∧ (y = 50)) ∧ 
  (∀ (m : ℕ), (30 * m + 75 * m) ≤ 700 → (∃ (m_values : Finset ℕ), 
    m_values = {2, 4, 6} ∧ (m ∈ m_values))) :=
  sorry

end book_costs_and_scenarios_l2270_227044


namespace car_trip_eq_560_miles_l2270_227067

noncomputable def car_trip_length (v L : ℝ) :=
  -- Conditions from the problem
  -- 1. Car travels for 2 hours before the delay
  let pre_delay_time := 2
  -- 2. Delay time is 1 hour
  let delay_time := 1
  -- 3. Post-delay speed is 2/3 of the initial speed
  let post_delay_speed := (2 / 3) * v
  -- 4. Car arrives 4 hours late under initial scenario:
  let late_4_hours_time := 2 + 1 + (3 * (L - 2 * v)) / (2 * v)
  -- Expected travel time without any delays is 2 + (L / v)
  -- Difference indicates delay of 4 hours
  let without_delay_time := (L / v)
  let time_diff_late_4 := (late_4_hours_time - without_delay_time = 4)
  -- 5. Delay 120 miles farther, car arrives 3 hours late
  let delay_120_miles_farther := 120
  let late_3_hours_time := 2 + delay_120_miles_farther / v + 1 + (3 * (L - 2 * v - 120)) / (2 * v)
  let time_diff_late_3 := (late_3_hours_time - without_delay_time = 3)

  -- Combining conditions to solve for L
  -- Goal: Prove L = 560
  L = 560 -> time_diff_late_4 ∧ time_diff_late_3

theorem car_trip_eq_560_miles (v : ℝ) : ∃ (L : ℝ), car_trip_length v L := 
by 
  sorry

end car_trip_eq_560_miles_l2270_227067


namespace trucks_transportation_l2270_227023

theorem trucks_transportation (k : ℕ) (H : ℝ) : 
  (∃ (A B C : ℕ), 
     A + B + C = k ∧ 
     A ≤ k / 2 ∧ B ≤ k / 2 ∧ C ≤ k / 2 ∧ 
     (0 ≤ (k - 2*A)) ∧ (0 ≤ (k - 2*B)) ∧ (0 ≤ (k - 2*C))) 
  →  (k = 7 → (2 : ℕ) = 2) :=
sorry

end trucks_transportation_l2270_227023


namespace grunters_win_all_6_games_l2270_227062

noncomputable def prob_no_overtime_win : ℚ := 0.54
noncomputable def prob_overtime_win : ℚ := 0.05
noncomputable def prob_win_any_game : ℚ := prob_no_overtime_win + prob_overtime_win
noncomputable def prob_win_all_6_games : ℚ := prob_win_any_game ^ 6

theorem grunters_win_all_6_games :
  prob_win_all_6_games = (823543 / 10000000) :=
by sorry

end grunters_win_all_6_games_l2270_227062


namespace weight_of_rod_l2270_227082

theorem weight_of_rod (length1 length2 weight1 weight2 weight_per_meter : ℝ)
  (h1 : length1 = 6) (h2 : weight1 = 22.8) (h3 : length2 = 11.25)
  (h4 : weight_per_meter = weight1 / length1) :
  weight2 = weight_per_meter * length2 :=
by
  -- The proof would go here
  sorry

end weight_of_rod_l2270_227082


namespace tangent_line_is_correct_l2270_227043

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := -3 * x + 2

-- Statement of the problem (to prove)
theorem tangent_line_is_correct :
  curve point_of_tangency.1 = point_of_tangency.2 ∧
  ∃ m b, (∀ x, (tangent_line x) = m * x + b) ∧
         tangent_line point_of_tangency.1 = point_of_tangency.2 ∧
         (∀ x, deriv (curve) x = -3 ↔ deriv (tangent_line) point_of_tangency.1 = -3) :=
by
  sorry

end tangent_line_is_correct_l2270_227043


namespace divisible_by_17_l2270_227008

theorem divisible_by_17 (n : ℕ) : 17 ∣ (2 ^ (5 * n + 3) + 5 ^ n * 3 ^ (n + 2)) := 
by {
  sorry
}

end divisible_by_17_l2270_227008


namespace largest_sphere_surface_area_in_cone_l2270_227057

theorem largest_sphere_surface_area_in_cone :
  (∀ (r : ℝ), (∃ (r : ℝ), r > 0 ∧ (1^2 + (3^2 - r^2) = 3^2)) →
    4 * π * r^2 ≤ 2 * π) :=
by
  sorry

end largest_sphere_surface_area_in_cone_l2270_227057


namespace tangent_slope_at_pi_over_four_l2270_227017

theorem tangent_slope_at_pi_over_four :
  deriv (fun x => Real.tan x) (Real.pi / 4) = 2 :=
sorry

end tangent_slope_at_pi_over_four_l2270_227017


namespace sqrt_of_26244_div_by_100_l2270_227093

theorem sqrt_of_26244_div_by_100 (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 :=
sorry

end sqrt_of_26244_div_by_100_l2270_227093


namespace least_number_to_add_l2270_227056

theorem least_number_to_add (x : ℕ) : (1021 + x) % 25 = 0 ↔ x = 4 := 
by 
  sorry

end least_number_to_add_l2270_227056


namespace profit_share_difference_correct_l2270_227064

noncomputable def profit_share_difference (a_capital b_capital c_capital b_profit : ℕ) : ℕ :=
  let total_parts := 4 + 5 + 6
  let part_size := b_profit / 5
  let a_profit := 4 * part_size
  let c_profit := 6 * part_size
  c_profit - a_profit

theorem profit_share_difference_correct :
  profit_share_difference 8000 10000 12000 1600 = 640 :=
by
  sorry

end profit_share_difference_correct_l2270_227064


namespace problem1_problem2_l2270_227063

noncomputable def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)
noncomputable def condition1 (a : ℝ) : Prop := 
  a = -1 ∧ (∃ x, p x a ∨ q x)

noncomputable def condition2 (a : ℝ) : Prop :=
  ∀ x, ¬ p x a → ¬ q x

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : condition1 a) : -6 ≤ x ∧ x < -3 ∨ 1 < x ∧ x ≤ 12 := 
sorry

theorem problem2 (a : ℝ) (h₂ : condition2 a) : -4 ≤ a ∧ a ≤ -2 :=
sorry

end problem1_problem2_l2270_227063


namespace Sidney_JumpJacks_Tuesday_l2270_227000

variable (JumpJacksMonday JumpJacksTuesday JumpJacksWednesday JumpJacksThursday : ℕ)
variable (SidneyTotalJumpJacks BrookeTotalJumpJacks : ℕ)

-- Given conditions
axiom H1 : JumpJacksMonday = 20
axiom H2 : JumpJacksWednesday = 40
axiom H3 : JumpJacksThursday = 50
axiom H4 : BrookeTotalJumpJacks = 3 * SidneyTotalJumpJacks
axiom H5 : BrookeTotalJumpJacks = 438

-- Prove Sidney's JumpJacks on Tuesday
theorem Sidney_JumpJacks_Tuesday : JumpJacksTuesday = 36 :=
by
  sorry

end Sidney_JumpJacks_Tuesday_l2270_227000


namespace find_x1_l2270_227095

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 := 
  sorry

end find_x1_l2270_227095


namespace johns_avg_speed_l2270_227059

/-
John cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour.
We want to prove that his average speed for the entire trip is 10.91 miles per hour.
-/

theorem johns_avg_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 10.91 :=
by
  sorry

end johns_avg_speed_l2270_227059


namespace unique_value_of_n_l2270_227073

theorem unique_value_of_n
  (n t : ℕ) (h1 : t ≠ 0)
  (h2 : 15 * t + (n - 20) * t / 3 = (n * t) / 2) :
  n = 50 :=
by sorry

end unique_value_of_n_l2270_227073


namespace countEquilateralTriangles_l2270_227058

-- Define the problem conditions
def numSmallTriangles := 18  -- The number of small equilateral triangles
def includesMarkedTriangle: Prop := True  -- All counted triangles include the marked triangle "**"

-- Define the main question as a proposition
def totalEquilateralTriangles : Prop :=
  (numSmallTriangles = 18 ∧ includesMarkedTriangle) → (1 + 4 + 1 = 6)

-- The theorem stating the number of equilateral triangles containing the marked triangle
theorem countEquilateralTriangles : totalEquilateralTriangles :=
  by
    sorry

end countEquilateralTriangles_l2270_227058


namespace weight_of_new_person_l2270_227047

theorem weight_of_new_person (W : ℝ) (N : ℝ) (h1 : (W + (8 * 2.5)) = (W - 20 + N)) : N = 40 :=
by
  sorry

end weight_of_new_person_l2270_227047


namespace remainder_18_pow_63_mod_5_l2270_227049

theorem remainder_18_pow_63_mod_5 :
  (18:ℤ) ^ 63 % 5 = 2 :=
by
  -- Given conditions
  have h1 : (18:ℤ) % 5 = 3 := by norm_num
  have h2 : (3:ℤ) ^ 4 % 5 = 1 := by norm_num
  sorry

end remainder_18_pow_63_mod_5_l2270_227049


namespace inequality_solution_range_l2270_227071

variable (a : ℝ)

def f (x : ℝ) := 2 * x^2 - 8 * x - 4

theorem inequality_solution_range :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x - a > 0) ↔ a < -4 := 
by
  sorry

end inequality_solution_range_l2270_227071


namespace green_disks_more_than_blue_l2270_227012

theorem green_disks_more_than_blue (total_disks : ℕ) (b y g : ℕ) (h1 : total_disks = 108)
  (h2 : b / y = 3 / 7) (h3 : b / g = 3 / 8) : g - b = 30 :=
by
  sorry

end green_disks_more_than_blue_l2270_227012


namespace brad_weighs_more_l2270_227045

theorem brad_weighs_more :
  ∀ (Billy Brad Carl : ℕ), 
    (Billy = Brad + 9) → 
    (Carl = 145) → 
    (Billy = 159) → 
    (Brad - Carl = 5) :=
by
  intros Billy Brad Carl h1 h2 h3
  sorry

end brad_weighs_more_l2270_227045


namespace ice_cream_maker_completion_time_l2270_227051

def start_time := 9
def time_to_half := 3
def end_time := start_time + 2 * time_to_half

theorem ice_cream_maker_completion_time :
  end_time = 15 :=
by
  -- Definitions: 9:00 AM -> 9, 12:00 PM -> 12, 3:00 PM -> 15
  -- Calculation: end_time = 9 + 2 * 3 = 15
  sorry

end ice_cream_maker_completion_time_l2270_227051


namespace rational_square_plus_one_positive_l2270_227001

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 :=
sorry

end rational_square_plus_one_positive_l2270_227001


namespace number_of_10_digit_numbers_divisible_by_66667_l2270_227098

def ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 : ℕ := 33

theorem number_of_10_digit_numbers_divisible_by_66667 :
  ∃ n : ℕ, n = ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 :=
by
  sorry

end number_of_10_digit_numbers_divisible_by_66667_l2270_227098


namespace exists_multiple_representations_l2270_227020

def V (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V n ∧ ¬∃ (p q : ℕ), p ∈ V n ∧ q ∈ V n ∧ p * q = m

theorem exists_multiple_representations (n : ℕ) (h : 2 < n) :
  ∃ r ∈ V n, ∃ s t u v : ℕ, 
    indecomposable n s ∧ indecomposable n t ∧ indecomposable n u ∧ indecomposable n v ∧ 
    r = s * t ∧ r = u * v ∧ (s ≠ u ∨ t ≠ v) :=
sorry

end exists_multiple_representations_l2270_227020
