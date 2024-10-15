import Mathlib

namespace NUMINAMATH_GPT_f_odd_and_periodic_l2275_227513

open Function

-- Define the function f : ℝ → ℝ satisfying the given conditions
variables (f : ℝ → ℝ)

-- Conditions
axiom f_condition1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom f_condition2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

-- Theorem statement
theorem f_odd_and_periodic : Odd f ∧ Periodic f 40 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_f_odd_and_periodic_l2275_227513


namespace NUMINAMATH_GPT_seven_digit_number_l2275_227556

theorem seven_digit_number (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
(h1 : a_1 + a_2 = 9)
(h2 : a_2 + a_3 = 7)
(h3 : a_3 + a_4 = 9)
(h4 : a_4 + a_5 = 2)
(h5 : a_5 + a_6 = 8)
(h6 : a_6 + a_7 = 11)
(h_digits : ∀ (i : ℕ), i ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] → i < 10) :
a_1 = 9 ∧ a_2 = 0 ∧ a_3 = 7 ∧ a_4 = 2 ∧ a_5 = 0 ∧ a_6 = 8 ∧ a_7 = 3 :=
by sorry

end NUMINAMATH_GPT_seven_digit_number_l2275_227556


namespace NUMINAMATH_GPT_mn_equals_neg3_l2275_227511

noncomputable def function_with_extreme_value (m n : ℝ) : Prop :=
  let f := λ x : ℝ => m * x^3 + n * x
  let f' := λ x : ℝ => 3 * m * x^2 + n
  f' (1 / m) = 0

theorem mn_equals_neg3 (m n : ℝ) (h : function_with_extreme_value m n) : m * n = -3 :=
sorry

end NUMINAMATH_GPT_mn_equals_neg3_l2275_227511


namespace NUMINAMATH_GPT_profit_percentage_l2275_227524

theorem profit_percentage (C S : ℝ) (hC : C = 800) (hS : S = 1080) :
  ((S - C) / C) * 100 = 35 := 
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2275_227524


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2275_227577

variables {a b : ℝ}

theorem sufficient_but_not_necessary_condition (h₁ : b < -4) : |a| + |b| > 4 :=
by {
    sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2275_227577


namespace NUMINAMATH_GPT_possible_distances_between_andrey_and_gleb_l2275_227588

theorem possible_distances_between_andrey_and_gleb (A B V G : Point) 
  (d_AB : ℝ) (d_VG : ℝ) (d_BV : ℝ) (d_AG : ℝ)
  (h1 : d_AB = 600) 
  (h2 : d_VG = 600) 
  (h3 : d_AG = 3 * d_BV) : 
  d_AG = 900 ∨ d_AG = 1800 :=
by {
  sorry
}

end NUMINAMATH_GPT_possible_distances_between_andrey_and_gleb_l2275_227588


namespace NUMINAMATH_GPT_calculate_fraction_l2275_227510

theorem calculate_fraction : 
  ∃ f : ℝ, (14.500000000000002 ^ 2) * f = 126.15 ∧ f = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l2275_227510


namespace NUMINAMATH_GPT_complex_number_problem_l2275_227560

theorem complex_number_problem (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
by {
  -- provide proof here
  sorry
}

end NUMINAMATH_GPT_complex_number_problem_l2275_227560


namespace NUMINAMATH_GPT_circumradius_of_triangle_ABC_l2275_227529

noncomputable def circumradius (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem circumradius_of_triangle_ABC :
  (circumradius 12 10 7 = 6) :=
by
  sorry

end NUMINAMATH_GPT_circumradius_of_triangle_ABC_l2275_227529


namespace NUMINAMATH_GPT_quadratic_roots_l2275_227578

theorem quadratic_roots (x : ℝ) : x^2 + 4 * x + 3 = 0 → x = -3 ∨ x = -1 :=
by
  intro h
  have h1 : (x + 3) * (x + 1) = 0 := by sorry
  have h2 : (x = -3 ∨ x = -1) := by sorry
  exact h2

end NUMINAMATH_GPT_quadratic_roots_l2275_227578


namespace NUMINAMATH_GPT_largest_natural_divisible_power_l2275_227579

theorem largest_natural_divisible_power (p q : ℤ) (hp : p % 5 = 0) (hq : q % 5 = 0) (hdiscr : p^2 - 4*q > 0) :
  ∀ (α β : ℂ), (α^2 + p*α + q = 0 ∧ β^2 + p*β + q = 0) → (α^100 + β^100) % 5^50 = 0 :=
sorry

end NUMINAMATH_GPT_largest_natural_divisible_power_l2275_227579


namespace NUMINAMATH_GPT_function_relationship_l2275_227514

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- The main statement we want to prove
theorem function_relationship (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on f (Set.Ici 0)) :
  f 1 > f (-10) :=
by sorry

end NUMINAMATH_GPT_function_relationship_l2275_227514


namespace NUMINAMATH_GPT_average_value_of_T_l2275_227538

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end NUMINAMATH_GPT_average_value_of_T_l2275_227538


namespace NUMINAMATH_GPT_decreasing_interval_for_function_l2275_227587

theorem decreasing_interval_for_function :
  ∀ (f : ℝ → ℝ) (ϕ : ℝ),
  (∀ x, f x = -2 * Real.tan (2 * x + ϕ)) →
  |ϕ| < Real.pi →
  f (Real.pi / 16) = -2 →
  ∃ a b : ℝ, 
  a = 3 * Real.pi / 16 ∧ 
  b = 11 * Real.pi / 16 ∧ 
  ∀ x, a < x ∧ x < b → ∀ y, x < y ∧ y < b → f y < f x :=
by sorry

end NUMINAMATH_GPT_decreasing_interval_for_function_l2275_227587


namespace NUMINAMATH_GPT_compare_logs_and_exp_l2275_227502

theorem compare_logs_and_exp :
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1 / 2)
  c < a ∧ a < b := 
sorry

end NUMINAMATH_GPT_compare_logs_and_exp_l2275_227502


namespace NUMINAMATH_GPT_area_AOC_is_1_l2275_227549

noncomputable def point := (ℝ × ℝ) -- Define a point in 2D space

def vector_add (v1 v2 : point) : point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_zero : point := (0, 0)

def scalar_mul (r : ℝ) (v : point) : point :=
  (r * v.1, r * v.2)

def vector_eq (v1 v2 : point) : Prop := 
  v1.1 = v2.1 ∧ v1.2 = v2.2

variables (A B C O : point)
variable (area_ABC : ℝ)

-- Conditions:
-- Point O is a point inside triangle ABC with an area of 4
-- \(\overrightarrow {OA} + \overrightarrow {OB} + 2\overrightarrow {OC} = \overrightarrow {0}\)
axiom condition_area : area_ABC = 4
axiom condition_vector : vector_eq (vector_add (vector_add O A) (vector_add O B)) (scalar_mul (-2) O)

-- Theorem to prove: the area of triangle AOC is 1
theorem area_AOC_is_1 : (area_ABC / 4) = 1 := 
sorry

end NUMINAMATH_GPT_area_AOC_is_1_l2275_227549


namespace NUMINAMATH_GPT_find_triples_l2275_227559

theorem find_triples (a b c : ℕ) :
  (∃ n : ℕ, 2^a + 2^b + 2^c + 3 = n^2) ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l2275_227559


namespace NUMINAMATH_GPT_arithmetic_expression_count_l2275_227527

theorem arithmetic_expression_count (f : ℕ → ℤ) 
  (h1 : f 1 = 9)
  (h2 : f 2 = 99)
  (h_recur : ∀ n ≥ 2, f n = 9 * (f (n - 1)) + 36 * (f (n - 2))) :
  ∀ n, f n = (7 / 10 : ℚ) * 12^n - (1 / 5 : ℚ) * (-3)^n := sorry

end NUMINAMATH_GPT_arithmetic_expression_count_l2275_227527


namespace NUMINAMATH_GPT_least_multiple_of_36_with_digit_product_multiple_of_9_l2275_227519

def is_multiple_of_36 (n : ℕ) : Prop :=
  n % 36 = 0

def product_of_digits_multiple_of_9 (n : ℕ) : Prop :=
  ∃ d : List ℕ, (n = List.foldl (λ x y => x * 10 + y) 0 d) ∧ (List.foldl (λ x y => x * y) 1 d) % 9 = 0

theorem least_multiple_of_36_with_digit_product_multiple_of_9 : ∃ n : ℕ, is_multiple_of_36 n ∧ product_of_digits_multiple_of_9 n ∧ n = 36 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_36_with_digit_product_multiple_of_9_l2275_227519


namespace NUMINAMATH_GPT_percentage_dried_fruit_of_combined_mix_l2275_227586

theorem percentage_dried_fruit_of_combined_mix :
  ∀ (weight_sue weight_jane : ℝ),
  (weight_sue * 0.3 + weight_jane * 0.6) / (weight_sue + weight_jane) = 0.45 →
  100 * (weight_sue * 0.7) / (weight_sue + weight_jane) = 35 :=
by
  intros weight_sue weight_jane H
  sorry

end NUMINAMATH_GPT_percentage_dried_fruit_of_combined_mix_l2275_227586


namespace NUMINAMATH_GPT_Anne_height_l2275_227553

-- Define the conditions
variables (S : ℝ)   -- Height of Anne's sister
variables (A : ℝ)   -- Height of Anne
variables (B : ℝ)   -- Height of Bella

-- Define the relations according to the problem's conditions
def condition1 (S : ℝ) := A = 2 * S
def condition2 (S : ℝ) := B = 3 * A
def condition3 (S : ℝ) := B - S = 200

-- Theorem statement to prove Anne's height
theorem Anne_height (S : ℝ) (A : ℝ) (B : ℝ)
(h1 : A = 2 * S) (h2 : B = 3 * A) (h3 : B - S = 200) : A = 80 :=
by sorry

end NUMINAMATH_GPT_Anne_height_l2275_227553


namespace NUMINAMATH_GPT_area_of_field_l2275_227564

theorem area_of_field (L W A : ℝ) (hL : L = 20) (hP : L + 2 * W = 25) : A = 50 :=
by
  sorry

end NUMINAMATH_GPT_area_of_field_l2275_227564


namespace NUMINAMATH_GPT_cubeRootThree_expression_value_l2275_227546

-- Define the approximate value of cube root of 3
def cubeRootThree : ℝ := 1.442

-- Lean theorem statement
theorem cubeRootThree_expression_value :
  cubeRootThree - 3 * cubeRootThree - 98 * cubeRootThree = -144.2 := by
  sorry

end NUMINAMATH_GPT_cubeRootThree_expression_value_l2275_227546


namespace NUMINAMATH_GPT_calculate_X_value_l2275_227518

theorem calculate_X_value : 
  let M := (2025 : ℝ) / 3
  let N := M / 4
  let X := M - N
  X = 506.25 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_X_value_l2275_227518


namespace NUMINAMATH_GPT_evaluate_expression_l2275_227503

theorem evaluate_expression (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (5 - x) + (5 - x) ^ 2 = 49 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l2275_227503


namespace NUMINAMATH_GPT_incorrect_statement_A_l2275_227544

def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def has_real_roots (a b c : ℝ) : Prop :=
  let delta := b^2 - 4 * a * c
  delta ≥ 0

theorem incorrect_statement_A (a b c : ℝ) (h₀ : a ≠ 0) :
  (∃ x : ℝ, parabola a b c x = 0) ∧ (parabola a b c (-b/(2*a)) < 0) → ¬ has_real_roots a b c := 
by
  sorry -- proof required here if necessary

end NUMINAMATH_GPT_incorrect_statement_A_l2275_227544


namespace NUMINAMATH_GPT_percentage_increase_l2275_227584

-- Defining the problem constants
def price (P : ℝ) : ℝ := P
def assets_A (A : ℝ) : ℝ := A
def assets_B (B : ℝ) : ℝ := B
def percentage (X : ℝ) : ℝ := X

-- Conditions
axiom price_company_B_double_assets : ∀ (P B: ℝ), price P = 2 * assets_B B
axiom price_seventy_five_percent_combined_assets : ∀ (P A B: ℝ), price P = 0.75 * (assets_A A + assets_B B)
axiom price_percentage_more_than_A : ∀ (P A X: ℝ), price P = assets_A A * (1 + percentage X / 100)

-- Theorem to prove
theorem percentage_increase : ∀ (P A B X : ℝ)
  (h1 : price P = 2 * assets_B B)
  (h2 : price P = 0.75 * (assets_A A + assets_B B))
  (h3 : price P = assets_A A * (1 + percentage X / 100)),
  percentage X = 20 :=
by
  intros P A B X h1 h2 h3
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_percentage_increase_l2275_227584


namespace NUMINAMATH_GPT_geometric_product_Pi8_l2275_227533

def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ}
variable (h_geom : geometric_sequence a)
variable (h_prod : a 4 * a 5 = 2)

theorem geometric_product_Pi8 :
  (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_product_Pi8_l2275_227533


namespace NUMINAMATH_GPT_average_of_four_l2275_227537

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end NUMINAMATH_GPT_average_of_four_l2275_227537


namespace NUMINAMATH_GPT_stan_weighs_5_more_than_steve_l2275_227539

theorem stan_weighs_5_more_than_steve
(S V J : ℕ) 
(h1 : J = 110)
(h2 : V = J - 8)
(h3 : S + V + J = 319) : 
(S - V = 5) :=
by
  sorry

end NUMINAMATH_GPT_stan_weighs_5_more_than_steve_l2275_227539


namespace NUMINAMATH_GPT_min_value_of_expression_l2275_227561

theorem min_value_of_expression (α β : ℝ) (h : α + β = π / 2) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 65 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2275_227561


namespace NUMINAMATH_GPT_photo_album_requirement_l2275_227589

-- Definition of the conditions
def pages_per_album : ℕ := 32
def photos_per_page : ℕ := 5
def total_photos : ℕ := 900

-- Calculation of photos per album
def photos_per_album := pages_per_album * photos_per_page

-- Calculation of required albums
noncomputable def albums_needed := (total_photos + photos_per_album - 1) / photos_per_album

-- Theorem to prove the required number of albums is 6
theorem photo_album_requirement : albums_needed = 6 :=
  by sorry

end NUMINAMATH_GPT_photo_album_requirement_l2275_227589


namespace NUMINAMATH_GPT_work_days_l2275_227501

theorem work_days (x : ℕ) (hx : 0 < x) :
  (1 / (x : ℚ) + 1 / 20) = 1 / 15 → x = 60 := by
sorry

end NUMINAMATH_GPT_work_days_l2275_227501


namespace NUMINAMATH_GPT_totalCats_l2275_227566

def whiteCats : Nat := 2
def blackCats : Nat := 10
def grayCats : Nat := 3

theorem totalCats : whiteCats + blackCats + grayCats = 15 := by
  sorry

end NUMINAMATH_GPT_totalCats_l2275_227566


namespace NUMINAMATH_GPT_complex_expression_identity_l2275_227591

open Complex

theorem complex_expression_identity
  (x y : ℂ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy : x^2 + x * y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_identity_l2275_227591


namespace NUMINAMATH_GPT_cylinder_base_area_l2275_227571

-- Definitions: Adding variables and hypotheses based on the problem statement.
variable (A_c A_r : ℝ) -- Base areas of the cylinder and the rectangular prism
variable (h1 : 8 * A_c = 6 * A_r) -- Condition from the rise in water levels
variable (h2 : A_c + A_r = 98) -- Sum of the base areas
variable (h3 : A_c / A_r = 3 / 4) -- Ratio of the base areas

-- Statement: The goal is to prove that the base area of the cylinder is 42.
theorem cylinder_base_area : A_c = 42 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_base_area_l2275_227571


namespace NUMINAMATH_GPT_cheaper_rock_cost_per_ton_l2275_227517

theorem cheaper_rock_cost_per_ton (x : ℝ) 
    (h1 : 24 * 1 = 24) 
    (h2 : 800 = 16 * x + 8 * 40) : 
    x = 30 :=
sorry

end NUMINAMATH_GPT_cheaper_rock_cost_per_ton_l2275_227517


namespace NUMINAMATH_GPT_equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l2275_227500

-- Define a transformation predicate for words
inductive transform : List Char -> List Char -> Prop
| xy_to_yyx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 'y'] ++ l2) (l1 ++ ['y', 'y', 'x'] ++ l2)
| yyx_to_xy : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 'y', 'x'] ++ l2) (l1 ++ ['x', 'y'] ++ l2)
| xt_to_ttx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 't'] ++ l2) (l1 ++ ['t', 't', 'x'] ++ l2)
| ttx_to_xt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 't', 'x'] ++ l2) (l1 ++ ['x', 't'] ++ l2)
| yt_to_ty : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 't'] ++ l2) (l1 ++ ['t', 'y'] ++ l2)
| ty_to_yt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 'y'] ++ l2) (l1 ++ ['y', 't'] ++ l2)

-- Reflexive and transitive closure of transform
inductive transforms : List Char -> List Char -> Prop
| base : ∀ l, transforms l l
| step : ∀ l m n, transform l m → transforms m n → transforms l n

-- Definitions for the words and their information
def word1 := ['x', 'x', 'y', 'y']
def word2 := ['x', 'y', 'y', 'y', 'y', 'x']
def word3 := ['x', 'y', 't', 'x']
def word4 := ['t', 'x', 'y', 't']
def word5 := ['x', 'y']
def word6 := ['x', 't']

-- Proof statements
theorem equivalent_xy_xxyy : transforms word1 word2 :=
by sorry

theorem not_equivalent_xyty_txy : ¬ transforms word3 word4 :=
by sorry

theorem not_equivalent_xy_xt : ¬ transforms word5 word6 :=
by sorry

end NUMINAMATH_GPT_equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l2275_227500


namespace NUMINAMATH_GPT_calculation_correctness_l2275_227592

theorem calculation_correctness : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end NUMINAMATH_GPT_calculation_correctness_l2275_227592


namespace NUMINAMATH_GPT_max_ratio_is_99_over_41_l2275_227570

noncomputable def max_ratio (x y : ℕ) (h1 : x > y) (h2 : x + y = 140) : ℚ :=
  if h : y ≠ 0 then (x / y : ℚ) else 0

theorem max_ratio_is_99_over_41 : ∃ (x y : ℕ), x > y ∧ x + y = 140 ∧ max_ratio x y (by sorry) (by sorry) = (99 / 41 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_max_ratio_is_99_over_41_l2275_227570


namespace NUMINAMATH_GPT_three_term_inequality_l2275_227576

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_three_term_inequality_l2275_227576


namespace NUMINAMATH_GPT_find_a_l2275_227583

variable (A B : Set ℤ) (a : ℤ)
variable (elem1 : 0 ∈ A) (elem2 : 1 ∈ A)
variable (elem3 : -1 ∈ B) (elem4 : 0 ∈ B) (elem5 : a + 3 ∈ B)

theorem find_a (h : A ⊆ B) : a = -2 := sorry

end NUMINAMATH_GPT_find_a_l2275_227583


namespace NUMINAMATH_GPT_train_length_l2275_227593

theorem train_length (time : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (speed_ms : ℚ) (length : ℚ) :
  time = 50 ∧ speed_kmh = 36 ∧ conversion_factor = 5 / 18 ∧ speed_ms = speed_kmh * conversion_factor ∧ length = speed_ms * time →
  length = 500 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2275_227593


namespace NUMINAMATH_GPT_tim_movie_marathon_duration_is_9_l2275_227580

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end NUMINAMATH_GPT_tim_movie_marathon_duration_is_9_l2275_227580


namespace NUMINAMATH_GPT_expand_expression_l2275_227541

variable (y : ℝ)

theorem expand_expression : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l2275_227541


namespace NUMINAMATH_GPT_find_y_when_x_is_6_l2275_227504

variable (x y : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (k : ℝ)

axiom inverse_proportional : 3 * x^2 * y = k
axiom initial_condition : 3 * 3^2 * 30 = k

theorem find_y_when_x_is_6 (h : x = 6) : y = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_6_l2275_227504


namespace NUMINAMATH_GPT_find_n_infinitely_many_squares_find_n_no_squares_l2275_227565

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P (n k l m : ℕ) : ℕ := n^k + n^l + n^m

theorem find_n_infinitely_many_squares :
  ∃ k, ∃ l, ∃ m, is_square (P 7 k l m) :=
by
  sorry

theorem find_n_no_squares :
  ∀ (k l m : ℕ) n, n ∈ [5, 6] → ¬is_square (P n k l m) :=
by
  sorry

end NUMINAMATH_GPT_find_n_infinitely_many_squares_find_n_no_squares_l2275_227565


namespace NUMINAMATH_GPT_probability_not_face_card_l2275_227545

-- Definitions based on the conditions
def total_cards : ℕ := 52
def face_cards  : ℕ := 12
def non_face_cards : ℕ := total_cards - face_cards

-- Statement of the theorem
theorem probability_not_face_card : (non_face_cards : ℚ) / (total_cards : ℚ) = 10 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_not_face_card_l2275_227545


namespace NUMINAMATH_GPT_counting_indistinguishable_boxes_l2275_227573

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end NUMINAMATH_GPT_counting_indistinguishable_boxes_l2275_227573


namespace NUMINAMATH_GPT_bucket_weight_full_l2275_227590

variable (c d : ℝ)

theorem bucket_weight_full (h1 : ∃ x y, x + (1 / 4) * y = c)
                           (h2 : ∃ x y, x + (3 / 4) * y = d) :
  ∃ x y, x + y = (3 * d - c) / 2 :=
by
  sorry

end NUMINAMATH_GPT_bucket_weight_full_l2275_227590


namespace NUMINAMATH_GPT_negation_equiv_l2275_227550

theorem negation_equiv (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l2275_227550


namespace NUMINAMATH_GPT_least_number_with_remainder_4_l2275_227557

theorem least_number_with_remainder_4 : ∃ n : ℕ, n = 184 ∧ 
  (∀ d ∈ [5, 9, 12, 18], (n - 4) % d = 0) ∧
  (∀ m : ℕ, (∀ d ∈ [5, 9, 12, 18], (m - 4) % d = 0) → m ≥ n) :=
by
  sorry

end NUMINAMATH_GPT_least_number_with_remainder_4_l2275_227557


namespace NUMINAMATH_GPT_find_abc_solutions_l2275_227531

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_solutions_l2275_227531


namespace NUMINAMATH_GPT_linear_func_is_direct_proportion_l2275_227543

theorem linear_func_is_direct_proportion (m : ℝ) : (∀ x : ℝ, (y : ℝ) → y = m * x + m - 2 → (m - 2 = 0) → y = 0) → m = 2 :=
by
  intros h
  have : m - 2 = 0 := sorry
  exact sorry

end NUMINAMATH_GPT_linear_func_is_direct_proportion_l2275_227543


namespace NUMINAMATH_GPT_total_spending_in_CAD_proof_l2275_227568

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

end NUMINAMATH_GPT_total_spending_in_CAD_proof_l2275_227568


namespace NUMINAMATH_GPT_find_first_number_l2275_227525

theorem find_first_number (x : ℕ) : 
    (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end NUMINAMATH_GPT_find_first_number_l2275_227525


namespace NUMINAMATH_GPT_math_problem_l2275_227562

theorem math_problem :
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l2275_227562


namespace NUMINAMATH_GPT_johns_original_earnings_l2275_227595

-- Define the conditions
def raises (original : ℝ) (percentage : ℝ) := original + original * percentage

-- The theorem stating the equivalent problem proof
theorem johns_original_earnings :
  ∃ (x : ℝ), raises x 0.375 = 55 ↔ x = 40 :=
sorry

end NUMINAMATH_GPT_johns_original_earnings_l2275_227595


namespace NUMINAMATH_GPT_area_of_triangle_l2275_227506

theorem area_of_triangle (S_x S_y S_z S : ℝ)
  (hx : S_x = Real.sqrt 7) (hy : S_y = Real.sqrt 6)
  (hz : ∃ k : ℕ, S_z = k) (hs : ∃ n : ℕ, S = n)
  : S = 7 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2275_227506


namespace NUMINAMATH_GPT_length_of_GH_l2275_227512

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end NUMINAMATH_GPT_length_of_GH_l2275_227512


namespace NUMINAMATH_GPT_gigi_mushrooms_l2275_227535

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end NUMINAMATH_GPT_gigi_mushrooms_l2275_227535


namespace NUMINAMATH_GPT_probability_solved_l2275_227516

theorem probability_solved (pA pB pA_and_B : ℚ) :
  pA = 2 / 3 → pB = 3 / 4 → pA_and_B = (2 / 3) * (3 / 4) →
  pA + pB - pA_and_B = 11 / 12 :=
by
  intros hA hB hA_and_B
  rw [hA, hB, hA_and_B]
  sorry

end NUMINAMATH_GPT_probability_solved_l2275_227516


namespace NUMINAMATH_GPT_max_b_value_l2275_227597

theorem max_b_value (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_b_value_l2275_227597


namespace NUMINAMATH_GPT_total_guppies_l2275_227563

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end NUMINAMATH_GPT_total_guppies_l2275_227563


namespace NUMINAMATH_GPT_Hadley_walked_to_grocery_store_in_2_miles_l2275_227523

-- Define the variables and conditions
def distance_to_grocery_store (x : ℕ) : Prop :=
  x + (x - 1) + 3 = 6

-- Stating the main proposition to prove
theorem Hadley_walked_to_grocery_store_in_2_miles : ∃ x : ℕ, distance_to_grocery_store x ∧ x = 2 := 
by sorry

end NUMINAMATH_GPT_Hadley_walked_to_grocery_store_in_2_miles_l2275_227523


namespace NUMINAMATH_GPT_number_of_keepers_l2275_227582

theorem number_of_keepers
  (h₁ : 50 * 2 = 100)
  (h₂ : 45 * 4 = 180)
  (h₃ : 8 * 4 = 32)
  (h₄ : 12 * 8 = 96)
  (h₅ : 6 * 8 = 48)
  (h₆ : 100 + 180 + 32 + 96 + 48 = 456)
  (h₇ : 50 + 45 + 8 + 12 + 6 = 121)
  (h₈ : ∀ K : ℕ, (2 * (K - 5) + 6 + 2 = 2 * K - 2))
  (h₉ : ∀ K : ℕ, 121 + K + 372 = 456 + (2 * K - 2)) :
  ∃ K : ℕ, K = 39 :=
by
  sorry

end NUMINAMATH_GPT_number_of_keepers_l2275_227582


namespace NUMINAMATH_GPT_problem_statement_l2275_227530

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2275_227530


namespace NUMINAMATH_GPT_ratio_15_to_1_l2275_227558

theorem ratio_15_to_1 (x : ℕ) (h : 15 / 1 = x / 10) : x = 150 := 
by sorry

end NUMINAMATH_GPT_ratio_15_to_1_l2275_227558


namespace NUMINAMATH_GPT_largest_inscribed_square_l2275_227505

-- Define the problem data
noncomputable def s : ℝ := 15
noncomputable def h : ℝ := s * (Real.sqrt 3) / 2
noncomputable def y : ℝ := s - h

-- Statement to prove
theorem largest_inscribed_square :
  y = (30 - 15 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_largest_inscribed_square_l2275_227505


namespace NUMINAMATH_GPT_find_C_value_l2275_227572

theorem find_C_value (A B C : ℕ) 
  (cond1 : A + B + C = 10) 
  (cond2 : B + A = 9)
  (cond3 : A + 1 = 3) :
  C = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_C_value_l2275_227572


namespace NUMINAMATH_GPT_sum_of_digits_base2_315_l2275_227575

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end NUMINAMATH_GPT_sum_of_digits_base2_315_l2275_227575


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2275_227596

-- a) Proof problem for \(x^2 + 5x + 6 < 0\)
theorem problem_a (x : ℝ) : x^2 + 5*x + 6 < 0 → -3 < x ∧ x < -2 := by
  sorry

-- b) Proof problem for \(-x^2 + 9x - 20 < 0\)
theorem problem_b (x : ℝ) : -x^2 + 9*x - 20 < 0 → x < 4 ∨ x > 5 := by
  sorry

-- c) Proof problem for \(x^2 + x - 56 < 0\)
theorem problem_c (x : ℝ) : x^2 + x - 56 < 0 → -8 < x ∧ x < 7 := by
  sorry

-- d) Proof problem for \(9x^2 + 4 < 12x\) (No solutions)
theorem problem_d (x : ℝ) : ¬ 9*x^2 + 4 < 12*x := by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2275_227596


namespace NUMINAMATH_GPT_daniel_paid_more_l2275_227515

noncomputable def num_slices : ℕ := 10
noncomputable def plain_cost : ℕ := 10
noncomputable def truffle_extra_cost : ℕ := 5
noncomputable def total_cost : ℕ := plain_cost + truffle_extra_cost
noncomputable def cost_per_slice : ℝ := total_cost / num_slices

noncomputable def truffle_slices_cost : ℝ := 5 * cost_per_slice
noncomputable def plain_slices_cost : ℝ := 5 * cost_per_slice

noncomputable def daniel_cost : ℝ := 5 * cost_per_slice + 2 * cost_per_slice
noncomputable def carl_cost : ℝ := 3 * cost_per_slice

noncomputable def payment_difference : ℝ := daniel_cost - carl_cost

theorem daniel_paid_more : payment_difference = 6 :=
by 
  sorry

end NUMINAMATH_GPT_daniel_paid_more_l2275_227515


namespace NUMINAMATH_GPT_quadratic_expression_value_l2275_227581

theorem quadratic_expression_value (x₁ x₂ : ℝ) (h₁ : x₁^2 - 3 * x₁ + 1 = 0) (h₂ : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^2 + 3 * x₂ + x₁ * x₂ - 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_expression_value_l2275_227581


namespace NUMINAMATH_GPT_family_of_four_children_includes_one_boy_one_girl_l2275_227574

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_family_of_four_children_includes_one_boy_one_girl_l2275_227574


namespace NUMINAMATH_GPT_sum_of_cubes_l2275_227528

def cubic_eq (x : ℝ) : Prop := x^3 - 2 * x^2 + 3 * x - 4 = 0

variables (a b c : ℝ)

axiom a_root : cubic_eq a
axiom b_root : cubic_eq b
axiom c_root : cubic_eq c

axiom sum_roots : a + b + c = 2
axiom sum_products_roots : a * b + a * c + b * c = 3
axiom product_roots : a * b * c = 4

theorem sum_of_cubes : a^3 + b^3 + c^3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l2275_227528


namespace NUMINAMATH_GPT_circle_diameter_percentage_l2275_227508

theorem circle_diameter_percentage (d_R d_S : ℝ) 
    (h : π * (d_R / 2)^2 = 0.04 * π * (d_S / 2)^2) : 
    d_R = 0.4 * d_S :=
by
    sorry

end NUMINAMATH_GPT_circle_diameter_percentage_l2275_227508


namespace NUMINAMATH_GPT_samantha_erased_length_l2275_227526

/--
Samantha drew a line that was originally 1 meter (100 cm) long, and then it was erased until the length was 90 cm.
This theorem proves that the amount erased was 10 cm.
-/
theorem samantha_erased_length : 
  let original_length := 100 -- original length in cm
  let final_length := 90 -- final length in cm
  original_length - final_length = 10 := 
by
  sorry

end NUMINAMATH_GPT_samantha_erased_length_l2275_227526


namespace NUMINAMATH_GPT_percentage_of_annual_decrease_is_10_l2275_227532

-- Define the present population and future population
def P_present : ℕ := 500
def P_future : ℕ := 450 

-- Calculate the percentage decrease
def percentage_decrease (P_present P_future : ℕ) : ℕ :=
  ((P_present - P_future) * 100) / P_present

-- Lean statement to prove the percentage decrease is 10%
theorem percentage_of_annual_decrease_is_10 :
  percentage_decrease P_present P_future = 10 :=
by
  unfold percentage_decrease
  sorry

end NUMINAMATH_GPT_percentage_of_annual_decrease_is_10_l2275_227532


namespace NUMINAMATH_GPT_part1_part2_l2275_227509

variables (a b c : ℝ)

theorem part1 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ab + bc + ac ≤ 1 / 3 := sorry

theorem part2 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  1 / a + 1 / b + 1 / c ≥ 9 := sorry

end NUMINAMATH_GPT_part1_part2_l2275_227509


namespace NUMINAMATH_GPT_min_students_with_same_score_l2275_227567

noncomputable def highest_score : ℕ := 83
noncomputable def lowest_score : ℕ := 30
noncomputable def total_students : ℕ := 8000
noncomputable def range_scores : ℕ := (highest_score - lowest_score + 1)

theorem min_students_with_same_score :
  ∃ k : ℕ, k = Nat.ceil (total_students / range_scores) ∧ k = 149 :=
by
  sorry

end NUMINAMATH_GPT_min_students_with_same_score_l2275_227567


namespace NUMINAMATH_GPT_minimum_ticket_cost_correct_l2275_227540

noncomputable def minimum_ticket_cost : Nat :=
let adults := 8
let children := 4
let adult_ticket_price := 100
let child_ticket_price := 50
let group_ticket_price := 70
let group_size := 10
-- Calculate the cost of group tickets for 10 people and regular tickets for 2 children
let total_cost := (group_size * group_ticket_price) + (2 * child_ticket_price)
total_cost

theorem minimum_ticket_cost_correct :
  minimum_ticket_cost = 800 := by
  sorry

end NUMINAMATH_GPT_minimum_ticket_cost_correct_l2275_227540


namespace NUMINAMATH_GPT_area_D_meets_sign_l2275_227534

-- Definition of conditions as given in the question
def condition_A (mean median : ℝ) : Prop := mean = 3 ∧ median = 4
def condition_B (mean : ℝ) (variance_pos : Prop) : Prop := mean = 1 ∧ variance_pos
def condition_C (median mode : ℝ) : Prop := median = 2 ∧ mode = 3
def condition_D (mean variance : ℝ) : Prop := mean = 2 ∧ variance = 3

-- Theorem stating that Area D satisfies the condition to meet the required sign
theorem area_D_meets_sign (mean variance : ℝ) (h : condition_D mean variance) : 
  (∀ day_increase, day_increase ≤ 7) :=
sorry

end NUMINAMATH_GPT_area_D_meets_sign_l2275_227534


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l2275_227522

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 4 * x^2 - 7 * x + 10 = 0} = {1, -2, 5} :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l2275_227522


namespace NUMINAMATH_GPT_number_of_common_terms_between_arithmetic_sequences_l2275_227552

-- Definitions for the sequences
def seq1 (n : Nat) := 2 + 3 * n
def seq2 (n : Nat) := 4 + 5 * n

theorem number_of_common_terms_between_arithmetic_sequences
  (A : Finset Nat := Finset.range 673)  -- There are 673 terms in seq1 from 2 to 2015
  (B : Finset Nat := Finset.range 403)  -- There are 403 terms in seq2 from 4 to 2014
  (common_terms : Finset Nat := (A.image seq1) ∩ (B.image seq2)) :
  common_terms.card = 134 := by
  sorry

end NUMINAMATH_GPT_number_of_common_terms_between_arithmetic_sequences_l2275_227552


namespace NUMINAMATH_GPT_custom_op_equality_l2275_227599

def custom_op (x y : Int) : Int :=
  x * y - 2 * x

theorem custom_op_equality : custom_op 5 3 - custom_op 3 5 = -4 := by
  sorry

end NUMINAMATH_GPT_custom_op_equality_l2275_227599


namespace NUMINAMATH_GPT_triangle_area_l2275_227585

noncomputable def area_of_triangle (l1 l2 l3 : ℝ × ℝ → Prop) (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem triangle_area :
  let A := (1, 6)
  let B := (-1, 6)
  let C := (0, 4)
  ∀ x y : ℝ, 
    (y = 6 → l1 (x, y)) ∧ 
    (y = 2 * x + 4 → l2 (x, y)) ∧ 
    (y = -2 * x + 4 → l3 (x, y)) →
  area_of_triangle l1 l2 l3 A B C = 1 :=
by 
  intros
  unfold area_of_triangle
  sorry

end NUMINAMATH_GPT_triangle_area_l2275_227585


namespace NUMINAMATH_GPT_heartsuit_properties_l2275_227520

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem heartsuit_properties (x y : ℝ) :
  (heartsuit x y ≥ 0) ∧ (heartsuit x y > 0 ↔ x ≠ y) := by
  -- Proof will go here 
  sorry

end NUMINAMATH_GPT_heartsuit_properties_l2275_227520


namespace NUMINAMATH_GPT_complete_square_transform_l2275_227547

theorem complete_square_transform (x : ℝ) :
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := 
sorry

end NUMINAMATH_GPT_complete_square_transform_l2275_227547


namespace NUMINAMATH_GPT_parallel_vectors_sin_cos_l2275_227548

theorem parallel_vectors_sin_cos (θ : ℝ) (a := (6, 3)) (b := (Real.sin θ, Real.cos θ))
  (h : (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2)) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_sin_cos_l2275_227548


namespace NUMINAMATH_GPT_at_least_2_boys_and_1_girl_l2275_227542

noncomputable def probability_at_least_2_boys_and_1_girl (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_with_0_boys := Nat.choose girls committee_size
  let ways_with_1_boy := Nat.choose boys 1 * Nat.choose girls (committee_size - 1)
  let ways_with_fewer_than_2_boys := ways_with_0_boys + ways_with_1_boy
  1 - (ways_with_fewer_than_2_boys / total_ways)

theorem at_least_2_boys_and_1_girl :
  probability_at_least_2_boys_and_1_girl 32 14 18 6 = 767676 / 906192 :=
by
  sorry

end NUMINAMATH_GPT_at_least_2_boys_and_1_girl_l2275_227542


namespace NUMINAMATH_GPT_negation_of_p_l2275_227569

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

theorem negation_of_p : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end NUMINAMATH_GPT_negation_of_p_l2275_227569


namespace NUMINAMATH_GPT_miles_driven_l2275_227507

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end NUMINAMATH_GPT_miles_driven_l2275_227507


namespace NUMINAMATH_GPT_min_value_frac_l2275_227551

theorem min_value_frac (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) (h3 : a * c = 4) : 
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, y = (1 / c + 9 / a) → y ≥ x :=
by sorry

end NUMINAMATH_GPT_min_value_frac_l2275_227551


namespace NUMINAMATH_GPT_tangent_line_eq_area_independent_of_a_l2275_227536

open Real

section TangentLineAndArea

def curve (x : ℝ) := x^2 - 1

def tangentCurvey (x : ℝ) := x^2

noncomputable def tangentLine (a : ℝ) (ha : a > 0) : (ℝ → ℝ) :=
  if a > 1 then λ x => (2*(a + 1)) * x - (a+1)^2
  else λ x => (2*(a - 1)) * x - (a-1)^2

theorem tangent_line_eq (a : ℝ) (ha : a > 0) :
  ∃ (line : ℝ → ℝ), (line = tangentLine a ha) :=
sorry

theorem area_independent_of_a (a : ℝ) (ha : a > 0) :
  (∫ x in (a - 1)..a, (tangentCurvey x - tangentLine a ha x)) +
  (∫ x in a..(a + 1), (tangentCurvey x - tangentLine a ha x)) = (2 / 3 : Real) :=
sorry

end TangentLineAndArea

end NUMINAMATH_GPT_tangent_line_eq_area_independent_of_a_l2275_227536


namespace NUMINAMATH_GPT_speed_boat_in_still_water_l2275_227555

-- Define the conditions
def speed_of_current := 20
def speed_upstream := 30

-- Define the effective speed given conditions
def effective_speed (speed_in_still_water : ℕ) := speed_in_still_water - speed_of_current

-- Theorem stating the problem
theorem speed_boat_in_still_water : 
  ∃ (speed_in_still_water : ℕ), effective_speed speed_in_still_water = speed_upstream ∧ speed_in_still_water = 50 := 
by 
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_speed_boat_in_still_water_l2275_227555


namespace NUMINAMATH_GPT_find_original_number_l2275_227598

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end NUMINAMATH_GPT_find_original_number_l2275_227598


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2275_227521

def setA : Set ℝ := { x : ℝ | x > -1 }
def setB : Set ℝ := { y : ℝ | 0 ≤ y ∧ y < 1 }

theorem intersection_of_A_and_B :
  (setA ∩ setB) = { z : ℝ | 0 ≤ z ∧ z < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2275_227521


namespace NUMINAMATH_GPT_brenda_more_than_jeff_l2275_227554

def emma_amount : ℕ := 8
def daya_amount : ℕ := emma_amount + (emma_amount * 25 / 100)
def jeff_amount : ℕ := (2 / 5) * daya_amount
def brenda_amount : ℕ := 8

theorem brenda_more_than_jeff :
  brenda_amount - jeff_amount = 4 :=
sorry

end NUMINAMATH_GPT_brenda_more_than_jeff_l2275_227554


namespace NUMINAMATH_GPT_num_possible_sums_l2275_227594

theorem num_possible_sums (s : Finset ℕ) (hs : s.card = 80) (hsub: s ⊆ Finset.range 121) : 
  ∃ (n : ℕ), (n = 3201) ∧ ∀ U, U = s.sum id → ∃ (U_min U_max : ℕ), U_min = 3240 ∧ U_max = 6440 ∧ (U_min ≤ U ∧ U ≤ U_max) :=
sorry

end NUMINAMATH_GPT_num_possible_sums_l2275_227594
