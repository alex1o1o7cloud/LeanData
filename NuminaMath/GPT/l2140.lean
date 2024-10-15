import Mathlib

namespace NUMINAMATH_GPT_radius_of_circle_l2140_214031

noncomputable def radius (α : ℝ) : ℝ :=
  5 / Real.sin (α / 2)

theorem radius_of_circle (c α : ℝ) (h_c : c = 10) :
  (radius α) = 5 / Real.sin (α / 2) := by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l2140_214031


namespace NUMINAMATH_GPT_number_of_paths_in_MATHEMATICIAN_diagram_l2140_214050

theorem number_of_paths_in_MATHEMATICIAN_diagram : ∃ n : ℕ, n = 8191 :=
by
  -- Define necessary structure
  -- Number of rows and binary choices
  let rows : ℕ := 12
  let choices_per_position : ℕ := 2
  -- Total paths calculation
  let total_paths := choices_per_position ^ rows
  -- Including symmetry and subtracting duplicate
  let final_paths := 2 * total_paths - 1
  use final_paths
  have : final_paths = 8191 :=
    by norm_num
  exact this

end NUMINAMATH_GPT_number_of_paths_in_MATHEMATICIAN_diagram_l2140_214050


namespace NUMINAMATH_GPT_PB_distance_eq_l2140_214000

theorem PB_distance_eq {
  A B C D P : Type
} (PA PD PC : ℝ) (hPA: PA = 6) (hPD: PD = 8) (hPC: PC = 10)
  (h_equidistant: ∃ y : ℝ, PA^2 + y^2 = PB^2 ∧ PD^2 + y^2 = PC^2) :
  ∃ PB : ℝ, PB = 6 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_PB_distance_eq_l2140_214000


namespace NUMINAMATH_GPT_a4_equals_9_l2140_214034

variable {a : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a4_equals_9 (h_geom : geometric_sequence a)
  (h_roots : ∃ a2 a6 : ℝ, a2^2 - 34 * a2 + 81 = 0 ∧ a6^2 - 34 * a6 + 81 = 0 ∧ a 2 = a2 ∧ a 6 = a6) :
  a 4 = 9 :=
sorry

end NUMINAMATH_GPT_a4_equals_9_l2140_214034


namespace NUMINAMATH_GPT_least_number_divisible_by_23_l2140_214074

theorem least_number_divisible_by_23 (n d : ℕ) (h_n : n = 1053) (h_d : d = 23) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_least_number_divisible_by_23_l2140_214074


namespace NUMINAMATH_GPT_CoveredAreaIs84_l2140_214039

def AreaOfStrip (length width : ℕ) : ℕ :=
  length * width

def TotalAreaWithoutOverlaps (numStrips areaOfOneStrip : ℕ) : ℕ :=
  numStrips * areaOfOneStrip

def OverlapArea (intersectionArea : ℕ) (numIntersections : ℕ) : ℕ :=
  intersectionArea * numIntersections

def ActualCoveredArea (totalArea overlapArea : ℕ) : ℕ :=
  totalArea - overlapArea

theorem CoveredAreaIs84 :
  let length := 12
  let width := 2
  let numStrips := 6
  let intersectionArea := width * width
  let numIntersections := 15
  let areaOfOneStrip := AreaOfStrip length width
  let totalAreaWithoutOverlaps := TotalAreaWithoutOverlaps numStrips areaOfOneStrip
  let totalOverlapArea := OverlapArea intersectionArea numIntersections
  ActualCoveredArea totalAreaWithoutOverlaps totalOverlapArea = 84 :=
by
  sorry

end NUMINAMATH_GPT_CoveredAreaIs84_l2140_214039


namespace NUMINAMATH_GPT_max_x_inequality_k_l2140_214042

theorem max_x_inequality_k (k : ℝ) (h : ∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) : k = 8 :=
sorry

end NUMINAMATH_GPT_max_x_inequality_k_l2140_214042


namespace NUMINAMATH_GPT_basketball_team_selection_l2140_214095

noncomputable def count_ways_excluding_twins (n k : ℕ) : ℕ :=
  let total_ways := Nat.choose n k
  let exhaustive_cases := Nat.choose (n - 2) (k - 2)
  total_ways - exhaustive_cases

theorem basketball_team_selection :
  count_ways_excluding_twins 12 5 = 672 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_selection_l2140_214095


namespace NUMINAMATH_GPT_width_of_room_l2140_214086

noncomputable def roomWidth (length : ℝ) (totalCost : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let area := totalCost / costPerSquareMeter
  area / length

theorem width_of_room :
  roomWidth 5.5 24750 1200 = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_width_of_room_l2140_214086


namespace NUMINAMATH_GPT_number_of_liars_l2140_214052

/-- There are 25 people in line, each of whom either tells the truth or lies.
The person at the front of the line says: "Everyone behind me is lying."
Everyone else says: "The person directly in front of me is lying."
Prove that the number of liars among these 25 people is 13. -/
theorem number_of_liars : 
  ∀ (persons : Fin 25 → Prop), 
    (persons 0 → ∀ n > 0, ¬persons n) →
    (∀ n : Nat, (1 ≤ n → n < 25 → persons n ↔ ¬persons (n - 1))) →
    (∃ l, l = 13 ∧ ∀ n : Nat, (0 ≤ n → n < 25 → persons n ↔ (n % 2 = 0))) :=
by
  sorry

end NUMINAMATH_GPT_number_of_liars_l2140_214052


namespace NUMINAMATH_GPT_power_function_solution_l2140_214082

def power_function_does_not_pass_through_origin (m : ℝ) : Prop :=
  (m^2 - m - 2) ≤ 0

def condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 3 = 1

theorem power_function_solution (m : ℝ) :
  power_function_does_not_pass_through_origin m ∧ condition m → (m = 1 ∨ m = 2) :=
by sorry

end NUMINAMATH_GPT_power_function_solution_l2140_214082


namespace NUMINAMATH_GPT_max_area_of_rectangle_l2140_214097

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 36) : (x * y) ≤ 81 :=
sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l2140_214097


namespace NUMINAMATH_GPT_expression_for_A_div_B_l2140_214054

theorem expression_for_A_div_B (x A B : ℝ)
  (h1 : x^3 + 1/x^3 = A)
  (h2 : x - 1/x = B) :
  A / B = B^2 + 3 := 
sorry

end NUMINAMATH_GPT_expression_for_A_div_B_l2140_214054


namespace NUMINAMATH_GPT_jeremy_goal_product_l2140_214020

theorem jeremy_goal_product 
  (g1 g2 g3 g4 g5 : ℕ) 
  (total5 : g1 + g2 + g3 + g4 + g5 = 13)
  (g6 g7 : ℕ) 
  (h6 : g6 < 10) 
  (h7 : g7 < 10) 
  (avg6 : (13 + g6) % 6 = 0) 
  (avg7 : (13 + g6 + g7) % 7 = 0) :
  g6 * g7 = 15 := 
sorry

end NUMINAMATH_GPT_jeremy_goal_product_l2140_214020


namespace NUMINAMATH_GPT_quadratic_eq_c_has_equal_roots_l2140_214075

theorem quadratic_eq_c_has_equal_roots (c : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + c = 0 ∧
                      ∀ y : ℝ, x^2 - 4 * x + c = 0 → y = x) : c = 4 := sorry

end NUMINAMATH_GPT_quadratic_eq_c_has_equal_roots_l2140_214075


namespace NUMINAMATH_GPT_triangular_pyramid_volume_l2140_214002

theorem triangular_pyramid_volume
  (b : ℝ) (h : ℝ) (H : ℝ)
  (b_pos : b = 4.5) (h_pos : h = 6) (H_pos : H = 8) :
  let base_area := (b * h) / 2
  let volume := (base_area * H) / 3
  volume = 36 := by
  sorry

end NUMINAMATH_GPT_triangular_pyramid_volume_l2140_214002


namespace NUMINAMATH_GPT_find_a_for_tangency_l2140_214068

-- Definitions of line and parabola
def line (x y : ℝ) : Prop := x - y - 1 = 0
def parabola (x y : ℝ) (a : ℝ) : Prop := y = a * x^2

-- The tangency condition for quadratic equations
def tangency_condition (a : ℝ) : Prop := 1 - 4 * a = 0

theorem find_a_for_tangency (a : ℝ) :
  (∀ x y, line x y → parabola x y a → tangency_condition a) → a = 1/4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_a_for_tangency_l2140_214068


namespace NUMINAMATH_GPT_dozens_in_each_box_l2140_214089

theorem dozens_in_each_box (boxes total_mangoes : ℕ) (h1 : boxes = 36) (h2 : total_mangoes = 4320) :
  (total_mangoes / 12) / boxes = 10 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_dozens_in_each_box_l2140_214089


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l2140_214005

theorem inscribed_sphere_radius 
  (a : ℝ) 
  (h_angle : ∀ (lateral_face : ℝ), lateral_face = 60) : 
  ∃ (r : ℝ), r = a * (Real.sqrt 3) / 6 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l2140_214005


namespace NUMINAMATH_GPT_arith_seq_a1_a2_a3_sum_l2140_214046

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a1_a2_a3_sum (a : ℕ → ℤ) (h_seq : arithmetic_seq a)
  (h1 : a 1 = 2) (h_sum : a 1 + a 2 + a 3 = 18) :
  a 4 + a 5 + a 6 = 54 :=
sorry

end NUMINAMATH_GPT_arith_seq_a1_a2_a3_sum_l2140_214046


namespace NUMINAMATH_GPT_valid_five_digit_integers_l2140_214062

/-- How many five-digit positive integers can be formed by arranging the digits 1, 1, 2, 3, 4 so 
that the two 1s are not next to each other -/
def num_valid_arrangements : ℕ :=
  36

theorem valid_five_digit_integers :
  ∃ n : ℕ, n = num_valid_arrangements :=
by
  use 36
  sorry

end NUMINAMATH_GPT_valid_five_digit_integers_l2140_214062


namespace NUMINAMATH_GPT_count_reflectional_symmetry_l2140_214040

def tetrominoes : List String := ["I", "O", "T", "S", "Z", "L", "J"]

def has_reflectional_symmetry (tetromino : String) : Bool :=
  match tetromino with
  | "I" => true
  | "O" => true
  | "T" => true
  | "S" => false
  | "Z" => false
  | "L" => false
  | "J" => false
  | _   => false

theorem count_reflectional_symmetry : 
  (tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end NUMINAMATH_GPT_count_reflectional_symmetry_l2140_214040


namespace NUMINAMATH_GPT_perimeter_of_rectangular_garden_l2140_214077

theorem perimeter_of_rectangular_garden (L W : ℝ) (h : L + W = 28) : 2 * (L + W) = 56 :=
by sorry

end NUMINAMATH_GPT_perimeter_of_rectangular_garden_l2140_214077


namespace NUMINAMATH_GPT_minimal_pyramid_height_l2140_214018

theorem minimal_pyramid_height (r x a : ℝ) (h₁ : 0 < r) (h₂ : a = 2 * r * x / (x - r)) (h₃ : x > 4 * r) :
  x = (6 + 2 * Real.sqrt 3) * r :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_minimal_pyramid_height_l2140_214018


namespace NUMINAMATH_GPT_coin_toss_fairness_l2140_214038

-- Statement of the problem as a Lean theorem.
theorem coin_toss_fairness (P_Heads P_Tails : ℝ) (h1 : P_Heads = 0.5) (h2 : P_Tails = 0.5) : 
  P_Heads = P_Tails ∧ P_Heads = 0.5 := 
sorry

end NUMINAMATH_GPT_coin_toss_fairness_l2140_214038


namespace NUMINAMATH_GPT_least_common_multiple_of_first_10_integers_l2140_214014

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end NUMINAMATH_GPT_least_common_multiple_of_first_10_integers_l2140_214014


namespace NUMINAMATH_GPT_quadratic_rewrite_l2140_214032

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 4) (h2 : 2 * d * e = 20) (h3 : e^2 + f = -24) :
  d * e = 10 :=
sorry

end NUMINAMATH_GPT_quadratic_rewrite_l2140_214032


namespace NUMINAMATH_GPT_find_2n_plus_m_l2140_214091

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 := 
sorry

end NUMINAMATH_GPT_find_2n_plus_m_l2140_214091


namespace NUMINAMATH_GPT_MinkyungHeight_is_correct_l2140_214073

noncomputable def HaeunHeight : ℝ := 1.56
noncomputable def NayeonHeight : ℝ := HaeunHeight - 0.14
noncomputable def MinkyungHeight : ℝ := NayeonHeight + 0.27

theorem MinkyungHeight_is_correct : MinkyungHeight = 1.69 :=
by
  sorry

end NUMINAMATH_GPT_MinkyungHeight_is_correct_l2140_214073


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2140_214011

variable (f : ℝ → ℝ)

def g (x : ℝ) : ℝ := f x - x - 1

theorem solution_set_of_inequality (h₁ : f 1 = 2) (h₂ : ∀ x, (deriv f x) < 1) :
  { x : ℝ | f x < x + 1 } = { x | 1 < x } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2140_214011


namespace NUMINAMATH_GPT_opposite_of_neg_five_l2140_214041

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end NUMINAMATH_GPT_opposite_of_neg_five_l2140_214041


namespace NUMINAMATH_GPT_solve_problem_l2140_214066

def is_solution (a : ℕ) : Prop :=
  a % 3 = 1 ∧ ∃ k : ℕ, a = 5 * k

theorem solve_problem : ∃ a : ℕ, is_solution a ∧ ∀ b : ℕ, is_solution b → a ≤ b := 
  sorry

end NUMINAMATH_GPT_solve_problem_l2140_214066


namespace NUMINAMATH_GPT_geometric_sequence_condition_l2140_214009

-- Given the sum of the first n terms of the sequence {a_n} is S_n = 2^n + c,
-- we need to prove that the sequence {a_n} is a geometric sequence if and only if c = -1.
theorem geometric_sequence_condition (c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S n = 2^n + c) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∃ q, ∀ n ≥ 1, a n = a 1 * q ^ (n - 1)) ↔ (c = -1) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l2140_214009


namespace NUMINAMATH_GPT_quadratic_roots_real_distinct_l2140_214047

theorem quadratic_roots_real_distinct (k : ℝ) :
  (k > (1/2)) ∧ (k ≠ 1) ↔
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k-1) * x1^2 + 2 * x1 - 2 = 0) ∧ ((k-1) * x2^2 + 2 * x2 - 2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_distinct_l2140_214047


namespace NUMINAMATH_GPT_product_closest_value_l2140_214060

-- Define the constants used in the problem
def a : ℝ := 2.5
def b : ℝ := 53.6
def c : ℝ := 0.4

-- Define the expression and the expected correct answer
def expression : ℝ := a * (b - c)
def correct_answer : ℝ := 133

-- State the theorem that the expression evaluates to the correct answer
theorem product_closest_value : expression = correct_answer :=
by
  sorry

end NUMINAMATH_GPT_product_closest_value_l2140_214060


namespace NUMINAMATH_GPT_minimum_value_of_f_l2140_214081

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 1 ∧ (∃ x₀ : ℝ, f x₀ = 1) := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2140_214081


namespace NUMINAMATH_GPT_evaluate_expression_l2140_214043

theorem evaluate_expression :
  4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2140_214043


namespace NUMINAMATH_GPT_rocking_chair_legs_l2140_214021

theorem rocking_chair_legs :
  let tables_4legs := 4 * 4
  let sofa_4legs := 1 * 4
  let chairs_4legs := 2 * 4
  let tables_3legs := 3 * 3
  let table_1leg := 1 * 1
  let total_legs := 40
  let accounted_legs := tables_4legs + sofa_4legs + chairs_4legs + tables_3legs + table_1leg
  ∃ rocking_chair_legs : Nat, total_legs = accounted_legs + rocking_chair_legs ∧ rocking_chair_legs = 2 :=
sorry

end NUMINAMATH_GPT_rocking_chair_legs_l2140_214021


namespace NUMINAMATH_GPT_sequence_a3_equals_1_over_3_l2140_214008

theorem sequence_a3_equals_1_over_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1) + 1)) : 
  a 3 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_sequence_a3_equals_1_over_3_l2140_214008


namespace NUMINAMATH_GPT_min_value_of_z_l2140_214027

noncomputable def min_z (x y : ℝ) : ℝ :=
  2 * x + (Real.sqrt 3) * y

theorem min_value_of_z :
  ∃ x y : ℝ, 3 * x^2 + 4 * y^2 = 12 ∧ min_z x y = -5 :=
sorry

end NUMINAMATH_GPT_min_value_of_z_l2140_214027


namespace NUMINAMATH_GPT_sector_area_l2140_214016

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 2
noncomputable def r : ℝ := l / θ

theorem sector_area :
  (1 / 2) * l * r = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sector_area_l2140_214016


namespace NUMINAMATH_GPT_solve_abs_eq_l2140_214099

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l2140_214099


namespace NUMINAMATH_GPT_lemonade_sales_l2140_214094

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_sales_l2140_214094


namespace NUMINAMATH_GPT_container_capacity_l2140_214007

theorem container_capacity (C : ℝ) (h₁ : C > 15) (h₂ : 0 < (81 : ℝ)) (h₃ : (337 : ℝ) > 0) :
  ((C - 15) / C) ^ 4 = 81 / 337 :=
sorry

end NUMINAMATH_GPT_container_capacity_l2140_214007


namespace NUMINAMATH_GPT_sequence_correctness_l2140_214017

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else -(2^(n - 1))

def partial_sum_S (n : ℕ) : ℤ := -2^n

theorem sequence_correctness (n : ℕ) (h : n ≥ 1) :
  (sequence_a 1 = -2) ∧ (∀ n ≥ 2, sequence_a (n + 1) = partial_sum_S n) ∧
  (sequence_a n = -(2^(n - 1))) ∧ (partial_sum_S n = -2^n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_correctness_l2140_214017


namespace NUMINAMATH_GPT_ratio_of_seconds_l2140_214076

theorem ratio_of_seconds (x : ℕ) :
  (12 : ℕ) / 8 = x / 240 → x = 360 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_seconds_l2140_214076


namespace NUMINAMATH_GPT_large_pyramid_tiers_l2140_214003

def surface_area_pyramid (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

theorem large_pyramid_tiers :
  (∃ n : ℕ, surface_area_pyramid n = 42) →
  (∃ n : ℕ, surface_area_pyramid n = 2352) →
  ∃ n : ℕ, surface_area_pyramid n = 2352 ∧ n = 24 :=
by
  sorry

end NUMINAMATH_GPT_large_pyramid_tiers_l2140_214003


namespace NUMINAMATH_GPT_expression_value_l2140_214006

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end NUMINAMATH_GPT_expression_value_l2140_214006


namespace NUMINAMATH_GPT_adam_earnings_l2140_214026

def lawns_to_mow : ℕ := 12
def lawns_forgotten : ℕ := 8
def earnings_per_lawn : ℕ := 9

theorem adam_earnings : (lawns_to_mow - lawns_forgotten) * earnings_per_lawn = 36 := by
  sorry

end NUMINAMATH_GPT_adam_earnings_l2140_214026


namespace NUMINAMATH_GPT_distinct_real_roots_l2140_214037

def operation (a b : ℝ) : ℝ := a^2 - a * b + b

theorem distinct_real_roots {x : ℝ} : 
  (operation x 3 = 5) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation x1 3 = 5 ∧ operation x2 3 = 5) :=
by 
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_distinct_real_roots_l2140_214037


namespace NUMINAMATH_GPT_sarahs_team_mean_score_l2140_214092

def mean_score_of_games (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sarahs_team_mean_score :
  mean_score_of_games [69, 68, 70, 61, 74, 62, 65, 74] = 67.875 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_team_mean_score_l2140_214092


namespace NUMINAMATH_GPT_train_crosses_signal_pole_in_20_seconds_l2140_214044

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 285
noncomputable def total_time_to_cross_platform : ℝ := 39

-- Define the speed of the train
noncomputable def train_speed : ℝ := (train_length + platform_length) / total_time_to_cross_platform

-- Define the expected time to cross the signal pole
noncomputable def time_to_cross_signal_pole : ℝ := train_length / train_speed

theorem train_crosses_signal_pole_in_20_seconds :
  time_to_cross_signal_pole = 20 := by
  sorry

end NUMINAMATH_GPT_train_crosses_signal_pole_in_20_seconds_l2140_214044


namespace NUMINAMATH_GPT_two_pipes_fill_time_l2140_214085

theorem two_pipes_fill_time (R : ℝ) (h1 : (3 : ℝ) * R * (8 : ℝ) = 1) : (2 : ℝ) * R * (12 : ℝ) = 1 :=
by 
  have hR : R = 1 / 24 := by linarith
  rw [hR]
  sorry

end NUMINAMATH_GPT_two_pipes_fill_time_l2140_214085


namespace NUMINAMATH_GPT_division_631938_by_625_l2140_214019

theorem division_631938_by_625 :
  (631938 : ℚ) / 625 = 1011.1008 :=
by
  -- Add a placeholder proof. We do not provide the solution steps.
  sorry

end NUMINAMATH_GPT_division_631938_by_625_l2140_214019


namespace NUMINAMATH_GPT_steve_keeps_total_money_excluding_advance_l2140_214024

-- Definitions of the conditions
def totalCopies : ℕ := 1000000
def advanceCopies : ℕ := 100000
def pricePerCopy : ℕ := 2
def agentCommissionRate : ℚ := 0.1

-- Question and final proof
theorem steve_keeps_total_money_excluding_advance :
  let totalEarnings := totalCopies * pricePerCopy
  let agentCommission := agentCommissionRate * totalEarnings
  let moneyKept := totalEarnings - agentCommission
  moneyKept = 1800000 := by
  -- Proof goes here, but we skip it for now
  sorry

end NUMINAMATH_GPT_steve_keeps_total_money_excluding_advance_l2140_214024


namespace NUMINAMATH_GPT_reciprocal_opposites_l2140_214004

theorem reciprocal_opposites (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end NUMINAMATH_GPT_reciprocal_opposites_l2140_214004


namespace NUMINAMATH_GPT_tree_height_equation_l2140_214023

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end NUMINAMATH_GPT_tree_height_equation_l2140_214023


namespace NUMINAMATH_GPT_total_expenditure_l2140_214067

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end NUMINAMATH_GPT_total_expenditure_l2140_214067


namespace NUMINAMATH_GPT_numerical_value_expression_l2140_214083

theorem numerical_value_expression (x y z : ℚ) (h1 : x - 4 * y - 2 * z = 0) (h2 : 3 * x + 2 * y - z = 0) (h3 : z ≠ 0) : 
  (x^2 - 5 * x * y) / (2 * y^2 + z^2) = 164 / 147 :=
by sorry

end NUMINAMATH_GPT_numerical_value_expression_l2140_214083


namespace NUMINAMATH_GPT_pigeon_count_correct_l2140_214065

def initial_pigeon_count : ℕ := 1
def new_pigeon_count : ℕ := 1
def total_pigeon_count : ℕ := 2

theorem pigeon_count_correct : initial_pigeon_count + new_pigeon_count = total_pigeon_count :=
by
  sorry

end NUMINAMATH_GPT_pigeon_count_correct_l2140_214065


namespace NUMINAMATH_GPT_range_of_m_l2140_214036

theorem range_of_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) (h4 : 4 / a + 1 / (b - 1) > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2140_214036


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l2140_214001

theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ), a^2 = 16 → b^2 = 8 → c^2 = a^2 - b^2 → e = c / a → e = (Real.sqrt 2) / 2 := 
by 
  intros a b c e ha hb hc he
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l2140_214001


namespace NUMINAMATH_GPT_smallest_a_for_polynomial_roots_l2140_214057

theorem smallest_a_for_polynomial_roots :
  ∃ (a b c : ℕ), 
         (∃ (r s t u : ℕ), r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧ r * s * t * u = 5160 ∧ a = r + s + t + u) 
    ∧  (∀ (r' s' t' u' : ℕ), r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧ r' * s' * t' * u' = 5160 ∧ r' + s' + t' + u' < a → false) 
    := sorry

end NUMINAMATH_GPT_smallest_a_for_polynomial_roots_l2140_214057


namespace NUMINAMATH_GPT_caps_difference_l2140_214084

theorem caps_difference (Billie_caps Sammy_caps : ℕ) (Janine_caps := 3 * Billie_caps)
  (Billie_has : Billie_caps = 2) (Sammy_has : Sammy_caps = 8) :
  Sammy_caps - Janine_caps = 2 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_caps_difference_l2140_214084


namespace NUMINAMATH_GPT_fourth_term_of_sequence_l2140_214049

-- Given conditions
def first_term : ℕ := 5
def fifth_term : ℕ := 1280

-- Definition of the common ratio
def common_ratio (a : ℕ) (b : ℕ) : ℕ := (b / a)^(1 / 4)

-- Function to calculate the nth term of a geometric sequence
def nth_term (a r n : ℕ) : ℕ := a * r^(n - 1)

-- Prove the fourth term of the geometric sequence is 320
theorem fourth_term_of_sequence 
    (a : ℕ) (b : ℕ) (a_pos : a = first_term) (b_eq : nth_term a (common_ratio a b) 5 = b) : 
    nth_term a (common_ratio a b) 4 = 320 := by
  sorry

end NUMINAMATH_GPT_fourth_term_of_sequence_l2140_214049


namespace NUMINAMATH_GPT_contrapositive_l2140_214093

variables (p q : Prop)

theorem contrapositive (hpq : p → q) : ¬ q → ¬ p :=
by sorry

end NUMINAMATH_GPT_contrapositive_l2140_214093


namespace NUMINAMATH_GPT_students_standing_together_l2140_214096

theorem students_standing_together (s : Finset ℕ) (h_size : s.card = 6) (a b : ℕ) (h_ab : a ∈ s ∧ b ∈ s) (h_ab_together : ∃ (l : List ℕ), l.length = 6 ∧ a :: b :: l = l):
  ∃ (arrangements : ℕ), arrangements = 240 := by
  sorry

end NUMINAMATH_GPT_students_standing_together_l2140_214096


namespace NUMINAMATH_GPT_original_selling_price_is_990_l2140_214012

theorem original_selling_price_is_990 
( P : ℝ ) -- original purchase price
( SP_1 : ℝ := 1.10 * P ) -- original selling price
( P_new : ℝ := 0.90 * P ) -- new purchase price
( SP_2 : ℝ := 1.17 * P ) -- new selling price
( h : SP_2 - SP_1 = 63 ) : SP_1 = 990 :=
by {
  -- This is just the statement, proof is not provided
  sorry
}

end NUMINAMATH_GPT_original_selling_price_is_990_l2140_214012


namespace NUMINAMATH_GPT_constant_term_expansion_eq_sixty_l2140_214045

theorem constant_term_expansion_eq_sixty (a : ℝ) (h : 15 * a = 60) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_expansion_eq_sixty_l2140_214045


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2140_214025

-- Lean 4 statements for the given problems:
theorem solve_equation1 (x : ℝ) (h : x ≠ 0) : (2 / x = 3 / (x + 2)) ↔ (x = 4) := by
  sorry

theorem solve_equation2 (x : ℝ) (h : x ≠ 2) : ¬(5 / (x - 2) + 1 = (x - 7) / (2 - x)) := by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2140_214025


namespace NUMINAMATH_GPT_white_pairs_coincide_l2140_214030

theorem white_pairs_coincide
  (red_triangles_half : ℕ)
  (blue_triangles_half : ℕ)
  (white_triangles_half : ℕ)
  (red_pairs : ℕ)
  (blue_pairs : ℕ)
  (red_white_pairs : ℕ)
  (red_triangles_total_half : red_triangles_half = 4)
  (blue_triangles_total_half : blue_triangles_half = 6)
  (white_triangles_total_half : white_triangles_half = 10)
  (red_pairs_total : red_pairs = 3)
  (blue_pairs_total : blue_pairs = 4)
  (red_white_pairs_total : red_white_pairs = 3) :
  ∃ w : ℕ, w = 5 :=
by
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l2140_214030


namespace NUMINAMATH_GPT_attraction_ticket_cost_l2140_214056

theorem attraction_ticket_cost
  (cost_park_entry : ℕ)
  (cost_attraction_parent : ℕ)
  (total_paid : ℕ)
  (num_children : ℕ)
  (num_parents : ℕ)
  (num_grandmother : ℕ)
  (x : ℕ)
  (h_costs : cost_park_entry = 5)
  (h_attraction_parent : cost_attraction_parent = 4)
  (h_family : num_children = 4 ∧ num_parents = 2 ∧ num_grandmother = 1)
  (h_total_paid : total_paid = 55)
  (h_equation : (num_children + num_parents + num_grandmother) * cost_park_entry + (num_parents + num_grandmother) * cost_attraction_parent + num_children * x = total_paid) :
  x = 2 := by
  sorry

end NUMINAMATH_GPT_attraction_ticket_cost_l2140_214056


namespace NUMINAMATH_GPT_Meadow_sells_each_diaper_for_5_l2140_214013

-- Define the conditions as constants
def boxes_per_week := 30
def packs_per_box := 40
def diapers_per_pack := 160
def total_revenue := 960000

-- Calculate total packs and total diapers
def total_packs := boxes_per_week * packs_per_box
def total_diapers := total_packs * diapers_per_pack

-- The target price per diaper
def price_per_diaper := total_revenue / total_diapers

-- Statement of the proof theorem
theorem Meadow_sells_each_diaper_for_5 : price_per_diaper = 5 := by
  sorry

end NUMINAMATH_GPT_Meadow_sells_each_diaper_for_5_l2140_214013


namespace NUMINAMATH_GPT_integer_ratio_condition_l2140_214090

variable {x y : ℝ}

theorem integer_ratio_condition (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, t = -2 := sorry

end NUMINAMATH_GPT_integer_ratio_condition_l2140_214090


namespace NUMINAMATH_GPT_minimum_loadings_to_prove_first_ingot_weighs_1kg_l2140_214070

theorem minimum_loadings_to_prove_first_ingot_weighs_1kg :
  ∀ (w : Fin 11 → ℕ), 
    (∀ i, w i = i + 1) →
    (∃ s₁ s₂ : Finset (Fin 11), 
       s₁.card ≤ 6 ∧ s₂.card ≤ 6 ∧ 
       s₁.sum w = 11 ∧ s₂.sum w = 11 ∧ 
       (∀ s : Finset (Fin 11), s.sum w = 11 → s ≠ s₁ ∧ s ≠ s₂) ∧
       (w 0 = 1)) := sorry -- Fill in the proof here

end NUMINAMATH_GPT_minimum_loadings_to_prove_first_ingot_weighs_1kg_l2140_214070


namespace NUMINAMATH_GPT_sum_of_tens_l2140_214010

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_l2140_214010


namespace NUMINAMATH_GPT_rational_root_even_denominator_l2140_214069

theorem rational_root_even_denominator
  (a b c : ℤ)
  (sum_ab_even : (a + b) % 2 = 0)
  (c_odd : c % 2 = 1) :
  ∀ (p q : ℤ), (q ≠ 0) → (IsRationalRoot : a * (p * p) + b * p * q + c * (q * q) = 0) →
    gcd p q = 1 → q % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_root_even_denominator_l2140_214069


namespace NUMINAMATH_GPT_x_coordinate_of_second_point_l2140_214022

variable (m n : ℝ)

theorem x_coordinate_of_second_point
  (h1 : m = 2 * n + 5)
  (h2 : (m + 5) = 2 * (n + 2.5) + 5) :
  (m + 5) = m + 5 :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_of_second_point_l2140_214022


namespace NUMINAMATH_GPT_minimize_costs_l2140_214079

def total_books : ℕ := 150000
def handling_fee_per_order : ℕ := 30
def storage_fee_per_1000_copies : ℕ := 40
def evenly_distributed_books : Prop := true --Assuming books are evenly distributed by default

noncomputable def optimal_order_frequency : ℕ := 10
noncomputable def optimal_batch_size : ℕ := 15000

theorem minimize_costs 
  (handling_fee_per_order : ℕ) 
  (storage_fee_per_1000_copies : ℕ) 
  (total_books : ℕ) 
  (evenly_distributed_books : Prop)
  : optimal_order_frequency = 10 ∧ optimal_batch_size = 15000 := sorry

end NUMINAMATH_GPT_minimize_costs_l2140_214079


namespace NUMINAMATH_GPT_inequality_solution_range_l2140_214059

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x+2| + |x-3| < a) ↔ a > 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l2140_214059


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2140_214098

-- Definitions translated from conditions
noncomputable def parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
noncomputable def a : ℝ := 2
noncomputable def c : ℝ := Real.sqrt 5

-- Eccentricity formula for the hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement to be proved
theorem hyperbola_eccentricity :
  eccentricity c a = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2140_214098


namespace NUMINAMATH_GPT_geometric_series_sum_l2140_214028

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  S = 341 / 1024 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S := a * (1 - r^n) / (1 - r)
  show S = 341 / 1024
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2140_214028


namespace NUMINAMATH_GPT_math_problem_correct_l2140_214078

noncomputable def math_problem : Prop :=
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40)

theorem math_problem_correct : math_problem := by
  sorry

end NUMINAMATH_GPT_math_problem_correct_l2140_214078


namespace NUMINAMATH_GPT_chocolate_bar_percentage_l2140_214048

theorem chocolate_bar_percentage (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
  (h1 : milk_chocolate = 25) (h2 : dark_chocolate = 25)
  (h3 : almond_chocolate = 25) (h4 : white_chocolate = 25) :
  (milk_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (dark_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (almond_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (white_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_percentage_l2140_214048


namespace NUMINAMATH_GPT_vector_intersecting_line_parameter_l2140_214051

theorem vector_intersecting_line_parameter :
  ∃ (a b s : ℝ), a = 3 * s + 5 ∧ b = 2 * s + 4 ∧
                   (∃ r, (a, b) = (3 * r, 2 * r)) ∧
                   (a, b) = (6, 14 / 3) :=
by
  sorry

end NUMINAMATH_GPT_vector_intersecting_line_parameter_l2140_214051


namespace NUMINAMATH_GPT_opposite_neg_inv_three_l2140_214015

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end NUMINAMATH_GPT_opposite_neg_inv_three_l2140_214015


namespace NUMINAMATH_GPT_cos_D_zero_l2140_214080

noncomputable def area_of_triangle (a b: ℝ) (sinD: ℝ) : ℝ := 1 / 2 * a * b * sinD

theorem cos_D_zero (DE DF : ℝ) (D : ℝ) (h1 : area_of_triangle DE DF (Real.sin D) = 98) (h2 : Real.sqrt (DE * DF) = 14) : Real.cos D = 0 :=
  by
  sorry

end NUMINAMATH_GPT_cos_D_zero_l2140_214080


namespace NUMINAMATH_GPT_pan_dimensions_l2140_214088

theorem pan_dimensions (m n : ℕ) : 
  (∃ m n, m * n = 48 ∧ (m-2) * (n-2) = 2 * (2*m + 2*n - 4) ∧ m > 2 ∧ n > 2) → 
  (m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
by
  sorry

end NUMINAMATH_GPT_pan_dimensions_l2140_214088


namespace NUMINAMATH_GPT_derivative_is_even_then_b_eq_zero_l2140_214061

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- The statement that the derivative is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Our main theorem
theorem derivative_is_even_then_b_eq_zero : is_even (f' a b c) → b = 0 :=
by
  intro h
  have h1 := h 1
  have h2 := h (-1)
  sorry

end NUMINAMATH_GPT_derivative_is_even_then_b_eq_zero_l2140_214061


namespace NUMINAMATH_GPT_min_value_quadratic_l2140_214033

theorem min_value_quadratic (x : ℝ) : 
  ∃ m, m = 3 * x^2 - 18 * x + 2048 ∧ ∀ x, 3 * x^2 - 18 * x + 2048 ≥ 2021 :=
by sorry

end NUMINAMATH_GPT_min_value_quadratic_l2140_214033


namespace NUMINAMATH_GPT_max_soap_boxes_l2140_214029

theorem max_soap_boxes :
  ∀ (L_carton W_carton H_carton L_soap_box W_soap_box H_soap_box : ℕ)
   (V_carton V_soap_box : ℕ) 
   (h1 : L_carton = 25) 
   (h2 : W_carton = 42)
   (h3 : H_carton = 60) 
   (h4 : L_soap_box = 7)
   (h5 : W_soap_box = 6)
   (h6 : H_soap_box = 10)
   (h7 : V_carton = L_carton * W_carton * H_carton)
   (h8 : V_soap_box = L_soap_box * W_soap_box * H_soap_box),
   V_carton / V_soap_box = 150 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_soap_boxes_l2140_214029


namespace NUMINAMATH_GPT_fraction_zero_x_eq_2_l2140_214058

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end NUMINAMATH_GPT_fraction_zero_x_eq_2_l2140_214058


namespace NUMINAMATH_GPT_gcd_gx_x_l2140_214035

-- Condition: x is a multiple of 7263
def isMultipleOf7263 (x : ℕ) : Prop := ∃ k : ℕ, x = 7263 * k

-- Definition of g(x)
def g (x : ℕ) : ℕ := (3*x + 4) * (9*x + 5) * (17*x + 11) * (x + 17)

-- Statement to be proven
theorem gcd_gx_x (x : ℕ) (h : isMultipleOf7263 x) : Nat.gcd (g x) x = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_l2140_214035


namespace NUMINAMATH_GPT_find_f_of_fraction_l2140_214072

noncomputable def f (t : ℝ) : ℝ := sorry

theorem find_f_of_fraction (x : ℝ) (h : f ((1-x^2)/(1+x^2)) = x) :
  f ((2*x)/(1+x^2)) = (1 - x) / (1 + x) ∨ f ((2*x)/(1+x^2)) = (x - 1) / (1 + x) :=
sorry

end NUMINAMATH_GPT_find_f_of_fraction_l2140_214072


namespace NUMINAMATH_GPT_find_R_l2140_214064

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → ¬ (m ∣ n)

theorem find_R :
  ∃ R : ℤ, R > 0 ∧ (∃ Q : ℤ, is_prime (R^3 + 4 * R^2 + (Q - 93) * R + 14 * Q + 10)) ∧ R = 5 :=
  sorry

end NUMINAMATH_GPT_find_R_l2140_214064


namespace NUMINAMATH_GPT_fraction_product_simplification_l2140_214053

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_simplification_l2140_214053


namespace NUMINAMATH_GPT_quadratic_non_real_roots_iff_l2140_214071

theorem quadratic_non_real_roots_iff (b : ℝ) : 
  (∀ x : ℂ, ¬ (x^2 + b*x + 16 = 0) → (b^2 - 4 * 1 * 16 < 0)) ↔ -8 < b ∧ b < 8 :=
by sorry

end NUMINAMATH_GPT_quadratic_non_real_roots_iff_l2140_214071


namespace NUMINAMATH_GPT_plant_height_increase_l2140_214063

theorem plant_height_increase (total_increase : ℕ) (century_in_years : ℕ) (decade_in_years : ℕ) (years_in_2_centuries : ℕ) (num_decades : ℕ) : 
  total_increase = 1800 →
  century_in_years = 100 →
  decade_in_years = 10 →
  years_in_2_centuries = 2 * century_in_years →
  num_decades = years_in_2_centuries / decade_in_years →
  total_increase / num_decades = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_plant_height_increase_l2140_214063


namespace NUMINAMATH_GPT_decreased_and_divided_l2140_214087

theorem decreased_and_divided (x : ℝ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 := by
  sorry

end NUMINAMATH_GPT_decreased_and_divided_l2140_214087


namespace NUMINAMATH_GPT_problem_statement_l2140_214055

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (a b : ℝ)

theorem problem_statement (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, HasDerivAt g (g' x) x)
                         (h3 : ∀ x, f' x < g' x)
                         (h4 : a = Real.log 2 / Real.log 5)
                         (h5 : b = Real.log 3 / Real.log 8) :
                         f a + g b > g a + f b := 
     sorry

end NUMINAMATH_GPT_problem_statement_l2140_214055
