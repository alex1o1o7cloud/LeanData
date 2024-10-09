import Mathlib

namespace sum_first_and_third_angle_l2079_207999

-- Define the conditions
variable (A : ℕ)
axiom C1 : A + 2 * A + (A - 40) = 180

-- State the theorem to be proven
theorem sum_first_and_third_angle : A + (A - 40) = 70 :=
by
  sorry

end sum_first_and_third_angle_l2079_207999


namespace total_dots_not_visible_l2079_207985

theorem total_dots_not_visible :
  let total_dots := 4 * 21
  let visible_sum := 1 + 2 + 3 + 3 + 4 + 5 + 5 + 6
  total_dots - visible_sum = 55 :=
by
  sorry

end total_dots_not_visible_l2079_207985


namespace birch_count_is_87_l2079_207962

def num_trees : ℕ := 130
def incorrect_signs (B L : ℕ) : Prop := B + L = num_trees ∧ L + 1 = num_trees - 1 ∧ B = 87

theorem birch_count_is_87 (B L : ℕ) (h1 : B + L = num_trees) (h2 : L + 1 = num_trees - 1) :
  B = 87 :=
sorry

end birch_count_is_87_l2079_207962


namespace polar_to_cartesian_coordinates_l2079_207974

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_coordinates :
  polar_to_cartesian 2 (2 / 3 * Real.pi) = (-1, Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_coordinates_l2079_207974


namespace october_birth_percentage_l2079_207911

theorem october_birth_percentage 
  (jan feb mar apr may jun jul aug sep oct nov dec total : ℕ) 
  (h_total : total = 100)
  (h_jan : jan = 2) (h_feb : feb = 4) (h_mar : mar = 8) (h_apr : apr = 5) 
  (h_may : may = 4) (h_jun : jun = 9) (h_jul : jul = 7) (h_aug : aug = 12) 
  (h_sep : sep = 8) (h_oct : oct = 6) (h_nov : nov = 5) (h_dec : dec = 4) : 
  (oct : ℕ) * 100 / total = 6 := 
by
  sorry

end october_birth_percentage_l2079_207911


namespace find_decimal_decrease_l2079_207993

noncomputable def tax_diminished_percentage (T C : ℝ) (X : ℝ) : Prop :=
  let new_tax := T * (1 - X / 100)
  let new_consumption := C * 1.15
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  new_revenue = original_revenue * 0.943

theorem find_decimal_decrease (T C : ℝ) (X : ℝ) :
  tax_diminished_percentage T C X → X = 18 := sorry

end find_decimal_decrease_l2079_207993


namespace smaller_circle_circumference_l2079_207909

theorem smaller_circle_circumference (r r2 : ℝ) : 
  (60:ℝ) / 360 * 2 * Real.pi * r = 8 →
  r = 24 / Real.pi →
  1 / 4 * (24 / Real.pi)^2 = (24 / Real.pi - 2 * r2) * (24 / Real.pi) →
  2 * Real.pi * r2 = 36 :=
  by
    intros h1 h2 h3
    sorry

end smaller_circle_circumference_l2079_207909


namespace equal_product_groups_exist_l2079_207939

def numbers : List ℕ := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

theorem equal_product_groups_exist :
  ∃ (g1 g2 : List ℕ), 
    g1.length = 5 ∧ g2.length = 5 ∧ 
    g1.prod = g2.prod ∧ g1.prod = 349188840 ∧ 
    (g1 ++ g2 = numbers ∨ g1 ++ g2 = numbers.reverse) :=
by
  sorry

end equal_product_groups_exist_l2079_207939


namespace find_C_l2079_207951

noncomputable def h (C D : ℝ) (x : ℝ) : ℝ := 2 * C * x - 3 * D ^ 2
def k (D : ℝ) (x : ℝ) := D * x

theorem find_C (C D : ℝ) (h_eq : h C D (k D 2) = 0) (hD : D ≠ 0) : C = 3 * D / 4 :=
by
  unfold h k at h_eq
  sorry

end find_C_l2079_207951


namespace find_original_number_l2079_207930

theorem find_original_number (c : ℝ) (h₁ : c / 12.75 = 16) (h₂ : 2.04 / 1.275 = 1.6) : c = 204 :=
by
  sorry

end find_original_number_l2079_207930


namespace volume_ratio_of_spheres_l2079_207995

theorem volume_ratio_of_spheres (r1 r2 r3 : ℝ) 
  (h : r1 / r2 = 1 / 2 ∧ r2 / r3 = 2 / 3) : 
  (4/3 * π * r3^3) = 3 * (4/3 * π * r1^3 + 4/3 * π * r2^3) :=
by
  sorry

end volume_ratio_of_spheres_l2079_207995


namespace nolan_monthly_savings_l2079_207906

theorem nolan_monthly_savings (m k : ℕ) (H : 12 * m = 36 * k) : m = 3 * k := 
by sorry

end nolan_monthly_savings_l2079_207906


namespace sufficient_not_necessary_l2079_207969

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end sufficient_not_necessary_l2079_207969


namespace rectangle_area_l2079_207926

-- Definitions:
variables (l w : ℝ)

-- Conditions:
def condition1 : Prop := l = 4 * w
def condition2 : Prop := 2 * l + 2 * w = 200

-- Theorem statement:
theorem rectangle_area (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 1600 :=
sorry

end rectangle_area_l2079_207926


namespace negate_proposition_p_l2079_207908

theorem negate_proposition_p (f : ℝ → ℝ) :
  (¬ ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) >= 0) ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
sorry

end negate_proposition_p_l2079_207908


namespace part_b_part_c_l2079_207977

-- Statement for part b: In how many ways can the figure be properly filled with the numbers from 1 to 5?
def proper_fill_count_1_to_5 : Nat :=
  8

-- Statement for part c: In how many ways can the figure be properly filled with the numbers from 1 to 7?
def proper_fill_count_1_to_7 : Nat :=
  48

theorem part_b :
  proper_fill_count_1_to_5 = 8 :=
sorry

theorem part_c :
  proper_fill_count_1_to_7 = 48 :=
sorry

end part_b_part_c_l2079_207977


namespace shortest_distance_from_vertex_to_path_l2079_207929

theorem shortest_distance_from_vertex_to_path
  (r l : ℝ)
  (hr : r = 1)
  (hl : l = 3) :
  ∃ d : ℝ, d = 1.5 :=
by
  -- Given a cone with a base radius of 1 cm and a slant height of 3 cm
  -- We need to prove the shortest distance from the vertex to the path P back to P is 1.5 cm
  sorry

end shortest_distance_from_vertex_to_path_l2079_207929


namespace total_output_equal_at_20_l2079_207925

noncomputable def total_output_A (x : ℕ) : ℕ :=
  200 + 20 * x

noncomputable def total_output_B (x : ℕ) : ℕ :=
  30 * x

theorem total_output_equal_at_20 :
  total_output_A 20 = total_output_B 20 :=
by
  sorry

end total_output_equal_at_20_l2079_207925


namespace find_common_difference_find_minimum_sum_minimum_sum_value_l2079_207978

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) (d : ℚ) := (-3 : ℚ) + n * d

-- Given conditions
def condition_1 : ℚ := -3
def condition_2 (d : ℚ) := 11 * a 4 d = 5 * a 7 d - 13
def common_difference : ℚ := 31 / 9

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (d : ℚ) := n * (-3 + (n - 1) * d / 2)

-- Defining the necessary theorems
theorem find_common_difference (d : ℚ) : condition_2 d → d = common_difference := by
  sorry

theorem find_minimum_sum (n : ℕ) : S n common_difference ≥ S 2 common_difference := by
  sorry

theorem minimum_sum_value : S 2 common_difference = -23 / 9 := by
  sorry

end find_common_difference_find_minimum_sum_minimum_sum_value_l2079_207978


namespace no_perfect_square_m_in_range_l2079_207902

theorem no_perfect_square_m_in_range : 
  ∀ m : ℕ, 4 ≤ m ∧ m ≤ 12 → ¬(∃ k : ℕ, 2 * m^2 + 3 * m + 2 = k^2) := by
sorry

end no_perfect_square_m_in_range_l2079_207902


namespace difference_sweaters_Monday_Tuesday_l2079_207938

-- Define conditions
def sweaters_knit_on_Monday : ℕ := 8
def sweaters_knit_on_Tuesday (T : ℕ) : Prop := T > 8
def sweaters_knit_on_Wednesday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Thursday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Friday : ℕ := 4

-- Define total sweaters knit in the week
def total_sweaters_knit (T : ℕ) : ℕ :=
  sweaters_knit_on_Monday + T + sweaters_knit_on_Wednesday T + sweaters_knit_on_Thursday T + sweaters_knit_on_Friday

-- Lean Theorem Statement
theorem difference_sweaters_Monday_Tuesday : ∀ T : ℕ, sweaters_knit_on_Tuesday T → total_sweaters_knit T = 34 → T - sweaters_knit_on_Monday = 2 :=
by
  intros T hT_total
  sorry

end difference_sweaters_Monday_Tuesday_l2079_207938


namespace sqrt_two_irrational_l2079_207967

theorem sqrt_two_irrational :
  ¬ ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (↑q / ↑p) ^ 2 = (2:ℝ) :=
sorry

end sqrt_two_irrational_l2079_207967


namespace fred_seashells_l2079_207980

-- Definitions based on conditions
def tom_seashells : Nat := 15
def total_seashells : Nat := 58

-- The theorem we want to prove
theorem fred_seashells : (15 + F = 58) → F = 43 := 
by
  intro h
  have h1 : F = 58 - 15 := by linarith
  exact h1

end fred_seashells_l2079_207980


namespace valid_inequalities_l2079_207915

theorem valid_inequalities (a b c : ℝ) (h : 0 < c) 
  (h1 : b > c - b)
  (h2 : c > a)
  (h3 : c > b - a) :
  a < c / 2 ∧ b < a + c / 2 :=
by
  sorry

end valid_inequalities_l2079_207915


namespace find_value_of_a_l2079_207905

theorem find_value_of_a (x a : ℝ) (h : 2 * x - a + 5 = 0) (h_x : x = -2) : a = 1 :=
by
  sorry

end find_value_of_a_l2079_207905


namespace inequality_one_inequality_two_l2079_207935

-- Problem (1)
theorem inequality_one {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : 2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

-- Problem (2)
theorem inequality_two {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : (a ^ 2 / b + b ^ 2 / c + c ^ 2 / a) ≥ 1 :=
sorry

end inequality_one_inequality_two_l2079_207935


namespace trajectory_of_point_P_l2079_207991

theorem trajectory_of_point_P :
  ∀ (x y : ℝ), 
  (∀ (m n : ℝ), n = 2 * m - 4 → (1 - m, -n) = (x - 1, y)) → 
  y = 2 * x :=
by
  sorry

end trajectory_of_point_P_l2079_207991


namespace simplest_square_root_among_choices_l2079_207904

variable {x : ℝ}

def is_simplest_square_root (n : ℝ) : Prop :=
  ∀ m, (m^2 = n) → (m = n)

theorem simplest_square_root_among_choices :
  is_simplest_square_root 7 ∧ ∀ n, n = 24 ∨ n = 1/3 ∨ n = 0.2 → ¬ is_simplest_square_root n :=
by
  sorry

end simplest_square_root_among_choices_l2079_207904


namespace platinum_earrings_percentage_l2079_207948

theorem platinum_earrings_percentage
  (rings_percentage ornaments_percentage : ℝ)
  (rings_percentage_eq : rings_percentage = 0.30)
  (earrings_percentage_eq : ornaments_percentage - rings_percentage = 0.70)
  (platinum_earrings_percentage : ℝ)
  (platinum_earrings_percentage_eq : platinum_earrings_percentage = 0.70) :
  ornaments_percentage * platinum_earrings_percentage = 0.49 :=
by 
  have earrings_percentage := 0.70
  have ornaments_percentage := 0.70
  sorry

end platinum_earrings_percentage_l2079_207948


namespace find_subtracted_number_l2079_207992

theorem find_subtracted_number (x y : ℝ) (h1 : x = 62.5) (h2 : (2 * (x + 5)) / 5 - y = 22) : y = 5 :=
sorry

end find_subtracted_number_l2079_207992


namespace stratified_sampling_sophomores_l2079_207941

theorem stratified_sampling_sophomores
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (total_selected : ℕ)
  (H_freshmen : freshmen = 550) (H_sophomores : sophomores = 700) (H_juniors : juniors = 750) (H_total_selected : total_selected = 100) :
  sophomores * total_selected / (freshmen + sophomores + juniors) = 35 :=
by
  sorry

end stratified_sampling_sophomores_l2079_207941


namespace probability_last_passenger_own_seat_is_half_l2079_207927

open Classical

-- Define the behavior and probability question:

noncomputable def probability_last_passenger_own_seat (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 2

-- The main theorem stating the probability for an arbitrary number of passengers n
-- The theorem that needs to be proved:
theorem probability_last_passenger_own_seat_is_half (n : ℕ) (h : n > 0) : 
  probability_last_passenger_own_seat n = 1 / 2 :=
by sorry

end probability_last_passenger_own_seat_is_half_l2079_207927


namespace find_n_l2079_207914

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem find_n 
  (n : ℕ)
  (h1 : (binom (n-6) 7) / binom n 7 = (6 * binom (n-7) 6) / binom n 7)
  : n = 48 := by
  sorry

end find_n_l2079_207914


namespace min_cubes_l2079_207932

-- Define the conditions as properties
structure FigureViews :=
  (front_view : ℕ)
  (side_view : ℕ)
  (top_view : ℕ)
  (adjacency_requirement : Bool)

-- Define the given views
def given_views : FigureViews := {
  front_view := 3,  -- as described: 2 cubes at bottom + 1 on top
  side_view := 3,   -- same as front view
  top_view := 3,    -- L-shape consists of 3 cubes
  adjacency_requirement := true
}

-- The theorem to state that the minimum number of cubes is 3
theorem min_cubes (views : FigureViews) : views.front_view = 3 ∧ views.side_view = 3 ∧ views.top_view = 3 ∧ views.adjacency_requirement = true → ∃ n, n = 3 :=
by {
  sorry
}

end min_cubes_l2079_207932


namespace decreasing_function_condition_l2079_207966

theorem decreasing_function_condition (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ≤ 3 → deriv f x ≤ 0) ↔ (m ≥ 1) :=
by 
  sorry

end decreasing_function_condition_l2079_207966


namespace value_of_expression_l2079_207944

theorem value_of_expression : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end value_of_expression_l2079_207944


namespace inv_composition_l2079_207961

theorem inv_composition (f g : ℝ → ℝ) (hf : Function.Bijective f) (hg : Function.Bijective g) (h : ∀ x, f⁻¹ (g x) = 2 * x - 4) : 
  g⁻¹ (f (-3)) = 1 / 2 :=
by
  sorry

end inv_composition_l2079_207961


namespace min_a4_in_arithmetic_sequence_l2079_207920

noncomputable def arithmetic_sequence_min_a4 (a1 d : ℝ) 
(S4 : ℝ := 4 * a1 + 6 * d)
(S5 : ℝ := 5 * a1 + 10 * d)
(a4 : ℝ := a1 + 3 * d) : Prop :=
  S4 ≤ 4 ∧ S5 ≥ 15 → a4 = 7

theorem min_a4_in_arithmetic_sequence (a1 d : ℝ) (h1 : 4 * a1 + 6 * d ≤ 4) 
(h2 : 5 * a1 + 10 * d ≥ 15) : 
arithmetic_sequence_min_a4 a1 d := 
by {
  sorry -- Proof is omitted
}

end min_a4_in_arithmetic_sequence_l2079_207920


namespace Taehyung_age_l2079_207964

variable (T U : Nat)

-- Condition 1: Taehyung is 17 years younger than his uncle
def condition1 : Prop := U = T + 17

-- Condition 2: Four years later, the sum of their ages is 43
def condition2 : Prop := (T + 4) + (U + 4) = 43

-- The goal is to prove that Taehyung's current age is 9, given the conditions above
theorem Taehyung_age : condition1 T U ∧ condition2 T U → T = 9 := by
  sorry

end Taehyung_age_l2079_207964


namespace sum_of_abc_l2079_207997

theorem sum_of_abc (a b c : ℕ) (h : a + b + c = 12) 
  (area_ratio : ℝ) (side_length_ratio : ℝ) 
  (ha : area_ratio = 50 / 98) 
  (hb : side_length_ratio = (Real.sqrt 50) / (Real.sqrt 98))
  (hc : side_length_ratio = (a * (Real.sqrt b)) / c) :
  a + b + c = 12 :=
by
  sorry

end sum_of_abc_l2079_207997


namespace thirteen_coins_value_l2079_207998

theorem thirteen_coins_value :
  ∃ (p n d q : ℕ), p + n + d + q = 13 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 141 ∧ 
                   2 ≤ p ∧ 2 ≤ n ∧ 2 ≤ d ∧ 2 ≤ q ∧ 
                   d = 3 :=
  sorry

end thirteen_coins_value_l2079_207998


namespace quadratic_inequality_l2079_207971

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_inequality (h : f b c (-1) = f b c 3) : f b c 1 < c ∧ c < f b c 3 :=
by
  sorry

end quadratic_inequality_l2079_207971


namespace eq_is_quadratic_iff_m_zero_l2079_207913

theorem eq_is_quadratic_iff_m_zero (m : ℝ) : (|m| + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 := by
  sorry

end eq_is_quadratic_iff_m_zero_l2079_207913


namespace hyperbola_slope_product_l2079_207954

open Real

theorem hyperbola_slope_product
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h : ∀ {x y : ℝ}, x ≠ 0 → (x^2 / a^2 - y^2 / b^2 = 1) → 
    ∀ {k1 k2 : ℝ}, (x = 0 ∨ y = 0) → (k1 * k2 = ((b^2) / (a^2)))) :
  (b^2 / a^2 = 3) :=
by 
  sorry

end hyperbola_slope_product_l2079_207954


namespace quadratic_unbounded_above_l2079_207907

theorem quadratic_unbounded_above : ∀ (x y : ℝ), ∃ M : ℝ, ∀ z : ℝ, M < (2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z) :=
by
  intro x y
  use 1000 -- Example to denote that for any point greater than 1000
  intro z
  have h1 : 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z ≥ 2 * 0^2 + 4 * 0 * y + 5 * y^2 + 8 * 0 - 6 * y + z := by sorry
  sorry

end quadratic_unbounded_above_l2079_207907


namespace gcd_of_360_and_150_is_30_l2079_207994

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l2079_207994


namespace parabola_standard_eq_l2079_207973

theorem parabola_standard_eq (h : ∃ (x y : ℝ), x - 2 * y - 4 = 0 ∧ (
                         (y = 0 ∧ x = 4 ∧ y^2 = 16 * x) ∨ 
                         (x = 0 ∧ y = -2 ∧ x^2 = -8 * y))
                         ) :
                         (y^2 = 16 * x) ∨ (x^2 = -8 * y) :=
by 
  sorry

end parabola_standard_eq_l2079_207973


namespace find_c_for_two_zeros_l2079_207903

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c_for_two_zeros (c : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) ↔ c = -2 ∨ c = 2 :=
sorry

end find_c_for_two_zeros_l2079_207903


namespace min_rectangle_area_l2079_207960

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end min_rectangle_area_l2079_207960


namespace digitalEarth_correct_l2079_207916

-- Define the possible descriptions of "Digital Earth"
inductive DigitalEarthDescription
| optionA : DigitalEarthDescription
| optionB : DigitalEarthDescription
| optionC : DigitalEarthDescription
| optionD : DigitalEarthDescription

-- Define the correct description according to the solution
def correctDescription : DigitalEarthDescription := DigitalEarthDescription.optionB

-- Define the theorem to prove the equivalence
theorem digitalEarth_correct :
  correctDescription = DigitalEarthDescription.optionB :=
sorry

end digitalEarth_correct_l2079_207916


namespace trajectory_of_center_of_P_l2079_207957

-- Define circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the conditions for the moving circle P
def externally_tangent (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = (1 + r)^2
def internally_tangent (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = (5 - r)^2

-- The statement we need to prove
theorem trajectory_of_center_of_P : ∃ (x y : ℝ), 
  (externally_tangent x y r) ∧ (internally_tangent x y r) →
  (x^2 / 9 + y^2 / 8 = 1) :=
by
  -- Proof will go here
  sorry

end trajectory_of_center_of_P_l2079_207957


namespace partA_partB_partC_partD_l2079_207942

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l2079_207942


namespace range_of_m_l2079_207963

def has_solution_in_interval (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), x^2 - 2 * x - 1 + m ≤ 0 

theorem range_of_m (m : ℝ) : has_solution_in_interval m ↔ m ≤ 2 := by 
  sorry

end range_of_m_l2079_207963


namespace time_for_trains_to_clear_l2079_207984

noncomputable def train_length_1 : ℕ := 120
noncomputable def train_length_2 : ℕ := 320
noncomputable def train_speed_1_kmph : ℚ := 42
noncomputable def train_speed_2_kmph : ℚ := 30

noncomputable def kmph_to_mps (speed: ℚ) : ℚ := (5/18) * speed

noncomputable def train_speed_1_mps : ℚ := kmph_to_mps train_speed_1_kmph
noncomputable def train_speed_2_mps : ℚ := kmph_to_mps train_speed_2_kmph

noncomputable def total_length : ℕ := train_length_1 + train_length_2
noncomputable def relative_speed : ℚ := train_speed_1_mps + train_speed_2_mps

noncomputable def collision_time : ℚ := total_length / relative_speed

theorem time_for_trains_to_clear : collision_time = 22 := by
  sorry

end time_for_trains_to_clear_l2079_207984


namespace original_number_of_people_is_fifteen_l2079_207901

/-!
The average age of all the people who gathered at a family celebration was equal to the number of attendees. 
Aunt Beta, who was 29 years old, soon excused herself and left. 
Even after Aunt Beta left, the average age of all the remaining attendees was still equal to their number.
Prove that the original number of people at the celebration is 15.
-/

theorem original_number_of_people_is_fifteen
  (n : ℕ)
  (s : ℕ)
  (h1 : s = n^2)
  (h2 : s - 29 = (n - 1)^2):
  n = 15 :=
by
  sorry

end original_number_of_people_is_fifteen_l2079_207901


namespace negation_proposition_l2079_207996

open Classical

variable (x : ℝ)

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l2079_207996


namespace batsman_average_after_17th_inning_l2079_207924

theorem batsman_average_after_17th_inning 
    (A : ℕ) 
    (hA : A = 15) 
    (runs_17th_inning : ℕ)
    (increase_in_average : ℕ) 
    (hscores : runs_17th_inning = 100)
    (hincrease : increase_in_average = 5) :
    (A + increase_in_average = 20) :=
by
  sorry

end batsman_average_after_17th_inning_l2079_207924


namespace tan_x_plus_pi_over_4_l2079_207933

theorem tan_x_plus_pi_over_4 (x : ℝ) (hx : Real.tan x = 2) : Real.tan (x + Real.pi / 4) = -3 :=
by
  sorry

end tan_x_plus_pi_over_4_l2079_207933


namespace rotor_permutations_l2079_207900

-- Define the factorial function for convenience
def fact : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * fact n

-- The main statement to prove
theorem rotor_permutations : (fact 5) / ((fact 2) * (fact 2)) = 30 := by
  sorry

end rotor_permutations_l2079_207900


namespace pq_false_implies_m_range_l2079_207919

def p : Prop := ∀ x : ℝ, abs x + x ≥ 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem pq_false_implies_m_range (m : ℝ) :
  (¬ (p ∧ q m)) → -2 < m ∧ m < 2 :=
by
  sorry

end pq_false_implies_m_range_l2079_207919


namespace investment_value_change_l2079_207953

theorem investment_value_change (k m : ℝ) : 
  let increaseFactor := 1 + k / 100
  let decreaseFactor := 1 - m / 100 
  let overallFactor := increaseFactor * decreaseFactor 
  let changeFactor := overallFactor - 1
  let percentageChange := changeFactor * 100 
  percentageChange = k - m - (k * m) / 100 := 
by 
  sorry

end investment_value_change_l2079_207953


namespace right_triangle_5_12_13_l2079_207981

theorem right_triangle_5_12_13 (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) : a^2 + b^2 = c^2 := 
by 
   sorry

end right_triangle_5_12_13_l2079_207981


namespace left_handed_classical_music_lovers_l2079_207965

-- Define the conditions
variables (total_people left_handed classical_music right_handed_dislike : ℕ)
variables (x : ℕ) -- x will represent the number of left-handed classical music lovers

-- State the assumptions based on conditions
axiom h1 : total_people = 30
axiom h2 : left_handed = 12
axiom h3 : classical_music = 20
axiom h4 : right_handed_dislike = 3
axiom h5 : 30 = x + (12 - x) + (20 - x) + 3

-- State the theorem to prove
theorem left_handed_classical_music_lovers : x = 5 :=
by {
  -- Skip the proof using sorry
  sorry
}

end left_handed_classical_music_lovers_l2079_207965


namespace find_number_l2079_207952

theorem find_number
  (n : ℕ)
  (h : 80641 * n = 806006795) :
  n = 9995 :=
by 
  sorry

end find_number_l2079_207952


namespace distance_to_airport_l2079_207950

theorem distance_to_airport
  (t : ℝ)
  (d : ℝ)
  (h1 : 45 * (t + 1) + 20 = d)
  (h2 : d - 65 = 65 * (t - 1))
  : d = 390 := by
  sorry

end distance_to_airport_l2079_207950


namespace digit_b_divisible_by_7_l2079_207990

theorem digit_b_divisible_by_7 (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) 
  (hdiv : (4000 + 110 * B + 3) % 7 = 0) : B = 0 :=
by
  sorry

end digit_b_divisible_by_7_l2079_207990


namespace shaded_rectangle_area_l2079_207956

def area_polygon : ℝ := 2016
def sides_polygon : ℝ := 18
def segments_persh : ℝ := 4

theorem shaded_rectangle_area :
  (area_polygon / sides_polygon) * segments_persh = 448 := 
sorry

end shaded_rectangle_area_l2079_207956


namespace circle_center_coordinates_l2079_207976

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0 ↔ (x - h)^2 + (y - k)^2 = 13) ∧ h = 2 ∧ k = -3 :=
sorry

end circle_center_coordinates_l2079_207976


namespace ball_more_than_bat_l2079_207955

theorem ball_more_than_bat :
  ∃ x y : ℕ, (2 * x + 3 * y = 1300) ∧ (3 * x + 2 * y = 1200) ∧ (y - x = 100) :=
by
  sorry

end ball_more_than_bat_l2079_207955


namespace average_wage_per_day_l2079_207943

variable (numMaleWorkers : ℕ) (wageMale : ℕ) (numFemaleWorkers : ℕ) (wageFemale : ℕ) (numChildWorkers : ℕ) (wageChild : ℕ)

theorem average_wage_per_day :
  numMaleWorkers = 20 →
  wageMale = 35 →
  numFemaleWorkers = 15 →
  wageFemale = 20 →
  numChildWorkers = 5 →
  wageChild = 8 →
  (20 * 35 + 15 * 20 + 5 * 8) / (20 + 15 + 5) = 26 :=
by
  intros
  -- Proof would follow here
  sorry

end average_wage_per_day_l2079_207943


namespace evaluate_expression_correct_l2079_207946

def evaluate_expression : ℚ :=
  let a := 17
  let b := 19
  let c := 23
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)
  numerator / denominator

theorem evaluate_expression_correct : evaluate_expression = 59 := 
by {
  -- proof skipped
  sorry
}

end evaluate_expression_correct_l2079_207946


namespace total_rods_required_l2079_207928

-- Define the number of rods needed per unit for each type
def rods_per_sheet_A : ℕ := 10
def rods_per_sheet_B : ℕ := 8
def rods_per_sheet_C : ℕ := 12
def rods_per_beam_A : ℕ := 6
def rods_per_beam_B : ℕ := 4
def rods_per_beam_C : ℕ := 5

-- Define the composition per panel
def sheets_A_per_panel : ℕ := 2
def sheets_B_per_panel : ℕ := 1
def beams_C_per_panel : ℕ := 2

-- Define the number of panels
def num_panels : ℕ := 10

-- Prove the total number of metal rods required for the entire fence
theorem total_rods_required : 
  (sheets_A_per_panel * rods_per_sheet_A + 
   sheets_B_per_panel * rods_per_sheet_B +
   beams_C_per_panel * rods_per_beam_C) * num_panels = 380 :=
by 
  sorry

end total_rods_required_l2079_207928


namespace topaz_sapphire_value_equal_l2079_207934

/-
  Problem statement: Given the following conditions:
  1. One sapphire and two topazes are three times more valuable than an emerald: S + 2T = 3E
  2. Seven sapphires and one topaz are eight times more valuable than an emerald: 7S + T = 8E
  
  Prove that the value of one topaz is equal to the value of one sapphire (T = S).
-/

theorem topaz_sapphire_value_equal
  (S T E : ℝ) 
  (h1 : S + 2 * T = 3 * E) 
  (h2 : 7 * S + T = 8 * E) :
  T = S := 
  sorry

end topaz_sapphire_value_equal_l2079_207934


namespace sum_of_2x2_table_is_zero_l2079_207968

theorem sum_of_2x2_table_is_zero {a b c d : ℤ} 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_eq : a + b = c + d)
  (prod_eq : a * c = b * d) :
  a + b + c + d = 0 :=
by sorry

end sum_of_2x2_table_is_zero_l2079_207968


namespace tanvi_rank_among_girls_correct_l2079_207912

def Vikas_rank : ℕ := 9
def Tanvi_rank : ℕ := 17
def girls_between : ℕ := 2
def Tanvi_rank_among_girls : ℕ := 8

theorem tanvi_rank_among_girls_correct (Vikas_rank Tanvi_rank girls_between Tanvi_rank_among_girls : ℕ) 
  (h1 : Vikas_rank = 9) 
  (h2 : Tanvi_rank = 17) 
  (h3 : girls_between = 2)
  (h4 : Tanvi_rank_among_girls = 8): 
  Tanvi_rank_among_girls = 8 := by
  sorry

end tanvi_rank_among_girls_correct_l2079_207912


namespace complex_point_quadrant_l2079_207958

def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_point_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  inFourthQuadrant z :=
by
  sorry

end complex_point_quadrant_l2079_207958


namespace find_pairs_of_square_numbers_l2079_207982

theorem find_pairs_of_square_numbers (a b k : ℕ) (hk : k ≥ 2) 
  (h_eq : (a * a + b * b) = k * k * (a * b + 1)) : 
  (a = k ∧ b = k * k * k) ∨ (b = k ∧ a = k * k * k) :=
by
  sorry

end find_pairs_of_square_numbers_l2079_207982


namespace positive_integer_solution_l2079_207972

theorem positive_integer_solution (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (1 / (x * x : ℝ) + 1 / (y * y : ℝ) + 1 / (z * z : ℝ) + 1 / (t * t : ℝ) = 1) ↔ (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by
  sorry

end positive_integer_solution_l2079_207972


namespace average_gas_mileage_round_trip_l2079_207937

theorem average_gas_mileage_round_trip
  (d : ℝ) (ms mr : ℝ)
  (h1 : d = 150)
  (h2 : ms = 35)
  (h3 : mr = 15) :
  (2 * d) / ((d / ms) + (d / mr)) = 21 :=
by
  sorry

end average_gas_mileage_round_trip_l2079_207937


namespace smallest_y_value_l2079_207940

theorem smallest_y_value : ∃ y : ℝ, 2 * y ^ 2 + 7 * y + 3 = 5 ∧ (∀ y' : ℝ, 2 * y' ^ 2 + 7 * y' + 3 = 5 → y ≤ y') := sorry

end smallest_y_value_l2079_207940


namespace smallest_integer_CC6_DD8_l2079_207947

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l2079_207947


namespace incorrect_statement_C_l2079_207983

theorem incorrect_statement_C : 
  (∀ x : ℝ, |x| = x → x = 0 ∨ x = 1) ↔ False :=
by
  -- Proof goes here
  sorry

end incorrect_statement_C_l2079_207983


namespace line_through_point_and_area_l2079_207910

theorem line_through_point_and_area (a b : ℝ) (x y : ℝ) 
  (hx : x = -2) (hy : y = 2) 
  (h_area : 1/2 * |a * b| = 1): 
  (2 * x + y + 2 = 0 ∨ x + 2 * y - 2 = 0) :=
  sorry

end line_through_point_and_area_l2079_207910


namespace time_to_fill_pond_l2079_207988

noncomputable def pond_capacity : ℝ := 200
noncomputable def normal_pump_rate : ℝ := 6
noncomputable def restriction_factor : ℝ := 2 / 3
noncomputable def restricted_pump_rate : ℝ := restriction_factor * normal_pump_rate

theorem time_to_fill_pond : pond_capacity / restricted_pump_rate = 50 := 
by 
  -- This is where the proof would go
  sorry

end time_to_fill_pond_l2079_207988


namespace number_of_married_men_at_least_11_l2079_207922

-- Definitions based only on conditions from a)
def total_men := 100
def men_with_tv := 75
def men_with_radio := 85
def men_with_ac := 70
def married_with_tv_radio_ac := 11

-- Theorem that needs to be proven based on the conditions
theorem number_of_married_men_at_least_11 : total_men ≥ married_with_tv_radio_ac :=
by
  sorry

end number_of_married_men_at_least_11_l2079_207922


namespace real_y_iff_x_ranges_l2079_207949

-- Definitions for conditions
variable (x y : ℝ)

-- Condition for the equation
def equation := 9 * y^2 - 6 * x * y + 2 * x + 7 = 0

-- Theorem statement
theorem real_y_iff_x_ranges :
  (∃ y : ℝ, equation x y) ↔ (x ≤ -2 ∨ x ≥ 7) :=
sorry

end real_y_iff_x_ranges_l2079_207949


namespace cos_value_l2079_207975

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 9 := by
  sorry

end cos_value_l2079_207975


namespace arithmetic_sequence_sum_l2079_207918

theorem arithmetic_sequence_sum :
  ∃ x y z d : ℝ, 
  d = (31 - 4) / 5 ∧ 
  x = 4 + d ∧ 
  y = x + d ∧ 
  z = 16 + d ∧ 
  (x + y + z) = 45.6 :=
by
  sorry

end arithmetic_sequence_sum_l2079_207918


namespace range_of_m_length_of_chord_l2079_207986

-- Definition of Circle C
def CircleC (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0

-- Definition of Circle D
def CircleD (x y : ℝ) := (x + 3)^2 + (y + 1)^2 = 16

-- Definition of Line l
def LineL (x y : ℝ) := x + 2*y - 4 = 0

-- Problem 1: Prove range of values for m
theorem range_of_m (m : ℝ) : (∀ x y, CircleC x y m) → m < 5 := by
  sorry

-- Problem 2: Prove length of chord MN
theorem length_of_chord (x y : ℝ) :
  CircleC x y 4 ∧ CircleD x y ∧ LineL x y →
  (∃ MN, MN = (4*Real.sqrt 5) / 5) := by
    sorry

end range_of_m_length_of_chord_l2079_207986


namespace point_not_in_first_quadrant_l2079_207936

theorem point_not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) : ¬ (m > 0 ∧ n > 0) :=
sorry

end point_not_in_first_quadrant_l2079_207936


namespace ratio_of_two_numbers_l2079_207923

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a > b) (h3 : a > 0) (h4 : b > 0) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_two_numbers_l2079_207923


namespace geometric_sequence_ratio_l2079_207987

theorem geometric_sequence_ratio (a b c q : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ b + c - a = x * q ∧ c + a - b = x * q^2 ∧ a + b - c = x * q^3 ∧ a + b + c = x) →
  q^3 + q^2 + q = 1 :=
by
  sorry

end geometric_sequence_ratio_l2079_207987


namespace simplify_expression_l2079_207959

theorem simplify_expression (y : ℝ) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^2) = (9 / 2) * y^3 :=
by sorry

end simplify_expression_l2079_207959


namespace expand_product_correct_l2079_207917

noncomputable def expand_product (x : ℝ) : ℝ :=
  3 * (x + 4) * (x + 5)

theorem expand_product_correct (x : ℝ) :
  expand_product x = 3 * x^2 + 27 * x + 60 :=
by
  unfold expand_product
  sorry

end expand_product_correct_l2079_207917


namespace number_of_black_squares_in_58th_row_l2079_207979

theorem number_of_black_squares_in_58th_row :
  let pattern := [1, 0, 0] -- pattern where 1 represents a black square
  let n := 58
  let total_squares := 2 * n - 1 -- total squares in the 58th row
  let black_count := total_squares / 3 -- number of black squares in the repeating pattern
  black_count = 38 :=
by
  let pattern := [1, 0, 0]
  let n := 58
  let total_squares := 2 * n - 1
  let black_count := total_squares / 3
  have black_count_eq_38 : 38 = (115 / 3) := by sorry
  exact black_count_eq_38.symm

end number_of_black_squares_in_58th_row_l2079_207979


namespace sum_of_terms_l2079_207945

theorem sum_of_terms (a d : ℕ) (h1 : a + d < a + 2 * d)
  (h2 : (a + d) * (a + 20) = (a + 2 * d) ^ 2)
  (h3 : a + 20 - a = 20) :
  a + (a + d) + (a + 2 * d) + (a + 20) = 46 :=
by
  sorry

end sum_of_terms_l2079_207945


namespace number_of_valid_pairs_l2079_207989

theorem number_of_valid_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * (x : ℝ) + b * (y : ℝ) = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) →
  ∃! pairs_count : ℕ, pairs_count = 72 :=
by
  sorry

end number_of_valid_pairs_l2079_207989


namespace Q_finishes_in_6_hours_l2079_207931

def Q_time_to_finish_job (T_Q : ℝ) : Prop :=
  let P_rate := 1 / 3
  let Q_rate := 1 / T_Q
  let work_together_2hr := 2 * (P_rate + Q_rate)
  let P_alone_work_40min := (2 / 3) * P_rate
  work_together_2hr + P_alone_work_40min = 1

theorem Q_finishes_in_6_hours : Q_time_to_finish_job 6 :=
  sorry -- Proof skipped

end Q_finishes_in_6_hours_l2079_207931


namespace min_draw_to_ensure_one_red_l2079_207970

theorem min_draw_to_ensure_one_red (b y r : ℕ) (h1 : b + y + r = 20) (h2 : b = y / 6) (h3 : r < y) : 
  ∃ n : ℕ, n = 15 ∧ ∀ d : ℕ, d < 15 → ∀ drawn : Finset (ℕ × ℕ × ℕ), drawn.card = d → ∃ card ∈ drawn, card.2 = r := 
sorry

end min_draw_to_ensure_one_red_l2079_207970


namespace scientific_notation_448000_l2079_207921

theorem scientific_notation_448000 :
  ∃ a n, (448000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 :=
by
  sorry

end scientific_notation_448000_l2079_207921
