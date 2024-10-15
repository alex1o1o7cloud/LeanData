import Mathlib

namespace NUMINAMATH_GPT_sum_of_solutions_of_quadratic_eq_l2130_213047

theorem sum_of_solutions_of_quadratic_eq :
  (∀ x : ℝ, 5 * x^2 - 3 * x - 2 = 0) → (∀ a b : ℝ, a = 5 ∧ b = -3 → -b / a = 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_of_quadratic_eq_l2130_213047


namespace NUMINAMATH_GPT_vector_ab_l2130_213091

theorem vector_ab
  (A B : ℝ × ℝ)
  (hA : A = (1, -1))
  (hB : B = (1, 2)) :
  (B.1 - A.1, B.2 - A.2) = (0, 3) :=
by
  sorry

end NUMINAMATH_GPT_vector_ab_l2130_213091


namespace NUMINAMATH_GPT_probability_only_one_l2130_213098

-- Define the probabilities
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complement probabilities
def not_P (P : ℚ) : ℚ := 1 - P
def P_not_A := not_P P_A
def P_not_B := not_P P_B
def P_not_C := not_P P_C

-- Expressions for probabilities where only one student solves the problem
def only_A_solves : ℚ := P_A * P_not_B * P_not_C
def only_B_solves : ℚ := P_B * P_not_A * P_not_C
def only_C_solves : ℚ := P_C * P_not_A * P_not_B

-- Total probability that only one student solves the problem
def P_only_one : ℚ := only_A_solves + only_B_solves + only_C_solves

-- The theorem to prove that the total probability matches
theorem probability_only_one : P_only_one = 11 / 24 := by
  sorry

end NUMINAMATH_GPT_probability_only_one_l2130_213098


namespace NUMINAMATH_GPT_new_tax_rate_is_30_percent_l2130_213007

theorem new_tax_rate_is_30_percent
  (original_rate : ℝ)
  (annual_income : ℝ)
  (tax_saving : ℝ)
  (h1 : original_rate = 0.45)
  (h2 : annual_income = 48000)
  (h3 : tax_saving = 7200) :
  (100 * (original_rate * annual_income - tax_saving) / annual_income) = 30 := 
sorry

end NUMINAMATH_GPT_new_tax_rate_is_30_percent_l2130_213007


namespace NUMINAMATH_GPT_sum_first_seven_terms_geometric_seq_l2130_213082

theorem sum_first_seven_terms_geometric_seq :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  S_7 = 127 / 192 := 
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  have h : S_7 = 127 / 192 := sorry
  exact h

end NUMINAMATH_GPT_sum_first_seven_terms_geometric_seq_l2130_213082


namespace NUMINAMATH_GPT_contradiction_divisible_by_2_l2130_213069

open Nat

theorem contradiction_divisible_by_2 (a b : ℕ) (h : (a * b) % 2 = 0) : a % 2 = 0 ∨ b % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_contradiction_divisible_by_2_l2130_213069


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_XYZ_l2130_213018

noncomputable def radius_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := area / s
  r

theorem radius_of_inscribed_circle_XYZ :
  radius_of_inscribed_circle 26 15 17 = 2 * Real.sqrt 42 / 29 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_XYZ_l2130_213018


namespace NUMINAMATH_GPT_john_change_received_is_7_l2130_213006

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end NUMINAMATH_GPT_john_change_received_is_7_l2130_213006


namespace NUMINAMATH_GPT_min_sum_one_over_xy_l2130_213015

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_min_sum_one_over_xy_l2130_213015


namespace NUMINAMATH_GPT_boys_in_first_group_l2130_213021

theorem boys_in_first_group (x : ℕ) (h₁ : 5040 = 360 * x) : x = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_boys_in_first_group_l2130_213021


namespace NUMINAMATH_GPT_max_sum_of_factors_l2130_213011

theorem max_sum_of_factors (A B C : ℕ) (h1 : A * B * C = 2310) (h2 : A ≠ B) (h3 : B ≠ C) (h4 : A ≠ C) (h5 : 0 < A) (h6 : 0 < B) (h7 : 0 < C) : 
  A + B + C ≤ 42 := 
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l2130_213011


namespace NUMINAMATH_GPT_A_oplus_B_eq_l2130_213026

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def symm_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M
def A : Set ℝ := {y | ∃ x:ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x:ℝ, y = -(x-1)^2 + 2}

theorem A_oplus_B_eq : symm_diff A B = {y | y ≤ 0} ∪ {y | y > 2} := by {
  sorry
}

end NUMINAMATH_GPT_A_oplus_B_eq_l2130_213026


namespace NUMINAMATH_GPT_unpainted_cubes_count_l2130_213002

theorem unpainted_cubes_count :
  let L := 6
  let W := 6
  let H := 3
  (L - 2) * (W - 2) * (H - 2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_unpainted_cubes_count_l2130_213002


namespace NUMINAMATH_GPT_find_d_l2130_213085

theorem find_d (m a b d : ℕ) 
(hm : 0 < m) 
(ha : m^2 < a ∧ a < m^2 + m) 
(hb : m^2 < b ∧ b < m^2 + m) 
(hab : a ≠ b)
(hd : m^2 < d ∧ d < m^2 + m ∧ d ∣ (a * b)) : 
d = a ∨ d = b :=
sorry

end NUMINAMATH_GPT_find_d_l2130_213085


namespace NUMINAMATH_GPT_dinosaur_dolls_distribution_l2130_213081

-- Defining the conditions
def num_dolls : ℕ := 5
def num_friends : ℕ := 2

-- Lean theorem statement
theorem dinosaur_dolls_distribution :
  (num_dolls * (num_dolls - 1) = 20) :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_dinosaur_dolls_distribution_l2130_213081


namespace NUMINAMATH_GPT_chocolate_bars_per_small_box_l2130_213088

theorem chocolate_bars_per_small_box (total_chocolate_bars small_boxes : ℕ) 
  (h1 : total_chocolate_bars = 442) 
  (h2 : small_boxes = 17) : 
  total_chocolate_bars / small_boxes = 26 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_per_small_box_l2130_213088


namespace NUMINAMATH_GPT_part_one_part_two_l2130_213068

noncomputable def f (a x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≥ -1/Real.exp 1) : a = 0 := sorry

theorem part_two {a x : ℝ} (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := sorry

end NUMINAMATH_GPT_part_one_part_two_l2130_213068


namespace NUMINAMATH_GPT_parallel_a_b_projection_a_onto_b_l2130_213075

noncomputable section

open Real

def a : ℝ × ℝ := (sqrt 3, 1)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem parallel_a_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_parallel : (a.1 / a.2) = (b θ).1 / (b θ).2) : θ = π / 6 := sorry

theorem projection_a_onto_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_proj : (sqrt 3 * cos θ + sin θ) = -sqrt 3) : b θ = (-1, 0) := sorry

end NUMINAMATH_GPT_parallel_a_b_projection_a_onto_b_l2130_213075


namespace NUMINAMATH_GPT_ratio_of_lemons_l2130_213005

theorem ratio_of_lemons :
  ∃ (L J E I : ℕ), 
  L = 5 ∧ 
  J = L + 6 ∧ 
  J = E / 3 ∧ 
  E = I / 2 ∧ 
  L + J + E + I = 115 ∧ 
  J / E = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_lemons_l2130_213005


namespace NUMINAMATH_GPT_greatest_integer_radius_of_circle_l2130_213078

theorem greatest_integer_radius_of_circle (r : ℕ) (A : ℝ) (hA : A < 80 * Real.pi) :
  r <= 8 ∧ r * r < 80 :=
sorry

end NUMINAMATH_GPT_greatest_integer_radius_of_circle_l2130_213078


namespace NUMINAMATH_GPT_trigonometric_identity_l2130_213071

noncomputable def π := Real.pi
noncomputable def tan (x : ℝ) := Real.sin x / Real.cos x

theorem trigonometric_identity (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : tan α = (1 + Real.sin β) / Real.cos β) :
  2 * α - β = π / 2 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l2130_213071


namespace NUMINAMATH_GPT_complex_number_equality_l2130_213027

theorem complex_number_equality (a b : ℂ) : a - b = 0 ↔ a = b := sorry

end NUMINAMATH_GPT_complex_number_equality_l2130_213027


namespace NUMINAMATH_GPT_true_statement_l2130_213097

variables {Plane Line : Type}
variables (α β γ : Plane) (a b m n : Line)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect_line (p q : Plane) : Line := sorry

-- Given conditions for the problem
variables (h1 : (α ≠ β)) (h2 : (parallel α β))
variables (h3 : (intersect_line α γ = a)) (h4 : (intersect_line β γ = b))

-- Statement verifying the true condition based on the above givens
theorem true_statement : parallel a b :=
by sorry

end NUMINAMATH_GPT_true_statement_l2130_213097


namespace NUMINAMATH_GPT_soldier_score_9_points_l2130_213037

-- Define the conditions and expected result in Lean 4
theorem soldier_score_9_points (shots : List ℕ) :
  shots.length = 10 ∧
  (∀ shot ∈ shots, shot = 7 ∨ shot = 8 ∨ shot = 9 ∨ shot = 10) ∧
  shots.count 10 = 4 ∧
  shots.sum = 90 →
  shots.count 9 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_soldier_score_9_points_l2130_213037


namespace NUMINAMATH_GPT_weight_of_replaced_sailor_l2130_213030

theorem weight_of_replaced_sailor (avg_increase : ℝ) (total_sailors : ℝ) (new_sailor_weight : ℝ) : 
  avg_increase = 1 ∧ total_sailors = 8 ∧ new_sailor_weight = 64 → 
  ∃ W, W = 56 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_weight_of_replaced_sailor_l2130_213030


namespace NUMINAMATH_GPT_shoot_down_probability_l2130_213070

-- Define the probabilities
def P_hit_nose := 0.2
def P_hit_middle := 0.4
def P_hit_tail := 0.1
def P_miss := 0.3

-- Define the condition: probability of shooting down the plane with at most 2 shots
def condition := (P_hit_tail + (P_hit_nose * P_hit_nose) + (P_miss * P_hit_tail))

-- Proving the probability matches the required value
theorem shoot_down_probability : condition = 0.23 :=
by
  sorry

end NUMINAMATH_GPT_shoot_down_probability_l2130_213070


namespace NUMINAMATH_GPT_problem_l2130_213032

theorem problem (m n : ℚ) (h : m - n = -2/3) : 7 - 3 * m + 3 * n = 9 := 
by {
  -- Place a sorry here as we do not provide the proof 
  sorry
}

end NUMINAMATH_GPT_problem_l2130_213032


namespace NUMINAMATH_GPT_adults_collectively_ate_l2130_213057

theorem adults_collectively_ate (A : ℕ) (C : ℕ) (total_cookies : ℕ) (share : ℝ) (each_child_gets : ℕ)
  (hC : C = 4) (hTotal : total_cookies = 120) (hShare : share = 1/3) (hEachChild : each_child_gets = 20)
  (children_gets : ℕ) (hChildrenGets : children_gets = C * each_child_gets) :
  children_gets = (2/3 : ℝ) * total_cookies → (share : ℝ) * total_cookies = 40 :=
by
  -- Placeholder for simplified proof
  sorry

end NUMINAMATH_GPT_adults_collectively_ate_l2130_213057


namespace NUMINAMATH_GPT_odd_function_increasing_function_l2130_213004

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 0.5

theorem odd_function (x : ℝ) : f (-x) = -f (x) :=
  by sorry

theorem increasing_function : ∀ x y : ℝ, x < y → f x < f y :=
  by sorry

end NUMINAMATH_GPT_odd_function_increasing_function_l2130_213004


namespace NUMINAMATH_GPT_initial_concentration_is_27_l2130_213061

-- Define given conditions
variables (m m_c : ℝ) -- initial mass of solution and salt
variables (x : ℝ) -- initial percentage concentration of salt
variables (h1 : m_c = (x / 100) * m) -- initial concentration definition
variables (h2 : m > 0) (h3 : x > 0) -- non-zero positive mass and concentration

theorem initial_concentration_is_27 (h_evaporated : (m / 5) * 2 * (x / 100) = m_c) 
  (h_new_concentration : (x + 3) = (m_c * 100) / (9 * m / 10)) 
  : x = 27 :=
by
  sorry

end NUMINAMATH_GPT_initial_concentration_is_27_l2130_213061


namespace NUMINAMATH_GPT_pets_percentage_of_cats_l2130_213049

theorem pets_percentage_of_cats :
  ∀ (total_pets dogs as_percentage bunnies cats_percentage : ℕ),
    total_pets = 36 →
    dogs = total_pets * as_percentage / 100 →
    as_percentage = 25 →
    bunnies = 9 →
    cats_percentage = (total_pets - (dogs + bunnies)) * 100 / total_pets →
    cats_percentage = 50 :=
by
  intros total_pets dogs as_percentage bunnies cats_percentage
  sorry

end NUMINAMATH_GPT_pets_percentage_of_cats_l2130_213049


namespace NUMINAMATH_GPT_javier_fraction_to_anna_zero_l2130_213096

-- Variables
variable (l : ℕ) -- Lee's initial sticker count
variable (j : ℕ) -- Javier's initial sticker count
variable (a : ℕ) -- Anna's initial sticker count

-- Initial conditions
def conditions (l j a : ℕ) : Prop :=
  j = 4 * a ∧ a = 3 * l

-- Javier's final stickers count
def final_javier_stickers (ja : ℕ) (j : ℕ) : ℕ :=
  ja

-- Anna's final stickers count (af = final Anna's stickers)
def final_anna_stickers (af : ℕ) : ℕ :=
  af

-- Lee's final stickers count (lf = final Lee's stickers)
def final_lee_stickers (lf : ℕ) : ℕ :=
  lf

-- Final distribution requirements
def final_distribution (ja af lf : ℕ) : Prop :=
  ja = 2 * af ∧ ja = 3 * lf

-- Correct answer, fraction of stickers given to Anna
def fraction_given_to_anna (j ja : ℕ) : ℚ :=
  ((j - ja) : ℚ) / (j : ℚ)

-- Lean theorem statement to prove
theorem javier_fraction_to_anna_zero
  (l j a ja af lf : ℕ)
  (h_cond : conditions l j a)
  (h_final : final_distribution ja af lf) :
  fraction_given_to_anna j ja = 0 :=
by sorry

end NUMINAMATH_GPT_javier_fraction_to_anna_zero_l2130_213096


namespace NUMINAMATH_GPT_ratio_expression_value_l2130_213000

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end NUMINAMATH_GPT_ratio_expression_value_l2130_213000


namespace NUMINAMATH_GPT_value_of_a_plus_d_l2130_213034

theorem value_of_a_plus_d (a b c d : ℕ) (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 10 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l2130_213034


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2130_213012

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (|a - b^2| + |b - a^2| ≤ 1) → ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ 
  ∃ (a b : ℝ), ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ ¬ (|a - b^2| + |b - a^2| ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2130_213012


namespace NUMINAMATH_GPT_cube_difference_positive_l2130_213029

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end NUMINAMATH_GPT_cube_difference_positive_l2130_213029


namespace NUMINAMATH_GPT_tickets_sold_l2130_213041

theorem tickets_sold (S G : ℕ) (hG : G = 388) (h_total : 4 * S + 6 * G = 2876) :
  S + G = 525 := by
  sorry

end NUMINAMATH_GPT_tickets_sold_l2130_213041


namespace NUMINAMATH_GPT_subset_0_in_X_l2130_213055

def X : Set ℝ := {x | x > -1}

theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end NUMINAMATH_GPT_subset_0_in_X_l2130_213055


namespace NUMINAMATH_GPT_range_of_a_l2130_213045

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x, ∃ y, y = (3 : ℝ) * x^2 + 2 * a * x + (a + 6) ∧ (y = 0)) :
  (a < -3 ∨ a > 6) :=
by { sorry }

end NUMINAMATH_GPT_range_of_a_l2130_213045


namespace NUMINAMATH_GPT_slices_per_person_l2130_213009

theorem slices_per_person
  (small_pizza_slices : ℕ)
  (large_pizza_slices : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_slices : ℕ)
  (bob_extra : ℕ)
  (susie_divisor : ℕ)
  (bill_slices : ℕ)
  (fred_slices : ℕ)
  (mark_slices : ℕ)
  (ann_slices : ℕ)
  (kelly_multiplier : ℕ) :
  small_pizza_slices = 4 →
  large_pizza_slices = 8 →
  small_pizzas_purchased = 4 →
  large_pizzas_purchased = 3 →
  george_slices = 3 →
  bob_extra = 1 →
  susie_divisor = 2 →
  bill_slices = 3 →
  fred_slices = 3 →
  mark_slices = 3 →
  ann_slices = 2 →
  kelly_multiplier = 2 →
  (2 * (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier))) =
    (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier)) :=
by
  sorry

end NUMINAMATH_GPT_slices_per_person_l2130_213009


namespace NUMINAMATH_GPT_minimum_value_of_z_l2130_213023

/-- Given the constraints: 
1. x - y + 5 ≥ 0,
2. x + y ≥ 0,
3. x ≤ 3,

Prove that the minimum value of z = (x + y + 2) / (x + 3) is 1/3.
-/
theorem minimum_value_of_z : 
  ∀ (x y : ℝ), 
    (x - y + 5 ≥ 0) ∧ 
    (x + y ≥ 0) ∧ 
    (x ≤ 3) → 
    ∃ (z : ℝ), 
      z = (x + y + 2) / (x + 3) ∧
      z = 1 / 3 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_minimum_value_of_z_l2130_213023


namespace NUMINAMATH_GPT_average_abcd_l2130_213084

-- Define the average condition of the numbers 4, 6, 9, a, b, c, d given as 20
def average_condition (a b c d : ℝ) : Prop :=
  (4 + 6 + 9 + a + b + c + d) / 7 = 20

-- Prove that the average of a, b, c, and d is 30.25 given the above condition
theorem average_abcd (a b c d : ℝ) (h : average_condition a b c d) : 
  (a + b + c + d) / 4 = 30.25 :=
by
  sorry

end NUMINAMATH_GPT_average_abcd_l2130_213084


namespace NUMINAMATH_GPT_inradius_of_triangle_area_three_times_perimeter_l2130_213060

theorem inradius_of_triangle_area_three_times_perimeter (A p s r : ℝ) (h1 : A = 3 * p) (h2 : p = 2 * s) (h3 : A = r * s) (h4 : s ≠ 0) :
  r = 6 :=
sorry

end NUMINAMATH_GPT_inradius_of_triangle_area_three_times_perimeter_l2130_213060


namespace NUMINAMATH_GPT_number_is_minus_three_l2130_213077

variable (x a : ℝ)

theorem number_is_minus_three (h1 : a = 0.5) (h2 : x / (a - 3) = 3 / (a + 2)) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_number_is_minus_three_l2130_213077


namespace NUMINAMATH_GPT_two_digit_numbers_division_condition_l2130_213051

theorem two_digit_numbers_division_condition {n x y q : ℕ} (h1 : 10 * x + y = n)
  (h2 : n % 6 = x)
  (h3 : n / 10 = 3) (h4 : n % 10 = y) :
  n = 33 ∨ n = 39 := 
sorry

end NUMINAMATH_GPT_two_digit_numbers_division_condition_l2130_213051


namespace NUMINAMATH_GPT_duration_of_each_turn_l2130_213072

-- Definitions based on conditions
def Wa := 1 / 4
def Wb := 1 / 12

-- Define the duration of each turn as T
def T : ℝ := 1 -- This is the correct answer we proved

-- Given conditions
def total_work_done := 6 * Wa + 6 * Wb

-- Lean statement to prove 
theorem duration_of_each_turn : T = 1 := by
  -- According to conditions, the total work done by a and b should equal the whole work
  have h1 : 3 * Wa + 3 * Wb = 1 := by sorry
  -- Let's conclude that T = 1
  sorry

end NUMINAMATH_GPT_duration_of_each_turn_l2130_213072


namespace NUMINAMATH_GPT_derivative_correct_l2130_213086

noncomputable def derivative_of_composite_function (x : ℝ) : Prop :=
  let y := (5 * x - 3) ^ 3
  let dy_dx := 3 * (5 * x - 3) ^ 2 * 5
  dy_dx = 15 * (5 * x - 3) ^ 2

theorem derivative_correct (x : ℝ) : derivative_of_composite_function x :=
by
  sorry

end NUMINAMATH_GPT_derivative_correct_l2130_213086


namespace NUMINAMATH_GPT_first_divisor_exists_l2130_213028

theorem first_divisor_exists (m d : ℕ) :
  (m % d = 47) ∧ (m % 24 = 23) ∧ (d > 47) → d = 72 :=
by
  sorry

end NUMINAMATH_GPT_first_divisor_exists_l2130_213028


namespace NUMINAMATH_GPT_sum_of_second_and_third_smallest_is_804_l2130_213089

noncomputable def sum_of_second_and_third_smallest : Nat :=
  let digits := [1, 6, 8]
  let second_smallest := 186
  let third_smallest := 618
  second_smallest + third_smallest

theorem sum_of_second_and_third_smallest_is_804 :
  sum_of_second_and_third_smallest = 804 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_second_and_third_smallest_is_804_l2130_213089


namespace NUMINAMATH_GPT_least_positive_divisible_l2130_213092

/-- The first five different prime numbers are given as conditions: -/
def prime1 := 2
def prime2 := 3
def prime3 := 5
def prime4 := 7
def prime5 := 11

/-- The least positive whole number divisible by the first five primes is 2310. -/
theorem least_positive_divisible :
  ∃ n : ℕ, n > 0 ∧ (n % prime1 = 0) ∧ (n % prime2 = 0) ∧ (n % prime3 = 0) ∧ (n % prime4 = 0) ∧ (n % prime5 = 0) ∧ n = 2310 :=
sorry

end NUMINAMATH_GPT_least_positive_divisible_l2130_213092


namespace NUMINAMATH_GPT_a_pow_10_add_b_pow_10_eq_123_l2130_213013

variable (a b : ℕ) -- better as non-negative integers for sequence progression

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a_pow_10_add_b_pow_10_eq_123 : a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_GPT_a_pow_10_add_b_pow_10_eq_123_l2130_213013


namespace NUMINAMATH_GPT_board_tiling_condition_l2130_213074

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end NUMINAMATH_GPT_board_tiling_condition_l2130_213074


namespace NUMINAMATH_GPT_total_ducats_is_160_l2130_213066

variable (T : ℤ) (a b c d e : ℤ) -- Variables to represent the amounts taken by the robbers

-- Conditions
axiom h1 : a = 81                                            -- The strongest robber took 81 ducats
axiom h2 : b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e    -- Each remaining robber took a different amount
axiom h3 : a + b + c + d + e = T                             -- Total amount of ducats
axiom redistribution : 
  -- Redistribution process leads to each robber having the same amount
  2*b + 2*c + 2*d + 2*e = T ∧
  2*(2*c + 2*d + 2*e) = T ∧
  2*(2*(2*d + 2*e)) = T ∧
  2*(2*(2*(2*e))) = T

-- Proof that verifies the total ducats is 160
theorem total_ducats_is_160 : T = 160 :=
by
  sorry

end NUMINAMATH_GPT_total_ducats_is_160_l2130_213066


namespace NUMINAMATH_GPT_seventh_observation_l2130_213042

-- Definitions from the conditions
def avg_original (x : ℕ) := 13
def num_observations_original := 6
def total_original := num_observations_original * (avg_original 0) -- 6 * 13 = 78

def avg_new := 12
def num_observations_new := num_observations_original + 1 -- 7
def total_new := num_observations_new * avg_new -- 7 * 12 = 84

-- The proof goal statement
theorem seventh_observation : (total_new - total_original) = 6 := 
  by
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_seventh_observation_l2130_213042


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l2130_213039

noncomputable def f (x : ℝ) : ℝ := 2^x - 3*x

theorem number_of_zeros_of_f : ∃ a b : ℝ, (f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ ∀ x : ℝ, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l2130_213039


namespace NUMINAMATH_GPT_find_original_number_l2130_213044

theorem find_original_number (x : ℝ)
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 :=
sorry

end NUMINAMATH_GPT_find_original_number_l2130_213044


namespace NUMINAMATH_GPT_sum_of_factors_l2130_213054

theorem sum_of_factors (x y : ℕ) :
  let exp := (27 * x ^ 6 - 512 * y ^ 6)
  let factor1 := (3 * x ^ 2 - 8 * y ^ 2)
  let factor2 := (3 * x ^ 2 + 8 * y ^ 2)
  let factor3 := (9 * x ^ 4 - 24 * x ^ 2 * y ^ 2 + 64 * y ^ 4)
  let sum := 3 + (-8) + 3 + 8 + 9 + (-24) + 64
  (factor1 * factor2 * factor3 = exp) ∧ (sum = 55) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_factors_l2130_213054


namespace NUMINAMATH_GPT_scale_total_length_l2130_213008

/-- Defining the problem parameters. -/
def number_of_parts : ℕ := 5
def length_of_each_part : ℕ := 18

/-- Theorem stating the total length of the scale. -/
theorem scale_total_length : number_of_parts * length_of_each_part = 90 :=
by
  sorry

end NUMINAMATH_GPT_scale_total_length_l2130_213008


namespace NUMINAMATH_GPT_total_hours_driven_l2130_213038

/-- Jade and Krista went on a road trip for 3 days. Jade drives 8 hours each day, and Krista drives 6 hours each day. Prove the total number of hours they drove altogether is 42. -/
theorem total_hours_driven (days : ℕ) (hours_jade_per_day : ℕ) (hours_krista_per_day : ℕ)
  (h1 : days = 3) (h2 : hours_jade_per_day = 8) (h3 : hours_krista_per_day = 6) :
  3 * 8 + 3 * 6 = 42 := 
by
  sorry

end NUMINAMATH_GPT_total_hours_driven_l2130_213038


namespace NUMINAMATH_GPT_Da_Yan_sequence_20th_term_l2130_213003

noncomputable def Da_Yan_sequence_term (n: ℕ) : ℕ :=
  if n % 2 = 0 then
    (n^2) / 2
  else
    (n^2 - 1) / 2

theorem Da_Yan_sequence_20th_term : Da_Yan_sequence_term 20 = 200 :=
by
  sorry

end NUMINAMATH_GPT_Da_Yan_sequence_20th_term_l2130_213003


namespace NUMINAMATH_GPT_rectangle_area_l2130_213093

theorem rectangle_area (a : ℕ) (h : 2 * (3 * a + 2 * a) = 160) : 3 * a * 2 * a = 1536 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2130_213093


namespace NUMINAMATH_GPT_slope_of_line_l2130_213046

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l2130_213046


namespace NUMINAMATH_GPT_simplify_fraction_sum_l2130_213076

theorem simplify_fraction_sum :
  (3 / 462) + (17 / 42) + (1 / 11) = 116 / 231 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_sum_l2130_213076


namespace NUMINAMATH_GPT_probability_of_blue_buttons_l2130_213095

theorem probability_of_blue_buttons
  (orig_red_A : ℕ) (orig_blue_A : ℕ)
  (removed_red : ℕ) (removed_blue : ℕ)
  (target_ratio : ℚ)
  (final_red_A : ℕ) (final_blue_A : ℕ)
  (final_red_B : ℕ) (final_blue_B : ℕ)
  (orig_buttons_A : orig_red_A + orig_blue_A = 16)
  (removed_buttons : removed_red = 3 ∧ removed_blue = 5)
  (final_buttons_A : final_red_A + final_blue_A = 8)
  (buttons_ratio : target_ratio = 2 / 3)
  (final_ratio_A : final_red_A + final_blue_A = target_ratio * 16)
  (red_in_A : final_red_A = orig_red_A - removed_red)
  (blue_in_A : final_blue_A = orig_blue_A - removed_blue)
  (red_in_B : final_red_B = removed_red)
  (blue_in_B : final_blue_B = removed_blue):
  (final_blue_A / (final_red_A + final_blue_A)) * (final_blue_B / (final_red_B + final_blue_B)) = 25 / 64 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_blue_buttons_l2130_213095


namespace NUMINAMATH_GPT_smallest_prime_sum_of_five_distinct_primes_l2130_213017

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct (a b c d e : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem smallest_prime_sum_of_five_distinct_primes :
  ∃ a b c d e : ℕ, is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ distinct a b c d e ∧ (a + b + c + d + e = 43) ∧ is_prime 43 :=
sorry

end NUMINAMATH_GPT_smallest_prime_sum_of_five_distinct_primes_l2130_213017


namespace NUMINAMATH_GPT_max_sum_of_positive_integers_with_product_144_l2130_213094

theorem max_sum_of_positive_integers_with_product_144 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 144 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 75 := 
by
  sorry

end NUMINAMATH_GPT_max_sum_of_positive_integers_with_product_144_l2130_213094


namespace NUMINAMATH_GPT_original_number_exists_l2130_213080

theorem original_number_exists :
  ∃ x : ℝ, 10 * x = x + 2.7 ∧ x = 0.3 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_number_exists_l2130_213080


namespace NUMINAMATH_GPT_ants_species_A_count_l2130_213067

theorem ants_species_A_count (a b : ℕ) (h1 : a + b = 30) (h2 : 2^5 * a + 3^5 * b = 3281) : 32 * a = 608 :=
by
  sorry

end NUMINAMATH_GPT_ants_species_A_count_l2130_213067


namespace NUMINAMATH_GPT_greater_number_is_twelve_l2130_213001

theorem greater_number_is_twelve (x : ℕ) (a b : ℕ) 
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x) 
  (h3 : a + b = 21) : 
  max a b = 12 :=
by 
  sorry

end NUMINAMATH_GPT_greater_number_is_twelve_l2130_213001


namespace NUMINAMATH_GPT_unique_rectangle_Q_l2130_213040

noncomputable def rectangle_Q_count (a : ℝ) :=
  let x := (3 * a) / 2
  let y := a / 2
  if x < 2 * a then 1 else 0

-- The main theorem
theorem unique_rectangle_Q (a : ℝ) (h : a > 0) :
  rectangle_Q_count a = 1 :=
sorry

end NUMINAMATH_GPT_unique_rectangle_Q_l2130_213040


namespace NUMINAMATH_GPT_crossword_solution_correct_l2130_213036

noncomputable def vertical_2 := "счет"
noncomputable def vertical_3 := "евро"
noncomputable def vertical_4 := "доллар"
noncomputable def vertical_5 := "вклад"
noncomputable def vertical_6 := "золото"
noncomputable def vertical_7 := "ломбард"

noncomputable def horizontal_1 := "обмен"
noncomputable def horizontal_2 := "система"
noncomputable def horizontal_3 := "ломбард"

theorem crossword_solution_correct :
  (vertical_2 = "счет") ∧
  (vertical_3 = "евро") ∧
  (vertical_4 = "доллар") ∧
  (vertical_5 = "вклад") ∧
  (vertical_6 = "золото") ∧
  (vertical_7 = "ломбард") ∧
  (horizontal_1 = "обмен") ∧
  (horizontal_2 = "система") ∧
  (horizontal_3 = "ломбард") :=
by
  sorry

end NUMINAMATH_GPT_crossword_solution_correct_l2130_213036


namespace NUMINAMATH_GPT_common_chord_through_vertex_l2130_213053

-- Define the structure for the problem
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

def passes_through (x y x_f y_f : ℝ) : Prop := (x - x_f) * (x - x_f) + y * y = 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The main statement to prove
theorem common_chord_through_vertex (p : ℝ)
  (A B C D : ℝ × ℝ)
  (hA : parabola A.snd A.fst p)
  (hB : parabola B.snd B.fst p)
  (hC : parabola C.snd C.fst p)
  (hD : parabola D.snd D.fst p)
  (hAB_f : passes_through A.fst A.snd (focus p).fst (focus p).snd)
  (hCD_f : passes_through C.fst C.snd (focus p).fst (focus p).snd) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + k = 0) → (y + k = 0) :=
by sorry

end NUMINAMATH_GPT_common_chord_through_vertex_l2130_213053


namespace NUMINAMATH_GPT_ed_money_left_after_hotel_stay_l2130_213019

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end NUMINAMATH_GPT_ed_money_left_after_hotel_stay_l2130_213019


namespace NUMINAMATH_GPT_find_m_pure_imaginary_l2130_213064

theorem find_m_pure_imaginary (m : ℝ) (h : m^2 + m - 2 + (m^2 - 1) * I = (0 : ℝ) + (m^2 - 1) * I) :
  m = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_pure_imaginary_l2130_213064


namespace NUMINAMATH_GPT_abc_divides_sum_pow_31_l2130_213031

theorem abc_divides_sum_pow_31 (a b c : ℕ) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) : 
  abc ∣ (a + b + c) ^ 31 := 
sorry

end NUMINAMATH_GPT_abc_divides_sum_pow_31_l2130_213031


namespace NUMINAMATH_GPT_equation_of_l_l2130_213087

-- Defining the equations of the circles
def circle_O (x y : ℝ) := x^2 + y^2 = 4
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 4 = 0

-- Assuming the line l makes circles O and C symmetric
def symmetric (l : ℝ → ℝ → Prop) := ∀ (x y : ℝ), l x y → 
  (∃ (x' y' : ℝ), circle_O x y ∧ circle_C x' y' ∧ (x + x') / 2 = x' ∧ (y + y') / 2 = y')

-- Stating the theorem to be proven
theorem equation_of_l :
  ∀ l : ℝ → ℝ → Prop, symmetric l → (∀ x y : ℝ, l x y ↔ x - y + 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_l_l2130_213087


namespace NUMINAMATH_GPT_arithmetic_base_conversion_l2130_213022

-- We start with proving base conversions

def convert_base3_to_base10 (n : ℕ) : ℕ := 1 * (3^0) + 2 * (3^1) + 1 * (3^2)

def convert_base7_to_base10 (n : ℕ) : ℕ := 6 * (7^0) + 5 * (7^1) + 4 * (7^2) + 3 * (7^3)

def convert_base9_to_base10 (n : ℕ) : ℕ := 6 * (9^0) + 7 * (9^1) + 8 * (9^2) + 9 * (9^3)

-- Prove the main equality

theorem arithmetic_base_conversion:
  (2468 : ℝ) / convert_base3_to_base10 121 + convert_base7_to_base10 3456 - convert_base9_to_base10 9876 = -5857.75 :=
by
  have h₁ : convert_base3_to_base10 121 = 16 := by native_decide
  have h₂ : convert_base7_to_base10 3456 = 1266 := by native_decide
  have h₃ : convert_base9_to_base10 9876 = 7278 := by native_decide
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_arithmetic_base_conversion_l2130_213022


namespace NUMINAMATH_GPT_diff_of_squares_l2130_213058

theorem diff_of_squares (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end NUMINAMATH_GPT_diff_of_squares_l2130_213058


namespace NUMINAMATH_GPT_ratio_of_segments_l2130_213052

theorem ratio_of_segments (E F G H : ℝ) (h_collinear : E < F ∧ F < G ∧ G < H)
  (hEF : F - E = 3) (hFG : G - F = 6) (hEH : H - E = 20) : (G - E) / (H - F) = 9 / 17 := by
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l2130_213052


namespace NUMINAMATH_GPT_ratio_of_gold_and_copper_l2130_213063

theorem ratio_of_gold_and_copper
  (G C : ℝ)
  (hG : G = 11)
  (hC : C = 5)
  (hA : (11 * G + 5 * C) / (G + C) = 8) : G = C :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_gold_and_copper_l2130_213063


namespace NUMINAMATH_GPT_inv_3i_minus_2inv_i_eq_neg_inv_5i_l2130_213025

-- Define the imaginary unit i such that i^2 = -1
def i : ℂ := Complex.I
axiom i_square : i^2 = -1

-- Proof statement
theorem inv_3i_minus_2inv_i_eq_neg_inv_5i : (3 * i - 2 * (1 / i))⁻¹ = -i / 5 :=
by
  -- Replace these steps with the corresponding actual proofs
  sorry

end NUMINAMATH_GPT_inv_3i_minus_2inv_i_eq_neg_inv_5i_l2130_213025


namespace NUMINAMATH_GPT_AM_GM_inequality_min_value_l2130_213035

theorem AM_GM_inequality_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_min_value_l2130_213035


namespace NUMINAMATH_GPT_kaleb_initial_games_l2130_213056

-- Let n be the number of games Kaleb started out with
def initial_games (n : ℕ) : Prop :=
  let sold_games := 46
  let boxes := 6
  let games_per_box := 5
  n = sold_games + boxes * games_per_box

-- Now we state the theorem
theorem kaleb_initial_games : ∃ n, initial_games n ∧ n = 76 :=
  by sorry

end NUMINAMATH_GPT_kaleb_initial_games_l2130_213056


namespace NUMINAMATH_GPT_solve_for_x_l2130_213024

theorem solve_for_x : ∃ x : ℚ, -3 * x - 8 = 4 * x + 3 ∧ x = -11 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2130_213024


namespace NUMINAMATH_GPT_ratio_of_average_speeds_l2130_213020

-- Definitions based on the conditions
def distance_AB := 600 -- km
def distance_AC := 300 -- km
def time_Eddy := 3 -- hours
def time_Freddy := 3 -- hours

def speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def speed_Eddy := speed distance_AB time_Eddy
def speed_Freddy := speed distance_AC time_Freddy

theorem ratio_of_average_speeds : (speed_Eddy / speed_Freddy) = 2 :=
by 
  -- Proof is skipped, so we use sorry
  sorry

end NUMINAMATH_GPT_ratio_of_average_speeds_l2130_213020


namespace NUMINAMATH_GPT_minjun_current_height_l2130_213073

variable (initial_height : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ)

theorem minjun_current_height
  (h_initial : initial_height = 1.1)
  (h_growth_last_year : growth_last_year = 0.2)
  (h_growth_this_year : growth_this_year = 0.1) :
  initial_height + growth_last_year + growth_this_year = 1.4 :=
by
  sorry

end NUMINAMATH_GPT_minjun_current_height_l2130_213073


namespace NUMINAMATH_GPT_carly_practice_backstroke_days_per_week_l2130_213062

theorem carly_practice_backstroke_days_per_week 
  (butterfly_hours_per_day : ℕ) 
  (butterfly_days_per_week : ℕ) 
  (backstroke_hours_per_day : ℕ) 
  (total_hours_per_month : ℕ)
  (weeks_per_month : ℕ)
  (d : ℕ)
  (h1 : butterfly_hours_per_day = 3)
  (h2 : butterfly_days_per_week = 4)
  (h3 : backstroke_hours_per_day = 2)
  (h4 : total_hours_per_month = 96)
  (h5 : weeks_per_month = 4)
  (h6 : total_hours_per_month - (butterfly_hours_per_day * butterfly_days_per_week * weeks_per_month) = backstroke_hours_per_day * d * weeks_per_month) :
  d = 6 := by
  sorry

end NUMINAMATH_GPT_carly_practice_backstroke_days_per_week_l2130_213062


namespace NUMINAMATH_GPT_point_on_curve_l2130_213016

theorem point_on_curve :
  let x := -3 / 4
  let y := 1 / 2
  x^2 = (y^2 - 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_point_on_curve_l2130_213016


namespace NUMINAMATH_GPT_step_count_initial_l2130_213014

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end NUMINAMATH_GPT_step_count_initial_l2130_213014


namespace NUMINAMATH_GPT_cover_points_with_circles_l2130_213043

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → min (dist (points i) (points j)) (min (dist (points j) (points k)) (dist (points i) (points k))) ≤ 1) :
  ∃ (a b : Fin n), ∀ (p : Fin n), dist (points p) (points a) ≤ 1 ∨ dist (points p) (points b) ≤ 1 := 
sorry

end NUMINAMATH_GPT_cover_points_with_circles_l2130_213043


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l2130_213083

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 4) : y = 4 - 5 * x :=
by
  /- Proof to be filled in here. -/
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l2130_213083


namespace NUMINAMATH_GPT_sum_123_consecutive_even_numbers_l2130_213010

theorem sum_123_consecutive_even_numbers :
  let n := 123
  let a := 2
  let d := 2
  let sum_arithmetic_series (n a l : ℕ) := n * (a + l) / 2
  let last_term := a + (n - 1) * d
  sum_arithmetic_series n a last_term = 15252 :=
by
  sorry

end NUMINAMATH_GPT_sum_123_consecutive_even_numbers_l2130_213010


namespace NUMINAMATH_GPT_largest_angle_is_120_l2130_213059

variable (d e f : ℝ)
variable (h1 : d + 3 * e + 3 * f = d^2)
variable (h2 : d + 3 * e - 3 * f = -4)

theorem largest_angle_is_120 (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -4) : 
  ∃ (F : ℝ), F = 120 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_is_120_l2130_213059


namespace NUMINAMATH_GPT_inequality_solution_empty_solution_set_l2130_213090

-- Problem 1: Prove the inequality and the solution range
theorem inequality_solution (x : ℝ) : (-7 < x ∧ x < 3) ↔ ( (x - 3)/(x + 7) < 0 ) :=
sorry

-- Problem 2: Prove the conditions for empty solution set
theorem empty_solution_set (a : ℝ) : (a > 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4*a*x + 4*a^2 + a ≤ 0) :=
sorry

end NUMINAMATH_GPT_inequality_solution_empty_solution_set_l2130_213090


namespace NUMINAMATH_GPT_graph_not_in_third_quadrant_l2130_213099

-- Define the conditions
variable (m : ℝ)
variable (h1 : 0 < m)
variable (h2 : m < 2)

-- Define the graph equation
noncomputable def line_eq (x : ℝ) : ℝ := (m - 2) * x + m

-- The proof problem: the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ¬ ∃ x y : ℝ, (x < 0 ∧ y < 0 ∧ y = (m - 2) * x + m) :=
sorry

end NUMINAMATH_GPT_graph_not_in_third_quadrant_l2130_213099


namespace NUMINAMATH_GPT_alberto_bikes_more_l2130_213065

-- Definitions of given speeds
def alberto_speed : ℝ := 15
def bjorn_speed : ℝ := 11.25

-- The time duration considered
def time_hours : ℝ := 5

-- Calculate the distances each traveled
def alberto_distance : ℝ := alberto_speed * time_hours
def bjorn_distance : ℝ := bjorn_speed * time_hours

-- Calculate the difference in distances
def distance_difference : ℝ := alberto_distance - bjorn_distance

-- The theorem to be proved
theorem alberto_bikes_more : distance_difference = 18.75 := by
    sorry

end NUMINAMATH_GPT_alberto_bikes_more_l2130_213065


namespace NUMINAMATH_GPT_max_consecutive_integers_lt_1000_l2130_213048

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end NUMINAMATH_GPT_max_consecutive_integers_lt_1000_l2130_213048


namespace NUMINAMATH_GPT_abs_eq_non_pos_2x_plus_4_l2130_213079

-- Condition: |2x + 4| = 0
-- Conclusion: x = -2
theorem abs_eq_non_pos_2x_plus_4 (x : ℝ) : (|2 * x + 4| = 0) → x = -2 :=
by
  intro h
  -- Here lies the proof, but we use sorry to indicate the unchecked part.
  sorry

end NUMINAMATH_GPT_abs_eq_non_pos_2x_plus_4_l2130_213079


namespace NUMINAMATH_GPT_taller_tree_height_l2130_213050

theorem taller_tree_height :
  ∀ (h : ℕ), 
    ∃ (h_s : ℕ), (h_s = h - 24) ∧ (5 * h = 7 * h_s) → h = 84 :=
by
  sorry

end NUMINAMATH_GPT_taller_tree_height_l2130_213050


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2130_213033

theorem sufficient_not_necessary_condition (x : ℝ) : (x > 0 → |x| = x) ∧ (|x| = x → x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2130_213033
