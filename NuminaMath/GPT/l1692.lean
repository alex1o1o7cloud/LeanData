import Mathlib

namespace NUMINAMATH_GPT_total_kids_l1692_169277

theorem total_kids (girls boys: ℕ) (h1: girls = 3) (h2: boys = 6) : girls + boys = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_kids_l1692_169277


namespace NUMINAMATH_GPT_mairiad_distance_ratio_l1692_169204

open Nat

theorem mairiad_distance_ratio :
  ∀ (x : ℕ),
  let miles_run := 40
  let miles_walked := 3 * miles_run / 5
  let total_distance := miles_run + miles_walked + x * miles_run
  total_distance = 184 →
  24 + x * 40 = 144 →
  (24 + 3 * 40) / 40 = 3.6 := 
sorry

end NUMINAMATH_GPT_mairiad_distance_ratio_l1692_169204


namespace NUMINAMATH_GPT_fruit_store_initial_quantities_l1692_169295

-- Definitions from conditions:
def total_fruit (a b c : ℕ) := a + b + c = 275
def sold_apples (a : ℕ) := a - 30
def added_peaches (b : ℕ) := b + 45
def sold_pears (c : ℕ) := c - c / 4
def final_ratio (a b c : ℕ) := (sold_apples a) / 4 = (added_peaches b) / 3 ∧ (added_peaches b) / 3 = (sold_pears c) / 2

-- The proof problem:
theorem fruit_store_initial_quantities (a b c : ℕ) (h1 : total_fruit a b c) 
  (h2 : final_ratio a b c) : a = 150 ∧ b = 45 ∧ c = 80 :=
sorry

end NUMINAMATH_GPT_fruit_store_initial_quantities_l1692_169295


namespace NUMINAMATH_GPT_part1_part2_l1692_169219

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ B a = B a ↔ a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A ∩ B a = B a ↔ a ≤ -1 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1692_169219


namespace NUMINAMATH_GPT_prob_sum_to_3_three_dice_correct_l1692_169253

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end NUMINAMATH_GPT_prob_sum_to_3_three_dice_correct_l1692_169253


namespace NUMINAMATH_GPT_proof_problem_l1692_169234

-- Definitions
def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

-- The equality proof statement
theorem proof_problem :
  (A ∩ B = {2, 5}) ∧
  ({x | x ∈ U ∧ ¬ (x ∈ A)} = {3, 4, 6}) ∧
  (A ∪ {x | x ∈ U ∧ ¬ (x ∈ B)} = {1, 2, 5, 6}) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1692_169234


namespace NUMINAMATH_GPT_geometric_seq_sum_l1692_169290

theorem geometric_seq_sum (a : ℝ) (q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) 
    (hS4 : a * (1 - q^4) / (1 - q) = 1) 
    (hS12 : a * (1 - q^12) / (1 - q) = 13) 
    : a * q^12 * (1 + q + q^2 + q^3) = 27 := 
by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_l1692_169290


namespace NUMINAMATH_GPT_unique_decomposition_of_two_reciprocals_l1692_169279

theorem unique_decomposition_of_two_reciprocals (p : ℕ) (hp : Nat.Prime p) (hp_ne_two : p ≠ 2) :
  ∃ (x y : ℕ), x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 2 / (p : ℝ)) := sorry

end NUMINAMATH_GPT_unique_decomposition_of_two_reciprocals_l1692_169279


namespace NUMINAMATH_GPT_sarah_score_l1692_169216

theorem sarah_score
  (hunter_score : ℕ)
  (john_score : ℕ)
  (grant_score : ℕ)
  (sarah_score : ℕ)
  (h1 : hunter_score = 45)
  (h2 : john_score = 2 * hunter_score)
  (h3 : grant_score = john_score + 10)
  (h4 : sarah_score = grant_score - 5) :
  sarah_score = 95 :=
by
  sorry

end NUMINAMATH_GPT_sarah_score_l1692_169216


namespace NUMINAMATH_GPT_bulb_illumination_l1692_169214

theorem bulb_illumination (n : ℕ) (h : n = 6) : 
  (2^n - 1) = 63 := by {
  sorry
}

end NUMINAMATH_GPT_bulb_illumination_l1692_169214


namespace NUMINAMATH_GPT_opposite_face_of_X_is_Y_l1692_169284

-- Define the labels for the cube faces
inductive Label
| X | V | Z | W | U | Y

-- Define adjacency relations
def adjacent (a b : Label) : Prop :=
  (a = Label.X ∧ (b = Label.V ∨ b = Label.Z ∨ b = Label.W ∨ b = Label.U)) ∨
  (b = Label.X ∧ (a = Label.V ∨ a = Label.Z ∨ a = Label.W ∨ a = Label.U))

-- Define the theorem to prove the face opposite to X
theorem opposite_face_of_X_is_Y : ∀ l1 l2 l3 l4 l5 l6 : Label,
  l1 = Label.X →
  l2 = Label.V →
  l3 = Label.Z →
  l4 = Label.W →
  l5 = Label.U →
  l6 = Label.Y →
  ¬ adjacent l1 l6 →
  ¬ adjacent l2 l6 →
  ¬ adjacent l3 l6 →
  ¬ adjacent l4 l6 →
  ¬ adjacent l5 l6 →
  ∃ (opposite : Label), opposite = Label.Y ∧ opposite = l6 :=
by sorry

end NUMINAMATH_GPT_opposite_face_of_X_is_Y_l1692_169284


namespace NUMINAMATH_GPT_joan_books_l1692_169209

theorem joan_books (initial_books sold_books result_books : ℕ) 
  (h_initial : initial_books = 33) 
  (h_sold : sold_books = 26) 
  (h_result : initial_books - sold_books = result_books) : 
  result_books = 7 := 
by
  sorry

end NUMINAMATH_GPT_joan_books_l1692_169209


namespace NUMINAMATH_GPT_sum_of_angles_of_parallelepiped_diagonal_lt_pi_l1692_169282

/-- In a rectangular parallelepiped, if the main diagonal forms angles α, β, and γ with the three edges meeting at a vertex, then the sum of these angles is less than π. -/
theorem sum_of_angles_of_parallelepiped_diagonal_lt_pi {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : 2 * α + 2 * β + 2 * γ < 2 * π) :
  α + β + γ < π := by
sorry

end NUMINAMATH_GPT_sum_of_angles_of_parallelepiped_diagonal_lt_pi_l1692_169282


namespace NUMINAMATH_GPT_calculate_expression_l1692_169208

theorem calculate_expression : -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (Real.pi / 3) = 6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1692_169208


namespace NUMINAMATH_GPT_total_opponents_points_is_36_l1692_169227
-- Import the Mathlib library

-- Define the conditions as Lean definitions
def game_scores : List ℕ := [3, 5, 6, 7, 8, 9, 11, 12]

def lost_by_two (n : ℕ) : Prop := n + 2 ∈ game_scores

def three_times_as_many (n : ℕ) : Prop := n * 3 ∈ game_scores

-- State the problem
theorem total_opponents_points_is_36 : 
  (∃ l1 l2 l3 w1 w2 w3 w4 w5 : ℕ, 
    game_scores = [l1, l2, l3, w1, w2, w3, w4, w5] ∧
    lost_by_two l1 ∧ lost_by_two l2 ∧ lost_by_two l3 ∧
    three_times_as_many w1 ∧ three_times_as_many w2 ∧ 
    three_times_as_many w3 ∧ three_times_as_many w4 ∧ 
    three_times_as_many w5 ∧ 
    l1 + 2 + l2 + 2 + l3 + 2 + ((w1 / 3) + (w2 / 3) + (w3 / 3) + (w4 / 3) + (w5 / 3)) = 36) :=
sorry

end NUMINAMATH_GPT_total_opponents_points_is_36_l1692_169227


namespace NUMINAMATH_GPT_polygon_sides_eq_7_l1692_169288

theorem polygon_sides_eq_7 (n : ℕ) (h : n * (n - 3) / 2 = 2 * n) : n = 7 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_7_l1692_169288


namespace NUMINAMATH_GPT_evaluate_expression_l1692_169229

theorem evaluate_expression :
  (2^1 - 3 + 5^3 - 2)⁻¹ * 3 = (3 : ℚ) / 122 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1692_169229


namespace NUMINAMATH_GPT_length_AB_l1692_169269

-- Definitions and conditions
variables (R r a : ℝ) (hR : R > r) (BC_eq_a : BC = a) (r_eq_4 : r = 4)

-- Length of AB
theorem length_AB (AB : ℝ) : AB = a * Real.sqrt (R / (R - 4)) :=
sorry

end NUMINAMATH_GPT_length_AB_l1692_169269


namespace NUMINAMATH_GPT_car_and_bicycle_distances_l1692_169297

noncomputable def train_speed : ℝ := 100 -- speed of the train in mph
noncomputable def car_speed : ℝ := (2 / 3) * train_speed -- speed of the car in mph
noncomputable def bicycle_speed : ℝ := (1 / 5) * train_speed -- speed of the bicycle in mph
noncomputable def travel_time_hours : ℝ := 30 / 60 -- travel time in hours, which is 0.5 hours

noncomputable def car_distance : ℝ := car_speed * travel_time_hours
noncomputable def bicycle_distance : ℝ := bicycle_speed * travel_time_hours

theorem car_and_bicycle_distances :
  car_distance = 100 / 3 ∧ bicycle_distance = 10 :=
by
  sorry

end NUMINAMATH_GPT_car_and_bicycle_distances_l1692_169297


namespace NUMINAMATH_GPT_a_equals_b_l1692_169245

theorem a_equals_b (a b : ℕ) (h : a^3 + a + 4 * b^2 = 4 * a * b + b + b * a^2) : a = b := 
sorry

end NUMINAMATH_GPT_a_equals_b_l1692_169245


namespace NUMINAMATH_GPT_cyclic_proportion_l1692_169267

variable {A B C p q r : ℝ}

theorem cyclic_proportion (h1 : A / B = p) (h2 : B / C = q) (h3 : C / A = r) :
  ∃ x y z, A = x ∧ B = y ∧ C = z ∧ x / y = p ∧ y / z = q ∧ z / x = r ∧
  x = (p^2 * q / r)^(1/3:ℝ) ∧ y = (q^2 * r / p)^(1/3:ℝ) ∧ z = (r^2 * p / q)^(1/3:ℝ) :=
by sorry

end NUMINAMATH_GPT_cyclic_proportion_l1692_169267


namespace NUMINAMATH_GPT_call_duration_l1692_169275

def initial_credit : ℝ := 30
def cost_per_minute : ℝ := 0.16
def remaining_credit : ℝ := 26.48

theorem call_duration :
  (initial_credit - remaining_credit) / cost_per_minute = 22 := 
sorry

end NUMINAMATH_GPT_call_duration_l1692_169275


namespace NUMINAMATH_GPT_find_k_when_root_is_zero_l1692_169201

-- Define the quadratic equation and what it implies
theorem find_k_when_root_is_zero (k : ℝ) (h : (k-1) * 0^2 + 6 * 0 + k^2 - k = 0) :
  k = 0 :=
by
  -- The proof steps would go here, but we're skipping it as instructed
  sorry

end NUMINAMATH_GPT_find_k_when_root_is_zero_l1692_169201


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l1692_169296

theorem sufficient_but_not_necessary (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / a) > (1 / b) :=
by
  sorry

theorem sufficient_but_not_necessary_rel (a b : ℝ) : 0 < a ∧ a < b ↔ (1 / a) > (1 / b) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l1692_169296


namespace NUMINAMATH_GPT_added_number_is_nine_l1692_169212

theorem added_number_is_nine (y : ℤ) : 
  3 * (2 * 4 + y) = 51 → y = 9 :=
by
  sorry

end NUMINAMATH_GPT_added_number_is_nine_l1692_169212


namespace NUMINAMATH_GPT_find_t_l1692_169248

variables {a b c r s t : ℝ}

theorem find_t (h1 : a + b + c = -3)
             (h2 : a * b + b * c + c * a = 4)
             (h3 : a * b * c = -1)
             (h4 : ∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 → (x = a ∨ x = b ∨ x = c))
             (h5 : ∀ y, y^3 + r*y^2 + s*y + t = 0 → (y = a + b ∨ y = b + c ∨ y = c + a))
             : t = 11 :=
sorry

end NUMINAMATH_GPT_find_t_l1692_169248


namespace NUMINAMATH_GPT_a5_value_l1692_169280

def sequence_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem a5_value (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → sequence_sum n a = (1 / 2 : ℚ) * (a n : ℚ) + 1) :
  a 5 = 2 := by
  sorry

end NUMINAMATH_GPT_a5_value_l1692_169280


namespace NUMINAMATH_GPT_largest_sum_36_l1692_169242

theorem largest_sum_36 : ∃ n : ℕ, ∃ a : ℕ, (n * a + (n * (n - 1)) / 2 = 36) ∧ ∀ m : ℕ, (m * a + (m * (m - 1)) / 2 = 36) → m ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_36_l1692_169242


namespace NUMINAMATH_GPT_part1_max_min_part2_cos_value_l1692_169251

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x + Real.pi / 6)

theorem part1_max_min (x : ℝ) (hx : -1/2 ≤ x ∧ x ≤ 1/2) : 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = 2) ∧ 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = -Real.sqrt 3) :=
sorry

theorem part2_cos_value (α : ℝ) (h : f (α / (2 * Real.pi)) = 1/4) : 
  Real.cos (2 * Real.pi / 3 - α) = -31/32 :=
sorry

end NUMINAMATH_GPT_part1_max_min_part2_cos_value_l1692_169251


namespace NUMINAMATH_GPT_solve_for_x_l1692_169246

theorem solve_for_x {x : ℝ} (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_solve_for_x_l1692_169246


namespace NUMINAMATH_GPT_secondary_spermatocytes_can_contain_two_y_chromosomes_l1692_169231

-- Definitions corresponding to the conditions
def primary_spermatocytes_first_meiotic_division_contains_y (n : Nat) : Prop := n = 1
def spermatogonia_metaphase_mitosis_contains_y (n : Nat) : Prop := n = 1
def secondary_spermatocytes_second_meiotic_division_contains_y (n : Nat) : Prop := n = 0 ∨ n = 2
def spermatogonia_prophase_mitosis_contains_y (n : Nat) : Prop := n = 1

-- The theorem statement equivalent to the given math problem
theorem secondary_spermatocytes_can_contain_two_y_chromosomes :
  ∃ n, (secondary_spermatocytes_second_meiotic_division_contains_y n ∧ n = 2) :=
sorry

end NUMINAMATH_GPT_secondary_spermatocytes_can_contain_two_y_chromosomes_l1692_169231


namespace NUMINAMATH_GPT_diff_PA_AQ_const_l1692_169255

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem diff_PA_AQ_const (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let P := (0, -sqrt 2)
  let Q := (0, sqrt 2)
  let A := (a, sqrt (a^2 + 1))
  distance P A - distance A Q = 2 := 
sorry

end NUMINAMATH_GPT_diff_PA_AQ_const_l1692_169255


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1692_169207

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

variable (S_eq_k_mul_n_plus_2 : ∀ n, S n = (n + 2) * (S 0 / (n + 2)))
variable (T_eq_k_mul_n_plus_1 : ∀ n, T n = (n + 1) * (T 0 / (n + 1)))

theorem arithmetic_sequence_ratio (h₁ : arithmetic_sequence a) (h₂ : arithmetic_sequence b)
  (h₃ : ∀ n, S n = sum_first_n_terms a n)
  (h₄ : ∀ n, T n = sum_first_n_terms b n)
  (h₅ : ∀ n, (S n) / (T n) = (n + 2) / (n + 1))
  : a 6 / b 8 = 13 / 16 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1692_169207


namespace NUMINAMATH_GPT_find_first_term_l1692_169233

noncomputable def first_term_of_arithmetic_sequence : ℝ := -19.2

theorem find_first_term
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1050)
  (h2 : 50 * (2 * a + 199 * d) = 4050) :
  a = first_term_of_arithmetic_sequence :=
by
  -- Given conditions
  have h1' : 2 * a + 99 * d = 21 := by sorry
  have h2' : 2 * a + 199 * d = 81 := by sorry
  -- Solve for d
  have hd : d = 0.6 := by sorry
  -- Substitute d into h1'
  have h_subst : 2 * a + 99 * 0.6 = 21 := by sorry
  -- Solve for a
  have ha : a = -19.2 := by sorry
  exact ha

end NUMINAMATH_GPT_find_first_term_l1692_169233


namespace NUMINAMATH_GPT_equilateral_triangle_area_ratio_l1692_169298

theorem equilateral_triangle_area_ratio :
  let side_small := 1
  let perim_small := 3 * side_small
  let total_fencing := 6 * perim_small
  let side_large := total_fencing / 3
  let area_small := (Real.sqrt 3) / 4 * side_small ^ 2
  let area_large := (Real.sqrt 3) / 4 * side_large ^ 2
  let total_area_small := 6 * area_small
  total_area_small / area_large = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_ratio_l1692_169298


namespace NUMINAMATH_GPT_pow_two_greater_than_square_l1692_169247

theorem pow_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2 ^ n > n ^ 2 :=
  sorry

end NUMINAMATH_GPT_pow_two_greater_than_square_l1692_169247


namespace NUMINAMATH_GPT_card_stack_partition_l1692_169210

theorem card_stack_partition (n k : ℕ) (cards : Multiset ℕ) (h1 : ∀ x ∈ cards, x ∈ Finset.range (n + 1)) (h2 : cards.sum = k * n!) :
  ∃ stacks : List (Multiset ℕ), stacks.length = k ∧ ∀ stack ∈ stacks, stack.sum = n! :=
sorry

end NUMINAMATH_GPT_card_stack_partition_l1692_169210


namespace NUMINAMATH_GPT_distance_CD_l1692_169265

theorem distance_CD (C D : ℝ × ℝ) (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) 
  (hC : C = (r₁, φ₁)) (hD : D = (r₂, φ₂)) (r₁_eq_5 : r₁ = 5) (r₂_eq_12 : r₂ = 12)
  (angle_diff : φ₁ - φ₂ = π / 3) : dist C D = Real.sqrt 109 :=
  sorry

end NUMINAMATH_GPT_distance_CD_l1692_169265


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_over_2_l1692_169203

theorem cos_alpha_plus_pi_over_2 (α : ℝ) (h : Real.sin α = 1/3) : 
    Real.cos (α + Real.pi / 2) = -(1/3) :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_pi_over_2_l1692_169203


namespace NUMINAMATH_GPT_solve_x_squared_solve_x_cubed_l1692_169266

-- Define the first problem with its condition and prove the possible solutions
theorem solve_x_squared {x : ℝ} (h : (x + 1)^2 = 9) : x = 2 ∨ x = -4 :=
sorry

-- Define the second problem with its condition and prove the possible solution
theorem solve_x_cubed {x : ℝ} (h : -2 * (x^3 - 1) = 18) : x = -2 :=
sorry

end NUMINAMATH_GPT_solve_x_squared_solve_x_cubed_l1692_169266


namespace NUMINAMATH_GPT_ballet_class_members_l1692_169224

theorem ballet_class_members (large_groups : ℕ) (members_per_large_group : ℕ) (total_members : ℕ) 
    (h1 : large_groups = 12) (h2 : members_per_large_group = 7) (h3 : total_members = large_groups * members_per_large_group) : 
    total_members = 84 :=
sorry

end NUMINAMATH_GPT_ballet_class_members_l1692_169224


namespace NUMINAMATH_GPT_calculate_x_l1692_169289

variable (a b x : ℝ)
variable (h1 : r = (3 * a) ^ (3 * b))
variable (h2 : r = a ^ b * x ^ b)
variable (h3 : x > 0)

theorem calculate_x (a b x : ℝ) (h1 : r = (3 * a) ^ (3 * b)) (h2 : r = a ^ b * x ^ b) (h3 : x > 0) : x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_GPT_calculate_x_l1692_169289


namespace NUMINAMATH_GPT_fewer_onions_correct_l1692_169281

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end NUMINAMATH_GPT_fewer_onions_correct_l1692_169281


namespace NUMINAMATH_GPT_slices_left_per_person_is_2_l1692_169263

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end NUMINAMATH_GPT_slices_left_per_person_is_2_l1692_169263


namespace NUMINAMATH_GPT_days_to_use_up_one_bag_l1692_169254

def rice_kg : ℕ := 11410
def bags : ℕ := 3260
def rice_per_day : ℚ := 0.25
def rice_per_bag : ℚ := rice_kg / bags

theorem days_to_use_up_one_bag : (rice_per_bag / rice_per_day) = 14 := by
  sorry

end NUMINAMATH_GPT_days_to_use_up_one_bag_l1692_169254


namespace NUMINAMATH_GPT_find_a5_plus_a7_l1692_169286

variable {a : ℕ → ℝ}

theorem find_a5_plus_a7 (h : a 3 + a 9 = 16) : a 5 + a 7 = 16 := 
sorry

end NUMINAMATH_GPT_find_a5_plus_a7_l1692_169286


namespace NUMINAMATH_GPT_base_case_n_equals_1_l1692_169271

variable {a : ℝ}
variable {n : ℕ}

theorem base_case_n_equals_1 (h1 : a ≠ 1) (h2 : n = 1) : 1 + a = 1 + a :=
by
  sorry

end NUMINAMATH_GPT_base_case_n_equals_1_l1692_169271


namespace NUMINAMATH_GPT_function_passes_through_A_l1692_169220

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + Real.log x / Real.log a

theorem function_passes_through_A 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≠ 1)
  : f a 2 = 4 := sorry

end NUMINAMATH_GPT_function_passes_through_A_l1692_169220


namespace NUMINAMATH_GPT_only_odd_integer_option_l1692_169259

theorem only_odd_integer_option : 
  (6 ^ 2 = 36 ∧ Even 36) ∧ 
  (23 - 17 = 6 ∧ Even 6) ∧ 
  (9 * 24 = 216 ∧ Even 216) ∧ 
  (96 / 8 = 12 ∧ Even 12) ∧ 
  (9 * 41 = 369 ∧ Odd 369)
:= by
  sorry

end NUMINAMATH_GPT_only_odd_integer_option_l1692_169259


namespace NUMINAMATH_GPT_find_g_of_7_l1692_169260

theorem find_g_of_7 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_g_of_7_l1692_169260


namespace NUMINAMATH_GPT_roots_geometric_progression_condition_l1692_169223

theorem roots_geometric_progression_condition 
  (a b c : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = -a)
  (h2 : x1 * x2 + x2 * x3 + x1 * x3 = b)
  (h3 : x1 * x2 * x3 = -c)
  (h4 : x2^2 = x1 * x3) :
  a^3 * c = b^3 :=
sorry

end NUMINAMATH_GPT_roots_geometric_progression_condition_l1692_169223


namespace NUMINAMATH_GPT_system_of_equations_solution_l1692_169230

theorem system_of_equations_solution
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1)
  (h2 : x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2)
  (h3 : x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3)
  (h4 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4)
  (h5 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) :
  x1 = 1 ∧ x2 = -1 ∧ x3 = 1 ∧ x4 = -1 ∧ x5 = 1 :=
by {
  -- proof steps go here
  sorry
}

end NUMINAMATH_GPT_system_of_equations_solution_l1692_169230


namespace NUMINAMATH_GPT_gravel_per_truckload_l1692_169270

def truckloads_per_mile : ℕ := 3
def miles_day1 : ℕ := 4
def miles_day2 : ℕ := 2 * miles_day1 - 1
def total_paved_miles : ℕ := miles_day1 + miles_day2
def total_road_length : ℕ := 16
def miles_remaining : ℕ := total_road_length - total_paved_miles
def remaining_truckloads : ℕ := miles_remaining * truckloads_per_mile
def barrels_needed : ℕ := 6
def gravel_per_pitch : ℕ := 5
def P : ℚ := barrels_needed / remaining_truckloads
def G : ℚ := gravel_per_pitch * P

theorem gravel_per_truckload :
  G = 2 :=
by
  sorry

end NUMINAMATH_GPT_gravel_per_truckload_l1692_169270


namespace NUMINAMATH_GPT_weight_of_empty_carton_l1692_169218

theorem weight_of_empty_carton
    (half_full_carton_weight : ℕ)
    (full_carton_weight : ℕ)
    (h1 : half_full_carton_weight = 5)
    (h2 : full_carton_weight = 8) :
  full_carton_weight - 2 * (full_carton_weight - half_full_carton_weight) = 2 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_empty_carton_l1692_169218


namespace NUMINAMATH_GPT_determine_OP_squared_l1692_169217

-- Define the given conditions
variable (O P : Point) -- Points: center O and intersection point P
variable (r : ℝ) (AB CD : ℝ) (E F : Point) -- radius, lengths of chords, midpoints of chords
variable (OE OF : ℝ) -- Distances from center to midpoints of chords
variable (EF : ℝ) -- Distance between midpoints
variable (OP : ℝ) -- Distance from center to intersection point

-- Conditions as given
axiom circle_radius : r = 30
axiom chord_AB_length : AB = 40
axiom chord_CD_length : CD = 14
axiom distance_midpoints : EF = 15
axiom distance_OE : OE = 20
axiom distance_OF : OF = 29

-- The proof problem: determine that OP^2 = 733 given the conditions
theorem determine_OP_squared :
  OP^2 = 733 :=
sorry

end NUMINAMATH_GPT_determine_OP_squared_l1692_169217


namespace NUMINAMATH_GPT_inequality_positive_numbers_l1692_169243

theorem inequality_positive_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (x + 2 * y + 3 * z)) + (y / (y + 2 * z + 3 * x)) + (z / (z + 2 * x + 3 * y)) ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_positive_numbers_l1692_169243


namespace NUMINAMATH_GPT_ratio_x_y_l1692_169268

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 3) : x / y = 1 / 23 := by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l1692_169268


namespace NUMINAMATH_GPT_largest_and_smallest_values_quartic_real_roots_l1692_169272

noncomputable def function_y (a b x : ℝ) : ℝ :=
  (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

theorem largest_and_smallest_values (a b : ℝ) (h : a > b) :
  ∃ x y, function_y a b x = y^2 ∧ y = a ∧ y = b :=
by
  sorry

theorem quartic_real_roots (a b y : ℝ) (h₁ : a > b) (h₂ : y > b) (h₃ : y < a) :
  ∃ x₀ x₁ x₂ x₃, function_y a b x₀ = y^2 ∧ function_y a b x₁ = y^2 ∧ function_y a b x₂ = y^2 ∧ function_y a b x₃ = y^2 :=
by
  sorry

end NUMINAMATH_GPT_largest_and_smallest_values_quartic_real_roots_l1692_169272


namespace NUMINAMATH_GPT_distance_between_points_l1692_169287

theorem distance_between_points :
  let (x1, y1) := (1, 2)
  let (x2, y2) := (6, 5)
  let d := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  d = Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1692_169287


namespace NUMINAMATH_GPT_smallest_k_l1692_169232

theorem smallest_k (k : ℕ) : 
  (∀ x, x ∈ [13, 7, 3, 5] → k % x = 1) ∧ k > 1 → k = 1366 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l1692_169232


namespace NUMINAMATH_GPT_determine_k_for_circle_l1692_169283

theorem determine_k_for_circle (x y k : ℝ) (h : x^2 + 14*x + y^2 + 8*y - k = 0) (r : ℝ) :
  r = 5 → k = 40 :=
by
  intros radius_eq_five
  sorry

end NUMINAMATH_GPT_determine_k_for_circle_l1692_169283


namespace NUMINAMATH_GPT_inequality_solution_set_l1692_169249

theorem inequality_solution_set (x : ℝ) : (3 - 2 * x) * (x + 1) ≤ 0 ↔ (x < -1) ∨ (x ≥ 3 / 2) :=
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1692_169249


namespace NUMINAMATH_GPT_value_of_c_plus_d_l1692_169241

theorem value_of_c_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : a + d = 2) : c + d = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_c_plus_d_l1692_169241


namespace NUMINAMATH_GPT_m_over_n_eq_l1692_169273

variables (m n : ℝ)
variables (x y x1 y1 x2 y2 x0 y0 : ℝ)

-- Ellipse equation
axiom ellipse_eq : m * x^2 + n * y^2 = 1

-- Line equation
axiom line_eq : x + y = 1

-- Points M and N on the ellipse
axiom M_point : m * x1^2 + n * y1^2 = 1
axiom N_point : m * x2^2 + n * y2^2 = 1

-- Midpoint of MN is P
axiom P_midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

-- Slope of OP
axiom slope_OP : y0 / x0 = (Real.sqrt 2) / 2

theorem m_over_n_eq : m / n = (Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_m_over_n_eq_l1692_169273


namespace NUMINAMATH_GPT_find_P20_l1692_169206

theorem find_P20 (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^2 + a * x + b) 
  (h_condition : P 10 + P 30 = 40) : P 20 = -80 :=
by {
  -- Additional statements to structure the proof can go here
  sorry
}

end NUMINAMATH_GPT_find_P20_l1692_169206


namespace NUMINAMATH_GPT_find_a_plus_b_l1692_169256

variable (r a b : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions on the sequence
axiom seq_def : seq 0 = 4096
axiom seq_rule : ∀ n, seq (n + 1) = seq n * r

-- Given value
axiom r_value : r = 1 / 4

-- Given intermediate positions in the sequence
axiom seq_a : seq 3 = a
axiom seq_b : seq 4 = b
axiom seq_5 : seq 5 = 4

-- Theorem to prove
theorem find_a_plus_b : a + b = 80 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1692_169256


namespace NUMINAMATH_GPT_meat_purchase_l1692_169215

theorem meat_purchase :
  ∃ x y : ℕ, 16 * x = y + 25 ∧ 8 * x = y - 15 ∧ y / x = 11 :=
by
  sorry

end NUMINAMATH_GPT_meat_purchase_l1692_169215


namespace NUMINAMATH_GPT_find_radius_l1692_169239

theorem find_radius (abbc: ℝ) (adbd: ℝ) (bccc: ℝ) (dcdd: ℝ) (R: ℝ)
  (h1: abbc = 4) (h2: adbd = 4) (h3: bccc = 2) (h4: dcdd = 1) :
  R = 5 :=
sorry

end NUMINAMATH_GPT_find_radius_l1692_169239


namespace NUMINAMATH_GPT_find_subtracted_number_l1692_169292

-- Given conditions
def t : ℕ := 50
def k : ℕ := 122
def eq_condition (n : ℤ) : Prop := t = (5 / 9 : ℚ) * (k - n)

-- The proof problem proving the number subtracted from k is 32
theorem find_subtracted_number : eq_condition 32 :=
by
  -- implementation here will demonstrate that t = 50 implies the number is 32
  sorry

end NUMINAMATH_GPT_find_subtracted_number_l1692_169292


namespace NUMINAMATH_GPT_books_per_shelf_l1692_169291

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ)
    (h₁ : mystery_shelves = 5)
    (h₂ : picture_shelves = 3)
    (h₃ : total_books = 32) :
    (total_books / (mystery_shelves + picture_shelves) = 4) :=
by
    sorry

end NUMINAMATH_GPT_books_per_shelf_l1692_169291


namespace NUMINAMATH_GPT_percentage_loss_l1692_169235

theorem percentage_loss (selling_price_with_loss : ℝ)
    (desired_selling_price_for_profit : ℝ)
    (profit_percentage : ℝ) (actual_selling_price : ℝ)
    (calculated_loss_percentage : ℝ) :
    selling_price_with_loss = 16 →
    desired_selling_price_for_profit = 21.818181818181817 →
    profit_percentage = 20 →
    actual_selling_price = 18.181818181818182 →
    calculated_loss_percentage = 12 → 
    calculated_loss_percentage = (actual_selling_price - selling_price_with_loss) / actual_selling_price * 100 := 
sorry

end NUMINAMATH_GPT_percentage_loss_l1692_169235


namespace NUMINAMATH_GPT_original_design_ratio_built_bridge_ratio_l1692_169244

-- Definitions
variables (v1 v2 r1 r2 : ℝ)

-- Conditions as per the problem
def original_height_relation : Prop := v1 = 3 * v2
def built_radius_relation : Prop := r2 = 2 * r1

-- Prove the required ratios
theorem original_design_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v1 / r1 = 3 / 4) := sorry

theorem built_bridge_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v2 / r2 = 1 / 8) := sorry

end NUMINAMATH_GPT_original_design_ratio_built_bridge_ratio_l1692_169244


namespace NUMINAMATH_GPT_no_solution_xy_in_nat_star_l1692_169285

theorem no_solution_xy_in_nat_star (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
by
  -- The proof would go here, but we'll leave it out for now.
  sorry

end NUMINAMATH_GPT_no_solution_xy_in_nat_star_l1692_169285


namespace NUMINAMATH_GPT_calc_delta_l1692_169258

noncomputable def delta (a b : ℝ) : ℝ :=
  (a^2 + b^2) / (1 + a * b)

-- Definition of the main problem as a Lean 4 statement
theorem calc_delta (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 4 > 0) :
  delta (delta 2 3) 4 = 6661 / 2891 :=
by
  sorry

end NUMINAMATH_GPT_calc_delta_l1692_169258


namespace NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_five_primes_l1692_169211

theorem arithmetic_mean_of_reciprocals_of_first_five_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7 + 1 / 11) / 5 = 2927 / 11550 := 
sorry

end NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_of_first_five_primes_l1692_169211


namespace NUMINAMATH_GPT_minimum_value_l1692_169299

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x^2 - x else
         if h : 1 < x ∧ x ≤ 2 then -2 * (x - 1)^2 + 6 * (x - 1) - 5
         else 0 -- extend as appropriate outside given ranges

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem minimum_value (x_1 x_2 : ℝ) (h1 : 1 < x_1 ∧ x_1 ≤ 2) : 
  (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1692_169299


namespace NUMINAMATH_GPT_contradiction_assumption_l1692_169262

theorem contradiction_assumption (x y : ℝ) (h1 : x > y) : ¬ (x^3 ≤ y^3) := 
by
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l1692_169262


namespace NUMINAMATH_GPT_period_pi_omega_l1692_169200

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  3 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 4 * (Real.cos (ω * x))^2

theorem period_pi_omega (ω : ℝ) (hω : ω > 0) (period_condition : ∀ x, f x ω = f (x + π) ω)
  (theta : ℝ) (h_f_theta : f theta ω = 1 / 2) :
  f (theta + π / 2) ω + f (theta - π / 4) ω = -13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_period_pi_omega_l1692_169200


namespace NUMINAMATH_GPT_dog_group_division_l1692_169294

theorem dog_group_division:
  let total_dogs := 12
  let group1_size := 4
  let group2_size := 5
  let group3_size := 3
  let Rocky_in_group1 := true
  let Bella_in_group2 := true
  (total_dogs == 12 ∧ group1_size == 4 ∧ group2_size == 5 ∧ group3_size == 3 ∧ Rocky_in_group1 ∧ Bella_in_group2) →
  (∃ ways: ℕ, ways = 4200)
  :=
  sorry

end NUMINAMATH_GPT_dog_group_division_l1692_169294


namespace NUMINAMATH_GPT_quadratic_roots_and_signs_l1692_169274

theorem quadratic_roots_and_signs :
  (∃ x1 x2 : ℝ, (x1^2 - 13*x1 + 40 = 0) ∧ (x2^2 - 13*x2 + 40 = 0) ∧ x1 = 5 ∧ x2 = 8 ∧ 0 < x1 ∧ 0 < x2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_and_signs_l1692_169274


namespace NUMINAMATH_GPT_people_who_came_to_game_l1692_169238

def total_seats : Nat := 92
def people_with_banners : Nat := 38
def empty_seats : Nat := 45

theorem people_who_came_to_game : (total_seats - empty_seats = 47) :=
by 
  sorry

end NUMINAMATH_GPT_people_who_came_to_game_l1692_169238


namespace NUMINAMATH_GPT_minimum_value_polynomial_l1692_169252

def polynomial (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + 4 * y^2 + 12 * x + 25

theorem minimum_value_polynomial : ∃ (m : ℝ), (∀ (x y : ℝ), polynomial x y ≥ m) ∧ m = 16 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_polynomial_l1692_169252


namespace NUMINAMATH_GPT_pq_sum_of_harmonic_and_geometric_sequences_l1692_169202

theorem pq_sum_of_harmonic_and_geometric_sequences
  (x y z : ℝ)
  (h1 : (1 / x - 1 / y) / (1 / y - 1 / z) = 1)
  (h2 : 3 * x * y = 7 * z) :
  ∃ p q : ℕ, (Nat.gcd p q = 1) ∧ p + q = 79 :=
by
  sorry

end NUMINAMATH_GPT_pq_sum_of_harmonic_and_geometric_sequences_l1692_169202


namespace NUMINAMATH_GPT_dart_lands_in_center_square_l1692_169222

theorem dart_lands_in_center_square (s : ℝ) (h : 0 < s) :
  let center_square_area := (s / 2) ^ 2
  let triangle_area := 1 / 2 * (s / 2) ^ 2
  let total_triangle_area := 4 * triangle_area
  let total_board_area := center_square_area + total_triangle_area
  let probability := center_square_area / total_board_area
  probability = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_dart_lands_in_center_square_l1692_169222


namespace NUMINAMATH_GPT_gcd_50403_40302_l1692_169213

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_50403_40302_l1692_169213


namespace NUMINAMATH_GPT_range_of_function_l1692_169236

noncomputable def function_range (x : ℝ) : ℝ :=
    (1 / 2) ^ (-x^2 + 2 * x)

theorem range_of_function : 
    (Set.range function_range) = Set.Ici (1 / 2) :=
by
    sorry

end NUMINAMATH_GPT_range_of_function_l1692_169236


namespace NUMINAMATH_GPT_minor_axis_length_l1692_169278

theorem minor_axis_length {x y : ℝ} (h : x^2 / 16 + y^2 / 9 = 1) : 6 = 6 :=
by
  sorry

end NUMINAMATH_GPT_minor_axis_length_l1692_169278


namespace NUMINAMATH_GPT_initial_sugar_amount_l1692_169228

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end NUMINAMATH_GPT_initial_sugar_amount_l1692_169228


namespace NUMINAMATH_GPT_upper_bound_exists_l1692_169205

theorem upper_bound_exists (U : ℤ) :
  (∀ n : ℤ, 1 < 4 * n + 7 ∧ 4 * n + 7 < U) →
  (∃ n_min n_max : ℤ, n_max = n_min + 29 ∧ 4 * n_max + 7 < U ∧ 4 * n_min + 7 > 1) →
  (U = 120) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_upper_bound_exists_l1692_169205


namespace NUMINAMATH_GPT_net_profit_calc_l1692_169293

theorem net_profit_calc:
  ∃ (x y : ℕ), x + y = 25 ∧ 1700 * x + 1800 * y = 44000 ∧ 2400 * x + 2600 * y = 63000 := by
  sorry

end NUMINAMATH_GPT_net_profit_calc_l1692_169293


namespace NUMINAMATH_GPT_juice_left_l1692_169221

theorem juice_left (total consumed : ℚ) (h_total : total = 1) (h_consumed : consumed = 4 / 6) :
  total - consumed = 2 / 6 ∨ total - consumed = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_juice_left_l1692_169221


namespace NUMINAMATH_GPT_problem_b_amount_l1692_169240

theorem problem_b_amount (a b : ℝ) (h1 : a + b = 1210) (h2 : (4/5) * a = (2/3) * b) : b = 453.75 :=
sorry

end NUMINAMATH_GPT_problem_b_amount_l1692_169240


namespace NUMINAMATH_GPT_bucket_weight_l1692_169264

theorem bucket_weight (x y p q : ℝ) 
  (h1 : x + (3 / 4) * y = p) 
  (h2 : x + (1 / 3) * y = q) :
  x + (5 / 6) * y = (6 * p - q) / 5 :=
sorry

end NUMINAMATH_GPT_bucket_weight_l1692_169264


namespace NUMINAMATH_GPT_vector_magnitude_parallel_l1692_169226

theorem vector_magnitude_parallel (x : ℝ) 
  (h1 : 4 / x = 2 / 1) :
  ( Real.sqrt ((4 + x) ^ 2 + (2 + 1) ^ 2) ) = 3 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_vector_magnitude_parallel_l1692_169226


namespace NUMINAMATH_GPT_harmon_high_school_proof_l1692_169257

noncomputable def harmon_high_school : Prop :=
  ∃ (total_players players_physics players_both players_chemistry : ℕ),
    total_players = 18 ∧
    players_physics = 10 ∧
    players_both = 3 ∧
    players_chemistry = (total_players - players_physics + players_both)

theorem harmon_high_school_proof : harmon_high_school :=
  sorry

end NUMINAMATH_GPT_harmon_high_school_proof_l1692_169257


namespace NUMINAMATH_GPT_expression_not_equal_33_l1692_169276

theorem expression_not_equal_33 (x y : ℤ) :
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end NUMINAMATH_GPT_expression_not_equal_33_l1692_169276


namespace NUMINAMATH_GPT_cubic_of_cubic_roots_correct_l1692_169225

variable (a b c : ℝ) (α β γ : ℝ)

-- Vieta's formulas conditions
axiom vieta1 : α + β + γ = -a
axiom vieta2 : α * β + β * γ + γ * α = b
axiom vieta3 : α * β * γ = -c

-- Define the polynomial whose roots are α³, β³, and γ³
def cubic_of_cubic_roots (x : ℝ) : ℝ :=
  x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3

-- Prove that this polynomial has α³, β³, γ³ as roots
theorem cubic_of_cubic_roots_correct :
  ∀ x : ℝ, cubic_of_cubic_roots a b c x = 0 ↔ (x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
sorry

end NUMINAMATH_GPT_cubic_of_cubic_roots_correct_l1692_169225


namespace NUMINAMATH_GPT_intersection_M_N_l1692_169250

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | x ≥ -1 / 3}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 / 3 ≤ x ∧ x < 4} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1692_169250


namespace NUMINAMATH_GPT_circle_area_increase_l1692_169261

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 125 := 
by {
  -- The proof will be written here.
  sorry
}

end NUMINAMATH_GPT_circle_area_increase_l1692_169261


namespace NUMINAMATH_GPT_solve_expression_l1692_169237

theorem solve_expression (x y z : ℚ)
  (h1 : 2 * x + 3 * y + z = 20)
  (h2 : x + 2 * y + 3 * z = 26)
  (h3 : 3 * x + y + 2 * z = 29) :
  12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = (computed_value : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1692_169237
