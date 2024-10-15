import Mathlib

namespace NUMINAMATH_GPT_domain_and_monotone_l134_13429

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem domain_and_monotone :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → ∃ y, f x = y) ∧
  ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f x1 < f x2 :=
by
  sorry

end NUMINAMATH_GPT_domain_and_monotone_l134_13429


namespace NUMINAMATH_GPT_find_N_l134_13405

theorem find_N :
  ∃ N : ℕ,
  (5 + 6 + 7 + 8 + 9) / 5 = (2005 + 2006 + 2007 + 2008 + 2009) / (N : ℝ) ∧ N = 1433 :=
sorry

end NUMINAMATH_GPT_find_N_l134_13405


namespace NUMINAMATH_GPT_tan_expression_value_l134_13428

noncomputable def sequence_properties (a b : ℕ → ℝ) :=
  (a 0 * a 5 * a 10 = -3 * Real.sqrt 3) ∧
  (b 0 + b 5 + b 10 = 7 * Real.pi) ∧
  (∀ n, a (n + 1) = a n * a 1) ∧
  (∀ n, b (n + 1) = b n + (b 1 - b 0))

theorem tan_expression_value (a b : ℕ → ℝ) (h : sequence_properties a b) :
  Real.tan (b 2 + b 8) / (1 - a 3 * a 7) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_expression_value_l134_13428


namespace NUMINAMATH_GPT_single_intersection_point_l134_13416

theorem single_intersection_point (k : ℝ) :
  (∃! x : ℝ, x^2 - 2 * x - k = 0) ↔ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_single_intersection_point_l134_13416


namespace NUMINAMATH_GPT_distance_covered_at_40_kmph_l134_13491

theorem distance_covered_at_40_kmph (x : ℝ) : 
  (x / 40 + (250 - x) / 60 = 5.4) → (x = 148) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_covered_at_40_kmph_l134_13491


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l134_13463

theorem domain_of_sqrt_function : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = {x : ℝ | 1 - x ≥ 0 ∧ x - Real.sqrt (1 - x) ≥ 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l134_13463


namespace NUMINAMATH_GPT_shadow_length_of_flagpole_l134_13450

theorem shadow_length_of_flagpole :
  ∀ (S : ℝ), (18 : ℝ) / S = (22 : ℝ) / 55 → S = 45 :=
by
  intro S h
  sorry

end NUMINAMATH_GPT_shadow_length_of_flagpole_l134_13450


namespace NUMINAMATH_GPT_expected_heads_of_fair_coin_l134_13454

noncomputable def expected_heads (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expected_heads_of_fair_coin :
  expected_heads 5 0.5 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_heads_of_fair_coin_l134_13454


namespace NUMINAMATH_GPT_individual_max_food_l134_13474

/-- Given a minimum number of guests and a total amount of food consumed,
    we want to find the maximum amount of food an individual guest could have consumed. -/
def total_food : ℝ := 319
def min_guests : ℝ := 160
def max_food_per_guest : ℝ := 1.99

theorem individual_max_food :
  total_food / min_guests <= max_food_per_guest := by
  sorry

end NUMINAMATH_GPT_individual_max_food_l134_13474


namespace NUMINAMATH_GPT_largest_angle_in_isosceles_triangle_l134_13498

-- Definitions of the conditions from the problem
def isosceles_triangle (A B C : ℕ) : Prop :=
  A = B ∨ B = C ∨ A = C

def angle_opposite_equal_side (θ : ℕ) : Prop :=
  θ = 50

-- The proof problem statement
theorem largest_angle_in_isosceles_triangle (A B C : ℕ) (θ : ℕ)
  : isosceles_triangle A B C → angle_opposite_equal_side θ → ∃ γ, γ = 80 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_isosceles_triangle_l134_13498


namespace NUMINAMATH_GPT_boats_distance_one_minute_before_collision_l134_13499

noncomputable def distance_between_boats_one_minute_before_collision
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := speed_boat1 + speed_boat2
  let relative_speed_per_minute := relative_speed / 60
  let time_to_collide := initial_distance / relative_speed_per_minute
  let distance_one_minute_before := initial_distance - (relative_speed_per_minute * (time_to_collide - 1))
  distance_one_minute_before

theorem boats_distance_one_minute_before_collision :
  distance_between_boats_one_minute_before_collision 5 21 20 = 0.4333 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_boats_distance_one_minute_before_collision_l134_13499


namespace NUMINAMATH_GPT_intersect_complement_eq_l134_13478

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}
def comp_B : Set ℕ := U \ B

theorem intersect_complement_eq :
  A ∩ comp_B = {4, 5} := by
  sorry

end NUMINAMATH_GPT_intersect_complement_eq_l134_13478


namespace NUMINAMATH_GPT_binom_divisibility_by_prime_l134_13415

-- Given definitions
variable (p k : ℕ) (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2)

-- Main theorem statement
theorem binom_divisibility_by_prime
  (hp : Nat.Prime p) (hk1 : 2 ≤ k) (hk2 : k ≤ p - 2) :
  Nat.choose (p - k + 1) k - Nat.choose (p - k - 1) (k - 2) ≡ 0 [MOD p] :=
sorry

end NUMINAMATH_GPT_binom_divisibility_by_prime_l134_13415


namespace NUMINAMATH_GPT_sum_faces_edges_vertices_triangular_prism_l134_13446

-- Given conditions for triangular prism:
def triangular_prism_faces : Nat := 2 + 3  -- 2 triangular faces and 3 rectangular faces
def triangular_prism_edges : Nat := 3 + 3 + 3  -- 3 top edges, 3 bottom edges, 3 connecting edges
def triangular_prism_vertices : Nat := 3 + 3  -- 3 vertices on the top base, 3 on the bottom base

-- Proof statement for the sum of the faces, edges, and vertices of a triangular prism
theorem sum_faces_edges_vertices_triangular_prism : 
  triangular_prism_faces + triangular_prism_edges + triangular_prism_vertices = 20 := by
  sorry

end NUMINAMATH_GPT_sum_faces_edges_vertices_triangular_prism_l134_13446


namespace NUMINAMATH_GPT_expected_pairs_correct_l134_13430

-- Define the total number of cards in the deck.
def total_cards : ℕ := 52

-- Define the number of black cards in the deck.
def black_cards : ℕ := 26

-- Define the number of red cards in the deck.
def red_cards : ℕ := 26

-- Define the expected number of pairs of adjacent cards such that one is black and the other is red.
def expected_adjacent_pairs := 52 * (26 / 51)

-- Prove that the expected_adjacent_pairs is equal to 1352 / 51.
theorem expected_pairs_correct : expected_adjacent_pairs = 1352 / 51 := 
by
  have expected_adjacent_pairs_simplified : 52 * (26 / 51) = (1352 / 51) := 
    by sorry
  exact expected_adjacent_pairs_simplified

end NUMINAMATH_GPT_expected_pairs_correct_l134_13430


namespace NUMINAMATH_GPT_minimize_q_l134_13473

noncomputable def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimize_q : ∃ x : ℝ, q x = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_q_l134_13473


namespace NUMINAMATH_GPT_polygon_sides_l134_13445

theorem polygon_sides (n : ℕ) (h : 3 * n * (n * (n - 3)) = 300) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l134_13445


namespace NUMINAMATH_GPT_perp_lines_iff_m_values_l134_13419

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end NUMINAMATH_GPT_perp_lines_iff_m_values_l134_13419


namespace NUMINAMATH_GPT_minimum_value_fraction_l134_13456

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

/-- Given that the function f(x) = log_a(4x-3) + 1 (where a > 0 and a ≠ 1) has a fixed point A(m, n), 
if for any positive numbers x and y, mx + ny = 3, 
then the minimum value of 1/(x+1) + 1/y is 1. -/
theorem minimum_value_fraction (a x y : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (hx : x + y = 3) : 
  (1 / (x + 1) + 1 / y) = 1 := 
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l134_13456


namespace NUMINAMATH_GPT_range_of_a_h_diff_l134_13403

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem range_of_a (a : ℝ) (h : a < 0) : (∀ x, 0 < x ∧ x < Real.log 3 → 
  (a * x - 1) / x < 0 ∧ Real.exp x + a ≠ 0 ∧ (a ≤ -3)) :=
sorry

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + Real.log x

theorem h_diff (a : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) : 
    x1 * x2 = 1/2 ∧ h a x1 - h a x2 > 3/4 - Real.log 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_h_diff_l134_13403


namespace NUMINAMATH_GPT_problem_statement_l134_13431

theorem problem_statement {m n : ℝ} 
  (h1 : (n + 2 * m) / (1 + m ^ 2) = -1 / 2) 
  (h2 : -(1 + n) + 2 * (m + 2) = 0) : 
  (m / n = -1) := 
sorry

end NUMINAMATH_GPT_problem_statement_l134_13431


namespace NUMINAMATH_GPT_Yvonne_laps_l134_13437

-- Definitions of the given conditions
def laps_swim_by_Yvonne (l_y : ℕ) : Prop := 
  ∃ l_s l_j, 
  l_s = l_y / 2 ∧ 
  l_j = 3 * l_s ∧ 
  l_j = 15

-- Theorem statement
theorem Yvonne_laps (l_y : ℕ) (h : laps_swim_by_Yvonne l_y) : l_y = 10 :=
sorry

end NUMINAMATH_GPT_Yvonne_laps_l134_13437


namespace NUMINAMATH_GPT_Davante_boys_count_l134_13459

def days_in_week := 7
def friends (days : Nat) := days * 2
def girls := 3
def boys (total_friends girls : Nat) := total_friends - girls

theorem Davante_boys_count :
  boys (friends days_in_week) girls = 11 :=
  by
    sorry

end NUMINAMATH_GPT_Davante_boys_count_l134_13459


namespace NUMINAMATH_GPT_compare_f_values_l134_13404

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f 0.5 ∧ f 0.5 < f 0.6 :=
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_compare_f_values_l134_13404


namespace NUMINAMATH_GPT_log_inequality_l134_13496

theorem log_inequality (a x y : ℝ) (ha : 0 < a) (ha_lt_1 : a < 1) 
(h : x^2 + y = 0) : 
  Real.log (a^x + a^y) / Real.log a ≤ Real.log 2 / Real.log a + 1 / 8 :=
sorry

end NUMINAMATH_GPT_log_inequality_l134_13496


namespace NUMINAMATH_GPT_smallest_positive_integer_exists_l134_13441

theorem smallest_positive_integer_exists
    (x : ℕ) :
    (x % 7 = 2) ∧
    (x % 4 = 3) ∧
    (x % 6 = 1) →
    x = 135 :=
by
    sorry

end NUMINAMATH_GPT_smallest_positive_integer_exists_l134_13441


namespace NUMINAMATH_GPT_fraction_of_crop_brought_to_BC_l134_13422

/-- Consider a kite-shaped field with sides AB = 120 m, BC = CD = 80 m, DA = 120 m.
    The angle between sides AB and BC is 120°, and between sides CD and DA is also 120°.
    Prove that the fraction of the crop brought to the longest side BC is 1/2. -/
theorem fraction_of_crop_brought_to_BC :
  ∀ (AB BC CD DA : ℝ) (α β : ℝ),
  AB = 120 ∧ BC = 80 ∧ CD = 80 ∧ DA = 120 ∧ α = 120 ∧ β = 120 →
  ∃ (frac : ℝ), frac = 1 / 2 :=
by
  intros AB BC CD DA α β h
  sorry

end NUMINAMATH_GPT_fraction_of_crop_brought_to_BC_l134_13422


namespace NUMINAMATH_GPT_factories_checked_by_second_group_l134_13433

theorem factories_checked_by_second_group 
(T : ℕ) (G1 : ℕ) (R : ℕ) 
(hT : T = 169) 
(hG1 : G1 = 69) 
(hR : R = 48) : 
T - (G1 + R) = 52 :=
by {
  sorry
}

end NUMINAMATH_GPT_factories_checked_by_second_group_l134_13433


namespace NUMINAMATH_GPT_part1_part2_part3_l134_13421

open Real

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

def M (m : ℝ) : Point := ⟨m - 2, 2 * m - 7⟩
def N (n : ℝ) : Point := ⟨n, 3⟩

-- Part 1
theorem part1 : 
  (M (7 / 2)).y = 0 ∧ (M (7 / 2)).x = 3 / 2 :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : abs (m - 2) = abs (2 * m - 7) → (m = 5 ∨ m = 3) :=
by
  sorry

-- Part 3
theorem part3 (m n : ℝ) : abs ((M m).y - 3) = 2 ∧ (M m).x = n - 2 → (n = 4 ∨ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l134_13421


namespace NUMINAMATH_GPT_find_larger_number_l134_13443

theorem find_larger_number (x y : ℤ) (h1 : 4 * y = 3 * x) (h2 : y - x = 12) : y = -36 := 
by sorry

end NUMINAMATH_GPT_find_larger_number_l134_13443


namespace NUMINAMATH_GPT_perfect_square_transformation_l134_13475

theorem perfect_square_transformation (a : ℤ) :
  (∃ x y : ℤ, x^2 + a = y^2) ↔ 
  ∃ α β : ℤ, α * β = a ∧ (α % 2 = β % 2) ∧ 
  ∃ x y : ℤ, x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_transformation_l134_13475


namespace NUMINAMATH_GPT_rectangle_width_l134_13455

theorem rectangle_width (width : ℝ) : 
  ∃ w, w = 14 ∧
  (∀ length : ℝ, length = 10 →
  (2 * (length + width) = 3 * 16)) → 
  width = w :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l134_13455


namespace NUMINAMATH_GPT_does_not_pass_first_quadrant_l134_13447

def linear_function (x : ℝ) : ℝ := -3 * x - 2

def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem does_not_pass_first_quadrant : ∀ (x : ℝ), ¬ in_first_quadrant x (linear_function x) := 
sorry

end NUMINAMATH_GPT_does_not_pass_first_quadrant_l134_13447


namespace NUMINAMATH_GPT_probability_mask_with_ear_loops_l134_13442

-- Definitions from the conditions
def production_ratio_regular : ℝ := 0.8
def production_ratio_surgical : ℝ := 0.2
def proportion_ear_loops_regular : ℝ := 0.1
def proportion_ear_loops_surgical : ℝ := 0.2

-- Theorem statement based on the translated proof problem
theorem probability_mask_with_ear_loops :
  production_ratio_regular * proportion_ear_loops_regular +
  production_ratio_surgical * proportion_ear_loops_surgical = 0.12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_mask_with_ear_loops_l134_13442


namespace NUMINAMATH_GPT_inequality_solution_l134_13452

variable {α : Type*} [LinearOrderedField α]
variable (a b x : α)

theorem inequality_solution (h1 : a < 0) (h2 : b = -a) :
  0 < x ∧ x < 1 ↔ ax^2 + bx > 0 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l134_13452


namespace NUMINAMATH_GPT_zoo_gorillas_sent_6_l134_13489

theorem zoo_gorillas_sent_6 (G : ℕ) : 
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  after_adding_meerkats = final_animals → G = 6 := 
by
  intros
  let initial_animals := 68
  let after_sending_gorillas := initial_animals - G
  let after_adopting_hippopotamus := after_sending_gorillas + 1
  let after_taking_rhinos := after_adopting_hippopotamus + 3
  let after_birth_lion_cubs := after_taking_rhinos + 8
  let after_adding_meerkats := after_birth_lion_cubs + (2 * 8)
  let final_animals := 90
  sorry

end NUMINAMATH_GPT_zoo_gorillas_sent_6_l134_13489


namespace NUMINAMATH_GPT_cos_even_function_l134_13462

theorem cos_even_function : ∀ x : ℝ, Real.cos (-x) = Real.cos x := 
by 
  sorry

end NUMINAMATH_GPT_cos_even_function_l134_13462


namespace NUMINAMATH_GPT_range_of_m_l134_13477

variables (f : ℝ → ℝ) (m : ℝ)

-- Assume f is a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x > f y

-- Assume f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem stating the main condition and the implication
theorem range_of_m (h_decreasing : is_decreasing f) (h_odd : is_odd f) (h_condition : f (m - 1) + f (2 * m - 1) > 0) : m > 2 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l134_13477


namespace NUMINAMATH_GPT_arithmetic_seq_a5_l134_13414

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_a5 (h1 : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a5_l134_13414


namespace NUMINAMATH_GPT_unrepresentable_integers_l134_13408

theorem unrepresentable_integers :
    {n : ℕ | ∀ a b : ℕ, a > 0 → b > 0 → n ≠ (a * (b + 1) + (a + 1) * b) / (b * (b + 1)) } =
    {1} ∪ {n | ∃ m : ℕ, n = 2^m + 2} :=
by
    sorry

end NUMINAMATH_GPT_unrepresentable_integers_l134_13408


namespace NUMINAMATH_GPT_minimum_value_l134_13434

theorem minimum_value : 
  ∀ a b : ℝ, 0 < a → 0 < b → a + 2 * b = 3 → (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l134_13434


namespace NUMINAMATH_GPT_line_intercepts_l134_13479

theorem line_intercepts (x y : ℝ) (P : ℝ × ℝ) (h1 : P = (1, 4)) (h2 : ∃ k : ℝ, (x + y = k ∨ 4 * x - y = 0) ∧ 
  ∃ intercepts_p : ℝ × ℝ, intercepts_p = (k / 2, k / 2)) :
  ∃ k : ℝ, (x + y - k = 0 ∧ k = 5) ∨ (4 * x - y = 0) :=
sorry

end NUMINAMATH_GPT_line_intercepts_l134_13479


namespace NUMINAMATH_GPT_evaluate_expression_l134_13406

theorem evaluate_expression (m n : ℝ) (h : 4 * m - 4 + n = 2) : 
  (m * (-2)^2 - 2 * (-2) + n = 10) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l134_13406


namespace NUMINAMATH_GPT_janice_items_l134_13469

theorem janice_items : 
  ∃ a b c : ℕ, 
    a + b + c = 60 ∧ 
    15 * a + 400 * b + 500 * c = 6000 ∧ 
    a = 50 := 
by 
  sorry

end NUMINAMATH_GPT_janice_items_l134_13469


namespace NUMINAMATH_GPT_trigonometric_identity_l134_13485

variable {α : Real}
variable (h : Real.cos α = -2 / 3)

theorem trigonometric_identity : 
  (Real.cos α = -2 / 3) → 
  (Real.cos (4 * Real.pi - α) * Real.sin (-α) / 
  (Real.sin (Real.pi / 2 + α) * Real.tan (Real.pi - α)) = Real.cos α) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l134_13485


namespace NUMINAMATH_GPT_missing_dimension_of_carton_l134_13471

theorem missing_dimension_of_carton (x : ℕ) 
  (h1 : 0 < x)
  (h2 : 0 < 48)
  (h3 : 0 < 60)
  (h4 : 0 < 8)
  (h5 : 0 < 6)
  (h6 : 0 < 5)
  (h7 : (x * 48 * 60) / (8 * 6 * 5) = 300) : 
  x = 25 :=
by
  sorry

end NUMINAMATH_GPT_missing_dimension_of_carton_l134_13471


namespace NUMINAMATH_GPT_parabola_vertex_point_l134_13401

theorem parabola_vertex_point (a b c : ℝ) 
    (h_vertex : ∃ a b c : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + c) 
    (h_vertex_coord : ∃ (h k : ℝ), h = 3 ∧ k = -5) 
    (h_pass : ∃ (x y : ℝ), x = 0 ∧ y = -2) :
    c = -2 := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_point_l134_13401


namespace NUMINAMATH_GPT_parallel_lines_iff_a_eq_neg3_l134_13423

theorem parallel_lines_iff_a_eq_neg3 (a : ℝ) :
  (∀ x y : ℝ, a * x + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 ≠ 0) ↔ a = -3 :=
sorry

end NUMINAMATH_GPT_parallel_lines_iff_a_eq_neg3_l134_13423


namespace NUMINAMATH_GPT_candy_bars_weeks_l134_13461

theorem candy_bars_weeks (buy_per_week : ℕ) (eat_per_4_weeks : ℕ) (saved_candies : ℕ) (weeks_passed : ℕ) :
  (buy_per_week = 2) →
  (eat_per_4_weeks = 1) →
  (saved_candies = 28) →
  (weeks_passed = 4 * (saved_candies / (4 * buy_per_week - eat_per_4_weeks))) →
  weeks_passed = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_candy_bars_weeks_l134_13461


namespace NUMINAMATH_GPT_midpoint_of_hyperbola_l134_13470

theorem midpoint_of_hyperbola :
  ∃ (A B : ℝ × ℝ),
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    (∃ (M : ℝ × ℝ), (M = (-1, -4)) ∧ 
      (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) ∧
    ¬(∃ (A B : ℝ × ℝ), (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧ 
      (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) ∨ 
      ((A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 2) ∨ 
      ((A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 3)) :=
sorry

end NUMINAMATH_GPT_midpoint_of_hyperbola_l134_13470


namespace NUMINAMATH_GPT_Pam_has_740_fruits_l134_13472

/-
Define the given conditions.
-/
def Gerald_apple_bags : ℕ := 5
def apples_per_Gerald_bag : ℕ := 30
def Gerald_orange_bags : ℕ := 4
def oranges_per_Gerald_bag : ℕ := 25

def Pam_apple_bags : ℕ := 6
def apples_per_Pam_bag : ℕ := 3 * apples_per_Gerald_bag
def Pam_orange_bags : ℕ := 4
def oranges_per_Pam_bag : ℕ := 2 * oranges_per_Gerald_bag

/-
Proving the total number of apples and oranges Pam has.
-/
def total_fruits_Pam : ℕ :=
    Pam_apple_bags * apples_per_Pam_bag + Pam_orange_bags * oranges_per_Pam_bag

theorem Pam_has_740_fruits : total_fruits_Pam = 740 := by
  sorry

end NUMINAMATH_GPT_Pam_has_740_fruits_l134_13472


namespace NUMINAMATH_GPT_cannot_determine_letters_afternoon_l134_13444

theorem cannot_determine_letters_afternoon
  (emails_morning : ℕ) (letters_morning : ℕ)
  (emails_afternoon : ℕ) (letters_afternoon : ℕ)
  (h1 : emails_morning = 10)
  (h2 : letters_morning = 12)
  (h3 : emails_afternoon = 3)
  (h4 : emails_morning = emails_afternoon + 7) :
  ¬∃ (letters_afternoon : ℕ), true := 
sorry

end NUMINAMATH_GPT_cannot_determine_letters_afternoon_l134_13444


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l134_13417

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (y^2 / 9 - x^2 / 4 = 1 →
  (y = (3 / 2) * x ∨ y = - (3 / 2) * x)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l134_13417


namespace NUMINAMATH_GPT_feeding_sequences_count_l134_13486

def num_feeding_sequences (num_pairs : ℕ) : ℕ :=
  num_pairs * num_pairs.pred * num_pairs.pred * num_pairs.pred.pred *
  num_pairs.pred.pred * num_pairs.pred.pred.pred * num_pairs.pred.pred.pred *
  1 * 1

theorem feeding_sequences_count (num_pairs : ℕ) (h : num_pairs = 5) :
  num_feeding_sequences num_pairs = 5760 := 
by
  rw [h]
  unfold num_feeding_sequences
  norm_num
  sorry

end NUMINAMATH_GPT_feeding_sequences_count_l134_13486


namespace NUMINAMATH_GPT_volunteer_comprehensive_score_l134_13451

theorem volunteer_comprehensive_score :
  let written_score := 90
  let trial_score := 94
  let interview_score := 92
  let written_weight := 0.30
  let trial_weight := 0.50
  let interview_weight := 0.20
  (written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight = 92.4) := by
  sorry

end NUMINAMATH_GPT_volunteer_comprehensive_score_l134_13451


namespace NUMINAMATH_GPT_find_f_2009_l134_13458

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom cond2 : f 1 = 2

theorem find_f_2009 : f 2009 = 2 := by
  sorry

end NUMINAMATH_GPT_find_f_2009_l134_13458


namespace NUMINAMATH_GPT_diagonals_of_polygon_l134_13492

theorem diagonals_of_polygon (f : ℕ → ℕ) (k : ℕ) (h_k : k ≥ 3) : f (k + 1) = f k + (k - 1) :=
sorry

end NUMINAMATH_GPT_diagonals_of_polygon_l134_13492


namespace NUMINAMATH_GPT_total_votes_cast_l134_13480

theorem total_votes_cast (b_votes c_votes total_votes : ℕ)
  (h1 : b_votes = 48)
  (h2 : c_votes = 35)
  (h3 : b_votes = (4 * total_votes) / 15) :
  total_votes = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l134_13480


namespace NUMINAMATH_GPT_factorable_iff_some_even_b_l134_13497

open Int

theorem factorable_iff_some_even_b (b : ℤ) :
  (∃ m n p q : ℤ,
    (35 : ℤ) = m * p ∧
    (35 : ℤ) = n * q ∧
    b = m * q + n * p) →
  (∃ k : ℤ, b = 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_factorable_iff_some_even_b_l134_13497


namespace NUMINAMATH_GPT_factorize_3x2_minus_3y2_l134_13412

theorem factorize_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_GPT_factorize_3x2_minus_3y2_l134_13412


namespace NUMINAMATH_GPT_right_triangle_area_l134_13487

theorem right_triangle_area (a b : ℝ) (ha : a = 3) (hb : b = 5) : 
  (1 / 2) * a * b = 7.5 := 
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_right_triangle_area_l134_13487


namespace NUMINAMATH_GPT_raviraj_distance_home_l134_13495

theorem raviraj_distance_home :
  let origin := (0, 0)
  let after_south := (0, -20)
  let after_west := (-10, -20)
  let after_north := (-10, 0)
  let final_pos := (-30, 0)
  Real.sqrt ((final_pos.1 - origin.1)^2 + (final_pos.2 - origin.2)^2) = 30 :=
by
  sorry

end NUMINAMATH_GPT_raviraj_distance_home_l134_13495


namespace NUMINAMATH_GPT_left_handed_like_jazz_l134_13483

theorem left_handed_like_jazz (total_people left_handed like_jazz right_handed_dislike_jazz : ℕ)
    (h1 : total_people = 30)
    (h2 : left_handed = 12)
    (h3 : like_jazz = 20)
    (h4 : right_handed_dislike_jazz = 3)
    (h5 : ∀ p, p = total_people - left_handed ∧ p = total_people - (left_handed + right_handed_dislike_jazz)) :
    ∃ x, x = 5 := by
  sorry

end NUMINAMATH_GPT_left_handed_like_jazz_l134_13483


namespace NUMINAMATH_GPT_max_marks_l134_13427

theorem max_marks (M : ℝ) (h_pass : 0.30 * M = 231) : M = 770 := sorry

end NUMINAMATH_GPT_max_marks_l134_13427


namespace NUMINAMATH_GPT_corveus_sleep_deficit_l134_13436

theorem corveus_sleep_deficit :
  let weekday_sleep := 5 -- 4 hours at night + 1-hour nap
  let weekend_sleep := 5 -- 5 hours at night, no naps
  let total_weekday_sleep := 5 * weekday_sleep
  let total_weekend_sleep := 2 * weekend_sleep
  let total_sleep := total_weekday_sleep + total_weekend_sleep
  let recommended_sleep_per_day := 6
  let total_recommended_sleep := 7 * recommended_sleep_per_day
  let sleep_deficit := total_recommended_sleep - total_sleep
  sleep_deficit = 7 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_corveus_sleep_deficit_l134_13436


namespace NUMINAMATH_GPT_min_value_expression_geq_twosqrt3_l134_13424

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  (1/(x-1)) + (3/(y-1))

theorem min_value_expression_geq_twosqrt3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1/x) + (1/y) = 1) : 
  min_value_expression x y >= 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_geq_twosqrt3_l134_13424


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l134_13482

noncomputable def b_value (b : ℝ) : Prop :=
  ∃ s : ℝ, 180 * s = b ∧ b * s = 75 / 32 ∧ b > 0

theorem geometric_sequence_b_value (b : ℝ) : b_value b → b = 20.542 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l134_13482


namespace NUMINAMATH_GPT_iris_jackets_l134_13453

theorem iris_jackets (J : ℕ) (h : 10 * J + 12 + 48 = 90) : J = 3 :=
by
  sorry

end NUMINAMATH_GPT_iris_jackets_l134_13453


namespace NUMINAMATH_GPT_sum_of_three_fractions_is_one_l134_13420

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end NUMINAMATH_GPT_sum_of_three_fractions_is_one_l134_13420


namespace NUMINAMATH_GPT_carol_optimal_strategy_l134_13425

-- Definitions of the random variables
def uniform_A (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def uniform_B (b : ℝ) : Prop := 0.25 ≤ b ∧ b ≤ 0.75
def winning_condition (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Carol's optimal strategy stated as a theorem
theorem carol_optimal_strategy : ∀ (a b c : ℝ), 
  uniform_A a → uniform_B b → (c = 7 / 12) → 
  winning_condition a b c → 
  ∀ (c' : ℝ), uniform_A c' → c' ≠ c → ¬(winning_condition a b c') :=
by
  sorry

end NUMINAMATH_GPT_carol_optimal_strategy_l134_13425


namespace NUMINAMATH_GPT_solve_equation_l134_13439

-- Define the conditions of the problem.
def equation (x : ℝ) : Prop := (5 - x / 3)^(1/3) = -2

-- Define the main theorem to prove that x = 39 is the solution to the equation.
theorem solve_equation : ∃ x : ℝ, equation x ∧ x = 39 :=
by
  existsi 39
  intros
  simp [equation]
  sorry

end NUMINAMATH_GPT_solve_equation_l134_13439


namespace NUMINAMATH_GPT_daily_pre_promotion_hours_l134_13440

-- Defining conditions
def weekly_additional_hours := 6
def hours_driven_in_two_weeks_after_promotion := 40
def days_in_two_weeks := 14
def hours_added_in_two_weeks := 2 * weekly_additional_hours

-- Math proof problem statement
theorem daily_pre_promotion_hours :
  (hours_driven_in_two_weeks_after_promotion - hours_added_in_two_weeks) / days_in_two_weeks = 2 :=
by
  sorry

end NUMINAMATH_GPT_daily_pre_promotion_hours_l134_13440


namespace NUMINAMATH_GPT_chocolates_bought_l134_13435

theorem chocolates_bought (cost_price selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price * 24 = selling_price)
  (h2 : gain_percent = 83.33333333333334)
  (h3 : selling_price = cost_price * 24 * (1 + gain_percent / 100)) :
  cost_price * 44 = selling_price :=
by
  sorry

end NUMINAMATH_GPT_chocolates_bought_l134_13435


namespace NUMINAMATH_GPT_value_of_a_minus_b_l134_13448

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = - (a + b)) :
  a - b = -2 ∨ a - b = -6 := sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l134_13448


namespace NUMINAMATH_GPT_number_of_cows_l134_13410

-- Definitions
variables (c h : ℕ)

-- Conditions
def condition1 : Prop := 4 * c + 2 * h = 20 + 2 * (c + h)
def condition2 : Prop := c + h = 12

-- Theorem
theorem number_of_cows : condition1 c h → condition2 c h → c = 10 :=
  by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_of_cows_l134_13410


namespace NUMINAMATH_GPT_michael_truck_meetings_l134_13488

theorem michael_truck_meetings :
  let michael_speed := 6
  let truck_speed := 12
  let pail_distance := 200
  let truck_stop_time := 20
  let initial_distance := pail_distance
  ∃ (meetings : ℕ), 
  (michael_speed, truck_speed, pail_distance, truck_stop_time, initial_distance, meetings) = 
  (6, 12, 200, 20, 200, 10) :=
sorry

end NUMINAMATH_GPT_michael_truck_meetings_l134_13488


namespace NUMINAMATH_GPT_total_visitors_over_two_days_l134_13490

-- Conditions given in the problem statement
def first_day_visitors : ℕ := 583
def second_day_visitors : ℕ := 246

-- The main problem: proving the total number of visitors over the two days
theorem total_visitors_over_two_days : first_day_visitors + second_day_visitors = 829 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_visitors_over_two_days_l134_13490


namespace NUMINAMATH_GPT_solution_set_of_inequality_l134_13465

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 2*x + 3 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l134_13465


namespace NUMINAMATH_GPT_a_squared_plus_b_squared_eq_sqrt_11_l134_13460

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_condition : a * b * (a - b) = 1

theorem a_squared_plus_b_squared_eq_sqrt_11 : a^2 + b^2 = Real.sqrt 11 := by
  sorry

end NUMINAMATH_GPT_a_squared_plus_b_squared_eq_sqrt_11_l134_13460


namespace NUMINAMATH_GPT_volume_inside_sphere_outside_cylinder_l134_13466

noncomputable def sphere_radius := 6
noncomputable def cylinder_diameter := 8
noncomputable def sphere_volume := 4/3 * Real.pi * (sphere_radius ^ 3)
noncomputable def cylinder_height := Real.sqrt ((sphere_radius * 2) ^ 2 - (cylinder_diameter) ^ 2)
noncomputable def cylinder_volume := Real.pi * ((cylinder_diameter / 2) ^ 2) * cylinder_height
noncomputable def volume_difference := sphere_volume - cylinder_volume

theorem volume_inside_sphere_outside_cylinder:
  volume_difference = (288 - 64 * Real.sqrt 5) * Real.pi :=
sorry

end NUMINAMATH_GPT_volume_inside_sphere_outside_cylinder_l134_13466


namespace NUMINAMATH_GPT_solve_for_x_l134_13449

theorem solve_for_x (x : ℝ) (h : (9 + 1/x)^(1/3) = -2) : x = -1/17 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l134_13449


namespace NUMINAMATH_GPT_value_of_f_12_l134_13476

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_12_l134_13476


namespace NUMINAMATH_GPT_odds_against_C_winning_l134_13402

theorem odds_against_C_winning (prob_A: ℚ) (prob_B: ℚ) (prob_C: ℚ)
    (odds_A: prob_A = 1 / 5) (odds_B: prob_B = 2 / 5) 
    (total_prob: prob_A + prob_B + prob_C = 1):
    ((1 - prob_C) / prob_C) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_odds_against_C_winning_l134_13402


namespace NUMINAMATH_GPT_problem_solution_l134_13407

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l134_13407


namespace NUMINAMATH_GPT_find_R_position_l134_13409

theorem find_R_position :
  ∀ (P Q R : ℤ), P = -6 → Q = -1 → Q = (P + R) / 2 → R = 4 :=
by
  intros P Q R hP hQ hQ_halfway
  sorry

end NUMINAMATH_GPT_find_R_position_l134_13409


namespace NUMINAMATH_GPT_Sam_total_books_l134_13494

/-- Sam's book purchases -/
def Sam_bought_books : Real := 
  let used_adventure_books := 13.0
  let used_mystery_books := 17.0
  let new_crime_books := 15.0
  used_adventure_books + used_mystery_books + new_crime_books

theorem Sam_total_books : Sam_bought_books = 45.0 :=
by
  -- The proof will show that Sam indeed bought 45 books in total
  sorry

end NUMINAMATH_GPT_Sam_total_books_l134_13494


namespace NUMINAMATH_GPT_M_intersect_N_l134_13457

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | x < 1}

theorem M_intersect_N : M ∩ N = {x | -1 < x ∧ x < 1} := 
by
  sorry

end NUMINAMATH_GPT_M_intersect_N_l134_13457


namespace NUMINAMATH_GPT_convert_and_compute_l134_13432

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2 * 4^2 + 3 * 4^1 + 1 * 4^0
  else if n = 21 then 2 * 4^1 + 1 * 4^0
  else if n = 3 then 3
  else 0

noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 135 then 2 * 4^2 + 1 * 4^1 + 3 * 4^0
  else 0

theorem convert_and_compute :
  base10_to_base4 ((base4_to_base10 231 / base4_to_base10 3) * base4_to_base10 21) = 213 :=
by {
  sorry
}

end NUMINAMATH_GPT_convert_and_compute_l134_13432


namespace NUMINAMATH_GPT_weights_problem_l134_13400

theorem weights_problem
  (a b c d : ℕ)
  (h1 : a + b = 280)
  (h2 : b + c = 255)
  (h3 : c + d = 290) 
  : a + d = 315 := 
  sorry

end NUMINAMATH_GPT_weights_problem_l134_13400


namespace NUMINAMATH_GPT_percentage_apples_sold_l134_13413

theorem percentage_apples_sold (A P : ℕ) (h1 : A = 600) (h2 : A * (100 - P) / 100 = 420) : P = 30 := 
by {
  sorry
}

end NUMINAMATH_GPT_percentage_apples_sold_l134_13413


namespace NUMINAMATH_GPT_find_the_number_l134_13493

theorem find_the_number : ∃ x : ℝ, (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ∧ x = 30 := 
by
  sorry

end NUMINAMATH_GPT_find_the_number_l134_13493


namespace NUMINAMATH_GPT_base_7_perfect_square_ab2c_l134_13468

-- Define the necessary conditions
def is_base_7_representation_of (n : ℕ) (a b c : ℕ) : Prop :=
  n = a * 7^3 + b * 7^2 + 2 * 7 + c

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Lean statement for the problem
theorem base_7_perfect_square_ab2c (n a b c : ℕ) (h1 : a ≠ 0) (h2 : is_base_7_representation_of n a b c) (h3 : is_perfect_square n) :
  c = 2 ∨ c = 3 ∨ c = 6 :=
  sorry

end NUMINAMATH_GPT_base_7_perfect_square_ab2c_l134_13468


namespace NUMINAMATH_GPT_smallest_N_l134_13467

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n
  
noncomputable def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

noncomputable def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

theorem smallest_N :
  ∃ N : ℕ, is_square (N / 2) ∧ is_cube (N / 3) ∧ is_fifth_power (N / 5) ∧
  N = 2^15 * 3^10 * 5^6 :=
by
  exists 2^15 * 3^10 * 5^6
  sorry

end NUMINAMATH_GPT_smallest_N_l134_13467


namespace NUMINAMATH_GPT_ellipse_condition_l134_13438

theorem ellipse_condition (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m) = 1) →
  (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_condition_l134_13438


namespace NUMINAMATH_GPT_maria_cupcakes_l134_13418

variable (initial : ℕ) (additional : ℕ) (remaining : ℕ)

theorem maria_cupcakes (h_initial : initial = 19) (h_additional : additional = 10) (h_remaining : remaining = 24) : initial + additional - remaining = 5 := by
  sorry

end NUMINAMATH_GPT_maria_cupcakes_l134_13418


namespace NUMINAMATH_GPT_smaller_number_is_72_l134_13426

theorem smaller_number_is_72
  (x : ℝ)
  (h1 : (3 * x - 24) / (8 * x - 24) = 4 / 9)
  : 3 * x = 72 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_72_l134_13426


namespace NUMINAMATH_GPT_find_g_l134_13411

variable (x : ℝ)

theorem find_g :
  ∃ g : ℝ → ℝ, 2 * x ^ 5 + 4 * x ^ 3 - 3 * x + 5 + g x = 3 * x ^ 4 + 7 * x ^ 2 - 2 * x - 4 ∧
                g x = -2 * x ^ 5 + 3 * x ^ 4 - 4 * x ^ 3 + 7 * x ^ 2 - x - 9 :=
sorry

end NUMINAMATH_GPT_find_g_l134_13411


namespace NUMINAMATH_GPT_inequality_proof_l134_13484

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) : (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l134_13484


namespace NUMINAMATH_GPT_train_length_l134_13464

theorem train_length (L S : ℝ) 
  (h1 : L = S * 15) 
  (h2 : L + 100 = S * 25) : 
  L = 150 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l134_13464


namespace NUMINAMATH_GPT_max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l134_13481

noncomputable def max_xy (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then x * y else 0

noncomputable def min_y_over_x_plus_4_over_y (x y : ℝ) : ℝ :=
if h : x > 0 ∧ y > 0 ∧ x + 2 * y = 4 then y / x + 4 / y else 0

theorem max_xy_is_2 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → max_xy x y = 2 :=
by
  intros x y hx hy hxy
  sorry

theorem min_y_over_x_plus_4_over_y_is_4 : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 4 → min_y_over_x_plus_4_over_y x y = 4 :=
by
  intros x y hx hy hxy
  sorry

end NUMINAMATH_GPT_max_xy_is_2_min_y_over_x_plus_4_over_y_is_4_l134_13481
