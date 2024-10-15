import Mathlib

namespace NUMINAMATH_GPT_painted_surface_area_of_pyramid_l2406_240650

/--
Given 19 unit cubes arranged in a 4-layer pyramid-like structure, where:
- The top layer has 1 cube,
- The second layer has 3 cubes,
- The third layer has 5 cubes,
- The bottom layer has 10 cubes,

Prove that the total painted surface area is 43 square meters.
-/
theorem painted_surface_area_of_pyramid :
  let layer1 := 1 -- top layer
  let layer2 := 3 -- second layer
  let layer3 := 5 -- third layer
  let layer4 := 10 -- bottom layer
  let total_cubes := layer1 + layer2 + layer3 + layer4
  let top_faces := layer1 * 1 + layer2 * 1 + layer3 * 1 + layer4 * 1
  let side_faces_layer1 := layer1 * 5
  let side_faces_layer2 := layer2 * 3
  let side_faces_layer3 := layer3 * 2
  let side_faces := side_faces_layer1 + side_faces_layer2 + side_faces_layer3
  let total_surface_area := top_faces + side_faces
  total_cubes = 19 → total_surface_area = 43 :=
by
  intros
  sorry

end NUMINAMATH_GPT_painted_surface_area_of_pyramid_l2406_240650


namespace NUMINAMATH_GPT_sum_of_vars_l2406_240608

variables (x y z w : ℤ)

theorem sum_of_vars (h1 : x - y + z = 7)
                    (h2 : y - z + w = 8)
                    (h3 : z - w + x = 4)
                    (h4 : w - x + y = 3) :
  x + y + z + w = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_vars_l2406_240608


namespace NUMINAMATH_GPT_floor_ceil_diff_l2406_240607

theorem floor_ceil_diff (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌊x⌋ + x - ⌈x⌉ = x - 1 :=
sorry

end NUMINAMATH_GPT_floor_ceil_diff_l2406_240607


namespace NUMINAMATH_GPT_find_the_number_l2406_240671

theorem find_the_number :
  ∃ X : ℝ, (66.2 = (6.620000000000001 / 100) * X) ∧ X = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l2406_240671


namespace NUMINAMATH_GPT_solve_problem_l2406_240628

theorem solve_problem (a b c d : ℤ) (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 5) : (b - d) ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2406_240628


namespace NUMINAMATH_GPT_cone_volume_l2406_240642

theorem cone_volume (r l: ℝ) (h: ℝ) (hr : r = 1) (hl : l = 2) (hh : h = Real.sqrt (l^2 - r^2)) : 
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by 
  sorry

end NUMINAMATH_GPT_cone_volume_l2406_240642


namespace NUMINAMATH_GPT_work_completion_time_l2406_240666

theorem work_completion_time (a b c : ℕ) (ha : a = 36) (hb : b = 18) (hc : c = 6) : (1 / (1 / a + 1 / b + 1 / c) = 4) := by
  sorry

end NUMINAMATH_GPT_work_completion_time_l2406_240666


namespace NUMINAMATH_GPT_sequences_count_n3_sequences_count_n6_sequences_count_n9_l2406_240618

inductive Shape
  | triangle
  | square
  | rectangle (k : ℕ)

open Shape

def transition (s : Shape) : List Shape :=
  match s with
  | triangle => [triangle, square]
  | square => [rectangle 1]
  | rectangle k =>
    if k = 0 then [rectangle 1] else [rectangle (k - 1), rectangle (k + 1)]

def count_sequences (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (shapes : List Shape) : ℕ :=
    if m = 0 then shapes.length
    else
      let next_shapes := shapes.bind transition
      aux (m - 1) next_shapes
  aux n [square]

theorem sequences_count_n3 : count_sequences 3 = 5 :=
  by sorry

theorem sequences_count_n6 : count_sequences 6 = 24 :=
  by sorry

theorem sequences_count_n9 : count_sequences 9 = 149 :=
  by sorry

end NUMINAMATH_GPT_sequences_count_n3_sequences_count_n6_sequences_count_n9_l2406_240618


namespace NUMINAMATH_GPT_maximize_area_playground_l2406_240665

noncomputable def maxAreaPlayground : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area_playground : ∀ (l w : ℝ),
  (2 * l + 2 * w = 400) ∧ (l ≥ 100) ∧ (w ≥ 60) → l * w ≤ maxAreaPlayground :=
by
  intros l w h
  sorry

end NUMINAMATH_GPT_maximize_area_playground_l2406_240665


namespace NUMINAMATH_GPT_sequence_term_l2406_240686

theorem sequence_term (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, (n + 1) * a n = 2 * n * a (n + 1)) : 
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_l2406_240686


namespace NUMINAMATH_GPT_sequence_has_both_max_and_min_l2406_240655

noncomputable def a_n (n : ℕ) : ℝ :=
  (n + 1) * ((-10 / 11) ^ n)

theorem sequence_has_both_max_and_min :
  ∃ (max min : ℝ) (N M : ℕ), 
    (∀ n : ℕ, a_n n ≤ max) ∧ (∀ n : ℕ, min ≤ a_n n) ∧ 
    (a_n N = max) ∧ (a_n M = min) := 
sorry

end NUMINAMATH_GPT_sequence_has_both_max_and_min_l2406_240655


namespace NUMINAMATH_GPT_west_1000_move_l2406_240637

def eastMovement (d : Int) := d  -- east movement positive
def westMovement (d : Int) := -d -- west movement negative

theorem west_1000_move : westMovement 1000 = -1000 :=
  by
    sorry

end NUMINAMATH_GPT_west_1000_move_l2406_240637


namespace NUMINAMATH_GPT_increase_in_circumference_by_2_cm_l2406_240677

noncomputable def radius_increase_by_two (r : ℝ) : ℝ := r + 2
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem increase_in_circumference_by_2_cm (r : ℝ) : 
    circumference (radius_increase_by_two r) - circumference r = 12.56 :=
by sorry

end NUMINAMATH_GPT_increase_in_circumference_by_2_cm_l2406_240677


namespace NUMINAMATH_GPT_apollonius_circle_equation_l2406_240641

theorem apollonius_circle_equation (x y : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (8, 0))
  (h : dist (x, y) A / dist (x, y) B = 1 / 2) : x^2 + y^2 = 16 := 
sorry

end NUMINAMATH_GPT_apollonius_circle_equation_l2406_240641


namespace NUMINAMATH_GPT_sin_x_solution_l2406_240674

theorem sin_x_solution (A B C x : ℝ) (h : A * Real.cos x + B * Real.sin x = C) :
  ∃ (u v : ℝ),  -- We assert the existence of u and v such that 
    Real.sin x = (A * C + B * u) / (A^2 + B^2) ∨ 
    Real.sin x = (A * C - B * v) / (A^2 + B^2) :=
sorry

end NUMINAMATH_GPT_sin_x_solution_l2406_240674


namespace NUMINAMATH_GPT_smaller_number_l2406_240604

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 := by
  sorry

end NUMINAMATH_GPT_smaller_number_l2406_240604


namespace NUMINAMATH_GPT_saber_toothed_frog_tails_l2406_240610

def tails_saber_toothed_frog (n k : ℕ) (x : ℕ) : Prop :=
  5 * n + 4 * k = 100 ∧ n + x * k = 64

theorem saber_toothed_frog_tails : ∃ x, ∃ n k : ℕ, tails_saber_toothed_frog n k x ∧ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_saber_toothed_frog_tails_l2406_240610


namespace NUMINAMATH_GPT_cole_round_trip_time_l2406_240670

/-- Prove that the total round trip time is 2 hours given the conditions -/
theorem cole_round_trip_time :
  ∀ (speed_to_work : ℝ) (speed_back_home : ℝ) (time_to_work_min : ℝ),
  speed_to_work = 50 → speed_back_home = 110 → time_to_work_min = 82.5 →
  ((time_to_work_min / 60) * speed_to_work + (time_to_work_min * speed_to_work / speed_back_home) / 60) = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cole_round_trip_time_l2406_240670


namespace NUMINAMATH_GPT_probability_win_all_games_l2406_240633

variable (p : ℚ) (n : ℕ)

-- Define the conditions
def probability_of_winning := p = 2 / 3
def number_of_games := n = 6
def independent_games := true

-- The theorem we want to prove
theorem probability_win_all_games (h₁ : probability_of_winning p)
                                   (h₂ : number_of_games n)
                                   (h₃ : independent_games) :
  p^n = 64 / 729 :=
sorry

end NUMINAMATH_GPT_probability_win_all_games_l2406_240633


namespace NUMINAMATH_GPT_remainder_when_587421_divided_by_6_l2406_240634

theorem remainder_when_587421_divided_by_6 :
  ¬ (587421 % 2 = 0) → (587421 % 3 = 0) → 587421 % 6 = 3 :=
by sorry

end NUMINAMATH_GPT_remainder_when_587421_divided_by_6_l2406_240634


namespace NUMINAMATH_GPT_min_handshakes_l2406_240695

theorem min_handshakes (n : ℕ) (h1 : n = 25) 
  (h2 : ∀ (p : ℕ), p < n → ∃ q r : ℕ, q ≠ r ∧ q < n ∧ r < n ∧ q ≠ p ∧ r ≠ p) 
  (h3 : ∃ a b c : ℕ, a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(∃ d : ℕ, (d = a ∨ d = b ∨ d = c) ∧ (¬(a = d ∨ b = d ∨ c = d)) ∧ d < n)) :
  ∃ m : ℕ, m = 28 :=
by
  sorry

end NUMINAMATH_GPT_min_handshakes_l2406_240695


namespace NUMINAMATH_GPT_wheel_radius_increase_proof_l2406_240652

noncomputable def radius_increase (orig_distance odometer_distance : ℝ) (orig_radius : ℝ) : ℝ :=
  let orig_circumference := 2 * Real.pi * orig_radius
  let distance_per_rotation := orig_circumference / 63360
  let num_rotations_orig := orig_distance / distance_per_rotation
  let num_rotations_new := odometer_distance / distance_per_rotation
  let new_distance := orig_distance
  let new_radius := (new_distance / num_rotations_new) * 63360 / (2 * Real.pi)
  new_radius - orig_radius

theorem wheel_radius_increase_proof :
  radius_increase 600 580 16 = 0.42 :=
by 
  -- The proof is skipped.
  sorry

end NUMINAMATH_GPT_wheel_radius_increase_proof_l2406_240652


namespace NUMINAMATH_GPT_pentagon_diagonal_l2406_240629

theorem pentagon_diagonal (a d : ℝ) (h : d^2 = a^2 + a * d) : 
  d = a * (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_pentagon_diagonal_l2406_240629


namespace NUMINAMATH_GPT_proof_complement_union_l2406_240636

-- Definition of the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the subset A
def A : Finset ℕ := {0, 3, 4}

-- Definition of the subset B
def B : Finset ℕ := {1, 3}

-- Definition of the complement of A in U
def complement_A : Finset ℕ := U \ A

-- Definition of the union of the complement of A and B
def union_complement_A_B : Finset ℕ := complement_A ∪ B

-- Statement of the theorem
theorem proof_complement_union :
  union_complement_A_B = {1, 2, 3} :=
sorry

end NUMINAMATH_GPT_proof_complement_union_l2406_240636


namespace NUMINAMATH_GPT_tom_splitting_slices_l2406_240698

theorem tom_splitting_slices :
  ∃ S : ℕ, (∃ t, t = 3/8 * S) → 
          (∃ u, u = 1/2 * (S - t)) → 
          (∃ v, v = u + t) → 
          (v = 5) → 
          (S / 2 = 8) :=
sorry

end NUMINAMATH_GPT_tom_splitting_slices_l2406_240698


namespace NUMINAMATH_GPT_complement_of_A_l2406_240632

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | abs (x - 1) > 1 }

-- Define the problem statement
theorem complement_of_A :
  ∀ x : ℝ, x ∈ compl A ↔ x ∈ Icc 0 2 :=
by
  intro x
  rw [mem_compl_iff, mem_Icc]
  sorry

end NUMINAMATH_GPT_complement_of_A_l2406_240632


namespace NUMINAMATH_GPT_range_of_m_three_zeros_l2406_240675

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_three_zeros_l2406_240675


namespace NUMINAMATH_GPT_not_possible_identical_nonzero_remainders_l2406_240619

theorem not_possible_identical_nonzero_remainders :
  ¬ ∃ (a : ℕ → ℕ) (r : ℕ), (r > 0) ∧ (∀ i : Fin 100, a i % (a ((i + 1) % 100)) = r) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_identical_nonzero_remainders_l2406_240619


namespace NUMINAMATH_GPT_gcd_of_polynomial_l2406_240676

theorem gcd_of_polynomial (x : ℕ) (hx : 32515 ∣ x) :
    Nat.gcd ((3 * x + 5) * (5 * x + 3) * (11 * x + 7) * (x + 17)) x = 35 :=
sorry

end NUMINAMATH_GPT_gcd_of_polynomial_l2406_240676


namespace NUMINAMATH_GPT_triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l2406_240638

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

def right_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_condition_A (a b c : ℝ) (h : triangle a b c) : 
  b^2 = (a + c) * (c - a) → right_triangle a c b := 
sorry

theorem triangle_condition_B (A B C : ℝ) (h : A + B + C = 180) : 
  A = B + C → 90 = A :=
sorry

theorem triangle_condition_C (A B C : ℝ) (h : A + B + C = 180) : 
  3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → 
  ¬ (right_triangle A B C) :=
sorry

theorem triangle_condition_D : 
  right_triangle 6 8 10 := 
sorry

theorem problem_solution (a b c : ℝ) (A B C : ℝ) (hABC : triangle a b c) : 
  (b^2 = (a + c) * (c - a) → right_triangle a c b) ∧
  ((A + B + C = 180) ∧ (A = B + C) → 90 = A) ∧
  (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → ¬ right_triangle a b c) ∧
  (right_triangle 6 8 10) → 
  ∃ (cond : Prop), cond = (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C) := 
sorry

end NUMINAMATH_GPT_triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l2406_240638


namespace NUMINAMATH_GPT_case1_case2_case3_l2406_240601

-- Definitions from conditions
def tens_digit_one : ℕ := sorry
def units_digit_one : ℕ := sorry
def units_digit_two : ℕ := sorry
def tens_digit_two : ℕ := sorry
def sum_units_digits_ten : Prop := units_digit_one + units_digit_two = 10
def same_digit : ℕ := sorry
def sum_tens_digits_ten : Prop := tens_digit_one + tens_digit_two = 10

-- The proof problems
theorem case1 (A B D : ℕ) (hBplusD : B + D = 10) :
  (10 * A + B) * (10 * A + D) = 100 * (A^2 + A) + B * D :=
sorry

theorem case2 (A B C : ℕ) (hAplusC : A + C = 10) :
  (10 * A + B) * (10 * C + B) = 100 * A * C + 100 * B + B^2 :=
sorry

theorem case3 (A B C : ℕ) (hAplusB : A + B = 10) :
  (10 * A + B) * (10 * C + C) = 100 * A * C + 100 * C + B * C :=
sorry

end NUMINAMATH_GPT_case1_case2_case3_l2406_240601


namespace NUMINAMATH_GPT_sum_of_digits_of_m_l2406_240689

-- Define the logarithms and intermediate expressions
noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_m :
  ∃ m : ℕ, log_b 3 (log_b 81 m) = log_b 9 (log_b 9 m) ∧ sum_of_digits m = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_m_l2406_240689


namespace NUMINAMATH_GPT_initial_number_of_friends_l2406_240605

theorem initial_number_of_friends (X : ℕ) (H : 3 * (X - 3) = 15) : X = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_friends_l2406_240605


namespace NUMINAMATH_GPT_no_integer_y_makes_Q_perfect_square_l2406_240600

def Q (y : ℤ) : ℤ := y^4 + 8 * y^3 + 18 * y^2 + 10 * y + 41

theorem no_integer_y_makes_Q_perfect_square :
  ¬ ∃ y : ℤ, ∃ b : ℤ, Q y = b^2 :=
by
  intro h
  rcases h with ⟨y, b, hQ⟩
  sorry

end NUMINAMATH_GPT_no_integer_y_makes_Q_perfect_square_l2406_240600


namespace NUMINAMATH_GPT_solve_for_x_l2406_240627

theorem solve_for_x (x y : ℝ) (h : 3 * x - 4 * y = 5) : x = (1 / 3) * (5 + 4 * y) :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l2406_240627


namespace NUMINAMATH_GPT_trig_expr_eval_sin_minus_cos_l2406_240654

-- Problem 1: Evaluation of trigonometric expression
theorem trig_expr_eval : 
    (Real.sin (-π / 2) + 3 * Real.cos 0 - 2 * Real.tan (3 * π / 4) - 4 * Real.cos (5 * π / 3)) = 2 :=
by 
    sorry

-- Problem 2: Given tangent value and angle constraints, find sine minus cosine
theorem sin_minus_cos {θ : ℝ} 
    (h1 : Real.tan θ = 4 / 3)
    (h2 : 0 < θ)
    (h3 : θ < π / 2) : 
    (Real.sin θ - Real.cos θ) = 1 / 5 :=
by 
    sorry

end NUMINAMATH_GPT_trig_expr_eval_sin_minus_cos_l2406_240654


namespace NUMINAMATH_GPT_pentagon_segment_condition_l2406_240606

-- Define the problem context and hypothesis
variable (a b c d e : ℝ)

theorem pentagon_segment_condition 
  (h₁ : a + b + c + d + e = 3)
  (h₂ : a ≤ b)
  (h₃ : b ≤ c)
  (h₄ : c ≤ d)
  (h₅ : d ≤ e) : 
  a < 3 / 2 ∧ b < 3 / 2 ∧ c < 3 / 2 ∧ d < 3 / 2 ∧ e < 3 / 2 := 
sorry

end NUMINAMATH_GPT_pentagon_segment_condition_l2406_240606


namespace NUMINAMATH_GPT_geometric_a1_value_l2406_240679

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_a1_value (a3 a5 : ℝ) (q : ℝ) : 
  a3 = geometric_sequence a1 q 3 →
  a5 = geometric_sequence a1 q 5 →
  a1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_a1_value_l2406_240679


namespace NUMINAMATH_GPT_proj_v_w_l2406_240663

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
  let w_dot_w := dot_product w w
  let v_dot_w := dot_product v w
  let scalar := v_dot_w / w_dot_w
  (scalar * w.1, scalar * w.2)

theorem proj_v_w :
  let v := (4, -3)
  let w := (12, 5)
  proj v w = (396 / 169, 165 / 169) :=
by
  sorry

end NUMINAMATH_GPT_proj_v_w_l2406_240663


namespace NUMINAMATH_GPT_problem_a_problem_b_l2406_240659

-- Define the conditions for problem (a):
variable (x y z : ℝ)
variable (h_xyz : x * y * z = 1)

theorem problem_a (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 :=
sorry

-- Define the conditions for problem (b):
variable (a b c : ℚ)

theorem problem_b (h_abc : a * b * c = 1) :
  ∃ (x y z : ℚ), x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ (x * y * z = 1) ∧ 
  (x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2406_240659


namespace NUMINAMATH_GPT_solve_system_l2406_240621

theorem solve_system : ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 8 * x + 5 * y = 7 ∧ x = 1 / 4 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2406_240621


namespace NUMINAMATH_GPT_arithmetic_progression_pairs_count_l2406_240613

theorem arithmetic_progression_pairs_count (x y : ℝ) 
  (h1 : x = (15 + y) / 2)
  (h2 : x + x * y = 2 * y) : 
  (∃ x1 y1, x1 = (15 + y1) / 2 ∧ x1 + x1 * y1 = 2 * y1 ∧ x1 = (9 + 3 * Real.sqrt 7) / 2 ∧ y1 = -6 + 3 * Real.sqrt 7) ∨ 
  (∃ x2 y2, x2 = (15 + y2) / 2 ∧ x2 + x2 * y2 = 2 * y2 ∧ x2 = (9 - 3 * Real.sqrt 7) / 2 ∧ y2 = -6 - 3 * Real.sqrt 7) := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_pairs_count_l2406_240613


namespace NUMINAMATH_GPT_find_possible_values_for_P_l2406_240696

theorem find_possible_values_for_P (x y P : ℕ) (h1 : x < y) :
  P = (x^3 - y) / (1 + x * y) → (P = 0 ∨ P ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_for_P_l2406_240696


namespace NUMINAMATH_GPT_acute_triangle_probability_l2406_240683

open Finset

noncomputable def isAcuteTriangleProb (n : ℕ) : Prop :=
  ∃ k : ℕ, (n = 2 * k ∧ (3 * (k - 2)) / (2 * (2 * k - 1)) = 93 / 125) ∨ (n = 2 * k + 1 ∧ (3 * (k - 1)) / (2 * (2 * k - 1)) = 93 / 125)

theorem acute_triangle_probability (n : ℕ) : isAcuteTriangleProb n → n = 376 ∨ n = 127 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_probability_l2406_240683


namespace NUMINAMATH_GPT_gcd_78_36_l2406_240658

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := 
by
  sorry

end NUMINAMATH_GPT_gcd_78_36_l2406_240658


namespace NUMINAMATH_GPT_problem_D_l2406_240699

-- Define the lines m and n, and planes α and β
variables (m n : Type) (α β : Type)

-- Define the parallel and perpendicular relations
variables (parallel : Type → Type → Prop) (perpendicular : Type → Type → Prop)

-- Assume the conditions of problem D
variables (h1 : perpendicular m α) (h2 : parallel n β) (h3 : parallel α β)

-- The proof problem statement: Prove that under these assumptions, m is perpendicular to n
theorem problem_D : perpendicular m n :=
sorry

end NUMINAMATH_GPT_problem_D_l2406_240699


namespace NUMINAMATH_GPT_complete_the_square_k_l2406_240688

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_k_l2406_240688


namespace NUMINAMATH_GPT_original_number_l2406_240603

theorem original_number (x : ℝ) (h : 1.35 * x = 935) : x = 693 := by
  sorry

end NUMINAMATH_GPT_original_number_l2406_240603


namespace NUMINAMATH_GPT_neznaika_discrepancy_l2406_240625

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end NUMINAMATH_GPT_neznaika_discrepancy_l2406_240625


namespace NUMINAMATH_GPT_olivia_total_pieces_l2406_240691

def initial_pieces_folder1 : ℕ := 152
def initial_pieces_folder2 : ℕ := 98
def used_pieces_folder1 : ℕ := 78
def used_pieces_folder2 : ℕ := 42

def remaining_pieces_folder1 : ℕ :=
  initial_pieces_folder1 - used_pieces_folder1

def remaining_pieces_folder2 : ℕ :=
  initial_pieces_folder2 - used_pieces_folder2

def total_remaining_pieces : ℕ :=
  remaining_pieces_folder1 + remaining_pieces_folder2

theorem olivia_total_pieces : total_remaining_pieces = 130 :=
  by sorry

end NUMINAMATH_GPT_olivia_total_pieces_l2406_240691


namespace NUMINAMATH_GPT_CarlyWorkedOnElevenDogs_l2406_240681

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end NUMINAMATH_GPT_CarlyWorkedOnElevenDogs_l2406_240681


namespace NUMINAMATH_GPT_cows_count_l2406_240640

theorem cows_count (D C : ℕ) (h1 : 2 * (D + C) + 32 = 2 * D + 4 * C) : C = 16 :=
by
  sorry

end NUMINAMATH_GPT_cows_count_l2406_240640


namespace NUMINAMATH_GPT_positive_integers_are_N_star_l2406_240673

def Q := { x : ℚ | true } -- The set of rational numbers
def N := { x : ℕ | true } -- The set of natural numbers
def N_star := { x : ℕ | x > 0 } -- The set of positive integers
def Z := { x : ℤ | true } -- The set of integers

theorem positive_integers_are_N_star : 
  ∀ x : ℕ, (x ∈ N_star) ↔ (x > 0) := 
sorry

end NUMINAMATH_GPT_positive_integers_are_N_star_l2406_240673


namespace NUMINAMATH_GPT_calculate_expression_l2406_240617

def inequality_holds (a b c d x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - d) / (x - c) ≥ 0

theorem calculate_expression : 
  ∀ (a b c d : ℝ),
    a < b ∧ b < d ∧
    (∀ x : ℝ, 
      (inequality_holds a b c d x ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32))) →
    a + 2 * b + 3 * c + 4 * d = 160 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l2406_240617


namespace NUMINAMATH_GPT_matrix_pow_50_l2406_240631

open Matrix

-- Define the given matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 2; -8, -5]

-- Define the expected result for C^50
def C_50 : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-199, -100; 400, 199]

-- Proposition asserting that C^50 equals the given result matrix
theorem matrix_pow_50 :
  C ^ 50 = C_50 := 
  by
  sorry

end NUMINAMATH_GPT_matrix_pow_50_l2406_240631


namespace NUMINAMATH_GPT_find_p_l2406_240648

theorem find_p (p q : ℚ) (h1 : 3 * p + 4 * q = 15) (h2 : 4 * p + 3 * q = 18) : p = 27 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2406_240648


namespace NUMINAMATH_GPT_employee_y_payment_l2406_240651

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 590) (h2 : X = 1.2 * Y) : Y = 268.18 := by
  sorry

end NUMINAMATH_GPT_employee_y_payment_l2406_240651


namespace NUMINAMATH_GPT_calculate_expression_l2406_240624

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2406_240624


namespace NUMINAMATH_GPT_denomination_of_bill_l2406_240609

def cost_berries : ℝ := 7.19
def cost_peaches : ℝ := 6.83
def change_received : ℝ := 5.98

theorem denomination_of_bill :
  (cost_berries + cost_peaches) + change_received = 20.0 := 
by 
  sorry

end NUMINAMATH_GPT_denomination_of_bill_l2406_240609


namespace NUMINAMATH_GPT_deanna_wins_l2406_240692

theorem deanna_wins (A B C D : ℕ) (total_games : ℕ) (total_wins : ℕ) (A_wins : A = 5) (B_wins : B = 2)
  (C_wins : C = 1) (total_games_def : total_games = 6) (total_wins_def : total_wins = 12)
  (total_wins_eq : A + B + C + D = total_wins) : D = 4 :=
by
  sorry

end NUMINAMATH_GPT_deanna_wins_l2406_240692


namespace NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l2406_240680

theorem general_term_formula :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) := 
  sorry

theorem sum_first_n_terms :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) →
  (∀ n, b n = 1 / (a n * a (n + 1))) →
  (∀ n, S n = 2 * (1 - 1 / (2 * n + 1))) →
  (S n = 4 * n / (2 * n + 1)) :=
  sorry

end NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l2406_240680


namespace NUMINAMATH_GPT_largest_log_value_l2406_240635

theorem largest_log_value :
  ∃ (x y z t : ℝ) (a b c : ℝ),
    x ≤ y ∧ y ≤ z ∧ z ≤ t ∧
    a = Real.log y / Real.log x ∧
    b = Real.log z / Real.log y ∧
    c = Real.log t / Real.log z ∧
    a = 15 ∧ b = 20 ∧ c = 21 ∧
    (∃ u v w, u = a * b ∧ v = b * c ∧ w = a * b * c ∧ w = 420) := sorry

end NUMINAMATH_GPT_largest_log_value_l2406_240635


namespace NUMINAMATH_GPT_non_zero_real_positive_integer_l2406_240672

theorem non_zero_real_positive_integer (x : ℝ) (h : x ≠ 0) : 
  (∃ k : ℤ, k > 0 ∧ (x - |x-1|) / x = k) ↔ x = 1 := 
sorry

end NUMINAMATH_GPT_non_zero_real_positive_integer_l2406_240672


namespace NUMINAMATH_GPT_jeopardy_episode_length_l2406_240615

-- Definitions based on the conditions
def num_episodes_jeopardy : ℕ := 2
def num_episodes_wheel : ℕ := 2
def wheel_twice_jeopardy (J : ℝ) : ℝ := 2 * J
def total_time_watched : ℝ := 120 -- in minutes

-- Condition stating the total time watched in terms of J
def total_watching_time_formula (J : ℝ) : ℝ :=
  num_episodes_jeopardy * J + num_episodes_wheel * (wheel_twice_jeopardy J)

theorem jeopardy_episode_length : ∃ J : ℝ, total_watching_time_formula J = total_time_watched ∧ J = 20 :=
by
  use 20
  simp [total_watching_time_formula, wheel_twice_jeopardy, num_episodes_jeopardy, num_episodes_wheel, total_time_watched]
  sorry

end NUMINAMATH_GPT_jeopardy_episode_length_l2406_240615


namespace NUMINAMATH_GPT_ratio_problem_l2406_240687

theorem ratio_problem
  (a b c d : ℝ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 49) :
  d / a = 1 / 122.5 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_ratio_problem_l2406_240687


namespace NUMINAMATH_GPT_prime_gt_10_exists_m_n_l2406_240685

theorem prime_gt_10_exists_m_n (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m + n < p ∧ p ∣ (5^m * 7^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_prime_gt_10_exists_m_n_l2406_240685


namespace NUMINAMATH_GPT_capacity_of_buckets_l2406_240612

theorem capacity_of_buckets :
  (∃ x : ℝ, 26 * x = 39 * 9) → (∃ x : ℝ, 26 * x = 351 ∧ x = 13.5) :=
by
  sorry

end NUMINAMATH_GPT_capacity_of_buckets_l2406_240612


namespace NUMINAMATH_GPT_mass_percentage_Al_in_Al2CO33_l2406_240653
-- Importing the required libraries

-- Define the necessary constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

-- Define the main theorem to prove the mass percentage of Al in Al2(CO3)3
theorem mass_percentage_Al_in_Al2CO33 :
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  simp [molar_mass_Al, molar_mass_C, molar_mass_O, molar_mass_Al2CO33, mass_Al_in_Al2CO33]
  -- Calculation result based on given molar masses
  sorry

end NUMINAMATH_GPT_mass_percentage_Al_in_Al2CO33_l2406_240653


namespace NUMINAMATH_GPT_original_faculty_number_l2406_240639

theorem original_faculty_number (x : ℝ) (h : 0.85 * x = 195) : x = 229 := by
  sorry

end NUMINAMATH_GPT_original_faculty_number_l2406_240639


namespace NUMINAMATH_GPT_each_album_contains_correct_pictures_l2406_240649

def pictures_in_each_album (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat) :=
  (pictures_per_album_phone + pictures_per_album_camera)

theorem each_album_contains_correct_pictures (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat)
  (h1 : pictures_phone = 80)
  (h2 : pictures_camera = 40)
  (h3 : albums = 10)
  (h4 : pictures_per_album_phone = 8)
  (h5 : pictures_per_album_camera = 4)
  : pictures_in_each_album pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera = 12 := by
  sorry

end NUMINAMATH_GPT_each_album_contains_correct_pictures_l2406_240649


namespace NUMINAMATH_GPT_nth_equation_l2406_240611

theorem nth_equation (n : ℕ) (hn : n ≠ 0) : 
  (↑n + 2) / ↑n - 2 / (↑n + 2) = ((↑n + 2)^2 + ↑n^2) / (↑n * (↑n + 2)) - 1 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l2406_240611


namespace NUMINAMATH_GPT_triangle_area_l2406_240630

theorem triangle_area (h b : ℝ) (Hhb : h < b) :
  let P := (0, b)
  let B := (b, 0)
  let D := (h, h)
  let PD := b - h
  let DB := b - h
  1 / 2 * PD * DB = 1 / 2 * (b - h) ^ 2 := by 
  sorry

end NUMINAMATH_GPT_triangle_area_l2406_240630


namespace NUMINAMATH_GPT_odd_periodic_function_l2406_240682

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

-- Problem statement
theorem odd_periodic_function (h_odd : odd_function f)
  (h_period : periodic_function f) (h_half : f 0.5 = 1) : f 7.5 = -1 :=
sorry

end NUMINAMATH_GPT_odd_periodic_function_l2406_240682


namespace NUMINAMATH_GPT_mike_can_buy_nine_games_l2406_240657

noncomputable def mike_dollars (initial_dollars : ℕ) (spent_dollars : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_dollars - spent_dollars) / game_cost

theorem mike_can_buy_nine_games : mike_dollars 69 24 5 = 9 := by
  sorry

end NUMINAMATH_GPT_mike_can_buy_nine_games_l2406_240657


namespace NUMINAMATH_GPT_simplify_fraction_l2406_240620

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2406_240620


namespace NUMINAMATH_GPT_part1_inequality_part2_min_value_l2406_240693

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  4^x + m * 2^x

theorem part1_inequality (x : ℝ) : f x (-3) > 4 → x > 2 :=
  sorry

theorem part2_min_value (h : (∀ x : ℝ, f x m + f (-x) m ≥ -4)) : m = -3 :=
  sorry

end NUMINAMATH_GPT_part1_inequality_part2_min_value_l2406_240693


namespace NUMINAMATH_GPT_andy_time_difference_l2406_240645

def time_dawn : ℕ := 20
def time_andy : ℕ := 46
def double_time_dawn : ℕ := 2 * time_dawn

theorem andy_time_difference :
  time_andy - double_time_dawn = 6 := by
  sorry

end NUMINAMATH_GPT_andy_time_difference_l2406_240645


namespace NUMINAMATH_GPT_boys_without_glasses_l2406_240626

def total_students_with_glasses : ℕ := 36
def girls_with_glasses : ℕ := 21
def total_boys : ℕ := 30

theorem boys_without_glasses :
  total_boys - (total_students_with_glasses - girls_with_glasses) = 15 :=
by
  sorry

end NUMINAMATH_GPT_boys_without_glasses_l2406_240626


namespace NUMINAMATH_GPT_children_division_into_circles_l2406_240602

theorem children_division_into_circles (n m k : ℕ) (hn : n = 5) (hm : m = 2) (trees_indistinguishable : true) (children_distinguishable : true) :
  ∃ ways, ways = 50 := 
by
  sorry

end NUMINAMATH_GPT_children_division_into_circles_l2406_240602


namespace NUMINAMATH_GPT_trace_bag_weight_is_two_l2406_240690

-- Given the conditions in the problem
def weight_gordon_bag₁ : ℕ := 3
def weight_gordon_bag₂ : ℕ := 7
def num_traces_bag : ℕ := 5

-- Total weight of Gordon's bags is 10
def total_weight_gordon := weight_gordon_bag₁ + weight_gordon_bag₂

-- Trace's bags weight
def total_weight_trace := total_weight_gordon

-- All conditions must imply this equation is true
theorem trace_bag_weight_is_two :
  (num_traces_bag * 2 = total_weight_trace) → (2 = 2) :=
  by
    sorry

end NUMINAMATH_GPT_trace_bag_weight_is_two_l2406_240690


namespace NUMINAMATH_GPT_tom_took_out_beads_l2406_240614

-- Definitions of the conditions
def green_beads : Nat := 1
def brown_beads : Nat := 2
def red_beads : Nat := 3
def beads_left_in_container : Nat := 4

-- Total initial beads
def total_beads : Nat := green_beads + brown_beads + red_beads

-- The Lean problem statement to prove
theorem tom_took_out_beads : (total_beads - beads_left_in_container) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tom_took_out_beads_l2406_240614


namespace NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l2406_240644

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, N = 800 ∧
  (40 : ℕ) / N = (2 : ℕ) / (40 : ℕ) := 
sorry

end NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l2406_240644


namespace NUMINAMATH_GPT_symmetric_axis_of_quadratic_l2406_240662

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := (x - 3) * (x + 5)

-- Prove that the symmetric axis of the quadratic function is the line x = -1
theorem symmetric_axis_of_quadratic : ∀ (x : ℝ), quadratic_function x = (x - 3) * (x + 5) → x = -1 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_symmetric_axis_of_quadratic_l2406_240662


namespace NUMINAMATH_GPT_smallest_d0_l2406_240667

theorem smallest_d0 (r : ℕ) (hr : r ≥ 3) : ∃ d₀, d₀ = 2^(r - 2) ∧ (7^d₀ ≡ 1 [MOD 2^r]) :=
by
  sorry

end NUMINAMATH_GPT_smallest_d0_l2406_240667


namespace NUMINAMATH_GPT_matrix_vec_addition_l2406_240661

def matrix := (Fin 2 → Fin 2 → ℤ)
def vector := Fin 2 → ℤ

def m : matrix := ![![4, -2], ![6, 5]]
def v1 : vector := ![-2, 3]
def v2 : vector := ![1, -1]

def matrix_vec_mul (m : matrix) (v : vector) : vector :=
  ![m 0 0 * v 0 + m 0 1 * v 1,
    m 1 0 * v 0 + m 1 1 * v 1]

def vec_add (v1 v2 : vector) : vector :=
  ![v1 0 + v2 0, v1 1 + v2 1]

theorem matrix_vec_addition :
  vec_add (matrix_vec_mul m v1) v2 = ![-13, 2] :=
by
  sorry

end NUMINAMATH_GPT_matrix_vec_addition_l2406_240661


namespace NUMINAMATH_GPT_vans_needed_l2406_240623

theorem vans_needed (boys girls students_per_van total_vans : ℕ) 
  (hb : boys = 60) 
  (hg : girls = 80) 
  (hv : students_per_van = 28) 
  (t : total_vans = (boys + girls) / students_per_van) : 
  total_vans = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_vans_needed_l2406_240623


namespace NUMINAMATH_GPT_largest_n_l2406_240697

theorem largest_n (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃ n : ℕ, n > 0 ∧ n = 10 ∧ n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 12 := 
sorry

end NUMINAMATH_GPT_largest_n_l2406_240697


namespace NUMINAMATH_GPT_divide_nuts_equal_l2406_240622

-- Define the conditions: sequence of 64 nuts where adjacent differ by 1 gram
def is_valid_sequence (seq : List Int) :=
  seq.length = 64 ∧ (∀ i < 63, (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ + 1) ∨ (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ - 1))

-- Main theorem statement: prove that the sequence can be divided into two groups with equal number of nuts and equal weights
theorem divide_nuts_equal (seq : List Int) (h : is_valid_sequence seq) :
  ∃ (s1 s2 : List Int), s1.length = 32 ∧ s2.length = 32 ∧ (s1.sum = s2.sum) :=
sorry

end NUMINAMATH_GPT_divide_nuts_equal_l2406_240622


namespace NUMINAMATH_GPT_cameron_list_count_l2406_240678

-- Definitions
def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b
def is_perfect_square (n : ℕ) : Prop := ∃ m, n = m * m
def is_perfect_cube (n : ℕ) : Prop := ∃ m, n = m * m * m

-- The main statement
theorem cameron_list_count :
  let smallest_square := 25
  let smallest_cube := 125
  (∀ n : ℕ, is_multiple_of n 25 → smallest_square ≤ n → n ≤ smallest_cube) →
  ∃ count : ℕ, count = 5 :=
by 
  sorry

end NUMINAMATH_GPT_cameron_list_count_l2406_240678


namespace NUMINAMATH_GPT_pieces_of_gum_l2406_240664

variable (initial_gum total_gum given_gum : ℕ)

theorem pieces_of_gum (h1 : given_gum = 16) (h2 : total_gum = 54) : initial_gum = 38 :=
by
  sorry

end NUMINAMATH_GPT_pieces_of_gum_l2406_240664


namespace NUMINAMATH_GPT_contractor_laborers_l2406_240684

theorem contractor_laborers (x : ℕ) (h : 9 * x = 15 * (x - 6)) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_contractor_laborers_l2406_240684


namespace NUMINAMATH_GPT_twenty_four_times_ninety_nine_l2406_240643

theorem twenty_four_times_ninety_nine : 24 * 99 = 2376 :=
by sorry

end NUMINAMATH_GPT_twenty_four_times_ninety_nine_l2406_240643


namespace NUMINAMATH_GPT_inner_circle_radius_is_sqrt_2_l2406_240646

noncomputable def radius_of_inner_circle (side_length : ℝ) : ℝ :=
  let semicircle_radius := side_length / 4
  let distance_from_center_to_semicircle_center :=
    Real.sqrt ((side_length / 2) ^ 2 + (side_length / 2) ^ 2)
  let inner_circle_radius := (distance_from_center_to_semicircle_center - semicircle_radius)
  inner_circle_radius

theorem inner_circle_radius_is_sqrt_2 (side_length : ℝ) (h: side_length = 4) : 
  radius_of_inner_circle side_length = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_inner_circle_radius_is_sqrt_2_l2406_240646


namespace NUMINAMATH_GPT_find_certain_number_l2406_240647

theorem find_certain_number :
  ∃ C, ∃ A B, (A + B = 15) ∧ (A = 7) ∧ (C * B = 5 * A - 11) ∧ (C = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l2406_240647


namespace NUMINAMATH_GPT_find_smaller_number_l2406_240669

theorem find_smaller_number
  (x y : ℝ) (m : ℝ)
  (h1 : x - y = 9) 
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 :=
by 
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2406_240669


namespace NUMINAMATH_GPT_twentieth_term_arithmetic_sequence_eq_neg49_l2406_240668

-- Definitions based on the conditions
def a1 : ℤ := 8
def d : ℤ := 5 - 8
def a (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The proof statement
theorem twentieth_term_arithmetic_sequence_eq_neg49 : a 20 = -49 :=
by 
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_twentieth_term_arithmetic_sequence_eq_neg49_l2406_240668


namespace NUMINAMATH_GPT_shirley_eggs_start_l2406_240656

theorem shirley_eggs_start (eggs_end : ℕ) (eggs_bought : ℕ) (eggs_start : ℕ) (h_end : eggs_end = 106) (h_bought : eggs_bought = 8) :
  eggs_start = eggs_end - eggs_bought → eggs_start = 98 :=
by
  intros h_start
  rw [h_end, h_bought] at h_start
  exact h_start

end NUMINAMATH_GPT_shirley_eggs_start_l2406_240656


namespace NUMINAMATH_GPT_probability_of_two_co_presidents_l2406_240694

noncomputable section

def binomial (n k : ℕ) : ℕ :=
  if h : n ≥ k then Nat.choose n k else 0

def club_prob (n : ℕ) : ℚ :=
  (binomial (n-2) 2 : ℚ) / (binomial n 4 : ℚ)

def total_probability : ℚ :=
  (1/4 : ℚ) * (club_prob 6 + club_prob 8 + club_prob 9 + club_prob 10)

theorem probability_of_two_co_presidents : total_probability = 0.2286 := by
  -- We expect this to be true based on the given solution
  sorry

end NUMINAMATH_GPT_probability_of_two_co_presidents_l2406_240694


namespace NUMINAMATH_GPT_minimum_throws_for_repetition_of_sum_l2406_240660

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end NUMINAMATH_GPT_minimum_throws_for_repetition_of_sum_l2406_240660


namespace NUMINAMATH_GPT_students_wrote_word_correctly_l2406_240616

-- Definitions based on the problem conditions
def total_students := 50
def num_cat := 10
def num_rat := 18
def num_croc := total_students - num_cat - num_rat
def correct_cat := 15
def correct_rat := 15
def correct_total := correct_cat + correct_rat

-- Question: How many students wrote their word correctly?
-- Correct Answer: 8

theorem students_wrote_word_correctly : 
  num_cat + num_rat + num_croc = total_students 
  → correct_cat = 15 
  → correct_rat = 15 
  → correct_total = 30 
  → ∀ (num_correct_words : ℕ), num_correct_words = correct_total - num_croc 
  → num_correct_words = 8 := by 
  sorry

end NUMINAMATH_GPT_students_wrote_word_correctly_l2406_240616
