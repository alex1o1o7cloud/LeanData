import Mathlib

namespace people_in_line_l335_335440

theorem people_in_line (n : ℕ) (h : n = 10) : n + 1 = 11 :=
by
  rw h
  exact nat.add_one 10

end people_in_line_l335_335440


namespace number_of_integers_between_sqrt3_and_sqrt5_l335_335874

theorem number_of_integers_between_sqrt3_and_sqrt5 :
  let S := {x : ℕ | 9 < x ∧ x < 25} in
  S.card = 15 :=
by sorry

end number_of_integers_between_sqrt3_and_sqrt5_l335_335874


namespace find_A_l335_335818

/-- Given that the equation Ax + 10y = 100 has two distinct positive integer solutions, prove that A = 10. -/
theorem find_A (A x1 y1 x2 y2 : ℕ) (h1 : A > 0) (h2 : x1 > 0) (h3 : y1 > 0) 
  (h4 : x2 > 0) (h5 : y2 > 0) (distinct_solutions : x1 ≠ x2 ∧ y1 ≠ y2) 
  (eq1 : A * x1 + 10 * y1 = 100) (eq2 : A * x2 + 10 * y2 = 100) : 
  A = 10 := sorry

end find_A_l335_335818


namespace triangle_isosceles_l335_335996

theorem triangle_isosceles 
  (A B C : ℝ) (hTriangle : A + B + C = π) 
  (hCondition : 2 * cos B * sin A = sin C) : A = B :=
by
  -- The actual proof is omitted
  sorry

end triangle_isosceles_l335_335996


namespace arithmetic_sequence_sum_l335_335111

theorem arithmetic_sequence_sum {a_1 d : ℝ} (h1 : a_1 + 3 * d = 8) (h2 : 6 * d = 4) :
  let S : ℕ → ℝ := λ n, n / 2 * (2 * a_1 + (n - 1) * d) in
  S 19 = 228 :=
by
  -- Defining the sum of the arithmetic sequence as S_n
  let S := λ n, n / 2 * (2 * a_1 + (n - 1) * d)
  -- Apply the let definition of S to confirm the statement
  exact sorry

end arithmetic_sequence_sum_l335_335111


namespace intersection_M_N_l335_335138

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l335_335138


namespace nobel_prize_laureates_at_workshop_l335_335375

theorem nobel_prize_laureates_at_workshop :
  ∃ (T W W_and_N N_no_W X N : ℕ), 
    T = 50 ∧ 
    W = 31 ∧ 
    W_and_N = 16 ∧ 
    (N_no_W = X + 3) ∧ 
    (T - W = 19) ∧ 
    (N_no_W + X = 19) ∧ 
    (N = W_and_N + N_no_W) ∧ 
    N = 27 :=
by
  sorry

end nobel_prize_laureates_at_workshop_l335_335375


namespace identify_incorrect_statements_l335_335142

variables (m n : Type) [Line m] [Line n]
variables (α β γ : Type) [Plane α] [Plane β] [Plane γ]

theorem identify_incorrect_statements :
  (∀ h1 : α ⊥ β, ∀ h2 : α ∩ β = m, ∀ h3 : n ⊥ m, n ⊥ α ∨ n ⊥ β → false) ∧
  (∀ h4 : ¬(m ⊥ α), ¬(∀ l : Type, [Line l] → l ⊥ m → l ∈ α) → false) ∧
  (∃ h5 : α ∩ β = m, ∃ h6 : n ‖ m, ∀ h7 : n ∉ α, ∀ h8 : n ∉ β, n ‖ α ∧ n ‖ β) ∧
  (∀ h9 : α ⊥ β, ∀ h10 : m ‖ n, ∀ h11 : n ⊥ β, m ‖ α ∨ m ∈ α → false) → 
  (α ∩ β = m ∧ α ⊥ β ∧ n ⊥ m ∧ m ‖ n ∧ n ⊥ β ∧ m ∈ α → ∃ (d : list nat), d = [1, 2, 4])
:= 
by sorry

end identify_incorrect_statements_l335_335142


namespace intersection_M_N_l335_335116

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l335_335116


namespace train_crossing_time_l335_335401

theorem train_crossing_time (speed_kmph : ℝ) (length_meters : ℝ) (speed_mps : ℝ) (time_seconds : ℝ) :
  speed_kmph = 240 → length_meters = 1400 → speed_mps = (speed_kmph * 1000) / 3600 → time_seconds = length_meters / speed_mps → time_seconds = 21 := 
by
  intros h_speed h_length h_conv h_time
  rw [h_speed, h_length, h_conv] at h_time
  rw [h_time]
  norm_num  -- This simplifies numerical expressions.
  sorry     -- Placeholder for the actual proof

end train_crossing_time_l335_335401


namespace zeros_of_f_l335_335526

noncomputable def f (a : ℝ) (x : ℝ) :=
if x ≤ 1 then a + 2^x else (1/2) * x + a

theorem zeros_of_f (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a ∈ Set.Ico (-2) (-1/2) :=
sorry

end zeros_of_f_l335_335526


namespace sum_distinct_x2_y2_z2_l335_335285

/-
Given positive integers x, y, and z such that
x + y + z = 30 and gcd(x, y) + gcd(y, z) + gcd(z, x) = 10,
prove that the sum of all possible distinct values of x^2 + y^2 + z^2 is 404.
-/
theorem sum_distinct_x2_y2_z2 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 30) 
  (h_gcd : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) : 
  x^2 + y^2 + z^2 = 404 :=
sorry

end sum_distinct_x2_y2_z2_l335_335285


namespace intersection_points_l335_335047

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

theorem intersection_points :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ ¬(x = x ∧ y = y) → 0 = 1 :=
sorry

end intersection_points_l335_335047


namespace surface_area_cube_equals_l335_335506

theorem surface_area_cube_equals (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 10) :
  let v_prism := a * b * c in
  let edge_cube := Real.cbrt v_prism in
  let surface_area_cube := 6 * (edge_cube ^ 2) in
  surface_area_cube = 6 * (Real.cbrt (5 * 7 * 10) ^ 2) :=
by
  sorry

end surface_area_cube_equals_l335_335506


namespace combination_8_5_eq_56_l335_335841

theorem combination_8_5_eq_56 : nat.choose 8 5 = 56 :=
by
  sorry

end combination_8_5_eq_56_l335_335841


namespace std_dev_of_data_set_l335_335317

noncomputable def data_set : List ℝ := [5, 7, 7, 8, 10, 11]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.sum) / lst.length

noncomputable def variance (lst : List ℝ) : ℝ :=
  (lst.map (λ x => (x - mean lst) ^ 2)).sum / lst.length

noncomputable def standard_deviation (lst : List ℝ) : ℝ :=
  Real.sqrt (variance lst)

theorem std_dev_of_data_set :
  standard_deviation data_set = 2 :=
sorry

end std_dev_of_data_set_l335_335317


namespace determine_k_if_even_function_l335_335964

noncomputable def f (x k : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem determine_k_if_even_function (k : ℝ) (h_even: ∀ x : ℝ, f x k = f (-x) k ) : k = 1 :=
by
  sorry

end determine_k_if_even_function_l335_335964


namespace polygon_diagonals_and_interior_angles_l335_335383

theorem polygon_diagonals_and_interior_angles (n : ℕ) (h1 : n = 25) (h2 : convex_polygon n) : 
  (number_of_diagonals n = 275) ∧ (sum_of_interior_angles n > 4000) :=
by
  sorry

noncomputable def number_of_diagonals (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
180 * (n - 2)

def convex_polygon (n : ℕ) : Prop := true  -- This is a placeholder for the actual definition of a convex polygon.

end polygon_diagonals_and_interior_angles_l335_335383


namespace range_f_real_range_f_set_range_f_interval1_range_f_interval2_range_f_interval3_l335_335060

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

-- Problem 1: Range of f(x) when x ∈ ℝ
theorem range_f_real : set.range f = set.Ici 1 := by
sory

-- Problem 2: Range of f(x) when x ∈ {-1, 0, 1}
theorem range_f_set : set.range (λ x : ℝ, if x ∈ {-1, 0, 1} then some (f x) else none) = {1, 2, 5} := by
sory

-- Problem 3: Range of f(x) when x ∈ [-1, 0]
theorem range_f_interval1 : set.range (λ x : ℝ, if x ∈ set.Icc (-1 : ℝ) 0 then some (f x) else none) = set.Icc 2 5 := by
sory

-- Problem 4: Range of f(x) when x ∈ [2, 3]
theorem range_f_interval2 : set.range (λ x : ℝ, if x ∈ set.Icc (2 : ℝ) 3 then some (f x) else none) = set.Icc 2 5 := by
sory

-- Problem 5: Range of f(x) when x ∈ [-1, 2]
theorem range_f_interval3 : set.range (λ x : ℝ, if x ∈ set.Icc (-1 : ℝ) 2 then some (f x) else none) = set.Icc 1 5 := by
sory

end range_f_real_range_f_set_range_f_interval1_range_f_interval2_range_f_interval3_l335_335060


namespace units_digit_sum_sequential_l335_335039

theorem units_digit_sum_sequential (S : Fin 8 → Nat) (H1 : S 1 = 1 * 1!)
                                   (H2 : S 2 = 2 * 2!) (H3 : S 3 = 3 * 3!)
                                   (H4 : S 4 = 4 * 4!) (H5 : S 5 = 5 * 5!)
                                   (H6 : S 6 = 6 * 6!) (H7 : S 7 = 7 * 7!):
                                   (S 1 + S 2 + S 3 + S 4 + S 5 + S 6 + S 7) % 10 = 9 :=
by
  sorry

end units_digit_sum_sequential_l335_335039


namespace smallest_number_of_students_l335_335571

theorem smallest_number_of_students (n : ℕ) (h1 : 1 ≤ n) (h2 : (∀ i, i < 8 → get_score i = 120)) (h3 : (∀ i, 8 ≤ i → i < n → 72 ≤ get_score i)) (h4 : (sum_scores (list.range n) = 84 * n)) : n = 32 :=
by
  sorry

end smallest_number_of_students_l335_335571


namespace collinear_XYZ_l335_335144

open EuclideanGeometry

variables {A B C H P L M N X Y Z : Point}

-- Given conditions in the problem
variable (h_orthocenter : IsOrthocenter H A B C)
variable (h_L_perp : Perpendicular H L (Line.mk P A))
variable (h_M_perp : Perpendicular H M (Line.mk P B))
variable (h_N_perp : Perpendicular H N (Line.mk P C))
variable (h_X_ext : IntersectsExtension (Line.mk H L) (Line.mk B C) X)
variable (h_Y_ext : IntersectsExtension (Line.mk H M) (Line.mk A C) Y)
variable (h_Z_ext : IntersectsExtension (Line.mk H N) (Line.mk A B) Z)

-- Statement to be proved
theorem collinear_XYZ : Collinear X Y Z :=
begin
  sorry
end

end collinear_XYZ_l335_335144


namespace product_is_square_of_quadratic_l335_335275

noncomputable def quadratic_trinomial := ℝ → ℝ 
noncomputable def derivative (f : ℝ → ℝ) := ℝ → ℝ 

variables (f g : quadratic_trinomial)
variables (h : quadratic_trinomial)

axiom f_is_quadratic : f = λ x, a * (x - u) ^ 2 + b
axiom g_is_quadratic : g = λ x, c * (x - v) ^ 2 + d

axiom inequality_condition : ∀ x : ℝ, (derivative f x) * (derivative g x) ≥ |f x| + |g x|

theorem product_is_square_of_quadratic : 
  ∃ h : quadratic_trinomial, f * g = λ x, (h x) ^ 2 :=
  sorry

end product_is_square_of_quadratic_l335_335275


namespace min_value_and_binomial_coefficient_l335_335926

theorem min_value_and_binomial_coefficient :
  let f (x : ℝ) := abs (x - 1) + abs (x + 7)
  let minimum_value : ℝ := 8
  minimum_value = 8 → (binom 8 5 = 56) := 
by
  intro h
  have r_val: 2 * 5 - 8 = -2 := by linarith
  have coeff := binom 8 5
  exact coeff

sorry

end min_value_and_binomial_coefficient_l335_335926


namespace determinant_zero_l335_335066

noncomputable theory
open_locale matrix

def matrix_A (α β : ℝ) : matrix (fin 3) (fin 3) ℝ :=
  ![![0, real.cos α, -real.sin α], 
    ![-real.cos α, 0, real.cos β], 
    ![real.sin α, -real.cos β, 0]]

theorem determinant_zero (α β : ℝ) : 
  matrix.det (matrix_A α β) = 0 :=
by {
  sorry
}

end determinant_zero_l335_335066


namespace area_of_triangle_PQR_l335_335000

def point (x y : ℝ) := (x, y)
def slope (l : ℝ) : Prop := ∀ (x₁ y₁ x₂ y₂ : ℝ), y₂ - y₁ = l * (x₂ - x₁)

noncomputable def P := point 2 4
noncomputable def Q := point 1 0
noncomputable def R := point (2/3) 0

def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

def line_through (p₁ p₂ : ℝ × ℝ) : ℝ → ℝ := 
  λ x, p₁.2 + (p₂.2 - p₁.2) / (p₂.1 - p₁.1) * (x - p₁.1)

theorem area_of_triangle_PQR :
  slope 2 → slope 3 → area_triangle P Q R = 2 / 3 :=
by
  sorry

end area_of_triangle_PQR_l335_335000


namespace solve_for_x_l335_335182

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = (9 / (x / 3))^2) : x = 23.43 :=
by {
  sorry
}

end solve_for_x_l335_335182


namespace find_coordinates_collinear_points_l335_335113

noncomputable theory

open_locale classical

variables {x y : ℝ}

def point (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), (B.1 - A.1, B.2 - A.2, B.3 - A.3) = (λ * (C.1 - A.1), λ * (C.2 + A.2), λ * (C.3 - A.3))

theorem find_coordinates_collinear_points :
  collinear (point 1 -2 11) (point 4 2 3) (point x y 15) → x = -1/2 ∧ y = -4 :=
by {
  intros h,
  sorry -- proof goes here
}

end find_coordinates_collinear_points_l335_335113


namespace num_cons_sets_sum_to_18_l335_335546

theorem num_cons_sets_sum_to_18 : ∃ unique (a n : ℕ), (n ≥ 2) ∧ (2 * a * n + n ^ 2 - n = 36) :=
by sorry

end num_cons_sets_sum_to_18_l335_335546


namespace remaining_students_l335_335705

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l335_335705


namespace find_n_l335_335049

noncomputable def t : ℕ → ℚ
| 1       := 2
| (n + 1) := if (n + 1) % 2 = 0 then 2 + t ((n + 1) / 2) else 1 / t n

theorem find_n (n : ℕ) (h : t n = 23 / 97) : n = 1905 := 
sorry

end find_n_l335_335049


namespace banner_enlargement_l335_335382

/-- 
Given an original banner of width 5 feet and height 3 feet,
and an enlarged banner width of 15 feet,
we want to prove that the height of the enlarged banner, when proportions are maintained, is 9 feet.
-/
theorem banner_enlargement :
  ∀ (original_width original_height new_width : ℕ),
    original_width = 5 →
    original_height = 3 →
    new_width = 15 →
    ∃ new_height : ℕ, new_height = (new_width * original_height) / original_width ∧ new_height = 9 :=
by
  intros original_width original_height new_width hw hh hnw
  use (new_width * original_height) / original_width
  rw [hw, hh, hnw]
  norm_num
  sorry

end banner_enlargement_l335_335382


namespace magician_guarantee_three_of_clubs_l335_335783

-- Definitions corresponding to the identified conditions
def deck_size : ℕ := 52
def num_discarded : ℕ := 51
def magician_choice (n : ℕ) (from_left : bool) : ℕ := if from_left then n else deck_size + 1 - n
def is_edge_position (position : ℕ) : Prop := position = 1 ∨ position = deck_size

-- Statement of the problem, translated to Lean
theorem magician_guarantee_three_of_clubs (initial_pos : ℕ) (H : is_edge_position initial_pos) :
    ∃ strategy, ∀ spectator_choice, 
                (∃ remaining_cards, 
                  remaining_cards = deck_size - num_discarded + 1 ∧ 
                  three_of_clubs ∈ remaining_cards) :=
begin
  -- proving strategy exists if the initial position is at the edge
  sorry
end

end magician_guarantee_three_of_clubs_l335_335783


namespace total_books_left_l335_335655

def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

theorem total_books_left : sandy_books + tim_books - benny_lost_books = 19 :=
by
  sorry

end total_books_left_l335_335655


namespace proposition_truths_l335_335525

-- Define the propositions
def proposition1 (l1 l2 : Line) (p : Plane) : Prop :=
  (l1 ∥ p) ∧ (l2 ∥ p) → (l1 ∥ l2)

def proposition2 (l1 l2 : Line) (l : Line) : Prop :=
  (l1 ⊥ l) ∧ (l2 ⊥ l) → (l1 ∥ l2)

def proposition3 (p1 p2 : Plane) (l : Line) : Prop :=
  (l ⊥ p2) ∧ (l ∈ p1) → (p1 ⊥ p2)

def proposition4 (p1 p2 : Plane) (l1 l2 : Line) : Prop :=
  (p1 ⊥ p2) ∧ (l1 ∈ p1) ∧ (l2 = line_of_intersection p1 p2) → (l1 ⊥ l2) → (l1 ⊥ p2)

-- The problem statement
theorem proposition_truths :
  ∀ l1 l2 l : Line, ∀ p1 p2 : Plane,
  ¬ (proposition1 l1 l2 p1) ∧ ¬ (proposition2 l1 l2 l) ∧
  (proposition3 p1 p2 l) ∧ (proposition4 p1 p2 l1 l2) :=
by
  sorry

end proposition_truths_l335_335525


namespace not_coplanar_if_basis_l335_335553

open_locale vector_space

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b c : V)

def is_basis (a b c : V) : Prop := 
  linear_independent ℝ ![a, b, c] ∧ span ℝ (![a, b, c] : list V) = ⊤

theorem not_coplanar_if_basis (h_basis : is_basis a b c) :
  ¬ coplanar ℝ ({a + b, a - b, c} : set V) :=
sorry

end not_coplanar_if_basis_l335_335553


namespace janet_used_paper_clips_l335_335222

def initial_paper_clips := 85
def found_paper_clips := 20
def given_away_per_friend := 5
def number_of_friends := 3
def end_day_paper_clips := 26

theorem janet_used_paper_clips :
  let total_initial := initial_paper_clips + found_paper_clips in
  let total_given_away := given_away_per_friend * number_of_friends in
  let remaining_after_giveaway := total_initial - total_given_away in
  total_initial + found_paper_clips - (remaining_after_giveaway - end_day_paper_clips) = 64 :=
  sorry

end janet_used_paper_clips_l335_335222


namespace difference_of_roots_l335_335075

theorem difference_of_roots : 
  let a := 6 + 3 * Real.sqrt 5
  let b := 3 + Real.sqrt 5
  let c := 1
  ∃ x1 x2 : ℝ, (a * x1^2 - b * x1 + c = 0) ∧ (a * x2^2 - b * x2 + c = 0) ∧ x1 ≠ x2 
  ∧ x1 > x2 ∧ (x1 - x2) = (Real.sqrt 6 - Real.sqrt 5) / 3 := 
sorry

end difference_of_roots_l335_335075


namespace cost_of_nuts_l335_335405

/--
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. 
One kilogram of nuts costs a certain amount N and one kilogram of dried fruit costs $8. 
His purchases cost $56. Prove that one kilogram of nuts costs $12.
-/
theorem cost_of_nuts (N : ℝ) 
  (h1 : 3 * N + 2.5 * 8 = 56) 
  : N = 12 := by
  sorry

end cost_of_nuts_l335_335405


namespace general_term_formula_l335_335466

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.sqrt n)

theorem general_term_formula :
  ∀ (n : ℕ), a n = 1 / Real.sqrt n :=
by
  intros
  rfl

end general_term_formula_l335_335466


namespace trapezoid_combined_area_correct_l335_335044

noncomputable def combined_trapezoid_area_proof : Prop :=
  let EF : ℝ := 60
  let GH : ℝ := 40
  let altitude_EF_GH : ℝ := 18
  let trapezoid_EFGH_area : ℝ := (1 / 2) * (EF + GH) * altitude_EF_GH

  let IJ : ℝ := 30
  let KL : ℝ := 25
  let altitude_IJ_KL : ℝ := 10
  let trapezoid_IJKL_area : ℝ := (1 / 2) * (IJ + KL) * altitude_IJ_KL

  let combined_area : ℝ := trapezoid_EFGH_area + trapezoid_IJKL_area

  combined_area = 1175

theorem trapezoid_combined_area_correct : combined_trapezoid_area_proof := by
  sorry

end trapezoid_combined_area_correct_l335_335044


namespace ornamental_rings_remaining_l335_335454

theorem ornamental_rings_remaining :
  let r := 100 in
  let T := 200 + r in
  let rings_after_sale := T - (3 * T / 4) in
  let rings_after_mothers_purchase := rings_after_sale + 300 in
  rings_after_mothers_purchase - 150 = 225 :=
by
  sorry

end ornamental_rings_remaining_l335_335454


namespace john_spent_fraction_on_peripherals_l335_335604

-- Define the problem parameters
def computer_cost : ℝ := 1500
def base_video_card_cost : ℝ := 300
def upgraded_video_card_cost : ℝ := 2 * base_video_card_cost
def total_spent : ℝ := 2100

-- Define the condition of the amount spent on monitor and peripherals
def monitor_peripherals_cost : ℝ := total_spent - (computer_cost + upgraded_video_card_cost - base_video_card_cost)

-- Define the fraction
def fraction_spent_on_peripherals : ℝ := monitor_peripherals_cost / computer_cost

-- The theorem to prove
theorem john_spent_fraction_on_peripherals : fraction_spent_on_peripherals = 1 / 5 :=
by
  sorry

end john_spent_fraction_on_peripherals_l335_335604


namespace piglet_straws_l335_335334

theorem piglet_straws (total_straws : ℕ) (straws_adult_pigs_ratio : ℚ) (straws_piglets_ratio : ℚ) (number_piglets : ℕ) :
  total_straws = 300 →
  straws_adult_pigs_ratio = 3/5 →
  straws_piglets_ratio = 1/3 →
  number_piglets = 20 →
  (total_straws * straws_piglets_ratio) / number_piglets = 5 := 
by
  intros
  sorry

end piglet_straws_l335_335334


namespace div_decimals_l335_335431

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end div_decimals_l335_335431


namespace cats_to_dogs_ratio_l335_335311

theorem cats_to_dogs_ratio
    (cats dogs : ℕ)
    (ratio : cats / dogs = 3 / 4)
    (num_cats : cats = 18) :
    dogs = 24 :=
by
    sorry

end cats_to_dogs_ratio_l335_335311


namespace minimum_buses_needed_l335_335377

theorem minimum_buses_needed (s b : ℕ) (h1 : s = 540) (h2 : b = 45) : 
  ∃ n : ℕ, 45 * n ≥ s ∧ (∀ m : ℕ, 45 * m ≥ s → n ≤ m) :=
by {
  use 12,
  split,
  { calc 45 * 12 = 540 : by norm_num
         ...    ≥ 540 : by linarith },
  { assume m h,
    have : m ≥ 12 := by linarith,
    exact this },
  sorry
}

end minimum_buses_needed_l335_335377


namespace true_propositions_l335_335933

theorem true_propositions : 
  (∀ x : ℝ, x^3 < 1 → x^2 + 1 > 0) ∧ (∀ x : ℚ, x^2 = 2 → false) ∧ 
  (∀ x : ℕ, x^3 > x^2 → false) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by 
  -- proof goes here
  sorry

end true_propositions_l335_335933


namespace Modified_OHara_triple_example_l335_335326

theorem Modified_OHara_triple_example : 
  (∀ a b c x : ℕ, sqrt (a : ℝ) + sqrt (b : ℝ) + sqrt (c : ℝ) = (x : ℝ) → (a = 16 ∧ b = 64 ∧ c = 9) → x = 15) :=
by
  intros a b c x ha
  sorry

end Modified_OHara_triple_example_l335_335326


namespace pancakes_eaten_by_older_is_12_l335_335170

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end pancakes_eaten_by_older_is_12_l335_335170


namespace ratio_of_cream_l335_335224

theorem ratio_of_cream (coffee_init : ℕ) (joe_coffee_drunk : ℕ) (cream_added : ℕ) (joann_total_drunk : ℕ) 
  (joann_coffee_init : ℕ := coffee_init)
  (joe_coffee_init : ℕ := coffee_init) (joann_cream_init : ℕ := cream_added)
  (joe_cream_init : ℕ := cream_added)
  (joann_drunk_cream_ratio : ℚ := joann_cream_init / (joann_coffee_init + joann_cream_init)) :
  (joe_cream_init / (joann_cream_init - joann_total_drunk * (joann_drunk_cream_ratio))) = 
  (6 / 5) := 
by
  sorry

end ratio_of_cream_l335_335224


namespace factorable_polynomial_sum_l335_335234

theorem factorable_polynomial_sum :
  let T := (∑ b in { b : ℤ | ∃ r s : ℤ, (r + s = -b) ∧ (r * s = 1008 * b) }, b)
  in |T| = 181440 :=
by
  sorry

end factorable_polynomial_sum_l335_335234


namespace positive_value_m_l335_335480

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l335_335480


namespace cannot_form_Shape_D_l335_335355

def unit_square : Type := { s : ℕ // s = 1 }

-- Define each shape as a type
def Shape_A (squares : list unit_square) : Prop :=
  squares.length = 6 ∧
  (∃ (rows cols : ℕ), rows = 2 ∧ cols = 3 ∧ squares.group_by (λ s, s) = rows * cols)

def Shape_B (squares : list unit_square) : Prop :=
  squares.length = 6 ∧
  (∃ (line end : ℕ), line = 4 ∧ end = 2 ∧ squares.group_by (λ s, s) = line + end)

def Shape_C (squares : list unit_square) : Prop :=
  squares.length = 6 ∧
  (∃ (rows cols : ℕ), rows = 3 ∧ cols = 2 ∧ squares.group_by (λ s, s) = rows * cols)

def Shape_D (squares : list unit_square) : Prop :=
  squares.length >= 7 ∧
  (∃ (cross rows cols : ℕ), cross = 1 ∧ rows > 2 ∧ cols > 2 ∧ squares.group_by (λ s, s) = cross * (rows + cols))

theorem cannot_form_Shape_D (squares : list unit_square) : ¬Shape_D squares :=
by sorry

end cannot_form_Shape_D_l335_335355


namespace determinant_is_zero_l335_335064

open Matrix

noncomputable def given_matrix (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, Real.cos α, -Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_is_zero (α β : ℝ) : det (given_matrix α β) = 0 := by
  sorry

end determinant_is_zero_l335_335064


namespace total_emails_received_l335_335766

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l335_335766


namespace flagpole_height_l335_335389

theorem flagpole_height (h : ℕ) (shadow_flagpole : ℕ) (height_building : ℕ) (shadow_building : ℕ) (similar_conditions : Prop) 
  (H1 : shadow_flagpole = 45) 
  (H2 : height_building = 24) 
  (H3 : shadow_building = 60) 
  (H4 : similar_conditions) 
  (H5 : h / 45 = 24 / 60) : h = 18 := 
by 
sorry

end flagpole_height_l335_335389


namespace find_minimum_fuse_length_l335_335576

def safeZone : ℝ := 70
def fuseBurningSpeed : ℝ := 0.112
def personSpeed : ℝ := 7
def minimumFuseLength : ℝ := 1.1

theorem find_minimum_fuse_length (x : ℝ) (h1 : x ≥ 0):
  (safeZone / personSpeed) * fuseBurningSpeed ≤ x :=
by
  sorry

end find_minimum_fuse_length_l335_335576


namespace uncolored_node_not_vertex_of_original_hexagon_l335_335794

theorem uncolored_node_not_vertex_of_original_hexagon
  (n : ℕ) (hex : fin 6 → ℕ → ℕ × ℕ)
  (division_points : ∀ (i : fin 6), fin n → ℝ × ℝ)
  (equilateral_triangle : ℕ × ℕ → ℕ → list (ℕ × ℕ))
  (colorable_nodes : list (ℕ × ℕ)) :
  (∃ (hex_vertex : ℕ × ℕ), hex_vertex ∈ division_points.val (0:fin 6) → 
     hex_vertex ∉ colorable_nodes) →
  False :=
begin
  sorry
end

end uncolored_node_not_vertex_of_original_hexagon_l335_335794


namespace no_unique_pairings_of_bracelets_l335_335867

theorem no_unique_pairings_of_bracelets :
  ∃ (bracelets : Finset ℕ) (days : Finset (Finset ℕ)),
  bracelets.card = 100 ∧ (∀ day ∈ days, (day.card = 3 ∧ day ⊆ bracelets)) ∧
  ¬(∀ pair ∈ bracelets.powersetLen 2, ∃! day ∈ days, pair ⊆ day) :=
by sorry

end no_unique_pairings_of_bracelets_l335_335867


namespace smallest_integer_satisfying_inequality_l335_335315

theorem smallest_integer_satisfying_inequality :
  ∃ n (h: n > 0), (√n - √(n - 1) < 0.01) ∧
  (∀ m (hm: m > 0), √m - √(m - 1) < 0.01 → n ≤ m) :=
begin
  use 2501,
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
    -- √2501 - √2500 < 0.01 by given problem.
    sorry,
  },
  {
    intros,
    -- Prove minimality, if √m - √(m - 1) < 0.01, then m ≥ 2501 by given problem.
    sorry,
  }
end

end smallest_integer_satisfying_inequality_l335_335315


namespace magician_guarantee_success_l335_335781

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end magician_guarantee_success_l335_335781


namespace sufficient_but_not_necessary_m_4_l335_335351

def line1 (m : ℝ) : ℝ → ℝ → ℝ := λ x y, (2 * m - 4) * x + (m + 1) * y + 2
def line2 (m : ℝ) : ℝ → ℝ → ℝ := λ x y, (m + 1) * x - m * y + 3

def slope (a b c : ℝ) : ℝ := -a / b

def perpendicular (m : ℝ) : Prop :=
  slope (2 * m - 4) (m + 1) 2 * slope (m + 1) (-m) 3 = -1

theorem sufficient_but_not_necessary_m_4 :
  (∀ x y : ℝ, line1 4 x y = 0 → line2 4 x y = 0 → perpendicular 4) ∧
  (∃ m : ℝ, m ≠ 4 ∧ perpendicular m) :=
sorry

end sufficient_but_not_necessary_m_4_l335_335351


namespace find_x_l335_335447

theorem find_x (x : ℚ) : |x + 3| = |x - 4| → x = 1/2 := 
by 
-- Add appropriate content here
sorry

end find_x_l335_335447


namespace smallest_pos_integer_n_l335_335662

theorem smallest_pos_integer_n 
  (x y : ℤ)
  (hx: ∃ k : ℤ, x = 8 * k - 2)
  (hy : ∃ l : ℤ, y = 8 * l + 2) :
  ∃ n : ℤ, n > 0 ∧ ∃ (m : ℤ), x^2 - x*y + y^2 + n = 8 * m ∧ n = 4 := by
  sorry

end smallest_pos_integer_n_l335_335662


namespace order_of_exponentials_l335_335496

theorem order_of_exponentials :
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  a < d ∧ d < b ∧ b < c :=
by
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  sorry

end order_of_exponentials_l335_335496


namespace travel_time_l335_335629

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l335_335629


namespace maximize_triangle_perimeter_l335_335412

-- Define the necessary parameters and conditions
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

-- Given angle C
variable (angle_C : real)

-- Given side AB
variables (A B : A) (AB : ℝ) 

-- To prove the triangle with largest perimeter
theorem maximize_triangle_perimeter 
  (hC : is_angle angle_C)
  (hAB : dist A B = AB) 
  : ∃ C, isosceles_triangle A B C ∧ (perimeter A B C) = max_perimeter A B :=
sorry

end maximize_triangle_perimeter_l335_335412


namespace inequality_holds_for_all_l335_335562

theorem inequality_holds_for_all (m : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 - m * x - 1) < 0) : -4 < m ∧ m ≤ 0 := 
sorry

end inequality_holds_for_all_l335_335562


namespace range_of_alpha_l335_335488

noncomputable def inequality_holds_for_all (α : ℝ) : Prop :=
  ∀ x : ℝ, 8 * x - (8 * complex.I * α) * x + real.cos α ^ 2 ≥ 0

theorem range_of_alpha :
  ∀ α : ℝ, 0 ≤ α → α ≤ π → inequality_holds_for_all α ↔ (0 ≤ α ∧ α ≤ π/6) ∨ (5*π/6 ≤ α ∧ α ≤ π) :=
begin
  intro α,
  intro h1,
  intro h2,
  sorry,
end

end range_of_alpha_l335_335488


namespace cost_of_one_dozen_lemons_l335_335336

theorem cost_of_one_dozen_lemons :
    (∀ (cost_per_lemon : ℕ), cost_per_lemon = 5) →
    (∀ (lemons_in_a_dozen : ℕ), lemons_in_a_dozen = 10) →
    ∃ (cost_of_one_dozen : ℕ), cost_of_one_dozen = 50 :=
begin
  intros,
  use 50,
  -- Proof skipped
  sorry
end

end cost_of_one_dozen_lemons_l335_335336


namespace sample_mean_experimental_group_median_combined_group_significant_difference_weight_increase_l335_335026

noncomputable def controlWeights : List ℝ := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]
noncomputable def experimentalWeights : List ℝ := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

theorem sample_mean_experimental_group : 
  (List.sum experimentalWeights) / (experimentalWeights.length : ℝ) = 19.8 :=
by
  sorry

noncomputable def combinedWeights : List ℝ := (controlWeights ++ experimentalWeights).qsort (<=)

theorem median_combined_group :
  (combinedWeights.get! 19 + combinedWeights.get! 20) / 2 = 23.4 :=
by
  sorry

theorem significant_difference_weight_increase :
  let a := 6; let b := 14; let c := 14; let d := 6; let n := 40;
  let K2 := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d)) in
  K2 > 3.841 :=
by
  sorry

end sample_mean_experimental_group_median_combined_group_significant_difference_weight_increase_l335_335026


namespace probability_specific_sequence_l335_335085

theorem probability_specific_sequence : 
  let total_arrangements := Nat.choose 8 5 in
  let specific_arrangement := 1 in
  (specific_arrangement / total_arrangements : ℚ) = (1 / 560 : ℚ) :=
by
  let total_arrangements := 560
  let specific_arrangement := 1
  sorry

end probability_specific_sequence_l335_335085


namespace f_properties_l335_335924

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * (sin x)^2 - sqrt 3 / 2

theorem f_properties :
  (∀ x, -1 <= f x ∧ f x <= 1) ∧
  (∀ x, f (x + π) = f x) ∧
  (f (π / 6) = 0) ∧
  (∀ x, f x = sin (2 * x - π / 3)) :=
begin
  sorry
end

end f_properties_l335_335924


namespace range_b_intersects_ellipse_l335_335086

open Real

noncomputable def line_intersects_ellipse (b : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < π → ∃ x y : ℝ, x = 2 * cos θ ∧ y = 4 * sin θ ∧ y = x + b

theorem range_b_intersects_ellipse :
  ∀ b : ℝ, line_intersects_ellipse b ↔ b ∈ Set.Icc (-2 : ℝ) (2 * sqrt 5) :=
by
  sorry

end range_b_intersects_ellipse_l335_335086


namespace largest_valid_subset_size_l335_335016

def is_valid_subset (S : Set ℕ) : Prop := ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b * 4

-- let universal_set be integers from 1 to 200
def universal_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 200}

noncomputable def largest_subset_size : ℕ := 188

theorem largest_valid_subset_size :
  ∃ (S : Set ℕ), is_valid_subset S ∧ universal_set ⊆ S ∧ S.card = largest_subset_size := sorry

end largest_valid_subset_size_l335_335016


namespace ara_final_height_is_59_l335_335822

noncomputable def initial_shea_height : ℝ := 51.2
noncomputable def initial_ara_height : ℝ := initial_shea_height + 4
noncomputable def final_shea_height : ℝ := 64
noncomputable def shea_growth : ℝ := final_shea_height - initial_shea_height
noncomputable def ara_growth : ℝ := shea_growth / 3
noncomputable def final_ara_height : ℝ := initial_ara_height + ara_growth

theorem ara_final_height_is_59 :
  final_ara_height = 59 := by
  sorry

end ara_final_height_is_59_l335_335822


namespace triangle_is_isosceles_l335_335994

noncomputable def is_isosceles {P Q R : Type*} [metric_space P] [metric_space Q] [metric_space R]
  (A : P) (B : Q) (C : R) (U : P) (V : Q) : Prop :=
dist A U = dist A V

theorem triangle_is_isosceles 
  (A B C X Y O1 O2 U V : Type*) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space Y]
  [metric_space O1] [metric_space O2] [metric_space U] [metric_space V]
  (h₁ : triangle A B C)
  (h₂ : on_line X B C)
  (h₃ : on_line Y B C)
  (h₄ : order B X Y C)
  (h₅ : dist B X * dist A C = dist C Y * dist A B)
  (h₆ : circumcenter A C X = O1)
  (h₇ : circumcenter A B Y = O2)
  (h₈ : intersect O1 O2 A B = U) 
  (h₉ : intersect O1 O2 A C = V) 
  : is_isosceles A U V :=
sorry

end triangle_is_isosceles_l335_335994


namespace a_general_formula_max_k_exists_l335_335899

-- Definitions based on problem conditions
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence a
def a (n : ℕ) : ℕ
| 1     := 2
| (n+1) := S n + 2

-- The general term formula for the sequence a_n should be 2^n
theorem a_general_formula (n : ℕ) : a n = 2^n := sorry

-- Additional sequences b_n and T_n based on the problem conditions
def b (n : ℕ) : ℝ := 1 / Real.log 2 (a n)
def T (n : ℕ) : ℝ := ∑ i in Finset.range (n+1) (2*n+1), b (n+i)

-- Proving that there exists a maximum k such that T_n > k/12 always holds, and k is 5
theorem max_k_exists (n : ℕ) : ∃ k : ℕ, (∀ m ≥ n, T m > k / 12) ∧ k = 5 :=
  sorry

end a_general_formula_max_k_exists_l335_335899


namespace sin_A_value_projection_BA_BC_l335_335969

theorem sin_A_value (A B C : ℝ) (a b c : ℝ) :
    cos (A - B) * cos B - sin (A - B) * sin (A + C) = -3 / 5 → sin A = 4 / 5 :=
by
  sorry

theorem projection_BA_BC (A B : ℝ) (a b c : ℝ) :
    a = 4 * sqrt 2 → b = 5 → (cos (A - B) * cos B - sin (A - B) * sin (A + _)) = -3 / 5 → 
    let sin_B := b * sin A / a,
    let c := sqrt (a^2 + b^2 - 2 * a * b * cos A) in
    let proj := c * cos B in proj = sqrt 2 / 2 :=
by
  sorry

end sin_A_value_projection_BA_BC_l335_335969


namespace proof_abc_div_def_l335_335556

def abc_div_def (a b c d e f : ℚ) : Prop := 
  a / b = 1 / 3 ∧ b / c = 2 ∧ c / d = 1 / 2 ∧ d / e = 3 ∧ e / f = 1 / 8 → (a * b * c) / (d * e * f) = 1 / 16

theorem proof_abc_div_def (a b c d e f : ℚ) :
  abc_div_def a b c d e f :=
by 
  sorry

end proof_abc_div_def_l335_335556


namespace monotonic_intervals_range_of_a_sum_greater_than_two_l335_335529

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / (a * x)

-- Question I: Monotonic intervals
theorem monotonic_intervals (a : ℝ) (ha : a > 0) : 
  (∀ x, (0 < x ∧ x < Real.exp 1) → 0 < deriv (λ x, f x a) x) ∧ 
  (∀ x, (x > Real.exp 1) → deriv (λ x, f x a) x < 0) :=
sorry

-- Question II: Range of a given f(x) ≤ x - 1/a
theorem range_of_a (a : ℝ) (h : ∀ x, 0 < x → f x a ≤ x - 1 / a) : 1 ≤ a :=
sorry

-- Question III: x₂ ln(x₁) + x₁ ln(x₂) = 0 implies x₁ + x₂ > 2
theorem sum_greater_than_two (x₁ x₂ : ℝ) (h : x₂ ≠ x₁) (hx : x₂ * Real.log x₁ + x₁ * Real.log x₂ = 0) : x₁ + x₂ > 2 :=
sorry

end monotonic_intervals_range_of_a_sum_greater_than_two_l335_335529


namespace measure_of_angle_e_l335_335059

def cube : Type :=
{ v : Type // v → ℝ × ℝ × ℝ }

noncomputable def angle_e_formed_by_space_diagonals (s : ℝ) [fact (s > 0)] : ℝ :=
real.arccos (real.sqrt 3 / 3)

theorem measure_of_angle_e (s : ℝ) [fact (s > 0)] : 
  angle_e_formed_by_space_diagonals s = real.arccos (real.sqrt 3 / 3) := 
sorry

end measure_of_angle_e_l335_335059


namespace cara_possible_pairs_l335_335041

theorem cara_possible_pairs :
  let total_people := 7
  let friends := total_people - 1
  (friends.choose 2) = 15 :=
by
  let total_people := 7
  let friends := total_people - 1
  have h : (friends.choose 2) = 15 := by
    rw [Nat.choose_eq_factorial_div_factorial (5+1) 2]
    norm_num
  exact h

end cara_possible_pairs_l335_335041


namespace pointP_in_quadrants_II_IV_l335_335671

noncomputable def pointP_coordinates : set (ℝ × ℝ) :=
  { (x,y) | (2 * x - 1) * (y + 1) = 0 }

theorem pointP_in_quadrants_II_IV (x y : ℝ) (h : (x, y) ∈ pointP_coordinates) : (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) :=
by
  sorry

end pointP_in_quadrants_II_IV_l335_335671


namespace minimum_value_of_f_l335_335469

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x) + (1 / (x^2 + 1 / x))

theorem minimum_value_of_f : ∀ (x : ℝ), x > 0 → f x ≥ f (real.cbrt (1 / 2)) :=
by
  sorry

end minimum_value_of_f_l335_335469


namespace sum_S_120_l335_335108

-- Define the sequence {a_n} where a_n = n
def a (n : ℕ) : ℕ := n

-- Define the modified sequence {d_n} based on the problem's condition
def d : ℕ → ℕ 
| 0     := 1
| (n+1) := if n < 9 then 2 else a (n - 5)

-- Define the sum S_n which is the sum of the first n terms of the sequence {d_n}
def S (n : ℕ) : ℕ := (Finset.range n).sum d

-- Prove that S_{120} equals 245
theorem sum_S_120 : S 120 = 245 := 
by
  -- Insert proof steps here
  sorry

end sum_S_120_l335_335108


namespace cookies_to_pints_l335_335331

-- Define the conditions as constants
constant quarts_per_18_cookies : ℕ := 3 -- 3 quarts to bake 18 cookies
constant pints_per_quart : ℕ := 2 -- 2 pints in a quart
constant cookies_baked : ℕ := 6 -- We want to find the number of pints needed to bake 6 cookies

-- We use a theorem to prove the final result
theorem cookies_to_pints (q18 : ℕ) (pq : ℕ) (c6 : ℕ) (h₁ : q18 = quarts_per_18_cookies)
    (h₂ : pq = pints_per_quart) (h₃ : c6 = cookies_baked) : (q18 * pq * c6) / 18 = 2 := by
  -- This is where the proof would go, but we will use 'sorry' for now.
  sorry

end cookies_to_pints_l335_335331


namespace jordan_trapezoid_height_l335_335429

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height_l335_335429


namespace range_of_a_l335_335527

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, ∀ y ∈ Set.Ioo (-1 : ℝ) 0, f a x ≤ f a y) →
  a ∈ Set.Ioo (-1 : ℝ) (-1 / (3 * Real.exp 1)) :=
sorry

end range_of_a_l335_335527


namespace roots_square_sum_l335_335554

theorem roots_square_sum (r s p q : ℝ) 
  (root_cond : ∀ x : ℝ, x^2 - 2 * p * x + 3 * q = 0 → (x = r ∨ x = s)) :
  r^2 + s^2 = 4 * p^2 - 6 * q :=
by
  sorry

end roots_square_sum_l335_335554


namespace cyclic_quad_iff_rectangle_l335_335384

/-- A figure is a cyclic quadrilateral if and only if it is a rectangle. -/
theorem cyclic_quad_iff_rectangle (Q : Type) [quadrilateral Q] : (cyclic_quadrilateral Q) ↔ (rectangle Q) := 
sorry

end cyclic_quad_iff_rectangle_l335_335384


namespace prove_arithmetic_seq_sum_first_n_terms_inequality_C_l335_335523

-- Problem statement:
variables {a : ℕ → ℕ} {n : ℕ}

-- Given conditions
def geometric_seq : Prop := ∀ n, a(n + 1) - 2 * a(n) = 2^n
def a1 : Prop := a 1 = 1
def a2 : Prop := a 2 = 4
def C : ℕ → ℝ := λ n, (2 * a n - 2 * n) / n

-- Problems to be proven
theorem prove_arithmetic_seq : 
  geometric_seq ∧ a1 ∧ a2 → 
  ∀ n, ∃ d, (a n) / (2^n) = (a 1) / (2^1) + (n - 1) * d := sorry

theorem sum_first_n_terms :
  geometric_seq ∧ a1 ∧ a2 → 
  ∑ i in finset.range n, a i = (n - 1) * (2^n) + 1 := sorry

theorem inequality_C :
  geometric_seq ∧ a1 ∧ a2 → 
  ∀ n ≥ 2, 
    (1 / 2 - (1 / 2)^n < ∑ i in finset.range (n-1), (1 / C (i+2))) ∧ 
    (∑ i in finset.range (n-1), (1 / C (i+2)) ≤ 1 - (1 / 2)^(n-1)) := sorry

end prove_arithmetic_seq_sum_first_n_terms_inequality_C_l335_335523


namespace angle_AD_BC_l335_335591

variables (a b c : ℝ)
variables (AD BD AB CD : ℝ)
variables (θ : ℝ)

-- Define the conditions
axiom h1 : AD = a
axiom h2 : BD = b
axiom h3 : AB = c
axiom h4 : CD = c
axiom h5 : ∠DAB + ∠BAC + ∠DAC = 180
axiom h6 : ∠DBA + ∠ABC + ∠DBC = 180

-- Define the angle between skew lines
def angle_between_skew_lines (x y : ℝ) := arccos (abs (b^2 - c^2) / a^2)

-- Prove that this angle corresponds to the given conditions
theorem angle_AD_BC : angle_between_skew_lines AD BD = θ :=
  sorry

end angle_AD_BC_l335_335591


namespace min_sum_of_3x3_table_with_distinct_sums_l335_335568

theorem min_sum_of_3x3_table_with_distinct_sums : 
  ∃ (table : Fin 3 × Fin 3 → ℕ), 
  (∀ i j : Fin 3, table i j ≥ 1) ∧ 
  (∀ i1 i2 : Fin 3, i1 ≠ i2 → (∑ j, table i1 j) ≠ (∑ j, table i2 j)) ∧ 
  (∀ j1 j2 : Fin 3, j1 ≠ j2 → (∑ i, table i j1) ≠ (∑ i, table i j2)) ∧ 
  (∑ i j, table i j = 17) := sorry

end min_sum_of_3x3_table_with_distinct_sums_l335_335568


namespace total_salary_after_strict_manager_l335_335198

-- Definitions based on conditions
def total_initial_salary (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  500 * x + (Finset.sum (Finset.range y) s) = 10000

def kind_manager_total (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  1500 * x + (Finset.sum (Finset.range y) s) + 1000 * y = 24000

def strict_manager_total (x y : ℕ) : ℕ :=
  500 * (x + y)

-- Lean statement to prove the required
theorem total_salary_after_strict_manager (x y : ℕ) (s : ℕ → ℕ) 
  (h_total_initial : total_initial_salary x y s) (h_kind_manager : kind_manager_total x y s) :
  strict_manager_total x y = 7000 := by
  sorry

end total_salary_after_strict_manager_l335_335198


namespace projection_vector_ratio_l335_335305

open Matrix

theorem projection_vector_ratio :
  let M := ![
    ![(3 : ℚ) / 17, -8 / 17],
    ![-8 / 17, 15 / 17]
  ]
  ∃ (x y : ℚ), M.mul_vec ![x, y] = ![x, y] → y / x = 7 / 4 :=
by
  intro M
  sorry

end projection_vector_ratio_l335_335305


namespace hyperbola_equation_l335_335076

def HyperbolaEquation (x y a b k : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = k

theorem hyperbola_equation 
  (x y : ℝ)
  (point_on_hyperbola : (x, y) = (2, 3))
  (asymptotes : ∀ x, y = √3 * x ∨ y = -√3 * x) : 
  HyperbolaEquation x y 1 √3 1 :=
by
  sorry

end hyperbola_equation_l335_335076


namespace ellipse_foci_x_axis_l335_335962

theorem ellipse_foci_x_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) : 0 < a ∧ a < b :=
sorry

end ellipse_foci_x_axis_l335_335962


namespace complex_numbers_angle_condition_l335_335598

theorem complex_numbers_angle_condition
  (n : ℕ)
  (Z : Fin n → ℂ)
  (hZ_sum : ∑ i, Z i = 0) :
  ∃ i j : Fin n, i ≠ j ∧ abs (arg (Z i) - arg (Z j)) ≥ (2 * Real.pi / 3) :=
sorry

end complex_numbers_angle_condition_l335_335598


namespace part1_part2_l335_335968

theorem part1 (a b c C : ℝ) (h : b - 1/2 * c = a * Real.cos C) (h1 : ∃ (A B : ℝ), Real.sin B - 1/2 * Real.sin C = Real.sin A * Real.cos C) :
  ∃ A : ℝ, A = 60 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 4 * (b + c) = 3 * b * c) (h2 : a = 2 * Real.sqrt 3) (h3 : b - 1/2 * c = a * Real.cos 60)
  (h4 : ∀ (A : ℝ), A = 60) : ∃ S : ℝ, S = 2 * Real.sqrt 3 :=
sorry

end part1_part2_l335_335968


namespace shape_area_is_36_l335_335211

def side_length : ℝ := 3
def num_squares : ℕ := 4
def area_square : ℝ := side_length ^ 2
def total_area : ℝ := num_squares * area_square

theorem shape_area_is_36 :
  total_area = 36 := by
  sorry

end shape_area_is_36_l335_335211


namespace triangle_area_PS_R_l335_335593

open Real

theorem triangle_area_PS_R (PQ QR PR : ℕ) (angle_PQR : ∠ PQR = 90) (PS_angle_bisector : PS is_angle_bisector) :
  PQ = 72 → QR = 29 → PR = 79 → 
  area_of_triangle PSR = 546 :=
by 
  intros hPQ hQR hPR 
  sorry

end triangle_area_PS_R_l335_335593


namespace range_of_m_l335_335149

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (h1 : ∀ x : ℝ, f (-x) = f x)
                   (h2 : ∀ a b : ℝ, a ≠ b → a ≤ 0 → b ≤ 0 → (f a - f b) / (a - b) < 0)
                   (h3 : f (m + 1) < f 2) : 
  ∃ m : ℝ, -3 < m ∧ m < 1 :=
sorry

end range_of_m_l335_335149


namespace complement_of_M_in_U_l335_335538

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M_in_U : compl M ∩ U = {1, 2, 3} :=
by
  sorry

end complement_of_M_in_U_l335_335538


namespace part1_part2_l335_335589

-- Define the conditions
def parametric_equations_of_line (t : ℝ) :=
  (x, y) = (1 + (1 / 2) * t, (sqrt 3 / 2) * t)

def polar_coordinate_equation_of_curve (ρ θ : ℝ) :=
  ρ * (sin (θ / 2))^2 = 1

-- Part 1: Finding the rectangular coordinate equation and polar coordinates of intersection points
theorem part1 (θ : ℝ) (ρ x y : ℝ) :
  polar_coordinate_equation_of_curve ρ θ →
  (ρ = sqrt (x^2 + y^2) ∧ x = ρ * cos θ) →
  (∃ y, y^2 = 4 * x + 4) ∧ 
  (∃θ₁ θ₂, θ₁ = π / 2 ∧ θ₂ = 3 * π / 2 ∧ 
    ((x = 0 ∧ y = 2 ∧ ρ = 2 ∧ θ = θ₁) ∨ 
     (x = 0 ∧ y = -2 ∧ ρ = 2 ∧ θ = θ₂))) :=
by
  intro h1 h2
  sorry

-- Part 2: Finding the value of 1/|PA| + 1/|PB|
theorem part2 (t : ℝ) (t1 t2 : ℝ) :
  let (A, B) := ((1 + (1 / 2) * t1, (sqrt 3 / 2) * t1), (1 + (1 / 2) * t2, (sqrt 3 / 2) * t2))
  let P := (1, 0)
  (polar_coordinate_equation_of_curve ρ θ → 
  parametric_equations_of_line t1 ∧
  parametric_equations_of_line t2 ∧
  C_intersects_line_l A B t →
  t1 + t2 = 8 / 3 ∧ -t1 * t2 = 32 / 3) →
  (1 / abs (PA A P) + 1 / abs (PB B P)) = sqrt 7 / 4 :=
by
  intro h1 h2
  sorry

end part1_part2_l335_335589


namespace value_of_second_part_l335_335958

theorem value_of_second_part 
  (total_amount : ℝ := 782) 
  (part1_ratio : ℝ := 1/2) 
  (part2_ratio : ℝ := 1/3) 
  (part3_ratio : ℝ := 3/4) :
  let total_proportion := part1_ratio + part2_ratio + part3_ratio in
  let value_of_one_part := total_amount / total_proportion in
  let value_of_second_part := value_of_one_part * part2_ratio in
  value_of_second_part ≈ 164.56 := by sorry

end value_of_second_part_l335_335958


namespace hexagon_dot_count_l335_335324

theorem hexagon_dot_count (n : ℕ) : 
  (n = 4) → (∑ i in finset.range (n-1), 6 * i + 1) + (6 * (n-1)) = 37 :=
by { intros, sorry }

end hexagon_dot_count_l335_335324


namespace max_circumference_of_circle_inside_parabola_l335_335006

-- Define the problem
def inside_parabola (x y : ℝ) : Prop := x^2 ≤ 4 * y
def passes_through_vertex (x y : ℝ) : Prop := x = 0 ∧ y = 0

def maximum_circumference : Prop :=
  ∀ (C : ℝ → ℝ → Prop) (x y : ℝ), 
    (inside_parabola x y ∧ passes_through_vertex x y ∧ C x y) →
    C x y = 4 * Real.pi

-- The theorem statement
theorem max_circumference_of_circle_inside_parabola : maximum_circumference :=
  sorry

end max_circumference_of_circle_inside_parabola_l335_335006


namespace kings_requirement_proof_l335_335777

open Classical

-- Define the conditions
def conditions (n : ℕ) : Prop :=
  ∃ (cities : Finset ℕ) (roads : Finset (ℕ × ℕ)),
    cities.card = n ∧ roads.card = n - 1 ∧
    (∀ d, d ∈ (finset.range (n * (n - 1) / 2)).attach ∧
    (∀ x y ∈ cities, shortest_path_dist cities roads x y = d))

-- Prove the cases for n = 6 and n = 1986
def kings_requirement_can_be_met_case_1: Prop :=
  conditions 6

def kings_requirement_cannot_be_met_case_2: Prop :=
  ¬ conditions 1986

-- Statement we need to prove
theorem kings_requirement_proof:
  kings_requirement_can_be_met_case_1 ∧ kings_requirement_cannot_be_met_case_2 :=
begin
  sorry,
end

end kings_requirement_proof_l335_335777


namespace probability_of_event_A_l335_335335

def probability_event_A : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes

-- Statement of the theorem
theorem probability_of_event_A :
  probability_event_A = 1 / 6 :=
by
  -- This is where the proof would go, replaced with sorry for now.
  sorry

end probability_of_event_A_l335_335335


namespace probability_intersect_first_quadrant_l335_335930

variable (a : ℝ)
variable h1 : a ∈ Set.Icc (-1 : ℝ) 5           -- a in the interval [-1, 5]
variable h2 : a ≠ -1                            -- a is not -1

noncomputable def line1 (x y : ℝ) : Prop := x - 2 * y - 1 = 0
noncomputable def line2 (x y : ℝ) : Prop := a * x + 2 * y + 2 * a = 0

theorem probability_intersect_first_quadrant : 
  ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ (0 < x ∧ 0 < y) → 
  (Set.Icc (-1 : ℝ) (0 : ℝ)).measure / (Set.Icc (-1 : ℝ) (5 : ℝ)).measure = 1 / 6 := by
  sorry

end probability_intersect_first_quadrant_l335_335930


namespace problem_solution_l335_335609

theorem problem_solution (a b c d x : ℚ) 
  (h1 : 2 * a + 2 = x) 
  (h2 : 3 * b + 3 = x) 
  (h3 : 4 * c + 4 = x) 
  (h4 : 5 * d + 5 = x) 
  (h5 : 2 * a + 3 * b + 4 * c + 5 * d + 6 = x) 
  : 2 * a + 3 * b + 4 * c + 5 * d = -10 / 3 := 
by 
  sorry

end problem_solution_l335_335609


namespace cost_price_souvenirs_maximum_profit_souvenirs_l335_335193

-- Statement for Part (1)
theorem cost_price_souvenirs :
  ∃ (x : ℕ), 10400 / x = 14000 / (x + 9) ∧ x = 26 ∧ (x + 9) = 35 :=
begin
  sorry
end

-- Statement for Part (2)
theorem maximum_profit_souvenirs :
  ∃ (m n : ℕ), 
  40 + 4 * m + 80 - 2 * n = 140 ∧ 
  n = 2 * m - 10 ∧ 
  let w := -12 * m^2 + 240 * m + 800 in 
  w = 2000 :=
begin
  sorry
end

end cost_price_souvenirs_maximum_profit_souvenirs_l335_335193


namespace normal_distribution_probability_l335_335792

noncomputable 
def normal_distribution (μ σ : ℝ) : probability_mass_function ℝ := sorry -- assuming a normal distribution function

variables {ξ : ℝ} {a : ℝ}

-- Given conditions
axiom xi_normal : normal_distribution 10 100 = ξ
axiom P_greater_than_11 : P(ξ > 11) = a

-- Prove the Lean theorem
theorem normal_distribution_probability :
  P(9 < ξ ∧ ξ ≤ 11) = 1 - 2 * a :=
    sorry

end normal_distribution_probability_l335_335792


namespace number_of_machines_is_four_l335_335063

-- Define the variables according to the conditions
def quarters_per_machine := 80
def dimes_per_machine := 100
def nickels_per_machine := 50
def pennies_per_machine := 120
def total_money := 165
def machine_money := (quarters_per_machine * 0.25) + (dimes_per_machine * 0.10) + (nickels_per_machine * 0.05) + (pennies_per_machine * 0.01)
def min_machines := 3
def max_machines := 5
def machines := total_money / machine_money

-- Prove that the number of machines is 4, given the conditions
theorem number_of_machines_is_four (h : min_machines ≤ machines ∧ machines ≤ max_machines) : machines = 4 :=
by
  have h_eq : machines = total_money / machine_money := rfl
  have h_calculation : machines = 4 := sorry
  exact h_calculation

end number_of_machines_is_four_l335_335063


namespace trigonometric_shift_l335_335333

theorem trigonometric_shift :
  (∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x - π / 12))) :=
by
  intros x
  rw [sin_add, cos_sub]
  sorry

end trigonometric_shift_l335_335333


namespace MaddiesMomCupsOfCoffeePerDay_l335_335621

noncomputable def cups_per_day (daily_cups: ℝ) :=
  let ounces_per_cup := 1.5
  let cost_per_bag := 8
  let ounces_per_bag := 10.5
  let weekly_milk_cost := 4
  let weekly_coffee_spend := 18
  let daily_cups_of_coffee := (weekly_coffee_spend - weekly_milk_cost) / cost_per_bag * ounces_per_bag / ounces_per_cup / 7
  daily_cups_of_coffee
  
theorem MaddiesMomCupsOfCoffeePerDay == 1.75 :
  cups_per_day = 1.75 :=
by
  sorry

end MaddiesMomCupsOfCoffeePerDay_l335_335621


namespace elizabeth_bracelets_l335_335870

theorem elizabeth_bracelets :
  ∀ (n m : ℕ), n = 100 → m = 3 →
  (∃ days : ℕ, ∀ (b1 b2 : ℕ), b1 < b2 → b1 < n → b2 < n 
  → (∃ day : ℕ, day < days ∧ day ∈ {1, 2, ..., days} ∧
     b1 ∈ {br | br < n ∧ ∃ s ∈ {1, 2, ..., 3}, br = bracelet(day, s)} ∧
     b2 ∈ {br | br < n ∧ ∃ s ∈ {1, 2, ..., 3}, br = bracelet(day, s)} )) = False :=
begin
  intros n m hn hm,
  rw [hn, hm],
  sorry
end

end elizabeth_bracelets_l335_335870


namespace num_balls_in_box_l335_335984

theorem num_balls_in_box (n : ℕ) (h1: 9 <= n) (h2: (9 : ℝ) / n = 0.30) : n = 30 :=
sorry

end num_balls_in_box_l335_335984


namespace sum_1_to_50_l335_335310

-- Given conditions: initial values, and the loop increments
def initial_index : ℕ := 1
def initial_sum : ℕ := 0
def loop_condition (i : ℕ) : Prop := i ≤ 50

-- Increment step for index and running total in loop
def increment_index (i : ℕ) : ℕ := i + 1
def increment_sum (S : ℕ) (i : ℕ) : ℕ := S + i

-- Expected sum output for the given range
def sum_up_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the sum of integers from 1 to 50
theorem sum_1_to_50 : sum_up_to_n 50 = 1275 := by
  sorry

end sum_1_to_50_l335_335310


namespace correct_statement_b_l335_335808

/-- Conditions as definitions -/
def statement_A : Prop := ∀ (base_pairs : Type), mutation(base_pairs) -> gene_mutation(base_pairs)
def statement_B : Prop := gene_mutations_occur() ∧ chromosomal_variations_occur()
def statement_C : Prop := chromosomal_translocation() -> (change_in_genes ∧ no_trats_effect)
def statement_D : Prop := treat_with_colchicine(haploid_seedlings) -> resulting_diploid_individuals

/-- The goal is to prove that statement B is correct given the conditions -/
theorem correct_statement_b (A B C D : Prop) (hA : ¬ statement_A) (hB : statement_B) 
                           (hC : ¬ statement_C) (hD : ¬ statement_D) : B = statement_B :=
by
  sorry

end correct_statement_b_l335_335808


namespace correct_propositions_l335_335918

theorem correct_propositions :
  let p := ∃ x : ℝ, tan x = 1
  let q := ∀ x : ℝ, x^2 - x + 1 > 0
  (p ∧ ¬q) = false ∧
  let f (x : ℝ) := x^3 - 3 * x^2 + 1
  let tangent_line := 0 = -3 
  (x = 5 → ¬(x^2 - 4 * x - 5 = 0)) ∧
  (tangent_line) :=
by
  sorry

end correct_propositions_l335_335918


namespace relation_y1_y2_y3_l335_335270

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -x^2 + 2*x + c

variables (x1 x2 x3 : ℝ)
variables (y1 y2 y3 c : ℝ)
variables (P1 : x1 = -1)
variables (P2 : x2 = 3)
variables (P3 : x3 = 5)
variables (H1 : y1 = quadratic_function x1 c)
variables (H2 : y2 = quadratic_function x2 c)
variables (H3 : y3 = quadratic_function x3 c)

theorem relation_y1_y2_y3 (c : ℝ) :
  (y1 = y2) ∧ (y1 > y3) :=
sor_問題ry

end relation_y1_y2_y3_l335_335270


namespace vasya_most_points_anya_least_possible_l335_335821

theorem vasya_most_points_anya_least_possible :
  ∃ (A B V : ℕ) (A_score B_score V_score : ℕ),
  A > B ∧ B > V ∧
  A_score = 9 ∧ B_score = 10 ∧ V_score = 11 ∧
  (∃ (words_common_AB words_common_AV words_only_B words_only_V : ℕ),
  words_common_AB = 6 ∧ words_common_AV = 3 ∧ words_only_B = 2 ∧ words_only_V = 4 ∧
  A = words_common_AB + words_common_AV ∧
  B = words_only_B + words_common_AB ∧
  V = words_only_V + words_common_AV ∧
  A_score = words_common_AB + words_common_AV ∧
  B_score = 2 * words_only_B + words_common_AB ∧
  V_score = 2 * words_only_V + words_common_AV) :=
sorry

end vasya_most_points_anya_least_possible_l335_335821


namespace pair_sum_ways_9_10_11_12_13_l335_335564

open Finset

def num_pairs_ways : Nat := 945

theorem pair_sum_ways_9_10_11_12_13 :
  ∃ pairs : Finset (Finset ℕ), 
    (pairs.card = 5) ∧ 
    (∀ pair ∈ pairs, pair.card = 2 ∧ Finset.sum pair ∈ {9, 10, 11, 12, 13}) ∧ 
    num_elements = univ.card 10
    := sorry

end pair_sum_ways_9_10_11_12_13_l335_335564


namespace pancakes_eaten_by_older_is_12_l335_335169

/-- Pancake problem conditions -/
def initial_pancakes : ℕ := 19
def final_pancakes : ℕ := 11
def younger_eats_per_cycle : ℕ := 1
def older_eats_per_cycle : ℕ := 3
def grandma_bakes_per_cycle : ℕ := 2
def net_reduction_per_cycle := younger_eats_per_cycle + older_eats_per_cycle - grandma_bakes_per_cycle
def total_pancakes_eaten_by_older (cycles : ℕ) := older_eats_per_cycle * cycles

/-- Calculate the cycles based on net reduction -/
def cycles : ℕ := (initial_pancakes - final_pancakes) / net_reduction_per_cycle

/-- Prove the number of pancakes the older grandchild eats is 12 based on given conditions --/
theorem pancakes_eaten_by_older_is_12 : total_pancakes_eaten_by_older cycles = 12 := by
  sorry

end pancakes_eaten_by_older_is_12_l335_335169


namespace symmetric_polynomial_evaluation_l335_335497

theorem symmetric_polynomial_evaluation :
  ∃ (a b : ℝ), (∀ x : ℝ, (x^2 + 3 * x) * (x^2 + a * x + b) = ((2 - x)^2 + 3 * (2 - x)) * ((2 - x)^2 + a * (2 - x) + b)) ∧
  ((3^2 + 3 * 3) * (3^2 + (-6) * 3 + 8) = -18) :=
sorry

end symmetric_polynomial_evaluation_l335_335497


namespace possible_values_of_k_l335_335424

open Nat

noncomputable def prime_roots_equation (a b k : ℕ) : Prop :=
  prime a ∧ prime b ∧ a + b = 63 ∧ a * b = k

theorem possible_values_of_k :
  ∃! k : ℕ, ∃ a b : ℕ, prime_roots_equation a b k :=
sorry

end possible_values_of_k_l335_335424


namespace largest_prime_divisor_l335_335410

theorem largest_prime_divisor (a b c d : ℕ) (h₁ : a = 8) (h₂ : b = 16) (h₃ : c = 20) (h₄ : d = 28) :
  ∃ p : ℕ, nat.prime p ∧ (∀ x ∈ [a, b, c, d], p ∣ x) ∧ (∀ q : ℕ, nat.prime q ∧ (∀ x ∈ [a, b, c, d], q ∣ x) → q ≤ p) :=
begin
  use 2,
  split,
  { exact nat.prime_two, },
  split,
  { intros x hx,
    rw list.mem_cons_iff at hx,
    rcases hx with rfl | hx,
    { exact dvd_refl 8, },
    rw list.mem_cons_iff at hx,
    rcases hx with rfl | hx,
    { exact dvd_refl 16, },
    rw list.mem_cons_iff at hx,
    rcases hx with rfl | hx,
    { exact dvd_refl 20, },
    { rw list.mem_singleton at hx,
      exact dvd_refl 28, },
  },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_divides⟩,
    by_contradiction h,
    have hq_neq_2 : q ≠ 2 := ne_of_gt h,
    have : ¬ q ∣ 28 := not_dvd_of pos_div_succ 28 h (mem_singleton.mp (hq_divides 28 (by rw [a₄]))),
    exact this hq_divides,
  },
end

end largest_prime_divisor_l335_335410


namespace sum_abc_eq_neg75_over_4_l335_335300

theorem sum_abc_eq_neg75_over_4
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + c = 25)
  (h2 : (1, 0), (-3, 0) ∈ set.points_on_curve y = ax^2 + bx + c) :
  a + b + c = -75 / 4 := sorry

end sum_abc_eq_neg75_over_4_l335_335300


namespace jane_change_l335_335221

def total_cost_before_discount (num_apples : ℕ) (cost_per_apple : ℝ) : ℝ :=
num_apples * cost_per_apple

def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
total_cost * discount_rate

def final_amount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
total_cost - discount_amount

def round_to_nearest_cent (amount : ℝ) : ℝ :=
(Float.ofReal amount).round.toReal / 100

noncomputable def calculate_change (payment_amount : ℝ) (final_amount : ℝ) : ℝ :=
payment_amount - final_amount

theorem jane_change :
  let num_apples := 5
  let cost_per_apple := 0.75
  let discount_rate := 0.10
  let payment_amount := 10
  let total_cost := total_cost_before_discount num_apples cost_per_apple
  let discount := discount_amount total_cost discount_rate
  let final_cost := final_amount total_cost discount
  let rounded_final_cost := round_to_nearest_cent final_cost
  calculate_change payment_amount rounded_final_cost = 6.62 :=
by
  sorry

end jane_change_l335_335221


namespace card_probability_multiple_l335_335021

def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

def count_multiples (n k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def inclusion_exclusion (a b c : ℕ) (n : ℕ) : ℕ :=
  (count_multiples n a) + (count_multiples n b) + (count_multiples n c) - 
  (count_multiples n (Nat.lcm a b)) - (count_multiples n (Nat.lcm a c)) - 
  (count_multiples n (Nat.lcm b c)) + 
  count_multiples n (Nat.lcm a (Nat.lcm b c))

theorem card_probability_multiple (n : ℕ) 
  (a b c : ℕ) (hne : n ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (inclusion_exclusion a b c n) / n = 47 / 100 := by
  sorry

end card_probability_multiple_l335_335021


namespace problem_solution_l335_335167

variables (a b c : ℝ × ℝ)
def vector_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

def vector_perpendicular (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

def vector_projection (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := v₁.1 * v₂.1 + v₁.2 * v₂.2
  let norm_sq := v₂.1 * v₂.1 + v₂.2 * v₂.2
  (dot_prod / norm_sq * v₂.1, dot_prod / norm_sq * v₂.2)

theorem problem_solution
  (x y : ℝ)
  (a_def : a = (1, -2))
  (b_def : b = (x, y))
  (c_def : c = (x - 1, y + 2)) :

  (vector_parallel a b → y = -2 * x) ∧
  (vector_perpendicular a b ∧ x + y = 3 →
    vector_projection a c = (-1/2) • c) :=
by
  sorry

end problem_solution_l335_335167


namespace writer_born_on_Wednesday_l335_335847

theorem writer_born_on_Wednesday :
  ∀ (birth_year anniversary_year : ℕ)
    (anniversary_day_of_week : String)
    (is_leap : ℕ → Bool),
    birth_year = 1821 →
    anniversary_year = 2121 →
    anniversary_day_of_week = "Monday" →
    (∀ y, is_leap y = (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))) →
    (birth_day_of_week : String),
    birth_day_of_week = "Wednesday" :=
by
  intros birth_year anniversary_year anniversary_day_of_week is_leap h1 h2 h3 h4
  sorry

end writer_born_on_Wednesday_l335_335847


namespace tetrahedron_incenters_not_coplanar_l335_335040

-- Define the points and tetrahedron
variables {A B C D I_A I_B I_C I_D : ℝ^3}

-- Conditions
def is_incenter (I : ℝ^3) (A B C : ℝ^3) : Prop :=
  -- I is the incenter of triangle A B C (definition details omitted)
  sorry

def tetrahedron (A B C D : ℝ^3) : Prop :=
  -- A B C D form a tetrahedron (definition details omitted)
  sorry

-- Main theorem statement
theorem tetrahedron_incenters_not_coplanar
  (h_tetrahedron : tetrahedron A B C D)
  (h_I_A : is_incenter I_A B C D)
  (h_I_B : is_incenter I_B A C D)
  (h_I_C : is_incenter I_C A B D)
  (h_I_D : is_incenter I_D A B C) :
  ¬ coplanar ({I_A, I_B, I_C, I_D} : set ℝ^3) :=
sorry

end tetrahedron_incenters_not_coplanar_l335_335040


namespace compute_c_l335_335323

noncomputable def is_not_perfect_square (n : ℕ) : Prop := 
  ∀ d : ℕ, d * d ≠ n

theorem compute_c (a b c k : ℕ) (h0 : c ≠ 0)
  (h1 : is_not_perfect_square c)
  (h2 : a + Real.sqrt (b + Real.sqrt c) ∈ {x | x^4 - 20 * x^3 + 108 * x^2 - k * x + 9 = 0}) :
  c = 7 :=
sorry

end compute_c_l335_335323


namespace product_of_a_b_l335_335890

theorem product_of_a_b :
  ∀ (a b : ℝ), (U = set.univ) → 
  (A = {x : ℝ | a ≤ x ∧ x ≤ b}) → 
  (\compl A = {x : ℝ | x < 3 ∨ x > 4}) → a * b = 12 := 
by
  intros a b hU hA hCompl
  sorry

end product_of_a_b_l335_335890


namespace measles_cases_in_1990_l335_335567

theorem measles_cases_in_1990
  (cases_1970 : ℕ)
  (cases_2000 : ℕ)
  (linear_decrease : ∀ t : ℕ, t ≥ 1970 ∧ t ≤ 2000 → ℕ) :
  cases_1970 = 350000 →
  cases_2000 = 600 →
  (∀ t : ℕ, t ≥ 1970 ∧ t ≤ 2000 → linear_decrease t = cases_1970 - ((cases_1970 - cases_2000) * (t - 1970) / (2000 - 1970))) →
  linear_decrease 1990 = 117060 :=
by
  intros
  sorry

end measles_cases_in_1990_l335_335567


namespace sum_positive_real_solutions_l335_335083

theorem sum_positive_real_solutions :
  ∀ (x : ℝ), 
  0 < x →
  2 * cos(2 * x) * (cos(2 * x) - cos(1008 * π^2 / x)) = cos(4 * x) - 1 →
  ∑ (solutions : Finset ℝ) (H : solutions.sum id = 41 * π), True :=
by
  sorry

end sum_positive_real_solutions_l335_335083


namespace division_exponent_rule_l335_335730

theorem division_exponent_rule (a : ℝ) (h : a ≠ 0) : (a^8) / (a^2) = a^6 :=
sorry

end division_exponent_rule_l335_335730


namespace total_passengers_per_day_l335_335810

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l335_335810


namespace exp_gt_pow_l335_335271

theorem exp_gt_pow (x : ℝ) (h_pos : 0 < x) (h_ne : x ≠ Real.exp 1) : Real.exp x > x ^ Real.exp 1 := by
  sorry

end exp_gt_pow_l335_335271


namespace possible_integer_roots_l335_335289

theorem possible_integer_roots (b c d e : ℤ) :
  ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4) ∧
            ∃ (r1 r2 r3 r4 : ℤ), 
              multiset.of_list [r1, r2, r3, r4].count (λ r, eval r (polynomial.C 1 * polynomial.X^4 + polynomial.C b * polynomial.X^3 + polynomial.C c * polynomial.X^2 + polynomial.C d * polynomial.X + polynomial.C e) = 0) = n := 
sorry

end possible_integer_roots_l335_335289


namespace intersection_M_N_l335_335130

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335130


namespace chord_length_cut_by_line_l335_335079

theorem chord_length_cut_by_line {x y : ℝ} (h_line : y = 3 * x) (h_circle : (x + 1) ^ 2 + (y - 2) ^ 2 = 25) :
  ∃ x1 x2 y1 y2, 
    (y1 = 3 * x1) ∧ (y2 = 3 * x2) ∧ 
    ((x1 + 1) ^ 2 + (y1 - 2) ^ 2 = 25) ∧ ((x2 + 1) ^ 2 + (y2 - 2) ^ 2 = 25) ∧ 
    (real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 3 * real.sqrt 10) := sorry

end chord_length_cut_by_line_l335_335079


namespace upper_pyramid_volume_l335_335795

noncomputable def volume_of_upper_smaller_pyramid : ℝ :=
  let base_edge_length := 10 * Real.sqrt 2 in
  let slant_edge_length := 12 in
  let plane_height := 4 in
  let initial_height := Real.sqrt (slant_edge_length^2 - (base_edge_length / Real.sqrt 2)^2) in
  let small_pyramid_height := initial_height - plane_height in
  let similarity_ratio := small_pyramid_height / initial_height in
  (base_edge_length * similarity_ratio)^2 * small_pyramid_height * (1 / 3)

theorem upper_pyramid_volume :
  volume_of_upper_smaller_pyramid = (1000 / 22) * (2 * Real.sqrt 11 - 4)^3 :=
by
  sorry

end upper_pyramid_volume_l335_335795


namespace intersection_M_N_l335_335115

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l335_335115


namespace a_12_eq_78_l335_335606

noncomputable def a : ℕ → ℕ
| 0       := 0  -- assuming a_0 = 0 for well-formedness, since a_1 is given as 1
| 1       := 1
| (m + n) := a m + a n + m * n

theorem a_12_eq_78 : a 12 = 78 :=
sorry

end a_12_eq_78_l335_335606


namespace num_lines_through_point_with_reciprocal_intercepts_l335_335778

def point (x y : ℝ) : Prop := true

def lineThroughPointAndReciprocalIntercepts (P : ℝ × ℝ) : ℕ :=
if P = (1, 4) then 2 else 0

theorem num_lines_through_point_with_reciprocal_intercepts (P : ℝ × ℝ) (hP : P = (1, 4)) :
  lineThroughPointAndReciprocalIntercepts P = 2 := by
  sorry

end num_lines_through_point_with_reciprocal_intercepts_l335_335778


namespace geometric_series_sum_l335_335880

theorem geometric_series_sum (a : ℝ) (h : -1 < a ∧ a < 1) 
    (T : ℝ → ℝ) (T_def : ∀ r, T(r) = 20 / (1 - r))
    (h_eq : T(a)^2 * T(-a)^2 = 640000) :
    T(a) + T(-a) = 80 := 
sorry

end geometric_series_sum_l335_335880


namespace initial_people_count_25_l335_335250

-- Definition of the initial number of people (X) and the condition
def initial_people (X : ℕ) : Prop := X - 8 + 13 = 30

-- The theorem stating that the initial number of people is 25
theorem initial_people_count_25 : ∃ (X : ℕ), initial_people X ∧ X = 25 :=
by
  -- We add sorry here to skip the actual proof
  sorry

end initial_people_count_25_l335_335250


namespace arrangement_count_l335_335485

-- Definitions based on given conditions
def Products : Type := {A, B, C, D, E}

-- Statements of the conditions
def together_relation (x y : Products) : Prop := (x = A ∧ y = B) ∨ (x = B ∧ y = A)
def not_together_relation (x y : Products) : Prop := (x = C ∧ y = D) ∨ (x = D ∧ y = C)

-- Main theorem statement
theorem arrangement_count : 
  ∃ (arrangements : Finset (Finset Products)), 
  (∀ (x y : Products), x ∈ arrangements → y ∈ arrangements → together_relation x y → x = A ∧ y = B) ∧
  (∀ (x y : Products), x ∈ arrangements → y ∈ arrangements → not_together_relation x y → x ≠ C ∧ y ≠ D) ∧
  arrangements.card = 36 :=
sorry

end arrangement_count_l335_335485


namespace find_number_of_girls_l335_335089

-- Definitions
variables (B G : ℕ)
variables (total children_holding_boys_hand children_holding_girls_hand : ℕ)
variables (children_counted_twice : ℕ)

-- Conditions
axiom cond1 : B + G = 40
axiom cond2 : children_holding_boys_hand = 22
axiom cond3 : children_holding_girls_hand = 30
axiom cond4 : total = 40

-- Goal
theorem find_number_of_girls (h : children_counted_twice = children_holding_boys_hand + children_holding_girls_hand - total) :
  G = 24 :=
sorry

end find_number_of_girls_l335_335089


namespace bn_arithmetic_sequence_an_general_term_l335_335214

noncomputable def a_sequence : ℕ → ℕ
| 1       := 1
| (n+2)   := 3 * a_sequence (n+1) + 3 ^ (n+1)

def b_sequence (n : ℕ) := a_sequence n / 3 ^ (n - 1)

theorem bn_arithmetic_sequence : ∀ n : ℕ, b_sequence (n + 1) - b_sequence n = 1 :=
sorry

theorem an_general_term (n : ℕ) : a_sequence n = n * 3 ^ (n - 1) :=
sorry

end bn_arithmetic_sequence_an_general_term_l335_335214


namespace difference_between_scores_l335_335201

variable (H F : ℕ)
variable (h_hajar_score : H = 24)
variable (h_sum_scores : H + F = 69)
variable (h_farah_higher : F > H)

theorem difference_between_scores : F - H = 21 := by
  sorry

end difference_between_scores_l335_335201


namespace find_smallest_n_l335_335238
open Real

theorem find_smallest_n :
  ∃ n : ℕ, (∃ m s : ℝ, m = (n + s)^3 ∧ (m ∈ ℤ) ∧ (n > 0) ∧ (0 < s ∧ s < 1/500)) ∧ n = 13 :=
by
  sorry

end find_smallest_n_l335_335238


namespace cost_of_article_l335_335957

theorem cost_of_article (C : ℝ) (G : ℝ)
    (h1 : G = 520 - C)
    (h2 : 1.08 * G = 580 - C) :
    C = 230 :=
by
    sorry

end cost_of_article_l335_335957


namespace no_real_roots_of_equation_l335_335045

theorem no_real_roots_of_equation :
  ∀ x : ℝ, x + real.sqrt (2 * x - 3) ≠ 5 :=
by sorry

end no_real_roots_of_equation_l335_335045


namespace largest_inscribed_triangle_area_l335_335831

-- Define the conditions: radius of the circle
def radius (D : Type) [MetricSpace D] : ℝ := 10

-- Define the diameter of the circle based on the radius
def diameter (D : Type) [MetricSpace D] : ℝ := 2 * radius D

-- Define the height of the inscribed triangle as the same as the radius
def height (D : Type) [MetricSpace D] : ℝ := radius D

-- Define the area of the largest inscribed triangle
def area_triangle (D : Type) [MetricSpace D] : ℝ :=
  1 / 2 * diameter D * height D

theorem largest_inscribed_triangle_area (D : Type) [MetricSpace D] :
  area_triangle D = 100 := 
by 
  -- Since we only need the statement, we use sorry here
  sorry

end largest_inscribed_triangle_area_l335_335831


namespace area_of_triangle_APQ_l335_335287

theorem area_of_triangle_APQ (b1 b2 : ℝ) (h_sum : b1 + b2 = 0) : 
  let A := (6, 8)
      P := (0, b1)
      Q := (0, b2) in
  (abs b1 = abs b2) ∧
  dist (6, 8) (0, 0) = 10 →
  euclideanDistance (0, b1) (0, b2) = 20 →
  let area : ℝ := ½ * 20 * 6 in
  area = 60 :=
begin
  sorry
end

end area_of_triangle_APQ_l335_335287


namespace find_sum_interval_l335_335038

theorem find_sum_interval : 
  let a := 19 / 6;
  let b := 35 / 8;
  let c := 73 / 12;
  let sum := a + b + c in
  13.5 < sum ∧ sum < 14 :=
by
  let a := 19 / 6;
  let b := 35 / 8;
  let c := 73 / 12;
  let sum := a + b + c;
  sorry

end find_sum_interval_l335_335038


namespace minimum_colors_l335_335844

noncomputable def minimum_colors_needed (n : ℕ) (h : n ≥ 2) : ℕ :=
n

theorem minimum_colors (n : ℕ) (h : n ≥ 2)
  (unique_hat_colors : ∀ i j : ℕ, i < n → j < n → i ≠ j → unique (hat_color i))
  (unique_ribbon_colors : ∀ i j k : ℕ, i < n → j < n → k < n → i ≠ j → j ≠ k → k ≠ i → unique (ribbon_color i j)) :
  minimum_colors_needed n h = n := 
sorry

end minimum_colors_l335_335844


namespace base5_to_base4_last_digit_l335_335439

theorem base5_to_base4_last_digit (n : ℕ) (h : n = 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4) : (n % 4 = 2) :=
by sorry

end base5_to_base4_last_digit_l335_335439


namespace triangle_is_isosceles_l335_335999

variables {A B C : ℝ}

theorem triangle_is_isosceles
  (h_angle_sum : A + B + C = π)
  (h_cos_sin : 2 * cos B * sin A = sin C) :
  A = B ∨ B = C ∨ A = C :=
begin
  sorry
end

end triangle_is_isosceles_l335_335999


namespace max_value_of_xm_minus_xn_l335_335908

noncomputable def max_value_xm_minus_xn (m n : ℕ) (h_mn : m ≠ n) (h_m_pos : 0 < m) (h_n_pos : 0 < n) : ℝ :=
  |m - n| * (n^n / m^m)^(1 / (m - n))

theorem max_value_of_xm_minus_xn (m n : ℕ) (h_mn : m ≠ n) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (x : ℝ) (h_x : x ∈ Ioo 0 1) :
  ∃ (x_max : ℝ), x_max ∈ Ioo 0 1 ∧ (∀ x ∈ Ioo 0 1, abs (x^m - x^n) ≤ abs ((x_max)^m - (x_max)^n)) ∧ 
  abs ((x_max)^m - (x_max)^n) = |m - n| * (n^n / m^m)^(1 / (m - n)) := sorry

end max_value_of_xm_minus_xn_l335_335908


namespace parabola_focus_coordinates_l335_335290

theorem parabola_focus_coordinates :
  let p := (-1:Rat) / 4,
      focus_x := 0,
      focus_y := -p in
  focus_x = 0 ∧ focus_y = 1/4 :=
by
  sorry

end parabola_focus_coordinates_l335_335290


namespace intersection_M_N_l335_335124

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335124


namespace arithmetic_sequence_a9_l335_335982

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a 1)
  (h2 : a 2 + a 4 = 2)
  (h5 : a 5 = 3) :
  a 9 = 7 :=
by
  sorry

end arithmetic_sequence_a9_l335_335982


namespace who_is_first_l335_335017

-- Define the participants
inductive Participant
| A | B | C | D
deriving DecidableEq

-- Define their statements
def A_statement_part1 := ∀ place : ℕ, place = 1 → place = 3
def A_statement_part2 := 3
def B_statement_part1 := 1
def B_statement_part2 := 4
def C_statement_part1 := 2
def C_statement_part2 := 3
def D_statement :=  ∀ pA pB pC : ℕ, (pA ≠ pB ∧ pA = pB ∧ pB ≠ pC ∧ pB = pC ∧ pC ≠ D ∧ pC = D)

-- Define D's integrity
def D_honest := ∀ (s : Prop), s → (s ∨ ¬s)

-- Define the winning condition
def winner := B

-- Prove the statement
theorem who_is_first (hA : ∀ place : ℕ, place = 1 ∨ place = 3) (hB : ∀ place : ℕ, place = 1 ∨ place = 4)
(hC : ∀ place : ℕ, place = 2 ∨ place = 3) (hD : ∀ s: Prop, s → s)
  : ∀ place : ℕ, place = (λ place, B) := sorry

end who_is_first_l335_335017


namespace number_of_special_three_digit_numbers_l335_335547

theorem number_of_special_three_digit_numbers : ∃ (n : ℕ), n = 3 ∧
  (∀ (A B C : ℕ), 
    (100 * A + 10 * B + C < 1000 ∧ 100 * A + 10 * B + C ≥ 100) ∧
    B = 2 * C ∧
    B = (A + C) / 2 → 
    (A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 312 ∨ 
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 642 ∨
     A = 3 * C ∧ C ≤ 3 ∧ B = 2 * C ∧ 100 * A + 10 * B + C = 963))
:= 
sorry

end number_of_special_three_digit_numbers_l335_335547


namespace arrange_abc_l335_335500

noncomputable def a := Real.log (1/2)
noncomputable def b := (1/3) ^ 0.8
noncomputable def c := 2 ^ (1/3)

theorem arrange_abc : a < b ∧ b < c := by
  sorry

end arrange_abc_l335_335500


namespace integral_solution_l335_335753

noncomputable def integral_problem : ℝ :=
  ∫ x in -Real.arccos (1 / Real.sqrt 10), 0, (3 * Real.tan x ^ 2 - 50) / (2 * Real.tan x + 7)

theorem integral_solution :
  integral_problem = 9 * Real.ln (sorry) :=
sorry

end integral_solution_l335_335753


namespace positive_three_digit_integers_l335_335174

theorem positive_three_digit_integers
  (digits : Multiset ℕ)
  (h_digits : digits = {1, 1, 4, 4, 7, 8, 8}) :
  ∃ count : ℕ, count = 52 ∧
  (∀ d ∈ digits, d ∈ {1, 4, 7, 8}) ∧
  (∀ d, digits.count d ≤ 2) :=
sorry

end positive_three_digit_integers_l335_335174


namespace sector_angle_l335_335670

theorem sector_angle (r α : ℝ) (h₁ : 2 * r + α * r = 4) (h₂ : (1 / 2) * α * r^2 = 1) : α = 2 :=
sorry

end sector_angle_l335_335670


namespace problem_l335_335274

-- Define the set {1, 2, ..., 1989}
def my_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 1989}

-- Define what it means for there to be a partition of this set into 117 subsets with 17 elements each and the same sum
def has_partition (s : set ℕ) (n : ℕ) (k : ℕ) (m : ℕ) :=
  ∃ (A : fin n → set ℕ), 
    (∀ i, A i ⊆ s) ∧                      -- Each Ai is a subset of s
    (∀ i j, i ≠ j → disjoint (A i) (A j)) ∧ -- The subsets are disjoint
    (∀ i, (A i).card = k) ∧                 -- Each subset has k elements
    (∃ t, ∀ i, (A i).sum id = t)            -- Every subset sums to t

theorem problem : has_partition my_set 117 17 1 :=
sorry

end problem_l335_335274


namespace calculate_mixed_juice_cost_l335_335688

noncomputable def mixed_juice_cost_per_litre (mixed_juice_cost_per_litre açaí_juice_cost_per_litre superfruit_cocktail_cost_per_litre total_litres_mixed_juice total_litres_açaí_juice : ℝ) : ℝ :=
  let total_cost_mixed_juice := total_litres_mixed_juice * mixed_juice_cost_per_litre
  let total_cost_açaí_juice := total_litres_açaí_juice * açaí_juice_cost_per_litre
  let total_litres_superfruit_cocktail := total_litres_mixed_juice + total_litres_açaí_juice
  let total_cost_superfruit_cocktail := superfruit_cocktail_cost_per_litre * total_litres_superfruit_cocktail
  let total_cost_mixed_juice := total_cost_superfruit_cocktail - total_cost_açaí_juice
  total_cost_mixed_juice / total_litres_mixed_juice

theorem calculate_mixed_juice_cost :
  mixed_juice_cost_per_litre 265.62 3104.35 1399.45 36 24 = 265.62 :=
by
  sorry

end calculate_mixed_juice_cost_l335_335688


namespace probability_between_652_760_l335_335203

noncomputable def binomial_prob (n : ℕ) (p : ℝ) : ℝ :=
  let μ := n * p
  let σ := Math.sqrt (n * p * (1 - p))
  let α := (652 - μ) / σ
  let β := (760 - μ) / σ
  toReal ((Gaussian.cdf β - Gaussian.cdf α))

theorem probability_between_652_760 (h₁: 1000 = 1000) (h₂: 0.7 = 0.7) :
  binomial_prob 1000 0.7 ≈ 0.999 :=
sorry

end probability_between_652_760_l335_335203


namespace faster_train_crossing_time_l335_335340

theorem faster_train_crossing_time : 
  ∀ (speed_fast speed_slow : ℝ) (length_fast : ℝ),
  speed_fast = 162 →
  speed_slow = 18 →
  length_fast = 1320 →
  (length_fast / (speed_fast - speed_slow) * 1000 / 3600 = 33) :=
begin
  intros speed_fast speed_slow length_fast h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry, -- actual proof would go here
end

end faster_train_crossing_time_l335_335340


namespace RahulMatchesProved_AnkitMatchesProved_l335_335989

-- Define the conditions as Lean statements
variables (RahulMatches AnkitMatches : ℕ) (RahulRuns AnkitRuns : ℚ) (RahulNextRuns AnkitNextRuns : ℚ)

-- Rahul's current average condition
def RahulAvg : ℚ := 46
def AnkitAvg : ℚ := 52
def RahulNewAvg : ℚ := 54

-- Mathematical representation of conditions
def RahulCondition : Prop :=
  RahulRuns / RahulMatches = RahulAvg ∧
  (RahulRuns + RahulNextRuns) / (RahulMatches + 1) = RahulNewAvg

def AnkitCondition : Prop :=
  AnkitRuns / AnkitMatches = AnkitAvg ∧
  (AnkitRuns + AnkitNextRuns) / (AnkitMatches + 1) = RahulNewAvg ∧
  AnkitRuns + AnkitNextRuns = RahulRuns + RahulNextRuns

-- Problem statement: Prove the number of matches each has played
theorem RahulMatchesProved (hRahul : RahulCondition (46 * RahulMatches) 78) : RahulMatches = 3 :=
sorry

theorem AnkitMatchesProved (hAnkit : AnkitCondition (52 * AnkitMatches) (RahulRuns + RahulNextRuns - 52 * AnkitMatches)) :
  AnkitMatches = 3 :=
sorry

end RahulMatchesProved_AnkitMatchesProved_l335_335989


namespace toadon_population_percentage_l335_335329

theorem toadon_population_percentage {pop_total G L T : ℕ}
    (h_total : pop_total = 80000)
    (h_gordonia : G = pop_total / 2)
    (h_lakebright : L = 16000)
    (h_total_population : pop_total = G + T + L) :
    (T * 100 / G) = 60 :=
by sorry

end toadon_population_percentage_l335_335329


namespace value_of_k_l335_335185

theorem value_of_k (k : ℤ) : 
  (∃ a b : ℤ, x^2 + k * x + 81 = a^2 * x^2 + 2 * a * b * x + b^2) → (k = 18 ∨ k = -18) :=
by
  sorry

end value_of_k_l335_335185


namespace total_students_in_faculty_l335_335586

theorem total_students_in_faculty :
  (let sec_year_num := 230
   let sec_year_auto := 423
   let both_subj := 134
   let sec_year_total := 0.80
   let at_least_one_subj := sec_year_num + sec_year_auto - both_subj
   ∃ (T : ℝ), sec_year_total * T = at_least_one_subj ∧ T = 649) := by
  sorry

end total_students_in_faculty_l335_335586


namespace general_equation_of_C_polar_radius_of_M_l335_335453

def line1 (t k : ℝ) : ℝ × ℝ :=
  ⟨2 + t, k * t⟩

def line2 (m k : ℝ) : ℝ × ℝ :=
  ⟨-2 + m, m / k⟩

def curveC : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), (p.1 ^ 2 - p.2 ^ 2 = 4)

def line3 (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) - Real.sqrt 2 = 0

def intersection_pointM (x y : ℝ) : Prop :=
  x + y - Real.sqrt 2 = 0 ∧ x ^ 2 - y ^ 2 = 4

theorem general_equation_of_C (k : ℝ) (t m : ℝ) :
  curveC (line1 t k) ∧ curveC (line2 m k) :=
sorry

theorem polar_radius_of_M (ρ x y : ℝ) (θ : ℝ) :
  line3 ρ θ ∧ intersection_pointM x y → ρ ^ 2 = 5 :=
sorry

end general_equation_of_C_polar_radius_of_M_l335_335453


namespace union_M_N_eq_N_l335_335537

open Set

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}

theorem union_M_N_eq_N : M ∪ N = N := sorry

end union_M_N_eq_N_l335_335537


namespace b107_mod_64_l335_335245

def b (n : ℕ) : ℕ := 5^n + 9^n

theorem b107_mod_64 : b 107 % 64 = 8 := 
by
  have h_mod : 107 % 32 = 11 := sorry
  have h5 : 5^11 % 64 = 5 := sorry
  have h9 : 9^11 % 64 = 3 := sorry
  have h_b107 : b 107 % 64 = (5 + 3) % 64 := by
    simp [b, h5, h9]
  show (5 + 3) % 64 = 8,
  by norm_num

end b107_mod_64_l335_335245


namespace calc_cos_sin_sum_l335_335698

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x
def slope_at_1 : ℝ := 1 + 2 / 1  -- derivative f' at x=1 yields 3

theorem calc_cos_sin_sum :
  let α := Real.arctan slope_at_1
  ∃ (α : ℝ), Real.tan α = slope_at_1
           ∧ Real.cos α + Real.sin α = 2 * Real.sqrt 10 / 5 :=
by
  let α := Real.arctan slope_at_1
  use α
  split
  · exact Real.tan_arctan slope_at_1
  sorry

end calc_cos_sin_sum_l335_335698


namespace arccos_sqrt3_over_2_eq_pi_over_6_l335_335432

theorem arccos_sqrt3_over_2_eq_pi_over_6 :
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by
  -- condition from the problem
  have h : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 :=
    by sorry -- this is a known identity we assume true
  -- now we infer the desired result using arccos
  rw Real.arccos_cos h
  sorry

end arccos_sqrt3_over_2_eq_pi_over_6_l335_335432


namespace monotonic_increasing_interval_l335_335093

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  sin x * sin x + sqrt 3 * (sin x * cos x)

-- Define the interval of interest
def interval := Set.Icc 0 (π / 2)

-- Lean statement to prove the monotonic increase in the interval [0, π/3]
theorem monotonic_increasing_interval :
  ∀ x ∈ interval, x ≤ π / 3 → (f ' x > 0) :=
by
  sorry

end monotonic_increasing_interval_l335_335093


namespace polynomial_coeff_sum_l335_335493

theorem polynomial_coeff_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) (h : (2 * x - 3) ^ 5 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 160 :=
sorry

end polynomial_coeff_sum_l335_335493


namespace unique_two_scoop_sundaes_l335_335029

open Nat

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : (nat.choose n 2) = 28 :=
by 
  rw h 
  simp 
  sorry

end unique_two_scoop_sundaes_l335_335029


namespace a4_is_27_l335_335934

-- Given the sum of the first n terms of the sequence
def Sn (n : ℕ) : ℚ := (1 / 2) * (3^n - 1)

-- Define the sequence {a_n}
def a (n : ℕ) : ℚ := Sn n - Sn (n-1)

theorem a4_is_27 : a 4 = 27 := 
by {
  unfold a Sn,
  have hS4 : Sn 4 = 41 / 2 := by norm_num,
  have hS3 : Sn 3 = 13 / 2 := by norm_num,
  rw [hS4, hS3],
  norm_num,
  sorry 
}

end a4_is_27_l335_335934


namespace arithmetic_sequence_difference_l335_335462

theorem arithmetic_sequence_difference :
  (∑ k in Finset.range 93, (1981 + k) - ∑ k in Finset.range 93, (201 + k)) = 165540 :=
by
  sorry

end arithmetic_sequence_difference_l335_335462


namespace n_squared_divisible_by_144_l335_335187

-- Definitions based on the conditions
variables (n k : ℕ)
def is_positive (n : ℕ) : Prop := n > 0
def largest_divisor_of_n_is_twelve (n : ℕ) : Prop := ∃ k, n = 12 * k
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

theorem n_squared_divisible_by_144
  (h1 : is_positive n)
  (h2 : largest_divisor_of_n_is_twelve n) :
  divisible_by (n * n) 144 :=
sorry

end n_squared_divisible_by_144_l335_335187


namespace sum_of_interior_angles_eq_1440_l335_335397

theorem sum_of_interior_angles_eq_1440 (h : ∀ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ)) : 
    (∃ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ) ∧ (n - 2) * 180 = 1440) :=
by
  sorry

end sum_of_interior_angles_eq_1440_l335_335397


namespace geometric_sequence_term_eq_l335_335575

theorem geometric_sequence_term_eq (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 / 2 → q = 1 / 2 → a₁ * q ^ (n - 1) = 1 / 32 → n = 5 :=
by
  intros ha₁ hq han
  sorry

end geometric_sequence_term_eq_l335_335575


namespace correct_conclusion_l335_335935

def vector (α : Type) := list α
def dot_product (v1 v2 : vector ℤ) : ℤ := list.sum (list.zip_with (*) v1 v2)

noncomputable def parallel (v1 v2 : vector ℤ) : Prop :=
  ∃ k : ℤ, v1 = v2.map (λ x, k * x)

noncomputable def perpendicular (v1 v2 : vector ℤ) : Prop :=
  dot_product v1 v2 = 0

def a := [-2, -3, 1]
def b := [2, 0, 4]
def c := [4, 6, -2]

theorem correct_conclusion : parallel a c ∧ perpendicular a b :=
by
  sorry

end correct_conclusion_l335_335935


namespace count_president_vp_secretary_l335_335637

theorem count_president_vp_secretary (total_members boys girls : ℕ) (total_members_eq : total_members = 30) 
(boys_eq : boys = 18) (girls_eq : girls = 12) :
  ∃ (ways : ℕ), 
  ways = (boys * girls * (boys - 1) + girls * boys * (girls - 1)) ∧
  ways = 6048 :=
by
  sorry

end count_president_vp_secretary_l335_335637


namespace intersection_M_N_l335_335137

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l335_335137


namespace harry_pencils_lost_l335_335820

-- Define the conditions
def anna_pencils : ℕ := 50
def harry_initial_pencils : ℕ := 2 * anna_pencils
def harry_current_pencils : ℕ := 81

-- Define the proof statement
theorem harry_pencils_lost :
  harry_initial_pencils - harry_current_pencils = 19 :=
by
  -- The proof is to be filled in
  sorry

end harry_pencils_lost_l335_335820


namespace average_birds_seen_l335_335257

theorem average_birds_seen (marcus birds: ℕ) (humphrey birds: ℕ) (darrel birds: ℕ) (isabella birds: ℕ) :
  marcus = 7 ∧ humphrey = 11 ∧ darrel = 9 ∧ isabella = 15 →
  (marcus + humphrey + darrel + isabella) / 4 = 10.5 :=
by
  intros h
  rcases h with ⟨h_marcus, ⟨h_humphrey, ⟨h_darrel, h_isabella⟩⟩⟩
  simp [h_marcus, h_humphrey, h_darrel, h_isabella]
  norm_num
  sorry

end average_birds_seen_l335_335257


namespace chocolates_per_student_class_7B_l335_335061

theorem chocolates_per_student_class_7B :
  (∃ (x : ℕ), 9 * x < 288 ∧ 10 * x > 300 ∧ x = 31) :=
by
  use 31
  -- proof steps omitted here
  sorry

end chocolates_per_student_class_7B_l335_335061


namespace probability_S7_eq_3_l335_335578

noncomputable def prob_S7_eq_3 : ℝ :=
  let α := 2/3 in
  let β := 1/3 in
  let num_ways := Nat.choose 7 5 in
  let red_prob := α^2 in
  let white_prob := β^5 in
  num_ways * red_prob * white_prob

theorem probability_S7_eq_3 : prob_S7_eq_3 = (Nat.choose 7 5 * (2/3)^2 * (1/3)^5) :=
by 
  sorry

end probability_S7_eq_3_l335_335578


namespace correct_statements_about_f_l335_335925

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + π / 6) + sin (ω * x - π / 6) + cos (ω * x)

theorem correct_statements_about_f (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x, f ω (x + π) = f ω x) :
  (ω = 2) ∧ (∀ φ : ℝ, f ω (x + φ) = f ω x → φ = π / 6) :=
by
  sorry

end correct_statements_about_f_l335_335925


namespace z_real_z_imaginary_l335_335887

-- Definitions and conditions from part a)
def z (m : ℝ) : ℂ := (m^2 - m - 6) / (m + 3) + (m^2 + 5m + 6) * complex.I

-- Proof problem for real number condition
theorem z_real (m : ℝ) (h : m ≠ -3) : z m = (m^2 - m - 6) / (m + 3) ↔ m = -2 :=
by {
  sorry
}

-- Proof problem for pure imaginary number condition
theorem z_imaginary (m : ℝ) (h₀ : m ≠ -3) (h₁ : ¬ (m^2 + 5m + 6 = 0)) : z m = (m^2 + 5m + 6) * complex.I ↔ m = 3 :=
by {
  sorry
}

end z_real_z_imaginary_l335_335887


namespace solve_quadratic_solve_inequalities_l335_335370
open Classical

-- Define the equation for Part 1
theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → (x = 1 ∨ x = 5) :=
by
  sorry

-- Define the inequalities for Part 2
theorem solve_inequalities (x : ℝ) : (x + 3 > 0) ∧ (2 * (x - 1) < 4) → (-3 < x ∧ x < 3) :=
by
  sorry

end solve_quadratic_solve_inequalities_l335_335370


namespace least_area_of_triangle_in_octagon_l335_335316

open Complex Real

-- Definition to represent the given conditions
def is_vertex_of_octagon (z : ℂ) (n : ℕ) : Prop :=
  z = (complex.ofReal (sqrt 3)) * exp ((2 * ↑π * I * n) / 8) - 6

-- The least possible area of triangle ABC formed by consecutive vertices
theorem least_area_of_triangle_in_octagon (A B C : ℂ)
  (hA : ∃ n : ℕ, is_vertex_of_octagon A n)
  (hB : ∃ n : ℕ, is_vertex_of_octagon B n)
  (hC : ∃ n : ℕ, is_vertex_of_octagon C n)
  (h_consecutive : ∃ n : ℕ, B = (complex.ofReal (sqrt 3)) * exp ((2 * ↑π * I * (n+1)) / 8) - 6 ∧
                              C = (complex.ofReal (sqrt 3)) * exp ((2 * ↑π * I * (n+2)) / 8) - 6) :
  (let area := (1/2) * abs (imag_part (B * conj C + C * conj A + A * conj B))
   in area = ((3/2) * sqrt 2 - 3/2)) := sorry

end least_area_of_triangle_in_octagon_l335_335316


namespace largest_integer_not_exceeding_A_l335_335610

-- Define the quadratic equation and its roots
theorem largest_integer_not_exceeding_A :
  let α := (1 + Real.sqrt 8085) / 2 in
  let β := (1 - Real.sqrt 8085) / 2 in
  α > β →
  let A := α^2 - 2 * β^2 + 2 * α * β + 3 * β + 7 in
  ⌊A⌋ = -6055 :=
by
  sorry

end largest_integer_not_exceeding_A_l335_335610


namespace find_positive_m_has_exactly_single_solution_l335_335481

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l335_335481


namespace rook_arrangement_count_l335_335473

theorem rook_arrangement_count : 
  ∃ (count : ℕ), count = 576 ∧ 
  (∀ (rooks : Fin 8 → Fin 8 × Fin 8),
    (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) ∧
    (∀ k : Fin 8, (rooks k).1 % 2 = (rooks k).2 % 2 → False)) := 
begin
  use 576,
  sorry,
end

end rook_arrangement_count_l335_335473


namespace CD_eq_CB_l335_335635

open EuclideanGeometry

theorem CD_eq_CB
  (K : Circle)
  (O : Point K)
  (A B : Point K)
  (C : Point K)
  (hA : OnChord O A B)
  (hB : OnChord O A B)
  (hC : OnChord O A B)
  (hD : ∃ D, SecondIntersectionOfCircumcircleOfACO K O A C D) :
  dist C D = dist C B :=
by
  sorry

end CD_eq_CB_l335_335635


namespace probability_top_card_diamond_l335_335798

theorem probability_top_card_diamond
  (cards : Finset (ℕ × ℕ))
  (h1 : cards.card = 60)
  (ranks : Finset ℕ)
  (h2 : ranks.card = 15)
  (suits : Finset ℕ)
  (h3 : suits.card = 4)
  (h4 : ∀ (r : ℕ), r ∈ ranks →
        (λ s : ℕ, (r, s)) '' suits ⊆ cards)
  (random_deck : list (ℕ × ℕ)) :
  (random_deck.head ∈ (λ r : ℕ, (r, 1)) '' ranks →
    (random_deck.head ∈ (λ r : ℕ, (r, 2)) '' ranks →
      (random_deck.head ∈ (λ r : ℕ, (r, 3)) '' ranks →
        (random_deck.head ∈ (λ r : ℕ, (r, 4)) '' ranks →
          ∃ probability : ℚ,
            probability = 1 / 4))) :=
begin
  sorry
end

end probability_top_card_diamond_l335_335798


namespace population_increase_l335_335091

/-- Conditions -/
def scale_factor_t0_t1 := 1 + 5 / 100
def scale_factor_t1_t2 := 1 + 10 / 100
def scale_factor_t2_t3 := 1 + 15 / 100

def overall_scale_factor := scale_factor_t0_t1 * scale_factor_t1_t2 * scale_factor_t2_t3

theorem population_increase :
  overall_scale_factor - 1 = 0.33075 := 
by
  sorry

end population_increase_l335_335091


namespace fraction_undefined_l335_335963

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (x = -1) := 
  sorry

end fraction_undefined_l335_335963


namespace no_positive_integers_satisfy_condition_l335_335464

theorem no_positive_integers_satisfy_condition :
  ∀ (n : ℕ), n > 0 → ¬∃ (a b m : ℕ), a > 0 ∧ b > 0 ∧ m > 0 ∧ 
  (a + b * Real.sqrt n) ^ 2023 = Real.sqrt m + Real.sqrt (m + 2022) := by
  sorry

end no_positive_integers_satisfy_condition_l335_335464


namespace successive_numbers_product_2652_l335_335694

theorem successive_numbers_product_2652 (n : ℕ) (h : n * (n + 1) = 2652) : n = 51 :=
sorry

end successive_numbers_product_2652_l335_335694


namespace find_positive_m_has_exactly_single_solution_l335_335482

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l335_335482


namespace mikails_age_l335_335259

-- Define the conditions
def dollars_per_year_old : ℕ := 5
def total_dollars_given : ℕ := 45

-- Main theorem statement
theorem mikails_age (age : ℕ) : (age * dollars_per_year_old = total_dollars_given) → age = 9 :=
by
  sorry

end mikails_age_l335_335259


namespace people_got_off_third_stop_l335_335403

theorem people_got_off_third_stop : 
  ∀ (p_1 p_2 p_3 p_4 n p_5 : ℕ), 
  p_1 = 10 →
  p_2 = 3 →
  p_3 = 2 * p_1 →
  p_4 = 2 →
  n = 12 →
  let remaining_after_second_stop := (p_1 - p_2) + p_3 in
  remaining_after_second_stop - p_5 + p_4 = n →
  p_5 = 17 :=
by
  intros p_1 p_2 p_3 p_4 n p_5 hp1 hp2 hp3 hp4 hn hfinal_eq
  sorry

end people_got_off_third_stop_l335_335403


namespace roots_sum_l335_335231

theorem roots_sum (p q r : ℝ) (w : ℂ) 
  (h₁ : ∃ (w : ℂ), 
       (z : ℂ) -> z^3 + p*z^2 + q*z + r = 0 
      → (z - (w + 5*complex.I)) * 
        (z - (w + 15*complex.I)) * 
        (z - (3*w - 6)) = 0) :
  p + q + r = -1 :=
sorry

end roots_sum_l335_335231


namespace problem1_problem2_l335_335507

-- 1. Definitions for arc length and area
def arc_length (α : ℝ) (R : ℝ) := α * R
def sector_area (α : ℝ) (R : ℝ) := (1/2) * α * R^2
def triangle_area (R : ℝ) (α : ℝ) := (1/2) * R^2 * Real.sin α
def segment_area (α : ℝ) (R : ℝ) := sector_area α R - triangle_area R α

-- 2. Problem 1: Prove the arc length and the segment area for given R = 10 and α = 60°
theorem problem1 (R : ℝ) (α : ℝ) (hα : α = π / 3) (hR : R = 10) :
  arc_length α R = (10 / 3) * π ∧ segment_area α R = 50 * (π / 3 - Real.sqrt 3 / 2) :=
by sorry

-- 3. Problem 2: Prove the α that maximizes the area of the sector with a given perimeter C
def perimeter (α : ℝ) (R : ℝ) := 2 * R + arc_length α R

theorem problem2 (C : ℝ) (hC : C > 0) :
  Exists (λ α : ℝ, α = 2 ∧ 
  ∀ (R : ℝ) (H : perimeter α R = C), sector_area α R = (C^2) / 16) :=
by sorry

end problem1_problem2_l335_335507


namespace modulus_of_z_l335_335907

theorem modulus_of_z (a : ℝ) (ha : (∀ b : ℝ, 2 * b = 1 → a = b) ∧ a ≠ -2) :
  complex.abs ((2 * a + 1 : ℂ) + complex.I * (real.sqrt 2)) = real.sqrt 6 := by sorry

end modulus_of_z_l335_335907


namespace least_possible_k_l335_335243

-- Definitions and conditions
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 100000 }

def has_size_A : Set ℕ → Prop := λ A, A ⊆ S ∧ A.card = 2010

-- The conjecture we want to prove
theorem least_possible_k :
  ∀ A : Set ℕ, has_size_A A → ∃ a b ∈ A, a ≠ b ∧ |a - b| ≤ 49 :=
by
  intro A hA
  sorry

end least_possible_k_l335_335243


namespace unit_digit_sum_l335_335746

theorem unit_digit_sum (n : ℕ) : 
  ((∑ k in finset.range (n+1), (2013 ^ k) % 10) % 10) = 3 :=
by
  sorry

end unit_digit_sum_l335_335746


namespace domain_of_f_l335_335291

theorem domain_of_f (x : ℝ) : (∃ y, f x = y) ↔ x ∈ set.Ici 2 := by
  let f := λ x, sqrt (x - 2)
  sorry

end domain_of_f_l335_335291


namespace books_per_shelf_l335_335015

def initial_coloring_books : ℕ := 86
def sold_coloring_books : ℕ := 37
def shelves : ℕ := 7

theorem books_per_shelf : (initial_coloring_books - sold_coloring_books) / shelves = 7 := by
  sorry

end books_per_shelf_l335_335015


namespace solve_problem1_solve_problem2_l335_335939

noncomputable def problem1 (a b c : ℝ) (B C : ℝ) : Prop :=
(b = 2 * a - 2 * c * cos B ∨ sqrt 3 * a = b * sin C + sqrt 3 * c * cos B) → C = π / 3

noncomputable def problem2 (a b : ℝ) (c : ℝ := 2) (D midpoint_AB : Prop) : Prop :=
(D → midpoint_AB) → (sqrt (a * a + b * b + (a * b)) ≤ sqrt 3) → true

theorem solve_problem1 (a b c B C : ℝ) : problem1 a b c B C :=
sorry

theorem solve_problem2 (a b : ℝ) (c : ℝ := 2) (D midpoint_AB: Prop) : problem2 a b :=
sorry

end solve_problem1_solve_problem2_l335_335939


namespace max_area_of_triangle_l335_335468

theorem max_area_of_triangle (AB AC BC : ℝ) : 
  AB = 4 → AC = 2 * BC → 
  ∃ (S : ℝ), (∀ (S' : ℝ), S' ≤ S) ∧ S = 16 / 3 :=
by
  sorry

end max_area_of_triangle_l335_335468


namespace liquid_X_percentage_l335_335620

theorem liquid_X_percentage (
  mass_A : ℝ, mass_B : ℝ,
  percentage_A : ℝ, percentage_B : ℝ,
  temp_A : ℝ, temp_B : ℝ,
  temp_factor : ℝ, temp_diff : ℝ
) : 
  let adjusted_percentage_A := percentage_A - (temp_diff / temp_factor) * 0.002,
      adjusted_percentage_B := percentage_B + (temp_diff / temp_factor) * 0.002,
      final_mass_A := mass_A * max (adjusted_percentage_A / 100) 0,
      final_mass_B := mass_B * (adjusted_percentage_B / 100),
      total_mass := mass_A + mass_B,
      total_liquid_X_mass := final_mass_A + final_mass_B,
      percentage_liquid_X_in_mixture := (total_liquid_X_mass / total_mass) * 100
  in percentage_liquid_X_in_mixture = 1.26
:=
by 
  have h1 : adjusted_percentage_A = 0.8 - (20 / 5) * 0.2 := by sorry,
  have h2 : adjusted_percentage_A = 0 := by sorry,  -- equivalently, Liquid X cannot be less than 0%
  have h3 : adjusted_percentage_B = 1.8 + (20 / 5) * 0.2 := by sorry,
  have h4 : adjusted_percentage_B = 2.6 := by sorry,
  have h5 : final_mass_A = 300 * 0 := by sorry,
  have h6 : final_mass_B = 700 * (2.6 / 100) := by sorry,
  have h7 : final_mass_B = 18.2 := by sorry,
  have percentage_liquid_X_in_mixture := (18.2 / 1000) * 100 := by sorry,
  show percentage_liquid_X_in_mixture = 1.26, from sorry

end liquid_X_percentage_l335_335620


namespace value_of_b_150_l335_335048

noncomputable def b : ℕ → ℕ
| 1       => 3
| (n + 1) => b n + 4 * n

theorem value_of_b_150 : b 150 = 44703 :=
sorry

end value_of_b_150_l335_335048


namespace trapezium_area_correct_l335_335362

-- Define the parameters given in the problem
def a : ℝ := 24
def b : ℝ := 18
def h : ℝ := 15

-- Define the area calculation based on the given conditions
def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proven
theorem trapezium_area_correct : trapezium_area a b h = 315 := by
  sorry

end trapezium_area_correct_l335_335362


namespace instantaneous_velocity_at_t3_l335_335819

-- Define the motion equation as a function
def motion_eq (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity as the derivative of the motion equation
def velocity (t : ℝ) : ℝ := deriv motion_eq t

-- Statement of the mathematically equivalent proof problem
theorem instantaneous_velocity_at_t3 : velocity 3 = 54 := by
  sorry

end instantaneous_velocity_at_t3_l335_335819


namespace each_girl_receives_six_sheets_l335_335708

variables (G B : ℕ)

noncomputable def number_of_sheets_per_girl (total_students boys_multiplier total_sheets leftover_sheets : ℕ) : ℕ :=
  let G := (total_students - boys_multiplier * total_students / (1 + boys_multiplier)) in
  let distributed_sheets := total_sheets - leftover_sheets in
  distributed_sheets / G

theorem each_girl_receives_six_sheets :
  G + B = 24 ∧ B = 2 * G ∧ 50 - 2 = 48 →
  number_of_sheets_per_girl 24 2 50 2 = 6 :=
by
  intro h
  sorry

end each_girl_receives_six_sheets_l335_335708


namespace max_value_of_a_l335_335896

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ :=
  n^2 + 2 * a * |(n : ℝ) - 2016|

def an (n : ℕ) (a : ℝ) : ℝ :=
  if n = 1 then Sn n a 
  else Sn n a - Sn (n - 1) a

theorem max_value_of_a (a : ℝ) (h1 : 0 < a) : a ≤ (1 / 2016) :=
sorry

end max_value_of_a_l335_335896


namespace arithmetic_sequence_sum_product_l335_335325

-- Define the variables a and d
variables (a d : ℤ)

-- Define the three numbers in terms of a and d
def num1 := a - d
def num2 := a
def num3 := a + d

-- Define the conditions
def cond_sum : Prop := num1 + num2 + num3 = 6
def cond_product : Prop := num1 * num2 * num3 = -10

-- Define the sets of solutions
def solution1 : set ℤ := {5, 2, -1}
def solution2 : set ℤ := {-1, 2, 5}

-- Define the main theorem
theorem arithmetic_sequence_sum_product :
  cond_sum a d ∧ cond_product a d → 
  {num1, num2, num3} = solution1 ∨ {num1, num2, num3} = solution2 :=
by sorry

end arithmetic_sequence_sum_product_l335_335325


namespace set_intersection_l335_335125

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l335_335125


namespace relation_y1_y2_y3_l335_335268

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -x^2 + 2*x + c

variables (x1 x2 x3 : ℝ)
variables (y1 y2 y3 c : ℝ)
variables (P1 : x1 = -1)
variables (P2 : x2 = 3)
variables (P3 : x3 = 5)
variables (H1 : y1 = quadratic_function x1 c)
variables (H2 : y2 = quadratic_function x2 c)
variables (H3 : y3 = quadratic_function x3 c)

theorem relation_y1_y2_y3 (c : ℝ) :
  (y1 = y2) ∧ (y1 > y3) :=
sor_問題ry

end relation_y1_y2_y3_l335_335268


namespace count_rectangles_4x4_grid_l335_335947

theorem count_rectangles_4x4_grid :
  let h_lines := 4
  let v_lines := 4
  binom h_lines 2 * binom v_lines 2 = 36 :=
by
  sorry

end count_rectangles_4x4_grid_l335_335947


namespace abs_distance_sum_l335_335106

noncomputable def center_trajectory : (ℝ × ℝ) → Prop :=
  λ p, p.snd ^ 2 = 4 * p.fst

noncomputable def minimized_circle : (ℝ × ℝ) → Prop :=
  λ p, p.fst ^ 2 + p.snd ^ 2 = 1

noncomputable def intersection_points (b : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.snd = 1 / 2 * p.fst + b) ∧ (center_trajectory p ∨ minimized_circle p)}

theorem abs_distance_sum {b : ℝ} :
  let F : ℝ × ℝ := (1, 0),
      parabola := {p | center_trajectory p},
      circle := {p | minimized_circle p},
      angle_complement := λ k1 k2 : ℝ, k1 + k2 = 0,
      distances := λ A B C D : ℝ × ℝ, (abs (A.fst - B.fst) + abs (C.fst - D.fst))
  in 
  ∀ A B C D : ℝ × ℝ, 
    A ∈ intersection_points b →
    B ∈ intersection_points b →
    C ∈ intersection_points b →
    D ∈ intersection_points b →
    angle_complement (A.snd / (A.fst - F.fst)) (C.snd / (C.fst - F.fst)) →
    distances A B C D = 36 * real.sqrt 5 / 5 :=
sorry

end abs_distance_sum_l335_335106


namespace math_problem_l335_335235

open Set

variables {a b c : Set.Point → Set.Point → Prop} 
variables {alpha beta : Set.Point → Set.Point → Set.Point → Prop}

-- Corresponds to proposition 1
def proposition1 : Prop :=
  ∀ {b alpha a c : Set.Point → Set.Point → Prop},
    (b ⊆ alpha ∧ ∀ (A : Set.Point), ∃ (B : Set.Point), c A B ∧ (a A B) ∧ ∀ (C : Set.Point), (alpha A C) → (c B C)) 
      → ((∀ (D E : Set.Point), b D E → ∀ (F G : Set.Point), c F G → Orthogonal D E G F (α)) 
        → ∀ (H I : Set.Point), a H I → ∀ (J K : Set.Point), b J K → Orthogonal H I J K (β))
        
-- Corresponds to proposition 2
def proposition2 : Prop :=
  ∀ {b alpha c : Set.Point → Set.Point → Prop},
    (b ⊆ alpha ∧ ∀ (L M : Set.Point), ¬(alpha L M) ∧ Parallel c alpha)
      → (Parallel b c)
      
-- Corresponds to proposition 3
def proposition3 : Prop :=
  ∀ {b alpha beta : Set.Point → Set.Point → Prop},
    b ⊆ alpha → Orthogonal b beta → Orthogonal alpha beta
    
-- Corresponds to proposition 4
def proposition4 : Prop :=
  ∀ {c alpha beta : Set.Point → Set.Point → Prop},
    (Orthogonal c alpha ∧ Orthogonal c beta)
      → Parallel alpha beta

theorem math_problem : proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end math_problem_l335_335235


namespace matrix_product_correct_l335_335433

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![2, 3, -1], ![0, 5, -4], ![-2, 5, 2]]
def B : Matrix (Fin 3) (Fin 3) ℤ := ![![3, -3, 0], ![2, 1, -4], ![5, 0, 1]]
def C : Matrix (Fin 3) (Fin 2) ℤ := ![![1, -1], ![0, 2], ![1, 0]]
def ABC := ![![ -6, -13 ], ![ -34, 20 ], ![ -4, 8 ]]

theorem matrix_product_correct : A ⬝ B ⬝ C = ABC := by sorry

end matrix_product_correct_l335_335433


namespace total_surface_area_of_pyramid_l335_335666

theorem total_surface_area_of_pyramid (h : ℝ) (angle_dihedral_base : ℝ) (H_angle : angle_dihedral_base = 60) :
  let total_surface_area := (3 * h^2 * Real.sqrt 3) / 2 in
  total_surface_area = (3 * h^2 * Real.sqrt 3) / 2 :=
by
  sorry

end total_surface_area_of_pyramid_l335_335666


namespace variance_of_y_eq_4_l335_335157

theorem variance_of_y_eq_4 (x : Fin 2017 → ℝ)
  (hxvar : (∑ i, (x i - (∑ i, x i) / 2017) ^ 2) / 2017 = 4) :
  let y (i : Fin 2017) := x i - 1 in
  (∑ i, (y i - (∑ i, y i) / 2017) ^ 2) / 2017 = 4 := by
  sorry

end variance_of_y_eq_4_l335_335157


namespace no_integer_coordinates_for_equilateral_triangle_l335_335416

-- Definitions of an equilateral triangle
def equilateral_triangle (a b c d e f : ℤ) : Prop :=
  let s1 := (a - c)^2 + (b - d)^2 in
  let s2 := (a - e)^2 + (b - f)^2 in
  let s3 := (c - e)^2 + (d - f)^2 in
  s1 = s2 ∧ s2 = s3

theorem no_integer_coordinates_for_equilateral_triangle :
  ∀ (a b c d e f : ℤ), ¬equilateral_triangle a b c d e f :=
by
  sorry

end no_integer_coordinates_for_equilateral_triangle_l335_335416


namespace power_add_zero_l335_335826

theorem power_add_zero : (-2)^(3^2) + (2)^(3^2) = 0 := by
  have n : ℤ := 3^2
  have term1 : ℤ := (-2)^n
  have term2 : ℤ := 2^n
  have h1 : term1 = -512 := by
    have h1_1 : n = 9 := by norm_num
    rw [h1_1]
    norm_num
  have h2 : term2 = 512 := by
    have h2_1 : n = 9 := by norm_num
    rw [h2_1]
    norm_num
  rw [h1, h2]
  norm_num
  done

end power_add_zero_l335_335826


namespace discount_difference_is_24_l335_335692

-- Definitions based on conditions
def smartphone_price : ℝ := 800
def single_discount_rate : ℝ := 0.25
def first_successive_discount_rate : ℝ := 0.20
def second_successive_discount_rate : ℝ := 0.10

-- Definitions of discounted prices
def single_discount_price (p : ℝ) (d1 : ℝ) : ℝ := p * (1 - d1)
def successive_discount_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ := 
  let intermediate_price := p * (1 - d1) 
  intermediate_price * (1 - d2)

-- Calculate the difference between the two final prices
def price_difference (p : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (single_discount_price p d1) - (successive_discount_price p d2 d3)

theorem discount_difference_is_24 :
  price_difference smartphone_price single_discount_rate first_successive_discount_rate second_successive_discount_rate = 24 := 
sorry

end discount_difference_is_24_l335_335692


namespace min_number_top_block_l335_335845

/--
Consider a new arrangement of cubical blocks where:
- The base layer has 12 blocks arranged in a 4x3 pattern.
- A second layer of 6 blocks rests on the base layer arranged such that each block sits on 2 base layer blocks.
- A third layer of 2 blocks is placed each on three of the second layer blocks.
- A single top block rests on the two third-layer blocks.
The blocks in the base layer are numbered from 1 to 12 in some order. 
Each block in subsequent layers gets a number that is the sum of the numbers of the blocks directly beneath it.
-/
theorem min_number_top_block (L1_blocks: Fin 12 → ℕ) :  
  let L2_blocks := λ i, 
    (L1_blocks (Fin.ofNat 2 * i) + L1_blocks (Fin.ofNat 2 * i + 1)) 
    in let L3_blocks := λ i, 
      (L2_blocks (Fin.ofNat 2 * i) + L2_blocks (Fin.ofNat 2 * i + 1) + L2_blocks (Fin.ofNat 3)) 
      in let L4_block := λ, 
        L3_blocks 0 + L3_blocks 1
        in
      (∀ i, i < 12 → L1_blocks i ∈ (1..12) ) →
      (∀ i j, L1_blocks i = L1_blocks j → i = j) →
      (L4_block = 89) :=
  sorry

end min_number_top_block_l335_335845


namespace part_I_part_II_l335_335161

-- Definition of the functions and their conditions
def f (a b x : ℝ) := (1 / 2) * a * x^2 + b * x
def g (x : ℝ) := 1 + log x

-- The question (I)
theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (b = 1) →
  ∃ I, ∀ x ∈ I, deriv (g x - f a b x) < 0 →
  a ∈ (-1/4, 0) ∪ (0, +∞) :=
sorry

-- The question (II)
theorem part_II (a b x1 x2: ℝ) (h_a : a ≠ 0) (h_x : 0 < x1 ∧ x1 < x2) :
  let T := (x1 + x2) / 2
  in ¬ (deriv g T = deriv (f a b T)) :=
sorry

end part_I_part_II_l335_335161


namespace inequality_solution_l335_335878

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := 3/4 ≤ x ∧ x ≤ 2

-- Theorem statement to prove the equivalence
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ solution_set x := by
  sorry

end inequality_solution_l335_335878


namespace total_weight_remaining_eggs_l335_335622

theorem total_weight_remaining_eggs :
  let large_egg_weight := 14
  let medium_egg_weight := 10
  let small_egg_weight := 6

  let box_A_weight := 4 * large_egg_weight + 2 * medium_egg_weight
  let box_B_weight := 6 * small_egg_weight + 2 * large_egg_weight
  let box_C_weight := 4 * large_egg_weight + 3 * medium_egg_weight
  let box_D_weight := 4 * medium_egg_weight + 4 * small_egg_weight
  let box_E_weight := 4 * small_egg_weight + 2 * medium_egg_weight

  total_weight := box_A_weight + box_C_weight + box_D_weight + box_E_weight
  total_weight = 270 := 
by
  sorry

end total_weight_remaining_eggs_l335_335622


namespace count_zeros_in_Q_l335_335849

theorem count_zeros_in_Q : 
  let R_k (k : ℕ) := (10^k - 1) / 9 
  let Q : ℕ := R_k 30 / R_k 6
  (∀ k : ℕ, Q = 1 + 10^6 + 10^12 + 10^18 + 10^24 → (nat.factors Q).count 0 = 25) :=
by {
  sorry
}

end count_zeros_in_Q_l335_335849


namespace greatest_integer_gcd_four_l335_335724

theorem greatest_integer_gcd_four {n : ℕ} (h1 : n < 150) (h2 : Nat.gcd n 12 = 4) : n <= 148 :=
by {
  sorry
}

end greatest_integer_gcd_four_l335_335724


namespace arithmetic_expression_l335_335721

theorem arithmetic_expression :
  8 / 2 - 3 - 10 + 3 * 9 = 18 :=
by linarith

end arithmetic_expression_l335_335721


namespace exists_divisible_by_2011_pow_2012_l335_335342

def sequence (u : ℕ → ℤ) : Prop :=
  u 0 = 0 ∧ u 1 = 0 ∧ ∀ n : ℕ, u (n + 2) = u (n + 1) + u n + 1

theorem exists_divisible_by_2011_pow_2012 (u : ℕ → ℤ) (h_seq : sequence u) :
  ∃ n : ℕ, n ≥ 1 ∧ 2011 ^ 2012 ∣ u n ∧ 2011 ^ 2012 ∣ u (n + 1) :=
by
  sorry

end exists_divisible_by_2011_pow_2012_l335_335342


namespace solve_for_x_l335_335551

theorem solve_for_x : (∃ x : ℝ, 2 ^ real.log2 7 = 3 * x - 4) → x = 11 / 3 :=
by
  sorry

end solve_for_x_l335_335551


namespace find_slope_for_given_area_l335_335422

theorem find_slope_for_given_area :
  ∀ m : ℝ, let c := 1 in
  let vertex := (0, 2) in
  let parabola := λ x : ℝ, x^2 + 2 in
  let line := λ x : ℝ, m * x + c in
  let intersection_points := {x : ℝ | parabola x = line x} in
  let base := 
    let x₁ := (m + Real.sqrt (m^2 - 4)) / 2 in
    let x₂ := (m - Real.sqrt (m^2 - 4)) / 2 in
    |x₁ - x₂| in
  let height := |2 - c| in
  let area := (1 / 2) * base * height in
  area = 12 -> m = Real.sqrt 580 ∨ m = - Real.sqrt 580 :=
sorry

end find_slope_for_given_area_l335_335422


namespace chord_length_is_2_l335_335152

noncomputable def chord_length (x y : ℝ) : ℝ :=
  (x - 2)^2 + y^2 - (4 : ℝ)

/-- Given a line l: y = √3 * x intersects the circle C: x^2 - 4x + y^2 = 0 at points A and B,
proving the length of the chord AB is 2 -/
theorem chord_length_is_2 {x y : ℝ} (h_line : y = real.sqrt(3) * x)
  (h_circle : x^2 - 4 * x + y^2 = 0) : 
  2 * real.sqrt(4 - (2 * real.sqrt(3))^2 / (3 + 1)) = 2 :=
by
  sorry

end chord_length_is_2_l335_335152


namespace frequency_and_student_count_in_range_79_89_l335_335825

-- Given 200 students took the test and the frequencies of test score ranges.
def students_count : ℕ := 200
def frequency_59_69 : ℝ := 0.1
def frequency_69_79 : ℝ := 0.3
def frequency_89_99 : ℝ := 0.2

-- The missing frequency and number of students in the score range 79.5 to 89.5.
theorem frequency_and_student_count_in_range_79_89 :
  let frequency_79_89 := 1 - (frequency_59_69 + frequency_69_79 + frequency_89_99)
  in frequency_79_89 = 0.4 ∧ (students_count * frequency_79_89).to_nat = 80 :=
by
  let frequency_79_89 := 1 - (frequency_59_69 + frequency_69_79 + frequency_89_99)
  have h1 : frequency_79_89 = 0.4 := by
    unfold frequency_79_89
    norm_num
  have h2 : (students_count * frequency_79_89).to_nat = 80 := by
    unfold students_count frequency_79_89
    norm_num
  exact ⟨h1, h2⟩

end frequency_and_student_count_in_range_79_89_l335_335825


namespace evaluate_expression_l335_335458

theorem evaluate_expression :
  81^(1/2:ℝ) * 8^(-1/3:ℝ) * 32^(1/5:ℝ) = 9 :=
by
  sorry

end evaluate_expression_l335_335458


namespace length_sum_of_intersections_l335_335915

noncomputable def curve_C := ∀ (x y : ℝ), x^2 + y^2 = 4 * x

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + (real.sqrt 3) / 2 * t, 1 / 2 * t)

def intersection_points (t : ℝ) := 
  let (x, y) := parametric_line t in
  x ^ 2 + y ^ 2 = 4 * x

theorem length_sum_of_intersections :
  (∀ (t : ℝ), intersection_points t) → 
  let t1 := (real.sqrt 3 + real.sqrt (3 + 12)) / 2 in
  let t2 := (real.sqrt 3 - real.sqrt (3 + 12)) / 2 in
  |t1| + |t2| = real.sqrt 15 :=
by
  sorry

end length_sum_of_intersections_l335_335915


namespace largest_number_l335_335449

theorem largest_number 
  (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)
  (hA : A = 0.986)
  (hB : B = 0.9851)
  (hC : C = 0.9869)
  (hD : D = 0.9807)
  (hE : E = 0.9819)
  : C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_l335_335449


namespace fourier_series_y_eq_l335_335854

noncomputable def fourier_series_decomposition (x : ℝ) : ℝ :=
  1 + (4 / Real.pi) * ∑' n, (if n = 0 then 0 else (-1)^(n-1) / n) * Real.sin (n * Real.pi / 2 * x)

theorem fourier_series_y_eq (x : ℝ) (h : x ∈ Ioo (-2 : ℝ) 2) : 
  x + 1 = fourier_series_decomposition x :=
sorry

end fourier_series_y_eq_l335_335854


namespace girls_together_arrangements_no_two_girls_adjacent_arrangements_exactly_three_between_arrangements_adjacent_not_next_to_arrangements_l335_335888

open Finset

section problem_conditions

def boys : Finset ℕ := range 4
def girls : Finset ℕ := range 3
def people : Finset ℕ := boys ∪ girls

-- Definitions for specific conditions
def girls_together (arrangement : List ℕ) : Prop :=
  ⟦arrangement ≃ filter (λ x, x ∈ girls) arrangement⟧

def no_two_girls_adjacent (arrangement : List ℕ) : Prop :=
  ∀ i ∈ range (arrangement.length - 1), arrangement[i] ∉ girls ∨ arrangement[i + 1] ∉ girls

def exactly_three_between (arrangement : List ℕ) (A B : ℕ) : Prop :=
  ∃ i j, arrangement[i] = A ∧ arrangement[j] = B ∧ abs (i - j) = 4

def adjacent_not_next_to (arrangement : List ℕ) (A B C : ℕ) : Prop :=
  ∃ i, arrangement[i] = A ∧ arrangement[i + 1] = B ∧ ∀ j, j ≠ i ∧ j ≠ i + 1 → arrangement[j] ≠ C

end problem_conditions

-- Theorems for the subproblems
theorem girls_together_arrangements :
  ∃ arrangements : Finset (List ℕ), girls_together arrangements ∧ arrangements.card = 720 := sorry

theorem no_two_girls_adjacent_arrangements :
  ∃ arrangements : Finset (List ℕ), no_two_girls_adjacent arrangements ∧ arrangements.card = 1440 := sorry

theorem exactly_three_between_arrangements (A B : ℕ) :
  ∃ arrangements : Finset (List ℕ), exactly_three_between arrangements A B ∧ arrangements.card = 720 := sorry

theorem adjacent_not_next_to_arrangements (A B C : ℕ) :
  ∃ arrangements : Finset (List ℕ), adjacent_not_next_to arrangements A B C ∧ arrangements.card = 960 := sorry

end girls_together_arrangements_no_two_girls_adjacent_arrangements_exactly_three_between_arrangements_adjacent_not_next_to_arrangements_l335_335888


namespace perpendicular_vectors_implies_k_eq_2_l335_335942

variable (k : ℝ)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, k)

theorem perpendicular_vectors_implies_k_eq_2 (h : (2 : ℝ) * (-1 : ℝ) + (1 : ℝ) * k = 0) : k = 2 := by
  sorry

end perpendicular_vectors_implies_k_eq_2_l335_335942


namespace num_students_above_threshold_l335_335303

open List

def student_heights : List ℝ := [161.5, 154.3, 143.7, 160.1, 158.0, 153.5, 147.8]
def threshold : ℝ := 150.0

def count_above_threshold (heights : List ℝ) (threshold : ℝ) : Nat :=
  heights.countp (λ h => h > threshold)

theorem num_students_above_threshold : count_above_threshold student_heights threshold = 5 := by
  sorry

end num_students_above_threshold_l335_335303


namespace sum_of_solutions_abs_inequality_l335_335727
-- Lean 4 statement for the given problem

theorem sum_of_solutions_abs_inequality :
  (∑ n in Finset.filter (λ n : ℤ, |n + 2| < |n - 3| ∧ |n - 3| < 10) (Finset.Icc (-7) 12), n) = -28 :=
by
  sorry

end sum_of_solutions_abs_inequality_l335_335727


namespace prove_sin_cos_15_equals_one_fourth_l335_335861

noncomputable def sin_cos_15_equals_one_fourth : Prop :=
  sin (π / 12) * cos (π / 12) = 1 / 4

theorem prove_sin_cos_15_equals_one_fourth :
  sin_cos_15_equals_one_fourth :=
by 
  -- The actual proof steps would go here
  sorry

end prove_sin_cos_15_equals_one_fourth_l335_335861


namespace base_eight_square_unique_digits_l335_335664

theorem base_eight_square_unique_digits :
  ∃ (A B C : ℕ), A = 2 ∧ B = 5 ∧ C = 6 ∧
  let num := (2 * 64) + (5 * 8) + 6 in
  let square := (num * num) in
  let digits := [2, 5, 6, 0, 1, 3, 4, 7] in
  ∀ d ∈ digits, ∃ i j, square = i * 8 + j ∧ d = j % 8 :=
sorry

end base_eight_square_unique_digits_l335_335664


namespace number_of_complex_solutions_l335_335472

theorem number_of_complex_solutions :
  (∀ z : ℂ, (z^4 - 1) / (z^3 + z^2 - 2z) = 0 ↔ ((z^2 + 1) * (z - 1) * (z + 1) = 0 ∧ z ≠ 0 ∧ (z - 1) ≠ 0 ∧ (z + 2) ≠ 0)) →
  ∃! n : ℕ, n = 2 :=
by
  sorry

end number_of_complex_solutions_l335_335472


namespace game_no_winning_move_for_first_player_l335_335353

theorem game_no_winning_move_for_first_player :
  ∀ (initial_move : Fin 8 × Fin 8), ∃ (strategy_for_player_2 : (Fin 8 × Fin 8) → (Fin 8 × Fin 8)),
    (∀ move_for_player_1 move_for_player_2, move_for_player_2 = strategy_for_player_2 move_for_player_1) →
    ∃ move_chain_1 move_chain_2, move_chain_2 = strategy_for_player_2 move_chain_1 ∧
    move_chain_1 ≠ move_chain_2 :=
begin
  sorry,
end

end game_no_winning_move_for_first_player_l335_335353


namespace Kaleb_spring_earnings_l335_335226

variables (S : ℝ) (summerEarnings : ℝ) (suppliesCost : ℝ) (finalAmount : ℝ)

-- Define the conditions as hypotheses
hypothesis h1 : finalAmount = 50
hypothesis h2 : summerEarnings = 50
hypothesis h3 : suppliesCost = 4
hypothesis h4 : finalAmount + suppliesCost = summerEarnings + S

-- State the theorem
theorem Kaleb_spring_earnings : S = 4 :=
by
  -- your proof steps would go here
  sorry

end Kaleb_spring_earnings_l335_335226


namespace trig_identity_l335_335178

theorem trig_identity (x : ℝ) (h : sin x + cos x + 2 * tan x + 2 * cot x + 2 * sec x + 2 * csc x = 10) : sin (2 * x) = 2 := 
by 
  sorry

end trig_identity_l335_335178


namespace arithmetic_square_root_of_a_minus_b_l335_335937

noncomputable def square_root_arithmetic (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.sqrt x else 0

theorem arithmetic_square_root_of_a_minus_b : ∀ (a b : ℝ),
    (∃ (k : ℝ), a + 3 = k ∧ (2 * a) - 6 = k ∧ k > 0) →
    (Real.cbrt b = -2) → 
    square_root_arithmetic (a - b) = 3 :=
  by
    intros a b h1 h2
    sorry

end arithmetic_square_root_of_a_minus_b_l335_335937


namespace psychology_charge_l335_335360

noncomputable def F : ℝ := 30 + A
def A : ℝ := 74 -- from solving the equations
def total_charge_5_hours : ℝ := F + 4 * A
def total_charge_3_hours : ℝ := F + 2 * A

theorem psychology_charge :
  total_charge_5_hours = 400 → total_charge_3_hours = 252 :=
by
  sorry

end psychology_charge_l335_335360


namespace sum_gives_remainder_l335_335754

noncomputable theory
open Nat

theorem sum_gives_remainder
  (n : ℕ)
  (nums : Fin n → ℕ)
  (coprime_prop : ∀ i, gcd (nums i) n = 1)
  (r : ℕ)
  (h_r : r < n)
  : ∃ (subset : Finset (Fin n)), (∑ i in subset, nums i) % n = r :=
sorry

end sum_gives_remainder_l335_335754


namespace no_unique_pairings_of_bracelets_l335_335868

theorem no_unique_pairings_of_bracelets :
  ∃ (bracelets : Finset ℕ) (days : Finset (Finset ℕ)),
  bracelets.card = 100 ∧ (∀ day ∈ days, (day.card = 3 ∧ day ⊆ bracelets)) ∧
  ¬(∀ pair ∈ bracelets.powersetLen 2, ∃! day ∈ days, pair ⊆ day) :=
by sorry

end no_unique_pairings_of_bracelets_l335_335868


namespace average_birds_seen_l335_335256

theorem average_birds_seen (marcus birds: ℕ) (humphrey birds: ℕ) (darrel birds: ℕ) (isabella birds: ℕ) :
  marcus = 7 ∧ humphrey = 11 ∧ darrel = 9 ∧ isabella = 15 →
  (marcus + humphrey + darrel + isabella) / 4 = 10.5 :=
by
  intros h
  rcases h with ⟨h_marcus, ⟨h_humphrey, ⟨h_darrel, h_isabella⟩⟩⟩
  simp [h_marcus, h_humphrey, h_darrel, h_isabella]
  norm_num
  sorry

end average_birds_seen_l335_335256


namespace perimeter_of_triangle_ABC_l335_335953

def condition_a_b (a b : ℝ) : Prop := |a - 2| + (b - 5)^2 = 0

def condition_c (x : ℝ) : Prop := (x - 3 > 3 * (x - 4)) ∧ ((4 * x - 1) / 6 < x + 1)

def largest_integer_solution (P : ℝ → Prop) : ℤ :=
  by {haveI := Classical.decEq ℤ, exact (Finset.Icc (-3 : ℤ) 4).filterMap (λ z, if P z then some z else none).max' trivial }

theorem perimeter_of_triangle_ABC (a b c : ℝ) 
  (h_ab : condition_a_b a b) 
  (h_c : largest_integer_solution condition_c = 4) : 
  a + b + ↑(largest_integer_solution condition_c) = 11 :=
sorry

end perimeter_of_triangle_ABC_l335_335953


namespace P_coordinates_correct_l335_335790

noncomputable def P_coordinates (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sqrt 3 * Real.sin α)

theorem P_coordinates_correct :
  ∀ (α : ℝ), |P_coordinates α = (1/2, 3/2)| :=
by
  sorry

end P_coordinates_correct_l335_335790


namespace greg_walked_three_dogs_for_nine_minutes_l335_335943

variables 
  (charge_per_dog : ℕ := 20)
  (charge_per_minute : ℕ := 1)
  (minutes_one_dog : ℕ := 10)
  (minutes_two_dogs : ℕ := 7)
  (total_earnings : ℕ := 171)
  (earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * minutes_one_dog)
  (earnings_two_dogs : ℕ := 2 * (charge_per_dog + charge_per_minute * minutes_two_dogs))
  (earnings_two_groups : ℕ := earnings_one_dog + earnings_two_dogs)
  (remaining_earnings : ℕ := total_earnings - earnings_two_groups)

theorem greg_walked_three_dogs_for_nine_minutes (minutes_three_dogs : ℕ) :
  (3 * (charge_per_dog + charge_per_minute * minutes_three_dogs) = remaining_earnings) → (minutes_three_dogs = 9) :=
begin
  -- This is where the proof would take place. 
  sorry
end

end greg_walked_three_dogs_for_nine_minutes_l335_335943


namespace triangle_isosceles_l335_335997

theorem triangle_isosceles 
  (A B C : ℝ) (hTriangle : A + B + C = π) 
  (hCondition : 2 * cos B * sin A = sin C) : A = B :=
by
  -- The actual proof is omitted
  sorry

end triangle_isosceles_l335_335997


namespace greatest_power_divides_factorial_l335_335107

open Nat

noncomputable def greatest_power_dividing_factorial (p : ℕ) (n : ℕ) [hp : Fact p.Prime] : ℕ :=
  ∑ i in Finset.range (n+1), n / p^i

theorem greatest_power_divides_factorial (p : ℕ) (n : ℕ) [hp : Fact p.Prime] :
  ∃ k : ℕ, (p^k ∣ n.factorial) ∧
  (∀ j : ℕ, (p^j ∣ n.factorial) → j ≤ k) :=
begin
  use greatest_power_dividing_factorial p n,
  sorry
end

end greatest_power_divides_factorial_l335_335107


namespace anns_age_l335_335418

theorem anns_age (a b : ℕ)
  (h1 : a + b = 72)
  (h2 : ∃ y, y = a - b)
  (h3 : b = a / 3 + 2 * (a - b)) : a = 36 :=
by
  sorry

end anns_age_l335_335418


namespace elizabeth_bracelets_l335_335869

theorem elizabeth_bracelets :
  ∀ (n m : ℕ), n = 100 → m = 3 →
  (∃ days : ℕ, ∀ (b1 b2 : ℕ), b1 < b2 → b1 < n → b2 < n 
  → (∃ day : ℕ, day < days ∧ day ∈ {1, 2, ..., days} ∧
     b1 ∈ {br | br < n ∧ ∃ s ∈ {1, 2, ..., 3}, br = bracelet(day, s)} ∧
     b2 ∈ {br | br < n ∧ ∃ s ∈ {1, 2, ..., 3}, br = bracelet(day, s)} )) = False :=
begin
  intros n m hn hm,
  rw [hn, hm],
  sorry
end

end elizabeth_bracelets_l335_335869


namespace arithmetic_sequence_general_formula_T_n_formula_l335_335510

-- Definitions
def a2 : ℕ := 5
def S4 : ℕ := 28
def b (n : ℕ) : ℕ := 2^n

-- General formula for the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := 4 * n - 3

-- The formula for T_n
def T (n : ℕ) : ℕ := (4 * n - 3) * 2 + ∑ i in finset.range (n - 1), (4 * (n-i-1) - 3) * 2^(i + 2)

theorem arithmetic_sequence_general_formula :
  ∀ n, a n = 4 * n - 3 := 
by sorry

theorem T_n_formula :
  ∀ n, T n = -8 * n - 10 + 5 * 2^(n + 1) :=
by sorry

end arithmetic_sequence_general_formula_T_n_formula_l335_335510


namespace geometric_configuration_l335_335202

theorem geometric_configuration :
  ∀ (AB A'B' AD DB A'D' D'B' x y : ℝ) (D D' P P' : ℝ) (A B A' B' : Prop),
  AB = 6 ∧ A'B' = 10 ∧ A = (AB / 2) ∧ B = (AB / 2) ∧ A' = (A'B' / 2) ∧ B' = (A'B' / 2) ∧
  P = 2 ∧ x = P ∧ (A + P + y + B') = 12 → x + y = 4 :=
begin
  sorry
end

end geometric_configuration_l335_335202


namespace cats_not_eating_tuna_or_chicken_l335_335980

theorem cats_not_eating_tuna_or_chicken (total_cats : ℕ) (cats_like_tuna : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) :
    total_cats = 75 →
    cats_like_tuna = 18 →
    cats_like_chicken = 55 →
    cats_like_both = 10 →
    (total_cats - ((cats_like_tuna - cats_like_both) + (cats_like_chicken - cats_like_both) + cats_like_both)) = 12 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Placeholder for detailed steps in the proof
  sorry

end cats_not_eating_tuna_or_chicken_l335_335980


namespace medium_bed_rows_l335_335168

theorem medium_bed_rows (large_top_beds : ℕ) (large_bed_rows : ℕ) (large_bed_seeds_per_row : ℕ) 
                         (medium_beds : ℕ) (medium_bed_seeds_per_row : ℕ) (total_seeds : ℕ) :
    large_top_beds = 2 ∧ large_bed_rows = 4 ∧ large_bed_seeds_per_row = 25 ∧
    medium_beds = 2 ∧ medium_bed_seeds_per_row = 20 ∧ total_seeds = 320 →
    ((total_seeds - (large_top_beds * large_bed_rows * large_bed_seeds_per_row)) / medium_bed_seeds_per_row) = 6 :=
by
  intro conditions
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := conditions
  sorry

end medium_bed_rows_l335_335168


namespace determine_polynomial_l335_335445

theorem determine_polynomial (p : ℝ → ℝ) (h₁ : p 3 = 10) (h₂ : ∀ x y : ℝ, p(x) * p(y) = p(x) + p(y) + p(x * y) - 2) : 
  ∀ x : ℝ, p(x) = x^2 + 1 :=
by
  sorry

end determine_polynomial_l335_335445


namespace find_positive_m_l335_335477

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l335_335477


namespace YaoMing_stride_impossible_l335_335023

-- Defining the conditions as Lean definitions.
def XiaoMing_14_years_old (current_year : ℕ) : Prop := current_year = 14
def sum_of_triangle_angles (angles : ℕ) : Prop := angles = 180
def CCTV5_broadcasting_basketball_game : Prop := ∃ t : ℕ, true -- Random event placeholder
def YaoMing_stride (stride_length : ℕ) : Prop := stride_length = 10

-- The main statement: Prove that Yao Ming cannot step 10 meters in one stride.
theorem YaoMing_stride_impossible (h1: ∃ y : ℕ, XiaoMing_14_years_old y) 
                                  (h2: ∃ a : ℕ, sum_of_triangle_angles a) 
                                  (h3: CCTV5_broadcasting_basketball_game) 
: ¬ ∃ s : ℕ, YaoMing_stride s := sorry

end YaoMing_stride_impossible_l335_335023


namespace sum_of_possible_values_l335_335236

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 5 * x + 6
def g (x : ℝ) : ℝ := 3 * x - 4

-- The problem statement in Lean
theorem sum_of_possible_values :
  g(f(3)) + g(f(2)) = 7 :=
by
  sorry

end sum_of_possible_values_l335_335236


namespace div_decimals_l335_335430

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end div_decimals_l335_335430


namespace green_turtles_1066_l335_335033

def number_of_turtles (G H : ℕ) : Prop :=
  H = 2 * G ∧ G + H = 3200

theorem green_turtles_1066 : ∃ G : ℕ, number_of_turtles G (2 * G) ∧ G = 1066 :=
by
  sorry

end green_turtles_1066_l335_335033


namespace sum_prime_factors_of_expected_value_l335_335788

noncomputable def permutation_sum_abs (n : ℕ) (a : Fin n → Fin n) : ℕ :=
  ∑ i, abs ((a i).val - i.val)

noncomputable def expected_permutation_sum_abs (n : ℕ) : ℕ :=
  (∑ a : Fin n → Fin n, permutation_sum_abs n a) / n!

theorem sum_prime_factors_of_expected_value :
  let n := 2012 in
  let S := expected_permutation_sum_abs n in
  primeFactorsSum S = 2083 :=
sorry

end sum_prime_factors_of_expected_value_l335_335788


namespace ratio_circumscribed_areas_l335_335011

-- Define the shapes and their properties
def same_perimeter (Q : ℝ) : Prop :=
  ∃ (w l : ℝ), 2 * (w + l) = Q ∧ l = 2 * w

-- Define the area calculations for the circles
def rectangle_circumscribed_area (Q : ℝ) : ℝ :=
  let w := Q / 6 in
  let l := 2 * w in
  let radius := (Q * real.sqrt 10) / 12
  π * radius^2

def triangle_circumscribed_area (Q : ℝ) : ℝ :=
  let s := Q / 3 in
  let radius := (Q * real.sqrt 3) / 9
  π * radius^2

-- Define the ratio calculation
def ratio_C_over_D (Q : ℝ) : ℝ :=
  (rectangle_circumscribed_area Q) / (triangle_circumscribed_area Q)

-- The statement we need to verify
theorem ratio_circumscribed_areas (Q : ℝ) (h : same_perimeter Q) : 
  ratio_C_over_D(Q) = 15 / 8 := 
sorry

end ratio_circumscribed_areas_l335_335011


namespace average_birds_seen_l335_335255

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end average_birds_seen_l335_335255


namespace actual_area_of_rhombus_is_correct_l335_335003

-- Definitions based on conditions
def scale_factor := 300 / 5  -- miles per inch
def short_diagonal_in_inches := 6  -- inches

-- Definitions based on derived calculations
def short_diagonal_in_miles := short_diagonal_in_inches * scale_factor

def side_length_of_triangle := (2 * short_diagonal_in_miles) / Real.sqrt 3
def area_of_one_triangle := (Real.sqrt 3 / 4) * side_length_of_triangle^2
def area_of_rhombus := 2 * area_of_one_triangle

-- Proof statement
theorem actual_area_of_rhombus_is_correct :
  area_of_rhombus = 86400 * Real.sqrt 3 :=
sorry

end actual_area_of_rhombus_is_correct_l335_335003


namespace inequality_pow_l335_335101

variable {n : ℕ}

theorem inequality_pow (hn : n > 0) : 
  (3:ℝ) / 2 ≤ (1 + (1:ℝ) / (2 * n)) ^ n ∧ (1 + (1:ℝ) / (2 * n)) ^ n < 2 := 
sorry

end inequality_pow_l335_335101


namespace intersection_M_N_l335_335121

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335121


namespace tangent_line_eq_F_monotonic_intervals_F_three_zeros_range_F_inequality_t_intervals_l335_335903

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.exp (-x) + x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * f x - g x

theorem tangent_line_eq (x : ℝ) : 
  let slope := Real.exp 1 - 1 in 
  let point := (1, Real.exp 1 - 1) in 
  y = slope * x 
  := sorry

theorem F_monotonic_intervals (a : ℝ) : 
  ∀ x ∈ Icc (-∞ : ℝ) (-1 : ℝ), differentiable_on ℝ (F e) 
  ∧ monotone_on (F e) Icc (-∞ : ℝ) (-1 : ℝ) 
  ∧ ∀ x ∈ Icc 0 ∞, differentiable_on ℝ (F e) 
  ∧ monotone_on (F e) Icc 0 ∞
  := sorry

theorem F_three_zeros_range (a : ℝ) (m : ℝ) :
  (Real.exp 1 - 1) < m ∧ m < 2 
  → ∃ x1 x2 x3, F e x1 = m ∧ F e x2 = m ∧ F e x3 = m
  := sorry

theorem F_inequality_t_intervals (t a : ℝ) (x1 x2 : ℝ) :
  0 < a ∧ a < 1 
  → (F a x1 = F a x1) 
  → (0 < t ∧ F a x1 + t * F a x2 > 0)
  := sorry

end tangent_line_eq_F_monotonic_intervals_F_three_zeros_range_F_inequality_t_intervals_l335_335903


namespace binomial_odd_coeff_sum_l335_335318

theorem binomial_odd_coeff_sum 
  (n : ℕ) (a₀ a₁ a₂ : ℕ → ℕ) 
  (h₁ : ∑ k in (filter (λ i, odd i) (range (n+1))), (binomial n k * (-1)^(n-k)) = 64) 
  (h₂ : (x - 1)^n = a₀ + a₁ * (x + 1) + a₂ * (x + 1)^2): 
  a₁ = 448 := 
sorry

end binomial_odd_coeff_sum_l335_335318


namespace proof_problem_l335_335618

def sequence_s (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n, S n = 2 * a n - 2^n

def cn_geometric (a : ℕ → ℕ) (c : ℕ → ℕ) :=
  ∀ n, c n = a (n + 1) - 2 * a n ∧ c n = 2^n ∧ ∃ r : ℕ, ∀ m, c (m + 1) = r * c m

def sum_t (T : ℕ → ℕ) :=
  ∀ n, T n = (∑ k in range n, (k + 1) / 2^(k + 1)) = (3 / 2) - (1 / 2^n) - ((n + 1) / 2^(n + 1))

theorem proof_problem :
  ∃ (a S : ℕ → ℕ) (T : ℕ → ℕ), sequence_s a S ∧
                                a 1 = 2 ∧ a 2 = 6 ∧
                                cn_geometric a (λ n, a (n + 1) - 2 * a n) ∧
                                sum_t T :=
  by
    sorry

end proof_problem_l335_335618


namespace min_AB_CD_l335_335534

-- Definitions
def parabola (y x : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 / 4
def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

-- Proposition
theorem min_AB_CD :
  ∃ (A B C D : ℝ × ℝ), 
  parabola A.2 A.1 ∧ parabola B.2 B.1 ∧ parabola C.2 C.1 ∧ parabola D.2 D.1 ∧
  circle A.1 A.2 ∧ circle B.1 B.2 ∧ circle C.1 C.2 ∧ circle D.1 D.2 ∧
  (∃ k : ℝ, line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2 ∧ 
                       line_through_focus k C.1 C.2 ∧ line_through_focus k D.1 D.2) ∧ 
  A.1 ≤ B.1 ∧ B.1 ≤ C.1 ∧ C.1 ≤ D.1 ∧
  |fst A - 1| + 1 = |A.1 - B.1| + 1 / 2 ∧
  |fst C - 1| + 1 = |C.1 - D.1| + 1 / 2 ∧
  min (abs ((fst A) + 1 / 2) + 4 * (abs ((fst D) + 1 / 2))) = (13 / 2) :=
sorry

end min_AB_CD_l335_335534


namespace intersecting_function_range_l335_335531

theorem intersecting_function_range (a : ℝ) (x0 y0 : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : x0 ≥ 2) 
  (h4 : y0 = (1/2)^x0) 
  (h5 : y0 = Real.log a x0) : 
  a ≥ 16 :=
sorry

end intersecting_function_range_l335_335531


namespace derivative_of_volume_is_surface_area_l335_335770

noncomputable def V_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem derivative_of_volume_is_surface_area (R : ℝ) (h : 0 < R) : 
  (deriv V_sphere R) = 4 * Real.pi * R^2 :=
by sorry

end derivative_of_volume_is_surface_area_l335_335770


namespace inradius_of_right_triangle_l335_335801

-- Definitions given in the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The main theorem statement
theorem inradius_of_right_triangle (a b c : ℕ) (h : is_right_triangle a b c) (h_sides : a = 9 ∧ b = 12 ∧ c = 15) : 
  let s := (a + b + c) / 2 in
  let A := (a * b) / 2 in
  let r := A / s in
  r = 3 :=
by
  -- This is where the proof would go
  sorry

end inradius_of_right_triangle_l335_335801


namespace veena_paid_fraction_l335_335759

structure BillPayments where
  Lasya_paid : ℚ
  Veena_paid : ℚ
  Akshitha_paid : ℚ
  fraction_veena_paid : ℚ

axiom initial_conditions (bp : BillPayments) :
  bp.Veena_paid = (1/2) * bp.Lasya_paid ∧
  bp.Akshitha_paid = (3/4) * bp.Veena_paid

noncomputable def prove_fraction_veena_paid (bp : BillPayments) : Prop :=
  bp.fraction_veena_paid = 4 / 15

theorem veena_paid_fraction {bp : BillPayments} (h : initial_conditions bp) :
  prove_fraction_veena_paid bp := sorry

end veena_paid_fraction_l335_335759


namespace Pollards_algorithm_complexity_l335_335279

-- Definitions for the initial conditions and properties
variable (u v : ℕ → ℕ) (n k : ℕ)
variable (u1 u2 : ℕ)
variable (p : ℕ) (is_prime_p : Nat.Prime p)
variable (upd_rule : ∀ k, u (k+1) = u k + 1 ∧ v (k+1) = v k + 2)

-- Statement of the main theorem
theorem Pollards_algorithm_complexity :
  (u 1 = u1) →
  (v 1 = u2) →
  (∀ k, u (k+1) = upd_rule.1 k) →
  (∀ k, v (k+1) = upd_rule.2 k) →
  (n > 1) →
  ∃ c : ℕ, ∀ k ≤ 2 * p, k <= c * p ∧ 
    (loop_time (Pollards_algorithm n) ≤ 2 * p * (log n)^2)
:= sorry

end Pollards_algorithm_complexity_l335_335279


namespace find_f_at_one_l335_335892

theorem find_f_at_one (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f(x) = a * x^4 + b * x^2 + 2 * x - 8)
  (h₂ : f(-1) = 10) : f(1) = -26 :=
by
  sorry

end find_f_at_one_l335_335892


namespace situps_difference_l335_335227

def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2
def emma_situps : ℕ := bob_situps / 3

theorem situps_difference : 
  (nathan_situps + bob_situps + emma_situps) - ken_situps = 60 := by
  sorry

end situps_difference_l335_335227


namespace quadratic_condition_l335_335267

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l335_335267


namespace indefinite_integral_l335_335843

theorem indefinite_integral :
  ∃ C : ℝ, ∀ x : ℝ, ∫ (8 * x - arctan (2 * x)) / (1 + 4 * x^2) dx = ln |(1 + 4 * x^2)| - (1 / 4) * (arctan (2 * x))^2 + C :=
by
  sorry

end indefinite_integral_l335_335843


namespace percentage_distribution_less_than_l335_335359

open Classical

noncomputable def percentage_less_than (m h : ℝ) (P : ℝ → ℝ) : ℝ :=
  P (m + h)

theorem percentage_distribution_less_than (m h : ℝ) (P : ℝ → ℝ) 
  (symmetric : ∀ x, P (m + x) = 1 - P (m - x)) 
  (h68 : P (m + h) - P (m - h) = 0.68) :
  percentage_less_than m h P = 0.84 :=
by
  have hm : P m = 0.5 := by sorry
  have h34 : P (m + h) - 0.5 = 0.34 := by sorry
  rw [percentage_less_than]
  calc
    P (m + h)
        = 0.5 + 0.34 := by rw [h34]; sorry
    _    = 0.84 := by norm_num

end percentage_distribution_less_than_l335_335359


namespace hyperbola_eccentricity_l335_335150

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : 2 * (1 : ℝ) + 1 = 0) (h2 : 0 < a) (h3 : 0 < b) 
  (h4 : b = 2 * a) : 
  (∃ e : ℝ, e = (Real.sqrt 5)) 
:= 
  sorry

end hyperbola_eccentricity_l335_335150


namespace correct_answers_count_l335_335205

/-- In an examination, a student scores 4 marks for every correct answer and loses 1 mark for every wrong answer.
    The student attempts all 60 questions and secures 120 marks.
    This theorem proves the number of correct answers attempted by the student. -/
theorem correct_answers_count (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 120) : c = 36 := 
by 
  field_simp [h1, h2]
  sorry

end correct_answers_count_l335_335205


namespace decreasing_interval_range_of_a_l335_335528

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem decreasing_interval :
  (∀ x > 0, deriv f x = 1 + log x) →
  { x : ℝ | 0 < x ∧ x < 1/e } = { x | 0 < x ∧ deriv f x < 0 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a * x - 6) →
  a ≤ 5 + log 2 :=
sorry

end decreasing_interval_range_of_a_l335_335528


namespace constant_added_to_data_l335_335413

theorem constant_added_to_data (X : List ℝ) (c : ℝ) (h : c ≠ 0) :
  (List.mean (X.map (λ x, x + c)) = List.mean X + c) ∧ 
  (List.variance (X.map (λ x, x + c)) = List.variance X) :=
by
  sorry

end constant_added_to_data_l335_335413


namespace game_scores_mod_five_l335_335204

theorem game_scores_mod_five :
  ∃ (s : Fin 20 → ℕ), StrictMono s ∧ (∑ i, s i) = 2020 ∧ (∑ i, s i ^ 3) % 5 = 0 :=
by
  sorry

end game_scores_mod_five_l335_335204


namespace repairs_cost_correctness_l335_335650

-- Define constants
def purchasePrice : ℝ := 45000
def sellingPrice : ℝ := 80000
def profitPercent : ℝ := 40.35

-- The repair cost we need to prove
def repairCost : ℝ := 12000

-- Define total investment and profit
def totalInvestment (R : ℝ) : ℝ := purchasePrice + R
def profit (R : ℝ) : ℝ := (profitPercent / 100) * (totalInvestment R)
def profitFromSale (R : ℝ) : ℝ := sellingPrice - (purchasePrice + R)

-- The proof statement
theorem repairs_cost_correctness : profitFromSale repairCost = profit repairCost :=
by
  sorry

end repairs_cost_correctness_l335_335650


namespace travel_time_l335_335630

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l335_335630


namespace trajectory_is_parabola_exists_point_P_l335_335207

-- Definitions: Conditions given in the problem
def is_tangent_to_circle (C : ℝ × ℝ) (r : ℝ) : Prop := 
  ∃ (x y : ℝ), (x - 1)^2 + y^2 = r^2 ∧ (C.1, C.2) = (x, y)

def is_tangent_to_line (C : ℝ × ℝ) (x_val : ℝ) : Prop := 
  C.1 = x_val

def parabola (x y : ℝ) : Prop := 
  y^2 = 4 * x

def moving_line_intersect (l : ℝ × ℝ → Prop) (T : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), m ≠ 0 ∧ (l A ∧ l B ∧ T A.fst A.snd ∧ T B.fst B.snd)

-- Statements: Proof goals
theorem trajectory_is_parabola (C : ℝ × ℝ) :
  is_tangent_to_circle C (1/2) → is_tangent_to_line C (-1/2) → parabola C.1 C.2 :=
sorry

theorem exists_point_P (T : ℝ → ℝ → Prop) (m : ℝ) (A B P : ℝ × ℝ) :
  (m < 0) → (moving_line_intersect (λ p, p.1 = m + t * p.2) T A B) → 
  ∃ P, (T P.1 P.2) ∧ P ≠ A ∧ P ≠ B ∧ (P.2^2 = -4 * m) :=
sorry

end trajectory_is_parabola_exists_point_P_l335_335207


namespace sum_of_sequence_max_sin_A_plus_sin_B_l335_335071

-- Problem 1: Sum of the sequence
theorem sum_of_sequence (n : ℕ) :
  (∑ k in Finset.range (n + 1), 1 / (k * (k + 1))) = n / (n + 1) := by
  sorry

-- Problem 2: Trigonometric identity in triangle
theorem max_sin_A_plus_sin_B (A B C a b c : ℝ) 
  (h : C = π / 3) (h1 : A + B = (2:ℝ) * π / 3) (h2 : c * Real.sin A = a * Real.cos C) :
  Real.sin A + Real.sin B = Real.sqrt 3 := by
  sorry

end sum_of_sequence_max_sin_A_plus_sin_B_l335_335071


namespace ellipse_equation_y_intercept_range_l335_335916

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := Real.sqrt 2
noncomputable def e := Real.sqrt 3 / 2
noncomputable def c := Real.sqrt 6
def M : ℝ × ℝ := (2, 1)

-- Condition: The ellipse equation form
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Question 1: Proof that the ellipse equation is as given
theorem ellipse_equation :
  ellipse x y ↔ (x^2) / 8 + (y^2) / 2 = 1 := sorry

-- Condition: Line l is parallel to OM
def slope_OM := 1 / 2
def line_l (m x y : ℝ) : Prop := y = slope_OM * x + m

-- Question 2: Proof of the range for y-intercept m given the conditions
theorem y_intercept_range (m : ℝ) :
  (-Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 2) ↔
  ∃ x1 y1 x2 y2,
    line_l m x1 y1 ∧ 
    line_l m x2 y2 ∧ 
    x1 ≠ x2 ∧ 
    y1 ≠ y2 ∧
    x1 * x2 + y1 * y2 < 0 := sorry

end ellipse_equation_y_intercept_range_l335_335916


namespace max_cut_trees_no_visible_stumps_l335_335976

-- Define the grid size and properties
def n := 100
def total_trees := n * n

-- Assume a function that determines if cutting trees will result in stumps visibility
def no_visible_stumps (cut : Fin n → Fin n → Bool) : Prop :=
  ∀ i j k l : Fin n, cut i j = true → cut k l = true → (i ≠ k ∧ j ≠ l)

-- Define the mathematical requirement
theorem max_cut_trees_no_visible_stumps :
  ∃ (cut : Fin n → Fin n → Bool), 
    (∑ i in Finset.finRange n, ∑ j in Finset.finRange n, if cut i j then 1 else 0) = 2500 ∧ 
    no_visible_stumps cut := by
  sorry

end max_cut_trees_no_visible_stumps_l335_335976


namespace find_cos2α_l335_335512

noncomputable def cos2α (tanα : ℚ) : ℚ :=
  (1 - tanα^2) / (1 + tanα^2)

theorem find_cos2α (h : tanα = (3 / 4)) : cos2α tanα = (7 / 25) :=
by
  rw [cos2α, h]
  -- here the simplification steps would be performed
  sorry

end find_cos2α_l335_335512


namespace shaina_chocolate_l335_335605

-- Define the conditions
def total_chocolate : ℚ := 48 / 5
def number_of_piles : ℚ := 4

-- Define the assertion to prove
theorem shaina_chocolate : (total_chocolate / number_of_piles) = (12 / 5) := 
by 
  sorry

end shaina_chocolate_l335_335605


namespace nonneg_integer_solutions_otimes_l335_335054

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l335_335054


namespace incorrect_statement_l335_335941

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := (2, 1)
noncomputable def c : ℝ × ℝ := (-4, -2)

-- Define the incorrect vector statement D
theorem incorrect_statement :
  ¬ ∀ (d : ℝ × ℝ), ∃ (k1 k2 : ℝ), d = (k1 * b.1 + k2 * c.1, k1 * b.2 + k2 * c.2) := sorry

end incorrect_statement_l335_335941


namespace parallelogram_side_lengths_l335_335787

theorem parallelogram_side_lengths (x y : ℚ) 
  (h1 : 12 * x - 2 = 10) 
  (h2 : 5 * y + 5 = 4) : 
  x + y = 4 / 5 := 
by 
  sorry

end parallelogram_side_lengths_l335_335787


namespace no_ratio_p_sq_terms_l335_335246

theorem no_ratio_p_sq_terms 
  (p : ℕ) (hp : Nat.Prime p) : 
  ∀ l : ℕ, l ≥ 1 → 
  ∀ m n : ℕ, 
  (m * (m + 1) / 2) = p^(2 * l) * (n * (n + 1) / 2) → 
  false := 
by 
  intro p hp l hl m n h 
  sorry

end no_ratio_p_sq_terms_l335_335246


namespace triangle_perimeter_l335_335522

-- Define roots of the quadratic equation and check the valid one.
theorem triangle_perimeter (a b : ℝ) (f := {x : ℝ | x^2 - 5*x + 4 = 0}) :
  a = 3 → b = 5 → (∀ x ∈ f, x = 4) → a + b + 4 = 12 :=
by
  intros h₁ h₂ h₃
  rw [← h₁, ← h₂]
  sorry

end triangle_perimeter_l335_335522


namespace weighted_average_correct_l335_335852

def english_marks := 51
def math_marks := 65
def physics_marks := 82
def chemistry_marks := 67
def biology_marks := 85
def history_marks := 63
def geography_marks := 78
def cs_marks := 90

def english_weight := 2
def math_weight := 3
def physics_weight := 2
def chemistry_weight := 1
def biology_weight := 1
def history_weight := 1
def geography_weight := 1
def cs_weight := 3

def total_weighted_marks := english_marks * english_weight + math_marks * math_weight + physics_marks * physics_weight +
                            chemistry_marks * chemistry_weight + biology_marks * biology_weight + history_marks * history_weight +
                            geography_marks * geography_weight + cs_marks * cs_weight

def total_weights := english_weight + math_weight + physics_weight + chemistry_weight + biology_weight + history_weight + 
                     geography_weight + cs_weight

def weighted_average := total_weighted_marks / total_weights.toFloat

theorem weighted_average_correct : weighted_average = 73.14 := by
  sorry

end weighted_average_correct_l335_335852


namespace part1_part2_part3_l335_335499

variable {x y z : ℝ}

-- Given condition
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)

theorem part1 : 
  (x / y + y / z + z / x) / 3 ≥ 1 := sorry

theorem part2 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3 := sorry

theorem part3 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x := sorry

end part1_part2_part3_l335_335499


namespace percentage_number_l335_335784

theorem percentage_number (b : ℕ) (h : b = 100) : (320 * b / 100) = 320 :=
by
  sorry

end percentage_number_l335_335784


namespace largest_integer_less_than_150_l335_335077

theorem largest_integer_less_than_150 :
  ∃ n : ℤ, n < 150 ∧ n % 8 = 5 ∧ (∀ m : ℤ, m < 150 ∧ m % 8 = 5 → m ≤ n) :=
by
  use 149
  split
  trivial
  split
  trivial
  sorry

end largest_integer_less_than_150_l335_335077


namespace common_ratio_of_geometric_sequence_l335_335774

theorem common_ratio_of_geometric_sequence :
  ∃ r : ℚ, 
  (let a := 12 in
   let second_term := -18 in
   a * r = second_term ∧
   a * r^2 = 27 ∧
   a * r^3 = -40.5 ∧
   r = -3/2) :=
sorry

end common_ratio_of_geometric_sequence_l335_335774


namespace find_positive_m_has_exactly_single_solution_l335_335483

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l335_335483


namespace find_y_values_l335_335240

noncomputable def satisfies_equation (x : ℝ) :=
  x^2 + 5 * (x / (x - 3))^2 = 50

def y_value (x : ℝ) :=
  ((x - 3)^2 * (x + 4)) / (2 * x - 5)

theorem find_y_values :
  ∃ y1 y2 : ℝ, (∃ x : ℝ, satisfies_equation x ∧ y_value x = y1) ∧
               (∃ x : ℝ, satisfies_equation x ∧ y_value x = y2) :=
sorry

end find_y_values_l335_335240


namespace range_of_a_l335_335237

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - 2 * x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x0 : ℝ, 0 < x0 ∧ x0 < 1 →
  (0 < (deriv (fun x => f a x - x)) x0)) →
  a > (4 / Real.exp (3 / 4)) :=
by
  intro h
  sorry

end range_of_a_l335_335237


namespace rainy_days_l335_335220

theorem rainy_days :
  ∃ (A : Finset ℕ) (H : A.card = 5), 
  ( ∃ (B : Finset (Finset ℕ)), 
      (∀ b ∈ B, b.card = 3 ∧ (∃ c : ℕ, c ∈ b ∧ c + 1 ∈ b ∧ c + 2 ∈ b)) ∧ 
      B.card = 9 ) := 
sorry

end rainy_days_l335_335220


namespace set_intersection_l335_335127

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l335_335127


namespace combinatorial_selection_l335_335580

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem combinatorial_selection (n_rows n_cols k : ℕ) (h1 : n_rows = 6) (h2 : n_cols = 6) (h3 : k = 4) :
  (choose n_rows k) * (choose n_cols k) * (Nat.factorial k) = 5400 := by
  sorry

#eval combinatorial_selection 6 6 4 rfl rfl rfl  -- This will return the expected result if proven correct

end combinatorial_selection_l335_335580


namespace triangle_propositions_l335_335995

theorem triangle_propositions (a b c : ℝ) (A B C : Real) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ABC : a ≠ b) (A_B_C_sum : A + B + C = π) : 
  (A:B:C=1:2:3 → a:b:c=1:\sqrt{3}:2) ∧ 
  (A > B → sin A > sin B) ∧ 
  (A = 30 * π / 180 → a = 3 → b = 4 → ∃ C', ∃ c', is_triangle ⟨a, b, c'⟩ ∧ is_triangle ⟨a, b, c⟩) → 
  false :=
begin
  sorry
end

end triangle_propositions_l335_335995


namespace coin_toss_sequence_count_l335_335979

theorem coin_toss_sequence_count 
  (n : ℕ) (HH HT TH TT : ℕ)
  (sequence_length : n = 15) 
  (count_HH : HH = 2) 
  (count_HT : HT = 3)
  (count_TH : TH = 4)
  (count_TT : TT = 5) : 
  number_of_valid_sequences n HH HT TH TT = 560 := sorry

end coin_toss_sequence_count_l335_335979


namespace gcd_204_85_l335_335872

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l335_335872


namespace min_value_achieved_at_pm_sqrt3_div_4_l335_335081

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (x^2 - real.sqrt 3 * |x| + 1) + real.sqrt (x^2 + real.sqrt 3 * |x| + 3)

theorem min_value_achieved_at_pm_sqrt3_div_4 :
  ∃ x : ℝ, f x = real.sqrt 7 ∧
           (∀ y : ℝ, f y ≥ real.sqrt 7) ∧
           (x = real.sqrt 3 / 4 ∨ x = -real.sqrt 3 / 4) :=
sorry

end min_value_achieved_at_pm_sqrt3_div_4_l335_335081


namespace area_ratio_lengths_AD_DC_l335_335566

-- Define the geometrical entities and given properties
variables {A B C D : Type} [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]
variables (triangle : Type) [decidable_triangle triangle] [inhabited triangle]
variables (angle : triangle → triangle → angle)

-- Define lengths and angles
variable (length : triangle → ℝ)
variable (angle_measure : angle → ℝ)

-- Given conditions
variable (BA BC AC : triangle)
variable (BD : angle)
variable (intersection_point : angle → triangle → triangle)
variable [decidable_eq (BA, BC, AC, BD)]

-- The angle bisector condition
variable (bisects : intersection_point BD AC = D)

-- Given lengths
variable [BA_eq_2BC : length BA = 2 * length BC]
variable [angle_ABC_is_120 : angle_measure (angle A B C) = 120]
variable [BC_eq_3 : length BC = 3]

-- The theorem to prove the ratio of the areas
theorem area_ratio (S1 S2 : ℝ) : 
  ∃ S1 S2, S1 / S2 = 1 / 2 :=
sorry

-- The theorem to prove lengths AD and DC
theorem lengths_AD_DC : 
  ∃ AD DC, length AD = 2 * real.sqrt 7 ∧ length DC = real.sqrt 7 :=
sorry

end area_ratio_lengths_AD_DC_l335_335566


namespace sequence_sum_bound_l335_335617

theorem sequence_sum_bound (a : ℕ+ → ℝ) 
  (h₁ : a 1 = 1)
  (h₂ : a 2 = 1/3)
  (h₃ : ∀ n : ℕ+, (1 + a n) * (1 + a (n + 2)) / ((1 + a (n + 1))^2) = a n * a (n + 2) / (a (n + 1))^2) :
  ∀ n : ℕ+, (∑ i in Finset.range n, a (i + 1)) < 34/21 :=
by
  sorry

end sequence_sum_bound_l335_335617


namespace distinct_real_roots_of_sum_l335_335498

variable {α : Type*} [linear_ordered_field α]

def quadratic (a b : α) (x : α) : α := x^2 + a * x + b

theorem distinct_real_roots_of_sum (n : ℕ) (a b : ℕ → α) (k : α)
  (h1 : 2 ≤ n)
  (h2 : ∀ i, 1 ≤ i → i ≤ n → b i = (a i)^2 / 4 - k)
  (h3 : ∀ i j, 1 ≤ i → i < j → j ≤ n → (a i - a j)^2 < 4 * k) :
  ∃ x1 x2 : α, x1 ≠ x2 ∧ (∑ i in finset.range n, quadratic (a i) (b i)) x = 0 :=
sorry

end distinct_real_roots_of_sum_l335_335498


namespace trailing_zeroes_500_fact_l335_335426

theorem trailing_zeroes_500_fact : 
  let count_multiples (n m : ℕ) := n / m 
  let count_5 := count_multiples 500 5
  let count_25 := count_multiples 500 25
  let count_125 := count_multiples 500 125
-- We don't count multiples of 625 because 625 > 500, thus its count is 0. 
-- Therefore: total trailing zeroes = count_5 + count_25 + count_125
  count_5 + count_25 + count_125 = 124 := sorry

end trailing_zeroes_500_fact_l335_335426


namespace C2_polar_to_cartesian_C1_C2_intersection_AB_l335_335987

-- Define parametric and polar equations as given
def C1_parametric (α t : ℝ) : ℝ × ℝ := (1 + t * cos α, 1 + t * sin α)

def C2_polar (ρ θ : ℝ) : Prop := ρ^2 = 4 * sqrt 2 * ρ * sin (θ + π / 4) - 4

-- Define the Cartesian form of C2
def C2_cartesian (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

-- Statement for Question (Ⅰ)
theorem C2_polar_to_cartesian :
  ∀ (ρ θ : ℝ), C2_polar ρ θ → ∃ x y : ℝ, ρ * cos θ = x ∧ ρ * sin θ = y ∧ C2_cartesian x y :=
by
  intros ρ θ h_polar
  use [ρ * cos θ, ρ * sin θ]
  sorry

-- Statement for Question (Ⅱ)
theorem C1_C2_intersection_AB :
  ∀ (α : ℝ), 
  ∃ t1 t2 : ℝ, 
    let (x1, y1) := C1_parametric α t1
    let (x2, y2) := C1_parametric α t2
    (C2_cartesian x1 y1 ∧ C2_cartesian x2 y2) → 
    (0 ≤ α ∧ α ≤ π) ∧ 
    (|((x1 - x2)^2 + (y1 - y2)^2)^0.5| ≤ 2 * sqrt 2) :=
by
  intro α
  sorry

end C2_polar_to_cartesian_C1_C2_intersection_AB_l335_335987


namespace intersection_M_N_l335_335120

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335120


namespace find_a_solution_l335_335443

theorem find_a_solution :
  ∃ (a : ℕ), 
  (∀ x : ℕ, (x > 0) → 
    ((∏ k in Finset.range (a + 1), (1 + (1 : ℚ) / (x + k))) = a - x) → 
    a = 7) :=
sorry

end find_a_solution_l335_335443


namespace find_alpha_l335_335518

open Real

theorem find_alpha
  (α β : ℝ)
  (h1: 0 < α ∧ α < π / 2)
  (h_a : (3/4, sin α))
  (h_b : (cos β, 1/3))
  (h_parallel : sin α * cos β = 1/4)
  : α = π / 12 ∨ α = 5 * π / 12 :=
by
  have h : 2 * α = π / 6 ∨ 2 α = 5 * π / 6,
  { sorry },
  cases h,
  { left,
    linarith, },
  { right,
    linarith, }
  sorry

end find_alpha_l335_335518


namespace sin_alpha_beta_inequality_l335_335272

theorem sin_alpha_beta_inequality (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2) :
  (sin α)^3 / (2*α - sin(2*α)) > (sin β)^2 / (2*β - sin(2*β)) := 
by
  -- Proof
  sorry

end sin_alpha_beta_inequality_l335_335272


namespace evaluate_expression_l335_335865

theorem evaluate_expression:
  (10⁻³ * 5⁻¹) / 10⁻⁴ = (1 / 50000000) :=
by
  sorry

end evaluate_expression_l335_335865


namespace min_positive_omega_l335_335301

theorem min_positive_omega :
  ∀ (ω : ℝ), (ω > 0) → 
  (∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = cos (ω * x + π / 4)) → 
  (∀ x : ℝ, f (x + π / 3) = sin(ω * x))) → 
  ω = 15 / 4 :=
by
  intro ω hω f hf hshift
  sorry

end min_positive_omega_l335_335301


namespace line_through_points_l335_335674

-- Define the points
variables (x1 y1 x2 y2 : ℝ)
variables (x1_eq : x1 = -2) (y1_eq : y1 = 3) (x2_eq : x2 = 1) (y2_eq : y2 = -6)

-- Define the slope and intercept
noncomputable def slope := (y2 - y1) / (x2 - x1)
noncomputable def intercept := y1 - slope * x1

-- Define the midpoint
def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

-- Define the equation of the line and m + b
def line_equation := ∀ x, slope * x + intercept = -3 * x - 3
def m_plus_b := slope + intercept

theorem line_through_points :
  x1_eq → y1_eq → x2_eq → y2_eq →
  slope = -3 ∧ intercept = -3 ∧
  (line_equation x1) ∧ (line_equation x2) ∧
  midpoint_x = -1 / 2 ∧ midpoint_y = -3 / 2 ∧
  m_plus_b = -6 :=
by {
  intros, 
  sorry 
}

end line_through_points_l335_335674


namespace sum_of_three_squares_l335_335022

-- Using the given conditions to define the problem.
variable (square triangle : ℝ)

-- Conditions
axiom h1 : square + triangle + 2 * square + triangle = 34
axiom h2 : triangle + square + triangle + 3 * square = 40

-- Statement to prove
theorem sum_of_three_squares : square + square + square = 66 / 7 :=
by
  sorry

end sum_of_three_squares_l335_335022


namespace value_of_derivative_at_1_l335_335524

noncomputable def f (x : ℝ) : ℝ := 2 * x * f' 1 + x^3

theorem value_of_derivative_at_1 : (deriv f 1) = -3 :=
by
  sorry

end value_of_derivative_at_1_l335_335524


namespace average_of_20_digits_l335_335288

theorem average_of_20_digits (f : Fin 14 → ℝ) (g : Fin 6 → ℝ) (A : ℝ) 
  (h1 : (∑ k, f k) / 14 = 390)
  (h2 : (∑ k, g k) / 6 = 756.67)
  (h3 : (∑ k, f k) + ∑ k, g k = 20 * A) : 
  A = 500.001 :=
sorry

end average_of_20_digits_l335_335288


namespace largest_prime_factor_of_expression_l335_335741

theorem largest_prime_factor_of_expression : 
  ∃ p, prime p ∧ p ≥ 2 ∧ 
  (∀ q, (q ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) → prime q → q ≤ p) ∧ 
  p = 241 :=
by
  sorry

end largest_prime_factor_of_expression_l335_335741


namespace math_ineq_l335_335367

variable {n : ℕ} (x : Fin (n + 1) → ℝ)

theorem math_ineq (h1 : ∀ i : Fin (n + 1), 0 < x i) (h2 : 2 ≤ n) :
  (Finset.univ.sum (λ i, (x i / x (i + 1) % (n + 1)) ^ n)) 
    ≥ (Finset.univ.sum (λ i, x (i + 1) % (n + 1) / x i)) := 
by
  sorry

end math_ineq_l335_335367


namespace intersection_of_M_and_N_l335_335156

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- The theorem to be proved
theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_M_and_N_l335_335156


namespace supermarket_selection_expected_value_l335_335196

noncomputable def small_supermarkets := 72
noncomputable def medium_supermarkets := 24
noncomputable def large_supermarkets := 12
noncomputable def total_supermarkets := small_supermarkets + medium_supermarkets + large_supermarkets
noncomputable def selected_supermarkets := 9

-- Problem (I)
noncomputable def small_selected := (small_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def medium_selected := (medium_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def large_selected := (large_supermarkets * selected_supermarkets) / total_supermarkets

theorem supermarket_selection :
  small_selected = 6 ∧ medium_selected = 2 ∧ large_selected = 1 :=
sorry

-- Problem (II)
noncomputable def further_analysis := 3
noncomputable def prob_small := small_selected / selected_supermarkets
noncomputable def E_X := prob_small * further_analysis

theorem expected_value :
  E_X = 2 :=
sorry

end supermarket_selection_expected_value_l335_335196


namespace angle_BIX_21_angle_BCA_54_l335_335264

theorem angle_BIX_21 (A B C I X : Type)
  [Inner product_space ℝ A] [Inner product_space ℝ B] [Inner product_space ℝ C] [Inner product_space ℝ I]
  [Inner product_space ℝ X]
  (hI : I = intersection_point_of_angle_bisectors A B C)
  (hX : X ∈ ℓ(B, C))
  (h1 : dist A I = dist B X)
  (h2 : dist A C = dist C X)
  (h3 : ∠B A C = 42) : 
  ∠B I X = 21 :=
sorry

theorem angle_BCA_54 (A B C I X : Type)
  [Inner product_space ℝ A] [Inner product_space ℝ B] [Inner product_space ℝ C] [Inner product_space ℝ I]
  [Inner product_space ℝ X]
  (hI : I = intersection_point_of_angle_bisectors A B C)
  (hX : X ∈ ℓ(B, C))
  (h1 : dist A I = dist B X)
  (h2 : dist A C = dist C X)
  (h3 : ∠B A C = 42) : 
  ∠B C A = 54 :=
sorry

end angle_BIX_21_angle_BCA_54_l335_335264


namespace find_x_minus_y_l335_335184

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 :=
by
  have h3 : x^2 - y^2 = (x + y) * (x - y) := by sorry
  have h4 : (x + y) * (x - y) = 8 * (x - y) := by sorry
  have h5 : 16 = 8 * (x - y) := by sorry
  have h6 : 16 = 8 * (x - y) := by sorry
  have h7 : x - y = 2 := by sorry
  exact h7

end find_x_minus_y_l335_335184


namespace focus_of_parabola_l335_335857

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the coordinates of the focus
def is_focus (x y : ℝ) : Prop := (x = 0) ∧ (y = 1)

-- The theorem statement
theorem focus_of_parabola : 
  (∃ x y : ℝ, parabola x y ∧ is_focus x y) :=
sorry

end focus_of_parabola_l335_335857


namespace arrange_pencils_l335_335218

-- Definition to express the concept of pencil touching
def pencil_touches (a b : Type) : Prop := sorry

-- Assume we have six pencils represented as 6 distinct variables.
variables (A B C D E F : Type)

-- Main theorem statement
theorem arrange_pencils :
  ∃ (A B C D E F : Type), (pencil_touches A B) ∧ (pencil_touches A C) ∧ 
  (pencil_touches A D) ∧ (pencil_touches A E) ∧ (pencil_touches A F) ∧ 
  (pencil_touches B C) ∧ (pencil_touches B D) ∧ (pencil_touches B E) ∧ 
  (pencil_touches B F) ∧ (pencil_touches C D) ∧ (pencil_touches C E) ∧ 
  (pencil_touches C F) ∧ (pencil_touches D E) ∧ (pencil_touches D F) ∧ 
  (pencil_touches E F) :=
sorry

end arrange_pencils_l335_335218


namespace prism_volume_and_surface_area_l335_335450

-- Define the inclined triangular prism with edge length 2 and lateral edge forming 60-degree angles with the base sides
structure Prism :=
  (edge_length : ℝ)
  (angle : ℝ)
  (has_edges_eq : edge_length = 2)
  (angle_eq : angle = (60:ℝ))

noncomputable def volume (p : Prism) : ℝ :=
  -- The formula for the volume would be computed based on the given prism
  2 * Real.sqrt 2

noncomputable def surface_area (p : Prism) : ℝ :=
  -- The formula for the surface area would be computed based on the given prism
  4 + 6 * Real.sqrt 3

theorem prism_volume_and_surface_area (p : Prism) (h : p.has_edges_eq) (h2 : p.angle_eq) :
  volume p = 2 * Real.sqrt 2 ∧ surface_area p = 4 + 6 * Real.sqrt 3 :=
by
  sorry

end prism_volume_and_surface_area_l335_335450


namespace redistribution_l335_335639

/-
Given:
- b = (12 / 13) * a
- c = (2 / 3) * b
- Person C will contribute 9 dollars based on the amount each person spent

Prove:
- Person C gives 6 dollars to Person A.
- Person C gives 3 dollars to Person B.
-/

theorem redistribution (a b c : ℝ) (h1 : b = (12 / 13) * a) (h2 : c = (2 / 3) * b) : 
  ∃ (x y : ℝ), x + y = 9 ∧ x = 6 ∧ y = 3 :=
by
  sorry

end redistribution_l335_335639


namespace find_positive_m_l335_335476

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l335_335476


namespace find_q_l335_335961

def lcm : ℕ → ℕ → ℕ := Nat.lcm

theorem find_q (q : ℕ) (H : lcm (lcm 12 q) (lcm 18 24) = 144) : q = 8 :=
by
  sorry

end find_q_l335_335961


namespace relation_y1_y2_y3_l335_335269

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -x^2 + 2*x + c

variables (x1 x2 x3 : ℝ)
variables (y1 y2 y3 c : ℝ)
variables (P1 : x1 = -1)
variables (P2 : x2 = 3)
variables (P3 : x3 = 5)
variables (H1 : y1 = quadratic_function x1 c)
variables (H2 : y2 = quadratic_function x2 c)
variables (H3 : y3 = quadratic_function x3 c)

theorem relation_y1_y2_y3 (c : ℝ) :
  (y1 = y2) ∧ (y1 > y3) :=
sor_問題ry

end relation_y1_y2_y3_l335_335269


namespace generating_function_correct_l335_335298

noncomputable def generating_function (x : ℝ) : ℝ := (1 + x) ^ -2

theorem generating_function_correct :
  ∀ (n : ℕ), (n > 0) → 
  seq.nth (λ k, (-1) ^ k * (k + 1)) n =
  (generating_function^[n]) 0 :=
sorry

end generating_function_correct_l335_335298


namespace airline_passenger_capacity_l335_335813

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l335_335813


namespace projectile_first_reaches_35m_l335_335673

theorem projectile_first_reaches_35m (t : ℝ) :
  let y := -4.9 * t^2 + 29.5 * t in
  y = 35 ↔ t = 10 / 7 := 
by
  sorry

end projectile_first_reaches_35m_l335_335673


namespace find_all_polynomials_l335_335615

noncomputable def polynomial_of_degree_n (n : ℕ) :=
  {P : Polynomial ℝ // P.degree = n}

theorem find_all_polynomials (n : ℕ) (P : Polynomial ℝ) 
  (h1 : P.degree = n) 
  (h2 : ∀ x : ℝ, P.eval x = 0 → P.eval (x + 1) = 1) 
  (h3 : ∃ (r : fin n → ℝ), ∀ i j, i ≠ j → r i ≠ r j ∧ ∀ i, P.eval (r i) = 0) 
  : ∃ b : ℝ, n = 1 ∧ P = X + Polynomial.C b := 
sorry

end find_all_polynomials_l335_335615


namespace sum_value_of_solutions_l335_335728

theorem sum_value_of_solutions (z : ℂ) (hz : z^8 = 1) :
  ∑ (z : ℂ) in {z | z^8 = 1}, (1 / (abs (1 + z))^2) = -76 := 
sorry

end sum_value_of_solutions_l335_335728


namespace range_of_a_l335_335365

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, (a + real.cos θ)^2 + (2 * a - real.sin θ)^2 ≤ 4) →
  a ∈ set.Icc (-real.sqrt 5 / 5) (real.sqrt 5 / 5) :=
by
  sorry

end range_of_a_l335_335365


namespace problem_solution_l335_335613

noncomputable def q (x : ℝ) : ℝ :=
sorry

theorem problem_solution :
  (∃ q : ℝ → ℝ,
    (∀ x, q x = x^3 + a*x^2 + b*x + c ∧ q 1 = 10 ∧ q 2 = 20 ∧ q 3 = 30)) →
    q(0) + q(4) = 40 :=
sorry

end problem_solution_l335_335613


namespace OI_perp_AB_l335_335978

open EuclideanGeometry

noncomputable def angle_bisector (A B C : Point) (A1 B1 I O : Point) : Prop :=
  ∠C = 90 ∧ is_angle_bisector A A1 B C ∧ is_angle_bisector B B1 A C ∧ lies_on_circumcenter_of_triangle O C A1 B1 ∧
  intersection_of_bisectors_is_incenter A A1 B B1 I ∧
  (is_perpendicular (line_segment O I) (line_segment A B))

theorem OI_perp_AB
  (A B C A1 B1 I O : Point)
  (h : angle_bisector A B C A1 B1 I O) :
  is_perpendicular (line_segment O I) (line_segment A B) := 
begin
  sorry,
end

end OI_perp_AB_l335_335978


namespace danny_bottle_caps_l335_335442

theorem danny_bottle_caps 
  (wrappers_park : Nat := 46)
  (caps_park : Nat := 50)
  (wrappers_collection : Nat := 52)
  (more_caps_than_wrappers : Nat := 4)
  (h1 : caps_park = wrappers_park + more_caps_than_wrappers)
  (h2 : wrappers_collection = 52) : 
  (∃ initial_caps : Nat, initial_caps + caps_park = wrappers_collection + more_caps_than_wrappers) :=
by 
  use 6
  sorry

end danny_bottle_caps_l335_335442


namespace monopoly_houses_l335_335278

/-
Given the initial number of houses and the series of transactions, prove the final number of houses for each player.
-/
theorem monopoly_houses (start_sean : ℕ) (start_karen : ℕ) (start_mark : ℕ) (start_lucy : ℕ) 
  (trade_sean : ℕ) (buy_sean : ℕ)
  (trade_karen : ℕ) (buy_karen : ℕ) (gift_karen : ℕ) (trade_with_mark : ℕ)
  (buy_mark : ℕ) (sell_mark : ℕ) (trade_with_karen: ℕ)
  (trade_lucy : ℕ) (buy_lucy : ℕ) (upgrade_lucy : ℕ) : 
  start_sean = 45 → trade_sean = 15 → buy_sean = 18 →
  start_karen = 30 → trade_karen = 48 → buy_karen = 10  → gift_karen = 8 → trade_with_mark = 15 →
  start_mark = 55 → buy_mark = 12 → sell_mark = 25 → trade_with_karen = 15  →
  start_lucy = 35 → trade_lucy = 8 → buy_lucy = 6 → upgrade_lucy = 20 →
  let sean_final := start_sean - trade_sean + buy_sean in
  let karen_final := 0 + buy_karen + gift_karen + trade_with_mark in
  let mark_final := start_mark + buy_mark - sell_mark - trade_with_karen in
  let lucy_final := start_lucy - trade_lucy + buy_lucy - upgrade_lucy in
  sean_final = 48 ∧ karen_final = 33 ∧ mark_final = 27 ∧ lucy_final = 13 :=
by
  intros
  unfold sean_final karen_final mark_final lucy_final
  sorry

end monopoly_houses_l335_335278


namespace positive_value_m_l335_335479

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l335_335479


namespace complex_coordinates_l335_335208

theorem complex_coordinates (a b : ℂ) (i : ℂ) (h_i : i = complex.I) :
  (a, b) = (4, -5) ↔ (a + b * complex.I) = (5 + 4 * complex.I) / complex.I :=
by
  sorry

end complex_coordinates_l335_335208


namespace second_fragment_velocity_l335_335388

-- Definitions
def initial_vertical_velocity : ℝ := 20 -- Initial velocity \( v_0 \)
def time_at_explosion : ℝ := 1 -- Time \( t \)
def gravity : ℝ := 10 -- Acceleration due to gravity \( g \)
def horizontal_velocity_first_fragment : ℝ := 48 -- Horizontal velocity \( v_{x1} \)

-- Proof Statement
theorem second_fragment_velocity :
  let v_y := initial_vertical_velocity - gravity * time_at_explosion in
  sqrt ((-horizontal_velocity_first_fragment) ^ 2 + v_y ^ 2) = 52 :=
by
  sorry

end second_fragment_velocity_l335_335388


namespace stickers_after_transactions_l335_335252

theorem stickers_after_transactions :
  let initial := 20
  let step1 := initial + 12
  let step2 := step1 + 25
  let step3 := step2 + 30
  let step4 := step3 + 5
  let step5 := step4 - 5
  let step6 := step5 - 8
  let step7 := step6 - 12
  step7 = 67 :=
by
  rw [←add_assoc, add_comm 12, add_assoc, add_comm 25, add_assoc, add_comm 30, add_assoc]
  rw add_comm
  sorry

end stickers_after_transactions_l335_335252


namespace employees_percentage_6_years_or_more_l335_335977

-- Definition of the marks for each year range
def marks : List ℕ := [3, 6, 5, 4, 2, 2, 3, 2, 1, 1]

-- Number of employees per mark (as a variable)
variable (y : ℕ)

-- Calculation of total employees
def total_employees := List.sum marks * y

-- Calculation of employees with 6 years or more
def employees_6_years_or_more := (marks[6]! + marks[7]! + marks[8]! + marks[9]!) * y

-- Statement of the theorem
theorem employees_percentage_6_years_or_more :
  (employees_6_years_or_more y : ℚ) / (total_employees y) * 100 = 24.14 :=
by sorry

end employees_percentage_6_years_or_more_l335_335977


namespace sum_a_1_to_99_l335_335244

def a (n : ℕ) : ℝ := 1 / ((↑n + 1) * real.sqrt (↑n) + ↑n * real.sqrt (↑n + 1))

theorem sum_a_1_to_99 : (∑ n in (finset.range 99).map (finset.nat.cast_embedding), a n) = 9 / 10 := 
  sorry

end sum_a_1_to_99_l335_335244


namespace unique_two_scoop_sundaes_l335_335030

open Nat

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : (nat.choose n 2) = 28 :=
by 
  rw h 
  simp 
  sorry

end unique_two_scoop_sundaes_l335_335030


namespace remainder_of_difference_l335_335748

open Int

theorem remainder_of_difference (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 3) (h : a > b) : (a - b) % 6 = 5 :=
  sorry

end remainder_of_difference_l335_335748


namespace spots_allocation_l335_335797

theorem spots_allocation : ∃ (n: ℕ) (k: ℕ), n = 9 ∧ k = 7 ∧ nat.choose n k = 36 := 
by {
  have n := 9,
  have k := 2,
  refine ⟨n, k, rfl, rfl, nat.choose n k = 36⟩,
  -- proof omitted
  sorry
}

end spots_allocation_l335_335797


namespace remove_terms_sum_equals_one_l335_335652

theorem remove_terms_sum_equals_one :
  let seq := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let remove := [1/12, 1/15]
  (seq.sum - remove.sum) = 1 :=
by
  sorry

end remove_terms_sum_equals_one_l335_335652


namespace integral_sin_cos_eq_0_5_l335_335068

noncomputable def integral_sin_cos_identity : ℝ :=
  ∫ x in 0..(real.pi / 2), real.sin x * real.cos x

theorem integral_sin_cos_eq_0_5 :
  integral_sin_cos_identity = 0.5 :=
by
  sorry

end integral_sin_cos_eq_0_5_l335_335068


namespace mgp_inequality_l335_335614

theorem mgp_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (1 / Real.sqrt (1 / 2 + a + a * b + a * b * c) + 
   1 / Real.sqrt (1 / 2 + b + b * c + b * c * d) + 
   1 / Real.sqrt (1 / 2 + c + c * d + c * d * a) + 
   1 / Real.sqrt (1 / 2 + d + d * a + d * a * b)) 
  ≥ Real.sqrt 2 := 
sorry

end mgp_inequality_l335_335614


namespace area_of_rearranged_arcs_l335_335758

-- Define the conditions.
variables (O : Point) (A B : Point)
hypothesis (radius_one : distance(O, A) = 1)
hypothesis (angle_right : ∠AOB = π / 2)

-- Define the theorem for the area of the rearranged shape.
theorem area_of_rearranged_arcs : 
  (let AB := distance(A, B) in
   let side_length := AB in
   let area_square := side_length^2 in
   area_square = 2) := 
by 
-- state each necessary logical step here, culminating in the proof.
sorry

end area_of_rearranged_arcs_l335_335758


namespace log_impossible_l335_335096

theorem log_impossible (log_5 : ℝ) (log_7 : ℝ) (h1 : log_5 = 0.6990) (h2 : log_7 = 0.8451) :
  ¬ ∃ (log_27 : ℝ), (log_27 = 3 * log_10 3) ∧ (log_10 3 = some_expression log_5 log_7) ∨ 
  ¬ ∃ (log_21 : ℝ), (log_21 = log_10 3 + log_7) ∧ (log_10 3 = some_expression log_5 log_7) :=
sorry

end log_impossible_l335_335096


namespace students_speaking_both_languages_l335_335411

theorem students_speaking_both_languages (total_students english_speakers japanese_speakers neither_speakers both_speakers: ℕ) 
(h1 : total_students = 50)
(h2 : english_speakers = 36)
(h3 : japanese_speakers = 20)
(h4 : neither_speakers = 8)
(h5 : total_students - neither_speakers = english_speakers - both_speakers + japanese_speakers - both_speakers - both_speakers + both_speakers) :
both_speakers = 14 :=
by 
  have h6 : total_students - neither_speakers = 42, from (h1 ▸ h4 ▸ rfl)
  rw h6 at h5 
  have h7 : 42 = english_speakers - both_speakers + japanese_speakers - both_speakers, from h5
  rw [h2, h3] at h7 
  have h8 : 42 = 36 + 20 - both_speakers, by linarith
  linarith
  sorry

end students_speaking_both_languages_l335_335411


namespace radii_ratio_interval_l335_335165

-- Define the trapezoid structure
structure Trapezoid :=
  (A B C D : Point)
  (base : segment C D)
  (parallelogram : parallelogram A B C D)

-- Define conditions
axiom trapezoid_data : Trapezoid
axiom E_on_BC : E ∈ line (B, C)
axiom E_cyclic_ACD : cyclic [A, C, D, E] -- E, A, C, D lie on the same circle
axiom ABCD_cyclic_BCA : cyclic [A, B, C] -- A, B, C lie on the same circle and tangent to CD

-- Lengths definitions
noncomputable def length_AB : ℝ := 12
noncomputable def ratio_BE_EC : ℝ := 4 / 5

-- Theorem to be proved
theorem radii_ratio_interval (AB : line × ℝ := 12)
  (BE_EC_ratio : (line → ℝ := 4 / 5)
  (BC len: ℝ) (AB : line * ℝ) : 
  let first_circle_radius : ℝ := calculate_radius_first_circle C E
  let second_circle_radius : ℝ := calculate_radius_second_circle A B :=
  \frac{first_circle_radius}{second_circle_radius} ∈ \left( \frac{2}{3}, \frac{4}{3} \right) := sorry

end radii_ratio_interval_l335_335165


namespace train_cross_time_l335_335400

noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kph : ℝ := 45
noncomputable def total_length : ℝ := 375

noncomputable def speed_mps := train_speed_kph * 1000 / 3600
noncomputable def distance := train_length + total_length

theorem train_cross_time : (distance / speed_mps) = 30 := by
  sorry

end train_cross_time_l335_335400


namespace row_col_sum_nonzero_l335_335502

def matrix_19x19 (A : matrix (fin 19) (fin 19) ℤ) := ∀ i j, A i j = 1 ∨ A i j = -1

def row_product (A : matrix (fin 19) (fin 19) ℤ) (i : fin 19) : ℤ :=
  ∏ j, A i j

def col_product (A : matrix (fin 19) (fin 19) ℤ) (j : fin 19) : ℤ :=
  ∏ i, A i j

theorem row_col_sum_nonzero (A : matrix (fin 19) (fin 19) ℤ)
  (h : matrix_19x19 A) :
  (∑ i, row_product A i + col_product A i) ≠ 0 := sorry

end row_col_sum_nonzero_l335_335502


namespace area_of_paper_l335_335012

-- Define the variables and conditions
variable (L W : ℝ)
variable (h1 : 2 * L + 4 * W = 34)
variable (h2 : 4 * L + 2 * W = 38)

-- Statement to prove
theorem area_of_paper : L * W = 35 := 
by
  sorry

end area_of_paper_l335_335012


namespace two_digit_numbers_no_repeats_from_123_l335_335948

theorem two_digit_numbers_no_repeats_from_123 : 
  {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (∃ a b, a ≠ b ∧ (a ∈ {1, 2, 3}) ∧ (b ∈ {1, 2, 3}) ∧ n = 10 * a + b)}.toFinset.card = 6 := by
  sorry

end two_digit_numbers_no_repeats_from_123_l335_335948


namespace isosceles_trapezoid_ratio_l335_335032

theorem isosceles_trapezoid_ratio (a b d_E d_G : ℝ) (h1 : a > b)
  (h2 : (1/2) * b * d_G = 3) (h3 : (1/2) * a * d_E = 7)
  (h4 : (1/2) * (a + b) * (d_E + d_G) = 24) :
  (a / b) = 7 / 3 :=
sorry

end isosceles_trapezoid_ratio_l335_335032


namespace exists_int_solution_l335_335219

theorem exists_int_solution (x : ℤ) :
  x ≡ 1 [MOD 6] ∧ x ≡ 9 [MOD 14] ∧ x ≡ 7 [MOD 15] ↔ x ≡ 37 [MOD 210] :=
by
  sorry

end exists_int_solution_l335_335219


namespace tap_filling_time_l335_335381

theorem tap_filling_time
  (T : ℝ)
  (H1 : 10 > 0) -- Second tap can empty the cistern in 10 hours
  (H2 : T > 0)  -- First tap's time must be positive
  (H3 : (1 / T) - (1 / 10) = (3 / 20))  -- Both taps together fill the cistern in 6.666... hours
  : T = 4 := sorry

end tap_filling_time_l335_335381


namespace problem_I_problem_II_l335_335539

noncomputable def f (x θ : ℝ) : ℝ := (sqrt 2 * sin (2 * x + θ))

def problem_conditions (θ : ℝ) (hθ : |θ| < π / 2) :=
  θ = π / 6

noncomputable def analyt_exp_f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x + π / 6)

def monot_decrease_interval (k : ℤ) : ℝ × ℝ := (k * π + π / 6, k * π + 2 * π / 3)

/- Problem I -/

theorem problem_I (x : ℝ) (θ : ℝ) (hθ : |θ| < π / 2) (h_symm : problem_conditions θ hθ) :
  f x θ = analyt_exp_f x ∧ ∀ k ∈ ℤ, analyt_exp_f x ∈ Icc (monot_decrease_interval k).1 (monot_decrease_interval k).2 :=
sorry

/- Problem II -/

noncomputable def side_a : ℝ := sqrt 7
noncomputable def circum_radius : ℝ := sqrt 7
noncomputable def circumcircle_area : ℝ := 7 * π

theorem problem_II (b c : ℝ) (A : ℝ) (hA : f A A = sqrt 2)
  (hb : b = 5) (hc : c = 2 * sqrt 3) :
  A = π / 6 ∧ side_a = sqrt (b^2 + c^2 - 2 * b * c * cos A) ∧ circum_radius = sqrt 7 ∧ circumcircle_area = 7 * π :=
sorry

end problem_I_problem_II_l335_335539


namespace count_odd_divisors_l335_335946

theorem count_odd_divisors (n : ℕ) (h : n < 100) : 
  {k : ℕ | k < 100 ∧ ∃ m : ℕ, k = m * m}.card = 9 :=
by 
sorry

end count_odd_divisors_l335_335946


namespace white_balls_count_l335_335583

theorem white_balls_count (a : ℕ) (h : 3 / (3 + a) = 3 / 7) : a = 4 :=
by sorry

end white_balls_count_l335_335583


namespace find_t_u_l335_335940

variables (V : Type) [InnerProductSpace ℝ V]
variables (a b p : V)
variable (h : ∥p - b∥ = 3 * ∥p - a∥)

theorem find_t_u : ∃ (t u : ℝ), t = 9 / 8 ∧ u = -1 / 8 :=
by
  use 9 / 8, -1 / 8
  sorry

end find_t_u_l335_335940


namespace starting_number_of_odd_integers_with_odd_factors_l335_335711

theorem starting_number_of_odd_integers_with_odd_factors (h : ∀ n, 
  n ∈ {x : ℕ | x ≤ 100 ∧ x % 2 = 1 ∧ (∃ k : ℕ, k^2 = x)} → n = 1 ∨ 
  n = 9 ∨ n = 25 ∨ n = 49 ∨ n = 81) : 
  ∃ m, m <= 100 ∧ m % 2 = 1 ∧ 
  (∀ a b, a ∈ {m, (m+1)..100} ∧ b ∈ {m, (m+1)..100} ∧ 
    a ≠ b → a ≠ b) ∧ (m = 1) :=
sorry

end starting_number_of_odd_integers_with_odd_factors_l335_335711


namespace intersection_M_N_l335_335136

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l335_335136


namespace probability_at_least_one_hardcover_l335_335628

-- Definitions for given conditions
def total_textbooks : ℕ := 15
def hardcover_textbooks : ℕ := 5
def selected_textbooks : ℕ := 3

-- Lean expression of the math proof problem
theorem probability_at_least_one_hardcover :
  ∃ (total hardcover selected : ℕ), total = total_textbooks ∧ hardcover = hardcover_textbooks ∧ selected = selected_textbooks ∧ 
  (∃ (P : ℚ), P = (1 - (nat.choose (total - hardcover) selected) / (nat.choose total selected)) ∧ P = (67 / 91)) :=
begin
  -- Define the variables according to the problem statement
  use [total_textbooks, hardcover_textbooks, selected_textbooks],
  split, refl,
  split, refl,
  split, refl,
  -- Provide the required probability
  use (67 / 91),
  split,
  { -- Compute the probability expression
    sorry },
  { -- Final comparison with the solution
    refl },
end

end probability_at_least_one_hardcover_l335_335628


namespace sum_of_erased_odd_numbers_l335_335701

theorem sum_of_erased_odd_numbers (k1 k2 : ℕ) (n1 n2 : ℕ) (total_sum : ℕ) :
  (1 + 3 + 5 + ... + (2*n1-1) = 961) ∧ (1 + 3 + 5 + ... + (2*n2-1) = 1001) ∧ 
  ((1 + 3 + 5 + ... + (2*k1-1)) + (1 + 3 + 5 + ... + (2*k2-1)) + total_sum = (1 + 3 + 5 + ... + (2*(k1+n1 + 1 + 1)*2))
→ total_sum = 154 :=
 by sorry

end sum_of_erased_odd_numbers_l335_335701


namespace positive_value_m_l335_335478

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end positive_value_m_l335_335478


namespace monotonically_increasing_function_l335_335024

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ {x y : ℝ}, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem monotonically_increasing_function :
  let A := λ x : ℝ, |x| + 1
  let B := λ x : ℝ, 1 / x
  let C := λ x : ℝ, -x^2 + 1
  let D := λ x : ℝ, -x * |x|
  is_monotonically_increasing A (Set.Ioi 0) ∧
  ¬ is_monotonically_increasing B (Set.Ioi 0) ∧
  ¬ is_monotonically_increasing C (Set.Ioi 0) ∧
  ¬ is_monotonically_increasing D (Set.Ioi 0) :=
by
  sorry

end monotonically_increasing_function_l335_335024


namespace magnitude_of_z_l335_335148

theorem magnitude_of_z 
  (z : ℂ)
  (h : (complex.i - 2) * z = 4 + 3 * complex.i) :
  |z| = real.sqrt 5 :=
sorry

end magnitude_of_z_l335_335148


namespace samuel_has_five_birds_l335_335376

theorem samuel_has_five_birds
  (birds_berries_per_day : ℕ)
  (total_berries_in_4_days : ℕ)
  (n_birds : ℕ)
  (h1 : birds_berries_per_day = 7)
  (h2 : total_berries_in_4_days = 140)
  (h3 : n_birds * birds_berries_per_day * 4 = total_berries_in_4_days) :
  n_birds = 5 := by
  sorry

end samuel_has_five_birds_l335_335376


namespace interest_difference_correct_l335_335672

-- Define the basic parameters and constants
def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10
def interest1 : ℝ := principal * rate * time1
def interest2 : ℝ := principal * rate * time2
def difference : ℝ := 143.998

-- Theorem statement: The difference between the interests is approximately Rs. 143.998
theorem interest_difference_correct :
  interest2 - interest1 = difference := sorry

end interest_difference_correct_l335_335672


namespace area_of_park_l335_335314

theorem area_of_park (x : ℕ) (rate_per_meter : ℝ) (total_cost : ℝ)
  (ratio_len_wid : ℕ × ℕ)
  (h_ratio : ratio_len_wid = (3, 2))
  (h_cost : total_cost = 140)
  (unit_rate : rate_per_meter = 0.50)
  (h_perimeter : 10 * x * rate_per_meter = total_cost) :
  6 * x^2 = 4704 :=
by
  sorry

end area_of_park_l335_335314


namespace combination_8_choose_2_l335_335028

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end combination_8_choose_2_l335_335028


namespace remaining_students_correct_l335_335702

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end remaining_students_correct_l335_335702


namespace largest_fraction_inequality_l335_335514

variable {a b c d e : ℝ}
variable {h₀ : 0 < a} {h₁ : a < b} {h₂ : b < c} {h₃ : c < d} {h₄ : d < e}

theorem largest_fraction_inequality 
  (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) :
  max (max (max (max (a+b+e)/(c+d) (a+c)/(b+d+e)) (b+c+e)/(a+d)) (b+d)/(a+c+e)) (c+d+e)/(a+b) = (c+d+e)/(a+b) := 
by
  sorry

end largest_fraction_inequality_l335_335514


namespace proof_problem_l335_335284

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem_l335_335284


namespace number_of_balanced_pairs_l335_335242

def M : Finset ℕ := Finset.range 18 \ {0}

def balanced_pair (a b c d : ℕ) : Prop := 
  a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b) % 17 = (c + d) % 17

theorem number_of_balanced_pairs : 476 = 
  card {p : Finset (ℕ × ℕ × ℕ × ℕ) | 
    ∃ (a b c d : ℕ), balanced_pair a b c d ∧ 
    (a, b, c, d) ∈ p} :=
by
  sorry
 
end number_of_balanced_pairs_l335_335242


namespace older_grandchild_pancakes_eaten_l335_335171

theorem older_grandchild_pancakes_eaten (initial_pancakes : ℕ) (remaining_pancakes : ℕ)
  (younger_eat_per_cycle : ℕ) (older_eat_per_cycle : ℕ) (bake_per_cycle : ℕ)
  (n : ℕ) 
  (h_initial : initial_pancakes = 19)
  (h_remaining : remaining_pancakes = 11)
  (h_younger_eat : younger_eat_per_cycle = 1)
  (h_older_eat : older_eat_per_cycle = 3)
  (h_bake : bake_per_cycle = 2)
  (h_reduction : initial_pancakes - remaining_pancakes = n * (younger_eat_per_cycle + older_eat_per_cycle - bake_per_cycle)) :
  older_eat_per_cycle * n = 12 :=
begin
  sorry
end

end older_grandchild_pancakes_eaten_l335_335171


namespace factorize_problem1_factorize_problem2_l335_335463

theorem factorize_problem1 (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

theorem factorize_problem2 (x y : ℝ) : 
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) :=
by sorry

end factorize_problem1_factorize_problem2_l335_335463


namespace complex_modulus_problem_l335_335557

theorem complex_modulus_problem
  (z : ℂ)
  (h : (3 - 4 * complex.I) * z = complex.abs (4 + 3 * complex.I)) :
  complex.abs z = 1 :=
by
  sorry

end complex_modulus_problem_l335_335557


namespace remaining_students_correct_l335_335703

def initial_groups : Nat := 3
def students_per_group : Nat := 8
def students_left_early : Nat := 2

def total_students (groups students_per_group : Nat) : Nat := groups * students_per_group

def remaining_students (total students_left_early : Nat) : Nat := total - students_left_early

theorem remaining_students_correct :
  remaining_students (total_students initial_groups students_per_group) students_left_early = 22 := by
  sorry

end remaining_students_correct_l335_335703


namespace car_travel_time_l335_335632

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l335_335632


namespace intersection_M_N_l335_335117

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l335_335117


namespace speed_of_second_fragment_l335_335386

-- Definition of the initial conditions
def initial_velocity : ℝ := 20 -- m/s
def time_after_flight : ℝ := 1  -- s
def gravity : ℝ := 10 -- m/s^2
def horizontal_velocity_first_fragment : ℝ := 48 -- m/s

-- Definition for vertical velocity after 1 second
def vertical_velocity (v0 : ℝ) (g : ℝ) (t : ℝ) : ℝ :=
  v0 - g * t

-- Vertical velocity of the second fragment is the same as that of the first fragment
def vertical_velocity_second_fragment := vertical_velocity initial_velocity gravity time_after_flight

-- Horizontal velocity of the second fragment
def horizontal_velocity_second_fragment := -horizontal_velocity_first_fragment

-- Magnitude of the velocity of the second fragment using Pythagorean theorem
def velocity_magnitude (vx vy : ℝ) : ℝ :=
  Real.sqrt (vx * vx + vy * vy)

-- Main theorem statement
theorem speed_of_second_fragment :
  velocity_magnitude horizontal_velocity_second_fragment vertical_velocity_second_fragment = 49.03 :=
by 
  sorry -- Proof omitted

end speed_of_second_fragment_l335_335386


namespace Jenny_Kenny_meet_l335_335601

noncomputable def conditions (Jenny_path : ℕ → ℝ → Prop) (Kenny_path : ℕ → ℝ → Prop) :=
  ∀ t : ℝ, ∃ (x_j y_j x_k y_k : ℝ), 
    Jenny_path t (x_j, y_j) ∧ Kenny_path t (x_k, y_k) ∧
    |x_j - x_k| = 300 ∧
    y_j = 150 ∧ y_k = -150 ∧
    (x_j = -75 + 2 * t) ∧
    (x_k = -75 + 4 * t) ∧
    (x_j^2 + y_j^2 = 75^2) ∧ (x_k^2 + y_k^2 = 75^2)

theorem Jenny_Kenny_meet (Jenny_path Kenny_path : ℕ → ℝ → Prop) (t : ℝ) :
  conditions Jenny_path Kenny_path →
  ∃ t : ℝ, t = 48 ∧ num_denom_sum t = 49 :=
sorry

end Jenny_Kenny_meet_l335_335601


namespace set_intersection_l335_335126

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l335_335126


namespace ascending_order_l335_335098

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb1 : -1 < b) (hb2 : b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end ascending_order_l335_335098


namespace number_of_real_zeros_l335_335474

def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

theorem number_of_real_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_real_zeros_l335_335474


namespace tangent_line_area_condition_l335_335189

theorem tangent_line_area_condition (a : ℝ) (h : 0 < a ∨ a < 0) :
  let f := λ x : ℝ, x^(-2)
  let f' := λ x : ℝ, -2 * x^(-3)
  let tangent_line := λ x : ℝ, (f' a) * (x - a) + f a
  -- Intersection points with axes
  let x_intercept := (a / 2)
  let y_intercept := (1 / a^2)
  -- Calculate area of the triangle
  (1 / 2) * |x_intercept| * |y_intercept| = 3 →
  a = 3 / 4 ∨ a = -3 / 4 :=
sorry

end tangent_line_area_condition_l335_335189


namespace shaded_area_correct_l335_335346

noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

def PQRS_side_length : ℝ := 6
def area_unshaded : ℝ := area_triangle 3 3

def area_shaded : ℝ :=
  (area_square 2) + (3 * 3) + (area_square 2) - area_unshaded

def shaded_percentage : ℝ :=
  (area_shaded / (area_square PQRS_side_length)) * 100

theorem shaded_area_correct :
  shaded_percentage = 34.72 := by
  sorry

end shaded_area_correct_l335_335346


namespace parallelepiped_ratio_l335_335786

variables {V : Type*} [InnerProductSpace ℝ V]

noncomputable def PSQRPTQS_div_PQPRPS (a b c : V) (h : ⟪a, b⟫ = 0) : ℝ :=
  (‖c‖^2 + ‖a - b‖^2 + ‖a + b + c‖^2 + ‖c - a‖^2) / (‖a‖^2 + ‖b‖^2 + ‖c‖^2)

theorem parallelepiped_ratio (a b c : V)
  (h : ⟪a, b⟫ = 0) :
  PSQRPTQS_div_PQPRPS a b c h =
  2 + (2 * ⟪b, c⟫) / (‖a‖^2 + ‖b‖^2 + ‖c‖^2) :=
begin
  sorry
end

end parallelepiped_ratio_l335_335786


namespace triangle_is_right_triangle_l335_335191

variables {a b c : ℝ} {C : ℝ}

-- We assume the basic triangle properties hold
axiom triangle_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_C : 0 < C ∧ C < π -- Angle in radians for a valid triangle

-- Given condition for the problem
axiom cosine_condition : a = b * Real.cos C

-- Conclusion to prove
theorem triangle_is_right_triangle : a = b * Real.cos C → a^2 + c^2 = b^2 := by
  intro h_cos
  -- Proof is omitted
  sorry

end triangle_is_right_triangle_l335_335191


namespace problem1_problem2_l335_335428

-- Problem 1
theorem problem1 (a : ℝ) : 3 * a ^ 2 - 2 * a + 1 + (3 * a - a ^ 2 + 2) = 2 * a ^ 2 + a + 3 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : x - 2 * (x - 3 / 2 * y) + 3 * (x - x * y) = 2 * x + 3 * y - 3 * x * y :=
by
  sorry

end problem1_problem2_l335_335428


namespace number_of_nonempty_sets_l335_335697

def is_subset (S : set ℕ) : Prop := S ⊆ {1, 2, 3, 4, 5}
def satisfies_condition (S : set ℕ) : Prop := ∀ a ∈ {1, 2, 3, 4, 5}, a ∈ S → 6 - a ∈ S

theorem number_of_nonempty_sets : 
  ∃ (S' : finset (set ℕ)), (∀ S ∈ S', is_subset S ∧ satisfies_condition S) ∧
                                 S'.card = 15 := 
sorry

end number_of_nonempty_sets_l335_335697


namespace check_good_numbers_l335_335755

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), ∀ k : Fin n, ∃ m : ℕ, k.val + a k + 1 = m^2

theorem check_good_numbers :
  ¬ is_good_number 11 ∧ is_good_number 13 ∧ is_good_number 15 ∧ 
  is_good_number 17 ∧ is_good_number 19 := 
by {
  split,
  { -- Proof that 11 is not a "good number"
    sorry },
  split,
  { -- Proof that 13 is a "good number"
    sorry },
  split,
  { -- Proof that 15 is a "good number"
    sorry },
  split,
  { -- Proof that 17 is a "good number"
    sorry },
  { -- Proof that 19 is a "good number"
    sorry }
}

end check_good_numbers_l335_335755


namespace eq_total_shaded_area_l335_335900

noncomputable def shaded_area_of_equilateral_triangle (n : ℕ) : ℝ :=
  let area_initial := (sqrt 3 / 4) * 8^2
  in area_initial * (1 / 3) * ((1 / 3) ^ n)

theorem eq_total_shaded_area : 
  let total_shaded_area := finset.sum (finset.range 100) shaded_area_of_equilateral_triangle
  (total_shaded_area : ℝ) = 8 * sqrt 3 :=
by
  sorry

end eq_total_shaded_area_l335_335900


namespace find_cos_alpha_l335_335143

noncomputable def alpha := Real.angle 

def condition1 (α : Real.angle) : Prop :=
  α > Real.pi ∧ α < 3 * Real.pi / 2

def condition2 (α : Real.angle) : Prop :=
  Real.tan α = 2

theorem find_cos_alpha (α : Real.angle) (h1 : condition1 α) (h2 : condition2 α) :
  Real.cos α = - (Real.sqrt 5) / 5 :=
sorry

end find_cos_alpha_l335_335143


namespace second_train_speed_l335_335760

-- Conditions
def L1 : ℝ := 280
def V1 : ℝ := 120
def L2 : ℝ := 220.04
def T : ℝ := 9

-- Relative speed conversion factor from kmph to m/s
def kmph_to_mps : ℝ := 5 / 18

-- Correct answer
def V2_correct : ℝ := 80.016

-- Problem statement
theorem second_train_speed :
  let V2 := V2_correct in
  (L1 + L2) = (V1 + V2) * kmph_to_mps * T :=
by
  sorry

end second_train_speed_l335_335760


namespace intersection_M_N_l335_335135

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l335_335135


namespace accurate_mass_l335_335761

variable (m1 m2 a b x : Real) -- Declare the variables

theorem accurate_mass (h1 : a * x = b * m1) (h2 : b * x = a * m2) : x = Real.sqrt (m1 * m2) := by
  -- We will prove the statement later
  sorry

end accurate_mass_l335_335761


namespace force_on_dam_correct_l335_335752

noncomputable def compute_force_on_dam (ρ g a b h : ℝ) : ℝ :=
  let pressure_at_depth (x : ℝ) := ρ * g * x
  let width_at_depth (x : ℝ) := b - x * (b - a) / h
  ∫ x in 0..h, pressure_at_depth x * width_at_depth x

theorem force_on_dam_correct :
  compute_force_on_dam 1000 10 7.2 12.0 5.0 = 1100000 := by
  sorry

end force_on_dam_correct_l335_335752


namespace grasshopper_jump_periodic_l335_335337

theorem grasshopper_jump_periodic (gamma : ℝ) (h_condition : ∃ l1 l2 : ℝ, (∃ x y : ℝ, l1 ≠ l2 ∧ (∀ x, 0 ≤ x) ∧ (∀ y, 0 ≤ y) ∧ ∃ gamma : ℝ, 0 < gamma < π ∧ ℝ.angle l1 l2 = gamma) ∧ (∀ jump_length : ℝ, jump_length = 1)) :
    (∃ T : ℕ, ∀ n : ℕ, ∃ m : ℕ, n = m + T ↔ γ / π ∈ ℚ) ↔ γ / π ∈ ℚ :=
by
  sorry

end grasshopper_jump_periodic_l335_335337


namespace rectangle_area_eq_140_l335_335304

theorem rectangle_area_eq_140 :
  let side_square := Real.sqrt 1225 in
  let radius_circle := side_square in
  let length_rectangle := (2 / 5) * radius_circle in
  let breadth_rectangle := 10 in
  length_rectangle * breadth_rectangle = 140 :=
by
  let side_square := Real.sqrt 1225
  let radius_circle := side_square
  let length_rectangle := (2 / 5) * radius_circle
  let breadth_rectangle := 10
  calc
    length_rectangle * breadth_rectangle = (2 / 5) * radius_circle * breadth_rectangle : by rw mul_assoc
                                      ... = (2 / 5) * 35 * 10 : by rw radius_circle
                                      ... = 140 : by norm_num

end rectangle_area_eq_140_l335_335304


namespace problem_I_simplify_problem_II_simplify_l335_335037

-- Problem (I)
theorem problem_I_simplify : 
  (0.027)^(1/3) - (6 + 1/4)^(1/2) + 256^(3/4) + (2 * Real.sqrt 2)^(2/3) - 3^(-1) + Real.pi^(0) = 967/15 :=
by
  sorry

-- Problem (II)
theorem problem_II_simplify (x : ℝ) : 
  (x - 1) / (x^(2/3) + x^(1/3) + 1) + (x + 1) / (x^(1/3) + 1) - (x - x^(1/3)) / (x^(1/3) - 1) = -x^(1/3) :=
by
  sorry

end problem_I_simplify_problem_II_simplify_l335_335037


namespace non_neg_int_solutions_l335_335055

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end non_neg_int_solutions_l335_335055


namespace ornamental_rings_remaining_l335_335455

theorem ornamental_rings_remaining :
  let r := 100 in
  let T := 200 + r in
  let rings_after_sale := T - (3 * T / 4) in
  let rings_after_mothers_purchase := rings_after_sale + 300 in
  rings_after_mothers_purchase - 150 = 225 :=
by
  sorry

end ornamental_rings_remaining_l335_335455


namespace sum_of_products_eq_131_l335_335319

theorem sum_of_products_eq_131 (a b c : ℝ) 
    (h1 : a^2 + b^2 + c^2 = 222)
    (h2 : a + b + c = 22) : 
    a * b + b * c + c * a = 131 :=
by
  sorry

end sum_of_products_eq_131_l335_335319


namespace problem1_problem2_l335_335097

-- Define the angle α and the given condition tan α = 3
variable (α : ℝ)

axiom tan_alpha : Real.tan α = 3

-- Lean statement for the first question
theorem problem1 : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 :=
by
  have h : Real.tan α = 3 := tan_alpha
  sorry

-- Lean statement for the second question
theorem problem2 : 1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = 2 :=
by
  have h : Real.tan α = 3 := tan_alpha
  sorry

end problem1_problem2_l335_335097


namespace minimum_value_l335_335954

theorem minimum_value (x : ℝ) (hx : 0 < x) : ∃ y, (y = x + 4 / (x + 1)) ∧ (∀ z, (x > 0 → z = x + 4 / (x + 1)) → 3 ≤ z) := sorry

end minimum_value_l335_335954


namespace determine_g_at_3_l335_335560

-- Definitions
def f (x : ℝ) := 4^x + 2^(x+1)
def symmetric_about_y_equals_x (f g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f x = y ↔ g y = x

-- Theorem statement
theorem determine_g_at_3 (g : ℝ → ℝ)
  (h_symm : symmetric_about_y_equals_x f g) :
  g 3 = 0 :=
sorry

end determine_g_at_3_l335_335560


namespace weight_combination_property_l335_335109

/--
Given a set of weights {1, 3, 9, 27}, prove that:
1. The heaviest object that can be weighed using these weights is 40 lb.
2. There are 40 distinct weights that can be measured using any combination of these weights.
-/
theorem weight_combination_property :
  let weights : List ℕ := [1, 3, 9, 27] in
  list.sum weights = 40 ∧ (∀ k, 1 ≤ k ∧ k ≤ 40 → ∃ ws, list.sum ws = k ∧ ws ⊆ weights) := by
  let weights : List ℕ := [1, 3, 9, 27]
  have h_sum : list.sum weights = 40 := by
    -- Putting in the summation proof as a placeholder.
    sorry
  have h_distinct : ∀ k, 1 ≤ k ∧ k ≤ 40 → ∃ ws, list.sum ws = k ∧ ws ⊆ weights := by
    -- Putting in the proof for distinct weights as a placeholder.
    sorry
  exact ⟨h_sum, h_distinct⟩

end weight_combination_property_l335_335109


namespace johns_drive_distance_l335_335603

/-- John's driving problem -/
theorem johns_drive_distance
  (d t : ℝ)
  (h1 : d = 25 * (t + 1.5))
  (h2 : d = 25 + 45 * (t - 1.25)) :
  d = 123.4375 := 
sorry

end johns_drive_distance_l335_335603


namespace current_price_after_increase_and_decrease_l335_335379

-- Define constants and conditions
def initial_price_RAM : ℝ := 50
def percent_increase : ℝ := 0.30
def percent_decrease : ℝ := 0.20

-- Define intermediate and final values based on conditions
def increased_price_RAM : ℝ := initial_price_RAM * (1 + percent_increase)
def final_price_RAM : ℝ := increased_price_RAM * (1 - percent_decrease)

-- Theorem stating the final result
theorem current_price_after_increase_and_decrease 
  (init_price : ℝ) 
  (inc : ℝ) 
  (dec : ℝ) 
  (final_price : ℝ) :
  init_price = 50 ∧ inc = 0.30 ∧ dec = 0.20 → final_price = 52 := 
  sorry

end current_price_after_increase_and_decrease_l335_335379


namespace linear_function_quadrants_l335_335001

theorem linear_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = (k + 1) * x + k - 2 → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ (-1 < k ∧ k < 2) := 
sorry

end linear_function_quadrants_l335_335001


namespace sequence_nonzero_l335_335100

noncomputable def a : ℕ → ℤ 
| 0 := 1
| 1 := 2
| (n + 2) := if ((a n) * (a (n + 1))).even then 5 * (a (n + 1)) - 3 * (a n) else (a (n + 1)) - (a n)

theorem sequence_nonzero (n : ℕ) : a n ≠ 0 := sorry

end sequence_nonzero_l335_335100


namespace sum_of_angles_is_540_l335_335421

variables (angle1 angle2 angle3 angle4 angle5 angle6 angle7 : ℝ)

theorem sum_of_angles_is_540
  (h : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540 :=
sorry

end sum_of_angles_is_540_l335_335421


namespace geometric_sequence_sum_l335_335515

theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : q ≠ 1)
    (hS2 : (a₁ * (1 - q^2)) / (1 - q) = 1)
    (hS4 : (a₁ * (1 - q^4)) / (1 - q) = 3) :
    (a₁ * (1 - q^8)) / (1 - q) = 15 := 
by
  sorry

end geometric_sequence_sum_l335_335515


namespace all_statements_are_incorrect_l335_335025

-- Definitions based on the conditions
def is_frustum (shape : Type) := 
  ∀ base section : set shape, parallel base section → 
  (lateral_edges_converge shape base section)

def statement1 (pyramid : Type) : Prop :=
  ∀ (plane : set pyramid) (base section : set pyramid),
  base ∩ section = ∅ → is_frustum (base ∪ section)

def statement2 (polyhedron : Type) : Prop :=
  ∀ (base1 base2 : set polyhedron) (faces : list (set polyhedron)),
  parallel base1 base2 →
  (∀ face ∈ faces, (is_trapezoid face)) → 
  is_frustum (base1 ∪ base2 ∪ ⋃₀ faces)

def statement3 (hexahedron : Type) : Prop :=
  ∀ (face1 face2 : set hexahedron) (faces : list (set hexahedron)),
  parallel face1 face2 →
  (length faces = 4) →
  (∀ face ∈ faces, is_isosceles_trapezoid face) → 
  is_frustum (face1 ∪ face2 ∪ ⋃₀ faces)

-- Proof that the given conditions don't define a frustum
theorem all_statements_are_incorrect : 
  ¬ statement1 pyramid ∧ ¬ statement2 polyhedron ∧ ¬ statement3 hexahedron := 
by
  sorry

end all_statements_are_incorrect_l335_335025


namespace brick_selection_path_count_l335_335587

-- Let dp[i][j] denote the number of ways to reach row i, column j
def dp : List (List Nat) :=
  let dp1 := [1, 1, 1, 1, 1]
  let dp2 := [2, 3, 3, 3, 2]
  let dp3 := [5, 8, 9, 8, 5]
  let dp4 := [13, 22, 25, 22, 13]
  [dp1, dp2, dp3, dp4]
  
-- Function to calculate the last row based on previous row
def next_row (prev : List Nat) : List Nat :=
  [prev[0] + prev[1],
   prev[0] + prev[1] + prev[2],
   prev[1] + prev[2] + prev[3],
   prev[2] + prev[3] + prev[4],
   prev[3] + prev[4]]

-- Calculate the 5th row
def dp5 := next_row dp[3]

-- Function to sum up the last row to get the total number of valid paths
def total_paths : Nat :=
  dp5.foldr (+) 0

theorem brick_selection_path_count : total_paths = 259 := by
  sorry

end brick_selection_path_count_l335_335587


namespace total_emails_vacation_l335_335764

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l335_335764


namespace infinitely_many_primes_with_f_x_congr_0_l335_335608

open Nat

theorem infinitely_many_primes_with_f_x_congr_0
  (f : ℕ → ℕ)
  (h_inj : Function.Injective f)
  (k : ℕ)
  (h_k_pos : k > 0)
  (h_upper_bound : ∀ n, f(n) ≤ n^k)
  : ∃ᶠ q in (PrimeSet : Filter ℕ), ∃ x : ℕ, Prime x ∧ f(x) ≡ 0 [MOD q] := 
sorry

end infinitely_many_primes_with_f_x_congr_0_l335_335608


namespace total_stick_length_l335_335600

theorem total_stick_length :
  ∃ (first second third fourth fifth : ℝ),
  first = 3 ∧
  second = 2 * first ∧
  third = second - 1 ∧
  fourth = third / 2 ∧
  fifth = 4 * fourth ∧
  first + second + third + fourth + fifth = 26.5 :=
by
  use [3, 6, 5, 2.5, 10]
  repeat { split, norm_num }
  sorry

end total_stick_length_l335_335600


namespace dice_even_odd_probability_l335_335452

theorem dice_even_odd_probability :
  let n_dice := 8
  let prob_even := (1 / 2 : ℝ)  -- probability a single die shows an even number (2, 4, or 6)
  let prob_odd := (1 / 2 : ℝ)   -- probability a single die shows an odd number (1, 3, or 5)
  let comb := nat.choose n_dice 4
  let prob_comb := (prob_even ^ 4) * (prob_odd ^ 4)
  let total_prob := comb * prob_comb
  total_prob = (35 / 128 : ℝ) :=
by {
  sorry  -- proof to be filled in
}

end dice_even_odd_probability_l335_335452


namespace ellipse_x_intercept_l335_335415

-- Definition of the problem parameters
def foci1 : ℝ × ℝ := (0, 2)
def foci2 : ℝ × ℝ := (6, 0)
def intercept1 : ℝ × ℝ := (0, 0)

-- The sum of distances constraint for an ellipse
def sum_of_distances {p1 p2 q : ℝ × ℝ} := dist p1 q + dist p2 q = 8

-- Proof problem statement
theorem ellipse_x_intercept :
  ∃ x : ℝ, x > 0 ∧ sum_of_distances foci1 foci2 (x, 0) ∧ x = 48 / 7 :=
by
  sorry

end ellipse_x_intercept_l335_335415


namespace remainder_of_3_pow_2015_mod_13_l335_335876

theorem remainder_of_3_pow_2015_mod_13 :
  (3 ^ 2015) % 13 = 9 :=
by
  have key_congruence : 3 ^ 3 % 13 = 1 := sorry
  have decomposition : 2015 = 3 * 671 + 2 := by
    calc
      2025 = 3 * 671 + 2 := by norm_num
  calc
    (3 ^ 2015) % 13
      = (3 ^ (3 * 671 + 2)) % 13 := by rw [decomposition]
  ... = ((3 ^ 3) ^ 671 * 3 ^ 2) % 13 := by rw [pow_add, pow_mul]
  ... = (1 ^ 671 * 3 ^ 2) % 13 := by rw [key_congruence]
  ... = (1 * 9) % 13 := by norm_num
  ... = 9 := by norm_num

end remainder_of_3_pow_2015_mod_13_l335_335876


namespace task_arrangement_possibilities_l335_335823

theorem task_arrangement_possibilities :
  let students := ["A", "B", "C", "D", "E"]
  let tasks := ["translation", "tour_guiding", "etiquette", "driving"]
  ∀ (A B C D E : string),
  A ∈ students → B ∈ students → C ∈ students → D ∈ students → E ∈ students →
  ∀ (translation tour_guiding etiquette driving : string),
  translation ∈ tasks → tour_guiding ∈ tasks → etiquette ∈ tasks → driving ∈ tasks →
  (A ≠ "D" ∧ A ≠ "E") → (B ≠ "D" ∧ B ≠ "E") → (C ≠ "D" ∧ C ≠ "E") → 
  (num_arrangements A B C D E translation tour_guiding etiquette driving = 78) := sorry

end task_arrangement_possibilities_l335_335823


namespace circle_has_most_axes_of_symmetry_l335_335807

def number_of_axes_of_symmetry (shape : Type) : ℕ :=
match shape with
| Circle     => 0 -- representing infinity
| Square     => 4
| Angle      => 1
| LineSegment => 1
| _          => 0

def most_axes_of_symmetry (shapes : List Type) : Type :=
shapes.foldl (λ acc s, if number_of_axes_of_symmetry s > number_of_axes_of_symmetry acc then s else acc) Circle

-- Define the shapes
@[derive DecidableEq]
inductive Shape
| Circle
| Square
| Angle
| LineSegment

open Shape

theorem circle_has_most_axes_of_symmetry :
  most_axes_of_symmetry [Circle, Square, Angle, LineSegment] = Circle := 
by sorry

end circle_has_most_axes_of_symmetry_l335_335807


namespace stone_count_121_l335_335062

theorem stone_count_121 : 
  let n := 11
  let full_cycle := 2 * n - 1
  let target := 121
  target % full_cycle = 19 →
  (find_stone : ℕ → ℕ) 
  (H : ∀ k, find_stone k = if k % full_cycle < n then k % full_cycle + 1 else 2 * n - (k % full_cycle))
  find_stone target = 10 := by
  sorry

end stone_count_121_l335_335062


namespace num_workers_in_factory_l335_335667

theorem num_workers_in_factory 
  (average_salary_total : ℕ → ℕ → ℕ)
  (old_supervisor_salary : ℕ)
  (average_salary_9_new : ℕ)
  (new_supervisor_salary : ℕ) :
  ∃ (W : ℕ), 
  average_salary_total (W + 1) 430 = W * 430 + 870 ∧ 
  average_salary_9_new = 9 * 390 ∧ 
  W + 1 = (9 * 390 - 510 + 870) / 430 := 
by {
  sorry
}

end num_workers_in_factory_l335_335667


namespace jason_birth_year_l335_335676

theorem jason_birth_year :
  ∀ (year_first_amc8 year_tenth_amc8 jason_age_at_tenth : ℕ), 
  year_first_amc8 = 1985 → 
  year_tenth_amc8 = year_first_amc8 + 9 →
  jason_age_at_tenth = 13 →
  jason_birth_year = year_tenth_amc8 - jason_age_at_tenth →
  jason_birth_year = 1981 :=
by
  intros year_first_amc8 year_tenth_amc8 jason_age_at_tenth
  intros h_year_first h_year_tenth h_jason_age h_jason_birth_year
  sorry

end jason_birth_year_l335_335676


namespace smallest_expression_l335_335959

theorem smallest_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  (y / x = 1 / 2) ∧ (y / x < x + y) ∧ (y / x < x * y) ∧ (y / x < x - y) ∧ (y / x < x / y) :=
by
  -- The proof is to be filled by the user
  sorry

end smallest_expression_l335_335959


namespace polar_equation_of_circle_slope_is_correct_l335_335986

noncomputable def circle_equation_polar :=
  ∀ (ρ α : ℝ),
  (∃ x y : ℝ, (x + 6) ^ 2 + y ^ 2 = 25 ∧ x = ρ * cos α ∧ y = ρ * sin α) →
  ρ^2 + 12 * ρ * cos α + 11 = 0

noncomputable def slope_of_line :=
  ∀ (α : ℝ),
  (∃ (t : ℝ -> ℝ), ∀ (x y : ℝ), x = t α * cos α ∧ y = t α * sin α ∧ 
  (∃ (A B : ℝ × ℝ), (x, y) = A ∧ (x, y) = B ∧ dist A B = √10)) →
  (∃ k : ℝ, k = ± (√15 / 3))
  
-- Add sorry to skip the proof
theorem polar_equation_of_circle : circle_equation_polar := sorry 

theorem slope_is_correct : slope_of_line := sorry

end polar_equation_of_circle_slope_is_correct_l335_335986


namespace intersection_M_N_l335_335123

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335123


namespace ratio_X_N_l335_335952

-- Given conditions as definitions
variables (P Q M N X : ℝ)
variables (hM : M = 0.40 * Q)
variables (hQ : Q = 0.30 * P)
variables (hN : N = 0.60 * P)
variables (hX : X = 0.25 * M)

-- Prove that X / N == 1 / 20
theorem ratio_X_N : X / N = 1 / 20 :=
by
  sorry

end ratio_X_N_l335_335952


namespace bob_has_winning_strategy_l335_335409

theorem bob_has_winning_strategy : ∃ winner : StringGamePlayer, winner = Bob ∧ has_winning_strategy winner (StringGame 2015) :=
by
  sorry

end bob_has_winning_strategy_l335_335409


namespace ellipse_eccentricity_l335_335230

variables {F1 F2 M : Type}
variable [metric_space F1]
variable [metric_space F2]
variable [metric_space M]

def eccentricity (C : ellipse) (e : ℝ) : Prop := 
∃ (F1 F2 : M) (M : M), 
    ellipse_foci C F1 F2 ∧ 
    on_ellipse C M ∧ 
    dist M F1 = 12 ∧ 
    dist M F2 = 16 ∧ 
    sin_angle M F2 F1 = 3/5 ∧ 
    (e = 5/7 ∨ e = 1/5)

theorem ellipse_eccentricity :
  ∃ e : ℝ, eccentricity C e :=
sorry

end ellipse_eccentricity_l335_335230


namespace one_def_and_two_def_mutually_exclusive_one_def_and_two_def_not_complementary_at_least_one_def_and_all_def_not_exclusive_nor_complementary_at_least_one_genuine_and_one_def_not_exclusive_nor_complementary_l335_335492

-- Definitions
def batch (genuine defective : ℕ) := genuine > 2 ∧ defective > 2

def select_two_items (genuine defective : ℕ) : list (ℕ × ℕ) := sorry  -- Placeholder, as the actual selection process is not specified

def exactly_one_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def exactly_two_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def at_least_one_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def all_defective (selection : list (ℕ × ℕ)) : Prop := sorry
def at_least_one_genuine (selection : list (ℕ × ℕ)) : Prop := sorry

-- Propositions based on the problem conditions
theorem one_def_and_two_def_mutually_exclusive (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, exactly_one_defective selection → ¬exactly_two_defective selection :=
sorry

theorem one_def_and_two_def_not_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, exactly_one_defective selection ∨ exactly_two_defective selection → ¬(exactly_one_defective selection ↔ ¬exactly_two_defective selection) :=
sorry

theorem at_least_one_def_and_all_def_not_exclusive_nor_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, ¬(at_least_one_defective selection ∧ ¬all_defective selection ∨ all_defective selection ∧ ¬at_least_one_defective selection) :=
sorry

theorem at_least_one_genuine_and_one_def_not_exclusive_nor_complementary (genuine defective : ℕ) (h : batch genuine defective) : 
  ∀ selection, ¬(at_least_one_genuine selection ∧ ¬at_least_one_defective selection ∨ at_least_one_genuine selection ∧ at_least_one_defective selection) :=
sorry

end one_def_and_two_def_mutually_exclusive_one_def_and_two_def_not_complementary_at_least_one_def_and_all_def_not_exclusive_nor_complementary_at_least_one_genuine_and_one_def_not_exclusive_nor_complementary_l335_335492


namespace largest_divisible_by_9_l335_335627

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by_9 (n : ℕ) : Prop :=
  sum_of_digits n % 9 = 0

noncomputable def digits_and_positions (n : ℕ) : List (ℕ × ℕ) :=
  n.digits 10 |>.reverse.enum

noncomputable def remove_digits (digits_positions : List (ℕ × ℕ)) (indexes : List ℕ) : ℕ :=
  let remaining_digits := digits_positions.filter (λ dp => ¬ List.elem dp.1 indexes)
  list_to_nat (remaining_digits.map Prod.snd).reverse

theorem largest_divisible_by_9 (n : ℕ) (h : n = 547654765476) : ∃ m, m ≤ n ∧ divisible_by_9 m :=
  ∃ (5476547646 : ℕ), sorry

#eval largest_divisible_by_9 547654765476 rfl

end largest_divisible_by_9_l335_335627


namespace range_of_m_l335_335914

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end range_of_m_l335_335914


namespace number_wall_problem_l335_335990

theorem number_wall_problem (m : ℤ) : 
  ((m + 5) + 16 + 18 = 56) → (m = 17) :=
by
  sorry

end number_wall_problem_l335_335990


namespace find_c_direction_vector_l335_335504

theorem find_c_direction_vector :
  let p1 := ⟨-7, 3⟩
  let p2 := ⟨-3, -1⟩
  let direction_vector := p2 - p1
  ∃ c : ℤ, direction_vector = ⟨4, c⟩ ∧ c = -4 :=
by
  sorry

end find_c_direction_vector_l335_335504


namespace solve_election_votes_l335_335983

def election_problem (V : ℕ) : Prop :=
  let winner_votes := 0.45 * V
  let second_votes := 0.35 * V
  let third_votes := V - (winner_votes + second_votes)
  winner_votes == 675 ∧ second_votes == 525 ∧ third_votes == 300 ∧ V == 1500

theorem solve_election_votes (V : ℕ) (h1 : winner_votes = 0.45 * V) (h2 : second_votes = 0.35 * V) (h3 : winner_votes - second_votes = 150) : election_problem V :=
by {
  sorry
}

end solve_election_votes_l335_335983


namespace max_squares_covered_by_card_l335_335769

theorem max_squares_covered_by_card : 
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  ∃ n, n = 9 :=
by
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  existsi 9
  sorry

end max_squares_covered_by_card_l335_335769


namespace num_of_adults_l335_335653

def students : ℕ := 22
def vans : ℕ := 3
def capacity_per_van : ℕ := 8

theorem num_of_adults : (vans * capacity_per_van) - students = 2 := by
  sorry

end num_of_adults_l335_335653


namespace cannot_sum_to_nine_l335_335052

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l335_335052


namespace alien_takes_home_l335_335816

variable (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ)

-- Conditions
def initial_abducted_people : abducted = 200 := rfl
def percentage_returned_people : returned_percentage = 0.8 := rfl
def people_taken_to_another_planet : taken_to_another_planet = 10 := rfl

-- The question to prove
def people_taken_home (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ) : ℕ :=
  let returned := (returned_percentage * abducted) in
  let remaining := abducted - returned in
  remaining - taken_to_another_planet

theorem alien_takes_home :
  people_taken_home 200 0.8 10 = 30 :=
by
  -- calculations directly in Lean or use sorry to represent the correctness
  sorry

end alien_takes_home_l335_335816


namespace tan_ratio_sum_l335_335179

theorem tan_ratio_sum (x y : ℝ) (hx : cos x ≠ 0) (hy : cos y ≠ 0) (hx' : sin x ≠ 0) (hy' : sin y ≠ 0)
  (h1 : sin x / cos y + sin y / cos x = 2)
  (h2 : cos x / sin y + cos y / sin x = 4) :
  (tan x / tan y + tan y / tan x = 6) :=
by
  sorry

end tan_ratio_sum_l335_335179


namespace find_quarters_l335_335390

def num_pennies := 123
def num_nickels := 85
def num_dimes := 35
def cost_per_scoop_cents := 300  -- $3 = 300 cents
def num_family_members := 5
def leftover_cents := 48

def total_cost_cents := num_family_members * cost_per_scoop_cents
def total_initial_cents := total_cost_cents + leftover_cents

-- Values of coins in cents
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25

def total_pennies_value := num_pennies * penny_value
def total_nickels_value := num_nickels * nickel_value
def total_dimes_value := num_dimes * dime_value
def total_initial_excluding_quarters := total_pennies_value + total_nickels_value + total_dimes_value

def total_quarters_value := total_initial_cents - total_initial_excluding_quarters
def num_quarters := total_quarters_value / quarter_value

theorem find_quarters : num_quarters = 26 := by
  sorry

end find_quarters_l335_335390


namespace trisect_midpoint_length_l335_335371

theorem trisect_midpoint_length
  (A D B C M : Type)
  (dist : A → A → ℝ)
  (trisect : dist A B = dist B C ∧ dist B C = dist C D)
  (midpoint : dist A M = dist M D)
  (MC_eq_4 : dist M C = 4)
  (AB_twice_MC : dist A B = 2 * dist M C) : dist A D = 24 := 
sorry

end trisect_midpoint_length_l335_335371


namespace straight_angle_mars_l335_335258

theorem straight_angle_mars : 
  (full_circle_lerps : ℝ) (straight_angle : ℝ) 
  (h : full_circle_lerps = 300) 
  (h2 : straight_angle = full_circle_lerps / 2) : 
  straight_angle = 150 :=
by
  rw [h, h2]
  norm_num
  sorry

end straight_angle_mars_l335_335258


namespace discount_on_item_l335_335277

noncomputable def discount_percentage : ℝ := 20
variable (total_cart_value original_price final_amount : ℝ)
variable (coupon_discount : ℝ)

axiom cart_value : total_cart_value = 54
axiom item_price : original_price = 20
axiom coupon : coupon_discount = 0.10
axiom final_price : final_amount = 45

theorem discount_on_item :
  ∃ x : ℝ, (total_cart_value - (x / 100) * original_price) * (1 - coupon_discount) = final_amount ∧ x = discount_percentage :=
by
  have eq1 := cart_value
  have eq2 := item_price
  have eq3 := coupon
  have eq4 := final_price
  sorry

end discount_on_item_l335_335277


namespace exists_infinite_ziba_numbers_in_form_l335_335341

def is_ziba (m : ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ m →
    ∃ S : Finset ℕ, (∀ x ∈ S, x ∣ m ∧ x > 0) ∧ (S.sum id = n)

theorem exists_infinite_ziba_numbers_in_form :
  ∃^∞ k : ℕ, is_ziba (k^2 + k + 2022) :=
sorry

end exists_infinite_ziba_numbers_in_form_l335_335341


namespace exists_fixed_point_l335_335366

open EuclideanGeometry

-- Define the entities and positions
variables (k₁ k₂ : Circle) (A B : Point)
variables (O₁ O₂ : Point) (P₁ P₂ : Point)

-- Conditions on circles k₁ and k₂
def circles_intersect (k₁ k₂ : Circle) (A B: Point) :=
  k₁.contains A ∧ k₁.contains B ∧ k₂.contains A ∧ k₂.contains B ∧ A ≠ B

-- Conditions on particles P₁ and P₂
def particles_move (k₁ k₂ : Circle) (A : Point) (P₁ P₂ : Point) (t : ℝ) :=
  ∀ t, k₁.contains (P₁ t) ∧ k₂.contains (P₂ t) ∧ (P₁ 0 = A) ∧ (P₂ 0 = A) ∧
       (P₁ 1 = A) ∧ (P₂ 1 = A)  -- particles return to A after full rotation

-- Theorem statement: Existence of fixed point P
theorem exists_fixed_point (k₁ k₂ : Circle) (A B : Point)
                          (h_intersect : circles_intersect k₁ k₂ A B)
                          (P₁ P₂ : ℝ → Point)
                          (h_move : particles_move k₁ k₂ A P₁ P₂) :
  ∃ P : Point, ∀ t : ℝ, dist P (P₁ t) = dist P (P₂ t) :=
sorry

end exists_fixed_point_l335_335366


namespace has_exactly_one_solution_l335_335088

theorem has_exactly_one_solution (a : ℝ) : 
  (∀ x : ℝ, 5^(x^2 + 2 * a * x + a^2) = a * x^2 + 2 * a^2 * x + a^3 + a^2 - 6 * a + 6) ↔ (a = 1) :=
sorry

end has_exactly_one_solution_l335_335088


namespace minimum_value_on_interval_l335_335682

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem minimum_value_on_interval : ∀ x ∈ set.Icc 2 3, f x ≥ f 2 :=
begin
  intro x,
  intro hx,
  -- Proof skipped
  sorry
end

end minimum_value_on_interval_l335_335682


namespace airline_passenger_capacity_l335_335812

def seats_per_row : Nat := 7
def rows_per_airplane : Nat := 20
def airplanes_owned : Nat := 5
def flights_per_day_per_airplane : Nat := 2

def seats_per_airplane : Nat := rows_per_airplane * seats_per_row
def total_seats : Nat := airplanes_owned * seats_per_airplane
def total_flights_per_day : Nat := airplanes_owned * flights_per_day_per_airplane
def total_passengers_per_day : Nat := total_flights_per_day * total_seats

theorem airline_passenger_capacity :
  total_passengers_per_day = 7000 := sorry

end airline_passenger_capacity_l335_335812


namespace finite_pos_int_set_condition_l335_335072

theorem finite_pos_int_set_condition (X : Finset ℕ) 
  (hX : ∀ a ∈ X, 0 < a) 
  (h2 : 2 ≤ X.card) 
  (hcond : ∀ {a b : ℕ}, a ∈ X → b ∈ X → a > b → b^2 / (a - b) ∈ X) :
  ∃ a : ℕ, X = {a, 2 * a} :=
by
  sorry

end finite_pos_int_set_condition_l335_335072


namespace intersection_of_equal_circles_l335_335717

-- Let A and B be points of intersection of two equal circles.
-- Let C and D be points of intersection of a line through A with the respective circles.
-- Let E be the midpoint of segment CD.
-- We need to prove BE is perpendicular to CD.
theorem intersection_of_equal_circles (A B C D E : Point)
  (h1 : circlesIntersectAtTwoPointsWithCentersAndEqualRadius A B C D)
  (h2 : lineThroughIntersectsCirclesAt A C D)
  (h3 : E = midpoint C D) :
  isPerpendicular (lineSegment B E) (lineSegment C D) :=
sorry

end intersection_of_equal_circles_l335_335717


namespace probability_calculation_l335_335084

open Classical

def probability_odd_sum_given_even_product :=
  let num_even := 4  -- even numbers: 2, 4, 6, 8
  let num_odd := 4   -- odd numbers: 1, 3, 5, 7
  let total_outcomes := 8^5
  let prob_all_odd := (num_odd / 8)^5
  let prob_even_product := 1 - prob_all_odd

  let ways_one_odd := 5 * num_odd * num_even^4
  let ways_three_odd := Nat.choose 5 3 * num_odd^3 * num_even^2
  let ways_five_odd := num_odd^5

  let favorable_outcomes := ways_one_odd + ways_three_odd + ways_five_odd
  let total_even_product_outcomes := total_outcomes * prob_even_product

  favorable_outcomes / total_even_product_outcomes

theorem probability_calculation :
  probability_odd_sum_given_even_product = rational_result := sorry

end probability_calculation_l335_335084


namespace aladdin_equator_travel_l335_335804

-- The definition of Aladdin's continuous movement over the equator.
def aladdin_movement (φ : ℝ → ℝ) : Prop := continuous φ

-- The main theorem to be proved
theorem aladdin_equator_travel (φ : ℝ → ℝ) (h_continuous : aladdin_movement φ) :
  ∃ t1 t2 : ℝ, |φ(t1) - φ(t2)| ≥ 1 := 
sorry

end aladdin_equator_travel_l335_335804


namespace median_roller_coaster_durations_l335_335699

def roller_coaster_durations : List ℕ := [15, 30, 30, 45, 55, 70, 80, 94, 100, 115, 132, 142, 158, 168, 185, 190, 195, 215, 230, 255, 255]

def median (xs : List ℕ) : ℕ :=
  let sorted_xs := xs.qsort (· ≤ ·)
  sorted_xs.get (sorted_xs.length / 2)

theorem median_roller_coaster_durations :
  median roller_coaster_durations = 100 :=
  sorry

end median_roller_coaster_durations_l335_335699


namespace part1_part2_l335_335619

variable {a b : ℝ}

noncomputable def in_interval (x: ℝ) : Prop :=
  -1/2 < x ∧ x < 1/2

theorem part1 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1/3 * a + 1/6 * b) < 1/4 := 
by sorry

theorem part2 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1 - 4 * a * b) > 2 * abs (a - b) := 
by sorry

end part1_part2_l335_335619


namespace cats_left_l335_335396

def initial_siamese_cats : ℕ := 12
def initial_house_cats : ℕ := 20
def cats_sold : ℕ := 20

theorem cats_left : (initial_siamese_cats + initial_house_cats - cats_sold) = 12 :=
by
sorry

end cats_left_l335_335396


namespace complex_point_second_quadrant_l335_335691

theorem complex_point_second_quadrant (i : ℂ) (h1 : i^4 = 1) :
  ∃ (z : ℂ), z = ((i^(2014))/(1 + i) * i) ∧ z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_second_quadrant_l335_335691


namespace desired_line_equation_exists_l335_335392

theorem desired_line_equation_exists :
  ∃ (a b c : ℝ), (a * 0 + b * 1 + c = 0) ∧
  (x - 3 * y + 10 = 0) ∧
  (2 * x + y - 8 = 0) ∧
  (a * x + b * y + c = 0) :=
by
  sorry

end desired_line_equation_exists_l335_335392


namespace distance_after_rest_l335_335354

-- Define the conditions
def distance_before_rest := 0.75
def total_distance := 1.0

-- State the theorem
theorem distance_after_rest :
  total_distance - distance_before_rest = 0.25 :=
by sorry

end distance_after_rest_l335_335354


namespace isosceles_trapezoid_ratio_l335_335582

theorem isosceles_trapezoid_ratio
  (a : ℝ)
  (h1 : a > 0)
  (ABCD_is_isosceles_trapezoid : ∃ AB CD hCE,
    CD = 2 * AB ∧
    hCE = AB / 2 ∧
    hCE = CE ∧
    (∃ E, right_triangle_ACE CD E hCE))
  : (AB / AC) = 2 * sqrt(5) / 5 := 
sorry

end isosceles_trapezoid_ratio_l335_335582


namespace solve_inequality_l335_335563

theorem solve_inequality (a : ℝ) :
  (∀ (x : ℝ), ¬ ((a - 2) * x^2 + 2 * (a - 2) * x - 4 ≥ 0)) ↔ (-2 < a ∧ a ≤ 2) := 
begin
  sorry
end

end solve_inequality_l335_335563


namespace sheryll_paid_total_l335_335570

-- Variables/conditions
variables (cost_per_book : ℝ) (num_books : ℕ) (discount_per_book : ℝ)

-- Given conditions
def assumption1 : cost_per_book = 5 := by sorry
def assumption2 : num_books = 10 := by sorry
def assumption3 : discount_per_book = 0.5 := by sorry

-- Theorem statement
theorem sheryll_paid_total : cost_per_book = 5 → num_books = 10 → discount_per_book = 0.5 → 
  (cost_per_book - discount_per_book) * num_books = 45 := by
  sorry

end sheryll_paid_total_l335_335570


namespace solve_for_x_l335_335555

theorem solve_for_x (x : ℂ) (h : x / complex.I = 1 - complex.I) : x = 1 + complex.I :=
by
  sorry

end solve_for_x_l335_335555


namespace hypotenuse_length_l335_335579

theorem hypotenuse_length (a b c : ℝ) 
  (h_right_angled : c^2 = a^2 + b^2) 
  (h_sum_squares : a^2 + b^2 + c^2 = 980) : 
  c = 70 :=
by
  sorry

end hypotenuse_length_l335_335579


namespace smallest_possible_sum_S_l335_335348

theorem smallest_possible_sum_S (n : ℕ) (R : ℕ) (S : ℕ) (d_i : ℕ → ℕ) :
  (∀ i, 1 ≤ d_i i ∧ d_i i ≤ 8) →
  (R = 2400 ∧ S = ∑ i in finset.range n, (8 - d_i i)) →
  S = 0 :=
by
  sorry

end smallest_possible_sum_S_l335_335348


namespace sequence_product_eq_l335_335725

theorem sequence_product_eq :
  (2 * (List.prod (List.map (λ n, (n + 2 : ℝ) / (n + 1)) (List.range (2009 - 2 + 1))))) = 5360 :=
by sorry

end sequence_product_eq_l335_335725


namespace union_M_N_l335_335372

noncomputable def M : Set ℝ := { x | x^2 - 3 * x = 0 }
noncomputable def N : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }

theorem union_M_N : M ∪ N = {0, 2, 3} :=
by {
  sorry
}

end union_M_N_l335_335372


namespace solution_l335_335294

theorem solution (f : ℝ → ℝ)
  (h_mono : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y))
  (h_func : ∀ x : ℝ, 0 < x → f(f(x) - log x - x^3) = 2)
  : f(real.exp(1)) = real.exp(1)^3 + 2 :=
sorry

end solution_l335_335294


namespace num_integers_x_polynomial_neg_l335_335490

theorem num_integers_x_polynomial_neg : ∀ x : ℤ, (x^4 - 55 * x^2 + 54 < 0) ↔ (x ∈ (⋃ n : ℕ in {2, 3, 4, 5, 6, 7}, {n, -n})).card = 12 :=
by sorry

end num_integers_x_polynomial_neg_l335_335490


namespace quadratic_condition_l335_335266

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l335_335266


namespace coefficients_divisible_by_5_l335_335663

theorem coefficients_divisible_by_5 
  (a b c d : ℤ) 
  (h : ∀ x : ℤ, 5 ∣ (a * x^3 + b * x^2 + c * x + d)) : 
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d := 
by {
  sorry
}

end coefficients_divisible_by_5_l335_335663


namespace ratio_corn_peas_on_barry_l335_335417

theorem ratio_corn_peas_on_barry (total_area ang_area bar_area corn_total peas_total 
                                  corn_ang peas_ang corn_bar peas_bar : ℕ) 
                                  (h_ang_total : ang_area + bar_area = total_area)
                                  (h_total : corn_total + peas_total = total_area)
                                  (h_ang_bar_ratio : 3 * bar_area = 2 * ang_area) 
                                  (h_total_ratio : 7 * peas_total = 3 * corn_total) 
                                  (h_ang_ratio : 4 * peas_ang = corn_ang) 
                                  (h_ang_corn : corn_ang + peas_ang = ang_area)
                                  (h_ang : ang_area = 60) 
                                  (h_total_land: total_area = 100)
                                  (h_corn : corn_total = 70)
                                  (h_peas : peas_total = 30)
                                  (h_bar_corn: corn_bar = corn_total - corn_ang)
                                  (h_bar_area: bar_area = ang_area / 1.5) 
                                  (h_bar_peas: peas_bar = peas_total - peas_ang)
                                 :
  11 * peas_bar = 9 * corn_bar := by
  sorry

end ratio_corn_peas_on_barry_l335_335417


namespace div_count_l335_335886

theorem div_count (n : ℕ) (h_pos : n > 0) (h_divisors_240_n_3 : (240 * n^3).d = 240) : ((216 * n^4).d = 156) :=
sorry

end div_count_l335_335886


namespace joan_kittens_total_l335_335223

-- Definition of the initial conditions
def joan_original_kittens : ℕ := 8
def neighbor_original_kittens : ℕ := 6
def joan_gave_away : ℕ := 2
def neighbor_gave_away : ℕ := 4
def joan_adopted_from_neighbor : ℕ := 3

-- The final number of kittens Joan has
def joan_final_kittens : ℕ :=
  let joan_remaining := joan_original_kittens - joan_gave_away
  let neighbor_remaining := neighbor_original_kittens - neighbor_gave_away
  let adopted := min joan_adopted_from_neighbor neighbor_remaining
  joan_remaining + adopted

theorem joan_kittens_total : joan_final_kittens = 8 := 
by 
  -- Lean proof would go here, but adding sorry for now
  sorry

end joan_kittens_total_l335_335223


namespace PM_equals_MQ_l335_335420

open EuclideanGeometry

variables {A B C D E F M P Q : Point}

-- Define conditions
def midpoint (AB M : Point) : Prop := dist AB 0 = 2 * dist A M
def line_intersect (P Q : Point) : Prop := ∃ l : Line, P ∈ l ∧ Q ∈ l
def on_line (P l : Point) (l : Line) : Prop := P ∈ l

-- Given conditions
axiom M_midpoint_AB : midpoint A B M
axiom CD_EF_intersect_M : line_intersect (C, D) ∧ line_intersect (E, F) ∧ on_line M CD ∧ on_line M EF
axiom CF_intersects_AB_at_P : on_line P CF ∧ CF ∈ line AB
axiom ED_intersects_AB_at_Q : on_line Q ED ∧ ED ∈ line AB

-- The statement to be proven
theorem PM_equals_MQ : dist P M = dist Q M :=
sorry

end PM_equals_MQ_l335_335420


namespace divide_6_books_into_three_parts_each_2_distribute_6_books_to_ABC_each_2_distribute_6_books_to_ABC_distribute_6_books_to_ABC_each_at_least_1_l335_335710

open Finset

-- Proof problem for Ⅰ
theorem divide_6_books_into_three_parts_each_2 : 
  ∃ (S : Finset (Finset ℕ)), S.card = 15 ∧ ∀ s ∈ S, s.card = 2 ∧ s.sum = 6 :=
sorry

-- Proof problem for Ⅱ
theorem distribute_6_books_to_ABC_each_2 : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 90 ∧ 
  ∀ s ∈ S, (∀ t ∈ s, t.card = 2) ∧ S.sum = 6 :=
sorry

-- Proof problem for Ⅲ
theorem distribute_6_books_to_ABC : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 729 ∧ S.sum = 6 :=
sorry

-- Proof problem for Ⅳ
theorem distribute_6_books_to_ABC_each_at_least_1 : 
  ∃ (S : Finset (Finset (Finset ℕ))), S.card = 481 ∧ S.sum = 6 ∧ 
  ∀ s ∈ S, (∀ t ∈ s, t.sum ≥ 1) :=
sorry

end divide_6_books_into_three_parts_each_2_distribute_6_books_to_ABC_each_2_distribute_6_books_to_ABC_distribute_6_books_to_ABC_each_at_least_1_l335_335710


namespace carols_birthday_l335_335751

def possible_dates : List (String × Nat) := [
    ("January", 4), ("January", 5), ("January", 11),
    ("March", 8),
    ("April", 8), ("April", 9),
    ("June", 5), ("June", 7),
    ("July", 13),
    ("October", 4), ("October", 7), ("October", 8)
]

theorem carols_birthday : ∃ m d, (m, d) ∈ possible_dates ∧ 
  (∀ x, x ≠ ("March", 8) ∧ x ≠ ("July", 13)) ∧
  (∀ y, y.2 ≠ 11 ∧ y.2 ≠ 9 ∧ y.2 ≠ 13) ∧
  (∀ z, z.2 ≠ 4 ∧ z.2 ≠ 5 ∧ z.2 ≠ 8) ∧
  ((m = "June" ∧ d = 7) ∨ (m = "October" ∧ d = 7) → m = "June") :=
by
  sorry

end carols_birthday_l335_335751


namespace speed_of_second_fragment_l335_335385

-- Definition of the initial conditions
def initial_velocity : ℝ := 20 -- m/s
def time_after_flight : ℝ := 1  -- s
def gravity : ℝ := 10 -- m/s^2
def horizontal_velocity_first_fragment : ℝ := 48 -- m/s

-- Definition for vertical velocity after 1 second
def vertical_velocity (v0 : ℝ) (g : ℝ) (t : ℝ) : ℝ :=
  v0 - g * t

-- Vertical velocity of the second fragment is the same as that of the first fragment
def vertical_velocity_second_fragment := vertical_velocity initial_velocity gravity time_after_flight

-- Horizontal velocity of the second fragment
def horizontal_velocity_second_fragment := -horizontal_velocity_first_fragment

-- Magnitude of the velocity of the second fragment using Pythagorean theorem
def velocity_magnitude (vx vy : ℝ) : ℝ :=
  Real.sqrt (vx * vx + vy * vy)

-- Main theorem statement
theorem speed_of_second_fragment :
  velocity_magnitude horizontal_velocity_second_fragment vertical_velocity_second_fragment = 49.03 :=
by 
  sorry -- Proof omitted

end speed_of_second_fragment_l335_335385


namespace monotonicity_f_range_of_a_l335_335919

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x^2) * Real.exp x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := (1 - 2 * x - x^2) * Real.exp x

-- Prove the monotonicity of f(x)
theorem monotonicity_f : 
  ∀ x, (x < -1 - Real.sqrt 2 → f' x < 0) ∧ 
       (-1 - Real.sqrt 2 < x ∧ x < -1 + Real.sqrt 2 → f' x > 0) ∧ 
       (-1 + Real.sqrt 2 < x → f' x < 0) := 
by 
  sorry

-- Prove that the range of values for a such that f(x) ≤ a x + 1 for x ≥ 0 is [1, +∞)
theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≥ 0 → f x ≤ a * x + 1)) ↔ a ∈ Set.Ici 1 := 
by 
  sorry

end monotonicity_f_range_of_a_l335_335919


namespace find_unique_digit_sets_l335_335190

theorem find_unique_digit_sets (a b c : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
 (h4 : 22 * (a + b + c) = 462) :
  (a = 4 ∧ b = 8 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 9 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 4 ∧ c = 9) ∨
  (a = 8 ∧ b = 9 ∧ c = 4) ∨ 
  (a = 9 ∧ b = 4 ∧ c = 8) ∨ 
  (a = 9 ∧ b = 8 ∧ c = 4) ∨
  (a = 5 ∧ b = 7 ∧ c = 9) ∨ 
  (a = 5 ∧ b = 9 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 9) ∨
  (a = 7 ∧ b = 9 ∧ c = 5) ∨ 
  (a = 9 ∧ b = 5 ∧ c = 7) ∨ 
  (a = 9 ∧ b = 7 ∧ c = 5) ∨
  (a = 6 ∧ b = 7 ∧ c = 8) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 6 ∧ c = 8) ∨
  (a = 7 ∧ b = 8 ∧ c = 6) ∨ 
  (a = 8 ∧ b = 6 ∧ c = 7) ∨ 
  (a = 8 ∧ b = 7 ∧ c = 6) :=
sorry

end find_unique_digit_sets_l335_335190


namespace working_together_days_l335_335361

noncomputable def efficiency_A : ℕ := 100
noncomputable def efficiency_B : ℕ := efficiency_A - 60
noncomputable def work_days_A : ℕ := 35
noncomputable def work_rate_A : ℚ := 1 / work_days_A
noncomputable def work_rate_B : ℚ := (efficiency_B : ℚ / efficiency_A) * work_rate_A
noncomputable def combined_work_rate : ℚ := work_rate_A + work_rate_B
noncomputable def combined_work_days : ℚ := 1 / combined_work_rate

theorem working_together_days:
  combined_work_days = 25 := by
  sorry

end working_together_days_l335_335361


namespace reciprocal_expression_equals_two_l335_335955

theorem reciprocal_expression_equals_two (x y : ℝ) (h : x * y = 1) : 
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end reciprocal_expression_equals_two_l335_335955


namespace solve_logarithm_problem_l335_335550

theorem solve_logarithm_problem (x y : ℝ) (h1 : 2^x = 9) (h2 : log 2 (8 / 3) = y) : x + 2 * y = 6 := by
  sorry

end solve_logarithm_problem_l335_335550


namespace percentage_decrease_hours_with_assistant_l335_335599

theorem percentage_decrease_hours_with_assistant :
  ∀ (B H H_new : ℝ), H_new = 0.9 * H → (H - H_new) / H * 100 = 10 :=
by
  intros B H H_new h_new_def
  sorry

end percentage_decrease_hours_with_assistant_l335_335599


namespace symmetric_point_correct_line_passes_second_quadrant_l335_335732

theorem symmetric_point_correct (x y: ℝ) (h_line : y = x + 1) :
  (x, y) = (-1, 2) :=
sorry

theorem line_passes_second_quadrant (m x y: ℝ) (h_line: m * x + y + m - 1 = 0) :
  (x, y) = (-1, 1) :=
sorry

end symmetric_point_correct_line_passes_second_quadrant_l335_335732


namespace XiaoYing_and_XiaoMing_under_point_light_l335_335035

-- Definition of the problem
def shorter_has_longer_shadow := 
  ∀ (h_Ying h_Ming l_Ying l_Ming : ℝ), 
    h_Ying < h_Ming → l_Ying > l_Ming → ∃ (light_source : Type), light_source = "point"

-- Statement of the theorem
theorem XiaoYing_and_XiaoMing_under_point_light :
  shorter_has_longer_shadow := 
begin
  sorry
end

end XiaoYing_and_XiaoMing_under_point_light_l335_335035


namespace max_airlines_l335_335768

theorem max_airlines (n : ℕ) (h : n ≥ 2) :
  ∃ m : ℕ, m = ⌊n / 2⌋ ∧
  (∀ (route : ℕ × ℕ), ∃ airline : ℕ, airline < m) ∧
  (∀ airline : ℕ, airline < m → ∀ (city1 city2 : ℕ), connected city1 city2 airline) := 
sorry

end max_airlines_l335_335768


namespace no_adjacent_alphabet_rearrangements_l335_335545

def is_valid_rearrangement (s : String) : Prop :=
  s ≠ "wx" ∧ s ≠ "xy" ∧ s ≠ "yz" ∧
  s ≠ "wxyz" ∧ s ≠ "wxzy" ∧ s ≠ "wyxz" ∧ s ≠ "wyzx" ∧ s ≠ "wzxy" ∧
  s ≠ "xwyz" ∧ s ≠ "xzwy" ∧ s ≠ "xzwy" ∧ s ≠ "xywz" ∧ s ≠ "xwyz" ∧
  s ≠ "ywzx" ∧ s ≠ "ywxz" ∧ s ≠ "yzxw" ∧ s ≠ "yxwz" ∧ s ≠ "yxzw" ∧
  s ≠ "zwxy" ∧ s ≠ "zxwy" ∧ s ≠ "zywx" ∧ s ≠ "zyxw"

theorem no_adjacent_alphabet_rearrangements :
  ∃ (s : List String), 
    (∀ (x ∈ s), is_valid_rearrangement x) ∧ s.length = 8 :=
sorry

end no_adjacent_alphabet_rearrangements_l335_335545


namespace tangency_point_is_ln2_l335_335910

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangency_point_is_ln2 (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) →
  (∃ m : ℝ, Real.exp m - Real.exp (-m) = 3 / 2) →
  m = Real.log 2 :=
by
  intro h1 h2
  sorry

end tangency_point_is_ln2_l335_335910


namespace no_two_digit_even_square_palindrome_l335_335949

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_two_digit : ℕ → Prop := λ n, 10 ≤ n ∧ n < 100

def is_perfect_square_of_even (n : ℕ) : Prop :=
  ∃ k : ℕ, even k ∧ k*k = n

theorem no_two_digit_even_square_palindrome :
  ∀ n : ℕ, is_two_digit n → is_perfect_square_of_even n → ¬ is_palindrome n :=
by sorry

end no_two_digit_even_square_palindrome_l335_335949


namespace imaginary_part_of_z_l335_335188

open Complex

theorem imaginary_part_of_z :
  ∃ z: ℂ, (3 - 4 * I) * z = abs (4 + 3 * I) ∧ z.im = 4 / 5 :=
by
  sorry

end imaginary_part_of_z_l335_335188


namespace center_of_symmetry_tan_shifted_l335_335856

theorem center_of_symmetry_tan_shifted (k : ℤ) : 
  let f := λ x : ℝ, Real.tan (π * x + π / 4)
  in f ((2 * k - 1) / 4) = 0 :=
sorry

end center_of_symmetry_tan_shifted_l335_335856


namespace sqrt2_sin_cos_geq_2_fourthroot_sin2t_l335_335369

-- Statement of the problem
theorem sqrt2_sin_cos_geq_2_fourthroot_sin2t (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π/2) :
  √2 * (Real.sin t + Real.cos t) ≥ 2 * (Real.sin (2 * t))^(1/4) := by
  sorry

end sqrt2_sin_cos_geq_2_fourthroot_sin2t_l335_335369


namespace yerema_can_prevent_foma_l335_335487

-- Define the coin pile and the rules of the game
structure Game :=
  (coins : List ℕ) -- List of coins with denominations 1 to 25
  (players : List String) -- List of players "Foma" and "Yerema"

-- Define the game conditions
def initial_game : Game :=
  { coins := List.range 25, -- Coins numbered 1 to 25
    players := ["Foma", "Yerema"] }

-- The question of whether Yerema can always prevent Foma from winning
theorem yerema_can_prevent_foma (g : Game) (initial_state : g = initial_game) : 
  ∃ strategy_yerema : (ℕ → Bool), ∀ strategy_foma : (ℕ → Bool),
  (∑ i in (g.coins.filter strategy_yerema), i) ≥ (∑ i in (g.coins.filter strategy_foma), i) :=
sorry

end yerema_can_prevent_foma_l335_335487


namespace sum_first_100_eq_5050_l335_335332

/-- The sum of the first 100 natural numbers is 5050. -/
theorem sum_first_100_eq_5050 : (∑ i in Finset.range 101, i) = 5050 :=
by
    sorry

end sum_first_100_eq_5050_l335_335332


namespace vector_statements_correctness_l335_335809

theorem vector_statements_correctness :
  (let s1 := (λ (a b m c o : Vector ℝ), a + m + b + o + c = a)) ∧ 
  (let s2 := (λ (a b : Vector ℝ), ∃ k : ℝ, a = (6, 2) ∧ b = (-3, k) ∧ 
              (cos θ > 0 → θ = π))) ∧ 
  (let s3 := (λ (e1 e2 : Vector ℝ), ¬ (∃ λ : ℝ, e1 = λ • e2))) ∧ 
  (let s4 := (λ (a b : Vector ℝ), ∃ θ : ℝ, a = (λ • b))) →
  (s1, s2, s3, s4).count (λ s, s) = 1 :=
sorry

end vector_statements_correctness_l335_335809


namespace percentage_of_truth_speakers_l335_335572

theorem percentage_of_truth_speakers
  (L : ℝ) (hL: L = 0.2)
  (B : ℝ) (hB: B = 0.1)
  (prob_truth_or_lies : ℝ) (hProb: prob_truth_or_lies = 0.4)
  (T : ℝ)
: T = prob_truth_or_lies - L + B :=
sorry

end percentage_of_truth_speakers_l335_335572


namespace nth_term_max_sum_first_n_terms_l335_335509

variable {α : Type*} [LinearOrder α] [HasAdd α] [HasNeg α] [HasSmul ℕ α] [HasSmul ℤ α]

-- Given conditions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a n - a (n + 1) = d

def condition (a : ℕ → ℤ) : Prop :=
  is_arithmetic_sequence a (-2) ∧ a 3 = a 2 + a 5

-- Proof statement for the nth term
theorem nth_term (a : ℕ → ℤ) (h : condition a) : ∀ n, a n = 8 - 2 * n :=
by
  sorry

-- Proof statement for the maximum sum of the first n terms
theorem max_sum_first_n_terms (a : ℕ → ℤ) (h : condition a) :
  ∃ n, S_n a n = 12 :=
by
  -- Definition of the sum of the first n terms S_n
  def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
    n * (2 * a 1 + (n - 1) * (-2)) / 2

  sorry

end nth_term_max_sum_first_n_terms_l335_335509


namespace lambda_mu_range_l335_335909

section
variables {A B C G P : Type} [affine_space V P]
variables {AB AC : V}
variables (lambda mu : ℝ)
variables [linear_independent ℝ ![AB, AC]]
variables [distinct_points A B C] -- Assume A, B, and C are distinct points.
variables (centroid_property : G = centroid ℝ ![A, B, C])
variables (inside_GBC : inside_triangle P G B C)
variables (AP_decomposition : vector_between A P = lambda • vector_between A B + mu • vector_between A C)

theorem lambda_mu_range (hG1 : lambda + mu < 1) (hG2 G_value : inside GB P) : (2/3 : ℝ) < lambda + mu ∧ lambda + mu < 1 :=
sorry
end

end lambda_mu_range_l335_335909


namespace circle_covers_four_points_l335_335981

theorem circle_covers_four_points :
  ∀ (points : List (ℝ × ℝ)), points.length = 76 ∧ ∀ (p : ℝ × ℝ), p ∈ points → 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 →
  ∃ (circle_center : ℝ × ℝ), ∃ (r : ℝ), r = 1 / 7 ∧ (List.filter (λ p, (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ r^2) points).length ≥ 4 :=
by
  sorry

end circle_covers_four_points_l335_335981


namespace geometric_mean_of_a2_a8_l335_335588

theorem geometric_mean_of_a2_a8 (a₁ q : ℝ) (a₁_eq : a₁ = 3) (q_eq : q = 2) :
  let a₂ := a₁ * q in
  let a₈ := a₁ * q^7 in
  Real.sqrt (a₂ * a₈) = 48 ∨ Real.sqrt (a₂ * a₈) = -48 :=
by
  sorry

end geometric_mean_of_a2_a8_l335_335588


namespace alien_home_planet_people_count_l335_335815

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end alien_home_planet_people_count_l335_335815


namespace min_sin_x_plus_sin_z_l335_335884

theorem min_sin_x_plus_sin_z
  (x y z : ℝ)
  (h1 : sqrt 3 * cos x = cot y)
  (h2 : 2 * cos y = tan z)
  (h3 : cos z = 2 * cot x) :
  sin x + sin z ≥ -7 * sqrt 2 / 6 := 
sorry

end min_sin_x_plus_sin_z_l335_335884


namespace count_valid_N_values_l335_335087

theorem count_valid_N_values : 
  {N : ℕ // N > 0} → (∃ m : ℕ, 144 = m * (N + 2)) ↔ {d : ℕ // d > 2 ∧ 144 % d = 0}.card = 13 :=
by {
  sorry
}

end count_valid_N_values_l335_335087


namespace max_value_t_min_value_inequality_l335_335928

noncomputable def exists_max_t (x t : ℝ) : Prop :=
|x + 1| ≥ |x - 2| + |t - 3|

theorem max_value_t :
  ∃ t, (∀ x, exists_max_t x t) → t ≤ 6 :=
by
  sorry

noncomputable def min_value_abc (a b c : ℝ) : Prop :=
a * b * c = 12 * real.sqrt 3

theorem min_value_inequality (a b c : ℝ) :
  min_value_abc a b c → (a + b)^2 + c^2 ≥ 36 :=
by
  sorry

end max_value_t_min_value_inequality_l335_335928


namespace line_through_point_area_triangle_l335_335391

theorem line_through_point_area_triangle :
  ∃ l : ℝ × ℝ × ℝ, (
    (l.1 = 8 ∨ l.1 = 2) ∧ 
    (l.2 = -5) ∧ 
    (l.3 = if l.1 = 8 then 20 else -10) ∧ 
    ∃ m b : ℝ, 
    (b = 5 * m - 4) ∧ 
    (5 * (m + 1) * (5 * m - 4) = 50) ∧ 
    l = (k, m, b)
  ) :=
sorry

end line_through_point_area_triangle_l335_335391


namespace sum_of_possible_n_l335_335718

theorem sum_of_possible_n :
  ∑ n in (Finset.filter (λ n, 4 < n ∧ n < 18) (Finset.Icc 0 17)), n = 91 :=
by
  -- This sum includes all integers n such that 4 < n < 18 (5, 6, ..., 17)
  sorry

end sum_of_possible_n_l335_335718


namespace find_maximum_value_l335_335080

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^4 + (Real.sin x) * (Real.cos x) + (Real.cos x)^4

theorem find_maximum_value :
  ∀ x: ℝ, f x ≤ 9 / 8 :=
begin
  sorry
end

end find_maximum_value_l335_335080


namespace problem_answer_is_A_l335_335677

-- Define each statement.
def statement1 : Prop := ∀ l m n : Line, (l ⊥ n) ∧ (m ⊥ n) → (l ∥ m)
def statement2 : Prop := ∀ l m t : Line, (l ∩ t ≠ ∅) ∧ (m ∩ t ≠ ∅) → (alternateInteriorAnglesEqual l m t → l ∥ m)
def statement3 : Prop := ∀ a b c : Line, (a ∥ b) ∧ (b ∥ c) → (a ∥ c)
def statement4 : Prop := ∀ P : Point, ∀ l : Line, ¬(P ∈ l) → (perpendicularSegmentIsShortest P l)
def statement5 : Prop := ∀ a b t : Line, (adjacentAnglesAngleBisectorsPerpendicular a b t)

-- Define what it means for the combination A to be correct.
def correct_comb_A : Prop :=
  statement1 ∧ statement3 ∧ statement4

-- The theorem to prove:
theorem problem_answer_is_A : correct_comb_A :=
by sorry

end problem_answer_is_A_l335_335677


namespace range_of_function_l335_335858

theorem range_of_function : 
  ∀ x ∈ set.Icc (Real.pi / 6) (5 * Real.pi / 6), 
  let sin_x := Real.sin x in 
  sin_x ∈ set.Icc (1 / 2) 1 →
  2 * sin_x^2 - 3 * sin_x + 1 ∈ set.Icc (-1 / 8) 0 :=
by
  intros x hx hsin_x
  let y := 2 * (sin_x : ℝ)^2 - 3 * sin_x + 1
  sorry

end range_of_function_l335_335858


namespace area_of_30_60_90_triangle_hypotenuse_6sqrt2_l335_335685

theorem area_of_30_60_90_triangle_hypotenuse_6sqrt2 :
  ∀ (a b c : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 3 * Real.sqrt 6 →
  c = 6 * Real.sqrt 2 →
  c = 2 * a →
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  intro a b c ha hb hc h2a
  sorry

end area_of_30_60_90_triangle_hypotenuse_6sqrt2_l335_335685


namespace cristine_final_lemons_l335_335851

def cristine_lemons_initial : ℕ := 12
def cristine_lemons_given_to_neighbor : ℕ := 1 / 4 * cristine_lemons_initial
def cristine_lemons_left_after_giving : ℕ := cristine_lemons_initial - cristine_lemons_given_to_neighbor
def cristine_lemons_exchanged_for_oranges : ℕ := 1 / 3 * cristine_lemons_left_after_giving
def cristine_lemons_left_after_exchange : ℕ := cristine_lemons_left_after_giving - cristine_lemons_exchanged_for_oranges

theorem cristine_final_lemons : cristine_lemons_left_after_exchange = 6 :=
by
  sorry

end cristine_final_lemons_l335_335851


namespace inequality_xyz_equality_condition_l335_335549

theorem inequality_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ 2 + x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  (x + y + z = 2 + x * y * z) ↔ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨
                                                  (x = 0 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 0 ∧ z = 1) ∨
                                                  (x = -1 ∧ y = 1 ∧ z = 0) :=
sorry

end inequality_xyz_equality_condition_l335_335549


namespace min_value_of_sin_x_plus_sin_z_thm_l335_335881

noncomputable def min_value_of_sin_x_plus_sin_z 
    (x y z : ℝ) 
    (h1 : sqrt 3 * Real.cos x = Real.cot y) 
    (h2 : 2 * Real.cos y = Real.tan z) 
    (h3 : Real.cos z = 2 * Real.cot x) : ℝ :=
  min (sin x + sin z)

theorem min_value_of_sin_x_plus_sin_z_thm 
    (x y z : ℝ)
    (h1 : sqrt 3 * Real.cos x = Real.cot y)
    (h2 : 2 * Real.cos y = Real.tan z)
    (h3 : Real.cos z = 2 * Real.cot x) :
  min_value_of_sin_x_plus_sin_z x y z h1 h2 h3 = -7 * sqrt 2 / 6 :=
sorry

end min_value_of_sin_x_plus_sin_z_thm_l335_335881


namespace sequence_b_n_l335_335057

theorem sequence_b_n (b : ℕ → ℝ) (h₁ : b 1 = 2) (h₂ : ∀ n, (b (n + 1))^3 = 64 * (b n)^3) : 
    b 50 = 2 * 4^49 :=
sorry

end sequence_b_n_l335_335057


namespace count_valid_n_l335_335544

theorem count_valid_n (h1 : ∀ n, n < 100) (h2 : ∀ m, (∃ k, k * (k + 1) = m) → m % 5 = 0) :
  ∃ n_set : Finset ℕ, (\sum n in n_set, 1) = 19 ∧ (∀ n ∈ n_set, ∃ m, (x^2 - n*x + m = 0) ∧ (∃ k, k * (k + 1) = m)) :=
by 
  sorry

end count_valid_n_l335_335544


namespace common_difference_is_3_l335_335404

noncomputable def whale_plankton_frenzy (x : ℝ) (y : ℝ) : Prop :=
  (9 * x + 36 * y = 450) ∧
  (x + 5 * y = 53)

theorem common_difference_is_3 :
  ∃ (x y : ℝ), whale_plankton_frenzy x y ∧ y = 3 :=
by {
  sorry
}

end common_difference_is_3_l335_335404


namespace no_eight_consecutive_odd_exponents_l335_335260

theorem no_eight_consecutive_odd_exponents :
  ¬ ∃ (n : ℕ), ∀ i : ℕ, i < 8 → (∀ p : ℕ, p.prime → odd (padic_val_nat p (n + i))) :=
by {
  sorry
}

end no_eight_consecutive_odd_exponents_l335_335260


namespace base7_digit_sum_l335_335181

theorem base7_digit_sum (A B C : ℕ) (hA : 1 ≤ A ∧ A < 7) (hB : 1 ≤ B ∧ B < 7) 
  (hC : 1 ≤ C ∧ C < 7) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eq : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) : 
  B + C = 6 := 
sorry

end base7_digit_sum_l335_335181


namespace fourth_individual_selected_is_16_l335_335505

-- Define the population
def population := List.range 20 |>.map (λ n => n + 1)  -- {1, 2, ..., 20}

-- Define the random number table
def random_table := [
  "1818", "0792", "4544", "1716", "5809", "7983", "8619",
  "6206", "7650", "0310", "5523", "6405", "0526", "6238"
]

-- Define a function to parse the random number table into a list of two-digit numbers
def parse_random_table (table : List String) : List Nat :=
  table.bind (λ row => row.toList.drop 2 |>.toArray.toArray.groupsOf 2 |>.map (Array.foldl (λ acc d => acc * 10 + (d.toNat - '0'.toNat)) 0))

-- Define the theorem statement
theorem fourth_individual_selected_is_16 : parse_random_table random_table |>.drop (4-1) |>.head = some 16 := 
by
  -- Placeholder for the proof
  sorry

end fourth_individual_selected_is_16_l335_335505


namespace determine_x_l335_335860

open Real

theorem determine_x (b x : ℝ) (h₀ : b > 1) (h₁ : x > 0) (h₂ : (4 * x) ^ log b 4 - (5 * x) ^ log b 5 = 0) : x = 4 / 5 :=
sorry

end determine_x_l335_335860


namespace token_exits_at_A2_l335_335971

-- Define the grid and its properties
inductive Row : Type
| A | B | C | D

inductive Col : Type
| col1 | col2 | col3 | col4

structure Cell : Type :=
(row : Row)
(col : Col)

-- Initial condition
def initial_position : Cell := { row := Row.C, col := Col.col2 }

-- Define the exit cell
def exit_cell : Cell := { row := Row.A, col := Col.col2 }

-- The theorem to prove
theorem token_exits_at_A2 :
  ∃ cell : Cell, cell = exit_cell ∧ token_exits_grid_from cell :=
sorry

end token_exits_at_A2_l335_335971


namespace range_of_k_l335_335380

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ (x, y) = (1, 2)) →
  (3 < k ∧ k < 7) :=
by
  intros hxy
  sorry

end range_of_k_l335_335380


namespace one_of_ten_is_one_l335_335756

theorem one_of_ten_is_one (a : ℕ → ℚ) (h_distinct: function.injective a)
  (h_sum: let odd_sum := ((finset.sum (finset.powerset_len 1 (finset.range 10))) +
                         (finset.sum (finset.powerset_len 3 (finset.range 10))) +
                         (finset.sum (finset.powerset_len 5 (finset.range 10))) +
                         (finset.sum (finset.powerset_len 7 (finset.range 10))) +
                         (finset.sum (finset.powerset_len 9 (finset.range 10))))
          let even_sum := ((finset.sum (finset.powerset_len 2 (finset.range 10))) +
                          (finset.sum (finset.powerset_len 4 (finset.range 10))) +
                          (finset.sum (finset.powerset_len 6 (finset.range 10))) +
                          (finset.sum (finset.powerset_len 8 (finset.range 10))) +
                          (finset.sum (finset.powerset_len 10 (finset.range 10))))
          (odd_sum = even_sum + 1)) :
  (∃ i, i < 10 ∧ a i = 1) :=
by {
  sorry
}

end one_of_ten_is_one_l335_335756


namespace measure_angle_YPZ_l335_335594

-- Define the conditions, including the triangle and its properties
variables {P X Y Z M N O : Type} [triangle : Triangle X Y Z] 
  (XM : Altitude X M) (YN : Altitude Y N) (ZO : Altitude Z O)
  (P_ortho : Orthocenter P X Y Z) 

-- Define the angles provided in the conditions
variables (angle_XYZ : Angle X Y Z) (angle_XZY : Angle X Z Y)
  [angle_XYZ_val : angle_XYZ = 65] [angle_XZY_val : angle_XZY = 37]

-- Goal to prove
theorem measure_angle_YPZ : angle_Y P Z = 143 := by
  sorry

end measure_angle_YPZ_l335_335594


namespace ratio_of_areas_ABC_DEF_l335_335503

-- Definitions of the points dividing the segments of triangle ABC
def PointOnSegment (A B P : Point) (ratio : ℚ) : Prop :=
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧ P = (1 - t) • A + t • B ∧ ratio = t / (1 - t)

variables {A B C D E F : Point}

-- Conditions
axiom hD : PointOnSegment A B D (1/2)
axiom hE : PointOnSegment B C E (1/2)
axiom hF : PointOnSegment C A F (1/2)

-- Goal
theorem ratio_of_areas_ABC_DEF : (area A B C) / (area D E F) = 27 / 8 :=
sorry

end ratio_of_areas_ABC_DEF_l335_335503


namespace diamond_expression_calculation_l335_335848

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_expression_calculation :
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 :=
by
  sorry

end diamond_expression_calculation_l335_335848


namespace largest_prime_factor_of_expression_l335_335737

theorem largest_prime_factor_of_expression :
  ∃ p : ℕ, prime p ∧ p = 241 ∧ ∀ q : ℕ, q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → prime q → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l335_335737


namespace B_finishes_remaining_work_in_3_days_l335_335743

theorem B_finishes_remaining_work_in_3_days
  (A_works_in : ℕ)
  (B_works_in : ℕ)
  (work_days_together : ℕ)
  (A_leaves : A_works_in = 4)
  (B_leaves : B_works_in = 10)
  (work_days : work_days_together = 2) :
  ∃ days_remaining : ℕ, days_remaining = 3 :=
by
  sorry

end B_finishes_remaining_work_in_3_days_l335_335743


namespace tangent_line_at_point_e_tangent_line_from_origin_l335_335530

-- Problem 1
theorem tangent_line_at_point_e (x y : ℝ) (h : y = Real.exp x) (h_e : x = Real.exp 1) :
    (Real.exp x) * x - y - Real.exp (x + 1) = 0 :=
sorry

-- Problem 2
theorem tangent_line_from_origin (x y : ℝ) (h : y = Real.exp x) :
    x = 1 →  Real.exp x * x - y = 0 :=
sorry

end tangent_line_at_point_e_tangent_line_from_origin_l335_335530


namespace intersection_M_N_l335_335119

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l335_335119


namespace count_integers_satisfying_conditions_l335_335945

theorem count_integers_satisfying_conditions :
  {n : ℤ | 200 < n ∧ n < 300 ∧ (∃ r : ℤ, n % 5 = r ∧ n % 7 = r)}.to_finset.card = 15 :=
by
  sorry

end count_integers_satisfying_conditions_l335_335945


namespace unique_line_passes_through_A_parallel_to_a_and_in_alpha_l335_335521

noncomputable def Point : Type := sorry
noncomputable def Line : Type := sorry
noncomputable def Plane : Type := sorry

variables (a : Line) (α : Plane) (A : Point)

-- Conditions
axiom line_parallel_plane (l : Line) (p : Plane) : Prop := sorry
axiom point_on_plane (pt : Point) (p : Plane) : Prop := sorry
axiom line_on_plane (l : Line) (p : Plane) : Prop := sorry
axiom line_parallel_line (l₁ l₂ : Line) : Prop := sorry
axiom passes_through_point (l : Line) (pt : Point) : Prop := sorry

-- The conditions given in the problem
axiom hyp_a_parallel_alpha : line_parallel_plane a α
axiom hyp_A_on_alpha : point_on_plane A α

-- Statement to prove
theorem unique_line_passes_through_A_parallel_to_a_and_in_alpha :
  ∃! (l : Line), passes_through_point l A ∧ line_parallel_line l a ∧ line_on_plane l α :=
sorry

end unique_line_passes_through_A_parallel_to_a_and_in_alpha_l335_335521


namespace num_words_at_least_one_vowel_l335_335173

-- Definitions based on conditions.
def letters : List Char := ['A', 'B', 'E', 'G', 'H']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'G', 'H']

-- The main statement posing the question and answer.
theorem num_words_at_least_one_vowel :
  let total_words := (letters.length) ^ 5
  let consonant_words := (consonants.length) ^ 5
  let result := total_words - consonant_words
  result = 2882 :=
by {
  let total_words := 5 ^ 5
  let consonant_words := 3 ^ 5
  let result := total_words - consonant_words
  have : result = 2882 := by sorry
  exact this
}

end num_words_at_least_one_vowel_l335_335173


namespace odd_function_derivative_condition_l335_335906

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def derivative_exists (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, f' x = (fderiv ℝ f x : ℝ)

theorem odd_function_derivative_condition
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_derivative : derivative_exists f f')
  (h_condition : ∀ x, x > 0 → x * f' x + 2 * f x > 0)
  (h_f2_eq_zero : f 2 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 2} :=
sorry

end odd_function_derivative_condition_l335_335906


namespace collatz_ninth_term_7_values_l335_335192

-- Define the Collatz function
def collatz : ℕ → ℕ
| n := if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the n-th term in the Collatz sequence
def collatz_seq (n : ℕ) : ℕ → ℕ
| 0 := n
| (k+1) := collatz (collatz_seq k)

-- Define the predicate that the 9th term in the sequence is 1
def collatz_ninth_term_is_one (n : ℕ) : Prop :=
  collatz_seq n 8 = 1

-- Prove that there are exactly 7 such numbers n
theorem collatz_ninth_term_7_values :
  ∃ s : Finset ℕ, s.card = 7 ∧ ∀ n, collatz_ninth_term_is_one n ↔ n ∈ s :=
sorry

end collatz_ninth_term_7_values_l335_335192


namespace evaluate_polynomial_at_2_l335_335460

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 1) = 31 := 
by 
  sorry

end evaluate_polynomial_at_2_l335_335460


namespace total_weight_remaining_eggs_l335_335623

theorem total_weight_remaining_eggs :
  let large_egg_weight := 14
  let medium_egg_weight := 10
  let small_egg_weight := 6

  let box_A_weight := 4 * large_egg_weight + 2 * medium_egg_weight
  let box_B_weight := 6 * small_egg_weight + 2 * large_egg_weight
  let box_C_weight := 4 * large_egg_weight + 3 * medium_egg_weight
  let box_D_weight := 4 * medium_egg_weight + 4 * small_egg_weight
  let box_E_weight := 4 * small_egg_weight + 2 * medium_egg_weight

  total_weight := box_A_weight + box_C_weight + box_D_weight + box_E_weight
  total_weight = 270 := 
by
  sorry

end total_weight_remaining_eggs_l335_335623


namespace problem_I_problem_II_1_problem_II_2_l335_335573

section
variables (boys_A girls_A boys_B girls_B : ℕ)
variables (total_students : ℕ)

-- Define the conditions
def conditions : Prop :=
  boys_A = 2 ∧ girls_A = 1 ∧ boys_B = 3 ∧ girls_B = 2 ∧ total_students = boys_A + girls_A + boys_B + girls_B

-- Problem (I)
theorem problem_I (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ arrangements, arrangements = 14400 := sorry

-- Problem (II.1)
theorem problem_II_1 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 13 / 14 := sorry

-- Problem (II.2)
theorem problem_II_2 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 6 / 35 := sorry
end

end problem_I_problem_II_1_problem_II_2_l335_335573


namespace monotonic_increase_interval_l335_335687

noncomputable def interval_of_monotonic_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
is_monotone_increasing f (Ioo a b)

def my_function (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_increase_interval :
  interval_of_monotonic_increase my_function (Real.sqrt 3 / 3) ⊤ :=
sorry

end monotonic_increase_interval_l335_335687


namespace smallest_degree_q_for_horizontal_asymptote_l335_335859

theorem smallest_degree_q_for_horizontal_asymptote :
  ∀ (q : polynomial ℝ), (∃ n : ℕ, q.degree = n ∧ (n ≥ 5)) ↔ ∃ q : polynomial ℝ, q.degree = 5 :=
sorry

end smallest_degree_q_for_horizontal_asymptote_l335_335859


namespace edward_spent_amount_l335_335451

-- Definitions based on the problem conditions
def initial_amount : ℕ := 18
def remaining_amount : ℕ := 2

-- The statement to prove: Edward spent $16
theorem edward_spent_amount : initial_amount - remaining_amount = 16 := by
  sorry

end edward_spent_amount_l335_335451


namespace depth_of_melted_ice_cream_l335_335799

theorem depth_of_melted_ice_cream (r_sphere r_cylinder : ℝ) (Vs : ℝ) (Vc : ℝ) :
  r_sphere = 3 →
  r_cylinder = 12 →
  Vs = (4 / 3) * Real.pi * r_sphere^3 →
  Vc = Real.pi * r_cylinder^2 * (1 / 4) →
  Vs = Vc →
  (1 / 4) = 1 / 4 := 
by
  intros hr_sphere hr_cylinder hVs hVc hVs_eq_Vc
  sorry

end depth_of_melted_ice_cream_l335_335799


namespace largest_prime_factor_l335_335735

-- Define the expressions given in the problem
def expression := 16^4 + 2 * 16^2 + 1 - 15^4

-- State the problem of finding the largest prime factor
theorem largest_prime_factor : ∃ p : ℕ, nat.prime p ∧ p ∣ expression ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ expression → q ≤ 241 :=
by {
  sorry  -- Proof needed
}

end largest_prime_factor_l335_335735


namespace normal_distribution_probability_l335_335911

variable {X : Type}

def normal_distribution (mean : ℝ) (variance : ℝ) (x : ℝ) : Type := sorry

theorem normal_distribution_probability :
  normal_distribution 102 16 X →
  (P (102 - 3 * 4 < X ∧ X ≤ 102 + 3 * 4) = 0.9974) →
  (P (X > 114) = 0.0013) :=
sorry

end normal_distribution_probability_l335_335911


namespace tangent_line_intersects_x_axis_at_neg_half_l335_335321

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp(x) + 2 * x + 1

def is_tangent_to_curve (line : ℝ → ℝ) (curve : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ k b, ∀ x, line x = k * x + b ∧ line x = curve x

theorem tangent_line_intersects_x_axis_at_neg_half :
  let P := (0 : ℝ, 1 : ℝ),
      line (x : ℝ) := 2 * x - 1 in 
  is_tangent_to_curve line f P → 
  ∃ x : ℝ, line x = 0 ∧ x = -1/2 := 
by 
  simp [is_tangent_to_curve, f]
  sorry

end tangent_line_intersects_x_axis_at_neg_half_l335_335321


namespace total_passengers_per_day_l335_335811

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l335_335811


namespace cost_of_50_tulips_l335_335824

theorem cost_of_50_tulips (c : ℕ → ℝ) :
  (∀ n : ℕ, n ≤ 40 → c n = n * (36 / 18)) ∧
  (∀ n : ℕ, n > 40 → c n = (40 * (36 / 18) + (n - 40) * (36 / 18)) * 0.9) ∧
  (c 18 = 36) →
  c 50 = 90 := sorry

end cost_of_50_tulips_l335_335824


namespace ratio_of_time_spent_l335_335195

theorem ratio_of_time_spent {total_minutes type_a_minutes type_b_minutes : ℝ}
  (h1 : total_minutes = 180)
  (h2 : type_a_minutes = 32.73)
  (h3 : type_b_minutes = total_minutes - type_a_minutes) :
  type_a_minutes / type_a_minutes = 1 ∧ type_b_minutes / type_a_minutes = 4.5 := by
  sorry

end ratio_of_time_spent_l335_335195


namespace distinct_light_arrangements_l335_335292

/-- The number of distinct arrangements of lights in a 3x2 grid, given at least one button is lit. -/
theorem distinct_light_arrangements : 
  let grid := fin 3 × fin 2 in
  let configurations := { f : grid → bool // ∃ p, f p = tt } in
  let distinct_configurations := (quotient.mk _) '' configurations in
  fintype.card distinct_configurations = 44 :=
by
  sorry  -- Proof omitted

end distinct_light_arrangements_l335_335292


namespace value_of_expression_l335_335094

theorem value_of_expression (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : 
  (x - 2 * y + 3 * z) / (x + y + z) = 8 / 9 := 
  sorry

end value_of_expression_l335_335094


namespace determinant_zero_l335_335067

noncomputable theory
open_locale matrix

def matrix_A (α β : ℝ) : matrix (fin 3) (fin 3) ℝ :=
  ![![0, real.cos α, -real.sin α], 
    ![-real.cos α, 0, real.cos β], 
    ![real.sin α, -real.cos β, 0]]

theorem determinant_zero (α β : ℝ) : 
  matrix.det (matrix_A α β) = 0 :=
by {
  sorry
}

end determinant_zero_l335_335067


namespace incorrect_judgment_D_l335_335893

theorem incorrect_judgment_D (p q : Prop) (hp : p = (2 + 3 = 5)) (hq : q = (5 < 4)) : 
  ¬((p ∧ q) ∧ (p ∨ q)) := by 
    sorry

end incorrect_judgment_D_l335_335893


namespace points_distances_1000_at_least_6000_l335_335597

theorem points_distances_1000_at_least_6000 : 
  ∃ (points : fin 1000 → ℝ × ℝ), 
    ∃ (d : ℝ), 
      (∀ i j, i ≠ j → (dist (points i) (points j) = d) → count (λ (i j), dist (points i) (points j) = d) ≥ 6000) :=
sorry

end points_distances_1000_at_least_6000_l335_335597


namespace plane_intersects_tetrahedron_surface_proportional_volumes_iff_incenter_l335_335789

theorem plane_intersects_tetrahedron_surface_proportional_volumes_iff_incenter
  (tetrahedron : Type)
  (V S r V1 S1 r1 : ℝ)
  (plane : Type)
  (intersects_edges : plane → tetrahedron → Prop) :
  (∃ plane, intersects_edges plane tetrahedron ∧ r = r1) ↔
  (∃ plane, intersects_edges plane tetrahedron ∧ V/S = V1/S1) :=
by
sorRFy

end plane_intersects_tetrahedron_surface_proportional_volumes_iff_incenter_l335_335789


namespace alien_home_planet_people_count_l335_335814

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end alien_home_planet_people_count_l335_335814


namespace sum_of_roots_l335_335966

theorem sum_of_roots : (x₁ x₂ : ℝ) → (h : 2 * x₁^2 + 6 * x₁ - 1 = 0) → (h₂ : 2 * x₂^2 + 6 * x₂ - 1 = 0) → x₁ + x₂ = -3 :=
by 
  sorry

end sum_of_roots_l335_335966


namespace max_elevator_distance_l335_335194

theorem max_elevator_distance :
  ∃ (floors : list ℕ), 
  floors.length = 11 ∧ 
  floors.head = 0 ∧ 
  floors.last = some 10 ∧ 
  (∀ i, i < 10 → (floors.nth i).isSome ∧ (floors.nth (i + 1)).isSome → (floors.nth i).get = (floors.nth (i + 1)).get + 1 ∨ (floors.nth i).get = (floors.nth (i + 1)).get - 1) ∧ 
  list.sum (list.map (λ n, 4) (list.filter (λ n, n < 10) floors.tail)) = 216 := 
sorry

end max_elevator_distance_l335_335194


namespace older_grandchild_pancakes_eaten_l335_335172

theorem older_grandchild_pancakes_eaten (initial_pancakes : ℕ) (remaining_pancakes : ℕ)
  (younger_eat_per_cycle : ℕ) (older_eat_per_cycle : ℕ) (bake_per_cycle : ℕ)
  (n : ℕ) 
  (h_initial : initial_pancakes = 19)
  (h_remaining : remaining_pancakes = 11)
  (h_younger_eat : younger_eat_per_cycle = 1)
  (h_older_eat : older_eat_per_cycle = 3)
  (h_bake : bake_per_cycle = 2)
  (h_reduction : initial_pancakes - remaining_pancakes = n * (younger_eat_per_cycle + older_eat_per_cycle - bake_per_cycle)) :
  older_eat_per_cycle * n = 12 :=
begin
  sorry
end

end older_grandchild_pancakes_eaten_l335_335172


namespace find_number_l335_335879

theorem find_number (x : ℕ) (n : ℕ) (h1 : x = 4) (h2 : x + n = 5) : n = 1 :=
by
  sorry

end find_number_l335_335879


namespace polynomial_as_difference_of_monotonic_polynomials_l335_335645

theorem polynomial_as_difference_of_monotonic_polynomials
  (f : Polynomial ℝ) :
  ∃ (F G : Polynomial ℝ), 
    (∀ x, 0 ≤ Polynomial.derivative F.eval x) ∧ 
    (∀ x, 0 ≤ Polynomial.derivative G.eval x) ∧ 
    (f = F - G) :=
by
  sorry

end polynomial_as_difference_of_monotonic_polynomials_l335_335645


namespace fifth_hexagon_dots_l335_335090

-- Definitions as per conditions
def dots_in_nth_layer (n : ℕ) : ℕ := 6 * (n + 2)

-- Function to calculate the total number of dots in the nth hexagon
def total_dots_in_hexagon (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + dots_in_nth_layer k) (dots_in_nth_layer 0)

-- The proof problem statement
theorem fifth_hexagon_dots : total_dots_in_hexagon 5 = 150 := sorry

end fifth_hexagon_dots_l335_335090


namespace area_of_triangle_of_tangency_points_l335_335832

-- Definition of the radii of the three circles
def r1 : ℝ := 2
def r2 : ℝ := 3
def r3 : ℝ := 4

-- Definition of the distances between the centers of the circles
def d12 : ℝ := r1 + r2
def d23 : ℝ := r2 + r3
def d13 : ℝ := r1 + r3

-- The semi-perimeter of the triangle formed by the centers
def semi_perimeter : ℝ := (d12 + d23 + d13) / 2

-- Using Heron's formula to calculate the area of the triangle formed by the centers
def area_triangle_centers : ℝ := real.sqrt (semi_perimeter * (semi_perimeter - d12) * (semi_perimeter - d23) * (semi_perimeter - d13))

-- The inradius of the triangle
def inradius : ℝ := area_triangle_centers / semi_perimeter

-- The expected area of the triangle formed by the points of tangency
def expected_area : ℝ := 6 * real.sqrt 6

-- Statement of the proof problem
theorem area_of_triangle_of_tangency_points : 
  ∃ (area : ℝ), area = expected_area :=
sorry

end area_of_triangle_of_tangency_points_l335_335832


namespace find_f_of_f_of_one_minus_i_l335_335891

def is_real (x : ℂ) : Prop := x.im = 0
def is_not_real (x : ℂ) : Prop := ¬ is_real x

noncomputable def f (x : ℂ) : ℂ :=
  if is_real x then 1 + x else (1 + complex.I) * x

theorem find_f_of_f_of_one_minus_i : f (f (1 - complex.I)) = 3 :=
by
  sorry

end find_f_of_f_of_one_minus_i_l335_335891


namespace angle_AFE_eq_AEF_l335_335034

noncomputable theory

-- Definitions based on conditions
variables {A B C D E F : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E] [DecidableEq F]
variable [Square ABCD]   -- ABCD is a square
variable [Parallel BE AC]  -- BE ∥ AC
variable [Equidistant AC CE] -- AC = CE
variable [Intersects (Line.segment EC) (Line.segment BA) F] -- extension of EC intersects extension of BA at F

-- Statement to prove
theorem angle_AFE_eq_AEF (ABCD_square: Square ABCD) (BE_parallel_AC: Parallel BE AC) (AC_eq_CE: Equidistant AC CE) 
    (EC_ext_BA_int_F: Intersects (Line.segment EC) (Line.segment BA) F) : Angle AFE = Angle AEF := 
  by
    sorry

end angle_AFE_eq_AEF_l335_335034


namespace expression_value_l335_335912

theorem expression_value (x y : ℝ) (h : x - 2 * y = 3) : 1 - 2 * x + 4 * y = -5 :=
by
  sorry

end expression_value_l335_335912


namespace marina_drive_l335_335626

theorem marina_drive (a b c : ℕ) (x : ℕ) 
  (h1 : 1 ≤ a) 
  (h2 : a + b + c ≤ 9)
  (h3 : 90 * (b - a) = 60 * x)
  (h4 : x = 3 * (b - a) / 2) :
  a = 1 ∧ b = 3 ∧ c = 5 ∧ a^2 + b^2 + c^2 = 35 :=
by {
  sorry
}

end marina_drive_l335_335626


namespace infections_first_wave_l335_335309

theorem infections_first_wave (x : ℕ)
  (h1 : 4 * x * 14 = 21000) : x = 375 :=
  sorry

end infections_first_wave_l335_335309


namespace intersection_M_N_l335_335139

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l335_335139


namespace total_emails_received_l335_335765

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l335_335765


namespace find_x_when_y_is_6_l335_335364

-- Condition for inverse variation
def inverse_var (k y : ℝ) (x : ℝ) : Prop := x = k / y^2

-- Given values
def given_value_x : ℝ := 1
def given_value_y : ℝ := 2
def new_value_y : ℝ := 6

-- The theorem to prove
theorem find_x_when_y_is_6 :
  ∃ k, inverse_var k given_value_y given_value_x → inverse_var k new_value_y (1/9) :=
by
  sorry

end find_x_when_y_is_6_l335_335364


namespace proof_problem_l335_335216

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (AB AC BC a b : vector A)

noncomputable def condition_triangle : Prop :=
∀ (A B C : A),
  dist B A = 2 ∧
  dist C A = 2 ∧
  dist B C = 2

noncomputable def condition_vectors : Prop :=
  ∃ (a b : vector A),
  AB = 2 • a ∧ AC = 2 • a + b

noncomputable def conclusion_a_unit_vector (a : vector A) : Prop :=
  ∥a∥ = 1

noncomputable def conclusion_b_parallel_BC (b BC : vector A) : Prop :=
  b = BC

noncomputable def conclusion_4a_plus_b_perp_BC (a b BC : vector A) : Prop :=
  (4 • a + b) ⬝ BC = 0

theorem proof_problem (h_triangle : condition_triangle A B C) (h_vectors : condition_vectors A B C a b) :
  conclusion_a_unit_vector a ∧ conclusion_b_parallel_BC b BC ∧ conclusion_4a_plus_b_perp_BC a b BC :=
by
  sorry

end proof_problem_l335_335216


namespace fahrenheit_to_celsius_conversion_l335_335720

theorem fahrenheit_to_celsius_conversion (F : ℤ) (hF : F = 100) : 
    (let C := (5 / 9 : ℚ) * (F - 32) in C ≈ 37.8) := by
  sorry

end fahrenheit_to_celsius_conversion_l335_335720


namespace problem1_problem2_problem3_l335_335938

-- Define the sets and their operations
def U := Set.univ ℝ
def A := {x : ℝ | 2 < x ∧ x < 9}
def B := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 2 - a}
def complement_U (s : Set ℝ) := {x : ℝ | x ∉ s}

-- Define the statements we need to prove
theorem problem1 :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} :=
sorry

theorem problem2 :
  B ∪ complement_U A = {x : ℝ | x ≤ 5 ∨ 9 ≤ x} :=
sorry

theorem problem3 (a : ℝ) :
  (C a ∪ complement_U B = U) → a ≤ -3 :=
sorry

end problem1_problem2_problem3_l335_335938


namespace fourth_largest_divisor_correct_l335_335679

def n : ℕ := 3430000000

def fourth_largest_divisor (n : ℕ) : ℕ :=
  let divisors := (List.range (n + 1)).filter (λ d, n % d = 0)
  (List.reverse divisors).nth 3

theorem fourth_largest_divisor_correct : fourth_largest_divisor n = 428750000 := by
  sorry

end fourth_largest_divisor_correct_l335_335679


namespace range_of_m_l335_335921

def f (x : ℝ) : ℝ := 2 * real.sqrt x + real.sqrt (5 - x)

theorem range_of_m (m : ℝ) : (∀ x ∈ set.Icc (0 : ℝ) 5, f x ≤ |m - 2|) ↔ m ∈ set.Iic (-3) ∪ set.Ici 7 :=
by {
  sorry
}

end range_of_m_l335_335921


namespace find_AB_l335_335985

variables (A B C D P : Type)
variable [plane_geometry A B C D P] 

-- Conditions
variables [is_rectangle A B C D]
variable [point_on_side P (side BC)]
variables (BP CP : ℝ)
variable (BP_eq : BP = 18)
variable (CP_eq : CP = 9)
variable (tan_angle_APD_eq : tan (angle A P D) = 2)

-- Variable to find
noncomputable def AB : ℝ :=
  sqrt 175.5

-- Theorem to prove
theorem find_AB :
  BP = 18 → CP = 9 → tan (angle A P D) = 2 → AB = sqrt 175.5 :=
sorry

end find_AB_l335_335985


namespace ratio_of_areas_l335_335714

theorem ratio_of_areas (A H I B C D E F G : Type) 
  [equilateral_triangle A H I] 
  (BC_parallel : ∀ BC, BC || HI)
  (DE_parallel : ∀ DE, DE || HI)
  (FG_parallel : ∀ FG, FG || HI)
  (AB_eq_BD_eq_DF_eq_FH : AB = BD ∧ BD = DF ∧ DF = FH)
  (AF_eq_fifth_sixth_AH : AF = (5/6) * AH) 
  : (area_trapezoid FGIH / area_triangle AHI) = 11/36 := sorry

end ratio_of_areas_l335_335714


namespace square_side_length_l335_335800

theorem square_side_length (s : ℝ) (h1 : 4 * s = 12) (h2 : s^2 = 9) : s = 3 :=
sorry

end square_side_length_l335_335800


namespace quadratic_condition_l335_335265

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end quadratic_condition_l335_335265


namespace number_of_M_partitions_l335_335229

-- Definitions for A and M
def A : Finset ℕ := Finset.range 2002 |>.image (λ x => x + 1)
def M : Set ℕ := {1001, 2003, 3005}

-- Definition of M-free set
def MFreeSet (B : Finset ℕ) : Prop :=
  ∀ {m n : ℕ}, m ∈ B → n ∈ B → m + n ∉ M

-- Definition of M-partition
def MPartition (A1 A2 : Finset ℕ) : Prop :=
  A1 ∪ A2 = A ∧ A1 ∩ A2 = ∅ ∧ MFreeSet A1 ∧ MFreeSet A2

-- The theorem to prove
theorem number_of_M_partitions :
  ∃ n : ℕ, n = 2 ^ 501 :=
sorry

end number_of_M_partitions_l335_335229


namespace triangle_area_60_triangle_area_120_l335_335646

variable (a b c : ℝ)

def area_60 (α : ℝ) (a b c : ℝ) : ℝ :=
  if α = 60 then (√3 / 4) * (a^2 - (b - c)^2) else 0

def area_120 (α : ℝ) (a b c : ℝ) : ℝ :=
  if α = 120 then (√3 / 12) * (a^2 - (b - c)^2) else 0

theorem triangle_area_60 (α : ℝ) (a b c : ℝ) : α = 60 → area_60 α a b c = (√3 / 4) * (a^2 - (b - c)^2) :=
by sorry

theorem triangle_area_120 (α : ℝ) (a b c : ℝ) : α = 120 → area_120 α a b c = (√3 / 12) * (a^2 - (b - c)^2) :=
by sorry

end triangle_area_60_triangle_area_120_l335_335646


namespace area_of_triangle_ACD_is_64_l335_335213

-- Define the given conditions
variables {A B C D : Type} [EuclideanGeometry A B C D]

noncomputable def length (A B : Type) : Real :=
match (A, B) with
  | (8, 8) => 8  -- Example values as given, usually would use a distance function

# Check if triangles can be isosceles
def is_isosceles_right {A B C : Type} (a b c : length A B) : Prop :=
(a = b ∧ a * sqrt 2 = c) ∨ (a = c ∧ a * sqrt 2 = b) 

# Define the triangles being isosceles right triangles
def triangle_ABC_is_isosceles_right (a b : length A B) : Prop :=
is_isosceles_right a b (length A C)

def triangle_BCD_is_isosceles_right (b d : length B D) : Prop :=
is_isosceles_right b d (length B C)

-- Create a proposition for the area of the triangle
def area_triangle_ACD (a b c : length A B) : Real :=
1 / 2 * (a * sqrt 2) * (b * sqrt 2)

-- Prove the area of triangle ACD to be 64 square units
theorem area_of_triangle_ACD_is_64 {A B C D : Type} (a b d : length A B) :
  triangle_ABC_is_isosceles_right a b → 
  triangle_BCD_is_isosceles_right b d →
  area_triangle_ACD a b d = 64 :=
by
  sorry

end area_of_triangle_ACD_is_64_l335_335213


namespace sum_first_1001_terms_l335_335516

-- Define the arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := 2 + (n - 2) * 1

-- Define the sequence b_n which is the floor of the logarithm base 10 of a_n
def b_n (n : ℕ) : ℤ := int.floor (real.log10 (a_n n))

-- Define the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℤ := (finset.range n).sum b_n

-- The statement to be proved
theorem sum_first_1001_terms : T 1001 = 1896 :=
by sorry

end sum_first_1001_terms_l335_335516


namespace intersection_M_N_l335_335131

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335131


namespace product_of_numbers_l335_335744

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := sorry

end product_of_numbers_l335_335744


namespace cat_clothing_probability_l335_335665

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end cat_clothing_probability_l335_335665


namespace shorts_cost_l335_335358

theorem shorts_cost :
  let football_cost := 3.75
  let shoes_cost := 11.85
  let zachary_money := 10
  let additional_needed := 8
  ∃ S, football_cost + shoes_cost + S = zachary_money + additional_needed ∧ S = 2.40 :=
by
  sorry

end shorts_cost_l335_335358


namespace cost_of_each_croissant_l335_335280

theorem cost_of_each_croissant 
  (quiches_price : ℝ) (num_quiches : ℕ) (each_quiche_cost : ℝ)
  (buttermilk_biscuits_price : ℝ) (num_biscuits : ℕ) (each_biscuit_cost : ℝ)
  (total_cost_with_discount : ℝ) (discount_rate : ℝ)
  (num_croissants : ℕ) (croissant_price : ℝ) :
  quiches_price = num_quiches * each_quiche_cost →
  each_quiche_cost = 15 →
  num_quiches = 2 →
  buttermilk_biscuits_price = num_biscuits * each_biscuit_cost →
  each_biscuit_cost = 2 →
  num_biscuits = 6 →
  discount_rate = 0.10 →
  (quiches_price + buttermilk_biscuits_price + (num_croissants * croissant_price)) * (1 - discount_rate) = total_cost_with_discount →
  total_cost_with_discount = 54 →
  num_croissants = 6 →
  croissant_price = 3 :=
sorry

end cost_of_each_croissant_l335_335280


namespace trains_meet_in_approx_17_45_seconds_l335_335750

noncomputable def train_meet_time
  (length1 length2 distance_between : ℕ)
  (speed1_kmph speed2_kmph : ℕ)
  : ℕ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_approx_17_45_seconds :
  train_meet_time 100 200 660 90 108 = 17 := by
  sorry

end trains_meet_in_approx_17_45_seconds_l335_335750


namespace longest_path_is_critical_path_l335_335581

noncomputable def longest_path_in_workflow_diagram : String :=
"Critical Path"

theorem longest_path_is_critical_path :
  (longest_path_in_workflow_diagram = "Critical Path") :=
  by
  sorry

end longest_path_is_critical_path_l335_335581


namespace problem1_problem2_l335_335828

-- Given conditions
variables (x y : ℝ)

-- Problem 1: Prove that ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = -xy
theorem problem1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = - (x * y) :=
sorry

-- Problem 2: Prove that (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2
theorem problem2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
sorry

end problem1_problem2_l335_335828


namespace cover_rectangle_with_polyomino_l335_335058

-- Defining the conditions under which the m x n rectangle can be covered by the given polyomino
theorem cover_rectangle_with_polyomino (m n : ℕ) :
  (6 ∣ (m * n)) →
  (m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) →
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) →
  ((3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ (m * n))) :=
sorry

end cover_rectangle_with_polyomino_l335_335058


namespace machine_production_in_10_seconds_l335_335960

def items_per_minute : ℕ := 150
def seconds_per_minute : ℕ := 60
def production_rate_per_second : ℚ := items_per_minute / seconds_per_minute
def production_time_in_seconds : ℕ := 10
def expected_production_in_ten_seconds : ℚ := 25

theorem machine_production_in_10_seconds :
  (production_rate_per_second * production_time_in_seconds) = expected_production_in_ten_seconds :=
sorry

end machine_production_in_10_seconds_l335_335960


namespace abs_diff_A4_B4_l335_335074

-- Definitions and conditions based on the problem
def A4 : ℕ := 1  -- A4 is 1 in base 4
def B4 : ℕ := 2  -- B4 is 2 in base 4

-- Prove the absolute difference of digits under given constraints
theorem abs_diff_A4_B4 : abs (A4 - B4) = 1 := by 
  sorry

end abs_diff_A4_B4_l335_335074


namespace min_shoes_to_ensure_pair_l335_335199

theorem min_shoes_to_ensure_pair :
  ∀ (n : ℕ), (h1: n = 24) (black_pairs brown_pairs : ℕ) (h2: black_pairs = 12 ∧ brown_pairs = 12),
  ∃ k, k = 25 :=
by
  intros n h1 black_pairs brown_pairs h2
  use 25
  sorry

end min_shoes_to_ensure_pair_l335_335199


namespace thanksgiving_chocolates_l335_335261

theorem thanksgiving_chocolates (x : ℝ) :
  (let first_day_remaining := (3/5) * x - 3 in
   let second_day_remaining := first_day_remaining - ((1/4) * first_day_remaining + 5) in
   let third_day_remaining := second_day_remaining in
   third_day_remaining = 10) ↔ x = 105 :=
by
  let first_day_remaining := (3/5) * x - 3
  let second_day_remaining := first_day_remaining - ((1/4) * first_day_remaining + 5)
  let third_day_remaining := second_day_remaining
  have h1 : third_day_remaining = 10 → x = 105, sorry
  have h2 : x = 105 → third_day_remaining = 10, sorry
  exact ⟨h1, h2⟩

end thanksgiving_chocolates_l335_335261


namespace fisherman_caught_total_fish_l335_335773

noncomputable def number_of_boxes : ℕ := 15
noncomputable def fish_per_box : ℕ := 20
noncomputable def fish_outside_boxes : ℕ := 6

theorem fisherman_caught_total_fish :
  number_of_boxes * fish_per_box + fish_outside_boxes = 306 :=
by
  sorry

end fisherman_caught_total_fish_l335_335773


namespace ratio_girls_total_members_l335_335975

theorem ratio_girls_total_members {p_boy p_girl : ℚ} (h_prob_ratio : p_girl = (3/5) * p_boy) (h_total_prob : p_boy + p_girl = 1) :
  p_girl / (p_boy + p_girl) = 3 / 8 :=
by
  sorry

end ratio_girls_total_members_l335_335975


namespace path_of_length_in_graph_l335_335973

theorem path_of_length_in_graph {n k : ℕ} (G : SimpleGraph (Fin n))
  (h_e : G.edgeFinset.card > (n * k) / 2) : 
  ∃ (p : List (Fin n)), p.length = k + 2 ∧ (∀ i, i < k + 1 → G.adj (p.nthLe i _ ) (p.nthLe (i + 1) _)) := 
begin
  sorry
end

end path_of_length_in_graph_l335_335973


namespace correct_inverse_proportion_equation_l335_335731

-- Define the conditions
def equation_A (x : ℝ) : ℝ := 2 / x
def equation_B (x : ℝ) : ℝ := 2 * x + 1
def equation_C (x : ℝ) : ℝ := (1 / 2) * x^2
def equation_D (x : ℝ) : ℝ := x / 2

-- Define the property of inverse proportionality
def is_inverse_proportional (f : ℝ → ℝ) : Prop :=
∃ k : ℝ, ∀ x ≠ 0, f x = k / x

-- The theorem to prove
theorem correct_inverse_proportion_equation :
  is_inverse_proportional equation_A ∧ ¬ is_inverse_proportional equation_B ∧ ¬ is_inverse_proportional equation_C ∧ ¬ is_inverse_proportional equation_D :=
by
  sorry

end correct_inverse_proportion_equation_l335_335731


namespace magician_guarantee_success_l335_335780

-- Definitions based on the conditions in part a).
def deck_size : ℕ := 52

def is_edge_position (position : ℕ) : Prop :=
  position = 0 ∨ position = deck_size - 1

-- Statement of the proof problem in part c).
theorem magician_guarantee_success (position : ℕ) : is_edge_position position ↔ 
  forall spectator_strategy : ℕ → ℕ, 
  exists magician_strategy : (ℕ → ℕ → ℕ), 
  forall t : ℕ, t = position →
  (∃ k : ℕ, t = magician_strategy k (spectator_strategy k)) :=
sorry

end magician_guarantee_success_l335_335780


namespace households_without_car_or_bike_l335_335577

/--
In a neighborhood having 90 households, some did not have either a car or a bike.
If 16 households had both a car and a bike and 44 had a car, and
there were 35 households with a bike only.
Prove that there are 11 households that did not have either a car or a bike.
-/
theorem households_without_car_or_bike
  (total_households : ℕ)
  (both_car_and_bike : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : both_car_and_bike = 16)
  (H3 : car = 44)
  (H4 : bike_only = 35) :
  ∃ N : ℕ, N = total_households - (car - both_car_and_bike + bike_only + both_car_and_bike) ∧ N = 11 :=
by {
  sorry
}

end households_without_car_or_bike_l335_335577


namespace missed_questions_proof_l335_335745

def num_missed_questions : ℕ := 180

theorem missed_questions_proof (F : ℕ) (h1 : 5 * F + F = 216) : F = 36 ∧ 5 * F = num_missed_questions :=
by {
  sorry
}

end missed_questions_proof_l335_335745


namespace tan_positive_in_third_quadrant_l335_335183

theorem tan_positive_in_third_quadrant (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) :
  real.tan θ > 0 :=
sorry

end tan_positive_in_third_quadrant_l335_335183


namespace number_is_100_l335_335262

theorem number_is_100 (n : ℕ) 
  (hquot : n / 11 = 9) 
  (hrem : n % 11 = 1) : 
  n = 100 := 
by 
  sorry

end number_is_100_l335_335262


namespace find_tan_angle_QDE_l335_335643

noncomputable def tan_angle_QDE (θ : ℝ) (a b c : ℝ) (cosθ sinθ : ℝ): Prop :=
  (DE EF FD : ℝ) (DE = 15) (EF = 16) (FD = 17) : θ = QDE 
  ∧ θ = QEF ∧ θ = QFD
  ∧ (15a + 16b + 17c) * cosθ = 385
  ∧ (15a + 16b + 17c) * sinθ = 240

theorem find_tan_angle_QDE : 
  ∃ (θ a b c cosθ sinθ: ℝ), tan_angle_QDE θ a b c cosθ sinθ → tan θ = 48 / 77 := 
  by sorry

end find_tan_angle_QDE_l335_335643


namespace symmetric_line_equation_l335_335186

theorem symmetric_line_equation (l : ℝ × ℝ → Prop)
  (h1 : ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0)
  (h2 : ∀ p : ℝ × ℝ, l p ↔ p = (0, 2) ∨ p = ⟨-3, 2⟩) :
  ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0 :=
by
  sorry

end symmetric_line_equation_l335_335186


namespace generating_function_l335_335296

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 0 then 1 else (-1)^n * (n + 1)

theorem generating_function (G : ℤ → ℤ) :
  (∀ n, G n = sequence n) → ∑ n in ℕ, G n * x^n = (1 + x)^(-2) :=
by
  intro h
  sorry

end generating_function_l335_335296


namespace total_number_recruits_l335_335312

theorem total_number_recruits 
  (x y z : ℕ)
  (h1 : x = 50)
  (h2 : y = 100)
  (h3 : z = 170)
  (h4 : x = 4 * (y - 50) ∨ y = 4 * (z - 170) ∨ x = 4 * (z - 170)) : 
  171 + (z - 170) = 211 :=
by
  sorry

end total_number_recruits_l335_335312


namespace largest_prime_factor_of_expression_l335_335738

theorem largest_prime_factor_of_expression :
  ∃ p : ℕ, prime p ∧ p = 241 ∧ ∀ q : ℕ, q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → prime q → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l335_335738


namespace problem_conditions_l335_335902

def odd_function {α β : Type*} [AddGroup α] [HasNeg β] (f : α → β) : Prop :=
  ∀ x, f (-x) = -f x

def f : ℝ → ℝ := sorry
def f_periodic : ℝ → ℝ := sorry

theorem problem_conditions
  (h_odd : odd_function f)
  (h_property : ∀ x, f x = f (2 - x))
  (h_value : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = Real.log2 (x + 7))
  (h_f_periodic : ∀ x, f (x + 4) = f x)
  :
  f 2021 = 3
  :=
sorry

end problem_conditions_l335_335902


namespace coeff_a3b3_in_expression_correct_l335_335343

noncomputable def coeff_a3b3_in_expression : ℕ :=
  let coeff_in_a_b := Nat.choose 6 3
  let coeff_in_c_c_inv := Nat.choose 8 4
  coeff_in_a_b * coeff_in_c_c_inv

theorem coeff_a3b3_in_expression_correct : coeff_a3b3_in_expression = 1400 :=
by
  simp [coeff_a3b3_in_expression, Nat.choose]
  sorry -- Detailed computations go here

end coeff_a3b3_in_expression_correct_l335_335343


namespace plate_arrangement_correctness_l335_335005

noncomputable def valid_plate_arrangements : ℕ :=
  361

theorem plate_arrangement_correctness (blue_plates red_plates green_plates orange_plates : ℕ)
  (h_blue : blue_plates = 5)
  (h_red : red_plates = 3)
  (h_green : green_plates = 2)
  (h_orange : orange_plates = 1) :
  let total_arrangements := 
        (if blue_plates + red_plates + green_plates + orange_plates = 11 then valid_plate_arrangements else 0) in
  total_arrangements = 361 :=
by
  sorry

end plate_arrangement_correctness_l335_335005


namespace length_BD_fraction_AD_l335_335644

theorem length_BD_fraction_AD (A B C D : Point) 
  (h1 : dist A B = 5 * dist B D) 
  (h2 : dist A C = 8 * dist C D) :
  dist B D = (1 / 6) * dist A D := 
sorry

end length_BD_fraction_AD_l335_335644


namespace hyperbola_eccentricity_l335_335519

theorem hyperbola_eccentricity (a b c e : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : F = (-c, 0)) 
(h₄ : ∀ (l : ℝ → ℝ) (A B : ℝ × ℝ), A = (a, 0) ∧ B = (0, b) → 
(l (-c) = 0) → (∃ l, line_symmetric_about l A B)) : 
e = 1 + Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_l335_335519


namespace set_intersection_l335_335128

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l335_335128


namespace sum_complex_seq_l335_335913

noncomputable def complex_seq (a : ℕ → ℂ) : Prop :=
  (∀ n, a(n+1)^2 - a(n) * a(n+1) + a(n)^2 = 0) ∧
  (∀ n, n ≥ 1 → a(n+1) ≠ a(n-1)) ∧
  a(1) = 1

theorem sum_complex_seq :
  ∀ (a : ℕ → ℂ), complex_seq a → (∑ n in Finset.range 2007, a n) = 2 :=
by
  sorry

end sum_complex_seq_l335_335913


namespace combination_8_5_eq_56_l335_335838

theorem combination_8_5_eq_56 : nat.choose 8 5 = 56 :=
by
  sorry

end combination_8_5_eq_56_l335_335838


namespace length_major_axis_ellipse_l335_335596

-- Define the problem conditions
structure Cylinder where
  radius : ℝ

structure Sphere where
  radius : ℝ

structure CenterDistance where
  d : ℝ

-- Define the conditions for the given problem
def base_cylinder : Cylinder := { radius := 6 }
def sphere1 : Sphere := { radius := 6 }
def sphere2 : Sphere := { radius := 6 }
def distance_centers : CenterDistance := { d := 13 }

-- Define the theorem to state the length of the major axis
theorem length_major_axis_ellipse (c : Cylinder) (s1 s2 : Sphere) (d : CenterDistance) :
  c.radius = 6 → s1.radius = 6 → s2.radius = 6 → d.d = 13 → 
  ∃ α, α.tangent_to s1 ∧ α.tangent_to s2 → ellipse.major_axis α = 13 :=
by
  sorry

end length_major_axis_ellipse_l335_335596


namespace find_n_l335_335437

def t : ℕ → ℚ
| 1     := 3
| (n+1) := if (n + 1) % 2 = 0 then 2 + t ((n + 1) / 2) else 2 / t n

theorem find_n : ∃ n : ℕ, t n = 7 / 29 :=
by {
  -- The proof will verify that n = 6 satisfies the given conditions,
  -- but the proof itself is omitted here.
  sorry
}

end find_n_l335_335437


namespace johns_meeting_distance_l335_335225

theorem johns_meeting_distance (d t: ℝ) 
    (h1 : d = 40 * (t + 1.5))
    (h2 : d - 40 = 60 * (t - 2)) :
    d = 420 :=
by sorry

end johns_meeting_distance_l335_335225


namespace prob_square_l335_335636

def total_figures := 10
def num_squares := 3
def num_circles := 4
def num_triangles := 3

theorem prob_square : (num_squares : ℚ) / total_figures = 3 / 10 :=
by
  rw [total_figures, num_squares]
  exact sorry

end prob_square_l335_335636


namespace lcm_of_product_of_mutually_prime_l335_335349

theorem lcm_of_product_of_mutually_prime (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.lcm a b = a * b :=
by
  sorry

end lcm_of_product_of_mutually_prime_l335_335349


namespace count_integers_between_3250_and_3500_with_increasing_digits_l335_335543

theorem count_integers_between_3250_and_3500_with_increasing_digits :
  ∃ n : ℕ, n = 20 ∧
    (∀ x : ℕ, 3250 ≤ x ∧ x ≤ 3500 →
      ∀ (d1 d2 d3 d4 : ℕ),
        d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
        (x = d1 * 1000 + d2 * 100 + d3 * 10 + d4) →
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)) :=
  sorry

end count_integers_between_3250_and_3500_with_increasing_digits_l335_335543


namespace no_solution_fraction_eq_l335_335491

theorem no_solution_fraction_eq (x m : ℝ) (h : (2 * x + m) / (x + 3) = 1) : m = 6 → false := by
  intro m_eq
  rw [m_eq] at h
  have denom_zero : x + 3 = 0 := by
    simp
  linarith
  sorry

end no_solution_fraction_eq_l335_335491


namespace arithmetic_geometric_sequence_min_value_l335_335501

theorem arithmetic_geometric_sequence_min_value (x y a b c d : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (arithmetic_seq : a = (x + y) / 2) (geometric_seq : c * d = x * y) :
  ( (a + b) ^ 2 ) / (c * d) ≥ 4 := 
by
  sorry

end arithmetic_geometric_sequence_min_value_l335_335501


namespace cos_alpha_line_l335_335155

theorem cos_alpha_line (α : ℝ) (h : ∃ x ≤ 0, tan α = -4 / 3 ∧ α ∈ Icc (π / 2) π) : cos α = -3 / 5 :=
sorry

end cos_alpha_line_l335_335155


namespace vertex_hyperbola_l335_335689

theorem vertex_hyperbola (a b : ℝ) (h_cond : 8 * a^2 + 4 * a * b = b^3) :
    let xv := -b / (2 * a)
    let yv := (4 * a - b^2) / (4 * a)
    (xv * yv = 1) :=
  by
  sorry

end vertex_hyperbola_l335_335689


namespace smallest_period_of_f_max_min_g_on_interval_l335_335923

noncomputable def f (x : ℝ) : ℝ := 
  2 * sqrt 3 * sin (x / 2 + π / 4) * cos (x / 2 + π / 4) - sin (x + π)

noncomputable def g (x : ℝ) : ℝ := 
  f (x - π / 6)

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * π :=
sorry

theorem max_min_g_on_interval : 
  is_max (g) [0, π] 2 ∧ is_min (g) [0, π] (-1) :=
sorry

end smallest_period_of_f_max_min_g_on_interval_l335_335923


namespace intersection_M_N_l335_335122

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335122


namespace triangle_isosceles_l335_335590

-- We are given a scalene triangle ABC with specific conditions and need to prove CDE is isosceles.
universe u
variables (A B C : Type u) [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Definitions of specific points and lines
variables (A' B' D E : Type u) [MetricSpace A'] [MetricSpace B'] [MetricSpace D] [MetricSpace E]
variables (Ω : Set (Point A B C)) (A'B' : Set (Point A' B'))

variables
  (angle_BAC angle_CBA : ℝ) (angle_ACB : ℝ := 60)
  (circumcircle : Circle (Point A B C) := Ω)

-- Conditions regarding parallel and intersecting lines
variables
  (parallel_AB'BC : ∥ AB' ∥ = ∥ BC ∥)
  (parallel_BA'AC : ∥ BA' ∥ = ∥ AC ∥)
  (intersect_AB'Omega : Set.intersect A'B' Ω = {D, E})

-- We need to prove that triangle CDE is isosceles
theorem triangle_isosceles : IsoscelesTriangle C D E :=
sorry

end triangle_isosceles_l335_335590


namespace sum_of_integers_remainders_l335_335350

theorem sum_of_integers_remainders (a b c : ℕ) :
  (a % 15 = 11) →
  (b % 15 = 13) →
  (c % 15 = 14) →
  ((a + b + c) % 15 = 8) ∧ ((a + b + c) % 10 = 8) :=
by
  sorry

end sum_of_integers_remainders_l335_335350


namespace sum_of_edges_proof_l335_335320

noncomputable def sum_of_edges (a r : ℝ) : ℝ :=
  let l1 := a / r
  let l2 := a
  let l3 := a * r
  4 * (l1 + l2 + l3)

theorem sum_of_edges_proof : 
  ∀ (a r : ℝ), 
  (a > 0 ∧ r > 0 ∧ (a / r) * a * (a * r) = 512 ∧ 2 * ((a^2 / r) + a^2 + a^2 * r) = 384) → sum_of_edges a r = 96 :=
by
  intros a r h
  -- We skip the proof here with sorry
  sorry

end sum_of_edges_proof_l335_335320


namespace count_whole_numbers_between_cuberoots_l335_335548

theorem count_whole_numbers_between_cuberoots : 
  ∃ (n : ℕ), n = 7 ∧ 
      ∀ x : ℝ, (3 < x ∧ x < 4 → ∃ k : ℕ, k = 4) ∧ 
                (9 < x ∧ x ≤ 10 → ∃ k : ℕ, k = 10) :=
sorry

end count_whole_numbers_between_cuberoots_l335_335548


namespace probability_a_squared_add_b_divisible_3_l335_335394

theorem probability_a_squared_add_b_divisible_3 :
  let a_values := {a : ℕ | 1 ≤ a ∧ a ≤ 10};
  let b_values := {b : ℤ | -10 ≤ b ∧ b ≤ -1};
  let favorable_pairs := {(a, b) ∈ a_values × b_values | (a^2 + b) % 3 = 0};
  let total_pairs := ∣a_values ∣ * ∣b_values ∣ in
  ∣favorable_pairs∣ = 37 → 37 / 100 :=
by
  sorry

end probability_a_squared_add_b_divisible_3_l335_335394


namespace parallelepiped_surface_area_l335_335395

theorem parallelepiped_surface_area (a b c : ℝ) (r V : ℝ) 
  (h1 : r = sqrt 3) 
  (h2 : V = 8) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = 12)
  (h4 : a * b * c = V) :
  2 * (a * b + b * c + c * a) = 24 :=
by
  sorry

end parallelepiped_surface_area_l335_335395


namespace cos_angle_RPQ_l335_335209

theorem cos_angle_RPQ (sin_RPQ : real := 3/5) (in_first_quadrant : 0 < angle_RPQ ∧ angle_RPQ < π / 2) :
  cos angle_RPQ = 4/5 := 
sorry

end cos_angle_RPQ_l335_335209


namespace log_sum_sequence_l335_335249

theorem log_sum_sequence : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n > 1, a n + a (n-1) = 2^n + 2^(n-1)) ∧ 
  let S := (λ n : ℕ, ∑ k in range n, a (k+1)) in log2 (S 2012 + 2) = 2013) :=
sorry

end log_sum_sequence_l335_335249


namespace base6_addition_correct_l335_335425

theorem base6_addition_correct : 
  (let a := 6^3 * 4 + 6^2 * 5 + 6 * 1 + 2,
       b := 6^4 * 2 + 6^3 * 3 + 6^2 * 4 + 6 * 5 + 3
   in (a + b = 6^4 * 3 + 6^3 * 4 + 2 * 6^2 + 4 * 6 + 5)) := sorry

end base6_addition_correct_l335_335425


namespace number_of_matches_team_a_l335_335286

-- Definitions based on the conditions
def won_fraction_a : ℚ := 3/5
def won_fraction_c : ℚ := 11/20
def matches_ratio_c : ℚ := 7/6
def matches_a : ℕ := 24  -- This is the assertion we need to prove

-- Assertion that team A has played 24 matches
theorem number_of_matches_team_a :
  (let a := matches_a in
   ∃ a : ℕ, 
   won_fraction_a * a = won_fraction_c * matches_ratio_c * a) :=
sorry

end number_of_matches_team_a_l335_335286


namespace symmetric_line_equation_l335_335486

-- Definitions based on conditions
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (-2, 0)
def folding_line : ℝ → ℝ := λ x, -x
def l1 (x y : ℝ) : Prop := 2*x + 3*y - 1 = 0

-- Statement to be proven
theorem symmetric_line_equation :
  ∃ l2 : ℝ → ℝ → Prop, 
    (∀x y, l2 x y ↔ l1 (-y) (-x)) ∧
    l2 = (λ x y, 3*x + 2*y + 1 = 0) := sorry

end symmetric_line_equation_l335_335486


namespace ratio_quadrilateral_l335_335050

theorem ratio_quadrilateral
    (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ)
    (h1 : AE / EB * BF / FC * CG / GD * DH / HA = 1)
    (h2 : E1F1 ∥ EF)
    (h3 : F1G1 ∥ FG)
    (h4 : G1H1 ∥ GH)
    (h5 : H1E1 ∥ HE)
    (h6 : E1A / AH1 = λ) :
  F1C / CG1 = λ := 
sorry

end ratio_quadrilateral_l335_335050


namespace set_intersection_l335_335129

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l335_335129


namespace three_digit_numbers_condition_l335_335307

theorem three_digit_numbers_condition (a b c : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c = 2 * ((10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)))
  ↔ (100 * a + 10 * b + c = 132 ∨ 100 * a + 10 * b + c = 264 ∨ 100 * a + 10 * b + c = 396) :=
by
  sorry

end three_digit_numbers_condition_l335_335307


namespace lcm_of_40_90_150_l335_335873

-- Definition to calculate the Least Common Multiple of three numbers
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Definitions for the given numbers
def n1 : ℕ := 40
def n2 : ℕ := 90
def n3 : ℕ := 150

-- The statement of the proof problem
theorem lcm_of_40_90_150 : lcm3 n1 n2 n3 = 1800 := by
  sorry

end lcm_of_40_90_150_l335_335873


namespace sum_of_first_n_terms_bn_formula_l335_335162

def a (n : ℕ) : ℕ := 2 * n - 1

def S_n (n : ℕ) : ℕ := n * n

def b_n (n : ℕ) : ℚ :=
  let S (m : ℕ) := S_n m
  List.foldr (fun (k : ℕ) (acc : ℚ) => acc * (1 - 1 / (S k))) 1 (List.range (n + 1)) * (1 - 1 / (S (n + 1)))

theorem sum_of_first_n_terms (n : ℕ) : ∑ i in finset.range n, a (i + 1) = S_n n := sorry

theorem bn_formula (n : ℕ) : b_n n = (n + 2 : ℚ) / (2 * (n + 1)) := sorry

end sum_of_first_n_terms_bn_formula_l335_335162


namespace min_sin_x_plus_sin_z_l335_335883

theorem min_sin_x_plus_sin_z
  (x y z : ℝ)
  (h1 : sqrt 3 * cos x = cot y)
  (h2 : 2 * cos y = tan z)
  (h3 : cos z = 2 * cot x) :
  sin x + sin z ≥ -7 * sqrt 2 / 6 := 
sorry

end min_sin_x_plus_sin_z_l335_335883


namespace range_of_t_l335_335532

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the foci points
def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation
def line (x y t : ℝ) : Prop := Real.sqrt 2 * x + Real.sqrt 3 * y + t = 0

-- Define the proof statement
theorem range_of_t (t : ℝ) : 
  (∃ x y : ℝ, circle x y ∧ line x y t) → 
  -5 ≤ t ∧ t ≤ 5 :=
sorry

end range_of_t_l335_335532


namespace surface_area_of_circumscribed_sphere_l335_335592

theorem surface_area_of_circumscribed_sphere
  (PA ⊥ plane ABC)
  (∠BAC = 120°)
  (AC = 2 * √3)
  (AB = √3)
  (PA = 4 * √2) :
  surface_area_of_circumscribed_sphere (pyramid P ABC) = 60 * π :=
sorry

end surface_area_of_circumscribed_sphere_l335_335592


namespace inradius_is_sqrt3_l335_335967

def inradius_of_triangle (a b c : ℝ) (B : ℝ) (S : ℝ) : ℝ :=
1 / 2 * (a + b + c) * S

theorem inradius_is_sqrt3 : 
  ∀ (a b c : ℝ) (A B C : ℝ), 
  (A + B + C = π) → 
  (B = π / 3) → 
  (b = 7) → 
  (20 = a * c * Real.cos B) → 
  let S := 1 / 2 * a * c * (Real.sin B) 
  in inradius_of_triangle a b c S = sqrt 3 := 
by
  intros,
  sorry

end inradius_is_sqrt3_l335_335967


namespace expected_value_of_winnings_equals_3_l335_335772

noncomputable def expected_value_of_winnings (p : ℕ → ℚ) : ℚ :=
  ∑ k in [2, 4, 6, 8], p k * k + p 3 * 2 + p 5 * 2

theorem expected_value_of_winnings_equals_3 :
  let p : ℕ → ℚ := λ k, if k ∈ [1, 2, 3, 4, 5, 6, 7, 8] then 1/8 else 0 in
  expected_value_of_winnings p = 3 := by
  sorry

end expected_value_of_winnings_equals_3_l335_335772


namespace eccentricities_proof_l335_335611

variable (e1 e2 m n c : ℝ)
variable (h1 : e1 = 2 * c / (m + n))
variable (h2 : e2 = 2 * c / (m - n))
variable (h3 : m ^ 2 + n ^ 2 = 4 * c ^ 2)

theorem eccentricities_proof :
  (e1 * e2) / (Real.sqrt (e1 ^ 2 + e2 ^ 2)) = (Real.sqrt 2) / 2 :=
by sorry

end eccentricities_proof_l335_335611


namespace least_positive_integer_x_l335_335344

theorem least_positive_integer_x :
  ∃ (x : ℕ), (x > 0) ∧ (((2 * x + 33) ^ 2) % 43 = 0) ∧ ∀ (y : ℕ), (y > 0) ∧ (((2 * y + 33) ^ 2) % 43 = 0) → x ≤ y :=
begin
  use 5,
  split,
  { exact nat.succ_pos' 4, },
  split,
  { exact nat.zero_mod 43, },
  intros y hy,
  sorry,
end

end least_positive_integer_x_l335_335344


namespace probability_same_length_l335_335233

/-- Define the set of all segments (sides and diagonals) of a regular hexagon. -/
def segments_of_hexagon : finset ℕ :=
  finset.range 15 -- There are 15 segments in total (6 sides + 9 diagonals).

/-- Define the number of segments of given lengths. -/
def num_sides := 6
def num_shorter_diagonals := 3
def num_longer_diagonals := 3

/-- Define the probability of selecting two segments of the same length from the hexagon. -/
theorem probability_same_length :
  (num_sides / (segments_of_hexagon.card - 1) * 5 +
   num_shorter_diagonals / (segments_of_hexagon.card - 1) * 2 +
   num_longer_diagonals / (segments_of_hexagon.card - 1) * 2) = 9 / 14 :=
by
  sorry

end probability_same_length_l335_335233


namespace total_distance_trip_l335_335014

-- Defining conditions
def time_paved := 2 -- hours
def time_dirt := 3 -- hours
def speed_dirt := 32 -- mph
def speed_paved := speed_dirt + 20 -- mph

-- Defining distances
def distance_dirt := speed_dirt * time_dirt -- miles
def distance_paved := speed_paved * time_paved -- miles

-- Proving total distance
theorem total_distance_trip : distance_dirt + distance_paved = 200 := by
  sorry

end total_distance_trip_l335_335014


namespace minimum_people_seated_adjacency_constraint_l335_335707

theorem minimum_people_seated_adjacency_constraint (chairs : ℕ) (n : ℕ) (h1 : chairs = 100) (h2 : ∀ m : ℕ, m ≥ n → (m = n ∨ m < n → ∃ k : ℕ, k ⟨m - n, chairs⟩ = ((k + 1))%chairs ∨ (k - 1)%chairs = ⟩)) : n = 34 :=
sorry

end minimum_people_seated_adjacency_constraint_l335_335707


namespace one_million_div_one_fourth_l335_335177

theorem one_million_div_one_fourth : (1000000 : ℝ) / (1 / 4) = 4000000 := by
  sorry

end one_million_div_one_fourth_l335_335177


namespace range_of_a_l335_335559

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (deriv (f x1 a) = 0) ∧ (deriv (f x2 a) = 0)) → a < 0 :=
by
  sorry

end range_of_a_l335_335559


namespace find_positive_m_l335_335475

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end find_positive_m_l335_335475


namespace probability_at_least_one_task_expectation_of_X_l335_335574

open ProbabilityTheory

def pA : ℚ := 3 / 4
def pB : ℚ := 3 / 4
def pC : ℚ := 2 / 3

-- Define the event that Jia completes at least one task
def P_at_least_one : ℚ :=
  1 - ((1 - pA) * (1 - pB) * (1 - pC))

theorem probability_at_least_one_task : P_at_least_one = 47 / 48 := by
  sorry

-- Define the random variable for the points earned
def X : Fin₄ → ℚ
| Fin₄.fz := 0
| Fin₄.fin1 := 1
| Fin₄.fin2 := 3
| Fin₄.fin3 := 6

-- Define the probabilities for the points
def pX (x : ℚ) : ℚ :=
  match x with
  | 0   => 7 / 16
  | 1   => 63 / 256
  | 3   => 21 / 256
  | 6   => 15 / 64
  | _   => 0

-- Calculate the expected value of X
def E_X : ℚ :=
  ∑ i, (X i) * (pX (X i))

theorem expectation_of_X : E_X = 243 / 128 := by
  sorry

end probability_at_least_one_task_expectation_of_X_l335_335574


namespace original_books_on_layers_l335_335762

-- Definitions based on conditions
def total_books : ℕ := 270
def books_moved_from_first_to_second : ℕ := 20
def books_moved_from_third_to_second : ℕ := 17
def books_per_layer_afterwards : ℕ := total_books / 3

-- Main theorem statement
theorem original_books_on_layers :
  (original_first_layer = 110) ∧ (original_second_layer = 53) ∧ (original_third_layer = 107) :=
begin
  -- Definitions based on changes and outcomes
  let original_first_layer := books_per_layer_afterwards + books_moved_from_first_to_second,
  let original_second_layer := books_per_layer_afterwards - (books_moved_from_first_to_second + books_moved_from_third_to_second),
  let original_third_layer := books_per_layer_afterwards + books_moved_from_third_to_second,
  -- Proof omitted
  sorry
end

end original_books_on_layers_l335_335762


namespace num_div_72_l335_335308

theorem num_div_72 (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) :
  (17 + a * 10 + ((b * 10 + 6) // 8) * 8 = a * 1000 + b * 100 + 76 → 
  (a + b = 4 ∨ a + b = 13)) :=
begin
  sorry
end

end num_div_72_l335_335308


namespace number_of_ways_to_fill_positions_l335_335709

theorem number_of_ways_to_fill_positions :
  ∑ i in {(3.choose 1) * (4.choose 2) * 2! + (4.choose 3) * 3!}, i = 60 :=
by
  sorry

end number_of_ways_to_fill_positions_l335_335709


namespace perpendicular_planes_parallel_l335_335105

-- Definitions for line, planes, and relationships of perpendicularity and parallelism
variables (l : Type) (α β : Type)
variables [plane α] [plane β] [line l]

-- Definitions for the relationships parallel and perpendicular
variable [relation_parallel_parallel : l ∥ α]
variable [relation_perpendicular_perpendicular : l ⟂ α]
variable [relation_plane : α ∥ β]

-- Stating the actual proof problem in Lean 4
theorem perpendicular_planes_parallel (h1: l ⟂ α) (h2: l ⟂ β) : α ∥ β :=
by
  sorry

end perpendicular_planes_parallel_l335_335105


namespace total_distance_theorem_l335_335002

def time_in_hours (minutes: ℝ) : ℝ :=
  minutes / 60

def total_distance_covered (time_flat : ℝ) (speed_flat : ℝ) (distance_flat : ℝ)
                           (time_uphill : ℝ) (speed_uphill : ℝ) (distance_uphill : ℝ)
                           (time_total : ℝ) (speed_downhill : ℝ) : ℝ :=
  let time_flat := distance_flat / speed_flat in
  let time_uphill := distance_uphill / speed_uphill in
  let time_remaining := time_total - (time_flat + time_uphill) in
  let distance_downhill := time_remaining * speed_downhill in
  distance_flat + distance_uphill + distance_downhill

theorem total_distance_theorem :
  total_distance_covered (2 / 6) 6 2
                         (3 / 4) 4 3
                         1.2 8 = 5.9336 := by
    sorry

end total_distance_theorem_l335_335002


namespace max_value_of_f_product_of_zeros_l335_335160

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b
 
theorem max_value_of_f (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) : f (1 / a) a b = -Real.log a - 1 + b :=
by
  sorry

theorem product_of_zeros (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) (hx_ne : x1 ≠ x2) : x1 * x2 < 1 / (a * a) :=
by
  sorry

end max_value_of_f_product_of_zeros_l335_335160


namespace perpendicular_k_value_exists_l335_335166

open Real EuclideanSpace

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 2)

theorem perpendicular_k_value_exists : ∃ k : ℝ, (vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) ∧ k = 5 / 4 := by
  sorry

end perpendicular_k_value_exists_l335_335166


namespace curve_shape_and_intersection_dot_product_PA_PB_l335_335535

-- Definitions of the conditions
def line_parametric_eq (theta : ℝ) : ℝ := 2 * real.sqrt 2 * real.sin (theta + real.pi / 4)

def curve_polar_eq (t : ℝ) : ℝ × ℝ := (t, 1 + 2 * t)

-- Theorems to prove
theorem curve_shape_and_intersection :
  (∃ r c, ∀ t, curve_polar_eq t = ((1 : ℝ) + r * real.cos t, (1 : ℝ) + r * real.sin t) 
    ∧ r = real.sqrt 2 ∧ c = (1, 1)) ∧
  (∃ theta, line_parametric_eq theta ∈ range (λ t, (curve_polar_eq t).1))
:= sorry

theorem dot_product_PA_PB :
  (∃ A B : ℝ × ℝ, A ∈ range curve_polar_eq ∧ B ∈ range curve_polar_eq
    ∧ let PA := (λ x, (0 : ℝ, 1 : ℝ) - x) in
    PA A • PA B = -1)
:= sorry

end curve_shape_and_intersection_dot_product_PA_PB_l335_335535


namespace largest_prime_factor_of_expression_l335_335739

theorem largest_prime_factor_of_expression : 
  ∃ p, prime p ∧ p ≥ 2 ∧ 
  (∀ q, (q ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) → prime q → q ≤ p) ∧ 
  p = 241 :=
by
  sorry

end largest_prime_factor_of_expression_l335_335739


namespace second_fragment_velocity_l335_335387

-- Definitions
def initial_vertical_velocity : ℝ := 20 -- Initial velocity \( v_0 \)
def time_at_explosion : ℝ := 1 -- Time \( t \)
def gravity : ℝ := 10 -- Acceleration due to gravity \( g \)
def horizontal_velocity_first_fragment : ℝ := 48 -- Horizontal velocity \( v_{x1} \)

-- Proof Statement
theorem second_fragment_velocity :
  let v_y := initial_vertical_velocity - gravity * time_at_explosion in
  sqrt ((-horizontal_velocity_first_fragment) ^ 2 + v_y ^ 2) = 52 :=
by
  sorry

end second_fragment_velocity_l335_335387


namespace back_parking_lot_filled_fraction_l335_335690

theorem back_parking_lot_filled_fraction
    (front_spaces : ℕ) (back_spaces : ℕ) (cars_parked : ℕ) (spaces_available : ℕ)
    (h1 : front_spaces = 52)
    (h2 : back_spaces = 38)
    (h3 : cars_parked = 39)
    (h4 : spaces_available = 32) :
    (back_spaces - (front_spaces + back_spaces - cars_parked - spaces_available)) / back_spaces = 1 / 2 :=
by
  sorry

end back_parking_lot_filled_fraction_l335_335690


namespace no_solution_for_12k_plus_7_l335_335649

theorem no_solution_for_12k_plus_7 (k : ℤ) :
  ∀ (a b c : ℕ), 12 * k + 7 ≠ 2^a + 3^b - 5^c := 
by sorry

end no_solution_for_12k_plus_7_l335_335649


namespace smallest_n_for_factorable_quadratic_l335_335446

open Int

theorem smallest_n_for_factorable_quadratic : ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 72 → 3 * B + A = n) ∧ n = 35 :=
by
  sorry

end smallest_n_for_factorable_quadratic_l335_335446


namespace terminating_decimal_representation_l335_335070

-- Definitions derived from conditions
def given_fraction : ℚ := 53 / (2^2 * 5^3)

-- The theorem we aim to state that expresses the question and correct answer
theorem terminating_decimal_representation : given_fraction = 0.106 :=
by
  sorry  -- proof goes here

end terminating_decimal_representation_l335_335070


namespace front_view_l335_335970

-- Define the cubes in each column as given by the conditions
def column1 := [3, 1]
def column2 := [2, 4, 2]
def column3 := [5, 2]

-- The function to get the tallest stack from a column
def tallest (column : List ℕ) : ℕ :=
  List.maximumD column 0

-- Define the condition for the front view determination problem
theorem front_view (front : List ℕ) : front = [tallest column1, tallest column2, tallest column3] :=
by
  -- Proof not required, so we use sorry
  sorry

end front_view_l335_335970


namespace committee_selection_license_plate_l335_335373

-- Problem 1
theorem committee_selection : 
  let members : Finset ℕ := (Finset.range 5) in
  let not_entertainment : Finset ℕ := {0, 1} in
  let eligible_for_entertainment : Finset ℕ := members \ not_entertainment in
  eligible_for_entertainment.card = 3 ∧ 
  (members \ {0}).card = 4 ∧ 
  (members \ {1}).card = 4 
  → Fintype.choose eligible_for_entertainment 1 * Fintype.perm (members \ {0}) 2 = 36
:= by sorry

-- Problem 2
theorem license_plate : 
  let english_letters : ℕ := 26 in
  let digits : ℕ := 10 in
  (english_letters ^ 2) * Fintype.arrangements digits 4 = 26^2 * Fintype.arrangements digits 4
:= by sorry

end committee_selection_license_plate_l335_335373


namespace combined_future_value_l335_335031

noncomputable def future_value (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem combined_future_value :
  let A1 := future_value 3000 0.05 3
  let A2 := future_value 5000 0.06 4
  let A3 := future_value 7000 0.07 5
  A1 + A2 + A3 = 19603.119 :=
by
  sorry

end combined_future_value_l335_335031


namespace solution_set_l335_335660

-- Define the system of equations
def system_of_equations (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2 * y + 4 ∧
  (2 * x)^2 - (y + 1)^2 = 3 ∧
  (2 * x - (y + 1)) * (2 * x + (y + 1)) = 3

-- Prove that the solutions to the system are the set we expect
theorem solution_set : 
  { (x, y) : ℤ × ℤ | system_of_equations x y } = { (1, 0), (1, -2), (-1, 0), (-1, -2) } := 
by 
  -- Proof omitted
  sorry

end solution_set_l335_335660


namespace vector_arithmetic_l335_335833

-- Define the vectors
def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (-1, 4)

-- Define scalar multiplications
def scalar_mult1 : ℝ × ℝ := (12, -20)  -- 4 * v1
def scalar_mult2 : ℝ × ℝ := (6, -18)   -- 3 * v2

-- Define intermediate vector operations
def intermediate_vector1 : ℝ × ℝ := (6, -2)  -- (12, -20) - (6, -18)

-- Final operation
def final_vector : ℝ × ℝ := (5, 2)  -- (6, -2) + (-1, 4)

-- Prove the main statement
theorem vector_arithmetic : 
  (4 : ℝ) • v1 - (3 : ℝ) • v2 + v3 = final_vector := by
  sorry  -- proof placeholder

end vector_arithmetic_l335_335833


namespace triangle_perimeter_l335_335846

variable (y : ℝ)

theorem triangle_perimeter (h₁ : 2 * y > y) (h₂ : y > 0) :
  ∃ (P : ℝ), P = 2 * y + y * Real.sqrt 2 :=
sorry

end triangle_perimeter_l335_335846


namespace angle_YXZ_25_degrees_l335_335993

theorem angle_YXZ_25_degrees
  (XYZ : Type)
  [triangle XYZ]
  (D : XYZ → Prop)
  (tangent : ∀ x y z : XYZ, D z → ∃ c, D c)
  (angle_XYZ : real := 75)
  (angle_YDZ : real := 40) :
  angle_XYZ - 75  ∧  angle_YDZ - 40  ∧ ∠ XYZ = 25 :=
sorry

end angle_YXZ_25_degrees_l335_335993


namespace shaded_region_perimeter_l335_335210

-- Define the given conditions
variable (C : Type) (circle : C → Prop)
variable (circumference : C → ℝ) (touches : C → C → Prop)
variable (centers_form_right_angle : C → C → C → Prop)

-- Condition that there are four circles, each with a given circumference
def four_identical_circles (c1 c2 c3 c4 : C) :=
  circle c1 ∧ circle c2 ∧ circle c3 ∧ circle c4 ∧
  (circumference c1 = 72) ∧ (circumference c2 = 72) ∧ (circumference c3 = 72) ∧ (circumference c4 = 72) ∧
  touches c1 c2 ∧ touches c2 c3 ∧ touches c3 c4 ∧ touches c4 c1 ∧ touches c1 c3 ∧ touches c2 c4 ∧
  centers_form_right_angle c1 c2 c3 ∧ centers_form_right_angle c2 c3 c4 ∧
  centers_form_right_angle c3 c4 c1 ∧ centers_form_right_angle c4 c1 c2

-- Proposition stating the perimeter of the shaded central region is 72
theorem shaded_region_perimeter (c1 c2 c3 c4 : C) (h : four_identical_circles c1 c2 c3 c4) : ℝ :=
  72

end shaded_region_perimeter_l335_335210


namespace arrangement_is_even_l335_335956

theorem arrangement_is_even : ∃ l : List ℕ, l.permutations.length = 648 ∧ 
  ∀ ⦃a b c d e f : ℕ⦄, l = [a, b, c, d, e, f] →
    a * b * c + d * e * f % 2 = 0 :=
begin
  sorry

end arrangement_is_even_l335_335956


namespace division_periodic_l335_335647

theorem division_periodic (n d : ℕ) (h1 : n = 41) (h2 : d = 61) : periodic (decimals (n/d)) := 
by
  sorry

end division_periodic_l335_335647


namespace evaluate_expression_l335_335459

theorem evaluate_expression : (Real.sqrt ((Real.sqrt 2)^4))^6 = 64 := by
  sorry

end evaluate_expression_l335_335459


namespace sum_of_products_composite_l335_335805

/-- Given 100 natural numbers from 2 to 101, divided into two groups each containing 50 numbers, 
    prove that the sum of the products of these two groups is a composite number. -/
theorem sum_of_products_composite :
  ∀ (s : Finset ℕ), s = Finset.range 102 \ {0, 1} →
  ∀ g1 g2 : Finset ℕ, (g1 ∪ g2 = s) ∧ (g1 ∩ g2 = ∅) ∧ (g1.card = 50) ∧ (g2.card = 50) →
  Nat.is_composite ((g1.prod id) + (g2.prod id)) :=
by
  intros s hs g1 g2 h_union_and_card
  sorry

end sum_of_products_composite_l335_335805


namespace generating_function_l335_335297

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 0 then 1 else (-1)^n * (n + 1)

theorem generating_function (G : ℤ → ℤ) :
  (∀ n, G n = sequence n) → ∑ n in ℕ, G n * x^n = (1 + x)^(-2) :=
by
  intro h
  sorry

end generating_function_l335_335297


namespace customers_stayed_behind_l335_335803

theorem customers_stayed_behind : ∃ x : ℕ, (x + (x + 5) = 11) ∧ x = 3 := by
  sorry

end customers_stayed_behind_l335_335803


namespace number_of_complex_solutions_l335_335471

theorem number_of_complex_solutions :
  (∀ z : ℂ, (z^4 - 1) / (z^3 + z^2 - 2z) = 0 ↔ ((z^2 + 1) * (z - 1) * (z + 1) = 0 ∧ z ≠ 0 ∧ (z - 1) ≠ 0 ∧ (z + 2) ≠ 0)) →
  ∃! n : ℕ, n = 2 :=
by
  sorry

end number_of_complex_solutions_l335_335471


namespace cylindrical_tank_half_filled_volume_l335_335771

theorem cylindrical_tank_half_filled_volume :
  ∀ (r h : ℝ), r = 5 ∧ h = 10 ∧ (∀ x, x = π) → (1 / 2 * π * r ^ 2 * h = 125 * π) :=
by
  intros r h 
  intro h_conditions
  cases h_conditions with r_eq5 h_conditions
  cases h_conditions with h_eq10 x_conditions
  sorry

end cylindrical_tank_half_filled_volume_l335_335771


namespace zero_relationship_l335_335927

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x - Real.sqrt x - 1

noncomputable def zero_f : ℝ := Classical.some (exists_zero f)
noncomputable def zero_g : ℝ := Classical.some (exists_zero g)
noncomputable def zero_h : ℝ := Classical.some (exists_zero h)

theorem zero_relationship :
  (f zero_f = 0) ∧ (g zero_g = 0) ∧ (h zero_h = 0) ∧ zero_f < zero_g ∧ zero_g < zero_h := sorry

end zero_relationship_l335_335927


namespace find_a2017_l335_335897

noncomputable def sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = 1 / (1 - a n)

def initial_condition (a : ℕ → ℝ) : Prop := a 1 = 1 / 2

theorem find_a2017 (a : ℕ → ℝ) 
  (h1 : sequence a) 
  (h2 : initial_condition a) : 
  a 2017 = 1 / 2 := sorry

end find_a2017_l335_335897


namespace angle_FHP_eq_angle_BAC_l335_335241

variables {A B C O H F P : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space O]
[metric_space H] [metric_space F] [metric_space P]
(AcuteABC : is_acute_triangle ABC) -- statement that ABC is an acute triangle
(BC_GT_CA : BC > CA) -- statement that BC is greater than CA
(O_is_circumcenter : circumcenter ABC = O) -- O is the circumcenter
(H_is_orthocenter : orthocenter ABC = H) -- H is the orthocenter
(F_foot_CH : foot (altitude C H) AB = F) -- F is the foot of the altitude from C to AB
(P_is_perpendicular : ∃ l, is_perpendicular l (line OF) ∧ P ∈ l ∧ P ∈ line CA) -- P is defined by the intersection
-- Statement to be proved
theorem angle_FHP_eq_angle_BAC :
  ∠ F H P = ∠ B A C := sorry

end angle_FHP_eq_angle_BAC_l335_335241


namespace find_x1_l335_335904

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
(h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 3 / 4) : 
x1 = 3 * real.sqrt(3) / 8 :=
sorry

end find_x1_l335_335904


namespace runner_beats_more_than_half_l335_335328

-- Definitions of runners and their race outcomes
inductive Runner
| A | B | C

def RaceOutcome : Type := Runner → Runner → Prop

constant race1 : RaceOutcome
constant race2 : RaceOutcome
constant race3 : RaceOutcome

-- Specific race outcomes
axiom race1_outcome : race1 Runner.A Runner.B ∧ race1 Runner.B Runner.C ∧ ¬ race1 Runner.C Runner.A
axiom race2_outcome : race2 Runner.B Runner.C ∧ race2 Runner.C Runner.A ∧ ¬ race2 Runner.A Runner.B
axiom race3_outcome : race3 Runner.C Runner.A ∧ race3 Runner.A Runner.B ∧ ¬ race3 Runner.B Runner.C

-- Definitions used in the proof problem
def beats (r : RaceOutcome) (x y : Runner) : Prop := r x y

-- Prove the conditions hold:
theorem runner_beats_more_than_half : 
  (∀ x y : Runner, x ≠ y → (count_beats x y > 1 / 2)) :=
by
  sorry

end runner_beats_more_than_half_l335_335328


namespace ratio_of_radii_l335_335200

-- Define radii of the smaller and larger circles
variable (a b : ℝ)

-- Define the condition provided in the problem
def area_condition : Prop := π * b^2 = 4 * π * a^2

-- The theorem to prove the ratio of the radii
theorem ratio_of_radii (h : area_condition a b) : a / b = 1 / Real.sqrt 5 := 
  sorry

end ratio_of_radii_l335_335200


namespace probability_real_roots_l335_335399

noncomputable def probability_real_roots_polynomial : ℝ :=
  let a_min := -15
  let a_max := 20
  let interval_length := a_max - a_min
  let discriminant_positive_min := (3 - Real.sqrt 12) / 2
  let discriminant_positive_max := (3 + Real.sqrt 12) / 2
  let excluded_interval_length := discriminant_positive_max - discriminant_positive_min
  1 - (excluded_interval_length / interval_length)

theorem probability_real_roots (a : ℝ) (h : a ∈ Icc (-15 : ℝ) 20) :
  probability_real_roots_polynomial = (35 - Real.sqrt 12) / 35 :=
by
  sorry

end probability_real_roots_l335_335399


namespace determinant_zero_of_triangle_angles_l335_335864

open Matrix

noncomputable def detMatrix : ℝ :=
  Matrix.det ![
    ![Real.sin A, Real.cos A, 1],
    ![Real.sin B, Real.cos B, 1],
    ![Real.sin C, Real.cos C, 1]
  ]

theorem determinant_zero_of_triangle_angles 
  (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (sum_of_angles : A + B + C = π) : 
  detMatrix A B C = 0 := 
by 
  -- proof goes here
  sorry

end determinant_zero_of_triangle_angles_l335_335864


namespace tony_spider_days_l335_335713

theorem tony_spider_days (crawl_rate : ℕ) (initial_height : ℕ) (move_up : ℕ) (wall_height : ℕ)
  (crawl_down_per_day : ℕ) (net_movement_up_per_day : ℕ) : ℕ :=
begin
  let height_after_n_days := initial_height + net_movement_up_per_day * crawl_rate,
  have h : 3 + 2 * crawl_rate ≥ 18,
  have h2 : crawl_rate = 8,
  exact h2
end

end tony_spider_days_l335_335713


namespace cody_marbles_l335_335043

theorem cody_marbles (M : ℕ) (h1 : M / 3 + 5 + 7 = M) : M = 18 :=
by
  have h2 : 3 * M / 3 + 3 * 5 + 3 * 7 = 3 * M := by sorry
  have h3 : 3 * M / 3 = M := by sorry
  have h4 : 3 * 7 = 21 := by sorry
  have h5 : M + 15 + 21 = 3 * M := by sorry
  have h6 : M = 18 := by sorry
  exact h6

end cody_marbles_l335_335043


namespace carrie_pants_l335_335042

theorem carrie_pants (P : ℕ) (shirts := 4) (pants := P) (jackets := 2)
  (shirt_cost := 8) (pant_cost := 18) (jacket_cost := 60)
  (total_cost := shirts * shirt_cost + jackets * jacket_cost + pants * pant_cost)
  (total_cost_half := 94) :
  total_cost = 188 → total_cost_half = 94 → total_cost = 2 * total_cost_half → P = 2 :=
by
  intros h_total h_half h_relation
  sorry

end carrie_pants_l335_335042


namespace locus_of_C_l335_335402

variable (A B C D : Point)
variable (AB_median : Line A B)
variable (AD_median : Line A D)
variable (BC : Line B C)
variable (D_midpoint : Midpoint D B C)

-- Defining lengths in the conditions
def AB_length : ℝ := 6
def AD_length : ℝ := 3

-- Condition specifications
axiom AB_fixed_length : length AB_median = AB_length
axiom AD_fixed_length : length AD_median = AD_length
axiom D_is_midpoint : Midpoint D B C

theorem locus_of_C : ∀ (C : Point),
  dist A C = AD_length :=
sorry

end locus_of_C_l335_335402


namespace trajectory_midpoint_l335_335641

variables (A B : ℝ × ℝ) (P : ℝ × ℝ)

def on_circle (A : ℝ × ℝ) : Prop := A.1^2 + A.2^2 = 4
def midpoint (A B P : ℝ × ℝ) : Prop := P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

theorem trajectory_midpoint (hA : on_circle A) (hP : midpoint A (0, 4) P) :
  P.1^2 + (P.2 - 2)^2 = 1 :=
sorry

end trajectory_midpoint_l335_335641


namespace minimum_triple_overlap_l335_335656

open Set

variables {α : Type} 

-- Percentages of each set
constants (pC pT pS : ℝ)
-- Assumed percentages
axiom Coffee : pC = 0.75
axiom Tea : pT = 0.60
axiom Soda : pS = 0.55

-- Pairwise overlap lower bounds
def overlap_CT := pC + pT - 1
def overlap_CS := pC + pS - 1
def overlap_TS := pT + pS - 1

-- Inclusion-Exclusion Principle calculation
def inclusion_exclusion (overlap_CT overlap_CS overlap_TS triple_overlap : ℝ) : Prop :=
    (pC + pT + pS - overlap_CT - overlap_CS - overlap_TS + triple_overlap ≤ 1)

-- We aim to prove that the minimum possible triple_overlap is 0
theorem minimum_triple_overlap : ∀ (triple_overlap : ℝ), triple_overlap ≥ 0 → 
    inclusion_exclusion overlap_CT overlap_CS overlap_TS triple_overlap :=
begin
  intros triple_overlap h,
  sorry
end

end minimum_triple_overlap_l335_335656


namespace exists_divisible_term_no_divisible_term_l335_335489

-- Definitions based on conditions
structure NiceSequence (a : ℕ → ℕ) : Prop :=
  (nice : ∀ n, a (2 * n) = 2 * a n)
  (strict_mono : strict_mono a)

-- Part (a)
theorem exists_divisible_term (a : ℕ → ℕ) (h : NiceSequence a) {p : ℕ} (hp : nat.prime p) (hpa : p > a 1) : 
  ∃ n, p ∣ a n := 
sorry

-- Part (b)
theorem no_divisible_term (p : ℕ) (hp : nat.prime p) (h2 : p > 2) :
  ∃ (a : ℕ → ℕ), NiceSequence a ∧ ∀ n, ¬ p ∣ a n :=
sorry

end exists_divisible_term_no_divisible_term_l335_335489


namespace parabola_vertex_in_fourth_quadrant_l335_335533

theorem parabola_vertex_in_fourth_quadrant (a c : ℝ) (h : -a > 0 ∧ c < 0) :
  a < 0 ∧ c < 0 :=
by
  sorry

end parabola_vertex_in_fourth_quadrant_l335_335533


namespace distance_between_intersections_l335_335871

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 12 * x

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y = 0

theorem distance_between_intersections :
  ∃ A B : ℝ × ℝ, parabola A.1 A.2 ∧ circle A.1 A.2 ∧ parabola B.1 B.2 ∧ circle B.1 B.2 ∧
    (A ≠ B) ∧ real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6 * real.sqrt 3 := sorry

end distance_between_intersections_l335_335871


namespace bear_weight_conversion_l335_335322

theorem bear_weight_conversion (kg_to_lb : ℝ) (bear_weight_kg : ℝ) (expected_weight_lb : ℝ) : 
  kg_to_lb = 0.4545 → bear_weight_kg = 300 → expected_weight_lb = 660 → 
  Float.round (bear_weight_kg / kg_to_lb) = expected_weight_lb :=
by
  intros h1 h2 h3
  sorry

end bear_weight_conversion_l335_335322


namespace distance_between_vertices_of_hyperbola_l335_335465

theorem distance_between_vertices_of_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0 → 
  2 * Real.sqrt (7 / 4) = Real.sqrt 7 :=
by
  intros x y h
  sorry

end distance_between_vertices_of_hyperbola_l335_335465


namespace number_of_correct_statements_l335_335414

theorem number_of_correct_statements :
  let statements := [
    "The altitude, median, and angle bisector of a triangle are all line segments",
    "Alternate interior angles are equal",
    "In the coordinate plane, points and ordered pairs correspond one-to-one",
    "Because a ⊥ b and a ⊥ c, therefore b ⊥ c",
    "In ∆ABC and ∆A'B'C', AB = A'B', BC = A'C', ∠B = ∠A', therefore ∆ABC is congruent to ∆A'B'C'"
  ] in
  let correct_statements := [true, false, true, false, true] in
  list.count (fun x => x = true) correct_statements = 3 :=
by
  sorry

end number_of_correct_statements_l335_335414


namespace dartboard_distribution_l335_335251

theorem dartboard_distribution : 
  ∃ (lists : Finset (Finset ℕ)), 
  lists.card = 15 ∧ 
  ∀ list ∈ lists, list.sum = 7 ∧ list.card = 5 ∧ 
  ∀ (a b ∈ list), a ≥ b → ∀ (c d ∈ list), d ≤ c → a ≥ c :=
sorry

end dartboard_distribution_l335_335251


namespace determine_f_when_alpha_l335_335146

noncomputable def solves_functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
∀ (x y : ℝ), 0 < x → 0 < y → f (f x + y) = α * x + 1 / (f (1 / y))

theorem determine_f_when_alpha (α : ℝ) (f : ℝ → ℝ) :
  (α = 1 → ∀ x, 0 < x → f x = x) ∧ (α ≠ 1 → ∀ f, ¬ solves_functional_equation f α) := by
  sorry

end determine_f_when_alpha_l335_335146


namespace find_length_of_chord_AB_l335_335393

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of points A and B
variables (x1 x2 y1 y2 : ℝ)

-- Define the conditions
def conditions : Prop := 
  parabola x1 y1 ∧ parabola x2 y2 ∧ (x1 + x2 = 4 / 3)

-- Define the length of chord AB
def length_of_chord_AB : ℝ := 
  (x1 + 1) + (x2 + 1)

-- Prove the length of chord AB
theorem find_length_of_chord_AB (x1 x2 y1 y2 : ℝ) (h : conditions x1 x2 y1 y2) :
  length_of_chord_AB x1 x2 = 10 / 3 :=
by
  sorry -- Proof is not required

end find_length_of_chord_AB_l335_335393


namespace students_ate_half_of_chocolates_without_nuts_l335_335357

def portion_of_chocolates_without_nuts_eaten
  (total_chocolates : ℕ)
  (portion_with_nuts : ℕ → ℕ)
  (portion_left : ℕ)
  (portion_ate_with_nuts : ℕ → ℕ)
  (portion_with_nuts_eaten : ℕ)
: ℕ :=
  (portion_with_nuts_eaten * portion_with_nuts (total_chocolates) / portion_with_nuts (total_chocolates))

theorem students_ate_half_of_chocolates_without_nuts
  (total_chocolates : ℕ)
  (portion_with_nuts : ℕ → ℕ := λ n, n / 2)
  (portion_left : ℕ)
  (portion_ate_with_nuts : ℕ → ℕ := λ n, 4 * (n / 5))  -- 80% is equivalent to 4/5
  : portion_of_chocolates_without_nuts_eaten total_chocolates portion_with_nuts portion_left portion_ate_with_nuts (total_chocolates / 2 - (portion_left - portion_ate_with_nuts (total_chocolates / 2))) = 0.5 :=
by
  sorry

end students_ate_half_of_chocolates_without_nuts_l335_335357


namespace distance_between_consecutive_trees_l335_335374

-- Definitions for the conditions
def yard_length : ℝ := 275
def number_of_trees : ℕ := 26
def number_of_intervals := number_of_trees - 1

-- Definition for the distance between trees
def distance_between_trees : ℝ := yard_length / number_of_intervals

-- Theorem stating the main proof problem
theorem distance_between_consecutive_trees :
  distance_between_trees = 11 := by
  sorry

end distance_between_consecutive_trees_l335_335374


namespace red_marbles_count_l335_335569

def num_red_marbles (R : ℕ) : Prop :=
  ∃ (G Y D : ℕ), 
    G = 3 * R ∧
    Y = (3 * R) / 5 ∧    -- 20% of 3R is (3R * 20 / 100) = (3R / 5)
    D = 88 ∧
    (R + G + Y + D = 12 * R)

theorem red_marbles_count : num_red_marbles 12 :=
by
  unfold num_red_marbles
  use (3 * 12)  -- G
  use (3 * 12) / 5  -- Y
  use 88  -- D
  split; try rfl
  show 3 * 12 = 3 * 12; rfl
  split;
  { show 12 * 3 / 5 = 12 * 3 / 5; rfl }
  split;
  { show 88 = 88; rfl }
  show 12 + 3 * 12 + (3 * 12) / 5 + 88 = 12 * 12
  calc
    12 + 3 * 12 + (3 * 12) / 5 + 88
       = 12 + 36 + 7 + 88 : by rfl
     ... = 12 * 12   : by norm_num

end red_marbles_count_l335_335569


namespace linear_function_not_valid_l335_335561

theorem linear_function_not_valid (k b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = -3 → x = -1 → y = -3) ∧
  ((∀ m n : ℝ, m = -b/k → n = b) → dist (0, b) (0, n) = dist (-b/k, 0) (m, 0)) →
  ¬ ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b → x = -1 → y = -3) ∧ y = -3x - 6 := 
begin
  sorry
end

end linear_function_not_valid_l335_335561


namespace median_salary_is_25000_l335_335019

-- Definitions for the given conditions
noncomputable def total_employees : ℕ := 70
noncomputable def salaries : list (ℕ × ℕ) := [
  (1, 140000),  -- President
  (8, 95000),   -- Vice-President
  (10, 78000),  -- Director
  (4, 60000),   -- Manager
  (7, 55000),   -- Associate Director
  (40, 25000)   -- Administrative Specialist
]

-- The proposition that we need to prove
theorem median_salary_is_25000 : 
  let sorted_salaries := (salaries.bind (λ (pos_salary : ℕ × ℕ), list.replicate pos_salary.fst pos_salary.snd)).insertion_sort (≤)
  in sorted_salaries.nth (35 - 1) = some 25000 ∧ sorted_salaries.nth (36 - 1) = some 25000 :=
by sorry

end median_salary_is_25000_l335_335019


namespace weekly_allowance_l335_335540

theorem weekly_allowance (A : ℝ) (H1 : A - (3/5) * A = (2/5) * A)
(H2 : (2/5) * A - (1/3) * ((2/5) * A) = (4/15) * A)
(H3 : (4/15) * A = 0.96) : A = 3.6 := 
sorry

end weekly_allowance_l335_335540


namespace distance_from_apex_l335_335715

/-- Proved distance from the apex of the right octagonal pyramid to the larger cross section -/
theorem distance_from_apex
  (A1 A2 : ℝ) (h_diff : ℝ)
  (area1 : A1 = 400 * Real.sqrt 2)
  (area2 : A2 = 900 * Real.sqrt 2)
  (height_difference : h_diff = 10) :
  ∃ h : ℝ, h = 30 :=
begin
  use 30,
  sorry
end

end distance_from_apex_l335_335715


namespace triangles_with_positive_integer_area_count_l335_335438

theorem triangles_with_positive_integer_area_count :
  let points := { p : (ℕ × ℕ) // 41 * p.1 + p.2 = 2017 }
  ∃ count, count = 600 ∧ ∀ (P Q : points), P ≠ Q →
    let area := (P.val.1 * Q.val.2 - Q.val.1 * P.val.2 : ℤ)
    0 < area ∧ (area % 2 = 0) := sorry

end triangles_with_positive_integer_area_count_l335_335438


namespace ellipse_properties_l335_335159

theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) 
  (f1 f2 a_vertex : ℝ × ℝ)
  (h_tri : ∃ (B : ℝ × ℝ), (B.fst = 0 ∧ B.snd = a) ∧ 
  (dist B ((-1), 0)) = 2 ∧ (dist B (1, 0)) = 2) 
  : (∃ (C : ℝ × ℝ → Prop), C = (λ P, (P.1^2) / a^2 + (P.2^2) / b^2 = 1) ∧
    ((1 : ℝ) / a = (1 / 2)) ∧ 
    (∃ (l : ℝ × ℝ → Prop), l = (λ P, (sqrt 5) * P.1 - 2 * P.2 - sqrt 5 = 0 ∨
    (sqrt 5) * P.1 + 2 * P.2 - sqrt 5 = 0))) :=
sorry

end ellipse_properties_l335_335159


namespace polar_to_rectangular_coords_l335_335850

theorem polar_to_rectangular_coords (r θ : ℝ) (cosθ sinθ : ℝ) 
  (h_r : r = 6) (h_θ : θ = π) (h_cos : cosθ = Real.cos π) (h_sin : sinθ = Real.sin π) :
  (r * cosθ, r * sinθ) = (-6, 0) := by
  rw [h_r, h_θ] at *
  rw [h_cos, h_sin]
  have cos_pi := Real.cos_pi
  have sin_pi := Real.sin_pi
  rw [cos_pi, sin_pi]
  simp [mul_neg, mul_zero]
  sorry

end polar_to_rectangular_coords_l335_335850


namespace max_expression_value_max_expression_achievable_l335_335467

noncomputable def max_expression : ℝ :=
  2

theorem max_expression_value (θ1 θ2 θ3 θ4 : ℝ) :
  cos θ1 * sin θ2 + cos θ2 * sin θ3 + cos θ3 * sin θ4 + cos θ4 * sin θ1 ≤ max_expression :=
sorry

theorem max_expression_achievable :
  ∃ θ1 θ2 θ3 θ4 : ℝ, cos θ1 * sin θ2 + cos θ2 * sin θ3 + cos θ3 * sin θ4 + cos θ4 * sin θ1 = max_expression :=
sorry

end max_expression_value_max_expression_achievable_l335_335467


namespace stratified_sampling_correct_l335_335776

-- Definition of the problem conditions
def num_students_class1 : ℕ := 54
def num_students_class2 : ℕ := 42
def total_students : ℕ := num_students_class1 + num_students_class2
def num_students_to_be_chosen : ℕ := 16
def sampling_probability : ℚ := num_students_to_be_chosen / total_students

-- Statements for the number of students chosen from each class
def num_students_chosen_class1 : ℕ := 9
def num_students_chosen_class2 : ℕ := 7

-- Proving the stratified sampling results
theorem stratified_sampling_correct : 
  (num_students_chosen_class1 = (num_students_class1 * sampling_probability).to_nat) ∧
  (num_students_chosen_class2 = (num_students_class2 * sampling_probability).to_nat) :=
by
  -- Proof goes here
  sorry

end stratified_sampling_correct_l335_335776


namespace inclination_angle_range_of_line_l335_335151

noncomputable def Point := Real × Real

def A : Point := (0, 2)
def B (m : ℝ) : Point := (-Real.sqrt 3, 3*m^2 + 12*m + 13)

def inclination_angle_range : Set (Real) :=
  {θ | 0 ≤ θ ∧ θ ≤ 30 ∨ 90 < θ ∧ θ < 180}

theorem inclination_angle_range_of_line (m : ℝ) :
  ∃ θ, θ ∈ inclination_angle_range ∧
  ∀ θ, let ⟨x1, y1⟩ := A;
           ⟨x2, y2⟩ := B m;
           tan θ = (y2 - y1) / (x2 - x1) in
  θ ∈ inclination_angle_range :=
sorry

end inclination_angle_range_of_line_l335_335151


namespace max_min_ratio_max_min_difference_max_min_sum_of_squares_l335_335114

theorem max_min_ratio (x y : ℝ) (H : x^2 + y^2 - 4 * x + 1 = 0) : 
  (y / x = sqrt 3) ∨ (y / x = -sqrt 3) :=
by sorry

theorem max_min_difference (x y : ℝ) (H : x^2 + y^2 - 4 * x + 1 = 0) : 
  (y - x = sqrt 6 - 2) ∨ (y - x = -sqrt 6 - 2) :=
by sorry

theorem max_min_sum_of_squares (x y : ℝ) (H : x^2 + y^2 - 4 * x + 1 = 0) : 
  (x^2 + y^2 = 7 + 4 * sqrt 3) ∨ (x^2 + y^2 = 7 - 4 * sqrt 3) :=
by sorry

end max_min_ratio_max_min_difference_max_min_sum_of_squares_l335_335114


namespace minimum_sum_of_distances_l335_335511

def point_on_parabola (a : ℝ) : ℝ × ℝ := (a^2, 2 * a)

def distance_to_line_l2 (a : ℝ) : ℝ := (a^2 + 1)

def distance_to_line_l1 (a : ℝ) : ℝ := (abs (4 * a^2 - 6 * a + 6)) / 5

def sum_of_distances (a : ℝ) : ℝ := (distance_to_line_l1 a) + (distance_to_line_l2 a)

theorem minimum_sum_of_distances :
  ∃ a : ℝ, sum_of_distances a = 2 := 
by
  have h_deriv : ∀ a : ℝ, deriv sum_of_distances a = (18 * a - 6) / 5
    := sorry
  
  have h_crit : deriv sum_of_distances (1 / 3) = 0
    := sorry

  have h_min_val : sum_of_distances (1 / 3) = 2 
    := sorry
  
  use (1 / 3)
  exact h_min_val

end minimum_sum_of_distances_l335_335511


namespace largest_prime_factor_of_expression_l335_335736

theorem largest_prime_factor_of_expression :
  ∃ p : ℕ, prime p ∧ p = 241 ∧ ∀ q : ℕ, q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → prime q → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l335_335736


namespace area_OBEC_is_27_l335_335779

noncomputable def area_quadrilateral_O_B_E_C : ℝ :=
  let A := (4 : ℝ, 0)
  let B := (0, 12)
  let C := (6, 0)
  let D := (0, 6)
  let E := (3, 3)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ := 1 / 2 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))
  area_triangle (0, 0) B E + area_triangle (0, 0) E C

theorem area_OBEC_is_27 : area_quadrilateral_O_B_E_C = 27 := by
  unfold area_quadrilateral_O_B_E_C
  sorry

end area_OBEC_is_27_l335_335779


namespace trig_identity_l335_335147

theorem trig_identity (α : ℝ) (h : 2 * sin (α - π / 3) = (2 - sqrt 3) * cos α) :
  sin (2 * α) + 3 * (cos α) ^ 2 = 7 / 5 :=
by sorry

end trig_identity_l335_335147


namespace sum_of_squares_of_roots_l335_335427

theorem sum_of_squares_of_roots:
  let q := 2 * (x^2) - 9 * x + 7
  (x1 x2 : ℝ) (h_roots : q.has_root x1 ∧ q.has_root x2):
  x1 + x2 = (9 / 2) → x1 * x2 = (7 / 2) → (x1^2 + x2^2) = 53 / 4 :=
by
  intros x1 x2 h_roots h_sum h_product
  let q := 2 * (x^2) - 9 * x + 7
  have h_sum_sq := (9 / 2) ^ 2
  have h_double_prod := 2 * (7 / 2)
  have h_rhs := h_sum_sq - h_double_prod
  exact h_rhs
  sorry

end sum_of_squares_of_roots_l335_335427


namespace solve_system_l335_335283

theorem solve_system :
  ∃ x y : ℝ, (x^2 - 9 * y^2 = 0 ∧ 2 * x - 3 * y = 6) ∧ (x = 6 ∧ y = 2) ∨ (x = 2 ∧ y = -2 / 3) :=
by
  -- The proof will go here
  sorry

end solve_system_l335_335283


namespace incorrect_tripling_statements_l335_335046

-- Definitions and conditions
def cylindrical_volume (r h : ℝ) : ℝ := π * r^2 * h
def square_area (s : ℝ) : ℝ := s^2
def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3
def fraction_eq (a b : ℝ) (b_nonzero : b ≠ 0) : Prop := a / b
def triple_negative (x : ℝ) (x_neg : x < 0) : Prop := 3 * x

-- Incorrect statements (Goal)
theorem incorrect_tripling_statements :
  ¬ (∀ s,  (square_area (3 * s) = 3 * square_area s)) ∧ ¬ (∀ r, (sphere_volume (3 * r) = 4 * sphere_volume r)) :=
by 
  -- provide proof here; it would involve calculations and logic as shown in the solution.
  sorry

end incorrect_tripling_statements_l335_335046


namespace painting_time_calculation_l335_335863

theorem painting_time_calculation :
  let doug_rate := (1 : ℚ) / 5
  let dave_rate := (1 : ℚ) / 7
  let ellen_rate := (1 : ℚ) / 9
  let combined_rate := doug_rate + dave_rate + ellen_rate

  (combined_rate * (t - 1) = 1) → t = 458 / 143 :=
by
  intros
  let doug_rate : ℚ := 1 / 5
  let dave_rate : ℚ := 1 / 7
  let ellen_rate : ℚ := 1 / 9
  let combined_rate : ℚ := doug_rate + dave_rate + ellen_rate
  let t := (458 : ℚ) / 143
  sorry

end painting_time_calculation_l335_335863


namespace probability_of_selection_l335_335950

-- Defining the conditions
def shirt_count := 5
def pants_count := 4
def socks_count := 7
def total_articles := shirt_count + pants_count + socks_count
def draw_count := 5

-- Function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℚ := nat.choose n k

-- Theorem statement
theorem probability_of_selection : 
  let total_ways_to_choose := binom total_articles draw_count
      choose_shirts := binom shirt_count 2
      choose_pants := binom pants_count 1
      choose_socks := binom socks_count 2
      favorable_ways := choose_shirts * choose_pants * choose_socks
  in (favorable_ways / total_ways_to_choose) = (35 / 182) :=
by
  sorry

end probability_of_selection_l335_335950


namespace log_inequalities_l335_335095

theorem log_inequalities (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  log x < log y < 0 → (0 < x ∧ x < y ∧ y < 1) :=
by
  intro h
  sorry

end log_inequalities_l335_335095


namespace allie_toy_count_l335_335806

theorem allie_toy_count (total_value : ℤ) (special_toy_value : ℤ) (other_toy_value : ℤ) (total_toys : ℤ) 
  (h1 : total_value = 52) (h2 : special_toy_value = 12) (h3 : other_toy_value = 5) : 
  total_toys = 9 :=
by
  -- assumption that number of $5 toys is x
  let x := (total_value - special_toy_value) / other_toy_value
  -- calculate total number of toys
  have ht : total_toys = x + 1, by sorry
  -- verify the total number of toys is 9
  have h_toys : ht = 9, by sorry
  -- hence the proven statement that total_toys = 9
  sorry

end allie_toy_count_l335_335806


namespace max_discount_l335_335767

def cost_price : ℝ := 700
def marked_price : ℝ := 1100
def min_profit_margin : ℝ := 0.10

def selling_price (discount_percentage : ℝ) : ℝ :=
  marked_price * (1 - discount_percentage / 100)

def profit_margin (selling_price : ℝ) : ℝ :=
  (selling_price - cost_price) / cost_price

theorem max_discount:
  ∃ (x : ℝ) (h : x ≤ 70), profit_margin (selling_price x) ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_l335_335767


namespace express_in_scientific_notation_l335_335974

theorem express_in_scientific_notation : (250000 : ℝ) = 2.5 * 10^5 := 
by {
  -- proof
  sorry
}

end express_in_scientific_notation_l335_335974


namespace triangle_is_isosceles_l335_335998

variables {A B C : ℝ}

theorem triangle_is_isosceles
  (h_angle_sum : A + B + C = π)
  (h_cos_sin : 2 * cos B * sin A = sin C) :
  A = B ∨ B = C ∨ A = C :=
begin
  sorry
end

end triangle_is_isosceles_l335_335998


namespace pure_imaginary_of_square_neg_l335_335917

variable {z z1 z2 x : ℂ}

theorem pure_imaginary_of_square_neg (hz : z ∈ ℂ) (hz_neg : z^2 < 0) : (∃ a : ℝ, z = a * complex.I) :=
by
  sorry

end pure_imaginary_of_square_neg_l335_335917


namespace tiling_difference_l335_335273

-- Define the board size
def board_size : ℕ :=
  533

-- Define a checkerboard pattern and the properties of positions (A and B)
structure Board (n : ℕ) :=
  (is_checkerboard : ∀ i j, (i + j) % 2 = 0 → (cell : (ℕ, ℕ)) → Prop)

-- Conditional properties of cells A and B
variables (A B : (ℕ, ℕ)) (same_color : (A.1 + A.2) % 2 = (B.1 + B.2) % 2)

-- Define the domino tiling problem
def can_tile (board : Board board_size) (cell : (ℕ, ℕ)) : Prop := sorry

-- Required proposition to prove
theorem tiling_difference
  (board : Board board_size)
  (A B : (ℕ, ℕ))
  (same_color : (A.1 + A.2) % 2 = (B.1 + B.2) % 2) :
  can_tile board A ≠ can_tile board B := sorry

end tiling_difference_l335_335273


namespace q_r_ratio_l335_335654

theorem q_r_ratio (total : ℕ) (p_ratio : ℕ) (q_ratio : ℕ) (r : ℕ) :
  total = 1210 ∧ p_ratio = 5 ∧ q_ratio = 4 ∧ r = 400 →
  (let q := 4 * 90 in q : r = 9 : 10) :=
sorry

end q_r_ratio_l335_335654


namespace PetyaWinsAgainstSasha_l335_335640

def MatchesPlayed (name : String) : Nat :=
if name = "Petya" then 12 else if name = "Sasha" then 7 else if name = "Misha" then 11 else 0

def TotalGames : Nat := 15

def GamesMissed (name : String) : Nat :=
if name = "Petya" then TotalGames - MatchesPlayed name else 
if name = "Sasha" then TotalGames - MatchesPlayed name else
if name = "Misha" then TotalGames - MatchesPlayed name else 0

def CanNotMissConsecutiveGames : Prop := True

theorem PetyaWinsAgainstSasha : (GamesMissed "Misha" = 4) ∧ CanNotMissConsecutiveGames → 
  ∃ (winsByPetya : Nat), winsByPetya = 4 :=
by
  sorry

end PetyaWinsAgainstSasha_l335_335640


namespace incorrect_relationship_f_pi4_f_pi_l335_335248

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_derivative_exists : ∀ x : ℝ, DifferentiableAt ℝ f x
axiom f_derivative_lt_sin2x : ∀ x : ℝ, 0 < x → deriv f x < (sin x) ^ 2
axiom f_symmetric_property : ∀ x : ℝ, f (-x) + f x = 2 * (sin x) ^ 2

theorem incorrect_relationship_f_pi4_f_pi : ¬ (f (π / 4) < f π) :=
by sorry

end incorrect_relationship_f_pi4_f_pi_l335_335248


namespace teaching_staff_in_sample_l335_335796

-- Given conditions
def total_staff : ℕ := 200
def administrative_staff : ℕ := 24
def teaching_support_ratio : ℕ × ℕ := (10, 1)
def sample_size : ℕ := 50

-- Prove the number of teaching staff to be included in the sample is 40
theorem teaching_staff_in_sample :
  let teaching_staff := 10 * (total_staff - administrative_staff) / 11,
      support_staff := (total_staff - administrative_staff) / 11,
      teaching_sample_ratio := teaching_staff / total_staff,
      teaching_sample := teaching_sample_ratio * sample_size
  in teaching_sample = 40 :=
by
  sorry

end teaching_staff_in_sample_l335_335796


namespace decrypt_message_l335_335368

-- Definitions of positions of Russian letters
def letterPosition (c : Char) : Nat := 
  match c with
  | 'А' => 1  | 'Б' => 2  | 'В' => 3  | 'Г' => 4  | 'Д' => 5
  | 'Е' => 6  | 'Ё' => 7  | 'Ж' => 8  | 'З' => 9  | 'И' => 10
  | 'Й' => 11 | 'К' => 12 | 'Л' => 13 | 'М' => 14 | 'Н' => 15
  | 'О' => 16 | 'П' => 17 | 'Р' => 18 | 'С' => 19 | 'Т' => 20
  | 'У' => 21 | 'Ф' => 22 | 'Х' => 23 | 'Ц' => 24 | 'Ч' => 25
  | 'Ш' => 26 | 'Щ' => 27 | 'Ъ' => 28 | 'Ы' => 29 | 'Ь' => 30
  | _ => 0 -- invalid character case

-- Function for decryption
def decryptLetter (cipherTextLetter : Char) (keyLetter : Char) : Char :=
  let decryptedPos := (letterPosition cipherTextLetter + 30 - letterPosition keyLetter) % 30
  match decryptedPos with
  | 0  => 'Я'
  | 1  => 'А'
  | 2  => 'Б'
  -- Add the rest of the cases here
  | _  => '?'

-- Condition: Decryption of each letter using keys А, Б, В should lead into our plaintext 'НАШКОРРЕСПОНДЕНТ'
def decryptionIsValid (cipherText : String) (plainText : String) : Prop :=
  cipherText.length = plainText.length ∧
  ∃ (keys : String),
    keys.length = cipherText.length ∧
    (∀ i, i < cipherText.length → 
    decryptLetter (cipherText.get ⟨i, sorry⟩) (keys.get ⟨i, sorry⟩) = plainText.get ⟨i, sorry⟩)

theorem decrypt_message : 
  decryptionIsValid 
    "РБЬНПТСИТСРРЕЗОХ" -- cipherText
    "НАШКОРРЕСПОНДЕНТ" -- plainText
:=
sorry

end decrypt_message_l335_335368


namespace intersection_M_N_l335_335118

open Set

def M : Set ℝ := { x | -4 < x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } :=
sorry

end intersection_M_N_l335_335118


namespace inequality_holds_l335_335102

theorem inequality_holds
    (k l m : ℤ) 
    (a b c : ℂ) 
    (h : ℕ) :
    abs k + abs l + abs m ≥ 2007 → 
    abs (k * a + l * b + m * c) > 1 / h := 
by sorry

end inequality_holds_l335_335102


namespace max_perimeter_of_rectangle_with_area_36_l335_335793

theorem max_perimeter_of_rectangle_with_area_36 :
  ∃ l w : ℕ, l * w = 36 ∧ (∀ l' w' : ℕ, l' * w' = 36 → 2 * (l + w) ≥ 2 * (l' + w')) ∧ 2 * (l + w) = 74 := 
sorry

end max_perimeter_of_rectangle_with_area_36_l335_335793


namespace trig_identity_proof_l335_335154

-- Define the point P given in the problem
def P := (4 / 5 : ℝ, 3 / 5 : ℝ)

-- Define the angle α such that its terminal side intersects the unit circle at point P
axiom alpha : ℝ
axiom intersects_unit_circle : cos alpha = 4 / 5 ∧ sin alpha = 3 / 5

-- Define the statement to prove
theorem trig_identity_proof :
  sin alpha = 3 / 5 ∧ cos alpha = 4 / 5 ∧ tan alpha = 3 / 4 ∧
  (sin (π + alpha) + 2 * sin (π / 2 - alpha)) / (2 * cos (π - alpha)) = -25 / 64 :=
by
  -- You can omit the proof details
  sorry

end trig_identity_proof_l335_335154


namespace cutting_wire_random_event_l335_335441

noncomputable def length : ℝ := sorry

def is_random_event (a : ℝ) : Prop := sorry

theorem cutting_wire_random_event (a : ℝ) (h : a > 0) :
  is_random_event a := 
by
  sorry

end cutting_wire_random_event_l335_335441


namespace triangle_min_ab_l335_335595

noncomputable def min_ab (a b c : ℝ) (A B C : ℝ) : ℝ :=
  a * b

theorem triangle_min_ab
  (a b c : ℝ)
  (A B C : ℝ)
  (sin_a : ℝ → ℝ)
  (cos_b : ℝ → ℝ)
  (sin_b : ℝ → ℝ)
  (sin_c : ℝ → ℝ)
  (cos_c : ℝ → ℝ)
  (two_sin_a_plus_sin_b_eq_two_sin_c_cos_b : 2 * sin_a A + sin_b B = 2 * sin_c C * cos_b B)
  (area_eq : (sqrt 3 / 2) * c = 1 / 2 * a * b * sin_c C) :
  let min_value := min_ab a b c A B C in
  min_value = 12 :=
sorry

end triangle_min_ab_l335_335595


namespace time_to_reach_ship_l335_335013

-- Conditions in Lean 4
def rate : ℕ := 22
def depth : ℕ := 7260

-- The theorem that we want to prove
theorem time_to_reach_ship : depth / rate = 330 := by
  sorry

end time_to_reach_ship_l335_335013


namespace imaginary_part_eq_neg2_l335_335686

-- Define the complex numbers and the conjugate.
def conj (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Define the given complex numbers.
def numerator : ℂ := ⟨4, 2⟩
def denominator : ℂ := ⟨-1, 2⟩

-- Define the expression.
def expr : ℂ := numerator / denominator

-- State the theorem that the imaginary part of the expression is -2.
theorem imaginary_part_eq_neg2 : expr.im = -2 := 
sorry

end imaginary_part_eq_neg2_l335_335686


namespace min_distance_parallel_lines_l335_335638

theorem min_distance_parallel_lines :
  let P := {p : ℝ × ℝ // 3 * p.1 + 4 * p.2 = 12}
  let Q := {q : ℝ × ℝ // 3 * q.1 + 4 * q.2 = -5 / 2}
  ∃ p : P, ∃ q : Q, ∥p.val - q.val∥ = 29 / 10 := by
sorry

end min_distance_parallel_lines_l335_335638


namespace geometric_transformation_l335_335683

theorem geometric_transformation (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = a * x + b) (h2 : ∀ x : ℝ, f (f x) = x) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ ¬ False) :=
by
  have h_eq := funext h2
  simp only [h1, function.funext_iff, forall_const] at h_eq
  sorry

end geometric_transformation_l335_335683


namespace ant_prob_bottom_vertex_l335_335435

theorem ant_prob_bottom_vertex :
  let top := 1
  let first_layer := 4
  let second_layer := 4
  let bottom := 1
  let prob_first_layer := 1 / first_layer
  let prob_second_layer := 1 / second_layer
  let prob_bottom := 1 / (second_layer + bottom)
  prob_first_layer * prob_second_layer * prob_bottom = 1 / 80 :=
by
  sorry

end ant_prob_bottom_vertex_l335_335435


namespace sum_of_first_n_natural_numbers_single_digit_l335_335176

theorem sum_of_first_n_natural_numbers_single_digit (n : ℕ) :
  (∃ a : ℕ, a ≤ 9 ∧ (a ≠ 0) ∧ 37 * (3 * a) = n * (n + 1) / 2) ↔ (n = 36) :=
by
  sorry

end sum_of_first_n_natural_numbers_single_digit_l335_335176


namespace largest_prime_factor_of_expression_l335_335740

theorem largest_prime_factor_of_expression : 
  ∃ p, prime p ∧ p ≥ 2 ∧ 
  (∀ q, (q ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) → prime q → q ≤ p) ∧ 
  p = 241 :=
by
  sorry

end largest_prime_factor_of_expression_l335_335740


namespace chocolate_eggs_total_weight_l335_335625

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end chocolate_eggs_total_weight_l335_335625


namespace circumcircle_tangent_to_fixed_circle_l335_335104

open EuclideanGeometry

-- Definitions representing the given conditions
variables {A O B C D E : Point}
variables {circleO : Circle}
variables {circleE : Circle}
variables {tangentBC : Line}
variables {circumcircleABC : Circle}
variables [FinitePointSet]

-- Conditions
def fixed_point_A_outside_circle_O (A : Point) (circleO : Circle) : Prop := 
  ¬ circleO.contains A

def points_B_C_on_tangents (A B C : Point) (tangentBJ tangentCK: Line) (circleO : Circle) : Prop := 
  IsTangent tangentBJ circleO ∧ tangentBJ.contains B ∧
  IsTangent tangentCK circleO ∧ tangentCK.contains C

def line_BC_tangent_at_D (B C D : Point) (tangentBC : Line) (circleO : Circle) : Prop := 
  IsTangent tangentBC circleO ∧ tangentBC.contains D

-- The theorem statement
theorem circumcircle_tangent_to_fixed_circle  
  (A O B C D E : Point) 
  (circleO : Circle) 
  (circleE : Circle) 
  (tangentBC : Line) 
  (circumcircleABC : Circle)
  [FinitePointSet] :
  fixed_point_A_outside_circle_O A circleO →
  points_B_C_on_tangents A B C (Line.through A B) (Line.through A C) circleO →
  line_BC_tangent_at_D B C D tangentBC circleO →
  Tangent circumcircleABC circleE :=
sorry

end circumcircle_tangent_to_fixed_circle_l335_335104


namespace a13_is_144_l335_335508

def sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * real.sqrt (a n) + 1

theorem a13_is_144 : ∀ (a : ℕ → ℝ) (n : ℕ), sequence a n → a 12 = 144 := 
by
  intro a n h,
  sorry

end a13_is_144_l335_335508


namespace find_stock_worth_before_fees_l335_335723

-- Given conditions
def cost_price : ℝ := 94.2 
def discount_rate : ℝ := 0.06
def brokerage_fee_rate : ℝ := 1 / 500

-- The worth of the stock before discount and brokerage fee is represented by X
noncomputable def stock_worth_before_fees (X : ℝ) : Prop :=
  cost_price = X - (discount_rate * X) + (brokerage_fee_rate * (X - (discount_rate * X)))

-- The theorem statement
theorem find_stock_worth_before_fees : ∃ X : ℝ, stock_worth_before_fees X ∧ X = 100 := by
  sorry

end find_stock_worth_before_fees_l335_335723


namespace area_of_triangle_DEF_l335_335009

variable (Q : Point) (ΔDEF : Triangle) (u1 u2 u3 : Triangle)

-- We assume that the lines drawn through Q parallel to the sides of ΔDEF divide it into u1, u2, and u3
/-
  Q is a point inside ΔDEF and lines drawn through Q parallel to the sides of ΔDEF divide it into u1, u2, u3
  with areas 16, 25, and 36 respectively.
-/
variable (h₁ : u1.area = 16)
variable (h₂ : u2.area = 25)
variable (h₃ : u3.area = 36)

-- Prove that the area of ΔDEF is 225
theorem area_of_triangle_DEF : ΔDEF.area = 225 := by
  sorry

end area_of_triangle_DEF_l335_335009


namespace sin_D_value_l335_335585

-- Define the right triangle DEF with the given conditions
variable {D E F : ℝ} (h1 : ∠E = 90) (h2 : 2 * real.sin D = 5 * real.cos D)

-- Define the mathematical equivalence to be proved
theorem sin_D_value :
  real.sin D = (5 * real.sqrt 29) / 29 :=
sorry

end sin_D_value_l335_335585


namespace all_a_n_are_integers_l335_335163

noncomputable def a : ℕ → ℤ
| 0       := 2
| 1       := 1
| 2       := 1
| 3       := 997
| (n+4) := (1993 + a (n+3) * a (n+2)) / a (n+1)

theorem all_a_n_are_integers : ∀ n : ℕ, ∃ z : ℤ, a n = z :=
by sorry

end all_a_n_are_integers_l335_335163


namespace inscribed_circle_radius_of_triangle_l335_335082

theorem inscribed_circle_radius_of_triangle (a b c : ℕ)
  (h₁ : a = 50) (h₂ : b = 120) (h₃ : c = 130) :
  ∃ r : ℕ, r = 20 :=
by sorry

end inscribed_circle_radius_of_triangle_l335_335082


namespace point_on_line_l335_335448

-- Define the points (3, 3) and (1, 0)
def P1 : ℝ × ℝ := (3, 3)
def P2 : ℝ × ℝ := (1, 0)

-- Define the point (x, 8)
def Px (x : ℝ) : ℝ × ℝ := (x, 8)

-- Define the slope of the line passing through the points (3, 3) and (1, 0)
def slope (P1 P2 : ℝ × ℝ) : ℝ := (P1.2 - P2.2) / (P1.1 - P2.1)

-- Define the proof statement
theorem point_on_line (x : ℝ) : 
  slope P1 P2 = 3 / 2 → 
  slope (Px x) P1 = 3 / 2 → 
  x = 19 / 3 :=
by
  intro h_slope_P1_P2 h_slope_Px_P1
  sorry

end point_on_line_l335_335448


namespace most_likely_is_odd_l335_335408

def numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 30}

def multiples_of_10 := {n ∈ numbers | n % 10 = 0}

def odd_numbers := {n ∈ numbers | n % 2 = 1}

def includes_digit_3 := {n ∈ numbers | ('3' ∈ n.digits 10)}

def multiples_of_5 := {n ∈ numbers | n % 5 = 0}

def includes_digit_2 := {n ∈ numbers | ('2' ∈ n.digits 10)}

theorem most_likely_is_odd :
  (∀ t ∈ {multiples_of_10, includes_digit_3, multiples_of_5, includes_digit_2}, 
    (odd_numbers.card > t.card)) :=
by
  sorry

end most_likely_is_odd_l335_335408


namespace find_m_n_l335_335073

theorem find_m_n :
  ∃ m n : ℕ, m! + 12 = n^2 ∧ (m, n) = (4, 6) :=
by
  use 4, 6
  split
  · calc (4! + 12) = 24 + 12 : by rw Nat.factorial_succ
                 ... = 36     : by norm_num
                 ... = 6^2    : by norm_num
  · sorry

end find_m_n_l335_335073


namespace remaining_students_l335_335704

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l335_335704


namespace cylinder_ellipse_major_axis_l335_335008

-- Given a right circular cylinder of radius 2
-- and a plane intersecting it forming an ellipse
-- with the major axis being 50% longer than the minor axis,
-- prove that the length of the major axis is 6.

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ) (major minor : ℝ),
    r = 2 → major = 1.5 * minor → minor = 2 * r → major = 6 :=
by
  -- Proof step to be filled by the prover.
  sorry

end cylinder_ellipse_major_axis_l335_335008


namespace nonneg_integer_solutions_otimes_l335_335053

noncomputable def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

theorem nonneg_integer_solutions_otimes :
  {x : ℕ | otimes 2 x ≥ 3} = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_otimes_l335_335053


namespace sequence_count_l335_335895

def num_sequences (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem sequence_count :
  let x := 490
  let y := 510
  let a : (n : ℕ) → ℕ := fun n => if n = 0 then 0 else if n = 1000 then 2020 else sorry
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (a (k + 1) - a k = 1 ∨ a (k + 1) - a k = 3)) →
  (∃ binomial_coeff, binomial_coeff = num_sequences 1000 490) :=
by sorry

end sequence_count_l335_335895


namespace lambda_value_l335_335905

theorem lambda_value (C : Set Point) (F1 F2 P : Point) (a λ : ℝ)
  (h_F1F2 : (F1, F2) ∈ Foci C)
  (h_P : P ∈ C)
  (h_angle : ∠F1PF2 = 60)
  (h_lambda_pos : λ > 1)
  (h_dist : dist P F1 = λ * dist P F2)
  (h_eccentricity : eccentricity C = sqrt 7 / 2) :
  λ = 3 :=
sorry

end lambda_value_l335_335905


namespace lecture_scheduling_l335_335004

theorem lecture_scheduling :
  (∃ schedule : List String, 
    (∀ schedule : List String, schedule.length = 7) ∧
    (∀ i j : ℕ, i < j → schedule[i] = "Smith" → schedule[j] = "Jones") ∧
    (∀ i j : ℕ, i < j → schedule[i] = "Green" → schedule[j] = "Brown") ∧
    (∃ valid_schedules : ℕ, valid_schedules = 1260)) sorry

end lecture_scheduling_l335_335004


namespace intersection_M_N_l335_335132

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335132


namespace marble_ratio_l335_335407

-- Let Allison, Angela, and Albert have some number of marbles denoted by variables.
variable (Albert Angela Allison : ℕ)

-- Given conditions.
axiom h1 : Angela = Allison + 8
axiom h2 : Allison = 28
axiom h3 : Albert + Allison = 136

-- Prove that the ratio of the number of marbles Albert has to the number of marbles Angela has is 3.
theorem marble_ratio : Albert / Angela = 3 := by
  sorry

end marble_ratio_l335_335407


namespace percentage_increase_is_50_l335_335785

def papaya_growth (P : ℝ) : Prop :=
  let growth1 := 2
  let growth2 := 2 * (1 + P / 100)
  let growth3 := 1.5 * growth2
  let growth4 := 2 * growth3
  let growth5 := 0.5 * growth4
  growth1 + growth2 + growth3 + growth4 + growth5 = 23

theorem percentage_increase_is_50 :
  ∃ (P : ℝ), papaya_growth P ∧ P = 50 := by
  sorry

end percentage_increase_is_50_l335_335785


namespace evaluate_expression_l335_335461

theorem evaluate_expression : 3 - 5 * (6 - 2^3) / 2 = 8 :=
by
  sorry

end evaluate_expression_l335_335461


namespace binom_8_5_eq_56_l335_335837

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l335_335837


namespace find_c_l335_335885

noncomputable def L (n : ℕ) : ℕ := Nat.lcm (Finset.range (n + 1)).extents

theorem find_c (n : ℕ) (h : 0 < n) :
  ∃ c : ℝ, c = 1 / (L n) ∧ (∀ (f : Polynomial ℤ) (a b : ℤ), 
  f.degree = n ∧ f.eval a ≠ f.eval b → c ≤ | (f.eval a - f.eval b) / (a - b) |) :=
by
  sorry

end find_c_l335_335885


namespace proof_numberOfCorrectStatements_l335_335295

-- Definition of the Gaussian function
def gaussian (x : ℝ) : ℝ := floor x

-- Definition of the fractional part function
def fractional (x : ℝ) : ℝ := x - gaussian x

-- Statement 1
def statement1 : Prop := gaussian (-4.1) = -4

-- Statement 2
def statement2 : Prop := fractional 3.5 = 0.5

-- Statement 3
def statement3 : Prop := ∀ x : ℝ, gaussian x = -3 ↔ -3 ≤ x ∧ x < -2

-- Statement 4
def statement4 : Prop := ∀ x : ℝ, (2.5 < x ∧ x ≤ 3.5) → (0 ≤ fractional x ∧ fractional x < 1)

-- Prove the number of correct statements
def numberOfCorrectStatements : ℕ := 
  if statement2 then
    if statement3 then
      if statement4 then
        3
      else
        2
    else
      1
  else
    0

theorem proof_numberOfCorrectStatements : numberOfCorrectStatements = 3 := 
  by sorry

end proof_numberOfCorrectStatements_l335_335295


namespace angle_B43_B44_B45_is_90_l335_335757

theorem angle_B43_B44_B45_is_90 :
  ∀ (B : ℕ → Point) (n : ℕ),
  (-- Conditions
  is_square (B 1) (B 2) (B 3) (B 4) ∧
  (∀ m : ℕ, B (m + 4) divides (B m) (B (m + 1)) 1 3)--
  -- Conclusion
  ∠ (B 43) (B 44) (B 45) = 90) :=
sorry

end angle_B43_B44_B45_is_90_l335_335757


namespace valid_four_digit_numbers_count_l335_335944

theorem valid_four_digit_numbers_count : 
  ∀ (digits : Finset ℕ), digits = {0, 1, 2, 3} → 
  ∀ (count: ℕ), count = (digits.perms.filter (λ p, p.head ≠ 0)).card → 
  count = 18 :=
by
  sorry

end valid_four_digit_numbers_count_l335_335944


namespace explicit_f_monotonic_intervals_g_closer_ln_l335_335855

noncomputable def f (x : ℝ) : ℝ := (f''(1) / 2) * Real.exp (2 * x - 2) + x^2 - 2 * f(0) * x
def g (x : ℝ) (a : ℝ) : ℝ := f (x / 2) - (1 / 4) * x^2 + (1 - a) * x + a

theorem explicit_f (f' f'' : ℝ → ℝ) (h1 : ∀ x, f'(x) = f'(1) * Real.exp(2 * x - 2) + 2 * x - 2 * f(0))
  (h2 : f(0) = (f'(1) / 2) * Real.exp(-2)) : ∀ x, f(x) = Real.exp(2 * x) + x^2 - 2 * x :=
sorry

theorem monotonic_intervals_g (a : ℝ) : 
  (∀ x, g (x) (a) = Real.exp(x) - a * (x - 1)) → 
  (∀ x, (a ≤ 0 ∨ (a > 0 → (∀ x < Real.log a, g'(x, a) < 0 ∧ ∀ x > Real.log a, g'(x, a) > 0))) :=
sorry

theorem closer_ln (a : ℝ) (x : ℝ) (h : a ≥ 2) (hx: x ≥ 1) :
  |Real.exp(1) - x| ≤ |e^(x-1) + a - Real.log(x)| :=
sorry

end explicit_f_monotonic_intervals_g_closer_ln_l335_335855


namespace statement2_true_l335_335007

def digit : ℕ := sorry

def statement1 : Prop := digit = 2
def statement2 : Prop := digit ≠ 3
def statement3 : Prop := digit = 5
def statement4 : Prop := digit ≠ 6

def condition : Prop := (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧
                        (¬ statement1 ∨ ¬ statement2 ∨ ¬ statement3 ∨ ¬ statement4)

theorem statement2_true (h : condition) : statement2 :=
sorry

end statement2_true_l335_335007


namespace monotonic_function_range_maximum_value_condition_function_conditions_l335_335920

-- Part (1): Monotonicity condition
theorem monotonic_function_range (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0) ↔ (m ≥ 3) := sorry

-- Part (2): Maximum value condition
theorem maximum_value_condition (m : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4) ↔ (m = -2) := sorry

-- Combined statement (optional if you want to show entire problem in one go)
theorem function_conditions (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0 ∧ 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4)) ↔ (m = -2 ∨ m ≥ 3) := sorry

end monotonic_function_range_maximum_value_condition_function_conditions_l335_335920


namespace sqrt_221_range_l335_335484

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 :=
by
  have h1 : 14^2 = 196 := rfl
  have h2 : 15^2 = 225 := rfl
  have h3 : 196 < 221 := by linarith
  have h4 : 221 < 225 := by linarith
  have sqrt_ineq : ∀ (a b : ℕ), a < b → Real.sqrt a < Real.sqrt b := sorry
  exact ⟨sqrt_ineq 196 221 h3, sqrt_ineq 221 225 h4⟩

end sqrt_221_range_l335_335484


namespace number_of_x_intercepts_l335_335444

-- Define the interval
def interval (a b : ℝ) := {x : ℝ | a < x ∧ x < b}

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (1 / x)

-- Lean statement to prove the number of x-intercepts
theorem number_of_x_intercepts : ∃ n : ℕ, n = 5729 ∧ 
  ∀ x ∈ interval 0.00005 0.0005, f x = 0 → true :=
begin
  sorry
end

end number_of_x_intercepts_l335_335444


namespace find_fx_neg_l335_335517

-- Conditions
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def func_nonneg (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = -x^2 + 4x

-- Prove
theorem find_fx_neg (f : ℝ → ℝ) (h_even : is_even_function f) (h_nonneg : func_nonneg f) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 4x :=
by
  intros x h
  have h_pos : -x > 0 := by linarith
  have h_f_neg_x : f (-x) = -x^2 + 4 * -x := h_nonneg f (-x) h_pos
  rw [h_even f, h_f_neg_x]
  simp
  sorry

end find_fx_neg_l335_335517


namespace min_value_frac_inv_l335_335901

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := abs (x - m + 1) - 2

theorem min_value_frac_inv (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_even : ∀ x, f x m = f (-x) m) (h_condition : f a m + f (2 * b) m = m) :
    (∃ (a b : ℝ), a + 2 * b = 5 ∧ (∀ a b, (0 < a ∧ 0 < b) → (a + 2 * b = 5 → (1 / a + 2 / b ≥ 9 / 5)))) :=
begin
  sorry
end

end min_value_frac_inv_l335_335901


namespace total_emails_vacation_l335_335763

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l335_335763


namespace hyperbola_eccentricity_sqrt_five_l335_335894

-- Define the hyperbola and its properties
variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Define the parabola equation
def parabola_eq (y x : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1

-- Define the intersection points P(1,2) lies on one of the asymptotes
def asymptote_intersect_parabola (P : ℝ × ℝ) : Prop :=
  -- Asymptote equation
  let (x, y) := P in b * x = a * y

-- The line passing through P and Q passes through the focus of the parabola
def pq_through_focus (P Q focus : ℝ × ℝ) : Prop :=
  let (xf, yf) := focus in -- Coordinates of the parabola's focus are (1,0)
  let (xp, yp) := P in
  let (xq, yq) := Q in
  (yp - yq) * (xf - xq) = (yf - yq) * (xp - xq)

-- Prove the eccentricity is sqrt(5)
theorem hyperbola_eccentricity_sqrt_five :
  ∀ (P Q : ℝ × ℝ),
    parabola_eq P.2 P.1 ∧ parabola_eq Q.2 Q.1 ∧
    asymptote_intersect_parabola P ∧ asymptote_intersect_parabola Q ∧
    pq_through_focus P Q (1, 0) →
    let c := Real.sqrt (a^2 + b^2) in
    let e := c / a in
    e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_sqrt_five_l335_335894


namespace square_and_product_l335_335936

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : (x = 42) ∧ ((x + 2) * (x - 2) = 1760) :=
by
  sorry

end square_and_product_l335_335936


namespace dishwasher_spending_l335_335862

theorem dishwasher_spending (E : ℝ) (h1 : E > 0) 
    (rent : ℝ := 0.40 * E)
    (left_over : ℝ := 0.28 * E)
    (spent : ℝ := 0.72 * E)
    (dishwasher : ℝ := spent - rent)
    (difference : ℝ := rent - dishwasher) :
    ((difference / rent) * 100) = 20 := 
by
  sorry

end dishwasher_spending_l335_335862


namespace min_consecutive_numbers_to_five_consecutive_ones_l335_335889

theorem min_consecutive_numbers_to_five_consecutive_ones : 
  ∃ (n : ℕ), n = 112 ∧ (∀ k < 112, consecutiveOnes (writeNumbersUpTo k) < 5) ∧ consecutiveOnes (writeNumbersUpTo 112) = 5 := by
  sorry

end min_consecutive_numbers_to_five_consecutive_ones_l335_335889


namespace symmetric_point_xoy_l335_335932

theorem symmetric_point_xoy (P : ℝ × ℝ × ℝ) (hP : P = (1, 2, 3)) :
  ∃ Q : ℝ × ℝ × ℝ, Q = (1, 2, -3) :=
by {
  use (1, 2, -3),
  sorry
}

end symmetric_point_xoy_l335_335932


namespace solve_equation_l335_335281

theorem solve_equation (x : ℝ) (h1 : x + 2 ≠ 0) (h2 : 3 - x ≠ 0) :
  (3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -15 / 2 :=
by
  sorry

end solve_equation_l335_335281


namespace inequality_k_factorial_l335_335648

theorem inequality_k_factorial (k : ℕ) (hk : k ≥ 1) : 
  (∑ i in Finset.range k, (i + 1) * (i + 1)!) < (k + 1)! :=
sorry

end inequality_k_factorial_l335_335648


namespace evaluate_expression_l335_335158

-- Given the mathematical problem and conditions:
def θ_ang (θ : ℝ) : Prop := (∃ θ : ℝ, tan θ = 3)

-- The expression to evaluate:
noncomputable def given_expression (θ : ℝ) : ℝ :=
  (sin (3 * π / 2 + θ) + 2 * cos (π - θ)) / (sin (π / 2 - θ) - sin (π - θ))

-- The target result:
theorem evaluate_expression (θ : ℝ) (h : θ_ang θ) : given_expression θ = 3 / 2 :=
sorry

end evaluate_expression_l335_335158


namespace binom_8_5_eq_56_l335_335835

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l335_335835


namespace ornamental_rings_remaining_l335_335457

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end ornamental_rings_remaining_l335_335457


namespace jungkook_has_smallest_collection_l335_335742

-- Define the collections
def yoongi_collection : ℕ := 7
def jungkook_collection : ℕ := 6
def yuna_collection : ℕ := 9

-- State the theorem
theorem jungkook_has_smallest_collection : 
  jungkook_collection = min yoongi_collection (min jungkook_collection yuna_collection) := 
by
  sorry

end jungkook_has_smallest_collection_l335_335742


namespace silk_dyeing_total_correct_l335_335675

open Real

theorem silk_dyeing_total_correct :
  let green := 61921
  let pink := 49500
  let blue := 75678
  let yellow := 34874.5
  let total_without_red := green + pink + blue + yellow
  let red := 0.10 * total_without_red
  let total_with_red := total_without_red + red
  total_with_red = 245270.85 :=
by
  sorry

end silk_dyeing_total_correct_l335_335675


namespace candy_store_spending_l335_335541

def initialAllowance : ℝ := 2.25
def arcadeSpending : ℝ := (3/5) * initialAllowance
def remainingAfterArcade : ℝ := initialAllowance - arcadeSpending
def toyStoreSpending : ℝ := (1/3) * remainingAfterArcade
def remainingAfterToyStore : ℝ := remainingAfterArcade - toyStoreSpending

theorem candy_store_spending : remainingAfterToyStore = 0.60 :=
by
  sorry

end candy_store_spending_l335_335541


namespace find_C_and_perimeter_l335_335992

noncomputable def triangle_properties (a b c A B C : ℝ) : Prop :=
  ∃ (a b c A B C : ℝ), 
  ∃ (triangle : ∃ (a b c A B C : ℝ), 
  ∃ ∠A ∠B ∠C : ℝ, 
    a = c * sin B / sin C ∧ b = c * sin A / sin C ∧ 
    2 * cos C * (a * cos B + b * cos A) = c) ∧ 
    (0 < A ∧ A < π) ∧ (0 < B ∧ B < π) ∧ (0 < C ∧ C < π) 

theorem find_C_and_perimeter : 
  ∀ (a b c A B C : ℝ),
  (triangle_properties a b c A B C) → 
  2 * cos C * (a * cos B + b * cos A) = c → 
  C = π / 3 ∧ c = sqrt 7 ∧ (1 / 2) * a * b * sin C = (3 * sqrt 3) / 2 → 
  a + b + c = 5 + sqrt 7 :=
by
  sorry

end find_C_and_perimeter_l335_335992


namespace first_figure_angle_x_l335_335584

theorem first_figure_angle_x 
  (AB DE : Line) (AB_parallel_DE : Parallel AB DE)
  (CDF DF : Triangle)
  (angle_CDF_55 : angle CDF = 55)
  (angle_CFD_25 : angle CFD = 25) :
  angle_x = 80 :=
by sorry

end first_figure_angle_x_l335_335584


namespace simplified_expression_is_correct_l335_335657

noncomputable def simplify_expression : ℂ :=
  (complex.mk (-2) (real.sqrt 7) / 3)^4 + (complex.mk (-2) (-real.sqrt 7) / 3)^4

theorem simplified_expression_is_correct : simplify_expression = 242 / 81 :=
  sorry

end simplified_expression_is_correct_l335_335657


namespace digit_sequence_1998_2000_l335_335020

theorem digit_sequence_1998_2000 : 
    ∃ (d1998 d1999 d2000 : ℕ), 
    (d1998 = 1) ∧ (d1999 = 4) ∧ (d2000 = 1) ∧ 
    (d1998 * 100 + d1999 * 10 + d2000 = 141) ∧
    (Σ (n ∈ {1}), 1 + 
     Σ (n ∈ (10..19)), 2 + 
     Σ (n ∈ (100..199)), 3 + 
     Σ (n ∈ (1000..(1000 + 419))), 4) ≥ 1998 :=
begin
  existsi 1,
  existsi 4,
  existsi 1,
  split, 
  rfl,
  split,
  rfl,
  split,
  rfl,
  split,
  exact rfl,
  sorry
end

end digit_sequence_1998_2000_l335_335020


namespace determinant_is_zero_l335_335065

open Matrix

noncomputable def given_matrix (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, Real.cos α, -Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_is_zero (α β : ℝ) : det (given_matrix α β) = 0 := by
  sorry

end determinant_is_zero_l335_335065


namespace largest_prime_factor_l335_335734

-- Define the expressions given in the problem
def expression := 16^4 + 2 * 16^2 + 1 - 15^4

-- State the problem of finding the largest prime factor
theorem largest_prime_factor : ∃ p : ℕ, nat.prime p ∧ p ∣ expression ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ expression → q ≤ 241 :=
by {
  sorry  -- Proof needed
}

end largest_prime_factor_l335_335734


namespace mod_inverse_7_29_l335_335470

theorem mod_inverse_7_29 : ∃ (a : ℕ), a = 25 ∧ (7 * a) % 29 = 1 := by
  use 25
  split
  sorry -- proof that 25 is the correct a
  sorry -- proof that (7 * 25) % 29 = 1

end mod_inverse_7_29_l335_335470


namespace maximum_value_of_f_l335_335145

noncomputable def f : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := f (n / 2) + n % 2

theorem maximum_value_of_f :
  ∃ n, 0 ≤ n ∧ n ≤ 1991 ∧ 11 * 14 * n ≥ 0 ∧ ∀ m, 0 ≤ m ∧ m ≤ 1991 → f m ≤ 10 :=
sorry

end maximum_value_of_f_l335_335145


namespace non_friendly_even_l335_335791

-- Define a polygon with 2k vertices where every two adjacent sides are perpendicular
structure Polygon :=
  (vertices : List (ℝ × ℝ))
  (adj_perpendicular : ∀ (v1 v2 : (ℝ × ℝ)), v1 ∈ vertices → v2 ∈ vertices → adjacent v1 v2 → perpendicular v1 v2)

-- Define adjacency of vertices in the polygon
def adjacent (v1 v2 : (ℝ × ℝ)) : Prop := 
  -- conditions defining adjacency of v1 and v2 in the polygon
  sorry

-- Define perpendicularity of angle bisectors at two vertices
def perpendicular (v1 v2 : (ℝ × ℝ)) : Prop :=
  -- conditions defining perpendicularity of angle bisectors at v1 and v2
  sorry

-- State the main theorem
theorem non_friendly_even (P : Polygon) (v : (ℝ × ℝ)) (H_v : v ∈ P.vertices) :
  ∃ n : ℕ, even n ∧ (∀ (u : (ℝ × ℝ)), u ∈ P.vertices → perpendicular v u → count (perpendicular v) P.vertices n) :=
sorry

end non_friendly_even_l335_335791


namespace clothing_percentage_l335_335633

variable (T : ℝ) -- Total amount excluding taxes.
variable (C : ℝ) -- Percentage of total amount spent on clothing.

-- Conditions
def spent_on_food := 0.2 * T
def spent_on_other_items := 0.3 * T

-- Taxes
def tax_on_clothing := 0.04 * (C * T)
def tax_on_food := 0.0
def tax_on_other_items := 0.08 * (0.3 * T)
def total_tax_paid := 0.044 * T

-- Statement to prove
theorem clothing_percentage : 
  0.04 * (C * T) + 0.08 * (0.3 * T) = 0.044 * T ↔ C = 0.5 :=
by
  sorry

end clothing_percentage_l335_335633


namespace num_of_nickels_l335_335197

theorem num_of_nickels (x : ℕ) (hx_eq_dimes : ∀ n, n = x → n = x) (hx_eq_quarters : ∀ n, n = x → n = 2 * x) (total_value : 5 * x + 10 * x + 50 * x = 1950) : x = 30 :=
sorry

end num_of_nickels_l335_335197


namespace count_integer_pairs_eq_eight_l335_335175

-- Define the problem condition
def equation (m n : ℤ) : Prop := m * n + n + 14 = (m - 1) ^ 2

-- Define the theorem we want to prove
theorem count_integer_pairs_eq_eight :
  (finset.filter (λ (p : ℤ × ℤ), equation p.1 p.2) (finset.univ.product finset.univ)).card = 8 := by
  sorry

end count_integer_pairs_eq_eight_l335_335175


namespace trajectory_eq_l335_335929

theorem trajectory_eq (x y : ℝ) (θ : ℝ) (P : ℝ × ℝ) (h1 : P = (x, y)) 
  (h2 : x * cos θ + y * sin θ = 1) (h3 : P.perpendicular_to_origin) : 
  x^2 + y^2 = 1 := 
by sorry

end trajectory_eq_l335_335929


namespace find_B_l335_335696

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem find_B (B : ℕ) : 
  let A := 24 in
  HCF A B = 13 ∧ LCM A B = 312 → B = 169 := 
by
  intro h
  let A := 24
  sorry

end find_B_l335_335696


namespace range_of_a_minus_b_l335_335513

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) : -3 < a - b ∧ a - b < 0 :=
by
  sorry

end range_of_a_minus_b_l335_335513


namespace course_duration_l335_335853

theorem course_duration (total_hours : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) (duration: ℕ) :
  total_hours = 336 →
  class_hours_per_week = 2 * 3 + 1 * 4 →
  homework_hours_per_week = 4 →
  duration = total_hours / (class_hours_per_week + homework_hours_per_week) →
  duration = 24 :=
begin
  intros h_total h_class h_homework h_duration,
  rw [h_total, h_class, h_homework] at h_duration,
  exact h_duration,
end

end course_duration_l335_335853


namespace count_improper_fractions_15_l335_335716

def isImproperFraction (num denom : ℕ) : Prop :=
  num >= denom

def possibleNumerators (denom : ℕ) (S : Finset ℕ) : Finset ℕ :=
  S.filter (fun x => isImproperFraction x denom)

def countImproperFractions (S : Finset ℕ) : ℕ :=
  S.sum (fun denom => (possibleNumerators denom S).card)

theorem count_improper_fractions_15 :
  let S : Finset ℕ := {3, 5, 7, 11, 13, 17}
  countImproperFractions S = 15 :=
by
  let S : Finset ℕ := {3, 5, 7, 11, 13, 17}
  have hs : ∀ denom ∈ S, (possibleNumerators denom S).card = 
    if denom = 3 then 0 else if denom = 5 then 5 else if denom = 7 then 4 else if denom = 11 then 3 else if denom = 13 then 2 else 1 := sorry
  have sum_correct : _ = 15 := sorry
  rw [countImproperFractions, finset_sum_eq_sum_filter],
  exact sum_correct

end count_improper_fractions_15_l335_335716


namespace mean_of_y_and_18_is_neg1_l335_335306

theorem mean_of_y_and_18_is_neg1 (y : ℤ) : 
  ((4 + 6 + 10 + 14) / 4) = ((y + 18) / 2) → y = -1 := 
by 
  -- Placeholder for the proof
  sorry

end mean_of_y_and_18_is_neg1_l335_335306


namespace range_of_a_decreasing_l335_335558

noncomputable def f (x a : ℝ) := logBase 3 (x^2 + a * x + a + 5)
def isDecreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem range_of_a_decreasing :
  (∃ a : ℝ, ∀ x : ℝ, x < 1 → ∃ y1 y2, isDecreasing (f x a) (Set.Iio y1))
  ↔ -3 ≤ a ∧ a ≤ -2 :=
sorry

end range_of_a_decreasing_l335_335558


namespace find_n_gizmos_l335_335352

-- Definitions based on conditions
def workers_per_gadgets (hours : ℕ) (workers : ℕ) (gadgets : ℕ) : ℝ :=
  gadgets / (workers * hours)

def workers_per_gizmos (hours : ℕ) (workers : ℕ) (gizmos : ℕ) : ℝ :=
  gizmos / (workers * hours)

-- The main theorem to be proven
theorem find_n_gizmos :
  let gadgets_per_hour := workers_per_gadgets 1 80 240 in
  let gizmos_per_hour := workers_per_gizmos 1 80 160 in
  let gadgets_per_3h := workers_per_gadgets 3 40 180 in
  let gizmos_per_3h := workers_per_gizmos 3 40 270 in
  let gadgets_per_2h := workers_per_gadgets 2 100 500 in
  let n := 100 * gizmos_per_3h * 2 in
  gadgets_per_hour = 3 → gizmos_per_hour = 2 → gadgets_per_3h = 3 → gizmos_per_3h = 2.25 → gadgets_per_2h = 3 → n = 450 :=
sorry

end find_n_gizmos_l335_335352


namespace perfect_apples_count_l335_335972

-- Definitions (conditions)
def total_apples := 30
def too_small_fraction := (1 : ℚ) / 6
def not_ripe_fraction := (1 : ℚ) / 3
def too_small_apples := (too_small_fraction * total_apples : ℚ)
def not_ripe_apples := (not_ripe_fraction * total_apples : ℚ)

-- Statement of the theorem (proof problem)
theorem perfect_apples_count : total_apples - too_small_apples - not_ripe_apples = 15 := by
  sorry

end perfect_apples_count_l335_335972


namespace tour_guide_and_spouse_handshakes_equal_l335_335634

   -- Defining the problem conditions
   def handshakes (n : ℕ) : Prop :=
     let participants := 2 * n in
     ∃ (handshake_counts : Finₓ participants → Finₓ (participants - 1)),
     (∀ i j : Finₓ participants, i ≠ j → handshake_counts i ≠ handshake_counts j) ∧
     (∀ k : Finₓ n, handshake_counts (Finₓ.ofNat (2 * k)) + handshake_counts (Finₓ.ofNat (2 * k + 1)) = (n - 1))

   -- Stating that the tour guide and their spouse shake hands with the same number of people
   theorem tour_guide_and_spouse_handshakes_equal (n : ℕ) : handshakes n :=
   sorry
   
end tour_guide_and_spouse_handshakes_equal_l335_335634


namespace find_percentage_of_other_investment_l335_335661

theorem find_percentage_of_other_investment
  (total_investment : ℝ) (specific_investment : ℝ) (specific_rate : ℝ) (total_interest : ℝ) 
  (other_investment : ℝ) (other_interest : ℝ) (P : ℝ) :
  total_investment = 17000 ∧
  specific_investment = 12000 ∧
  specific_rate = 0.04 ∧
  total_interest = 1380 ∧
  other_investment = total_investment - specific_investment ∧
  other_interest = total_interest - specific_rate * specific_investment ∧ 
  other_interest = (P / 100) * other_investment
  → P = 18 :=
by
  intros
  sorry

end find_percentage_of_other_investment_l335_335661


namespace combination_8_5_eq_56_l335_335839

theorem combination_8_5_eq_56 : nat.choose 8 5 = 56 :=
by
  sorry

end combination_8_5_eq_56_l335_335839


namespace relationship_between_A_and_p_l335_335726

variable {x y p : ℝ}

theorem relationship_between_A_and_p (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : x ≠ y * 2) (h4 : x ≠ p * y)
  (A : ℝ) (hA : A = (x^2 - 3 * y^2) / (3 * x^2 + y^2))
  (hEq : (p * x * y) / (x^2 - (2 + p) * x * y + 2 * p * y^2) - y / (x - 2 * y) = 1 / 2) :
  A = (9 * p^2 - 3) / (27 * p^2 + 1) := 
sorry

end relationship_between_A_and_p_l335_335726


namespace sqrt_square_eq_abs_l335_335922

theorem sqrt_square_eq_abs (x : ℝ) : sqrt (x ^ 2) = |x| :=
sorry

end sqrt_square_eq_abs_l335_335922


namespace football_area_l335_335276

theorem football_area (PQRS_is_square : ∀ (P Q R S : ℝ×ℝ), square P Q R S) 
    (circle_Q : ∀ (Q P T U : ℝ×ℝ), circle_centered_at Q P T U)
    (circle_S : ∀ (S P V W : ℝ×ℝ), circle_centered_at S P V W)
    (PQ_eq_4 : ∀ (P Q : ℝ×ℝ), dist P Q = 4) :
    let area_II_III := 8 * real.pi - 16
    in area_II_III = 8 * real.pi - 16 :=
by
  sorry

end football_area_l335_335276


namespace find_m_l335_335436

namespace MathProof

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

-- State the problem
theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -15 := by
  sorry

end MathProof

end find_m_l335_335436


namespace solve_car_production_l335_335378

def car_production_problem : Prop :=
  ∃ (NorthAmericaCars : ℕ) (TotalCars : ℕ) (EuropeCars : ℕ),
    NorthAmericaCars = 3884 ∧
    TotalCars = 6755 ∧
    EuropeCars = TotalCars - NorthAmericaCars ∧
    EuropeCars = 2871

theorem solve_car_production : car_production_problem := by
  sorry

end solve_car_production_l335_335378


namespace charles_total_money_l335_335830

variable (found_pennies : Nat)
variable (found_nickels : Nat)
variable (found_dimes : Nat)
variable (home_nickels : Nat)
variable (home_dimes : Nat)
variable (home_quarter : Nat)

theorem charles_total_money :
  (6 * 0.01 + 4 * 0.05 + 3 * 0.10) + (3 * 0.05 + 2 * 0.10 + 1 * 0.25) = 1.16 := by
  have p_found := (6 : ℝ) * 0.01
  have n_found := (4 : ℝ) * 0.05
  have d_found := (3 : ℝ) * 0.10
  have n_home := (3 : ℝ) * 0.05
  have d_home := (2 : ℝ) * 0.10
  have q_home := (1 : ℝ) * 0.25
  let total_found := p_found + n_found + d_found
  let total_home := n_home + d_home + q_home
  let total_money := total_found + total_home
  show total_money = 1.16
  sorry

end charles_total_money_l335_335830


namespace total_people_who_eat_vegetarian_l335_335363

def people_who_eat_only_vegetarian := 16
def people_who_eat_both_vegetarian_and_non_vegetarian := 12

-- We want to prove that the total number of people who eat vegetarian is 28
theorem total_people_who_eat_vegetarian : 
  people_who_eat_only_vegetarian + people_who_eat_both_vegetarian_and_non_vegetarian = 28 :=
by 
  sorry

end total_people_who_eat_vegetarian_l335_335363


namespace inequality_arith_geo_mean_l335_335141

theorem inequality_arith_geo_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
by
  sorry

end inequality_arith_geo_mean_l335_335141


namespace binom_8_5_eq_56_l335_335834

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l335_335834


namespace total_cost_7_sandwiches_6_sodas_l335_335423

-- Definitions based on conditions
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def bulk_discount : ℕ := 10
def total_items (sandwiches sodas : ℕ) : ℕ := sandwiches + sodas

-- The main problem: Prove the total cost given the conditions
theorem total_cost_7_sandwiches_6_sodas : 
  let total_cost := (7 * sandwich_cost) + (6 * soda_cost)
  in if total_items 7 6 > 10 then total_cost - bulk_discount else total_cost = 36 :=
by
  sorry

end total_cost_7_sandwiches_6_sodas_l335_335423


namespace train_times_valid_l335_335339

-- Define the parameters and conditions
def trainA_usual_time : ℝ := 180 -- minutes
def trainB_travel_time : ℝ := 810 -- minutes

theorem train_times_valid (t : ℝ) (T_B : ℝ) 
  (cond1 : (7 / 6) * t = t + 30)
  (cond2 : T_B = 4.5 * t) : 
  t = trainA_usual_time ∧ T_B = trainB_travel_time :=
by
  sorry

end train_times_valid_l335_335339


namespace car_travel_time_l335_335631

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l335_335631


namespace magician_guarantee_three_of_clubs_l335_335782

-- Definitions corresponding to the identified conditions
def deck_size : ℕ := 52
def num_discarded : ℕ := 51
def magician_choice (n : ℕ) (from_left : bool) : ℕ := if from_left then n else deck_size + 1 - n
def is_edge_position (position : ℕ) : Prop := position = 1 ∨ position = deck_size

-- Statement of the problem, translated to Lean
theorem magician_guarantee_three_of_clubs (initial_pos : ℕ) (H : is_edge_position initial_pos) :
    ∃ strategy, ∀ spectator_choice, 
                (∃ remaining_cards, 
                  remaining_cards = deck_size - num_discarded + 1 ∧ 
                  three_of_clubs ∈ remaining_cards) :=
begin
  -- proving strategy exists if the initial position is at the edge
  sorry
end

end magician_guarantee_three_of_clubs_l335_335782


namespace geometric_mean_property_l335_335706

theorem geometric_mean_property
  (parabola : Type)
  (C F : parabola)
  (circle : Type)
  (A B E D : circle)
  (vertex_def : is_vertex C)
  (focus_def : is_focus F)
  (circle_centered : is_centered_circle C F)
  (intersection_A : is_intersection A)
  (intersection_B : is_intersection B)
  (intersection_E : is_intersection_line_AB CF E)
  (diametric_opposite : is_diametric_opposite D F) :
  FD * FE = DE ^ 2 :=
sorry

end geometric_mean_property_l335_335706


namespace sum_of_solutions_eq_neg_two_l335_335616

noncomputable def f : ℝ → ℝ :=
λ x, if x < -3 then 3 * x + 6 else -x^2 - 2 * x + 2

theorem sum_of_solutions_eq_neg_two :
  (∑ x in ({x : ℝ | f x = -6}) (λ x, x)) = -2 := 
begin
  sorry
end

end sum_of_solutions_eq_neg_two_l335_335616


namespace ornamental_rings_remaining_l335_335456

-- Definitions based on conditions
variable (initial_stock : ℕ) (final_stock : ℕ)

-- Condition 1
def condition1 := initial_stock + 200 = 3 * initial_stock

-- Condition 2
def condition2 := final_stock = (200 + initial_stock) * 1 / 4 - (200 + initial_stock) / 4 + 300 - 150

-- Theorem statement to prove the final stock is 225
theorem ornamental_rings_remaining
  (h1 : condition1 initial_stock)
  (h2 : condition2 initial_stock final_stock) :
  final_stock = 225 :=
sorry

end ornamental_rings_remaining_l335_335456


namespace digit_difference_l335_335747

theorem digit_difference {X Y : ℕ} (h : 10 * X + Y - (10 * Y + X) = 72) : X - Y = 8 :=
by
  have h1 : 9 * (X - Y) = 72 := by linarith
  have h2 : X - Y = 72 / 9 := by linarith
  exact h2

end digit_difference_l335_335747


namespace possible_values_of_n_l335_335069

theorem possible_values_of_n (n : ℕ) (h₁ : ∃ (S : set (line ℝ²)), S.card = n ∧ ∀ d ∈ S, (S \ {d}).intersect_count d = 10) :
  n = 11 ∨ n = 12 ∨ n = 15 ∨ n = 20 :=
sorry

end possible_values_of_n_l335_335069


namespace team_answer_prob_team_expected_score_l335_335802

theorem team_answer_prob (P1 P2 P3 : ℝ) (h1 : P1 = 3 / 4) (h2 : (1 - P1) * (1 - P3) = 1 / 12) (h3 : P2 * P3 = 1 / 4) :
  (1 - (1 - P1) * (1 - P2) * (1 - P3)) = 91 / 96 :=
by
  sorry

theorem team_expected_score (P1 P2 P3 : ℝ) (h1 : P1 = 3 / 4) (h2 : (1 - P1) * (1 - P3) = 1 / 12) (h3 : P2 * P3 = 1 / 4) :
  30 * (10 * (91 / 96)) - 100 = 1475 / 8 :=
by
  sorry

end team_answer_prob_team_expected_score_l335_335802


namespace initial_books_l335_335602

theorem initial_books (sold_monday sold_tuesday sold_wednesday sold_thursday sold_friday : ℕ)
  (percentage_not_sold : ℝ)
  (h_monday : sold_monday = 75)
  (h_tuesday : sold_tuesday = 50)
  (h_wednesday : sold_wednesday = 64)
  (h_thursday : sold_thursday = 78)
  (h_friday : sold_friday = 135)
  (h_percentage_not_sold : percentage_not_sold = 55.333333333333336) :
  let total_sold := (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday) in
  let percentage_sold := 100 - percentage_not_sold in
  let x_initial_books := total_sold / (percentage_sold / 100) in
  x_initial_books = 900 := 
sorry

end initial_books_l335_335602


namespace sum_of_transformed_sequence_range_of_d_l335_335110

-- Statement for Part 1
theorem sum_of_transformed_sequence (d : ℝ) (n : ℕ)
    (h1 : d > 0)
    (h2 : (2 + d)^2 = 4 + 6 * d) :
      (Finset.range n).sum (fun k => (2 * k - 1) / 2^k) = 3 - (2 * n + 3) / 2^n := sorry

-- Statement for Part 2
theorem range_of_d (d : ℝ)
    (h1 : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, 1 / ((2*i - 1) * (2*(i+1) - 1))) > 2015 / 2016) :
      0 < d ∧ d < 1 / 2015 := sorry

end sum_of_transformed_sequence_range_of_d_l335_335110


namespace chocolate_eggs_total_weight_l335_335624

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end chocolate_eggs_total_weight_l335_335624


namespace Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l335_335651

-- Problem 1: Factorization
theorem Problem1_factorize (a : ℝ) : a^2 - 8 * a + 15 = (a - 3) * (a - 5) :=
  sorry

-- Problem 2: Minimum Perimeter of triangle ABC
theorem Problem2_min_perimeter_triangle (a b c : ℝ) 
  (h : a^2 + b^2 - 14 * a - 8 * b + 65 = 0) (hc : ∃ k : ℤ, 2 * k + 1 = c) : 
  a + b + c ≥ 16 :=
  sorry

-- Problem 3: Maximum Value of the Polynomial
theorem Problem3_max_value_polynomial : 
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, y ≠ -1 → -2 * x^2 - 4 * x + 3 ≥ -2 * y^2 - 4 * y + 3 :=
  sorry

end Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l335_335651


namespace g_difference_l335_335612

def g (n : ℕ) : ℚ :=
  (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

theorem g_difference (m : ℕ) : 
  g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) :=
by
  sorry

end g_difference_l335_335612


namespace find_A_find_bc_sum_l335_335140

-- Definitions of the angles and sides of the triangle
variables (A B C : ℝ) (a b c : ℝ)

-- Vectors definitions
def m : ℝ × ℝ := (-Math.cos (A / 2), Math.sin (A / 2))
def n : ℝ × ℝ := (Math.cos (A / 2), Math.sin (A / 2))

-- Problem statement
theorem find_A (h1 : m A B C a b c · n A B C a b c = 1/2) : A = 2 * Real.pi / 3 :=
sorry

theorem find_bc_sum (h2 : a = 2 * Real.sqrt 3) (h3 : Real.sqrt 3 = (1 / 2) * b * c * Math.sin A) : b + c = 4 :=
sorry

end find_A_find_bc_sum_l335_335140


namespace avg_annual_growth_rate_optimal_room_price_l335_335018

-- Problem 1: Average Annual Growth Rate
theorem avg_annual_growth_rate (visitors_2021 visitors_2023 : ℝ) (years : ℕ) (visitors_2021_pos : 0 < visitors_2021) :
  visitors_2023 > visitors_2021 → visitors_2023 / visitors_2021 = 2.25 → 
  ∃ x : ℝ, (1 + x)^2 = 2.25 ∧ x = 0.5 :=
by sorry

-- Problem 2: Optimal Room Price for Desired Profit
theorem optimal_room_price (rooms : ℕ) (base_price cost_per_room desired_profit : ℝ)
  (rooms_pos : 0 < rooms) :
  base_price = 180 → cost_per_room = 20 → desired_profit = 9450 → 
  ∃ y : ℝ, (y - cost_per_room) * (rooms - (y - base_price) / 10) = desired_profit ∧ y = 230 :=
by sorry

end avg_annual_growth_rate_optimal_room_price_l335_335018


namespace arthur_bought_2_hamburgers_on_second_day_l335_335419

theorem arthur_bought_2_hamburgers_on_second_day
  (H D X: ℕ)
  (h1: 3 * H + 4 * D = 10)
  (h2: D = 1)
  (h3: 2 * X + 3 * D = 7):
  X = 2 :=
by
  sorry

end arthur_bought_2_hamburgers_on_second_day_l335_335419


namespace sum_of_areas_eq_100m2_impossible_l335_335829

theorem sum_of_areas_eq_100m2_impossible (n : ℕ) 
  (h_side_lengths : ∀ k, 1 ≤ k ∧ k ≤ n → (2 * k - 1 ≥ 0)) :
  let sum_of_areas := ∑ k in Finset.range n, (2 * (k + 1) - 1)^2
  in sum_of_areas ≠ 10^4 :=
by sorry

end sum_of_areas_eq_100m2_impossible_l335_335829


namespace intersection_M_N_l335_335134

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335134


namespace smallest_n_points_circle_l335_335877

theorem smallest_n_points_circle (n : ℕ) (h : n = 91) : 
  ∃ (pairs : ℕ), 
    pairs ≥ 2005 ∧ 
    ∀ (P : Finset (ℝ × ℝ)), 
      P.card = n → 
      ∃ (A B : Finset (ℝ × ℝ)), 
        A ≠ B ∧ 
        (angle A.val B.val) ≤ 120 :=
  sorry

end smallest_n_points_circle_l335_335877


namespace swimmers_meetings_l335_335338

/-- Two swimmers at opposite ends of a 90-foot pool swim at rates of 
3 feet per second and 2 feet per second, respectively. 
They swim back and forth for 12 minutes without loss of time at the turns. 
Prove that the number of times they pass each other is 20. -/
theorem swimmers_meetings (length_pool : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) 
  (time_minutes : ℕ) (length_pool_eq : length_pool = 90) (rate_swimmer1_eq : rate_swimmer1 = 3) 
  (rate_swimmer2_eq : rate_swimmer2 = 2) (time_minutes_eq : time_minutes = 12) : 
  let time_seconds := time_minutes * 60 in 
  let distance_covered swimmer_rate := time_seconds * swimmer_rate in
  let times_pass_each_other := 
    (distance_covered rate_swimmer1 / length_pool) + (distance_covered rate_swimmer2 / length_pool) - 1 in
  times_pass_each_other = 20 :=
by
  /- Proof is not required -/
  sorry

end swimmers_meetings_l335_335338


namespace combination_8_choose_2_l335_335027

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end combination_8_choose_2_l335_335027


namespace alpha_necessary_not_sufficient_for_beta_l335_335552

def alpha (x : ℝ) : Prop := x^2 = 4
def beta (x : ℝ) : Prop := x = 2

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x : ℝ, beta x → alpha x) ∧ ¬(∀ x : ℝ, alpha x → beta x) :=
by
  sorry

end alpha_necessary_not_sufficient_for_beta_l335_335552


namespace double_root_possible_values_l335_335398

theorem double_root_possible_values (s : ℤ) 
  (h1 : ∃ b₃ b₂ b₁ : ℤ, (λ x, x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 50) = 0)
  (h2 : ∀ (P : ℤ → ℤ), (P = λ x, x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 50) → (x - s)^2 ∣ P x): 
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 := 
sorry

end double_root_possible_values_l335_335398


namespace first_set_broken_percent_l335_335092

-- Defining some constants
def firstSetTotal : ℕ := 50
def secondSetTotal : ℕ := 60
def secondSetBrokenPercent : ℕ := 20
def totalBrokenMarbles : ℕ := 17

-- Define the function that calculates broken marbles from percentage
def brokenMarbles (percent marbles : ℕ) : ℕ := (percent * marbles) / 100

-- Theorem statement
theorem first_set_broken_percent :
  ∃ (x : ℕ), brokenMarbles x firstSetTotal + brokenMarbles secondSetBrokenPercent secondSetTotal = totalBrokenMarbles ∧ x = 10 :=
by
  sorry

end first_set_broken_percent_l335_335092


namespace find_length_DC_l335_335991

namespace MathProof

-- Definition of the variables and conditions
variables {A B C D F E : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited F] [Inhabited E]

-- Given the following conditions for the trapezoid ABCD
def trapezoid_ABCD_parallel_DC (AB DC : ℝ) (par : AB = DC) : Prop := par

def sides_AB_BC (AB BC : ℝ) : Prop := AB = 5 ∧ BC = 4 * real.sqrt 2

def angles (angle_BCD angle_CDA : ℝ) : Prop := angle_BCD = 45 ∧ angle_CDA = 45

-- Required to prove the length of DC
theorem find_length_DC (angle_BCD angle_CDA : ℝ) (BC : ℝ) (AB DC : ℝ) 
  (H1 : trapezoid_ABCD_parallel_DC AB DC AB)
  (H2 : sides_AB_BC AB BC)
  (H3 : angles angle_BCD angle_CDA) : 
  DC = 13 :=
begin
  sorry
end

end MathProof

end find_length_DC_l335_335991


namespace proof_l335_335215

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (S : ℝ)

-- Given conditions
def condition_1 : a = 2 := sorry
def condition_2 : C = real.pi / 3 := sorry

-- Proof problem (1)
def proof_problem_1 : Prop :=
  (A = real.pi / 4) → (c = real.sqrt 6)

-- Proof problem (2)
def proof_problem_2 : Prop :=
  (S = real.sqrt 3) → (b = 2) ∧ (c = 2)

theorem proof :
  (condition_1 ∧ condition_2) →
  (proof_problem_1 ∧ proof_problem_2)
:= sorry

end proof_l335_335215


namespace hockey_league_games_l335_335749

def num_teams : ℕ := 18
def encounters_per_pair : ℕ := 10
def num_games (n : ℕ) (k : ℕ) : ℕ := (n * (n - 1)) / 2 * k

theorem hockey_league_games :
  num_games num_teams encounters_per_pair = 1530 :=
by
  sorry

end hockey_league_games_l335_335749


namespace quadratic_function_properties_l335_335164

def quadratic_function (a b : ℝ) (x : ℝ) := a * x^2 + b * x + a

noncomputable def f (x : ℝ) : ℝ := quadratic_function (-2) 7 x

theorem quadratic_function_properties :
  (∀ x : ℝ, quadratic_function a b (x + 7/4) = quadratic_function a b (7/4 - x)) →
  (∀ x : ℝ, quadratic_function a b x = 7 * x + a → ∃! x, quadratic_function a b x = 7*x + a) →
  a = -2 ∧ b = 7 ∧ (range (f) = set.Iic (33/8)) :=
sorry

end quadratic_function_properties_l335_335164


namespace generating_function_correct_l335_335299

noncomputable def generating_function (x : ℝ) : ℝ := (1 + x) ^ -2

theorem generating_function_correct :
  ∀ (n : ℕ), (n > 0) → 
  seq.nth (λ k, (-1) ^ k * (k + 1)) n =
  (generating_function^[n]) 0 :=
sorry

end generating_function_correct_l335_335299


namespace KLMN_is_parallelogram_l335_335010

noncomputable def is_parallelogram {K L M N : ℝ} : Prop :=
  parallel K L M N ∧ parallel L M N K ∧ length K L = length M N ∧ length L M = length N K

theorem KLMN_is_parallelogram
  (S S1 S2 : Circle ℝ)
  (A B C D K N L M : Point ℝ)
  (r R : ℝ)
  (h1 : Circle.inscribed S ABCD)
  (h2 : Circle.tangent_at S1 S A ∧ Circle.tangent_at S2 S C ∧ S1.radius = S2.radius ∧ S1.radius = r)
  (h3 : Circle.intersects_at S1 AB K ∧ Circle.intersects_at S1 AD N)
  (h4 : Circle.intersects_at S2 BC L ∧ Circle.intersects_at S2 CD M) :
  is_parallelogram K L M N :=
sorry

end KLMN_is_parallelogram_l335_335010


namespace max_of_3x_plus_y_l335_335293

theorem max_of_3x_plus_y (x y : ℝ) (h : x^2 + y^2 / 3 = 1) : 
  3 * x + y ≤ 2 * Real.sqrt 3 :=
begin
  sorry
end

end max_of_3x_plus_y_l335_335293


namespace greatest_difference_hundreds_digit_l335_335678

theorem greatest_difference_hundreds_digit : 
  ∀ (y : ℕ), (∃ (y : ℕ), 0 ≤ y ∧ y ≤ 9 ∧ (632 + 100 * y) % 4 = 0) → 
  (y <= 9 ∧ ∃ y_max y_min, 0 ≤ y_min ∧ y_min ≤ 9 ∧ 0 ≤ y_max ∧ y_max ≤ 9 ∧ (632 + 100 * y_min) % 4 = 0 ∧ (632 + 100 * y_max) % 4 = 0 ∧ y_max - y_min = 9) :=
begin
  sorry
end

end greatest_difference_hundreds_digit_l335_335678


namespace email_scam_check_l335_335356

-- Define the condition for receiving an email about winning a car
def received_email (info: String) : Prop :=
  info = "You received an email informing you that you have won a car. You are asked to provide your mobile phone number for contact and to transfer 150 rubles to a bank card to cover the postage fee for sending the invitation letter."

-- Define what indicates a scam
def is_scam (info: String) : Prop :=
  info = "Request for mobile number already known to the sender and an upfront payment."

-- Proving that the information in the email implies it is a scam
theorem email_scam_check (info: String) (h1: received_email info) : is_scam info :=
by
  sorry

end email_scam_check_l335_335356


namespace exponentiation_identity_l335_335722

theorem exponentiation_identity :
  (3 ^ 12 * (9 : ℤ) ^ (-3 : ℤ)) ^ 2 = 531441 := 
by
  -- Definitions and properties are implicitly included through Lean's library imports
  sorry

end exponentiation_identity_l335_335722


namespace albert_lisa_combinations_l335_335406

def valid_combinations (albert: Finset ℕ) (lisa: Finset ℕ) :=
  (count: ℕ) (combinations: Finset (ℕ × ℕ × ℕ × ℕ)) :
    -- This represents the number of valid combinations
      (∀ x ∈ lisa, ∀ a b c ∈ albert, a * b * c = 2 * x) 
      ∧
      -- And the set of valid (albert1, albert2, albert3, lisa)
      ((a × b × c × l).fst.snd.snd.snd ∈ combinations → a * b * c = 2 * l
      ∧ 
      -- Ensure unique combinations (order matters).
      MultiSet.card combinations.to_List = count)

theorem albert_lisa_combinations : valid_combinations (Finset.range 1 11) (Finset.range 1 9) 42 :=
sorry

end albert_lisa_combinations_l335_335406


namespace true_statements_for_family_of_lines_l335_335247

-- Definitions of the problem conditions
def family_of_lines (θ : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1 * Real.cos θ + (p.2 - 2) * Real.sin θ = 1 }

def lies_on_any_line (P : ℝ × ℝ) : Prop :=
  ∃ θ ∈ Set.Icc 0 (2 * Real.pi), P ∈ family_of_lines θ

def regular_n_sided_polygon (n : ℕ) (n_ge_3 : n ≥ 3) : Prop :=
  ∃ (vertices : Fin n → (ℝ × ℝ)), ∀ i, ∃ θ ∈ Set.Icc 0 (2 * Real.pi), (vertices i) ∈ family_of_lines θ

def equal_area_equilateral_triangles : Prop :=
  ∀ Δ₁ Δ₂ : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ),
  (∀ v : (ℝ × ℝ), v ∈ [Δ₁.1, Δ₁.2.1, Δ₁.2.2] → ∃ θ ∈ Set.Icc 0 (2 * Real.pi), v ∈ family_of_lines θ) →
  (∀ v : (ℝ × ℝ), v ∈ [Δ₂.1, Δ₂.2.1, Δ₂.2.2] → ∃ θ ∈ Set.Icc 0 (2 * Real.pi), v ∈ family_of_lines θ) →
  let area (Δ : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) :=
  0.5 * abs (Δ.1.1 * (Δ.2.1.2 - Δ.2.2.2) + Δ.2.1.1 * (Δ.2.2.2 - Δ.1.2) + Δ.2.2.1 * (Δ.1.2 - Δ.2.1.2))
  in area Δ₁ = area Δ₂

-- Lean 4 statement to prove that statements B, C, and D are true
theorem true_statements_for_family_of_lines : 
  (¬ ∃ (P : ℝ × ℝ), ∀ θ ∈ Set.Icc 0 (2 * Real.pi), P ∈ family_of_lines θ) ∧ 
  (∃ (P : ℝ × ℝ), ¬ lies_on_any_line P) ∧ 
  (∀ n : ℕ, n ≥ 3 → regular_n_sided_polygon n (Nat.le_refl n)) ∧ 
  equal_area_equilateral_triangles :=
  by 
    -- Proof omitted
    sorry

end true_statements_for_family_of_lines_l335_335247


namespace value_of_x_l335_335565

theorem value_of_x :
  ∃ x : ℝ, x = 1.13 * 80 :=
sorry

end value_of_x_l335_335565


namespace accumulation_point_of_S_l335_335228

/-- Define the set S. -/
def S : Set ℝ := { x | ∃ m n : ℤ, x = (Real.pi^n) / (1992^m) }

/-- Prove that every x ≥ 0 is an accumulation point of S. -/
theorem accumulation_point_of_S (x : ℝ) (hx : x ≥ 0) : ∃ (y ∈ S), ∀ ε > 0, ∃ (z ∈ S), z ≠ y ∧ |z - x| < ε :=
sorry

end accumulation_point_of_S_l335_335228


namespace hyperbola_inequality_l335_335931

-- Define point P on the hyperbola in terms of a and b
theorem hyperbola_inequality (a b : ℝ) (h : (3*a + 3*b)^2 / 9 - (a - b)^2 = 1) : |a + b| ≥ 1 :=
sorry

end hyperbola_inequality_l335_335931


namespace complementary_sets_count_l335_335866

-- Define the attributes
inductive Animal | cat | dog | bird
inductive Color | red | blue | green
inductive Shade | light | medium | dark
inductive Type | wild | domestic | exotic

-- Define a card as a tuple of attributes
structure Card :=
  (animal : Animal)
  (color : Color)
  (shade : Shade)
  (type : Type)

-- Define a complementary set predicate
def is_complementary_set (c1 c2 c3 : Card) : Prop :=
  (c1.animal = c2.animal ∧ c2.animal = c3.animal ∨ c1.animal ≠ c2.animal ∧ c2.animal ≠ c3.animal ∧ c1.animal ≠ c3.animal) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∨ c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.type = c2.type ∧ c2.type = c3.type ∨ c1.type ≠ c2.type ∧ c2.type ≠ c3.type ∧ c1.type ≠ c3.type)

-- Lean statement to prove the number of complementary sets is 720
theorem complementary_sets_count : 
  ∃ (S : Finset (Finset Card)), S.card = 720 ∧ 
    ∀ (s ∈ S), ∃ (c1 c2 c3 : Card), s = {c1, c2, c3} ∧ is_complementary_set c1 c2 c3 :=
sorry

end complementary_sets_count_l335_335866


namespace length_of_median_DN_l335_335206

-- Defining conditions
variables (D E F N : Type) [metric_space D] [metric_space E] [metric_space F]
variables [metric_space N] (DE : E) (DF : F) (DEF : D → E → F → Prop)
variables (midpoint : E → F → N → Prop)
variables (right_angle : ∀ d e f, DEF d e f → angle d e f = π / 2)

-- Assume given conditions
def conditions (d : D) (e : E) (f : F) (n : N) : Prop :=
  DEF d e f ∧ right_angle d e f ∧
  distance d e = 5 ∧
  distance d f = 12 ∧
  midpoint e f n

-- Prove the length of median DN is 6.5 cm
theorem length_of_median_DN (d : D) (e : E) (f : F) (n : N)
  (h : conditions d e f n) : distance d n = 6.5 :=
sorry

end length_of_median_DN_l335_335206


namespace double_integral_value_l335_335842

noncomputable def integral_I : ℝ :=
  ∫ x in 0..2, ∫ y in (x^2 + x - 3)..(3 / 2 * x), (x + y) ∂y ∂x

theorem double_integral_value : integral_I = 14 / 5 := by
  sorry

end double_integral_value_l335_335842


namespace distance_to_asymptote_of_hyperbola_l335_335520

-- Definitions of the conditions
def parabola_focus : (ℝ × ℝ) := (0, 2)

def hyperbola (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def eccentricity (e : ℝ) (a : ℝ) (c : ℝ) : Prop :=
  e = c / a

def distance_from_focus_to_asymptote (focus : ℝ × ℝ) (asymptote : ℝ → ℝ) :=
  let d := λ (x0 y0 : ℝ) A B C => abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)
  d (focus.1) (focus.2) (1) (real.sqrt(3)) (0)
  
-- The proof statement
theorem distance_to_asymptote_of_hyperbola
  (m n : ℝ)
  (h_fparabola : parabola_focus = (0, 2))
  (h_hyperbola : hyperbola m n)
  (h_eccentricity : eccentricity 2 1 2)
  (h_n : n = 1)
  (h_m : m = -(1/3)) :
  distance_from_focus_to_asymptote (0, 2) (λ x, -x / (real.sqrt(3))) = real.sqrt(3) :=
sorry

end distance_to_asymptote_of_hyperbola_l335_335520


namespace hyperbola_and_line_pq_l335_335668

-- Definitions based on conditions
def center_origin (c : ℝ) : Prop := c = 0

def imaginary_axis_length (b : ℝ) : Prop := 2 * sqrt 6 = 2 * b

def right_focus_is (c : ℝ) : Prop := c > 0

def line_l (a c : ℝ) (A : ℝ × ℝ) : Prop := A = (a^2 / c, 0)

def distances_relation (O F : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  ∥O - F∥ = 3 * ∥O - A∥

def hyperbola_equation (h : ℝ × ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, h (x, y) = x^2 / a^2 - y^2 / b^2 - 1

def correct_equation_of_hyperbola (a b : ℝ) : Prop :=
  a = sqrt 3 ∧ b = sqrt 6

-- The main statement we want to prove
theorem hyperbola_and_line_pq :
  ∃ a b : ℝ, correct_equation_of_hyperbola a b ∧
  let hyperbola := λ p : ℝ × ℝ, p.1^2 / a^2 - p.2^2 / b^2 - 1 in
  (∀ c, right_focus_is c →
  ∃ A : ℝ × ℝ, line_l a c A ∧ distances_relation (0, 0) (c, 0) A) →
  ∃ k : ℝ, k ∈ ℝ ∧ (k ≠ sqrt 2 ∧ k ≠ -sqrt 2) → 
  (∀ (F P Q : ℝ × ℝ), 
    P ≠ Q ∧
    ∀ x y, x = x - sqrt 2 * y - 3 ∨ x = x + sqrt 2 * y - 3) → 
  hyperbola (x, 0) = 0 :=
sorry

end hyperbola_and_line_pq_l335_335668


namespace number_of_sheep_l335_335695

theorem number_of_sheep (S H : ℕ)
  (ratio : S / H = 5 / 7)
  (horse_food_daily : 230)
  (total_food_daily : 12880)
  (food_equation : H * horse_food_daily = total_food_daily) :
  S = 40 :=
by
  sorry

end number_of_sheep_l335_335695


namespace largest_prime_factor_l335_335733

-- Define the expressions given in the problem
def expression := 16^4 + 2 * 16^2 + 1 - 15^4

-- State the problem of finding the largest prime factor
theorem largest_prime_factor : ∃ p : ℕ, nat.prime p ∧ p ∣ expression ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ expression → q ≤ 241 :=
by {
  sorry  -- Proof needed
}

end largest_prime_factor_l335_335733


namespace vector_subtraction_l335_335494

def a : ℝ × ℝ × ℝ := (-5, 1, 3)
def b : ℝ × ℝ × ℝ := (3, -1, 2)

theorem vector_subtraction : a.1 - 2 * b.1 = -11 ∧ a.2 - 2 * b.2 = 3 ∧ a.3 - 2 * b.3 = -1 :=
by
  sorry

end vector_subtraction_l335_335494


namespace relationship_between_x_y_z_l335_335253

variables (x y z : ℝ)
def F := 50
def S := 75

theorem relationship_between_x_y_z 
  (h1 : x * z = F) 
  (h2 : y * z = S) : 
  y = 1.5 * x :=
begin 
  sorry 
end

end relationship_between_x_y_z_l335_335253


namespace arithmetic_sequence_sum_l335_335103

open Nat

theorem arithmetic_sequence_sum : 
  let d := λ (n : ℕ), (1996 / (n - 1)) in
  (∀ n : ℕ, d n ∈ ℕ ∧ 1 + (n - 1) * d n = 1997 ∧ n > 3) → 
  (Finset.sum (Finset.filter (λ n, d n ∈ ℕ ∧ 1 + (n - 1) * d n = 1997 ∧ n > 3) (Finset.range 1998)) id) = 3501 :=
by
  sorry

end arithmetic_sequence_sum_l335_335103


namespace percentage_of_square_shaded_l335_335212

theorem percentage_of_square_shaded {PQRS : Type} [Square PQRS] 
  (divided_into_four : PQRS = divide_into_four_identical_squares PQRS)
  (one_shaded : is_shaded (one_of_four_small_squares PQRS)) :
  shaded_percentage PQRS = 25 :=
by
  sorry

end percentage_of_square_shaded_l335_335212


namespace price_decrease_l335_335693

theorem price_decrease (current_price original_price : ℝ) (h1 : current_price = 684) (h2 : original_price = 900) :
  ((original_price - current_price) / original_price) * 100 = 24 :=
by
  sorry

end price_decrease_l335_335693


namespace solve_equation_l335_335282

noncomputable def is_solution (x : ℝ) : Prop :=
  (x / (2 * Real.sqrt 2) + (5 * Real.sqrt 2) / 2) * Real.sqrt (x^3 - 64 * x + 200) = x^2 + 6 * x - 40

noncomputable def conditions (x : ℝ) : Prop :=
  (x^3 - 64 * x + 200) ≥ 0 ∧ x ≥ 4

theorem solve_equation :
  (∀ x, is_solution x → conditions x) = (x = 6 ∨ x = 1 + Real.sqrt 13) :=
by sorry

end solve_equation_l335_335282


namespace recurring_decimal_exceeds_fixed_decimal_l335_335036

theorem recurring_decimal_exceeds_fixed_decimal :
  let x := (9 : ℚ) / 11       -- representation of 0.\overline{81}
  let y := (81 : ℚ) / 100    -- representation of 0.81
  x - y = (9 : ℚ) / 1100 :=  -- the result we want to show
by
  have lemma1 : x = (9 : ℚ) / 11 := rfl
  have lemma2 : y = (81 : ℚ) / 100 := rfl
  sorry

end recurring_decimal_exceeds_fixed_decimal_l335_335036


namespace average_birds_seen_l335_335254

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end average_birds_seen_l335_335254


namespace arithmetic_geometric_sequence_general_term_arithmetic_geometric_sequence_sum_l335_335112

noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)

theorem arithmetic_geometric_sequence_general_term :
  (∀ n : ℕ, a_n n = 2^(n-1)) :=
begin
  sorry
end

theorem arithmetic_geometric_sequence_sum (n : ℕ) :
  (finset.range n).sum a_n = 2^n - 1 :=
begin
  sorry
end

end arithmetic_geometric_sequence_general_term_arithmetic_geometric_sequence_sum_l335_335112


namespace min_value_of_sin_x_plus_sin_z_thm_l335_335882

noncomputable def min_value_of_sin_x_plus_sin_z 
    (x y z : ℝ) 
    (h1 : sqrt 3 * Real.cos x = Real.cot y) 
    (h2 : 2 * Real.cos y = Real.tan z) 
    (h3 : Real.cos z = 2 * Real.cot x) : ℝ :=
  min (sin x + sin z)

theorem min_value_of_sin_x_plus_sin_z_thm 
    (x y z : ℝ)
    (h1 : sqrt 3 * Real.cos x = Real.cot y)
    (h2 : 2 * Real.cos y = Real.tan z)
    (h3 : Real.cos z = 2 * Real.cot x) :
  min_value_of_sin_x_plus_sin_z x y z h1 h2 h3 = -7 * sqrt 2 / 6 :=
sorry

end min_value_of_sin_x_plus_sin_z_thm_l335_335882


namespace find_ordered_pair_l335_335875

theorem find_ordered_pair (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  18 * m * n = 72 - 9 * m - 4 * n ↔ (m = 8 ∧ n = 36) := 
by 
  sorry

end find_ordered_pair_l335_335875


namespace value_of_x_in_fractions_when_written_as_fraction_is_99900_form_l335_335347

theorem value_of_x_in_fractions (x : ℝ) (h : x = 0.421571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571571ก่ย
∧ 1000 * x = 112 * 0.42156240967281313233522 + 0.421156247849871041225577authentication

1. 999 * x = 420.735
2. x = 420735 / 1000      ∧ 1000 × x>10+%g57'°01^0-0?
3. x = 4207359 / 999000

open_locale nat

theorem when_written_as_fraction_is_99900_form :
  x = 4207359 := sorryb ∧        lean<pre-contained∇ elementa and autonomous >

end value_of_x_in_fractions_when_written_as_fraction_is_99900_form_l335_335347


namespace numberOfValidConfigurations_l335_335434

noncomputable theory

-- Definition of a valid configuration
def isValidConfiguration (g : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 16) ∧
  (∀ i j, i < 3 → j < 3 → g i j < g (i + 1) j ∧ g i j < g i (j + 1)) ∧
  g 3 3 = 16

-- Statement to prove the number of valid configurations
theorem numberOfValidConfigurations : Σ' (g : Matrix (Fin 4) (Fin 4) ℕ), isValidConfiguration g = 4096 :=
sorry

end numberOfValidConfigurations_l335_335434


namespace number_of_elements_in_intersection_l335_335536

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (2 - x^2) }
def setB : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem number_of_elements_in_intersection :
  Finset.card (Finset.filter (λ x, x ∈ setA) (Finset.filter (λ x, x ∈ setB) (Finset.range 5))) = 3 :=
by
  sorry

end number_of_elements_in_intersection_l335_335536


namespace compute_fractional_sum_l335_335239

variable {R : Type*} [CommRing R]

def r : R := sorry
def s : R := sorry
def t : R := sorry

noncomputable def polynomial : Polynomial R := X^3 - 5 * X^2 + 6 * X - 9

axiom roots_of_polynomial:
  Polynomial.RootOf polynomial r ∧ Polynomial.RootOf polynomial s ∧ Polynomial.RootOf polynomial t

axiom vieta_sum :
  r + s + t = 5

axiom vieta_product_of_pairs :
  r * s + r * t + s * t = 6

axiom vieta_product :
  r * s * t = 9

theorem compute_fractional_sum :
  (r * s / t) + (s * t / r) + (t * r / s) = -6 :=
by
  sorry

end compute_fractional_sum_l335_335239


namespace meaningful_sqrt_l335_335965

theorem meaningful_sqrt (a : ℝ) (h : a - 4 ≥ 0) : a ≥ 4 :=
sorry

end meaningful_sqrt_l335_335965


namespace number_to_add_for_average_l335_335345

theorem number_to_add_for_average (x : ℝ) : (6 + 16 + 8 + 12 + 21 + x) / 6 = 17 → x = 39 := 
by 
  intro h
  -- hint: Add the necessary assumptions for Lean to process the equations
  suffices : 63 + x = 102, from
    suffices x = 39, by exact this
  sorry

end number_to_add_for_average_l335_335345


namespace a1_through_a2013_sum_l335_335951

theorem a1_through_a2013_sum :
  ∀ (a : ℕ → ℝ), (1 - 2 * (1/2))^2013 = (∑ i in range 2014, a i * ((1/2) ^ i)) →
  a 0 = 1 →
  (∑ i in range 2013, a (i+1) / 2^(i+1)) = -1 :=
by
  intros a h_power_series h_a0
  -- Proof will go here
  sorry

end a1_through_a2013_sum_l335_335951


namespace min_box_coeff_l335_335180

theorem min_box_coeff (a b c d : ℤ) (h_ac : a * c = 40) (h_bd : b * d = 40) : 
  ∃ (min_val : ℤ), min_val = 89 ∧ (a * d + b * c) ≥ min_val :=
sorry

end min_box_coeff_l335_335180


namespace find_m_l335_335680

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 3

theorem find_m :
  (∀ x, -2 ≤ x → differentiable_at ℝ (quadratic_function m) x ∧ derivative (quadratic_function m) x ≥ 0) ∧
  (∀ x, x ≤ -2 → differentiable_at ℝ (quadratic_function m) x ∧ derivative (quadratic_function m) x ≤ 0) →
  m = -8 :=
sorry

end find_m_l335_335680


namespace product_of_axes_l335_335642

-- Definitions based on conditions
def ellipse (a b : ℝ) : Prop :=
  a^2 - b^2 = 64

def triangle_incircle_diameter (a b : ℝ) : Prop :=
  b + 8 - a = 4

-- Proving that (AB)(CD) = 240
theorem product_of_axes (a b : ℝ) (h₁ : ellipse a b) (h₂ : triangle_incircle_diameter a b) : 
  (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_of_axes_l335_335642


namespace cannot_sum_to_nine_l335_335051

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l335_335051


namespace trains_speed_ratio_l335_335719

-- Define the conditions
variables (V1 V2 L1 L2 : ℝ)
axiom time1 : L1 = 27 * V1
axiom time2 : L2 = 17 * V2
axiom timeTogether : L1 + L2 = 22 * (V1 + V2)

-- The theorem to prove the ratio of the speeds
theorem trains_speed_ratio : V1 / V2 = 7.8 :=
sorry

end trains_speed_ratio_l335_335719


namespace area_trianglе_D_O_E_l335_335263

noncomputable def triangleArea (p : ℝ) : ℝ := 
  (1 / 2) * 15 * p

theorem area_trianglе_D_O_E (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 15) :
  triangleArea p = (15 * p) / 2 :=
by
  unfold triangleArea
  -- additional steps would be here, but we use sorry since proof is not required
  sorry

end area_trianglе_D_O_E_l335_335263


namespace extrema_of_f_l335_335729

noncomputable def f (x : ℝ) : ℝ := Math.sin (Math.cos x ^ 2) + Math.sin (Math.sin x ^ 2)

theorem extrema_of_f (x : ℝ) : ∃ k : ℤ, x = k * (Real.pi / 4) → 
  is_extremum (f x) := 
sorry

end extrema_of_f_l335_335729


namespace derivatives_at_zero_l335_335607

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : ∀ n : ℕ, f (1 / (n + 1)) = (n + 1)^2 / ((n + 1)^2 + 1)

theorem derivatives_at_zero :
  f 0 = 1 ∧ 
  deriv f 0 = 0 ∧ 
  deriv (deriv f) 0 = -2 ∧ 
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by
  sorry

end derivatives_at_zero_l335_335607


namespace remainder_of_sum_is_13_l335_335327

theorem remainder_of_sum_is_13 
  (x y z : ℤ) 
  (hx : x % 20 = 7) 
  (hy : y % 20 = 11) 
  (hz : z % 20 = 15) : 
  (x + y + z) % 20 = 13 := 
begin
  sorry
end

end remainder_of_sum_is_13_l335_335327


namespace solve_for_X_l335_335658

noncomputable def X : ℝ := 3^(80/27)

theorem solve_for_X (X : ℝ) :
  (sqrt (X^3) = 81 * (81)^(1/9)) → X = 3^(80/27) :=
by
  intro h,
  sorry

end solve_for_X_l335_335658


namespace residue_mod_13_l335_335827

theorem residue_mod_13 :
  (250 ≡ 3 [MOD 13]) → 
  (20 ≡ 7 [MOD 13]) → 
  (5^2 ≡ 12 [MOD 13]) → 
  ((250 * 11 - 20 * 6 + 5^2) % 13 = 3) :=
by 
  sorry

end residue_mod_13_l335_335827


namespace last_digit_of_7_to_the_7_l335_335078

theorem last_digit_of_7_to_the_7 :
  (7 ^ 7) % 10 = 3 :=
by
  sorry

end last_digit_of_7_to_the_7_l335_335078


namespace find_min_max_S_l335_335700

-- Define the sequence of non-negative reals and conditions
def s (n : ℕ) (i : Fin n) : ℝ := sorry

-- Given conditions
axiom sum_s (n : ℕ) : n = 2004 → (∑ i, s n i) = 2
axiom cyclic_product_sum (n : ℕ) : n = 2004 → (∑ i, (s n i) * (s n ((i + 1) % n))) = 1

-- Defining S
def S (n : ℕ) : ℝ := ∑ i, (s n i) ^ 2

-- The proof problem: finding the min and max value of S
theorem find_min_max_S (n : ℕ) (h_n : n = 2004) :
  (∀ (s : Fin n → ℝ), ∀ h1 : (∑ i, s i) = 2, ∀ h2 : (∑ i, (s i) * (s ((i + 1) % n))) = 1,
    ∃ (min_S max_S : ℝ), min_S = 3 / 2 ∧ max_S = 2 ∧ S = min_S ∧ S = max_S) := 
sorry

end find_min_max_S_l335_335700


namespace Problem_statements_l335_335495

theorem Problem_statements (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = a * b) :
  (a + b ≥ 4) ∧
  ¬(a * b ≤ 4) ∧
  (a + 4 * b ≥ 9) ∧
  (1 / a ^ 2 + 2 / b ^ 2 ≥ 2 / 3) :=
by sorry

end Problem_statements_l335_335495


namespace solve_for_y_l335_335659

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4)) : y = 1296 := by
  sorry

end solve_for_y_l335_335659


namespace parabola_y_intercept_l335_335542

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- State the theorem
theorem parabola_y_intercept : 
  ∃! y : ℝ, ∃ x : ℝ, x = 0 ∧ y = parabola x :=
begin
  -- Proof here
  sorry
end

end parabola_y_intercept_l335_335542


namespace binom_8_5_eq_56_l335_335836

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l335_335836


namespace MN_length_l335_335330

variable (a b : ℝ)
variable (AD BC : Real)
variable (M N : Real)
variables [fact (AD = a)] [fact (BC = b)]

theorem MN_length (a b : ℝ) (AD BC M N : ℝ) (h1 : AD = a) (h2 : BC = b) (MN_parallel_to_bases : MN a b) :
    MN = (2 * a * b) / (a + b) :=
sorry

end MN_length_l335_335330


namespace smallest_possible_n_l335_335684

-- Definitions needed for the problem
variable (x n : ℕ) (hpos : 0 < x)
variable (m : ℕ) (hm : m = 72)

-- The conditions as already stated
def gcd_cond := Nat.gcd 72 n = x + 8
def lcm_cond := Nat.lcm 72 n = x * (x + 8)

-- The proof statement
theorem smallest_possible_n (h_gcd : gcd_cond x n) (h_lcm : lcm_cond x n) : n = 8 :=
by 
  -- Intuitively outline the proof
  sorry

end smallest_possible_n_l335_335684


namespace segment_AB_length_l335_335988

-- Define the problem conditions
variables (AB CD h : ℝ)
variables (x : ℝ)
variables (AreaRatio : ℝ)
variable (k : ℝ := 5 / 2)

-- The given conditions
def condition1 : Prop := AB = 5 * x ∧ CD = 2 * x
def condition2 : Prop := AB + CD = 280
def condition3 : Prop := h = AB - 20
def condition4 : Prop := AreaRatio = k

-- The statement to prove
theorem segment_AB_length (h k : ℝ) (x : ℝ) :
  (AB = 5 * x ∧ CD = 2 * x) ∧ (AB + CD = 280) ∧ (h = AB - 20) ∧ (AreaRatio = k) → AB = 200 :=
by 
  sorry

end segment_AB_length_l335_335988


namespace intersection_square_distance_eq_l335_335669

theorem intersection_square_distance_eq :
  let C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 9 }
  let C2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 4)^2 = 5 }
  let inters := C1 ∩ C2
  let points := (λ p : ℝ × ℝ, p ∈ inters) 
  let x1 := (1 : ℝ) + real.sqrt 2.75
  let x2 := (1 : ℝ) - real.sqrt 2.75
  let y := (2.5 : ℝ)
  let p1 := (x1, y)
  let p2 := (x2, y)
  let dist_sq : ℝ := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  dist_sq = 11 :=
by
  sorry

end intersection_square_distance_eq_l335_335669


namespace non_neg_int_solutions_l335_335056

def operation (a b : ℝ) : ℝ := a * (a - b) + 1

theorem non_neg_int_solutions (x : ℕ) :
  2 * (2 - x) + 1 ≥ 3 ↔ x = 0 ∨ x = 1 := by
  sorry

end non_neg_int_solutions_l335_335056


namespace part_one_part_two_l335_335898

def sequence (a : ℕ → ℕ) := ∀ n : ℕ, a (n + 1) - 3 * a n = 2 * 3^n

theorem part_one (a : ℕ → ℕ) (h₀ : a 1 = 3) (h : sequence a) :
  ∃ d : ℚ, ∀ n : ℕ, (a (n + 1) / 3^(n + 1)) - (a n / 3^n) = d :=
by
  sorry

theorem part_two (a : ℕ → ℕ) (h₀ : a 1 = 3) (h : sequence a) :
  ∑ i in Finset.range n, a (i + 1) = n * 3^n :=
by
  sorry

end part_one_part_two_l335_335898


namespace find_min_value_l335_335099

theorem find_min_value (a x y : ℝ) (h : y = -x^2 + 3 * Real.log x) : ∃ x, ∃ y, (a - x)^2 + (a + 2 - y)^2 = 8 :=
by
  sorry

end find_min_value_l335_335099


namespace function_properties_l335_335681

noncomputable def f (x : ℝ) : ℝ := sin (x + π/4) ^ 2 + cos (x - π/4) ^ 2 - 1

theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
by
  sorry

end function_properties_l335_335681


namespace selection_of_representatives_l335_335775

theorem selection_of_representatives 
  (females : ℕ) (males : ℕ)
  (h_females : females = 3) (h_males : males = 4) :
  (females ≥ 1 ∧ males ≥ 1) →
  (females * (males * (males - 1) / 2) + (females * (females - 1) / 2 * males) = 30) := 
by
  sorry

end selection_of_representatives_l335_335775


namespace rhombus_area_24_l335_335153

-- Definitions based only on conditions
def is_rhombus (a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0)

def perimeter_eq_20 (s : ℝ) : Prop :=
  4 * s = 20

def ratio_diagonals (d1 d2 : ℝ) : Prop :=
  d1 / d2 = 4 / 3

def area_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

-- The theorem to be proved
theorem rhombus_area_24 {s d1 d2 : ℝ} 
  (hs : perimeter_eq_20 s)
  (hr : ratio_diagonals d1 d2)
  (hdiag : is_rhombus d1 d2 ∧ sqrt ((d1 / 2)^2 + (d2 / 2)^2) = s) :
  area_rhombus d1 d2 = 24 :=
sorry

end rhombus_area_24_l335_335153


namespace alien_takes_home_l335_335817

variable (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ)

-- Conditions
def initial_abducted_people : abducted = 200 := rfl
def percentage_returned_people : returned_percentage = 0.8 := rfl
def people_taken_to_another_planet : taken_to_another_planet = 10 := rfl

-- The question to prove
def people_taken_home (abducted : ℕ) (returned_percentage : ℚ) (taken_to_another_planet : ℕ) : ℕ :=
  let returned := (returned_percentage * abducted) in
  let remaining := abducted - returned in
  remaining - taken_to_another_planet

theorem alien_takes_home :
  people_taken_home 200 0.8 10 = 30 :=
by
  -- calculations directly in Lean or use sorry to represent the correctness
  sorry

end alien_takes_home_l335_335817


namespace ratio_DE_EF_l335_335217

-- Declare the triangle points A, B, C
variables {A B C : Type} [add_comm_group A]

-- Declare vectors pointing to points D and E on the segments AB and BC respectively
variables (D E F: A)
-- Declare scalars based on given ratios
variables (d_ratio e_ratio : ℚ)

-- Point D divides segment AB in the ratio 4:1
def D_point (A B : A) : A := (1/5) • A + (4/5) • B

-- Point E divides segment BC in the ratio 2:3
def E_point (B C : A) : A := (2/5) • B + (3/5) • C

-- F is the intersection point of lines DE and AC
def F_point (D E A C : A) : A := 10 • E + -5 • D

-- Proving the required ratio DE / EF = 1/5
theorem ratio_DE_EF (A B C D E F : A) 
  (hD : D = D_point A B) 
  (hE : E = E_point B C) 
  (hF : F = F_point D E A C) :
  ∃ ratio : ℚ, ratio = 1 / 5 := 
begin
  use (1 / 5),
  sorry
end

end ratio_DE_EF_l335_335217


namespace ratio_fifteenth_term_l335_335232

variables (a d b e : ℝ)

def S (n : ℕ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)
def T (n : ℕ) : ℝ := (n / 2) * (2 * b + (n - 1) * e)

theorem ratio_fifteenth_term 
  (h : ∀ n : ℕ, (S n / T n) = (5 * n + 3 : ℝ) / (3 * n + 20 : ℝ)) : 
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by
  sorry

end ratio_fifteenth_term_l335_335232


namespace exponential_function_fixed_point_l335_335302

theorem exponential_function_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : (0, 1) ∈ set_of (λ (p : ℝ × ℝ), p.2 = a ^ p.1) :=
by
  sorry

end exponential_function_fixed_point_l335_335302


namespace area_triangle_SQM_eq_6_l335_335313

-- Define the conditions of the problem
def length_PQ := 8
def width_PS := 6
def diagonal_PR := (Real.sqrt (length_PQ^2 + width_PS^2) : ℝ)
def segment_length := diagonal_PR / 4

-- Define the height calculation from the area of triangle PSR
def height := (2 * (length_PQ * width_PS) / diagonal_PR)

-- Calculate the area of triangle SQM where QM is the base and height is the one derived from above
def area_SQM := (segment_length * height) / 2

-- The proof statement
theorem area_triangle_SQM_eq_6 :
  area_SQM = 6 := 
sorry

end area_triangle_SQM_eq_6_l335_335313


namespace intersection_M_N_l335_335133

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l335_335133


namespace combination_8_5_eq_56_l335_335840

theorem combination_8_5_eq_56 : nat.choose 8 5 = 56 :=
by
  sorry

end combination_8_5_eq_56_l335_335840


namespace unique_integers_sum_l335_335712

theorem unique_integers_sum :
  ∃ (b2 b3 b4 b5 b6 b7 b8 : ℤ), 
  5 / 8 = b2 / 2! + b3 / 3! + b4 / 4! + b5 / 5! + b6 / 6! + b7 / 7! + b8 / 8! ∧
  (0 ≤ b2 ∧ b2 < 2) ∧
  (0 ≤ b3 ∧ b3 < 3) ∧
  (0 ≤ b4 ∧ b4 < 4) ∧
  (0 ≤ b5 ∧ b5 < 5) ∧
  (0 ≤ b6 ∧ b6 < 6) ∧
  (0 ≤ b7 ∧ b7 < 7) ∧
  (0 ≤ b8 ∧ b8 < 8) ∧
  b2 + b3 + b4 + b5 + b6 + b7 + b8 = 4 :=
by
  sorry

end unique_integers_sum_l335_335712
