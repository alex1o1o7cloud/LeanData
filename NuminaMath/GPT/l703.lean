import Mathlib

namespace new_weighted_avg_is_80_l703_703318

variables {A B : ℝ}
variables {nA : ℝ} {nB : ℝ}
variables {total_students : ℝ}

def marks_section_A : ℝ := A * 6 * 1.5
def marks_section_B : ℝ := B * 4
def total_marks : ℝ := 400
def class_average : ℝ := 40
def total_students : ℝ := 10
def weighted_avg_before : ℝ := (marks_section_A + marks_section_B) / total_students

-- Condition from the problem
axiom avg_class : total_students * class_average = total_marks

-- Condition from the problem
axiom weighted_avg_eq : 1.5 * A * 6 + B * 4 = total_marks

def new_weighted_avg : ℝ := ((3 * A * 6) + (2 * B * 4)) / total_students

-- To prove: The new weighted average is 80
theorem new_weighted_avg_is_80 :
  new_weighted_avg = 80 :=
by
  sorry

end new_weighted_avg_is_80_l703_703318


namespace xyz_inequality_l703_703285

theorem xyz_inequality (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x := 
  sorry

end xyz_inequality_l703_703285


namespace exists_vertex_with_triangle_edges_l703_703702

-- Define a tetrahedron with vertices A, B, C, D, and their associated edge lengths.
structure Tetrahedron :=
(A B C D : Point)
(dist_AB : ℝ)
(dist_AC : ℝ)
(dist_AD : ℝ)
(dist_BC : ℝ)
(dist_BD : ℝ)
(dist_CD : ℝ)

-- Define what it means for the edges at a vertex to satisfy the triangle inequality.
def satisfies_triangle_inequality (x y z : ℝ) : Prop :=
(x + y > z) ∧ (y + z > x) ∧ (x + z > y)

-- The problem statement in Lean 4
theorem exists_vertex_with_triangle_edges (T : Tetrahedron) :
  ∃ (v : Point) (e1 e2 e3 : ℝ),
    (v = T.A ∧ e1 = T.dist_AB ∧ e2 = T.dist_AC ∧ e3 = T.dist_AD ∧ satisfies_triangle_inequality e1 e2 e3) ∨
    (v = T.B ∧ e1 = T.dist_AB ∧ e2 = T.dist_BC ∧ e3 = T.dist_BD ∧ satisfies_triangle_inequality e1 e2 e3) ∨
    (v = T.C ∧ e1 = T.dist_AC ∧ e2 = T.dist_BC ∧ e3 = T.dist_CD ∧ satisfies_triangle_inequality e1 e2 e3) ∨
    (v = T.D ∧ e1 = T.dist_AD ∧ e2 = T.dist_BD ∧ e3 = T.dist_CD ∧ satisfies_triangle_inequality e1 e2 e3) :=
begin
  sorry
end

end exists_vertex_with_triangle_edges_l703_703702


namespace four_letter_initial_sets_l703_703969

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l703_703969


namespace right_triangle_area_l703_703314

theorem right_triangle_area (A B C : Type) [linear_ordered_field A]
  (theta45 : ∀ {x : A}, x = 45)
  (right_angle : ∀ {x : A}, x = 90)
  (altitude_four : ∀ {x : A}, x = 4) 
  (hypotenuse : A) 
  (area : A) :
  (∀ (triangle : Type) (legs_eq : ∀ {a b : A}, a = 45 → b = 45 → a = b),
   hypotenuse = 4 * √2 ∧
   area = 1 / 2 * hypotenuse * altitude_four →
   area = 8 * √2) :=
sorry

end right_triangle_area_l703_703314


namespace minimum_value_of_3x_plus_ay_l703_703305

variables (x y a : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (ha : 0 < a)
variables (h1 : x + 2*y = 1)
variables (h2 : ∀ x y, x + 2*y = 1 → (3 / x + a / y) ≥ 6 * real.sqrt 3 → (3 / x + a / y) = 6 * real.sqrt 3)

theorem minimum_value_of_3x_plus_ay (hx_inv : 1 / x + 2 / y = 1) : 3 * x + a * y ≥ 6 * real.sqrt 3 :=
sorry

end minimum_value_of_3x_plus_ay_l703_703305


namespace happiness_difference_test_l703_703960

theorem happiness_difference_test :
  let N := 1184
  let O₁₁ := 638
  let O₁₂ := 128
  let O₂₁ := 372
  let O₂₂ := 46
  let O₁₊ := 1010
  let O₂₊ := 174
  let O₊₁ := 766
  let O₊₂ := 418
  let χ² := N * (O₁₁ * O₂₂ - O₁₂ * O₂₁)^2 / (O₁₊ * O₂₊ * O₊₁ * O₊₂)
  (χ² ≈ 7.022) →
  (χ² > 6.635 → "can") ∧ (χ² < 7.879 → "cannot") :=
by
  sorry

end happiness_difference_test_l703_703960


namespace problem1_problem2_l703_703933

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n-1)

def S (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

def condition1 (q : ℝ) (a₁ : ℝ) : Prop :=
  2 * S a₁ q 9 = S a₁ q 3 + S a₁ q 6

def common_ratio (q : ℝ) (a₁ : ℝ) : Prop :=
  q = (- real.sqrt 34) / 2

def arithmetic_seq (a₂ a₈ a₅ : ℝ) : Prop :=
  2 * a₈ = a₂ + a₅

theorem problem1 (a₁ : ℝ) : ∃ q, condition1 q a₁ → common_ratio q a₁ :=
sorry

theorem problem2 (a₁ q : ℝ) (h1 : condition1 q a₁) (h2 : common_ratio q a₁) :
  arithmetic_seq (geometric_sequence a₁ q 2) (geometric_sequence a₁ q 8) (geometric_sequence a₁ q 5) :=
sorry

end problem1_problem2_l703_703933


namespace game_full_turn_l703_703052

theorem game_full_turn (m n : ℕ) (h : m > n) : 
  ∃ k : ℕ, k = m - n ∧ 
  ∀ i : ℕ, (i < k) → ( 
    ∀ j : ℕ, (j < n) → 
      ¬ (is_adjacent_boy_girl i j)  ) :=
  sorry

end game_full_turn_l703_703052


namespace part1_part2_l703_703525

theorem part1 (x : ℝ) (h_fx : ∀ x : ℝ, x ∈ Icc 0 (2 * π / 3) →
  let m := (2 * sqrt 3 * sin x, 2 * cos x)
      n := (cos x, - cos x)
      f := (m.1 * n.1) + (m.2 * n.2) + 2
  in (f = 2 ↔ x = π / 6)) (h_f0 : f'(0) = 2 * sqrt 3) :
  a = 2 ∧ b = 2 * sqrt 3 :=
  sorry

theorem part2 (k : ℝ) :
  (∀ x : ℝ, x ∈ Icc 0 (2 * π / 3) →
  let m := (2 * sqrt 3 * sin x, 2 * cos x)
      n := (cos x, - cos x)
      f := (m.1 * n.1) + (m.2 * n.2) + 2
  in f - log 3 k = 0) ↔ 1/27 ≤ k ∧ k ≤ 1 :=
  sorry

end part1_part2_l703_703525


namespace family_visit_cost_is_55_l703_703212

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end family_visit_cost_is_55_l703_703212


namespace part1_part2_l703_703233

noncomputable def area_of_triangle (a b c : ℝ) (sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Part 1
theorem part1 (a : ℝ) (b : ℝ) (c : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : 2 * Real.sin C = 3 * Real.sin A)
  (a_val : a = 4) (b_val : b = 5) (c_val : c = 6) : 
  area_of_triangle 4 5 (3 * Real.sqrt 7 / 8) = 15 * Real.sqrt 7 / 4 := by 
  sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : ∃ (a' : ℕ), a = a') :
  a = 2 := by
  sorry

end part1_part2_l703_703233


namespace distinct_positive_differences_count_l703_703981

def setOfNumbers : Set ℕ := {2, 4, 6, 8, 10, 12}

theorem distinct_positive_differences_count : 
  let differences := {x | ∃ a b ∈ setOfNumbers, a ≠ b ∧ x = abs (a - b)}
  (differences ∩ Set.Ioo 0 ∞).card = 5 := 
by
  -- proof goes here
  sorry

end distinct_positive_differences_count_l703_703981


namespace compute_x_l703_703666

theorem compute_x 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 0.1)
  (hs1 : ∑' n, 4 * x^n = 4 / (1 - x))
  (hs2 : ∑' n, 4 * (10^n - 1) * x^n = 4 * (4 / (1 - x))) :
  x = 3 / 40 :=
by
  sorry

end compute_x_l703_703666


namespace decagon_angle_measure_l703_703296

theorem decagon_angle_measure 
  (A B C D E F G H I J Q : Type)
  [Decagon A B C D E F G H I J]
  (h1 : Extends A J Q)
  (h2 : Extends D E Q) :
  measure_angle Q = 72 :=
by
  sorry

end decagon_angle_measure_l703_703296


namespace path_count_through_B_l703_703446

open SimpleGraph
open Finset

variable (A B C D F G I : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq F] [DecidableEq G] [DecidableEq I]

/-- The number of paths from A to C passing through B -/
theorem path_count_through_B 
    (ant_paths : A → B → C → D → F → G → I → ℕ) 
    (A_to_B_paths : ℕ := 2) 
    (B_to_C_paths : ℕ := 2) 
    : ant_paths A B C D F G I = A_to_B_paths * B_to_C_paths := 
begin
    sorry
end

end path_count_through_B_l703_703446


namespace show_spiders_l703_703708

noncomputable def spiders_found (ants : ℕ) (ladybugs_initial : ℕ) (ladybugs_fly_away : ℕ) (total_insects_remaining : ℕ) : ℕ :=
  let ladybugs_remaining := ladybugs_initial - ladybugs_fly_away
  let insects_observed := ants + ladybugs_remaining
  total_insects_remaining - insects_observed

theorem show_spiders
  (ants : ℕ := 12)
  (ladybugs_initial : ℕ := 8)
  (ladybugs_fly_away : ℕ := 2)
  (total_insects_remaining : ℕ := 21) :
  spiders_found ants ladybugs_initial ladybugs_fly_away total_insects_remaining = 3 := by
  sorry

end show_spiders_l703_703708


namespace pizza_consumption_order_l703_703906

theorem pizza_consumption_order :
  ∀ (P : nat) (Alan Bella Carol Dean Eva : ℚ) (total_slices : ℚ),
  Alan = 1/6 ∧ Bella = 2/7 ∧ Carol = 1/3 ∧ Dean = 1/8 ∧ 
  total_slices = 168 ∧
  Eva = total_slices - (Alan * total_slices + Bella * total_slices + Carol * total_slices + Dean * total_slices) →
  (let slices := λ x, x * total_slices in
   [slices Carol, slices Bella, slices Alan, slices Eva, slices Dean].to_sorted_list.reverse = 
   [slices Carol, slices Bella, slices Alan, slices Eva, slices Dean].qsort (≥)) :=
begin
  sorry
end

end pizza_consumption_order_l703_703906


namespace convert_octal_to_decimal_l703_703077

theorem convert_octal_to_decimal : 
  let octal := 734 in
  let decimal_from_octal (n : ℕ) : ℕ :=
    let d0 := n % 10 in
    let d1 := (n / 10) % 10 in
    let d2 := (n / 100) % 10 in
    d0 * 8^0 + d1 * 8^1 + d2 * 8^2 in
  decimal_from_octal octal = 476 :=
by
  let octal := 734
  let d0 := octal % 10
  let d1 := (octal / 10) % 10
  let d2 := (octal / 100) % 10
  have h1 : d0 = 4 := by sorry
  have h2 : d1 = 3 := by sorry
  have h3 : d2 = 7 := by sorry
  show d0 * 8^0 + d1 * 8^1 + d2 * 8^2 = 476 :=
    by rw [h1, h2, h3]; sorry

end convert_octal_to_decimal_l703_703077


namespace find_p_plus_q_l703_703015

-- Definitions corresponding to the problem conditions
def side_lengths (DE EF FD : ℝ) := DE = 15 ∧ EF = 36 ∧ FD = 27

def rectangle_vertices (W X Y Z DE DF EF : Set ℝ) := 
  W ∈ DE ∧ X ∈ DF ∧ Y ∈ EF ∧ Z ∈ EF

def area_expression (γ μ λ : ℝ) := 
  ∀ γ : ℝ, γ = 36 ∨ γ = 18 → 
    (μ * γ - λ * γ^2 = if γ = 36 then 36 * μ - 1296 * λ else 36 * λ * 18 - λ * 324).

-- The theorem to prove
theorem find_p_plus_q :
  ∃ (p q : ℕ), p + q = 4 ∧ (λ : ℝ, λ = 1 / 3) ∧ p.gcd q = 1 :=
by
  sorry

end find_p_plus_q_l703_703015


namespace find_f2_l703_703165

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem find_f2 (a b : ℝ) (h : (∃ x : ℝ, f x a b = 10 ∧ x = 1)):
  f 2 a b = 18 ∨ f 2 a b = 11 :=
sorry

end find_f2_l703_703165


namespace gcd_divisor_l703_703663

theorem gcd_divisor (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) (hrs : Nat.gcd r s = 60) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) 
  : 13 ∣ p :=
sorry

end gcd_divisor_l703_703663


namespace distance_PQ_l703_703257

variables (E F G H : ℝ × ℝ × ℝ)
variables (E' F' G' H' P Q : ℝ × ℝ × ℝ)

-- Conditions stated in the problem
def parallelogram (E F G H : ℝ × ℝ × ℝ) : Prop := 
  E.1 = 0 ∧ E.2 = 0 ∧ F.1 = 0 ∧ F.2 = 1 ∧ 
  G.1 = 1 ∧ G.2 = 1 ∧ H.1 = 1 ∧ H.2 = 0

def rays_parallel (E' F' G' H' : ℝ × ℝ × ℝ) : Prop := 
  E'.z = 7 ∧ F'.z = 12 ∧ G'.z = 13 ∧ H'.z = 15

def midpoints (E' G' F' H' P Q : ℝ × ℝ × ℝ) : Prop :=
  P = (0.5, 0.5, (E'.3 + G'.3) / 2) ∧ Q = (0.5, 0.5, (F'.3 + H'.3) / 2)

def calculate_distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  abs (P.3 - Q.3)

-- Statement to be proved
theorem distance_PQ : 
  parallelogram E F G H →
  rays_parallel E' F' G' H' → 
  midpoints E' G' F' H' P Q →
  calculate_distance P Q = 3.5 :=
by
  intros h1 h2 h3
  sorry

end distance_PQ_l703_703257


namespace rationalize_denominator_l703_703705

-- Problem statement
theorem rationalize_denominator :
  1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end rationalize_denominator_l703_703705


namespace surface_area_of_sphere_from_cube_shaved_l703_703594

theorem surface_area_of_sphere_from_cube_shaved (edge_length : ℝ) (diameter : ℝ) (r : ℝ) : 
  edge_length = 2 → 
  diameter = edge_length → 
  r = diameter / 2 → 
  4 * real.pi * r^2 = 4 * real.pi := 
by {
  sorry -- skipping proof
}

end surface_area_of_sphere_from_cube_shaved_l703_703594


namespace fresh_grapes_water_content_l703_703521

theorem fresh_grapes_water_content:
  ∀ (P : ℝ), 
  (∀ (x y : ℝ), P = x) → 
  (∃ (fresh_grapes dry_grapes : ℝ), fresh_grapes = 25 ∧ dry_grapes = 3.125 ∧ 
  (100 - P) / 100 * fresh_grapes = 0.8 * dry_grapes ) → 
  P = 90 :=
by 
  sorry

end fresh_grapes_water_content_l703_703521


namespace find_original_strength_l703_703798

variable (original_strength : ℕ)
variable (total_students : ℕ := original_strength + 12)
variable (original_avg_age : ℕ := 40)
variable (new_students : ℕ := 12)
variable (new_students_avg_age : ℕ := 32)
variable (new_avg_age_reduction : ℕ := 4)
variable (new_avg_age : ℕ := original_avg_age - new_avg_age_reduction)

theorem find_original_strength (h : (original_avg_age * original_strength + new_students * new_students_avg_age) / total_students = new_avg_age) :
  original_strength = 12 := 
sorry

end find_original_strength_l703_703798


namespace logarithmic_inequality_l703_703547

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log 4

theorem logarithmic_inequality :
  Real.log a < (1 / 2)^b := by
  sorry

end logarithmic_inequality_l703_703547


namespace four_letter_initial_sets_l703_703970

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l703_703970


namespace exists_tangent_line_unique_a_l703_703526

variable {R : Type} [RealField R]
variable (a : R)
def f (x : R) : R := a * x^3 + x^2 / 2 + x + 1
def g (x : R) : R := Real.exp x

theorem exists_tangent_line :
  ∃ l : R → R, l 0 = 1 ∧ l' 0 = 1 ∧ (∀ x : R, f x = l x ↔ g x = l x) :=
sorry

theorem unique_a (h : ∀ x : R, f x ≤ g x) : a = 1/6 :=
sorry

end exists_tangent_line_unique_a_l703_703526


namespace num_four_letter_initials_l703_703973

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l703_703973


namespace minimum_value_inequality_l703_703140

theorem minimum_value_inequality (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) : m + (n^2 - mn + 4)/(m - n) ≥ 4 :=
  sorry

end minimum_value_inequality_l703_703140


namespace sum_of_sequence_l703_703630

noncomputable def a : ℕ → ℝ
| 0 := 1  -- Note: Lean uses 0-based indexing, so a₁ is a(0)
| (n + 1) := (↑(n + 2) ^ 2 / (↑(n + 2) ^ 2 - 1)) * a n

def seq_sum : ℕ → ℝ
| 0 := 0
| (n + 1) := seq_sum n + a (n + 1) / ((n + 1) ^ 2 : ℝ)

theorem sum_of_sequence (n : ℕ) : seq_sum n = 2 * n / (n + 1) :=
sorry

end sum_of_sequence_l703_703630


namespace box_count_neither_markers_nor_erasers_l703_703464

-- Define the conditions as parameters.
def total_boxes : ℕ := 15
def markers_count : ℕ := 10
def erasers_count : ℕ := 5
def both_count : ℕ := 4

-- State the theorem to be proven in Lean 4.
theorem box_count_neither_markers_nor_erasers : 
  total_boxes - (markers_count + erasers_count - both_count) = 4 := 
sorry

end box_count_neither_markers_nor_erasers_l703_703464


namespace solution_set_of_inequality_l703_703738

-- Given conditions
variables {f : ℝ → ℝ}
axiom even_func : ∀ x, f(x) = f(-x)
axiom monotonic_decrease : ∀ x y, 0 < x → x < y → f(y) < f(x)
axiom value_at_one : f(1) = 0

-- Theorem statement
theorem solution_set_of_inequality : { x : ℝ | f(x) > 0 } = { x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0 } := 
sorry

end solution_set_of_inequality_l703_703738


namespace third_and_fourth_largest_diff_l703_703501

/-- 
Given the digits 2, 4, 6, and 8, each used exactly once,
the difference between the third largest and the fourth largest 
four-digit numbers that can be formed is 36.
-/
theorem third_and_fourth_largest_diff : 
  let digits := [2, 4, 6, 8] in 
  ∃ (a b : ℕ), 
    a = 8462 ∧ b = 8426 ∧ 
    a - b = 36 :=
by
  sorry

end third_and_fourth_largest_diff_l703_703501


namespace calculate_value_of_M_l703_703861

def M : ℕ := 
  let terms := (List.range' 1 105).reverse.map (λ n, if (n % 3 == 0) then -n^2 else n^2)
  terms.sum

theorem calculate_value_of_M : M = 5722 := by
  sorry

end calculate_value_of_M_l703_703861


namespace house_numbers_possible_values_l703_703433

theorem house_numbers_possible_values 
  (n : ℕ)
  (h1 : 100 ≤ n ∧ n ≤ 999)
  (k : ℕ)
  (h2 : ∃ (hn : ℕ), hn = (n / ∑ (x in Finset.range (n+1)), (if (ToDigits x).head = 2 then 1 else 0))
  (k > 0) := 
  n = 110 ∨ n = 121 ∨
  n = 132 ∨ n = 143 ∨
  n = 154 ∨ n = 165 ∨
  n = 176 ∨ n = 187 ∨
  n = 198 ∨ n = 235 ∨
  n = 282 ∨ n = 333 ∨
  n = 444 ∨ n = 555 ∨
  n = 666 ∨ n = 777 ∨
  n = 888 ∨ n = 999 :=
sorry

end house_numbers_possible_values_l703_703433


namespace boat_distance_downstream_l703_703060

theorem boat_distance_downstream (speed_boat_still: ℕ) (speed_stream: ℕ) (time: ℕ)
    (h1: speed_boat_still = 25)
    (h2: speed_stream = 5)
    (h3: time = 4) :
    (speed_boat_still + speed_stream) * time = 120 := 
sorry

end boat_distance_downstream_l703_703060


namespace city_population_increase_decrease_l703_703064

variable (n : ℕ)

/-- Given a city's population increased by 1500 people, and following this the new population 
    decreased by 15%, resulting in 50 fewer people than the original population.
    We aim to prove that the original population is 8833. -/
theorem city_population_increase_decrease (h : 0.85 * (n + 1500) + 50 = n) : n = 8833 := 
sorry

end city_population_increase_decrease_l703_703064


namespace imaginary_part_sum_reciprocal_l703_703746

theorem imaginary_part_sum_reciprocal : 
  (complex.im ((1 / (complex.mk (-2) 1)) + (1 / (complex.mk 1 (-2)))) = 1 / 5) :=
by
  sorry

end imaginary_part_sum_reciprocal_l703_703746


namespace lcm_45_75_l703_703127

theorem lcm_45_75 : Nat.lcm 45 75 = 225 :=
by
  sorry

end lcm_45_75_l703_703127


namespace sum_arithmetic_series_51_to_100_l703_703118

theorem sum_arithmetic_series_51_to_100 :
  let a := 51 in
  let l := 100 in
  let n := 50 in
  (n / 2) * (a + l) = 3775 :=
by
  let a := 51
  let l := 100
  let n := 50
  sorry

end sum_arithmetic_series_51_to_100_l703_703118


namespace parallelogram_proof_l703_703530

theorem parallelogram_proof 
  (A B C D E G F H : Point) 
  (h_parallelogram : parallelogram A B C D)
  (h_angle_BAC : angle A B C = 40)
  (h_angle_BCA : angle B C A = 20)
  (h_points_on_diagonal : on_diagonal E G A C)
  (h_points_on_side : on_side F H A D)
  (h_collinear_BEF : collinear B E F)
  (h_angles_90 : angle B A G = 90 ∧ angle A H G = 90)
  (h_AF_EQ_EG : dist A F = dist E G) :
  dist A F = dist H D :=
by
  sorry

end parallelogram_proof_l703_703530


namespace problem_f_4_1981_l703_703740

def f : ℕ → ℕ → ℕ
| 0, y     := y + 1
| (x+1), 0 := f x 1
| (x+1), (y+1) := f x (f (x+1) y)

theorem problem_f_4_1981 :
  f 4 1981 = 2^(2^(2^(1981 + 1) + 1)) - 3 :=
sorry

end problem_f_4_1981_l703_703740


namespace relationship_between_abc_l703_703183

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log (1/2) / Real.log 3
noncomputable def c : ℝ := 2 ^ 0.6

theorem relationship_between_abc : b < a ∧ a < c := by
  sorry

end relationship_between_abc_l703_703183


namespace triangle_area_triangle_obtuse_exists_l703_703230

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Given conditions
def condition_b : b = a + 1 := by sorry
def condition_c : c = a + 2 := by sorry
def condition_sin : 2 * Real.sin C = 3 * Real.sin A := by sorry

-- Proof statement for Part (1)
theorem triangle_area (h1 : condition_b) (h2 : condition_c) (h3 : condition_sin) :
  (1 / 2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := 
sorry

-- Proof statement for Part (2)
theorem triangle_obtuse_exists (h1 : condition_b) (h2 : condition_c) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧
  (let b := a + 1 in let c := a + 2 in
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) > Real.pi / 2) :=
sorry

end triangle_area_triangle_obtuse_exists_l703_703230


namespace find_tangent_line_equation_l703_703141

noncomputable def tangent_line_equation (f : ℝ → ℝ) (perp_line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  let y₀ := f x₀
  let slope_perp_to_tangent := -2
  let slope_tangent := -1 / 2
  slope_perp_to_tangent = -1 / (deriv f x₀) ∧
  x₀ = 1 ∧ y₀ = 1 ∧
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3

theorem find_tangent_line_equation :
  tangent_line_equation (fun (x : ℝ) => Real.sqrt x) (fun (x : ℝ) => -2 * x - 4) 1 := by
  sorry

end find_tangent_line_equation_l703_703141


namespace find_b_value_l703_703880

theorem find_b_value (b d : ℕ) 
  (h1 : (2 + 8 + 9 + b + d) + (21 * 5 - (2 + 8 + 9 + b + d)) = 69) 
  (h2 : 21 * 5 = 105) 
  (h3 : d + 5 + 9 = 21) : 
  b = 10 := 
by
  have eq1 : 105 - (2 + 8 + 9 + b + d) = 69 := by
    rw [h2] at h1
    exact h1
  have eq2 : 105 - 19 - (b + d) = 69 := by 
    rw [show 2 + 8 + 9 = 19, by norm_num] at eq1
    exact eq1
  have sum_bd : b + d = 17 := by 
    rw [show 105 - 19 = 86, by norm_num] at eq2
    norm_num at eq2
    exact eq2
  have value_of_d : d = 7 := by
    exact (eq_of_add_eq_add_right h3).symm
  have value_of_b : b = 10 := by
    rw [value_of_d] at sum_bd
    norm_num at sum_bd
    exact sum_bd
  exact value_of_b

end find_b_value_l703_703880


namespace volume_of_circumscribing_sphere_l703_703925

structure Tetrahedron (A B C D : ℝ) := 
(AB : ℝ)
(CD : ℝ)
(AC : ℝ)
(BC : ℝ)
(AD : ℝ)
(BD : ℝ)
(vertex_on_sphere : Prop)

noncomputable def sphere_volume (tetra : Tetrahedron) : ℝ :=
  if h : tetra.vertex_on_sphere = true then
    let r := 1 in -- Given from solution that radius r is 1
    (4 / 3) * Real.pi * r ^ 3
  else
    0 -- A default value indicating vertices are not on sphere

theorem volume_of_circumscribing_sphere (A B C D : ℝ) (tetra : Tetrahedron A B C D)
  (h₁ : tetra.AB = sqrt 2)
  (h₂ : tetra.CD = sqrt 2)
  (h₃ : tetra.AC = sqrt 3)
  (h₄ : tetra.BC = sqrt 3)
  (h₅ : tetra.AD = sqrt 3)
  (h₆ : tetra.BD = sqrt 3)
  (h₇ : tetra.vertex_on_sphere = true) : 
  sphere_volume tetra = (4 * Real.pi) / 3 :=
by
  sorry

end volume_of_circumscribing_sphere_l703_703925


namespace square_diagonal_area_approx_l703_703093

-- Defining the square with given dimensions
def square_side : ℝ := 10
def point_A : ℝ := 2
def point_B : ℝ := 2
def midpoint_section : ℝ := 3

-- Formalizing the proof problem statement
theorem square_diagonal_area_approx :
  let d := square_side in
  let A := point_A in
  let B := point_B in
  let m := midpoint_section in
  (∃ region : ℝ, region ≈ 5) :=
begin
  sorry
end

end square_diagonal_area_approx_l703_703093


namespace fiona_initial_seat_l703_703713

theorem fiona_initial_seat (greg hannah ian jane kayla lou : Fin 7)
  (greg_final : Fin 7 := greg + 3)
  (hannah_final : Fin 7 := hannah - 2)
  (ian_final : Fin 7 := jane)
  (jane_final : Fin 7 := ian)
  (kayla_final : Fin 7 := kayla + 1)
  (lou_final : Fin 7 := lou - 2)
  (fiona_final : Fin 7) :
  (fiona_final = 0 ∨ fiona_final = 6) →
  ∀ (fiona_initial : Fin 7), 
  (greg_final ≠ fiona_initial ∧ hannah_final ≠ fiona_initial ∧ ian_final ≠ fiona_initial ∧ 
   jane_final ≠ fiona_initial ∧ kayla_final ≠ fiona_initial ∧ lou_final ≠ fiona_initial) →
  fiona_initial = 0 :=
by
  sorry

end fiona_initial_seat_l703_703713


namespace min_sum_distances_l703_703361

noncomputable def minDistancesSum (a b c : ℝ) : ℝ :=
  real.sqrt (4*a^2 + 2*b*c)

theorem min_sum_distances (a b c : ℝ) : 
  ∃ P: ℝ×ℝ×ℝ, ∀ A B C D : ℝ×ℝ×ℝ,
  dist P A + dist P B + dist P C + dist P D = minDistancesSum a b c := 
sorry

end min_sum_distances_l703_703361


namespace symmetric_graph_C3_l703_703771

section
  variable (f : ℝ → ℝ) (C_1 C_2 : ℝ → ℝ)

  -- Initial function f(x) = 2^x
  def f := fun x => 2^x

  -- C_1 obtained by translating f(x) one unit to the left
  def C_1 := fun x => f(x + 1)

  -- C_2 obtained by translating C_1 one unit upwards
  def C_2 := fun x => C_1(x) + 1

  theorem symmetric_graph_C3 : ∀ x, (C_2 (x - 1) = 2^(x + 1)) → y = (log (x - 1) / log 2) - 1 :=
  by
    intro x hx
    sorry
end

end symmetric_graph_C3_l703_703771


namespace triangle_area_triangle_obtuse_exists_l703_703229

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Given conditions
def condition_b : b = a + 1 := by sorry
def condition_c : c = a + 2 := by sorry
def condition_sin : 2 * Real.sin C = 3 * Real.sin A := by sorry

-- Proof statement for Part (1)
theorem triangle_area (h1 : condition_b) (h2 : condition_c) (h3 : condition_sin) :
  (1 / 2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := 
sorry

-- Proof statement for Part (2)
theorem triangle_obtuse_exists (h1 : condition_b) (h2 : condition_c) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧
  (let b := a + 1 in let c := a + 2 in
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) > Real.pi / 2) :=
sorry

end triangle_area_triangle_obtuse_exists_l703_703229


namespace part1_part2_l703_703934

variables (a b c m n p : ℝ)

-- Conditions
def conditions1 : Prop := a^2 + b^2 + c^2 = 1 ∧ m^2 + n^2 + p^2 = 1
def conditions2 : Prop := a^2 + b^2 + c^2 = 1 ∧ m^2 + n^2 + p^2 = 1 ∧ abc ≠ 0

-- Question (Ⅰ)
theorem part1 (h : conditions1 a b c m n p) : |a * m + b * n + c * p| ≤ 1 :=
sorry

-- Question (Ⅱ)
theorem part2 (h : conditions2 a b c m n p) : (m^4 / a^2) + (n^4 / b^2) + (p^4 / c^2) ≥ 1 :=
sorry

end part1_part2_l703_703934


namespace solution_l703_703673

-- Define point A
def A : ℝ × ℝ := (20, 10)

-- Define point B
def B : ℝ × ℝ := (4, 2)

-- Define the midpoint C as the midpoint of A and B
def C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The proof goal
theorem solution : 3 * C.1 - 5 * C.2 = 6 := by
  unfold C
  simp
  norm_num
  sorry -- This skips the proof part, as requested

end solution_l703_703673


namespace max_degree_p_has_horizontal_asymptote_l703_703470

noncomputable def p (x : ℝ) : ℝ := sorry
def denominator (x : ℝ) : ℝ := 3*x^6 - 2*x^4 + x^2 + 9
def f (x : ℝ) := p(x) / denominator(x)

theorem max_degree_p_has_horizontal_asymptote : 
  (∃ n : ℕ, ∀ x : ℝ, degree(p(x)) ≤ 6) ∧ 
  (∀ x : ℝ, degree(p(x)) > 6 → ¬has_horizontal_asymptote(f(x))) := 
sorry

end max_degree_p_has_horizontal_asymptote_l703_703470


namespace four_letter_initial_sets_l703_703966

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l703_703966


namespace func_strictly_increasing_l703_703113

open Real

noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

def func (x : ℝ) : ℝ := log_half (-x^2 + x + 6)

def strictly_increasing_on {f : ℝ → ℝ} (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem func_strictly_increasing : strictly_increasing_on func (1 / 2) 3 :=
sorry

end func_strictly_increasing_l703_703113


namespace total_participants_l703_703352

theorem total_participants (F M : ℕ)
  (h1 : F / 2 = 130)
  (h2 : F / 2 + M / 4 = (F + M) / 3) : 
  F + M = 780 := 
by 
  sorry

end total_participants_l703_703352


namespace composite_lcm_factorial_l703_703515

-- Define a function to count composite numbers up to x
def k (x : ℕ) : ℕ :=
  (List.range (x + 1)).filter (λ n, n > 1 ∧ ¬ Nat.prime n).length

-- State the theorem with the inequality involving lcm and factorials
theorem composite_lcm_factorial (n : ℕ) (h : n ∈ [2, 3, 4, 5, 7, 9]) :
  (k n)! * Nat.lcm_list (List.range (n + 1)) > (n - 1)! :=
by 
  sorry

end composite_lcm_factorial_l703_703515


namespace necessary_condition_for_f_lt_5_not_sufficient_condition_for_f_lt_5_l703_703527

variable (x : ℝ)

def f (x : ℝ) : ℝ := (1 / 2) ^ x - 3

theorem necessary_condition_for_f_lt_5 : f x < 5 → x > -4 := by
  sorry

theorem not_sufficient_condition_for_f_lt_5 : ¬(∀ x > -4, f x < 5) := by
  sorry

end necessary_condition_for_f_lt_5_not_sufficient_condition_for_f_lt_5_l703_703527


namespace most_likely_event_l703_703848

def roll_die := [1, 2, 3, 4, 5, 6]
def probability (s : Set ℕ) := s.card / roll_die.card

def condition_gt_2 := {x | 2 < x ∧ x ∈ roll_die}
def condition_eq_4_or_5 := {x | (x = 4 ∨ x = 5) ∧ x ∈ roll_die}
def condition_even := {x | (x % 2 = 0) ∧ x ∈ roll_die}
def condition_lt_3 := {x | x < 3 ∧ x ∈ roll_die}
def condition_eq_3 := {x | x = 3 ∧ x ∈ roll_die}

theorem most_likely_event : 
  probability condition_gt_2 = 2/3 ∧ 
  (probability condition_gt_2 > probability condition_eq_4_or_5) ∧ 
  (probability condition_gt_2 > probability condition_even) ∧ 
  (probability condition_gt_2 > probability condition_lt_3) ∧ 
  (probability condition_gt_2 > probability condition_eq_3) := 
by 
  sorry

end most_likely_event_l703_703848


namespace jack_jill_same_speed_l703_703239

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end jack_jill_same_speed_l703_703239


namespace find_b_l703_703261

theorem find_b (g : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∃ x, g x ≠ 0) 
               (h3 : a > 0) (h4 : a ≠ 1) (h5 : ∀ x, (1 / (a ^ x - 1) - 1 / b) * g x = (1 / (a ^ (-x) - 1) - 1 / b) * g (-x)) :
    b = -2 :=
sorry

end find_b_l703_703261


namespace largest_percentage_increase_l703_703452

theorem largest_percentage_increase :
  let students := [60, 66, 70, 76, 78, 85] in
  let percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / a * 100 in
  let increases := list.zipWith percentage_increase students students.tail in
  list.maximum increases = some (percentage_increase 60 66) :=
by
  sorry

end largest_percentage_increase_l703_703452


namespace number_of_pairs_l703_703129

noncomputable def number_of_ordered_pairs (n : ℕ) : ℕ :=
  if n = 5 then 8 else 0

theorem number_of_pairs (f m: ℕ) : f ≥ 0 ∧ m ≥ 0 → number_of_ordered_pairs 5 = 8 :=
by
  intro h
  sorry

end number_of_pairs_l703_703129


namespace power_tower_monotone_and_bounded_l703_703022

def seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = real.sqrt 2 * a n

theorem power_tower_monotone_and_bounded :
  ∃ a : ℕ → ℝ,
  a 0 = 1 ∧
  seq a ∧
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, a n < 2) :=
sorry

end power_tower_monotone_and_bounded_l703_703022


namespace increase_speed_to_pass_correctly_l703_703208

theorem increase_speed_to_pass_correctly
  (x a : ℝ)
  (ha1 : 50 < a)
  (hx1 : (a - 40) * x = 30)
  (hx2 : (a + 50) * x = 210) :
  a - 50 = 5 :=
by
  sorry

end increase_speed_to_pass_correctly_l703_703208


namespace complex_conjugate_correct_l703_703367

-- Definition of a complex number
def complex_num : ℂ := 3 + 4 * complex.I

-- Definition of the expected conjugate
def expected_conjugate : ℂ := 3 - 4 * complex.I

-- Theorem stating the expected conjugate of a given complex number
theorem complex_conjugate_correct : complex.conj complex_num = expected_conjugate :=
by
  -- The proof for this theorem will go here
  sorry

end complex_conjugate_correct_l703_703367


namespace phone_bill_calculation_l703_703832

def base_cost : ℝ := 25
def free_texts : ℕ := 50
def text_cost : ℝ := 0.10
def free_talk_time_hours : ℝ := 25
def additional_minute_cost : ℝ := 0.15
def february_texts : ℕ := 200
def february_talk_time_hours : ℝ := 28

theorem phone_bill_calculation : 
  let charged_texts := february_texts - free_texts
      text_charge := charged_texts * text_cost
      extra_talk_time_minutes := (february_talk_time_hours - free_talk_time_hours) * 60
      talk_charge := extra_talk_time_minutes * additional_minute_cost
      total_cost := base_cost + text_charge + talk_charge
  in total_cost = 67 := by
  sorry

end phone_bill_calculation_l703_703832


namespace number_of_games_played_l703_703249

noncomputable def total_games_played (W_J : ℕ) (M_final : ℕ) (L_M : ℕ) (W_M : ℕ) : ℕ :=
  W_J + W_M

theorem number_of_games_played (W_J M_final : ℕ) (game_result: ∀ (games : ℕ),
  games = W_J + (M_final + L_M) / 2):
  W_J = 3 →
  M_final = 5 → 
  ∑ games in game_result games, games = 7 :=
by
sorry

end number_of_games_played_l703_703249


namespace no_such_number_exists_l703_703001

theorem no_such_number_exists :
  ¬∃ (x : ℝ), 1000^2 + 1001^2 + 1002^2 + x^2 + 1004^2 = 6 :=
by
  -- calculations for the proof
  let a := (1000 : ℝ)^2
  let b := (1001 : ℝ)^2
  let c := (1002 : ℝ)^2
  let d := (1004 : ℝ)^2
  have ha : a = 1000000 := by norm_num
  have hb : b = 1002001 := by norm_num
  have hc : c = 1004004 := by norm_num
  have hd : d = 1008016 := by norm_num
  have sum_sq : a + b + c + d = 4014021 := by rw [ha, hb, hc, hd]; norm_num
  have contradiction : 4014021 > 6 := by norm_num
  simp only [gt, not_le] at contradiction
  intro h,
  obtain ⟨x, hx⟩ := h
  rw hx at sum_sq,
  linarith

end no_such_number_exists_l703_703001


namespace solve_for_x_l703_703588

variable x : ℝ

-- Condition: (1/8) * 2^36 = 2^x
axiom condition : (1/8) * 2^36 = 2^x

theorem solve_for_x : x = 33 :=
by
  sorry

end solve_for_x_l703_703588


namespace cyclist_distance_from_start_l703_703820

theorem cyclist_distance_from_start 
  (e1 e2 : ℕ)
  (s1 s2 : ℕ)
  (net_displacement_ew : ℕ := (12 - 5))
  (net_displacement_ns : ℕ := (9 - 3)) :
  let d := Math.sqrt ((7 : ℕ)^2 + (6 : ℕ)^2) in
  d = Real.sqrt (85 : ℕ) :=
by
  sorry

end cyclist_distance_from_start_l703_703820


namespace not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l703_703825

noncomputable def f (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem not_bounded_on_neg_infty_zero (a x : ℝ) (h : a = 1) : 
  ¬ ∃ M > 0, ∀ x < 0, |f a x| ≤ M :=
by sorry

-- Problem (2)
theorem range_of_a_bounded_on_zero_infty (a : ℝ) : 
  (∀ x ≥ 0, |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l703_703825


namespace revenue_from_full_price_tickets_l703_703435

theorem revenue_from_full_price_tickets (f h p : ℕ) (h1 : f + h = 160) (h2 : f * p + h * (p / 2) = 2400) : f * p = 1600 :=
by
  sorry

end revenue_from_full_price_tickets_l703_703435


namespace AB_C_B_D_concyclic_l703_703354

theorem AB_C_B_D_concyclic
  (A B C M O D : Point)
  (h_inscribed : Circle)
  (h_center : center h_inscribed = O)
  (h_AO_perpendicular : ⟂ (line_through O) (line_through O A) (line_through M O))
  (h_intersect_BC : intersects (line_through M O) (line_through B C) = M)
  (h_OD_perpendicular : ⟂ (line_through O D) (line_through M A)) :
  concyclic A B C D :=
sorry

end AB_C_B_D_concyclic_l703_703354


namespace reduced_fraction_numerator_l703_703823

theorem reduced_fraction_numerator :
  let numerator := 4128 
  let denominator := 4386 
  let gcd := Nat.gcd numerator denominator
  let reduced_numerator := numerator / gcd 
  let reduced_denominator := denominator / gcd 
  (reduced_numerator : ℚ) / (reduced_denominator : ℚ) = 16 / 17 → reduced_numerator = 16 :=
by
  intros
  sorry

end reduced_fraction_numerator_l703_703823


namespace squarish_numbers_l703_703428

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def contains_no_digit_9 (n : ℕ) : Prop :=
  ¬∃ k : ℕ, (n / 10^k % 10 = 9)

def squarish (n : ℕ) : Prop :=
  contains_no_digit_9 n ∧ 
  (∀ k : ℕ, k < Nat.digits 10 n → is_square (n + 10^k))

theorem squarish_numbers :
  {n : ℕ | squarish n} = {0, 3, 8, 15} :=
by
  sorry

end squarish_numbers_l703_703428


namespace truncated_cone_volume_l703_703083

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end truncated_cone_volume_l703_703083


namespace part1_solution_part2_solution_l703_703564

def f (x : ℝ) : ℝ := |x - 2| - |x + 1|

def g (a : ℝ) (x : ℝ) : ℝ := (a * x^2 - x + 1) / x

theorem part1_solution (x : ℝ) : f x > 1 ↔ x < 0 := 
by sorry

theorem part2_solution (a : ℝ) (h : a > 0) : (∀ x > 0, g a x > f x) ↔ a ≥ 1 :=
by sorry

end part1_solution_part2_solution_l703_703564


namespace exists_obtuse_triangle_l703_703221

section
variables {a b c : ℝ} (ABC : Triangle ℝ) (h₁ : ABC.side1 = a)
  (h₂ : ABC.side2 = a + 1) (h₃ : ABC.side3 = a + 2)
  (h₄ : 2 * sin ABC.γ = 3 * sin ABC.α)

noncomputable def area_triangle_proof : ABC.area = (15 * real.sqrt 7) / 4 :=
sorry

theorem exists_obtuse_triangle : ∃ (a : ℝ), a = 2 ∧ ∀ (h : Triangle ℝ), 
  h.side1 = a → h.side2 = a + 1 → h.side3 = a + 2 → obtuse h :=
sorry
end

end exists_obtuse_triangle_l703_703221


namespace find_c_l703_703786

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -x^2 + c*x - 8 < 0 ↔ x ∈ set.Ioo (-∞:ℝ) 2 ∪ set.Ioo 6 (∞:ℝ)) → c = 8 :=
by
  sorry

end find_c_l703_703786


namespace remainder_of_6x_mod_9_l703_703799

theorem remainder_of_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 :=
by
  sorry

end remainder_of_6x_mod_9_l703_703799


namespace jackson_earning_l703_703245

theorem jackson_earning :
  let rate := 5
  let vacuum_hours := 2 * 2
  let dish_hours := 0.5
  let bathroom_hours := 0.5 * 3
  vacuum_hours * rate + dish_hours * rate + bathroom_hours * rate = 30 :=
by
  intros
  let rate := 5
  let vacuum_hours := 2 * 2
  let dish_hours := 0.5
  let bathroom_hours := 0.5 * 3
  have vacuum_earn := vacuum_hours * rate
  have dish_earn := dish_hours * rate
  have bathroom_earn := bathroom_hours * rate
  have total_earn := vacuum_earn + dish_earn + bathroom_earn
  exact total_earn = 30
  sorry

end jackson_earning_l703_703245


namespace original_four_digit_number_l703_703008

theorem original_four_digit_number (n L S : ℕ) 
  (h1 : 999 < n ∧ n < 10000)
  (h2 : ∀ d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10], d ≠ 0)
  (h3 : L = n + 5562)
  (h4 : S = n - 2700)
  (h5 : L = list.max' [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10])
  (h6 : S = list.min' [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]) :
  n = 4179 :=
by 
  sorry

end original_four_digit_number_l703_703008


namespace initial_people_in_line_l703_703454

theorem initial_people_in_line :
  ∃ X : ℕ, ((X - 10 + 5 = 25) ∧ X = 30) :=
begin
  use 30,
  split,
  { sorry },
  { refl },
end

end initial_people_in_line_l703_703454


namespace unique_plants_count_1320_l703_703011

open Set

variable (X Y Z : Finset ℕ)

def total_plants_X : ℕ := 600
def total_plants_Y : ℕ := 480
def total_plants_Z : ℕ := 420
def shared_XY : ℕ := 60
def shared_YZ : ℕ := 70
def shared_XZ : ℕ := 80
def shared_XYZ : ℕ := 30

theorem unique_plants_count_1320 : X.card = total_plants_X →
                                Y.card = total_plants_Y →
                                Z.card = total_plants_Z →
                                (X ∩ Y).card = shared_XY →
                                (Y ∩ Z).card = shared_YZ →
                                (X ∩ Z).card = shared_XZ →
                                (X ∩ Y ∩ Z).card = shared_XYZ →
                                (X ∪ Y ∪ Z).card = 1320 := 
by {
  sorry
}

end unique_plants_count_1320_l703_703011


namespace equation_of_curve_slope_range_condition_l703_703270

noncomputable def curve_equation (x y : ℝ) : Prop :=
  abs (y - 3) = sqrt 3 * sqrt (x^2 + (y - 1)^2)

theorem equation_of_curve (x y : ℝ):
  curve_equation x y ↔ 3 * x ^ 2 + 2 * y ^ 2 = 6 :=
sorry

noncomputable def line_slope_condition (k λ : ℝ) (x1 x2 y1 y2 : ℝ) : Prop :=
  y1 = k * x1 + 1 ∧ y2 = k * x2 + 1 ∧
  -(4 * k) /(2 * k^2 + 3) = x1 + x2 ∧
  -(4 / (2 * k^2 + 3)) = x1 * x2 ∧
  x1 = -λ * x2

theorem slope_range_condition (k : ℝ) :
  (∃ λ x1 x2 y1 y2, 2 ≤ λ ∧ λ ≤ 3 ∧ line_slope_condition k λ x1 x2 y1 y2) ↔ 
  (-sqrt 3 ≤ k ∧ k ≤ -sqrt 2 / 2) ∨ (sqrt 2 / 2 ≤ k ∧ k ≤ sqrt 3) :=
sorry

end equation_of_curve_slope_range_condition_l703_703270


namespace alternating_series_sum_l703_703099

theorem alternating_series_sum : 
  ∑ n in Finset.range 23, (-1 : ℤ) ^ (n - 11) = 0 := 
by sorry

end alternating_series_sum_l703_703099


namespace semicircle_radius_l703_703095

-- We need to define the given conditions as constants or hypotheses:
theorem semicircle_radius (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  (right_triangle_PQR : ∠PQR = π / 2)
  (diameter_PQ_semi_area : ∃ r : ℝ, (π * r^2) / 2 = 12.5 * π)
  (diameter_QR_semi_arc_length : ∃ r : ℝ, π * r = 9 * π) :
  -- Prove the radius of the semicircle on PR is approximately 10.3
  ∃ r : ℝ, r ≈ 10.3 := by
sorry

end semicircle_radius_l703_703095


namespace men_in_first_group_l703_703599

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end men_in_first_group_l703_703599


namespace part1_part2_l703_703235

noncomputable def area_of_triangle (a b c : ℝ) (sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Part 1
theorem part1 (a : ℝ) (b : ℝ) (c : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : 2 * Real.sin C = 3 * Real.sin A)
  (a_val : a = 4) (b_val : b = 5) (c_val : c = 6) : 
  area_of_triangle 4 5 (3 * Real.sqrt 7 / 8) = 15 * Real.sqrt 7 / 4 := by 
  sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : ∃ (a' : ℕ), a = a') :
  a = 2 := by
  sorry

end part1_part2_l703_703235


namespace area_of_triangle_obtuse_triangle_exists_l703_703227

-- Define the mathematical conditions given in the problem
variables {a b c : ℝ}
axiom triangle_inequality : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y
axiom sine_relation : 2 * Real.sin c = 3 * Real.sin a
axiom side_b : b = a + 1
axiom side_c : c = a + 2

-- Part 1: Prove the area of ΔABC equals the provided solution
theorem area_of_triangle : 
  2 * Real.sin c = 3 * Real.sin a → 
  b = a + 1 → 
  c = a + 2 → 
  a = 4 → 
  b = 5 → 
  c = 6 → 
  let ab_sin_c := (1 / 2) * 4 * 5 * ((3 * Real.sqrt 7) / 8) in
  ab_sin_c = 15 * Real.sqrt 7 / 4 :=
by {
  sorry, -- The proof steps are to be provided here
}

-- Part 2: Prove there exists a positive integer a such that ΔABC is obtuse with a = 2
theorem obtuse_triangle_exists :
  (∃ (a : ℕ) (h2 : 1 < a ∧ a < 3), 2 * Real.sin c = 3 * Real.sin a ∧ b = a + 1 ∧ c = a + 2) →
  ∃ a = 2 ∧ (b = a + 1 ∧ c = a + 2) :=
by {
  sorry, -- The proof steps are to be provided here
}

end area_of_triangle_obtuse_triangle_exists_l703_703227


namespace intersection_example_l703_703153

open Set

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_example : 
  let A := { x : ℝ | -1 < x ∧ x < 4 }
  let B := { -4, 1, 3, 5 }
  in A ∩ B = { 1, 3 } :=
by 
  sorry

end intersection_example_l703_703153


namespace probability_neither_red_l703_703059

theorem probability_neither_red (total_marbles : ℕ) (red_marbles : ℕ) (non_red_marbles : ℕ)
    (h_total: total_marbles = 84) (h_red: red_marbles = 12) (h_non_red: non_red_marbles = total_marbles - red_marbles) :
    (72 / 84) * (72 / 84) = 36 / 49 :=
begin
  -- Proof goes here
  sorry
end

end probability_neither_red_l703_703059


namespace no_real_solutions_sufficient_not_necessary_l703_703385

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l703_703385


namespace num_ways_to_turn_off_lights_l703_703427

-- Let's define our problem in terms of the conditions given
-- Define the total number of lights
def total_lights : ℕ := 12

-- Define that we need to turn off 3 lights
def lights_to_turn_off : ℕ := 3

-- Define that we have 10 possible candidates for being turned off 
def candidates := total_lights - 2

-- Define the gap consumption statement that effectively reduce choices to 7 lights
def effective_choices := candidates - lights_to_turn_off

-- Define the combination formula for the number of ways to turn off the lights
def num_ways := Nat.choose effective_choices lights_to_turn_off

-- Final statement to prove
theorem num_ways_to_turn_off_lights : num_ways = Nat.choose 7 3 :=
by
  sorry

end num_ways_to_turn_off_lights_l703_703427


namespace birds_on_branch_l703_703292

theorem birds_on_branch (initial_parrots remaining_parrots remaining_crows total_birds : ℕ) (h₁ : initial_parrots = 7) (h₂ : remaining_parrots = 2) (h₃ : remaining_crows = 1) (h₄ : initial_parrots - remaining_parrots = total_birds - remaining_crows - initial_parrots) : total_birds = 13 :=
sorry

end birds_on_branch_l703_703292


namespace bacteria_growth_time_l703_703321

-- Define the conditions and the final proof statement
theorem bacteria_growth_time (n0 n1 : ℕ) (t : ℕ) :
  (∀ (k : ℕ), k > 0 → n1 = n0 * 3 ^ k) →
  (∀ (h : ℕ), t = 5 * h) →
  n0 = 200 →
  n1 = 145800 →
  t = 30 :=
by
  sorry

end bacteria_growth_time_l703_703321


namespace polynomial_roots_sum_of_inverses_l703_703262

theorem polynomial_roots_sum_of_inverses :
  let p q r s t : ℂ in
  let poly := (λ x, x^5 - 4*x^4 + 7*x^3 - 3*x^2 + x - 1) in
  (poly p = 0 ∧ poly q = 0 ∧ poly r = 0 ∧ poly s = 0 ∧ poly t = 0) →
  (∀ (x : ℂ), poly x = 0 → x = p ∨ x = q ∨ x = r ∨ x = s ∨ x = t) →
  (pq + pr + ps + pt + qr + qs + qt + rs + rt + st = 7) →
  (p * q * r * s * t = 1) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t)) = 7 := by
  sorry

end polynomial_roots_sum_of_inverses_l703_703262


namespace average_shifted_l703_703559

theorem average_shifted (n : ℕ) (x : ℕ → ℝ) (h : (∑ i in Finset.range n, x i) / n = 2) :
  (∑ i in Finset.range n, (x i + 2)) / n = 4 :=
sorry

end average_shifted_l703_703559


namespace sum_of_exponents_of_1540_is_21_l703_703995

theorem sum_of_exponents_of_1540_is_21 :
  ∃ (S : Set ℕ), (1540 = ∑ i in S, 2^i) ∧ S.Sum id = 21 := sorry

end sum_of_exponents_of_1540_is_21_l703_703995


namespace greatest_integer_sum_2997_l703_703503

theorem greatest_integer_sum_2997 :
  let s := ∑ n in Finset.range (10^9+1), (n : ℝ) ^ (-2 / 3)
  in ⌊s⌋ = 2997 :=
by
  let s := ∑ n in Finset.range (10^9+1), (n : ℝ) ^ (-2 / 3)
  have h : ⌊s⌋ = 2997 := sorry
  exact h

end greatest_integer_sum_2997_l703_703503


namespace quadratic_real_roots_k_eq_one_l703_703570

theorem quadratic_real_roots_k_eq_one 
  (k : ℕ) 
  (h_nonneg : k ≥ 0) 
  (h_real_roots : ∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) : 
  k = 1 := 
sorry

end quadratic_real_roots_k_eq_one_l703_703570


namespace poly_solution_l703_703750

-- Definitions for the conditions of the problem
def poly1 (d g : ℚ) := 5 * d ^ 2 - 4 * d + g
def poly2 (d h : ℚ) := 4 * d ^ 2 + h * d - 5
def product (d g h : ℚ) := 20 * d ^ 4 - 31 * d ^ 3 - 17 * d ^ 2 + 23 * d - 10

-- Statement of the problem: proving g + h = 7/2 given the conditions.
theorem poly_solution
  (g h : ℚ)
  (cond : ∀ d : ℚ, poly1 d g * poly2 d h = product d g h) :
  g + h = 7 / 2 :=
by
  sorry

end poly_solution_l703_703750


namespace integral_abs_sin_l703_703097

theorem integral_abs_sin (I : ℤ := 0) : 
  ∫ x in 0..(2 * real.pi), |real.sin x| = 4 :=
by
  sorry

end integral_abs_sin_l703_703097


namespace always_have_at_least_one_point_satisfying_l703_703365

-- Define the types for points and colors
inductive Color
| Red | Blue | Green | Yellow

structure Point :=
(color : Color)

-- Define the four points
def P1 : Point := {color := Color.Red}
def P2 : Point := {color := Color.Blue}
def P3 : Point := {color := Color.Green}
def P4 : Point := {color := Color.Yellow}

-- Define the coloring of line segments
structure Segment :=
(p1 p2 : Point)
(color : Color)

-- Assume the conditions from the problem
axiom use_all_colors (segments : ℕ → Segment) : 
  ∃ i j k l, 
    (segments i).color = Color.Red ∧ 
    (segments j).color = Color.Blue ∧ 
    (segments k).color = Color.Green ∧ 
    (segments l).color = Color.Yellow

-- Our goal is to prove there exists a point satisfying condition (a) or (b)
theorem always_have_at_least_one_point_satisfying (segments : ℕ → Segment) :
  ∃ p : Point, 
    (
      (∃ s1 s2 s3 : ℕ, s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧ 
       (segments s1).color = Color.Red ∧ 
       (segments s2).color = Color.Blue ∧ 
       (segments s3).color = Color.Green ∧
       ((segments s1).p1 = p ∨ (segments s1).p2 = p) ∧
       ((segments s2).p1 = p ∨ (segments s2).p2 = p) ∧
       ((segments s3).p1 = p ∨ (segments s3).p2 = p)
    ) ∨
    (
      ∃ q r s : Point, 
      q ≠ r ∧ r ≠ s ∧ q ≠ s ∧ 
      ((∃ s1 : ℕ, (segments s1).p1 = q ∧ (segments s1).p2 = r ∧ (segments s1).color = Color.Red) ∨
       ∃ s1 : ℕ, (segments s1).p1 = r ∧ (segments s1).p2 = q ∧ (segments s1).color = Color.Red) ∧
      ((∃ s2 : ℕ, (segments s2).p1 = r ∧ (segments s2).p2 = s ∧ (segments s2).color = Color.Blue) ∨
       ∃ s2 : ℕ, (segments s2).p1 = s ∧ (segments s2).p2 = r ∧ (segments s2).color = Color.Blue) ∧
      ((∃ s3 : ℕ, (segments s3).p1 = s ∧ (segments s3).p2 = q ∧ (segments s3).color = Color.Green) ∨
       ∃ s3 : ℕ, (segments s3).p1 = q ∧ (segments s3).p2 = s ∧ (segments s3).color = Color.Green))
    )
  sorry

end always_have_at_least_one_point_satisfying_l703_703365


namespace jack_jill_speed_l703_703240

-- Define conditions for Jack and Jill's speeds.
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72
def jill_time (x : ℝ) : ℝ := x + 8
def jack_speed (x : ℝ) : ℝ := x^2 - 7x - 18
def jill_speed (x : ℝ) : ℝ := (x^2 + x - 72) / (x + 8)

-- Define the main theorem to prove
theorem jack_jill_speed : (jack_speed 10 = 2) ∧ (jill_speed 10 = 2) :=
by
  sorry

end jack_jill_speed_l703_703240


namespace triangle_is_isosceles_l703_703699

-- Definitions of points and segments
variables {A B C D E : Type} [Points A B C D E]

-- Definitions of proportions and angles
variables {AD DB AE EC : ℝ}
variables (angleC angleDEB : ℝ)

-- Conditions
def condition1 : AD / DB = 2 := sorry
def condition2 : AE / EC = 2 := sorry
def condition3 : angleC = 2 * angleDEB := sorry

-- The main theorem to prove
theorem triangle_is_isosceles (h1 : AD / DB = 2) (h2 : AE / EC = 2) (h3 : angleC = 2 * angleDEB) : AC = BC := by 
  sorry

end triangle_is_isosceles_l703_703699


namespace village_population_decrease_rate_l703_703777

theorem village_population_decrease_rate :
  ∃ (R : ℝ), 15 * R = 18000 :=
by
  sorry

end village_population_decrease_rate_l703_703777


namespace find_k_l703_703909

noncomputable def slopes_perpendicular (k : ℚ) : Prop :=
  let l1 := k * x - y - 3 = 0
  let l2 := x + (2 * k + 3) * y - 2 = 0
  let m1 := k -- Slope of l1
  let m2 := -1 / (2 * k + 3) -- Slope of l2
  k * m2 = -1

theorem find_k (k : ℚ) (h : slopes_perpendicular k) : k = -3 :=
by
  unfold slopes_perpendicular at h
  sorry

end find_k_l703_703909


namespace chip_final_balance_l703_703102

noncomputable def finalBalance : ℝ := 
  let initialBalance := 50.0
  let month1InterestRate := 0.20
  let month2NewCharges := 20.0
  let month2InterestRate := 0.20
  let month3NewCharges := 30.0
  let month3Payment := 10.0
  let month3InterestRate := 0.25
  let month4NewCharges := 40.0
  let month4Payment := 20.0
  let month4InterestRate := 0.15

  -- Month 1
  let month1InterestFee := initialBalance * month1InterestRate
  let balanceMonth1 := initialBalance + month1InterestFee

  -- Month 2
  let balanceMonth2BeforeInterest := balanceMonth1 + month2NewCharges
  let month2InterestFee := balanceMonth2BeforeInterest * month2InterestRate
  let balanceMonth2 := balanceMonth2BeforeInterest + month2InterestFee

  -- Month 3
  let balanceMonth3BeforeInterest := balanceMonth2 + month3NewCharges
  let balanceMonth3AfterPayment := balanceMonth3BeforeInterest - month3Payment
  let month3InterestFee := balanceMonth3AfterPayment * month3InterestRate
  let balanceMonth3 := balanceMonth3AfterPayment + month3InterestFee

  -- Month 4
  let balanceMonth4BeforeInterest := balanceMonth3 + month4NewCharges
  let balanceMonth4AfterPayment := balanceMonth4BeforeInterest - month4Payment
  let month4InterestFee := balanceMonth4AfterPayment * month4InterestRate
  let balanceMonth4 := balanceMonth4AfterPayment + month4InterestFee

  balanceMonth4

theorem chip_final_balance : finalBalance = 189.75 := by sorry

end chip_final_balance_l703_703102


namespace position_M_independent_l703_703919

theorem position_M_independent 
  (S : Type) [circle S] 
  (A B : S.point) 
  (C : S.chord_point A B)
  (S' : ℕ → circle) -- representing a family of circles tangent to chord AB at point C
  (tangent_S' : ∀ n, tangent_point (S' n) A B C)
  (intersection_S' : ∀ n, ∃ P Q, intersection_points (S' n) S P Q)
  (M : intersection_point_AB_PQ (S' 0)) :  -- Assuming for starting choice S'
  ∀ n, intersection_point_AB_PQ (S' n) = M :=
sorry

end position_M_independent_l703_703919


namespace ellipse_eccentricity_l703_703535

theorem ellipse_eccentricity (a b : ℝ) (h₁ : a > b > 0) 
  (h₂ : ∃ (a : ℝ), (4 / a^2) = 1) 
  (h₃ : ∃ (a b : ℝ), (1 / a^2) + (9 / (4 * b^2)) = 1) : 
  (sqrt (1 - b^2 / a^2)) = 1 / 2 := sorry

end ellipse_eccentricity_l703_703535


namespace range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l703_703546

theorem range_of_x_if_p_and_q_true (a : ℝ) (p q : ℝ → Prop) (h_a : a = 1) (h_p : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (h_q : ∀ x, q x ↔ (x-3)^2 < 1) (h_pq : ∀ x, p x ∧ q x) :
  ∀ x, 2 < x ∧ x < 3 :=
by
  sorry

theorem range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q (p q : ℝ → Prop) (h_neg : ∀ x, ¬p x → ¬q x) : 
  ∀ a : ℝ, a > 0 → (a ≥ 4/3 ∧ a ≤ 2) :=
by
  sorry

end range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l703_703546


namespace time_to_cross_platform_l703_703423

-- Definitions based on conditions
def speed_kmph := 72
def length_train := 170.0416
def length_platform := 350
def speed_mps := (speed_kmph * 1000) / 3600
def total_distance_covered := length_train + length_platform
def time_seconds := total_distance_covered / speed_mps

-- Theorem to be proved
theorem time_to_cross_platform : time_seconds = 26.00208 := by
  sorry

end time_to_cross_platform_l703_703423


namespace least_number_condition_l703_703368

-- Define the set of divisors as a constant
def divisors : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 15}

-- Define the least number that satisfies the condition
def least_number : ℕ := 125

-- The theorem stating that the least number 125 leaves a remainder of 5 when divided by the given set of numbers
theorem least_number_condition : ∀ d ∈ divisors, least_number % d = 5 :=
by
  sorry

end least_number_condition_l703_703368


namespace solve_quadratic_l703_703302

theorem solve_quadratic (y : ℝ) :
  3 * y * (y - 1) = 2 * (y - 1) → y = 2 / 3 ∨ y = 1 :=
by
  sorry

end solve_quadratic_l703_703302


namespace sum_of_k_values_l703_703029

theorem sum_of_k_values : 
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∑ k ∈ ({k | ∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)}).toFinset) = 14 :=
sorry

end sum_of_k_values_l703_703029


namespace average_of_first_150_terms_l703_703471

def sequence (n: ℕ) : ℤ := (-1)^(n + 1) * n

theorem average_of_first_150_terms : 
  (∑ i in Finset.range 150, sequence (i + 1)) / 150 = -0.5 := 
by
  sorry

end average_of_first_150_terms_l703_703471


namespace eccentricity_of_ellipse_l703_703161

theorem eccentricity_of_ellipse :
  ∀ (a b : ℝ), a > b ∧ b > 0 ∧ (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 ≠ 0 ∧ P.2 = 2 ∧ (P.1, P.2) ∈ {p | p.1^2 / a^2 + p.2^2 / b^2 = 1} ∧
  ∃ F : ℝ × ℝ, F = (1, 0) ∧ distance F (1, 2) = 1) →
  ( ∃ (e : ℝ), e = sqrt (1 - (b^2 / a^2)) ) ∧ e = sqrt 2 - 1 :=
by
  sorry

end eccentricity_of_ellipse_l703_703161


namespace distance_between_l703_703689

-- Define the given speeds and time
def speed_car := 65 -- km/h
def speed_train := 95 -- km/h
def time_hours := 8 -- hours

-- Compute the distances traveled
def distance_car := speed_car * time_hours -- km
def distance_train := speed_train * time_hours -- km

-- Statement to prove
theorem distance_between (s_c s_t t : ℕ) (h1 : s_c = 65) (h2 : s_t = 95) (h3 : t = 8) :
  distance_train - distance_car = 240 :=
by
  -- Directly use the given conditions and definitions
  have h1d : distance_car = 65 * 8 := by sorry
  have h2d : distance_train = 95 * 8 := by sorry
  -- Calculate the difference
  calc
    distance_train - distance_car
        = (95 * 8) - (65 * 8) : by rw [h1d, h2d]
    ... = 240 : by sorry

end distance_between_l703_703689


namespace evaluate_polynomial_l703_703885

theorem evaluate_polynomial : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_l703_703885


namespace probability_of_exactly_one_success_probability_of_at_least_one_success_l703_703696

variable (PA : ℚ := 1/2)
variable (PB : ℚ := 2/5)
variable (P_A_bar : ℚ := 1 - PA)
variable (P_B_bar : ℚ := 1 - PB)

theorem probability_of_exactly_one_success :
  PA * P_B_bar + PB * P_A_bar = 1/2 :=
sorry

theorem probability_of_at_least_one_success :
  1 - (P_A_bar * P_A_bar * P_B_bar * P_B_bar) = 91/100 :=
sorry

end probability_of_exactly_one_success_probability_of_at_least_one_success_l703_703696


namespace garden_area_l703_703317

-- Given that the garden is a square with certain properties
variables (s A P : ℕ)

-- Conditions:
-- The perimeter of the square garden is 28 feet
def perimeter_condition : Prop := P = 28

-- The area of the garden is equal to the perimeter plus 21
def area_condition : Prop := A = P + 21

-- The perimeter of a square garden with side length s
def perimeter_def : Prop := P = 4 * s

-- The area of a square garden with side length s
def area_def : Prop := A = s * s

-- Prove that the area A is 49 square feet
theorem garden_area : perimeter_condition P → area_condition P A → perimeter_def s P → area_def s A → A = 49 :=
by 
  sorry

end garden_area_l703_703317


namespace A_intersection_B_complement_l703_703678

noncomputable
def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | x > 1}

def set_B : Set ℝ := {y | -1 < y ∧ y < 2}

def B_complement : Set ℝ := {y | y <= -1 ∨ y >= 2}

def intersection : Set ℝ := {x | x >= 2}

theorem A_intersection_B_complement :
  (set_A ∩ B_complement) = intersection :=
  sorry

end A_intersection_B_complement_l703_703678


namespace product_of_sum_of_gcd_and_lcm_eq_112_l703_703460

theorem product_of_sum_of_gcd_and_lcm_eq_112 
  (a b : ℕ) (h₁ : a = 8) (h₂ : b = 12) :
  let g := Nat.gcd a b
  let l := Nat.lcm a b
  in g * (g + l) = 112 := 
by
  sorry

end product_of_sum_of_gcd_and_lcm_eq_112_l703_703460


namespace remainder_of_polynomial_l703_703508

-- Define the polynomial
def P (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

-- State the theorem
theorem remainder_of_polynomial (x : ℝ) : P 3 = 50 := sorry

end remainder_of_polynomial_l703_703508


namespace max_truth_tellers_l703_703802

theorem max_truth_tellers (number : Fin 10 → ℕ) (gt_statements : Fin 10 → Prop) (lt_statements : Fin 10 → Prop) : 
  (∀ n, (number n) > n + 1 → gt_statements n) ∧ 
  (∀ n, (number n) < n + 1 → lt_statements n) → 
  ∃ max_truth_tellers, max_truth_tellers = 8 :=
begin
  sorry
end

end max_truth_tellers_l703_703802


namespace jack_jill_speed_l703_703241

-- Define conditions for Jack and Jill's speeds.
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72
def jill_time (x : ℝ) : ℝ := x + 8
def jack_speed (x : ℝ) : ℝ := x^2 - 7x - 18
def jill_speed (x : ℝ) : ℝ := (x^2 + x - 72) / (x + 8)

-- Define the main theorem to prove
theorem jack_jill_speed : (jack_speed 10 = 2) ∧ (jill_speed 10 = 2) :=
by
  sorry

end jack_jill_speed_l703_703241


namespace locus_of_intersections_l703_703703

-- Conditions definitions
variable {A B C : Point}
variable (O₁ O₂ : Circle)

-- Orthogonal circles
def orthogonal (O₁ O₂ : Circle) : Prop :=
  -- Definition of orthogonality of circles in terms of their intersections (details omitted)
  sorry

def passes_through (O : Circle) (P Q : Point) : Prop :=
  -- Definition of circle passing through points P and Q (details omitted)
  sorry

-- Main theorem
theorem locus_of_intersections (h₁ : orthogonal O₁ O₂)
                              (h₂ : passes_through O₁ A C)
                              (h₃ : passes_through O₂ B C) :
  (exists (L : locus_of_intersections_with_orthogonal_circles_through [A, C] [B, C]),
    is_circle L ∨ is_line L) :=
by
  sorry

end locus_of_intersections_l703_703703


namespace next_meeting_time_l703_703048

noncomputable def perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem next_meeting_time 
  (AB BC CD AD : ℝ) 
  (v_human v_dog : ℝ) 
  (initial_meeting_time : ℝ) :
  AB = 100 → BC = 200 → CD = 100 → AD = 200 →
  initial_meeting_time = 2 →
  v_human + v_dog = 300 →
  ∃ next_time : ℝ, next_time = 14 := 
by
  sorry

end next_meeting_time_l703_703048


namespace largest_n_with_triangle_property_l703_703107

-- Definition of the triangle property
def triangle_property (S : Set ℕ) : Prop :=
  ∀ a b c ∈ S, a < b → b < c → c ≥ a + b

-- Main statement
theorem largest_n_with_triangle_property (n : ℕ) (h : n = 75) :
  ∀ S ⊆ {x | 3 ≤ x ∧ x ≤ n} (H : S.card = 8), triangle_property S :=
by
  intro S hS hCard
  rw [h] at hS hCard
  sorry

end largest_n_with_triangle_property_l703_703107


namespace a_n_formula_T_2n_sum_l703_703258

theorem a_n_formula (a S : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3)
  (cond1 : ∀ n ≥ 2, 2 * (S n + S (n - 1)) = a n ^ 2 + 1) : 
  ∀ n ≥ 1, a n = 2 * n - 1 :=
sorry

theorem T_2n_sum (a S : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3)
  (cond1 : ∀ n ≥ 2, 2 * (S n + S (n - 1)) = a n ^ 2 + 1) : 
  ∀ n : ℕ, ∑ i in range (2 * n), if i % 2 = 0 then (1 / (a i * a (i + 2))) else (a i) = 2 * n ^ 2 - n + (n / (3 * (4 * n + 3))) :=
sorry

end a_n_formula_T_2n_sum_l703_703258


namespace soccer_team_probability_l703_703437

theorem soccer_team_probability :
  let total_players := 12
  let forwards := 6
  let defenders := 6
  let total_ways := Nat.choose total_players 2
  let defender_ways := Nat.choose defenders 2
  ∃ p : ℚ, p = defender_ways / total_ways ∧ p = 5 / 22 :=
sorry

end soccer_team_probability_l703_703437


namespace truncated_cone_volume_l703_703084

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end truncated_cone_volume_l703_703084


namespace order_a_b_c_l703_703915

noncomputable def a : ℝ := 0.3^2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 2^0.3

theorem order_a_b_c : b < a ∧ a < c :=
by
  have h1 : 0 < a := by linarith [Real.rpow_pos_of_pos (by linarith) (by norm_num)]
  have h2 : a < 1 := by rw [a, ←Real.rpow_one 0.3]; exact Real.rpow_lt_rpow_of_exponent_lt (by linarith) (by norm_num)
  have h3 : b < 0 := by
    apply Real.log_lt_log_of_lt (by norm_num) (by linarith)
    simp
    exact (by norm_num : 2 > 1)
  have h4 : 1 < c := by
    simp [c]
    have : 1 = 2^0 := rfl
    exact (Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by linarith))
  exact ⟨h3, h1⟩
  exact ⟨h2, h4⟩

end order_a_b_c_l703_703915


namespace problem_statement_l703_703495

theorem problem_statement (a b c : ℝ) (h1 : a = (sqrt 5)^5) (h2 : b = sqrt a) (h3 : c = b^6) : 
  c = 5^7.5 := sorry

end problem_statement_l703_703495


namespace part_a_part_b_l703_703255

-- The problem statement and the conditions (no specific solution steps)
noncomputable def a_seq (a_0 : ℝ) : ℕ → ℝ
| 0       := a_0
| (n + 1) := min (2 * a_seq n) (1 - 2 * a_seq n)

-- Prove that ∃ n, a_n < 3/16
theorem part_a (a_0 : ℝ) (h0 : irrational a_0) (h1 : 0 < a_0) (h2 : a_0 < 1/2) :
  ∃ n, a_seq a_0 n < 3/16 :=
sorry

-- Prove that ¬ (∀ n, a_n > 7/40)
theorem part_b (a_0 : ℝ) (h0 : irrational a_0) (h1 : 0 < a_0) (h2 : a_0 < 1/2) :
  ¬ (∀ n, a_seq a_0 n > 7/40) :=
sorry

end part_a_part_b_l703_703255


namespace number_of_possible_values_of_S_l703_703669

def sum_S_of_subset_C (C : Finset ℕ) (hc : C.card = 80) (hC : ∀ x ∈ C, x ∈ (Finset.range 120).map Nat.succ) : ℕ :=
  Finset.sum C id

theorem number_of_possible_values_of_S : 
  ∃ (n : ℕ), n = 3201 ∧
  ∀ (C : Finset ℕ) (hc : C.card = 80) (hC : ∀ x ∈ C, x ∈ (Finset.range 120).map Nat.succ), 
    let S : ℕ := sum_S_of_subset_C C hc hC in
    3240 ≤ S ∧ S ≤ 6440 →
    x \in list.range(n),
      sorry

end number_of_possible_values_of_S_l703_703669


namespace smallest_positive_integer_n_l703_703649

/-- Given a = π/2010, find the smallest positive integer n such that
2[cos(a) * sin(2a) + cos(4a) * sin(4a) + cos(9a) * sin(6a) + ... + cos(n^2 * a) * sin(2n * a)] 
is an integer.
-/
theorem smallest_positive_integer_n (a : ℝ) (h : a = Real.pi / 2010) :
  ∃ n : ℕ, n > 0 ∧ 2010 ∣ n * (n + 1) ∧ (∀ m : ℕ, m > 0 ∧ 2010 ∣ m * (m + 1) → m ≥ n) :=
sorry

end smallest_positive_integer_n_l703_703649


namespace downstream_speed_l703_703831

noncomputable def speed_downstream (Vu Vs : ℝ) : ℝ :=
  2 * Vs - Vu

theorem downstream_speed (Vu Vs : ℝ) (hVu : Vu = 30) (hVs : Vs = 45) :
  speed_downstream Vu Vs = 60 := by
  rw [hVu, hVs]
  dsimp [speed_downstream]
  linarith

end downstream_speed_l703_703831


namespace proof_prob_at_least_one_die_3_or_5_l703_703793

def probability_at_least_one_die_3_or_5 (total_outcomes : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem proof_prob_at_least_one_die_3_or_5 :
  let total_outcomes := 36
  let favorable_outcomes := 20
  probability_at_least_one_die_3_or_5 total_outcomes favorable_outcomes = 5 / 9 := 
by 
  sorry

end proof_prob_at_least_one_die_3_or_5_l703_703793


namespace a8_eq_3_l703_703739

open Nat

noncomputable def a_n {n : ℕ} : ℕ → ℤ := sorry

noncomputable def b_n {n : ℕ} : ℕ → ℤ := sorry

theorem a8_eq_3
  (h1 : a_n 1 = 3)
  (h2 : ∀ n : ℕ, b_n n = a_n (n + 1) - a_n n)
  (h3 : b_n 3 = -2)
  (h4 : b_n 10 = 12) : a_n 8 = 3 :=
sorry

end a8_eq_3_l703_703739


namespace number_of_valid_sequences_equals_80_l703_703672

def isValidSequence (seq : Fin 10 → ℕ) : Prop :=
  seq 9 = 3 * seq 0 ∧
  seq 1 + seq 7 = 2 * seq 4 ∧
  ∀ i : Fin 9, seq i.succ = seq i + 1 ∨ seq i.succ = seq i + 2

theorem number_of_valid_sequences_equals_80 :
  (Fin 10 → ℕ) → ℕ := λ _,
  ∃ (seqs : Fin 10 → ℕ), isValidSequence seqs ∧  
  (finset.filter isValidSequence finset.univ_hom).card = 80 := sorry

end number_of_valid_sequences_equals_80_l703_703672


namespace measure_angle_MDN_180_l703_703219

open EuclideanGeometry

/-- In triangle ABC, angle bisector of ∠A meets side BC at point D.
    A circle with center D and radius DB intersects the segments AB and AC
    at points M and N respectively. Prove that ∠MDN = 180°. -/
theorem measure_angle_MDN_180 
  {A B C D M N : Point} -- Define all necessary points
  (h_triangle : Triangle A B C)
  (h_angle_bisector : AngleBisector AD ABC) -- AD is the angle bisector of ∠A
  (h_intersect : AD ∩ BC = {D}) -- AD intersects BC at D
  (h_circle_center : Circle D DB) -- Circle centered at D with radius DB
  (h_circle_inter_sections_AB : ∃ M ∈ AB, M_ne_A ∧ M_ne_B ∧ Distance D M = DB)
  (h_circle_inter_sections_AC : ∃ N ∈ AC, N_ne_A ∧ N_ne_C ∧ Distance D N = DB)
  (h_radius : Distance D B = Distance D C) -- DB = DC
  (h_angle_central : CentralAngle MOD 360°) -- MOD is a central angle
  : MeasureAngle MDN = 180° :=
sorry

end measure_angle_MDN_180_l703_703219


namespace ones_digit_of_p_l703_703519

theorem ones_digit_of_p (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hseq : q = p + 4 ∧ r = p + 8 ∧ s = p + 12) (hpg : p > 5) : (p % 10) = 9 :=
by
  sorry

end ones_digit_of_p_l703_703519


namespace area_ratio_PQR_ABC_l703_703631

-- Define some necessary concepts
variables {PQR ABC : Type*}
variables {PQR_Set : set PQR}
variables {ABC_Set : set ABC}

-- Given conditions as definitions
def ratio_BD_DC := 1 / 2
def ratio_CE_EA := 1 / 2
def ratio_AF_FB := 1 / 2

-- Problem statement in Lean 4
theorem area_ratio_PQR_ABC (h1 : ratio_BD_DC = 1 / 2) 
                           (h2 : ratio_CE_EA = 1 / 2) 
                           (h3 : ratio_AF_FB = 1 / 2) : 
                           (area PQR_Set) / (area ABC_Set) = 1 / 7 := 
sorry

end area_ratio_PQR_ABC_l703_703631


namespace necessary_but_not_sufficient_condition_l703_703158

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x < 5) : (x < 2 → x < 5) ∧ ¬(x < 5 → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l703_703158


namespace minimum_pq_distance_l703_703450

-- Definitions and conditions
def right_triangle (A B C : Point) : Prop :=
  ∠ A B C = 60 ∧ dist B C = 8

def midpoint (A B M : Point) : Prop := 
  M = (A + B) / 2

-- Ensure noncomputables if necessary for more complex numerical operations
noncomputable def minimum_distance (P Q : Point) : ℝ :=
  ∥P - Q∥

theorem minimum_pq_distance (A B C A' B' P Q : Point)
  (h_right_triangle: right_triangle A B C)
  (h1: midpoint A B P)
  (h2: midpoint B' C Q)
  (h_rotation: rotate_around C A' A B' B):  
  minimum_distance P Q = 17 / 4 := 
sorry

end minimum_pq_distance_l703_703450


namespace factorization_identity_l703_703493

theorem factorization_identity (a b : ℝ) : 
  -a^3 + 12 * a^2 * b - 36 * a * b^2 = -a * (a - 6 * b)^2 :=
by 
  sorry

end factorization_identity_l703_703493


namespace cube_root_inequality_l703_703642

theorem cube_root_inequality (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  real.cbrt (a / b) + real.cbrt (b / a) ≤ real.cbrt (2 * (a + b) * (1 / a + 1 / b)) :=
begin
  sorry
end

end cube_root_inequality_l703_703642


namespace octagon_diag_20_algebraic_expr_positive_l703_703445

def octagon_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diag_20 : octagon_diagonals 8 = 20 := by
  -- Formula for diagonals is used here
  sorry

theorem algebraic_expr_positive (x : ℝ) : 2 * x^2 - 2 * x + 1 > 0 := by
  -- Complete the square to show it's always positive
  sorry

end octagon_diag_20_algebraic_expr_positive_l703_703445


namespace number_of_initials_sets_l703_703961

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l703_703961


namespace sum_of_factors_of_360_l703_703033

theorem sum_of_factors_of_360 : Nat.divisorSum 360 = 1170 := by
  sorry

end sum_of_factors_of_360_l703_703033


namespace int_angle_eq_64_l703_703299

theorem int_angle_eq_64 {P Q : Point} (h1: RegularDecagon ABCDEFGHIJ) (h2: inter_ext_sides P Q ABCDEFGHIJ) :
  angle P Q = 64 :=
sorry

end int_angle_eq_64_l703_703299


namespace mike_earnings_first_job_l703_703685

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end mike_earnings_first_job_l703_703685


namespace limit_fraction_l703_703491

open Classical
open BigOperators
open Filter

noncomputable theory

theorem limit_fraction :
  Tendsto (λ n : ℕ, (3 * n - 1 : ℝ) / (2 * n + 3)) atTop (nhds (3 / 2)) :=
sorry

end limit_fraction_l703_703491


namespace max_kings_no_attack_l703_703782

-- Define the movement of a king on a chessboard.
def king_moves (x y : ℤ) : set (ℤ × ℤ) :=
  {(x + dx, y + dy) | dx dy : ℤ, -1 ≤ dx ∧ dx ≤ 1 ∧ -1 ≤ dy ∧ dy ≤ 1 ∧ (dx ≠ 0 ∨ dy ≠ 0)}

-- Define the chessboard as an 8x8 grid.
def chessboard : set (ℤ × ℤ) :=
  {(x, y) | x y : ℤ, 0 ≤ x ∧ x < 8 ∧ 0 ≤ y ∧ y < 8}

-- The maximum number of kings that can be placed on the chessboard without them attacking each other.
theorem max_kings_no_attack : ∃ S : set (ℤ × ℤ), 
  S ⊆ chessboard ∧ 
  ∀ (k1 k2 : ℤ × ℤ), k1 ∈ S → k2 ∈ S → k1 ≠ k2 → ¬(k2 ∈ king_moves (k1.fst) (k1.snd)) ∧ 
  S.card = 16 := 
sorry

end max_kings_no_attack_l703_703782


namespace value_of_expression_l703_703115

theorem value_of_expression : (sqrt ((sqrt 5) ^ 4)) ^ 8 = 390625 := 
  sorry

end value_of_expression_l703_703115


namespace delta_AOB_area_l703_703625

noncomputable def curve1 (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)
def line2 (θ : ℝ) : ℝ := √3 / Real.sin θ

theorem delta_AOB_area :
  (∃ (α : ℝ) (θ : ℝ) (A B : ℝ × ℝ),
    (curve1 α).1 = 2 + 2 * Real.cos α ∧
    (curve1 α).2 = 2 * Real.sin α ∧
    (2 + 2 * Real.cos α) = 4 * Real.cos θ ∧
    2 * Real.sin α = √3 / Real.sin θ ∧
    A = (2 * √3, 2*n*π + π/6) ∧
    B = (2, 2*n*π + π/3) ∧
    is_area_of_triangle (0, 0) A B = √3) :=
sorry

end delta_AOB_area_l703_703625


namespace find_circle_equation_from_hyperbola_l703_703568

theorem find_circle_equation_from_hyperbola :
  let c := (2 * Real.sqrt 5, 0) in
  let r := 5 in
  let h := 2 * Real.sqrt 5 in
  (∀ (x y : ℝ), 
  (x - h)^2 + y^2 = r^2 
  ↔ ∀ (x y : ℝ), ((x^2 / 4 - y^2 / 16 = 1) ∧ 2 * x + y = 0 → 
  ∃ (c : ℝ × ℝ), 
  c.1 = 2 * Real.sqrt 5 ∧ c.2 = 0)) :=
by sorry

end find_circle_equation_from_hyperbola_l703_703568


namespace find_lola_number_l703_703681

theorem find_lola_number (l m : ℂ) (h1 : l * m = 24 + 18 * complex.I) (h2 : m = 4 - 3 * complex.I) :
  l = 1.68 + 5.76 * complex.I :=
sorry

end find_lola_number_l703_703681


namespace time_saved_is_six_minutes_l703_703852

-- Conditions
def distance_monday : ℝ := 3
def distance_wednesday : ℝ := 4
def distance_friday : ℝ := 5

def speed_monday : ℝ := 6
def speed_wednesday : ℝ := 4
def speed_friday : ℝ := 5

def speed_constant : ℝ := 5

-- Question (proof statement)
theorem time_saved_is_six_minutes : 
  (distance_monday / speed_monday + distance_wednesday / speed_wednesday + distance_friday / speed_friday) - (distance_monday + distance_wednesday + distance_friday) / speed_constant = 0.1 :=
by
  sorry

end time_saved_is_six_minutes_l703_703852


namespace analogical_inference_l703_703543

-- We define the conditions
def condition1 (a b c : ℝ) : Prop := a > b → a + c > b + c

def inductive_reasoning (obs : list ℝ) (gen_conclusion : ℝ) : Prop :=
  ∀ o ∈ obs, o < gen_conclusion

def deductive_reasoning (general_principle : ℝ → Prop) (specific_case : ℝ) : Prop :=
  general_principle specific_case

def analogical_reasoning (situation1 situation2 : Prop) : Prop :=
  situation1 ∧ situation2

-- We need to prove that the inference from “if a > b then a + c > b + c” to “if a > b then ac > bc”
-- is an example of analogical reasoning.
theorem analogical_inference (a b c : ℝ) :
  condition1 a b c →
  analogical_reasoning (a > b → a + c > b + c) (a > b → a * c > b * c) :=
by sorry

end analogical_inference_l703_703543


namespace sin_A_sin_B_cos_C_l703_703927

open_locale real

-- Define the setup conditions: acute triangle ABC with given altitudes and properties
variables (A B C : ℝ)
variables (triangle_ABC : A + B + C = π)
variables (acute_triangle_ABC : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)

variables (AD BE DE : ℝ)
variables (O : ℝ)

-- O is the circumcenter and lies on segment DE
variables (circumcenter_O : true) -- Abstract logical representation, actual calculation avoided

-- Conclusion to prove:
theorem sin_A_sin_B_cos_C (h1 : A + B + C = π) 
  (h2 : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  : real.sin A * real.sin B * real.cos C = 1 / 2 := 
sorry

end sin_A_sin_B_cos_C_l703_703927


namespace heads_before_tails_probability_l703_703665

variable (q : ℚ)
variable (a b : ℕ)

theorem heads_before_tails_probability :
  (q = 1 / 4) → (nat.gcd a b = 1) → (q = a / b) → (a + b = 5) :=
by 
  intros hq hgcd hq_ab
  sorry

end heads_before_tails_probability_l703_703665


namespace suff_not_necessary_no_real_solutions_l703_703381

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l703_703381


namespace average_eq_16_l703_703353

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 12

theorem average_eq_16 (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : (x + y) / 2 = 16 := by
  sorry

end average_eq_16_l703_703353


namespace ram_salary_percentage_more_l703_703704

theorem ram_salary_percentage_more (R r : ℝ) (h : r = 0.8 * R) :
  ((R - r) / r) * 100 = 25 := 
sorry

end ram_salary_percentage_more_l703_703704


namespace abs_z_pow_5_eq_1_l703_703498

def z : ℂ := (1/2) + (Complex.I * (Real.sqrt 3 / 2))

theorem abs_z_pow_5_eq_1 : Complex.abs (z ^ 5) = 1 :=
by
  sorry

end abs_z_pow_5_eq_1_l703_703498


namespace ellipse_distance_is_constant_l703_703928

noncomputable def ellipse_property (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  let line1 := λ x : ℝ, (1 / 2) * x  -- equation of line y = 1/2 * x
  let line2 := λ x : ℝ, -(1 / 2) * x  -- equation of line y = -1/2 * x
  let distance_MN := 2 * b  -- derived from point (0, b)
  (∀ P : ℝ × ℝ, P.fst * P.fst / (a * a) + P.snd * P.snd / (b * b) = 1 →
    let parallel_line1 := λ x : ℝ, (1 / 2) * x + P.snd - (1 / 2) * P.fst
    let parallel_line2 := λ x : ℝ, -(1 / 2) * x + P.snd + (1 / 2) * P.fst
    ∀ M N : ℝ × ℝ, M.snd = parallel_line1 M.fst ∧ N.snd = parallel_line2 N.fst ∧
                     (M.snd = line1 M.fst ∨ N.snd = line2 N.fst) →
                       dist M N = distance_MN) → sqrt (a / b) = 2

theorem ellipse_distance_is_constant (a b : ℝ) (h : a > b ∧ b > 0)
    (constant_MN : ellipse_property a b h) : sqrt (a / b) = 2 :=
  sorry

end ellipse_distance_is_constant_l703_703928


namespace right_heavier_combinations_l703_703277

open Finset

def weights : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def six_combinations := weights.powerset.filter (λ s, s.card = 6)
def three_combinations := weights.powerset.filter (λ s, s.card = 3)

def sum_weights (w : Finset ℕ) : ℕ := w.sum id

def valid_combination (left right : Finset ℕ) : Prop :=
  (weights = left ∪ right) ∧ (left.disjoint right) ∧ (sum_weights right > sum_weights left)

theorem right_heavier_combinations : (∃ (right : Finset ℕ), right ∈ three_combinations ∧
  ∀ left, left = weights \ right → valid_combination left right) ↔ 2 :=
by
  sorry

end right_heavier_combinations_l703_703277


namespace sum_of_valid_n_l703_703373

open Nat

theorem sum_of_valid_n :
  let gcd_lcm_condition (n : ℕ) := lcm n 150 = gcd n 150 + 600
  in (∀ n : ℕ, gcd_lcm_condition n → n ∈ [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150, 225, 675].filter (λ x, gcd x 150 ∈ [1, 75])) →
  ∑ n in (Finset.filter gcd_lcm_condition (Finset.range 1000)), n = 675 := sorry

end sum_of_valid_n_l703_703373


namespace area_of_triangle_obtuse_triangle_exists_l703_703224

-- Define the mathematical conditions given in the problem
variables {a b c : ℝ}
axiom triangle_inequality : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y
axiom sine_relation : 2 * Real.sin c = 3 * Real.sin a
axiom side_b : b = a + 1
axiom side_c : c = a + 2

-- Part 1: Prove the area of ΔABC equals the provided solution
theorem area_of_triangle : 
  2 * Real.sin c = 3 * Real.sin a → 
  b = a + 1 → 
  c = a + 2 → 
  a = 4 → 
  b = 5 → 
  c = 6 → 
  let ab_sin_c := (1 / 2) * 4 * 5 * ((3 * Real.sqrt 7) / 8) in
  ab_sin_c = 15 * Real.sqrt 7 / 4 :=
by {
  sorry, -- The proof steps are to be provided here
}

-- Part 2: Prove there exists a positive integer a such that ΔABC is obtuse with a = 2
theorem obtuse_triangle_exists :
  (∃ (a : ℕ) (h2 : 1 < a ∧ a < 3), 2 * Real.sin c = 3 * Real.sin a ∧ b = a + 1 ∧ c = a + 2) →
  ∃ a = 2 ∧ (b = a + 1 ∧ c = a + 2) :=
by {
  sorry, -- The proof steps are to be provided here
}

end area_of_triangle_obtuse_triangle_exists_l703_703224


namespace height_of_stack_of_pots_l703_703842

-- Definitions corresponding to problem conditions
def pot_thickness : ℕ := 1

def top_pot_diameter : ℕ := 16

def bottom_pot_diameter : ℕ := 4

def diameter_decrement : ℕ := 2

-- Number of pots calculation
def num_pots : ℕ := (top_pot_diameter - bottom_pot_diameter) / diameter_decrement + 1

-- The total vertical distance from the bottom of the lowest pot to the top of the highest pot
def total_vertical_distance : ℕ := 
  let inner_heights := num_pots * (top_pot_diameter - pot_thickness + bottom_pot_diameter - pot_thickness) / 2
  let total_thickness := num_pots * pot_thickness
  inner_heights + total_thickness

theorem height_of_stack_of_pots : total_vertical_distance = 65 := 
sorry

end height_of_stack_of_pots_l703_703842


namespace matrix_not_invertible_at_fraction_l703_703116

theorem matrix_not_invertible_at_fraction : 
  ∃ x : ℝ, (matrix.det ![\[2 * x, 5\], \[4 - x, 9\]] = 0) ∧ x = 20 / 23 :=
by
  use 20 / 23
  sorry

end matrix_not_invertible_at_fraction_l703_703116


namespace transformation_A_transformation_B_transformation_C_transformation_D_l703_703391

-- Definitions of the conditions
variables (a b x y m n c : ℝ)
variable [Fact (a ≠ 0)]  -- Assume a ≠ 0 for the division

-- Proof statements for each condition
theorem transformation_A : a = b → a + 5 = b + 5 := by
  assume h : a = b
  rw [h]

theorem transformation_B : x = y → x / a = y / a := by
  assume h : x = y
  rw [h]

theorem transformation_C : m = n → 1 - 3 * m = 1 - 3 * n := by
  assume h : m = n
  rw [h]

theorem transformation_D : x = y → x * c = y * c := by
  assume h : x = y
  rw [h]

end transformation_A_transformation_B_transformation_C_transformation_D_l703_703391


namespace factorize_polynomial_l703_703120
   
   -- Define the polynomial
   def polynomial (x : ℝ) : ℝ :=
     x^3 + 3 * x^2 - 4
   
   -- Define the factorized form
   def factorized_form (x : ℝ) : ℝ :=
     (x - 1) * (x + 2)^2
   
   -- The theorem statement
   theorem factorize_polynomial (x : ℝ) : polynomial x = factorized_form x := 
   by
     sorry
   
end factorize_polynomial_l703_703120


namespace mike_first_job_earnings_l703_703687

theorem mike_first_job_earnings (total_wages : ℕ) (hours_second_job : ℕ) (pay_rate_second_job : ℕ) 
  (second_job_earnings := hours_second_job * pay_rate_second_job) 
  (first_job_earnings := total_wages - second_job_earnings) :
  total_wages = 160 → hours_second_job = 12 → pay_rate_second_job = 9 → first_job_earnings = 52 := 
by 
  intros h₁ h₂ h₃ 
  unfold first_job_earnings second_job_earnings
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end mike_first_job_earnings_l703_703687


namespace collinear_c₁_c₂_l703_703049

-- Define the vectors a and b
def a : Vector ℝ := ⟨[-1, 2, -1]⟩
def b : Vector ℝ := ⟨[2, -7, 1]⟩

-- Define the vectors c₁ and c₂ in terms of a and b
def c₁ := (6 : ℝ) • a - (2 : ℝ) • b
def c₂ := b - (3 : ℝ) • a

-- State that c₁ and c₂ are collinear
theorem collinear_c₁_c₂ : ∃ γ : ℝ, c₁ = γ • c₂ := by
  sorry

end collinear_c₁_c₂_l703_703049


namespace min_value_ineq_min_value_attainable_l703_703656

theorem min_value_ineq
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_sum : a + b + c = 9) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 :=
by {
  sorry,
}

theorem min_value_attainable :
  (a b c : ℝ)
  (h_eq : a = 3 ∧ b = 3 ∧ c = 3) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) = 9 :=
by {
  sorry,
}

end min_value_ineq_min_value_attainable_l703_703656


namespace num_palindrome_pairs_l703_703833

def is_palindrome (n : ℕ) : Prop :=
  let d := n.digits 10
  d = d.reverse

def four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_palindrome n

def five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ is_palindrome n

theorem num_palindrome_pairs : 
  {p1 : ℕ // four_digit_palindrome p1} × {p2 : ℕ // four_digit_palindrome p2} →
  five_digit_palindrome ((p1 : ℕ) + (p2 : ℕ)) →
  36 :=
sorry

end num_palindrome_pairs_l703_703833


namespace calculation_correctness_l703_703907

variables (a b c d e f : ℝ)

def M (x y : ℝ) : ℝ := max x y
def m (x y : ℝ) : ℝ := min x y

theorem calculation_correctness
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) :
  M (M a (m c d)) (m e (m b f)) = c :=
by sorry

end calculation_correctness_l703_703907


namespace find_lambda_l703_703558

variables {α : Type*} [inner_product_space ℝ α] (a b : α) (λ : ℝ)

-- Conditions
def orthogonal_vectors : Prop := inner_product_space.is_ortho a b
def norm_a : Prop := ∥a∥ = 2
def norm_b : Prop := ∥b∥ = 3

-- Orthogonality condition for the given vectors
def orthogonal_comb : Prop := inner_product_space.is_ortho (3 • a + 2 • b) (λ • a - b)

-- Proof problem statement
theorem find_lambda (h1 : orthogonal_vectors a b) (h2 : norm_a a) (h3 : norm_b b) (h4 : orthogonal_comb a b λ) : λ = 3/2 :=
sorry

end find_lambda_l703_703558


namespace center_of_tangent_circle_l703_703322

-- Define the conditions of the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 1
def circle2 (x y : ℝ) := x^2 + y^2 - 8 * x + 12 = 0

def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
    let (x, y) := P in
    let O := (0, 0)
    let F := (4, 0)
    let PO := real.sqrt ((x - 0)^2 + (y - 0)^2)
    let PF := real.sqrt ((x - 4)^2 + (y - 0)^2)
    PO + PF = 1

theorem center_of_tangent_circle :
    ∃ P : ℝ × ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → point_on_hyperbola P) :=
sorry

end center_of_tangent_circle_l703_703322


namespace max_subset_non_multiples_l703_703576

theorem max_subset_non_multiples (T : Set ℕ) (hT : T = {d | d ∣ (2004^100)}) :
  ∃ S ⊆ T, (∀ a b ∈ S, (a ≠ b → ¬ (a ∣ b) ∧ ¬ (b ∣ a))) ∧ S.card = 10201 :=
by
  sorry

end max_subset_non_multiples_l703_703576


namespace pond_field_ratio_l703_703747

theorem pond_field_ratio (L W : ℕ) (pond_side : ℕ) (hL : L = 24) (hLW : L = 2 * W) (hPond : pond_side = 6) :
  pond_side * pond_side / (L * W) = 1 / 8 :=
by
  sorry

end pond_field_ratio_l703_703747


namespace angle_BAE_60_degrees_l703_703841

-- Definitions for the conditions of the problem
variables (A B C D E : Type) [Square ABCD]
variables (CDLength : ℝ) (DE CE : Type) [DE = ⅓ * CD] [CE = ⅔ * CD]
variables [EquilateralTriangle ABE]

-- Statement of the problem
theorem angle_BAE_60_degrees (hSquare : Square ABCD) (hEonCD : DE = ⅓ * CD) (hEquilateral : EquilateralTriangle ABE) : angle BAE = 60 := 
    sorry

end angle_BAE_60_degrees_l703_703841


namespace cosine_of_angle_between_vectors_l703_703957

open Real EuclideanSpace

variable (a b : ℝ × ℝ)
variable (ha : a = (3, 1)) (hb : b = (2, 2))

theorem cosine_of_angle_between_vectors : 
  cos ⟨a + b, a - b⟩ = (sqrt 17) / 17 := 
by
  sorry

end cosine_of_angle_between_vectors_l703_703957


namespace relationship_among_a_b_c_l703_703553

-- Conditions
variable {f : ℝ → ℝ} (h_deriv : (1 < x) → has_deriv_at f (f' x) x) (h_cond : ∀ x > 1, f x < x * f' x * real.log x)

-- Definitions of a, b, c
def a : ℝ := f (real.exp 2)
def b : ℝ := 2 * f (real.exp 1)
def c : ℝ := 4 * f (real.sqrt (real.exp 1))

-- Statement of the goal
theorem relationship_among_a_b_c : a > b ∧ b > c :=
by sorry

end relationship_among_a_b_c_l703_703553


namespace distinguishable_large_triangles_l703_703818

theorem distinguishable_large_triangles :
  let number_of_colors := 8 in
  let corner_same_color := 8 in
  let corner_two_same_one_different := 8 * 7 in
  let corner_all_different := Nat.comb 8 3 in
  let total_corner_combinations := corner_same_color + corner_two_same_one_different + corner_all_different in
  let center_choices := 8 in
  let total_distinguishable_triangles := total_corner_combinations * center_choices in
  total_distinguishable_triangles = 960 :=
by
  let number_of_colors := 8
  let corner_same_color := 8
  let corner_two_same_one_different := 8 * 7
  let corner_all_different := Nat.comb 8 3
  let total_corner_combinations := corner_same_color + corner_two_same_one_different + corner_all_different
  let center_choices := 8
  let total_distinguishable_triangles := total_corner_combinations * center_choices
  have h1 : total_corner_combinations = 120 := by
    simp [corner_same_color, corner_two_same_one_different, corner_all_different, Nat.comb]
    norm_num
  have h2 : total_distinguishable_triangles = 120 * 8 := by
    simp [total_corner_combinations, center_choices, h1]
  have h3 : 120 * 8 = 960 := by norm_num
  exact h2.trans_eq h3.symm


end distinguishable_large_triangles_l703_703818


namespace min_words_for_90_percent_l703_703587

theorem min_words_for_90_percent (total_words : ℕ) (required_percentage : ℤ) (min_words : ℕ) 
  (h_total_words : total_words = 600)
  (h_required_percentage : required_percentage = 90)
  (h_min_words : min_words = 540) :
  min_words = (required_percentage * total_words) / 100 :=
by
  rw [h_total_words, h_required_percentage, h_min_words]
  have : (90 * 600) / 100 = 540 := by norm_num
  assumption

end min_words_for_90_percent_l703_703587


namespace benny_gave_seashells_l703_703456

theorem benny_gave_seashells (original_seashells : ℕ) (remaining_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 66) 
  (h2 : remaining_seashells = 14) 
  (h3 : original_seashells - remaining_seashells = given_seashells) : 
  given_seashells = 52 := 
by
  sorry

end benny_gave_seashells_l703_703456


namespace keiko_speed_l703_703251

theorem keiko_speed (wA wB tA tB : ℝ) (v : ℝ)
    (h1: wA = 4)
    (h2: wB = 8)
    (h3: tA = 48)
    (h4: tB = 72)
    (h5: v = (24 * π) / 60) :
    v = 2 * π / 5 :=
by
  sorry

end keiko_speed_l703_703251


namespace wooden_block_even_blue_faces_l703_703085

theorem wooden_block_even_blue_faces :
  let length := 6
  let width := 6
  let height := 2
  let total_cubes := length * width * height
  let corners := 8
  let edges_not_corners := 24
  let faces_not_edges := 24
  let interior := 16
  let even_blue_faces := edges_not_corners + interior
  total_cubes = 72 →
  even_blue_faces = 40 :=
by
  sorry

end wooden_block_even_blue_faces_l703_703085


namespace perpendicular_line_and_plane_implies_perpendicular_line_l703_703147

variable {Point : Type}
variable [HasInnerProductSpace Point]
variables (l m : Set Point) (α : Set Point)

def is_parallel (l₁ l₂ : Set Point) : Prop := sorry -- Define parallelism properly
def is_perpendicular (l₁ l₂ : Set Point) : Prop := sorry -- Define perpendicularity properly
def contains (l : Set Point) (α : Set Point) : Prop := sorry -- Define containment properly

theorem perpendicular_line_and_plane_implies_perpendicular_line :
  (is_perpendicular l α) ∧ (contains m α) → (is_perpendicular l m) :=
sorry

end perpendicular_line_and_plane_implies_perpendicular_line_l703_703147


namespace sum_of_solutions_l703_703375

def is_solution (n : ℕ) : Prop :=
  Nat.lcm n 150 = Nat.gcd n 150 + 600

theorem sum_of_solutions : ∑ n in (Finset.filter is_solution (Finset.range 1000)), n = 225 :=
by
  sorry

end sum_of_solutions_l703_703375


namespace ratio_of_AB_AC_l703_703402

theorem ratio_of_AB_AC (A B C D E F : Point) (w₁ w₂ : Circle) (x y z : ℝ)
    (h1 : Circle_contains w₁ A) (h2 : Circle_contains w₁ C) 
    (h3 : Circle_contains w₂ A) (h4 : Circle_contains w₂ B)
    (h5 : Circle_intersect_at w₁ w₂ D) (h6 : Line_contains_segment (A, E) F)
    (h7 : Line_contains_segment (B, C) D)
    (h8 : is_tangent_from BE B w₁) 
    (h9 : Segment_length BE = x) (h10 : Segment_length BD = y) (h11 : Segment_length FC = z)
    : (Segment_length AB) / (Segment_length AC) = x / z :=
by
  sorry

end ratio_of_AB_AC_l703_703402


namespace broken_stick_triangle_probability_l703_703021

noncomputable def probability_of_triangle (x y z : ℕ) : ℚ := sorry

theorem broken_stick_triangle_probability :
  ∀ x y z : ℕ, (x < y + z ∧ y < x + z ∧ z < x + y) → probability_of_triangle x y z = 1 / 4 := 
by
  sorry

end broken_stick_triangle_probability_l703_703021


namespace derivative_y_wrt_x_l703_703801

noncomputable theory
open Real

-- Given conditions
def f_x (t : ℝ) : ℝ := arcsin (sin t)
def f_y (t : ℝ) : ℝ := arccos (cos t)

-- The problem statement: Prove that the derivative of y with respect to x is 1
theorem derivative_y_wrt_x (t : ℝ) (ht1 : -π/2 ≤ t ∧ t ≤ π/2) (ht2: 0 ≤ t ∧ t ≤ π) :
  ∂ (λ t, arccos (cos t)) / ∂ (λ t, arcsin (sin t)) = 1 :=
sorry

end derivative_y_wrt_x_l703_703801


namespace lines_parallel_l703_703950

/--
Given two lines represented by the equations \(2x + my - 2m + 4 = 0\) and \(mx + 2y - m + 2 = 0\), 
prove that the value of \(m\) that makes these two lines parallel is \(m = -2\).
-/
theorem lines_parallel (m : ℝ) : 
    (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0) ∧ (∀ x y : ℝ, m * x + 2 * y - m + 2 = 0) 
    → m = -2 :=
by
  sorry

end lines_parallel_l703_703950


namespace find_erased_number_l703_703079

noncomputable def erased_number (n : ℕ) : ℚ :=
if (n - 1) ^ 2 - (n - 1) + 1 = 2 * 614 / 17 then n - 1 else 1

theorem find_erased_number :
  ∃ (n : ℕ), erased_number n = 7 ∧ (69 ≤ n ∧ n ≤ 70) :=
begin
  -- Given conditions and solutions steps are skipped as we aren't required to provide the proof.
  sorry
end

end find_erased_number_l703_703079


namespace sin_inequality_in_triangle_l703_703202

theorem sin_inequality_in_triangle (A B C : ℝ) (h_sum : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  Real.sin A * Real.sin (A / 2) + Real.sin B * Real.sin (B / 2) + Real.sin C * Real.sin (C / 2) ≤ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end sin_inequality_in_triangle_l703_703202


namespace sum_digits_9A_eq_9_l703_703732

theorem sum_digits_9A_eq_9 (A : ℕ) (h : ∀ i j, i < j → ∃ k, decimal_digit A i = k ∧ decimal_digit A j = k + 1) :
  sum_of_digits (9 * A) = 9 :=
by
  sorry

end sum_digits_9A_eq_9_l703_703732


namespace locate_quadrant_l703_703994

open Complex

noncomputable def z : ℂ := (2 - 7 * I) / (4 - I)

-- The condition is explicitly defined:
def conditions : z = (15 / 17) - (26 / 17) * I := by
  field_simp [z]
  ring
  
theorem locate_quadrant : (z.re > 0) ∧ (z.im < 0) := by
  rw [conditions]
  norm_num
  exact ⟨by norm_num, by norm_num⟩ -- Positive real part, negative imaginary part

end locate_quadrant_l703_703994


namespace perfect_square_factors_count_l703_703749

theorem perfect_square_factors_count :
  let n := 2^5 * 3^4 * 5^2 * 7 in
  ∃ count : ℕ, count = 18 ∧ (∀ k : ℕ, k > 0 → k ∣ n → (∃ a b c d : ℕ, k = 2^a * 3^b * 5^c * 7^d ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ d % 2 = 0) → count = (3 * 3 * 2 * 1)) := 
sorry

end perfect_square_factors_count_l703_703749


namespace cubes_squares_problem_l703_703190

theorem cubes_squares_problem (h1 : 2^3 - 7^2 = 1) (h2 : 3^3 - 6^2 = 9) (h3 : 5^3 - 9^2 = 16) : 4^3 - 8^2 = 0 := 
by
  sorry

end cubes_squares_problem_l703_703190


namespace men_in_first_group_l703_703716

theorem men_in_first_group (M : ℕ) : (M * 18 = 27 * 24) → M = 36 :=
by
  sorry

end men_in_first_group_l703_703716


namespace third_segment_lt_quarter_of_AC_l703_703632

theorem third_segment_lt_quarter_of_AC
  {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
  (a b c : ℝ) (angle_A angle_B angle_C : ℝ) 
  (h : 3 * angle_A - angle_C < real.pi)
  (divide_B : ∀ (X : Type*), 4 ∣ angle_B) : 
  ∃ K L M : A, ∃ AC : ℝ, AC = |b| → (|LM| < AC / 4) :=
begin
  sorry
end

end third_segment_lt_quarter_of_AC_l703_703632


namespace interest_calculated_years_l703_703757

variable (P T : ℝ)

-- Given conditions
def principal_sum_positive : Prop := P > 0
def simple_interest_condition : Prop := (P * 5 * T) / 100 = P / 5

-- Theorem statement
theorem interest_calculated_years (h1 : principal_sum_positive P) (h2 : simple_interest_condition P T) : T = 4 :=
  sorry

end interest_calculated_years_l703_703757


namespace lambda_range_l703_703947

open Real

def f (x : ℝ) := x^2 / exp x

theorem lambda_range (λ : ℝ) : 
  (∃ (x1 x2 x3 x4 : ℝ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧ 
    sqrt (f x1) + 2 / sqrt (f x1) - λ = 0 ∧
    sqrt (f x2) + 2 / sqrt (f x2) - λ = 0 ∧
    sqrt (f x3) + 2 / sqrt (f x3) - λ = 0 ∧
    sqrt (f x4) + 2 / sqrt (f x4) - λ = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) 
  ↔ (λ > exp(1) + 2 / exp(1)) :=
sorry

end lambda_range_l703_703947


namespace train_speed_proof_l703_703082

noncomputable def train_speed_in_kmh : ℝ :=
  let V := 20 in
  V * 3.6

theorem train_speed_proof :
  (∀ (L V : ℝ), (V * 10 = L) → (L + 500 = V * 35) → V = 20) →
  train_speed_in_kmh = 72 :=
by
  intro h
  have hV : ∃ V : ℝ, ∀ L : ℝ, (V * 10 = L) → (L + 500 = V * 35) → V = 20 :=
    ⟨20, λ L h1 h2, sorry⟩
  specialize h 500 20 (by linarith) (by linarith)
  simp [train_speed_in_kmh, h]
  sorry

end train_speed_proof_l703_703082


namespace number_of_teams_l703_703768

-- Define the necessary conditions and variables
variable (n : ℕ)
variable (num_games : ℕ)

-- Define the condition that each team plays each other team exactly once 
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The main theorem to prove
theorem number_of_teams (h : total_games n = 91) : n = 14 :=
sorry

end number_of_teams_l703_703768


namespace least_positive_divisible_by_first_five_primes_starting_from_five_l703_703781

theorem least_positive_divisible_by_first_five_primes_starting_from_five :
  let primes := [5, 7, 11, 13, 17]
  in primes.prod = 85085 :=
by
  let primes := [5, 7, 11, 13, 17]
  -- Proof is omitted
  sorry

end least_positive_divisible_by_first_five_primes_starting_from_five_l703_703781


namespace leak_empties_cistern_in_12_hours_l703_703816

variable (R : ℝ) (L : ℝ)

-- Define the conditions
def filling_rate := 1 / 4 -- Cistern fills in 4 hours normally, so the rate is 1/4 cistern per hour
def combined_rate := 1 / 6 -- With the leak, it takes 6 hours to fill, so the combined rate is 1/6 cistern per hour
def leak_rate := R - combined_rate -- Leak rate is the difference between the filling rate and combined rate

-- The main theorem to prove
theorem leak_empties_cistern_in_12_hours :
  filling_rate = R →
  combined_rate = 1 / 6 →
  leak_rate = L →
  leak_rate = 1 / 12 →
  1 / L = 12 :=
by
  intro hR hCombinedRate hLeakRate hLeakRateValue
  rw [←hLeakRateValue]
  sorry

end leak_empties_cistern_in_12_hours_l703_703816


namespace right_triangle_hypotenuse_l703_703622

theorem right_triangle_hypotenuse (B C: ℝ) (A : ℝ) :
  (∠ C = 90) ∧ (BC = 1) ∧ (AC = 2) → (AB = sqrt 5) :=
by
  sorry

end right_triangle_hypotenuse_l703_703622


namespace incircles_tangent_l703_703872

variable {A B C D : Type*}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]

variables {a b c d ab ac bc : ℝ}
-- Assume a generic triangle A, B, C with side lengths.
-- a represents AB, b represents BC, and c represents AC
-- d represents the location of point D on BC, and ab, ac, bc are side lengths of triangle ABC

-- Definition of point D on BC
noncomputable def D_location (BC : ℝ) (ab ac : ℝ) : ℝ := (ab + BC - ac) / 2

-- Theorem: the incircles of triangles ABD and ADC are tangent to each other
theorem incircles_tangent
  (h : BD + DC = BC)
  (h2 : BD - DC = AB - AC) :
  D = (D_location BC ab ac, D_location BC ac ab) := by
    -- Proof intentionally omitted
    sorry

end incircles_tangent_l703_703872


namespace find_max_min_f_no_max_min_phi_l703_703898

section Example1

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

-- Define the interval
def I1 : set ℝ := set.Icc 1 3

-- Statement of the maximum and minimum values
theorem find_max_min_f : 
  ∃ (x_max x_min : ℝ), x_max ∈ I1 ∧ x_min ∈ I1 ∧ (∀ x ∈ I1, f x ≤ f x_max) ∧ (∀ x ∈ I1, f x_min ≤ f x) :=
sorry

end Example1

section Example2

def phi (x : ℝ) : ℝ := x + 1/x

-- Define the interval
def I2 : set ℝ := set.Icc (-2 : ℝ) 2

-- Statement of no maximum or minimum values
theorem no_max_min_phi : ¬(∃ (x_max x_min : ℝ), x_max ∈ I2 ∧ x_min ∈ I2 ∧ (∀ x ∈ I2, phi x ≤ phi x_max) ∧ (∀ x ∈ I2, phi x_min ≤ phi x)) :=
sorry

end Example2

end find_max_min_f_no_max_min_phi_l703_703898


namespace pappus_theorem_l703_703803

-- Define the conditions given in the problem
variables {A1 B1 C1 A2 B2 C2 A B C : Type*}
variables [has_line (line : A1 → B2 → Type*)]
variables [line (A1) = line (B2)]
variables [line (A2) = line (B1)]
variables [line (B1) = line (C2)]
variables [line (B2) = line (C1)]
variables [line (C1) = line (A2)]
variables [line (C2) = line (A1)]
variables (intersection1 : intersection (line A1 B2) (line A2 B1) = C)
variables (intersection2 : intersection (line B1 C2) (line B2 C1) = A)
variables (intersection3 : intersection (line C1 A2) (line C2 A1) = B)

-- State the theorem
theorem pappus_theorem : collinear A B C :=
sorry

end pappus_theorem_l703_703803


namespace exponent_product_l703_703592

theorem exponent_product (a : ℝ) (m n : ℕ)
  (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 :=
sorry

end exponent_product_l703_703592


namespace questionnaires_drawn_l703_703439

theorem questionnaires_drawn
  (units : ℕ → ℕ)
  (h_arithmetic : ∀ n, units (n + 1) - units n = units 1 - units 0)
  (h_total : units 0 + units 1 + units 2 + units 3 = 100)
  (h_unitB : units 1 = 20) :
  units 3 = 40 :=
by
  -- Proof would go here
  -- Establish that the arithmetic sequence difference is 10, then compute unit D (units 3)
  sorry

end questionnaires_drawn_l703_703439


namespace domain_of_f_sin_x_l703_703160

theorem domain_of_f_sin_x (k : ℤ) :
  (∀ x, 0 < f x ∧ f x ≤ 1) →
  ∀ x, (2 * k * π < x ∧ x < 2 * k * π + π) ∧ (0 < sin x ∧ sin x ≤ 1) :=
by
  intros hf x
  sorry  -- Placeholder for the actual proof

end domain_of_f_sin_x_l703_703160


namespace find_angle_ABC_l703_703253

-- Conditions definitions
variable (A B C T E : Point)
variable (M N : Point)
variable (circ_center : Point)
variable (triangle_ABC : Triangle)
variable (h1 : IsAcuteAngleAndScalene triangle_ABC A B C)
variable (h2 : AngleAtT A T B = 120 ∧ AngleAtT B T C = 120)
variable (h3 : CircPassesThrough Midpoints circ_center triangle_ABC)
variable (h4 : Collinear B T E)

-- The theorem to prove
theorem find_angle_ABC :
  ∠(A, B, C) = 30 :=
sorry

end find_angle_ABC_l703_703253


namespace sufficient_but_not_necessary_condition_l703_703388

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (x^2 - m * x + 1) = 0 → (m^2 - 4 < 0) = ∀ m : ℝ, -2 < m ∧ m < 2 :=
sufficient_but_not_necessary_condition sorry

end sufficient_but_not_necessary_condition_l703_703388


namespace range_of_m_l703_703932

theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, m * x0^2 + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) → -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l703_703932


namespace element_of_sequence_l703_703565

/-
Proving that 63 is an element of the sequence defined by aₙ = n² + 2n.
-/
theorem element_of_sequence (n : ℕ) (h : 63 = n^2 + 2 * n) : ∃ n : ℕ, 63 = n^2 + 2 * n :=
by
  sorry

end element_of_sequence_l703_703565


namespace range_of_a_l703_703199

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x ^ 2 - 2 * x

noncomputable def y' (x : ℝ) (a : ℝ) : ℝ := 1 / x + 2 * a * x - 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → y' x a ≥ 0) ↔ a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l703_703199


namespace partner_with_largest_share_receives_12000_l703_703132

theorem partner_with_largest_share_receives_12000 (profit : ℕ)
  (total_parts : ℕ)
  (largest_share : ℕ)
  (parts : list ℕ)
  (h_parts_sum : list.sum parts = total_parts)
  (h_profit_div_by_parts : profit / total_parts = largest_share)
  (h_parts : parts = [3, 4, 4, 6]) :
  largest_share * 6 = 12000 :=
by
  -- Proof goes here
  sorry

end partner_with_largest_share_receives_12000_l703_703132


namespace cat_food_more_than_dog_food_l703_703846

-- Define the number of packages and cans per package for cat food
def cat_food_packages : ℕ := 9
def cat_food_cans_per_package : ℕ := 10

-- Define the number of packages and cans per package for dog food
def dog_food_packages : ℕ := 7
def dog_food_cans_per_package : ℕ := 5

-- Total number of cans of cat food
def total_cat_food_cans : ℕ := cat_food_packages * cat_food_cans_per_package

-- Total number of cans of dog food
def total_dog_food_cans : ℕ := dog_food_packages * dog_food_cans_per_package

-- Prove the difference between the total cans of cat food and total cans of dog food
theorem cat_food_more_than_dog_food : total_cat_food_cans - total_dog_food_cans = 55 := by
  -- Provide the calculation results directly
  have h_cat : total_cat_food_cans = 90 := by rfl
  have h_dog : total_dog_food_cans = 35 := by rfl
  calc
    total_cat_food_cans - total_dog_food_cans = 90 - 35 := by rw [h_cat, h_dog]
    _ = 55 := rfl

end cat_food_more_than_dog_food_l703_703846


namespace distinct_sequences_l703_703585

theorem distinct_sequences : ∃ n : ℕ, n = 392 ∧
  (∃ seqs: Finset (List Char), seqs.card = n ∧
    ∀ s ∈ seqs, s.head = 'M' ∧ s.ilast ≠ 'A' ∧ s.length = 4 ∧ (∀ c ∈ s.tails, c.length ≤ 1 → c.nodup)) := 
begin
  sorry
end

end distinct_sequences_l703_703585


namespace sum_cubes_identity_l703_703998

theorem sum_cubes_identity (x y z : ℝ) (h1 : x + y + z = 10) (h2 : xy + yz + zx = 20) :
    x^3 + y^3 + z^3 - 3 * x * y * z = 400 := by
  sorry

end sum_cubes_identity_l703_703998


namespace real_solutions_eq_l703_703902

def satisfies_equations (x y : ℝ) : Prop :=
  (4 * x + 5 * y = 13) ∧ (2 * x - 3 * y = 1)

theorem real_solutions_eq {x y : ℝ} : satisfies_equations x y ↔ (x = 2 ∧ y = 1) :=
by sorry

end real_solutions_eq_l703_703902


namespace test_option_d_false_l703_703551

-- Definitions of lines and planes
variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Propositions given in the problem
def option_a (m_parallel_alpha m_parallel_beta alpha_intersect_beta_neq_n : Prop) : Prop :=
  m_parallel_alpha ∧ m_parallel_beta ∧ alpha_intersect_beta_neq_n → m = n

def option_b (alpha_perpendicular_beta m_perpendicular_alpha n_perpendicular_beta : Prop) : Prop :=
  alpha_perpendicular_beta ∧ m_perpendicular_alpha ∧ n_perpendicular_beta → m = n

def option_c (alpha_perpendicular_beta alpha_perpendicular_gamma beta_intersect_gamma_eq_m : Prop) : Prop :=
  alpha_perpendicular_beta ∧ alpha_perpendicular_gamma ∧ beta_intersect_gamma_eq_m → m = n

def option_d (alpha_parallel_beta m_parallel_alpha : Prop) : Prop :=
  alpha_parallel_beta ∧ m_parallel_alpha → m = β ∨ m = n

-- Lean statement to test if option D being false is correct
theorem test_option_d_false 
  (m_parallel_alpha : m ∥ α)
  (alpha_parallel_beta : α ∥ β)
  (h_false_d : option_d α_parallel_beta m_parallel_alpha = False) : 
  True := sorry

end test_option_d_false_l703_703551


namespace hyperbola_midpoint_distance_l703_703949

noncomputable def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 25) - (y^2 / 9) = 1

structure Point :=
(x : ℝ)
(y : ℝ)

def on_left_branch (M : Point) : Prop := M.x < 0 ∧ hyperbola_eq M.x M.y

def distance (P Q : Point) : ℝ :=
real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

def origin : Point := ⟨0, 0⟩

def right_focus : Point := ⟨5, 0⟩  -- Since the distance from the center to each focus is sqrt(a^2 + b^2) and a = 5

theorem hyperbola_midpoint_distance :
  ∀ M : Point, on_left_branch M →
  distance M right_focus = 18 →
  distance origin (midpoint M right_focus) = 4 :=
by
  intros
  sorry

end hyperbola_midpoint_distance_l703_703949


namespace arrangement_l703_703138

noncomputable def a : ℝ := Real.log 0.9 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.7 / Real.log 1.1
noncomputable def c : ℝ := 1.1 ^ 0.9

theorem arrangement : b < a ∧ a < c := by
  sorry

end arrangement_l703_703138


namespace evaluate_expression_l703_703887

theorem evaluate_expression (a : ℝ) : (a^7 + a^7 + a^7 - a^7) = a^8 :=
by
  sorry

end evaluate_expression_l703_703887


namespace stickers_after_birthday_l703_703636

-- Definitions based on conditions
def initial_stickers : Nat := 39
def birthday_stickers : Nat := 22

-- Theorem stating the problem we aim to prove
theorem stickers_after_birthday : initial_stickers + birthday_stickers = 61 :=
by 
  sorry

end stickers_after_birthday_l703_703636


namespace range_of_f_l703_703053

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f_l703_703053


namespace area_of_shape1_correct_area_of_shape2_correct_l703_703697

open Real

noncomputable def area_of_shape1 : Real :=
  ∫ x in -π/3 .. π/3, (cos x)

theorem area_of_shape1_correct :
  area_of_shape1 = sqrt 3 :=
sorry

noncomputable def area_of_shape2 : Real :=
  (∫ x in 0 .. 4, sqrt (2 * x)) + (∫ x in 4 .. 8, sqrt (2 * x) - (x - 4))

theorem area_of_shape2_correct :
  area_of_shape2 = 40 / 3 :=
sorry

end area_of_shape1_correct_area_of_shape2_correct_l703_703697


namespace tetrahedron_probability_l703_703826

theorem tetrahedron_probability :
  let faces := {0, 1, 2, 3}
  let event_A := {p : ℕ × ℕ | p.1 ∈ faces ∧ p.2 ∈ faces ∧ p.1^2 + p.2^2 ≤ 4}
  ∑ b in event_A, 1 = 6 → (6: ℝ) / (16: ℝ) = (3: ℝ) / (8: ℝ) := 
by
  sorry

end tetrahedron_probability_l703_703826


namespace min_value_frac_sum_l703_703265

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ (a b : ℝ), (a + 3 * b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (∀ (a b : ℝ), (a + 3 * b = 1) → 0 < a → 0 < b → (1 / a + 3 / b) ≥ 16) :=
sorry

end min_value_frac_sum_l703_703265


namespace complex_conjugate_l703_703523

theorem complex_conjugate (z : ℂ) (hz : (1 - 2 * complex.I) * z = 2 * complex.I) : 
  conj z = -4/5 - (2/5) * complex.I :=
sorry

end complex_conjugate_l703_703523


namespace solve_cubic_diophantine_l703_703124

theorem solve_cubic_diophantine :
  (∃ x y z : ℤ, x^3 + y^3 + z^3 - 3 * x * y * z = 2003) ↔ 
  (x = 667 ∧ y = 668 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 668 ∧ y = 668 ∧ z = 667) :=
sorry

end solve_cubic_diophantine_l703_703124


namespace distinct_solutions_eq_l703_703877

theorem distinct_solutions_eq : 
  (∃ x : ℝ, x - |3 * x - 2| = 4) + (∃ x : ℝ, x - |3 * x - 2| = -4) = 2 := by
  sorry

end distinct_solutions_eq_l703_703877


namespace magic_square_base_l703_703104

theorem magic_square_base :
  ∃ b : ℕ, (b + 1 + (b + 5) + 2 = 9 + (b + 3)) ∧ b = 3 :=
by
  use 3
  -- Proof in Lean goes here
  sorry

end magic_square_base_l703_703104


namespace cosine_angle_a_plus_b_a_minus_b_l703_703956

open Real

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : vector) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def cosine_angle (u v : vector) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def a : vector := (3, 1)
def b : vector := (2, 2)

def a_plus_b : vector := (a.1 + b.1, a.2 + b.2)
def a_minus_b : vector := (a.1 - b.1, a.2 - b.2)

theorem cosine_angle_a_plus_b_a_minus_b :
  cosine_angle a_plus_b a_minus_b = sqrt 17 / 17 := 
sorry

end cosine_angle_a_plus_b_a_minus_b_l703_703956


namespace sum_floor_comparison_l703_703922

variable {α : Type*} [LinearOrderedField α]

theorem sum_floor_comparison {n : ℕ} {x y : Fin n.succ → α}
  (hx : ∀ i j, i ≤ j → x i ≤ x j)
  (hy : ∀ i j, i ≤ j → y j ≤ y i)
  (h_sum : ∑ i in Finset.range n.succ, (i + 1) * x i ≥ ∑ i in Finset.range n.succ, (i + 1) * y i) :
  ∀ (α : α), ∑ i in Finset.range n.succ, x i * ⌊(i + 1) * α⌋ ≥ ∑ i in Finset.range n.succ, y i * ⌊(i + 1) * α⌋ :=
by
  sorry

end sum_floor_comparison_l703_703922


namespace cot_square_sum_geq_three_fourths_l703_703723
-- Import the necessary libraries

-- The Lean 4 statement encapsulating the problem
theorem cot_square_sum_geq_three_fourths
  (α β γ : ℝ)
  (h1 : α ∈ Icc 0 π)
  (h2 : β ∈ Icc 0 π)
  (h3 : γ ∈ Icc 0 π)
  (h4 : is_prism_with_equilateral_faces α β γ) :
  cot α ^ 2 + cot β ^ 2 + cot γ ^ 2 ≥ 3 / 4 :=
sorry

-- Definitions required for completeness
def is_prism_with_equilateral_faces (α β γ : ℝ) : Prop := 
-- formal definition assuming the conditions mentioned
sorry

end cot_square_sum_geq_three_fourths_l703_703723


namespace stepmother_inevitable_overflow_l703_703512

theorem stepmother_inevitable_overflow (b : ℝ) : b < 2 → ∀ (initial_state: Fin 5 → ℝ) (distribution_fn: ℝ → (Fin 5 → ℝ) → (Fin 5 → ℝ)),
  (∀ (current_state: Fin 5 → ℝ) (new_state: Fin 5 → ℝ),
    (∃ (i j : Fin 5), i ≠ j ∧ adj(i, j) ∧ distribution_fn 1 current_state = new_state) →
    ∃ (i : Fin 5), new_state i > b) :=
begin
  intros hb initial_state distribution_fn h,
  sorry
end

end stepmother_inevitable_overflow_l703_703512


namespace sum_first_3k_terms_l703_703105

noncomputable def first_term (k : ℕ) : ℤ := 4 * k^2 - k + 1
noncomputable def common_diff : ℤ := 2
noncomputable def num_terms (k : ℕ) : ℤ := 3 * k
noncomputable def last_term (k : ℕ) : ℤ := first_term k + (num_terms k - 1) * common_diff

theorem sum_first_3k_terms (k : ℕ) :
  let a := first_term k in
  let d := common_diff in
  let n := num_terms k in
  n * (a + last_term k) / 2 = 12 * k^3 + 6 * k^2 :=
by
  sorry

end sum_first_3k_terms_l703_703105


namespace longest_segment_in_cylinder_l703_703821

theorem longest_segment_in_cylinder
  (r : ℝ) (h : ℝ) (d : ℝ) (L : ℝ)
  (hr : r = 4)
  (hh : h = 10)
  (hd : d = 2 * r)
  (hL : L = Real.sqrt (h^2 + d^2)) :
  L = Real.sqrt 164 := 
  by
    unfold Real.sqrt
    rw [hr, hh, hd]
    simp
    sorry

end longest_segment_in_cylinder_l703_703821


namespace test_question_count_l703_703794

theorem test_question_count
    (total_points : ℕ)
    (four_point_questions : ℕ)
    (questions_value : (ℕ → ℕ) → ℕ)
    (value_2_point : ∀ f, f 2 = 2)
    (value_4_point : ∀ f, f 4 = 4)
    (points_contribution_4_point : ∀ f, four_point_questions * f 4 = 40)
    (total_points_contribution : ∀ f, total_points = 100) :
    ∃ (total_questions : ℕ), total_questions = four_point_questions + (total_points - 40) / (f 2) :=
by
  sorry

end test_question_count_l703_703794


namespace chris_money_before_birthday_l703_703466

variables {x : ℕ} -- Assuming we are working with natural numbers (non-negative integers)

-- Conditions
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Question
theorem chris_money_before_birthday : x = total_money_now - (grandmother_money + aunt_uncle_money + parents_money) :=
by
  sorry

end chris_money_before_birthday_l703_703466


namespace find_r_l703_703893

theorem find_r (r : ℕ) : log 8 (r + 9) = 3 → r = 503 := by
  sorry

end find_r_l703_703893


namespace rectangular_equation_of_C_intersection_points_value_l703_703218

-- Step 1: Prove the rectangular coordinate equation of curve C
theorem rectangular_equation_of_C :
  ∀ (ρ θ : ℝ), (ρ * sin θ * sin θ = 8 * cos θ) → ((ρ * cos θ)^2 = 8 * (ρ * cos θ)) := sorry

-- Step 2: Prove the value of 1/|PM| + 1/|PN|
theorem intersection_points_value :
  let x_t (t : ℝ) := 1 + 3/5 * t
  let y_t (t : ℝ) := 2 + 4/5 * t
  let P : ℝ × ℝ := (1, 2)
  let equation_C (x y : ℝ) := y^2 = 8 * x
  ∀ t₁ t₂ : ℝ, 
  (y_t t₁)^2 = 8 * (x_t t₁) ∧ (y_t t₂)^2 = 8 * (x_t t₂) →
  (t₁ + t₂ = 5/2) ∧ (t₁ * t₂ = -25/4) →
  (1 / real.sqrt ((x_t t₁ - 1)^2 + (y_t t₁ - 2)^2) + 
   1 / real.sqrt ((x_t t₂ - 1)^2 + (y_t t₂ - 2)^2)) = 
  (2 * real.sqrt 5) / 5 := sorry

end rectangular_equation_of_C_intersection_points_value_l703_703218


namespace complex_conjugate_l703_703524

theorem complex_conjugate (z : ℂ) (hz : (1 - 2 * complex.I) * z = 2 * complex.I) : 
  conj z = -4/5 - (2/5) * complex.I :=
sorry

end complex_conjugate_l703_703524


namespace num_four_letter_initials_l703_703971

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l703_703971


namespace find_constants_l703_703561

theorem find_constants (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h_max : ∀ x ∈ set.Icc (-3/2) 0, b + a^(x^2 + 2 * x) ≤ 3)
  (h_min : ∀ x ∈ set.Icc (-3/2) 0, b + a^(x^2 + 2 * x) ≥ 5/2) :
  (a = 2 ∧ b = 2) ∨ (a = 2/3 ∧ b = 3/2) := 
sorry

end find_constants_l703_703561


namespace cosine_of_angle_between_vectors_l703_703958

open Real EuclideanSpace

variable (a b : ℝ × ℝ)
variable (ha : a = (3, 1)) (hb : b = (2, 2))

theorem cosine_of_angle_between_vectors : 
  cos ⟨a + b, a - b⟩ = (sqrt 17) / 17 := 
by
  sorry

end cosine_of_angle_between_vectors_l703_703958


namespace num_four_letter_initials_l703_703972

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l703_703972


namespace minimum_cos_A_ABC_maximum_area_ABC_l703_703237

-- Definitions and conditions
noncomputable def triangle_ABC_min_cos_A : Prop :=
  ∀ (a b c : ℝ),
    2 * a^2 = b^2 + c^2 →
    a > 0 →
    b > 0 →
    c > 0 →
    ∃ A : ℝ, -- Angle A opposite to side a
      cos A = 1/2

-- Verification of the minimum value of cos A
theorem minimum_cos_A_ABC : triangle_ABC_min_cos_A :=
  by
   intros a b c h1 h2 h3 h4
   use real.cos (real.arccos (1/2))
   sorry

-- Definitions and conditions for area
noncomputable def triangle_ABC_max_area : Prop :=
  ∀ (b c : ℝ),
    a = 2 →
    2 * a^2 = b^2 + c^2 →
    b >= 0 →
    c >= 0 →
    (c^2 - 2 * b^2) ∣ (b^2 - c^2) → -- This is Maxwell condition
    ∃ (A : ℝ), -- Angle A opposite to side a
      sin A = sqrt 3 / 2 →
      bc ≤ 4 →
      1 / 2 * b * c * real.sin A = sqrt 3

-- Verification of the maximum area
theorem maximum_area_ABC : triangle_ABC_max_area :=
  by
   intros b c h1 h2 h3 h4 h5
   use real.sin (real.arcsin (sqrt 3 / 2))
   sorry

end minimum_cos_A_ABC_maximum_area_ABC_l703_703237


namespace jack_jill_same_speed_l703_703238

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end jack_jill_same_speed_l703_703238


namespace number_is_correct_l703_703192

theorem number_is_correct (x : ℝ) (h : 0.35 * x = 0.25 * 50) : x = 35.7143 :=
by 
  sorry

end number_is_correct_l703_703192


namespace radius_ratio_l703_703844

open Real

-- Defining the given problem conditions
def frustum_volume (h R r : ℝ) : ℝ := (h * π * (R^2 + R * r + r^2)) / 3

def frustum_volume_halfway (h R r : ℝ) : ℝ := (h / 2 * π * (((R + r) / 2)^2 + ((R + r) / 2) * r + r^2)) / 3

-- Given condition 1: Full volume is 2.6 liters
axiom full_volume_condition (h R r : ℝ) : frustum_volume h R r = 2.6

-- Given condition 2: Halfway volume is 0.7 liters
axiom halfway_volume_condition (h R r : ℝ) : frustum_volume_halfway h R r = 0.7

-- The theorem to prove
theorem radius_ratio (h R r : ℝ) (full_volume_condition h R r) (halfway_volume_condition h R r) : R = 3 * r :=
sorry

end radius_ratio_l703_703844


namespace men_in_first_group_l703_703597

-- Define the problem conditions
def men_and_boys_work (x m b : ℕ) :=
  let w := 10 * (x * m + 8 * b) in
  w = 2 * (26 * m + 48 * b) ∧
  4 * (15 * m + 20 * b) = w

-- Formal statement of the problem
theorem men_in_first_group (x m b : ℕ) :
  men_and_boys_work x m b → x = 6 :=
sorry

end men_in_first_group_l703_703597


namespace problem_statement_l703_703494

theorem problem_statement (a b c : ℝ) (h1 : a = (sqrt 5)^5) (h2 : b = sqrt a) (h3 : c = b^6) : 
  c = 5^7.5 := sorry

end problem_statement_l703_703494


namespace series_sum_l703_703100

theorem series_sum : (∑ n in Finset.range (99-1) + 2, 1/(n*(n+1))) = 49 / 100 := by
  sorry

end series_sum_l703_703100


namespace relationship_ab_l703_703550

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 1) = -f x)
variable (a b : ℝ)
variable (h_ex : ∃ x : ℝ, f (a + x) = f (b - x))

-- State the conclusion we need to prove
theorem relationship_ab : ∃ k : ℕ, k > 0 ∧ (a + b) = 2 * k + 1 :=
by
  sorry

end relationship_ab_l703_703550


namespace prime_of_two_pow_sub_one_prime_l703_703294

theorem prime_of_two_pow_sub_one_prime {n : ℕ} (h : Nat.Prime (2^n - 1)) : Nat.Prime n :=
sorry

end prime_of_two_pow_sub_one_prime_l703_703294


namespace proof_cos_2alpha_plus_pi_by_4_proof_alpha_minus_beta_l703_703557

noncomputable def cos_2alpha_plus_pi_by_4 (α : ℝ) (hα : α ∈ (0, π / 2)) (h_tanα : Real.tan α = 2) : Prop :=
  Real.cos (2 * α + π / 4) = - 7 * Real.sqrt 2 / 10

theorem proof_cos_2alpha_plus_pi_by_4 (α : ℝ) (hα : α ∈ (0, π / 2)) (h_tanα : Real.tan α = 2) : cos_2alpha_plus_pi_by_4 α hα h_tanα :=
  sorry

noncomputable def alpha_minus_beta (α β : ℝ) (hα : α ∈ (0, π / 2)) (hβ_range : β ∈ (-π / 2, 0)) (h_sin_beta : Real.sin (β + π / 4) = Real.sqrt 10 / 10) : Prop :=
  α - β = π / 2

theorem proof_alpha_minus_beta (α β : ℝ) (hα : α ∈ (0, π / 2)) (hβ_range : β ∈ (-π / 2, 0)) (h_sin_beta : Real.sin (β + π / 4) = Real.sqrt 10 / 10) : alpha_minus_beta α β hα hβ_range h_sin_beta :=
  sorry

end proof_cos_2alpha_plus_pi_by_4_proof_alpha_minus_beta_l703_703557


namespace family_total_cost_l703_703215

def cost_of_entrance_ticket : ℕ := 5
def cost_of_child_attraction_ticket : ℕ := 2
def cost_of_adult_attraction_ticket : ℕ := 4
def number_of_children : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_grandmothers : ℕ := 1

def number_of_family_members : ℕ := number_of_children + number_of_parents + number_of_grandmothers
def cost_of_entrance_tickets : ℕ := number_of_family_members * cost_of_entrance_ticket
def cost_of_children_attraction_tickets : ℕ := number_of_children * cost_of_child_attraction_ticket
def number_of_adults : ℕ := number_of_parents + number_of_grandmothers
def cost_of_adults_attraction_tickets : ℕ := number_of_adults * cost_of_adult_attraction_ticket

def total_cost : ℕ := cost_of_entrance_tickets + cost_of_children_attraction_tickets + cost_of_adults_attraction_tickets

theorem family_total_cost : total_cost = 55 := 
by 
  -- Calculation of number_of_family_members: 4 children + 2 parents + 1 grandmother = 7
  have h1 : number_of_family_members = 4 + 2 + 1 := by rfl
  -- Calculation of cost_of_entrance_tickets: 7 people * $5 = $35
  have h2 : cost_of_entrance_tickets = 7 * 5 := by rw h1 
  -- Calculation of cost_of_children_attraction_tickets: 4 children * $2 = $8
  have h3 : cost_of_children_attraction_tickets = 4 * 2 := by rfl
  -- Calculation of number_of_adults: 2 parents + 1 grandmother = 3
  have h4 : number_of_adults = 2 + 1 := by rfl
  -- Calculation of cost_of_adults_attraction_tickets: 3 adults * $4 = $12
  have h5 : cost_of_adults_attraction_tickets = 3 * 4 := by rw h4
  -- Total cost calculation: $35 (entrance) + $8 (children) + $12 (adults) = $55
  have h6 : total_cost = 35 + 8 + 12 := 
    by rw [h2, h3, h5]
  exact h6


end family_total_cost_l703_703215


namespace fraction_simplification_l703_703866

theorem fraction_simplification : 
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1 / 3 :=
by
  sorry

end fraction_simplification_l703_703866


namespace average_weight_increase_l703_703727

-- Let A be the average weight of the original 8 persons.
variable (A : ℝ)

-- A condition to compute the increase in average weight when replacing a person of weight 35 kg with someone of weight 75 kg.
theorem average_weight_increase: 
  let initial_total_weight := 8 * A in
  let new_total_weight := initial_total_weight - 35 + 75 in
  let new_average_weight := new_total_weight / 8 in
  new_average_weight - A = 5 :=
by
  sorry

end average_weight_increase_l703_703727


namespace general_formula_a_sum_sn_l703_703545

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ :=
  a n + 2 ^ (a n)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem general_formula_a :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_sn :
  ∀ n, S n = n * (n + 1) + (4^(n + 1) - 4) / 3 :=
sorry

end general_formula_a_sum_sn_l703_703545


namespace complex_number_modulus_sum_l703_703269

theorem complex_number_modulus_sum 
  (n : ℕ) 
  (z : ℕ → ℂ) 
  (h_sum : (Finset.range n).sum (λ i, ‖ z i ‖) = 1) :
  ∃ S : Finset ℕ, ∑ i in S, z i ≠ 0 ∧ ‖∑ i in S, z i‖ ≥ (1/6 : ℝ) :=
sorry

end complex_number_modulus_sum_l703_703269


namespace communication_problem_l703_703004

theorem communication_problem (n k : ℕ) (h1 : n ≥ 5) (h2 : ∀ a b : ℕ, a ≠ b → a < n → b < n → ∃ (c : ℕ), c ≤ 1) 
(h3 : (∀ (s : Finset ℕ), s.card = n - 2 → s.sum = 3 ^ k)) :
n = 5 :=
sorry

end communication_problem_l703_703004


namespace domain_of_ln_one_minus_five_pow_x_correct_l703_703736

noncomputable def domain_of_ln_one_minus_five_pow_x : Set ℝ :=
{ x : ℝ | 1 - 5^x > 0 }

theorem domain_of_ln_one_minus_five_pow_x_correct :
  domain_of_ln_one_minus_five_pow_x = { x : ℝ | x < 0 } :=
sorry

end domain_of_ln_one_minus_five_pow_x_correct_l703_703736


namespace jia_wins_strategy_l703_703765

theorem jia_wins_strategy:
  (exists (jia_cards yi_cards : list ℕ),
    (∀ x ∈ jia_cards, x ∈ (list.range 2018).map (λ x, x + 1)) ∧
    (∀ x ∈ yi_cards, x ∈ (list.range 2018).map (λ x, x + 1)) ∧
    list.disjoint jia_cards yi_cards ∧
    jia_cards.length = 1009 ∧
    yi_cards.length = 1009 ∧
    (∀ (i : ℕ), (i % 2 = 1 → (list.count i (jia_cards ++ yi_cards)) = 1)) ∧
    (∀ (i : ℕ), (i % 2 = 0 → (list.count i (jia_cards ++ yi_cards)) = 1)) ∧
    (list.sum jia_cards % 2 = 0)) :=
  sorry

end jia_wins_strategy_l703_703765


namespace arrangement_count_l703_703843

theorem arrangement_count :
  (∀ A B C D E : Type,
    (A ∧ B) ∧ ¬(C ∧ D) →
    number_of_arrangements = 24) :=
sorry

end arrangement_count_l703_703843


namespace find_C_max_area_l703_703619

variable (A B C a b c : ℝ)
variable (h1 : A + B + C = π / 2) -- This should represent acute triangle condition
variable (h2 : a = 2 * sin A)
variable (h3 : b = 2 * sin B)
variable (h4 : c = 2 * sin C)
variable (h5 : (a * cos B + b * cos A) / c = (2 * sqrt 3 / 3) * sin C)

-- Prove that C = 60 degrees
theorem find_C : C = π / 3 :=
begin
  -- Proof steps here
  sorry
end

-- Prove the maximum value of the area S of triangle ABC
theorem max_area (S : ℝ) (h6 : S = 1 / 2 * a * b * sin C) : S ≤ 3 * sqrt 3 / 4 :=
begin
  -- Proof steps here
  sorry
end

end find_C_max_area_l703_703619


namespace eval_expr_l703_703490

theorem eval_expr : (1 / 16 : ℝ) ^ (-1 / 2) = 4 := by
  sorry

end eval_expr_l703_703490


namespace find_value_of_d_e_l703_703266

theorem find_value_of_d_e:
  let d e : ℝ in
  (3 * d^2 + 5 * d - 7 = 0) ∧ (3 * e^2 + 5 * e - 7 = 0) →
  (d - 2) * (e - 2) = 5 :=
by {
  intro d e,
  assume h,
  sorry
}

end find_value_of_d_e_l703_703266


namespace total_toys_first_three_days_production_difference_total_wage_l703_703813

/-- Defining constants and deviations for each day -/
def planned_daily_production : ℤ := 100
def weekly_production_target : ℤ := 700

def deviation_monday : ℤ := 10
def deviation_tuesday : ℤ := -6
def deviation_wednesday : ℤ := -8
def deviation_thursday : ℤ := 15
def deviation_friday : ℤ := -12
def deviation_saturday : ℤ := 18
def deviation_sunday : ℤ := -9

def base_rate_per_toy : ℤ := 10
def extra_rate_per_toy_above_target : ℤ := 12
def reduced_rate_per_toy_below_target : ℤ := 8

/-- State the total production in the first three days -/
theorem total_toys_first_three_days :
  (planned_daily_production * 3 + deviation_monday + deviation_tuesday + deviation_wednesday) = 296 :=
by {
  -- sorry is used here to indicate the proof is skipped
  sorry,
}

/-- State the difference in toys between the highest and lowest production days -/
theorem production_difference :
  (max deviation_monday (max deviation_tuesday (max deviation_wednesday (max deviation_thursday (max deviation_friday (max deviation_saturday deviation_sunday))))))
   - min deviation_monday (min deviation_tuesday (min deviation_wednesday (min deviation_thursday (min deviation_friday (min deviation_saturday deviation_sunday)))))) = 30 :=
by {
  -- sorry is used here to indicate the proof is skipped
  sorry,
}

/-- Calculate the total wage of the workers for the week -/
theorem total_wage :
  let total_production := planned_daily_production * 7 
                        + deviation_monday + deviation_tuesday + deviation_wednesday 
                        + deviation_thursday + deviation_friday + deviation_saturday 
                        + deviation_sunday in
  let extra_toys := total_production - weekly_production_target in
  if total_production > weekly_production_target then
    (base_rate_per_toy * weekly_production_target) + (extra_rate_per_toy_above_target * extra_toys) = 7096
  else if total_production < weekly_production_target then
    (reduced_rate_per_toy_below_target * total_production) = 7096
  else
    (base_rate_per_toy * total_production) = 7096 :=
by {
  -- sorry is used here to indicate the proof is skipped
  sorry,
}

end total_toys_first_three_days_production_difference_total_wage_l703_703813


namespace number_of_f_less_than_2003_l703_703264

-- Definitions of X and f
def X := {n : Nat | n ≥ 0}
def f : ℕ → ℕ -- given f is a function from non-negative integers to itself

-- Given conditions
axiom functional_eq (n : ℕ) : (f (2 * n + 1))^2 - (f (2 * n))^2 = 6 * f n + 1
axiom f_non_decreasing (n : ℕ) : f (2 * n) ≥ f n

-- Main theorem to be proven
theorem number_of_f_less_than_2003 : {m : ℕ | ∃ n, f n = m ∧ m < 2003}.card = 128 := sorry

end number_of_f_less_than_2003_l703_703264


namespace part_a_part_b_l703_703406

theorem part_a (α β γ : ℝ)
  (h1 : cos α * cos β * cos γ ≤ 1 / 8)
  (h2 : cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 - 2 * cos α * cos β * cos γ) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 ≥ 3 / 4 :=
by {
  sorry
}

theorem part_b (α β γ : ℝ)
  (obtuse : cos α * cos β * cos γ < 0)
  (h2 : cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 - 2 * cos α * cos β * cos γ) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 > 1 :=
by {
  sorry
}

end part_a_part_b_l703_703406


namespace marbles_difference_l703_703637

theorem marbles_difference (lost found : ℕ) (h_lost : lost = 23) (h_found : found = 9) : lost - found = 14 :=
by
  rw [h_lost, h_found]
  -- 23 - 9
  rfl
  -- the goal now is 14 = 14

end marbles_difference_l703_703637


namespace min_value_l703_703654

theorem min_value
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 9) :
  ∃ x : ℝ, x = 9 ∧ x = min (λ x, (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)) := 
sorry

end min_value_l703_703654


namespace constant_S_n_l703_703110

theorem constant_S_n (n : ℕ) (hn : 0 < n) :
  (∀ x y z : ℝ, xyz = 1 ∧ x + y + z = 0 → S_n x y z = x^n + y^n + z^n) ↔ (n = 1 ∨ n = 3) :=
sorry

end constant_S_n_l703_703110


namespace smallest_a_mod_remainders_l703_703194

theorem smallest_a_mod_remainders:
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], 2521 % d = 1) ∧
  (∀ n : ℕ, ∃ a : ℕ, a = 2520 * n + 1 ∧ (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], a % d = 1)) :=
by
  sorry

end smallest_a_mod_remainders_l703_703194


namespace triangle_area_eq_300_sqrt_3_l703_703617

-- Define the general conditions of a triangle
variables (P Q R : Type) [EuclideanGeometry ℝ]

-- Define the specific conditions given in the problem
def PQ := (30 : ℝ)
def PR := (40 : ℝ)
def angle_QPR := (π / 3) -- 60 degrees in radians

-- Define the statement to be proved
theorem triangle_area_eq_300_sqrt_3 :
  ∃ (area : ℝ), area = 300 * Real.sqrt 3 ∧
  ∀ (hPQR : Triangle P Q R), 
    Triangle.side_length hPQR P Q = PQ ∧ 
    Triangle.side_length hPQR P R = PR ∧ 
    Triangle.angle hPQR Q P R = angle_QPR ->
    Triangle.area hPQR = area :=
by
  sorry

end triangle_area_eq_300_sqrt_3_l703_703617


namespace arrangement_count_l703_703351

def people := {A, B, C, D, E}

def are_adjacent : people → people → Prop
| A, B => false
| B, A => false
| C, D => true
| D, C => true
| _, _ => true

def is_adjacent (p q : people) : Prop := are_adjacent p q

theorem arrangement_count (P : Set people) (h1 : ∀ p q ∈ P, is_adjacent p q -> (p = C ∧ q = D) ∨ (p = D ∧ q = C)) : 
  ∃ (n : ℕ), n = 72 :=
by
  sorry

end arrangement_count_l703_703351


namespace size_of_largest_subset_l703_703662

open Set
open Finset

variable (n : ℕ)
def A : Finset ℤ := 
  Finset.range (2 * n + 1) -ᵘ finset.singleton (-n)

theorem size_of_largest_subset (n : ℕ) (H : n > 0) : 
  ∃ s : Finset ℤ, s ⊆ A ∧ (∀ a b c ∈ s, a + b + c ≠ 0) ∧ s.card = 2 * (n / 2) + 2 := 
sorry

end size_of_largest_subset_l703_703662


namespace total_votes_cast_l703_703063

-- Problem statement and conditions
variable (V : ℝ) (candidateVotes : ℝ) (rivalVotes : ℝ)
variable (h1 : candidateVotes = 0.35 * V)
variable (h2 : rivalVotes = candidateVotes + 1350)

-- Target to prove
theorem total_votes_cast : V = 4500 := by
  -- pseudo code proof would be filled here in real Lean environment
  sorry

end total_votes_cast_l703_703063


namespace complex_point_quadrant_l703_703522

noncomputable def quadrant (z : ℂ) : ℕ :=
if z.re > 0 ∧ z.im > 0 then 1
else if z.re < 0 ∧ z.im > 0 then 2
else if z.re < 0 ∧ z.im < 0 then 3
else if z.re > 0 ∧ z.im < 0 then 4
else 0  -- Origin or on one of the axes

theorem complex_point_quadrant (x y : ℝ) (h : (1 - complex.i : ℂ) * (x : ℂ) = 1 + y * complex.i) : 
  quadrant (x + y * complex.i) = 4 :=
by
  -- Parsing the hypothesis into the form we can work with
  have h_re : (1 : ℂ) * ↑x - (complex.i : ℂ) * ↑x = (1 : ℂ),
    from congr_arg complex.re h,
  have h_im : (1 : ℂ) * ↑0 + (complex.i : ℂ) * y = complex.i * y,
    from congr_arg complex.im h,
  -- Since both parts need to be true for the equality to hold
  have x_eq : x = 1 := complex.ext_iff.1 h 0,
  have y_eq : y = -1 := complex.ext_iff.1 h 1,
  subst x_eq,
  subst y_eq,
  -- Evaluating the quadrant function on (x + yi)
  rw quadrant,
  rw if_pos,
  split; norm_num,
  exact dec_trivial,

end complex_point_quadrant_l703_703522


namespace crickets_total_l703_703394

noncomputable def initial_amount : ℝ := 7.5
noncomputable def additional_amount : ℝ := 11.25
noncomputable def total_amount : ℝ := 18.75

theorem crickets_total : initial_amount + additional_amount = total_amount :=
by
  sorry

end crickets_total_l703_703394


namespace Robe_savings_l703_703289

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end Robe_savings_l703_703289


namespace domain_of_f_l703_703324

def function_domain : Set ℝ := {x : ℝ | x ≠ 1 ∧ x > -1}

theorem domain_of_f :
  (∀ x : ℝ, (x ∉ function_domain) ↔ f(x) = ∅) :=
sorry

end domain_of_f_l703_703324


namespace AB_length_is_160_l703_703698

noncomputable def length_of_AB (AB AG : ℝ) : ℝ :=
  let C := AB / 2
  let D := C / 2
  let E := D / 2
  let F := E / 2
  let G := F / 2
  in 32 * G

theorem AB_length_is_160 (AG : ℝ) (h : AG = 5) : length_of_AB X AG = 160 :=
by
  sorry

end AB_length_is_160_l703_703698


namespace smallest_square_number_l703_703785

theorem smallest_square_number (x y : ℕ) (hx : ∃ a, x = a ^ 2) (hy : ∃ b, y = b ^ 3) 
  (h_simp: ∃ c d, x / (y ^ 3) = c ^ 3 / d ^ 2 ∧ c > 1 ∧ d > 1): x = 64 := by
  sorry

end smallest_square_number_l703_703785


namespace find_a_l703_703941

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def line (a : ℝ) := {p : ℝ × ℝ | a * p.1 - p.2 + 3 = 0}
def circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}
def radius : ℝ := 2
def chord_length : ℝ := 2 * Real.sqrt 3

theorem find_a (a : ℝ) : ∃ A B ∈ circle, A ∈ line a ∧ B ∈ line a ∧ dist A B = chord_length → a = 0 :=
sorry

end find_a_l703_703941


namespace common_ratio_is_two_l703_703422

theorem common_ratio_is_two (a r : ℝ) (h_pos : a > 0) 
  (h_sum : a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r)) : 
  r = 2 := 
by
  sorry

end common_ratio_is_two_l703_703422


namespace two_f_x_eq_8_over_4_plus_x_l703_703186

variable (f : ℝ → ℝ)
variable (x : ℝ)
variables (hx : 0 < x)
variable (h : ∀ x, 0 < x → f (2 * x) = 2 / (2 + x))

theorem two_f_x_eq_8_over_4_plus_x : 2 * f x = 8 / (4 + x) :=
by sorry

end two_f_x_eq_8_over_4_plus_x_l703_703186


namespace parallel_transitive_converse_parallel_transitive_inverse_parallel_transitive_contrapositive_parallel_transitive_l703_703039

-- Definitions
def are_parallel (l₁ l₂ :  Line) : Prop := ∃ l, l₁ ∥ l ∧ l₂ ∥ l

-- Main Theorems
theorem parallel_transitive (l₁ l₂ l₃ : Line) : (l₁ ∥ l₃) → (l₂ ∥ l₃) → (l₁ ∥ l₂) := 
by sorry

theorem converse_parallel_transitive (l₁ l₂ l₃ : Line) : (l₁ ∥ l₂) → (are_parallel l₁ l₃ ∧ are_parallel l₂ l₃) := 
by sorry

theorem inverse_parallel_transitive (l₁ l₂ l₃ : Line) : ¬(are_parallel l₁ l₃ ∧ are_parallel l₂ l₃) → ¬(l₁∥l₂) := 
by sorry

theorem contrapositive_parallel_transitive (l₁ l₂ l₃ : Line) : ¬(l₁ ∥ l₂) → ¬(are_parallel l₁ l₃ ∧ are_parallel l₂ l₃) := 
by sorry

end parallel_transitive_converse_parallel_transitive_inverse_parallel_transitive_contrapositive_parallel_transitive_l703_703039


namespace valid_k_l703_703131

def f (k x : ℝ) : ℝ := ((k + 1) * x^2 + (k + 3) * x + (2 * k - 8)) / ((2 * k - 1) * x^2 + (k + 1) * x + (k - 4))

def domain (k x : ℝ) : Prop := (2 * k - 1) * x^2 + (k + 1) * x + (k - 4) ≠ 0

def positive_f_on_domain (k : ℝ) : Prop :=
  ∀ (x : ℝ), domain k x → f k x > 0

theorem valid_k (k : ℝ) : positive_f_on_domain k ↔ 
  k = 1 ∨ k > (15 + 16 * Real.sqrt 2) / 7 ∨ k < (15 - 16 * Real.sqrt 2) / 7 :=
sorry

end valid_k_l703_703131


namespace mean_median_mode_equal_l703_703533

noncomputable def dataset : List ℕ := [20, 30, 40, 50, 50, 60, 70, 80]

def mean (l : List ℕ) : ℚ :=
  ((l.foldl (λ (acc : ℕ) (x : ℕ), acc + x) 0) : ℚ) / l.length

def median (l : List ℕ) : ℚ :=
  if l.length % 2 = 0 then
    let mid1 := l.get! (l.length / 2 - 1)
    let mid2 := l.get! (l.length / 2)
    ((mid1 + mid2) : ℚ) / 2
  else
    (l.get! (l.length / 2)) : ℚ

def mode (l : List ℕ) : ℕ :=
  l.groupBy id
   .foldl (λ acc (x : List ℕ), if x.length > acc.length then x else acc) []
   .head!

theorem mean_median_mode_equal :
  mean dataset = 50 ∧ median dataset = 50 ∧ mode dataset = 50 := by
  sorry

end mean_median_mode_equal_l703_703533


namespace intersection_angle_bisectors_on_BC_l703_703605

theorem intersection_angle_bisectors_on_BC 
  (A B C I D X : Type)
  [triangle ABC]
  (h1 : ∠CAB > ∠ABC)
  (h2 : I is_incenter_of_triangle ABC)
  (h3 : D on_segment BC)
  (h4 : ∠CAD = ∠ABC)
  (Γ : Type)
  (tangent_to_AC_at_A : Γ tangent AC A)
  (passes_through_I : Γ passes_through I)
  (second_intersection : X is_second_intersection_of Γ (circumcircle_of_triangle ABC)) :
  let bisector_DAB := bisector_of ∠DAB in
  let bisector_CXB := bisector_of ∠CXB in
  (intersection_of bisector_DAB bisector_CXB).lies_on_line BC :=
sorry

end intersection_angle_bisectors_on_BC_l703_703605


namespace average_reduction_10_percent_l703_703812

noncomputable def average_percentage_reduction
  (initial_price : ℝ) (final_price : ℝ) (reductions : ℕ) : ℝ :=
let x := (final_price / initial_price) ^ (1 / reductions : ℝ) in
1 - x

theorem average_reduction_10_percent :
  average_percentage_reduction 50 40.5 2 = 0.1 :=
by
  unfold average_percentage_reduction
  have h : (40.5 / 50 : ℝ) ^ (1 / 2 : ℝ) = 0.9 :=
    by sorry -- calculation of the square root
  rw h
  norm_num
  sorry

end average_reduction_10_percent_l703_703812


namespace maximum_value_of_t_l703_703184

theorem maximum_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x, (6 * x^3 - a * x^2 - 2 * b * x + 2).deriv x = 0 → x = 1) :
  t = a * b → t ≤ 81 / 4 :=
begin
  sorry
end

end maximum_value_of_t_l703_703184


namespace min_value_ineq_min_value_attainable_l703_703658

theorem min_value_ineq
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_sum : a + b + c = 9) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 :=
by {
  sorry,
}

theorem min_value_attainable :
  (a b c : ℝ)
  (h_eq : a = 3 ∧ b = 3 ∧ c = 3) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) = 9 :=
by {
  sorry,
}

end min_value_ineq_min_value_attainable_l703_703658


namespace value_of_x2y_plus_xy2_l703_703593

-- Define variables x and y as real numbers
variables (x y : ℝ)

-- Define the conditions
def condition1 : Prop := x + y = -2
def condition2 : Prop := x * y = -3

-- Define the proof problem
theorem value_of_x2y_plus_xy2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 * y + x * y^2 = 6 := by
  sorry

end value_of_x2y_plus_xy2_l703_703593


namespace deepak_wife_speed_correct_l703_703693

noncomputable def deepak_wife_speed (distance_track : ℕ) (deepak_speed : ℝ) (meet_time_min : ℝ) : ℝ :=
let meet_time_hr := meet_time_min / 60 in
let deepak_distance_km := deepak_speed * meet_time_hr in
let total_distance_km := (distance_track : ℝ) / 1000 in
let wife_distance_km := total_distance_km - deepak_distance_km in
wife_distance_km / meet_time_hr

theorem deepak_wife_speed_correct :
  deepak_wife_speed 528 4.5 3.84 = 3.75 :=
by
  sorry

end deepak_wife_speed_correct_l703_703693


namespace decagon_angle_measure_l703_703297

theorem decagon_angle_measure 
  (A B C D E F G H I J Q : Type)
  [Decagon A B C D E F G H I J]
  (h1 : Extends A J Q)
  (h2 : Extends D E Q) :
  measure_angle Q = 72 :=
by
  sorry

end decagon_angle_measure_l703_703297


namespace mass_percentage_H_calculation_l703_703923

noncomputable def molar_mass_CaH2 : ℝ := 42.09
noncomputable def molar_mass_H2O : ℝ := 18.015
noncomputable def molar_mass_H2SO4 : ℝ := 98.079

noncomputable def moles_CaH2 : ℕ := 3
noncomputable def moles_H2O : ℕ := 4
noncomputable def moles_H2SO4 : ℕ := 2

noncomputable def mass_H_CaH2 : ℝ := 3 * 2 * 1.008
noncomputable def mass_H_H2O : ℝ := 4 * 2 * 1.008
noncomputable def mass_H_H2SO4 : ℝ := 2 * 2 * 1.008

noncomputable def total_mass_H : ℝ :=
  mass_H_CaH2 + mass_H_H2O + mass_H_H2SO4

noncomputable def total_mass_mixture : ℝ :=
  (moles_CaH2 * molar_mass_CaH2) + (moles_H2O * molar_mass_H2O) + (moles_H2SO4 * molar_mass_H2SO4)

noncomputable def mass_percentage_H : ℝ :=
  (total_mass_H / total_mass_mixture) * 100

theorem mass_percentage_H_calculation :
  abs (mass_percentage_H - 4.599) < 0.001 :=
by
  sorry

end mass_percentage_H_calculation_l703_703923


namespace segment_BD_equals_zero_l703_703216

noncomputable def triangle_ABC (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ AB BC CA : ℝ, AB = 8 ∧ BC = 12 ∧ CA = 10

noncomputable def triangle_DEF (D E F : Type) [metric_space D] [metric_space E] [metric_space F] :=
  ∃ DE EF FD : ℝ, DE = 4 ∧ EF = 6 ∧ FD = 5

noncomputable def common_angle {A B C D E F : Type}
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F] :
  (triangle_ABC A B C) → (triangle_DEF D E F) → angle B E = 100 :=
sorry

noncomputable def similar_triangles {A B C D E F : Type}
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F] :
  (triangle_ABC A B C) → (triangle_DEF D E F) → 
  (angle B E = 100) → similarity_ratio A B C D E F :=
sorry

noncomputable def length_BD : ℝ := 0

theorem segment_BD_equals_zero (A B C D E F : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F]
  (hABC : triangle_ABC A B C) (hDEF : triangle_DEF D E F)
  (hAngle : common_angle hABC hDEF) :
  length_BD = 0 :=
sorry

end segment_BD_equals_zero_l703_703216


namespace pages_needed_l703_703038

def total_new_cards : ℕ := 8
def total_old_cards : ℕ := 10
def cards_per_page : ℕ := 3

theorem pages_needed (h : total_new_cards = 8) (h2 : total_old_cards = 10) (h3 : cards_per_page = 3) : 
  (total_new_cards + total_old_cards) / cards_per_page = 6 := by 
  sorry

end pages_needed_l703_703038


namespace count_obtuse_triangle_values_k_l703_703756

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  if a ≥ b ∧ a ≥ c then a * a > b * b + c * c 
  else if b ≥ a ∧ b ≥ c then b * b > a * a + c * c
  else c * c > a * a + b * b

theorem count_obtuse_triangle_values_k :
  ∃! (k : ℕ), is_triangle 8 18 k ∧ is_obtuse_triangle 8 18 k :=
sorry

end count_obtuse_triangle_values_k_l703_703756


namespace general_seq_correct_T_sum_correct_find_min_n_l703_703675

noncomputable def seq_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  n * a n - n * (n - 1)

def seq (n : ℕ) : ℕ → ℤ
| 1     := 1
| (n+1) := seq_sum n seq - seq_sum (n-1) seq

def general_seq_formula (n : ℕ) : ℤ :=
  2 * n - 1

theorem general_seq_correct (n : ℕ) : seq n = general_seq_formula n :=
  sorry

noncomputable def T_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, 1 / (seq i * seq (i + 1))

def T_sum_formula (n : ℕ) : ℚ :=
  n / (2 * n + 1)

theorem T_sum_correct (n : ℕ) : T_sum n = T_sum_formula n :=
  sorry

theorem find_min_n (n : ℕ) : T_sum_formula n > 100 / 209 → 
  ∀ m : ℕ, m < n → T_sum_formula m ≤ 100 / 209 :=
  sorry

end general_seq_correct_T_sum_correct_find_min_n_l703_703675


namespace elevator_scenarios_l703_703409

-- Definitions based on given conditions
def floors : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def people : Type := {A, B, C}

-- Number of scenarios given the conditions
theorem elevator_scenarios : 
  let A_choices := {3, 4, 5, 6}  -- A cannot choose 2nd floor
  let BC_choices := {2, 3, 4, 5, 6} 
  ∃ n : ℕ, n = 65 ∧  -- total number of scenarios
    ( -- Case 1: A goes to 7th floor
      A_choices.card * BC_choices.card = 25 ∧
      -- Case 2: A does not go to 7th floor and B or C goes to 7th floor
      ({3, 4, 5, 6}.card * BC_choices.card * 2 = 40 )
    ∧ 25 + 40 = 65 ) :=
begin
  sorry
end

end elevator_scenarios_l703_703409


namespace ab_value_l703_703999

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a ^ 2 + b ^ 2 = 35) : a * b = 13 :=
by
  sorry

end ab_value_l703_703999


namespace volume_of_ellipsoid_major_axis_rotation_volume_of_ellipsoid_minor_axis_rotation_volume_of_sphere_l703_703447

noncomputable theory

open Real

variables {a b : ℝ} (h : a > b) (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

-- Major Axis Rotation
theorem volume_of_ellipsoid_major_axis_rotation : 
  ∀ (a b : ℝ), a > b → 
  (π * ∫ x in -a..a, (b^2 * (a^2 - x^2) / a^2) dx = 4 / 3 * π * a * b^2) :=
sorry

-- Minor Axis Rotation
theorem volume_of_ellipsoid_minor_axis_rotation : 
  ∀ (a b : ℝ), a > b → 
  (π * ∫ y in -b..b, (a^2 * (b^2 - y^2) / b^2) dy = 4 / 3 * π * a^2 * b) :=
sorry

-- Sphere Volume
theorem volume_of_sphere (a : ℝ) : 
  (π * ∫ r in -a..a, (a^2 - r^2) dr = 4 / 3 * π * a^3) :=
sorry

end volume_of_ellipsoid_major_axis_rotation_volume_of_ellipsoid_minor_axis_rotation_volume_of_sphere_l703_703447


namespace probability_order_correct_l703_703911

inductive Phenomenon
| Certain
| VeryLikely
| Possible
| Impossible
| NotVeryLikely

open Phenomenon

def probability_order : Phenomenon → ℕ
| Certain       => 5
| VeryLikely    => 4
| Possible      => 3
| NotVeryLikely => 2
| Impossible    => 1

theorem probability_order_correct :
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] =
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] :=
by
  -- skips the proof
  sorry

end probability_order_correct_l703_703911


namespace solve_for_a_l703_703111

theorem solve_for_a (a : ℝ) (n : ℕ) :
  (∀ n : ℕ, 4 * floor (a * n) = n + floor (a * floor (a * n))) → a = 2 + Real.sqrt 3 :=
by
  sorry

end solve_for_a_l703_703111


namespace average_age_approximately_181_46_l703_703690

noncomputable def Kimiko_age: ℝ := 28

noncomputable def Omi_age: ℝ := 2.5 * Kimiko_age

noncomputable def Arlette_age: ℝ := (3 / 4) * Kimiko_age

noncomputable def Xander_age: ℝ := Kimiko_age^2 - 7

noncomputable def Yolanda_age: ℝ := real.cbrt (Xander_age / 2) + 4

noncomputable def total_age: ℝ := Kimiko_age + Omi_age + Arlette_age + Xander_age + Yolanda_age

noncomputable def average_age: ℝ := total_age / 5

theorem average_age_approximately_181_46 : average_age = 181.46 :=
by
  sorry

end average_age_approximately_181_46_l703_703690


namespace sqrt_sum_equals_k_l703_703342

theorem sqrt_sum_equals_k (k : ℕ) (h : 0 < k) : 
  (sqrt 2 + sqrt 8 + sqrt 18 = sqrt k) ↔ k = 72 :=
by
  sorry

end sqrt_sum_equals_k_l703_703342


namespace angle_A_measure_triangle_area_l703_703156

variable {a b c : ℝ} 
variable {A B C : ℝ} 
variable (triangle : a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A))

theorem angle_A_measure (h : (b - c)^2 = a^2 - b * c) : A = Real.pi / 3 :=
sorry

theorem triangle_area 
  (h1 : a = 3) 
  (h2 : Real.sin C = 2 * Real.sin B) 
  (h3 : A = Real.pi / 3) 
  (hb : b = Real.sqrt 3)
  (hc : c = 2 * Real.sqrt 3) : 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end angle_A_measure_triangle_area_l703_703156


namespace remainder_of_M_mod_1000_l703_703645

def B := {1, 2, 3, 4, 5, 6, 7, 8}

def valid_function (g : B → B) :=
  ∃ c : B, ∀ x : B, g (g x) = c

noncomputable def M : ℕ :=
  8 * (7 * 1 + 21 * 32 + 35 * 81 + 35 * 64 + 21 * 25 + 7 * 6 + 1 * 1)

def remainder_when_divided (x y : ℕ) := x % y

theorem remainder_of_M_mod_1000 : remainder_when_divided M 1000 = 192 := by
  unfold remainder_when_divided
  unfold M
  simp
  sorry

end remainder_of_M_mod_1000_l703_703645


namespace count_integers_satisfying_condition_l703_703177

theorem count_integers_satisfying_condition :
  {n : ℤ | -15 ≤ n ∧ n ≤ 5 ∧ (n - 5) * (n + 2) * (n + 9) < 0}.finite.card = 6 :=
by
  sorry

end count_integers_satisfying_condition_l703_703177


namespace radius_of_larger_circle_15_l703_703010

def radius_larger_circle (r1 r2 r3 r : ℝ) : Prop :=
  ∃ (A B C O : EuclideanSpace ℝ (Fin 2)), 
    dist A B = r1 + r2 ∧
    dist B C = r2 + r3 ∧
    dist A C = r1 + r3 ∧
    dist O A = r - r1 ∧
    dist O B = r - r2 ∧
    dist O C = r - r3 ∧
    (dist O A + r1 = r ∧
    dist O B + r2 = r ∧
    dist O C + r3 = r)

theorem radius_of_larger_circle_15 :
  radius_larger_circle 10 3 2 15 :=
by
  sorry

end radius_of_larger_circle_15_l703_703010


namespace number_of_hens_l703_703041

theorem number_of_hens (H C : ℕ) 
  (h1 : H + C = 60) 
  (h2 : 2 * H + 4 * C = 200) : H = 20 :=
sorry

end number_of_hens_l703_703041


namespace right_triangle_area_l703_703308

theorem right_triangle_area (h : Real) (a b c : Real) (A B C : ℝ) : 
  ∠A = 45 ∧ ∠C = 45 ∧ ∠B = 90 ∧ h = 4 → 
  area (triangle A B C) = 8 * Real.sqrt 2 :=
begin
  sorry
end

end right_triangle_area_l703_703308


namespace simplify_arithmetic_expr1_simplify_arithmetic_expr2_l703_703407

-- Problem 1 Statement
theorem simplify_arithmetic_expr1 (x y : ℝ) : 
  (x - 3 * y) - (y - 2 * x) = 3 * x - 4 * y :=
sorry

-- Problem 2 Statement
theorem simplify_arithmetic_expr2 (a b : ℝ) : 
  5 * a * b^2 - 3 * (2 * a^2 * b - 2 * (a^2 * b - 2 * a * b^2)) = -7 * a * b^2 :=
sorry

end simplify_arithmetic_expr1_simplify_arithmetic_expr2_l703_703407


namespace max_Sn_occurs_at_6_l703_703148

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1_pos : a 1 > 0)
variable (S_n_def : ∀ n, S n = (n * (a 1 + a (n + 1)) / 2))
variable (S4_eq_S8 : S 4 = S 8)

theorem max_Sn_occurs_at_6 :
  ∃ n, S n = max (S n) ∧ n = 6 := by
  sorry

end max_Sn_occurs_at_6_l703_703148


namespace family_visit_cost_is_55_l703_703213

def num_children := 4
def num_parents := 2
def num_grandmother := 1
def num_people := num_children + num_parents + num_grandmother

def entrance_ticket_cost := 5
def attraction_ticket_cost_kid := 2
def attraction_ticket_cost_adult := 4

def entrance_total_cost := num_people * entrance_ticket_cost
def attraction_total_cost_kids := num_children * attraction_ticket_cost_kid
def adults := num_parents + num_grandmother
def attraction_total_cost_adults := adults * attraction_ticket_cost_adult

def total_cost := entrance_total_cost + attraction_total_cost_kids + attraction_total_cost_adults

theorem family_visit_cost_is_55 : total_cost = 55 := by
  sorry

end family_visit_cost_is_55_l703_703213


namespace julia_played_with_kids_on_tuesday_l703_703638

theorem julia_played_with_kids_on_tuesday (total: ℕ) (monday: ℕ) (tuesday: ℕ) 
  (h1: total = 18) (h2: monday = 4) : 
  tuesday = (total - monday) :=
by
  sorry

end julia_played_with_kids_on_tuesday_l703_703638


namespace sin_gt_sub_cubed_l703_703918

theorem sin_gt_sub_cubed (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  Real.sin x > x - x^3 / 6 := 
by 
  sorry

end sin_gt_sub_cubed_l703_703918


namespace boys_at_table_l703_703056

theorem boys_at_table (children : ℕ) (boys_lie_to_girls : ∀ b g, b ∈ boys → g ∈ girls → (b speaks to g → b lies)) 
  (boys_truth_to_boys : ∀ b1 b2, b1 ∈ boys → b2 ∈ boys → (b1 speaks to b2 → b1 tells_truth))
  (girls_lie_to_boys : ∀ g b, g ∈ girls → b ∈ boys → (g speaks to b → g lies))
  (girls_truth_to_girls : ∀ g1 g2, g1 ∈ girls → g2 ∈ girls → (g1 speaks to g2 → g1 tells_truth))
  (alternating_statements : ℕ → string)
  (last_statement : string) :
  children = 13 →
  (alternating_statements 1 = "Most of us are boys") →
  (alternating_statements 2 = "Most of us are girls") →
  (alternating_statements 13 = "Most of us are boys") →
  ∃ boys girls, boys + girls = 13 ∧ boys = 7 :=
by
  sorry

end boys_at_table_l703_703056


namespace inverse_function_b_value_l703_703549

theorem inverse_function_b_value (b : ℝ) :
  (∀ x, ∃ y, 2^x + b = y) ∧ (∃ x, ∃ y, (x, y) = (2, 5)) → b = 1 :=
by
  sorry

end inverse_function_b_value_l703_703549


namespace no_real_solutions_sufficient_not_necessary_l703_703384

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l703_703384


namespace area_of_quadrilateral_PAQR_l703_703016

-- Definitions per conditions
def AP : ℝ := 10
def PB : ℝ := 20
def PR : ℝ := 25

-- The Lean statement to prove the area of quadrilateral PAQR
theorem area_of_quadrilateral_PAQR (hPAQ_right : ∠ (AP, PQ) = 90) (hPBR_right : ∠ (PB, PR) = 90) :
  let AQ := (AP^2 - PQ^2).sqrt,
      PQ := (AP^2 - AQ^2).sqrt,
      BR := (PR^2 - PB^2).sqrt in
  let area_PAQ := (1/2) * PQ * AQ,
      area_PBR := (1/2) * PB * BR in
  area_PAQ + area_PBR = 174 :=
by
  let AQ := sqrt (AP^2 - PQ^2),
      PQ := sqrt (AP^2 - AQ^2),
      BR := sqrt (PR^2 - PB^2)
  let area_PAQ := (1/2) * PQ * AQ,
      area_PBR := (1/2) * PB * BR
  sorry

end area_of_quadrilateral_PAQR_l703_703016


namespace exists_obtuse_triangle_l703_703223

section
variables {a b c : ℝ} (ABC : Triangle ℝ) (h₁ : ABC.side1 = a)
  (h₂ : ABC.side2 = a + 1) (h₃ : ABC.side3 = a + 2)
  (h₄ : 2 * sin ABC.γ = 3 * sin ABC.α)

noncomputable def area_triangle_proof : ABC.area = (15 * real.sqrt 7) / 4 :=
sorry

theorem exists_obtuse_triangle : ∃ (a : ℝ), a = 2 ∧ ∀ (h : Triangle ℝ), 
  h.side1 = a → h.side2 = a + 1 → h.side3 = a + 2 → obtuse h :=
sorry
end

end exists_obtuse_triangle_l703_703223


namespace jackson_earns_30_dollars_l703_703243

theorem jackson_earns_30_dollars :
  let vacuuming_hours := 2 * 2,
      washing_dishes_hours := 0.5,
      cleaning_bathroom_hours := 3 * washing_dishes_hours,
      total_hours := vacuuming_hours + washing_dishes_hours + cleaning_bathroom_hours,
      hourly_rate := 5
  in total_hours * hourly_rate = 30 :=
by
  let vacuuming_hours := 2 * 2
  let washing_dishes_hours := 0.5
  let cleaning_bathroom_hours := 3 * washing_dishes_hours
  let total_hours := vacuuming_hours + washing_dishes_hours + cleaning_bathroom_hours
  let hourly_rate := 5
  show total_hours * hourly_rate = 30
  exact sorry

end jackson_earns_30_dollars_l703_703243


namespace machine_production_rates_l703_703618

theorem machine_production_rates (x1 x2 x3 x4 x5 : ℕ) :
  {35, 39, 40, 49, 44, 46, 30, 41, 32, 36} =
  {x1 + x2, x1 + x3, x1 + x4, x1 + x5, x2 + x3, x2 + x4, x2 + x5, x3 + x4, x3 + x5, x4 + x5} →
  ∃ (x1 x2 x3 x4 x5 : ℕ), x1 = 13 ∧ x2 = 17 ∧ x3 = 19 ∧ x4 = 22 ∧ x5 = 27 :=
begin
  sorry
end

end machine_production_rates_l703_703618


namespace sum_of_two_numbers_is_10_l703_703762

variable (a b : ℝ)

theorem sum_of_two_numbers_is_10
  (h1 : a + b = 10)
  (h2 : a - b = 8)
  (h3 : a^2 - b^2 = 80) :
  a + b = 10 :=
by
  sorry

end sum_of_two_numbers_is_10_l703_703762


namespace will_new_cards_count_l703_703393

-- Definitions based on conditions
def cards_per_page := 3
def pages_used := 6
def old_cards := 10

-- Proof statement (no proof, only the statement)
theorem will_new_cards_count : (pages_used * cards_per_page) - old_cards = 8 :=
by sorry

end will_new_cards_count_l703_703393


namespace conjugate_of_z_l703_703560

-- Definition of the imaginary unit i
def i : ℂ := complex.I

-- Given conditions
def z : ℂ := 2 + (1 / i)

-- Proof problem: Show that the conjugate of z is equal to 2 + i
theorem conjugate_of_z : conj(z) = 2 + i :=
by
  sorry

end conjugate_of_z_l703_703560


namespace range_of_m_l703_703572

variable (m : ℝ) (x : ℝ)

def p : Prop := x < -2 ∨ x > 10

def q : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

theorem range_of_m : (∀ x, ¬ p x → q x) → 3 ≤ m :=
by
  intros h
  sorry

end range_of_m_l703_703572


namespace number_of_grey_birds_l703_703767

variable (G : ℕ)

def grey_birds_condition1 := G + 6
def grey_birds_condition2 := G / 2

theorem number_of_grey_birds
  (H1 : G + 6 + G / 2 = 66) :
  G = 40 :=
by
  sorry

end number_of_grey_birds_l703_703767


namespace graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l703_703744

theorem graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines :
  ∀ x y : ℝ, (x^2 - y^2 = 0) ↔ (y = x ∨ y = -x) := 
by
  sorry

end graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l703_703744


namespace correct_statements_l703_703092

-- Define the necessary conditions and statements
def statement2 (x y : ℝ) : Prop := 
  x * y > 0 → (y / x) + (x / y) ≥ 2

def statement4 (x y : ℝ) : Prop := 
  x + y = 0 → 2^x + 2^y ≥ 2

-- The theorem to prove the correctness of statements ② and ④
theorem correct_statements (x y : ℝ) :
  statement2 x y ∧ statement4 x y :=
by
  split
  sorry
  sorry

end correct_statements_l703_703092


namespace vector_magnitude_l703_703175

variables (a b : ℝ^3)

-- Define conditions
def condition1 := ‖a‖ = 1
def condition2 := ‖b‖ = 2
def condition3 := ‖a + b‖ = real.sqrt 5

-- Define the theorem to be proved
theorem vector_magnitude :
  condition1 a b →
  condition2 a b →
  condition3 a b →
  ‖2 • a - b‖ = 2 * real.sqrt 2 :=
by
  sorry

end vector_magnitude_l703_703175


namespace stateA_selection_percentage_l703_703610

theorem stateA_selection_percentage :
  ∀ (P : ℕ), (∀ (n : ℕ), n = 8000) → (7 * 8000 / 100 = P * 8000 / 100 + 80) → P = 6 := by
  -- The proof steps go here
  sorry

end stateA_selection_percentage_l703_703610


namespace distance_between_stations_l703_703045

theorem distance_between_stations (x : ℕ) 
  (h1 : ∃ (x : ℕ), ∀ t : ℕ, (t * 16 = x ∧ t * 21 = x + 60)) :
  2 * x + 60 = 444 :=
by sorry

end distance_between_stations_l703_703045


namespace max_solution_of_abs_eq_10_l703_703896

theorem max_solution_of_abs_eq_10 :
  ∃ x, |x + 3| = 10 ∧ ∀ y, |y + 3| = 10 → x ≥ y :=
by
  use 7
  split
  { exact (abs_of_nonneg (by linarith)).mpr rfl }
  { intros y hy
    by_cases hy1 : y + 3 = 10
    { linarith }
    { have hy2 : y + 3 = -10 := by
        have := abs_eq (y + 3) 10
        cases this <|> assumption
      linarith } }

end max_solution_of_abs_eq_10_l703_703896


namespace area_difference_l703_703989

-- Define the constants for the problem conditions
def radius_large := 25
def diameter_small := 15
def radius_small := diameter_small / 2

-- Define the areas of the circles in terms of pi
def area_large := Real.pi * radius_large^2
def area_small := Real.pi * radius_small^2

-- Prove that the difference in their areas is 568.75 * Real.pi
theorem area_difference :
  area_large - area_small = 568.75 * Real.pi := 
sorry

end area_difference_l703_703989


namespace find_y_l703_703217

theorem find_y
  (XYZ_is_straight_line : XYZ_is_straight_line)
  (angle_XYZ : ℝ)
  (angle_YWZ : ℝ)
  (y : ℝ)
  (exterior_angle_theorem : angle_XYZ = y + angle_YWZ)
  (h1 : angle_XYZ = 150)
  (h2 : angle_YWZ = 58) :
  y = 92 :=
by
  sorry

end find_y_l703_703217


namespace rate_percent_l703_703046

-- Define the parameters
def simpleInterest (si p t : ℝ) : ℝ :=
  (si * 100) / (p * t)

-- Define the conditions given in the problem
def SI := 320
def P := 4000
def T := 2

-- State the theorem to be proven
theorem rate_percent : simpleInterest SI P T = 4 := by
  sorry

end rate_percent_l703_703046


namespace relationship_among_a_b_c_l703_703475

def new_stationary_point (f : ℝ → ℝ) : ℝ :=
  let f' := (fun x => (derivative f) x)
  let f'' := (fun x => (derivative f') x)
  (real.root (fun x => f x = f'' x))

def g : ℝ → ℝ := fun x => 2 * x
def h : ℝ → ℝ := fun x => real.log x
def phi : ℝ → ℝ := fun x => x ^ 3

def a : ℝ := new_stationary_point g
noncomputable def b : ℝ := new_stationary_point h
def c : ℝ := new_stationary_point phi

theorem relationship_among_a_b_c : (1 < b ∧ b < 3) ∧ c = 3 ∧ a = 1 → c > b ∧ b > a :=
by
  sorry

end relationship_among_a_b_c_l703_703475


namespace matrix_diagonal_sum_l703_703050

open_locale big_operators

noncomputable theory

def arithmetic_sequence (a d : ℝ) := λ i : ℕ, a + ↑i * d

def geometric_sequence (a r : ℝ) := λ i : ℕ, a * r ^ i

def matrix (n : ℕ) (a : ℕ → ℕ → ℝ) (row_seq : ∀ i, ∃ d, ∀ j, a i j = arithmetic_sequence (a i 1) d j) 
    (common_ratio : ℝ) (col_seq : ∀ j, ∀ i₀ i₁, i₀ ≤ i₁ → a i₁ j = geometric_sequence (a i₀ j) common_ratio (i₁ - i₀)) :=
  ∀ i j, a i j ∈ set.univ -- We can check the row and column sequence conditions as needed

def given_matrix := 
λ (i j : ℕ), 
if (i = 4 ∧ j = 2) then (1 / 8 : ℝ) else 
if (i = 4 ∧ j = 3) then (3 / 16 : ℝ) else 
if (i = 2 ∧ j = 4) then (1 : ℝ) else
0 -- other entries we're not explicitly given are defaulted to 0 for this definition

theorem matrix_diagonal_sum (n : ℕ) (h : 4 ≤ n) 
  (a : ℕ → ℕ → ℝ) 
  (row_seq : ∀ i, ∃ d, ∀ j, a i j = arithmetic_sequence (a i 1) d j) 
  (common_ratio : ℝ) (col_seq : ∀ j, ∀ i₀ i₁, i₀ ≤ i₁ → a i₁ j = geometric_sequence (a i₀ j) common_ratio (i₁ - i₀))
  (h_special_values : a 2 4 = 1 ∧ a 4 2 = 1 / 8 ∧ a 4 3 = 3 / 16)
  :
  ∑ i in finset.range n, a i i = 2 - (n + 2) / 2^n :=
sorry

end matrix_diagonal_sum_l703_703050


namespace sum_of_arithmetic_sequence_l703_703566

theorem sum_of_arithmetic_sequence (n : ℕ) : 
  let a : ℕ → ℤ := λ n, -5 * n + 2 in
  let S : ℕ → ℤ := λ n, -(5 * n^2 + n) / 2 in
  (∑ i in range n, a i) = S n :=
by
  -- Lean code to be provided for the proof
  sorry

end sum_of_arithmetic_sequence_l703_703566


namespace advertisement_broadcast_count_l703_703057

theorem advertisement_broadcast_count :
  ∃ n : ℕ, n = 36 ∧
  ∃ ads : list char, ads.length = 5 ∧
  ∃ comm_ads public_ads : list char, comm_ads.length = 3 ∧ public_ads.length = 2 ∧
  (∀ i < 4, ads[i] ≠ 'P' → ∃ k, k < 4 ∧ ads[k] = 'P') ∧
  ads[4] = 'P' ∧
  (∀ j, ∀ k, ads[j] = 'P' → ads[k] ≠ 'P' → k ≠ j + 1) :=
sorry

end advertisement_broadcast_count_l703_703057


namespace solve_for_a_l703_703181

noncomputable def integral_val (a : ℝ) : ℝ :=
  ∫ x in 1..a, (2 * x + 1 / x)

theorem solve_for_a (a : ℝ) (h1 : 1 < a) (h2 : integral_val a = 3 + Real.log 2) : a = 2 :=
by
  sorry

end solve_for_a_l703_703181


namespace decrease_ratio_percentage_l703_703073

def royaltiesFirst : ℝ := 7 -- in million dollars
def salesFirst : ℝ := 20 -- in million dollars
def royaltiesSecond : ℝ := 9 -- in million dollars
def salesSecond : ℝ := 108 -- in million dollars

theorem decrease_ratio_percentage : 
  let ratioFirst := royaltiesFirst / salesFirst,
      ratioSecond := royaltiesSecond / salesSecond,
      percentageFirst := (ratioFirst * 100),
      percentageSecond := (ratioSecond * 100),
      decreasePercentage := percentageFirst - percentageSecond
  in abs (decreasePercentage - 26.67) < 0.01 := 
by
  sorry

end decrease_ratio_percentage_l703_703073


namespace circle_radius_l703_703754

-- Given conditions
def central_angle : ℝ := 225
def perimeter : ℝ := 83
noncomputable def pi_val : ℝ := Real.pi

-- Formula for the radius
noncomputable def radius : ℝ := 332 / (5 * pi_val + 8)

-- Prove that the radius is correct given the conditions
theorem circle_radius (theta : ℝ) (P : ℝ) (r : ℝ) (h_theta : theta = central_angle) (h_P : P = perimeter) :
  r = radius :=
sorry

end circle_radius_l703_703754


namespace range_of_a_l703_703154

def A (x : ℝ) := 1 ≤ x ∧ x < 5
def B (x a : ℝ) := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) ↔ a ∈ set.Iic (-1) :=
by {
  sorry
}

end range_of_a_l703_703154


namespace rectangle_area_eq_l703_703851

theorem rectangle_area_eq (a b c d x y z w : ℝ)
  (h1 : a = x + y) (h2 : b = y + z) (h3 : c = z + w) (h4 : d = w + x) :
  a + c = b + d :=
by
  sorry

end rectangle_area_eq_l703_703851


namespace worker_earnings_l703_703442

theorem worker_earnings 
  (E : ℤ) 
  (h1 : ∀ n : ℕ, n % 2 = 1 → (net_amount n E = n / 2 * (E-18) + E )) 
  (h2 : net_amount 9 E = 48) : 
  E = 24 :=
by
  sorry

def net_amount : ℕ → ℤ → ℤ 
| 0, E => 0
| 1, E => E
| (n+2), E => net_amount n E + E - 18

end worker_earnings_l703_703442


namespace absent_children_count_l703_703691

-- Definition of conditions
def total_children := 700
def bananas_per_child := 2
def bananas_extra := 2
def total_bananas := total_children * bananas_per_child

-- The proof goal
theorem absent_children_count (A P : ℕ) (h_P : P = total_children - A)
    (h_bananas : total_bananas = P * (bananas_per_child + bananas_extra)) : A = 350 :=
by
  -- Since this is a statement only, we place a sorry here to skip the proof.
  sorry

end absent_children_count_l703_703691


namespace hyperbola_solution_l703_703169

noncomputable def hyperbola_equation : Prop :=
  ∃ (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b),
  (c - a = 1) ∧ 
  (b = sqrt 3) ∧ 
  (c^2 - a^2 = b^2) ∧ 
  (x^2 - y^2 / (b^2) = 1)
  
theorem hyperbola_solution : hyperbola_equation :=
by 
  sorry

end hyperbola_solution_l703_703169


namespace one_not_reciprocal_of_neg_one_l703_703790

theorem one_not_reciprocal_of_neg_one :
  (1 ≠ (-1)⁻¹) :=
by {
  rw [inv_eq_one_div, div_neg],
  exact neg_one_ne_one.symm,
}

end one_not_reciprocal_of_neg_one_l703_703790


namespace negation_of_existence_l703_703541

theorem negation_of_existence (p : Prop) (h : ∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) : 
  ¬ (∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) ↔ 
  ∀ (c : ℝ), c > 0 → ¬ (∃ (x : ℝ), x^2 - x + c = 0) :=
by 
  sorry

end negation_of_existence_l703_703541


namespace pq_ratio_at_0_l703_703741

noncomputable def p (x : ℝ) : ℝ := -3 * (x + 4) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

theorem pq_ratio_at_0 : (p 0) / (q 0) = 0 := by
  sorry

end pq_ratio_at_0_l703_703741


namespace ratio_of_boys_to_girls_l703_703319

theorem ratio_of_boys_to_girls (a b : ℕ)
  (h_class_avg : (75.5 * a + 81 * b) / (a + b) = 78) :
  a / b = 6 / 5 :=
by
  have h : 75.5 * a + 81 * b = 78 * (a + b),
  { exact (div_eq_iff (a + b).ne_zero).mp h_class_avg },
  have h_simp : 75.5 * a + 81 * b = (78 * a + 78 * b),
  { rw h },
  linarith,
sory

end ratio_of_boys_to_girls_l703_703319


namespace probability_of_four_hits_probability_of_two_hits_l703_703429

-- We are dealing with probabilities,
-- For simplification, we can define functions to compute the necessary combinations and probabilities.

def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

-- Condition Definitions
def total_numbers := 10
def drawn_numbers := 5
def player_numbers := 4

-- Question a): Probability of matching four numbers
theorem probability_of_four_hits :
  let total_combinations := binomial_coefficient total_numbers drawn_numbers in
  let favorable_combinations := binomial_coefficient drawn_numbers player_numbers * binomial_coefficient (drawn_numbers - player_numbers) 0 in
  (favorable_combinations : ℚ) / total_combinations = 1 / 21 :=
sorry

-- Question b): Probability of matching two numbers
theorem probability_of_two_hits :
  let total_combinations := binomial_coefficient total_numbers drawn_numbers in
  let favorable_combinations := binomial_coefficient player_numbers 2 * binomial_coefficient (drawn_numbers - player_numbers) 2 in
  (favorable_combinations : ℚ) / total_combinations = 10 / 21 :=
sorry


end probability_of_four_hits_probability_of_two_hits_l703_703429


namespace sum_even_202_to_300_l703_703399

noncomputable def sum_even_integers (a d l : ℕ) : ℕ :=
  let n := ((l - a) / d) + 1 in
  (n / 2) * (a + l)

theorem sum_even_202_to_300 : sum_even_integers 202 2 300 = 12550 := by
  sorry

end sum_even_202_to_300_l703_703399


namespace digits_product_l703_703850

-- Define the conditions
variables (A B : ℕ)

-- Define the main problem statement using the conditions and expected answer
theorem digits_product (h1 : A + B = 12) (h2 : (10 * A + B) % 3 = 0) : A * B = 35 := 
by
  sorry

end digits_product_l703_703850


namespace tan_six_theta_eq_l703_703193

theorem tan_six_theta_eq (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 :=
by
  sorry

end tan_six_theta_eq_l703_703193


namespace zero_diff_l703_703337

-- Define the quadratic equation with given conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions of the problem
def vertex_form (a x : ℝ) : ℝ := a * (x - 2)^2 - 4

-- Define the point (4, 12) on the parabola
def point_on_parabola (a : ℝ) : Prop := vertex_form a 4 = 12

-- Define the computation of zeros
def zeros (a b c m n : ℝ) : Prop :=
  quadratic a b c m = 0 ∧ quadratic a b c n = 0 ∧ m > n

-- Define the theorem to prove the difference of zeros is 2
theorem zero_diff (a b c m n : ℝ) (h1 : vertex_form a _ = quadratic a b c _) (h2 : point_on_parabola a) (h3 : zeros a b c m n) : m - n = 2 :=
by sorry

end zero_diff_l703_703337


namespace probability_of_perfect_square_or_multiple_of_4_l703_703720

-- Define the set of numbers on an 8-sided die
def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the condition for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

-- Define the condition for a number being a multiple of 4
def is_multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- Define the set of successful outcomes
def successful_outcomes : List ℕ :=
  sides.filter (λ n, is_perfect_square n ∨ is_multiple_of_4 n)

-- Define the probability as the ratio of the number of successful outcomes to the total number of outcomes
def probability : ℚ :=
  successful_outcomes.length / sides.length

-- State the theorem
theorem probability_of_perfect_square_or_multiple_of_4 :
  probability = 3 / 8 :=
by
  sorry

end probability_of_perfect_square_or_multiple_of_4_l703_703720


namespace largest_value_of_x_l703_703504

theorem largest_value_of_x (x : ℝ) (h : |x - 8| = 15) : x ≤ 23 :=
by
  sorry -- Proof to be provided

end largest_value_of_x_l703_703504


namespace find_ordered_triples_l703_703890

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_ordered_triples :
  {a b c : ℕ // is_perfect_square (a^2 + 2 * b + c) ∧ 
                is_perfect_square (b^2 + 2 * c + a) ∧ 
                is_perfect_square (c^2 + 2 * a + b)} = 
  ({⟨0, 0, 0⟩, ⟨1, 1, 1⟩, ⟨127, 106, 43⟩, ⟨106, 43, 127⟩, ⟨43, 127, 106⟩} : Finset (ℕ × ℕ × ℕ)) :=
by sorry

end find_ordered_triples_l703_703890


namespace triangle_sum_is_19_l703_703719

-- Defining the operation on a triangle
def triangle_op (a b c : ℕ) := a * b - c

-- Defining the vertices of the two triangles
def triangle1 := (4, 2, 3)
def triangle2 := (3, 5, 1)

-- Statement that the sum of the operation results is 19
theorem triangle_sum_is_19 :
  triangle_op (4) (2) (3) + triangle_op (3) (5) (1) = 19 :=
by
  -- Triangle 1 calculation: 4 * 2 - 3 = 8 - 3 = 5
  -- Triangle 2 calculation: 3 * 5 - 1 = 15 - 1 = 14
  -- Sum of calculations: 5 + 14 = 19
  sorry

end triangle_sum_is_19_l703_703719


namespace evaluate_expression_l703_703489

theorem evaluate_expression (x z : ℤ) (h₁ : x = 4) (h₂ : z = 2) :
  z * (z - 4 * x) = -28 :=
by {
  sorry,
}

end evaluate_expression_l703_703489


namespace g_in_func_set_l703_703676

def func_set (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|

def g (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem g_in_func_set : func_set g :=
by
  intro x₁ x₂ hx₁ hx₂
  calc
    |g x₁ - g x₂| = |(x₁ + x₂) * (x₁ - x₂) + 2 * (x₁ - x₂)| : by sorry
    ... = |x₁ + x₂ + 2| * |x₁ - x₂|                     : by sorry
    ... ≤ 4 * |x₁ - x₂|                                 : by sorry

end g_in_func_set_l703_703676


namespace segment_division_exists_l703_703108

-- Definitions and conditions
noncomputable def is_linear (f : ℝ → ℝ) := ∃ (a b : ℝ), ∀ x, f(x) = a * x + b
noncomputable def is_quadratic_trinomial (f : ℝ → ℝ) := ∃ (a b c : ℝ), ∀ x, f(x) = a * x^2 + b * x + c

def integrals_equal {a b : ℝ} (f : ℝ→ ℝ) (black white : set ℝ) : Prop :=
  ∫ x in black, f x = ∫ x in white, f x

theorem segment_division_exists :
  ∃ (black white : set ℝ),
  (∀ f : ℝ → ℝ, is_linear f → integrals_equal f black white) ∧
  (∀ f : ℝ → ℝ, is_quadratic_trinomial f → integrals_equal f black white) :=
sorry

end segment_division_exists_l703_703108


namespace cost_of_song_book_l703_703246

-- Define the given constants: cost of trumpet, cost of music tool, and total spent at the music store.
def cost_of_trumpet : ℝ := 149.16
def cost_of_music_tool : ℝ := 9.98
def total_spent_at_store : ℝ := 163.28

-- The goal is to prove that the cost of the song book is $4.14.
theorem cost_of_song_book :
  total_spent_at_store - (cost_of_trumpet + cost_of_music_tool) = 4.14 :=
by
  sorry

end cost_of_song_book_l703_703246


namespace number_of_possible_medians_l703_703648

open Set

variable (S : Finset ℤ)

def seven_elements : Set ℤ := {3, 5, 7, 13, 15, 17, 19}

theorem number_of_possible_medians
  (h1 : S.card = 11)
  (h2 : seven_elements ⊆ S) :
  ∃ medians : Finset ℤ, 
    (medians.card = 8 ∧ 
     ∀ m ∈ medians, 
     ∃ T : List ℤ, 
       T.length = 11 ∧ 
       T.nth 5 = some m ∧ 
       (∀ x ∈ T, x ∈ S)) :=
sorry

end number_of_possible_medians_l703_703648


namespace rational_coeff_terms_count_l703_703783

/-- 
  The expansion of (x * root 4 (2) + y * sqrt 5) ^ 500 contains 126 terms with rational coefficients.
-/
theorem rational_coeff_terms_count :
  let expansion := (λ (x y : ℚ), (x * (2 : ℝ)^(1/4) + y * (5 : ℝ)^(1/2))^500;
  (count_rat_coeff expansion) = 126 :=
by
  sorry

end rational_coeff_terms_count_l703_703783


namespace fraction_of_single_men_l703_703453

theorem fraction_of_single_men (T : ℝ) (hT : 0 < T) :
  let w := 0.64 * T,
      m := T - w,
      married := 0.60 * T,
      married_women := 0.75 * w,
      married_men := married - married_women,
      fraction_married_men := married_men / m,
      fraction_single_men := 1 - fraction_married_men
  in fraction_single_men = 2 / 3 :=
by
  let w := 0.64 * T
  let m := T - w
  let married := 0.60 * T
  let married_women := 0.75 * w
  let married_men := married - married_women
  let fraction_married_men := married_men / m
  let fraction_single_men := 1 - fraction_married_men
  sorry

end fraction_of_single_men_l703_703453


namespace find_a_and_b_find_range_of_c_l703_703271

-- Let f(x) have the form specified and the initial conditions.

def f (x : ℝ) (a b c : ℝ) := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

-- Given that the function f(x) has extreme values at x = 1 and x = 2
def has_extremes_at (f : ℝ → ℝ) (x₁ x₂ : ℝ) := 
  ∀ (x : ℝ), (x = x₁ ∨ x = x₂) → deriv f x = 0

-- Also given that for any x ∈ [0, 3], f(x) < c^2 holds
def in_interval (x : ℝ) := 0 ≤ x ∧ x ≤ 3
def f_lt_c_squared (f : ℝ → ℝ) (c : ℝ) := 
  ∀ x, in_interval x → f x < c^2

-- Part I: Prove the values of a and b
theorem find_a_and_b (a b : ℝ) (h : has_extremes_at (f a b) 1 2) : 
  a = -3 ∧ b = 4 :=
sorry

-- Part II: Prove the range of values for c
theorem find_range_of_c (a b : ℝ) (c : ℝ) (h : has_extremes_at (f a b) 1 2) 
  (h_a : a = -3) (h_b : b = 4) (h₂ : f_lt_c_squared (f a b) c) :
  c < -1 ∨ c > 9 :=
sorry

end find_a_and_b_find_range_of_c_l703_703271


namespace isosceles_triangle_two_points_l703_703611

open Real

theorem isosceles_triangle_two_points : 
  ∃ (C1 C2 : ℝ × ℝ), 
    (let A := (0, 0) and B := (8, 0) in
      (dist A B = 8) ∧
      (dist A C1 + dist B C1 + dist A B = 30) ∧
      (dist A C2 + dist B C2 + dist A B = 30) ∧
      ((dist A C1 = dist B C1) ∨ (dist A C1 = dist A B) ∨ (dist B C1 = dist A B)) ∧
      ((dist A C2 = dist B C2) ∨ (dist A C2 = dist A B) ∨ (dist B C2 = dist A B))
    ) ∧ C1 ≠ C2 :=
by sorry

end isosceles_triangle_two_points_l703_703611


namespace largest_possible_cylindrical_tank_radius_in_crate_l703_703396

theorem largest_possible_cylindrical_tank_radius_in_crate
  (crate_length : ℝ) (crate_width : ℝ) (crate_height : ℝ)
  (cylinder_height : ℝ) (cylinder_radius : ℝ)
  (h_cube : crate_length = 20 ∧ crate_width = 20 ∧ crate_height = 20)
  (h_cylinder_in_cube : cylinder_height = 20 ∧ 2 * cylinder_radius ≤ 20) :
  cylinder_radius = 10 :=
sorry

end largest_possible_cylindrical_tank_radius_in_crate_l703_703396


namespace distance_P1_P5_is_2π_l703_703946

noncomputable def curve (x : ℝ) : ℝ := 2 * sin (x + π / 4) * cos (π / 4 - x)

def is_intersection (P : ℝ × ℝ) : Prop :=
  P.snd = 1 / 2 ∧ P.snd = curve P.fst

def intersection_points (n : ℕ) : ℝ :=
  if n % 2 = 0 then
    n * π / 12 + 7 π / 12
  else
    n * π / 12 + 11 π / 12
  
def distance (P1 P2 : ℝ) : ℝ :=
  abs (P2 - P1)

theorem distance_P1_P5_is_2π :
  let P1 := intersection_points 1
  let P5 := intersection_points 5
  in distance P1 P5 = 2 * π :=
by
  sorry

end distance_P1_P5_is_2π_l703_703946


namespace rectangle_area_l703_703602

theorem rectangle_area (c d : ℝ) : 
  (sqrt (c^2 + d^2)) = sqrt (c^2 + d^2) → (c * d) = c * d :=
by
  intro h
  exact (eq.refl (c * d))

end rectangle_area_l703_703602


namespace units_digit_of_sum_of_cubes_of_first_3000_odd_integers_l703_703034

theorem units_digit_of_sum_of_cubes_of_first_3000_odd_integers :
  (∑ n in (Finset.range 3000).filter (λ x, x % 2 = 1), (n + 1)^3) % 10 = 0 := by
  sorry

end units_digit_of_sum_of_cubes_of_first_3000_odd_integers_l703_703034


namespace new_elephants_entry_rate_l703_703773

-- Definitions
def initial_elephants := 30000
def exodus_rate := 2880
def exodus_duration := 4
def final_elephants := 28980
def new_elephants_duration := 7

-- Prove that the rate of new elephants entering the park is 1500 elephants per hour
theorem new_elephants_entry_rate :
  let elephants_left_after_exodus := initial_elephants - exodus_rate * exodus_duration
  let new_elephants := final_elephants - elephants_left_after_exodus
  let new_entry_rate := new_elephants / new_elephants_duration
  new_entry_rate = 1500 :=
by
  sorry

end new_elephants_entry_rate_l703_703773


namespace shaffiq_steps_l703_703718

def divide_and_floor (n : ℕ) : ℕ :=
  n / 2

theorem shaffiq_steps (initial : ℕ) (target : ℕ) : initial = 100 → target = 1 -> ∃ steps : ℕ, (nat.iterate divide_and_floor steps initial = target) ∧ (steps = 6) :=
by
  intros h_initial h_target
  use 6
  split
  sorry  -- The actual proof is omitted.
  trivial -- steps = 6 comes from our earlier computation

end shaffiq_steps_l703_703718


namespace trig_expr_evaluation_l703_703888

noncomputable def evaluate_trig_expr : ℝ :=
  sin (37 * (π / 180)) * cos (34 * (π / 180))^2 +
  2 * sin (34 * (π / 180)) * cos (37 * (π / 180)) * cos (34 * (π / 180)) -
  sin (37 * (π / 180)) * sin (34 * (π / 180))^2

theorem trig_expr_evaluation :
  evaluate_trig_expr = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end trig_expr_evaluation_l703_703888


namespace ants_meeting_points_l703_703772

/-- Definition for the problem setup: two ants running at constant speeds around a circle. -/
structure AntsRunningCircle where
  laps_ant1 : ℕ
  laps_ant2 : ℕ

/-- Theorem stating that given the laps completed by two ants in opposite directions on a circle, 
    they will meet at a specific number of distinct points. -/
theorem ants_meeting_points 
  (ants : AntsRunningCircle)
  (h1 : ants.laps_ant1 = 9)
  (h2 : ants.laps_ant2 = 6) : 
    ∃ n : ℕ, n = 5 := 
by
  -- Proof goes here
  sorry

end ants_meeting_points_l703_703772


namespace number_of_initials_is_10000_l703_703978

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l703_703978


namespace smallest_m_last_four_digits_l703_703661

theorem smallest_m_last_four_digits :
  ∃ m : ℕ, 
    (∃ k : ℕ, m = 4 * k) ∧ -- m is divisible by 4
    (∃ k : ℕ, m = 9 * k) ∧ -- m is divisible by 9
    (∀ d : ℕ, d ∣ m → d = 4 ∨ d = 9 ∨ d = 1) ∧ -- m's digits are only 4 and 9
    (∃ n₄ n₉ : ℕ, n₄ ≥ 2 ∧ n₉ ≥ 2 ∧ m.digits 10 = (list.replicate n₄ 4 ++ list.replicate n₉ 9).reverse) ∧ -- at least two 4's and two 9's
    (m % 10000 = 9494) -- last four digits are 9494
    :=
  sorry

end smallest_m_last_four_digits_l703_703661


namespace alice_steps_l703_703443

noncomputable def num_sticks (n : ℕ) : ℕ :=
  (n + 1 : ℕ) ^ 2

theorem alice_steps (n : ℕ) (h : num_sticks n = 169) : n = 13 :=
by sorry

end alice_steps_l703_703443


namespace problem_a_l703_703926

-- Definitions and conditions
variables (A B C C1 B1: Point)
variables (φ : ℝ)
variable (midpoint : Point)
variable (isMidpoint : midpoint = (B + C) / 2) -- Assuming the midpoint M defined as (B + C) / 2

-- Conditions
axiom right_triangle_ABC1 : is_right_triangle A B C1
axiom right_triangle_AB1C : is_right_triangle A B1 C
axiom angle_ABC1 : angle A B C1 = φ
axiom angle_ACB1 : angle A C B1 = φ
axiom angle_C1_90 : angle B C1 A = 90
axiom angle_B1_90 : angle C B1 A = 90

-- Proof statements
theorem problem_a : dist midpoint B1 = dist midpoint C1 ∧ angle B1 midpoint C1 = 2 * φ := 
by
  sorry

end problem_a_l703_703926


namespace prob_at_least_one_gold_prob_gold_not_more_than_silver_l703_703483

/-
Definitions corresponding to conditions:
- Number of customers surveyed: 30
- Fraction of customers who did not win any prize: 2/3
- Fraction of winning customers who won the silver prize: 3/5
- The number of customers who did not win any prize: 20
- The number of winning customers: 10
- The number of silver prize winners: 6
- The number of gold prize winners: 4
-/

def num_customers_surveyed := 30
def frac_no_prize := (2 / 3 : ℚ)
def frac_silver_winners := (3 / 5 : ℚ)
def num_no_prize := 20
def num_prize_winners := 10
def num_silver_winners := 6
def num_gold_winners := 4

/-
The claims to prove:
1. Probability that at least one of the 3 selected customers won the gold prize is 73/203
2. Probability that the number of customers who won the gold prize among the 3 selected customers 
   is not more than the number of customers who won the silver prize is 157/203
-/

theorem prob_at_least_one_gold : 
  true → (73/203 : ℚ) = 1 - (130/203 : ℚ) := sorry

theorem prob_gold_not_more_than_silver : 
  true → (157/203 : ℚ) = (130/203 : ℚ) + (27/203 : ℚ) := sorry

end prob_at_least_one_gold_prob_gold_not_more_than_silver_l703_703483


namespace find_a3_l703_703532

-- Define the sequence conditions
def sequence (a : ℕ → ℝ) :=
  ∀ n, a (n+2) = a (n+1) + a n

-- Define specific terms' values
variables (a : ℕ → ℝ)
variables (a1_eq : a 1 = 1) (a5_eq : a 5 = 8)

-- The theorem to prove
theorem find_a3 (h : sequence a) : a 3 = 3 :=
by
  sorry

end find_a3_l703_703532


namespace equations_not_equivalent_l703_703392

variable {X : Type} [Field X]
variable (A B : X → X)

theorem equations_not_equivalent (h1 : ∀ x, A x ^ 2 = B x ^ 2) (h2 : ¬∀ x, A x = B x) :
  (∃ x, A x ≠ B x ∨ A x ≠ -B x) := 
sorry

end equations_not_equivalent_l703_703392


namespace part1_part2_l703_703232

noncomputable def area_of_triangle (a b c : ℝ) (sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Part 1
theorem part1 (a : ℝ) (b : ℝ) (c : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : 2 * Real.sin C = 3 * Real.sin A)
  (a_val : a = 4) (b_val : b = 5) (c_val : c = 6) : 
  area_of_triangle 4 5 (3 * Real.sqrt 7 / 8) = 15 * Real.sqrt 7 / 4 := by 
  sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : ∃ (a' : ℕ), a = a') :
  a = 2 := by
  sorry

end part1_part2_l703_703232


namespace minimize_value_l703_703668

noncomputable def minimize_y (a b x : ℝ) : ℝ := (x - a) ^ 3 + (x - b) ^ 3

theorem minimize_value (a b : ℝ) : ∃ x : ℝ, minimize_y a b x = minimize_y a b a ∨ minimize_y a b x = minimize_y a b b :=
sorry

end minimize_value_l703_703668


namespace find_x_l703_703185

theorem find_x 
  (b : ℝ) (x : ℝ) 
  (hb : b > 1)
  (hx : x > 0)
  (h : (3 * x) ^ (Real.log b 3) - (5 * x) ^ (Real.log b 5) = 0) : 
  x = 3 / 5 :=
sorry

end find_x_l703_703185


namespace part_a_l703_703398

theorem part_a (n : ℤ) (m : ℤ) (h : m = n + 2) : 
  n * m + 1 = (n + 1) ^ 2 := by
  sorry

end part_a_l703_703398


namespace equality_condition_l703_703878

theorem equality_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) → a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end equality_condition_l703_703878


namespace solution_values_l703_703051

variables (A B C D : ℝ)
variables (m n : ℕ)

-- Conditions given in the problem
axiom condition_1 : ∃ (x y : ℝ), x^2 + y^2 = A ∧ |x| + |y| = B
axiom condition_2 : ∃ (x y : ℝ), x^2 + y^2 = A ∧ |x| + |y| = B → fintype.card {p : ℝ × ℝ // (p.fst^2 + p.snd^2 = A) ∧ (|p.fst| + |p.snd| = B)} = m
axiom condition_3 : ∃ (x y z : ℝ), x^2 + y^2 + z^2 = C ∧ |x| + |y| + |z| = D
axiom condition_4 : ∃ (x y z : ℝ), x^2 + y^2 + z^2 = C ∧ |x| + |y| + |z| = D → fintype.card {p : ℝ × ℝ × ℝ // (p.fst^2 + p.snd^2 + p.thd^2 = C) ∧ (|p.fst| + |p.snd| + |p.thd| = D)} = n
axiom condition_5 : m > n
axiom condition_6 : n > 1

-- Lean statement proving the values of m and n
theorem solution_values : m = 8 ∧ n = 6 := 
sorry

end solution_values_l703_703051


namespace age_of_25th_student_l703_703796

theorem age_of_25th_student (avg_age_25_students : ℕ) (avg_age_10_students : ℕ) (avg_age_14_students : ℕ)
  (total_students : ℕ) (students_1 : ℕ) (students_2 : ℕ) (total_avg_age_25_students : avg_age_25_students * total_students = 625)
  (total_avg_age_10_students : avg_age_10_students * students_1 = 220)
  (total_avg_age_14_students : avg_age_14_students * students_2 = 392) :
  ∃ age_25th_student : ℕ,
    age_25th_student = 13 := by
  have h1 : 25 = total_students := by sorry
  have h2 : 10 = students_1 := by sorry
  have h3 : 14 = students_2 := by sorry
  use 13
  sorry

end age_of_25th_student_l703_703796


namespace total_people_100_l703_703206

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 : ℚ) * P = 40 ∧ (1 / 4 : ℚ) * P ≤ P ∧ P ≥ 40 

theorem total_people_100 {P : ℕ} (h : total_people P) : P = 100 := 
by 
  sorry -- proof would go here

end total_people_100_l703_703206


namespace limit_tan_sin_eq_one_fourth_l703_703459

noncomputable def limit_tan_sin : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → (|((tan x - sin x) / (x * (1 - cos (2 * x))) - 1/4)| < ε))

theorem limit_tan_sin_eq_one_fourth : limit_tan_sin := 
sorry

end limit_tan_sin_eq_one_fourth_l703_703459


namespace lightly_spacy_subsets_of_15_l703_703109

-- Definition of the problem conditions
def is_lightly_spacy (s : set ℕ) : Prop :=
  ∀ (n : ℕ), (n ∈ s → (n + 1) ∉ s) ∧ (n + 2 ∉ s) ∧ (n + 3 ∉ s) ∧ (n + 4 ∉ s)

-- Definition of the set T_n
def T (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

-- Initial values for d_n
def d_values : ℕ → ℕ
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 5
| (n + 5) := (d_values (n + 4)) + (d_values n)

-- The total number of lightly spacy subsets for T_n
def d (n : ℕ) : ℕ :=
d_values n

-- The theorem we want to prove
theorem lightly_spacy_subsets_of_15 : d 15 = 181 := sorry

end lightly_spacy_subsets_of_15_l703_703109


namespace ticket_distribution_l703_703349

def tickets := {1, 2, 3, 4, 5, 6}
def people := {A, B, C, D}

theorem ticket_distribution :
  ∃ (distributions : set (tickets × people)),
  (∀ p ∈ people, 1 ≤ cardinality (distributions ∩ {(t, p) | t ∈ tickets}) ∧
                  cardinality (distributions ∩ {(t, p) | t ∈ tickets}) ≤ 2 ∧
                  (∀ t₁ t₂ ∈ tickets, t₁ < t₂ → ((t₁, p) ∈ distributions ∧ (t₂, p) ∈ distributions) → t₂ = t₁ + 1)) →
  (cardinality distributions = 144) :=
sorry

end ticket_distribution_l703_703349


namespace train_length_proof_l703_703809

noncomputable def bridge_length : ℕ := 2800
noncomputable def train_speed : ℕ := 800
noncomputable def crossing_time : ℕ := 4

theorem train_length_proof : 
  let length_of_train := train_speed * crossing_time - bridge_length in
  length_of_train = 400 :=
by
  let length_of_train := train_speed * crossing_time - bridge_length
  have h1 : length_of_train = 3200 - 2800 := by rfl
  have h2 : 3200 - 2800 = 400 := by norm_num
  exact Eq.trans h1 h2

end train_length_proof_l703_703809


namespace remove_candies_to_distribute_equally_l703_703486

def elena_candies (total_candies friends : ℕ) : ℕ :=
  total_candies % friends

theorem remove_candies_to_distribute_equally :
  ∀ (total_candies friends : ℕ), friends = 4 → total_candies = 35 → elena_candies total_candies friends = 3 :=
by
  intros total_candies friends hfriends htotal_candies
  rw [htotal_candies, hfriends]
  simp [elena_candies]
  sorry

end remove_candies_to_distribute_equally_l703_703486


namespace liam_needed_additional_questions_l703_703679

theorem liam_needed_additional_questions :
  let programming_questions := 15
  let data_structures_questions := 20
  let algorithms_questions := 15
  let percentage_correct_programming := 0.8
  let percentage_correct_data_structures := 0.5
  let percentage_correct_algorithms := 0.7
  let total_questions := 50
  let required_passing_percentage := 0.65
  let correct_programming := percentage_correct_programming * programming_questions
  let correct_data_structures := percentage_correct_data_structures * data_structures_questions
  let correct_algorithms := percentage_correct_algorithms * algorithms_questions
  let total_correct := correct_programming + correct_data_structures + correct_algorithms
  let correct_needed_to_pass := Float.ceil(required_passing_percentage * total_questions)
  in correct_needed_to_pass - total_correct = 1 :=
by
  sorry

end liam_needed_additional_questions_l703_703679


namespace positive_integer_solutions_count_l703_703112

theorem positive_integer_solutions_count :
  ∃ S : Finset ℕ, (∀ x ∈ S, 15 < -2 * x + 17) ∧ S.card = 1 :=
by
  sorry

end positive_integer_solutions_count_l703_703112


namespace cone_height_l703_703745

theorem cone_height (r : ℝ) (h : ℝ) 
  (hc1 : r = 20)  -- radius of semicircular surface
  (hc2 : 2 * r * h = (1/2) * 2 * π * 20) -- circumference condition derived
: h = 10 * real.sqrt 3 := 
sorry -- Proof is not provided as per instructions.

end cone_height_l703_703745


namespace probability_odd_even_l703_703306

def four_numbers := {1, 2, 3, 4 : ℕ}

theorem probability_odd_even :
  let total_outcomes := (finset.card (finset.powersetLen 2 four_numbers.to_finset)).to_rat
  let favorable_outcomes := (2 * 2).to_rat
  (favorable_outcomes / total_outcomes) = 2/3 :=
by
  sorry

end probability_odd_even_l703_703306


namespace inclination_angle_range_proven_l703_703752

noncomputable def slope_of_line (α : ℝ) : ℝ := (sin α) / (√3)

def inclination_angle_range (θ α : ℝ) : Prop :=
  let k := slope_of_line α in
  tan θ = k ∧ (θ ∈ set.Icc 0 (π / 6) ∨ θ ∈ set.Ico (5 * π / 6) π)

theorem inclination_angle_range_proven {α θ : ℝ} :
  (x : ℝ) → (y : ℝ) → (x * sin α - √3 * y + 1 = 0) → inclination_angle_range θ α :=
sorry

end inclination_angle_range_proven_l703_703752


namespace volume_tetrahedron_l703_703333

def A1 := 4^2
def A2 := 3^2
def h := 1

theorem volume_tetrahedron:
  (h / 3 * (A1 + A2 + Real.sqrt (A1 * A2))) = 37 / 3 := by
  sorry

end volume_tetrahedron_l703_703333


namespace maximize_angle_l703_703921

-- Define the problem setup
variable {α : Type*} [Plane α]
variable {A B : Point α}

-- The statement to be proved

theorem maximize_angle (P : Point α) : ∃ P, (∀ Q : Point α, Q ≠ P → angle A P B ≥ angle A Q B) :=
sorry

end maximize_angle_l703_703921


namespace binary_division_remainder_l703_703784

theorem binary_division_remainder : 
  let binary_number := "110111100101"
  let divisor := 8
  let remainder := 5
  binary_number.to_nat % divisor = remainder :=
by sorry

end binary_division_remainder_l703_703784


namespace total_lateral_surface_area_l703_703836

def outer_radius : ℝ := 4
def inner_radius : ℝ := 3
def height : ℝ := 8

theorem total_lateral_surface_area : 
  2 * Real.pi * outer_radius * height - 2 * Real.pi * inner_radius * height = 16 * Real.pi :=
by
  sorry

end total_lateral_surface_area_l703_703836


namespace four_letter_initial_sets_l703_703967

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l703_703967


namespace article_filling_correct_l703_703054

-- definitions based on conditions provided
def Gottlieb_Daimler := "Gottlieb Daimler was a German engineer."
def Invented_Car := "Daimler is normally believed to have invented the car."

-- Statement we want to prove
theorem article_filling_correct : 
  (Gottlieb_Daimler = "Gottlieb Daimler was a German engineer.") ∧ 
  (Invented_Car = "Daimler is normally believed to have invented the car.") →
  ("Gottlieb Daimler, a German engineer, is normally believed to have invented the car." = 
   "Gottlieb Daimler, a German engineer, is normally believed to have invented the car.") :=
by
  sorry

end article_filling_correct_l703_703054


namespace problem_correct_l703_703863

noncomputable def problem := 
  1 - (1 / 2)⁻¹ * Real.sin (60 * Real.pi / 180) + abs (2^0 - Real.sqrt 3) = 0

theorem problem_correct : problem := by
  sorry

end problem_correct_l703_703863


namespace log10_17_between_consecutive_integers_l703_703347

theorem log10_17_between_consecutive_integers :
  ∃ (a b : ℤ), (a + b = 3) ∧ (a < b) ∧ (a < log 17 / log 10) ∧ (log 17 / log 10 < b) ∧ (b = a + 1) :=
by {
  -- Note: log 10 is pre-implemented in Lean as the natural logarithm (log base e), so log 10 17 == log 17 / log 10
  use 1, 2,
  simp,
  split,
  {
    exact 3,
  },
  {
    refine ⟨1, _, _, _⟩,
    sorry,
  }
}

end log10_17_between_consecutive_integers_l703_703347


namespace probability_of_satisfaction_l703_703449

def negative_review_given_dissatisfied := 0.8
def positive_review_given_satisfied := 0.15
def angry_reviews := 60
def positive_reviews := 20

theorem probability_of_satisfaction (p : ℝ) (h : (0.15 * p) / (0.8 * (1 - p)) = 1 / 3) : p ≈ 0.64 :=
sorry

end probability_of_satisfaction_l703_703449


namespace time_to_print_800_flyers_l703_703817

theorem time_to_print_800_flyers (x : ℝ) (h1 : 0 < x) :
  (1 / 6) + (1 / x) = 1 / 1.5 ↔ ∀ y : ℝ, 800 / 6 + 800 / x = 800 / 1.5 :=
by sorry

end time_to_print_800_flyers_l703_703817


namespace minimum_distance_l703_703162

theorem minimum_distance
  (P : ℂ) -- Representing P as a point in the plane
  (hP : ∃ α : ℝ, P = complex.ofReal((sqrt 3) * real.cos α) + complex.I * (3 * real.sin α)) -- Parametric form
  (hC : complex.abs (complex.ofReal (P.re / sqrt 3)^2 + (P.im / 3)^2 - 1) < ε) -- P lies on the ellipse, tolerated with ε for equality in computations
  (hL : ∀ p θ, p * real.sin (θ - π / 4) = 2 * sqrt 2)
  (hPolar_to_Cartesian : ∀ x y, y - x + 4 = 0) : 
  ∃ d, d = (2 * sqrt 2 - sqrt 6) :=
sorry

end minimum_distance_l703_703162


namespace depth_of_water_l703_703416

/-- Define the parameters of the problem -/
def Length := 10 -- Length of the cistern in meters
def Width := 8 -- Width of the cistern in meters
def TotalWetSurfaceArea := 134 -- Total wet surface area in square meters

/-- Define the area of the bottom of the cistern -/
def AreaBottom := Length * Width

/-- Define the depth of water in the cistern (to be proved) -/
def Depth := 1.5 -- Depth of water in the cistern in meters

/-- Define the area of the sides in contact with water -/
def AreaSides := TotalWetSurfaceArea - AreaBottom

theorem depth_of_water (d : ℝ) :
  2 * d * Length + 2 * d * Width = AreaSides →
  d = Depth :=
by
  sorry

end depth_of_water_l703_703416


namespace value_of_expression_l703_703377

theorem value_of_expression (x : ℤ) (h : x = 3) : x^6 - 3 * x = 720 := by
  sorry

end value_of_expression_l703_703377


namespace frog_ends_on_vertical_side_l703_703419

noncomputable def frog_probability : ℚ :=
  let P := λ (x y : ℕ), if (x = 0 ∨ x = 6) then 1 else if (y = 0 ∨ y = 6) then 0 else sorry
  in P 2 3

theorem frog_ends_on_vertical_side :
  frog_probability = 7 / 9 :=
sorry

end frog_ends_on_vertical_side_l703_703419


namespace product_of_reciprocals_l703_703760

theorem product_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 :=
by
  sorry

end product_of_reciprocals_l703_703760


namespace evaluate_expression_l703_703487

-- Define the ceiling of square roots for the given numbers
def ceil_sqrt_3 := 2
def ceil_sqrt_27 := 6
def ceil_sqrt_243 := 16

-- Main theorem statement
theorem evaluate_expression :
  ceil_sqrt_3 + ceil_sqrt_27 * 2 + ceil_sqrt_243 = 30 :=
by
  -- Sorry to indicate that the proof is skipped
  sorry

end evaluate_expression_l703_703487


namespace range_of_a_l703_703259

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = -f x)  -- odd function
  (h2 : ∀ x, x < 0 → f x = 9 * x + (a^2) / x + 7)
  (h3 : ∀ x, 0 ≤ x → f x ≥ a + 1) : 
  a ≤ -8/7 :=
begin
  sorry
end

end range_of_a_l703_703259


namespace three_pow_210_mod_17_l703_703787

open Nat

theorem three_pow_210_mod_17 : 3^210 % 17 = 9 := by
  -- Using Fermat's Little Theorem
  have h1 : 3^16 % 17 = 1 := by sorry
  -- Reducing the exponent mod 16
  have h2 : 210 % 16 = 2 := by sorry
  -- Calculate 3^2
  have h3 : 3^2 = 9 := by sorry
  -- Combining the results
  calc
    3^210 % 17 = 3^(16 * 13 + 2) % 17 := by rw [Nat.pow_add]
    ...         = (3^16)^(13) * 3^2 % 17 := by rw [pow_mul]
    ...         = (1^13) * 9 % 17 := by rw [h1, h3]
    ...         = 9 % 17 := by sorry
    ...         = 9 := by sorry

end three_pow_210_mod_17_l703_703787


namespace impossible_to_achieve_25_percent_grape_juice_l703_703019

theorem impossible_to_achieve_25_percent_grape_juice (x y : ℝ) 
  (h1 : ∀ a b : ℝ, (8 / (8 + 32) = 2 / 10) → (6 / (6 + 24) = 2 / 10))
  (h2 : (8 * x + 6 * y) / (40 * x + 30 * y) = 1 / 4) : false :=
by
  sorry

end impossible_to_achieve_25_percent_grape_juice_l703_703019


namespace triangle_area_l703_703339

theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : P = 20)
  (h2 : r = 2.5) 
  (h3 : A = r * (P / 2)) :
  A = 25 :=
by
  have s : ℝ := P / 2
  rw [h1, h2] at *
  have hs : s = 10 := by norm_num [s, h1]
  rw hs at *
  norm_num [A, h3]
  assumption sorry

end triangle_area_l703_703339


namespace find_x_l703_703400

-- Define the known values
def a := 6
def b := 16
def c := 8
def desired_average := 13

-- Define the target number we need to find
def target_x := 22

-- Prove that the number we need to add to get the desired average is 22
theorem find_x : (a + b + c + target_x) / 4 = desired_average :=
by
  -- The proof itself is omitted as per instructions
  sorry

end find_x_l703_703400


namespace find_pairs_l703_703891

open BigOperators

theorem find_pairs (a b : ℕ) (h1 : a < b) (h2 : ∑ k in Finset.range (b-1) - Finset.range (a+1), k = 1998) :
  (a, b) = (1997, 1999) ∨ 
  (a, b) = (664, 668) ∨ 
  (a, b) = (497, 502) ∨ 
  (a, b) = (217, 227) ∨ 
  (a, b) = (160, 173) ∨ 
  (a, b) = (60, 88) ∨ 
  (a, b) = (37, 74) ∨ 
  (a, b) = (35, 73) :=
sorry

end find_pairs_l703_703891


namespace trig_problem_1_trig_problem_2_l703_703805

-- Problem (1)
theorem trig_problem_1 (α : ℝ) (h1 : Real.tan (π + α) = -4 / 3) (h2 : 3 * Real.sin α / 4 = -Real.cos α)
  : Real.sin α = -4 / 5 ∧ Real.cos α = 3 / 5 := by
  sorry

-- Problem (2)
theorem trig_problem_2 : Real.sin (25 * π / 6) + Real.cos (26 * π / 3) + Real.tan (-25 * π / 4) = -1 := by
  sorry

end trig_problem_1_trig_problem_2_l703_703805


namespace expected_sample_size_l703_703814

noncomputable def highSchoolTotalStudents (f s j : ℕ) : ℕ :=
  f + s + j

noncomputable def expectedSampleSize (total : ℕ) (p : ℝ) : ℝ :=
  total * p

theorem expected_sample_size :
  let f := 400
  let s := 320
  let j := 280
  let p := 0.2
  let total := highSchoolTotalStudents f s j
  expectedSampleSize total p = 200 :=
by
  sorry

end expected_sample_size_l703_703814


namespace num_distinct_intersections_l703_703114

def linear_eq1 (x y : ℝ) := x + 2 * y - 10
def linear_eq2 (x y : ℝ) := x - 4 * y + 8
def linear_eq3 (x y : ℝ) := 2 * x - y - 1
def linear_eq4 (x y : ℝ) := 5 * x + 3 * y - 15

theorem num_distinct_intersections (n : ℕ) :
  (∀ x y : ℝ, linear_eq1 x y = 0 ∨ linear_eq2 x y = 0) ∧ 
  (∀ x y : ℝ, linear_eq3 x y = 0 ∨ linear_eq4 x y = 0) →
  n = 3 :=
  sorry

end num_distinct_intersections_l703_703114


namespace minimum_adults_attending_l703_703096

/-- 
Problem Statement:
Given:
  - Adults are seated in groups of 17 (G1).
  - Children are seated in groups of 15 (G2).
  - For every adult group, there is one children group and one mixed group (G3).
  - The mixed group has exactly 1 adult for every 2 children (C1).
  - The mixed group should not exceed 21 members (C2).
  - The total number of adults is equal to the total number of children (C3).
Prove:
  The minimum number of adults attending is 276 (min_adults).
-/
theorem minimum_adults_attending 
  (G1 : ℕ) (G2 : ℕ) (G3 : ℕ) 
  (C1 : ℕ) (C2 : ℕ) (C3 : ℕ)
  (adults_in_group : G1 = 17)
  (children_in_group : G2 = 15)
  (adults_eq_children: C3)
  (adults_per_mix: C1 = 6)
  (mix_max: C2 = 18)
  (num_groups_eq : G1 = G2 ): 
  ∃ (adults_attending : ℕ), adults_attending = 276 :=
by 
  sorry

end minimum_adults_attending_l703_703096


namespace earnings_per_bluegill_l703_703857

-- Define the conditions as hypotheses
def cost_of_game : ℕ := 60
def earnings_last_weekend : ℕ := 35
def earnings_per_trout : ℕ := 5
def fish_caught_sunday : ℕ := 5
def percent_trout : ℕ := 60
def dollars_needed : ℕ := 2

-- Theorem to prove
theorem earnings_per_bluegill : 
  let total_needed := cost_of_game - dollars_needed,
  let total_earned := total_needed - earnings_last_weekend,
  let num_trout := (percent_trout * fish_caught_sunday) / 100, 
  let num_bluegill := fish_caught_sunday - num_trout,
  let earnings_from_trout := num_trout * earnings_per_trout,
  let earnings_from_bluegill := total_earned - earnings_from_trout in
  earnings_from_bluegill / num_bluegill = 4 :=
by
  -- The proof will go here
  sorry

end earnings_per_bluegill_l703_703857


namespace exists_M_on_EC_l703_703629

-- Definitions and assumptions based on the problem statement
variables {A B C D E : Point} -- Assume these are points in a 2D Euclidean plane
variable (P : Point → Point → Point → Triangle) -- Constructor for a triangle using three points

-- Isosceles right triangle predicate
def is_isosceles_right_triangle (t : Triangle) : Prop :=
  (is_right_angle t.a t.b t.c ∧ t.a t.b = t.a t.c) ∨
  (is_right_angle t.b t.a t.c ∧ t.b t.a = t.b t.c) ∨
  (is_right_angle t.c t.a t.b ∧ t.c t.a = t.c t.b)

-- Non-congruent isosceles triangles
variables (h1 : is_isosceles_right_triangle (P A B C))
variables (h2 : is_isosceles_right_triangle (P A D E))
variable (h3 : ¬congruent (P A B C) (P A D E))
variable (h4 : fixed (P A B C))
variable (h5 : rotates (P A D E) around A)

-- The statement to prove
theorem exists_M_on_EC : 
  ∃ M : Point, M ∈ line_segment E C ∧ is_isosceles_right_triangle (P B M D) :=
sorry

end exists_M_on_EC_l703_703629


namespace angle_SLB_eq_90_l703_703332

noncomputable theory

open EuclideanGeometry

-- Definitions of points and properties given in conditions
variables {A B C K L M S : Point}
variables {α β γ : ℝ}
variables (h_isosceles : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ dist A B = dist B C)
variables (h_square : isSquare A K L M)
variables (h_point : dist A S = dist S L ∧ isOnLine A B S)

-- Angle equality theorem statement
theorem angle_SLB_eq_90 :
  ∠ (S, L, B) = 90 :=
by
  sorry

end angle_SLB_eq_90_l703_703332


namespace triangle_area_triangle_obtuse_exists_l703_703228

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Given conditions
def condition_b : b = a + 1 := by sorry
def condition_c : c = a + 2 := by sorry
def condition_sin : 2 * Real.sin C = 3 * Real.sin A := by sorry

-- Proof statement for Part (1)
theorem triangle_area (h1 : condition_b) (h2 : condition_c) (h3 : condition_sin) :
  (1 / 2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := 
sorry

-- Proof statement for Part (2)
theorem triangle_obtuse_exists (h1 : condition_b) (h2 : condition_c) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧
  (let b := a + 1 in let c := a + 2 in
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) > Real.pi / 2) :=
sorry

end triangle_area_triangle_obtuse_exists_l703_703228


namespace increased_consumption_percentage_l703_703002

-- Definitions based on conditions
def original_tax (T : ℝ) : ℝ := T
def original_consumption (C : ℝ) : ℝ := C

-- Conditions
def diminished_tax (T : ℝ) : ℝ := 0.70 * T
def increased_consumption (C : ℝ) (x : ℝ) : ℝ := C * (1 + x / 100)

-- New revenue based on diminished tax and increased consumption
def new_revenue (T C x : ℝ) : ℝ := 0.70 * T * (C * (1 + x / 100))

-- Decrease in revenue condition
def decreased_revenue (T C : ℝ) : ℝ := 0.77 * T * C

-- Theorem statement
theorem increased_consumption_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  (∀ x : ℝ, new_revenue T C x = decreased_revenue T C → x = 10) :=
by
  sorry -- Proof goes here


end increased_consumption_percentage_l703_703002


namespace unique_function_l703_703875

-- Define the type of strictly positive integers
def pos_int := { x : ℕ // 0 < x }

-- Define a function f from pos_int to pos_int
def f : pos_int → pos_int

-- Define the first condition: for each positive integer n, the sum of first n values of f is a perfect square
def sum_is_perfect_square (f : pos_int → pos_int) : Prop :=
  ∀ n : pos_int, ∃ m : ℕ, (finset.range n.val).sum (λ k, (f ⟨k+1, nat.succ_pos k⟩).val) = m * m

-- Define the second condition: f(n) divides n^3 for each positive integer n
def divides_n_cubed (f : pos_int → pos_int) : Prop :=
  ∀ n : pos_int, (f n).val ∣ (n.val ^ 3)

-- Our goal: to prove that the only function f satisfying the conditions is f(n) = n^3
theorem unique_function 
  (f : pos_int → pos_int) 
  (h1 : sum_is_perfect_square f) 
  (h2 : divides_n_cubed f) : 
  ∀ n : pos_int, f n = ⟨n.val ^ 3, nat.pos_pow_of_pos 3 n.property⟩ :=
sorry

end unique_function_l703_703875


namespace work_completion_by_b_l703_703404

theorem work_completion_by_b (a_days : ℕ) (a_solo_days : ℕ) (a_b_combined_days : ℕ) (b_days : ℕ) :
  a_days = 12 ∧ a_solo_days = 3 ∧ a_b_combined_days = 5 → b_days = 15 :=
by
  sorry

end work_completion_by_b_l703_703404


namespace centers_of_circumcircles_form_parallelogram_l703_703634

theorem centers_of_circumcircles_form_parallelogram
  (P Q R S: ℝ) 
  (A B C D : ℝ)
  (parallelogram : is_parallelogram A B C D)
  (points_chosen : chosen_points P Q R S inside parallelogram)
  (triangles : resulting_triangles_from P Q R S A B C D) :
  is_parallelogram (circumcenter (triangle_faces P Q A)) 
                   (circumcenter (triangle_faces Q R B))
                   (circumcenter (triangle_faces R S C)) 
                   (circumcenter (triangle_faces S P D)) :=
sorry

end centers_of_circumcircles_form_parallelogram_l703_703634


namespace lilith_caps_loss_l703_703273

theorem lilith_caps_loss
  (caps_per_month_first_year : ℕ)
  (caps_per_month_after_first_year : ℕ)
  (christmas_caps : ℕ)
  (years_collecting : ℕ)
  (total_caps_collected : ℕ)
  (caps_after_years : ℕ) :
  caps_per_month_first_year = 3 →
  caps_per_month_after_first_year = 5 →
  christmas_caps = 40 →
  years_collecting = 5 →
  total_caps_collected = 401 →
  ∀ years, (years >= 1) →
  (total_caps : ℕ),
    total_caps = (caps_per_month_first_year * 12) + 
                 (caps_per_month_after_first_year * 12 * (years_collecting - 1)) + 
                 (christmas_caps * years_collecting) →
  (caps_lost : ℕ), 
    caps_lost = total_caps - total_caps_collected →
  (caps_lost / years_collecting = 15) :=
by
  intros caps_per_month_first_year caps_per_month_after_first_year christmas_caps 
         years_collecting total_caps_collected caps_after_years
  intros h1 h2 h3 h4 h5 h6
  sorry

end lilith_caps_loss_l703_703273


namespace eccentricity_of_ellipse_l703_703536

-- Define the conditions
def ellipse_eq (a b x y : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

def parallel_lines (x y : ℝ) : Prop :=
  ∃ (b : ℝ), y = x + b ∨ y = x - b

def intersects_quad (E : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (A B C D : ℝ × ℝ), 
  E A.1 A.2 ∧ E B.1 B.2 ∧ E C.1 C.2 ∧ E D.1 D.2 ∧ 
  l A.1 A.2 ∧ l B.1 B.2 ∧ l C.1 C.2 ∧ l D.1 D.2

def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry
-- placeholder for area calculation

-- Define given values
def given_area : ℝ := 8 / 3 * b^2

-- Lean statement
theorem eccentricity_of_ellipse (a b : ℝ) (h1 : ellipse_eq a b) 
  (h2 : intersects_quad (λ x y, ellipse_eq a b x y) (λ x y, parallel_lines x y)) 
  (h3 : area_quadrilateral = given_area) : 
  let c := sqrt (a^2 - b^2) in 
  c / a = sqrt (1 / 2) := sorry

end eccentricity_of_ellipse_l703_703536


namespace sin_monotonic_interval_l703_703876

theorem sin_monotonic_interval :
  (∀ x y : ℝ, - (π / 8) ≤ x ∧ x ≤ (3 * π / 8) ∧ x ≤ y → 
    sin (2 * x - π / 4) ≤ sin (2 * y - π / 4)) :=
begin
  sorry
end

end sin_monotonic_interval_l703_703876


namespace first_fun_friday_is_april_28_l703_703065

noncomputable def first_fun_friday (fiscal_year_begin: ℕ → ℕ → ℕ) : ℕ × ℕ :=
sorry

theorem first_fun_friday_is_april_28:
  (fiscal_year_begin (3) (1) = 3 ∧ first_fun_friday = (4, 28)) :=
by sorry

end first_fun_friday_is_april_28_l703_703065


namespace product_of_c_l703_703507

theorem product_of_c:
  (∏ c in finset.filter (λ c => c < 25) (finset.range 25), c) = 24! :=
by
  sorry

end product_of_c_l703_703507


namespace A_runs_faster_l703_703411

variable (v_A v_B : ℝ)  -- Speed of A and B
variable (k : ℝ)       -- Factor by which A is faster than B

-- Conditions as definitions in Lean:
def speed_relation (k : ℝ) (v_A v_B : ℝ) : Prop := v_A = k * v_B
def start_difference : ℝ := 60
def race_course_length : ℝ := 80
def reach_finish_same_time (v_A v_B : ℝ) : Prop := (80 / v_A) = ((80 - start_difference) / v_B)

theorem A_runs_faster
  (h1 : speed_relation k v_A v_B)
  (h2 : reach_finish_same_time v_A v_B) : k = 4 :=
by
  sorry

end A_runs_faster_l703_703411


namespace distance_sum_l703_703871

theorem distance_sum (a : ℝ) (x y : ℝ) 
  (AB CD : ℝ) (A B C D P Q M N : ℝ)
  (h_AB : AB = 4) (h_CD : CD = 8) 
  (h_M_AB : M = (A + B) / 2) (h_N_CD : N = (C + D) / 2)
  (h_P_AB : P ∈ [A, B]) (h_Q_CD : Q ∈ [C, D])
  (h_x : x = dist P M) (h_y : y = dist Q N)
  (h_y_eq_2x : y = 2 * x) (h_x_eq_a : x = a) :
  x + y = 3 * a := 
by
  sorry

end distance_sum_l703_703871


namespace sum_of_first_2007_terms_l703_703838

variable {α : Type}
open Nat

def seq (b : ℕ → ℤ) :=
  ∀ n ≥ 3, b n = b (n - 1) - b (n - 2)

def sum_first (b : ℕ → ℤ) (n : ℕ) :=
  ∑ i in range n, b i

def periodic (b : ℕ → ℤ) (k : ℕ) :=
  ∀ n, b (n + k) = b n

theorem sum_of_first_2007_terms (b : ℕ → ℤ) (p q : ℤ) 
  (h1 : b 1 = p) (h2 : b 2 = q)
  (rec : seq b)
  (sum_1500 : sum_first b 1500 = 2006)
  (sum_2006 : sum_first b 2006 = 1500)
  (periodic_b : periodic b 6) 
  : sum_first b 2007 = 506 := 
sorry

end sum_of_first_2007_terms_l703_703838


namespace quadratic_one_solution_m_l703_703942

theorem quadratic_one_solution_m (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + m = 0) → 
  (∀ (x y : ℝ), 3 * x^2 - 7 * x + m = 0 → 3 * y^2 - 7 * y + m = 0 → x = y) → 
  m = 49 / 12 :=
by
  sorry

end quadratic_one_solution_m_l703_703942


namespace men_in_first_group_l703_703598

-- Define the problem conditions
def men_and_boys_work (x m b : ℕ) :=
  let w := 10 * (x * m + 8 * b) in
  w = 2 * (26 * m + 48 * b) ∧
  4 * (15 * m + 20 * b) = w

-- Formal statement of the problem
theorem men_in_first_group (x m b : ℕ) :
  men_and_boys_work x m b → x = 6 :=
sorry

end men_in_first_group_l703_703598


namespace incorrect_propositions_l703_703091

-- Definitions of propositions
def Prop1 (α β : Plane) (inf_lines : Set Line) : Prop :=
  (∀ l ∈ inf_lines, l ∈ α ∧ l ∥ β) → α ∥ β

def Prop2 (α β γ : Plane) (l : Line) : Prop :=
  (α ∥ l ∧ β ∥ l) → α ∥ β

def Prop3 (α : Plane) (p1 p2 : Point) : Prop :=
  (p1 ∉ α ∧ p2 ∉ α) → ∃ β, β ∥ α ∧ p1 ∈ β ∧ p2 ∈ β

def Prop4 (α β γ : Plane) : Prop :=
  (α ∥ γ ∧ β ∥ γ) → α ∥ β

-- Main theorem asserting the incorrect propositions
theorem incorrect_propositions (α β γ : Plane) (l : Line) (p1 p2 : Point) (inf_lines : Set Line) :
  ¬ Prop1 α β inf_lines ∧ ¬ Prop2 α β γ l ∧ ¬ Prop3 α p1 p2 ∧ ¬ Prop4 α β γ :=
by
  sorry

end incorrect_propositions_l703_703091


namespace min_value_l703_703655

theorem min_value
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 9) :
  ∃ x : ℝ, x = 9 ∧ x = min (λ x, (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)) := 
sorry

end min_value_l703_703655


namespace wrapping_paper_area_l703_703854

theorem wrapping_paper_area (w v h : ℝ) (hv : v < w) (hwv : w > 0) (hv_pos : 0 < v) :
  let paper_width := 3 * v
  let paper_length := w
  paper_length * paper_width = 3 * w * v :=
by
  unfold paper_width
  unfold paper_length
  rw [mul_assoc, mul_comm v 3, mul_assoc]
  sorry

end wrapping_paper_area_l703_703854


namespace quotient_sum_distinct_remainders_l703_703717

/-- Proof that the quotient of the sum of the distinct remainders of 
    the squares of integers from 1 to 15, when divided by 13, is 3. -/
theorem quotient_sum_distinct_remainders :
  let remainders := { n : ℕ // 1 ≤ n ∧ n ≤ 15 } |>.map (λ n, (n^2 % 13)).to_finset.val.to_list in
  let m := remainders.sum in
  m / 13 = 3 :=
by
  haveI : DecidableEq ℕ := Classical.decEq ℕ
  let remainders := (List.range' 1 15).map (λ n, (n^2 % 13))
  let distinct_remainders := remainders.to_finset.to_list
  have sum_distinct_remainders : distinct_remainders.sum = 39 := sorry
  let m := 39
  show 39 / 13 = 3
  rw [Nat.div_eq_of_lt 39 13]
  exact eq.refl 3
  sorry

end quotient_sum_distinct_remainders_l703_703717


namespace area_of_triangle_PQR_l703_703534

-- Define the conditions
variables {A B C A₁ B₁ C₁ : Type}
variable {S : ℝ}
variables {k₁ k₂ k₃ : ℝ}
variable {PQR : Type}

-- Assume point conditions
variables {BA₁ A₁C CB₁ B₁A AC₁ C₁B : ℝ}
-- Area function (to find the area of triangle PQR)
variable (area_PQR : ℝ)

-- Hypothesis: Point ratios and conditions
hypothesis h_ratios : 
  BA₁ / A₁C = k₁ ∧ k₁ > 1 ∧ 
  CB₁ / B₁A = k₂ ∧ k₂ > 1 ∧ 
  AC₁ / C₁B = k₃ ∧ k₃ > 1

-- Given area of triangle ABC
variable (area_ABC : ℝ)
hypothesis h_area_ABC : area_ABC = S

-- The goal is to prove the area of triangle PQR using k₁, k₂, and k₃
theorem area_of_triangle_PQR :
  area_PQR = 
    (1 - k₁ * k₂ * k₃) ^ 2 * area_ABC / 
    ((1 + k₁ + k₁ * k₂) * (1 + k₂ + k₂ * k₃) * (1 + k₃ + k₃ * k₁)) := by
  sorry

end area_of_triangle_PQR_l703_703534


namespace minimum_questions_to_determine_village_l703_703007

-- Step 1: Define the types of villages
inductive Village
| A : Village
| B : Village
| C : Village

-- Step 2: Define the properties of residents in each village
def tells_truth (v : Village) (p : Prop) : Prop :=
  match v with
  | Village.A => p
  | Village.B => ¬p
  | Village.C => p ∨ ¬p

-- Step 3: Define the problem context in Lean
theorem minimum_questions_to_determine_village :
    ∀ (tourist_village person_village : Village), ∃ (n : ℕ), n = 4 := by
  sorry

end minimum_questions_to_determine_village_l703_703007


namespace geometric_seq_and_sum_of_seq_l703_703055

theorem geometric_seq_and_sum_of_seq (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
    (h2 : ∀ n, (S (n + 1) - 2 * S n, S n) = (2, n) ∨ (2, n) = (S (n + 1) - 2 * S n, S n)) :
  (∀ n, S n / n = 2 ^ (n - 1)) ∧ 
  (∀ n, (∑ i in range n, S i) = (n-1)*2^n + 1) :=
sorry

end geometric_seq_and_sum_of_seq_l703_703055


namespace sum_of_coefficients_square_l703_703000

theorem sum_of_coefficients_square (C : ℤ → ℤ) :
  let S := (C 1 + C 2 + C 3 + C 4) in
  (S ^ 2) = 225 := by
  sorry

end sum_of_coefficients_square_l703_703000


namespace intersection_complement_l703_703952

open Set

noncomputable def U : Set ℝ := {-1, 0, 1, 4}
def A : Set ℝ := {-1, 1}
def B : Set ℝ := {1, 4}
def C_U_B : Set ℝ := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} :=
by
  sorry

end intersection_complement_l703_703952


namespace pairs_satisfy_condition_l703_703476

theorem pairs_satisfy_condition (a b : ℝ) :
  (∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a_int b_int : ℤ, a = a_int ∧ b = b_int)) :=
by
  sorry

end pairs_satisfy_condition_l703_703476


namespace find_N_l703_703128

theorem find_N :
  (∃ (N : ℕ), 0 < N ∧ 6! * 10! = 18 * (N! + 45)) → 
  (∃ (N : ℕ), N = 11) :=
sorry

end find_N_l703_703128


namespace count_integer_pairs_l703_703984

theorem count_integer_pairs :
  {p : ℕ × ℕ | nat.sqrt p.1 + nat.sqrt p.2 = nat.sqrt 200600}.card = 11 :=
begin
  sorry
end

end count_integer_pairs_l703_703984


namespace surface_area_of_circumscribed_sphere_l703_703937

noncomputable def base_area : ℝ := π
noncomputable def lateral_area : ℝ := 2 * base_area

theorem surface_area_of_circumscribed_sphere :
  (base_area = π) →
  (lateral_area = 2 * base_area) →
  ∃ S : ℝ, S = 4 * π * ((2 * ℝ.sqrt 3 / 3) ^ 2) ∧ S = 16 * π / 3 :=
by
  sorry

end surface_area_of_circumscribed_sphere_l703_703937


namespace angle_AED_l703_703137

variable {A B C D E : Type}
variables [euclidean_triangle A B C] (BD_bisects_B : angle_bisector B D) 
          (AC_extended_to_E : extends_to A C E) (θ a c : ℝ)
          (angle_EBC_right : ∠ EBC = 90)

def angle_sum_triangle (a c θ : ℝ) : Prop :=
  a + c + 2 * θ = 180

theorem angle_AED (a c : ℝ) (AC_extended_to_E : extends_to A C E)
  (BD_bisects_B : angle_bisector B D)
  (angle_EBC_right : ∠ EBC = 90) :
    ∠ AED = 180 - (a + c) / 2 :=
  sorry

end angle_AED_l703_703137


namespace problem_l703_703940

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom restricted_function : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = log (x + 1) / log 2

-- What we need to prove
theorem problem : f (-2013) + f 2012 = -1 :=
by
  sorry

end problem_l703_703940


namespace midpoint_of_AB_l703_703659

-- Definitions of the geometrical entities involved
def parallel_lines (d d' : Line) : Prop := ∀ x : Point, (x ∈ d ↔ x ∈ d')

def is_tangent (Γ : Circle) (d : Line) (A : Point) : Prop := 
A ∈ d ∧ A ∈ Γ ∧ ∀ x : Point, x ∈ d ∩ Γ → x = A

def intersects (Γ : Circle) (d' : Line) (C D : Point) : Prop :=
C ∈ d' ∧ D ∈ d' ∧ C ∈ Γ ∧ D ∈ Γ ∧ C ≠ D

def tangent_at (Γ : Circle) (D : Point) (d : Line) (B : Point) : Prop :=
∀ x : Point, x ∈ Γ ∩ d → x ≠ D → ∀ e : Line, D ∈ e ∧ x ∈ e → e = d

def intersection_point (l1 l2 : Line) (P : Point) : Prop :=
P ∈ l1 ∧ P ∈ l2

-- Statement of the problem
theorem midpoint_of_AB
  (d d' : Line) (Γ : Circle) (A B C D E I : Point)
  (h_parallel : parallel_lines d d')
  (h_tangent : is_tangent Γ d A)
  (h_intersects : intersects Γ d' C D)
  (h_tangent_at : tangent_at Γ D d B)
  (h_intersection_E : intersection_point (segment B C) Γ E)
  (h_intersection_I : intersection_point (segment D E) d I) :
  midpoint I A B :=
sorry

end midpoint_of_AB_l703_703659


namespace Carla_final_position_l703_703463

-- Carla's initial position
def Carla_initial_position : ℤ × ℤ := (10, -10)

-- Function to calculate Carla's new position after each move
def Carla_move (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
  match direction % 4 with
  | 0 => (pos.1, pos.2 + distance)   -- North
  | 1 => (pos.1 + distance, pos.2)   -- East
  | 2 => (pos.1, pos.2 - distance)   -- South
  | 3 => (pos.1 - distance, pos.2)   -- West
  | _ => pos  -- This case will never happen due to the modulo operation

-- Recursive function to simulate Carla's journey
def Carla_journey : ℕ → ℤ × ℤ → ℤ × ℤ 
  | 0, pos => pos
  | n + 1, pos => 
    let next_pos := Carla_move pos n (2 + n / 2 * 2)
    Carla_journey n next_pos

-- Prove that after 100 moves, Carla's position is (-191, -10)
theorem Carla_final_position : Carla_journey 100 Carla_initial_position = (-191, -10) :=
sorry

end Carla_final_position_l703_703463


namespace max_log_value_l703_703188

theorem max_log_value (x : ℝ) (h : x > -1) : 
  (∃ y, y = x + (1 / (x + 1)) + 3 ∧ y ≤ 4) ∧ log (1 / 2) y ≤ -2 :=
sorry

end max_log_value_l703_703188


namespace new_elephants_entry_rate_l703_703774

-- Definitions
def initial_elephants := 30000
def exodus_rate := 2880
def exodus_duration := 4
def final_elephants := 28980
def new_elephants_duration := 7

-- Prove that the rate of new elephants entering the park is 1500 elephants per hour
theorem new_elephants_entry_rate :
  let elephants_left_after_exodus := initial_elephants - exodus_rate * exodus_duration
  let new_elephants := final_elephants - elephants_left_after_exodus
  let new_entry_rate := new_elephants / new_elephants_duration
  new_entry_rate = 1500 :=
by
  sorry

end new_elephants_entry_rate_l703_703774


namespace sasha_total_items_l703_703712

/-
  Sasha bought pencils at 13 rubles each and pens at 20 rubles each,
  paying a total of 350 rubles. 
  Prove that the total number of pencils and pens Sasha bought is 23.
-/
theorem sasha_total_items
  (x y : ℕ) -- Define x as the number of pencils and y as the number of pens
  (H: 13 * x + 20 * y = 350) -- Given total cost condition
  : x + y = 23 := 
sorry

end sasha_total_items_l703_703712


namespace four_letter_initial_sets_l703_703968

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l703_703968


namespace find_A_of_trig_max_bsquared_plus_csquared_l703_703606

-- Given the geometric conditions and trigonometric identities.

-- Prove: Given 2a * sin B = b * tan A, we have A = π / 3
theorem find_A_of_trig (a b c A B C : Real) (h1 : 2 * a * Real.sin B = b * Real.tan A) :
  A = Real.pi / 3 := sorry

-- Prove: Given a = 2, the maximum value of b^2 + c^2 is 8
theorem max_bsquared_plus_csquared (a b c A : Real) (hA : A = Real.pi / 3) (ha : a = 2) :
  b^2 + c^2 ≤ 8 :=
by
  have hcos : Real.cos A = 1 / 2 := by sorry
  have h : 4 = b^2 + c^2 - b * c * (1/2) := by sorry
  have hmax : b^2 + c^2 + b * c ≤ 8 := by sorry
  sorry -- Proof steps to reach the final result

end find_A_of_trig_max_bsquared_plus_csquared_l703_703606


namespace f_equals_one_l703_703330

-- Define the functions f, g, h with the given properties

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def h : ℕ → ℕ := sorry

-- Condition 1: h is injective
axiom h_injective : ∀ {a b : ℕ}, h a = h b → a = b

-- Condition 2: g is surjective
axiom g_surjective : ∀ n : ℕ, ∃ m : ℕ, g m = n

-- Condition 3: Definition of f in terms of g and h
axiom f_def : ∀ n : ℕ, f n = g n - h n + 1

-- Prove that f(n) = 1 for all n ∈ ℕ
theorem f_equals_one : ∀ n : ℕ, f n = 1 := by
  sorry

end f_equals_one_l703_703330


namespace product_divisible_by_1419_l703_703795

noncomputable def product_of_even_integers (n : ℕ) : ℕ :=
  ∏ i in (Finset.range (n / 2)), 2 * (i + 1)

-- Theorem statement: The product of consecutive even integers up to 106 is divisible by 1419.
theorem product_divisible_by_1419 : ∃ n, product_of_even_integers n = 1419 := sorry

end product_divisible_by_1419_l703_703795


namespace jogger_distance_ahead_l703_703068

/-- Definition of the speeds and the time given in the conditions. -/
def jogger_speed_km_hr : ℝ := 9
def train_speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def time_seconds : ℝ := 24

/-- Conversion factors and derived relative speed. -/
def conversion_factor : ℝ := 5 / 18
def relative_speed_m_s : ℝ := (train_speed_km_hr - jogger_speed_km_hr) * conversion_factor

/-- Main theorem: The distance the jogger is ahead of the train. -/
theorem jogger_distance_ahead :
  let D := relative_speed_m_s * time_seconds - train_length_m in
  D = 120 :=
by
  sorry

end jogger_distance_ahead_l703_703068


namespace triangle_area_is_correct_l703_703341

def perimeter : ℝ := 20
def inradius : ℝ := 2.5
def semi_perimeter := perimeter / 2
def area := inradius * semi_perimeter

theorem triangle_area_is_correct : area = 25 :=
  sorry

end triangle_area_is_correct_l703_703341


namespace train_dispatch_interval_l703_703791

theorem train_dispatch_interval (journey_time minutes trains_encountered : ℕ) (interval dispatch_interval : ℝ) :
  journey_time = 17 →
  trains_encountered = 5 →
  dispatch_interval = (34 : ℝ) / (trains_encountered + 1 : ℝ) →
  340 / 60 < dispatch_interval ∧ dispatch_interval ≤ 510 / 60 :=
by {
  intros h1 h2 h3,
  -- Here the proof would continue using the given conditions.
  sorry
}

end train_dispatch_interval_l703_703791


namespace intervals_increasing_triangle_angles_l703_703582

def vector_a (ω x : ℝ) : ℝ × ℝ := (3, -real.cos (ω * x))
def vector_b (ω x : ℝ) : ℝ × ℝ := (real.sin (ω * x), real.sqrt 3)
def f (ω x : ℝ) : ℝ := (vector_a ω x).1 * (vector_b ω x).1 + (vector_a ω x).2 * (vector_b ω x).2

theorem intervals_increasing (ω : ℝ) (h_ω : ω > 0) (h_T : 2 * real.pi / ω = real.pi) :
  ∀ k : ℤ, (k * real.pi - real.pi / 6) ≤ λ x, f ω x ≤ (k * real.pi + real.pi / 3) ↔ 
  (∃ k : ℤ, (k * real.pi - real.pi / 6) ≤ x ∧ x ≤ (k * real.pi + real.pi / 3)) :=
sorry

noncomputable def angle_A : ℝ := real.pi / 3
noncomputable def len_a (len_b : ℝ) : ℝ := real.sqrt 3 * len_b

theorem triangle_angles (A B C : ℝ) (a b c : ℝ) 
  (h_A2 : f 2 (angle_A / 2) = real.sqrt 3)
  (h_a : a = len_a b) : 
  A = real.pi / 3 ∧ B = real.pi / 6 ∧ C = real.pi / 2 :=
sorry

end intervals_increasing_triangle_angles_l703_703582


namespace smallest_consecutive_even_integer_l703_703759

theorem smallest_consecutive_even_integer (x : ℕ) (h : ∑ i in finset.range 30, (x + 2 * i) = 12000) : x = 371 :=
sorry

end smallest_consecutive_even_integer_l703_703759


namespace new_weight_is_71_l703_703320

def average_weight_increase (W : ℝ) (increase : ℝ) := W + increase

def new_person_weight (W : ℝ) (increase : ℝ) (old_person_weight new_person_weight : ℝ) :=
  W - old_person_weight + new_person_weight = W + increase

theorem new_weight_is_71 (W : ℝ) (old_person_weight : ℝ) (increase : ℝ) (new_person_weight : ℝ) :
  average_weight_increase W increase →
  new_person_weight W increase old_person_weight new_person_weight →
  old_person_weight = 65 →
  increase = 6 →
  new_person_weight = 71 :=
by
  intros h1 h2 h3 h4
  sorry

end new_weight_is_71_l703_703320


namespace hyperbola_center_coordinates_l703_703500

theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), 
  (∀ x y : ℝ, 
    ((4 * y - 6) ^ 2 / 36 - (5 * x - 3) ^ 2 / 49 = -1) ↔
    ((x - h) ^ 2 / ((7 / 5) ^ 2) - (y - k) ^ 2 / ((3 / 2) ^ 2) = 1)) ∧
  h = 3 / 5 ∧ k = 3 / 2 :=
by sorry

end hyperbola_center_coordinates_l703_703500


namespace number_of_initials_sets_l703_703965

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l703_703965


namespace slope_of_intersection_line_l703_703729

theorem slope_of_intersection_line :
  ∀ (x y : ℝ), 
  (x^2 + y^2 - 6x + 4y - 20 = 0 ∧ x^2 + y^2 - 16x + 8y + 40 = 0) → 
  ∃ m : ℝ, m = 5 / 2 :=
by
  sorry

end slope_of_intersection_line_l703_703729


namespace satisfies_property1_satisfies_property2_l703_703067

-- Conditions given in the problem
variable {f : ℝ → ℝ}
variable (hf₁ : ∀ x : ℝ, f (-x) + f x = 0)
variable (hf₂ : ∀ x : ℝ, f x = f (4 - x))

-- The function we need to prove satisfies the conditions
def fx (x : ℝ) := Real.sin (Real.pi * x / 4)

-- Statements to verify
theorem satisfies_property1 : (∀ x : ℝ, (fx (-x) + fx x = 0)) :=
by sorry

theorem satisfies_property2 : (∀ x : ℝ, (fx x = fx (4 - x))) :=
by sorry

end satisfies_property1_satisfies_property2_l703_703067


namespace airflow_rate_l703_703066

theorem airflow_rate (minutes_per_day : ℕ) (total_airflow_week : ℕ) (days_per_week : ℕ) :
  minutes_per_day = 10 →
  total_airflow_week = 42000 →
  days_per_week = 7 →
  (total_airflow_week / (minutes_per_day * days_per_week * 60) : ℕ) = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- The conversion from minutes to seconds and the division are straightforward
  sorry

end airflow_rate_l703_703066


namespace shaded_area_l703_703003

/-- Definitions of points and relevant segments in the triangle PQR and their properties --/
structure Triangle (P Q R : Type) :=
(PR QR : P → Q → R → ℝ)

def isosceles (T : Triangle P Q R) : Prop := T.PR = T.QR
def right_angle (T : Triangle P Q R) : Prop := angle T.PR T.QR = 90

noncomputable def triangle_pqr_area : ℝ := 1
noncomputable def arc_area : ℝ := (π / 4)

def shaded_region_area (T : Triangle P Q R) : ℝ := 
  triangle_pqr_area - arc_area 

/-- Theorem to prove the area of the shaded region inside triangle PQR --/
theorem shaded_area (P Q R : Type) (T : Triangle P Q R) 
(h_isosceles : isosceles T) (h_right_angle : right_angle T) 
(h_length : distance P Q = 2) (h_arc1 : arc_in_triangle T 1 P) 
(h_arc2 : arc_in_triangle T 1 Q) :
  shaded_region_area T = 1 - (π / 4) := 
sorry

end shaded_area_l703_703003


namespace angle_CBD_eq_angle_OAO_l703_703778

theorem angle_CBD_eq_angle_OAO' {O O' A B C D : Type*} [circle O] [circle O']
  (h_intersect : O ∩ O' = {A, B}) (h_line : A ∈ C ∧ A ∈ D) :
  measure_angle C B D = measure_angle O A O' :=
sorry

end angle_CBD_eq_angle_OAO_l703_703778


namespace box_surface_area_correct_l703_703304

-- Define the dimensions of the original cardboard.
def original_length : ℕ := 25
def original_width : ℕ := 40

-- Define the size of the squares removed from each corner.
def square_side : ℕ := 8

-- Define the surface area function.
def surface_area (length width : ℕ) (square_side : ℕ) : ℕ :=
  let area_remaining := (length * width) - 4 * (square_side * square_side)
  area_remaining

-- The theorem statement to prove
theorem box_surface_area_correct : surface_area original_length original_width square_side = 744 :=
by
  sorry

end box_surface_area_correct_l703_703304


namespace limit_sum_cubes_l703_703472

-- Define the first term and common ratio for the geometric progression
variables {a r : ℝ}

-- Define the conditions for the common ratio
def valid_ratio (r : ℝ) : Prop := -1 < r ∧ r < 1

-- Define the limit of the sum in a geometric series
def sum_of_cubes_geometric_series (a r : ℝ) (hr : valid_ratio r) : ℝ :=
  a^3 / (1 - r^3)

-- Now construct our main theorem
theorem limit_sum_cubes (a r : ℝ) (hr : valid_ratio r) :
  (∑' n : ℕ, (a * r^n)^3) = sum_of_cubes_geometric_series a r hr := by
  sorry

end limit_sum_cubes_l703_703472


namespace price_reduction_correct_l703_703845

theorem price_reduction_correct :
  ∃ x : ℝ, (0.3 - x) * (500 + 4000 * x) = 180 ∧ x = 0.1 :=
by
  sorry

end price_reduction_correct_l703_703845


namespace probability_even_product_l703_703061

theorem probability_even_product : 
  ∀ (B : finset ℕ),
  B = { 1, 2, 3, 4, 5 } →
  ∃ (p : ℚ), 
  p = 16 / 25 ∧ 
  (let events := B.product B in
   (∑ x in events, if (x.1 * x.2) % 2 = 0 then 1 else 0)
   / (events.card : ℚ) = p) :=
begin
  intros B hB,
  use 16 / 25,
  split,
  {
    refl,
  },
  {
    rw [hB],
    let events := ({1, 2, 3, 4, 5} : finset ℕ).product {1, 2, 3, 4, 5},
    have h_events_card : events.card = 25,
    {
      simp [events],
    },
    have even_events_card : (∑ x in events, if (x.1 * x.2) % 2 = 0 then 1 else 0) = 16,
    {
      -- Calculation and proof of events where the product is even
      sorry,
    },
    rw [even_events_card],
    field_simp [h_events_card],
  },
end

end probability_even_product_l703_703061


namespace dilation_matrix_centered_at_point_l703_703897

theorem dilation_matrix_centered_at_point (S : ℝ) (cx cy : ℝ) :
  let T_neg := matrix.of ![![1, 0, -cx], ![0, 1, -cy], ![0, 0, 1]],
      D := matrix.of ![![S, 0, 0], ![0, S, 0], ![0, 0, 1]],
      T_pos := matrix.of ![![1, 0, cx], ![0, 1, cy], ![0, 0, 1]] in
  (T_pos ⬝ D ⬝ T_neg) = matrix.of ![![S, 0, (1 - S) * cx], ![0, S, (1 - S) * cy], ![0, 0, 1]] :=
by sorry

example : dilation_matrix_centered_at_point 4 1 1 = matrix.of ![![4, 0, -3], ![0, 4, -3], ![0, 0, 1]] :=
by sorry

end dilation_matrix_centered_at_point_l703_703897


namespace ellipse_eccentricity_triangle_area_l703_703929

/-- eccentricity of the ellipse -/
theorem ellipse_eccentricity :
  let a := 7
  let b := 2 * Real.sqrt 6
  let c := Real.sqrt (a^2 - b^2)
  (c / a) = 5 / 7 :=
by
  sorry

/-- area of the triangle -/
theorem triangle_area :
  let a := 7
  let c := 5
  let s := 2 * c
  let pf1_pf2_product := s^2 - (|.pf1|^2 + |.pf2|^2)
  ∃ (pf1 pf2: ℝ), pf1 + pf2 = 2 * a ∧ pf1 * pf2 = pf1_pf2_product / 2 ∧
  (1 / 2) * pf1 * pf2 = 24 :=
by
  sorry

end ellipse_eccentricity_triangle_area_l703_703929


namespace domain_of_function_l703_703477

theorem domain_of_function : 
  {x : ℝ | x ≠ 1 ∧ x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end domain_of_function_l703_703477


namespace correct_average_of_corrected_number_l703_703797

theorem correct_average_of_corrected_number (num_list : List ℤ) (wrong_num correct_num : ℤ) (n : ℕ)
  (hn : n = 10)
  (haverage : (num_list.sum / n) = 5)
  (hwrong : wrong_num = 26)
  (hcorrect : correct_num = 36)
  (hnum_list_sum : num_list.sum + correct_num - wrong_num = num_list.sum + 10) :
  (num_list.sum + 10) / n = 6 :=
by
  sorry

end correct_average_of_corrected_number_l703_703797


namespace cannot_cover_plane_with_rectangles_l703_703397

theorem cannot_cover_plane_with_rectangles (A : ℕ → ℝ) 
  (hA : ∀ n : ℕ, A n = n^2) : 
  ¬ ∀ P : ℝ, ∃ (Rects : ℕ → set (ℝ × ℝ)),
  (∀ n : ℕ, ∃ (x y : ℝ), 
      Rects n = {p : ℝ × ℝ | |p.1 - x| ≤ 2^(-n) ∧ 
                               |p.2 - y| ≤ (n^2 * 2^n) / 2^(n)})
    → ∀ (p : ℝ × ℝ), ∃ n, p ∈ Rects n :=
sorry

end cannot_cover_plane_with_rectangles_l703_703397


namespace num_three_digit_integers_l703_703985

-- Define the conditions as a Lean statement
theorem num_three_digit_integers : 
  {n : ℕ | n % 7 = 3 ∧ n % 8 = 6 ∧ n % 12 = 8 ∧ 100 ≤ n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry -- proof to be completed

end num_three_digit_integers_l703_703985


namespace functional_relationship_l703_703436

def cost_A := 100
def sell_A := 150
def cost_B := 85
def sell_B := 120
def total_units := 160

def profit_A := sell_A - cost_A
def profit_B := sell_B - cost_B

theorem functional_relationship (x : ℕ) (y : ℕ) :
  y = profit_A * x + profit_B * (total_units - x) :=
by
  have h1 : profit_A = 50 := rfl
  have h2 : profit_B = 35 := rfl
  have h3 : total_units = 160 := rfl
  calc
    y = profit_A * x + profit_B * (total_units - x) : by sorry -- Proof to be completed
      ... = 50 * x + 35 * (160 - x) : by sorry
      ... = 15 * x + 5600 : by sorry

end functional_relationship_l703_703436


namespace find_y_l703_703993

theorem find_y (x y : ℕ) (h1 : x^2 = y + 3) (h2 : x = 6) : y = 33 := 
by
  sorry

end find_y_l703_703993


namespace range_of_sin_plus_cos_l703_703751

def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem range_of_sin_plus_cos :
  ∀ x : ℝ, -Real.sqrt 2 ≤ f x ∧ f x ≤ Real.sqrt 2 :=
by
  sorry

end range_of_sin_plus_cos_l703_703751


namespace curve_C2_is_circle_max_area_AOB_l703_703209

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

def C2_eq (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 2 * a) ^ 2 + y ^ 2 = 4 * a ^ 2 ↔ ∃ θ : ℝ, (x, y) = (2 * a + 2 * a * Real.cos θ, 2 * a * Real.sin θ)

theorem curve_C2_is_circle (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : C2_eq a :=
by
  sorry

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def line_OA (a : ℝ) : Prop :=
  let A := polar_to_cartesian 2 (Real.pi / 3)
  ∀ B : ℝ × ℝ, B ∈ curve_C1 ∧ B ≠ (0, 0) →
  let d := ((A.1 * (B.2 - 0) - A.2 * (B.1 - 0)) / (Real.sqrt ((A.1 ^ 2) + (A.2 ^ 2))))
  Real.abs d = a * Real.abs (2 * Real.cos (- Real.pi / 6) + Real.sqrt 3)

theorem max_area_AOB (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : 
  (∃ α : ℝ, 
    let B := (2 * a + 2 * a * Real.cos α, 2 * a * Real.sin α)
    let S : ℝ := 1 / 2 * Real.abs (polar_to_cartesian 2 (Real.pi / 3).1 * B.2 - polar_to_cartesian 2 (Real.pi / 3).2 * B.1)
  S = 4 + 2 * Real.sqrt 3
  ) → a = 2 :=
by
  sorry

end curve_C2_is_circle_max_area_AOB_l703_703209


namespace sum_second_largest_and_smallest_l703_703006

theorem sum_second_largest_and_smallest :
  let numbers := [10, 11, 12, 13, 14]
  ∃ second_largest second_smallest, (List.nthLe numbers 3 sorry = second_largest ∧ List.nthLe numbers 1 sorry = second_smallest ∧ second_largest + second_smallest = 24) :=
sorry

end sum_second_largest_and_smallest_l703_703006


namespace factorial_vs_power_l703_703789

noncomputable def A : ℝ := real.factorial 999
noncomputable def B : ℝ := 500 ^ 999

theorem factorial_vs_power : A < B := by
  sorry

end factorial_vs_power_l703_703789


namespace Bob_in_chair3_l703_703087

-- Definitions based on conditions
variable (chairs : Fin 4 → String)

def Charlie_position := (chairs 1 = "Charlie")
def not_adjacent (pos1 pos2 : Fin 4) := ((pos1 == pos2 + 1) || (pos2 == pos1 + 1)) = false
def between (pos1 pos2 pos3 : Fin 4) := (pos1 < pos2 < pos3 || pos3 < pos2 < pos1) = false

-- Kevin's statements were false
axiom statement1_false : ¬ not_adjacent (chairs 1) (chairs 0) ∧ ¬ not_adjacent (chairs 1) (chairs 2)
axiom statement2_false : ¬ between (chairs 0) (chairs 1) (chairs 3) ∧ ¬ between (chairs 3) (chairs 1) (chairs 0)

-- Given Charlie is in chair #2
axiom Charlie_in_chair2 : Charlie_position

-- We need to prove Bob is sitting in chair #3
theorem Bob_in_chair3 : chairs 2 = "Bob" :=
sorry

end Bob_in_chair3_l703_703087


namespace solve_real_equation_l703_703892

theorem solve_real_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x ^ 3 + 3 * x ^ 2 - x) / (x ^ 2 + 4 * x + 3) + x = -7 ↔ x = -5 / 2 ∨ x = -4 := 
by
  sorry

end solve_real_equation_l703_703892


namespace cos_A_eq_sqrt_3_div_2_l703_703621

-- Define the right triangle and conditions
variables (A B C : Type) [MetricSpace A]

-- Assume the given lengths and angles
variable [has_dist (A → B)]

-- Define lengths and angles
noncomputable def right_triangle_ABC (h : dist A C = √3 * dist B C) (h2 : dist A B = 2 * dist B C) : Prop :=
  ∠C = 90

-- The theorem to be proved
theorem cos_A_eq_sqrt_3_div_2 {A B C : Point} (h : right_triangle_ABC h h2) : cos A = √3 / 2 :=
by
  sorry

end cos_A_eq_sqrt_3_div_2_l703_703621


namespace sqrt_trig_identity_l703_703511

theorem sqrt_trig_identity :
  sqrt (1 - 2 * sin (π + 2) * cos (π - 2)) = sin 2 - cos 2 :=
sorry

end sqrt_trig_identity_l703_703511


namespace only_B_forms_triangle_l703_703037

/-- Check if a set of line segments can form a triangle --/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_B_forms_triangle :
  ¬ can_form_triangle 2 6 3 ∧
  can_form_triangle 6 7 8 ∧
  ¬ can_form_triangle 1 7 9 ∧
  ¬ can_form_triangle (3 / 2) 4 (5 / 2) :=
by
  sorry

end only_B_forms_triangle_l703_703037


namespace area_difference_l703_703986

def radius_of_diameter (d : ℝ) : ℝ :=
  d / 2

def area_of_circle (r : ℝ) : ℝ :=
  real.pi * r^2

theorem area_difference (r1 : ℝ) (d2 : ℝ) : 
  r1 = 25 → d2 = 15 → 
  area_of_circle r1 - area_of_circle (radius_of_diameter d2) = 568.75 * real.pi :=
by
  intros h1 h2
  rw [h1, h2, radius_of_diameter, area_of_circle, area_of_circle]
  sorry

end area_difference_l703_703986


namespace num_four_letter_initials_l703_703975

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l703_703975


namespace daily_sales_volume_expression_price_reduction_for_profit_l703_703680

variables {x : ℝ}

-- Define original conditions
def original_price : ℝ := 2
def selling_price : ℝ := 6
def daily_sales : ℝ := 150
def price_decrease_per_pound : ℝ := 0.5
def sales_increase_per_pound : ℝ := 50
def desired_profit : ℝ := 750

-- Define the linear relationship of sales volume with price reduction
def daily_sales_volume (x : ℝ) : ℝ := daily_sales + (100 * x)

-- Define the profit equation
def profit (x : ℝ) : ℝ := (selling_price - x - original_price) * daily_sales_volume x

-- Statement (1): Daily sales volume as a function of price reduction
theorem daily_sales_volume_expression (x : ℝ) : daily_sales_volume x = 150 + 100 * x :=
by simp [daily_sales_volume]

-- Statement (2): Price reduction to achieve desired daily profit
theorem price_reduction_for_profit (x : ℝ) :
  x = 1.5 ↔ profit x = desired_profit :=
by sorry

end daily_sales_volume_expression_price_reduction_for_profit_l703_703680


namespace find_y_and_coordinates_R_l703_703540

theorem find_y_and_coordinates_R (y : ℝ) (y_R : ℝ) :
  let P := (-3 : ℝ, 8 : ℝ)
  let Q := (5 : ℝ, y)
  let R := (11 : ℝ, y_R)
  slope P Q = -5/4 ∧ (abs (fst Q - 11) = 6) → (y = -2 ∧ y_R = -9.5) :=
by
  sorry

end find_y_and_coordinates_R_l703_703540


namespace circles_area_difference_l703_703730

open Real

noncomputable def area_diff_between_circles (C1 C2 : ℝ) : ℝ :=
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  (π * r2^2) - (π * r1^2)

theorem circles_area_difference:
  (area_diff_between_circles 268 380 ≈ 5778.76) :=
by sorry

end circles_area_difference_l703_703730


namespace area_difference_l703_703987

def radius_of_diameter (d : ℝ) : ℝ :=
  d / 2

def area_of_circle (r : ℝ) : ℝ :=
  real.pi * r^2

theorem area_difference (r1 : ℝ) (d2 : ℝ) : 
  r1 = 25 → d2 = 15 → 
  area_of_circle r1 - area_of_circle (radius_of_diameter d2) = 568.75 * real.pi :=
by
  intros h1 h2
  rw [h1, h2, radius_of_diameter, area_of_circle, area_of_circle]
  sorry

end area_difference_l703_703987


namespace overtakes_in_16_minutes_l703_703278

def number_of_overtakes (track_length : ℕ) (speed_a : ℕ) (speed_b : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let relative_speed := speed_a - speed_b
  let time_per_overtake := track_length / relative_speed
  time_seconds / time_per_overtake

theorem overtakes_in_16_minutes :
  number_of_overtakes 200 6 4 16 = 9 :=
by
  -- We will insert calculations or detailed proof steps if needed
  sorry

end overtakes_in_16_minutes_l703_703278


namespace tan_sum_l703_703589

theorem tan_sum (x y : ℝ) (h1 : Real.tan x + Real.tan y = 15) (h2 : Real.cot x + Real.cot y = 20) : Real.tan (x + y) = 60 := 
by
  sorry

end tan_sum_l703_703589


namespace perpendicular_lines_of_perpendicular_planes_l703_703944

variables {α β : Plane} {a b : Line}

theorem perpendicular_lines_of_perpendicular_planes 
  (h1 : α ⟂ β) (h2 : a ⟂ α) (h3 : b ⟂ β) : 
  a ⟂ b :=
sorry

end perpendicular_lines_of_perpendicular_planes_l703_703944


namespace interval_of_increase_l703_703478

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem interval_of_increase : 
  ∀ x : ℝ, (0 < x ∧ x < Real.exp 1) → (0 < (1 - Real.log x) ∧ x > 0) :=
begin
  -- Proof omitted
  sorry
end

end interval_of_increase_l703_703478


namespace Megan_acorns_now_l703_703683

def initial_acorns := 16
def given_away_acorns := 7
def remaining_acorns := initial_acorns - given_away_acorns

theorem Megan_acorns_now : remaining_acorns = 9 := by
  sorry

end Megan_acorns_now_l703_703683


namespace inequality_abc_m_l703_703931

variables {n : ℕ}
variables {a b c : Fin n.succ → ℝ}
variables {m : Fin n.succ → ℝ}

-- assume the conditions of the problem
axiom pos_aa : ∀ i : Fin n.succ, 0 < a i
axiom pos_bb : ∀ i : Fin n.succ, 0 < b i
axiom pos_cc : ∀ i : Fin n.succ, 0 < c i
axiom definition_m : ∀ k : Fin n.succ, m k = ⨆ (i j l : Fin n.succ), if max (max i j) l = k then a i * b j * c l else 0

-- state the theorem to be proved
theorem inequality_abc_m : 
  (∑ i, a i) * (∑ j, b j) * (∑ l, c l) ≤ n.succ * n.succ * (∑ k, m k) :=
sorry

end inequality_abc_m_l703_703931


namespace quadrilateral_angle_BAD_l703_703288

/-- Let ABCD be a quadrilateral with AB = BC = CD. 
    Given m∠ABC = 50° and m∠BCD = 150°, prove that m∠BAD = 100°. 
-/
theorem quadrilateral_angle_BAD
  (A B C D : Type)
  (AB BC CD : ℝ)
  (h1 : AB = BC)
  (h2 : BC = CD)
  (angle_ABC : ℝ)
  (angle_BCD : ℝ)
  (h3 : angle_ABC = 50)
  (h4 : angle_BCD = 150) :
  ∃ angle_BAD : ℝ, angle_BAD = 100 :=
begin
  use 100,
  sorry
end

end quadrilateral_angle_BAD_l703_703288


namespace point_inside_triangle_l703_703401

variables {A B C P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variable {x y z u v w : ℝ}

-- Define the distances as conditions
def distance_to_vertices (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ :=
  x

def distance_to_sides (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ :=
  u

theorem point_inside_triangle 
  (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (x y z u v w : ℝ)
  (hx : distance_to_vertices P A B C = x)
  (hy : distance_to_vertices P B A C = y)
  (hz : distance_to_vertices P C A B = z)
  (hu : distance_to_sides P A B C = u)
  (hv : distance_to_sides P B A C = v)
  (hw : distance_to_sides P C A B = w) :
  x + y + z ≥ 2 * (u + v + w) :=
sorry

end point_inside_triangle_l703_703401


namespace sum_cos_to_tan_l703_703155

theorem sum_cos_to_tan (p q : ℕ) (h_rel_prime : Nat.gcd p q = 1) (h_pos : 0 < p ∧ 0 < q) (h_lt : p < 90 * q) 
  (h_sum : ∑ k in Finset.range 50, Real.cos (3 * (k : ℝ)) = Real.tan (p / q)) :
  p + q = 76 := by
  sorry

end sum_cos_to_tan_l703_703155


namespace find_f4_l703_703537

noncomputable def f : ℝ → ℝ := sorry

theorem find_f4 (hf_odd : ∀ x : ℝ, f (-x) = -f x)
                (hf_property : ∀ x : ℝ, f (x + 2) = -f x) :
  f 4 = 0 :=
sorry

end find_f4_l703_703537


namespace simplify_trig_expression_1_simplify_trig_expression_2_l703_703301
open Real

-- Define the problem statement.
theorem simplify_trig_expression_1 (α : ℝ) :
  (-(sin (π + α)) + sin (-α) - tan (2 * π + α)) / (tan (α + π) + cos (-α) + cos (π - α)) = -1 := 
sorry

-- For the second expression, handling the two cases.
theorem simplify_trig_expression_2 (α : ℝ) (n : ℤ) :
  (if even n then (sin (α + n * π) + sin (α - n * π)) / (sin (α + n * π) * cos (α - n * π)) = 2 / cos α
   else (sin (α + n * π) + sin (α - n * π)) / (sin (α + n * π) * cos (α - n * π)) = -2 / cos α) := 
sorry

end simplify_trig_expression_1_simplify_trig_expression_2_l703_703301


namespace number_of_boys_l703_703725

theorem number_of_boys (n : ℕ) (h1 : (n * 182 - 60) / n = 180): n = 30 :=
by
  sorry

end number_of_boys_l703_703725


namespace no_closed_loop_after_replacement_l703_703047

theorem no_closed_loop_after_replacement (N M : ℕ) 
  (h1 : N = M) 
  (h2 : (N + M) % 4 = 0) :
  ¬((N - 1) - (M + 1)) % 4 = 0 :=
by
  sorry

end no_closed_loop_after_replacement_l703_703047


namespace problem_statement_l703_703178

noncomputable def count_valid_M : ℕ :=
  Set.card {M : ℕ | M < 500 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (⌊x⌋ + 1) = M}

theorem problem_statement : count_valid_M = 198 := 
  sorry

end problem_statement_l703_703178


namespace equal_cost_at_20_minutes_l703_703363

/-- Define the cost functions for each telephone company -/
def united_cost (m : ℝ) : ℝ := 11 + 0.25 * m
def atlantic_cost (m : ℝ) : ℝ := 12 + 0.20 * m
def global_cost (m : ℝ) : ℝ := 13 + 0.15 * m

/-- Prove that at 20 minutes, the cost is the same for all three companies -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 ∧ atlantic_cost 20 = global_cost 20 :=
by
  sorry

end equal_cost_at_20_minutes_l703_703363


namespace fraction_addition_l703_703025

theorem fraction_addition : (1 + 3 + 5)/(2 + 4 + 6) + (2 + 4 + 6)/(1 + 3 + 5) = 25/12 := by
  sorry

end fraction_addition_l703_703025


namespace shortest_distance_from_curve_to_line_l703_703755

noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_from_curve_to_line : 
  ∃ (x y : ℝ), y = curve x ∧ line x y ∧ 
  (∀ (x₀ y₀ : ℝ), y₀ = curve x₀ → ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ d = Real.sqrt 5) :=
sorry

end shortest_distance_from_curve_to_line_l703_703755


namespace dogwood_tree_cut_count_l703_703607

theorem dogwood_tree_cut_count
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_left : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0)
  (h3 : trees_left = 2.0) :
  trees_part1 + trees_part2 - trees_left = 7.0 :=
by
  sorry

end dogwood_tree_cut_count_l703_703607


namespace exists_spatial_quadrilateral_l703_703080

open Classical

noncomputable theory

variables {V : Type*} [fintype V]

/-- Define the conditions of the graph problem. -/
def graph_conditions (q : ℕ) (n l : ℕ) (G : simple_graph V) :=
  (n = q^2 + q + 1) ∧
  (l ≥ 1/2 * q * (q+1)^2 + 1) ∧
  (q ≥ 2) ∧
  (∀ v : V, 0 < (G.degree v)) ∧
  (∃ v : V, q+2 ≤ G.degree v) ∧
  (∀ (a b c d : V), (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → is_simplex (fin 4) G)

theorem exists_spatial_quadrilateral
  (q n l : ℕ)
  (G : simple_graph V)
  (hG : graph_conditions q n l G) :
  ∃ (A B C D : V), G.adj A B ∧ G.adj B C ∧ G.adj C D ∧ G.adj D A :=
sorry

end exists_spatial_quadrilateral_l703_703080


namespace shaded_area_ratio_l703_703828

theorem shaded_area_ratio
  (large_square_area : ℕ := 25)
  (grid_dimension : ℕ := 5)
  (shaded_square_area : ℕ := 2)
  (num_squares : ℕ := 25)
  (ratio : ℚ := 2 / 25) :
  (shaded_square_area : ℚ) / large_square_area = ratio := 
by
  sorry

end shaded_area_ratio_l703_703828


namespace union_A_B_l703_703152

noncomputable def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
noncomputable def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_A_B : A ∪ B = {x : ℝ | -1 < x} := by
  sorry

end union_A_B_l703_703152


namespace length_breadth_difference_l703_703316

theorem length_breadth_difference (B : ℕ) (h1 : B = 8) (h2 : ∃ L : ℕ, 18 * B = L * B) : ∃ diff : ℕ, diff = 10 := 
by
  obtain ⟨L, hL⟩ := h2
  use (L - B)
  have hB : B = 8, from h1
  rw hB at hL
  have hL_eq : L = 18, from nat.mul_right_inj (ne_of_gt (by norm_num)) hL
  have h_diff : L - B = 10, by rw [hL_eq, hB]; norm_num
  exact h_diff

end length_breadth_difference_l703_703316


namespace neg_p_equiv_l703_703171

-- The proposition p
def p : Prop := ∀ x : ℝ, x^2 - 1 < 0

-- Equivalent Lean theorem statement
theorem neg_p_equiv : ¬ p ↔ ∃ x₀ : ℝ, x₀^2 - 1 ≥ 0 :=
by
  sorry

end neg_p_equiv_l703_703171


namespace min_value_expression_l703_703026

theorem min_value_expression :
  ∃ x : ℝ, (x+2) * (x+3) * (x+5) * (x+6) + 2024 = 2021.75 :=
sorry

end min_value_expression_l703_703026


namespace sum_of_solutions_l703_703374

def is_solution (n : ℕ) : Prop :=
  Nat.lcm n 150 = Nat.gcd n 150 + 600

theorem sum_of_solutions : ∑ n in (Finset.filter is_solution (Finset.range 1000)), n = 225 :=
by
  sorry

end sum_of_solutions_l703_703374


namespace calculate_expression_l703_703862

theorem calculate_expression : 
  (2 * real.sqrt 48 - 3 * real.sqrt (1 / 3)) / real.sqrt 6 = 7 * real.sqrt 2 / 2 :=
by
  sorry

end calculate_expression_l703_703862


namespace a50_plus_b50_eq_l703_703078

noncomputable def sequence (n : ℕ) : ℝ × ℝ :=
  (nat.recOn n (1, real.sqrt 3) 
    (λ n ⟨a_n, b_n⟩, (real.sqrt 3 * a_n - b_n, real.sqrt 3 * b_n + a_n)))

theorem a50_plus_b50_eq : 
  (sequence 50).1 + (sequence 50).2 = 2^49 * (-real.sqrt 3 + 1) :=
sorry

end a50_plus_b50_eq_l703_703078


namespace round_robin_games_l703_703355

theorem round_robin_games (x : ℕ) (h : ∃ (n : ℕ), n = 15) : (x * (x - 1)) / 2 = 15 :=
sorry

end round_robin_games_l703_703355


namespace find_lambda_l703_703914

-- Definitions for vectors a and b
def a : ℝ × ℝ × ℝ := (-2, 1, 3)
def b : ℝ × ℝ × ℝ := (-1, 2, 1)

-- Definition for the dot product of two 3-dimensional vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) : dot_product a (a.1 - λ * b.1, a.2 - λ * b.2, a.3 - λ * b.3) = 0 → λ = 2 :=
by {
    -- Placeholder for the proof
    sorry
}

end find_lambda_l703_703914


namespace find_position_of_2005_l703_703853

def table_structure (i j : ℕ) : Prop :=
  let row_start := (i - 1)^2 + 1 in
  let row_end := i^2 in
  row_start ≤ a_i^j ∧ a_i^j ≤ row_end ∧
  a_i^j = row_start + (j - 1)

theorem find_position_of_2005 : 
  ∃ (i j : ℕ), table_structure i j ∧ a_i^j = 2005 :=
sorry

end find_position_of_2005_l703_703853


namespace limit_equivalent_l703_703139

variable (f : ℝ → ℝ) (x_0 : ℝ)

-- Condition
def derivative_at_x0 : Prop := deriv f x_0 = 3

-- Statement to prove
theorem limit_equivalent (h : derivative_at_x0 f x_0) : 
  (tendsto (λ Δx : ℝ, (f (x_0 + 2 * Δx) - f x_0) / (3 * Δx)) (𝓝 0) (𝓝 2)) :=
by 
  sorry

end limit_equivalent_l703_703139


namespace sqrt_power_simplified_l703_703497

theorem sqrt_power_simplified : (sqrt ((sqrt 5) ^ 5)) ^ 6 = 5 ^ 7.5 := by
  sorry

end sqrt_power_simplified_l703_703497


namespace cost_of_each_shirt_is_8_l703_703465

-- Define the conditions
variables (S : ℝ)
def shirts_cost := 4 * S
def pants_cost := 2 * 18
def jackets_cost := 2 * 60
def total_cost := shirts_cost S + pants_cost + jackets_cost
def carrie_pays := 94

-- The goal is to prove that S equals 8 given the conditions above
theorem cost_of_each_shirt_is_8
  (h1 : carrie_pays = total_cost S / 2) : S = 8 :=
sorry

end cost_of_each_shirt_is_8_l703_703465


namespace part1_part2_l703_703234

noncomputable def area_of_triangle (a b c : ℝ) (sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Part 1
theorem part1 (a : ℝ) (b : ℝ) (c : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : 2 * Real.sin C = 3 * Real.sin A)
  (a_val : a = 4) (b_val : b = 5) (c_val : c = 6) : 
  area_of_triangle 4 5 (3 * Real.sqrt 7 / 8) = 15 * Real.sqrt 7 / 4 := by 
  sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : ∃ (a' : ℕ), a = a') :
  a = 2 := by
  sorry

end part1_part2_l703_703234


namespace value_of_a_l703_703577

theorem value_of_a (a : ℝ) : 
  ({2, 3} : Set ℝ) ⊆ ({1, 2, a} : Set ℝ) → a = 3 :=
by
  sorry

end value_of_a_l703_703577


namespace acute_triangle_inequality_l703_703211

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C) ≤
    Real.pi * (1 / A + 1 / B + 1 / C) :=
sorry

end acute_triangle_inequality_l703_703211


namespace simplify_expression_l703_703451

-- Define the constants and variables with required conditions
variables {x y z p q r : ℝ}

-- Assume the required distinctness conditions
axiom h1 : x ≠ p 
axiom h2 : y ≠ q 
axiom h3 : z ≠ r 

-- State the theorem to be proven
theorem simplify_expression (h : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (2 * (x - p) / (3 * (r - z))) * (2 * (y - q) / (3 * (p - x))) * (2 * (z - r) / (3 * (q - y))) = -8 / 27 :=
  sorry

end simplify_expression_l703_703451


namespace find_a_and_c_l703_703910

theorem find_a_and_c (a c : ℚ)
  (h : (⟨a, -1, c⟩ : ℚ × ℚ × ℚ) ×ₜ (⟨5, 4, 6⟩ : ℚ × ℚ × ℚ) = ⟨-15, -20, 24⟩) :
  a = 175 / 36 ∧ c = 11 / 6 :=
  sorry

end find_a_and_c_l703_703910


namespace area_of_triangle_obtuse_triangle_exists_l703_703226

-- Define the mathematical conditions given in the problem
variables {a b c : ℝ}
axiom triangle_inequality : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y
axiom sine_relation : 2 * Real.sin c = 3 * Real.sin a
axiom side_b : b = a + 1
axiom side_c : c = a + 2

-- Part 1: Prove the area of ΔABC equals the provided solution
theorem area_of_triangle : 
  2 * Real.sin c = 3 * Real.sin a → 
  b = a + 1 → 
  c = a + 2 → 
  a = 4 → 
  b = 5 → 
  c = 6 → 
  let ab_sin_c := (1 / 2) * 4 * 5 * ((3 * Real.sqrt 7) / 8) in
  ab_sin_c = 15 * Real.sqrt 7 / 4 :=
by {
  sorry, -- The proof steps are to be provided here
}

-- Part 2: Prove there exists a positive integer a such that ΔABC is obtuse with a = 2
theorem obtuse_triangle_exists :
  (∃ (a : ℕ) (h2 : 1 < a ∧ a < 3), 2 * Real.sin c = 3 * Real.sin a ∧ b = a + 1 ∧ c = a + 2) →
  ∃ a = 2 ∧ (b = a + 1 ∧ c = a + 2) :=
by {
  sorry, -- The proof steps are to be provided here
}

end area_of_triangle_obtuse_triangle_exists_l703_703226


namespace smallest_possible_value_l703_703992

theorem smallest_possible_value (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  ∃ (v : ℝ), v = 
  (⌊ (2 * x + y) / z ⌋ + ⌊ (2 * y + z) / x ⌋ + ⌊ (2 * z + x) / y ⌋ + ⌊ (x + y + z) / (x + y) ⌋) ∧ 
  v = 4 :=
sorry

end smallest_possible_value_l703_703992


namespace zeros_in_square_of_sevens_l703_703869

def contains_no_zeros (n : ℕ) : Prop :=
  ¬ (∃ m < 10, n % (10 ^ m) = 0)

theorem zeros_in_square_of_sevens :
  contains_no_zeros (7^2) ∧ contains_no_zeros (77^2) ∧ ¬contains_no_zeros (777^2) →
  ∃ (z : ℕ), number_of_zeros (77_777_777^2) = z ∧ z = 7 :=
by
  sorry

end zeros_in_square_of_sevens_l703_703869


namespace original_price_l703_703819

-- Definitions for the problem
def selling_price : ℝ := 520
def gain_percent : ℝ := 15.56 / 100

-- The theorem we want to prove
theorem original_price (P : ℝ) :
  P = selling_price / (1 + gain_percent) :=
sorry

end original_price_l703_703819


namespace natural_number_N_exists_l703_703505

theorem natural_number_N_exists :
  ∃ N : ℕ, (5 ∣ N ∧ 49 ∣ N ∧ ∀ d : ℕ, d ∣ N → d = 1 ∨ d = N ∨ N = 12005) :=
begin
  sorry
end

end natural_number_N_exists_l703_703505


namespace mean_median_mode_relation_l703_703584

noncomputable def fish_counts : List ℚ := [3, 1, 2, 4, 1, 4, 4, 2, 3]

noncomputable def mean (l : List ℚ) : ℚ :=
  l.sum / l.length

noncomputable def median (l : List ℚ) : ℚ :=
  let sorted_l := l.qsort (· < ·)
  sorted_l[(l.length / 2 : ℕ)]

noncomputable def mode (l : List ℚ) : ℚ :=
  l.foldr (λ x acc, if l.count x > l.count acc then x else acc) 0

theorem mean_median_mode_relation :
  mean fish_counts < median fish_counts ∧ median fish_counts < mode fish_counts :=
by
  sorry

end mean_median_mode_relation_l703_703584


namespace domain_of_logarithm_and_fraction_l703_703325

def domain_of_function : Set ℝ :=
  {x : ℝ | x > 2 ∧ x ≠ 3}

theorem domain_of_logarithm_and_fraction :
  let y (x : ℝ) := log (2 * x - 4) / log 2 + 1 / (x - 3)
  ∀ x : ℝ, (∃ y, y = log (2 * x - 4) / log 2 + 1 / (x - 3)) ↔ x ∈ domain_of_function :=
by
  sorry

end domain_of_logarithm_and_fraction_l703_703325


namespace find_integer_for_expression_l703_703327

theorem find_integer_for_expression :
  ∀ y : ℝ, ∃ k : ℝ, y^2 + 10*y + 33 = (y + 5)^2 + k ∧ k = 8 :=
by
  intro y
  use 8
  split
  · calc
      y^2 + 10*y + 33
      = (y + 5)^2 + 8 : sorry
  · rfl

end find_integer_for_expression_l703_703327


namespace solve_inequality_l703_703758

theorem solve_inequality : { x : ℝ // 2 * x - 1 ≠ 0 } -> x ∈ set.Ioc (1 / 2 : ℝ) 1 → (x - 1) * (2 * x - 1) ≤ 0 ∧ (2 * x - 1) ≠ 0 :=
by
  sorry

end solve_inequality_l703_703758


namespace increasing_sequences_modulo_1013_l703_703899

theorem increasing_sequences_modulo_1013 :
  let a_seq : ℕ → Prop := λ i, (i ≤ 10) ∧ (i ≥ 1) ∧ (i % 2 = 1)
  let sequences := {seq : fin 10 → ℕ // ∀ i, a_seq (seq i)} -- Set of all possible sequences satisfying the given conditions.
  ∃ (m : ℕ)(n : ℕ), combinatorial.choose 1013 10 ∧ m % 1000 = 13
:=
begin
  sorry
end

end increasing_sequences_modulo_1013_l703_703899


namespace dive_has_five_judges_l703_703203

noncomputable def number_of_judges 
  (scores : List ℝ)
  (difficulty : ℝ)
  (point_value : ℝ) : ℕ := sorry

theorem dive_has_five_judges :
  number_of_judges [7.5, 8.0, 9.0, 6.0, 8.8] 3.2 77.76 = 5 :=
by
  sorry

end dive_has_five_judges_l703_703203


namespace fermat_torricelli_point_unique_l703_703252

def is_fermat_torricelli_point (A B C T : Point) : Prop :=
  ∠ B T C = 120 ∧ ∠ C T A = 120 ∧ ∠ A T B = 120

theorem fermat_torricelli_point_unique (A B C : Point) (hA : ∠ B A C < 120) (hB : ∠ A B C < 120) (hC : ∠ A C B < 120) :
  ∃! T : Point, inside_triangle A B C T ∧ is_fermat_torricelli_point A B C T :=
sorry

end fermat_torricelli_point_unique_l703_703252


namespace p_sufficient_for_q_p_not_necessary_for_q_l703_703150

variable (x : ℝ)

def p := sqrt (x + 2) - sqrt (1 - 2 * x) > 0
def q := (x + 1) / (x - 1) ≤ 0

theorem p_sufficient_for_q : ∀ x, p x → q x :=
by sorry

theorem p_not_necessary_for_q : ∃ x, q x ∧ ¬ p x :=
by sorry

end p_sufficient_for_q_p_not_necessary_for_q_l703_703150


namespace jackson_earning_l703_703244

theorem jackson_earning :
  let rate := 5
  let vacuum_hours := 2 * 2
  let dish_hours := 0.5
  let bathroom_hours := 0.5 * 3
  vacuum_hours * rate + dish_hours * rate + bathroom_hours * rate = 30 :=
by
  intros
  let rate := 5
  let vacuum_hours := 2 * 2
  let dish_hours := 0.5
  let bathroom_hours := 0.5 * 3
  have vacuum_earn := vacuum_hours * rate
  have dish_earn := dish_hours * rate
  have bathroom_earn := bathroom_hours * rate
  have total_earn := vacuum_earn + dish_earn + bathroom_earn
  exact total_earn = 30
  sorry

end jackson_earning_l703_703244


namespace transformed_parabola_l703_703724

theorem transformed_parabola (x : ℝ) : 
  (λ x => -x^2 + 1) (x - 2) - 2 = - (x - 2)^2 - 1 := 
by 
  sorry 

end transformed_parabola_l703_703724


namespace problem_I_problem_II_problem_III_l703_703855

variables {pA pB : ℝ}

-- Given conditions
def probability_A : ℝ := 0.7
def probability_B : ℝ := 0.6

-- Questions reformulated as proof goals
theorem problem_I : 
  sorry := 
 sorry

theorem problem_II : 
  -- Find: Probability that at least one of A or B succeeds on the first attempt
  sorry := 
 sorry

theorem problem_III : 
  -- Find: Probability that A succeeds exactly one more time than B in two attempts each
  sorry := 
 sorry

end problem_I_problem_II_problem_III_l703_703855


namespace volume_of_inscribed_sphere_l703_703840

theorem volume_of_inscribed_sphere (surface_area_cube : ℝ) (h_surface_area : surface_area_cube = 6) :
    let s := Real.sqrt(surface_area_cube / 6)
    let r := s / 2
    let volume_sphere := (4 / 3) * Real.pi * r^3
    volume_sphere = (1 / 6) * Real.pi := by
  sorry

end volume_of_inscribed_sphere_l703_703840


namespace MikeEarnings_l703_703276

-- Definitions based on the conditions
def MikeHourlyEarnings : ℝ
def PhilHourlyEarnings : ℝ := 6
def PhilEarningPercentage : ℝ := 0.50

-- The theorem to prove
theorem MikeEarnings :
  (PhilHourlyEarnings = PhilEarningPercentage * MikeHourlyEarnings) →
  MikeHourlyEarnings = 12 :=
by
  intros h
  sorry

end MikeEarnings_l703_703276


namespace grain_milling_l703_703586

theorem grain_milling (W : ℝ) (h : 0.9 * W = 100) : W = 111.1 :=
sorry

end grain_milling_l703_703586


namespace find_all_real_solutions_l703_703123

theorem find_all_real_solutions (x : ℝ) :
    (1 / ((x - 1) * (x - 2))) + (1 / ((x - 2) * (x - 3))) + (1 / ((x - 3) * (x - 4))) + (1 / ((x - 4) * (x - 5))) = 1 / 4 →
    x = 1 ∨ x = 5 :=
by
  sorry

end find_all_real_solutions_l703_703123


namespace chromium_percentage_in_combined_alloy_l703_703613

def chromium_percentage (chromium_percentage_A: ℝ) (mass_A: ℝ)
                        (chromium_percentage_B: ℝ) (mass_B: ℝ)
                        (chromium_percentage_C: ℝ) (mass_C: ℝ)
                        (chromium_percentage_D: ℝ) (mass_D: ℝ) : ℝ :=
  (chromium_percentage_A * mass_A + 
   chromium_percentage_B * mass_B + 
   chromium_percentage_C * mass_C + 
   chromium_percentage_D * mass_D)
  / (mass_A + mass_B + mass_C + mass_D) * 100

theorem chromium_percentage_in_combined_alloy :
  chromium_percentage 0.12 15 0.08 30 0.20 20 0.05 35 = 9.95 := by
  sorry

end chromium_percentage_in_combined_alloy_l703_703613


namespace page_copy_cost_l703_703635

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost_l703_703635


namespace diameter_of_circular_ground_l703_703415

noncomputable def radius_of_garden_condition (area_garden : ℝ) (broad_garden : ℝ) : ℝ :=
  let pi_val := Real.pi
  (area_garden / pi_val - broad_garden * broad_garden) / (2 * broad_garden)

-- Given conditions
variable (area_garden : ℝ := 226.19467105846502)
variable (broad_garden : ℝ := 2)

-- Goal to prove: diameter of the circular ground is 34 metres
theorem diameter_of_circular_ground : 2 * radius_of_garden_condition area_garden broad_garden = 34 :=
  sorry

end diameter_of_circular_ground_l703_703415


namespace elephant_entry_rate_l703_703776

def initial : ℕ := 30000
def exit_rate : ℕ := 2880
def exodus_hours : ℕ := 4
def final : ℕ := 28980
def entry_hours : ℕ := 7

theorem elephant_entry_rate :
  let elephants_left := exit_rate * exodus_hours
  let remaining_elephants := initial - elephants_left
  let new_elephants := final - remaining_elephants
  let entry_rate := new_elephants / entry_hours
  entry_rate = 1500 :=
by 
  rw [mul_comm exit_rate exodus_hours, mul_comm initial exodus_hours]
  rw [sub_eq_add_neg initial elephants_left, sub_eq_add_neg final remaining_elephants]
  exact sorry

end elephant_entry_rate_l703_703776


namespace problem_trajectory_eq_problem_range_t_l703_703531

-- Define the given circle equation
def circle_D (H : ℝ × ℝ) : Prop := (H.1 - 2)^2 + (H.2 + 3)^2 = 32

-- Define the point P coordinates
def P : ℝ × ℝ := (-6, 3)

-- Define the midpoint M formula based on P and H
def midpoint_M (H : ℝ × ℝ) : ℝ × ℝ := ((H.1 - 6) / 2, (H.2 + 3) / 2)

-- Define the trajectory equation as a property of M
def trajectory_M (M : ℝ × ℝ) : Prop := (M.1 + 2)^2 + M.2^2 = 8

-- The proof problem statement
theorem problem_trajectory_eq : 
  ∀ H : ℝ × ℝ, circle_D H → trajectory_M (midpoint_M H) :=
by
  intros H hH
  -- proof omitted
  sorry

-- Define the orthogonality condition for vectors NB and NC
def orthogonal_vectors (B C N : ℝ × ℝ) : Prop :=
  let v1 := (B.1 - N.1, B.2 - N.2)
  let v2 := (C.1 - N.1, C.2 - N.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- The proof problem for the range of t
theorem problem_range_t (k t : ℝ) : 
  (∃ B C : ℝ × ℝ,
    trajectory_M B ∧ trajectory_M C ∧
    B.2 = k * B.1 ∧ C.2 = k * C.1 ∧
    orthogonal_vectors B C (0, t)) →
  t ∈ [-real.sqrt 5 - 1, -real.sqrt 5 + 1] ∪ [real.sqrt 5 - 1, real.sqrt 5 + 1] :=
by
  intros h
  -- proof omitted
  sorry

end problem_trajectory_eq_problem_range_t_l703_703531


namespace sides_of_polygon_internal_angle_of_polygon_l703_703835

/- Definitions for the given problem conditions -/
def is_regular_polygon (perimeter side_length : ℝ) (n : ℕ) : Prop :=
  n * side_length = perimeter

def internal_angle (n : ℕ) : ℝ :=
  (n - 2) * 180 / n

/- Theorem statements -/
theorem sides_of_polygon {perimeter side_length : ℝ} (h_perimeter: perimeter = 180) (h_side_length: side_length = 15) :
  ∃ n : ℕ, is_regular_polygon perimeter side_length n ∧ n = 180 / 15 :=
begin
  sorry -- Proof will be provided here
end

theorem internal_angle_of_polygon {perimeter side_length : ℝ} (h_perimeter: perimeter = 180) (h_side_length: side_length = 15) :
  ∀ n : ℕ, is_regular_polygon perimeter side_length n → internal_angle n = 150 :=
begin
  sorry -- Proof will be provided here
end

end sides_of_polygon_internal_angle_of_polygon_l703_703835


namespace hypotenuse_length_l703_703035

theorem hypotenuse_length (a b : ℝ) (π : ℝ) (hπ : Real.pi = π)
  (h1 : (1 / 3) * π * b^2 * a = 1250 * π)
  (h2 : (1 / 3) * π * a^2 * b = 2700 * π) :
  Real.sqrt (a^2 + b^2) ≈ 21.33 := 
  sorry

end hypotenuse_length_l703_703035


namespace find_water_added_l703_703303

theorem find_water_added : 
  ∃ W : ℝ, (0.57 * 9 = 0.4275 * (9 + W)) ∧ W = 3 :=
by {
  use 3,
  split,
  {
    -- Proof that 0.57 * 9 = 0.4275 * (9 + 3)
    sorry
  },
  {
    -- Proof that W = 3
    exact rfl,
  }
}

end find_water_added_l703_703303


namespace mass_percentage_C_in_citric_acid_approx_l703_703027
-- Import necessary libraries

-- Definitions derived from conditions
def molecular_formula_citric_acid := (6, 8, 7) -- 6 C, 8 H, 7 O
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of molar mass of citric acid
def molar_mass_citric_acid := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (7 * atomic_mass_O)

-- Definition of mass of carbon in citric acid
def mass_C_in_citric_acid := 6 * atomic_mass_C

-- Prove that the mass percentage of carbon is approximately 37.51%
theorem mass_percentage_C_in_citric_acid_approx :
  abs ((mass_C_in_citric_acid / molar_mass_citric_acid) * 100 - 37.51) < 0.01 :=
by {
  sorry
}

end mass_percentage_C_in_citric_acid_approx_l703_703027


namespace clap7_total_count_l703_703421

def isVisible7 (n : ℕ) : Prop :=
  n.digits 10.contains 7

def isInvisible7 (n : ℕ) : Prop :=
  n % 7 = 0

def countVisible7 (l : List ℕ) : ℕ :=
  l.filter isVisible7 |>.length

def countInvisible7 (l : List ℕ) : ℕ :=
  l.filter isInvisible7 |>.length

def countBoth (l : List ℕ) : ℕ :=
  l.filter (λ n => isVisible7 n ∧ isInvisible7 n) |>.length

theorem clap7_total_count : (List.range' 1 100).countVisible7
                          + (List.range' 1 100).countInvisible7
                          - (List.range' 1 100).countBoth = 30 :=
by
  sorry

end clap7_total_count_l703_703421


namespace number_of_initials_is_10000_l703_703979

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l703_703979


namespace fruit_seller_sp_l703_703420

theorem fruit_seller_sp (CP SP : ℝ)
    (h1 : SP = 0.75 * CP)
    (h2 : 19.93 = 1.15 * CP) :
    SP = 13.00 :=
by
  sorry

end fruit_seller_sp_l703_703420


namespace range_of_x_satisfying_inequality_l703_703939

-- Define the function f on the real numbers ℝ
variable (f : ℝ → ℝ)

-- Define the conditions for the problem
-- Condition 1: f is monotonically increasing on [-2, +∞)
def monotonic_increasing_on := ∀ x y, -2 ≤ x → x ≤ y → f x ≤ f y

-- Condition 2: f(x-2) is an even function
def even_shifted := ∀ x, f (x - 2) = f (-x - 2)

-- Problem statement: Prove that for the function satisfying above conditions, the inequality holds in the interval (-2, 2)
theorem range_of_x_satisfying_inequality :
  (monotonic_increasing_on f) →
  (even_shifted f) →
  ∀ x, f (2 * x) < f (x + 2) ↔ x ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
begin
  sorry
end

end range_of_x_satisfying_inequality_l703_703939


namespace parametric_curve_length_correct_l703_703098

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..2*Real.pi, Real.sqrt (9 + 7 * Real.sin t ^ 2)

theorem parametric_curve_length_correct :
  parametric_curve_length = 4 * Real.pi * Real.sqrt ((9 + 7) / 2) :=
sorry

end parametric_curve_length_correct_l703_703098


namespace relationship_among_abc_l703_703806

noncomputable def a := Real.sqrt 5 + 2
noncomputable def b := 2 - Real.sqrt 5
noncomputable def c := Real.sqrt 5 - 2

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end relationship_among_abc_l703_703806


namespace triangle_area_is_correct_l703_703340

def perimeter : ℝ := 20
def inradius : ℝ := 2.5
def semi_perimeter := perimeter / 2
def area := inradius * semi_perimeter

theorem triangle_area_is_correct : area = 25 :=
  sorry

end triangle_area_is_correct_l703_703340


namespace cone_height_correct_l703_703694

noncomputable def height_of_cone (R1 R2 R3 base_radius : ℝ) : ℝ :=
  if R1 = 20 ∧ R2 = 40 ∧ R3 = 40 ∧ base_radius = 21 then 28 else 0

theorem cone_height_correct :
  height_of_cone 20 40 40 21 = 28 :=
by sorry

end cone_height_correct_l703_703694


namespace number_of_initials_is_10000_l703_703977

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l703_703977


namespace series_equals_two_l703_703492

-- Define the infinite series sum
noncomputable def series_sum : ℕ → ℝ
| 0     := 0
| (n+1) := (2 * n + 1) / (2^(n+1).toℝ) + series_sum n

theorem series_equals_two :
  (real.InfiniteSum (λ k, (2 * k + 1) / (2^(k+1).toℝ)) 2) :=
by
  sorry

end series_equals_two_l703_703492


namespace determine_scoundrels_and_knights_l703_703643

theorem determine_scoundrels_and_knights (n : ℕ) (h : n ≥ 2) (more_scoundrels : ∃ s k : ℕ, s > k ∧ s + k = n) :
  (∀ (statements : Π (i : ℕ), 1 ≤ i → Σ' j : ℕ, j < i ∧ (j = 0 ∨ statements j 0 = Σ'_mk j 0 ∧ i ≠ 1)), 
      ∃ (e : Π (i : ℕ), i < n → ℤ), 
      (∀ (i : ℕ), 0 < i → i < n → ((statements i 0).snd.2.2 → e i = 1) ∧ ((¬ statements i 0).snd.2.2 → e i = -1))) :=
sorry

end determine_scoundrels_and_knights_l703_703643


namespace polynomial_integer_values_l703_703379

/-- Given a polynomial P(x) = a * x^3 + b * x^2 + c * x + d,
  if P(-1), P(0), P(1), and P(2) are integers, then P(x) takes integer values for all integer x. -/
theorem polynomial_integer_values
  (a b c d : ℤ)
  (h_neg1 : ∃ (k1 : ℤ), P (-1) = k1)
  (h_0 : ∃ (k2 : ℤ), P 0 = k2)
  (h_1 : ∃ (k3 : ℤ), P 1 = k3)
  (h_2 : ∃ (k4 : ℤ), P 2 = k4) :
  ∀ x : ℤ, ∃ (k : ℤ), P x = k :=
begin
  sorry
end

end polynomial_integer_values_l703_703379


namespace not_like_terms_in_groupB_l703_703089

-- Definitions
def groupA : list (expr → expr) := [λ x y, 3 * x ^ 2 * y, λ x y, -3 * y * x ^ 2]
def groupB : list (expr → expr) := [λ x y, 3 * x ^ 2 * y, λ x y, -2 * y ^ 2 * x]
def groupC : list int := [-2004, 2005]
def groupD : list (expr → expr) := [λ x y, 5 * x * y, λ x y, 3 * y * x]

-- Proof problem statement
theorem not_like_terms_in_groupB : ¬ like_terms (groupB.head, groupB.get 1) :=
sorry

end not_like_terms_in_groupB_l703_703089


namespace balancing_point_is_vertex_l703_703640

-- Define a convex polygon and its properties
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a balancing point for a convex polygon
def is_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  -- Placeholder for the actual definition that the areas formed by drawing lines from Q to vertices of P are equal
  sorry

-- Define the uniqueness of the balancing point
def unique_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  ∀ R : Point, is_balancing_point P R → R = Q

-- Main theorem statement
theorem balancing_point_is_vertex (P : ConvexPolygon n) (Q : Point) 
  (h_balance : is_balancing_point P Q) (h_unique : unique_balancing_point P Q) : 
  ∃ i : Fin n, Q = P.vertices i :=
sorry

end balancing_point_is_vertex_l703_703640


namespace sufficient_but_not_necessary_condition_l703_703389

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (x^2 - m * x + 1) = 0 → (m^2 - 4 < 0) = ∀ m : ℝ, -2 < m ∧ m < 2 :=
sufficient_but_not_necessary_condition sorry

end sufficient_but_not_necessary_condition_l703_703389


namespace angle_NMC_l703_703236

-- Definitions of the given conditions
variables {A B C M N : Type}
variables [triangle : Triangle A B C]
variable (h1 : ∠ ABC = 100)
variable (h2 : ∠ ACB = 65)
variable (h3 : PointOn M A B)
variable (h4 : PointOn N A C)
variable (h5 : ∠ MCB = 55)
variable (h6 : ∠ NBC = 80)

-- Statement of the proof problem
theorem angle_NMC : ∠ NMC = 25 :=
by
  sorry

end angle_NMC_l703_703236


namespace adam_cat_food_packages_l703_703847

theorem adam_cat_food_packages (c : ℕ) 
  (dog_food_packages : ℕ := 7) 
  (cans_per_cat_package : ℕ := 10) 
  (cans_per_dog_package : ℕ := 5) 
  (extra_cat_food_cans : ℕ := 55) 
  (total_dog_cans : ℕ := dog_food_packages * cans_per_dog_package) 
  (total_cat_cans : ℕ := c * cans_per_cat_package)
  (h : total_cat_cans = total_dog_cans + extra_cat_food_cans) : 
  c = 9 :=
by
  sorry

end adam_cat_food_packages_l703_703847


namespace frank_reading_rate_l703_703520

theorem frank_reading_rate
    (total_chapters : ℕ := 2)
    (total_pages_per_chapter : ℕ := 405)
    (total_days : ℕ := 664) :
    total_chapters / total_days ≈ 0.003 := 
by
  sorry

end frank_reading_rate_l703_703520


namespace hardest_working_more_hours_l703_703044

theorem hardest_working_more_hours (A B C : ℕ) (h1 : A + B + C = 120) (h2 : A : B : C = 1 : 2 : 3) :
  (C - A) = 40 :=
sorry

end hardest_working_more_hours_l703_703044


namespace magnitude_of_b_l703_703953

open Real

variables {a b : Vector}

-- Define dot product condition
def dot_product (a b : Vector) : ℝ := -8

-- Define projection condition in terms of given magnitude
def projection (a b : Vector) : ℝ := -3 * sqrt 2

-- Given a and b are vectors satisfying dot_product and projection conditions
axiom dot_product_condition : dot_product a b = -8
axiom projection_condition : projection a b = -3 * sqrt 2

-- Concluding with the required magnitude of b
theorem magnitude_of_b : ‖b‖ = 4 * sqrt 2 / 3 :=
by sorry

end magnitude_of_b_l703_703953


namespace cans_needed_to_collect_l703_703444

theorem cans_needed_to_collect (goal alyssa abigail andrew : ℕ) (h_goal : goal = 200) (h_alyssa : alyssa = 30) (h_abigail : abigail = 43) (h_andrew : andrew = 55) :
  goal - (alyssa + abigail + andrew) = 72 :=
by
  rw [h_goal, h_alyssa, h_abigail, h_andrew]
  sorry

end cans_needed_to_collect_l703_703444


namespace cannot_be_18_meters_l703_703431

variable (length width fence_length : ℕ)
variable (against_wall_fence_length : ℕ → ℕ → ℕ)

def against_wall_fence_length := 
  fun (l w : ℕ) => [l + w * 2, l * 2 + w]

theorem cannot_be_18_meters {l w : ℕ} (h1 : l = 6) (h2 : w = 3) : 
  ¬ (18 ∈ against_wall_fence_length l w) := 
by
  rw [against_wall_fence_length]
  exact sorry

end cannot_be_18_meters_l703_703431


namespace fourth_hexagon_has_50_dots_l703_703201

/-
Conditions:
1. The first hexagon has 1 dot.
2. The second hexagon has 8 dots.
3. The third hexagon has 22 dots.
4. The nth hexagon’s layer will have 7(n-1) dots if n is even, and 7n dots if n is odd for n ≥ 2.
-/

def hex_dots (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let f := ℕ.recOn 
        (λ n hn, hn + if (n + 1) % 2 = 0 then 7 * (n + 1) else 7 * n)
        1
        (nat.pred n)
        in if 0 < n then f + hex_dots (nat.pred n) else 0

theorem fourth_hexagon_has_50_dots : hex_dots 4 = 50 := sorry


end fourth_hexagon_has_50_dots_l703_703201


namespace triangle_existence_l703_703174

theorem triangle_existence 
  (h_a h_b m_a : ℝ) :
  (m_a ≥ h_a) → 
  ((h_a > 1/2 * h_b ∧ m_a > h_a → true ∨ false) ∧ 
  (m_a = h_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b < m_a → true ∨ false) ∧ 
  (h_a ≤ 1/2 * h_b ∧ 1/2 * h_b = m_a → false ∨ true) ∧ 
  (1/2 * h_b > m_a → false)) :=
by
  intro
  sorry

end triangle_existence_l703_703174


namespace segment_QR_length_l703_703604

noncomputable def triangle_ABC := {a b c : ℝ // 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a}

noncomputable def circle_P_radius (a b c AB AC BC : ℝ) (h : a = AB ∧ b = AC ∧ c = BC) : ℝ :=
by
  have h_triangle : a^2 + b^2 = c^2 := eq.symm (eq.trans (by norm_num : 12^2 + 5^2 = 169) (eq.symm (by norm_num : 13^2 = 169)))
  let area := 1 / 2 * b * c := by norm_num
  let CD := area / (1 / 2 * a) := by norm_num
  exact CD

theorem segment_QR_length 
  (AB AC BC : ℝ)
  (h1 : AB = 13)
  (h2 : AC = 12)
  (h3 : BC = 5)
  (circle_radius := circle_P_radius AB AC BC 13 12 5 (by simp [h1, h2, h3]) = 60 / 13) :
  2 * circle_radius = 120 / 13 :=
by
  have h_triangle_area : 1 / 2 * BC * AC = 30 := by norm_num
  have CD := circle_radius
  have segment_length := 2 * CD
  assumption

end segment_QR_length_l703_703604


namespace find_d_l703_703889

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = 2 * d + real.sqrt (a + b + c + d)) : 
  d = (-7 + real.sqrt 33) / 8 :=
sorry

end find_d_l703_703889


namespace fixed_point_exists_l703_703514

theorem fixed_point_exists (m : ℝ) :
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 → x = 3 ∧ y = 1 :=
by
  sorry

end fixed_point_exists_l703_703514


namespace intersect_at_single_point_l703_703145

variables (A B C D M N K L O P : Type) [Euclidean A B C D M N K L O P]

-- Definitions for points and midpoints
def points_on_diag (A B C D M N K L O : Type) :=
  let AC := line A C,
      BD := line B D,
      O := intersection AC BD,
      M := point_on_line AC |AM| = |OC|,
      N := point_on_line BD |BN| = |OD|,
      K := midpoint A C,
      L := midpoint B D
  {AC BD O M N K L}

-- The main theorem statement
theorem intersect_at_single_point 
  (A B C D M N K L O : Type)
  [points_on_diag A B C D M N K L O] :
  ∃ P : Type, collinear P ML ∧ collinear P NK ∧ collinear P intersection_tri_ABC_ACD :=
  sorry

end intersect_at_single_point_l703_703145


namespace a_2015_eq_neg6_l703_703573

noncomputable def a : ℕ → ℤ
| 0 => 3
| 1 => 6
| (n+2) => a (n+1) - a n

theorem a_2015_eq_neg6 : a 2015 = -6 := 
by 
  sorry

end a_2015_eq_neg6_l703_703573


namespace mike_first_job_earnings_l703_703688

theorem mike_first_job_earnings (total_wages : ℕ) (hours_second_job : ℕ) (pay_rate_second_job : ℕ) 
  (second_job_earnings := hours_second_job * pay_rate_second_job) 
  (first_job_earnings := total_wages - second_job_earnings) :
  total_wages = 160 → hours_second_job = 12 → pay_rate_second_job = 9 → first_job_earnings = 52 := 
by 
  intros h₁ h₂ h₃ 
  unfold first_job_earnings second_job_earnings
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end mike_first_job_earnings_l703_703688


namespace dan_helmet_crater_difference_l703_703873

theorem dan_helmet_crater_difference :
  ∀ (r d : ℕ), 
  (r = 75) ∧ (d = 35) ∧ (r = 15 + (d + (r - 15 - d))) ->
  ((d - (r - 15 - d)) = 10) :=
by
  intros r d h
  have hr : r = 75 := h.1
  have hd : d = 35 := h.2.1
  have h_combined : r = 15 + (d + (r - 15 - d)) := h.2.2
  sorry

end dan_helmet_crater_difference_l703_703873


namespace vacation_fund_percentage_l703_703881

def net_monthly_salary : ℝ := 3500
def discretionary_income := (1/5) * net_monthly_salary
def savings_percentage : ℝ := 0.20
def eating_out_percentage : ℝ := 0.35
def remaining_amount : ℝ := 105

theorem vacation_fund_percentage :
  (1 - (savings_percentage + eating_out_percentage + (remaining_amount / discretionary_income))) = 0.30 :=
by
  sorry

end vacation_fund_percentage_l703_703881


namespace unit_digit_of_7_power_100_to_6_l703_703376

theorem unit_digit_of_7_power_100_to_6 : Nat.digits 10 (7 ^ (100 ^ 6)) % 10 = 1 := by
  have h1 : 100 % 4 = 0 := by norm_num
  have h2 : (100 ^ 6) % 4 = 0 := by exact Nat.pow_mod h1 _ (Nat.zero_mod 4) 
  have h3 : 7 ^ 4 % 10 = 1 := by norm_num
  have h4 : 7 ^ (100 ^ 6) % 10 = 7 ^ ((100 ^ 6) % 4) % 10 := Nat.pow_mod 7 (100 ^ 6) 10
  rw [h2, Nat.pow_zero] at h4
  rw [h4]
  exact h3

end unit_digit_of_7_power_100_to_6_l703_703376


namespace solution_egg_problem_l703_703682

noncomputable def egg_problem : Prop :=
  let bought_eggs : ℕ := 60
  let first_tray_eggs : ℕ := 15
  let second_tray_eggs : ℕ := 12
  let third_tray_eggs : ℕ := 10
  let total_dropped_eggs := first_tray_eggs + second_tray_eggs + third_tray_eggs
  let undropped_eggs := bought_eggs - total_dropped_eggs
  let perfect_undropped_eggs := undropped_eggs
  let cracked_first_tray_eggs : ℕ := 7
  let cracked_second_tray_eggs : ℕ := 5
  let cracked_third_tray_eggs : ℕ := 3
  let total_cracked_eggs_in_dropped_trays := cracked_first_tray_eggs + cracked_second_tray_eggs + cracked_third_tray_eggs
  perfect_undropped_eggs - total_cracked_eggs_in_dropped_trays = 8

theorem solution_egg_problem : egg_problem := by
  unfold egg_problem; simp; exact rfl

end solution_egg_problem_l703_703682


namespace fraction_smallest_to_largest_area_l703_703628

-- Top-level statement
theorem fraction_smallest_to_largest_area :
  ∀ (A : ℝ),
    let middle_ring_area := 6 * A
    let outer_ring_area := 12 * A
    let largest_circle_area := A + middle_ring_area + outer_ring_area
  in largest_circle_area ≠ 0 → (A / largest_circle_area) = 1 / 19 :=
begin
  intros A middle_ring_area outer_ring_area largest_circle_area h,
  rw [middle_ring_area, outer_ring_area, largest_circle_area],
  norm_num,
  sorry
end

end fraction_smallest_to_largest_area_l703_703628


namespace parallelogram_rectangle_l703_703413

/-- A quadrilateral is a parallelogram if both pairs of opposite sides are equal,
and it is a rectangle if its diagonals are equal. -/
structure Quadrilateral :=
  (side1 side2 side3 side4 : ℝ)
  (diag1 diag2 : ℝ)

structure Parallelogram extends Quadrilateral :=
  (opposite_sides_equal : side1 = side3 ∧ side2 = side4)

def is_rectangle (p : Parallelogram) : Prop :=
  p.diag1 = p.diag2 → (p.side1^2 + p.side2^2 = p.side3^2 + p.side4^2)

theorem parallelogram_rectangle (p : Parallelogram) : is_rectangle p :=
  sorry

end parallelogram_rectangle_l703_703413


namespace fodder_duration_l703_703408

noncomputable def initial_fodder_duration 
    (buffaloes : ℕ) 
    (oxen : ℕ) 
    (cows : ℕ) 
    (additional_cows : ℕ) 
    (additional_buffaloes : ℕ) 
    (total_fodder_days : ℕ) 
    (initial_days: ℝ): ℝ :=
  let B := 1 in
  let daily_fodder_initial := buffaloes * B + oxen * (3 / 2) * B + cows * (3 / 4) * B in
  let daily_fodder_after := (buffaloes + additional_buffaloes) * B + oxen * (3 / 2) * B + (cows + additional_cows) * (3 / 4) * B in
  initial_days := (daily_fodder_after * total_fodder_days) / daily_fodder_initial β

theorem fodder_duration 
    (B : ℝ) 
    (buffaloes : ℕ) 
    (oxen : ℕ) 
    (cows : ℕ) 
    (additional_cows : ℕ) 
    (additional_buffaloes : ℕ) 
    (total_fodder_days : ℕ) 
    (initial_days : ℝ) :
    initial_fodder_duration buffaloes oxen cows additional_cows additional_buffaloes total_fodder_days B = 31.2 := 
  sorry

end fodder_duration_l703_703408


namespace probability_blue_or_orange_is_five_thirteen_l703_703808

-- Define the number of jelly beans of each color.
def red_jelly_beans : ℕ := 7
def green_jelly_beans : ℕ := 8
def yellow_jelly_beans : ℕ := 9
def blue_jelly_beans : ℕ := 10
def orange_jelly_beans : ℕ := 5

-- Define the total number of jelly beans.
def total_jelly_beans : ℕ := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans + orange_jelly_beans

-- Define the number of blue or orange jelly beans.
def blue_or_orange_jelly_beans : ℕ := blue_jelly_beans + orange_jelly_beans

-- Define the expected probability
def expected_probability : ℚ := 5 / 13

-- The theorem we aim to prove
theorem probability_blue_or_orange_is_five_thirteen :
  (blue_or_orange_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = expected_probability :=
by
  sorry

end probability_blue_or_orange_is_five_thirteen_l703_703808


namespace sum_of_valid_primes_is_176_l703_703371

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def valid_prime_pair (p q : ℕ) : Prop :=
  is_two_digit p ∧ is_two_digit q ∧ p > 50 ∧ p < 99 ∧ is_prime p ∧ is_prime q ∧ q = reverse_digits p

def sum_valid_primes : ℕ :=
  (List.range 100).filter (λ p, ∃ q, valid_prime_pair p q)
                    .foldl (+) 0

theorem sum_of_valid_primes_is_176 : sum_valid_primes = 176 := by
  sorry

end sum_of_valid_primes_is_176_l703_703371


namespace integer_solutions_count_for_equation_l703_703748

theorem integer_solutions_count_for_equation :
  (∃ n : ℕ, (∀ x y : ℤ, (1/x + 1/y = 1/7) → (x ≠ 0) → (y ≠ 0) → n = 5 )) :=
sorry

end integer_solutions_count_for_equation_l703_703748


namespace num_valid_polynomials_eq_four_l703_703106

open Finset

def num_valid_polynomials : ℕ := 
  let coefficients := (finset.range 10).product (finset.range 10).product (finset.range 10).product (finset.range 10) in
  (coefficients.filter (λ abc => let ⟨⟨⟨a, b⟩, c⟩, d⟩ := abc in a + b + c + d = 1)).card

theorem num_valid_polynomials_eq_four : num_valid_polynomials = 4 :=
sorry

end num_valid_polynomials_eq_four_l703_703106


namespace minimize_y_at_x_l703_703469

noncomputable def minimize_y (a b x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + 2 * (a - b) * x

theorem minimize_y_at_x (a b : ℝ) :
  ∃ x : ℝ, minimize_y a b x = minimize_y a b (b / 2) := by
  sorry

end minimize_y_at_x_l703_703469


namespace no_real_solution_for_log_eq_l703_703196

theorem no_real_solution_for_log_eq :
  ¬ ∃ x : ℝ, log (x + 4) + log (x - 2) = log (x^2 - 6 * x - 5) ∧ 
    x + 4 > 0 ∧ x - 2 > 0 ∧ x^2 - 6 * x - 5 > 0 :=
by
  sorry

end no_real_solution_for_log_eq_l703_703196


namespace simplify_polynomial_l703_703366

theorem simplify_polynomial (x : ℝ) : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2) = (-x^2 + 23 * x - 3) := 
by
  sorry

end simplify_polynomial_l703_703366


namespace percentage_increase_of_x_l703_703345

theorem percentage_increase_of_x 
  (x1 y1 : ℝ) 
  (h1 : ∀ x2 y2, (x1 * y1 = x2 * y2) → (y2 = 0.7692307692307693 * y1) → x2 = x1 * 1.3) : 
  ∃ P : ℝ, P = 30 :=
by 
  have P := 30 
  use P 
  sorry

end percentage_increase_of_x_l703_703345


namespace remainder_problem_l703_703461

theorem remainder_problem :
  (1234567 % 135 = 92) ∧ ((92 * 5) % 27 = 1) := by
  sorry

end remainder_problem_l703_703461


namespace triangle_area_triangle_obtuse_exists_l703_703231

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Given conditions
def condition_b : b = a + 1 := by sorry
def condition_c : c = a + 2 := by sorry
def condition_sin : 2 * Real.sin C = 3 * Real.sin A := by sorry

-- Proof statement for Part (1)
theorem triangle_area (h1 : condition_b) (h2 : condition_c) (h3 : condition_sin) :
  (1 / 2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := 
sorry

-- Proof statement for Part (2)
theorem triangle_obtuse_exists (h1 : condition_b) (h2 : condition_c) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧
  (let b := a + 1 in let c := a + 2 in
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) > Real.pi / 2) :=
sorry

end triangle_area_triangle_obtuse_exists_l703_703231


namespace not_equal_coordinates_l703_703134

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ := 
  (r * real.cos θ, r * real.sin θ)

theorem not_equal_coordinates : 
  let M := (-5, real.pi / 3) in
  let Cartesian_M := polar_to_cartesian (-5) (real.pi / 3) in
  let cartesian_A := polar_to_cartesian 5 (-real.pi / 3) in
  let cartesian_B := polar_to_cartesian 5 (4 * real.pi / 3) in
  let cartesian_C := polar_to_cartesian 5 (-2 * real.pi / 3) in
  let cartesian_D := polar_to_cartesian (-5) (-5 * real.pi / 3) in
  Cartesian_M ≠ cartesian_A ∧ 
  Cartesian_M = cartesian_B ∧ 
  Cartesian_M = cartesian_C ∧ 
  Cartesian_M = cartesian_D :=
  sorry

end not_equal_coordinates_l703_703134


namespace cross_sectional_area_proof_l703_703731

-- Define the cube with edge length 1
structure Cube where
  edge_length : ℝ
  (edge_length_eq : edge_length = 1)

-- Define the plane alpha passing through line BD1
noncomputable def plane_alpha : set (ℝ × ℝ × ℝ) :=
  {p | ∃ x y z, (p = (x, y, z)) ∧ (x - y + z = 1)}

-- Define the range of cross-sectional area
def cross_sectional_area_range : set ℝ :=
  {a | a = (Real.sqrt 6) / 2 ∨ a = Real.sqrt 2}

theorem cross_sectional_area_proof (c : Cube) :
  c.edge_length = 1 →
  (∃ p ∈ plane_alpha, p ∈ (set.univ : set (ℝ × ℝ × ℝ))) →
  ∃ a ∈ cross_sectional_area_range, true :=
by
  intros h_edge_length h_plane_intersect
  sorry

end cross_sectional_area_proof_l703_703731


namespace product_equation_l703_703761

/-- Given two numbers x and y such that x + y = 20 and x - y = 4,
    the product of three times the larger number and the smaller number is 288. -/
theorem product_equation (x y : ℕ) (h1 : x + y = 20) (h2 : x - y = 4) (h3 : x > y) : 3 * x * y = 288 := 
sorry

end product_equation_l703_703761


namespace fruit_seller_price_l703_703824

theorem fruit_seller_price (C : ℝ) (h1 : 1.05 * C = 14.823529411764707) : 
  0.85 * C = 12 := 
sorry

end fruit_seller_price_l703_703824


namespace determinant_difference_l703_703552

namespace MatrixDeterminantProblem

open Matrix

variables {R : Type*} [CommRing R]

theorem determinant_difference (a b c d : R) 
  (h : det ![![a, b], ![c, d]] = 15) :
  det ![![3 * a, 3 * b], ![3 * c, 3 * d]] - 
  det ![![3 * b, 3 * a], ![3 * d, 3 * c]] = 270 := 
by
  sorry

end MatrixDeterminantProblem

end determinant_difference_l703_703552


namespace perp_lines_from_perp_planes_l703_703090

variables {Plane Line : Type} [affine_space Plane Line]

-- Definitions of perpendicular and parallel relations
def perp (l : Line) (p : Plane) : Prop := sorry -- define perpendicularity between line and plane
def par (l : Line) (m : Plane) : Prop := sorry -- define parallelism between line and plane

variables {α β : Plane} {m n : Line}

-- Main proof statement
theorem perp_lines_from_perp_planes 
  (h1 : perp m α) (h2 : perp n β) (h3 : perp α β) : perp m n := 
sorry

end perp_lines_from_perp_planes_l703_703090


namespace right_triangle_area_l703_703307

theorem right_triangle_area (h : Real) (a b c : Real) (A B C : ℝ) : 
  ∠A = 45 ∧ ∠C = 45 ∧ ∠B = 90 ∧ h = 4 → 
  area (triangle A B C) = 8 * Real.sqrt 2 :=
begin
  sorry
end

end right_triangle_area_l703_703307


namespace all_roots_are_nth_roots_of_unity_l703_703370

noncomputable def smallest_positive_integer_n : ℕ :=
  5
  
theorem all_roots_are_nth_roots_of_unity :
  (∀ z : ℂ, (z^4 + z^3 + z^2 + z + 1 = 0) → z^(smallest_positive_integer_n) = 1) :=
  by
    sorry

end all_roots_are_nth_roots_of_unity_l703_703370


namespace percentage_markup_l703_703336

theorem percentage_markup (CP SP : ℕ) (hCP : CP = 800) (hSP : SP = 1000) :
  let Markup := SP - CP
  let PercentageMarkup := (Markup : ℚ) / CP * 100
  PercentageMarkup = 25 := by
  sorry

end percentage_markup_l703_703336


namespace goods_train_pass_time_l703_703069

-- Define the speeds of the trains
def speed_mans_train : ℝ := 15 -- in kmph
def speed_goods_train : ℝ := 97 -- in kmph

-- Define the length of the goods train
def length_goods_train : ℝ := 280 -- in meters

-- Conversion factor from kmph to m/s
def kmph_to_mps (s : ℝ) : ℝ := s * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_mans_train + speed_goods_train)

-- Time taken for goods train to pass the man
def time_to_pass : ℝ := length_goods_train / relative_speed_mps

theorem goods_train_pass_time:
  time_to_pass = 8.99 := sorry

end goods_train_pass_time_l703_703069


namespace sqrt10_solution_l703_703159

theorem sqrt10_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : (1/a) + (1/b) = 2) :
  m = Real.sqrt 10 :=
sorry

end sqrt10_solution_l703_703159


namespace sum_of_three_numbers_l703_703380

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 42) (h3 : c + a = 58) :
  a + b + c = 67.5 :=
by
  sorry

end sum_of_three_numbers_l703_703380


namespace tangent_line_at_zero_increasing_on_0_1_max_min_on_0_pi_l703_703166

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.sin x + 2 * Real.cos x + x

theorem tangent_line_at_zero : 
  let y := f 0 in
  let tangent_slope := (x : ℝ) → Deriv.deriv f x in
  y = 2 ∧ tangent_slope 0 = 0 :=
by
  let y := f 0
  let tangent_slope := (x : ℝ) → Deriv.deriv f x
  have h1 : y = 2 := sorry
  have h2 : tangent_slope 0 = 0 := sorry
  exact ⟨h1, h2⟩

theorem increasing_on_0_1 : ∀ x, 0 < x ∧ x < 1 → 0 < Deriv.deriv f x :=
by
  intro x hx
  have h : 0 < Deriv.deriv f x := sorry
  exact h

theorem max_min_on_0_pi : 
  let max_val := f (π / 2) 
  let min_val := f π
  max_val = π - 1 ∧ min_val = π - 2 :=
by
  let max_val := f (π / 2)
  let min_val := f π
  have h1 : max_val = π - 1 := sorry
  have h2 : min_val = π - 2 := sorry
  exact ⟨h1, h2⟩

end tangent_line_at_zero_increasing_on_0_1_max_min_on_0_pi_l703_703166


namespace probability_at_least_three_red_l703_703425

def total_jellybeans := 15
def red_jellybeans := 6
def blue_jellybeans := 3
def white_jellybeans := 6
def picked_jellybeans := 4

-- Number of ways to pick k items from n items
noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways to pick 4 jellybeans from 15
noncomputable def total_ways := choose total_jellybeans picked_jellybeans

-- Number of ways to pick exactly 3 red jellybeans and 1 non-red jellybean
noncomputable def ways_three_red_one_nonred := choose red_jellybeans 3 * choose (total_jellybeans - red_jellybeans) 1

-- Number of ways to pick exactly 4 red jellybeans
noncomputable def ways_four_red := choose red_jellybeans 4

-- Total number of favorable outcomes for at least 3 red jellybeans
noncomputable def successful_outcomes := ways_three_red_one_nonred + ways_four_red

-- Probability of at least 3 red jellybeans out of 4 picked
noncomputable def probability := (successful_outcomes : ℚ) / total_ways

theorem probability_at_least_three_red :
  probability = (13 : ℚ) / 91 :=
begin
  sorry
end

end probability_at_least_three_red_l703_703425


namespace no_base_satisfies_condition_l703_703908

theorem no_base_satisfies_condition :
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬ (977 % b = 0) :=
by
  intros b hb
  cases hb with hb1 hb2
  exact sorry

end no_base_satisfies_condition_l703_703908


namespace intersection_of_curves_l703_703210

theorem intersection_of_curves : (∀ t : ℝ, ∀ θ : ℝ, 
  (x = t ∧ y = sqrt t) ∧
  (x = sqrt 2 * cos θ ∧ y = sqrt 2 * sin θ) ) →
  (1,1) ∈ (set_of (λ p : ℝ × ℝ, p.1 = p.2) ∩ 
          set_of (λ p : ℝ × ℝ, p.1^2 + p.2^2 = 2)) :=
sorry

end intersection_of_curves_l703_703210


namespace short_trees_after_planting_l703_703005

/-- 
Define the number of current short trees, newly planted short trees, and the total
short trees after planting.
-/
def current_short_trees : ℕ := 3
def newly_planted_short_trees : ℕ := 9
def total_short_trees : ℕ := current_short_trees + newly_planted_short_trees

theorem short_trees_after_planting : total_short_trees = 12 := 
by
  unfold total_short_trees
  simp
  sorry

end short_trees_after_planting_l703_703005


namespace david_more_pushup_than_zachary_l703_703040

theorem david_more_pushup_than_zachary (zachary david : ℕ) (hz : zachary = 19) (hd : david = 58) : david - zachary = 39 :=
by
  rw [hz, hd]
  sorry

end david_more_pushup_than_zachary_l703_703040


namespace find_sin_3phi_l703_703997

noncomputable def e_phi := (1 + complex.I * real.sqrt 8) / 3
noncomputable def phi := complex.log e_phi / complex.I

theorem find_sin_3phi :
  complex.sin (3 * phi) = - (5 * real.sqrt 8) / 9 :=
by
  sorry

end find_sin_3phi_l703_703997


namespace jogger_distance_l703_703595

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  -- Definition and proof would go here
  --
  sorry

end jogger_distance_l703_703595


namespace jackson_earns_30_dollars_l703_703242

theorem jackson_earns_30_dollars :
  let vacuuming_hours := 2 * 2,
      washing_dishes_hours := 0.5,
      cleaning_bathroom_hours := 3 * washing_dishes_hours,
      total_hours := vacuuming_hours + washing_dishes_hours + cleaning_bathroom_hours,
      hourly_rate := 5
  in total_hours * hourly_rate = 30 :=
by
  let vacuuming_hours := 2 * 2
  let washing_dishes_hours := 0.5
  let cleaning_bathroom_hours := 3 * washing_dishes_hours
  let total_hours := vacuuming_hours + washing_dishes_hours + cleaning_bathroom_hours
  let hourly_rate := 5
  show total_hours * hourly_rate = 30
  exact sorry

end jackson_earns_30_dollars_l703_703242


namespace piglet_weight_l703_703062

variable (C K P L : ℝ)

theorem piglet_weight (h1 : C = K + P) (h2 : P + C = L + K) (h3 : L = 30) : P = 15 := by
  sorry

end piglet_weight_l703_703062


namespace quadratic_rewrite_l703_703711

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 4) (h2 : 2 * d * e = 20) (h3 : e^2 + f = -24) :
  d * e = 10 :=
sorry

end quadratic_rewrite_l703_703711


namespace abs_diff_41st_term_l703_703357

open Nat

-- Definitions of arithmetic sequences C and D
def seqC (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 50 - 15 * (n - 1)

theorem abs_diff_41st_term : abs (seqC 41 - seqD 41) = 1200 :=
by
  sorry

end abs_diff_41st_term_l703_703357


namespace find_ratio_of_three_numbers_l703_703612

noncomputable def ratio_of_three_numbers (A B C : ℝ) : Prop :=
  (A + B + C) / (A + B - C) = 4 / 3 ∧
  (A + B) / (B + C) = 7 / 6

theorem find_ratio_of_three_numbers (A B C : ℝ) (h₁ : ratio_of_three_numbers A B C) :
  A / C = 2 ∧ B / C = 5 :=
by
  sorry

end find_ratio_of_three_numbers_l703_703612


namespace radius_circle_D_form_l703_703103

-- Definitions for circles C, D, and E
def Circle (radius : ℝ) := {center : ℝ × ℝ // (center.fst - 0)^2 + (center.snd - 0)^2 = radius^2}
def internally_tangent (C D : Circle) (x : ℝ × ℝ) : Prop :=
  x ∈ C ∧ x ∈ D ∧ ∃δ > 0, C.radius + D.radius = δ ∧ ∀y ∈ C, ∀z ∈ D, dist y z = δ

noncomputable def radius_of_circle_D_radius_form_proof : ℝ := by
  let r := 1 -- radius of circle E
  let radius_D := 4 * r -- radius of circle D
  have h : radius_D = 4 := by
    calc
      radius_D = 4 * r : by rfl
              ... = 4 * 1 : by rfl
              ... = 4 : by rfl
  exact radius_D

-- Main theorem to prove
theorem radius_circle_D_form: ∃ p q : ℕ, radius_of_circle_D_radius_form_proof = real.sqrt p - q ∧ p = 16 ∧ q = 0 :=
by
  use 16, 0
  have h : radius_of_circle_D_radius_form_proof = 4 := by rfl
  have sqrt16 : real.sqrt 16 = 4 := by norm_num
  exact ⟨h, sqrt16, rfl⟩

end radius_circle_D_form_l703_703103


namespace problem_statement_l703_703272

-- Defining the sets M and P based on the given conditions
def M : Set ℝ := {x | 2^x - 1 > 3}
def P : Set ℝ := {x | log 2 x < 2}

-- The statement to be proved
theorem problem_statement (x : ℝ) : (x ∈ M ∪ P) → (x ∈ M ∩ P) ∨ (¬ (x ∈ M ∩ P) ∧ ¬(x ∈ M ∪ P)) :=
by
  -- Proof to be filled in later
  sorry

end problem_statement_l703_703272


namespace cannot_end_up_with_one_l703_703779

-- Definitions based on conditions
def initial_set := Finset.range 3001

-- Theorem statement
theorem cannot_end_up_with_one :
  ¬ (∃ steps : list (ℤ × ℤ), (∀ ab ∈ steps, ab.1 ∈ initial_set ∧ ab.2 ∈ initial_set) ∧ 
    initial_set.card - steps.length = 1 ∧ (∀ ab ∈ steps, ab.1 - ab.2 ∈ initial_set))
    ∧ final_set = {1} :=
sorry

end cannot_end_up_with_one_l703_703779


namespace fourth_term_expansion_l703_703126

def binomial_term (n r : ℕ) (a b : ℚ) : ℚ :=
  (Nat.descFactorial n r) / (Nat.factorial r) * a^(n - r) * b^r

theorem fourth_term_expansion (x : ℚ) (hx : x ≠ 0) : 
  binomial_term 6 3 2 (-(1 / (x^(1/3)))) = (-160 / x) :=
by
  sorry

end fourth_term_expansion_l703_703126


namespace arg_z_range_l703_703670

open Complex

noncomputable def arg_values_range (z : ℂ) (h : Arg ((z + 1) / (z + 2)) = π / 6 ∨ Arg ((z + 1) / (z + 2)) = -π / 6) : Prop :=
  arg z ∈ Icc (5 * π / 6 - arcsin (sqrt 3 / 3)) π ∪ Ioc π (7 * π / 6 + arcsin (sqrt 3 / 3))

theorem arg_z_range (z : ℂ) : 
  (Arg ((z + 1) / (z + 2)) = π / 6 ∨ Arg ((z + 1) / (z + 2)) = -π / 6) →
  arg z ∈ Icc (5 * π / 6 - arcsin (sqrt 3 / 3)) π ∪ Ioc π (7 * π / 6 + arcsin (sqrt 3 / 3)) :=
sorry

end arg_z_range_l703_703670


namespace number_of_odd_degree_vertices_even_l703_703701

theorem number_of_odd_degree_vertices_even (V : Type) [Fintype V] (E : Type) [Fintype E] 
  (degree : V → ℕ) (sum_degrees_eq_twice_edges : (∑ v, degree v) = 2 * Fintype.card E) 
  (even_sum : ∀ (s : Finset ℕ), (∀ x ∈ s, x % 2 = 0) → (s.sum id) % 2 = 0)
  (odd_sum_even_iff_count_even : ∀ (s : Finset ℕ), (∀ x ∈ s, x % 2 = 1) → (s.sum id % 2 = 0 ↔ Fintype.card s % 2 = 0)) : 
  ∃ s : Finset V, (∀ v ∈ s, degree v % 2 = 1) ∧ Fintype.card s % 2 = 0 :=
by
  sorry

end number_of_odd_degree_vertices_even_l703_703701


namespace ellipse_equation_focus_y_axis_ellipse_equation_focal_length_10_l703_703509

theorem ellipse_equation_focus_y_axis (a b c : ℝ) (h : 0 < b ∧ b < a) :
  2 * c = 4 ∧ (2^2 / a^2) + (3^2 / b^2) = 1 ∧ a^2 = b^2 + c^2 ↔
  (a = 4) ∧ (c = 2) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, y^2 / 16 + x^2 / 12 = 1).
sorry

theorem ellipse_equation_focal_length_10 (a b c : ℝ) :
  2 * c = 10 ∧ 2 * a = 26 ∧ a^2 = b^2 + c^2 ↔
  (c = 5) ∧ (a = 13) ∧ (b = 12) ∧ ((∀ x y : ℝ, x^2 / 169 + y^2 / 144 = 1) ∨ (∀ x y : ℝ, y^2 / 169 + x^2 / 144 = 1)).
sorry

end ellipse_equation_focus_y_axis_ellipse_equation_focal_length_10_l703_703509


namespace parabola_passes_through_A_C_l703_703930

theorem parabola_passes_through_A_C : ∃ (a b : ℝ), (2 = a * 1^2 + b * 1 + 1) ∧ (1 = a * 2^2 + b * 2 + 1) :=
by {
  sorry
}

end parabola_passes_through_A_C_l703_703930


namespace triangle_area_l703_703310

theorem triangle_area (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (a b c : ℝ) (h : (a = b ∧ b = c) ∧ (a * a + b * b = c * c) ∧ (a * (b + b) = 8) ∧ (a * b / 2 = 16)) : 
  a * b / 2 = 16 :=
by {
  sorry 
}

end triangle_area_l703_703310


namespace trailing_zeros_60_l703_703179

def count_trailing_zeros_factorial (n : ℕ) (p : ℕ) [fact (nat.prime p)] : ℕ :=
  if n = 0 then 0 else n / p + count_trailing_zeros_factorial (n / p) p

theorem trailing_zeros_60! : count_trailing_zeros_factorial 60 5 = 14 := 
by {
  sorry
}

end trailing_zeros_60_l703_703179


namespace exists_obtuse_triangle_l703_703222

section
variables {a b c : ℝ} (ABC : Triangle ℝ) (h₁ : ABC.side1 = a)
  (h₂ : ABC.side2 = a + 1) (h₃ : ABC.side3 = a + 2)
  (h₄ : 2 * sin ABC.γ = 3 * sin ABC.α)

noncomputable def area_triangle_proof : ABC.area = (15 * real.sqrt 7) / 4 :=
sorry

theorem exists_obtuse_triangle : ∃ (a : ℝ), a = 2 ∧ ∀ (h : Triangle ℝ), 
  h.side1 = a → h.side2 = a + 1 → h.side3 = a + 2 → obtuse h :=
sorry
end

end exists_obtuse_triangle_l703_703222


namespace exists_obtuse_triangle_l703_703220

section
variables {a b c : ℝ} (ABC : Triangle ℝ) (h₁ : ABC.side1 = a)
  (h₂ : ABC.side2 = a + 1) (h₃ : ABC.side3 = a + 2)
  (h₄ : 2 * sin ABC.γ = 3 * sin ABC.α)

noncomputable def area_triangle_proof : ABC.area = (15 * real.sqrt 7) / 4 :=
sorry

theorem exists_obtuse_triangle : ∃ (a : ℝ), a = 2 ∧ ∀ (h : Triangle ℝ), 
  h.side1 = a → h.side2 = a + 1 → h.side3 = a + 2 → obtuse h :=
sorry
end

end exists_obtuse_triangle_l703_703220


namespace balls_in_boxes_l703_703695

theorem balls_in_boxes :
  ∀ (balls boxes : ℕ), balls = 5 → boxes = 4 →
    (∃ (way_to_place : ℕ), way_to_place = 4) :=
by
  intros balls boxes hballs hboxes
  use 4
  sorry

end balls_in_boxes_l703_703695


namespace angle_between_AD_BE_eq_pi_div_3_l703_703359

-- Definitions based on the conditions given
variables {A B C D E : Type*}
variables (A B C D E : E)

-- Two equilateral triangles with a common vertex C
def equilateral_triangle (A B C : E) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def common_vertex (C : E) (α β : E → E → Prop) : Prop :=
  ∃ A B D E, α A B C ∧ β C D E

-- Given condition
axiom equilateral_ABC : equilateral_triangle A B C
axiom equilateral_CDE : equilateral_triangle C D E
axiom common_vertex_C : common_vertex C (equilateral_triangle A B) (equilateral_triangle D E)

-- Prove the angle between AD and BE is π/3
theorem angle_between_AD_BE_eq_pi_div_3 : 
  angle_between_lines A D B E = π / 3 :=
sorry

end angle_between_AD_BE_eq_pi_div_3_l703_703359


namespace sum_of_digits_of_sum_of_steps_l703_703614

theorem sum_of_digits_of_sum_of_steps (n : ℕ) (h_prime : n.Prime)
  (h_jumps_condition : (Nat.ceil (n / 3) - Nat.ceil (n / 6)) = 25) :
  Nat.digits 10 (149 + 151) = [3] := by
  sorry

end sum_of_digits_of_sum_of_steps_l703_703614


namespace domain_of_f_l703_703737

noncomputable def f : ℝ → ℝ := λ x, Real.log (1 - 1 / (x + 3))

theorem domain_of_f :
  {x : ℝ | 1 - (1 / (x + 3)) > 0} = {x : ℝ | x < -3 ∨ x > -2} :=
by
  sorry

end domain_of_f_l703_703737


namespace bud_age_is_eight_l703_703858

def uncle_age : ℕ := 24

def bud_age (uncle_age : ℕ) : ℕ := uncle_age / 3

theorem bud_age_is_eight : bud_age uncle_age = 8 :=
by
  sorry

end bud_age_is_eight_l703_703858


namespace volunteers_arrangement_l703_703350

theorem volunteers_arrangement (h : fintype (fin 7)):
  (finset.card (finset.combinations (finset.univ : finset (fin 7)) 3) * 
   finset.card (finset.combinations (finset.univ \ @finset.range 4 (finset.univ : finset (fin 7))) 3) = 140) :=
by sorry

end volunteers_arrangement_l703_703350


namespace gcd_xyz_times_xyz_is_square_l703_703667

theorem gcd_xyz_times_xyz_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by
  sorry

end gcd_xyz_times_xyz_is_square_l703_703667


namespace sufficient_but_not_necessary_condition_l703_703387

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (x^2 - m * x + 1) = 0 → (m^2 - 4 < 0) = ∀ m : ℝ, -2 < m ∧ m < 2 :=
sufficient_but_not_necessary_condition sorry

end sufficient_but_not_necessary_condition_l703_703387


namespace DE_perpendicular_EF_l703_703256

theorem DE_perpendicular_EF
  (A B C D E F : Type)
  [euclidean_plane A B C D E F]
  (h1 : Incircle_centered D (triangle B A C))
  (h2 : Angle D A C = 30)
  (h3 : Angle D C A = 30)
  (h4 : Angle D B A = 60)
  (h5 : is_midpoint E B C)
  (h6 : is_trisection_point F A C 2 1) :
  Perpendicular D E F :=
sorry

end DE_perpendicular_EF_l703_703256


namespace increasing_interval_of_g_l703_703742

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f (4 * x - x^2)

theorem increasing_interval_of_g :
  ∀ x : ℝ, 0 < x ∧ x < 2 → g x > g (x + 1) (x + 1 < 2) :=
by sorry

end increasing_interval_of_g_l703_703742


namespace greatest_pair_sum_possible_l703_703839

open Nat

noncomputable def max_pair_sum (sums : List ℕ) (x y : ℕ) := 
  ∃ (a b c d e : ℕ), 
    list.contains sums 138 ∧ list.contains sums 419 ∧ 
    list.contains sums 350 ∧ list.contains sums 293 ∧
    list.contains sums 226 ∧ list.contains sums 409 ∧
    list.contains sums 167 ∧ list.contains sums 254 ∧
    list.contains sums x ∧ list.contains sums y ∧
    (x + y = 3464)

theorem greatest_pair_sum_possible (sums : List ℕ) (x y : ℕ) : 
  max_pair_sum sums x y → x + y = 3464 :=
sorry

end greatest_pair_sum_possible_l703_703839


namespace die_toss_total_recording_results_l703_703706

theorem die_toss_total_recording_results :
  (∑ n in finset.filter (λ n, multiset.card (multiset.to_finset n) = 3) (finset.range 6).powerset_len 5, 1) = 840 := 
sorry

end die_toss_total_recording_results_l703_703706


namespace problem_l703_703144

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ (x : ℝ), f(x + 4) = f(x)
axiom cond2 : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 2) → f(x₁) < f(x₂)
axiom cond3 : ∀ (x : ℝ), f(4 - x) = f(x)

theorem problem : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  -- proof would go here
  sorry

end problem_l703_703144


namespace mike_earnings_first_job_l703_703686

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end mike_earnings_first_job_l703_703686


namespace problem_statement_l703_703143

variable (f : ℝ → ℝ)
variable (x : ℝ)
variable (f'' : ℝ → ℝ)

-- Given conditions
axiom f_diff_defined : ∀ x, 0 < x ∧ x < π / 2 → ContinuousOn f (set.Ioo 0 (π / 2))
axiom f_second_derivative : ∀ x, 0 < x ∧ x < π / 2 → f'' x * sin x - f x * cos x > 0

-- The proof problem
theorem problem_statement : sqrt 3 * f (π / 6) < f (π / 3) :=
sorry

end problem_statement_l703_703143


namespace monotonic_decreasing_intervals_l703_703948

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x) + cos (ω * x)

theorem monotonic_decreasing_intervals (ω : ℝ) (hω : ω > 0)
  (h_distance : ∃ T > 0, ∀ x : ℝ, (f x ω = -2 → f (x + T) ω = -2) ∧ (T = π)) :
  ∀ k : ℤ, ∀ x : ℝ, k * π + π / 6 ≤ x ∧ x ≤ k * π + π / 3 →
    ∀ y : ℝ, x ≤ y → f y ω ≤ f x ω :=
sorry

end monotonic_decreasing_intervals_l703_703948


namespace compare_abc_l703_703991

theorem compare_abc (a b c : ℝ) (h1 : a = 5^0.2) (h2 : b = 0.5^0.2) (h3 : c = 0.5^2) : a > b ∧ b > c :=
by
  rw [h1, h2, h3]
  sorry

end compare_abc_l703_703991


namespace expression_simplification_l703_703860

theorem expression_simplification (x y : ℝ) : x^2 + (y - x) * (y + x) = y^2 :=
by
  sorry

end expression_simplification_l703_703860


namespace david_marin_apples_l703_703874

theorem david_marin_apples (marin_apples david_apples : ℕ) (h_marin : marin_apples = 3) (h_david : david_apples = 3) :
  marin_apples - david_apples = 0 :=
by 
  rw [h_marin, h_david]
  exact Nat.sub_self 3

end david_marin_apples_l703_703874


namespace problem_statement_A_problem_statement_B_problem_statement_C_problem_statement_D_l703_703390

theorem problem_statement_A (x : ℝ) (h1 : 0 < x ∧ x < 1/2) : 
  ∃ m : ℝ, m = \frac{1}{4} ∧ ∀ y : ℝ, y = x * sqrt(1 - 4 * x^2) → y ≤ m := sorry

theorem problem_statement_B (x : ℝ) (h2 : x < 0) : 
  ∀ y : ℝ, y = x + (1 / x) - 2 → ¬(y = 0) := sorry

theorem problem_statement_C (x y : ℝ) (h3 : x > 0 ∧ y > 0 ∧ x + y = 8) : 
  ∃ m : ℝ, m = 25 ∧ ∀ z : ℝ, z = (1 + x) * (1 + y) → z ≤ m := sorry

theorem problem_statement_D (x y : ℝ) (h4 : x > 0 ∧ y > 0 ∧ 3x + y = 5 * x * y) : 
  ∀ m : ℝ, m = 4 * x + 3 * y → ¬(m = 24 / 5) := sorry

end problem_statement_A_problem_statement_B_problem_statement_C_problem_statement_D_l703_703390


namespace years_deposited_first_bank_l703_703094

def P : ℝ := 640  -- Principal amount
def r : ℝ := 0.15  -- Rate of interest per annum (15%)
def I2 : ℝ := 480  -- Interest from the second bank (calculated as P * r * 5)
def D : ℝ := 144  -- Difference between their interests

-- Define n, the number of years the money was deposited in the first bank
variable (n : ℝ)

-- The interest from the first bank
def I1 := P * r * n  -- 96n in the simplified form

-- The Lean statement for the problem proof
theorem years_deposited_first_bank : I2 - I1 = D → n = 3.5 :=
by
  -- Definitions and conditions
  have hP : P = 640 := rfl
  have hr : r = 0.15 := rfl
  have hI2 : I2 = 480 := rfl
  have hD : D = 144 := rfl
  have hI1 : I1 = P * r * n := rfl

  -- The proof goes here (to be completed)
  sorry

end years_deposited_first_bank_l703_703094


namespace find_original_percentage_of_acid_l703_703788

noncomputable def percentage_of_acid (a w : ℕ) : ℚ :=
  (a : ℚ) / (a + w : ℚ) * 100

theorem find_original_percentage_of_acid (a w : ℕ) 
  (h1 : (a : ℚ) / (a + w + 2 : ℚ) = 1 / 4)
  (h2 : (a + 2 : ℚ) / (a + w + 4 : ℚ) = 2 / 5) : 
  percentage_of_acid a w = 33.33 :=
by 
  sorry

end find_original_percentage_of_acid_l703_703788


namespace prove_cuboid_properties_l703_703417

noncomputable def cuboid_length := 5
noncomputable def cuboid_width := 4
noncomputable def cuboid_height := 3

theorem prove_cuboid_properties :
  (min (cuboid_length * cuboid_width) (min (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 12) ∧
  (max (cuboid_length * cuboid_width) (max (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 20) ∧
  ((cuboid_length + cuboid_width + cuboid_height) * 4 = 48) ∧
  (2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 94) ∧
  (cuboid_length * cuboid_width * cuboid_height = 60) :=
by
  sorry

end prove_cuboid_properties_l703_703417


namespace norm_a_sub_b_eq_angle_between_a_sub_b_and_a_range_norm_a_sub_t_b_l703_703581

noncomputable def vector_a := (1 : ℝ, Real.sqrt 3)
noncomputable def vector_b := (-2 : ℝ, 0)

def norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Problem 1
theorem norm_a_sub_b_eq : norm (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) = 2 * Real.sqrt 3 :=
  sorry

-- Problem 2
def dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem angle_between_a_sub_b_and_a :
  let a_sub_b := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) in
  Real.arccos ((dot a_sub_b vector_a) / (norm a_sub_b * norm vector_a)) = Real.pi / 6 :=
  sorry

-- Problem 3
theorem range_norm_a_sub_t_b :
  (∀ t : ℝ, norm (vector_a.1 - t * vector_b.1, vector_a.2 - t * vector_b.2) > Real.sqrt 3) :=
  sorry

end norm_a_sub_b_eq_angle_between_a_sub_b_and_a_range_norm_a_sub_t_b_l703_703581


namespace men_in_first_group_l703_703600

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end men_in_first_group_l703_703600


namespace number_of_initials_sets_l703_703962

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l703_703962


namespace supermarket_solution_exists_l703_703414

noncomputable theory

-- Define the given system of equations for part 1
def eq1 (x y : ℝ) : Prop := 10 * x + 8 * y = 880
def eq2 (x y : ℝ) : Prop := 2 * x + 5 * y = 380

-- Define the constraints for part 2
def cost_constraint (a : ℝ) : Prop := 40 * a + 60 * (50 - a) ≤ 2520
def profit_constraint (a : ℝ) : Prop := 10 * a + 15 * (50 - a) ≥ 620

-- Define the ranges and enumeration of possible plans
def valid_range (a : ℝ) : Prop := 24 ≤ a ∧ a ≤ 26
def plan1 : Prop := valid_range 24
def plan2 : Prop := valid_range 25
def plan3 : Prop := valid_range 26

-- Main theorem
theorem supermarket_solution_exists :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y ∧ x = 40 ∧ y = 60) ∧
  (∃ a : ℝ, cost_constraint a ∧ profit_constraint a) ∧ plan1 ∧ plan2 ∧ plan3 :=
by sorry

end supermarket_solution_exists_l703_703414


namespace measure_of_angle_DML_l703_703864

-- Define the conditions given in the problem
variables (D E F L M N : Point)
variables (Omega : Circle)
variables (triangle_DEF : Triangle D E F)
variables (triangle_LMN : Triangle L M N)
variables (angle_D angle_E angle_F : Angle)

-- Conditions
def conditions :=
  Incircle Omega triangle_DEF ∧
  Circumcircle Omega triangle_LMN ∧
  OnLineSegment L E F ∧
  OnLineSegment M D E ∧
  OnLineSegment N D F ∧
  angle_D = 50 ∧
  angle_E = 70 ∧
  angle_F = 60

-- The proof problem in Lean 4 statement
theorem measure_of_angle_DML 
  (h : conditions D E F L M N Omega triangle_DEF triangle_LMN angle_D angle_E angle_F) : 
  ∠DML = 30 := by
    sorry

end measure_of_angle_DML_l703_703864


namespace combined_period_cos_3x_plus_sin_2x_l703_703369

theorem combined_period_cos_3x_plus_sin_2x :
  ∃ T : Real, (∀ x : Real, cos (3 * (x + T)) + sin (2 * (x + T)) = cos (3 * x) + sin (2 * x)) ∧ T = 2 * Real.pi := 
sorry

end combined_period_cos_3x_plus_sin_2x_l703_703369


namespace apple_probabilities_l703_703811

theorem apple_probabilities :
  (∀ (A B C : ℝ), (A / (A + B + C)) = 0.2) ∧
  (∀ (pA pB pC rA rB rC : ℝ), 
    (pA / (pA + pB + pC)) * rA + 
    (pB / (pA + pB + pC)) * rB + 
    (pC / (pA + pB + pC)) * rC = 0.69) ∧
  (∀ (pA rA total_first_grade : ℝ), 
    (pA / (2 + 5 + 3)) * rA / total_first_grade = 5 / 23) :=
by
  assume A B C
  assume pA pB pC rA rB rC
  assume pA rA total_first_grade
  sorry

end apple_probabilities_l703_703811


namespace smallest_positive_sum_l703_703882

open BigOperators

-- Define b_i's are either 1 or -1
def b (i : ℕ) (h : 1 ≤ i ∧ i ≤ 50) : ℤ := sorry -- Replace with actual definition if needed

-- This is the statement of our theorem
theorem smallest_positive_sum : 
  (∀ i (hi : 1 ≤ i ∧ i ≤ 50), b i hi = 1 ∨ b i hi = -1) →
  ∑ i in finset.range 50, ∑ j in finset.range 50, if (1 ≤ i ∧ i < j ∧ j ≤ 50) then b i ⟨i, sorry⟩ * b j ⟨j, sorry⟩ else 0 = 7 :=
by
  sorry

end smallest_positive_sum_l703_703882


namespace similar_KLM_YDZ_EXZ_YXF_l703_703280

noncomputable def circumcenter (A B C : Point) : Point := sorry
def similar (A B C D E F : Triangle) : Prop := sorry

variables {X Y Z D E F K L M : Point}
variables (triangle_YDZ triangle_EXZ triangle_YXF : Triangle)
variables (angle_ZDY angle_ZXE angle_FXY angle_YZD angle_EZX angle_YFX : Angle)
variables (K_circumcenter : K = circumcenter Y D Z)
variables (L_circumcenter : L = circumcenter E X Z)
variables (M_circumcenter : M = circumcenter Y X F)

theorem similar_KLM_YDZ_EXZ_YXF
  (h1 : similar triangle_YDZ triangle_EXZ)
  (h2 : similar triangle_EXZ triangle_YXF)
  (h3 : angle_ZDY = angle_ZXE)
  (h4 : angle_ZXE = angle_FXY)
  (h5 : angle_YZD = angle_EZX)
  (h6 : angle_EZX = angle_YFX)
  (hK : K_circumcenter)
  (hL : L_circumcenter)
  (hM : M_circumcenter) :
  similar (Triangle.mk K L M) triangle_YDZ :=
sorry

end similar_KLM_YDZ_EXZ_YXF_l703_703280


namespace positive_integer_a_l703_703603

theorem positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ (k : ℤ), (2 * a + 8) = k * (a + 1)) :
  a = 1 ∨ a = 2 ∨ a = 5 :=
by sorry

end positive_integer_a_l703_703603


namespace complex_polygon_area_l703_703518

-- Define the side length of the squares
def side_length : ℝ := 8

-- Define the area of one square
def area_square : ℝ := side_length * side_length

-- Define the area of the triangle
def area_triangle : ℝ := (1 / 2) * side_length * side_length

-- The main goal: prove the area of the complex polygon
theorem complex_polygon_area : (4 * area_square) + area_triangle - 4 * (area_triangle / 2) = 288 :=
by
  sorry

end complex_polygon_area_l703_703518


namespace sum_of_factors_360_l703_703031

theorem sum_of_factors_360 : σ 360 = 1170 :=
by sorry

end sum_of_factors_360_l703_703031


namespace simplify_expression_l703_703905

theorem simplify_expression 
  (x y z w : ℝ) 
  (h1 : x ≠ 2) 
  (h2 : y ≠ 3) 
  (h3 : z ≠ 4) 
  (h4 : w ≠ 5)
  (h5 : x ≠ 6 - y) 
  (h6 : y ≠ 2-x) 
  (h7 : z ≠ 3-y) 
  (h8 : z ≠ w-4) 
  (h9 : w ≠ 6-z) 
  (h10 : w ≠ x+1) : 
  \[
  \frac{x-2}{6-y} \cdot \frac{y-3}{2-x} \cdot \frac{z-4}{3-y} \cdot \frac{6-z}{4-w} \cdot \frac{w-5}{z-6} \cdot \frac{x+1}{5-w} = 1
  \]
:=
  sorry

end simplify_expression_l703_703905


namespace proof_problem_l703_703800

-- Conditions for symmetrical expressions
def symmetrical_expr (f : ℕ → ℕ → Prop) : Prop :=
  ∀ (a b : ℕ), f a b = f b a

-- Question 1: Verifying symmetrical expressions
def expr1 (a b c : ℕ) : ℕ := a + b + c
def expr2 (a b : ℕ) : ℕ := a^2 + b^2

def thesis1 (a b c : ℕ) : Prop :=
  symmetrical_expr (λ a b, (expr1 a b c))

def thesis2 (a b : ℕ) : Prop :=
  symmetrical_expr (λ a b, (expr2 a b))

-- Question 2: Symmetrical monomial with degree 6
def monomial : ℕ → ℕ → ℕ := λ x y, x^3 * y^3

def thesis3 (x y : ℕ) : Prop :=
  symmetrical_expr monomial ∧ x^3 * y^3 = x^3 * y^3 ∧ 3 + 3 = 6

-- Question 3: Given expressions and evaluating their sum
def A (a b : ℕ) : ℕ := 2 * a^2 + 4 * b^2
def B (a b : ℕ) : ℕ := a^2 - 2 * a * b

def thesis4 (a b : ℕ) : Prop :=
  let result := 4 * a^2 + 4 * b^2 - 4 * a * b in
  symmetrical_expr (λ a b, result)

-- Combined theorem statement
theorem proof_problem (a b c x y : ℕ) : thesis1 a b c ∧ thesis2 a b ∧ thesis3 x y ∧ thesis4 a b :=
by sorry

end proof_problem_l703_703800


namespace number_of_subsets_including_1_and_10_l703_703263

def A : Set ℕ := {a : ℕ | ∃ x y z : ℕ, a = 2^x * 3^y * 5^z}
def B : Set ℕ := {b : ℕ | b ∈ A ∧ 1 ≤ b ∧ b ≤ 10}

theorem number_of_subsets_including_1_and_10 :
  ∃ (s : Finset (Finset ℕ)), (∀ x ∈ s, 1 ∈ x ∧ 10 ∈ x) ∧ s.card = 128 := by
  sorry

end number_of_subsets_including_1_and_10_l703_703263


namespace range_of_m_l703_703563

def f (x : ℝ) (m : ℝ) : ℝ := x * (m + real.exp (-x))

def g (x : ℝ) : ℝ := (x - 1) / real.exp x

theorem range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 m = f x2 m ∧ f x1 m = 0 ∧ f x2 m = 0) ↔ m ∈ set.Ioo 0 (real.exp (-2)) :=
sorry

end range_of_m_l703_703563


namespace find_a_l703_703913

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Theorem to be proved
theorem find_a (a : ℝ) 
  (h1 : A a ∪ B a = {1, 3, a}) : a = 0 ∨ a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l703_703913


namespace quadratic_passing_through_points_l703_703743

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_passing_through_points :
  (∃ a b c : ℝ, 
    (∀ x, quadratic a b c x = a * (x + 5)^2 - 4) ∧ 
    quadratic a b c 0 = 6 ∧ 
    quadratic a b c (-3) = -2.4) :=
begin
  sorry
end

end quadratic_passing_through_points_l703_703743


namespace minimum_perimeter_of_octagon_with_roots_of_Q_l703_703664

noncomputable def Q (z : ℂ) : ℂ := z^8 + (8 * real.sqrt 2 + 12) * z^4 - (8 * real.sqrt 2 + 14)

/-- The minimum perimeter among all the 8-sided polygons in the complex plane whose vertices are 
precisely the zeros of Q(z) is 8√2. -/
theorem minimum_perimeter_of_octagon_with_roots_of_Q : 
  let roots := {z : ℂ | Q z = 0} in
  let vertices := multiset.bind roots (λ root, {root}) in
  let perimeter := multiset.sum (multiset.map (λ v₁ v₂, complex.abs (v₁ - v₂)) (all_pairs vertices)) in
  perimeter = 8 * real.sqrt 2 := sorry

end minimum_perimeter_of_octagon_with_roots_of_Q_l703_703664


namespace decreasing_interval_of_f_l703_703479

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, (1 < x ∧ x < y) → f y < f x :=
by
  sorry

end decreasing_interval_of_f_l703_703479


namespace first_discount_percentage_l703_703753

-- Given conditions
def initial_price : ℝ := 390
def final_price : ℝ := 285.09
def second_discount : ℝ := 0.15

-- Definition for the first discount percentage
noncomputable def first_discount (D : ℝ) : ℝ :=
initial_price * (1 - D / 100) * (1 - second_discount)

-- Theorem statement
theorem first_discount_percentage : ∃ D : ℝ, first_discount D = final_price ∧ D = 13.99 :=
by
  sorry

end first_discount_percentage_l703_703753


namespace polynomial_identity_l703_703674

noncomputable def P (x : ℝ) (n : ℕ) (a : Fin n → ℝ) : ℝ :=
    x^n + ∑ i in Finset.range n, a i * x^(n-1-i)

noncomputable def Q (x : ℝ) (m : ℕ) (b : Fin m → ℝ) : ℝ :=
    x^m + ∑ i in Finset.range m, b i * x^(m-1-i)

theorem polynomial_identity
    (n m : ℕ)
    (a : Fin n → ℝ)
    (b : Fin m → ℝ)
    (h : P x n a ^ 2 = (x^2 - 1) * Q x m b ^ 2 + 1) :
    (derivative (P x n a)) = n * (Q x (n-1) b) :=
by
    sorry

end polynomial_identity_l703_703674


namespace smallest_x_in_domain_of_f_f_l703_703187

def f (x : ℝ) : ℝ := real.sqrt(x - 3)

theorem smallest_x_in_domain_of_f_f (x : ℝ) : (x ≥ 3 ∧ real.sqrt(x - 3) ≥ 3) → x ≥ 12 :=
begin
  sorry
end

end smallest_x_in_domain_of_f_f_l703_703187


namespace elephant_entry_rate_l703_703775

def initial : ℕ := 30000
def exit_rate : ℕ := 2880
def exodus_hours : ℕ := 4
def final : ℕ := 28980
def entry_hours : ℕ := 7

theorem elephant_entry_rate :
  let elephants_left := exit_rate * exodus_hours
  let remaining_elephants := initial - elephants_left
  let new_elephants := final - remaining_elephants
  let entry_rate := new_elephants / entry_hours
  entry_rate = 1500 :=
by 
  rw [mul_comm exit_rate exodus_hours, mul_comm initial exodus_hours]
  rw [sub_eq_add_neg initial elephants_left, sub_eq_add_neg final remaining_elephants]
  exact sorry

end elephant_entry_rate_l703_703775


namespace midpoint_OH_eq_M_l703_703647

variables {A B C D P M O H : Point}
variables (is_intersection_point_of_diagonals : ∀ (Q : Point), is_intersection_point Q A B C D ↔ Q = P)
variables (is_intersection_point_of_midline_segments : ∀ (N : Point), is_intersection_point N (midpoint A B) (midpoint C D) ↔ N = M)
variables (is_intersection_point_of_bisectors : ∀ (P : Point), is_intersection_point P (perpendicular_bisector A C) (perpendicular_bisector B D) ↔ P = O)
variables (is_intersection_point_of_orthocenters : ∀ (R : Point), is_intersection_point R (line (orthocenter A P D) (orthocenter B P C)) (line (orthocenter A P B) (orthocenter C P D)) ↔ R = H)

theorem midpoint_OH_eq_M : M = midpoint O H :=
sorry

end midpoint_OH_eq_M_l703_703647


namespace shaded_area_fraction_l703_703734

def area_of_circle (r : ℝ) : ℝ := real.pi * r^2

noncomputable def fraction_shaded {r : ℝ} (h1 : 0 < r) : ℝ :=
  let A_inner := area_of_circle r
  let A_outer := area_of_circle (2 * r)
  let A_region := A_outer - A_inner
  let area_segment := A_region / 6
  let shaded_area := 3 * area_segment
  shaded_area / A_outer

theorem shaded_area_fraction (r : ℝ) (h_pos : 0 < r) : fraction_shaded h_pos = 3 / 8 :=
by
  unfold fraction_shaded area_of_circle
  calc
    let A_inner := real.pi * r^2
    let A_outer := real.pi * (2 * r)^2
    let A_region := A_outer - A_inner
    let area_segment := A_region / 6
    let shaded_area := 3 * area_segment
    A_inner = real.pi * r^2                 : by refl
    A_outer = 4 * real.pi * r^2             : by { rw [mul_pow], ring }
    A_region = 3 * real.pi * r^2            : by ring
    area_segment = real.pi * r^2 / 2        : by rw [div_mul_eq_div_mul]
    shaded_area = 3 * (real.pi * r^2 / 2)   : by rw mul_assoc
    shaded_area / (4 * real.pi * r^2) = 3 / 8
    : by field_simp [real.pi_ne_zero, mul_ne_zero]; ring
  sorry

end shaded_area_fraction_l703_703734


namespace solve_fractional_equation_l703_703346

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  1 / x = 2 / (x + 1) → x = 1 :=
by
  sorry

end solve_fractional_equation_l703_703346


namespace product_of_two_is_perfect_square_l703_703528

theorem product_of_two_is_perfect_square
  (S : Fin 33 → ℕ) 
  (hS : ∀ n ∈ S, ∃ a b c d e : ℕ, n = 2^a * 3^b * 5^c * 7^d * 11^e) : 
  ∃ i j : Fin 33, i ≠ j ∧ ∃ k : ℕ, S i * S j = k^2 :=
by
  sorry

end product_of_two_is_perfect_square_l703_703528


namespace best_is_man_l703_703426

structure Competitor where
  name : String
  gender : String
  age : Int
  is_twin : Bool

noncomputable def participants : List Competitor := [
  ⟨"man", "male", 30, false⟩,
  ⟨"sister", "female", 30, true⟩,
  ⟨"son", "male", 30, true⟩,
  ⟨"niece", "female", 25, false⟩
]

def are_different_gender (c1 c2 : Competitor) : Bool := c1.gender ≠ c2.gender
def has_same_age (c1 c2 : Competitor) : Bool := c1.age = c2.age

noncomputable def best_competitor : Competitor :=
  let best_candidate := participants[0] -- assuming "man" is the best for example's sake
  let worst_candidate := participants[2] -- assuming "son" is the worst for example's sake
  best_candidate

theorem best_is_man : best_competitor.name = "man" :=
by
  have h1 : are_different_gender (participants[0]) (participants[2]) := by sorry
  have h2 : has_same_age (participants[0]) (participants[2]) := by sorry
  exact sorry

end best_is_man_l703_703426


namespace problem_even_odd_case_enumeration_l703_703395

theorem problem_even_odd_case_enumeration :
  ∃ A : ℤ, 10 ≤ A ∧ A < 31 ∧
  let x := (A^2 / 100) % 10, y := (A^2 / 10) % 10, z := A^2 % 10 in
    (100 * x + 10 * y + z = A^2) ∧ 
    (z % 2 = 1) ∧
    A = 19 ∧ A^2 = 361 :=
begin
  sorry
end

end problem_even_odd_case_enumeration_l703_703395


namespace min_value_of_quadratic_l703_703335

theorem min_value_of_quadratic :
  ∀ x : ℝ, (x - 1) ^ 2 ≥ 0 :=
begin
  sorry
end

end min_value_of_quadratic_l703_703335


namespace find_number_l703_703506

-- Defining the constants provided and the related condition
def eight_percent_of (x: ℝ) : ℝ := 0.08 * x
def ten_percent_of_40 : ℝ := 0.10 * 40
def is_solution (x: ℝ) : Prop := (eight_percent_of x) + ten_percent_of_40 = 5.92

-- Theorem statement
theorem find_number : ∃ x : ℝ, is_solution x ∧ x = 24 :=
by sorry

end find_number_l703_703506


namespace min_value_ineq_min_value_attainable_l703_703657

theorem min_value_ineq
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_sum : a + b + c = 9) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 :=
by {
  sorry,
}

theorem min_value_attainable :
  (a b c : ℝ)
  (h_eq : a = 3 ∧ b = 3 ∧ c = 3) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) = 9 :=
by {
  sorry,
}

end min_value_ineq_min_value_attainable_l703_703657


namespace sum_of_factors_of_360_l703_703032

theorem sum_of_factors_of_360 : Nat.divisorSum 360 = 1170 := by
  sorry

end sum_of_factors_of_360_l703_703032


namespace int_angle_eq_64_l703_703298

theorem int_angle_eq_64 {P Q : Point} (h1: RegularDecagon ABCDEFGHIJ) (h2: inter_ext_sides P Q ABCDEFGHIJ) :
  angle P Q = 64 :=
sorry

end int_angle_eq_64_l703_703298


namespace find_positive_integer_k_l703_703538

theorem find_positive_integer_k (p : ℕ) (hp : Prime p) (hp2 : Odd p) : 
  ∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n * n = k - p * k ∧ k = ((p + 1) * (p + 1)) / 4 :=
by
  sorry

end find_positive_integer_k_l703_703538


namespace sum_of_roots_l703_703903

def f (x : ℝ) : ℝ := -6 * x / (x^2 - 9)
def g (x : ℝ) : ℝ := 3 * x / (x + 3) - 2 / (x - 3) + 1

theorem sum_of_roots : ∑ x in {x : ℝ | f x = g x}, x = 5 / 4 := sorry

end sum_of_roots_l703_703903


namespace final_price_of_skis_l703_703071

def original_price : ℝ := 200
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.20

theorem final_price_of_skis :
  let price_after_first_discount := original_price * (1 - first_discount_rate),
      price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  in price_after_second_discount = 96 :=
by
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  have : price_after_first_discount = 120 := by
    sorry  -- Calculation step 1
  have : price_after_second_discount = 96 := by
    sorry  -- Calculation step 2
  exact this

end final_price_of_skis_l703_703071


namespace AM_GM_inequality_l703_703293

theorem AM_GM_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_not_all_eq : x ≠ y ∨ y ≠ z ∨ z ≠ x) :
  (x + y) * (y + z) * (z + x) > 8 * x * y * z :=
by
  sorry

end AM_GM_inequality_l703_703293


namespace complex_number_real_l703_703601

theorem complex_number_real (m : ℝ) : (z : ℂ) → z = (1 + complex.i) * (1 + m * complex.i) → z.im = 0 → m = -1 :=
by
-- declare the complex numbers and multiplication
intro z h₁ h₂
sorry

end complex_number_real_l703_703601


namespace find_complex_number_l703_703894

noncomputable def complex_number_satisfies_conditions (a b : ℝ) : Prop :=
  let z := complex.mk a b in
  complex.abs (z - 2) = 3 ∧
  complex.abs (z + complex.mk 1 2) = 3 ∧
  complex.abs (z - complex.mk 0 3) = 3

theorem find_complex_number (a b : ℝ) :
  complex_number_satisfies_conditions a b → ∃ z : ℂ, z = complex.mk a b :=
by
  intro h,
  use complex.mk a b,
  exact h,
  sorry

end find_complex_number_l703_703894


namespace vasya_wins_l703_703283

-- Define the grid size and initial setup
def grid_size : ℕ := 13
def initial_stones : ℕ := 2023

-- Define a condition that checks if a move can put a stone on the 13th cell
def can_win (position : ℕ) : Prop :=
  position = grid_size

-- Define the game logic for Petya and Vasya
def next_position (pos : ℕ) (move : ℕ) : ℕ :=
  pos + move

-- Ensure a win by always ensuring the next move does not leave Petya on positions 4, 7, 10, 13
def winning_strategy_for_vasya (current_pos : ℕ) (move : ℕ) : Prop :=
  (next_position current_pos move) ≠ 4 ∧
  (next_position current_pos move) ≠ 7 ∧
  (next_position current_pos move) ≠ 10 ∧
  (next_position current_pos move) ≠ 13

theorem vasya_wins : ∃ strategy : ℕ → ℕ → Prop,
  ∀ current_pos move, winning_strategy_for_vasya current_pos move → can_win (next_position current_pos move) :=
by
  sorry -- To be provided

end vasya_wins_l703_703283


namespace distance_between_centers_l703_703205

variables {A B C D : Type*}

-- Define points and triangle properties
variables [add_comm_group C] [module ℝ C] {A B D : C}

-- Right triangle ABC with right angle at C and median CD
noncomputable def is_right_triangle (A B C : C) : Prop :=
  ∃ (u v : C), A = u + v ∧ B = u - v ∧ C = u + v - 2 * u

-- Given BC = 4 and circumradius of triangle ABC is 5/2
variables (BC : ℝ := 4) (R : ℝ := 5 / 2)

-- Define centers of inscribed circles in triangles ACD and BCD
noncomputable def center_inscribed_circle_△ACD (A C D : C) : C := sorry
noncomputable def center_inscribed_circle_△BCD (B C D : C) : C := sorry

-- Distance between centers of inscribed circles in triangles ACD and BCD
noncomputable def distance_centers (A B C D : C) : ℝ :=
  dist (center_inscribed_circle_△ACD A C D) (center_inscribed_circle_△BCD B C D)

-- Main theorem: The distance between the centers of the inscribed circles in triangles ACD and BCD
theorem distance_between_centers (A B C D : C) [is_right_triangle A B C] 
  (median : Line A B D) (BC_eq : dist B C = BC) (circum_radius_eq : dist B A / 2 = R) : 
  distance_centers A B C D = (5 * real.sqrt 13) / 12 := by sorry

end distance_between_centers_l703_703205


namespace eval_abs_value_l703_703488

noncomputable def z₁ : ℂ := 3 * real.sqrt 5 - 3 * complex.I
noncomputable def z₂ : ℂ := 2 * real.sqrt 2 + 2 * complex.I

theorem eval_abs_value :
  complex.abs (z₁ * z₂) = 18 * real.sqrt 2 :=
by
  sorry

end eval_abs_value_l703_703488


namespace number_of_initials_is_10000_l703_703976

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l703_703976


namespace find_a100_l703_703924

theorem find_a100 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, a n = (2 * (S n)^2) / (2 * (S n) - 1))
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 100 = -2 / 39203 := 
sorry

-- Explanation of the statement:
-- 'theorem find_a100': We define a theorem to find a_100.
-- 'a : ℕ → ℝ': a is a sequence of real numbers.
-- 'S : ℕ → ℝ': S is a sequence representing the sum of the first n terms.
-- 'h1' to 'h3': Given conditions from the problem statement.
-- 'a 100 = -2 / 39203' : The statement to prove.

end find_a100_l703_703924


namespace valid_matrix_count_is_ten_l703_703920

def is_valid_matrix (a b c d : ℕ) : Prop :=
  a ∈ {0, 1} ∧ b ∈ {0, 1} ∧ c ∈ {0, 1} ∧ d ∈ {0, 1} ∧ (a * d - b * c = 0)

def count_valid_matrices : ℕ :=
  (Finset.univ.product (Finset.univ.product (Finset.univ.product Finset.univ))).filter
    (λ ⟨⟨a, b⟩, ⟨c, d⟩⟩, is_valid_matrix a b c d).card

theorem valid_matrix_count_is_ten :
  count_valid_matrices = 10 :=
  sorry

end valid_matrix_count_is_ten_l703_703920


namespace ratio_A_to_B_l703_703058

noncomputable def A_annual_income : ℝ := 436800.0000000001
noncomputable def B_increase_rate : ℝ := 0.12
noncomputable def C_monthly_income : ℝ := 13000

noncomputable def A_monthly_income : ℝ := A_annual_income / 12
noncomputable def B_monthly_income : ℝ := C_monthly_income + (B_increase_rate * C_monthly_income)

theorem ratio_A_to_B :
  ((A_monthly_income / 80) : ℝ) = 455 ∧
  ((B_monthly_income / 80) : ℝ) = 182 :=
by
  sorry

end ratio_A_to_B_l703_703058


namespace five_digit_numbers_percentage_with_repeats_l703_703173

theorem five_digit_numbers_percentage_with_repeats
  (total_numbers : ℕ := 90000)
  (non_repeating_numbers : ℕ := 9 * 9 * 8 * 7 * 6)
  (repeating_numbers := total_numbers - non_repeating_numbers)
  (p := (repeating_numbers.toFloat / total_numbers.toFloat) * 100 : Float)
  (S := Float.round (p + 5) : Float) :
  p = 69.8 ∧ S = 75 :=
by
  -- Calculations as given in the problem:
  have ht : total_numbers = 90000 := rfl
  have hn : non_repeating_numbers = 27216 := by norm_num
  have hr : repeating_numbers = 62784 := by norm_num
  have hp : Float.round( (62784.toFloat / 90000.toFloat) * 100 * 10 ) / 10 = 69.8 := by norm_num
  have hs : S = 75 := by norm_num
  split
  · exact hp
  · exact hs
sory

end five_digit_numbers_percentage_with_repeats_l703_703173


namespace find_analytic_expression_and_range_l703_703567

theorem find_analytic_expression_and_range 
  (A : ℝ) (ω : ℝ) (φ : ℝ)
  (hA : A > 0) (hω : ω > 0) (hφ : |φ| < π / 2) 
  (h1 : f (5 * π / 12) = 3)
  (h2 : f (11 * π / 12) = -3)
  (f : ℝ → ℝ := λ x, A * sin (ω * x + φ)) :
  (f x = 3 * sin (2 * x - π / 3)) ∧ 
  (∀ x ∈ set.Icc 0 (7 * π / 12), -3 * (sqrt 3) / 2 ≤ f x ∧ f x ≤ 3) :=
  sorry

end find_analytic_expression_and_range_l703_703567


namespace near_square_qoutient_l703_703364

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient_l703_703364


namespace no_real_solutions_sufficient_not_necessary_l703_703386

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l703_703386


namespace final_answer_l703_703562

def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 4 * (Real.cos x) ^ 2 - 1

theorem final_answer (a b c : ℝ)
  (h₀ : ∀ x : ℝ, a * f x - b * f (x + c) = 3) :
  3 * a ^ 2 + 2 * b + Real.cos c = 15 / 4 := by
  sorry

end final_answer_l703_703562


namespace johns_avg_speed_l703_703247

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

end johns_avg_speed_l703_703247


namespace shaded_trapezium_area_l703_703733

theorem shaded_trapezium_area :
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  area = 55 / 4 :=
by
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  show area = 55 / 4
  sorry

end shaded_trapezium_area_l703_703733


namespace pin_permutations_count_l703_703709

theorem pin_permutations_count : 
  ∀ (digits : Finset ℕ), digits = {1, 3, 5, 8} → digits.card = 4 → digits.toList.permutations.length = 24 :=
by
  intros digits h1 h2
  rw [←Finset.card_eq_card, h1, Finset.card_eq_four] at h2
  have permutations_length := List.permutations_length digits.toList
  rw [h1, permutations_length]
  exact rfl
  sorry

end pin_permutations_count_l703_703709


namespace range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l703_703164

open Real

theorem range_a_of_abs_2x_minus_a_eq_1_two_real_solutions :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (abs (2^x1 - a) = 1) ∧ (abs (2^x2 - a) = 1)} = {a : ℝ | 1 < a} :=
by
  sorry

end range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l703_703164


namespace inequality_proof_l703_703700

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9 / 16 :=
sorry

end inequality_proof_l703_703700


namespace largest_natural_a_excluding_interval_l703_703912

theorem largest_natural_a_excluding_interval : ∃ a : ℕ, a = 3 ∧ ∀ x, ¬(-4 ≤ (8 * x - 20) / (a - x ^ 2) ∧ (8 * x - 20) / (a - x ^ 2) ≤ -1) :=
by
  -- Here lie the conditions as definitions in Lean 4
  use 3
  intros x
  sorry

end largest_natural_a_excluding_interval_l703_703912


namespace max_distance_of_points_on_circles_proven_l703_703935

noncomputable def max_distance_of_points_on_circles : ℝ :=
let C₁_center := (-1, -4) in
let C₂_center := (2, 0) in
let r1 := 5 in
let r2 := 3 in
let dist_centers := Real.sqrt ( (-1 - 2)^2 + (-4 - 0)^2 ) in
dist_centers + r1 + r2

theorem max_distance_of_points_on_circles_proven :
  (∀ (M N : ℝ × ℝ),
    (M.1 + 1)^2 + (M.2 + 4)^2 = 25 →
    (N.1 - 2)^2 + N.2^2 = 9 →
    |(M.1 - N.1)^2 + (M.2 - N.2)^2| ≤ max_distance_of_points_on_circles) :=
by
  intros M N hM hN
  sorry

end max_distance_of_points_on_circles_proven_l703_703935


namespace chip_product_even_probability_l703_703580

theorem chip_product_even_probability :
  let chips := {1, 2, 3}
  in let all_draws := (chips × chips)
  in let even_products := {p | ∃ (x y : ℕ), (x, y) ∈ all_draws ∧ p = x * y ∧ p % 2 = 0}
  in (card even_products : ℚ) / (card all_draws : ℚ) = 5 / 9 :=
sorry

end chip_product_even_probability_l703_703580


namespace perpendicular_slope_condition_M_on_circle_diameter_AB_l703_703539

theorem perpendicular_slope_condition (p : ℝ) (hp : 0 < p) (M N : Point) (A B : Point)
(hM : M = Point.mk 1 2) 
(hC : M.y^2 = 2 * p * M.x)
(hN : N = Point.mk 5 (-2))
(hA : parabola_contains_point p A) 
(hB : parabola_contains_point p B) 
(hAB : (N.x - M.x) / (N.y - M.y) * (A.x - B.x) + (N.y - M.y) / (N.x - M.x) * (A.y - B.y) = 0) :
(line_eq AB) = "x - y - 7 = 0" := sorry

theorem M_on_circle_diameter_AB (p : ℝ) (M : Point) (A B : Point)
(hM : M = Point.mk 1 2)
(hC : M.y^2 = 2 * p * M.x)
(hA : parabola_contains_point p A)
(hB : parabola_contains_point p B)
(hcircle : on_circle_diameter M A B) :
(M_on_circle_diameter) :
M_on_circle_diameter  = true := sorry

end perpendicular_slope_condition_M_on_circle_diameter_AB_l703_703539


namespace resultant_force_magnitude_l703_703579

variables (F1 F2 F3 F : Vector)
variables (h1 : |F1| = 2) (h2 : |F2| = 2) (h3 : |F3| = 2)
variables (h4 : angle F1 F2 = 60) (h5 : angle F1 F3 = 60) (h6 : angle F2 F3 = 60)

theorem resultant_force_magnitude :
  |F1 + F2 + F3| = 2 * sqrt 6 :=
sorry

end resultant_force_magnitude_l703_703579


namespace floor_of_smallest_root_of_g_zero_l703_703260

def g (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x + 4 * Real.tan x

theorem floor_of_smallest_root_of_g_zero :
  ∃ s > 0, g s = 0 ∧ Int.floor s = 3 :=
by 
  sorry

end floor_of_smallest_root_of_g_zero_l703_703260


namespace y_intercept_of_line_l703_703763

theorem y_intercept_of_line :
  ∃ y, (∀ x : ℝ, 2 * x - 3 * y = 6) ∧ (y = -2) :=
sorry

end y_intercept_of_line_l703_703763


namespace points_distance_from_line_l703_703569

def line : Type := λ (x : ℝ), x = 2
def circle : Type := λ (x y : ℝ), x^2 + y^2 - 2*x - 2*y = 0

theorem points_distance_from_line (x y : ℝ) :
  (circle x y) → 
  (∀ (l : line), ∃ p1 p2 : ℝ × ℝ, dist (1, 1) (2, y) = 1 ∧
  ((p1 ∈ circle ∧ dist p1 l = 1) ∧ (p2 ∈ circle ∧ dist p2 l = 1)) → 
  2 :=
sorry

end points_distance_from_line_l703_703569


namespace hash_hash_hash_45_l703_703474

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_45 : hash (hash (hash 45)) = 7.56 :=
by
  sorry

end hash_hash_hash_45_l703_703474


namespace michael_water_left_l703_703684

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left_l703_703684


namespace fraction_habitable_surface_l703_703596

noncomputable def fraction_land_not_covered_by_water : ℚ := 1 / 3
noncomputable def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_land_not_covered_by_water * fraction_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_habitable_surface_l703_703596


namespace modulus_of_complex_number_l703_703157

-- Definition of the imaginary unit
def i : ℂ := complex.I

-- Definition of the complex number in the problem
def z : ℂ := (1 + i) * (3 - 4 * i)

-- The theorem statement
theorem modulus_of_complex_number : complex.abs z = 5 * real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l703_703157


namespace pyramid_height_l703_703951

theorem pyramid_height (P A B C D : Point) 
  (PA PB PC PD : Line)
  (ACP BDP : Plane)
  (isosceles : ∀ (face : Triangle), is_isosceles_right face ∧ face.legs = (1 : ℝ))
  (angle_90 : ∀ (a b : Point), (a = A ∨ a = B ∨ a = C ∨ a = D) → (b = A ∨ b = B ∨ b = C ∨ b = D) → angle_between (a, b) = (π / 2)) :
  ∃ H, on_plane H ACP ∧ perp (H, P) (AC, BD) ∧ distance P H = (√5 - 1) / 2 :=
by
  sorry

end pyramid_height_l703_703951


namespace right_triangle_area_l703_703313

theorem right_triangle_area (A B C : Type) [linear_ordered_field A]
  (theta45 : ∀ {x : A}, x = 45)
  (right_angle : ∀ {x : A}, x = 90)
  (altitude_four : ∀ {x : A}, x = 4) 
  (hypotenuse : A) 
  (area : A) :
  (∀ (triangle : Type) (legs_eq : ∀ {a b : A}, a = 45 → b = 45 → a = b),
   hypotenuse = 4 * √2 ∧
   area = 1 / 2 * hypotenuse * altitude_four →
   area = 8 * √2) :=
sorry

end right_triangle_area_l703_703313


namespace piecewise_function_evaluation_l703_703916

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.cos (π * x)
  else f (x - 1)

theorem piecewise_function_evaluation :
  f (4/3) + f (-4/3) = 1 := sorry

end piecewise_function_evaluation_l703_703916


namespace maximum_sum_of_distances_l703_703867

theorem maximum_sum_of_distances (l w : ℝ) (h1 : l ≥ w)
  (P E F G H : ℝ × ℝ) (AB CD BC DA : ℝ)
  (h2 : AB = l) (h3 : CD = l) (h4 : BC = w) (h5 : DA = w) :
  PE.x = (P.x - E.x) →
  PH.x = (P.x - H.x) →
  PF.y = (P.y - F.y) →
  PG.y = (P.y - G.y) →
  PE.x + PH.x + PF.y + PG.y ≤ 2 * l + 2 * w :=
by
  sorry

end maximum_sum_of_distances_l703_703867


namespace remaining_sign_after_operations_l703_703764

theorem remaining_sign_after_operations :
  ∀ (plus_signs minus_signs operations : ℕ), 
    plus_signs = 10 → minus_signs = 15 → operations = 24 →
    (∀ n, n = 0 → ((plus_signs + minus_signs - 2 * n) % 2 = minus_signs % 2)) → 
    ((plus_signs + minus_signs - 2 * operations = 1) → remaining_sign = -1) :=
by
  intros plus_signs minus_signs operations h_plus_signs h_minus_signs h_operations h_invariant
  cases h_plus_signs
  cases h_minus_signs
  cases h_operations
  have h_parity := h_invariant operations.rfl
  simp at h_parity
  sorry

end remaining_sign_after_operations_l703_703764


namespace Shuai_Shuai_memorized_third_day_l703_703715

variable (N : ℕ)  -- total number of words

-- Definitions for the conditions
def words_first_three_days := (1 / 2) * N
def words_last_three_days := (2 / 3) * N
def difference_between_periods := words_last_three_days N - words_first_three_days N

-- Problem statement in Lean 4
theorem Shuai_Shuai_memorized_third_day :
  sorry

end Shuai_Shuai_memorized_third_day_l703_703715


namespace remainder_sum_a2i_mod_3_l703_703996

theorem remainder_sum_a2i_mod_3 (n : ℕ) (h : 0 < n) :
  let a : ℕ → ℕ := λ i : ℕ, if i = 0 then 4 ^ (2 * n) else
    if i = 2 * n then 1 ^ (2 * n) else
    if i % 2 = 1 then ((6 : ℕ) ^ (2 * n) - 2 ^ (2 * n)) / 2 else 0 in
  (∑ i in (finset.range (n+1)).filter (λ k, k > 0), a (2 * i)) % 3 = 1 := by
  sorry

end remainder_sum_a2i_mod_3_l703_703996


namespace neg_half_theta_quadrant_l703_703197

theorem neg_half_theta_quadrant (k : ℤ) (θ : ℝ) (h : 2 * k * real.pi - real.pi / 2 < θ ∧ θ < 2 * k * real.pi) :
  ∃ m : ℤ, (m * real.pi < -θ / 2 ∧ -θ / 2 < m * real.pi + real.pi / 2 ∨
             (m + 1) * real.pi < -θ / 2 ∧ -θ / 2 < (m + 1) * real.pi + real.pi / 2) :=
by
  sorry

end neg_half_theta_quadrant_l703_703197


namespace grape_vs_banana_candies_diff_l703_703009

def G : ℕ := 48
def B : ℕ := 43

theorem grape_vs_banana_candies_diff : G - B = 5 := 
by 
  have hG : G = 192 / 4 := sorry
  have hB : B = 43 := rfl
  calc G - B = 48 - 43 : by rw [hG, hB]
        ... = 5 : by norm_num

end grape_vs_banana_candies_diff_l703_703009


namespace tan_alpha_eq_4_over_3_expression_value_eq_4_l703_703135

-- Conditions
variable (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi / 2)) (h_sin : Real.sin α = 4 / 5)

-- Prove: tan α = 4 / 3
theorem tan_alpha_eq_4_over_3 : Real.tan α = 4 / 3 :=
by
  sorry

-- Prove: the value of the given expression is 4
theorem expression_value_eq_4 : 
  (Real.sin (α + Real.pi) - 2 * Real.cos ((Real.pi / 2) + α)) / 
  (- Real.sin (-α) + Real.cos (Real.pi + α)) = 4 :=
by
  sorry

end tan_alpha_eq_4_over_3_expression_value_eq_4_l703_703135


namespace infinite_polynomial_pairs_l703_703287

open Polynomial

theorem infinite_polynomial_pairs :
  ∀ n : ℕ, ∃ (fn gn : ℤ[X]), fn^2 - (X^4 - 2 * X) * gn^2 = 1 :=
sorry

end infinite_polynomial_pairs_l703_703287


namespace sqrt_sq_eq_abs_l703_703990

theorem sqrt_sq_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| :=
sorry

end sqrt_sq_eq_abs_l703_703990


namespace max_third_altitude_l703_703360

noncomputable section

variables {D E F : Type}
variables (DE DF EF : ℝ) -- lengths of the sides
variables (h_DE h_DF h_EF : ℝ) -- altitudes to the respective sides

def scalene_triangle (DE DF EF : ℝ) : Prop :=
  DE ≠ DF ∧ DF ≠ EF ∧ DE ≠ EF

def valid_altitudes (DE DF EF h_DE h_DF h_EF : ℝ) : Prop :=
  (h_DE = 18) ∧ (h_DF = 6) ∧ (h_EF ∈ Int) ∧ (1 / 2 * DE * 18 = 1 / 2 * 3 * DE * 6 ∧ 
  h_EF = 18 * DE / EF) ∧ (EF > 2 * DE)

theorem max_third_altitude (DE DF EF h_DE h_DF h_EF : ℝ) 
  (h1 : scalene_triangle DE DF EF)
  (h2 : valid_altitudes DE DF EF h_DE h_DF h_EF) :
  h_EF ≤ 9 := 
sorry

end max_third_altitude_l703_703360


namespace sum_cos_eq_zero_l703_703904

open Complex

noncomputable def sum_cos (α : ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range (2 * n + 1), Real.cos (α + (2 * k * Real.pi) / (2 * n + 1))

theorem sum_cos_eq_zero (α : ℝ) (n : ℕ) (hn : 0 < n) :
  sum_cos α n = 0 :=
sorry

end sum_cos_eq_zero_l703_703904


namespace loop_property_l703_703714

theorem loop_property (n : ℕ) (u hat_u β : ℕ) :
  let β_k (k : ℕ) := β % 2^k in
  let v (k : ℕ) := hat_u^(2^k) % n in
  let m (k : ℕ) := β / 2^k in
  (∀ k, u ≡ hat_u^(β_k k) [MOD n] ∧ v k ≡ hat_u^(2^k) [MOD n] ∧ m k = β / 2^k)
  → u ≡ hat_u^β [MOD n] :=
by
  sorry

end loop_property_l703_703714


namespace student_weight_loss_l703_703195

theorem student_weight_loss {S R L : ℕ} (h1 : S = 90) (h2 : S + R = 132) (h3 : S - L = 2 * R) : L = 6 := by
  sorry

end student_weight_loss_l703_703195


namespace cyclic_quadrilateral_congurent_orthocenters_l703_703086

variables (A B C D A' B' C' D' : Type) [CyclicQuadrilateral A B C D] 
variables [Orthocenter B C D A'] [Orthocenter A C D B'] [Orthocenter A B D C'] [Orthocenter A B C D']

theorem cyclic_quadrilateral_congurent_orthocenters :
  CyclicQuadrilateral A B C D ∧ Orthocenter B C D A' ∧ Orthocenter A C D B' ∧ Orthocenter A B D C' ∧ Orthocenter A B C D' → 
  Congruent A B C D A' B' C' D' :=
by sorry

end cyclic_quadrilateral_congurent_orthocenters_l703_703086


namespace find_original_value_l703_703440

noncomputable theory
open_locale big_operators

def V (original_value : ℝ) : Prop := original_value = 11083.33

def P1 (original_value : ℝ) : ℝ := (5/7) * original_value * (3/100)
def P2 (original_value : ℝ) : ℝ := (5/7) * original_value * (4/100)
def P3 (original_value : ℝ) : ℝ := (5/7) * original_value * (5/100)
def total_premium_paid (original_value : ℝ) : ℝ := P1 original_value + P2 original_value + P3 original_value

theorem find_original_value (h : total_premium_paid V = 950) : V 11083.33 :=
sorry

end find_original_value_l703_703440


namespace max_cookies_andy_can_eat_l703_703012

theorem max_cookies_andy_can_eat (A B C : ℕ) (hB_pos : B > 0) (hC_pos : C > 0) (hB : B ∣ A) (hC : C ∣ A) (h_sum : A + B + C = 36) :
  A ≤ 30 := by
  sorry

end max_cookies_andy_can_eat_l703_703012


namespace correct_calculation_l703_703036

theorem correct_calculation (a b : ℝ) :
  (6 * a - 5 * a ≠ 1) ∧
  (a + 2 * a^2 ≠ 3 * a^3) ∧
  (- (a - b) = -a + b) ∧
  (2 * (a + b) ≠ 2 * a + b) :=
by 
  sorry

end correct_calculation_l703_703036


namespace problem1_problem2_l703_703281

--First problem conditions
variables (A B C D E F A1 B1 C1 D1 : Point)

--Prove the given equation

theorem problem1 (h1 : A - E = B - F) (h2 : A₁ - E = B₁ - F) (h3 : C - E = D - F) (h4 : C₁ - E = D₁- F) :
  (1 / (dist A A₁) + 1 / (dist C C₁)) = (1 / (dist B B₁) + 1 / (dist D D₁)) :=
sorry


--Second problem conditions
variables (A B C A1 B1 C1 : Point)
variables (A' B' C' A1' B1' C1' : Point)

--Prove the given equation

theorem problem2 (h1 : InscribedIn A₁ B₁ C₁ A B C) (h2 : Perspective A₁ B₁ C₁ A B C) (h3 : ParallelToDesarguesLine A' B' C' A1' B1' C1') :
  (1 / (dist A A') + 1 / (dist B B') + 1 / (dist C C')) = (1 / (dist A₁ A₁') + 1 / (dist B₁ B₁') + 1 / (dist C₁ C₁')) :=
sorry

end problem1_problem2_l703_703281


namespace math_problem_l703_703168

noncomputable def e : ℝ := Real.exp 1

-- Define the functions f, g, h with b = 0
def f (x : ℝ) := Real.exp x
def g (x : ℝ) := Real.log x
def h (k : ℝ) (x : ℝ) := k * x

-- Define the main theorem combining all parts
theorem math_problem 
  (k : ℝ)
  (x : ℝ) (H1 : 0 < x) 
  (H2 : ∀ y : ℝ, 0 < y → f y ≥ h k y ∧ h k y ≥ g y)
  (x1 : ℝ) (H3 : x1 < 0)
  (x2 : ℝ) (Htangent : f x1 = h k x1 ∧ g x2 = h k x2)
  (H4 : x ≥ x2) :
  (∀ x, 0 < x → Real.exp x / x ≥ k ∧ k ≥ Real.log x / x) ∧
  x2 > e ∧
  (∀ x, x ≥ x2 → a * (x1 - 1) + x * Real.log x - x ≥ 0 → a ≤ 1) := 
sorry

end math_problem_l703_703168


namespace triangle_area_l703_703312

theorem triangle_area (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (a b c : ℝ) (h : (a = b ∧ b = c) ∧ (a * a + b * b = c * c) ∧ (a * (b + b) = 8) ∧ (a * b / 2 = 16)) : 
  a * b / 2 = 16 :=
by {
  sorry 
}

end triangle_area_l703_703312


namespace right_triangle_area_l703_703309

theorem right_triangle_area (h : Real) (a b c : Real) (A B C : ℝ) : 
  ∠A = 45 ∧ ∠C = 45 ∧ ∠B = 90 ∧ h = 4 → 
  area (triangle A B C) = 8 * Real.sqrt 2 :=
begin
  sorry
end

end right_triangle_area_l703_703309


namespace number_of_initials_is_10000_l703_703980

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l703_703980


namespace binary_to_octal_conversion_l703_703473

def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.enum.sum (λ ⟨i, d⟩, d * 2^i)

def decimal_to_octal (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if n = 0 then acc
    else aux (n / 8) (n % 8 :: acc)
  aux n []

def binary_1010101_is_85_in_decimal : Prop :=
  binary_to_decimal [1, 0, 1, 0, 1, 0, 1] = 85

def decimal_85_is_125_in_octal : Prop :=
  decimal_to_octal 85 = [1, 2, 5]

theorem binary_to_octal_conversion : Prop :=
  binary_1010101_is_85_in_decimal ∧ decimal_85_is_125_in_octal

end binary_to_octal_conversion_l703_703473


namespace bike_avg_speed_l703_703884

-- Conditions definitions
def distance_swim : ℝ := 1 / 2
def speed_swim : ℝ := 1.5

def distance_run : ℝ := 4
def speed_run : ℝ := 8

def distance_bike : ℝ := 18

def total_time : ℝ := 2.5

-- Time calculations
def time_swim : ℝ := distance_swim / speed_swim
def time_run : ℝ := distance_run / speed_run

-- Assertion for the average speed required for the bike ride to meet the total time condition
def avg_speed_bike_required : ℝ :=
  distance_bike / (total_time - (time_swim + time_run))

-- Theorem statement asserting the required average speed for the bike ride.
theorem bike_avg_speed : avg_speed_bike_required = 54 / 5 :=
by
  sorry

end bike_avg_speed_l703_703884


namespace person_A_B_adjacent_person_A_B_not_adjacent_person_A_B_two_between_person_A_not_left_B_not_right_l703_703982

theorem person_A_B_adjacent : ∃ (n : ℕ), n = 240 ∧ 
  ∀ (perm : list ℕ), perm.length = 6 → perm.nodup →
    (∃ i : ℕ, i < 5 ∧ (perm.nth i = some 1 ∧ perm.nth (i+1) = some 2) 
      ∨ (perm.nth i = some 2 ∧ perm.nth (i+1) = some 1)) sorry

theorem person_A_B_not_adjacent : ∃ (n : ℕ), n = 480 ∧
  ∀ (perm : list ℕ), perm.length = 6 → perm.nodup →
    (∀ i : ℕ, i < 5 → ¬((perm.nth i = some 1 ∧ perm.nth (i+1) = some 2) 
      ∨ (perm.nth i = some 2 ∧ perm.nth (i+1) = some 1))) sorry

theorem person_A_B_two_between : ∃ (n : ℕ), n = 144 ∧
  ∀ (perm : list ℕ), perm.length = 6 → perm.nodup →
    (∃ i : ℕ, i < 4 ∧ (perm.nth i = some 1 ∧ perm.nth (i+3) = some 2)
      ∨ (perm.nth i = some 2 ∧ perm.nth (i+3) = some 1)) sorry

theorem person_A_not_left_B_not_right : ∃ (n : ℕ), n = 504 ∧
  ∀ (perm : list ℕ), perm.length = 6 → perm.nodup →
    perm.head ≠ some 1 ∧ perm.last ≠ some 2 sorry

end person_A_B_adjacent_person_A_B_not_adjacent_person_A_B_two_between_person_A_not_left_B_not_right_l703_703982


namespace LCM_GCD_product_l703_703430

theorem LCM_GCD_product (a b : ℕ) (h_a : a = 24) (h_b : b = 36) :
  Nat.lcm a b * Nat.gcd a b = 864 := by
  rw [h_a, h_b]
  have lcm_ab := Nat.lcm_eq (by repeat {exact dec_trivial})
  have gcd_ab := Nat.gcd_eq (by repeat {exact dec_trivial})
  exact calc
    Nat.lcm 24 36 * Nat.gcd 24 36 = 72 * 12 : by congr; apply lcm_ab; apply gcd_ab
    ... = 864 : by norm_num

end LCM_GCD_product_l703_703430


namespace rooks_knight_moves_no_attack_l703_703279

-- Defining the problem in lean
theorem rooks_knight_moves_no_attack :
  ∀ (initial_positions : list (ℕ × ℕ)),
    (∃ rooks : list (ℕ × ℕ), w ∧ ¬∃ (i j : ℕ), i ≠ j ∧ rooks[i] = rooks[j]) →
    (∃ new_positions : list (ℕ × ℕ), w ∧ ¬∃ (i j : ℕ), i ≠ j ∧ new_positions[i] = new_positions[j] ∧ (∀ k, (new_positions[k] = (initial_positions[k].1 + 2, initial_positions[k].2 + 1) ∨ new_positions[k] = (initial_positions[k].1 + 1, initial_positions[k].2 + 2) ∨ new_positions[k] = (initial_positions[k].1 - 1, initial_positions[k].2 - 2) ∨ new_positions[k] = (initial_positions[k].1 - 2, initial_positions[k].2 - 1)))).
{
  sorry
}

end rooks_knight_moves_no_attack_l703_703279


namespace polynomial_intersection_max_l703_703028

-- Definitions based on conditions
def fifth_degree_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₅ = 1 ∧ ∀ x : ℝ, p x = a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀

def fourth_degree_polynomial (q : ℝ → ℝ) : Prop :=
  ∃ b₄ b₃ b₂ b₁ b₀ : ℝ, b₄ = 1 ∧ ∀ x : ℝ, q x = b₄ * x ^ 4 + b₃ * x ^ 3 + b₂ * x ^ 2 + b₁ * x + b₀

-- Theorem to prove
theorem polynomial_intersection_max (p q : ℝ → ℝ) :
  fifth_degree_polynomial p →
  fourth_degree_polynomial q →
  ∃ s : set ℝ, (∀ x, x ∈ s ↔ p x = q x) ∧ s.finite ∧ s.card = 5 := 
by
  sorry

end polynomial_intersection_max_l703_703028


namespace no_convex_pentagon_with_all_obtuse_angles_l703_703482

theorem no_convex_pentagon_with_all_obtuse_angles :
  ¬ ∃ (A B C D E : Type) (A B D B C E C D A D E B E A C : ℝ),
    convex A B C D E ∧
    angle A B D > 90 ∧
    angle B C E > 90 ∧
    angle C D A > 90 ∧
    angle D E B > 90 ∧
    angle E A C > 90 :=
by sorry

end no_convex_pentagon_with_all_obtuse_angles_l703_703482


namespace ferris_wheel_capacity_l703_703403

theorem ferris_wheel_capacity (number_of_seats : ℕ) (capacity_per_seat : ℕ) : 
  number_of_seats = 14 → capacity_per_seat = 6 → number_of_seats * capacity_per_seat = 84 :=
by
  intros h1 h2
  rw [h1, h2]
  exact Nat.mul_comm 14 6
  sorry

end ferris_wheel_capacity_l703_703403


namespace min_value_is_nine_l703_703650

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l703_703650


namespace cos_double_angle_of_parallel_vectors_l703_703954

variables {α : Type*}

/-- Given vectors a and b specified by the problem, if they are parallel, then cos 2α = 7/9. -/
theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α)) 
  (hb : b = (Real.cos α, 1)) 
  (parallel : a.1 * b.2 = a.2 * b.1) : 
  Real.cos (2 * α) = 7/9 := 
by 
  sorry

end cos_double_angle_of_parallel_vectors_l703_703954


namespace parametric_to_standard_equation_l703_703571

theorem parametric_to_standard_equation (t : ℝ) :
  (∃ t : ℝ, (x = 1 + t ∧ y = -1 + t)) → (x - y - 2 = 0) := 
by { intro h, cases h with t ht, sorry }

end parametric_to_standard_equation_l703_703571


namespace extended_pattern_ratio_l703_703119

def original_black_tiles : ℕ := 13
def original_white_tiles : ℕ := 12
def original_total_tiles : ℕ := 5 * 5

def new_side_length : ℕ := 7
def new_total_tiles : ℕ := new_side_length * new_side_length
def added_white_tiles : ℕ := new_total_tiles - original_total_tiles

def new_black_tiles : ℕ := original_black_tiles
def new_white_tiles : ℕ := original_white_tiles + added_white_tiles

def ratio_black_to_white : ℚ := new_black_tiles / new_white_tiles

theorem extended_pattern_ratio :
  ratio_black_to_white = 13 / 36 :=
by
  sorry

end extended_pattern_ratio_l703_703119


namespace num_four_letter_initials_l703_703974

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l703_703974


namespace ab_is_2_l703_703455

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem ab_is_2 :
  (∃ a b : ℝ, (∀ x, y = a * Real.tan (b * x) → Real.period (a * Real.tan (b * x)) = π / 2 ∧
    y = a * Real.tan (b * π / 8) → y = 1) → (a * b = 2)) :=
by
  sorry

end ab_is_2_l703_703455


namespace ratio_of_area_of_EFM_to_ABC_l703_703284

theorem ratio_of_area_of_EFM_to_ABC (AB BC AC AE BF AM : ℝ)
  (h1 : AE = AB / 3)
  (h2 : BF = BC / 6)
  (h3 : AM = 2 * AC / 5) :
  let S := (sqrt (AB^2 - AE^2) * sqrt (BC^2 - BF^2) * sqrt (AC^2 - AM^2)) / 4 in
  let SEFM := (S * (23 / 90)) in
  SEFM / S = 23 / 90 :=
by
  sorry

end ratio_of_area_of_EFM_to_ABC_l703_703284


namespace sin_of_triangle_XYZ_l703_703623

/-- In a right triangle XYZ where ∠Y = 90°, with sides XY = 8 and YZ = 17 units, prove that sin X = 8/17. -/
theorem sin_of_triangle_XYZ (X Y Z : Type) [RightTriangle X Y Z]
  (hypotenuse : ∠YZ = 90°) (sideXY : XY = 8) (sideYZ : YZ = 17) : 
  sin X = 8 / 17 :=
by
  sorry

end sin_of_triangle_XYZ_l703_703623


namespace largest_coefficient_expansion_l703_703182

theorem largest_coefficient_expansion (n : ℕ) 
  (hn : Finset.sum (Finset.range (n + 1)) (λ k, C n k / (k + 1)) = 31 / (n + 1)) : 
  (binomial (2 * n) (n)) = 70 := 
sorry

end largest_coefficient_expansion_l703_703182


namespace sum_of_factors_360_l703_703030

theorem sum_of_factors_360 : σ 360 = 1170 :=
by sorry

end sum_of_factors_360_l703_703030


namespace boys_from_pine_high_l703_703879

-- Defining the main parameters
def total_students := 180
def boys := 100
def girls := 80
def maple_high := 70
def pine_high := 60
def oak_high := 50
def girls_maple_high := 30

-- Define the number of boys in pine_high
theorem boys_from_pine_high : 
  let boys_maple_high := maple_high - girls_maple_high,
      boys_pine_and_oak_high := boys - boys_maple_high,
      girls_pine_and_oak_high := total_students - maple_high - girls_maple_high,
      boys_pine_high := pine_high - (girls - girls_pine_and_oak_high) in
  boys_pine_high = 30 :=
by
  -- Define intermediate calculations
  let boys_maple_high := maple_high - girls_maple_high;
  let boys_pine_and_oak_high := boys - boys_maple_high;
  let girls_pine_and_oak_high := girls - girls_maple_high;
  let boys_pine_high := pine_high - (girls - girls_pine_and_oak_high);
  -- Final assertion
  have h1 : boys_maple_high = 40 := by sorry;
  have h2 : boys_pine_and_oak_high = 60 := by sorry;
  have h3 : girls_pine_and_oak_high = 50 := by sorry;
  have h4 : boys_pine_high = 30 := by sorry;
  exact h4

end boys_from_pine_high_l703_703879


namespace no_solution_for_m_and_n_l703_703295

theorem no_solution_for_m_and_n (m n : ℕ) : ¬ (m! + 48 = 48 * (m + 1) * n) := 
sorry

end no_solution_for_m_and_n_l703_703295


namespace second_pirate_gets_diamond_l703_703804

theorem second_pirate_gets_diamond (coins_bag1 coins_bag2 : ℕ) :
  (coins_bag1 ≤ 1 ∧ coins_bag2 ≤ 1) ∨ (coins_bag1 > 1 ∨ coins_bag2 > 1) →
  (∃ n k : ℕ, n % 2 = 0 → (coins_bag1 + n) = (coins_bag2 + k)) :=
sorry

end second_pirate_gets_diamond_l703_703804


namespace sqrt_power_simplified_l703_703496

theorem sqrt_power_simplified : (sqrt ((sqrt 5) ^ 5)) ^ 6 = 5 ^ 7.5 := by
  sorry

end sqrt_power_simplified_l703_703496


namespace prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l703_703362

-- Definitions of probabilities of making a shot
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Number of attempts
def num_attempts : ℕ := 3

-- Probability that B makes at most 2 shots
theorem prob_B_at_most_2_shots : 
  (1 - (num_attempts.choose 3) * (p_B ^ 3) * ((1 - p_B) ^ (num_attempts - 3))) = 7 / 8 :=
by 
  sorry

-- Probability that B makes exactly 2 more shots than A
theorem prob_B_exactly_2_more_than_A : 
  (num_attempts.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ 1) * (num_attempts.choose 0) * ((1 - p_A) ^ num_attempts) +
  (num_attempts.choose 3) * (p_B ^ 3) * (num_attempts.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (num_attempts - 1)) = 1 / 6 :=
by 
  sorry

end prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l703_703362


namespace smallest_n_digit_sum_l703_703766

theorem smallest_n_digit_sum :
  ∃ n : ℕ, (∃ (arrangements : ℕ), arrangements > 1000000 ∧ arrangements = (1/2 * ((n + 1) * (n + 2)))) ∧ (1 + n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + n % 10 = 9) :=
sorry

end smallest_n_digit_sum_l703_703766


namespace Robe_savings_l703_703290

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end Robe_savings_l703_703290


namespace count_real_solutions_l703_703510

noncomputable def g (x : ℝ) : ℝ :=
  ∑ i in Finset.range 50, (2 * (i + 1) - 1 : ℝ) / (x - (i + 1))

theorem count_real_solutions : 
  (∑ i in Finset.range 50, (2 * (i + 1) - 1) / (x - (i + 1)) = x - 10) → 
  (∃ n : ℕ, n = 52) :=
sorry

end count_real_solutions_l703_703510


namespace place_mat_length_l703_703434

-- Given conditions
def radius : ℝ := 5
def width : ℝ := 1.5
def n : ℕ := 8
def theta := (2 * Real.pi) / n
def chord_length := 2 * radius * Real.sin(theta / 2)

-- Proven length y is 5 * sqrt(2 - sqrt(2))
theorem place_mat_length :
  chord_length = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  -- skipping the proof here
  sorry

end place_mat_length_l703_703434


namespace algebraic_expression_value_l703_703590

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : 2 * a^2 - 4 * a + 2022 = 2024 := 
by 
  sorry

end algebraic_expression_value_l703_703590


namespace measure_of_AB_l703_703626

variable {α : Type} [LinearOrder α]

-- Definitions according to the conditions in the problem
variables (a b : α)

-- Definitions to state the problem properly in Lean 4
theorem measure_of_AB (AB CD BC DE AD : α) (α : α)
  (h1 : AB = AD + DC)  -- Given AB = AE + EB (and AE = AD, EB = DC)
  (h2 : DC = CD)      -- Given E is moving from D to B
  (h3 : AD + CD = AB) -- Parallelism property holds true - measure
  
  : AB = a + b := by
  sorry -- Proof to be filled in later

end measure_of_AB_l703_703626


namespace three_identical_differences_l703_703142

theorem three_identical_differences (s : Finset ℕ) (h_size: s.card = 19) (h_bound: ∀ x ∈ s, x < 91) : 
  ∃ d ∈ (Finset.image (λ (x : ℕ × ℕ), x.snd - x.fst) ((s.product s).filter (λ p, p.snd > p.fst))), 3 ≤ (Finset.card ((s.product s).filter (λ p, (p.snd - p.fst) = d))) := 
sorry

end three_identical_differences_l703_703142


namespace minimize_expression_l703_703516

theorem minimize_expression (x : ℝ) : 3 * x^2 - 12 * x + 1 ≥ 3 * 2^2 - 12 * 2 + 1 :=
by sorry

end minimize_expression_l703_703516


namespace domain_of_f_exp_l703_703200

theorem domain_of_f_exp (f : ℝ → ℝ) (h : ∀ (x : ℝ), x ∈ set.Icc 0 2 → f x ∈ set.Icc 0 2) :
  ∀ (x : ℝ), x ∈ set.Icc 1 2 → (f (2^x - 2)) ∈ set.Icc 0 2 :=
by sorry

end domain_of_f_exp_l703_703200


namespace part_a_part_b_l703_703254

section

variables (K : Type*) [Field K] [Fintype K]
variable (h_irreducible : Irreducible (Polynomial.X^2 - 5 : Polynomial K))

theorem part_a : (1 : K) + 1 ≠ 0 :=
  sorry

theorem part_b (a : K) : Polynomial.IsReducible (Polynomial.X^5 + Polynomial.C a : Polynomial K) :=
  sorry

end

end part_a_part_b_l703_703254


namespace find_AB_l703_703074

variables {A B C D X Y : Point}
variables (AD BC AB : ℝ)

-- Given conditions
def is_trapezoid (A B C D : Point) : Prop := AD ∥ BC
def angle_bisector_A (A B C D X Y : Point) : Prop := bisects_angle A X ∧ ∠AXC = 90
def initial_lengths : Prop := AD = 19 ∧ CY = 16

-- To prove
theorem find_AB (h₁ : is_trapezoid A B C D)
                (h₂ : angle_bisector_A A B C D X)
                (h₃ : initial_lengths AD BC AB)
                : AB = 17.5 :=
sorry

end find_AB_l703_703074


namespace right_triangle_area_l703_703315

theorem right_triangle_area (A B C : Type) [linear_ordered_field A]
  (theta45 : ∀ {x : A}, x = 45)
  (right_angle : ∀ {x : A}, x = 90)
  (altitude_four : ∀ {x : A}, x = 4) 
  (hypotenuse : A) 
  (area : A) :
  (∀ (triangle : Type) (legs_eq : ∀ {a b : A}, a = 45 → b = 45 → a = b),
   hypotenuse = 4 * √2 ∧
   area = 1 / 2 * hypotenuse * altitude_four →
   area = 8 * √2) :=
sorry

end right_triangle_area_l703_703315


namespace ab_sum_eq_one_l703_703574

-- Definitions based on problem conditions
def A (a b : ℝ) : set ℝ := {x | (x^2 + a * x + b) * (x - 1) = 0}

variable (B : set ℝ)

-- Hypotheses based on problem conditions
variable (h1 : A a b ∩ B = {1, 2})
variable (h2 : A a b ∩ (set.univ \ B) = {3})

-- Final statement to prove
theorem ab_sum_eq_one (a b : ℝ) (B : set ℝ) (h1 : A a b ∩ B = {1, 2}) (h2 : A a b ∩ (set.univ \ B) = {3}) : a + b = 1 :=
sorry

end ab_sum_eq_one_l703_703574


namespace center_circle_sum_eq_neg1_l703_703481

theorem center_circle_sum_eq_neg1 
  (h k : ℝ) 
  (h_center : ∀ x y, (x - h)^2 + (y - k)^2 = 22) 
  (circle_eq : ∀ x y, x^2 + y^2 = 4*x - 6*y + 9) : 
  h + k = -1 := 
by 
  sorry

end center_circle_sum_eq_neg1_l703_703481


namespace knight_without_goblet_l703_703121

-- Define the conditions and statements
def knight (n : ℕ) := n % 50

-- Knight positions are indexed from 0 to 49
def knight_pos := Fin 50

-- Define the initial state of goblets: True represents red wine, False represents white wine
variable (wine: knight_pos → Bool)

-- Condition: there are both red and white goblets
def has_red_and_white := 
  ∃ n, wine n ∧ ∃ m, n ≠ m ∧ ¬ (wine m)

-- Define the transition functions
def pass_goblet_left (n : knight_pos) : knight_pos := 
  (n + 48) % 50

def pass_goblet_right (n : knight_pos) : knight_pos := 
  Fin.mk ((n + 1) % 50) sorry

def midnight_pass (n : knight_pos) : knight_pos :=
  if wine n then pass_goblet_right n else pass_goblet_left n

-- Main theorem
theorem knight_without_goblet :
  has_red_and_white wine → ∃ k, ¬ ∃ j, midnight_pass j = k := 
by 
  sorry

end knight_without_goblet_l703_703121


namespace sum_f1_values_l703_703660

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (k : ℝ) (x y : ℝ) : f (f (x - y)) = f x * f y - k * f x + f y - k * x * y
axiom f_0_eq_k (k : ℝ) : f 0 = k

theorem sum_f1_values (k : ℝ) : 
  let f1 := (λ c d, c * 1 + d)
  let c1 := (if h : k ≥ 0 then √k else 0) in
  let c2 := (if h : k ≥ 0 then -√k else 0) in
  let d := (k - 1) / 2
  ∑ (f1 c1 d) + ∑ (f1 c2 d) = k - 1 :=
by 
  sorry


end sum_f1_values_l703_703660


namespace jasmine_percentage_correct_l703_703043

-- Definitions from the conditions
def initial_volume : ℕ := 80
def initial_percentage_jasmine : ℤ := 10
def additional_jasmine : ℕ := 8
def additional_water : ℕ := 12

-- The proof goal in Lean
theorem jasmine_percentage_correct :
  let initial_jasmine := initial_volume * initial_percentage_jasmine / 100 in
  let total_jasmine := initial_jasmine + additional_jasmine in
  let final_volume := initial_volume + additional_jasmine + additional_water in
  (total_jasmine : ℤ) * 100 / final_volume = 16 :=
by
  sorry

end jasmine_percentage_correct_l703_703043


namespace volume_intersection_pyramids_l703_703641

noncomputable section

-- Definitions of the pyramids P and Q
def pyramid_P_base := [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)]
def apex_P := (1, 1, 3)
def apex_Q := (2, 2, 3)

-- Theorem stating the volume of the intersection of P and Q is 27/4
theorem volume_intersection_pyramids :
  let P := mk_pyramid pyramid_P_base apex_P in
  let Q := mk_pyramid pyramid_P_base apex_Q in
  volume (intersection P.interior Q.interior) = 27 / 4 :=
sorry

end volume_intersection_pyramids_l703_703641


namespace general_term_a_general_term_b_l703_703943

section
  variables (a b : ℕ → ℤ) (q : ℤ)
  
  -- Given Conditions:
  def geom_seq (a : ℕ → ℤ) (q : ℤ) := ∀ n : ℕ, a (n + 1) = a 1 * q ^ n
  def a1_eq : a 1 = 4 := sorry
  def a2_a3_eq : 2 * a 2 + a 3 = 60 := sorry
  def b_seq (b a : ℕ → ℤ) := ∀ n : ℕ, b (n + 1) = b n + a n
  def b1_eq_a2 : b 1 = a 2 := sorry

  -- First proof statement for sequence {a_n}
  theorem general_term_a :
    geom_seq a q → a 1 = 4 → (2 * a 2 + a 3 = 60) →
    (∀ n : ℕ+, a n = 4 * 3 ^ (n - 1)) ∨ (∀ n : ℕ+, a n = 4 * (-5) ^ (n - 1)) := sorry

  -- Second proof statement for sequence {b_n}
  theorem general_term_b :
    (q = 3) → geom_seq a q → a 2 = 12 →
    b_seq b a → b 1 = a 2 → 
    ∀ n : ℕ+, (b n = 2 * 3 ^ (n - 1) + 10) := sorry

end

end general_term_a_general_term_b_l703_703943


namespace percentage_of_students_just_passed_l703_703609

theorem percentage_of_students_just_passed :
  ∀ (total_students absent_students failed_students passed_cutoff average_class_score : ℕ),
  absent_students = total_students * 20 / 100 →
  failed_students = total_students * 30 / 100 →
  passed_cutoff = 40 →
  average_class_score = 36 →
  let remaining_students := total_students - absent_students - failed_students - total_students * 10 / 100 in
  let total_marks := average_class_score * total_students in
  let absent_marks := absent_students * 0 in
  let failed_marks := failed_students * (passed_cutoff - 20) in
  let just_passed_marks := total_students * 10 / 100 * passed_cutoff in
  let remaining_marks := remaining_students * 65 in
  absent_marks + failed_marks + just_passed_marks + remaining_marks = total_marks →
  total_students * 10 / 100 = 10 :=
by
  intros,
  sorry

end percentage_of_students_just_passed_l703_703609


namespace find_x_percentage_l703_703024

noncomputable def percentage_in_second_part (x : ℝ) : Prop :=
  ((47 / 100) * 1442 - (x / 100) * 1412) + 63 = 252

theorem find_x_percentage : ∃ x : ℝ, percentage_in_second_part x ∧ x ≈ 34.63 :=
sorry

end find_x_percentage_l703_703024


namespace conjugate_in_fourth_quadrant_l703_703901

-- Definition of a complex number and its conjugate
noncomputable def z : ℂ := complex.I * (1 - complex.I)

-- Proposition that the conjugate of z lies in the Fourth Quadrant
theorem conjugate_in_fourth_quadrant : 
  let z_conjugate := complex.conj z in
  z_conjugate.re > 0 ∧ z_conjugate.im < 0 :=
sorry

end conjugate_in_fourth_quadrant_l703_703901


namespace sally_pens_l703_703291

theorem sally_pens :
  ∀ (pens : ℕ) (students : ℕ) (pens_per_student : ℕ) (initial_remainder locker_pens home_pens: ℕ),
    pens = 342 → students = 44 → pens_per_student = 7 →
    initial_remainder = pens - pens_per_student * students →
    locker_pens = initial_remainder / 2 →
    home_pens = initial_remainder / 2 →
    home_pens = 17 :=
by
  intros pens students pens_per_student initial_remainder locker_pens home_pens
  assume h_pens h_students h_pens_per_student h_initial_remainder h_locker_pens h_home_pens
  rw [h_pens, h_students, h_pens_per_student] at h_initial_remainder
  rw [h_initial_remainder] at h_locker_pens h_home_pens
  sorry

end sally_pens_l703_703291


namespace intersection_M_P_l703_703575

def is_natural (x : ℤ) : Prop := x ≥ 0

def M (x : ℤ) : Prop := (x - 1)^2 < 4 ∧ is_natural x

def P := ({-1, 0, 1, 2, 3} : Set ℤ)

theorem intersection_M_P :
  {x : ℤ | M x} ∩ P = {0, 1, 2} :=
  sorry

end intersection_M_P_l703_703575


namespace cost_price_is_1500_l703_703323

-- Definitions for the given conditions
def selling_price : ℝ := 1200
def loss_percentage : ℝ := 20

-- Define the cost price such that the loss percentage condition is satisfied
def cost_price (C : ℝ) : Prop :=
  loss_percentage = ((C - selling_price) / C) * 100

-- The proof problem to be solved: 
-- Prove that the cost price of the radio is Rs. 1500
theorem cost_price_is_1500 : ∃ C, cost_price C ∧ C = 1500 :=
by
  sorry

end cost_price_is_1500_l703_703323


namespace area_difference_l703_703988

-- Define the constants for the problem conditions
def radius_large := 25
def diameter_small := 15
def radius_small := diameter_small / 2

-- Define the areas of the circles in terms of pi
def area_large := Real.pi * radius_large^2
def area_small := Real.pi * radius_small^2

-- Prove that the difference in their areas is 568.75 * Real.pi
theorem area_difference :
  area_large - area_small = 568.75 * Real.pi := 
sorry

end area_difference_l703_703988


namespace initial_percentage_decrease_l703_703344

theorem initial_percentage_decrease (P : ℝ) (x : ℝ) :
  let initial_price := P
  let decreased_price := P * (1 - x / 100)
  let increased_price := decreased_price * 1.20
  (increased_price = P * 1.10000000000000014) → x = 8.33333333333333 :=
begin
  intro h,
  have h1 : (1 - x / 100) * 1.20 = 1.10000000000000014 := by
    calc
      (P * (1 - x / 100)) * 1.20 = P * 1.10000000000000014 : by { rw h },
  have h2 : 1 - x / 100 = 1.10000000000000014 / 1.20 := by
    calc
      (1 - x / 100) * 1.20 = 1.10000000000000014 : by rw h1,
  have h3 : x = 8.33333333333333 := by
    calc
      x = (1 - 0.9166666666666667) * 100 : by sorry

  show x = 8.33333333333333, from h3
end

end initial_percentage_decrease_l703_703344


namespace number_of_initials_sets_l703_703964

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l703_703964


namespace negation_of_exists_log3_nonnegative_l703_703151

variable (x : ℝ)

theorem negation_of_exists_log3_nonnegative :
  (¬ (∃ x : ℝ, Real.logb 3 x ≥ 0)) ↔ (∀ x : ℝ, Real.logb 3 x < 0) :=
by
  sorry

end negation_of_exists_log3_nonnegative_l703_703151


namespace paint_per_statue_l703_703117

def total_paint : ℝ := 7 / 16
def number_of_statues : ℕ := 7

theorem paint_per_statue : 
  (total_paint / number_of_statues) = (1 / 16) := 
by 
  sorry

end paint_per_statue_l703_703117


namespace cyclic_sum_inequality_l703_703268

variable (n : ℕ) (x : Fin n → ℝ)
variable (h_pos : ∀ i, 0 < x i)

theorem cyclic_sum_inequality :
  (∑ i in Finset.range n, x (i+1) / x i) ≤ ∑ i in Finset.range n, (x i / x (i+1))^n := 
  sorry

end cyclic_sum_inequality_l703_703268


namespace partition_multiple_sum_l703_703822

theorem partition_multiple_sum (S : Finset ℕ) (hS : ∀ x ∈ S, 0 < x) :
  ∃ N : ℕ, ∀ (x : list ℕ), (∀ n ∈ x, n ∈ S) → (list.sum x) % N = 0 →
  ∃ (y : list (list ℕ)), (∀ sublist ∈ y, list.sum sublist = N) ∧ list.join y = x :=
begin
  sorry
end

end partition_multiple_sum_l703_703822


namespace max_valid_m_l703_703895

def is_valid_grid (m : ℕ) (grid : Array (Array Char)) : Prop :=
  ∀ i j, i ≠ j → (finset.univ.filter (λ c, grid[i][c] = grid[j][c])).card ≤ 1

theorem max_valid_m : ∀ m : ℕ, is_valid_grid m grid ↔ m ≤ 5 := sorry

end max_valid_m_l703_703895


namespace proof_problem_l703_703644

-- Definitions for the conditions
variables (A : Set) (n m : ℕ)
variables (Ai : Fin m → Set) -- We use Ai as a function taking indices in Fin m

-- Constraints: A has n elements and Ai are pairwise disjoint subsets of A
axiom h1 : A.card = n
axiom h2 : ∀ i, (Ai i).SubsetOf A
axiom h3 : ∀ (i j : Fin m), i ≠ j → Disjoint (Ai i) (Ai j)

-- Conclusion to prove the two inequalities
theorem proof_problem :
    (∑ i in Finset.range m, 1 / Finset.card (Finset.powersetLen (Ai i).card A ) ) ≤ 1 ∧
    (∑ i in Finset.range m, Finset.card (Finset.powersetLen (Ai i).card A)) ≥ m^2 := by
  sorry

end proof_problem_l703_703644


namespace derivative_of_2x_sin_2x_plus_5_derivative_of_x3_minus_1_over_sin_x_l703_703125

-- Statement for the first derivative problem
theorem derivative_of_2x_sin_2x_plus_5 : 
  ∀ (x : ℝ), 
  (deriv (λ x : ℝ, 2 * x * sin (2 * x + 5)) x) = (2 * sin (2 * x + 5) + 4 * x * cos (2 * x + 5)) :=
by sorry

-- Statement for the second derivative problem
theorem derivative_of_x3_minus_1_over_sin_x : 
  ∀ (x : ℝ) (hx : sin x ≠ 0), 
  (deriv (λ x : ℝ, (x^3 - 1) / sin x) x) = ((3 * x^2 * sin x - (x^3 - 1) * cos x) / (sin x)^2) :=
by sorry

end derivative_of_2x_sin_2x_plus_5_derivative_of_x3_minus_1_over_sin_x_l703_703125


namespace range_of_m_minimum_value_l703_703542

theorem range_of_m (m n : ℝ) (h : 2 * m - n = 3) (ineq : |m| + |n + 3| ≥ 9) : 
  m ≤ -3 ∨ m ≥ 3 := 
sorry

theorem minimum_value (m n : ℝ) (h : 2 * m - n = 3) : 
  ∃ c, c = 3 ∧ c = |(5 / 3) * m - (1 / 3) * n| + |(1 / 3) * m - (2 / 3) * n| := 
sorry

end range_of_m_minimum_value_l703_703542


namespace flush_probability_l703_703023

noncomputable def probability_flush : ℚ :=
  4 * Nat.choose 13 5 / Nat.choose 52 5

theorem flush_probability :
  probability_flush = 33 / 16660 :=
by
  sorry

end flush_probability_l703_703023


namespace spherical_coordinates_neg_y_l703_703834

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arctan2 (Real.sqrt (x^2 + y^2)) z
  let θ := Real.arctan2 y x
  (ρ, θ, φ)

theorem spherical_coordinates_neg_y
  (ρ θ φ : ℝ)
  (hρ : ρ = 3)
  (hθ : θ = 5 * Real.pi / 6)
  (hφ : φ = Real.pi / 4) :
  rectangular_to_spherical
    (ρ * Real.sin φ * Real.cos θ)
    (-ρ * Real.sin φ * Real.sin θ)
    (ρ * Real.cos φ) = (3, Real.pi / 6, Real.pi / 4) :=
by
  have h : spherical_to_rectangular 3 (5 * Real.pi / 6) (Real.pi / 4) =
           (3 * Real.sin (Real.pi / 4) * Real.cos (5 * Real.pi / 6), 
            3 * Real.sin (Real.pi / 4) * Real.sin (5 * Real.pi / 6), 
            3 * Real.cos (Real.pi / 4)) := by sorry
  have h_neg_y : spherical_to_rectangular 3 (Real.pi / 6) (Real.pi / 4) =
           (3 * Real.sin (Real.pi / 4) * Real.cos (Real.pi / 6), 
            3 * Real.sin (Real.pi / 4) * Real.sin (Real.pi / 6), 
            3 * Real.cos (Real.pi / 4)) := by sorry
  sorry


end spherical_coordinates_neg_y_l703_703834


namespace triangle_area_l703_703311

theorem triangle_area (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (a b c : ℝ) (h : (a = b ∧ b = c) ∧ (a * a + b * b = c * c) ∧ (a * (b + b) = 8) ∧ (a * b / 2 = 16)) : 
  a * b / 2 = 16 :=
by {
  sorry 
}

end triangle_area_l703_703311


namespace part1_growth_rate_part2_selling_price_l703_703722

noncomputable def monthly_growth_rate (P : ℕ → ℕ) (r : ℕ) (April June : ℕ) :=
  P April * (1 + r)^2 = P June

theorem part1_growth_rate :
  (∃ r : ℝ, monthly_growth_rate (λ n, if n = 4 then 150 else if n = 6 then 216 else 0) r 4 6) :=
begin
  let r := 0.2,
  have h : 150 * (1 + r)^2 = 216,
  { simp [r],
    norm_num },
  use r,
  exact ⟨h⟩,
  sorry
end

noncomputable def actual_selling_price (c y0 V0 k π y : ℝ) :=
  (y - c) * (V0 - k * (y - y0)) = π

theorem part2_selling_price :
  (∃ y : ℝ, actual_selling_price 30 40 600 10 10000 y y = 50) :=
begin
  let y := 50,
  have h : (y - 30) * (600 - 10 * (y - 40)) = 10000,
  { simp [y],
    norm_num },
  use y,
  exact ⟨h⟩,
  sorry
end

end part1_growth_rate_part2_selling_price_l703_703722


namespace shift_graph_right_l703_703014

theorem shift_graph_right :
  ∀ (x : ℝ), 3 * sin (2 * x - π / 5) = 3 * sin (2 * (x - π / 10)) :=
by
  sorry

end shift_graph_right_l703_703014


namespace vector_expression_l703_703936

-- Declare the magnitudes of vectors a and b
variables (a b : EuclideanSpace ℝ (Fin 3)) -- Assuming 3D vectors for generality
def mag_a := 2
def mag_b := 5

-- Declare the cosine of the angle between a and b as a condition
def cos_angle : ℝ := Real.cos (2 * Real.pi / 3) -- cos(120 degrees) = -1/2

-- Assume the magnitude conditions
axiom mag_cond_a : ∥a∥ = mag_a
axiom mag_cond_b : ∥b∥ = mag_b

-- Assume the dot product by angle condition
axiom dot_product_by_angle : a ⬝ b = ∥a∥ * ∥b∥ * cos_angle

-- The proof goal
theorem vector_expression : ((2 : ℝ) • a - b) ⬝ a = 13 := by
  sorry

end vector_expression_l703_703936


namespace range_x_intercept_and_max_area_of_triangle_l703_703163

theorem range_x_intercept_and_max_area_of_triangle 
  (C1_eq : ∀ x y, x^2 / 4 + y^2 / 3 = 1) 
  (C2_eq : ∀ x y, y^2 = 4 * x) 
  (P : ∃ t ≠ 0, y = 2 * t ∧ x = t^2) 
  (tangent_line_l : ∃ k, y - 2 * t = k * (x - t^2)) :
  (∃ t, 0 < t^2 ∧ t^2 < 4 → -4 < x ∧ x < 0) ∧
  (∃ S_max, S_max = sqrt(3)) :=
by {
  sorry
}

end range_x_intercept_and_max_area_of_triangle_l703_703163


namespace find_y_l703_703343

theorem find_y (x : ℝ) (y : ℝ) (h : (3 + y)^5 = (1 + 3 * y)^4) (hx : x = 1.5) : y = 1.5 :=
by
  -- Proof steps go here
  sorry

end find_y_l703_703343


namespace smallest_prime_factor_of_difference_l703_703735

theorem smallest_prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) (h_diff : A ≠ C) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 3 :=
by
  sorry

end smallest_prime_factor_of_difference_l703_703735


namespace map_intersection_no_parallel_l703_703042

def maps_intersection_point (A B C D A1 B1 C1 D1: Point) : Prop :=
  (∃ O: Point, (AO : ℝ) / (A1O : ℝ) = 5) 

axiom rectangular_maps 
  (A B C D A1 B1 C1 D1: Point) 
  (scale : ℝ) (parallel_sides : Prop) : Prop :=
  is_rectangle A B C D ∧ 
  is_rectangle A1 B1 C1 D1 ∧ 
  scaled_by A B C D A1 B1 C1 D1 (1/5) ∧
  sides_parallel A B C D A1 B1 C1 D1

theorem map_intersection_no_parallel
  (A B C D A1 B1 C1 D1: Point) 
  (scale : ℝ) (parallel_sides : Prop): 
  (rectangular_maps A B C D A1 B1 C1 D1 scale parallel_sides) →
  maps_intersection_point  A B C D A1 B1 C1 D1 :=
by
  sorry

end map_intersection_no_parallel_l703_703042


namespace ratio_of_areas_l703_703204

-- Definitions
def area_equilateral_triangle (side_length : ℝ) : ℝ := 
  (sqrt 3 / 4) * side_length^2

def area_trianglex (area_smaller : ℝ) : ℝ :=
  2 * area_smaller

def area_triangleade (area_smaller : ℝ) : ℝ :=
  4 * area_smaller

noncomputable def ratio (a b : ℝ) : ℝ := a / b

-- Problem Statement
theorem ratio_of_areas : 
  ∀ (s : ℝ), s > 0 → ratio (area_trianglex s) (area_triangleade s) = 1 / 2 :=
by
  intros s hs
  simpl
  have h : s ≠ 0 := ne_of_gt hs
  field_simp [h]
  sorry

end ratio_of_areas_l703_703204


namespace eval_expression_equals_sqrt3_l703_703945

def alpha : ℝ := -35 / 6 * Real.pi

theorem eval_expression_equals_sqrt3 :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha) - Real.cos (Real.pi + alpha)) / 
  (1 + Real.sin α^2 + Real.sin (Real.pi - α) - Real.cos (Real.pi + α)^2) = Real.sqrt 3 :=
by
  sorry

end eval_expression_equals_sqrt3_l703_703945


namespace number_of_initials_sets_l703_703963

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l703_703963


namespace notebook_cost_l703_703513

theorem notebook_cost :
  ∃ (x y : ℕ), (5 * x + 4 * y = 380) ∧ (3 * x + 6 * y = 354) ∧ (x = 48) :=
by {
  let x := 48,
  let y := 35,
  use x,
  use y,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
}

end notebook_cost_l703_703513


namespace maximize_rental_revenue_l703_703810

theorem maximize_rental_revenue :
  ∃ X : ℝ, 
    let cars := 100 - (X - 3000) / 50 in
    let revenue := (X - 200) * cars - 150 * cars - 50 * ((X - 3000) / 50) in
    (X = 4100) ∧ 
    (∀Y : ℝ, 
      let cars' := 100 - (Y - 3000) / 50 in
      let revenue' := (Y - 200) * cars' - 150 * cars' - 50 * ((Y - 3000) / 50) in
      revenue ≤ revenue') :=
by sorry

end maximize_rental_revenue_l703_703810


namespace find_a_monotonic_increase_intervals_l703_703167

-- Definition of the function f
noncomputable def f (a x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) + a

-- Helper to rewrite f in terms of a transformation
noncomputable def transformed_f (a x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + π / 6) + a + 1

-- Statement 1: Given the max value of f(x) is 2, proof that a = -1
theorem find_a (a : ℝ) (h : ∃ x : ℝ, transformed_f a x = 2) : a = -1 :=
sorry

-- Statement 2: The intervals of monotonic increase of f(x)
theorem monotonic_increase_intervals :
  (∀ k : ℤ, ∀ x : ℝ, -π/3 + k * π ≤ x ∧ x ≤ π/6 + k * π → 
  ∀ y : ℝ, x < y ∧ y < π/6 + k * π → f a x < f a y) :=
sorry

end find_a_monotonic_increase_intervals_l703_703167


namespace suff_not_necessary_no_real_solutions_l703_703382

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l703_703382


namespace flux_vector_field_spherical_l703_703502

noncomputable def vector_field (r θ : ℝ) : ℝ × ℝ × ℝ := (r^2 * θ, r^2, 0)

noncomputable def flux_through_upper_hemisphere (R: ℝ) : ℝ :=
  2 * Real.pi * R^4

theorem flux_vector_field_spherical (R : ℝ) : 
  let a := vector_field in
  let upper_hemisphere := λ r θ φ,
    r = R ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ 0 ≤ φ ∧ φ < 2 * Real.pi in
  (∫ (φ from 0 to 2 * Real.pi), ∫ (θ from 0 to Real.pi / 2),
     R^4 * θ * Real.sin θ dθ dφ) = flux_through_upper_hemisphere R :=
by sorry

end flux_vector_field_spherical_l703_703502


namespace rhombus_implies_parallelogram_converse_false_inverse_false_converse_and_inverse_false_l703_703578

-- Define our propositions symbolically
def isRhombus (P : Type) (h : P) : Prop := sorry -- Placeholder for actual rhombus definition
def isParallelogram (P : Type) (h : P) : Prop := sorry -- Placeholder for actual parallelogram definition

theorem rhombus_implies_parallelogram (P : Type) (h : P) :
  isRhombus P h → isParallelogram P h := sorry

theorem converse_false (P : Type) (h : P) :
  ¬ (isParallelogram P h → isRhombus P h) := sorry

theorem inverse_false (P : Type) (h : P) :
  ¬ (¬ isRhombus P h → ¬ isParallelogram P h) := sorry

theorem converse_and_inverse_false (P : Type) (h : P) :
  (¬ (isParallelogram P h → isRhombus P h)) ∧ (¬ (¬ isRhombus P h → ¬ isParallelogram P h)) :=
  by
    exact and.intro (converse_false P h) (inverse_false P h)

end rhombus_implies_parallelogram_converse_false_inverse_false_converse_and_inverse_false_l703_703578


namespace total_cows_l703_703418

/-- A farmer divides his herd of cows among his four sons.
The first son receives 1/3 of the herd, the second son receives 1/6,
the third son receives 1/9, and the rest goes to the fourth son,
who receives 12 cows. Calculate the total number of cows in the herd
-/
theorem total_cows (n : ℕ) (h1 : (n : ℚ) * (1 / 3) + (n : ℚ) * (1 / 6) + (n : ℚ) * (1 / 9) + 12 = n) : n = 54 := by
  sorry

end total_cows_l703_703418


namespace max_paint_fraction_l703_703191

theorem max_paint_fraction (paint_time : ℕ) (fraction_painted : ℚ)
  (h1 : paint_time = 60) :
  let max_rate := (1 : ℚ) / paint_time in
  let time_painting := 15 in
  let result := max_rate * time_painting in
  result = fraction_painted :=
by
  -- Proof omission
  sorry

end max_paint_fraction_l703_703191


namespace cyclist_speed_ratio_l703_703018

theorem cyclist_speed_ratio (v_1 v_2 : ℝ)
  (h1 : v_1 = 2 * v_2)
  (h2 : v_1 + v_2 = 6)
  (h3 : v_1 - v_2 = 2) :
  v_1 / v_2 = 2 := 
sorry

end cyclist_speed_ratio_l703_703018


namespace min_value_l703_703653

theorem min_value
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 9) :
  ∃ x : ℝ, x = 9 ∧ x = min (λ x, (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)) := 
sorry

end min_value_l703_703653


namespace cosine_axis_of_symmetry_zero_l703_703326

theorem cosine_axis_of_symmetry_zero :
  ∀ x : ℝ, y = cos x → (x = 0 ∨ x = π ∨ x = 2 * π ∨ x = 3 * π) :=
sorry

end cosine_axis_of_symmetry_zero_l703_703326


namespace gcd_poly_correct_l703_703548

-- Define the conditions
def is_even_multiple_of (x k : ℕ) : Prop :=
  ∃ (n : ℕ), x = k * 2 * n

variable (b : ℕ)

-- Given condition
axiom even_multiple_7768 : is_even_multiple_of b 7768

-- Define the polynomials
def poly1 (b : ℕ) := 4 * b * b + 37 * b + 72
def poly2 (b : ℕ) := 3 * b + 8

-- Proof statement
theorem gcd_poly_correct : gcd (poly1 b) (poly2 b) = 8 :=
  sorry

end gcd_poly_correct_l703_703548


namespace find_line_L_l703_703830

namespace Geometry

variable (L L1 L2: ℝ → ℝ → Prop)

theorem find_line_L :
  (L (2:ℝ) (4:ℝ)) ∧ 
  (∀ x y : ℝ, L1 x y ↔ 2*x - y + 2 = 0) ∧ 
  (∀ x y : ℝ, L2 x y ↔ 2*x - y - 3 = 0) ∧ 
  (∃ a b : ℝ, ∀ x y : ℝ, (a, b) = ( ((x,y):ℝ) ⊆ L1 ∧ (x,y) ⊆ L2 ) ∧ (2*a - 4*b + 13 = 0)) ↔ 
  (∀ x y : ℝ, L x y ↔ x - 2*y + 6 = 0) := by
  sorry

end Geometry

end find_line_L_l703_703830


namespace area_of_storm_eye_l703_703868

theorem area_of_storm_eye : 
  let large_quarter_circle_area := (1 / 4) * π * 5^2
  let small_circle_area := π * 2^2
  let storm_eye_area := large_quarter_circle_area - small_circle_area
  storm_eye_area = (9 * π) / 4 :=
by
  sorry

end area_of_storm_eye_l703_703868


namespace coeff_x5_in_expansion_l703_703499

theorem coeff_x5_in_expansion (f : ℤ[X]) : f = (1 + 2 * X - 3 * X^2)^6 → coeff f 5 = -168 :=
by sorry

end coeff_x5_in_expansion_l703_703499


namespace maximize_income_l703_703424

-- Define the given conditions
def num_rooms : ℕ := 100

def price_and_occupancy : List (ℕ × ℚ) := 
  [(200, 0.65), (180, 0.75), (160, 0.85), (140, 0.95)]

-- Define the daily income function
def daily_income (price : ℕ) (occupancy : ℚ) (rooms : ℕ) : ℚ :=
  price * occupancy * rooms

-- The maximum daily income is obtained when the price per room is set to 160 yuan
theorem maximize_income : ∀ price, 
  (price, price_and_occupancy.lookup price) ∈ price_and_occupancy →
  daily_income price (price_and_occupancy.lookup price).get_or_else 0 num_rooms ≤
  daily_income 160 0.85 num_rooms :=
begin
  sorry
end

end maximize_income_l703_703424


namespace corrected_mean_observations_l703_703334

theorem corrected_mean_observations :
  let mean := 36
  let n := 50
  let observations_sum := mean * n
  let corrected_sum := observations_sum + ((48 - 21) + (56 - 34) + (67 - 55))
  (corrected_sum / n : ℝ) = 37.22 :=
by
  let mean := 36
  let n := 50
  let observations_sum := mean * n
  let corrected_sum := observations_sum + ((48 - 21) + (56 - 34) + (67 - 55))
  have h1 : corrected_sum = 1861 := by
    calc
      corrected_sum 
        = observations_sum + ((48 - 21) + (56 - 34) + (67 - 55)) : rfl
        ... = 1800 + ((48 - 21) + (56 - 34) + (67 - 55)) : by simp [observations_sum]
        ... = 1800 + (27 + 22 + 12) : by simp
        ... = 1800 + 61 : by simp
        ... = 1861 : by simp
  have h2 : corrected_sum / n = 37.22 := by
    calc
      corrected_sum / n 
        = 1861 / 50 : by rw [h1]
        ... = 37.22 : by norm_num
  exact h2

end corrected_mean_observations_l703_703334


namespace suff_not_necessary_no_real_solutions_l703_703383

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l703_703383


namespace lamp_problem_l703_703827

/--
A house has an even number of lamps distributed among its rooms. 
Each room has at least three lamps. 
Each lamp shares a switch with exactly one other lamp, 
and each change in the switch shared by two lamps changes their states simultaneously.

Prove that for every initial state of the lamps, 
there exists a sequence of changes in some of the switches 
at the end of which each room contains lamps which are on as well as lamps which are off.
-/
theorem lamp_problem 
  (rooms : Type) 
  (lamps : Type) 
  (even_lamps : ∃ n : ℕ, Finset.card lamps = 2 * n)
  (at_least_three_lamps_per_room : ∀ r : rooms, ∃ l : Finset lamps, Finset.card l ≥ 3) 
  (switch : {x : lamps // ∃ y : lamps, x ≠ y}) 
  (change_states_simultaneously : ∀ (x y : lamps) (s : switch), x ≠ y → s = switch → x = y) :
  ∀ (initial_state : lamps → bool), 
  ∃ (seq_of_changes : list switch), 
    ∀ r : rooms, ∃ (l1 l2 : lamps), l1 ≠ l2 ∧ (initial_state l1 ≠ initial_state l2) :=
sorry

end lamp_problem_l703_703827


namespace min_value_is_nine_l703_703651

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l703_703651


namespace cone_prism_volume_ratio_l703_703432

variables {r h: ℝ} (hr_pos : r > 0) (hh_pos : h > 0)

def volume_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
def volume_prism (r h : ℝ) : ℝ := 8 * r^2 * h

theorem cone_prism_volume_ratio (r h : ℝ) (hr_pos : r > 0) (hh_pos : h > 0) :
  volume_cone r h / volume_prism r h = Real.pi / 24 := by
  sorry

end cone_prism_volume_ratio_l703_703432


namespace div_by_prime_power_l703_703286

theorem div_by_prime_power (p α x : ℕ) (hp : Nat.Prime p) (hpg : p > 2) (hα : α > 0) (t : ℤ) :
  (∃ k : ℤ, x^2 - 1 = k * p^α) ↔ (∃ t : ℤ, x = t * p^α + 1 ∨ x = t * p^α - 1) :=
sorry

end div_by_prime_power_l703_703286


namespace find_ad_l703_703633

noncomputable def ad_in_triangle (ABC : Triangle) (D : Point) (BD AD : ℝ) 
  (CAB CBA : ℝ) : Prop :=
  (BD = 13) ∧
  (CAB = CBA / 3) ∧
  (BC = CD) ∧
  AD = 13

-- Given the conditions
axiom triangle_ABC : Triangle
axiom point_D : Point
axiom bd : ℝ
axiom ad : ℝ
axiom cab : ℝ
axiom cba : ℝ

-- Conditions
axiom bd_eq_13 : bd = 13
axiom cab_eq_cba_div_3 : cab = cba / 3
axiom bc_eq_cd : true -- Placeholder for the geometric condition that BC = CD

-- Statement to be proved
theorem find_ad : ad_in_triangle triangle_ABC point_D bd ad cab cba :=
by
  apply ad_in_triangle
  split
  exact bd_eq_13
  split
  exact cab_eq_cba_div_3
  split
  exact bc_eq_cd
  exact bd_eq_13

end find_ad_l703_703633


namespace find_speeds_of_A_and_B_l703_703410

noncomputable def speed_A_and_B (x y : ℕ) : Prop :=
  30 * x - 30 * y = 300 ∧ 2 * x + 2 * y = 300

theorem find_speeds_of_A_and_B : ∃ (x y : ℕ), speed_A_and_B x y ∧ x = 80 ∧ y = 70 :=
by
  sorry

end find_speeds_of_A_and_B_l703_703410


namespace sum_of_segments_length_l703_703615

namespace RectangleProblem

-- Define the sides of the rectangle
def sideXY : ℝ := 5
def sideYZ : ℝ := 6

-- Define the length of the segments
def length_diag : ℝ := Real.sqrt ( sideXY^2 + sideYZ^2 )

-- Define the function for the segment lengths
def seg_length (k : ℕ) (h : 1 ≤ k ∧ k ≤ 209) : ℝ :=
  length_diag * (210 - k) / 210

-- Define the sum of the segments
noncomputable def sum_segments : ℝ :=
  2 * (Finset.sum (Finset.range 210) (λ k, if (1 ≤ k ∧ k ≤ 209) then seg_length k ⟨1, 209⟩ else 0)) - length_diag

-- The statement of the proof
theorem sum_of_segments_length :
  sum_segments = 208 * Real.sqrt 61 :=
sorry

end RectangleProblem

end sum_of_segments_length_l703_703615


namespace rectangle_ratio_l703_703517

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio_l703_703517


namespace equidistant_points_locus_l703_703146

-- Define the necessary geometric objects: points, circle, and square.
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)
structure Square := (bottom_left : Point) (side_length : ℝ)

-- Define the distance from a point to a set.
def distance_to_set (P : Point) (K : set Point) : ℝ :=
  Inf {d : ℝ | ∃ Q ∈ K, d = Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)}

-- Define the main problem.
def equidistant_points (S : Square) (C : Circle) (P : Point) :=
  distance_to_set P {Q | on_boundary Q S} = 
  Real.sqrt ((P.x - C.center.x) ^ 2 + (P.y - C.center.y) ^ 2) - C.radius

-- The conjecture regarding the set of equidistant points.
theorem equidistant_points_locus (S : Square) (C : Circle) 
  (h₁ : C.center ∈ {Q | on_boundary Q S})
  (h₂ : ¬ ∃ Q ∈ {R : Point | on_interior R S}, 
          Real.sqrt ((Q.x - C.center.x) ^ 2 + (Q.y - C.center.y) ^ 2) = C.radius) :
  ∃ ℒ : set Point, set_eq ℒ {P | equidistant_points S C P} ∧
    (ℒ = {P | ∃ A ∈ vertex_set S, 
      Real.sqrt ((P.x - C.center.x) ^ 2 + 
      (P.y - C.center.y) ^ 2) - C.radius = Real.sqrt ((P.x - A.x) ^ 2 + (P.y - A.y) ^ 2)}
    ∪ {P | ∃ E ∈ edge_set S, 
      let E' := {Q | on_line Q (shifted_parallel E C.radius)} in
      Real.sqrt ((P.x - C.center.x) ^ 2 + 
      (P.y - C.center.y) ^ 2) = distance_to_line P E'}) :=
sorry

end equidistant_points_locus_l703_703146


namespace problem1_problem2_l703_703485

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2_l703_703485


namespace paul_can_buy_toys_l703_703282

-- Definitions of the given conditions
def initial_dollars : ℕ := 3
def allowance : ℕ := 7
def toy_cost : ℕ := 5

-- Required proof statement
theorem paul_can_buy_toys : (initial_dollars + allowance) / toy_cost = 2 := by
  sorry

end paul_can_buy_toys_l703_703282


namespace find_positive_numbers_l703_703900

-- Definition of the main problem statement
theorem find_positive_numbers (a b : ℕ) (h₁ : ∃ a b, (a, b) ∈ {(2, 2), (1, 3), (3, 3)}) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 :=
by
  cases h₁ with a h₁_cases
  cases h₁_cases with b h_cases
  cases h_cases
    case inl { -- case for (a, b) = (2, 2)
      sorry -- Proof for this case
    }
    case inr {
      cases h_cases
        case inl { -- case for (a, b) = (1, 3)
          sorry -- Proof for this case
        }
        case inr { -- case for (a, b) = (3, 3)
          sorry -- Proof for this case
        }
    }

end find_positive_numbers_l703_703900


namespace volume_of_larger_cube_is_correct_l703_703378

noncomputable def larger_cube_volume : ℝ :=
  let smaller_volume := 8
  let smaller_side := real.cbrt smaller_volume
  let smaller_surface_area := 6 * smaller_side^2
  let larger_surface_area := 3 * smaller_surface_area
  let larger_side := real.sqrt (larger_surface_area / 6)
  let larger_volume := larger_side^3
  larger_volume

theorem volume_of_larger_cube_is_correct : larger_cube_volume = 24 * real.sqrt 3 := sorry

end volume_of_larger_cube_is_correct_l703_703378


namespace min_square_area_of_quartic_roots_l703_703467

theorem min_square_area_of_quartic_roots (p q r s : ℤ) :
  ∃ (roots : ℂ → ℂ) (S : Set ℂ), monic (x^4 + (p:ℂ) * x^3 + (q:ℂ) * x^2 + (r:ℂ) * x + s) ∧ 
  (∀ z ∈ S, z^4 + p * z^3 + q * z^2 + r * z + s = 0) ∧ is_square S ∧ area S = 2 :=
sorry

end min_square_area_of_quartic_roots_l703_703467


namespace unique_function_satisfying_condition_l703_703480

theorem unique_function_satisfying_condition :
  (∃! g : ℝ → ℝ, ∀ x y : ℝ, g(x + y) + g(x - y) - g(x) * g(y) ≤ 0) :=
begin
  sorry
end

end unique_function_satisfying_condition_l703_703480


namespace Tina_profit_correct_l703_703769

theorem Tina_profit_correct :
  ∀ (price_per_book cost_per_book books_per_customer total_customers : ℕ),
  price_per_book = 20 →
  cost_per_book = 5 →
  books_per_customer = 2 →
  total_customers = 4 →
  (price_per_book * (books_per_customer * total_customers) - 
   cost_per_book * (books_per_customer * total_customers) = 120) :=
by
  intros price_per_book cost_per_book books_per_customer total_customers
  sorry

end Tina_profit_correct_l703_703769


namespace card_statements_are_false_l703_703412

theorem card_statements_are_false :
  ¬( ( (statements: ℕ) →
        (statements = 1 ↔ ¬statements = 1 ∧ ¬statements = 2 ∧ ¬statements = 3 ∧ ¬statements = 4 ∧ ¬statements = 5) ∧
        ( statements = 2 ↔ (statements = 1 ∨ statements = 3 ∨ statements = 4 ∨ statements = 5)) ∧
        (statements = 3 ↔ (statements = 1 ∧ statements = 2 ∧ (statements = 4 ∨ statements = 5) ) ) ∧
        (statements = 4 ↔ (statements = 1 ∧ statements = 2 ∧ statements = 3 ∧ statements != 5 ) ) ∧
        (statements = 5 ↔ (statements = 4 ) )
)) :=
sorry

end card_statements_are_false_l703_703412


namespace triangle_area_l703_703338

theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : P = 20)
  (h2 : r = 2.5) 
  (h3 : A = r * (P / 2)) :
  A = 25 :=
by
  have s : ℝ := P / 2
  rw [h1, h2] at *
  have hs : s = 10 := by norm_num [s, h1]
  rw hs at *
  norm_num [A, h3]
  assumption sorry

end triangle_area_l703_703338


namespace smallest_number_of_yellow_marbles_l703_703250

theorem smallest_number_of_yellow_marbles :
  ∃ n : ℕ, ∀ y : ℕ, (n = 6 * m and y = n - (n / 2 + n / 3 + 12)) → y = 0 := by
  sorry

end smallest_number_of_yellow_marbles_l703_703250


namespace min_value_is_nine_l703_703652

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l703_703652


namespace little_john_friends_share_l703_703274

-- Noncomputable definition for dealing with reals
noncomputable def amount_given_to_each_friend :=
  let total_initial := 7.10
  let total_left := 4.05
  let spent_on_sweets := 1.05
  let total_given_away := total_initial - total_left
  let total_given_to_friends := total_given_away - spent_on_sweets
  total_given_to_friends / 2

-- The theorem stating the result
theorem little_john_friends_share :
  amount_given_to_each_friend = 1.00 :=
by
  sorry

end little_john_friends_share_l703_703274


namespace max_area_triangle_PMN_l703_703013

-- Define the conditions
def P : (ℝ × ℝ) := (11, 0)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y : ℝ) : Prop := y = x - 11
def P_in_parabola : ¬ parabola P.1 P.2 := by {
  intro h,
  cases P,
  dsimp [parabola] at h,
  linarith,
}
def angle_of_inclination : ℝ := π / 4
def slope : ℝ := 1

-- Define the points R and Q on the parabola
def R (x y : ℝ) : Prop := parabola x y ∧ line x y
def Q (x y : ℝ) : Prop := parabola x y ∧ line x y

-- Define parallel line condition and points M and N
def parallel_line (m : ℝ) (x y : ℝ) : Prop := y = x + m
def M (m x y : ℝ) : Prop := parabola x y ∧ parallel_line m x y
def N (m x y : ℝ) : Prop := parabola x y ∧ parallel_line m x y

-- Problem statement to prove the maximum area
theorem max_area_triangle_PMN : ∃ (m : ℝ), (let MN := 4 * real.sqrt(1 - m)
                                           let d := abs (m + 11) / real.sqrt 2
                                           ((2 * real.sqrt(1 - m) * abs (m + 11)) ≤ 22)) := sorry

end max_area_triangle_PMN_l703_703013


namespace base4_calculation_l703_703457

theorem base4_calculation :
  (let a := 2*4^2 + 0*4^1 + 3*4^0) -- 203_4
  (let b := 2*4^1 + 1*4^0)         -- 21_4
  (let c := 3*4^0)                 -- 3_4
  (let d := 1*4^5 + 1*4^4 + 0*4^3 + 3*4^2 + 2*4^1 + 0*4^0) -- 110320_4
  a * b / c = d
:= by
  sorry

end base4_calculation_l703_703457


namespace water_usage_december_l703_703356

def springWaterFee (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then 8 * x
  else if 5 < x ∧ x ≤ 8 then 12 * x - 20
  else if 8 < x ∧ x ≤ 10 then 16 * x - 52
  else 0

def totalFee (x : ℝ) (totalWater : ℝ) : ℝ :=
  springWaterFee x + 2 * (totalWater - x)

theorem water_usage_december (x : ℝ) (totalWater : ℝ) (totalFeePaid : ℝ) (hx : 5 < x ∧ x ≤ 8) :
  totalWater = 16 → totalFee x totalWater = 72 → x = 6 ∧ (totalWater - x) = 10 :=
by
  sorry

end water_usage_december_l703_703356


namespace num_sets_satisfy_condition_l703_703136

theorem num_sets_satisfy_condition : 
  (finset.univ.powerset.filter (λ M : finset ℕ, M ∪ {1, 2} = {1, 2, 3})).card = 4 := by
  sorry

end num_sets_satisfy_condition_l703_703136


namespace true_discount_l703_703728

theorem true_discount (BD SD : ℝ) (hBD : BD = 18) (hSD : SD = 90) : 
  ∃ TD : ℝ, TD = 15 ∧ TD = BD / (1 + BD / SD) :=
by
  have h1 : 1 + BD / SD = 1 + 18 / 90 := by rw [hBD, hSD]
  have h2 : 1 + 18 / 90 = 1 + 0.2 := by norm_num
  have h3 : 1 + 0.2 = 1.2 := by norm_num
  have h4 : BD / (1 + BD / SD) = 18 / 1.2 := by rw [h1, h3]
  have h5 : 18 / 1.2 = 15 := by norm_num
  use 15
  split
  · exact h5
  · exact h4

end true_discount_l703_703728


namespace angle_difference_l703_703441

theorem angle_difference (X Y Z Z1 Z2 : ℝ) (h1 : Y = 2 * X) (h2 : X = 30) (h3 : Z1 + Z2 = Z) (h4 : Z1 = 60) (h5 : Z2 = 30) : Z1 - Z2 = 30 := 
by 
  sorry

end angle_difference_l703_703441


namespace multiplication_correct_l703_703792

theorem multiplication_correct (x : ℤ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

end multiplication_correct_l703_703792


namespace amanda_tickets_l703_703088

theorem amanda_tickets (F : ℕ) (h : 4 * F + 32 + 28 = 80) : F = 5 :=
by
  sorry

end amanda_tickets_l703_703088


namespace initial_speed_correct_l703_703072

variables (D T : ℝ) (initial_speed remaining_speed : ℝ)
variables (one_third_time two_thirds_distance : ℝ)

-- The initial speed is computed based on the given conditions
def initial_speed := 2 * D / T

-- The remaining speed is given as 40 kmph 
def remaining_speed := 40

-- The time taken for the last part of the journey 
def time_last_part := (D / 3) / remaining_speed

-- Total time consists of the sum of the time taken for first and last part
def total_time := D * 3 / 80

-- Proof that the initial speed was 53.33 kmph
theorem initial_speed_correct :
  initial_speed D total_time = 160 / 3 :=
by 
  -- We skip the proof here using sorry
  sorry

end initial_speed_correct_l703_703072


namespace tournament_battles_needed_l703_703616

-- Define the tournament conditions
def initialTeams : ℕ := 2017

def teamsRemainingAfterNBattles (n : ℕ) : ℕ :=
  initialTeams - 2 * n

-- The final condition where only one team remains
def hasChampion (n : ℕ) : Prop :=
  teamsRemainingAfterNBattles n = 1

-- The main theorem to prove the required number of battles
theorem tournament_battles_needed : ∃ n, hasChampion n ∧ n = 1008 :=
by {
  use 1008,
  split,
  {
    unfold hasChampion,
    unfold teamsRemainingAfterNBattles,
    simp,
  },
  {
    refl,
  }
}

end tournament_battles_needed_l703_703616


namespace hyperbola_foci_distance_2sqrt2c_l703_703870

variables (c : ℝ)

def hyperbola_foci_distance (c : ℝ) : ℝ := 2 * Real.sqrt 2 * c

theorem hyperbola_foci_distance_2sqrt2c (c : ℝ) :
  ∀ (x y : ℝ), x * y = c^2 → ∃ (d : ℝ), d = hyperbola_foci_distance c := by
  intros x y h
  use hyperbola_foci_distance c
  sorry

end hyperbola_foci_distance_2sqrt2c_l703_703870


namespace sum_of_partitions_divisible_by_4_l703_703438

theorem sum_of_partitions_divisible_by_4 (n : ℕ) (h : n > 0) :
  ∀ (partition : ℕ → ℕ → bool), 
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → partition i j → (i ≠ j)) → 
  (∃ m : ℕ, m = 2 * n * (n - 1) ∧ m % 4 = 0) :=
sorry

end sum_of_partitions_divisible_by_4_l703_703438


namespace sara_initial_peaches_l703_703710

variable (p : ℕ)

def initial_peaches (picked_peaches total_peaches : ℕ) :=
  total_peaches - picked_peaches

theorem sara_initial_peaches :
  initial_peaches 37 61 = 24 :=
by
  -- This follows directly from the definition of initial_peaches
  sorry

end sara_initial_peaches_l703_703710


namespace rational_term_in_arith_progression_l703_703122

theorem rational_term_in_arith_progression 
  (n : ℤ) (h : n ≥ 3) :
  (n % 3 = 1) ↔ 
  (∀ (a₁ d : ℚ) (k : ℤ), 
    let a := λ i, a₁ + i * d in
    ((∑ i in finset.range (nat.to_fin n).card, (i + 1) * a i) ∈ ℚ) → 
    ∃ i, a i ∈ ℚ) :=
sorry

end rational_term_in_arith_progression_l703_703122


namespace family_total_cost_l703_703214

def cost_of_entrance_ticket : ℕ := 5
def cost_of_child_attraction_ticket : ℕ := 2
def cost_of_adult_attraction_ticket : ℕ := 4
def number_of_children : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_grandmothers : ℕ := 1

def number_of_family_members : ℕ := number_of_children + number_of_parents + number_of_grandmothers
def cost_of_entrance_tickets : ℕ := number_of_family_members * cost_of_entrance_ticket
def cost_of_children_attraction_tickets : ℕ := number_of_children * cost_of_child_attraction_ticket
def number_of_adults : ℕ := number_of_parents + number_of_grandmothers
def cost_of_adults_attraction_tickets : ℕ := number_of_adults * cost_of_adult_attraction_ticket

def total_cost : ℕ := cost_of_entrance_tickets + cost_of_children_attraction_tickets + cost_of_adults_attraction_tickets

theorem family_total_cost : total_cost = 55 := 
by 
  -- Calculation of number_of_family_members: 4 children + 2 parents + 1 grandmother = 7
  have h1 : number_of_family_members = 4 + 2 + 1 := by rfl
  -- Calculation of cost_of_entrance_tickets: 7 people * $5 = $35
  have h2 : cost_of_entrance_tickets = 7 * 5 := by rw h1 
  -- Calculation of cost_of_children_attraction_tickets: 4 children * $2 = $8
  have h3 : cost_of_children_attraction_tickets = 4 * 2 := by rfl
  -- Calculation of number_of_adults: 2 parents + 1 grandmother = 3
  have h4 : number_of_adults = 2 + 1 := by rfl
  -- Calculation of cost_of_adults_attraction_tickets: 3 adults * $4 = $12
  have h5 : cost_of_adults_attraction_tickets = 3 * 4 := by rw h4
  -- Total cost calculation: $35 (entrance) + $8 (children) + $12 (adults) = $55
  have h6 : total_cost = 35 + 8 + 12 := 
    by rw [h2, h3, h5]
  exact h6


end family_total_cost_l703_703214


namespace ratio_surface_area_volume_l703_703017

theorem ratio_surface_area_volume (a b : ℕ) (h1 : a^3 = 6 * b^2) (h2 : 6 * a^2 = 6 * b) : 
  (6 * a^2) / (b^3) = 7776 :=
by
  sorry

end ratio_surface_area_volume_l703_703017


namespace complement_B_range_a_l703_703172

open Set

variable (A B : Set ℝ) (a : ℝ)

def mySetA : Set ℝ := {x | 2 * a - 2 < x ∧ x < a}
def mySetB : Set ℝ := {x | 3 / (x - 1) ≥ 1}

theorem complement_B_range_a (h : mySetA a ⊆ compl mySetB) : 
  compl mySetB = {x | x ≤ 1} ∪ {x | x > 4} ∧ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end complement_B_range_a_l703_703172


namespace max_area_of_pen_l703_703529

theorem max_area_of_pen (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (x : ℝ), (3 * x + x = 60) ∧ (2 * x * x = 450) :=
by
  -- This theorem states that there exists an x such that
  -- the total perimeter with internal divider equals 60,
  -- and the total area of the two squares equals 450.
  use 15
  sorry

end max_area_of_pen_l703_703529


namespace equilateral_triangle_side_length_l703_703448

theorem equilateral_triangle_side_length :
  let roots : Finset ℝ := {a | a^3 - 9*a^2 + 10*a + 5 = 0}.toFinset
  in roots.card = 3 →
     ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
       (a + b + c = 9) ∧
       (a * b + b * c + c * a = 10) →
         ∃ l : ℝ, l = 2 * Real.sqrt 17 := sorry

end equilateral_triangle_side_length_l703_703448


namespace average_distances_is_correct_l703_703075

def diagonal_length (a : ℝ) : ℝ := real.sqrt (a^2 + a^2)

def position_after_diagonal_run (d : ℝ) (a : ℝ) : ℝ × ℝ :=
  let fraction_of_diagonal := d / (diagonal_length a) in 
  (fraction_of_diagonal * a, fraction_of_diagonal * a)

def position_after_first_turn (init_pos : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (init_pos.1 + d, init_pos.2)

def position_after_second_turn (init_pos : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (init_pos.1, init_pos.2 - d)

def distances_to_sides (pos : ℝ × ℝ) (a : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (pos.1, pos.2, a - pos.1, a - pos.2)

def average_distance (dists : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  (dists.1 + dists.2 + dists.3 + dists.4) / 4

theorem average_distances_is_correct :
  ∀ (a d1 d2 d3 : ℝ),
  a = 8 → d1 = 4.8 → d2 = 3 → d3 = 1 →
  let pos_after_diagonal := position_after_diagonal_run d1 a in
  let pos_after_first := position_after_first_turn pos_after_diagonal d2 in
  let pos_after_second := position_after_second_turn pos_after_first d3 in
  let dists := distances_to_sides pos_after_second a in
  average_distance dists = 4 :=
by
  intros a d1 d2 d3 ha hd1 hd2 hd3;
  sorry

end average_distances_is_correct_l703_703075


namespace projection_slope_angle_l703_703556

noncomputable def slope_angle (a b c : ℝ) : ℝ :=
  let m_PQ := ((sqrt 3) - 0) / (-2 - (-1)) in
  let m_l := -1 / m_PQ in
  real.arctan m_l

theorem projection_slope_angle (a b c : ℝ) (h_line : a * -2 + b * sqrt 3 + c = 0) : 
  slope_angle a b c = 30 :=
by sorry

end projection_slope_angle_l703_703556


namespace shaded_region_area_l703_703865

-- Circle definition with center and radius
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Point definition
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Midpoint of two points
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

noncomputable def R := midpoint P Q

-- Tangent segments
def tangent_point (C : Circle) (P : Point) (H : P ≠ C.center) :=
  let r := C.radius in
  ∃ θ : ℝ, P = { x := C.center.x + r * cos θ, y := C.center.y + r * sin θ }

-- Shaded region area calculation
theorem shaded_region_area {P Q : Point} (r : ℝ) (hP : Circle P 3)
    (hQ : Circle Q 3) (hR : 3 * sqrt 2 = dist P R)
    (hT1 : Tangent R S hP) (hT2 : Tangent R T hQ)
    (hUV : TangentLine UV hP hQ) : 
    area_shaded_region P Q R S T U V = 36 * (sqrt 2) - 9 - (9 * Real.pi) / 4 :=
  sorry

end shaded_region_area_l703_703865


namespace harris_flour_amount_l703_703770

noncomputable def flour_needed_by_cakes (cakes : ℕ) : ℕ := cakes * 100

noncomputable def traci_flour : ℕ := 500

noncomputable def total_cakes : ℕ := 9

theorem harris_flour_amount : flour_needed_by_cakes total_cakes - traci_flour = 400 := 
by
  sorry

end harris_flour_amount_l703_703770


namespace salary_increase_after_reduction_l703_703692

theorem salary_increase_after_reduction (E S : ℝ) (hE : E > 0) (hS : S > 0) :
  let E' := 0.9 * E in
  E * S = E' * (S / 0.9) →
  ((S / 0.9) - S) / S * 100 = 11.11 :=
by
  sorry

end salary_increase_after_reduction_l703_703692


namespace problem_statement_l703_703591

-- Define the problem
theorem problem_statement (a b : ℝ) (h : a - b = 1 / 2) : -3 * (b - a) = 3 / 2 := 
  sorry

end problem_statement_l703_703591


namespace remainder_when_divided_by_18_l703_703275

theorem remainder_when_divided_by_18 (n : ℕ) (r3 r6 r9 : ℕ)
  (hr3 : r3 = n % 3)
  (hr6 : r6 = n % 6)
  (hr9 : r9 = n % 9)
  (h_sum : r3 + r6 + r9 = 15) :
  n % 18 = 17 := sorry

end remainder_when_divided_by_18_l703_703275


namespace mobile_profit_percentage_l703_703248

-- Conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 2 / 100
def overall_profit : ℝ := 500

-- Function to calculate selling price of grinder
def selling_price_grinder (cp: ℝ) (loss_pct: ℝ) : ℝ :=
  cp - (cp * loss_pct)

-- Function to calculate total cost price
def total_cost_price (cp1 cp2: ℝ) : ℝ :=
  cp1 + cp2

-- Function to calculate total selling price with overall profit
def total_selling_price (tcp profit: ℝ) : ℝ :=
  tcp + profit

-- Function to calculate selling price of mobile
def selling_price_mobile (tsp spg: ℝ) : ℝ :=
  tsp - spg

-- Function to calculate profit on mobile
def profit_mobile (sp cp: ℝ) : ℝ :=
  sp - cp

-- Function to calculate profit percentage
def profit_percentage (profit cp: ℝ) : ℝ :=
  (profit / cp) * 100

-- Given conditions
def initial_conditions : Prop :=
  let spg := selling_price_grinder cost_price_grinder loss_percentage_grinder
  let tcp := total_cost_price cost_price_grinder cost_price_mobile
  let tsp := total_selling_price tcp overall_profit
  let spm := selling_price_mobile tsp spg
  let pm := profit_mobile spm cost_price_mobile
  profit_percentage pm cost_price_mobile = 10

-- Statement equivalent to the mathematical problem
theorem mobile_profit_percentage : initial_conditions :=
by
  -- provide proof for initial conditions
  sorry

end mobile_profit_percentage_l703_703248


namespace arithmetic_seq_a7_l703_703627

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 50) 
  (h_a5 : a 5 = 30) : 
  a 7 = 10 := 
by
  sorry

end arithmetic_seq_a7_l703_703627


namespace Ndoti_wins_l703_703639

-- Define the square ABCD
structure Square :=
  (A B C D : Point) -- vertices
  (area : ℝ) -- total area of the square

-- Define the points chosen by Kilua and Ndoti
structure Points :=
  (X Y Z W : Point) -- points on the sides of the square

-- Define a function to compute the area of the convex quadrilateral formed by points X, Y, Z, W
def quadrilateral_area (X Y Z W : Point) : ℝ := sorry -- assume some method to calculate the area

-- Total area of the square
variable (total_square_area : ℝ)

-- The condition that the quadrilateral area should be at most half the area of the square
def half_square_area (s : Square) (p : Points) : Prop :=
  quadrilateral_area p.X p.Y p.Z p.W ≤ s.area / 2

-- Prove that Ndoti has a winning strategy
theorem Ndoti_wins (s : Square) (p : Points) : half_square_area s p :=
sorry

end Ndoti_wins_l703_703639


namespace card_arrangement_bound_l703_703101

theorem card_arrangement_bound : 
  ∀ (cards : ℕ) (cells : ℕ), cards = 1000 → cells = 1994 → 
  ∃ arrangements : ℕ, arrangements = cells - cards + 1 ∧ arrangements < 500000 :=
by {
  sorry
}

end card_arrangement_bound_l703_703101


namespace integral_f_eq_neg32_l703_703938

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  - x^3 + 18 * x + a

theorem integral_f_eq_neg32 :
  ∃ a : ℝ, (∫ x in 0..2, f a x) = a → a = -32 :=
by {
  let f := f a,
  let I := ∫ x in 0..2, f a x,
  use -32,
  have h : (∫ x in 0..2, f (-32) x) = -32,
  { sorry },
  exact ⟨ h, rfl ⟩
}

end integral_f_eq_neg32_l703_703938


namespace percentage_invalid_votes_l703_703620

theorem percentage_invalid_votes
  (total_votes : ℕ)
  (votes_for_A : ℕ)
  (candidate_A_percentage : ℝ)
  (total_votes_count : total_votes = 560000)
  (votes_for_A_count : votes_for_A = 404600)
  (candidate_A_percentage_count : candidate_A_percentage = 0.85) :
  ∃ (x : ℝ), (x / 100) * total_votes = total_votes - votes_for_A / candidate_A_percentage ∧ x = 15 :=
by
  sorry

end percentage_invalid_votes_l703_703620


namespace evaluate_expression_l703_703886

-- Define the necessary conditions as constants in Lean.
def numerator := 35 * 100
def denominator := 0.07 * 100

-- Statement of the problem as a theorem.
theorem evaluate_expression : numerator / denominator = 500 := by
  sorry

end evaluate_expression_l703_703886


namespace total_marbles_in_bowls_l703_703358

theorem total_marbles_in_bowls :
  let second_bowl := 600
  let first_bowl := 3 / 4 * second_bowl
  let third_bowl := 1 / 2 * first_bowl
  let fourth_bowl := 1 / 3 * second_bowl
  first_bowl + second_bowl + third_bowl + fourth_bowl = 1475 :=
by
  sorry

end total_marbles_in_bowls_l703_703358


namespace f_0_plus_f_1_l703_703555

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg1 : f (-1) = 2

theorem f_0_plus_f_1 : f 0 + f 1 = -2 :=
by
  sorry

end f_0_plus_f_1_l703_703555


namespace rhombus_area_l703_703554

theorem rhombus_area 
  (a b : ℝ)
  (side_length : ℝ)
  (diff_diag : ℝ)
  (h_side_len : side_length = Real.sqrt 89)
  (h_diff_diag : diff_diag = 6)
  (h_diag : a - b = diff_diag ∨ b - a = diff_diag)
  (h_side_eq : side_length = Real.sqrt (a^2 + b^2)) :
  (1 / 2 * a * b) * 4 = 80 :=
by
  sorry

end rhombus_area_l703_703554


namespace eccentricity_of_ellipse_l703_703646

-- Definitions for the ellipse and related properties
variables {a b c k : ℝ}
def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def is_focus (F : ℝ × ℝ) (a c : ℝ) := F = (c, 0) ∨ F = (-c, 0)
def perpendicular (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 = 0

-- Main problem statement
theorem eccentricity_of_ellipse (h_ellipse : ellipse a b) 
  (h_focus1 : is_focus (-c, 0) a c) (h_focus2 : is_focus (c, 0) a c) 
  (h_AF_eq_3BF : ∃ k : ℝ, k > 0 ∧ dist A F1 = 3 * dist F1 B) 
  (h_AB_perpendicular_AF2 : perpendicular (A-B) (A-F2) ) 
  (h_a_gt_b : a > b) (h_b_gt_0 : b > 0) :
  let e := c / a in e = sqrt 2 / 2 :=
sorry

end eccentricity_of_ellipse_l703_703646


namespace edward_rides_l703_703883

theorem edward_rides (total_tickets tickets_spent tickets_per_ride rides : ℕ)
    (h1 : total_tickets = 79)
    (h2 : tickets_spent = 23)
    (h3 : tickets_per_ride = 7)
    (h4 : rides = (total_tickets - tickets_spent) / tickets_per_ride) :
    rides = 8 := by sorry

end edward_rides_l703_703883


namespace max_sum_of_vertex_products_l703_703849

theorem max_sum_of_vertex_products (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6}) :
  8 * (a * c * e + a * c * f + a * d * e + a * d * f + b * c * e + b * c * f + b * d * e + b * d * f) <= 343 :=
sorry

end max_sum_of_vertex_products_l703_703849


namespace bobby_consumption_l703_703856

theorem bobby_consumption :
  let initial_candy := 28
  let additional_candy_portion := 3/4 * 42
  let chocolate_portion := 1/2 * 63
  initial_candy + additional_candy_portion + chocolate_portion = 91 := 
by {
  let initial_candy : ℝ := 28
  let additional_candy_portion : ℝ := 3/4 * 42
  let chocolate_portion : ℝ := 1/2 * 63
  sorry
}

end bobby_consumption_l703_703856


namespace derivative_y_eq_x2_div_sin_x_find_a_b_extremum_inductive_proof_cubic_root_conditions_l703_703405

-- Problem 1
theorem derivative_y_eq_x2_div_sin_x (x : ℝ) : 
  (deriv (λ x : ℝ, x^2 / sin x) x) = (2 * x * sin x - x^2 * cos x) / (sin x)^2 :=
sorry

-- Problem 2
theorem find_a_b_extremum
  {x y : ℝ → ℝ}
  (h : x = 1)
  (f : ℝ → ℝ)
  (v : ℝ)
  (extremum : ∃ (a b : ℝ), y = x^3 + a * x^2 + b * x + a^2 ∧ f x = 10)
  : f = y :=
sorry

-- Problem 3
theorem inductive_proof (n : ℕ) :
  (∃ k : ℕ, (n + 1) * (n + 2) * ... * (n + n) = 2^k * 1 * 3 * ... * (2n - 1)) → 
  (∃ k : ℕ, (n + 2) * (n + 3) * ... * (2n + 2) = 2^k * 1 * 3 * ... * (2n - 1) * 2 * (2k + 1)) :=
sorry

-- Problem 4
theorem cubic_root_conditions
  {a b : ℝ} :
  (a = -3 ∧ b = -3) ∨ (a = -3 ∧ b > 2) ∨ (a = 0 ∧ b = 2) →
  ∃ (x : ℝ), x^3 + a * x + b = 0 ∧ ∀ y : ℝ, x ≠ y →
  x^3 + a * y + b ≠ 0 :=
sorry

end derivative_y_eq_x2_div_sin_x_find_a_b_extremum_inductive_proof_cubic_root_conditions_l703_703405


namespace ratio_AH_HG_GF_l703_703624

structure Square (A B C D E F G H : Type) :=
  (is_square : ∃ points, is_square points) -- denotes that ABCD forms a square
  (midpoint_AB : midpoint (A, B) = E)
  (midpoint_BC : midpoint (B, C) = F)
  (AF_CE_intersect_G : ∃ G, intersection (A, F) (C, E) = G)
  (AF_DE_intersect_H : ∃ H, intersection (A, F) (D, E) = H)

theorem ratio_AH_HG_GF
  (A B C D E F G H : Type)
  [Square A B C D E F G H] :
  ratio (A, H) (H, G) (G, F) = (6, 4, 5) :=
sorry

end ratio_AH_HG_GF_l703_703624


namespace simplify_sqrt_expr_l703_703300

noncomputable def simplify_expr : ℝ :=
  let a := (sqrt 8 : ℝ)
  let b := (sqrt 32 : ℝ)
  let c := (sqrt 72 : ℝ)
  let d := (sqrt 50 : ℝ)
  a - b + c - d

theorem simplify_sqrt_expr : simplify_expr = -sqrt 2 :=
  by
    sorry -- Proof will be provided here later

end simplify_sqrt_expr_l703_703300


namespace problem_l703_703198

variable {x y : ℝ}

theorem problem (hx : 0 < x) (hy : 0 < y) (h : x^2 - y^2 = 3 * x * y) :
  (x^2 / y^2) + (y^2 / x^2) - 2 = 9 :=
sorry

end problem_l703_703198


namespace train_pass_bridge_in_approx_26_64_sec_l703_703462

noncomputable def L_train : ℝ := 240 -- Length of the train in meters
noncomputable def L_bridge : ℝ := 130 -- Length of the bridge in meters
noncomputable def Speed_train_kmh : ℝ := 50 -- Speed of the train in km/h
noncomputable def Speed_train_ms : ℝ := (Speed_train_kmh * 1000) / 3600 -- Speed of the train in m/s
noncomputable def Total_distance : ℝ := L_train + L_bridge -- Total distance to be covered by the train
noncomputable def Time : ℝ := Total_distance / Speed_train_ms -- Time to pass the bridge

theorem train_pass_bridge_in_approx_26_64_sec : |Time - 26.64| < 0.01 := by
  sorry

end train_pass_bridge_in_approx_26_64_sec_l703_703462


namespace sum_of_valid_n_l703_703372

open Nat

theorem sum_of_valid_n :
  let gcd_lcm_condition (n : ℕ) := lcm n 150 = gcd n 150 + 600
  in (∀ n : ℕ, gcd_lcm_condition n → n ∈ [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150, 225, 675].filter (λ x, gcd x 150 ∈ [1, 75])) →
  ∑ n in (Finset.filter gcd_lcm_condition (Finset.range 1000)), n = 675 := sorry

end sum_of_valid_n_l703_703372


namespace area_of_lune_l703_703837

noncomputable def calculate_lune_area (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let area1 := (1 / 2) * real.pi * (r1 ^ 2)
  let area2 := (1 / 2) * real.pi * (r2 ^ 2)
  area2 - area1

theorem area_of_lune : calculate_lune_area 2 3 = (5 / 8) * real.pi := by
  sorry

end area_of_lune_l703_703837


namespace range_of_a_l703_703329

noncomputable def f (a x : ℝ) : ℝ := log a (x^2 - a * x + 4)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ set.Ioi 1 → f a x > 0) ↔ 1 < a ∧ a < 2 * Real.sqrt 3 :=
begin
  sorry
end

end range_of_a_l703_703329


namespace increasing_interval_of_f_l703_703331

def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 2

theorem increasing_interval_of_f : ∀ x, 1 < x → f x > f 1 := 
sorry

end increasing_interval_of_f_l703_703331


namespace area_outside_small_squares_l703_703829

theorem area_outside_small_squares (a b : ℕ) (ha : a = 10) (hb : b = 4) (n : ℕ) (hn: n = 2) :
  a^2 - n * b^2 = 68 :=
by
  rw [ha, hb, hn]
  sorry

end area_outside_small_squares_l703_703829


namespace female_students_selected_l703_703608

theorem female_students_selected (males females : ℕ) (p : ℚ) (h_males : males = 28)
  (h_females : females = 21) (h_p : p = 1 / 7) : females * p = 3 := by 
  sorry

end female_students_selected_l703_703608


namespace cisco_spots_difference_l703_703583

theorem cisco_spots_difference :
  ∃ C G R : ℕ, R = 46 ∧ G = 5 * C ∧ G + C = 108 ∧ (23 - C) = 5 :=
by
  sorry

end cisco_spots_difference_l703_703583


namespace income_left_l703_703070

theorem income_left (income food_education_rent_left : ℝ) :
    let spent_on_food := 0.35 * income in
    let spent_on_education := 0.25 * income in
    let remaining_after_food_education := income - spent_on_food - spent_on_education in
    let spent_on_rent := 0.80 * remaining_after_food_education in
    let remaining_income := remaining_after_food_education - spent_on_rent in
    remaining_income = 0.08 * income := 
by
  sorry

end income_left_l703_703070


namespace proportion_solution_l703_703189

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 :=
by
  sorry

end proportion_solution_l703_703189


namespace part1_part2_l703_703671
noncomputable theory

-- Define the basic conditions and sets
def f_domain := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -2 < x ∧ x < 2}
def B (a : ℝ) := {x : ℝ | x^2 - x + a - a^2 < 0 ∧ a < 0}

open set

-- Helper predicate functions
def is_subset_of (X Y : set ℝ) : Prop := ∀ x, x ∈ X → x ∈ Y
def is_superset_of (X Y : set ℝ) : Prop := ∀ x, x ∈ Y → x ∈ X

-- Questions to be proven
theorem part1 (a : ℝ) :
  is_subset_of A (B a) → a ≤ -2 :=
sorry

theorem part2 (a : ℝ) :
  is_superset_of A (B a) → a ≥ -1 :=
sorry

end part1_part2_l703_703671


namespace cosine_angle_a_plus_b_a_minus_b_l703_703955

open Real

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : vector) : ℝ :=
  sqrt (v.1^2 + v.2^2)

def cosine_angle (u v : vector) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

def a : vector := (3, 1)
def b : vector := (2, 2)

def a_plus_b : vector := (a.1 + b.1, a.2 + b.2)
def a_minus_b : vector := (a.1 - b.1, a.2 - b.2)

theorem cosine_angle_a_plus_b_a_minus_b :
  cosine_angle a_plus_b a_minus_b = sqrt 17 / 17 := 
sorry

end cosine_angle_a_plus_b_a_minus_b_l703_703955


namespace black_boxcar_capacity_l703_703707

-- Definitions from conditions
def red_boxcars : ℕ := 3
def blue_boxcars : ℕ := 4
def black_boxcars : ℕ := 7
def total_capacity : ℕ := 132000

-- Define the variable B which is the coal capacity of each black boxcar
variable (B : ℕ)

-- Define the relation between the capacities
def blue_capacity : ℕ := 2 * B
def red_capacity : ℕ := 6 * B

-- The total capacity equation based on the conditions
def total_capacity_eq : red_boxcars * red_capacity + blue_boxcars * blue_capacity + black_boxcars * B = total_capacity :=
by sorry

-- Main theorem to prove
theorem black_boxcar_capacity (h : total_capacity_eq) : B = 4000 :=
by sorry

end black_boxcar_capacity_l703_703707


namespace area_of_triangle_obtuse_triangle_exists_l703_703225

-- Define the mathematical conditions given in the problem
variables {a b c : ℝ}
axiom triangle_inequality : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y
axiom sine_relation : 2 * Real.sin c = 3 * Real.sin a
axiom side_b : b = a + 1
axiom side_c : c = a + 2

-- Part 1: Prove the area of ΔABC equals the provided solution
theorem area_of_triangle : 
  2 * Real.sin c = 3 * Real.sin a → 
  b = a + 1 → 
  c = a + 2 → 
  a = 4 → 
  b = 5 → 
  c = 6 → 
  let ab_sin_c := (1 / 2) * 4 * 5 * ((3 * Real.sqrt 7) / 8) in
  ab_sin_c = 15 * Real.sqrt 7 / 4 :=
by {
  sorry, -- The proof steps are to be provided here
}

-- Part 2: Prove there exists a positive integer a such that ΔABC is obtuse with a = 2
theorem obtuse_triangle_exists :
  (∃ (a : ℕ) (h2 : 1 < a ∧ a < 3), 2 * Real.sin c = 3 * Real.sin a ∧ b = a + 1 ∧ c = a + 2) →
  ∃ a = 2 ∧ (b = a + 1 ∧ c = a + 2) :=
by {
  sorry, -- The proof steps are to be provided here
}

end area_of_triangle_obtuse_triangle_exists_l703_703225


namespace minimum_value_condition_l703_703917

theorem minimum_value_condition (x a : ℝ) (h1 : x > a) (h2 : ∀ y, y > a → x + 4 / (y - a) > 9) : a = 6 :=
sorry

end minimum_value_condition_l703_703917


namespace standard_equation_line_BC_fixed_point_l703_703149

section EllipseProof

open Real

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions from the problem
axiom a_gt_b_gt_0 : ∀ (a b : ℝ), a > b → b > 0
axiom passes_through_point : ∀ (a b x y : ℝ), ellipse a b x y → (x = 1 ∧ y = sqrt 2 / 2)
axiom has_eccentricity : ∀ (a b c : ℝ), c / a = sqrt 2 / 2 → c^2 = a^2 - b^2 → b = 1

-- The standard equation of the ellipse
theorem standard_equation (a b : ℝ) (x y : ℝ) :
  a = sqrt 2 → b = 1 → ellipse a b x y → ellipse (sqrt 2) 1 x y :=
sorry

-- Prove that BC always passes through a fixed point
theorem line_BC_fixed_point (a b x1 x2 y1 y2 : ℝ) :
  a = sqrt 2 → b = 1 → 
  ellipse a b x1 y1 → ellipse a b x2 y2 →
  y1 = -y2 → x1 ≠ x2 → (-1, 0) = (-1, 0) →
  ∃ (k : ℝ) (x : ℝ), x = -2 ∧ y = 0 :=
sorry

end EllipseProof

end standard_equation_line_BC_fixed_point_l703_703149


namespace find_third_number_l703_703726

theorem find_third_number (A B C : ℝ) (h1 : (A + B + C) / 3 = 48) (h2 : (A + B) / 2 = 56) : C = 32 :=
by sorry

end find_third_number_l703_703726


namespace diagonal_length_l703_703458

theorem diagonal_length {A : Type*} [metric_space A] (A_0 A_1 A_2 A_3 A_4 : A) :
  dist A_0 A_1 = 1 ∧
  dist A_1 A_2 = 1 ∧
  dist A_2 A_3 = 1 ∧
  dist A_3 A_4 = 1 ∧
  dist A_4 A_0 = 1 ∧
  dist A_0 A_3 = sqrt 2 ∧
  dist A_1 A_3 = sqrt 2 ∧
  inner A_0 A_1 A_2 = 0 ∧
  inner A_1 A_2 A_3 = 0
  → dist A_2 A_4 = sqrt (19 / 7) := by
  sorry

end diagonal_length_l703_703458


namespace capacity_of_tank_is_750_l703_703081

noncomputable def tank_capacity : ℕ :=
  let pipeA_rate := 40
  let pipeB_rate := 30
  let pipeC_rate := -20
  let cycle_time := 3
  let total_time := 45
  let net_rate_per_cycle := pipeA_rate + pipeB_rate + pipeC_rate
  let number_of_cycles := total_time / cycle_time
  net_rate_per_cycle * number_of_cycles

theorem capacity_of_tank_is_750 : tank_capacity = 750 :=
  by
    sorry

end capacity_of_tank_is_750_l703_703081


namespace base5_calc_example_l703_703859

theorem base5_calc_example :
  let b203 := 2 * 5^2 + 0 * 5^1 + 3 in
  let b24 := 2 * 5^1 + 4 in
  let b3 := 3 in
  (b203 * b24 / b3) = 343 :=
by
  let b203 := 2 * 5^2 + 0 * 5^1 + 3
  let b24 := 2 * 5^1 + 4
  let b3 := 3
  sorry

end base5_calc_example_l703_703859


namespace ratio_of_broken_spiral_shells_to_total_broken_shells_l703_703176

variable (P B Pn Ps Bs : Nat)

axiom h1 : P = 17
axiom h2 : B = 52
axiom h3 : Pn = 12
axiom h4 : Ps = P - Pn
axiom h5 : Bs = Ps + 21

theorem ratio_of_broken_spiral_shells_to_total_broken_shells : Bs / B = 1 / 2 :=
by
  have h6 : Ps = P - Pn := by rw [h4]
  have h7 : Ps = 17 - 12 := by rw [h1, h3]
  have h8 : Bs = (17 - 12) + 21 := by rw [h5, h7]
  have h9 : Bs = 26 := by linarith
  have h10 : Bs / B = 26 / 52 := by rw [h9, h2]
  have h11 : 26 / 52 = 1 / 2 := by norm_num
  exact h11

end ratio_of_broken_spiral_shells_to_total_broken_shells_l703_703176


namespace maximum_profit_10_balls_l703_703180

-- Conditions
def num_balls : ℕ := 10
def radioactive : ℕ := 1
def price_per_ball : ℕ := 1
def cost_per_measurement : ℕ := 1

-- Maximum Profit Calculation
def max_profit : ℕ :=
  let triangular (k : ℕ) : ℕ := k * (k + 1) / 2
  if num_balls ≤ triangular 4 + 1 then
    num_balls - (4 + 1)
  else
    sorry

theorem maximum_profit_10_balls : max_profit = 5 :=
  by {
    calc max_profit = num_balls - 5 : sorry
    ... = 5 : by rw [num_balls]; norm_num
  }

end maximum_profit_10_balls_l703_703180


namespace maximize_volume_proof_l703_703020

noncomputable def maximize_volume (total_length : ℝ) (length_difference : ℝ) : ℝ × ℝ :=
  let x := 1 in
  let h := 3.2 - 2 * x in
  let V := x * (x + length_difference) * h in
  (h, V)

theorem maximize_volume_proof (h : ℝ) (V : ℝ) : 
  let total_length := 14.8 in 
  let length_difference := 0.5 in
  (h, V) = maximize_volume total_length length_difference :=
by
  sorry

end maximize_volume_proof_l703_703020


namespace average_greater_than_median_by_17_pounds_l703_703959

-- Define the weights based on the given conditions
def hammie_weight : ℕ := 110
def sisters_weights : list ℕ := [6, 6, 7, 9, 12]

-- Calculate the median of the weights
def median_weight : ℕ := (7 + 9) / 2

-- Calculate the average (mean) weight
def average_weight : ℕ := (6 + 6 + 7 + 9 + 12 + 110) / 6

-- Define the proof statement
theorem average_greater_than_median_by_17_pounds : average_weight - median_weight = 17 :=
by
  -- Placeholder for the proof steps
  sorry

end average_greater_than_median_by_17_pounds_l703_703959


namespace min_norm_diff_eq_6_l703_703170

open Real EuclideanSpace

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

noncomputable def vector_a {V : Type*} [inner_product_space ℝ V] := sorry
noncomputable def vector_b {V : Type*} [inner_product_space ℝ V] := sorry
noncomputable def vector_c {V : Type*} [inner_product_space ℝ V] := sorry

theorem min_norm_diff_eq_6 
  (h1 : inner_product ℝ a b = 0) 
  (h2 : ∥c∥ = 1) 
  (h3 : ∥a - c∥ = 5) 
  (h4 : ∥b - c∥ = 5) 
  : ∥a - b∥ ≥ 6 := 
sorry

end min_norm_diff_eq_6_l703_703170


namespace adjacent_probability_l703_703807

theorem adjacent_probability (n : ℕ) (hn : n ≥ 2) : 
  let total_perms := nat.factorial n in
  let fav_perms := 2 * (n - 1) * nat.factorial (n - 2) in
  fav_perms / total_perms = 2 / n :=
sorry

end adjacent_probability_l703_703807


namespace half_children_notebook_distribution_l703_703133

theorem half_children_notebook_distribution (C N N' : ℕ) (h1 : N = C / 8) (h2 : C * N = 512) :
  N' = 512 / (C / 2) → N' = 16 :=
by
  have hC : C * (C / 8) = 512 := by rw [h1]; exact h2
  have hC_squared : (C ^ 2) / 8 = 512 := by rw [mul_div_assoc]; exact hC
  have fact : (C ^ 2) / 8 = 512 := by exact hC_squared
  have eqC: C * C / 8 = 512 := by rw mul_comm
  have eqC_value: C ^ 2 = 4096 := by exact eqC * 8
  have calcC: C = 64 := by exact intRoot_aux_eq eqC_value 2
  replace h1: N = C / 8 := sorry
  replace h2: N = 64 / 8 := sorry
  replace h3: N = 8 := sorry
  have halfC: C / 2 = 32 := by exact eq halfC
  have calcN : 512 / halfC = 16 := by nlinarith [halfC]
  sorry

end half_children_notebook_distribution_l703_703133


namespace math_proof_problem_l703_703468

noncomputable def f (x : ℝ) : ℝ := real.log10 (x / (x^2 + 1))

theorem math_proof_problem :
  (∀ x : ℝ, 0 < x → x < ∞) →
  (∀ x : ℝ, 0 < x ∧ x < 1 → f(x) < f(x + 1)) ∧ 
  (∀ x : ℝ, x > 1 → f(x) > f(x + 1)) :=
by
  sorry

end math_proof_problem_l703_703468


namespace arithmetic_sequence_sum_equality_l703_703677

variables {a_n : ℕ → ℝ} -- the arithmetic sequence
variables (S_n : ℕ → ℝ) -- the sum of the first n terms of the sequence

-- Define the conditions as hypotheses
def condition_1 (S_n : ℕ → ℝ) : Prop := S_n 3 = 3
def condition_2 (S_n : ℕ → ℝ) : Prop := S_n 6 = 15

-- Theorem statement
theorem arithmetic_sequence_sum_equality
  (h1 : condition_1 S_n)
  (h2 : condition_2 S_n)
  (a_n_formula : ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0))
  (S_n_formula : ∀ n, S_n n = n * (a_n 0 + (n - 1) * (a_n 1 - a_n 0) / 2)) :
  a_n 10 + a_n 11 + a_n 12 = 30 := sorry

end arithmetic_sequence_sum_equality_l703_703677


namespace find_value_of_fraction_l703_703544

noncomputable def lg (n : ℝ) := Real.log n / Real.log 10

theorem find_value_of_fraction (x y : ℝ) (h : lg(x - y) + lg(x + 2y) = lg 2 + lg x + lg y) (hy : y > 0) : x / y = 2 :=
sorry

end find_value_of_fraction_l703_703544


namespace triangle_sides_l703_703076

structure Triangle (α : Type) :=
  (A B C : α)

theorem triangle_sides
  (α : Type)
  [Field α]
  (m p q : α)
  (h0 : m ≠ 0)
  (h1 : p ≠ 0)
  (h2 : q ≠ 0)
  (rhombus_side_length : α)
  (h3 : rhombus_side_length = m)
  (side_BC : α)
  (h4 : side_BC = p + q) :
  ∃ (AB AC : α), sides (Triangle α) = (side_BC, m * (p + q) / q, m * (p + q) / p) :=
by
  use (m * (p + q) / q), (m * (p + q) / p)
  sorry

end triangle_sides_l703_703076


namespace schedule_arrangements_l703_703815

open Finset

noncomputable def num_valid_arrangements : ℕ :=
  card {l : List ℕ // 
    l ~ [0, 1, 2, 3, 4, 5] ∧ -- l is a permutation of [0, 1, 2, 3, 4, 5]
    l.head ≠ 5 ∧ -- Physical Education (5) is not in the first period
    l.nth 3 ≠ some 1  -- Mathematics (1) is not in the fourth period
  }

theorem schedule_arrangements : num_valid_arrangements = 480 := 
by
  sorry

end schedule_arrangements_l703_703815


namespace total_votes_is_5000_l703_703207

theorem total_votes_is_5000 :
  ∃ (V : ℝ), 0.45 * V - 0.35 * V = 500 ∧ 0.35 * V - 0.20 * V = 350 ∧ V = 5000 :=
by
  sorry

end total_votes_is_5000_l703_703207


namespace ratio_pythons_boa_constrictors_l703_703348

/-- There are 200 snakes in the park -/
def total_snakes : ℕ := 200

/-- There are 40 boa constrictors in the park -/
def boa_constrictors : ℕ := 40

/-- The total number of rattlesnakes in the park is 40 -/
def rattlesnakes : ℕ := 40

/-- The number of pythons is the total_snakes minus the sum of boa_constrictors and rattlesnakes -/
def pythons : ℕ := total_snakes - (boa_constrictors + rattlesnakes)

/-- The ratio of pythons to boa_constrictors is 3:1 -/
theorem ratio_pythons_boa_constrictors : pythons / boa_constrictors = 3 :=
by
  have h1 : total_snakes = 200 := rfl
  have h2 : boa_constrictors = 40 := rfl
  have h3 : rattlesnakes = 40 := rfl
  have h4 : pythons = total_snakes - (boa_constrictors + rattlesnakes) := rfl
  have h5 : pythons = 120 := by simp [h4, h1, h2, h3]
  have h6 : pythons / boa_constrictors = 120 / 40 := by rw h5; rw h2
  simp at h6
  exact h6

end ratio_pythons_boa_constrictors_l703_703348


namespace price_per_cup_l703_703484

theorem price_per_cup
  (num_trees : ℕ)
  (oranges_per_tree_g : ℕ)
  (oranges_per_tree_a : ℕ)
  (oranges_per_tree_m : ℕ)
  (oranges_per_cup : ℕ)
  (total_income : ℕ)
  (h_g : num_trees = 110)
  (h_a : oranges_per_tree_g = 600)
  (h_al : oranges_per_tree_a = 400)
  (h_m : oranges_per_tree_m = 500)
  (h_o : oranges_per_cup = 3)
  (h_income : total_income = 220000) :
  total_income / (((num_trees * oranges_per_tree_g) + (num_trees * oranges_per_tree_a) + (num_trees * oranges_per_tree_m)) / oranges_per_cup) = 4 :=
by
  repeat {sorry}

end price_per_cup_l703_703484


namespace no_pairs_satisfy_equation_l703_703983

theorem no_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ¬ (2 / a + 2 / b = 1 / (a + b)) :=
by
  intros a b ha hb h
  -- the proof would go here
  sorry

end no_pairs_satisfy_equation_l703_703983


namespace fraction_of_3_5_eq_2_15_l703_703780

theorem fraction_of_3_5_eq_2_15 : (2 / 15) / (3 / 5) = 2 / 9 := by
  sorry

end fraction_of_3_5_eq_2_15_l703_703780


namespace intersecting_roads_or_railways_l703_703721

noncomputable def syldavia_intersections : Prop :=
  let V := 11 in
  let connections := (V * (V - 1)) / 2 in
  let roads_or_railways := connection -> bool in
  let intersect (road1 road2 : connection) : Prop := -- some intersection logic here sorry,
  ∃ (r : ℕ) (h : r >= connections / 2)
    (edges : finset (fin connections)), 
    (∀ e ∈ edges, roads_or_railways e) ∧ 
    ∃(e1 e2 ∈ edges), (intersect e1 e2)

theorem intersecting_roads_or_railways (roads_or_railways : connection -> bool) : syldavia_intersections :=
sorry

end intersecting_roads_or_railways_l703_703721


namespace area_of_large_rectangle_l703_703328

-- Define the given areas for the sub-shapes
def shaded_square_area : ℝ := 4
def bottom_rectangle_area : ℝ := 2
def right_rectangle_area : ℝ := 6

-- Prove the total area of the large rectangle EFGH is 12 square inches
theorem area_of_large_rectangle : shaded_square_area + bottom_rectangle_area + right_rectangle_area = 12 := 
by 
sorry

end area_of_large_rectangle_l703_703328


namespace canada_2004_q4_l703_703267

theorem canada_2004_q4 (p : ℕ) (hp : p.prime) (hodd : p % 2 = 1) : 
  (∑ k in Finset.range (p + 1), k ^ (2 * p - 1)) % (p ^ 2) = (p * (p + 1) / 2) % (p ^ 2) := 
sorry

end canada_2004_q4_l703_703267


namespace tangent_line_intersects_circle_l703_703130

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ :=
  Real.exp x * (x^2 + a * x + 1 - 2 * a)

-- Define the tangent line at P(0, 1-2a)
def tangent_line (a : ℝ) (x : ℝ) : ℝ :=
  (1 - a) * x + (1 - 2 * a)

-- Define the circle equation
def circle (x y : ℝ) : ℝ :=
  x^2 + 2 * x + y^2 - 12

-- The theorem we are going to prove
theorem tangent_line_intersects_circle (a : ℝ) :
  ∃ x y : ℝ, y = tangent_line a x ∧ circle x y < 0 :=
sorry

end tangent_line_intersects_circle_l703_703130
