import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.FunctionPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Data.Fin.Vec
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Syntax
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Log
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.TwoDimensions.Ellipse
import Mathlib.MeasureTheory.Probability.Variable.Normal
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactics.ByContra
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Complex

namespace fixed_point_on_line_AC_l222_222999

-- Given definitions and conditions directly from a)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_P (x y : ℝ) : Prop := ∃ t : ℝ, x = t * y - 1
def reflection_across_x_axis (y : ℝ) : ℝ := -y

-- The final proof statement translating c)
theorem fixed_point_on_line_AC
  (A B C P : ℝ × ℝ)
  (hP : P = (-1, 0))
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hAB : ∃ t : ℝ, line_through_P A.1 A.2 ∧ line_through_P B.1 B.2)
  (hRef : C = (B.1, reflection_across_x_axis B.2)) :
  ∃ x y : ℝ, (x, y) = (1, 0) ∧ line_through_P x y := 
sorry

end fixed_point_on_line_AC_l222_222999


namespace root_expression_value_l222_222090

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value_l222_222090


namespace shortest_distance_point_to_line_segment_l222_222004

theorem shortest_distance_point_to_line_segment :
  let a : (ℝ × ℝ × ℝ) := (3, 4, 1)
  let b : (ℝ × ℝ × ℝ) := (1, 0, -1)
  let c : (ℝ × ℝ × ℝ) := (3, 10, 3)
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  let v := (1 + 2 * t, 10 * t, -1 + 4 * t)
  ∥v.1 - a.1, v.2 - a.2, v.3 - a.3∥ = 
  (Real.sqrt ((36 / 59)^2 + (-40 / 59)^2 + (-116 / 59)^2)) :=
begin
  sorry
end

end shortest_distance_point_to_line_segment_l222_222004


namespace combined_exceeds_limit_l222_222687

-- Let Zone A, Zone B, and Zone C be zones on a road.
-- Let pA be the percentage of motorists exceeding the speed limit in Zone A.
-- Let pB be the percentage of motorists exceeding the speed limit in Zone B.
-- Let pC be the percentage of motorists exceeding the speed limit in Zone C.
-- Each zone has an equal amount of motorists.

def pA : ℝ := 15
def pB : ℝ := 20
def pC : ℝ := 10

/-
Prove that the combined percentage of motorists who exceed the speed limit
across all three zones is 15%.
-/
theorem combined_exceeds_limit :
  (pA + pB + pC) / 3 = 15 := 
by sorry

end combined_exceeds_limit_l222_222687


namespace right_triangle_ratio_l222_222633

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end right_triangle_ratio_l222_222633


namespace minimum_value_of_function_l222_222940

theorem minimum_value_of_function : ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_of_function_l222_222940


namespace trinomials_real_roots_inequality_l222_222913

theorem trinomials_real_roots_inequality :
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ¬ (∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q))) >
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q)) :=
sorry

end trinomials_real_roots_inequality_l222_222913


namespace cos_angle_COB_l222_222099

variable {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variable {A B C O : V}
variable {triangle : affine_simplex ℝ V 2}
variable {r : ℝ}

-- Conditions
variable (h₁ : circumcenter triangle = O)
variable (h₂ : 2 * (triangle.points 0 -ᵥ O) + 2 * (triangle.points 1 -ᵥ O) + (triangle.points 2 -ᵥ O) = 0)
variable (h₃ : dist (triangle.points 1) (triangle.points 2) = 2)

-- Statement to prove
theorem cos_angle_COB : real_inner (triangle.points 2 -ᵥ O) (triangle.points 1 -ᵥ O) = (-1/4) * (∥triangle.points 2 -ᵥ O∥ * ∥triangle.points 1 -ᵥ O∥) := 
sorry

end cos_angle_COB_l222_222099


namespace min_value_rationalize_sqrt_denominator_l222_222191

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222191


namespace lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l222_222137

def total_area_of_triangles_and_quadrilateral (A B Q : ℝ) : ℝ :=
  A + B + Q

def lena_triangles_and_quadrilateral_area (A B Q : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_quadrilateral A B Q

def total_area_of_triangles_and_pentagon (C D P : ℝ) : ℝ :=
  C + D + P

def vasya_triangles_and_pentagon_area (C D P : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_pentagon C D P

theorem lena_can_form_rectangles (A B Q : ℝ) (h : lena_triangles_and_quadrilateral_area A B Q) :
  lena_triangles_and_quadrilateral_area A B Q :=
by 
-- We assume the definition holds as given
sorry

theorem vasya_can_form_rectangles (C D P : ℝ) (h : vasya_triangles_and_pentagon_area C D P) :
  vasya_triangles_and_pentagon_area C D P :=
by 
-- We assume the definition holds as given
sorry

theorem lena_and_vasya_can_be_right (A B Q C D P : ℝ)
  (hlena : lena_triangles_and_quadrilateral_area A B Q)
  (hvasya : vasya_triangles_and_pentagon_area C D P) :
  lena_triangles_and_quadrilateral_area A B Q ∧ vasya_triangles_and_pentagon_area C D P :=
by 
-- Combining both assumptions
exact ⟨hlena, hvasya⟩

end lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l222_222137


namespace correct_propositions_count_l222_222989

theorem correct_propositions_count :
  let prop1 := ¬ ∀ a : ℝ, a ≠ 0 → ¬ (a^2 + a = 0)
  let prop2 := ¬ ∀ a b : ℝ, a > b → (2^a > 2^b - 1)
  let prop3 := ∃ x : ℝ, x^2 + 1 < 1 in
  (nat_of_bool prop1 + nat_of_bool prop2 + nat_of_bool prop3) = 2 := 
by
  sorry

end correct_propositions_count_l222_222989


namespace value_of_f_2012_l222_222542

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4

theorem value_of_f_2012 (a b α β : ℝ) (h : f 2011 a b α β = 5) : f 2012 a b α β = 3 :=
by
  -- Definitions and conditions provided:
  have h1 : f 2011 a b α β = -a * Real.sin α - b * Real.cos β + 4 := sorry
  have h2 : -a * Real.sin α - b * Real.cos β + 4 = 5 := h
  have h3 : -a * Real.sin α - b * Real.cos β = 1 := by linarith
  have h4 : a * Real.sin α + b * Real.cos β = -1 := sorry
  -- Required to show: f 2012 a b α β = 3
  calc
    f 2012 a b α β 
        = a * Real.sin (π * 2012 + α) + b * Real.cos (π * 2012 + β) + 4 : sorry
    ... = a * Real.sin α + b * Real.cos β + 4 : sorry
    ... = -1 + 4 : by rw [h4]
    ... = 3 : by linarith

end value_of_f_2012_l222_222542


namespace min_sum_rect_box_l222_222235

-- Define the main theorem with the given constraints
theorem min_sum_rect_box (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_vol : a * b * c = 2002) : a + b + c ≥ 38 :=
  sorry

end min_sum_rect_box_l222_222235


namespace sum_of_digits_base_8_rep_of_888_l222_222353

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222353


namespace min_water_filled_cells_l222_222492

-- Declaration of the grid size and the initial conditions.
def gridSize := 14
def initial_water_row : Fin gridSize → Fin gridSize → Prop := λ r c, r = 0

-- Function for water spread which happens unless blocked by a sandbag.
def spread_water (water : Fin gridSize → Fin gridSize → Prop) (sandbags : Fin gridSize → Fin gridSize → Prop) : Fin gridSize → Fin gridSize → Prop :=
λ r c, water r c ∨ (∃ (dr dc : ℤ), abs dr + abs dc = 1 ∧ 0 ≤ r + dr ∧ r + dr < gridSize ∧ 0 ≤ c + dc ∧ c + dc < gridSize ∧ water (⟨r + dr, sorry⟩) (⟨c + dc, sorry⟩) ∧ ¬sandbags r c)

-- Vasya places sandbags in any three cells not occupied by water.
def place_sandbags (current_sandbags : Fin gridSize → Fin gridSize → Prop) : Fin gridSize → Fin gridSize → Prop :=
λ r c, current_sandbags r c ∨ sorry -- logic for placing new sandbags

-- Prove the minimum number of water-filled cells is 37 after optimal sandbag placement.
theorem min_water_filled_cells : ∃ sandbag_placement_strategy : (Fin gridSize → Fin gridSize → Prop) → (Fin gridSize → Fin gridSize → Prop),
  let final_water_distribution := sorry -- final water spread logic using spread_water
  in ∀ sandbags, (place_sandbags sandbags) = sandbag_placement_strategy (place_sandbags sandbags)
    → (final_water_distribution ∘ spread_water (initial_water_row) (place_sandbags sandbags)).cards ≥ 37 := sorry

end min_water_filled_cells_l222_222492


namespace find_n_l222_222154

theorem find_n :
  ∑ k in finset.range (n - 8), 1 / (k + 9) * sqrt (k + 11) + (k + 11) * sqrt (k + 9) = 1 / 9 → 
  n = 79 := 
sorry

end find_n_l222_222154


namespace sum_a_1_to_2014_eq_2014_l222_222575

def f (n : ℕ) : ℤ :=
if n % 2 = 1 then n^2 else - (n^2)

def a (n : ℕ) : ℤ :=
f n + f (n + 1)

theorem sum_a_1_to_2014_eq_2014 : (∑ n in Finset.range 2014, a (n + 1)) = 2014 :=
by
  sorry

end sum_a_1_to_2014_eq_2014_l222_222575


namespace sum_of_digits_base8_888_l222_222410

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222410


namespace min_value_of_f_inequality_for_a_b_l222_222577

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 2)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  intro x
  sorry

theorem inequality_for_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : 1/a + 1/b = Real.sqrt 3) : 
  1/a^2 + 2/b^2 ≥ 2 := by
  sorry

end min_value_of_f_inequality_for_a_b_l222_222577


namespace correct_sqrt_multiplication_l222_222818

theorem correct_sqrt_multiplication :
  ∀ (a b : ℝ), (a = 2) → (b = 3) → (sqrt a * sqrt b = sqrt (a * b)) :=
by
  intros a b ha hb
  rw [ha, hb]
  exact Eq.refl (sqrt (2 * 3))

end correct_sqrt_multiplication_l222_222818


namespace einstein_fundraising_l222_222307

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l222_222307


namespace find_BD_l222_222102

noncomputable def BD_of_triangle_ABC (A B C D : ℝ) (h1 : A = 0) (h2 : B = 5) (h3 : C = 10) (h4 : D = 12.5) : ℝ :=
  sqrt (75.25) - 2.5

theorem find_BD {BD : ℝ} {A B C D : ℝ} 
  (h1 : AC = BC := 10)
  (h2 : AB = 5) 
  (h3 : ∃ (D : ℝ), B < D ∧ CD = 13):
  BD = sqrt 75.25 - 2.5 :=
sorry

end find_BD_l222_222102


namespace sum_of_digits_base8_l222_222394

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222394


namespace opposite_of_num_l222_222752

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222752


namespace rooster_ratio_l222_222165

theorem rooster_ratio (R H : ℕ) 
  (h1 : R + H = 80)
  (h2 : R + (1 / 4) * H = 35) :
  R / 80 = 1 / 4 :=
  sorry

end rooster_ratio_l222_222165


namespace hyperbola_transverse_axis_l222_222998

noncomputable def hyperbola_transverse_axis_length (a b : ℝ) : ℝ :=
  2 * a

theorem hyperbola_transverse_axis {a b : ℝ} (h : a > 0) (h_b : b > 0) 
  (eccentricity_cond : Real.sqrt 2 = Real.sqrt (1 + b^2 / a^2))
  (area_cond : ∃ x y : ℝ, x^2 = -4 * Real.sqrt 3 * y ∧ y * y / a^2 - x^2 / b^2 = 1 ∧ 
                 Real.sqrt 3 = 1 / 2 * (2 * Real.sqrt (3 - a^2)) * Real.sqrt 3) :
  hyperbola_transverse_axis_length a b = 2 * Real.sqrt 2 :=
by
  sorry

end hyperbola_transverse_axis_l222_222998


namespace incorrect_statement_A_l222_222820

def quadrilateral_with_equal_sides_is_rhombus (Q : Type) := ∀ (a b c d : Q), by sorry
def quad_opposite_sides_parallel_and_equal_is_parallelogram (Q : Type) := ∀ (a b c d : Q), by sorry
def three_lines_intersect_pairs_no_common_point_plane (Q : Type) := ∀ (a b c : Q), by sorry
def quad_two_pairs_opposite_sides_parallel_is_parallelogram (Q : Type) := ∀ (a b c d : Q), by sorry

theorem incorrect_statement_A
  (H1 : quadrilateral_with_equal_sides_is_rhombus Q)
  (H2 : quad_opposite_sides_parallel_and_equal_is_parallelogram Q)
  (H3 : three_lines_intersect_pairs_no_common_point_plane Q)
  (H4 : quad_two_pairs_opposite_sides_parallel_is_parallelogram Q) : 
  ¬ quadrilateral_with_equal_sides_is_rhombus (Type) :=
sorry

end incorrect_statement_A_l222_222820


namespace area_of_triangle_l222_222938

theorem area_of_triangle :
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 34 :=
by {
  -- Definitions
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  -- Proof (normally written here, but omitted with 'sorry')
  sorry
}

end area_of_triangle_l222_222938


namespace exists_large_triplet_sum_l222_222898

theorem exists_large_triplet_sum (a : ℕ → ℕ) (h : (a 1, a 2, ..., a 10) ∈ {σ | σ ∈ permutations {1, 2, ..., 10}}) :
    ∃ i, (a i + a (i % 10 + 1) + a ((i + 1) % 10 + 1)) ≥ 17 :=
by sorry

end exists_large_triplet_sum_l222_222898


namespace f_2022_eq_2021_l222_222543

noncomputable def f : ℕ → ℝ → ℝ
| 0, x := x
| 1, x := -(2 * x + 7) / (x + 3)
| (n + 2), x := f 1 (f (n + 1) x)

theorem f_2022_eq_2021 : f 2022 2021 = 2021 := by
  sorry

end f_2022_eq_2021_l222_222543


namespace greatest_possible_value_of_k_l222_222263

theorem greatest_possible_value_of_k :
  ∃ k : ℝ, 
    (∀ (x: ℝ), x^2 + k * x + 8 = 0 → 
      ∃ α β : ℝ, α ≠ β ∧ α - β = sqrt 73) → 
      k = sqrt 105 := 
sorry

end greatest_possible_value_of_k_l222_222263


namespace bisection_method_midpoint_value_l222_222794

theorem bisection_method_midpoint_value
  (f : ℝ → ℝ)
  (h1 : f 1 < 0)
  (h2 : f 1.5 > 0)
  (h3 : f 2 > 0) :
  let x0 := 1.25
  in x0 = (1 + 1.5) / 2 :=
by
  let x0 := (1 + 1.5) / 2
  have hx0 : x0 = 1.25 := rfl
  exact hx0

end bisection_method_midpoint_value_l222_222794


namespace player_two_wins_best_strategy_l222_222534

theorem player_two_wins_best_strategy (n : ℕ) (hn : n % 2 = 1) :
  ∃ best_strategy_player1 best_strategy_player2,
    (∃ game_outcome : string, game_outcome = "Player 2 wins") :=
by
  sorry

end player_two_wins_best_strategy_l222_222534


namespace least_four_digit_palindrome_div_by_5_l222_222801

noncomputable def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in s = s.reverse

theorem least_four_digit_palindrome_div_by_5 : 
  ∃ n : ℕ, is_palindrome n ∧ 1000 ≤ n ∧ n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 5 = 0 → n ≤ m := 
sorry

end least_four_digit_palindrome_div_by_5_l222_222801


namespace original_price_l222_222496

theorem original_price (p q: ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) : 
  let x := 20000 / (10000^2 - (p^2 + q^2) * 10000 + p^2 * q^2)
  (x : ℝ) * (1 - p^2 / 10000) * (1 - q^2 / 10000) = 2 :=
by
  sorry

end original_price_l222_222496


namespace sum_of_digits_in_8_pow_2004_l222_222811

theorem sum_of_digits_in_8_pow_2004 : 
  let n := 8 ^ 2004,
      tens_digit := (n / 10) % 10,
      units_digit := n % 10
  in tens_digit + units_digit = 7 :=
by
  sorry

end sum_of_digits_in_8_pow_2004_l222_222811


namespace minor_premise_incorrect_verification_l222_222439

-- Define the conditions
def exponential_function (a : ℝ) (hx : x : ℝ) : ℝ := a ^ x
def power_function (alpha : ℝ) (x : ℝ) : ℝ := x ^ alpha 

-- The major premise: Exponential functions are increasing for a > 1
axiom exp_increasing (a : ℝ) (h : a > 1) : StrictMono (exponential_function a)

-- The minor premise: The statement should consider y = x ^ α is an exponential function
axiom minor_premise_incorrect (α : ℝ) (h : α > 1) : ¬ (power_function α = exponential_function a)

-- The conclusion appropriately derived
theorem minor_premise_incorrect_verification
  (α : ℝ) (a : ℝ) (h1 : α > 1) (h2 : a > 1) :
  ¬ (power_function α = exponential_function a) :=
by
  apply minor_premise_incorrect; assumption

end minor_premise_incorrect_verification_l222_222439


namespace domain_of_h_l222_222711

variable {α : Type*}

-- Define the function f based on the given domain
variable (f : ℝ → α) (hf : ∀ x, -8 ≤ x ∧ x ≤ 4 → f x = f x)

-- Define the function h based on the definition h(x) = f(3x + 1)
noncomputable def h (x : ℝ) : α := f (3 * x + 1)

-- Define the domain of a function
def domain (g : ℝ → α) : set ℝ := {x : ℝ | ∃ y, g x = y}

-- The problem statement: Prove that the domain of h is [-3, 1]
theorem domain_of_h : domain h = set.Icc (-3 : ℝ) (1 : ℝ) :=
sorry

end domain_of_h_l222_222711


namespace centroid_of_ABC_l222_222584

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 5, y := 5 }
def B : Point := { x := 8, y := -3 }
def C : Point := { x := -4, y := 1 }

def centroid (A B C : Point) : Point :=
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3 }

theorem centroid_of_ABC :
  centroid A B C = { x := 3, y := 1 } :=
by
  sorry

end centroid_of_ABC_l222_222584


namespace factor_tree_X_value_l222_222630

theorem factor_tree_X_value :
  let F := 2 * 5
  let G := 7 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  X = 16170 := by
sorry

end factor_tree_X_value_l222_222630


namespace locus_of_perpendicular_bisector_and_tangent_is_perpendicular_l222_222027

open EuclideanGeometry

noncomputable def circle (O : Point) (R : ℝ) := sorry
noncomputable def point_on_circle (M O : Point) (R : ℝ) := sorry
noncomputable def perpendicular_bisector (A M : Point) := sorry
noncomputable def tangent (M O : Point) (R : ℝ) := sorry
noncomputable def locus (N : Point) (L1 L2 : Line) := sorry
noncomputable def line_perpendicular (N : Point) (L : Line) := sorry

theorem locus_of_perpendicular_bisector_and_tangent_is_perpendicular
  (O A M : Point) (R : ℝ) :
  circle O R →
  point_on_circle M O R →
  (∀ N, locus N (perpendicular_bisector A M) (tangent M O R) →
         line_perpendicular N (line_through O A)) :=
sorry

end locus_of_perpendicular_bisector_and_tangent_is_perpendicular_l222_222027


namespace sum_of_digits_base8_888_l222_222330

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222330


namespace calculate_value_l222_222905

theorem calculate_value : (3 * 12 + 18) / (6 - 3) = 18 := by
  -- conditions
  let num := 3 * 12 + 18
  let denom := 6 - 3
  have h1 : num = 54 := by
    calc
      3 * 12 = 36     : by norm_num
      36 + 18 = 54    : by norm_num
  have h2 : denom = 3 := by
    calc
      6 - 3 = 3       : by norm_num
  have h3 : 54 / 3 = 18 := by norm_num
  show (3 * 12 + 18) / (6 - 3) = 18 from h3
  sorry

end calculate_value_l222_222905


namespace spider_socks_and_shoes_l222_222477

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem spider_socks_and_shoes :
  let total_items := 20,
      legs := 10
  in (∏ i in finset.range total_items, i + 1) / (2 ^ legs) = factorial 20 / 2^10 :=
by sorry

end spider_socks_and_shoes_l222_222477


namespace rationalize_denominator_min_value_l222_222183

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222183


namespace average_of_second_set_l222_222717

open Real

theorem average_of_second_set 
  (avg6 : ℝ)
  (n1 n2 n3 n4 n5 n6 : ℝ)
  (avg1_set : ℝ)
  (avg3_set : ℝ)
  (h1 : avg6 = 3.95)
  (h2 : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = avg6)
  (h3 : (n1 + n2) / 2 = 3.6)
  (h4 : (n5 + n6) / 2 = 4.400000000000001) :
  (n3 + n4) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_l222_222717


namespace trajectory_of_M_is_parabola_l222_222121

variables {α : Type*}
variables (A B C D A₁ B₁ C₁ D₁ M : α)
variables (dist_plane_M_AADD1A1 dist_line_M_BC : α → ℝ)

-- Definitions for points being on planes
def on_plane_ABB1A1 (M : α) : Prop := sorry
def on_plane_ADD1A1 (A D D₁ A₁ M : α) : Prop := sorry

-- Distance from point M to plane ADD1A1
def distance_to_plane (M : α) (plane : α → Prop) : ℝ := sorry

-- Distance from point M to line BC
def distance_to_line (M : α) (line : α → Prop) : ℝ := sorry

axiom distance_equal (M : α) : distance_to_plane M (on_plane_ADD1A1 A D D₁ A₁) =
                              distance_to_line M (on_plane_ABB1A1 A B B₁ A₁)

theorem trajectory_of_M_is_parabola : 
  ∀ M, on_plane_ABB1A1 M → 
       distance_to_plane M (on_plane_ADD1A1 A D D₁ A₁) = 
       distance_to_line M (on_plane_ABB1A1 A B B₁ A₁) → 
       (Parabola_definition M := sorry) :=
by
  sorry

end trajectory_of_M_is_parabola_l222_222121


namespace inv_dist_sum_maximization_l222_222861

variables {A B C O B1 C1 : Type}
variables [InnerProductSpace ℝ O]

-- Define the point inside the angle
def point_in_angle (A B C O : Point) : Prop :=
  angle A O B < π ∧ angle A O C < π ∧ angle B O C < π

-- Given conditions for the problem
axiom B1_on_AB : lies_on B1 A B
axiom C1_on_AC : lies_on C1 A C
axiom O_inside_ABC : point_in_angle A B C O

-- Define the function to maximize
def inv_dist_sum (B O : O) (C O : O) : ℝ :=
  (1 / dist B O) + (1 / dist C O)

-- Statement of the proof problem
theorem inv_dist_sum_maximization :
  ∃ B1 C1, lies_on B1 A B ∧ lies_on C1 A C ∧ inv_dist_sum B1 O C1 O = inv_dist_sum O B O C :=
sorry

end inv_dist_sum_maximization_l222_222861


namespace unit_digit_of_expression_l222_222289

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l222_222289


namespace inscribed_to_circumscribed_ratio_l222_222635

theorem inscribed_to_circumscribed_ratio (a b : ℕ)
  (h1 : 6 = a) (h2 : 8 = b) :
  let c := (a^2 + b^2).sqrt
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2 in
  inscribed_radius / circumscribed_radius = 2 / 5 :=
by
  have ha : a = 6 := h1
  have hb : b = 8 := h2
  let c : ℕ := Int.sqrt ((a: ℤ)^2 + (b: ℤ)^2) -- Hypotenuse length
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2
  have hc : c = 10 := by sorry
  have h_inscribed : inscribed_radius = 2 := by sorry
  have h_circumscribed : circumscribed_radius = 5 := by sorry
  rw [h_inscribed, h_circumscribed]
  norm_num

end inscribed_to_circumscribed_ratio_l222_222635


namespace remainder_sum_of_squares_25_mod_6_l222_222806

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem remainder_sum_of_squares_25_mod_6 :
  (sum_of_squares 25) % 6 = 5 :=
by
  sorry

end remainder_sum_of_squares_25_mod_6_l222_222806


namespace find_general_term_Tn_greater_than_one_l222_222970

open Real

variable {a : ℕ → ℝ}

-- Conditions for the arithmetic sequence
axiom a3 : a 3 = -4
axiom a1_a10 : a 1 + a 10 = 2

-- General formula for the sequence
def general_term (n : ℕ) : ℝ := 2 * n - 10

-- Proof that the derived general term matches the conditions.
theorem find_general_term (n : ℕ) :
  a 3 = -4 ∧ (a 1 + a 10 = 2) → a n = 2 * n - 10 := by
  intro h
  cases h with h1 h2
  have eq_a1 : a 1 = -8 := by sorry -- Derived from solving the equations
  have eq_d : (a (n + 1) - a n) = 2 := by sorry -- Derived from solving the equations
  induction n with n ih
  · sorry -- Base case for n = 0
  · sorry -- Inductive step

-- Conditions for the sequence {b_n}
def b (n : ℕ) : ℝ := 3 ^ (a n)
def T (n : ℕ) : ℝ := (List.range n).map b |>.prod

-- Proof that T_n > 1 for n > 9.
theorem Tn_greater_than_one {n : ℕ}:
  (∀ n, a n = log 3 (b n)) ∧ (∀ n, T n = (List.range n).map b |>.prod) → T n > 1 → n > 9 := by
  intro h _ 
  have t_eq : T n = 3 ^ (n ^ 2 - 9 * n) := by sorry
  have inequality : 3 ^ (n ^ 2 - 9 * n) > 1 → n ^ 2 - 9 * n > 0 := by sorry
  have factorization : n ^ 2 - 9 * n = n * (n - 9) := by sorry
  have final_step : n * (n - 9) > 0 → n > 9 := by sorry
  exact final_step

end find_general_term_Tn_greater_than_one_l222_222970


namespace opposite_of_neg_half_is_half_l222_222738

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222738


namespace sqrt_13_between_3_and_4_l222_222607

theorem sqrt_13_between_3_and_4 : 
  let a := Real.sqrt 13 in 3 < a ∧ a < 4 :=  sorry

end sqrt_13_between_3_and_4_l222_222607


namespace perpendicular_lines_find_n_l222_222275

noncomputable def find_n (k1 k2 : ℝ) (n : ℝ) : Prop :=
  (Polynomial.root (Polynomial.leadingCoeffquadPoly 2 8 n) k1) ∧
  (Polynomial.root (Polynomial.leadingCoeffquadPoly 2 8 n) k2) ∧
  (k1 * k2 = -1)

theorem perpendicular_lines_find_n (k1 k2 : ℝ) (n : ℝ) (h : find_n k1 k2 n) : n = -2 := by
  sorry

end perpendicular_lines_find_n_l222_222275


namespace sum_of_digits_base8_888_l222_222383

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222383


namespace einstein_fundraising_l222_222308

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l222_222308


namespace greatest_possible_k_l222_222267

theorem greatest_possible_k (k : ℂ) (h : ∃ x1 x2 : ℂ, x1 ≠ x2 ∧ x1 + x2 = -k ∧ x1 * x2 = 8 ∧ |x1 - x2| = sqrt 73) : k = sqrt 105 :=
sorry

end greatest_possible_k_l222_222267


namespace symmetry_of_g_about_point_l222_222966

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 5)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + 2 * π / 5)

theorem symmetry_of_g_about_point :
  (∀ x : ℝ, ω > 0 ∧ (∀ x, f ω x = f ω (x + π)) → (∀ x, g x = g (-(x + π / 10) - π / 10))) →
  (g (- π / 10) = 0) :=
begin
  -- Proof will go here
  sorry
end

end symmetry_of_g_about_point_l222_222966


namespace angle_symmetry_trapezoid_l222_222725

open EuclideanGeometry

variables {P : Type*} [EuclideanSpace P]

-- Definitions from the conditions
def is_trapezoid (A B C D : P) (AD BC : ℝ) (hAD : A ≠ D) (hBC : B ≠ C) : Prop :=
  ∃ (O : P), is_diagonal A C O ∧ is_diagonal B D O ∧ are_symmetric_about O B B' ∧ are_symmetric_about O C C'

-- Main theorem statement
theorem angle_symmetry_trapezoid (A B C D B' C' O : P) (AD BC : ℝ)
  (h_trap : is_trapezoid A B C D AD BC A_ne_D B_ne_C) :
  ∠ C'AC = ∠ B'DB :=
sorry

end angle_symmetry_trapezoid_l222_222725


namespace Jack_can_form_rectangle_l222_222132

theorem Jack_can_form_rectangle : 
  ∃ (a b : ℕ), 
  3 * a = 2016 ∧ 
  4 * a = 2016 ∧ 
  4 * b = 2016 ∧ 
  3 * b = 2016 ∧ 
  (503 * 4 + 3 * 9 = 2021) ∧ 
  (2 * 3 = 4) :=
by 
  sorry

end Jack_can_form_rectangle_l222_222132


namespace identify_irrational_number_l222_222493

theorem identify_irrational_number :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (sqrt 3 = a / b) :=
by
  sorry

end identify_irrational_number_l222_222493


namespace nth_term_l222_222517

theorem nth_term (b : ℕ → ℝ) (h₀ : b 1 = 1)
  (h_rec : ∀ n ≥ 1, (b (n + 1))^2 = 36 * (b n)^2) : 
  b 50 = 6^49 :=
by
  sorry

end nth_term_l222_222517


namespace sum_of_digits_base8_888_l222_222413

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222413


namespace intersection_of_A_and_B_l222_222974

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l222_222974


namespace sum_of_base_8_digits_888_l222_222337

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222337


namespace opposite_neg_half_l222_222745

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222745


namespace greatest_possible_k_l222_222269

theorem greatest_possible_k (k : ℝ) (h : ∀ x, x^2 + k * x + 8 = 0) (diff_roots : (∀ a b, a ≠ b → a - b = sqrt 73)) : k = sqrt 105 :=
by
  sorry

end greatest_possible_k_l222_222269


namespace right_triangle_perimeter_l222_222867

noncomputable def perimeter_of_right_triangle (x : ℝ) : ℝ :=
  let y := x + 15
  let c := Real.sqrt (x^2 + y^2)
  x + y + c

theorem right_triangle_perimeter
  (h₁ : ∀ a b : ℝ, a * b = 2 * 150)  -- The area condition
  (h₂ : ∀ a b : ℝ, b = a + 15)       -- One leg is 15 units longer than the other
  : perimeter_of_right_triangle 11.375 = 66.47 :=
by
  sorry

end right_triangle_perimeter_l222_222867


namespace determine_quadrilateral_shape_l222_222512

def Point := (ℝ × ℝ)

def is_rectangle (A B C D : Point) : Prop :=
  A = (0, 0) ∧ B = (0, 6) ∧ C = (3, 6) ∧ D = (3, 0)

def line_through (A B : Point) (angle : ℝ) : Point → Prop :=
  fun P => P.1 * angle = P.2

def intersection (l1 l2 : Point → Prop) : Point → Prop :=
  λ P, l1 P ∧ l2 P

def is_trapezoid (A B C D : Point) : Prop :=
  ¬ (C.1 = D.1 ∧ A.1 = B.1) ∧ (A.2 = B.2)

theorem determine_quadrilateral_shape :
  ∀ (A B C D : Point),
    is_rectangle A B C D →
    let l1 := line_through A B 1
    let l2 := line_through A B 3.73
    let l3 := line_through B A (-1)
    let l4 := line_through B A (-3.73)
    let P := intersection l1 l3
    let Q := intersection l2 l4
    is_trapezoid A B P Q :=
sorry

end determine_quadrilateral_shape_l222_222512


namespace log2_inequality_to_exponential_inequality_l222_222832

theorem log2_inequality_to_exponential_inequality :
  (∀ a b : ℝ, 2^a > 2^b → a > b) ∧ (∀ a b : ℝ, a > b → (a > 0 ∧ b > 0) ∧ (2^a > 2^b)) ∧ ¬ (∀ a b : ℝ, 2^a > 2^b ↔ log a / log 2 > log b / log 2) :=
by
  sorry

end log2_inequality_to_exponential_inequality_l222_222832


namespace sum_of_digits_base_8_rep_of_888_l222_222348

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222348


namespace square_land_area_l222_222822

theorem square_land_area (side_length : ℕ) (h₁ : side_length = 25) (shape_square : true) : 
    side_length * side_length = 625 :=
by 
  rw h₁
  norm_num

end square_land_area_l222_222822


namespace sum_of_digits_base8_888_l222_222381

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222381


namespace sum_of_digits_base8_888_l222_222379

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222379


namespace part1_max_value_of_abs_fx_part2_maximum_M_l222_222961

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then 
    x^2 + a * x + 1 - a 
  else 
    f (x + 2) a

-- Part (I) Statement
theorem part1_max_value_of_abs_fx (a : ℝ) (h : a = -8) : ∀ x, -6 ≤ x ∧ x ≤ 5 → |f x a| ≤ 9 := 
sorry

-- Part (II) Statement
theorem part2_maximum_M (a : ℝ) (h : -2 ≤ a ∧ a ≤ 4) : ∃ M, (|f x a| ≤ 3 ∀ x, x ∈ set.Icc 0 M) ∧ ∀ a ∈ set.Icc -2 4, M = 2 ↔ a = -2 :=
sorry

end part1_max_value_of_abs_fx_part2_maximum_M_l222_222961


namespace data_set_variance_zero_l222_222622

theorem data_set_variance_zero (x : ℝ) (h : variance [3, 3, 3, x] = 0) : x = 3 := 
sorry

end data_set_variance_zero_l222_222622


namespace continuous_at_2_l222_222438

-- Define the function f
def f (x : ℝ) : ℝ := -5 * x^2 - 8

-- Define continuity at a point
def is_continuous_at (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x - f x0| < ε

-- The theorem to prove the continuity of the function at x0 = 2
theorem continuous_at_2 : is_continuous_at f 2 :=
by
  intro ε ε_pos
  let δ := ε / 25
  use δ
  split
  · linarith
  · intro x hx
    have : |f x - f 2| < ε, sorry
    assumption

end continuous_at_2_l222_222438


namespace sixInchCubeWeight_sixInchCubeValue_l222_222467

-- Define the conditions 
def fourInchCubeWeight : ℝ := 5
def fourInchCubeValue : ℝ := 1200
def volumeRatio : ℝ := (6^3) / (4^3) -- Volume ratio between the six-inch and four-inch cubes

-- Define the questions as propositions
theorem sixInchCubeWeight : 
  (4^3 * fourInchCubeWeight * volumeRatio) = (4^3 * 16.875) :=
by
  sorry

theorem sixInchCubeValue : 
  (4^3 * fourInchCubeValue * volumeRatio) = (4^3 * 4050) :=
by
  sorry

end sixInchCubeWeight_sixInchCubeValue_l222_222467


namespace shaded_area_l222_222527

theorem shaded_area (R : ℝ) (hR : 0 < R) : 
  let α := π / 4 in
  let S₀ := (π * R^2) / 2 in
  S₀ = (π * R^2) / 2 :=
by sorry

end shaded_area_l222_222527


namespace dart_land_probability_l222_222848

noncomputable def octagon_side : ℝ := 2 + Real.sqrt 2

noncomputable def area_of_octagon (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

noncomputable def area_of_square (s : ℝ) : ℝ := (s * Real.sqrt 2)^2

noncomputable def probability (a_square : ℝ) (a_octagon : ℝ) : ℝ := a_square / a_octagon

theorem dart_land_probability :
  let s := octagon_side;
  let a_square := area_of_square s;
  let a_octagon := area_of_octagon s in
  abs (probability a_square a_octagon - 0.4) < 0.01 := 
by
  sorry

end dart_land_probability_l222_222848


namespace maximum_pieces_is_seven_l222_222253

noncomputable def max_pieces (PIE PIECE : ℕ) (n : ℕ) : Prop :=
  PIE = PIECE * n ∧ natDigits 10 PIE = List.nodup (natDigits 10 PIE) ∧ natDigits 10 PIECE = List.nodup (natDigits 10 PIECE)

theorem maximum_pieces_is_seven :
  max_pieces 95207 13601 7 :=
sorry

end maximum_pieces_is_seven_l222_222253


namespace charity_event_probability_l222_222897

theorem charity_event_probability :
  let A_days := 3
  let total_days := 5
  let A_total_ways := Nat.choose total_days A_days
  let consecutive_days := 3
  let probability := consecutive_days / A_total_ways

  A_total_ways = 10 → -- A₅³ is the number of ways B, C, and D can be chosen to participate.
  probability = 1 / 20
:=
by
  sorry

end charity_event_probability_l222_222897


namespace theta_central_l222_222987

noncomputable def sum_cis_arithmetic_sequence : ℝ :=
  let terms := [60, 70, 80, 90, 100, 110, 120, 130, 140]
  terms.sum (λ θ, Complex.exp (Real.pi * θ / 180))

theorem theta_central : ∀ θ, θ ∈ [60, 70, 80, 90, 100, 110, 120, 130, 140] →
  let sum := sum_cis_arithmetic_sequence
  ∃ r, sum = r * Complex.exp (Real.pi * 100 / 180) ∧ 0 ≤ 100 ∧ 100 < 360 :=
begin
  intro θ,
  cases List.mem_cons_iff θ 60 with H1 IH,
  { subst θ },
  cases List.mem_cons_iff θ 70 with H2 IH,
  { subst θ },
  cases List.mem_cons_iff θ 80 with H3 IH,
  { subst θ },
  cases List.mem_cons_iff θ 90 with H4 IH,
  { subst θ },
  cases List.mem_cons_iff θ 100 with H5 IH,
  { subst θ },
  cases List.mem_cons_iff θ 110 with H6 IH,
  { subst θ },
  cases List.mem_cons_iff θ 120 with H7 IH,
  { subst θ },
  cases List.mem_cons_iff θ 130 with H8 IH,
  { subst θ },
  cases List.mem_cons_iff θ 140 with H9 IH,
  { subst θ },
  { sorry }
end

end theta_central_l222_222987


namespace number_of_classes_min_wins_for_class2101_l222_222787

-- Proof Problem for Q1
theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 := sorry

-- Proof Problem for Q2
theorem min_wins_for_class2101 (y : ℕ) (h : y + (9 - y) = 9 ∧ 2 * y + (9 - y) >= 14) : y >= 5 := sorry

end number_of_classes_min_wins_for_class2101_l222_222787


namespace cevians_are_perpendicular_l222_222689

theorem cevians_are_perpendicular
  {A B C C1 A1 B1 X : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace C1] [MetricSpace A1] [MetricSpace B1 ]
  [MetricSpace X]
  (h₁ : C1 ∈ Line (A, B))
  (h₂ : A1 ∈ Line (B, C))
  (h₃ : B1 ∈ Line (C, A))
  (h₄ : Concurrent (Line (A, A1)) (Line (B, B1)) (Line (C, C1)) X)
  (h₅ : Angle (Line (A1, C1)) (Line (C1, B)) = Angle (Line (B1, C1)) (Line (C1, A))) :
  Perpendicular (Line (C, C1)) (Line (A, B)) :=
sorry

end cevians_are_perpendicular_l222_222689


namespace simplify_expression_l222_222507

theorem simplify_expression (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 :=
by
  sorry

end simplify_expression_l222_222507


namespace Veenapaniville_high_schools_l222_222644

theorem Veenapaniville_high_schools :
  ∃ (districtA districtB districtC : ℕ),
    districtA + districtB + districtC = 50 ∧
    (districtA + districtB + districtC = 50) ∧
    (∃ (publicB parochialB privateB : ℕ), 
      publicB + parochialB + privateB = 17 ∧ privateB = 2) ∧
    (∃ (publicC parochialC privateC : ℕ),
      publicC = 9 ∧ parochialC = 9 ∧ privateC = 9 ∧ publicC + parochialC + privateC = 27) ∧
    districtB = 17 ∧
    districtC = 27 →
    districtA = 6 := by
  sorry

end Veenapaniville_high_schools_l222_222644


namespace main_theorem_l222_222032

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = ∑ k in finset.range n, a (k + 1) / 2

def general_term (a : ℕ → ℝ) : Prop :=
  ∀ n, a n =
  if n = 1 then 1 else (1 / 2) * (3 / 2) ^ (n - 2)

def log_seq (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, b n = Real.log (3 * a (n + 1)) / Real.log (3 / 2)

def sum_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∑ k in finset.range n, 1 / (b k * b (k + 1))

theorem main_theorem (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  sequence a →
  general_term a →
  log_seq b a →
  sum_terms T b →
  ∀ n, T n = n / (n + 1) :=
by
  intro h1 h2 h3 h4
  sorry

end main_theorem_l222_222032


namespace valid_four_digit_count_l222_222079

namespace FourDigitNumbers

/-- Define the digits allowed in the thousands place -/
def choices_thousands : Finset ℕ := {1, 2, 4, 7, 9}

/-- Define the digits allowed in the other three places -/
def choices_other : Finset ℕ := {0, 1, 2, 4, 7, 9}

/-- Function to count the number of valid four-digit numbers -/
def count_valid_numbers : ℕ :=
  (choices_thousands.card) * (choices_other.card ^ 3)

/-- Theorem stating that the count of valid four-digit numbers is 1512 -/
theorem valid_four_digit_count : count_valid_numbers = 1512 := by
  rw [count_valid_numbers]
  rw [Finset.card_insert_of_not_mem (by decide : 1 ≠ 2) (by simp)]
  rw [Finset.card_insert_of_not_mem (by decide : 2 ≠ 4) (by simp)]
  rw [Finset.card_insert_of_not_mem (by decide : 4 ≠ 7) (by simp)]
  rw [Finset.card_insert_of_not_mem (by decide : 7 ≠ 9) (by simp)]
  rw [Finset.card_singleton]

  apply sorry  -- add the proof steps here

end FourDigitNumbers

end valid_four_digit_count_l222_222079


namespace cheaperCandy_cost_is_5_l222_222459

def cheaperCandy (C : ℝ) : Prop :=
  let expensiveCandyCost := 20 * 8
  let cheaperCandyCost := 40 * C
  let totalWeight := 20 + 40
  let totalCost := 60 * 6
  expensiveCandyCost + cheaperCandyCost = totalCost

theorem cheaperCandy_cost_is_5 : cheaperCandy 5 :=
by
  unfold cheaperCandy
  -- SORRY is a placeholder for the proof steps, which are not required
  sorry 

end cheaperCandy_cost_is_5_l222_222459


namespace f_at_seven_l222_222565

variable {𝓡 : Type*} [CommRing 𝓡] [OrderedAddCommGroup 𝓡] [Module ℝ 𝓡]

-- Assuming f is a function from ℝ to ℝ with the given properties
variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function.
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Condition 2: f(x + 2) = -f(x) for all x.
def periodic_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = - f x 

-- Condition 3: f(x) = 2x^2 when x ∈ (0, 2)
def interval_definition (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_seven
  (h_odd : odd_function f)
  (h_periodic : periodic_negation f)
  (h_interval : interval_definition f) :
  f 7 = -2 :=
by
  sorry

end f_at_seven_l222_222565


namespace opposite_of_neg_half_is_half_l222_222737

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222737


namespace bear_hunting_l222_222840

theorem bear_hunting
    (mother_meat_req : ℕ) (cub_meat_req : ℕ) (num_cubs : ℕ) (num_animals_daily : ℕ)
    (weekly_meat_req : mother_meat_req = 210)
    (weekly_meat_per_cub : cub_meat_req = 35)
    (number_of_cubs : num_cubs = 4)
    (animals_hunted_daily : num_animals_daily = 10)
    (total_weekly_meat : mother_meat_req + num_cubs * cub_meat_req = 350) :
    ∃ w : ℕ, (w * num_animals_daily * 7 = 350) ∧ w = 5 :=
by
  sorry

end bear_hunting_l222_222840


namespace collinear_points_count_l222_222886

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l222_222886


namespace least_four_digit_palindrome_div_by_5_l222_222799

noncomputable def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in s = s.reverse

theorem least_four_digit_palindrome_div_by_5 : 
  ∃ n : ℕ, is_palindrome n ∧ 1000 ≤ n ∧ n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 5 = 0 → n ≤ m := 
sorry

end least_four_digit_palindrome_div_by_5_l222_222799


namespace correct_calculation_problem_statement_l222_222602

theorem correct_calculation :
  ∀ (a b c : ℕ), a - b = c ↔ a = b + c :=
begin
  intros a b c,
  split;
  intro h,
  { rw h,
    exact Nat.add_sub_cancel b c },
  { rw h,
    exact Nat.add_sub_cancel' b c },
end

theorem problem_statement :
  ∀ a b c d : ℕ, (a + b = c) → (c - b = d) → (a = d) :=
by {
  intros a b c d habc hcd,
  rw ←hcd,
  rw Nat.add_sub_of_le,
  exact Nat.le_of_add_eq (eq.symm habc),
  sorry,
}

def number_to_subtract := 399
def initial_value := 514
def incorrect_result := 913
def correct_result := 115

example : ((initial_value + number_to_subtract = incorrect_result) ∧ (initial_value - number_to_subtract = correct_result)) :=
by {
  split,
  {
    -- 514 + 399 = 913
    exact rfl,
  },
  {
    -- 514 - 399 = 115
    exact rfl,
  } 
} 

end correct_calculation_problem_statement_l222_222602


namespace find_250th_term_l222_222319

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

noncomputable def sequence_omitting_squares_and_threes : ℕ → ℕ
| 0       => 0  -- Technically doesn't matter, starting from sequence being empty
| (n + 1) =>
  let next := n + 1
  if is_perfect_square next ∨ next % 3 = 0 then
    sequence_omitting_squares_and_threes n
  else
    next

theorem find_250th_term :
  sequence_omitting_squares_and_threes 250 = 377 :=
sorry

end find_250th_term_l222_222319


namespace quadratic_solution_l222_222779

theorem quadratic_solution :
  ∀ x : ℝ, (3 * x - 1) * (2 * x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 :=
by
  sorry

end quadratic_solution_l222_222779


namespace continuous_at_2_l222_222437

-- Define the function f
def f (x : ℝ) : ℝ := -5 * x^2 - 8

-- Define continuity at a point
def is_continuous_at (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x0| < δ → |f x - f x0| < ε

-- The theorem to prove the continuity of the function at x0 = 2
theorem continuous_at_2 : is_continuous_at f 2 :=
by
  intro ε ε_pos
  let δ := ε / 25
  use δ
  split
  · linarith
  · intro x hx
    have : |f x - f 2| < ε, sorry
    assumption

end continuous_at_2_l222_222437


namespace number_of_boys_in_biology_class_l222_222300

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l222_222300


namespace hat_price_after_discounts_l222_222871

-- Defining initial conditions
def initial_price : ℝ := 15
def first_discount_percent : ℝ := 0.25
def second_discount_percent : ℝ := 0.50

-- Defining the expected final price after applying both discounts
def expected_final_price : ℝ := 5.625

-- Lean statement to prove the final price after both discounts is as expected
theorem hat_price_after_discounts : 
  let first_reduced_price := initial_price * (1 - first_discount_percent)
  let second_reduced_price := first_reduced_price * (1 - second_discount_percent)
  second_reduced_price = expected_final_price := sorry

end hat_price_after_discounts_l222_222871


namespace problem1_problem2_l222_222066

-- Problem (1)
theorem problem1 (x : ℝ) : (2 * |x - 1| ≥ 1) ↔ (x ≤ 1/2 ∨ x ≥ 3/2) := sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : a > 0) : (∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) ↔ a ≥ 2 := sorry

end problem1_problem2_l222_222066


namespace maximum_pieces_l222_222248

theorem maximum_pieces :
  ∀ (ПИРОГ КУСОК : ℕ) (h1 : ПИРОГ = 95207) (h2 : КУСОК = 13601),
    (ПИРОГ = КУСОК * 7) ∧ (ПИРОГ < 100000) ∧ (ПИРОГ.to_digits.nodup) → 
    7 = 7 :=
by { sorry }

end maximum_pieces_l222_222248


namespace triangle_side_ratio_l222_222100

theorem triangle_side_ratio
  (a b c A B C : ℝ)
  (cosB cosC : ℝ)
  (h₁ : b * cos C + c * cos B = 2 * b)
  (h₂ : a = b * (sin A / sin B)) -- Law of Sines
  (h₃ : sin A = 2 * sin B)       -- Derived relationship from steps
  : a / b = 2 := 
by {
  sorry,
}

end triangle_side_ratio_l222_222100


namespace sum_of_digits_base8_l222_222385

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222385


namespace find_BD_l222_222101

noncomputable def BD_of_triangle_ABC (A B C D : ℝ) (h1 : A = 0) (h2 : B = 5) (h3 : C = 10) (h4 : D = 12.5) : ℝ :=
  sqrt (75.25) - 2.5

theorem find_BD {BD : ℝ} {A B C D : ℝ} 
  (h1 : AC = BC := 10)
  (h2 : AB = 5) 
  (h3 : ∃ (D : ℝ), B < D ∧ CD = 13):
  BD = sqrt 75.25 - 2.5 :=
sorry

end find_BD_l222_222101


namespace LindasOriginalSavings_l222_222160

theorem LindasOriginalSavings : 
  (∃ S : ℝ, (1 / 4) * S = 200) ∧ 
  (3 / 4) * S = 600 ∧ 
  (∀ F : ℝ, 0.80 * F = 600 → F = 750) → 
  S = 800 :=
by
  sorry

end LindasOriginalSavings_l222_222160


namespace solve_system_l222_222239

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem solve_system (a b : ℝ) :
  (f a b 1 = 4) ∧ (f a b 0 = 2) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_system_l222_222239


namespace opposite_of_neg_half_l222_222758

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222758


namespace range_b_extreme_value_g_range_k_monotonic_f_l222_222576

noncomputable def f (x k : ℝ) := Real.exp x - k * x^2

/-- Question 1 -/
theorem range_b_extreme_value_g :
  ∀ b : ℝ, (∀ x : ℝ, ∃ f g : ℝ → ℝ, g x = f x * (x^2 - b * x + 2) → 
  (k = 0 → ∃ c : ℂ, differentiable ℝ c) → 
  (∃ x : ℝ, g x = 0)) ↔ b ∈ (-∞, -2) ∪ (2, ∞) := sorry

/-- Question 2 -/
theorem range_k_monotonic_f :
  ∀ k : ℝ, (∀ x ∈ set.Ioi 0, differentiable ℝ (f x k) ∧ (Real.exp x - 2 * k * x) ≥ 0) ↔ k ∈ set.Iic (Real.e / 2) := sorry

end range_b_extreme_value_g_range_k_monotonic_f_l222_222576


namespace opposite_of_half_l222_222766

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222766


namespace find_missing_numbers_l222_222167

namespace CardPuzzle

def consecutive_not_asc_desc (cards : List ℕ) : Prop :=
  ∀ i, i ≤ cards.length - 3 → ¬(cards[i] < cards[i+1] ∧ cards[i+1] < cards[i+2]) ∧ ¬(cards[i] > cards[i+1] ∧ cards[i+1] > cards[i+2])

theorem find_missing_numbers {cards : List ℕ} (h_len : cards.length = 9)
  (h_distinct : cards.nodup)
  (h_missing : ∃ A B C, (cards.filter (λ n, n ≠ A ∧ n ≠ B ∧ n ≠ C)).length = 6 ∧ consecutive_not_asc_desc (cards.filter (λ n, n ≠ A ∧ n ≠ B ∧ n ≠ C))) :
  ∃ (A B C : ℕ), A = 5 ∧ B = 2 ∧ C = 9 :=
by
  sorry

end CardPuzzle

end find_missing_numbers_l222_222167


namespace einstein_needs_more_money_l222_222305

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l222_222305


namespace sequence_sum_correct_l222_222587

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n % 3 == 1 then 1 else if n % 3 == 2 then 2 else 3

theorem sequence_sum_correct :
  (∑ k in Finset.range 2004, sequence_sum (k + 1)) = 4008 :=
by
  simp only [sequence_sum]
  sorry

end sequence_sum_correct_l222_222587


namespace percentage_increase_soda_l222_222945

theorem percentage_increase_soda :
  let original_price_candy := 15 / 1.25 in
  let original_price_soda := 16 - original_price_candy in
  let new_price_soda := 6 in
  ((new_price_soda - original_price_soda) / original_price_soda * 100) = 50 :=
by
  sorry

end percentage_increase_soda_l222_222945


namespace Nori_gave_more_to_Lea_l222_222683

noncomputable def Nori_crayons_initial := 4 * 8
def Mae_crayons := 5
def Nori_crayons_left := 15
def Crayons_given_to_Lea := Nori_crayons_initial - Mae_crayons - Nori_crayons_left
def Crayons_difference := Crayons_given_to_Lea - Mae_crayons

theorem Nori_gave_more_to_Lea : Crayons_difference = 7 := by
  sorry

end Nori_gave_more_to_Lea_l222_222683


namespace find_b_l222_222095

-- Define the given hyperbola equation and conditions
def hyperbola (x y : ℝ) (b : ℝ) : Prop := x^2 - y^2 / b^2 = 1
def asymptote_line (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem to prove
theorem find_b (b : ℝ) (hb : b > 0) :
    (∀ x y : ℝ, hyperbola x y b → asymptote_line x y) → b = 2 :=
by 
  sorry

end find_b_l222_222095


namespace unit_digit_of_expression_is_4_l222_222284

theorem unit_digit_of_expression_is_4 :
  Nat.unitsDigit ((2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) - 1) = 4 :=
by
  sorry

end unit_digit_of_expression_is_4_l222_222284


namespace maximum_distance_with_tire_switching_l222_222952

theorem maximum_distance_with_tire_switching :
  ∀ (x y : ℕ),
    (∀ (front rear : ℕ), (front = 24000) ∧ (rear = 36000)) →
    x < 24000 →
    (y = min (24000 - x) (36000 - x)) →
    (x + y = 48000) :=
by {
  intros x y h_front_rear x_lt y_def,
  obtain ⟨front_eq, rear_eq⟩ := h_front_rear,
  rw [front_eq, rear_eq] at *,
  cases x_lt,
  sorry
}

end maximum_distance_with_tire_switching_l222_222952


namespace radius_of_third_circle_l222_222790

theorem radius_of_third_circle (r1 r2 : ℝ) (shaded_area : ℝ) (r3 : ℝ) 
  (h1 : r1 = 19) 
  (h2 : r2 = 29) 
  (h3 : shaded_area = π * (r2^2 - r1^2)) 
  (h4 : shaded_area = π * 480) 
  (h5 : π * r3^2 = shaded_area) :
  r3 = 4 * real.sqrt 30 := 
sorry

end radius_of_third_circle_l222_222790


namespace harry_minimal_plums_l222_222074

theorem harry_minimal_plums :
  ∃ n m p : ℕ, n > 0 ∧ 2012 + 13 * n = 13 * ((7 * n)//5) ∧ 3 * ((7 * n)//5) = 2 * ((7 * n)//5) + ((7 * n)//5) - 7 * n ∧
  ∃ k : ℕ,  k = 1 ∧ 2025 = 2012 + 13 * k :=
sorry

end harry_minimal_plums_l222_222074


namespace part1_part2_l222_222067

variables {x y a b : ℝ}

-- Define the line l1 as 2x + 4y - 1 = 0 and the point A(3, 0)
def line_l1 (x y : ℝ) := 2 * x + 4 * y - 1 = 0
def point_A := (3, 0 : ℝ × ℝ)

-- Define the line l2 passing through A and parallel to l1
def line_l2 (x y : ℝ) := x + 2 * y - 3 = 0

-- Prove that l2 passing through point A is indeed line l2
theorem part1 : line_l2 x y :=
by sorry

-- Let the point B(a, b) be the symmetric point of A with respect to line l1
def point_B := (2, -6 : ℝ × ℝ)

-- Prove that the coordinates of B are (2, -6)
theorem part2 (a b : ℝ) (h1 : 2 * ((a + 3) / 2) + 4 * (b / 2) - 1 = 0) (h2 : b = 2 * (a - 3)) :
  (a, b) = (2, -6) :=
by sorry

end part1_part2_l222_222067


namespace collinear_points_count_l222_222885

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l222_222885


namespace farmer_land_l222_222166

variable (T : ℝ) -- Total land owned by the farmer

def is_cleared (T : ℝ) : ℝ := 0.90 * T
def cleared_barley (T : ℝ) : ℝ := 0.80 * is_cleared T
def cleared_potato (T : ℝ) : ℝ := 0.10 * is_cleared T
def cleared_tomato : ℝ := 90
def cleared_land (T : ℝ) : ℝ := cleared_barley T + cleared_potato T + cleared_tomato

theorem farmer_land (T : ℝ) (h : cleared_land T = is_cleared T) : T = 1000 := sorry

end farmer_land_l222_222166


namespace us_supermarkets_l222_222831

variable (C : ℕ) (US : ℕ) (Canada : ℕ)

axiom total_supermarkets : US + Canada = 60
axiom more_in_us : US = Canada + 14

theorem us_supermarkets : US = 37 := by
  have Canada_eq: Canada = 23 := by
    have eq1: 2 * Canada + 14 = 60 := by
      calc
        2 * Canada + 14 = Canada + Canada + 14 := by ring
        _ = Canada + (Canada + 14) := by ring
        _ = US + Canada := by rw [more_in_us]
        _ = 60 := by exact total_supermarkets
    have eq2: 2 * Canada = 46 := by
      calc
        2 * Canada = 60 - 14 := by rw [eq1]
        _ = 46 := by norm_num
    exact eq_of_mul_eq_mul_left (show 2 ≠ 0 by norm_num) eq2
  have US_eq: US = Canada + 14 := by exact more_in_us
  exact eq_of_eq_true (by norm_num)

end us_supermarkets_l222_222831


namespace cos_double_angle_unit_circle_l222_222572

theorem cos_double_angle_unit_circle (α y₀ : ℝ) (h : (1/2)^2 + y₀^2 = 1) : 
  Real.cos (2 * α) = -1/2 :=
by 
  -- The proof is omitted
  sorry

end cos_double_angle_unit_circle_l222_222572


namespace circumscribed_sphere_surface_area_l222_222986

noncomputable def surface_area_of_circumscribed_sphere_from_volume (V : ℝ) : ℝ :=
  let s := V^(1/3 : ℝ)
  let d := s * Real.sqrt 3
  4 * Real.pi * (d / 2) ^ 2

theorem circumscribed_sphere_surface_area (V : ℝ) (h : V = 27) : surface_area_of_circumscribed_sphere_from_volume V = 27 * Real.pi :=
by
  rw [h]
  unfold surface_area_of_circumscribed_sphere_from_volume
  sorry

end circumscribed_sphere_surface_area_l222_222986


namespace chlorination_reaction_l222_222941

theorem chlorination_reaction:
  (c2h6 moles cl2 moles : ℝ) 
  (h_c2h6 : c2h6 = 1) 
  (h_cl2 : cl2 = 6) :
  (hcl : ℝ) 
  (h_reaction : c2h6 + 6 * cl2 = 1 * 6 + 6 * hcl) ->
  hcl = 6 :=
by
  intro hcl h_reaction
  have hc2h6_def := h_c2h6
  have hcl2_def := h_cl2
  sorry

end chlorination_reaction_l222_222941


namespace days_to_finish_together_l222_222487

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l222_222487


namespace problem1_problem2_l222_222441

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end problem1_problem2_l222_222441


namespace more_crayons_given_to_Lea_than_Mae_l222_222685

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end more_crayons_given_to_Lea_than_Mae_l222_222685


namespace inclination_angle_range_l222_222620

theorem inclination_angle_range (k : ℝ) (h : -1 ≤ k ∧ k ≤ real.sqrt 3) : 
  ∃ α : ℝ, α ∈ ([0, real.pi / 3] ∪ [3 * real.pi / 4, real.pi)) ∧ real.tan α = k := 
sorry

end inclination_angle_range_l222_222620


namespace six_digit_multiples_of_5_count_l222_222890

/-- 
  Among the six-digit numbers formed by the digits 0, 1, 2, 3, 4, 5 without repetition, 
  there are a total of 216 such numbers that are multiples of 5.
-/
theorem six_digit_multiples_of_5_count : 
  (card { n : ℕ | (0 ≤ n ∧ n < 10^6) ∧ (∀ (d : ℕ), d < 6 → ((n / 10^d) % 10) ∈ {0,1,2,3,4,5}) ∧ (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧ n % 5 = 0 }) = 216 :=
by
  /- sorry we skip the proof for now -/
  sorry

end six_digit_multiples_of_5_count_l222_222890


namespace find_m_l222_222026

noncomputable def a (n : ℕ) (h : n > 0) : ℝ := Real.log (n + 2) / Real.log (n + 1)

theorem find_m (h : ∀ n, n > 0 → a n h * a (n + 1) h = 2016) :
  m = 2 ^ 2016 - 2 := sorry

end find_m_l222_222026


namespace sin_phi_in_square_l222_222445

-- Define the basic geometry objects
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : (B.1 - A.1) = side_length ∧ (C.1 - B.1) = side_length ∧ (D.1 - C.1) = side_length ∧ (A.1 - D.1) = side_length ∧
               (B.2 - A.2) = side_length ∧ (C.2 - B.2) = side_length ∧ (D.2 - C.2) = side_length ∧ (A.2 - D.2) = side_length)

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Problem translation
theorem sin_phi_in_square (s : Square) (P Q : ℝ × ℝ) (φ : ℝ) :
  let A := s.A, B := s.B, C := s.C;
  P = midpoint A B ∧ Q = midpoint B C →
  sin φ = 3 / 5 :=
by
  sorry

end sin_phi_in_square_l222_222445


namespace unit_digit_of_expression_l222_222287

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l222_222287


namespace sum_of_digits_base_8_888_is_13_l222_222404

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222404


namespace polynomial_condition_l222_222151

-- Define the sum of digits function
def S (n : ℕ) : ℕ := (n.digits 10).sum

-- State the main theorem
theorem polynomial_condition
  (P : ℕ → ℤ)
  (h_poly : ∃ (coeffs : List ℤ), ∀ n, P n = (List.sum (coeffs.mapWithIndex (λ i a, a * n ^ i))))
  (h_pos : ∀ n ≥ 2016, P n > 0)
  (h_sum : ∀ n ≥ 2016, S (Int.toNat (P n)) = P (S n)) :
  (∃ c, c ∈ Finset.range 10 ∧ P = (λ _, c)) ∨ (P = id) := 
sorry

end polynomial_condition_l222_222151


namespace hermione_utility_l222_222592

theorem hermione_utility (h : ℕ) : (h * (10 - h) = (4 - h) * (h + 2)) ↔ h = 4 := by
  sorry

end hermione_utility_l222_222592


namespace sum_of_digits_base8_888_l222_222412

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222412


namespace g_equation_and_value_l222_222024

-- Define the function g with the given condition
def g : ℝ → ℝ := λ x, sorry

-- State the main theorem: g satisfies the given equation and the required value g(2)
theorem g_equation_and_value :
  (∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) → g 2 = 14 :=
by
  intro h
  sorry

end g_equation_and_value_l222_222024


namespace ellipse_cond_l222_222277

theorem ellipse_cond (m n : ℝ) (h : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ m > n > 0 :=
sorry

end ellipse_cond_l222_222277


namespace periodicity_sum_2019_2021_cos_satisfies_l222_222564

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_sym : ∀ x, f (4 - x) = f x)

theorem periodicity : ∀ x, f (x + 8) = f x := 
by
  sorry

theorem sum_2019_2021: f 2019 + f 2020 + f 2021 = 0 :=
by
  sorry

theorem cos_satisfies : 
  ∀ x, (cos ((π / 4) * x + π / 2)) = f x →
  (∀ x, f (4 - x) = f x) ∧ ∀ x, f (-x) = -f x :=
by
  sorry

end periodicity_sum_2019_2021_cos_satisfies_l222_222564


namespace parallel_perpendicular_trans_l222_222977

variables {Plane Line : Type}

-- Definitions in terms of lines and planes
variables (α β γ : Plane) (a b : Line)

-- Definitions of parallel and perpendicular
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- The mathematical statement to prove
theorem parallel_perpendicular_trans :
  (parallel a b) → (perpendicular b α) → (perpendicular a α) :=
by sorry

end parallel_perpendicular_trans_l222_222977


namespace abs_cube_root_neg8_l222_222231

def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_val (x : ℝ) : ℝ := if x < 0 then -x else x

theorem abs_cube_root_neg8 : abs_val (cube_root (-8)) = 2 := by
  sorry

end abs_cube_root_neg8_l222_222231


namespace probability_of_4_in_decimal_rep_l222_222690
noncomputable def decimal_rep_5_div_7 : Real := 5 / 7

theorem probability_of_4_in_decimal_rep :
  (0.714285714285 ≈ decimal_rep_5_div_7) →
  ∃ digit_set : Finset ℕ, 
  digit_set = {7, 1, 4, 2, 8, 5} ∧ 
  (1 / 6 : ℚ) = 1 / digit_set.card :=
by
  intros h
  use {7, 4, 1, 2, 8, 5}
  split
  sorry

end probability_of_4_in_decimal_rep_l222_222690


namespace polynomial_solutions_l222_222526

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 - x + 3)) →
  (P = 0 ∨ ∃ n : ℕ, P = (Polynomial.C 1) * (Polynomial.X^2 - 2 * Polynomial.X + 3)^n) :=
by
  sorry

end polynomial_solutions_l222_222526


namespace find_k_collinear_l222_222072

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (1, 2)

theorem find_k_collinear : ∃ k : ℝ, (1 - 2 * k, 3 - k) = (-k, k) * c ∧ k = -1/3 :=
by
  sorry

end find_k_collinear_l222_222072


namespace uniquely_identify_figure_l222_222593

structure Figure where
  is_curve : Bool
  has_axis_of_symmetry : Bool
  has_center_of_symmetry : Bool

def Circle : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Ellipse : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := false }
def Triangle : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }
def Square : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Rectangle : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Parallelogram : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := true }
def Trapezoid : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }

theorem uniquely_identify_figure (figures : List Figure) (q1 q2 q3 : Figure → Bool) :
  ∀ (f : Figure), ∃! (f' : Figure), 
    q1 f' = q1 f ∧ q2 f' = q2 f ∧ q3 f' = q3 f :=
by
  sorry

end uniquely_identify_figure_l222_222593


namespace sum_integers_80_to_90_l222_222809

theorem sum_integers_80_to_90 : ∑ i in Finset.range (91 - 80) + 80, i = 935 := by
  sorry

end sum_integers_80_to_90_l222_222809


namespace intercepts_sum_mod_17_l222_222510

theorem intercepts_sum_mod_17 : 
  ∃ x_0 y_0 : ℤ, 0 ≤ x_0 ∧ x_0 < 17 ∧ 0 ≤ y_0 ∧ y_0 < 17 ∧
    (5 * x_0 ≡ -1 [MOD 17]) ∧ (3 * y_0 ≡ 1 [MOD 17]) ∧ (x_0 + y_0 = 7) :=
by
  use 1
  use 6
  split
  split
  exact zero_le_one
  exact by norm_num
  split
  exact zero_le_six
  exact by norm_num
  split
  calc 5 * 1 = 5 : by ring
         ... ≡ -1 [MOD 17] : by norm_num
  split
  calc 3 * 6 = 18 : by ring
         ... ≡ 1 [MOD 17] : by norm_num
  calc 1 + 6 = 7 : by ring
  sorry

end intercepts_sum_mod_17_l222_222510


namespace sum_of_base8_digits_888_l222_222357

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222357


namespace rationalize_denominator_min_value_l222_222184

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222184


namespace problem1_problem2_l222_222969

def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We add this case for Lean to handle zero index
  else if n = 1 then 2
  else 2^(n-1)

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) a

theorem problem1 (n : ℕ) :
  a n = 
  if n = 1 then 2
  else 2^(n-1) :=
sorry

theorem problem2 (n : ℕ) :
  S n = 2^n :=
sorry

end problem1_problem2_l222_222969


namespace algebraic_expression_value_l222_222091

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m - 2 = 0) : 2 * m^2 - 2 * m = 4 := by
  sorry

end algebraic_expression_value_l222_222091


namespace integer_satisfy_sqrt_range_l222_222935

theorem integer_satisfy_sqrt_range :
  {(x : ℤ) | 7 > Real.sqrt (x.toReal) ∧ Real.sqrt (x.toReal) > 5}.toFinset.card = 23 :=
by
  sorry

end integer_satisfy_sqrt_range_l222_222935


namespace max_sum_of_extremes_l222_222731

theorem max_sum_of_extremes (squares circles : fin 5 → ℕ) (h1 : list.perm squares.to_list [1, 2, 4, 5, 6, 9, 10, 11, 13]) (h2 : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → circles k = squares i + squares j) :
 max (squares 0) (squares 4) + min (squares 0) (squares 4)  = 20 :=
 by {
 sorry
}

end max_sum_of_extremes_l222_222731


namespace maximum_pieces_l222_222250

theorem maximum_pieces :
  ∀ (ПИРОГ КУСОК : ℕ) (h1 : ПИРОГ = 95207) (h2 : КУСОК = 13601),
    (ПИРОГ = КУСОК * 7) ∧ (ПИРОГ < 100000) ∧ (ПИРОГ.to_digits.nodup) → 
    7 = 7 :=
by { sorry }

end maximum_pieces_l222_222250


namespace train_cross_time_l222_222874

-- Given conditions
def train_length : ℝ := 50  -- length of train in meters
def train_speed_kmh : ℝ := 144  -- speed of train in km/h

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Converted speed in m/s
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor

-- Expected time taken for the train to cross the pole
noncomputable def expected_time : ℝ := 1.25

-- Theorem to prove
theorem train_cross_time : (train_length / train_speed_ms = expected_time) :=
by
  sorry

end train_cross_time_l222_222874


namespace associates_hired_l222_222465

-- Given conditions as definitions
def current_partners : ℕ := 14
def initial_ratio (partners associates : ℕ) : Prop := 2 * associates = 63 * partners
def new_ratio (partners associates : ℕ) : Prop := partners = 1 * associates

-- Define functions to find the number of associates initially and after hiring
def current_associates (partners associates : ℕ) : ℕ := 
  if h : initial_ratio partners associates then associates else 0

def hired_associates (partners initial_associates hired : ℕ) : ℕ := 
  if h : new_ratio partners (initial_associates + hired) then hired else 0

-- Prove the equivalent statement
theorem associates_hired : hired_associates current_partners (current_associates current_partners 441) 35 = 35 :=
by
  sorry

end associates_hired_l222_222465


namespace prod_z1_z2_real_z_purely_imaginary_find_m_l222_222972

-- Definitions for the first part
def z1 : ℂ := 1 + Complex.i
def z2 (x y : ℝ) : ℂ := x + y * Complex.i 
def real_if_imaginary_zero (z : ℂ) : Prop := z.im = 0

-- The first proof statement
theorem prod_z1_z2_real (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : real_if_imaginary_zero (z1 * z2 x y)) : x / y = -1 :=
sorry

-- Definitions for the second part
def conj_z1 : ℂ := Complex.conj z1
def z_expr (m : ℝ) : ℂ := (z1 / conj_z1) ^ 2022 + (m^2 - m - 1) - (m + 1) * Complex.i

-- The second proof statement
theorem z_purely_imaginary_find_m (m : ℝ) (h : z_expr m = Complex.i * (z_expr m).im) : m = 2 :=
sorry

end prod_z1_z2_real_z_purely_imaginary_find_m_l222_222972


namespace find_divisor_l222_222726

theorem find_divisor : ∃ (divisor : ℕ), ∀ (quotient remainder dividend : ℕ), quotient = 14 ∧ remainder = 7 ∧ dividend = 301 → (dividend = divisor * quotient + remainder) ∧ divisor = 21 :=
by
  sorry

end find_divisor_l222_222726


namespace rationalize_denominator_min_sum_l222_222175

-- Defining the problem in Lean 4
theorem rationalize_denominator_min_sum :
  let A := 5
  let x := 1
  let y := 2
  let z := 1
  let a := 4
  (sqrt 50) / (sqrt 25 - sqrt 5) = (A * x * sqrt y + z) / a ∧ A + x + y + z + a = 13 := sorry

end rationalize_denominator_min_sum_l222_222175


namespace exists_root_between_l222_222051

-- Given definitions and conditions
variables (a b c : ℝ)
variables (ha : a ≠ 0)
variables (x1 x2 : ℝ)
variable (h1 : a * x1^2 + b * x1 + c = 0)    -- root of the first equation
variable (h2 : -a * x2^2 + b * x2 + c = 0)   -- root of the second equation

-- Proof statement
theorem exists_root_between (a b c : ℝ) (ha : a ≠ 0) (x1 x2 : ℝ)
    (h1 : a * x1^2 + b * x1 + c = 0) (h2 : -a * x2^2 + b * x2 + c = 0) :
    ∃ x3 : ℝ, 
      (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) ∧ 
      (1 / 2 * a * x3^2 + b * x3 + c = 0) :=
sorry

end exists_root_between_l222_222051


namespace distance_center_to_line_l222_222119

-- Definition of the parametric equation of the line l in the Cartesian coordinate system
def line_param_eq (t : ℝ) : ℝ × ℝ :=
  (3 - (real.sqrt 2 / 2) * t, real.sqrt 5 + (real.sqrt 2 / 2) * t)

-- Definition of the polar equation of circle C in the polar coordinate system
def circle_polar_eq (theta : ℝ) : ℝ :=
  2 * real.sqrt 5 * real.sin theta

-- Definitions for the circle in Cartesian coordinates
def circle_center : ℝ × ℝ := (0, real.sqrt 5)
def radius : ℝ := real.sqrt 5

-- Parametric equation in Cartesian converted form
noncomputable def line_in_cartesian (x y : ℝ) : Prop :=
  x + y - real.sqrt 5 = 3

-- Distance from a point to a line in Cartesian form
noncomputable def point_line_distance (x0 y0 a b c : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / real.sqrt (a ^ 2 + b ^ 2)

-- Parameters of line equation for distance calculation
def a : ℝ := 1
def b : ℝ := 1
def c : ℝ := -(real.sqrt 5 + 3)
def point : ℝ × ℝ := (0, real.sqrt 5)

-- Thm (I): Distance from the center of the circle to the line
theorem distance_center_to_line : point_line_distance (0) (real.sqrt 5) a b c = (3 * real.sqrt 2) / 2 :=
  sorry

-- Thm (II): Sum of distances |PA| + |PB|
noncomputable theorem sum_distances_PA_PB (t : ℝ) : 
  let P := (3, real.sqrt 5),
    A := line_param_eq t, 
    B := line_param_eq (-t) 
  in (abs (t) + abs (-t)) = 3 * real.sqrt 2 :=
  sorry

end distance_center_to_line_l222_222119


namespace geometric_sum_property_l222_222124

variable {a : ℕ → ℝ}

axiom geometric_sequence (r : ℝ) : ∀ n, a (n + 1) = a n * r

noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)

theorem geometric_sum_property 
  (h1 : a 1 + a 2 = 40) 
  (h2 : a 3 + a 4 = 60) 
  (h3 : geometric_sequence r) : 
  a 7 + a 8 = 135 :=
sorry

end geometric_sum_property_l222_222124


namespace multiply_fractions_l222_222903

theorem multiply_fractions :
  (2 / 9) * (5 / 14) = 5 / 63 :=
by
  sorry

end multiply_fractions_l222_222903


namespace sum_of_digits_base8_888_l222_222366

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222366


namespace sum_of_base_8_digits_888_l222_222336

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222336


namespace sum_of_valid_x_l222_222008

theorem sum_of_valid_x :
  (∑ x in {x : ℤ | let S := {107, 122, 127, 137, 152, x}
                      in (645 + x) / 6 = if x < 107 then 124.5
                         else if x > 152 then 132
                         else if 127 < x ∧ x < 137 then (127 + x) / 2
                         else (127 + 137) / 2}, x) = 234 :=
by
  sorry

end sum_of_valid_x_l222_222008


namespace tetrahedron_opposite_edges_perpendicular_iff_l222_222171

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB CD AC BD AD BC : ℝ)

-- Definition of tetrahedron
@[class] structure Tetrahedron (A B C D : Type) :=
  (AB : A)
  (CD : B)
  (AC : C)
  (BD : D)
  (AD BC : ℝ)

theorem tetrahedron_opposite_edges_perpendicular_iff 
  (t : Tetrahedron A B C D)
  : (t.AB * t.CD = 0 ∧ t.AC * t.BD = 0 ∧ t.AD * t.BC = 0) ↔ 
    (t.AB * t.AB + t.CD * t.CD = t.AC * t.AC + t.BD * t.BD ∧ 
     t.AC * t.AC + t.BD * t.BD = t.AD * t.AD + t.BC * t.BC) :=
sorry

end tetrahedron_opposite_edges_perpendicular_iff_l222_222171


namespace lying_dwarf_number_is_possible_l222_222432

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l222_222432


namespace sum_of_digits_base8_888_l222_222367

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222367


namespace hyperbola_equation_l222_222030

theorem hyperbola_equation (a b : ℝ) : 
  (∃ S : ℝ → ℝ → Prop,  
    (∃ p1 p2 : ℝ × ℝ, S p1 p2 ∧ p1 = (3, 0)) ∧ 
    (y = -√2 * x ∧ (S x y) = (↑⟦↑⟮x⟯²⟯ / (a²) - ↑⟦y⟯² / 12⏩ = 1))) :=
  S = (λ x y, (x^2 / 3 - y^2 / 6 = 1)) :=
sorry

end hyperbola_equation_l222_222030


namespace option_B_functions_same_l222_222420

theorem option_B_functions_same :
  (∀ u v, f u = sqrt ((1 + u) / (1 - u)) ∧ g v = sqrt ((1 + v) / (1 - v)) → f = g) :=
by {
  intro u v,
  intro h,
  cases h with h_f h_g,
  simp [h_f, h_g],
  sorry
}

end option_B_functions_same_l222_222420


namespace sum_of_valid_n_eq_18_l222_222007

theorem sum_of_valid_n_eq_18 :
  ∑ n in {n : ℤ | (∃ x : ℤ, n^2 - 17*n + 72 = x^2) ∧ (24 % n = 0)}, n = 18 :=
sorry

end sum_of_valid_n_eq_18_l222_222007


namespace max_pieces_is_seven_l222_222259

-- Define what it means for a number to have all distinct digits
def all_digits_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.nodup)

-- Define the main proof problem
theorem max_pieces_is_seven :
  ∃ (n : ℕ) (PIE : ℕ) (PIECE : ℕ),
  (PIE = PIECE * n) ∧
  (PIE >= 10000) ∧ (PIE < 100000) ∧
  all_digits_distinct PIE ∧
  all_digits_distinct PIECE ∧
  ∀ m, (m > n) → (¬ (∃ P' PIECE', (P' = PIECE' * m) ∧
   (P' >= 10000) ∧ (P' < 100000) ∧ all_digits_distinct P' ∧ all_digits_distinct PIECE'))
:= sorry

end max_pieces_is_seven_l222_222259


namespace final_acid_concentration_l222_222108

def volume1 : ℝ := 2
def concentration1 : ℝ := 0.40
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.60

theorem final_acid_concentration :
  ((concentration1 * volume1 + concentration2 * volume2) / (volume1 + volume2)) = 0.52 :=
by
  sorry

end final_acid_concentration_l222_222108


namespace probability_symmetry_line_l222_222292

theorem probability_symmetry_line (points : Finset (ℕ × ℕ))
  (h_count_points : points.card = 81)
  (P : ℕ × ℕ)
  (h_P_center : P = (4, 4))
  (Q : ℕ × ℕ)
  (h_Q_random : Q ∈ points ∧ Q ≠ P) :
  (∃ sym_points : Finset (ℕ × ℕ), sym_points.card = 32 ∧ (Q ∈ sym_points)) →
  (80:ℚ)⁻¹ * 32 = (2:ℚ)/5 :=
begin
  -- sorry is used to skip the proof
  sorry
end

end probability_symmetry_line_l222_222292


namespace find_sin_x_l222_222014

theorem find_sin_x (a b x : ℝ) (h₀ : a > b) (h₁ : b > 0) (h₂ : 0 < x) (h₃ : x < π / 2) (h₄ : cot x = (a^2 - b^2) / (2 * a * b)) :
  sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end find_sin_x_l222_222014


namespace find_AB_length_l222_222117

-- Consider a rectangle ABCD
variable (A B C D P : Point)
variable (AB BC CD DA : ℝ)

-- P is a point on BC
variable (BP CP : ℝ)
variable (tan_angle_APD : ℝ)

-- Here are the given conditions
variable (h1 : BP = 16)
variable (h2 : CP = 8)
variable (h3 : tan_angle_APD = 3)

-- Our goal is to prove AB = 16
theorem find_AB_length (h_ABCD : Rectangle A B C D)
                       (h_P_on_BC : On P (LineSegment B C))
                       (h_BP : distance B P = 16)
                       (h_CP : distance C P = 8)
                       (h_tan_APD : tan (angle A P D) = 3) :
                       distance A B = 16 := 
sorry

end find_AB_length_l222_222117


namespace circle_equation_l222_222001

theorem circle_equation 
  (x y : ℝ)
  (passes_origin : (x, y) = (0, 0))
  (intersects_line : ∃ (x y : ℝ), 2 * x - y + 1 = 0)
  (intersects_circle : ∃ (x y :ℝ), x^2 + y^2 - 2 * x - 15 = 0) : 
  x^2 + y^2 + 28 * x - 15 * y = 0 :=
sorry

end circle_equation_l222_222001


namespace equivalent_single_discount_rate_l222_222841

-- Definitions based on conditions
def original_price : ℝ := 120
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.15
def combined_discount_rate : ℝ := 0.3625  -- This is the expected result

-- The proof problem statement
theorem equivalent_single_discount_rate :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 
  (original_price * (1 - combined_discount_rate)) := 
sorry

end equivalent_single_discount_rate_l222_222841


namespace sum_of_digits_in_8_pow_2004_l222_222810

theorem sum_of_digits_in_8_pow_2004 : 
  let n := 8 ^ 2004,
      tens_digit := (n / 10) % 10,
      units_digit := n % 10
  in tens_digit + units_digit = 7 :=
by
  sorry

end sum_of_digits_in_8_pow_2004_l222_222810


namespace line_BD_tangent_to_circumcircle_TSH_l222_222514

open EuclideanGeometry

variables {A B C D H S T : Point}

theorem line_BD_tangent_to_circumcircle_TSH
  (hABCD : ConvexQuadrilateral A B C D)
  (hAngles : ∠ABC = 90 ∧ ∠CDA = 90)
  (hH_perp : foot_of_perpendicular A B D = H)
  (hS_on_AB : lies_on S AB)
  (hT_on_AD : lies_on T AD)
  (hH_inside_SCT : lies_in_triangle H SCT)
  (hAngleCondition1 : ∠CHS - ∠CSB = 90)
  (hAngleCondition2 : ∠THC - ∠DTC = 90) :
  is_Tangent (circumcircle (T S H)) (line_through B D) :=
sorry

end line_BD_tangent_to_circumcircle_TSH_l222_222514


namespace distance_between_intersections_l222_222113

noncomputable section

open Real

def point := ℝ × ℝ × ℝ

def start_point : point := (3, 0, -1)
def end_point : point := (1, -4, -5)
def sphere_center : point := (1, 1, 1)
def sphere_radius : ℝ := 2

-- Distance function between two points
def dist (a b : point) : ℝ :=
  sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2)

-- Line equation parameterized by t
def line (t : ℝ) : point :=
  (3 - 2*t, -4*t, -1 - 4*t)

-- Prove the distance between the intersection points with the sphere
theorem distance_between_intersections : 
  dist (line t1) (line t2) = 4 / 3 :=
by sorry

end distance_between_intersections_l222_222113


namespace blocks_selection_count_l222_222453

theorem blocks_selection_count :
  let n := 6 in
  let k := 4 in
  (Nat.choose n k * Nat.choose n k * Nat.factorial k = 5400) :=
by
  let n := 6
  let k := 4
  have h1 : Nat.choose n k = 15 := by sorry
  have h2 : Nat.factorial k = 24 := by sorry
  calc
    Nat.choose n k * Nat.choose n k * Nat.factorial k
      = 15 * 15 * 24 : by rw [h1, h1, h2]
  ... = 5400 : by norm_num

end blocks_selection_count_l222_222453


namespace cos_sum_identity_least_positive_integer_n_find_least_positive_integer_n_l222_222002

theorem cos_sum_identity :
  (∑ k in Finset.range 39 + 50, (1 : ℝ) / (Real.cos k.1 * Real.cos (k.1 + 1))) = 1 / Real.cos 1 :=
sorry

theorem least_positive_integer_n :
  ∀ n : ℕ, n > 0 → (∑ k in Finset.range (n - 1) + 50, 1 / (Real.cos k.1 * Real.cos (k.1 + 1))) = 1 / Real.cos n :=
sorry

-- Using the previous theorem to deduce the least positive integer:
theorem find_least_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ (∑ k in Finset.range 39 + 50, 1 / (Real.cos k.1 * Real.cos (k.1 + 1))) = 1 / Real.cos n :=
begin
  use 1,
  split,
  { exact nat.one_pos },
  { exact cos_sum_identity }
end

end cos_sum_identity_least_positive_integer_n_find_least_positive_integer_n_l222_222002


namespace desargues_theorem_l222_222316

open_locale classical

variables {Point : Type} [incidence_geometry Point]

-- Definitions of points and intersections
variables (A B C A' B' C' O P Q R : Point)

-- Conditions
axiom intersect_AA' : ∃ O, line_through A A' = line_through B B' ∧ line_through B' = line_through C C'
axiom intersect_P : ∃ P, line_through B C = line_through B' C'
axiom intersect_Q : ∃ Q, line_through C A = line_through C' A'
axiom intersect_R : ∃ R, line_through A B = line_through A' B'

theorem desargues_theorem
  (hO : intersect_AA')
  (hP : intersect_P)
  (hQ : intersect_Q)
  (hR : intersect_R) :
  collinear P Q R := sorry

end desargues_theorem_l222_222316


namespace problem_1_problem_2_l222_222965

open Complex

theorem problem_1 (m : ℝ) (h : (conj (1 + m * I) * (3 + I)).im = 0) :
  let z := 1 + m * I in
  let z1 := (m + 2 * I) / (1 - I) in
  abs z1 = Real.sqrt 26 / 2 :=
by
  sorry

theorem problem_2 (m a : ℝ) (h1 : (conj (1 + m * I) * (3 + I)).im = 0) :
  let z := 1 + m * I in
  let z2 := (a - I) / z in
  (z2.re > 0 ∧ z2.im > 0) → a > 1 / 3 :=
by
  sorry

end problem_1_problem_2_l222_222965


namespace sum_of_coordinates_l222_222700

theorem sum_of_coordinates (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 20) : x + y = 2 :=
sorry

end sum_of_coordinates_l222_222700


namespace limit_proof_l222_222962

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem limit_proof :
  (filter.at_top.limsup (λ Δx : ℝ, (f (2 + 3 * Δx) - f 2) / Δx)) = -3 / 4 :=
sorry

end limit_proof_l222_222962


namespace opposite_of_num_l222_222748

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222748


namespace basic_astrophysics_degrees_l222_222456

theorem basic_astrophysics_degrees :
  let microphotonics_pct := 12
  let home_electronics_pct := 24
  let food_additives_pct := 15
  let gmo_pct := 29
  let industrial_lubricants_pct := 8
  let total_budget_percentage := 100
  let full_circle_degrees := 360
  let given_pct_sum := microphotonics_pct + home_electronics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct
  let astrophysics_pct := total_budget_percentage - given_pct_sum
  let astrophysics_degrees := (astrophysics_pct * full_circle_degrees) / total_budget_percentage
  astrophysics_degrees = 43.2 := by
  sorry

end basic_astrophysics_degrees_l222_222456


namespace max_value_expression_l222_222010

theorem max_value_expression (x : ℝ) : 
  (∃ y : ℝ, y = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16) ∧ 
                ∀ z : ℝ, 
                (∃ x : ℝ, z = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16)) → 
                y ≥ z) → 
  ∃ y : ℝ, y = 1 / 16 := 
sorry

end max_value_expression_l222_222010


namespace greatest_possible_k_l222_222266

theorem greatest_possible_k (k : ℂ) (h : ∃ x1 x2 : ℂ, x1 ≠ x2 ∧ x1 + x2 = -k ∧ x1 * x2 = 8 ∧ |x1 - x2| = sqrt 73) : k = sqrt 105 :=
sorry

end greatest_possible_k_l222_222266


namespace opposite_of_half_l222_222767

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222767


namespace rationalize_denominator_min_value_l222_222189

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222189


namespace sum_of_k_l222_222727

theorem sum_of_k : ∃ (k_vals : List ℕ), 
  (∀ k ∈ k_vals, ∃ α β : ℤ, α + β = k ∧ α * β = -20) 
  ∧ k_vals.sum = 29 :=
by 
  sorry

end sum_of_k_l222_222727


namespace intersection_points_of_inscribed_polygons_l222_222220

theorem intersection_points_of_inscribed_polygons :
  let p6 := (6 : ℕ)
  let p7 := (7 : ℕ)
  let p8 := (8 : ℕ)
  let p9 := (9 : ℕ)
  ∀ (circle : Type) (inscribed : circle → ℕ → Prop),
    (inscribed circle p6) ∧ (inscribed circle p7) ∧ (inscribed circle p8) ∧ (inscribed circle p9) →
    (∀ (m n : ℕ), inscribed circle m → inscribed circle n → (m ≠ n) → (∀ v, ¬ shares_vertex v m n)) →
    (∀ (s1 s2 s3 : ℕ), inscribed circle s1 → inscribed circle s2 → inscribed circle s3 → 
     (s1 ≠ s2) → (s2 ≠ s3) → (s1 ≠ s3) → ∀ pt, ¬ common_intersection pt s1 s2 s3) →
    (number_of_intersections circle p6 p7 p8 p9 = 80) :=
by
  intros _ _ _ _ _
  sorry

end intersection_points_of_inscribed_polygons_l222_222220


namespace cost_of_each_folder_is_six_l222_222674

variable (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
          (erasers_per_pencils : ℕ) (cost_per_pencil : ℕ) 
          (cost_per_eraser : ℕ) (total_spent : ℕ) 
          (cost_of_paints : ℕ) (total_classes_supply_cost : ℕ)

-- Given conditions
def conditions := 
  classes = 6 ∧ 
  folders_per_class = 1 ∧
  pencils_per_class = 3 ∧
  erasers_per_pencils = 6 ∧
  cost_per_pencil = 2 ∧
  cost_per_eraser = 1 ∧
  total_spent = 80 ∧
  cost_of_paints = 5 ∧
  total_classes_supply_cost = 44

-- Define the cost of each folder
def folder_cost (folder_total_spent : ℕ) (folders_needed : ℕ) : ℕ :=
  folder_total_spent / folders_needed

-- The remaining cost to be spent on folders
def remaining_cost := total_spent - total_classes_supply_cost

-- Prove that the cost of each folder is $6
theorem cost_of_each_folder_is_six (h : conditions) : folder_cost remaining_cost 6 = 6 := 
  sorry

end cost_of_each_folder_is_six_l222_222674


namespace sum_of_base_8_digits_888_l222_222344

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222344


namespace range_of_a_l222_222126

variable (a : ℝ)
variable (x : ℝ)

noncomputable def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (h : ∀ x, otimes (x - a) (x + a) < 1) : - 1 / 2 < a ∧ a < 3 / 2 :=
sorry

end range_of_a_l222_222126


namespace book_configurations_l222_222468

theorem book_configurations (n : ℕ) (h_n : n = 8) : 
  (2 ≤ n - 1) → (1 ≤ n - 2) → ∃ (c : ℕ), c = 6 :=
by
  intro h1 h2
  use 6
  sorry

end book_configurations_l222_222468


namespace opposite_of_neg_half_l222_222756

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222756


namespace equations_not_equivalent_l222_222895

theorem equations_not_equivalent : 
  (∀ x : ℝ, (2 * real.sqrt (x + 5) = x + 2) ↔ (x = 4)) ∧ 
  (∀ x : ℝ, (4 * (x + 5) = (x + 2) ^ 2) ↔ (x = 4 ∨ x = -4)) → 
  (∃ x : ℝ, (2 * real.sqrt (x + 5) = x + 2) ∧ ¬ (4 * (x + 5) = (x + 2) ^ 2)) :=
by
  sorry

end equations_not_equivalent_l222_222895


namespace rationalize_denominator_min_value_l222_222203

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222203


namespace sum_f_vals_l222_222960

def f (n : ℤ) : ℝ := Real.sin (n * Real.pi / 4)

theorem sum_f_vals : (Finset.range 2008).sum (λ i, f (i + 1)) = 0 := by
  sorry

end sum_f_vals_l222_222960


namespace limit_find_l222_222670

noncomputable def f (n : ℕ) : ℝ :=
  (List.range (2 * n + 1)).map (λ k, n^2 + (k:ℝ)^2).prod

theorem limit_find : 
  (tendsto (λ (n : ℕ), (f n)^(1/(n:ℝ)) / n^4) atTop (𝓝 (Real.exp (2 * Real.log 5 + 2 * Real.arctan 2 - 4)))) :=
begin
  sorry
end

end limit_find_l222_222670


namespace sum_of_digits_base_8_rep_of_888_l222_222350

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222350


namespace rationalize_denominator_l222_222213

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222213


namespace bisect_area_of_trapezoid_l222_222724

-- Define the vertices of the quadrilateral
structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 16, y := 0 }
def C : Point := { x := 8, y := 8 }
def D : Point := { x := 0, y := 8 }

-- Define the equation of a line
structure Line :=
  (slope : ℚ)
  (intercept : ℚ)

-- Define the condition for parallel lines
def parallel (L1 L2 : Line) : Prop :=
  L1.slope = L2.slope

-- Define the diagonal AC and the required line
def AC : Line := { slope := 1, intercept := 0 }
def bisecting_line : Line := { slope := 1, intercept := -4 }

-- The area of trapezoid
def trapezoid_area : ℚ := (8 * (16 + 8)) / 2

-- Proof that the required line is parallel to AC and bisects the area of the trapezoid
theorem bisect_area_of_trapezoid :
  parallel bisecting_line AC ∧ 
  (1 / 2) * (8 * (16 + bisecting_line.intercept)) = trapezoid_area / 2 :=
by
  sorry

end bisect_area_of_trapezoid_l222_222724


namespace rationalize_denominator_min_value_l222_222199

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222199


namespace possible_values_of_n_l222_222821

-- Conditions: Definition of equilateral triangles and squares with side length 1
def equilateral_triangle_side_length_1 : Prop := ∀ (a : ℕ), 
  ∃ (triangle : ℕ), triangle * 60 = 180 * (a - 2)

def square_side_length_1 : Prop := ∀ (b : ℕ), 
  ∃ (square : ℕ), square * 90 = 180 * (b - 2)

-- Definition of convex n-sided polygon formed using these pieces
def convex_polygon_formed (n : ℕ) : Prop := 
  ∃ (a b c d : ℕ), 
    a + b + c + d = n ∧ 
    60 * a + 90 * b + 120 * c + 150 * d = 180 * (n - 2)

-- Equivalent proof problem
theorem possible_values_of_n :
  ∃ (n : ℕ), (5 ≤ n ∧ n ≤ 12) ∧ convex_polygon_formed n :=
sorry

end possible_values_of_n_l222_222821


namespace range_of_a_l222_222619

theorem range_of_a (a : ℝ) (h : (∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2)) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l222_222619


namespace maximum_distance_with_tire_switching_l222_222951

theorem maximum_distance_with_tire_switching :
  ∀ (x y : ℕ),
    (∀ (front rear : ℕ), (front = 24000) ∧ (rear = 36000)) →
    x < 24000 →
    (y = min (24000 - x) (36000 - x)) →
    (x + y = 48000) :=
by {
  intros x y h_front_rear x_lt y_def,
  obtain ⟨front_eq, rear_eq⟩ := h_front_rear,
  rw [front_eq, rear_eq] at *,
  cases x_lt,
  sorry
}

end maximum_distance_with_tire_switching_l222_222951


namespace consecutive_zeros_in_2017_factorial_l222_222504

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def legendre_exponent (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    let rec count (n : ℕ) (acc : ℕ) :=
      if n < p then acc else count (n / p) (acc + n / p)
    in count n 0

theorem consecutive_zeros_in_2017_factorial :
  legendre_exponent 2017 5 = 502 := 
by
  sorry

end consecutive_zeros_in_2017_factorial_l222_222504


namespace constant_term_binomial_expansion_l222_222278

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (3 * (1:ℚ) - (1 / (2 * 3 * (1:ℚ))) ^ 6) * (- (1 / 2) ^ 3) * (nat.choose 6 3) = -5 / 2 :=
by {
  sorry  -- the proof goes here
}

end constant_term_binomial_expansion_l222_222278


namespace gobbleian_words_count_l222_222688

-- Define the set of words possible using Gobbleian language rules
def num_words_possible (alphabet_size : ℕ) (max_length : ℕ) : ℕ :=
  let counts := List.range (max_length + 1)
  counts.sum (λ n, if n = 0 then 0 else alphabet_size ^ n)

-- Problem conditions
def alphabet_size : ℕ := 7
def max_length : ℕ := 4

-- Expected answer
def expected_total_words : ℕ := 2800

-- The proof statement
theorem gobbleian_words_count :
  num_words_possible alphabet_size max_length = expected_total_words := by
  sorry

end gobbleian_words_count_l222_222688


namespace sum_of_digits_base8_888_l222_222373

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222373


namespace opposite_neg_half_l222_222746

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222746


namespace find_a_l222_222240

theorem find_a (a x_0 : ℝ) (h_tangent: (ax_0^3 + 1 = x_0) ∧ (3 * a * x_0^2 = 1)) : a = 4 / 27 :=
sorry

end find_a_l222_222240


namespace expr_undefined_iff_l222_222013

theorem expr_undefined_iff (b : ℝ) : ¬ ∃ y : ℝ, y = (b - 1) / (b^2 - 9) ↔ b = -3 ∨ b = 3 :=
by 
  sorry

end expr_undefined_iff_l222_222013


namespace balloons_cost_indeterminate_l222_222536

theorem balloons_cost_indeterminate (b1 b2 b3 total : ℕ) (cost : ℕ → ℕ) :
  b1 = 5 ∧ b2 = 6 ∧ b3 = 7 ∧ total = b1 + b2 + b3 ∧ total = 18 → 
  ∃ (p : ℕ), cost p = "indeterminate" :=
by
  assume h,
  have p_per_balloon : ∀ cost, cost (b1 + b2 + b3) = cost 18 → cost = "indeterminate",
  sorry

end balloons_cost_indeterminate_l222_222536


namespace baseball_to_football_ratio_l222_222131

theorem baseball_to_football_ratio (total_cards : ℕ) (baseball_cards : ℕ) (football_cards : ℕ)
  (h_total : total_cards = 125)
  (h_baseball : baseball_cards = 95)
  (h_football : football_cards = total_cards - baseball_cards) :
  (baseball_cards : ℚ) / football_cards = 19 / 6 :=
by
  sorry

end baseball_to_football_ratio_l222_222131


namespace sqrt_multiplication_correctness_l222_222815

theorem sqrt_multiplication_correctness : 
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 :=
by 
  rw [Real.mul_self_sqrt (by norm_num : 0 ≤ 2), 
      Real.mul_self_sqrt (by norm_num : 0 ≤ 3), 
      Real.sqrt_mul (by norm_num : 0 ≤ 2) (by norm_num : 0 ≤ 3)]
  norm_num
  sorry

end sqrt_multiplication_correctness_l222_222815


namespace sum_of_digits_base8_888_l222_222405

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222405


namespace period_of_y_max_value_of_y_no_central_symmetry_no_transform_l222_222541

def f (x : ℝ) := Real.sin (x + Real.pi / 4)
def g (x : ℝ) := Real.cos (x - Real.pi / 4)
def y (x : ℝ) := f x * g x

theorem period_of_y : ∀ x, y (x + Real.pi) = y x := by
  -- proof needed
  sorry

theorem max_value_of_y : ∀ x, y x ≤ 1 := by
  -- proof needed
  sorry

theorem no_central_symmetry : ¬ (∀ x, y (2 * (Real.pi / 4) - x) = - y x) := by
  -- proof needed
  sorry

theorem no_transform : ¬ (∀ x, f (x - Real.pi / 2) = g x) := by
  -- proof needed
  sorry

end period_of_y_max_value_of_y_no_central_symmetry_no_transform_l222_222541


namespace right_triangle_ratio_l222_222634

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end right_triangle_ratio_l222_222634


namespace razorback_tshirt_profit_l222_222715

theorem razorback_tshirt_profit
  (total_tshirts_sold : ℕ)
  (tshirts_sold_arkansas_game : ℕ)
  (money_made_arkansas_game : ℕ) :
  total_tshirts_sold = 163 →
  tshirts_sold_arkansas_game = 89 →
  money_made_arkansas_game = 8722 →
  money_made_arkansas_game / tshirts_sold_arkansas_game = 98 :=
by 
  intros _ _ _
  sorry

end razorback_tshirt_profit_l222_222715


namespace inequality_sum_l222_222545

theorem inequality_sum (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 2 ≤ n) :
  (∑ i in finset.range n, 1 / (a + (i + 1) * b)) < n / (Real.sqrt ((a + 1/2 * b) * (a + (2 * n + 1) / 2 * b))) :=
by
  sorry

end inequality_sum_l222_222545


namespace two_exp_sum_lt_four_l222_222971

theorem two_exp_sum_lt_four (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : 
  2^a + 2^b + 2^c < 4 := 
  sorry

end two_exp_sum_lt_four_l222_222971


namespace rationalize_denominator_min_value_l222_222187

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222187


namespace Jori_water_left_l222_222135

theorem Jori_water_left (a b : ℚ) (h1 : a = 7/2) (h2 : b = 7/4) : a - b = 7/4 := by
  sorry

end Jori_water_left_l222_222135


namespace lying_dwarf_possible_numbers_l222_222430

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l222_222430


namespace eliminate_denominator_l222_222706

theorem eliminate_denominator (x : ℝ) : 6 - (x - 2) / 2 = x → 12 - x + 2 = 2 * x :=
by
  intro h
  sorry

end eliminate_denominator_l222_222706


namespace ellipse_chord_equation_l222_222094

theorem ellipse_chord_equation 
  (chord_bisected : (∃ A B : ℝ × ℝ, ((A.1^2 / 36) + (A.2^2 / 9) = 1 ∧ (B.1^2 / 36) + (B.2^2 / 9) = 1) ∧ (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2)) : 
  ∃ (a b c : ℝ), a * 4 + b * 2 + c = 0 ∧  a = 1 ∧ b = 2 ∧ c = -8 :=
by
  sorry

end ellipse_chord_equation_l222_222094


namespace sum_of_digits_base_8_rep_of_888_l222_222347

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222347


namespace relationship_between_a_b_c_l222_222022

theorem relationship_between_a_b_c (a b c : ℕ) (h1 : a = 2^40) (h2 : b = 3^32) (h3 : c = 4^24) : a < c ∧ c < b := by
  -- Definitions as per conditions
  have ha : a = 32^8 := by sorry
  have hb : b = 81^8 := by sorry
  have hc : c = 64^8 := by sorry
  -- Comparisons involving the bases
  have h : 32 < 64 := by sorry
  have h' : 64 < 81 := by sorry
  -- Resultant comparison
  exact ⟨by sorry, by sorry⟩

end relationship_between_a_b_c_l222_222022


namespace prob_at_least_one_first_class_expected_daily_profit_production_increase_decision_l222_222454

section 
open ProbabilityTheory 

-- Given conditions
def prob_first_class : ℚ := 0.5
def prob_second_class : ℚ := 0.4
def prob_third_class : ℚ := 0.1

def profit_first_class : ℚ := 0.8
def profit_second_class : ℚ := 0.6
def profit_third_class : ℚ := -0.3

def daily_output : ℕ := 2

-- Proof statements
theorem prob_at_least_one_first_class : 
  let prob_event := (prob_first_class * prob_first_class) + 
                    2 * (prob_first_class * (1 - prob_first_class))
  in prob_event = 0.75 := 
  by sorry

theorem expected_daily_profit : 
  let exp_profit := -0.6 * (prob_third_class ^ 2) + 0.3 * (2 * prob_second_class * prob_third_class) +
                    0.5 * (2 * prob_first_class * prob_third_class) + 1.2 * (prob_second_class ^ 2) +
                    1.4 * (2 * prob_first_class * prob_second_class) + 1.6 * (prob_first_class ^ 2)
  in exp_profit = 1.22 :=
  by sorry

theorem production_increase_decision: 
  let avg_profit_per_unit := 1.22 / daily_output
  let net_profit (n : ℕ) := avg_profit_per_unit * n - (n - log (n))
  in ∀ n : ℕ, net_profit n ≤ 0 := 
  by sorry

end

end prob_at_least_one_first_class_expected_daily_profit_production_increase_decision_l222_222454


namespace perpendicular_line_equation_l222_222528

theorem perpendicular_line_equation :
  (∃ (x y: ℝ), x + y - 3 = 0 ∧ 2 * x - y = 0 ∧ (line_equation : ℝ → ℝ → Prop) (x_intersection = 1) (y_intersection = 2) 
  (∀ k, line_equation k -2 = false) :
  (∀ x y : ℝ, line_equation x y ↔ x - 2 * y + 3 = 0) := sorry

end perpendicular_line_equation_l222_222528


namespace quadrilateral_identity_l222_222862

variable (A B C D I K L M N : Point)
variable (AB BC CD DA IK IM IL IN : ℝ)

-- Axioms or conditions
axiom quadrilateral_circumscribed : circumscribed_quadrilateral A B C D I
axiom MidK : midpoint K A B
axiom MidL : midpoint L B C
axiom MidM : midpoint M C D
axiom MidN : midpoint N D A
axiom given_condition : AB * CD = 4 * IK * IM

-- The theorem to be proved
theorem quadrilateral_identity : BC * AD = 4 * IL * IN :=
sorry

end quadrilateral_identity_l222_222862


namespace min_value_one_over_a_plus_one_over_b_l222_222018

theorem min_value_one_over_a_plus_one_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hgeom : real.sqrt 2 = real.sqrt (2^a * 2^b)) : 
    real.sqrt 2 = real.sqrt (2^a * 2^b) → a + b = 1 → (∀ a b : ℝ, 0 < a → 0 < b → a + b = 1 → 4 = 1/a + 1/b) :=
sorry

end min_value_one_over_a_plus_one_over_b_l222_222018


namespace angle_between_diagonals_of_regular_decagon_l222_222290

theorem angle_between_diagonals_of_regular_decagon :
  let vertices := {D1, D2, D3, D4, D5, D6, D7, D8, D9, D10}
  let regular_decagon (v : set ℝ) := v.card = 10 ∧ ∃ (O : ℝ), ∀ d ∈ v, ∥d - O∥ = ∥O∥
  let central_angle (n : ℕ) := 360 / n
  let subtended_angle (d1 d2 : ℝ) := (2 : ℕ) * (central_angle 10)
  let angle_between_diagonals := (1 / 2 : ℝ) * (subtended_angle D1 D3 + subtended_angle D2 D5)
  ∀ (D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ), 
    regular_decagon vertices →
    angle_between_diagonals = 90 :=
by
  sorry

end angle_between_diagonals_of_regular_decagon_l222_222290


namespace solve_for_x_l222_222449

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l222_222449


namespace sum_of_base_8_digits_888_l222_222339

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222339


namespace part_II_l222_222656

-- Define the given functions f and g
def f (x : ℝ) := x^2 + x
def g (x : ℝ) := -x^2 + x

-- Define h based on the above functions and a parameter λ
def h (x : ℝ) (λ : ℝ) := g x - λ * f x + 1

-- Prove that the range of λ for h(x) to be increasing on [-1, 1] is -3 <= λ <= -1/3
theorem part_II (λ : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (h (x + 1) λ ≥ h x λ)) ↔ (-3 ≤ λ ∧ λ ≤ -1/3) :=
begin
  sorry
end

end part_II_l222_222656


namespace total_students_l222_222775

theorem total_students (N : ℕ) (num_provincial : ℕ) (sample_provincial : ℕ) 
(sample_experimental : ℕ) (sample_regular : ℕ) (sample_sino_canadian : ℕ) 
(ratio : ℕ) 
(h1 : num_provincial = 96) 
(h2 : sample_provincial = 12) 
(h3 : sample_experimental = 21) 
(h4 : sample_regular = 25) 
(h5 : sample_sino_canadian = 43) 
(h6 : ratio = num_provincial / sample_provincial) 
(h7 : ratio = 8) 
: N = ratio * (sample_provincial + sample_experimental + sample_regular + sample_sino_canadian) := 
by 
  sorry

end total_students_l222_222775


namespace problem_range_and_length_l222_222994

def f (x : ℝ) := 4 * sin x * sin (x + real.pi / 3)

noncomputable def A : ℝ := real.pi / 3
noncomputable def b : ℝ := 2
noncomputable def c : ℝ := 4
noncomputable def AD : ℝ := sqrt 7

theorem problem_range_and_length :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ real.pi / 2 → (0 ≤ f x ∧ f x ≤ 3)) ∧
  (∀ x : ℝ, f x ≤ f A) ∧
  (b = 2) ∧
  (c = 4) ∧
  (AD = sqrt 7) :=
by
  sorry

end problem_range_and_length_l222_222994


namespace prism_volume_l222_222556

-- Define the basic setup
structure TriangularPrism :=
  (a h : ℝ) -- edge length of the base and height of the prism

structure Sphere :=
  (R : ℝ) -- radius of the sphere

-- Define the conditions and the equivalent proof problem
theorem prism_volume (a h : ℝ) (R : ℝ) 
  (vertices_on_sphere : all_vertices_on_sphere)
  (same_size_circles : intersecting_circles_same_size)
  (surface_area : 4 * π * R^2 = 20 * π)
  (eq1 : (a / 2)^2 + (h / 2)^2 = (a / √3)^2)
  (eq2 : (h / 2)^2 + (a / √3)^2 = R^2) :
  ∃ V, V = 6 * √3 :=
by
  -- skeleton of the proof
  sorry

end prism_volume_l222_222556


namespace rhombus_perimeter_l222_222865

-- Define the lengths of the diagonals
def d1 : ℝ := 5  -- Length of the first diagonal
def d2 : ℝ := 12 -- Length of the second diagonal

-- Calculate the perimeter and state the theorem
theorem rhombus_perimeter : ((d1 / 2)^2 + (d2 / 2)^2).sqrt * 4 = 26 := by
  -- Sorry is placed here to denote the proof
  sorry

end rhombus_perimeter_l222_222865


namespace prove_f_pi_over_4_l222_222573

-- Define the given conditions
def given_cos_phi : ℝ := -4 / 5
def given_sin_phi : ℝ := 3 / 5
def given_axes_distance : ℝ := π / 2

-- Define the function f(x) = sin(ω x + φ)
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Define the period from the condition
def period_distance_equals_pi_over_2 (ω : ℝ) : Prop :=
  (2 * π / ω) = 2 * (π / 2)

-- The theorem to prove
theorem prove_f_pi_over_4 :
  period_distance_equals_pi_over_2 2 →
  f (π / 4) 2 given_cos_phi = -4 / 5 :=
by sorry

end prove_f_pi_over_4_l222_222573


namespace sum_of_base_8_digits_888_l222_222343

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222343


namespace gcd_221_195_l222_222729

-- Define the two numbers
def a := 221
def b := 195

-- Statement of the problem: the gcd of a and b is 13
theorem gcd_221_195 : Nat.gcd a b = 13 := 
by
  sorry

end gcd_221_195_l222_222729


namespace value_of_120abc_l222_222990

theorem value_of_120abc (a b c d : ℝ) 
    (h1 : 10 * a = 20) 
    (h2 : 6 * b = 20) 
    (h3 : c^2 + d^2 = 50) 
    : 120 * a * b * c = 800 * real.sqrt (50 - d^2) :=
by
  sorry

end value_of_120abc_l222_222990


namespace cube_root_of_6880_l222_222981

theorem cube_root_of_6880 :
  ∀ (c₁ : ℝ) (c₂ : ℝ),
  c₁ = 4.098 →
  c₂ = 1.902 →
  ∛6880 = 19.02 :=
by
  intros c₁ c₂ hc₁ hc₂
  sorry

end cube_root_of_6880_l222_222981


namespace local_minimum_bounded_area_l222_222500

noncomputable def f (x : ℝ) : ℝ := x * (1 - x^2) * Real.exp (x^2)

theorem local_minimum : f (-1 / Real.sqrt 2) = -Real.sqrt (Real.exp 1) / (2 * Real.sqrt 2) :=
sorry

theorem bounded_area : (∫ x in -1..1, f x) = Real.exp 1 - 2 :=
sorry

end local_minimum_bounded_area_l222_222500


namespace opposite_neg_one_half_l222_222761

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222761


namespace solve_for_x_l222_222448

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l222_222448


namespace sum_of_digits_base_8_888_is_13_l222_222397

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222397


namespace rationalize_denominator_min_value_l222_222185

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222185


namespace cos_beta_gamma_extrema_l222_222562

variable (α β γ k : ℝ)

theorem cos_beta_gamma_extrema (h0 : 0 < k) (h1 : k < 2)
  (h2 : cos α + k * cos β + (2 - k) * cos γ = 0)
  (h3 : sin α + k * sin β + (2 - k) * sin γ = 0) :
  -1 ≤ cos (β - γ) ∧ cos (β - γ) ≤ -0.5 :=
by
  sorry

end cos_beta_gamma_extrema_l222_222562


namespace reflected_ray_equation_l222_222474

noncomputable def symmetric_point (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ :=
sorry -- Definition of the symmetric point

theorem reflected_ray_equation :
  let P : ℝ × ℝ := (2, 3)
  let L : ℝ × ℝ → Prop := λ p, p.1 + p.2 = -1
  let Q : ℝ × ℝ := (1, 1)
  ∃ a b : ℝ, symmetric_point P L = (a, b) ∧ 4 * p.1 - 5 * p.2 + 1 = 0 :=
begin
  sorry -- The proof goes here
end

end reflected_ray_equation_l222_222474


namespace min_value_rationalize_sqrt_denominator_l222_222195

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222195


namespace Jack_can_sail_4_days_l222_222242

def wind_speeds : list ℕ := [15, 18, 17, 21, 22, 25, 16]
def sail_condition (speed : ℕ) : Prop := speed < 20

theorem Jack_can_sail_4_days :
  list.countp sail_condition wind_speeds = 4 :=
by sorry

end Jack_can_sail_4_days_l222_222242


namespace ratio_of_cost_to_selling_price_l222_222691

-- Define the conditions in Lean
variable (C S : ℝ) -- C is the cost price per pencil, S is the selling price per pencil
variable (h : 90 * C - 40 * S = 90 * S)

-- Define the statement to be proved
theorem ratio_of_cost_to_selling_price (C S : ℝ) (h : 90 * C - 40 * S = 90 * S) : (90 * C) / (90 * S) = 13 :=
by
  sorry

end ratio_of_cost_to_selling_price_l222_222691


namespace sum_of_digits_base8_888_l222_222332

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222332


namespace sum_of_transformed_roots_equals_one_l222_222668

theorem sum_of_transformed_roots_equals_one 
  {α β γ : ℝ} 
  (hα : α^3 - α - 1 = 0) 
  (hβ : β^3 - β - 1 = 0) 
  (hγ : γ^3 - γ - 1 = 0) : 
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
sorry

end sum_of_transformed_roots_equals_one_l222_222668


namespace no_distinct_nat_numbers_eq_l222_222172

theorem no_distinct_nat_numbers_eq (x y z t : ℕ) (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t) 
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) : x ^ x + y ^ y ≠ z ^ z + t ^ t := 
by 
  sorry

end no_distinct_nat_numbers_eq_l222_222172


namespace proof_problem_l222_222537

open Real

variables (x y z : ℝ)
def AB : ℝ × ℝ × ℝ := (1, 5, -2)
def BC : ℝ × ℝ × ℝ := (3, 1, z)
def BP : ℝ × ℝ × ℝ := (x - 1, y, -3)

-- Conditions
def AB_perp_BC : Prop := (fst AB * fst BC + snd AB * snd BC + trd AB * trd BC = 0)
def BP_perp_AB : Prop := ((fst BP) * (fst AB) + (snd BP) * (snd AB) + (trd BP) * (trd AB) = 0)
def BP_perp_BC : Prop := ((fst BP) * (fst BC) + (snd BP) * (snd BC) + (trd BP) * (trd BC) = 0)

theorem proof_problem
  (h₁ : AB_perp_BC)
  (h₂ : BP_perp_AB)
  (h₃ : BP_perp_BC) :
  (x + y + z = 53 / 7) :=
sorry

end proof_problem_l222_222537


namespace min_positive_period_of_f_max_min_of_f_on_interval_l222_222991

-- Define the given function f
def f (x : ℝ) : ℝ := 2 * Real.sin (π - x) * Real.cos x + Real.cos (2 * x)

-- Definition of the minimum positive period in terms of a statement
theorem min_positive_period_of_f : 
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π :=
sorry

-- Defining the interval for x
def interval := Set.Icc (π / 4) (π / 2)

-- Finding the maximum and minimum values on the given interval
theorem max_min_of_f_on_interval :
  ∃ (max_val min_val : ℝ), 
  (∀ x ∈ interval, f(x) ≤ max_val) ∧ (∀ x ∈ interval, f(x) ≥ min_val) ∧ 
  max_val = 1 ∧ min_val = -1 :=
sorry

end min_positive_period_of_f_max_min_of_f_on_interval_l222_222991


namespace three_digit_number_l222_222950

theorem three_digit_number (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9) 
  (h : 100 * a + 10 * b + c = 3 * (10 * (a + b) + c)) : 100 * a + 10 * b + c = 135 :=
  sorry

end three_digit_number_l222_222950


namespace monotonic_decreasing_interval_l222_222733

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (2 * x)

theorem monotonic_decreasing_interval :
  { x : ℝ | x < -1 / 2 } = { x : ℝ | ∃ y, f' y < 0} :=
by
  -- Proof required
  sorry

end monotonic_decreasing_interval_l222_222733


namespace max_pieces_l222_222244

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.nodup digits

def five_digits (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem max_pieces :
  ∀ (n : ℕ) (КУСОК ПИРОГ : ℕ), 
    ПИРОГ = КУСОК * n → 
    five_digits ПИРОГ → 
    distinct_digits ПИРОГ → 
    n ≤ 7 :=
begin
  intros n КУСОК ПИРОГ h1 h2 h3,
  -- skip the proof
  sorry
end

end max_pieces_l222_222244


namespace sum_of_digits_base8_888_l222_222371

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222371


namespace length_MN_l222_222554

variables (A B C D M N : Type) 
variables [Trapezoid A B C D]
variables [Base AD : length = 3]
variables [Base BC : length = 18]
variables [Diagonal AC]
variables [Ratio AM MC : 1 2]
variables [Line M : parallel_to_base]
variables [Intersection N : line_parallel_to_base_intersects_diagonal]

theorem length_MN :
  length_segment MN = 4 :=
by sorry

end length_MN_l222_222554


namespace evaluate_arithmetic_expression_l222_222523

theorem evaluate_arithmetic_expression :
  let A := (finset.range 100).sum (λ n, 2001 + n)
  let B := (finset.range 100).sum (λ n, 201 + n)
  (A - B + 1500) = 181500 :=
by
  sorry

end evaluate_arithmetic_expression_l222_222523


namespace simplify_expression_l222_222227

theorem simplify_expression (a: ℤ) (h₁: a ≠ 0) (h₂: a ≠ 1) (h₃: a ≠ -3) :
  (2 * a = 4) → a = 2 :=
by
  sorry

end simplify_expression_l222_222227


namespace factorial_div_42_40_l222_222509

theorem factorial_div_42_40 : (42! / 40! = 1722) := 
by 
  sorry

end factorial_div_42_40_l222_222509


namespace volume_of_locations_eq_27sqrt6pi_over_8_l222_222835

noncomputable def volumeOfLocationSet : ℝ :=
  let sqrt2_inv := 1 / (2 * Real.sqrt 2)
  let points := [ (sqrt2_inv, sqrt2_inv, sqrt2_inv),
                  (sqrt2_inv, sqrt2_inv, -sqrt2_inv),
                  (sqrt2_inv, -sqrt2_inv, sqrt2_inv),
                  (-sqrt2_inv, sqrt2_inv, sqrt2_inv) ]
  let condition (x y z : ℝ) : Prop :=
    4 * (x^2 + y^2 + z^2) + 3 / 2 ≤ 15
  let r := Real.sqrt (27 / 8)
  let volume := (4/3) * Real.pi * r^3
  volume

theorem volume_of_locations_eq_27sqrt6pi_over_8 :
  volumeOfLocationSet = 27 * Real.sqrt 6 * Real.pi / 8 :=
sorry

end volume_of_locations_eq_27sqrt6pi_over_8_l222_222835


namespace sum_of_digits_base_8_888_is_13_l222_222400

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222400


namespace maximum_pieces_is_seven_l222_222254

noncomputable def max_pieces (PIE PIECE : ℕ) (n : ℕ) : Prop :=
  PIE = PIECE * n ∧ natDigits 10 PIE = List.nodup (natDigits 10 PIE) ∧ natDigits 10 PIECE = List.nodup (natDigits 10 PIECE)

theorem maximum_pieces_is_seven :
  max_pieces 95207 13601 7 :=
sorry

end maximum_pieces_is_seven_l222_222254


namespace arithmetic_sequence_value_l222_222639

theorem arithmetic_sequence_value (a : ℕ → ℕ) (m : ℕ) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 4) 
  (h_a5 : a 5 = m) 
  (h_a7 : a 7 = 16) : 
  m = 10 := 
by
  sorry

end arithmetic_sequence_value_l222_222639


namespace part1_correct_part2_correct_l222_222852

-- Define the probabilities of selecting from each workshop
def P_A1 : ℝ := 0.25
def P_A2 : ℝ := 0.35
def P_A3 : ℝ := 0.4

-- Define the probabilities of a defective product given it was produced in each workshop
def P_B_A1 : ℝ := 0.05
def P_B_A2 : ℝ := 0.04
def P_B_A3 : ℝ := 0.02

-- Compute the total probability of selecting a defective product using the law of total probability
def P_B : ℝ := P_A1 * P_B_A1 + P_A2 * P_B_A2 + P_A3 * P_B_A3

-- Compute the conditional probability that a defective product was produced in workshop A using Bayes' theorem
def P_A1_B : ℝ := (P_A1 * P_B_A1) / P_B

-- The theorem to check if computations are correct
theorem part1_correct : P_B = 0.0345 := by sorry
theorem part2_correct : P_A1_B = 0.36 := by sorry

end part1_correct_part2_correct_l222_222852


namespace area_of_triangle_l222_222730

-- Define the conditions
def hypotenuse : ℝ := 20
def angle_A : ℝ := 30 * Real.pi / 180 -- Convert degrees to radians

-- Theorem: Area of a right triangle with hypotenuse and given angle is 50√3 square inches
theorem area_of_triangle : 
  ∀ {a b c : ℝ}, 
  a = 10 →
  b = 10 * Real.sqrt 3 →
  c = hypotenuse →
  angle_A = 30 * Real.pi / 180 →
  ∃ (A : ℝ), A = 1 / 2 * a * b ∧ A = 50 * Real.sqrt 3 :=
by
  -- Provide a proof here
  sorry

end area_of_triangle_l222_222730


namespace min_ABCD_sum_correct_ABCD_l222_222207

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222207


namespace sum_of_coefficients_l222_222279

theorem sum_of_coefficients {x : ℝ} (hx : x = 1) : 
  let expr := (Real.sqrt x + 2 / (3 * x)) ^ 4 in 
  let sum_coeff := (1 + 2) ^ 4 in 
  sum_coeff = 81 := 
by 
  sorry

end sum_of_coefficients_l222_222279


namespace sum_of_base8_digits_888_l222_222355

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222355


namespace find_a_l222_222557

-- Definitions
variable (a : ℝ)
variable (PA PB PC : ℝ)
variable (S : ℝ := 9 * Real.pi)

-- Conditions
def are_mutually_perpendicular (PA PB PC : ℝ) : Prop := True -- There isn't a direct way to specify perpendicularity for lengths in Lean, typically handled with vectors
def length_condition_one : PA = 2 * a := rfl
def length_condition_two : PB = 2 * a := rfl
def length_condition_three : PC = a := rfl
def surface_area_condition : S = 9 * Real.pi := rfl

-- The theorem to prove
theorem find_a
  (PA PB PC : ℝ)
  (h1: are_mutually_perpendicular PA PB PC)
  (h2: PA = 2 * a)
  (h3: PB = 2 * a)
  (h4: PC = a)
  (h5: S = 9 * Real.pi) : a = 1 := sorry

end find_a_l222_222557


namespace lollipop_cost_is_1_50_l222_222908

-- Define all given conditions
def lollipop_cost (l : ℝ) :=
  let gummies_cost := 2 * 2 in
  let total_cost := 15 - 5 in
  let lollipops_cost := total_cost - gummies_cost in
  l = lollipops_cost / 4

-- The statement we need to prove
theorem lollipop_cost_is_1_50 : lollipop_cost 1.50 :=
by 
  unfold lollipop_cost 
  sorry

end lollipop_cost_is_1_50_l222_222908


namespace rationalize_denominator_l222_222215

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222215


namespace correct_statements_given_conditions_l222_222053

-- Defining the conditions as Lean definitions
def condition1 : Prop := ∀ (A B C D : Point), ¬coplanar A B C D → (¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D)
def condition2 : Prop := ∀ (P Q R : Point), distinct P Q R → (∀ (π₁ π₂ : Plane), π₁ ≠ π₂ → ((P ∈ π₁) ∧ (P ∈ π₂) ∧ (Q ∈ π₁) ∧ (Q ∈ π₂) ∧ (R ∈ π₁) ∧ (R ∈ π₂)) → false)
def condition3 : Prop := ∀ (ℓ₁ ℓ₂ : Line), (¬ (∃ (P : Point), (P ∈ ℓ₁) ∧ (P ∈ ℓ₂))) → skew ℓ₁ ℓ₂
def condition4 : Prop := ∀ (ℓ₁ ℓ₂ ℓ₃ : Line), (skew ℓ₁ ℓ₂) ∧ (skew ℓ₁ ℓ₃) → skew ℓ₂ ℓ₃
def condition5 : Prop := ∀ (ℓ₁ ℓ₂ ℓ₃ : Line), (skew ℓ₂ ℓ₃) ∧ (∃ (P : Point), (P ∈ ℓ₁) ∧ (P ∈ ℓ₂)) → (∃ (π₁ π₂ : Plane), π₁ ≠ π₂ ∧ (P ∈ π₁) ∧ (P ∈ π₂))

-- Statement concluding the correct answers given the conditions
theorem correct_statements_given_conditions
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) :
  {1, 5} :=
begin
  sorry
end

end correct_statements_given_conditions_l222_222053


namespace rationalize_denominator_and_min_sum_l222_222181

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222181


namespace find_x_l222_222732

theorem find_x (x : ℚ) : (8 + 12 + 24) / 3 = (16 + x) / 2 → x = 40 / 3 :=
by
  intro h
  sorry

end find_x_l222_222732


namespace part_a_part_b_part_c_part_d_l222_222702

-- Part a
theorem part_a (x : ℝ) : 
  (5 / x - x / 3 = 1 / 6) ↔ x = 6 := 
by
  sorry

-- Part b
theorem part_b (a : ℝ) : 
  ¬ ∃ a, (1 / 2 + a / 4 = a / 4) := 
by
  sorry

-- Part c
theorem part_c (y : ℝ) : 
  (9 / y - y / 21 = 17 / 21) ↔ y = 7 := 
by
  sorry

-- Part d
theorem part_d (z : ℝ) : 
  (z / 8 - 1 / z = 3 / 8) ↔ z = 4 := 
by
  sorry

end part_a_part_b_part_c_part_d_l222_222702


namespace rationalize_denominator_l222_222217

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222217


namespace cos_half_pi_plus_alpha_l222_222016

theorem cos_half_pi_plus_alpha (α : ℝ) (h : sin (-α) = sqrt 5 / 3) : cos (π / 2 + α) = sqrt 5 / 3 :=
sorry

end cos_half_pi_plus_alpha_l222_222016


namespace three_digit_integers_count_l222_222078

theorem three_digit_integers_count :
  let digits := {1, 3, 3, 4, 4, 4, 7}
  in 
  (∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * 100 + b * 10 + c ≥ 100 ∧ 
                a * 100 + b * 10 + c ≤ 999 → True) ∧
  (∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
                ((a = b ∧ b ≠ c) ∨ (a ≠ b ∧ b = c) ∨ (a = c ∧ b ≠ c)) ∧ 
                a * 100 + b * 10 + c ≥ 100 ∧ 
                a * 100 + b * 10 + c ≤ 999 → True) ∧
  (∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
                a = b ∧ b = c ∧ 
                a * 100 + b * 10 + c ≥ 100 ∧ 
                a * 100 + b * 10 + c ≤ 999 → True) →
  43
:= by
  sorry

end three_digit_integers_count_l222_222078


namespace min_value_of_reciprocal_sum_l222_222020

theorem min_value_of_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : Float.ofReal (Real.sqrt (2^a * 2^b)) = Float.ofReal (Real.sqrt 2)) : 
  minValue (λ x, x = (1/a + 1/b)) = 4 :=
by
  -- Proof omitted
  sorry

end min_value_of_reciprocal_sum_l222_222020


namespace nadia_played_minutes_l222_222681

-- conditions
def mistakes_per_notes (x mistakes notes : ℕ) := x mistakes * 40 = 3 * notes
def notes_per_minute (x notes minute : ℕ) := x notes * 1 = 60 * minute
def total_mistakes (x total mistakes minute : ℕ) := x total * 8 * 4.5 = mistakes * minute

-- question
theorem nadia_played_minutes
  (h1 : mistakes_per_notes 3 40)
  (h2 : notes_per_minute 60 1)
  (h3 : total_mistakes 36 4.5)
  : minutes = 8 := 
by
  sorry

end nadia_played_minutes_l222_222681


namespace lying_dwarf_number_is_possible_l222_222433

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l222_222433


namespace find_length_of_ab_l222_222623

noncomputable def length_of_ab (A B C : ℝ) (cos_A : ℝ) (cos_B : ℝ) (a_c: ℝ) : ℝ :=
let sin_A := real.sqrt(1 - cos_A^2),
    sin_B := real.sqrt(1 - cos_B^2),
    sin_C := sin_A * cos_B + cos_A * sin_B in
  a_c * sin_C / sin_B

theorem find_length_of_ab (A B C : ℝ) (h1 : cos A = 3 / 5) (h2 : cos B = 5 / 13) (h3 : AC = 3) :
  length_of_ab A B C (3 / 5) (5 / 13) 3 = 14 / 5 := by
  sorry

end find_length_of_ab_l222_222623


namespace short_answer_question_time_l222_222659

-- Definitions from the conditions
def minutes_per_paragraph := 15
def minutes_per_essay := 60
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15
def total_minutes := 4 * 60

-- Auxiliary calculations
def total_minutes_essays := num_essays * minutes_per_essay
def total_minutes_paragraphs := num_paragraphs * minutes_per_paragraph
def total_minutes_used := total_minutes_essays + total_minutes_paragraphs

-- The time per short-answer question is 3 minutes
theorem short_answer_question_time (x : ℕ) : (total_minutes - total_minutes_used) / num_short_answer_questions = 3 :=
by
  -- x is defined as the time per short-answer question
  let x := (total_minutes - total_minutes_used) / num_short_answer_questions
  have time_for_short_answer_questions : total_minutes - total_minutes_used = 45 := by sorry
  have time_per_short_answer_question : 45 / num_short_answer_questions = 3 := by sorry
  have x_equals_3 : x = 3 := by sorry
  exact x_equals_3

end short_answer_question_time_l222_222659


namespace opposite_neg_one_half_l222_222764

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222764


namespace sequence_properties_l222_222034

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)
  (h_arith : arithmetic_seq a 2)
  (h_sum_prop : sum_seq a S)
  (h_ratio : ∀ n, S (2 * n) / S n = 4)
  (b : ℕ → ℤ) (T : ℕ → ℤ)
  (h_b : ∀ n, b n = a n * 2 ^ (n - 1))

-- Prove the sequences
theorem sequence_properties :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, T n = (2 * n - 3) * 2^n + 3) :=
by
  sorry

end sequence_properties_l222_222034


namespace not_P_4_given_not_P_5_l222_222473

-- Define the proposition P for natural numbers
def P (n : ℕ) : Prop := sorry

-- Define the statement we need to prove
theorem not_P_4_given_not_P_5 (h1 : ∀ k : ℕ, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 := by
  sorry

end not_P_4_given_not_P_5_l222_222473


namespace even_product_probability_l222_222522

theorem even_product_probability :
  let chips := [1, 2, 3, 4]
  let total_outcomes := chips.product(chips).length
  let favorable_outcomes := (chips.product(chips)).filter (λ p, (p.fst * p.snd) % 2 = 0)
  (favorable_outcomes.length : ℚ) / total_outcomes = 3 / 4 :=
sorry

end even_product_probability_l222_222522


namespace f_alpha_value_l222_222023

theorem f_alpha_value (α : ℝ)
  (h1 : α ∈ (3 * Real.pi / 2, 2 * Real.pi))  -- α in third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  let f := λ α : ℝ, (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.tan (-α + Real.pi)) /
                    (-Real.tan (-α - Real.pi) * Real.cos (Real.pi / 2 - α)) in
  f α = 2 * Real.sqrt 6 / 5 :=
by
  sorry

end f_alpha_value_l222_222023


namespace pipe_stack_height_4_layers_l222_222447

-- Define the radius and conditions described in the problem
def pipe_radius := 10 -- radius in cm
def pipe_diameter := 2 * pipe_radius -- diameter in cm
def equilateral_triangle_height (side : ℝ) := side * (Real.sqrt 3) / 2 -- height of an equilateral triangle calculation

-- Define total height calculation using conditions
noncomputable def total_height (layers : ℕ) : ℝ :=
  let base_height := pipe_radius in
  let additional_height := (layers - 1) * equilateral_triangle_height pipe_diameter in
  base_height + additional_height

-- Theorem to prove the total height for 4 layers is 10 + 30√3 cm
theorem pipe_stack_height_4_layers :
  total_height 4 = 10 + 30 * Real.sqrt 3 :=
sorry

end pipe_stack_height_4_layers_l222_222447


namespace quadrupled_volume_l222_222461

theorem quadrupled_volume {initial_volume : ℕ} (h1 : initial_volume = 5) :
  let factor := 4 in
  let new_volume := initial_volume * (factor ^ 3) in
  new_volume = 320 :=
by
  have h2 : factor = 4 := rfl
  have h3 : factor ^ 3 = 64 := by norm_num
  have h4 : initial_volume = 5 := h1
  have h5 : new_volume = initial_volume * 64 := by rw [← h3, ← h2, h4, nat.mul_comm]
  exact h5.trans (by norm_num)

end quadrupled_volume_l222_222461


namespace sum_of_digits_base_8_rep_of_888_l222_222349

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222349


namespace max_pieces_l222_222247

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.nodup digits

def five_digits (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem max_pieces :
  ∀ (n : ℕ) (КУСОК ПИРОГ : ℕ), 
    ПИРОГ = КУСОК * n → 
    five_digits ПИРОГ → 
    distinct_digits ПИРОГ → 
    n ≤ 7 :=
begin
  intros n КУСОК ПИРОГ h1 h2 h3,
  -- skip the proof
  sorry
end

end max_pieces_l222_222247


namespace sum_of_digits_base8_l222_222392

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222392


namespace normal_distribution_interval_probability_l222_222627

noncomputable def normal_distribution_probability (X : ℝ → ℝ) (μ σ : ℝ) :=
  ∀ a b : ℝ, probability X a b = 
    if a < μ ∧ μ < b then
      (mathieu.standard_normal_distribution (b - μ) - mathieu.standard_normal_distribution (a - μ))
    else if μ ≤ a then
      (mathieu.standard_normal_distribution (b - μ))
    else
      (1 - mathieu.standard_normal_distribution (a - μ))

theorem normal_distribution_interval_probability (X : ℝ) (σ : ℝ) (h1 : prob X (0, 1) = 0.4)
  (h2 : ∀ a b : ℝ, probability X a b = normal_distribution_probability X 1 σ) :
  prob X (0, +∞) = 0.9 :=
by
  sorry

end normal_distribution_interval_probability_l222_222627


namespace bhanu_income_percentage_l222_222901

variable {I P : ℝ}

theorem bhanu_income_percentage (h₁ : 300 = (P / 100) * I)
                                  (h₂ : 210 = 0.3 * (I - 300)) :
  P = 30 :=
by
  sorry

end bhanu_income_percentage_l222_222901


namespace cos_7x_eq_cos_5x_has_7_solutions_l222_222645

open Real

theorem cos_7x_eq_cos_5x_has_7_solutions : 
  ∀ x ∈ Icc 0 pi, 
  (cos (7 * x) = cos (5 * x)) → 
  (∃ k:ℤ, x = k * pi ∧ 0 ≤ k ∧ k ≤ 1) ∨ 
  (∃ k:ℤ, 0 ≤ k ∧ k ≤ 6 ∧ x = k * pi / 6) :=
sorry

end cos_7x_eq_cos_5x_has_7_solutions_l222_222645


namespace translation_correct_l222_222312

variable (x : ℝ)

def original_function : ℝ → ℝ := λ x, sin (2 * x)
def translated_left_function : ℝ → ℝ := λ x, sin (2 * (x + 1))
def final_function : ℝ → ℝ := λ x, sin (2 * x + 2) + 1

theorem translation_correct :
  final_function x = sin(2 * x + 2) + 1 := 
by
  sorry

end translation_correct_l222_222312


namespace rectangle_area_is_11_l222_222911

-- Define the conditions
constants (ABCD : Type) -- Represent the rectangle ABCD as a type
constants (smaller_square1 smaller_square2 larger_square : Type) -- Represent the 3 squares
constants (area : ℝ → ℝ) -- Define a area function from side length to area
constants (side_length : Type → ℝ) -- Define a function for getting side lengths

axiom smaller_square_area : ∀ (s : Type), s = smaller_square1 ∨ s = smaller_square2 → area (side_length s) = 1
axiom larger_square_area : area (side_length larger_square) = 9
axiom nonoverlapping : ∀ (s1 s2 : Type), (s1 = smaller_square1 ∨ s1 = smaller_square2 ∨ s1 = larger_square) ∧
                                             (s2 = smaller_square1 ∨ s2 = smaller_square2 ∨ s2 = larger_square) → s1 ≠ s2
axiom correct_side_length : ∀ (s : Type), s = larger_square → side_length s = 3

noncomputable theory

-- The proof
theorem rectangle_area_is_11 : area (10 + 10 + 0) = 11 := by
  sorry

end rectangle_area_is_11_l222_222911


namespace intersection_points_of_inscribed_polygons_l222_222219

theorem intersection_points_of_inscribed_polygons :
  let p6 := (6 : ℕ)
  let p7 := (7 : ℕ)
  let p8 := (8 : ℕ)
  let p9 := (9 : ℕ)
  ∀ (circle : Type) (inscribed : circle → ℕ → Prop),
    (inscribed circle p6) ∧ (inscribed circle p7) ∧ (inscribed circle p8) ∧ (inscribed circle p9) →
    (∀ (m n : ℕ), inscribed circle m → inscribed circle n → (m ≠ n) → (∀ v, ¬ shares_vertex v m n)) →
    (∀ (s1 s2 s3 : ℕ), inscribed circle s1 → inscribed circle s2 → inscribed circle s3 → 
     (s1 ≠ s2) → (s2 ≠ s3) → (s1 ≠ s3) → ∀ pt, ¬ common_intersection pt s1 s2 s3) →
    (number_of_intersections circle p6 p7 p8 p9 = 80) :=
by
  intros _ _ _ _ _
  sorry

end intersection_points_of_inscribed_polygons_l222_222219


namespace temperature_on_tuesday_l222_222718

variable (T W Th F : ℕ)

-- Conditions
def cond1 : Prop := (T + W + Th) / 3 = 32
def cond2 : Prop := (W + Th + F) / 3 = 34
def cond3 : Prop := F = 44

-- Theorem statement
theorem temperature_on_tuesday : cond1 T W Th → cond2 W Th F → cond3 F → T = 38 :=
by
  sorry

end temperature_on_tuesday_l222_222718


namespace pencil_sharpening_and_breaking_l222_222654

/-- Isha's pencil initially has a length of 31 inches. After sharpening, it has a length of 14 inches.
Prove that:
1. The pencil was shortened by 17 inches.
2. Each half of the pencil, after being broken in half, is 7 inches long. -/
theorem pencil_sharpening_and_breaking 
  (initial_length : ℕ) 
  (length_after_sharpening : ℕ) 
  (sharpened_length : ℕ) 
  (half_length : ℕ) 
  (h1 : initial_length = 31) 
  (h2 : length_after_sharpening = 14) 
  (h3 : sharpened_length = initial_length - length_after_sharpening) 
  (h4 : half_length = length_after_sharpening / 2) : 
  sharpened_length = 17 ∧ half_length = 7 := 
by {
  sorry
}

end pencil_sharpening_and_breaking_l222_222654


namespace two_cubic_meters_to_cubic_feet_l222_222597

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end two_cubic_meters_to_cubic_feet_l222_222597


namespace problem_statement_l222_222553

-- Conditions for the sequence {a_n}
def a : ℕ → ℤ
| 0     := 1                  -- Corresponds to a_1 = 1
| 1     := 2                  -- Corresponds to a_2 = 2
| (n+2) := - a n              -- Corresponds to a_{n+2} = -a_n

-- Problem statement
theorem problem_statement :
  (∀ n, b n = a (n + 1) - a n → ∃ r, b 1 = 1 ∧ (∀ n, b (n + 1) = r * b n) ∧ r = -1) ∧
  (∀ n, a n = (1 + (-1 : ℤ)^n) / 2) :=
sorry

end problem_statement_l222_222553


namespace max_daily_profit_l222_222878

noncomputable def daily_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then (5*x/3 - x^3/180)
  else if 12 < x ∧ x ≤ 20 then (1/2 * x)
  else 0

theorem max_daily_profit : ∃ (x : ℝ), 0 < x ∧ x ≤ 20 ∧ 
  (daily_profit x = 100 / 9) :=
begin
  use 10,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { rw daily_profit,
    simp,
    sorry, -- the proof goes here
  }
end

end max_daily_profit_l222_222878


namespace integer_expression_50_integers_l222_222946

theorem integer_expression_50_integers :
  (set.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (factorial (n^2 + n - 1) / ((factorial (n + 1))^n)).denom = 1) (set.Icc 1 60)).card = 50 := sorry

end integer_expression_50_integers_l222_222946


namespace sum_of_digits_base8_888_l222_222329

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222329


namespace basketball_cost_l222_222133

-- Initial conditions
def initial_amount : Nat := 50
def cost_jerseys (n price_per_jersey : Nat) : Nat := n * price_per_jersey
def cost_shorts : Nat := 8
def remaining_amount : Nat := 14

-- Derived total spent calculation
def total_spent (initial remaining : Nat) : Nat := initial - remaining
def known_cost (jerseys shorts : Nat) : Nat := jerseys + shorts

-- Prove the cost of the basketball
theorem basketball_cost :
  let jerseys := cost_jerseys 5 2
  let shorts := cost_shorts
  let total_spent := total_spent initial_amount remaining_amount
  let known_cost := known_cost jerseys shorts
  total_spent - known_cost = 18 := 
by
  sorry

end basketball_cost_l222_222133


namespace paths_from_C_to_D_l222_222599

theorem paths_from_C_to_D : 
  let total_steps := 10
  let up_steps := 3
  let right_steps := 7
  in combinatorial.combinations total_steps up_steps = 120 := by
sorry

end paths_from_C_to_D_l222_222599


namespace euler_totient_problem_l222_222923

open Nat

theorem euler_totient_problem 
  (n : ℕ) : 
  (∃ a : ℤ, a ^ (φ n / 2) ≡ -1 [MOD n]) ↔ 
    n = 4 ∨ (∃ p : ℕ, p.Prime ∧ (n = p ^ k) ∧ 1 ≤ k) ∨ 
      (∃ p : ℕ, p.Prime ∧ (n = 2 * p ^ k) ∧ 1 ≤ k) := 
sorry

end euler_totient_problem_l222_222923


namespace smallest_k_l222_222068

noncomputable def segment_points_partition (points : List (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (M N : List (ℝ × ℝ)), 
    (M.length + N.length = 2020) ∧ 
    (M ∪ N = points) ∧ 
    (M ∩ N = [] ) ∧ 
    (M.sum (λ p, p.2) ≤ k) ∧ 
    (N.sum (λ p, p.1) ≤ k)

theorem smallest_k {points : List (ℝ × ℝ)} :
  (∀ p ∈ points, 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 + p.2 = 1) ∧ points.length = 2020 →
  ∃ k : ℕ, segment_points_partition points k ∧ 
    (∀ m : ℕ, segment_points_partition points m → k ≤ m) :=
sorry

end smallest_k_l222_222068


namespace probability_of_no_math_test_l222_222773

def probability_of_math_test : ℚ := 4 / 7

theorem probability_of_no_math_test : probability_of_math_test = 4 / 7 → (1 - probability_of_math_test) = 3 / 7 :=
by
  assume h : probability_of_math_test = 4 / 7
  sorry

end probability_of_no_math_test_l222_222773


namespace sum_of_digits_base_8_rep_of_888_l222_222346

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222346


namespace smallest_part_division_l222_222604

theorem smallest_part_division (S : ℚ) (P1 P2 P3 : ℚ) (total : ℚ) :
  (P1, P2, P3) = (1, 2, 3) →
  total = 64 →
  S = total / (P1 + P2 + P3) →
  S = 10 + 2/3 :=
by
  sorry

end smallest_part_division_l222_222604


namespace cubert_6880_l222_222978

theorem cubert_6880 (h1: real.cbrt 68.8 = 4.098) (h2: real.cbrt 6.88 = 1.902) : real.cbrt 6880 = 19.02 := by
  sorry

end cubert_6880_l222_222978


namespace min_value_condition_range_of_m_l222_222146

noncomputable def f (x : ℝ) : ℝ := (1 + x) ^ 2 - 2 * Real.log (1 + x)

theorem min_value_condition (x : ℝ) (h : x ∈ Icc 0 1) :
  f x ≤ 4 - 2 * Real.log 2 :=
sorry

theorem range_of_m (x : ℝ) (h : x ∈ Icc 0 1) (m : ℝ) :
  (∃ x₀, x₀ ∈ Icc 0 1 ∧ f x₀ - m ≤ 0) ↔ m ≥ 1 :=
sorry

end min_value_condition_range_of_m_l222_222146


namespace radius_of_semicircle_on_BC_l222_222499

theorem radius_of_semicircle_on_BC (A B C : ℝ) (r_inscribed : ℝ)
  (h_right_angle : ∠ B = π / 2)
  (h_AB_diameter : 2 * r_AB = AB)
  (h_AC_diameter : 2 * r_AC = AC)
  (h_area_AB : π * r_AB^2 / 2 = 10 * π)
  (h_arc_AC : π * r_AC = 10 * π)
  (h_inscribed_circle : r_inscribed = 2)
  (h_ABC : A = B) -- A, B, and C are points forming triangle ABC
  : ∃ r_BC: ℝ, r_BC = 2 * Real.sqrt 30 := by
  sorry

end radius_of_semicircle_on_BC_l222_222499


namespace corrected_mean_l222_222828

-- Definitions based on conditions
def initial_mean : ℝ := 36
def num_observations : ℕ := 50
def incorrect_observation : ℝ := 23
def correct_observation : ℝ := 30

-- Lean statement representing the proof problem
theorem corrected_mean : 
  let initial_sum := initial_mean * num_observations in
  let difference := correct_observation - incorrect_observation in
  let corrected_sum := initial_sum + difference in
  let new_mean := corrected_sum / num_observations in
  new_mean = 36.14 :=
by
  sorry

end corrected_mean_l222_222828


namespace bob_monthly_hours_l222_222930

noncomputable def total_hours_in_month : ℝ :=
  let daily_hours := 10
  let weekly_days := 5
  let weeks_in_month := 4.33
  daily_hours * weekly_days * weeks_in_month

theorem bob_monthly_hours :
  total_hours_in_month = 216.5 :=
by
  sorry

end bob_monthly_hours_l222_222930


namespace max_value_proof_l222_222143

noncomputable def max_value_expr (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * (a - x) * (x + real.sqrt (x^2 + b^2)) + c * x

theorem max_value_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  ∃ x : ℝ, 0 ≤ x ∧ max_value_expr a b c x = (3 - c) / 2 * b^2 + 9 * a^2 / 2 :=
  sorry

end max_value_proof_l222_222143


namespace min_ABCD_sum_correct_ABCD_l222_222205

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222205


namespace distinct_wave_numbers_count_l222_222859

def is_wave_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  n < 100000 ∧ n ≥ 10000 ∧
  digits.length = 5 ∧
  digits.nodup ∧
  (digits.nth 1 > digits.nth 0) ∧
  (digits.nth 1 > digits.nth 2) ∧
  (digits.nth 3 > digits.nth 2) ∧
  (digits.nth 3 > digits.nth 4)

/- Theorem: The number of distinct five-digit "wave numbers" that can be formed using the digits 
1, 2, 3, 4, and 5 without repeating any digits is 16. -/
theorem distinct_wave_numbers_count : finset.card (finset.filter is_wave_number (finset.range 100000)) = 16 :=
  sorry

end distinct_wave_numbers_count_l222_222859


namespace find_percentage_reduced_each_year_l222_222638

-- Define the conditions
def initial_men := 150
def initial_women := 90
def final_population := 140.78099890167377

-- Define the unknown and the equation
noncomputable def percentage_reduced_each_year (x : ℝ) : Prop :=
    let a := initial_men * (1 - x / 100) ^ 2 in
    let b := initial_women in
    let p := (a ^ 2 + b ^ 2) ^ (1 / 2) in
    p = final_population

-- Declare the theorem we want to prove
theorem find_percentage_reduced_each_year : ∃ (x : ℝ), percentage_reduced_each_year x :=
begin
   sorry
end

end find_percentage_reduced_each_year_l222_222638


namespace smallest_sum_ending_2050306_l222_222808

/--
Given nine consecutive natural numbers starting at n,
prove that the smallest sum of these nine numbers ending in 2050306 is 22050306.
-/
theorem smallest_sum_ending_2050306 
  (n : ℕ) 
  (hn : ∃ m : ℕ, 9 * m = (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ∧ 
                 (9 * m) % 10^7 = 2050306) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) = 22050306 := 
sorry

end smallest_sum_ending_2050306_l222_222808


namespace part_1_solution_part_2_solution_l222_222578

noncomputable def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

theorem part_1_solution (x : ℝ) : f(x) ≥ -2 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
sorry

theorem part_2_solution (a : ℝ) : (∀ x : ℝ, f(x) ≤ x - a) ↔ a ≤ -2 :=
sorry

end part_1_solution_part_2_solution_l222_222578


namespace find_c_value_l222_222241

theorem find_c_value (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 4) (h3 : x2 = 5) (h4 : y2 = 0) (c : ℝ)
  (h5 : 3 * ((x1 + x2) / 2) - 2 * ((y1 + y2) / 2) = c) : c = 5 :=
sorry

end find_c_value_l222_222241


namespace rationalize_denominator_l222_222216

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222216


namespace shaded_area_of_rotated_semicircle_l222_222937

def semicircle_shaded_area (R : ℝ) : ℝ :=
  (2 * real.pi * R^2) / 3

theorem shaded_area_of_rotated_semicircle (R : ℝ) (alpha : ℝ) (h : alpha = real.pi / 3) :
  semicircle_shaded_area R = (2 * real.pi * R^2) / 3 :=
by
  rw [semicircle_shaded_area, h]
  sorry

end shaded_area_of_rotated_semicircle_l222_222937


namespace rationalize_denominator_min_value_l222_222197

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222197


namespace pi_sub_alpha_in_first_quadrant_l222_222605

theorem pi_sub_alpha_in_first_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) : 0 < π - α ∧ π - α < π / 2 :=
by
  sorry

end pi_sub_alpha_in_first_quadrant_l222_222605


namespace sum_of_base8_digits_888_l222_222363

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222363


namespace inequality_a4b_to_abcd_l222_222036

theorem inequality_a4b_to_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end inequality_a4b_to_abcd_l222_222036


namespace ellipse_focus_major_minor_l222_222237

theorem ellipse_focus_major_minor (a b m : ℝ) 
  (h1 : ∀ (x y : ℝ), \frac{x^2}{b^2} + \frac{y^2}{a^2} = 1) 
  (h2 : 2 * b = a) : m = 1 / 4 :=
sorry

end ellipse_focus_major_minor_l222_222237


namespace min_value_rationalize_sqrt_denominator_l222_222194

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222194


namespace min_value_rationalize_sqrt_denominator_l222_222193

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222193


namespace moles_of_HCl_formed_l222_222532

-- Define the reaction as given in conditions
def reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) := C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Define the initial moles of reactants
def moles_C2H6 : ℝ := 2
def moles_Cl2 : ℝ := 2

-- State the expected moles of HCl produced
def expected_moles_HCl : ℝ := 4

-- The theorem stating the problem to prove
theorem moles_of_HCl_formed : ∃ HCl : ℝ, reaction moles_C2H6 moles_Cl2 0 HCl ∧ HCl = expected_moles_HCl :=
by
  -- Skipping detailed proof with sorry
  sorry

end moles_of_HCl_formed_l222_222532


namespace count_integers_between_bounds_l222_222598

theorem count_integers_between_bounds :
  (Real.toIntFloor (-7.5 * 3.1415) + 1) + (Real.toIntCeil (11 * 3.1415) - 1) + 1 = 58 := 
by
  sorry

end count_integers_between_bounds_l222_222598


namespace find_rate_percent_l222_222826

theorem find_rate_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : 2420 = P * (1 + r / 100)^2) 
  (h2 : 3025 = P * (1 + r / 100)^3) : 
  r = 25 :=
by
  sorry

end find_rate_percent_l222_222826


namespace only_first_term_is_prime_l222_222925

def sequence_term (n : ℕ) : ℕ :=
  47 * (10 ^ (2 * (n - 1)) + 10 ^ (2 * (n - 2)) + ... + 10^2 + 1)

theorem only_first_term_is_prime : ∃ (n : ℕ) (hn : n ≥ 1), 
  (∀ k, k ≤ n → (prime (sequence_term 1) ∧ (∀ i, 2 ≤ i → i ≤ k → ¬ prime (sequence_term i)))) →
  n = 1 := by
  sorry

end only_first_term_is_prime_l222_222925


namespace volume_of_open_box_l222_222469

theorem volume_of_open_box :
  let original_length := 46
  let original_width := 36
  let square_cut_length := 8
  let new_length := original_length - 2 * square_cut_length
  let new_width := original_width - 2 * square_cut_length
  let height := square_cut_length
  new_length * new_width * height = 4800 :=
begin
  sorry
end

end volume_of_open_box_l222_222469


namespace frog_arrangements_l222_222297

-- Definitions for the problem
def frogs := ["G", "G", "R", "R", "R", "B", "Y"]
def green (f: String): Prop := f = "G"
def red (f: String): Prop := f = "R"
def adjacent (f1 f2 : String) : Prop := ∃ n, frogs.nth n = some f1 ∧ frogs.nth (n + 1) = some f2

-- Key condition: Green frogs cannot sit next to red frogs
def valid_arrangement (arr: List String) : Prop :=
  ∀ (n : Nat), n < arr.length - 1 → ¬ (green (arr.nth! n) ∧ red (arr.nth! (n + 1))) ∧ ¬ (red (arr.nth! n) ∧ green (arr.nth! (n + 1)))

-- The statement to prove: there are 120 valid arrangements
theorem frog_arrangements : ∃ (arrangements : Finset (List String)), arrangements.card = 120 ∧ ∀ arr ∈ arrangements, valid_arrangement arr := 
sorry

end frog_arrangements_l222_222297


namespace expression_evaluation_l222_222505

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l222_222505


namespace remaining_water_l222_222919

def initial_water : ℚ := 3
def water_used : ℚ := 4 / 3

theorem remaining_water : initial_water - water_used = 5 / 3 := 
by sorry -- skipping the proof for now

end remaining_water_l222_222919


namespace maximum_value_of_f_in_unit_interval_l222_222093

noncomputable def f (a b c : ℝ) : ℝ :=
  a * (1 - a + a * b) * (1 - a * b + a * b * c) * (1 - c)

theorem maximum_value_of_f_in_unit_interval :
  ∃ (a b c : ℝ), a ∈ Icc 0 1 ∧ b ∈ Icc 0 1 ∧ c ∈ Icc 0 1 ∧ f a b c = 8 / 27 ∧
  ∀ (x y z : ℝ), x ∈ Icc 0 1 → y ∈ Icc 0 1 → z ∈ Icc 0 1 → f x y z ≤ 8 / 27 :=
by
  sorry

end maximum_value_of_f_in_unit_interval_l222_222093


namespace normal_distribution_half_probability_l222_222569

variable (σ : ℝ)

theorem normal_distribution_half_probability 
  (ξ : ℝ → Prop)
  (h : ∀ (x : ℝ), ξ x ↔ true )
  (μ : ℝ)
  (hξ : Normal μ σ)
  (hx : μ = 2016):
  P (ξ < 2016) = 1 / 2 := by
  sorry

end normal_distribution_half_probability_l222_222569


namespace problem_1_problem_2_problem_3_l222_222416

theorem problem_1 : 
  ∀ x : ℝ, x^2 - 2 * x + 5 = (x - 1)^2 + 4 := 
sorry

theorem problem_2 (n : ℝ) (h : ∀ x : ℝ, x^2 + 2 * n * x + 3 = (x + 5)^2 - 25 + 3) : 
  n = -5 := 
sorry

theorem problem_3 (a : ℝ) (h : ∀ x : ℝ, (x^2 + 6 * x + 9) * (x^2 - 4 * x + 4) = ((x + a)^2 + b)^2) : 
  a = -1/2 := 
sorry

end problem_1_problem_2_problem_3_l222_222416


namespace find_number_l222_222807

theorem find_number (x : ℝ) (h : 3034 - x / 200.4 = 3029) : x = 1002 :=
sorry

end find_number_l222_222807


namespace trig_identity_l222_222040

open Real

theorem trig_identity (α : ℝ) (h : tan α = 2) :
  2 * cos (2 * α) + 3 * sin (2 * α) - sin (α) ^ 2 = 2 / 5 :=
by sorry

end trig_identity_l222_222040


namespace two_digit_number_condition_l222_222482

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l222_222482


namespace ratio_XQ_QY_l222_222497

theorem ratio_XQ_QY (XQ QY : ℝ) (h1 : XQ + QY = 6) (total_area : 12) (h2 : (total_area / 2) = 6)
  (unit_square : 1) (triangle_base : 6) (triangle_height : triangle_base * (unit_square / 2) = 5) :
  XQ = 2 * QY :=
sorry

end ratio_XQ_QY_l222_222497


namespace sum_of_base_8_digits_888_l222_222342

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222342


namespace sum_abc_is_neg_9_l222_222127

variable {a b c : ℤ}

-- Defining the coordinates of the point M
def M : (ℤ × ℤ × ℤ) := (a, b + 3, 2 * c + 1)

-- Defining the coordinates of the symmetric point M'
def M' : (ℤ × ℤ × ℤ) := (-4, -2, 15)

-- Defining the symmetric conditions
def symmetry_condition (M M' : ℤ × ℤ × ℤ) : Prop :=
  M.1 = - M'.1 ∧ 
  M.2 = M'.2 ∧ 
  M.3 = M'.3

-- Main statement to prove
theorem sum_abc_is_neg_9 (h : symmetry_condition M M') : a + b + c = -9 := 
by
  sorry

end sum_abc_is_neg_9_l222_222127


namespace remainder_of_product_mod_10_l222_222324

theorem remainder_of_product_mod_10 :
  (1265 * 4233 * 254 * 1729) % 10 = 0 := by
  sorry

end remainder_of_product_mod_10_l222_222324


namespace sum_of_digits_base_8_888_is_13_l222_222399

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222399


namespace calculate_outlet_requirements_l222_222462

def outlets_needed := 10
def suites_outlets_needed := 15
def num_standard_rooms := 50
def num_suites := 10
def type_a_percentage := 0.40
def type_b_percentage := 0.60
def type_c_percentage := 1.0

noncomputable def total_outlets_needed := 500 + 150
noncomputable def type_a_outlets_needed := 0.40 * 500
noncomputable def type_b_outlets_needed := 0.60 * 500
noncomputable def type_c_outlets_needed := 150

theorem calculate_outlet_requirements :
  total_outlets_needed = 650 ∧
  type_a_outlets_needed = 200 ∧
  type_b_outlets_needed = 300 ∧
  type_c_outlets_needed = 150 :=
by
  sorry

end calculate_outlet_requirements_l222_222462


namespace conditional_probability_l222_222628

theorem conditional_probability :
  let P_B : ℝ := 0.15
  let P_A : ℝ := 0.05
  let P_A_and_B : ℝ := 0.03
  let P_B_given_A := P_A_and_B / P_A
  P_B_given_A = 0.6 :=
by
  sorry

end conditional_probability_l222_222628


namespace min_value_sum_l222_222057

noncomputable def f (x : ℝ) : ℝ := 3 * x - x^3

theorem min_value_sum : 
  let a := -1
  let b := f a in 
  a + b = -3 := by
  sorry

end min_value_sum_l222_222057


namespace minimum_number_of_tiles_l222_222475

def tile_width_in_inches : ℕ := 6
def tile_height_in_inches : ℕ := 4
def region_width_in_feet : ℕ := 3
def region_height_in_feet : ℕ := 8

def inches_to_feet (i : ℕ) : ℚ :=
  i / 12

def tile_width_in_feet : ℚ :=
  inches_to_feet tile_width_in_inches

def tile_height_in_feet : ℚ :=
  inches_to_feet tile_height_in_inches

def tile_area_in_square_feet : ℚ :=
  tile_width_in_feet * tile_height_in_feet

def region_area_in_square_feet : ℚ :=
  region_width_in_feet * region_height_in_feet

def number_of_tiles : ℚ :=
  region_area_in_square_feet / tile_area_in_square_feet

theorem minimum_number_of_tiles :
  number_of_tiles = 144 := by
    sorry

end minimum_number_of_tiles_l222_222475


namespace cost_of_four_books_l222_222315

theorem cost_of_four_books
  (H : 2 * book_cost = 36) :
  4 * book_cost = 72 :=
by
  sorry

end cost_of_four_books_l222_222315


namespace prove_circle_equation_l222_222868

noncomputable def circle_equation (
  total_students : ℕ,
  freshmen : ℕ,
  sophomores : ℕ,
  seniors : ℕ,
  sampled_students : ℕ,
  a : ℕ,
  b : ℕ,
  angle_BAC : ℝ,
  A : ℝ × ℝ,
  intersects : (ℝ × ℝ) → (ℝ × ℝ) → Prop
) : Prop :=
  total_students = 2500 ∧
  freshmen = 1000 ∧
  sophomores = 900 ∧
  seniors = 600 ∧
  sampled_students = 100 ∧
  angle_BAC = 120 ∧
  A = (1, -1) ∧
  (∃ line : ℝ → ℝ → ℝ, line = λ x y, 5 * x + 3 * y + 1) ∧
  (∃ B C : ℝ × ℝ, intersects B C (λ x y, 5 * x + 3 * y + 1) ∧
    ∠ B A C = angle_BAC) ∧
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = 18 / 17)

theorem prove_circle_equation :
  circle_equation 2500 1000 900 600 100 40 24 120 (1, -1) (λ B C line_eq, line_eq (B.1) (B.2) = 0 ∧ line_eq (C.1) (C.2) = 0) :=
begin
  sorry
end

end prove_circle_equation_l222_222868


namespace sum_of_base8_digits_888_l222_222364

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222364


namespace correct_transformation_l222_222912

noncomputable def transformation_preserves_pattern (P Q : ℝ) (l : list {triangle : Type, circle : Type})
  (is_pattern : ∀ i : ℕ, l.nth i = some (if i % 2 = 0 then {triangle := ℝ} else {circle := ℝ})) : Prop := 
∃ t : ℝ, (∀ x : ℝ, l = l.map (λ y => y + t)) ∧ 
¬ (∃ p : ℝ, (l = l.map (λ y => rotate y p)))
  ∧ 
¬ (∃ q : ℝ, q ⊥ l ∧ (l = l.map (λ y => reflect y q)))

theorem correct_transformation : ∃ t : ℝ, transformation_preserves_pattern = t :=
begin
  sorry
end

end correct_transformation_l222_222912


namespace min_ABCD_sum_correct_ABCD_l222_222208

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222208


namespace part1_part2_part3_l222_222973

-- Define the given functions
def f (x : ℝ) (a : ℝ) := 1 - a * (1 / 2) ^ x + (1 / 4) ^ x
def g (x : ℝ) (a : ℝ) := log (1 / 2) ((1 - a * x) / (x - 1))

-- Conditions
def g_is_odd (a : ℝ) : Prop := ∀ x, g (-x) a = - g x a

-- Part (1)
theorem part1 (h : g_is_odd a) : a = -1 := 
sorry

-- Part (2)
theorem part2 (a : ℝ) (h : a = -1) :
  ∃ m, ∃ x : ℝ, x ∈ set.Icc (-3 : ℝ) 2 → f x a + m = 0 ∧ m ∈ set.Icc (-57 : ℝ) (-3 / 4) := 
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ set.Ici (0 : ℝ), |f x a| ≤ 5) : a ∈ set.Icc (-7 : ℝ) 3 := 
sorry

end part1_part2_part3_l222_222973


namespace square_AC_eq_AH_add_EF_l222_222649
-- Import the entire Mathlib library to ensure we have all necessary definitions.

-- Define the problem in Lean using the identified conditions and question from the problem statement.
theorem square_AC_eq_AH_add_EF (A B C D E F H : Type) [square ABCD]
  (h1 : ⟨AE ⊥ BC⟩)
  (h2 : ⟨AF ⊥ CD⟩)
  (h3 : H is orthocenter of triangle AEF) :
  AC^2 = AH^2 + EF^2 :=
sorry

end square_AC_eq_AH_add_EF_l222_222649


namespace max_value_of_expr_l222_222963

theorem max_value_of_expr (x : ℝ) (h : x ≠ 0) : 
  (∀ y : ℝ, y = (x^2) / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) → y ≤ 1/8) :=
sorry

end max_value_of_expr_l222_222963


namespace opposite_neg_half_l222_222744

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222744


namespace smallest_integer_n_is_148_l222_222009

theorem smallest_integer_n_is_148 :
  ∃ (x : Fin 148 → ℝ), (∑ i, x i = 1200) ∧ (∑ i, (x i) ^ 4 = 800000) :=
sorry

end smallest_integer_n_is_148_l222_222009


namespace minimum_chambers_l222_222678

theorem minimum_chambers (n : ℕ) :
  (∀ (c : Chamber), ∃ (t₁ t₂ t₃ : Tunnel), 
    leads_to t₁ c ∧ leads_to t₂ c ∧ leads_to t₃ c ∧
    connected_via_tunnels c t₁ t₂ t₃) ∧
  (∀ (c₁ c₂ : Chamber), ∃ (p : TunnelPath), 
    path_exists p c₁ c₂) ∧
  (∃ (t : Tunnel), removes_and_splits t) →
  n = 10 :=
by
  sorry

end minimum_chambers_l222_222678


namespace intersection_is_correct_l222_222673

open Set

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_is_correct : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_is_correct_l222_222673


namespace percent_of_number_l222_222452

theorem percent_of_number (N : ℝ) (h : (4 / 5) * (3 / 8) * N = 24) : 2.5 * N = 200 :=
by
  sorry

end percent_of_number_l222_222452


namespace rationalize_denominator_min_value_l222_222186

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222186


namespace smallest_positive_period_tan_alpha_minus_pi_four_l222_222995

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 6) + cos x

theorem smallest_positive_period : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 2 * π :=
by
  sorry

theorem tan_alpha_minus_pi_four (α : ℝ) (h_α_first_quadrant : 0 < α ∧ α < π / 2)
  (h_f_alpha_pi_three : f (α + π / 3) = 4 / 5) :
  tan (α - π / 4) = -1 / 7 :=
by
  sorry

end smallest_positive_period_tan_alpha_minus_pi_four_l222_222995


namespace einstein_needs_more_money_l222_222306

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l222_222306


namespace opposite_of_neg_half_is_half_l222_222740

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222740


namespace find_omega_l222_222061

theorem find_omega (ω α β : ℝ) (h₁ : f x = sin (ω * x - π / 6) + 1/2)
  (h₂ : f α = -1/2) (h₃ : f β = 1/2) (h₄ : abs (α - β) = 3 * π / 4) : ω = 2 / 3 := 
sorry

end find_omega_l222_222061


namespace wire_service_percent_do_not_cover_politics_l222_222931

variable (total_reporters : ℕ) (politics_reporters : ℕ) (local_politics_reporters : ℕ)

def percent_do_not_cover_politics 
  (total_reporters : ℕ) 
  (local_politics_reporters : ℕ) 
  (politics_reporters : ℕ) 
  (h1 : total_reporters = 100)
  (h2 : 0.2 * total_reporters = local_politics_reporters)
  (h3 : 0.8 * politics_reporters = local_politics_reporters) : ℕ :=
  100 * (total_reporters - politics_reporters) / total_reporters

theorem wire_service_percent_do_not_cover_politics
  (total_reporters : ℕ) (local_politics_reporters : ℕ) (politics_reporters : ℕ)
  (h1 : total_reporters = 100)
  (h2 : 0.2 * total_reporters = local_politics_reporters)
  (h3 : 0.8 * politics_reporters = local_politics_reporters) :
  percent_do_not_cover_politics total_reporters local_politics_reporters politics_reporters h1 h2 h3 = 75 :=
by
  sorry

end wire_service_percent_do_not_cover_politics_l222_222931


namespace opposite_neg_half_l222_222747

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222747


namespace number_of_triangles_l222_222544

theorem number_of_triangles (n : ℕ) (h1 : n ≥ 3) : 
  ∑ (i : ℕ) in (finset.range n).powerset.len 3, 1 = nat.choose n 3 := 
sorry

end number_of_triangles_l222_222544


namespace preserved_connectedness_after_edge_deletion_l222_222520

-- Definitions to capture the conditions provided
structure Graph (V : Type) :=
  (E : Type)
  (incidence : E → V × V)
  (adj : V → V → Prop := λ u v, ∃ e, incidence e = (u, v) ∨ incidence e = (v, u))

def is_colored (G : Graph V) (N : ℕ) :=
  ∃ (color : G.E → fin N), ∀ v : V, ∃! c : fin N, ∃ e : G.E, (G.incidence e).fst = v ∨ (G.incidence e).snd = v ∧ (color e) = c

def is_connected (G : Graph V) :=
  ∀ u v : V, ∃ (p : list V), p.head? = some u ∧ p.last? = some v ∧ ∀ i, i < p.length - 1 → G.adj (p.nth_le i sorry) (p.nth_le (i+1) sorry)

-- The main theorem to state the proof problem
theorem preserved_connectedness_after_edge_deletion
  {V : Type} {G : Graph V} {N : ℕ}
  (hC : is_connected G)
  (hColored : is_colored G N) :
   ∃ G' : Graph V,
     ((∃ color : G.E → fin N,
        ∀ (v : V), ∃! (c : fin (N-1)), ∃ e : G.E, 
          (G.incidence e).fst = v ∨ (G.incidence e).snd = v ∧ 
          (color e) = c) ∧ is_connected G'.

end preserved_connectedness_after_edge_deletion_l222_222520


namespace correct_statements_l222_222296

-- Definitions based on the conditions given in the problem

def statement1 (a b : ℝ) : Prop :=
a > b ↔ a^2 > b^2

def statement2 (A B : Set ℝ) : Prop :=
(A ∩ B) = B → B = ∅

def statement3 (x : ℝ) : Prop :=
x = 3 → x^2 - 2 * x - 3 = 0 ∧ ¬(x^2 - 2 * x - 3 = 0 → x = 3)

def statement4 (m : ℝ) : Prop :=
(∃ q : ℚ, m = q) ↔ (∃ r : ℝ, m = r)

-- The final theorem to prove which statements are correct
theorem correct_statements :
  ¬statement1 ∧ statement2 ∧ statement3 ∧ statement4 :=
by
  intro a b A B x m
  sorry

end correct_statements_l222_222296


namespace find_some_number_l222_222092

theorem find_some_number (x some_number : ℝ) (h1 : (27 / 4) * x - some_number = 3 * x + 27) (h2 : x = 12) :
  some_number = 18 :=
by
  sorry

end find_some_number_l222_222092


namespace find_BD_l222_222104
-- Import the entire Mathlib library

-- Define the points A, B, C, and D, and their respective distances
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define the distances
variables (AB AC BC CD : ℝ)
variables(h1 : AC = 10)(h2 : BC = 10)(h3 : AB = 5)(h4 : CD = 13)

-- Define the points relationship
variables (B_between_A_D : B ∈ segment A D)
  
-- The theorem statement
theorem find_BD (BD : ℝ) : BD = real.sqrt 75.25 - 2.5 :=
sorry

end find_BD_l222_222104


namespace line_contains_point_l222_222948

theorem line_contains_point (k : ℤ) : (∀ x y : ℤ, 3 - k * y = -4 * x → (x, y) = (2, -1)) → k = -11 :=
by
  intro H
  specialize H 2 (-1)
  simp at H
  exact H.symm

end line_contains_point_l222_222948


namespace range_of_3a_minus_b_l222_222021

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a) (ha' : a < 2) (hb : 1 < b) (hb' : b < 4) : 
  -19 < 3 * a - b ∧ 3 * a - b < 5 :=
by
  sorry

end range_of_3a_minus_b_l222_222021


namespace part1_part2_l222_222551

open Classical

variable (a : ℝ)

-- Define the sequence of polynomials using a recursive relation
def f : ℕ → (ℝ → ℝ)
| 0     => fun x => 1
| (n+1) => fun x => x * f n x + f n (a * x)

-- Prove the symmetry property
theorem part1 (n : ℕ) (x : ℝ) :
  f a n x = x^n * f a n (1 / x) := sorry

-- Define an explicit expression for the polynomials
noncomputable def explicit_f (n : ℕ) (x : ℝ) : ℝ :=
  1 + ∑ j in Finset.range n,
    ((∏ i in Finset.range j, a^(n-i) - 1) / (∏ i in Finset.range j, a - 1)) * x^(j+1)

-- Prove the explicit expression matches the recursive definition
theorem part2 (n : ℕ) (x : ℝ) :
  f a n x = explicit_f a n x := sorry

end part1_part2_l222_222551


namespace no_two_distinct_real_roots_l222_222788

-- Definitions of the conditions and question in Lean 4
theorem no_two_distinct_real_roots (a : ℝ) (h : a ≥ 1) : ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*x1 + a = 0) ∧ (x2^2 - 2*x2 + a = 0) :=
sorry

end no_two_distinct_real_roots_l222_222788


namespace solve_functional_equation_l222_222933

theorem solve_functional_equation
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, (∀ x, f x = d * x^2 + c) ∧ (∀ x, g x = d * x^2 + c) :=
sorry

end solve_functional_equation_l222_222933


namespace halfway_miles_proof_l222_222311

def groceries_miles : ℕ := 10
def haircut_miles : ℕ := 15
def doctor_miles : ℕ := 5

def total_miles : ℕ := groceries_miles + haircut_miles + doctor_miles

theorem halfway_miles_proof : total_miles / 2 = 15 := by
  -- calculation to follow
  sorry

end halfway_miles_proof_l222_222311


namespace sum_of_digits_base_8_888_is_13_l222_222403

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222403


namespace painting_percentage_l222_222591

noncomputable def area_painting (length_painting width_painting : ℝ) : ℝ :=
  length_painting * width_painting

noncomputable def area_trapezoid (top_base bottom_base height : ℝ) : ℝ :=
  (1 / 2) * (top_base + bottom_base) * height
  
noncomputable def percentage_painting_on_wall (length_painting width_painting top_base bottom_base height : ℝ) : ℝ :=
  (area_painting length_painting width_painting / area_trapezoid top_base bottom_base height) * 100

theorem painting_percentage (length_painting width_painting top_base bottom_base height : ℝ) 
  (h1 : length_painting = 3.5) (h2 : width_painting = 7.2) 
  (h3 : top_base = 9.6) (h4 : bottom_base = 14) 
  (h5 : height = 10) : 
  percentage_painting_on_wall length_painting width_painting top_base bottom_base height ≈ 21.36 := 
by
  sorry

end painting_percentage_l222_222591


namespace no_common_integer_solution_l222_222508

theorem no_common_integer_solution :
  ∀ x : ℤ, ¬ (14 * x + 5 ≡ 0 [MOD 9] ∧ 17 * x - 5 ≡ 0 [MOD 12]) :=
by
  assume x
  sorry

end no_common_integer_solution_l222_222508


namespace probability_defective_product_probability_A1_given_defective_l222_222851

theorem probability_defective_product (P_A1 P_A2 P_A3 P_B_given_A1 P_B_given_A2 P_B_given_A3 : ℝ)
  (h1 : P_A1 = 0.25) (h2 : P_A2 = 0.35) (h3 : P_A3 = 0.4) 
  (h4 : P_B_given_A1 = 0.05) (h5 : P_B_given_A2 = 0.04) (h6 : P_B_given_A3 = 0.02) :
  (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) = 0.0345 := 
by 
  -- Proof skipped
  sorry

theorem probability_A1_given_defective (P_A1 P_A2 P_A3 P_B_given_A1 P_B_given_A2 P_B_given_A3 : ℝ)
  (h1 : P_A1 = 0.25) (h2 : P_A2 = 0.35) (h3 : P_A3 = 0.4) 
  (h4 : P_B_given_A1 = 0.05) (h5 : P_B_given_A2 = 0.04) (h6 : P_B_given_A3 = 0.02)
  (P_B : ℝ) (hP_B : P_B = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) :
  (P_A1 * P_B_given_A1 / P_B) ≈ 0.36 := 
by 
  -- Proof skipped
  sorry

end probability_defective_product_probability_A1_given_defective_l222_222851


namespace find_X_l222_222944

theorem find_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ X = 0.3 :=
by
  sorry

end find_X_l222_222944


namespace angles_arithmetic_sequence_sides_l222_222574

theorem angles_arithmetic_sequence_sides (A B C a b c : ℝ)
  (h_angle_ABC : A + B + C = 180)
  (h_arithmetic_sequence : 2 * B = A + C)
  (h_cos_B : A * A + c * c - b * b = 2 * a * c)
  (angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < 180 ∧ B < 180 ∧ C < 180) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end angles_arithmetic_sequence_sides_l222_222574


namespace projection_of_right_angle_aob_l222_222701

-- Definitions
variable (α : Plane) -- A fixed plane
variable (A B O : Point) -- Points defining right angle AOB
variable (right_angle_aob : rightAngle A O B) -- Right angle at O between A and B

-- Assumptions
variable (proj_angle : Angle) -- Projection angle of AOB onto plane α
variable (orthogonal : ∀ p : Plane, (A, B, O lie_in p) → p ⊥ α → proj_angle = 0 ∨ proj_angle = 180) -- When plane containing AOB is perpendicular to α
variable (parallel : ∀ p : Plane, (A, B, O lie_in p) → p ∥ α → proj_angle = 90) -- When plane containing AOB is parallel to α
variable (neither : ∀ p : Plane, (A, B, O lie_in p) → ¬ (p ⊥ α) → ¬ (p ∥ α) → (acute proj_angle ∨ obtuse proj_angle)) -- When plane containing AOB is neither parallel nor perpendicular to α

-- The theorem to prove
theorem projection_of_right_angle_aob : 
  (proj_angle = 0) ∨ (acute proj_angle) ∨ (proj_angle = 90) ∨ (obtuse proj_angle) ∨ (proj_angle = 180) := sorry

end projection_of_right_angle_aob_l222_222701


namespace min_ABCD_sum_correct_ABCD_l222_222204

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222204


namespace words_lost_in_oz_l222_222646

-- Define constants for the number of letters and the forbidden letter position
def total_letters : ℕ := 69
def forbidden_letter_pos : ℕ := 7

-- Define the number of one-letter words lost
def one_letter_words_lost : ℕ := 1

-- Define the number of two-letter words lost
def two_letter_words_lost : ℕ := total_letters + total_letters - 1

-- Calculate the total number of words lost
def total_words_lost : ℕ := one_letter_words_lost + two_letter_words_lost

-- Prove that the total number of words lost is 138
theorem words_lost_in_oz : total_words_lost = 138 := by
  unfold total_letters forbidden_letter_pos one_letter_words_lost two_letter_words_lost total_words_lost
  calc 1 + (69 + 69 - 1) = 1 + 137 : by rw [Nat.add_sub_cancel_left]
  ...                       = 138 : by rw [Nat.add_comm]

end words_lost_in_oz_l222_222646


namespace opposite_of_neg_half_l222_222759

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222759


namespace number_of_winners_is_four_l222_222490

-- Definitions for each person winning
def A_wins : Prop := true
def B_wins : Prop := true
def C_wins : Prop := true
def D_wins : Prop := true

-- Conditions from the problem
def cond1 : Prop := A_wins → B_wins
def cond2 : Prop := B_wins → (C_wins ∨ ¬A_wins)
def cond3 : Prop := ¬D_wins → (A_wins ∧ ¬C_wins)
def cond4 : Prop := D_wins → A_wins

-- Known facts from the problem
def known1 : A_wins := by true.intro
def known2 : D_wins := by true.intro

-- Proof problem statement
theorem number_of_winners_is_four : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ known1 ∧ known2 → 
  (A_wins ∧ B_wins ∧ C_wins ∧ D_wins) 
:= by
  sorry

end number_of_winners_is_four_l222_222490


namespace rationalize_denominator_and_min_sum_l222_222179

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222179


namespace probability_distribution_xi_l222_222772

theorem probability_distribution_xi (a : ℝ) (ξ : ℕ → ℝ) (h1 : ξ 1 = a / (1 * 2))
  (h2 : ξ 2 = a / (2 * 3)) (h3 : ξ 3 = a / (3 * 4)) (h4 : ξ 4 = a / (4 * 5))
  (h5 : (ξ 1) + (ξ 2) + (ξ 3) + (ξ 4) = 1) :
  ξ 1 + ξ 2 = 5 / 6 :=
by
  sorry

end probability_distribution_xi_l222_222772


namespace sum_of_base_8_digits_888_l222_222341

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222341


namespace max_value_x_1_minus_3x_is_1_over_12_l222_222956

open Real

noncomputable def max_value_of_x_1_minus_3x (x : ℝ) : ℝ :=
  x * (1 - 3 * x)

theorem max_value_x_1_minus_3x_is_1_over_12 :
  ∀ x : ℝ, 0 < x ∧ x < 1 / 3 → max_value_of_x_1_minus_3x x ≤ 1 / 12 :=
by
  intros x h
  sorry

end max_value_x_1_minus_3x_is_1_over_12_l222_222956


namespace sum_of_digits_base8_888_l222_222377

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222377


namespace soccer_balls_are_20_l222_222294

variable (S : ℕ)
variable (num_baseballs : ℕ) (num_volleyballs : ℕ)
variable (condition_baseballs : num_baseballs = 5 * S)
variable (condition_volleyballs : num_volleyballs = 3 * S)
variable (condition_total : num_baseballs + num_volleyballs = 160)

theorem soccer_balls_are_20 :
  S = 20 :=
by
  sorry

end soccer_balls_are_20_l222_222294


namespace opposite_of_neg_half_is_half_l222_222739

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222739


namespace rationalize_denominator_l222_222212

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222212


namespace sum_of_digits_base8_888_l222_222408

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222408


namespace constant_S13_l222_222643

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |> List.sum

-- Defining the given conditions as hypotheses
variable {a : ℕ → ℤ} {d : ℤ}
variable (h_arith : arithmetic_sequence a d)
variable (constant_sum : a 2 + a 4 + a 15 = k)

-- Goal to prove: S_13 is a constant
theorem constant_S13 (k : ℤ) :
  sum_first_n_terms a 13 = k :=
  sorry

end constant_S13_l222_222643


namespace naturals_less_than_10_l222_222524

theorem naturals_less_than_10 :
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end naturals_less_than_10_l222_222524


namespace rhombus_side_length_l222_222781

theorem rhombus_side_length (L S m : ℝ) (h1 : 0 < L) (h2 : 0 < S) (h3 : m^2 = (L^2 / 4) - S) :
  m = sqrt ((L^2 - 4 * S) / 2) :=
by sorry

end rhombus_side_length_l222_222781


namespace proof_problem_l222_222709

variables {a b c : Real}

theorem proof_problem (h1 : a < 0) (h2 : |a| < |b|) (h3 : |b| < |c|) (h4 : b < 0) :
  (|a * b| < |b * c|) ∧ (a * c < |b * c|) ∧ (|a + b| < |b + c|) :=
by
  sorry

end proof_problem_l222_222709


namespace rationalize_denominator_and_min_sum_l222_222177

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222177


namespace equation_one_solution_equation_two_real_solutions_l222_222707

-- Problem 1: Solve 2x^2 - 4x - 3 = 0
theorem equation_one_solution (x : ℝ) : 2 * x^2 - 4 * x - 3 = 0 ↔ x = 1 + (1 / 2) * real.sqrt (10) ∨ x = 1 - (1 / 2) * real.sqrt (10) := by
  sorry

-- Problem 2: Solve (x^2 + x)^2 - x^2 - x = 30
theorem equation_two_real_solutions (x : ℝ) : (x^2 + x)^2 - x^2 - x = 30 ↔ x = -3 ∨ x = 2 := by
  sorry

end equation_one_solution_equation_two_real_solutions_l222_222707


namespace keith_attended_games_l222_222303

def total_games : ℕ := 8
def missed_games : ℕ := 4
def attended_games (total : ℕ) (missed : ℕ) : ℕ := total - missed

theorem keith_attended_games : attended_games total_games missed_games = 4 := by
  sorry

end keith_attended_games_l222_222303


namespace unique_wins_log2_q_l222_222304

theorem unique_wins_log2_q :
  let n := 30,
      games := (n * (n - 1)) / 2,
      total_possibilities := 2 ^ games,
      factorial_30 := (Finset.range n).prod Nat.succ,
      powers_of_2_in_30_fact :=
        (List.range (Nat.log 30 2 + 1)).sum (λ k, 30 / 2^k),
      q := 2 ^ (games - powers_of_2_in_30_fact) in
      log 2 q = 409 :=
by
  let n := 30
  let games := (n * (n - 1)) / 2
  let total_possibilities := 2 ^ games
  let factorial_30 := (Finset.range n).prod Nat.succ
  let powers_of_2_in_30_fact := (List.range (Nat.log 30 2 + 1)).sum (λ k, 30 / 2^k)
  let q := 2 ^ (games - powers_of_2_in_30_fact)
  have h1 : log 2 (2 ^ (games - powers_of_2_in_30_fact)) = games - powers_of_2_in_30_fact := by sorry
  have h2 : games - powers_of_2_in_30_fact = 409 := by sorry
  exact h1.symm.trans h2

end unique_wins_log2_q_l222_222304


namespace unit_digit_of_product_is_4_l222_222281

theorem unit_digit_of_product_is_4 :
  let expr := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1 in
  expr % 10 = 4 :=
by
  -- define the expression 
  let expr : ℕ := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1
  -- ensure the equivalence of unit digit
  show expr % 10 = 4
  sorry -- proof goes here

end unit_digit_of_product_is_4_l222_222281


namespace range_m_l222_222218

theorem range_m (m : ℝ) : 
  (∀ x : ℝ, ((m * x - 1) * (x - 2) > 0) ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_m_l222_222218


namespace sum_of_digits_of_8_pow_2004_l222_222812

-- Define the problem statement
theorem sum_of_digits_of_8_pow_2004 :
  let n := 8 ^ 2004 in 
  (n % 100) / 10 + ((n % 100) % 10) = 7 :=
by 
  sorry

end sum_of_digits_of_8_pow_2004_l222_222812


namespace sum_of_base8_digits_888_l222_222362

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222362


namespace sqrt_multiplication_correctness_l222_222816

theorem sqrt_multiplication_correctness : 
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 :=
by 
  rw [Real.mul_self_sqrt (by norm_num : 0 ≤ 2), 
      Real.mul_self_sqrt (by norm_num : 0 ≤ 3), 
      Real.sqrt_mul (by norm_num : 0 ≤ 2) (by norm_num : 0 ≤ 3)]
  norm_num
  sorry

end sqrt_multiplication_correctness_l222_222816


namespace largest_time_for_77_degrees_l222_222105

-- Define the initial conditions of the problem
def temperature_eqn (t : ℝ) : ℝ := -t^2 + 14 * t + 40

-- Define the proposition we want to prove
theorem largest_time_for_77_degrees : ∃ t, temperature_eqn t = 77 ∧ t = 11 := 
sorry

end largest_time_for_77_degrees_l222_222105


namespace product_equivalence_l222_222611

theorem product_equivalence 
  (a b c d e f : ℝ) 
  (h1 : a + b + c + d + e + f = 0) 
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) : 
  (a + c) * (a + d) * (a + e) * (a + f) = (b + c) * (b + d) * (b + e) * (b + f) :=
by
  sorry

end product_equivalence_l222_222611


namespace speed_of_person_l222_222892

-- Defining the given conditions
def speed_of_escalator : ℝ := 11
def length_of_escalator : ℝ := 126
def time_taken : ℝ := 9

-- The main statement to be proved
theorem speed_of_person :
  let distance_escalator_covers := speed_of_escalator * time_taken,
      remaining_distance := length_of_escalator - distance_escalator_covers,
      speed_of_person_relative := remaining_distance / time_taken,
      actual_speed_of_person := speed_of_person_relative + speed_of_escalator
  in actual_speed_of_person = 14 := sorry

end speed_of_person_l222_222892


namespace interest_years_l222_222479

def simple_interest (P R N : ℝ) := (P * R * N) / 100

theorem interest_years (R : ℝ) (N : ℝ) (h : simple_interest 200 (R + 5) N = simple_interest 200 R N + 100) :
  N = 100 :=
by
  sorry

end interest_years_l222_222479


namespace work_problem_l222_222842

theorem work_problem (days_B : ℝ) (h : (1 / 20) + (1 / days_B) = 1 / 8.571428571428571) : days_B = 15 :=
sorry

end work_problem_l222_222842


namespace triangle_area_correct_l222_222624

noncomputable def area_of_triangle
  (a : ℝ) (A B : ℝ) (h_a : a = Real.sqrt 2) (h_A : A = Real.pi / 4) (h_B : B = Real.pi / 3) : 
  ℝ := 
  let b := (a * Real.sin B) / (Real.sin A) in
  let C := Real.pi - (A + B) in
  let sin_C := Real.sin C in
  (1 / 2) * a * b * sin_C

theorem triangle_area_correct :
  ∀ (a A B : ℝ) (h_a : a = Real.sqrt 2) (h_A : A = Real.pi / 4) (h_B : B = Real.pi / 3),
  area_of_triangle a A B h_a h_A h_B = (3 + Real.sqrt 3) / 4 :=
by
  intros
  sorry

end triangle_area_correct_l222_222624


namespace arthur_walked_total_miles_l222_222501

def blocks_east := 8
def blocks_north := 15
def blocks_west := 3
def block_length := 1/2

def total_blocks := blocks_east + blocks_north + blocks_west
def total_miles := total_blocks * block_length

theorem arthur_walked_total_miles : total_miles = 13 := by
  sorry

end arthur_walked_total_miles_l222_222501


namespace students_with_same_grade_l222_222280

theorem students_with_same_grade :
  let total_students := 40
  let students_with_same_A := 3
  let students_with_same_B := 2
  let students_with_same_C := 6
  let students_with_same_D := 1
  let total_same_grade_students := students_with_same_A + students_with_same_B + students_with_same_C + students_with_same_D
  total_same_grade_students = 12 →
  (total_same_grade_students / total_students) * 100 = 30 :=
by
  sorry

end students_with_same_grade_l222_222280


namespace sum_of_digits_base8_888_l222_222376

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222376


namespace find_four_digit_number_l222_222843

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l222_222843


namespace midpoint_KL_on_circumcircle_l222_222833

-- Definitions based on the conditions given in the problem
variables {A B C K L: Type}
variables {triangle_ABC : Triangle A B C}

-- Defining K as the intersection point of the internal angle bisector of ∠B and external angle bisector of ∠C
def K_definition := is_intersection_angle_bisectors A B C

-- Defining L as the intersection point of the internal angle bisector of ∠C and external angle bisector of ∠B
def L_definition := is_intersection_angle_bisectors A C B

-- Main theorem statement
theorem midpoint_KL_on_circumcircle 
  (triangle_ABC : Triangle A B C)
  (K_definition : K_definition)
  (L_definition : L_definition) :
  midpoint K L ∈ circumcircle triangle_ABC :=
by
  -- Proof steps would go here
  sorry

end midpoint_KL_on_circumcircle_l222_222833


namespace sum_of_digits_base8_888_l222_222365

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222365


namespace intersection_A_B_l222_222561

open Set

variable {t : ℝ}

def setA : Set ℝ := { t | -3 < t ∧ t < -1 }
def setB : Set ℝ := { t | t ≤ -2 ∨ t ≥ 0 }
def setC : Set ℝ := (-3 : ℝ) <' (-2 : ℝ)]

theorem intersection_A_B :
  (setA ∩ setB) = setC := by
sorry

end intersection_A_B_l222_222561


namespace find_k_l222_222957

-- Define the variables and their relations
noncomputable def a (k : ℝ) : ℝ := k / 2
noncomputable def b (k : ℝ) : ℝ := k / 3

-- Define the main theorem to prove
theorem find_k : ∀ (k : ℝ), k ≠ 1 ∧ (2 * a k + b k = (a k) * (b k)) → k = 18 :=
begin
  sorry
end

end find_k_l222_222957


namespace possible_values_of_b_number_of_possible_values_of_b_l222_222230

theorem possible_values_of_b (b : ℕ) (h1 : 2 ≤ b) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : b = 5 ∨ b = 6 :=
begin
  sorry
end

theorem number_of_possible_values_of_b : ℕ :=
nat.card {b : ℕ | 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4}.to_finset = 2

end possible_values_of_b_number_of_possible_values_of_b_l222_222230


namespace rationalize_denominator_and_min_sum_l222_222182

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222182


namespace maximum_value_l222_222140

variables (a b c : ℝ)
variables (a_vec b_vec c_vec : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ‖a_vec‖ = 2
axiom norm_b : ‖b_vec‖ = 3
axiom norm_c : ‖c_vec‖ = 4

theorem maximum_value : 
  (‖(a_vec - (3:ℝ) • b_vec)‖^2 + ‖(b_vec - (3:ℝ) • c_vec)‖^2 + ‖(c_vec - (3:ℝ) • a_vec)‖^2) ≤ 377 :=
by
  sorry

end maximum_value_l222_222140


namespace part1_correct_part2_correct_l222_222853

-- Define the probabilities of selecting from each workshop
def P_A1 : ℝ := 0.25
def P_A2 : ℝ := 0.35
def P_A3 : ℝ := 0.4

-- Define the probabilities of a defective product given it was produced in each workshop
def P_B_A1 : ℝ := 0.05
def P_B_A2 : ℝ := 0.04
def P_B_A3 : ℝ := 0.02

-- Compute the total probability of selecting a defective product using the law of total probability
def P_B : ℝ := P_A1 * P_B_A1 + P_A2 * P_B_A2 + P_A3 * P_B_A3

-- Compute the conditional probability that a defective product was produced in workshop A using Bayes' theorem
def P_A1_B : ℝ := (P_A1 * P_B_A1) / P_B

-- The theorem to check if computations are correct
theorem part1_correct : P_B = 0.0345 := by sorry
theorem part2_correct : P_A1_B = 0.36 := by sorry

end part1_correct_part2_correct_l222_222853


namespace sum_of_digits_base8_888_l222_222372

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222372


namespace opposite_neg_half_l222_222742

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222742


namespace quotient_remainder_correct_l222_222323

noncomputable def P : ℚ[X] := 10 * X^4 + 5 * X^3 - 9 * X^2 + 7 * X + 2
noncomputable def D : ℚ[X] := 3 * X^2 + 2 * X + 1
noncomputable def Q : ℚ[X] := (10 / 3) * X^2 - (5 / 9) * X - (193 / 243)
noncomputable def R : ℚ[X] := (592 / 27) * X + (179 / 27)

theorem quotient_remainder_correct :
  P = D * Q + R ∧ degree R < degree D :=
by
  sorry

end quotient_remainder_correct_l222_222323


namespace problem_l222_222699

-- Define the problem conditions and the statement that needs to be proved
theorem problem:
  ∀ (x : ℝ), (x ∈ Set.Icc (-1) m) ∧ ((1 - (-1)) / (m - (-1)) = 2 / 5) → m = 4 := by
  sorry

end problem_l222_222699


namespace percent_increase_area_l222_222274

-- Given conditions
def diameter1 : ℝ := 14
def radius1 : ℝ := diameter1 / 2
def area1 : ℝ := π * radius1^2

def diameter2 : ℝ := 18
def radius2 : ℝ := diameter2 / 2
def area2 : ℝ := π * radius2^2

-- The main statement to prove
theorem percent_increase_area :
  ((area2 - area1) / area1) * 100 ≈ 65.31 :=
by
  -- Placeholder proof
  sorry

end percent_increase_area_l222_222274


namespace least_four_digit_palindrome_div_by_5_l222_222800

noncomputable def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in s = s.reverse

theorem least_four_digit_palindrome_div_by_5 : 
  ∃ n : ℕ, is_palindrome n ∧ 1000 ≤ n ∧ n < 10000 ∧ n % 5 = 0 ∧ ∀ m : ℕ, is_palindrome m ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 5 = 0 → n ≤ m := 
sorry

end least_four_digit_palindrome_div_by_5_l222_222800


namespace correct_calculation_A_l222_222819

theorem correct_calculation_A : (sqrt 2 * sqrt 3 = sqrt 6) ∧ 
  ¬ (sqrt 5 - sqrt 3 = sqrt 2) ∧ 
  ¬ (2 + sqrt 3 = 2 * sqrt 3) ∧ 
  ¬ (sqrt 8 / sqrt 2 = 4) := by sorry

end correct_calculation_A_l222_222819


namespace ab_plus_cd_eq_neg_346_over_9_l222_222088

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l222_222088


namespace Dan_picked_9_plums_l222_222676

-- Define the constants based on the problem
def M : ℕ := 4 -- Melanie's plums
def S : ℕ := 3 -- Sally's plums
def T : ℕ := 16 -- Total plums picked

-- The number of plums Dan picked
def D : ℕ := T - (M + S)

-- The theorem we want to prove
theorem Dan_picked_9_plums : D = 9 := by
  sorry

end Dan_picked_9_plums_l222_222676


namespace distinct_gcd_values_of_ab_l222_222417

theorem distinct_gcd_values_of_ab (a b : ℕ) (h : gcd a b * lcm a b = 600) : 
  ∃ (S : Set ℕ), S.card = 14 ∧ ∀ x ∈ S, x = gcd a b :=
sorry

end distinct_gcd_values_of_ab_l222_222417


namespace sum_of_digits_base8_l222_222387

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222387


namespace minimum_edges_for_16_yellow_to_green_impossible_to_turn_all_23_yellow_to_green_l222_222708

-- Problem (a): Prove that to turn all 16 lamps green, at least 8 edges need to be touched.
theorem minimum_edges_for_16_yellow_to_green (A : Type) (L : set A) (W : set (A × A)) 
    (h : ∀ l ∈ L, isYellow l) (hL : L.card = 16) (hW : ∀ e ∈ W, e.fst ∈ L ∧ e.snd ∈ L):
  ∃ S ⊆ W, S.card ≥ 8 ∧ (∀ e ∈ S, changes_color e) := 
sorry

-- Problem (b): Prove that it is impossible to turn all 23 lamps green by touching any number of edges.
theorem impossible_to_turn_all_23_yellow_to_green (A : Type) (L : set A) (W : set (A × A)) 
    (h : ∀ l ∈ L, isYellow l) (hL : L.card = 23) (hW : ∀ e ∈ W, e.fst ∈ L ∧ e.snd ∈ L):
  ¬ ∃ S ⊆ W, (∀ e ∈ S, changes_color e) ∧ (∀ l ∈ L, isGreen l) := 
sorry

end minimum_edges_for_16_yellow_to_green_impossible_to_turn_all_23_yellow_to_green_l222_222708


namespace complement_U_M_l222_222159

theorem complement_U_M :
  let U := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  let M := {x : ℤ | ∃ k : ℤ, x = 4 * k}
  {x | x ∈ U ∧ x ∉ M} = {x : ℤ | ∃ k : ℤ, x = 4 * k - 2} :=
by
  sorry

end complement_U_M_l222_222159


namespace range_of_a_l222_222156

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x)
noncomputable def f_prime (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def f_double_prime (x : ℝ) : ℝ := -1 / (1 + x)^2
noncomputable def g (x : ℝ) : ℝ := x * f_double_prime x

theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) : 
  (f x) - a * (g x) ≥ 0 → a ≤1 :=
by
  let F (x : ℝ) := f x - a * g x
  let F' (x : ℝ) := ((1 : ℝ) - a) / (1 + x)^2
  have hF' : ∀ x, x ≥ 0 → F' x ≥ 0, from sorry,
  have h : (1 : ℝ) - a ≥ 0, from sorry,
  exact h

end range_of_a_l222_222156


namespace min_value_rationalize_sqrt_denominator_l222_222196

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222196


namespace question_to_answer_l222_222063

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * Real.log x + 3

theorem question_to_answer {m : ℝ} {a : ℝ} (h₁ : 2 ≤ a) (h₂ : a ≤ 3)
  (h₃ : ∀ x₁ x₂ : ℝ, 4 ≤ x₁ → 4 ≤ x₂ → x₁ ≠ x₂ → (f x₂ a - f x₁ a) / (x₁ - x₂) < 2 * m) :
  -⟨19, 4⟩ ≤ m :=
sorry

end question_to_answer_l222_222063


namespace sum_of_base8_digits_888_l222_222356

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222356


namespace min_N_value_l222_222921

theorem min_N_value : 
  ∃ (N : ℕ), (∀ (grid : Matrix (Fin 10) (Fin 10) ℕ), 
    (∀ i j, grid i j = 1 ∨ grid i j = 2) ∧
    (∃ (ones twos : ℕ), 
      (ones + twos = 100) ∧ 
      (ones = 50) ∧ 
      (twos = 50) ∧ 
      (∑ i, (∏ j, grid i j) + 
       ∑ j, (∏ i, grid i j)) = N)) → 
  N ≥ 640 :=
by
  sorry

end min_N_value_l222_222921


namespace larger_cube_remains_heavier_l222_222083

-- Define the properties of the two cubes
variables {cube : Type} [has_volume cube]

-- Define the larger and smaller cube
variables (smaller_cube larger_cube : cube)

-- Define the relation for volume
definition larger_volume (c1 c2 : cube) [has_volume c1] [has_volume c2] : Prop :=
  volume c1 > volume c2

-- Define the condition of the hole being drilled into the smaller cube
definition hole_drilled (c1 : cube) [has_volume c1] (volume_removed : ℝ) : Prop :=
  volume c1 > volume_removed

-- State the main theorem describing the conditions under which the larger cube remains heavier
theorem larger_cube_remains_heavier 
  [has_volume larger_cube] [has_volume smaller_cube]
  (h1 : larger_volume larger_cube smaller_cube)
  (h2 : hole_drilled smaller_cube (volume smaller_cube)) :
  larger_volume larger_cube (smaller_cube - volume smaller_cube) :=
sorry

end larger_cube_remains_heavier_l222_222083


namespace sum_of_digits_base_8_888_is_13_l222_222401

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222401


namespace part1_part2_part3_l222_222060

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := log x - a * x + b / x

-- Condition: f(x) + f(1/x) = 0
def condition (x a b : ℝ) : Prop := f x a b + f (1/x) a b = 0

-- Prove a = -2 when f has a tangent at x=1 passing through (0, -5)
theorem part1 (x a b : ℝ) (h : condition x a b) (tangent_line : ∃ m, ∀ (y : ℝ), y = f 1 a b + m * (y - 1) → (0, -5) ∈ tangent_line) :
  a = -2 := sorry

-- Prove f(a^2 / 2) > 0 given 0 < a < 1
theorem part2 (a : ℝ) (h : 0 < a ∧ a < 1) : f (a^2 / 2) a a > 0 := sorry

-- Prove the range of a when f has three distinct zeros
theorem part3 (a b : ℝ) (h : condition a a b) (h2 : number_of_zeros f = 3) :
  0 < a ∧ a < 1 / 2 := sorry

end part1_part2_part3_l222_222060


namespace symmetry_probability_l222_222291

-- Define the setting of the problem
def grid_points : ℕ := 121
def grid_size : ℕ := 11
def center_point : (ℕ × ℕ) := (6, 6)
def total_points : ℕ := grid_points - 1
def symmetric_lines : ℕ := 4
def points_per_line : ℕ := 10
def total_symmetric_points : ℕ := symmetric_lines * points_per_line
def probability : ℚ := total_symmetric_points / total_points

-- Theorem statement
theorem symmetry_probability 
  (hp: grid_points = 121) 
  (hs: grid_size = 11) 
  (hc: center_point = (6, 6))
  (htp: total_points = 120)
  (hsl: symmetric_lines = 4)
  (hpl: points_per_line = 10)
  (htsp: total_symmetric_points = 40)
  (hp: probability = 1 / 3) : 
  probability = 1 / 3 :=
by 
  sorry

end symmetry_probability_l222_222291


namespace value_of_a_l222_222982

def f (x : ℝ) : ℝ := x^2 + 9
def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 25) : a = 3 :=
by
  sorry

end value_of_a_l222_222982


namespace time_to_pass_platform_correct_l222_222481

-- Definitions based on the problem conditions
def train_speed_kmh : ℝ := 54  -- speed of the train in km/hr

def platform_length_m : ℝ := 300.024  -- length of the platform in meters

def time_pass_man_s : ℝ := 20  -- time for the train to pass the man in seconds

def speed_mps (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600  -- conversion from km/hr to m/s

def train_speed_mps : ℝ := speed_mps train_speed_kmh  -- speed of the train in m/s

def train_length_m (speed : ℝ) (time : ℝ) : ℝ := speed * time  -- length of the train in meters

def total_distance_m (train_length : ℝ) (platform_length : ℝ) : ℝ := train_length + platform_length  -- total distance when passing the platform

def time_pass_platform_s (distance : ℝ) (speed : ℝ) : ℝ := distance / speed  -- time to pass the platform

-- Correct answer
def correct_time_pass_platform : ℝ := 40.0016

-- Theorem statement
theorem time_to_pass_platform_correct : 
    time_pass_platform_s (total_distance_m (train_length_m train_speed_mps time_pass_man_s) platform_length_m) train_speed_mps 
    = correct_time_pass_platform := by
  sorry

end time_to_pass_platform_correct_l222_222481


namespace equal_sum_sequence_even_odd_l222_222834

-- Define the sequence a_n
variable {a : ℕ → ℤ}

-- Define the condition of the equal-sum sequence
def equal_sum_sequence (a : ℕ → ℤ) : Prop := ∀ n, a n + a (n + 1) = a (n + 1) + a (n + 2)

-- Statement to prove the odd terms are equal and the even terms are equal
theorem equal_sum_sequence_even_odd (a : ℕ → ℤ) (h : equal_sum_sequence a) : (∀ n, a (2 * n) = a 0) ∧ (∀ n, a (2 * n + 1) = a 1) :=
by
  sorry

end equal_sum_sequence_even_odd_l222_222834


namespace equal_diagonals_only_in_square_and_pentagon_l222_222837

-- Define regular polygon and its properties
structure RegularPolygon (n : ℕ) :=
  (is_regular : ∀ i j, distance (vertex i) (vertex j) = distance (vertex j) (vertex k))
  (vertices : fin n → Point)

-- Define the types of polygons
inductive PolygonType
  | triangle : PolygonType
  | quadrilateral : PolygonType
  | pentagon : PolygonType
  | hexagon_and_beyond : ℕ → PolygonType

open PolygonType

-- The actual Lean statement
theorem equal_diagonals_only_in_square_and_pentagon :
  ∀ (p : RegularPolygon n), (∀ i j, i ≠ j → is_diagonal i j → distance (vertices i) (vertices j) = distance (vertices k) (vertices l)) → 
  (p.type = quadrilateral ∨ p.type = pentagon) :=
by 
  sorry

end equal_diagonals_only_in_square_and_pentagon_l222_222837


namespace all_mushrooms_can_be_good_l222_222858

def is_bad (worms : ℕ) : Prop := worms ≥ 10

def redistribute_worms (bad_mushrooms good_mushrooms : ℕ) (initial_worms : ℕ) 
  (bad_worms_per_mushroom : ℕ) (good_worms_per_mushroom : ℕ) : bool :=
  if bad_mushrooms * bad_worms_per_mushroom + good_mushrooms * good_worms_per_mushroom = initial_worms then
    true
  else
    false

theorem all_mushrooms_can_be_good :
  let bad_mushrooms := 90 in
  let good_mushrooms := 10 in
  let initial_worms := 900 in
  let bad_worms_per_mushroom := 10 in
  let good_worms_per_mushroom := 0 in
  redistribute_worms bad_mushrooms good_mushrooms initial_worms bad_worms_per_mushroom good_worms_per_mushroom = true → 
  ∀ worms_in_final_mushroom,
  worms_in_final_mushroom < 10 := 
by {
  let redistributed_worms_per_mushroom := 9;
  intros;
  sorry
}

end all_mushrooms_can_be_good_l222_222858


namespace train_speed_before_accident_l222_222480

theorem train_speed_before_accident (d v : ℝ) (hv_pos : v > 0) (hd_pos : d > 0) :
  (d / ((3/4) * v) - d / v = 35 / 60) ∧
  (d - 24) / ((3/4) * v) - (d - 24) / v = 25 / 60 → 
  v = 64 :=
by
  sorry

end train_speed_before_accident_l222_222480


namespace sum_of_digits_base8_l222_222389

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222389


namespace symmetry_axis_eq_find_a_b_l222_222055

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x - π / 3) + b

theorem symmetry_axis_eq (a b : ℝ) (k : ℤ) : 
∀ x : ℝ, f a b x = f a b (x + k * π / 2) ↔ x = k * π / 2 + 5 * π / 12 := 
sorry

theorem find_a_b (a b : ℝ) (h₁ : ∀ x ∈ Set.Icc (0 : ℝ) (π / 2), f a b x = -2 → 2 * x - π / 3 = -π / 3)
  (h₂ : ∀ x ∈ Set.Icc (0 : ℝ) (π / 2), f a b x = sqrt 3 → 2 * x - π / 3 = π / 2) : 
  a = 2 ∧ b = sqrt 3 - 2 := sorry

end symmetry_axis_eq_find_a_b_l222_222055


namespace opposite_of_half_l222_222769

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222769


namespace range_g_minus_x_l222_222914

noncomputable def g : ℝ → ℝ
| x := if -2 <= x ∧ x <= -1 then -1
       else if -1 < x ∧ x <= 0 then -1
       else if 0 < x ∧ x <= 1 then 0
       else if 1 < x ∧ x <= 2 then 1
       else if 2 < x ∧ x <= 3 then 1
       else if 3 < x ∧ x <= 4 then 2
       else 0

theorem range_g_minus_x : (range (λ x, g x - x) = set.Icc (-2:ℝ) 1) :=
by
  sorry

end range_g_minus_x_l222_222914


namespace lines_coplanar_l222_222049

theorem lines_coplanar 
  (a b c : ℝ^3)
  (E : (ℝ^3)) 
  (F : (ℝ^3)) 
  (G : (ℝ^3)) 
  (H : (ℝ^3)) 
  (K : (ℝ^3)) 
  (L : (ℝ^3)) 
  (h_E : E = 0.5 • c) 
  (h_F : F = 0.5 • a) 
  (h_G : G = 0.5 • a - E) 
  (h_H : H = 0.5 • a + G) 
  (h_K : K = 0.5 • b + a) 
  (h_L : L = 0.5 • a - b) :
(E - F) + (G - H) + (K - L) = 0 :=
sorry

end lines_coplanar_l222_222049


namespace find_possible_y_values_l222_222672

noncomputable def validYValues (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) : Set ℝ :=
  { y | y = (x - 3)^2 * (x + 4) / (2 * x - 4) }

theorem find_possible_y_values (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) :
  validYValues x hx = {39, 6} :=
sorry

end find_possible_y_values_l222_222672


namespace min_value_one_over_a_plus_one_over_b_l222_222017

theorem min_value_one_over_a_plus_one_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hgeom : real.sqrt 2 = real.sqrt (2^a * 2^b)) : 
    real.sqrt 2 = real.sqrt (2^a * 2^b) → a + b = 1 → (∀ a b : ℝ, 0 < a → 0 < b → a + b = 1 → 4 = 1/a + 1/b) :=
sorry

end min_value_one_over_a_plus_one_over_b_l222_222017


namespace smallest_possible_n_l222_222295

theorem smallest_possible_n :
  ∃ n : ℕ, (∃ (A B C D : ℕ), (a = 110 * A ∧ b = 110 * B ∧ c = 110 * C ∧ d = 110 * D ∧
                        gcd A B C D = 1 ∧ lcm A B C D * 110 = n ∧
                        ∃ (k : ℕ), count_quadruplets_with_gcd_lcm 110 110000 = k) ∧ 
              n = 198000) := 
sorry

end smallest_possible_n_l222_222295


namespace hyperbola_equation_standard_area_triangle_OAB_l222_222582

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Given conditions
variables {a b : ℝ}
-- Eccentricity condition: c / a = sqrt(5)
axiom eccentricity_sqrt5 (c : ℝ) : c / a = sqrt 5
-- Imaginary axis length condition: 2b = 4
axiom imaginary_axis_length : 2 * b = 4
-- Relationship among a, b, and c: c^2 = a^2 + b^2
axiom c_squared : ∀ {c : ℝ}, c^2 = a^2 + b^2

-- Line through (0,1) with slope 45 degrees
def line_through_point (x y : ℝ) : Prop := y = x + 1

-- Points of intersection A and B and calculating their coordinates
variables {x1 y1 x2 y2 : ℝ}
axiom points_A_B : line_through_point x1 y1 ∧ hyperbola a b x1 y1 ∧ line_through_point x2 y2 ∧ hyperbola a b x2 y2

-- Use Vieta's formulas and expressions for AB
axiom vieta_x1x2 : x1 + x2 = 2/3
axiom vieta_x1x2_product : x1 * x2 = -5/3

-- Distance from the origin to the line
def distance_from_origin_to_line : ℝ := sqrt 2 / 2

-- Prove standard equation of the hyperbola
theorem hyperbola_equation_standard (a b : ℝ) : (a > 0) ∧ (b > 0) ∧ (hyperbola a b x y → (a = 1) ∧ (b = 2)) :=
by
  have a_eq : a = 1 := sorry
  have b_eq : b = 2 := sorry
  exact ⟨a_eq, b_eq, sorry⟩

-- Prove area of triangle OAB
theorem area_triangle_OAB (x1 y1 x2 y2 : ℝ) : (area_of_triangle_OAB = 4/3) :=
by
  have ab_length : abs (8 * sqrt 2 / |3|) := sorry
  have d := distance_from_origin_to_line
  exact (1 / 2) * ab_length * d = 4 / 3


end hyperbola_equation_standard_area_triangle_OAB_l222_222582


namespace toms_crab_buckets_l222_222309

def crabs_per_bucket := 12
def price_per_crab := 5
def weekly_earnings := 3360

theorem toms_crab_buckets : (weekly_earnings / (crabs_per_bucket * price_per_crab)) = 56 := by
  sorry

end toms_crab_buckets_l222_222309


namespace maximum_pieces_is_seven_l222_222252

noncomputable def max_pieces (PIE PIECE : ℕ) (n : ℕ) : Prop :=
  PIE = PIECE * n ∧ natDigits 10 PIE = List.nodup (natDigits 10 PIE) ∧ natDigits 10 PIECE = List.nodup (natDigits 10 PIECE)

theorem maximum_pieces_is_seven :
  max_pieces 95207 13601 7 :=
sorry

end maximum_pieces_is_seven_l222_222252


namespace abc_equal_183_l222_222666

noncomputable def A (x : ℝ) : ℝ := ∑ k in (Set.Ici 0), (x^(3*k) / ((Nat.factorial (3*k)) : ℝ))
noncomputable def B (x : ℝ) : ℝ := ∑ k in (Set.Ici 0), (x^(3*k+1) / ((Nat.factorial (3*k+1)) : ℝ))
noncomputable def C (x : ℝ) : ℝ := ∑ k in (Set.Ici 0), (x^(3*k+2) / ((Nat.factorial (3*k+2)) : ℝ))

theorem abc_equal_183 (x : ℝ) (hx : 0 < x) (h : A x ^ 3 + B x ^ 3 + C x ^ 3 + 8 * A x * B x * C x = 2014) :
  A x * B x * C x = 183 :=
sorry

end abc_equal_183_l222_222666


namespace part1_part2_part3_l222_222064

noncomputable def f (x m : ℝ) := 9^x - 2 * 3^(x + m)

noncomputable def g (x m : ℝ) := f x m + f (-x) m

-- Part 1
theorem part1 (x : ℝ) (h : f x 1 ≤ 27) : x ≤ 2 :=
sorry

-- Part 2
theorem part2 (x1 x2 m : ℝ) (hx : x1 * x2 = m^2) (hx1 : x1 > 0) (hx2 : x2 > x1) (hm : m > 0) :
  f x2 m > f x1 m :=
sorry

-- Part 3
theorem part3 (m : ℝ) (h : ∃ x, g x m = -11) : m = 1 :=
sorry

end part1_part2_part3_l222_222064


namespace factorize_expression_l222_222932

theorem factorize_expression (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 :=
by
  sorry

end factorize_expression_l222_222932


namespace graph_transformation_l222_222996

def transform_graph (f : ℝ → ℝ) : ℝ → ℝ :=
  let shifted := f ∘ (λ x, x + (π / 6))
  λ x, shifted (x / 2)

theorem graph_transformation :
  transform_graph (λ x : ℝ, sin (2 * x - π / 3)) = λ x, sin x :=
by
  sorry

end graph_transformation_l222_222996


namespace product_of_repeating_decimal_and_22_l222_222942

noncomputable def repeating_decimal_to_fraction : ℚ :=
  0.45 + 0.0045 * (10 ^ (-2 : ℤ))

theorem product_of_repeating_decimal_and_22 : (repeating_decimal_to_fraction * 22 = 10) :=
by
  sorry

end product_of_repeating_decimal_and_22_l222_222942


namespace investment_total_correct_l222_222893

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end investment_total_correct_l222_222893


namespace area_enclosed_by_circle_l222_222797

theorem area_enclosed_by_circle : 
  (∀ x y : ℝ, x^2 + y^2 + 10 * x + 24 * y = 0) → 
  (π * 13^2 = 169 * π):=
by
  intro h
  sorry

end area_enclosed_by_circle_l222_222797


namespace find_250th_term_l222_222318

-- Define what it means for a number not to be a perfect square
def is_not_perfect_square (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ≠ n

-- Define what it means for a number not to be a multiple of 3
def is_not_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 ≠ 0

-- Define the sequence that omits perfect squares and multiples of 3
def filtered_sequence : ℕ → ℕ
| 0     := 1
| (n+1) := Nat.find (λ m, m > filtered_sequence n ∧ is_not_perfect_square m ∧ is_not_multiple_of_3 m)

-- Define the property to be proved: the 250th term is 350
theorem find_250th_term : filtered_sequence 249 = 350 :=
by
  sorry

end find_250th_term_l222_222318


namespace coin_order_l222_222705

theorem coin_order (F C E A D B : Type) :
  (F ≠ C) → (F ≠ E) → (F ≠ A) → (F ≠ D) → (F ≠ B) →
  (C ≠ E) → (C ≠ A) → (C ≠ D) → (C ≠ B) →
  (E ≠ A) → (E ≠ D) → (E ≠ B) →
  (A ≠ D) → (A ≠ B) →
  (D ≠ B) →
  (above F C) → (above F E) → (above F A) → (above F D) → (above F B) →
  (above C A) → (above C B) → (above C D) →
  (above E A) → (above E B) → (coincides E D) →
  (above D B) →
  order_coins [F, C, E, D, A, B] := sorry

end coin_order_l222_222705


namespace inscribed_to_circumscribed_ratio_l222_222636

theorem inscribed_to_circumscribed_ratio (a b : ℕ)
  (h1 : 6 = a) (h2 : 8 = b) :
  let c := (a^2 + b^2).sqrt
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2 in
  inscribed_radius / circumscribed_radius = 2 / 5 :=
by
  have ha : a = 6 := h1
  have hb : b = 8 := h2
  let c : ℕ := Int.sqrt ((a: ℤ)^2 + (b: ℤ)^2) -- Hypotenuse length
  let inscribed_radius := (a + b - c) / 2
  let circumscribed_radius := c / 2
  have hc : c = 10 := by sorry
  have h_inscribed : inscribed_radius = 2 := by sorry
  have h_circumscribed : circumscribed_radius = 5 := by sorry
  rw [h_inscribed, h_circumscribed]
  norm_num

end inscribed_to_circumscribed_ratio_l222_222636


namespace find_250th_term_l222_222317

-- Define what it means for a number not to be a perfect square
def is_not_perfect_square (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ≠ n

-- Define what it means for a number not to be a multiple of 3
def is_not_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 ≠ 0

-- Define the sequence that omits perfect squares and multiples of 3
def filtered_sequence : ℕ → ℕ
| 0     := 1
| (n+1) := Nat.find (λ m, m > filtered_sequence n ∧ is_not_perfect_square m ∧ is_not_multiple_of_3 m)

-- Define the property to be proved: the 250th term is 350
theorem find_250th_term : filtered_sequence 249 = 350 :=
by
  sorry

end find_250th_term_l222_222317


namespace smallest_element_in_T_l222_222667

-- Define the set of integers from 1 to 20
def I := { n : ℤ | 1 ≤ n ∧ n ≤ 20 }

-- Define prime check function
def is_prime (n : ℤ) : Prop := 
  nat.prime (int.to_nat n)

-- Define a valid set T based on the given conditions
def valid_set_T (T : set ℤ) : Prop :=
  -- T is a subset of I
  T ⊆ I ∧
  -- T contains 8 elements
  T.card = 8 ∧
  -- If x < y and x, y ∈ T, then y is not a multiple of x
  (∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧
  -- The largest number in T is a prime number
  is_prime (T.sup id)

-- Statement to prove the smallest element in T
theorem smallest_element_in_T : ∃ T : set ℤ, valid_set_T T ∧ T.min id = 4 := 
sorry

end smallest_element_in_T_l222_222667


namespace money_invested_time_l222_222012

theorem money_invested_time :
  let P := 810
  let R := 4.783950617283951 / 100 -- Convert percentage to decimal
  let SI := 155
  let time := (SI * 100) / (P * R)
  abs (time - 4) < 1 :=
by
  let P := 810
  let R := 4.783950617283951 / 100
  let SI := 155
  let time := (SI * 100) / (P * R)
  show abs (time - 4) < 1
  sorry

end money_invested_time_l222_222012


namespace product_of_union_eq_zero_l222_222621

open Set

noncomputable def A (x y : ℝ) : Set ℝ := {2 * x, 3 * y}
noncomputable def B (x y : ℝ) : Set ℝ := {6, x * y}

theorem product_of_union_eq_zero (x y : ℝ) (h : ∃! a, a ∈ A x y ∧ a ∈ B x y) : 
  let union := A x y ∪ B x y in
  ∏ a in union, a = 0 :=
sorry

end product_of_union_eq_zero_l222_222621


namespace inequality_solution_l222_222047

theorem inequality_solution (a b m x : ℝ) :
  (∃ a = 1 ∧ ∃ b = 2 ∧ (s : Set ℝ) (h_s : s = { x | m < x ∧ x < 2 } ↔ m < 2 ∨ s = ∅ ↔ m = 2 ∨ s = { x | 2 < x ∧ x < m } ↔ m > 2)) →
  SetOf (x^2 - 3 * a * x + b > 0) = SetOf (x < 1 ∨ x > 2) :=
by
  sorry

end inequality_solution_l222_222047


namespace find_angle_C_find_a_plus_b_l222_222129

-- Define the geometric conditions
variable {A B C : ℝ}
variable {a b c : ℝ}

-- Condition 1: The sides opposite to angles A, B, and C are a, b, and c respectively.
def sides_opposite (a b c : ℝ) (A B C : ℝ) (hA : a = opposite A) (hB : b = opposite B) (hC : c = opposite C) : Prop :=
  a = opposite A ∧ b = opposite B ∧ c = opposite C

-- Condition 2: Given relation 2cos(C)(a*cos(B) + b*cos(A)) = c
def given_relation (a b c A B C : ℝ) : Prop :=
  2 * real.cos C * (a * real.cos B + b * real.cos A) = c

-- Condition 3: c = sqrt(7)
def condition_c_sqrt7 (c : ℝ) : Prop :=
  c = real.sqrt 7

-- Condition 4: The area of triangle ABC is 3sqrt(3)/2
def area_of_triangle (a b C : ℝ) (Area_given : ℝ) : Prop :=
  (1 / 2) * a * b * real.sin C = Area_given

-- Theorems to Prove
theorem find_angle_C {a b c A B C : ℝ} 
  (h1 : sides_opposite a b c A B C)
  (h2 : given_relation a b c A B C) :
  C = π / 3 :=
sorry

theorem find_a_plus_b {a b c A B C : ℝ} 
  (h1 : sides_opposite a b c A B C)
  (h2 : given_relation a b c A B C)
  (h3 : condition_c_sqrt7 c)
  (h4 : area_of_triangle a b C (3 * real.sqrt 3 / 2)) :
  a + b = 5 :=
sorry

end find_angle_C_find_a_plus_b_l222_222129


namespace greatest_possible_k_l222_222271

theorem greatest_possible_k (k : ℝ) (h : ∀ x, x^2 + k * x + 8 = 0) (diff_roots : (∀ a b, a ≠ b → a - b = sqrt 73)) : k = sqrt 105 :=
by
  sorry

end greatest_possible_k_l222_222271


namespace common_factor_of_polynomial_l222_222232

theorem common_factor_of_polynomial :
  ∀ (x : ℝ), (2 * x^2 - 8 * x) = 2 * x * (x - 4) := by
  sorry

end common_factor_of_polynomial_l222_222232


namespace sum_of_digits_base_8_rep_of_888_l222_222352

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222352


namespace intersection_points_in_circle_l222_222222

open Classical

noncomputable def num_sides := [6, 7, 8, 9]

def intersection_points (n m : ℕ) : ℕ :=
  if n < m then 2 * n else 2 * m

def total_intersections : ℕ :=
  ∑ i in num_sides.toFinset, ∑ j in num_sides.toFinset, if i < j then intersection_points i j else 0

theorem intersection_points_in_circle : total_intersections = 80 := by
  sorry

end intersection_points_in_circle_l222_222222


namespace taxi_fare_l222_222657

noncomputable def initial_fee : ℝ := 2.25
noncomputable def cost_first_segment : ℝ := 0.35
noncomputable def length_first_segment : ℝ := 0.25
noncomputable def max_first_segment : ℝ := 2

noncomputable def cost_second_segment : ℝ := 0.25
noncomputable def length_second_segment : ℝ := 2 / 5
noncomputable def end_second_segment : ℝ := 5

noncomputable def cost_final_segment : ℝ := 0.15
noncomputable def length_final_segment : ℝ := 0.5

noncomputable def peak_hour_surcharge : ℝ := 3.0
noncomputable def total_miles : ℝ := 7.8

theorem taxi_fare :
  ∀ (initial_fee cost_first_segment length_first_segment max_first_segment
      cost_second_segment length_second_segment end_second_segment
      cost_final_segment length_final_segment peak_hour_surcharge total_miles : ℝ),
  initial_fee + 
  (cost_first_segment * (max_first_segment / length_first_segment)) + 
  (cost_second_segment * ((end_second_segment - max_first_segment) / length_second_segment)) + 
  (cost_final_segment * ((total_miles - end_second_segment) / length_final_segment).ceil) + 
  peak_hour_surcharge = 12.70 :=
by
  intros
  sorry

end taxi_fare_l222_222657


namespace sum_of_digits_base8_888_l222_222334

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222334


namespace tan_beta_l222_222538

noncomputable def tan_eq_2 (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) : Real :=
2

theorem tan_beta (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) :
  Real.tan β = tan_eq_2 α β h1 h2 := by
  sorry

end tan_beta_l222_222538


namespace sum_of_digits_base_8_rep_of_888_l222_222345

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222345


namespace tan_of_cos_and_range_complex_trig_expression_l222_222039

variables (α : ℝ) (h_cos : cos α = -sqrt 5 / 5) (h_range : π < α ∧ α < 3 * π / 2)

theorem tan_of_cos_and_range (h_cos : cos α = -sqrt 5 / 5) (h_range : π < α ∧ α < 3 * π / 2) : 
  tan α = 2 :=
sorry

theorem complex_trig_expression (h_cos : cos α = -sqrt 5 / 5) (h_range : π < α ∧ α < 3 * π / 2) :
  (3 * sin (π + α) + cos (3 * π - α)) / (sin (3 * π / 2 + α) + 2 * sin (α - 2 * π)) = -7 / 3 :=
sorry

end tan_of_cos_and_range_complex_trig_expression_l222_222039


namespace mike_picked_32_limes_l222_222881

theorem mike_picked_32_limes (total_limes : ℕ) (alyssa_limes : ℕ) (mike_limes : ℕ) 
  (h1 : total_limes = 57) (h2 : alyssa_limes = 25) (h3 : mike_limes = total_limes - alyssa_limes) : 
  mike_limes = 32 :=
by
  sorry

end mike_picked_32_limes_l222_222881


namespace sufficient_condition_not_necessary_condition_main_theorem_l222_222975

open Real

-- Definition of α
def α := π / 6

-- Define the condition that tan(2α) = sqrt(3)
def tanCondition := tan (2 * α) = sqrt 3

-- Proof that α = π / 6 is a sufficient condition for tan(2 * α) = sqrt 3
theorem sufficient_condition : (α = π / 6) → tanCondition :=
by
  intro h
  rw [h, α]
  sorry

-- Proof that α = π / 6 is not a necessary condition for tan(2 * α) = sqrt 3
theorem not_necessary_condition : ¬ ((tanCondition) → (α = π / 6)) :=
by
  intro h
  have h' : tan (2 * (π / 3)) = sqrt 3 := sorry
  have neq : π / 3 ≠ π / 6 := sorry
  specialize h h'
  contradiction

-- Main theorem which connects both proofs to match "A" condition
theorem main_theorem : (α = π / 6) ∧ ¬ (α = π / 6 ↔ tanCondition) :=
by
  split
  apply sufficient_condition
  apply not_necessary_condition

end sufficient_condition_not_necessary_condition_main_theorem_l222_222975


namespace no_natural_number_solution_l222_222928

theorem no_natural_number_solution :
  ¬∃ (n : ℕ), ∃ (k : ℕ), (n^5 - 5*n^3 + 4*n + 7 = k^2) :=
sorry

end no_natural_number_solution_l222_222928


namespace packets_of_candy_bought_l222_222902

theorem packets_of_candy_bought
    (candies_per_day_weekday : ℕ)
    (candies_per_day_weekend : ℕ)
    (days_weekday : ℕ)
    (days_weekend : ℕ)
    (weeks : ℕ)
    (candies_per_packet : ℕ)
    (total_candies : ℕ)
    (packets_bought : ℕ) :
    candies_per_day_weekday = 2 →
    candies_per_day_weekend = 1 →
    days_weekday = 5 →
    days_weekend = 2 →
    weeks = 3 →
    candies_per_packet = 18 →
    total_candies = (candies_per_day_weekday * days_weekday + candies_per_day_weekend * days_weekend) * weeks →
    packets_bought = total_candies / candies_per_packet →
    packets_bought = 2 :=
by
  intros
  sorry

end packets_of_candy_bought_l222_222902


namespace table_transformation_max_n_moves_l222_222120

theorem table_transformation_max_n_moves (n : ℕ) (initial_config: ℕ → ℕ → Bool) 
  (config_transformation: (ℕ → ℕ → Bool) → Prop) :
  (∃ moves : list (ℕ ⊕ ℕ), 
    length moves ≤ n 
    ∧ config_transformation (apply_moves moves initial_config)) :=
sorry

end table_transformation_max_n_moves_l222_222120


namespace sum_of_digits_base8_l222_222390

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222390


namespace residue_mod_neg_935_mod_24_l222_222518

theorem residue_mod_neg_935_mod_24 : (-935) % 24 = 1 :=
by
  sorry

end residue_mod_neg_935_mod_24_l222_222518


namespace number_of_distinct_ordered_pairs_l222_222037

noncomputable def num_distinct_ordered_pairs (s : Finset ℕ) (A B : Finset ℕ) : ℕ :=
  if (A ∩ B).nonempty ∧ (A ∪ B = s) ∧ A ≠ B then 1 else 0

theorem number_of_distinct_ordered_pairs :
  ∑ x ∈ (Finset.powerset (Finset.range 6) \ (Finset.singleton ∅)), 
    ∑ y ∈ (Finset.powerset (Finset.range 6) \ (Finset.singleton ∅)), 
      num_distinct_ordered_pairs (Finset.range 6) x y = 211 :=
by
  sorry

end number_of_distinct_ordered_pairs_l222_222037


namespace coin_flips_count_l222_222613

noncomputable def probability_tails_first_flip : ℝ := 1 / 2

noncomputable def probability_heads_last_two_flips : ℝ := (1 / 2) * (1 / 2)

noncomputable def total_probability (n : ℕ) : ℝ :=
  probability_tails_first_flip * (1 / 2)^(n - 3) * probability_heads_last_two_flips

theorem coin_flips_count (n : ℕ) (h1 : probability_tails_first_flip = 1 / 2)
  (h2 : total_probability n = 0.125) : n = 4 :=
begin
  sorry,
end

end coin_flips_count_l222_222613


namespace compare_areas_of_regular_shapes_l222_222560

theorem compare_areas_of_regular_shapes :
  ∀ (l : ℝ) (h₁ : 12 * l > 0), 
  let S3 := 4 * Real.sqrt 3 * l^2,
      S4 := 9 * l^2,
      S6 := 6 * Real.sqrt 3 * l^2 in
  S6 > S4 ∧ S4 > S3 := 
by
  intro l h₁
  let S3 := 4 * Real.sqrt 3 * l^2
  let S4 := 9 * l^2
  let S6 := 6 * Real.sqrt 3 * l^2
  sorry

end compare_areas_of_regular_shapes_l222_222560


namespace convert_spherical_coordinates_l222_222548

theorem convert_spherical_coordinates (
  ρ θ φ : ℝ
) (h1 : ρ = 5) (h2 : θ = 3 * Real.pi / 4) (h3 : φ = 9 * Real.pi / 4) : 
ρ = 5 ∧ 0 ≤ 7 * Real.pi / 4 ∧ 7 * Real.pi / 4 < 2 * Real.pi ∧ 0 ≤ Real.pi / 4 ∧ Real.pi / 4 ≤ Real.pi :=
by
  sorry

end convert_spherical_coordinates_l222_222548


namespace problem_statement_l222_222955

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ (ab < 1) :=
by
  sorry

end problem_statement_l222_222955


namespace crop_planting_ways_l222_222854

def Crop := {corn, wheat, soybeans, potatoes, rice}

def Grid (r c : Nat) := { sections : Array (Array (Option Crop)) // sections.size = r ∧ sections.all (λ row => row.size = c) }

def isValidGrid (g : Grid 3 2) : Bool :=
  g.sections.all (λ row, row.all (λ section, section.isSome)) ∧
  ∀ (i j : Nat), i < 3 ∧ j < 2 → (
    (g.sections[i][j] = some corn → (if i > 0 then g.sections[i-1][j] ≠ some wheat ∧ g.sections[i-1][j] ≠ some rice else True) ∧
                                   (if i < 2 then g.sections[i+1][j] ≠ some wheat ∧ g.sections[i+1][j] ≠ some rice else True) ∧
                                   (if j > 0 then g.sections[i][j-1] ≠ some wheat ∧ g.sections[i][j-1] ≠ some rice else True) ∧
                                   (if j < 1 then g.sections[i][j+1] ≠ some wheat ∧ g.sections[i][j+1] ≠ some rice else True)) ∧
    (g.sections[i][j] = some potatoes → (if i > 0 then g.sections[i-1][j] ≠ some soybeans ∧ g.sections[i-1][j] ≠ some rice else True) ∧
                                       (if i < 2 then g.sections[i+1][j] ≠ some soybeans ∧ g.sections[i+1][j] ≠ some rice else True) ∧
                                       (if j > 0 then g.sections[i][j-1] ≠ some soybeans ∧ g.sections[i][j-1] ≠ some rice else True) ∧
                                       (if j < 1 then g.sections[i][j+1] ≠ some soybeans ∧ g.sections[i][j+1] ≠ some rice else True)))

theorem crop_planting_ways : ∃ (g : List (Grid 3 2)), g.length = 215 ∧ g.all isValidGrid := 
sorry

end crop_planting_ways_l222_222854


namespace complex_subtraction_l222_222796

-- Define the complex numbers c and d
def c : ℂ := 5 - 3 * complex.I
def d : ℂ := 2 + 4 * complex.I

-- State that c - 3 * d is equal to -1 - 15 * I
theorem complex_subtraction : c - 3 * d = -1 - 15 * complex.I := 
sorry

end complex_subtraction_l222_222796


namespace screen_time_morning_l222_222713

def total_screen_time : ℕ := 120
def evening_screen_time : ℕ := 75
def morning_screen_time : ℕ := 45

theorem screen_time_morning : total_screen_time - evening_screen_time = morning_screen_time := by
  sorry

end screen_time_morning_l222_222713


namespace option_d_always_correct_l222_222041

variable {a b : ℝ}

theorem option_d_always_correct (h1 : a < b) (h2 : b < 0) (h3 : a < 0) :
  (a + 1 / b)^2 > (b + 1 / a)^2 :=
by
  -- Lean proof code would go here.
  sorry

end option_d_always_correct_l222_222041


namespace sum_of_digits_base8_888_l222_222409

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222409


namespace Cally_colored_shirts_l222_222906

theorem Cally_colored_shirts (C : ℕ) (hcally : 10 + 7 + 6 = 23) (hdanny : 6 + 8 + 10 + 6 = 30) (htotal : 23 + 30 + C = 58) : 
  C = 5 := 
by
  sorry

end Cally_colored_shirts_l222_222906


namespace nat_values_of_x_l222_222936

theorem nat_values_of_x :
  (∃ (x : ℕ), 2^(x - 5) = 2 ∧ x = 6) ∧
  (∃ (x : ℕ), 2^x = 512 ∧ x = 9) ∧
  (∃ (x : ℕ), x^5 = 243 ∧ x = 3) ∧
  (∃ (x : ℕ), x^4 = 625 ∧ x = 5) :=
  by {
    sorry
  }

end nat_values_of_x_l222_222936


namespace opposite_neg_half_l222_222743

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l222_222743


namespace avg_terminals_used_l222_222629

noncomputable def avgTerminalsUsed (n : ℕ) (p : ℝ) := 
  if h : 0 ≤ p ∧ p ≤ 1 then n * p else 0 

theorem avg_terminals_used (n : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : avgTerminalsUsed n p = n * p := 
  by 
  unfold avgTerminalsUsed 
  simp [h] 
  sorry

end avg_terminals_used_l222_222629


namespace factorize1_factorize2_factorize3_l222_222525

-- Proof problem 1: Prove m^2 + 4m + 4 = (m + 2)^2
theorem factorize1 (m : ℝ) : m^2 + 4 * m + 4 = (m + 2)^2 :=
sorry

-- Proof problem 2: Prove a^2 b - 4ab^2 + 3b^3 = b(a-b)(a-3b)
theorem factorize2 (a b : ℝ) : a^2 * b - 4 * a * b^2 + 3 * b^3 = b * (a - b) * (a - 3 * b) :=
sorry

-- Proof problem 3: Prove (x^2 + y^2)^2 - 4x^2 y^2 = (x + y)^2 (x - y)^2
theorem factorize3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

end factorize1_factorize2_factorize3_l222_222525


namespace elimination_of_3_cliques_l222_222516

open Finset

variable {V : Type} [DecidableEq V]

-- Define what it means to be a k-clique
def is_k_clique (G : SimpleGraph V) (k : ℕ) (C : Finset V) : Prop :=
  C.card = k ∧ ∀ (a b : V), a ∈ C → b ∈ C → a ≠ b → G.adj a b

-- Assume conditions: Every two 3-cliques share at least one vertex and no 5-cliques exist
variable (G : SimpleGraph V)
variable (H1 : ∀ C1 C2 : Finset V, is_k_clique G 3 C1 → is_k_clique G 3 C2 → (C1 ∩ C2).nonempty)
variable (H2 : ¬ ∃ C : Finset V, is_k_clique G 5 C)

-- The proof goal: There exist at most two vertices whose removal eliminates all 3-cliques
theorem elimination_of_3_cliques (G : SimpleGraph V) (H1 : ∀ C1 C2 : Finset V, is_k_clique G 3 C1 → is_k_clique G 3 C2 → (C1 ∩ C2).nonempty)
  (H2 : ¬ ∃ C : Finset V, is_k_clique G 5 C) : ∃ (S : Finset V), S.card ≤ 2 ∧ ∀ C : Finset V, is_k_clique G 3 C → (C ∩ S).nonempty :=
sorry

end elimination_of_3_cliques_l222_222516


namespace coefficient_a7_l222_222025

theorem coefficient_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (x : ℝ) 
  (h : x^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 
          + a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 
          + a_8 * (x - 1)^8 + a_9 * (x - 1)^9) : 
  a_7 = 36 := 
by
  sorry

end coefficient_a7_l222_222025


namespace time_elephants_l222_222795

def total_time := 130
def time_seals := 13
def time_penguins := 8 * time_seals

theorem time_elephants : total_time - (time_seals + time_penguins) = 13 :=
by
  sorry

end time_elephants_l222_222795


namespace line_intersects_ellipse_chord_length_condition_l222_222988

noncomputable def ellipse (x y : ℝ) := 4 * x^2 + y^2 = 1

noncomputable def line (x m : ℝ) := x + m

noncomputable def chord_length (m : ℝ) : ℝ :=
  Real.sqrt 2 * Real.sqrt ((-2 * m / 5) ^ 2 - 4 * (m ^ 2 - 1) / 5) = 2 * Real.sqrt 10 / 5

theorem line_intersects_ellipse (m : ℝ) : 
  (-Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2) ↔ 
  ∃ x y, ellipse x y ∧ y = line x m := sorry

theorem chord_length_condition (m : ℝ) : 
  chord_length m = 2 * Real.sqrt 10 / 5 ↔ m = 0 := sorry

end line_intersects_ellipse_chord_length_condition_l222_222988


namespace second_derivative_at_pi_over_2_l222_222058

def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem second_derivative_at_pi_over_2 :
  deriv (deriv f) (Real.pi / 2) = 1 :=
by
  sorry

end second_derivative_at_pi_over_2_l222_222058


namespace cartesian_equation_of_trajectory_maximum_area_of_triangle_l222_222641

/-- Prove that the Cartesian equation of the trajectory of point Q is x + y = 4, given the polar
    equation ρ = sin θ + cos θ and the condition |OP|*|OQ| = 4. -/
theorem cartesian_equation_of_trajectory
  (ρ θ : ℝ)
  (condition1 : ρ = sin θ + cos θ)
  (OP OQ : ℝ)
  (condition2 : OP * OQ = 4) :
  ∃ (x y : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ ∧ x + y = 4 :=
sorry

/-- Prove that the maximum area of the triangle MOP is 2√2,
    given point M with coordinates (4, 3π/4) and the polar equation ρ = sin θ + cos θ. -/
theorem maximum_area_of_triangle
  (M : ℝ × ℝ)
  (conditionM : M = (4, 3 * Real.pi / 4))
  (ρ θ : ℝ)
  (condition1 : ρ = sin θ + cos θ) :
  ∃ max_area : ℝ, max_area = 2 * Real.sqrt 2 :=
sorry

end cartesian_equation_of_trajectory_maximum_area_of_triangle_l222_222641


namespace U_vec3_l222_222782

-- Definitions of vectors
def vec1 : ℝ × ℝ × ℝ := (2, 4, 3)
def vec2 : ℝ × ℝ × ℝ := (-3, 2, 4)
def vec3 : ℝ × ℝ × ℝ := (1, 6, 7)
def vecU1V1 := (3, -2, 1)
def vecU2V2 := (-1, 2, 3)

-- Cross product definition
def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v1.2 * v2.3 - v1.3 * v2.2), (v1.3 * v2.1 - v1.1 * v2.3), (v1.1 * v2.2 - v1.2 * v2.1))

-- Transformation U definition (can leave it abstract to apply conditions)
axiom U : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ

-- Conditions:
axiom linearity : ∀ (a b : ℝ) (v w : ℝ × ℝ × ℝ), U (a • v + b • w) = a • U v + b • U w
axiom cross_prod_preserving : ∀ (v w : ℝ × ℝ × ℝ), U (cross_product v w) = cross_product (U v) (U w)
axiom U_vec1 : U vec1 = vecU1V1
axiom U_vec2 : U vec2 = vecU2V2

-- The proof statement
theorem U_vec3 : U vec3 = (-7/2, 4, -2) :=
  sorry

end U_vec3_l222_222782


namespace cut_geometry_into_identical_parts_l222_222918

-- Assuming N is the total number of grid squares and the figure is represented 
-- as a finite set of points within a rectangular grid.
variable (N : ℕ) 
variable (grid : ℕ × ℕ → Prop) -- A property that represents the grid line structure 

-- Defining the problem statement
theorem cut_geometry_into_identical_parts (h_symmetric : ∃ line, 
  ∀ (x y : ℕ × ℕ), grid x → grid y → x = y ∨ x = line ∨ y = line):
  ∃ (part1 part2 : set (ℕ × ℕ)), (∀ x, grid x → x ∈ part1 ∪ part2) ∧ 
  (∀ x, x ∈ part1 → x ∉ part2) ∧ (∀ x, x ∈ part2 → x ∉ part1) ∧ 
  set.card part1 = set.card part2 ∧ 
  ∃ symmetric_line, 
  (∀ x, (x ∈ part1 ↔ symmetric_line x)) ∧ 
  (∀ x, (x ∈ part2 ↔ symmetric_line x)) :=
begin
  sorry
end

end cut_geometry_into_identical_parts_l222_222918


namespace solve_for_x_l222_222450

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l222_222450


namespace greatest_possible_k_l222_222268

theorem greatest_possible_k (k : ℂ) (h : ∃ x1 x2 : ℂ, x1 ≠ x2 ∧ x1 + x2 = -k ∧ x1 * x2 = 8 ∧ |x1 - x2| = sqrt 73) : k = sqrt 105 :=
sorry

end greatest_possible_k_l222_222268


namespace number_of_oranges_l222_222694

theorem number_of_oranges :
  ∃ o m : ℕ, (m + 6 * m + o = 20) ∧ (6 * m > o) ∧ (2 ≤ m) ∧ (m ≤ 2) ∧ (o = 6) :=
begin
  -- instantiating variables m and o
  use [2, 6],
  -- prove and this would skip the proof,
  split, linarith, 
  split, linarith,
  split, linarith,
  linarith,
end

end number_of_oranges_l222_222694


namespace solve_sine_equation_l222_222823

theorem solve_sine_equation (x : ℝ) (k : ℤ) (h : |Real.sin x| ≠ 1) :
  (8.477 * ((∑' n, Real.sin x ^ n) / (∑' n, ((-1 : ℝ) * Real.sin x) ^ n)) = 4 / (1 + Real.tan x ^ 2)) 
  ↔ (x = (-1)^k * (Real.pi / 6) + k * Real.pi) :=
by
  sorry

end solve_sine_equation_l222_222823


namespace cube_root_of_6880_l222_222980

theorem cube_root_of_6880 :
  ∀ (c₁ : ℝ) (c₂ : ℝ),
  c₁ = 4.098 →
  c₂ = 1.902 →
  ∛6880 = 19.02 :=
by
  intros c₁ c₂ hc₁ hc₂
  sorry

end cube_root_of_6880_l222_222980


namespace percent_increase_expenditure_l222_222625

theorem percent_increase_expenditure (cost_per_minute_2005 minutes_2005 minutes_2020 total_expenditure_2005 total_expenditure_2020 : ℕ)
  (h1 : cost_per_minute_2005 = 10)
  (h2 : minutes_2005 = 200)
  (h3 : minutes_2020 = 2 * minutes_2005)
  (h4 : total_expenditure_2005 = minutes_2005 * cost_per_minute_2005)
  (h5 : total_expenditure_2020 = minutes_2020 * cost_per_minute_2005) :
  ((total_expenditure_2020 - total_expenditure_2005) * 100 / total_expenditure_2005) = 100 :=
by
  sorry

end percent_increase_expenditure_l222_222625


namespace coefficient_of_x7_in_expansion_l222_222721

theorem coefficient_of_x7_in_expansion :
  let f := (2 * x - 1) * ((1 / x) + 2 * x) ^ 6 
  in coefficient x 7 ((2 * x - 1) * ((1 / x) + 2 * x) ^ 6) = 128 :=
begin
  sorry
end

end coefficient_of_x7_in_expansion_l222_222721


namespace chicken_farm_l222_222785

def total_chickens (roosters hens : ℕ) : ℕ := roosters + hens

theorem chicken_farm (roosters hens : ℕ) (h1 : 2 * hens = roosters) (h2 : roosters = 6000) : 
  total_chickens roosters hens = 9000 :=
by
  sorry

end chicken_farm_l222_222785


namespace cubic_meters_to_cubic_feet_l222_222595

theorem cubic_meters_to_cubic_feet :
  (let feet_per_meter := 3.28084
  in (feet_per_meter ^ 3) * 2 = 70.6294) :=
by
  sorry

end cubic_meters_to_cubic_feet_l222_222595


namespace interval_of_decrease_l222_222618

/-- Given the derivative f'(x) = 2x - 4, proving that the interval of decrease for the function f(x-1) is (-∞, 3). -/
theorem interval_of_decrease (f : ℝ → ℝ) (hf_deriv : ∀ x, deriv f x = 2 * x - 4) :
  (∀ x, deriv (λ x, f (x - 1)) x < 0 ↔ x < 3) :=
sorry

end interval_of_decrease_l222_222618


namespace sum_of_digits_base8_888_l222_222331

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222331


namespace batman_game_cost_l222_222310

theorem batman_game_cost (total_spent superman_cost : ℝ) 
  (H1 : total_spent = 18.66) (H2 : superman_cost = 5.06) :
  total_spent - superman_cost = 13.60 :=
by
  sorry

end batman_game_cost_l222_222310


namespace greatest_possible_value_of_k_l222_222264

theorem greatest_possible_value_of_k :
  ∃ k : ℝ, 
    (∀ (x: ℝ), x^2 + k * x + 8 = 0 → 
      ∃ α β : ℝ, α ≠ β ∧ α - β = sqrt 73) → 
      k = sqrt 105 := 
sorry

end greatest_possible_value_of_k_l222_222264


namespace graph_covered_by_rectangles_l222_222422

theorem graph_covered_by_rectangles
  (F : ℝ → ℝ)
  (h_inc : ∀ x1 x2, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ 1 → F x1 ≤ F x2)
  (h_def : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ F x ∧ F x ≤ 1) :
  ∀ n : ℕ, 
  ∃ (N : ℕ) (rectangles : list (set (ℝ × ℝ))),
  (N = n ∧ ∀ r ∈ rectangles, ∃ (a b c d : ℝ),
    (0 ≤ a ∧ a ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ d ∧ d ≤ 1 ∧ 
    (set.prod (set.Icc a b) (set.Icc c d) ⊆ r) ∧
    set.measure (set.prod (set.Icc a b) (set.Icc c d)) = 1/(2*n))) :=
sorry

end graph_covered_by_rectangles_l222_222422


namespace max_subset_l222_222150

open Set

noncomputable def max_n (n : ℕ) :=
  ∃ A : Finset ℕ, A ⊆ (Finset.range 2022).filter (λ x, x > 0) ∧
    (∀ x y ∈ A, x ≠ y → ¬(Nat.coprime x y) ∧ ¬(Nat.divisible x y)) ∧
    A.card = n

theorem max_subset :
  max_n 505 :=
begin
  -- Proof omitted
  sorry
end

end max_subset_l222_222150


namespace number_of_irrational_numbers_is_one_l222_222494

theorem number_of_irrational_numbers_is_one :
  let numbers := [-3.14, 0, Real.pi, 22 / 7, 0.1010010001]
  in (count_irrational numbers = 1) :=
by
  sorry

def count_irrational (nums : List ℝ) : ℕ :=
  nums.countp (λ x, ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b)

end number_of_irrational_numbers_is_one_l222_222494


namespace probability_defective_product_probability_A1_given_defective_l222_222850

theorem probability_defective_product (P_A1 P_A2 P_A3 P_B_given_A1 P_B_given_A2 P_B_given_A3 : ℝ)
  (h1 : P_A1 = 0.25) (h2 : P_A2 = 0.35) (h3 : P_A3 = 0.4) 
  (h4 : P_B_given_A1 = 0.05) (h5 : P_B_given_A2 = 0.04) (h6 : P_B_given_A3 = 0.02) :
  (P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) = 0.0345 := 
by 
  -- Proof skipped
  sorry

theorem probability_A1_given_defective (P_A1 P_A2 P_A3 P_B_given_A1 P_B_given_A2 P_B_given_A3 : ℝ)
  (h1 : P_A1 = 0.25) (h2 : P_A2 = 0.35) (h3 : P_A3 = 0.4) 
  (h4 : P_B_given_A1 = 0.05) (h5 : P_B_given_A2 = 0.04) (h6 : P_B_given_A3 = 0.02)
  (P_B : ℝ) (hP_B : P_B = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3) :
  (P_A1 * P_B_given_A1 / P_B) ≈ 0.36 := 
by 
  -- Proof skipped
  sorry

end probability_defective_product_probability_A1_given_defective_l222_222850


namespace sum_of_digits_base8_888_l222_222380

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222380


namespace A_B_work_together_finish_l222_222489
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l222_222489


namespace horner_operations_count_l222_222793

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

theorem horner_operations_count : 
  let x := 0.4 in
  let total_operations := 12 in
  (∃ (multiplications additions : ℕ), 
    multiplications = 6 ∧ 
    additions = 6 ∧ 
    multiplications + additions = total_operations) :=
by
  sorry

end horner_operations_count_l222_222793


namespace required_speed_l222_222824

-- The car covers 504 km in 6 hours initially.
def distance : ℕ := 504
def initial_time : ℕ := 6
def initial_speed : ℕ := distance / initial_time

-- The time that is 3/2 times the initial time.
def factor : ℚ := 3 / 2
def new_time : ℚ := initial_time * factor

-- The speed required to cover the same distance in the new time.
def new_speed : ℚ := distance / new_time

-- The proof statement
theorem required_speed : new_speed = 56 := by
  sorry

end required_speed_l222_222824


namespace find_chord_line_eq_l222_222616

theorem find_chord_line_eq (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ)
    (hP : P = (1, 1)) (hC : C = (3, 0)) (hr : r = 3)
    (circle_eq : ∀ (x y : ℝ), (x - 3)^2 + y^2 = r^2) :
    ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = -1 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := by
  sorry

end find_chord_line_eq_l222_222616


namespace part_a_part_b_l222_222712

-- Definitions for conditions
def colored (plane : Type) := plane → (ℕ → bool)

-- Statement for part (a)
theorem part_a (plane : Type) (color : colored plane)
    (h : ∀ A B C : plane, ¬(equilateral_triangle A B C) ∨ (color A = color B ∧ color B = color C)) :
    ∃ A B C : plane, (color A = color B ∧ color B = color C) ∧ midpoint A B C :=
begin
  sorry
end

-- Statement for part (b)
theorem part_b (plane : Type) (color : colored plane) :
    ∃ A B C : plane, equilateral_triangle A B C ∧ (color A = color B ∧ color B = color C) :=
begin
  sorry
end

end part_a_part_b_l222_222712


namespace linear_function_not_in_third_quadrant_l222_222774

noncomputable def quadratic_eq_roots_sum (a b : ℝ) : Prop := a + b = 2
noncomputable def quadratic_eq_roots_product (a b : ℝ) : Prop := a * b = -3
noncomputable def linear_function (a b : ℝ) : ℝ → ℝ := λ x, (a * b - 1) * x + (a + b)

theorem linear_function_not_in_third_quadrant (a b : ℝ) (quad_sum : quadratic_eq_roots_sum a b) (quad_prod : quadratic_eq_roots_product a b) :
  ¬ ∃ x y : ℝ, y = linear_function a b x ∧ x < 0 ∧ y < 0 :=
sorry

end linear_function_not_in_third_quadrant_l222_222774


namespace subset_contains_square_product_l222_222225

variable V : Set ℕ := { x | x > 0 ∧ x < 26 }
variable S : Set ℕ
variable hS : S ⊆ V ∧ S.card ≥ 17

theorem subset_contains_square_product (S ⊆ V ∧ S.card ≥ 17) :
  ∃ x y ∈ S, x ≠ y ∧ ∃ z : ℕ, x * y = z ^ 2 := by
sorry

end subset_contains_square_product_l222_222225


namespace f2023_plus_f2024_l222_222147

-- Define the function f with given properties
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_periodic : ∀ x : ℝ, f(x + 4) = f(x)
axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)
axiom f_neg3 : f(-3) = -3

-- State the theorem to prove f(2023) + f(2024) = 3
theorem f2023_plus_f2024 : f(2023) + f(2024) = 3 := by
  sorry

end f2023_plus_f2024_l222_222147


namespace greatest_possible_value_of_k_l222_222265

theorem greatest_possible_value_of_k :
  ∃ k : ℝ, 
    (∀ (x: ℝ), x^2 + k * x + 8 = 0 → 
      ∃ α β : ℝ, α ≠ β ∧ α - β = sqrt 73) → 
      k = sqrt 105 := 
sorry

end greatest_possible_value_of_k_l222_222265


namespace least_four_digit_palindrome_divisible_by_5_l222_222803

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : ∃ n, is_palindrome n ∧ is_divisible_by_5 n ∧ is_four_digit n ∧ ∀ m, is_palindrome m ∧ is_divisible_by_5 m ∧ is_four_digit m → n ≤ m :=
by
  -- proof steps will be here
  sorry

end least_four_digit_palindrome_divisible_by_5_l222_222803


namespace part1_part2_part3_l222_222033

variable (a b S T: ℕ → ℝ)

-- Definitions based on conditions
axiom a_eq : ∀ n : ℕ, a n = n + 1
axiom S_eq : ∀ n : ℕ, S n = (n * (n + 3)) / 2
axiom b_eq : ∀ n : ℕ, b n = 2^n * (n - 1) / (n * a n)
axiom T_eq : ∀ (n : ℕ+), T n = (∑ i in Finset.range n, b (i + 1))

-- Goal: proving the correctness of the general formula for a_n, 
-- sum of sequence b_n and the range of λ.
theorem part1 : ∀ n : ℕ, a n = n + 1 :=
by
  intro n
  exact a_eq n

theorem part2 : ∀ (n : ℕ+), T n = (2^(n + 1) / (n + 1)) - 2 :=
by
  intro n
  have : ∀ k, b k = 2^k * (k - 1) / (k * a k) := b_eq
  rw [b_eq]
  sorry -- Summing and simplifying will be done here

theorem part3 : ∃ λ : ℝ, (∀ n : ℕ+, T n + 2 > λ * S n) ∧ λ < 4 / 9 :=
by
  use 4 / 9
  have : ∀ n, S n = (n * (n + 3)) / 2 := S_eq
  intro n
  rw [S_eq, T_eq]
  sorry -- Inequality to be shown here for the correct λ

end part1_part2_part3_l222_222033


namespace sum_of_digits_base8_888_l222_222369

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222369


namespace problem1_problem2_l222_222440

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end problem1_problem2_l222_222440


namespace cone_generators_angle_distance_l222_222783

theorem cone_generators_angle_distance 
  (R H h : ℝ) 
  (hR_pos : R > 0)
  (hH_pos : H > 0)
  (hh_pos : h > 0) 
  (quarter: Real.pi = 2 * angle (0, 1)) : 
  let angle_between_generators := arccos ((h * H) / sqrt ((H^2 + R^2) * (h^2 + R^2))),
      distance_between_generators := R * (H + h) / sqrt (H^2 + R^2)
  in
  angle_between_generators = arccos (h * H / sqrt ((H^2 + R^2) * (h^2 + R^2))) 
  ∧ distance_between_generators = R * (H + h) / sqrt (H^2 + R^2) :=
by
  sorry

end cone_generators_angle_distance_l222_222783


namespace general_term_formula_sum_first_n_terms_l222_222985

open Nat

section
variables {S : ℕ → ℕ} {a : ℕ → ℕ}

-- Define sequence and sum conditions
axiom sum_condition : ∀ n, S n = 2 * a n - a 1
axiom arithmetic_condition : a 1 + a 3 = 2 * (a 2 + 1)

-- Proving the general term formula for the sequence {a_n}
theorem general_term_formula : (∀ n, a n = 2^n) :=
sorry

-- Define the sequence {b_n} as given in problem II and its sum
noncomputable def b : ℕ → ℕ 
| 0     := 1
| 1     := 0
| (n+2) := 2^(n+2) - (n+2) - 2

noncomputable def T : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := 2^(n+3) - ((n+2)^2 + 5 * (n+2)) / 2

-- Proving the sum of the first n terms of {a_n - n - 2}
theorem sum_first_n_terms : (∀ n, T n = 2^(n+1) - (n^2 + 5 * n) / 2) :=
sorry

end

end general_term_formula_sum_first_n_terms_l222_222985


namespace apple_distribution_l222_222680

theorem apple_distribution (total_apples : ℝ)
  (time_anya time_varya time_sveta total_time : ℝ)
  (work_anya work_varya work_sveta : ℝ) :
  total_apples = 10 →
  time_anya = 20 →
  time_varya = 35 →
  time_sveta = 45 →
  total_time = (time_anya + time_varya + time_sveta) →
  work_anya = (total_apples * time_anya / total_time) →
  work_varya = (total_apples * time_varya / total_time) →
  work_sveta = (total_apples * time_sveta / total_time) →
  work_anya = 2 ∧ work_varya = 3.5 ∧ work_sveta = 4.5 := by
  sorry

end apple_distribution_l222_222680


namespace min_ABCD_sum_correct_ABCD_l222_222210

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222210


namespace opposite_of_half_l222_222768

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222768


namespace player_A_min_score_l222_222686

theorem player_A_min_score (A B : ℕ) (hA_first_move : A = 1) (hB_next_move : B = 2) : 
  ∃ k : ℕ, k = 64 :=
by
  sorry

end player_A_min_score_l222_222686


namespace opposite_of_num_l222_222751

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222751


namespace PQ_passes_through_M_l222_222138

-- Define a parallelogram
variables {α : Type*} [normed_group α] [normed_space ℝ α]

structure Parallelogram (A B C D : α) : Prop :=
(parallel_AB_CD : ∃ u v : ℝ, B = A + u • (D - C))
(parallel_AD_BC : ∃ u v : ℝ, D = A + u • (C - B))

-- Midpoint function
def midpoint (P Q : α) : α := (P + Q) / 2

-- Define the key points
variables (A B C D X Y P Q M : α)

-- Condition: ABCD is a parallelogram
axiom parallelogram_ABCD : Parallelogram A B C D

-- Condition: X and Y lie on segments AB and CD
axiom X_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = A + t • (B - A)
axiom Y_on_CD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = C + s • (D - C)

-- Condition: Intersections
axiom P_is_intersection_AY_DX : (∃ t1 t2 : ℝ, P = A + t1 • (Y - A) ∧ P = D + t2 • (X - D))
axiom Q_is_intersection_BY_DX : (∃ u1 u2 : ℝ, Q = B + u1 • (Y - B) ∧ Q = D + u2 • (X - D))

-- Define the midpoint M of diagonal BD
axiom M_is_midpoint_BD : M = midpoint B D

-- Conclusion: The line PQ passes through the fixed point M
theorem PQ_passes_through_M :
  ∃ l : ℝ, Q = P + l • (M - P) :=
sorry

end PQ_passes_through_M_l222_222138


namespace shaded_area_correct_l222_222123

noncomputable def total_shaded_area (r1 r2 r3 : ℝ) : ℝ :=
  let area1 := Real.pi * r1^2
  let area2 := Real.pi * r2^2
  let area3 := Real.pi * r3^2
  area2 - (area1 + area3)

theorem shaded_area_correct :
  total_shaded_area 3 6 1 ≈ 81.7 :=
by
  sorry

end shaded_area_correct_l222_222123


namespace max_pieces_is_seven_l222_222258

-- Define what it means for a number to have all distinct digits
def all_digits_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.nodup)

-- Define the main proof problem
theorem max_pieces_is_seven :
  ∃ (n : ℕ) (PIE : ℕ) (PIECE : ℕ),
  (PIE = PIECE * n) ∧
  (PIE >= 10000) ∧ (PIE < 100000) ∧
  all_digits_distinct PIE ∧
  all_digits_distinct PIECE ∧
  ∀ m, (m > n) → (¬ (∃ P' PIECE', (P' = PIECE' * m) ∧
   (P' >= 10000) ∧ (P' < 100000) ∧ all_digits_distinct P' ∧ all_digits_distinct PIECE'))
:= sorry

end max_pieces_is_seven_l222_222258


namespace sum_of_base8_digits_888_l222_222361

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222361


namespace ellipse_vertices_distance_l222_222000

theorem ellipse_vertices_distance :
  ∀ {a b : ℝ} (h_a : a = 8) (h_b : b = 7),
    (∀ x y : ℝ, (x^2 / 49) + (y^2 / 64) = 1) → 
    2 * a = 16 :=
by
  intros a b h_a h_b h
  rw [h_a]
  norm_num
  sorry

end ellipse_vertices_distance_l222_222000


namespace sum_of_digits_base_8_rep_of_888_l222_222354

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222354


namespace cos_theta_minus_phi_l222_222612

theorem cos_theta_minus_phi (θ φ : ℝ) :
  exp (complex.I * θ) = (4 / 5) + (3 / 5) * complex.I ∧
  exp (complex.I * φ) = -(5 / 13) + (12 / 13) * complex.I →
  real.cos (θ - φ) = - (16 / 65) :=
by
  sorry

end cos_theta_minus_phi_l222_222612


namespace calc_expr_eq_simplify_expr_eq_l222_222442

-- Problem 1: Calculation
theorem calc_expr_eq : 
  ((1 / 2) ^ (-2) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20) = 3 - 2 * Real.sqrt 5 := 
  by
  sorry

-- Problem 2: Simplification
theorem simplify_expr_eq (x : ℝ) (hx : x ≠ 0): 
  ((x^2 - 2 * x + 1) / (x^2 - 1) / (x - 1) / (x^2 + x)) = x := 
  by
  sorry

end calc_expr_eq_simplify_expr_eq_l222_222442


namespace array_rows_equal_columns_l222_222632

theorem array_rows_equal_columns (m n : ℕ) (A : ℕ → ℕ → ℝ)
  (h1 : ∀ i, ∃ j, 0 ≤ A i j)
  (h2 : ∀ j, ∃ i, 0 ≤ A i j)
  (h3 : ∀ i j, A i j > 0 → ∑ k in range m, A k j = ∑ l in range n, A i l) :
  m = n :=
sorry

end array_rows_equal_columns_l222_222632


namespace candy_cost_l222_222460

theorem candy_cost (C : ℝ) 
  (h1 : 20 * C + 80 * 5 = 100 * 6) : 
  C = 10 := 
by
  sorry

end candy_cost_l222_222460


namespace find_other_person_money_l222_222677

noncomputable def other_person_money (mias_money : ℕ) : ℕ :=
  let x := (mias_money - 20) / 2
  x

theorem find_other_person_money (mias_money : ℕ) (h_mias_money : mias_money = 110) : 
  other_person_money mias_money = 45 := by
  sorry

end find_other_person_money_l222_222677


namespace smallest_N_for_given_k_l222_222549

theorem smallest_N_for_given_k (k : ℕ) (hk : k > 0) :
  ∃ N : ℕ, (∀ (s : Finset ℕ), s.card = 2 * k → (s.sum id > N ∧ 
    (∀ t : Finset ℕ, t ⊆ s → t.card = k → t.sum id ≤ N / 2))) ∧ 
  N = 2 * k^3 + 3 * k^2 + 3 * k :=
begin
  sorry,
end

end smallest_N_for_given_k_l222_222549


namespace dice_probability_l222_222464

theorem dice_probability (X₁ X₂ X₃ X₄ : ℕ) (h₁ : X₁ ∈ finset.range 1 7)
  (h₂ : X₂ ∈ finset.range 1 7) (h₃ : X₃ ∈ finset.range 1 7) 
  (h₄ : X₄ ∈ finset.range 1 7) (h_sum : X₁ + X₂ + X₃ = X₄) :
  (∃ n ∈ finset.singleton 2, X₁ = n ∨ X₂ = n ∨ X₃ = n ∨ X₄ = n) →
  (probability_of_event (λ X₁ X₂ X₃ X₄, ∃ n ∈ finset.singleton 2, X₁ = n ∨ X₂ = n ∨ X₃ = n ∨ X₄ = n) 
   (all_valid_outcomes)) = 3 / 5 :=
begin
  sorry
end

end dice_probability_l222_222464


namespace find_250th_term_l222_222320

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

noncomputable def sequence_omitting_squares_and_threes : ℕ → ℕ
| 0       => 0  -- Technically doesn't matter, starting from sequence being empty
| (n + 1) =>
  let next := n + 1
  if is_perfect_square next ∨ next % 3 = 0 then
    sequence_omitting_squares_and_threes n
  else
    next

theorem find_250th_term :
  sequence_omitting_squares_and_threes 250 = 377 :=
sorry

end find_250th_term_l222_222320


namespace florist_break_even_l222_222466

theorem florist_break_even :
  ∃ x : ℕ, 
    -- initial condition: she had 37 roses
    let initial_roses : ℕ := 37 in
    -- she sold 16 roses
    let sold_roses : ℕ := 16 in
    -- final condition after selling and picking more roses: she had 40 roses
    let final_roses : ℕ := 40 in
    -- remaining roses after selling
    let remaining_roses := initial_roses - sold_roses in
    -- final count includes the picked roses
    (remaining_roses + x = final_roses)
    -- conclusion: she picked 19 roses
    ∧ x = 19 :=
by
  sorry

end florist_break_even_l222_222466


namespace albert_number_l222_222880

theorem albert_number :
  ∃ (n : ℕ), (1 / (n : ℝ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) ∧ 
             ∃ m : ℕ, (1 / (m : ℝ) + 1 / 2 = 1 / 3 + 2 / (m + 1)) ∧ m ≠ n :=
sorry

end albert_number_l222_222880


namespace P_cannot_be_reached_l222_222855

-- Define the positions on the board as an enumeration
inductive Position
| S | P | Q | R | T | W

-- Define the movement rules
def move_three (pos : Position) (direction : String) : Option Position :=
  match pos, direction with
  | Position.S, "right" => some Position.W
  | _, _ => none

def move_two (pos : Position) (direction : String) : Option Position :=
  match pos, direction with
  | Position.W, "up" => some Position.T
  | Position.W, "left" => some Position.S
  | _, _ => none

-- Define a way to combine moves
def can_reach (start end : Position) : Prop :=
  ∃ direction1 direction2,
    move_three start direction1 = some Position.W ∧
    move_two Position.W direction2 = some end

-- Prove that P cannot be reached from S
theorem P_cannot_be_reached : ¬ can_reach Position.S Position.P :=
  sorry

end P_cannot_be_reached_l222_222855


namespace prove_inequality_l222_222045

open Real

noncomputable def ellipse_condition (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def perpendicular_bisector_intersects_x_axis (a b x0 : ℝ) : Prop :=
  - (a^2 - b^2) / a < x0 ∧ x0 < (a^2 - b^2) / a

theorem prove_inequality (a b x0 : ℝ) 
  (h1 : ∃ x y, ellipse_condition x y a b) 
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : ∀ x1 y1 x2 y2, 
    ellipse_condition x1 y1 a b → 
    ellipse_condition x2 y2 a b →
    let x' := (x1 + x2) / 2 in
    let y' := (y1 + y2) / 2 in
    let x0 := x' * ((a^2 - b^2) / a^2) in
    - (a^2 - b^2) / a < x0 ∧ x0 < (a^2 - b^2) / a
  ) : perpendicular_bisector_intersects_x_axis a b x0 :=
sorry

end prove_inequality_l222_222045


namespace expected_survivors_l222_222626

theorem expected_survivors :
  let initial_population := 700
  let p1 := 1 / 10
  let p2 := 2 / 10
  let p3 := 3 / 10
  let s1 := 1 - p1
  let s2 := 1 - p2
  let s3 := 1 - p3
  let after_first_month := initial_population * s1
  let after_second_month := after_first_month * s2
  let after_third_month := after_second_month * s3
  (after_third_month ≈ 353) :=
by
  -- Definitions
  let initial_population := 700
  let p1 := 1 / 10
  let p2 := 2 / 10
  let p3 := 3 / 10
  let s1 := 1 - p1
  let s2 := 1 - p2
  let s3 := 1 - p3
  let after_first_month := initial_population * s1
  let after_second_month := after_first_month * s2
  let after_third_month := after_second_month * s3
  -- Statement to prove
  show after_third_month ≈ 353 from sorry

end expected_survivors_l222_222626


namespace rationalize_denominator_and_min_sum_l222_222176

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222176


namespace locus_of_centers_of_tangent_circles_l222_222003

-- Definitions from problem conditions
variables {ℝ : Type*} [metric_space ℝ] [normed_group ℝ]
          {line l : set (ℝ × ℝ)}
          {point M : ℝ × ℝ}

def is_on_line (p : ℝ × ℝ) (l : set (ℝ × ℝ)) : Prop := sorry

def is_perpendicular_to (l1 l2 : set (ℝ × ℝ)) : Prop := sorry

-- Hypotheses based on the conditions
axiom M_on_l (hM : M ∈ l) : is_on_line M l

axiom line_m_perpendicular_to_l (hlm : is_perpendicular_to m l) : 
  m = {p : ℝ × ℝ | ∃ (a : ℝ), a ≠ 0 ∧ p = (M.1, M.2 + a)}

-- Equivalent proof problem
theorem locus_of_centers_of_tangent_circles (center : ℝ × ℝ) :
  (∀ r, center = (M.1, M.2 + r) ∧ r ≠ 0 → 
   ∃ circle, tangent_to_line circle l M ∧ center_of_circle circle = center) :=
begin
  sorry
end

end locus_of_centers_of_tangent_circles_l222_222003


namespace greatest_possible_difference_l222_222427

def is_reverse (q r : ℕ) : Prop :=
  let q_tens := q / 10
  let q_units := q % 10
  let r_tens := r / 10
  let r_units := r % 10
  (q_tens = r_units) ∧ (q_units = r_tens)

theorem greatest_possible_difference (q r : ℕ) (hq1 : q ≥ 10) (hq2 : q < 100)
  (hr1 : r ≥ 10) (hr2 : r < 100) (hrev : is_reverse q r) (hpos_diff : q - r < 30) :
  q - r ≤ 27 :=
by
  sorry

end greatest_possible_difference_l222_222427


namespace proof_problem_l222_222836

-- Define sets
def N_plus : Set ℕ := {x | x > 0}  -- Positive integers
def Z : Set ℤ := {x | true}        -- Integers
def Q : Set ℚ := {x | true}        -- Rational numbers

-- Lean problem statement
theorem proof_problem : 
  (0 ∉ N_plus) ∧ 
  (((-1)^3 : ℤ) ∈ Z) ∧ 
  (π ∉ Q) :=
by
  sorry

end proof_problem_l222_222836


namespace find_a_and_sin_A_l222_222141

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Conditions
axiom sin_A_eq_3sin_B : sin A = 3 * sin B
axiom C_eq_pi_div_3 : C = Real.pi / 3
axiom c_eq_sqrt_7 : c = Real.sqrt 7

-- Statement to prove
theorem find_a_and_sin_A (h1 : sin A = 3 * sin B)
                         (h2 : C = Real.pi / 3)
                         (h3 : c = Real.sqrt 7) : 
                         (a = 3 ∧ sin A = 3 * Real.sqrt 21 / 14) :=
by { sorry }

end find_a_and_sin_A_l222_222141


namespace count_positive_integer_b_for_log_count_valid_b_l222_222600

theorem count_positive_integer_b_for_log (b : ℕ) : 
  ∃ n : ℕ, n > 0 ∧ b^n = 15625 → b ∈ {5, 25, 125, 15625} :=
sorry

theorem count_valid_b : 
  finset.card {b : ℕ | ∃ n : ℕ, n > 0 ∧ b^n = 15625 } = 4 :=
sorry

end count_positive_integer_b_for_log_count_valid_b_l222_222600


namespace volunteer_recommendations_l222_222106

def num_recommendations (boys girls : ℕ) (total_choices chosen : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_choices chosen
  let invalid_combinations := Nat.choose boys chosen
  total_combinations - invalid_combinations

theorem volunteer_recommendations : num_recommendations 4 3 7 4 = 34 := by
  sorry

end volunteer_recommendations_l222_222106


namespace greatest_possible_k_l222_222270

theorem greatest_possible_k (k : ℝ) (h : ∀ x, x^2 + k * x + 8 = 0) (diff_roots : (∀ a b, a ≠ b → a - b = sqrt 73)) : k = sqrt 105 :=
by
  sorry

end greatest_possible_k_l222_222270


namespace sum_of_digits_base_8_888_is_13_l222_222402

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222402


namespace rationalize_denominator_and_min_sum_l222_222180

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222180


namespace sum_of_digits_base8_888_l222_222375

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222375


namespace forty_percent_of_number_l222_222426

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 0.4 * N = 204 :=
sorry

end forty_percent_of_number_l222_222426


namespace president_vp_committee_count_l222_222115

theorem president_vp_committee_count :
  let people := 10
  let choose (n k : ℕ) := Nat.choose n k
  let ways_president := 10
  let ways_vp := 9
  let ways_committee := choose 8 2
  in ways_president * ways_vp * ways_committee = 2520 :=
by
  sorry

end president_vp_committee_count_l222_222115


namespace range_of_a_minus_b_l222_222550

noncomputable def quadratic_function (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

def distinct_zeros (a b : ℝ) : Prop :=
  let Δ := a^2 - 4*b in Δ > 0

theorem range_of_a_minus_b (a b : ℝ) :
  distinct_zeros a b →
  (∃ x1 x2 x3 x4 : ℝ, x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
                      (x2 - x1 = x3 - x2) ∧ (x3 - x2 = x4 - x3) ∧
                      quadratic_function (x1^2 + 2*x1 - 1) a b = 0 ∧
                      quadratic_function (x2^2 + 2*x2 - 1) a b = 0 ∧
                      quadratic_function (x3^2 + 2*x3 - 1) a b = 0 ∧
                      quadratic_function (x4^2 + 2*x4 - 1) a b = 0) →
  a - b ≤ 25 / 9 :=
by
  sorry

end range_of_a_minus_b_l222_222550


namespace sum_of_digits_base8_888_l222_222414

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222414


namespace functions_A_are_same_functions_C_are_same_functions_D_are_same_l222_222888

-- Definition of the functions
def F_A (x : ℝ) : ℝ := 1
def G_A (x : ℝ) : ℝ := (x - 1) ^ 0

def F_C (x : ℝ) : ℝ := Real.sqrt (2 + x) * Real.sqrt (2 - x)
def G_C (x : ℝ) : ℝ := Real.sqrt (4 - x ^ 2)

def F_D (x : ℝ) : ℝ := (x + 1) ^ 2
def G_D (t : ℝ) : ℝ := t ^ 2 + 2 * t + 1

-- Prove that the functions in each pair are the same
theorem functions_A_are_same : ∀ x : ℝ, F_A x = G_A x :=
by sorry

theorem functions_C_are_same : ∀ x : ℝ, F_C x = G_C x :=
by sorry

theorem functions_D_are_same : ∀ x : ℝ, F_D x = G_D x :=
by sorry

end functions_A_are_same_functions_C_are_same_functions_D_are_same_l222_222888


namespace correct_propositions_l222_222887

-- Define the functions in question
noncomputable def f1 (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)
noncomputable def f2 (x : ℝ) : ℝ := Real.cos (x + π / 3)
noncomputable def f3 (x : ℝ) : ℝ := Real.tan (x + π / 3)
noncomputable def f4 (x : ℝ) : ℝ := 3 * Real.sin (2 * x + π / 3)
noncomputable def f4_translated (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

-- Define the propositions
def prop1 := ∀ x ∈ Ioo (-π/3) (π/6), f1 x > f1 (x - 10^(-10))
def prop2 := ∀ x, f2 (2 * (π/6) - x) = f2 x
def prop3 := ∀ x, f3 (2 * (π/6) - x) = f3 x
def prop4 := ∀ x, f4 (x + π/6) = f4_translated x

-- Lean statement asserting the correct propositions
theorem correct_propositions : prop2 ∧ prop4 := by {
  sorry -- Proof not needed as stated in instructions
}

end correct_propositions_l222_222887


namespace num_valid_values_m_l222_222162

def num_values_m : ℕ :=
  let n := 540
  let divisors := n.divisors
  let valid_divisors := divisors.filter (λ d => d > 1 ∧ d < n)
  valid_divisors.length

theorem num_valid_values_m : num_values_m = 22 := 
by
  sorry

end num_valid_values_m_l222_222162


namespace sum_of_base8_digits_888_l222_222358

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222358


namespace units_digit_of_factorial_sum_l222_222415

theorem units_digit_of_factorial_sum : 
  (∑ i in finset.range 1000, (nat.factorial i % 10)) % 10 = 3 := 
by
  -- As Lean code, we assert the given sum and conditions
  sorry

end units_digit_of_factorial_sum_l222_222415


namespace investment_total_correct_l222_222894

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end investment_total_correct_l222_222894


namespace solve_for_x_l222_222451

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l222_222451


namespace find_area_triangle_OPQ_l222_222502

noncomputable def area_triangle_OPQ (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : (a + 2) * (b + 3) = 6) : ℝ :=
  let P := (b, 2 / b) in
  let Q := (2 / a, a) in
  let O := (0 : ℝ, 0 : ℝ) in
  1/2 * real.abs (O.1 * (P.2 - Q.2) + P.1 * (Q.2 - O.2) + Q.1 * (O.2 - P.2))

theorem find_area_triangle_OPQ (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : (a + 2) * (b + 3) = 6) : 
  area_triangle_OPQ a b ha hb h = 8 / 3 :=
sorry

end find_area_triangle_OPQ_l222_222502


namespace prime_sums_count_l222_222228

/--
Given a sequence where the first term is the prime number 3, and each successive term is the sum of the previous term and the next prime number, prove that out of the first fifteen terms of this sequence, exactly three of them are prime.
-/
theorem prime_sums_count : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
  let sums := list.scanl (+) 0 primes
  let sum_first_15 := sums.take 15
  let prime_sums := sum_first_15.filter (λ n, nat.prime n)
  prime_sums.length = 3 :=
by
  sorry

end prime_sums_count_l222_222228


namespace number_of_subsets_of_set_l222_222610

theorem number_of_subsets_of_set (x y : ℝ) 
  (z : ℂ) (hz : z = (2 - (1 : ℂ) * Complex.I) / (1 + (2 : ℂ) * Complex.I))
  (hx : z.re = x) (hy : z.im = y) : 
  (Finset.powerset ({x, 2^x, y} : Finset ℝ)).card = 8 :=
by
  sorry

end number_of_subsets_of_set_l222_222610


namespace sum_of_digits_base8_888_l222_222382

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222382


namespace ab_cd_value_l222_222086

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l222_222086


namespace find_four_digit_number_l222_222845

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l222_222845


namespace find_radius_l222_222984

theorem find_radius (r : ℝ) (h : ∃ A B : ℝ × ℝ, 
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 36 ∧ 
  A.1 ^ 2 + A.2 ^ 2 = r ^ 2 ∧ B.1 ^ 2 + B.2 ^ 2 = r ^ 2 ∧ 
  (1 * A.1 - sqrt 3 * A.2 + 8 = 0) ∧ 
  (1 * B.1 - sqrt 3 * B.2 + 8 = 0)) :
  r = 5 :=
sorry

end find_radius_l222_222984


namespace original_number_is_57_l222_222470

theorem original_number_is_57 :
  ∃ n : ℕ, (digits n).prod * n = 1995 ∧ n = 57 := 
by {
  use 57,
  split,
  sorry, -- Proof that (digits 57).prod * 57 = 1995 goes here.
  refl
}

end original_number_is_57_l222_222470


namespace num_subsets_P_l222_222038

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def P : Set ℕ := M ∩ N

theorem num_subsets_P : (P.cardinality = 4) :=
  by sorry

end num_subsets_P_l222_222038


namespace unit_digit_of_expression_is_4_l222_222286

theorem unit_digit_of_expression_is_4 :
  Nat.unitsDigit ((2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) - 1) = 4 :=
by
  sorry

end unit_digit_of_expression_is_4_l222_222286


namespace range_of_a_l222_222050

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 ∧ sin^2 x - 2 * sin x - a = 0) ↔ a ∈ set.Icc (-1) 3 :=
by {
  sorry
}

end range_of_a_l222_222050


namespace min_ABCD_sum_correct_ABCD_l222_222209

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222209


namespace sum_of_digits_base8_888_l222_222411

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222411


namespace probability_of_hitting_target_at_least_once_l222_222110

-- Define the constant probability of hitting the target in a single shot
def p_hit : ℚ := 2 / 3

-- Define the probability of missing the target in a single shot
def p_miss := 1 - p_hit

-- Define the probability of missing the target in all 3 shots
def p_miss_all_3 := p_miss ^ 3

-- Define the probability of hitting the target at least once in 3 shots
def p_hit_at_least_once := 1 - p_miss_all_3

-- Provide the theorem stating the solution
theorem probability_of_hitting_target_at_least_once :
  p_hit_at_least_once = 26 / 27 :=
by
  -- sorry is used to indicate the theorem needs to be proved
  sorry

end probability_of_hitting_target_at_least_once_l222_222110


namespace toothpicks_in_12th_stage_l222_222786

def toothpicks_in_stage (n : ℕ) : ℕ :=
  3 * n

theorem toothpicks_in_12th_stage : toothpicks_in_stage 12 = 36 :=
by
  -- Proof steps would go here, including simplification and calculations, but are omitted with 'sorry'.
  sorry

end toothpicks_in_12th_stage_l222_222786


namespace mary_screws_sections_l222_222675

def number_of_sections (initial_screws : Nat) (multiplier : Nat) (screws_per_section : Nat) : Nat :=
  let additional_screws := initial_screws * multiplier
  let total_screws := initial_screws + additional_screws
  total_screws / screws_per_section

theorem mary_screws_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end mary_screws_sections_l222_222675


namespace ab_plus_cd_eq_neg_346_over_9_l222_222087

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end ab_plus_cd_eq_neg_346_over_9_l222_222087


namespace radius_of_inscribed_circle_in_symmetric_trapezoid_l222_222455

theorem radius_of_inscribed_circle_in_symmetric_trapezoid :
  ∃ (r : ℝ), 
    (∃ (R : ℝ) (O K : ℝ), R = 1 ∧ R / 2 = O - K ∧ 
    ∃ (a b : ℝ), a ≥ b ∧ 
    r = sqrt (a * b) ∧ r = sqrt (9 / 40)) :=
sorry

end radius_of_inscribed_circle_in_symmetric_trapezoid_l222_222455


namespace sum_frac_parts_eq_1007_l222_222976

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := int.floor x

-- Define the fractional part function
def frac_part (x : ℝ) : ℝ := x - ↑(floor x)

-- Prove that the sum of the fractional parts is 1007
theorem sum_frac_parts_eq_1007 :
  (\sum k in finset.range 2015, frac_part (k.succ / 2015)) = 1007 := 
by
  sorry

end sum_frac_parts_eq_1007_l222_222976


namespace largest_possible_number_l222_222651

theorem largest_possible_number 
    (total_people : ℕ) (not_working : ℕ) (have_families : ℕ) (like_sing_shower : ℕ)
    (h1 : total_people = 100)
    (h2 : not_working = 50)
    (h3 : have_families = 25)
    (h4 : like_sing_shower = 75) :
    let working := total_people - not_working
    let without_families := total_people - have_families
    in min (min working without_families) like_sing_shower = 50 :=
by
  sorry

end largest_possible_number_l222_222651


namespace probability_same_color_is_one_third_l222_222314

-- Define a type for colors
inductive Color 
| red 
| white 
| blue 

open Color

-- Define the function to calculate the probability of the same color selection
def sameColorProbability : ℚ :=
  let total_outcomes := 3 * 3
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

-- Theorem stating that the probability is 1/3
theorem probability_same_color_is_one_third : sameColorProbability = 1 / 3 :=
by
  -- Steps of proof will be provided here
  sorry

end probability_same_color_is_one_third_l222_222314


namespace alpha_parallel_beta_l222_222567

-- Definitions for lines and planes
constant Line : Type
constant Plane : Type
constant α β : Plane
constant m n l1 l2 : Line

-- Conditions
constant m_in_α : m ∈ α
constant n_in_α : n ∈ α
constant l1_in_β : l1 ∈ β
constant l2_in_β : l2 ∈ β
constant l1_l2_intersect : intersecting l1 l2

-- Parallel predicate
constant parallel : Line → Line → Prop
constant parallel_plane : Plane → Plane → Prop

-- Theorem
theorem alpha_parallel_beta (h1 : parallel m l1) (h2 : parallel n l2) : parallel_plane α β :=
sorry

end alpha_parallel_beta_l222_222567


namespace sum_of_digits_of_8_pow_2004_l222_222813

-- Define the problem statement
theorem sum_of_digits_of_8_pow_2004 :
  let n := 8 ^ 2004 in 
  (n % 100) / 10 + ((n % 100) % 10) = 7 :=
by 
  sorry

end sum_of_digits_of_8_pow_2004_l222_222813


namespace function_properties_l222_222728

def f (x : ℝ) : ℝ := 2 * (sin (x - π / 4))^2 - 1

theorem function_properties : 
  (∀ x : ℝ, f (-x) = -f(x)) ∧ (∀ x : ℝ, f (x + π) = f(x)) :=
by
  sorry

end function_properties_l222_222728


namespace people_left_gym_l222_222900

theorem people_left_gym (initial : ℕ) (additional : ℕ) (current : ℕ) (H1 : initial = 16) (H2 : additional = 5) (H3 : current = 19) : (initial + additional - current) = 2 :=
by
  sorry

end people_left_gym_l222_222900


namespace petya_wins_optimal_l222_222168

-- Definitions and conditions based on the given problem
def n : ℕ := 8 -- Example size of the grid, can be generalized later
def initial_board := Array.init n (λ _ => Array.init n (λ _ => false)) -- All cells are initially white
def initial_board_with_rook := initial_board.set! 0 (initial_board.get! 0).set! 0 true  -- Top-left corner is black

-- A function to determine if a cell is within the grid bounds and white
def is_white (board : Array (Array Bool)) (i j : ℕ) : Prop :=
  i < n ∧ j < n ∧ ¬board.get! i |>.get! j

-- Function to move the rook and mark cells black
def move_rook (board : Array (Array Bool)) (start end_ : (ℕ × ℕ)) : Array (Array Bool) :=
  -- Details of move_rook function implementation would be here
  board -- Placeholder, actual implementation needed

-- Game conditions and question as a theorem
theorem petya_wins_optimal :
  ∀ (n : ℕ) (initial_board : Array (Array Bool)),
  (initial_board.get! 0 |>.get! 0 = true) →  -- Initial board has rook in top-left corner (black)
  (∀ (move : (ℕ × ℕ) → (ℕ × ℕ)),  -- For any valid move
    board' = move_rook initial_board move.fst move.snd →
    ∀ (board' : Array (Array Bool)),
      ---- Proof steps should show Petya's strategy ----
      sorry
    ) →
  (true) -- This represents that Petya wins as per the analysis

end petya_wins_optimal_l222_222168


namespace probability_fav_song_not_played_completely_l222_222849

theorem probability_fav_song_not_played_completely :
  (∃ songs : List ℕ, songs.length = 12 ∧
     (∀ (i : ℕ), 0 ≤ i ∧ i < 12 → (songs.get i = 40 + i * 40)) ∧
     (∃ fav_song_index : ℕ, fav_song_index = 6 ∧ (songs.get fav_song_index = 280)) ∧
     (let total_duration := 12 * 40 + (12 * 11 / 2) * 40 in
      total_duration = 5 * 60 + 40) ∧
  ∃ k : ℚ, k = 5 / 6 := 
by 
sorry

end probability_fav_song_not_played_completely_l222_222849


namespace sequence_properties_l222_222647

theorem sequence_properties (a : ℕ → ℤ) (h1 : a 1 + a 2 = 5)
  (h2 : ∀ n, n % 2 = 1 → a (n + 1) - a n = 1)
  (h3 : ∀ n, n % 2 = 0 → a (n + 1) - a n = 3) :
  (a 1 = 2 ∧ a 2 = 3) ∧
  (∀ n, ∃ d, ∀ k, a (2 * k + 1) = a 1 + k * d) ∧
  (¬ ∃ r, ∀ k, a (2 * k) = a 2 * (r ^ k)) ∧
  (∀ n, (n % 2 = 1 → a n = 2 * n) ∧ (n % 2 = 0 → a n = 2 * n - 1)) := 
sorry

end sequence_properties_l222_222647


namespace min_area_quadrilateral_PACB_l222_222149

def circle (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
def line (x y : ℝ) := 3*x + 4*y + 3 = 0
def center_and_radius : ℝ × ℝ × ℝ := (1, 1, 1)  -- center (1, 1) and radius 1

theorem min_area_quadrilateral_PACB : 
  ∀ P : ℝ × ℝ, line P.1 P.2 → (circle P.1 P.2 = false) → 
  (let r := sqrt(3) in
   ∀ A B : ℝ × ℝ, (A ≠ B) → 
   (circle A.1 A.2) ∧ (circle B.1 B.2) →
   (3*P.1 + 4*P.2 + 3 = 0) →
   (P.1, P.2) ∈ line → 
   (let d := 2 in
   (∃ C D : ℝ × ℝ, |P.1 - C.1| * |P.2 - C.2| = r →
                        |P.1 - D.1| * |P.2 - D.2| = r) →
   let area := r * r in
   area = sqrt(3))) :=
sorry

end min_area_quadrilateral_PACB_l222_222149


namespace complex_modulus_l222_222233

theorem complex_modulus (z : ℂ) (h : (z - complex.I) * complex.I = 1 + complex.I) : complex.abs z = 1 :=
sorry

end complex_modulus_l222_222233


namespace ab_cd_value_l222_222085

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l222_222085


namespace percentage_of_non_swan_geese_l222_222663

theorem percentage_of_non_swan_geese (n : ℕ) (h : n > 0) :
  let geese := 0.4 * n,
      swans := 0.2 * n,
      non_swans := n - swans
  in (geese / non_swans) = 0.5 :=
by
  sorry

end percentage_of_non_swan_geese_l222_222663


namespace max_distance_with_optimal_swapping_l222_222953

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end max_distance_with_optimal_swapping_l222_222953


namespace fraction_of_total_cost_for_raisins_l222_222909

-- Define variables and constants
variable (R : ℝ) -- cost of a pound of raisins

-- Define the conditions as assumptions
variable (cost_of_nuts : ℝ := 4 * R)
variable (cost_of_dried_berries : ℝ := 2 * R)

variable (total_cost : ℝ := 3 * R + 4 * cost_of_nuts + 2 * cost_of_dried_berries)
variable (cost_of_raisins : ℝ := 3 * R)

-- Main statement that we want to prove
theorem fraction_of_total_cost_for_raisins :
  cost_of_raisins / total_cost = 3 / 23 := by
  sorry

end fraction_of_total_cost_for_raisins_l222_222909


namespace opposite_neg_one_half_l222_222762

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222762


namespace no_tangent_to_x_axis_max_integer_a_for_inequality_l222_222056

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (a / 2) * x^2

theorem no_tangent_to_x_axis (a : ℝ) : ¬∃ t : ℝ, f t a = 0 ∧ (t * Real.exp t - a * t) = 0 := sorry

theorem max_integer_a_for_inequality : 
  (∃ a : ℤ, (∀ x1 x2 : ℝ, x2 > 0 → f (x1 + x2) a - f (x1 - x2) a > -2 * x2) ∧ 
             (∀ b : ℤ, b > a → ∃ x1 x2 : ℝ, x2 > 0 ∧ f (x1 + x2) b - f (x1 - x2) b ≤ -2 * x2)) ∧ a = ↑3 := sorry

end no_tangent_to_x_axis_max_integer_a_for_inequality_l222_222056


namespace proof_two_digit_number_l222_222484

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l222_222484


namespace contrapositive_even_sum_l222_222234

theorem contrapositive_even_sum (a b : ℕ) :
  (¬(a % 2 = 0 ∧ b % 2 = 0) → ¬(a + b) % 2 = 0) ↔ (¬((a + b) % 2 = 0) → ¬(a % 2 = 0 ∧ b % 2 = 0)) :=
by
  sorry

end contrapositive_even_sum_l222_222234


namespace digits_in_number_l222_222302

def four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def contains_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  (n / 1000 = d1 ∨ n / 100 % 10 = d1 ∨ n / 10 % 10 = d1 ∨ n % 10 = d1) ∧
  (n / 1000 = d2 ∨ n / 100 % 10 = d2 ∨ n / 10 % 10 = d2 ∨ n % 10 = d2) ∧
  (n / 1000 = d3 ∨ n / 100 % 10 = d3 ∨ n / 10 % 10 = d3 ∨ n % 10 = d3)

def exactly_two_statements_true (s1 s2 s3 : Prop) : Prop :=
  (s1 ∧ s2 ∧ ¬s3) ∨ (s1 ∧ ¬s2 ∧ s3) ∨ (¬s1 ∧ s2 ∧ s3)

theorem digits_in_number (n : ℕ) 
  (h1 : four_digit_number n)
  (h2 : contains_digits n 1 4 5 ∨ contains_digits n 1 5 9 ∨ contains_digits n 7 8 9)
  (h3 : exactly_two_statements_true (contains_digits n 1 4 5) (contains_digits n 1 5 9) (contains_digits n 7 8 9)) :
  contains_digits n 1 4 5 ∧ contains_digits n 1 5 9 :=
sorry

end digits_in_number_l222_222302


namespace sum_of_base_8_digits_888_l222_222335

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222335


namespace acute_angle_cosine_l222_222606

theorem acute_angle_cosine (A : ℝ) (h1 : cos A = 1 / 2) (h2 : 0 < A ∧ A < π / 2) : A = π / 3 := 
sorry

end acute_angle_cosine_l222_222606


namespace fraction_auto_installment_credit_extended_by_finance_companies_l222_222899

def total_consumer_installment_credit : ℝ := 291.6666666666667
def auto_instalment_percentage : ℝ := 0.36
def auto_finance_companies_credit_extended : ℝ := 35

theorem fraction_auto_installment_credit_extended_by_finance_companies :
  auto_finance_companies_credit_extended / (auto_instalment_percentage * total_consumer_installment_credit) = 1 / 3 :=
by
  sorry

end fraction_auto_installment_credit_extended_by_finance_companies_l222_222899


namespace sum_of_digits_base8_888_l222_222325

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222325


namespace x_squared_plus_y_squared_l222_222603

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^3 = 8) (h2 : x * y = 5) : 
  x^2 + y^2 = -6 := by
  sorry

end x_squared_plus_y_squared_l222_222603


namespace composition_of_homotheties_l222_222174

-- Define points A1 and A2 and the coefficients k1 and k2
variables (A1 A2 : ℂ) (k1 k2 : ℂ)

-- Definition of homothety
def homothety (A : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - A) + A

-- Translation vector in case 1
noncomputable def translation_vector (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 = 1 then (1 - k1) * A1 + (k1 - 1) * A2 else 0 

-- Center A in case 2
noncomputable def center (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 ≠ 1 then (k2 * (1 - k1) * A1 + (1 - k2) * A2) / (k1 * k2 - 1) else 0

-- The final composition of two homotheties
noncomputable def composition (A1 A2 : ℂ) (k1 k2 : ℂ) (z : ℂ) : ℂ :=
  if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
  else homothety (center A1 A2 k1 k2) (k1 * k2) z

-- The theorem to prove
theorem composition_of_homotheties 
  (A1 A2 : ℂ) (k1 k2 : ℂ) : ∀ z : ℂ,
  composition A1 A2 k1 k2 z = if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
                              else homothety (center A1 A2 k1 k2) (k1 * k2) z := 
by sorry

end composition_of_homotheties_l222_222174


namespace pyramid_volume_approx_l222_222866

noncomputable def side_square (surface_area_pyramid : ℝ) : ℝ :=
  (surface_area_pyramid * 3 / 7) ^ (1/2)

noncomputable def area_triangle (area_square : ℝ) : ℝ :=
  area_square / 3

noncomputable def height_pyramid (side_length : ℝ) (area_triangle : ℝ) : ℝ :=
  (area_triangle * 2) / side_length

noncomputable def volume_pyramid (area_square : ℝ) (height : ℝ) : ℝ :=
  (area_square * height) / 3

theorem pyramid_volume_approx :
  ∀ (surface_area_pyramid : ℝ),
  surface_area_pyramid = 648 →
  volume_pyramid (side_square surface_area_pyramid ^ 2) (height_pyramid (side_square surface_area_pyramid) (area_triangle (side_square surface_area_pyramid ^ 2))) ≈ 940.296 :=
by
  assume (surface_area_pyramid : ℝ)
  assume h1 : surface_area_pyramid = 648
  sorry

end pyramid_volume_approx_l222_222866


namespace assertion_1_assertion_2_assertion_3_l222_222519

-- Assertion 1
theorem assertion_1 (P Q : ℝ × ℝ) (l : set (ℝ × ℝ)) (hl : ∃ a b : ℝ, l = {p | p.snd = a * p.fst + b})
  (hP : P.snd > 0) (hQ : Q.snd > 0) (hPQ : P ≠ Q) : ¬ (∃ C1 C2 : set (ℝ × ℝ), 
  C1 ≠ C2 ∧ (∀ P, P ∈ C1 ∧ P ∈ l) ∧ (∀ Q, Q ∈ C2 ∧ Q ∈ l)) :=
sorry

-- Assertion 2
theorem assertion_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b = 1) : 
  ¬ (log a b + log b a ≥ 2) :=
sorry

-- Assertion 3
theorem assertion_3 (A B : set (ℝ × ℝ)) (h : ∀ r : ℝ, r ≥ 0 → ∀ (x y : ℝ), x^2 + y^2 ≤ r^2 ∨ (x, y) ∈ A → (x^2 + y^2 ≤ r^2 ∨ (x, y) ∈ B)) :
  ¬ (A ⊆ B) :=
sorry

end assertion_1_assertion_2_assertion_3_l222_222519


namespace find_a_values_l222_222949
open Real

def system_of_equations (a x y : ℝ) : Prop :=
  (x - 5) * sin a - (y - 5) * cos a = 0 ∧
  ((x + 1)^2 + (y + 1)^2 - 4) * ((x + 1)^2 + (y + 1)^2 - 16) = 0

theorem find_a_values 
  (a : ℝ) (ha : system_of_equations a x y)
  : a = π / 4 + arcsin (sqrt 2 / 6) + π * n ∨ 
    a = π / 4 - arcsin (sqrt 2 / 6) + π * n 
  where n : ℤ :=
sorry

end find_a_values_l222_222949


namespace maximum_pieces_l222_222249

theorem maximum_pieces :
  ∀ (ПИРОГ КУСОК : ℕ) (h1 : ПИРОГ = 95207) (h2 : КУСОК = 13601),
    (ПИРОГ = КУСОК * 7) ∧ (ПИРОГ < 100000) ∧ (ПИРОГ.to_digits.nodup) → 
    7 = 7 :=
by { sorry }

end maximum_pieces_l222_222249


namespace ratio_of_cards_l222_222734

noncomputable def ratio_ellis_orion (total_cards : ℕ) (difference : ℕ) :=
  let x := (total_cards - difference) / 2
  in  (x + difference) / x

theorem ratio_of_cards (total_cards : ℕ) (difference : ℕ) 
    (h1 : total_cards = 500) 
    (h2 : difference = 50) : 
    ratio_ellis_orion total_cards difference = 11 / 9 :=
  by
  sorry

end ratio_of_cards_l222_222734


namespace opposite_of_neg_half_is_half_l222_222741

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222741


namespace ratio_of_a_to_b_l222_222261

theorem ratio_of_a_to_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
    (h_x : x = 1.25 * a) (h_m : m = 0.40 * b) (h_ratio : m / x = 0.4) 
    : (a / b) = 4 / 5 := by
  sorry

end ratio_of_a_to_b_l222_222261


namespace continuity_at_x0_l222_222435

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end continuity_at_x0_l222_222435


namespace minimum_bus_door_height_l222_222243

-- Definitions based on the problem conditions
def normal_distribution_height : Real → Real → Real → Real := sorry  -- Placeholder for the PDF of the normal distribution

def mu : Real := 170  -- mean
def sigma : Real := 7  -- standard deviation

-- Given probabilities
axiom prob_mu_minus_sigma_to_mu_plus_sigma : 0.6826 = 
  (normal_distribution_height mu sigma (mu - sigma)) -
  (normal_distribution_height mu sigma (mu + sigma))

axiom prob_mu_minus_2sigma_to_mu_plus_2sigma : 0.9544 = 
  (normal_distribution_height mu sigma (mu - 2 * sigma)) -
  (normal_distribution_height mu sigma (mu + 2 * sigma))

axiom prob_mu_minus_3sigma_to_mu_plus_3sigma : 0.9974 =
  (normal_distribution_height mu sigma (mu - 3 * sigma)) -
  (normal_distribution_height mu sigma (mu + 3 * sigma))

-- Prove the required height for the bus doors
theorem minimum_bus_door_height : ∃ h : Real, h = 184 ∧ 
  ((1 - (normal_distribution_height mu sigma h)) ≤ 0.0228) :=
by
  sorry

end minimum_bus_door_height_l222_222243


namespace unit_digit_of_expression_l222_222288

theorem unit_digit_of_expression :
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  (expr - 1) % 10 = 4 :=
by
  let expr := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)
  sorry

end unit_digit_of_expression_l222_222288


namespace find_a8_l222_222015

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8_l222_222015


namespace sum_of_digits_base8_888_l222_222370

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222370


namespace total_marbles_l222_222601

theorem total_marbles (num_green : ℕ) (r b g y total : ℕ) 
  (h_ratio : r = 1) (h_ratio_b : b = 3) (h_ratio_g : g = 2) (h_ratio_y : y = 4)
  (h_sum : r + b + g + y = 10)
  (h_green : g → num_green = 24):
  total = 120 := 
by
  sorry

end total_marbles_l222_222601


namespace circle_tangency_problem_l222_222910

theorem circle_tangency_problem
  (r_C : ℝ) (r_D : ℝ)
  (h1 : r_C = 36)
  (h2 : 1 ≤ r_D ∧ r_D < 36)
  (h3 : ∃ n : ℕ, (36 / r_D).denom = 1 ∧ 36 / r_D = n ) :
  ∃ (s : ℕ), s.card = 8 := sorry

end circle_tangency_problem_l222_222910


namespace sum_of_digits_base8_l222_222391

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222391


namespace tetrahedron_surface_area_l222_222098

theorem tetrahedron_surface_area (V : ℝ) (hV : V = 9) : 
  ∃ S : ℝ, S = 18 * Real.sqrt 3 :=
by
  use (18 * Real.sqrt 3)
  sorry

end tetrahedron_surface_area_l222_222098


namespace problem_statement_l222_222964

theorem problem_statement {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) ∧ 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) < 2 :=
by
  sorry

end problem_statement_l222_222964


namespace correct_operation_is_multiplication_by_3_l222_222471

theorem correct_operation_is_multiplication_by_3
  (x : ℝ)
  (percentage_error : ℝ)
  (correct_result : ℝ := 3 * x)
  (incorrect_result : ℝ := x / 5)
  (error_percentage : ℝ := (correct_result - incorrect_result) / correct_result * 100) :
  percentage_error = 93.33333333333333 → correct_result / x = 3 :=
by
  intro h
  sorry

end correct_operation_is_multiplication_by_3_l222_222471


namespace max_pieces_l222_222246

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.nodup digits

def five_digits (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem max_pieces :
  ∀ (n : ℕ) (КУСОК ПИРОГ : ℕ), 
    ПИРОГ = КУСОК * n → 
    five_digits ПИРОГ → 
    distinct_digits ПИРОГ → 
    n ≤ 7 :=
begin
  intros n КУСОК ПИРОГ h1 h2 h3,
  -- skip the proof
  sorry
end

end max_pieces_l222_222246


namespace rationalize_denominator_min_value_l222_222188

theorem rationalize_denominator_min_value :
  ∃ (A B C D : ℤ), D > 0 ∧ ∃ k : ℕ, prime_pos k ∧ ¬ ∃ l : ℕ, l^2 ∣ B ∧ 
  (5 * sqrt 2) / (5 - sqrt 5) = (A * sqrt B + C) / D ∧ 
  A + B + C + D = 12 := sorry

end rationalize_denominator_min_value_l222_222188


namespace number_of_oranges_l222_222695

theorem number_of_oranges :
  ∃ o m : ℕ, (m + 6 * m + o = 20) ∧ (6 * m > o) ∧ (2 ≤ m) ∧ (m ≤ 2) ∧ (o = 6) :=
begin
  -- instantiating variables m and o
  use [2, 6],
  -- prove and this would skip the proof,
  split, linarith, 
  split, linarith,
  split, linarith,
  linarith,
end

end number_of_oranges_l222_222695


namespace days_to_finish_together_l222_222486

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l222_222486


namespace opposite_neg_one_half_l222_222760

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222760


namespace projection_correct_l222_222558

noncomputable def vec_proj (u v : Vect ℝ 3) : Vect ℝ 3 :=
  let v_dot_v := v.dot v in
  if v_dot_v = 0 then 0 else (u.dot v / v_dot_v) • v

theorem projection_correct :
  let w := (2:ℝ) • ⟨[2, 4, -1]⟩ 
  let proj₁ := ⟨[1, -2, 3]⟩
  let proj₂ := ⟨[4, -1, 2]⟩
  vec_proj proj₁ w = ⟨[-2, -4, 1]⟩ →
  vec_proj proj₂ w = ⟨[4/21, 8/21, -1/21]⟩ :=
by
  intros w proj₁ proj₂ h
  sorry

end projection_correct_l222_222558


namespace median_length_of_right_triangle_l222_222118

def is_right_triangle (Δ : Triangle) : Prop :=
  Δ.angle DEF = π / 2

def midpoint (N : Point, E F : Point) : Prop :=
  distance N E = distance N F / 2

def length_of (A B : Point) : ℝ :=
  distance A B

def hypotenuse_length (Δ : Triangle) [is_right_triangle Δ] (DE DF : ℝ) : ℝ :=
  Real.sqrt (DE^2 + DF^2)

theorem median_length_of_right_triangle
  (D E F N : Point)
  (h_right : is_right_triangle ⟨D, E, F⟩)
  (h_midpoint : midpoint N E F)
  (h_length_DE : length_of D E = 5)
  (h_length_DF : length_of D F = 12) :
  length_of D N = 6.5 :=
sorry

end median_length_of_right_triangle_l222_222118


namespace least_four_digit_palindrome_divisible_by_5_l222_222802

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : ∃ n, is_palindrome n ∧ is_divisible_by_5 n ∧ is_four_digit n ∧ ∀ m, is_palindrome m ∧ is_divisible_by_5 m ∧ is_four_digit m → n ≤ m :=
by
  -- proof steps will be here
  sorry

end least_four_digit_palindrome_divisible_by_5_l222_222802


namespace car_average_speed_l222_222830

def average_speed (d1 d2 t1 t2 : ℝ) := (d1 + d2) / (t1 + t2)

theorem car_average_speed : average_speed 50 60 1 1 = 55 := 
begin
  sorry
end

end car_average_speed_l222_222830


namespace seating_arrangement_l222_222114

theorem seating_arrangement :
  let A := 1
  let B := 2
  let C := 3
  let D := 4
  let E := 5
  let F := 6 in
  let people := [A, B, C, D, E, F] in
  let ab_unit := 2 * 5! / 5 in
  ab_unit = 48 :=
by
  sorry

end seating_arrangement_l222_222114


namespace least_four_digit_palindrome_divisible_by_5_l222_222804

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem least_four_digit_palindrome_divisible_by_5 : ∃ n, is_palindrome n ∧ is_divisible_by_5 n ∧ is_four_digit n ∧ ∀ m, is_palindrome m ∧ is_divisible_by_5 m ∧ is_four_digit m → n ≤ m :=
by
  -- proof steps will be here
  sorry

end least_four_digit_palindrome_divisible_by_5_l222_222804


namespace problem_1_problem_2_problem_3_l222_222062

def f (a b x : ℝ) := a * log x / log 2 + b * log x / log 3 + 2

theorem problem_1 (a b : ℝ) : f a b 1 = 2 :=
by
sorry

theorem problem_2 (a b x : ℝ) : f a b x + f a b (1 / x) = 4 :=
by
sorry

theorem problem_3 (a b : ℝ) : 
  (f a b 1 + ∑ i in (Finset.range 2013).filter (≠ 0), f a b i + f a b (1 / i)) = 8050 :=
by
sorry

end problem_1_problem_2_problem_3_l222_222062


namespace new_average_weight_l222_222293

theorem new_average_weight (w_avg : ℝ) (n : ℕ) (w1 w2 : ℝ) (new_avg : ℝ) :
  n = 7 → w_avg = 76 → w1 = 110 → w2 = 60 →
  new_avg = (n * w_avg + w1 + w2) / (n + 2) → new_avg = 78 :=
by
  intros hn hw_avg hw1 hw2 hnew_avg
  simp [hn, hw_avg, hw1, hw2] at hnew_avg
  exact hnew_avg

end new_average_weight_l222_222293


namespace constructed_triangles_cover_base_l222_222031

-- Define the Pyramid and the conditions
structure Pyramid (n : ℕ) :=
(base : Fin n → Point)
(apex : Point)
(congruent_to_base : ∀ i : Fin n, Triangle (apex, base i, base (i + 1) % n) ≃ Triangle (base i, constructed_vertex i, base (i + 1) % n))
(same_side : ∀ i : Fin n, OnSameSide (line (base i, base (i + 1) % n)) (base (i + 1) % n) (constructed_vertex i))

-- Prove that the constructed triangles cover the entire base
theorem constructed_triangles_cover_base (pyramid : Pyramid n) :
  ∀ p : Point, p ∈ convex_hull (set.range (pyramid.base)) → ∃ i : Fin n, p ∈ triangle (pyramid.constructed_vertex i, pyramid.base i, pyramid.base (i + 1) % n) :=
sorry

end constructed_triangles_cover_base_l222_222031


namespace max_sides_convex_polygon_with_four_right_angles_l222_222847

theorem max_sides_convex_polygon_with_four_right_angles :
  ∀ (n : ℕ), (n ≥ 3) →
  (∑ i in (finset.range n).filter (λ k, k ≠ 3 ∨ k ≠ 2 ∨ k ≠ 1), (90 : ℝ)) = 360 →
  (∀ i < n - 4, (a₁ + a₂ + ... + aₙ₋₄) < 90) →
  ∃ (n_max : ℕ), n_max = 4 :=
by
  sorry

end max_sides_convex_polygon_with_four_right_angles_l222_222847


namespace rationalize_denominator_and_min_sum_l222_222178

theorem rationalize_denominator_and_min_sum (A B C D : ℕ) :
  (∀ x, x = (5 * sqrt 2 + sqrt 10) / 4) →
  (D > 0) →
  (∀ p, prime p → ¬ (p ^ 2 ∣ B)) →
  (0 < A ∧  A = 5) →
  (0 < B ∧ B = 10) →
  (0 ≤ C ∧ (C = 1)) →
  (0 < D ∧ D = 4) →
  (A + B + C + D = 20) :=
by
  intros hx hD hB hA h_BC hC hD_sum
  sorry

end rationalize_denominator_and_min_sum_l222_222178


namespace Nori_gave_more_to_Lea_l222_222682

noncomputable def Nori_crayons_initial := 4 * 8
def Mae_crayons := 5
def Nori_crayons_left := 15
def Crayons_given_to_Lea := Nori_crayons_initial - Mae_crayons - Nori_crayons_left
def Crayons_difference := Crayons_given_to_Lea - Mae_crayons

theorem Nori_gave_more_to_Lea : Crayons_difference = 7 := by
  sorry

end Nori_gave_more_to_Lea_l222_222682


namespace unit_digit_of_product_is_4_l222_222283

theorem unit_digit_of_product_is_4 :
  let expr := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1 in
  expr % 10 = 4 :=
by
  -- define the expression 
  let expr : ℕ := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1
  -- ensure the equivalence of unit digit
  show expr % 10 = 4
  sorry -- proof goes here

end unit_digit_of_product_is_4_l222_222283


namespace side_length_irrational_l222_222776

theorem side_length_irrational (s : ℝ) (h : s^2 = 3) : ¬∃ (r : ℚ), s = r := by
  sorry

end side_length_irrational_l222_222776


namespace find_BD_l222_222103
-- Import the entire Mathlib library

-- Define the points A, B, C, and D, and their respective distances
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define the distances
variables (AB AC BC CD : ℝ)
variables(h1 : AC = 10)(h2 : BC = 10)(h3 : AB = 5)(h4 : CD = 13)

-- Define the points relationship
variables (B_between_A_D : B ∈ segment A D)
  
-- The theorem statement
theorem find_BD (BD : ℝ) : BD = real.sqrt 75.25 - 2.5 :=
sorry

end find_BD_l222_222103


namespace sum_of_base8_digits_888_l222_222359

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222359


namespace every_n_has_good_multiple_l222_222547

-- Definitions related to the problem statements
def is_divisor (d n : ℕ) : Prop := d ∣ n
def is_nontrivial_divisor (d n : ℕ) : Prop := is_divisor d n ∧ d > 1

def is_good (n : ℕ) : Prop := 
  ∃ (D : Finset ℕ), (∀ d ∈ D, is_nontrivial_divisor d n) ∧ D.sum = n - 1

-- Theorem proving that every natural number has a good multiple
theorem every_n_has_good_multiple (n : ℕ) : ∃ m : ℕ, is_good (m * n) :=
sorry

end every_n_has_good_multiple_l222_222547


namespace point_P_minimizes_PA_PB_sum_l222_222839

open Real EuclideanGeometry

variables {O A B P K : Point}

def equidistant (O A B : Point) : Prop :=
    dist O A = dist O B

noncomputable def point_minimizes_sum {O A B : Point}
    (equidist_OA_OB : equidistant O A B)
    (K : ℝ) (hK : K > (dist O A)) : Point :=
    let C := Circle O K in
    let A' := point_on_ray OA (K^2 / (dist O A)) in
    let B' := point_on_ray OB (K^2 / (dist O B)) in
    if intersects A' B' C then
        intersection_points A' B' C
    else
        closest_point_on_perpendicular_bisector A' B' C

theorem point_P_minimizes_PA_PB_sum
    (equidist_OA_OB : equidistant O A B)
    (P : Point)
    (K : ℝ) (hK : K > (dist O A))
    (hOP : dist O P = K) :
    P = point_minimizes_sum equidist_OA_OB K hK :=
sorry

end point_P_minimizes_PA_PB_sum_l222_222839


namespace sum_of_digits_base8_888_l222_222326

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222326


namespace boys_in_biology_is_25_l222_222299

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l222_222299


namespace sum_of_digits_base8_l222_222393

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222393


namespace multiply_by_3_l222_222491

variable (x : ℕ)  -- Declare x as a natural number

-- Define the conditions
def condition : Prop := x + 14 = 56

-- The goal to prove
theorem multiply_by_3 (h : condition x) : 3 * x = 126 := sorry

end multiply_by_3_l222_222491


namespace collinear_points_in_cube_l222_222883

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l222_222883


namespace total_amount_of_money_l222_222857

theorem total_amount_of_money (N50 N500 : ℕ) (h1 : N50 = 97) (h2 : N50 + N500 = 108) : 
  50 * N50 + 500 * N500 = 10350 := by
  sorry

end total_amount_of_money_l222_222857


namespace min_value_rationalize_sqrt_denominator_l222_222192

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222192


namespace RT_length_l222_222446

theorem RT_length
  (PQ RS : ℝ) (T : Point)
  (h1 : PQ = 3 * RS)
  (PR : ℝ)
  (h2 : PR = 15)
  (PR_intersects_diagonals_at_T : ∃ T, Point T) :
  length_RT = 15 / 4 := by
  sorry

end RT_length_l222_222446


namespace T_n_formula_l222_222540

-- Define the function f(x)
def f (x : ℝ) : ℝ := x / (3 * x + 1)

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := f (a n)

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (3^n - 2) * 2^(n - 1)

-- Define the sum of the first n terms of sequence T_n
def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), b i / a i

-- The theorem to be proved
theorem T_n_formula (n : ℕ) : T n = 3^n - 2^n + 1 := sorry

end T_n_formula_l222_222540


namespace train_length_calculation_l222_222875

theorem train_length_calculation 
  (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) 
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 25) 
  (h_train_speed_kmph : train_speed_kmph = 57.6) : 
  ∃ train_length, train_length = 250 :=
by
  sorry

end train_length_calculation_l222_222875


namespace opposite_neg_one_half_l222_222763

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222763


namespace opposite_neg_one_half_l222_222765

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l222_222765


namespace Priya_driving_speed_l222_222169

/-- Priya's driving speed calculation -/
theorem Priya_driving_speed
  (time_XZ : ℝ) (rate_back : ℝ) (time_ZY : ℝ)
  (midway_condition : time_XZ = 5)
  (speed_back_condition : rate_back = 60)
  (time_back_condition : time_ZY = 2.0833333333333335) :
  ∃ speed_XZ : ℝ, speed_XZ = 50 :=
by
  have distance_ZY : ℝ := rate_back * time_ZY
  have distance_XZ : ℝ := 2 * distance_ZY
  have speed_XZ : ℝ := distance_XZ / time_XZ
  existsi speed_XZ
  sorry

end Priya_driving_speed_l222_222169


namespace max_pieces_l222_222245

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  list.nodup digits

def five_digits (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem max_pieces :
  ∀ (n : ℕ) (КУСОК ПИРОГ : ℕ), 
    ПИРОГ = КУСОК * n → 
    five_digits ПИРОГ → 
    distinct_digits ПИРОГ → 
    n ≤ 7 :=
begin
  intros n КУСОК ПИРОГ h1 h2 h3,
  -- skip the proof
  sorry
end

end max_pieces_l222_222245


namespace ordered_triples_unique_l222_222082

theorem ordered_triples_unique :
  ∃! (a b c : ℤ), a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ (log a b = c^2) ∧ (a + b + c = 100) := sorry

end ordered_triples_unique_l222_222082


namespace cost_effectiveness_of_large_pie_l222_222273

noncomputable def area (diameter : ℝ) : ℝ :=
  let radius := diameter / 2
  π * radius^2

noncomputable def price_per_cm2 (price : ℝ) (diameter : ℝ) : ℝ :=
  price / area diameter

theorem cost_effectiveness_of_large_pie :
  let small_pie_diameter := 30
  let small_pie_price := 30
  let large_pie_diameter := 40
  let large_pie_price := 40
  price_per_cm2 large_pie_price large_pie_diameter < price_per_cm2 small_pie_price small_pie_diameter :=
by
  sorry

end cost_effectiveness_of_large_pie_l222_222273


namespace inequality_proof_l222_222993

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) : 
  1 / a + 4 / b ≥ 9 / 4 :=
by
  sorry

end inequality_proof_l222_222993


namespace donut_combinations_l222_222503

theorem donut_combinations (donuts types : ℕ) (at_least_one : ℕ) :
  donuts = 7 ∧ types = 5 ∧ at_least_one = 4 → ∃ combinations : ℕ, combinations = 100 :=
by
  intros h
  sorry

end donut_combinations_l222_222503


namespace no_such_function_l222_222531

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x - y :=
by
  sorry

end no_such_function_l222_222531


namespace min_quotient_base_twelve_l222_222873

theorem min_quotient_base_twelve : 
  ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 10) ∧ (1 ≤ b ∧ b ≤ 10) ∧ (1 ≤ c ∧ c ≤ 10) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∀ (a' b' c' : ℕ), (1 ≤ a' ∧ a' ≤ 10) ∧ (1 ≤ b' ∧ b' ≤ 10) ∧ (1 ≤ c' ∧ c' ≤ 10) ∧ (a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c') →
   (144 * a + 12 * b + c) / (a + b + c) ≥ 24.5) :=
sorry

end min_quotient_base_twelve_l222_222873


namespace arithmetic_geometric_sum_l222_222142

theorem arithmetic_geometric_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : a 3 = a 1 + 2 * d) (h3 : a 5 = a 1 + 4 * d) (h4 : (a 3) ^ 2 = a 1 * a 5)
  (h5 : d ≠ 0) : S n = (n^2 + 7 * n) / 4 := sorry

end arithmetic_geometric_sum_l222_222142


namespace find_a_l222_222059

noncomputable def f (x a : ℝ) := log x - a / x

theorem find_a (a : ℝ) (h_a_positive : a > 0)
  (h_f_min : ∃ (x : ℝ), x ∈ set.Icc 1 (real.exp 1) ∧ f x a = 3 / 2) :
  a = -real.sqrt (real.exp 1) :=
sorry

end find_a_l222_222059


namespace quadratic_real_solutions_l222_222097

theorem quadratic_real_solutions (m : ℝ) :
  (∃ (x : ℝ), m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_solutions_l222_222097


namespace min_ABCD_sum_correct_ABCD_l222_222206

theorem min_ABCD_sum : ∃ (A B C D : ℤ), 
  (D > 0) ∧ 
  (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ 
  (∃ m n : ℚ, (m = A * n.sqrt B + (C : ℚ)) ∧ (n = D.inv) ∧ ((⌜(5 : ℚ) * ℚ.sqrt 2 + ℚ.sqrt 10) / (4 : ℚ)⌝ = m * n)) ∧
  (A + B + C + D = 20) :=
sorry

noncomputable def A := 5
noncomputable def B := 10
noncomputable def C := 1
noncomputable def D := 4

theorem correct_ABCD : ((D > 0) ∧ (¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)) ∧ (A + B + C + D = 20)) :=
by {
  split,
  -- D > 0
  exact dec_trivial,
  split,
  -- ¬ ∃ p : ℤ, (prime p) ∧ (p * p ∣ B)
  sorry,
  -- A + B + C + D = 20
  exact dec_trivial
}

end min_ABCD_sum_correct_ABCD_l222_222206


namespace train_cross_time_l222_222877

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 255.03
noncomputable def train_speed_ms : ℝ := 12.5
noncomputable def distance_to_travel : ℝ := train_length + bridge_length
noncomputable def expected_time : ℝ := 30.0024

theorem train_cross_time :
  (distance_to_travel / train_speed_ms) = expected_time :=
by sorry

end train_cross_time_l222_222877


namespace arithmetic_series_sum_l222_222904

theorem arithmetic_series_sum : 
  let seq := λ n : ℕ, 2 * n - 1 in
  let n := 11 in
  let sum := ∑ i in Finset.range n, seq (i + 1) in
  sum = 121 :=
by 
  sorry

end arithmetic_series_sum_l222_222904


namespace opposite_of_half_l222_222771

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222771


namespace sum_first_11_terms_is_55_l222_222046

-- Given conditions:
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}
axiom a1 : ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d
axiom s1 : ∀ n : ℕ, S_n n = n * (a_n 1 + a_n n) / 2
axiom cond : 2 * a_n 7 - a_n 8 = 5

-- Prove:
theorem sum_first_11_terms_is_55 : S_n 11 = 55 :=
sorry

end sum_first_11_terms_is_55_l222_222046


namespace min_value_rationalize_sqrt_denominator_l222_222190

theorem min_value_rationalize_sqrt_denominator : 
  (∃ A B C D : ℤ, (0 < D ∧ ∀ p : ℕ, prime p → p^2 ∣ B → false) ∧ 
  (A * B.sqrt + C) / D = (50.sqrt) / (25.sqrt - 5.sqrt) ∧ 
  D = 5 ∧ A = 5 ∧ B = 2 ∧ C = 1 ∧ A + B + C + D = 12) := sorry

end min_value_rationalize_sqrt_denominator_l222_222190


namespace convex_polyhedron_faces_l222_222697

theorem convex_polyhedron_faces (n : ℕ) (h : n > 0) (poly : polyhedron) (h_convex : convex poly) (h_faces : num_faces poly = 10 * n) :
  ∃ k, count_faces_with_sides poly k ≥ n :=
by
  sorry

end convex_polyhedron_faces_l222_222697


namespace positive_n_value_l222_222947

theorem positive_n_value
    (n : ℝ)
    (h1 : |complex.of_real 2 + complex.I * n| = 4 * real.sqrt 5) :
    n = 2 * real.sqrt 19 := by
  sorry

end positive_n_value_l222_222947


namespace solve_fraction_equation_l222_222276

theorem solve_fraction_equation : ∀ (x : ℝ), (x + 2) / (2 * x - 1) = 1 → x = 3 :=
by
  intros x h
  sorry

end solve_fraction_equation_l222_222276


namespace sequence_a_formula_sequence_b_formula_sequence_c_sum_l222_222968

noncomputable def sequence_a (n : ℕ) : ℕ := n + 1
noncomputable def sequence_b (n : ℕ) : ℕ := 2 ^ n
noncomputable def sequence_c (n : ℕ) : ℚ := 
  n / (2 ^ n) - 1 / (sequence_a n * sequence_a (n + 1))

theorem sequence_a_formula :
  ∀ n : ℕ, sequence_a n = n + 1 :=
sorry

theorem sequence_b_formula :
  ∀ n : ℕ, sequence_b n = 2 ^ n :=
sorry

theorem sequence_c_sum (n : ℕ) :
  let T_n := (Finset.range (n + 1)).sum sequence_c
  in T_n = (3 / 2 : ℚ) - (n + 2) / (2 ^ n) + 1 / (n + 2) :=
sorry

end sequence_a_formula_sequence_b_formula_sequence_c_sum_l222_222968


namespace ball_probability_l222_222637

theorem ball_probability :
  ∀ (total_balls red_balls white_balls : ℕ),
  total_balls = 10 → red_balls = 6 → white_balls = 4 →
  -- Given conditions: Total balls, red balls, and white balls.
  -- First ball drawn is red
  ∀ (first_ball_red : true),
  -- Prove that the probability of the second ball being red is 5/9.
  (red_balls - 1) / (total_balls - 1) = 5/9 :=
by
  intros total_balls red_balls white_balls h_total h_red h_white first_ball_red
  sorry

end ball_probability_l222_222637


namespace cubert_6880_l222_222979

theorem cubert_6880 (h1: real.cbrt 68.8 = 4.098) (h2: real.cbrt 6.88 = 1.902) : real.cbrt 6880 = 19.02 := by
  sorry

end cubert_6880_l222_222979


namespace perfect_square_expression_l222_222922

theorem perfect_square_expression (a b c k x y : ℝ) :
  k = (a + c) / 2 + (1 / 2) * Real.sqrt((a - c) ^ 2 + 4 * b ^ 2) ∨
  k = (a + c) / 2 - (1 / 2) * Real.sqrt((a - c) ^ 2 + 4 * b ^ 2) →
  ∃ t : ℝ, t^2 = a * x^2 + 2 * b * x * y + c * y^2 - k * (x^2 + y^2) :=
sorry

end perfect_square_expression_l222_222922


namespace even_number_sum_odd_composites_l222_222148

theorem even_number_sum_odd_composites (n : ℕ) (h₁ : even n) (h₂ : 40 ≤ n) :
  ∃ (a b : ℕ), odd a ∧ odd b ∧ composite a ∧ composite b ∧ n = a + b := 
by
  sorry

end even_number_sum_odd_composites_l222_222148


namespace not_symmetric_about_point_l222_222579

noncomputable def f (x : ℝ) : ℝ := 
  real.cos (2 * x + real.pi / 3) + sqrt 3 * real.sin (2 * x + real.pi / 3) + 1

theorem not_symmetric_about_point :
  ¬ (∃ y, ∀ x : ℝ, f (-x - π / 4) = y - f (x - π / 4)) :=
sorry

end not_symmetric_about_point_l222_222579


namespace last_three_digits_of_primitive_polynomial_pairs_l222_222224

def is_primitive (p : Polynomial ℤ) : Prop :=
  p.coeffs.gcd = 1

def valid_coeff (n : ℤ) : Prop :=
  n ∈ {1, 2, 3, 4, 5}

theorem last_three_digits_of_primitive_polynomial_pairs :
  let polys := {p : Polynomial ℤ | (∀ i ∈ p.support, valid_coeff (p.coeff i)) ∧ is_primitive p}
  let N := (polys.card)^2
  N % 1000 = 689 :=
sorry

end last_three_digits_of_primitive_polynomial_pairs_l222_222224


namespace picnic_problem_l222_222472

variables (M W C A : ℕ)

theorem picnic_problem
  (H1 : M + W + C = 200)
  (H2 : A = C + 20)
  (H3 : M = 65)
  (H4 : A = M + W) :
  M - W = 20 :=
by sorry

end picnic_problem_l222_222472


namespace boys_in_biology_is_25_l222_222298

-- Definition of the total number of students in the Physics class
def physics_class_students : ℕ := 200

-- Definition of the total number of students in the Biology class
def biology_class_students : ℕ := physics_class_students / 2

-- Condition that there are three times as many girls as boys in the Biology class
def girls_boys_ratio : ℕ := 3

-- Calculate the total number of "parts" in the Biology class (3 parts girls + 1 part boys)
def total_parts : ℕ := girls_boys_ratio + 1

-- The number of boys in the Biology class
def boys_in_biology : ℕ := biology_class_students / total_parts

-- The statement to prove the number of boys in the Biology class is 25
theorem boys_in_biology_is_25 : boys_in_biology = 25 := by
  sorry

end boys_in_biology_is_25_l222_222298


namespace correct_sqrt_multiplication_l222_222817

theorem correct_sqrt_multiplication :
  ∀ (a b : ℝ), (a = 2) → (b = 3) → (sqrt a * sqrt b = sqrt (a * b)) :=
by
  intros a b ha hb
  rw [ha, hb]
  exact Eq.refl (sqrt (2 * 3))

end correct_sqrt_multiplication_l222_222817


namespace part_a_part_b_l222_222035

variable (α : ℝ) (irrational_α : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ α = p / (q:ℚ))
variable (h0 : 0 < α) (h1 : α < 1/2)

noncomputable def α_seq : ℕ → ℝ
| 0     := α
| (n+1) := min (2 * α_seq n) (1 - 2 * α_seq n)

theorem part_a : ∃ n : ℕ, α_seq α < 3 / 16 := sorry

theorem part_b : ¬ ∀ n : ℕ, α_seq α > 7 / 40 := sorry

end part_a_part_b_l222_222035


namespace base10_to_base4_85_l222_222798

theorem base10_to_base4_85 : 
  ∀ (n : ℕ), 
  n = 85 → n = 1 * 4^3 + 1 * 4^2 + 1 * 4^1 + 1 * 4^0  :=
by 
  intro n,
  intro h,
  rw h,
  sorry

end base10_to_base4_85_l222_222798


namespace cost_difference_l222_222515

-- Define the costs
def cost_chocolate : ℕ := 3
def cost_candy_bar : ℕ := 7

-- Define the difference to be proved
theorem cost_difference :
  cost_candy_bar - cost_chocolate = 4 :=
by
  -- trivial proof steps
  sorry

end cost_difference_l222_222515


namespace pairs_removed_when_X_is_8_all_cards_removed_when_X_is_19_number_of_X_for_two_cards_left_l222_222136

variables (cards : set ℕ) (X : ℕ)

-- Here we define the deck as a set of numbers from 1 to 18
def deck : set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 18}

-- Define the condition C1: the set of pairs with a given sum X
def pair_sum_X (X : ℕ) : set (ℕ × ℕ) := 
  {p : ℕ × ℕ | p.1 ∈ deck ∧ p.2 ∈ deck ∧ p.1 + p.2 = X ∧ p.1 ≠ p.2}

-- Proof for (a)
theorem pairs_removed_when_X_is_8 : 
  pair_sum_X 8 = {(1, 7), (2, 6), (3, 5)} :=
sorry

-- Proof for (b)
theorem all_cards_removed_when_X_is_19 : 
  (∀ p ∈ pair_sum_X 19, p.1 ∈ deck ∧ p.2 ∈ deck) ∧
  (∀ c ∈ deck, (c, 19-c) ∈ pair_sum_X 19) :=
sorry

-- Proof for (c)
theorem number_of_X_for_two_cards_left : 
  (∃! X : ℕ, (deck.card - (pair_sum_X X).card = 2)) :=
sorry

end pairs_removed_when_X_is_8_all_cards_removed_when_X_is_19_number_of_X_for_two_cards_left_l222_222136


namespace unique_solution_l222_222934

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
    (∀ x, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
    (∀ x, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

theorem unique_solution (f : ℝ → ℝ) :
    is_solution f → (∀ x, x ≥ 1 → f x = x + 1) := 
sorry

end unique_solution_l222_222934


namespace total_reptiles_l222_222272

theorem total_reptiles 
  (reptiles_in_s1 : ℕ := 523)
  (reptiles_in_s2 : ℕ := 689)
  (reptiles_in_s3 : ℕ := 784)
  (reptiles_in_s4 : ℕ := 392)
  (reptiles_in_s5 : ℕ := 563)
  (reptiles_in_s6 : ℕ := 842) :
  reptiles_in_s1 + reptiles_in_s2 + reptiles_in_s3 + reptiles_in_s4 + reptiles_in_s5 + reptiles_in_s6 = 3793 :=
by
  sorry

end total_reptiles_l222_222272


namespace tetrahedron_edge_length_l222_222777

-- Definitions
def is_tetrahedron (W X Y Z : Type) (d : W → X → ℝ) :=
  let edges := [d W X, d W Y, d W Z, d X Y, d X Z, d Y Z]
  ∧ d W Z = 42

noncomputable def length_of_XY (W X Y Z : Type) (d : W → X → ℝ) :=
  if is_tetrahedron W X Y Z d then 14 else 0

-- Problem statement
theorem tetrahedron_edge_length
  (W X Y Z : Type)
  (d : W → X → ℝ)
  (h : is_tetrahedron W X Y Z d) :
  length_of_XY W X Y Z d = 14 :=
sorry

end tetrahedron_edge_length_l222_222777


namespace opposite_of_num_l222_222749

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222749


namespace acute_angle_inequality_l222_222043

theorem acute_angle_inequality (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := 
sorry

end acute_angle_inequality_l222_222043


namespace max_pieces_is_seven_l222_222257

-- Define what it means for a number to have all distinct digits
def all_digits_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.nodup)

-- Define the main proof problem
theorem max_pieces_is_seven :
  ∃ (n : ℕ) (PIE : ℕ) (PIECE : ℕ),
  (PIE = PIECE * n) ∧
  (PIE >= 10000) ∧ (PIE < 100000) ∧
  all_digits_distinct PIE ∧
  all_digits_distinct PIECE ∧
  ∀ m, (m > n) → (¬ (∃ P' PIECE', (P' = PIECE' * m) ∧
   (P' >= 10000) ∧ (P' < 100000) ∧ all_digits_distinct P' ∧ all_digits_distinct PIECE'))
:= sorry

end max_pieces_is_seven_l222_222257


namespace rationalize_denominator_l222_222214

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222214


namespace opposite_of_num_l222_222750

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222750


namespace tan_sin_cos_eq_two_l222_222958

-- Define the assumption
variable (α : ℝ) (h : sin α + cos α = Real.sqrt 2)

-- Target theorem
theorem tan_sin_cos_eq_two : tan α + (cos α / sin α) = 2 := by
  sorry

end tan_sin_cos_eq_two_l222_222958


namespace two_digit_number_condition_l222_222483

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end two_digit_number_condition_l222_222483


namespace anais_more_toys_than_kamari_l222_222498

theorem anais_more_toys_than_kamari (total_toys : ℕ) (kamari_toys : ℕ) (h1 : total_toys = 160) (h2 : kamari_toys = 65) : 
  ∃ anais_more_toys : ℕ, anais_more_toys = 30 :=
by
  -- We can calculate Anais's toys from the given conditions
  let anais_toys := total_toys - kamari_toys
  -- We know the total toys is 160 and Kamari has 65 toys
  have h_total : anais_toys = 160 - 65, from calc
    anais_toys = total_toys - kamari_toys   : rfl
            ... = 160 - 65                   : by rw [h1, h2]
  -- Calculate the difference between Anais's toys and Kamari's toys
  let anais_more_toys := anais_toys - kamari_toys
  -- Prove the difference is 30
  have h_diff : anais_more_toys = 95 - 65, from calc
    anais_more_toys = anais_toys - kamari_toys : rfl
                    ... = 95 - 65              : by rw h_total
  -- Now we know 95 - 65 is 30
  have final_proof : anais_more_toys = 30, from nat.sub_eq_of_eq_add (nat.sub_eq_of_eq_add rfl)
  -- Return existence of such a number
  use 30
  exact final_proof

end anais_more_toys_than_kamari_l222_222498


namespace equilateral_triangle_pigeonhole_principle_l222_222891

theorem equilateral_triangle_pigeonhole_principle :
  ∀ (T : Type) (triangle : T) (side_length : ℝ)
  (points : Fin 5 → T) (point_inside : ∀ i, is_inside points i triangle),
  equilateral triangle side_length 2 →
  (∃ (i j : Fin 5), i ≠ j ∧ distance (points i) (points j) < 1) := 
sorry

end equilateral_triangle_pigeonhole_principle_l222_222891


namespace find_b_squared_l222_222920

theorem find_b_squared (a b : ℝ) (z : ℂ) :
  (0 < a) ∧ (0 < b) ∧ (|Complex.ofReal a + Complex.I * b| = 5) ∧
  (∀ z : ℂ, |(Complex.ofReal a + Complex.I * b) * z - z| = |(Complex.ofReal a + Complex.I * b) * z - 1|) →
  b^2 = 99 / 4 :=
by
  intros h
  -- Proof omitted
  sorry

end find_b_squared_l222_222920


namespace arithmetic_sequence_sum_range_l222_222158

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range 
  (a d : ℝ)
  (h1 : 1 ≤ a + 3 * d) 
  (h2 : a + 3 * d ≤ 4)
  (h3 : 2 ≤ a + 4 * d)
  (h4 : a + 4 * d ≤ 3) 
  : 0 ≤ S_n a d 6 ∧ S_n a d 6 ≤ 30 := 
sorry

end arithmetic_sequence_sum_range_l222_222158


namespace calc_3_delta_4_l222_222608

def delta (c d : ℝ) : ℝ := (c + d) / (1 + c * d^2)

theorem calc_3_delta_4 : delta 3 4 = 7 / 49 :=
by
  -- Using the given definition, we calculate:
  -- delta 3 4 = (3 + 4) / (1 + 3 * (4^2))
  --            = 7 / (1 + 3 * 16)
  --            = 7 / (1 + 48)
  --            = 7 / 49
  sorry

end calc_3_delta_4_l222_222608


namespace smallest_m_exceeds_15_l222_222943

noncomputable def sum_of_digits (n : ℤ) : ℕ :=
  n.to_nat.digits 10 |>.sum

def fractional_part_sum_of_digits_exceeds_15 (m : ℕ) : Prop :=
  sum_of_digits (( (10 : ℤ) ^ m) / (3 ^ m) ) > 15

theorem smallest_m_exceeds_15 :
  ∃ m, fractional_part_sum_of_digits_exceeds_15 m ∧
    ∀ k, k < m → ¬ fractional_part_sum_of_digits_exceeds_15 k :=
sorry

end smallest_m_exceeds_15_l222_222943


namespace complement_union_l222_222588

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_union : U \ (A ∪ B) = {4} := by
  sorry

end complement_union_l222_222588


namespace max_parts_divided_by_two_planes_l222_222792

theorem max_parts_divided_by_two_planes : 
  ∀ (P1 P2 : Plane), 
    (P1 ∥ P2 → num_parts_divided_by_planes P1 P2 ≤ 3) ∧
    (¬(P1 ∥ P2) → num_parts_divided_by_planes P1 P2 ≤ 4) → 
    ∃ P1 P2, num_parts_divided_by_planes P1 P2 = 4 :=
begin
  sorry
end

end max_parts_divided_by_two_planes_l222_222792


namespace zeros_in_decimal_representation_l222_222077

def term_decimal_zeros (x : ℚ) : ℕ := sorry  -- Function to calculate the number of zeros in the terminating decimal representation.

theorem zeros_in_decimal_representation :
  term_decimal_zeros (1 / (2^7 * 5^9)) = 8 :=
sorry

end zeros_in_decimal_representation_l222_222077


namespace initial_percentage_increase_l222_222260

-- Definition of the problem conditions in Lean
variables {P x : ℝ} 
def initial_increase (P : ℝ) (x : ℝ) := P * (1 + x / 100)
def final_decrease (P : ℝ) (x : ℝ) := initial_increase P x * 0.85
def single_increase (P : ℝ) := P * 1.19

-- Lean 4 statement to prove the problem
theorem initial_percentage_increase : ∀ (P : ℝ) (x : ℝ), 
  final_decrease P x = single_increase P → x = 40 :=
by {
  -- introduce the variables and hypothesis
  intros P x h,
  
  -- write the hypothesis in terms of an equation
  unfold final_decrease single_increase initial_increase at h,
  sorry
}

end initial_percentage_increase_l222_222260


namespace max_tanB_cotC_l222_222153

theorem max_tanB_cotC {a b c x0 y0 z0 : ℝ}
  (h1 : b > max a c)
  (h2 : a * (z0 / x0) + b * (2 * y0 / x0) + c = 0)
  (h3 : (z0 / y0)^2 + (x0 / y0)^2 / 4 = 1) :
  ∃ B C, tan B * cot C ≤ 5 / 3 :=
by
  sorry

end max_tanB_cotC_l222_222153


namespace minimum_weighings_to_identify_fake_coin_l222_222882

theorem minimum_weighings_to_identify_fake_coin :
  ∀ (coins : Fin 13 → ℝ), (∃ i, (coins i = 0) ∧ (∀ j ≠ i, coins j = 1)) →
  ∃ strategy : list (Fin 13 × Fin 13) × list (Fin 13 → bool),
  length strategy.1 = 3 ∧
  (∀ outcome : Fin 13 → Fin 13 → bool,
    determine_fake_coin coins outcome strategy.1 strategy.2 = some i)
  sorry

end minimum_weighings_to_identify_fake_coin_l222_222882


namespace park_area_is_102400_l222_222829

noncomputable def park_length_breadth_ratio : ℕ → ℕ → Prop :=
λ l b, l = 4 * b

noncomputable def cyclist_speed_kmph : ℕ := 12

noncomputable def time_minutes : ℕ := 8

noncomputable def speed_m_per_min : ℕ := (cyclist_speed_kmph * 1000) / 60

noncomputable def perimeter_of_park (speed time : ℕ) : ℕ :=
speed * time

noncomputable def park_perimeter_eq_length_breadth (l b : ℕ) :=
2 * (l + b)

noncomputable def find_area (l b : ℕ) : ℕ :=
l * b

theorem park_area_is_102400 :
  ∃ (l b : ℕ), park_length_breadth_ratio l b ∧
               perimeter_of_park speed_m_per_min time_minutes = park_perimeter_eq_length_breadth l b ∧
               find_area l b = 102400 :=
sorry

end park_area_is_102400_l222_222829


namespace trajectory_of_M_area_ratio_OPQ_BOM_l222_222069

-- Definitions of the parabola and other initial conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus_F : (ℝ × ℝ) := (1, 0)
def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)
def midpoint (A B M : ℝ × ℝ) : Prop := (M.1 = (A.1 + B.1) / 2) ∧ (M.2 = (A.2 + B.2) / 2)
def origin_O : (ℝ × ℝ) := (0, 0)
def line_intersect_x_eq_neg4 (point : ℝ × ℝ) : Prop := point.1 = -4

-- Question (Ⅰ): The equation of the trajectory of the moving point \( M \).
theorem trajectory_of_M (A B M : ℝ × ℝ) (k : ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2 ∧ 
  midpoint A B M →
  (M.2)^2 = 2 * M.1 - 2 :=
sorry

-- Question (Ⅱ): The ratio of the area of triangles \( \triangle OPQ \) and \( \triangle BOM \).
theorem area_ratio_OPQ_BOM (A B P Q M : ℝ × ℝ) (y1 y2 : ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line_intersect_x_eq_neg4 P ∧ line_intersect_x_eq_neg4 Q ∧ 
  midpoint A B M ∧
  P.2 = -16 / y1 ∧ Q.2 = -16 / y2 →
  ∃ (k : ℝ), y1 + y2 = 4 * k ∧ y1 * y2 = -4 ∧ 
  8 * abs (y1 - y2) / ((1/4) * abs (y1 - y2)) = 32 :=
sorry

end trajectory_of_M_area_ratio_OPQ_BOM_l222_222069


namespace brick_height_l222_222846

theorem brick_height (H : ℝ) 
    (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
    (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℝ)
    (volume_wall: wall_length = 900 ∧ wall_width = 500 ∧ wall_height = 1850)
    (volume_brick: brick_length = 21 ∧ brick_width = 10)
    (num_bricks_value: num_bricks = 4955.357142857142) :
    (H = 0.8) :=
by {
  sorry
}

end brick_height_l222_222846


namespace number_of_valid_c_values_l222_222134

theorem number_of_valid_c_values : 
  (∀ (c : ℕ), 0 < c ∧ c < 100) → 
  let S := 72800 + 136 * c in 
  80000 ≤ S ∧ S ≤ 85000 → 
  37 :=
by 
  sorry

end number_of_valid_c_values_l222_222134


namespace tangent_line_at_1_range_of_a_for_two_extreme_points_f_at_x1_less_than_zero_l222_222581

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x

theorem tangent_line_at_1 (a : ℝ) (h : a = 2) :
  ∃ m b, (∀ x, f x a = m * x + b) ∧ (m = -2) ∧ (b = -2) := by
  sorry

theorem range_of_a_for_two_extreme_points :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ (∀ x, f' x a = 0 → x = x1 ∨ x = x2)) →
  0 < a ∧ a < 1 / Real.e :=
  by sorry

theorem f_at_x1_less_than_zero
  (a : ℝ) (x1 x2 : ℝ)
  (h : ∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ (∀ x, f' x a = 0 → x = x1 ∨ x = x2)) :
  f x1 a < 0 := by
  sorry

end tangent_line_at_1_range_of_a_for_two_extreme_points_f_at_x1_less_than_zero_l222_222581


namespace number_of_matches_among_quitting_players_l222_222107

-- Definitions for conditions
variable (n r : ℕ) -- n is the total number of players initially, r is matches among the three quitting players
variable (totalMatches : ℕ := 50) -- Total matches played is 50

-- Each quitting player played 2 matches
def quittingPlayerMatches : ℕ := 6 - r

-- Remaining players after three quitting
def remainingPlayers : ℕ := n - 3

-- Matches played among remaining players using binomial coefficient nC2
def matchesAmongRemainingPlayers : ℕ := (remainingPlayers * (remainingPlayers - 1)) / 2

-- Hypothesis based on the problem statement
def hypothesis : Prop := matchesAmongRemainingPlayers + quittingPlayerMatches = totalMatches

-- The proof statement
theorem number_of_matches_among_quitting_players :
  hypothesis n r → r = 1 :=
by
  -- proof omitted
  sorry

end number_of_matches_among_quitting_players_l222_222107


namespace square_root_expression_correct_l222_222223

/-- Given conditions for the digits a, b, c, d, e, f, g, and h 
    to satisfy the equation and constraints provided. -/
theorem square_root_expression_correct (a b c d e f g h : ℕ)
  (h1 : ∀ n, n = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
  (h2 : sqrt (a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) = g * 100 + f * 10 + c)
  (h3 : d * 100 + c * 10 + d = (d * 10 + f) * f)
  (h4 : b * 1000 + b * 100 + e * 10 + f = (c * 10 + h) * 10 + c) :
  a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f = 157609 :=
sorry

end square_root_expression_correct_l222_222223


namespace problem_M_plus_N_l222_222084

theorem problem_M_plus_N (M N : ℝ) (H1 : 4/7 = M/77) (H2 : 4/7 = 98/(N^2)) : M + N = 57.1 := 
sorry

end problem_M_plus_N_l222_222084


namespace find_vector_magnitude_l222_222716

variables {a b : EuclideanSpace ℝ (fin 2)}
variables (θ : ℝ)
noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (fin 2)) : ℝ := 
  real.angle.angleCos (a, b)

def magnitude (v : EuclideanSpace ℝ (fin 2)) : ℝ := 
  ∥v∥

theorem find_vector_magnitude 
  (h1 : angle_between_vectors a b = real.pi / 3)
  (h2 : magnitude a = 2)
  (h3 : magnitude b = 1) :
  magnitude (a - 2 • b) = 2 :=
by
  sorry

end find_vector_magnitude_l222_222716


namespace quadratic_conversion_l222_222513

theorem quadratic_conversion (x : ℝ) :
  (2*x - 1)^2 = (x + 1)*(3*x + 4) →
  ∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a*x^2 + b*x + c = 0 :=
by simp [pow_two, mul_add, add_mul, mul_comm]; sorry

end quadratic_conversion_l222_222513


namespace max_value_of_f_on_I_l222_222529

-- Define the function
def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the interval
def I : Set ℝ := Set.Icc 0 2

-- Statement to prove the maximum value
theorem max_value_of_f_on_I : ∃ (c : ℝ), c ∈ I ∧ ∀ x ∈ I, f x ≤ f 1 := by
  use 1
  split
  . exact Set.mem_Icc.mpr ⟨le_refl 0, by norm_num⟩
  . intro x hx
    have : f 1 = 1 / Real.exp 1 := by simp [f]
    sorry -- Proof required here, representing the steps to show f(x) <= f(1)

end max_value_of_f_on_I_l222_222529


namespace find_x2_plus_y2_plus_z2_l222_222915

-- Define the matrix N
def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, 2 * y, 0], ![-x, y, z], ![x, -y, z]]

-- Define the transpose of N
def N_transpose (x y z : ℝ) := (N x y z)ᵀ

-- Define the identity matrix I
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

-- The theorem to be proven
theorem find_x2_plus_y2_plus_z2 (x y z : ℝ) (h : N_transpose x y z ⬝ N x y z = I) :
    x^2 + y^2 + z^2 = 1 :=
by
  sorry

end find_x2_plus_y2_plus_z2_l222_222915


namespace chessboard_problem_l222_222924

theorem chessboard_problem (n k : ℕ) (hn : 0 < n) (hk : 0 < k)
    (positive_sum : ∀ (chessboard : matrix (fin n) (fin n) ℤ), 0 < matrix.sum chessboard)
    (negative_k_subboard : ∀ (chessboard : matrix (fin n) (fin n) ℤ), 
      ∀ i j : ℕ, i + k ≤ n → j + k ≤ n →  matrix.sum (submatrix chessboard (fin.of_nat_lt ⟨i, _⟩) (fin.of_nat_lt ⟨j, _⟩)) < 0) :
    ¬(k ∣ n) :=
  sorry

end chessboard_problem_l222_222924


namespace train_speed_is_correct_l222_222876

noncomputable def train_length : ℕ := 900
noncomputable def platform_length : ℕ := train_length
noncomputable def time_in_minutes : ℕ := 1
noncomputable def distance_covered : ℕ := train_length + platform_length
noncomputable def speed_m_per_minute : ℕ := distance_covered / time_in_minutes
noncomputable def speed_km_per_hr : ℕ := (speed_m_per_minute * 60) / 1000

theorem train_speed_is_correct :
  speed_km_per_hr = 108 :=
by
  sorry

end train_speed_is_correct_l222_222876


namespace max_cosA_cosB_l222_222130

open Real

theorem max_cosA_cosB {A B : ℝ} (h_triangle : A + B < π) (h_AB : A > 0 ∧ B > 0)
  (h_sin : sin A * sin B = (2 - sqrt 3) / 4) :
  ∃ M, M = (2 + sqrt 3) / 4 ∧ ∀ (x y : ℝ), h_AB → x = A → y = B → cos x * cos y ≤ M :=
sorry

end max_cosA_cosB_l222_222130


namespace People_Distribution_l222_222838

theorem People_Distribution 
  (total_people : ℕ) 
  (total_buses : ℕ) 
  (equal_distribution : ℕ) 
  (h1 : total_people = 219) 
  (h2 : total_buses = 3) 
  (h3 : equal_distribution = total_people / total_buses) : 
  equal_distribution = 73 :=
by 
  intros 
  sorry

end People_Distribution_l222_222838


namespace min_value_of_expression_l222_222155

theorem min_value_of_expression (x y : ℝ) : 
  ∃ m : ℝ, m = (xy - 1)^2 + (x + y)^2 ∧ (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ m) := 
sorry

end min_value_of_expression_l222_222155


namespace sum_of_digits_base8_l222_222388

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222388


namespace rationalize_denominator_min_value_l222_222198

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222198


namespace equilateral_triangle_area_with_inscribed_circle_l222_222313

noncomputable def radius_of_inscribed_circle (area_circle: ℝ) : ℝ :=
  real.sqrt (area_circle / real.pi)

def side_length_of_equilateral_triangle (radius: ℝ) : ℝ :=
  radius * 2 * real.sqrt 3

def height_of_equilateral_triangle (radius: ℝ) : ℝ :=
  2 * radius

def area_of_equilateral_triangle (radius: ℝ) : ℝ :=
  let side_length := side_length_of_equilateral_triangle radius
  in 1 / 2 * side_length * height_of_equilateral_triangle radius

theorem equilateral_triangle_area_with_inscribed_circle :
  area_of_equilateral_triangle 3 = 18 * real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_with_inscribed_circle_l222_222313


namespace minimum_distance_between_curve_and_line_range_of_slope_for_intersection_l222_222028

open Real

noncomputable def rho := sqrt 2

def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 2

def line_l (alpha t : ℝ) : ℝ × ℝ := (2 + t * cos alpha, 2 + t * sin alpha)

def line_standard_form (alpha : ℝ) : ℝ → ℝ → Prop :=
λ x y, ∃ t : ℝ, x = 2 + t * cos alpha ∧ y = 2 + t * sin alpha

theorem minimum_distance_between_curve_and_line (alpha : ℝ) (h : alpha = 3 / 4 * π) :
  ∀ (P Q : ℝ × ℝ), curve_C P.1 P.2 → line_standard_form alpha Q.1 Q.2 → dist P Q ≥ sqrt 2 := sorry

theorem range_of_slope_for_intersection (k : ℝ) :
  (∃ alpha, k = tan alpha) →
  (∀ (x y : ℝ), (curve_C x y ∧ line_standard_form alpha x y) → (∃! P Q, P ≠ Q)) ↔ (2 - sqrt 3 < k ∧ k < 2 + sqrt 3) := sorry

end minimum_distance_between_curve_and_line_range_of_slope_for_intersection_l222_222028


namespace clipping_per_friend_l222_222664

def GluePerClipping : Nat := 6
def TotalGlue : Nat := 126
def TotalFriends : Nat := 7

theorem clipping_per_friend :
  (TotalGlue / GluePerClipping) / TotalFriends = 3 := by
  sorry

end clipping_per_friend_l222_222664


namespace sum_of_digits_base8_888_l222_222327

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222327


namespace cubic_meters_to_cubic_feet_l222_222594

theorem cubic_meters_to_cubic_feet :
  (let feet_per_meter := 3.28084
  in (feet_per_meter ^ 3) * 2 = 70.6294) :=
by
  sorry

end cubic_meters_to_cubic_feet_l222_222594


namespace math_proof_problem_l222_222029

-- Define the function f
def f (x : ℝ) (n : ℝ) (m : ℝ) : ℝ := (n - 2^x) / (2^(x+1) + m)

-- Define the conditions as hypotheses
variables {x : ℝ} {m n : ℝ}

-- Define the main theorem to prove
theorem math_proof_problem (h1 : ∀ x: ℝ, f(x, 1, 2) = -f(-x, 1, 2)) :
  m = 2 ∧ n = 1 ∧ (∀ x : ℝ, x ∈ [1/2, 3] → ∀ k : ℝ, f(k*x^2, 1, 2) + f(2*x-1, 1, 2) > 0 → k < -1) := 
sorry

end math_proof_problem_l222_222029


namespace count_divisors_from_1_to_10_l222_222081

open Nat

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_divisors_count : ℕ :=
  let n := 30_240
  let divisors := [1, 2, 3, 4, 5, 6, 8, 9, 10].filter (λ d => is_divisor d n)
  divisors.length

theorem count_divisors_from_1_to_10 (n : ℕ) (h : n = 30_240) : valid_divisors_count = 9 := by
  sorry

end count_divisors_from_1_to_10_l222_222081


namespace exists_k0_for_plane_division_l222_222698

theorem exists_k0_for_plane_division :
  ∃ k0 : ℕ, (∃ n : ℕ, n > k0 ∧ ∀ k : ℕ, k > k0 → ∃ (lines : ℕ), not_all_parallel lines ∧ regions lines = k) ∧ k0 = 5 :=
begin
  sorry
end

end exists_k0_for_plane_division_l222_222698


namespace compute_d_for_ellipse_l222_222495

theorem compute_d_for_ellipse
  (in_first_quadrant : true)
  (is_tangent_x_axis : true)
  (is_tangent_y_axis : true)
  (focus1 : (ℝ × ℝ) := (5, 4))
  (focus2 : (ℝ × ℝ) := (d, 4)) :
  d = 3.2 := by
  sorry

end compute_d_for_ellipse_l222_222495


namespace range_of_a_l222_222089

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + a + 3 ≥ 0) ↔ a ∈ set.Ici 0 := sorry

end range_of_a_l222_222089


namespace count_valid_numbers_correct_l222_222076

def odd_digits : Finset ℕ := {1, 3, 5, 7, 9}

def count_valid_numbers : ℕ :=
  let all_numbers := 5^5 in
  let invalid_numbers := 5 * 4^4 in
  all_numbers - invalid_numbers

theorem count_valid_numbers_correct : count_valid_numbers = 1845 := by
  sorry

end count_valid_numbers_correct_l222_222076


namespace range_of_a_l222_222571

theorem range_of_a :
  (∀ x : ℝ, f'(x) = -exp x - 1) →
  (∀ x : ℝ, g'(x) = a - 2 * sin x) →
  (∀ x : ℝ, (f'(x) * g'(x) = -1)) → 
  (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l222_222571


namespace limit_expression_l222_222506

open Real

noncomputable def limit_cos_exp (f g : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → abs (f x - g x) < ε

theorem limit_expression :
  limit_cos_exp (λ x, (1 - cos x) / (exp (3 * x) - 1)^2) (λ x, 1 / 18) :=
sorry

end limit_expression_l222_222506


namespace maximum_pieces_l222_222251

theorem maximum_pieces :
  ∀ (ПИРОГ КУСОК : ℕ) (h1 : ПИРОГ = 95207) (h2 : КУСОК = 13601),
    (ПИРОГ = КУСОК * 7) ∧ (ПИРОГ < 100000) ∧ (ПИРОГ.to_digits.nodup) → 
    7 = 7 :=
by { sorry }

end maximum_pieces_l222_222251


namespace campers_with_red_hair_l222_222719

variables (T B G K R : ℕ)

-- Given conditions
def condition1 : Prop := B = 25
def condition2 : Prop := B = 0.5 * T
def condition3 : Prop := G + K = 15
def question : Prop := R = T - (B + G + K)

-- The target statement we need to prove
theorem campers_with_red_hair : condition1 ∧ condition2 ∧ condition3 → question :=
sorry

end campers_with_red_hair_l222_222719


namespace simplify_expression_l222_222226

noncomputable def problem_expression : ℝ :=
  (0.25)^(-2) + 8^(2/3) - real.log 25 / real.log 10 - 2 * (real.log 2 / real.log 10)

theorem simplify_expression : problem_expression = 18 :=
by
  sorry

end simplify_expression_l222_222226


namespace count_perfect_square_n_l222_222080

theorem count_perfect_square_n : 
  (set.count {n : ℤ | 4 ≤ n ∧ n ≤ 15 ∧ ∃ k : ℤ, 1 * n^4 + 2 * n^3 + 3 * n^2 + 2 * n + 1 = k^2} = 12) :=
sorry

end count_perfect_square_n_l222_222080


namespace opposite_of_neg_half_l222_222754

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222754


namespace find_sequence_value_l222_222533
-- Import the required libraries

-- Declare the problem within Lean's logic framework
theorem find_sequence_value (x : ℕ → ℝ)
  (h_distinct: ∀ n m, n ≥ 2 → m ≥ 2 → n ≠ m → x n ≠ x m)
  (h_rel: ∀ n, n ≥ 2 → x n = (x (n - 1) + 398 * x n + x (n + 1)) / 400) :
  sqrt ((x 2023 - x 2) / 2021 * (2022 / (x 2023 - x 1))) + 2021 = 2022 :=
by
  sorry

end find_sequence_value_l222_222533


namespace A_B_work_together_finish_l222_222488
noncomputable def work_rate_B := 1 / 12
noncomputable def work_rate_A := 2 * work_rate_B
noncomputable def combined_work_rate := work_rate_A + work_rate_B

theorem A_B_work_together_finish (hB: work_rate_B = 1/12) (hA: work_rate_A = 2 * work_rate_B) :
  (1 / combined_work_rate) = 4 :=
by
  -- Placeholder for the proof, we don't need to provide the proof steps
  sorry

end A_B_work_together_finish_l222_222488


namespace f_2017_equal_2017_l222_222710

noncomputable def f : ℝ → ℝ := sorry

axiom fx_plus_3_le : ∀ x : ℝ, f(x + 3) ≤ f(x) + 3
axiom fx_plus_2_ge : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2
axiom f1 : f(1) = 1

theorem f_2017_equal_2017 : f(2017) = 2017 :=
by
  exact sorry

end f_2017_equal_2017_l222_222710


namespace sum_red_equals_sum_blue_l222_222457

variable (r1 r2 r3 r4 b1 b2 b3 b4 w1 w2 w3 w4 : ℝ)

theorem sum_red_equals_sum_blue (h : (r1 + w1 / 2) + (r2 + w2 / 2) + (r3 + w3 / 2) + (r4 + w4 / 2) 
                                 = (b1 + w1 / 2) + (b2 + w2 / 2) + (b3 + w3 / 2) + (b4 + w4 / 2)) : 
  r1 + r2 + r3 + r4 = b1 + b2 + b3 + b4 :=
by sorry

end sum_red_equals_sum_blue_l222_222457


namespace pages_per_day_l222_222827

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 63) (h2 : days = 3) : total_pages / days = 21 :=
by
  sorry

end pages_per_day_l222_222827


namespace hyperbola_eccentricity_is_sqrt_5_l222_222065

noncomputable def hyperbola_eccentricity (m : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  real.sqrt (1 + (b/a)^2) -- Defining eccentricity

theorem hyperbola_eccentricity_is_sqrt_5
  (m : ℝ) (h : m = 4) : 
  hyperbola_eccentricity m (1/2) 1 = real.sqrt 5 :=
  by {
      sorry
  } 

end hyperbola_eccentricity_is_sqrt_5_l222_222065


namespace kendra_words_learned_l222_222660

theorem kendra_words_learned (Goal : ℕ) (WordsNeeded : ℕ) (WordsAlreadyLearned : ℕ) 
  (h1 : Goal = 60) (h2 : WordsNeeded = 24) :
  WordsAlreadyLearned = Goal - WordsNeeded :=
sorry

end kendra_words_learned_l222_222660


namespace acute_angled_triangle_perimeter_gt_4R_l222_222173

variable {T : Type} [Triangle T]
variable {P R : ℝ} -- Perimeter and Circumradius
variable (acute : T → Prop) -- The triangle is acute-angled
variable (perimeter : T → ℝ) -- Function to get the perimeter of the triangle
variable (circumradius : T → ℝ) -- Function to get the circumradius of the triangle

theorem acute_angled_triangle_perimeter_gt_4R (t : T) (h_acute : acute t) :
  perimeter t > 4 * circumradius t := 
sorry

end acute_angled_triangle_perimeter_gt_4R_l222_222173


namespace circumscribed_sphere_surface_area_l222_222552

-- Define the setup and conditions for the right circular cone and its circumscribed sphere
theorem circumscribed_sphere_surface_area (PA PB PC AB R : ℝ)
  (h1 : AB = Real.sqrt 2)
  (h2 : PA = 1)
  (h3 : PB = 1)
  (h4 : PC = 1)
  (h5 : R = Real.sqrt 3 / 2 * PA) :
  4 * Real.pi * R ^ 2 = 3 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l222_222552


namespace no_second_quadrant_l222_222096

theorem no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (x < 0 → 3 * x + k - 2 ≤ 0)) → k ≤ 2 :=
by
  intro h
  sorry

end no_second_quadrant_l222_222096


namespace rationalize_denominator_l222_222211

theorem rationalize_denominator : 
  ∃ (A B C D : ℤ), D > 0 ∧ (¬ ∃ (p : ℤ), prime p ∧ p^2 ∣ B) ∧
  (A * √ 2 + C) / D = ((√ 50) / (√ 25 - √ 5)) ∧
  A + B + C + D = 12 := 
by
  sorry

end rationalize_denominator_l222_222211


namespace opposite_of_neg_half_l222_222757

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222757


namespace minimum_value_128_l222_222145

theorem minimum_value_128 (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_prod: a * b * c = 8) : 
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 := 
by
  sorry

end minimum_value_128_l222_222145


namespace coeff_x_squared_l222_222570

theorem coeff_x_squared (n : ℕ) (t h : ℕ)
  (h_t : t = 4^n) 
  (h_h : h = 2^n) 
  (h_sum : t + h = 272)
  (C : ℕ → ℕ → ℕ) -- binomial coefficient notation, we'll skip the direct proof of properties for simplicity
  : (C 4 4) * (3^0) = 1 := 
by 
  /-
  Proof steps (informal, not needed in Lean statement):
  Since the sum of coefficients is t, we have t = 4^n.
  For the sum of binomial coefficients, we have h = 2^n.
  Given t + h = 272, solve for n:
    4^n + 2^n = 272 
    implies 2^n = 16, so n = 4.
  Substitute into the general term (\(T_{r+1}\):
    T_{r+1} = C_4^r * 3^(4-r) * x^((8+r)/6)
  For x^2 term, set (8+r)/6 = 2, yielding r = 4.
  The coefficient is C_4^4 * 3^0 = 1.
  -/
  sorry

end coeff_x_squared_l222_222570


namespace coefficient_x3_expansion_l222_222939

/-- The coefficient of x^3 in the expansion of (1 - x)^6 * (1 + x + x^2) is -11. -/
theorem coefficient_x3_expansion :
  ∑ r in {0, 1, 2, 3, 4, 5, 6}, (-1)^r * (Nat.choose 6 r) * (coeff (X^r * (1 + X + X^2)) 3) = -11 :=
by
  sorry

end coefficient_x3_expansion_l222_222939


namespace max_distance_with_optimal_swapping_l222_222954

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end max_distance_with_optimal_swapping_l222_222954


namespace grandfather_age_l222_222814

theorem grandfather_age :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 10 * a + b = a + b^2 ∧ 10 * a + b = 89 :=
by
  sorry

end grandfather_age_l222_222814


namespace find_f_of_minus_five_l222_222238

theorem find_f_of_minus_five (a b : ℝ) (f : ℝ → ℝ) (h1 : f 5 = 7) (h2 : ∀ x, f x = a * x + b * Real.sin x + 1) : f (-5) = -5 :=
by
  sorry

end find_f_of_minus_five_l222_222238


namespace calc_expr_eq_simplify_expr_eq_l222_222443

-- Problem 1: Calculation
theorem calc_expr_eq : 
  ((1 / 2) ^ (-2) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20) = 3 - 2 * Real.sqrt 5 := 
  by
  sorry

-- Problem 2: Simplification
theorem simplify_expr_eq (x : ℝ) (hx : x ≠ 0): 
  ((x^2 - 2 * x + 1) / (x^2 - 1) / (x - 1) / (x^2 + x)) = x := 
  by
  sorry

end calc_expr_eq_simplify_expr_eq_l222_222443


namespace min_filtrations_l222_222125

open Real

noncomputable def log2 := 0.3010
noncomputable def log3 := 0.4771
noncomputable def log8_div_10 : ℝ := log10 (8/10)

theorem min_filtrations (n : ℤ) : 
  (∀ n, 0.8^n < 0.05) → n ≥ 14 :=
by {
  -- Only theorem statement is required
  sorry
}

end min_filtrations_l222_222125


namespace min_value_expr_l222_222530

theorem min_value_expr (x : ℝ) (h : x = 10 + 10 * real.sqrt 2) :
    (x^2 + 100) / (x - 10) = 20 + 20 * real.sqrt 2 := by
  sorry

end min_value_expr_l222_222530


namespace minimal_difference_partition_l222_222784

noncomputable def flowerbed_areas : List ℕ := [9, 16, 25, 64, 81]

theorem minimal_difference_partition :
  ∃ (group1 group2 : List ℕ), (group1 ++ group2 = flowerbed_areas) ∧
  (list.sum group1, list.sum group2) = (97, 98) :=
by {
  sorry
}

end minimal_difference_partition_l222_222784


namespace two_cubic_meters_to_cubic_feet_l222_222596

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end two_cubic_meters_to_cubic_feet_l222_222596


namespace hyperbola_major_axis_length_l222_222997

theorem hyperbola_major_axis_length
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (hyp : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → True)
  (OM_MF2_perp : OM ⊥ MF2)
  (M_on_asymptote : M_on_asymptote (a b) M)
  (area_triangle : ∃ OM MF2 : ℝ, 1/2 * OM * MF2 = 16)
  (same_eccentricity : eccentricity a b = eccentricity 4 2) :
  2 * a = 16 := 
sorry

end hyperbola_major_axis_length_l222_222997


namespace max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l222_222568

-- Problem (1)
theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x*y ≤ 4 :=
sorry

-- Additional statement to show when the maximum is achieved
theorem max_xy_is_4 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x = 4 ∧ y = 1 ↔ x*y = 4 :=
sorry

-- Problem (2)
theorem min_x_plus_y (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x + y ≥ 9 :=
sorry

-- Additional statement to show when the minimum is achieved
theorem min_x_plus_y_is_9 (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x = 6 ∧ y = 3 ↔ x + y = 9 :=
sorry

end max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l222_222568


namespace sum_of_base8_digits_888_l222_222360

def sumDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n < base then n else
  sumDigits (n / base) base + (n % base)

theorem sum_of_base8_digits_888 : sumDigits 888 8 = 13 := by
  sorry

end sum_of_base8_digits_888_l222_222360


namespace concurrency_O4X_O5Y_O6Z_l222_222429

-- Define the points A, B, C, P, R, T as points in the Euclidean plane ℝ²
variables {A B C P R T : ℝ × ℝ}

-- Define O4, O5, O6 as centers of squares APQR, PBT V, and TCRS respectively
def O4 := (A + P + Q + R) / 4
def O5 := (P + B + T + V) / 4
def O6 := (T + C + R + S) / 4

-- Define X, Y, Z as midpoints of sides BC, CA, AB respectively
def X := (B + C) / 2
def Y := (C + A) / 2
def Z := (A + B) / 2

-- Define the perpendicular bisectors as lines passing through O4, O5, O6 and X, Y, Z respectively
def perp_bis_O4X := Line_through O4 X
def perp_bis_O5Y := Line_through O5 Y
def perp_bis_O6Z := Line_through O6 Z

-- The theorem statement in Lean confirming the concurrency at the circumcenter of triangle ABC
theorem concurrency_O4X_O5Y_O6Z :
  Concurrent perp_bis_O4X perp_bis_O5Y perp_bis_O6Z := sorry

end concurrency_O4X_O5Y_O6Z_l222_222429


namespace find_four_digit_number_l222_222844

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l222_222844


namespace complex_number_solution_l222_222048

theorem complex_number_solution (z : ℂ) (h : (2 + complex.I) * z = 3 + 4 * complex.I) : z = 2 + complex.I := 
sorry

end complex_number_solution_l222_222048


namespace num_int_vals_not_satisfying_inequality_l222_222011

-- Definitions
def quadratic_expr (x : ℤ) : ℤ := 3 * x^2 + 11 * x + 4

-- Statement of the problem
theorem num_int_vals_not_satisfying_inequality : 
  {x : ℤ | quadratic_expr x ≤ 21}.to_finset.card = 19 := 
sorry

end num_int_vals_not_satisfying_inequality_l222_222011


namespace together_time_l222_222825

theorem together_time (P_time Q_time : ℝ) (hP : P_time = 4) (hQ : Q_time = 6) : (1 / ((1 / P_time) + (1 / Q_time))) = 2.4 :=
by
  sorry

end together_time_l222_222825


namespace sum_of_digits_base8_l222_222386

theorem sum_of_digits_base8 (n : ℕ) (h : n = 888) : 
  (let d0 := (n % 8) 
       n1 := (n / 8)
       d1 := (n1 % 8) 
       n2 := (n1 / 8)
       d2 := (n2 % 8)
       n3 := (n2 / 8) in
   d0 + d1 + d2 + n3) = 13 :=
by {
  have h₁ : n = 888,
  { exact h, },
  let b8 := 8,
  let step1 := n % b8,
  let n1 := n / b8,
  let step2 := n1 % b8,
  let n2 := n1 / b8,
  let step3 := n2 % b8,
  let n3 := n2 / b8,
  have h₂ : step1 = 0,
  { rw h₁, norm_num, },
  have h₃ : step2 = 7,
  { rw h₁, norm_num, },
  have h₄ : step3 = 5,
  { rw h₁, norm_num, },
  have h₅ : n3 = 1,
  { rw h₁, norm_num, },
  rw [h₂, h₃, h₄, h₅],
  norm_num,
  sorry
}

end sum_of_digits_base8_l222_222386


namespace Ava_vs_Mia_l222_222164

variable (David_shells : ℕ) (Total_shells : ℕ)
variable (Mia_shells_multiplier : ℕ) (Alice_shells_divider : ℕ)
variable (Mia_shells : ℕ) (Ava_shells : ℕ) (Alice_shells : ℕ)

-- Given conditions
def David_shells := 15
def Mia_shells_multiplier := 4
def Alice_shells_divider := 2
def Total_shells := 195
def Mia_shells := Mia_shells_multiplier * David_shells
def Alice_shells (Ava_shells : ℕ) := Ava_shells / Alice_shells_divider

-- Prove that Ava has 20 more shells than Mia
theorem Ava_vs_Mia 
  (Ava_shells Mia_shells Alice_shells Total_shells : ℕ)
  (David_shells = 15)
  (Mia_shells = 4 * David_shells)
  (Alice_shells = Ava_shells / 2)
  (15 + Mia_shells + Ava_shells + Alice_shells = 195) :
  Ava_shells - Mia_shells = 20 := 
  by 
    sorry

end Ava_vs_Mia_l222_222164


namespace correct_equation_l222_222419

theorem correct_equation (A B C D : Prop) : 
  A ↔ (sqrt 3) ^ 2 = 3 ∧
  B ↔ sqrt ((-3) ^ 2) = -3 ∧
  C ↔ sqrt (3 ^ 3) = 3 ∧
  D ↔ (-sqrt 3) ^ 2 = -3 ∧
  A ∧ ¬B ∧ ¬C ∧ ¬D :=
by {sorry}

end correct_equation_l222_222419


namespace opposite_of_num_l222_222753

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l222_222753


namespace team_B_score_third_game_l222_222879

theorem team_B_score_third_game (avg_points : ℝ) (additional_needed : ℝ) (total_target : ℝ) (P : ℝ) :
  avg_points = 61.5 → additional_needed = 330 → total_target = 500 →
  2 * avg_points + P + additional_needed = total_target → P = 47 :=
by
  intros avg_points_eq additional_needed_eq total_target_eq total_eq
  rw [avg_points_eq, additional_needed_eq, total_target_eq] at total_eq
  sorry

end team_B_score_third_game_l222_222879


namespace sum_of_digits_base_8_rep_of_888_l222_222351

theorem sum_of_digits_base_8_rep_of_888 : 
  let n := 888
  let base_8_rep := (1, 5, 7, 0)
  let digit_sum := 1 + 5 + 7 + 0
  is_base_8_rep (n : ℕ) (base_8_rep : ℕ × ℕ × ℕ × ℕ) → 
  digit_sum = 13 := 
by
  sorry

end sum_of_digits_base_8_rep_of_888_l222_222351


namespace sum_of_digits_base8_888_l222_222328

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222328


namespace medians_greater_than_29_square_l222_222170

-- Define the condition of an acute triangle in Lean
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 > 2 * max a b^2 + min (a b^2)

-- Define the median lengths based on side lengths
def median_length (a b c : ℝ) : ℝ := 
  (Math.sqrt (2 * b^2 + 2 * c^2 - a^2)) / 4

-- Prove the main inequality problem
theorem medians_greater_than_29_square (a b c r : ℝ)
  (hacute: is_acute_triangle a b c) :
  median_length a b c ^ 2 + (median_length b c a ^ 2) > 29 * r^2 := 
by
  sorry -- proof goes here

end medians_greater_than_29_square_l222_222170


namespace F_minimum_value_neg_inf_to_0_l222_222609

variable (f g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) := ∀ x, h (-x) = - (h x)

theorem F_minimum_value_neg_inf_to_0 
  (hf_odd : is_odd f) 
  (hg_odd : is_odd g)
  (hF_max : ∀ x > 0, f x + g x + 2 ≤ 8) 
  (hF_reaches_max : ∃ x > 0, f x + g x + 2 = 8) :
  ∀ x < 0, f x + g x + 2 ≥ -4 :=
by
  sorry

end F_minimum_value_neg_inf_to_0_l222_222609


namespace intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l222_222870

variables (x y z : ℕ)

-- Conditions
axiom condition_1 : ∀ (t: ℕ), t = (6 : ℕ) → y * z = 6 * (y - x)
axiom condition_2 : ∀ (t: ℕ), t = (3 : ℕ) → y * z = 3 * (y + x)

-- Proof statements
theorem intervals_between_trolleybuses : z = 4 :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

theorem sportsman_slower_than_trolleybus : y = 3 * x :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

end intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l222_222870


namespace rationalize_denominator_min_value_l222_222201

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222201


namespace ratio_AE_EQ_l222_222696

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_square (A B C D : Point) : Prop :=
  A.x = 0 ∧ A.y = 0 ∧
  B.x = 1 ∧ B.y = 0 ∧
  C.x = 1 ∧ C.y = 1 ∧
  D.x = 0 ∧ D.y = 1

def point_on_line (A B P : Point) (r s : ℝ) : Prop :=
  P.x = (r * B.x + s * A.x) / (r + s) ∧
  P.y = (r * B.y + s * A.y) / (r + s)

def lines_intersect (A B C D : Point) (E : Point) : Prop :=
  let m1 := (B.y - A.y) / (B.x - A.x)
  let m2 := (D.y - C.y) / (D.x - C.x)
  ∃ x y, E.x = x ∧ E.y = y ∧ 
           y = m1 * (x - A.x) + A.y ∧
           y = m2 * (x - C.x) + C.y

theorem ratio_AE_EQ (A B C D P Q E : Point)
  (h_square : is_square A B C D)
  (h_P_on_AB : point_on_line A B P 2 3)
  (h_Q_on_BC : point_on_line B C Q 3 1)
  (h_E_intersect : lines_intersect D P A Q E) :
  (dist A E) / (dist E Q) = 4 / 9 :=
sorry

end ratio_AE_EQ_l222_222696


namespace sum_of_digits_base8_888_l222_222378

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222378


namespace total_chess_games_played_l222_222428

theorem total_chess_games_played : finset.card (finset.pairs_of_card 2 (finset.range 10)) = 45 := by
  sorry

end total_chess_games_played_l222_222428


namespace range_y_minus_2x_l222_222617

theorem range_y_minus_2x (x y : ℝ) (hx : -2 ≤ x ∧ x ≤ 1) (hy : 2 ≤ y ∧ y ≤ 4) :
  0 ≤ y - 2 * x ∧ y - 2 * x ≤ 8 :=
sorry

end range_y_minus_2x_l222_222617


namespace expansion_term_nine_coeff_term_containing_x3_coeff_diff_eq_162_term_with_x3_l222_222983

theorem expansion_term_nine_coeff {x : ℝ} (n : ℕ) (T3_coeff T2_coeff : ℝ) 
  (hT3 : T3_coeff = C(n, 2) * 4) 
  (hT2 : T2_coeff = C(n, 1) * (-2)) 
  (h_diff : T3_coeff = T2_coeff + 162) : n = 9 :=
by 
  -- The proof goes here
  sorry

theorem term_containing_x3 {x : ℝ} (n : ℕ) (r : ℕ)
  (h_n : n = 9) (h_r : r = 1) :
  ∃ T2_term : ℝ, T2_term = C(9, 1) * (-2) * x^3 :=
by 
  -- The proof goes here
  exists x = 3
  exact -18 * x^3


open_locale big_operators   -- If needed appropriate library for combinations C(n, r)

noncomputable theory
-- Note: combinatorics.C is used for binomial coefficients in some Lean versions, adjust as needed.
def C (n k : ℕ) : ℕ := nat.choose n k

/-- The theorem proving the first part of the problem. -/
theorem coeff_diff_eq_162 (n : ℕ) : 4 * (C n 2) = -2 * (C n 1) + 162 → n = 9 :=
sorry

/-- The theorem proving the term containing x^3 in the expansion. -/
theorem term_with_x3 (n : ℕ) : (∃ (T₂ : ℝ), T₂ = C n 1 * -2 * x^3) → n = 9 :=
 sorry

end expansion_term_nine_coeff_term_containing_x3_coeff_diff_eq_162_term_with_x3_l222_222983


namespace factorize_poly_part_a_factorize_poly_part_b_l222_222424

-- Part (a)
theorem factorize_poly_part_a : 
  (x : ℝ) → x^8 + x^4 + 1 = (x^4 + x^2 + 1) * (x^4 - x^2 + 1) :=
by
  sorry

-- Part (b)
theorem factorize_poly_part_b : 
  (x : ℝ) → x^8 + x^4 + 1 = (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + real.sqrt 3 * x + 1) * (x^2 - real.sqrt 3 * x + 1) :=
by
  sorry

end factorize_poly_part_a_factorize_poly_part_b_l222_222424


namespace coeff_x3_of_expansion_l222_222052

theorem coeff_x3_of_expansion : 
  (coeff (x_pow 3) (expand ((1 - (1/x)) * (1 + x)^5))) = 5 := 
sorry

end coeff_x3_of_expansion_l222_222052


namespace intersection_points_in_circle_l222_222221

open Classical

noncomputable def num_sides := [6, 7, 8, 9]

def intersection_points (n m : ℕ) : ℕ :=
  if n < m then 2 * n else 2 * m

def total_intersections : ℕ :=
  ∑ i in num_sides.toFinset, ∑ j in num_sides.toFinset, if i < j then intersection_points i j else 0

theorem intersection_points_in_circle : total_intersections = 80 := by
  sorry

end intersection_points_in_circle_l222_222221


namespace sum_of_digits_base_8_888_is_13_l222_222398

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222398


namespace emma_travel_time_l222_222478

noncomputable def emma_time (highway_length_miles : ℝ) (highway_width_feet : ℝ)
  (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
let highway_length_feet := highway_length_miles * mile_to_feet
let loop_diameter := highway_width_feet
let loop_radius := loop_diameter / 2
let number_of_loops := highway_length_feet / loop_diameter |> floor
let loop_circumference := 2 * Math.PI * loop_radius
let total_distance_feet := number_of_loops * loop_circumference
let total_distance_miles := total_distance_feet / mile_to_feet in
total_distance_miles / speed_mph

theorem emma_travel_time : emma_time 2 50 4 5280 = Math.PI / 2 :=
by
  sorry

end emma_travel_time_l222_222478


namespace molecular_weight_of_BaBr2_l222_222322

theorem molecular_weight_of_BaBr2 
    (atomic_weight_Ba : ℝ)
    (atomic_weight_Br : ℝ)
    (moles : ℝ)
    (hBa : atomic_weight_Ba = 137.33)
    (hBr : atomic_weight_Br = 79.90) 
    (hmol : moles = 8) :
    (atomic_weight_Ba + 2 * atomic_weight_Br) * moles = 2377.04 :=
by 
  sorry

end molecular_weight_of_BaBr2_l222_222322


namespace min_value_C_l222_222070

def x1 : ℝ := 2 + Real.sqrt 5
def x2 : ℝ := 2 - Real.sqrt 5

noncomputable def a (n : ℕ+) : ℤ :=
  Int.floor (x1^n + (1 / 2^n))

theorem min_value_C (C : ℝ) : C = 1 / 288 → 
  ∀ n : ℕ+, ∑ k in Finset.range n, ((1 : ℝ) / (a k * a (k + 2))) ≤ C :=
by
  sorry

end min_value_C_l222_222070


namespace func_equiv_set1_func_neq_set2_domain1_not_func_equiv_set2_l222_222889

variables {x : ℝ}

theorem func_equiv_set1 : 
  ∀ x ∈ set.Icc (-1 : ℝ) 1, (sqrt (1 - x^2) / abs (x + 2)) = (sqrt (1 - x^2) / (x + 2)) := 
by 
  sorry

theorem func_neq_set2_domain1 : 
  ∀ x ∈ set.Ici (2 : ℝ), sqrt (x - 1) * sqrt (x - 2) ≠ sqrt (x^2 - 3 * x + 2) := 
by 
  sorry

theorem not_func_equiv_set2 :
  ∃ x, (x ∈ set.Ici 2) ∧ (sqrt (x^2 - 3*x + 2)).nonpos ∧ sqrt (x - 1) * sqrt (x - 2) = sqrt (x^2 - 3*x + 2) ↔ false := 
by 
  sorry

end func_equiv_set1_func_neq_set2_domain1_not_func_equiv_set2_l222_222889


namespace opposite_of_half_l222_222770

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end opposite_of_half_l222_222770


namespace correct_conclusions_count_l222_222992

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem correct_conclusions_count :
  (∃ (a b c : ℝ), c = 0 ∧ (∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) 2 → 
  (f' x = 3 * x^2 + 2 * a * x + b) ∧ f' 1 = -1 ∧ f' (-1) = -1) 
  ∧ (f 0 = 0) ∧ f = (λ x, x^3 - 4 * x))
  → (set.count 
      { p | p = (f = (λ x, x^3 - 4 * x)) 
         ∨ (∃ e, e > 0 ∧ x ∈ set.Icc (-2 : ℝ) 2 → e = 1) 
         ∨ (∀ (x ∈ set.Icc (-2 : ℝ) 2), f x + f (-x) = 0)} = 2) :=
sorry

end correct_conclusions_count_l222_222992


namespace max_min_difference_abc_l222_222586

theorem max_min_difference_abc (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
    let M := 1
    let m := -1/2
    M - m = 3/2 :=
by
  sorry

end max_min_difference_abc_l222_222586


namespace arrangement_problem_l222_222116

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangement_problem 
  (p1 p2 p3 p4 p5 : Type)  -- Representing the five people
  (youngest : p1)         -- Specifying the youngest
  (oldest : p5)           -- Specifying the oldest
  (unique_people : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5) -- Ensuring five unique people
  : (factorial 5) - (factorial 4 * 2) = 72 :=
by sorry

end arrangement_problem_l222_222116


namespace rationalize_denominator_min_value_l222_222202

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222202


namespace find_a_l222_222714

def fibonacci : ℕ → ℕ
| 1       => 1
| 2       => 1
| (n + 3) => fibonacci (n + 1) + fibonacci (n + 2)

theorem find_a (a b d : ℕ) (h1 : fibonacci 1 = 1)
  (h2 : fibonacci 2 = 1)
  (h3 : ∀ n ≥ 3, fibonacci n = fibonacci (n - 1) + fibonacci (n - 2))
  (h4 : d = b + 2)
  (h5 : a + b + d = 1000)
  (h6 : fibonacci a < fibonacci b)
  (h7 : fibonacci b < fibonacci d)
  (h8 : 2 * fibonacci b = fibonacci a + fibonacci d) : a = 332 :=
by
  sorry

end find_a_l222_222714


namespace B_minus_A_pi_over_2_area_of_triangle_l222_222128

open Real

-- Definitions and conditions
variables (A B C a b : ℝ)

def is_triangle (A B C a b : ℝ) : Prop :=
B > π / 2 ∧ collinear ([cos A, b], [sin A, a]) 

-- Proof (1)
theorem B_minus_A_pi_over_2 (h_triangle : is_triangle A B C a b) : 
  B - A = π / 2 := sorry

-- Proof (2)
theorem area_of_triangle (h_triangle : is_triangle (π / 6) (2 * π / 3) (π / 6) 2 2.sqrt 3) :
  let a := 2 in let b := 2.sqrt 3 in 
  (1 / 2) * a * b * sin (π / 6) = sqrt 3 := sorry

end B_minus_A_pi_over_2_area_of_triangle_l222_222128


namespace decimal_to_base_five_l222_222916

theorem decimal_to_base_five (n : ℕ) (h : n = 175) : nat.to_digits 5 n = [1, 2, 0, 0] :=
by
  rw [h]
  sorry

end decimal_to_base_five_l222_222916


namespace sum_of_digits_base_8_888_is_13_l222_222395

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222395


namespace unit_digit_of_expression_is_4_l222_222285

theorem unit_digit_of_expression_is_4 :
  Nat.unitsDigit ((2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) - 1) = 4 :=
by
  sorry

end unit_digit_of_expression_is_4_l222_222285


namespace hyperbola_equation_l222_222583

theorem hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (eccentricity : Real.sqrt 2 = b / a)
  (line_through_FP_parallel_to_asymptote : ∃ c : ℝ, c = Real.sqrt 2 * a ∧ ∀ P : ℝ × ℝ, P = (0, 4) → (P.2 - 0) / (P.1 + c) = 1) :
  (∃ (a b : ℝ), a = b ∧ (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2) → 
  (∃ x y : ℝ, ((x^2 / 8) - (y^2 / 8) = 1)) :=
by
  sorry

end hyperbola_equation_l222_222583


namespace well_capacity_l222_222791

theorem well_capacity (rate1 rate2 time : ℕ) (h_rate1 : rate1 = 48) (h_rate2 : rate2 = 192) (h_time : time = 5) : (rate1 * time + rate2 * time) = 1200 :=
by
  have h1 : rate1 * time = 48 * 5, by rw [h_rate1, h_time]
  have h2 : rate2 * time = 192 * 5, by rw [h_rate2, h_time]
  have total : rate1 * time + rate2 * time = 48 * 5 + 192 * 5, by rw [h1, h2]
  -- Skipping the proof steps, which would simply involve calculating 48 * 5 + 192 * 5 calc
  -- 48 * 5 + 192 * 5 = 240 + 960 = 1200
  sorry

end well_capacity_l222_222791


namespace tetrahedron_altitudes_intersect_l222_222111

theorem tetrahedron_altitudes_intersect
  (A B C D : Point ℝ) 
  (h : ∃ (s : Sphere ℝ), 
       s.contains (midpoint A B) ∧ 
       s.contains (midpoint A C) ∧ 
       s.contains (midpoint A D) ∧ 
       s.contains (midpoint B C) ∧ 
       s.contains (midpoint B D) ∧ 
       s.contains (midpoint C D)) : 
  ∃ P : Point ℝ, is_intersection_point_of_all_altitudes A B C D P := 
sorry

end tetrahedron_altitudes_intersect_l222_222111


namespace cost_of_socks_l222_222521

theorem cost_of_socks (cost_shirt_no_discount cost_pants_no_discount cost_shirt_discounted cost_pants_discounted cost_socks_discounted total_savings team_size socks_cost_no_discount : ℝ) 
    (h1 : cost_shirt_no_discount = 7.5)
    (h2 : cost_pants_no_discount = 15)
    (h3 : cost_shirt_discounted = 6.75)
    (h4 : cost_pants_discounted = 13.5)
    (h5 : cost_socks_discounted = 3.75)
    (h6 : total_savings = 36)
    (h7 : team_size = 12)
    (h8 : 12 * (7.5 + 15 + socks_cost_no_discount) - 12 * (6.75 + 13.5 + 3.75) = 36)
    : socks_cost_no_discount = 4.5 :=
by
  sorry

end cost_of_socks_l222_222521


namespace vector_expression_result_l222_222590

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def vector_dot_product (v1 v2 : Vector2) : ℝ :=
  v1.x * v1.y + v2.x * v2.y

def vector_scalar_mul (c : ℝ) (v : Vector2) : Vector2 :=
  { x := c * v.x, y := c * v.y }

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

noncomputable def a : Vector2 := { x := 2, y := -1 }
noncomputable def b : Vector2 := { x := 3, y := -2 }

theorem vector_expression_result :
  vector_dot_product
    (vector_sub (vector_scalar_mul 3 a) b)
    (vector_sub a (vector_scalar_mul 2 b)) = -15 := by
  sorry

end vector_expression_result_l222_222590


namespace part1_part2_l222_222444

-- Part 1: Expression simplification
theorem part1 (a : ℝ) : (a - 3)^2 + a * (4 - a) = -2 * a + 9 := 
by
  sorry

-- Part 2: Solution set of inequalities
theorem part2 (x : ℝ) : 
  (3 * x - 5 < x + 1) ∧ (2 * (2 * x - 1) ≥ 3 * x - 4) ↔ (-2 ≤ x ∧ x < 3) := 
by
  sorry

end part1_part2_l222_222444


namespace sum_of_digits_base8_888_l222_222406

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222406


namespace rationalize_denominator_min_value_l222_222200

theorem rationalize_denominator_min_value
  (sqrt50 : ℝ := Real.sqrt 50)
  (sqrt25 : ℝ := Real.sqrt 25)
  (sqrt5 : ℝ := Real.sqrt 5)
  (A : ℤ := 5)
  (B : ℤ := 2)
  (C : ℤ := 1)
  (D : ℤ := 4)
  (expression : ℝ := 5 * √2 + √10)
  (denom : ℝ := 4)
  : (sqrt50 / (sqrt25 - sqrt5)) = (expression / denom) ∧ 
    (A + B + C + D) = 12 :=
by
  sorry

end rationalize_denominator_min_value_l222_222200


namespace yeast_counting_procedure_l222_222122

def yeast_counting_conditions (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool) : Prop :=
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true

theorem yeast_counting_procedure :
  ∀ (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool),
  yeast_counting_conditions counting_method shake_test_tube_needed dilution_needed →
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true :=
by
  intros counting_method shake_test_tube_needed dilution_needed h_condition
  exact h_condition

end yeast_counting_procedure_l222_222122


namespace limit_sub_sqrt_sequence_l222_222157

noncomputable def a : ℕ → ℝ
| 0            := 3
| 1            := 7
| (n + 2) := ((a (n + 1)) * (a (n + 1) - 2) + 4) / a n

theorem limit_sub_sqrt_sequence :
  (∀ sequence a,
    a 1 = 3 →
    a 2 = 7 →
    (∀ n ≥ 3, a (n-1) * (a (n-1) - 2) = a (n-2) * a n - 4) →
    (lim_{n → ∞} (sqrt (a n) - floor (sqrt (a n))) = 1/2)) :=
sorry

end limit_sub_sqrt_sequence_l222_222157


namespace min_fraction_value_l222_222585

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem min_fraction_value : ∀ x ∈ (Set.Ici (7 / 4)), (f x)^2 + 2 / (f x) ≥ 81 / 28 :=
by
  sorry

end min_fraction_value_l222_222585


namespace construction_days_behind_without_additional_workers_l222_222860

-- Definitions for initial and additional workers and their respective efficiencies and durations.
def initial_workers : ℕ := 100
def initial_worker_efficiency : ℕ := 1
def total_days : ℕ := 150

def additional_workers_1 : ℕ := 50
def additional_worker_efficiency_1 : ℕ := 2
def additional_worker_start_day_1 : ℕ := 30

def additional_workers_2 : ℕ := 25
def additional_worker_efficiency_2 : ℕ := 3
def additional_worker_start_day_2 : ℕ := 45

def additional_workers_3 : ℕ := 15
def additional_worker_efficiency_3 : ℕ := 4
def additional_worker_start_day_3 : ℕ := 75

-- Define the total additional work units done by the extra workers.
def total_additional_work_units : ℕ := 
  (additional_workers_1 * additional_worker_efficiency_1 * (total_days - additional_worker_start_day_1)) +
  (additional_workers_2 * additional_worker_efficiency_2 * (total_days - additional_worker_start_day_2)) +
  (additional_workers_3 * additional_worker_efficiency_3 * (total_days - additional_worker_start_day_3))

-- Define the days the initial workers would have taken to do the additional work.
def initial_days_for_additional_work : ℕ := 
  (total_additional_work_units + (initial_workers * initial_worker_efficiency) - 1) / (initial_workers * initial_worker_efficiency)

-- Define the total days behind schedule.
def days_behind_schedule : ℕ := (total_days + initial_days_for_additional_work) - total_days

-- Define the theorem to prove.
theorem construction_days_behind_without_additional_workers : days_behind_schedule = 244 := 
  by 
  -- This translates to manually verifying the outcome.
  -- A detailed proof can be added later.
  sorry

end construction_days_behind_without_additional_workers_l222_222860


namespace proof_find_abc_side_length_l222_222458

noncomputable def circle_radius (area : ℝ) : ℝ :=
  real.sqrt (area / real.pi)

noncomputable def oa_length : ℝ := 5

noncomputable def abc_side_length (o_radius oa_length : ℝ) : ℝ := 
  let s_sqrt_3_div_2 := (real.sqrt 3) / 2
  let result := real.sqrt (4 * ((o_radius ^ 2) - (oa_length ^ 2)) / (4 + 3))
  result

def find_abc_side_length : Prop :=
  let circle_area := 100 * real.pi
  let circle_radius := circle_radius circle_area
  let s := abc_side_length circle_radius oa_length
  s = 5

theorem proof_find_abc_side_length : find_abc_side_length := sorry

end proof_find_abc_side_length_l222_222458


namespace least_number_divisible_by_11_l222_222321

theorem least_number_divisible_by_11 (n : ℕ) (k : ℕ) (h₁ : n = 2520 * k + 1) (h₂ : 11 ∣ n) : n = 12601 :=
sorry

end least_number_divisible_by_11_l222_222321


namespace distance_between_vertices_l222_222926

-- Given a hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := 4 * x ^ 2 - 24 * x - y ^ 2 - 6 * y + 34 = 0

-- Statement to prove
theorem distance_between_vertices : 
  ∀ x y : ℝ, hyperbola_eq x y → True := -- Placeholder for actual property
begin
  sorry -- Proof to be filled in.
end

end distance_between_vertices_l222_222926


namespace polygon_sides_l222_222780

theorem polygon_sides (n : ℕ) : (n - 2) * 180 = 3 * 360 → n = 8 :=
by
  intros h,
  sorry

end polygon_sides_l222_222780


namespace correct_option_is_C_l222_222421

def is_linear_system (eq1 eq2 : Π {x y : ℝ}, Prop) : Prop :=
  ∃ a b c d e f : ℝ, eq1 = (λ x y, a*x + b*y = c) ∧ eq2 = (λ x y, d*x + e*y = f)

def optionA (x y : ℝ) : Prop := (x - y^2 = 5, x + 3*y = 16)
def optionB (x y z : ℝ) : Prop := (2*x + 3*z = 5, y - x = 2)
def optionC (x y : ℝ) : Prop := (x + y = 11, 5*x - 3*y = -7)
def optionD (x y : ℝ) : Prop := (2*x + y = 1, -x + 3/y = -3)

theorem correct_option_is_C : is_linear_system optionA.1 optionA.2 = false
  ∧ is_linear_system optionB.1 optionB.2 = false
  ∧ is_linear_system optionC.1 optionC.2
  ∧ is_linear_system optionD.1 optionD.2 = false := by
  sorry

end correct_option_is_C_l222_222421


namespace new_recipe_water_l222_222262

theorem new_recipe_water (flour water sugar : ℕ)
  (h_orig : flour = 10 ∧ water = 6 ∧ sugar = 3)
  (h_new : ∀ (new_flour new_water new_sugar : ℕ), 
            new_flour = 10 ∧ new_water = 3 ∧ new_sugar = 3)
  (h_sugar : sugar = 4) :
  new_water = 4 := 
  sorry

end new_recipe_water_l222_222262


namespace number_of_players_in_hockey_club_l222_222642

-- Defining the problem parameters
def cost_of_gloves : ℕ := 6
def cost_of_helmet := cost_of_gloves + 7
def total_cost_per_set := cost_of_gloves + cost_of_helmet
def total_cost_per_player := 2 * total_cost_per_set
def total_expenditure : ℕ := 3120

-- Defining the target number of players
def num_players : ℕ := total_expenditure / total_cost_per_player

theorem number_of_players_in_hockey_club : num_players = 82 := by
  sorry

end number_of_players_in_hockey_club_l222_222642


namespace area_of_trajectory_l222_222589

theorem area_of_trajectory (A B : ℝ × ℝ) (P : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (1, 0))
  (h : dist P A = real.sqrt 2 * dist P B) : 
  let S := set_of (λ P, dist P A = real.sqrt 2 * dist P B) in
  let trajectory := metric.sphere (3, 0) (2 * real.sqrt 2) in
  S = trajectory ∧ (π * (2 * real.sqrt 2) ^ 2) = 8 * π :=
by 
  sorry

end area_of_trajectory_l222_222589


namespace no_integral_solution_l222_222703

theorem no_integral_solution (n k m l : ℤ) (h_l : l ≥ 2) (h_k1 : 4 ≤ k) (h_k2 : k ≤ n - 4) :
  ¬ ∃ m : ℤ, (n.choose k) = m^l :=
by
  sorry

end no_integral_solution_l222_222703


namespace circle_equation_of_parabola_focus_l222_222236

theorem circle_equation_of_parabola_focus :
  let focus := (1, 0) in
  let radius := 1 in
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 -> x ^ 2 + y ^ 2 - 2 * x = 0) :=
begin
  sorry
end

end circle_equation_of_parabola_focus_l222_222236


namespace sum_of_base_8_digits_888_l222_222338

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222338


namespace average_of_c_and_d_l222_222735

variable (c d e : ℝ)

theorem average_of_c_and_d
  (h1: (4 + 6 + 9 + c + d + e) / 6 = 20)
  (h2: e = c + 6) :
  (c + d) / 2 = 47.5 := by
sorry

end average_of_c_and_d_l222_222735


namespace figure_100_squares_l222_222109

theorem figure_100_squares :
  (∃ (f : ℕ → ℕ), 
    f 0 = 1 ∧ 
    f 1 = 7 ∧ 
    f 2 = 19 ∧ 
    f 3 = 37 ∧ 
    ∀ n, f n = 3 * n ^ 2 + 3 * n + 1) →
  ∃ f, f 100 = 30301 :=
begin
  intro h,
  cases h with f h_f,
  use f,
  rw h_f.right.right.right,
  rw h_f.right.right.left,
  rw h_f.right.left,
  rw h_f.left,
  sorry
end

end figure_100_squares_l222_222109


namespace remainder_1560th_term_is_zero_l222_222511

def seq_term (n : ℕ) : ℕ := 
  let k := (int.natAbs (Int.floor (Real.sqrt (8 * n + 1)) - 1) / 2) in 
  k

theorem remainder_1560th_term_is_zero : (seq_term 1560) % 8 = 0 :=
by 
  sorry

end remainder_1560th_term_is_zero_l222_222511


namespace iced_coffee_days_per_week_l222_222161

theorem iced_coffee_days_per_week (x : ℕ) (h1 : 5 * 4 = 20)
  (h2 : 20 * 52 = 1040)
  (h3 : 2 * x = 2 * x)
  (h4 : 52 * (2 * x) = 104 * x)
  (h5 : 1040 + 104 * x = 1040 + 104 * x)
  (h6 : 1040 + 104 * x - 338 = 1040 + 104 * x - 338)
  (h7 : (0.75 : ℝ) * (1040 + 104 * x) = 780 + 78 * x) :
  x = 3 :=
by
  sorry

end iced_coffee_days_per_week_l222_222161


namespace proof_two_digit_number_l222_222485

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l222_222485


namespace mika_initial_stickers_l222_222679

theorem mika_initial_stickers :
  let store_stickers := 26.0
  let birthday_stickers := 20.0 
  let sister_stickers := 6.0 
  let mother_stickers := 58.0 
  let total_stickers := 130.0 
  ∃ x : Real, x + store_stickers + birthday_stickers + sister_stickers + mother_stickers = total_stickers ∧ x = 20.0 := 
by 
  sorry

end mika_initial_stickers_l222_222679


namespace determinant_zero_l222_222144

-- Define the polynomial whose roots are a, b, and c.
variables {a b c s p q : ℂ}

-- Conditions given by Vieta's formulas.
axiom root1 : a^3 - s * a^2 + p * a + q = 0
axiom root2 : b^3 - s * b^2 + p * b + q = 0
axiom root3 : c^3 - s * c^2 + p * c + q = 0

-- Define the determinant of the given matrix.
def determinant (a b c : ℂ) : ℂ :=
  a * det!!(a b,
            c a,
            b c) - b * det!!(c b, 
                             b a) + c * det!!(c a,
                                              b c)

-- Prove that the determinant is 0 given the conditions.
theorem determinant_zero (ha : a^3 - s * a^2 + p * a + q = 0)
                         (hb : b^3 - s * b^2 + p * b + q = 0)
                         (hc : c^3 - s * c^2 + p * c + q = 0) :
  determinant a b c = 0 :=
sorry

end determinant_zero_l222_222144


namespace option_B_incorrect_l222_222071

noncomputable def line := Type
noncomputable def plane := Type

variables (a b c : line) (alpha beta : plane)

-- Conditions
axiom lines_are_different : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom planes_are_different : alpha ≠ beta 
axiom b_in_alpha : b ⊂ alpha
axiom c_not_in_alpha : ¬ c ⊂ alpha

-- Question:
-- Which of the following statements is incorrect?
-- Option B: The converse of "If b ⊥ beta, then alpha ⊥ beta"
theorem option_B_incorrect : ¬ (∀ (b beta : line) (alpha beta : plane), (β ⊥ b → β ⊥ alpha) → (β ⊥ alpha → β ⊥ b)) :=
sorry

end option_B_incorrect_l222_222071


namespace more_crayons_given_to_Lea_than_Mae_l222_222684

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end more_crayons_given_to_Lea_than_Mae_l222_222684


namespace A_lent_5000_to_B_l222_222856

noncomputable def principalAmountB
    (P_C : ℝ)
    (r : ℝ)
    (total_interest : ℝ)
    (P_B : ℝ) : Prop :=
  let I_B := P_B * r * 2
  let I_C := P_C * r * 4
  I_B + I_C = total_interest

theorem A_lent_5000_to_B :
  principalAmountB 3000 0.10 2200 5000 :=
by
  sorry

end A_lent_5000_to_B_l222_222856


namespace analytical_expression_and_intervals_of_increase_l222_222054

def f (x : ℝ) (ω : ℝ) : ℝ := 4 * cos (π / 3 - ω * x) * cos (ω * x) - 1

theorem analytical_expression_and_intervals_of_increase (ω : ℝ) (hω : ω > 0) : 
  (∀ x, f x ω = 2 * sin (2 * x + π / 6)) ∧ 
  (by interval_cases (x ∈ set.interval (0:ℝ) (2*π), 
  { exact x ∈ set.interval (0:ℝ) (π / 6) ∨ x ∈ set.Ioo (2 * π / 3) (7 * π / 6) ∨ x ∈ set.Ioo (5 * π / 3) (2 * π) })) := 
sorry

end analytical_expression_and_intervals_of_increase_l222_222054


namespace min_distance_origin_to_line_l222_222044

theorem min_distance_origin_to_line (a b : ℝ) (h : a + 2 * b = Real.sqrt 5) : 
  Real.sqrt (a^2 + b^2) ≥ 1 :=
sorry

end min_distance_origin_to_line_l222_222044


namespace polynomial_factor_sum_abs_l222_222139

theorem polynomial_factor_sum_abs :
  let S := ∑ b in { b : ℤ | ∃ r s : ℤ, r + s = -b ∧ r * s = 2008 * b }, b
  |S| = 88352 :=
sorry

end polynomial_factor_sum_abs_l222_222139


namespace find_k_l222_222425

-- Define the conditions and the question
theorem find_k (t k : ℝ) (h1 : t = 50) (h2 : t = (5 / 9) * (k - 32)) : k = 122 := by
  -- Proof will go here
  sorry

end find_k_l222_222425


namespace sum_of_digits_base8_888_l222_222374

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222374


namespace minimize_distance_l222_222722

-- Definitions of points and distances
structure Point where
  x : ℝ
  y : ℝ

def distanceSquared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition points A, B, and C
def A := Point.mk 7 3
def B := Point.mk 3 0

-- Mathematical problem: Find the value of k that minimizes the sum of distances squared
theorem minimize_distance : ∃ k : ℝ, ∀ k', 
  (distanceSquared A (Point.mk 0 k) + distanceSquared B (Point.mk 0 k) ≤ 
   distanceSquared A (Point.mk 0 k') + distanceSquared B (Point.mk 0 k')) → 
  k = 3 / 2 :=
by
  sorry

end minimize_distance_l222_222722


namespace smallest_positive_period_max_value_on_interval_l222_222580

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + (sqrt 3 / 2) * cos (2 * x) + 1 / 2

theorem smallest_positive_period : ∀ (x : ℝ), f (x + π) = f x :=
sorry

theorem max_value_on_interval : ∀ x ∈ Icc 0 (π / 4), f x ≤ 3 / 2 :=
sorry

end smallest_positive_period_max_value_on_interval_l222_222580


namespace measure_angle_BAC_l222_222652

theorem measure_angle_BAC (A B C X Y : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] [MetricSpace Y]
  (AX XY YB BC : ℝ) (h1 : AX = XY) (h2 : XY = YB) (h3 : YB = BC) (h4 : ∠ BAC = 150) : ∠ BAC = 7.5 :=
by
  sorry

end measure_angle_BAC_l222_222652


namespace fx_le_x2_l222_222434

theorem fx_le_x2 {f : ℝ → ℝ} (M : ℝ) (h1 : ∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2))
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs(f(x)) ≤ M) :
  ∀ x : ℝ, 0 ≤ x → f(x) ≤ x^2 :=
sorry

end fx_le_x2_l222_222434


namespace base6_to_base10_product_zero_l222_222229

theorem base6_to_base10_product_zero
  (c d e : ℕ)
  (h : (5 * 6^2 + 3 * 6^1 + 2 * 6^0) = (100 * c + 10 * d + e)) :
  (c * e) / 10 = 0 :=
by
  sorry

end base6_to_base10_product_zero_l222_222229


namespace maximum_value_inequality_l222_222073

noncomputable def f : ℝ → ℝ :=
  λ x => x^2 - 4 * x + 8

theorem maximum_value_inequality (a : ℝ) (h : a ∈ Icc (1 : ℝ) a) (h_max : ∀ x ∈ Icc (1 : ℝ) a, f x ≤ f a) : 3 ≤ a :=
by
  sorry

end maximum_value_inequality_l222_222073


namespace mia_and_mom_toys_solution_l222_222163

def mia_and_mom_toys_problem : Prop :=
  ∀ (T_total : ℕ) (R_mom R_mia : ℕ),
    T_total = 50 →
    R_mom = 4 →
    R_mia = 3 →
    let cycle_time := 45 in
    let net_gain := R_mom - R_mia in
    let cycles := 47 in
    let last_cycle_time := cycle_time in
    let total_time := (cycles * cycle_time) + last_cycle_time in
    (total_time / 60 = 36)

theorem mia_and_mom_toys_solution : mia_and_mom_toys_problem :=
begin
  sorry
end

end mia_and_mom_toys_solution_l222_222163


namespace coefficient_of_x_squared_is_one_l222_222614

theorem coefficient_of_x_squared_is_one (a b c d : ℚ) (h : |a| + |b| + |c| + |d| = 12) :
  (x y : ℚ) (h_eq : (x^2 + x - 12) = (a*x + b) * (c*x + d)) → 
  ∃ k, k = 1 :=
by
  -- proof omitted
  sorry

end coefficient_of_x_squared_is_one_l222_222614


namespace maximum_pieces_is_seven_l222_222255

noncomputable def max_pieces (PIE PIECE : ℕ) (n : ℕ) : Prop :=
  PIE = PIECE * n ∧ natDigits 10 PIE = List.nodup (natDigits 10 PIE) ∧ natDigits 10 PIECE = List.nodup (natDigits 10 PIECE)

theorem maximum_pieces_is_seven :
  max_pieces 95207 13601 7 :=
sorry

end maximum_pieces_is_seven_l222_222255


namespace change_in_mean_and_median_l222_222662

def initial_participants : List ℕ := [15, 18, 14, 20, 14]
def correct_participants : List ℕ := [15, 23, 14, 20, 14]

def mean (l : List ℕ) : ℚ :=
  l.sum / l.length

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem change_in_mean_and_median :
  mean correct_participants - mean initial_participants = 1 ∧
  median correct_participants = median initial_participants :=
by
  sorry

end change_in_mean_and_median_l222_222662


namespace algebraic_expression_value_l222_222959

-- Define the premises as a Lean statement
theorem algebraic_expression_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a * (b + c) + b * (a + c) + c * (a + b) = -1 :=
sorry

end algebraic_expression_value_l222_222959


namespace minimum_line_segments_l222_222669

noncomputable def find_min_line_segments (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

theorem minimum_line_segments (n : ℕ)
  (h₁ : ∀ (i j k l : ℕ), 
          i < j ∧ j < k ∧ k < l ∧ l < n → 
          ∃ (a b c : ℕ), 
            a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
            ({a, b, c} ⊆ {i, j, k, l}) ∧ 
            (a, b) < n ∧ (b, c) < n ∧ (c, a) < n) :
  find_min_line_segments n = (n - 1) * (n - 2) / 2 :=
sorry

end minimum_line_segments_l222_222669


namespace coefficient_of_x3_in_binomial_expansion_l222_222720

theorem coefficient_of_x3_in_binomial_expansion :
  (Polynomial.expand (Polynomial.X - Polynomial.C (1 / Polynomial.X^2)) 6).coeff 3 = -6 :=
by
  sorry

end coefficient_of_x3_in_binomial_expansion_l222_222720


namespace total_stamps_collected_l222_222075

-- Conditions
def harry_stamps : ℕ := 180
def sister_stamps : ℕ := 60
def harry_three_times_sister : harry_stamps = 3 * sister_stamps := 
  by
  sorry  -- Proof will show that 180 = 3 * 60 (provided for completeness)

-- Statement to prove
theorem total_stamps_collected : harry_stamps + sister_stamps = 240 :=
  by
  sorry

end total_stamps_collected_l222_222075


namespace avg_gas_mileage_correct_l222_222872

noncomputable def total_distance : ℝ :=
  150 + 50 + 150

noncomputable def gasoline_sedan : ℝ :=
  150 / 25

noncomputable def gasoline_suv : ℝ :=
  50 / 15

noncomputable def gasoline_hybrid : ℝ :=
  150 / 50

noncomputable def total_gasoline_used : ℝ :=
  gasoline_sedan + gasoline_suv + gasoline_hybrid

noncomputable def avg_gas_mileage (total_distance : ℝ) (total_gasoline_used : ℝ) : ℝ :=
  total_distance / total_gasoline_used

theorem avg_gas_mileage_correct :
  avg_gas_mileage total_distance total_gasoline_used ≈ 28.38 :=
sorry

end avg_gas_mileage_correct_l222_222872


namespace spherical_coordinates_of_point_l222_222917

def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := Real.atan2 y x
  let φ := Real.acos (z / ρ)
  (ρ, θ, φ)

theorem spherical_coordinates_of_point :
  rectangular_to_spherical 4 (-4 * Real.sqrt 3) (-4) = (4 * Real.sqrt 5, 4 * Real.pi / 3, 2 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_of_point_l222_222917


namespace series_converges_to_one_l222_222927

noncomputable def infinite_series := ∑' n, (3^n) / (3^(2^n) + 2)

theorem series_converges_to_one :
  infinite_series = 1 := by
  sorry

end series_converges_to_one_l222_222927


namespace ants_in_park_l222_222864

theorem ants_in_park:
  let width_meters := 100
  let length_meters := 130
  let cm_per_meter := 100
  let ants_per_sq_cm := 1.2
  let width_cm := width_meters * cm_per_meter
  let length_cm := length_meters * cm_per_meter
  let area_sq_cm := width_cm * length_cm
  let total_ants := ants_per_sq_cm * area_sq_cm
  total_ants = 156000000 := by
  sorry

end ants_in_park_l222_222864


namespace max_pieces_is_seven_l222_222256

-- Define what it means for a number to have all distinct digits
def all_digits_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (digits.nodup)

-- Define the main proof problem
theorem max_pieces_is_seven :
  ∃ (n : ℕ) (PIE : ℕ) (PIECE : ℕ),
  (PIE = PIECE * n) ∧
  (PIE >= 10000) ∧ (PIE < 100000) ∧
  all_digits_distinct PIE ∧
  all_digits_distinct PIECE ∧
  ∀ m, (m > n) → (¬ (∃ P' PIECE', (P' = PIECE' * m) ∧
   (P' >= 10000) ∧ (P' < 100000) ∧ all_digits_distinct P' ∧ all_digits_distinct PIECE'))
:= sorry

end max_pieces_is_seven_l222_222256


namespace sum_of_digits_base8_888_l222_222407

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  (let digits := [1, 5, 7, 0] in 
    digits.sum = 13) := by
  sorry

end sum_of_digits_base8_888_l222_222407


namespace trig_identity_evaluation_l222_222929

theorem trig_identity_evaluation :
  let θ1 := 70 * Real.pi / 180 -- angle 70 degrees in radians
  let θ2 := 10 * Real.pi / 180 -- angle 10 degrees in radians
  let θ3 := 20 * Real.pi / 180 -- angle 20 degrees in radians
  (Real.tan θ1 * Real.cos θ2 * (Real.sqrt 3 * Real.tan θ3 - 1) = -1) := 
by 
  sorry

end trig_identity_evaluation_l222_222929


namespace range_of_c_l222_222042

def P (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → (c ^ x1) > (c ^ x2)
def q (c : ℝ) : Prop := ∀ x : ℝ, x > (1 / 2) → (2 * c * x - c) > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1)
  (h3 : ¬ (P c ∧ q c)) (h4 : (P c ∨ q c)) :
  (1 / 2) < c ∧ c < 1 :=
by
  sorry

end range_of_c_l222_222042


namespace part1_C_value_part2_c_value_l222_222653

-- Definitions used in Lean 4 statement based on the conditions given in the problem
variables {a b c A B C : ℝ}
variables (h1 : c * sin C - a * sin A = (b - a) * sin B)
variables (h2 : b = 4)
variables (area : ℝ)
def triangle_area : ℝ := 6 * sqrt 3

-- The results we need to prove
theorem part1_C_value (h_area : area = triangle_area) : C = π / 3 :=
sorry

theorem part2_c_value (h_area : area = triangle_area) : c = 2 * sqrt 7 :=
sorry

end part1_C_value_part2_c_value_l222_222653


namespace sum_of_base_8_digits_888_l222_222340

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end sum_of_base_8_digits_888_l222_222340


namespace sum_of_ages_l222_222661

-- Define ages of Kiana and her twin brothers
variables (kiana_age : ℕ) (twin_age : ℕ)

-- Define conditions
def age_product_condition : Prop := twin_age * twin_age * kiana_age = 162
def age_less_than_condition : Prop := kiana_age < 10
def twins_older_condition : Prop := twin_age > kiana_age

-- The main problem statement
theorem sum_of_ages (h1 : age_product_condition twin_age kiana_age) (h2 : age_less_than_condition kiana_age) (h3 : twins_older_condition twin_age kiana_age) :
  twin_age * 2 + kiana_age = 20 :=
sorry

end sum_of_ages_l222_222661


namespace trip_first_part_distance_l222_222463

theorem trip_first_part_distance (x : ℝ) :
  let total_distance : ℝ := 60
  let speed_first : ℝ := 48
  let speed_remaining : ℝ := 24
  let avg_speed : ℝ := 32
  (x / speed_first + (total_distance - x) / speed_remaining = total_distance / avg_speed) ↔ (x = 30) :=
by sorry

end trip_first_part_distance_l222_222463


namespace correct_equation_l222_222418

theorem correct_equation (A B C D : Prop) : 
  A ↔ (sqrt 3) ^ 2 = 3 ∧
  B ↔ sqrt ((-3) ^ 2) = -3 ∧
  C ↔ sqrt (3 ^ 3) = 3 ∧
  D ↔ (-sqrt 3) ^ 2 = -3 ∧
  A ∧ ¬B ∧ ¬C ∧ ¬D :=
by {sorry}

end correct_equation_l222_222418


namespace largest_B_is_9_l222_222648

def is_divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_divisible_by_four (n : ℕ) : Prop :=
  n % 4 = 0

def largest_B_divisible_by_3_and_4 (B : ℕ) : Prop :=
  is_divisible_by_three (21 + B) ∧ is_divisible_by_four 32

theorem largest_B_is_9 : largest_B_divisible_by_3_and_4 9 :=
by
  have h1 : is_divisible_by_three (21 + 9) := by sorry
  have h2 : is_divisible_by_four 32 := by sorry
  exact ⟨h1, h2⟩

end largest_B_is_9_l222_222648


namespace doctor_visit_cost_l222_222789

theorem doctor_visit_cost (cast_cost : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) (visit_cost : ℝ) :
  cast_cost = 200 → insurance_coverage = 0.60 → out_of_pocket = 200 → 0.40 * (visit_cost + cast_cost) = out_of_pocket → visit_cost = 300 :=
by
  intros h_cast h_insurance h_out_of_pocket h_equation
  sorry

end doctor_visit_cost_l222_222789


namespace continuity_at_x0_l222_222436

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end continuity_at_x0_l222_222436


namespace range_not_includes_3_l222_222535

theorem range_not_includes_3 (b c : ℝ) (h : 3 ∈ Set.range (λ x : ℝ, x^2 - b*x + c) → 3 ∈ Set.range (λ x : ℝ, x^2 - b*x + c)) :
  c >= 3 ∧ (-real.sqrt (4 * c - 12) < b ∧ b < real.sqrt (4 * c - 12)) :=
sorry

end range_not_includes_3_l222_222535


namespace smaller_integer_of_two_digits_l222_222005

theorem smaller_integer_of_two_digits (a b : ℕ) (ha : 10 ≤ a ∧ a ≤ 99) (hb: 10 ≤ b ∧ b ≤ 99) (h_diff : a ≠ b)
  (h_eq : (a + b) / 2 = a + b / 100) : a = 49 ∨ b = 49 := 
by
  sorry

end smaller_integer_of_two_digits_l222_222005


namespace median_length_of_right_triangle_l222_222640

theorem median_length_of_right_triangle (DE EF : ℝ) (hDE : DE = 5) (hEF : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let N := (EF / 2)
  let DN := DF / 2
  DN = 6.5 :=
by
  sorry

end median_length_of_right_triangle_l222_222640


namespace left_focus_of_ellipse_l222_222723

open Real

-- Define the conditions: parameterization of the ellipse
def ellipse_x (θ : ℝ) : ℝ := 4 * cos θ
def ellipse_y (θ : ℝ) : ℝ := 3 * sin θ

-- Define the standard form of the ellipse
def ellipse_standard (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Prove that the coordinates of the left focus of the ellipse are (- √7, 0)
theorem left_focus_of_ellipse : ∃ (x y : ℝ), ellipse_standard x y ∧ (x = - √7) ∧ (y = 0) :=
by
  -- skip proof
  sorry

end left_focus_of_ellipse_l222_222723


namespace opposite_of_neg_half_l222_222755

theorem opposite_of_neg_half : ∃ x : ℚ, -1/2 + x = 0 ∧ x = 1/2 :=
by {
  use 1/2,
  split,
  { norm_num },
  { refl }
}

end opposite_of_neg_half_l222_222755


namespace john_walks_farther_than_nina_l222_222658

theorem john_walks_farther_than_nina (john_distance nina_distance : ℝ) (h_john : john_distance = 0.7) (h_nina : nina_distance = 0.4) : john_distance - nina_distance = 0.3 :=
by
  intros,
  rw [h_john, h_nina],
  norm_num

end john_walks_farther_than_nina_l222_222658


namespace problem_correctness_l222_222563

theorem problem_correctness (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  (m = 2 ∨ m = -2) ∧ (m^2 + (a + b) / 2 + (- (x * y)) ^ 2023 = 3) := 
by
  sorry

end problem_correctness_l222_222563


namespace determinant_formula_l222_222152

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V) (t : ℝ) (D : ℝ)
noncomputable def D_t : ℝ := (a ∙ (b + t • d) × c)

theorem determinant_formula :
  det ![(a × (b + t • d)), (b × c), (c × a)] = D_t * D :=
sorry

end determinant_formula_l222_222152


namespace unit_digit_of_product_is_4_l222_222282

theorem unit_digit_of_product_is_4 :
  let expr := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1 in
  expr % 10 = 4 :=
by
  -- define the expression 
  let expr : ℕ := ((2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)) - 1
  -- ensure the equivalence of unit digit
  show expr % 10 = 4
  sorry -- proof goes here

end unit_digit_of_product_is_4_l222_222282


namespace simplify_trig1_is_minus_one_simplify_trig2_is_two_l222_222704

noncomputable def simplify_trig1 : Real :=
  (sin (35 * Real.pi / 180))^2 - 1 / 2 / (cos (10 * Real.pi / 180) * cos (80 * Real.pi / 180))

noncomputable def simplify_trig2 (α : Real) : Real :=
  ((1 / tan (α / 2)) - (tan (α / 2))) * (1 - cos (2 * α)) / (sin (2 * α))

theorem simplify_trig1_is_minus_one : simplify_trig1 = -1 := 
  by
  sorry

theorem simplify_trig2_is_two (α : Real) : simplify_trig2 α = 2 := 
  by
  sorry

end simplify_trig1_is_minus_one_simplify_trig2_is_two_l222_222704


namespace leon_ordered_gaming_chairs_l222_222665

variable (G : ℕ)

-- Conditions from the problem
def toy_organizers_cost : ℤ := 3 * 78
def gaming_chairs_cost : ℤ := 83 * G
def total_cost_before_fee : ℤ := toy_organizers_cost + gaming_chairs_cost
def delivery_fee : ℤ := (0.05 * ↑total_cost_before_fee : ℤ)  -- need type casting as ℤ
def total_cost : ℤ := total_cost_before_fee + delivery_fee

-- Given total payment
def payment : ℤ := 420

-- The theorem we need to prove
theorem leon_ordered_gaming_chairs (h : total_cost = payment) : G = 2 := by
  sorry

end leon_ordered_gaming_chairs_l222_222665


namespace number_of_boys_in_biology_class_l222_222301

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l222_222301


namespace num_integer_sided_triangles_with_perimeter_15_eq_7_l222_222615

def isValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def hasPerimeter15 (a b c : ℕ) : Prop :=
  a + b + c = 15

theorem num_integer_sided_triangles_with_perimeter_15_eq_7 :
  { (a, b, c) : ℕ × ℕ × ℕ // a ≤ b ∧ b ≤ c ∧ hasPerimeter15 a b c ∧ isValidTriangle a b c }.to_finset.card = 7 :=
by
  sorry

end num_integer_sided_triangles_with_perimeter_15_eq_7_l222_222615


namespace cannot_tile_with_tetrominoes_l222_222907

def can_be_tiled (board : Fin 6 × Fin 6 → ℕ) : Prop :=
  ∀ tile : Fin 4 → Fin 6 × Fin 6,
  (∀ i j, tile 0 = (i, j) → tile 1 = (i + 1, j) ∨ tile 1 = (i - 1, j) ∨ tile 1 = (i, j + 1) ∨ tile 1 = (i, j - 1)) →
  (∀ i j k l, tile 0 ≠ tile 1 ∧ tile 1 ≠ tile 2 ∧ tile 2 ≠ tile 3) →
  ∑ i, board (tile i) = 6 * 6

theorem cannot_tile_with_tetrominoes :
  ∀ board : Fin 6 × Fin 6 → ℕ, (∀ i j : Fin 6, board (i, j) = (i + j) % 4) →
  ¬ can_be_tiled board :=
by {
  sorry
}

end cannot_tile_with_tetrominoes_l222_222907


namespace concert_duration_is_805_l222_222863

def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

def total_duration (hours : ℕ) (extra_minutes : ℕ) : ℕ :=
  hours_to_minutes hours + extra_minutes

theorem concert_duration_is_805 : total_duration 13 25 = 805 :=
by
  -- Proof skipped
  sorry

end concert_duration_is_805_l222_222863


namespace min_value_of_reciprocal_sum_l222_222019

theorem min_value_of_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : Float.ofReal (Real.sqrt (2^a * 2^b)) = Float.ofReal (Real.sqrt 2)) : 
  minValue (λ x, x = (1/a + 1/b)) = 4 :=
by
  -- Proof omitted
  sorry

end min_value_of_reciprocal_sum_l222_222019


namespace no_pos_reals_floor_prime_l222_222896

open Real
open Nat

theorem no_pos_reals_floor_prime : 
  ∀ (a b : ℝ), (0 < a) → (0 < b) → ∃ n : ℕ, ¬ Prime (⌊a * n + b⌋) :=
by
  intro a b a_pos b_pos
  sorry

end no_pos_reals_floor_prime_l222_222896


namespace petya_oranges_l222_222692

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end petya_oranges_l222_222692


namespace lying_dwarf_possible_numbers_l222_222431

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l222_222431


namespace belfried_total_payroll_correct_l222_222476

-- Total payroll to be determined
def total_payroll_belfried (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_threshold : ℝ) := 
  tax_free_threshold + tax_paid / tax_rate

-- Define constants as given in the problem
def tax_free_threshold : ℝ := 200000
def tax_rate : ℝ := 0.002
def tax_paid : ℝ := 200

-- Prove the total payroll is equal to the given solution
theorem belfried_total_payroll_correct : 
  total_payroll_belfried tax_paid tax_rate tax_free_threshold = 300000 := by
  sorry

end belfried_total_payroll_correct_l222_222476


namespace petya_oranges_l222_222693

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end petya_oranges_l222_222693


namespace parallel_vectors_dot_product_l222_222650

variables (α : ℝ)

noncomputable def vector_a := (Real.cos α, Real.sin α)
noncomputable def vector_b := (Real.sin (α + π/6), Real.cos (α + π/6))

theorem parallel_vectors (h : (Real.cos α * Real.cos (α + π/6) - Real.sin α * Real.sin (α + π/6) = 0)) (hα : 0 < α ∧ α < π/2) :
  α = π / 6 := 
sorry

theorem dot_product (h : Real.tan (2 * α) = -1 / 7) (hα : 0 < α ∧ α < π/2) :
  (vector_a α).fst * (vector_b α).fst + (vector_a α).snd * (vector_b α).snd = (Real.sqrt 6 - 7 * Real.sqrt 2) / 20 := 
sorry

end parallel_vectors_dot_product_l222_222650


namespace max_area_of_triangle_triangle_can_be_right_angled_perimeter_when_A_equals_2C_area_of_triangle_AOB_when_A_equals_2C_l222_222555

namespace TriangleProofs

open Real

variables {A B C : ℝ} {a b c : ℝ} {O : Point} (h1 : a = 6) 
(h2 : 4 * sin B = 5 * sin C) (h3 : A = 2 * C)

-- Statement ①: The maximum area of triangle ABC is 40
theorem max_area_of_triangle (h1 : a = 6) (h2 : 4 * sin B = 5 * sin C) : 
  (1/2) * a * c * sin B = 40 := sorry

-- Statement ②: Triangle ABC can be a right-angled triangle
theorem triangle_can_be_right_angled (h1 : a = 6) (h2 : 4 * sin B = 5 * sin C) : 
  ∃ (B : ℝ), B = π / 2 := sorry

-- Statement ③: When A = 2C, the perimeter of triangle ABC is 15
theorem perimeter_when_A_equals_2C (h1 : a = 6) (h2 : 4 * sin B = 5 * sin C)
  (h3 : A = 2 * C) : a + b + c = 15 := sorry

-- Statement ④: When A = 2C, if O is the incenter of triangle ABC
-- the area of triangle AOB is sqrt(7)
theorem area_of_triangle_AOB_when_A_equals_2C (h1 : a = 6) 
(h2 : 4 * sin B = 5 * sin C) (h3 : A = 2 * C)
(hO : incenter O A B C) : 
  (1/2) * A * c * inradius O A B C = sqrt 7 := sorry

end TriangleProofs

end max_area_of_triangle_triangle_can_be_right_angled_perimeter_when_A_equals_2C_area_of_triangle_AOB_when_A_equals_2C_l222_222555


namespace sum_of_digits_base_8_888_is_13_l222_222396

noncomputable def sum_of_digits_base_8_rep_of_888 : ℕ :=
  let n : ℕ := 888
  let base_8_rep : ℕ := 1 * 8^3 + 5 * 8^2 + 7 * 8^1 + 0 * 8^0
  in 1 + 5 + 7 + 0

theorem sum_of_digits_base_8_888_is_13 : sum_of_digits_base_8_rep_of_888 = 13 :=
  sorry

end sum_of_digits_base_8_888_is_13_l222_222396


namespace find_b_plus_c_l222_222539

theorem find_b_plus_c (a b c d : ℝ) 
    (h₁ : a + d = 6) 
    (h₂ : a * b + a * c + b * d + c * d = 40) : 
    b + c = 20 / 3 := 
sorry

end find_b_plus_c_l222_222539


namespace angle_relationship_l222_222112

variables {A B C H L D : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited H] [inhabited L] [inhabited D]
variables (α β γ δ : ℝ) -- Angles in radians or degrees depending on context

def is_triangle (ABC : Type) : Prop :=
  ∃ (A B C : Type) (α β γ : ℝ), α + β + γ = π

def altitude (A H : Type) (BC : Type) : Prop :=
  ∃ (H : Type), -- Altitude is perpendicular from A to BC
  true

def angle_bisector (A L : Type) (BC : Type) : Prop :=
  ∃ (L : Type), -- L bisects the angle at A
  true

def median (A D : Type) (BC : Type) : Prop :=
  ∃ (D : Type), -- D is the midpoint of BC
  true

def angles (A H L D : Type) (α β γ δ : ℝ) : Prop :=
  (α = β ↔ γ = δ)

theorem angle_relationship
  (ABC : Type) [is_triangle ABC]
  (A H L D : Type)
  (α β γ δ : ℝ)
  (h₁ : α + β = γ + δ)
  (h₂ : altitude A H ABC)
  (h₃: angle_bisector A L ABC)
  (h₄: median A D ABC) :
  angles A H L D α β γ δ ↔ α = π / 2 :=
sorry

end angle_relationship_l222_222112


namespace find_f_2018_l222_222559

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_functional_eq : ∀ x : ℝ, f x = - (1 / f (x + 3))
axiom f_at_4 : f 4 = -2018

theorem find_f_2018 : f 2018 = -2018 :=
  sorry

end find_f_2018_l222_222559


namespace distance_formula_example_l222_222805

variable (x1 y1 x2 y2 : ℝ)

theorem distance_formula_example : dist (3, -1) (-4, 3) = Real.sqrt 65 :=
by
  let x1 := 3
  let y1 := -1
  let x2 := -4
  let y2 := 3
  sorry

end distance_formula_example_l222_222805


namespace trajectory_equation_minimum_AB_l222_222967

/-- Let a moving circle \( C \) passes through the point \( F(0, 1) \).
    The center of the circle \( C \), denoted as \( (x, y) \), is above the \( x \)-axis and the
    distance from \( (x, y) \) to \( F \) is greater than its distance to the \( x \)-axis by 1.
    We aim to prove that the trajectory of the center is \( x^2 = 4y \). -/
theorem trajectory_equation {x y : ℝ} (h : y > 0) (hCF : Real.sqrt (x^2 + (y - 1)^2) - y = 1) : 
  x^2 = 4 * y :=
sorry

/-- Suppose \( A \) and \( B \) are two distinct points on the curve \( x^2 = 4y \). 
    The tangents at \( A \) and \( B \) intersect at \( P \), and \( AP \perp BP \). 
    Then the minimum value of \( |AB| \) is 4. -/
theorem minimum_AB {x₁ x₂ : ℝ} 
  (h₁ : y₁ = (x₁^2) / 4) (h₂ : y₂ = (x₂^2) / 4)
  (h_perp : x₁ * x₂ = -4) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d = 4 :=
sorry

end trajectory_equation_minimum_AB_l222_222967


namespace probability_of_break_in_first_50_meters_l222_222423

theorem probability_of_break_in_first_50_meters (total_length favorable_length : ℝ) 
  (h_total_length : total_length = 320) 
  (h_favorable_length : favorable_length = 50) : 
  (favorable_length / total_length) = 0.15625 := 
sorry

end probability_of_break_in_first_50_meters_l222_222423


namespace sum_of_digits_base8_888_l222_222384

/-- Definition of the function to convert a number to a specified base -/
def convert_to_base (n : ℕ) (b : ℕ) : list ℕ := 
  if b ≤ 1 then [n] 
  else let rec aux (n: ℕ) (acc: list ℕ) : list ℕ :=
    if n = 0 then acc 
    else aux (n / b) ((n % b) :: acc) 
  in aux n []

/-- Definition of the function to sum the digits in a list -/
def sum_of_digits (digits : list ℕ) : ℕ :=
  digits.sum

/-- Problem statement: sum of the digits in the base 8 representation of 888 -/
theorem sum_of_digits_base8_888 : sum_of_digits (convert_to_base 888 8) = 13 := 
  sorry

end sum_of_digits_base8_888_l222_222384


namespace trajectory_of_M_is_ellipse_fixed_point_on_y_axis_l222_222655

noncomputable def circle (x y : ℝ) (cx cy : ℝ) (r : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 = r^2

def symmetric (x y : ℝ) : (ℝ × ℝ) := (-x, -y)

def point_on_line (x y k : ℝ) (y_int : ℝ) : Prop :=
  y = k * x + y_int

-- Condition 1: Point P is on the circle centered at (1,0) with radius sqrt(8)
def pointP_on_circle (x y : ℝ) : Prop :=
  circle x y 1 0 (real.sqrt 8)

-- Condition 2: Point F2 is symmetric to F1 with respect to the origin
def pointF2 (x2 y2 : ℝ) : Prop :=
  (x2, y2) = symmetric 1 0

-- First question: Prove the trajectory of M is the ellipse with equation x^2/2 + y^2 = 1
theorem trajectory_of_M_is_ellipse :
  ∃ M : ℝ × ℝ, ∀ x y : ℝ, (∃ (hx : pointP_on_circle x y) (hx2 : pointF2 1 0), true) →
    (circle (fst M) (snd M) 0 0 1 → (x^2 / 2 + y^2 = 1)) := sorry

-- Second question: Prove there exists a fixed point Q(0, -1) on the y-axis
theorem fixed_point_on_y_axis (k : ℝ) :
  (∃ Q : ℝ × ℝ, ∀ x1 y1 x2 y2 : ℝ, (∃ (hline : point_on_line x1 y1 k (1/3)) (hline2 : point_on_line x2 y2 k (1/3)),
  (x1 + x2)/2 = 0 ∧ y1 = k * x1 + 1/3 ∧ y2 = k * x2 + 1/3) → Q = (0, -1)) := sorry

end trajectory_of_M_is_ellipse_fixed_point_on_y_axis_l222_222655


namespace sum_of_digits_base8_888_l222_222333

theorem sum_of_digits_base8_888 : 
  let n := 888 in
  let base := 8 in
  let digits := [1, 5, 7, 0] in
  digits.sum = 13 := 
by
  sorry

end sum_of_digits_base8_888_l222_222333


namespace solution_set_f_neg_l222_222546

-- Definitions as per conditions
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)
variable (h_dom : ∀ x > 0, True)
variable (h_der : ∀ x > 0, deriv f x = f' x)
variable (h_cond : ∀ x > 0, x * f' x > f x)
variable (h_f2 : f 2 = 0)

-- Proof statement
theorem solution_set_f_neg : {x | f x < 0} = Ioo 0 2 :=
sorry

end solution_set_f_neg_l222_222546


namespace jacket_price_l222_222869

theorem jacket_price (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) (sales_tax : ℝ) : 
  original_price = 150 → initial_discount = 0.25 → additional_discount = 10 → sales_tax = 0.10 →
  (original_price * (1 - initial_discount) - additional_discount) * (1 + sales_tax) = 112.75 :=
by
  assume h1 : original_price = 150
  assume h2 : initial_discount = 0.25
  assume h3 : additional_discount = 10
  assume h4 : sales_tax = 0.10
  sorry

end jacket_price_l222_222869


namespace smallest_positive_solution_of_tan3x_tan2x_eq_sec2x_l222_222006

theorem smallest_positive_solution_of_tan3x_tan2x_eq_sec2x :
  ∃ (x : ℝ), x > 0 ∧ tan (3 * x) - tan (2 * x) = sec (2 * x) ∧ x = π / 6 :=
by 
  -- proof will be here
  sorry

end smallest_positive_solution_of_tan3x_tan2x_eq_sec2x_l222_222006


namespace solution_set_absolute_value_inequality_l222_222778

theorem solution_set_absolute_value_inequality (x : ℝ) :
  (|x-3| + |x-5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solution_set_absolute_value_inequality_l222_222778


namespace opposite_of_neg_half_is_half_l222_222736

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l222_222736


namespace k_not_congruent_one_mod_3_l222_222671

theorem k_not_congruent_one_mod_3 
  (k : ℕ) 
  (h_pos : 0 < k) 
  (h_part : ∃ A B C : finset ℕ, 
              (X : finset ℕ := finset.range (3 ^ 31 + k + 1) \ finset.range (3 ^ 31) ∧
              disjoint A B ∧ disjoint A C ∧ disjoint B C ∧ 
              (A ∪ B ∪ C = X) ∧ 
              (A.sum id = B.sum id) ∧ (B.sum id = C.sum id))) 
  : k % 3 ≠ 1 :=
sorry

end k_not_congruent_one_mod_3_l222_222671


namespace sum_of_digits_base8_888_l222_222368

def base8_representation (n : Nat) : List Nat :=
  let rec helper (n : Nat) (acc : List Nat) : List Nat :=
    if n == 0 then acc else helper (n / 8) ((n % 8) :: acc)
  helper n []

def sum_of_list (l : List Nat) : Nat :=
  l.foldl (· + ·) 0

theorem sum_of_digits_base8_888 : 
  sum_of_list (base8_representation 888) = 13 := 
  by
    sorry

end sum_of_digits_base8_888_l222_222368


namespace collinear_points_in_cube_l222_222884

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l222_222884


namespace conjugate_of_fraction_l222_222566

noncomputable def conjugate_complex_number (z : ℂ) : ℂ := conj z

theorem conjugate_of_fraction (i : ℂ) (h : i * i = -1) : 
  conjugate_complex_number ((i - 2) / i) = 1 - 2 * i := by 
  sorry

end conjugate_of_fraction_l222_222566


namespace fishing_problem_l222_222631

theorem fishing_problem :
  ∃ (x y : ℕ), 
    (x + y = 70) ∧ 
    (∃ k : ℕ, x = 9 * k) ∧ 
    (∃ m : ℕ, y = 17 * m) ∧ 
    x = 36 ∧ 
    y = 34 := 
by
  sorry

end fishing_problem_l222_222631
