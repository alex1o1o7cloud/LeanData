import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Matrix.Determinant
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics.CombinatorialGame
import Mathlib.Data.Finset
import Mathlib.Data.List.Mathlib
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Polygon.Basic
import Mathlib.Geometry.Projective
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Basic
import data.set

namespace subset_three_sum_zero_l777_777574

noncomputable def S (n : ℕ) : Set ℤ := { x | - (2 * n - 1) ≤ x ∧ x ≤ 2 * n - 1 }

theorem subset_three_sum_zero (n : ℕ) (A : Set ℤ) (h1 : A ⊆ S n) (h2 : A.card = 2 * n + 1) :
  ∃ a b c ∈ A, a + b + c = 0 :=
sorry

end subset_three_sum_zero_l777_777574


namespace bees_on_second_day_l777_777591

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end bees_on_second_day_l777_777591


namespace defective_pens_l777_777903

theorem defective_pens :
  ∃ D N : ℕ, (N + D = 9) ∧ (N / 9 * (N - 1) / 8 = 5 / 12) ∧ (D = 3) :=
by
  sorry

end defective_pens_l777_777903


namespace fraction_simplification_l777_777416

open Real -- Open the Real namespace for real number operations

theorem fraction_simplification (a x : ℝ) : 
  (sqrt (a^2 + x^2) - (x^2 + a^2) / sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 := 
sorry

end fraction_simplification_l777_777416


namespace range_of_function_l777_777516

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.logb 5 (3^x + m)

theorem range_of_function :
  ∀ {m : ℝ}, f 3 m = 2 → (λ x : ℝ, -x^(m / 3)) '' (Set.Ioi 0) = Set.Iio 0 :=
by
  sorry

end range_of_function_l777_777516


namespace inequality_holds_for_all_x_iff_a_in_range_l777_777467

theorem inequality_holds_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ (-4 < a ∧ a < -1) :=
by
  sorry

end inequality_holds_for_all_x_iff_a_in_range_l777_777467


namespace y_intercept_3x_minus_4y_eq_12_l777_777670

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l777_777670


namespace weight_of_each_hardcover_book_l777_777927

theorem weight_of_each_hardcover_book
  (weight_limit : ℕ := 80)
  (hardcover_books : ℕ := 70)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (textbook_weight : ℕ := 2)
  (knick_knack_weight : ℕ := 6)
  (over_weight : ℕ := 33)
  (total_weight : ℕ := hardcover_books * x + textbooks * textbook_weight + knick_knacks * knick_knack_weight)
  (weight_eq : total_weight = weight_limit + over_weight) :
  x = 1 / 2 :=
by {
  sorry
}

end weight_of_each_hardcover_book_l777_777927


namespace calc_expression_l777_777277

variable {x : ℝ}

theorem calc_expression :
    (2 + 3 * x) * (-2 + 3 * x) = 9 * x ^ 2 - 4 := sorry

end calc_expression_l777_777277


namespace relationship_abc_l777_777507

noncomputable def a : ℝ := Real.log 0.7 / Real.log 2
noncomputable def b : ℝ := 0.7^2
noncomputable def c : ℝ := 2^0.3

theorem relationship_abc : a < b ∧ b < c := 
by
  sorry

end relationship_abc_l777_777507


namespace inradius_of_right_triangle_l777_777603

theorem inradius_of_right_triangle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = (1/2) * (a + b - c) :=
sorry

end inradius_of_right_triangle_l777_777603


namespace salary_reduction_l777_777247

theorem salary_reduction (S : ℝ) (R : ℝ) 
  (h : (S - (R / 100) * S) * (4 / 3) = S) :
  R = 25 := 
  sorry

end salary_reduction_l777_777247


namespace no_such_hexagon_exists_l777_777317

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777317


namespace a_1_value_l777_777444

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)

axiom a_n_def : ∀ n ≥ 2, a n + 2 * (S n) * (S (n - 1)) = 0
axiom S_5_value : S 5 = 1/11
axiom summation_def : ∀ k ≥ 1, S k = S (k - 1) + a k

theorem a_1_value : a 1 = 1/3 := by
  sorry

end a_1_value_l777_777444


namespace geometric_sequence_general_term_geometric_sequence_sum_n_l777_777441

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end geometric_sequence_general_term_geometric_sequence_sum_n_l777_777441


namespace XY_perpendicular_to_median_l777_777646

variables {A B C P X Y M: Type} [MetricSpace A B] [MetricSpace B C] [MetricSpace C A] 
          [MetricSpace A P] [MetricSpace B P] [MetricSpace P X] [MetricSpace P Y] 
          [MetricSpace A Y] [MetricSpace B X] [MetricSpace M C]

-- Define the triangle inscribed in the circle
def inscribed_triangle (A B C : Type) : Prop := 
  True -- Placeholder for the definition of inscribed triangle in a circle

-- Define tangents intersecting at P
def tangents_intersect (A B P : Type) : Prop := 
  True -- Placeholder for the definition of tangents intersecting at a point

-- Define orthogonal projections X and Y on AC and BC
def orthogonal_projections (P A C X B Y : Type) : Prop :=
  True -- Placeholder for the definition of orthogonal projections

-- Define median CM
def is_median (C M A B : Type) : Prop := 
  True -- Placeholder for the definition of a median

-- The required proof that XY is perpendicular to the median CM
theorem XY_perpendicular_to_median (h1 : inscribed_triangle A B C)
  (h2 : tangents_intersect A B P) 
  (h3 : orthogonal_projections P A C X P B C Y) 
  (h4 : is_median C M A B) :
  ∀ (XY : Type), XY ⊥ CM :=
by
  sorry

end XY_perpendicular_to_median_l777_777646


namespace sum_integers_neg50_to_75_l777_777272

theorem sum_integers_neg50_to_75 : 
  ∑ k in Finset.range (75 + 1) \ Finset.range (-50), k = 1575 :=
by
  sorry

end sum_integers_neg50_to_75_l777_777272


namespace min_value_expression_l777_777813

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ c : ℝ, c = (1 / (2 * x) + x / (y + 1)) ∧ c = 5 / 4) :=
sorry

end min_value_expression_l777_777813


namespace midpoint_is_excenter_l777_777618

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

theorem midpoint_is_excenter (A B C : Point) (hABC : is_isosceles_triangle A B C) 
  (O : Point) (circumcircle : Circle) (circle : Circle) (hTangent1 : is_tangent circle circumcircle) 
  (hTangent2 : is_tangent circle (line_through A B)) (hTangent3 : is_tangent circle (line_through A C))
  (P Q : Point) (hPQ1 : is_tangent_at circle (line_through A B) P) (hPQ2 : is_tangent_at circle (line_through A C) Q) :
  is_excenter (midpoint P Q) A :=
sorry

end midpoint_is_excenter_l777_777618


namespace probability_of_same_length_l777_777962

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777962


namespace collinear_D_E_H_l777_777898

-- Define the triangle with vertices A, B, C
variables {A B C D E H I : Point}
variables (h1 : Length AB < Length AC)
variables (h2 : OnLine D (LineThrough C A))
variables (h3 : OnLine E (LineThrough B A))
variables (h4 : Length CD = Length AB)
variables (h5 : Length BE = Length AC)
variables (h6 : SameSide A D E (LineThrough B C))
variables (h7 : IsIncenter I (Triangle A B C))
variables (h8 : IsOrthocenter H (Triangle B C I))

theorem collinear_D_E_H :
  Collinear D E H :=
sorry

end collinear_D_E_H_l777_777898


namespace no_such_hexagon_exists_l777_777320

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777320


namespace percentage_B_D_l777_777808

variables (A B C D : ℝ)

-- Conditions as hypotheses
theorem percentage_B_D
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) : 
  B = 1.1115 * D :=
sorry

end percentage_B_D_l777_777808


namespace radius_of_circle_Q_l777_777782

-- Definitions based on problem conditions
structure Circle (α : Type _) :=
(center : α)
(radius : ℝ) 

variables {α : Type _} [MetricSpace α]

-- Centers of circles P, Q, R, S
variables (X Y Z W : α)
-- Radii of circles P, Q, R, S
variables (rP rQ rR rS : ℝ)
-- Tangency points (internal and external)
variables (T : α)

-- Conditions
axiom circle_P : Circle α := ⟨X, 2⟩
axiom circle_Q : Circle α := ⟨Y, rQ⟩
axiom circle_R : Circle α := ⟨Z, rQ⟩
axiom circle_S : Circle α := ⟨W, rS⟩

-- Circle properties for tangency and radii relationships
axiom tangency_cond_1 : dist X Y = 2 + rQ
axiom tangency_cond_2 : dist X Z = 2 + rQ
axiom tangency_cond_3 : dist Y Z = 2 * rQ
axiom tangency_cond_4 : dist W X = 2
axiom tangency_cond_5 : dist W Y = 2 + rQ
axiom tangency_cond_6 : dist W Z = 2 + rQ

-- Proof problem statement
theorem radius_of_circle_Q : rQ = 16 / 9 := sorry

end radius_of_circle_Q_l777_777782


namespace sum_of_roots_of_quadratic_l777_777687

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l777_777687


namespace sum_of_roots_l777_777693

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l777_777693


namespace smallest_n_for_fraction_with_digits_439_l777_777620

theorem smallest_n_for_fraction_with_digits_439 (m n : ℕ) (hmn : Nat.gcd m n = 1) (hmn_pos : 0 < m ∧ m < n) (digits_439 : ∃ X : ℕ, (m : ℚ) / n = (439 + 1000 * X) / 1000) : n = 223 :=
by
  sorry

end smallest_n_for_fraction_with_digits_439_l777_777620


namespace sinx_tanx_condition_l777_777045

theorem sinx_tanx_condition (x : ℝ) : (sin x = (Real.sqrt 2) / 2) → (¬ (tan x = 1)) ∧ (tan x = 1 → sin x = (Real.sqrt 2) / 2) :=
by
  sorry

end sinx_tanx_condition_l777_777045


namespace range_of_k_l777_777514

/-- If the function y = (k + 1) * x is decreasing on the entire real line, then k < -1. -/
theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x < y → (k + 1) * x > (k + 1) * y) : k < -1 :=
sorry

end range_of_k_l777_777514


namespace hexagon_shaded_area_fraction_l777_777146

theorem hexagon_shaded_area_fraction (O A B C D E F Y Z : Point)
(h1 : is_center O (hexagon A B C D E F))
(h2 : is_midpoint Y C D)
(h3 : is_midpoint Z D E)
(h4 : shaded_region (triangle COY) (triangle DOY) (triangle EZO)) :
  shaded_area_fraction (hexagon A B C D E F) = 1/4 :=
sorry

end hexagon_shaded_area_fraction_l777_777146


namespace geometric_locus_of_fourth_vertex_l777_777413

theorem geometric_locus_of_fourth_vertex
  (O : ℝ × ℝ) (R : ℝ)
  (x0 y0: ℝ) (x1 y1 : ℝ) (x2 y2 : ℝ)
  (h1 : (x1 - O.1)^2 + (y1 - O.2)^2 = R^2)
  (h2 : (x2 - O.1)^2 + (y2 - O.2)^2 = R^2)
  (h3 : (x0 - O.1)^2 + (y0 - O.2)^2 < R^2) :
  ∃ x4 y4 : ℝ, x4 = x1 + x2 - x0 ∧ y4 = y1 + y2 - y0 :=
begin
  use [x1 + x2 - x0, y1 + y2 - y0],
  split;
  refl,
end

end geometric_locus_of_fourth_vertex_l777_777413


namespace cos_120_eq_neg_one_half_l777_777234

theorem cos_120_eq_neg_one_half :
  (cos 120) = - (1 / 2) :=
by
  sorry

end cos_120_eq_neg_one_half_l777_777234


namespace y_intercept_of_line_l777_777676

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l777_777676


namespace Carlos_does_not_pay_Bernardo_l777_777102

-- Define the initial amounts paid
def LeRoy_paid : ℝ := 100
def Bernardo_paid : ℝ := 150
def Carlos_paid_eur : ℝ := 120
def exchange_rate : ℝ := 1.10

-- Define Carlos' payment in dollars
def Carlos_paid : ℝ := Carlos_paid_eur * exchange_rate

-- Calculate total expenses
def total_expenses : ℝ := LeRoy_paid + Bernardo_paid + Carlos_paid

-- Calculate each person's share
def equal_share : ℝ := total_expenses / 3

-- Define who needs to pay whom
def Carlos_to_Bernardo : ℝ := 0

-- Main statement ensuring Carlos does not need to pay Bernardo anything
theorem Carlos_does_not_pay_Bernardo :
  Carlos_to_Bernardo = 0 := sorry

end Carlos_does_not_pay_Bernardo_l777_777102


namespace problem1_eval_problem2_eval_l777_777283

-- Problem 1
theorem problem1_eval :
  (1 : ℚ) * (-4.5) - (-5.6667) - (2.5) - 7.6667 = -9 := 
by
  sorry

-- Problem 2
theorem problem2_eval :
  (-(4^2) / (-2)^3) - ((4 / 9) * ((-3 / 2)^2)) = 1 := 
by
  sorry

end problem1_eval_problem2_eval_l777_777283


namespace distinct_roots_in_intervals_l777_777292

variables (a : ℝ) (x : ℝ)

-- Define the quadratic equations for different cases
def quadratic_nonnegative_case (a x : ℝ) : Prop := 
  x^2 - 2*a*x + (4*a - 3) = 0

def quadratic_negative_case (a x : ℝ) : Prop := 
  x = (4*a - 3) / (2*a)

-- Statement to prove distinct roots in specific intervals
theorem distinct_roots_in_intervals :
  (∀ x ≥ 0, quadratic_nonnegative_case a x → 
    ((a ∈ set.Ico (3/4) 1 ∪ set.Ioi 3) → (x ≠ 0 ∧ x ≠ -1))) ∧
  (∀ x < 0, quadratic_negative_case a x → 
    ((a ∈ set.Ioo 0 (3/4)) → (x ≠ 0 ∧ x ≠ -1))) :=
sorry

end distinct_roots_in_intervals_l777_777292


namespace smallest_k_independent_expression_l777_777293

theorem smallest_k_independent_expression:
  ∃ k, k ≥ 1 ∧ (∀ x, (sin (k * x) * (sin x) ^ k + cos (k * x) * (cos x) ^ k - (cos (2 * x)) ^ k) = (sin (3 * x) * (sin x) ^ 3 + cos (3 * x) * (cos x) ^ 3 - (cos (2 * x)) ^ 3) ∧ k = 3) :=
sorry

end smallest_k_independent_expression_l777_777293


namespace max_sin_sum_acute_triangle_l777_777888

theorem max_sin_sum_acute_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_angle_A : A ≤ π / 2) (h_angle_B : B ≤ π / 2) (h_angle_C : C ≤ π / 2) :
  (sin A + sin B + sin C) ≤ (3 * (Real.sin (π / 3))) :=
sorry

end max_sin_sum_acute_triangle_l777_777888


namespace simple_interest_rate_l777_777264

-- Define the conditions
def principal : ℝ := 750
def amount : ℝ := 1050
def time : ℝ := 5

-- Define the simple interest equation as a function
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Define the amount equation
def amount_with_interest (P SI : ℝ) : ℝ :=
  P + SI

-- Define the problem statement
theorem simple_interest_rate :
  ∃ R : ℝ, amount_with_interest principal (simple_interest principal R time) = amount :=
by {
  -- Let the required rate be R
  let R := (amount - principal) * 100 / (principal * time)
  use R
  
  -- Substitute R into the equations
  sorry
}

end simple_interest_rate_l777_777264


namespace sufficient_condition_for_inequality_l777_777636

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a > 4 :=
by 
  sorry

end sufficient_condition_for_inequality_l777_777636


namespace solution_x_y_l777_777289

theorem solution_x_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
    x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) :=
by {
    sorry
}

end solution_x_y_l777_777289


namespace number_of_valid_k_is_96_l777_777014

theorem number_of_valid_k_is_96 :
  ∃ N : ℕ, N = 96 ∧ (∀ k : ℤ, |k| < 150 → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) ↔ k ∈ valid_ks)
where valid_ks : Set ℤ :=
  {k : ℤ | |k| < 150 ∧ ∃ x : ℤ, 3 * x^2 + k * x + 8 = 0} :=
sorry

end number_of_valid_k_is_96_l777_777014


namespace find_number_lemma_l777_777804

theorem find_number_lemma (x : ℝ) (a b c d : ℝ) (h₁ : x = 5) 
  (h₂ : a = 0.47 * 1442) (h₃ : b = 0.36 * 1412) 
  (h₄ : c = a - b) (h₅ : d + c = x) : 
  d = -164.42 :=
by
  sorry

end find_number_lemma_l777_777804


namespace no_such_hexagon_and_point_l777_777393

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777393


namespace smallest_number_with_conditions_l777_777794

theorem smallest_number_with_conditions (n : ℕ) 
  (h1 : 8 ∣ n) 
  (h2 : 5 ∣ n) 
  (h3 : ∀ p : ℕ, p.prime → p ≠ 2 → p ≠ 5 → p ∣ n → (p = 3 ∨ p = 7)) : 
  n = 840 := 
sorry

end smallest_number_with_conditions_l777_777794


namespace A_equals_expected_A_union_B_equals_expected_l777_777479

-- Define the sets A and B based on given conditions
def setA_condition := {x : ℝ | ∃ y : ℝ, y = log (x^2 - 2*x)}
def setB_condition := {y : ℝ | ∃ x : ℝ, y = x^0.5 + 1}

-- Define the expected forms of sets A and B
def setA_expected := {x : ℝ | x < 0 ∨ x > 2}
def A_union_B_expected := {x : ℝ | x < 0 ∨ x ≥ 1}

-- Prove that setA_condition equals setA_expected
theorem A_equals_expected : setA_condition = setA_expected := by
  sorry

-- Prove that setA_condition union setB_condition equals A_union_B_expected
theorem A_union_B_equals_expected : setA_condition ∪ setB_condition = A_union_B_expected := by
  sorry

end A_equals_expected_A_union_B_equals_expected_l777_777479


namespace cyclic_quad_eq_l777_777471

open Real Angle

theorem cyclic_quad_eq (A B C D P : Point) (h_cyclic: CyclicQuadrilateral A B C D)
  (h_angles : ∠PA B = ∠PB C ∧ ∠PB C = ∠PC D ∧ ∠PC D = ∠PD A) :
  dist A B * dist C D = dist B C * dist A D :=
by
  sorry

end cyclic_quad_eq_l777_777471


namespace find_fraction_value_l777_777459

-- Given conditions
variables {α : ℝ} (h : real.tan α = 3)

-- The proposition to prove
theorem find_fraction_value (h : real.tan α = 3) : 
  (sin α + 3 * cos α) / (2 * sin α + 5 * cos α) = 6 / 11 :=
sorry

end find_fraction_value_l777_777459


namespace circumscribed_sphere_radius_of_pyramid_l777_777447

-- Define the vertices and their relationships
variables (S A B C : Type) [EuclideanSpace ℝ S] [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C]

-- Define the lengths of the edges
def SA_length := 2
def SB_length := 4
def SC_length := 4

-- Define the condition that the edges are mutually perpendicular
def mutually_perpendicular (SA SB SC : ℝ) : Prop := 
  is_orthogonal SA SB ∧ is_orthogonal SB SC ∧ is_orthogonal SC SA

-- The main statement
theorem circumscribed_sphere_radius_of_pyramid :
  mutually_perpendicular (SA_length : ℝ) (SB_length : ℝ) (SC_length : ℝ) →
  (circumscribed_sphere_radius_of_pyramid (triangle_pyramid S A B C) = 3) :=
sorry

end circumscribed_sphere_radius_of_pyramid_l777_777447


namespace hexagon_probability_same_length_l777_777978

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777978


namespace findPrincipalAmount_l777_777222

noncomputable def principalAmount (r : ℝ) (t : ℝ) (diff : ℝ) : ℝ :=
  let n := 2 -- compounded semi-annually
  let rate_per_period := (1 + r / n)
  let num_periods := n * t
  (diff / (rate_per_period^num_periods - 1 - r * t))

theorem findPrincipalAmount :
  let r := 0.05
  let t := 3
  let diff := 25
  abs (principalAmount r t diff - 2580.39) < 0.01 := 
by 
  sorry

end findPrincipalAmount_l777_777222


namespace coefficient_x_neg1_l777_777568

noncomputable def a : ℝ :=
  ∫ x in 0..(Real.pi / 2), (Real.sin x + Real.cos x)

theorem coefficient_x_neg1 :
  ∑ k in Finset.range 7, (-(1:ℝ))^k * (Nat.choose 6 k) * 2^(6 - k) * (λ x : ℝ, x^(3 - k)) a⁻¹  = 60 := 
sorry

end coefficient_x_neg1_l777_777568


namespace no_such_hexagon_exists_l777_777329

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777329


namespace f_maps_S_to_S_exists_t_smallest_t_if_conditions_met_l777_777561

def S := {s : ℕ × ℕ | nat.coprime s.1 s.2 ∧ s.2 % 2 = 0 ∧ s.1 < s.2}

noncomputable def f (s : ℕ × ℕ) : ℕ × ℕ :=
  let n0 := s.2 / (2 ^ (nat.find (λ k, s.2 % (2 ^ k) = 0 ∧ s.2 / (2 ^ k) % 2 = 1)))
  in (n0, s.1 + s.2 - n0)

theorem f_maps_S_to_S (s : ℕ × ℕ) (h : s ∈ S) : f s ∈ S :=
sorry

theorem exists_t (s : ℕ × ℕ) (h : s ∈ S) : ∃ t : ℕ, t ≤ (s.1 + s.2 + 1) / 4 ∧ (f^[t] s) = s :=
sorry

theorem smallest_t_if_conditions_met (s : ℕ × ℕ) (h : s ∈ S)
  (prime_sum : nat.prime (s.1 + s.2))
  (h_nodiv : ∀ k, 1 ≤ k ∧ k ≤ s.1 + s.2 - 2 → (2 ^ k - 1) % (s.1 + s.2) ≠ 0) :
  ∃ t : ℕ, t = (s.1 + s.2 + 1) / 4 ∧ (f^[t] s) = s :=
sorry

end f_maps_S_to_S_exists_t_smallest_t_if_conditions_met_l777_777561


namespace johns_contribution_correct_l777_777048

noncomputable def average_contribution_before : Real := sorry
noncomputable def total_contributions_by_15 : Real := 15 * average_contribution_before
noncomputable def new_average_contribution : Real := 150
noncomputable def johns_contribution : Real := average_contribution_before * 15 + 1377.3

-- The theorem we want to prove
theorem johns_contribution_correct :
  (new_average_contribution = (total_contributions_by_15 + johns_contribution) / 16) ∧
  (new_average_contribution = 2.2 * average_contribution_before) :=
sorry

end johns_contribution_correct_l777_777048


namespace star_computation_l777_777508

def star (x y : ℝ) := x * y - 3 * x + y

theorem star_computation :
  (star 5 8) - (star 8 5) = 12 := by
  sorry

end star_computation_l777_777508


namespace exists_smallest_positive_real_in_interval_l777_777814

def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x + 3 * Real.tan x

theorem exists_smallest_positive_real_in_interval :
  ∃ x ∈ Set.Ioo 3 4, f x = 0 ∧ ∀ y ∈ Set.Ioo 0 x, f y ≠ 0 := 
sorry

end exists_smallest_positive_real_in_interval_l777_777814


namespace max_value_a_b2_c4_l777_777989

-- Definitions
variables {a b c : ℝ}
variable h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0
variable h_sum : a + b + c = 3

-- Theorem statement
theorem max_value_a_b2_c4 : a + b^2 + c^4 ≤ 3 :=
by 
  -- Placeholder for the proof
  sorry

end max_value_a_b2_c4_l777_777989


namespace problem1_problem2_problem3_l777_777843

-- Define the quadratic function and constraints for the problem
def quadratic_function (a b x : ℝ) : ℝ := a * x^2 - b * x + 2

-- Problem 1: Axis of symmetry and vertex coordinates for given specific values
theorem problem1 (a : ℝ) (h : a ≠ 0) :
    quadratic_function (-2) (-4) = -2 * (x - 1)^2 + 4 ∧ axis_of_symmetry (-2) (-4) = 1 ∧ vertex_coordinates (-2) (-4) = (1, 4) :=
sorry

-- Problem 2: Given points Q and P symmetric about the origin on the graph
theorem problem2 (a : ℝ) (m t : ℝ) (h : a ≠ 0) :
    quadratic_function (-2) 4 m = t ∧ quadratic_function (-2) 4 (-m) = -t → m = ±1 :=
sorry

-- Problem 3: Given points A and B on the graph with provided conditions
theorem problem3 (a y1 y2 : ℝ) (h : a ≠ 0) :
    passes_through (quadratic_function a (a + 2)) (1, 0) ∧ 
    let A := (1/2, y1) in let B := (1/2 + 3/a, y2) in
    (a > 0 → y2 > y1) ∧ (a < 0 → y1 > y2) :=
sorry

end problem1_problem2_problem3_l777_777843


namespace count_int_values_in_range_l777_777028

theorem count_int_values_in_range : 
  ∃ (n_values : Finset ℤ), 
    (∀ n ∈ n_values, -100 < n^3 ∧ n^3 < 100) ∧ 
    n_values.card = 9 :=
begin
  sorry
end

end count_int_values_in_range_l777_777028


namespace second_fragment_speed_l777_777739

def initial_velocity : ℝ := 20 -- m/s
def time_of_explosion : ℝ := 1 -- s
def gravity : ℝ := 10 -- m/s^2
def first_fragment_horizontal_velocity : ℝ := 48 -- m/s

noncomputable def speed_of_second_fragment : ℝ :=
  let vertical_velocity_at_explosion := initial_velocity - gravity * time_of_explosion
  let v2x := -first_fragment_horizontal_velocity
  let v2y := 2 * vertical_velocity_at_explosion -- since mass fragments are equal
  real.sqrt (v2x^2 + v2y^2)

theorem second_fragment_speed : speed_of_second_fragment = 52 := by
  sorry

end second_fragment_speed_l777_777739


namespace value_of_f_l777_777000

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 3^x

theorem value_of_f :
  f (1 / 4) = -2 :=
by
  sorry

end value_of_f_l777_777000


namespace point_in_fourth_quadrant_l777_777993

def i : ℂ := complex.I

def z : ℂ := (2 + i) * (1 - i)

theorem point_in_fourth_quadrant (z : ℂ) (hx : z = (2 + i) * (1 - i)) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end point_in_fourth_quadrant_l777_777993


namespace fraction_clever_integers_divisible_by_25_l777_777788

def is_clever_integer (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 30 ≤ n ∧ n ≤ 150 ∧ (n.digits 10).sum = 10

def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0

theorem fraction_clever_integers_divisible_by_25 :
  (∃ (S : Finset ℕ), (∀ n ∈ S, is_clever_integer n) ∧ (S.filter is_divisible_by_25).card / S.card = 1 / 8) :=
sorry

end fraction_clever_integers_divisible_by_25_l777_777788


namespace no_such_hexagon_exists_l777_777362

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777362


namespace evaluate_expression_l777_777274

variable (a : Real) (h1 : a ≠ 0) (h2 : 128 = 2^7) (h3 : -64 = -(2^6))

theorem evaluate_expression : 
  ( (1/8) * a^0 + ( (1 / (8 * a) ) )^0 - 128^(-1/3) - (-64)^(-2/3) ) = 1 + 7/25 := 
by
  -- sorry is used to skip the proof
  sorry

end evaluate_expression_l777_777274


namespace max_PB_dot_PC_l777_777541

noncomputable def A : Point := (0, 0)
def B : Point := (3 / 2, (3 * Real.sqrt 3) / 2)
def C : Point := (4, 0)
def P (θ : ℝ) : Point := (2 * Real.cos θ, 2 * Real.sin θ)

def PB (θ : ℝ) : Vector := (B.1 - 2 * Real.cos θ, B.2 - 2 * Real.sin θ)
def PC (θ : ℝ) : Vector := (C.1 - 2 * Real.cos θ, C.2 - 2 * Real.sin θ)

def dotProduct (u v : Vector) : ℝ := u.1 * v.1 + u.2 * v.2

def PB_dot_PC (θ : ℝ) : ℝ :=
  let u := PB θ
  let v := PC θ
  dotProduct u v

theorem max_PB_dot_PC : ∃ θ : ℝ, PB_dot_PC θ = 10 + 2 * Real.sqrt 37 :=
sorry

end max_PB_dot_PC_l777_777541


namespace matrix_B_pow48_l777_777103

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 2], ![0, -2, 0]]

theorem matrix_B_pow48 :
  B ^ 48 = ![![0, 0, 0], ![0, 16^12, 0], ![0, 0, 16^12]] :=
by sorry

end matrix_B_pow48_l777_777103


namespace teta_beta_gamma_l777_777011

theorem teta_beta_gamma : 
  ∃ T E T' A B E' T'' A' G A'' M M' A''' A'''' : ℕ, 
  TETA = T * 1000 + E * 100 + T' * 10 + A ∧ 
  BETA = B * 1000 + E' * 100 + T'' * 10 + A' ∧ 
  GAMMA = G * 10000 + A'' * 1000 + M * 100 + M' * 10 + A''' ∧
  TETA + BETA = GAMMA ∧ 
  A = A'''' ∧ E = E' ∧ T = T' ∧ T' = T'' ∧ A = A' ∧ A = A'' ∧ A = A''' ∧ M = M' ∧ 
  T ≠ E ∧ T ≠ A ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧
  E ≠ A ∧ E ≠ B ∧ E ≠ G ∧ E ≠ M ∧
  A ≠ B ∧ A ≠ G ∧ A ≠ M ∧
  B ≠ G ∧ B ≠ M ∧
  G ≠ M ∧
  TETA = 4940 ∧ BETA = 5940 ∧ GAMMA = 10880
  :=
sorry

end teta_beta_gamma_l777_777011


namespace ratio_of_engineers_to_designers_l777_777617

-- Definitions of the variables
variables (e d : ℕ)

-- Conditions:
-- 1. The average age of the group is 45
-- 2. The average age of engineers is 40
-- 3. The average age of designers is 55

theorem ratio_of_engineers_to_designers (h : (40 * e + 55 * d) / (e + d) = 45) : e / d = 2 :=
by
-- Placeholder for the proof
sorry

end ratio_of_engineers_to_designers_l777_777617


namespace probability_five_collinear_in_5x5_grid_l777_777529

/-- In a square array of 25 dots (5x5 grid), what is the probability that five randomly chosen dots are collinear? -/
theorem probability_five_collinear_in_5x5_grid :
  (12 : ℚ) / Nat.choose 25 5 = 2 / 8855 := by
  sorry

end probability_five_collinear_in_5x5_grid_l777_777529


namespace move_point_l777_777534

def move_left (p : Point) (d : ℝ) : Point := ⟨p.x - d, p.y⟩
def move_up (p : Point) (d : ℝ) : Point := ⟨p.x, p.y + d⟩

def initialA : Point := ⟨2, 1⟩

theorem move_point :
  let A' := move_up (move_left initialA 3) 1
  A' = ⟨-1, 2⟩ :=
by sorry

end move_point_l777_777534


namespace no_such_convex_hexagon_and_point_l777_777343

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777343


namespace a_2004_interval_l777_777644

theorem a_2004_interval {a : ℕ → ℝ} (h : ∀ n : ℕ, a (n + 1) = a n * (a n + 2)) :
  a 2004 ∈ set.Ici (-1) :=
sorry

end a_2004_interval_l777_777644


namespace proof_x_minus_y_squared_l777_777501

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l777_777501


namespace minimum_n_minus_m_l777_777009

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_l777_777009


namespace no_such_hexagon_exists_l777_777331

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777331


namespace geometric_sequence_properties_l777_777438

-- Define the geometric sequence with the given conditions
def geometric_sequence (a₃ : ℕ → ℝ) := (a₃ 3 = 12) ∧ (a₃ 8 = 3 / 8)

-- Define the general formula for the n-th term of a geometric sequence
def general_term (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
def sum_of_geometric_sequence (a₁ q : ℝ) (S_n : ℕ → ℝ) (n : ℕ) : Prop :=
  S_n n = a₁ * (1 - q^n) / (1 - q)

-- The proof problem statement
theorem geometric_sequence_properties : 
  ∃ a₁ q S_n : ℝ,
  geometric_sequence (λ n, general_term a₁ q n)
  →
  ∀ n, (general_term a₁ q n = 48 * (1 / 2)^(n - 1)) 
       ∧ (S_n n = 93 → n = 5) :=
begin
  sorry
end

end geometric_sequence_properties_l777_777438


namespace problem_false_proposition_l777_777481

def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0

theorem problem_false_proposition : ¬ (p ∧ q) :=
by
  sorry

end problem_false_proposition_l777_777481


namespace auntie_em_can_park_l777_777748

-- Define the total number of parking spaces
def total_spaces : ℕ := 20

-- Define the number of cars that have already parked
def parked_cars : ℕ := 14

-- Define the number of empty spaces required for Auntie Em's vehicle
def required_adjacent_spaces : ℕ := 3

-- Probability that Auntie Em can park
def prob_can_park : ℚ := 19 / 25

-- Formalize the theorem statement
theorem auntie_em_can_park :
  ∃ (total_spaces parked_cars required_adjacent_spaces : ℕ),
    total_spaces = 20 ∧
    parked_cars = 14 ∧
    required_adjacent_spaces = 3 ∧
    (prob_can_park = 19 / 25) :=
begin
  use [total_spaces, parked_cars, required_adjacent_spaces],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  exact sorry
end

end auntie_em_can_park_l777_777748


namespace no_such_convex_hexagon_and_point_l777_777337

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777337


namespace percentage_of_A_l777_777207

-- Define variables and assumptions
variables (A B : ℕ)
def total_payment := 580
def payment_B := 232

-- Define the proofs of the conditions provided in the problem
axiom total_payment_eq : A + B = total_payment
axiom B_eq : B = payment_B
noncomputable def percentage_paid_to_A := (A / B) * 100

-- Theorem to prove the percentage of the payment to A compared to B
theorem percentage_of_A : percentage_paid_to_A = 150 :=
by
 sorry

end percentage_of_A_l777_777207


namespace probability_same_length_segments_of_regular_hexagon_l777_777947

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777947


namespace no_such_hexagon_and_point_l777_777389

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777389


namespace base9_minus_base6_l777_777405

-- Definitions from conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 325 => 3 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Main theorem statement
theorem base9_minus_base6 : base9_to_base10 325 - base6_to_base10 231 = 175 :=
by
  sorry

end base9_minus_base6_l777_777405


namespace unequal_circles_cannot_have_one_or_four_common_tangents_l777_777206

theorem unequal_circles_cannot_have_one_or_four_common_tangents 
  (r1 r2 : ℝ) (h : r1 ≠ r2) :
  ∀ (n : ℕ), n = 1 ∨ n = 4 → ¬ (∃ P Q : set (ℝ × ℝ), 
    (is_common_tangent P r1 r2 ∧ is_common_tangent Q r1 r2 ∧ finset.card {P, Q} = n)) :=
by
  assume n,
  assume hn : n = 1 ∨ n = 4,
  assume hpq : ∃ P Q : set (ℝ × ℝ), 
    (is_common_tangent P r1 r2 ∧ is_common_tangent Q r1 r2 ∧ finset.card {P, Q} = n),
  sorry

end unequal_circles_cannot_have_one_or_four_common_tangents_l777_777206


namespace unit_circle_sector_arc_length_l777_777068

theorem unit_circle_sector_arc_length (r S l : ℝ) (h1 : r = 1) (h2 : S = 1) (h3 : S = 1 / 2 * l * r) : l = 2 :=
by
  sorry

end unit_circle_sector_arc_length_l777_777068


namespace exists_K_l777_777921

noncomputable def K_internally_tangent (A B C I P Q R K : Point) : Prop :=
  let incircle := Circle I (dist I P)
  let circumcircle := Circle (center B C) (dist B C)
  tangent incircle circumcircle ∧ has_point K incircle ∧ has_point K circumcircle

variables
  (A B C I P Q R : Point)
  (h_triangle : triangle A B C)
  (h_incircle : ∃ I, incircle A B C I P Q R)

theorem exists_K (A B C I P Q R : Point)
  (h_triangle : triangle A B C)
  (h_incircle : ∃ I, incircle A B C I P Q R) :
  ∃ K : Point, K_internally_tangent A B C I P Q R K :=
sorry

end exists_K_l777_777921


namespace cake_consumption_after_seven_trips_l777_777708

theorem cake_consumption_after_seven_trips : let remaining_fraction := (1 / 3) in
  let trips := 7 in
  let eaten_fraction := (1 - (remaining_fraction ^ trips)) / (1 - remaining_fraction) in
  eaten_fraction = 1093 / 2187 :=
by
  -- Sorry skips the proof step.
  sorry

end cake_consumption_after_seven_trips_l777_777708


namespace packets_of_sugar_per_week_l777_777753

theorem packets_of_sugar_per_week (total_grams : ℕ) (packet_weight : ℕ) (total_packets : ℕ) :
  total_grams = 2000 →
  packet_weight = 100 →
  total_packets = total_grams / packet_weight →
  total_packets = 20 := 
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 

end packets_of_sugar_per_week_l777_777753


namespace circle_intersect_sum_eq_400_l777_777143

theorem circle_intersect_sum_eq_400 {A B C X Y : Type} [MetricSpace X]
  (AX AB CY BC : ℝ) (d : ℝ) 
  (h1 : 2 * 10 = 20) -- Diameter of the circle
  (h2 : AX * AB = d^2 - 100) -- Power of Point theorem for AX
  (h3 : CY * BC = d^2 - 100) -- Power of Point theorem for CY
  : AX * AB + CY * BC = 400 := 
  by 
  have sum_eq_2d2_minus_200 : AX * AB + CY * BC = 2 * (d^2 - 100) :=
    by rw [h2, h3]
  have simpl_sum : 2 * (d^2 - 100) = 400 :=
    by rw [show d^2 - 100 = 100 from sorry, mul_assoc, mul_comm 2 100]
  rw [sum_eq_2d2_minus_200, simpl_sum]

end circle_intersect_sum_eq_400_l777_777143


namespace kevin_cards_found_l777_777100

theorem kevin_cards_found : ∀ (initial_cards found_cards total_cards : Nat), 
  initial_cards = 7 → 
  total_cards = 54 → 
  total_cards - initial_cards = found_cards →
  found_cards = 47 :=
by
  intros initial_cards found_cards total_cards h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end kevin_cards_found_l777_777100


namespace integer_solutions_of_inequality_system_l777_777614

theorem integer_solutions_of_inequality_system :
  { x : ℤ | (3 * x - 2) / 3 ≥ 1 ∧ 3 * x + 5 > 4 * x - 2 } = {2, 3, 4, 5, 6} :=
by {
  sorry
}

end integer_solutions_of_inequality_system_l777_777614


namespace inequality_system_solution_range_l777_777056

theorem inequality_system_solution_range (x m : ℝ) :
  (∃ x : ℝ, (x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 :=
by
  sorry

end inequality_system_solution_range_l777_777056


namespace problem_part_I_problem_part_II_l777_777840

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

def M : set ℝ := { x | x < 1 / 2 ∨ x > 5 / 2 }

theorem problem_part_I : { x : ℝ | f x > 2 } = M := by
  sorry

theorem problem_part_II (x : ℝ) (a b : ℝ) (h : a ≠ 0) (hx : ¬ (x < 1 / 2 ∨ x > 5 / 2)) :
  abs (a + b) + abs (a - b) ≥ abs a * f x := by
  sorry

end problem_part_I_problem_part_II_l777_777840


namespace no_such_hexagon_and_point_l777_777387

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777387


namespace petes_average_speed_l777_777771

theorem petes_average_speed
    (map_distance : ℝ := 5) 
    (time_taken : ℝ := 1.5) 
    (map_scale : ℝ := 0.05555555555555555) :
    (map_distance / map_scale) / time_taken = 60 := 
by
    sorry

end petes_average_speed_l777_777771


namespace no_such_hexagon_exists_l777_777355

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777355


namespace find_x_l777_777417

theorem find_x (x : ℝ) (h : sqrt (9 - 2 * x) = 5) : x = -8 :=
sorry

end find_x_l777_777417


namespace sum_of_roots_l777_777697

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l777_777697


namespace no_such_hexagon_and_point_l777_777388

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777388


namespace odd_function_m_l777_777812

def f (m : ℝ) (x : ℝ) : ℝ := m + 2 / (3^x - 1)

theorem odd_function_m (m : ℝ) (f_odd : ∀ x : ℝ, f m x = -f m (-x)) : m = 1 :=
by
  sorry

end odd_function_m_l777_777812


namespace area_of_triangle_ADE_l777_777526

noncomputable def triangle_area (A B C: ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_ADE (A B D E F : ℝ × ℝ) (h₁ : A.1 = 0 ∧ A.2 = 0) (h₂ : B.1 = 8 ∧ B.2 = 0)
  (h₃ : D.1 = 8 ∧ D.2= 8) (h₄ : E.1 = 4 * 3 / 5 ∧ E.2 = 0) 
  (h₅ : F.1 = 0 ∧ F.2 = 12) :
  triangle_area A D E = 288 / 25 := 
sorry

end area_of_triangle_ADE_l777_777526


namespace radius_of_circle_Q_l777_777781

-- Definitions based on problem conditions
structure Circle (α : Type _) :=
(center : α)
(radius : ℝ) 

variables {α : Type _} [MetricSpace α]

-- Centers of circles P, Q, R, S
variables (X Y Z W : α)
-- Radii of circles P, Q, R, S
variables (rP rQ rR rS : ℝ)
-- Tangency points (internal and external)
variables (T : α)

-- Conditions
axiom circle_P : Circle α := ⟨X, 2⟩
axiom circle_Q : Circle α := ⟨Y, rQ⟩
axiom circle_R : Circle α := ⟨Z, rQ⟩
axiom circle_S : Circle α := ⟨W, rS⟩

-- Circle properties for tangency and radii relationships
axiom tangency_cond_1 : dist X Y = 2 + rQ
axiom tangency_cond_2 : dist X Z = 2 + rQ
axiom tangency_cond_3 : dist Y Z = 2 * rQ
axiom tangency_cond_4 : dist W X = 2
axiom tangency_cond_5 : dist W Y = 2 + rQ
axiom tangency_cond_6 : dist W Z = 2 + rQ

-- Proof problem statement
theorem radius_of_circle_Q : rQ = 16 / 9 := sorry

end radius_of_circle_Q_l777_777781


namespace no_such_hexagon_exists_l777_777316

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777316


namespace dot_product_ps_l777_777987

theorem dot_product_ps (p q r s : ℝ^3)
(hp : p.norm = 1) (hq : q.norm = 1) (hr : r.norm = 1) (hs : s.norm = 1)
(hpq : p.dot q = -1/13) (hpr : p.dot r = -1/13) (hqr : q.dot r = -1/13)
(hqs : q.dot s = -1/13) (hrs : r.dot s = -1/13) :
  p.dot s = -37/39 := 
sorry

end dot_product_ps_l777_777987


namespace abs_value_of_complex_l777_777856

theorem abs_value_of_complex : 
  ∀ (z : ℂ), z = (3 : ℝ) + (1 : ℝ) * Complex.I → Complex.abs z = Real.sqrt 10 := 
by
  intros z h
  rw h
  simp
  sorry

end abs_value_of_complex_l777_777856


namespace problem1_problem2_l777_777428

def f (α : Real) : Real := 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos (3 / 2 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) * Real.sin (Real.pi + α))

theorem problem1 : f (-13 * Real.pi / 3) = 1 / 2 :=
  sorry

theorem problem2 (α : Real) (h : Real.cos (α - Real.pi / 2) = 3 / 5) (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : f α = -4 / 5 :=
  sorry

end problem1_problem2_l777_777428


namespace sum_reciprocal_di_ge_one_l777_777434

noncomputable def di (P : ℕ → Complex) (n i : ℕ) : ℂ :=
  (∏ j in (Finset.filter (λ j, j ≠ i) (Finset.range n)), abs (P j - P i))

theorem sum_reciprocal_di_ge_one (P : ℕ → ℂ) (n : ℕ) (h_unit_circle : ∀ i, i < n → abs (P i) = 1) (h_distinct : Function.Injective P) :
  (∑ i in (Finset.range n), (1 / di P n i)) ≥ 1 :=
sorry

end sum_reciprocal_di_ge_one_l777_777434


namespace monotonic_increasing_max_min_values_l777_777863

def f(x : ℝ) := x^2 - 2 * x

theorem monotonic_increasing : ∀ x1 x2, (2 ≤ x1) → (x1 < x2) → (x2 ≤ 5) → f(x1) < f(x2) :=
by
  intros x1 x2 h1 h2 h3
  sorry

theorem max_min_values : (set.min'' (f '' set.Icc 2 5) = 0) ∧ (set.max'' (f '' set.Icc 2 5) = 15) :=
by
  split
  · sorry
  · sorry

end monotonic_increasing_max_min_values_l777_777863


namespace range_of_m_for_ellipse_l777_777512

theorem range_of_m_for_ellipse (m : ℝ) :
  (∃ x y : ℝ, x^2 / (m + 2) - y^2 / (m + 1) = 1) →
  m ∈ set.Ioo (-2 : ℝ) (-3 / 2) ∪ set.Ioo (-3 / 2) (-1) :=
by sorry

end range_of_m_for_ellipse_l777_777512


namespace no_such_hexagon_exists_l777_777346

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777346


namespace no_such_hexagon_exists_l777_777349

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777349


namespace hyperbola_eccentricity_l777_777893

variable (x y : ℝ)

def ellipse (x y : ℝ) := (x^2 / 5) + y^2 = 1

def semi_major_axis_ellipse := sqrt 5
def semi_minor_axis_ellipse := 1
def c_ellipse := sqrt (semi_major_axis_ellipse^2 - semi_minor_axis_ellipse^2)

def hyperbola (x y : ℝ) := (x^2 / 4) - y^2 = 1
def semi_major_axis_hyperbola := 2
def c_hyperbola := sqrt (semi_major_axis_hyperbola^2 + semi_minor_axis_ellipse^2)

def eccentricity_hyperbola := c_hyperbola / semi_major_axis_hyperbola

theorem hyperbola_eccentricity :
  c_ellipse = 2 ∧ c_hyperbola = sqrt 5 ∧ (c_hyperbola / semi_major_axis_hyperbola) = sqrt(5) / 2 :=
by
  sorry

end hyperbola_eccentricity_l777_777893


namespace equidistant_points_eq_two_l777_777816

noncomputable def number_of_equidistant_points (O : Point) (r d : ℝ) 
  (h1 : d > r) : ℕ := 
2

theorem equidistant_points_eq_two (O : Point) (r d : ℝ) 
  (h1 : d > r) : number_of_equidistant_points O r d h1 = 2 :=
by
  sorry

end equidistant_points_eq_two_l777_777816


namespace prove_x_minus_y_squared_l777_777495

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l777_777495


namespace probability_of_same_length_segments_l777_777952

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777952


namespace find_a_and_b_find_min_max_l777_777862

def f (a b x : ℝ) : ℝ := a * x^3 - 5 * x^2 - b * x

theorem find_a_and_b (a b : ℝ)
  (extreme_point : ∀ x, (f' = 3 * a * x^2 - 10 * x - b) → f'(3) = 0)
  (f1_eq_neg1 : f a b 1 = -1) :
  a = 1 ∧ b = -3 := sorry

theorem find_min_max (a b : ℝ)
  (h : a = 1 ∧ b = -3) :
  ∃ min max : ℝ, min = f 1 (-3) 3 ∧ max = f 1 (-3) 4 ∧ min = -9 ∧ max = -4 := sorry

end find_a_and_b_find_min_max_l777_777862


namespace no_such_hexagon_exists_l777_777360

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777360


namespace inscribed_prism_iff_l777_777661

noncomputable def isInscribedPrism (cylinder_height : ℝ) (prism_height : ℝ) (is_inscribed_polygon : Prop) : Prop :=
  prism_height = cylinder_height ∧ is_inscribed_polygon

theorem inscribed_prism_iff (cylinder_height prism_height : ℝ) (is_inscribed_polygon : Prop) :
  isInscribedPrism cylinder_height prism_height is_inscribed_polygon ↔
  (prism_height = cylinder_height ∧ is_inscribed_polygon) :=
begin
  sorry
end

end inscribed_prism_iff_l777_777661


namespace odd_n_cube_plus_one_not_square_l777_777722

theorem odd_n_cube_plus_one_not_square (n : ℤ) (h : n % 2 = 1) : ¬ ∃ (x : ℤ), x^2 = n^3 + 1 :=
by
  sorry

end odd_n_cube_plus_one_not_square_l777_777722


namespace balls_in_boxes_l777_777490

theorem balls_in_boxes :
  let balls := 7
  let boxes := 3
  let red := 1
  let indistinguishable := 2 in
  64 = 
    (choose balls balls) +
    (choose balls (balls - 1)) +
    (choose balls (balls - 2)) +
    (choose balls (balls - 3)) :=
sorry

end balls_in_boxes_l777_777490


namespace product_of_distances_correct_l777_777533

variables (P : ℝ×ℝ) (t : ℝ)

-- Define the parametric equations for curve C1
def C1 (t : ℝ) := (1 + (real.sqrt 3)/2 * t, 1 + 1/2 * t)

-- Define the polar coordinate equation for curve C2
def C2 (x y : ℝ) := x^2 + y^2 = 4

-- Fix the point P(1,1)
def P := (1, 1)

noncomputable def product_of_distances : ℝ := 
  let tA := some tA in
  let tB := some tB in
  abs (tA * tB)

theorem product_of_distances_correct:
  product_of_distances P (C1 t) (C2 (1 + (real.sqrt 3)/2 * t) (1 + 1/2 * t)) = 2 :=
sorry

end product_of_distances_correct_l777_777533


namespace plot_length_l777_777629

theorem plot_length (b : ℕ) (cost_per_meter total_cost : ℕ)
  (h1 : cost_per_meter = 2650 / 100)  -- Since Lean works with integers, use 2650 instead of 26.50
  (h2 : total_cost = 5300)
  (h3 : 2 * (b + 16) + 2 * b = total_cost / cost_per_meter) :
  b + 16 = 58 :=
by
  -- Above theorem aims to prove the length of the plot is 58 meters, given the conditions.
  sorry

end plot_length_l777_777629


namespace probability_same_length_segments_of_regular_hexagon_l777_777945

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777945


namespace complex_magnitude_l777_777055

theorem complex_magnitude (z : ℂ) (h : (complex.I : ℂ) * z = complex.sqrt 3 - complex.I) : abs z = 2 := by
  sorry

end complex_magnitude_l777_777055


namespace permutation_difference_l777_777270

-- Definitions based on the problem conditions
def P (n k : ℕ) : ℕ := (List.range k).map (λ i => n - i).prod

-- Given conditions expressed as definitions
def P_8_4 : ℕ := P 8 4
def P_8_2 : ℕ := P 8 2

-- The statement to be proven
theorem permutation_difference :
  P_8_4 - 2 * P_8_2 = 1568 := by
  sorry

end permutation_difference_l777_777270


namespace prove_x_minus_y_squared_l777_777494

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l777_777494


namespace modulus_of_complex_l777_777801

open Complex

theorem modulus_of_complex
  (Z : ℂ) (h : Z = 1 / (I - 1)) :
  |Z| = (Real.sqrt 2) / 2 :=
by
  sorry

end modulus_of_complex_l777_777801


namespace gcd_lcm_product_l777_777802

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 1350 :=
by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l777_777802


namespace sum_roots_eq_six_l777_777688

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l777_777688


namespace equation_of_line_l777_777623

theorem equation_of_line (θ : ℝ) (b : ℝ) (k : ℝ) (y x : ℝ) :
  θ = Real.pi / 4 ∧ b = 2 ∧ k = Real.tan θ ∧ k = 1 ∧ y = k * x + b ↔ y = x + 2 :=
by
  intros
  sorry

end equation_of_line_l777_777623


namespace peter_pairs_of_pants_l777_777263

-- Define the conditions
def shirt_cost_condition (S : ℕ) : Prop := 2 * S = 20
def pants_cost (P : ℕ) : Prop := P = 6
def purchase_condition (P S : ℕ) (number_of_pants : ℕ) : Prop :=
  P * number_of_pants + 5 * S = 62

-- State the proof problem:
theorem peter_pairs_of_pants (S P number_of_pants : ℕ) 
  (h1 : shirt_cost_condition S)
  (h2 : pants_cost P) 
  (h3 : purchase_condition P S number_of_pants) :
  number_of_pants = 2 := by
  sorry

end peter_pairs_of_pants_l777_777263


namespace prove_f_squared_minus_t_squared_l777_777905

def unit_circle (O : Point) := radius O = 1
def parallel (MN PQ : Line) := ∃ OR : Line, parallel MN OR ∧ parallel PQ OR
def chord_lengths (MP PQ NR MN : ℝ) (t : ℝ) := MP = t ∧ PQ = t ∧ NR = t ∧ MN = MP + PQ + NR - t
def is_chord_MN (O : Point) (MN : Line) := length MN = f

theorem prove_f_squared_minus_t_squared (O : Point) (MN PQ : Line) (MP PQ NR MN_len : ℝ) (t f : ℝ) :
  unit_circle O →
  parallel MN PQ →
  chord_lengths MP PQ NR MN_len t →
  is_chord_MN O MN → (f^2 - t^2 = 4) :=
sorry

end prove_f_squared_minus_t_squared_l777_777905


namespace neg_one_to_zero_l777_777269

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l777_777269


namespace regular_hexagon_same_length_probability_l777_777971

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777971


namespace hexagon_probability_same_length_l777_777977

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777977


namespace option_A_incorrect_option_B_correct_option_C_correct_option_D_correct_l777_777920

-- Option A: Prove the radius of the circumcircle is not 4
theorem option_A_incorrect (a : ℝ) (A : ℝ) (R : ℝ)
  (h1 : a = 2)
  (h2 : A = 30) :
  R ≠ 4 :=
sorry

-- Option B: Given the ratio condition, prove angle A is 45 degrees
theorem option_B_correct (a : ℝ) (b : ℝ) (A : ℝ) (B : ℝ)
  (h : a / cos A = b / sin B) :
  A = 45 :=
sorry

-- Option C: Prove that triangle ABC is obtuse angled
theorem option_C_correct (A B C : ℝ)
  (h : sin^2 A + sin^2 B + cos^2 C < 1) :
  ∃ C', C' > 90 :=
sorry

-- Option D: Prove the area of the triangle is as specified
theorem option_D_correct (AB AC B : ℝ) (area : ℝ)
  (h1 : AB = sqrt 3)
  (h2 : AC = 1)
  (h3 : B = 30) :
  area = sqrt 3 / 4 ∨ area = sqrt 3 / 2 :=
sorry

end option_A_incorrect_option_B_correct_option_C_correct_option_D_correct_l777_777920


namespace find_prob_l777_777460

-- Define the normal distribution with mean μ and variance σ²
variables {μ σ : ℝ}

def normal_prob_cond (ξ : ℝ → ℝ) : Prop :=
  (ξ ∼ Normal μ σ) ∧
  ((ξ > 4) = (ξ < 2)) ∧
  (P(ξ ≤ 0) = 0.2)

theorem find_prob (ξ : ℝ → ℝ) (h : normal_prob_cond ξ) :
  P(0 < ξ < 6) = 0.6 :=
by 
  sorry

end find_prob_l777_777460


namespace proof_problem_sum_result_l777_777430

open Real BigOperators

noncomputable def proof_problem (a b c k : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (abc = 1) ∧ (k ≥ 3) → 
  (∑ i in Finset.univ, (1 / (a^(k) * (b + c))) ≥ 3 / 2)

-- Formal statement written mathematically equivalent to the problem.
theorem proof_problem_sum_result :
  ∀ (a b c k : ℝ),
    proof_problem a b c k := 
  by
  intro a b c k
  rw proof_problem
  sorry -- proof omitted

end proof_problem_sum_result_l777_777430


namespace geom_seq_sum_elems_l777_777084

theorem geom_seq_sum_elems (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geom_seq_sum_elems_l777_777084


namespace inequality_proof_l777_777469

variable {a b : ℕ → ℝ}

-- Conditions: {a_n} is a geometric sequence with positive terms, {b_n} is an arithmetic sequence, a_6 = b_8
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

axiom a_pos_terms : ∀ n : ℕ, a n > 0
axiom a_geom_seq : is_geometric a
axiom b_arith_seq : is_arithmetic b
axiom a6_eq_b8 : a 6 = b 8

-- Prove: a_3 + a_9 ≥ b_9 + b_7
theorem inequality_proof : a 3 + a 9 ≥ b 9 + b 7 :=
by sorry

end inequality_proof_l777_777469


namespace quantity_of_pure_milk_after_two_dilutions_l777_777760

-- Definitions based on the given conditions
def initial_quantity_milk : ℝ := 90
def removed_quantity : ℝ := 9

-- Function representing milk quantity after each replacement
def remaining_milk (initial : ℝ) (removed : ℝ) : ℝ :=
  initial - removed

def milk_fraction (milk : ℝ) (total : ℝ) : ℝ :=
  milk / total

def milk_after_dilution (milk : ℝ) (total : ℝ) (removed : ℝ) : ℝ :=
  milk * (1 - removed / total)

-- Theorem that proves the final quantity of milk
theorem quantity_of_pure_milk_after_two_dilutions :
  let first_remaining := remaining_milk initial_quantity_milk removed_quantity in
  let second_remaining := milk_after_dilution first_remaining initial_quantity_milk removed_quantity in
  second_remaining = 72.9 :=
by
  let first_remaining := remaining_milk initial_quantity_milk removed_quantity
  let second_remaining := milk_after_dilution first_remaining initial_quantity_milk removed_quantity
  have h1 : first_remaining = 81 := by
    simp [remaining_milk, initial_quantity_milk, removed_quantity]
  have h2 : milk_fraction first_remaining initial_quantity_milk = 0.9 := by
    simp [milk_fraction, first_remaining, initial_quantity_milk]
    rw h1
    norm_num
  have h3 : second_remaining = first_remaining - first_remaining * (removed_quantity / initial_quantity_milk) := by
    simp [milk_after_dilution, first_remaining, initial_quantity_milk, removed_quantity]
  rw [h1, h2] at h3
  rw h3
  norm_num
  simp
  norm_num [initial_quantity_milk, removed_quantity, first_remaining]
  sorry

end quantity_of_pure_milk_after_two_dilutions_l777_777760


namespace no_such_convex_hexagon_and_point_exists_l777_777366

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777366


namespace no_such_hexagon_and_point_l777_777395

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777395


namespace c_finishes_work_in_18_days_l777_777050

theorem c_finishes_work_in_18_days (A B C : ℝ) 
  (h1 : A = 1 / 12) 
  (h2 : B = 1 / 9) 
  (h3 : A + B + C = 1 / 4) : 
  1 / C = 18 := 
    sorry

end c_finishes_work_in_18_days_l777_777050


namespace y_intercept_of_line_l777_777677

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l777_777677


namespace area_triangle_a0b0c0_twice_hexagon_ac1ba1cb1_l777_777535

variables {A B C A1 B1 C1 A0 B0 C0 : Type} [triangle : triangle A B C]
variables {circumcircle : circle A B C} 
variables {intersect_angle_bisector : intersection_angle_bisector A B C A1}
variables {intersect_angle_bisector : intersection_angle_bisector B A C B1}
variables {intersect_angle_bisector : intersection_angle_bisector C A B C1}
variables {line_intersections : line A1 A (exterior_angle_bisector B C) A0 B0 C0}

theorem area_triangle_a0b0c0_twice_hexagon_ac1ba1cb1 :
  area (triangle A0 B0 C0) = 2 * area (polygon [A, C1, B, A1, C, B1]) :=
sorry

end area_triangle_a0b0c0_twice_hexagon_ac1ba1cb1_l777_777535


namespace probability_of_same_length_l777_777961

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777961


namespace count_odd_values_f_up_to_100_l777_777805

def f : ℕ → ℕ
| 1        := 1
| (2 * n)  := 3 * f n + 2
| (2 * n + 1) := 2 * f n

theorem count_odd_values_f_up_to_100 : 
  (Finset.filter (λ x, (f x % 2 = 1)) (Finset.range 101)).card = 7 :=
by
  sorry

end count_odd_values_f_up_to_100_l777_777805


namespace largest_consecutive_sum_35_l777_777632

theorem largest_consecutive_sum_35 :
  ∃ n, (∃ a, (∑ i in finset.range n, (a + i)) = 35 ∧ ∀ m, m > n → (∃ a, (∑ i in finset.range m, (a + i)) ≠ 35)) ∧ n = 5 :=
sorry

end largest_consecutive_sum_35_l777_777632


namespace chalkboard_area_l777_777135

theorem chalkboard_area (width : ℝ) (h₁ : width = 3.5) (length : ℝ) (h₂ : length = 2.3 * width) : 
  width * length = 28.175 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end chalkboard_area_l777_777135


namespace arrival_times_correct_l777_777622

noncomputable section

-- Definitions based on the conditions
def distance_AF := 185
def distance_AB := 20
def distance_AC := 70
def distance_AD := 100
def distance_AE := 161
def distance_BC := distance_AC - distance_AB
def distance_CD := distance_AD - distance_AC
def distance_DE := distance_AE - distance_AD
def total_travel_time_in_hours := 142 / 60

-- Function defining the time taken including stops
def travel_time_with_stops (v : ℝ) (u_difference : ℝ) (total_distance : ℝ) (stops : List ℝ) (stop_time : ℝ): ℝ :=
  (40 / v) + (105 / (v + u_difference)) + (40 / v) + list.sum stops * stop_time

-- Speed of the train
def uphill_speed := 80
def flat_speed := uphill_speed + 10

-- Arrival times for the stations
def arrival_time_B := 8 + (15 / 60)
def arrival_time_C := 8 + (53 / 60)
def arrival_time_D := 9 + (21 / 60)
def arrival_time_E := 10 + (34 / 60)

theorem arrival_times_correct :
  ∀ (uphill_speed flat_speed : ℝ),
    flat_speed = uphill_speed + 10 ∧
    travel_time_with_stops uphill_speed 10 distance_AF [0.25, 0.25, 0.25, 0.25] (0.05) = total_travel_time_in_hours →
    (8 + (20 / uphill_speed) = arrival_time_B) ∧
    (8 + (53 / flat_speed) = arrival_time_C) ∧
    (9 + (21 / flat_speed) = arrival_time_D) ∧
    (10 + (34 / uphill_speed) = arrival_time_E) := sorry

end arrival_times_correct_l777_777622


namespace sum_of_roots_l777_777695

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l777_777695


namespace simplify_expression_l777_777164

theorem simplify_expression : 
  (2 / (Real.sqrt 2 + Real.sqrt 3)) + (Real.sqrt 2 / (4 * Real.sqrt (97 + 56 * Real.sqrt 3))) = (3 * Real.sqrt 2) / 4 :=
by
  have h₁ : Real.sqrt (97 + 56 * Real.sqrt 3) = 7 + 4 * Real.sqrt 3, sorry
  calc
    -- Steps will go here, skipped as "sorry"
    _ = (3 * Real.sqrt 2) / 4 := sorry

end simplify_expression_l777_777164


namespace solve_for_x_l777_777114

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem solve_for_x (x : ℝ) : star 6 x = 45 ↔ x = 19 / 3 := by
  sorry

end solve_for_x_l777_777114


namespace seashells_given_l777_777156

theorem seashells_given (original_seashells : ℕ) (current_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 35) 
  (h2 : current_seashells = 17) 
  (h3 : given_seashells = original_seashells - current_seashells) : 
  given_seashells = 18 := 
by 
  sorry

end seashells_given_l777_777156


namespace no_such_hexagon_exists_l777_777326

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777326


namespace interest_rate_of_first_account_l777_777217

theorem interest_rate_of_first_account (r : ℝ) 
  (h1 : 7200 = 4000 + 4000)
  (h2 : 4000 * r = 4000 * 0.10) : 
  r = 0.10 :=
sorry

end interest_rate_of_first_account_l777_777217


namespace sequence_becomes_constant_l777_777715

def sequence_constant_eventually (n : ℕ) : Prop :=
  ∃ N, ∀ k ≥ N, ∃ b, ∀ m ≥ k, a m = b

def a : ℕ → ℕ
| 0     := 0
| (k+1) := sorry -- Define as stated in the problem, but implementation not needed for the statement

theorem sequence_becomes_constant (n : ℕ) (h : 0 < n) :
  sequence_constant_eventually n :=
sorry

end sequence_becomes_constant_l777_777715


namespace sum_roots_eq_six_l777_777692

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l777_777692


namespace frog_friendship_probability_l777_777935
open BigOperators

-- Definitions of the problem conditions:
def num_frogs : ℕ := 16

def friendship_probability : ℚ := 1 / 2

-- Statement of the problem:
theorem frog_friendship_probability :
  ∃ (a b : ℕ), (∀ i : fin num_frogs, (set.filter (λ j : fin num_frogs, i ≠ j ∧ frog_are_friends i j) finset.univ).card % 4 = 0) →
    gcd a b = 1 ∧
    (a, b) = (1167, 2 ^ 41) ∧
    (a : ℚ) / b = friendship_probability :=
sorry

-- Define the friendship relation
noncomputable def frog_are_friends (i j : fin num_frogs) : Prop := sorry

end frog_friendship_probability_l777_777935


namespace gabriel_forgot_days_l777_777424

def days_in_july : ℕ := 31
def days_taken : ℕ := 28

theorem gabriel_forgot_days : days_in_july - days_taken = 3 := by
  sorry

end gabriel_forgot_days_l777_777424


namespace num_zeros_of_g_l777_777475

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then |real.log x| else 2^|x|

def g (x : ℝ) : ℝ :=
2 * (f x)^2 - 3 * (f x) + 1

theorem num_zeros_of_g : 
  (∀ y, g y = 0 → ∃ x, f x = y) ∧
  (∃ l : list ℝ, l.length = 5 ∧ ∀ z ∈ l, g z = 0) :=
begin
  sorry
end

end num_zeros_of_g_l777_777475


namespace mandy_book_length_l777_777132

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end mandy_book_length_l777_777132


namespace maximize_projection_area_theorem_l777_777491

noncomputable def maximize_projection_area (a b c : ℝ) :=
  maximize (λ θ φ : ℝ, projection_area a b c θ φ)

axiom projection_area: ℝ → ℝ → ℝ → ℝ → ℝ → ℝ

theorem maximize_projection_area_theorem (a b c : ℝ) :
  let diagonal_length := real.sqrt (a^2 + b^2 + c^2) in
  let condition := ∃ θ φ : ℝ, 
                    diagonal_length = 
                    ∥projection_area a b c θ φ∥ in
  maximize_projection_area a b c = condition := sorry

end maximize_projection_area_theorem_l777_777491


namespace sum_inverse_distance_product_l777_777431

noncomputable def point_on_unit_circle (i : ℕ) (n : ℕ) [fact (i < n)] : ℂ :=
  by sorry  -- Definition placeholder for n distinct points on the unit circle

noncomputable def distance_product (i : ℕ) (n : ℕ) : ℝ :=
  let points := λ j, point_on_unit_circle j n in
  (finset.range n).product (λ j, if j ≠ i then abs (points j - points i) else 1)

theorem sum_inverse_distance_product (n : ℕ) (hn : 0 < n) : 
  ∑ i in finset.range n, 1 / distance_product i n >= 1 :=
by sorry  -- Proof placeholder for the theorem


end sum_inverse_distance_product_l777_777431


namespace hyperbola_range_a_l777_777819

theorem hyperbola_range_a (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (eqn : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) (focus : ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) :
  (∀ m : ℝ, m = Real.tan (π / 3) → ∀ x y : ℝ, x = 4 → x^2 / a^2 - y^2 / b^2 = 1 → y = m * (x - 4) → (x, y) = (4, 0)) → 0 < a ∧ a ≤ 2 :=
by
  skip_proof sorry

end hyperbola_range_a_l777_777819


namespace similar_triangles_l777_777995

variables {A B C D M : Type*} [circle A B C D]
variables [is_collinear A B M]
variables [is_collinear C D M]
variables [is_external M (circle A B C D)]

theorem similar_triangles (h1 : cyclic_quad A B C D)
                          (h2 : line (A, B) ∩ line (C, D) = M)
                          (h3 : is_collinear A B M)
                          (h4 : is_collinear C D M)
                          (h5 : is_external M (circle A B C D))
                          : similar (triangle M A C) (triangle M D B) := 
sorry

end similar_triangles_l777_777995


namespace distance_between_parallel_lines_l777_777411

noncomputable def distance_parallel_lines : ℝ :=
  let a : ℝ := 5
  let b : ℝ := 12
  let m : ℝ := 3
  let n : ℝ := 5
  abs(m - n) / real.sqrt(a^2 + b^2)

theorem distance_between_parallel_lines :
  distance_parallel_lines = 2 / 13 :=
by
  sorry

end distance_between_parallel_lines_l777_777411


namespace difference_between_a_b_l777_777466

theorem difference_between_a_b (a b : ℝ) (d : ℝ) : 
  (a - b = d) → (a ^ 2 + b ^ 2 = 150) → (a * b = 25) → d = 10 :=
by
  sorry

end difference_between_a_b_l777_777466


namespace choose_supermarket_80_choose_supermarket_130_actual_amount_paid_discount_rate_B_not_necessarily_increase_l777_777769

-- Definitions for supermarkets' discount policies
def supermarket_A_discount (x : ℕ) : ℕ := x * 4 / 5
def supermarket_B_discount (x : ℕ) : ℕ :=
  if x < 100 then x else x - 30

-- Part 1: Supermarket choice for specific shopping amounts
def choose_supermarket (x : ℕ) : string :=
  if x = 80 then "A"
  else if x = 130 then "B"
  else "check amount"

-- Part 2: Analytical expressions for actual paid amount
def actual_amount_paid_A (x : ℕ) : ℕ := supermarket_A_discount x
def actual_amount_paid_B (x : ℕ) : ℕ := supermarket_B_discount x

-- Part 3: Discount rate comparison example
def discount_rate_A (x : ℕ) : ℕ := 20
def discount_rate_B (x : ℕ) : ℕ :=
  if x = 100 then 30
  else if x = 120 then 25
  else 0 -- General case/hypothetical

-- Propositions to prove
theorem choose_supermarket_80 : choose_supermarket 80 = "A" := by sorry
theorem choose_supermarket_130 : choose_supermarket 130 = "B" := by sorry

theorem actual_amount_paid (x : ℕ) (h : 0 ≤ x ∧ x < 200) :
  ∃ y_A y_B, y_A = actual_amount_paid_A x ∧ 
             y_B = actual_amount_paid_B x ∧
             ((x < 100 ∧ y_A < y_B) ∨ 
              (100 ≤ x ∧ x < 150 ∧ y_B < y_A) ∨ 
              (x = 150 ∧ y_A = y_B) ∨ 
              (150 < x ∧ x < 200 ∧ y_A < y_B)) := by sorry

theorem discount_rate_B_not_necessarily_increase : discount_rate_B 100 = 30 ∧ discount_rate_B 120 = 25 := by sorry

end choose_supermarket_80_choose_supermarket_130_actual_amount_paid_discount_rate_B_not_necessarily_increase_l777_777769


namespace find_v1013_l777_777175

def g : ℕ → ℕ :=
| 1 := 5
| 2 := 3
| 3 := 4
| 4 := 1
| 5 := 2
| _ := 0

def v : ℕ → ℕ
| 0 := 3
| (n+1) := g (v n)

theorem find_v1013 : v 1013 = 5 :=
by
  -- Placeholder to skip the proof
  sorry

end find_v1013_l777_777175


namespace sum_of_roots_of_quadratic_l777_777702

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l777_777702


namespace no_convex_hexagon_with_point_M_l777_777311

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777311


namespace sandy_correct_sums_l777_777157

theorem sandy_correct_sums :
  ∃ x y : ℕ, x + y = 30 ∧ 3 * x - 2 * y = 60 ∧ x = 24 :=
by
  sorry

end sandy_correct_sums_l777_777157


namespace lakers_win_nba_finals_in_7_games_l777_777616

noncomputable def nba_finals_probability_lakers_win_in_7_games : ℚ :=
  let p_lakers_win := 1 / 3 in
  let p_celtics_win := 2 / 3 in
  let comb_6_3 := Nat.choose 6 3 in
  let prob_6_games := comb_6_3 * (p_lakers_win ^ 3 * p_celtics_win ^ 3) in
  let prob_7th_game := p_lakers_win in
  prob_6_games * prob_7th_game

theorem lakers_win_nba_finals_in_7_games :
  nba_finals_probability_lakers_win_in_7_games = 160 / 2187 := sorry

end lakers_win_nba_finals_in_7_games_l777_777616


namespace no_such_hexagon_and_point_l777_777390

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777390


namespace unit_digit_of_expression_is_six_l777_777088

noncomputable def expression (a : ℝ) : ℝ :=
  ( (sqrt ((a - 2) * (abs a - 1)) + sqrt ((a - 2) * (1 - abs a))) / (1 + 1 / (1 - a)) + (5 * a + 1) / (1 - a) )^1988

def is_valid_a (a : ℝ) : Prop :=
  (a - 2) * (abs a - 1) ≥ 0 ∧ (a - 2) * (1 - abs a) ≥ 0

def unit_digit (n : ℕ) : ℕ :=
  n % 10

theorem unit_digit_of_expression_is_six (a : ℝ) (h_valid_a : is_valid_a a) :
  unit_digit (expression a).to_nat = 6 :=
sorry

end unit_digit_of_expression_is_six_l777_777088


namespace perfect_square_octal_last_digit_l777_777531

theorem perfect_square_octal_last_digit (a b c : ℕ) (n : ℕ) (h1 : a ≠ 0) (h2 : (abc:ℕ) = n^2) :
  c = 1 :=
sorry

end perfect_square_octal_last_digit_l777_777531


namespace problem_f3_is_neg2_l777_777179

theorem problem_f3_is_neg2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (1 + x) = -f (1 - x)) (h3 : f 1 = 2) : f 3 = -2 :=
sorry

end problem_f3_is_neg2_l777_777179


namespace parabola_correct_options_l777_777187

theorem parabola_correct_options (A B C : ℝ × ℝ) (F : ℝ × ℝ) (hF : F = (1,0)) :
  centroid F A B C →
  (abs (dist F A) + abs (dist F B) + abs (dist F C) = 6) ∧
  (∀ P : ℝ × ℝ, (P.1)^2 = 4 * P.2 → (dist P F = 4 → dist P (0, 0) = real.sqrt 21)) := 
sorry

end parabola_correct_options_l777_777187


namespace no_such_hexagon_exists_l777_777333

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777333


namespace seq_max_min_terms_l777_777191

noncomputable def a (n: ℕ) : ℝ := 1 / (2^n - 18)

theorem seq_max_min_terms : (∀ (n : ℕ), n > 5 → a 5 > a n) ∧ (∀ (n : ℕ), n ≠ 4 → a 4 < a n) :=
by 
  sorry

end seq_max_min_terms_l777_777191


namespace sculptor_requires_4_blocks_l777_777751

-- Define the conditions using Lean definitions
def cylinder_radius : ℝ := 2.5
def cylinder_height : ℝ := 9
def cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * cylinder_height

def block_length : ℝ := 8
def block_width : ℝ := 3
def block_height : ℝ := 2
def block_volume : ℝ := block_length * block_width * block_height

-- The main theorem statement
theorem sculptor_requires_4_blocks : 
  ⌈(cylinder_volume / block_volume)⌉ = 4 := 
by 
  sorry

end sculptor_requires_4_blocks_l777_777751


namespace y_intercept_3x_minus_4y_eq_12_l777_777673

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l777_777673


namespace letters_contain_straight_line_only_l777_777522

theorem letters_contain_straight_line_only
  (both : ℕ)
  (dot_only : ℕ)
  (total : ℕ)
  (dot_straight_line : both = 11)
  (only_dot : dot_only = 5)
  (total_letters : total = 40)
  : ∃ (X : ℕ), X = 24 := by
   have h1 : both + dot_only = 16 := by rw [dot_straight_line, only_dot]; norm_num
   have h2 : total = 16 + (X + 11) - 11 := by rw [total_letters]; norm_num
   have h3 : 16 + X = 40 := by linarith
   exact ⟨24, by linarith⟩.

end letters_contain_straight_line_only_l777_777522


namespace focus_of_parabola_l777_777800

def parabola_focus (a k : ℕ) : ℚ :=
  1 / (4 * a) + k

theorem focus_of_parabola :
  parabola_focus 9 6 = 217 / 36 :=
by
  sorry

end focus_of_parabola_l777_777800


namespace radius_of_circle_Q_l777_777779

theorem radius_of_circle_Q 
  (externally_tangent : ∀ (P Q R S : Circle), tangent_outside P Q ∧ tangent_outside Q R ∧ tangent_outside R P ∧ tangent_inside S P ∧ tangent_inside S Q ∧ tangent_inside S R)
  (Q_R_congruent : circle_Q.radius = circle_R.radius)
  (P_radius : circle_P.radius = 2)
  (P_center_in_S : S_center ∈ circle_P) :
  circle_Q.radius = 16 / 9 := 
sorry

end radius_of_circle_Q_l777_777779


namespace intersection_of_intervals_l777_777057

theorem intersection_of_intervals :
  let A := {x : ℝ | x < -3}
  let B := {x : ℝ | x > -4}
  A ∩ B = {x : ℝ | -4 < x ∧ x < -3} :=
by
  sorry

end intersection_of_intervals_l777_777057


namespace midpoints_intersect_l777_777737

variables {A B C D E F M1 M2 M3 M4 M5 M6 M7 M8 : Type} [AffineSpace ℝ]

-- Define hexagon and midpoints
def Hexagon (A B C D E F : ℝ) : Prop := 
  -- Add your conditions for a convex hexagon here
    
def Midpoint (x y m : ℝ) : Prop :=
  m = (x + y) / 2

def segments_intersect (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) : Prop :=
  -- Add the proper intersection checking here

theorem midpoints_intersect (A B C D E F M1 M2 M3 M4 M5 M6 M7 M8 : ℝ)
  (hHex : Hexagon A B C D E F)
  (hM1 : Midpoint A B M1)
  (hM2 : Midpoint C D M2)
  (hM3 : Midpoint M1 M2 M3)
  (hM4 : Midpoint E F M4)
  (hM5 : Midpoint A F M5)
  (hM6 : Midpoint B C M6)
  (hM7 : Midpoint M5 M6 M7)
  (hM8 : Midpoint D E M8) :
  segments_intersect M3 M4 M7 M8 :=
begin
  sorry
end

end midpoints_intersect_l777_777737


namespace slope_proof_l777_777681

noncomputable def slope_between_midpoints : ℚ :=
  let p1 := (2, 3)
  let p2 := (4, 5)
  let q1 := (7, 3)
  let q2 := (8, 7)

  let midpoint (a b : ℚ × ℚ) : ℚ × ℚ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  let m1 := midpoint p1 p2
  let m2 := midpoint q1 q2

  (m2.2 - m1.2) / (m2.1 - m1.1)

theorem slope_proof : slope_between_midpoints = 2 / 9 := by
  sorry

end slope_proof_l777_777681


namespace triangle_areas_l777_777922

theorem triangle_areas (A B C O A_1 A_2 : Point)
  (h1 : dist A B = 3)
  (h2 : dist A C = 4)
  (h3 : angle A B C = 60 * π / 180)
  (circumcenter : circumcenter A B C = O)
  (angle_bisector : ∃ A_1, is_angle_bisector A A_1 (segment B C))
  (intersect : ∃ A_2, line A A_1 ∧ on_circle A_2 (circumcircle A B C))
  : area (triangle O A_2 C) = 13 / (4 * √3) ∧ area (triangle A_1 A_2 C) = 13 / (7 * √3) :=
sorry

end triangle_areas_l777_777922


namespace elizabeth_net_profit_is_50_l777_777401

variables (ingredient_cost : ℕ) (bags_made : ℕ) (price_per_bag : ℕ)
variables (bags_sold_initial : ℕ) (discounted_bags : ℕ) (discounted_price : ℕ)
variable total_cost : ℕ := bags_made * ingredient_cost
variable revenue_initial : ℕ := bags_sold_initial * price_per_bag
variable revenue_discounted : ℕ := discounted_bags * discounted_price
variable total_revenue : ℕ := revenue_initial + revenue_discounted
variable net_profit : ℕ := total_revenue - total_cost

theorem elizabeth_net_profit_is_50 :
  ingredient_cost = 3 ∧ bags_made = 20 ∧ price_per_bag = 6 ∧ 
  bags_sold_initial = 15 ∧ discounted_bags = 5 ∧ discounted_price = 4 →
  net_profit = 50 :=
by 
  intros h, 
  cases h with h1 h2, 
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end elizabeth_net_profit_is_50_l777_777401


namespace shaded_region_area_eq_l777_777254

noncomputable def areaShadedRegion : ℝ :=
  let side_square := 14
  let side_triangle := 18
  let height := 14
  let H := 9 * Real.sqrt 3
  let BF := (side_square + side_triangle, height - H)
  let base_BF := BF.1 - 0
  let height_BF := BF.2
  let area_triangle_BFH := 0.5 * base_BF * height_BF
  let total_triangle_area := 0.5 * side_triangle * height
  let area_half_BFE := 0.5 * total_triangle_area
  area_half_BFE - area_triangle_BFH

theorem shaded_region_area_eq :
  areaShadedRegion = 9 * Real.sqrt 3 :=
by 
 sorry

end shaded_region_area_eq_l777_777254


namespace probability_same_length_segments_l777_777938

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777938


namespace f_zero_add_f_neg_three_l777_777997

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_add (x y : ℝ) : f x + f y = f (x + y)

axiom f_three : f 3 = 4

theorem f_zero_add_f_neg_three : f 0 + f (-3) = -4 :=
by
  sorry

end f_zero_add_f_neg_three_l777_777997


namespace stone_board_configuration_l777_777290

theorem stone_board_configuration (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h_even : even m → n ≥ m - 1) (h_odd : odd m → n ≥ m) :
    (∃ stones : ℕ → ℕ, (∀ i, i < m → stones i ≤ n) ∧
                         (∀ i j, i < m → j < m → i ≠ j → stones i ≠ stones j) ∧
                         (∀ c, c < n → ∃ t, ∀ r, r < m → stones r = t)) ↔ (even m → n ≥ m - 1) ∧ (odd m → n ≥ m) :=
sorry

end stone_board_configuration_l777_777290


namespace calculate_time_after_duration_l777_777273

def start_time : String := "2023-01-03T03:00:00"
def duration_minutes : Int := 1950
def expected_time : String := "2023-01-04T11:30:00"

theorem calculate_time_after_duration :
  calculate_time start_time duration_minutes = expected_time :=
sorry

end calculate_time_after_duration_l777_777273


namespace find_line_through_p_intersecting_c_find_circle_through_p_tangent_to_c_l777_777472

theorem find_line_through_p_intersecting_c (l : ℝ → ℝ) :
  (∀ x y, x^2 + y^2 + 2*x - 6*y + 5 = 0) → 
  (l(-2) = -1) → 
  (∃ (a b : ℝ), ∀ x, l(x) = a*x + b) → 
  (∀ x y : ℝ, l(x)^2 + y^2 + 2*l(x) - 6*y + 5 = 0 ∧ |l(1) - l(5)| = 4) → 
  l = (λ x : ℝ, 15/8 * x + 11/4) ∨ l = (λ x : ℝ, -2) := 
by {
  sorry -- Proof omitted
}

theorem find_circle_through_p_tangent_to_c (M : ℝ → ℝ → ℝ) :
  (∀ x y, x^2 + y^2 + 2*x - 6*y + 5 = 0) → 
  (M(-2, -1) = 0) → 
  (M(0, 1) = 0) → 
  (∃ (D E F : ℝ), ∀ x y, x^2 + y^2 + D*x + E*y + F = 0 → M(x, y) = 0) → 
  (∃ (R : ℝ), ∀ x y, M(x, y) = 0 → (sqrt (x^2 + y^2) = R)) → 
  (M(0, 1) = 0 ∧ 2*x + y - 1 = 0) →
  (∀ x y, M(x, y) = x^2 + y^2 - 4*x + 6*y - 7) :=
by {
  sorry -- Proof omitted
}

end find_line_through_p_intersecting_c_find_circle_through_p_tangent_to_c_l777_777472


namespace books_given_away_l777_777600

theorem books_given_away (initial_books books_sold books_left books_given : ℕ) 
    (h1 : initial_books = 134)
    (h2 : books_sold = 27)
    (h3 : books_left = 68)
    (h4 : initial_books - books_sold - books_given = books_left) : 
    books_given = 39 := 
by
  -- We define the values to help Lean understand the context
  have hb : initial_books = (134 : ℕ) := h1
  have hs : books_sold = (27 : ℕ) := h2
  have hl : books_left = (68 : ℕ) := h3
  -- Subtract the sold books from the initial books
  have hx : initial_books - books_sold = (107 : ℕ) := by simp [hb, hs]
  -- Subtract the remaining books from the books after selling
  have hy : 107 - books_given = books_left := h4
  -- Using the facts we know
  rw [hl, hy]
  simp [hx]
  sorry

end books_given_away_l777_777600


namespace complement_union_eq_l777_777480

open Set

variable (U M N : Set ℕ)
variable hU : U = {1, 2, 3, 4, 5, 6}
variable hM : M = {2, 3, 5}
variable hN : N = {4, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {1, 6} :=
by
  sorry

end complement_union_eq_l777_777480


namespace triangle_area_l777_777542

/-- In triangle ABC, with side BC = 24, 
  given that the incircle trisects the median AD, 
  the area of the triangle can be expressed as 128 * sqrt 16, 
  thus p + q = 144 where p and q are integers, 
  and q is not divisible by the square of a prime. -/
theorem triangle_area {A B C : Point} (BC : dist B C = 24)
  (median_trisection : ∃ D, is_median AD ∧ incircle_trisects_median AD):
  ∃ (p q : ℕ), 
    (area_of_triangle ABC = p * real.sqrt q) ∧
    (¬ ∃ r, prime r ∧ r * r ∣ q) ∧
    (p + q = 144) :=
sorry

end triangle_area_l777_777542


namespace triangle_identity_l777_777546

variable (a b c : ℝ)
variable (cosA cosB cosC : ℝ)
variable (A B C : ℝ)

axiom angle_opposite_to_side (triangle_ABC : ℝ → ℝ → ℝ → Prop) (A B C : ℝ) (a b c : ℝ) :
  angle_opposite_to_side A a triangle_ABC ∧ angle_opposite_to_side B b triangle_ABC ∧ angle_opposite_to_side C c triangle_ABC

theorem triangle_identity
  (triangle_ABC : ℝ → ℝ → ℝ → Prop)
  (A B C : ℝ)
  (a b c : ℝ)
  (angle_opposite_to_side : ∀ (A B C a b c : ℝ), angle_opposite_to_side A a triangle_ABC ∧ angle_opposite_to_side B b triangle_ABC ∧ angle_opposite_to_side C c triangle_ABC)
  (cos_law_A : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cosA)
  (cos_law_B : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * cosB)
  (cos_law_C : c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * cosC)
  : (a ^ 2 + b ^ 2 + c ^ 2 = 2 * (b * c * cosA + a * c * cosB + a * b * cosC)) :=
sorry

end triangle_identity_l777_777546


namespace solution_set_of_f_x_gt_2_minimum_value_of_f_l777_777584

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solution_set_of_f_x_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7} ∪ {x : ℝ | x > 5 / 3} :=
by 
  sorry

theorem minimum_value_of_f : ∃ x : ℝ, f x = -9 / 2 :=
by 
  sorry

end solution_set_of_f_x_gt_2_minimum_value_of_f_l777_777584


namespace closest_desert_function_relation_l777_777061

noncomputable def f_a (x : ℕ) : ℝ := 0.2 * x
noncomputable def f_b (x : ℕ) : ℝ := 2^x / 10
noncomputable def f_c (x : ℕ) : ℝ := (x^2 + 2 * x) / 10
noncomputable def f_d (x : ℕ) : ℝ := 0.2 + Real.log (x) / Real.log (16)

theorem closest_desert_function_relation :
    ∀ x ∈ {1, 2, 3}, abs (f_b x - desert_increase x) < abs (f_a x - desert_increase x) ∧
                      abs (f_b x - desert_increase x) < abs (f_c x - desert_increase x) ∧
                      abs (f_b x - desert_increase x) < abs (f_d x - desert_increase x) := 
by 
  -- Proof is omitted
  sorry

-- Auxiliary function for desert increase
noncomputable def desert_increase (x : ℕ) : ℝ :=
  if x = 1 then 0.2 else if x = 2 then 0.4 else if x = 3 then 0.76 else 0

end closest_desert_function_relation_l777_777061


namespace no_such_hexagon_exists_l777_777297

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777297


namespace range_of_f_l777_777177

def floor (x : ℝ) : ℤ := ⌊x⌋ -- Lean syntax for floor function.
def f (x : ℝ) : ℝ := (floor (2 * x) : ℝ) - 2 * x -- Defining the function f.

theorem range_of_f : Set.Ioc (-1 : ℝ) 0 = ∃ y : ℝ, f y :=
begin
  sorry,
end

end range_of_f_l777_777177


namespace sin_alpha_value_l777_777470

noncomputable def terminal_side_condition (α : ℝ) : Prop :=
  let P := (-1, -2*real.sqrt 2)
  let r := real.sqrt (1 + 8)
  let sin_alpha_plus_pi_over_6 := -2*real.sqrt 2 / r
  let cos_alpha_plus_pi_over_6 := -1 / r
  sin (α + real.pi / 6) = sin_alpha_plus_pi_over_6 ∧
  cos (α + real.pi / 6) = cos_alpha_plus_pi_over_6

theorem sin_alpha_value (α : ℝ) (h : terminal_side_condition α) :
  sin α = (1 - 2 * real.sqrt 6) / 6 :=
by sorry

end sin_alpha_value_l777_777470


namespace Rohit_initial_south_distance_l777_777610

theorem Rohit_initial_south_distance (d : ℝ) :
  ∃ d, (let final_pos := (λ d, (20 + 15, 0)) ∘ (λ d, (20, 0)) ∘ (λ d, (20, -d)) ∘ (λ d, (0, -d))
  in final_pos d = (35, 0)) → d = 20 :=
begin
  sorry
end

end Rohit_initial_south_distance_l777_777610


namespace map_C_D_by_180_deg_rotation_l777_777215

def C := (-3, 2)
def C' := (3, -2)
def D := (-2, 5)
def D' := (2, -5)

def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (λ (x y : ℝ), (-x, -y)) p.1 p.2

theorem map_C_D_by_180_deg_rotation :
  rotate_180_deg C = C' ∧ rotate_180_deg D = D' :=
sorry

end map_C_D_by_180_deg_rotation_l777_777215


namespace square_difference_l777_777498

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l777_777498


namespace no_convex_hexagon_with_point_M_l777_777314

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777314


namespace length_of_other_wood_pieces_l777_777167

theorem length_of_other_wood_pieces (total_wood : ℕ) (wood_needed_6_pieces : ℕ) (other_wood_lengths : ℕ) : 
  (6 * 4 = wood_needed_6_pieces) → (total_wood = 28) → (total_wood - wood_needed_6_pieces = 4) :=
by
  intro h1 h2
  have h3 : wood_needed_6_pieces = 24 :=
    by rw [h1]
  rw [h3, h2]
  norm_num
  sorry

end length_of_other_wood_pieces_l777_777167


namespace ab_zero_l777_777831

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l777_777831


namespace cos_phi_of_square_midpoints_l777_777232

def is_square (A B C D : ℝ × ℝ) : Prop :=
  (A.1 = 0 ∧ A.2 = 0) ∧
  (B.1 = 0 ∧ B.2 = 4) ∧
  (C.1 = 4 ∧ C.2 = 4) ∧
  (D.1 = 4 ∧ D.2 = 0)

def is_midpoint (X Y Z : ℝ × ℝ) : Prop :=
  (Z.1 = (X.1 + Y.1) / 2) ∧ (Z.2 = (X.2 + Y.2) / 2)

theorem cos_phi_of_square_midpoints :
  ∀ (A B C D P Q : ℝ × ℝ),
    is_square A B C D →
    is_midpoint A B P →
    is_midpoint B C Q →
    ∃ φ : ℝ, cos φ = 9 / 10 :=
by
  sorry

end cos_phi_of_square_midpoints_l777_777232


namespace prob_arithmetic_sequence_prob_geometric_sequence_expected_value_xi_l777_777601

-- 1. Probability that x, y, z form an arithmetic sequence when n = 3
theorem prob_arithmetic_sequence (n x y z : ℕ) (h_n : n = 3) (h_sum : x + y + z = n) :
    (2 * y = x + z) → 
    (∃! p, p = 5 / 108) :=
sorry

-- 2. Probability that x, y, z form a geometric sequence when n = 6
theorem prob_geometric_sequence (n x y z : ℕ) (h_n : n = 6) (h_sum : x + y + z = n) :
    (y^2 = x * z) → 
    (∃! p, p = 1 / 1296) :=
sorry

-- 3. Expected value of the absolute difference in number of balls between Box A and Box B when n = 4
theorem expected_value_xi (n : ℕ) (A B : ℕ) (h_n : n = 4) :
    (Eξ = (|A - B| : real)) → 
    (∃! e, e ≈ 1.01) :=
sorry

end prob_arithmetic_sequence_prob_geometric_sequence_expected_value_xi_l777_777601


namespace equivalent_problem_l777_777087

noncomputable theory

open Real

-- Definitions given in conditions
def polar_eq (theta a : ℝ) : ℝ := 2 * (sin theta) + 2 * a * (cos theta)

def parametric_x (t : ℝ) : ℝ := -2 + (sqrt 2) / 2 * t
def parametric_y (t : ℝ) : ℝ := (sqrt 2) / 2 * t

variables (a : ℝ) (h₁ : a > 0) 

-- Definitions deduced from the solution
def curve_c_eq (x y a : ℝ) : Prop := (x - a)^2 + (y - 1)^2 = a^2 + 1
def line_l_eq (x y : ℝ) : Prop := y = x + 2

def P := (-2 : ℝ, 0 : ℝ)
def PM (M : ℝ × ℝ) := sqrt ((M.fst - P.fst)^2 + (M.snd - P.snd)^2)
def PN (N : ℝ × ℝ) := sqrt ((N.fst - P.fst)^2 + (N.snd - P.snd)^2)

theorem equivalent_problem :
  (curve_c_eq x y a) ∧ 
  (line_l_eq x y) ∧ 
  (PM M + PN N = 5 * sqrt 2) → a = 2 := 
sorry

end equivalent_problem_l777_777087


namespace find_ab_l777_777834

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l777_777834


namespace b_50_l777_777823

-- Define the sequence and T_n
def seq (n : ℕ) : ℤ := sorry -- b_n, to be defined later via conditions

def T (n : ℕ) : ℤ := (list.range (n+1)).sum (seq ∘ nat.succ)

-- Given conditions
axiom b1 : seq 1 = 1

axiom b_rec (n : ℕ) (hn : 2 ≤ n) : seq n = 3 * (T n)^2 / (3 * (T n) - 2)

-- Theorem to prove
theorem b_50 : seq 50 = -3 / 21460 := sorry

end b_50_l777_777823


namespace prime_divides_2_pow_n_minus_n_infinte_times_l777_777148

theorem prime_divides_2_pow_n_minus_n_infinte_times (p : ℕ) (hp : Nat.Prime p) : ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end prime_divides_2_pow_n_minus_n_infinte_times_l777_777148


namespace cos_alpha_sub_pi_over_4_sin_square_half_alpha_sin4alpha_cos2alpha_l777_777844

theorem cos_alpha_sub_pi_over_4 (α : ℝ) (hα : α ∈ Ioo (π / 2) π) (h_sin : Real.sin α = 4 / 5) : 
  Real.cos (α - π / 4) = Real.sqrt 2 / 10 :=
sorry

theorem sin_square_half_alpha_sin4alpha_cos2alpha (α : ℝ) (hα : α ∈ Ioo (π / 2) π) (h_sin : Real.sin α = 4 / 5) : 
  Real.sin(α / 2)^2 + (Real.sin(4 * α) * Real.cos(2 * α)) / (1 + Real.cos(4 * α)) = -8 / 25 :=
sorry

end cos_alpha_sub_pi_over_4_sin_square_half_alpha_sin4alpha_cos2alpha_l777_777844


namespace determinant_new_matrix_eq_2D_l777_777564

-- Define the vectors a, b, c in an appropriate vector space
variables {V : Type*} [inner_product_space ℝ V]
variable (a b c : V)

-- Define the determinant D as given in the problem
def D : ℝ := ⟪a, b × c⟫

-- Main statement: Prove that the determinant of the new matrix equals 2D
theorem determinant_new_matrix_eq_2D :
  let a' := (2:ℝ) • a,
      b' := b + c,
      c' := c + a
  in ⟪a', b' × c'⟫ = 2 * D :=
by
  sorry

end determinant_new_matrix_eq_2D_l777_777564


namespace probability_of_same_length_l777_777964

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777964


namespace max_a_monotonic_decreasing_l777_777001

theorem max_a_monotonic_decreasing {a : ℝ} :
  (∀ x ≥ 1, f(x) = a * x - x^3 ∧ f'(x) = a - 3 * x^2 ∧ f'(x) ≤ 0) → a ≤ 3 :=
begin
  sorry
end

end max_a_monotonic_decreasing_l777_777001


namespace solve_for_p_l777_777477

variables (a b p : ℝ) (x y k : ℝ)

-- Conditions from the problem
axiom a_pos : a = 2 * sqrt 2
axiom b_pos : b = p / 2
axiom p_pos : p > 0
axiom hyperbola : x^2 / a^2 - y^2 / b^2 = 1
axiom parabola : x^2 = 2 * p * y
axiom line : k = p / (4 * sqrt 2),
-- line y = kx - 1 is tangent to the parabola
axiom line_tangent : ∃ y x, y = k * x - 1 ∧ x^2 = 2 * p * y

-- Goal: Prove p = 4
theorem solve_for_p : p = 4 :=
sorry

end solve_for_p_l777_777477


namespace points_per_basket_l777_777136

variable (M S B : ℕ)
variable (total_points : ℕ)
variable (p : ℕ)

-- Conditions
def matthew_points : Prop := M = 9
def shawn_points : Prop := S = 6
def total_baskets_made : Prop := B = 5
def points_sum : Prop := total_points = M + S

-- Theorem statement to be proved
theorem points_per_basket (hM : matthew_points) (hS : shawn_points) (hB : total_baskets_made) (hT : points_sum) : p = total_points / B := by
  sorry

end points_per_basket_l777_777136


namespace no_such_hexagon_exists_l777_777356

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777356


namespace class_performance_proof_l777_777730

-- Define the performance records
def performance_records : List Int := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]

-- Define the conditions
def standard_jumps : Int := 160
def points_above : Int := 1
def points_below : Float := 0.5
def reward_threshold : Float := 65.0

-- Theorem stating the proof problem
theorem class_performance_proof :
  let best_performance := performance_records.maximum?.getOrElse 0;
  let worst_performance := performance_records.minimum?.getOrElse 0;
  let difference := best_performance - worst_performance;
  let total_adjustments := performance_records.sum;
  let average_jumps := standard_jumps + total_adjustments / List.length performance_records;
  let points_above_total := performance_records.filter(λ x => x > 0).sum;
  let points_below_total := (performance_records.filter(λ x => x < 0).map(λ x => Float.ofInt (-x)).sum) * points_below;
  let total_score := Float.ofInt points_above_total - points_below_total;
  difference = 27 ∧ average_jumps = standard_jumps + 6 ∧ total_score > reward_threshold := 
by
  -- Placeholder for the actual proof.
  sorry

end class_performance_proof_l777_777730


namespace sum_first_10_terms_l777_777643

theorem sum_first_10_terms {a : ℕ → ℕ} (h₀ : a 1 = 1) 
  (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) - a n = n + 1) : 
  ∑ i in Finset.range 10, (1 / a (i + 1) : ℚ) = 20 / 11 := sorry

end sum_first_10_terms_l777_777643


namespace mandy_book_length_l777_777133

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end mandy_book_length_l777_777133


namespace student_marks_l777_777754

def max_marks : ℕ := 600
def passing_percentage : ℕ := 30
def fail_by : ℕ := 100

theorem student_marks :
  ∃ x : ℕ, x + fail_by = (passing_percentage * max_marks) / 100 :=
sorry

end student_marks_l777_777754


namespace fib_mod_5_l777_777015

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := (fib n) + (fib (n+1))

theorem fib_mod_5 (k : ℕ) (h : k > 0) : 5 ∣ fib (5 * k) :=
by
  sorry

end fib_mod_5_l777_777015


namespace ribbon_deficit_l777_777934

-- Definitions based on the conditions
def total_ribbon_available : ℕ := 18
def gifts_count : ℕ := 6
def ribbon_per_gift : ℕ := 2
def bows_count : ℕ := gifts_count
def ribbon_per_bow : ℝ := 1.5

-- Lean statement for the math proof problem
theorem ribbon_deficit : 
  let total_required_ribbon_gifts := gifts_count * ribbon_per_gift in
  let total_required_ribbon_bows := bows_count * ribbon_per_bow in
  let total_required_ribbon := total_required_ribbon_gifts + total_required_ribbon_bows in
  total_required_ribbon > total_ribbon_available
:= 
by
  let total_required_ribbon_gifts := gifts_count * ribbon_per_gift in
  let total_required_ribbon_bows := bows_count * ribbon_per_bow in
  let total_required_ribbon := total_required_ribbon_gifts + total_required_ribbon_bows in
  have total_required_ribbon = 21, from sorry,
  show total_required_ribbon > total_ribbon_available, from sorry

end ribbon_deficit_l777_777934


namespace no_such_hexagon_exists_l777_777350

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777350


namespace no_such_hexagon_and_point_l777_777386

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777386


namespace quadrilateral_inscribed_circle_parallel_l777_777077

variables {α : Type*} [euclidean_geometry α]

/-- Given a quadrilateral inscribed in a circle, where diagonals intersect at points M and N, and an angle condition, prove parallelism. -/
theorem quadrilateral_inscribed_circle_parallel (A B C D O M N K : α)
  (h_circle : ∀ P Q, P, Q ∈ circle O → (P - O).length = (Q - O).length)
  (h_insc : ∀ P, P ∈ {A, B, C, D} → P ∈ circle O)
  (h_int_diag1 : ∃ R, line AC ∩ line BD = {R} ∧ R = M)
  (h_int_diag2 : ∃ S, line AD ∩ line BC = {S} ∧ S = N)
  (h_angle : ∠ A N B = 90)
  (h_reflect : K = reflection O (line AB)) :
  parallel (line O M) (line K N) := 
  sorry

end quadrilateral_inscribed_circle_parallel_l777_777077


namespace initial_loss_percentage_l777_777258

theorem initial_loss_percentage 
  (CP : ℝ := 250) 
  (SP : ℝ) 
  (h1 : SP + 50 = 1.10 * CP) : 
  (CP - SP) / CP * 100 = 10 := 
sorry

end initial_loss_percentage_l777_777258


namespace sheets_for_given_height_l777_777238

theorem sheets_for_given_height (sheets_in_bundle : ℕ) (thickness_bundle : ℝ) (desired_height : ℝ) :
  sheets_in_bundle = 400 →
  thickness_bundle = 4 →
  desired_height = 6 →
  let thickness_per_sheet := thickness_bundle / sheets_in_bundle in
  desired_height / thickness_per_sheet = 600 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  have thickness_per_sheet := 4 / 400 : ℝ
  have desired_sheets   := 6 / (4 / 400) : ℝ
  show desired_sheets = 600
  calc desired_sheets = 6 / (4 / 400) : by sorry
                  ... = 6 * (400 / 4)   : by sorry
                  ... = 6 * 100        : by sorry
                  ... = 600            : by sorry

end sheets_for_given_height_l777_777238


namespace quadratic_solution_a_plus_b2_l777_777165

theorem quadratic_solution_a_plus_b2 :
  (∃ a b : ℂ, (λ x : ℂ, 3 * x^2 - 6 * x + 11 = 0) a) →
  (∀ a b : ℂ, a = 1 ∧ b = 2 * real.sqrt 6 / 3 → a + b^2 = 11 / 3) :=
by
  sorry

end quadratic_solution_a_plus_b2_l777_777165


namespace original_price_proof_l777_777749

-- Let's define the conditions for this problem
def original_price (P : ℝ) : Prop :=
  let S := 1.35 * P in
  let Profit := S - P in
  Profit = 1080

-- Now we state the theorem
theorem original_price_proof : ∃ P : ℝ, original_price P ∧ P = 3085.71 :=
by
  -- This proof is omitted; this is a statement only.
  sorry

end original_price_proof_l777_777749


namespace negation_proposition_p_l777_777013

open Classical

variable (n : ℕ)

def proposition_p : Prop := ∃ n : ℕ, 2^n > 100

theorem negation_proposition_p : ¬ proposition_p ↔ ∀ n : ℕ, 2^n ≤ 100 := 
by sorry

end negation_proposition_p_l777_777013


namespace monopoly_produces_durable_competitive_market_prefers_durable_l777_777784

-- Define the conditions
structure EconomicContext (periods : Nat) where
  benefit_per_period : Real
  durable_cost : Real
  low_quality_cost : Real

def monopoly_condition (S : Real) : Prop :=
  28 > 2 * (20 - S)

def competitive_condition (S : Real) : Prop :=
  28 > 2 * 20 - 2 * S

-- Define the proof problem
theorem monopoly_produces_durable {S : Real} (context : EconomicContext 2) :
  context.benefit_per_period = 20 ∧ context.durable_cost = 12 →
  monopoly_condition S → S < 6 :=
begin
  intros h ctx_cond,
  sorry -- This is where the proof would go
end

theorem competitive_market_prefers_durable {S : Real} (context : EconomicContext 2):
  context.benefit_per_period = 20 ∧ context.durable_cost = 12 →
  competitive_condition S → S > 6 :=
begin
  intros h ctx_cond,
  sorry -- This is where the proof would go
end

end monopoly_produces_durable_competitive_market_prefers_durable_l777_777784


namespace min_value_of_diff_l777_777008

noncomputable def f (x : ℝ) : ℝ :=
  sin x * sin (x + π / 3) - 1 / 4

theorem min_value_of_diff {m n : ℝ} (h : m < n) (h_f_range : ∀ x ∈ set.Icc m n, f x ∈ set.Icc (-1 / 2) (1 / 4)) :
  n - m = 2 * π / 3 :=
sorry

end min_value_of_diff_l777_777008


namespace sweets_neither_red_nor_green_l777_777195

theorem sweets_neither_red_nor_green (total_sweets : ℕ) (red_sweets : ℕ) (green_sweets : ℕ) 
  (h_total : total_sweets = 285) (h_red : red_sweets = 49) (h_green : green_sweets = 59) :
  total_sweets - (red_sweets + green_sweets) = 177 :=
by 
  rw [h_total, h_red, h_green]
  sorry

end sweets_neither_red_nor_green_l777_777195


namespace correct_statement_a_l777_777707

theorem correct_statement_a (x y : ℝ) (h : x + y < 0) : x^2 - y > x :=
sorry

end correct_statement_a_l777_777707


namespace probability_of_same_length_l777_777963

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777963


namespace regular_hexagon_same_length_probability_l777_777975

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777975


namespace probability_of_same_length_l777_777965

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777965


namespace more_than_half_millet_l777_777091

def initial_millet := 0.25
def initial_other := 0.75
def millet_increase_interval := 0.05
def max_millet := 0.50

-- Function to calculate millet added on a given day
def millet_added (day : ℕ) : ℝ :=
  if day <= 5 then initial_millet + (millet_increase_interval * (day - 1))
  else max_millet

-- Function to calculate total millet after birds' consumption
def total_millet (day : ℕ) : ℝ :=
  if day = 1 then initial_millet
  else total_millet (day - 1) * 0.7 + millet_added day

-- Function to calculate the total seed content
def total_seeds (day : ℕ) : ℝ := day

theorem more_than_half_millet : ∃ n ∈ (ℕ : Type), total_millet n / total_seeds n > 0.5 :=
by 
  sorry

end more_than_half_millet_l777_777091


namespace smaller_number_approx_l777_777650

noncomputable def a (x : ℝ) : ℝ := 7 * x
noncomputable def b (x : ℝ) : ℝ := 11 * x

theorem smaller_number_approx (x : ℝ) (hx : x = (135 + Real.sqrt 1841) / 32) : 
  a x ≈ 38.92 :=
by
  sorry

end smaller_number_approx_l777_777650


namespace count_int_values_in_range_l777_777027

theorem count_int_values_in_range : 
  ∃ (n_values : Finset ℤ), 
    (∀ n ∈ n_values, -100 < n^3 ∧ n^3 < 100) ∧ 
    n_values.card = 9 :=
begin
  sorry
end

end count_int_values_in_range_l777_777027


namespace center_of_square_l777_777575

theorem center_of_square (O : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (hAB : dist A B = 1) 
  (hA : A = (0, 0)) 
  (hB : B = (1, 0)) 
  (hC : C = (1, 1)) 
  (hD : D = (0, 1)) 
  (h_sum_squares : (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = 2): 
  O = (1/2, 1/2) :=
by sorry

end center_of_square_l777_777575


namespace inequality_problem_l777_777039

-- Given a < b < 0, we want to prove a^2 > ab > b^2
theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
sorry

end inequality_problem_l777_777039


namespace square_difference_l777_777497

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l777_777497


namespace no_four_distinct_integers_with_product_plus_2006_perfect_square_l777_777768

theorem no_four_distinct_integers_with_product_plus_2006_perfect_square : 
  ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ k1 k2 k3 k4 k5 k6 : ℕ, a * b + 2006 = k1^2 ∧ 
                          a * c + 2006 = k2^2 ∧ 
                          a * d + 2006 = k3^2 ∧ 
                          b * c + 2006 = k4^2 ∧ 
                          b * d + 2006 = k5^2 ∧ 
                          c * d + 2006 = k6^2) := 
sorry

end no_four_distinct_integers_with_product_plus_2006_perfect_square_l777_777768


namespace total_pencils_is_5_l777_777194

-- Define the initial number of pencils and the number of pencils Tim added
def initial_pencils : Nat := 2
def pencils_added_by_tim : Nat := 3

-- Prove the total number of pencils is equal to 5
theorem total_pencils_is_5 : initial_pencils + pencils_added_by_tim = 5 := by
  sorry

end total_pencils_is_5_l777_777194


namespace conditional_probability_l777_777237

/- Conditions for the problem. -/
variables {Ω : Type} [ProbabilitySpace Ω]

-- There are 4 products, 3 first-class and 1 second-class
def num_products : ℕ := 4
def num_first_class : ℕ := 3
def num_second_class : ℕ := 1

-- Events A and B
def event_A (ω : Ω) : Prop := first_draw_first_class ω
def event_B (ω : Ω) : Prop := second_draw_first_class ω

-- Probabilities
def P (p : Prop) [decidable p] : ℝ := probability_measure p

-- The goal is to find P(B|A), which is the conditional probability of event B given event A.
theorem conditional_probability (hA : P event_A > 0) :
  P (event_B ∧ event_A) / P event_A = 2 / 3 :=
sorry

end conditional_probability_l777_777237


namespace number_of_polynomials_with_condition_l777_777415

theorem number_of_polynomials_with_condition : 
  ∃ (P : ℕ → ℂ) (n : ℕ), 
  (∑ i in finset.range (n + 1), |P i| + n = 4) ∧ 
  (number_of_such_polynomials P n = 16) :=
begin
  sorry
end

end number_of_polynomials_with_condition_l777_777415


namespace area_of_T_prime_l777_777565

noncomputable def area_of_transformed_region {T : Type} (area_T : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
|det A| * area_T

theorem area_of_T_prime :
  let area_T := 8
  let A := ![![3, 0], ![5, 4]] : Matrix (Fin 2) (Fin 2) ℝ
  area_of_transformed_region area_T A = 96 :=
by
  sorry

end area_of_T_prime_l777_777565


namespace symmedian_length_l777_777407

theorem symmedian_length (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ AS : ℝ, AS = (b * c^2 / (b^2 + c^2)) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) :=
sorry

end symmedian_length_l777_777407


namespace quadratic_eq_complete_square_l777_777249

theorem quadratic_eq_complete_square (x p q : ℝ) (h : 9 * x^2 - 54 * x + 63 = 0) 
(h_trans : (x + p)^2 = q) : p + q = -1 := sorry

end quadratic_eq_complete_square_l777_777249


namespace first_player_wins_l777_777063

def winning_strategy (m n : ℕ) : Prop :=
  if m = 1 ∧ n = 1 then false else true

theorem first_player_wins (m n : ℕ) :
  winning_strategy m n :=
by
  sorry

end first_player_wins_l777_777063


namespace probability_same_length_segments_of_regular_hexagon_l777_777944

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777944


namespace range_of_m_l777_777861

-- Definitions and the main problem statement
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ (-4 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l777_777861


namespace correct_average_weight_l777_777221

def average_weight_correct (
  n : ℕ, -- The number of boys in the class
  initial_avg_weight : ℚ, -- The initial average weight calculated in kg
  misread_weight : ℚ, -- The misread weight in kg
  actual_weight : ℚ -- The actual weight in kg
) : ℚ := sorry -- The correct average weight

theorem correct_average_weight :
  let n := 20
  let initial_avg_weight := 58.4
  let misread_weight := 56
  let actual_weight := 68
  average_weight_correct n initial_avg_weight misread_weight actual_weight = 59 :=
  sorry

end correct_average_weight_l777_777221


namespace number_of_terms_in_ap_is_eight_l777_777185

theorem number_of_terms_in_ap_is_eight
  (n : ℕ) (a d : ℝ)
  (even : n % 2 = 0)
  (sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 24)
  (sum_even : (n / 2 : ℝ) * (2 * a + n * d) = 30)
  (last_exceeds_first : (n - 1) * d = 10.5) :
  n = 8 :=
by sorry

end number_of_terms_in_ap_is_eight_l777_777185


namespace problem1_problem2_l777_777278

-- Define the first calculation
def calc1 : ℝ := 
  (Real.sqrt 27 * Real.sqrt 2 / Real.sqrt (2 / 3)) - 
  ((Real.sqrt 12 + 3 * Real.sqrt 6) * Real.sqrt 3)

-- Define the second calculation
def calc2 : ℝ := 
  (3 - Real.sqrt 2) ^ 2 - 
  (Real.sqrt 5 + Real.sqrt 3) * 
  (Real.sqrt 5 - Real.sqrt3)

-- Prove the first calculation equals 3 - 9 * sqrt(2)
theorem problem1 : calc1 = 3 - 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the second calculation equals 9 - 6 * sqrt(2)
theorem problem2 : calc2 = 9 - 6 * Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l777_777278


namespace proof_x_minus_y_squared_l777_777503

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l777_777503


namespace intersection_of_M_N_equals_0_1_open_interval_l777_777125

def M : Set ℝ := { x | x ≥ 0 }
def N : Set ℝ := { x | x^2 < 1 }

theorem intersection_of_M_N_equals_0_1_open_interval :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } := 
sorry

end intersection_of_M_N_equals_0_1_open_interval_l777_777125


namespace gmat_problem_l777_777734

theorem gmat_problem 
  (B : ℝ) (NoQ: ℝ) (AB : ℝ)
  (hB : B = 0.65)
  (hNoQ : NoQ = 0.05)
  (hAB : AB = 0.55) :
  ∃ A : ℝ, A = 0.85 := 
by
  -- Define the percentage of at least one correct answer
  let OneCorrect := 1 - NoQ
  -- Use inclusion-exclusion principle
  let A := OneCorrect - (B - AB)
  -- Substitute the values to get A = 0.85
  have hA : A = 1 - 0.05 - (0.65 - 0.55), from sorry
  -- Verify that A = 0.85
  exact ⟨A, hA⟩

end gmat_problem_l777_777734


namespace no_such_convex_hexagon_exists_l777_777378

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777378


namespace jon_fluid_intake_per_week_l777_777932

theorem jon_fluid_intake_per_week :
  ( ∀ (h1 : Nat := 16),
      (h2 : Nat := 4),
      (b1 : Nat := 16),
      (f1 : Nat := 16 * (1 + 25 / 100)),
      (d1 : Nat := 2),
      (w1 : Nat := 7),
      fluid_intake_per_week : Nat /- 728 ounces per week -/)
   → fluid_intake_per_week =
     ((h1 / h2 * b1) + (f1 * d1)) * w1 :=
begin
  intros,
  sorry,
end

end jon_fluid_intake_per_week_l777_777932


namespace probability_of_at_least_one_three_l777_777242

noncomputable def is_valid_roll (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 8

noncomputable def valid_rolls : List ℕ := [1,2,3,4,5,6,7,8]

noncomputable def at_least_one_three (rolls : List ℕ) : Prop := 3 ∈ rolls

theorem probability_of_at_least_one_three :
  ∀ (X1 X2 X3 X4 : ℕ),
    is_valid_roll X1 →
    is_valid_roll X2 →
    is_valid_roll X3 →
    is_valid_roll X4 →
    (X1 + X2 + X3 = X4) →
    (∃ n : ℚ, n = (5:ℚ) / 12) :=
  sorry

end probability_of_at_least_one_three_l777_777242


namespace problem_statement_l777_777065

/-- Given a geometric sequence where the first term a₁ = 2 and the fourth term a₄ = 16,
  and another sequence bₙ defined as log₁₀(aₙ), prove that bₙ is an arithmetic sequence and 
  find the sum of the first n terms. -/
theorem problem_statement (n : ℕ) (h1 : ∀ n, a n = 2 ^ n) (h2 : ∀ n, b n = log 10 (a n)) :
    (T n) = n * (n + 1) / 2 := 
    sorry

end problem_statement_l777_777065


namespace solveForT_l777_777612

theorem solveForT (t : ℝ) : 4 * 4^t + real.sqrt (16 * 16^t) = 40 → t = 1.5 :=
by
  intro h
  sorry

end solveForT_l777_777612


namespace value_of_f_neg_four_thirds_l777_777585

noncomputable def f : ℝ → ℝ 
| x := if 0 < x then Real.cos (π * x) else f (x + 1) - 1

theorem value_of_f_neg_four_thirds : f (- 4 / 3) = -5 / 2 :=
  sorry

end value_of_f_neg_four_thirds_l777_777585


namespace part1_part2_l777_777992

-- Define the functions and conditions
def f (a x : ℝ) : ℝ := -a * x + 1
def g (x : ℝ) : ℝ := x^2

-- Part 1: Prove given the condition
theorem part1 (a : ℝ) (h : ∀ x ∈ set.Ici (2:ℝ), f a x / g x > 0) : a ∈ set.Iic 0 := sorry

-- Part 2: Prove the conditions collectively implies the different range of a
theorem part2 (a : ℝ) 
  (h1 : ∀ x ∈ set.Ici (2:ℝ), f a x + g x > 0)
  (h2 : ∀ x ∈ set.Ici (2:ℝ), f a x - g x > 0)
  (h3 : ∀ x ∈ set.Ici (2:ℝ), f a x * g x > 0)
  (h4 : ∀ x ∈ set.Ici (2:ℝ), f a x / g x > 0) : a ∈ set.Ioo 0 (5/2) := 
  sorry

end part1_part2_l777_777992


namespace range_of_a_l777_777429

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → (x < a ∨ x > a + 4)) ∧ ¬(∀ x : ℝ, (x < a ∨ x > a + 4) → -2 ≤ x ∧ x ≤ 1) ↔
  a > 1 ∨ a < -6 :=
by {
  sorry
}

end range_of_a_l777_777429


namespace compute_difference_a_b_l777_777793

-- Define the initial amounts paid by Alex, Bob, and Carol
def alex_paid := 120
def bob_paid := 150
def carol_paid := 210

-- Define the total amount and equal share
def total_costs := alex_paid + bob_paid + carol_paid
def equal_share := total_costs / 3

-- Define the amounts Alex and Carol gave to Bob, satisfying their balances
def a := equal_share - alex_paid
def b := carol_paid - equal_share

-- Lean 4 statement to prove a - b = 30
theorem compute_difference_a_b : a - b = 30 := by
  sorry

end compute_difference_a_b_l777_777793


namespace no_such_hexagon_exists_l777_777304

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777304


namespace product_expression_equals_fraction_l777_777271

theorem product_expression_equals_fraction :
  ∏ n in (Finset.range 11).map (Function.Embedding.add 2), (1 - (1 / (n^2))) = 13 / 24 :=
by
  sorry

end product_expression_equals_fraction_l777_777271


namespace find_age_l777_777720

-- Define the age variables
variables (P Q : ℕ)

-- Define the conditions
def condition1 : Prop := (P - 3) * 3 = (Q - 3) * 4
def condition2 : Prop := (P + 6) * 6 = (Q + 6) * 7

-- Prove that, given the conditions, P equals 15
theorem find_age (h1 : condition1 P Q) (h2 : condition2 P Q) : P = 15 :=
sorry

end find_age_l777_777720


namespace min_set_satisfying_conditions_l777_777398

open Int

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def satisfies_conditions (S : Finset ℕ) : Prop :=
  (∀ a ∈ S, a ≤ 100) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, gcd a c = 1 ∧ gcd b c = 1) ∧
  (∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ gcd a b > 1 ∧ gcd b d > 1)

theorem min_set_satisfying_conditions : ∃ S : Finset ℕ, satisfies_conditions S ∧ S.card = 14 :=
by
  -- Proof omitted
  sorry

end min_set_satisfying_conditions_l777_777398


namespace max_true_statements_l777_777570

theorem max_true_statements (c d : ℝ) :
  (¬(1 / c < 1 / d) ∧ c^3 > d^3 ∧ c < d ∧ c < 0 ∧ d < 0) → 
  (∃ S : set (Prop), S = {¬(1 / c < 1 / d), c^3 > d^3, c < d, c < 0, d < 0} ∧ S.card = 4) :=
by
  sorry

end max_true_statements_l777_777570


namespace incorrect_value_in_polynomial_progression_l777_777532

noncomputable def polynomial_values (x : ℕ) : ℕ :=
  match x with
  | 0 => 1
  | 1 => 9
  | 2 => 35
  | 3 => 99
  | 4 => 225
  | 5 => 441
  | 6 => 784
  | 7 => 1296
  | _ => 0  -- This is a dummy value just to complete the function

theorem incorrect_value_in_polynomial_progression :
  ¬ (∃ (a b c d : ℝ), ∀ x : ℕ,
    polynomial_values x = (a * x ^ 3 + b * x ^ 2 + c * x + d + if x ≤ 7 then 0 else 1)) :=
by
  intro h
  sorry

end incorrect_value_in_polynomial_progression_l777_777532


namespace determine_p_l777_777915

theorem determine_p (p x1 x2 : ℝ) 
  (h_eq : ∀ x, x^2 + p * x + 3 = 0)
  (h_root_relation : x2 = 3 * x1)
  (h_vieta1 : x1 + x2 = -p)
  (h_vieta2 : x1 * x2 = 3) :
  p = 4 ∨ p = -4 := 
sorry

end determine_p_l777_777915


namespace probability_of_same_length_segments_l777_777959

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777959


namespace hexagon_probability_same_length_l777_777979

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777979


namespace red_side_probability_l777_777728

noncomputable def num_cards  := 6
noncomputable def num_black_black := 3
noncomputable def num_black_red := 1
noncomputable def num_red_red := 2

-- Defining the probability function
noncomputable def probability_red_side (total_red_sides : ℕ) (desired_red_sides : ℕ) : ℚ :=
  desired_red_sides / total_red_sides

-- The probability that the other side is red, given that one side is red
theorem red_side_probability :
  let total_red_sides := (num_red_red * 2) + 1 in
  let desired_red_sides := num_red_red * 2 in
  probability_red_side total_red_sides desired_red_sides = 4 / 5 := by
  sorry

end red_side_probability_l777_777728


namespace min_value_a_plus_one_over_a_minus_one_l777_777886

theorem min_value_a_plus_one_over_a_minus_one (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ (a = 2 → a + 1 / (a - 1) = 3) :=
by
  -- Translate the mathematical proof problem into a Lean 4 theorem statement.
  sorry

end min_value_a_plus_one_over_a_minus_one_l777_777886


namespace dilation_example_l777_777621

-- Define the dilation function
def dilation (center scale: ℂ) (point: ℂ) : ℂ :=
  center + scale * (point - center)

-- Prove that dilation of -2 - 2i centered at 1 + 2i with scale factor 4 is -11 - 14i
theorem dilation_example : dilation (1 + 2 * complex.I) 4 (-2 - 2 * complex.I) = -11 - 14 * complex.I := 
  sorry

end dilation_example_l777_777621


namespace max_value_of_f_l777_777631

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 3) * cos x

open Real

theorem max_value_of_f :
  ∀ x ∈ Icc (π / 6) (π / 3), f x ≤ 0 ∧ ∃ y ∈ Icc (π / 6) (π / 3), f y = 0 := by
  sorry

end max_value_of_f_l777_777631


namespace no_such_hexagon_exists_l777_777298

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777298


namespace octahedron_volume_from_unit_cube_l777_777280

theorem octahedron_volume_from_unit_cube :
  ∀ (s : ℝ), s = 1 → 
  volume_of_octahedron_formed_by_centers_of_unit_cube_faces s = 1 / 3 :=
by
  sorry

noncomputable def volume_of_octahedron_formed_by_centers_of_unit_cube_faces (s : ℝ) : ℝ :=
  let base_area := s * s
  let height := s / 2
  2 * (1 / 3) * base_area * height

end octahedron_volume_from_unit_cube_l777_777280


namespace probability_of_e_theorem_l777_777080

noncomputable def probability_of_selecting_e : ℚ :=
  2 / 7

theorem probability_of_e_theorem (letters: list ℕ) (h_len: letters.length = 7) 
  (h_e: letters.count 2 = 2): probability_of_selecting_e = 2 / 7 :=
by sorry

end probability_of_e_theorem_l777_777080


namespace solve_problem_l777_777081

noncomputable def area_segment (a c : ℝ) : ℝ := 1/2 * a * (a + c)
noncomputable def radius (r a : ℝ) (c : ℝ) : ℝ := r
noncomputable def area_circle (r : ℝ) : ℝ := real.pi * r ^ 2
noncomputable def probability (S1 S2 : ℝ) : ℝ := S1 / S2

def problem_statement : Prop :=
  let c := 6 in
  let a := 1 in
  let S1 := area_segment a c in
  let r := 5 in
  let S2 := area_circle r in
  let P := probability S1 S2 in
  P = 7 / 150

theorem solve_problem : problem_statement := sorry

end solve_problem_l777_777081


namespace smallest_positive_period_2π_minimum_value_on_interval_l777_777006

def f (x : ℝ) : ℝ := 4 * sin (x / 2) * sin (x / 2 + π / 6) + 2 * sqrt 3 * (cos x - 1)

theorem smallest_positive_period_2π : (∀ x, f (x + 2 * π) = f x ∧ ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 2 * π) := 
by sorry

theorem minimum_value_on_interval : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * π / 3 → f x = -sqrt 3 
:= 
by sorry

end smallest_positive_period_2π_minimum_value_on_interval_l777_777006


namespace kevin_found_cards_l777_777099

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end kevin_found_cards_l777_777099


namespace sum_of_roots_l777_777694

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l777_777694


namespace find_pattern_in_sequence_l777_777924

-- Define the initial sequence and its transformation
def sequence : List Nat := [111, 213, 141, 516, 171, 819, 202, 122]

-- Define the corresponding simpler sequence
def simple_sequence (n : Nat) : Nat := 10 + n

-- Assertion that needs proof
theorem find_pattern_in_sequence :
  ∃ f : Nat → Nat, ∀ n : Nat, n < sequence.length → f (simple_sequence n) = sequence.nth_le n (sorry : n < sequence.length) :=
sorry

end find_pattern_in_sequence_l777_777924


namespace minimum_n_minus_m_l777_777010

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_l777_777010


namespace probability_same_length_segments_of_regular_hexagon_l777_777946

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777946


namespace no_such_hexagon_and_point_l777_777392

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777392


namespace concyclic_NA_I₁I₂_l777_777984

variables (A B C : Point) (Γ : Circle)
variables (N M : Point)
variables (I₁ I₂ : Point)

-- Assume that A, B, and C form a triangle inscribed in circle Γ.
def triangle_inscribed (A B C : Point) (Γ : Circle) : Prop :=
  A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ

-- N is the midpoint of the arc BC containing A
def midpoint_arc (N A B C : Point) (Γ : Circle) : Prop :=
  N ∈ Γ ∧ arc_contains (Γ, N) (B, C) A

-- M is the midpoint of segment BC.
def midpoint_segment (M B C : Point) : Prop :=
  midpoint M B C

-- I₁ is the incenter of triangle BAM.
def incenter (I : Point) (A B M : Point) : Prop :=
  incenter_of I (triangle A B M)

-- I₂ is the incenter of triangle CAM.
def incenter' (I : Point) (A C M : Point) : Prop :=
  incenter_of I (triangle A C M)

-- The final statement: points N, A, I₁, and I₂ are concyclic.
theorem concyclic_NA_I₁I₂ (A B C : Point) (Γ : Circle) (N M : Point) (I₁ I₂ : Point) :
  triangle_inscribed A B C Γ →
  midpoint_arc N A B C Γ →
  midpoint_segment M B C →
  incenter I₁ A B M →
  incenter' I₂ A C M →
  concyclic N A I₁ I₂ :=
by sorry

end concyclic_NA_I₁I₂_l777_777984


namespace total_hamburger_varieties_l777_777485

def num_condiments : ℕ := 9
def num_condiment_combinations : ℕ := 2 ^ num_condiments
def num_patties_choices : ℕ := 4
def num_bread_choices : ℕ := 2

theorem total_hamburger_varieties : num_condiment_combinations * num_patties_choices * num_bread_choices = 4096 :=
by
  -- conditions
  have h1 : num_condiments = 9 := rfl
  have h2 : num_condiment_combinations = 2 ^ num_condiments := rfl
  have h3 : num_patties_choices = 4 := rfl
  have h4 : num_bread_choices = 2 := rfl

  -- correct answer
  sorry

end total_hamburger_varieties_l777_777485


namespace find_number_l777_777213

theorem find_number (x : ℕ) : (x + 12) / 4 = 12 ∧ (x + 12) % 4 = 3 → x = 39 :=
by
  -- condition
  intro h,
  have h₁ : (x + 12) / 4 = 12, from h.left,
  have h₂ : (x + 12) % 4 = 3, from h.right,
  -- the proof would go here, skipped with sorry
  sorry

end find_number_l777_777213


namespace angle_MHB_proof_l777_777518

-- Definitions of the given conditions
def triangle (A B C : Type) := { angle_A angle_B angle_C : ℝ // angle_A + angle_B + angle_C = 180 }

def is_altitude {P Q R : Type} (A : P) (H : Q) (B C : R) := True -- to be defined concretely
def is_median {P Q R : Type} (C : P) (M : Q) (A B : R) := True -- to be defined concretely

-- Variables corresponding to the vertices and points in the problem
variables {A B C H M : Type}

-- Given values
def triangle_ABC := triangle A B C
def angle_A := 80 : ℝ
def angle_B := 60 : ℝ
def angle_C := 40 : ℝ
def altitude_AH := is_altitude A H B C
def median_CM := is_median C M A B

-- The theorem statement
theorem angle_MHB_proof : triangle_ABC → altitude_AH → median_CM → ∠MHB = 30 :=
by
  intro triangle_ABC altitude_AH median_CM
  sorry

end angle_MHB_proof_l777_777518


namespace analytical_expression_of_f_max_value_of_m_l777_777005

noncomputable def f (x : ℝ) (ω : ℝ) (a : ℝ) : ℝ :=
  2 * (cos (ω * x))^2 + 2 * √3 * sin (ω * x) * cos (ω * x) + a

theorem analytical_expression_of_f
    (hω : 0 < ω) 
    (ha : a = -2)
    (h_max : ∀ x, 2 * (cos (ω * x))^2 + 2 * √3 * sin (ω * x) * cos (ω * x) + a ≤ 1)
    (h_T : ∀ x, f x ω a = f (x + (π / ω)) ω a) :
    f = λ x, 2 * sin(2 * x + π / 6) - 1 :=
  sorry

noncomputable def g (x : ℝ) : ℝ :=
  f (2 * x - π / 6) 1 -2

theorem max_value_of_m
    (h_min_g : ∀ m, m ∈ [0, π / 3] → min (g 0) (g m) = g 0) :
    ∀ m, m ≤ π / 3 :=
  sorry

end analytical_expression_of_f_max_value_of_m_l777_777005


namespace ratio_cp_pa_l777_777919

theorem ratio_cp_pa {A B C D M P : Type} 
    (hABC : Triangle A B C)
    (hAB : dist A B = 20)
    (hAC : dist A C = 11)
    (hAangleBisector : ∃ D, angle_bisector A ∩ line B C = D)
    (hADMidpoint : M = midpoint A D)
    (hPint : ∃ P, line A C ∩ line B M = P)
    (hSimilarTriangles : ∀ D', parallel (line B P) (line D D') → similar_triangles (Triangle B P C) (Triangle D D' C))
    (hAngleBisectorThm : ∀ {D : Type}, is_angle_bisector A B C D → dist B D / dist D C = dist A B / dist A C) :
    let m := 31
    let n := 20
    m + n = 51 :=
by
  sorry

end ratio_cp_pa_l777_777919


namespace chess_tournament_third_place_wins_l777_777524

theorem chess_tournament_third_place_wins :
  ∀ (points : Fin 8 → ℕ)
  (total_games : ℕ)
  (total_points : ℕ),
  (total_games = 28) →
  (∀ i j : Fin 8, i ≠ j → points i ≠ points j) →
  ((points 1) = (points 4 + points 5 + points 6 + points 7)) →
  (points 2 > points 4) →
  ∃ (games_won : Fin 8 → Fin 8 → Prop),
  (games_won 2 4) :=
by
  sorry

end chess_tournament_third_place_wins_l777_777524


namespace pipe_rate_ratio_l777_777145

theorem pipe_rate_ratio :
  let RA := (1:ℝ) / 12
  let RB := ((1:ℝ) / 3) / 12
  RB / RA = (1:ℝ) / 3 :=
by
  let RA := (1:ℝ) / 12
  let RB := ((1:ℝ) / 3) / 12
  have h1 : RA = (1:ℝ) / 12 := rfl
  have h2 : RB = (1:ℝ) / 36 := by
    calc
      ((1:ℝ) / 3) / 12 = (1:ℝ) / 3 * 1 / 12  := by ring
      ... = (1:ℝ) / 36 := by ring
  calc
    RB / RA = (1:ℝ) / 36 / (1:ℝ) / 12 := rfl
    ... = (1:ℝ) / 3 := by ring

end pipe_rate_ratio_l777_777145


namespace no_such_convex_hexagon_exists_l777_777382

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777382


namespace log_expression_value_l777_777462

theorem log_expression_value (x : ℝ) (hx : x < 1) (h : (Real.log x / Real.log 10)^3 - 2 * (Real.log (x^3) / Real.log 10) = 150) :
  (Real.log x / Real.log 10)^4 - (Real.log (x^4) / Real.log 10) = 645 := 
sorry

end log_expression_value_l777_777462


namespace problem_l777_777818

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^2 - 2 else if x > 0 then -(2 * (-x)^2 - 2) else 0

theorem problem (x : ℝ) (h1 : ∀ x : ℝ, f x + f (-x) = 0) (h2 : ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 2) : 
  f (f (-1)) + f 2 = -6 :=
by
  sorry

end problem_l777_777818


namespace walk_back_to_starting_point_l777_777034

-- Define a prime condition
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the walking steps function
def walk_steps (n : ℕ) : ℤ :=
  if is_prime n then 2 else -2

-- Define the number of steps taken from move 2 to n
def total_steps (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, if i + 2 ≥ 2 ∧ i + 2 ≤ 30 then walk_steps (i + 2) else 0

-- Theorem statement
theorem walk_back_to_starting_point : total_steps 29 = -16 := by
  sorry

end walk_back_to_starting_point_l777_777034


namespace complex_number_in_third_quadrant_l777_777461

def i : Complex := ⟨0, 1⟩

theorem complex_number_in_third_quadrant :
  (let z := ((1 - i) ^ 2) / (1 + i) in
   z = -1 - i ∧ z.re < 0 ∧ z.im < 0) :=
by
  let z := ((1 - i) ^ 2) / (1 + i)
  sorry

end complex_number_in_third_quadrant_l777_777461


namespace inner_circle_radius_l777_777563

-- Define the properties of the rectangle ABCD and the given circles
structure Rectangle where
  A B C D : ℝ × ℝ
  AB : ℝ
  BC : ℝ
  AB_len : AB = 8
  BC_len : BC = 6

-- Define the properties of the inner circle
structure InnerCircle where
  O : ℝ × ℝ
  r : ℝ

-- Properties of tangency (circle centered at A with radius 4,
-- circle centered at B with radius 4, circle centered at C with radius 3,
-- circle centered at D with radius 3)
axiom Tangency (rect : Rectangle) (inner : InnerCircle) :
  (dist rect.A inner.O) = inner.r + 4 ∧
  (dist rect.B inner.O) = inner.r + 4 ∧
  (dist rect.C inner.O) = inner.r + 3 ∧
  (dist rect.D inner.O) = inner.r + 3

-- Lean statement asserting the radius of the inner circle contained within the rectangle
-- and tangent to the given circles is 11 - sqrt(66)
theorem inner_circle_radius {rect : Rectangle} {inner : InnerCircle}
  (h_tangent : Tangency rect inner) :
  inner.r = 11 - Real.sqrt 66 :=
by
  sorry

end inner_circle_radius_l777_777563


namespace smallest_next_smallest_sum_l777_777201

-- Defining the set of numbers as constants
def nums : Set ℕ := {10, 11, 12, 13}

-- Define the smallest number in the set
def smallest : ℕ := 10

-- Define the next smallest number in the set
def next_smallest : ℕ := 11

-- The main theorem statement
theorem smallest_next_smallest_sum : smallest + next_smallest = 21 :=
by 
  sorry

end smallest_next_smallest_sum_l777_777201


namespace probability_of_same_length_segments_l777_777958

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777958


namespace no_such_convex_hexagon_and_point_exists_l777_777374

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777374


namespace log_eq_3_implies_x_eq_512_l777_777038

theorem log_eq_3_implies_x_eq_512 (x : ℝ) (h : log 8 x = 3) : x = 512 := by
  -- Skipping the proof steps as instructed
  sorry 

end log_eq_3_implies_x_eq_512_l777_777038


namespace no_such_convex_hexagon_and_point_exists_l777_777371

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777371


namespace y_intercept_of_line_l777_777675

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l777_777675


namespace percentage_increase_book_price_l777_777634

theorem percentage_increase_book_price (OldP NewP : ℕ) (hOldP : OldP = 300) (hNewP : NewP = 330) :
  ((NewP - OldP : ℕ) / OldP : ℚ) * 100 = 10 := by
  sorry

end percentage_increase_book_price_l777_777634


namespace no_such_hexagon_exists_l777_777296

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777296


namespace circuit_length_is_365_l777_777092

-- Definitions based on given conditions
def runs_morning := 7
def runs_afternoon := 3
def total_distance_week := 25550
def total_runs_day := runs_morning + runs_afternoon
def total_runs_week := total_runs_day * 7

-- Statement of the problem to be proved
theorem circuit_length_is_365 :
  total_distance_week / total_runs_week = 365 :=
sorry

end circuit_length_is_365_l777_777092


namespace no_such_hexagon_and_point_l777_777394

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777394


namespace no_such_hexagon_exists_l777_777335

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777335


namespace no_such_class_exists_l777_777202

theorem no_such_class_exists : ¬ ∃ (b g : ℕ), (3 * b = 5 * g) ∧ (32 < b + g) ∧ (b + g < 40) :=
by {
  -- Proof goes here
  sorry
}

end no_such_class_exists_l777_777202


namespace prob_Q_closer_to_F_is_half_l777_777545

noncomputable def prob_Q_closer_to_F 
  (D E F Q : Type) 
  [IsTriangle D E F]
  (DE : ℝ) (EF : ℝ) (DF : ℝ) 
  (hDE : DE = 6) (hEF : EF = 8) (hDF : DF = 10) : ℝ :=
  have h_right : is_right_triangle D E F ∧ hypotenuse D E F = DF := sorry
  -- Compute the area of the triangle
  let area_triangle := (1 / 2) * DF * EF
  -- Compute the area of the desired region
  let area_region := (DF / 2) * (EF / 2)
  -- Compute the probability
  area_region / area_triangle

theorem prob_Q_closer_to_F_is_half
  (D E F Q : Type)
  [IsTriangle D E F]
  (DE : ℝ) (EF : ℝ) (DF : ℝ)
  (hDE : DE = 6) (hEF : EF = 8) (hDF : DF = 10) :
  prob_Q_closer_to_F D E F Q DE EF DF hDE hEF hDF = 1 / 2 :=
sorry

end prob_Q_closer_to_F_is_half_l777_777545


namespace greatest_possible_large_chips_l777_777648

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (h1 : s + l = 54)
  (h2 : ∃ p : ℕ, nat.prime p ∧ s = l + p) : 
  l ≤ 26 :=
sorry

end greatest_possible_large_chips_l777_777648


namespace probability_of_same_length_segments_l777_777956

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777956


namespace main_problem_l777_777858

-- Define the ellipse equation with the point that lies on it
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
∀ x y : ℝ, (x, y) = (2 * Real.sqrt 2, Real.sqrt 2) → x^2 / a^2 + y^2 / b^2 = 1

-- Tangent line condition
def tangent_line (x y : ℝ) : Prop :=
x + 2 * y - 4 * Real.sqrt 2 = 0

-- Existence of point P satisfying the given vector condition
def exists_point_P (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
∃ P : ℝ × ℝ, P = (0, 4) ∧ ∀ (M A B : ℝ × ℝ) (k : ℝ), 
  M = (0, 1) → 
  P.1 = 0 → 
  (M.2 = k * M.1 + 1) →
  tangent_line (2 * Real.sqrt 2) (Real.sqrt 2) →
  ∃ λ : ℝ, (P - M) = λ * ((P - A) / |P - A| + (P - B) / |P - B|)

-- Main theorem
theorem main_problem (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos) →
  (tangent_line (2 * Real.sqrt 2) (Real.sqrt 2)) →
  (exists_point_P a b a_pos b_pos) :=
by
  sorry

end main_problem_l777_777858


namespace rectangles_on_grid_l777_777776

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem rectangles_on_grid (cols rows : ℕ) (h₁ : cols = 4) (h₂ : rows = 3) :
  combination 4 2 * combination 3 2 = 18 := by
  rw [h₁, h₂]
  have h_comb4_2 : combination 4 2 = 6 := rfl
  have h_comb3_2 : combination 3 2 = 3 := rfl
  rw [h_comb4_2, h_comb3_2]
  norm_num
  sorry

end rectangles_on_grid_l777_777776


namespace neg_one_power_zero_l777_777266

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l777_777266


namespace initial_numbers_possible_l777_777203

noncomputable def transformation (a b c : ℤ) : Prop :=
  ∃ x y z, (x = a ∨ x = b ∨ x = c) ∧ 
            (y = a ∨ y = b ∨ y = c) ∧ y ≠ x ∧ 
            z = a + b + c - x - y - 1

theorem initial_numbers_possible :
  transformation 17 1967 1983 ∧
  (∃ n1 n2 n3, transformation n1 n2 n3 ∧ 
   transformation ... (repeat application of transformation) ∧ 
   (n1 = 3 ∧ n2 = 3 ∧ n3 = 3)) ∧
  ¬ (∃ m1 m2 m3, transformation m1 m2 m3 ∧ 
   transformation ... (repeat application of transformation) ∧ 
   (m1 = 2 ∧ m2 = 2 ∧ m3 = 2)) :=
sorry

end initial_numbers_possible_l777_777203


namespace rectangle_geometry_square_geometry_l777_777923

section geometric_properties

-- Define a rectangle with its inherent properties.
structure Rectangle (α : Type) [LinearOrder α] :=
  (A B C D : α)
  (right_angles : ∀ {p q r s : α}, right_angle p q r s = true)

-- Define a square as a special type of rectangle.
structure Square (α : Type) [LinearOrder α] extends Rectangle α :=
  (equal_sides : ∀ {p q r s : α}, p = q ∧ q = r ∧ r = s ∧ s = p)
  (perpendicular_diagonals : ∀ {p q r s : α}, perpendicular p q r s = true)

-- Provide the conditions extracted from the initial problem statement.
variable {α : Type} [LinearOrder α]

def Rectangle_properties (R : Rectangle α) : Prop :=
  (∀ {p q r s : α}, p = q ∧ q = r ∧ r = s ∧ s = p) ∧
  (→ symmetries_count R = 2)

def Square_properties (S : Square α) : Prop :=
  (perpendicular_diagonals S) ∧
  (symmetries_count S = 4)

-- Finally, assert the conclusions derived from the solution provided earlier.
theorem rectangle_geometry
  (R : Rectangle α) :
  Rectangle_properties R :=
sorry

theorem square_geometry
  (S : Square α) :
  Square_properties S :=
sorry

end geometric_properties

end rectangle_geometry_square_geometry_l777_777923


namespace consistent_values_l777_777807

theorem consistent_values (a x: ℝ) :
    (12 * x^2 + 48 * x - a + 36 = 0) ∧ ((a + 60) * x - 3 * (a - 20) = 0) ↔
    ((a = -12 ∧ x = -2) ∨ (a = 0 ∧ x = -1) ∨ (a = 180 ∧ x = 2)) := 
by
  -- proof steps should be filled here
  sorry

end consistent_values_l777_777807


namespace eq_solution_l777_777787

-- Define the recursive function f_n
def f : ℕ → ℝ → ℝ
| 0, x => sqrt (x^2 + 48)
| (n+1), x => sqrt (x^2 + 6 * f n x)

-- Theorem stating that the only solution to f_n(x) = 2x for all n is x = 4
theorem eq_solution : ∀ n : ℕ, ∀ x : ℝ, (f n x = 2 * x) → (x = 4) :=
by
  intros n x h
  sorry

end eq_solution_l777_777787


namespace no_such_hexagon_exists_l777_777347

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777347


namespace nico_borrowed_at_least_three_books_on_monday_l777_777593

-- Define the conditions
def reads_first_book : ℕ := 20
def reads_second_book : ℕ := 12
def total_read_until_wednesday : ℕ := 51

-- The proof problem statement
theorem nico_borrowed_at_least_three_books_on_monday (reads_first_book reads_second_book total_read_until_wednesday : ℕ) :
  reads_first_book = 20 →
  reads_second_book = 12 →
  total_read_until_wednesday = 51 →
  ∃ n : ℕ, n ≥ 3 :=
by
  assume h1 : reads_first_book = 20
  assume h2 : reads_second_book = 12
  assume h3 : total_read_until_wednesday = 51
  -- Proof
  sorry

end nico_borrowed_at_least_three_books_on_monday_l777_777593


namespace bees_on_second_day_l777_777590

theorem bees_on_second_day (bees_first_day : ℕ) (tripling_factor : ℕ) (h1 : bees_first_day = 144) (h2 : tripling_factor = 3) :
  let bees_second_day := bees_first_day * tripling_factor
  in bees_second_day = 432 := 
by
  intros
  have h3 : bees_second_day = 144 * 3 := by rw [h1, h2]
  rw h3
  norm_num
  exact rfl

end bees_on_second_day_l777_777590


namespace roots_sum_arithmetic_sequence_l777_777110

noncomputable def sum_of_roots (p : Polynomial ℝ) : ℝ :=
let coeffs := p.coeffs in
let n := p.natDegree in
- (coeffs.get (n - 1) / coeffs.get (n))

theorem roots_sum_arithmetic_sequence (P : Polynomial ℝ) (h : P ≠ 0) :
  ∃ d : ℝ, 
    let S := λ k, sum_of_roots (P.derivative^[k]) in
    ∀ m n, 
      m < P.natDegree ∧ n < P.natDegree → 
      S (n + 1) - S n = d ∧
      S (m + 1) - S m = d ∧
      S 1 - S 0 = - (P.coeff (P.natDegree - 1) / P.coeff P.natDegree / P.natDegree) := 
sorry

end roots_sum_arithmetic_sequence_l777_777110


namespace problem_equivalence_l777_777625

-- Definitions of the statements
def statement1 (p q : Prop) : Prop := (¬ (p ∧ q)) → (¬ p ∧ ¬ q)
def statement2 : Prop := (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧ (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1)
def statement3 : Prop := (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))
def statement4 : Prop := (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- Stating the proof problem.
theorem problem_equivalence : ¬ statement1 ∧ statement2 ∧ statement3 ∧ statement4 :=
by
  -- Only proving the equivalence, no actual proof steps required.
  sorry

end problem_equivalence_l777_777625


namespace perfect_square_divisors_count_l777_777489

noncomputable def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map Nat.factorial |>.prod

theorem perfect_square_divisors_count : 
  (∀ n, perfect_square_divisors (product_factorials 10) = 1920) := sorry

end perfect_square_divisors_count_l777_777489


namespace train_length_l777_777657

noncomputable def length_of_first_train (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  let v1_m_per_s := v1 * 1000 / 3600
  let v2_m_per_s := v2 * 1000 / 3600
  let relative_speed := v1_m_per_s + v2_m_per_s
  let combined_length := relative_speed * t
  combined_length - l2

theorem train_length (l2 : ℝ) (v1 : ℝ) (v2 : ℝ) (t : ℝ) (h_l2 : l2 = 200) 
  (h_v1 : v1 = 100) (h_v2 : v2 = 200) (h_t : t = 3.6) : length_of_first_train l2 v1 v2 t = 100 := by
  sorry

end train_length_l777_777657


namespace no_convex_hexagon_with_point_M_l777_777309

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777309


namespace no_convex_hexagon_with_point_M_l777_777308

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777308


namespace prove_x_minus_y_squared_l777_777493

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l777_777493


namespace plane_coloring_impossible_l777_777404

-- Define points and their colors
inductive Color
| red
| blue

def Point : Type := ℝ × ℝ

-- Define that every point on a plane is colored either red or blue
def color (p : Point) : Color := sorry

-- Given conditions
def not_all_same_color : Prop := ∃ p q : Point, p ≠ q ∧ color p ≠ color q
def exactly_one_blue_on_circumference (p : Point) : Prop := 
    ∃! q : Point, dist p q = 1 ∧ color q = Color.blue 

-- The main theorem
theorem plane_coloring_impossible :
  (∃ p, ∀ q, dist p q = 1 → color q = Color.blue) → false :=
begin
  assume h,
  sorry -- Proof follows here
end

end plane_coloring_impossible_l777_777404


namespace smallest_value_of_3a_plus_2_l777_777036

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  3 * a + 2 = 1 / 2 :=
sorry

end smallest_value_of_3a_plus_2_l777_777036


namespace Monge_point_foot_of_altitude_on_circumcircle_l777_777149

-- Define the types for points and the plane.
variable {Point : Type}
variable (A B C D T O G H : Point) (plane_ABC : Set Point)

-- Define the functions for distance, centroid, Monge point, etc.
variable (distance : Point → Point → Real)
variable (circumcenter_of_face : Point → Point → Point → Point)
variable (centroid_of_tetrahedron : Point → Point → Point → Point → Point)
variable (Monge_point_of_tetrahedron : Point → Point → Point → Point → Point)
variable (foot_of_perpendicular : Point → Set Point → Point)

-- Define the given conditions as hypotheses
theorem Monge_point_foot_of_altitude_on_circumcircle :
  (Monge_point_of_tetrahedron A B C D = T) →
  (circumcenter_of_face A B C = O) →
  (centroid_of_tetrahedron A B C D = G) →
  (distance G T = distance G O) →
  (T ∈ plane_ABC) →
  (foot_of_perpendicular D plane_ABC = H) →
  (H ∈ circumcenter_of_face A B C)
:= by 
  sorry

end Monge_point_foot_of_altitude_on_circumcircle_l777_777149


namespace sum_binom_tan_sec_identity_l777_777606

theorem sum_binom_tan_sec_identity
  (n : ℕ)
  (x : ℝ) :
  (∑ k in Finset.range (n + 1), Nat.choose n k * (Real.tan (x / 2))^(2 * k) * (1 + 2^k / (1 - (Real.tan (x / 2))^2)^k)) = (Real.sec (x / 2))^(2 * n) + (Real.sec x)^n :=
by
  sorry

end sum_binom_tan_sec_identity_l777_777606


namespace area_inequality_l777_777085

theorem area_inequality {ABC A1 B1 C1 A0 B0 C0 : Type} 
  (hA1 : ∀ (ABC : Type), ∃ (P : Type), P = intersection_angle_bisector_circumcircle ABC A) 
  (hB1 : ∀ (ABC : Type), ∃ (P : Type), P = intersection_angle_bisector_circumcircle ABC B) 
  (hC1 : ∀ (ABC : Type), ∃ (P : Type), P = intersection_angle_bisector_circumcircle ABC C) 
  (hA0 : ∀ (ABC A1 : Type), ∃ (P : Type), P = intersection_external_angle_bisector ABC A1) 
  (hB0 : ∀ (ABC B1 : Type), ∃ (P : Type), P = intersection_external_angle_bisector ABC B1) 
  (hC0 : ∀ (ABC C1 : Type), ∃ (P : Type), P = intersection_external_angle_bisector ABC C1) :
  area_triangle A0 B0 C0 ≥ 4 * area_triangle ABC := sorry

end area_inequality_l777_777085


namespace cannot_determine_right_triangle_by_angle_ratio_l777_777544

theorem cannot_determine_right_triangle_by_angle_ratio 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13)
  (h3 : a / b = 1 / √2) (h4 : b / c = √2 / √3)
  (h5 : A / B = 3 / 4) (h6 : B / C = 4 / 5) :
  ¬ (C = 90) :=
by
  sorry

end cannot_determine_right_triangle_by_angle_ratio_l777_777544


namespace max_residents_in_block_l777_777083

theorem max_residents_in_block 
  (total_floors : ℕ) (half_floors: ℕ)
  (apartments_per_half : ℕ → ℕ)
  (max_residents_per_apartment : ℕ)
  (condition1 : total_floors = 12)
  (condition2 : half_floors = total_floors / 2)
  (condition3a : apartments_per_half 0 = 6)
  (condition3b : apartments_per_half 1 = 5)
  (condition4 : max_residents_per_apartment = 4) :
  let total_apartments := half_floors * apartments_per_half 0 + half_floors * apartments_per_half 1 in
  let max_residents := total_apartments * max_residents_per_apartment in
  max_residents = 264 := 
  by
  sorry

end max_residents_in_block_l777_777083


namespace equation_vs_equality_l777_777767

theorem equation_vs_equality : 
  (∀ (e : Type), (∃ (x : e), true) → e = true) →
  false :=
by
  intros
  sorry

end equation_vs_equality_l777_777767


namespace john_finishes_fourth_task_at_l777_777930

theorem john_finishes_fourth_task_at (h1 : John_begins_first_task_at = 9 * 60)
                                     (h2 : John_finishes_third_task_at = 11 * 60 + 30)
                                     (h3 : ∀ t, is_task_duration_equal t) :
  fourth_task_finishes_at = 12 * 60 + 20 :=

sorry

end john_finishes_fourth_task_at_l777_777930


namespace find_angle_C_max_area_condition_l777_777897

-- Definitions from the conditions
variables {A B C : ℝ} {a b c : ℝ}
variable h1 : sqrt 3 * (a - c * real.cos B) = b * real.sin C

-- Proof of part 1: Finding angle C
theorem find_angle_C (h : sqrt 3 * (a - c * real.cos B) = b * real.sin C) : C = π / 3 :=
sorry

-- Definition of area of triangle
noncomputable def triangle_area (a b C : ℝ) : ℝ :=
  (1 / 2) * a * b * real.sin C

-- Proof of part 2: Maximizing the area with given c = 2
theorem max_area_condition (h : c = 2) (h2 : C = π / 3) : 
  ∃ a b : ℝ, a = 2 ∧ b = 2 ∧ triangle_area a b C = sqrt 3 :=
sorry

end find_angle_C_max_area_condition_l777_777897


namespace count_four_digit_even_nonzero_digits_l777_777881

/-- The number of distinct four-digit positive integers with only even non-zero digits is 256. -/
theorem count_four_digit_even_nonzero_digits : 
  let digits := {2, 4, 6, 8}
  let four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d ∈ Int.toString n, d ∈ digits}
  4^4 = 256 := by
  sorry

end count_four_digit_even_nonzero_digits_l777_777881


namespace positive_difference_eq_505_l777_777210

theorem positive_difference_eq_505 :
  let a := (8^2 - 8) / 8,
      b := (8^2 * 8^2) / 8 in
  abs (b - a) = 505 := by
  sorry

end positive_difference_eq_505_l777_777210


namespace sum_fiftieth_powers_100_gon_l777_777640

noncomputable def sum_fiftieth_powers_all_sides_and_diagonals (n : ℕ) (R : ℝ) : ℝ := sorry
-- Define the sum of 50-th powers of all the sides and diagonals for a general n-gon inscribed in a circle of radius R

theorem sum_fiftieth_powers_100_gon (R : ℝ) : 
  sum_fiftieth_powers_all_sides_and_diagonals 100 R = sorry := sorry

end sum_fiftieth_powers_100_gon_l777_777640


namespace no_such_convex_hexagon_and_point_l777_777338

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777338


namespace distinct_four_digit_even_integers_count_l777_777880

theorem distinct_four_digit_even_integers_count : 
  ∃ n : ℕ, n = 256 ∧ 
  (∀ thousands hundreds tens units : ℕ, 
    thousands ∈ {2, 4, 6, 8} ∧ 
    hundreds ∈ {2, 4, 6, 8} ∧ 
    tens ∈ {2, 4, 6, 8} ∧ 
    units ∈ {2, 4, 6, 8} → 
    n = 4^4) := 
by
  sorry

end distinct_four_digit_even_integers_count_l777_777880


namespace kevin_found_cards_l777_777098

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end kevin_found_cards_l777_777098


namespace trader_profit_l777_777758

noncomputable def profit_percentage (P : ℝ) : ℝ :=
  let purchased_price := 0.72 * P
  let market_increase := 1.05 * purchased_price
  let expenses := 0.08 * market_increase
  let net_price := market_increase - expenses
  let first_sale_price := 1.50 * net_price
  let final_sale_price := 1.25 * first_sale_price
  let profit := final_sale_price - P
  (profit / P) * 100

theorem trader_profit
  (P : ℝ) 
  (hP : 0 < P) :
  profit_percentage P = 30.41 :=
by
  sorry

end trader_profit_l777_777758


namespace sum_roots_eq_six_l777_777689

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l777_777689


namespace part_a_part_b_l777_777106

-- Given definitions and conditions
variables (A B C P Q T_A T_B T_C : Point)
variables (circumcircle : Circle)
variables (isogonal_conjugate : Point → Point → Triangle → Prop)
variables (similar_orientation : Triangle → Triangle → Prop)

def tasty_pair (P Q : Point) (ABC : Triangle) : Prop :=
  isogonal_conjugate P Q ABC

-- Part (a) statement:
theorem part_a (P Q : Point) (ABC : Triangle)
  [hq1 : Q ≠ A]
  [hq2 : Q ≠ B]
  [hq3 : Q ≠ C]
  (T_A_property : similar_orientation (Triangle.mk T_A B P) (Triangle.mk T_A Q C))
  (T_B_property : similar_orientation (Triangle.mk T_B C P) (Triangle.mk T_B Q A))
  (T_C_property : similar_orientation (Triangle.mk T_C A P) (Triangle.mk T_C Q B))
  (is_tasty_pair : tasty_pair P Q ABC) :
  (T_A, T_B, T_C) ⊆ circumcircle :=
sorry

-- Part (b) statement:
theorem part_b (P Q : Point) (ABC Tabc : Triangle)
  [is_congruent : congruent (Triangle.mk A B C) (Triangle.mk T_A T_B T_C)]
  (T_A_property : similar_orientation (Triangle.mk T_A B P) (Triangle.mk T_A Q C))
  (T_B_property : similar_orientation (Triangle.mk T_B C P) (Triangle.mk T_B Q A))
  (T_C_property : similar_orientation (Triangle.mk T_C A P) (Triangle.mk T_C Q B))
  (is_tasty_pair_abc : tasty_pair P Q ABC) :
  tasty_pair P Q (Triangle.mk T_A T_B T_C) :=
sorry

end part_a_part_b_l777_777106


namespace production_company_keeps_60_percent_l777_777246

noncomputable def openingWeekendRevenue : ℝ := 120
noncomputable def productionCost : ℝ := 60
noncomputable def profit : ℝ := 192
noncomputable def totalRevenue : ℝ := 3.5 * openingWeekendRevenue
noncomputable def amountKept : ℝ := profit + productionCost
noncomputable def percentageKept : ℝ := (amountKept / totalRevenue) * 100

theorem production_company_keeps_60_percent :
  percentageKept = 60 :=
by
  sorry

end production_company_keeps_60_percent_l777_777246


namespace tenth_triangular_number_l777_777174

theorem tenth_triangular_number : 
  let T : ℕ → ℕ := λ n : ℕ, n * (n + 1) / 2 in
  T 10 = 55 :=
by
  sorry

end tenth_triangular_number_l777_777174


namespace f_ln_half_eq_neg_3_l777_777474

namespace ProofProblem

-- Define the function
def f (a x : ℝ) : ℝ := a * x + 5^x / (5^x + 1)

-- Given conditions
variable (a : ℝ)

-- The specific given value
axiom f_ln2_eq_4 : f a (Real.log 2) = 4

-- The statement of the problem to be proved
theorem f_ln_half_eq_neg_3 : f a (Real.log (1 / 2)) = -3 :=
sorry

end ProofProblem

end f_ln_half_eq_neg_3_l777_777474


namespace find_principal_l777_777257

theorem find_principal (R : ℝ) : ∃ P : ℝ, (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 100 :=
by {
  use 200,
  sorry
}

end find_principal_l777_777257


namespace part1_part2_l777_777586

-- Part (1)
theorem part1 (a b : ℕ) (h₁ : a = 12) (h₂ : b = 5) : (c : ℕ) = 13 :=
by
  have hc : c = sqrt (a^2 + b^2) := sorry

-- Part (2)
theorem part2 (c b : ℕ) (h₁ : c = 10) (h₂ : b = 9) : (a : ℝ) = real.sqrt 19 :=
by
  have ha : a = real.sqrt (c^2 - b^2) := sorry

end part1_part2_l777_777586


namespace baseball_team_opponents_total_score_l777_777725

theorem baseball_team_opponents_total_score : 
  ∀ (scores : List ℕ) (loss_game_scores : List ℕ) (triple_score_wins : List ℕ) (double_score_wins : List ℕ),
  scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  loss_game_scores = [1, 2, 3, 4, 5, 6] →
  triple_score_wins = [10, 11, 12] →
  double_score_wins = [7, 8, 9] →
  (∀ (x : ℕ), x ∈ loss_game_scores → ∃ (opponent_x : ℕ), opponent_x = x + 2) →
  (∀ (y : ℕ), y ∈ triple_score_wins → ∃ (opponent_y : ℚ), opponent_y = y / 3) →
  (∀ (z : ℕ), z ∈ double_score_wins → ∃ (opponent_z : ℚ), opponent_z = z / 2) →
  (List.sum (List.map (λ x, x + 2) loss_game_scores) + 
   List.sum (List.map (λ y, (y : ℚ) / 3) triple_score_wins).toNat + 
   List.sum (List.map (λ z, (z : ℚ) / 2) double_score_wins).toNat) = 56 :=
by
  intros scores loss_game_scores triple_score_wins double_score_wins
  intros hscores hloss htriple hdouble h_loss_game_scores h_triple_score_wins h_double_score_wins
  rw [hscores, hloss, htriple, hdouble]
  simp at *
  sorry

end baseball_team_opponents_total_score_l777_777725


namespace work_done_by_external_force_l777_777216

-- Definitions from the conditions
def hookes_law (F k l : ℝ) : Prop := F = k * l

def original_length := 0.9
def compression_force_per_cm := 0.049
def spring_constant := compression_force_per_cm * 100 -- Converting to per meter
def compressed_from := 0.7
def compressed_to := 0.5

-- Main theorem statement
theorem work_done_by_external_force : 
  ∫ x in (compressed_to..compressed_from), spring_constant * x = 0.294 := by
  sorry

end work_done_by_external_force_l777_777216


namespace jill_lavender_candles_l777_777928

variable (L C A : ℕ)

def scented_candles (L C A : ℕ) : Prop :=
  (L = 2 * C) ∧ (A = 10) ∧ (C = 1.5 * A)

theorem jill_lavender_candles (h : scented_candles L C A) : L = 30 :=
by
  have h1 : A = 10 := h.2.1
  have h2 : C = 1.5 * A := h.2.2
  have h3 : C = 15 := by rw [h1, mul_comm, Nat.floor_div_eq_div (by norm_num : 1 < (15 / (1:ℕ) : ℚ)) (by norm_num)]
  have h4 : L = 2 * C := h.1
  rw [h3] at h4
  exact h4
  sorry -- Here we skip the detailed final proof step considering we will provide a complete proof in a more context-rich environment.


end jill_lavender_candles_l777_777928


namespace no_such_hexagon_exists_l777_777325

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777325


namespace cannot_be_external_diagonals_l777_777705

theorem cannot_be_external_diagonals (a b c : ℕ) : 
  ¬(3^2 + 4^2 = 6^2) :=
by
  sorry

end cannot_be_external_diagonals_l777_777705


namespace AK_eq_BL_l777_777873

open EuclideanGeometry

-- Define the setting
variables {k l : Circle} {K L A B : Point}

-- Assume given conditions
axiom tangents_touch : IsTangent K k ∧ IsTangent L l
axiom circles_intersect : Intersect k l
axiom K_on_k : OnCircle K k
axiom L_on_l : OnCircle L l
axiom A_on_k : OnCircle A k
axiom B_on_l : OnCircle B l
axiom line_KL : IsLine K L

-- Prove the theorem
theorem AK_eq_BL (h : tangents_touch ∧ circles_intersect ∧ K_on_k ∧ L_on_l ∧ A_on_k ∧ B_on_l ∧ line_KL) :
  distance A K = distance B L :=
by
  sorry

end AK_eq_BL_l777_777873


namespace sweets_neither_red_nor_green_l777_777196

theorem sweets_neither_red_nor_green (total_sweets : ℕ) (red_sweets : ℕ) (green_sweets : ℕ) 
  (h_total : total_sweets = 285) (h_red : red_sweets = 49) (h_green : green_sweets = 59) :
  total_sweets - (red_sweets + green_sweets) = 177 :=
by 
  rw [h_total, h_red, h_green]
  sorry

end sweets_neither_red_nor_green_l777_777196


namespace correctness_of_conclusions_l777_777078

-- Define our setting
def sicos (θ : ℝ) (x0 y0 r : ℝ) : ℝ := (x0 + y0) / r

def positive_cosine (x : ℝ) : ℝ := (2:ℝ)^(1/2) * sin (x + (Real.pi / 4))

theorem correctness_of_conclusions (θ : ℝ) (x0 y0 r : ℝ)
  (h1 : r > 0)
  (h2 : positive_cosine (3 * Real.pi / 4) = 0)
  (h3 : ∀ k : ℤ, ∀ x : ℝ,
    2 * k * Real.pi + Real.pi / 4 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 4 → deriv sin (x + Real.pi / 4) < 0)
  (h4 : ∀ x : ℝ, positive_cosine x ≠ 3 / 2) :
-- Conclusion 2 corresponds to symmetry at (3π/4, 0)
  (positive_cosine (3 * Real.pi / 4) = 0) ∧
-- Conclusion 4 corresponds to no intersection point with y = 3/2
  (∀ x : ℝ, positive_cosine x ≠ 3 / 2) :=
by
  sorry

end correctness_of_conclusions_l777_777078


namespace alan_total_payment_l777_777763

theorem alan_total_payment :
  let D := 24 -- price of one CD by "The Dark"
  let avn_price := 12 -- price of one CD by "AVN"
  let total_other_cds_cost := 2 * D + avn_price -- total cost of "The Dark" and "AVN" CDs
  let nineties_cd_cost := 0.40 * total_other_cds_cost -- cost of 5 CDs of 90s music
  let total_cost_before_discounts := total_other_cds_cost + nineties_cd_cost
  let discount := 0.10 * total_cost_before_discounts -- 10% discount
  let discounted_cost := total_cost_before_discounts - discount
  let sales_tax := 0.08 * discounted_cost -- 8% sales tax
  let final_amount := discounted_cost + sales_tax
  round(final_amount * 100) / 100 = 81.65 := sorry

end alan_total_payment_l777_777763


namespace combined_market_value_two_years_later_l777_777286

theorem combined_market_value_two_years_later:
  let P_A := 8000
  let P_B := 10000
  let P_C := 12000
  let r_A := 0.20
  let r_B := 0.15
  let r_C := 0.10

  let V_A_year_1 := P_A - r_A * P_A
  let V_A_year_2 := V_A_year_1 - r_A * P_A
  let V_B_year_1 := P_B - r_B * P_B
  let V_B_year_2 := V_B_year_1 - r_B * P_B
  let V_C_year_1 := P_C - r_C * P_C
  let V_C_year_2 := V_C_year_1 - r_C * P_C

  V_A_year_2 + V_B_year_2 + V_C_year_2 = 21400 :=
by
  sorry

end combined_market_value_two_years_later_l777_777286


namespace probability_same_length_segments_of_regular_hexagon_l777_777950

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777950


namespace problem_l777_777851

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable (d r : ℝ)
variable (n : ℕ)

-- Conditions for arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Conditions for geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, b (n + 1) = b n * r

-- Specific conditions given for sequences
def conditions_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
a 1 + (a 1 + 4 * d) + (a 1 + 8 * d) = 9

def conditions_b (b : ℕ → ℝ) (r : ℝ) : Prop :=
b 2 * (b 2 * r ^ 3) * (b 2 * r ^ 6) = 3 * real.sqrt 3

-- Main statement to be proved
theorem problem (a : ℕ → ℝ) (b : ℕ → ℝ) (d r : ℝ) : 
  is_arithmetic_sequence a d → 
  is_geometric_sequence b r → 
  conditions_a a d → 
  conditions_b b r → 
  (a 2 + a 8) / (1 + b 2 * b 8) = 3 / 2 :=
by 
  sorry

end problem_l777_777851


namespace central_projection_preserves_lines_l777_777605

-- Define the conditions as parameters

variables (α₁ α₂ : Type) [ProjectivePlane α₁] [ProjectivePlane α₂]
variable (O : Point)

-- Define what it means for a line to be exceptional
def exceptional (l : Line α₁) : Prop :=
  ∃ P : Point, P ∈ l ∧ P = O

-- Define the central projection from α₁ to α₂.
def central_projection (P : Point α₁) : Point α₂ :=
  if P ≠ O then some_projection_function P O else some_infinity_point_on_α₂

-- Define the theorem statement
theorem central_projection_preserves_lines (l : Line α₁) (h : ¬ exceptional l) :
  ∃ m : Line α₂, ∀ P : Point, P ∈ l → (central_projection O P) ∈ m :=
begin
  sorry
end

end central_projection_preserves_lines_l777_777605


namespace no_such_hexagon_exists_l777_777318

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777318


namespace fraction_sum_l777_777053

variable {a b : ℝ}

theorem fraction_sum (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) (h1 : a^2 + a - 2007 = 0) (h2 : b^2 + b - 2007 = 0) :
  (1/a + 1/b) = 1/2007 :=
by
  sorry

end fraction_sum_l777_777053


namespace calculate_expression_l777_777778

theorem calculate_expression :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 :=
by
  sorry

end calculate_expression_l777_777778


namespace circles_externally_tangent_l777_777633

noncomputable def M := (3 : ℝ, 0 : ℝ)
noncomputable def N := (0 : ℝ, -4 : ℝ)

noncomputable def radius1 : ℝ := 3
noncomputable def radius2 : ℝ := 2

theorem circles_externally_tangent :
  dist M N = radius1 + radius2 := by
  sorry

end circles_externally_tangent_l777_777633


namespace must_occur_five_times_in_august_l777_777615

variable (N : ℕ) -- Year N

-- The following definitions are implied from the problem.

def has_five_fridays_in_july (year : ℕ) : Prop :=
  ∃ (days : list ℕ), (∀ day ∈ days, 1 ≤ day ∧ day ≤ 31) ∧ 
  (∀ i ∈ {0, 1, 2, 3, 4}, ((days.nth i).get_or_else 0) % 7 = 5) ∧
  (days.length = 5)

def has_31_days (month : ℕ) : Prop :=
  month = 31

-- The final proof statement
theorem must_occur_five_times_in_august (h1 : has_five_fridays_in_july N)
    (h2 : has_31_days 31) : 
    ∃ day : string, (day = "Tuesday") ∧ occurs_five_times_in_august day :=
sorry

end must_occur_five_times_in_august_l777_777615


namespace probability_same_length_segments_of_regular_hexagon_l777_777951

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777951


namespace mandy_reads_books_of_480_pages_l777_777131

def pages_at_age6 : ℕ := 8

def pages_at_age12 (p6 : ℕ) : ℕ := 5 * p6

def pages_at_age20 (p12 : ℕ) : ℕ := 3 * p12

def pages_presently (p20 : ℕ) : ℕ := 4 * p20

theorem mandy_reads_books_of_480_pages :
  let p6 := pages_at_age6,
  let p12 := pages_at_age12 p6,
  let p20 := pages_at_age20 p12,
  let ppresent := pages_presently p20
  in ppresent = 480 :=
by
  sorry

end mandy_reads_books_of_480_pages_l777_777131


namespace calculate_total_weight_AlBr3_l777_777775

-- Definitions for the atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90

-- Definition for the molecular weight of AlBr3
def molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br

-- Number of moles
def number_of_moles : ℝ := 5

-- Total weight of 5 moles of AlBr3
def total_weight_5_moles_AlBr3 : ℝ := molecular_weight_AlBr3 * number_of_moles

-- Desired result
def expected_total_weight : ℝ := 1333.40

-- Statement to prove that total_weight_5_moles_AlBr3 equals the expected total weight
theorem calculate_total_weight_AlBr3 :
  total_weight_5_moles_AlBr3 = expected_total_weight :=
sorry

end calculate_total_weight_AlBr3_l777_777775


namespace work_completion_time_l777_777709

theorem work_completion_time (W : ℝ) : 
  let A_effort := 1 / 11
  let B_effort := 1 / 20
  let C_effort := 1 / 55
  (2 * A_effort + B_effort + C_effort) = 1 / 4 → 
  8 * (2 * A_effort + B_effort + C_effort) = 1 :=
by { sorry }

end work_completion_time_l777_777709


namespace max_f_in_interval_l777_777917

-- Define the new operation
def op (a b : ℝ) : ℝ :=
  if a >= b then a else b^2

-- Define the function f
def f (x : ℝ) : ℝ :=
  (op 1 x) * x + (op 2 x)

-- The theorem statement
theorem max_f_in_interval : ∃ x ∈ set.Icc (-2 : ℝ) 2, ∀ y ∈ set.Icc (-2 : ℝ) 2, f y ≤ f x ∧ f x = 10 :=
by
  sorry

end max_f_in_interval_l777_777917


namespace expression_may_not_hold_true_l777_777811

theorem expression_may_not_hold_true (a b x y : ℝ) (h : a = b) : a * x = b * y → x = y :=
begin
  intro h1,
  sorry,
end

end expression_may_not_hold_true_l777_777811


namespace hexagon_probability_same_length_l777_777982

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777982


namespace probability_two_consecutive_heads_four_tosses_is_one_fourth_l777_777241
-- Import the full Mathlib library

-- Define the problem statement in Lean
theorem probability_two_consecutive_heads_four_tosses_is_one_fourth :
  let outcomes := 2^4 in
  let favorable_outcomes := 4 in
  favorable_outcomes / outcomes = (1 : ℚ)/4 := by
  sorry

end probability_two_consecutive_heads_four_tosses_is_one_fourth_l777_777241


namespace y_intercept_of_line_l777_777666

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l777_777666


namespace joanne_earnings_l777_777552

theorem joanne_earnings :
  let main_job_hourly_wage := 16.00
  let part_time_job_hourly_wage := 13.50
  let main_job_hours_per_day := 8
  let part_time_job_hours_per_day := 2
  let number_of_days := 5

  let main_job_daily_earnings := main_job_hours_per_day * main_job_hourly_wage
  let main_job_weekly_earnings := main_job_daily_earnings * number_of_days
  let part_time_job_daily_earnings := part_time_job_hours_per_day * part_time_job_hourly_wage
  let part_time_job_weekly_earnings := part_time_job_daily_earnings * number_of_days

  (main_job_weekly_earnings + part_time_job_weekly_earnings = 775)
:= by
  simp only [
    main_job_hourly_wage, part_time_job_hourly_wage,
    main_job_hours_per_day, part_time_job_hours_per_day,
    number_of_days,
    main_job_daily_earnings, main_job_weekly_earnings,
    part_time_job_daily_earnings, part_time_job_weekly_earnings
  ]
  sorry

end joanne_earnings_l777_777552


namespace question1_question2_l777_777090

-- Definitions and Conditions
variables {A B C : ℝ} {a b c : ℝ}
def sides (s : ℝ) : Prop := s > 0
def angles (α : ℝ) : Prop := 0 < α ∧ α < π

-- Given conditions
axiom triangle (a b c : ℝ) : sides a ∧ sides b ∧ sides c
axiom angleC (A B C : ℝ) : angles A ∧ angles B ∧ angles C
axiom cos_condition (C A B : ℝ) : 3 * Real.cos C = 2 * Real.sin A * Real.sin B

-- Lean statements
theorem question1 (h : 3 * Real.cos C = 2 * Real.sin A * Real.sin B) : 
  ∃ x, x = (2 * Real.sqrt 3) / 3 ∧ x = Real.sin C / (Real.sin A * Real.sin B) := 
sorry

theorem question2 (h1 : A = π / 6) (h2 : a = Real.sqrt 7)
  (h3 : sides a) (h4 : angles A) : 
  ∃ c area, c = 5 / 3 ∧ area = (5 * Real.sqrt 3) / 4 :=
sorry

end question1_question2_l777_777090


namespace angle_of_inclination_tangent_line_at_point_l777_777638

theorem angle_of_inclination_tangent_line_at_point :
  let y := fun x => x^3 - 2 * x + 4
  let y' := fun x => deriv y x
  let point := (1 : ℝ, y 1)
  let slope := y' 1
  let α := Real.arctan slope
  0 ≤ α ∧ α ≤ Real.pi →
  α = Real.pi / 4 :=
by
  sorry

end angle_of_inclination_tangent_line_at_point_l777_777638


namespace probability_one_white_ball_conditional_probability_P_B_given_A_l777_777901

-- Definitions for Problem 1
def red_balls : Nat := 4
def white_balls : Nat := 2
def total_balls : Nat := red_balls + white_balls

def C (n k : ℕ) : ℕ := n.choose k

theorem probability_one_white_ball :
  (C 2 1 * C 4 2 : ℚ) / C 6 3 = 3 / 5 :=
by sorry

-- Definitions for Problem 2
def total_after_first_draw : Nat := total_balls - 1
def remaining_red_balls : Nat := red_balls - 1

theorem conditional_probability_P_B_given_A :
  (remaining_red_balls : ℚ) / total_after_first_draw = 3 / 5 :=
by sorry

end probability_one_white_ball_conditional_probability_P_B_given_A_l777_777901


namespace no_convex_hexagon_with_point_M_l777_777307

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777307


namespace maximum_area_of_region_S_l777_777527

-- Define the four circles with their respective radii
structure Circle (r : ℝ) := 
(radius : ℝ := r)

def circle1 : Circle := { radius := 1 }
def circle3 : Circle := { radius := 3 }
def circle5 : Circle := { radius := 5 }
def circle7 : Circle := { radius := 7 }

-- Define the line l, which the circles are tangent to at point A
-- For simplicity, we can abstract the line l and point A without defining them explicitly here, since we are only concerned with the areas
axiom line_l_and_point_A : Prop

-- Define the function that calculates the area of a circle
def area (c: Circle) : ℝ := π * c.radius^2

-- Define the region S as the union of the four circles
def region_S (c1 c3 c5 c7 : Circle) : ℝ := area c7 + area c5 - area c3 - area c1

-- State the maximum area theorem to be proven
theorem maximum_area_of_region_S (c1 c3 c5 c7 : Circle) (h1 : c1.radius = 1) (h3 : c3.radius = 3) (h5 : c5.radius = 5) (h7 : c7.radius = 7) : 
  region_S c1 c3 c5 c7 = 65 * π :=
by
  sorry

end maximum_area_of_region_S_l777_777527


namespace max_f_value_find_A_and_side_a_l777_777284

noncomputable def f (x : ℝ) : ℝ := sin x * sin (π / 2 + x) + cos x ^ 2

theorem max_f_value : Sup (set.range f) = (sqrt 2 + 1) / 2 := sorry

variables {A B : ℝ} {a b : ℝ}
axiom f_A : f A = 1
axiom A_B_sum : A + B = 7 * π / 12
axiom side_b : b = sqrt 6

theorem find_A_and_side_a :
  A = π / 4 ∧ a = 2 :=
begin
  sorry
end

end max_f_value_find_A_and_side_a_l777_777284


namespace sinx_tanx_condition_l777_777046

theorem sinx_tanx_condition (x : ℝ) : (sin x = (Real.sqrt 2) / 2) → (¬ (tan x = 1)) ∧ (tan x = 1 → sin x = (Real.sqrt 2) / 2) :=
by
  sorry

end sinx_tanx_condition_l777_777046


namespace mural_costs_l777_777556

def trapezoid_area (a b h : ℝ) : ℝ :=
  0.5 * (a + b) * h

def paint_cost (area cost_per_sqm tax_rate : ℝ) : ℝ :=
  let base_cost := area * cost_per_sqm
  base_cost + base_cost * tax_rate

def labor_cost (area rate_per_hour area_per_hour : ℝ) : ℝ :=
  let hours_needed := (area / area_per_hour).ceil
  hours_needed * rate_per_hour

def total_cost (paint_cost labor_cost additional_supplies : ℝ) : ℝ :=
  paint_cost + labor_cost + additional_supplies

theorem mural_costs
  (a b h : ℝ) (paint_cost_per_sqm tax_rate extra_cost : ℝ)
  (area_per_hour rate_per_hour : ℝ)
  (H₁ : a = 6) (H₂ : b = 4) (H₃ : h = 3)
  (H₄ : paint_cost_per_sqm = 4) (H₅ : tax_rate = 0.05)
  (H₆ : extra_cost = 30) (H₇ : area_per_hour = 2) (H₈ : rate_per_hour = 15) :
  trapezoid_area a b h = 15 ∧ total_cost (paint_cost (trapezoid_area a b h) paint_cost_per_sqm tax_rate) (labor_cost (trapezoid_area a b h) rate_per_hour area_per_hour) extra_cost = 213 := by
  sorry

end mural_costs_l777_777556


namespace probability_of_same_length_segments_l777_777957

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777957


namespace parabola_line_equations_and_triangle_area_l777_777868

theorem parabola_line_equations_and_triangle_area :
  (∀ (F A B : ℝ × ℝ) (x y : ℝ) (FA FB : ℝ × ℝ),
    let parabola_eq := y^2 = 4 * x,
        focus_F := F = (1, 0),
        P := (2, 0),
        line_AB := (A_x = 2 ∧ A_y > 0) ∨ (A_y = k * (A_x - 2)) ∧ (A_x + A_y * ⋆ = 0),
        intersect_A: parabola_eq ∧ line_AB,
        intersect_B: parabola_eq ∧ line_AB ∧ A ≠ B
    in
    (FA = (A_x - 1, A_y) ∧ FB = (B_x - 1, B_y) ∧ FA ⋅ FB = -11)
    → ((line_eq1 : (x - y - 2 = 0) ∨ (x + y - 2 = 0)) ∧ 
         (min_area_triangle_ABF = 2 * sqrt 2)))
:= 
  sorry

end parabola_line_equations_and_triangle_area_l777_777868


namespace sum_of_roots_of_quadratic_l777_777684

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l777_777684


namespace parabola_intersections_l777_777656

noncomputable def intersection_points : set (ℝ × ℝ) :=
  {p | (∃ (x : ℝ), p = (x, 2 * x^2 + 5 * x - 3) ∧ (2 * x^2 + 5 * x - 3 = x^2 + 2))}

theorem parabola_intersections :
  intersection_points = {(-5, 27), (1, 3)} :=
by
  sorry

end parabola_intersections_l777_777656


namespace probability_of_same_length_l777_777967

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777967


namespace square_difference_l777_777496

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l777_777496


namespace performance_difference_average_jumps_total_score_exceeds_threshold_l777_777732

def performance_records : List Int := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]
def standard_jumps : Int := 160
def reward_threshold : Int := 65

theorem performance_difference : 
  List.maximum performance_records - List.minimum performance_records = 27 := sorry

theorem average_jumps : 
  (standard_jumps + (List.sum performance_records / List.length performance_records)) = 166 := sorry

theorem total_score_exceeds_threshold : 
  let points_above := List.sum (List.filter (λ x => x > 0) performance_records)
  let points_below := List.sum (List.filter (λ x => x < 0) performance_records)
  (points_above * 1 + points_below * (-0.5)) > reward_threshold := sorry 

end performance_difference_average_jumps_total_score_exceeds_threshold_l777_777732


namespace seating_arrangements_count_l777_777260

-- Define people
inductive Person
| Alice | Bob | Carla | Derek | Eric | Fiona

open Person

-- Define the conditions as properties on seats
def satisfies_conditions (arrangement : List Person) : Prop :=
  let alice_next_to (x : Person) : Prop := ∃ i, i < arrangement.length - 1 ∧ arrangement[i] = Alice ∧ arrangement[i + 1] = x
  let derek_next_to (x : Person) : Prop := ∃ i, i < arrangement.length - 1 ∧ arrangement[i] = Derek ∧ arrangement[i + 1] = x
  ¬ alice_next_to Bob ∧ ¬ alice_next_to Carla ∧ ¬ derek_next_to Eric ∧ ¬ derek_next_to Fiona

-- Define the proof statement
theorem seating_arrangements_count : (List.permutations [Alice, Bob, Carla, Derek, Eric, Fiona].filter satisfies_conditions).length = 256 := 
sorry

end seating_arrangements_count_l777_777260


namespace percentage_increase_l777_777635

theorem percentage_increase (P : ℝ) (x : ℝ) :
  P * (1 + x / 100) * 0.8 = 1.04 * P -> x = 30 :=
by
  intro h
  have : (1 + x / 100) * 0.8 = 1.04 := by linarith
  sorry

end percentage_increase_l777_777635


namespace y_intercept_of_line_l777_777664

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l777_777664


namespace sock_pairs_l777_777904

def total_ways (n_white n_brown n_blue n_red : ℕ) : ℕ :=
  n_blue * n_white + n_blue * n_brown + n_blue * n_red

theorem sock_pairs (n_white n_brown n_blue n_red : ℕ) (h_white : n_white = 5) (h_brown : n_brown = 4) (h_blue : n_blue = 2) (h_red : n_red = 1) :
  total_ways n_white n_brown n_blue n_red = 20 := by
  -- insert the proof steps here
  sorry

end sock_pairs_l777_777904


namespace find_x_arccos_l777_777425

theorem find_x_arccos :
  ∀ (x : ℝ), arccos(2 * x - 1) = real.pi / 4 → x = (real.sqrt 2 + 2) / 4 :=
by
  -- Proof would go here but is not required
  sorry

end find_x_arccos_l777_777425


namespace parallel_vectors_implies_x_eq_neg_one_l777_777427

variables (x : ℝ)

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors_implies_x_eq_neg_one : (a.1 / b.1 = a.2 / b.2) → x = -1 := by
  intro h
  sorry

end parallel_vectors_implies_x_eq_neg_one_l777_777427


namespace modular_arithmetic_problem_l777_777571

-- Define the conditions:
def n : ℤ := 21  -- We know n = 21 from the conditions 3n ≡ 1 mod 31 and 0 ≤ n < 31.

-- State and define the problem:
theorem modular_arithmetic_problem :
  0 ≤ n ∧ n < 31 ∧ (3 * n % 31 = 1) → ((2 ^ n) ^ 3 - 2) % 31 = 6 := by
  intros
  sorry

end modular_arithmetic_problem_l777_777571


namespace ab_is_zero_l777_777838

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l777_777838


namespace chord_length_l777_777783

/-- Define the radii of the circles C1 and C2 respectively. Also define the collinearity of their centers
, i.e., O1, O2, and O3. Then, formulate the problem of finding the length of the chord AB. --/
theorem chord_length (r1 r2 r3 : ℝ) (HO1 O3T AB : ℝ) (m n p : ℕ) (h_r1 : r1 = 5) (h_r2 : r2 = 12)
  (h_r3 : r3 = 17) (h_ho1 : HO1 = 5) (h_o3t : O3T = 22 / 12) (h_ab : AB = (real.sqrt 10283) / 3)
  (h_rel_prime : nat.coprime m p) (h_square_free : ∀ k : ℕ, k^2 ∣ n → k = 1) :
  AB = (real.sqrt 10283) / 3 := by
sorry

end chord_length_l777_777783


namespace chess_tournament_games_l777_777719

theorem chess_tournament_games (n : ℕ) (h1 : n = 18) (h2 : ∀ i j, i ≠ j → (i ∈ finset.range n → j ∈ finset.range n → true)) : 
  2 * n * (n - 1) = 612 :=
by
  sorry

end chess_tournament_games_l777_777719


namespace temperature_in_middle_of_night_l777_777597

theorem temperature_in_middle_of_night :
  let initial_temp := -3
  let noon_temp := initial_temp + 6
  let midnight_temp := noon_temp - 7
  midnight_temp = -4 :=
by
  -- Definitions and proof sketch here
  let initial_temp : ℝ := -3
  let noon_temp := initial_temp + 6
  let midnight_temp := noon_temp - 7
  show midnight_temp = -4, from sorry

end temperature_in_middle_of_night_l777_777597


namespace proof_x_minus_y_squared_l777_777502

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l777_777502


namespace hexagon_probability_same_length_l777_777976

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777976


namespace real_roots_range_l777_777423

theorem real_roots_range (a : ℝ) : (∃ x : ℝ, a*x^2 + 2*x - 1 = 0) ↔ (a >= -1 ∧ a ≠ 0) :=
by 
  sorry

end real_roots_range_l777_777423


namespace no_such_convex_hexagon_and_point_exists_l777_777367

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777367


namespace find_xy_l777_777803

theorem find_xy (x y : ℕ) (h : 5 ^ (-6 : ℤ) = 5 ^ (56 / x : ℤ) / (5 ^ (32 / y : ℤ) * 5 ^ (44 / x : ℤ))) :
  x = 3 ∧ y = 4 :=
by
  sorry

end find_xy_l777_777803


namespace symmetric_points_y_axis_l777_777079

theorem symmetric_points_y_axis :
  ∀ (m n : ℝ), (m + 4 = 0) → (n = 3) → (m + n) ^ 2023 = -1 :=
by
  intros m n Hm Hn
  sorry

end symmetric_points_y_axis_l777_777079


namespace length_of_bridge_l777_777746

-- Definitions of the conditions in Lean 4
def walking_speed_km_per_hr := 10
def time_crossing_bridge_min := 18

-- Unit conversions
def walking_speed_m_per_min := walking_speed_km_per_hr * 1000 / 60

theorem length_of_bridge :
  (walking_speed_m_per_min * time_crossing_bridge_min).round = 3000 :=
by
  sorry

end length_of_bridge_l777_777746


namespace not_a_term_l777_777865

def sequence (n : ℕ) : ℤ := 25 - 2 * n

theorem not_a_term (n : ℕ) : ¬ ∃ n : ℕ, sequence n = 2 :=
by
  sorry

end not_a_term_l777_777865


namespace y_intercept_of_line_l777_777665

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l777_777665


namespace angles_equal_l777_777128

/-
Given two circles intersecting at points P and Q, 
and a line intersecting the segment PQ at points A, B, C, and D in that order,
prove that the angle ∠APB is equal to the angle ∠CQD.
-/
theorem angles_equal 
  (circles : Circle → Circle)
  (P Q A B C D : Point)
  (h_intersectPQ : ∃ c1 c2, c1 ≠ c2 ∧ c1 ∩ c2 = {P, Q})
  (h_line : Line)
  (h_intersectLine : ∃ points, h_line ∩ [P, Q] = {A, B, C, D}) :
  ∠APB = ∠CQD :=
sorry

end angles_equal_l777_777128


namespace isosceles_triangle_largest_angle_l777_777074

theorem isosceles_triangle_largest_angle (a b c : ℝ) 
  (h1 : a = b)
  (h2 : c + 50 + 50 = 180) : 
  c = 80 :=
by sorry

end isosceles_triangle_largest_angle_l777_777074


namespace true_inequality_B_true_inequality_C_l777_777214

-- Given conditions and definitions
def sin_lt_x : Prop := ∀ x : ℝ, sin x < x

def exp_ge_exp_val : Prop := ∀ x : ℝ, exp x ≥ x * exp 1

def ln_lt_sqrt_x : Prop := ∀ x : ℝ, x > 0 → log x < real.sqrt x
  
def ln_ge_1_minus_div_x : Prop := ∀ x : ℝ, x > 0 → log x ≥ (1 - x) / x

-- New proof goals according to the correct solutions
theorem true_inequality_B : exp_ge_exp_val := sorry

theorem true_inequality_C : ln_lt_sqrt_x := sorry

end true_inequality_B_true_inequality_C_l777_777214


namespace probability_same_length_segments_l777_777942

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777942


namespace solve_time_for_n_presses_l777_777549

-- Definitions based on conditions
def number_of_papers : ℕ := 500000
def time_for_40_presses : ℝ := 12
def number_of_presses_40 : ℕ := 40
def time_for_30_presses : ℝ := 15.999999999999998
def number_of_presses_30 : ℕ := 30

-- Main statement to be proven.
-- The statement confirms that given the conditions, the time T for n presses equals 480 hours / n
theorem solve_time_for_n_presses (n : ℕ) (h1 : number_of_presses_40 * time_for_40_presses = 480)
    (h2 : number_of_presses_30 * time_for_30_presses ≈ 480) :
    ∃ T : ℝ, T = 480 / n :=
by 
  sorry

end solve_time_for_n_presses_l777_777549


namespace magnitude_of_b_l777_777876

variables (a b : ℝ)(a_add_b : ℝ)
variables (dot_product : ℝ)(a_magnitude : ℝ := 1)(a_add_b_magnitude : ℝ := sqrt 7)(a_b_angle : ℝ := π / 3)

theorem magnitude_of_b (ha : a_magnitude = 1)
  (hab : a_add_b_magnitude = sqrt 7)
  (hangle : dot_product = π / 3) :
  a_add_b = 2 :=
  sorry

end magnitude_of_b_l777_777876


namespace post_office_packages_per_day_l777_777190

noncomputable def total_days (months: ℕ) (days_per_month: ℕ) : ℕ :=
  months * days_per_month

noncomputable def total_letters (letters_per_day: ℕ) (total_days: ℕ) : ℕ :=
  letters_per_day * total_days

noncomputable def total_packages (total_mail: ℕ) (total_letters: ℕ) : ℕ :=
  total_mail - total_letters

noncomputable def packages_per_day (total_packages: ℕ) (total_days: ℕ) : ℕ :=
  total_packages / total_days

theorem post_office_packages_per_day:
  let months := 6 in
  let days_per_month := 30 in
  let letters_per_day := 60 in
  let total_mail := 14400 in
  let total_days := total_days months days_per_month in
  let total_letters := total_letters letters_per_day total_days in
  let total_packages := total_packages total_mail total_letters in
  packages_per_day total_packages total_days = 20 := by
  sorry

end post_office_packages_per_day_l777_777190


namespace s_is_contrapositive_of_p_l777_777515

variables (A B : Prop)

-- Definitions based on given problem conditions
def p := A → B
def r := ¬A → ¬B
def s := ¬B → ¬A

-- Lean 4 statement to prove
theorem s_is_contrapositive_of_p : s = (¬B → ¬A) := 
sorry

end s_is_contrapositive_of_p_l777_777515


namespace hyperbola_foci_coincide_with_ellipse_major_axis_endpoints_l777_777513

theorem hyperbola_foci_coincide_with_ellipse_major_axis_endpoints :
  (∀ (m : ℝ), hyperbola_with_foci_eq_ellipse_major_axis_endpoints m → m = 2) :=
by
  -- Definitions
  /- The hyperbola equation -/
  def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

  /- The ellipse equation -/
  def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

  /- Condition of the problem -/
  def hyperbola_with_foci_eq_ellipse_major_axis_endpoints (m : ℝ) : Prop :=
    ∃ a : ℝ, a = 2 ∧ ∃ x y : ℝ, hyperbola 2 m x y ∧ ellipse (sqrt 3) 2 x y ∧ 
      (x = 0 ∧ (y = 2 ∨ y = -2))

  -- Sorry to skip the proof
  sorry

end hyperbola_foci_coincide_with_ellipse_major_axis_endpoints_l777_777513


namespace no_such_convex_hexagon_and_point_exists_l777_777372

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777372


namespace parallel_vectors_dot_product_condition_l777_777842

-- Definitions for the problem
def m (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def n : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)

-- Proof problem (1) statement
theorem parallel_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 3) (h1 : ∃ k : ℝ, m x = k • n) :
  x = Real.pi / 3 :=
sorry

-- Proof problem (2) statement
theorem dot_product_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 3) (h1 : (m x).fst * n.fst + (m x).snd * n.snd = 3 / 5) :
  Real.sin (x - Real.pi / 12) = -Real.sqrt 2 / 10 :=
sorry

end parallel_vectors_dot_product_condition_l777_777842


namespace sum_of_segments_on_sides_of_triangle_l777_777654

theorem sum_of_segments_on_sides_of_triangle 
  (a b c : ℝ) 
  (h_acute : a² + b² > c²) 
  (h1 : a = 7) 
  (h2 : b = 4) 
  (h3 : c = 5) 
  (y : ℝ) 
  (h_y : y = 20 / 7) :
  a + b + c + y = 132 / 7 := by
  sorry

end sum_of_segments_on_sides_of_triangle_l777_777654


namespace largest_prime_factor_5292_l777_777678

theorem largest_prime_factor_5292 :
  ∃ p, nat.prime p ∧ p ∣ 5292 ∧ ∀ q, nat.prime q ∧ q ∣ 5292 → q ≤ p :=
sorry

end largest_prime_factor_5292_l777_777678


namespace fly_distance_bound_l777_777243

-- Definitions based on the conditions
variables {a b : ℝ} -- sides of the right triangle
def hypotenuse := 5
def right_triangle (a b : ℝ) : Prop := a^2 + b^2 = hypotenuse^2
def fly_right_angle_turns := ∀ (n : ℕ), n = 10 -- fly makes right-angle turns exactly 10 times

-- The proof problem
theorem fly_distance_bound (a b : ℝ) (h_triangle : right_triangle a b) (h_turns : fly_right_angle_turns) :
  (∑ i in finset.range 10, if i % 2 = 0 then b else a) ≤ 10 := sorry

end fly_distance_bound_l777_777243


namespace geometric_sequence_common_ratio_l777_777064

variable {a1 a3 a2 a4 q : ℝ}

theorem geometric_sequence_common_ratio (h1 : a1 * a3 = 16) (h2 : a2 + a4 = 12)
  (h3 : a3 = a1 * q ^ 2) (h4 : a4 = a1 * q ^ 3) (h5 : a2 = a1 * q) : q = ± Real.sqrt 2 :=
by
  sorry

end geometric_sequence_common_ratio_l777_777064


namespace g_triple_application_l777_777887

def g (x : ℕ) : ℕ := 7 * x + 3

theorem g_triple_application : g (g (g 3)) = 1200 :=
by
  sorry

end g_triple_application_l777_777887


namespace no_angle_sin_eq_sec_no_angle_cos_eq_csc_l777_777611

theorem no_angle_sin_eq_sec (α : ℝ) : sin α ≠ (1 / cos α) :=
by
sorry

theorem no_angle_cos_eq_csc (α : ℝ) : cos α ≠ (1 / sin α) :=
by
sorry

end no_angle_sin_eq_sec_no_angle_cos_eq_csc_l777_777611


namespace inequality_transformation_l777_777506

variable {a b c d : ℝ}

theorem inequality_transformation
  (h1 : a < b)
  (h2 : b < 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (d / a) < (c / a) :=
by
  sorry

end inequality_transformation_l777_777506


namespace density_function_Z_l777_777547

-- Define the probability density functions for X and Y.
def f1 (x : ℝ) : ℝ := if 0 ≤ x then exp (-x) else 0
def f2 (y : ℝ) : ℝ := if 0 ≤ y then (1/2) * exp (-y / 2) else 0

-- Define the convolution of f1 and f2 to get the density function of Z = X + Y.
noncomputable def g (z : ℝ) : ℝ :=
  if 0 ≤ z then
    ∫ x in 0..z, f1 x * f2 (z - x)
  else 0

-- The target density function of Z = X + Y
def target_g (z : ℝ) : ℝ :=
  if 0 ≤ z then exp (-z / 2) * (1 - exp (-z / 2)) else 0

-- Prove that g equals the target density function target_g.
theorem density_function_Z :
  ∀ z : ℝ, g z = target_g z :=
by
  sorry

end density_function_Z_l777_777547


namespace lines_concurrence_l777_777180

open Real Topology Geometry

noncomputable def proof_problem := 
  let A : Point := sorry
  let O : Point := sorry
  let C : Point := sorry
  let ω : Circle := sorry 
  let M : Point := sorry
  let N : Point := sorry
  let K : Point := sorry
  let L : Point := sorry

-- Conditions: A, O, C, ω form triangle inscribed in circle ω
-- M, N, K, L are defined as intersections of AB, BC, CD, DA with ω

theorem lines_concurrence :
  concurrent (line_through M N) (line_through K L) (tangent_line ω M) (tangent_line ω K) :=
sorry

end lines_concurrence_l777_777180


namespace parallel_PL_CD_l777_777895

open EuclideanGeometry

noncomputable theory

-- Definitions
variables {A B C D E G P L : Point}

-- Conditions
axiom midpoint_D : is_midpoint D A B
axiom midpoint_E : is_midpoint E A C
axiom intersect_G : intersect BE CD G
axiom circumcircle_intersect_P : intersects (circumcircle (triangle ABE)) (circumcircle (triangle ACD)) P ∧ P ≠ A
axiom extension_AG_L : intersects (extension AG) (circumcircle (triangle ACD)) L ∧ L ≠ A

-- Statement to be proven
theorem parallel_PL_CD :
  PL ∥ CD :=
sorry

end parallel_PL_CD_l777_777895


namespace sum_roots_eq_six_l777_777691

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l777_777691


namespace sequence_geometric_and_general_formula_find_minimum_n_l777_777822

theorem sequence_geometric_and_general_formula 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) :=
sorry

theorem find_minimum_n 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (b T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n)
  (h2 : ∀ n : ℕ, b n = (2 * n + 1) * a n + (2 * n + 1))
  (h3 : T 0 = 0)
  (h4 : ∀ n : ℕ, T (n + 1) = T n + b (n + 1)) :
  ∃ n : ℕ, n ≥ 1 ∧ (T n - 2) / (2 * n - 1) > 2010 :=
sorry

end sequence_geometric_and_general_formula_find_minimum_n_l777_777822


namespace total_questions_attempted_l777_777073

theorem total_questions_attempted (C W T : ℕ) (hC : C = 42) (h_score : 4 * C - W = 150) : T = C + W → T = 60 :=
by
  sorry

end total_questions_attempted_l777_777073


namespace greatest_area_triangle_APC_l777_777204

variables (A B C P : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
variables (a b c : ℝ) (s : ℝ)
variables (AB BC CA : ℝ)
variables (hB max_area : ℝ)

axiom sides : AB = 10 ∧ BC = 17 ∧ CA = 21
axiom on_circle : P lies_on (circle_with_diameter AB)

def semi_perimeter : ℝ := (AB + BC + CA) / 2

def area_triangle_ABC (s AB BC CA : ℝ) : ℝ :=
  Math.sqrt (s * (s - AB) * (s - BC) * (s - CA))

def altitude_B : ℝ := 2 * area_triangle_ABC s AB BC CA / CA

noncomputable def max_distance_p_to_ac : ℝ := (AB / 2) + (altitude_B / 2)

noncomputable def greatest_possible_area (AC max_distance_p_to_ac : ℝ) : ℝ :=
  (1 / 2) * AC * max_distance_p_to_ac

theorem greatest_area_triangle_APC : greatest_possible_area CA max_distance_p_to_ac = 94.5 :=
by
  have h_sides := sides,
  have h_semi_perimeter : semi_perimeter = 24 := sorry,
  have h_area_triangle : area_triangle_ABC semi_perimeter AB BC CA = 84 := sorry,
  have h_altitude_B : altitude_B = 8 := sorry,
  have h_max_distance : max_distance_p_to_ac = 9 := sorry,
  have h_greatest_possible_area : greatest_possible_area = 94.5 := sorry,
  exact h_greatest_possible_area

end greatest_area_triangle_APC_l777_777204


namespace centers_do_not_always_intersect_other_rectangles_l777_777250

theorem centers_do_not_always_intersect_other_rectangles (R : Type) [rectangle R] (divisions : list (R × list R)) :
  ∃ p1 p2 : R, (∃ q : R, ¬ intersects_center_line p1 p2 q)
sorry

end centers_do_not_always_intersect_other_rectangles_l777_777250


namespace probability_same_length_segments_l777_777941

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777941


namespace tulip_price_correct_l777_777134

-- Initial conditions
def first_day_tulips : ℕ := 30
def first_day_roses : ℕ := 20
def second_day_tulips : ℕ := 60
def second_day_roses : ℕ := 40
def third_day_tulips : ℕ := 6
def third_day_roses : ℕ := 16
def rose_price : ℝ := 3
def total_revenue : ℝ := 420

-- Question: What is the price of one tulip?
def tulip_price (T : ℝ) : ℝ :=
    first_day_tulips * T + first_day_roses * rose_price +
    second_day_tulips * T + second_day_roses * rose_price +
    third_day_tulips * T + third_day_roses * rose_price

-- Proof problem statement
theorem tulip_price_correct (T : ℝ) : tulip_price T = total_revenue → T = 2 :=
by
  sorry

end tulip_price_correct_l777_777134


namespace probability_red_or_white_l777_777235

theorem probability_red_or_white (total_marbles : ℕ) (blue_marbles : ℕ) (red_marbles : ℕ) :
  total_marbles = 150 → blue_marbles = 24 → red_marbles = 37 →
  (total_marbles - blue_marbles - red_marbles) / total_marbles + red_marbles / total_marbles = 21 / 25 :=
by
  intros h_total h_blue h_red
  rw [h_total, h_blue, h_red]
  have h_white : total_marbles - blue_marbles - red_marbles = 89 := by sorry
  sorry

end probability_red_or_white_l777_777235


namespace perpendicular_vectors_l777_777875

variable (a b : ℝ × ℝ) (x : ℝ)

noncomputable def vec_a := (1, 2 : ℝ)
noncomputable def vec_b := (x, 4 : ℝ)

theorem perpendicular_vectors :
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) → x = -8 := by
  sorry

end perpendicular_vectors_l777_777875


namespace average_velocity_eq_l777_777867

noncomputable def motion_eq : ℝ → ℝ := λ t => 1 - t + t^2

theorem average_velocity_eq (Δt : ℝ) :
  (motion_eq (3 + Δt) - motion_eq 3) / Δt = 5 + Δt :=
by
  sorry

end average_velocity_eq_l777_777867


namespace total_germs_l777_777913

-- Define variables and constants
namespace BiologyLab

def petri_dishes : ℕ := 75
def germs_per_dish : ℕ := 48

-- The goal is to prove that the total number of germs is as expected.
theorem total_germs : (petri_dishes * germs_per_dish) = 3600 :=
by
  -- Proof is omitted for this example
  sorry

end BiologyLab

end total_germs_l777_777913


namespace no_such_hexagon_exists_l777_777352

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777352


namespace expression_equals_sqrt2_l777_777276

theorem expression_equals_sqrt2 :
  (1 + Real.pi)^0 + 2 - abs (-3) + 2 * Real.sin (Real.pi / 4) = Real.sqrt 2 := by
  sorry

end expression_equals_sqrt2_l777_777276


namespace not_always_divisible_by_40_l777_777042

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem not_always_divisible_by_40 (p : ℕ) (hp_prime : is_prime p) (hp_geq7 : p ≥ 7) : ¬ (∀ p : ℕ, is_prime p ∧ p ≥ 7 → 40 ∣ (p^2 - 1)) := 
sorry

end not_always_divisible_by_40_l777_777042


namespace number_of_non_integer_power_terms_in_expansion_of_expression_l777_777040

-- Definitions for Lean
def a : ℝ := 2 * ∫ (x : ℝ) in -3..3, (x + |x|)

theorem number_of_non_integer_power_terms_in_expansion_of_expression :
  a = 18 →
  (let n := 18 in 
   let terms_with_non_integer_powers := (range (n + 1)).filter (λ r, 
     let power := 9 - (5 * r / 6) in ¬ ∃ k : ℤ, (power : ℝ) = k) in
   terms_with_non_integer_powers.length = 15) :=
by {
  sorry
}

end number_of_non_integer_power_terms_in_expansion_of_expression_l777_777040


namespace f_l777_777003

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then -1 / Real.sqrt x else 1 + x^2

def f_prime (x : ℝ) : ℝ :=
  if x > 0 then -1 / (2 * (x * Real.sqrt x))
  else if x < 0 then 2 * x
  else 0 -- This is only a placeholder, f' is not defined at x = 0

theorem f'_mul (h_pos : (0 : ℝ) < 4) (h_neg : (-1 : ℝ) < 0) :
  (f_prime 4) * (f_prime (-1)) = -1 / 8 :=
by
  have h_f4 : f_prime 4 = 1 / 16 := by
    dsimp [f_prime] ; sorry
  have h_f_neg1 : f_prime (-1) = -2 := by
    dsimp [f_prime] ; sorry
  rw [h_f4, h_f_neg1]
  dsimp
  norm_num

end f_l777_777003


namespace least_four_digit_7_heavy_l777_777259

-- Define 7-heavy number
def is7Heavy (n : ℕ) : Prop := n % 7 > 3

-- Define least four-digit number
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Theorem statement
theorem least_four_digit_7_heavy : ∃ n : ℕ, isFourDigit n ∧ is7Heavy n ∧ (∀ m : ℕ, isFourDigit m ∧ is7Heavy m → n ≤ m) :=
begin
  use 1000,
  split,
  { split,
    { exact nat.le_refl 1000 },
    { exact nat.le_of_lt_succ (nat.lt_succ_iff.mpr (nat.le_of_eq (nat.refl 1000))) } },
  { split,
    { exact dec_trivial },
    { intros m hm,
      cases hm with h1 h2,
      cases h1 with h3 h4,
      apply nat.le_of_lt_succ,
      exact nat.succ_le_of_lt h3 } }
end

end least_four_digit_7_heavy_l777_777259


namespace probability_same_length_segments_of_regular_hexagon_l777_777949

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777949


namespace count_numbers_200_to_499_with_1_l777_777033

def number_contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, (n / 10^k) % 10 = d

noncomputable def count_numbers_with_digit (start : ℕ) (end_ : ℕ) (d : ℕ) : ℕ :=
  (nat.range (end_ - start + 1)).count (λ i, number_contains_digit (start + i) d)

theorem count_numbers_200_to_499_with_1 :
  count_numbers_with_digit 200 499 1 = 57 :=
sorry

end count_numbers_200_to_499_with_1_l777_777033


namespace bob_distance_from_start_l777_777740

-- Assumptions based on given conditions
variables (s : ℝ) (walked_distance : ℝ)
def regular_octagon (s : ℝ) : Prop := sorry -- Placeholder for the definition of a regular octagon
def walk_along_octagon (s : ℝ) (distance : ℝ) : ℝ × ℝ := sorry -- Function to calculate final position after walking

-- Given conditions
axiom side_length : s = 3
axiom total_walk : walked_distance = 10
axiom is_regular_oct : regular_octagon s

-- Main problem statement
theorem bob_distance_from_start :
  ∀ (s walked_distance : ℝ), regular_octagon s → walk_along_octagon s walked_distance = (3 - 3 * (Real.sqrt 2) / 2, 3 * (Real.sqrt 2) / 2 - 4) →
  Real.sqrt ((3 - 3 * (Real.sqrt 2) / 2) ^ 2 + (3 * (Real.sqrt 2) / 2 - 4) ^ 2) = Real.sqrt 17 :=
by
  assume s walked_distance
  assume h1 : regular_octagon s
  assume h2 : walk_along_octagon s walked_distance = (3 - 3 * (Real.sqrt 2) / 2, 3 * (Real.sqrt 2) / 2 - 4)
  sorry

-- Note: The axiom and definition are placeholders. They need to be properly defined to match the conditions described.

end bob_distance_from_start_l777_777740


namespace sum_of_roots_of_quadratic_l777_777685

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l777_777685


namespace determine_A_plus_B_l777_777791

theorem determine_A_plus_B :
  ∃ (A B : ℚ), ((∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → 
  (Bx - 23) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) ∧
  (A + B = 11 / 9)) :=
sorry

end determine_A_plus_B_l777_777791


namespace range_of_exponential_function_l777_777505

theorem range_of_exponential_function :
  (∀ x : ℝ, (2:ℝ)^(x^2 + 1) ≤ (1 / 4)^(x - 2)) → (set.range (λ x, (2:ℝ)^x) = set.Icc (1 / 8) 2) :=
by
  intro h
  sorry

end range_of_exponential_function_l777_777505


namespace regular_hexagon_same_length_probability_l777_777969

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777969


namespace smallest_k_for_tournament_l777_777900

/-- 
  In a k-player tournament for k > 1,
  every player plays every other player exactly once. 
  Prove the smallest value of k such that it is possible 
  that for any two players, there was a third player who beat both of them is k = 7.
-/
theorem smallest_k_for_tournament (k : ℕ) (h : k > 1) :
  (∀ i j, ∃ m, m ≠ i ∧ m ≠ j ∧ m beats i ∧ m beats j) ↔ k = 7 :=
sorry

end smallest_k_for_tournament_l777_777900


namespace probability_of_same_length_l777_777960

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777960


namespace rebecca_income_percentage_l777_777153

-- Define Rebecca's initial income
def rebecca_initial_income : ℤ := 15000
-- Define Jimmy's income
def jimmy_income : ℤ := 18000
-- Define the increase in Rebecca's income
def rebecca_income_increase : ℤ := 7000

-- Define the new income for Rebecca after increase
def rebecca_new_income : ℤ := rebecca_initial_income + rebecca_income_increase
-- Define the new combined income
def new_combined_income : ℤ := rebecca_new_income + jimmy_income

-- State the theorem to prove that Rebecca's new income is 55% of the new combined income
theorem rebecca_income_percentage : 
  (rebecca_new_income * 100) / new_combined_income = 55 :=
sorry

end rebecca_income_percentage_l777_777153


namespace polynomial_example_properties_l777_777151

open Polynomial

noncomputable def polynomial_example : Polynomial ℚ :=
- (1 / 2) * (X^2 + X - 1) * (X^2 + 1)

theorem polynomial_example_properties :
  ∃ P : Polynomial ℚ, (X^2 + 1) ∣ P ∧ (X^3 + 1) ∣ (P - 1) :=
by
  use polynomial_example
  -- To complete the proof, one would typically verify the divisibility properties here.
  sorry

end polynomial_example_properties_l777_777151


namespace doug_fires_l777_777396

theorem doug_fires (D : ℝ) (Kai_fires : ℝ) (Eli_fires : ℝ) 
    (hKai : Kai_fires = 3 * D)
    (hEli : Eli_fires = 1.5 * D)
    (hTotal : D + Kai_fires + Eli_fires = 110) : 
  D = 20 := 
by
  sorry

end doug_fires_l777_777396


namespace PQ_bisects_BC_l777_777104

open EuclideanGeometry

variable {A B C D E F D1 D2 P Q : Point}

-- Conditions from step a)
axiom hABC : Triangle A B C
axiom hD : Foot D A B C
axiom hE : Foot E B A C
axiom hF : Foot F C A B
axiom hD1 : PointOnLine D1 E F ∧ Distance D F = Distance D1 E ∧ Between E D1 F
axiom hD2 : PointOnLine D2 E F ∧ Distance D E = Distance D2 F ∧ Between F E D2
axiom hP : BisectorOnLine D D1 P A B
axiom hQ : BisectorOnLine D D2 Q A C

-- Prove that PQ bisects BC
theorem PQ_bisects_BC : BisectsLine P Q B C :=
  sorry

end PQ_bisects_BC_l777_777104


namespace transform_sine_graph_l777_777653

theorem transform_sine_graph :
  ∀ (x : ℝ), shift (3 * sin (x - π / 3)) (-2 * π / 3) = 3 * sin (x + π / 3) :=
by
  sorry

end transform_sine_graph_l777_777653


namespace angle_A_is_30_degrees_l777_777483

theorem angle_A_is_30_degrees {A : ℝ} (hA_acute : 0 < A ∧ A < π / 2) (hA_sin : Real.sin A = 1 / 2) : A = π / 6 :=
sorry

end angle_A_is_30_degrees_l777_777483


namespace polynomial_int_coeff_n_P_deg_le_2_l777_777119

open Polynomial

theorem polynomial_int_coeff_n_P_deg_le_2 (P : Polynomial ℤ) :
    let n_P : ℤ := (Finset.filter (λ k, P.eval k ^ 2 = 1) (Finset.image Int.ofNat (Finset.range (P.degree.toNat + 1)))).card in
    n_P - P.degree ≤ 2 :=
by
  sorry

end polynomial_int_coeff_n_P_deg_le_2_l777_777119


namespace no_such_hexagon_exists_l777_777357

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777357


namespace bisect_LM_by_K_l777_777714

variable {triangle : Type} [Inhabit triangle]

structure Triangle (α : Type) :=
(center_L : Point α) -- The center of the circle passing through the excenters
(orthocenter_M : Point α) -- The orthocenter of the triangle
(medial_incenter_K : Point α) -- The incenter of the medial triangle

def bisects (A B midpoint : Point α) : Prop :=
dist A midpoint = dist midpoint B

theorem bisect_LM_by_K {T : Triangle α} :
  bisects T.center_L T.orthocenter_M T.medial_incenter_K :=
sorry

end bisect_LM_by_K_l777_777714


namespace max_planes_determined_by_15_points_l777_777248

theorem max_planes_determined_by_15_points : 
  (∀ (points : Finset (EuclideanSpace ℝ 3)), points.card = 15 → (∀ (p1 p2 p3 p4 : Points), p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 → ¬AffineIndependent ℝ ![p1, p2, p3, p4]) → choose 15 3 = 455) := 
sorry

end max_planes_determined_by_15_points_l777_777248


namespace arithmetic_sequence_common_difference_l777_777070

theorem arithmetic_sequence_common_difference (a d : ℕ) (n : ℕ) :
  a = 5 →
  (a + (n - 1) * d = 50) →
  (n * (a + (a + (n - 1) * d)) / 2 = 275) →
  d = 5 := 
by
  intros ha ha_n hs_n
  sorry

end arithmetic_sequence_common_difference_l777_777070


namespace inequality_sum_pow_l777_777436

theorem inequality_sum_pow
  (n : ℕ)
  (x : Fin n → ℝ)
  (a m s : ℝ)
  (hx_pos : ∀ i, 0 < x i)
  (ha_pos : 0 < a)
  (hm_pos : 0 < m)
  (h_sum : ∑ i, x i = s) :
  ∑ i, (a + (1 / (x i) ^ m)) ^ n ≥ n * (a + (n / s) ^ m) ^ n := by
  sorry

end inequality_sum_pow_l777_777436


namespace total_weight_of_nuts_l777_777736

theorem total_weight_of_nuts :
  let w_almonds := 0.14
  let w_pecans := 0.38
  w_almonds + w_pecans = 0.52 :=
by
  let w_almonds := 0.14
  let w_pecans := 0.38
  show w_almonds + w_pecans = 0.52,
  sorry

end total_weight_of_nuts_l777_777736


namespace g_at_3_l777_777176

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2

theorem g_at_3 : g 3 = 0 :=
by
  sorry

end g_at_3_l777_777176


namespace y_intercept_of_line_l777_777663

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l777_777663


namespace no_such_hexagon_exists_l777_777359

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777359


namespace num_integers_satisfying_cube_count_integers_satisfying_cube_l777_777025

theorem num_integers_satisfying_cube (n : ℤ) :
  (-100 < n^3 ∧ n^3 < 100) ↔ n ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4} := by
  sorry

theorem count_integers_satisfying_cube :
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := by
  sorry

end num_integers_satisfying_cube_count_integers_satisfying_cube_l777_777025


namespace seq_terms_are_integers_l777_777717

theorem seq_terms_are_integers
  (a : ℕ → ℕ)
  (h0 : a 0 = 1)
  (h1 : a 1 = 1)
  (rec : ∀ n > 0, a (n + 1) * a (n - 1) = a n * (a n + 1)) :
  ∀ n, a n ∈ ℤ := sorry

end seq_terms_are_integers_l777_777717


namespace range_of_a_range_of_t_l777_777583

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a ≥ f x) ↔ a ∈ set.Ici (-5 / 2) :=
by
  sorry

theorem range_of_t (t : ℝ) : (∀ x : ℝ, f x ≥ -t^2 - (5 / 2) * t - 1) ↔ t ∈ (set.Ici (1 / 2) ∪ set.Iic (-3)) :=
by
  sorry

end range_of_a_range_of_t_l777_777583


namespace find_number_is_16_l777_777713

noncomputable def find_number : ℝ :=
  let a := 0.15 * 40
  let b := λ x : ℝ, 0.25 * x + 2
  classical.some (classical.em (∃ x : ℝ, a = b x))

theorem find_number_is_16 : find_number = 16 :=
by
  sorry

end find_number_is_16_l777_777713


namespace part1_part2_l777_777991

def f (x a : ℝ) : ℝ := x * Real.log a - a * Real.log x

def g (x : ℝ) (a n : ℕ) : ℝ := ∑ k in Finset.range n, f (x^(k+1)) a

theorem part1 (a : ℝ) (h1 : 1 < a) : (∀ x ≥ 4, f x a ≥ 0) ↔ (2 ≤ a ∧ a ≤ 4) :=
sorry

theorem part2 (a : ℝ) (x : ℝ) (n : ℕ) (h1 : 1 < a) (h2 : 0 < x) (h3 : x < 1) (h4 : 1 ≤ n) :
  ¬ (g x n a + g x (3 * n) a = 2 * g x (2 * n) a) :=
sorry

end part1_part2_l777_777991


namespace more_student_tickets_l777_777652

-- Definitions of given conditions
def student_ticket_price : ℕ := 6
def nonstudent_ticket_price : ℕ := 9
def total_sales : ℕ := 10500
def total_tickets : ℕ := 1700

-- Definitions of the variables for student and nonstudent tickets
variables (S N : ℕ)

-- Lean statement of the problem
theorem more_student_tickets (h1 : student_ticket_price * S + nonstudent_ticket_price * N = total_sales)
                            (h2 : S + N = total_tickets) : S - N = 1500 :=
by
  sorry

end more_student_tickets_l777_777652


namespace arithmetic_sequence_common_difference_l777_777448

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1)
  (h2 : a 3 + a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l777_777448


namespace range_of_a_l777_777454

variables (a : ℝ)

def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, log 0.5 (x^2 + 2*x + a) = y

def q (a : ℝ) : Prop := ∀ (x y : ℝ), x < y → -(5 - 2*a)^x > -(5 - 2*a)^y

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l777_777454


namespace find_sin_phi_l777_777129

variables {R : Type*} [RealField R]
variables (u v w : EuclideanSpace R (Fin 3))

noncomputable def sin_angle_u_v (h1 : ‖u‖ = 2) (h2 : ‖v‖ = 4) (h3 : ‖w‖ = 6) (h4 : u × (u × v) = w) : Real :=
  let φ := Real.angle u v in
  Real.sin φ

theorem find_sin_phi (u v w : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖u‖ = 2) 
  (h2 : ‖v‖ = 4) 
  (h3 : ‖w‖ = 6) 
  (h4 : u × (u × v) = w) : 
  sin_angle_u_v u v w h1 h2 h3 h4 = 3 / 8 := 
by
  sorry

end find_sin_phi_l777_777129


namespace num_three_digit_integers_l777_777878

def valid_digits : Multiset ℕ := {1, 1, 4, 4, 4, 7}

/-- 
  The number of different positive three-digit integers that can be formed 
  using only the digits in the set $\{1, 1, 4, 4, 4, 7\}$, with no digit 
  used more times than it appears in the given set.
-/
theorem num_three_digit_integers : 
  let num1 : ℕ := 1
    let num4 : ℕ := 4
    let num7 : ℕ := 7
    (∀n : ℕ, n ∈ {num1, num4, num7} -> n ∈ valid_digits) 
    → 31 := 
sorry

end num_three_digit_integers_l777_777878


namespace unique_solution_to_exponential_poly_equation_l777_777031

noncomputable def polynomial_has_unique_real_solution : Prop :=
  ∃! x : ℝ, (2 : ℝ)^(3 * x + 3) - 3 * (2 : ℝ)^(2 * x + 1) - (2 : ℝ)^x + 1 = 0

theorem unique_solution_to_exponential_poly_equation :
  polynomial_has_unique_real_solution :=
sorry

end unique_solution_to_exponential_poly_equation_l777_777031


namespace sequoia_taller_than_maple_l777_777551

def height_maple_tree : ℚ := 13 + 3/4
def height_sequoia : ℚ := 20 + 1/2

theorem sequoia_taller_than_maple : (height_sequoia - height_maple_tree) = 6 + 3/4 :=
by
  sorry

end sequoia_taller_than_maple_l777_777551


namespace odd_factors_not_div_by_3_l777_777487

theorem odd_factors_not_div_by_3 (n : ℕ) (h : n = 2^2 * 3^2 * 7) :
  (finset.filter (λ x, odd x ∧ ¬ (3 ∣ x)) (finset.divisors n)).card = 2 :=
by {
  sorry
}

end odd_factors_not_div_by_3_l777_777487


namespace inequality_holds_l777_777504

theorem inequality_holds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := 
by
  sorry

end inequality_holds_l777_777504


namespace y_intercept_of_line_l777_777669

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l777_777669


namespace sum_of_roots_of_quadratic_l777_777683

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l777_777683


namespace similarity_coordinates_C_l777_777908

theorem similarity_coordinates_C (A B C : ℝ × ℝ) (ratio : ℝ) :
  A = (1,2) ∧ B = (2,1) ∧ C = (3,2) ∧ ratio = 2 →
  (exists C' : ℝ × ℝ, (C' = (6,4)) ∨ (C' = (-6,-4))) :=
by { intro h, sorry }

end similarity_coordinates_C_l777_777908


namespace divide_into_equal_parts_l777_777152

-- Definitions based on the conditions from step a
variables {A B C D O K : Type} 
variables (AC BD : A → B → C → A → D → K → Prop)

-- Given conditions as Lean definitions
def is_cyclic_quadrilateral (ABCD : A × B × C × D) : Prop :=
  ∃ circle, Circle.center circle = O ∧ Circle.circumscribes circle ABCD

def perpendicular_diagonals (AC BD : A → B → C → A → D → K → Prop): Prop :=
  ∀ (KA KB KC KD : Prop), KA ∧ KB ∧ KC ∧ KD → AC BD KA KB KC KD
   
def intersection_point (K : A → B → C → A → D → K → Prop) : Prop :=
  ∃ KA KB, AC K KA KB = BD K KA KB
   
-- Combining all the conditions
theorem divide_into_equal_parts (ABCD : A × B × C × D) 
  (circumcircle : is_cyclic_quadrilateral ABCD)
  (perp : perpendicular_diagonals AC BD)
  (inter : intersection_point K) :
  broken_line AO divides ABCD into two equal areas := 
sorry

end divide_into_equal_parts_l777_777152


namespace no_such_hexagon_exists_l777_777301

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777301


namespace min_balls_to_guarantee_12_l777_777727

theorem min_balls_to_guarantee_12 :
  ∀ (r g y b k : ℕ), 
    r = 30 → 
    g = 22 → 
    y = 18 → 
    b = 15 → 
    k = 10 → 
    (∃ n : ℕ, n = 55 ∧ 
      ∀ d : ℕ, d = n → 
        ∃ c : ℕ, (r - 12 < (r - r + d) ∨ g - 12 < (g - g + d) ∨ y - 12 < (y - y + d) ∨ b - 12 < (b - b + d) ∨ k - 10 < (k - k + d)))
sorry

end min_balls_to_guarantee_12_l777_777727


namespace no_such_hexagon_exists_l777_777327

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777327


namespace greg_additional_money_needed_l777_777023

noncomputable def price_scooter := 90
noncomputable def price_helmet := 30
noncomputable def price_lock := 15
noncomputable def discount_rate := 0.10
noncomputable def sales_tax_rate := 0.10
noncomputable def gift_card := 20
noncomputable def amount_saved := 57

theorem greg_additional_money_needed :
  let discounted_price_scooter := price_scooter * (1 - discount_rate)
      total_price_without_tax := discounted_price_scooter + price_helmet + price_lock
      sales_tax := sales_tax_rate * total_price_without_tax
      total_price_with_tax := total_price_without_tax + sales_tax
      price_after_gift_card := total_price_with_tax - gift_card
      additional_money_needed := price_after_gift_card - amount_saved
  in additional_money_needed = 61.60 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end greg_additional_money_needed_l777_777023


namespace elizabeth_net_profit_is_50_l777_777402

variables (ingredient_cost : ℕ) (bags_made : ℕ) (price_per_bag : ℕ)
variables (bags_sold_initial : ℕ) (discounted_bags : ℕ) (discounted_price : ℕ)
variable total_cost : ℕ := bags_made * ingredient_cost
variable revenue_initial : ℕ := bags_sold_initial * price_per_bag
variable revenue_discounted : ℕ := discounted_bags * discounted_price
variable total_revenue : ℕ := revenue_initial + revenue_discounted
variable net_profit : ℕ := total_revenue - total_cost

theorem elizabeth_net_profit_is_50 :
  ingredient_cost = 3 ∧ bags_made = 20 ∧ price_per_bag = 6 ∧ 
  bags_sold_initial = 15 ∧ discounted_bags = 5 ∧ discounted_price = 4 →
  net_profit = 50 :=
by 
  intros h, 
  cases h with h1 h2, 
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end elizabeth_net_profit_is_50_l777_777402


namespace max_profit_correctness_l777_777256

noncomputable def daily_purchase_max_profit := 
  let purchase_price := 4.2
  let selling_price := 6
  let return_price := 1.2
  let days_sold_10kg := 10
  let days_sold_6kg := 20
  let days_in_month := 30
  let profit_function (x : ℝ) := 
    10 * x * (selling_price - purchase_price) + 
    days_sold_6kg * 6 * (selling_price - purchase_price) + 
    days_sold_6kg * (x - 6) * (return_price - purchase_price)
  (6, profit_function 6)

theorem max_profit_correctness : daily_purchase_max_profit = (6, 324) :=
  sorry

end max_profit_correctness_l777_777256


namespace average_rent_before_increase_correct_l777_777741

def original_rent := 1400
def rent_increase_percentage := 0.20
def new_mean_rent := 870
def number_of_friends := 4

def average_rent_before_increase : Prop :=
  let total_rent_after_increase := new_mean_rent * number_of_friends
  let increase_for_one_friend := rent_increase_percentage * original_rent
  let total_rent_before_increase := total_rent_after_increase - increase_for_one_friend
  let average_rent_before_increase := total_rent_before_increase / number_of_friends
  average_rent_before_increase = 800

theorem average_rent_before_increase_correct : average_rent_before_increase :=
by
  sorry

end average_rent_before_increase_correct_l777_777741


namespace remainder_b96_div_50_l777_777569

theorem remainder_b96_div_50 (b : ℕ → ℕ) (h : ∀ n, b n = 7^n + 9^n) : b 96 % 50 = 2 :=
by
  -- The proof is omitted.
  sorry

end remainder_b96_div_50_l777_777569


namespace find_B_l777_777455

universe u

variable {α : Type u}
variable (a b c d : α)
variable (A : Set α)
variable (B : Set α)

def A_def : A = {a, b, c, d} := rfl

def size_B : B.card = 2 := sorry
def B_subset_A : B ⊆ A := sorry
def cond_1 : a ∈ B → c ∈ B := sorry
def cond_2 : d ∉ B → c ∉ B := sorry
def cond_3 : d ∈ B → b ∉ B := sorry

theorem find_B : B = {c, d} :=
by
  sorry

end find_B_l777_777455


namespace find_lambda_l777_777018

def vector := (ℝ × ℝ)

def a : vector := (-1, 1)
def b : vector := (1, 2)

def parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_lambda (λ : ℝ) : 
  parallel (2 * (a.1, a.2) + b) (a.1 - λ * b.1, a.2 - λ * b.2) → λ = -1/2 :=
by
  sorry

end find_lambda_l777_777018


namespace area_of_triangle_XYZ_is_756_l777_777181

-- Define the conditions of the problem
variables (X Y Z P Q R : Point)
variables (r XP PY height XY : ℝ)

-- Assumptions based on the problem statement:
-- Radius of the inscribed circle
axiom radius_incircle : r = 15

-- XP, PY lengths
axiom XP_length : XP = 18
axiom PY_length : PY = 24

-- Height from Z to XY
axiom height_from_Z : height = 36

-- Line segment XY
def XY := XP + PY

-- The area of triangle XYZ given base and height
def area_triangle_XYZ_base_height (XY : ℝ) (height : ℝ) : ℝ := 
  0.5 * XY * height

-- The area of triangle XYZ as provided by the solution
def area_triangle_XYZ_solution : ℝ :=
  radius_incircle * (42 + 8.4)

-- Prove the area of triangle XYZ is 756 square units
theorem area_of_triangle_XYZ_is_756 :
  area_triangle_XYZ_base_height XY height = 756 :=
by
  sorry

end area_of_triangle_XYZ_is_756_l777_777181


namespace area_of_smaller_circle_l777_777205

noncomputable def radius_smaller_circle : ℝ := sorry
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle

-- Given: PA = AB = 5
def PA : ℝ := 5
def AB : ℝ := 5

-- Final goal: The area of the smaller circle is 5/3 * π
theorem area_of_smaller_circle (r_s : ℝ) (rsq : r_s^2 = 5 / 3) : (π * r_s^2 = 5/3 * π) :=
by
  exact sorry

end area_of_smaller_circle_l777_777205


namespace area_under_g_l777_777160

noncomputable def g (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

def x0 := Real.pi / 3

theorem area_under_g :
  ∫ x in 0..x0, g x = 3 / 4 :=
by
  sorry

end area_under_g_l777_777160


namespace choose_numbers_l777_777578

theorem choose_numbers (n : ℕ) (h : n > 0) :
  ∃ S ⊆ finset.range (2^n + 1), S.card ≥ (2^(n-1) + n) ∧ 
  (∀ x y ∈ S, x ≠ y → ¬ (x + y) ∣ (x * y)) :=
by
  sorry

end choose_numbers_l777_777578


namespace flight_distance_to_Recife_l777_777703

theorem flight_distance_to_Recife (speed : ℝ) (distance_traveled : ℝ) (remaining_time_hours : ℝ) :
  speed = 864 ∧ distance_traveled = 1222 ∧ remaining_time_hours = 4 / 3 →
  distance_traveled + speed * remaining_time_hours ≈ 2400 :=
begin
  sorry
end

end flight_distance_to_Recife_l777_777703


namespace bertha_descendants_without_daughters_l777_777772

-- Definitions based on conditions
def num_daughters : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30
def daughters_with_daughters := (total_daughters_and_granddaughters - num_daughters) / 6

-- The number of Bertha's daughters who have no daughters:
def daughters_without_daughters := num_daughters - daughters_with_daughters
-- The number of Bertha's granddaughters:
def num_granddaughters := total_daughters_and_granddaughters - num_daughters
-- All granddaughters have no daughters:
def granddaughters_without_daughters := num_granddaughters

-- The total number of daughters and granddaughters without daughters
def total_without_daughters := daughters_without_daughters + granddaughters_without_daughters

-- Main theorem statement
theorem bertha_descendants_without_daughters :
  total_without_daughters = 26 :=
by
  sorry

end bertha_descendants_without_daughters_l777_777772


namespace line_intersects_circle_l777_777869

noncomputable def point := ℝ × ℝ

def line (p : point) : Prop := p.1 + p.2 = 2

def circle (p : point) : Prop := (p.1 - 1) ^ 2 + p.2 ^ 2 = 1

theorem line_intersects_circle :
  ∃ (p : point), line p ∧ circle p :=
sorry

end line_intersects_circle_l777_777869


namespace ab_is_zero_l777_777839

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l777_777839


namespace trapezoid_circumcenter_l777_777824

variables {A B C D : Type*} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
variables (A B C D : ℝ → ℝ)

-- Definition of a trapezoid
def is_trapezoid (A B C D : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] :=
  ∃ (base AD : set (A × D)), base ⊆ AD ∧ ∃ (B C : set (B × C)), B ⊆ C

-- Definition of the circumcenter of a triangle
def circumcenter (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] :=
  ∃ O, ∀ X, dist O X = dist O A ∧ dist O X = dist O B ∧ dist O X = dist O C

-- Statement: given the conditions, prove the required property
theorem trapezoid_circumcenter (A B C D : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
  (h_trap : is_trapezoid A B C D)
  (h_circcenter_ABC_on_BD : ∃ O, circumcenter A B C O ∧ O ∈ (line B D)) :
  ∃ P, circumcenter A B D P ∧ P ∈ (line A C) :=
sorry

end trapezoid_circumcenter_l777_777824


namespace no_such_hexagon_exists_l777_777324

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777324


namespace log_base_change_l777_777566

open Real

variables (y : ℝ) (l : ℝ)

theorem log_base_change (H1 : log 3 5 = y) (H2 : 3^y = 5) : log 9 125 = (3 / 2) * y :=
sorry

end log_base_change_l777_777566


namespace circumcircle_eq_l777_777872

noncomputable def A : (ℝ × ℝ) := (8, 0)
noncomputable def B : (ℝ × ℝ) := (0, 6)
noncomputable def O : (ℝ × ℝ) := (0, 0)
noncomputable def P : (ℝ × ℝ) := (-1, 5)

theorem circumcircle_eq :
  let C := (4, 3)
  ∃ eq_circle : (ℝ × ℝ → Prop), eq_circle := (λ (p : ℝ × ℝ), (p.1 - 4)^2 + (p.2 - 3)^2 = 25) ∧
  ∃ eq_tangent : ((ℝ × ℝ) * (ℝ × ℝ) → Prop),
    eq_tangent := (λ (p₁ p₂ : ℝ × ℝ), p₁.1 = -1 ∨ 21 * p₂.1 - 20 * p₂.2 + 121 = 0) :=
by
  sorry

end circumcircle_eq_l777_777872


namespace black_percentage_is_57_l777_777999

noncomputable def radius (n : ℕ) : ℝ := 3 * (n + 1)
noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

def black_areas_sum : ℝ :=
  let areas := List.init 4 (λ k => area (radius (2 * k)))
  areas.sum - areas.tail.sum

def total_area : ℝ := area (radius 6)

def black_percentage : ℝ := (black_areas_sum / total_area) * 100

theorem black_percentage_is_57 :
  black_percentage = 57 :=
sorry

end black_percentage_is_57_l777_777999


namespace area_outside_doghouse_l777_777773

def side_length_of_hexagon : ℝ := 2
def rope_length : ℝ := 3
def accessible_area : ℝ := 9 * Real.pi - 6 * Real.sqrt 3

theorem area_outside_doghouse :
  let hex_area := (3 * Real.sqrt 3) / 2 * (side_length_of_hexagon ^ 2)
  let circular_sector_area := Real.pi * (rope_length ^ 2)
  (circular_sector_area - hex_area) = accessible_area :=
by
  sorry

end area_outside_doghouse_l777_777773


namespace no_such_hexagon_exists_l777_777299

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777299


namespace largest_circle_center_is_A_l777_777183

-- Define the given lengths of the pentagon's sides
def AB : ℝ := 16
def BC : ℝ := 14
def CD : ℝ := 17
def DE : ℝ := 13
def AE : ℝ := 14

-- Define the radii of the circles centered at points A, B, C, D, E
variables (R_A R_B R_C R_D R_E : ℝ)

-- Conditions based on the problem statement
def radius_conditions : Prop :=
  R_A + R_B = AB ∧
  R_B + R_C = BC ∧
  R_C + R_D = CD ∧
  R_D + R_E = DE ∧
  R_E + R_A = AE

-- The main theorem to prove
theorem largest_circle_center_is_A (h : radius_conditions R_A R_B R_C R_D R_E) :
  10 ≥ R_A ∧ R_A ≥ R_B ∧ R_A ≥ R_C ∧ R_A ≥ R_D ∧ R_A ≥ R_E :=
by sorry

end largest_circle_center_is_A_l777_777183


namespace max_norm_of_c_l777_777830

theorem max_norm_of_c
  (a b : ℝ × ℝ)
  (c : ℝ × ℝ)
  (h1 : a.1^2 + a.2^2 = 1) -- a is a unit vector
  (h2 : b.1^2 + b.2^2 = 1) -- b is a unit vector
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) -- a and b are perpendicular
  (h4 : (a.1 - c.1) * (√3 * b.1 - c.1) + (a.2 - c.2) * (√3 * b.2 - c.2) = 1) -- dot product condition
  : ∀ x y : ℝ, x = c.1 → y = c.2 → (x^2 + y^2) ≤ (√2 + 1)^2 :=
  by
    sorry

end max_norm_of_c_l777_777830


namespace expected_number_of_balls_fixed_l777_777139

-- Defining the total number of balls
def numBalls : ℕ := 9

-- Defining the swaps made by Chris (first with second ball) and Silva (fourth with fifth ball)
def chrisSwap (balls : Fin numBalls → Fin numBalls) : Fin numBalls → Fin numBalls :=
  fun i => if i = 0 then balls 1 else if i = 1 then balls 0 else balls i

def silvaSwap (balls : Fin numBalls → Fin numBalls) : Fin numBalls → Fin numBalls :=
  fun i => if i = 3 then balls 4 else if i = 4 then balls 3 else balls i

-- Combined swap function
def combinedSwap (balls : Fin numBalls → Fin numBalls) : Fin numBalls → Fin numBalls :=
  silvaSwap ∘ chrisSwap balls

-- The expected number of balls that remain in their original positions
def expectedBallsInOriginalPosition : ℕ :=
  (Finset.univ.filter (λ i => combinedSwap id i = i)).card

theorem expected_number_of_balls_fixed : expectedBallsInOriginalPosition = 5 := by
  -- Proof is not required, so we use sorry
  sorry

end expected_number_of_balls_fixed_l777_777139


namespace digit_zero_count_l777_777029

theorem digit_zero_count (n : ℕ) (h₁ : n ≤ 2376) : 
  (∑ i in Finset.range (2376 + 1), if '0' ∈ (i.digits 10).toFinset then 1 else 0) = 578 :=
sorry

end digit_zero_count_l777_777029


namespace no_convex_hexagon_with_point_M_l777_777315

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777315


namespace booksSoldOnMonday_l777_777931

noncomputable def booksStock : ℕ := 620
noncomputable def unsoldPercentage : ℝ := 54.83870967741935 / 100
noncomputable def booksSoldTuesday : ℕ := 82
noncomputable def booksSoldWednesday : ℕ := 60
noncomputable def booksSoldThursday : ℕ := 48
noncomputable def booksSoldFriday : ℕ := 40

theorem booksSoldOnMonday :
  let unsoldBooks : ℕ := (booksStock * unsoldPercentage).round
  let totalBooksSoldTtoF : ℕ := booksSoldTuesday + booksSoldWednesday + booksSoldThursday + booksSoldFriday
  let totalBooksSold : ℕ := booksStock - unsoldBooks
  let booksSoldMonday : ℕ := totalBooksSold - totalBooksSoldTtoF
  booksSoldMonday = 50 := by
  sorry

end booksSoldOnMonday_l777_777931


namespace problem_l777_777850

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable (d r : ℝ)
variable (n : ℕ)

-- Conditions for arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Conditions for geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, b (n + 1) = b n * r

-- Specific conditions given for sequences
def conditions_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
a 1 + (a 1 + 4 * d) + (a 1 + 8 * d) = 9

def conditions_b (b : ℕ → ℝ) (r : ℝ) : Prop :=
b 2 * (b 2 * r ^ 3) * (b 2 * r ^ 6) = 3 * real.sqrt 3

-- Main statement to be proved
theorem problem (a : ℕ → ℝ) (b : ℕ → ℝ) (d r : ℝ) : 
  is_arithmetic_sequence a d → 
  is_geometric_sequence b r → 
  conditions_a a d → 
  conditions_b b r → 
  (a 2 + a 8) / (1 + b 2 * b 8) = 3 / 2 :=
by 
  sorry

end problem_l777_777850


namespace radius_of_sector_eq_seven_l777_777848

-- Define the given conditions
def central_angle : ℝ := (5 * Real.pi) / 7
def perimeter : ℝ := 5 * Real.pi + 14

-- The goal is to prove that the radius is 7
theorem radius_of_sector_eq_seven (r : ℝ) (h_angle : central_angle = (5 * Real.pi) / 7) (h_perimeter : perimeter = 5 * Real.pi + 14) :
  (5 / 7 * Real.pi * r + 2 * r = 5 * Real.pi + 14) → r = 7 :=
by
  sorry

end radius_of_sector_eq_seven_l777_777848


namespace correct_number_of_propositions_l777_777825

variables (l m n : Line) (α β : Plane)

-- Definition of conditions
def prop1 : Prop := (m ∥ l ∧ n ∥ l) → m ∥ n
def prop2 : Prop := (m ⟂ α ∧ m ∥ β) → α ⟂ β
def prop3 : Prop := (m ∥ α ∧ n ∥ α) → m ∥ n
def prop4 : Prop := (m ⟂ β ∧ α ⟂ β) → (m ∥ α ∨ m ⊆ α)

-- The proof problem to be solved
theorem correct_number_of_propositions : 
  ∃ props_correct : ℕ,
    ((if prop1 then 1 else 0) + 
     (if prop2 then 1 else 0) +
     (if prop3 then 1 else 0) +
     (if prop4 then 1 else 0)) = props_correct
    ∧ props_correct = 3 :=
begin
  sorry
end

end correct_number_of_propositions_l777_777825


namespace no_such_convex_hexagon_exists_l777_777385

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777385


namespace negation_ln_eq_x_minus_1_l777_777184

theorem negation_ln_eq_x_minus_1 :
  ¬(∃ x : ℝ, 0 < x ∧ Real.log x = x - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by 
  sorry

end negation_ln_eq_x_minus_1_l777_777184


namespace parabola_intersects_x_axis_l777_777188

theorem parabola_intersects_x_axis :
  ∀ m : ℝ, (m^2 - m - 1 = 0) → (-2 * m^2 + 2 * m + 2023 = 2021) :=
by 
intros m hm
/-
  Given condition: m^2 - m - 1 = 0
  We need to show: -2 * m^2 + 2 * m + 2023 = 2021
-/
sorry

end parabola_intersects_x_axis_l777_777188


namespace no_such_hexagon_exists_l777_777328

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777328


namespace no_such_convex_hexagon_and_point_exists_l777_777375

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777375


namespace find_z_l777_777473

-- Definitions of the conditions
def equation_1 (x y : ℝ) : Prop := x^2 - 3 * x + 6 = y - 10
def equation_2 (y z : ℝ) : Prop := y = 2 * z
def x_value (x : ℝ) : Prop := x = -5

-- Lean theorem statement
theorem find_z (x y z : ℝ) (h1 : equation_1 x y) (h2 : equation_2 y z) (h3 : x_value x) : z = 28 :=
sorry

end find_z_l777_777473


namespace dogs_sold_correct_l777_777770

-- Definitions based on conditions
def ratio_cats_to_dogs (cats dogs : ℕ) := 2 * dogs = cats

-- Given conditions
def cats_sold := 16
def dogs_sold := 8

-- The theorem to prove
theorem dogs_sold_correct (h : ratio_cats_to_dogs cats_sold dogs_sold) : dogs_sold = 8 :=
by
  sorry

end dogs_sold_correct_l777_777770


namespace positive_when_x_negative_l777_777047

theorem positive_when_x_negative (x : ℝ) (h : x < 0) : (x / |x|)^2 > 0 := by
  sorry

end positive_when_x_negative_l777_777047


namespace performance_difference_average_jumps_total_score_exceeds_threshold_l777_777733

def performance_records : List Int := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]
def standard_jumps : Int := 160
def reward_threshold : Int := 65

theorem performance_difference : 
  List.maximum performance_records - List.minimum performance_records = 27 := sorry

theorem average_jumps : 
  (standard_jumps + (List.sum performance_records / List.length performance_records)) = 166 := sorry

theorem total_score_exceeds_threshold : 
  let points_above := List.sum (List.filter (λ x => x > 0) performance_records)
  let points_below := List.sum (List.filter (λ x => x < 0) performance_records)
  (points_above * 1 + points_below * (-0.5)) > reward_threshold := sorry 

end performance_difference_average_jumps_total_score_exceeds_threshold_l777_777733


namespace exists_rational_for_floor_eq_l777_777604

theorem exists_rational_for_floor_eq : 
  ∃ (c d : ℤ), d < 100 ∧ ∀ (k : ℕ), 1 ≤ k → k ≤ 99 → 
  (Int.floor (k * (c : ℚ) / d) = Int.floor (k * 73 / 100)) :=
by 
  sorry

end exists_rational_for_floor_eq_l777_777604


namespace no_such_convex_hexagon_and_point_l777_777341

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777341


namespace second_worker_time_l777_777761

noncomputable theory
open_locale classical

def rate_worker1 : ℝ := 1 / 5
def rate_worker2 (T : ℝ) : ℝ := 1 / T
def combined_rate (T : ℝ) : ℝ := rate_worker1 + rate_worker2 T
def combined_time : ℝ := 2.2222222222222223

theorem second_worker_time :
  ∃ T : ℝ, combined_rate T = 1 / combined_time ∧ T = 1.4285714285714286 :=
begin
  use 1.4285714285714286,
  split,
  {
    calc
    combined_rate 1.4285714285714286 = 1 / 5 + 1 / 1.4285714285714286 : by refl
    ... = 1 / 5 + 0.7 : by norm_num
    ... = 0.9 : by norm_num
    ... = 1 / 2.2222222222222223 : by norm_num,
  },
  {
    refl,
  }
end

end second_worker_time_l777_777761


namespace max_metro_speed_l777_777914

variable (R S v : ℝ)

theorem max_metro_speed (h1 : S > 0) (h2 : R > 0)
    (yegor : 12 * (S / 24) > S / v)
    (nikita : 6 * (S / 2) < (S + R) / v) :
    v ≤ 23 :=
  sorry

end max_metro_speed_l777_777914


namespace expectation_variance_inequality_l777_777582

noncomputable def Expectation (X : Type) [discrete_distribution X] : ℝ := sorry

noncomputable def Variance (X : Type) [discrete_distribution X] : ℝ := sorry

theorem expectation_variance_inequality (X : Type) [discrete_distribution X] (h : Expectation X ≠ -1) :
  Expectation (λ x => (X x + 1) ^ 2) > Variance X :=
sorry

end expectation_variance_inequality_l777_777582


namespace tetrahedron_edge_RS_l777_777192

noncomputable theory
open_locale classical

structure EdgeLengths := 
  (PQ QR RP QS PS RS : ℕ)

theorem tetrahedron_edge_RS
  (lengths : EdgeLengths)
  (H_edge_set : {lengths.PQ, lengths.QR, lengths.RP, lengths.QS, lengths.PS, lengths.RS} = {8, 14, 19, 28, 37, 42})
  (H_PQ_42 : lengths.PQ = 42) :
  lengths.RS = 14 :=
sorry

end tetrahedron_edge_RS_l777_777192


namespace no_such_hexagon_exists_l777_777334

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777334


namespace sqrt_floor_equality_l777_777208

theorem sqrt_floor_equality (n : ℤ) : 
  ⌊ (Real.sqrt (n.toReal) + Real.sqrt (n.toReal + 1)) ⌋ = ⌊ Real.sqrt (4 * n.toReal + 2) ⌋ := 
sorry

end sqrt_floor_equality_l777_777208


namespace perp_bisector_bisects_AD_l777_777996

theorem perp_bisector_bisects_AD
  (A B C M P Q D : Point)
  (hABC : Triangle A B C)
  (hM_midpoint_BC : midpoint M B C)
  (hCircle_MA : Circle M A)
  (hP_on_AB : lies_on P (line A B))
  (hQ_on_AC : lies_on Q (line A C))
  (hP_intersects_circle : on_circle P (Circle M A))
  (hQ_intersects_circle : on_circle Q (Circle M A))
  (hTangents_meet_D : tangent (Circle M A) P ∧ tangent (Circle M A) Q)
  (hTangents_meet_at : concurrent (tangent (Circle M A) P) (tangent (Circle M A) Q) D) :
  bisects (perpendicular_bisector_line B C) (segment A D) :=
sorry

end perp_bisector_bisects_AD_l777_777996


namespace find_ab_l777_777836

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l777_777836


namespace magnitude_of_angle_A_value_of_a_l777_777543

-- Given conditions in the problem
variables (A B C a b c : ℝ)
variables (m n : ℝ × ℝ)
hypothesis h1 : m = (-Real.cos B, Real.sin C)
hypothesis h2 : n = (-Real.cos C, -Real.sin B)
hypothesis h3 : m.1 * n.1 + m.2 * n.2 = 1 / 2
hypothesis h4 : A + B + C = Real.pi
hypothesis h5 : b + c = 4
hypothesis h6 : (1 / 2) * b * c * Real.sin A = Real.sqrt 3

-- First part: Prove the magnitude of angle A
theorem magnitude_of_angle_A : A = 2 * Real.pi / 3 := by
  sorry

-- Second part: Prove the value of a
theorem value_of_a : a = 2 * Real.sqrt 3 := by
  sorry

end magnitude_of_angle_A_value_of_a_l777_777543


namespace distinct_closed_paths_l777_777442

theorem distinct_closed_paths (n : ℕ) (h : n > 0) : 
  (n! * (n - 1)! / (2 * n) = n! * (n - 1)! / 2) := 
begin
  sorry
end

end distinct_closed_paths_l777_777442


namespace walter_chore_days_l777_777660

-- Definitions for the conditions
variables (b w : ℕ)  -- b: days regular, w: days exceptionally well

-- Conditions
def days_eq : Prop := b + w = 15
def earnings_eq : Prop := 3 * b + 4 * w = 47

-- The theorem stating the proof problem
theorem walter_chore_days (hb : days_eq b w) (he : earnings_eq b w) : w = 2 :=
by
  -- We only need to state the theorem; the proof is omitted.
  sorry

end walter_chore_days_l777_777660


namespace mark_charged_more_hours_l777_777599

theorem mark_charged_more_hours (P K M : ℕ) 
  (h_total : P + K + M = 144)
  (h_pat_kate : P = 2 * K)
  (h_pat_mark : P = M / 3) : M - K = 80 := 
by
  sorry

end mark_charged_more_hours_l777_777599


namespace no_such_hexagon_exists_l777_777351

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777351


namespace no_such_convex_hexagon_and_point_l777_777345

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777345


namespace kevin_feeds_each_toad_3_worms_l777_777097

theorem kevin_feeds_each_toad_3_worms
  (num_toads : ℕ) (minutes_per_worm : ℕ) (hours_to_minutes : ℕ) (total_minutes : ℕ)
  (H1 : num_toads = 8)
  (H2 : minutes_per_worm = 15)
  (H3 : hours_to_minutes = 60)
  (H4 : total_minutes = 6 * hours_to_minutes)
  :
  total_minutes / minutes_per_worm / num_toads = 3 :=
sorry

end kevin_feeds_each_toad_3_worms_l777_777097


namespace sqrt_expression_meaningful_range_l777_777891

theorem sqrt_expression_meaningful_range (x : ℝ) :
  (∃ (y : ℝ), y = sqrt (x - 3)) → x ≥ 3 :=
by
  sorry

end sqrt_expression_meaningful_range_l777_777891


namespace number_of_real_solutions_l777_777422

theorem number_of_real_solutions (c : ℝ) : 
  (∃ c1 c2 : ℝ, c1 ≠ c2 ∧ (|1/2 - c * I| = 3/4) ∧ (|1/2 - c2 * I| = 3/4)) → 
  2 :=
sorry

end number_of_real_solutions_l777_777422


namespace max_value_of_f_in_interval_l777_777004

def f (x φ : ℝ) : ℝ := sqrt 3 * sin (2 * x - φ) - cos (2 * x - φ)

theorem max_value_of_f_in_interval :
  ∀ (φ : ℝ), |φ| < π / 2 → (∀ x, f x φ = f (-x) φ) → (∀ x ∈ Icc (-π / 6) (π / 3), f x φ ≤ 1) := sorry

end max_value_of_f_in_interval_l777_777004


namespace cos2_minus_sin2_pi_over_12_l777_777275

theorem cos2_minus_sin2_pi_over_12 : 
  (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2 = Real.cos (Real.pi / 6) := 
by
  sorry

end cos2_minus_sin2_pi_over_12_l777_777275


namespace hexagon_probability_same_length_l777_777983

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777983


namespace circle_equation_tangent_line1_tangent_line2_l777_777845

-- Definitions of points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)

-- Equation for the circle given the point constraints
def circle_eq : Prop := 
  ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) ↔ ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0))

-- Equations for the tangent lines passing through point P and tangent to the circle
def tangent_eq1 : Prop := 
  P.1 = 2

def tangent_eq2 : Prop :=
  4 * P.1 - 3 * P.2 + 1 = 0

-- Statements to be proven
theorem circle_equation : circle_eq := 
  sorry 

theorem tangent_line1 : tangent_eq1 := 
  sorry 

theorem tangent_line2 : tangent_eq2 := 
  sorry 

end circle_equation_tangent_line1_tangent_line2_l777_777845


namespace minimum_value_l777_777572

noncomputable def minValue (x y : ℝ) : ℝ := (2 / x) + (3 / y)

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 20) : minValue x y = 1 :=
sorry

end minimum_value_l777_777572


namespace no_such_convex_hexagon_exists_l777_777379

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777379


namespace bees_on_second_day_l777_777589

theorem bees_on_second_day (bees_first_day : ℕ) (tripling_factor : ℕ) (h1 : bees_first_day = 144) (h2 : tripling_factor = 3) :
  let bees_second_day := bees_first_day * tripling_factor
  in bees_second_day = 432 := 
by
  intros
  have h3 : bees_second_day = 144 * 3 := by rw [h1, h2]
  rw h3
  norm_num
  exact rfl

end bees_on_second_day_l777_777589


namespace intersection_correct_l777_777262

open Set

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_correct : M ∩ N = {0, 1, 2} :=
by 
  -- Proof omitted
  sorry

end intersection_correct_l777_777262


namespace smallest_positive_addition_to_palindrome_l777_777747

def is_palindrome(n : Nat) : Prop :=
  let s := n.toString
  s = s.reverse

def next_palindrome (n : Nat) (m : Nat) : Prop :=
  is_palindrome (n + m) ∧ (∀ k : Nat, k < m → ¬ is_palindrome (n + k))

theorem smallest_positive_addition_to_palindrome :
  ∃ (m : Nat), next_palindrome 53209 m ∧ m = 126 :=
sorry

end smallest_positive_addition_to_palindrome_l777_777747


namespace no_such_hexagon_exists_l777_777363

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777363


namespace value_of_x_minus_y_l777_777035

theorem value_of_x_minus_y (x y : ℝ) 
    (h1 : 3015 * x + 3020 * y = 3025) 
    (h2 : 3018 * x + 3024 * y = 3030) :
    x - y = 11.1167 :=
sorry

end value_of_x_minus_y_l777_777035


namespace no_such_convex_hexagon_and_point_l777_777342

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777342


namespace increasing_intervals_l777_777002

def f (x : ℝ) : ℝ := Real.exp x * (x^2 + x + 1)

noncomputable def f' (x : ℝ) : ℝ := (Real.exp x) * (x^2 + 3 * x + 2)

theorem increasing_intervals :
  (∀ x, f' x > 0 → (x > -1 ∨ x < -2)) ∧ (∀ x, ((x > -1 ∨ x < -2) → f' x > 0)): sorry

end increasing_intervals_l777_777002


namespace min_value_of_diff_l777_777007

noncomputable def f (x : ℝ) : ℝ :=
  sin x * sin (x + π / 3) - 1 / 4

theorem min_value_of_diff {m n : ℝ} (h : m < n) (h_f_range : ∀ x ∈ set.Icc m n, f x ∈ set.Icc (-1 / 2) (1 / 4)) :
  n - m = 2 * π / 3 :=
sorry

end min_value_of_diff_l777_777007


namespace no_convex_hexagon_with_point_M_l777_777310

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777310


namespace no_such_hexagon_exists_l777_777302

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777302


namespace range_of_m_l777_777827

-- Define the discriminant of a quadratic equation
def discriminant(a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition p: The equation x^2 - 2x + m = 0 has two distinct real roots
def p (m : ℝ) : Prop := discriminant 1 (-2) m > 0

-- Proposition q: The function y = (m + 2)x - 1 is monotonically increasing
def q (m : ℝ) : Prop := m + 2 > 0

-- The main theorem stating the conditions and proving the range of m
theorem range_of_m (m : ℝ) (hpq : p m ∨ q m) (hpnq : ¬(p m ∧ q m)) : m ≤ -2 ∨ m ≥ 1 := sorry

end range_of_m_l777_777827


namespace minimum_throws_needed_l777_777925

noncomputable def throws_needed (p : ℝ) (q : ℝ) (lg_approx : ℝ) (target_prob : ℝ) : ℝ :=
    if (q ≠ 0 ∧ p ≠ 1 ∧ lg_approx ≠ 0) then
      let lg_p_neg := log (1 - p) / log q,
          lg_target := log target_prob / log q
      in ⌈(- lg_target / lg_p_neg).ceil.to_nat⌉₊
    else 0

theorem minimum_throws_needed :
  throws_needed 0.6 2 0.3 0.1 = 3 :=
by
  unfold throws_needed
  split_ifs
  sorry

end minimum_throws_needed_l777_777925


namespace joey_age_next_multiple_digit_sum_l777_777929

theorem joey_age_next_multiple_digit_sum:
  ∀ C J Z : ℕ,
    Z = 1 ->
    J = C + 2 ->
    -- "first of 8 birthdays" condition simplifies to J = 26 by the problem specifics
    (∀ n : ℕ, J = 26 + n * 25 → J + 25 = 26 + 25) ->
    (J + 25 = 51) ->
    digit_sum (J + 25) = 6 :=
by
  intros C J Z HZ HJ Hrepeats Hnext
  unfold digit_sum
  -- Specific calculations skipped: substitute and simplify directly
  sorry

# Function to calculate sum of digits (for completeness)
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

end joey_age_next_multiple_digit_sum_l777_777929


namespace triangle_area_ratio_l777_777117

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_ratio :
  let A := triangle_area 20 20 24 in
  let B := triangle_area 20 20 28 in
  A / B = 24 / 25 :=
by
  sorry

end triangle_area_ratio_l777_777117


namespace length_CF_l777_777985

/-!
Let \( C \) be a point not on line \( AF \) and \( D \) a point on line \( AF \) such that \( CD \perp AF. \)
Meanwhile, \( B \) is a point on line \( CF \) such that \( AB \perp CF. \)
Given that \( AB = 6 \), \( CD = 10 \), and \( AF = 7 \), prove that the length of \( CF \) is \( \frac{35}{3} \).
-/

theorem length_CF (C D B F A: Type)
  [not_on_line_C : ¬ (C ∈ (line_through A F))]
  [on_line_D : D ∈ (line_through A F)]
  (h1 : perpendicular (C - D) (A - F))
  (h2 : B ∈ (line_through C F))
  (h3 : perpendicular (A - B) (C - F))
  (hAB : length (A - B) = 6)
  (hCD : length (C - D) = 10)
  (hAF : length (A - F) = 7) :
  length (C - F) = 35 / 3 := 
by 
  sorry

end length_CF_l777_777985


namespace quadratic_function_range_l777_777641

def range_of_quadratic_function : Set ℝ :=
  {y : ℝ | y ≥ 2}

theorem quadratic_function_range :
  ∀ x : ℝ, (∃ y : ℝ, y = x^2 - 4*x + 6 ∧ y ∈ range_of_quadratic_function) :=
by
  sorry

end quadratic_function_range_l777_777641


namespace count_combinations_30_cents_l777_777877

theorem count_combinations_30_cents : 
  let number_of_combinations (target : ℕ) : ℕ := 
    (List.range' 0 (target / 5 + 1)).sum (λ n, if (target - n * 5) % 25 == 0 then 1 else 0) 
    + (if target % 25 == 0 then 1 else 0) 
    + (if target % 5 == 0 then 1 else 0) 
    + 1 := 
  number_of_combinations 30 = 11 := 
by sorry

end count_combinations_30_cents_l777_777877


namespace mandy_reads_books_of_480_pages_l777_777130

def pages_at_age6 : ℕ := 8

def pages_at_age12 (p6 : ℕ) : ℕ := 5 * p6

def pages_at_age20 (p12 : ℕ) : ℕ := 3 * p12

def pages_presently (p20 : ℕ) : ℕ := 4 * p20

theorem mandy_reads_books_of_480_pages :
  let p6 := pages_at_age6,
  let p12 := pages_at_age12 p6,
  let p20 := pages_at_age20 p12,
  let ppresent := pages_presently p20
  in ppresent = 480 :=
by
  sorry

end mandy_reads_books_of_480_pages_l777_777130


namespace regular_hexagon_same_length_probability_l777_777974

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777974


namespace proof_x_minus_y_squared_l777_777500

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l777_777500


namespace three_digit_combinations_count_l777_777486

theorem three_digit_combinations_count : 
  let digits := [1, 2, 3, 4],
      num_digits := 3,
      valid_numbers := {
        x : Fin 1000 // x represents 3 digits, range 0-999
          | let d1 := x / 100, d2 := (x % 100) / 10, d3 := x % 10 in
            d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧
            (d1 = d2 -> d1 ≠ d3) ∧ (d1 = d3 -> d1 ≠ d2) ∧ (d2 = d3 -> d2 ≠ d1)
      } in
  valid_numbers.card = 60 :=
by
  sorry

end three_digit_combinations_count_l777_777486


namespace yellow_surface_area_min_fraction_l777_777744

/-- 
  Given a larger cube with 4-inch edges, constructed from 64 smaller cubes (each with 1-inch edge),
  where 50 cubes are colored blue, and 14 cubes are colored yellow. 
  If the large cube is crafted to display the minimum possible yellow surface area externally,
  then the fraction of the surface area of the large cube that is yellow is 7/48.
-/
theorem yellow_surface_area_min_fraction (n_smaller_cubes blue_cubes yellow_cubes : ℕ) 
  (edge_small edge_large : ℕ) (surface_area_larger_cube yellow_surface_min : ℕ) :
  edge_small = 1 → edge_large = 4 → n_smaller_cubes = 64 → 
  blue_cubes = 50 → yellow_cubes = 14 →
  surface_area_larger_cube = 96 → yellow_surface_min = 14 → 
  (yellow_surface_min : ℚ) / (surface_area_larger_cube : ℚ) = 7 / 48 := 
by 
  intros h_edge_small h_edge_large h_n h_blue h_yellow h_surface_area h_yellow_surface
  sorry

end yellow_surface_area_min_fraction_l777_777744


namespace find_ab_l777_777835

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end find_ab_l777_777835


namespace laser_path_total_distance_l777_777745

noncomputable def laser_path_distance : ℝ :=
  let A : ℝ × ℝ := (2, 4)
  let D : ℝ × ℝ := (10, 4)
  let B : ℝ × ℝ := (0, 4)
  let C : ℝ × ℝ := (10, 0)
  let distance := λ (p1 p2 : ℝ × ℝ), Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  distance A B + distance B C + distance C D

theorem laser_path_total_distance : laser_path_distance = 6 + 2 * Real.sqrt 29 := by
  sorry

end laser_path_total_distance_l777_777745


namespace no_such_hexagon_and_point_l777_777391

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l777_777391


namespace find_side_c_l777_777896

noncomputable def cos_120 : ℝ := real.cos (2 * real.pi / 3)

theorem find_side_c (a b C : ℝ) (h_a : a = 3) (h_b : b = 5) (h_C : C = 2 * real.pi / 3) :
  (a^2 + b^2 - 2 * a * b * real.cos C = 49) ∧ (real.sqrt (a^2 + b^2 - 2 * a * b * real.cos C) = 7) :=
by
  rw [h_a, h_b, h_C]
  have h_cos120 : real.cos (2 * real.pi / 3) = -1/2 := real.cos_two_pi_div_three
  rw [h_cos120]
  norm_num
  split
  { norm_num }
  { norm_num }

end find_side_c_l777_777896


namespace no_convex_hexagon_with_point_M_l777_777312

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777312


namespace y_intercept_3x_minus_4y_eq_12_l777_777671

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l777_777671


namespace eqn_locus_C_pq_fixed_point_l777_777581

variables (A B M : Point)
variables (T : Point)
variables (C : Set Point)

-- Defining points A and B
def A := ({x := -2, y := 0} : Point)
def B := ({x := 2, y := 0} : Point)

-- Defining the line slopes k_AM and k_BM
def k_AM (M : Point) : Real := M.y / (M.x + 2)
def k_BM (M : Point) : Real := M.y / (M.x - 2)

-- Locus condition for point M
def locus (M : Point) : Prop := k_AM M * k_BM M = 1 / 4

-- Set of points forming the locus C
def C := {M : Point | locus M}

-- The equation of locus C
theorem eqn_locus_C (x y : Real) : (M : Point) ∈ C ↔ (x^2 / 4 - y^2 = 1 ∧ x ≠ 2 ∧ x ≠ -2) := sorry

-- Defining a moving point T on the line x = 1
def moving_T (t : Real) : Point := ({x := 1, y := t} : Point)

-- Proving the line PQ passes through a fixed point
theorem pq_fixed_point (T : Point) (P Q : Point) (C : Set Point) :
    P ≠ Q → P ∈ C → Q ∈ C → lies_on_line P (line AT) → lies_on_line Q (line BT) → line PQ intersects (point (4, 0)) := sorry

end eqn_locus_C_pq_fixed_point_l777_777581


namespace find_double_pieces_l777_777587

theorem find_double_pieces (x : ℕ) 
  (h1 : 100 + 2 * x + 150 + 660 = 1000) : x = 45 :=
by sorry

end find_double_pieces_l777_777587


namespace problem_statement_l777_777558

def g (a b c : ℝ) : ℝ :=
  if a + b + c ≤ 5 then (a * b - a + c) / (2 * a)
  else (a * b - b - c) / (-2 * b)

theorem problem_statement : g 1 2 1 + g 3 2 1 = 1 / 4 := by
  sorry

end problem_statement_l777_777558


namespace sum_inverse_distance_product_l777_777432

noncomputable def point_on_unit_circle (i : ℕ) (n : ℕ) [fact (i < n)] : ℂ :=
  by sorry  -- Definition placeholder for n distinct points on the unit circle

noncomputable def distance_product (i : ℕ) (n : ℕ) : ℝ :=
  let points := λ j, point_on_unit_circle j n in
  (finset.range n).product (λ j, if j ≠ i then abs (points j - points i) else 1)

theorem sum_inverse_distance_product (n : ℕ) (hn : 0 < n) : 
  ∑ i in finset.range n, 1 / distance_product i n >= 1 :=
by sorry  -- Proof placeholder for the theorem


end sum_inverse_distance_product_l777_777432


namespace allocation_of_branches_and_toys_l777_777226

-- Define the number of branches and toys as natural numbers
variables (b t : ℕ)

-- Define the conditions
def condition_1 := t = b + 1
def condition_2 := 2 * b = t - 1

-- The main theorem to prove
theorem allocation_of_branches_and_toys (hb : b = 3) (ht : t = 4) :
  condition_1 b t ∧ condition_2 b t :=
begin
  -- This is the correct number of branches and toys
  -- Proof is omitted
  sorry
end

end allocation_of_branches_and_toys_l777_777226


namespace elizabeth_net_profit_l777_777399

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end elizabeth_net_profit_l777_777399


namespace num_ways_to_choose_officers_l777_777598

-- Definition of the total number of members
def total_members : ℕ := 30
-- Definition of boys' count
def boys : ℕ := 15
-- Definition of girls' count
def girls : ℕ := 15

-- Conditions for the selection process 
axiom same_gender_president_vicepresident (p v : ℕ) : p < total_members ∧ v < total_members ∧ (p < boys ∧ v < boys ∨ p ≥ boys ∧ v ≥ boys)

axiom different_gender_president_treasurer (p t : ℕ) : (p < boys ∧ t ≥ boys ∨ p >= boys ∧ t < boys)

-- The theorem stating the number of ways to choose president, vice-president, and treasurer
theorem num_ways_to_choose_officers : 
  (Σ p : ℕ, Σ v : ℕ, Σ t : ℕ, same_gender_president_vicepresident p v ∧ different_gender_president_treasurer p t) = 6300 := 
by 
  -- Proof is omitted using sorry
  sorry

end num_ways_to_choose_officers_l777_777598


namespace no_such_convex_hexagon_and_point_l777_777340

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777340


namespace find_a_find_h_min_find_m_range_l777_777864

noncomputable def f (a x : ℝ) : ℝ := a^x
noncomputable def g (a x m : ℝ) : ℝ := a^(2*x) + m
noncomputable def h (a x m : ℝ) : ℝ := g(a, x, m) - 2 * m * f(a, x)

-- Given conditions
axiom m_pos : ∀ m : ℝ, m > 0 → True
axiom a_pos : ∀ a : ℝ, a > 0 → True
axiom a_neq_one : ∀ a : ℝ, a ≠ 1 → True

-- Problem I
theorem find_a (a : ℝ) (h : a + a⁻¹ = 5/2) : a = 2 ∨ a = 1/2 := by
  sorry

-- Problem II
theorem find_h_min (a m : ℝ) (h₁ : a > 1) (h₂ : ∀ x, 0 ≤ x ∧ x ≤ 1 → True) : 
  ∃ H, (H = if m < 1 then 1 - m else if 1 ≤ m ∧ m ≤ 2 then m - m^2 else 4 - 3*m) := by
  sorry

-- Problem III
theorem find_m_range (a m : ℝ) (h₁ : a > 1) (h₂ : ∀ x, 0 ≤ x ∧ x ≤ 1 → |(f(a, x) - m * g(a, x, m)) / f(a, x)| ≤ 1) : 
  m ∈ Set.Ico 0 2 := by
  sorry

end find_a_find_h_min_find_m_range_l777_777864


namespace max_value_n_l777_777108

def set_of_two_elements (s : Set ℕ) := ∃ (a b : ℕ), a ≠ b ∧ s = {a, b}

noncomputable def max_n := 4033

def disjoint_sum_sets (sets : List (Set ℕ)) : Prop :=
  ∀ i j k l,
    i < sets.length → j < sets.length → k < sets.length → l < sets.length →
    i < j → k < l → 
    let S1 := {a + b | a ∈ sets[i], b ∈ sets[j]} in
    let S2 := {a + b | a ∈ sets[k], b ∈ sets[l]} in
    S1 ≠ S2

theorem max_value_n (A : List (Set ℕ)) (h₁ : ∀ s ∈ A, set_of_two_elements s) 
    (h₂ : disjoint_sum_sets A) : 
      A.length ≤ max_n := by 
  sorry

end max_value_n_l777_777108


namespace circle_properties_l777_777109

theorem circle_properties :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 4 * y - 25 = -y^2 + 10 * x + 49 → (x - 5)^2 + (y - 2)^2 = s^2) ∧
  c = 5 ∧ d = 2 ∧ s = Real.sqrt 103 ∧ c + d + s = 7 + Real.sqrt 103 :=
by
  sorry

end circle_properties_l777_777109


namespace ratio_BD_BO_l777_777062

-- Definitions of the conditions
def circle (O : Point) (radius : Real) : Set Point := {P | dist P O = radius}

def is_tangent_to (P Q : Point) (O : Point) (C : Set Point) : Prop :=
  dist Q O > dist P O ∧ ∃ (T : Point), T ∈ C ∧ Q = T

def is_isosceles (A B C : Point) : Prop :=
  dist A B = dist B C

def angle (A B C : Point) : Real := sorry

def segment_intersect (A B O : Point) (C : Set Point) : Point := sorry

variables {O A C B D : Point}
variable {circle_center : O}
variable {radius : Real}
variable {circle_def : circle O radius}
variable {is_tangent_BA : is_tangent_to B A O (circle O radius)}
variable {is_tangent_BC : is_tangent_to B C O (circle O radius)}
variable {isosceles_triangle : is_isosceles A B C}
variable {angle_ABC_100 : angle A B C = 100}
variable {D_def : D = segment_intersect B O O (circle O radius)}

theorem ratio_BD_BO : (dist B D) / (dist B O) = 1 / 2 := sorry

end ratio_BD_BO_l777_777062


namespace prime_sum_eq_11_over_13_l777_777870

theorem prime_sum_eq_11_over_13 (p q : ℕ) (hp : p.Prime) (hq : q.Prime) (cond : p * q + p^2 + q^2 = 199) (h_lt : p < q)
  (h_p : p = 2) (h_q : q = 13) :
  (∑ k in Finset.range (q - 1), 2 / (k + p) * (k + p + 1)) = 11 / 13 := by
  sorry

end prime_sum_eq_11_over_13_l777_777870


namespace domain_of_f_range_of_f_strictly_decreasing_f_solve_inequality_l777_777859

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 + real.sqrt (1 - x^2)) - real.sqrt (1 + x)

theorem domain_of_f : ∀ x, f x = real.sqrt (1 + real.sqrt (1 - x^2)) - real.sqrt (1 + x) → -1 ≤ x ∧ x ≤ 1 := 
sorry

theorem range_of_f : ∀ y, ∃ x, f x = y ∧ 1 - real.sqrt (2) ≤ y ∧ y ≤ 1 := 
sorry 

theorem strictly_decreasing_f : ∀ (x1 x2 : ℝ), -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ x1 < x2 → f x1 > f x2 := 
sorry

theorem solve_inequality : ∀ (x : ℝ), f x > 1/2 → -1 ≤ x ∧ x < (3 * real.sqrt 2 - 1 - real.sqrt (13 + 6 * real.sqrt 2)) / 8 := 
sorry

end domain_of_f_range_of_f_strictly_decreasing_f_solve_inequality_l777_777859


namespace sum_of_numbers_l777_777682

theorem sum_of_numbers : (4.75 + 0.303 + 0.432) = 5.485 := 
by  
  sorry

end sum_of_numbers_l777_777682


namespace sweets_neither_red_nor_green_l777_777197

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end sweets_neither_red_nor_green_l777_777197


namespace x_coordinate_undetermined_l777_777918

theorem x_coordinate_undetermined
    (h1 : ∃ (x1 : ℝ), 4 = x1 ∧ 0 = 3)
    (h2 : ∃ (x1 : ℝ), -12 = x1) :
    ∀ x1 : ℝ, ∃ y2 : ℝ, y2 ∉ {-16 * (3 / (x1 - 4))} :=
by
  sorry

end x_coordinate_undetermined_l777_777918


namespace triangle_inequality_transformed_l777_777576

variables {a b c : ℝ}

noncomputable def a' := a / (a + 1)
noncomputable def b' := b / (b + 1)
noncomputable def c' := c / (c + 1)

theorem triangle_inequality_transformed 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (habc : a + b > c) (hbac : a + c > b) (hcba : b + c > a) :
  a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a' := 
  sorry

end triangle_inequality_transformed_l777_777576


namespace tan_alpha_equiv_l777_777810

-- Define the theorem statement
theorem tan_alpha_equiv :
  ∀ (α : ℝ), tan α = -3 → (2 * sin α - cos α) / (2 * sin α + cos α) = 7 / 5 :=
by
  intros α h_tan
  -- Placeholder for the proof
  sorry

end tan_alpha_equiv_l777_777810


namespace quadratic_function_min_value_l777_777821

theorem quadratic_function_min_value (a b c : ℝ) (h_a : a > 0) (h_b : b ≠ 0) 
(h_f0 : |c| = 1) (h_f1 : |a + b + c| = 1) (h_fn1 : |a - b + c| = 1) :
∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a*x^2 + b*x + c) ∧
  (|f 0| = 1) ∧ (|f 1| = 1) ∧ (|f (-1)| = 1) ∧
  (f 0 = -(5/4) ∨ f 1 = -(5/4) ∨ f (-1) = -(5/4)) :=
by
  sorry

end quadratic_function_min_value_l777_777821


namespace tank_empty_time_l777_777218

theorem tank_empty_time (V : ℝ) (r_inlet r_outlet1 r_outlet2 : ℝ) (I : V = 20 * 12^3)
  (r_inlet_val : r_inlet = 5) (r_outlet1_val : r_outlet1 = 9) 
  (r_outlet2_val : r_outlet2 = 8) : 
  (V / ((r_outlet1 + r_outlet2) - r_inlet) = 2880) :=
by
  sorry

end tank_empty_time_l777_777218


namespace no_such_hexagon_exists_l777_777364

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777364


namespace number_partitions_correct_l777_777414

noncomputable def number_partitions (n : ℕ) : ℕ :=
2^(n-1)

theorem number_partitions_correct (n : ℕ) :
  ∑ (A1 A2 A3 : Finset ℕ) in (Finset.powerset (Finset.range (n + 1))),
    (∀ (x : ℕ), (x ∈ A1 → x ∉ A2 ∧ x ∉ A3) ∧ (x ∈ A2 → x ∉ A1 ∧ x ∉ A3) ∧ (x ∈ A3 → x ∉ A1 ∧ x ∉ A2)) →
    (∀ (set : Finset ℕ), (set = A1 ∨ set = A2 ∨ set = A3) →
    Finset.Pack_interleaved set) →
    (A1.nonempty ∧ A2.nonempty ∧ A3.nonempty → 
    ((Finset.min' A1 A1.nonempty ∈ (2 * ℕ)) ∨ 
    (Finset.min' A2 A2.nonempty ∈ (2 * ℕ)) ∨ 
    (Finset.min' A3 A3.nonempty ∈ (2 * ℕ))) ∧
    ((Finset.min' A1 A1.nonempty ∈ (2 * ℕ) ∧ Finset.min' A2 A2.nonempty ∉ (2 * ℕ) ∧ Finset.min' A3 A3.nonempty ∉ (2 * ℕ)) ∨
    (Finset.min' A1 A1.nonempty ∉ (2 * ℕ) ∧ Finset.min' A2 A2.nonempty ∈ (2 * ℕ) ∧ Finset.min' A3 A3.nonempty ∉ (2 * ℕ)) ∨
    (Finset.min' A1 A1.nonempty ∉ (2 * ℕ) ∧ Finset.min' A2 A2.nonempty ∉ (2 * ℕ) ∧ Finset.min' A3 A3.nonempty ∈ (2 * ℕ)))) →
    number_partitions n = 2^(n-1) :=
by
  sorry

end number_partitions_correct_l777_777414


namespace determine_n_l777_777037

theorem determine_n (n : ℕ) (h : 9^4 = 3^n) : n = 8 :=
by {
  sorry
}

end determine_n_l777_777037


namespace sin_tan_condition_l777_777043

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end sin_tan_condition_l777_777043


namespace y_intercept_of_line_l777_777667

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l777_777667


namespace find_m_values_l777_777828

open set

theorem find_m_values :
  let A := {x : ℝ | x^2 - x - 6 = 0},
      B := λ m : ℝ, {x : ℝ | x * m = 1} in
  {m | ∀ x, x ∈ A ∩ B m ↔ x ∈ B m} = {0, -1/2, 1/3} :=
by sorry

end find_m_values_l777_777828


namespace remaining_files_calc_l777_777024

-- Definitions based on given conditions
def music_files : ℕ := 27
def video_files : ℕ := 42
def deleted_files : ℕ := 11

-- Theorem statement to prove the number of remaining files
theorem remaining_files_calc : music_files + video_files - deleted_files = 58 := by
  sorry

end remaining_files_calc_l777_777024


namespace cherry_pie_degrees_l777_777899

theorem cherry_pie_degrees (total_students : ℕ) (choco_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ)
  (other_pies : ℕ) (cherry_preference : ℚ) :
  total_students = 48 →
  choco_pie = 18 →
  apple_pie = 12 →
  blueberry_pie = 8 →
  other_pies = 3 →
  cherry_preference = 10 / 3 →
  (cherry_preference / total_students * 360).round = 25 := by
  intros
  sorry

end cherry_pie_degrees_l777_777899


namespace ingested_fat_correct_l777_777742

def initial_fat_herring := 40
def initial_fat_eel := 20
def initial_fat_pike := initial_fat_eel + 10
def initial_fat_salmon := 35
def initial_fat_halibut := 50

def cooking_loss_rate := 0.10
def indigestible_rate := 0.08

def remaining_fat (initial_fat : ℝ) : ℝ := initial_fat * (1 - cooking_loss_rate)
def digestible_fat (remaining_fat : ℝ) : ℝ := remaining_fat * (1 - indigestible_rate)

def count_herrings := 40
def count_eels := 30
def count_pikes := 25
def count_salmons := 20
def count_halibuts := 15

def total_ingested_fat : ℝ :=
  (digestible_fat (remaining_fat initial_fat_herring) * count_herrings) +
  (digestible_fat (remaining_fat initial_fat_eel) * count_eels) +
  (digestible_fat (remaining_fat initial_fat_pike) * count_pikes) +
  (digestible_fat (remaining_fat initial_fat_salmon) * count_salmons) +
  (digestible_fat (remaining_fat initial_fat_halibut) * count_halibuts)

theorem ingested_fat_correct : total_ingested_fat = 3643.2 :=
by
  sorry

end ingested_fat_correct_l777_777742


namespace normalize_differential_eq_l777_777155

noncomputable def normalize_system (x : ℝ → ℝ) (y : ℝ → ℝ) :
  ((ℝ → ℝ) × (ℝ → ℝ) × (ℝ → ℝ)) :=
  let x1 := x;
  let x2 := fun t => (deriv (λ (u : ℝ), x u)) t;
  let x3 := y;
  (x1, x2, x3)

theorem normalize_differential_eq (x y : ℝ → ℝ) :
  (∀ t, deriv (λ t, deriv (λ (u : ℝ), x u) t) t = y t) ∧
  (∀ t, t^3 * (deriv y t) = 2 * x t) →
  let (x1, x2, x3) := normalize_system x y in
  (∀ t, deriv x1 t = x2 t) ∧
  (∀ t, deriv x2 t = x3 t) ∧
  (∀ t, deriv x3 t = 2 * x1 t / t^3) :=
by 
  intros h;
  sorry

end normalize_differential_eq_l777_777155


namespace hexagon_probability_same_length_l777_777980

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777980


namespace circle_radius_l777_777639

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2*y = 0 → ∃ r : ℝ, r = 1 :=
by
  sorry

end circle_radius_l777_777639


namespace find_value_of_a_l777_777219

theorem find_value_of_a (a : ℝ) 
  (h : (2 * a + 16 + 3 * a - 8) / 2 = 69) : a = 26 := 
by
  sorry

end find_value_of_a_l777_777219


namespace upward_walk_to_Felix_l777_777525

structure Point where
  x : ℝ
  y : ℝ

def DennisLocation : Point := ⟨8, -15⟩
def EmilyLocation : Point := ⟨3, 20⟩
def FelixLocation : Point := ⟨5, 5⟩

def midpoint (p1 p2 : Point) : Point := 
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def meetingPoint : Point := midpoint DennisLocation EmilyLocation

theorem upward_walk_to_Felix : (FelixLocation.y - meetingPoint.y) = 2.5 :=
by
  sorry

end upward_walk_to_Felix_l777_777525


namespace sum_distances_to_vertices_ge_4sqrt3_l777_777817

/-- Given a cube with edge length 1, prove that for any point M inside the cube,
    the sum of the distances from M to the vertices of the cube is at least 4√3. -/
theorem sum_distances_to_vertices_ge_4sqrt3 (M : EuclideanSpace ℝ 3) 
    (A B C D A1 B1 C1 D1 : EuclideanSpace ℝ 3)
    (hA : dist A B = 1) (hB : dist B C = 1) (hC : dist C D = 1)
    (hAA1 : dist A A1 = 1) (hBB1 : dist B B1 = 1) (hCC1 : dist C C1 = 1) (hDD1 : dist D D1 = 1)
    (hAC1 : dist A C1 = sqrt 3) (hBD1 : dist B D1 = sqrt 3) (hCA1 : dist C A1 = sqrt 3)
    (hDB1 : dist D B1 = sqrt 3) :
  dist M A + dist M B + dist M C + dist M D + dist M A1 + dist M B1 + dist M C1 + dist M D1 ≥ 4 * sqrt 3 := 
sorry

end sum_distances_to_vertices_ge_4sqrt3_l777_777817


namespace no_such_hexagon_exists_l777_777354

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777354


namespace determinant_of_matrix_A_l777_777403

variable (y : ℝ)

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [y + 2, 2y, 2y],
    [2y, y + 2, 2y],
    [2y, 2y, y + 2]
  ]

theorem determinant_of_matrix_A : Matrix.det (matrix_A y) = 5 * y^3 - 10 * y^2 + 12 * y + 8 := by
  sorry

end determinant_of_matrix_A_l777_777403


namespace neg_one_to_zero_l777_777268

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l777_777268


namespace part_a_part_b_l777_777421

noncomputable def a_n (n : ℕ) : ℕ := 
  (n * (n + 1) / 2) % 10

theorem part_a (h : a_n 21 = 1) :
  a_n 21 = 1 :=
by
  exact h

theorem part_b (h : ∑ i in range 2016, a_n i = 8055) :
  ∑ i in range 2016, a_n i = 8055 :=
by
  exact h

end part_a_part_b_l777_777421


namespace polar_equation_C2_distance_AB_l777_777797

open Real

-- Define curves C1 and C2
def C1 (t : ℝ) : ℝ × ℝ := (1 + 2 * t, 2 - 2 * t)
def C2 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) : ℝ × ℝ := (2 * cos θ + 2, 2 * sin θ)

-- Part (I): Polar equation of the curve C2
theorem polar_equation_C2 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) : ∀ θ, (C2 θ hθ).fst ^ 2 + (C2 θ hθ).snd ^ 2 = 4 * cos θ ^ 2 := by
  sorry

-- Part (II): Distance |AB| where C1 intersects C2
theorem distance_AB : ∃ A B : ℝ × ℝ, A ≠ B ∧ (∃ t₁ t₂ : ℝ, C1 t₁ = A ∧ C1 t₂ = B ∧ (C2 θ (by norm_num : 0 ≤ θ ∧ θ ≤ 2 * π)) = A ∧ (C2 θ (by norm_num : 0 ≤ θ ∧ θ ≤ 2 * π)) = B) ∧ dist A B = sqrt 14 := by
  sorry

end polar_equation_C2_distance_AB_l777_777797


namespace find_pairs_l777_777409

theorem find_pairs (x y : ℝ) (h1 : |x| + |y| = 1340) (h2 : x^3 + y^3 + 2010 * x * y = 670^3) :
  x + y = 670 ∧ x * y = -673350 :=
sorry

end find_pairs_l777_777409


namespace allocation_of_branches_and_toys_l777_777227

-- Define the number of branches and toys as natural numbers
variables (b t : ℕ)

-- Define the conditions
def condition_1 := t = b + 1
def condition_2 := 2 * b = t - 1

-- The main theorem to prove
theorem allocation_of_branches_and_toys (hb : b = 3) (ht : t = 4) :
  condition_1 b t ∧ condition_2 b t :=
begin
  -- This is the correct number of branches and toys
  -- Proof is omitted
  sorry
end

end allocation_of_branches_and_toys_l777_777227


namespace no_such_hexagon_exists_l777_777323

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777323


namespace unique_quotient_is_9742_l777_777517

theorem unique_quotient_is_9742 :
  ∃ (d4 d3 d2 d1 : ℕ),
    (d2 = d1 + 2) ∧
    (d4 = d3 + 2) ∧
    (0 ≤ d1 ∧ d1 ≤ 9) ∧
    (0 ≤ d2 ∧ d2 ≤ 9) ∧
    (0 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d4 * 1000 + d3 * 100 + d2 * 10 + d1 = 9742) :=
by sorry

end unique_quotient_is_9742_l777_777517


namespace centers_of_incircles_form_rectangle_l777_777450

theorem centers_of_incircles_form_rectangle
  (A B C D : Point)
  (h : InscribedQuadrilateral A B C D)
  (A' B' C' D' : Point)
  (hA' : IsIncenter A B C A')
  (hB' : IsIncenter A B D B')
  (hC' : IsIncenter A C D C')
  (hD' : IsIncenter B C D D') :
  IsRectangle A' B' C' D' :=
sorry

end centers_of_incircles_form_rectangle_l777_777450


namespace count_possible_m_l777_777093

theorem count_possible_m :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
    ∀ (P Q R : ℝ × ℝ), -- coordinates of points forming the triangle
    let M₁ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
    let M₂ := ((P.1 + R.1) / 2, (P.2 + R.2) / 2) in
      -- medians' slopes
      let slope1 := (M₁.2 - P.2) / (M₁.1 - P.1) in
      let slope2 := (M₂.2 - P.2) / (M₂.1 - P.1) in
      ((slope1 = 5 ∧ slope2 = 4 * 5) ∧ m₁ = 20) ∨
      ((slope1 = 5 ∧ (P.2 - M₂.2) / (P.1 - M₂.1) = (5 / 4)) ∧ m₂ = (5/4)) :=
sorry

end count_possible_m_l777_777093


namespace interest_rate_of_second_part_l777_777756

theorem interest_rate_of_second_part 
  (total_sum : ℝ) (P2 : ℝ) (interest1_rate : ℝ) 
  (time1 : ℝ) (time2 : ℝ) (interest2_value : ℝ) : 
  (total_sum = 2704) → 
  (P2 = 1664) → 
  (interest1_rate = 0.03) → 
  (time1 = 8) → 
  (interest2_value = interest1_rate * (total_sum - P2) * time1) → 
  (time2 = 3) → 
  1664 * r * time2 = interest2_value → 
  r = 0.05 := 
by sorry

end interest_rate_of_second_part_l777_777756


namespace partial_fraction_decomposition_l777_777121

noncomputable def polynomial := λ x: ℝ => x^3 - 24 * x^2 + 88 * x - 75

theorem partial_fraction_decomposition
  (p q r A B C : ℝ)
  (hpq : p ≠ q)
  (hpr : p ≠ r)
  (hqr : q ≠ r)
  (hroots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (hdecomposition: ∀ s: ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
                      1 / polynomial s = A / (s - p) + B / (s - q) + C / (s - r)) :
  (1 / A + 1 / B + 1 / C = 256) := sorry

end partial_fraction_decomposition_l777_777121


namespace min_moves_to_equalize_coins_l777_777716

def num_coins : List ℕ := [8, 14, 18, 6, 10, 20, 15]

theorem min_moves_to_equalize_coins :
  (∃ k, k = 22) ∧ ∀ (boxes : List ℕ) (h: boxes = num_coins),
    (∑ b in boxes, b) % boxes.length = 0 ∧ 
    -- condition additional that we move coins to minimum moves
    sorry := 
begin
  sorry,
end

end min_moves_to_equalize_coins_l777_777716


namespace sum_of_roots_l777_777696

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l777_777696


namespace ab_is_zero_l777_777837

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end ab_is_zero_l777_777837


namespace no_such_convex_hexagon_exists_l777_777384

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777384


namespace widgets_after_15_days_l777_777550

theorem widgets_after_15_days 
  : let a1 := 2, d := 3, n := 15, an := 3 * n - 1 in
    let Sn := (n / 2) * (a1 + an) in
    Sn = 345 :=
by
  let a1 := 2
  let d := 3
  let n := 15
  let an := 3 * n - 1
  let Sn := (n / 2) * (a1 + an)
  sorry

end widgets_after_15_days_l777_777550


namespace units_digit_of_5_pow_35_minus_6_pow_21_l777_777704

-- Definition for extracting the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Proposition stating the conditions and question
theorem units_digit_of_5_pow_35_minus_6_pow_21 :
  units_digit (5^35 - 6^21) = 9 :=
by
  -- We rely on the given conditions without proving them here
  have h1 : units_digit (5^35) = 5 := sorry
  have h2 : units_digit (6^21) = 6 := sorry
  -- The main proof logic
  have h3 : units_digit (5^35 - 6^21) = units_digit (5 - 6) := sorry
  have h4 : units_digit (5 - 6) = units_digit (-1) := sorry
  have h5 : units_digit (-1) = 9 := sorry
  exact h5

end units_digit_of_5_pow_35_minus_6_pow_21_l777_777704


namespace license_plate_count_l777_777883

theorem license_plate_count :
  let possibilities := 36
  let pairs := 3
  (pairs * possibilities * possibilities * (possibilities - 1)) = 136080 :=
by
  let possibilities := 36
  let pairs := 3
  -- license plate construction
  calc 3 * 36 * 36 * 35 = 136080 : sorry

end license_plate_count_l777_777883


namespace centrally_symmetric_equidecomposable_l777_777231

-- Assume we have a type for Polyhedra
variable (Polyhedron : Type)

-- Conditions
variable (sameVolume : Polyhedron → Polyhedron → Prop)
variable (centrallySymmetricFaces : Polyhedron → Prop)
variable (equidecomposable : Polyhedron → Polyhedron → Prop)

-- Theorem statement
theorem centrally_symmetric_equidecomposable 
  (P Q : Polyhedron) 
  (h1 : sameVolume P Q) 
  (h2 : centrallySymmetricFaces P) 
  (h3 : centrallySymmetricFaces Q) :
  equidecomposable P Q := 
sorry

end centrally_symmetric_equidecomposable_l777_777231


namespace possible_integer_lengths_for_third_side_l777_777049

theorem possible_integer_lengths_for_third_side (x : ℕ) : (8 < x ∧ x < 19) ↔ (4 ≤ x ∧ x ≤ 18) :=
sorry

end possible_integer_lengths_for_third_side_l777_777049


namespace circle_intersection_unique_k_l777_777537

theorem circle_intersection_unique_k :
  ∃! k : ℝ, k ≥ 0 ∧ ∀ z : ℂ, |z| = k ↔ (∃ x y : ℝ, z = x + y * I ∧ (x + 3.75)^2 + y^2 = 5.0625) :=
sorry

end circle_intersection_unique_k_l777_777537


namespace y_intercept_3x_minus_4y_eq_12_l777_777672

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end y_intercept_3x_minus_4y_eq_12_l777_777672


namespace decompose_trig_expression_l777_777288

theorem decompose_trig_expression (x : ℝ) :
  1 - (sin x)^5 - (cos x)^5 = (1 - sin x) * (1 - cos x) * (3 + 2 * (sin x + cos x) + 2 * (sin x * cos x) + (sin x * cos x) * (sin x + cos x)) :=
by
  sorry

end decompose_trig_expression_l777_777288


namespace smallest_positive_period_sin_cos_squared_l777_777294

theorem smallest_positive_period_sin_cos_squared : 
  ∃ T > 0, (∀ x, y = (sin x + cos x)^2 → y = y ∧ y = (sin (x + T) + cos (x + T))^2) ∧ T = π :=
by 
  sorry

end smallest_positive_period_sin_cos_squared_l777_777294


namespace sum_gcd_lcm_l777_777107

theorem sum_gcd_lcm (A B : ℕ) (hA : A = Nat.gcd 10 (Nat.gcd 15 25)) (hB : B = Nat.lcm 10 (Nat.lcm 15 25)) :
  A + B = 155 :=
by
  sorry

end sum_gcd_lcm_l777_777107


namespace cannot_all_lie_inside_circumcircle_l777_777105

variables {ABC : Type} [triangle ABC]

structure Point (ABC : Type) [triangle ABC] where
  P A1 B1 C1 A2 B2 C2 : ABC

-- Let P be a point inside triangle ABC

theorem cannot_all_lie_inside_circumcircle (ABC : Type) [triangle ABC]
  (P : Point ABC) (A1 B1 C1 A2 B2 C2 : ABC)
  (AP_meets_BC_at_A1 : line_meets AP BC A1)
  (BP_meets_CA_at_B1 : line_meets BP CA B1)
  (CP_meets_AB_at_C1 : line_meets CP AB C1)
  (A1_midpoint_PA2 : midpoint A1 (P, A2))
  (B1_midpoint_PB2 : midpoint B1 (P, B2))
  (C1_midpoint_PC2 : midpoint C1 (P, C2))
  : ¬ (inside_circumcircle A2 ABC ∧ inside_circumcircle B2 ABC ∧ inside_circumcircle C2 ABC) := sorry

end cannot_all_lie_inside_circumcircle_l777_777105


namespace find_expression_of_odd_function_range_of_m_l777_777178

-- Problem 1: Finding the expression for f(x)
theorem find_expression_of_odd_function (a b c : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f(-x) = -f(x))
  (h_eq : ∀ x, f(x) = ax^3 + bx^2 + cx)
  (h_A : f(-sqrt 2) = sqrt 2)
  (h_B : f(2*sqrt 2) = 10 * sqrt 2) :
  f = λ x, x^3 - 3 * x :=
by sorry

-- Problem 2: Range of m for three distinct real roots
theorem range_of_m (m : ℝ) :
  (∀ x, (x^3 - 3*x + m = 0) → (∃! r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)) ↔ -2 < m ∧ m < 2 :=
by sorry

end find_expression_of_odd_function_range_of_m_l777_777178


namespace closest_approximation_l777_777619

theorem closest_approximation :
  (69.28 * 0.004) / x = 9.237333333333334 → x ≈ 0.03 := sorry

end closest_approximation_l777_777619


namespace sum_of_roots_of_quadratic_l777_777701

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l777_777701


namespace sinTripleAngle_sin18Degrees_l777_777659

-- Statement 1: Triple angle formula for sine
theorem sinTripleAngle (α : ℝ) : 
  sin (3 * α) = 3 * sin α - 4 * (sin α) ^ 3 := 
by 
  sorry

-- Statement 2: Value of sin 18 degrees
theorem sin18Degrees : 
  sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sinTripleAngle_sin18Degrees_l777_777659


namespace initial_tickets_l777_777792

-- Definitions of the conditions
def ferris_wheel_rides : ℕ := 2
def roller_coaster_rides : ℕ := 3
def log_ride_rides : ℕ := 7

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 1

def additional_tickets_needed : ℕ := 6

-- Calculate the total number of tickets needed
def total_tickets_needed : ℕ := 
  (ferris_wheel_rides * ferris_wheel_cost) +
  (roller_coaster_rides * roller_coaster_cost) +
  (log_ride_rides * log_ride_cost)

-- The proof statement
theorem initial_tickets : ∀ (initial_tickets : ℕ), 
  total_tickets_needed - additional_tickets_needed = initial_tickets → 
  initial_tickets = 20 :=
by
  intros initial_tickets h
  sorry

end initial_tickets_l777_777792


namespace regular_hexagon_same_length_probability_l777_777968

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777968


namespace sum_first_five_l777_777536

noncomputable def a : ℕ → ℝ := fun n => a₁ + (n - 1) * d

def a₃_is_2 (a₃ : ℝ) (h₃ : a₃ = 2) : Prop :=
  a 3 = a₃

def sum_first_5_terms (a₁ d : ℝ) : ℝ :=
  a₁ + (a₁ + d) + (a₁ + 2 * d) + (a₁ + 3 * d) + (a₁ + 4 * d)

theorem sum_first_five {a₁ d : ℝ} (h₃ : a₁ + 2 * d = 2) : sum_first_5_terms a₁ d = 10 :=
by sorry

end sum_first_five_l777_777536


namespace tailor_cut_skirt_l777_777757

theorem tailor_cut_skirt (x_pants x_add x_skirt : ℝ) (h1 : x_pants = 0.5) (h2 : x_add = 0.25) (h3 : x_skirt = x_pants + x_add) :
  x_skirt = 0.75 :=
by
  rw [h1, h2] at h3
  assumption

end tailor_cut_skirt_l777_777757


namespace no_such_convex_hexagon_and_point_exists_l777_777369

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777369


namespace frustum_volume_fraction_l777_777255

-- Define the original pyramid's base edge length and altitude
def base_edge : ℝ := 24
def altitude : ℝ := 15

-- Define the smaller pyramid's altitude
def smaller_pyramid_altitude : ℝ := altitude / 3

-- Define the volume ratio of similar pyramids based on their linear dimensions ratio cubed
def volume_ratio (h1 h2 : ℝ) : ℝ := (h2 / h1) ^ 3

-- Calculate the volume ratio for the smaller pyramid
def smaller_pyramid_volume_ratio : ℝ := volume_ratio altitude smaller_pyramid_altitude

-- Define the remaining volume ratio of the frustum
def remaining_frustum_volume_ratio : ℝ := 1 - smaller_pyramid_volume_ratio

-- Theorem (statement only, no proof required)
theorem frustum_volume_fraction :
  remaining_frustum_volume_ratio = 26 / 27 :=
sorry

end frustum_volume_fraction_l777_777255


namespace ab_zero_l777_777832

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l777_777832


namespace elizabeth_net_profit_l777_777400

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end elizabeth_net_profit_l777_777400


namespace volume_and_area_of_pyramid_l777_777154

-- Define the base of the pyramid.
def rect (EF FG : ℕ) : Prop := EF = 10 ∧ FG = 6

-- Define the perpendicular relationships and height of the pyramid.
def pyramid (EF FG PE : ℕ) : Prop := 
  rect EF FG ∧
  PE = 10 ∧ 
  (PE > 0) -- Given conditions include perpendicular properties, implying height is positive.

-- Problem translation: Prove the volume and area calculations.
theorem volume_and_area_of_pyramid (EF FG PE : ℕ) 
  (h1 : rect EF FG) 
  (h2 : PE = 10) : 
  (1 / 3 * EF * FG * PE = 200 ∧ 1 / 2 * EF * FG = 30) := 
by
  sorry

end volume_and_area_of_pyramid_l777_777154


namespace k_values_equation_satisfied_l777_777291

theorem k_values_equation_satisfied : 
  {k : ℕ | k > 0 ∧ ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s} = {2, 3, 4, 8} :=
by
  sorry

end k_values_equation_satisfied_l777_777291


namespace ab_zero_l777_777833

theorem ab_zero (a b : ℝ)
  (h1 : a + b = 5)
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 :=
by
  sorry

end ab_zero_l777_777833


namespace sum_of_roots_of_quadratic_l777_777698

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l777_777698


namespace largest_multiple_of_7_smaller_than_neg_55_l777_777209

theorem largest_multiple_of_7_smaller_than_neg_55 : ∃ m : ℤ, m % 7 = 0 ∧ m < -55 ∧ ∀ n : ℤ, n % 7 = 0 → n < -55 → n ≤ m :=
sorry

end largest_multiple_of_7_smaller_than_neg_55_l777_777209


namespace probability_of_same_length_segments_l777_777955

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777955


namespace positive_diff_between_median_and_mode_eq_16_l777_777680

/-
  List of numbers derived from the stem and leaf plot:
  tens_units: List Nat := [21, 22, 22, 23, 25, 32, 32, 32, 36, 36, 41, 41, 47, 48, 49, 50, 53, 53, 54, 57, 60, 61, 65, 65, 68]
-/

def tens_units: List Nat := [21, 22, 22, 23, 25, 32, 32, 32, 36, 36, 41, 41, 47, 48, 49, 50, 53, 53, 54, 57, 60, 61, 65, 65, 68]

noncomputable def median (l : List Nat) : Nat :=
  let sorted := l.sorted
  sorted.get! (sorted.length / 2)

noncomputable def modes (l : List Nat) : List Nat := 
  l.groupBy id
  |>.map (λ g => (g.head!, g.length))
  |>.filter (λ p => p.2 = l.groupBy id |>.map List.length |>.maximum)
  |>.map Prod.fst

noncomputable def positive_differences (a : Nat) (ls : List Nat) : List Nat :=
  ls.map (λ b => if b > a then b - a else a - b)

theorem positive_diff_between_median_and_mode_eq_16 :
  positive_differences (median tens_units) (modes tens_units) = [16] ∨ positive_differences (median tens_units) (modes tens_units) = [26, 16] := 
by 
  -- proof omitted
  sorry

end positive_diff_between_median_and_mode_eq_16_l777_777680


namespace expenditure_fraction_l777_777799

variable (B : ℝ)
def cost_of_book (x y : ℝ) (B : ℝ) := x = 0.30 * (B - 2 * y)
def cost_of_coffee (x y : ℝ) (B : ℝ) := y = 0.10 * (B - x)

theorem expenditure_fraction (x y : ℝ) (B : ℝ) 
  (hx : cost_of_book x y B) 
  (hy : cost_of_coffee x y B) : 
  (x + y) / B = 31 / 94 :=
sorry

end expenditure_fraction_l777_777799


namespace mrs_lopez_total_cost_l777_777138

def ticket_price (age : ℕ) : ℝ :=
  if age < 3 then 0
  else if age <= 12 then 8
  else if age >= 60 then 9
  else 11

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price - (price * discount / 100)

def total_ticket_cost : ℝ :=
  let lopez_and_husband := 2 * ticket_price 35
  let parents := 2 * ticket_price 73
  let children := ticket_price 7 + ticket_price 10 + ticket_price 14
  let nephews := ticket_price 6 + ticket_price 17
  let aunt := ticket_price 65
  let friends := 2 * ticket_price 33
  let total_cost := lopez_and_husband + parents + children + nephews + aunt + friends
  let husband_discount := apply_discount (ticket_price 35) 25
  let parents_discount := apply_discount (2 * ticket_price 73) 15
  let nephew_discount := apply_discount (ticket_price 17) 10
  total_cost
  - (ticket_price 35 - husband_discount)
  - (2 * ticket_price 73 - parents_discount)
  - (ticket_price 17 - nephew_discount)

theorem mrs_lopez_total_cost : total_ticket_cost = 110.45 := by
  sorry

end mrs_lopez_total_cost_l777_777138


namespace andrew_age_l777_777261

theorem andrew_age 
  (g a : ℚ)
  (h1: g = 16 * a)
  (h2: g - 20 - (a - 20) = 45) : 
 a = 17 / 3 := by 
  sorry

end andrew_age_l777_777261


namespace smallest_common_factor_n_l777_777211

theorem smallest_common_factor_n :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (9 * n - 2) ∧ d ∣ (7 * n + 3)) ∧ (∀ k : ℕ, 0 < k ∧ k < n → (¬ ∃ d > 1, d ∣ (9 * k - 2) ∧ d ∣ (7 * k + 3))) :=
by {
  use 23,
  split,
  { exact nat.lt_add_one_iff.mpr (nat.succ_pos 22) },
  split,
  { use 41,
    split,
    { exact nat.prime.two_lt nat.prime_41 },
    split,
    { exact nat.dvd_sub' (show 41 ∣ 205, from by norm_num) (show 41 ∣ 2, from nat.dvd_refl 2) },
    { exact nat.dvd_sub' (show 41 ∣ 164, from by norm_num) (show 41 ∣ 3, from nat.dvd_refl 3) } },
  intro k,
  cases' Classical.em (∃ d : ℕ, d > 1 ∧ d ∣ 9 * k - 2 ∧ d ∣ 7 * k + 3) with h h;
  { contradiction }
}

end smallest_common_factor_n_l777_777211


namespace geometric_series_sum_l777_777798

theorem geometric_series_sum :
  let a1 := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 5
  let s_n := a1 * (1 - r^n) / (1 - r)
  s_n = 205 / 1024 :=
by 
  let a1 := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 5
  let s_n := a1 * (1 - r^n) / (1 - r)
  have h1 : r ^ n = (-1 / 4) ^ 5 := by rfl
  have h2 : (-1 / 4) ^ 5 = -1 / 1024 := by norm_num
  have h3 : a1 * (1 - (-1 / 1024)) = a1 * (1 + 1 / 1024) := by ring
  have h4 : a1 * (1 + 1 / 1024) = (1 / 4) * (1025 / 1024) := by field_simp; ring
  have h5 : (1 / 4) * (1025 / 1024) / (1 + 1 / 4) = (1 / 4) * (1025 / 1024) / (5 / 4) := by ring
  have h6 : (1 / 4) * (1025 / 1024) / (5 / 4) = (1025 / 4096) / (5 / 4) := by field_simp; ring
  have h7 : (1025 / 4096) * 4 / 5 = 205 / 1024 := by field_simp; norm_num
  exact h7

end geometric_series_sum_l777_777798


namespace max_three_cell_corners_l777_777251

-- Define the grid size
def grid_height : ℕ := 7
def grid_width : ℕ := 14

-- Define the concept of a three-cell corner removal
def three_cell_corner (region : ℕ) : ℕ := region / 3

-- Define the problem statement in Lean
theorem max_three_cell_corners : three_cell_corner (grid_height * grid_width) = 32 := by
  sorry

end max_three_cell_corners_l777_777251


namespace tram_length_proof_l777_777588
-- Import the necessary library

-- Define the conditions
def tram_length : ℕ := 32 -- The length of the tram we want to prove

-- The main theorem to be stated
theorem tram_length_proof (L : ℕ) (v : ℕ) 
  (h1 : v = L / 4)  -- The tram passed by Misha in 4 seconds
  (h2 : v = (L + 64) / 12)  -- The tram passed through a tunnel of 64 meters in 12 seconds
  : L = tram_length :=
by
  sorry

end tram_length_proof_l777_777588


namespace minimum_largest_median_value_l777_777521

def minimum_largest_median (grid : matrix (fin 5) (fin 18) ℕ) (h1 : ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 90): ℕ :=
  let medians := λ j, (list.sort (≤) [(grid 0 j), (grid 1 j), (grid 2 j), (grid 3 j), (grid 4 j)]).nth_le 2 (by decide)
  in medians.max' (by { use medians 0, intros j, cases j; try {repeat {constructor}} })

theorem minimum_largest_median_value: ∀ (grid : matrix (fin 5) (fin 18) ℕ) (h1 : ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 90), 
  minimum_largest_median grid h1 = 54 :=
sorry

end minimum_largest_median_value_l777_777521


namespace find_unknown_number_l777_777410

theorem find_unknown_number
  (n : ℕ)
  (h_lcm : Nat.lcm n 1491 = 5964) :
  n = 4 :=
sorry

end find_unknown_number_l777_777410


namespace back_wheel_revolutions_l777_777594

theorem back_wheel_revolutions 
(radius_front radius_back : ℝ) 
(revs_front : ℕ) 
(h_radius_front : radius_front = 4) 
(h_radius_back : radius_back = 0.5) 
(h_revs_front : revs_front = 150) : 
  (2 * Math.pi * radius_front * revs_front) / (2 * Math.pi * radius_back) = 1200 := 
by
  rw [h_radius_front, h_radius_back, h_revs_front]
  calc
    (2 * Math.pi * 4 * 150) / (2 * Math.pi * 0.5)
      = (8 * Math.pi * 150) / (Math.pi)             : by simp [two_mul, mul_assoc]
  ... = 1200                                       : by simp [div_mul_eq_mul_div, mul_div_cancel_left, ne_of_gt, real.pi_pos]

end back_wheel_revolutions_l777_777594


namespace sum_of_roots_of_quadratic_l777_777686

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end sum_of_roots_of_quadratic_l777_777686


namespace square_difference_l777_777499

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l777_777499


namespace christmas_tree_problem_l777_777228

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end christmas_tree_problem_l777_777228


namespace num_integers_satisfying_cube_count_integers_satisfying_cube_l777_777026

theorem num_integers_satisfying_cube (n : ℤ) :
  (-100 < n^3 ∧ n^3 < 100) ↔ n ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4} := by
  sorry

theorem count_integers_satisfying_cube :
  {n : ℤ | -100 < n^3 ∧ n^3 < 100}.to_finset.card = 9 := by
  sorry

end num_integers_satisfying_cube_count_integers_satisfying_cube_l777_777026


namespace acute_triangle_perpendiculars_equal_l777_777069

theorem acute_triangle_perpendiculars_equal
  (A B C D E M N : Type)
  [acute_triangle : ∀ {α : Type}, α → α → α → α]
  (altitude_AD : ∀ {α : Type}, α → α → α)
  (altitude_CE : ∀ {α : Type}, α → α → α)
  (perpendicular_AM : ∀ {α : Type}, α → α → α)
  (perpendicular_CN : ∀ {α : Type}, α → α → α)
  (ME DN : Type) 
  (h1 : acute_triangle A B C)
  (h2 : altitude_AD A D)
  (h3 : altitude_CE C E)
  (h4 : perpendicular_AM A M)
  (h5 : perpendicular_CN C N)
  (h6 : ME = AE * cos (angle B C A))
  (h7 : DN = AC * cos (angle A B C))
  (h8 : ME = DN) : 
  ME = DN :=
by
  sorry

end acute_triangle_perpendiculars_equal_l777_777069


namespace probability_same_length_segments_l777_777943

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777943


namespace sweets_neither_red_nor_green_l777_777198

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end sweets_neither_red_nor_green_l777_777198


namespace probability_same_length_segments_l777_777937

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777937


namespace area_calc_l777_777140

noncomputable def region_area (n : ℕ) : ℝ :=
  if |x| + |y| ≤ n then 2 * n^2 else 0
  
theorem area_calc (n : ℝ) : 
  (∀ n, 
    (region_area 1 = 2) ∧ 
    (region_area 2 = 8) → 
    (region_area n = 2 * n^2)) :=
begin
  sorry
end

end area_calc_l777_777140


namespace volume_of_original_cube_l777_777144

theorem volume_of_original_cube (s : ℝ) (h : (s + 2) * (s - 3) * s - s^3 = 26) : s^3 = 343 := 
sorry

end volume_of_original_cube_l777_777144


namespace evaluate_expression_l777_777774

theorem evaluate_expression : 10 * 0.2 * 5 * 0.1 + 5 = 6 :=
by
  -- transformed step-by-step mathematical proof goes here
  sorry

end evaluate_expression_l777_777774


namespace rook_main_diagonal_exists_l777_777141

def rook_moves_adjacently (board : list (list ℕ)) : Prop :=
  ∀(i j : ℕ), i < 10 → j < 10 → 
    (board i j) = 1 → 
    ∃(di dj : ℕ), di ≤ 1 ∧ dj ≤ 1 ∧ 
      ¬ (board (i + di) (j + dj)) = 1

def visits_all_cells_once (board : list (list ℕ)) : Prop :=
  ∀(i j : ℕ), i < 10 → j < 10 → ∃(move : ℕ), board i j = move

def main_diagonal_condition (board : list (list ℕ)) : Prop :=
  ∀(k : ℕ), k < 10 → 
  ∃(m n : ℕ), m < 9 → n < 9 → 
  (board m m) = (board (m + 1) m) ∧ 
  (board m m) = (board m (m + 1))

theorem rook_main_diagonal_exists (board : list (list ℕ)) : 
  rook_moves_adjacently board →
  visits_all_cells_once board →
  main_diagonal_condition board :=
begin
  sorry
end

end rook_main_diagonal_exists_l777_777141


namespace no_such_convex_hexagon_and_point_l777_777339

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777339


namespace distinct_quadrilateral_areas_l777_777397

theorem distinct_quadrilateral_areas (A B C D E F : ℝ) 
  (h : A + B + C + D + E + F = 156) :
  ∃ (Q1 Q2 Q3 : ℝ), Q1 = 78 ∧ Q2 = 104 ∧ Q3 = 104 :=
sorry

end distinct_quadrilateral_areas_l777_777397


namespace problem_part1_problem_part2_l777_777458

open Real

variables {α : ℝ}

theorem problem_part1 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) :
  sin α * cos α = 3 / 10 := sorry

theorem problem_part2 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) (h3 : sin α * cos α = 3 / 10) :
  sin α + cos α = - (2 * sqrt 10 / 5) := sorry

end problem_part1_problem_part2_l777_777458


namespace arrangements_round_table_l777_777906

theorem arrangements_round_table (n : ℕ) (h : n = 5) : (n - 1)! = 24 :=
by {
  have n_eq_5 : n = 5 := h,
  have factorial_4 : 4! = 24 := by norm_num,
  rw [n_eq_5],
  exact factorial_4,
}

end arrangements_round_table_l777_777906


namespace max_students_seated_l777_777766

theorem max_students_seated : 
  (∑ i in Finset.range 25, Nat.ceil ((6 + 2 * (i + 1)) / 2) = 400) :=
sorry

end max_students_seated_l777_777766


namespace prove_x_minus_y_squared_l777_777492

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l777_777492


namespace schedule_problem_l777_777032

def num_schedule_ways : Nat :=
  -- total ways to pick 3 out of 6 periods and arrange 3 courses
  let total_ways := Nat.choose 6 3 * Nat.factorial 3
  -- at least two consecutive courses (using Principle of Inclusion and Exclusion)
  let two_consecutive := 5 * 6 * 4
  let three_consecutive := 4 * 6
  let invalid_ways := two_consecutive + three_consecutive
  total_ways - invalid_ways

theorem schedule_problem (h : num_schedule_ways = 24) : num_schedule_ways = 24 := by {
  exact h
}

end schedule_problem_l777_777032


namespace y_intercept_of_line_l777_777674

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end y_intercept_of_line_l777_777674


namespace CDsUniqueToSusanAndMaria_l777_777170

theorem CDsUniqueToSusanAndMaria (s m shared : ℕ) (h_s : s = 23) (h_m : m = 9) (h_shared : shared = 14) :
  (s - shared) + m = 18 :=
by
  rw [h_s, h_shared, h_m]
  sorry

end CDsUniqueToSusanAndMaria_l777_777170


namespace point_M_coordinates_l777_777857

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 4 * x

-- Define the condition given in the problem: instantaneous rate of change
def rate_of_change (a : ℝ) : Prop := f' a = -4

-- Define the point on the curve
def point_M (a b : ℝ) : Prop := f a = b

-- Proof statement
theorem point_M_coordinates : 
  ∃ (a b : ℝ), rate_of_change a ∧ point_M a b ∧ a = -1 ∧ b = 3 :=  
by
  sorry

end point_M_coordinates_l777_777857


namespace find_n_independent_polynomials_l777_777419

-- Define n-independent polynomial
def n_independent (P : Polynomial ℝ) (n : ℝ) : Prop := 
  P.eval n = 0 ∧ 
  ∀ Q ∈ {Q : Polynomial ℝ | ∃ p, p.is_permutation_of_coeff P ∧ Q = p}, 
    ∃ Q1, Q1 = Q.swap_coeffs ∧ Q1.eval n = 0

-- Main theorem to prove:
theorem find_n_independent_polynomials : 
  ∀ (P : Polynomial ℝ), P.degree = 2000 → (∃ n, n_independent P n) → ∃ n, n = 0 ∨ n = 1 :=
begin
  assume P hPdegree hnindep,
  sorry
end

end find_n_independent_polynomials_l777_777419


namespace row_product_const_implies_col_product_const_l777_777990

theorem row_product_const_implies_col_product_const {n : ℕ} (a b : Fin n → ℝ) 
(h : ∃ c : ℝ, ∀ i : Fin n, ∏ j : Fin n, (a i + b j) = c) : 
∃ C : ℝ, ∀ j : Fin n, ∏ i : Fin n, (a i + b j) = C := 
sorry

end row_product_const_implies_col_product_const_l777_777990


namespace division_by_fraction_l777_777488

theorem division_by_fraction (a b c d : ℚ) (hd : d ≠ 0) : (a / b) / (c / d) = (a / b) * (d / c) := by
  sorry

example : (9 / 5) / (1 / 3) = 27 / 5 := by
  apply division_by_fraction
  norm_num

end division_by_fraction_l777_777488


namespace ribbon_length_l777_777750

noncomputable def length_of_ribbon (h : ℝ) (d : ℝ) (turns : ℕ) : ℝ :=
  let circumference := d * Real.pi
  let width := turns * circumference
  Real.sqrt (h^2 + width^2)

theorem ribbon_length :
  length_of_ribbon 20 3 7 ≈ 68.9 :=
by
  sorry

end ribbon_length_l777_777750


namespace chord_length_of_parabola_l777_777465

-- Define the parabola
def parabola := { p : ℝ × ℝ // p.2 ^ 2 = 4 * (√3) * p.1 }

-- Define the condition that M is the midpoint of A and B with x-coordinate 2
def isMidpoint (A B M : ℝ × ℝ) : Prop := M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the chord length function
noncomputable def chordLength (A B : ℝ × ℝ) : ℝ := abs (A.1 - B.1)

-- Define the x-coordinate 2 condition for midpoint
def midpointXCondition (A B : ℝ × ℝ) : Prop := (A.1 + B.1) / 2 = 2

-- The main theorem statement
theorem chord_length_of_parabola (A B : parabola) 
  (h1 : midpointXCondition A.1 B.1) :
  chordLength A.1 B.1 = 4 + 2 * (√3) :=
by 
  sorry

end chord_length_of_parabola_l777_777465


namespace average_of_expressions_l777_777790

theorem average_of_expressions (y : ℝ) :
  (1 / 3:ℝ) * ((2 * y + 5) + (3 * y + 4) + (7 * y - 2)) = 4 * y + 7 / 3 :=
by sorry

end average_of_expressions_l777_777790


namespace sum_roots_eq_six_l777_777690

theorem sum_roots_eq_six : 
  ∀ x : ℝ, (x - 3) ^ 2 = 16 → (x - 3 = 4 ∨ x - 3 = -4) → (let x₁ := 3 + 4 in let x₂ := 3 - 4 in x₁ + x₂ = 6) := by
  sorry

end sum_roots_eq_six_l777_777690


namespace slope_angle_of_line_l777_777193

theorem slope_angle_of_line (t : ℝ) :
  let x := 3 + t * sin (30 * Real.pi / 180)
  let y := -t * cos (30 * Real.pi / 180)
  ∃ θ : ℝ, θ = 120 * Real.pi / 180 :=
  sorry

end slope_angle_of_line_l777_777193


namespace no_such_hexagon_exists_l777_777319

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777319


namespace arithmetic_sequence_general_formula_and_extremum_l777_777449

noncomputable def a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_general_formula_and_extremum :
  (a 1 + a 4 = 8) ∧ (a 2 * a 3 = 15) →
  (∃ c d : ℤ, (∀ n : ℕ, a n = c * n + d) ∨ (∀ n : ℕ, a n = -c * n + d)) ∧
  ((∃ n_min : ℕ, n_min > 0 ∧ S n_min = 1) ∧ (∃ n_max : ℕ, n_max > 0 ∧ S n_max = 16)) :=
by
  sorry

end arithmetic_sequence_general_formula_and_extremum_l777_777449


namespace no_such_hexagon_exists_l777_777303

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777303


namespace joanne_total_weekly_earnings_l777_777554

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end joanne_total_weekly_earnings_l777_777554


namespace cannot_determine_geometric_sum_l777_777645

-- Definitions for geometric sequences
def is_geometric_seq (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), (∀ (n : ℕ), s (n+1) = r * s n)

-- Problem statement: Prove that given sequences are geometric, we cannot determine if their sum is always geometric
theorem cannot_determine_geometric_sum (a b : ℕ → ℝ) 
  (h_a : is_geometric_seq a) 
  (h_b : is_geometric_seq b) : ¬(∃ c : ℕ → ℝ, c = λ n, a n + b n ∧ is_geometric_seq c) := 
sorry

end cannot_determine_geometric_sum_l777_777645


namespace composite_function_simplification_l777_777406

variable {x : ℝ}

noncomputable def y : ℝ := sin (log (sqrt x))

theorem composite_function_simplification (h : x > 0) : 
  y = sin (1 / 2 * log x) := by
sorry

end composite_function_simplification_l777_777406


namespace probability_same_length_segments_of_regular_hexagon_l777_777948

theorem probability_same_length_segments_of_regular_hexagon (sides diagonals : ℕ) 
  (total_segments : ℕ) (prob_same_length : ℚ) 
  (h_sides : sides = 6)
  (h_diagonals : diagonals = 9)
  (h_total_segments : total_segments = sides + diagonals)
  : prob_same_length = 17 / 35 :=
by
  have total_combinations := total_segments * (total_segments - 1) / 2
  have prob_sides := (sides * (sides - 1) / 2 : ℚ) / total_combinations
  have prob_diagonals := (diagonals * (diagonals - 1) / 2 : ℚ) / total_combinations
  have prob_sum := prob_sides + prob_diagonals
  have h_prob_sides : prob_sides = (6 * 5 / 2) / (15 * 14 / 2) :=
    by sorry
  have h_prob_diagonals : prob_diagonals = (9 * 8 / 2) / (15 * 14 / 2) :=
    by sorry
  calc
    prob_same_length = prob_sum : by sorry
    ... = 17 / 35 : by {rw [←h_prob_sides, ←h_prob_diagonals], sorry}

end probability_same_length_segments_of_regular_hexagon_l777_777948


namespace number_of_cats_l777_777082

theorem number_of_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 1212) 
  (h2 : dogs = 567) 
  (h3 : cats = total_animals - dogs) : 
  cats = 645 := 
by 
  sorry

end number_of_cats_l777_777082


namespace line_passes_point_l777_777173

theorem line_passes_point (k : ℝ) :
  ((1 + 4 * k) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k)) = 0 :=
by
  sorry

end line_passes_point_l777_777173


namespace number_of_new_birds_l777_777573

theorem number_of_new_birds (B F L S : ℕ) (h1 : B = 29) (h2 : B + F = 42) (h3 : L = 0.4 * (B + F)) (h4 : S = 0.6 * (B + F)) (h5 : F = 13):
  L = 17 ∧ S = 25 :=
by 
  sorry

end number_of_new_birds_l777_777573


namespace plot_length_more_than_breadth_l777_777182

noncomputable def cost_of_fencing (b : ℕ) : ℕ := 26.50 * (2 * 64 + 2 * b)

theorem plot_length_more_than_breadth : ∀ (b x : ℕ), cost_of_fencing b = 5300 ∧ b + x = 64 → x = 28 :=
by
  intro b x
  intro h
  sorry

end plot_length_more_than_breadth_l777_777182


namespace joanne_earnings_l777_777553

theorem joanne_earnings :
  let main_job_hourly_wage := 16.00
  let part_time_job_hourly_wage := 13.50
  let main_job_hours_per_day := 8
  let part_time_job_hours_per_day := 2
  let number_of_days := 5

  let main_job_daily_earnings := main_job_hours_per_day * main_job_hourly_wage
  let main_job_weekly_earnings := main_job_daily_earnings * number_of_days
  let part_time_job_daily_earnings := part_time_job_hours_per_day * part_time_job_hourly_wage
  let part_time_job_weekly_earnings := part_time_job_daily_earnings * number_of_days

  (main_job_weekly_earnings + part_time_job_weekly_earnings = 775)
:= by
  simp only [
    main_job_hourly_wage, part_time_job_hourly_wage,
    main_job_hours_per_day, part_time_job_hours_per_day,
    number_of_days,
    main_job_daily_earnings, main_job_weekly_earnings,
    part_time_job_daily_earnings, part_time_job_weekly_earnings
  ]
  sorry

end joanne_earnings_l777_777553


namespace induction_base_case_l777_777718

theorem induction_base_case : (-1 : ℤ) + 3 - 5 + (-1)^2 * 1 = (-1 : ℤ) := sorry

end induction_base_case_l777_777718


namespace trailing_zeros_2014_factorial_l777_777777

-- Define Legendre's formula to count the power of a prime factor within a factorial
def legendre (n p : ℕ) : ℕ :=
  let rec loop (k : ℕ) (sum : ℕ) : ℕ :=
    if p^k > n then sum else loop (k + 1) (sum + n / p^k)
  loop 1 0

-- Define the number n
def n := 2014

-- Define the prime factor p we care about
def p := 5

-- Calculate the number of trailing zeros in n!
def trailing_zeros (n p : ℕ) : ℕ :=
  legendre n p

theorem trailing_zeros_2014_factorial : trailing_zeros 2014 5 = 501 := by
  sorry

end trailing_zeros_2014_factorial_l777_777777


namespace length_of_bridge_l777_777630

-- Defining the conditions from the problem
def length_of_train : ℝ := 110
def speed_km_per_hr : ℝ := 45
def crossing_time_sec : ℝ := 30

-- Converting speed from km/hr to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

-- Total distance covered during the crossing
def total_distance : ℝ := speed_m_per_s * crossing_time_sec

-- The proof problem statement
theorem length_of_bridge : ∃ L : ℝ, L = total_distance - length_of_train :=
by 
  sorry

end length_of_bridge_l777_777630


namespace total_amount_spent_l777_777096

namespace KeithSpending

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost : ℝ := 112.46
def total_cost : ℝ := 387.85

theorem total_amount_spent : speakers_cost + cd_player_cost + tires_cost = total_cost :=
by sorry

end KeithSpending

end total_amount_spent_l777_777096


namespace hyperbola_eccentricity_l777_777866

noncomputable def hyperbola (a b : ℝ) : set (ℝ × ℝ) :=
{p | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

noncomputable def circle (a : ℝ) : set (ℝ × ℝ) :=
{p | p.1^2 + p.2^2 = a^2 / 4}

variable (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
variable (F E P : ℝ × ℝ)
variable (OP OE OF : ℝ × ℝ)

-- Given conditions
axiom h3 : F = (-c, 0)
axiom h4 : E ∈ circle a
axiom h5 : F ∈ hyperbola a b
axiom h6 : 2 • OE - OF = OP

-- Goal
theorem hyperbola_eccentricity :
  ∃ e : ℝ, e = (Real.sqrt 10) / 2 :=
sorry

end hyperbola_eccentricity_l777_777866


namespace measure_angle_ACB_eq_33_l777_777538

noncomputable def angle_ABC(angle_ABD : ℝ) : ℝ :=
  180 - angle_ABD

noncomputable def angle_ACB(angle_BAC : ℝ, angle_ABC : ℝ) : ℝ :=
  180 - (angle_BAC + angle_ABC)

theorem measure_angle_ACB_eq_33
  (angle_ABD : ℝ)
  (angle_BAC : ℝ)
  (angle_ABC_plus_ABD : angle_ABD = 120)
  (sum_of_angles_in_triangle : 180 - angle_ABC angle_ABD - angle_BAC = 33)
  : angle_ACB angle_BAC (angle_ABC angle_ABD) = 33 := by
  sorry

end measure_angle_ACB_eq_33_l777_777538


namespace a11_a12_a13_eq_105_l777_777567

variable (a : ℕ → ℝ) -- Define the arithmetic sequence
variable (d : ℝ) -- Define the common difference

-- Assume the conditions given in step a)
axiom arith_seq (n : ℕ) : a n = a 0 + n * d
axiom sum_3_eq_15 : a 0 + a 1 + a 2 = 15
axiom prod_3_eq_80 : a 0 * a 1 * a 2 = 80
axiom pos_diff : d > 0

theorem a11_a12_a13_eq_105 : a 10 + a 11 + a 12 = 105 :=
sorry

end a11_a12_a13_eq_105_l777_777567


namespace proof_problem_l777_777445

variables {a : ℕ → ℝ} {n : ℕ}

-- Given conditions
def seq_condition (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, n^2 * a (n - 2) * a (n + 2) = (n^2 - 4) * a n^2

-- Define correctness of each option
def CorrectOptionB (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 3 ∧ ∀ n ∈ ℕ, a (2 * n + 1) = 2 * n + 1

def CorrectOptionC (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = n

-- Proof problem statement
theorem proof_problem (a : ℕ → ℝ) 
    (h : seq_condition a) : CorrectOptionB a ∧ CorrectOptionC a :=
sorry

end proof_problem_l777_777445


namespace chemical_reactions_correct_l777_777295

-- Define the chemical formulas and reactions
def tertiary_alcohol : Type := sorry
def ketone : Type := sorry
def primary_alkane : Type := sorry
def ester : Type := sorry
def carboxylic_acid : Type := sorry
def alcohol : Type := sorry

def R₃COH : tertiary_alcohol := sorry
def R₂CO : ketone := sorry
def RH : primary_alkane := sorry
def RCOOR' : ester := sorry
def RCOOH : carboxylic_acid := sorry
def R'OH : alcohol := sorry
def K₂Cr₂O₇ : Type := sorry
def Cl₂ : Type := sorry

-- Define the reactions
def oxidation (alcohol : tertiary_alcohol) (oxidizing_agent : Type) : ketone := sorry
def substitution (alkane : primary_alkane) (halogen : Type) : Type := sorry
def hydrolysis (ester : ester) (water : Type) : Type := sorry

-- The equivalence to prove that given the conditions, the final products are correct.
theorem chemical_reactions_correct :
  oxidation R₃COH K₂Cr₂O₇ = R₂CO ∧
  substitution RH Cl₂ = (some result type ) ∧ -- need more specifics for substitution results
  hydrolysis RCOOR' (some water type) = (RCOOH, R'OH) :=
sorry

end chemical_reactions_correct_l777_777295


namespace comparison_of_average_speeds_l777_777279

noncomputable def average_speed_car_a (D u v : ℝ) : ℝ :=
  D / ((D / (2 * u)) + (D / (2 * v)))

noncomputable def average_speed_car_b (t u v : ℝ) : ℝ :=
  ((u + v) / 2)

theorem comparison_of_average_speeds
  (u v : ℝ)
  (hu : 0 < u)
  (hv : 0 < v) :
  average_speed_car_a 1 u v ≤ average_speed_car_b 1 u v :=
begin
  sorry
end

end comparison_of_average_speeds_l777_777279


namespace modulus_of_z_l777_777511

variable (z : ℂ)
variable (i : ℂ)
variable (h : i^2 = -1)

theorem modulus_of_z (h₁ : z * (1 + i) = 2 * i) : |z| = Real.sqrt 2 :=
by sorry

end modulus_of_z_l777_777511


namespace rice_cost_per_plate_l777_777735

-- Definitions based on the conditions from the problem
def total_plates := 100
def chicken_cost_per_plate := 0.40
def total_food_cost := 50.0

-- Statement to prove
theorem rice_cost_per_plate (x : ℝ) (h1 : x = (total_food_cost - total_plates * chicken_cost_per_plate) / total_plates) : x = 0.10 := by
  sorry

end rice_cost_per_plate_l777_777735


namespace f_2019_value_l777_777841

noncomputable def f : ℝ → ℝ :=
  λ x, if 1 ≤ x ∧ x < 3 then log 3 x + 1 else sorry -- f(x) has other values elsewhere which are not specified

lemma fx_periodic (x : ℝ) : f (x + 2) = f x :=
  sorry -- By condition, f(x) is periodic with period 2

lemma f_def (h : 1 ≤ 1 ∧ 1 < 3) : f 1 = log 3 1 + 1 :=
  sorry -- By condition, f(x) = log_3 x + 1 for x in [1, 3)

theorem f_2019_value : f 2019 = 1 :=
by
  have p : 2019 = 1 + 2 * 1009 := by norm_num
  rw [p]
  have periodic := fx_periodic (1 + 2 * 1009)
  rw [periodic]
  apply f_def
  exact ⟨by norm_num, by norm_num⟩

end f_2019_value_l777_777841


namespace triangle_perimeter_max_l777_777287

theorem triangle_perimeter_max (R : ℝ) :
  (∀ (x ∈ set.Ioo 0 π), convex_on (set.Ioo 0 π) sin) →
  (∀ A B C : ℝ, A + B + C = π → A ∈ set.Ioo 0 π → B ∈ set.Ioo 0 π → C ∈ set.Ioo 0 π →
    let a := sin A
    let b := sin B
    let c := sin C
    a + b + c ≤ 3 * sin (π / 3) →
    2 * R * (a + b + c) ≤ 3 * sqrt 3 * R) :=
s

end triangle_perimeter_max_l777_777287


namespace angle_A_is_75_l777_777608

-- Let quadrilateral ABCD be given, with vertices A, B, C, and D.
variables (A B C D : Type) [quadrilateral A B C D] [parallelogram A B C D]

-- Given that within triangle BCD, ∠DCB measures 75 degrees.
variable (angle_DCB : angle (D C B) = 75)

-- And triangle BCD is isosceles with BC = CD.
variable (isosceles_BCD : isosceles (B C D) (segment B C) (segment C D))

-- Prove that the measure of ∠A is 75 degrees.
theorem angle_A_is_75 : angle (A) = 75 := by
  sorry

end angle_A_is_75_l777_777608


namespace range_f_domain_minus1_1_g_a_relationship_and_domain_exists_x2_for_g_eq_fx2_plus_m_l777_777860

noncomputable def f (x : ℝ) := 4^x - 2^x

noncomputable def a (s t : ℝ) := 2^s + 2^t
noncomputable def b (s t : ℝ) := 2^(s + t)

noncomputable def g (a : ℝ) := (a^2 - a) / 2

theorem range_f_domain_minus1_1 : 
  ∀ x ∈ Icc (-1 : ℝ) 1, ∃ y ∈ Icc (-1/4 : ℝ) 2, f x = y := 
sorry

theorem g_a_relationship_and_domain :
  ∀ s t : ℝ, f s + f t = 0 → b s t = g (a s t) ∧ 1 < a s t ∧ a s t ≤ 2 := 
sorry

theorem exists_x2_for_g_eq_fx2_plus_m:
  ∀ x1 ∈ Ioc 1 2, ∃ x2 ∈ Icc (-1 : ℝ) 1, ∃ m ∈ Icc (-1 : ℝ) (1/4), g x1 = f x2 + m := 
sorry

end range_f_domain_minus1_1_g_a_relationship_and_domain_exists_x2_for_g_eq_fx2_plus_m_l777_777860


namespace log_ab_is_pi_l777_777240

open Real

noncomputable def log_ab (a b : ℝ) : ℝ :=
(log b) / (log a)

theorem log_ab_is_pi (a b : ℝ)  (ha_pos: 0 < a) (ha_ne_one: a ≠ 1) (hb_pos: 0 < b) 
  (cond1 : log (a ^ 3) = log (b ^ 6)) (cond2 : cos (π * log a) = 1) : log_ab a b = π :=
by
  sorry

end log_ab_is_pi_l777_777240


namespace simplify_expression_eq_69_l777_777163

theorem simplify_expression_eq_69 : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end simplify_expression_eq_69_l777_777163


namespace radius_of_circles_l777_777826

-- Define points A, B, and C
def point (α : Type) := prod α α
def A : point ℝ := (-1, 2)
def B : point ℝ := (1, 2)
def C : point ℝ := (5, -2)

-- Define a metric space for the point
instance : metric_space (point ℝ) := sorry

-- Distance function
def dist (p q : point ℝ) : ℝ := sorry

-- Define the conditions and goal
theorem radius_of_circles (r : ℝ) (M N : point ℝ) :
  dist M B = r ∧ dist N B = r ∧ dist A B = dist C B :=
  r = sqrt 10 := sorry

end radius_of_circles_l777_777826


namespace maximize_intersection_area_trapezoid_l777_777596

/-- Given a trapezoid ABCD with bases AD and BC, and a point K on AD,
find the point M on BC such that the area of the common part of triangles
AMD and BKC is maximized. Prove that this point M satisfies the ratio
BM / MC == AK / KD. -/
theorem maximize_intersection_area_trapezoid (A B C D K M : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_K_on_AD : on_line K A D)
  (M_on_BC : on_line M B C) :
  let AK := line_segment A K
      KD := line_segment K D
      BM := line_segment B M
      MC := line_segment M C in
  (area (triangle A M D) ∩ area (triangle B K C) ≤ area (triangle A B D)) ∧
  (area (triangle A M D) ∩ area (triangle B K C) ≤ area (triangle B C D)) ↔
  (BM.length / MC.length = AK.length / KD.length) :=
sorry

end maximize_intersection_area_trapezoid_l777_777596


namespace relationship_between_y_values_l777_777452

noncomputable def y_relation (y1 y2 y3 c : ℝ) : Prop :=
let f : ℝ → ℝ := λ x, 3 * x ^ 2 - 6 * x + c in
    f (-3) = y1 ∧ f (-1) = y2 ∧ f (5) = y3 ∧ (y1 = y3 ∧ y1 > y2)

theorem relationship_between_y_values (y1 y2 y3 c : ℝ) :
  y_relation y1 y2 y3 c := sorry

end relationship_between_y_values_l777_777452


namespace cross_colors_differ_l777_777765

-- Hypothesis: All cells of a grid are colored in 5 colors.
variables {Color : Type} [fintype Color] [decidable_eq Color] (A : ℕ → ℕ → Color)

-- Hypothesis: In any figure of the form 1x5 strip, all colors are different.
def valid_strip (i j : ℕ) : Prop :=
  (finset.univ.image (A i ∘ (j + .)).to_finset = finset.univ)

-- The target is to prove that in any figure of the 2x2 form, all colors are different.
theorem cross_colors_differ (i j : ℕ) (h_strip : ∀ i j, valid_strip A i j) : 
  ¬ (A i j = A (i + 1) j) ∧ 
  ¬ (A i j = A i (j + 1)) ∧ 
  ¬ (A i j = A (i + 1) (j + 1)) ∧ 
  ¬ (A (i + 1) j = A i (j + 1)) ∧ 
  ¬ (A (i + 1) j = A (i + 1) (j + 1)) ∧ 
  ¬ (A i (j + 1) = A (i + 1) (j + 1)) :=
begin
  sorry
end

end cross_colors_differ_l777_777765


namespace frisbee_sales_l777_777752

/-- A sporting goods store sold some frisbees, with $3 and $4 price points.
The total receipts from frisbee sales were $204. The fewest number of $4 frisbees that could have been sold is 24.
Prove the total number of frisbees sold is 60. -/
theorem frisbee_sales (x y : ℕ) (h1 : 3 * x + 4 * y = 204) (h2 : 24 ≤ y) : x + y = 60 :=
by {
  -- Proof skipped
  sorry
}

end frisbee_sales_l777_777752


namespace root_interval_l777_777123

noncomputable def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ x ∈ (set.Ioo 1.25 1.5), f x = 0 :=
sorry

end root_interval_l777_777123


namespace largest_prime_factor_sum_of_divisors_180_l777_777118

theorem largest_prime_factor_sum_of_divisors_180 : 
  let N := ∑ d in (finset.divisors 180), d
  in prime_factors N ∪ {13} = prime_factors N :=
by
  sorry

end largest_prime_factor_sum_of_divisors_180_l777_777118


namespace regular_decagon_interior_angle_l777_777051

theorem regular_decagon_interior_angle (n : ℕ) (h1 : n = 10) (h2 : ∀ i j : ℕ, (i < n ∧ j < n) → interior_angle n i = interior_angle n j) : 
  interior_angle n 0 = 144 :=
by
  sorry

end regular_decagon_interior_angle_l777_777051


namespace find_fx_for_negative_x_l777_777420

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem find_fx_for_negative_x (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_given : given_function f) :
  ∀ x, (x < 0) → f x = x + x^2 :=
by
  sorry

end find_fx_for_negative_x_l777_777420


namespace joanne_total_weekly_earnings_l777_777555

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end joanne_total_weekly_earnings_l777_777555


namespace days_to_complete_work_alone_l777_777225

noncomputable def work_rate_x (Dx : ℕ) : ℝ := 1 / Dx
def work_rate_y : ℝ := 1 / 20
def combined_work_rate (Dx : ℕ) : ℝ := work_rate_x Dx + work_rate_y
def fraction_work_completed (Dx : ℕ) : ℝ := 5 * combined_work_rate Dx
def target_fraction : ℝ := 0.6071428571428572

theorem days_to_complete_work_alone : ∃ Dx : ℕ, fraction_work_completed Dx = target_fraction ∧ Dx ≈ 14 :=
  sorry

end days_to_complete_work_alone_l777_777225


namespace horse_grazing_area_l777_777743

theorem horse_grazing_area (field_length : ℝ) (field_width : ℝ) (rope_length : ℝ) :
  field_length = 45 → field_width = 25 → rope_length = 22 → 
  ∃ (A_graze : ℝ), A_graze = 121 * Real.pi := 
by
  intros h1 h2 h3
  use 121 * Real.pi
  exact sorry

end horse_grazing_area_l777_777743


namespace avg_age_of_team_is_23_l777_777171

-- Conditions
def captain_age := 24
def wicket_keeper_age := captain_age + 7

def remaining_players_avg_age (team_avg_age : ℝ) := team_avg_age - 1
def total_team_age (team_avg_age : ℝ) := 11 * team_avg_age
def total_remaining_players_age (team_avg_age : ℝ) := 9 * remaining_players_avg_age team_avg_age

-- Proof statement
theorem avg_age_of_team_is_23 (team_avg_age : ℝ) :
  total_team_age team_avg_age = captain_age + wicket_keeper_age + total_remaining_players_age team_avg_age → 
  team_avg_age = 23 :=
by
  sorry

end avg_age_of_team_is_23_l777_777171


namespace prime_factors_of_N_l777_777789

theorem prime_factors_of_N (N : ℕ) (h : log 2 (log 3 (log 5 (N^2))) = 5) :
  ∃! p, p.prime ∧ p ∣ N :=
sorry

end prime_factors_of_N_l777_777789


namespace three_digit_odd_count_l777_777159

theorem three_digit_odd_count : 
  let digits_even := {0, 2}
  let digits_odd := {1, 3, 5}
  ∃ (count : ℕ), 
    (count = 18) ∧
    (∀ h t u, h ∈ digits_even ∧ t ∈ digits_odd ∧ u ∈ digits_odd ∧ u ≠ t ∧ ((100 * h + 10 * t + u) % 2 = 1) 
      → true) :=
by { sorry }

end three_digit_odd_count_l777_777159


namespace fixed_deposit_maturity_with_interest_l777_777890

-- Define the conditions
def initial_principal : ℝ := 1000
def annual_interest_rate : ℝ := 0.14
def monthly_subsidy : ℝ := 0.07
def monthly_interest_rate_excess : ℝ := 0.02
def three_years : ℝ := 3
def additional_half_year : ℝ := 0.5

-- Define the expected result
noncomputable def total_principal_and_interest : ℝ := 4390.4

-- Full problem statement in Lean with the conditions translated to definitions
theorem fixed_deposit_maturity_with_interest 
  (initial_principal annual_interest_rate monthly_subsidy monthly_interest_rate_excess three_years additional_half_year : ℝ) :
  let total_amount := (initial_principal + initial_principal * annual_interest_rate * three_years + initial_principal * monthly_subsidy * 12 * three_years) * (1 + monthly_interest_rate_excess * (additional_half_year * 12)) in
  total_amount = total_principal_and_interest := 
by
  sorry

end fixed_deposit_maturity_with_interest_l777_777890


namespace pentagon_paintings_l777_777738

theorem pentagon_paintings (A B C D E : ℕ) (colors : Finset ℕ)
  (h_convex : true)  -- Assuming the pentagon is convex, we don't need an actual condition for convexity in this context.
  (h_distinct : ∀ x ∈ colors, ∀ y ∈ colors, x ≠ y) : 
  (∀ u v ∈ (Finset.pair A colors), u ≠ v) →
  Finset.card colors = 5 →
  ∃ m : ℕ, m = 1020 :=
begin
  sorry
end

end pentagon_paintings_l777_777738


namespace rectangular_prism_diagonal_l777_777252

theorem rectangular_prism_diagonal :
  ∀ (h : ℝ), h = 12 →
  ∀ (l : ℝ), l = 2 * h →
  ∀ (w : ℝ), w = l / 2 →
  ∃ (d : ℝ), d = 12 * Real.sqrt 6 :=
by
  intros h h_eq l l_eq w w_eq
  use 12 * Real.sqrt 6
  sorry

end rectangular_prism_diagonal_l777_777252


namespace no_such_hexagon_exists_l777_777300

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777300


namespace find_base_l777_777509

theorem find_base (x y : ℕ) (b : ℕ) (h1 : 3 ^ x * b ^ y = 19683) (h2 : x - y = 9) (h3 : x = 9) : b = 1 := 
by
  sorry

end find_base_l777_777509


namespace total_fuel_usage_is_250_l777_777094

-- Define John's fuel consumption per km
def fuel_consumption_per_km : ℕ := 5

-- Define the distance of the first trip
def distance_trip1 : ℕ := 30

-- Define the distance of the second trip
def distance_trip2 : ℕ := 20

-- Define the fuel usage calculation
def fuel_usage_trip1 := distance_trip1 * fuel_consumption_per_km
def fuel_usage_trip2 := distance_trip2 * fuel_consumption_per_km
def total_fuel_usage := fuel_usage_trip1 + fuel_usage_trip2

-- Prove that the total fuel usage is 250 liters
theorem total_fuel_usage_is_250 : total_fuel_usage = 250 := by
  sorry

end total_fuel_usage_is_250_l777_777094


namespace sum_first_10_terms_is_75_l777_777626

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def seq (n : ℕ) : ℕ := S_n(n) / n

noncomputable def sum_first_10_terms : ℕ := (List.range 10).sum (λ n, seq (n + 1))

theorem sum_first_10_terms_is_75 : sum_first_10_terms = 75 :=
by
  sorry

end sum_first_10_terms_is_75_l777_777626


namespace volume_equal_rotation_l777_777651

theorem volume_equal_rotation (a b x : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (4 / 3) * π * a^2 * b = (4 / 3) * π * a^2 * b + π * a^2 * x + (2 / 3) * π * a^2 * x → x = 0 :=
by
  intro h
  linarith

end volume_equal_rotation_l777_777651


namespace abc_l777_777912

theorem abc.geometry_problem
  (A B C D E F H P Q R : Type)
  [triangle : Triangle A B C]
  [altitudes : IsAltitude A D (Line B C) ∧ IsAltitude B E (Line A C) ∧ IsAltitude C F (Line A B)]
  [orthocenter : IsOrthocenter H A B C]
  [circle_AH : CircleContains AH A ∧ CircleContains AH H]
  [intersections : CircleContains AH Q ∧ CircleContains AH P ∧ Q ≠ A ∧ P ≠ A ∧ Q ∈ (Line A B) ∧ P ∈ (Line A C)]
  [circle_OPQ : CircleContains (Circumcircle P Q) O]
  [tangency : TangentToCircle (Circumcircle O P Q) (Line B C) R]
  : (CR / BR) = (ED / FD) := sorry

end abc_l777_777912


namespace total_people_l777_777066

-- Definitions of the given conditions
variable (I N B Ne T : ℕ)

-- These variables represent the given conditions
axiom h1 : I = 25
axiom h2 : N = 23
axiom h3 : B = 21
axiom h4 : Ne = 23

-- The theorem we want to prove
theorem total_people : T = 50 :=
by {
  sorry -- We denote the skipping of proof details.
}

end total_people_l777_777066


namespace remainder_of_1000th_term_div_3_l777_777285

noncomputable def sequence1000thTerm : ℕ :=
  let n := nat.find (λ n, (n * (n + 1)) / 2 ≥ 1000) in
  n

theorem remainder_of_1000th_term_div_3 : (sequence1000thTerm % 3) = 0 :=
sorry

end remainder_of_1000th_term_div_3_l777_777285


namespace percentage_A_to_B_l777_777655

variable (A B : ℕ)
variable (total : ℕ := 570)
variable (B_amount : ℕ := 228)

theorem percentage_A_to_B :
  (A + B = total) →
  B = B_amount →
  (A = total - B_amount) →
  ((A / B_amount : ℚ) * 100 = 150) :=
sorry

end percentage_A_to_B_l777_777655


namespace hyperbola_from_circle_conditions_l777_777855

theorem hyperbola_from_circle_conditions :
  let C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2*x - 5*y + 4 = 0
  let H : ℝ := 1
  let K : ℝ := 1
  let a : ℝ := 1
  let b : ℝ := √15
  ∀ x y : ℝ, C (0 : ℝ) 1 ∧ C (0 : ℝ) 4 → 
    (y - K)^2 / a^2 - x^2 / b^2 = 1 := sorry

end hyperbola_from_circle_conditions_l777_777855


namespace B_cardinality_l777_777016

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_even (n : ℕ) : Prop := n % 2 = 0

def B : Finset ℕ := 
  {x | ∃ a ∈ A, ∃ b ∈ A, is_even (a * b) ∧ x = a * b}.toFinset

theorem B_cardinality : B.card = 15 := 
sorry

end B_cardinality_l777_777016


namespace least_value_d_eq_d0_all_permutations_with_d0_l777_777559

-- Defining the necessary components.
def permutation (n : ℕ) := {l : list ℕ // l.perm (list.range n)}

def set_A (n : ℕ) : set (permutation n) :=
  {a | ∀ S : finset ℕ, S.card < n → ¬(S.image (λ k, (a.1.nth_le k (k.prop)))) = S}

def d (a : permutation 2003) : ℕ :=
  finset.sum (finset.range 2003) (λ k, (a.1.nth_le k (by linarith)) - k) ^ 2

noncomputable def d_0 : ℕ := 4010006

theorem least_value_d_eq_d0 :
  ∃ a ∈ set_A 2003, d a = d_0 := sorry

theorem all_permutations_with_d0 :
  ∀ a ∈ set_A 2003, d a = d_0 ↔ ∃ k, a = {l := list.rotate (list.range 2003) k, perm := sorry} := sorry

end least_value_d_eq_d0_all_permutations_with_d0_l777_777559


namespace no_such_hexagon_exists_l777_777332

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777332


namespace hexagon_division_parts_new_hexagon_area_correct_l777_777230

-- Given conditions
def hexagon_area : ℝ := 144 -- in cm²
def original_parts : ℕ := 6 * 4 -- total parts the original hexagon is divided into

-- Statement proving number of divided parts
theorem hexagon_division_parts (H: original_parts = 24): true := by {
  sorry
}

-- Given condition: division forming new hexagon
def new_hexagon_area : ℝ :=
  hexagon_area / 3 -- area of the new hexagon formed

-- Statement proving area of new hexagon composed of quadrilaterals
theorem new_hexagon_area_correct (H: new_hexagon_area = 48): true := by {
  sorry
}

end hexagon_division_parts_new_hexagon_area_correct_l777_777230


namespace xy_product_l777_777464

noncomputable def f (t : ℝ) : ℝ := Real.sqrt (t^2 + 1) - t + 1

theorem xy_product (x y : ℝ)
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) :
  x * y = 1 := by
  sorry

end xy_product_l777_777464


namespace area_BDGF_is_338_l777_777560

-- Let's define the problem in Lean 4
variable {A B C D E F G : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]

noncomputable def area_of_quadrilateral (A B C D E F G : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G] : ℝ := 
  -- conditions implied to be used later in the proof
  sorry

theorem area_BDGF_is_338 {A B C D E F G : Type*} 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]
  (h₁ : ∠ BAC = 90)
  (h₂ : AB = 24)
  (h₃ : AC = 10)
  (h₄ : constr_square B C D E)
  (h₅ : bisect_angle A B C F G) :
  area_of_quadrilateral A B C D E F G = 338 := 
sorry

end area_BDGF_is_338_l777_777560


namespace function_form_l777_777562

def satisfies_condition (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → ⌊ (f (m * n) : ℚ) / n ⌋ = f m

theorem function_form (f : ℕ → ℤ) (h : satisfies_condition f) :
  ∃ r : ℝ, ∀ n : ℕ, 
    (f n = ⌊ (r * n : ℝ) ⌋) ∨ (f n = ⌈ (r * n : ℝ) ⌉ - 1) := 
  sorry

end function_form_l777_777562


namespace farming_problem_proof_l777_777795

-- Define the problem conditions
def prob_eggplant : ℚ := 1 / 3
def prob_cucumber : ℚ := 2 / 3
def emergence_rate_eggplant : ℚ := 0.95
def emergence_rate_cucumber : ℚ := 0.98
def num_plots : ℕ := 4

-- Define the main proof problem
theorem farming_problem_proof :
  -- Probability of exactly 2 plots planting eggplants
  (∃ (n_eggplants : ℕ), n_eggplants = 2 ∧ (num_plots.choose n_eggplants : ℚ) * (prob_eggplant ^ n_eggplants) * (prob_cucumber ^ (num_plots - n_eggplants)) = 8 / 27) ∧
  -- Germination rate of seeds for each plot
  (prob_eggplant * emergence_rate_eggplant + prob_cucumber * emergence_rate_cucumber = 0.97) ∧
  -- Distribution of the random variable X
  (∀ (X : ℕ), X ∈ {0, 1, 2} ∧ 
    (if X = 0 then 1 / 25 else
      if X = 1 then 16 / 25 else
      if X = 2 then 8 / 25 else 0
    ) ∧
    -- Expected value of X
    (0 * (1 / 25) + 1 * (16 / 25) + 2 * (8 / 25) = 32 / 25)
  ) :=
  sorry

end farming_problem_proof_l777_777795


namespace no_such_hexagon_exists_l777_777361

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777361


namespace sequence_general_term_l777_777253

def sequence (a : ℕ → ℕ) :=
  a 1 = 3 ∧ a 2 = 9 ∧ ∀ n > 1, a (n + 1) = 4 * a n - 3 * a (n - 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence a) :
  ∀ n, a n = 3^n :=
by
  sorry

end sequence_general_term_l777_777253


namespace sin_tan_condition_l777_777044

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end sin_tan_condition_l777_777044


namespace radius_of_circle_Q_l777_777780

theorem radius_of_circle_Q 
  (externally_tangent : ∀ (P Q R S : Circle), tangent_outside P Q ∧ tangent_outside Q R ∧ tangent_outside R P ∧ tangent_inside S P ∧ tangent_inside S Q ∧ tangent_inside S R)
  (Q_R_congruent : circle_Q.radius = circle_R.radius)
  (P_radius : circle_P.radius = 2)
  (P_center_in_S : S_center ∈ circle_P) :
  circle_Q.radius = 16 / 9 := 
sorry

end radius_of_circle_Q_l777_777780


namespace rectangle_area_EFGH_l777_777907

open Real

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem rectangle_area_EFGH :
  let E := (1 : ℝ, -5 : ℝ),
      F := (2001 : ℝ, 195 : ℝ),
      H := (3 : ℝ, -25 : ℝ) in
  let y := -25 in
  slope E F * slope E H = -1 ∧
  distance E F * distance E H = 40_400 :=
by 
  -- Coordinates of points
  let E := (1 : ℝ, -5 : ℝ)
  let F := (2001 : ℝ, 195 : ℝ)
  let H := (3 : ℝ, -25 : ℝ)
  
  -- Slopes and intermediate point calculation
  have h_slope_EF : slope E F = 1 / 10 := by sorry
  have h_slope_EH : slope E H * (1 / 10) = -1 := by sorry
  
  -- Proof of perpendicular slopes
  have h_perpendicular : slope E F * slope E H = -1 := by sorry
  
  -- Length calculations
  have h_distance_EF : distance E F = 200 * sqrt 101 := by sorry
  have h_distance_EH : distance E H = 2 * sqrt 101 := by sorry

  -- Area calculation
  have h_area : (distance E F * distance E H = 40_400) := by sorry
  
  use h_perpendicular, h_area
  sorry

end rectangle_area_EFGH_l777_777907


namespace total_lunch_bill_l777_777158

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h1 : cost_hotdog = 5.36) (h2 : cost_salad = 5.10) : 
  cost_hotdog + cost_salad = 10.46 := 
by 
  sorry

end total_lunch_bill_l777_777158


namespace range_of_a_l777_777478

variable (a x : ℝ)

def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem range_of_a (h : A ∪ B a = B a) : a ≤ 0 :=
begin
  -- Sorry is used to omit the proof
  sorry,
end

end range_of_a_l777_777478


namespace sin_transformation_l777_777426

theorem sin_transformation (alpha : ℝ) (h : cos (π/3 + alpha) = 1/3) :
  sin (5/6 * π + alpha) = 1/3 :=
by
  sorry

end sin_transformation_l777_777426


namespace within_acceptable_range_l777_777759

def flour_weight : ℝ := 25.18
def flour_label : ℝ := 25
def tolerance : ℝ := 0.25

theorem within_acceptable_range  :
  (flour_label - tolerance) ≤ flour_weight ∧ flour_weight ≤ (flour_label + tolerance) :=
by
  sorry

end within_acceptable_range_l777_777759


namespace magnitude_of_2a_sub_b_is_sqrt7_l777_777020

open Real EuclideanSpace

noncomputable def magnitude_of_vector {n : ℕ} (a b : EuclideanSpace (Fin n) ℝ) (h₁ : ⟪a, b⟫ = 0) (ha : ‖a‖ = 1) (hb : ‖b‖ = sqrt 3) : ℝ :=
  ‖2 • a - b‖

theorem magnitude_of_2a_sub_b_is_sqrt7 
{a b : EuclideanSpace (Fin 2) ℝ} 
(h₁ : ⟪a, b⟫ = 0) 
(ha : ‖a‖ = 1) 
(hb : ‖b‖ = sqrt 3) 
: magnitude_of_vector a b h₁ ha hb = sqrt 7 := 
sorry

end magnitude_of_2a_sub_b_is_sqrt7_l777_777020


namespace projection_matrix_equiv_l777_777988

-- Define the projection matrix P_1
def P1 : Matrix (Fin 2) (Fin 2) ℚ := 
  ⟦ ⟨(4/5 : ℚ), (2/5 : ℚ)⟩, ⟨(2/5 : ℚ), (1/5 : ℚ)⟩ ⟧

-- Define the projection matrix P_2
def P2 : Matrix (Fin 2) (Fin 2) ℚ :=
  ⟦ ⟨(1/2 : ℚ), (1/2 : ℚ)⟩, ⟨(1/2 : ℚ), (1/2 : ℚ)⟩ ⟧

-- Define the product matrix
def P2P1 : Matrix (Fin 2) (Fin 2) ℚ :=
  P2.mul P1

-- The statement to prove
theorem projection_matrix_equiv :
  P2P1 = ⟦⟨(3/5 : ℚ), (3/10 : ℚ)⟩, ⟨(3/5 : ℚ), (3/10 : ℚ)⟩⟧ :=
by
  sorry

end projection_matrix_equiv_l777_777988


namespace probability_same_length_segments_l777_777939

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777939


namespace mike_planted_50_l777_777916

-- Definitions for conditions
def mike_morning (M : ℕ) := M
def ted_morning (M : ℕ) := 2 * M
def mike_afternoon := 60
def ted_afternoon := 40
def total_planted (M : ℕ) := mike_morning M + ted_morning M + mike_afternoon + ted_afternoon

-- Statement to prove
theorem mike_planted_50 (M : ℕ) (h : total_planted M = 250) : M = 50 :=
by
  sorry

end mike_planted_50_l777_777916


namespace original_solution_concentration_l777_777712

variable (C : ℝ) -- Concentration of the original solution as a percentage.
variable (v_orig : ℝ := 12) -- 12 ounces of the original vinegar solution.
variable (w_added : ℝ := 50) -- 50 ounces of water added.
variable (v_final_pct : ℝ := 7) -- Final concentration of 7%.

theorem original_solution_concentration :
  (C / 100 * v_orig = v_final_pct / 100 * (v_orig + w_added)) →
  C = (v_final_pct * (v_orig + w_added)) / v_orig :=
sorry

end original_solution_concentration_l777_777712


namespace probability_at_least_one_even_l777_777451

theorem probability_at_least_one_even (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5}) : 
  (∃ (comb : Finset (Finset ℕ)), comb.card = (S.card.choose 3) ∧ 
    (∀ (x : Finset ℕ), x ∈ comb → x.card = 3 ∧ 
      ∃ (rem : Finset ℕ), rem = S \ x ∧ rem.card = 2 ∧ 
      (1 ≤ rem.filter (λ n, n % 2 = 0).card)) ∧ 
    ((S.card.choose 2).toRational / (S.card.choose 3).toRational = 0.3)) → 
  ((1 - (3 / 10 : ℚ)) = 7 / 10) :=
by 
  intros,
  sorry

end probability_at_least_one_even_l777_777451


namespace rectangle_diagonal_length_l777_777189

theorem rectangle_diagonal_length (p : ℝ) (r_lw : ℝ) (l w d : ℝ) 
    (h_p : p = 84) 
    (h_ratio : r_lw = 5 / 2) 
    (h_l : l = 5 * (p / 2) / 7) 
    (h_w : w = 2 * (p / 2) / 7) 
    (h_d : d = Real.sqrt (l ^ 2 + w ^ 2)) :
  d = 2 * Real.sqrt 261 :=
by
  sorry

end rectangle_diagonal_length_l777_777189


namespace tiling_tromino_l777_777926

theorem tiling_tromino (m n : ℕ) : (∀ t : ℕ, (t = 3) → (3 ∣ m * n)) → (m * n % 6 = 0) → (m * n % 6 = 0) :=
by
  sorry

end tiling_tromino_l777_777926


namespace simplify_expression_l777_777162

variable {R : Type*} [Field R]

theorem simplify_expression (x y z : R) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
sorry

end simplify_expression_l777_777162


namespace find_a_l777_777017

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a ^ 2}

theorem find_a (a : ℝ) (h : A ∪ B a = {0, 1, 2, 4}) : a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l777_777017


namespace geometric_sequence_general_term_geometric_sequence_sum_n_l777_777440

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end geometric_sequence_general_term_geometric_sequence_sum_n_l777_777440


namespace missing_number_l777_777721

theorem missing_number (x : ℝ) : (306 / x) * 15 + 270 = 405 ↔ x = 34 := 
by
  sorry

end missing_number_l777_777721


namespace diameter_of_wheel_is_approx_0_66_l777_777726

noncomputable def diameter_of_wheel : ℝ :=
  let π : ℝ := Real.pi in
  let distance : ℝ := 1000 -- distance in meters
  let revolutions : ℝ := 482.5323296660876 -- number of complete revolutions in 1 km
  let circumference : ℝ := distance / revolutions in
  circumference / π

theorem diameter_of_wheel_is_approx_0_66 :
  abs (diameter_of_wheel - 0.66) < 0.01 :=
by
  sorry

end diameter_of_wheel_is_approx_0_66_l777_777726


namespace total_coins_l777_777998

def piles_of_quarters := 5
def piles_of_dimes := 5
def coins_per_pile := 3

theorem total_coins (piles_of_quarters piles_of_dimes coins_per_pile : ℕ) : 
  piles_of_quarters = 5 ∧ piles_of_dimes = 5 ∧ coins_per_pile = 3 →
  (piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 30) :=
by
  intro h
  cases h with hq hd
  rw [hq, hd.2.1, hd.2.2]
  sorry

#print axioms total_coins

end total_coins_l777_777998


namespace all_stones_equal_weight_l777_777647

variable {α : Type}
variable (x : Fin 13 → ℕ)

-- Define the condition that any 12 stones can be balanced
def balance_property (x : Fin 13 → ℕ) : Prop :=
  ∀ (i : Fin 13), 
    ∃ (s1 s2 : Finset (Fin 13)), 
      (s1.card = 6 ∧ s2.card = 6 ∧ s1 ∪ s2 = (Finset.univ.erase i) ∧ 
      (∑ j in s1, x j) = ∑ j in s2, x j)

theorem all_stones_equal_weight (h : balance_property x) :
  ∀ i j : Fin 13, x i = x j :=
sorry

end all_stones_equal_weight_l777_777647


namespace f_is_function_g_is_not_function_h_is_not_function_l777_777871

-- Define the sets A and B
def A : Set := {a, b, c}
def B : Set := {x, y, z}

-- Define the mappings f, g, and h
def f : a → B := {a ↦ y, b ↦ z, c ↦ x}
def g : a → B := {a ↦ y, c ↦ x}
def h : a → B := {a ↦ y, b ↦ z, c ↦ x, c ↦ z}

-- Proof statements
theorem f_is_function : Function f :=
  sorry

theorem g_is_not_function : ¬Function g :=
  sorry

theorem h_is_not_function : ¬Function h :=
  sorry

end f_is_function_g_is_not_function_h_is_not_function_l777_777871


namespace muffins_per_batch_is_six_l777_777809

def batches_per_month (total_batches : ℕ) (months : ℕ) : ℕ :=
  total_batches / months

def muffins_per_batch (total_students : ℕ) (batches_per_month : ℕ) : ℕ :=
  total_students / batches_per_month

theorem muffins_per_batch_is_six (total_students : ℕ) (total_batches : ℕ) (months : ℕ) :
  total_students = 24 ∧ total_batches = 36 ∧ months = 9 →
  muffins_per_batch total_students (batches_per_month total_batches months) = 6 :=
by 
  intros h,
  rcases h with ⟨hs, hb, hm⟩,
  simp [hs, hb, hm],
  unfold batches_per_month muffins_per_batch,
  simp,
  sorry

end muffins_per_batch_is_six_l777_777809


namespace queen_middle_school_teachers_l777_777609

theorem queen_middle_school_teachers
  (students : ℕ) 
  (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 6)
  (h_classes_per_teacher : classes_per_teacher = 5)
  (h_students_per_class : students_per_class = 25) : 
  (students * classes_per_student / students_per_class) / classes_per_teacher = 72 :=
by
  sorry

end queen_middle_school_teachers_l777_777609


namespace sum_of_coordinates_l777_777169

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (x : ℝ) : ℝ := sorry

theorem sum_of_coordinates (h1 : (2, 5) ∈ (λ x, 2 * f x '' set.univ))
    (h2 : f 2 = 5 / 2)
    (h3 : f_inv (5 / 2) = 2) :
  (5 / 2) + (f_inv (5 / 2) / 3) = 19 / 6 :=
by
  have h4 : 5 = 2 * f 2 := sorry
  have h5 : f 2 = 5 / 2 := sorry
  have h6 : f_inv (5 / 2) = 2 := sorry
  have h7 : (5 / 2) + (2 / 3) = 19 / 6 := sorry
  exact h7

end sum_of_coordinates_l777_777169


namespace no_such_hexagon_exists_l777_777321

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777321


namespace ratio_Sprime_S_eq_one_fourth_l777_777846

variables {A B C D E F I1 I2 I3 : Type} [EuclideanGeometry A B C D E F I1 I2 I3]

-- Given conditions as definitions
def S : ℝ := sorry                  -- Area of triangle ABC
def D_midpoint : midpoint B C D := sorry
def E_midpoint : midpoint C A E := sorry
def F_midpoint : midpoint A B F := sorry
def I1_incenter : incenter A E F I1 := sorry
def I2_incenter : incenter B F D I2 := sorry
def I3_incenter : incenter C D E I3 := sorry

-- Theorem to prove the ratio
theorem ratio_Sprime_S_eq_one_fourth :
  let Sprime := area_triangle I1 I2 I3 in
  Sprime / S = 1 / 4 :=
sorry

end ratio_Sprime_S_eq_one_fourth_l777_777846


namespace minimize_sum_distances_l777_777453

variables (Q : ℕ → ℝ) (Q_pos : ∀ n, Q (n+1) > Q n)

theorem minimize_sum_distances :
  ∃ Qopt, (Qopt = Q 5) ∧ 
  ∀ Q', (Σ (i : ℕ) in (finset.range 9), abs (Q Q' - Q i)) ≥ (Σ (i : ℕ) in (finset.range 9), abs (Q 5 - Q i)) :=
sorry

end minimize_sum_distances_l777_777453


namespace find_m_l777_777019

noncomputable def slope (l : ℝ → ℝ) : ℝ := sorry

def line1 := λ x : ℝ, 4 * x + 1
def line2 := λ x : ℝ, x
def line3 (m : ℝ) := λ x : ℝ, (2 * x - 3) / m

def symmetric_slope (slope : ℝ) (axis_slope : ℝ) := -((slope - axis_slope) / (1 + slope * axis_slope))

theorem find_m (m : ℝ) :
  symmetric_slope (-4) 1 * slope (line3 m) = -1 → 
  m = 1 / 2 :=
by
  sorry

end find_m_l777_777019


namespace vector_coordinates_l777_777022

theorem vector_coordinates :
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1) :=
by
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  show (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1)
  sorry

end vector_coordinates_l777_777022


namespace range_of_a_l777_777127

def symmetric_about_two (f : ℝ → ℝ) : Prop := ∀ x, f (2 + x) = f (2 - x)

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem range_of_a 
  (f : ℝ → ℝ)
  (h1 : symmetric_about_two f)
  (h2 : decreasing_on_interval f 2 (real.of_nat 2))
  (h3 : f(0) = f(4)) :
  ∀ a, f a ≥ f 0 → (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l777_777127


namespace volume_of_open_box_l777_777245

-- Definitions based on the conditions
def metallic_sheet_length : ℝ := 48
def metallic_sheet_width : ℝ := 36
def square_side_length : ℝ := 7

-- Calculation of the box dimensions
def new_length : ℝ := metallic_sheet_length - 2 * square_side_length
def new_width : ℝ := metallic_sheet_width - 2 * square_side_length
def height : ℝ := square_side_length

-- The main theorem we want to prove
theorem volume_of_open_box : new_length * new_width * height = 5244 := by
  sorry

end volume_of_open_box_l777_777245


namespace sum_of_arithmetic_progression_l777_777150

theorem sum_of_arithmetic_progression (n : ℕ) : 
  let row (k : ℕ) := (k + 1) * (2k + 1) in
  ∑ i in range (2 * n + 1), row i = (2 * n - 1) ^ 2 :=
sorry

end sum_of_arithmetic_progression_l777_777150


namespace no_such_convex_hexagon_and_point_exists_l777_777370

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777370


namespace hyperbola_equation_l777_777628

theorem hyperbola_equation : 
  (∃ a b : ℝ, 
    (focus1 (Ellipse (8:ℝ) (4:ℝ)) = (± 2, 0) ∧ focus2 (Ellipse (8:ℝ) (4:ℝ)) = (± 2, 0)) ∧
    asymptote (Hyperbola a b) = (λ x : ℝ, sqrt 3 * x) ∧
    hyperbola_eq := x^2 - (y^2 / 3) = 1) :=
sorry

end hyperbola_equation_l777_777628


namespace total_length_of_lines_in_S_l777_777986

def S : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | let x := p.1, y := p.2 in 
               abs(abs(abs(x)) - 3) - 1 + abs(abs(abs(y)) - 3) - 1 = 2}

theorem total_length_of_lines_in_S : 
  let total_length := 16 * real.sqrt 2 in
  ∑ (p ∈ S), 1 = total_length := 
sorry

end total_length_of_lines_in_S_l777_777986


namespace EF_side_length_l777_777086

def square_side_length (n : ℝ) : Prop := n = 10

def distance_parallel_line (d : ℝ) : Prop := d = 6.5

def area_difference (a : ℝ) : Prop := a = 13.8

theorem EF_side_length :
  ∃ (x : ℝ), square_side_length 10 ∧ distance_parallel_line 6.5 ∧ area_difference 13.8 ∧ x = 5.4 :=
sorry

end EF_side_length_l777_777086


namespace probability_same_length_segments_l777_777936

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777936


namespace right_angled_triangles_count_l777_777624

/-- Suppose a solid with specific three views. The problem is to determine the number of right-angled triangles in the solid. -/
def solid_with_views : Type := sorry  -- As the problem's specific graph & properties are not formally specified

axiom solid_conditions : solid_with_views

theorem right_angled_triangles_count (s : solid_with_views) : 
  count_right_angled_triangles s = 3 := 
begin
  sorry  -- We'll skip the proof for this example
end

end right_angled_triangles_count_l777_777624


namespace toll_for_18_wheel_truck_l777_777224

-- Define the total number of wheels, wheels on the front axle, 
-- and wheels on each of the other axles.
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4

-- Define the formula for calculating the toll.
def toll_formula (x : ℕ) : ℝ := 2.50 + 0.50 * (x - 2)

-- Calculate the number of other axles.
def calc_other_axles (wheels_left : ℕ) (wheels_per_axle : ℕ) : ℕ :=
wheels_left / wheels_per_axle

-- Statement to prove the final toll is $4.00.
theorem toll_for_18_wheel_truck : toll_formula (
  1 + calc_other_axles (total_wheels - front_axle_wheels) other_axle_wheels
) = 4.00 :=
by sorry

end toll_for_18_wheel_truck_l777_777224


namespace relationship_among_abc_l777_777113

noncomputable def a := Real.log 3 / Real.log 7
noncomputable def b := Real.log 7 / Real.log (1/3)
noncomputable def c := 3 ^ 0.7

theorem relationship_among_abc : b < a ∧ a < c :=
by {
    sorry
}

end relationship_among_abc_l777_777113


namespace hallie_made_100_per_painting_l777_777484

-- Define conditions
def num_paintings : ℕ := 3
def total_money_made : ℕ := 300

-- Define the goal
def money_per_painting : ℕ := total_money_made / num_paintings

theorem hallie_made_100_per_painting :
  money_per_painting = 100 :=
sorry

end hallie_made_100_per_painting_l777_777484


namespace greatest_possible_n_l777_777052

theorem greatest_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 8100) : n ≤ 8 :=
by
  -- Intentionally left uncommented.
  sorry

end greatest_possible_n_l777_777052


namespace parking_problem_l777_777200

theorem parking_problem (n : ℕ) :
  ∃ (a : ℕ → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i j, i ≠ j → a i ≠ a j) → (n + 1) ^ (n - 1) :=
by
  sorry

end parking_problem_l777_777200


namespace count_four_digit_even_nonzero_digits_l777_777882

/-- The number of distinct four-digit positive integers with only even non-zero digits is 256. -/
theorem count_four_digit_even_nonzero_digits : 
  let digits := {2, 4, 6, 8}
  let four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d ∈ Int.toString n, d ∈ digits}
  4^4 = 256 := by
  sorry

end count_four_digit_even_nonzero_digits_l777_777882


namespace largest_option_B_l777_777706

noncomputable def A := Real.sqrt (Real.cbrt 56)
noncomputable def B := Real.sqrt (Real.cbrt 3584)
noncomputable def C := Real.sqrt (Real.cbrt 2744)
noncomputable def D := Real.sqrt (Real.cbrt 392)
noncomputable def E := Real.sqrt (Real.cbrt 448)

theorem largest_option_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_option_B_l777_777706


namespace min_value_expr_l777_777112

theorem min_value_expr (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (∃ (a b c : ℝ), a + b + c = 3 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + sqrt (a * b) + real.cbrt (a * b * c) = 3)  :=
sorry

end min_value_expr_l777_777112


namespace cone_height_correct_l777_777468

noncomputable def cone_height (radius : ℝ) (central_angle : ℝ) : ℝ := 
  let base_radius := (central_angle * radius) / (2 * Real.pi)
  let height := Real.sqrt (radius ^ 2 - base_radius ^ 2)
  height

theorem cone_height_correct:
  cone_height 3 (2 * Real.pi / 3) = 2 * Real.sqrt 2 := 
by
  sorry

end cone_height_correct_l777_777468


namespace no_such_hexagon_exists_l777_777330

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l777_777330


namespace complex_mod_z_l777_777579

def complex_mod (z : ℂ) : ℝ := complex.abs z

def z : ℂ := (1 - complex.i) / (1 + complex.i) + 2 * complex.i

theorem complex_mod_z : complex_mod z = 1 :=
by sorry

end complex_mod_z_l777_777579


namespace sunglasses_count_l777_777595

theorem sunglasses_count (total_people : ℕ)
  (women_percentage : ℚ) (men_percentage : ℚ)
  (women_sunglasses_percentage : ℚ) (men_sunglasses_percentage : ℚ)
  (children_sunglasses_percentage : ℚ)
  (total_women total_men total_children women_wearing_sunglasses men_wearing_sunglasses children_wearing_sunglasses : ℕ) :
  total_people = 3000 →
  women_percentage = 2 / 5 →
  men_percentage = 1 / 3 →
  women_sunglasses_percentage = 0.25 →
  men_sunglasses_percentage = 0.15 →
  children_sunglasses_percentage = 0.10 →
  total_women = women_percentage * total_people →
  total_men = men_percentage * total_people →
  total_children = total_people - (total_women + total_men) →
  total_women = 1200 →
  total_men = 1000 →
  total_children = 800 →
  women_wearing_sunglasses = women_sunglasses_percentage * total_women →
  men_wearing_sunglasses = men_sunglasses_percentage * total_men →
  children_wearing_sunglasses = children_sunglasses_percentage * total_children →
  (women_wearing_sunglasses + men_wearing_sunglasses + children_wearing_sunglasses) = 530 :=
begin
  sorry
end

end sunglasses_count_l777_777595


namespace find_number_l777_777212

theorem find_number (x : ℕ) (h : (9 * x) / 3 = 27) : x = 9 :=
by
  sorry

end find_number_l777_777212


namespace polynomial_evaluation_bound_l777_777012

theorem polynomial_evaluation_bound {n : ℕ} (a : Fin (n-1) → ℝ) 
  (h : ∀ i, 0 ≤ a i) :
  ∀ (P : ℝ → ℝ), P 2 ≥ 3^n :=
begin
  -- Definitions needed to match the problem conditions
  let P := λ x, x^n + (Finset.univ.sum (λ i : Fin (n-1), a i * x^(n - 1 - i))) + 1,
  have P_has_n_roots : ∃ α : Fin n → ℝ, (∀ i, P (α i) = 0) := sorry,
  -- Our conclusion
  exact sorry,
end

end polynomial_evaluation_bound_l777_777012


namespace class_performance_proof_l777_777731

-- Define the performance records
def performance_records : List Int := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]

-- Define the conditions
def standard_jumps : Int := 160
def points_above : Int := 1
def points_below : Float := 0.5
def reward_threshold : Float := 65.0

-- Theorem stating the proof problem
theorem class_performance_proof :
  let best_performance := performance_records.maximum?.getOrElse 0;
  let worst_performance := performance_records.minimum?.getOrElse 0;
  let difference := best_performance - worst_performance;
  let total_adjustments := performance_records.sum;
  let average_jumps := standard_jumps + total_adjustments / List.length performance_records;
  let points_above_total := performance_records.filter(λ x => x > 0).sum;
  let points_below_total := (performance_records.filter(λ x => x < 0).map(λ x => Float.ofInt (-x)).sum) * points_below;
  let total_score := Float.ofInt points_above_total - points_below_total;
  difference = 27 ∧ average_jumps = standard_jumps + 6 ∧ total_score > reward_threshold := 
by
  -- Placeholder for the actual proof.
  sorry

end class_performance_proof_l777_777731


namespace range_of_a_l777_777892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x^3 - a*x)

theorem range_of_a {a : ℝ} (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∀ x ∈ Ioo (-1/2) (0 : ℝ), ∀ y ∈ Ioo x 0, f a y > f a x) ↔ a ∈ set.Ico (3 / 4) 1 :=
sorry

end range_of_a_l777_777892


namespace coordinates_of_C_prime_l777_777910

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end coordinates_of_C_prime_l777_777910


namespace regular_hexagon_same_length_probability_l777_777970

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777970


namespace increase_in_area_correct_l777_777723

-- Define the dimensions of the original rectangular garden
def length_rect := 60
def width_rect := 20

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Calculate the side length of the square garden using the same perimeter.
def side_square := perimeter_rect / 4

-- Define the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Define the area of the square garden
def area_square := side_square * side_square

-- Define the increase in area after reshaping
def increase_in_area := area_square - area_rect

-- Prove that the increase in the area is 400 square feet
theorem increase_in_area_correct : increase_in_area = 400 := by
  -- The proof is omitted
  sorry

end increase_in_area_correct_l777_777723


namespace find_a_if_f_even_l777_777041

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (Real.exp (3 * x) + 1) + a * x

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Main theorem statement
theorem find_a_if_f_even :
  ∀ a : ℝ, is_even_function (f a) → a = -3 / 2 := by
  sorry

end find_a_if_f_even_l777_777041


namespace no_such_hexagon_exists_l777_777353

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777353


namespace B_share_is_correct_l777_777762

variable (x : ℝ) -- Investment of B

-- Investments based on conditions
def A_investment := 3 * x
def B_investment := x
def C_investment := (3 / 2) * x
def D_investment := (1 / 2) * x

-- Capital months calculations
def A_capital_months := A_investment * 9
def B_capital_months := B_investment * 6
def C_capital_months := C_investment * 4
def D_capital_months := D_investment * 12

-- Total capital months
def total_capital_months := A_capital_months + B_capital_months + C_capital_months + D_capital_months

-- Total profit
def total_profit := 5280

-- Computing B's share
def B_share := (B_capital_months / total_capital_months) * total_profit

-- Theorem to prove B's share is Rs. 704
theorem B_share_is_correct : B_share = 704 := by
  -- Proof is omitted
  sorry

end B_share_is_correct_l777_777762


namespace prime_count_in_list_l777_777785

-- Define the sequence given in the problem
def a (n : ℕ) : ℕ :=
  47 * (10^(2*n) + 10^n + 1)

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Prove the statement that only one number in the sequence is prime
theorem prime_count_in_list : (finset.filter is_prime (finset.range 100).map a).card = 1 :=
by
  sorry

end prime_count_in_list_l777_777785


namespace no_such_convex_hexagon_exists_l777_777381

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777381


namespace gold_purity_l777_777637

theorem gold_purity : 
  (∀ (K : ℝ), (K = 24) ↔ (K = 12 + 12) ∧ (50% purity) = 12 ∧ (100% purity) = 24) 
  -> ∃ choice, choice = "D" :=
begin
  sorry
end

end gold_purity_l777_777637


namespace quadrilateral_ratio_l777_777607

theorem quadrilateral_ratio {R : ℝ} (r : ℝ) (hR : R = π * r^2) 
  (AC_is_diameter : ∃ (A C : ℝ), A ≠ C ∧ AC = 2 * r) 
  (angle_DAC_15 : m ∠ DAC = 15) 
  (angle_BAC_45 : m ∠ BAC = 45) : 
  let ABCD_area_ratio := (3 : ℝ) / (2 * π) in
  ∃ (a b c : ℕ), a = 3 ∧ b = 0 ∧ c = 2 ∧ a + b + c = 5 :=
by 
  sorry

end quadrilateral_ratio_l777_777607


namespace intersection_property_l777_777457

variable (A B : Set ℝ)
def A := {x : ℝ | x < 3}
def B := {x : ℝ | x > 1}

theorem intersection_property : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
  by sorry

end intersection_property_l777_777457


namespace time_interval_for_birth_and_death_rates_l777_777528

-- Define the parameters and assumptions
def average_birth_rate : ℕ := 7
def average_death_rate : ℕ := 3
def net_population_increase_per_day : ℕ := 172800
def seconds_per_day : ℕ := 86400

-- Define the time interval in days and prove it is approximately 2 seconds
theorem time_interval_for_birth_and_death_rates :
  let net_increase_per_interval := average_birth_rate - average_death_rate
  let T_in_days := (1 : ℝ) / (net_population_increase_per_day / net_increase_per_interval)
  let T_in_seconds := T_in_days * (seconds_per_day : ℝ)
  T_in_seconds ≈ 2 :=
by
  let net_increase_per_interval := average_birth_rate - average_death_rate
  let T_in_days := (1 : ℝ) / (net_population_increase_per_day / net_increase_per_interval)
  let T_in_seconds := T_in_days * (seconds_per_day : ℝ)
  -- Skipping the proof steps
  sorry

end time_interval_for_birth_and_death_rates_l777_777528


namespace GPA_of_rest_of_classroom_l777_777627

variable (n : ℕ) (x : ℝ)
variable (H1 : ∀ n, n > 0)
variable (H2 : (15 * n + 2 * n * x) / (3 * n) = 17)

theorem GPA_of_rest_of_classroom (n : ℕ) (H1 : ∀ n, n > 0) (H2 : (15 * n + 2 * n * x) / (3 * n) = 17) : x = 18 := by
  sorry

end GPA_of_rest_of_classroom_l777_777627


namespace bees_on_second_day_l777_777592

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end bees_on_second_day_l777_777592


namespace neg_one_power_zero_l777_777267

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l777_777267


namespace regular_hexagon_same_length_probability_l777_777973

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777973


namespace circles_ordered_by_radius_l777_777281

def circle_X_radius : ℝ := 2 * Real.pi
def circle_Y_circumference : ℝ := 12 * Real.pi
def circle_Z_area : ℝ := 16 * Real.pi
def half_circle_W_area : ℝ := 8 * Real.pi

noncomputable def circle_Y_radius : ℝ := circle_Y_circumference / (2 * Real.pi)
noncomputable def circle_Z_radius : ℝ := Real.sqrt (circle_Z_area / Real.pi)
noncomputable def half_circle_W_radius : ℝ := Real.sqrt ((2 * half_circle_W_area) / Real.pi)

theorem circles_ordered_by_radius :
  let r_X := circle_X_radius;
  let r_Y := circle_Y_radius;
  let r_Z := circle_Z_radius;
  let r_W := half_circle_W_radius;
  [r_Z, r_W, r_Y, r_X].sort <= [r_Z, r_W, r_Y, r_X] :=
by
  sorry

end circles_ordered_by_radius_l777_777281


namespace monic_cubic_polynomial_l777_777408

-- Definition of the given conditions
def α : ℝ := real.cbrt 2 + 1

-- The polynomial P(x)
def P (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x - 3

-- Statement we need to prove
theorem monic_cubic_polynomial : P(α) = 0 :=
sorry

end monic_cubic_polynomial_l777_777408


namespace distinct_four_digit_even_integers_count_l777_777879

theorem distinct_four_digit_even_integers_count : 
  ∃ n : ℕ, n = 256 ∧ 
  (∀ thousands hundreds tens units : ℕ, 
    thousands ∈ {2, 4, 6, 8} ∧ 
    hundreds ∈ {2, 4, 6, 8} ∧ 
    tens ∈ {2, 4, 6, 8} ∧ 
    units ∈ {2, 4, 6, 8} → 
    n = 4^4) := 
by
  sorry

end distinct_four_digit_even_integers_count_l777_777879


namespace households_with_only_bike_l777_777220

theorem households_with_only_bike
  (N : ℕ) (H_no_car_or_bike : ℕ) (H_car_bike : ℕ) (H_car : ℕ)
  (hN : N = 90)
  (h_no_car_or_bike : H_no_car_or_bike = 11)
  (h_car_bike : H_car_bike = 16)
  (h_car : H_car = 44) :
  ∃ (H_bike_only : ℕ), H_bike_only = 35 :=
by {
  sorry
}

end households_with_only_bike_l777_777220


namespace sum_of_roots_of_quadratic_l777_777700

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l777_777700


namespace student_scores_correct_answers_l777_777072

variable (c w : ℕ)

theorem student_scores_correct_answers :
  (c + w = 60) ∧ (4 * c - w = 130) → c = 38 :=
by
  intro h
  sorry

end student_scores_correct_answers_l777_777072


namespace Karlee_initial_grapes_l777_777095

theorem Karlee_initial_grapes (G S Remaining_Fruits : ℕ)
  (h1 : S = (3 * G) / 5)
  (h2 : Remaining_Fruits = 96)
  (h3 : Remaining_Fruits = (3 * G) / 5 + (9 * G) / 25) :
  G = 100 := by
  -- add proof here
  sorry

end Karlee_initial_grapes_l777_777095


namespace rainfall_comparison_l777_777142

-- Define the conditions
def rainfall_mondays (n_mondays : ℕ) (rain_monday : ℝ) : ℝ :=
  n_mondays * rain_monday

def rainfall_tuesdays (n_tuesdays : ℕ) (rain_tuesday : ℝ) : ℝ :=
  n_tuesdays * rain_tuesday

def rainfall_difference (total_monday : ℝ) (total_tuesday : ℝ) : ℝ :=
  total_tuesday - total_monday

-- The proof statement
theorem rainfall_comparison :
  rainfall_difference (rainfall_mondays 13 1.75) (rainfall_tuesdays 16 2.65) = 19.65 := by
  sorry

end rainfall_comparison_l777_777142


namespace general_term_a_n_sum_of_first_n_terms_b_n_to_S_n_l777_777849

-- Given conditions
variables {a : ℕ → ℚ}
hypothesis h1 : ∀ n : ℕ, n ≥ 1 → ∃ d : ℚ, (1 / a (n+1)) - (1 / a n) = d
hypothesis h2 : a 3 = 1 / 5
hypothesis h3 : a 2 = 3 * a 5

-- To prove
theorem general_term_a_n :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 * n - 1) :=
by
  sorry

theorem sum_of_first_n_terms_b_n_to_S_n :
  ∀ n : ℕ, n ≥ 1 → 
  let b := λ n, a n * a (n + 1),
  S := λ n, ∑ k in finset.range n, b k 
  in S n = n / (2 * n + 1) :=
by
  sorry

end general_term_a_n_sum_of_first_n_terms_b_n_to_S_n_l777_777849


namespace no_such_hexagon_exists_l777_777305

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l777_777305


namespace equation_of_tangent_circle_l777_777412

noncomputable def circle_eq_tangent_to_line (A B C : ℝ) (x1 y1 : ℝ) (line : ℝ -> ℝ -> Prop) :=
  let d := |A * x1 + B * y1 + C| / real.sqrt (A^2 + B^2) in
  let center_x := -1 in
  let center_y := 3 in
  let radius := 2 * real.sqrt 2 in
  (center_x + 1)^2 + (center_y - 3)^2 = 8

theorem equation_of_tangent_circle : 
  circle_eq_tangent_to_line (1:ℝ) (-1:ℝ) (0:ℝ) (-1:ℝ) (3:ℝ) (λ x y, x - y = 0) := 
  sorry

end equation_of_tangent_circle_l777_777412


namespace range_of_x_l777_777446

theorem range_of_x (a : ℕ → ℝ) (x : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_condition : ∀ n, a (n + 1)^2 + a n^2 < (5 / 2) * a (n + 1) * a n)
  (h_a2 : a 2 = 3 / 2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) : 2 < x ∧ x < 3 := by
  sorry

end range_of_x_l777_777446


namespace total_cakes_correct_l777_777724

-- Define the initial number of full-size cakes
def initial_cakes : ℕ := 350

-- Define the number of additional full-size cakes made
def additional_cakes : ℕ := 125

-- Define the number of half-cakes made
def half_cakes : ℕ := 75

-- Convert half-cakes to full-size cakes, considering only whole cakes
def half_to_full_cakes := (half_cakes / 2)

-- Total full-size cakes calculation
def total_cakes :=
  initial_cakes + additional_cakes + half_to_full_cakes

-- Prove the total number of full-size cakes
theorem total_cakes_correct : total_cakes = 512 :=
by
  -- Skip the proof
  sorry

end total_cakes_correct_l777_777724


namespace probability_of_same_length_l777_777966

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l777_777966


namespace determine_polynomial_q_l777_777168

noncomputable def q (x : ℝ) : ℝ := x^4 - 19.35*x^3 + 564.88*x^2 - 225.41*x - 150

-- Theorem stating that q(x) is the unique monic quartic polynomial satisfying the conditions
theorem determine_polynomial_q : 
  (∃ q : ℝ → ℝ, 
    (∀ x : ℝ, q(x) = x^4 - 19.35*x^3 + 564.88*x^2 - 225.41*x - 150) ∧
     q.is_monic ∧
     q.root (5 - 3*complex.i) ∧ 
     q.evaluate 0 = -150) := 
begin 
  use q,
  split, {
    intros x,
    refl,
  },
  split, {
    sorry -- Proof that q is monic
  },
  split, {
    sorry -- Proof that 5-3i and its conjugate are roots
  }, {
    rw q,
    simp,
    norm_num,
  },
end

end determine_polynomial_q_l777_777168


namespace proof_f_f_f_3_l777_777557

def f (n : ℤ) : ℤ :=
  if n < 5
  then n^2 + 1
  else 2 * n - 3

theorem proof_f_f_f_3 :
  f (f (f 3)) = 31 :=
by 
  -- Here, we skip the proof as instructed
  sorry

end proof_f_f_f_3_l777_777557


namespace average_speed_of_car_l777_777223

theorem average_speed_of_car (d1 d2 t1 t2 : ℝ) (h1 : d1 = 95) (h2 : d2 = 60) (h3 : t1 = 1) (h4 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 77.5 :=
by
  have total_distance : d1 + d2 = 155 :=
    by rw [h1, h2]; norm_num,
  have total_time : t1 + t2 = 2 :=
    by rw [h3, h4]; norm_num,
  rw [← total_distance, ← total_time],
  norm_num,
  sorry

end average_speed_of_car_l777_777223


namespace find_b_value_l777_777510

theorem find_b_value (x y z : ℝ) (u t : ℕ) (h_pos_xyx : x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ t > 0)
  (h1 : (x + y - z) / z = 1) (h2 : (x - y + z) / y = 1) (h3 : (-x + y + z) / x = 1) 
  (ha : (x + y) * (y + z) * (z + x) / (x * y * z) = 8) (hu_t : u + t + u * t = 34) : (u + t = 10) :=
by
  sorry

end find_b_value_l777_777510


namespace no_such_hexagon_exists_l777_777358

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777358


namespace hexagon_probability_same_length_l777_777981

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l777_777981


namespace sum_reciprocal_di_ge_one_l777_777433

noncomputable def di (P : ℕ → Complex) (n i : ℕ) : ℂ :=
  (∏ j in (Finset.filter (λ j, j ≠ i) (Finset.range n)), abs (P j - P i))

theorem sum_reciprocal_di_ge_one (P : ℕ → ℂ) (n : ℕ) (h_unit_circle : ∀ i, i < n → abs (P i) = 1) (h_distinct : Function.Injective P) :
  (∑ i in (Finset.range n), (1 / di P n i)) ≥ 1 :=
sorry

end sum_reciprocal_di_ge_one_l777_777433


namespace A_alone_approx_30_l777_777729

noncomputable def workTogetherInDays : ℝ := 19.411764705882355
noncomputable def workByBInDays : ℝ := 55

noncomputable def workAlone (A : ℝ) : ℝ :=
  1 / A + 1 / workByBInDays
   
noncomputable def canDoWorkAloneIn (A : ℝ) : Prop :=
  workAlone A = 1 / workTogetherInDays

theorem A_alone_approx_30 :
  ∃ (A : ℝ), A ≈ 30 ∧ canDoWorkAloneIn A :=
begin
  sorry
end

end A_alone_approx_30_l777_777729


namespace probability_same_length_segments_l777_777940

theorem probability_same_length_segments (T : Finset ℕ) (sides : Finset ℕ) (diagonals : Finset ℕ) :
  sides.card = 6 →
  diagonals.card = 9 →
  (∃ d1 d2 : Finset ℕ, d1.card = 3 ∧ d2.card = 3 ∧ d1 ∩ d2 = ∅ ∧ diagonals = d1 ∪ d2) →
  T = sides ∪ diagonals →
  (∃ n : ℕ, (T.card.choose 2) = n * 5 ∧ ∃ f : ℕ, (∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0) = f * 5) →
  ((∑ x in T, ∑ y in (T \ {x}), if x = y then 1 else 0).toRat / (T.card.choose 2).toRat = (1 : ℚ) / 5) :=
by
  sorry

end probability_same_length_segments_l777_777940


namespace kevin_cards_found_l777_777101

theorem kevin_cards_found : ∀ (initial_cards found_cards total_cards : Nat), 
  initial_cards = 7 → 
  total_cards = 54 → 
  total_cards - initial_cards = found_cards →
  found_cards = 47 :=
by
  intros initial_cards found_cards total_cards h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end kevin_cards_found_l777_777101


namespace problem_I_problem_II_l777_777126

open Set Real

-- Problem (I)
theorem problem_I (x : ℝ) :
  (|x - 2| ≥ 4 - |x - 1|) ↔ x ∈ Iic (-1/2) ∪ Ici (7/2) :=
by
  sorry

-- Problem (II)
theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 1/2/n = 1) :
  m + 2 * n ≥ 4 :=
by
  sorry

end problem_I_problem_II_l777_777126


namespace wheel_diameter_l777_777236

noncomputable def diameter_of_wheel (revolutions : ℝ) (distance : ℝ) : ℝ :=
  let circumference := distance / revolutions in
  circumference / Real.pi

theorem wheel_diameter : diameter_of_wheel 624.4536030972898 1000 ≈ 0.5097 := 
by
  sorry

end wheel_diameter_l777_777236


namespace no_convex_hexagon_with_point_M_l777_777313

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777313


namespace sandwiches_ordered_l777_777602

-- Define the cost per sandwich
def cost_per_sandwich : ℝ := 5

-- Define the delivery fee
def delivery_fee : ℝ := 20

-- Define the tip percentage
def tip_percentage : ℝ := 0.10

-- Define the total amount received
def total_received : ℝ := 121

-- Define the equation representing the total amount received
def total_equation (x : ℝ) : Prop :=
  cost_per_sandwich * x + delivery_fee + (cost_per_sandwich * x + delivery_fee) * tip_percentage = total_received

-- Define the theorem that needs to be proved
theorem sandwiches_ordered (x : ℝ) : total_equation x ↔ x = 18 :=
sorry

end sandwiches_ordered_l777_777602


namespace christmas_tree_problem_l777_777229

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end christmas_tree_problem_l777_777229


namespace no_such_convex_hexagon_and_point_exists_l777_777373

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777373


namespace no_such_convex_hexagon_exists_l777_777376

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777376


namespace isosceles_triangle_points_count_l777_777067

noncomputable def count_isosceles_points :
  ℕ :=
  let A := (3, 3)
  let B := (5, 3)
  let grid_points : List (ℕ × ℕ) :=
    List.join (List.map (λ x, List.map (λ y, (x, y)) (List.range 7)) (List.range 7))
  let is_perpendicular_bisector := grid_points.filter (λ P, P.1 = 4 ∧ P ≠ (4, 3))
  let is_2_units_away := grid_points.filter (λ P, (P = (3, 1)) ∨ (P = (3, 5)) ∨ (P = (5, 1)) ∨ (P = (5, 5)))
  is_perpendicular_bisector.length + is_2_units_away.length -- Expected to be 10

theorem isosceles_triangle_points_count :
  count_isosceles_points = 10 :=
by {
  sorry
}

end isosceles_triangle_points_count_l777_777067


namespace quadratic_has_minimum_value_positive_l777_777786

def quadratic_function (x : ℝ) : ℝ :=
  3 * x^2 - 9 * x + 6

theorem quadratic_has_minimum_value_positive :
  (∃ x : ℝ, quadratic_function x = quadratic_function (1.5) ∧ x = 1.5) :=
begin
  sorry
end

end quadratic_has_minimum_value_positive_l777_777786


namespace parking_lot_empty_first_time_at_11_30_l777_777239

open Nat

noncomputable def departure_time (S : ℕ) : ℝ := 6 * (S - 1)
noncomputable def entry_number (S : ℕ) : ℕ := S - 15
noncomputable def entry_time (y : ℕ) : ℝ := 8 * y + 3

theorem parking_lot_empty_first_time_at_11_30 (S x y : ℕ) (h : S > 55) :
  x = departure_time S ∧ S = y + 15 ∧ 8 * y > x - 3 → x = 330 :=
begin
  intros ⟨hx1, hx2, hx3⟩,
  have h4 : S = 56, from sorry,
  rw [hx1, departure_time] at hx1,
  norm_num at hx1,
  exact hx1
end

end parking_lot_empty_first_time_at_11_30_l777_777239


namespace no_such_convex_hexagon_exists_l777_777383

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777383


namespace density_and_convergence_l777_777577

-- Definitions of f, phi (varphi), and f_sigma
variables {f : ℝ → ℝ}
variables {phi : ℝ → ℂ}
variables {f_sigma : ℝ → ℝ → ℝ}

-- Define the conditions
def is_density_function (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ f x) ∧ (∫ x, f x = 1)

def characteristic_function (phi : ℝ → ℂ) (f : ℝ → ℝ) : Prop :=
  ∀ t, phi t = ∫ x, complex.exp (complex.I * t * x) * (f x)

def f_sigma_def (sigma : ℝ) (f : ℝ → ℝ) (phi : ℝ → ℂ) : ℝ → ℝ :=
  λ x, (1 / (2 * real.pi)) * ∫ (t : ℝ) in (set.Icc (-1/sigma) (1/sigma)), 
      complex.exp (-complex.I * t * x) * phi t * (1 - (sigma * |t|))

-- The two main claims
theorem density_and_convergence (sigma : ℝ) (hσ : 0 < sigma) :
  (is_density_function f ∧ characteristic_function phi f) →
  is_density_function (f_sigma_def sigma f phi) ∧
  (λ σ → ∫ x, abs (f x - (f_sigma_def σ f phi) x) ∂ measure_theory.measure_space.volume) → 0 :=
by
  intro hf 
  sorry

end density_and_convergence_l777_777577


namespace equivalent_representations_l777_777233

theorem equivalent_representations :
  (16 / 20 = 24 / 30) ∧
  (80 / 100 = 4 / 5) ∧
  (4 / 5 = 0.8) :=
by 
  sorry

end equivalent_representations_l777_777233


namespace max_elements_is_3_l777_777755

-- Define the condition for any two distinct elements x and y in the set
def condition (S : Set ℝ) : Prop :=
  ∀ {x y : ℝ}, x ∈ S → y ∈ S → x ≠ y → (x + y - 1)^2 = x * y + 1

-- Define the proposition that the maximum number of elements in this set is 3
theorem max_elements_is_3 (S : Set ℝ) (hS : condition S) : Finset.card S ≤ 3 :=
  sorry

end max_elements_is_3_l777_777755


namespace no_such_convex_hexagon_and_point_l777_777336

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777336


namespace arrangement_count_l777_777076

-- Define the conditions: Wilma, Paul, and Alice refuse to sit next to each other
def wilma := "Wilma"
def paul := "Paul"
def alice := "Alice"
def people := ["Person1", "Person2", "Person3", "Person4", "Person5", "Person6", "Person7", wilma, paul, alice]

-- Define the problem: finding the number of valid arrangements of people
def number_of_arrangements : ℕ :=
  9! - (3 * (8! * 2!) - 7! * 3!)

/-- The number of arrangements of 9 people such that Wilma, Paul, and Alice refuse to sit next to each other is 181200. -/
theorem arrangement_count : number_of_arrangements = 181200 :=
sorry

end arrangement_count_l777_777076


namespace shaded_area_eq_triangle_area_l777_777539

-- Define the geometric setup
variables {A B C : ℝ} -- vertices of the triangle
variable (h_right_triangle : A^2 + B^2 = C^2) -- A is the hypotenuse

-- Define the semicircle areas
noncomputable def semicircle_area (d : ℝ) : ℝ :=
  (1/2) * Real.pi * (d / 2)^2

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1/2) * a * b

-- Define the conditions
axiom ha : a + b = semicircle_area AB
axiom hc : c + d = semicircle_area BC
axiom hT : b + d + T = semicircle_area AC

-- Lean statement to prove the problem
theorem shaded_area_eq_triangle_area 
  (A B C : ℝ) (h_right_triangle : A^2 + B^2 = C^2) (a b c d T : ℝ) 
  (ha : a + b = semicircle_area A) 
  (hc : c + d = semicircle_area B) 
  (hT : b + d + T = semicircle_area C) : 
  a + c = T :=
by sorry

end shaded_area_eq_triangle_area_l777_777539


namespace largest_prime_factor_5292_l777_777679

theorem largest_prime_factor_5292 :
  ∃ p, nat.prime p ∧ p ∣ 5292 ∧ ∀ q, nat.prime q ∧ q ∣ 5292 → q ≤ p :=
sorry

end largest_prime_factor_5292_l777_777679


namespace solve_system_of_equations_l777_777166

theorem solve_system_of_equations (x y : ℝ) :
  (x^4 + (7/2) * x^2 * y + 2 * y^3 = 0) ∧
  (4 * x^2 + 7 * x * y + 2 * y^3 = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -1) ∨ (x = -11 / 2 ∧ y = -11 / 2) :=
sorry

end solve_system_of_equations_l777_777166


namespace problem_statement_l777_777435

theorem problem_statement 
  (n : ℕ) 
  (n_ge_3 : n ≥ 3) 
  (a : ℕ → ℝ) 
  (a_pos : ∀ i, 2 ≤ i → i ≤ n → 0 < a i) 
  (a_prod_eq_1 : ∏ i in Finset.Icc 2 n, a i = 1) :
  (∏ i in Finset.Icc 2 n, (1 + a i)^i) > n^n :=
sorry

end problem_statement_l777_777435


namespace at_least_one_not_less_than_one_third_l777_777124

theorem at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 :=
sorry

end at_least_one_not_less_than_one_third_l777_777124


namespace no_such_hexagon_exists_l777_777348

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l777_777348


namespace no_convex_hexagon_with_point_M_l777_777306

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l777_777306


namespace triangle_ABC_l777_777520

theorem triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hb : b = 3) (hcosB : real.cos B = 1/3)
  (hcos_rule : b^2 = a^2 + c^2 - 2*a*c*real.cos B) : c = 3 ∧ real.cos (A - C) = 23/27 := by {
  have hc : c = 3, from sorry,
  have hcosC : real.cos C = 1/3, from sorry,
  have hsinC : real.sin C = 2 * real.sqrt 2 / 3, from sorry,
  have hsinA : real.sin A = (4 * real.sqrt 2) / 9, from sorry,
  have hcosA : real.cos A = 7 / 9, from sorry,
  have hcosAC : real.cos (A - C) = (7 / 9) * (1 / 3) + ((4 * real.sqrt 2) / 9) * ((2 * real.sqrt 2) / 3), from sorry,
  split,
  exact hc,
  exact hcosAC,
}

end triangle_ABC_l777_777520


namespace ellipse_equation_with_m_l777_777540

theorem ellipse_equation_with_m (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m ∈ Set.Ioi 5 := 
sorry

end ellipse_equation_with_m_l777_777540


namespace prime_dates_in_2004_l777_777030

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0
  
def prime_days_in_february_2004 : ℕ := 
  if 2004 % 4 = 0 ∧ (2004 % 100 ≠ 0 ∨ 2004 % 400 = 0) then -- Leap year condition
    10 -- As calculated, February has 10 prime dates
  else 
    0 -- Not a leap year

def prime_days_in_march_2004 : ℕ := 11 -- Prime days in March
def prime_days_in_may_2004 : ℕ := 11 -- Prime days in May
def prime_days_in_july_2004 : ℕ := 11 -- Prime days in July
def prime_days_in_november_2004 : ℕ := 10 -- Prime days in November

def total_prime_dates_2004 : ℕ :=
  prime_days_in_february_2004 + prime_days_in_march_2004 + prime_days_in_may_2004 +
  prime_days_in_july_2004 + prime_days_in_november_2004

theorem prime_dates_in_2004 : total_prime_dates_2004 = 53 := by
  unfold total_prime_dates_2004
  unfold prime_days_in_february_2004
  unfold prime_days_in_march_2004
  unfold prime_days_in_may_2004
  unfold prime_days_in_july_2004
  unfold prime_days_in_november_2004
  simp -- To simplify the calculations
  sorry -- Proof to be filled


end prime_dates_in_2004_l777_777030


namespace probability_of_B_l777_777075

-- Define the events and their probabilities according to the problem description
def A₁ := "Event where a red ball is taken from bag A"
def A₂ := "Event where a white ball is taken from bag A"
def A₃ := "Event where a black ball is taken from bag A"
def B := "Event where a red ball is taken from bag B"

-- Types of bags A and B containing balls
structure Bag where
  red : Nat
  white : Nat
  black : Nat

-- Initial bags
def bagA : Bag := ⟨ 3, 2, 5 ⟩
def bagB : Bag := ⟨ 3, 3, 4 ⟩

-- Probabilities of each event in bagA
def P_A₁ : ℚ := 3 / 10
def P_A₂ : ℚ := 2 / 10
def P_A₃ : ℚ := 5 / 10

-- Probability of event B under conditions A₁, A₂, A₃
def P_B_given_A₁ : ℚ := 4 / 11
def P_B_given_A₂ : ℚ := 3 / 11
def P_B_given_A₃ : ℚ := 3 / 11

-- Goal: Prove that the probability of drawing a red ball from bag B (P(B)) is 3/10
theorem probability_of_B : 
  (P_A₁ * P_B_given_A₁ + P_A₂ * P_B_given_A₂ + P_A₃ * P_B_given_A₃) = (3 / 10) :=
by
  -- Placeholder for the proof
  sorry

end probability_of_B_l777_777075


namespace parallel_vectors_angle_l777_777482

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (1 / 2, Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

theorem parallel_vectors_angle (α : ℝ) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (vec_a α).1 = k * (vec_b α).1 ∧ (vec_a α).2 = k * (vec_b α).2) (h_acute : 0 < α ∧ α < π / 2) :
  α = π / 4 :=
sorry

end parallel_vectors_angle_l777_777482


namespace compute_expression_l777_777282

theorem compute_expression :
  (-9 * 5) - (-7 * -2) + (11 * -4) = -103 :=
by
  sorry

end compute_expression_l777_777282


namespace parallelogram_area_correct_l777_777710

-- Definitions of the conditions
def base : ℕ := 12
def height : ℕ := 18

-- Definition of the area function for a parallelogram
def parallelogram_area (b h : ℕ) : ℕ := b * h

-- Statement of the theorem
theorem parallelogram_area_correct : parallelogram_area base height = 216 := by
  -- This is where the proof would go
  sorry

end parallelogram_area_correct_l777_777710


namespace solution_correct_l777_777199

-- Definition of the dot configuration of a single cube
structure Cube :=
(faces : Fin 6 → Nat)
(h_face_dot : ∃! i : Fin 6, faces i = 3)
(h_two_faces_two : ∃ i j : Fin 6, i ≠ j ∧ faces i = 2 ∧ faces j = 2)
(h_three_faces_one : ∀ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k → faces i = 1 ∧ faces j = 1 ∧ faces k = 1)

-- Definition of the condition that two touching faces have the same number of dots
def faces_match_on_touch (c1 c2 : Cube) (f1 f2 : Fin 6) : Prop :=
c1.faces f1 = c2.faces f2

-- Definition of the configuration
structure Configuration :=
(cubes : Fin 7 → Cube)
(h_faces_touch : ∀ {i j : Fin 7} (f1 f2 : Fin 6), touching i j f1 f2 → faces_match_on_touch (cubes i) (cubes j) f1 f2)

-- touching is a placeholder definition indicating when two faces of different cubes touch each other.
constant touching : Fin 7 → Fin 7 → Fin 6 → Fin 6 → Prop

noncomputable def problem_statement : Prop :=
∃ (A B C : Nat),
  Configuration → A = 2 ∧ B = 2 ∧ C = 3

theorem solution_correct :
  problem_statement :=
by
  sorry

end solution_correct_l777_777199


namespace min_value_reciprocal_sum_l777_777116

theorem min_value_reciprocal_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 2) : 
  (∃ c : ℝ, c = (1/x) + (1/y) + (1/z) ∧ c ≥ 9/2) :=
by
  -- proof would go here
  sorry

end min_value_reciprocal_sum_l777_777116


namespace triangles_intersection_area_is_zero_l777_777060

-- Define the vertices of the two triangles
def vertex_triangle_1 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (0, 2)
| ⟨1, _⟩ => (2, 1)
| ⟨2, _⟩ => (0, 0)

def vertex_triangle_2 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (2, 2)
| ⟨1, _⟩ => (0, 1)
| ⟨2, _⟩ => (2, 0)

-- The area of the intersection of the two triangles
def area_intersection (v1 v2 : Fin 3 → (ℝ × ℝ)) : ℝ :=
  0

-- The theorem to prove
theorem triangles_intersection_area_is_zero :
  area_intersection vertex_triangle_1 vertex_triangle_2 = 0 :=
by
  -- Proof is omitted here.
  sorry

end triangles_intersection_area_is_zero_l777_777060


namespace collinear_a_b_l777_777021

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -2)

-- Definition of collinearity of vectors
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

-- Statement to prove
theorem collinear_a_b : collinear a b :=
by
  sorry

end collinear_a_b_l777_777021


namespace no_such_convex_hexagon_and_point_exists_l777_777368

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l777_777368


namespace alice_winning_strategy_l777_777764

theorem alice_winning_strategy (x y : ℝ) : 
  ∃ (strategy : (ℝ × ℝ) → Prop), 
  ∀ (card1 card2 : ℝ × ℝ), strategy card1 ∨ strategy card2 :=
by 
  -- Card Expressions
  let exprs := [x + y, x - y, x^2 + xy + y^2, x^2 - xy + y^2]

  -- Winning Strategy
  let strategy := λ (cards : ℝ × ℝ), 
    cards = (x - y, x^2 + xy + y^2) ∨ 
    cards = (x - y, x^2 - xy + y^2) ∨
    cards = (x + y, x^2 - xy + y^2) ∨
    cards = (x + y, x^2 + xy + y^2) ∨
    cards = (x^2 - xy + y^2, x^2 + xy + y^2)

  have strategy_exists : ∃ (s : (ℝ × ℝ) → Prop), 
    ∀ (c1 c2 : ℝ × ℝ), s c1 ∨ s c2,
  from Exists.intro strategy 
    (λ c1 c2, 
      -- Case Analysis
      cases c1 with a1 b1,
      cases c2 with a2 b2,
      -- Always choosing cards such that Alice wins according to the strategies described
      sorry),

  exact strategy_exists

end alice_winning_strategy_l777_777764


namespace postcards_initial_count_l777_777265

theorem postcards_initial_count (P : ℕ) 
  (h1 : ∀ n, n = P / 2)
  (h2 : ∀ n, n = (P / 2) * 15 / 5) 
  (h3 : P / 2 + 3 * P / 2 = 36) : 
  P = 18 := 
sorry

end postcards_initial_count_l777_777265


namespace elapsed_time_l777_777137

variable (totalDistance : ℕ) (runningSpeed : ℕ) (distanceRemaining : ℕ)

theorem elapsed_time (h1 : totalDistance = 120) (h2 : runningSpeed = 4) (h3 : distanceRemaining = 20) :
  (totalDistance - distanceRemaining) / runningSpeed = 25 := by
sorry

end elapsed_time_l777_777137


namespace centroid_area_condition_l777_777548

variables {α : Type*} [linear_ordered_field α]

noncomputable def centroid (A B C : euclidean_geometry.point α) : euclidean_geometry.point α :=
((A + B + C) / 3 : euclidean_geometry.point α)

theorem centroid_area_condition
  (A B C : euclidean_geometry.point α)
  (G : euclidean_geometry.point α := centroid A B C)
  (N : euclidean_geometry.point α)
  (P : euclidean_geometry.point α)
  (hN : euclidean_geometry.on_boundary N (euclidean_geometry.triangle A B C))
  (hP : euclidean_geometry.in_or_on_boundary P (euclidean_geometry.triangle A B C)) :
  euclidean_geometry.area_of_triangle G N P ≥
    euclidean_geometry.area_of_triangle A B C / 6 :=
sorry

end centroid_area_condition_l777_777548


namespace inverse_h_l777_777115

def f (x : ℝ) : ℝ := 5 * x + 6
def g (x : ℝ) : ℝ := 4 * x - 7
def h (x : ℝ) : ℝ := f (g (x))

theorem inverse_h (y : ℝ) : h⁻¹(y) = (y + 29) / 20 :=
by
  -- inverse function theorem: h⁻¹ is inverse of h
  sorry

end inverse_h_l777_777115


namespace no_such_convex_hexagon_exists_l777_777377

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777377


namespace find_y_for_projection_l777_777874

theorem find_y_for_projection :
  ∃ y : ℝ, let v := ![1, y, 4]
  let w := ![3, 2, 1]
  let proj_w_v := ( ( (v ⬝ w) / (w ⬝ w) ) • w )
  proj_w_v = ![2, 4 / 3, 2 / 3] ∧ y = 7 / 6 :=
by
  sorry

end find_y_for_projection_l777_777874


namespace total_cooking_and_movie_time_l777_777933

-- Definitions based on conditions
def first_movie_duration : ℝ := 1.5 -- hours
def second_movie_duration : ℝ := first_movie_duration + 0.5 -- 30 minutes longer in hours
def popcorn_time : ℝ := 10 / 60 -- converting minutes to hours
def fries_time : ℝ := 2 * popcorn_time -- twice the popcorn time in hours

-- Lean statement to prove the total time
theorem total_cooking_and_movie_time :
  first_movie_duration + second_movie_duration + popcorn_time + fries_time = 4 := 
by
  sorry

end total_cooking_and_movie_time_l777_777933


namespace problem1_problem2_l777_777902

-- Define the total number of balls for clarity
def total_red_balls : ℕ := 4
def total_white_balls : ℕ := 6
def total_balls_drawn : ℕ := 4

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := n.choose k

-- Problem 1: Prove that the number of ways to draw 4 balls that include both colors is 194
theorem problem1 :
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) +
  (binom total_red_balls 1 * binom total_white_balls 3) = 194 :=
  sorry

-- Problem 2: Prove that the number of ways to draw 4 balls where the number of red balls is at least the number of white balls is 115
theorem problem2 :
  (binom total_red_balls 4 * binom total_white_balls 0) +
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) = 115 :=
  sorry

end problem1_problem2_l777_777902


namespace find_x_l777_777418

theorem find_x (x : ℕ) (h1 : (31 : ℕ) ≤ 100) (h2 : (58 : ℕ) ≤ 100) (h3 : (98 : ℕ) ≤ 100) (h4 : 0 < x) (h5 : x ≤ 100)
               (h_mean_mode : ((31 + 58 + 98 + x + x) / 5 : ℚ) = 1.5 * x) : x = 34 :=
by
  sorry

end find_x_l777_777418


namespace tower_surface_area_l777_777796

theorem tower_surface_area : 
  let volumes := [1, 64, 27, 216, 125, 343, 8, 512] in
  let side_lengths := volumes.map (λ v => int.ofNat (nat.cbrt v)) in
  let surface_area s := 6 * s^2 in
  let bottom_surface_area := 6 in
  let adjusted_surface_areas := [80, 45, 180, 125, 245, 20, 320] in
  bottom_surface_area + adjusted_surface_areas.sum = 1021 :=
by sorry

end tower_surface_area_l777_777796


namespace smallest_m_integer_series_zero_l777_777120

noncomputable def b : ℝ := Real.pi / 1806

theorem smallest_m_integer_series_zero (m : ℕ) : 
  (2 * ∑ k in Finset.range (m+1), ((Real.cos ((k:ℝ)^2 * b)) * Real.sin ((k:ℝ) * b))) = (0 : ℝ) ↔ 
  m = 903 := 
begin 
  sorry -- the proof is not required
end

end smallest_m_integer_series_zero_l777_777120


namespace lines_no_common_point_parallel_or_skew_l777_777059

theorem lines_no_common_point_parallel_or_skew
  (L1 L2 : AffineSpace ℝ 3) :
  (∀ P, ¬(P ∈ L1 ∧ P ∈ L2)) → (are_parallel L1 L2 ∨ are_skew L1 L2) :=
by
  sorry

end lines_no_common_point_parallel_or_skew_l777_777059


namespace problem_C_properties_l777_777172

noncomputable def curve_C (x y k : ℝ) :=
  abs (x + 1) * abs (y - 1) = k ^ 2

theorem problem_C_properties (k : ℝ) (hk : k > 0) :
  ¬ curve_C (-1) 1 k ∧ 
  (∀ x y, curve_C x y k ↔ curve_C (-2 - x) (2 - y) k) ∧
  (∀ x y A B, curve_C x y k → 
      (| x + 1 | ≤ |A| ∧ | y - 1 | ≤ |B| → |A| + |B| ≥ 2k)) ∧
  (∀ x y, curve_C x y k → 
      let P0 := (x, y),
          P1 := (-2 - x, y),
          P2 := ((2 * -1) - (x), 2*1 - (y)),
          P3 := (x, (2 * -y)) in 
      2 * abs (x + 1) * 2 * abs (y - 1) = 4 * k ^ 2) := 
by
  sorry

end problem_C_properties_l777_777172


namespace algebraic_expression_problem_l777_777894

-- Define the conditions and the target statement to verify.
theorem algebraic_expression_problem (x : ℝ) 
  (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by 
  -- Add sorry to skip the proof.
  sorry

end algebraic_expression_problem_l777_777894


namespace no_such_hexagon_exists_l777_777365

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l777_777365


namespace find_norm_b_l777_777853

variables (a b : V)
variables (dot_product : a ⬝ b = -8)
variables (proj : (a ⬝ b) / ∥b∥ = -3 * Real.sqrt 2)

theorem find_norm_b : ∥b∥ = (4 * Real.sqrt 2) / 3 :=
  sorry

end find_norm_b_l777_777853


namespace hypotenuse_length_roots_l777_777058

theorem hypotenuse_length_roots (x y : ℝ) 
  (h_eq : 2 * x^2 - 8 * x + 7 = 0) 
  (h_eq' : 2 * y^2 - 8 * y + 7 = 0) 
  (h_triangle : x * y = 7 / 2 ∧ x + y = 4) :
  real.sqrt (x^2 + y^2) = 3 :=
by
  sorry

end hypotenuse_length_roots_l777_777058


namespace total_enemies_l777_777530

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end total_enemies_l777_777530


namespace solve_for_a_plus_b_l777_777852

theorem solve_for_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, (-1 < x ∧ x < 1 / 3) → ax^2 + bx + 1 > 0) →
  a * (-3) + b = -5 :=
by
  intro h
  -- Here we can use the proofs provided in the solution steps.
  sorry

end solve_for_a_plus_b_l777_777852


namespace no_such_convex_hexagon_exists_l777_777380

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l777_777380


namespace probability_of_same_length_segments_l777_777954

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777954


namespace roots_single_circle_or_line_l777_777147

open Complex

theorem roots_single_circle_or_line (a b c d : ℂ) (n : ℕ) (h : n > 0) :
  ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → 
  (∃ r : ℝ, ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → |z - b| = r ∨ ∃ m : ℝ, ∀ z : ℂ, a * (z - b)^n = c * (z - d)^n → z = m * (z - d)) :=
by
  sorry

end roots_single_circle_or_line_l777_777147


namespace gcd_factorial8_factorial10_factorial_8_calculation_l777_777806

/-- Definition of factorial -/
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

/-- The greatest common divisor of 8! and 10! is 8! (which equals 40320) -/
theorem gcd_factorial8_factorial10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
by sorry

/-- Calculation of 8! -/
theorem factorial_8_calculation : factorial 8 = 40320 :=
by sorry

end gcd_factorial8_factorial10_factorial_8_calculation_l777_777806


namespace value_of_expression_l777_777711

theorem value_of_expression : (85 + 32 / 113) * 113 = 9635 :=
by
  sorry

end value_of_expression_l777_777711


namespace sequence_property_l777_777443

def sequence (a : ℕ → ℕ) : Prop := 
  (a 1 = 1) ∧ 
  (a 2 = 5) ∧ 
  (a 3 = 6) ∧ 
  (∀ n, n ≥ 4 → a n = 3 * a (n-1) - a (n-2) - 2 * a (n-3))

theorem sequence_property (a : ℕ → ℕ) (h : sequence a) : 
  ∀ n, n ≥ 4 → a n > 3 * 2^(n-2) :=
sorry

end sequence_property_l777_777443


namespace length_of_rectangle_is_16_l777_777054

-- Given Conditions
def side_length_sq (a : ℝ) : ℝ := 8 -- Side length of the square
def width_rect (b : ℝ) : ℝ := 4 -- Width of the rectangle

-- Proof Problem Statement
theorem length_of_rectangle_is_16 :
  ∃ (L : ℝ), (side_length_sq 8 * side_length_sq 8 = width_rect 4 * L) ∧ L = 16 :=
begin
  sorry
end

end length_of_rectangle_is_16_l777_777054


namespace similarity_coordinates_C_l777_777909

theorem similarity_coordinates_C (A B C : ℝ × ℝ) (ratio : ℝ) :
  A = (1,2) ∧ B = (2,1) ∧ C = (3,2) ∧ ratio = 2 →
  (exists C' : ℝ × ℝ, (C' = (6,4)) ∨ (C' = (-6,-4))) :=
by { intro h, sorry }

end similarity_coordinates_C_l777_777909


namespace sum_seq_inverse_l777_777642

def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 75
  | k + 1 => (seq k)^2 - (seq k)

theorem sum_seq_inverse :
  ∑' n, (1 / (seq n - 1)) = 1 / 75 :=
by
  sorry

end sum_seq_inverse_l777_777642


namespace find_b_l777_777476

def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  3 * x^2 + b * x + c

theorem find_b (b c : ℝ) (h : tendsto (fun d => (f (b + d) b c - f b b c) / d) (𝓝 0) (𝓝 14)) :
  b = 2 :=
by
  unfortunately the completion of the proof is omitted

end find_b_l777_777476


namespace max_distance_l777_777994

theorem max_distance (z : ℂ) (hz : complex.abs z = 3) :
  ∃ d : ℝ, d = 1529.104 ∧ ∀ w : ℂ, w = (1 + 2*complex.I) * z^4 - z^6 → complex.abs w ≤ d :=
sorry

end max_distance_l777_777994


namespace part1_part2_l777_777580

variable (x a : ℝ) (p q r : Prop)

-- Definition of p, q, and r
def condition_p : Prop := x^2 - x - 2 ≤ 0
def condition_q : Prop := (x - 3) / x < 0
def condition_r : Prop := (x - (a + 1)) * (x + (2 * a - 1)) ≤ 0

-- Solution 1: Prove range of x if p ∧ q is true
theorem part1 (hp : condition_p x) (hq : condition_q x) : 0 < x ∧ x ≤ 2 := by
  sorry

-- Solution 2: Prove range of a if p is a sufficient but not necessary condition for r
theorem part2 (hsufficient : ∀ (x : ℝ), condition_p x → condition_r x) (hnecessary : ∃ (x : ℝ), condition_r x ∧ ¬condition_p x) : a > 1 := by
  sorry

end part1_part2_l777_777580


namespace sqrt_floor_eq_l777_777161

theorem sqrt_floor_eq (n : ℕ) (hn : 0 < n) :
    ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧ 
    ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by
  sorry

end sqrt_floor_eq_l777_777161


namespace ratio_of_B_to_C_l777_777244

theorem ratio_of_B_to_C
  (A B C : ℕ) 
  (h1 : A = B + 2) 
  (h2 : A + B + C = 47) 
  (h3 : B = 18) : B / C = 2 := 
by 
  sorry

end ratio_of_B_to_C_l777_777244


namespace hyperbola_eccentricity_proof_l777_777820

noncomputable def hyperbola_eccentricity (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) : 
    ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_proof (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) :    
    hyperbola_eccentricity a b k1 k2 ha hb C_on_hyperbola slope_condition minimized_expr = Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_proof_l777_777820


namespace coordinates_of_C_prime_l777_777911

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end coordinates_of_C_prime_l777_777911


namespace terminal_side_second_quadrant_l777_777884

def isSecondQuadrant (α : ℝ) : Prop :=
  (Real.pi / 2) < α ∧ α < Real.pi

theorem terminal_side_second_quadrant (α : ℝ) (h : α = 3) : isSecondQuadrant α :=
by
  rw [h]
  exact ⟨by norm_num, by norm_num⟩

end terminal_side_second_quadrant_l777_777884


namespace ab_sum_eq_2927_l777_777111

theorem ab_sum_eq_2927 (a b : ℕ) (h_ab_rel_prime : Nat.coprime a b)
  (h_fraction : (a : ℚ) / b = (16 : ℚ) / 21 + (81 : ℚ) / 640) : a + b = 2927 :=
by
  sorry

end ab_sum_eq_2927_l777_777111


namespace solve_system_l777_777854

-- Define the system of linear equations
def system_of_eqns : list (list ℚ) :=
  [[1, 1, 2], 
   [2, -1, 1]]

-- Define the expected solution
def solution := (1, 1)

-- The theorem stating that the system is solved by the given solution
theorem solve_system : 
  (∀ (x y : ℚ), 
    (system_of_eqns.head.head * x + system_of_eqns.head.tail.head * y = system_of_eqns.head.tail.tail.head) ∧ 
    (system_of_eqns.tail.head.head * x + system_of_eqns.tail.head.tail.head * y = system_of_eqns.tail.head.tail.tail.head)) → 
    solution = (1, 1) :=
  by
    intros x y h
    sorry

end solve_system_l777_777854


namespace problem_statement_l777_777437

variable {f : ℝ → ℝ}

def monotone_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x

theorem problem_statement (h1: ∀ x : ℝ, f x ∈ set.univ)
  (h2: monotone_decreasing_on_interval f (-∞) 5)
  (h3: ∀ t : ℝ, f (5 + t) = f (5 - t)) :
  f 9 < f (-1) ∧ f (-1) < f 13 := 
sorry

end problem_statement_l777_777437


namespace no_such_hexagon_exists_l777_777322

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l777_777322


namespace sum_of_roots_of_quadratic_l777_777699

theorem sum_of_roots_of_quadratic :
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16 in
  (∀ x, f x = 0 → x = 7 ∨ x = -1) →
  (let sum_of_roots := 7 + (-1) in sum_of_roots = 6) :=
by
  sorry

end sum_of_roots_of_quadratic_l777_777699


namespace geometry_problem_l777_777463

-- Definitions of points and distances in a circle
variables {A B C D M P Q : Point} [incircle : Incircle O]
variables {rAB : Diameter O A B} {rC : OnCircle O C} {rD : OnCircle O D}
variables {mM : MidPoint M C D} {pP : Perpendicular M AC P} {qQ : Perpendicular M AD Q}

theorem geometry_problem 
  (h1: Diameter O A B) 
  (h2: OnCircle O C) 
  (h3: OnCircle O D)
  (h4: MidPoint M C D) 
  (h5: Perpendicular M AC P) 
  (h6: Perpendicular M AD Q)
  (BC BD MP MQ MC : ℝ) :
  BC MP + BD MQ = 2 (MC)^2 := sorry

end geometry_problem_l777_777463


namespace election_result_l777_777071

noncomputable def election_proof (V : ℝ) : Prop :=
  let a_percentage := 0.32
  let b_percentage := 0.42
  let c_percentage := 0.26
  let votes_c := 0.42 * V - 1908 in
  0.26 * V = votes_c ∧ V = 1908 / 0.16

theorem election_result :
  election_proof 11925 :=
by
  unfold election_proof
  simp
  split
  { norm_num }
  { norm_num }

end election_result_l777_777071


namespace ab_eq_3bc_md_perp_bd_iff_ab_eq_3bc_l777_777089

section
variables {A B C D M : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables {AB BC : ℝ}

-- Conditions
def is_midpoint (M A B : A) : Prop := dist A M = dist M B
def is_foot_of_bisector (D : B) : Prop --{B: Type*} [metric_space B] := sorry -- need the proper condition here for angle bisector's foot
def is_perpendicular (MD BD : Type*) : Prop := sorry -- put the actual perpendicular condition

-- Given Conditions for the Problem
axiom midpoint_condition : is_midpoint M A B
axiom bisector_condition : is_foot_of_bisector D
axiom perpendicular_condition : is_perpendicular MD BD

-- Part (b): Prove that \( |AB| = 3|BC| \)
theorem ab_eq_3bc (h1 : midpoint_condition) (h2 : bisector_condition) (h3 : perpendicular_condition) : dist A B = 3 * dist B C :=
sorry

-- Part (c): Prove the Converse
theorem md_perp_bd_iff_ab_eq_3bc (h2 : bisector_condition) : (is_perpendicular MD BD ↔ dist A B = 3 * dist B C) :=
sorry

end

end ab_eq_3bc_md_perp_bd_iff_ab_eq_3bc_l777_777089


namespace solve_eq1_solve_eq2_l777_777613

theorem solve_eq1 (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2 / 3) :=
by sorry

theorem solve_eq2 (x : ℝ) : x^2 - 4 * x - 5 = 0 ↔ (x = 5 ∨ x = -1) :=
by sorry

end solve_eq1_solve_eq2_l777_777613


namespace number_of_light_configs_l777_777649

theorem number_of_light_configs (n : ℕ) : (n = 5) -> (2 ^ n = 32) :=
by
  intro h
  rw h
  exact by norm_num

end number_of_light_configs_l777_777649


namespace regular_hexagon_same_length_probability_l777_777972

theorem regular_hexagon_same_length_probability :
  let T : Finset (Finite_Field) := sorry, -- Define the actual set T una don the constructible geometry module in lean mathematical library
  let total_segments := 15,
  let sides := 6,
  let shorter_diagonals := 6,
  let longer_diagonals := 3,
  let prob_side := (6 / 15) * (5 / 14),
  let prob_shorter := (6 / 15) * (5 / 14),
  let prob_longer := (3 / 15) * (2 / 14),
  let total_prob := (prob_side + prob_shorter + prob_longer),
  (total_prob = (11 / 35)) := sorry

end regular_hexagon_same_length_probability_l777_777972


namespace triangle_side_length_l777_777519

theorem triangle_side_length (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] :
  ∀ (a b c : Real) (cosBC : Real),
  a = 5 ∧ b = 7 ∧ cosBC = 1/2 → 
  ∃ (BC : Real), BC = 8 := by 
  intros a b c cosBC h,
  cases h with ha hcb,
  cases hcb with hb hcos,
  sorry

end triangle_side_length_l777_777519


namespace find_sin2_l777_777829

noncomputable def sin_cos_identity (alpha : ℝ) :=
  sin alpha + cos alpha = 1 / 3

theorem find_sin2 (alpha : ℝ) (h : sin_cos_identity alpha) : 
  sin^2 (π / 4 - alpha) = 17 / 18 := 
sorry

end find_sin2_l777_777829


namespace probability_exactly_two_heads_and_two_tails_l777_777889

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails_l777_777889


namespace calculate_product_l777_777122

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product_l777_777122


namespace number_is_26_l777_777186

theorem number_is_26 : 2 * (4^2) - 6 = 26 := by
  calc
    2 * (4^2) - 6 = 2 * 16 - 6 : by rw [pow_two]
                ... = 32 - 6    : by norm_num
                ... = 26        : by norm_num

end number_is_26_l777_777186


namespace correct_total_count_l777_777523

variable (x : ℕ)

-- Define the miscalculation values
def value_of_quarter := 25
def value_of_dime := 10
def value_of_half_dollar := 50
def value_of_nickel := 5

-- Calculate the individual overestimations and underestimations
def overestimation_from_quarters := (value_of_quarter - value_of_dime) * (2 * x)
def underestimation_from_half_dollars := (value_of_half_dollar - value_of_nickel) * x

-- Calculate the net correction needed
def net_correction := overestimation_from_quarters - underestimation_from_half_dollars

theorem correct_total_count :
  net_correction x = 15 * x :=
by
  sorry

end correct_total_count_l777_777523


namespace range_of_t_l777_777847

open Real

noncomputable def circleF : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2 * x - 3 = 0

noncomputable def ellipseC (x y : ℝ) : Prop :=
  ∃ a b : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ a = √13 / 3 ∧ b = √13 / 2∧

theorem range_of_t :
  (-\frac{\sqrt{7}}{7}, \frac{\sqrt{7}}{7}) :=
begin
  sorry
end

end range_of_t_l777_777847


namespace sin_sum_identity_triangle_area_l777_777658

theorem sin_sum_identity (θ φ : ℝ) : 
  sin θ + sin φ = 2 * sin ((θ + φ) / 2) * cos ((θ - φ) / 2) := 
sorry

theorem triangle_area (R A B C : ℝ) 
  (radius_condition : R = 2)
  (angle_condition : sin (2 * A) + sin (A - B + C) = sin (C - A - B) + 1 / 2) :
  let sinA := sin A in
  let sinB := sin B in
  let sinC := sin C in
  2 * R^2 * sinA * sinB * sinC = 1 :=
sorry

end sin_sum_identity_triangle_area_l777_777658


namespace y_intercept_of_line_l777_777668

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l777_777668


namespace y_intercept_of_line_l777_777662

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l777_777662


namespace tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l777_777885

variable {α : Real}

theorem tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5 (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l777_777885


namespace no_such_convex_hexagon_and_point_l777_777344

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l777_777344


namespace circle_equation_center_on_x_axis_l777_777815

theorem circle_equation_center_on_x_axis
  (center : ℝ × ℝ) (eqn : ℝ × ℝ → Prop)
  (hx : center.2 = 0)
  (hA : eqn (5, 2))
  (hB : eqn (-1, 4))
  (hc : ∀ (p : ℝ × ℝ), eqn p ↔ (p.1 - center.1)^2 + p.2^2 = (5 - center.1)^2 + 2^2) :
  eqn (λ p : ℝ × ℝ, (p.1 - 1)^2 + p.2^2 = 20) := by
  sorry

end circle_equation_center_on_x_axis_l777_777815


namespace number_of_proper_subsets_of_A_inter_B_is_3_l777_777456

open Set

def A : Set ℕ := {x | -2 ≤ x ∧ x ≤ 1}
def B : Set ℤ := {x | abs x ≤ 2}
def A_inter_B : Set ℤ := {x : ℤ | x ∈ A ∩ B}

theorem number_of_proper_subsets_of_A_inter_B_is_3 : 
  (A_inter_B.card - 1) = 3 :=
by
  sorry

end number_of_proper_subsets_of_A_inter_B_is_3_l777_777456


namespace geometric_sequence_properties_l777_777439

-- Define the geometric sequence with the given conditions
def geometric_sequence (a₃ : ℕ → ℝ) := (a₃ 3 = 12) ∧ (a₃ 8 = 3 / 8)

-- Define the general formula for the n-th term of a geometric sequence
def general_term (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
def sum_of_geometric_sequence (a₁ q : ℝ) (S_n : ℕ → ℝ) (n : ℕ) : Prop :=
  S_n n = a₁ * (1 - q^n) / (1 - q)

-- The proof problem statement
theorem geometric_sequence_properties : 
  ∃ a₁ q S_n : ℝ,
  geometric_sequence (λ n, general_term a₁ q n)
  →
  ∀ n, (general_term a₁ q n = 48 * (1 / 2)^(n - 1)) 
       ∧ (S_n n = 93 → n = 5) :=
begin
  sorry
end

end geometric_sequence_properties_l777_777439


namespace probability_of_same_length_segments_l777_777953

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end probability_of_same_length_segments_l777_777953
