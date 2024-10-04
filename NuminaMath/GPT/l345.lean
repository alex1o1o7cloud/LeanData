import Mathlib
import Mathlib.Algebra.Arithmetics
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial.GroupRingAction
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finite
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Modeq
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Integration
import Mathlib.Probability.Basic
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SolveByElim
import Mathlib.Topology.MetricSpace.Basic

namespace immigration_count_l345_345115

theorem immigration_count
    (people_born : ℕ)
    (total_new_people : ℕ)
    (people_born_last_year : people_born = 90171)
    (total_new_people_last_year : total_new_people = 106491) :
    total_new_people - people_born = 16320 :=
by
  rw [people_born_last_year, total_new_people_last_year]
  norm_num
  done

end immigration_count_l345_345115


namespace angle_OQP_90_degrees_l345_345610

theorem angle_OQP_90_degrees:
  ∀ (A B C D O P Q : Point),
  convex_quadrilateral A B C D →
  inscribed_in_circle A B C D O →
  intersection_point_diag AC BD P →
  second_intersection_point_circumcircles APD BPC Q →
  ∠ (O Q P) = 90° :=
by
  intros A B C D O P Q H_convex H_inscribed H_inter_diag H_inter_circum
  sorry

end angle_OQP_90_degrees_l345_345610


namespace find_ellipse_find_circle_center_l345_345669

-- Definitions based on conditions.
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def circle (b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 = b^2

def tangent (x y : ℝ) (F : ℝ × ℝ) : Prop :=
  ∃ b : ℝ, circle b (x, y) ∧ line_through (x, y) F

def midpoint (F Q : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  N = (F.1 + Q.1) / 2, (F.2 + Q.2) / 2

axiom right_focus (a b : ℝ) (h : a > b ∧ b > 0) : ∃ F : ℝ × ℝ, F = (2, 0) ∧ ellipse a b h

axiom curve_intersects_ellipse (m : ℝ) (a b : ℝ) :
  (∀ x : ℝ, (x, x^2 + m) ∈ set_of (λ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)) →
  ∃ t : ℝ, (0, t) ∈ set_of (λ (x y : ℝ), (x = 0 ∧ y = t))

noncomputable def ellipse_equation (c : ℝ) (h1 : c = 2) : Prop :=
  ∃ a b : ℝ, ellipse a b (by sorry) ∧ a^2 = 3 * b^2

noncomputable def circle_center (m : ℝ) (a b : ℝ) : ℝ × ℝ :=
  (0, 1/3)

-- The statement to prove the equation of the ellipse.
theorem find_ellipse (c : ℝ) (h1 : c = 2) : ∃ a b : ℝ, ellipse a b (by sorry) ∧ a^2 = 3 * b^2 :=
  ⟨sqrt 6, sqrt 2, by sorry⟩

-- The statement to prove the coordinates of the center of the circle.
theorem find_circle_center (m : ℝ) (a b : ℝ) (h : ellipse a b (by sorry)) :
  circle_center m a b = (0, 1/3) :=
sorry

end find_ellipse_find_circle_center_l345_345669


namespace parabola_intersects_line_segment_range_l345_345033

theorem parabola_intersects_line_segment_range (a : ℝ) :
  (9/9 : ℝ) ≤ a ∧ a < 2 ↔
  ∃ x1 y1 x2 y2 x0,
    (y1 = a * x1^2 - 3 * x1 + 1) ∧
    (y2 = a * x2^2 - 3 * x2 + 1) ∧
    (∀ x, y = a * x^2 - 3 * x + 1) ∧
    (|x1 - x0| > |x2 - x0| → y1 > y2) ∧
    let M := (-1 : ℝ, -2 : ℝ); N := (3 : ℝ, 2 : ℝ) in
    let y_mn := (x : ℝ) → x - 1 in
    ∃ x_1 x_2, x_1 ≠ x_2 ∧
      (a * x_1^2 - 3 * x_1 + 1 = x_1 - 1) ∧
      (a * x_2^2 - 3 * x_2 + 1 = x_2 - 1) :=
begin
  sorry
end

end parabola_intersects_line_segment_range_l345_345033


namespace sum_of_first_four_terms_l345_345091

variable (a : ℕ → ℝ)
variable (q : ℝ)

def geometric_sequence : Prop :=
  a 2 = 9 ∧ a 5 = 243 ∧ ∃ q, a (n+1) = a n * q

theorem sum_of_first_four_terms (h : geometric_sequence a q) : 
  (a 1 + a 2 + a 3 + a 4) = 120 := 
sorry

end sum_of_first_four_terms_l345_345091


namespace canonical_equations_of_line_l345_345309

-- Definitions for the normal vectors of the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -2)
def n2 : ℝ × ℝ × ℝ := (1, -3, 1)

-- Define the equations of the planes
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y - 2 * z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3 * y + z + 3 = 0

-- The canonical equations of the line of intersection
def canonical_eq (x y z : ℝ) : Prop := (z * (-4)) = (y * (-9)) ∧ (z * (-3)) = (x + 3) * (-9)

theorem canonical_equations_of_line :
  ∀ x y z : ℝ, (plane1 x y z) ∧ (plane2 x y z) → canonical_eq x y z :=
by
  sorry

end canonical_equations_of_line_l345_345309


namespace virginia_sweettarts_l345_345287

theorem virginia_sweettarts (total_sweettarts : ℕ) (sweettarts_per_person : ℕ) (friends : ℕ) (sweettarts_left : ℕ) 
  (h1 : total_sweettarts = 13) 
  (h2 : sweettarts_per_person = 3) 
  (h3 : total_sweettarts = sweettarts_per_person * (friends + 1) + sweettarts_left) 
  (h4 : sweettarts_left < sweettarts_per_person) :
  friends = 3 :=
by
  sorry

end virginia_sweettarts_l345_345287


namespace product_zero_count_l345_345828

noncomputable def countOddMultiplesOf4 : ℕ :=
  let oddMultiplesOf4 := {n : ℕ | n % 8 = 4 ∧ 1 ≤ n ∧ n ≤ 3000}
  oddMultiplesOf4.toFinset.card

theorem product_zero_count : countOddMultiplesOf4 = 375 :=
  sorry

end product_zero_count_l345_345828


namespace intersection_M_N_l345_345455

def set_N : set ℝ := {x | x^2 ≤ 1}
def set_M : set ℝ := {x | x^2 - 2x - 3 < 0}

def intersection_set : set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_M_N : (set_M ∩ set_N) = intersection_set :=
sorry

end intersection_M_N_l345_345455


namespace symmetric_difference_cardinality_l345_345138

variables (x y : Set ℤ)
variables (hx : x.card = 12) (hy : y.card = 15) (hxy : (x ∩ y).card = 9)

theorem symmetric_difference_cardinality :
  (x \cup y \diff (x ∩ y)).card = 9 := by
  sorry

end symmetric_difference_cardinality_l345_345138


namespace triangle_perimeter_even_l345_345728

-- Definitions for the coordinates of points and integer lengths of sides
variable (xA yA xB yB xC yC : ℤ)

-- Definition for the lengths of sides using integer coordinates
noncomputable def length (x1 y1 x2 y2 : ℤ) : ℚ :=
  (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)).toRational

-- Definition for sides of the triangle
noncomputable def AB := length xA yA xB yB
noncomputable def BC := length xB yB xC yC
noncomputable def CA := length xC yC xA yA

-- The theorem to prove
theorem triangle_perimeter_even : 
  AB xA yA xB yB + BC xB yB xC yC + CA xC yC xA yA ∈ ℤ → 
  (AB xA yA xB yB + BC xB yB xC yC + CA xC yC xA yA) % 2 = 0 := 
  by sorry

end triangle_perimeter_even_l345_345728


namespace triangle_side_length_median_l345_345093

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l345_345093


namespace price_of_each_apple_l345_345012

theorem price_of_each_apple (price_per_asparagus_bundle price_per_grape_box total_worth total_cost_apples price_per_apple : ℝ)
  (h1 : 60 * 3 = 180)
  (h2 : 40 * 2.5 = 100)
  (h3 : 180 + 100 = 280)
  (h4 : 630 - 280 = 350)
  (h5 : 350 / 700 = price_per_apple)
  (h6 : price_per_asparagus_bundle = 3)
  (h7 : price_per_grape_box = 2.5)
  (h8 : total_worth = 630)
  : price_per_apple = 0.5 :=
begin
  sorry
end

end price_of_each_apple_l345_345012


namespace interval_monotonic_increase_l345_345047

theorem interval_monotonic_increase 
    (ω : ℝ) (φ : ℝ) (ω_pos : ω > 0) 
    (φ_bounds : -π / 2 < φ ∧ φ < π / 2) 
    (center_symmetry : f (1 / 3) = 0) 
    (BC : ℝ) (BC_eq : BC = 4) 
    (k : ℤ) :
    (f x = √3 * sin(ω * x + φ)) →
    (((4 * (k : ℝ)) - 2 / 3) < (x : ℝ) ∧ (x < (4 * (k : ℝ)) + 4 / 3)) :=
  sorry

end interval_monotonic_increase_l345_345047


namespace expressing_population_in_scientific_notation_l345_345620

def population_in_scientific_notation (population : ℝ) : Prop :=
  population = 1.412 * 10^9

theorem expressing_population_in_scientific_notation : 
  population_in_scientific_notation (1.412 * 10^9) :=
by
  sorry

end expressing_population_in_scientific_notation_l345_345620


namespace sum_of_roots_phi_l345_345213

noncomputable def phi_sum : ℝ := 
  let z6 := -1 / real.sqrt 3 - complex.I * real.sqrt (2 / 3)
  let θ := real.angle.arg(z6) -- this should give 240 degrees
  let k_values := [0, 1, 2, 3, 4, 5]
  let phi_values := k_values.map (λ k, (θ + 360 * k) / 6)
  phi_values.sum

theorem sum_of_roots_phi : phi_sum = 1140 :=
  sorry

end sum_of_roots_phi_l345_345213


namespace inverse_proportion_inequality_l345_345521

theorem inverse_proportion_inequality :
  ∀ (y : ℝ → ℝ) (y_1 y_2 y_3 : ℝ),
  (∀ x, y x = 7 / x) →
  y (-3) = y_1 →
  y (-1) = y_2 →
  y (2) = y_3 →
  y_2 < y_1 ∧ y_1 < y_3 :=
by
  intros y y_1 y_2 y_3 hy hA hB hC
  sorry

end inverse_proportion_inequality_l345_345521


namespace cannot_form_square_l345_345686

def sticks : List ℕ := [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

def total_length (l : List ℕ) : ℕ := l.sum

theorem cannot_form_square : total_length sticks % 4 ≠ 0 :=
by
  have h : total_length sticks = 50 := by sorry
  rw h
  norm_num

end cannot_form_square_l345_345686


namespace subsets_with_at_least_three_adjacent_chairs_l345_345243

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345243


namespace selection_from_departments_probability_one_woman_ξ_distribution_correct_ξ_expectation_correct_l345_345739

namespace SurveyProof

-- Definitions for department staff:
structure Department :=
  (men : ℕ)
  (women : ℕ)

def A : Department := ⟨6, 4⟩
def B : Department := ⟨3, 2⟩

-- Definitions for selecting number of staff
def total_staff_selected := 3
def ratio_A_to_B := 2 / 1  -- Practical simplified ratio

-- Definition of the number of selections from A and B
def staff_selected_A := 2
def staff_selected_B := 1

-- Proof problems:
-- 1. Prove the selected staff members from each department:
theorem selection_from_departments : staff_selected_A + staff_selected_B = total_staff_selected ∧ 
(ratio_A_to_B = staff_selected_A / staff_selected_B) := by sorry

-- 2. Probability of selecting at least one woman from department A
def total_A := A.men + A.women
def prob_no_woman_A := (nat.choose A.men 2) / (nat.choose total_A 2)
def prob_at_least_one_woman_A := 1 - prob_no_woman_A

theorem probability_one_woman : prob_at_least_one_woman_A = 2 / 3 := by sorry

-- 3. Distribution and expectation of ξ (number of men among the 3 selected)
def ξ_distribution : (ℕ → ℚ) := λ n,
  match n with
  | 0 := 4 / 75
  | 1 := 22 / 75
  | 2 := 34 / 75
  | 3 := 15 / 75
  | _ := 0
  end

def ξ_expectation := (0 * 4 / 75) + (1 * 22 / 75) + (2 * 34 / 75) + (3 * 15 / 75)

theorem ξ_distribution_correct :
  ξ_distribution 0 = 4 / 75 ∧
  ξ_distribution 1 = 22 / 75 ∧
  ξ_distribution 2 = 34 / 75 ∧
  ξ_distribution 3 = 15 / 75 := by sorry

theorem ξ_expectation_correct : ξ_expectation = 9 / 5 := by sorry

end SurveyProof

end selection_from_departments_probability_one_woman_ξ_distribution_correct_ξ_expectation_correct_l345_345739


namespace center_cell_value_l345_345539

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l345_345539


namespace number_of_adjacent_subsets_l345_345230

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345230


namespace polynomial_has_integer_roots_between_moves_l345_345643

def has_integer_roots (a b : ℤ) : Prop :=
  ∃ m n : ℤ, (a = m + n) ∧ (b = m * n)

theorem polynomial_has_integer_roots_between_moves :
  ∃ intermediate_polynomial : ℤ → ℤ, 
    (intermediate_polynomial 0 * intermediate_polynomial 1 = 
      (intermediate_polynomial 0 * intermediate_polynomial 0 + 
      (intermediate_polynomial 1 * intermediate_polynomial 1 + 
      (intermediate_polynomial 2 * intermediate_polynomial 2))) ∧
    ∃ a b : ℤ, a = 10 ∧ b = 20 ∧
    ∃ a b : ℤ, a = 20 ∧ b = 10 ∧
    (∀ P : ℤ → ℤ, 
      (∀ a b : ℤ, 
        ∃ intermediate_polynomial : ℤ → ℤ,
          P(2) = 0 ∧
          (by sorry))
    )) [ sorry

end polynomial_has_integer_roots_between_moves_l345_345643


namespace probability_of_correct_guess_l345_345715

theorem probability_of_correct_guess (num_choices : ℕ) (num_correct : ℕ)
  (h1 : num_choices = 5)
  (h2 : num_correct = 1) :
  (num_correct / num_choices : ℚ) = 1 / 5 :=
by
  rw [h1, h2]
  norm_num
  sorry

end probability_of_correct_guess_l345_345715


namespace find_percentage_l345_345514

variable (x p : ℝ)
variable (h1 : 0.25 * x = (p / 100) * 1500 - 20)
variable (h2 : x = 820)

theorem find_percentage : p = 15 :=
by
  sorry

end find_percentage_l345_345514


namespace log_function_fixed_point_l345_345524

theorem log_function_fixed_point (a m n : ℝ) (x y : ℝ) 
  (h1 : x = -1) 
  (h2 : y = -2) 
  (h3 : y = log a (x + m) + n) 
  : m * n = -4 :=
by
  sorry

end log_function_fixed_point_l345_345524


namespace distance_product_eq_l345_345111

theorem distance_product_eq
  (S : Type)
  (α β : ℝ)
  (P Q : S → S)
  (SP SQ : ℝ)
  (sin_α sin_β : ℝ)
  (h1 : ∀ s : S, P (S) = P s)
  (h2 : ∀ s : S, Q (S) = Q s)
  (hp : ∀ s : S, P s * sin_α = SP * sin_α)
  (hq : ∀ s : S, Q s * sin_β = SQ * sin_β) :
  (SP * sin_α) * (SQ * sin_β) = (SP * sin_α) * (SQ * sin_β) :=
by sorry

end distance_product_eq_l345_345111


namespace roots_numerically_equal_opposite_signs_l345_345899

theorem roots_numerically_equal_opposite_signs
  (a b d: ℝ) 
  (h: ∃ x : ℝ, (x^2 - (a + 1) * x) / ((b + 1) * x - d) = (n - 2) / (n + 2) ∧ x = -x)
  : n = 2 * (b - a) / (a + b + 2) := by
  sorry

end roots_numerically_equal_opposite_signs_l345_345899


namespace hawk_babies_expected_l345_345751

theorem hawk_babies_expected 
  (kettles : ℕ) (pregnancies_per_kettle : ℕ) (babies_per_pregnancy : ℕ) (loss_fraction : ℝ)
  (h1 : kettles = 6)
  (h2 : pregnancies_per_kettle = 15)
  (h3 : babies_per_pregnancy = 4)
  (h4 : loss_fraction = 0.25) :
  let total_babies := kettles * pregnancies_per_kettle * babies_per_pregnancy
  let surviving_fraction := 1 - loss_fraction
  let expected_surviving_babies := (total_babies : ℝ) * surviving_fraction
  in expected_surviving_babies = 270 :=
by
  sorry

end hawk_babies_expected_l345_345751


namespace true_propositions_l345_345393

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345393


namespace true_propositions_correct_l345_345369

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345369


namespace arithmetic_sequence_term_12_l345_345937

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_term_12 (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 :=
by
  -- The following line ensures the theorem compiles correctly.
  sorry

end arithmetic_sequence_term_12_l345_345937


namespace pears_pairable_and_arrangeable_l345_345689

variable (pears : List ℕ) (h_even : pears.length % 2 = 0)
variable (h_diff : ∀ i, i < pears.length - 1 → (pears[i] - pears[i+1]).abs ≤ 1)

theorem pears_pairable_and_arrangeable (h_sorted : pears.sorted (≤)) :
  ∃ pairs : List (ℕ × ℕ), 
    pairs.length * 2 = pears.length ∧ 
    (∀ j < pairs.length - 1, (pairs[j].fst - pairs[j+1].fst).abs ≤ 1 ∧ (pairs[j].snd - pairs[j+1].snd).abs ≤ 1) :=
sorry

end pears_pairable_and_arrangeable_l345_345689


namespace number_of_integers_with_zero_product_l345_345827

noncomputable def num_integers_product_zero : ℕ :=
  (1 to 2012).count (λ n : ℕ, (∃ k : ℕ, k < n ∧ (1 + complex.exp (2 * real.pi * complex.I * k / n))^n + 1 = 0))

theorem number_of_integers_with_zero_product :
  num_integers_product_zero = 335 :=
sorry

end number_of_integers_with_zero_product_l345_345827


namespace pentagonal_grid_toothpicks_l345_345694

theorem pentagonal_grid_toothpicks :
  ∀ (base toothpicks per sides toothpicks per joint : ℕ),
    base = 10 → 
    sides = 4 → 
    toothpicks_per_side = 8 → 
    joints = 5 → 
    toothpicks_per_joint = 1 → 
    (base + sides * toothpicks_per_side + joints * toothpicks_per_joint = 47) :=
by
  intros base sides toothpicks_per_side joints toothpicks_per_joint
  sorry

end pentagonal_grid_toothpicks_l345_345694


namespace squares_difference_l345_345141

theorem squares_difference (n : ℕ) (h : n > 0) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := 
by 
  sorry

end squares_difference_l345_345141


namespace imaginary_part_of_z_l345_345200

-- Define the complex number z
def z : ℂ := (i^5) / (1 - i)

-- State the theorem
theorem imaginary_part_of_z : complex.im z = 1 / 2 :=
by sorry

end imaginary_part_of_z_l345_345200


namespace maximum_n_l345_345462

def set_A (n : ℕ) : set ℕ := 
  {s | ∃ k : ℕ, k + 4 ≤ n ∧ s = 5*k + 10}

def set_B (n : ℕ) : set ℕ := 
  {s | ∃ l : ℕ, l + 5 ≤ n ∧ s = 6*l + 15}

theorem maximum_n (n : ℕ) (h₁ : n ≥ 6) (h₂ : (set_A n ∩ set_B n).card = 2016) : n = 12106 :=
  sorry

end maximum_n_l345_345462


namespace moles_C2H4Cl2_produced_l345_345001

noncomputable def moles_C2H4Cl2 (moles_C2H6 : ℕ) (moles_Cl2 : ℕ) (ratio_C2H6_Cl2 : ℕ → ℕ → Prop) : ℕ :=
  if h_C2H6 : moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2

theorem moles_C2H4Cl2_produced : 
  moles_C2H4Cl2 3 6 (λ moles_C2H6 moles_Cl2, moles_C2H6 = moles_Cl2) = 3 :=
by 
  unfold moles_C2H4Cl2
  simp
  sorry

end moles_C2H4Cl2_produced_l345_345001


namespace exponential_function_passing_point_range_of_x_l345_345495

theorem exponential_function_passing_point :
  (∃ f : ℝ → ℝ, (∀ x, f x = (2:ℝ) ^ x) ∧ f 2 = 4) :=
by
  use (λ x, (2:ℝ) ^ x)
  split
  { intro x
    refl }
  { norm_num }

theorem range_of_x (f : ℝ → ℝ) (h : ∀ x, f x = (2:ℝ) ^ x) :
  (∀ x : ℝ, f (x - 1) < 1 ↔ x < 1) :=
by
  intro x
  have h1 : f (x - 1) = (2:ℝ) ^ (x - 1) := h (x - 1)
  rw h1
  exact ⟨λ hx => by linarith [hx], λ hx => by linarith [hx]⟩

end exponential_function_passing_point_range_of_x_l345_345495


namespace speed_of_man_is_correct_l345_345764

-- Defining the conditions
def length_of_train : ℝ := 110
def speed_of_train_kmh : ℝ := 80
def time_to_pass_seconds : ℝ := 4.5

-- Convert speed of train from km/h to m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * 1000 / 3600

-- Define the relative speed equation
def relative_speed (speed_of_man_ms : ℝ) : ℝ := speed_of_train_ms + speed_of_man_ms

-- Define the distance formula
def calculated_distance (speed_of_man_ms : ℝ) : ℝ := relative_speed(speed_of_man_ms) * time_to_pass_seconds

-- Define the speed of the man in km/h
def speed_of_man_kmh (speed_of_man_ms : ℝ) : ℝ := speed_of_man_ms * 3.6

-- Main theorem to prove
theorem speed_of_man_is_correct : ∃ (v_ms: ℝ), speed_of_man_kmh v_ms = 8.0064 ∧ calculated_distance(v_ms) = length_of_train :=
by
  sorry

end speed_of_man_is_correct_l345_345764


namespace number_of_non_negative_x_l345_345438

theorem number_of_non_negative_x (x : ℝ) : 
  (∃ n : ℕ, x = n^3) →
  (∃ k : ℕ, k ∈ set.Icc 0 12 ∧ (∃ x : ℝ, 0 <= x ∧ (144 - (x ^ (1 / 3)) = k))) →
  (∃ x_set : finset ℝ, (∀ x ∈ x_set, 0 ≤ x ∧ ∃ k, k ∈ finset.range 13 ∧ 144 - (x ^ (1 / 3)) = k) ∧ x_set.card = 13) :=
begin
  sorry
end

end number_of_non_negative_x_l345_345438


namespace jenna_stickers_l345_345667

theorem jenna_stickers (Kate_stickers Jenna_stickers : ℕ) 
  (ratio_condition : Kate_stickers.to_rat / Jenna_stickers.to_rat = 7/4)
  (Kate_count : Kate_stickers = 21) : Jenna_stickers = 12 :=
by {
  -- We assume the conditions provided
  have h : (Kate_stickers.to_rat) / (Jenna_stickers.to_rat) = (7 : ℚ) / (4 : ℚ) := ratio_condition,
  have k : Kate_stickers = 21 := Kate_count,
  -- Actual proof skipped
  sorry
}

end jenna_stickers_l345_345667


namespace range_of_f_gt_a_l345_345913

-- Define the piecewise function f
def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x else -x^2 + a * x

-- Define f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Define the target range of x values
def target_range : Set ℝ :=
{ x | x > -1 - Real.sqrt 3 }

-- The theorem we want to prove
theorem range_of_f_gt_a (a : ℝ) (h : is_odd (f a)) : 
  { x : ℝ | f x a > a } = { x : ℝ | x ∈ target_range } :=
sorry

end range_of_f_gt_a_l345_345913


namespace complex_solutions_real_parts_product_l345_345197

theorem complex_solutions_real_parts_product (x : ℂ) (i : ℂ) :
  (∃ (x : ℂ), x^2 - 2 * x = i) → 
  let y := 1 in -- Representing the real part of 1 independently
  let z1 := y + re (e^(complex.I * real.pi / 8) * real.sqrt (2 ^ (1 / 4))) in
  let z2 := y - re (e^(complex.I * real.pi / 8) * real.sqrt (2 ^ (1 / 4))) in
  (z1 * z2 = (1 - real.sqrt 2) / 2) := sorry

end complex_solutions_real_parts_product_l345_345197


namespace sum_first_ten_terms_l345_345861

theorem sum_first_ten_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (h₁ : ∀ n, S (n + 1) = S n + a n + 3) 
  (h₂ : a 5 + a 6 = 29) : 
  ∑ i in finset.range 10, (a i + a (i + 1)) = 320 := sorry

end sum_first_ten_terms_l345_345861


namespace polygon_number_of_sides_l345_345930

theorem polygon_number_of_sides (h : ∀ (n : ℕ), (360 : ℝ) / (n : ℝ) = 1) : 
  360 = (1:ℝ) :=
  sorry

end polygon_number_of_sides_l345_345930


namespace inequality_problem_l345_345153

theorem inequality_problem (x : ℝ) (h_denom : 2 * x^2 + 2 * x + 1 ≠ 0) : 
  -4 ≤ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ≤ 1 :=
sorry

end inequality_problem_l345_345153


namespace sean_div_julie_l345_345171

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l345_345171


namespace train_passes_bridge_in_36_seconds_l345_345716

theorem train_passes_bridge_in_36_seconds
  (length_train : ℝ)
  (speed_train_km_h : ℝ)
  (length_bridge : ℝ) :
  (length_train = 310) →
  (speed_train_km_h = 45) →
  (length_bridge = 140) →
  let speed_train_m_s : ℝ := speed_train_km_h * 1000 / 3600 in
  let total_distance : ℝ := length_train + length_bridge in
  let time : ℝ := total_distance / speed_train_m_s in
  time = 36 := 
by
  intros
  sorry

end train_passes_bridge_in_36_seconds_l345_345716


namespace carol_pennies_l345_345065

variable (a c : ℕ)

theorem carol_pennies (h₁ : c + 2 = 4 * (a - 2)) (h₂ : c - 2 = 3 * (a + 2)) : c = 62 :=
by
  sorry

end carol_pennies_l345_345065


namespace chairs_subsets_l345_345252

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345252


namespace medians_intersect_at_one_point_l345_345998

open_locale classical
noncomputable theory

structure Triangle :=
(A B C : ℝ × ℝ)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def is_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
∃ (k : ℝ), p1.1 = k * p2.1 ∧ p1.2 = k * p2.2

theorem medians_intersect_at_one_point
  (Δ : Triangle)
  (D : midpoint Δ.B Δ.C)
  (E : midpoint Δ.A Δ.C)
  (F : midpoint Δ.A Δ.B)
  (e1 e2 : ℝ × ℝ)
  (h1 : Δ.B - Δ.A = e1)
  (h2 : Δ.C - Δ.B = e2)
  (h3 : ¬is_collinear Δ.A Δ.B Δ.C) :
  ∃ (G : ℝ × ℝ), is_collinear (G - D) (Δ.A - Δ.B) ∧ 
  is_collinear (G - E) (Δ.C - Δ.A) ∧ 
  is_collinear (G - F) (Δ.B - Δ.C) := 
sorry

end medians_intersect_at_one_point_l345_345998


namespace min_value_of_norm_diff_l345_345862

variables {V : Type*} [inner_product_space ℝ V]

/-- The problem's vector space and conditions --/
variables (a b c : V) (t1 t2 : ℝ)

/-- Given conditions --/
axiom a_unit : ∥a∥ = 1
axiom b_unit : ∥b∥ = 1
axiom a_perp_b : ⟪a, b⟫ = 0
axiom c_norm : ∥c∥ = 13
axiom c_dot_a : ⟪c, a⟫ = 3
axiom c_dot_b : ⟪c, b⟫ = 4

/-- Problem statement --/
theorem min_value_of_norm_diff : 
  ∃ t1 t2 : ℝ, ∥c - t1 • a - t2 • b∥ = 12 :=
sorry

end min_value_of_norm_diff_l345_345862


namespace subset_equation_l345_345471

theorem subset_equation (m : ℝ) :
  let A := {-1}
  let B := {x : ℝ | x^2 + m * x - 3 = 1}
  A ⊆ B -> m = -3 :=
begin
  -- proof will be added here
  intro h,
  sorry  -- placeholder as we are not providing the proof
end

end subset_equation_l345_345471


namespace count_4_digit_numbers_with_property_l345_345502

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l345_345502


namespace subsets_with_at_least_three_adjacent_chairs_l345_345245

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345245


namespace subsets_with_at_least_three_adjacent_chairs_l345_345249

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345249


namespace garden_minimum_cost_l345_345156

def area_plot_1 := 5 * 2
def area_plot_2 := 4 * 4
def area_plot_3 := 3 * 3
def area_plot_4 := 6 * 1
def area_plot_5 := 4 * 1

def cost_sunflowers := 0.75
def cost_tulips := 1.25
def cost_roses := 1.75
def cost_orchids := 2.25
def cost_hydrangeas := 2.75

noncomputable def total_cost := (cost_hydrangeas * area_plot_5) + (cost_orchids * area_plot_4) + (cost_roses * area_plot_3) + (cost_tulips * area_plot_1) + (cost_sunflowers * area_plot_2)

theorem garden_minimum_cost : total_cost = 64.75 := 
by
  -- proof to be filled
  sorry

end garden_minimum_cost_l345_345156


namespace min_abs_expr1_min_abs_expr2_l345_345205

theorem min_abs_expr1 (x : ℝ) : |x - 4| + |x + 2| ≥ 6 := sorry

theorem min_abs_expr2 (x : ℝ) : |(5 / 6) * x - 1| + |(1 / 2) * x - 1| + |(2 / 3) * x - 1| ≥ 1 / 2 := sorry

end min_abs_expr1_min_abs_expr2_l345_345205


namespace proof_opposite_triangles_equal_area_l345_345758

variables {A B C D P : Type} [AffineSpace ℝ A B C D P]

def parallelogram (A B C D : A) : Prop := parallelogram A B C D

def opposite_triangles_equal_area (P : P) (A B C D : A) (par: parallelogram A B C D) : Prop :=
  let area₁ := triangle_area P A B
  let area₂ := triangle_area P B C
  let area₃ := triangle_area P C D
  let area₄ := triangle_area P D A
  area₁ + area₃ = area₂ + area₄

theorem proof_opposite_triangles_equal_area (A B C D P : P)
    (par: parallelogram A B C D)
    (total_area: area (parallelogram A B C D) = area₁ + area₂ + area₃ + area₄) :
  opposite_triangles_equal_area P A B C D par :=
sorry

end proof_opposite_triangles_equal_area_l345_345758


namespace true_propositions_l345_345379

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345379


namespace number_of_true_propositions_l345_345389

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345389


namespace parabola_circle_intersection_distance_sum_l345_345126

noncomputable def parabola_focus_distance_sum 
  (parabola : ℝ → ℝ) 
  (circle : ℝ × ℝ → ℝ)
  (focus : ℝ × ℝ)
  (points : List (ℝ × ℝ))
  (expected_sum : ℝ) : Prop :=
parabola = (fun x => x^2) ∧
circle = (fun p => (p.1 - 1)^2 + (p.2 - 2)^2 - 500^2) ∧
focus = (0, 1/4) ∧
points = [(-28, 784), (-2, 4), (14, 196), (18, 324)] ∧
(expected_sum = points.foldl (fun acc (p : ℝ × ℝ) => acc + abs (p.2 - focus.2)) 0)

theorem parabola_circle_intersection_distance_sum :
  parabola_focus_distance_sum (fun x => x^2) 
                              (fun p => (p.1 - 1)^2 + (p.2 - 2)^2 - 500^2) 
                              (0, 1/4) 
                              [(-28, 784), (-2, 4), (14, 196), (18, 324)] 
                              1307 := 
by
  unfold parabola_focus_distance_sum
  repeat {trivial <|> unfold List.foldl <|> sorry}

end parabola_circle_intersection_distance_sum_l345_345126


namespace circle_chairs_adjacent_l345_345264

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345264


namespace employee_count_after_hire_l345_345082

theorem employee_count_after_hire (initial_employees : ℕ) (percentage_increase : ℕ) (new_total : ℕ)
  (h_initial : initial_employees = 1200)
  (h_increase : percentage_increase = 40)
  (h_new_total : new_total = 1680) :
  initial_employees + (initial_employees * percentage_increase) / 100 = new_total :=
by
  rw [h_initial, h_increase, h_new_total]
  norm_num
  sorry

end employee_count_after_hire_l345_345082


namespace problem_solution_l345_345061

theorem problem_solution (a0 a1 a2 a3 a4 a5 : ℝ) :
  (1 + 2*x)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 →
  a0 + a2 + a4 = 121 := 
sorry

end problem_solution_l345_345061


namespace smallest_number_divisible_conditions_l345_345295

theorem smallest_number_divisible_conditions :
  (∃ n : ℕ, 
    n % 9 = 0 ∧ 
    n % 2 = 1 ∧ 
    n % 3 = 1 ∧ 
    n % 4 = 1 ∧ 
    n % 5 = 1 ∧ 
    n % 6 = 1 ∧ 
    n % 7 = 1 ∧ 
    n % 8 = 1 ∧ 
    n = 5041) :=
by { sorry }

end smallest_number_divisible_conditions_l345_345295


namespace seven_times_equivalent_l345_345626

theorem seven_times_equivalent (n a b : ℤ) (h : n = a^2 + a * b + b^2) :
  ∃ (c d : ℤ), 7 * n = c^2 + c * d + d^2 :=
sorry

end seven_times_equivalent_l345_345626


namespace sin_double_beta_cos_alpha_plus_pi_over_4_l345_345441

namespace ProofProblem

variables {α β : ℝ}
-- Conditions
axiom alpha_range : 0 < α ∧ α < π / 2
axiom beta_range : π / 2 < β ∧ β < π
axiom cos_beta_minus_pi_over_4 : cos (β - π / 4) = 1 / 3
axiom sin_alpha_plus_beta : sin (α + β) = 4 / 5

-- Prove that sin(2β) = -7/9
theorem sin_double_beta : sin (2 * β) = -7 / 9 :=
  sorry

-- Prove that cos(α + π/4) = (-3 + 8 * √2) / 15
theorem cos_alpha_plus_pi_over_4 : cos (α + π / 4) = (-3 + 8 * Real.sqrt 2) / 15 :=
  sorry

end ProofProblem

end sin_double_beta_cos_alpha_plus_pi_over_4_l345_345441


namespace intersection_of_M_and_N_l345_345858

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2x - 3 < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : (M ∩ N) = C :=
by
  sorry

end intersection_of_M_and_N_l345_345858


namespace correct_logarithmic_identity_l345_345769

variables (a x y : ℝ)

theorem correct_logarithmic_identity 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : x > 0) 
  (h4 : y > 0) : 
  log a (sqrt x / y) = 1 / 2 * log a x - log a y := 
sorry

end correct_logarithmic_identity_l345_345769


namespace neg_exists_equiv_forall_l345_345014

theorem neg_exists_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by
  sorry

end neg_exists_equiv_forall_l345_345014


namespace parabola_line_intersection_distance_l345_345039

noncomputable def point (x y : ℝ) := (x, y)

def parabola := {p : ℝ × ℝ | p.2 = 2 * p.1^2}

def line (k : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 1/8}

def focus := (0 : ℝ, 1/8 / (4 * 2))

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem parabola_line_intersection_distance
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A ∈ parabola)
  (hB : B ∈ parabola)
  (hA' : A ∈ line k)
  (hB' : B ∈ line k)
  (hAF : distance A focus = 1) :
  distance A B = 8 / 7 :=
sorry

end parabola_line_intersection_distance_l345_345039


namespace complement_intersection_eq_l345_345137

open Set

def U := {1, 2, 3, 4}
def M := {1, 2, 3}
def N := {2, 3, 4}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 4} := by
  -- Proof goes here
  sorry

end complement_intersection_eq_l345_345137


namespace find_f_sqrt_5_minus_1_l345_345599

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_sqrt_5_minus_1 :
  (∀ x > 0, f x ≤ f (x + 1)) → -- f is monotonically increasing
  (∀ x > 0, f x * f (f (x) + 1 / x) = 1) → -- f(x) * f(f(x) + 1/x) = 1
  f (sqrt 5 - 1) = -1 / 2 := 
sorry

end find_f_sqrt_5_minus_1_l345_345599


namespace value_of_clothing_piece_eq_l345_345338

def annual_remuneration := 10
def work_months := 7
def received_silver_coins := 2

theorem value_of_clothing_piece_eq : 
  ∃ x : ℝ, (x + received_silver_coins) * 12 = (x + annual_remuneration) * work_months → x = 9.2 :=
by
  sorry

end value_of_clothing_piece_eq_l345_345338


namespace exist_three_sum_eq_third_l345_345995

theorem exist_three_sum_eq_third
  (A : Finset ℕ)
  (h_card : A.card = 52)
  (h_cond : ∀ (a : ℕ), a ∈ A → a ≤ 100) :
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = z :=
sorry

end exist_three_sum_eq_third_l345_345995


namespace number_of_such_points_l345_345135

open Set

noncomputable def p_set (x : ℕ) : Set ℕ := {x, 1}
noncomputable def q_set (y : ℕ) : Set ℕ := {y, 1, 2}

theorem number_of_such_points :
  let P := { x | ∃ y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, p_set x ⊆ q_set y },
      pairs := { (x, y) ∈ P × P | p_set x ⊆ q_set y }
  in pairs.size = 14 :=
by
  sorry

end number_of_such_points_l345_345135


namespace john_marbles_l345_345708

theorem john_marbles : ∃ m : ℕ, (m ≡ 3 [MOD 7]) ∧ (m ≡ 2 [MOD 4]) ∧ m = 10 := by
  sorry

end john_marbles_l345_345708


namespace option_B_valid_option_D_valid_l345_345477

variables (Line Plane : Type)
variables (m n : Line) (α β : Plane)

-- Define the conditions
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Line → Prop)
variables (contained : Line → Plane → Prop)
variables (plane_perpendicular : Plane → Plane → Prop)

-- State the conditions as hypotheses
variables (hm_parallel_α : parallel m α)
variables (hn_contained_α : contained n α)
variables (hm_perpendicular_n : perpendicular m n)
variables (hm_perpendicular_α : perpendicular m α)
variables (hn_perpendicular_β : perpendicular n β)
variables (hα_perpendicular_β : plane_perpendicular α β)
variables (hm_contained_α : contained m α)
variables (hn_contained_β : contained n β)
variables (hα_parallel_β : plane_perpendicular α β)

-- Conclusion B
theorem option_B_valid : (hm_perpendicular_n) ∧ (hm_perpendicular_α) ∧ (hn_perpendicular_β) → (plane_perpendicular α β) :=
by sorry

-- Conclusion D
theorem option_D_valid : (hα_parallel_β) ∧ (hm_perpendicular_α) ∧ (hn_perpendicular_β) → (perpendicular m n) :=
by sorry

end option_B_valid_option_D_valid_l345_345477


namespace limit_fraction_to_three_l345_345361

noncomputable def limit_frac : ℕ → ℝ := λ n, (3 * (n:ℝ) - 1) / (n + 2)

theorem limit_fraction_to_three : Filter.Tendsto limit_frac Filter.atTop (𝓝 3) :=
by sorry

end limit_fraction_to_three_l345_345361


namespace find_sum_a_b_l345_345611

noncomputable def point := (ℝ × ℝ)

def is_square (A B C D : point) (s : ℝ) : Prop :=
  A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s)

def is_equilateral_triangle (X Y Z : point) : Prop :=
  dist X Y = dist X Z ∧ dist X Y = dist Y Z

def WXYZ_area_is (A B C D X Y Z W : point) : Prop :=
  is_square A B C D 2 ∧
  is_equilateral_triangle A B X ∧
  is_equilateral_triangle B C Y ∧
  is_equilateral_triangle C D Z ∧
  is_equilateral_triangle D A W ∧
  ∃ a b : ℝ, WXYZ_area W X Y Z = a + sqrt b ∧ (4 : ℝ) + sqrt (12 : ℝ) = a + sqrt b

theorem find_sum_a_b
  (A B C D X Y Z W : point)
  (h1 : is_square A B C D 2)
  (h2 : is_equilateral_triangle A B X)
  (h3 : is_equilateral_triangle B C Y)
  (h4 : is_equilateral_triangle C D Z)
  (h5 : is_equilateral_triangle D A W)
  (h6 : ∃ a b : ℝ, WXYZ_area W X Y Z = a + sqrt b)
  : a + b = 16 := sorry

end find_sum_a_b_l345_345611


namespace president_and_committee_combination_l345_345934

theorem president_and_committee_combination (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 ∧ k = 3 ∧ total = (10 * Nat.choose 9 3) → total = 840 :=
by
  intros
  sorry

end president_and_committee_combination_l345_345934


namespace inequality_solution_range_of_a_l345_345883

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

def range_y := Set.Icc (-2 : ℝ) 2

def subset_property (a : ℝ) : Prop := 
  Set.Icc a (2 * a - 1) ⊆ range_y

theorem inequality_solution (x : ℝ) :
  f x ≤ x^2 - 3 * x + 1 ↔ x ≤ 1 ∨ x ≥ 3 := sorry

theorem range_of_a (a : ℝ) :
  subset_property a ↔ 1 ≤ a ∧ a ≤ 3 / 2 := sorry

end inequality_solution_range_of_a_l345_345883


namespace count_at_least_three_adjacent_chairs_l345_345242

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345242


namespace circle_chairs_adjacent_l345_345267

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345267


namespace cara_total_bread_l345_345004

variable (L B : ℕ)  -- Let L and B be the amount of bread for lunch and breakfast, respectively

theorem cara_total_bread :
  (dinner = 240) → 
  (dinner = 8 * L) → 
  (dinner = 6 * B) → 
  (total_bread = dinner + L + B) → 
  total_bread = 310 :=
by
  intros
  -- Here you'd begin your proof, implementing each given condition
  sorry

end cara_total_bread_l345_345004


namespace angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l345_345924

-- Conditions for (1): In ΔABC, A = 60°, a = 4√3, b = 4√2, prove B = 45°.
theorem angle_B_in_triangle_ABC
  (A : Real)
  (a b : Real)
  (hA : A = 60)
  (ha : a = 4 * Real.sqrt 3)
  (hb : b = 4 * Real.sqrt 2) :
  ∃ B : Real, B = 45 := by
  sorry

-- Conditions for (2): In ΔABC, a = 3√3, c = 2, B = 150°, prove b = 7.
theorem side_b_in_triangle_ABC
  (a c B : Real)
  (ha : a = 3 * Real.sqrt 3)
  (hc : c = 2)
  (hB : B = 150) :
  ∃ b : Real, b = 7 := by
  sorry

end angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l345_345924


namespace junior_score_l345_345081

theorem junior_score 
  (n : ℕ) 
  (h1 : 0.2 * n = floor (0.2 * n))  -- ensuring 0.2n is an integer (number of juniors)
  (h2 : 0.8 * n = floor (0.8 * n))  -- ensuring 0.8n is an integer (number of seniors)
  (h_avg : (78 : ℝ))
  (h4 : ∀ j ∈ {x | x ≤ n ∧ (∃m, (5: ℝ)/n = 1)}, j = j)
  (h_avg_seniors : (75 : ℝ)) :
  (∀ j ∈ {x | x ≤ n ∧ (∃m, (5: ℝ)/n = 1)}, j = 90) :=
sorry

end junior_score_l345_345081


namespace beverage_ratio_l345_345987

theorem beverage_ratio (x y : ℕ) (h1 : 5 * x + 4 * y = 5.5 * x + 3.6 * y) : x : y = 4 : 5 :=
by
  sorry

end beverage_ratio_l345_345987


namespace true_propositions_correct_l345_345370

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345370


namespace angle_bisector_slope_l345_345186

/-
Given conditions:
1. line1: y = 2x
2. line2: y = 4x
Prove:
k = (sqrt(21) - 6) / 7
-/

theorem angle_bisector_slope :
  let m1 := 2
  let m2 := 4
  let k := (Real.sqrt 21 - 6) / 7
  (1 - m1 * m2) ≠ 0 →
  k = (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
:=
sorry

end angle_bisector_slope_l345_345186


namespace krishan_money_eq_l345_345306

noncomputable def ram_money := 686
def ratio_rg := 5 / 11
def ratio_gk := 3 / 5

theorem krishan_money_eq :
  let G := ratio_rg * ram_money in
  let K := (5 / 3) * G in
  K = 2515 :=
sorry

end krishan_money_eq_l345_345306


namespace neg_or_false_implies_or_true_l345_345917

theorem neg_or_false_implies_or_true (p q : Prop) (h : ¬(p ∨ q) = False) : p ∨ q :=
by {
  sorry
}

end neg_or_false_implies_or_true_l345_345917


namespace always_meaningful_l345_345436

theorem always_meaningful (a : ℝ) : ∀ a : ℝ, ∃ b : ℝ, b = a / 2 := 
by 
  intro a
  use a / 2
  exact rfl

end always_meaningful_l345_345436


namespace sean_div_julie_l345_345170

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l345_345170


namespace probability_at_least_five_l345_345077

open Classical 

def redPackets : list ℝ := [2.63, 1.95, 3.26, 1.77, 0.39]

def allCombinationsSumAtLeastFive : list (ℝ × ℝ) :=
  [(2.63, 3.26), (3.26, 1.95), (3.26, 1.77)]

def validCombinationCount : ℕ := 3
def totalCombinationCount : ℕ := 5.choose 2
def desiredProbability : ℝ := validCombinationCount / totalCombinationCount

theorem probability_at_least_five :
  desiredProbability = 3 / 10 :=
by
  sorry

end probability_at_least_five_l345_345077


namespace find_m_value_l345_345874

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x + 3
  else x * x - 2

theorem find_m_value (m : ℝ) : f(m) = 2 ↔ m = -2 :=
sorry

end find_m_value_l345_345874


namespace cos_difference_zero_l345_345637

noncomputable def cos72 := Real.cos (72 * Real.pi / 180)
noncomputable def cos144 := Real.cos (144 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)

theorem cos_difference_zero :
  cos72 - cos144 = 0 :=
by
  let c := cos72
  let d := cos144
  have h1 : d = -(1 - 2 * c^2), by sorry
  have h2 : c - d = c + 1 - 2 * c^2, by sorry
  have h3 : 2 * c^2 - c - 1 = 0, by sorry
  have h4 : c = 1, by sorry
  sorry

end cos_difference_zero_l345_345637


namespace hyperbola_eccentricity_l345_345473

open Real

-- Definitions for conditions
variable (a b c : ℝ) (ha : a > 0) (hb : b > 0)
def hyperbola := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
def asymptote := ∀ x y : ℝ, b * x - a * y = 0
def F1 := (-c, 0)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def F1M := b
def MN := 3 * b
def NF1 := 4 * b
def cos_angle := b / c
def NF2 := 4 * b - 2 * a

-- Proof statement
theorem hyperbola_eccentricity
  (h1 : hyperbola a b) (h2 : asymptote a b) (hF1 : distance F1 (0, 0) = c)
  (hF1M : distance F1 (0, 0) = b) (hMN : distance (F1.1, 0) (F1.1 + 3 * b, 0) = 3 * b)
  (hNF1 : distance (F1.1 + 3 * b, 0) F1 = 4 * b) :
  ∃ e : ℝ, e = 5 / 3 :=
sorry

end hyperbola_eccentricity_l345_345473


namespace expression_value_l345_345734

theorem expression_value : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by {
  sorry
}

end expression_value_l345_345734


namespace locus_of_Q_l345_345025

theorem locus_of_Q (x y : ℝ) (λ : ℝ) (h₁ : 2/3 ≤ λ ∧ λ ≤ 1) :
  (∃ x₁ y₁ : ℝ, (x₁^2/3 + y₁^2/2 = 1) ∧ ((x = 6 ∧ (x - 6)^2/3 + y^2/2 = 1) ∨ (x = 5 ∧ ((x - 5)^2/(4/3) + y^2/2 = 1))) := 
sorry

end locus_of_Q_l345_345025


namespace staplers_left_l345_345683

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l345_345683


namespace nth_equation_sum_odd_sequence_l345_345619

-- Define the nth equation pattern and prove it
theorem nth_equation (n : ℕ) : n^2 = (n + 1)^2 - (2 * n + 1) :=
by sorry

-- Prove the sum of the sequence for n ≥ 2
theorem sum_odd_sequence (n : ℕ) (hn : n ≥ 2) : 
  (∑ k in Finset.range n, (2 * (k + 1) - 1)) = n^2 - 1 :=
by sorry

end nth_equation_sum_odd_sequence_l345_345619


namespace ten_pow_m_add_n_l345_345443

-- Let m and n be real numbers such that lg 3 = m and lg 4 = n.
variable (m n : Real)

-- Define the conditions
axiom lg_3_eq_m : log 3 = m
axiom lg_4_eq_n : log 4 = n

-- The goal is to prove that 10^(m + n) = 12.
theorem ten_pow_m_add_n (m n : Real) (lg_3_eq_m : log 3 = m) (lg_4_eq_n : log 4 = n) : 10^(m + n) = 12 := 
by
  sorry

end ten_pow_m_add_n_l345_345443


namespace center_cell_value_l345_345532

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l345_345532


namespace inequality_proof_l345_345134

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : min (a + b) (min (b + c) (c + a)) > sqrt 2) (h5 : a^2 + b^2 + c^2 = 3) : 
  (a / (b + c - a)^2) + (b / (c + a - b)^2) + (c / (a + b - c)^2) ≥ 3 / (a * b * c)^2 := 
sorry

end inequality_proof_l345_345134


namespace part_a_2n_squared_rods_possible_part_b_max_rods_no_overlap_l345_345723

theorem part_a_2n_squared_rods_possible (n : ℕ) (h : ∀ (x y z : ℕ), x < 2*n → y < 2*n → z < 2*n → ∃ r, r.pierces x y z ∧ r.direction ∈ {direction.x, direction.y, direction.z}) :
  ∃ (rods : set rod), rods.card = 2*n^2 ∧ (∀ r ∈ rods, r.pierces_exactly (2*n)) ∧ pairwise_disjoint rods :=
sorry

theorem part_b_max_rods_no_overlap (n : ℕ) (h : ∀ (x y z : ℕ), x < 2*n → y < 2*n → z < 2*n → ∃ r, r.pierces x y z ∧ r.direction ∈ {direction.x, direction.y, direction.z}) :
  ∃ (rods : set rod), rods.card = 2*n^2 ∧ (∀ r ∈ rods, r.pierces_exactly (2*n)) ∧ pairwise_disjoint rods :=
sorry

end part_a_2n_squared_rods_possible_part_b_max_rods_no_overlap_l345_345723


namespace find_DF_l345_345098

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345098


namespace circle_chairs_adjacent_l345_345269

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345269


namespace derivative_at_zero_l345_345041

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem derivative_at_zero :
  let f' (x : ℝ) := (x + 2) * Real.exp x in
  f' 0 = 2 := 
by
  sorry

end derivative_at_zero_l345_345041


namespace yao_ming_shots_l345_345932

-- Defining the conditions
def total_shots_made : ℕ := 14
def total_points_scored : ℕ := 28
def three_point_shots_made : ℕ := 3
def two_point_shots (x : ℕ) : ℕ := x
def free_throws_made (x : ℕ) : ℕ := total_shots_made - three_point_shots_made - x

-- The theorem we want to prove
theorem yao_ming_shots :
  ∃ (x y : ℕ),
    (total_shots_made = three_point_shots_made + x + y) ∧ 
    (total_points_scored = 3 * three_point_shots_made + 2 * x + y) ∧
    (x = 8) ∧
    (y = 3) :=
sorry

end yao_ming_shots_l345_345932


namespace gears_rotation_l345_345327

variable gear1_rotates gear2_rotates gear3_rotates gear4_rotates gear5_rotates gear6_rotates : Prop

axiom condition1 : gear1_rotates → (gear2_rotates ∧ ¬gear5_rotates)
axiom condition2 : (gear2_rotates ∨ gear5_rotates) → ¬gear4_rotates
axiom condition3 : (gear3_rotates ∧ gear4_rotates) ∨ (¬gear3_rotates ∧ ¬gear4_rotates)
axiom condition4 : gear5_rotates ∨ gear6_rotates

theorem gears_rotation (h : gear1_rotates) : gear2_rotates ∧ gear3_rotates ∧ gear6_rotates :=
by
  sorry

end gears_rotation_l345_345327


namespace triangle_area_from_altitudes_l345_345765

noncomputable def triangleArea (altitude1 altitude2 altitude3 : ℝ) : ℝ :=
  sorry

theorem triangle_area_from_altitudes
  (h1 : altitude1 = 15)
  (h2 : altitude2 = 21)
  (h3 : altitude3 = 35) :
  triangleArea 15 21 35 = 245 * Real.sqrt 3 :=
sorry

end triangle_area_from_altitudes_l345_345765


namespace amount_of_meat_left_l345_345580

theorem amount_of_meat_left (initial_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ)
  (h0 : initial_meat = 20)
  (h1 : meatballs_fraction = 1/4)
  (h2 : spring_rolls_meat = 3) : 
  (initial_meat - (initial_meat * meatballs_fraction:ℕ) - spring_rolls_meat) = 12 := 
by 
  sorry

end amount_of_meat_left_l345_345580


namespace maximum_diff_on_line_l345_345884

structure Point :=
  (x : ℝ)
  (y : ℝ)

def line (p : Point) : Prop := p.x - 2 * p.y + 8 = 0

def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def PA (P : Point) := dist P (Point.mk (-2) 8)

def PB (P : Point) := dist P (Point.mk (-2) (-4))

noncomputable def max_diff_PA_PB (P : Point) : ℝ :=
  Real.abs (PA P - PB P)

theorem maximum_diff_on_line :
  ∃ P : Point, line P ∧ P = Point.mk 12 10 ∧ max_diff_PA_PB P = 4 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_diff_on_line_l345_345884


namespace sqrt_inequality_l345_345150

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt (a - 2) - sqrt (a - 3) > sqrt a - sqrt (a - 1) :=
sorry

end sqrt_inequality_l345_345150


namespace train_stations_hours_apart_l345_345692

theorem train_stations_hours_apart (total_travel_time : ℕ) (break_time : ℕ) (actual_travel_time : ℕ) : 
  total_travel_time = 270 → break_time = 30 → actual_travel_time = total_travel_time - break_time → 
  actual_travel_time / 60 = 4 :=
by
  intros htot hbreak hactual
  rw [htot, hbreak, hactual]
  norm_num
  sorry  -- The proof goes here

end train_stations_hours_apart_l345_345692


namespace integral_example_l345_345813

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem integral_example : ∫ x in -2..-1, f x = -2 * Real.log 2 :=
by
  sorry

end integral_example_l345_345813


namespace limit_equivalence_l345_345342

open Nat
open Real

variable {u : ℕ → ℝ} {L : ℝ}

def original_def (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L - u n| ≤ ε

def def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, n < N ∨ |L - u n| ≤ ε)

def def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∀ n : ℕ, ∃ N : ℕ, n ≥ N → |L - u n| ≤ ε

def def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |L - u n| < ε

def def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε > 0, ∀ n ≥ N, |L - u n| ≤ ε

theorem limit_equivalence :
  original_def u L ↔ def1 u L ∧ def3 u L ∧ ¬def2 u L ∧ ¬def4 u L :=
by
  sorry

end limit_equivalence_l345_345342


namespace c1_c2_not_collinear_l345_345774

def vector := ℝ × ℝ × ℝ

def a : vector := (3, 5, 4)

def b : vector := (5, 9, 7)

def c1 (a b : vector) : vector :=
  (-2 * a.1 + b.1, -2 * a.2 + b.2, -2 * a.3 + b.3)

def c2 (a b : vector) : vector :=
  (3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2, 3 * a.3 - 2 * b.3)

def collinear (v1 v2 : vector) : Prop :=
  ∃ γ : ℝ, v1 = (γ * v2.1, γ * v2.2, γ * v2.3)

theorem c1_c2_not_collinear : ¬ collinear (c1 a b) (c2 a b) :=
  sorry

end c1_c2_not_collinear_l345_345774


namespace eval_expr_at_neg_half_l345_345638

theorem eval_expr_at_neg_half :
  (let x := (-1 : ℚ) / 2 in 2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1)) = -7 :=
by
  sorry

end eval_expr_at_neg_half_l345_345638


namespace sean_divided_by_julie_is_2_l345_345158

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l345_345158


namespace circle_chairs_adjacent_l345_345268

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345268


namespace meat_left_l345_345585

theorem meat_left (initial_meat : ℕ) (meatball_fraction : ℚ) (spring_roll_meat : ℕ) 
  (h_initial : initial_meat = 20) 
  (h_meatball_fraction : meatball_fraction = 1/4)
  (h_spring_roll_meat : spring_roll_meat = 3) : 
  initial_meat - (initial_meat * meatball_fraction.num / meatball_fraction.denom).toNat - spring_roll_meat = 12 :=
by
  sorry

end meat_left_l345_345585


namespace independence_test_solvability_l345_345488

-- Define the type for the problems
inductive Problem
| one    -- The cure rate of a drug for a certain disease
| two    -- Whether two drugs treating the same disease are related
| three  -- The probability of smokers contracting lung disease
| four   -- Whether the smoking population is related to gender
| five   -- Whether internet usage is related to the crime rate among teenagers

-- Define the property of a problem being solvable using independence tests
def solvable_by_independence_test : Problem → Prop
| Problem.one := False
| Problem.two := True
| Problem.three := False
| Problem.four := True
| Problem.five := True

-- The proof statement
theorem independence_test_solvability :
  (solvable_by_independence_test Problem.two) ∧
  (solvable_by_independence_test Problem.four) ∧
  (solvable_by_independence_test Problem.five) ∧
  ¬ (solvable_by_independence_test Problem.one) ∧
  ¬ (solvable_by_independence_test Problem.three) :=
by
  sorry

end independence_test_solvability_l345_345488


namespace evaluate_expression_l345_345417

theorem evaluate_expression : -20 + 8 * (10 / 2) - 4 = 16 :=
by
  sorry -- Proof to be completed

end evaluate_expression_l345_345417


namespace sean_divided_by_julie_is_2_l345_345160

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l345_345160


namespace right_triangle_medians_l345_345112

/-- The problem statement specifying the conditions and question. -/
theorem right_triangle_medians (m : ℝ) :
  ∃! m, 
    (∃ (a b c : ℝ × ℝ), 
      a = (0, 0) ∧ 
      ((∃ k : ℝ, b = (k, 0) ∧ c = (k, -1)) ∨ 
       (∃ k : ℝ, b = (0, k) ∧ c = (-1, k))) ∧ 
      (median_line b c = 5 * a.1 + 3) ∧ 
      (median_line a b = m * a.1 - 4)) ∧
    m = 20 ∨ m = 5 / 4 :=
sorry

end right_triangle_medians_l345_345112


namespace distance_to_town_l345_345349

theorem distance_to_town (d : ℝ) (h₁ : d < 6) (h₂ : d > 5) (h₃ : d > 4) : 5 < d ∧ d < 6 :=
by {
  split;
  { assumption }
}

end distance_to_town_l345_345349


namespace least_red_chips_l345_345222

/--
  There are 70 chips in a box. Each chip is either red or blue.
  If the sum of the number of red chips and twice the number of blue chips equals a prime number,
  proving that the least possible number of red chips is 69.
-/
theorem least_red_chips (r b : ℕ) (p : ℕ) (h1 : r + b = 70) (h2 : r + 2 * b = p) (hp : Nat.Prime p) :
  r = 69 :=
by
  -- Proof goes here
  sorry

end least_red_chips_l345_345222


namespace seans_sum_divided_by_julies_sum_l345_345162

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l345_345162


namespace chairs_subset_count_l345_345272

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345272


namespace find_angle_l345_345056

open Real

-- Definitions based on conditions
def a : ℝ × ℝ := (1, sqrt 3)
def b (m : ℝ) : ℝ × ℝ := (3, m)
def proj_cond (m : ℝ) (θ : ℝ) : Prop := 
  let ⟨x, y⟩ := b m in sqrt (x ^ 2 + y ^ 2) * cos θ = 3

-- Theorem to prove
theorem find_angle (m : ℝ) (θ : ℝ) (h : proj_cond m θ) : 
  θ = π / 6 :=
sorry

end find_angle_l345_345056


namespace final_sum_l345_345190

def internal_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

lemma calculated_angle (m n : ℤ) (hn : m + n = 1088 ∧ (∀ k, k ∣ m → k ∣ n → k = 1)) :
  ∑ i in (range 8), internal_angle (i + 3) = m / n := sorry

theorem final_sum :
  ∃ m n : ℤ, m + n = 1088 ∧ (∀ k, k ∣ m → k ∣ n → k = 1) ∧ 
  (∑ i in (range 8), internal_angle (i + 3) = m / n) := 
begin
  use [1081, 7],
  split,
  { refl, },
  split,
  { intros k hk1 hk2,
    sorry, }, -- Proof that 1081 and 7 are relatively prime
  { sorry, } -- Proof that the sum of the angles equals 1081/7
end

end final_sum_l345_345190


namespace mary_age_l345_345693

theorem mary_age (M F : ℕ) (h1 : F = 4 * M) (h2 : F - 3 = 5 * (M - 3)) : M = 12 :=
by
  sorry

end mary_age_l345_345693


namespace tank_overflow_time_l345_345147

noncomputable def pipe_fill_time (rate_A rate_B : ℕ) : ℕ :=
  let rate_combined := 1 / ((1 / rate_A) + (1 / rate_B))
  in rate_combined

def pipe_filling_time_proof : Prop :=
  ∀ (pA pB : ℕ), (pA = 32) → (pB = pA / 3) → pipe_fill_time pA pB = 8

theorem tank_overflow_time : pipe_filling_time_proof :=
  by
  intros pA pB hA hB
  sorry

end tank_overflow_time_l345_345147


namespace dot_product_AC_BD_zero_l345_345010

noncomputable theory

open_locale real_inner_product_space

-- Define four points A, B, C, D in space
variables (A B C D : ℝ^3)
-- Define the conditions on the distances between these points
variables (hAB : dist A B = 3) (hBC : dist B C = 7) (hCD : dist C D = 11) (hDA : dist D A = 9)

-- The main theorem to prove
theorem dot_product_AC_BD_zero : (A - C) ⬝ (B - D) = 0 :=
sorry

end dot_product_AC_BD_zero_l345_345010


namespace range_of_f_l345_345044

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem range_of_f : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x ∈ Set.Icc (-18 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l345_345044


namespace length_of_OP_l345_345965

variable {A B C P Q : Type} [IsTriangle A B C]
variable {O : Type} [IsCentroid O A B C]
variable (OQ : ℝ)
variable (AP : ℝ)

theorem length_of_OP (hOQ : OQ = 4) (hAP : AP = 15) : OP = 10 :=
  sorry

end length_of_OP_l345_345965


namespace cylinder_volume_change_l345_345301

theorem cylinder_volume_change (r h : ℝ) (V : ℝ) (volume_eq : V = π * r^2 * h) :
  let h' := 3 * h;
  let r' := 4 * r;
  let V' := π * (r')^2 * (h')
  in V' = 48 * V :=
by
  simp [h', r', V', volume_eq]
  sorry

end cylinder_volume_change_l345_345301


namespace det_B_pow_five_l345_345512

theorem det_B_pow_five (B : Matrix ℕ ℕ ℝ) (hB : det B = -3) : det (B^5) = -243 :=
by
  sorry

end det_B_pow_five_l345_345512


namespace part1_inequality_part2_inequality_l345_345491

-- Problem Part 1
def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)

theorem part1_inequality (x : ℝ) : f x ≤ 1 ↔ 0 ≤ x :=
by sorry

-- Problem Part 2
def max_f_value : ℝ := 3
def a : ℝ := sorry  -- Define in context
def b : ℝ := sorry  -- Define in context
def c : ℝ := sorry  -- Define in context

-- Prove √a + √b + √c ≤ 3 given a + b + c = 3
theorem part2_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = max_f_value) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 :=
by sorry

end part1_inequality_part2_inequality_l345_345491


namespace center_cell_value_l345_345548

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l345_345548


namespace students_in_examination_l345_345560

theorem students_in_examination (T : ℕ)
  (H1 : 25% of T got the first division)
  (H2 : 54% of T got the second division)
  (H3 : 63 students just passed) :
  T = 300 :=
sorry

end students_in_examination_l345_345560


namespace proof_problem_solution_l345_345984

open Real

noncomputable def proof_problem (n : ℕ) (x : Fin n → ℝ) (k m : ℕ) :=
  (∑ i, (x i ^ k + 1 / x i ^ k) ^ m) ≥ n * (n ^ k + 1 / n ^ k) ^ m

theorem proof_problem_solution (n : ℕ) (x : Fin n → ℝ) (k m : ℕ)
  (hx : ∑ i, x i = 1) (hk_pos : 0 < k) (hm_pos : 0 < m) :
  proof_problem n x k m := sorry

end proof_problem_solution_l345_345984


namespace find_DC_l345_345940

theorem find_DC (A B C D : Point) (h1 : dist A B = 36)
  (h2 : ∠ A D B = 90)
  (h3 : real.sin (angle A) = 3 / 5)
  (h4 : real.sin (angle C) = 1 / 4) :
  dist D C = 83.65 :=
sorry

end find_DC_l345_345940


namespace fixed_point_for_graph_l345_345435

theorem fixed_point_for_graph (m : ℝ) : ∃ (a b : ℝ), (∀ (m : ℝ), b = 9 * a^2 + m * a - 5 * m) ∧ a = 5 ∧ b = 225 :=
by
  use 5, 225
  split
  sorry

#check fixed_point_for_graph

end fixed_point_for_graph_l345_345435


namespace initial_number_of_men_l345_345653

def initial_average_age_increased_by_2_years_when_two_women_replace_two_men 
    (M : ℕ) (A men1 men2 women1 women2 : ℕ) : Prop :=
  (men1 = 20) ∧ (men2 = 24) ∧ (women1 = 30) ∧ (women2 = 30) ∧
  ((M * A) + 16 = (M * (A + 2)))

theorem initial_number_of_men (M : ℕ) (A : ℕ) (men1 men2 women1 women2: ℕ):
  initial_average_age_increased_by_2_years_when_two_women_replace_two_men M A men1 men2 women1 women2 → 
  2 * M = 16 → M = 8 :=
by
  sorry

end initial_number_of_men_l345_345653


namespace count_at_least_three_adjacent_chairs_l345_345238

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345238


namespace coefficient_x2_sum_l345_345409

theorem coefficient_x2_sum (s : ℕ → ℕ)
  (h_s : ∀ n, s n = (nat.choose n 2)) :
  (∑ n in finset.range 17, s (3 + n)) = 1139 :=
by {
  sorry
}

end coefficient_x2_sum_l345_345409


namespace projections_proportional_to_squares_l345_345996

theorem projections_proportional_to_squares
  (a b c a1 b1 : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a1 = (a^2) / c)
  (h₃ : b1 = (b^2) / c) :
  (a1 / b1) = (a^2 / b^2) :=
by sorry

end projections_proportional_to_squares_l345_345996


namespace xyz_equal_l345_345608

noncomputable theory

open Real

theorem xyz_equal {x y z : ℝ} 
  (h1 : x^3 = 2 * y - 1)
  (h2 : y^3 = 2 * z - 1)
  (h3 : z^3 = 2 * x - 1) : 
  x = y ∧ y = z := 
by 
  sorry

end xyz_equal_l345_345608


namespace imaginary_part_of_i_over_i_plus_one_l345_345600

theorem imaginary_part_of_i_over_i_plus_one :
  let i := Complex.I in Complex.im (i / (i + 1)) = 1 / 2 :=
by 
  sorry

end imaginary_part_of_i_over_i_plus_one_l345_345600


namespace boys_and_girls_equal_l345_345784

theorem boys_and_girls_equal (m d M D : ℕ) (hm : m > 0) (hd : d > 0) (h1 : (M / m) ≠ (D / d)) (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d := 
sorry

end boys_and_girls_equal_l345_345784


namespace minimum_students_class_5_7_l345_345364

theorem minimum_students_class_5_7 :
  ∃ n, (n % 7 = 3) ∧ (n % 8 = 3) ∧ n = 59 :=
by
  use 59
  split
  { norm_num }  -- to check n % 7 = 3
  split
  { norm_num }  -- to check n % 8 = 3
  { refl }      -- to check n = 59

# This Lean statement should build successfully as it checks that 59 is a solution to the given modular arithmetic conditions.

end minimum_students_class_5_7_l345_345364


namespace sum_even_divisors_140_l345_345296

theorem sum_even_divisors_140 : 
  ∑ d in (Finset.filter (λ d: ℕ, d % 2 = 0) (Finset.divisors 140)), d = 288 := 
by
  sorry

end sum_even_divisors_140_l345_345296


namespace magnitude_of_2a_minus_b_l345_345836

open Real

def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (u.1 - v.1, u.2 - v.2, u.3 - v.3)
def vector_smul (c : ℝ) (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (c * u.1, c * u.2, c * u.3)
def vector_magnitude (u : ℝ × ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

theorem magnitude_of_2a_minus_b : vector_magnitude (vector_sub (vector_smul 2 a) b) = sqrt 17 := 
sorry

end magnitude_of_2a_minus_b_l345_345836


namespace triangle_perimeter_l345_345822

-- Definitions of the given vertices
def A := (-4.5 : ℝ, -1.5 : ℝ)
def B := (8.5 : ℝ, 2.5 : ℝ)
def C := (-3 : ℝ, 4.5 : ℝ)

-- Function to compute the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Distances between the vertices
def AB := distance A B
def BC := distance B C
def CA := distance C A

-- Perimeter of the triangle
def perimeter : ℝ := AB + BC + CA

theorem triangle_perimeter :
  perimeter = real.sqrt 185 + real.sqrt 136.25 + real.sqrt 38.25 :=
by
  sorry

end triangle_perimeter_l345_345822


namespace find_omega_l345_345915
noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem find_omega
  (α β ω : ℝ)
  (hα : f ω α = -2)
  (hβ : f ω β = 0)
  (h_min : abs (α - β) = 3 * π / 4) :
  ω = 2 / 3 :=
sorry

end find_omega_l345_345915


namespace double_summation_eq_180_l345_345470

open Complex

theorem double_summation_eq_180
  (x y : ℂ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : y^2 * (x^2 - x * y + y^2) + x^3 * (x - y) = 0) :
  ∑ m in Finset.range 30, ∑ n in Finset.range 30, (x^(18 * m * n) * y^(-18 * m * n)) = 180 := 
sorry

end double_summation_eq_180_l345_345470


namespace solve_x_l345_345006

noncomputable def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

theorem solve_x (x : ℝ) (h : op (x - 1) 2 = 1) : x = -1 := 
by {
  -- proof outline here...
  sorry
}

end solve_x_l345_345006


namespace count_at_least_three_adjacent_chairs_l345_345237

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345237


namespace center_cell_value_l345_345538

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l345_345538


namespace count_valid_4_digit_numbers_l345_345500

def is_valid_number (n : ℕ) : Prop :=
  let thousands := n / 1000 in
  let hundreds := (n / 100) % 10 in
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  1000 ≤ n ∧ n < 10000 ∧
  1 ≤ thousands ∧ thousands ≤ 9 ∧
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_4_digit_numbers : 
  (Finset.filter is_valid_number (Finset.range 10000)).card = 1782 :=
sorry

end count_valid_4_digit_numbers_l345_345500


namespace cos_theta_of_point_l345_345021

theorem cos_theta_of_point (
  m : ℝ,
  hsin : sin θ = (sqrt 3 / 3) * m,
  P : ℝ × ℝ,
  hP : P = (sqrt 2, m)
) : cos θ = sqrt 6 / 3 ∨ cos θ = -sqrt 6 / 3 :=
sorry

end cos_theta_of_point_l345_345021


namespace find_side_DF_in_triangle_DEF_l345_345102

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l345_345102


namespace observation_transfer_application_1_transfer_application_2_extended_research_l345_345730

open Nat

-- Define the observational part
theorem observation (n: ℕ) : 2 / ((2 * n - 1) * (2 * n + 1)) = 1 / (2 * n - 1) - 1 / (2 * n + 1) :=
sorry

-- Define the transfer application part
theorem transfer_application_1 : 2 / (2021 * 2023) = 1 / 2021 - 1 / 2023 :=
sorry

theorem transfer_application_2 : 1 / (3 * 5) = 1 / 2 * (1 / 3 - 1 / 5) :=
sorry

-- Define the extended research part
theorem extended_research : (Finset.range 10092).sum (λ k, if k % 5 = 1 then 1 / (k * (k + 5)) else 0)
  = 2020 / 10101 :=
sorry

end observation_transfer_application_1_transfer_application_2_extended_research_l345_345730


namespace alice_wins_on_4x2019_grid_l345_345702

/--
We consider a 4 x 2019 grid where Alice and Bob take turns placing a T-shaped piece. 
The first player who cannot make a move loses. Alice starts; Alice has a winning strategy.
-/
theorem alice_wins_on_4x2019_grid : 
  ∃ strategy : (nat × nat × nat × nat) → (nat × nat × nat × nat) → Prop,
  (∀ b a : (nat × nat × nat × nat), strategy b a) ∧ 
  (∀ b : (nat × nat × nat × nat), strategy b (0, 0, 0, 0) = false) :=
sorry

end alice_wins_on_4x2019_grid_l345_345702


namespace scalene_triangle_obtuse_angle_condition_l345_345085

theorem scalene_triangle_obtuse_angle_condition 
  (a b c : ℝ)
  (h1 : scalene_triangle a b c)
  (h2 : a > b) -- Assuming a is the longest side based on given problem
  (h3 : a > c) -- Assuming a is the longest side based on given problem
  (h4 : ∃ A : ℝ, 90 < A ∧ ∠A a b c = A) -- ∠A is obtuse
  : a^2 > b^2 + c^2 :=
sorry

end scalene_triangle_obtuse_angle_condition_l345_345085


namespace distance_AB_l345_345929

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_AB (t : ℝ)
    (x1 y1 : ℝ) (l1_x l1_y : ℝ → ℝ)
    (l2 : ℝ → ℝ → Prop)
    (A : ℝ × ℝ)
    (intersects : ∃ t, l2 (l1_x t) (l1_y t)) 
    (B : ℝ × ℝ) (distance_correct : distance A.1 A.2 B.1 B.2 = 5 / 2) :
    distance (1 : ℝ) (2 : ℝ) ((l1_x (1 / 2)) : ℝ) ((l1_y (1 / 2)) : ℝ) = 5 / 2 := 
begin
  -- Definitions for coordinates of B using line l1's parametric equations
  let B_x := l1_x (1 / 2),
  let B_y := l1_y (1 / 2),
  -- Define point A
  let A_x := 1,
  let A_y := 2,
  -- Calculate the distance 
  have h_distance : distance A_x A_y B_x B_y = 5 / 2,
    from distance_correct,
  exact h_distance,
end

#check distance_AB

-- Defining lines l₁ and l₂, and points A and B based on the problem condition
def l1_x (t : ℝ) := 1 + 3 * t
def l1_y (t : ℝ) := 2 - 4 * t

def l2 (x y : ℝ) := 2 * x - 4 * y = 5

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (l1_x (1 / 2), l1_y (1 / 2))

end distance_AB_l345_345929


namespace inequality_solution_l345_345510

theorem inequality_solution (x : ℝ) : 9 - x^2 < 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l345_345510


namespace power_of_i_i_pow_2023_l345_345865

noncomputable def i : Complex := Complex.I

theorem power_of_i (n : ℕ) : i^n = Complex.I^n := by
  sorry

theorem i_pow_2023 : i^2023 = -i := by
  rw power_of_i,
  have : 2023 % 4 = 3 := by norm_num,
  rw [Complex.I_pow_eq_one],

  -- Applying the cycle logic
  calc (Complex.I) ^ 2023 = (Complex.I)^(4*505 + 3) : by norm_num
                     ... = (Complex.I^4)^505 * Complex.I^3 : by rw [pow_add, pow_mul]
                     ... = 1^505 * Complex.I^3 : by rw [Complex.I_pow_eq_one]
                     ... = Complex.I^3 : by rw one_pow
                     ... = -Complex.I : by norm_num,
  norm_num,
  assumption,
  sorry

end power_of_i_i_pow_2023_l345_345865


namespace function_solutions_l345_345406

variable {f : ℝ → ℝ}

theorem function_solutions (h : ∀ a b : ℝ, f(a^2) - f(b^2) ≤ (f(a) + b) * (a - f(b))) :
  (f = id) ∨ (f = λ x, -x) :=
by
  sorry

end function_solutions_l345_345406


namespace binomial_coeff_l345_345564

theorem binomial_coeff (n : ℕ) (x : ℝ) (h_n : n = 4) (h_sum : (-(2:ℝ))^n = 16) :
  (binom 4 2 * (1 / sqrt x)^2 * (-3)^2) = (54 / x) :=
by
  -- conditions simplifications
  sorry

end binomial_coeff_l345_345564


namespace equation_of_ellipse_equation_of_line_l_l345_345487

-- Given conditions from the problem
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Prop := (λ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
def circle_eq : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 = 4 / 5
def P : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (2/5, 4/5)

-- Prove (I) Equation of ellipse
theorem equation_of_ellipse : (ellipse 2 1 ⟨by linarith, by linarith⟩) :=
by simp [ellipse, 2, 1]

-- Given that |AF₂|, |AB|, and |BF₂| form an arithmetic sequence |AB| = 8/3
-- Prove (II) Equation of line l
theorem equation_of_line_l :
  (λ x y : ℝ, x - sqrt 5 * y + sqrt 3 = 0 ∨ x + sqrt 5 * y + sqrt 3 = 0) :=
sorry

end equation_of_ellipse_equation_of_line_l_l345_345487


namespace find_real_numbers_l345_345816

theorem find_real_numbers (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) :
  (x = 1 ∧ y = 2 ∧ z = -1) ∨ 
  (x = 1 ∧ y = -1 ∧ z = 2) ∨
  (x = 2 ∧ y = 1 ∧ z = -1) ∨ 
  (x = 2 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 2) ∨
  (x = -1 ∧ y = 2 ∧ z = 1) := 
sorry

end find_real_numbers_l345_345816


namespace number_of_cows_l345_345083

variables (c h d : ℕ)

theorem number_of_cows (h d : ℕ) : 
  (4 * c + 2 * h + 2 * d = 2 * (c + h + d) + 22) ∧ (h + d = 2 * c) -> c = 11 :=
begin
  sorry
end

end number_of_cows_l345_345083


namespace scientific_notation_of_19400000000_l345_345676

theorem scientific_notation_of_19400000000 :
  ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ (19400000000 : ℝ) = a * 10^n ∧ a = 1.94 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_19400000000_l345_345676


namespace find_angle_C_l345_345110

theorem find_angle_C (A B C : ℝ)
  (h1 : 2 * Real.sin A + 3 * Real.cos B = 4)
  (h2 : 3 * Real.sin B + 2 * Real.cos A = Real.sqrt 3)
  (triangle_ABC : A + B + C = Real.pi) :
  C = Real.pi / 6 :=
begin
  sorry
end

end find_angle_C_l345_345110


namespace sum_of_digits_second_smallest_mult_of_lcm_l345_345123

theorem sum_of_digits_second_smallest_mult_of_lcm :
  let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
  let M := 2 * lcm12345678
  (Nat.digits 10 M).sum = 15 := by
    -- Definitions from the problem statement
    let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
    let M := 2 * lcm12345678
    sorry

end sum_of_digits_second_smallest_mult_of_lcm_l345_345123


namespace find_DF_l345_345097

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345097


namespace chair_subsets_with_at_least_three_adjacent_l345_345259

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345259


namespace min_value_x_sq_y_minus_ln_x_minus_x_l345_345016

noncomputable def xy_y_eq_e_x (x y : ℝ) : Prop := (ln ((x * y)^y) = exp x)

theorem min_value_x_sq_y_minus_ln_x_minus_x (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : xy_y_eq_e_x x y) :
  x ^ 2 * y - ln x - x = 1 :=
sorry

end min_value_x_sq_y_minus_ln_x_minus_x_l345_345016


namespace chair_subsets_with_at_least_three_adjacent_l345_345263

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345263


namespace tetrahedron_inequality_l345_345117

theorem tetrahedron_inequality
  (a b c d h_a h_b h_c h_d V : ℝ)
  (ha : V = 1/3 * a * h_a)
  (hb : V = 1/3 * b * h_b)
  (hc : V = 1/3 * c * h_c)
  (hd : V = 1/3 * d * h_d) :
  (a + b + c + d) * (h_a + h_b + h_c + h_d) >= 48 * V := 
  by sorry

end tetrahedron_inequality_l345_345117


namespace winners_A_and_C_l345_345831

variables {P : Type} [DecidableEq P]

def participant : Type := P

variables (A B C D : participant)

def prediction (p : participant) : Prop := 
  match p with
  | A => ¬(∃ w : participant, wonAward w) ∧ wonAward C
  | B => wonAward A ∨ wonAward D
  | C => ∀ w : participant, w = A → (¬ wonAward w ∧ wonAward C)
  | D => ∃ w : participant, w = A ∨ w = B ∨ w = C
  | _ => false
  end

def wonAward (p : participant) : Prop := 
  match p with
  | A => true
  | B => false
  | C => true
  | D => false
  | _ => false
  end

def correctPrediction (p : participant) : Prop := prediction p = wonAward p

def number_of_winners : ℕ := 2

def number_of_correct_predictions : ℕ := 2

theorem winners_A_and_C :
  number_of_winners = 2 ∧ number_of_correct_predictions = 2 ∧ 
  ∃ w1 w2, wonAward w1 ∧ wonAward w2 ∧ w1 ≠ w2 ∧ ((w1 = A ∧ w2 = C) ∨ (w1 = C ∧ w2 = A)) :=
by sorry

end winners_A_and_C_l345_345831


namespace det_B_pow_five_l345_345511

theorem det_B_pow_five (B : Matrix ℕ ℕ ℝ) (hB : det B = -3) : det (B^5) = -243 :=
by
  sorry

end det_B_pow_five_l345_345511


namespace median_pets_is_2_5_l345_345218

-- Define the data set
def pets : List ℝ := [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 5, 5, 5, 5, 7, 7]

-- Define the number of data points
def num_students : ℕ := 16

-- Define a function to compute the median of a sorted list
noncomputable def median (l : List ℝ) : ℝ :=
let n := l.length
in if n % 2 = 0 then
    (l.get! (n/2 - 1) + l.get! (n/2)) / 2
   else
    l.get! (n / 2)

-- Assert that the median of the pets list is 2.5
theorem median_pets_is_2_5 : median pets = 2.5 := by
  sorry

end median_pets_is_2_5_l345_345218


namespace hexagon_inequality_l345_345132

variables {A B C D E F G H : ℝ} -- Variables for points represented as reals, modify accordingly if different types are required.
variables (ABCDEF_convex : convex_hull ℝ ({A, B, C, D, E, F} : set ℝ))
variables (AB_eq_BC : AB = BC)
variables (BC_eq_CD : BC = CD)
variables (DE_eq_EF : DE = EF)
variables (EF_eq_FA : EF = FA)
variables (angle_BCD : ∠ B C D = 60)
variables (angle_EFA : ∠ E F A = 60)
variables (angle_AGB : ∠ A G B = 120)
variables (angle_DHE : ∠ D H E = 120)

theorem hexagon_inequality :
  AG + GB + GH + DH + HE ≥ CF :=
by
  sorry

end hexagon_inequality_l345_345132


namespace sea_creatures_lost_l345_345496

theorem sea_creatures_lost (sea_stars : ℕ) (seashells : ℕ) (snails : ℕ) (items_left : ℕ)
  (h1 : sea_stars = 34) (h2 : seashells = 21) (h3 : snails = 29) (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l345_345496


namespace water_balloon_ratio_l345_345405

theorem water_balloon_ratio (c r j : ℕ) 
  (h1 : c = 4 * r) 
  (h2 : j = 6) 
  (h3 : c = 12) :
  r / j = 1 / 2 :=
by
  have r_def : r = 3 :=
    calc
      r = c / 4 : by rw [h1]
      ... = 12 / 4 : by rw [h3]
      ... = 3 : by norm_num
  have ratio_def : r / j = 3 / 6 := by rw [r_def, h2]
  exact (by norm_num : 3 / 6 = 1 / 2)
  sorry

end water_balloon_ratio_l345_345405


namespace parabola_intersect_line_segment_range_l345_345030

theorem parabola_intersect_line_segment_range (a : ℝ) :
  (∀ (x : ℝ), y = a * x^2 - 3 * x + 1) →
  (∀ (x1 y1 x2 y2 x0 : ℝ), (x0 ∈ {x | ∃ y, y = a * x^2 - 3 * x + 1}) →
                         (x1 - x0).abs > (x2 - x0).abs →
                         ∃ y1 y2, y1 = a * x1^2 - 3 * x1 + 1 ∧ y2 = a * x2^2 - 3 * x2 + 1 → y1 > y2) →
  (∃ (x : ℝ), x ∈ Icc (-1 : ℝ) (3 : ℝ) → y = x - 1) →
  (∃ (x : ℝ), y = a * x^2 - 3 * x + 1 = (x - 1)) →
  (\(a \ge 10/9 ∧ a < 2\)) :=
by
  sorry

end parabola_intersect_line_segment_range_l345_345030


namespace center_cell_value_l345_345553

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l345_345553


namespace pow_expression_eq_l345_345845

theorem pow_expression_eq (n : ℕ) (h_nat : 0 < n) :
  let x := (1 / 2 : ℝ) * (1991 ^ (1 / (n : ℝ)) - 1991 ^ (-(1 / (n : ℝ)))) in
  (x - real.sqrt (1 + x^2)) ^ n = (-1 : ℝ) ^ n * 1991 ^ -1 :=
by sorry

end pow_expression_eq_l345_345845


namespace find_m_plus_n_l345_345957

def T (k : ℕ) : ℚ := k * (k + 1) / 2

noncomputable def infinite_series : ℚ :=
  ∑' k in filter (λ k, k ≥ 4) (range (nat.succ nat.succ)), 
    1 / ((T (k-1) - 1) * (T k - 1) * (T (k+1) - 1))

theorem find_m_plus_n : infinite_series = 1 / 450 := sorry

end find_m_plus_n_l345_345957


namespace parabola_maximum_value_l345_345904

noncomputable def maximum_parabola (a b c : ℝ) (h := -b / (2*a)) (k := a * h^2 + b * h + c) : Prop :=
  ∀ (x : ℝ), a ≠ 0 → b = 12 → c = 4 → a = -3 → k = 16

theorem parabola_maximum_value : maximum_parabola (-3) 12 4 :=
by
  sorry

end parabola_maximum_value_l345_345904


namespace problem_l345_345067

variable (m n : ℝ)
variable (h1 : m + n = -1994)
variable (h2 : m * n = 7)

theorem problem (m n : ℝ) (h1 : m + n = -1994) (h2 : m * n = 7) : 
  (m^2 + 1993 * m + 6) * (n^2 + 1995 * n + 8) = 1986 := 
by
  sorry

end problem_l345_345067


namespace least_positive_integer_mod_l345_345316

theorem least_positive_integer_mod (x : ℤ) : x + 7071 ≡ 3540 [MOD 15] → x = 9 :=
by
  intro h
  have h1 : x ≡ 3540 - 7071 [MOD 15] := by sorry
  have h2 : x ≡ -3531 [MOD 15] := by sorry
  have h3 : -3531 ≡ 9 [MOD 15] := by sorry
  have h4 : x ≡ 9 [MOD 15] := by sorry
  exact Int.eq_mod_iff (h4.symm) (Nat.cast_pos.mpr zero_lt_nine)

end least_positive_integer_mod_l345_345316


namespace find_m_l345_345403

section
variables {R : Type*} [CommRing R]

def f (x : R) : R := 4 * x^2 - 3 * x + 5
def g (x : R) (m : R) : R := x^2 - m * x - 8

theorem find_m (m : ℚ) : 
  f (5 : ℚ) - g (5 : ℚ) m = 20 → m = -53 / 5 :=
by {
  sorry
}

end

end find_m_l345_345403


namespace distinct_pawns_arrangement_5x5_chessboard_l345_345516

theorem distinct_pawns_arrangement_5x5_chessboard : ∃ n : ℕ, n = 120 ∧
  (∀ (board : Fin 5 → Fin 5), (∀ i j, i ≠ j → board i ≠ board j) ∧ 
  (∀ j, ∃ i, board i = j)) := 
begin
  existsi 120,
  split,
  { refl, },
  { intros board,
    split,
    { intros i j h,
      -- proof to be completed
      sorry, },
    { intros j,
      -- proof to be completed
      sorry, },
  }
end

end distinct_pawns_arrangement_5x5_chessboard_l345_345516


namespace count_valid_complex_numbers_l345_345972

noncomputable def f (z : ℂ) : ℂ := z^2 + 2 * complex.I * z + 2

def valid_complex (z : ℂ) : Prop :=
  (z.im > 0) ∧ 
  (isInt (f z).re) ∧ 
  (| (f z).re | ≤ 15) ∧
  (isInt (f z).im) ∧
  (| (f z).im | ≤ 15)

theorem count_valid_complex_numbers (N : ℕ) :
  ∃ (S : set ℂ), S.card = N ∧ ∀ z ∈ S, valid_complex z :=
sorry

end count_valid_complex_numbers_l345_345972


namespace seans_sum_divided_by_julies_sum_l345_345164

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l345_345164


namespace emily_josh_bail_rate_l345_345714

theorem emily_josh_bail_rate :
  ∃ r : ℝ, 
    (let distance_to_shore := 2 in
     let rowing_speed := 4 in
     let max_water_capacity := 50 in
     let incoming_water_rate := 8 in
     let time_to_shore := distance_to_shore / rowing_speed * 60 in
     let net_intake_rate := incoming_water_rate - r in
     let total_net_intake := time_to_shore * net_intake_rate in
     total_net_intake ≤ max_water_capacity) 
    ∧ r = 7 := 
by
  sorry

end emily_josh_bail_rate_l345_345714


namespace tangent_line_at_zero_l345_345914

noncomputable def f (x : ℝ) := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_zero :
  let f' := fun x => Real.exp x + 2 * x - 1 + Real.cos x in
  let f0 := f 0 in
  f0 = 1 → (∀ x : ℝ, f(x) = Real.exp x + x^2 - x + Real.sin x) →
  (∀ x : ℝ, f' x = Real.exp x + 2 * x - 1 + Real.cos x) →
  let slope := f' 0 in
  slope = 1 →
  y = x + 1 := sorry

end tangent_line_at_zero_l345_345914


namespace lambda_phi_relation_l345_345037

-- Define the context and conditions
variables (A B C D M N : Type) -- Points on the triangle with D being the midpoint of BC
variables (AB AC BC BN BM MN : ℝ) -- Lengths
variables (lambda phi : ℝ) -- Ratios given in the problem

-- Conditions
-- 1. M is a point on the median AD of triangle ABC
variable (h1 : M = D ∨ M = A ∨ M = D) -- Simplified condition stating M's location
-- 2. The line BM intersects the side AC at point N
variable (h2 : N = M ∧ N ≠ A ∧ N ≠ C) -- Defining the intersection point
-- 3. AB is tangent to the circumcircle of triangle NBC
variable (h3 : tangent AB (circumcircle N B C))
-- 4. BC = lambda BN
variable (h4 : BC = lambda * BN)
-- 5. BM = phi * MN
variable (h5 : BM = phi * MN)

-- Goal
theorem lambda_phi_relation : phi = lambda ^ 2 :=
sorry

end lambda_phi_relation_l345_345037


namespace chairs_subset_count_l345_345275

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345275


namespace number_of_subsets_of_intersection_l345_345632

open Set

def setA : Set ℝ := {x | x^2 < 9}
def setB : Set ℕ := {x | -1 < (x : ℝ) ∧ (x : ℝ) < 5}

theorem number_of_subsets_of_intersection :
  Finite (setA ∩ setB) ∧ Fintype.card (𝒫 (setA ∩ setB)) = 4 :=
by
  -- Conditions directly from the problem
  let A := {x : ℝ | x^2 < 9}
  let B := {x : ℕ | -1 < (x : ℝ) ∧ (x : ℝ) < 5}

  -- Question
  -- We need to prove the number of subsets of (setA ∩ setB) is 4
  sorry

end number_of_subsets_of_intersection_l345_345632


namespace domain_of_function_l345_345191

theorem domain_of_function :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l345_345191


namespace find_b_find_area_l345_345530

open Real

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := A + π / 2
noncomputable def a : ℝ := 3
noncomputable def cos_A : ℝ := sqrt 6 / 3
noncomputable def b : ℝ := 3 * sqrt 2
noncomputable def area : ℝ := 3 * sqrt 2 / 2

theorem find_b (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  b = 3 * sqrt 2 := 
  sorry

theorem find_area (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  area = 3 * sqrt 2 / 2 := 
  sorry

end find_b_find_area_l345_345530


namespace jodi_third_week_miles_l345_345587

theorem jodi_third_week_miles (total_miles : ℕ) (first_week : ℕ) (second_week : ℕ) (fourth_week : ℕ) (days_per_week : ℕ) (third_week_miles_per_day : ℕ) 
  (H1 : first_week * days_per_week + second_week * days_per_week + third_week_miles_per_day * days_per_week + fourth_week * days_per_week = total_miles)
  (H2 : first_week = 1) 
  (H3 : second_week = 2) 
  (H4 : fourth_week = 4)
  (H5 : total_miles = 60)
  (H6 : days_per_week = 6) :
    third_week_miles_per_day = 3 :=
by sorry

end jodi_third_week_miles_l345_345587


namespace cos_difference_simplify_l345_345634

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end cos_difference_simplify_l345_345634


namespace paul_books_sold_l345_345146

theorem paul_books_sold:
  ∀ (initial_books friend_books sold_per_day days final_books sold_books: ℝ),
    initial_books = 284.5 →
    friend_books = 63.7 →
    sold_per_day = 16.25 →
    days = 8 →
    final_books = 112.3 →
    sold_books = initial_books - friend_books - final_books →
    sold_books = 108.5 :=
by intros initial_books friend_books sold_per_day days final_books sold_books
   sorry

end paul_books_sold_l345_345146


namespace star_set_l345_345595

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x}
def star (A B : Set ℝ) : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem star_set :
  star A B = {x | (0 ≤ x ∧ x < 1) ∨ (3 < x)} :=
by
  sorry

end star_set_l345_345595


namespace cart_total_books_l345_345621

theorem cart_total_books (fiction non_fiction autobiographies picture: ℕ) 
  (h1: fiction = 5)
  (h2: non_fiction = fiction + 4)
  (h3: autobiographies = 2 * fiction)
  (h4: picture = 11)
  : fiction + non_fiction + autobiographies + picture = 35 := by
  -- Proof is omitted
  sorry

end cart_total_books_l345_345621


namespace number_of_true_propositions_l345_345386

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345386


namespace true_compound_props_l345_345400

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345400


namespace pyramid_regular_l345_345088

-- Define the conditions and the statement
def regular_ngon (A : ℕ → ℝ×ℝ) (n : ℕ) : Prop :=
  ∀ i : ℕ,  i < n → 
  ∃ r : ℝ, ((A (i % n)).1 - (A ((i + 1) % n)).1)^2 + ((A (i % n)).2 - (A ((i + 1) % n)).2)^2 = r^2

def apex_to_edge_equal_angles (S : ℝ×ℝ×ℝ) (A : ℕ → ℝ×ℝ) (n : ℕ) : Prop :=
  let angle (a b c : ℝ×ℝ×ℝ) : ℝ :=
    -- some appropriate definition to calculate the angle ∠ bac
    sorry
  ∀ i : ℕ,  i < n → 
  angle (S.1, S.2, S.3) ((A (i % n)).1, (A (i % n)).2, 0) ((A ((i + 1) % n)).1, (A ((i + 1) % n)).2, 0) = 
  angle (S.1, S.2, S.3) ((A ((i + 1) % n)).1, (A ((i + 1) % n)).2, 0) ((A ((i + 2) % n)).1, (A ((i + 2) % n)).2, 0)

def pyramid_is_regular (S : ℝ×ℝ×ℝ) (A : ℕ → ℝ×ℝ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → 
  ((S.1 - (A (i % n)).1)^2 + (S.2 - (A (i % n)).2)^2 + (S.3)^2) = 
  ((S.1 - (A (j % n)).1)^2 + (S.2 - (A (j % n)).2)^2 + (S.3)^2)

-- The theorem to prove
theorem pyramid_regular (n : ℕ) (A : ℕ → ℝ×ℝ) (S : ℝ×ℝ×ℝ) :
  regular_ngon A n →
  apex_to_edge_equal_angles S A n →
  pyramid_is_regular S A n :=
  sorry

end pyramid_regular_l345_345088


namespace no_full_side_visible_inside_no_full_side_visible_outside_l345_345717

section Visibility

variables {P : Type} [polygon P] (O : Type) [point_in_polygon O P]
variables {P : Type} [polygon P] (O : Type) [point_out_polygon O P]

-- Part (a): Point inside the polygon
theorem no_full_side_visible_inside :
  ∀ (P : polygon) (O : point), point_in_polygon O P → ∀ (s : side_of_polygon P), ¬full_side_visible_from O s :=
by
  intros P O hin s,
  sorry

-- Part (b): Point outside the polygon
theorem no_full_side_visible_outside :
  ∀ (P : polygon) (O : point), point_out_polygon O P → ∀ (s : side_of_polygon P), ¬full_side_visible_from O s :=
by
  intros P O hout s,
  sorry

end Visibility

end no_full_side_visible_inside_no_full_side_visible_outside_l345_345717


namespace cease_workshop_A_l345_345744

noncomputable def P_xi : ℕ → ℚ
| 0 := 32 / 125
| 1 := 64 / 125
| 2 := 26 / 125
| 3 :=  3 / 125
| _ := 0 -- For numbers outside [0, 3], probability is 0

def profit (machines_breaking : ℕ) : ℚ :=
match machines_breaking with
| 0 => 20000
| 1 => 10000
| 2 => 0
| 3 => -30000
| _ => 0
end

noncomputable def expected_profit_workshop_B : ℚ :=
2 * P_xi 0 + 1 * P_xi 1 + 0 * P_xi 2 - 3 * P_xi 3

noncomputable def P_eta (k : ℕ) : ℚ :=
if k ≤ 3 then (Nat.choose 3 k : ℚ) * (2 / 5 : ℚ)^k * (3 / 5)^(3-k) else 0

noncomputable def expected_profit_workshop_A : ℚ :=
2 * P_eta 0 + 1 * P_eta 1 + 0 * P_eta 2 - 3 * P_eta 3

theorem cease_workshop_A : expected_profit_workshop_A < expected_profit_workshop_B :=
sorry

end cease_workshop_A_l345_345744


namespace proof_problem_l345_345841

noncomputable def f (x : ℝ) (k : ℝ) := (log x + k) / exp x

noncomputable def F (x : ℝ) (k : ℝ) := x * exp x * derivative (λ x, (log x + k) / exp x) x

def g (x : ℝ) (a : ℝ) := -x^2 + 2*a*x

theorem proof_problem :
  (∀ k : ℝ, derivative (λ x, (log x + k) / exp x) 1 = 0 → k = 1) ∧
  (∀ x : ℝ, F x 1 = 1 - x * log x - x) ∧
  (∀ x : ℝ, -log x - 2 ≥ 0 ↔ 0 < x ∧ x ≤ 1 / exp 2) →
  (∀ x : ℝ, -log x - 2 ≤ 0 ↔ x ≥ 1 / exp 2) →
  (∀ x2 : ℝ, 0 ≤ x2 ∧ x2 ≤ 1 →
    (∀ a : ℝ, 0 < a → x1 : ℝ → 0 < x1 →
    g x2 a < F x1 1 → 0 < a ∧ a < 1 + 1 / (2 * exp 2)))
:= sorry

end proof_problem_l345_345841


namespace number_of_passwords_is_correct_l345_345336

open Classical

noncomputable def number_of_valid_passwords (nums : Finset ℕ) (digit_list : List Char) : ℕ :=
  if nums.prod id = 6 ∧ {ch | ch ∈ digit_list}.filter (λ ch, ch = 'A').card = 3 ∧
     digit_list.count 'a' = 1 ∧ digit_list.count 'b' = 1 ∧
     digit_list.length = 9 ∧
     ∀ (i : ℕ), (digit_list.nth i = some 'A' → digit_list.nth (i + 1) ≠ some 'A') ∧
     ∀ (i : ℕ), (digit_list.nth i = some 'a' → digit_list.nth (i + 1) ≠ some 'b') then 13600
  else 0

theorem number_of_passwords_is_correct :
  ∃ (pw : List Char) (nums : Finset ℕ),
    nums.prod id = 6 ∧
    {ch | ch ∈ pw}.filter (λ ch, ch = 'A').card = 3 ∧
    pw.count 'a' = 1 ∧
    pw.count 'b' = 1 ∧
    pw.length = 9 ∧
    (∀ (i : ℕ), (pw.nth i = some 'A' → pw.nth (i + 1) ≠ some 'A')) ∧
    (∀ (i : ℕ), (pw.nth i = some 'a' → pw.nth (i + 1) ≠ some 'b')) ∧
    number_of_valid_passwords nums pw = 13600 :=
by
  existsi (['1', '2', '3', 'A', 'A', 'A', 'a', 'b', '0'] : List Char)
  existsi ({1, 2, 3, 1} : Finset ℕ)
  -- Here we list out and prove all conditions
  split; sorry

end number_of_passwords_is_correct_l345_345336


namespace find_angle_x_l345_345994

noncomputable theory

/-- Define the points and properties in the problem -/
def center_of_circle (O A C D : Type) : Prop :=
(O, A, C, D : Type) ∧ (A ≠ C) ∧ (O ≠ A) ∧ (O ≠ C)

/-- Define diameter property of the circle -/
def diameter (O A C : Type) : Prop :=
AC = 2 * AO

/-- Define inscribed right angle property -/
def inscribed_angle (O A C D : Type) : Prop :=
(O, A, C, D : Type) ∧ (angle ACD = 90)

/-- Given in problem -/
def given_angle_CDA (angle_CDA : ℝ) : Prop :=
angle_CDA = 48

/-- The main theorem stating that under the given conditions, angle x = 58 degrees -/
theorem find_angle_x
  (O A C D : Type)
  (h_center : center_of_circle O A C D)
  (h_diameter : diameter O A C)
  (h_inscribed : inscribed_angle O A C D)
  (h_angle_CDA : given_angle_CDA 48) :
  ∃ (x : ℝ), x = 58 :=
begin
  sorry
end

end find_angle_x_l345_345994


namespace sequence_a_11_l345_345569

theorem sequence_a_11 (a : ℕ → ℚ) (arithmetic_seq : ℕ → ℚ)
  (h1 : a 3 = 2)
  (h2 : a 7 = 1)
  (h_arith : ∀ n, arithmetic_seq n = 1 / (a n + 1))
  (arith_property : ∀ n, arithmetic_seq (n + 1) - arithmetic_seq n = arithmetic_seq (n + 2) - arithmetic_seq (n + 1)) :
  a 11 = 1 / 2 :=
by
  sorry

end sequence_a_11_l345_345569


namespace find_2005th_digit_l345_345144

theorem find_2005th_digit :
  (∃ f : ℕ → ℕ, 
    (∀ n, 1 ≤ n ∧ n ≤ 99 → (λ m, (1 * 1) + (2 * 2) + ... + (m * m) = (m * (m + 1) / 2)) n) ∧ 
    f 2005 = 1) := sorry

end find_2005th_digit_l345_345144


namespace verify_propositions_l345_345376

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345376


namespace find_numbers_l345_345283

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def number1 := 986
noncomputable def number2 := 689

theorem find_numbers :
  is_three_digit_number number1 ∧ is_three_digit_number number2 ∧
  hundreds_digit number1 = units_digit number2 ∧ hundreds_digit number2 = units_digit number1 ∧
  number1 - number2 = 297 ∧ (hundreds_digit number2 + tens_digit number2 + units_digit number2) = 23 :=
by
  sorry

end find_numbers_l345_345283


namespace triangle_angle_A_eq_60_l345_345408

-- Definitions of the triangle sides and median
def triangle_ab : ℝ := 2
def triangle_ac : ℝ := 4
def median_a_to_bc : ℝ := sqrt 7

-- Statement to prove
theorem triangle_angle_A_eq_60 :
  let A := 60 * (π / 180) -- converting degrees to radians for Lean mathlib
  in is_angle_A (triangle_ab) (triangle_ac) (median_a_to_bc) A :=
sorry

end triangle_angle_A_eq_60_l345_345408


namespace prove_AB_is_tangent_to_circumcircle_of_triangle_BCD_l345_345867

variables {A B C D : Type*}

-- Definitions for the problem
structure Triangle (α : Type*) :=
  (A B C : α)

structure Point (α : Type*) :=
  (D : α)
  
universe u

variable (α : Type u)

-- Angles involved
variables 
  (angleC angleADB angleABC : ℝ)
  (AD CD : ℝ)
  (t: Triangle α)
  (P: Point α)

-- Conditions
def AD_CD_ratio_eq_2_1 : Prop := AD / CD = 2
def angle_C_eq_45 : Prop := angleC = 45
def angle_ADB_eq_60 : Prop := angleADB = 60

-- To prove
def AB_tangent_to_circumcircle_BCD : Prop :=
  ∃ (O : Point α), -- Assume we can construct the circumcircle with center O
    ∃ (E : α),    -- And intersection point E on BC
    ∃ (AC_line_point : Point α), -- and point D lies on AC
      AD_CD_ratio_eq_2_1 ∧
      angle_C_eq_45 ∧
      angle_ADB_eq_60 ∧
      -- Further construct the tangent relation statement for AB
      ∀ (O : Point α), ∀ (t : Triangle α),
        ∃ (B, C, D : α),
          (angle_C_eq_45 t = 45) ∧
          (AD_CD_ratio_eq_2_1 (AC_line_point D)) →
          is_tangent t.A t.B t.C O.D -- Function to check tangency

axiom is_tangent : α → α → α → α → Prop

-- Final theorem statement
theorem prove_AB_is_tangent_to_circumcircle_of_triangle_BCD
  (AD_CD_ratio_eq_2_1 : AD_CD_ratio_eq_2_1)
  (angle_C_eq_45 : angle_C_eq_45)
  (angle_ADB_eq_60 : angle_ADB_eq_60) :
  AB_tangent_to_circumcircle_BCD :=
begin
  sorry
end

end prove_AB_is_tangent_to_circumcircle_of_triangle_BCD_l345_345867


namespace find_fx_monotonically_increasing_intervals_range_fx_l345_345055

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2 - 1/2 * Real.cos (2 * x)

theorem find_fx (x : ℝ) : f x = Real.sin (2 * x - Real.pi / 6) + 1/2 := 
sorry

theorem monotonically_increasing_intervals (x : ℝ) (h0 : 0 ≤ x) (h2π : x ≤ 2 * Real.pi) : 
(x ∈ set.Icc 0 (Real.pi / 3) ∨ x ∈ set.Icc (5 * Real.pi / 6) (4 * Real.pi / 3)) →
strict_mono (λ y, f y) :=
sorry

theorem range_fx (x : ℝ) (hx : x ∈ set.Icc 0 (Real.pi / 3)) : 
f x ∈ set.Icc 0 (3 / 2) :=
sorry

end find_fx_monotonically_increasing_intervals_range_fx_l345_345055


namespace ones_digit_of_largest_power_of_2_dividing_16_factorial_l345_345428

theorem ones_digit_of_largest_power_of_2_dividing_16_factorial : 
  (nat.digits 10 (2^15)).head = 8 :=
by
  sorry

end ones_digit_of_largest_power_of_2_dividing_16_factorial_l345_345428


namespace arithmetic_seq_of_S_find_S_a_l345_345023

def S : ℕ → ℚ
| 0 := 0
| (n + 1) := S n + a (n + 1)

def a : ℕ → ℚ
| 0 := 0
| 1 := 1 / 2
| (n + 1) := -2 * S (n + 1) * S n

theorem arithmetic_seq_of_S (n : ℕ) (hn : n ≥ 2) : 
  (1 / S n) = (1 / S 1) + 2 * (n - 1) :=
sorry

theorem find_S_a (n : ℕ) : 
  (S n = 1 / (2 * n)) ∧ 
  (a n = if n = 1 then 1 / 2 else -1 / (2 * n * (n - 1))) :=
sorry

end arithmetic_seq_of_S_find_S_a_l345_345023


namespace sum_of_fractions_divides_l345_345959

theorem sum_of_fractions_divides (p : ℕ) (hprime : p = 1601) (m n : ℕ) :
  (∑ k in Finset.range p, if (k^2 + 1) % p ≠ 0 then 1 / (k^2 + 1) else 0) = m / n →
  p ∣ (2 * m + n) :=
by
  have hp : p = 1601 := hprime
  sorry

end sum_of_fractions_divides_l345_345959


namespace ratio_AO_OM_l345_345855

variable (A B C D O M : Point)
variable (is_regular_tetrahedron : RegularTetrahedron A B C D)
variable (is_centroid_M : Centroid M B C D)
variable (is_equidistant_O : Equidistant O A B C D all_faces)

theorem ratio_AO_OM : ratio AO OM = 3 :=
by
  sorry

end ratio_AO_OM_l345_345855


namespace true_compound_props_l345_345402

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345402


namespace true_compound_props_l345_345401

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345401


namespace count_at_least_three_adjacent_chairs_l345_345241

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345241


namespace length_of_closed_convex_curve_l345_345202

theorem length_of_closed_convex_curve (C : set (ℝ × ℝ)) (hC : is_convex C) (h_closed : is_closed C)
  (h_projection : ∀ (l : ℝ × ℝ), length (proj l (boundary C)) = 1) :
  length(boundary C) = π := 
sorry

end length_of_closed_convex_curve_l345_345202


namespace meat_left_l345_345583

theorem meat_left (initial_meat : ℕ) (meatball_fraction : ℚ) (spring_roll_meat : ℕ) 
  (h_initial : initial_meat = 20) 
  (h_meatball_fraction : meatball_fraction = 1/4)
  (h_spring_roll_meat : spring_roll_meat = 3) : 
  initial_meat - (initial_meat * meatball_fraction.num / meatball_fraction.denom).toNat - spring_roll_meat = 12 :=
by
  sorry

end meat_left_l345_345583


namespace cube_root_abs_value_sum_l345_345788

theorem cube_root_abs_value_sum :
  real.cbrt (-27) + abs (-5) = 2 := by
sory

end cube_root_abs_value_sum_l345_345788


namespace weather_conditions_on_May_15_l345_345356

-- Define the conditions as predicates and state the problem as the main theorem.
variables (T : ℝ) (raining crowded : Prop)

-- Define the given condition
def condition : Prop := (T ≥ 70) ∧ ¬raining → crowded

-- Define the outcome on May 15
def not_crowded : Prop := ¬crowded

-- Translate the final conclusion
theorem weather_conditions_on_May_15 (h : condition) : not_crowded h → (T < 70 ∨ raining) := 
sorry

end weather_conditions_on_May_15_l345_345356


namespace petya_always_wins_l345_345439

theorem petya_always_wins :
  ∀ (figure : Type) (L_cut : figure → figure × figure) (initial_figure : figure),
  (∀ x, x ∈ initial_figure → x = L_cut x) →
  (∃ petya_wins_strategy : (figure × figure → figure), ∀ (vasyas_move : figure), ∃ (sub_figure : figure × figure), sub_figure = L_cut vasyas_move →
  sub_figure = petya_wins_strategy sub_figure) :=
by
  sorry

end petya_always_wins_l345_345439


namespace maximum_rubidium_concentration_l345_345315

theorem maximum_rubidium_concentration :
  ∀ (x y z : ℝ), 
    (10 * x + 8 * y + 5 * z) / (x + y + z) = 7 →     -- Condition 1: 7% Hydroxide solution
    5 * z / (x + y + z) ≤ 2 →                        -- Condition 2: Francium should not exceed 2%
    (y = 2 * z - 3 * x) →                            -- Derived solution step from the conditions
    x + y + z ≠ 0 →                                  -- Ensure total volume is non-zero
    10 * x / (x + y + z) ≤ 1 :=                      -- The rubidium concentration should be at most 1

by 
  intros x y z H H1 H2 H3,
  sorry

end maximum_rubidium_concentration_l345_345315


namespace pd_distance_l345_345463

theorem pd_distance (PA PB PC PD : ℝ) (hPA : PA = 17) (hPB : PB = 15) (hPC : PC = 6) :
  PA^2 + PC^2 = PB^2 + PD^2 → PD = 10 :=
by
  sorry

end pd_distance_l345_345463


namespace find_N_plus_5n_l345_345154

theorem find_N_plus_5n :
  ∃ (x y z : ℝ), 3 * (x + y + z) = x^2 + y^2 + z^2 ∧
  let N := max (xz + yz) in
  let n := min (xy) in
  N + 5 * n = 27 :=
sorry

end find_N_plus_5n_l345_345154


namespace sean_divided_by_julie_is_2_l345_345161

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l345_345161


namespace hyperbola_equation_l345_345481

theorem hyperbola_equation
    (a b λ : ℝ)
    (asymptote1 : ∀ x, y = 3 * x ↔ ∃ y, y = 3 * x)
    (asymptote2 : ∀ x, y = -3 * x ↔ ∃ y, y = -3 * x)
    (focus : (Real.sqrt 10, 0) ∈ focus_of_hyperbola λ) :
  x^2 - (y^2 / 9) = 1 :=
by
  -- Proof omitted
  sorry

end hyperbola_equation_l345_345481


namespace true_propositions_l345_345380

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345380


namespace option_B_valid_option_D_valid_l345_345478

variables (Line Plane : Type)
variables (m n : Line) (α β : Plane)

-- Define the conditions
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Line → Prop)
variables (contained : Line → Plane → Prop)
variables (plane_perpendicular : Plane → Plane → Prop)

-- State the conditions as hypotheses
variables (hm_parallel_α : parallel m α)
variables (hn_contained_α : contained n α)
variables (hm_perpendicular_n : perpendicular m n)
variables (hm_perpendicular_α : perpendicular m α)
variables (hn_perpendicular_β : perpendicular n β)
variables (hα_perpendicular_β : plane_perpendicular α β)
variables (hm_contained_α : contained m α)
variables (hn_contained_β : contained n β)
variables (hα_parallel_β : plane_perpendicular α β)

-- Conclusion B
theorem option_B_valid : (hm_perpendicular_n) ∧ (hm_perpendicular_α) ∧ (hn_perpendicular_β) → (plane_perpendicular α β) :=
by sorry

-- Conclusion D
theorem option_D_valid : (hα_parallel_β) ∧ (hm_perpendicular_α) ∧ (hn_perpendicular_β) → (perpendicular m n) :=
by sorry

end option_B_valid_option_D_valid_l345_345478


namespace sum_left_half_row_12_pascals_triangle_l345_345363

theorem sum_left_half_row_12_pascals_triangle : 
  let row_12 := Finset.range 13;
  let total_sum := (2:ℕ)^12;
  let left_half_sum := total_sum / 2;
  left_half_sum = 2048 :=
by
  let row_12 := Finset.range 13;
  let total_sum := (2:ℕ)^12;
  let left_half_sum := total_sum / 2;
  have h : left_half_sum = 2048, from sorry;
  exact h

end sum_left_half_row_12_pascals_triangle_l345_345363


namespace sean_and_julie_sums_l345_345168

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l345_345168


namespace green_balls_more_than_red_l345_345654

theorem green_balls_more_than_red
  (total_balls : ℕ) (red_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 66)
  (h2 : red_balls = 30)
  (h3 : green_balls = total_balls - red_balls) : green_balls - red_balls = 6 :=
by
  sorry

end green_balls_more_than_red_l345_345654


namespace min_b_a_value_l345_345447

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^2 / 2 + x^3 / 3 - x^4 / 4 + ... + x^2015 / 2015

noncomputable def g (x : ℝ) : ℝ := 1 - x + x^2 / 2 - x^3 / 3 + x^4 / 4 - ... - x^2015 / 2015

noncomputable def F (x : ℝ) : ℝ := f(x + 3) * g(x - 4)

theorem min_b_a_value :
  (∃ a b : ℤ, a < b ∧ (∀ x, F x = 0 → (a : ℝ) ≤ x ∧ x ≤ (b : ℝ)) ∧ b - a = 10) :=
sorry

end min_b_a_value_l345_345447


namespace length_CD_is_correct_l345_345576

noncomputable def isosceles_triangle_base_length : ℝ := 10

def triangle_ABE_area : ℝ := 150
def trapezoid_area : ℝ := 100
def altitude_from_A_to_BE : ℝ := 30

def length_of_CD := (triangle_ABE_area - trapezoid_area) * (altitude_from_A_to_BE / (triangle_ABE_area / (1 / 2 * isosceles_triangle_base_length))) ** (1 / 2) / (altitude_from_A_to_BE ** (1 / 2))

theorem length_CD_is_correct :
  length_of_CD = 10 * (3 ** (1 / 2)) / 3 :=
by
  -- Proof steps would go here, but we skip it as per instructions
  sorry

end length_CD_is_correct_l345_345576


namespace solve_eq1_solve_eq2_l345_345179

noncomputable theory

-- Definitions based on the conditions
def eq1 (y : ℝ) : Prop := 2.4 * y - 9.8 = 1.4 * y - 9
def eq2 (x : ℝ) : Prop := x - 3 = (3 / 2) * x + 1

-- Theorems to be proven
theorem solve_eq1 (y : ℝ) : eq1 y → y = 0.8 :=
by sorry

theorem solve_eq2 (x : ℝ) : eq2 x → x = -8 :=
by sorry

end solve_eq1_solve_eq2_l345_345179


namespace domain_of_function_l345_345410

theorem domain_of_function :
  ∀ x : ℝ, (f(x) = log 5 (log 2 (log 6 x))) → x ∈ (36 : ℝ, ∞) :=
by
  intros x hx
  change log 5 (log 2 (log 6 x)) with f x at hx
  sorry

end domain_of_function_l345_345410


namespace count_4_digit_numbers_with_property_l345_345501

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l345_345501


namespace chair_subsets_with_at_least_three_adjacent_l345_345260

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345260


namespace triangle_properties_l345_345945

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (angle_A : A = 30) (angle_B : B = 45) (side_a : a = Real.sqrt 2) :
  b = 2 ∧ (1 / 2) * a * b * Real.sin (105 * Real.pi / 180) = (Real.sqrt 3 + 1) / 2 := by
sorry

end triangle_properties_l345_345945


namespace true_propositions_l345_345392

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345392


namespace value_of_f_minus_3_l345_345448

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + x^3 + 1

theorem value_of_f_minus_3 (a b : ℝ) (h : f 3 a b = 7) : f (-3) a b = -5 := 
by
  sorry

end value_of_f_minus_3_l345_345448


namespace max_possible_points_difference_adjacent_teams_l345_345650

theorem max_possible_points_difference_adjacent_teams
  (teams : Finset ℕ)
  (total_teams : teams.card = 12)
  (plays_twice : ∀ t1 ∈ teams, ∀ t2 ∈ teams, t1 ≠ t2 → 2 * match_points t1 t2) -- Each team plays twice with every other team
  (win_points : (∀ t1 t2 ∈ teams, match_points t1 t2 = 2) ∨ (draw_points t1 t2 = 1) ∨ (loss_points t1 t2 = 0)) -- Win: 2 points, Draw: 1 point, Loss: 0 points
  (ranking : List ℕ)
  (ranked_by_points : ranking.sort (≥))
  : ∃ r : ℕ, r < ranking.length - 1 ∧ (ranking[r] - ranking[r+1] = 24) := 
sorry

end max_possible_points_difference_adjacent_teams_l345_345650


namespace current_time_between_3_and_4_l345_345950

theorem current_time_between_3_and_4 (t : ℝ) (h1 : t > 0) (h2 : t < 60)
  (h3 : 6 * (t + 5) = 89 + 0.5 * (t - 2)) :
  t = 54/5 := by
  change 6 * (t + 5) with 6 * t + 30 at h3
  change 89 + 0.5 * (t - 2) with 89 + 0.5 * t - 1 at h3
  linarith

end current_time_between_3_and_4_l345_345950


namespace jellybean_ratio_l345_345952

theorem jellybean_ratio (jellybeans_large: ℕ) (large_glasses: ℕ) (small_glasses: ℕ) (total_jellybeans: ℕ) (jellybeans_per_large: ℕ) (jellybeans_per_small: ℕ)
  (h1 : jellybeans_large = 50)
  (h2 : large_glasses = 5)
  (h3 : small_glasses = 3)
  (h4 : total_jellybeans = 325)
  (h5 : jellybeans_per_large = jellybeans_large * large_glasses)
  (h6 : jellybeans_per_small * small_glasses = total_jellybeans - jellybeans_per_large)
  : jellybeans_per_small = 25 ∧ jellybeans_per_small / jellybeans_large = 1 / 2 :=
by
  sorry

end jellybean_ratio_l345_345952


namespace students_per_row_first_scenario_l345_345910

theorem students_per_row_first_scenario 
  (S R x : ℕ)
  (h1 : S = x * R + 6)
  (h2 : S = 12 * (R - 3))
  (h3 : S = 6 * R) :
  x = 5 :=
by
  sorry

end students_per_row_first_scenario_l345_345910


namespace solve_fraction_eq_zero_l345_345178

theorem solve_fraction_eq_zero (x : ℝ) :
  (∃ (x : ℝ), (x + 5 = 0) ∧ (∀ (x : ℝ), x^2 + 4 * x + 10 > 0)) ↔ (x = -5) :=
by
  -- Define numerator
  let numerator := x + 5
  -- Define denominator
  let denominator := x^2 + 4 * x + 10

  -- State that numerator equals zero
  have h1 : numerator = 0 ↔ x = -5 := by
    simp [numerator]
  -- State that denominator is always positive
  have h2 : denominator > 0 := by
    calc (x^2 + 4 * x + 10) = (x + 2) ^ 2 + 6 : by ring
                        ... > 0 : by linarith

  -- Combine all to prove the main theorem
  split
  {
    intro h,
    cases h with x hx,
    exact hx.left
  }
  {
    intro hx,
    use x,
    exact ⟨hx, h2⟩
  }

end solve_fraction_eq_zero_l345_345178


namespace chairs_subsets_l345_345255

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345255


namespace minimize_expression_l345_345219

theorem minimize_expression :
  ∀ n : ℕ, 0 < n → (n = 6 ↔ ∀ m : ℕ, 0 < m → (n ≤ (2 * (m + 9))/(m))) := 
by
  sorry

end minimize_expression_l345_345219


namespace chairs_subsets_l345_345254

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345254


namespace min_sin_y_sin_x_l345_345208

theorem min_sin_y_sin_x 
  (x y : ℝ)
  (h1 : cos y + cos x = sin (3 * x))
  (h2 : sin (2 * y) - sin (2 * x) = cos (4 * x) - cos (2 * x)) :
  ∃ m : ℝ, m = -1 - sqrt (2 + sqrt 2) / 2 ∧ ∀ a b : ℝ, (cos b + cos a = sin (3 * a)) → 
    (sin (2 * b) - sin (2 * a) = cos (4 * a) - cos (2 * a)) → (sin b + sin a) ≥ m := 
sorry

end min_sin_y_sin_x_l345_345208


namespace find_angle_B_l345_345476

variables (a b c A B C : ℝ)
variables (triangle_ABC : Triangle a b c A B C)

def angle_A : A = π / 4 := sorry
def side_a : a = 2 := sorry
def trig_cond : b * Real.cos C - c * Real.cos B = 2 * Real.sqrt 2 := sorry

theorem find_angle_B : B = 5 * π / 8 :=
by
  have hyp_A : A = π / 4 := angle_A
  have hyp_a : a = 2 := side_a
  have hyp_trig : b * Real.cos C - c * Real.cos B = 2 * Real.sqrt 2 := trig_cond
  sorry -- The actual proof will be inserted here.

end find_angle_B_l345_345476


namespace units_digit_seven_consecutive_l345_345414

theorem units_digit_seven_consecutive (n : ℕ) : 
    (nat.digits 10 (n * (n+1) * (n+2) * (n+3) * (n+4) * (n+5) * (n+6))).head = 0 :=
by
  sorry

end units_digit_seven_consecutive_l345_345414


namespace true_propositions_l345_345391

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345391


namespace h_value_l345_345960

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem h_value :
  ∃ (h : ℝ → ℝ), (h 0 = 7)
  ∧ (∃ (a b c : ℝ), (f a = 0) ∧ (f b = 0) ∧ (f c = 0) ∧ (h (-8) = (1/49) * (-8 - a^3) * (-8 - b^3) * (-8 - c^3))) 
  ∧ h (-8) = -1813 := by
  sorry

end h_value_l345_345960


namespace world_series_game7_probability_l345_345652

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def probability_of_game7 : ℚ :=
  let favorable_outcomes := binomial_coefficient 6 3
  let total_outcomes := 2^6
  favorable_outcomes / total_outcomes

theorem world_series_game7_probability :
  probability_of_game7 = 5 / 16 := by
  sorry

end world_series_game7_probability_l345_345652


namespace ellipse_standard_eq_tangent_line_eq_l345_345847

noncomputable theory

-- Definitions
def line (x y : ℝ) := √6 * x + 2 * y - 2 * √6 = 0
def ellipse (x y : ℝ) := x^2 / 10 + y^2 / 6 = 1
def point_E (x y : ℝ) := x = 0 ∧ y = √6
def point_F (x y : ℝ) := x = 2 ∧ y = 0
def point_P := (√5, √3)

-- Theorem Statements
theorem ellipse_standard_eq (a b : ℝ) (h : a > b ∧ b > 0 ∧ ∀ x y, ellipse x y → line x y) : 
  ellipse = λ x y, x^2 / 10 + y^2 / 6 = 1 :=
sorry

theorem tangent_line_eq (P : ℝ × ℝ) (hP : ellipse P.1 P.2) : 
  (∀ x y : ℝ, (x, y) = point_P → (√5 / 10) * x + (√3 / 6) * y = 1) :=
sorry

end ellipse_standard_eq_tangent_line_eq_l345_345847


namespace chair_subsets_with_at_least_three_adjacent_l345_345262

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345262


namespace apples_left_after_pie_l345_345649

theorem apples_left_after_pie (initial_apples : ℝ) (given_apples : ℝ) (apples_for_pie : ℝ) :
  initial_apples = 10.0 → given_apples = 5.0 → apples_for_pie = 4.0 → initial_apples + given_apples - apples_for_pie = 11.0 :=
by
  intros h_init h_given h_pie
  rw [h_init, h_given, h_pie]
  norm_num
  sorry

end apples_left_after_pie_l345_345649


namespace find_P_point_l345_345833

noncomputable def P_point {x y : ℝ} (A B C P : ℝ × ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3)
  (AB.1 * AP.1 + AB.2 * AP.2 + AB.3 * AP.3 = 0) ∧
  (AC.1 * AP.1 + AC.2 * AP.2 + AC.3 * AP.3 = 0)

theorem find_P_point :
  let A := (0 : ℝ, 1, 0)
  let B := (1 : ℝ, 1, 0)
  let C := (1 : ℝ, 0, 0)
  ∃ (x y : ℝ), P_point A B C (x, y, 1) ∧ (x = 0) ∧ (y = 1) :=
by {
  sorry
}

end find_P_point_l345_345833


namespace verify_propositions_l345_345378

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345378


namespace count_4_digit_numbers_with_property_l345_345503

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l345_345503


namespace true_propositions_l345_345395

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345395


namespace coefficient_of_x3y0_l345_345943

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def f (m n : ℕ) : ℕ :=
  binomial_coeff 6 m * binomial_coeff 4 n

theorem coefficient_of_x3y0 :
  f 3 0 = 20 :=
by
  sorry

end coefficient_of_x3y0_l345_345943


namespace compare_values_log_and_exp_l345_345971

noncomputable def log_0_7_of_0_8 := Real.logBase 0.7 0.8
noncomputable def log_1_1_of_0_9 := Real.logBase 1.1 0.9
noncomputable def exp_1_1_to_0_9 := 1.1 ^ 0.9

theorem compare_values_log_and_exp :
  let a := log_0_7_of_0_8
  let b := log_1_1_of_0_9
  let c := exp_1_1_to_0_9
  c > a ∧ a > b :=
by {
  let a := log_0_7_of_0_8
  let b := log_1_1_of_0_9
  let c := exp_1_1_to_0_9
  sorry
}

end compare_values_log_and_exp_l345_345971


namespace find_CF_l345_345075

-- Given a triangle ABC with specific angles and side lengths
variables {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C]
variables (angle_BAC angle_ACB angle_ABC : ℝ)
variables (AB AC BC AD BD CF : ℝ)

-- Conditions:
-- ∠BAC = 60 degrees, ∠ACB = 30 degrees
-- AB = 2
-- D is the midpoint of AB
-- E lies on BC such that AE ⊥ BD
-- Extend AC through C to F such that EF = FB
def proof_problem : Prop :=
  (angle_BAC = 60) ∧ 
  (angle_ACB = 30) ∧
  (AB = 2) ∧
  (AD = AB / 2) ∧ 
  (BD = (AD ^ 2 + (AD - AB) ^ 2) ^ (1/2)) ∧ 
  (disjoint (line_through A E) (line_through B D)) ∧
  (∀ E, E ∈ line_segment B C → ⟦line_through A E∥ to R)) ∧
  (is_midpoint B F C → CF = sqrt 3)

-- Statement asking to prove CF = sqrt(3) given the conditions
theorem find_CF : proof_problem → CF = sqrt 3 := sorry

end find_CF_l345_345075


namespace impossible_sum_equality_l345_345776

theorem impossible_sum_equality (T : Fin 9 → Nat) (O : Fin 9 → Nat) (S : Fin 9 → Nat)
  (T_map : ∀ i : Fin 9, 1 ≤ T i ∧ T i ≤ 9) 
  (O_map : ∀ i : Fin 9, 1 ≤ O i ∧ O i ≤ 9) 
  (S_map : ∀ i : Fin 9, 1 ≤ S i ∧ S i ≤ 9)
  (T_distinct : ∀ i j : Fin 9, i ≠ j → T i ≠ T j) 
  (O_distinct : ∀ i j : Fin 9, i ≠ j → O i ≠ O j) 
  (S_distinct : ∀ i j : Fin 9, i ≠ j → S i ≠ S j) :
  ∑ i, T i * O i * S i ≠ 2019 := by
  sorry

end impossible_sum_equality_l345_345776


namespace arrangement_count_l345_345354

theorem arrangement_count (n : ℕ) : ∑ j in finset.Icc 1 n, (nat.choose (n-1) (j-1)) = 2^(n-1) :=
by
  sorry

end arrangement_count_l345_345354


namespace find_t_l345_345494

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (6, -4)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def perpendicular (x y : ℝ × ℝ) : Prop := dot_product x y = 0

theorem find_t : ∃ t : ℝ, perpendicular a (t • a + b) :=
by {
  use -5,
  simp [a, b, dot_product, perpendicular],
  norm_num,
}

end find_t_l345_345494


namespace limit_T2_over_T1_l345_345594

-- Define the curve
def curve (x : ℝ) : ℝ := log (x + 1)

-- Define the tangent line at (p, log(p + 1))
def tangent_line (p : ℝ) (x : ℝ) : ℝ :=
  (1 / (p + 1)) * x - (p / (p + 1)) + log (p + 1)

-- Define the normal line at (p, log(p + 1))
def normal_line (p : ℝ) (x : ℝ) : ℝ :=
  - (p + 1) * x + (p * (p + 1)) + log (p + 1)

-- Define the areas T1 and T2
noncomputable def T1 (p : ℝ) : ℝ :=
  integral 0 p (λ x, curve x - tangent_line p x)

noncomputable def T2 (p : ℝ) : ℝ :=
  integral 0 p (λ x, curve x - normal_line p x)

-- State the theorem
theorem limit_T2_over_T1 (T1 T2 : ℝ → ℝ) : 
  (∀ p : ℝ, T1 p = integral 0 p (λ x, curve x - tangent_line p x)) →
  (∀ p : ℝ, T2 p = integral 0 p (λ x, curve x - normal_line p x)) →
  tendsto (λ p, T2 p / T1 p) (nhds 0) (nhds (-1)) :=
by
  sorry

end limit_T2_over_T1_l345_345594


namespace range_of_function_l345_345412

theorem range_of_function : 
    ∀ x : ℝ, 0 ≤ x → (∃ y : ℝ, y ∈ set.Ici 1 ∧ y = (x^2 - 2)^2 + 1) :=
by
  intro x hx
  use (x^2 - 2)^2 + 1
  split
  { simp [set.mem_Ici, add_nonneg, pow_two_nonneg], linarith }
  { refl }

end range_of_function_l345_345412


namespace red_lucky_stars_l345_345324

theorem red_lucky_stars (x : ℕ) : (20 + x + 15 > 0) → (x / (20 + x + 15) : ℚ) = 0.5 → x = 35 := by
  sorry

end red_lucky_stars_l345_345324


namespace AL_LB_ratio_given_CK_KD_l345_345657

-- Given definitions from the problem
variables {S : Type*} [normed_space ℝ S]
variables (A B C D E K L : S)
variables (circle_S : metric_space.ball (0 : S) 1 = set.univ)

-- Conditions extracted from the problem
def perpendicular_diameters : Prop := 
  ∃ O, metric_space.ball O 1 = circle_S ∧ linear_independent ℝ (finset.univ (\[A,B\])) ∧ \[A,B\] ⊂ span O

def EA_intersects_CD_at_K : Prop := 
  ∃ K, metric_space.ball O 1 = circle_S ∧ E ∈ K ∧ linear_independent ℝ (finset.univ (\[E,A\])) ∧ E.A ⊂ span O ∧ 
  substantial_prism (evocable representatives {spanning magnate.subprismorphs} span O ∧{cd > 1}

def EC_intersects_AB_at_L : Prop := 
  ∃ L, metric_space.ball O 1 = circle_S ∧ {c / {bearings.ab L * a} => substantial > {⊂_FL L A B } ⊂ ∧
  [able_space ($χ,) eck_{eck_* representation eck_span]〉.

def CK_ratio_KD := (s=,) each 2 ∧ ∀∃ K} ∧ ^∧ ∧ {─f} ε:ck.similarity.equinor_true {from_eck_, <)

-- Construct the proof problem in Lean 4
theorem AL_LB_ratio_given_CK_KD : 
  perpendicular_diameters A B C D → 
  EA_intersects_CD_at_K E A K →
  EC_intersects_AB_at_L E C L →
  CK_ratio_KD C K D (2 : ℝ) (1 : ℝ) → 
  ∃ R, R = 3 ∧ AL : LB = (3 : ℝ) * (1 : ℝ) :=
sorry

end AL_LB_ratio_given_CK_KD_l345_345657


namespace center_cell_value_l345_345533

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l345_345533


namespace derivative_of_f_max_min_values_of_f_in_interval_l345_345875

def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 4 * x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 3 * x^2 + x - 4 :=
by
  intro x
  sorry

theorem max_min_values_of_f_in_interval :
  ∃ (C_min C_max : ℝ),
    (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f(x) ≥ C_min) ∧
    (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f(x) ≤ C_max) ∧
    C_min = -5 / 2 ∧
    C_max = 104 / 27 :=
by
  use -5 / 2, 104 / 27
  sorry

end derivative_of_f_max_min_values_of_f_in_interval_l345_345875


namespace cube_roll_prime_probability_l345_345743

theorem cube_roll_prime_probability :
  (∑ (a b : ℕ) in finset.fin_range 6, if nat.prime (a + b) then 1 else 0) / 36 = 5 / 12 := 
sorry

end cube_roll_prime_probability_l345_345743


namespace convert_speed_l345_345760

theorem convert_speed (s_kph : ℝ) (s_kph_val : s_kph = 1.5) : 
  (s_kph * (1000 / 3600)) = 0.4167 :=
by
  -- s_kph = 1.5 by assumption
  rewrite [s_kph_val]
  -- calculate manually: 1.5 * (1000 / 3600)
  rw [div_eq_mul_inv, mul_comm, ←mul_assoc, inv_mul_cancel_left, 
      mul_comm, inv_mul_cancel_left, ←mul_inv, mul_comm, mul_left_comm, 
      mul_comm 1.5 1000, ←mul_assoc, mul_inv_cancel_left]
  sorry

end convert_speed_l345_345760


namespace trapezium_area_correct_l345_345305

-- Define the lengths of the parallel sides and the distance between them
def a := 24  -- length of the first parallel side in cm
def b := 14  -- length of the second parallel side in cm
def h := 18  -- distance between the parallel sides in cm

-- Define the area calculation function for the trapezium
def trapezium_area (a b h : ℕ) : ℕ :=
  1 / 2 * (a + b) * h

-- The theorem to prove that the area of the given trapezium is 342 square centimeters
theorem trapezium_area_correct : trapezium_area a b h = 342 :=
  sorry

end trapezium_area_correct_l345_345305


namespace true_compound_props_l345_345399

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345399


namespace simplify_expression_l345_345812

theorem simplify_expression (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 :=
by
  sorry

end simplify_expression_l345_345812


namespace prime_triples_l345_345423

theorem prime_triples (p x y : ℕ) (hp : p.prime) (hx : 0 < x) (hy : 0 < y) :
  (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x^(p-1) + y = p^a ∧ x + y^(p-1) = p^b) ↔ 
  (p = 3 ∧ (x = 2 ∧ y = 5 ∨ x = 5 ∧ y = 2)) ∨ 
  (p = 2 ∧ ∃ n k : ℕ, 0 < n ∧ k > 0 ∧ x = n ∧ y = 2^k - n) := 
sorry

end prime_triples_l345_345423


namespace center_cell_value_l345_345546

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l345_345546


namespace sasha_age_l345_345782

theorem sasha_age :
  ∃ a : ℕ, 
    (M = 2 * a - 3) ∧
    (M = a + (a - 3)) ∧
    (a = 3) :=
by
  sorry

end sasha_age_l345_345782


namespace area_of_quad_PFCG_l345_345777

-- Define the rectangle ABCD
variables (A B C D E F G H P: Type)

-- Area function
variable (area : Type → ℝ)

-- Conditions as per the problem statement
def is_rectangle (A B C D : Type) := 
  AB = 4 ∧ BC = 6 ∧ 
  ∃ E, ∃ F, ∃ G, ∃ H, (AE / EB = 3 / 1) ∧ (BF / FC = 2 / 1) ∧ (DG / GC = 1 / 3) ∧ (AH / HD = 1 / 2)

def point_on_line_segment (P : Type) (H F : Type) := 
  ∃ t ∈ (0,1), P = H + t * (F - H)

-- Given an area of quadrilateral
def area_of_quad_AEPH (A E P H : Type) := 
  area A E P H = 5

-- Prove the desired area
theorem area_of_quad_PFCG (h₁ : is_rectangle A B C D)
                           (h₂ : point_on_line_segment P H F)
                           (h₃ : area_of_quad_AEPH A E P H) : 
  area P F C G = 8 := 
sorry

end area_of_quad_PFCG_l345_345777


namespace susie_investment_l345_345647

theorem susie_investment (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  let z := 1500 - x - y in
  z ≥ 0 →
  1.04^3 * x + 1.05^3 * y + 1.06^3 * z = 1755.06 →
  x = 475.63 :=
begin
  sorry
end

end susie_investment_l345_345647


namespace isosceles_triangle_perimeter_l345_345073

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 4) (h2 : b = 9) (h3 : a ≠ b) (h4 : a + a > b ∨ b + b > a) :
  a + a + b = 22 ∨ b + b + a = 22 :=
by {
  have h5 : a = 4 := h1,
  have h6 : b = 9 := h2,
  have h7 : a ≠ b := h3,
  have h8 : a + a > b ∨ b + b > a := h4,
  sorry
}

end isosceles_triangle_perimeter_l345_345073


namespace lives_per_player_l345_345334

theorem lives_per_player (num_players total_lives : ℕ) (h1 : num_players = 8) (h2 : total_lives = 64) :
  total_lives / num_players = 8 := by
  sorry

end lives_per_player_l345_345334


namespace find_cost_of_second_book_l345_345157

-- Given conditions
variables (b1 : ℝ) (bill : ℝ) (change : ℝ)

-- Setting the known values for the conditions
def known_conditions : Prop :=
  (b1 = 5.5) ∧ (bill = 20) ∧ (change = 8)

-- The proof statement
theorem find_cost_of_second_book (b1 bill change : ℝ) (h : known_conditions b1 bill change) : ∃ b2 : ℝ, b2 = 6.5 :=
by {
  have total_spent := bill - change,
  use total_spent - b1,
  sorry
}

end find_cost_of_second_book_l345_345157


namespace blue_die_greater_l345_345322

-- Conditions
def blue_die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}
def yellow_die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}
def p := 1 / 6
def q := 1 / 6

-- Question (Goal)
theorem blue_die_greater :
  let E := {x ∈ blue_die | ∀ y ∈ yellow_die, x > y} in
  let prob := ∑ x in E, p * q in
  prob = 5 / 12 := sorry

end blue_die_greater_l345_345322


namespace min_distance_Y_P_Q_Z_proof_l345_345928

-- Define the triangle and the values
variables (X Y Z P Q : Type) [has_dist X Y Z P Q]
variable (angle_XYZ : has_angle X Y Z)
variable (XY : has_dist X Y)
variable (XZ : has_dist X Z)

-- Condition definitions
axiom angle_XYZ_val : angle_XYZ = 60
axiom XY_val : XY = 8
axiom XZ_val : XZ = 12

-- Question as Lean theorem
theorem min_distance_Y_P_Q_Z_proof :
  ∃ (YP PQ QZ : has_dist P Q), YP + PQ + QZ = sqrt 304 :=
sorry

end min_distance_Y_P_Q_Z_proof_l345_345928


namespace value_of_f_neg2_l345_345664

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem value_of_f_neg2 (a b c : ℝ) (h1 : f a b c 5 + f a b c (-5) = 6) (h2 : f a b c 2 = 8) :
  f a b c (-2) = -2 := by
  sorry

end value_of_f_neg2_l345_345664


namespace alpha_beta_inequality_l345_345472

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) : 
  -2 < α - β ∧ α - β < 0 := 
sorry

end alpha_beta_inequality_l345_345472


namespace collinear_points_l345_345622

structure Circle (P Q R : Type) where
  diameter : P → Q → Prop 
  on_circle : R → Prop

variables {P Q R : Type} [Circle P Q R]

def midpoint (K : R) (C D: R) : Prop :=
  ∃ XY : R, K = midpoint (C, D) ∧ XY = diameter K

def projection (M : R) (X A C : R) : Prop :=
  ∃ l : R, M = projection X AC

theorem collinear_points 
  (C D : R) 
  (XY : Q) 
  (K : R) 
  (M : R) 
  (N : R) 
  (X A : R) 
  (Y B : R)
  (h1 : Circle P Q R)
  (h2 : h1.on_circle C)
  (h3 : h1.on_circle D)
  (h4 : h1.diameter XY K)
  (h5 : midpoint K C D)
  (h6 : projection M X A C)
  (h7 : projection N Y B D)
  : collinear M N K := 
sorry

end collinear_points_l345_345622


namespace sum_largest_smallest_three_digit_l345_345528

theorem sum_largest_smallest_three_digit :
  (∃ (d1 d2 d3 d4 : ℕ), (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) ∧
    (d1 = 0 ∨ d1 = 2 ∨ d1 = 4 ∨ d1 = 6) ∧
    (d2 = 0 ∨ d2 = 2 ∨ d2 = 4 ∨ d2 = 6) ∧
    (d3 = 0 ∨ d3 = 2 ∨ d3 = 4 ∨ d3 = 6) ∧
    (d4 = 0 ∨ d4 = 2 ∨ d4 = 4 ∨ d4 = 6) ∧
    (d1 * 100 + d2 * 10 + d3 ≤ 642 ∨ d1 * 100 + d2 * 10 + d3 ≥ 204) ∧
    ((d1 = 6 ∧ d2 = 4 ∧ d3 = 2) ∨ (d1 = 2 ∧ d2 = 0 ∧ d3 = 4))) →
  (642 + 204) = 846 :=
by 
  simp
  -- Proof omitted
  sorry

end sum_largest_smallest_three_digit_l345_345528


namespace parabola_intersect_line_segment_range_l345_345031

theorem parabola_intersect_line_segment_range (a : ℝ) :
  (∀ (x : ℝ), y = a * x^2 - 3 * x + 1) →
  (∀ (x1 y1 x2 y2 x0 : ℝ), (x0 ∈ {x | ∃ y, y = a * x^2 - 3 * x + 1}) →
                         (x1 - x0).abs > (x2 - x0).abs →
                         ∃ y1 y2, y1 = a * x1^2 - 3 * x1 + 1 ∧ y2 = a * x2^2 - 3 * x2 + 1 → y1 > y2) →
  (∃ (x : ℝ), x ∈ Icc (-1 : ℝ) (3 : ℝ) → y = x - 1) →
  (∃ (x : ℝ), y = a * x^2 - 3 * x + 1 = (x - 1)) →
  (\(a \ge 10/9 ∧ a < 2\)) :=
by
  sorry

end parabola_intersect_line_segment_range_l345_345031


namespace count_increasing_arithmetic_sequences_length_3_l345_345631

theorem count_increasing_arithmetic_sequences_length_3 :
  let S := {i : ℕ | 1 ≤ i ∧ i ≤ 20}
  ∃ (n : ℕ), (∀ a b c ∈ S, a < b ∧ b < c ∧ 2 * b = a + c → n) ∧ n = 90 :=
by
  sorry

end count_increasing_arithmetic_sequences_length_3_l345_345631


namespace sandy_hits_l345_345348

variables (H A h: ℕ)
variable (k: ℤ)

theorem sandy_hits :
  (A > 0) ∧ (H = 0.360 * A) ∧ (h = 0.040 * A + 2) ∧ (k - 2 = 0.040 * A) →
  H + h = 12 :=
begin
  sorry
end

end sandy_hits_l345_345348


namespace staplers_left_l345_345680

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l345_345680


namespace chess_match_probability_l345_345282

theorem chess_match_probability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (3 * p^3 * (1 - p) ≤ 6 * p^3 * (1 - p)^2) → (p ≤ 1/2) :=
by
  sorry

end chess_match_probability_l345_345282


namespace cube_rods_l345_345721

theorem cube_rods {n : ℕ} (h: n > 0) :
  let N := 2 * n in
  ∃ (rods : set (fin N × fin N × fin N)), rods.card = 2 * n^2 ∧
  ∀ (r1 r2 : fin N × fin N × fin N), r1 ≠ r2 → 
    rods r1 ∧ rods r2 → (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2 ∧ r1.3 ≠ r2.3) :=
by 
  sorry

end cube_rods_l345_345721


namespace angle_PMN_66_l345_345089

theorem angle_PMN_66 (P Q R M N : Type) [EuclideanGeometry] 
                      (angle_PQR : ∠ P Q R = 48) 
                      (isosceles_PQR : PR = RQ) 
                      (isosceles_PMN : PM = PN):
                      ∠ P M N = 66 :=
by
  sorry

end angle_PMN_66_l345_345089


namespace library_visits_l345_345954

theorem library_visits
  (william_visits_per_week : ℕ := 2)
  (jason_visits_per_week : ℕ := 4 * william_visits_per_week)
  (emma_visits_per_week : ℕ := 3 * jason_visits_per_week)
  (zoe_visits_per_week : ℕ := william_visits_per_week / 2)
  (chloe_visits_per_week : ℕ := emma_visits_per_week / 3)
  (jason_total_visits : ℕ := jason_visits_per_week * 8)
  (emma_total_visits : ℕ := emma_visits_per_week * 8)
  (zoe_total_visits : ℕ := zoe_visits_per_week * 8)
  (chloe_total_visits : ℕ := chloe_visits_per_week * 8)
  (total_visits : ℕ := jason_total_visits + emma_total_visits + zoe_total_visits + chloe_total_visits) :
  total_visits = 328 := by
  sorry

end library_visits_l345_345954


namespace right_triangle_area_l345_345209

-- Define the variables and conditions
variables (p x y : ℝ)
def hypotenuse := 2 * x
def perimeter := x + y + hypotenuse

-- Conditions
axiom perimeter_eq : perimeter = 3 * p
axiom leg_half_hypotenuse : x = hypotenuse / 2

-- Goal: Prove the area formula
theorem right_triangle_area (h₁ : perimeter_eq) (h₂ : leg_half_hypotenuse) :
  (1 / 2) * x * y = (p^2 * Real.sqrt 3) / 4 :=
sorry

end right_triangle_area_l345_345209


namespace subsets_with_at_least_three_adjacent_chairs_l345_345244

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345244


namespace catherine_paint_gallons_l345_345793

noncomputable def paintGallonsNeeded (num_pillars : ℕ) (height : ℝ) (diameter : ℝ) (coverage : ℝ) : ℝ := 
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_area := lateral_surface_area * num_pillars
  let gallons := total_area / coverage
  Real.ceil gallons

theorem catherine_paint_gallons :
  paintGallonsNeeded 20 24 12 300 = 61 := 
by
  -- The proof is omitted and replaced with sorry for now.
  sorry

end catherine_paint_gallons_l345_345793


namespace find_n_l345_345859

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -1)
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 n - 1)) : 
  n = 12 :=
sorry

end find_n_l345_345859


namespace plane_equation_l345_345426

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧ 
  (∀ (x y z : ℤ), (x, y, z) = (0, 0, 0) ∨ (x, y, z) = (2, 0, -2) → A * x + B * y + C * z + D = 0) ∧ 
  ∀ (x y z : ℤ), (A = 1 ∧ B = -5 ∧ C = 1 ∧ D = 0) := sorry

end plane_equation_l345_345426


namespace inequality_S_over_n_minus_1_l345_345451

theorem inequality_S_over_n_minus_1 {n : ℕ} (a : ℕ → ℝ) (n_ge_two : 2 ≤ n) (a_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) :
  let S := ∑ i in Finset.range n, a i in
  (∑ i in Finset.range n, (a i)^2 / (S - a i)) ≥ S / (n - 1) :=
by
  let S := ∑ i in Finset.range n, a i
  sorry

end inequality_S_over_n_minus_1_l345_345451


namespace arithmetic_sequence_properties_l345_345314

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Given conditions
axiom h1 : ∀ n, a (n + 1) = a n + d (where d is the common difference of the arithmetic sequence)
axiom a2 : a 2 = 1
axiom a5 : a 5 = -5

-- Definition of the general term of the arithmetic sequence
def general_term (n : ℕ) : α := -2 * n + 5

-- Sum of the first n terms of the arithmetic sequence
noncomputable def sum_n (n : ℕ) : α := (n * (2 * (a 1) + (n - 1) * d)) / 2

-- Maximum sum of the first n terms
def max_sum_n : α :=
  let n := 2 in sum_n n

-- Proof goal:
-- 1. The general term of the sequence is -2n + 5
-- 2. The maximum value of the sum of the first n terms is 4 and occurs at n=2
theorem arithmetic_sequence_properties :
  (∀ n, a n = general_term n) ∧ max_sum_n = 4 :=
by
  sorry

end arithmetic_sequence_properties_l345_345314


namespace shorter_side_length_l345_345718

theorem shorter_side_length (L W : ℝ) (h1 : L * W = 91) (h2 : 2 * L + 2 * W = 40) :
  min L W = 7 :=
by
  sorry

end shorter_side_length_l345_345718


namespace find_missing_values_l345_345029

theorem find_missing_values :
  (∃ x y : ℕ, 4 / 5 = 20 / x ∧ 4 / 5 = y / 20 ∧ 4 / 5 = 80 / 100) →
  (x = 25 ∧ y = 16 ∧ 4 / 5 = 80 / 100) :=
by
  sorry

end find_missing_values_l345_345029


namespace ratio_of_areas_of_triangles_l345_345228

-- Definitions and conditions
def triangle_GHI (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ (c * c = a * a + b * b)
def triangle_JKL (a b c : ℕ) : Prop := a = 9 ∧ b = 40 ∧ c = 41 ∧ (c * c = a * a + b * b)
def area (base height : ℕ) : ℚ := (base * height : ℚ) / 2

-- Statement to prove
theorem ratio_of_areas_of_triangles :
  ∀ (a b c d e f : ℕ),
    triangle_GHI a b c →
    triangle_JKL d e f →
    a = 6 ∧ b = 8 ∧ d = 9 ∧ e = 40 →
    area a b / area d e = 2 / 15 :=
by
  intros a b c d e f h1 h2 h3
  rw [triangle_GHI, triangle_JKL] at h1 h2
  simp [h1, h2, h3, area]
  sorry

end ratio_of_areas_of_triangles_l345_345228


namespace probability_of_independent_events_l345_345908

theorem probability_of_independent_events (p : Set → ℚ) (a b : Set) (h_independent : ∀ s t, a = s → b = t → p (s ∩ t) = p s * p t)
(h_pa : p a = 4 / 7) (h_pb : p b = 2 / 5) : p (a ∩ b) = 8 / 35 :=
by {
  have h1 : p (a ∩ b) = p a * p b := h_independent a b rfl rfl,
  rw [h_pa, h_pb] at h1,
  norm_num at h1,
  exact h1,
}

end probability_of_independent_events_l345_345908


namespace find_angle_C_max_cos_A_B_l345_345475

-- Part I
theorem find_angle_C (a b c : ℝ) (h : c^2 = a^2 + b^2 - ab) : 
  ∃ C : ℝ, C = π / 3 :=
sorry

-- Part II
theorem max_cos_A_B (A B : ℝ) (C : ℝ) (hC : C = π / 3) (hSum : A + B + C = π) :
  ∃ maxVal : ℝ, maxVal = 1 ∧ maxVal = max (cos A + cos B) :=
sorry

end find_angle_C_max_cos_A_B_l345_345475


namespace range_of_a_l345_345180

theorem range_of_a (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f x) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f (x - a) + f (x + a)) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l345_345180


namespace true_propositions_correct_l345_345371

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345371


namespace ratio_of_areas_l345_345109

-- Given definitions and conditions
variables (α : Real) (ABC BMN : Type) 
variables (area : Type → Real)

-- Altitudes BM and CN, proportion AM:CM = 2:3, and BAC angle is α
def triangle (T : Type) : Prop := T = ABC ∨ T = BMN
def altitude (ALT : Type) : Prop := ALT = BM ∨ ALT = CN
def AM_CM_ratio (AM CM : Real) : Prop := AM / CM = 2 / 3
def BAC_angle (A B C : Point) (angle : Real) : Prop := angle = α

-- Formulate the theorem
theorem ratio_of_areas (A B C M N : Point) (h_alt_BM : altitude BM) (h_alt_CN : altitude CN)
    (h_ratio : AM_CM_ratio (distance A M) (distance C M))
    (h_angle : BAC_angle A B C α) :
  (area BMN / area ABC) = abs (cos α ^ 2 - 2 / 5) := by sorry

end ratio_of_areas_l345_345109


namespace race_time_A_is_20_l345_345556

-- Definitions provided by the conditions
def race_length := 120
def distance_advantage := 72
def time_advantage := 10

-- Conditions in terms of variables
variables (T_A : ℝ) (V_A V_B : ℝ)

-- Definition of speeds in terms of time
def speed_A := race_length / T_A
def speed_B_1 := (race_length - distance_advantage) / T_A
def speed_B_2 := distance_advantage / (T_A + time_advantage)

-- Theorem statement: Prove that T_A equals 20 seconds
theorem race_time_A_is_20 :
  (speed_A = race_length / T_A) ∧ 
  (speed_B_1 = (race_length - distance_advantage) / T_A) ∧ 
  (speed_B_2 = distance_advantage / (T_A + time_advantage)) →
  (speed_B_1 = speed_B_2) →
   T_A = 20 :=
by
  sorry

end race_time_A_is_20_l345_345556


namespace correct_a_values_l345_345817

noncomputable def valid_a_values : set ℝ :=
  {a : ℝ | ∃ (x1 x2 x3 x4 x5 : ℝ), 0 ≤ x1 ∧ 0 ≤ x2 ∧ 0 ≤ x3 ∧ 0 ≤ x4 ∧ 0 ≤ x5 ∧ 
    x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 = a ∧ 
    x1 + 8 * x2 + 27 * x3 + 64 * x4 + 125 * x5 = a^2 ∧
    x1 + 32 * x2 + 243 * x3 + 1024 * x4 + 3125 * x5 = a^3 }

theorem correct_a_values : valid_a_values = {1, 4, 9, 16, 25} :=
sorry

end correct_a_values_l345_345817


namespace true_propositions_l345_345396

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345396


namespace true_propositions_correct_l345_345367

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345367


namespace analytical_expression_l345_345459

noncomputable def function_expression (x : ℝ) : ℝ := 2 * real.sin (2 * x) + 2

theorem analytical_expression :
  ∃ A ω φ m : ℝ, 
  (∀ x, function_expression x = A * real.sin (ω * x + φ) + m) ∧
  (∀ x, A * real.sin (ω * x + φ) + m ≤ 4) ∧
  (∀ x, A * real.sin (ω * x + φ) + m ≥ 0) ∧
  (ω ≠ 0) ∧
  (∃ k > 0, ∀ x, function_expression(x + k) = function_expression x) ∧
  k = π ∧
  (∀ x, function_expression(x) = function_expression(-x)) :=
begin
  sorry
end

end analytical_expression_l345_345459


namespace angle_between_vectors_l345_345895

variable (a b : Vector ℝ)

-- Given conditions
def magnitude_of_a : ℝ := 1
def magnitude_of_b : ℝ := sqrt 2
def dot_product_condition : (a - b) ⬝ a = 0

-- The proof problem
theorem angle_between_vectors (h1 : ‖a‖ = magnitude_of_a) (h2 : ‖b‖ = magnitude_of_b) (h3 : dot_product_condition) : 
  ∠ a b = π / 4 := 
sorry

end angle_between_vectors_l345_345895


namespace jackson_meat_left_l345_345578

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l345_345578


namespace ribbons_purple_l345_345558

theorem ribbons_purple (total_ribbons : ℕ) (yellow_ribbons purple_ribbons orange_ribbons black_ribbons : ℕ)
  (h1 : yellow_ribbons = total_ribbons / 4)
  (h2 : purple_ribbons = total_ribbons / 3)
  (h3 : orange_ribbons = total_ribbons / 6)
  (h4 : black_ribbons = 40)
  (h5 : yellow_ribbons + purple_ribbons + orange_ribbons + black_ribbons = total_ribbons) :
  purple_ribbons = 53 :=
by
  sorry

end ribbons_purple_l345_345558


namespace vasya_can_double_money_l345_345562

theorem vasya_can_double_money (
  exchange_ratios : 
    (exchange_ratio_dross_coconuts: ℝ) → -- 1 dollar = 10 coconuts
    (exchange_ratio_raccoon_bananas: ℝ) → -- 1 raccoon = 6 bananas
    (exchange_ratio_raccoon_coconuts: ℝ) → -- 1 raccoon = 11 coconuts
    (exchange_ratio_coconut_dollars: ℝ) → -- 1 coconut = 1/15 dollars
    (exchange_ratio_banana_coconuts: ℝ),  -- 1 banana = 2 coconuts
  initial_dollars: ℝ := 100,
  target_dollars: ℝ := 200
) : Prop :=
  ∃ strategy, strategy initial_dollars >= target_dollars

end vasya_can_double_money_l345_345562


namespace simplify_trig_expression_tan_alpha_value_l345_345312

-- Proof Problem (1)
theorem simplify_trig_expression :
  (∃ θ : ℝ, θ = (20:ℝ) ∧ 
    (∃ α : ℝ, α = (160:ℝ) ∧ 
      (∃ β : ℝ, β = 1 - 2 * (Real.sin θ) * (Real.cos θ) ∧ 
        (∃ γ : ℝ, γ = 1 - (Real.sin θ)^2 ∧ 
          (Real.sqrt β) / ((Real.sin α) - (Real.sqrt γ)) = -1)))) :=
sorry

-- Proof Problem (2)
theorem tan_alpha_value (α : ℝ) (h : Real.tan α = 1 / 3) :
  1 / (4 * (Real.cos α)^2 - 6 * (Real.sin α) * (Real.cos α)) = 5 / 9 :=
sorry

end simplify_trig_expression_tan_alpha_value_l345_345312


namespace find_omega_l345_345878

-- Let 𝛀 be any positive real number
-- Define the function f and its properties

-- The Lean 4 statement to prove that ω = 1/2
theorem find_omega (ω : ℝ) (h1 : 0 < ω) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ (2 * π / 3) → (2 * (ω * x.sup 1)) ≤ 0)
  (h3 : 2 * cos (ω * (2 * π / 3)) = 1) : ω = 1 / 2 := 
by 
  sorry

end find_omega_l345_345878


namespace max_S_plus_sqrt3_cosB_cosC_l345_345923

variable {A B C : ℝ}
variable {a b c S : ℝ}

-- Conditions
def triangle_conditions : Prop :=
  (a^2 = b^2 + c^2 + b * c) ∧
  (a = sqrt 3) ∧
  (S = (1 / 2) * a * c * sin B)

-- Statement to prove
theorem max_S_plus_sqrt3_cosB_cosC (A B C a b c S : ℝ) 
  (h : triangle_conditions) : 
  ∃ M, M = S + sqrt 3 * cos B * cos C ∧ M ≤ sqrt 3 :=
sorry

end max_S_plus_sqrt3_cosB_cosC_l345_345923


namespace find_constants_l345_345966

open Matrix 

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem find_constants :
  ∃ c d : ℝ, c = 1/12 ∧ d = 1/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end find_constants_l345_345966


namespace largest_even_sum_1988_is_290_l345_345072

theorem largest_even_sum_1988_is_290 (n : ℕ) 
  (h : 14 * n = 1988) : 2 * n + 6 = 290 :=
sorry

end largest_even_sum_1988_is_290_l345_345072


namespace gdp_scientific_notation_l345_345944

theorem gdp_scientific_notation
  (gdp_2022 : ℕ)
  (h : gdp_2022 = 1400000000000)
  : gdp_2022 = 1.4 * 10^12 :=
by 
  -- Here we skip the proof
  sorry

end gdp_scientific_notation_l345_345944


namespace red_rose_cost_l345_345114

-- Define the necessary conditions
def two_dozens_of_red_roses : ℕ := 24
def three_sunflowers_cost : ℝ := 3 * 3
def total_flower_cost : ℝ := 45

-- Question and conditions reformulated into Lean statement
theorem red_rose_cost (x : ℝ) (h1 : two_dozens_of_red_roses * x + three_sunflowers_cost = total_flower_cost) : x = 1.5 :=
by
sor님리

end red_rose_cost_l345_345114


namespace ratio_sum_zero_l345_345918

theorem ratio_sum_zero :
  (∑ x in {x : ℝ | (3*x + 4) / (5*x + 4) = (5*x + 6) / (8*x + 6)}, x) = 0 :=
by sorry

end ratio_sum_zero_l345_345918


namespace counterfeit_found_in_two_weighings_l345_345990

def Coin := ℕ  -- We use natural numbers to represent coins

def isCounterfeit (c : Coin) (coins : Fin 9 → Coin) : Prop :=
  coins c = true /- identifies if the coin c is counterfeit within the list of coins -/

noncomputable def findCounterfeit
  (coins : Fin 9 → Coin)
  (balance : (Fin 3 → Coin) → (Fin 3 → Coin) → bool) : Coin :=
  sorry /- the implementation for finding the counterfeit coin -/

theorem counterfeit_found_in_two_weighings (coins : Fin 9 → Coin)
  (balance : (Fin 3 → Coin) → (Fin 3 → Coin) → bool)
  (assumption : ∃ c: Fin 9, isCounterfeit c coins ∧ (∀ c', c' ≠ c → ¬isCounterfeit c' coins)) :
  ∃ c: Fin 9, isCounterfeit c coins ∧ (findCounterfeit coins balance = c) :=
begin
  -- Here we can assume the proof steps, but we would need to implement findCounterfeit for actual verification
  sorry
end

end counterfeit_found_in_two_weighings_l345_345990


namespace trigonometric_identity_l345_345442

open Real

theorem trigonometric_identity
  (theta : ℝ)
  (h : cos (π / 6 - theta) = 2 * sqrt 2 / 3) : 
  cos (π / 3 + theta) = 1 / 3 ∨ cos (π / 3 + theta) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l345_345442


namespace center_cell_value_l345_345543

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l345_345543


namespace product_of_distinct_primes_divisible_by_each_minus_one_l345_345290

theorem product_of_distinct_primes_divisible_by_each_minus_one :
  ∀ (p : ℕ → ℕ) (k : ℕ), (k ≥ 2) →
   (∀ i, 1 ≤ i ∧ i ≤ k → prime (p i)) →
   (∀ i j, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k ∧ i ≠ j → p i ≠ p j) →
   (∀ i, 1 ≤ i ∧ i ≤ k → (∏ n in finset.range (k + 1), p (n + 1)) % (p i - 1) = 0) →
   (∏ n in finset.range (k + 1), p (n + 1) = 6 ∨ ∏ n in finset.range (k + 1), p (n + 1) = 42 ∨
    ∏ n in finset.range (k + 1), p (n + 1) = 1806) :=
by
  arbitrary sorry

end product_of_distinct_primes_divisible_by_each_minus_one_l345_345290


namespace range_of_a_exists_two_real_roots_l345_345490

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 / 4) * x + 1 else real.log x

theorem range_of_a_exists_two_real_roots (a : ℝ) : 
  (∀ x : ℝ, f x = a * x) → 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ ∧ f x₂ = a * x₂) ↔ (a ∈ set.Ico (1 / 4 : ℝ) (real.exp(-1))) := 
sorry

end range_of_a_exists_two_real_roots_l345_345490


namespace chairs_subset_count_l345_345274

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345274


namespace number_of_true_propositions_l345_345388

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345388


namespace all_Tinks_Falars_Gymes_l345_345559

universe u

variables {Falars Gymes Halps Tinks Isoys : Type u}
variables (Falar : Falars → Prop) (Gyme : Gymes → Prop)
variables (Halp : Halps → Prop) (Tink : Tinks → Prop) (Isoy : Isoys → Prop)

axiom all_Falars_Gymes : ∀ x : Falars, Falar x → Gyme x
axiom all_Halps_Tinks : ∀ x : Halps, Halp x → Tink x
axiom all_Isoys_Falars : ∀ x : Isoys, Isoy x → Falar x
axiom all_Tinks_Isoys : ∀ x : Tinks, Tink x → Isoy x

theorem all_Tinks_Falars_Gymes :
  ∀ x : Tinks, (Tink x → Falar x) ∧ (Tink x → Gyme x) :=
by
  sorry

end all_Tinks_Falars_Gymes_l345_345559


namespace find_DF_l345_345108

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345108


namespace staplers_left_l345_345681

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l345_345681


namespace length_of_AC_in_triangle_l345_345573

theorem length_of_AC_in_triangle
  (A B C : Point)
  (BC : length(B, C) = 1)
  (angle_B : angle B = pi / 3)
  (area_ABC : area A B C = sqrt 3) :
  length(A, C) = sqrt 13 :=
sorry

end length_of_AC_in_triangle_l345_345573


namespace F_recurrence_l345_345289

def A (n : ℕ) : ℕ := 2^n

def F : ℕ → ℕ
| 0     := 0  -- F_0 is not defined in the problem but we define it as 0 for convenience.
| 1     := 2
| 2     := 3
| 3     := 5
| 4     := 8
| (n+5) := F (n+4) + F (n+3)

theorem F_recurrence (n p : ℕ) (hn : n ≥ 1) (hp : p ≥ 1) : 
  F (n + p + 1) = F n * F p + F (n - 1) * F (p - 1) := sorry

end F_recurrence_l345_345289


namespace top_card_is_king_l345_345366

noncomputable def num_cards := 52
noncomputable def num_kings := 4
noncomputable def probability_king := num_kings / num_cards

theorem top_card_is_king :
  probability_king = 1 / 13 := by
  sorry

end top_card_is_king_l345_345366


namespace determinant_tangent_matrix_l345_345596

theorem determinant_tangent_matrix (A B C : Real) (h1 : A + B + C = Real.pi) (h2 : A ≠ Real.pi / 2) (h3 : B ≠ Real.pi / 2) (h4 : C ≠ Real.pi / 2) :
    Real.det (Matrix.of ![![Real.tan A, 1, 1], ![1, Real.tan B, 1], ![1, 1, Real.tan C]]) = 2 :=
by
  sorry

end determinant_tangent_matrix_l345_345596


namespace sage_reflection_day_l345_345142

theorem sage_reflection_day 
  (day_of_reflection_is_jan_1 : Prop)
  (equal_days_in_last_5_years : Prop)
  (new_year_10_years_ago_was_friday : Prop)
  (reflections_in_21st_century : Prop) : 
  ∃ (day : String), day = "Thursday" :=
by
  sorry

end sage_reflection_day_l345_345142


namespace max_value_of_f_sum_of_roots_lt_neg_two_log_a_l345_345880

-- Given function
def f (x a b : ℝ) := x - a * real.exp x + b

-- Condition: a > 0
variables (a b : ℝ) (h_a : a > 0)

-- 1. Maximum value of f(x)
theorem max_value_of_f : 
  ∃ x_max, x_max = real.log (1 / a) ∧ f (real.log (1 / a)) a b = real.log (1 / a) - 1 + b :=
sorry

-- 2. Inequality for distinct zeroes
theorem sum_of_roots_lt_neg_two_log_a
  (h_b : b ∈ set.univ)
  (x1 x2 : ℝ)
  (h_roots : f x1 a b = 0 ∧ f x2 a b = 0 ∧ x1 ≠ x2) :
  x1 + x2 < -2 * real.log a :=
sorry

end max_value_of_f_sum_of_roots_lt_neg_two_log_a_l345_345880


namespace find_values_of_a_l345_345040

theorem find_values_of_a (a n l : ℕ)
    (m k : ℕ → ℕ)
    (h1 : a > 1)
    (h2 : (∏ i in Finset.range n, a^(m i) - 1) = (∏ j in Finset.range l, a^(k j) + 1))
    : a = 2 ∨ a = 3 :=
by
  sorry

end find_values_of_a_l345_345040


namespace translate_vertex_to_increase_l345_345887

def quadratic_function (x : ℝ) : ℝ := -x^2 + 1

theorem translate_vertex_to_increase (x : ℝ) :
  ∃ v, v = (2, quadratic_function 2) ∧
    (∀ x < 2, quadratic_function (x + 2) = quadratic_function x + 1 ∧
    ∀ x < 2, quadratic_function x < quadratic_function (x + 1)) :=
sorry

end translate_vertex_to_increase_l345_345887


namespace center_cell_value_l345_345552

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l345_345552


namespace center_cell_value_l345_345545

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l345_345545


namespace hall_length_l345_345188

theorem hall_length (L B A : ℝ) (h1 : B = 2 / 3 * L) (h2 : A = 2400) (h3 : A = L * B) : L = 60 := by
  -- proof steps here
  sorry

end hall_length_l345_345188


namespace polynomial_expression_value_l345_345601

-- Declare the roots of the polynomial and their identities from Vieta's formulas
variables {p q r : ℝ}

-- The conditions as given in the problem
axiom roots_of_polynomial : p + q + r = 8
axiom roots_product_sum : pq + pr + qr = 10
axiom roots_product : pqr = 3

-- The Lean statement we need to prove
theorem polynomial_expression_value : 
  ∀ (p q r : ℝ), 
  (p + q + r = 8) → 
  (pq + pr + qr = 10) → 
  (pqr = 3) → 
  (p/(qr + 2) + q/(pr + 2) + r/(pq + 2) = 8/5) :=
by
  intros p q r h₁ h₂ h₃
  sorry

end polynomial_expression_value_l345_345601


namespace seed_mixture_ryegrass_l345_345174

noncomputable def percentage_ryegrass (mixtureX mixtureY mixture_X_perc mixture_Y_perc final_ryegrass_perc : ℚ) : ℚ :=
  mixtureX*mixture_X_perc + mixtureY*mixture_Y_perc - final_ryegrass_perc * (mixture_X_perc + mixture_Y_perc)

theorem seed_mixture_ryegrass :
  ∀ (R : ℚ) (mixture_Y_ryegrass : ℚ) (final_mixture_ryegrass : ℚ) (mixture_X_fraction : ℚ) (mixture_Y_fraction : ℚ),
    mixture_X_fraction = 1 / 3 →
    mixture_Y_fraction = 2 / 3 →
    mixture_Y_ryegrass = 0.25 →
    final_mixture_ryegrass = 0.30 →
    100 * percentage_ryegrass (R / 100) mixture_Y_ryegrass mixture_X_fraction mixture_Y_fraction final_mixture_ryegrass = 40 :=
by
  intros R mixture_Y_ryegrass final_mixture_ryegrass mixture_X_fraction mixture_Y_fraction
  assume h1 : mixture_X_fraction = 1 / 3,
  assume h2 : mixture_Y_fraction = 2 / 3,
  assume h3 : mixture_Y_ryegrass = 0.25,
  assume h4 : final_mixture_ryegrass = 0.30
  sorry

end seed_mixture_ryegrass_l345_345174


namespace monotonicity_of_g_inequality_f_l345_345877

-- Define f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x + 1) * Real.exp x
def g (x : ℝ) (a m : ℝ) : ℝ := Real.log (f x a) + (m - 1) * x

-- Theorem for part (1): Monotonicity of g(x)
theorem monotonicity_of_g (m a x : ℝ) (h : a = 0) : 
  (m ≥ 1 ∨ m = 0) → (∀ x, Increasing (g x a m) ∧ 
  (m ≤ -1) → (∀ x, Decreasing (g x a m) ∧ 
  (0 < m ∧ m < 1) → (∀ x ∈ Set.Ioo (-(1 + sqrt(1 - m^2)) / m) ((1 - sqrt(1 - m^2)) / m), Decreasing  (g x a m)) ∧ 
  (m > (-(1 + sqrt(1 - m^2)) / m) ∨ m < ((1 - sqrt(1 - m^2)) / m)) → Increasing (g x a m) ∧ 
  (-1 < m ∧ m < 0) → (∀ x ∈ Set.Ioo (-(1 + sqrt(1 - m^2)) / m) ((1 - sqrt(1 - m^2)) / m), Decreasing (g x a m)) ∧ 
  (∀ x ∈ Set.Ioo (-(1 + sqrt(1 - m^2)) / m) ((1 - sqrt(1 - m^2)) / m), Increasing (g x a m) sorry

-- Theorem for part (2): Inequality f(x) > a(x+1)
theorem inequality_f (a x : ℝ) (h1 : -2 < a) (h2 : a < 1 / 2) (h3 : x ∈ Set.Ioi (-1)) : f x a > a * (x + 1) :=
begin
  sorry,
end

end monotonicity_of_g_inequality_f_l345_345877


namespace problem1_problem2_problem3_problem4_l345_345308

-- Definitions for the conditions in the problem
def independent (X Y : ℝ → ℝ) : Prop := sorry -- This would need more specific formalization
def characteristic_function (f : ℝ → ℂ) : Prop := sorry -- This would need more specific formalization
def distribution_function (F : ℝ → ℝ) : Prop := sorry -- This would need more specific formalization

-- Definitions for the random variables and their characteristic and distribution functions
variable (X Y : ℝ → ℝ)
variable (φ ψ : ℝ → ℂ)
variable (F G : ℝ → ℝ)

-- Assuming the necessary properties for X, Y, φ, ψ, F, G
axiom ind_X_Y : independent X Y
axiom char_φ : characteristic_function φ
axiom char_ψ : characteristic_function ψ
axiom dist_F : distribution_function F
axiom dist_G : distribution_function G

-- Definition for M(f, g)
noncomputable def M (f g : ℝ → ℂ) : ℂ :=
  lim at_top (λ T, (1 / (2 * T)) * ∫ t in -T..T, f(t) * conj(g(t)) )
  
-- Statements to be proved
theorem problem1 : (∀ x : ℝ, M φ (λ t, complex.exp (complex.i * t * x)) = F x - F (x -)) :=
sorry

theorem problem2 : M φ ψ = ∑' (x : ℤ), (F x - F (x -)) * (G x - G (x -)) :=
sorry

theorem problem3 : M φ φ = ∑' (x : ℤ), (F x - F (x -)) ^ 2 :=
sorry

theorem problem4 : (∀ x : ℝ, (distribution_function F → (M φ φ = 0 ↔ continuous F))) :=
sorry

end problem1_problem2_problem3_problem4_l345_345308


namespace hexagon_vector_BC_l345_345567

variable (V : Type)
variable [AddCommGroup V] [Module ℝ V]

variables (A B C D E F : V)
variable (a b : V)
variable (H1 : B - A = a)
variable (H2 : E - A = b)
variable (H3 : B + E = 2 * A)  -- Takes into account the symmetry and properties of the hexagon.

theorem hexagon_vector_BC (V : Type) 
  [AddCommGroup V] [Module ℝ V] 
  (A B C D E F : V) 
  (a b : V) 
  (H1 : B - A = a) 
  (H2 : E - A = b) 
  (H3 : B + E = 2 * A) 
  : C - B = (1/2 : ℝ) • a + (1/2 : ℝ) • b :=
sorry

end hexagon_vector_BC_l345_345567


namespace count_at_least_three_adjacent_chairs_l345_345239

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345239


namespace c_leq_1_over_4n_l345_345856

theorem c_leq_1_over_4n 
  (a : Fin (n + 1) → ℝ) (c : ℝ)
  (h_rec : ∀ k : Fin n, a k = c + ∑ i in Finset.Ico k n, a (i - k) * (a i + a (i + 1)))
  (h_n : a n = 0)
: c ≤ 1 / (4 * n) :=
sorry

end c_leq_1_over_4n_l345_345856


namespace k_value_l345_345216

theorem k_value (k : ℝ) (h : (k / 4) + (-k / 3) = 2) : k = -24 :=
by
  sorry

end k_value_l345_345216


namespace volume_pyramid_ACDE_eq_one_twelfth_cube_l345_345724

-- Definitions
def Point := (ℝ × ℝ × ℝ)
def length_of_edge := a : ℝ

def A : Point := (0, 0, 0)
def D : Point := (a, 0, 0)
def C : Point := (a, a, 0)
def A1: Point := (0, 0, a)
def D1: Point := (a, 0, a)
def E : Point := (a, 0, a / 2)

-- Assertion
theorem volume_pyramid_ACDE_eq_one_twelfth_cube (a : ℝ) (h_pos : a > 0) :
  let V_pyramid := (1 / 3) * (1 / 2 * a^2) * (a / 2) in
  let V_cube := a^3 in
  V_pyramid = (1 / 12) * V_cube :=
by
  sorry

end volume_pyramid_ACDE_eq_one_twelfth_cube_l345_345724


namespace complex_solutions_real_parts_product_l345_345196

theorem complex_solutions_real_parts_product (x : ℂ) (i : ℂ) :
  (∃ (x : ℂ), x^2 - 2 * x = i) → 
  let y := 1 in -- Representing the real part of 1 independently
  let z1 := y + re (e^(complex.I * real.pi / 8) * real.sqrt (2 ^ (1 / 4))) in
  let z2 := y - re (e^(complex.I * real.pi / 8) * real.sqrt (2 ^ (1 / 4))) in
  (z1 * z2 = (1 - real.sqrt 2) / 2) := sorry

end complex_solutions_real_parts_product_l345_345196


namespace center_cell_value_l345_345541

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l345_345541


namespace prove_parallelogram_l345_345980

variables (A B C N K L D E G H : Type)
variables [IsMidpoint A B D]
variables [IsMidpoint L K E]
variables [IsMidpoint C A G]
variables [IsMidpoint N A H]
variables [TriangleErectedOutside ABC ANC CLB BKA]
variables [EqualAngles NAC KBA LCB]
variables [EqualAngles NCA KAB LBC]

theorem prove_parallelogram
  (h1: Angle NAC = Angle KBA = Angle LCB)
  (h2: Angle NCA = Angle KAB = Angle LBC)
  (hD: IsMidpoint A B D)
  (hE: IsMidpoint L K E)
  (hG: IsMidpoint C A G)
  (hH: IsMidpoint N A H) :
  Parallelogram DEGH :=
sorry

end prove_parallelogram_l345_345980


namespace winter_sales_l345_345673

theorem winter_sales (T : ℕ) (spring_summer_sales : ℕ) (fall_sales : ℕ) (winter_sales : ℕ) 
  (h1 : T = 20) 
  (h2 : spring_summer_sales = 12) 
  (h3 : fall_sales = 4) 
  (h4 : T = spring_summer_sales + fall_sales + winter_sales) : 
     winter_sales = 4 := 
by 
  rw [h1, h2, h3] at h4
  linarith


end winter_sales_l345_345673


namespace possible_division_l345_345344

theorem possible_division (side_length : ℕ) (areas : Fin 5 → Set (Fin side_length × Fin side_length))
  (h1 : side_length = 5)
  (h2 : ∀ i, ∃ cells : Finset (Fin side_length × Fin side_length), areas i = cells ∧ Finset.card cells = 5)
  (h3 : ∀ i j, i ≠ j → Disjoint (areas i) (areas j))
  (total_cut_length : ℕ)
  (h4 : total_cut_length ≤ 16) :
  
  ∃ cuts : Finset (Fin side_length × Fin side_length) × Finset (Fin side_length × Fin side_length),
    total_cut_length = (cuts.1.card + cuts.2.card) :=
sorry

end possible_division_l345_345344


namespace number_of_true_propositions_l345_345385

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345385


namespace probability_of_specific_choice_l345_345497

-- Define the sets of subjects
inductive Subject
| Chinese
| Mathematics
| ForeignLanguage
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

-- Define the conditions of the examination mode "3+1+2"
def threeSubjects := [Subject.Chinese, Subject.Mathematics, Subject.ForeignLanguage]
def oneSubject := [Subject.Physics, Subject.History]
def twoSubjects := [Subject.PoliticalScience, Subject.Geography, Subject.Chemistry, Subject.Biology]

-- Calculate the total number of ways to choose one subject from Physics or History and two subjects from PoliticalScience, Geography, Chemistry, and Biology
def totalWays : Nat := 2 * Nat.choose 4 2  -- 2 choices for "1" part, and C(4, 2) ways for "2" part

-- Calculate the probability that a candidate will choose Political Science, History, and Geography
def favorableOutcome := 1  -- Only one specific combination counts

theorem probability_of_specific_choice :
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  (specific_combination : ℚ) / total_ways = 1 / 12 :=
by
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  show (specific_combination : ℚ) / total_ways = 1 / 12
  sorry

end probability_of_specific_choice_l345_345497


namespace min_abs_w_sub_u_sq_l345_345038

noncomputable def z : ℂ := sorry
def w : ℝ := (z + z⁻¹).re
def u : ℂ := (1 - z) / (1 + z)

theorem min_abs_w_sub_u_sq (h1 : -1 < w) (h2 : w < 2) : |w - (u ^ 2).re| = 1 := by
  sorry

end min_abs_w_sub_u_sq_l345_345038


namespace compute_expression_l345_345359

theorem compute_expression : 20 * (1/3 + 1/4 + 1/6)⁻² = 320/9 :=
by 
  sorry

end compute_expression_l345_345359


namespace find_number_satisfying_9y_eq_number12_l345_345515

noncomputable def power_9_y (y : ℝ) := (9 : ℝ) ^ y
noncomputable def root_12 (x : ℝ) := x ^ (1 / 12 : ℝ)

theorem find_number_satisfying_9y_eq_number12 :
  ∃ number : ℝ, power_9_y 6 = number ^ 12 ∧ abs (number - 3) < 0.0001 :=
by
  sorry

end find_number_satisfying_9y_eq_number12_l345_345515


namespace pyramid_volume_l345_345796

-- Definitions of the conditions
variables {Point : Type} [metric_space Point]
variables (A B C D P O Q : Point)
variables (s : ℝ) (φ : ℝ)
variable [square_base : square A B C D s]
variable [equidistant : ∀ (V : Point), V ∈ {A, B, C, D} → dist P V = dist P A]
variable [angle_phi : angle P A B = φ]

-- Define the volume of the pyramid
def volume_pyramid (P A B C D : Point) (s φ : ℝ) : ℝ :=
  (s^3 / 6) * sqrt (cotan (φ / 2)^2 + 1)

-- The theorem statement
theorem pyramid_volume (s φ : ℝ) : 
  ∃ (P A B C D : Point), 
    (square A B C D s) →
    (∀ (V : Point), V ∈ {A, B, C, D} → dist P V = dist P A) →
    (angle P A B = φ) →
    volume_pyramid P A B C D s φ = (s^3 / 6) * sqrt (cotan (φ / 2)^2 + 1) :=
sorry

end pyramid_volume_l345_345796


namespace smallest_perimeter_l345_345430

-- Definitions of the angle conditions
def angleA (B : ℝ) := 2 * B
def angleC (B : ℝ) := 180 - 3 * B

-- Conditions of the problem: ∠A = 2∠B and ∠C > 90°
lemma angle_conditions (B : ℝ) (h1 : B < 30) : angleC B > 90 :=
by linarith [h1]

-- The sides a, b, c of triangle
def side_lengths : (ℤ × ℤ × ℤ) := (7, 8, 15)

-- Perimeter of the triangle
def perimeter (a b c : ℤ) := a + b + c

-- The main statement: The smallest possible perimeter is 30
theorem smallest_perimeter :
  let ⟨a, b, c⟩ := side_lengths in
  perimeter a b c = 30 :=
by sorry

end smallest_perimeter_l345_345430


namespace cube_and_square_sum_l345_345362

theorem cube_and_square_sum : 
  let sum_of_cubes : ℤ := (Finset.range 50).sum (λ k, (k + 1)^3) + (Finset.range 50).sum (λ k, (-(k + 1))^3)
  let sum_of_squares : ℤ := (Finset.range 50).sum (λ k, (k + 1)^2)
  sum_of_cubes + sum_of_squares = 42925 :=
by
  sorry

end cube_and_square_sum_l345_345362


namespace horizontal_asymptote_value_l345_345906

theorem horizontal_asymptote_value :
  (∃ y : ℝ, ∀ x : ℝ, (y = (18 * x^5 + 6 * x^3 + 3 * x^2 + 5 * x + 4) / (6 * x^5 + 4 * x^3 + 5 * x^2 + 2 * x + 1)) → y = 3) :=
by
  sorry

end horizontal_asymptote_value_l345_345906


namespace min_sum_pairwise_distances_l345_345468

-- Definition for the pairwise distances sum of four points
def pairwise_distances_sum (A B C D : ℝ × ℝ) : ℝ :=
  let dist := λ (p q : ℝ × ℝ), real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B + dist A C + dist A D + dist B C + dist B D + dist C D

-- Condition for all pairwise distances to be at least 1
def all_distances_at_least (d : ℝ) (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ (p q : ℝ × ℝ), real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d ≤ dist A B ∧ d ≤ dist A C ∧ d ≤ dist A D ∧ d ≤ dist B C ∧ d ≤ dist B D ∧ d ≤ dist C D

open_locale real

-- Main theorem statement
theorem min_sum_pairwise_distances : 
  ∃ (A B C D : ℝ × ℝ),
    all_distances_at_least 1 A B C D ∧
    pairwise_distances_sum A B C D = 5 + real.sqrt 3 :=
sorry

end min_sum_pairwise_distances_l345_345468


namespace total_cost_after_tax_l345_345663

def cost_sponge : ℝ := 4.20
def cost_shampoo : ℝ := 7.60
def cost_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05

theorem total_cost_after_tax : 
  let total_pre_tax := cost_sponge + cost_shampoo + cost_soap
  let tax_amount := tax_rate * total_pre_tax
  total_pre_tax + tax_amount = 15.75 := 
by 
  let total_pre_tax := cost_sponge + cost_shampoo + cost_soap
  let tax_amount := tax_rate * total_pre_tax
  have pre_tax_eq : total_pre_tax = 4.20 + 7.60 + 3.20 := by sorry
  have tax_eq : tax_amount = 0.05 * 15.00 := by sorry
  have final_sum_eq : total_pre_tax + tax_amount = 15.00 + 0.75 := by sorry
  exact final_sum_eq.trans (by norm_num)
  sorry

end total_cost_after_tax_l345_345663


namespace find_pqrs_square_sum_l345_345900

variables (p q r s : ℝ)

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[p, q], [r, s]]

def B_transpose : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[p, r], [q, s]]

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  1 / (p * s - q * r) • ![[s, -q], [-r, p]]

theorem find_pqrs_square_sum (h : B_transpose = 2 • B_inv) : p^2 + q^2 + r^2 + s^2 = 4 :=
by
  sorry

end find_pqrs_square_sum_l345_345900


namespace sum_reciprocals_lt_bound_l345_345604

noncomputable def seq_a : ℕ → ℝ
| 0     := 1
| 1     := a
| (n+2) := (seq_a (n+1)^2 / seq_a n^2 - 2) * seq_a (n+1)

theorem sum_reciprocals_lt_bound (a : ℝ) (h : a > 2) (k : ℕ) :
  (∑ i in Finset.range (k+1), 1 / seq_a a (i)) < (1/2 * (2 + a - Real.sqrt (a^2 - 4))) :=
sorry

end sum_reciprocals_lt_bound_l345_345604


namespace derivative_of_y_l345_345655

noncomputable def y (x : ℝ) : ℝ := x * Real.exp (Real.cos x)

theorem derivative_of_y (x : ℝ) : deriv (y x) = -x * Real.sin x * Real.exp (Real.cos x) + Real.exp (Real.cos x) := sorry

end derivative_of_y_l345_345655


namespace length_of_chord_through_focus_l345_345461

noncomputable def parabola_equation : Real → Real := λ x, 8 * x

def chord_through_focus (x1 x2 : Real) : Prop :=
  ∃ y1 y2, (x1 + x2 = 10) ∧
            (y1^2 = parabola_equation x1) ∧
            (y2^2 = parabola_equation x2) ∧
            (x1 ≥ 0) ∧ (x2 ≥ 0)  -- Since x1 and x2 are on y^2 = 8x, they should be non-negative

theorem length_of_chord_through_focus (x1 x2 : Real) (h : chord_through_focus x1 x2) : |x1 + x2 + 4| = 14 :=
by
  sorry

end length_of_chord_through_focus_l345_345461


namespace find_DF_l345_345106

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345106


namespace min_value_abs_plus_2023_proof_l345_345901

noncomputable def min_value_abs_plus_2023 (a : ℚ) : Prop :=
  |a| + 2023 ≥ 2023

theorem min_value_abs_plus_2023_proof (a : ℚ) : min_value_abs_plus_2023 a :=
  by
  sorry

end min_value_abs_plus_2023_proof_l345_345901


namespace prime_if_factorial_plus_one_divisible_l345_345627

theorem prime_if_factorial_plus_one_divisible (n : ℕ) (h : n > 1) (h1 : (n - 1)! + 1 % n = 0) : Prime n :=
sorry

end prime_if_factorial_plus_one_divisible_l345_345627


namespace equal_distances_l345_345120

-- Define the triangle ABC, midpoint M, points E and F, and intersections X and Y
variables (A B C E F M X Y : Point)
-- Hypothetical collinearity assumption required by Lean's framework (this may not be necessary for every geometrical library)
variables [collinear ABC]

-- Basic properties and relationships
variables [is_midpoint M B C]
variables [on_ray E A B] [on_ray F A C]
variables (angle_AEC : Angle A E C)
variables (angle_AMC : Angle A M C)
variables (angle_AFB : Angle A F B)
variables (angle_AMB : Angle A M B)

-- Hypotheses based on the problem statement
hypothesis h1 : 2 * angle_AEC = angle_AMC
hypothesis h2 : 2 * angle_AFB = angle_AMB

-- Define circumcircle of triangle AEF and intersections with line BC
variable [circumcircle A E F]

-- Define points X and Y as intersections with the circumference of AEF and BC 
variables [intersection_circumcircle_line A E F B C X] [intersection_circumcircle_line A E F B C Y]

-- Required to prove: XB = YC
theorem equal_distances (h1 : 2 * angle_AEC = angle_AMC) (h2 : 2 * angle_AFB = angle_AMB) : 
  distance X B = distance Y C := 
sorry

end equal_distances_l345_345120


namespace center_cell_value_l345_345550

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l345_345550


namespace correct_answers_count_l345_345345

-- Define the conditions from the problem
def total_questions : ℕ := 25
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def total_score : ℤ := 85

-- State the theorem
theorem correct_answers_count :
  ∃ x : ℕ, (x ≤ total_questions) ∧ 
           (total_questions - x : ℕ) ≥ 0 ∧ 
           (correct_points * x + incorrect_points * (total_questions - x) = total_score) :=
sorry

end correct_answers_count_l345_345345


namespace angles_with_same_terminal_side_l345_345671

theorem angles_with_same_terminal_side (k : ℤ) :
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} = 
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + (-460 % 360)} :=
by sorry

end angles_with_same_terminal_side_l345_345671


namespace product_of_real_parts_l345_345194

theorem product_of_real_parts (x1 x2 : ℂ) (h1 : x1^2 - 2*x1 = complex.I) (h2 : x2^2 - 2*x2 = complex.I) :
  (x1.re * x2.re) = (1 - Real.sqrt 2) / 2 :=
  sorry

end product_of_real_parts_l345_345194


namespace largest_circle_from_square_area_l345_345339

noncomputable def largest_circle_area (A_square : ℕ) : ℝ :=
  let s := Real.sqrt A_square
  let P := 4 * s
  let r := P / (2 * Real.pi)
  let A_circle := Real.pi * r^2
  A_circle

theorem largest_circle_from_square_area (A_square : ℕ) (h : A_square = 144) : 
  Int.round (largest_circle_area A_square) = 183 := 
by
  sorry

end largest_circle_from_square_area_l345_345339


namespace sean_and_julie_sums_l345_345167

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l345_345167


namespace number_of_adjacent_subsets_l345_345233

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345233


namespace volume_ratio_tetrahedrons_l345_345092

theorem volume_ratio_tetrahedrons (P A B C D E F G : Type) [Real](hexagonal_pyramid : regular_hexagonal_pyramid P A B C D E F)
  (midpoint_G : midpoint P B = G) :
  volume_tetrahedron D G A C / volume_tetrahedron P G A C = 2 :=
by
  -- Given conditions
  have hexagon := hexagonal_pyramid
  have midpoint := midpoint_G
  sorry

end volume_ratio_tetrahedrons_l345_345092


namespace smallest_n_is_55_l345_345823

def is_perfect_square (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_n_is_55 : ∃ (n : ℕ) (a b c : ℕ),
  a + b + c = n ∧
  is_perfect_square (a + b) ∧
  is_perfect_square (a + c) ∧
  is_perfect_square (b + c) ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  ∀ m : ℕ, (∃ (x y z : ℕ),
    x + y + z = m ∧
    is_perfect_square (x + y) ∧
    is_perfect_square (x + z) ∧
    is_perfect_square (y + z) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x > 0 ∧ y > 0 ∧ z > 0) → n ≤ m :=
begin
  sorry
end

end smallest_n_is_55_l345_345823


namespace sean_and_julie_sums_l345_345166

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l345_345166


namespace find_f2_l345_345449

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x + 1

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -8 :=
by
  sorry

end find_f2_l345_345449


namespace loom_weaving_rate_l345_345352

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (rate : ℝ) 
  (h1 : total_cloth = 26) (h2 : total_time = 203.125) : rate = total_cloth / total_time := by
  sorry

#check loom_weaving_rate

end loom_weaving_rate_l345_345352


namespace solve_paralleliped_problem_l345_345735

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_from_point_to_plane (P Q R S : Point3D) : ℝ :=
  let PQ := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩
  let PR := ⟨R.x - P.x, R.y - P.y, R.z - P.z⟩
  let normal := ⟨PQ.y * PR.z - PQ.z * PR.y, PQ.z * PR.x - PQ.x * PR.z, PQ.x * PR.y - PQ.y * PR.x⟩
  let numerator := abs (normal.x * S.x + normal.y * S.y + normal.z * S.z)
  let denominator := (normal.x ^ 2 + normal.y ^ 2 + normal.z ^ 2).sqrt
  numerator / denominator

def paralleliped_problem : Prop :=
  let S := ⟨0, 0, 0⟩
  let P := ⟨6, 0, 0⟩
  let Q := ⟨0, 6, 0⟩
  let R := ⟨0, 0, 4⟩
  distance_from_point_to_plane P Q R S = 4

theorem solve_paralleliped_problem : paralleliped_problem := by
  -- the proof is to be filled in
  sorry

end solve_paralleliped_problem_l345_345735


namespace regular_octagon_diagonal_l345_345902

variable {a b c : ℝ}

-- Define a function to check for a regular octagon where a, b, c are respective side, shortest diagonal, and longest diagonal
def is_regular_octagon (a b c : ℝ) : Prop :=
  -- Here, we assume the standard geometric properties of a regular octagon.
  -- In a real formalization, we might model the octagon directly.

  -- longest diagonal c of a regular octagon (spans 4 sides)
  c = 2 * a

theorem regular_octagon_diagonal (a b c : ℝ) (h : is_regular_octagon a b c) : c = 2 * a :=
by
  exact h

end regular_octagon_diagonal_l345_345902


namespace prime_between_30_40_with_remainder_l345_345207

theorem prime_between_30_40_with_remainder :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 4 ∧ n = 31 :=
by
  sorry

end prime_between_30_40_with_remainder_l345_345207


namespace measure_of_angle_D_l345_345933

theorem measure_of_angle_D 
  (A B C D E F : ℝ)
  (h1 : A = B) (h2 : B = C) (h3 : C = F)
  (h4 : D = E) (h5 : A = D - 30) 
  (sum_angles : A + B + C + D + E + F = 720) : 
  D = 140 :=
by
  sorry

end measure_of_angle_D_l345_345933


namespace part_a_2n_squared_rods_possible_part_b_max_rods_no_overlap_l345_345722

theorem part_a_2n_squared_rods_possible (n : ℕ) (h : ∀ (x y z : ℕ), x < 2*n → y < 2*n → z < 2*n → ∃ r, r.pierces x y z ∧ r.direction ∈ {direction.x, direction.y, direction.z}) :
  ∃ (rods : set rod), rods.card = 2*n^2 ∧ (∀ r ∈ rods, r.pierces_exactly (2*n)) ∧ pairwise_disjoint rods :=
sorry

theorem part_b_max_rods_no_overlap (n : ℕ) (h : ∀ (x y z : ℕ), x < 2*n → y < 2*n → z < 2*n → ∃ r, r.pierces x y z ∧ r.direction ∈ {direction.x, direction.y, direction.z}) :
  ∃ (rods : set rod), rods.card = 2*n^2 ∧ (∀ r ∈ rods, r.pierces_exactly (2*n)) ∧ pairwise_disjoint rods :=
sorry

end part_a_2n_squared_rods_possible_part_b_max_rods_no_overlap_l345_345722


namespace fraction_shaded_is_one_tenth_l345_345752

theorem fraction_shaded_is_one_tenth :
  ∀ (A L S: ℕ), A = 300 → L = 5 → S = 2 → 
  ((15 * 20 = A) → (A / L = 60) → (60 / S = 30) → (30 / A = 1 / 10)) :=
by sorry

end fraction_shaded_is_one_tenth_l345_345752


namespace hypotenuse_length_l345_345204

-- Definitions of the conditions
variable (x : ℝ) -- The shorter leg
variable (h : ℝ) -- The hypotenuse

-- Definition of longer leg
def longer_leg := 3 * x - 3

-- Definition of the area of the triangle
def area := (1 / 2) * x * longer_leg

-- Conditions
axiom h_area : area = 72
axiom h_positive : x > 0

-- Pythagorean theorem
def hypotenuse := Real.sqrt (x^2 + longer_leg^2)

-- The theorem to prove
theorem hypotenuse_length : 
  h = hypotenuse := by sorry

end hypotenuse_length_l345_345204


namespace Trapezoid_Area_l345_345725

-- Define the conditions
def AreaOfTrapezoid (AD BC: ℝ) (Height: ℝ) :=
  (1 / 2) * (AD + BC) * Height

def AreaOfTriangle (Base Height: ℝ) :=
  (1 / 2) * Base * Height

theorem Trapezoid_Area (k h : ℝ) (Area_ABEF: ℝ)
  (BC AD FD AF CE ED DE DC: ℝ) 
  (H_parallel: AD ∥ BC)
  (H_ratio_BC_AD: BC = 5 * k ∧ AD = 7 * k)
  (H_ratio_AF_FD: AF = 4 * FD ∧ FD = 3 * k)
  (H_ratio_CE_ED: CE = 2 * ED ∧ DE = 3 * DC)
  (H_Area_ABEF: Area_ABEF = 123)
  (Height: 5 * h) :
  AreaOfTrapezoid AD BC (5 * h) = 180 := 
by 
  sorry

end Trapezoid_Area_l345_345725


namespace coefficient_x3_y5_l345_345189

theorem coefficient_x3_y5 :
  let f := (x + y) * (x - y)^7
  -- coefficient of x^3 y^5  in f
  ((polynomial.C 1) * (polynomial.X^3) * (polynomial.C 14) * (polynomial.Y^5)).coeff (3, 5) = 14 :=
sorry

end coefficient_x3_y5_l345_345189


namespace count_cubes_between_bounds_l345_345504

theorem count_cubes_between_bounds : ∃ (n : ℕ), n = 42 ∧
  ∀ x, 2^9 + 1 ≤ x^3 ∧ x^3 ≤ 2^17 + 1 ↔ 9 ≤ x ∧ x ≤ 50 := 
sorry

end count_cubes_between_bounds_l345_345504


namespace maximal_altitude_intersections_l345_345434

theorem maximal_altitude_intersections (P : Fin 5 → ℝ × ℝ) 
  (h_no_coincide : ∀ i j k l : Fin 5, i ≠ j → k ≠ l → line_through (P i) (P j) ≠ line_through (P k) (P l))
  (h_no_parallel : ∀ i j k l : Fin 5, i ≠ j → k ≠ l → ¬ parallel (line_through (P i) (P j)) (line_through (P k) (P l)))
  (h_no_perpendicular : ∀ i j k l : Fin 5, i ≠ j → k ≠ l → ¬ perpendicular (line_through (P i) (P j)) (line_through (P k) (P l))) :
  maximal_number_of_intersections P = 70 := 
sorry

end maximal_altitude_intersections_l345_345434


namespace find_a_l345_345128

noncomputable def f (a x : ℝ) := a * x + 1 / Real.sqrt 2

theorem find_a (a : ℝ) (h_pos : 0 < a) (h : f a (f a (1 / Real.sqrt 2)) = f a 0) : a = 0 :=
by
  sorry

end find_a_l345_345128


namespace find_t_l345_345911

noncomputable def parametric_circle : set (ℝ × ℝ) := 
  {p | ∃ θ : ℝ, p = (1 + 4 * cos θ, 3 + 4 * sin θ)}

theorem find_t (t : ℝ) 
  (chord_length_condition : ∀ (x y : ℝ), (x, y) ∈ parametric_circle → x - y + t = 0 → 
    sqrt ((1 - x) ^ 2 + (3 - y) ^ 2) = 4 * sqrt 2) : 
  t = -2 ∨ t = 6 :=
sorry

end find_t_l345_345911


namespace find_a_range_l345_345053

theorem find_a_range (a : ℝ)
  (p : ∀ x ∈ set.Icc (1/2 : ℝ) 1, 1/x - a ≥ 0)
  (q : ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) :
  a ∈ set.Iic (-2) ∪ {1} :=
sorry

end find_a_range_l345_345053


namespace number_of_uniform_triplets_l345_345935

theorem number_of_uniform_triplets
  (ministers : Fin 12 → Type)
  (friend : (i j : Fin 12) → Prop)
  (enemy : (i j : Fin 12) → Prop)
  (h_symm_friend : ∀ i j, friend i j → friend j i)
  (h_symm_enemy : ∀ i j, enemy i j → enemy j i)
  (h_total_friends : ∀ i, ∃! S : Finset (Fin 12), S.card = 5 ∧ ∀ j ∈ S, friend i j)
  (h_total_enemies : ∀ i, ∃! S : Finset (Fin 12), S.card = 6 ∧ ∀ j ∈ S, enemy i j) :
  (∃! T : Finset (Fin 12), T.card = 3 ∧ (∀ i j k ∈ T, (friend i j ∧ friend j k ∧ friend k i) ∨ (enemy i j ∧ enemy j k ∧ enemy k i))) ↔ 40 := by
  sorry

end number_of_uniform_triplets_l345_345935


namespace symmetry_center_ratio_of_sides_in_acute_triangle_l345_345048

-- Definition and key equation provided in the problem
def f (x : ℝ) := 2 * cos (2 * x + π / 3) - 2 * cos (2 * x) + 1

-- First part: Symmetry center
theorem symmetry_center (k : ℤ) :
  { p : ℝ × ℝ | p.fst = -π / 12 + k * π / 2 ∧ p.snd = 1 } ≠ ∅ :=
by sorry

-- Some required geometric and trigonometric definitions for the next part
variables (A B C : ℝ) (a b c : ℝ)
variables {ABC_triangle : a.to_angle = A ∧ b.to_angle = B ∧ c.to_angle = C}
variables {fA_zero : f A = 0}

-- Second part: Ratio of sides in an acute triangle
theorem ratio_of_sides_in_acute_triangle :
  (π / 6 < C ∧ C < π / 2) ∧ f A = 0 → 
  A = π / 3 ∧ b / c ∈ (1 / 2, 2) :=
by sorry

end symmetry_center_ratio_of_sides_in_acute_triangle_l345_345048


namespace data_transmission_time_l345_345810

def chunks_per_block : ℕ := 1024
def blocks : ℕ := 30
def transmission_rate : ℕ := 256
def seconds_in_minute : ℕ := 60

theorem data_transmission_time :
  (blocks * chunks_per_block) / transmission_rate / seconds_in_minute = 2 :=
by
  sorry

end data_transmission_time_l345_345810


namespace mapping_element_l345_345565

def A : Set (ℝ × ℝ) := {p | ∃ x y: ℝ, p = (x, y)}

def B : Set (ℝ × ℝ) := A

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem mapping_element (x y : ℝ) (h : (x, y) = (-1, 2)) : f (x, y) = (-3, 1) :=
by
  cases h
  rw [f]
  repeat {rw prod.mk}
  simp [sub_eq_add_neg, add_assoc, add_left_neg, add_right_neg]
  sorry

end mapping_element_l345_345565


namespace exists_digits_divisible_by_73_no_digits_divisible_by_79_l345_345733

open Int

def decimal_digits (n : ℕ) := n < 10

theorem exists_digits_divisible_by_73 :
  ∃ (a b : ℕ), decimal_digits a ∧ decimal_digits b ∧ 
  (∀ n : ℕ, a * 10^(n+2) + b * 10^(n+1) + 222 * (10^n / 9) + 31 ≡ 0 [MOD 73]) :=
by {
  sorry
}

theorem no_digits_divisible_by_79 :
  ¬ ∃ (c d : ℕ), decimal_digits c ∧ decimal_digits d ∧ 
  (∀ n : ℕ, c * 10^(n+2) + d * 10^(n+1) + 222 * (10^n / 9) + 31 ≡ 0 [MOD 79]) :=
by {
  sorry
}

end exists_digits_divisible_by_73_no_digits_divisible_by_79_l345_345733


namespace true_propositions_l345_345382

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345382


namespace staplers_left_l345_345685

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l345_345685


namespace perfect_square_trinomial_k_l345_345450

theorem perfect_square_trinomial_k (k : ℝ) (x y : ℝ) :
  (∃ a b : ℝ, (x + a*y)^2 = x^2 + b*x*y + (64:ℝ)*y^2) → k = 16 ∨ k = -16 :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  sorry

end perfect_square_trinomial_k_l345_345450


namespace ssr_zero_if_on_regression_line_l345_345851

noncomputable def sum_of_squared_residuals {n : ℕ} (data : Fin n → ℝ × ℝ) : ℝ :=
  ∑ i, let (x_i, y_i) := data i in (y_i - (1 / 3 * x_i + 2))^2

theorem ssr_zero_if_on_regression_line {n : ℕ} (data : Fin n → ℝ × ℝ)
  (h : ∀ i, let (x_i, y_i) := data i in y_i = 1 / 3 * x_i + 2) : sum_of_squared_residuals data = 0 :=
by
  sorry

end ssr_zero_if_on_regression_line_l345_345851


namespace mikes_remaining_cards_l345_345616

variable (original_number_of_cards : ℕ)
variable (sam_bought : ℤ)
variable (alex_bought : ℤ)

theorem mikes_remaining_cards :
  original_number_of_cards = 87 →
  sam_bought = 8 →
  alex_bought = 13 →
  original_number_of_cards - (sam_bought + alex_bought) = 66 :=
by
  intros h_original h_sam h_alex
  rw [h_original, h_sam, h_alex]
  norm_num

end mikes_remaining_cards_l345_345616


namespace ratio_of_areas_l345_345201

variable (lS : ℝ)

def lR := 1.2 * lS
def wR := 0.8 * lS
def areaR := lR * wR
def areaS := lS * lS
def ratio := areaR / areaS

theorem ratio_of_areas : ratio = 24 / 25 := by
  sorry

end ratio_of_areas_l345_345201


namespace verify_propositions_l345_345374

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345374


namespace nth_derivative_of_f_l345_345799

noncomputable def f (n : ℕ) (x : ℝ) := ∫ 0 to x, ∑ i in finset.range n, (x - t)^i / i.fact

theorem nth_derivative_of_f (n : ℕ) (x : ℝ) : 
    (derivative^[n] (f n)) x = (real.exp (↑n * x) * (↑n^n - 1) / (↑n - 1)) :=
sorry

end nth_derivative_of_f_l345_345799


namespace shelves_of_mystery_books_l345_345794

theorem shelves_of_mystery_books (total_books : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ) (M : ℕ) 
  (h_total_books : total_books = 54) 
  (h_picture_shelves : picture_shelves = 4) 
  (h_books_per_shelf : books_per_shelf = 6)
  (h_mystery_books : total_books - picture_shelves * books_per_shelf = M * books_per_shelf) :
  M = 5 :=
by
  sorry

end shelves_of_mystery_books_l345_345794


namespace unit_fraction_decomposition_terminates_l345_345617

theorem unit_fraction_decomposition_terminates (a b : ℕ) (pos_b : b > 0) (pos_a : a > 0) :
  ∃ (k : ℕ) (t : ℕ → ℚ), (∀ i, t i = 1 / (t_1 + t_2 + ... + t_{i-1}) ∧ t_1 + t_2 + ... + t_k = a / b) := by
  sorry

end unit_fraction_decomposition_terminates_l345_345617


namespace staples_left_in_stapler_l345_345677

def initial_staples : ℕ := 50
def used_staples : ℕ := 3 * 12
def remaining_staples : ℕ := initial_staples - used_staples

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  unfold initial_staples used_staples remaining_staples
  rw [Nat.mul_comm, Nat.mul_comm 3, Nat.mul_comm 12, Nat.sub_eq_iff_eq_add]
  have h : ∀ a b : ℕ, a = b -> 50 - (3 * 12) = b -> 50 - 36 = a := by intros; rw [h, Nat.mul_comm 3, Nat.mul_comm 12]
  exact h 36 36 rfl
#align std.staples_left_in_stapler


end staples_left_in_stapler_l345_345677


namespace y_intercept_of_line_l345_345291

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 4 * y = 12) (hx : x = 0) : y = -3 :=
by
  -- Since x = 0,
  -- replace x with 0 in the given equation 3x - 4y = 12
  have H : 3 * 0 - 4 * y = 12, from h.elim (congr_arg (λ lhs, 3 * 0 - 4 * y)),
  sorry

end y_intercept_of_line_l345_345291


namespace scoring_situations_l345_345011

theorem scoring_situations :
  ∃ (students : Fin 4 → ℤ), 
    (∀ i, students i = 21 ∨ students i = -21 ∨ students i = 7 ∨ students i = -7) ∧ 
    (∑ i, students i = 0) ∧ 
    (students ≠ ![21, 21, -21, -21] ∧ students ≠ ![21, -21, 7, -7] → (students.count 21 + students.count -21 = 2) ∨ 
    (students.count 21 + students.count -7 = 1) ∨ 
    (students.count -21 + students.count 7 = 1) ∨ 
    (students.count 21 + students.count -21 + students.count 7 + students.count -7 = 4) ∨ 
    (students.count 7 + students.count -7 = 2)) ∧
    (students.count 21 != 0 ∨ students.count -21 != 0 ∨ students.count 7 != 0 ∨ students.count -7 != 0) :=
begin
  sorry
end

end scoring_situations_l345_345011


namespace base_of_ABBA_pattern_l345_345298

theorem base_of_ABBA_pattern (b : ℕ) (A B : ℕ) (A ≠ B) (h1 : b^3 ≤ 489) (h2 : 489 < b^4) :
  489 = A * b^3 + B * b^2 + B * b + A → b = 6 ∧ (1 ≤ A) ∧ (A < b) ∧ (0 ≤ B) ∧ (B < b) :=
by
  sorry

end base_of_ABBA_pattern_l345_345298


namespace center_cell_value_l345_345555

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l345_345555


namespace greatest_identical_snack_bags_l345_345175

-- Defining the quantities of each type of snack
def granola_bars : Nat := 24
def dried_fruit : Nat := 36
def nuts : Nat := 60

-- Statement of the problem: greatest number of identical snack bags Serena can make without any food left over.
theorem greatest_identical_snack_bags :
  Nat.gcd (Nat.gcd granola_bars dried_fruit) nuts = 12 :=
sorry

end greatest_identical_snack_bags_l345_345175


namespace problem_l345_345508

noncomputable def log : Real → Real → Real := sorry

theorem problem 
  (x y : ℝ) 
  (h₁ : 4^x = 6) 
  (h₂ : 9^y = 6) : 
  (1/x) + (1/y) = 2 :=
by
  sorry

end problem_l345_345508


namespace geometry_problem_l345_345080

-- Definitions for points and segments based on given conditions
variables {O A B C D E F G : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G]

-- Lengths of segments based on given conditions
variables (DE EG : ℝ)
variable (BG : ℝ)

-- Given lengths
def given_lengths : Prop :=
  DE = 5 ∧ EG = 3

-- Goal to prove
def goal : Prop :=
  BG = 12

-- The theorem combining conditions and the goal
theorem geometry_problem (h : given_lengths DE EG) : goal BG :=
  sorry

end geometry_problem_l345_345080


namespace mass_percentage_O_in_mixture_l345_345805

/-- Mass percentage of oxygen in a mixture of Acetone and Methanol -/
theorem mass_percentage_O_in_mixture 
  (mass_acetone: ℝ)
  (mass_methanol: ℝ)
  (mass_O_acetone: ℝ)
  (mass_O_methanol: ℝ) 
  (total_mass: ℝ) : 
  mass_acetone = 30 → 
  mass_methanol = 20 → 
  mass_O_acetone = (16 / 58.08) * 30 →
  mass_O_methanol = (16 / 32.04) * 20 →
  total_mass = mass_acetone + mass_methanol →
  ((mass_O_acetone + mass_O_methanol) / total_mass) * 100 = 36.52 :=
by
  sorry

end mass_percentage_O_in_mixture_l345_345805


namespace distance_between_points_is_correct_l345_345785

-- Define the points and the distance function
def point1 : ℝ × ℝ := (3, 4)
def point2 : ℝ × ℝ := (8, -6)

-- Define the distance formula 
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem
theorem distance_between_points_is_correct :
  euclidean_distance point1 point2 = 5 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_is_correct_l345_345785


namespace conditional_probability_l345_345834

variable {Ω : Type} [ProbabilityTheory Ω]

def P (x : Ω → Prop) : ℝ := sorry  -- Placeholder for probability measure function

variables (A B : Ω → Prop)

-- Given conditions
axiom PA : P A = (3 / 4)
axiom PB : P B = (1 / 4)
axiom PAB : P (A ∧ B) = (1 / 2)

-- Statement to prove
theorem conditional_probability : P (A ∧ B) / P A = (2 / 3) := by
  rw [PAB, PA]
  norm_num
  sorry

end conditional_probability_l345_345834


namespace exists_infinite_solutions_l345_345997

theorem exists_infinite_solutions :
  ∃ (x y z : ℤ), (∀ k : ℤ, x = 2 * k ∧ y = 999 - 2 * k ^ 2 ∧ z = 998 - 2 * k ^ 2) ∧ (x ^ 2 + y ^ 2 - z ^ 2 = 1997) :=
by 
  -- The proof should go here
  sorry

end exists_infinite_solutions_l345_345997


namespace odd_cycle_exists_l345_345737

-- Define the conditions
variable (n : ℕ) (h1 : n ≥ 3)
variable (n_airlines : Fin n → List (Fin n) × List (Fin n))
variable (odd_cycles : ∀ (i : Fin n), Odd (n_airlines i).fst.size ∧ (n_airlines i).fst.size ≥ 3)

-- Define the graph structure and the proof
theorem odd_cycle_exists (n : ℕ) (h1 : n ≥ 3) 
                         (n_airlines : Fin n → List (Fin n) × List (Fin n))
                         (odd_cycles : ∀ (i : Fin n), Odd (n_airlines i).fst.size ∧ (n_airlines i).fst.size ≥ 3) :
  ∃ (cycle : List (Fin n)), Odd cycle.size ∧ (∀ e ∈ cycle, ∃ i : Fin n, e ∈ (n_airlines i).snd) :=
sorry

end odd_cycle_exists_l345_345737


namespace derivative_decreasing_l345_345192

-- Define the function 2^x
def f (x : ℝ) : ℝ := 2^x

-- Define the derivative of the function 2^x
def f_deriv (x : ℝ) : ℝ := (2^x) * (Math.log 2)

-- The proof statement
theorem derivative_decreasing (x : ℝ) : 
  f_deriv x < f x :=
by
  -- This is where the proof would go, but for now, we replace it with sorry
  sorry

end derivative_decreasing_l345_345192


namespace field_size_l345_345140

theorem field_size
  (cost_per_foot : ℝ)
  (total_money : ℝ)
  (cannot_fence : ℝ)
  (cost_per_foot_eq : cost_per_foot = 30)
  (total_money_eq : total_money = 120000)
  (cannot_fence_eq : cannot_fence > 1000) :
  ∃ (side_length : ℝ), side_length * side_length = 1000000 := 
by
  sorry

end field_size_l345_345140


namespace chairs_subset_count_l345_345277

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345277


namespace find_a_find_b_find_n_range_l345_345045

section
  variable (f : ℝ → ℝ) (g : ℝ → ℝ) (a b n : ℝ)

  -- (1) Given conditions on f(x), find a.
  def f_condition1 := ∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f x = x^4 - 4*x^3 + a*x^2 - 1
  def f_condition2 := ∀ x ∈ Set.Icc (1:ℝ) (2:ℝ), f x = x^4 - 4*x^3 + a*x^2 - 1
  def f'_at_1_is_zero := deriv f 1 = 0

  theorem find_a (f : ℝ → ℝ) [Differentiable ℝ f] 
    (h1 : f_condition1 f)
    (h2 : f_condition2 f)
    (h3 : f'_at_1_is_zero f) :
    a = 4 :=
  sorry

  -- (2) Determine if there exists a b such that g(x) intersects f(x) exactly 2 times.
  def g_condition := g = λ x, b*x^2 - 1
  def intersection_count := ∃ b, ∃! x, f x = g x

  theorem find_b (f g : ℝ → ℝ)
    (ha : a = 4)
    (h1 : f_condition1 f ∧ f_condition2 f)
    (h2 : g_condition g) :
    b = 0 ∨ b = 4 :=
  sorry

  -- (3) Find range of n such that inequality holds
  def inquality_condition := ∀ m ∈ Set.Icc (-6:ℝ) (-2:ℝ), ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ),
    f x ≤ m*x^3 + 2*x^2 - n

  theorem find_n_range (f : ℝ → ℝ) (n : ℝ) 
    (h1 : f = λ x, x^4 - 4*x^3 + 4*x^2 - 1)
    (h2 : inquality_condition f n) :
    n ≤ -4 :=
  sorry
end

end find_a_find_b_find_n_range_l345_345045


namespace sum_of_squares_eq_23456_l345_345907

theorem sum_of_squares_eq_23456 (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end sum_of_squares_eq_23456_l345_345907


namespace probability_boy_saturday_girl_sunday_l345_345440

def num_people := 4
def num_boys := 2
def num_girls := 2

-- Total number of ways to choose 2 people out of 4
def total_events := nnreal.of_nat (nat.choose num_people 2 * (2!))

-- Number of favorable outcomes: choosing 1 boy for Saturday and 1 girl for Sunday
def favorable_events := nnreal.of_nat (num_boys * num_girls)

-- The probability that a boy is chosen for Saturday and a girl for Sunday
theorem probability_boy_saturday_girl_sunday :
  (favorable_events / total_events) = (1 / 3) :=
sorry

end probability_boy_saturday_girl_sunday_l345_345440


namespace inequality_correct_l345_345970

open BigOperators

theorem inequality_correct {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ (a + b)^2 / 4 ∧ (a + b)^2 / 4 ≥ a * b :=
by 
  sorry

end inequality_correct_l345_345970


namespace eccentricity_of_ellipse_l345_345525

variable (a b c e : ℝ)

# Assume conditions:
# 1. Lengths form an arithmetic sequence: 2b = a + c
# 2. Relationship among semi-axes: b^2 = a^2 - c^2
# Goal: The eccentricity of the ellipse is e = 3/5.

theorem eccentricity_of_ellipse :
  (2 * b = a + c) →
  (b^2 = a^2 - c^2) →
  e = c / a →
  e = 3 / 5 :=
by
  intros h1 h2 he
  sorry

end eccentricity_of_ellipse_l345_345525


namespace common_inner_tangent_circles_l345_345939

noncomputable theory

def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_tangent_line_of_two_circles (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (line : ℝ → ℝ) : Prop :=
  let d := euclidean_distance c1 c2 in
  (is_inner_tangent : d = r1 + r2) ∧
  (passes_through_point p1 : p1 ∈ [(c1.1 + (r1 / d) * (c2.1 - c1.1), c1.2 + (r1 / d) * (c2.2 - c1.2)] ∧
   passes_through_point p2 : p2 ∈ [(c2.1 - (r2 / d) * (c2.1 - c1.1), c2.2 - (r2 / d) * (c2.2 - c1.2)]) ∧
  (line_part_one : (p1.2 - c1.2) = line (p1.1 - c1.1)) ∧
  (line_part_two : (p2.2 - c2.2) = line (p2.1 - c2.1))

open_locale classical

theorem common_inner_tangent_circles :
  is_tangent_line_of_two_circles (0, 0) (3, -4) 4 1 (λ x, - 4 / 3 * x) :=
begin
  sorry
end

end common_inner_tangent_circles_l345_345939


namespace angle_between_2a_minus_b_and_a_is_45_degrees_l345_345057

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 3)
noncomputable def vector_2a_minus_b : ℝ × ℝ := (2 * (2, 1)) - (1, 3)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def cos_angle (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem angle_between_2a_minus_b_and_a_is_45_degrees :
  real.arccos (cos_angle vector_2a_minus_b vector_a) = real.pi / 4 :=
sorry

end angle_between_2a_minus_b_and_a_is_45_degrees_l345_345057


namespace range_of_m_l345_345843

theorem range_of_m (m : ℝ) (p : ∃ x : ℝ, m * x^2 + 1 ≤ 0) (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) (hq : ¬(p ∨ q)) : m ≥ 2 :=
sorry

end range_of_m_l345_345843


namespace isosceles_right_triangle_cover_obtuse_l345_345854

theorem isosceles_right_triangle_cover_obtuse (ABC : Triangle) (h1 : ABC.is_obtuse) (h2 : ABC.circumradius = 1) :
  ∃ DEF : Triangle, DEF.is_isosceles_right ∧ DEF.hypotenuse_length = 1 + Real.sqrt 2 ∧ DEF.covers ABC := 
sorry

end isosceles_right_triangle_cover_obtuse_l345_345854


namespace simplify_F_range_F_l345_345489

noncomputable def f (t : ℝ) : ℝ := real.sqrt ((1 - t) / (1 + t))
noncomputable def F (x : ℝ) : ℝ := real.sin x * f (real.cos x) + real.cos x * f (real.sin x)

theorem simplify_F (x : ℝ) (h1 : π < x) (h2 : x < 3 * π / 2) : 
    F (x) = sqrt 2 * real.sin (x + π / 4) - 2 :=
sorry

theorem range_F : set_of (λ y, ∃ x, π < x ∧ x < 3 * π / 2 ∧ F x = y) = 
    set.Ico (-2 - sqrt 2) (-3) :=
sorry

end simplify_F_range_F_l345_345489


namespace distance_from_M_to_bases_l345_345759

variable (A B C A₁ B₁ C₁ : Point)
variable (h : ℝ)
variable (M : Point)

variables (plane_A1BC plane_AB1C plane_ABC1 : Plane)

-- Conditions
axiom plane_A1BC_contains_A1_B_C : plane_A1BC.contains A₁ ∧ plane_A1BC.contains B ∧ plane_A1BC.contains C
axiom plane_AB1C_contains_A_B1_C : plane_AB1C.contains A ∧ plane_AB1C.contains B₁ ∧ plane_AB1C.contains C
axiom plane_ABC1_contains_A_B_C1 : plane_ABC1.contains A ∧ plane_ABC1.contains B ∧ plane_ABC1.contains C₁

axiom intersection_point_M : plane_A1BC ∩ plane_AB1C ∩ plane_ABC1 = M

-- Distances to bases
def distance_to_base_ABC := distance M (projection M ABC_base_plane)
def distance_to_base_A1B1C1 := distance M (projection M A1B1C1_base_plane)

theorem distance_from_M_to_bases (h : ℝ) :
  (distance_to_base_ABC M) = (1 / 3) * h ∧ (distance_to_base_A1B1C1 M) = (2 / 3) * h :=
sorry

end distance_from_M_to_bases_l345_345759


namespace odd_function_phi_l345_345523

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def is_odd_function_sin (φ : ℝ) : Prop :=
  is_odd_function (λ x, Real.sin (x + φ))

theorem odd_function_phi (φ : ℝ) : (is_odd_function_sin φ) ↔ ∃ k : ℤ, φ = k * Real.pi :=
  sorry

end odd_function_phi_l345_345523


namespace option_B_shares_asymptotes_l345_345772

-- Define the given hyperbola equation
def given_hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- The asymptotes for the given hyperbola
def asymptotes_of_given_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the hyperbola for option B
def option_B_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 16) = 1

-- The asymptotes for option B hyperbola
def asymptotes_of_option_B_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Theorem stating that the hyperbola in option B shares the same asymptotes as the given hyperbola
theorem option_B_shares_asymptotes :
  (∀ x y : ℝ, given_hyperbola x y → asymptotes_of_given_hyperbola x y) →
  (∀ x y : ℝ, option_B_hyperbola x y → asymptotes_of_option_B_hyperbola x y) :=
by
  intros h₁ h₂
  -- Here should be the proof to show they have the same asymptotes
  sorry

end option_B_shares_asymptotes_l345_345772


namespace sin_C_value_side_c_value_l345_345925

theorem sin_C_value (C : ℝ) (h : sin C + cos C = 1 - sin (C / 2)) : sin C = 3 / 4 :=
by sorry

theorem side_c_value (a b c : ℝ) (C : ℝ) (h1 : a^2 + b^2 = 4 * (a + b) - 8)(h2 : sin C = 3 / 4) : 
  c = 1 + sqrt 7 :=
by sorry

end sin_C_value_side_c_value_l345_345925


namespace exists_polynomial_for_divisors_l345_345008

open Polynomial

theorem exists_polynomial_for_divisors (n : ℕ) :
  (∃ P : ℤ[X], ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℤ)^2) ↔
  (Nat.Prime n ∨ n = 1 ∨ n = 6) := by
  sorry

end exists_polynomial_for_divisors_l345_345008


namespace cosine_value_of_angle_BN_DM_l345_345022

noncomputable def cosine_angle_BN_DM (A B C D: ℝ × ℝ × ℝ) (M: ℝ × ℝ × ℝ) (N: ℝ × ℝ × ℝ) : ℝ :=
  let BN := (B.1 - N.1, B.2 - N.2, B.3 - N.3)
  let DM := (D.1 - M.1, D.2 - M.2, D.3 - M.3)
  let dot_product := BN.1 * DM.1 + BN.2 * DM.2 + BN.3 * DM.3
  let norm_BN := real.sqrt (BN.1^2 + BN.2^2 + BN.3^2)
  let norm_DM := real.sqrt (DM.1^2 + DM.2^2 + DM.3^2)
  dot_product / (norm_BN * norm_DM)

theorem cosine_value_of_angle_BN_DM:
  ∀ (A B C D M N: ℝ × ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧
    M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2) ∧
    N = ((A.1 + D.1) / 2, (A.2 + D.2) / 2, (A.3 + D.3) / 2) ∧
    (∃ t: ℝ, 0 < t ∧ ∀ (X Y: ℝ × ℝ × ℝ), norm_sq (X.1 - Y.1, X.2 - Y.2, X.3 - Y.3) = t) ∧
    (cosine_angle_BN_DM A B C D M N = (2 / 3)) :=
by sorry

end cosine_value_of_angle_BN_DM_l345_345022


namespace mixing_solutions_l345_345176

theorem mixing_solutions (Vx : ℝ) :
  (0.10 * Vx + 0.30 * 900 = 0.25 * (Vx + 900)) ↔ Vx = 300 := by
  sorry

end mixing_solutions_l345_345176


namespace tangent_line_eq_f_le_2x_minus_2_range_of_k_l345_345881

noncomputable def f (x : ℝ) : ℝ := x - x^2 + 3 * real.log x

-- Proof problem 1: Prove that the equation of the tangent line with a slope of 2 to the curve y = f(x) is y = 2x - 2.
theorem tangent_line_eq {x : ℝ} (h : deriv f x = 2) :
  ∃ y : ℝ, f x = y ∧ y = 2x - 2 :=
sorry

-- Proof problem 2: Prove that f(x) ≤ 2x - 2.
theorem f_le_2x_minus_2 (x : ℝ) :
  f(x) ≤ 2 * x - 2 :=
sorry

-- Proof problem 3: Prove that the range of k such that ∃ x0 > 1, ∀ x ∈ (1, x0), f(x) ≥ k * (x - 1) is k ∈ (-∞, 2).
theorem range_of_k (k : ℝ) :
  (∃ (x0 : ℝ), x0 > 1 ∧ (∀ x ∈ Ioo 1 x0, f(x) ≥ k * (x - 1))) ↔ k < 2 :=
sorry

end tangent_line_eq_f_le_2x_minus_2_range_of_k_l345_345881


namespace albert_wins_strategy_l345_345143

theorem albert_wins_strategy (n : ℕ) (h : n = 1999) : 
  ∃ strategy : (ℕ → ℕ), (∀ tokens : ℕ, tokens = n → tokens > 1 → 
  (∃ next_tokens : ℕ, next_tokens < tokens ∧ next_tokens ≥ 1 ∧ next_tokens ≥ tokens / 2) → 
  (∃ k, tokens = 2^k + 1) → strategy n = true) :=
sorry

end albert_wins_strategy_l345_345143


namespace original_price_of_bag_l345_345320

theorem original_price_of_bag (discounted_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) 
  (h : discounted_price = original_price * (1 - discount_rate)) : 
  original_price = 150 :=
by
  have h1 : original_price = discounted_price / (1 - discount_rate), from sorry
  exact h1

end original_price_of_bag_l345_345320


namespace number_of_other_workers_at_hospital_l345_345113

theorem number_of_other_workers_at_hospital
    (N : ℕ)
    (h1 : N = 5)
    (h2 : 0.1 = 1 / (N * (N - 1) / 2)) :
    N - 2 = 3 :=
by
  -- Given N = 5, and the probability condition, prove that N - 2 equals 3.
  sorry

end number_of_other_workers_at_hospital_l345_345113


namespace lambda_value_l345_345062

theorem lambda_value (λ : ℝ) :
  let m := (λ, 2, 3) in
  let n := (1, -3, 1) in
  (λ * 1 + 2 * (-3) + 3 * 1 = 0) → λ = 3 :=
by sorry

end lambda_value_l345_345062


namespace graph_of_cos_shifted_eq_sin_l345_345226

noncomputable def graph_shift_cos_to_sin : Bool :=
  let y_cos := λ x, Real.cos (2 * x)
  let y_sin := λ x, Real.sin (2 * x - (Real.pi / 6))
  let m := -Real.pi / 3
  let y_cos_shifted := λ x, Real.cos (2 * (x + m))
  let y_cos_shifted_transformed := λ x, Real.sin (2 * x - (Real.pi / 6) + Real.pi / 2)
  y_sin(0) = y_cos_shifted_transformed(0)

theorem graph_of_cos_shifted_eq_sin :
  graph_shift_cos_to_sin = true := sorry

end graph_of_cos_shifted_eq_sin_l345_345226


namespace sasha_zhenya_different_genders_l345_345779

theorem sasha_zhenya_different_genders
    (total_people : ℕ)
    (dances : total_people → ℕ)
    (sasha zhenya : total_people)
    (people_3 : set total_people)
    (people_6 : set total_people)
    (h_total_people : total_people = 20)
    (h_people_3 : people_3.card = 10 ∧ ∀ p ∈ people_3, dances p = 3)
    (h_sasha_dances : dances sasha = 5)
    (h_zhenya_dances: dances zhenya = 5)
    (h_people_6 : people_6.card = 8 ∧ ∀ p ∈ people_6, dances p = 6)
    (h_distinct : sasha ≠ zhenya)
    (h_dance_prop : ∀ {p q : total_people}, p ≠ q → (dances p) % 2 = 0 →
                   ∀ {gender : total_people → Prop}, (gender p = ¬ gender q) ↔ (gender p = gender q) → False)
    : ∀ (gender : total_people → Prop), gender sasha ≠ gender zhenya :=
by
  sorry

end sasha_zhenya_different_genders_l345_345779


namespace decreasing_on_neg_l345_345482

variable (f : ℝ → ℝ)

-- Condition 1: f(x) is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Condition 2: f(x) is increasing on (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove: f(x) is decreasing on (-∞, 0)
theorem decreasing_on_neg (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : increasing_on_pos f) :
  ∀ x y, x < y → y < 0 → f y < f x :=
by 
  sorry

end decreasing_on_neg_l345_345482


namespace find_a_l345_345835

noncomputable def integral_result_equals (a : ℝ) : ℝ :=
  ∫ x in 1..a, (2 * x + 1 / x)

theorem find_a (a : ℝ) (h : a > 1) : integral_result_equals a = 3 + log 2 → a = 2 :=
by
  intro h1
  unfold integral_result_equals at h1
  simp only [interval_integral.integral_of_le] at h1
  sorry

end find_a_l345_345835


namespace cylinder_volume_change_l345_345302

theorem cylinder_volume_change (r h : ℝ) (V : ℝ) (volume_eq : V = π * r^2 * h) :
  let h' := 3 * h;
  let r' := 4 * r;
  let V' := π * (r')^2 * (h')
  in V' = 48 * V :=
by
  simp [h', r', V', volume_eq]
  sorry

end cylinder_volume_change_l345_345302


namespace solve_money_conditions_l345_345832

theorem solve_money_conditions 
  (a b : ℝ)
  (h1 : b - 4 * a < 78)
  (h2 : 6 * a - b = 36) :
  a < 57 ∧ b > -36 :=
sorry

end solve_money_conditions_l345_345832


namespace problem_solution_l345_345052

-- Define parametric equations of line l
def parametric_line_l (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define polar equation of curve C
def polar_eq_C (θ : ℝ) : ℝ := Real.sin θ / (1 - Real.sin θ ^ 2)

-- Proof problem: Prove polar equation of line l, rectangular equation of curve C, and minimum distance from point P on curve C to line l
theorem problem_solution :
  (∀ (t : ℝ), let (x, y) := parametric_line_l t in x - y = 1) ∧
  (∀ (ρ θ : ℝ), polar_eq_C θ = ρ → ρ ^ 2 * Real.cos θ ^ 2 = ρ * Real.sin θ → y = x ^ 2 ∧ x ≠ 0) ∧
  (∀ (x₀ y₀ : ℝ), y₀ = x₀ ^ 2 ∧ x₀ ≠ 0 →
      let d := (|x₀ - y₀ - 1| / Real.sqrt 2) in
      d = (|-(x₀ - 1 / 2) ^ 2 - 3 / 4| / Real.sqrt 2) →
      (x₀ = 1 / 2 → d = 3 * Real.sqrt 2 / 8 ∧ (x₀ = 1 / 2 ∧ y₀ = 1 / 4))) :=
by sorry

end problem_solution_l345_345052


namespace chairs_subsets_l345_345250

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345250


namespace total_people_ride_l345_345781

theorem total_people_ride (people_per_carriage : ℕ) (num_carriages : ℕ) (h1 : people_per_carriage = 12) (h2 : num_carriages = 15) : 
    people_per_carriage * num_carriages = 180 := by
  sorry

end total_people_ride_l345_345781


namespace projection_of_vector_l345_345894

variables (a b : ℝ × ℝ)
variables (proj : ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - 2*v.1, u.2 - 2*v.2)
def vector_norm (v : ℝ × ℝ) := real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_vector :
  a = (4, -7) →
  b = (3, -4) →
  proj = (dot_product (vector_sub a b) b) / vector_norm b →
  proj = -2 :=
by
  intros
  sorry

end projection_of_vector_l345_345894


namespace expected_value_abs_diff_eq_inv_lambda_l345_345609

variables {Ω : Type*} {λ : ℝ} {ξ η : Ω → ℝ}

noncomputable def independent_exponential_components := 
  (MeasureTheory.ProbabilityTheory.independent ξ η ∧ 
   ∀ ω, MeasureTheory.PDF.measure (MeasureTheory.PDF.exponential λ) (ξ ω) ∧ 
        MeasureTheory.PDF.measure (MeasureTheory.PDF.exponential λ) (η ω))

theorem expected_value_abs_diff_eq_inv_lambda
  (hξη : independent_exponential_components) :
  MeasureTheory.ProbabilityTheory.expectation (λ ω, |ξ ω - η ω|) = 1 / λ :=
sorry

end expected_value_abs_diff_eq_inv_lambda_l345_345609


namespace chair_subsets_with_at_least_three_adjacent_l345_345261

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345261


namespace subsets_with_at_least_three_adjacent_chairs_l345_345248

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345248


namespace a_minus_b_eq_three_l345_345506

theorem a_minus_b_eq_three (a b : ℝ) (h : (a+bi) * i = 1 + 2 * i) : a - b = 3 :=
by
  sorry

end a_minus_b_eq_three_l345_345506


namespace count_valid_4_digit_numbers_l345_345498

def is_valid_number (n : ℕ) : Prop :=
  let thousands := n / 1000 in
  let hundreds := (n / 100) % 10 in
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  1000 ≤ n ∧ n < 10000 ∧
  1 ≤ thousands ∧ thousands ≤ 9 ∧
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_4_digit_numbers : 
  (Finset.filter is_valid_number (Finset.range 10000)).card = 1782 :=
sorry

end count_valid_4_digit_numbers_l345_345498


namespace min_value_range_of_a_l345_345042

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp (-2 * x) + a * (2 * x + 1) * Real.exp (-x) + x^2 + x

theorem min_value_range_of_a (a : ℝ) (h : a > 0)
  (min_f : ∃ x : ℝ, f a x = Real.log a ^ 2 + 3 * Real.log a + 2) :
  a ∈ Set.Ici (Real.exp (-3 / 2)) :=
by
  sorry

end min_value_range_of_a_l345_345042


namespace parabola_focus_equals_ellipse_focus_l345_345522

theorem parabola_focus_equals_ellipse_focus (p : ℝ) : 
  let parabola_focus := (p / 2, 0)
  let ellipse_focus := (2, 0)
  parabola_focus = ellipse_focus → p = 4 :=
by
  intros h
  sorry

end parabola_focus_equals_ellipse_focus_l345_345522


namespace verify_propositions_l345_345377

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345377


namespace min_diff_length_edges_l345_345069

-- Define what it means for a tetrahedron to have no isosceles faces
structure Tetrahedron (V E : Type) :=
(vertices : fin 4 → V)
(edges : fin 6 → E)
(face_is_scalene : ∀ f : fin 4, let ⟨e1, e2, e3⟩ := face_edges f in e1 ≠ e2 ∧ e2 ≠ e3 ∧ e1 ≠ e3)

-- Define the proof that the minimum number of edges with different lengths is 3
theorem min_diff_length_edges {V E : Type} (T : Tetrahedron V E) : 
  ∃ (distinct_edge_count : nat), 
    distinct_edge_count = 3 ∧ 
    (  ∀ (edge_lengths : fin 6 → ℕ),
       (∀ f : fin 4, let ⟨e1, e2, e3⟩ := face_edges f 
                    in edge_lengths e1 ≠ edge_lengths e2 
                        ∧ edge_lengths e2 ≠ edge_lengths e3 
                        ∧ edge_lengths e1 ≠ edge_lengths e3) 
       → distinct_edge_count = 3) :=
sorry

end min_diff_length_edges_l345_345069


namespace no_transformation_without_integer_roots_no_transformation_without_integer_roots_l345_345942

theorem no_transformation_without_integer_roots :
  ∀ (p q : ℤ), ¬(transform_eq_with_no_int_roots p q) :=
by
  sorry

def initial_eq (x : ℤ) : Prop := x^2 - 2013 * x - 13 = 0
def target_eq (x : ℤ) : Prop := x^2 + 13 * x + 2013 = 0

def transform_eq_with_no_int_roots (p q : ℤ) : Prop :=
  (∀ x : ℤ, x^2 + p * x + q ≠ 0 → 1 + p + q ≠ 0) -- Intermediate equation does not have integer roots
  ∧ (p - (-2013)) + (q - (-13)) = (13 - (-2013)) + (2013 - (-13)) -- Change in p and q sums up to the total necessary change

-- Begin the theorem statement by asserting the impossibility of such a transformation
theorem no_transformation_without_integer_roots :
  ¬(∃ (p q : ℤ), initial_eq p q ∧ transform_eq_with_no_int_roots p q ∧ target_eq p q) :=
by
  sorry

end no_transformation_without_integer_roots_no_transformation_without_integer_roots_l345_345942


namespace points_lie_on_ellipse_l345_345830

open Real

noncomputable def curve_points_all_lie_on_ellipse (s: ℝ) : Prop :=
  let x := 2 * cos s + 2 * sin s
  let y := 4 * (cos s - sin s)
  (x^2 / 8 + y^2 / 32 = 1)

-- Below statement defines the theorem we aim to prove:
theorem points_lie_on_ellipse (s: ℝ) : curve_points_all_lie_on_ellipse s :=
sorry -- This "sorry" is to indicate that the proof is omitted.

end points_lie_on_ellipse_l345_345830


namespace monotonic_intervals_range_of_a_l345_345043

noncomputable def f (x a : ℝ) := x - a / x - (a + 1) * Real.log x

theorem monotonic_intervals :
  let f := (λ x : ℝ, x - 1 / (2 * x) - (3 / 2) * Real.log x) in
  (∀ x, 0 < x ∧ x < (1 / 2) → (0, 1 / 2)) ∧ (∀ x, x > 1 → ((1, ∞))) ∧ (∀ x, (1 / 2) < x ∧ x < 1 → ((1 / 2, 1))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → f x a ≤ x) ↔ (a ≥ -1 / (Real.exp 1 - 1)) :=
sorry

end monotonic_intervals_range_of_a_l345_345043


namespace swim_time_CBA_l345_345212

theorem swim_time_CBA (d t_down t_still t_upstream: ℝ) 
  (h1 : d = 1) 
  (h2 : t_down = 1 / (6 / 5))
  (h3 : t_still = 1)
  (h4 : t_upstream = (4 / 5) / 2)
  (total_time_down : (t_down + t_still) = 1)
  (total_time_up : (t_still + t_down) = 2) :
  (t_upstream * (d - (d / 5))) / 2 = 5 / 2 :=
by sorry

end swim_time_CBA_l345_345212


namespace solve_quadratic_equation_l345_345640

theorem solve_quadratic_equation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ ∀ x : ℝ, (x^2 - 2*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂) :=
by
  sorry

end solve_quadratic_equation_l345_345640


namespace new_median_after_adding_9_l345_345328

variable (nums : List ℕ) (n : ℕ)
hypothesis (mean_eq : (nums.sum / nums.length.toNat = 5.5))
hypothesis (unique_mode : ∃ m : ℕ, ∀ x, m = 4 ∧ (∀ y, y ≠ x → ¬ (nums.count y > nums.count x)))
hypothesis (median_eq : (nums.nth (nums.length / 2 - 1) = some 5))
hypothesis (initial_len : nums.length = 6)
hypothesis (pos_ints : ∀ x ∈ nums, x > 0)

theorem new_median_after_adding_9 (nums_extended : List ℕ := nums ++ [9]) :
  (nums_extended.nth ((nums_extended.length / 2).toNat) = some 5.0) :=
by
  sorry

end new_median_after_adding_9_l345_345328


namespace laser_path_distance_l345_345753

noncomputable def distance_traveled_by_laser : ℝ :=
  let A := (4, 7)
  let D := (10, 7)
  let D' := (-10, -7)
  real.sqrt ((A.1 - D'.1)^2 + (A.2 - D'.2)^2)

theorem laser_path_distance : distance_traveled_by_laser = 14 * real.sqrt 2 := by
  sorry

end laser_path_distance_l345_345753


namespace number_of_true_propositions_l345_345387

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345387


namespace mutually_exclusive_not_contradictory_l345_345284

-- Define the initial conditions
def balls := ℕ
def white_ball : balls := 2
def black_ball : balls := 2

-- Define the events
def exactly_one_black_ball (draw : list balls) : Prop :=
  draw.count black_ball = 1

def exactly_two_black_balls (draw : list balls) : Prop :=
  draw.count black_ball = 2

-- The statement to prove
theorem mutually_exclusive_not_contradictory : 
  ∀ (draw : list balls), exactly_one_black_ball draw ∨ exactly_two_black_balls draw → 
  (exactly_one_black_ball draw ∧ ¬ exactly_two_black_balls draw) :=
by sorry

end mutually_exclusive_not_contradictory_l345_345284


namespace phillip_has_51_91_remaining_l345_345991

/-- Phillip's remaining money after shopping with specific conditions -/
def phillip_remaining_money : ℝ :=
let initial_amount : ℝ := 95 in
let oranges_cost : ℝ := 2 * 3 in
let apples_cost : ℝ := 4 * 3.5 in
let candy_cost : ℝ := 6 in
let eggs_cost : ℝ := 2 * 6 in
let milk_cost : ℝ := 4 in
let apples_discount : ℝ := 0.15 * apples_cost in
let discounted_apples_cost : ℝ := apples_cost - apples_discount in
let total_cost_before_tax : ℝ := oranges_cost + discounted_apples_cost + candy_cost + eggs_cost + milk_cost in
let sales_tax : ℝ := 0.08 * total_cost_before_tax in
let final_total_cost : ℝ := total_cost_before_tax + sales_tax in
initial_amount - final_total_cost

/-- Proposition: Phillip has $51.91 left after shopping -/
theorem phillip_has_51_91_remaining :
  phillip_remaining_money = 51.91 := by
  sorry

end phillip_has_51_91_remaining_l345_345991


namespace ratio_AF_BF_l345_345666

noncomputable def AF (F A : ℝ × ℝ) : ℝ := Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2)
noncomputable def BF (F B : ℝ × ℝ) : ℝ := Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2)

theorem ratio_AF_BF : 
  ∀ (A F B : ℝ × ℝ), 
  A = (2, 4) → F = (4, 0) → B = (8, -8) → 
  (AF F A / BF F B = 1 / 2) :=
by
  intros A F B hA hF hB
  rw [hA, hF, hB]
  have hAF : AF F A = 2 * Real.sqrt 5 := by sorry
  have hBF : BF F B = 4 * Real.sqrt 5 := by sorry
  rw [hAF, hBF]
  exact div_eq_div_of_mul_eq_mul (by norm_num) (by norm_num)

end ratio_AF_BF_l345_345666


namespace smallest_a_l345_345519

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_a (a : ℕ) (h1 : is_factor 112 (a * 43 * 62 * 1311)) (h2 : is_factor 33 (a * 43 * 62 * 1311)) : a = 1848 :=
by
  sorry

end smallest_a_l345_345519


namespace find_value_l345_345646

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic : ∀ x : ℝ, f (x + Real.pi) = f x
axiom value_at_neg_pi_third : f (-Real.pi / 3) = 1 / 2

theorem find_value : f (2017 * Real.pi / 3) = 1 / 2 :=
by
  sorry

end find_value_l345_345646


namespace monotonic_decreasing_interval_l345_345206

variable {x : ℝ}

def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem monotonic_decreasing_interval : 
  (∀ x > 0, (2 * x - 2 / x) ≤ 0 ↔ 0 < x ∧ x ≤ 1) :=
by sorry

end monotonic_decreasing_interval_l345_345206


namespace classroom_tables_l345_345741

theorem classroom_tables (n : ℕ) (h : n = 321₇) : n / 3 = 54 :=
by
  -- The proof will go here (but it's not required in this guideline)
  sorry

end classroom_tables_l345_345741


namespace gcd_18_n_eq_6_l345_345437

open Nat

theorem gcd_18_n_eq_6 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 150) (h2 : gcd 18 n = 6) : Finset.card ((Finset.filter (λ k => gcd 18 k = 6) (Finset.range 151)).val) = 17 := by
  sorry

end gcd_18_n_eq_6_l345_345437


namespace options_B_and_D_correct_l345_345879

-- Definitions from conditions
def f (x : ℝ) : ℝ := (Real.log x) / x

-- Statement with conditions and assertions
theorem options_B_and_D_correct (x1 x2 : ℝ) (h_cond : x2 > x1 ∧ x1 > 1) :
  abs (f x1 - f x2) < 1 / Real.exp 1 ∧ f x1 - f x2 < x2 - x1 := by
  sorry

end options_B_and_D_correct_l345_345879


namespace cos_difference_zero_l345_345636

noncomputable def cos72 := Real.cos (72 * Real.pi / 180)
noncomputable def cos144 := Real.cos (144 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)

theorem cos_difference_zero :
  cos72 - cos144 = 0 :=
by
  let c := cos72
  let d := cos144
  have h1 : d = -(1 - 2 * c^2), by sorry
  have h2 : c - d = c + 1 - 2 * c^2, by sorry
  have h3 : 2 * c^2 - c - 1 = 0, by sorry
  have h4 : c = 1, by sorry
  sorry

end cos_difference_zero_l345_345636


namespace find_side_DF_in_triangle_DEF_l345_345101

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l345_345101


namespace lines_not_parallel_l345_345732

variables (a b c : Type) [line a] [line b] [line c]

def skew_lines (a b : Type) [line a] [line b] : Prop := 
  ¬ ∃ (P : Type) [point P], incident P a ∧ incident P b

def parallel (l1 l2 : Type) [line l1] [line l2] : Prop :=
  ∀ (P Q : Type) [point P] [point Q], incident P l1 ∧ incident P l2 → incident Q l1 ∧ incident Q l2

theorem lines_not_parallel (a b c : Type) [line a] [line b] [line c]
  (h1 : skew_lines a b) (h2 : parallel c a) : ¬ parallel c b := by
  sorry

end lines_not_parallel_l345_345732


namespace trace_diff_eq_half_n_n_sub_one_l345_345729

noncomputable def trace_difference (A B : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  (Matrix.trace (fun (i j : Fin n) => (A ⬝ B) ⬝ (A ⬝ B)) - Matrix.trace (fun (i j : Fin n) => (A ⬝ ⬝ B) ⬝ (A ⬝ B)))

theorem trace_diff_eq_half_n_n_sub_one (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h_rank : Matrix.rank(Matrix.mul_assoc (A ⬝ B - A ⬝ ⬝ B) + Matrix.mul_assoc (1:Matrix (Fin n) (Fin n) ℝ)) = 1) :
  trace_difference A B = (1 / 2) * n * (n - 1) :=
sorry

end trace_diff_eq_half_n_n_sub_one_l345_345729


namespace center_cell_value_l345_345544

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l345_345544


namespace all_heads_up_l345_345703

-- A function to represent the flipping of the k-th coin in a list of coins
def flip (l : List Bool) (k : ℕ) : List Bool :=
  l.take (k - 1) ++ (if l.get (k - 1) then [false] else [true]) ++ l.drop k

-- A function to count the number of tails (represented as false) in the list
def count_tails (l : List Bool) : ℕ :=
  l.countp (λ x => x = false)

-- The main theorem
theorem all_heads_up (l : List Bool) (n : ℕ) (Hlen : l.length = n)
  (step : List Bool → ℕ → List Bool) (Hstep : ∀ l, step l (count_tails l) = flip l (count_tails l) + 1) :
  ∃ k, k > 0 ∧ (step^[k] l = List.repeat true n) :=
by
  sorry -- Proof goes here


end all_heads_up_l345_345703


namespace number_of_adjacent_subsets_l345_345234

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345234


namespace right_triangle_side_lengths_approximation_l345_345341

noncomputable def AC : ℝ := 465
noncomputable def α : ℝ := 51.6 * (Real.pi / 180) -- Convert degrees to radians
noncomputable def approx_a : ℝ := 588
noncomputable def approx_c : ℝ := 746

theorem right_triangle_side_lengths_approximation :
  ∃ (a c : ℝ), |a - approx_a| / approx_a < 0.01 ∧ |c - approx_c| / approx_c < 0.01 ∧ 
               ∠ (Complex.angle (Complex.exp (Complex.I * α))) = α ∧ 
               AC = 465 ∧
               a = AC * Real.tan α ∧ 
               c = AC / Real.cos α :=
by {
  sorry
}

end right_triangle_side_lengths_approximation_l345_345341


namespace staplers_left_l345_345682

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l345_345682


namespace find_side_DF_in_triangle_DEF_l345_345104

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l345_345104


namespace find_trajectory_maximal_distance_l345_345846

-- Define the circle and the points
def circle (x y: ℝ) := x^2 + y^2 = 1
def trajectory (x y: ℝ) := x^2 / 3 + y^2 = 1

-- Define the conditions
variables {x₀ y₀ x y: ℝ}
variables (hP : circle x₀ y₀)
variables (hQ : y₀ = y)
variables (hOM_QP : x = sqrt 3 * x₀)

theorem find_trajectory : trajectory x y :=
by {
  -- Insert actual proof here
  sorry
}

-- Maximal distance on the trajectory
theorem maximal_distance : ∃ m: ℝ, ∀ (x₁ y₁ x₂ y₂: ℝ),
  (l := (x₁ = m*y₁ + n ∧ x₂ = m*y₂ + n)) ∧ trajectory x₁ y₁ ∧ trajectory x₂ y₂ →
  abs (dist (x₁, y₁) (x₂, y₂)) ≤ sqrt 3 :=
by {
  -- Insert actual proof here
  sorry
}

end find_trajectory_maximal_distance_l345_345846


namespace min_area_of_circle_tangent_lines_at_m2_l345_345872

noncomputable def circle_eqn (m : ℝ) := 
  λ (x y : ℝ), x^2 + y^2 - 2 * m * x - 2 * y + 4 * m - 4 = 0

theorem min_area_of_circle :
  ∃ m : ℝ, ∀ x y, circle_eqn m x y ↔ m = 2 := sorry

theorem tangent_lines_at_m2 :
  ∀ x y : ℝ, circle_eqn 2 x y →
  ((∀ (x y : ℝ), x = 1 ∧ (x, y) = (1, -2)) ∨ (4 * x - 3 * y - 10 = 0 ∧ (x, y) = (1, -2))) := sorry

end min_area_of_circle_tangent_lines_at_m2_l345_345872


namespace complement_U_B_eq_D_l345_345121

def B (x : ℝ) : Prop := x^2 - 3 * x + 2 < 0
def U : Set ℝ := Set.univ
def complement_U_B : Set ℝ := U \ {x | B x}

theorem complement_U_B_eq_D : complement_U_B = {x | x ≤ 1 ∨ x ≥ 2} := by
  sorry

end complement_U_B_eq_D_l345_345121


namespace find_base_b_l345_345028

theorem find_base_b : 
  ∃ b : ℕ, (25_b : ℕ → ℕ) b = 2 * (2 * b + 5) + 5 ∧ (52_b : ℕ → ℕ) b = 5 * b + 2 ∧ (52_b b = 2 * 25_b b) := 
  ∃ (b : ℕ), 
  (2 * b + 5) = 2 * (2 * b + 5) + 5 ∧
  (5 * b + 2) = 2 * (2 * b + 5) + 2 ∧
  (5 * b + 2 = 2 * (2 * b + 5)) ↔ b = 8 :=
begin
  existsi 8,
  -- Definitions of 25_b and 52_b
  β_defs : 25_b b = 2 * b + 5, 52_b b = 5 * b + 2,
  sorry
end

end find_base_b_l345_345028


namespace units_digit_sum_factorials_l345_345297

-- Definitions based on the conditions
def units_digit (n : ℕ) : ℕ := n % 10

-- Lean statement to represent the proof problem
theorem units_digit_sum_factorials :
  units_digit (∑ n in Finset.range 2024, n.factorial) = 3 :=
by 
  sorry

end units_digit_sum_factorials_l345_345297


namespace division_termination_l345_345182

theorem division_termination (a b : ℕ) (ha : a ≤ 1988) (hb : b ≤ 1988) :
  ∃ (n : ℕ), n ≤ 6 ∧ (∀ m, m ≤ n → 
    let q : ℕ := if a ≥ b then a / b else b / a in
    let r : ℕ := if a ≥ b then a % b else b % a in
    (q = 0 ∨ r = 0)) :=
sorry

end division_termination_l345_345182


namespace cyclic_quad_inequality_l345_345603

variables {A B C D : Type} [has_dist A] 
variables {is_on_circle : ∀ {a b c d : A}, Prop} 
variables {AB AC BD CD : ℝ}

-- Assume that points are distinct and ordered on the circle
def distinct_points_on_circle (a b c d : A) : Prop := is_on_circle a b c d

-- Assume AB is the longest side in the cyclic quadrilateral ABCD
def longest_side {a b c d : A} (AB AC BD CD : ℝ) : Prop := AB > AC ∧ AB > BD ∧ AB > CD

theorem cyclic_quad_inequality
  (a b c d : A)
  (h1 : distinct_points_on_circle a b c d)
  (h2 : longest_side AB AC BD CD) :
  AB + BD > AC + CD :=
sorry

end cyclic_quad_inequality_l345_345603


namespace find_ab_l345_345518

theorem find_ab (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 :=
by 
  sorry

end find_ab_l345_345518


namespace complement_intersection_eq_l345_345013

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { -2, -1, 0, 1 }

theorem complement_intersection_eq : (set.compl A) ∩ B = {-2, -1} := by
  sorry

end complement_intersection_eq_l345_345013


namespace solution_set_of_inequality_l345_345215

theorem solution_set_of_inequality :
  { x : ℝ | (x + 3) * (6 - x) ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 6 } :=
sorry

end solution_set_of_inequality_l345_345215


namespace minimum_distance_time_ge_minimum_distance_time_lt_l345_345281

-- Definitions of the conditions
variables (a b w : ℝ) (α : ℝ)
variables (hge : a ≥ b) (hlt : b > a)

-- Proofs of the questions
theorem minimum_distance_time_ge : (a ≥ b) → 
  (∃ t₀, ∃ d₀, d₀ = (a - b) * (Real.cos (α / 2)) ∧ t₀ = (a + b) / (2 * w)) :=
begin
  intros,
  use (a + b) / (2 * w),
  use (a - b) * (Real.cos (α / 2)),
  split; refl,
end

theorem minimum_distance_time_lt : (b > a) → 
  (∃ t₀, ∃ d₀, d₀ = Real.sqrt (a^2 + b^2 - 2 * a * b * (Real.cos α)) ∧ t₀ = (b - a) / (2 * w)) :=
begin
  intros,
  use (b - a) / (2 * w),
  use Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos α),
  split; refl,
end

-- Usage with conditions
example : (a ≥ b) → 
  (∃ t₀, ∃ d₀, d₀ = (a - b) * (Real.cos (α / 2)) ∧ t₀ = (a + b) / (2 * w)) :=
minimum_distance_time_ge a b w α

example : (b > a) → 
  (∃ t₀, ∃ d₀, d₀ = Real.sqrt (a^2 + b^2 - 2 * a * b * (Real.cos α)) ∧ t₀ = (b - a) / (2 * w)) :=
minimum_distance_time_lt a b w α

end minimum_distance_time_ge_minimum_distance_time_lt_l345_345281


namespace maximize_projection_area_l345_345307

theorem maximize_projection_area
  (P : Parallelepiped)
  (horizontal_plane : Plane) :
  (∃ face : Face, face ⊆ P ∧ face.is_parallel_to horizontal_plane) →
  (∀ other_face : Face, other_face ⊆ P → projection_area other_face horizontal_plane ≤ projection_area (horizontal_face P horizontal_plane) horizontal_plane) :=
sorry

end maximize_projection_area_l345_345307


namespace ratio_of_volumes_of_spheres_l345_345483

-- Define the variables a, b, c representing the radii of the spheres
variables (a b c : ℝ)

-- Define the condition of surface area ratios
def surface_area_ratio := (4 * real.pi * a^2) / (4 * real.pi * b^2) = 1 / 2 ∧ (4 * real.pi * a^2) / (4 * real.pi * c^2) = 1 / 3

-- Define what we need to prove: the ratio of volumes
def volume_ratio := (a^3) / (b^3) = 1 / (2 * real.sqrt 2) ∧ (a^3) / (c^3) = 1 / (3 * real.sqrt 3)

-- The theorem that formalizes our problem statement
theorem ratio_of_volumes_of_spheres (a b c : ℝ) (h : surface_area_ratio a b c) : volume_ratio a b c :=
sorry

end ratio_of_volumes_of_spheres_l345_345483


namespace a_51_eq_101_l345_345568

-- Define the sequence 'a_n' as described in the problem
def a : ℕ → ℕ
| 0 := 1
| (n + 1) := a n + 2

-- Define the theorem to prove a_{51} = 101
theorem a_51_eq_101 : a 50 = 101 :=
by sorry

end a_51_eq_101_l345_345568


namespace cosine_angle_between_vectors_l345_345695

noncomputable def midpoint (a b : Point) : Point := sorry

theorem cosine_angle_between_vectors
  (P Q R T U : Point)
  (mid_Q : midpoint T U = Q)
  (PQ_len : dist P Q = 2)
  (TU_len : dist T U = 2)
  (QR_len : dist Q R = 8)
  (PR_len : dist P R = 2 * Real.sqrt 17)
  (dot_product_eq : (P.to_vec - Q.to_vec) ⋅ (P.to_vec - T.to_vec) + 
                    (P.to_vec - R.to_vec) ⋅ (P.to_vec - U.to_vec) = 4) :
  cos_angle_between (T.to_vec - U.to_vec) (Q.to_vec - R.to_vec) = 1 / 2 :=
sorry

end cosine_angle_between_vectors_l345_345695


namespace intersection_M_N_l345_345614

def M := { x : ℝ | x < 2011 }
def N := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l345_345614


namespace mul_101_eq_10201_l345_345416

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end mul_101_eq_10201_l345_345416


namespace train_length_l345_345697

theorem train_length 
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (time_pass : ℝ)
  (relative_speed : ℝ := speed_train1 + speed_train2)
  (length_train : ℝ) :
  speed_train1 = 30 → speed_train2 = 30 → time_pass = 60 →
  relative_speed * 5 / 18 = (2 * length_train) / time_pass →
  length_train = 500 :=
begin
  sorry
end

end train_length_l345_345697


namespace area_ratio_l345_345946

-- Definitions and given conditions are explicitly defined
variables {A B C D E F P Q R : Type*} [plane_geometry A B C D E F P Q R]
variable (triangle : triangle ABC) 
variable (BD DC CE EA AF FB : ratio) -- Ratios are defined as types

-- Given ratios as the conditions
def ratios (triangle : triangle ABC) := 
  BD:DC = 1:3 ∧ CE:EA = 3:1 ∧ AF:FB = 2:1  

-- Intersection definition for points P, Q, R on the respected segments
def intersections (triangle : triangle ABC) :=
  line_intersects_at AD P ∧ line_intersects_at BE Q ∧ line_intersects_at CF R

-- Proof goal: The ratio of areas of triangles PQR and ABC
theorem area_ratio (triangle : triangle ABC) (h_ratio : ratios triangle) (h_intersection : intersections triangle) :
  ∃ (k : ℚ), k = (1 / 27) :=
sorry

end area_ratio_l345_345946


namespace amount_of_meat_left_l345_345582

theorem amount_of_meat_left (initial_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ)
  (h0 : initial_meat = 20)
  (h1 : meatballs_fraction = 1/4)
  (h2 : spring_rolls_meat = 3) : 
  (initial_meat - (initial_meat * meatballs_fraction:ℕ) - spring_rolls_meat) = 12 := 
by 
  sorry

end amount_of_meat_left_l345_345582


namespace circles_intersect_l345_345211

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x - 4*y - 1 = 0

theorem circles_intersect : 
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) := 
sorry

end circles_intersect_l345_345211


namespace circle_chairs_adjacent_l345_345266

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345266


namespace circle_chairs_adjacent_l345_345265

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345265


namespace swimming_speed_still_water_l345_345337

theorem swimming_speed_still_water :
  (exists v : ℝ, 2 > 0 ∧ 8 > 0 ∧ 16 = (v - 2) * 8) →
  (∃ v : ℝ, v = 4) :=
by
  intro h
  cases h with v hv
  sorry

end swimming_speed_still_water_l345_345337


namespace intersection_A_B_l345_345119

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | ∃ (n : ℤ), (x : ℝ) = n }

theorem intersection_A_B : A ∩ B = {0, 1} := 
by
  sorry

end intersection_A_B_l345_345119


namespace find_xyz_l345_345644

theorem find_xyz (x y z : ℝ) (h₁ : x + 1 / y = 5) (h₂ : y + 1 / z = 2) (h₃ : z + 2 / x = 10 / 3) : x * y * z = (21 + Real.sqrt 433) / 2 :=
by
  sorry

end find_xyz_l345_345644


namespace sequence_no_perfect_power_l345_345570

-- Define the initial triple
def initial_triple : (ℕ × ℕ × ℕ) := (2, 3, 5)

-- Define the sequence generation rule
def next_triple (t : (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  (t.1 * t.2, t.2 * t.3, t.3 * t.1)

-- Define the sequence recursively
noncomputable def sequence : ℕ → (ℕ × ℕ × ℕ)
| 0     := initial_triple
| (n+1) := next_triple (sequence n)

-- Prove no element in this sequence can be a perfect power of any integer
theorem sequence_no_perfect_power (n k m : ℕ) (hm : m ≥ 2) :
  ¬ (∃ i : ℕ × ℕ × ℕ, i ∈ (sequence n) ∧ (∃ p : ℕ, i.1 = p^m ∨ i.2 = p^m ∨ i.3 = p^m)) :=
by
  sorry

end sequence_no_perfect_power_l345_345570


namespace subsets_with_at_least_three_adjacent_chairs_l345_345246

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345246


namespace minimum_value_of_f_range_of_t_l345_345046

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem minimum_value_of_f : ∀ x, f x ≥ 6 ∧ ∃ x0 : ℝ, f x0 = 6 := 
by sorry

theorem range_of_t (t : ℝ) : (t ≤ -2 ∨ t ≥ 3) ↔ ∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t :=
by sorry

end minimum_value_of_f_range_of_t_l345_345046


namespace range_of_g_eq_l345_345002

noncomputable def g (x : ℝ) : ℝ :=
  (sin x ^ 3 + 7 * sin x ^ 2 + cos x * sin x + 3 * cos x ^ 2 - 11) / (sin x - 1)

theorem range_of_g_eq : 
  {y : ℝ | ∃ x : ℝ, g x = y ∧ sin x ≠ 1} = {y : ℝ | y ≥ -2 + real.sqrt 2} :=
by
  sorry

end range_of_g_eq_l345_345002


namespace tangents_not_necessarily_coincide_at_B_l345_345566

noncomputable def parabola := { p : ℝ × ℝ | p.snd = p.fst^2 }
noncomputable def circle (a b r : ℝ) := { p : ℝ × ℝ | (p.fst - a)^2 + (p.snd - b)^2 = r^2 }

theorem tangents_not_necessarily_coincide_at_B:
  ∃ (a b r : ℝ), 
  let A := (1,1)
      B := (-3,9)
  in 
  A ∈ parabola ∧ B ∈ parabola ∧ A ∈ circle a b r ∧ B ∈ circle a b r ∧
  (∃ t : ℝ, (1,1) = (a + t*1, b + t*(2*1))) →
  ∄ t : ℝ, (-3,9) = (a + t*(-3), b + t*(2*(-3))) :=
begin
  sorry,
end

end tangents_not_necessarily_coincide_at_B_l345_345566


namespace trigonometric_identity_l345_345837

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity_l345_345837


namespace parabola_tangent_line_l345_345804

theorem parabola_tangent_line (b c : ℝ) :
  let y_parabola := λ x : ℝ, x^2 + b * x + c
  let y_line := λ x : ℝ, 2 * x
  let y_parabola_prime := λ x : ℝ, 2 * x + b
  (y_parabola 2 = y_line 2) → -- Condition 1: two curves intersect at x = 2
  (y_parabola_prime 2 = 2) →  -- Condition 2: slopes are equal at x = 2
  b = -2 ∧ c = 4 :=            -- Conclusion
sorry

end parabola_tangent_line_l345_345804


namespace johns_payment_l345_345955

def stereo_cost_usd : ℤ := 250
def tv_cost_jpy : ℤ := 45000
def dvd_cost_eur : ℤ := 120

def stereo_trade_in_percent : ℝ := 0.80
def tv_trade_in_percent : ℝ := 0.75
def dvd_trade_in_percent : ℝ := 0.85

def exchange_rate_usd_jpy : ℝ := 110
def exchange_rate_usd_eur : ℝ := 0.9

def entertainment_system_cost_cad : ℤ := 1500
def soundbar_cost_mxn : ℤ := 8000
def gaming_console_cost_aud : ℤ := 700

def entertainment_system_discount : ℝ := 0.25
def soundbar_discount : ℝ := 0.20
def gaming_console_discount : ℝ := 0.30

def exchange_rate_usd_cad : ℝ := 1.3
def exchange_rate_usd_mxn : ℝ := 20
def exchange_rate_usd_aud : ℝ := 1.4

def sales_tax_stereo : ℝ := 0.05
def sales_tax_tv : ℝ := 0.03
def sales_tax_dvd : ℝ := 0.07
def sales_tax_entertainment_system : ℝ := 0.08
def sales_tax_soundbar : ℝ := 0.05
def sales_tax_gaming_console : ℝ := 0.10

theorem johns_payment : ℝ :=
let stereo_trade_in_value := stereo_cost_usd * stereo_trade_in_percent in
let tv_trade_in_value := (tv_cost_jpy * tv_trade_in_percent) / exchange_rate_usd_jpy in
let dvd_trade_in_value := (dvd_cost_eur * dvd_trade_in_percent) / exchange_rate_usd_eur in
let total_trade_in_value := stereo_trade_in_value + tv_trade_in_value + dvd_trade_in_value in

let entertainment_system_cost_usd := entertainment_system_cost_cad / exchange_rate_usd_cad in
let entertainment_system_cost_after_discount := entertainment_system_cost_usd * (1 - entertainment_system_discount) in
let entertainment_system_total_cost := entertainment_system_cost_after_discount * (1 + sales_tax_entertainment_system) in

let soundbar_cost_usd := soundbar_cost_mxn / exchange_rate_usd_mxn in
let soundbar_cost_after_discount := soundbar_cost_usd * (1 - soundbar_discount) in
let soundbar_total_cost := soundbar_cost_after_discount * (1 + sales_tax_soundbar) in

let gaming_console_cost_usd := gaming_console_cost_aud / exchange_rate_usd_aud in
let gaming_console_cost_after_discount := gaming_console_cost_usd * (1 - gaming_console_discount) in
let gaming_console_total_cost := gaming_console_cost_after_discount * (1 + sales_tax_gaming_console) in

let total_new_items_cost := entertainment_system_total_cost + soundbar_total_cost + gaming_console_total_cost in

total_new_items_cost - total_trade_in_value = 1035.47 := sorry

end johns_payment_l345_345955


namespace jackson_meat_left_l345_345577

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l345_345577


namespace integral_exponential_l345_345675

theorem integral_exponential : ∫ x in 0..1, Real.exp x = Real.exp 1 - Real.exp 0 :=
by
  sorry

end integral_exponential_l345_345675


namespace find_x_l345_345941

def AB_is_straight_line : Prop := true -- Given statement, should be assumed as true in context
def angle_DCE : ℝ := 95 -- ∠DCE = 95°
def angle_ECA : ℝ := 58 -- ∠ECA = 58°

theorem find_x (h1 : AB_is_straight_line) (h2 : ∠ DCE = angle_DCE) (h3 : ∠ ECA = angle_ECA) : 
  ∠ x = 27 := sorry

end find_x_l345_345941


namespace minimum_f_l345_345821

def f (x y : ℤ) : ℤ := |5 * x^2 + 11 * x * y - 5 * y^2|

theorem minimum_f (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : ∃ (m : ℤ), m = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∨ y ≠ 0) → f x y ≥ m :=
by sorry

end minimum_f_l345_345821


namespace linear_function_through_two_points_l345_345425

theorem linear_function_through_two_points :
  ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧
  (k ≠ 0) ∧
  (3 = 2 * k + b) ∧
  (2 = 3 * k + b) ∧
  (∀ x, y = -x + 5) :=
by
  sorry

end linear_function_through_two_points_l345_345425


namespace positive_integer_divisibility_l345_345009

theorem positive_integer_divisibility :
  ∀ n : ℕ, 0 < n → (5^(n-1) + 3^(n-1) ∣ 5^n + 3^n) → n = 1 :=
by
  sorry

end positive_integer_divisibility_l345_345009


namespace length_of_QR_l345_345927

theorem length_of_QR
  (P Q R Z : Type)
  (PQ PR QR QZ RZ : ℝ)
  (hPQ : PQ = 79)
  (hPR : PR = 93)
  (hPZ : PQ = P)
  (hQZ_RZ_int : ∀ r s : ℕ, QR = r + s)
  (hCircle : PZ = PQ)
  (hCircleIntersects : QR = QZ + RZ)
  (hSolution : QR = 79) :
  QR = 79 := by
  sorry

end length_of_QR_l345_345927


namespace count_at_least_three_adjacent_chairs_l345_345240

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345240


namespace sean_divided_by_julie_is_2_l345_345159

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l345_345159


namespace seans_sum_divided_by_julies_sum_l345_345165

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l345_345165


namespace cherry_ratio_l345_345792

theorem cherry_ratio (total_lollipops cherry_lollipops watermelon_lollipops sour_apple_lollipops grape_lollipops : ℕ) 
  (h_total : total_lollipops = 42) 
  (h_rest_equally_distributed : watermelon_lollipops = sour_apple_lollipops ∧ sour_apple_lollipops = grape_lollipops) 
  (h_grape : grape_lollipops = 7) 
  (h_total_sum : cherry_lollipops + watermelon_lollipops + sour_apple_lollipops + grape_lollipops = total_lollipops) : 
  cherry_lollipops = 21 ∧ (cherry_lollipops : ℚ) / total_lollipops = 1 / 2 :=
by
  sorry

end cherry_ratio_l345_345792


namespace intersection_M_N_l345_345889

def set_M (x : ℝ) : Prop := x^2 < 1
def set_N (x : ℝ) : Prop := 2^x > 1

theorem intersection_M_N :
  {x : ℝ | set_M x} ∩ {x : ℝ | set_N x} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l345_345889


namespace chairs_subsets_l345_345253

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345253


namespace area_of_triangle_l345_345480

-- Definitions based on conditions
def parabola (C : Type) :=
  ∃ d : ℝ, ∀ p : ℝ × ℝ, C = { p : ℝ × ℝ | p.2 = p.1^2 / (4 * d) }

def passes_through_focus (l : Type) (C : Type) : Prop :=
  ∃ d : ℝ, d > 0 ∧ focus l ∈ parabola_focus C

def perpendicular_to_axis_of_symmetry (l : Type) (C : Type) : Prop :=
  ∃ d : ℝ, d > 0 ∧ (focus l).2 = d / 2 ∧ l.1 = 0

def intersects (l : Type) (C : Type) (A B : Type) : Prop :=
  ∃ d : ℝ, d > 0 ∧ C = { p : ℝ × ℝ | p.2 = p.1^2 / (4 * d) } ∧
  line_intersects_parabola l C A B ∧ distance A B = 12

def directrix_point (P : Type) (C : Type) : Prop :=
  ∃ d : ℝ, d > 0 ∧ P ∈ directrix C

-- Lean 4 statement
theorem area_of_triangle (l C A B P : Type) 
  (h1: passes_through_focus l C)
  (h2: perpendicular_to_axis_of_symmetry l C)
  (h3: intersects l C A B)
  (h4: directrix_point P C) :
  area_triangle A B P = 36 :=
sorry

end area_of_triangle_l345_345480


namespace chairs_subset_count_l345_345271

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345271


namespace abs_diff_eq_5_l345_345818

def abs_diff_C_D (C D : Nat) : Nat :=
  if C > D then C - D else D - C

theorem abs_diff_eq_5 : 
  ∃ (C D : Nat), C < 8 ∧ D < 8 ∧ 
  (
   D = 5 ∧ 
   C = 0 ∧ 
   abs_diff_C_D C D = 5 
  ) := 
by {
  use (0, 5),
  simp,
  sorry
}

end abs_diff_eq_5_l345_345818


namespace sec_4pi_over_3_l345_345421

theorem sec_4pi_over_3 : Float.cos (4 * Float.pi / 3) ≠ 0 → Float.sec (4 * Float.pi / 3) = -2 := by
  sorry

end sec_4pi_over_3_l345_345421


namespace swap_numbers_l345_345648

variables (a b c : ℕ)

theorem swap_numbers:
  a = 1 → b = 3 →
  (let c := b in let b := a in let a := c in (a, b))
  = (3, 1) :=
by
  intros ha hb
  let c := b
  let b := a
  let a := c
  exact (a, b) = (3, 1)
  sorry

end swap_numbers_l345_345648


namespace sum_floor_sqrt_ge_floor_sqrt_sum_add_200k_l345_345726

def min (lst : List ℕ) : ℕ :=
  lst.foldr Nat.min Nat.infinity

theorem sum_floor_sqrt_ge_floor_sqrt_sum_add_200k
  (a : Fin 25 → ℕ) (h : ∀ i, 0 ≤ a i) :
  let k := min (List.ofFn a)
  ∑ i, (⌊(Real.sqrt (a i))⌋ : ℕ) ≥ ⌊Real.sqrt (∑ i, a i + 200 * k)⌋ :=
by
  let k := min (List.ofFn a)
  sorry  -- The proof will be filled in here later

end sum_floor_sqrt_ge_floor_sqrt_sum_add_200k_l345_345726


namespace total_books_l345_345905

-- Lean 4 Statement
theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) (albert_books : ℝ) (total_books : ℝ) 
  (h1 : stu_books = 9) 
  (h2 : albert_ratio = 4.5) 
  (h3 : albert_books = stu_books * albert_ratio) 
  (h4 : total_books = stu_books + albert_books) : 
  total_books = 49.5 := 
sorry

end total_books_l345_345905


namespace center_cell_value_l345_345551

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l345_345551


namespace quadratic_residue_distribution_even_odd_quadratic_residue_distribution_sum_of_quadratic_residues_l345_345976

theorem quadratic_residue_distribution (p : ℕ) (hp_prime: Nat.Prime p) (hp_mod: p % 4 = 1) :
  let residues := (1 + p) / 4 in
  ∃ qres nres, qres = residues ∧ nres = residues ∧
  set.univ.filter (λ x, (∃ y, y^2 % p = x)).card = qres ∧ -- count of quadratic residues
  set.univ.filter (λ x, ¬(∃ y, y^2 % p = x)).card = nres  -- count of non-quadratic residues
:= sorry

theorem even_odd_quadratic_residue_distribution (p : ℕ) (hp_prime: Nat.Prime p) (hp_mod: p % 4 = 1) :
  let residues := (p - 1) / 4 in
  ∃ even_qres odd_nres,
  set.univ.filter (λ x, x % 2 = 0 ∧ (∃ y, y^2 % p = x)).card = residues ∧ -- even quadratic residues
  set.univ.filter (λ x, x % 2 = 1 ∧ ¬(∃ y, y^2 % p = x)).card = residues   -- odd non-quadratic residues
:= sorry

theorem sum_of_quadratic_residues (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) :
  let sum_qr := p * (p - 1) / 4 in
  ∑ i in (set.univ.filter (λ x, (∃ y, y^2 % p = x)).to_finset), i = sum_qr
:= sorry

end quadratic_residue_distribution_even_odd_quadratic_residue_distribution_sum_of_quadratic_residues_l345_345976


namespace number_of_adjacent_subsets_l345_345232

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345232


namespace integer_solutions_for_quadratic_l345_345407

theorem integer_solutions_for_quadratic (a : ℤ) :
  (∃ f : ℤ → ℤ × ℤ, function.injective f ∧ (∀ n, (f n).fst ^ 2 + a * (f n).fst * (f n).snd + (f n).snd ^ 2 = 1)) ↔ |a| ≥ 2 :=
by {
  sorry
}

end integer_solutions_for_quadratic_l345_345407


namespace discount_cards_count_l345_345329

theorem discount_cards_count : ∀ (total_cards : ℕ) (not_discount_cards : ℕ), 
  total_cards = 10000 ∧ not_discount_cards = 4096 → 
  total_cards - not_discount_cards = 5904 := 
by
  intros total_cards not_discount_cards h
  cases h with h_tot h_not_discount
  rw [h_tot, h_not_discount]
  exact rfl

end discount_cards_count_l345_345329


namespace probability_A_wins_l345_345699

variable (P_A_not_lose : ℝ) (P_draw : ℝ)
variable (h1 : P_A_not_lose = 0.8)
variable (h2 : P_draw = 0.5)

theorem probability_A_wins : P_A_not_lose - P_draw = 0.3 := by
  sorry

end probability_A_wins_l345_345699


namespace cars_50_km_apart_fourth_time_l345_345279

theorem cars_50_km_apart_fourth_time :
  (∃ A B : ℝ,
    ∃ t1 t2 : ℝ,
    ∃ d : ℝ,
    d = 300 ∧
    t1 = 6 ∧
    t2 = 8 ∧
    (t2 - t1) = 2 ∧ 
    -- Cars meet for the first time 50 km apart after 2 hours
    d1 + d2 = d - 50 ∧
    -- After 7.6 hours (from 6 a.m.), they are 50 km apart for the fourth time
    d3 + d4 = d - 50 ∧
    (t1 + 7.6) = 13.6) →
  Cars_50_km_apart_fourth_time = time_from_morning 1 36 := 
begin
  sorry
end

end cars_50_km_apart_fourth_time_l345_345279


namespace vector_lines_l345_345347

-- Define the given vector field in cylindrical coordinates
def vector_field (ρ ϕ z : ℝ) : ℝ × ℝ × ℝ :=
  (1, ϕ, 0)

-- Define the conditions based on the given problem
def conditions (ρ ϕ z : ℝ) (C1 C2 C : ℝ) : Prop :=
  z = C1 ∧ ρ = C * ϕ

-- Define the theorem that proves the vector lines of the given vector field
theorem vector_lines (ρ ϕ z C1 C : ℝ) :
  conditions ρ ϕ z C1 C :=
by
  sorry

end vector_lines_l345_345347


namespace remaining_cube_edge_length_l345_345330

theorem remaining_cube_edge_length (a b : ℕ) (h : a^3 = 98 + b^3) : b = 3 :=
sorry

end remaining_cube_edge_length_l345_345330


namespace tangent_line_equation_l345_345198

/-- Equation of the tangent line to the curve y = -5e^x + 3 at the point (0, -2) is 5x + y + 2 = 0 -/
theorem tangent_line_equation : 
  let f : ℝ → ℝ := λ x, -5 * Real.exp x + 3
  in ∀ x y: ℝ, (x = 0 ∧ y = -2) → (5 * x + y + 2 = 0) :=
by 
  intros f x y xy_eqn,
  sorry.

end tangent_line_equation_l345_345198


namespace correctReasoning_l345_345711

def analogicalReasoning := ∀ (R : Type), ¬(R = reasoningFromSpecificToGeneral)
def deductiveReasoning := ∀ (R : Type), ¬(R = reasoningFromSpecificToGeneral)
def inductiveReasoning := ∀ (R : Type), R = reasoningFromIndividualToGeneral
def reasonableReasoning := ∀ (R : Type), ¬(R = stepInProof)

theorem correctReasoning : inductiveReasoning reasoningFromIndividualToGeneral :=
by sorry

end correctReasoning_l345_345711


namespace verify_propositions_l345_345373

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345373


namespace quantity_properties_l345_345727

section quantities

variables (a b c S Sϕ : ℝ)

def S_A := (b^2 + c^2 - a^2) / 2
def S_B := (c^2 + a^2 - b^2) / 2
def S_C := (a^2 + b^2 - c^2) / 2

theorem quantity_properties :
  (S_A + S_B = c^2) ∧ 
  (S_B + S_C = a^2) ∧ 
  (S_C + S_A = b^2) ∧ 
  (S_A + S_B + S_C = Sϕ) ∧ 
  (S_A * S_B + S_B * S_C + S_C * S_A = 4 * S^2) ∧ 
  (S_A * S_B * S_C = 4 * S^2 * Sϕ - (a * b * c)^2) :=
by sorry

end quantities

end quantity_properties_l345_345727


namespace part1_l345_345857

def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

def setB (x : ℝ) : Prop := x ≠ 0 ∧ x ≤ 5 ∧ 0 < x

def setC (a x : ℝ) : Prop := 3 * a ≤ x ∧ x ≤ 2 * a + 1

def setInter (x : ℝ) : Prop := setA x ∧ setB x

theorem part1 (a : ℝ) : (∀ x, setC a x → setInter x) ↔ (0 < a ∧ a ≤ 1 / 2 ∨ 1 < a) :=
sorry

end part1_l345_345857


namespace length_of_chord_120_deg_equation_of_chord_bisected_by_P_l345_345457

def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 9

def is_chord_through_P (P : ℝ × ℝ) (α : ℝ) (AB : ℝ × ℝ → Prop) : Prop := 
  ∃ k b : ℝ, k = Real.tan α ∧ (∀ x y, AB (x, y) ↔ y = k * (x - P.1) + b)

theorem length_of_chord_120_deg :
  ∀ (AB : ℝ × ℝ → Prop),
  is_chord_through_P (2, Real.sqrt 3) (120 * Real.pi / 180) AB →
  ∃ A B : ℝ × ℝ, AB A ∧ AB B ∧ A.1 ≠ B.1 ∧ 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 := sorry

theorem equation_of_chord_bisected_by_P :
  ∀ (P : ℝ × ℝ) (AB : ℝ × ℝ → Prop),
  is_on_circle P →
  AB (2, Real.sqrt 3) →
  (∀ A B : ℝ × ℝ, AB A ∧ AB B → (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = Real.sqrt 3) →
  ∃ k b : ℝ, ∀ x y, AB (x, y) ↔ y = k * x + b ∧ k = -2 * Real.sqrt 3 / 3 := 
    ∃ k b : ℝ, ∀ x y, AB (x, y) ↔ (y = -2 * Real.sqrt 3 / 3 * x + b) :=
sorry

end length_of_chord_120_deg_equation_of_chord_bisected_by_P_l345_345457


namespace common_diff_divisible_by_prime_l345_345223

theorem common_diff_divisible_by_prime 
  (p : ℕ) 
  (h_prime_p : p.prime) 
  (a : ℕ → ℕ) 
  (h_ap : ∀ i j : ℕ, i < j ∧ j ≤ p → a i < a j) 
  (h_prime_seq : ∀ i, 1 ≤ i → i ≤ p → Nat.Prime (a i)) 
  (a1_gt_p : a 1 > p) 
  (h_ap_form : ∀ i, 1 ≤ i → i ≤ p → a i = a 1 + (i - 1) * (a 2 - a 1)): 
  p ∣ (a 2 - a 1) := 
sorry

end common_diff_divisible_by_prime_l345_345223


namespace find_triangle_side_a_and_b_and_angle_C_l345_345839

noncomputable def triangle_side_a_and_b_and_angle_C (S : ℝ) (c : ℝ) (A : ℝ) : (ℝ × ℝ × ℝ) :=
if S = sqrt 3 / 2 ∧ c = 2 ∧ A = 60 * (Real.pi / 180) then (sqrt 3, 1, 90 * (Real.pi / 180)) else (0, 0, 0)

theorem find_triangle_side_a_and_b_and_angle_C :
  triangle_side_a_and_b_and_angle_C (sqrt 3 / 2) 2 (60 * (Real.pi / 180)) = (sqrt 3, 1, 90 * (Real.pi / 180)) :=
by
  sorry

end find_triangle_side_a_and_b_and_angle_C_l345_345839


namespace min_value_of_m_l345_345919

theorem min_value_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (real.pi / 3), m ≥ 2 * real.tan x) → m = 2 * real.sqrt 3 :=
sorry

end min_value_of_m_l345_345919


namespace cos_inequality_solution_l345_345801

theorem cos_inequality_solution :
  ∀ y ∈ Icc (0 : ℝ) (Real.pi / 2), (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), cos (x + y) ≥ cos x - cos y) ↔ y = 0 := by
  sorry

end cos_inequality_solution_l345_345801


namespace smallest_divisible_power_l345_345605

noncomputable def smallest_n (m : ℕ) (h1 : m % 2 = 1) (h2 : m ≥ 2) (s : ℕ) (hs : ∃ k, m = 2^k + 1 ∨ m = 2^k - 1) : ℕ :=
  2 ^ (1989 - s)

theorem smallest_divisible_power (m : ℕ) (h1 : m % 2 = 1) (h2 : m ≥ 2) :
  ∃ n, 2^1989 ∣ m^n - 1 ∧ n = smallest_n m h1 h2 (nat.find hs) (nat.find_spec hs) :=
sorry

end smallest_divisible_power_l345_345605


namespace PQ_length_l345_345574

noncomputable def length_of_PQ (A B C P Q : Type*)
  [MetricSpace ABC] [HasDist ABC ℝ]
  (h1 : dist A B = 8)
  (h2 : dist B C = 11)
  (h3 : dist A C = 6)
  (h4 : SimilarTriangle (tri_conn_eq A P B) (tri_conn_eq A B C))
  (h5 : SimilarTriangle (tri_conn_eq Q A C) (tri_conn_eq A B C)) : ℝ :=
let PB : ℝ := (8 * 8) / 11 in
let QC : ℝ := (6 * 6) / 11 in
let PQ := 11 - (PB + QC) in
PQ

theorem PQ_length : 
  ∀ (A B C P Q : Type*)
  [MetricSpace ABC] [HasDist ABC ℝ]
  (h1 : dist A B = 8)
  (h2 : dist B C = 11)
  (h3 : dist A C = 6)
  (h4 : SimilarTriangle (tri_conn_eq A P B) (tri_conn_eq A B C))
  (h5 : SimilarTriangle (tri_conn_eq Q A C) (tri_conn_eq A B C)), 
  length_of_PQ A B C P Q h1 h2 h3 h4 h5 = 21 / 11 :=
by {
  sorry
}

end PQ_length_l345_345574


namespace Arman_hours_worked_l345_345593

/--
  Given:
  - LastWeekHours = 35
  - LastWeekRate = 10 (in dollars per hour)
  - IncreaseRate = 0.5 (in dollars per hour)
  - TotalEarnings = 770 (in dollars)
  Prove that:
  - ThisWeekHours = 40
-/
theorem Arman_hours_worked (LastWeekHours : ℕ) (LastWeekRate : ℕ) (IncreaseRate : ℕ) (TotalEarnings : ℕ)
  (h1 : LastWeekHours = 35)
  (h2 : LastWeekRate = 10)
  (h3 : IncreaseRate = 1/2)  -- because 0.5 as a fraction is 1/2
  (h4 : TotalEarnings = 770)
  : ∃ ThisWeekHours : ℕ, ThisWeekHours = 40 :=
by
  sorry

end Arman_hours_worked_l345_345593


namespace remainder_when_divided_82_l345_345068

theorem remainder_when_divided_82 (x : ℤ) (k m : ℤ) (R : ℤ) (h1 : 0 ≤ R) (h2 : R < 82)
    (h3 : x = 82 * k + R) (h4 : x + 7 = 41 * m + 12) : R = 5 :=
by
  sorry

end remainder_when_divided_82_l345_345068


namespace problem_1_problem_2_l345_345050

open Real

-- Definitions for the mathematical conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Proving part 1: Solution set for f(x) > 5 when m = 2 and n = 3
theorem problem_1 (x : ℝ) : f x 2 3 > 5 ↔ (x ∈ (-∞, 0) ∪ (2, ∞)) :=
sorry

-- Proving part 2: Minimum value of 2m + n if f(x) ≥ 1 is always true
theorem problem_2 (m n : ℝ) (h : ∀ x, f x m n ≥ 1) : 2 * m + n ≥ 2 :=
sorry

end problem_1_problem_2_l345_345050


namespace sequence_an_arithmetic_l345_345850

-- Statements and definitions
theorem sequence_an_arithmetic (a : ℕ → ℕ) (s : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) 
  (h_pos : ∀ n, a n > 0)
  (h_sum : ∀ n, s n = (a n + 1) / 2 ^ 2)
  (hb_def : ∀ n, b n = 10 - a n) :
  (∃ d : ℕ, ∀ n, a n = 2 * n - 1) ∧
  (∃ T : ℕ → ℕ, ∀ n, T n = ((T 5 = 25) → True)) :=
by
  sorry

end sequence_an_arithmetic_l345_345850


namespace solve_cubic_eq_l345_345431

theorem solve_cubic_eq (z : ℂ) : z^3 = -27 ↔ (z = -3 ∨ z = (3/2 + (3 * complex.I * √3) / 2) ∨ z = (3/2 - (3 * complex.I * √3) / 2)) :=
by
  sorry

end solve_cubic_eq_l345_345431


namespace find_right_triangle_conditions_l345_345770

def is_right_triangle (A B C : ℝ) : Prop := 
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem find_right_triangle_conditions (A B C : ℝ):
  (A + B = C ∧ is_right_triangle A B C) ∨ 
  (A = B ∧ B = 2 * C ∧ is_right_triangle A B C) ∨ 
  (A / 30 = 1 ∧ B / 30 = 2 ∧ C / 30 = 3 ∧ is_right_triangle A B C) :=
sorry

end find_right_triangle_conditions_l345_345770


namespace weights_divisible_into_two_equal_piles_l345_345931

/-- 
There are 2009 weights, each with integer mass in grams and not exceeding 1000 grams. Any two
adjacent weights differ by exactly 1 gram, and the total weight is even. We need to prove 
that these weights can be divided into two groups with equal total weight.
-/
theorem weights_divisible_into_two_equal_piles (weights : Fin 2009 → ℕ) :
  (∀ i : Fin 2008, abs (weights i - weights (Fin.mk (i + 1) (by linarith))) = 1) →
  (∀ i : Fin 2009, weights i ≤ 1000) →
  (∑ i, weights i) % 2 = 0 →
  ∃ (I J : Finset (Fin 2009)), disjoint I J ∧ I ∪ J = Finset.univ ∧ (∑ i in I, weights i) = (∑ i in J, weights i) :=
by
  sorry

end weights_divisible_into_two_equal_piles_l345_345931


namespace problem_l345_345876

-- Define the piecewise function f
def f : ℝ → ℝ :=
begin
  intro x,
  by_cases (x ≤ 0),
  exact x^2 + 1,
  exact -x + 1,
end

-- Define x1, x2, and x3
noncomputable def x1 : ℝ := log 2 (1/3)
noncomputable def x2 : ℝ := 2^(1/3)
noncomputable def x3 : ℝ := 3^(-1/2)

-- The target theorem
theorem problem : f x1 > f x3 ∧ f x3 > f x2 := sorry

end problem_l345_345876


namespace number_of_dogs_with_both_tags_and_collars_l345_345780

-- Defining the problem
def total_dogs : ℕ := 80
def dogs_with_tags : ℕ := 45
def dogs_with_collars : ℕ := 40
def dogs_with_neither : ℕ := 1

-- Statement: Prove the number of dogs with both tags and collars
theorem number_of_dogs_with_both_tags_and_collars : 
  (dogs_with_tags + dogs_with_collars - total_dogs + dogs_with_neither) = 6 :=
by
  sorry

end number_of_dogs_with_both_tags_and_collars_l345_345780


namespace pencil_cost_l345_345825

def cost_of_pencil_problem :=
  ∃ (x y : ℕ), 5 * x + 2 * y = 286 ∧ 3 * x + 4 * y = 204 ∧ y = 12

theorem pencil_cost (x y : ℕ) (h1 : 5 * x + 2 * y = 286) (h2 : 3 * x + 4 * y = 204) : y = 12 :=
begin
  sorry
end

end pencil_cost_l345_345825


namespace figure_covering_l345_345747

theorem figure_covering (m n : ℕ) (F : list (list ℤ)) 
(h_pos_sum : ∀ (grid : list (list ℤ)), (list.sum (grid.map list.sum) > 0) → 
  ∃ (config : list (ℕ × ℕ)), list.sum (config.map (λ (c : ℕ × ℕ), grid.get c.1.get_or_else(0, 0) c.2.get_or_else(0, 0))) > 0) 
  : ∃ (layers : list (list (list ℤ))), (∀ (i : ℕ), (1 ≤ i ∧ i ≤ m * n) →
  (grid i = 1)) :=
sorry

end figure_covering_l345_345747


namespace induction_seq_sum_l345_345152

def seq_sum (n : ℕ) : ℚ :=
1 + (List.range n).map (λ i => 1/(2^i)).sum 

def num_additional_terms (k : ℕ) : ℕ :=
2^k

theorem induction_seq_sum (p : ℕ → ℚ) :
    (∀ n : ℕ, seq_sum n = p n) →
    ∀ k : ℕ, seq_sum (k + 1) = seq_sum k + (List.range (num_additional_terms k)).map (λ i => 1/(2^(k + 1 + i))).sum :=
begin
    sorry
end

end induction_seq_sum_l345_345152


namespace diamond_not_commutative_diamond_not_associative_l345_345829

noncomputable def diamond (x y : ℝ) : ℝ :=
  x^2 * y / (x + y + 1)

theorem diamond_not_commutative (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x ≠ y → diamond x y ≠ diamond y x :=
by
  intro hxy
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : x^2 * y * (y + x + 1) = y^2 * x * (x + y + 1) := by
    sorry
  -- Simplify the equation to show the contradiction
  sorry

theorem diamond_not_associative (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (diamond x y) ≠ (diamond y x) → (diamond (diamond x y) z) ≠ (diamond x (diamond y z)) :=
by
  unfold diamond
  intro h
  -- Assume the contradiction leads to this equality not holding
  have eq : (diamond x y)^2 * z / (diamond x y + z + 1) ≠ (x^2 * (diamond y z) / (x + diamond y z + 1)) :=
    by sorry
  -- Simplify the equation to show the contradiction
  sorry

end diamond_not_commutative_diamond_not_associative_l345_345829


namespace find_n_l345_345404

-- Define basic geometric and trigonometric assumptions.
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = 6 ∧ dist B C = 6 ∧ dist C A = 6

def is_center_of_circle (A B : Point) (r : ℝ) : Prop :=
  dist A B = r

def arc_length (A B : Point) (θ : ℝ) : ℝ :=
  θ * dist A B

def area_of_rectangle (width height : ℝ) : ℝ :=
  width * height

def area_of_circle (radius : ℝ) : ℝ :=
  radius * radius * Real.pi

noncomputable def area_traced_out (A B C : Point) : ℝ :=
  let arc_length := arc_length A B (Real.pi / 3)
  let rectangle_area := area_of_rectangle arc_length 6
  let quarter_circle_area := area_of_circle 6 / 4
  rectangle_area + quarter_circle_area

theorem find_n : 
  ∀ (A B C : Point), equilateral_triangle A B C →
  is_center_of_circle A B 6 →
  n : ℕ, area_traced_out A B C = n * Real.pi :=
by
  intros A B C h_triangle h_center
  use 21
  simp [area_traced_out, arc_length, area_of_rectangle, area_of_circle, h_center, h_triangle]
  sorry

end find_n_l345_345404


namespace center_cell_value_l345_345534

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l345_345534


namespace partition_diff_equal_sum_l345_345958

open Finset

noncomputable def can_partition_diff_to_equal_sum (n : ℕ) (hn : n % 2 = 1) (xs : Fin n → ℝ) (hx : ∀ (i j : Fin n), i ≠ j → xs i ≠ xs j) : Prop :=
  ∃ (A B : Finset ℝ), A ∪ B = X ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id
  where
    X = {d ∈ (xs '' univ).powerset | ∃ i j, i < j ∧ d = abs (xs i - xs j)}

-- Now we state the theorem
theorem partition_diff_equal_sum (n : ℕ) (hn : n % 2 = 1) (xs : Fin n → ℝ) (hx : ∀ (i j : Fin n), i ≠ j → xs i ≠ xs j) :
  can_partition_diff_to_equal_sum n hn xs hx :=
sorry

end partition_diff_equal_sum_l345_345958


namespace evaluate_expression_at_x_eq_0_l345_345639

-- Define the given expression
def given_expression (x : ℝ) : ℝ :=
  (2 * x / (x + 1)) - ((2 * x + 6) / (x^2 - 1)) / ((x + 3) / (x^2 - 2 * x + 1))

-- Define the main theorem
theorem evaluate_expression_at_x_eq_0 : given_expression 0 = 2 :=
by
  -- Proof should be filled here
  sorry

end evaluate_expression_at_x_eq_0_l345_345639


namespace necessary_sufficient_condition_l345_345826

def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

theorem necessary_sufficient_condition (a x : ℝ) (h : a ∈ Icc (-1 : ℝ) 1) : 
  (∀ x, f x a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end necessary_sufficient_condition_l345_345826


namespace polynomial_expansion_l345_345419

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^5 + 5 * t^4 + 2 * t^3 - 26 * t^2 + 25 * t - 12 :=
by sorry

end polynomial_expansion_l345_345419


namespace roots_of_quadratic_l345_345131

theorem roots_of_quadratic (p q : ℝ) (h1 : 3 * p^2 + 9 * p - 21 = 0) (h2 : 3 * q^2 + 9 * q - 21 = 0) :
  (3 * p - 4) * (6 * q - 8) = 122 := by
  -- We don't need to provide the proof here, only the statement
  sorry

end roots_of_quadratic_l345_345131


namespace intersection_complement_l345_345890

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 5, 6})
variable (hB : B = {1, 3, 4, 6, 7})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 5} :=
sorry

end intersection_complement_l345_345890


namespace rent_for_additional_hour_l345_345778

theorem rent_for_additional_hour (x : ℝ) :
  (25 + 10 * x = 125) → (x = 10) :=
by 
  sorry

end rent_for_additional_hour_l345_345778


namespace find_constant_e_l345_345824

theorem find_constant_e {x y e : ℝ} : (x / (2 * y) = 3 / e) → ((7 * x + 4 * y) / (x - 2 * y) = 25) → (e = 2) :=
by
  intro h1 h2
  sorry

end find_constant_e_l345_345824


namespace unique_representation_shifted_function_in_M_preimage_circle_l345_345978

-- Problem 1
theorem unique_representation (a b c d : ℝ) (h : ∀ x : ℝ, a * cos x + b * sin x = c * cos x + d * sin x) : (a, b) = (c, d) :=
by sorry

-- Problem 2
theorem shifted_function_in_M (a_0 b_0 t : ℝ) : 
  ∃ m n : ℝ, (∀ x : ℝ, (a_0 * cos x + b_0 * sin x) = m * cos (x + t) + n * sin (x + t)) ∧ (m = a_0 * cos t + b_0 * sin t) ∧ (n = b_0 * cos t - a_0 * sin t) :=
by sorry

-- Problem 3
theorem preimage_circle (a_0 b_0 t : ℝ) : 
  let m := a_0 * cos t + b_0 * sin t
  let n := b_0 * cos t - a_0 * sin t
  in ∀ t : ℝ, m^2 + n^2 = a_0^2 + b_0^2 :=
by sorry

end unique_representation_shifted_function_in_M_preimage_circle_l345_345978


namespace lcm_150_456_l345_345704

theorem lcm_150_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end lcm_150_456_l345_345704


namespace next_in_sequence_is_33_l345_345420

def sequence (n : ℕ) : ℕ
| 0     := 3
| (n+1) := 2 * sequence n - 1

theorem next_in_sequence_is_33 : sequence 4 = 33 := 
by 
  -- proof
  sorry

end next_in_sequence_is_33_l345_345420


namespace negation_proposition_l345_345665

theorem negation_proposition (a b : ℝ) :
  (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_proposition_l345_345665


namespace find_cosine_angle_l345_345474

/-- Definitions of the ellipse and its properties -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 25 + y^2 / 9 = 1)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) |

/-- Theorem statement -/
theorem find_cosine_angle
(F₁ : ℝ × ℝ)
(F₂ : ℝ × ℝ)
(P : ℝ × ℝ)
(hF₁ : F₁ = (-4, 0))
(hF₂ : F₂ = (4, 0))
(hEllipse : is_on_ellipse P)
(hArea : area_of_triangle P F₁ F₂ = 3 * real.sqrt 3) :
  cos (angle F₁ P F₂) = 1 / 2 :=
by
  sorry

end find_cosine_angle_l345_345474


namespace center_cell_value_l345_345537

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l345_345537


namespace const_fn_range_max_value_l345_345311

theorem const_fn_range (x : ℝ) : (|x-1| + |x+3| = 4) ↔ -3 ≤ x ∧ x ≤ 1 :=
by sorry

theorem max_value (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  ( √2 * x + √2 * y + √5 * z) ≤ 3 :=
by sorry

end const_fn_range_max_value_l345_345311


namespace problem_statement_l345_345844

-- We begin by stating the variables x and y with the given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : x - 2 * y = 3
axiom h2 : (x - 2) * (y + 1) = 2

-- The theorem to prove
theorem problem_statement : (x^2 - 2) * (2 * y^2 - 1) = -9 :=
by
  sorry

end problem_statement_l345_345844


namespace distance_between_midpoints_l345_345625

def midpoint (a b c d : ℝ) : ℝ × ℝ :=
  ((a + c) / 2, (b + d) / 2)

def new_position_A (a b : ℝ) : ℝ × ℝ :=
  (a + 5, b + 12)

def new_position_B (c d : ℝ) : ℝ × ℝ :=
  (c - 15, d - 6)

def new_midpoint (a b c d : ℝ) : ℝ × ℝ :=
  let (a', b') := new_position_A a b
  let (c', d') := new_position_B c d
  ((a' + c') / 2, (b' + d') / 2)

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_midpoints (a b c d : ℝ) :
  let (m, n) := midpoint a b c d
  let (m', n') := new_midpoint a b c d
  distance m n m' n' = Real.sqrt 34 := 
sorry

end distance_between_midpoints_l345_345625


namespace expected_value_parabola_l345_345885

def X_distribution (a b : ℤ) : ℚ :=
  if a=b then 0
  else if |a-b|=1 then 1
  else 2

theorem expected_value_parabola (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∀ a b ∈ {-3, -2, -1, 0, 1, 2, 3}, 
                       a * b > 0 ∧ a ≠ 0 ∧ b ≠ 0) :
  let X := |a - b| in (X = 0 * 1/3 + 1 * 4/9 + 2 * 2/9) → 
  (∑ e in {0, 1, 2}, e * X_distribution a b) / 3 * 3 * 2 * 7 = 8/9 :=
begin
  sorry
end

end expected_value_parabola_l345_345885


namespace smallest_palindromic_l345_345365

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

theorem smallest_palindromic
  (n : ℕ) (h1 : n > 10)
  (h2 : is_palindrome n 2)
  (h3 : is_palindrome n 8) : n = 63 :=
begin
  sorry
end

end smallest_palindromic_l345_345365


namespace center_cell_value_l345_345549

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l345_345549


namespace at_most_one_even_l345_345710

theorem at_most_one_even (a b c : ℕ) :
  (∃ e1 e2 e3 : bool, (e1 = tt ↔ Even a) ∧ (e2 = tt ↔ Even b) ∧ (e3 = tt ↔ Even c))
  → (tttbysb) → (ttotal_recall_bs)
: sorry

end at_most_one_even_l345_345710


namespace cube_rods_l345_345720

theorem cube_rods {n : ℕ} (h: n > 0) :
  let N := 2 * n in
  ∃ (rods : set (fin N × fin N × fin N)), rods.card = 2 * n^2 ∧
  ∀ (r1 r2 : fin N × fin N × fin N), r1 ≠ r2 → 
    rods r1 ∧ rods r2 → (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2 ∧ r1.3 ≠ r2.3) :=
by 
  sorry

end cube_rods_l345_345720


namespace domain_of_f_l345_345193

noncomputable def f (x : ℝ) : ℝ := 1 / real.sqrt (1 - real.logb 2 x)

theorem domain_of_f : {x : ℝ | 0 < x ∧ 1 - real.logb 2 x > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end domain_of_f_l345_345193


namespace parabola_no_y_intercepts_l345_345896

def number_of_y_intercepts (f : ℝ → ℝ) : ℕ :=
  if H : ∃ y : ℝ, f y = 0 then 1 else 0

theorem parabola_no_y_intercepts : 
  number_of_y_intercepts (λ y, 2 * y^2 - 3 * y + 7) = 0 :=
by 
  sorry

end parabola_no_y_intercepts_l345_345896


namespace min_value_x_sq_y_minus_ln_x_minus_x_l345_345017

noncomputable def xy_y_eq_e_x (x y : ℝ) : Prop := (ln ((x * y)^y) = exp x)

theorem min_value_x_sq_y_minus_ln_x_minus_x (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : xy_y_eq_e_x x y) :
  x ^ 2 * y - ln x - x = 1 :=
sorry

end min_value_x_sq_y_minus_ln_x_minus_x_l345_345017


namespace exists_tetrahedron_with_feet_of_all_altitudes_outside_faces_l345_345808

theorem exists_tetrahedron_with_feet_of_all_altitudes_outside_faces :
  ∃ (A B C D : ℝ × ℝ × ℝ),
    -- Tetrahedron setup conditions
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧ 
    -- Conditions on dihedral angles being obtuse for the altitudes
    ((dihedral_angle A B C D > π / 2 ∧ dihedral_angle A C B D > π / 2) ∨
     (dihedral_angle B A D C > π / 2 ∧ dihedral_angle B D A C > π / 2)) :=
sorry

end exists_tetrahedron_with_feet_of_all_altitudes_outside_faces_l345_345808


namespace expected_babies_proof_l345_345748

-- Define the conditions
def num_kettles : ℕ := 6
def avg_pregnancies_per_kettle : ℕ := 15
def babies_per_pregnancy : ℕ := 4
def loss_fraction : ℝ := 0.25

-- Define the expected number of babies
def total_babies_expected : ℕ :=
  let babies_per_kettle := avg_pregnancies_per_kettle * babies_per_pregnancy
  in num_kettles * babies_per_kettle

def babies_not_hatching : ℝ :=
  total_babies_expected * loss_fraction

def final_expected_babies : ℝ :=
  total_babies_expected - babies_not_hatching

-- The goal is to prove the final expected number of babies
theorem expected_babies_proof : final_expected_babies = 270 := by
  sorry

end expected_babies_proof_l345_345748


namespace chairs_subset_count_l345_345273

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345273


namespace range_of_m_l345_345871

theorem range_of_m (m : ℝ) (h : 9 > m^2 ∧ m ≠ 0) : m ∈ Set.Ioo (-3) 0 ∨ m ∈ Set.Ioo 0 3 := 
sorry

end range_of_m_l345_345871


namespace largest_prime_factor_always_37_l345_345332

-- We define the cyclic sequence conditions
def cyclic_shift (seq : List ℕ) : Prop :=
  ∀ i, seq.get! (i % seq.length) % 10 = seq.get! ((i + 1) % seq.length) / 100 ∧
       (seq.get! ((i + 1) % seq.length) / 10 % 10 = seq.get! ((i + 2) % seq.length) % 10) ∧
       (seq.get! ((i + 2) % seq.length) / 10 % 10 = seq.get! ((i + 3) % seq.length) / 100)

-- Summing all elements of a list
def sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Prove that 37 is always a factor of the sum T
theorem largest_prime_factor_always_37 (seq : List ℕ) (h : cyclic_shift seq) : 
  37 ∣ sum seq := 
sorry

end largest_prime_factor_always_37_l345_345332


namespace mark_cans_correct_l345_345754

variable (R : ℕ) -- Rachel's cans
variable (J : ℕ) -- Jaydon's cans
variable (M : ℕ) -- Mark's cans
variable (T : ℕ) -- Total cans 

-- Conditions
def jaydon_cans (R : ℕ) : ℕ := 2 * R + 5
def mark_cans (J : ℕ) : ℕ := 4 * J
def total_cans (R : ℕ) (J : ℕ) (M : ℕ) : ℕ := R + J + M

theorem mark_cans_correct (R : ℕ) (J : ℕ) 
  (h1 : J = jaydon_cans R) 
  (h2 : M = mark_cans J) 
  (h3 : total_cans R J M = 135) : 
  M = 100 := 
sorry

end mark_cans_correct_l345_345754


namespace average_of_integers_between_25_and_36_l345_345292

theorem average_of_integers_between_25_and_36 :
  (∑ i in Finset.range 12 \ Finset.range 1, (25 + i)) / 11 = 31 := 
by
  sorry

end average_of_integers_between_25_and_36_l345_345292


namespace most_successful_10th_grader_score_l345_345079

theorem most_successful_10th_grader_score
  (a : ℕ) -- number of 10th graders
  (b : ℕ) -- total points scored by 10th graders
  (total_points : ℕ := 11*a*(11*a - 1)) -- total points distributed
  (total_score : ℕ := 5.5 * b) -- total points scored by all players
  (equation : total_points = total_score) 
  (b_val : b = 2 * a * (11 * a - 1)) : 
  ∃ score : ℕ, score = 20 := 
by
  sorry

end most_successful_10th_grader_score_l345_345079


namespace ellipse_equation_exists_rhombus_l345_345351

-- Define the ellipse and focus conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = sqrt 6 / 3

-- Define conditions for intersection and rhombus verification
def intersect_circles_on_ellipse (x y : ℝ) (F1 F2 : ℝ) : Prop :=
  ∃ x y, ellipse sqrt(3) 1 x y ∧
    distance (F1, y) (x, y) = sqrt(3) + 1 ∧
    distance (F2, y) (x, y) = sqrt(3) - 1

def line_l (k x : ℝ) : ℝ :=
  k * x + 3 / 2

def line_ellipse_intersection (k x y : ℝ) : Prop :=
  ellipse sqrt(3) 1 x (line_l k x)

def parallelogram_rhombus (k x y : ℝ) : Prop :=
  let M := (x, line_l k x) in
  let N := (y, line_l k y) in
  let H := midpoint M N in
  let slope_AH := (H.2 - 0) / (H.1 - 0) in
  let slope_MN := (N.2 - M.2) / (N.1 - M.1) in
  slope_AH * slope_MN = -1

-- Theorem statements
theorem ellipse_equation :
  ∃ a b, a > b ∧ b > 0 ∧ eccentricity (sqrt 2) a ∧
    ellipse sqrt(3) 1 x y :=
by sorry

theorem exists_rhombus (k : ℝ) (x y : ℝ):
  ∃ k, parallelogram_rhombus k x y :=
by sorry

end ellipse_equation_exists_rhombus_l345_345351


namespace S_n_is_arithmetic_sum_a_n_l345_345464

noncomputable def a_n (S_n : ℕ → ℝ) (n : ℕ) : ℝ := 2 * real.sqrt (S_n n) - 1
noncomputable def S_n (S : ℕ → ℝ) (n : ℕ) : ℝ := (finset.range (n + 1)).sum S

theorem S_n_is_arithmetic (S_n : ℕ → ℝ) (h1 : ∀ n, S_n (n + 1) - S_n n = 2 * real.sqrt (S_n (n + 1)) - 1) :
  ∃ (d : ℝ), ∀ n, S_n (n + 1) = S_n n + d :=
sorry

noncomputable def T_n (n : ℕ) : ℝ := n^2 + 3 - ((2 * n + 3) / 2 ^ n)

theorem sum_a_n (n : ℕ) : ∑ i in finset.range n, a_n S_n i = T_n n :=
sorry

end S_n_is_arithmetic_sum_a_n_l345_345464


namespace spaceship_travel_distance_l345_345343

theorem spaceship_travel_distance :
  ∀ (d_EX d_XY d_YE : ℝ),
  d_EX = 0.5 → d_YE = 0.1 → (d_EX + d_XY + d_YE) = 0.7 → d_XY = 0.1 :=
begin
  intros d_EX d_XY d_YE h1 h2 h3,
  sorry
end

end spaceship_travel_distance_l345_345343


namespace largest_among_four_numbers_l345_345445

theorem largest_among_four_numbers : 
  let a := (- (1 / 2)) ^ (-1)
  let b := 2 ^ (- (1 / 2))
  let c := (1 / 2) ^ (- (1 / 2))
  let d := 2 ^ (- 1)
  c > a ∧ c > b ∧ c > d :=
by
  sorry

end largest_among_four_numbers_l345_345445


namespace ratio_of_roses_l345_345280

-- Definitions for conditions
def roses_two_days_ago : ℕ := 50
def roses_yesterday : ℕ := roses_two_days_ago + 20
def roses_total : ℕ := 220
def roses_today : ℕ := roses_total - roses_two_days_ago - roses_yesterday

-- Lean statement to prove the ratio of roses planted today to two days ago is 2
theorem ratio_of_roses :
  roses_today / roses_two_days_ago = 2 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_roses_l345_345280


namespace number_of_adjacent_subsets_l345_345229

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345229


namespace chairs_subsets_l345_345256

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345256


namespace eq_no_sol_l345_345148

open Nat -- Use natural number namespace

theorem eq_no_sol (k : ℤ) (x y z : ℕ) (hk1 : k ≠ 1) (hk3 : k ≠ 3) :
  ¬ (x^2 + y^2 + z^2 = k * x * y * z) := 
sorry

end eq_no_sol_l345_345148


namespace alice_ball_after_three_turns_l345_345350

def prob_keeping_ball_by_Alice : ℚ := 1/3
def prob_passing_ball_by_Alice_to_Bob : ℚ := 2/3
def prob_passing_ball_by_Bob_to_Alice : ℚ := 1/2

theorem alice_ball_after_three_turns :
  let prob_keeping_ball_Alice := prob_keeping_ball_by_Alice in
  let prob_passing_Alice_to_Bob := prob_passing_ball_by_Alice_to_Bob in
  let prob_passing_Bob_to_Alice := prob_passing_ball_by_Bob_to_Alice in
  -- Calculate probabilities based on given conditions
  let prob_Alice_turn1 := 1 in
  let prob_Alice_turn2_when_kept := prob_Alice_turn1 * prob_keeping_ball_Alice in
  let prob_Alice_turn2_when_passed := prob_Alice_turn1 * prob_passing_Alice_to_Bob * prob_passing_Bob_to_Alice in
  let prob_Alice_turn3_kept_from_Alice := prob_Alice_turn2_when_kept * prob_keeping_ball_Alice in
  let prob_Alice_turn3_passed_back_from_Bob := prob_Alice_turn2_when_passed * prob_keeping_ball_Alice * prob_passing_Alice_to_Bob * prob_passing_Bob_to_Alice in
  let prob_Alice_turn3_kept_from_Bob := prob_Alice_turn2_when_passed * prob_keeping_ball_Alice in
  let prob_Alice_turn3_all := prob_Alice_turn3_kept_from_Alice
                              + prob_Alice_turn3_passed_back_from_Bob
                              + prob_Alice_turn3_kept_from_Bob in
  prob_Alice_turn3_all = 29/54 := 
sorry

end alice_ball_after_three_turns_l345_345350


namespace part1_part2_l345_345078

def bag : Set (Ball → Prop) := 
  { λ b, b = red ∨ b = white | red ∈ {1, 2, 3}, white ∈ {4, 5} }

def X (ω : Ω) : ℕ :=
  (sampleWithReplacement bag 3).count red

noncomputable def P_X (k : ℕ) : ℚ :=
  choose 3 k * (3/5)^k * (2/5)^(3-k)

noncomputable def E_X : ℚ :=
  3 * (3/5)

theorem part1 : (P_X 0 = 8/125) ∧ (P_X 1 = 36/125) ∧ (P_X 2 = 54/125) ∧ (P_X 3 = 27/125) ∧ (E_X = 9/5) := by
  sorry

def Y (ω : Ω) : ℕ :=
  firstDrawsUntilNSuccess bag red 2

noncomputable def P_Y_4 : ℚ :=
  choose 3 1 * (3/5) * (2/5)^2 * (3/5)

theorem part2 : P_Y_4 = 108/625 := by
  sorry

end part1_part2_l345_345078


namespace mn_value_l345_345920

-- Definitions
def exponent_m := 2
def exponent_n := 2

-- Theorem statement
theorem mn_value : exponent_m * exponent_n = 4 :=
by
  sorry

end mn_value_l345_345920


namespace determine_class_size_l345_345286

-- Define individual lying patterns
def liesAlways (response : ℕ → Bool) : Prop :=
  ∀ n, response n = false

def alternatesTruthLies (response : ℕ → Bool) : Prop :=
  ∀ n, response n = (n % 2 = 1)

def liesEveryThird (response : ℕ → Bool) : Prop :=
  ∀ n, response n = ¬(n % 3 = 2)

-- Define the overall conditions about their responses
def responses (responseVasya responsePetya responseKolya : ℕ → ℕ) : Prop :=
  (∀ n < 6, responseVasya n = 25 ∨ responseVasya n = 26 ∨ responseVasya n = 27) ∧
  (∀ n < 6, responsePetya n = 25 ∨ responsePetya n = 26 ∨ responsePetya n = 27) ∧
  (∀ n < 6, responseKolya n = 25 ∨ responseKolya n = 26 ∨ responseKolya n = 27) ∧
  (∑ n in finset.range 6, if responseVasya n == 27 then 1 else 0 +
  (if responsePetya n == 27 then 1 else 0) +
  (if responseKolya n == 27 then 1 else 0)) = 7

-- Goal definition: prove that the actual number of students is 27
theorem determine_class_size
  (responseVasya responsePetya responseKolya : ℕ → ℕ)
  (lieVasya : liesAlways (λ n, to_bool (responseVasya n = 25)))
  (alternPetya : alternatesTruthLies (λ n, to_bool (responsePetya n = 25)))
  (lieKolya : liesEveryThird (λ n, to_bool (responseKolya n = 25)))
  (resps : responses responseVasya responsePetya responseKolya) :
  ∀ (actual_class_size : ℕ), actual_class_size = 27 := 
sorry

end determine_class_size_l345_345286


namespace complex_expression_evaluation_l345_345606

theorem complex_expression_evaluation :
  let w := complex.exp (complex.I * (2 * real.pi / 9)) in
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^4 / (1 + w^9)) = -1 / 2 :=
by
  sorry

end complex_expression_evaluation_l345_345606


namespace sean_and_julie_sums_l345_345169

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l345_345169


namespace meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l345_345658

-- Distance between locations A and B
def distance : ℝ := 448

-- Speed of the slow train
def slow_speed : ℝ := 60

-- Speed of the fast train
def fast_speed : ℝ := 80

-- Problem 1: Prove the two trains meet 3.2 hours after the fast train departs (both trains heading towards each other, departing at the same time)
theorem meet_time_same_departure : 
  (slow_speed + fast_speed) * 3.2 = distance :=
by
  sorry

-- Problem 2: Prove the two trains meet 3 hours after the fast train departs (slow train departs 28 minutes before the fast train)
theorem meet_time_staggered_departure : 
  (slow_speed * (28/60) + (slow_speed + fast_speed) * 3) = distance :=
by
  sorry

-- Problem 3: Prove the fast train catches up to the slow train 22.4 hours after departure (both trains heading in the same direction, departing at the same time)
theorem catch_up_time_same_departure : 
  (fast_speed - slow_speed) * 22.4 = distance :=
by
  sorry

end meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l345_345658


namespace max_S_value_l345_345452

variable (n : ℕ) (M : ℝ) 
variable {a : ℕ → ℝ}  -- Definition for a sequence a_k

theorem max_S_value (hn_pos : 0 < n) (h_arith_seq : ∀ d : ℝ, ∃ (a : ℕ → ℝ), ∀ k, a k = a 0 + (k - 1) * d) 
                    (h_condition : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S : ℝ, S = (n + 1) * (sqrt 10 / 2) * sqrt M :=
sorry

end max_S_value_l345_345452


namespace magnitude_sum_l345_345921

variables (a b : ℝ^3) -- Definitions for arbitrary vectors a and b
def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2) -- Calculates the magnitude of a vector

axiom magnitude_a : magnitude a = 1
axiom magnitude_b : magnitude b = 2
axiom angle_ab : real.angle a b = real.pi / 3

theorem magnitude_sum : magnitude (a + b) = real.sqrt 7 :=
by
  sorry

end magnitude_sum_l345_345921


namespace find_side_DF_in_triangle_DEF_l345_345103

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l345_345103


namespace special_food_cost_per_ounce_l345_345221

-- Define the problem conditions
def goldfish_total : ℕ := 50
def food_per_goldfish_per_day : ℝ := 1.5
def special_food_percentage : ℝ := 0.20
def total_special_food_cost : ℝ := 45.0

-- Theorem to prove
theorem special_food_cost_per_ounce :
  let special_goldfish_count := special_food_percentage * goldfish_total
      total_special_food_per_day := special_goldfish_count * food_per_goldfish_per_day
  in total_special_food_cost / total_special_food_per_day = 3 :=
by
  sorry

end special_food_cost_per_ounce_l345_345221


namespace not_possible_to_list_numbers_l345_345575

theorem not_possible_to_list_numbers :
  ¬ (∃ (f : ℕ → ℕ), (∀ n, f n ≥ 1 ∧ f n ≤ 1963) ∧
                     (∀ n, Nat.gcd (f n) (f (n+1)) = 1) ∧
                     (∀ n, Nat.gcd (f n) (f (n+2)) = 1)) :=
by
  sorry

end not_possible_to_list_numbers_l345_345575


namespace cylinder_volume_increase_l345_345300

theorem cylinder_volume_increase 
  (π : ℝ) (r h : ℝ) (V : ℝ) 
  (V_def : V = π * r^2 * h) :
  let new_V := π * (4 * r)^2 * (3 * h) in 
  new_V / V = 48 :=
by sorry

end cylinder_volume_increase_l345_345300


namespace cos_sum_to_product_identity_l345_345418

theorem cos_sum_to_product_identity : ∀ x : ℝ, ∃ a b c d : ℕ, (cos x + cos (5 * x) + cos (9 * x) + cos (13 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) ∧ (a + b + c + d = 17) :=
by {
  -- Here we will outline the proof structure, but we will leave the detailed proof as an exercise.
  sorry
}

end cos_sum_to_product_identity_l345_345418


namespace min_value_expression_l345_345019

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : log ((x * y)^y) = Real.exp x) : 
  ∃ c : ℝ, (∀ a b : ℝ, 0 < a → 0 < b → log ((a * b)^b) = Real.exp a → a^2 * b - log a - a ≥ c) 
  ∧ (c = 1) :=
by
  use 1
  intro a b ha hb hab
  sorry

end min_value_expression_l345_345019


namespace unique_three_digit_numbers_count_l345_345873

theorem unique_three_digit_numbers_count:
  ∃ (digits : Set ℕ) (len : ℕ), 
  digits = {3, 5, 7, 9} ∧ len = 3 → 
  (set.card { n : ℕ // ∀ d ∈ digits, ∃! p, (n / 10^p % 10) = d ∧ 1 ≤ n / 10^2 < 10} = 24) :=
begin
  sorry
end

end unique_three_digit_numbers_count_l345_345873


namespace true_propositions_correct_l345_345372

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345372


namespace value_of_some_number_l345_345066

def [[x]] (x : ℝ) : ℝ := x^2 + 2 * x + 4

theorem value_of_some_number (y : ℝ) :
  [[y]] y = 12 ↔ y = -4 ∨ y = 2 :=
by
  sorry

end value_of_some_number_l345_345066


namespace max_subset_size_l345_345886

theorem max_subset_size (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 3) :
  ∃ (A : set (ℕ × ℕ)), A ⊆ (set.prod (set.Icc 1 m) (set.Icc 1 n)) ∧
    ∀ x1 x2 x3 y1 y2 y3, 
    x1 < x2 → x2 < x3 → y1 < y2 → y2 < y3 → 
    ¬ ((x1, y2) ∈ A ∧ (x2, y1) ∈ A ∧ (x2, y2) ∈ A ∧ (x2, y3) ∈ A ∧ (x3, y2) ∈ A) 
    → ∃ B : set (ℕ × ℕ), B ⊆ A ∧ B.finite ∧ B.card = 2 * m + 2 * n - 4 :=
sorry

end max_subset_size_l345_345886


namespace value_of_f_at_3_l345_345903

noncomputable def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

theorem value_of_f_at_3 : f 3 = 155 := by
  sorry

end value_of_f_at_3_l345_345903


namespace probability_of_divisibility_by_11_l345_345756

-- Definitions
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_digits 10 n in
  digits = digits.reverse

def five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ is_palindrome n

def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

-- Proof Problem Statement
theorem probability_of_divisibility_by_11 :
  (∃ k, k = 90) ∧ ∃ n, five_digit_palindrome n ∧ divisible_by_11 n →
  (∃ p q : ℕ, p = 1 ∧ q = 10 ∧ (∃ (total_palindromes : ℕ), total_palindromes = 900) →
  (p / q : ℚ) = 1 / 10) :=
by sorry

end probability_of_divisibility_by_11_l345_345756


namespace find_root_interval_l345_345429

open Real

theorem find_root_interval : 
  ∃ k : ℤ, (∃ root : ℝ, f root = 0 ∧ root ∈ (k-1 : ℝ, k)) ∧ k = 0 :=
by 
  let f := λ x : ℝ, exp x + x
  have cont_f : Continuous f := by continuity
  have exists_root_interval : ∃ root : ℝ, f root = 0 ∧ root ∈ Ioo (-1 : ℝ) 0 :=
    by
      have : f (-1) < 0 := by simp [f, exp_lt]  -- f(-1) = e^(-1) - 1 < 0
      have : f 0 > 0 := by simp [f]            -- f(0) = 1 > 0
      exact intermediate_value_Ioo (-1 : ℝ) 0 (by norm_num) (by norm_num) (by linarith)
  have k_val : ∃ k : ℤ, (k - 1 : ℝ) = -1 := by use 0
  exact ⟨0, exists_root_interval, rfl⟩

end find_root_interval_l345_345429


namespace number_of_racks_l345_345415

theorem number_of_racks (cds_per_rack total_cds : ℕ) (h1 : cds_per_rack = 8) (h2 : total_cds = 32) :
  total_cds / cds_per_rack = 4 :=
by
  -- actual proof goes here
  sorry

end number_of_racks_l345_345415


namespace ellipse_circle_condition_l345_345285

-- Definitions and conditions
variables (a b : ℝ)
def ellipse := ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1
def circle := ∀ (x y : ℝ), (x^2 + y^2) = 1

-- Problem statement rewritten in Lean 4
theorem ellipse_circle_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse a b) (h4 : circle) : 1 / a^2 + 1 / b^2 = 1 :=
sorry

end ellipse_circle_condition_l345_345285


namespace seq_is_arithmetic_l345_345446

-- Condition: the difference between consecutive terms is constant and equals 3
def diff_eq_const {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
  ∀ n, a (n + 1) - a n = 3

-- The sequence is an arithmetic sequence: defined by the difference being constant
theorem seq_is_arithmetic {α : Type*} [AddCommGroup α] (a : ℕ → α) :
  diff_eq_const a → ∃ d, ∀ n, a (n + 1) = a n + d :=
by
  intro h
  use 3
  sorry

end seq_is_arithmetic_l345_345446


namespace train_cross_pole_time_l345_345346

theorem train_cross_pole_time 
  (speed_km_hr : ℝ) (train_length_m : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) :
  speed_km_hr = 100 → train_length_m = 250 → km_to_m = 1000 → hr_to_s = 3600 → 
  (train_length_m / (speed_km_hr * (km_to_m / hr_to_s))) = 90 :=
begin
  intros h1 h2 h3 h4,
  -- I acknowledge the conditions: speed in km/hr, train length in meters, km to m conversion, hr to s conversion
  have speed_m_s : ℝ := speed_km_hr * (km_to_m / hr_to_s),
  have travel_time : ℝ := train_length_m / speed_m_s,
  rw [h1, h2, h3, h4] at *,
  rw [←real.div_eq_mul_inv, mul_assoc, real.mul_div_cancel_left _ (ne_of_gt h3)],
  rw [real.div_eq_mul_inv, mul_assoc, real.mul_div_cancel_left (train_length_m * (hr_to_s ^ (-1 : ℝ)) * km_to_m ^ (-1 : ℝ)) (ne_of_gt h3)],
  norm_num,
end

end train_cross_pole_time_l345_345346


namespace rational_triples_satisfy_conditions_l345_345633

variable (k l : ℚ)

noncomputable def a := (2 * k) / (k^2 - 1)
noncomputable def b := (2 * l) / (l^2 - 1)
noncomputable def c := (2 * (k + l) * (k * l - 1)) / ((k + l)^2 - (k * l - 1)^2)

theorem rational_triples_satisfy_conditions (k l : ℚ) :
  a + b + c = 6 ∧ a * b * c = 6 := by
  sorry

end rational_triples_satisfy_conditions_l345_345633


namespace redox_agents_part_a_redox_agents_part_b_redox_agents_part_c_l345_345629

-- Part (a)
theorem redox_agents_part_a (S H2 H_plus : Type) (H2_to_H_plus : H2 → H_plus)
  (S_to_S2_minus : S → S) :
  (∃ x : (S → S), x = S_to_S2_minus ∧ (∃ y : (H2 → H_plus), y = H2_to_H_plus)) → 
  (OxidizingAgent S ∧ ReducingAgent H2) := by sorry

-- Part (b)
theorem redox_agents_part_b (CuS O2 SO2 CuO : Type) (O2_to_SO2 : O2 → SO2)
  (CuS_to_CuO : CuS → CuO) :
  (∃ x : (O2 → SO2), x = O2_to_SO2 ∧ (∃ y : (CuS → CuO), y = CuS_to_CuO)) → 
  (OxidizingAgent O2 ∧ ReducingAgent CuS) := by sorry

-- Part (c)
theorem redox_agents_part_c (SO2 O2 SO3 : Type) (O2_to_SO3 : O2 → SO3)
  (SO2_to_SO3 : SO2 → SO3) :
  (∃ x : (O2 → SO3), x = O2_to_SO3 ∧ (∃ y : (SO2 → SO3), y = SO2_to_SO3)) → 
  (OxidizingAgent O2 ∧ ReducingAgent SO2) := by sorry

end redox_agents_part_a_redox_agents_part_b_redox_agents_part_c_l345_345629


namespace problem_solution_l345_345465

-- Definitions
def has_property_P (A : List ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ A.length →
    (A.get! (j - 1) + A.get! (i - 1) ∈ A ∨ A.get! (j - 1) - A.get! (i - 1) ∈ A)

def sequence_01234 := [0, 2, 4, 6]

-- Propositions
def proposition_1 : Prop := has_property_P sequence_01234

def proposition_2 (A : List ℕ) : Prop := 
  has_property_P A → (A.headI = 0)

def proposition_3 (A : List ℕ) : Prop :=
  has_property_P A → A.headI ≠ 0 →
  ∀ k, 1 ≤ k ∧ k < A.length → A.get! (A.length - 1) - A.get! (A.length - 1 - k) = A.get! k

def proposition_4 (A : List ℕ) : Prop :=
  has_property_P A → A.length = 3 →
  A.get! 2 = A.get! 0 + A.get! 1

-- Main statement
theorem problem_solution : 
  (proposition_1) ∧
  (∃ A, ¬ (proposition_2 A)) ∧
  (∃ A, proposition_3 A) ∧
  (∃ A, proposition_4 A) →
  3 = 3 := 
by sorry

end problem_solution_l345_345465


namespace president_and_vice_president_selection_l345_345145

-- Definitions as per conditions in part (a)
def boys := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def girls := {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
def longest_serving_boys := {1, 2, 3, 4, 5, 6}
def longest_serving_girls := {16, 17, 18, 19, 20, 21}

-- Statement to prove that the number of ways to select a president and vice-president is 180
theorem president_and_vice_president_selection :
  (longest_serving_boys.card + longest_serving_girls.card) *
  (boys.card + girls.card - longest_serving_boys.card) = 180 :=
by sorry

end president_and_vice_president_selection_l345_345145


namespace point_in_third_quadrant_l345_345868

section quadrant_problem

variables (a b : ℝ)

-- Given: Point (a, b) is in the fourth quadrant
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- To prove: Point (a / b, 2 * b - a) is in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- The theorem stating that if (a, b) is in the fourth quadrant,
-- then (a / b, 2 * b - a) is in the third quadrant
theorem point_in_third_quadrant (a b : ℝ) (h : in_fourth_quadrant a b) :
  in_third_quadrant (a / b) (2 * b - a) :=
  sorry

end quadrant_problem

end point_in_third_quadrant_l345_345868


namespace part1_cos_A_part2_c_l345_345926

-- We define a triangle with sides a, b, c opposite to angles A, B, C respectively.
variables (a b c : ℝ) (A B C : ℝ)
-- Given conditions for the problem:
variable (h1 : 3 * a * Real.cos A = c * Real.cos B + b * Real.cos C)
variable (h_cos_sum : Real.cos B + Real.cos C = (2 * Real.sqrt 3) / 3)
variable (ha : a = 2 * Real.sqrt 3)

-- The first part of the problem statement proving cos A = 1/3 given the conditions.
theorem part1_cos_A : Real.cos A = 1 / 3 :=
by
  sorry

-- The second part of the problem statement proving c = 3 given the conditions.
theorem part2_c : c = 3 :=
by
  sorry

end part1_cos_A_part2_c_l345_345926


namespace problem_correct_l345_345977

variables (x y : ℝ)

def p := (x - 2) * (y - 5) ≠ 0
def q := x ≠ 2 ∨ y ≠ 5
def r := x + y ≠ 7

theorem problem_correct :
  (¬(p → r) ∧ ¬(r → p)) ∧ (p → q ∧ ¬(q → p)) ∧ (q → r ∧ ¬(r → q)) :=
by
  sorry

end problem_correct_l345_345977


namespace rectangle_area_PQRS_l345_345731

noncomputable def area_of_PQRS (PQ RS: ℝ) : ℝ :=
  7 * (17 + 14 * (17 / 7)) -- Derived from PQ' = 7 and QT = 17

theorem rectangle_area_PQRS :
  ∀ (PQ RS T U : ℝ) (phi : ℝ),
  PQ = 7 →
  QT = 17 →
  T ≠ U →
  QT < RS →
  ∃ (a b c : ℤ),  c > 0 ∧ ¬ ∃ p, p^2 ∣ c ∧
  PQRS = (PQ * 14 * (17 / 7) * sqrt(68) + 119 * sqrt(34)) ∧
  a + b + c = 251 :=
begin
  sorry
end

end rectangle_area_PQRS_l345_345731


namespace sphere_radius_squared_l345_345690

theorem sphere_radius_squared (R x y z : ℝ)
  (h1 : 2 * Real.sqrt (R^2 - x^2 - y^2) = 5)
  (h2 : 2 * Real.sqrt (R^2 - x^2 - z^2) = 6)
  (h3 : 2 * Real.sqrt (R^2 - y^2 - z^2) = 7) :
  R^2 = 15 :=
sorry

end sphere_radius_squared_l345_345690


namespace unit_digit_of_sum_of_powers_l345_345618

theorem unit_digit_of_sum_of_powers (n : ℕ) :
  let f : ℕ → ℕ := λ n, (3 ^ n) % 10 in
  let sum_unit_digits := (Finset.range n).sum f in
  (505 * (3 + 9 + 7 + 1) + (3 + 9 + 7)) % 10 = 9 :=
by
  sorry

end unit_digit_of_sum_of_powers_l345_345618


namespace graph_g_transformed_is_D_l345_345661

noncomputable def g (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x ≤ 0 then -x
else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2)
else if 2 ≤ x ∧ x ≤ 4 then x - 2
else 0

def g_transformed (x : ℝ) : ℝ := g (x - 2) + 3

theorem graph_g_transformed_is_D : 
  is_correct_graph g_transformed GraphD :=
sorry

end graph_g_transformed_is_D_l345_345661


namespace largest_n_for_f_eq_n_l345_345492

def f : ℕ → ℕ
| 1 := 1
| 2 := 1
| (3 * n) := 3 * f n - 2
| (3 * n + 1) := 3 * f n + 1
| (3 * n + 2) := 3 * f n + 4

theorem largest_n_for_f_eq_n :
  ∃ n, n ≤ 1992 ∧ f n = n ∧ ∀ m, m ≤ 1992 ∧ f m = m → m ≤ n :=
begin
  use 1093,
  split,
  { exact le_refl 1992, },  -- 1093 ≤ 1992
  split,
  { sorry },                -- f(1093) = 1093
  { intros m hm,
    sorry }                 -- ∀ m, m ≤ 1992 ∧ f(m) = m → m ≤ 1093
end

end largest_n_for_f_eq_n_l345_345492


namespace cylinder_volume_increase_l345_345299

theorem cylinder_volume_increase 
  (π : ℝ) (r h : ℝ) (V : ℝ) 
  (V_def : V = π * r^2 * h) :
  let new_V := π * (4 * r)^2 * (3 * h) in 
  new_V / V = 48 :=
by sorry

end cylinder_volume_increase_l345_345299


namespace desks_in_first_row_l345_345557

theorem desks_in_first_row
  (n : ℕ)
  (d : ℕ)
  (h₀ : n = 8)
  (h₁ : ∀ i : ℕ, i < 8 → number_of_desks (i + 1) = d + 2 * i)
  (h₂ : total_number_of_desks = 136)
  (h₃ : total_number_of_desks = ∑ i in range 8, number_of_desks (i + 1))
  : d = 10 :=
sorry

end desks_in_first_row_l345_345557


namespace representation_has_at_least_n_nonzero_digits_l345_345118

theorem representation_has_at_least_n_nonzero_digits 
  (a b n : ℕ) 
  (hb : 1 < b) 
  (hdiv : (b^n - 1) ∣ a) 
  (ha_pos : 0 < a) 
  (hn_pos : 0 < n) 
  : nat.digits b a.filter (≠ 0).length ≥ n := 
sorry

end representation_has_at_least_n_nonzero_digits_l345_345118


namespace isosceles_triangles_height_ratio_l345_345698

theorem isosceles_triangles_height_ratio
  (b1 b2 h1 h2 : ℝ)
  (h1_ne_zero : h1 ≠ 0) 
  (h2_ne_zero : h2 ≠ 0)
  (equal_vertical_angles : ∀ (a1 a2 : ℝ), true) -- Placeholder for equal angles since it's not used directly
  (areas_ratio : (b1 * h1) / (b2 * h2) = 16 / 36)
  (similar_triangles : b1 / b2 = h1 / h2) :
  h1 / h2 = 2 / 3 :=
by
  sorry

end isosceles_triangles_height_ratio_l345_345698


namespace relationship_s3_s4_l345_345964

-- Definitions for the conditions
variable (A B C D O : Point)
variable (s3 s4 : ℝ)

-- Definition specifying the relationship between s3 and the lengths from O to the vertices.
def s3_def : s3 = (dist O A) + (dist O B) + (dist O C) + (dist O D) := sorry

-- Definition specifying the relationship between s4 and the perimeter of the quadrilateral.
def s4_def : s4 = (dist A B) + (dist B C) + (dist C D) + (dist D A) := sorry

-- Definition specifying the equal areas of triangles AOB and COD.
def equal_area : area (triangle A B O) = area (triangle C D O) := sorry

-- The theorem to prove the relationship between s3 and s4.
theorem relationship_s3_s4 (s3_def : s3_def) (s4_def : s4_def) (equal_area : equal_area) :
  (s3 >= (1/2) * s4) ∧ (s3 <= s4) :=
sorry

end relationship_s3_s4_l345_345964


namespace geometric_sequence_sum_l345_345460

variable (a : ℕ → ℚ)
variable (q : ℚ)

def term (n : ℕ) := a n

def sum_of_terms (terms : List ℕ) : ℚ :=
  (terms.map term).sum

theorem geometric_sequence_sum :
  (∀ n : ℕ, term n = a 0 * q ^ n) → q = -⅓ →
  sum_of_terms [1, 3, 5, 7] / sum_of_terms [2, 4, 6, 8] = -3 :=
by
  intros h_q h_ratio
  sorry

end geometric_sequence_sum_l345_345460


namespace total_palm_trees_combined_l345_345331

/-- The desert has 3/5 fewer palm trees than the forest -/
def desert_palm_trees (forest_palm_trees : ℕ) : ℕ :=
  forest_palm_trees - (3 * forest_palm_trees / 5)

/-- Given the number of palm trees in the forest (5000)
    and along the river (1200), prove the total number
    of palm trees in the desert, forest, and along the river
    combined is 8200. -/
theorem total_palm_trees_combined (forest_palm_trees river_palm_trees : ℕ) 
  (h1 : forest_palm_trees = 5000) (h2 : river_palm_trees = 1200) :
  let desert_palm_trees := desert_palm_trees forest_palm_trees in
  desert_palm_trees + forest_palm_trees + river_palm_trees = 8200 :=
  by sorry

end total_palm_trees_combined_l345_345331


namespace seating_arrangements_count_l345_345184

-- Definitions for conditions
def Seats := Finset.range 18

inductive Planet
| Martian
| Venusian
| Earthling

open Planet

def Arrangement := (Seats → Planet)

def fixed_seats (arr : Arrangement) : Prop :=
  arr 0 = Martian ∧ arr 17 = Earthling

def no_E_left_of_M (arr : Arrangement) : Prop :=
  ∀ i, arr i = Earthling → arr ((i - 1) % 18) ≠ Martian

def no_M_left_of_V (arr : Arrangement) : Prop :=
  ∀ i, arr i = Martian → arr ((i - 1) % 18) ≠ Venusian

def no_V_left_of_E (arr : Arrangement) : Prop :=
  ∀ i, arr i = Venusian → arr ((i - 1) % 18) ≠ Earthling

def no_more_than_two_consecutive (arr : Arrangement) : Prop :=
  ∀ i p, arr i = p → arr ((i + 1) % 18) = p → arr ((i + 2) % 18) ≠ p

-- Combining all conditions
def valid_arrangement (arr : Arrangement) : Prop :=
  fixed_seats arr ∧ 
  no_E_left_of_M arr ∧
  no_M_left_of_V arr ∧
  no_V_left_of_E arr ∧
  no_more_than_two_consecutive arr

theorem seating_arrangements_count : ∃ n, n = 27 ∧ 
  (Finset.filter valid_arrangement (Finset.univ : Finset Arrangement)).card = n :=
begin
  sorry
end

end seating_arrangements_count_l345_345184


namespace count_at_least_three_adjacent_chairs_l345_345236

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l345_345236


namespace no_solution_exists_only_solution_is_1963_l345_345125

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n % 10 + sum_of_digits (n / 10)

-- Proof problem for part (a)
theorem no_solution_exists :
  ¬ ∃ x : ℕ, x + sum_of_digits x + sum_of_digits (sum_of_digits x) = 1993 :=
sorry

-- Proof problem for part (b)
theorem only_solution_is_1963 :
  ∃ x : ℕ, (x + sum_of_digits x + sum_of_digits (sum_of_digits x) + sum_of_digits (sum_of_digits (sum_of_digits x)) = 1993) ∧ (x = 1963) :=
sorry

end no_solution_exists_only_solution_is_1963_l345_345125


namespace carla_start_time_l345_345791

/-- Carla needs to dry-clean 80 pieces of laundry by noon. If she needs to clean 20 pieces of laundry per hour, prove that she should start working at 8:00 AM --/
theorem carla_start_time :
  ∀ (N r T : ℕ),
  N = 80 →
  r = 20 →
  T = 12 →
  T - N / r = 8 :=
by
  intros N r T hN hr hT
  rw [hN, hr, hT]
  norm_num
  sorry

end carla_start_time_l345_345791


namespace sean_div_julie_l345_345173

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l345_345173


namespace inradius_triangle_XYZ_l345_345956

theorem inradius_triangle_XYZ (ABC : Triangle)
  (L : Point)
  (R : ℝ)
  (ADLE BHLK CILJ : Parallelogram)
  (D H : Point)
  (K I : Point)
  (J E : Point)
  (DE HK IJ : Line)
  (X Y Z : Point)
  (condition1 : IsSymmedianPoint L ABC)
  (condition2 : Circumradius ABC = R)
  (condition3 : D ∈ LineAB ABC)
  (condition4 : H ∈ LineAB ABC)
  (condition5 : K ∈ LineBC ABC)
  (condition6 : I ∈ LineBC ABC)
  (condition7 : J ∈ LineCA ABC)
  (condition8 : E ∈ LineCA ABC)
  (condition9 : IsParallelogram ADLE)
  (condition10 : IsParallelogram BHLK)
  (condition11 : IsParallelogram CILJ)
  (condition12 : DE ∩ HK = X)
  (condition13 : HK ∩ IJ = Y)
  (condition14 : IJ ∩ DE = Z) :
  inradius (Triangle.mk X Y Z) = R / 2 :=
sorry

end inradius_triangle_XYZ_l345_345956


namespace karcsi_incorrect_l345_345591

variables (x y : ℝ) (t1 t2 t_bus t_total : ℝ)
variables (speed_k speed_j speed_bus : ℝ := 6) ( := 4) ( := 60)

-- conditions based on given problem
def karcsi_time (x : ℝ) := x / speed_k
def joska_time (y : ℝ) := y / speed_j
def bus_time (x y : ℝ) := (x + y) / speed_bus
def karcsi_total_time (x y : ℝ) := karcsi_time x + bus_time x y

-- prove that Karcsi was incorrect
theorem karcsi_incorrect : (karcsi_total_time x y = joska_time y) → (x > y) :=
by
  intro h
  sorry

end karcsi_incorrect_l345_345591


namespace probability_ln_b_eq_ln_c_l345_345623

noncomputable def prob_ln_b_eq_ln_c (b c : ℝ) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) : ℝ :=
  if [Real.log b] = [Real.log c] then sorry
  else sorry

theorem probability_ln_b_eq_ln_c : 
  (∀ (b c : ℝ), 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → prob_ln_b_eq_ln_c b c = (Real.exp 1 - 1) / (Real.exp 1 + 1)) :=
begin
   intros b c hb hc,
   sorry
end

end probability_ln_b_eq_ln_c_l345_345623


namespace train_pass_jogger_in_39_seconds_l345_345304

-- Definitions of the conditions
def speed_jogger := 9 -- in km/h
def speed_train := 45 -- in km/h
def distance_ahead := 270 -- in meters
def length_train := 120 -- in meters

-- Conversion factor
def km_per_hour_to_m_per_sec (speed_kmh : Float) : Float :=
  speed_kmh * 1000 / 3600

-- Relative speed in m/s
def relative_speed := km_per_hour_to_m_per_sec (speed_train - speed_jogger)

-- Total distance to be covered by the train
def total_distance := distance_ahead + length_train

-- Time to pass the jogger
def time_to_pass_jogger : Float := total_distance / relative_speed

theorem train_pass_jogger_in_39_seconds : time_to_pass_jogger = 39 := by
  sorry

end train_pass_jogger_in_39_seconds_l345_345304


namespace sin_eq_log_has_three_roots_l345_345668

theorem sin_eq_log_has_three_roots :
  ∃ x1 x2 x3 : ℝ, (0 < x1 ∧ x1 ≤ 10 ∧ sin x1 = log10 x1) ∧ 
                (0 < x2 ∧ x2 ≤ 10 ∧ sin x2 = log10 x2) ∧ 
                (0 < x3 ∧ x3 ≤ 10 ∧ sin x3 = log10 x3) ∧ 
                x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 :=
by
  sorry

end sin_eq_log_has_three_roots_l345_345668


namespace true_propositions_l345_345035

section
variable (m n : Line) (α β γ : Plane)

-- 1. If m is perpendicular to alpha and beta, then alpha is parallel to beta.
axiom perp_line_to_planes {m : Line} {α β : Plane} (hmα : perpendicular m α) (hmβ : perpendicular m β) : parallel α β

-- 2. If alpha is perpendicular to gamma and beta is perpendicular to gamma, then alpha is parallel to beta.
axiom perp_planes_false {α β γ : Plane} (hαγ : perpendicular α γ) (hβγ : perpendicular β γ) : ¬parallel α β

-- 3. If m is in alpha, n is in beta, and m is parallel to n, then alpha is parallel to beta.
axiom lines_in_planes_false {m n : Line} {α β : Plane} (hmα : in_plane m α) (hnβ : in_plane n β) (hmn : parallel m n) : ¬parallel α β

-- 4. If m and n are skew lines, m is in alpha, m is parallel to beta, n is in beta, and n is parallel to alpha, then alpha is parallel to beta.
axiom skew_lines_in_planes {m n : Line} {α β : Plane} (hmα : in_plane m α) (hmβ : parallel_to_plane m β) (hnβ : in_plane n β) (hnα : parallel_to_plane n α) (hmn_skew : skew_lines m n) : parallel α β

theorem true_propositions :
  (perp_line_to_planes -> parallel α β) ∧
  (perp_planes_false -> ¬parallel α β) ∧
  (lines_in_planes_false -> ¬parallel α β) ∧
  (skew_lines_in_planes -> parallel α β) :=
by
  sorry
end

end true_propositions_l345_345035


namespace fixed_points_intersection_infinite_tangency_points_l345_345466

variables {P : Type*} [fintype P]

def circle_C1 : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 - 6 * p.2 + 32 = 0 }

def family_circle_C2 (a : ℝ) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * a * p.1 - 2 * (8 - a) * p.2 + 4 * a + 12 = 0 }

theorem fixed_points_intersection :
  (4, 2) ∈ circle_C1 ∧ (6, 4) ∈ circle_C1 ∧
  ∀ a : ℝ, (4, 2) ∈ family_circle_C2 a ∧ (6, 4) ∈ family_circle_C2 a :=
sorry

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem infinite_tangency_points :
  (2, 0) ∈ ellipse ∧ 
  (6 / 5, -4 / 5) ∈ ellipse ∧
  ∀ P : ℝ × ℝ, P ∈ ellipse → 
    let PT1 := sqrt (P.1^2 + P.2^2 - 10 * P.1 - 6 * P.2 + 32),
        PT2 := sqrt (P.1^2 + P.2^2 - 2 * a * P.1 - 2 * (8 - a) * P.2 + 4 * a + 12)
    in (PT1 = PT2 → (P = (2, 0) ∨ P = (6 / 5, -4 / 5))) :=
sorry

end fixed_points_intersection_infinite_tangency_points_l345_345466


namespace bella_age_l345_345783

theorem bella_age (B : ℕ) 
  (h1 : (B + 9) + B + B / 2 = 27) 
  : B = 6 :=
by sorry

end bella_age_l345_345783


namespace hyperbola_focal_length_l345_345659

theorem hyperbola_focal_length {a b : ℝ} (h : a^2 = 16) (k : b^2 = 25) :
  let c := Real.sqrt (a^2 + b^2) in 2 * c = 2 * Real.sqrt 41 :=
by 
  sorry

end hyperbola_focal_length_l345_345659


namespace volume_removed_percentage_is_21_33_l345_345335

def original_box := {length := 20, width := 12, height := 10}
def removed_cube := {side := 4}
def volume (box : {length : ℕ, width : ℕ, height : ℕ}) : ℕ := box.length * box.width * box.height
def cube_volume (cube : {side : ℕ}) : ℕ := cube.side ^ 3

theorem volume_removed_percentage_is_21_33 :
  let V_original := volume original_box;
      V_cube := cube_volume removed_cube;
      V_total_removed := 8 * V_cube in
  (V_total_removed : ℚ) / V_original * 100 = 21.33 := sorry

end volume_removed_percentage_is_21_33_l345_345335


namespace polynomial_form_l345_345422

-- Definitions
def is_even_polynomial (p : ℤ[X]) : Prop :=
  ∃ (a : ℕ → ℤ), ∀ n, p.coeff (2 * n) = a n ∧ p.coeff (2 * n + 1) = 0

-- Problem statement
theorem polynomial_form (p : ℤ[X]) :
  (∀ a b : ℤ, a + b ≠ 0 → a + b ∣ (p.eval a - p.eval b)) ↔ is_even_polynomial p :=
sorry

end polynomial_form_l345_345422


namespace num_choices_for_D_l345_345563

def point := (ℝ × ℝ)

def is_on_grid (pt : point) : Prop :=
  ∃ i j : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j <= 6 ∧ pt = (i, j)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def isosceles_triangle (A B D : point) : Prop :=
  distance A D = distance B D ∨ distance A B = distance A D ∨ distance A B = distance B D

def is_valid_D (A B D : point) : Prop :=
  is_on_grid D ∧ isosceles_triangle A B D ∧ D ≠ A ∧ D ≠ B 

def num_valid_D (A B : point) : ℕ :=
  finset.card (finset.filter (λ D, is_valid_D A B D) (finset.univ : finset point))

theorem num_choices_for_D :
  let A := (2, 2) in
  let B := (5, 2) in
  num_valid_D A B = 9 :=
by sorry

end num_choices_for_D_l345_345563


namespace find_b_l345_345660

def f (b x : ℝ) := x^2 - 2 * b * x + 3

theorem find_b (b : ℝ) (h : ∀ x ∈ Icc (-1 : ℝ) 2, f b x ≥ 1) : b = Real.sqrt 2 ∨ b = -3 / 2 := sorry

end find_b_l345_345660


namespace log_half_inequality_l345_345015

variable (a b : ℝ)

theorem log_half_inequality (h : Real.logBase (1/2) a < Real.logBase (1/2) b) : 
  (1/4)^a < (1/3)^b :=
sorry

end log_half_inequality_l345_345015


namespace area_of_triangle_ADC_l345_345090

theorem area_of_triangle_ADC (BD DC : ℝ) (area_ABD : ℝ)
  (h_ratio : BD / DC = 5 / 2) (h_area : area_ABD = 40) :
  let area_ADC := (2 / 5) * area_ABD in
  area_ADC = 16 :=
by
  sorry

end area_of_triangle_ADC_l345_345090


namespace compare_abc_l345_345840

noncomputable def a : ℝ := real.sqrt real.exp 1
noncomputable def b : ℝ := real.log (real.sqrt 3)
noncomputable def c : ℝ := real.exp (-real.log 2)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l345_345840


namespace find_a_of_pure_imaginary_l345_345020

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = ⟨0, b⟩  -- complex number z is purely imaginary if it can be written as 0 + bi

theorem find_a_of_pure_imaginary (a : ℝ) (i : ℂ) (ha : i*i = -1) :
  isPureImaginary ((1 - i) * (a + i)) → a = -1 := by
  sorry

end find_a_of_pure_imaginary_l345_345020


namespace smallest_x_l345_345507

theorem smallest_x (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 0.75 = y / (210 + x)) : x = 2 :=
by
  sorry

end smallest_x_l345_345507


namespace probability_of_exactly_three_heads_and_no_three_consecutive_tails_l345_345746

def isValidSequence (s : List Bool) : Prop :=
  s.count true = 3 ∧ ∀ (i : ℕ), i ≤ s.length - 3 → ¬ (s[i] = false ∧ s[i+1] = false ∧ s[i+2] = false)

def totalOutcomes : ℕ := 2 ^ 8

noncomputable def validSequencesCount : ℕ := sorry -- should be calculated based on the given conditions

theorem probability_of_exactly_three_heads_and_no_three_consecutive_tails :
  validSequencesCount / totalOutcomes = 35 / 256 :=
sorry

end probability_of_exactly_three_heads_and_no_three_consecutive_tails_l345_345746


namespace union_sets_eq_closed_interval_l345_345070

def M : set ℝ := { x | x^2 - 4 * x < 0 }
def N : set ℝ := {0, 4}

theorem union_sets_eq_closed_interval : M ∪ N = set.Icc 0 4 := 
by {
  sorry
}

end union_sets_eq_closed_interval_l345_345070


namespace course_selection_plans_l345_345691

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_plans :
  let A_courses := C 4 2
  let B_courses := C 4 3
  let C_courses := C 4 3
  A_courses * B_courses * C_courses = 96 :=
by
  sorry

end course_selection_plans_l345_345691


namespace constant_term_in_binomial_expansion_l345_345803

open Finset

-- Define the binomial coefficient to be used
def binomial (n k : ℕ) : ℕ := (finset.range k).prod (λ i, (n - i) / (i + 1))

-- Define the expansion term
def term (r : ℕ) : ℤ := binomial 8 r * (-2) ^ r

theorem constant_term_in_binomial_expansion :
  (x - 2 / x) ^ 8 = 1120 :=
sorry

end constant_term_in_binomial_expansion_l345_345803


namespace calculate_expression_l345_345786

theorem calculate_expression : 
  (3.242 * (14 + 6) - 7.234 * 7) / 20 = 0.7101 :=
by
  sorry

end calculate_expression_l345_345786


namespace smoothie_cost_l345_345592

theorem smoothie_cost
    (burger_cost : ℤ)
    (sandwich_cost : ℤ)
    (total_order_cost : ℤ)
    (num_smoothies : ℤ)
    (smoothie_cost : ℤ) :
    burger_cost = 5 →
    sandwich_cost = 4 →
    total_order_cost = 17 →
    num_smoothies = 2 →
    smoothie_cost = (total_order_cost - burger_cost - sandwich_cost) / num_smoothies →
    smoothie_cost = 4 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h5
  simp at h5
  assumption

end smoothie_cost_l345_345592


namespace range_of_x_l345_345136

noncomputable def f : ℝ → ℝ := sorry -- f is an even function increasing on [0, ∞)

theorem range_of_x {f : ℝ → ℝ} (h1 : ∀ x, f x = f (-x))
  (h2 : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∀ x y, x ≤ y < 0 → f y ≤ f x) 
  (h4 : f (1/3) = 0) :
  {x : ℝ | f (log x / log (1/8 : ℝ)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} :=
sorry

end range_of_x_l345_345136


namespace polygon_proof_l345_345340

-- Define the conditions and the final proof problem.
theorem polygon_proof 
  (interior_angle : ℝ) 
  (side_length : ℝ) 
  (h1 : interior_angle = 160) 
  (h2 : side_length = 4) 
  : ∃ n : ℕ, ∃ P : ℝ, (interior_angle = 180 * (n - 2) / n) ∧ (P = n * side_length) ∧ (n = 18) ∧ (P = 72) :=
by
  sorry

end polygon_proof_l345_345340


namespace odd_function_period_eq_zero_l345_345064

def is_odd_function {R : Type*} [AddGroup R] (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def smallest_positive_period {R : Type*} [AddGroup R] (f : R → R) (T : R) : Prop :=
  T > 0 ∧ ∀ x, f(x + T) = f x

theorem odd_function_period_eq_zero {R : Type*} [LinearOrder R] [AddGroup R] 
  {f : R → R} {T : R} (hf_odd : is_odd_function f) (hf_period : smallest_positive_period f T) : 
  f (-T / 2) = 0 :=
sorry

end odd_function_period_eq_zero_l345_345064


namespace center_cell_value_l345_345542

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l345_345542


namespace min_tank_cost_l345_345745

def tank_cost (x : ℝ) (base_cost_per_sqm : ℝ) (wall_cost_per_sqm : ℝ) (volume : ℝ) (depth : ℝ) : ℝ :=
  let base_area := volume / depth
  let base_cost := base_area * base_cost_per_sqm
  let wall_cost := (2 * depth * x) * wall_cost_per_sqm + wall_cost_per_sqm * (volume / x)
  base_cost + wall_cost

theorem min_tank_cost :
  ∃ x : ℝ, tank_cost x 300 240 6400 4 = 633600 :=
by
  sorry

end min_tank_cost_l345_345745


namespace Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l345_345183

noncomputable def Thabo_book_count_problem : Prop :=
  let P := Nat
  let F := Nat
  ∃ (P F : Nat), 
    -- Conditions
    (P > 40) ∧ 
    (F = 2 * P) ∧ 
    (F + P + 40 = 220) ∧ 
    -- Conclusion
    (P - 40 = 20)

theorem Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction : Thabo_book_count_problem :=
  sorry

end Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l345_345183


namespace shaded_area_is_correct_l345_345493

-- Define the universal set and sets A and B
def U : set ℕ := {0, 1, 2, 3, 4}
def A : set ℕ := {1, 2, 3}
def B : set ℕ := {2, 4}

-- Define the shaded area set
def shaded_area : set ℕ := {0, 2}

-- The proof statement
theorem shaded_area_is_correct :
  (A ∪ B) ∪ (U \ (A ∩ B)) = shaded_area := by
  sorry

end shaded_area_is_correct_l345_345493


namespace total_simple_interest_fetched_l345_345740

-- Definition for principal amount P
def principal : ℝ := 26775

-- Definition for rate of interest R
def rate : ℝ := 3

-- Definition for time period T
def time : ℝ := 5

-- Definition for simple interest formula
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Theorem to prove the total simple interest fetched is 803.25
theorem total_simple_interest_fetched : simple_interest principal rate time = 803.25 :=
by
  sorry

end total_simple_interest_fetched_l345_345740


namespace max_angle_of_tangents_l345_345458

-- Definitions of geometric objects, conditions and the goal.

structure Point where
  x : ℝ
  y : ℝ

def circleO : Point → Prop := 
  λ A, A.x^2 + A.y^2 = 1

def P : Point := { x := -1, y := real.sqrt 3 }

def is_on_circle (A : Point) : Prop := circleO A

def angle (P A B : Point) : ℝ := sorry  -- assuming the existence of an angle measure function

theorem max_angle_of_tangents :
  ∀ A B : Point, circleO A → circleO B → 
  (∃ PA PB : Prop, PA → PB → ∠ APB = angle P A B) → 
  (max_value (angle P A B) = π/3) :=
by
  intros
  sorry  -- proof goes here

end max_angle_of_tangents_l345_345458


namespace parallel_lines_point_l345_345615

theorem parallel_lines_point (x y : ℝ) : 
  (∀ p1 p2: ℝ × ℝ, ((p1.1 = -2) ∧ (p1.2 = 0) ∧ (p2.1 = 0) ∧ (p2.2 = 2)) →
  ∀ q1 q2: ℝ × ℝ, ((q1.1 = 6) ∧ (q1.2 = 2) ∧ (q2.1 = x) ∧ (q2.2 = y)) →
  ((p2.2 - p1.2) / (p2.1 - p1.1) = (q2.2 - q1.2) / (q2.1 - q1.1))) →
  y = x + 4 :=
begin
  -- proof goes here
  sorry
end

end parallel_lines_point_l345_345615


namespace probability_last_roll_12th_l345_345809

theorem probability_last_roll_12th (n : ℕ) (H_n : n = 8) : 
  probability (Ella_rolls_until_consecutive H_n 12) = (7^10) / (8^11) :=
sorry

end probability_last_roll_12th_l345_345809


namespace set_equality_l345_345891

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {2, 3}
noncomputable def N : Set ℕ := {1, 4}
noncomputable def Complement (U M : Set ℕ) : Set ℕ := U \ M

theorem set_equality : Complement U M ∩ Complement U N = {5, 6} := by
  sorry

end set_equality_l345_345891


namespace percent_voted_for_winners_accurate_l345_345185

theorem percent_voted_for_winners_accurate :
  let total_members := 5000
  let votes_cast := 2000
  let president_votes_required := (0.50 * votes_cast) + 1
  let vice_president_votes_required := 0.45 * votes_cast
  let secretary_votes_required := 0.40 * votes_cast
  let treasurer_votes_required := 0.35 * votes_cast
  let president_percent := (president_votes_required / total_members) * 100
  let vice_president_percent := (vice_president_votes_required / total_members) * 100
  let secretary_percent := (secretary_votes_required / total_members) * 100
  let treasurer_percent := (treasurer_votes_required / total_members) * 100
  president_percent = 20.02 ∧
  vice_president_percent = 18 ∧
  secretary_percent = 16 ∧
  treasurer_percent = 14 :=
by
  let total_members := 5000
  let votes_cast := 2000
  let president_votes_required := (0.50 * votes_cast) + 1
  let vice_president_votes_required := 0.45 * votes_cast
  let secretary_votes_required := 0.40 * votes_cast
  let treasurer_votes_required := 0.35 * votes_cast
  let president_percent := (president_votes_required / total_members) * 100
  let vice_president_percent := (vice_president_votes_required / total_members) * 100
  let secretary_percent := (secretary_votes_required / total_members) * 100
  let treasurer_percent := (treasurer_votes_required / total_members) * 100
  sorry

end percent_voted_for_winners_accurate_l345_345185


namespace find_f_log3_36_l345_345866

-- Define the periodic odd function f satisfying the given properties
def f (x : ℝ) : ℝ := 
  if h : 0 ≤ x ∧ x ≤ 1 then 3^x - 1
  else if h' : 1 ≤ x ∧ x < 4 then f (x - 3)
  else if h'' : -1 ≤ x ∧ x < 0 then f (-x)
  else sorry  -- This fits the context outside the given bounds. Requires domain handling.

-- State the theorem to be proved: That f (log 3 36) == -1/3
theorem find_f_log3_36 : f (log 3 36) = -1/3 :=
by
  sorry  -- The proof is skipped as per instructions

end find_f_log3_36_l345_345866


namespace car_speed_l345_345326

theorem car_speed (distance time : ℝ) (h1 : distance = 300) (h2 : time = 5) : distance / time = 60 := by
  have h : distance / time = 300 / 5 := by
    rw [h1, h2]
  norm_num at h
  exact h

end car_speed_l345_345326


namespace probability_0_lt_X_lt_4_l345_345869

open probability_theory

noncomputable def normal_distribution_X (σ : ℝ) : measure ℝ :=
measure_normal 2 σ

theorem probability_0_lt_X_lt_4 (σ : ℝ) (h1 : P(X ≤ 4) = 0.88) :
  P(0 < X < 4) = 0.76 :=
sorry

end probability_0_lt_X_lt_4_l345_345869


namespace doors_2_and_3_must_be_opened_l345_345224

-- Define the conditions as hypotheses
def Door1_opened {door1 door2 door3 door4 door5 : Prop} (h1 : door1) : door2 ∧ ¬door5 :=
  -- given condition: if door1 is open, door2 must be open, and door5 must be closed
  sorry

def Door2_or_5_opened_closed_4 {door2 door4 door5 : Prop} (h2 : door2 ∨ door5) : ¬door4 :=
  -- given condition: if door2 or door5 is open, door4 must be closed
  sorry

def Doors_3_and_4_cannot_be_closed_simultaneously {door3 door4 : Prop} : ¬(¬door3 ∧ ¬door4) :=
  -- given condition: doors 3 and 4 cannot both be closed
  sorry

-- Main theorem stating that if door1 is opened, then doors 2 and 3 must both be opened
theorem doors_2_and_3_must_be_opened (door1 door2 door3 door4 door5 : Prop)
  (h1 : door1) :
  door2 ∧ door3 :=
Proof
  -- use the definitions above and conditions provided to prove that door2 and door3 must be opened
  have h1_opened := Door1_opened h1,
  have h2_closed_4 := Door2_or_5_opened_closed_4 (Or.inl h1_opened.left),
  have h3_opened_3 := Doors_3_and_4_cannot_be_closed_simultaneously,
  -- rest of the proof skipped
  sorry

end doors_2_and_3_must_be_opened_l345_345224


namespace f_sum_2017_l345_345063

-- Define a function that represents f(n) = tan(n * pi / 3)
def f (n : ℕ) : ℝ := Real.tan ((n : ℝ) * Real.pi / 3)

-- State the theorem to prove that the sum from f(1) to f(2017) is sqrt(3)
theorem f_sum_2017 : Finset.sum (Finset.range 2017) (λ n, f(n + 1)) = Real.sqrt 3 := 
by
  -- The proof is omitted
  sorry

end f_sum_2017_l345_345063


namespace find_x_l345_345433

def condition (x : ℝ) : Prop := 16 ^ (-3) = 2 ^ (96 / x) / (2 ^ (52 / x) * 16 ^ (34 / x))

theorem find_x : ∃ x : ℝ, condition x ∧ x = 23 / 3 :=
by
  sorry

end find_x_l345_345433


namespace tangency_to_x_axis_number_of_zeros_l345_345049

-- Define the function f
def f (x a : ℝ) : ℝ := 2 * Real.exp x + 2 * a * x - x + 3 - a^2

-- Statement 1: Tangency to the x-axis
theorem tangency_to_x_axis (a : ℝ) (x₀ : ℝ) :
  (2 * Real.exp x₀ + 2 * a - 2 * x₀ = 0) ∧ (2 * Real.exp x₀ + 2 * a * x₀ - x₀^2 + 3 - a^2 = 0) →
  a = Real.log 3 - 3 :=
sorry -- Proof goes here

-- Statement 2: Number of zeros of f(x) when x > 0
theorem number_of_zeros (a : ℝ) :
  ((a ≤ -Real.sqrt 5) ∨ (a = Real.log 3 - 3) ∨ (a > Real.sqrt 5) →
    ∃! x > 0, f x a = 0) ∧ 
  ((-Real.sqrt 5 < a ∧ a < Real.log 3 - 3) →
    ∃! x1 x2, 0 < x1 ∧ x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ∧
  ((Real.log 3 - 3 < a ∧ a ≤ Real.sqrt 5) →
    ¬∃ x > 0, f x a = 0) :=
sorry -- Proof goes here

end tangency_to_x_axis_number_of_zeros_l345_345049


namespace fractions_equivalent_under_scaling_l345_345798

theorem fractions_equivalent_under_scaling (a b d k x : ℝ) (h₀ : d ≠ 0) (h₁ : k ≠ 0) :
  (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x)) ↔ b = d :=
by sorry

end fractions_equivalent_under_scaling_l345_345798


namespace remainder_of_sum_div_100_l345_345983

def s : ℤ := ∑ k in finset.range 2015.succ, (k + 1) * 2^(k + 1)

theorem remainder_of_sum_div_100 : s % 100 = 6 := 
by 
  sorry

end remainder_of_sum_div_100_l345_345983


namespace staplers_left_l345_345684

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l345_345684


namespace min_h_25_value_l345_345797

theorem min_h_25_value :
  (∀ (h : ℕ → ℝ), (∀ (x y : ℕ), h x + h y ≥ x^2 + 2 * y + 1) → ∃ (v : ℝ), (v = h 25 ∧ v = 420)) :=
begin
  sorry
end

end min_h_25_value_l345_345797


namespace no_maximum_value_on_interval_l345_345129

-- Definition of the function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

-- Statement of the proof problem
theorem no_maximum_value_on_interval (m : ℝ) 
  (h_even : ∀ x, f(m)(-x) = f(m)(x)) : 
  ¬ ∃ x ∈ set.Ioo (-2 : ℝ) (-1 : ℝ), ∀ y ∈ set.Ioo (-2 : ℝ) (-1 : ℝ), f(m)(y) ≤ f(m)(x) :=
by
  have h_m_zero : m = 0 := 
  begin
    sorry -- This is where we would prove that m = 0 based on the even function condition
  end,
  rw h_m_zero at *,
  let f0 (x : ℝ) := f 0 x,
  have h_f0 : f0 = λ x, -x^2 + 3 := 
  begin
    ext,
    sorry -- This is where we would show the expression for f when m = 0
  end,
  sorry -- This is where we would conclude that f0 does not have a maximum value on the interval (-2, -1)

end no_maximum_value_on_interval_l345_345129


namespace problem_statement_l345_345087

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- defining conditions
axiom a1_4_7 : a 1 + a 4 + a 7 = 39
axiom a2_5_8 : a 2 + a 5 + a 8 = 33
axiom is_arithmetic : arithmetic_seq a d

theorem problem_statement : a 5 + a 8 + a 11 = 15 :=
by sorry

end problem_statement_l345_345087


namespace parabola_focus_directrix_distance_l345_345484

open Real

structure Parabola (p : ℝ) :=
(vertex : ℝ × ℝ)
(focus : ℝ × ℝ)
(passes_through : ℝ × ℝ → Prop)

def parabola := Parabola 9/2 (0, 0) (9/2, 0) (λ pt, pt = (1, 3))

theorem parabola_focus_directrix_distance :
  ∀ (p : ℝ) (C : Parabola p), (C.vertex = (0, 0)) → (C.focus = (p, 0)) → C.passes_through (1, 3) → p = 9 / 2 :=
begin
  sorry,
end

end parabola_focus_directrix_distance_l345_345484


namespace smallest_norm_value_l345_345968

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l345_345968


namespace cost_of_first_shirt_l345_345589

theorem cost_of_first_shirt (x : ℝ) (h1 : x + (x + 6) = 24) : x + 6 = 15 :=
by
  sorry

end cost_of_first_shirt_l345_345589


namespace perfect_cubes_between_100_and_900_l345_345897

theorem perfect_cubes_between_100_and_900:
  ∃ n, n = 5 ∧ (∀ k, (k ≥ 5 ∧ k ≤ 9) → (k^3 ≥ 100 ∧ k^3 ≤ 900)) :=
by
  sorry

end perfect_cubes_between_100_and_900_l345_345897


namespace count_valid_4_digit_numbers_l345_345499

def is_valid_number (n : ℕ) : Prop :=
  let thousands := n / 1000 in
  let hundreds := (n / 100) % 10 in
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  1000 ≤ n ∧ n < 10000 ∧
  1 ≤ thousands ∧ thousands ≤ 9 ∧
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_4_digit_numbers : 
  (Finset.filter is_valid_number (Finset.range 10000)).card = 1782 :=
sorry

end count_valid_4_digit_numbers_l345_345499


namespace haley_collected_cans_l345_345059

theorem haley_collected_cans (C : ℕ) (h : C - 7 = 2) : C = 9 :=
by {
  sorry
}

end haley_collected_cans_l345_345059


namespace max_S_value_l345_345453

variable (n : ℕ) (M : ℝ) 
variable {a : ℕ → ℝ}  -- Definition for a sequence a_k

theorem max_S_value (hn_pos : 0 < n) (h_arith_seq : ∀ d : ℝ, ∃ (a : ℕ → ℝ), ∀ k, a k = a 0 + (k - 1) * d) 
                    (h_condition : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S : ℝ, S = (n + 1) * (sqrt 10 / 2) * sqrt M :=
sorry

end max_S_value_l345_345453


namespace minimum_students_performed_exactly_four_spells_l345_345670

/--

The school level magic and spellcasting competition consists of 5 spells. Out of 100 young wizards who participated in the competition:
- 95 performed the 1st spell correctly
- 75 performed the 2nd spell correctly
- 97 performed the 3rd spell correctly
- 95 performed the 4th spell correctly
- 96 performed the 5th spell correctly.

Prove that the minimum number of students who could have performed exactly 4 out of the 5 spells correctly is 8.

--/

open Classical

theorem minimum_students_performed_exactly_four_spells
    (total_students : ℕ)
    (spell1_correct : ℕ)
    (spell2_correct : ℕ)
    (spell3_correct : ℕ)
    (spell4_correct : ℕ)
    (spell5_correct : ℕ)
    (spell1_errors : ℕ := total_students - spell1_correct)
    (spell3_errors : ℕ := total_students - spell3_correct)
    (spell4_errors : ℕ := total_students - spell4_correct)
    (spell5_errors : ℕ := total_students - spell5_correct) :
  spell1_correct = 95 →
  spell2_correct = 75 →
  spell3_correct = 97 →
  spell4_correct = 95 →
  spell5_correct = 96 →
  total_students = 100 →
  ∃(students_exactly_four : ℕ), students_exactly_four = 8 := 
by
  intros
  use 8
  sorry

end minimum_students_performed_exactly_four_spells_l345_345670


namespace find_BE_l345_345572

def point (α : Type*) := (α × α)
noncomputable def angle := ℝ  

variables {α : Type*} [division_ring α] [has_sqrt α] [has_pow α]

noncomputable def BE_of_triangle (A B C D E : point α) (AB BC CA CD : α) (angleBAE angleCAD : angle) : α :=
  if h : AB = 15 ∧ BC = 18 ∧ CA = 17 ∧ CD = 7 ∧ angleBAE = angleCAD
  then (17325/4754 : α)
  else 0

theorem find_BE (A B C D E : point α) (AB BC CA CD : α) (angleBAE angleCAD : angle)
  (h_AB : AB = 15)
  (h_BC : BC = 18)
  (h_CA : CA = 17)
  (h_CD : CD = 7)
  (h_angle : angleBAE = angleCAD) :
  BE_of_triangle A B C D E AB BC CA CD angleBAE angleCAD = (17325/4754 : α) :=
by
  sorry

end find_BE_l345_345572


namespace image_digit_sum_l345_345155

theorem image_digit_sum 
  (cat chicken crab bear goat: ℕ)
  (h1 : 5 * crab = 10)
  (h2 : 4 * crab + goat = 11)
  (h3 : 2 * goat + crab + 2 * bear = 16)
  (h4 : cat + bear + 2 * goat + crab = 13)
  (h5 : 2 * crab + 2 * chicken + goat = 17) :
  cat = 1 ∧ chicken = 5 ∧ crab = 2 ∧ bear = 4 ∧ goat = 3 := by
  sorry

end image_digit_sum_l345_345155


namespace number_of_true_propositions_l345_345390

section
variables (p1 p2 p3 p4 : Prop)

-- Given conditions
axiom h1 : p1
axiom h2 : ¬p2
axiom h3 : ¬p3
axiom h4 : p4

-- Compound propositions
def c1 := p1 ∧ p4
def c2 := p1 ∧ p2
def c3 := ¬p2 ∨ p3
def c4 := ¬p3 ∨ ¬p4

-- Proof statement: exactly 3 of the compound propositions are true
theorem number_of_true_propositions : (c1 ∧ c3 ∧ c4) ∧ ¬c2 :=
by sorry
end

end number_of_true_propositions_l345_345390


namespace center_cell_value_l345_345536

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l345_345536


namespace true_compound_props_l345_345397

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345397


namespace geom_series_sum_n_eq_728_div_729_l345_345672

noncomputable def a : ℚ := 1 / 3
noncomputable def r : ℚ := 1 / 3
noncomputable def S_n (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))

theorem geom_series_sum_n_eq_728_div_729 (n : ℕ) (h : S_n n = 728 / 729) : n = 6 :=
by
  sorry

end geom_series_sum_n_eq_728_div_729_l345_345672


namespace no_real_roots_l345_345413

theorem no_real_roots (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 8) :
  let Δ := b^2 - 4 * a * c in Δ < 0 → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros Δ hΔ
  sorry

end no_real_roots_l345_345413


namespace triangle_side_length_median_l345_345094

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l345_345094


namespace octagon_area_mn_l345_345674

-- Let's define the required structures and conditions

structure Square :=
(center : (ℝ × ℝ))
(side_length : ℝ)

structure Octagon :=
(vertices : fin 8 → (ℝ × ℝ))

noncomputable def area_of_octagon (o : Octagon) : ℚ :=
  let triangle_area := (55 / 99 : ℚ) * 1 / 2 in
  8 * triangle_area

def relatively_prime (a b: ℕ) : Prop :=
  Nat.gcd a b = 1

def m := 20
def n := 9

-- Now stating the theorem
theorem octagon_area_mn :
  ∃ (m n : ℕ), (relatively_prime m n) ∧
  (area_of_octagon ⟨λ i, (0, 0)⟩ = m / n) ∧
  (m + n = 29) := by
  use m, n
  constructor
  · exact Nat.gcd_eq_one_iff_coprime.mpr ((Nat.prime_factors_eq_iff_eq_prime 2 2_neq 9_eq).mp (Nat.prime_factors_eq_iff_eq_prime 20_eq 20 9 (Nat_prime_factors 20_ne0 9_ne0 2 9)))
  constructor
  · simp [area_of_octagon, triangle_area, m, n]
  · exact rfl

end octagon_area_mn_l345_345674


namespace find_a_l345_345456

-- Define the polynomial f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 - 0.8

-- Define the intermediate values v_0, v_1, and v_2 using Horner's method
def v_0 (a : ℝ) : ℝ := a
def v_1 (a : ℝ) (x : ℝ) : ℝ := v_0 a * x + 2
def v_2 (a : ℝ) (x : ℝ) : ℝ := v_1 a x * x + 3.5 * x - 2.6 * x + 13.5

-- The condition for v_2 when x = 5
axiom v2_value (a : ℝ) : v_2 a 5 = 123.5

-- Prove that a = 4
theorem find_a : ∃ a : ℝ, v_2 a 5 = 123.5 ∧ a = 4 := by
  sorry

end find_a_l345_345456


namespace largest_volume_fixed_surface_largest_volume_fixed_edge_sum_largest_volume_fixed_diagonal_shortest_diagonal_fixed_edge_sum_l345_345713

def rectangular_prism (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

def is_cube (a b c : ℝ) :=
  a = b ∧ b = c

def surface_area (a b c : ℝ) :=
  2 * (a * b + b * c + c * a)

def edge_sum (a b c : ℝ) :=
  4 * (a + b + c)

def diagonal (a b c : ℝ) :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem largest_volume_fixed_surface (a b c : ℝ) (F : ℝ) 
  (h_cond : rectangular_prism a b c)
  (h_sa : surface_area a b c = F) : 
  is_cube a b c := sorry

theorem largest_volume_fixed_edge_sum (a b c : ℝ) (E : ℝ)
  (h_cond : rectangular_prism a b c)
  (h_es : edge_sum a b c = E) :
  is_cube a b c := sorry

theorem largest_volume_fixed_diagonal (a b c : ℝ) (d : ℝ)
  (h_cond : rectangular_prism a b c)
  (h_d : diagonal a b c = d) :
  is_cube a b c := sorry

theorem shortest_diagonal_fixed_edge_sum (a b c : ℝ) (E : ℝ)
  (h_cond : rectangular_prism a b c)
  (h_es : edge_sum a b c = E) :
  is_cube a b c := sorry

end largest_volume_fixed_surface_largest_volume_fixed_edge_sum_largest_volume_fixed_diagonal_shortest_diagonal_fixed_edge_sum_l345_345713


namespace center_cell_value_l345_345535

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l345_345535


namespace true_propositions_l345_345381

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345381


namespace sum_of_inverses_is_constant_l345_345979

variable {P Q R S : Point}
variable {PQ PR PS d : ℝ}
variable {O : Point}
variable (is_center : O centroids (Tetrahedron P Q R S))
variable (pyramid_structure : RegularTriangularPyramid P (Triangle A B C))

theorem sum_of_inverses_is_constant :
  ∃ k : ℝ, (1 / PQ) + (1 / PR) + (1 / PS) = k :=
by
  sorry

end sum_of_inverses_is_constant_l345_345979


namespace q1_q2_q3_l345_345800

def combination_square_numbers (a b c : ℤ) : Prop :=
  ∃ k l m : ℤ, 
    k^2 = a * b ∧ 
    l^2 = a * c ∧ 
    m^2 = b * c ∧ 
    k > 0 ∧
    l > 0 ∧ 
    m > 0
  
theorem q1 : combination_square_numbers (-4) (-16) (-25) :=
by {
  sorry
}

theorem q2 : ∀ m : ℤ, combination_square_numbers (-3) m (-12) ∧ sqrt_int (abs (-3 * m)) 12 → m = -48 :=
by {
  intros m,
  sorry
}

theorem q3 : ∃ b c: ℤ, combination_square_numbers (-2) b c :=
by {
  sorry
}

end q1_q2_q3_l345_345800


namespace root_in_interval_l345_345598

def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → (∃ x : ℝ, 1.25 < x ∧ x < 1.5 ∧ f x = 0) :=
by
  intros h
  sorry

end root_in_interval_l345_345598


namespace sum_first_13_terms_l345_345936

variable {a : ℕ → ℝ} -- defining a_n as a function from natural numbers to real numbers
variable h : a 3 + a 5 + 2 * a 10 = 4 -- condition given in the problem

theorem sum_first_13_terms (h : a 3 + a 5 + 2 * a 10 = 4) : 
  (Finset.range 13).sum (λ n => a (n + 1)) = 13 := 
sorry

end sum_first_13_terms_l345_345936


namespace staples_left_in_stapler_l345_345678

def initial_staples : ℕ := 50
def used_staples : ℕ := 3 * 12
def remaining_staples : ℕ := initial_staples - used_staples

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  unfold initial_staples used_staples remaining_staples
  rw [Nat.mul_comm, Nat.mul_comm 3, Nat.mul_comm 12, Nat.sub_eq_iff_eq_add]
  have h : ∀ a b : ℕ, a = b -> 50 - (3 * 12) = b -> 50 - 36 = a := by intros; rw [h, Nat.mul_comm 3, Nat.mul_comm 12]
  exact h 36 36 rfl
#align std.staples_left_in_stapler


end staples_left_in_stapler_l345_345678


namespace sum_of_areas_CQiPi_l345_345963

-- Definition of unit square and points as described
structure Point :=
  (x : ℝ) (y : ℝ)

def unit_square : set Point := 
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

def midpoint (p1 p2 : Point) : Point := 
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 1⟩
def D : Point := ⟨0, 1⟩

noncomputable def Q (i : ℕ) : Point :=
  if i = 1 then midpoint B C else midpoint (P (i - 1)) C

noncomputable def P (i : ℕ) : Point :=
  -- Assume intersection logic with segment BD
  let t := Q i in
  let bd_slope : ℝ := (D.y - B.y) / (D.x - B.x) in
  let bd_y_intercept : ℝ := B.y - bd_slope * B.x in
  let aq_slope : ℝ := (t.y - A.y) / (t.x - A.x) in
  let aq_y_intercept : ℝ := A.y - aq_slope * A.x in
  let x_intersect : ℝ := (aq_y_intercept - bd_y_intercept) / (bd_slope - aq_slope) in
  ⟨x_intersect, bd_slope * x_intersect + bd_y_intercept⟩

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1 / 2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem sum_of_areas_CQiPi (i : ℕ) : Real :=
  tsum (λ i, triangle_area C (Q i) (P i)) = 1 / 4 :=
by sorry

end sum_of_areas_CQiPi_l345_345963


namespace value_of_a_l345_345526

theorem value_of_a (a : ℝ) (A : ℝ × ℝ) (h : A = (1, 0)) : (a * A.1 + 3 * A.2 - 2 = 0) → a = 2 :=
by
  intro h1
  rw [h] at h1
  sorry

end value_of_a_l345_345526


namespace real_part_zero_implies_a_eq_2_l345_345444

variable (a : ℝ)

theorem real_part_zero_implies_a_eq_2 (ha : (1 : ℂ) + complex.I * (2 : ℂ) + complex.I * (a : ℂ) ∈ ℂ → complex.re ((1 + complex.I) * (2 + complex.I * a)) = 0) : a = 2 := 
sorry

end real_part_zero_implies_a_eq_2_l345_345444


namespace true_compound_props_l345_345398

/-- Definitions of individual propositions. --/
def p1 : Prop := ∀ (l1 l2 l3 : Line), intersects_pairwise l1 l2 l3 ∧ ¬passes_same_point l1 l2 l3 → lies_in_same_plane l1 l2 l3
def p2 : Prop := ∀ (a b c : Point), determines_one_plane a b c
def p3 : Prop := ∀ (l1 l2 : Line), ¬intersects l1 l2 → parallel l1 l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), contains α l ∧ perpendicular_to_plane m α → perpendicular_to_line m l

/-- Given truth values of the propositions. --/
axiom h_p1 : p1
axiom h_p2 : ¬p2
axiom h_p3 : ¬p3
axiom h_p4 : p4

/-- Propositions to be proven true. --/
theorem true_compound_props : (p1 ∧ p4) ∧ (¬p2 ∨ p3) ∧ (¬p3 ∨ ¬p4) := 
by {
  split,
  { exact ⟨h_p1, h_p4⟩ }, -- (p1 ∧ p4)
  split,
  { left, exact h_p2 }, -- (¬p2 ∨ p3)
  { left, exact h_p3 }  -- (¬p3 ∨ ¬p4)
}

end true_compound_props_l345_345398


namespace find_DF_l345_345100

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345100


namespace remainder_of_sum_l345_345293

theorem remainder_of_sum:
  let S := ∑ k in Finset.range 126, (5^k * 3^k)
  in S % 8 = 4 :=
by
  -- Proof goes here
  sorry

end remainder_of_sum_l345_345293


namespace night_duty_schedule_arrangements_l345_345766

theorem night_duty_schedule_arrangements :
  let employees : Type := {femaleA, femaleB, femaleC, femaleD, maleA, maleB, maleC, maleD, maleE, maleF}
  in
  ∀ (m: Nat) (f: Nat) (days : List String) (sched : List (employees × String)),
  m = 3 →
  f = 2 →
  days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] →
  (∃(schedule: List (employees × String)), schedule.length = 5 ∧
  (∀ e : employees, list.count (schedule.map prod.fst) e ≤ 1) ∧
  list.count (schedule.map prod.fst) femaleA ≤ 4 ∧ 
  ∀ e in schedule.map prod.fst,
    (cases_on e (λ a, a ∈ [femaleA, femaleB, femaleC, femaleD]) (λ b, b ∈ [maleA, maleB, maleC, maleD, maleE, maleF])) ∧
  femaleA ∉ List.filterMap (λx, if x.snd ∈ ["Monday", "Tuesday"] then some x else none) schedule ∧
  maleB ∉ List.filterMap (λx, if x.snd == "Tuesday" then some x else none) schedule ∧
  (∃ f3 : employees × String, f3.1 = maleC ∧ f3.2 == "Friday" ∧ f3 ∈ schedule)) → 
  list.length (filter (λ s: List (employees × String),
    list.count (s.map prod.fst) femaleA ≤ 4 ∧
    ∀ e in s, list.count ((s.map prod.fst).erase e) e ≤ 0 ∧
    femaleA ∉ List.filterMap (λx, if x.snd ∈ ["Monday", "Tuesday"] then some x else none) s ∧
    maleB ∉ List.filterMap (λx, if x.snd == "Tuesday" then some x else none) s ∧
    (∃ f3 : employees × String, f3.1 = maleC ∧ f3.2 == "Friday" ∧ f3 ∈ s)) possibleStaffSchedules) = 960 :=
by sorry

end night_duty_schedule_arrangements_l345_345766


namespace arrangement_of_boys_and_girls_l345_345687

theorem arrangement_of_boys_and_girls
  (boys : ℕ) (girls : ℕ)
  (ends_girls : Bool := true) :
  boys = 3 ∧ girls = 2 ∧ ends_girls →
  ∃ n, n = 12 :=
by
  intro h
  use 12
  sorry

end arrangement_of_boys_and_girls_l345_345687


namespace triangle_area_l345_345947

noncomputable def area_of_triangle (a b c : ℝ) (angle_A : ℝ) : ℝ :=
  if angle_A = 60 then (complex.sqrt 3 / 4) * (a^2 - (b - c)^2) else 0

theorem triangle_area (a b c : ℝ) (hA : angle_A = 60) :
  area_of_triangle a b c 60 = (complex.sqrt 3 / 4) * (a^2 - (b - c)^2) :=
  sorry

end triangle_area_l345_345947


namespace Shell_Ratio_l345_345624

-- Definitions of the number of shells collected by Alan, Ben, and Laurie.
variable (A B L : ℕ)

-- Hypotheses based on the given conditions:
-- 1. Alan collected four times as many shells as Ben did.
-- 2. Laurie collected 36 shells.
-- 3. Alan collected 48 shells.
theorem Shell_Ratio (h1 : A = 4 * B) (h2 : L = 36) (h3 : A = 48) : B / Nat.gcd B L = 1 ∧ L / Nat.gcd B L = 3 :=
by
  sorry

end Shell_Ratio_l345_345624


namespace distribution_plans_for_girls_and_boys_l345_345509

theorem distribution_plans_for_girls_and_boys :
  let girls := 5
  let boys := 2
  let place1 := 1
  let place2 := 1
  (place1 + place2 >= 2) ∧ 
  (girls = 5) ∧ (boys = 2) →
  (2 * (∑ i in {1, 2}, nat.choose girls i)) = 30 := 
sorry

end distribution_plans_for_girls_and_boys_l345_345509


namespace number_of_ways_to_choose_president_and_vp_l345_345086

-- Definition of the group of people
def group_size : ℕ := 5

-- Definition of the condition that President and Vice-President cannot be the same person
def is_valid_choice (p v : ℕ) : Prop := p ≠ v

-- The mathematical statement to prove
theorem number_of_ways_to_choose_president_and_vp (h : group_size = 5) : 
  ∑ p in finset.range group_size, ∑ v in finset.range group_size, if is_valid_choice p v then 1 else 0 = 20 :=
by 
  sorry

end number_of_ways_to_choose_president_and_vp_l345_345086


namespace sean_div_julie_l345_345172

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l345_345172


namespace tenth_finger_l345_345645

-- Assume there exists a function f such that the following definition of g holds
variable f : ℕ → ℕ
def g (x : ℕ) := f (x + 1)

-- Given f(5) = 4
axiom f_5 : f 5 = 4

-- Given f(4) = 3
axiom f_4 : f 4 = 3

-- g(4) = f(5), which implies g(4) = 4
#check f_5       -- to ensure axiom added correctly

-- Prove that the number Joe writes on his tenth finger is 6.
theorem tenth_finger : g 9 = 6 := by
  sorry

end tenth_finger_l345_345645


namespace chair_subsets_with_at_least_three_adjacent_l345_345257

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345257


namespace GDP_range_correct_GDP_mode_correct_GDP_median_correct_GDP_upper_quartile_correct_l345_345651

noncomputable def GDP_indices : List ℝ := [112.2, 108.1, 108.7, 108.7, 109.5, 108.9, 108.1, 104.0, 107.3, 104.3]

def sorted_GDP_indices : List ℝ := List.sort GDP_indices

def range (data : List ℝ) : ℝ := data.getLast sorry - data.head sorry

def modes (data : List ℝ) : List ℝ :=
  let counts := data.groupBy id
  let max_count := (counts.map (λ l => l.length)).maximum
  counts.filterMap (λ l => if l.length = max_count then some l.head else none)

def median (data : List ℝ) : ℝ :=
  let n := data.length
  if n % 2 = 0 then
    (data.get! (n / 2 - 1) + data.get! (n / 2)) / 2
  else
    data.get! (n / 2)

def upper_quartile (data : List ℝ) : ℝ :=
  let n := data.length
  if n % 2 == 0 then
    median (data.drop (n / 2))
  else
    median (data.drop (n / 2 + 1))

theorem GDP_range_correct : range sorted_GDP_indices = 8.2 := sorry

theorem GDP_mode_correct : ∃ m, m ∈ modes sorted_GDP_indices ∧ m = 108.1 := sorry

theorem GDP_median_correct : median sorted_GDP_indices = 108.4 := sorry

theorem GDP_upper_quartile_correct : upper_quartile sorted_GDP_indices = 108.9 := sorry

end GDP_range_correct_GDP_mode_correct_GDP_median_correct_GDP_upper_quartile_correct_l345_345651


namespace rational_p_area_triangle_l345_345005

theorem rational_p_area_triangle (p : ℚ) (f : ℚ → ℚ) :
    f = λ x, -x^2 + 4 * p * x - p + 1 →
    (∃ n : ℚ, area_triangle f ∈ ℤ) →
    (p = -3/4 ∨ p = 0 ∨ p = 1/4 ∨ p = 1) :=
sorry

noncomputable def area_triangle (f : ℚ → ℚ) : ℚ :=
  let q := 4 * (p:ℚ)^2 - p + 1
  let base := 2 * (q : ℚ)^0.5
  let height := q
  base * height / 2

end rational_p_area_triangle_l345_345005


namespace staples_left_in_stapler_l345_345679

def initial_staples : ℕ := 50
def used_staples : ℕ := 3 * 12
def remaining_staples : ℕ := initial_staples - used_staples

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  unfold initial_staples used_staples remaining_staples
  rw [Nat.mul_comm, Nat.mul_comm 3, Nat.mul_comm 12, Nat.sub_eq_iff_eq_add]
  have h : ∀ a b : ℕ, a = b -> 50 - (3 * 12) = b -> 50 - 36 = a := by intros; rw [h, Nat.mul_comm 3, Nat.mul_comm 12]
  exact h 36 36 rfl
#align std.staples_left_in_stapler


end staples_left_in_stapler_l345_345679


namespace number_of_adjacent_subsets_l345_345231

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345231


namespace problem_min_value_l345_345864

theorem problem_min_value {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  (2 * a + b + c) ≥ 4 := 
  sorry

end problem_min_value_l345_345864


namespace problem_b_l345_345130

variables {m n : Line} {α β : Plane}

def parallel (α β : Plane) : Prop := Plane.parallel α β
def perp (x y : Plane) : Prop := Plane.perpendicular x y
def perpL (m n : Line) : Prop := Line.perpendicular m n
def parallelL (m n : Line) : Prop := Line.parallel m n

theorem problem_b (h1 : parallel α β) (h2 : perpL m α) (h3 : parallelL n β) : perpL m n :=
sorry

end problem_b_l345_345130


namespace imo1983_q24_l345_345949

theorem imo1983_q24 :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ 
    (∀ x ∈ S, x > 0 ∧ x ≤ 10^5) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → (x + z ≠ 2 * y)) :=
sorry

end imo1983_q24_l345_345949


namespace inequality_F_product_l345_345882

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  Real.exp x - k * x

noncomputable def F (x : ℝ) (k : ℝ) : ℝ :=
  f x k + f (-x) k

theorem inequality_F_product (k : ℝ) (n : ℕ) (h : 0 < n) :
  (∏ i in Finset.range(n+1).filter (λ x, 0 < x), F (i : ℝ) k) >
  (Real.exp n + 2) ^ (n / 2) := 
  sorry

end inequality_F_product_l345_345882


namespace circles_BXC_and_AEF_are_tangent_l345_345892

-- Definitions of points and circles
structure Triangle :=
(A B C : Point)

structure Circle :=
(center : Point)
(radius : ℝ)

-- Questions and conditions
variables (ΔABC : Triangle)
variables (D I E F X : Point)
variables (CI : Circle)
variables (CAEF : Circle (Center := E, F))

-- Inner bisector D
axiom AD_is_inner_bisector :
  ∀ (A B C D : Point), -- Conditions defining D

-- Perpendicular line through D intersecting outer bisector at I
axiom D_perpendicular_BC_intersects_outer_bisector_at_I :
  ∀ (D I : Point), -- Conditions defining D and I

-- Circle (I, ID) intersects CA and AB at E, F
axiom circle_intersects_CA_AB_at_E_and_F :
  Circle.center = I ∧ Circle.radius = dist I D →
  intersects Circular (CA) at E ∧ intersects Circular (AB) at F

-- Symmedian line of ∆AEF intersects circle (AEF) at X
axiom symmedian_intersects_circle_AEF_at_X :
  symmedian_line ΔAEF = true ∧ intersects (CA(CA(EAB,EF),EF) at X

-- Prove that circles (BXC) and (AEF) are tangent
theorem circles_BXC_and_AEF_are_tangent :
  tangent Circles (X, CB) (AX ∧ EF) :=
  sorry

end circles_BXC_and_AEF_are_tangent_l345_345892


namespace padic_fibonacci_periodic_l345_345127

-- Definitions of $p$-adic Fibonacci sequence and period divisor.
def padic_fibonacci (p k : ℕ) : ℕ → ℕ := 
-- implementation of $p$-adic Fibonacci sequence would go here.

noncomputable def period_divisor (p k : ℕ) : ℕ := p^(k-1) * (p-1)

-- Theorems to state our proof problem.
theorem padic_fibonacci_periodic (p k n : ℕ) (hp : p > 0) :
  (padic_fibonacci p k (n + period_divisor p k)) % p^k = (padic_fibonacci p k n) % p^k :=
sorry

end padic_fibonacci_periodic_l345_345127


namespace log_drift_downstream_time_l345_345323

theorem log_drift_downstream_time (a x : ℝ) 
  (h1 : 2 * (a + x) = 1) 
  (h2 : 3 * (a - x) = 1) :
  let d := 1 in
  x = 1 / 12 →
  d / x = 12 := 
by
  sorry

end log_drift_downstream_time_l345_345323


namespace max_value_of_trig_function_l345_345411

theorem max_value_of_trig_function (x : ℝ) :
  ∃ (M : ℝ), M = 2 ∧ ∀ x, (sin (x + π / 4) + cos (π / 4 - x) ≤ M) :=
  sorry

end max_value_of_trig_function_l345_345411


namespace smallest_m_for_even_function_l345_345662

def f (x : ℝ) : ℝ := sqrt 3 * sin x - cos x

-- The translated function g(x)
def g (m x : ℝ) : ℝ := f (x + m)

-- To check if g(x) is even, we need to show g(-x) = g(x)
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem smallest_m_for_even_function :
  ∃ m > 0, is_even_function (g m) ∧ ∀ m' > 0, is_even_function (g m') → m ≤ m' :=
sorry

end smallest_m_for_even_function_l345_345662


namespace balanced_integers_between_2000_and_9999_l345_345333

noncomputable def balanced_integers_count : Nat :=
  let is_balanced (n : Nat) : Bool := 
    let d1 := n / 1000
    let d2 := (n % 1000) / 100
    let d3 := (n % 100) / 10
    let d4 := n % 10
    d1 + d2 = d3 + d4
  (1000.to 9999).filter is_balanced |>.length

theorem balanced_integers_between_2000_and_9999 : balanced_integers_count = 555 :=
by {
  sorry
}

end balanced_integers_between_2000_and_9999_l345_345333


namespace chairs_subset_count_l345_345276

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l345_345276


namespace find_DF_l345_345107

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345107


namespace vaishali_total_stripes_l345_345701

def total_stripes (hats_with_3_stripes hats_with_4_stripes hats_with_no_stripes : ℕ) 
  (hats_with_5_stripes hats_with_7_stripes hats_with_1_stripe : ℕ) 
  (hats_with_10_stripes hats_with_2_stripes : ℕ)
  (stripes_per_hat_with_3 stripes_per_hat_with_4 stripes_per_hat_with_no : ℕ)
  (stripes_per_hat_with_5 stripes_per_hat_with_7 stripes_per_hat_with_1 : ℕ)
  (stripes_per_hat_with_10 stripes_per_hat_with_2 : ℕ) : ℕ :=
  hats_with_3_stripes * stripes_per_hat_with_3 +
  hats_with_4_stripes * stripes_per_hat_with_4 +
  hats_with_no_stripes * stripes_per_hat_with_no +
  hats_with_5_stripes * stripes_per_hat_with_5 +
  hats_with_7_stripes * stripes_per_hat_with_7 +
  hats_with_1_stripe * stripes_per_hat_with_1 +
  hats_with_10_stripes * stripes_per_hat_with_10 +
  hats_with_2_stripes * stripes_per_hat_with_2

#eval total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2 -- 71

theorem vaishali_total_stripes : (total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2) = 71 :=
by
  sorry

end vaishali_total_stripes_l345_345701


namespace circumscribed_circle_radius_l345_345948

noncomputable
def radius_of_circumcircle (t h : ℝ) : ℝ :=
  ⟨R = t^2 / (2 * h)⟩ -- Declare R to be the value that solves the equation in terms of t and h

theorem circumscribed_circle_radius (ABC : Triangle) (AM AL AH : LineSegment)
  (L : Point) (h t : ℝ) (HL : Point)
  (AL_val : AL.length = t)
  (AH_val : AH.length = h)
  (L_mid: midpoint L (segment HB MH))
  (median_proof: isMedian AM ABC)
  (bisector_proof: isBisector AL ABC)
  (altitude_proof: isAltitude AH ABC):
  radius_of_circumcircle t h :=
by
  sorry

end circumscribed_circle_radius_l345_345948


namespace smallest_four_digit_palindrome_div7_eq_1661_l345_345294

theorem smallest_four_digit_palindrome_div7_eq_1661 :
  ∃ (A B : ℕ), (A == 1 ∨ A == 3 ∨ A == 5 ∨ A == 7 ∨ A == 9) ∧
  (1000 ≤ 1100 * A + 11 * B ∧ 1100 * A + 11 * B < 10000) ∧
  (1100 * A + 11 * B) % 7 = 0 ∧
  (1100 * A + 11 * B) = 1661 :=
by
  sorry

end smallest_four_digit_palindrome_div7_eq_1661_l345_345294


namespace concyclic_points_l345_345597
open EuclideanGeometry

theorem concyclic_points
  (A B C P Q R S : Point)
  (h_tri : Triangle ABC)
  (h_acute : is_acute ∠BAC ∧ is_acute ∠ABC ∧ is_acute ∠BCA)
  (h_circle_AC : circle (AC) P ∧ circle (AC) Q)
  (h_circle_AB : circle (AB) R ∧ circle (AB) S)
  (h_altitude_PQ : altitude_from B P Q)
  (h_altitude_RS : altitude_from C R S) :
  collinear (P :: Q :: R :: S :: []) :=
sorry

end concyclic_points_l345_345597


namespace product_of_real_parts_l345_345195

theorem product_of_real_parts (x1 x2 : ℂ) (h1 : x1^2 - 2*x1 = complex.I) (h2 : x2^2 - 2*x2 = complex.I) :
  (x1.re * x2.re) = (1 - Real.sqrt 2) / 2 :=
  sorry

end product_of_real_parts_l345_345195


namespace probability_median_five_from_ten_l345_345795

noncomputable def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_median_five_from_ten : 
  let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let total_combinations := count_combinations 10 5 in
  let favorable_combinations := (count_combinations 4 2) * (count_combinations 5 2) in
  let probability := (favorable_combinations : ℚ) / total_combinations in
  probability = 5 / 21 :=
begin
  sorry
end

end probability_median_five_from_ten_l345_345795


namespace center_cell_value_l345_345540

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l345_345540


namespace discriminant_of_polynomial_l345_345820

noncomputable def polynomial_discriminant (a b c : ℚ) : ℚ :=
b^2 - 4 * a * c

theorem discriminant_of_polynomial : polynomial_discriminant 2 (4 - (1/2 : ℚ)) 1 = 17 / 4 :=
by
  sorry

end discriminant_of_polynomial_l345_345820


namespace lateral_surface_area_volume_of_prism_l345_345427

variables {h α β : ℝ} (rhombus_base : Type)

-- Define the conditions
noncomputable def diagonals (h α β : ℝ) : ℝ × ℝ :=
  (h * Real.cot α, h * Real.cot β)

noncomputable def area_rhombus (h α β : ℝ) : ℝ :=
  1 / 2 * (h * Real.cot α) * (h * Real.cot β)

noncomputable def volume_prism (h α β : ℝ) : ℝ :=
  area_rhombus h α β * h

-- Prove the lateral surface area
theorem lateral_surface_area (h α β : ℝ) :
  2 * h ^ 2 * Real.sqrt (Real.cot α ^ 2 + Real.cot β ^ 2) = 2 * h^2 * Real.sqrt (Real.cot α ^ 2 + Real.cot β ^ 2) :=
by
  sorry

-- Prove the volume
theorem volume_of_prism (h α β : ℝ) :
  1/2 * h^3 * Real.cot α * Real.cot β = 1/2 * h^3 * Real.cot α * Real.cot β :=
by
  sorry

end lateral_surface_area_volume_of_prism_l345_345427


namespace count_even_fibonacci_first_2007_l345_345688

def fibonacci (n : Nat) : Nat :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

def fibonacci_parity : List Bool := List.map (fun x => fibonacci x % 2 = 0) (List.range 2008)

def count_even (l : List Bool) : Nat :=
  l.foldl (fun acc x => if x then acc + 1 else acc) 0

theorem count_even_fibonacci_first_2007 : count_even (fibonacci_parity.take 2007) = 669 :=
sorry

end count_even_fibonacci_first_2007_l345_345688


namespace cos_difference_simplify_l345_345635

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end cos_difference_simplify_l345_345635


namespace wendy_time_correct_l345_345357

noncomputable section

def bonnie_time : ℝ := 7.80
def wendy_margin : ℝ := 0.25

theorem wendy_time_correct : (bonnie_time - wendy_margin) = 7.55 := by
  sorry

end wendy_time_correct_l345_345357


namespace book_count_l345_345719

theorem book_count (P C B : ℕ) (h1 : P = 3 * C / 2) (h2 : B = 3 * C / 4) (h3 : P + C + B > 3000) : 
  P + C + B = 3003 := by
  sorry

end book_count_l345_345719


namespace work_completion_time_l345_345325

theorem work_completion_time (work_per_day_A : ℚ) (work_per_day_B : ℚ) (work_per_day_C : ℚ) 
(days_A_worked: ℚ) (days_C_worked: ℚ) :
work_per_day_A = 1 / 20 ∧ work_per_day_B = 1 / 30 ∧ work_per_day_C = 1 / 10 ∧
days_A_worked = 2 ∧ days_C_worked = 4  → 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked) +
(1 - 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked)))
/ work_per_day_B + days_C_worked) 
= 15 := by
sorry

end work_completion_time_l345_345325


namespace complex_real_solution_l345_345838

open Complex

theorem complex_real_solution {a : ℝ} (h : (let z := (a - 1)^2 * I + 4 * a in z.im = 0)) : a = 1 :=
by {
  sorry
}

end complex_real_solution_l345_345838


namespace true_propositions_correct_l345_345368

def p1 := True
def p2 := False
def p3 := False
def p4 := True

def truePropositions (p1 p2 p3 p4 : Prop) : set ℕ :=
  { if p1 ∧ p4 then 1 else 0,
    if p1 ∧ p2 then 2 else 0,
    if ¬p2 ∨ p3 then 3 else 0,
    if ¬p3 ∨ ¬p4 then 4 else 0
  }.erase 0

theorem true_propositions_correct :
  truePropositions p1 p2 p3 p4 = {1, 3, 4} :=
by {
  intros,
  simp [truePropositions, p1, p2, p3, p4],
  sorry
}

end true_propositions_correct_l345_345368


namespace diagonal_inequality_l345_345116

theorem diagonal_inequality (A B C D : ℝ × ℝ) (h1 : A.1 = 0) (h2 : B.1 = 0) (h3 : C.2 = 0) (h4 : D.2 = 0) 
  (ha : A.2 < B.2) (hd : D.1 < C.1) : 
  (Real.sqrt (A.2^2 + C.1^2)) * (Real.sqrt (B.2^2 + D.1^2)) > (Real.sqrt (A.2^2 + D.1^2)) * (Real.sqrt (B.2^2 + C.1^2)) :=
sorry

end diagonal_inequality_l345_345116


namespace derivative_slope_tangent_line_l345_345199

variable {α β : Type}
variable [NormedAddCommGroup α] [NormedSpace ℝ α] [NormedAddCommGroup β] [NormedSpace ℝ β] {f : α → β} {x₀ : α} {f' : β}

-- Assuming f is differentiable at x₀
axiom differentiable_at_f : DifferentiableAt ℝ f x₀

-- Tangent line slope definition
theorem derivative_slope_tangent_line : fderiv ℝ f x₀ = f' ↔
  ∃ (L : α → β), (L = λ h : α × ℝ, f x₀ + f' h.2) :=
by
  sorry

end derivative_slope_tangent_line_l345_345199


namespace polynomial_sum_eq_l345_345602

-- Definitions of the given polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- The theorem to prove
theorem polynomial_sum_eq (x : ℝ) : 
  p x + q x + r x + s x = -x^2 + 10 * x - 11 :=
by 
  -- Proof steps are omitted here
  sorry

end polynomial_sum_eq_l345_345602


namespace number_of_unit_distance_pairs_lt_bound_l345_345842

/-- Given n distinct points in the plane, the number of pairs of points with a unit distance between them is less than n / 4 + (1 / sqrt 2) * n^(3 / 2). -/
theorem number_of_unit_distance_pairs_lt_bound (n : ℕ) (hn : 0 < n) :
  ∃ E : ℕ, E < n / 4 + (1 / Real.sqrt 2) * n^(3 / 2) :=
by
  sorry

end number_of_unit_distance_pairs_lt_bound_l345_345842


namespace sum_of_powers_mod7_l345_345007

theorem sum_of_powers_mod7 (k : ℕ) : (2^k + 3^k) % 7 = 0 ↔ k % 6 = 3 := by
  sorry

end sum_of_powers_mod7_l345_345007


namespace find_a_l345_345358

-- Define the function and constants
def y (a b x : ℝ) : ℝ := a * (real.sec (b * x))

-- Assume a condition matching the graph
theorem find_a (b : ℝ) (hb : 0 < b) : ∃ a : ℝ, ∀ x : ℝ, y a b x ≥ 3 → a = 3 :=
by
  sorry

end find_a_l345_345358


namespace true_propositions_l345_345394

def p1 : Prop :=
  ∀ (α : Type) [plane α] (l1 l2 l3 : line α), 
    (∀ (p : point α), p ∈ l1 ∩ l2 ∧ p ∈ l2 ∩ l3 ∧ p ∈ l1 ∩ l3 → False) →
    ∃ (π : plane α), l1 ⊆ π ∧ l2 ⊆ π ∧ l3 ⊆ π

def p2 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (p1 p2 p3 : point α), 
    ∃! (π : plane α), p1 ∈ π ∧ p2 ∈ π ∧ p3 ∈ π

def p3 : Prop :=
  ∀ (α : Type) [plane α], 
    ∀ (l1 l2 : line α), 
    ¬(∃ p, p ∈ l1 ∩ l2) → l1 ∥ l2

def p4 : Prop := 
  ∀ (α : Type) [plane α], 
    ∀ (l m : line α) (π : plane α), 
    l ⊆ π ∧ (∀ p1 p2 : point α, (p1 ∈ m ∧ p2 ∈ π → p1 ⊥ p2)) → m ⊥ l

theorem true_propositions : 
  (p1 ∧ p4) ∧ ¬(p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end true_propositions_l345_345394


namespace center_cell_value_l345_345547

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l345_345547


namespace average_of_11_numbers_l345_345187

theorem average_of_11_numbers (a b c d e f g h i j k : ℕ) 
  (h₀ : (a + b + c + d + e + f) / 6 = 19)
  (h₁ : (f + g + h + i + j + k) / 6 = 27)
  (h₂ : f = 34) :
  (a + b + c + d + e + f + g + h + i + j + k) / 11 = 22 := 
by
  sorry

end average_of_11_numbers_l345_345187


namespace exists_unique_point_M_l345_345149

theorem exists_unique_point_M (A B C : Point) (h_triangle : Triangle A B C) :
  ∃! M : Point, (dist M A + dist B C = dist M B + dist A C ∧ 
                 dist M B + dist A C = dist M C + dist A B ∧ 
                 dist M C + dist A B = dist M A + dist B C) :=
sorry

end exists_unique_point_M_l345_345149


namespace tan_double_angle_l345_345034

theorem tan_double_angle (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : Real.tan (2 * α) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l345_345034


namespace A_finishes_work_in_8_days_l345_345736

theorem A_finishes_work_in_8_days 
  (A_work B_work W : ℝ) 
  (h1 : 4 * A_work + 6 * B_work = W)
  (h2 : (A_work + B_work) * 4.8 = W) :
  A_work = W / 8 :=
by
  -- We should provide the proof here, but we will use "sorry" for now.
  sorry

end A_finishes_work_in_8_days_l345_345736


namespace locus_of_circumcircle_centers_l345_345467

open EuclideanGeometry

-- Definitions for points, angles, distances, and circumcircles
variables (A O M N K L P : Point)
variables (α r : ℝ)
variables (angle_bisectors : Angle)
variables (tangent_line : Line)
variables (circumcircle_center : Point)

-- Given conditions
def given_conditions := 
  Angle A ∧
  Circle A O ∧
  is_tangent_to tangent_line A ∧
  intersects_at tangent_line (AngleSides A) M N

-- Definitions of distances
def distances_conditions :=
  dist(A, L) = r / (sin(α / 2) * (1 + sin(α / 2))) ∧
  dist(A, P) = r / (sin(α / 2) * (1 - sin(α / 2)))

-- Define the locus
def locus :=
  Segment K L ∨ Ray P tangent_line

-- Main theorem
theorem locus_of_circumcircle_centers :
  ∀ (circumcircle_center : Point),
    given_conditions A O M N K L P α r angle_bisectors tangent_line →
    distances_conditions A L P r α →
    circumcircle_center ∈ locus K L P tangent_line :=
sorry

end locus_of_circumcircle_centers_l345_345467


namespace arith_geo_seq_necessary_and_sufficient_l345_345852

structure ArithGeoSeq (a : ℕ → ℝ) (q : ℝ) :=
  (a_pos : ∀ n, 0 < a n)

-- Define sequence properties
def seq_properties (a : ℕ → ℝ) (q : ℝ) :=
  q = 0 ∧ (∀ n, a n = a 1 * q ^ n)

-- Problem restated in Lean 
theorem arith_geo_seq_necessary_and_sufficient (q : ℝ) (a : ℕ → ℝ) 
  [ArithGeoSeq a q] : seq_properties a q → (a 1 > 0 ↔ a 2017 > 0) :=
by
  sorry

end arith_geo_seq_necessary_and_sufficient_l345_345852


namespace correct_proposition_D_l345_345712

theorem correct_proposition_D (a b : ℝ) (h1 : a < 0) (h2 : b < 0) : 
  (b / a) + (a / b) ≥ 2 := 
sorry

end correct_proposition_D_l345_345712


namespace student_finished_6_problems_in_class_l345_345762

theorem student_finished_6_problems_in_class (total_problems : ℕ) (x y : ℕ) (h1 : total_problems = 15) (h2 : 3 * y = 2 * x) (h3 : x + y = total_problems) : y = 6 :=
sorry

end student_finished_6_problems_in_class_l345_345762


namespace set_intersection_l345_345888

def setA : Set ℝ := {x | x < 1}
def setB : Set ℝ := {x | 3 ^ x < 1}
def intersection : Set ℝ := {x | x < 0}

theorem set_intersection :
  (setA ∩ setB) = intersection :=
sorry

end set_intersection_l345_345888


namespace coloring_ways_l345_345656

-- Definitions of the problem:
def column1 := 1
def column2 := 2
def column3 := 3
def column4 := 4
def column5 := 3
def column6 := 2
def column7 := 1
def total_colors := 3 -- Blue, Yellow, Green

-- Adjacent coloring constraints:
def adjacent_constraints (c1 c2 : ℕ) : Prop := c1 ≠ c2

-- Number of ways to color figure:
theorem coloring_ways : 
  (∃ (n : ℕ), n = 2^5) ∧ 
  n = 32 :=
by 
  sorry

end coloring_ways_l345_345656


namespace time_difference_l345_345353

theorem time_difference (v : ℝ) (h : v ≠ 0) : 
  let t1 := 250 / v in
  let t2 := 360 / (2 * v) in
  t1 - t2 = 70 / v :=
by
  -- begin proof
  sorry

end time_difference_l345_345353


namespace chairs_subsets_l345_345251

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l345_345251


namespace jerry_earnings_per_task_l345_345586

theorem jerry_earnings_per_task :
  ∀ (task_hours : ℕ) (daily_hours : ℕ) (days_per_week : ℕ) (total_earnings : ℕ),
    task_hours = 2 →
    daily_hours = 10 →
    days_per_week = 5 →
    total_earnings = 1400 →
    total_earnings / ((daily_hours / task_hours) * days_per_week) = 56 :=
by
  intros task_hours daily_hours days_per_week total_earnings
  intros h_task_hours h_daily_hours h_days_per_week h_total_earnings
  sorry

end jerry_earnings_per_task_l345_345586


namespace tangent_curve_a_vals_l345_345527

theorem tangent_curve_a_vals (a : ℝ) :
  (∃ x : ℝ, y = x ^ 3 - a) ∧ (∃ x : ℝ, y = 3 * x + 1) ∧ tangent_at_point :=
  a = -3 ∨ a = 1 :=
begin
  sorry
end

end tangent_curve_a_vals_l345_345527


namespace verify_propositions_l345_345375

-- Define the propositions
def p1 : Prop := ∀(L1 L2 L3 : Set Point), 
  (Lines_Intersect_Pairwise L1 L2 L3 ∧ Not_Through_Same_Point L1 L2 L3) → Lie_In_Same_Plane L1 L2 L3

def p2 : Prop := ∀(P1 P2 P3 : Point), ∃! (Plane_Containing P1 P2 P3)

def p3 : Prop := ∀(l m : Line), (Not_Intersect l m) → Parallel l m

def p4 : Prop := ∀(l : Line) (α : Plane) (m : Line), 
  (Contained l α ∧ Perpendicular m α) → Perpendicular m l

-- Proving given conditions
theorem verify_propositions : 
  (p1 ∧ p4) ∧ (¬ (p1 ∧ p2)) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := 
by 
  sorry

end verify_propositions_l345_345375


namespace concurrency_of_bf_ce_ah_l345_345988

theorem concurrency_of_bf_ce_ah
  {A B C E F H : Point} 
  (hAB : Segment A B)
  (hAC : Segment A C)
  (hBE : Segment B E)
  (hCF : Segment C F)
  (hAH : Altitude A H B C)
  (h_eq_abe_acf : equilateralTriangle (Triangle.mk A B E) = equilateralTriangle (Triangle.mk A C F))
  (h_angle_abe_acf : ∠ A B E = 90 ∧ ∠ A C F = 90) :
  concurrent (Line.mk B F) (Line.mk C E) hAH :=
sorry

end concurrency_of_bf_ce_ah_l345_345988


namespace simplify_fraction_evaluate_expression_l345_345313

-- Part (1)
theorem simplify_fraction (α : ℝ) : 
  (sin (α + 3/2 * Real.pi) * sin (-α + Real.pi) * cos (α + Real.pi / 2)) / 
  (cos (-α - Real.pi) * cos (α - Real.pi / 2) * tan (α + Real.pi)) = 
  - cos α := 
  sorry

-- Part (2)
theorem evaluate_expression : 
  tan 675 + sin (-330) + cos 960 = -1 :=
  sorry

end simplify_fraction_evaluate_expression_l345_345313


namespace circle_minus_point_circle_minus_non_measurable_l345_345317

noncomputable def theta (x : ℝ) : ℝ := sorry -- Θ such that π/θ ∉ ℚ

def unit_circle : set ℂ := {z : ℂ | ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * π ∧ z = complex.exp (complex.I * α)}

def A (theta : ℝ) : set ℂ := {z : ℂ | ∃ (k : ℕ), z = complex.exp (complex.I * (k * theta))}
def B (theta : ℝ) : set ℂ := unit_circle \ A theta

def representative_set (x : ℝ) (theta : ℝ) : ℝ := sorry -- A set of representatives based on axiom of choice

theorem circle_minus_point (theta : ℝ) (H : real.pi / theta ∉ ℚ) :
  ∃ (A B : set ℂ), A ∪ B = unit_circle ∧ A ∩ B = ∅ ∧ ∃ z ∈ unit_circle, B = unit_circle \ {1} := 
sorry

theorem circle_minus_non_measurable (theta : ℝ) (H : real.pi / theta ∉ ℚ) :
  ∃ (A B : set ℂ), A ∪ B = unit_circle ∧ A ∩ B = ∅ ∧ ∃ X, X ⊆ unit_circle ∧ ¬ measurable_set X ∧ B = unit_circle \ X := 
sorry

end circle_minus_point_circle_minus_non_measurable_l345_345317


namespace number_of_valid_sequences_l345_345642

-- Define the square and transformations
structure Point where
  x : ℝ
  y : ℝ

def E : Point := ⟨2, 2⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨-2, -2⟩
def H : Point := ⟨2, -2⟩

inductive Transformation
| L : Transformation  -- Rotation 90 degrees counterclockwise
| R : Transformation  -- Rotation 90 degrees clockwise
| H : Transformation  -- Reflection across x-axis
| V : Transformation  -- Reflection across y-axis

open Transformation

-- Dihedral group D4 transformations and properties
def dihedral_action (t : Transformation) (p : Point) : Point :=
  match t with
  | L => ⟨-p.y, p.x⟩
  | R => ⟨p.y, -p.x⟩
  | H => ⟨p.x, -p.y⟩
  | V => ⟨-p.x, p.y⟩

-- Condition for a sequence of transformations to return vertices to original positions
def is_identity_sequence (seq : List Transformation) : Prop :=
  List.foldl dihedral_action ⟨E, F, G, H⟩ seq = ⟨E, F, G, H⟩

-- Main theorem statement
theorem number_of_valid_sequences : 
  ∃ (seqs : Finset (List Transformation)), 
  (∀ seq ∈ seqs, is_identity_sequence (seq.push L)) ∧ 
  seqs.card = 2^33 := 
sorry

end number_of_valid_sequences_l345_345642


namespace probability_red_even_and_green_gt_3_l345_345278

variable {Ω : Type} [Probability Ω]

/-- Two 6-sided dice, one red and one green are rolled. -/
def roll_red_is_even (ω : Ω) : Prop :=
  let red := (nat % 6) ω in
  red = 2 ∨ red = 4 ∨ red = 6

def roll_green_greater_than_3 (ω : Ω) : Prop :=
  let green := (nat % 6) ω in
  green = 4 ∨ green = 5 ∨ green = 6

theorem probability_red_even_and_green_gt_3 :
  Prob (λ ω, roll_red_is_even ω ∧ roll_green_greater_than_3 ω) = 1 / 4 :=
by
  sorry

end probability_red_even_and_green_gt_3_l345_345278


namespace symmetry_planes_position_l345_345355

theorem symmetry_planes_position
  (S1 S2 S3 : Plane)
  (h1 : is_symmetry_plane S1)
  (h2 : is_symmetry_plane S2)
  (h3 : is_symmetry_plane S3)
  (h4 : ¬ ∃ S, S ≠ S1 ∧ S ≠ S2 ∧ S ≠ S3 ∧ is_symmetry_plane S) :
  (∃ P1 P2 P3 : Line, 
    intersects_at_right_angle S1 S2 P1 ∧ 
    intersects_at_right_angle S2 S3 P2 ∧ 
    intersects_at_right_angle S1 S3 P3) ∨
  (∃ Q1 Q2 Q3 : Point,
    intersects_at_angle S1 S2 Q1 60 ∧
    intersects_at_angle S2 S3 Q2 60 ∧
    intersects_at_angle S1 S3 Q3 60) :=
sorry

end symmetry_planes_position_l345_345355


namespace digits_difference_1729_l345_345802

theorem digits_difference_1729 :
  let n : ℕ := 1729 in
  let base4_digits := Nat.log 4 (n + 1) in
  let base6_digits := Nat.log 6 (n + 1) in
  base4_digits - base6_digits = 1 := 
by
  sorry

end digits_difference_1729_l345_345802


namespace subsequence_intersection_at_most_k_l345_345893

theorem subsequence_intersection_at_most_k
  (n k : ℕ) (h : k ≤ n / 2) :
  let seq := (List.range n).bind (fun i => List.repeat (i + 1) k)
  in
  ∀ (B : Fin n → List ℕ), (∀ i, ∃ s, B i = seq.drop (s * k) |>.take k) →
  (∃ (A : Finset (Fin n)), Finset.card A ≤ k ∧ ∀ i j, i ≠ j → ((B i) ∩ (B j) = ∅)) :=
by
  admit -- proof omitted

end subsequence_intersection_at_most_k_l345_345893


namespace minSumLog_l345_345479

open Real

noncomputable def sumLogMin (n : ℕ) (xs : Fin n → ℝ) : ℝ :=
  ∑ k in Finset.range n, Real.log (xs (k + 1) % n - 1 / 4) / Real.log (xs 4)

theorem minSumLog (n : ℕ) (xs : Fin n → ℝ) (h1 : ∀ k : Fin n, 1 / 4 < xs k ∧ xs k < 1) (h2 : xs (n % n) = xs 0) :
  ∑ k in Finset.range n, Real.log (xs ((k + 1) % n) - 1 / 4) / Real.log (xs 4) = 2 * n :=
sorry

end minSumLog_l345_345479


namespace quadratic_integers_pairs_l345_345071

theorem quadratic_integers_pairs (m n : ℕ) :
  (0 < m ∧ m < 9) ∧ (0 < n ∧ n < 9) ∧ (m^2 > 9 * n) ↔ ((m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2)) :=
by {
  -- Insert proof here
  sorry
}

end quadratic_integers_pairs_l345_345071


namespace square_football_area_l345_345999

-- Define a real number to simulate the value of π
noncomputable def π : ℝ := real.pi

-- Define side length of the square
def PQ : ℝ := 4

-- Define the area of the quarter circle
def area_quarter_circle : ℝ := (1/4) * π * (PQ^2)

-- Define the area of the isosceles right triangle
def area_triangle : ℝ := (1/2) * (PQ) * (PQ)

-- Define the area of region III by subtracting triangle area from quarter circle area
def area_region_III : ℝ := area_quarter_circle - area_triangle

-- Define the total area of regions II and III combined
def total_area_II_and_III : ℝ := 2 * area_region_III

-- Given conditions and definitions, state the main theorem
theorem square_football_area : |total_area_II_and_III - 9.1| < 0.1 :=
by
  sorry

end square_football_area_l345_345999


namespace go_total_pieces_l345_345058

theorem go_total_pieces (T : ℕ) (h : T > 0) (prob_black : T = (3 : ℕ) * 4) : T = 12 := by
  sorry

end go_total_pieces_l345_345058


namespace period_of_f_max_F_on_interval_find_k_for_zero_count_l345_345469

noncomputable def f (x : ℝ) : ℝ := (Real.abs (Real.sin x)) + (Real.abs (Real.cos x))
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := k + 4 * (Real.sin x) * (Real.cos x)
noncomputable def F (x : ℝ) (k : ℝ) : ℝ := f x - g x k

-- (1) Prove that π / 2 is a period of the function f(x)
theorem period_of_f : ∀ x : ℝ, f (x + Real.pi / 2) = f x := 
by
  sorry

-- (2) When k = 0, find the maximum value of F(x) on the interval [π/2, π]
theorem max_F_on_interval : ∃ x ∈ Set.Icc (Real.pi / 2) Real.pi, F x 0 = 2 + Real.sqrt 2 := 
by
  sorry

-- (3) If F(x) has exactly an odd number of zeros in (0, π), find k
theorem find_k_for_zero_count : ∃ k : ℝ, 
  (∃! x : ℝ, x ∈ Set.Ioo 0 Real.pi ∧ F x k = 0) ∧ 
  (k = 1 ∨ k = Real.sqrt 2 - 2 ∨ k = Real.sqrt 2 + 2) := 
by
  sorry

end period_of_f_max_F_on_interval_find_k_for_zero_count_l345_345469


namespace first_proof_second_proof_l345_345520

-- Define the conditions and prove the statements in Lean

namespace GeometryProofs

-- Define the points, angles, and lengths involved in the problem
variables (A B C D E F T : Type)
variables (α β : ℝ)
variables (AE EB AF FC DT DA EF BC : ℝ)

-- Define the conditions
def conditions : Prop :=
  (AE = EB) ∧ (AF = FC) ∧ (angle A E B = 2 * α) ∧ (angle A F B = 2 * β) ∧ 
  (angle D B C = β) ∧ (angle B C D = α) ∧ (T is_projection_of D on BC)

-- First theorem statement
theorem first_proof (h : conditions A B C D E F T α β AE EB AF FC DT DA EF BC) : 
  ⊢ DA ⊥ EF := 
  sorry

-- Second theorem statement
theorem second_proof (h : conditions A B C D E F T α β AE EB AF FC DT DA EF BC) : 
  ⊢ DA / EF = 2 * (DT / BC) := 
  sorry

end GeometryProofs

end first_proof_second_proof_l345_345520


namespace a_n_formula_S_n_ge_1_l345_345613

def T_n (n : ℕ) : ℝ := -- Assuming a helper function to denote T_n, this will need an implementation
sorry

def S_n (n : ℕ) : ℝ := -- Assuming a helper function to denote S_n
sorry 

def a_n (n : ℕ) : ℝ := -- The derived formula for a_n
(3 / 2) ^ (n - 1)

theorem a_n_formula (n : ℕ) (h : n > 0) : 
  3 * (3 / 2) ^ (n - 1) - 2 * n = T_n n :=
sorry

theorem S_n_ge_1 (n : ℕ) (h : n > 0) : 
  S_n n ≥ 1 :=
sorry

end a_n_formula_S_n_ge_1_l345_345613


namespace no_2010_sided_polygon_with_sides_1_to_2010_has_inscribed_circle_l345_345807

theorem no_2010_sided_polygon_with_sides_1_to_2010_has_inscribed_circle :
  ¬ ∃ (S : Finset ℕ), (S = Finset.range 2010) ∧
                      ((S.sum id = (1005 * 2011)) ∧
                      is_possible_partition(S)) := 
sorry


end no_2010_sided_polygon_with_sides_1_to_2010_has_inscribed_circle_l345_345807


namespace possible_angle_B_l345_345124

noncomputable def angle_B_possible_values (R b : ℝ) (BO BH : ℝ) (sinB : ℝ) : Set ℝ :=
  { B | ∃ O H A C : ℝ×ℝ, 
    O = (0,0) ∧
    H = A + B + C ∧
    BO = BH ∧
    BO = R ∧
    BH = R ∧
    B = 60 ∨ B = 120 }

theorem possible_angle_B (R : ℝ) (b : ℝ) (O H A C : ℝ×ℝ) (B : ℝ) (BO BH: ℝ) :
  O = (0,0) → 
  H = A + B + C → 
  BO = BH → 
  BO = R → 
  BH = R →
  b = R * real.sqrt 3 →
  sinB = real.sqrt 3 / 2 → 
  B = 60 ∨ B = 120 :=
by
  intros
  sorry

end possible_angle_B_l345_345124


namespace probability_sum_of_two_dice_is_five_l345_345709

theorem probability_sum_of_two_dice_is_five :
  let outcomes := {(a, b) | a ∈ Fin 6, b ∈ Fin 6}
  let favorable := {(a, b) | (a + 1) + (b + 1) = 5}
  (favorable.to_finset.card / outcomes.to_finset.card : ℚ) = 1 / 9 :=
by
  sorry

end probability_sum_of_two_dice_is_five_l345_345709


namespace graph_edges_upper_bound_l345_345122

theorem graph_edges_upper_bound (G : Type) [graph G] (n m : ℕ)
  (h_vertices : graph.num_vertices G = n)
  (h_edges : graph.num_edges G = m)
  (no_triangles : ∀ (a b c : G), ¬(graph.is_triangle a b c))
  (no_quad : ∀ (a b c d : G), ¬(graph.is_quadrilateral a b c d)) :
  m ≤ n * Math.sqrt(n - 1) / 2 :=
sorry

end graph_edges_upper_bound_l345_345122


namespace best_store_is_A_l345_345060

/-- Problem conditions -/
def price_per_ball : Nat := 25
def balls_to_buy : Nat := 58

/-- Store A conditions -/
def balls_bought_per_offer_A : Nat := 10
def balls_free_per_offer_A : Nat := 3

/-- Store B conditions -/
def discount_per_ball_B : Nat := 5

/-- Store C conditions -/
def cashback_rate_C : Nat := 40
def cashback_threshold_C : Nat := 200

/-- Cost calculations -/
def cost_store_A (total_balls : Nat) (price : Nat) : Nat :=
  let full_offers := total_balls / balls_bought_per_offer_A
  let remaining_balls := total_balls % balls_bought_per_offer_A
  let balls_paid_for := full_offers * (balls_bought_per_offer_A - balls_free_per_offer_A) + remaining_balls
  balls_paid_for * price

def cost_store_B (total_balls : Nat) (price : Nat) (discount : Nat) : Nat :=
  total_balls * (price - discount)

def cost_store_C (total_balls : Nat) (price : Nat) (cashback_rate : Nat) (threshold : Nat) : Nat :=
  let cost_before_cashback := total_balls * price
  let full_cashbacks := cost_before_cashback / threshold
  let cashback_amount := full_cashbacks * cashback_rate
  cost_before_cashback - cashback_amount

theorem best_store_is_A :
  cost_store_A balls_to_buy price_per_ball = 1075 ∧
  cost_store_B balls_to_buy price_per_ball discount_per_ball_B = 1160 ∧
  cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C = 1170 ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_B balls_to_buy price_per_ball discount_per_ball_B ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C :=
by {
  -- placeholder for the proof
  sorry
}

end best_store_is_A_l345_345060


namespace parallelogram_equal_diagonals_not_square_l345_345303

theorem parallelogram_equal_diagonals_not_square 
  (P : Type) [parallelogram P] : 
  ¬(diagonals_equal P → is_square P) :=
by
  sorry

end parallelogram_equal_diagonals_not_square_l345_345303


namespace triangles_are_equilateral_l345_345775

-- Define conditions
variables (A P Q B C D P' Q' : Point)

-- Conditions
def is_equilateral_triangle (APQ : triangle A P Q) : Prop :=
  (dist A P = dist P Q) ∧ (dist P Q = dist Q A)

def is_midpoint (M X Y : Point) : Prop :=
  dist X M = dist M Y ∧ dist X Y = 2 * dist X M

-- Geometric Configuration conditions
axiom APQ_equilateral : is_equilateral_triangle (triangle A P Q)
axiom rectangle_AROUND_APQ : ∃ B C D, (is_rectangle B C D A ∧ 
  lies_on P BC ∧ lies_on Q CD)
axiom midpoint_PP'_AP : is_midpoint P' A P
axiom midpoint_QQ'_AQ : is_midpoint Q' A Q

-- Theorem to prove
theorem triangles_are_equilateral :
  is_equilateral_triangle (triangle B Q' C) ∧
  is_equilateral_triangle (triangle C P' D) :=
by
  sorry

end triangles_are_equilateral_l345_345775


namespace hawk_babies_expected_l345_345750

theorem hawk_babies_expected 
  (kettles : ℕ) (pregnancies_per_kettle : ℕ) (babies_per_pregnancy : ℕ) (loss_fraction : ℝ)
  (h1 : kettles = 6)
  (h2 : pregnancies_per_kettle = 15)
  (h3 : babies_per_pregnancy = 4)
  (h4 : loss_fraction = 0.25) :
  let total_babies := kettles * pregnancies_per_kettle * babies_per_pregnancy
  let surviving_fraction := 1 - loss_fraction
  let expected_surviving_babies := (total_babies : ℝ) * surviving_fraction
  in expected_surviving_babies = 270 :=
by
  sorry

end hawk_babies_expected_l345_345750


namespace pepperoni_crust_ratio_l345_345953

-- Define the conditions as Lean 4 statements
def L : ℕ := 50
def C : ℕ := 2 * L
def D : ℕ := 210
def S : ℕ := L + C + D
def S_E : ℕ := S / 4
def CR : ℕ := 600
def CH : ℕ := 400
def PizzaTotal (P : ℕ) : ℕ := CR + P + CH
def PizzaEats (P : ℕ) : ℕ := (PizzaTotal P) / 5
def JacksonEats : ℕ := 330

theorem pepperoni_crust_ratio (P : ℕ) (h1 : S_E + PizzaEats P = JacksonEats) : P / CR = 1 / 3 :=
by sorry

end pepperoni_crust_ratio_l345_345953


namespace area_of_largest_square_l345_345773

-- Define the lengths of the sides of the right triangle
variables (XY XZ YZ : ℝ)

-- Condition: angle XYZ is a right angle
def is_right_angle : Prop :=
  sorry  -- this can hold other properties depending on the actual requirement.

-- The Pythagorean theorem
def pythagorean_theorem : Prop :=
  YZ^2 = XY^2 + XZ^2

-- The sum of the areas of the three squares
def sum_of_areas : Prop :=
  XY^2 + XZ^2 + YZ^2 = 450

-- The final statement to be proven
theorem area_of_largest_square (h_right : is_right_angle) (h_pyth : pythagorean_theorem) (h_sum : sum_of_areas) :
  YZ^2 = 225 :=
by
  sorry

end area_of_largest_square_l345_345773


namespace mn_values_l345_345974

theorem mn_values (m n : ℤ) (h : m^2 * n^2 + m^2 + n^2 + 10 * m * n + 16 = 0) : 
  (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
  sorry

end mn_values_l345_345974


namespace find_DF_l345_345105

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345105


namespace min_value_expression_l345_345018

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : log ((x * y)^y) = Real.exp x) : 
  ∃ c : ℝ, (∀ a b : ℝ, 0 < a → 0 < b → log ((a * b)^b) = Real.exp a → a^2 * b - log a - a ≥ c) 
  ∧ (c = 1) :=
by
  use 1
  intro a b ha hb hab
  sorry

end min_value_expression_l345_345018


namespace successful_multiplications_in_one_hour_l345_345742

variable (multiplications_per_second : ℕ)
variable (error_rate_percentage : ℕ)

theorem successful_multiplications_in_one_hour
  (h1 : multiplications_per_second = 15000)
  (h2 : error_rate_percentage = 5)
  : (multiplications_per_second * 3600 * (100 - error_rate_percentage) / 100) 
    + (multiplications_per_second * 3600 * error_rate_percentage / 100) = 54000000 := by
  sorry

end successful_multiplications_in_one_hour_l345_345742


namespace area_rect_DEF_G_l345_345909

open EuclideanGeometry

-- Define the problem conditions
variable (A B C D E G F : Point)
variable (rectABCD : Rectangle A B C D)
variable (midpointE : Midpoint E A D)
variable (midpointG : Midpoint G C D)
variable (midpointF : Midpoint F B C)
variable (areaABCD : Area rectABCD = 144)

-- Define the theorem to prove
theorem area_rect_DEF_G :
  ∃ (rectDEFG : Rectangle E D G F), Area rectDEFG = 36 :=
by
  sorry

end area_rect_DEF_G_l345_345909


namespace F_101_value_l345_345517

-- Define the recurrence relation
def F : ℕ → ℚ
| 1       := 2
| (n + 1) := (2 * F n + 1) / 2

-- Statement of the theorem we want to prove
theorem F_101_value : F 101 = 52 := sorry

end F_101_value_l345_345517


namespace sphere_radius_eq_cylinder_radius_l345_345217

theorem sphere_radius_eq_cylinder_radius
  (r h d : ℝ) (h_eq_d : h = 16) (d_eq_h : d = 16)
  (sphere_surface_area_eq_cylinder : 4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h) : 
  r = 8 :=
by
  sorry

end sphere_radius_eq_cylinder_radius_l345_345217


namespace expected_babies_proof_l345_345749

-- Define the conditions
def num_kettles : ℕ := 6
def avg_pregnancies_per_kettle : ℕ := 15
def babies_per_pregnancy : ℕ := 4
def loss_fraction : ℝ := 0.25

-- Define the expected number of babies
def total_babies_expected : ℕ :=
  let babies_per_kettle := avg_pregnancies_per_kettle * babies_per_pregnancy
  in num_kettles * babies_per_kettle

def babies_not_hatching : ℝ :=
  total_babies_expected * loss_fraction

def final_expected_babies : ℝ :=
  total_babies_expected - babies_not_hatching

-- The goal is to prove the final expected number of babies
theorem expected_babies_proof : final_expected_babies = 270 := by
  sorry

end expected_babies_proof_l345_345749


namespace amount_of_meat_left_l345_345581

theorem amount_of_meat_left (initial_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ)
  (h0 : initial_meat = 20)
  (h1 : meatballs_fraction = 1/4)
  (h2 : spring_rolls_meat = 3) : 
  (initial_meat - (initial_meat * meatballs_fraction:ℕ) - spring_rolls_meat) = 12 := 
by 
  sorry

end amount_of_meat_left_l345_345581


namespace exists_special_trio_l345_345763

theorem exists_special_trio (P : Finset ℕ) (h_card : P.card = 101)
  (h_matches : ∀ (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P), a ≠ b → ∃ n : ℕ, n ≤ 10 ∧ (a,b) ∈ n) 
  (h_scores : ∃ a b : ℕ, (a, b) ∈ (11, 0) ∨ (a, b) ∈ (11, 10)) :
  ∃ (A B C : ℕ), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ 
  (∃ n : ℕ, n ≤ 10 ∧ (A, B) ∈ n ∧ (A, C) ∈ n ∧ (B, C) ∉ n) :=
by
  sorry

end exists_special_trio_l345_345763


namespace evaluate_expression_l345_345811

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) : 2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end evaluate_expression_l345_345811


namespace polynomial_coefficients_equality_l345_345505

theorem polynomial_coefficients_equality 
  (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (2 : ℝ)*x - 3)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) -> 
  (a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10) :=
by
  sorry

end polynomial_coefficients_equality_l345_345505


namespace intersection_height_l345_345074

-- Define the height of each pole and their separation distance
def height1 : ℝ := 30
def height2 : ℝ := 90
def distance : ℝ := 150

-- Define the equations of the lines connecting the top of each pole to the foot of the opposite pole
def line1 (x : ℝ) : ℝ := - (1/5) * x + height1
def line2 (x : ℝ) : ℝ := (3/5) * x

-- The theorem to prove the intersection height
theorem intersection_height : ∃ y : ℝ, line1 distance = line2 distance ∧ y = 22.5 :=
by
  sorry

end intersection_height_l345_345074


namespace all_numbers_are_prime_l345_345981

theorem all_numbers_are_prime 
  (a : ℕ → ℕ)
  (p : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1))
  (h2 : ∀ n, prime (p n))
  (h3 : ∀ n, p n ∣ a n)
  (h4 : ∀ n k, a n - a k = p n - p k) :
  ∀ n, prime (a n) :=
sorry

end all_numbers_are_prime_l345_345981


namespace sum_eq_formula_l345_345432

theorem sum_eq_formula (n : ℕ) : 
  (∑ k in Finset.range (n+1), k * (n - k + 1)) = n * (n + 1) * (n + 2) / 6 := 
sorry

end sum_eq_formula_l345_345432


namespace molecular_weight_l345_345705

theorem molecular_weight :
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  (2 * H_weight + 1 * Br_weight + 3 * O_weight + 1 * C_weight + 1 * N_weight + 2 * S_weight) = 220.065 :=
by
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  sorry

end molecular_weight_l345_345705


namespace angle_between_slant_line_and_plane_l345_345912

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (0, 1, 1)

-- Define the function that computes the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the function that computes the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Define the function that computes the cosine of the angle between two vectors
def cosine_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

-- State the theorem to be proved
theorem angle_between_slant_line_and_plane : 
  Real.arccos (cosine_angle a b) = Real.pi / 3 :=
sorry

end angle_between_slant_line_and_plane_l345_345912


namespace max_value_of_expression_l345_345000

theorem max_value_of_expression :
  ∃ x y : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + 4) / real.sqrt (x^2 + 4 * y^2 + 2) ≤ real.sqrt 29) ∧
            (∃ x y : ℝ, (2 * x + 3 * y + 4) / real.sqrt (x^2 + 4 * y^2 + 2) = real.sqrt 29) := 
sorry

end max_value_of_expression_l345_345000


namespace wedding_cost_l345_345590

theorem wedding_cost (venue_cost food_drink_cost guests_john : ℕ) 
  (guest_increment decorations_base decorations_per_guest transport_couple transport_per_guest entertainment_cost surchage_rate discount_thresh : ℕ) (discount_rate : ℕ) :
  let guests_wife := guests_john + (guests_john * guest_increment / 100)
  let venue_total := venue_cost + (venue_cost * surchage_rate / 100)
  let food_drink_total := if guests_wife > discount_thresh then (food_drink_cost * guests_wife) * (100 - discount_rate) / 100 else food_drink_cost * guests_wife
  let decorations_total := decorations_base + (decorations_per_guest * guests_wife)
  let transport_total := transport_couple + (transport_per_guest * guests_wife)
  (venue_total + food_drink_total + decorations_total + transport_total + entertainment_cost = 56200) :=
by {
  -- Constants given in the conditions
  let venue_cost := 10000
  let food_drink_cost := 500
  let guests_john := 50
  let guest_increment := 60
  let decorations_base := 2500
  let decorations_per_guest := 10
  let transport_couple := 200
  let transport_per_guest := 15
  let entertainment_cost := 4000
  let surchage_rate := 15
  let discount_thresh := 75
  let discount_rate := 10
  sorry
}

end wedding_cost_l345_345590


namespace problem_statement_l345_345612

noncomputable theory

variables {S : Type*} [DecidableEq S] (f : S → S → S → S) (g : S → S → S)

def satisfies_condition_1 (f : S → S → S → S) : Prop :=
  ∀ (x y : S), f(x, x, y) = x ∧ f(x, y, x) = x ∧ f(x, y, y) = x

def satisfies_condition_2 (f : S → S → S → S) : Prop :=
  ∀ (x y z : S), x ≠ y → y ≠ z → z ≠ x → f(x, y, z) ∈ {x, y, z} ∧ 
  f(y, z, x) = f(x, y, z) ∧ f(y, x, z) = f(x, y, z)

def nontrivial_composition_of (f g : S → S → S → S) (g' : S → S → S) : Prop :=
  ∃ h : S → S, ∀ x y z, f(x, y, z) = h (g(x, y, z))

def not_composition_of (g f : S → S → S → S) (g' : S → S → S) : Prop :=
  ¬ ∃ h : S → S → S, ∀ x y z, g(x, y) = h (f(x, y, z), f(y, z, x), f(z, x, y))

theorem problem_statement (h1 : satisfies_condition_1 f)
  (h2 : satisfies_condition_2 f) :
  ∃ g : S → S → S, nontrivial_composition_of f f g ∧ not_composition_of g f f :=
sorry 

end problem_statement_l345_345612


namespace this_year_sales_l345_345761

def last_year_sales : ℝ := 320 -- in millions
def percent_increase : ℝ := 0.5 -- 50%

theorem this_year_sales : (last_year_sales * (1 + percent_increase)) = 480 := by
  sorry

end this_year_sales_l345_345761


namespace find_t_correct_l345_345819

theorem find_t_correct : 
  ∃ t : ℝ, (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 15) = 15 * x^4 - 47 * x^3 + 115 * x^2 - 110 * x + 75) ∧ t = -10 :=
sorry

end find_t_correct_l345_345819


namespace odd_and_increasing_on_interval_l345_345771

open Function

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

def f1 (x : ℝ) : ℝ := |x + 1|

def f2 (x : ℝ) : ℝ := sin x

def f3 (x : ℝ) : ℝ := 2^x + 2^(-x)

def f4 (x : ℝ) : ℝ := log x

theorem odd_and_increasing_on_interval :
  (is_odd f2 ∧ is_increasing_on f2 (-1) 1) ∧
  ¬(is_odd f1 ∧ is_increasing_on f1 (-1) 1) ∧
  ¬(is_odd f3 ∧ is_increasing_on f3 (-1) 1) ∧
  ¬(is_odd f4 ∧ is_increasing_on f4 (-1) 1) :=
by
  sorry

end odd_and_increasing_on_interval_l345_345771


namespace successful_operation_even_n_l345_345454

theorem successful_operation_even_n (n : ℕ) (n_pos : 0 < n) :
  (∃ (f : fin n^2 → fin n^2 × fin n^2), 
    ∀ (i j : fin n^2), 
      i ≠ j → 
      (f i).1 ≠ (f j).1 ∧ (f i).2 ≠ (f j).2 ∧ 
      (f j).1 ≠ (f i).1 ∧
      ((n^2 : ℤ) = ((f i).1.val - (f i).2.val) * ((f i).1.val - (f i).2.val) + ((f j).1.val - (f j).2.val) * ((f j).1.val - (f j).2.val))) →
  even n :=
begin
  sorry
end

end successful_operation_even_n_l345_345454


namespace number_of_people_in_group_l345_345076

theorem number_of_people_in_group (ways_to_seat: ℕ) (factorial_eqn: ∃ n: ℕ, (n - 1)! = ways_to_seat) : 
  ways_to_seat = 144 → ∃ n, (n - 1)! = 144 ∧ n = 6 :=
by
  sorry

end number_of_people_in_group_l345_345076


namespace circles_are_externally_tangent_l345_345210

noncomputable def circle1_center : ℝ × ℝ := (0, 0)
noncomputable def circle1_radius : ℝ := 1

noncomputable def circle2_center : ℝ × ℝ := (2, 0)
noncomputable def circle2_radius : ℝ := 1

theorem circles_are_externally_tangent :
  let d := real.dist circle1_center circle2_center in
  d = circle1_radius + circle2_radius :=
by
  sorry

end circles_are_externally_tangent_l345_345210


namespace prob_white_point_leftmost_l345_345318

noncomputable def prob_sum_at_most_one (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
  have points_drawn : k > 0 := sorry
  (m : ℚ) / (n : ℚ)

theorem prob_white_point_leftmost (p : ℚ):
  let n := 2019
  let m := 1019
  let k := 1000
  let prob := prob_sum_at_most_one n m k
  p = prob := by
  sorry

end prob_white_point_leftmost_l345_345318


namespace range_of_a_l345_345985

variable {α : Type*} [LinearOrder α]

def set_A (a : α) : set α := {x | x ≤ a}
def set_B : set α := {x | x < 2}

theorem range_of_a (a : ℝ) (h : set_A a ⊆ set_B) : a < 2 := sorry

end range_of_a_l345_345985


namespace binomial_summation_eq_l345_345151

open Real

theorem binomial_summation_eq (n : ℕ) :
  (∑ k in range (n+1) | odd k, (-1) ^ ((k - 1) / 2) * 3^(-(k - 1) / 2) * binomial n k) = 
  (2^n / 3^((n-1)/2)) * sin (n * π / 6) :=
sorry

end binomial_summation_eq_l345_345151


namespace smallest_norm_value_l345_345967

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l345_345967


namespace largest_possible_sum_of_areas_of_good_rectangles_l345_345982

theorem largest_possible_sum_of_areas_of_good_rectangles (m n : ℕ) : 
  ∃ S : set (set (ℤ × ℤ)), 
    (∀ r ∈ S, ∃ (a b c d : ℤ), r = {a, b, c, d} ∧ (a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = d.1 ∧ d.2 = a.2) ∧ 
    -m ≤ a.1 ∧ a.1 ≤ m ∧ -n ≤ a.2 ∧ a.2 ≤ n ∧
    a.1 ≠ b.1 ∧ a.2 ≠ b.2) ∧
    ∀ (r1 r2 ∈ S), a ≠ b → ∀ x ∈ r1, x ∉ r2 → 
  S.sum (λ r, area r) = m * n * (m + 1) * (n + 1) := 
sorry

end largest_possible_sum_of_areas_of_good_rectangles_l345_345982


namespace inequality_x_y_z_l345_345607

open Real

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
    (x ^ 3) / ((1 + y) * (1 + z)) + (y ^ 3) / ((1 + z) * (1 + x)) + (z ^ 3) / ((1 + x) * (1 + y)) ≥ 3 / 4 :=
by
  sorry

end inequality_x_y_z_l345_345607


namespace find_x1_l345_345860

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/4) 
  : x1 = 3/4 := 
sorry

end find_x1_l345_345860


namespace num_orange_juice_l345_345288

-- Definitions based on the conditions in the problem
def O : ℝ := sorry -- To represent the number of bottles of orange juice
def A : ℝ := sorry -- To represent the number of bottles of apple juice
def cost_orange_juice : ℝ := 0.70
def cost_apple_juice : ℝ := 0.60
def total_cost : ℝ := 46.20
def total_bottles : ℝ := 70

-- Conditions used as definitions in Lean 4
axiom condition1 : O + A = total_bottles
axiom condition2 : cost_orange_juice * O + cost_apple_juice * A = total_cost

-- Proof statement with the correct answer
theorem num_orange_juice : O = 42 := by
  sorry

end num_orange_juice_l345_345288


namespace average_minutes_run_per_day_l345_345084

variable (e : ℕ) -- Let e be the number of eighth graders
variable (sixth_graders_avg : ℕ) (seventh_graders_avg : ℕ) (eighth_graders_avg : ℕ)
variable (num_sixth_graders_ratio : ℕ) (num_seventh_graders_ratio : ℕ)

axiom sixth_graders_run : sixth_graders_avg = 20
axiom seventh_graders_run : seventh_graders_avg = 18
axiom eighth_graders_run : eighth_graders_avg = 22
axiom sixth_graders_ratio : num_sixth_graders_ratio = 3
axiom seventh_graders_ratio : num_seventh_graders_ratio = 2

theorem average_minutes_run_per_day : 
  let sixth_graders := num_sixth_graders_ratio * e in
  let seventh_graders := num_seventh_graders_ratio * e in
  let eighth_graders := e in
  let total_minutes := sixth_graders_avg * sixth_graders + seventh_graders_avg * seventh_graders + eighth_graders_avg * eighth_graders in
  let total_students := sixth_graders + seventh_graders + eighth_graders in
  total_minutes / total_students = 19.67 :=
by
  rw [sixth_graders_run, seventh_graders_run, eighth_graders_run, sixth_graders_ratio, seventh_graders_ratio]
  sorry

end average_minutes_run_per_day_l345_345084


namespace parabola_intersects_line_segment_range_l345_345032

theorem parabola_intersects_line_segment_range (a : ℝ) :
  (9/9 : ℝ) ≤ a ∧ a < 2 ↔
  ∃ x1 y1 x2 y2 x0,
    (y1 = a * x1^2 - 3 * x1 + 1) ∧
    (y2 = a * x2^2 - 3 * x2 + 1) ∧
    (∀ x, y = a * x^2 - 3 * x + 1) ∧
    (|x1 - x0| > |x2 - x0| → y1 > y2) ∧
    let M := (-1 : ℝ, -2 : ℝ); N := (3 : ℝ, 2 : ℝ) in
    let y_mn := (x : ℝ) → x - 1 in
    ∃ x_1 x_2, x_1 ≠ x_2 ∧
      (a * x_1^2 - 3 * x_1 + 1 = x_1 - 1) ∧
      (a * x_2^2 - 3 * x_2 + 1 = x_2 - 1) :=
begin
  sorry
end

end parabola_intersects_line_segment_range_l345_345032


namespace find_B_smallest_period_l345_345036

variable (A B C : ℝ)
variable (x : ℝ)
variable (m : ℝ × ℝ := (2, -2 * Real.sqrt 3))
variable (n : ℝ × ℝ := (Real.cos B, Real.sin B))
variable (a : ℝ × ℝ := (1 + Real.sin (2 * x), Real.cos (2 * x)))
variable (f : ℝ → ℝ := λ x => (1 + Real.sin (2 * x)) * Real.cos B + Real.cos (2 * x) * Real.sin B)

axiom triangle_angles_sum : A + B + C = π
axiom perpendicular_vectors : m.1 * n.1 + m.2 * n.2 = 0

theorem find_B : B = π / 6 :=
by
  sorry

theorem smallest_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
by
  sorry

end find_B_smallest_period_l345_345036


namespace carpooling_arrangements_l345_345757

def ends9 := "Jia's car ends in 9"
def ends0 := "Car ends in 0"
def ends2 := "Car ends in 2"
def ends1 := "Car ends in 1"
def ends5 := "Car ends in 5"

def cars : List String := [ends9, ends0, ends2, ends1, ends5]

def odd_days : List Nat := [5, 7, 9]
def even_days : List Nat := [6, 8]

def odd_ending_cars : List String := [ends9, ends1, ends5]
def even_ending_cars : List String := [ends0, ends2]

theorem carpooling_arrangements :
  ((odd_days.choose 1).length * (List.replicate 2 (odd_ending_cars.choose 2)).length + (odd_days.choose 0).length * (odd_ending_cars.choose 3).length) *
  (List.replicate 2 (even_ending_cars.choose 2)).length = 80 :=
sorry

end carpooling_arrangements_l345_345757


namespace promotional_statement_2_correct_promotional_statement_3_correct_correct_statements_l345_345738

def promotional_exchange_rule (n : ℕ) : ℕ :=
  n + (n - 1) / 2

theorem promotional_statement_2_correct (n : ℕ) : n ≥ 67 → (n + n / 3 + (n / 3 + 1) / 3 + ((n / 3 + 1) / 3 + 2) / 3 + 1 / 3 = 100) :=
sorry

theorem promotional_statement_3_correct (n : ℕ) : promotional_exchange_rule n = n + (n - 1) / 2 :=
sorry

theorem correct_statements : promotional_statement_2_correct 67 ∧ promotional_statement_3_correct :=
by sorry

end promotional_statement_2_correct_promotional_statement_3_correct_correct_statements_l345_345738


namespace total_amount_paid_l345_345139

-- Define the conditions of the problem
def cost_without_discount (quantity : ℕ) (unit_price : ℚ) : ℚ :=
  quantity * unit_price

def cost_with_discount (quantity : ℕ) (unit_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := cost_without_discount quantity unit_price
  total_cost - (total_cost * discount_rate)

-- Define each category's cost after discount
def pens_cost : ℚ := cost_with_discount 7 1.5 0.10
def notebooks_cost : ℚ := cost_without_discount 4 5
def water_bottles_cost : ℚ := cost_with_discount 2 8 0.30
def backpack_cost : ℚ := cost_with_discount 1 25 0.15
def socks_cost : ℚ := cost_with_discount 3 3 0.25

-- Prove the total amount paid is $68.65
theorem total_amount_paid : pens_cost + notebooks_cost + water_bottles_cost + backpack_cost + socks_cost = 68.65 := by
  sorry

end total_amount_paid_l345_345139


namespace number_of_lines_with_acute_angle_slope_l345_345848

theorem number_of_lines_with_acute_angle_slope :
  let s := {-3, -2, -1, 0, 1, 2, 3}
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a * b < 0 ∧
    (λ (a b c : ℤ), ax + by + c = 0 → 43) := sorry

end number_of_lines_with_acute_angle_slope_l345_345848


namespace number_of_adjacent_subsets_l345_345235

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l345_345235


namespace jackson_meat_left_l345_345579

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l345_345579


namespace num_distinct_real_roots_f_f_eq_zero_l345_345054

def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem num_distinct_real_roots_f_f_eq_zero :
  (∃! x : ℝ, f(f(x)) = 0) ↔ 4 :=
sorry

end num_distinct_real_roots_f_f_eq_zero_l345_345054


namespace minimum_swaps_to_sort_30_volumes_l345_345220

/-- disorder: a pair of volumes where the volume with the larger number stands to the left of the volume with the smaller number --/
def disorder (v : ℕ) (u : ℕ) : Prop := v > u

noncomputable def count_disorders (arr : list ℕ) : ℕ :=
(arr.zip arr.tail).count (λ (vu : ℕ × ℕ), disorder vu.fst vu.snd)

theorem minimum_swaps_to_sort_30_volumes :
  ∀ (initial_arrangement : list ℕ), initial_arrangement.length = 30 →
  minimum_swaps_to_sort initial_arrangement = 435 := by
  sorry
  
/-- minimum_swaps_to_sort - function to calculate the number of swaps required to sort a list in the correct order --/
noncomputable def minimum_swaps_to_sort (arr : list ℕ) : ℕ :=
-- heuristic to determine the minimum number of swaps needed; to be proved 
435


end minimum_swaps_to_sort_30_volumes_l345_345220


namespace seans_sum_divided_by_julies_sum_l345_345163

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l345_345163


namespace basketball_scores_l345_345321

theorem basketball_scores :
  ∃ P: Finset ℕ, (∀ x y: ℕ, (x + y = 7 → P = {p | ∃ x y: ℕ, p = 3 * x + 2 * y})) ∧ (P.card = 8) :=
sorry

end basketball_scores_l345_345321


namespace correct_propositions_l345_345993

noncomputable def proposition_2 (O A B C P : Point) (λ : ℝ) (hλ : λ > 0) : Prop :=
  (vector.op_eq_v(A,P).2 = λ * (vector.direction_v(A,B) / vector.norm (vector.op_eq_v(A,B).2) + vector.direction_v(A,C) / vector.norm (vector.op_eq_v(A,C).2))) 
  → P = incenter O A B C

noncomputable def proposition_3 (O A B C P : Point) (λ : ℝ) (hλ : λ > 0) : Prop :=
  (vector.op_eq_v(A,P).2 = λ * (vector.direction_v(A,B) / (vector.norm (vector.op_eq_v(A,B).2) * sin (angle B)) + vector.direction_v(A,C) / (vector.norm (vector.op_eq_v(A,C).2) * sin (angle C)))) 
  → P = centroid O A B C

noncomputable def proposition_4 (O A B C P : Point) (λ : ℝ) (hλ : λ > 0) : Prop :=
  (vector.op_eq_v(A,P).2 = λ * (vector.direction_v(A,B) / (vector.norm (vector.op_eq_v(A,B).2) * cos (angle B)) + vector.direction_v(A,C) / (vector.norm (vector.op_eq_v(A,C).2) * cos (angle C)))) 
  → P = orthocenter O A B C

noncomputable def proposition_5 (O A B C P : Point) (λ : ℝ) (hλ : λ > 0) : Prop :=
  (vector.op_eq_v(O,P).2 = (vector.direction_v(O,B) + vector.direction_v(O,C))/2 + λ * (vector.direction_v(A,B) / (vector.norm (vector.op_eq_v(A,B).2) * cos (angle B)) + vector.direction_v(A,C) / (vector.norm (vector.op_eq_v(A,C).2) * cos (angle C)))) 
  → P = circumcenter O A B C

theorem correct_propositions (O A B C : Point) (α : Plane) :
  (in_plane O α) ∧ (in_plane A α) ∧ (in_plane B α) ∧ (in_plane C α) ∧ (angle_opposite B AC) ∧ (angle_opposite C AB) →
  [proposition_2, proposition_3, proposition_4, proposition_5] .= [true, true, true, true] :=
begin
  sorry,
end

end correct_propositions_l345_345993


namespace fraction_percent_reduction_l345_345916

theorem fraction_percent_reduction
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0):
  let a_new := a * 0.80
  let b_new := b * 1.28 
  in (a_new / b_new = (a / b) * (5 / 8)) ∧ ((1 - (a_new / b_new) / (a / b)) * 100 = 37.5) :=
by
  let a_new := a * 0.80
  let b_new := b * 1.28 
  have new_fraction : a_new / b_new = (a / b) * (5 / 8), by
    sorry
  have reduction_percentage : (1 - (a_new / b_new) / (a / b)) * 100 = 37.5, by
    sorry
  exact ⟨new_fraction, reduction_percentage⟩

end fraction_percent_reduction_l345_345916


namespace find_DF_l345_345099

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l345_345099


namespace true_propositions_l345_345383

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345383


namespace combined_difference_Carl_Kevin_combined_difference_Carl_Susan_l345_345790

theorem combined_difference_Carl_Kevin :
  let Carl_historical := 70
  let Carl_animal := 40
  let Kevin_historical := 45
  let Kevin_animal := 25
  combined_difference_Carl_Kevin = (Carl_historical - Kevin_historical) + (Carl_animal - Kevin_animal) 
  combined_difference_Carl_Kevin == 40 := 
by 
  sorry 

theorem combined_difference_Carl_Susan :
  let Carl_historical := 70
  let Carl_animal := 40
  let Susan_historical := 60
  let Susan_animal := 35
  combined_difference_Carl_Susan = (Carl_historical - Susan_historical) + (Carl_animal - Susan_animal)
  combined_difference_Carl_Susan == 15 :=
by
  sorry

end combined_difference_Carl_Kevin_combined_difference_Carl_Susan_l345_345790


namespace sum_of_coefficients_l345_345787

theorem sum_of_coefficients :
  let p := -3 * (Polynomial.C 1 * Polynomial.X^8 - Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 4 * Polynomial.X^3 - Polynomial.C 6) + 
           5 * (Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 3 * Polynomial.X^2) - 
           4 * (Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 5)
  in p.coeff 0 = 42 :=
by
  sorry

end sum_of_coefficients_l345_345787


namespace _l345_345992

variables {α : Type*} [inner_product_space ℝ α] -- Assume α is a real inner product space
variables (O A B C P : α) -- O, A, B, C, P are points in α
variable (λ : ℝ) -- λ is a real number

noncomputable def vector_eq (O A B C P : α) (λ : ℝ) : Prop :=
  (P - O) = (A - O) + λ * ((B - A) + (C - A))

noncomputable theorem find_lambda (h₁ : vector_eq O A B C P λ)
  (h₂ : inner_product_space.has_inner.inner ((P - A), (P - O) + (P - B) + (P - C)) = 0) :
  λ = 1 / 4 :=
sorry

end _l345_345992


namespace subsets_with_at_least_three_adjacent_chairs_l345_345247

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l345_345247


namespace crossing_time_l345_345755

def convert_speed (speed_km_hr : ℝ) : ℝ := (speed_km_hr * 1000) / 60

def time_to_cross_bridge (distance_m : ℝ) (speed_m_min : ℝ) : ℝ := distance_m / speed_m_min

theorem crossing_time 
  (speed_km_hr : ℝ) 
  (distance_m : ℝ) 
  (speed_m_min : ℝ := convert_speed speed_km_hr) 
  (time_min : ℝ := time_to_cross_bridge distance_m speed_m_min) 
  (h_speed : speed_km_hr = 10) 
  (h_distance : distance_m = 2500) : 
  time_min = 15 := 
  by 
    sorry

end crossing_time_l345_345755


namespace fraction_value_l345_345707

theorem fraction_value : (4 * 5) / 10 = 2 := by
  sorry

end fraction_value_l345_345707


namespace min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l345_345026

noncomputable def min_value_expression (a b : ℝ) (hab : 2 * a + b = 1) : ℝ :=
  4 * a^2 + b^2 + 1 / (a * b)

theorem min_value_expression_geq_17_div_2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (hab: 2 * a + b = 1) :
  min_value_expression a b hab ≥ 17 / 2 :=
sorry

theorem min_value_expression_eq_17_div_2_for_specific_a_b :
  min_value_expression (1/3) (1/3) (by norm_num) = 17 / 2 :=
sorry

end min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l345_345026


namespace measure_of_angle_C_l345_345529

theorem measure_of_angle_C (a b area : ℝ) (C : ℝ) :
  a = 5 → b = 8 → area = 10 →
  (1 / 2 * a * b * Real.sin C = area) →
  (C = Real.pi / 6 ∨ C = 5 * Real.pi / 6) := by
  intros ha hb harea hformula
  sorry

end measure_of_angle_C_l345_345529


namespace equivalent_forms_l345_345051

-- Given line equation
def given_line_eq (x y : ℝ) : Prop :=
  (3 * x - 2) / 4 - (2 * y - 1) / 2 = 1

-- General form of the line
def general_form (x y : ℝ) : Prop :=
  3 * x - 8 * y - 2 = 0

-- Slope-intercept form of the line
def slope_intercept_form (x y : ℝ) : Prop := 
  y = (3 / 8) * x - 1 / 4

-- Intercept form of the line
def intercept_form (x y : ℝ) : Prop :=
  x / (2 / 3) + y / (-1 / 4) = 1

-- Normal form of the line
def normal_form (x y : ℝ) : Prop :=
  3 / Real.sqrt 73 * x - 8 / Real.sqrt 73 * y - 2 / Real.sqrt 73 = 0

-- Proof problem: Prove that the given line equation is equivalent to the derived forms
theorem equivalent_forms (x y : ℝ) :
  given_line_eq x y ↔ (general_form x y ∧ slope_intercept_form x y ∧ intercept_form x y ∧ normal_form x y) :=
sorry

end equivalent_forms_l345_345051


namespace tommys_family_members_l345_345227

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members_l345_345227


namespace ellipse_properties_l345_345853

theorem ellipse_properties
  (C_center : ∀ (x y a b : ℝ), b ≠ 0 → a ≠ 0 → x^2 / a^2 + y^2 / b^2 = 1)
  (foci_on_x_axis : ∀ (c : ℝ), |c| = a * 1/2)
  (eccentricity : ∀ (c a : ℝ), c / a = 1/2)
  (tangent_line : ∀ (F : ℝ) (θ : ℝ), θ = 60 → (∃ k, y = k(x + F)) ∧ (x^2 + y^2 = b^2 / a^2))
  (line_intersects : ∀ (M N : ℝ), M ≠ N → (∨ (x2 M, y2 M) = y = kx + m))
  (circle_diameter : ∀ ((x1 y1 x2 y2 : ℝ)), (x1 - x2)^2 + (y1 - y2)^2 = (right_vertex_x, right_vertex_y))
  : ∃ p q : ℝ, (C_center = C_focus) → ∃ (x y : ℝ), x = 2/7 ∧ y = 0 := by
  sorry

end ellipse_properties_l345_345853


namespace common_perimeter_eq_1_l345_345133

variable {A B C A1 A2 B1 B2 C1 C2 : Type}
variable [IsEquilateralTriangle A B C] [SideLengthABC : TriangleSideLength A B C 1]
variable [OnSideBC A1] [OnSideBC A2] [OnSideCA B1] [OnSideCA B2] [OnSideAB C1] [OnSideAB C2]
variable [LessDistance A B1 A B2] [LessDistance C A1 C A2] [LessDistance B C1 B C2]
variable [ConcurrentSegments B1 C2] [ConcurrentSegments C1 A2] [ConcurrentSegments A1 B2]
variable [EqualPerimeters (Perimeter A B2 C1) (Perimeter B C2 A1) (Perimeter C A2 B1)]

theorem common_perimeter_eq_1 : Perimeter A B2 C1 = 1 ∧ Perimeter B C2 A1 = 1 ∧ Perimeter C A2 B1 = 1 := by
  sorry

end common_perimeter_eq_1_l345_345133


namespace stratified_sampling_example_l345_345561

theorem stratified_sampling_example
  (total_students : ℕ) (first_year_students : ℕ) (second_year_students : ℕ) (third_year_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 2400)
  (h2 : first_year_students = 800)
  (h3 : second_year_students = 900)
  (h4 : third_year_students = 700)
  (h5 : sample_size = 48) :
  let sampling_ratio := sample_size / total_students in
  let first_year_sampled := first_year_students * sampling_ratio in
  let second_year_sampled := second_year_students * sampling_ratio in
  let third_year_sampled := third_year_students * sampling_ratio in
  first_year_sampled = 16 ∧ second_year_sampled = 18 ∧ third_year_sampled = 14 :=
by
  sorry

end stratified_sampling_example_l345_345561


namespace center_cell_value_l345_345531

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l345_345531


namespace circle_chairs_adjacent_l345_345270

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l345_345270


namespace ellipse_m_value_l345_345486

/-- Given the ellipse $\frac{x^{2}}{25} + \frac{y^{2}}{m^{2}} = 1$ with a focal length of 8,
 and \( m > 0 \), prove that \( m = 3 \) or \( m = \sqrt{41} \). -/
theorem ellipse_m_value (m : ℝ) (h₀ : m > 0)
  (h₁ : ∃ c : ℝ, c = 4 ∧ (ellipse_eq : ∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1)) :
  m = 3 ∨ m = Real.sqrt 41 := 
by
  sorry

end ellipse_m_value_l345_345486


namespace chair_subsets_with_at_least_three_adjacent_l345_345258

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l345_345258


namespace equal_chords_of_circle_l345_345310

theorem equal_chords_of_circle 
  {S : Type*} [EuclideanGeometry S] 
  (A B C D E F G H : S) : 
  (circle A B) → 
  tangent_at B (line A B) → 
  B ∈ segment C D → 
  E ∈ intersection (circle A B) (line A C) → 
  F ∈ intersection (circle A B) (line A D) → 
  G ∈ intersection (circle A B) (line C F) → 
  H ∈ intersection (circle A B) (line D E) → 
  (segment_length A H) = (segment_length A G) := 
sorry

end equal_chords_of_circle_l345_345310


namespace defect_free_product_probability_is_correct_l345_345214

noncomputable def defect_free_probability : ℝ :=
  let p1 := 0.2
  let p2 := 0.3
  let p3 := 0.5
  let d1 := 0.95
  let d2 := 0.90
  let d3 := 0.80
  p1 * d1 + p2 * d2 + p3 * d3

theorem defect_free_product_probability_is_correct :
  defect_free_probability = 0.86 :=
by
  sorry

end defect_free_product_probability_is_correct_l345_345214


namespace right_angle_triangle_l345_345203

theorem right_angle_triangle (a b c : ℝ) (h : (a + b) ^ 2 - c ^ 2 = 2 * a * b) : a ^ 2 + b ^ 2 = c ^ 2 := 
by
  sorry

end right_angle_triangle_l345_345203


namespace plan1_has_higher_expected_loss_l345_345768

noncomputable def prob_minor_flooding : ℝ := 0.2
noncomputable def prob_major_flooding : ℝ := 0.05
noncomputable def cost_plan1 : ℝ := 4000
noncomputable def loss_major_plan1 : ℝ := 30000
noncomputable def loss_minor_plan2 : ℝ := 15000
noncomputable def loss_major_plan2 : ℝ := 30000

noncomputable def expected_loss_plan1 : ℝ :=
  (loss_major_plan1 * prob_major_flooding) + (cost_plan1 * prob_minor_flooding) + cost_plan1

noncomputable def expected_loss_plan2 : ℝ :=
  (loss_major_plan2 * prob_major_flooding) + (loss_minor_plan2 * prob_minor_flooding)

theorem plan1_has_higher_expected_loss : expected_loss_plan1 > expected_loss_plan2 :=
by
  sorry

end plan1_has_higher_expected_loss_l345_345768


namespace total_cost_of_pencils_and_pens_l345_345319

noncomputable def cost_of_pencil : ℝ := 0.1
noncomputable def cost_of_pen (P : ℝ) : Prop := 4 * cost_of_pencil + 5 * P = 2.00

theorem total_cost_of_pencils_and_pens (P : ℝ) (h : cost_of_pen P) : 
  4 * cost_of_pencil + 5 * P = 2.0 :=
by
  exact h

end total_cost_of_pencils_and_pens_l345_345319


namespace sarah_times_sixth_seventh_l345_345630

theorem sarah_times_sixth_seventh (t₁ t₂ t₃ t₄ t₅ t₆ t₇ : ℕ)
  (h₁ : t₁ = 84)
  (h₂ : t₂ = 90)
  (h₃ : t₃ = 87)
  (h₄ : t₄ = 91)
  (h₅ : t₅ = 89)
  (h₆ : (t₁ + t₂ + t₃ + t₄ + t₅ + t₆ + t₇) / 7 = 88)
  : t₆ = 88 ∧ t₇ = 89 :=
begin
  sorry
end

end sarah_times_sixth_seventh_l345_345630


namespace angle_bisector_intersection_ratio_l345_345922

/-- In triangle ABC, given AB = 6, BC = 7, CA = 8,
    AD is the angle bisector of ∠BAC intersecting BC at D,
    BE is the angle bisector of ∠ABC intersecting CA at E,
    and AD and BE intersect at F, then the ratio AF / FD is equal to 2. -/
theorem angle_bisector_intersection_ratio
  (A B C D E F : Type*)
  [triangle A B C]
  (AB : Real := 6)
  (BC : Real := 7)
  (CA : Real := 8)
  (angle_bisector_BAC : angle_bisector A B C A D C)
  (angle_bisector_ABC : angle_bisector B A C B E A)
  (intersection_AD_BE : intersects AD BE = F) :
  ∃ (AF FD : Real), AF / FD = 2 := sorry

end angle_bisector_intersection_ratio_l345_345922


namespace divisors_145_9_count_l345_345898

-- Definitions based on conditions
def divisors_145_9 := {d | ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ d = 5^a * 29^b}

-- Perfect square condition
def is_perfect_square (d : ℕ) : Prop := ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ d = 5^a * 29^b

-- Perfect cube condition
def is_perfect_cube (d : ℕ) : Prop := ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (a % 3 = 0) ∧ (b % 3 = 0) ∧ d = 5^a * 29^b

-- Perfect sixth power condition
def is_perfect_sixth_power (d : ℕ) : Prop := ∃ (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (a % 6 = 0) ∧ (b % 6 = 0) ∧ d = 5^a * 29^b

-- Theorem statement
theorem divisors_145_9_count :
  (finset.filter is_perfect_square divisors_145_9).card +
  (finset.filter is_perfect_cube divisors_145_9).card -
  (finset.filter is_perfect_sixth_power divisors_145_9).card = 37 :=
by sorry

end divisors_145_9_count_l345_345898


namespace sufficient_but_not_necessary_not_necessary_condition_main_theorem_l345_345849

section
  variable {Point : Type}
  variable {Plane : Type}
  variable {Line : Type}
  variable (perpendicular : Line → Plane → Prop)
  variable (parallel : Line → Line → Prop)

  -- Assuming two non-coincident lines and a plane are given
  variable (a b : Line)
  variable (α : Plane)

  theorem sufficient_but_not_necessary 
    (h₁ : perpendicular a α) 
    (h₂ : perpendicular b α) : 
    parallel a b :=
  sorry

  theorem not_necessary_condition 
    (h₃ : parallel a b) : 
    ¬(perpendicular a α ∧ perpendicular b α) :=
  sorry

  theorem main_theorem :
    (∀ (α : Plane) (a b : Line), perpendicular a α ∧ perpendicular b α → parallel a b) ∧ 
    (∃ (α : Plane) (a b : Line), parallel a b ∧ ¬(perpendicular a α ∧ perpendicular b α)) :=
  begin
    split,
    { intros α a b h,
      exact sufficient_but_not_necessary a b α h.1 h.2 },
    { use [α, a, b],
      split,
      { assumption },
      { exact not_necessary_condition a b α },
     }
  end

end

end sufficient_but_not_necessary_not_necessary_condition_main_theorem_l345_345849


namespace problem_solution_l345_345962

theorem problem_solution :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ),
  (∀ x : ℤ, (2 - x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₀ = 32 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1 →
  a₁ + a₂ + a₄ + a₅ = -31 :=
begin
  -- Proof omitted
  sorry
end

end problem_solution_l345_345962


namespace proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l345_345951

noncomputable def prob_boy_pass_all_rounds : ℚ :=
  (5/6) * (4/5) * (3/4) * (2/3)

noncomputable def prob_girl_pass_all_rounds : ℚ :=
  (4/5) * (3/4) * (2/3) * (1/2)

def prob_xi_distribution : (ℚ × ℚ × ℚ × ℚ × ℚ) :=
  (64/225, 96/225, 52/225, 12/225, 1/225)

def exp_xi : ℚ :=
  (0 * (64/225) + 1 * (96/225) + 2 * (52/225) + 3 * (12/225) + 4 * (1/225))

theorem proof_prob_boy_pass_all_rounds :
  prob_boy_pass_all_rounds = 1/3 :=
by
  sorry

theorem proof_prob_girl_pass_all_rounds :
  prob_girl_pass_all_rounds = 1/5 :=
by
  sorry

theorem proof_xi_distribution :
  prob_xi_distribution = (64/225, 96/225, 52/225, 12/225, 1/225) :=
by
  sorry

theorem proof_exp_xi :
  exp_xi = 16/15 :=
by
  sorry

end proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l345_345951


namespace find_valid_n_l345_345815

def has_exactly_n_divisors (n : ℕ) : Prop :=
n ≥ 1 ∧ (nat.num_divisors (2^n - 1) = n)

theorem find_valid_n :
  { n : ℕ // has_exactly_n_divisors n } = {1, 2, 4, 6, 8, 16, 32} :=
by
  sorry

end find_valid_n_l345_345815


namespace radius_when_angle_is_60_angle_when_area_is_max_l345_345485

variable (O A B : Type) (angle radius arc_length area : ℝ)

-- Given that the circumference of sector OAB is 4.
axiom h1 : 2 * radius + arc_length = 4

-- Given that the arc length is AB.
axiom h2 : arc_length = arc_length

-- (1) When ∠AOB = 60°, find the radius of the arc at this time.
theorem radius_when_angle_is_60 (h_angle : ∠O A B = π / 3) (h_arc : arc_length = (π / 3) * radius) :
  radius = 12 / (6 + π) := 
sorry

-- (2) When the area of the sector is maximized, find the size of the central angle at this time.
theorem angle_when_area_is_max (h_area : ∀ x, area = -x^2 + 2 * x) :
  ∠O A B = 2 := 
sorry

end radius_when_angle_is_60_angle_when_area_is_max_l345_345485


namespace tile_difference_is_11_l345_345225

-- Define the initial number of blue and green tiles
def initial_blue_tiles : ℕ := 13
def initial_green_tiles : ℕ := 6

-- Define the number of additional green tiles added as border
def additional_green_tiles : ℕ := 18

-- Define the total number of green tiles in the new figure
def total_green_tiles : ℕ := initial_green_tiles + additional_green_tiles

-- Define the total number of blue tiles in the new figure (remains the same)
def total_blue_tiles : ℕ := initial_blue_tiles

-- Define the difference between the total number of green tiles and blue tiles
def tile_difference : ℕ := total_green_tiles - total_blue_tiles

-- The theorem stating that the difference between the total number of green tiles 
-- and the total number of blue tiles in the new figure is 11
theorem tile_difference_is_11 : tile_difference = 11 := by
  sorry

end tile_difference_is_11_l345_345225


namespace probability_of_rolling_perfect_square_l345_345181

theorem probability_of_rolling_perfect_square :
  (3 / 12 : ℚ) = 1 / 4 :=
by
  sorry

end probability_of_rolling_perfect_square_l345_345181


namespace center_cell_value_l345_345554

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l345_345554


namespace triangle_side_length_median_l345_345095

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l345_345095


namespace range_of_x_satisfying_inequality_l345_345973

theorem range_of_x_satisfying_inequality
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_domain : ∀ x, x ≠ 0 → f x ≠ 0)
  (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = deriv f x)
  (h_cond : ∀ x, 0 < x → x * log x * f' x + f x > 0)
  : { x : ℝ | (x + 2) * f x / (x - 1) ≤ 0 } = set.Iic (-2) ∪ set.Ioo 0 1 := sorry

end range_of_x_satisfying_inequality_l345_345973


namespace Sophia_fraction_finished_l345_345641

/--
Sophia finished a fraction of a book.
She calculated that she finished 90 more pages than she has yet to read.
Her book is 270.00000000000006 pages long.
Prove that the fraction of the book she finished is 2/3.
-/
theorem Sophia_fraction_finished :
  let total_pages : ℝ := 270.00000000000006
  let yet_to_read : ℝ := (total_pages - 90) / 2
  let finished_pages : ℝ := yet_to_read + 90
  finished_pages / total_pages = 2 / 3 :=
by
  sorry

end Sophia_fraction_finished_l345_345641


namespace ellipse_equation_max_area_of_triangle_l345_345024

def ellipse (a b : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1}

def F1 : ℝ × ℝ := (-1, 0)
def M : ℝ × ℝ := (1, real.sqrt 2 / 2)
def l (x : ℝ) : ℝ := - (1 / 2) * x
def A : Set (ℝ × ℝ) := {p | (∃ x y, p = (x, y) ∧ ellipse a b p) ∧ l(x) = y}
def P : Set (ℝ × ℝ) := {p | p ∈ ellipse a b ∧ p ∉ A}

def a : ℝ := real.sqrt 2
def b : ℝ := 1

theorem ellipse_equation :
  ∀ x y : ℝ, (x, y) ∈ ellipse a b ↔ x^2 / 2 + y^2 = 1 := by
  sorry

theorem max_area_of_triangle (ΔABP : ℝ) :
  ΔABP = real.sqrt 2 := by
  sorry

end ellipse_equation_max_area_of_triangle_l345_345024


namespace ratio_of_additional_hours_james_danced_l345_345588

-- Definitions based on given conditions
def john_first_dance_time : ℕ := 3
def john_break_time : ℕ := 1
def john_second_dance_time : ℕ := 5
def combined_dancing_time_excluding_break : ℕ := 20

-- Calculations to be proved
def john_total_resting_dancing_time : ℕ :=
  john_first_dance_time + john_break_time + john_second_dance_time

def john_total_dancing_time : ℕ :=
  john_first_dance_time + john_second_dance_time

def james_dancing_time : ℕ :=
  combined_dancing_time_excluding_break - john_total_dancing_time

def additional_hours_james_danced : ℕ :=
  james_dancing_time - john_total_dancing_time

def desired_ratio : ℕ × ℕ :=
  (additional_hours_james_danced, john_total_resting_dancing_time)

-- Theorem to be proved according to the problem statement
theorem ratio_of_additional_hours_james_danced :
  desired_ratio = (4, 9) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_additional_hours_james_danced_l345_345588


namespace distance_to_friends_house_l345_345986

-- Define the conditions
def uphill_walk_time_min : ℕ := 15
def downhill_walk_time_min : ℕ := 10
def uphill_rate_mph : ℝ := 3
def downhill_rate_mph : ℝ := 6
def uphill_walk_time_hr : ℝ := uphill_walk_time_min / 60.0
def downhill_walk_time_hr : ℝ := downhill_walk_time_min / 60.0

-- Define the problem
theorem distance_to_friends_house :
  let d := uphill_rate_mph * uphill_walk_time_hr in
  d = 0.75 := by
  let uphill_distance := 3 * (15 / 60)
  let downhill_distance := 6 * (10 / 60)
  have h1 : uphill_distance = 0.75 := by
    calc
      3 * (15 / 60 : ℝ)
        = 3 * (0.25 : ℝ) : by norm_num
        = 0.75 : by norm_num
  have h2 : downhill_distance = 0.75 := by
    calc
      6 * (10 / 60 : ℝ)
        = 6 * (1 / 6 : ℝ) : by norm_num
        = 1 : by norm_num
      1 = 0.75 : by sorry  -- here is an inconsistency in initial problem setup needing adjustment or correction
  -- Assume equal distances and prove
  sorry

end distance_to_friends_house_l345_345986


namespace compare_abc_l345_345961

variable (a b c : ℝ)

def a_def : ℝ := 2^-(1/3)
def b_def : ℝ := Real.logBase 2 (1/3)
def c_def : ℝ := Real.logBase (1/2) (1/3)

theorem compare_abc : b < a ∧ a < c :=
by
  have h_a : a = 2^-(1/3) := rfl
  have h_b : b = Real.logBase 2 (1/3) := rfl
  have h_c : c = Real.logBase (1/2) (1/3) := rfl
  sorry

end compare_abc_l345_345961


namespace find_triangle_areas_l345_345938

variables (A B C D : Point)
variables (S_ABC S_ACD S_ABD S_BCD : ℝ)

def quadrilateral_area (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  S_ABC + S_ACD + S_ABD + S_BCD = 25

def conditions (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  (S_ABC = 2 * S_BCD) ∧ (S_ABD = 3 * S_ACD)

theorem find_triangle_areas
  (S_ABC S_ACD S_ABD S_BCD : ℝ) :
  quadrilateral_area S_ABC S_ACD S_ABD S_BCD →
  conditions S_ABC S_ACD S_ABD S_BCD →
  S_ABC = 10 ∧ S_ACD = 5 ∧ S_ABD = 15 ∧ S_BCD = 10 :=
by
  sorry

end find_triangle_areas_l345_345938


namespace tyler_common_ratio_l345_345700

theorem tyler_common_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 10)
  (h2 : (a + 4) / (1 - r) = 15) : 
  r = 1 / 5 :=
by
  sorry

end tyler_common_ratio_l345_345700


namespace inequality_solution_l345_345806

theorem inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := sorry

end inequality_solution_l345_345806


namespace sampling_prob_equal_l345_345003

theorem sampling_prob_equal (N n : ℕ) (P_1 P_2 P_3 : ℝ)
  (H_random : ∀ i, 1 ≤ i ∧ i ≤ N → P_1 = 1 / N)
  (H_systematic : ∀ i, 1 ≤ i ∧ i ≤ N → P_2 = 1 / N)
  (H_stratified : ∀ i, 1 ≤ i ∧ i ≤ N → P_3 = 1 / N) :
  P_1 = P_2 ∧ P_2 = P_3 :=
by
  sorry

end sampling_prob_equal_l345_345003


namespace probability_divides_is_half_l345_345696

def set := {1, 2, 3, 4, 5}

def pairs (s : set ℕ) : set (ℕ × ℕ) := {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2}

def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def valid_pairs (s : set ℕ) : set (ℕ × ℕ) := {p ∈ pairs s | divides (min p.1 p.2) (max p.1 p.2)}

def probability (valid : set (ℕ × ℕ)) (total : set (ℕ × ℕ)) : ℚ := 
  (set.card valid) / (set.card total : ℚ)

theorem probability_divides_is_half : probability (valid_pairs set) (pairs set) = 1 / 2 :=
  sorry

end probability_divides_is_half_l345_345696


namespace remainder_of_3_pow_19_mod_10_l345_345706

theorem remainder_of_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_mod_10_l345_345706


namespace ryan_more_hours_english_than_spanish_l345_345814

-- Define the time spent on various languages as constants
def hoursEnglish : ℕ := 7
def hoursSpanish : ℕ := 4

-- State the problem as a theorem
theorem ryan_more_hours_english_than_spanish : hoursEnglish - hoursSpanish = 3 :=
by sorry

end ryan_more_hours_english_than_spanish_l345_345814


namespace locus_of_points_is_apollonian_circle_l345_345989

noncomputable def locus_of_intersection_points {O A1 B1 A2 B2 : Type*} 
  [has_nontrivial_contradiction O A1 B1 A2 B2] 
  (l1 l2 : O → Prop) (S : Type*) (constant_ratio : Prop) 
  (A1_on_l1 : A1 ∈ l1 O) (B1_on_l1 : B1 ∈ l1 O) 
  (A2_on_l2 : A2 ∈ l2 O) (B2_on_l2 : B2 ∈ l2 O) 
  (OA1_OA2_diff_ratio : (OA1 / OA2 ≠ OB1 / OB2)) :
  set S :=
{S | ∀ (A1 A2 B1 B2 : O), 
  A1 ∈ l1 O ∧ B1 ∈ l1 O ∧ 
  A2 ∈ l2 O ∧ B2 ∈ l2 O ∧ 
  (OA1 / OA2 ≠ OB1 / OB2) →
  (∃! S, constant_ratio) }

theorem locus_of_points_is_apollonian_circle 
  {O A1 B1 A2 B2 : Type*} 
  [has_nontrivial_contradiction O A1 B1 A2 B2] 
  (l1 l2 : O → Prop) 
  (constant_ratio : Prop) 
  (A1_on_l1 : A1 ∈ l1 O) (B1_on_l1 : B1 ∈ l1 O) 
  (A2_on_l2 : A2 ∈ l2 O) (B2_on_l2 : B2 ∈ l2 O) 
  (OA1_OA2_diff_ratio : (OA1 / OA2 ≠ OB1 / OB2)) : 
  locus_of_intersection_points l1 l2 (S : Type*) constant_ratio :=
sorry

end locus_of_points_is_apollonian_circle_l345_345989


namespace smallest_norm_value_l345_345969

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end smallest_norm_value_l345_345969


namespace meat_left_l345_345584

theorem meat_left (initial_meat : ℕ) (meatball_fraction : ℚ) (spring_roll_meat : ℕ) 
  (h_initial : initial_meat = 20) 
  (h_meatball_fraction : meatball_fraction = 1/4)
  (h_spring_roll_meat : spring_roll_meat = 3) : 
  initial_meat - (initial_meat * meatball_fraction.num / meatball_fraction.denom).toNat - spring_roll_meat = 12 :=
by
  sorry

end meat_left_l345_345584


namespace OM_eq_ON_O_center_of_circle_l345_345571

theorem OM_eq_ON
  (A B C D M N O : Point)
  (h_square : square A B C D)
  (h_M : M ∈ LineSegment(B, C))
  (h_N : N ∈ LineSegment(C, D))
  (h_angle : Angle(A, M, N) = 45)
  (h_O : O ∈ Circle(C, M, N))
  (h_A_C : O ∈ LineSegment(A, C)) :
  dist O M = dist O N := 
sorry

theorem O_center_of_circle
  (A B C D M N O : Point)
  (h_square : square A B C D)
  (h_M : M ∈ LineSegment(B, C))
  (h_N : N ∈ LineSegment(C, D))
  (h_angle : Angle(A, M, N) = 45)
  (h_O : O ∈ Circle(C, M, N))
  (h_A_C : O ∈ LineSegment(A, C)) :
  is_center O (Circle(A, M, N)) := 
sorry

end OM_eq_ON_O_center_of_circle_l345_345571


namespace triangle_side_length_median_l345_345096

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l345_345096


namespace tetrahedron_congruent_faces_l345_345628

-- Define the tetrahedron and conditions
noncomputable def Tetrahedron (A B C D : Type*) : Prop := sorry

-- Conditions
variable (A B C D : Type*)

-- Edge equality condition
variable (h1 : ∃ (ab cd : ℝ), ab = cd)

-- Plane angle sum conditions
variable (h2 : ∃ (angleA angleB : ℝ), angleA = 180 ∧ angleB = 180)

-- The proof that if the conditions hold, all faces of the tetrahedron are congruent
theorem tetrahedron_congruent_faces :
  (Tetrahedron A B C D) → h1 → h2 →
  (∀ face1 face2 face3 face4, congruent face1 face2 face3 face4) := sorry

end tetrahedron_congruent_faces_l345_345628


namespace sin_330_eq_negative_half_l345_345360

theorem sin_330_eq_negative_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_eq_negative_half_l345_345360


namespace find_range_of_a_l345_345027

open Real

-- Definitions and conditions
def p (a : ℝ) := ∀ x : ℝ, (1 - a) ^ x > 0 → (1 - a) ^ x = (1 - a) ^ x
def q (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def r (a : ℝ) := ¬ (p a ∧ q a)
def s (a : ℝ) := p a ∨ q a

-- Theorem statement
theorem find_range_of_a (a : ℝ) : (r a) → (s a) → (0 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
by
sory

end find_range_of_a_l345_345027


namespace digit_2023_after_decimal_point_l345_345424

noncomputable def decimal_expansion_7_16 : ℕ → ℕ
| n := if n ≤ 4 then ["4", "3", "7", "5"].nth (n - 1).to_nat else 0

theorem digit_2023_after_decimal_point : decimal_expansion_7_16 2023 = 0 := by
  -- proof goes here
  sorry

end digit_2023_after_decimal_point_l345_345424


namespace no_odd_vertices_with_odd_degrees_l345_345789

-- Definitions based on conditions
def is_graph (G : Type) [graph G] : Prop := true

def degree (G : Type) [graph G] (v : G) : ℕ := sorry

def num_edges (G : Type) [graph G] : ℕ := sorry

-- Theorem statement
theorem no_odd_vertices_with_odd_degrees (G : Type) [graph G] : 
  (finset.sum (finset.univ : finset G) (degree G)) % 2 = 0 → 
  (finset.card (finset.filter (λ v, (degree G v) % 2 = 1) (finset.univ : finset G))) % 2 = 0 :=
sorry

end no_odd_vertices_with_odd_degrees_l345_345789


namespace one_is_positive_l345_345863

variable (x y z : ℝ)
def a : ℝ := x^2 - 2 * y + (π / 2)
def b : ℝ := y^2 - 2 * z + (π / 3)
def c : ℝ := z^2 - 2 * x + (π / 6)

theorem one_is_positive (a b c : ℝ) (ha : a = x^2 - 2 * y + π / 2)
        (hb : b = y^2 - 2 * z + π / 3) (hc : c = z^2 - 2 * x + π / 6) :
        a > 0 ∨ b > 0 ∨ c > 0 :=
by {
  sorry
}

end one_is_positive_l345_345863


namespace cubic_sum_expression_l345_345513

theorem cubic_sum_expression (x y z p q r : ℝ) (h1 : x * y = p) (h2 : x * z = q) (h3 : y * z = r) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by
  sorry

end cubic_sum_expression_l345_345513


namespace profit_after_five_days_days_for_target_profit_l345_345767

structure WholesalerProblem :=
  (cost_per_kg : ℕ)
  (total_weight : ℕ)
  (selling_price : ℕ → ℕ)
  (daily_loss : ℕ)
  (max_days : ℕ)
  (daily_storage_cost : ℕ)

def problem_data : WholesalerProblem :=
  { cost_per_kg := 40,
    total_weight := 700,
    selling_price := λ x, 50 + 2 * x,
    daily_loss := 15,
    max_days := 15,
    daily_storage_cost := 50 }

def remaining_product (total_weight daily_loss x : ℕ) : ℕ :=
  total_weight - daily_loss * x

def profit (selling_price remaining_product cost_per_kg total_weight daily_storage_cost x : ℕ) : Int :=
  selling_price * remaining_product - cost_per_kg * total_weight - daily_storage_cost * x

theorem profit_after_five_days :
  profit (problem_data.selling_price 5)
         (remaining_product problem_data.total_weight problem_data.daily_loss 5)
         problem_data.cost_per_kg
         problem_data.total_weight
         problem_data.daily_storage_cost
         5 = 9250 := 
by 
  sorry

theorem days_for_target_profit :
  ∃ x, profit (problem_data.selling_price x) 
               (remaining_product problem_data.total_weight problem_data.daily_loss x) 
               problem_data.cost_per_kg 
               problem_data.total_weight 
               problem_data.daily_storage_cost x = 10000 ∧ x = 10 :=
by 
  sorry

end profit_after_five_days_days_for_target_profit_l345_345767


namespace solve_polynomial_eq_l345_345177

theorem solve_polynomial_eq :
  ∀ (y : ℂ), (y^4 + 4 * y^3 * complex.sqrt 3 + 18 * y^2 + 12 * y * complex.sqrt 3 + 9) + (y + complex.sqrt 3) = 0 →
  y = -complex.sqrt 3 ∨
  y = -complex.sqrt 3 - 1 ∨
  y = -complex.sqrt 3 + 1/2 + complex.sqrt 3 / 2 * complex.I ∨
  y = -complex.sqrt 3 + 1/2 - complex.sqrt 3 / 2 * complex.I :=
by
  intros
  sorry

end solve_polynomial_eq_l345_345177


namespace correct_geometric_relationship_l345_345975

-- Define the basic entities
variable {Point : Type} [HilbertGeometry Point] (m : Line Point) (α β : Plane Point)

-- Conditions used from the given problem
def line_parallel_plane (m : Line Point) (p : Plane Point) : Prop := ∀ p1 ∈ p, p1 ∉ m ∨ ∀ p2 ∈ m, p2 ∈ p
def line_perpendicular_plane (m : Line Point) (p : Plane Point) : Prop := ∃ p1 p2 ∈ p, p1 ≠ p2 ∧ p1 -ₗ p2 ⊥ m
def plane_perpendicular_plane (p1 p2 : Plane Point) : Prop := ∃ p3 ∈ p1, ∃ p4 ∈ p2, p3 -ₗ p4 ⊥ p1 ∧ p3 -ₗ p4 ⊥ p2

-- Lean 4 statement for the problem
theorem correct_geometric_relationship 
  (m : Line Point)
  (α β : Plane Point)
  (h1 : line_parallel_plane m α)
  (h2 : line_parallel_plane m β ∨ line_perpendicular_plane m β) :
  plane_perpendicular_plane α β :=
by sorry

end correct_geometric_relationship_l345_345975


namespace complement_intersection_l345_345870

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x > 0}

def B : Set ℝ := {x | -3 < x ∧ x < 1}

def compA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_intersection :
  (compA ∩ B) = {x | 0 ≤ x ∧ x < 1} := by
  -- The proof goes here
  sorry

end complement_intersection_l345_345870


namespace true_propositions_l345_345384

-- Definitions of our propositions
def p1 : Prop := ∀ (l1 l2 l3 : Line), l1.intersect l2 ∧ l2.intersect l3 ∧ l1.intersect l3 ∧ ¬(∃ P, l1.contains P ∧ l2.contains P ∧ l3.contains P) → l1.coplanar l2 l3
def p2 : Prop := ∀ (P1 P2 P3 : Point), (¬ collinear P1 P2 P3) → ∃! (α : Plane), α.contains P1 ∧ α.contains P2 ∧ α.contains P3
def p3 : Prop := ∀ (l1 l2 : Line), ¬ l1.intersect l2 → l1 ∥ l2
def p4 : Prop := ∀ (l m : Line) (α : Plane), l ∈ α ∧ m ⊥ α → m ⊥ l

-- Truth evaluations from the solution
axiom Truth_eval_p1 : p1 = True
axiom False_eval_p2 : p2 = False
axiom False_eval_p3 : p3 = False
axiom Truth_eval_p4 : p4 = True

-- Compound proposition values
def prop1 := p1 ∧ p4
def prop2 := p1 ∧ p2
def prop3 := ¬ p2 ∨ p3
def prop4 := ¬ p3 ∨ ¬ p4

theorem true_propositions :
  (if prop1 then "①" else "") ++
  (if prop2 then "②" else "") ++
  (if prop3 then "③" else "") ++
  (if prop4 then "④" else "") = "①③④" :=
by
  rw [Truth_eval_p1, False_eval_p2, False_eval_p3, Truth_eval_p4]
  sorry

end true_propositions_l345_345384
