import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearOrder
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Calculus.Fderiv.Linear
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Binomial.MultiChoose
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean
import Mathlib.GroupTheory.Subgroup
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Real
import Set

namespace vertex_of_parabola_l791_791456

theorem vertex_of_parabola (c d: ℝ) (h: ∀ x: ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ 6 ≤ x)) :
  (1, 25) = vertex_of_parabola (-x^2 + c * x + d) :=
by
  sorry

end vertex_of_parabola_l791_791456


namespace functions_identical_1_functions_not_identical_2_l791_791226

section
variable {x : ℝ}

def f1 (x : ℝ) : ℝ := x / (x * x)
def g1 (x : ℝ) : ℝ := 1 / x

-- Problem 1: Prove that f1 and g1 are identical
theorem functions_identical_1 : ∀ x, x ≠ 0 → f1 x = g1 x :=
by
  sorry

def f2 (x : ℝ) : ℝ := x
def g2 (x : ℝ) : ℝ := Real.sqrt (x * x)

-- Problem 2: Prove that f2 and g2 are not identical
theorem functions_not_identical_2 : ∃ x, f2 x ≠ g2 x :=
by
  sorry
end

end functions_identical_1_functions_not_identical_2_l791_791226


namespace perpendicular_bisects_l791_791413

theorem perpendicular_bisects (A D P B C H M : Point)
(angle_BDA angle_BCA angle_BPH angle_MPH angle_MDP angle_MPD : ℝ) :
(Triangle A D P)
→ M = midpoint A D
→ H = foot_of_perpendicular P B C
→ ∠ B D A = ∠ B C A
→ ∠ B P H = ∠ M P H
→ ∠ M P H = 90
→ ∠ M D P = ∠ M P D
→ is_median P M A D → midpoint A D = M := sorry

end perpendicular_bisects_l791_791413


namespace standard_robot_weight_l791_791769

variable (S : ℕ) -- Define the variable for the standard robot's weight
variable (MaxWeight : ℕ := 210) -- Define the variable for the maximum weight of a robot, which is 210 pounds
variable (MinWeight : ℕ) -- Define the variable for the minimum weight of the robot

theorem standard_robot_weight (h1 : 2 * MinWeight ≥ MaxWeight) 
                             (h2 : MinWeight = S + 5) 
                             (h3 : MaxWeight = 210) :
  100 ≤ S ∧ S ≤ 105 := 
by
  sorry

end standard_robot_weight_l791_791769


namespace age_of_17th_student_is_75_l791_791796

variables (T A : ℕ)

def avg_17_students := 17
def avg_5_students := 14
def avg_9_students := 16
def total_17_students := 17 * avg_17_students
def total_5_students := 5 * avg_5_students
def total_9_students := 9 * avg_9_students
def age_17th_student : ℕ := total_17_students - (total_5_students + total_9_students)

theorem age_of_17th_student_is_75 :
  age_17th_student = 75 := by sorry

end age_of_17th_student_is_75_l791_791796


namespace tetrahedron_coloring_l791_791099

-- Define the problem as a Lean 4 statement.
theorem tetrahedron_coloring (h1 : is_regular_tetrahedron T) 
(h2 : split_face_into_four_congruent_equilateral_triangles T) 
(h3 : rotational_symmetry T) 
(h4 : use_two_colors T)
: distinct_colorings T = 48 :=
sorry

end tetrahedron_coloring_l791_791099


namespace centers_of_rectangles_form_rectangle_l791_791902

variables {A B C D O₁ O₂ O₃ O₄ : Type*}
  [cyclic_quadrilateral A B C D]
  [is_rectangle O₁ A B A opposite_side_of_CD]
  [is_rectangle O₂ B C B opposite_side_of_DA]
  [is_rectangle O₃ C D C opposite_side_of_AB]
  [is_rectangle O₄ D A D opposite_side_of_BC]

/-- The centers of the rectangles formed on each side of a cyclic quadrilateral form a rectangle. -/
theorem centers_of_rectangles_form_rectangle
  (h₁ : center O₁ (rectangle O₁))
  (h₂ : center O₂ (rectangle O₂))
  (h₃ : center O₃ (rectangle O₃))
  (h₄ : center O₄ (rectangle O₄)) :
  is_rectangle h₁ h₂ h₃ h₄ :=
sorry

end centers_of_rectangles_form_rectangle_l791_791902


namespace campers_afternoon_proof_campers_afternoon_l791_791194

-- Define the number of campers who went rowing in the morning
def morning_rowers : ℕ := 13

-- Define the total number of campers who went rowing
def total_rowers : ℕ := 34

-- Define the number of campers who went rowing in the afternoon
def afternoon_rowers : ℕ := total_rowers - morning_rowers

-- The theorem stating that the number of campers who went rowing in the afternoon is 21
theorem campers_afternoon (afternoon_rowers = 21) : Prop :=
  afternoon_rowers = 21

-- Now we need to prove this theorem
theorem proof_campers_afternoon : campers_afternoon (afternoon_rowers = 21) :=
begin
  sorry -- the proof is omitted
end

end campers_afternoon_proof_campers_afternoon_l791_791194


namespace striped_shorts_difference_l791_791356

variable (students : ℕ)
variable (striped_shirts checkered_shirts shorts : ℕ)

-- Conditions
variable (Hstudents : students = 81)
variable (Hstriped : striped_shirts = 2 * checkered_shirts)
variable (Hcheckered : checkered_shirts = students / 3)
variable (Hshorts : shorts = checkered_shirts + 19)

-- Goal
theorem striped_shorts_difference :
  striped_shirts - shorts = 8 :=
sorry

end striped_shorts_difference_l791_791356


namespace circle_tangent_to_x_axis_l791_791342

theorem circle_tangent_to_x_axis (b : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + 4 * x + 2 * b * y + c = 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ y : ℝ, y = -b ↔ y = 2)) ↔ (b = 2 ∨ b = -2) :=
sorry

end circle_tangent_to_x_axis_l791_791342


namespace angle_ZQY_measure_l791_791422

theorem angle_ZQY_measure (r : ℝ) (h1 : midpoint Z XY)
  (h2 : midpoint W YZ)
  (h3 : semi_circle XY)
  (h4 : semi_circle YZ)
  (h5 : equally_divides ZQ XY YZ) :
  ∠ZQY = 112.5 :=
sorry

end angle_ZQY_measure_l791_791422


namespace find_base_of_triangle_l791_791611

def triangle_base (area : ℝ) (height : ℝ) (base : ℝ) : Prop :=
  area = (base * height) / 2

theorem find_base_of_triangle : triangle_base 24 8 6 :=
by
  -- Simplification and computation steps are omitted as per the instruction
  sorry

end find_base_of_triangle_l791_791611


namespace problem_even_n_l791_791976

theorem problem_even_n (n : ℕ) (hn : n ≥ 2) :
  (∃ a : Fin n → ℝ, 
      (∑ i, a i = 0) ∧ 
      (∑ i, (a i)^2 = 1) ∧ 
      (∑ i, a i^3 * real.sqrt n = 2 * (finset.max_below (finset.range n) (λ i, a i) * real.sqrt n - 1))) ↔ 
  (∃ (k: ℕ), n = 2 * k) :=
begin
  sorry
end

end problem_even_n_l791_791976


namespace angle_CDM_eq_90_l791_791033

-- Define the trapezoid, with vertices A, B, C, D
variables {A B C D K M : Type} [AddGroup A] [AddGroup B] [AddGroup C]
           [AddGroup D] [AddGroup K] [AddGroup M]

-- Given conditions
-- AD and BC are parallel
variable (h_parallel : ∃ (v1 v2 : A), v1 ≠ v2 ∧ (∃ f : A → A, f v1 = v1 + v2) ∧ (∃ g : A → A, g v1 = v1 + v2))
-- AB = BC = BD
variable (h_eq1 : ∃ x y z : B, x = y ∧ y = z)
-- BK is the height from B, intersecting AC at M
variable (h_height : ∃ u v : K, u ≠ v ∧ ∃ l : K → K, l u = v)

-- Goal: prove ∠CDM = 90°
theorem angle_CDM_eq_90 
  (h_parallel : ∃ (v1 v2 : A), v1 ≠ v2 ∧ (∃ f : A → A, f v1 = v1 + v2) ∧ (∃ g : A → A, g v1 = v1 + v2))
  (h_eq1 : ∃ x y z : B, x = y ∧ y = z)
  (h_height : ∃ u v : K, u ≠ v ∧ ∃ l : K → K, l u = v) :
  ∠CDM = 90° :=
sorry

end angle_CDM_eq_90_l791_791033


namespace expand_expression_l791_791600

variables {R : Type*} [CommRing R] (x : R)

theorem expand_expression : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 :=
by sorry

end expand_expression_l791_791600


namespace sequence_upper_bound_l791_791644

theorem sequence_upper_bound (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, a (n + 1) = (1 + 2 * a n) / (1 + a n)) →
  ∀ n : ℕ, a (n + 1) < (1 + Real.sqrt 5) / 2 :=
by
  -- Base Case
  assume h1 : a 1 = 1,
  assume h2 : ∀ n : ℕ, a (n + 1) = (1 + 2 * a n) / (1 + a n),
  sorry

end sequence_upper_bound_l791_791644


namespace mean_computation_l791_791700

theorem mean_computation (x y : ℝ) 
  (h1 : (28 + x + 70 + 88 + 104) / 5 = 67)
  (h2 : (if x < 50 ∧ x < 62 then if y < 62 then ((28 + y) / 2 = 81) else ((62 + x) / 2 = 81) else if y < 50 then ((y + 50) / 2 = 81) else if y < 62 then ((50 + y) / 2 = 81) else ((50 + x) / 2 = 81)) -- conditions for median can be simplified and expanded as necessary
) : (50 + 62 + 97 + 124 + x + y) / 6 = 82.5 :=
sorry

end mean_computation_l791_791700


namespace product_of_distances_l791_791991

noncomputable def hyperbola := {P : ℝ × ℝ | P.1^2 - P.2^2 = 1}

def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

theorem product_of_distances (P : ℝ × ℝ) (hP : P ∈ hyperbola) (h_angle : real.angle (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) = real.pi / 2) :
  (real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2) :=
sorry

end product_of_distances_l791_791991


namespace calculate_interest_rate_l791_791561

theorem calculate_interest_rate 
  (A P : ℝ) (n t : ℕ) (r : ℝ) 
  (hA : A = 3087)
  (hP : P = 2800)
  (hn : n = 1)
  (ht : t = 2) :
  r = 0.05 :=
by
  have h1 : A = P * (1 + r / n) ^ (n * t) := sorry
  have h2 : 3087 = 2800 * (1 + r / 1) ^ (1 * 2) := sorry
  have h3 : 3087 / 2800 = (1 + r) ^ 2 := sorry
  have h4 : 1.1025 = (1 + r) ^ 2 := sorry
  have h5 : real.sqrt 1.1025 = 1 + r := sorry
  have h6 : 1.05 = 1 + r := sorry
  have h7 : 1.05 - 1 = r := sorry
  have h8 : 0.05 = r := sorry
  exact h8.symm

end calculate_interest_rate_l791_791561


namespace extremum_values_l791_791442

noncomputable def f (x : ℝ) := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  ∀ x : ℝ, -2 < x ∧ x < 2 → 
  (f(-1) = 5 ∧ ∀ y : ℝ, -2 < y ∧ y < 2 → f y > -∞) :=
by
  sorry

end extremum_values_l791_791442


namespace inclination_of_line_through_A_B_is_135_degrees_l791_791445

-- Define a point in 2D
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definitions for points A and B
def A : Point := { x := -2, y := 0 }
def B : Point := { x := -5, y := 3 }

-- Define the slope function
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

-- Define the inclination function
def inclination (m : ℝ) : ℝ :=
  if m = 0 then 0 else if m > 0 then 180 * real.arctan(m) / real.pi else 180 + 180 * real.arctan(m) / real.pi

-- Problems' main statement
theorem inclination_of_line_through_A_B_is_135_degrees :
  inclination (slope A B) = 135 :=
by
  sorry

end inclination_of_line_through_A_B_is_135_degrees_l791_791445


namespace min_diagonal_of_rectangle_l791_791701

theorem min_diagonal_of_rectangle (l w : ℝ) (h1 : 2 * l + 2 * w = 30) : 
  exists (d : ℝ), d = real.sqrt 112.5 ∧ (∀ l w, 2 * l + 2 * w = 30 → real.sqrt (l^2 + w^2) ≥ d) := sorry

end min_diagonal_of_rectangle_l791_791701


namespace grains_of_rice_in_teaspoon_is_10_l791_791469

noncomputable def grains_of_rice_per_teaspoon : ℕ :=
  let grains_per_cup := 480
  let tablespoons_per_half_cup := 8
  let teaspoons_per_tablespoon := 3
  grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)

theorem grains_of_rice_in_teaspoon_is_10 : grains_of_rice_per_teaspoon = 10 :=
by
  sorry

end grains_of_rice_in_teaspoon_is_10_l791_791469


namespace revenue_95_percent_l791_791002

-- Definitions based on the conditions
variables (C : ℝ) (n : ℝ)
def revenue_full : ℝ := 1.20 * C
def tickets_sold_percentage : ℝ := 0.95

-- Statement of the theorem based on the problem translation
theorem revenue_95_percent (C : ℝ) :
  (tickets_sold_percentage * revenue_full C) = 1.14 * C :=
by
  sorry -- Proof to be provided

end revenue_95_percent_l791_791002


namespace F_six_eq_seven_l791_791749

def F : ℕ → ℝ
| 1 := a
| 2 := a
| 3 := a
| (n + 1) := (F n * F (n - 1) + 1) / (F (n - 2))

theorem F_six_eq_seven (a : ℝ) : F 6 = 7 :=
by sorry

end F_six_eq_seven_l791_791749


namespace exists_point_in_interior_with_bound_l791_791066

-- Define the closed unit disk D
def D : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 ≤ 1}

-- Ensure f is defined on D and in the range [-1, 1]
def f (p : ℝ × ℝ) : ℝ := 
  sorry -- f is not explicitly defined, so it is left as a sorry placeholder

-- Define the partial derivatives of f
def f1 (p : ℝ × ℝ) : ℝ := 
  sorry -- ∂f/∂x is not explicitly defined

def f2 (p : ℝ × ℝ) : ℝ := 
  sorry -- ∂f/∂y is not explicitly defined

-- Conditions for f in D
axiom f_in_D : ∀ (p : ℝ × ℝ), p ∈ D → -1 ≤ f p ∧ f p ≤ 1

-- Define the interior of D
def interior_D : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 < 1}

-- Main theorem statement
theorem exists_point_in_interior_with_bound :
  ∃ (p : ℝ × ℝ), p ∈ interior_D ∧ (f1 p) ^ 2 + (f2 p) ^ 2 < 16 :=
by
  sorry -- Proof is skipped

end exists_point_in_interior_with_bound_l791_791066


namespace right_triangle_45_45_90_altitude_l791_791431

theorem right_triangle_45_45_90_altitude:
  ∀ (A B C D : Type) 
  (AB BC AC : ℝ)
  (h : ℝ),
  -- Conditions
  (right_angle : B = 90) 
  (angle_A : A = 45) 
  (angle_C : C = 45)
  (BD : ℝ) (BD = 4)

  (h = BD * sqrt 2) 
  -- Proof goals: Hypotenuse and area
  (hypotenuse_length : h = 8)
  (triangle_area : (h^2 / 4) = 16) :=
  sorry

end right_triangle_45_45_90_altitude_l791_791431


namespace charge_paid_by_b_l791_791174

theorem charge_paid_by_b (T : ℕ) (h_a h_b h_c : ℕ) (h_total : T = 520) (ha_cond : h_a = 7) (hb_cond : h_b = 8) (hc_cond : h_c = 11) :
  let total_hours := h_a + h_b + h_c,
      fraction_b := h_b / total_hours,
      charge_b := T * fraction_b
  in charge_b = 160 :=
by
  sorry

end charge_paid_by_b_l791_791174


namespace problem_statement_l791_791864

notation "→" => Vector -- Assuming this notation represents vector as needed.

variable (A B C D P Q : Type)
variable (AB PA BQ PC BA QC PQ QC QP CQ : A → B → Vector)
variable (PQ : Vector)

-- Conditions
def exprA : Vector := AB + (PA + BQ)
def exprB : Vector := (AB + PC) + (BA - QC)
def exprC : Vector := QC - QP + CQ
def exprD : Vector := PA + AB - BQ

-- Problem Statement
theorem problem_statement :
  (exprA = PQ) ∧ (exprB = PQ) ∧ (exprC = PQ) → ¬(exprD = PQ) :=
sorry

end problem_statement_l791_791864


namespace servings_in_one_week_l791_791973

theorem servings_in_one_week (daily_servings : ℕ) (days_in_week : ℕ) (total_servings : ℕ)
  (h1 : daily_servings = 3)
  (h2 : days_in_week = 7)
  (h3 : total_servings = daily_servings * days_in_week) :
  total_servings = 21 := by
  sorry

end servings_in_one_week_l791_791973


namespace edge_length_of_cut_off_cube_l791_791633

theorem edge_length_of_cut_off_cube (V_large V_remaining : ℕ) 
  (h1 : V_large = 1000) (h2 : V_remaining = 488) : 
  ∃ x : ℕ, x ^ 3 = 64 ∧ 1000 - 8 * x ^ 3 = V_remaining := by
    use 4
    split
    · norm_num
    · exact h2.symm
appointment

end edge_length_of_cut_off_cube_l791_791633


namespace min_distinct_people_l791_791467

/--
There are 10 photos, each showing three people from left to right: a son, a father, and the father's sibling.
Given that the ten individuals in the middle of the photos (the fathers) are all different.
Prove that the minimum number of distinct people appearing in these 10 photos is 16.
-/
theorem min_distinct_people (photos : Fin 10 → (ℕ × ℕ × ℕ))
  (distinct_fathers : ∀ (i j : Fin 10), i ≠ j → photos i.2 ≠ photos j.2) :
  ∃ n, n = 16 :=
by
  -- Conditions: 10 photos with distinct fathers
  have h_distinct_fathers : ∀ (i j : Fin 10), i ≠ j → photos i.2 ≠ photos j.2 := distinct_fathers
  -- Conclusion: The minimum number of distinct people is 16
  use 16
  sorry

end min_distinct_people_l791_791467


namespace anna_savings_percent_l791_791223

theorem anna_savings_percent :
  let original_jacket_price := 100
  let original_shirt_price := 50
  let original_pants_price := 70
  let discount_jacket := 0.30
  let discount_shirt := 0.50
  let discount_pants := 0.25
  let original_total_price := original_jacket_price + original_shirt_price + original_pants_price
  let savings_jacket := original_jacket_price * discount_jacket
  let savings_shirt := original_shirt_price * discount_shirt
  let savings_pants := original_pants_price * discount_pants
  let total_savings := savings_jacket + savings_shirt + savings_pants
  let percent_saved := (total_savings / original_total_price) * 100
  percent_saved ≈ 32.95 := 
by sorry

end anna_savings_percent_l791_791223


namespace meaningful_fraction_iff_l791_791697

theorem meaningful_fraction_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (2 - x)) ↔ x ≠ 2 := by
  sorry

end meaningful_fraction_iff_l791_791697


namespace find_x_l791_791692

theorem find_x : ∃ x : ℝ, 45 - (28 - (37 - (x - 15))) = 54 ∧ x = 15 :=
begin
  use 15,
  split,
  { -- proving the condition
    norm_num,
  },
  { -- proving the solution is correct
    refl,
  }
end

end find_x_l791_791692


namespace compute_exp_l791_791944

theorem compute_exp : 3 * 3^4 + 9^30 / 9^28 = 324 := 
by sorry

end compute_exp_l791_791944


namespace A_lt_one_l791_791385

def A := (Finset.range (2 ^ 11 - 2 ^ 10)).sum (λ k, (1 / (2 ^ 10 + k)))

theorem A_lt_one : A < 1 :=
by sorry

end A_lt_one_l791_791385


namespace number_of_isosceles_cross_sections_l791_791480

theorem number_of_isosceles_cross_sections (A B C D : Type) [RegularTetrahedron A B C D] : 
  ∃ n : ℕ, n = 18 ∧ 
           (∀ T : IsoscelesTriangle, 
            ∃ plane : Plane, 
            plane.contains_points {A, X, Y} ∧ 
            angle_between_plane_and_base plane BCD = 75) :=
sorry

end number_of_isosceles_cross_sections_l791_791480


namespace PQ_passes_through_H_l791_791285

namespace Geometry

variables {A B C D P Q H : Type}

axiom cyclic_quadrilateral (A B C D : Type) : Prop
axiom symmetric_points (C P Q : Type) (AB AD : Type) : Prop
axiom orthocenter (H : Type) (triangle : Type) : Prop
axiom line_passes_through (line point1 point2 : Type) : Prop

theorem PQ_passes_through_H
  (ABCD_cyclic : cyclic_quadrilateral A B C D)
  (P_symmetric : symmetric_points C P P AB)
  (Q_symmetric : symmetric_points C Q Q AD)
  (H_orthocenter : orthocenter H (triangle A B D)) :
  ∃ L : Type, line_passes_through L P Q ∧ line_passes_through L H :=
sorry

end Geometry

end PQ_passes_through_H_l791_791285


namespace interval_of_x_l791_791264

theorem interval_of_x (x : ℝ) : 
  (2 < 3 * x ∧ 3 * x < 3) ∧ 
  (2 < 4 * x ∧ 4 * x < 3) ↔ 
  x ∈ set.Ioo (2 / 3) (3 / 4) := 
sorry

end interval_of_x_l791_791264


namespace both_owners_count_l791_791574

-- Define the sets and counts as given in the conditions
variable (total_students : ℕ) (rabbit_owners : ℕ) (guinea_pig_owners : ℕ) (both_owners : ℕ)

-- Assume the values given in the problem
axiom total : total_students = 50
axiom rabbits : rabbit_owners = 35
axiom guinea_pigs : guinea_pig_owners = 40

-- The theorem to prove
theorem both_owners_count : both_owners = rabbit_owners + guinea_pig_owners - total_students := by
  sorry

end both_owners_count_l791_791574


namespace rational_coefficients_count_l791_791858

theorem rational_coefficients_count : 
  ∃ n, n = 84 ∧ ∀ k, (0 ≤ k ∧ k ≤ 500) → 
            (k % 3 = 0 ∧ (500 - k) % 2 = 0) → 
            n = 84 :=
by
  sorry

end rational_coefficients_count_l791_791858


namespace find_AX_length_l791_791733

variables (A B C X : Point) (AC BC CX AX XB : ℝ)

noncomputable def AX_length : Prop :=
  ∃ (AX XB : ℝ), 
  AX / XB = 3 / 2 ∧ AC = 27 ∧ BC = 35 ∧ CX = 30 ∧ AX = 45

theorem find_AX_length :
  AX_length A B C X AC BC CX AX XB := sorry

end find_AX_length_l791_791733


namespace allocate_square_plots_l791_791507

theorem allocate_square_plots (x y : ℤ) (h : x > y) :
  ∃ u v : ℤ, u^2 + v^2 = 2 * (x^2 + y^2) :=
by {
  use [x + y, x - y],
  -- sorry can be used to skip the actual detailed proof which is not required here.
  sorry
}

end allocate_square_plots_l791_791507


namespace gcf_60_90_l791_791487

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l791_791487


namespace find_roots_approximation_l791_791268

/--
Given the equation (∛3)^x = x, we are to show that the roots, approximated to two decimal places, are x₁ = 1 and x₂ = 2.48.
-/
theorem find_roots_approximation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2.48 ∧
  ((real.cbrt 3) ^ x₁ = x₁) ∧ ((real.cbrt 3) ^ x₂ = x₂) :=
by
  -- Proof skipped
  sorry

end find_roots_approximation_l791_791268


namespace allocate_plots_l791_791509

theorem allocate_plots (x y : ℕ) (h : x > y) : 
  ∃ u v : ℕ, (u^2 + v^2 = 2 * (x^2 + y^2)) :=
by
  sorry

end allocate_plots_l791_791509


namespace vector_b_value_l791_791683

theorem vector_b_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -2)
  2 • a + b = (3, 2) → b = (1, -2) :=
by
  intros
  sorry

end vector_b_value_l791_791683


namespace find_values_of_a_l791_791961

noncomputable def f : ℝ → ℝ :=
λ x, if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
     else if x ≥ 0 then Real.exp (x - 1)
     else 0  -- here we add a default value for f(x) when x is not in the domain for safety.

theorem find_values_of_a : {a : ℝ | f a = 1} = {1, -Real.sqrt 2 / 2} :=
by  
   sorry

end find_values_of_a_l791_791961


namespace area_ratio_l791_791811

theorem area_ratio (l b r : ℝ) (h1 : l = 2 * b) (h2 : 6 * b = 2 * π * r) :
  (l * b) / (π * r ^ 2) = 2 * π / 9 :=
by {
  sorry
}

end area_ratio_l791_791811


namespace find_b2_l791_791817

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 25) (h10 : b 10 = 105)
  (h_arith : ∀ n, n ≥ 3 → b n = (∑ i in finset.range (n-1).succ, b i) / (n-1)) :
  b 2 = 185 :=
sorry

end find_b2_l791_791817


namespace profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l791_791026

noncomputable def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def p (x : ℕ) : ℝ := R x - C x
noncomputable def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_function_is_correct : ∀ x, p x = -20 * x^2 + 2500 * x - 4000 := 
by 
  intro x
  sorry

theorem marginal_profit_function_is_correct : ∀ x, 0 < x ∧ x ≤ 100 → Mp x = -40 * x + 2480 := 
by 
  intro x
  sorry

theorem profit_function_max_value : ∃ x, (x = 62 ∨ x = 63) ∧ p x = 74120 :=
by 
  sorry

theorem marginal_profit_function_max_value : ∃ x, x = 1 ∧ Mp x = 2440 :=
by 
  sorry

theorem profit_and_marginal_profit_max_not_equal : ¬ (∃ x y, (x = 62 ∨ x = 63) ∧ y = 1 ∧ p x = Mp y) :=
by 
  sorry

end profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l791_791026


namespace determine_triangle_area_l791_791444

noncomputable def triangle_area_proof : Prop :=
  let height : ℝ := 2
  let angle_ratio : ℝ := 2 / 1
  let smaller_base_part : ℝ := 1
  let larger_base_part : ℝ := 7 / 3
  let base := smaller_base_part + larger_base_part
  let area := (1 / 2) * base * height
  area = 11 / 3

theorem determine_triangle_area : triangle_area_proof :=
by
  sorry

end determine_triangle_area_l791_791444


namespace range_of_a_eq_two_distinct_real_roots_l791_791310

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 / 4) * x + 1 else Real.log x

theorem range_of_a_eq_two_distinct_real_roots :
  ∃ (a : ℝ), (a ∈ set.Ioo (1 / 4) (1 / Real.exp 1)) ∧ 
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ ∧ f x₂ = a * x₂) :=
sorry

end range_of_a_eq_two_distinct_real_roots_l791_791310


namespace trig_problem_l791_791276

-- Translate the conditions and problems into Lean 4:
theorem trig_problem (α : ℝ) (h1 : Real.tan α = 2) :
    (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end trig_problem_l791_791276


namespace polygon_never_returns_to_initial_state_l791_791920

-- Define the irregular polygon inscribed in a circle.
variables {n : ℕ} (a : fin n → ℝ)

-- Condition that a vertex is unstable if it divides its corresponding arc into two unequal parts.
def unstable_vertex (a : fin n → ℝ) (i : fin n) : Prop :=
  (a i + a (i + 1)) / 2 ≠ a i

-- Define the transformation applied every second.
def transform (a : fin n → ℝ) : fin n → ℝ :=
  λ i, (a i + a (i + 1)) / 2

-- Formal proof that no matter how many seconds pass, the polygon will never be the same as the initial polygon.
theorem polygon_never_returns_to_initial_state (a : fin n → ℝ) :
  ∀ t : ℕ, transform^[t] a ≠ a := 
sorry

end polygon_never_returns_to_initial_state_l791_791920


namespace naomi_regular_bikes_l791_791929
-- Import necessary libraries

-- Define the condition and the proof problem
theorem naomi_regular_bikes (R C : ℕ) (h1 : C = 11) 
  (h2 : 2 * R + 4 * C = 58) : R = 7 := 
  by 
  -- Include all necessary conditions as assumptions
  have hC : C = 11 := h1
  have htotal : 2 * R + 4 * C = 58 := h2
  -- Skip the proof itself
  sorry

end naomi_regular_bikes_l791_791929


namespace cos_angle_neg_seq_l791_791245

theorem cos_angle_neg_seq (α : ℝ) : 
  (∀ (n : ℕ), real.cos (2^n * α) < 0) ↔ 
  ∃ (k : ℤ), α = 2*k*real.pi + (2/3*real.pi) ∨ α = 2*k*real.pi - (2/3*real.pi) :=
by
  sorry

end cos_angle_neg_seq_l791_791245


namespace even_product_of_sums_l791_791767

def cards : List (ℕ × ℕ) :=
  [(1, 2), (3, 4), (5, 6), ..., (99, 98)]  -- Assume this list is filled correctly

theorem even_product_of_sums :
  ∃ (a b : Fin 99 → ℕ), 
    (∀ i, (i < 50 → odd (a i) ∧ b i ∈ { x | x % 2 = 0 ∨ x % 2 ≠ 0 }) ∧ 
           (i ≥ 50 → even (a i) ∧ b i ∈ { x | x % 2 = 0 ∨ x % 2 ≠ 0 })) ∧
    even (∏ i in Finset.range 99, (a i + b i)) :=
by 
  sorry

end even_product_of_sums_l791_791767


namespace father_l791_791892

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S) (h2 : F + 15 = 2 * (S + 15)) : F = 45 :=
sorry

end father_l791_791892


namespace length_of_AB_l791_791070

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_AB :
  let O := (0, 0)
  let A := (54^(1/3), 0)
  let B := (0, 54^(1/3))
  distance A B = 54^(1/3) * Real.sqrt 2 :=
by
  sorry

end length_of_AB_l791_791070


namespace hcf_36_84_l791_791518

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := 
by 
s -- This indicates that the proof is omitted

end hcf_36_84_l791_791518


namespace boys_in_first_group_l791_791505

theorem boys_in_first_group (x : ℕ) (h₁ : 5040 = 360 * x) : x = 14 :=
by {
  sorry
}

end boys_in_first_group_l791_791505


namespace find_a_find_b_find_c_find_d_l791_791184

def tetrahedron_OA_OB_OC : Prop :=
  let O := (0 : ℝ, 0, 0)
  let A := (6 * x : ℝ, 0, 0)
  let B := (0 : ℝ, 6 * x, 0)
  let C := (0 : ℝ, 0, 6 * x)
  ∃ (OA OB OC : ℝ),
    OA = 6 * x ∧ OB = 6 * x ∧ OC = 6 * x ∧
    (A.1 - O.1) * (B.2 - O.2) * (C.3 - O.3) ≠ 0

theorem find_a (h : tetrahedron_OA_OB_OC) : ∃ (a : ℝ), V = 36 * x^3 :=
by
  sorry

theorem find_b (h : tetrahedron_OA_OB_OC) : ∃ (b : ℝ), area_ABC = 18 * sqrt(3) * x^2 :=
by
  sorry

theorem find_c (h : tetrahedron_OA_OB_OC) : ∃ (c : ℝ), dist_O_to_ABC = 2 * sqrt(3) * x :=
by
  sorry

theorem find_d (h : tetrahedron_OA_OB_OC) : ∃ (d : ℝ), sin_theta = sqrt(d) / 3 :=
by
  sorry

end find_a_find_b_find_c_find_d_l791_791184


namespace f_2009_equals_4021_l791_791546

-- Define the function and conditions
def f (x : ℝ) : ℝ

-- Given condition for the function f
axiom functional_equation : ∀ x y : ℝ, f(x) * f(y) - f(2 * x * y + 3) + 3 * f(x + y) - 3 * f(x) = -6 * x

-- Prove that f(2009) = 4021
theorem f_2009_equals_4021 : f 2009 = 4021 := 
sorry

end f_2009_equals_4021_l791_791546


namespace solve_for_x_l791_791010

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l791_791010


namespace total_minutes_l791_791018

def count_minutes (h : ℕ) : ℕ :=
  if h = 0 then 0 else h

theorem total_minutes : ∑ h in Finset.range 12, count_minutes h = 66 :=
by
  sorry

end total_minutes_l791_791018


namespace max_constant_l791_791527

theorem max_constant (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i in Finset.range n, a i ^ 2) * (∑ i in Finset.range n, ∑ j in Finset.range n, max i j * a i * a j) ≥
  (π ^ 2 / 16) * (∑ i in Finset.range n, a i) ^ 4 :=
by sorry

end max_constant_l791_791527


namespace tim_income_less_than_juan_l791_791762

-- Definitions of the conditions
variables {T J M : ℝ}
def mart_income_condition1 (M T : ℝ) : Prop := M = 1.40 * T
def mart_income_condition2 (M J : ℝ) : Prop := M = 0.84 * J

-- The proof goal
theorem tim_income_less_than_juan (T J M : ℝ) 
(h1: mart_income_condition1 M T) 
(h2: mart_income_condition2 M J) : 
T = 0.60 * J :=
by
  sorry

end tim_income_less_than_juan_l791_791762


namespace final_dollars_final_euros_final_rubles_l791_791568

-- Defining constants and initial amounts
def initial_euros : ℝ := 3000
def initial_dollars : ℝ := 4000
def initial_rubles : ℝ := 240000

def interest_rate_eur : ℝ := 0.021
def interest_rate_usd : ℝ := 0.021
def interest_rate_rub : ℝ := 0.079

def sell_rate_eur_to_rub : ℝ := 60.10
def buy_rate_rub_to_usd : ℝ := 58.90
def sell_rate_usd_to_rub : ℝ := 58.50
def buy_rate_rub_to_eur : ℝ := 61.20

-- Function to compound interest yearly
def compound_interest (principal rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

-- After year 1
def year1_euros : ℝ := compound_interest initial_euros interest_rate_eur 1
def year1_dollars : ℝ := compound_interest initial_dollars interest_rate_usd 1
def year1_rubles : ℝ := compound_interest initial_rubles interest_rate_rub 1

-- Conversion after year 1
def euros_sold : ℝ := 1000
def rubles_from_euros : ℝ := euros_sold * sell_rate_eur_to_rub
def dollars_purchased : ℝ := rubles_from_euros / buy_rate_rub_to_usd

def year2_euros : ℝ :=
  compound_interest (year1_euros - euros_sold) interest_rate_eur 1
def year2_dollars : ℝ :=
  compound_interest (year1_dollars + dollars_purchased) interest_rate_usd 1
def year2_rubles : ℝ :=
  compound_interest year1_rubles interest_rate_rub 1

-- Conversion after year 2
def dollars_sold : ℝ := 2000
def rubles_from_dollars : ℝ := dollars_sold * sell_rate_usd_to_rub
def euros_purchased : ℝ := rubles_from_dollars / buy_rate_rub_to_eur

def year3_euros : ℝ :=
  compound_interest (year2_euros + euros_purchased) interest_rate_eur 1
def year3_dollars : ℝ :=
  compound_interest (year2_dollars - dollars_sold) interest_rate_usd 1
def year3_rubles : ℝ :=
  compound_interest year2_rubles interest_rate_rub 1

-- Statements to prove
theorem final_dollars : round year3_dollars = 3286 := by sorry
theorem final_euros : round year3_euros = 4040 := by sorry
theorem final_rubles : round year3_rubles = 301504 := by sorry

end final_dollars_final_euros_final_rubles_l791_791568


namespace largest_five_digit_number_is_99200_l791_791850

def largest_five_digit_number_sum_20 (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧ (n.digits.sum = 20)

theorem largest_five_digit_number_is_99200 : ∃ n : ℕ, largest_five_digit_number_sum_20 n ∧ n = 99200 :=
by
  sorry

end largest_five_digit_number_is_99200_l791_791850


namespace probability_floor_sqrt_50x_given_floor_sqrt_x_eq_11_zero_l791_791062

open Real

noncomputable def probability_of_floor_sqrt_50x_conditioned_floor_sqrt_x_eq_11 : ℝ → Prop :=
λ x, if x ∈ set.Icc 50 150 then
    if floor (sqrt x) = 11 then
        (if floor (sqrt (50 * x)) = 110 then 0 else 1)
    else 0
else 0

theorem probability_floor_sqrt_50x_given_floor_sqrt_x_eq_11_zero :
  ∀ x : ℝ, probability_of_floor_sqrt_50x_conditioned_floor_sqrt_x_eq_11 x = 0 :=
by
  intro x
  unfold probability_of_floor_sqrt_50x_conditioned_floor_sqrt_x_eq_11
  -- we need to handle details about x's range, but the overall proof is left out with "sorry"
  sorry

end probability_floor_sqrt_50x_given_floor_sqrt_x_eq_11_zero_l791_791062


namespace number_of_valid_b_l791_791619

noncomputable def count_valid_b (x : ℕ) (b : ℕ) : Prop :=
  (3 * x > 4 * x - 4) ∧ (4 * x - b > -8)

theorem number_of_valid_b :
  ∃ (b_values : Finset ℕ), (∀ b ∈ b_values, count_valid_b 3 b) ∧
  (¬ ∃ (b ∈ b_values) (x : ℕ), x ≠ 3 ∧ count_valid_b x b) ∧ 
  b_values.card = 4 :=
by
  sorry

end number_of_valid_b_l791_791619


namespace Cubs_win_World_Series_probability_l791_791104

noncomputable def prob_Cubs_win_series : ℚ := 
  ∑ k in finset.range 4, 
    (nat.choose (3 + k) k) * (4 / 5)^4 * (1 / 5)^k

theorem Cubs_win_World_Series_probability : 
  (prob_Cubs_win_series * 100).round = 97 := 
by
  sorry

end Cubs_win_World_Series_probability_l791_791104


namespace sound_intensity_relationship_l791_791155

def sound_intensity_to_decibel (I : ℝ) : ℝ := 10 * log10 I + 120

theorem sound_intensity_relationship
  (I1 I2 I3 : ℝ)
  (h1 : 10^(-3) < I1 ∧ I1 < 10^(-2))
  (h2 : 10^(-2) < I2 ∧ I2 < 10^(-1))
  (h3 : 10^(1.5) < I3 ∧ I3 < 10^(2.5)) :
  I1 * I3 > I2^2 :=
sorry

end sound_intensity_relationship_l791_791155


namespace ball_distribution_l791_791770

theorem ball_distribution :
  ∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ x1 x2 x3, f x1 x2 x3 → x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) ∧
    (∃ (count : ℕ), (count = 15) ∧ (∀ x1 x2 x3, f x1 x2 x3 → count = 15)) :=
sorry

end ball_distribution_l791_791770


namespace guards_return_to_original_post_l791_791543

theorem guards_return_to_original_post (
  (guards : Fin 12 → ℝ) :
  ∀ i, guards.times(0) = 0⟩
  (period: ℝ) : period = (60 : ℝ)
  
  (velocities: Fin 12 → ℝ): 
  ∀ i, velocities i = (period / 60)

  (rules: ℝ → Fin 12 → Fin 12): 
  (t: ℝ) 
  ∀ i, 
  (if rules(t) = true 
  then guards i) after (i.e.,rules)? 
  else change direction 

  (time: ℝ) : time = (12 /2 : ℝ))

  ∀ (t : ℝ), t = (60/2 : ℝ) → 
  ∀ i, 
  guards(t) = guards(times(0))
  
  := 
by 
  sorry

end guards_return_to_original_post_l791_791543


namespace smallest_positive_integer_l791_791862
-- Import the required library

-- State the problem in Lean
theorem smallest_positive_integer (x : ℕ) (h : 5 * x ≡ 17 [MOD 31]) : x = 13 :=
sorry

end smallest_positive_integer_l791_791862


namespace problem_statement_l791_791300

variables (a b : ℝ) (h_a : a > 0) (h_b : b > 0)

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus_of_parabola : ℝ × ℝ := (1, 0)
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def P : ℝ × ℝ := (3/2, real.sqrt 6)

theorem problem_statement :
  hyperbola (P.1) (P.2) ∧ parabola (P.1) (P.2) ∧ 
  distance P focus_of_parabola = 5/2 ∧
  ( ∀ x y, hyperbola x y → x / (a) = y / (b) ∧ x / (a) = -y / (b)) :=
sorry

end problem_statement_l791_791300


namespace line_passing_through_intersection_and_origin_l791_791682

theorem line_passing_through_intersection_and_origin :
  ∃ (l : ℝ → ℝ → Prop), 
  ((∀ x y, l (2 * x + y - 3 = 0) (x + 4 * y - 2 = 0)) ∧ 
   (∀ x y, l (x - 10 * y = 0) ((x, y) = (0, 0)))) := 
sorry

end line_passing_through_intersection_and_origin_l791_791682


namespace triangle_altitudes_problem_l791_791038

variable {A B C P Q H : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C] [MetricSpace P] [MetricSpace Q] [MetricSpace H]
variables (HP HQ : ℕ) (AH BH BP PC AQ QC : ℕ)

axiom altitudes_intersect : ∃ H, hp H = 6 ∧ hq H = 3

theorem triangle_altitudes_problem
  (H_ap : HP = 6)
  (H_hq : HQ = 3)
  (HP_def : HP = 6)
  (HQ_def : HQ = 3)
  (assumption_on_ratios : ∀ (AH BH : ℕ), BH / 6 = AH / 3 → BH = 2 * AH):
  (BP * PC - AQ * QC = 27) :=
by
  refine ⟨27, _⟩
  sorry

end triangle_altitudes_problem_l791_791038


namespace find_g_inv_f_neg_10_l791_791926

variable {α : Type*} {β : Type*} [invFun : Inverse β α]

theorem find_g_inv_f_neg_10 (f g : α → β) 
  (hf : Function.LeftInverse f g) 
  (hg : Function.RightInverse f g) 
  (h_inv : ∀ x, f (g x) = 3 * x - 2) :
  g⁻¹ (f (-10 : α)) = (-8 / 3 : α) :=
sorry

end find_g_inv_f_neg_10_l791_791926


namespace distinct_paths_in_4x4_grid_with_corner_removed_solution_l791_791233

theorem distinct_paths_in_4x4_grid_with_corner_removed : ℕ :=
  let grid : List (List (Option ℕ)) := [
    [some 1, some 1, some 1, some 1],
    [some 1, some 1, some 1, some 1],
    [some 1, some 1, some 1, some 1],
    [none, some 1, some 1, some 1]
  ]
  -- Define the recursive counting function
  let rec countPaths (currentSquare : ℕ × ℕ) (visitedSquares : List (ℕ × ℕ)) : ℕ :=
    if visitedSquares.length = 15 then 1
    else
      let moves := [(-1, 0), (1, 0), (0, -1), (0, 1)] -- up, down, left, right
      let neighbors := moves.map (λ move => 
                                let newSquare := (currentSquare.1 + move.1, currentSquare.2 + move.2)
                                if newSquare.1 < 0 ∨ newSquare.1 > 3 ∨ newSquare.2 < 0 ∨ newSquare.2 > 3 
                                   ∨ (newSquare = (3, 0)) ∨ (newSquare ∈ visitedSquares)
                                then none
                                else some newSquare
                              )
      let validNeighbors := neighbors.filterMap id
      validNeighbors.map (λ neighbor => countPaths neighbor (neighbor :: visitedSquares)).sum

  let initialSquares := [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), 
                         (1,3), (2,0), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
  
  initialSquares.map (λ start => countPaths start [start]).sum

theorem solution: distinct_paths_in_4x4_grid_with_corner_removed = 14007 :=
  sorry

end distinct_paths_in_4x4_grid_with_corner_removed_solution_l791_791233


namespace find_length_of_DE_l791_791016

noncomputable def length_of_DE {D E F : Type*} [EuclideanSpace F] (EF DF : ℝ) (hEF : EF = 8) (hDF : DF = 5)
  (hMedPerp : ∀ M N : F, midpoint D E M ∧ midpoint E F N → ⟪D - M, E - N⟫ = 0) : ℝ :=
  Real.sqrt(89 / 5)

theorem find_length_of_DE {D E F : Type*} [EuclideanSpace F] (EF DF : ℝ) (hEF : EF = 8) (hDF : DF = 5)
  (hMedPerp : ∀ M N : F, midpoint D E M ∧ midpoint E F N → ⟪D - M, E - N⟫ = 0) :
  length_of_DE EF DF hEF hDF hMedPerp = Real.sqrt (89 / 5) :=
sorry

end find_length_of_DE_l791_791016


namespace series_convergence_l791_791410

noncomputable def P_j (x z : ℂ) (j : ℕ) : ℂ :=
  (∏ k in range j, (1 - z * x^k)) / (∏ k in range j, (z - x^(k + 1)))

theorem series_convergence (x z : ℂ) (h_x : |x| < 1) (h_z : |z| > 1) :
  1 + ∑ j in range ∞, (1 + x^j) * P_j x z j = 0 :=
sorry

end series_convergence_l791_791410


namespace range_of_a_for_circle_l791_791799

theorem range_of_a_for_circle (a : ℝ) : 
  -2 < a ∧ a < 2/3 ↔ 
  ∃ (x y : ℝ), (x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1) = 0 :=
sorry

end range_of_a_for_circle_l791_791799


namespace fruit_seller_prices_l791_791203

theorem fruit_seller_prices:
  ∃ (CP_m CP_a CP_o : ℝ), 
    (0.8 * CP_m = 5) ∧ 
    (1.1 * CP_a = 7) ∧ 
    (1.15 * CP_o = 4) ∧ 
    (1.05 * CP_m = 6.5625) ∧ 
    (1.05 * CP_a = 6.6818) ∧ 
    (1.05 * CP_o = 3.6522) := 
by {
  use [6.25, 6.3636, 3.4783],
  split; norm_num, -- for 0.8 * 6.25 = 5
  split; norm_num, -- for 1.1 * 6.3636 = 7
  split; norm_num, -- for 1.15 * 3.4783 = 4
  split; norm_num, -- for 1.05 * 6.25 = 6.5625
  split; norm_num, -- for 1.05 * 6.3636 = 6.6818
  norm_num  -- for 1.05 * 3.4783 = 3.6522
}

end fruit_seller_prices_l791_791203


namespace max_pencils_l791_791738

theorem max_pencils 
  (p : ℕ → ℝ)
  (h_price1 : ∀ n : ℕ, n ≤ 10 → p n = 0.75 * n)
  (h_price2 : ∀ n : ℕ, n > 10 → p n = 0.75 * 10 + 0.65 * (n - 10))
  (budget : ℝ) (h_budget : budget = 10) :
  ∃ n : ℕ, p n ≤ budget ∧ (∀ m : ℕ, p m ≤ budget → m ≤ 13) :=
by {
  sorry
}

end max_pencils_l791_791738


namespace two_colorable_map_l791_791412

theorem two_colorable_map (n : ℕ) (regions : ℕ) (lines_circles : set ℕ) :
  ∀ (n : ℕ), (∃ regions (r : set regions) (l : set lines_circles), colorable_with_two_colors r l) := 
sorry

-- Definitions referred:
def colorable_with_two_colors (regions : set ℕ) (boundaries : set ℕ) : Prop :=
∀ (r1 r2 : regions), (r1 ≠ r2) → (adjacent r1 r2 boundaries) → (different_color r1 r2)

def adjacent (r1 r2 : ℕ) (boundaries : set ℕ) : Prop := 
-- Definition of adjacency based on shared boundary segment or arc
sorry

def different_color (r1 r2 : ℕ) : Prop := 
-- Definition for ensuring two regions are of different colors
sorry

end two_colorable_map_l791_791412


namespace find_x_l791_791271

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l791_791271


namespace area_of_quadrilateral_BEIH_correct_l791_791170

open Int

noncomputable def area_of_quadrilateral_BEIH : ℚ :=
  let B := (0 : ℚ, 0 : ℚ)
  let A := (0 : ℚ, 3 : ℚ)
  let C := (4 : ℚ, 0 : ℚ)
  let D := (4 : ℚ, 3 : ℚ)
  let E := (0 : ℚ, 1.5 : ℚ)
  let F := (2 : ℚ, 0 : ℚ)
  let I := (8 / 9 : ℚ, 20 / 9 : ℚ)
  let H := (6 / 5 : ℚ, 6 / 5 : ℚ)

  let area := (1 / 2) * abs (
    B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2
    - (E.2 * B.1 + I.2 * E.1 + H.2 * I.1 + B.2 * H.1)
  )

  area

theorem area_of_quadrilateral_BEIH_correct : area_of_quadrilateral_BEIH = 184 / 405 := by
  sorry

end area_of_quadrilateral_BEIH_correct_l791_791170


namespace sum_of_possible_values_l791_791401

theorem sum_of_possible_values (x y : ℝ) 
  (h : x * y - 2 * x / y ^ 3 - 2 * y / x ^ 3 = 4) : 
  (x - 2) * (y - 2) = 1 := 
sorry

end sum_of_possible_values_l791_791401


namespace inclination_angle_of_focal_chord_l791_791665

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end inclination_angle_of_focal_chord_l791_791665


namespace rigid_body_one_fixed_point_l791_791735

theorem rigid_body_one_fixed_point :
  ∃ (P : point) (motion : rigid_body_motion), ( ∀ Q ≠ P, Q ∈ rigid_body -> Q ≠ fixed_point) :=
sorry

end rigid_body_one_fixed_point_l791_791735


namespace interval_proof_l791_791262

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l791_791262


namespace distance_bob_walked_when_met_l791_791086

theorem distance_bob_walked_when_met (distance_XY walk_rate_Yolanda walk_rate_Bob : ℕ)
  (start_time_Yolanda start_time_Bob : ℕ) (y_distance b_distance : ℕ) (t : ℕ)
  (h1 : distance_XY = 65)
  (h2 : walk_rate_Yolanda = 5)
  (h3 : walk_rate_Bob = 7)
  (h4 : start_time_Yolanda = 0)
  (h5 : start_time_Bob = 1)
  (h6 : y_distance = walk_rate_Yolanda * (t + start_time_Bob))
  (h7 : b_distance = walk_rate_Bob * t)
  (h8 : y_distance + b_distance = distance_XY) : 
  b_distance = 35 := 
sorry

end distance_bob_walked_when_met_l791_791086


namespace omega_range_l791_791698

theorem omega_range 
  (ω : ℝ) 
  (hω : ω > 0) 
  (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = sin (ω * x + (π / 3))) 
  (h_max : ∀ x ∈ set.Icc 0 (2 * π), f x ≤ 1)
  (h_min : ∀ x ∈ set.Icc 0 (2 * π), f x ≥ -1)
  (h_unique_max : ∃! x ∈ set.Icc 0 (2 * π), f x = 1)
  (h_unique_min : ∃! x ∈ set.Icc 0 (2 * π), f x = -1) :
  (7 / 12 : ℝ) ≤ ω ∧ ω < (13 / 12 : ℝ) := 
by 
  sorry

end omega_range_l791_791698


namespace probability_white_given_popped_l791_791196

theorem probability_white_given_popped :
  let P_white := 3 / 5
  let P_yellow := 2 / 5
  let P_popped_given_white := 2 / 5
  let P_popped_given_yellow := 4 / 5
  let P_white_and_popped := P_white * P_popped_given_white
  let P_yellow_and_popped := P_yellow * P_popped_given_yellow
  let P_popped := P_white_and_popped + P_yellow_and_popped
  let P_white_given_popped := P_white_and_popped / P_popped
  P_white_given_popped = 3 / 7 :=
by sorry

end probability_white_given_popped_l791_791196


namespace hockey_match_intermediate_goal_l791_791123

theorem hockey_match_intermediate_goal (score_dynamo final_score_spartak : ℕ) (final_score_dynamo : score_dynamo ≥ 8) (final_score_spartak_eq : final_score_spartak = 5) :
  ∃ t, (t : ℝ) ∈ Icc (0 : ℝ) 60 ∧
       ((score_dynamo - t * (8 - 0) / 60) = (t * (5 - 0) / 60)) := 
sorry

end hockey_match_intermediate_goal_l791_791123


namespace thirteenth_more_likely_friday_l791_791774

theorem thirteenth_more_likely_friday :
  (∃ freq : ℕ → ℕ, 
    freq (13, 5) > freq (13, 1) ∧
    freq (13, 5) > freq (13, 2) ∧
    freq (13, 5) > freq (13, 3) ∧
    freq (13, 5) > freq (13, 4) ∧
    freq (13, 5) > freq (13, 6) ∧
    freq (13, 5) > freq (13, 0)) :=
by
  -- Assume Gregorian calendar with 400-year cycle, leap years, and week pattern.
  admit

end thirteenth_more_likely_friday_l791_791774


namespace set_diff_cardinality_l791_791696

open Set

theorem set_diff_cardinality :
  let A := {1/2, 1/3, 1/5} : Set ℚ
  let B := {2/10, 4/10, 5/10, 7/10} : Set ℚ
  finset.card (A \ B : Finset ℚ) = 1 := by
  let A := {1/2, 1/3, 1/5} : Set ℚ
  let B := {2/10, 4/10, 5/10, 7/10} : Set ℚ
  sorry

end set_diff_cardinality_l791_791696


namespace hyperbola_properties_l791_791547

theorem hyperbola_properties (foci_on_x_axis : Prop) 
  (angle_between_asymptotes_eq_pi_over_3 : Prop) 
  (focal_distance_eq_12 : Prop) : 
  (∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (((foci_on_x_axis → (angle_between_asymptotes_eq_pi_over_3 → 
      focal_distance_eq_12 → (a^2 + b^2 = 36) ∧ (2 * a * (36 - a^2).sqrt = sqrt 3 * (2 * a^2 - 36)))) ) → 
      ((∃ (ecc : ℝ), ecc = (6 / sqrt a^2) ∧ 
      ((a^2 = 27 ∧ b^2 = 9 ∧ (∃ ecc2, ecc2 = (6 / sqrt 27) ∧ (ecc2 = 2 * (sqrt 3 / 3)))) ∨ 
      (a^2 = 9 ∧ b^2 = 27 ∧ (∃ ecc3, ecc3 = (6 / sqrt 9) ∧ (ecc3 = 2)))))))
   )) := sorry

end hyperbola_properties_l791_791547


namespace area_of_306090_triangle_l791_791577

-- Conditions
def is_306090_triangle (a b c : ℝ) : Prop :=
  a / b = 1 / Real.sqrt 3 ∧ a / c = 1 / 2

-- Given values
def hypotenuse : ℝ := 6

-- To prove
theorem area_of_306090_triangle :
  ∃ (a b c : ℝ), is_306090_triangle a b c ∧ c = hypotenuse ∧ (1 / 2) * a * b = (9 * Real.sqrt 3) / 2 :=
by
  sorry

end area_of_306090_triangle_l791_791577


namespace contrapositive_statement_l791_791118

theorem contrapositive_statement :
  (∀ n : ℕ, (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0) →
  (∀ n : ℕ, n % 10 ≠ 0 → ¬(n % 2 = 0 ∧ n % 5 = 0)) :=
by
  sorry

end contrapositive_statement_l791_791118


namespace last_student_calls_out_l791_791534

-- Define the transformation rules as a function
def next_student (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

-- Define the sequence generation function
noncomputable def student_number : ℕ → ℕ
| 0       => 1  -- the 1st student starts with number 1
| (n + 1) => next_student (student_number n)

-- The main theorem to prove
theorem last_student_calls_out (n : ℕ) : student_number 2013 = 12 :=
sorry

end last_student_calls_out_l791_791534


namespace vector_zero_or_scalar_zero_l791_791620

variables {Vector : Type} [AddCommGroup Vector] [Module ℝ Vector]

theorem vector_zero_or_scalar_zero (a : Vector) (λ : ℝ) : λ • a = 0 ↔ λ = 0 ∨ a = 0 := by
  sorry

end vector_zero_or_scalar_zero_l791_791620


namespace julia_played_tag_l791_791049

theorem julia_played_tag (monday_kids tuesday_kids : ℕ) (hm : monday_kids = 7) (ht : tuesday_kids = 13):
  monday_kids + tuesday_kids = 20 :=
by
  rw [hm, ht]
  exact Nat.add_comm 7 13

end julia_played_tag_l791_791049


namespace count_valid_pairs_l791_791593

/-- Define the predicate for the valid pairs -/
def valid_pair (m n : ℤ) : Prop :=
  1 ≤ m ∧ m ≤ 3013 ∧ 3^n < 2^m ∧ 2^m < 2^(m + 3) ∧ 2^(m + 3) < 3^(n + 1)

/-- Count the number of pairs (m, n) that satisfy the given conditions -/
theorem count_valid_pairs (n_pairs : ℤ) :
  n_pairs = 2797 ↔
  (n_pairs = (Finset.card (Finset.image₂ (uncurry valid_pair) (Finset.Icc 1 3013) (Finset.Icc 1 1000)))) :=
sorry

end count_valid_pairs_l791_791593


namespace find_f2023_pi_over_6_l791_791455

def f : ℕ → ℝ → ℝ
| 0, x := 3 * Real.sin x
| (n+1), x := 9 / (3 - f n x)

theorem find_f2023_pi_over_6 : f 2023 (Real.pi / 6) = 6 := by
  sorry

end find_f2023_pi_over_6_l791_791455


namespace perimeters_equal_l791_791907

noncomputable def side_length_square := 15 -- cm
noncomputable def length_rectangle := 18 -- cm
noncomputable def area_rectangle := 216 -- cm²

theorem perimeters_equal :
  let perimeter_square := 4 * side_length_square
  let width_rectangle := area_rectangle / length_rectangle
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  perimeter_square = perimeter_rectangle :=
by
  sorry

end perimeters_equal_l791_791907


namespace rectangle_area_l791_791802

-- Definitions of conditions
def width : ℝ := 5
def length : ℝ := 2 * width

-- The goal is to prove the area is 50 square inches given the length and width
theorem rectangle_area : length * width = 50 := by
  have h_length : length = 2 * width := by rfl
  have h_width : width = 5 := by rfl
  sorry

end rectangle_area_l791_791802


namespace streetlights_count_l791_791106

theorem streetlights_count
  (num_squares : ℕ)
  (streetlights_per_square : ℕ)
  (unused_streetlights : ℕ)
  (h_squares : num_squares = 15)
  (h_per_square : streetlights_per_square = 12)
  (h_unused : unused_streetlights = 20)
  : (num_squares * streetlights_per_square + unused_streetlights = 200) :=
by
  simp [h_squares, h_per_square, h_unused]
  sorry

end streetlights_count_l791_791106


namespace Tony_age_at_end_of_period_l791_791151

-- Definitions based on the conditions in a):
def hours_per_day := 2
def days_worked := 60
def total_earnings := 1140
def earnings_per_hour (age : ℕ) := age

-- The main property we need to prove: Tony's age at the end of the period is 12 years old
theorem Tony_age_at_end_of_period : ∃ age : ℕ, (2 * age * days_worked = total_earnings) ∧ age = 12 :=
by
  sorry

end Tony_age_at_end_of_period_l791_791151


namespace tony_rope_length_l791_791838

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l791_791838


namespace f_is_even_l791_791040

def f (x : ℝ) : ℝ := 2^(x^2 - 4) - x^2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by 
  sorry

end f_is_even_l791_791040


namespace number_of_natural_solutions_for_x_y_z_number_of_non_neg_solutions_for_x1_to_x10_statement_l791_791188

-- Problem 1: Number of natural number solutions for x + y + z = 15
theorem number_of_natural_solutions_for_x_y_z (x y z : ℕ) (h : x + y + z = 15) : 
  (card {t : ℕ × ℕ × ℕ | t.1 + t.2.1 + t.2.2 = 15}) = 91 :=
sorry

-- Problem 2: Number of non-negative integer solutions for 2x1 + x2 + ... + x10 = 3
theorem number_of_non_neg_solutions_for_x1_to_x10_statement (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℕ) 
  (h : 2 * x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 = 3) : 
  (card {t : ℕ^10 | 2 * t.1 + t.2 + t.3 + t.4 + t.5 + t.6 + t.7 + t.8 + t.9 + t.10 = 3}) = 174 :=
sorry

end number_of_natural_solutions_for_x_y_z_number_of_non_neg_solutions_for_x1_to_x10_statement_l791_791188


namespace remainder_of_division_l791_791161

def polynomial : ℚ[X] := 6 * X^8 - 2 * X^7 - 10 * X^6 + 3 * X^4 + 5 * X^3 - 15
def divisor : ℚ[X] := 3 * X - 6

theorem remainder_of_division :
  polynomial % divisor = 713 := 
  sorry

end remainder_of_division_l791_791161


namespace edge_length_of_small_cube_l791_791636

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end edge_length_of_small_cube_l791_791636


namespace ellipse_equation_tangent_line_properties_and_area_l791_791028

-- Definitions of Ellipse and relevant properties
def ellipse_center_origin (E : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), ((x, y) ∈ E) → ∃ a b : ℝ, a > b ∧ 
    ((x^2 / a^2) + (y^2 / b^2) = 1)

def passes_through_point (E : Set (ℝ × ℝ)) (A : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, ((x, y) ∈ E) → x = 0 ∧ y = 1

def orthogonal_vector (P Q R : ℝ × ℝ) : Prop :=
  let u := (fst Q - fst P, snd Q - snd P)
  let v := (fst R - fst P, snd R - snd P)
  u.1 * v.1 + u.2 * v.2 = 0

-- Main theorems
theorem ellipse_equation : 
  ∃ E : Set (ℝ × ℝ), ellipse_center_origin E ∧ passes_through_point E (0, 1) ∧ orthogonal_vector (0, 1) F₁ F₂ ∧ 
  ∀ x y : ℝ, (x, y) ∈ E ↔ (x^2/2 + y^2 = 1) := 
sorry

-- Definitions of line and circle, critical points and area
def line_passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  l (-real.sqrt 3) = 0

def line_tangent_to_circle (l : ℝ → ℝ) (r : ℝ) (Q : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 = r^2) ∧ l x = y 

def area_of_triangle (P Q R : (ℝ × ℝ)) : ℝ :=
  abs((fst P * (snd Q - snd R) + fst Q * (snd R - snd P) + fst R * (snd P - snd Q)) / 2)

-- Theorem stating final results
theorem tangent_line_properties_and_area :
  ∃ (l : ℝ → ℝ) (r : ℝ) (P Q : ℝ × ℝ), line_passes_through l (-real.sqrt 3, 0) ∧ l = (λ x, x + real.sqrt 3) ∧ 
  (line_tangent_to_circle l r Q) ∧ 
  r = real.sqrt (3/2) ∧
  area_of_triangle (0, 0) P Q = 1/4 :=
sorry

end ellipse_equation_tangent_line_properties_and_area_l791_791028


namespace solve_for_x_l791_791011

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end solve_for_x_l791_791011


namespace allocate_plots_l791_791508

theorem allocate_plots (x y : ℕ) (h : x > y) : 
  ∃ u v : ℕ, (u^2 + v^2 = 2 * (x^2 + y^2)) :=
by
  sorry

end allocate_plots_l791_791508


namespace find_f_2_l791_791122

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2 (h1 : ∀ x1 x2 : ℝ, f (x1 * x2) = f x1 + f x2) (h2 : f 8 = 3) : f 2 = 1 :=
by
  sorry

end find_f_2_l791_791122


namespace bowling_tournament_orders_l791_791023

theorem bowling_tournament_orders :
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  total_orders = 32 :=
by
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  show total_orders = 32
  sorry

end bowling_tournament_orders_l791_791023


namespace largest_five_digit_sum_twenty_l791_791852

theorem largest_five_digit_sum_twenty : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 20) ∧ (∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m.digits.sum = 20 → m ≤ n)) ∧ n = 99200 :=
sorry

end largest_five_digit_sum_twenty_l791_791852


namespace James_trout_pounds_l791_791044

def pounds_trout (T : ℝ) : Prop :=
  let salmon := 1.5 * T
  let tuna := 2 * T
  T + salmon + tuna = 1100

theorem James_trout_pounds :
  ∃ T : ℝ, pounds_trout T ∧ T = 244 :=
sorry

end James_trout_pounds_l791_791044


namespace solve_system_of_equations_l791_791134

theorem solve_system_of_equations :
  {p : ℝ × ℝ | 
    (p.1^2 + p.2 + 1) * (p.2^2 + p.1 + 1) = 4 ∧
    (p.1^2 + p.2)^2 + (p.2^2 + p.1)^2 = 2} =
  {(0, 1), (1, 0), 
   ( (-1 + Real.sqrt 5) / 2, (-1 + Real.sqrt 5) / 2),
   ( (-1 - Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2) } :=
by
  sorry

end solve_system_of_equations_l791_791134


namespace problem_statement_l791_791298

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h_diff : differentiable ℝ f)
  (h_f0 : f 0 = 1)
  (h_deriv_cond : ∀ x, x ≠ -1 → (x + 1) * (fderiv ℝ f x - f x) > 0) :
  (∀ x, x > -1 → (f x / exp x > f (x + 1) / exp (x + 1))) ∧
  (exp 2 * f e > exp (exp 1) * f 2) :=
by
  sorry

end problem_statement_l791_791298


namespace sum_solution_l791_791575

theorem sum_solution : 
  (∑ n in Finset.range 1979, n * (1980 - n)) = 1979 * 990 * 661 := 
by
  sorry

end sum_solution_l791_791575


namespace gcf_60_90_l791_791489

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l791_791489


namespace error_percent_area_measurement_l791_791720

theorem error_percent_area_measurement :
  ∀ (L W : ℝ), 
  let L' := 1.20 * L in
  let W' := 0.90 * W in
  let A := L * W in
  let A' := L' * W' in
  (A' - A) / A * 100 = 8 := 
by
  intros L W
  let L' := 1.20 * L
  let W' := 0.90 * W
  let A := L * W
  let A' := L' * W'
  sorry

end error_percent_area_measurement_l791_791720


namespace man_speed_theorem_l791_791570

noncomputable def train_length : ℝ := 800
noncomputable def train_speed_kmh : ℝ := 63
noncomputable def crossing_time : ℝ := 47.99616030717543

noncomputable def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (1000 / 3600)
noncomputable def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh
noncomputable def relative_speed (length : ℝ) (time : ℝ) : ℝ := length / time
noncomputable def man_speed_ms (train_speed : ℝ) (relative_speed : ℝ) : ℝ := train_speed - relative_speed
noncomputable def ms_to_kmh (ms : ℝ) : ℝ := ms * (3600 / 1000)
noncomputable def man_speed_kmh : ℝ := ms_to_kmh (man_speed_ms train_speed_ms (relative_speed train_length crossing_time))

theorem man_speed_theorem : man_speed_kmh ≈ 2.9916 := by
  sorry

end man_speed_theorem_l791_791570


namespace pasture_rent_share_l791_791869

theorem pasture_rent_share (x : ℕ) (H1 : (45 / (10 * x + 60 + 45)) * 245 = 63) : 
  x = 7 :=
by {
  sorry
}

end pasture_rent_share_l791_791869


namespace jackie_work_hours_l791_791377

theorem jackie_work_hours :
  ∀ (T E S F : ℕ), T = 24 ∧ E = 3 ∧ S = 8 ∧ F = 5 → T - (E + S + F) = 8 := by
  intros T E S F h
  rcases h with ⟨hT, hE, hS, hF⟩
  calc
    T - (E + S + F) = 24 - (3 + 8 + 5) : by { rw [hT, hE, hS, hF] }
    ...              = 24 - 16          : by rfl
    ...              = 8               : by rfl

end jackie_work_hours_l791_791377


namespace intersection_of_A_and_B_l791_791679

noncomputable def setA : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def setB : Set ℝ := {y | 1 < y}

theorem intersection_of_A_and_B : setA ∩ {x | ∃ y, y = 2^x ∧ x > 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l791_791679


namespace composite_positive_integer_property_l791_791957

open Nat

theorem composite_positive_integer_property :
  ∀ n : ℕ, (n > 1 ∧ ¬Prime n) →
  (∃ (d : ℕ → ℕ) (k : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ k → d i ∣ n ∧ 1 = d 1 ∧ d k = n ∧ ∀ (i : ℕ), 2 ≤ i ∧ i ≤ k → d i = (d (i - 1) + 1)) →
    (n = 4)) := 
by
  sorry

end composite_positive_integer_property_l791_791957


namespace sum_terms_zero_l791_791646

variable {α : Type _} [linear_ordered_field α]
variables (a : ℕ → α) (S : ℕ → α)
variables A B C : EuclideanSpace ℝ (Fin 2)

-- Given data
variable (a1009 : α := a 1009)
variable (a1 : α := a 1)
variable (a2017 : α := a 2017)
variable (Sn : ℕ → α := S)

-- Coordinates of points
variable (A_coord : EuclideanSpace ℝ (Fin 2) := ![a1009, 1])
variable (B_coord : EuclideanSpace ℝ (Fin 2) := ![2, -1])
variable (C_coord : EuclideanSpace ℝ (Fin 2) := ![2, 2])


theorem sum_terms_zero (a_seq : ∀ n, a n = a1 + (n - 1) * d) 
  (sum_seq : ∀ n, S n = n * (a 1 + a n)) 
  (same_proj : ⟪A_coord, C_coord⟫ = ⟪B_coord, C_coord⟫) 
  (a1009_zero : a 1009 = 0) 
  (proj_eq : 2 * a1009 + 2 = 2 * 2 - 1 * 2) :
  S 2017 = 0 :=
by
  -- calling the hypotheses to satisfy the conditions
  sorry

end sum_terms_zero_l791_791646


namespace red_ball_probability_l791_791715

theorem red_ball_probability :
  let red_balls := 2
      black_balls := 3
      total_balls := red_balls + black_balls
  in (red_balls / total_balls) = 2 / 5 :=
by
  sorry

end red_ball_probability_l791_791715


namespace total_students_l791_791359

def class_sizes (start : ℕ) : List ℕ :=
  List.init 5 (λ i => start - 2 * i)

theorem total_students (h : class_sizes 27 = [27, 25, 23, 21, 19]) : 
  List.sum (class_sizes 27) = 115 :=
by
  sorry

end total_students_l791_791359


namespace circle_non_intersecting_square_l791_791193

theorem circle_non_intersecting_square {rect_width rect_height : ℝ} (unit_squares : set (set (ℝ × ℝ))) :
  rect_width = 20 ∧ rect_height = 25 → 
  (∀ s, s ∈ unit_squares → (∃ a b : ℝ, s = { p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ a + 1 ∧ b ≤ p.2 ∧ p.2 ≤ b + 1 })) →
  (card unit_squares = 120) →
  (∃ (cx cy : ℝ), (0.5 ≤ cx ∧ cx ≤ rect_width - 0.5) ∧ (0.5 ≤ cy ∧ cy ≤ rect_height - 0.5) ∧
    ∀ s ∈ unit_squares, ∀ (p : ℝ × ℝ), p ∈ s → dist (cx, cy) p ≥ 0.5) :=
by
  sorry

end circle_non_intersecting_square_l791_791193


namespace find_c_value_l791_791286

theorem find_c_value 
  (a c : ℝ) 
  (h1 : deriv (λ x : ℝ, a * x ^ 3 + c) 1 = 6) 
  (h2 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (a * 2 ^ 3 + c) = 20) :
  c = 4 :=
sorry

end find_c_value_l791_791286


namespace finite_set_of_congruent_polygons_cover_square_l791_791946

theorem finite_set_of_congruent_polygons_cover_square (M K : convex_polygon) (H1 : ∀ side ∈ M.sides ∪ M.diagonals, is_integer side.length) (H2 : square K) :
  ∃ S : finset (translation_reflection M), (⋃ P ∈ S, P) = K ∧ ∀ p ∈ interior_points K, ∃ n ∈ ℕ, ∀ P ∈ S, p ∈ (P \ sides P) ↔ multiplicity p S = n :=
by sorry

end finite_set_of_congruent_polygons_cover_square_l791_791946


namespace sufficient_but_not_necessary_condition_l791_791182

def sufficient_condition (a : ℝ) : Prop := 
  (a > 1) → (1 / a < 1)

def necessary_condition (a : ℝ) : Prop := 
  (1 / a < 1) → (a > 1)

theorem sufficient_but_not_necessary_condition (a : ℝ) : sufficient_condition a ∧ ¬necessary_condition a := by
  sorry

end sufficient_but_not_necessary_condition_l791_791182


namespace max_theater_members_l791_791135

theorem max_theater_members (N : ℕ) :
  (∃ (k : ℕ), (N = k^2 + 3)) ∧ (∃ (n : ℕ), (N = n * (n + 9))) → N ≤ 360 :=
by
  sorry

end max_theater_members_l791_791135


namespace find_other_number_l791_791178

theorem find_other_number (A B : ℕ) (h1 : A = 24) (hHCF_reciprocal : (nat.gcd A B) = 17) (hLCM_reciprocal : (nat.lcm A B) = 312) : B = 221 :=
by {
  -- the statement of the theorem in Lean
  sorry
}

end find_other_number_l791_791178


namespace area_ratios_of_triangle_l791_791773

theorem area_ratios_of_triangle (ABC A B C A₁ B₁ C₁ : Point)
  (hA₁ : A₁ ∈ Segment(A B))
  (hB₁ : B₁ ∈ Segment(B C))
  (hC₁ : C₁ ∈ Segment(C A))
  (concurrent: Ceva(ABC, A₁, B₁, C₁)) :
  area(A₁ B₁ C₁) / area(ABC) ≤ 1 / 4 := 
  sorry

end area_ratios_of_triangle_l791_791773


namespace point_equal_distances_l791_791555

theorem point_equal_distances (x y : ℝ) (hx : y = x) (hxy : y - 4 = -x) (hline : x + y = 4) : x = 2 :=
by sorry

end point_equal_distances_l791_791555


namespace smaller_number_is_25_l791_791464

theorem smaller_number_is_25 (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end smaller_number_is_25_l791_791464


namespace largest_five_digit_sum_twenty_l791_791854

theorem largest_five_digit_sum_twenty : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 20) ∧ (∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m.digits.sum = 20 → m ≤ n)) ∧ n = 99200 :=
sorry

end largest_five_digit_sum_twenty_l791_791854


namespace problem_l791_791694

theorem problem (a b : ℚ) (h : a / b = 6 / 5) : (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := 
by 
  sorry

end problem_l791_791694


namespace tangent_line_at_1_interval_of_increase_minimum_value_on_interval_l791_791998

open Real

noncomputable def f (a : ℝ) (x : ℝ) := a * x - log x

theorem tangent_line_at_1 (a : ℝ) (h_a : a = 2) :
  let y := f 2 1
  ∈d (f 2)' 1 = 1 :=
  sorry

theorem interval_of_increase (a : ℝ) (h_extremum : (f a)' 1 = 0) :
  a = 1 → ∀ x, x ∈ Ioi 1 → (f 1)' x > 0 :=
  sorry

theorem minimum_value_on_interval (a : ℝ) (h_min : ∃! x ∈ Ioc 0 e, f a x = 3) :
  a = exp 2 :=
  sorry

end tangent_line_at_1_interval_of_increase_minimum_value_on_interval_l791_791998


namespace all_fruits_fallen_by_twelfth_day_l791_791830

noncomputable def magical_tree_falling_day : Nat :=
  let total_fruits := 58
  let initial_day_falls := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].foldl (· + ·) 0
  let continuation_falls := [1, 2].foldl (· + ·) 0
  let total_days := initial_day_falls + continuation_falls
  12

theorem all_fruits_fallen_by_twelfth_day :
  magical_tree_falling_day = 12 :=
by
  sorry

end all_fruits_fallen_by_twelfth_day_l791_791830


namespace plates_per_meal_l791_791482

theorem plates_per_meal 
  (people : ℕ) (meals_per_day : ℕ) (total_days : ℕ) (total_plates : ℕ) 
  (h_people : people = 6) 
  (h_meals : meals_per_day = 3) 
  (h_days : total_days = 4) 
  (h_plates : total_plates = 144) 
  : (total_plates / (people * meals_per_day * total_days)) = 2 := 
  sorry

end plates_per_meal_l791_791482


namespace calculation_result_l791_791230

theorem calculation_result :
  5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 :=
by sorry

end calculation_result_l791_791230


namespace exists_pos_integer_m_divisible_by_1988_l791_791280

noncomputable def f : ℤ → ℤ := λ x, 3 * x + 2

theorem exists_pos_integer_m_divisible_by_1988 :
  ∃ m : ℕ, f^[100] m % 1988 = 0 := 
sorry

end exists_pos_integer_m_divisible_by_1988_l791_791280


namespace woman_works_finish_days_l791_791883

theorem woman_works_finish_days (M W : ℝ) 
  (hm_work : ∀ n : ℝ, n * M = 1 / 100)
  (hw_work : ∀ men women : ℝ, (10 * M + 15 * women) * 6 = 1) :
  W = 1 / 225 :=
by
  have man_work := hm_work 1
  have woman_work := hw_work 10 W
  sorry

end woman_works_finish_days_l791_791883


namespace no_such_function_exists_l791_791958

namespace ProofProblem

open Nat

-- Declaration of the proposed function
def f : ℕ+ → ℕ+ := sorry

-- Statement to be proved
theorem no_such_function_exists : 
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f^[n] n = n + 1 :=
by
  sorry

end ProofProblem

end no_such_function_exists_l791_791958


namespace find_sequence_expression_l791_791372

def sequence (a : ℕ → ℚ) : Prop :=
  a 2 = 3 / 2 ∧ a 3 = 7 / 3 ∧ ∃ r : ℚ, ∀ n : ℕ, n > 0 → (n * a n + 1) = r^(n-1)

theorem find_sequence_expression (a : ℕ → ℚ) (h : sequence a) : ∀ n, a n = (2^n - 1) / n :=
by
  sorry

end find_sequence_expression_l791_791372


namespace count_triangles_l791_791951

-- Define the conditions for the problem
def P (x1 x2 : ℕ) : Prop := 37 * x1 ≤ 2022 ∧ 37 * x2 ≤ 2022

def valid_points (x y : ℕ) : Prop := 37 * x + y = 2022

def area_multiple_of_3 (x1 x2 : ℕ): Prop :=
  (∃ k : ℤ, 3 * k = x1 - x2) ∧ x1 ≠ x2 ∧ P x1 x2

-- The final theorem to prove the number of such distinct triangles
theorem count_triangles : 
  (∃ (n : ℕ), n = 459 ∧ 
    ∃ x1 x2 : ℕ, area_multiple_of_3 x1 x2 ∧ x1 ≠ x2) :=
by
  sorry

end count_triangles_l791_791951


namespace MarcoScoresAreCorrect_l791_791816

noncomputable def MarcoTestScores : List ℕ := [94, 82, 76, 75, 64]

theorem MarcoScoresAreCorrect : 
  ∀ (scores : List ℕ),
    scores = [82, 76, 75] ∧ 
    (∃ t4 t5, t4 < 95 ∧ t5 < 95 ∧ 82 ≠ t4 ∧ 82 ≠ t5 ∧ 76 ≠ t4 ∧ 76 ≠ t5 ∧ 75 ≠ t4 ∧ 75 ≠ t5 ∧ 
       t4 ≠ t5 ∧
       (82 + 76 + 75 + t4 + t5 = 5 * 85) ∧ 
       (82 + 76 = t4 + t5)) → 
    (scores = [94, 82, 76, 75, 64]) := 
by 
  sorry

end MarcoScoresAreCorrect_l791_791816


namespace gcd_60_90_l791_791491

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end gcd_60_90_l791_791491


namespace geometric_sequence_S6_l791_791299

-- We first need to ensure our definitions match the given conditions.
noncomputable def a1 : ℝ := 1 -- root of x^2 - 5x + 4 = 0
noncomputable def a3 : ℝ := 4 -- root of x^2 - 5x + 4 = 0

-- Definition of the geometric sequence
noncomputable def q : ℝ := 2 -- common ratio derived from geometric sequence where a3 = a1 * q^2

-- Definition of the n-th term of the geometric sequence
noncomputable def a (n : ℕ) : ℝ := a1 * q^((n : ℝ) - 1)

-- Definition of the sum of the first n terms of the geometric sequence
noncomputable def S (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)

-- The theorem we want to prove
theorem geometric_sequence_S6 : S 6 = 63 :=
  by sorry

end geometric_sequence_S6_l791_791299


namespace min_value_fraction_l791_791752

theorem min_value_fraction (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_sum : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_fraction_l791_791752


namespace chinese_remainder_theorem_count_l791_791103

theorem chinese_remainder_theorem_count :
  (card {x ∈ Finset.range 2024 // x % 3 = 1 ∧ x % 5 = 1}) = 135 :=
sorry

end chinese_remainder_theorem_count_l791_791103


namespace part1_k_value_part2_ratio_l791_791183

-- Definition according to the condition in Part 1
def inverse_proportion (k x : ℝ) : ℝ := k / x

-- Definition according to the condition in Part 2
def given_ratio_condition (a b : ℝ) : Prop := (a - b) / b = 2 / 3

-- Define the theorem for part 1
theorem part1_k_value (k x y : ℝ) (hx : x = -4) (hy : y = 3/4) (h : inverse_proportion k x = y) : k = -3 := by
  sorry

-- Define the theorem for part 2
theorem part2_ratio (a b : ℝ) (h : given_ratio_condition a b) : a / b = 5 / 3 := by
  sorry

end part1_k_value_part2_ratio_l791_791183


namespace solution_set_of_inequality_l791_791333

def f (x : ℝ) : ℝ := log x + 2^x + sqrt x - 1

theorem solution_set_of_inequality :
  {x : ℝ | f x > f (2 * x - 4)} = {x : ℝ | 2 < x ∧ x < 4 ∧ 0 < x} := 
by 
  sorry

end solution_set_of_inequality_l791_791333


namespace mean_of_remaining_four_numbers_l791_791110

theorem mean_of_remaining_four_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 → (a + b + c + d) / 4 = 88.75 :=
by
  intro h
  sorry

end mean_of_remaining_four_numbers_l791_791110


namespace non_intersecting_intervals_finite_or_countable_l791_791090

-- Statement of the problem
theorem non_intersecting_intervals_finite_or_countable (I : set (set ℝ)) 
  (h_pairwise : pairwise (disjoint on I))
  (h_rational_point : ∀ i ∈ I, ∃ q ∈ i, q ∈ ℚ) : 
  I.finite ∨ I.countable :=
sorry

end non_intersecting_intervals_finite_or_countable_l791_791090


namespace composite_shape_area_l791_791966

theorem composite_shape_area : 
  ∀ (ABC : Triangle) (ABDE BCHI CAFG : Square) (octagon : RegularOctagon),
  ABC.equilateral 2 →
  Square.side_length ABDE 2 ∧ Square.side_length BCHI 2 ∧ Square.side_length CAFG 2 →
  Square.attached ABDE ABC AB ∧ Square.attached BCHI ABC BC ∧ Square.attached CAFG ABC CA →
  RegularOctagon.attached octagon CAFG FG →
  CompositeShape.area (CompositeShape.mk hexagon octagon) = 4*√3 + 8*√2 + 2 := 
sorry

end composite_shape_area_l791_791966


namespace ratio_of_segments_l791_791147

variable {α : Type} [LinearOrderedField α] 

-- Define geometric points and lines as needed
variables (A B C A1 B1 M : α × α)

-- Define conditions for the isosceles triangle and intersecting line
def isosceles_triangle (A B C : α × α) : Prop :=
(AB_dist : dist A B = dist A C)

def intersects_sides (M A1 B1 A B C : α × α) : Prop :=
(on_base_M : ∃ ratio : α, M = (1 - ratio) • A + ratio • B) ∧
(on_side_CA : ∃ t : α, A1 = (1 - t) • C + t • A) ∧
(on_side_CB : ∃ s : α, B1 = (1 - s) • C + s • B)

-- Define the theorem that needs to be proved
theorem ratio_of_segments (A B C A1 B1 M : α × α)
  (h_iso : isosceles_triangle A B C)
  (h_intersects : intersects_sides M A1 B1 A B C) :
  dist A1 A / dist A1 M = dist B1 B / dist B1 M :=
sorry

end ratio_of_segments_l791_791147


namespace point_C_coordinates_l791_791326

-- Given vectors and points
def vectorAB := (1, 3)
def vectorBC := (2, 1)
def pointA := (-1, 2)

-- Definition of vector addition
def vector_add (v1 v2 : ℕ × ℕ) : ℕ × ℕ := (v1.1 + v2.1, v1.2 + v2.2)

-- Definition of point translation
def point_translate (p : ℕ × ℕ) (v : ℕ × ℕ) : ℕ × ℕ := (p.1 + v.1, p.2 + v.2)

theorem point_C_coordinates :
  let vectorAC := vector_add vectorAB vectorBC in
  point_translate pointA vectorAC = (2, 6) :=
by
  -- proof would be here
  sorry

end point_C_coordinates_l791_791326


namespace total_oil_volume_l791_791214

theorem total_oil_volume (total_bottles : ℕ) (bottles_250ml : ℕ) (bottles_300ml : ℕ)
    (volume_250ml : ℕ) (volume_300ml : ℕ) (total_volume_ml : ℚ) 
    (total_volume_l : ℚ) (h1 : total_bottles = 35)
    (h2 : bottles_250ml = 17) (h3 : bottles_300ml = total_bottles - bottles_250ml)
    (h4 : volume_250ml = 250) (h5 : volume_300ml = 300) 
    (h6 : total_volume_ml = bottles_250ml * volume_250ml + bottles_300ml * volume_300ml)
    (h7 : total_volume_l = total_volume_ml / 1000) : 
    total_volume_l = 9.65 := 
by 
  sorry

end total_oil_volume_l791_791214


namespace range_of_inverse_l791_791690

noncomputable def f (x : ℝ) : ℝ := log10 (x + 1)

theorem range_of_inverse :
  ∀ y : ℝ, ∃ x : ℝ, f x = y → (x > -1) :=
by
  sorry

end range_of_inverse_l791_791690


namespace not_associative_l791_791389

variable {T : Type} [NonZeroReal T]

def op (a b : T) : T := 3 * a * b

theorem not_associative : ¬ ∀ a b c : T, op a (op b c) = op (op a b) c :=
by sorry

end not_associative_l791_791389


namespace gummies_remain_l791_791740

theorem gummies_remain
  (initial_candies : ℕ)
  (sibling_candies_per : ℕ)
  (num_siblings : ℕ)
  (best_friend_fraction : ℝ)
  (cousin_fraction : ℝ)
  (kept_candies : ℕ)
  (result : ℕ)
  (h_initial : initial_candies = 500)
  (h_sibling_candies_per : sibling_candies_per = 35)
  (h_num_siblings : num_siblings = 3)
  (h_best_friend_fraction : best_friend_fraction = 0.5)
  (h_cousin_fraction : cousin_fraction = 0.25)
  (h_kept_candies : kept_candies = 50)
  (h_result : result = 99) : 
  (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋ - 
  ⌊cousin_fraction * (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋)⌋ 
  - kept_candies) = result := 
by {
  sorry
}

end gummies_remain_l791_791740


namespace Janet_needs_to_save_more_l791_791046

theorem Janet_needs_to_save_more :
  let savings := 2225
  let rent_per_month := 1250
  let advance_rent_months := 2
  let deposit := 500
  let total_needed := (rent_per_month * advance_rent_months) + deposit
  let additional_needed := total_needed - savings
  in
    additional_needed = 775 := sorry

end Janet_needs_to_save_more_l791_791046


namespace domain_of_f_l791_791120

theorem domain_of_f (x : ℝ) :
  (tan x - 1 ≥ 0 ∧ 4 - x^2 ≥ 0) ↔ (x ∈ Set.Icc (-2 : ℝ) (-(π / 2)) ∪ Set.Ico (π / 4) (π / 2)) := by
    sorry

end domain_of_f_l791_791120


namespace find_x_l791_791244

noncomputable def isCorrectValue (x : ℝ) : Prop :=
  ⌊x⌋ + x = 13.4

theorem find_x (x : ℝ) (h : isCorrectValue x) : x = 6.4 :=
  sorry

end find_x_l791_791244


namespace greatest_perimeter_among_four_pieces_l791_791045

/--
Given an isosceles triangle with a base of 12 inches and a height of 15 inches,
the greatest perimeter among the four pieces of equal area obtained by cutting
the triangle into four smaller triangles is approximately 33.43 inches.
-/
theorem greatest_perimeter_among_four_pieces :
  let base : ℝ := 12
  let height : ℝ := 15
  ∃ (P : ℝ), P = (3 + Real.sqrt (225 + 4) + Real.sqrt (225 + 9)) ∧ abs (P - 33.43) < 0.01 := sorry

end greatest_perimeter_among_four_pieces_l791_791045


namespace triangles_sticks_not_proportional_l791_791808

theorem triangles_sticks_not_proportional :
  ∀ (n_triangles n_sticks : ℕ), 
  (∃ k : ℕ, n_triangles = k * n_sticks) 
  ∨ 
  (∃ k : ℕ, n_triangles * n_sticks = k) 
  → False :=
by
  sorry

end triangles_sticks_not_proportional_l791_791808


namespace ticket_cost_l791_791928

theorem ticket_cost (a : ℝ)
  (h1 : ∀ c : ℝ, c = a / 3)
  (h2 : 3 * a + 5 * (a / 3) = 27.75) :
  6 * a + 9 * (a / 3) = 53.52 := 
sorry

end ticket_cost_l791_791928


namespace max_rectangles_after_cuts_l791_791179

theorem max_rectangles_after_cuts :
  let total_cells := 8 * 8 in
  let cut_cells := 3 * (2 * 2) in
  let remaining_cells := total_cells - cut_cells in
  let max_rectangles := remaining_cells / 3 in
  ∃ n : ℕ, n = max_rectangles ∧ n = 17 :=
begin
  let total_cells := 8 * 8,
  let cut_cells := 3 * (2 * 2),
  let remaining_cells := total_cells - cut_cells,
  let max_rectangles := remaining_cells / 3,
  use max_rectangles,
  split,
  { refl },
  { rw max_rectangles,
    norm_num,
    exact 17 },
end

end max_rectangles_after_cuts_l791_791179


namespace pythagorean_theorem_from_general_theorem_l791_791936

theorem pythagorean_theorem_from_general_theorem (a b c : ℝ) (gamma : ℝ) :
  gamma = real.pi / 2 → c^2 = a^2 + b^2 - 2 * a * b * real.cos gamma → c^2 = a^2 + b^2 :=
by
  intro h_gamma h_general_theorem
  have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
  rw [h_gamma, h_cos_90] at h_general_theorem
  simp at h_general_theorem
  exact h_general_theorem

end pythagorean_theorem_from_general_theorem_l791_791936


namespace relatively_prime_in_set_l791_791012

theorem relatively_prime_in_set (s : Finset ℕ) (n : ℕ) (h_distinct: s.card = n+1) (h_range: ∀ x ∈ s, x ∈ Finset.range (2*n+1)) :
  ∃ a b ∈ s, a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end relatively_prime_in_set_l791_791012


namespace number_of_correct_statements_l791_791918

theorem number_of_correct_statements :
  let P1 := ∀ (l : Line) (P Q : Plane), (P ∥ l) → (Q ∥ l) → (P ∥ Q)
  let P2 := ∀ (P Q R : Plane), (P ∥ R) → (Q ∥ R) → (P ∥ Q)
  let P3 := ∀ (l m n : Line), (l ⊥ n) → (m ⊥ n) → (l ∥ m)
  let P4 := ∀ (l m : Line) (P : Plane), (l ⊥ P) → (m ⊥ P) → (l ∥ m)
  (¬ P1) ∧ P2 ∧ (¬ P3) ∧ P4 → count_if_true [P1, P2, P3, P4] = 2 :=
by {
  sorry
}

end number_of_correct_statements_l791_791918


namespace convex_polygon_sides_eq_longest_diag_at_most_two_l791_791330

theorem convex_polygon_sides_eq_longest_diag_at_most_two
  (n : ℕ)
  (polygon : list ℝ)
  (h_convex : convex polygon)
  (h_sides : ∀ (i j : ℕ), 1 ≤ i < j ≤ n → polygon[i] = max (polygon.diagonal_lengths))
  (h_trian_ineq : ∀ (a b c : ℝ), a + b ≥ c) :
  count (λ side, side = max (polygon.diagonal_lengths)) polygon ≤ 2 :=
sorry

end convex_polygon_sides_eq_longest_diag_at_most_two_l791_791330


namespace range_of_m_l791_791653

noncomputable def prop_p (m : ℝ) : Prop :=
0 < m ∧ m < 1 / 3

noncomputable def prop_q (m : ℝ) : Prop :=
0 < m ∧ m < 15

theorem range_of_m (m : ℝ) : (prop_p m ∧ ¬ prop_q m) ∨ (¬ prop_p m ∧ prop_q m) ↔ 1 / 3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l791_791653


namespace find_x_l791_791270

theorem find_x (x : ℝ) :
  18^3 = (16^2) / 4 * 3^(6 * x) → 
  x = Real.log 91.125 / (6 * Real.log 3) :=
by
  sorry

end find_x_l791_791270


namespace sequence_general_term_l791_791006

-- Define the sequence based on the conditions provided.
def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 2 then -1 / 3
  else if n = 3 then 1 / 4
  else if n = 4 then -1 / 5
  else (1:ℚ) / 1 -- Dummy value, since the first four terms are explicitly given.

-- Define the general term of the sequence as per the problem statement.
def general_term (n : ℕ) : ℚ :=
  (-1)^(n + 1) / (n + 1)

-- The proof statement that must be shown.
theorem sequence_general_term :
  ∀ n, (n < 5) → general_term n = sequence (n + 1) :=
by
  sorry

end sequence_general_term_l791_791006


namespace count_symbols_l791_791702

theorem count_symbols (n : ℕ) (hn : n = 15) : (n - 1) = 14 :=
by
  rw [hn]
  norm_num

end count_symbols_l791_791702


namespace correct_operation_l791_791510

theorem correct_operation (a b : ℝ) : 
  (2 * a^2 + a^2 = 3 * a^2) ∧ 
  (a^3 * a^3 ≠ 2 * a^3) ∧ 
  (a^9 / a^3 ≠ a^3) ∧ 
  (¬(7 * a * b - 5 * a = 2)) :=
by 
  sorry

end correct_operation_l791_791510


namespace interval_solution_l791_791258

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l791_791258


namespace plane_equation_correct_l791_791867

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def plane_eq (n : Point3D) (A : Point3D) : Point3D → ℝ :=
  fun P => n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

def is_perpendicular_plane (A B C : Point3D) (D : Point3D → ℝ) : Prop :=
  let BC := vector_sub C B
  D = plane_eq BC A

theorem plane_equation_correct :
  let A := { x := 7, y := -5, z := 1 }
  let B := { x := 5, y := -1, z := -3 }
  let C := { x := 3, y := 0, z := -4 }
  is_perpendicular_plane A B C (fun P => -2 * P.x + P.y - P.z + 20) :=
by
  sorry

end plane_equation_correct_l791_791867


namespace area_of_shaded_region_l791_791153

def radius_of_first_circle : ℝ := 4
def radius_of_second_circle : ℝ := 5
def radius_of_third_circle : ℝ := 2
def radius_of_fourth_circle : ℝ := 9

theorem area_of_shaded_region :
  π * (radius_of_fourth_circle ^ 2) - π * (radius_of_first_circle ^ 2) - π * (radius_of_second_circle ^ 2) - π * (radius_of_third_circle ^ 2) = 36 * π :=
by {
  sorry
}

end area_of_shaded_region_l791_791153


namespace triangle_inequality_proof_l791_791734

theorem triangle_inequality_proof (a b : ℕ) (ha : a > 1) (hb : b > 1) 
  (h1 : a = b) : b - a = 0 :=
by 
  rw h1
  exact Nat.sub_self a

end triangle_inequality_proof_l791_791734


namespace limit_derivative_transformation_l791_791292

variables {f : ℝ → ℝ}
noncomputable def d_delta_x := λ h : ℝ, h = 0

theorem limit_derivative_transformation 
  (H: ∀ y : ℝ, ∃ δ > 0, ∀ ε > 0, ∀ h : ℝ, 0 < abs h < δ → abs ( (f (1 + 3 * h) - f (1) ) / h - f'(1) ) < ε ) 
  (H_deriv : f'(1) = 1) :
  ∃ l : ℝ, tendsto (λ Δx : ℝ, (f(1 + 3 * Δx) - f(1)) / Δx) (nhds 0) (nhds l) ∧ l = 3 :=
by
  sorry

end limit_derivative_transformation_l791_791292


namespace determine_d_l791_791239

theorem determine_d :
  ∃ d : ℝ, (let a := 1; let b := -10; let c := d + 4 in
             let h := -b / (2 * a) in
             let k := c - b^2 / (4 * a) in
             k = 0) → d = 21 :=
sorry

end determine_d_l791_791239


namespace sqrt_inequality_l791_791556

theorem sqrt_inequality (x : ℝ) (h₀ : x > 0) : (sqrt x > 3 * x - 2) ↔ (4 / 9 < x ∧ x < 1) :=
sorry

end sqrt_inequality_l791_791556


namespace maximum_possible_topper_score_l791_791177

noncomputable def max_topper_score (scores : Fin 10 → ℕ) : ℕ :=
  let top5_scores := (Finset.univ.image scores).sort (≤).reverse.take 5
  top5_scores.head

theorem maximum_possible_topper_score
  (scores : Fin 10 → ℕ)
  (avg_score : (∑ i, scores i) / 10 = 60)
  (remaining_avg_score : (∑ i in Finset.filter (λ j, scores j ∈ (Finset.univ.image scores).erase (Finset.univ.image scores).sort(≤).reverse.take 5) Finset.univ, scores i) / 5 = 55)
  (all_passed : ∀ i, scores i ≥ 40)
  (distinct_top5 : Function.Injective (λ i, (Finset.univ.image scores).sort(≤).reverse.take 5))
  : max_topper_score scores = 100 := sorry

end maximum_possible_topper_score_l791_791177


namespace length_PQ_is_8_l791_791039

variable (P Q R X Y Z : Type) [Plane P Q R, Line Q := 90, Length QR := 8, PQ : ℝ]

theorem length_PQ_is_8 : (least_length XZ = 3.2) → (Length PQ = 8) := by
  sorry

end length_PQ_is_8_l791_791039


namespace necessary_but_not_sufficient_l791_791438

noncomputable def necessary_but_not_sufficient_condition := 
  ∀ x : ℝ, (x-1)*(x-2) = 0 → (x ≠ 1) → (x-1 ≠ 0)

lemma problem_equivalence (x : ℝ) : 
  (x-1)*(x-2) = 0 ↔ (x = 1 ∨ x = 2) := sorry

theorem necessary_but_not_sufficient : necessary_but_not_sufficient_condition :=
by {
  intros x h hx1,
  rw problem_equivalence at h,
  cases h,
  { contradiction },
  { intro hx1_,
    contradiction }
}

end necessary_but_not_sufficient_l791_791438


namespace empty_bags_even_condition_l791_791623

theorem empty_bags_even_condition (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
  (∃ (f : ℕ → ℕ × ℕ), (∀ k, (let (m', n') := f k in 
    (if m' = 0 ∧ n' = 0 then True else (1 ≤ m' ∧ 1 ≤ n') ∧ (
        (∃ p, m' = m - p ∧ n' = n - p) ∨
        (∃ q, m' = 3 * m ∧ n' = n ∨ m' = m ∧ n' = 3 * n))
    )))) ↔ even (m + n) :=
sorry

end empty_bags_even_condition_l791_791623


namespace problem_statement_l791_791924

/-- Define the four functions on the interval [0, 1] --/
def f_1 (x : ℝ) : ℝ := x
def f_2 (x : ℝ) : ℝ := x^2
def f_3 (x : ℝ) : ℝ := abs (x - 1/2)
def f_4 (x : ℝ) : ℝ := -x

/-- Define the interval [0, 1] --/
def I := set.Icc 0 1

/-- Define the convexity property for functions --/
def is_convex (f : ℝ → ℝ) := 
  ∀ (x₁ x₂ : ℝ) (λ : ℝ), (x₁ ∈ I) → (x₂ ∈ I) → (λ ∈ I) → 
   f(x₁ * λ + x₂ * (1 - λ)) ≤ λ * f(x₁) + (1 - λ) * f(x₂)

/-- Proof statement that f_1 and f_3 are convex functions --/
theorem problem_statement : 
  is_convex f_1 ∧ is_convex f_3 :=
by 
  sorry

end problem_statement_l791_791924


namespace jam_fraction_left_after_dinner_l791_791520

noncomputable def jam_left_after_dinner (initial: ℚ) (lunch_fraction: ℚ) (dinner_fraction: ℚ) : ℚ :=
  initial - (initial * lunch_fraction) - ((initial - (initial * lunch_fraction)) * dinner_fraction)

theorem jam_fraction_left_after_dinner :
  jam_left_after_dinner 1 (1/3) (1/7) = (4/7) :=
by
  sorry

end jam_fraction_left_after_dinner_l791_791520


namespace circle_center_radius_sum_eq_l791_791387

theorem circle_center_radius_sum_eq :
  let D : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 - 14*y + 73 = -y^2 + 8*x}
  ∃ a b r,
    (∀ x y, (x, y) ∈ D ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = 11 + 2*Real.sqrt 2 :=
by
  sorry

end circle_center_radius_sum_eq_l791_791387


namespace parallel_vectors_magnitude_l791_791684

variable (x : ℝ)

def a := (1 : ℝ, 2 : ℝ)
def b := (-1 : ℝ, x)

theorem parallel_vectors_magnitude :
  (1 / -1 = 2 / x) → (Real.sqrt ((-1)^2 + x^2) = Real.sqrt 5) :=
by
  intro h
  sorry

end parallel_vectors_magnitude_l791_791684


namespace jill_nail_count_l791_791047

theorem jill_nail_count {N : ℕ} 
    (h_purple : 6)
    (h_blue : 8)
    (h_difference : (8 * 100 / N) - ((N - 14) * 100 / N) = 10) : 
    N = 20 :=
  sorry

end jill_nail_count_l791_791047


namespace maximize_profits_l791_791886

variable (m : ℝ) (x : ℝ)

def w1 (m x : ℝ) := (8 - m) * x - 30
def w2 (x : ℝ) := -0.01 * x^2 + 8 * x - 80

theorem maximize_profits : 
  (4 ≤ m ∧ m < 5.1 → ∀ x, 0 ≤ x ∧ x ≤ 500 → w1 m x ≥ w2 x) ∧
  (m = 5.1 → ∀ x ≤ 300, w1 m 500 = w2 300) ∧
  (m > 5.1 ∧ m ≤ 6 → ∀ x, 0 ≤ x ∧ x ≤ 300 → w2 x ≥ w1 m x) :=
  sorry

end maximize_profits_l791_791886


namespace usual_time_to_catch_bus_l791_791847

theorem usual_time_to_catch_bus (S T : ℝ) (h : S / (4 / 5 * S) = (T + 3) / T) : T = 12 :=
by 
  sorry

end usual_time_to_catch_bus_l791_791847


namespace slope_of_l3_l791_791759

variables {Point : Type} [LinearOrderedField Point] (A B C : Point × Point)
variables (l1 l2 l3 : Point → Point → Prop)

-- Define lines and points
def line1 (x y : Point) : Prop := 4 * x - 3 * y = 2
def line2 (x y : Point) : Prop := y = 2
def pointA : Point × Point := (1, -3)
def pointB (x y : Point) : Prop := line1 x y ∧ line2 x y
def line3 (m x y xA yA : Point) (slope_pos : m > 0) : Prop :=
  y - yA = m * (x - xA)

-- Point C on line l2
def pointC (x y: Point) : Prop := line2 x y

-- Area condition
def area_triangle (x1 y1 x2 y2 x3 y3 : Point) : Point := 
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

-- Statement combining all conditions and proving the slope of l3
theorem slope_of_l3 (xB yB : Point) (hB : pointB xB yB)
  (xC1 yC1 xC2 yC2 : Point) (hC1 : pointC xC1 yC1) (hC2 : pointC xC2 yC2) 
  (s_A : (area_triangle 1 (-3) xB yB xC1 yC1) = 6 ∨ (area_triangle 1 (-3) xB yB xC2 yC2) = 6)
  : ∃ m : Point, line3 m 1 (-3) xC1 yC1 ∧ m = 25 / 17 :=
sorry

end slope_of_l3_l791_791759


namespace find_angle_AD_P_l791_791792

-- Definitions to encapsulate the conditions
variables (P : Type) [plane P]
variables (A B C D : P) -- Vertices of the square
variables (α β : ℝ) -- Given angles α and β
variable x : ℝ -- Unknown angle to be found

-- Conditions
variables (h1 : angle_between_planes (plane_square ABCD) P = α)
variables (h2 : angle_between_line_plane AB P = β)

-- Main proof statement
theorem find_angle_AD_P (P : Type) [plane P]
  (A B C D : P) (α β x : ℝ) 
  (h1 : angle_between_planes (plane_square ABCD) P = α)
  (h2 : angle_between_line_plane AB P = β) :
  x = arcsin (sqrt (sin (α - β) * sin (α + β))) :=
begin
  sorry
end

end find_angle_AD_P_l791_791792


namespace surface_area_of_circumscribed_sphere_of_triangular_pyramid_l791_791641

theorem surface_area_of_circumscribed_sphere_of_triangular_pyramid
  (a : ℝ)
  (h₁ : a > 0) : 
  ∃ S, S = (27 * π / 32 * a^2) := 
by
  sorry

end surface_area_of_circumscribed_sphere_of_triangular_pyramid_l791_791641


namespace least_value_of_x_l791_791959

theorem least_value_of_x (x : ℝ) : (4 * x^2 + 8 * x + 3 = 1) → (-1 ≤ x) :=
by
  intro h
  sorry

end least_value_of_x_l791_791959


namespace RS_is_8_l791_791036

open Real

noncomputable def distance (p1 p2 : Point) : ℝ := sorry

structure Point := (x : ℝ) (y : ℝ)

structure Triangle (A B C : Point) :=
(PQ : distance A B = 10)
(QR : distance B C = 10)
(PS : distance A (midpoint A C) = 12)
(QS : distance B (midpoint A C) = 6)
(S_on_PR : ∃ t, (midpoint A C).x = (1 - t) * A.x + t * C.x ∧ (midpoint A C).y = (1 - t) * A.y + t * C.y)

def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

theorem RS_is_8 {A B C : Point} (T : Triangle A B C) : distance B (midpoint A C) = 8 :=
sorry

end RS_is_8_l791_791036


namespace sin_cos_2x_domain_of_f_l791_791881

-- Proof Problem for Question 1
theorem sin_cos_2x (x : ℝ) (h : 2 * sin x = cos x) : (cos (2 * x)) / (1 + sin (2 * x)) = 1 / 3 :=
by sorry

-- Proof Problem for Question 2
theorem domain_of_f (x : ℝ) (k : ℤ) (h₁ : sin x > 1 / 2) (h₂ : tan x ≤ 1) :
  ∃ k : ℤ, x ∈ set.union (set.Ioc (2 * k * π + π / 6) (2 * k * π + π / 4)) (set.Ioc (2 * k * π + π / 2) (2 * k * π + 5 * π / 6)) :=
by sorry

end sin_cos_2x_domain_of_f_l791_791881


namespace interval_proof_l791_791261

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l791_791261


namespace contrapositive_of_proposition_l791_791437

theorem contrapositive_of_proposition (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0 → x ≠ 2) → (x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  assume h : x^2 - 3 * x + 2 ≠ 0 → x ≠ 2
  assume h_contra : x = 2
  sorry

end contrapositive_of_proposition_l791_791437


namespace subgroup_equality_example_l791_791582

theorem subgroup_equality_example :
  ∃ (x y : ℤ), x > 0 ∧ H = H_xy ∧ x = 7 ∧ y = 5 := 
by
  let H := Subgroup.closure ({ (3, 8), (4, -1), (5, 4) } : Set (ℤ × ℤ))
  let H_xy := Subgroup.closure ({ (0, 7), (1, 5) } : Set (ℤ × ℤ))
  use 7
  use 5
  split
  { norm_num },  -- For x > 0
  split
  { sorry },    -- Placeholder for proving H = H_xy
  split
  { refl },
  { refl }

end subgroup_equality_example_l791_791582


namespace max_value_of_f_l791_791806

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^3 + Real.cos (2 * x) - (Real.cos x)^2 - Real.sin x

theorem max_value_of_f :
  ∃ x : ℝ, f x = 5 / 27 ∧ ∀ y : ℝ, f y ≤ 5 / 27 :=
sorry

end max_value_of_f_l791_791806


namespace girl_name_correct_l791_791204

-- The Russian alphabet positions as a Lean list
def russianAlphabet : List (ℕ × Char) := [(1, 'А'), (2, 'Б'), (3, 'В'), (4, 'Г'), (5, 'Д'), (6, 'Е'), (7, 'Ё'), 
                                           (8, 'Ж'), (9, 'З'), (10, 'И'), (11, 'Й'), (12, 'К'), (13, 'Л'), 
                                           (14, 'М'), (15, 'Н'), (16, 'О'), (17, 'П'), (18, 'Р'), (19, 'С'), 
                                           (20, 'Т'), (21, 'У'), (22, 'Ф'), (23, 'Х'), (24, 'Ц'), (25, 'Ч'), 
                                           (26, 'Ш'), (27, 'Щ'), (28, 'Ъ'), (29, 'Ы'), (30, 'Ь'), (31, 'Э'), 
                                           (32, 'Ю'), (33, 'Я')]

-- The sequence of numbers representing the girl's name
def nameSequence : ℕ := 2011533

-- The corresponding name derived from the sequence
def derivedName : String := "ТАНЯ"

-- The equivalence proof statement
theorem girl_name_correct : 
  (nameSequence = 2011533 → derivedName = "ТАНЯ") :=
by
  intro h
  sorry

end girl_name_correct_l791_791204


namespace no_nat_p_prime_and_p6_plus_6_prime_l791_791603

theorem no_nat_p_prime_and_p6_plus_6_prime (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (p^6 + 6)) : False := 
sorry

end no_nat_p_prime_and_p6_plus_6_prime_l791_791603


namespace inequality_solution_l791_791979

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-3/4) ∪ Set.Ioc 4 5 ∪ Set.Ioi 5) ↔ 
  (x+2) ≠ 0 ∧ (x-2) ≠ 0 ∧ (4 * (x^2 - 1) * (x-2) - (x+2) * (7 * x - 6)) / (4 * (x+2) * (x-2)) ≥ 0 := 
by
  sorry

end inequality_solution_l791_791979


namespace shortest_distance_between_circles_is_correct_l791_791499

noncomputable def circle1_center : ℝ × ℝ := (6, 3)
noncomputable def circle1_radius : ℝ := real.sqrt 8

noncomputable def circle2_center : ℝ × ℝ := (-5, -4)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_between_centers : ℝ :=
real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

noncomputable def shortest_distance : ℝ :=
distance_between_centers - (circle1_radius + circle2_radius)

theorem shortest_distance_between_circles_is_correct :
  shortest_distance = 7.21 := by
  sorry

end shortest_distance_between_circles_is_correct_l791_791499


namespace triangle_area_with_median_l791_791980

theorem triangle_area_with_median (a b m : ℝ) (area : ℝ) 
  (h_a : a = 6) (h_b : b = 8) (h_m : m = 5) : 
  area = 24 :=
sorry

end triangle_area_with_median_l791_791980


namespace exists_positive_m_l791_791282

noncomputable def f (x : ℕ) : ℕ := 3 * x + 2

def f_iter (x : ℕ) (n : ℕ) : ℕ :=
if h : n = 0 then x else iterate (λ y, f y) n x

def divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem exists_positive_m : ∃ m : ℕ, m > 0 ∧ (divisible_by (f_iter m 100) 1988) :=
sorry

end exists_positive_m_l791_791282


namespace percentage_of_number_l791_791000

theorem percentage_of_number (N P : ℝ) (h1 : 0.60 * N = 240) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_of_number_l791_791000


namespace find_monic_cubic_polynomial_with_root_l791_791242

-- Define the monic cubic polynomial
def Q (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 6

-- Define the root condition we need to prove
theorem find_monic_cubic_polynomial_with_root (a : ℝ) (ha : a = (5 : ℝ)^(1/3) + 1) : Q a = 0 :=
by
  -- Proof goes here (omitted)
  sorry

end find_monic_cubic_polynomial_with_root_l791_791242


namespace Andrews_age_l791_791922

theorem Andrews_age (a g : ℝ) (h1 : g = 15 * a) (h2 : g - a = 55) : a = 55 / 14 :=
by
  /- proof will go here -/
  sorry

end Andrews_age_l791_791922


namespace mod_inverse_11_1000_l791_791856

theorem mod_inverse_11_1000 : ∃ (x : ℤ), 0 ≤ x ∧ x < 1000 ∧ 11 * x % 1000 = 1 :=
by
  use 91
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end mod_inverse_11_1000_l791_791856


namespace triangle_abc_C_triangle_abc_area_l791_791351

variable (c b : ℝ) (B C : ℝ)

axiom h_c : c = √3
axiom h_b : b = 1
axiom h_B : B = 30

theorem triangle_abc_C : C = 60 ∨ C = 120 :=
sorry

theorem triangle_abc_area : 
  (C = 60 → (1/2) * b * c = √3 / 2) ∧ 
  (C = 120 → (1/2) * b * c * sin (60 : ℝ) = √3 / 4) :=
sorry

end triangle_abc_C_triangle_abc_area_l791_791351


namespace solution_l791_791975

def N_star := { n : ℕ // n > 0 }

def f (n : N_star) : N_star := sorry

theorem solution (f : N_star → N_star) :
  (∀ m n : N_star, (f m).val * (f m).val + (f n).val ∣ (m.val * m.val + n.val) * (m.val * m.val + n.val)) →
  (∀ n : N_star, f n = n) := 
sorry

end solution_l791_791975


namespace weight_of_B_l791_791871

theorem weight_of_B (A B C : ℝ) (h1 : (A + B + C) / 3 = 45) (h2 : (A + B) / 2 = 40) (h3 : (B + C) / 2 = 46) : B = 37 :=
by
  sorry

end weight_of_B_l791_791871


namespace number_of_possible_values_for_r_l791_791447

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def is_five_place_decimal (r : ℚ) : Prop :=
  ∃ (a b c d e : ℕ), 
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
    r = (a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e) * 10^(-5)

def closest_fraction_to_r (r : ℚ) : Prop :=
  let fractions := [2/9, 1/4, 3/10, 3/11] in
  3/11 = fractions.min_by (λ f, abs (r - f))

def value_range (r : ℚ) : Prop :=
  26137 / 10^5 ≤ r ∧ r ≤ 28635 / 10^5

theorem number_of_possible_values_for_r :
  ∃ (n : ℕ), n = 2499 ∧ 
  ∀ (r : ℚ),
    is_five_place_decimal r →
    closest_fraction_to_r r →
    value_range r →
    n = 2499 :=
by {
  sorry
}

end number_of_possible_values_for_r_l791_791447


namespace grains_in_one_tsp_l791_791470

-- Definitions based on conditions
def grains_in_one_cup : Nat := 480
def half_cup_is_8_tbsp : Nat := 8
def one_tbsp_is_3_tsp : Nat := 3

-- Theorem statement
theorem grains_in_one_tsp :
  let grains_in_half_cup := grains_in_one_cup / 2
  let grains_in_one_tbsp := grains_in_half_cup / half_cup_is_8_tbsp
  grains_in_one_tbsp / one_tbsp_is_3_tsp = 10 :=
by
  sorry

end grains_in_one_tsp_l791_791470


namespace number_of_girls_l791_791832

theorem number_of_girls (T B : ℕ) (hT : T = 62) (hB : B = 27) : T - B = 35 := by
  rw [hT, hB]
  norm_num
  sorry

end number_of_girls_l791_791832


namespace bottles_remaining_l791_791403

-- Define the initial number of bottles.
def initial_bottles : ℝ := 45.0

-- Define the number of bottles Maria drank.
def maria_drinks : ℝ := 14.0

-- Define the number of bottles Maria's sister drank.
def sister_drinks : ℝ := 8.0

-- The value that needs to be proved.
def bottles_left : ℝ := initial_bottles - maria_drinks - sister_drinks

-- The theorem statement.
theorem bottles_remaining :
  bottles_left = 23.0 :=
by
  sorry

end bottles_remaining_l791_791403


namespace even_function_m_eq_neg_one_l791_791345

theorem even_function_m_eq_neg_one (m : ℝ) :
  (∀ x : ℝ, (m - 1)*x^2 - (m^2 - 1)*x + (m + 2) = (m - 1)*(-x)^2 - (m^2 - 1)*(-x) + (m + 2)) →
  m = -1 :=
  sorry

end even_function_m_eq_neg_one_l791_791345


namespace cube_volume_l791_791461

theorem cube_volume (sum_edges : ℝ) (h : sum_edges = 48) : 
  ∃ V : ℝ, V = 64 := 
by
  let s : ℝ := sum_edges / 12
  let V : ℝ := s^3
  use V
  have hs : s = 4 := by
    calc
      s = 48 / 12 : by rw [h]
      ... = 4 : by norm_num
  rw hs
  show V = 64
  calc
    V = s^3 : by rfl
    ... = 4^3 : by rw hs
    ... = 64 : by norm_num

end cube_volume_l791_791461


namespace divide_2xn_rectangle_l791_791331

noncomputable def r (n : ℕ) : ℝ :=
  if n = 1 then 2
  else if n = 2 then 8
  else 6 * r (n - 1) - 7 * r (n - 2)

theorem divide_2xn_rectangle (n : ℕ) (h : n > 0) :
  r n = (3 - real.sqrt 2) / 4 * (3 + real.sqrt 2)^n + (3 + real.sqrt 2) / 4 * (3 - real.sqrt 2)^n :=
by
  sorry

end divide_2xn_rectangle_l791_791331


namespace select_medical_team_l791_791622

open Nat

theorem select_medical_team : 
  let male_doctors := 5
  let female_doctors := 4
  let selected_doctors := 3
  (male_doctors.choose 1 * female_doctors.choose 2 + male_doctors.choose 2 * female_doctors.choose 1) = 70 :=
by
  sorry

end select_medical_team_l791_791622


namespace max_positive_iff_a_gt_neg5_l791_791605

def f (x a : ℝ) : ℝ := x^3 + 3*x + a - 9
def g (x a : ℝ) : ℝ := a + 2^(5-x) - 3^(x-1)

theorem max_positive_iff_a_gt_neg5 (a : ℝ) :
  (∀ x : ℝ, max (f x a) (g x a) > 0) ↔ a > -5 := sorry

end max_positive_iff_a_gt_neg5_l791_791605


namespace boss_salary_percentage_increase_l791_791485

theorem boss_salary_percentage_increase (W B : ℝ) (h : W = 0.2 * B) : ((B / W - 1) * 100) = 400 := by
sorry

end boss_salary_percentage_increase_l791_791485


namespace find_side_lengths_l791_791109

variable (P Q R S : Type)
variable [Field P]
variable [Field Q]
variable [Field R]
variable [Field S]

def radius : ℝ := 5
def lengthPQ : ℝ := 6
def lengthQR : ℝ := 6
def areaPQRS : ℝ := 48
def lengthRS : ℝ := 4 * Real.sqrt 13
def lengthSP : ℝ := 4 * Real.sqrt 13

theorem find_side_lengths
  (h_area : PQRS.Area = areaPQRS)
  (h_lengthPQ : P.length Q = lengthPQ)
  (h_lengthQR : Q.length R = lengthQR)
  (h_equal_lengths : R.length S = S.length P)
  (h_circumcircle : Mathlib.Circumference {P, Q, R} radius) :
  R.length S = lengthRS ∧ S.length P = lengthSP :=
sorry

end find_side_lengths_l791_791109


namespace circle_center_tangent_lines_l791_791540

theorem circle_center_tangent_lines 
    (center : ℝ × ℝ)
    (h1 : 3 * center.1 + 4 * center.2 = 10)
    (h2 : center.1 = 3 * center.2) : 
    center = (30 / 13, 10 / 13) := 
by {
  sorry
}

end circle_center_tangent_lines_l791_791540


namespace valid_water_amounts_l791_791875

noncomputable def can_obtain_30_percent_syrup (N : ℕ) : Prop :=
  ∃ (A B C : ℕ) (steps : ℕ → List (ℕ × ℕ × ℕ)), 
    A = 3 ∧ B = N ∧ C = 0 ∧
    (∀ t ∈ steps, 
      ∃ (A' B' C' : ℕ), (t = (A', B', C')) ∧ 
      (A + B + C = A' + B' + C') ∧ 
      ((A' = 0 ∨ B' = 0 ∨ C' = 0) ∨ 
      (A' = 3 ∨ A' = B' ∨ A' = C'))) ∧ 
    (steps.maximum (λ t, t.1) = 10 ∧ 
     (steps.maximum (λ t, t.1)/= 3 ∧ 
     (steps.maximum (λ t, t.2) = 10 ∧ 
      steps.maximum (λ t, t.2) ≠ 3 ∧ 
      (steps.maximum (λ t, t.3) = 10 ∧ 
       steps.maximum (λ t, t.3) ≠ 3)))

theorem valid_water_amounts (N : ℕ) : can_obtain_30_percent_syrup N ↔ ¬ N % 3 = 0 :=
by
  sorry

end valid_water_amounts_l791_791875


namespace December_sales_fraction_l791_791594

namespace DepartmentStore

variable (A Q P : ℝ)

theorem December_sales_fraction :
  let first_year_Dec := Q * A,
      second_year_Dec := Q * A * (1 + P / 100),
      third_year_Dec := Q * A * (1 + P / 100)^2,
      total_Dec := first_year_Dec + second_year_Dec + third_year_Dec,
      total_sales := 3 * 11 * A + total_Dec 
  in total_Dec / total_sales = Q * (1 + (1 + P / 100) + (1 + P / 100)^2) / (3 * 11 + Q + Q * (1 + P / 100) + Q * (1 + P / 100)^2) :=
by
  sorry

end DepartmentStore

end December_sales_fraction_l791_791594


namespace fraction_zero_implies_value_l791_791009

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l791_791009


namespace triangle_sym_Barycentric_l791_791374

theorem triangle_sym_Barycentric (X Y Z P Q R : Point) 
  (hP : lineThrough X Y ∧ P ∈ lineThrough X Y ∧ barycentric_coord P X Y = (4, 1))
  (hQ : lineThrough Y Z ∧ Q ∈ lineThrough Y Z ∧ barycentric_coord Q Y Z = (4, 1))
  (hR : R = lineIntersection (lineThrough P Q) (lineThrough X Z)) :
  PQ / QR = 1 / 5 :=
  sorry

end triangle_sym_Barycentric_l791_791374


namespace parallelogram_base_l791_791247

theorem parallelogram_base (A h b : ℝ) (hA : A = 375) (hh : h = 15) : b = 25 :=
by
  sorry

end parallelogram_base_l791_791247


namespace exists_universal_accessible_city_l791_791708

-- Define the basic structure for cities and flights
structure Country :=
  (City : Type)
  (accessible : City → City → Prop)

namespace Country

-- Define the properties of accessibility in the country
variables {C : Country}

-- Axiom: Each city is accessible from itself
axiom self_accessible (A : C.City) : C.accessible A A

-- Axiom: For any two cities, there exists a city from which both are accessible
axiom exists_intermediate (P Q : C.City) : ∃ R : C.City, C.accessible R P ∧ C.accessible R Q

-- Definition of the main theorem
theorem exists_universal_accessible_city :
  ∃ U : C.City, ∀ A : C.City, C.accessible U A :=
sorry

end Country

end exists_universal_accessible_city_l791_791708


namespace angle_between_planes_zero_l791_791606

-- Define the planes in the 3D space
def plane1 : ℝ × ℝ × ℝ × ℝ := (2, -6, 14, -1)
def plane2 : ℝ × ℝ × ℝ × ℝ := (5, -15, 35, -3)

-- Compute the normal vectors of the planes
def normal_vector (plane : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (plane.1, plane.2, plane.3)

def n1 := normal_vector plane1
def n2 := normal_vector plane2

-- Compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Compute the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Prove that the angle between the planes is zero
theorem angle_between_planes_zero : 
  let cos_phi := dot_product n1 n2 / (magnitude n1 * magnitude n2) in
  Real.arccos cos_phi = 0 :=
by
  let cos_phi := dot_product n1 n2 / (magnitude n1 * magnitude n2)
  sorry

end angle_between_planes_zero_l791_791606


namespace discount_is_25_percent_l791_791968

-- Conditions
def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def total_spent : ℝ := 135
def nieces_number : ℝ := 3

-- Given total cost calculation
def total_cost_per_niece : ℝ := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
def total_cost_without_discount : ℝ := nieces_number * total_cost_per_niece
def discount_amount : ℝ := total_cost_without_discount - total_spent
def discount_percentage : ℝ := (discount_amount / total_cost_without_discount) * 100

-- The proof goal
theorem discount_is_25_percent : discount_percentage = 25 :=
by
  sorry

end discount_is_25_percent_l791_791968


namespace fraction_of_automobile_installment_credit_extended_by_finance_companies_l791_791228

theorem fraction_of_automobile_installment_credit_extended_by_finance_companies
  (total_consumer_credit : ℝ)
  (percentage_auto_credit : ℝ)
  (credit_extended_by_finance_companies : ℝ)
  (total_auto_credit_fraction : percentage_auto_credit = 0.36)
  (total_consumer_credit_value : total_consumer_credit = 475)
  (credit_extended_by_finance_companies_value : credit_extended_by_finance_companies = 57) :
  credit_extended_by_finance_companies / (percentage_auto_credit * total_consumer_credit) = 1 / 3 :=
by
  -- The proof part will go here.
  sorry

end fraction_of_automobile_installment_credit_extended_by_finance_companies_l791_791228


namespace perfect_square_of_factorials_l791_791865

open Nat

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_of_factorials :
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  is_perfect_square E3 :=
by
  -- definition of E1, E2, E3, E4, E5 as expressions given conditions
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  
  -- specify that E3 is the perfect square
  show is_perfect_square E3

  sorry

end perfect_square_of_factorials_l791_791865


namespace annie_bought_figurines_l791_791224

theorem annie_bought_figurines:
  let televisions := 5
  let cost_per_television := 50
  let total_spent := 260
  let cost_per_figurine := 1
  let cost_of_televisions := televisions * cost_per_television
  let remaining_money := total_spent - cost_of_televisions
  remaining_money / cost_per_figurine = 10 :=
by
  sorry

end annie_bought_figurines_l791_791224


namespace bake_sale_money_made_l791_791942

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end bake_sale_money_made_l791_791942


namespace election_result_l791_791361

-- Definitions
variable {V : ℝ} -- Total number of votes
def votes_A := 0.35 * V  -- Candidate A's votes
def votes_C := 0.25 * V  -- Candidate C's votes
def votes_B := 0.4 * V  -- Candidate B's votes

-- Additional Conditions
def B_more_than_A := votes_B = votes_A + 2460

-- Questions
def percentage_votes_B := 100 - (35 + 25)

-- Proof Statement
theorem election_result (h : B_more_than_A) : 
  percentage_votes_B = 40 ∧ V = 49200 := by
  sorry

end election_result_l791_791361


namespace B_D_G_X_concyclic_l791_791053

-- Definitions representing the geometric setup
variables (A B C D E F G X : Type)
variable [acute triangle ABC] -- ABC is an acute triangle
variable [AC : length_of_side A C > length_of_side A B] -- AC > AB
variable [intouch_triangle DEF ABC] -- DEF is the intouch triangle of ABC
variable [D_on_BC : point_on_segment D B C] -- D is on BC
variable [E_on_AC : point_on_segment E A C] -- E is on AC
variable [F_on_AB : point_on_segment F A B] -- F is on AB
variable [G_intersection : perpendicular_intersection D E F G A B] -- G is the intersection of the perpendicular from D to EF with AB
variable [X_intersection : circumcircle_intersection ABC AEF X] -- X is the intersection of (ABC) and (AEF)

-- The theorem to prove
theorem B_D_G_X_concyclic : cyclic {B, D, G, X} :=
sorry

end B_D_G_X_concyclic_l791_791053


namespace polynomial_expansion_identity_l791_791688

theorem polynomial_expansion_identity (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  let p := (2 + real.sqrt 3)^6
  let q := (-2 + real.sqrt 3)^6
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = p) →
  (a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 = q) →
  (a_0 + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5)^2 = 1 :=
by {
  intros,
  sorry
}

end polynomial_expansion_identity_l791_791688


namespace speed_of_stream_l791_791173

theorem speed_of_stream (v_d v_u : ℝ) (h_d : v_d = 13) (h_u : v_u = 8) :
  (v_d - v_u) / 2 = 2.5 :=
by
  -- Insert proof steps here
  sorry

end speed_of_stream_l791_791173


namespace algebraic_identity_l791_791995

theorem algebraic_identity (a : ℚ) (h : a + a⁻¹ = 3) : a^2 + a⁻¹^2 = 7 := 
  sorry

end algebraic_identity_l791_791995


namespace average_inversions_l791_791287

-- Definition of inversion in a permutation
def is_inversion (σ : List ℕ) (j k : ℕ) : Prop :=
  1 ≤ j ∧ j < k ∧ k ≤ σ.length ∧ σ.nth_le (j - 1) sorry > σ.nth_le (k - 1) sorry

-- Number of inversions in a permutation
def num_inversions (σ : List ℕ) : ℕ :=
  ((List.range (σ.length)).bind (λ j, (List.range (σ.length)).filter (λ k, is_inversion σ (j + 1) (k + 1)))).length

-- The average number of inversions over all permutations of (1, 2, 3, ..., n)
theorem average_inversions (n : ℕ) : ∑ σ in (List.permutations (List.range n)), num_inversions σ / (n.factorial : ℝ) = n * (n - 1) / 4 := sorry

end average_inversions_l791_791287


namespace polynomial_factor_c_zero_l791_791269

theorem polynomial_factor_c_zero (c q : ℝ) :
    ∃ q : ℝ, (3*q + 6 = 0 ∧ c = 6*q + 12) ↔ c = 0 :=
by
  sorry

end polynomial_factor_c_zero_l791_791269


namespace find_some_multiplier_l791_791598

theorem find_some_multiplier (some_multiplier : ℤ)
    (h : | 5 - some_multiplier * (3 - 12) | - | 5 - 11 | = 71) :
    some_multiplier = 8 := by
  sorry

end find_some_multiplier_l791_791598


namespace find_m_l791_791340

theorem find_m (m : ℕ) :
  (∀ x : ℝ, -2 * x ^ 2 + 5 * x - 2 <= 9 / m) →
  m = 8 :=
sorry

end find_m_l791_791340


namespace line_parabola_vertex_unique_l791_791990

theorem line_parabola_vertex_unique (a : ℝ) :
    (∃ x : ℝ, x^3 - 3 * a * x + a^2 = x + a) → a = 0 :=
begin
  sorry
end

end line_parabola_vertex_unique_l791_791990


namespace servings_of_peanut_butter_l791_791206

-- Definitions from conditions
def total_peanut_butter : ℚ := 35 + 4/5
def serving_size : ℚ := 2 + 1/3

-- Theorem to be proved
theorem servings_of_peanut_butter :
  total_peanut_butter / serving_size = 15 + 17/35 := by
  sorry

end servings_of_peanut_butter_l791_791206


namespace smallest_integer_y_l791_791861

theorem smallest_integer_y (y : ℤ) (h : 8 - 3 * y ≤ 23) : y ≥ -5 :=
by {
  sorry
}

example : ∃ y : ℤ, (8 - 3 * y ≤ 23) ∧ (∀ z : ℤ, 8 - 3 * z ≤ 23 → y ≤ z) :=
by {
  use -5,
  split,
  { linarith, },
  { intros z hz,
    have : -3 * y ≤ -3 * z, from hz,
    linarith, }
}

end smallest_integer_y_l791_791861


namespace probability_green_marbles_correct_l791_791195

noncomputable def probability_of_two_green_marbles : ℚ :=
  let total_marbles := 12
  let green_marbles := 7
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green

theorem probability_green_marbles_correct :
  probability_of_two_green_marbles = 7 / 22 := by
    sorry

end probability_green_marbles_correct_l791_791195


namespace tunnel_length_l791_791133

-- Defining the conditions
def train_length : ℝ := 1
def train_speed_mph : ℝ := 60
def time_diff_minutes : ℝ := 3

-- Computing required values from the conditions
def train_speed_mpm : ℝ := train_speed_mph / 60
def distance_covered_by_front : ℝ := train_speed_mpm * time_diff_minutes

-- Proof statement
theorem tunnel_length (train_length : ℝ) (train_speed_mph : ℝ) (time_diff_minutes : ℝ) 
  (train_speed_mpm : ℝ := train_speed_mph / 60)
  (distance_covered_by_front : ℝ := train_speed_mpm * time_diff_minutes) : 
  distance_covered_by_front - train_length = 2 := 
  by
  sorry

end tunnel_length_l791_791133


namespace sum_of_digits_in_repeating_decimal_l791_791578

theorem sum_of_digits_in_repeating_decimal (n : ℕ) (h_n_period : n = 400) :
  (let dec : String := "000102030405...9998009901" in -- ensuring dec is the repeating sequence
  let digits := dec.toList.map (λ ch, (ch.toNat - '0'.toNat)) -- converting char to digit
  let period_digits := digits.take n in -- take the first n digits of the repeating sequence
  period_digits.sum) = 1800 :=
by 
  sorry -- proof goes here

end sum_of_digits_in_repeating_decimal_l791_791578


namespace striped_shirts_more_than_shorts_l791_791353

theorem striped_shirts_more_than_shorts :
  ∀ (total_students striped_students checkered_students short_students : ℕ),
    total_students = 81 →
    striped_students = total_students * 2 / 3 →
    checkered_students = total_students - striped_students →
    short_students = checkered_students + 19 →
    striped_students - short_students = 8 :=
by
  intros total_students striped_students checkered_students short_students
  sorry

end striped_shirts_more_than_shorts_l791_791353


namespace how_many_divisors_31752_l791_791686

theorem how_many_divisors_31752 : 
  let n := 31752
  count (λ d, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ n % d = 0) [1, 2, 3, 4, 5, 6, 7, 8, 9] = 7 :=
by sorry

end how_many_divisors_31752_l791_791686


namespace isosceles_triangle_base_angle_l791_791921

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle : a + b + c = 180)
  (h_iso : a = b ∨ b = c ∨ a = c) (h_interior : a = 50 ∨ b = 50 ∨ c = 50) :
  c = 50 ∨ c = 65 :=
by sorry

end isosceles_triangle_base_angle_l791_791921


namespace number_of_positive_integers_l791_791960

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def polynomial (n : ℕ) : ℤ :=
  n^3 - 6 * n^2 + 17 * n - 19

theorem number_of_positive_integers (n : ℕ) :
  ∃ k : ℕ, k = 3 ∧ 
  (∀ m : ℕ, m ≤ n → (is_prime (polynomial m) → m = 3 ∨ m = 4 ∨ m = 5)) :=
sorry

end number_of_positive_integers_l791_791960


namespace exists_c_for_int_zero_l791_791055

variable {f : ℝ → ℝ}
variable [ContinuousOn [0,1] f]

theorem exists_c_for_int_zero (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_int : ∫ x in 0..1, f x = 0) :
  ∃ c ∈ (0 : ℝ)..(1 : ℝ), ∫ x in 0..c, x * f x = 0 :=
begin
  sorry
end

end exists_c_for_int_zero_l791_791055


namespace count_correct_statements_l791_791221

theorem count_correct_statements
  (S1 : "The circumcenter of a triangle is the intersection point of the perpendicular bisectors of the three sides of the triangle." → True)
  (S2 : "If two arcs have equal degrees, then they are congruent arcs." → False)
  (S3 : "The diameter is the longest chord in a circle." → True)
  (S4 : "If two chords are equal, then the central angles they subtend are equal." → False) :
  ([S1, S2, S3, S4].filter (λ s, s)).length = 2 :=
  sorry

end count_correct_statements_l791_791221


namespace interval_intersection_l791_791252

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l791_791252


namespace value_of_Q_l791_791399

theorem value_of_Q (n : ℕ) (h : n = 2010) :
  let Q := (∏ k in finset.range (n - 2) + 2, (1 - 1 / (k + 3))) in Q = 1 / 1005 :=
by
  unfold_projs,
  subst h,
  norm_num,
  sorry

end value_of_Q_l791_791399


namespace simplify_expression_eq_l791_791531

theorem simplify_expression_eq (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a - 1/a) / ((a^2 - 2 * a + 1) / a) = (a + 1) / (a - 1) :=
by
  sorry

end simplify_expression_eq_l791_791531


namespace infinitely_many_primes_such_that_ap_plus_b_is_composite_l791_791088

open Nat

theorem infinitely_many_primes_such_that_ap_plus_b_is_composite 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ∃^∞ p, Prime p ∧ ¬ Prime (a * p + b) :=
by
  sorry

end infinitely_many_primes_such_that_ap_plus_b_is_composite_l791_791088


namespace common_divisors_120_90_45_l791_791685

def divisors (n : ℤ) : set ℤ :=
  {d | d ∣ n}

def common_divisors_count (a b c : ℤ) := 
  (divisors a ∩ divisors b ∩ divisors c).size

theorem common_divisors_120_90_45 : common_divisors_count 120 90 45 = 8 :=
by 
  sorry

end common_divisors_120_90_45_l791_791685


namespace min_val_of_f_l791_791446

-- Define max function
def max (a b : ℝ) : ℝ :=
if a >= b then a else b

-- Define the function that we're interested in
def f (x : ℝ) : ℝ := max (x^2) (x + 2)

-- The statement that we need to prove
theorem min_val_of_f : ∃ x : ℝ, ∀ y : ℝ, f y >= f x ∧ f x = 1 := by
  sorry

end min_val_of_f_l791_791446


namespace polynomial_ratio_l791_791677

noncomputable theory

-- Definitions of polynomials f and Q
def f (x : ℝ) : ℝ := x^2009 + 19 * x^2008 + 1

def has_distinct_real_roots (p : ℝ → ℝ) : Prop :=
  ∃ s : Fin 2009 → ℝ, (∀ i j : Fin 2009, i ≠ j → s i ≠ s j) ∧ (∀ x, p x = ∏ i, (x - s i))

def Q (z : ℝ) : ℝ :=
  let s : Fin 2009 → ℝ := classical.some (classical.some_spec (classical.some_spec (Exists.intro (λ x, f x = 0) sorry)))
  in ∏ i, (z - (s i + 1 / s i))

-- The main theorem statement
theorem polynomial_ratio: 
  (has_distinct_real_roots f) →
  (∀ j : Fin 2009, Q (classical.some (classical.some_spec (classical.some_spec (Exists.intro (λ x, f x = 0) sorry))) j + 1 / classical.some (classical.some_spec (classical.some_spec (Exists.intro (λ x, f x = 0) sorry))) j) = 0) →
  Q 1 / Q (-1) = 289 / 291 :=
begin
  sorry
end

end polynomial_ratio_l791_791677


namespace proof_about_lines_l791_791168

def statement_A (a b x y : ℝ) : Prop := (x / a + y / b = 1) -> (x = 0 ∧ y = 0)

def statement_B {ℓ₁ ℓ₂ : Type} [AffineSpace ℝ ↑ℓ₁] [AffineSpace ℝ ℓ₂] {a₁ a₂ b₁ b₂ : ℝ} (slope₁ slope₂ : ℝ) : Prop := 
(slope₁ ≠ slope₂ ∧ (a₁/b₁ ≠ a₂/b₂))

def statement_C (x1 y1 x2 y2 x y : ℝ) : Prop := (x - x1) * (y2 - y1) = (y - y1) * (x2 - x1)

def statement_D (m1 : ℝ) : Prop := ¬(m1 = 0)

theorem proof_about_lines : ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end proof_about_lines_l791_791168


namespace circles_externally_tangent_l791_791129

noncomputable def circle_center_radius (h : ℝ) (k : ℝ) (r : ℝ) (x y : ℝ) : Prop :=
(x - h)^2 + (y - k)^2 = r^2

theorem circles_externally_tangent :
  (∀ x y, circle_center_radius 4 0 3 x y ↔ (x - 4)^2 + y^2 = 9) ∧
  (∀ x y, circle_center_radius 0 3 2 x y ↔ x^2 + (y - 3)^2 = 4) →
  dist (4, 0) (0, 3) = 5 →
  5 = 3 + 2 →
  externally_tangent (circle_center_radius 4 0 3) (circle_center_radius 0 3 2) :=
by
  sorry

-- Auxiliary function to calculate the distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define external tangency
definition externally_tangent
  (c1 c2 : ℝ → ℝ → Prop) : Prop :=
∃ (h1 k1 r1 h2 k2 r2 : ℝ),
  (∀ x y, c1 x y ↔ (x - h1)^2 + (y - k1)^2 = r1^2) ∧
  (∀ x y, c2 x y ↔ (x - h2)^2 + (y - k2)^2 = r2^2) ∧
  dist (h1, k1) (h2, k2) = r1 + r2

end circles_externally_tangent_l791_791129


namespace triangle_is_equilateral_l791_791303

-- Define the conditions as assumptions
variables (a b c : ℝ) 
hypothesis1 : 2 * b = a + c
hypothesis2 : 2 * Real.sqrt b = Real.sqrt a + Real.sqrt c

-- Theorem statement
theorem triangle_is_equilateral 
  (h1 : 2 * b = a + c)
  (h2 : 2 * Real.sqrt b = Real.sqrt a + Real.sqrt c) : 
  a = b ∧ b = c := 
sorry

end triangle_is_equilateral_l791_791303


namespace triangle_inequalities_equalities_hold_when_equilateral_l791_791411

theorem triangle_inequalities (ABC : Triangle) (r S p R : ℝ) 
(h_r : r = inradius ABC)
(h_S : S = area ABC)
(h_p : p = semi_perimeter ABC)
(h_R : R = circumradius ABC) :
r ≤ (Real.sqrt (Real.sqrt (3 * S))) / 3 ∧
(Real.sqrt (Real.sqrt (3 * S))) / 3 ≤ (Real.sqrt 3 * p) / 9 ∧
(Real.sqrt 3 * p) / 9 ≤ (1 / 2) * R :=
begin
  sorry
end

-- Additional theorem to check when equalities hold
theorem equalities_hold_when_equilateral (ABC : Triangle) (r S p R : ℝ) 
(h_r : r = inradius ABC)
(h_S : S = area ABC)
(h_p : p = semi_perimeter ABC)
(h_R : R = circumradius ABC)
(h_equilateral : is_equilateral ABC) :
r = (Real.sqrt (Real.sqrt (3 * S))) / 3 ∧
(Real.sqrt (Real.sqrt (3 * S))) / 3 = (Real.sqrt 3 * p) / 9 ∧
(Real.sqrt 3 * p) / 9 = (1 / 2) * R :=
begin
  sorry
end

end triangle_inequalities_equalities_hold_when_equilateral_l791_791411


namespace scientific_notation_correct_l791_791114

-- Define the conditions
def circumference_circular_orbit_chinese_space_station := 43000000

-- Define the target scientific notation representation
def scientific_notation_representation := 4.3 * 10 ^ 7

-- State the problem: Prove that the given number in scientific notation is equal to the circumference
theorem scientific_notation_correct : 
  circumference_circular_orbit_chinese_space_station = scientific_notation_representation :=
by
  sorry

end scientific_notation_correct_l791_791114


namespace semicircle_radius_l791_791797

theorem semicircle_radius (a : ℝ) (h₁ : a = 48) (h₂ : a > 0) : 
  let inner_square_side := (a / 2) * Real.sqrt 2 in
  let radius := inner_square_side / 2 in
  radius = 12 := by
  sorry

end semicircle_radius_l791_791797


namespace cube_root_minimization_l791_791576

def c := 18
def d := 1
def n := 5832

theorem cube_root_minimization : (nat.cubeRoot n = c * nat.cubeRoot d) → (c + d = 19) := by
  intro h
  sorry

end cube_root_minimization_l791_791576


namespace interval_of_x_l791_791266

theorem interval_of_x (x : ℝ) : 
  (2 < 3 * x ∧ 3 * x < 3) ∧ 
  (2 < 4 * x ∧ 4 * x < 3) ↔ 
  x ∈ set.Ioo (2 / 3) (3 / 4) := 
sorry

end interval_of_x_l791_791266


namespace circumradius_of_triangle_l791_791889

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : (c / 2) = 7.5 := 
by
  have hc : c = 15 := h₃
  have radius : ℝ := c / 2
  show radius = 7.5
  sorry

end circumradius_of_triangle_l791_791889


namespace number_of_sets_satisfying_condition_l791_791125

-- Define the elements a, b, c as distinct
axiom a b c : Type

-- The given condition: the union of {a, b} and set A equals {a, b, c}
def satisfies_condition (A : set a) : Prop := ({a, b} ∪ A = {a, b, c})

-- The proof goal: Prove there are 4 sets that satisfy the condition
theorem number_of_sets_satisfying_condition : 
  (finset.filter satisfies_condition (finset.powerset {a, b, c})).card = 4 :=
sorry

end number_of_sets_satisfying_condition_l791_791125


namespace variance_scaled_l791_791138

theorem variance_scaled {s^2 : ℝ} (s2 : ℝ) (h : s2 = s^2) : 4 * s^2 = 4 * s2 :=
by
  sorry

end variance_scaled_l791_791138


namespace probability_of_perpendicular_edges_l791_791947

def is_perpendicular_edge (e1 e2 : ℕ) : Prop :=
-- Define the logic for identifying perpendicular edges here
sorry

def total_outcomes : ℕ := 81

def favorable_outcomes : ℕ :=
-- Calculate the number of favorable outcomes here
20 + 6 + 18

theorem probability_of_perpendicular_edges : 
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 44 / 81 := by
-- Proof for calculating the probability
sorry

end probability_of_perpendicular_edges_l791_791947


namespace siblings_pizza_order_l791_791987

theorem siblings_pizza_order :
  let anna_fraction := (1 : ℝ) / 6
  let ben_fraction := (1 : ℝ) / 4
  let carol_fraction := (1 : ℝ) / 3
  let dennis_fraction := (1 : ℝ) / 8
  let elly_fraction := 1 - (anna_fraction + ben_fraction + carol_fraction + dennis_fraction)
  let pizza_parts := [
    (carol_fraction, "Carol"),
    (ben_fraction, "Ben"),
    (anna_fraction, "Anna"),
    (dennis_fraction, "Dennis"),
    (elly_fraction, "Elly")
  ]
  ∀ sorted_parts in list.sort (λ a b => a.1 > b.1) pizza_parts,
  sorted_parts = [
    (carol_fraction, "Carol"),
    (ben_fraction, "Ben"),
    (anna_fraction, "Anna"),
    (dennis_fraction, "Dennis"),
    (elly_fraction, "Elly")
  ] :=
by sorry

end siblings_pizza_order_l791_791987


namespace geometric_sequence_common_ratio_l791_791005

variable {n : ℕ}
variable {T_n : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {q : ℝ}

-- The common ratio q of the geometric sequence and product of the first n terms T_n.
def common_ratio (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

def product_of_first_n_terms (T_n : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), T_n n = ∏ i in finset.range n, b (i + 1)

theorem geometric_sequence_common_ratio (b : ℕ → ℝ) (q : ℝ) (T_n : ℕ → ℝ) (h1 : common_ratio b q) (h2 : product_of_first_n_terms T_n b) :
  ∃ c : ℝ, ∀ n : ℕ, T_n n = c * (q^n) := 
by
  sorry

end geometric_sequence_common_ratio_l791_791005


namespace fourth_product_is_correct_l791_791722
-- Import Mathlib for broader library support

-- Define the row of random numbers explicitly
def second_row : List ℕ := [7216, 2884, 3660, 1073, 4366, 7575]

-- Function to select the 'n-th' product number from the row following the described method
def product_number (n : ℕ) : ℕ :=
  second_row.drop 2 -- Drop the first two groups
             .init -- Remove the last number to fit the problem's 50 selections
             .nth_le (n-1) sorry -- Select the nth number

-- Define the target product number for the fourth selection
def fourth_product_number := 447

-- The theorem stating that the number of the fourth product selected is 447
theorem fourth_product_is_correct : product_number 4 = fourth_product_number := 
  sorry

end fourth_product_is_correct_l791_791722


namespace intersection_sets_l791_791323

theorem intersection_sets :
  (∀ x : ℝ, 0 < x ∧ x < 4 → ∃ y : ℝ, y = 3 * x + 2 ∧ 2 < y ∧ y < 14) :=
by
  intro x
  intro hx
  dsimp at hx
  sorry

end intersection_sets_l791_791323


namespace Maria_age_in_2010_l791_791732

theorem Maria_age_in_2010 (m : ℝ) (m_age_in_2000 : m = 20.5) (sum_birth_years : (2000 - m) + (2000 - 3 * m) = 3918) :
  m + 10 = 30.5 :=
by
  have h : 4000 - 4 * m = 3918
  {
    calc
      (2000 - m) + (2000 - 3 * m) = 4000 - 4 * m : by linarith
                                ... = 3918         : sum_birth_years
  }
  have : -4 * m = -82
  {
    linarith
  }
  have hm : m = 20.5
  {
    linarith
  }
  have Maria_age_2010 : m + 10 = 30.5
  {
    linarith
  }
  exact Maria_age_2010

end Maria_age_in_2010_l791_791732


namespace percentage_psychology_liberal_arts_l791_791927

variable (T : ℝ)
variable (H1 : 0.80 * T = T_freshmen)
variable (H2 : 0.60 * T_freshmen = T_liberal_arts_freshmen)
variable (H3 : 0.24 * T = T_psychology_liberal_arts_freshmen)

theorem percentage_psychology_liberal_arts (T_freshmen T_liberal_arts_freshmen T_psychology_liberal_arts_freshmen : ℝ)  
: (T_psychology_liberal_arts_freshmen / T_liberal_arts_freshmen) * 100 = 50 :=
by
  have H_freshmen : T_freshmen = 0.80 * T := by exact H1
  have H_liberal_arts : T_liberal_arts_freshmen = 0.60 * T_freshmen := by exact H2
  have H_psychology : T_psychology_liberal_arts_freshmen = 0.24 * T := by exact H3
  sorry

end percentage_psychology_liberal_arts_l791_791927


namespace smallest_integer_larger_than_970_l791_791501

noncomputable def smallest_integer_larger_than : ℕ :=
  Nat.ceil ((Real.sqrt 3 + Real.sqrt 2) ^ 6)

theorem smallest_integer_larger_than_970 :
  smallest_integer_larger_than = 970 :=
begin
  sorry,
end

end smallest_integer_larger_than_970_l791_791501


namespace range_of_a_l791_791754

def proposition_p (a : ℝ) : Prop :=
  (a + 6) * (a - 7) < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4 * x + a < 0

def neg_q (a : ℝ) : Prop :=
  a ≥ 4

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ neg_q a) ↔ a ∈ Set.Ioo (-6 : ℝ) (7 : ℝ) ∪ Set.Ici (4 : ℝ) :=
sorry

end range_of_a_l791_791754


namespace dima_wins_optimally_l791_791216

/-- Mitya and Dima play a game by erasing numbers from 1 to 100 on a board. They take turns, starting with Mitya, and the game ends when only two numbers remain. Mitya wins if the sum of the final two numbers is divisible by 7; otherwise, Dima wins. This theorem proves that Dima wins if both play optimally. -/
theorem dima_wins_optimally : 
  ∀ (board : Finset ℕ), 
  (∀ n, n ∈ board → n ≥ 1 ∧ n ≤ 100) → 
  Mitya_goes_first → 
  (∃ final_two : Finset ℕ, final_two.card = 2 
    ∧ (¬ (final_two.sum % 7 = 0) → (dima_wins board))) := 
by
  intros,
  sorry

end dima_wins_optimally_l791_791216


namespace tiles_needed_l791_791831

-- Definitions for the problem
def width_wall : ℕ := 36
def length_wall : ℕ := 72
def width_tile : ℕ := 3
def length_tile : ℕ := 4

-- The area of the wall
def A_wall : ℕ := width_wall * length_wall

-- The area of one tile
def A_tile : ℕ := width_tile * length_tile

-- The number of tiles needed
def number_of_tiles : ℕ := A_wall / A_tile

-- Proof statement
theorem tiles_needed : number_of_tiles = 216 := by
  sorry

end tiles_needed_l791_791831


namespace max_and_min_values_l791_791805

noncomputable def f (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_and_min_values :
  (∀ x ∈ Icc (-1 : ℝ) 2, f 1 ≥ f x) ∧
  (∀ x ∈ Icc (-1 : ℝ) 2, f x ≥ f 2) :=
by 
  sorry

end max_and_min_values_l791_791805


namespace exists_pos_integer_m_divisible_by_1988_l791_791281

noncomputable def f : ℤ → ℤ := λ x, 3 * x + 2

theorem exists_pos_integer_m_divisible_by_1988 :
  ∃ m : ℕ, f^[100] m % 1988 = 0 := 
sorry

end exists_pos_integer_m_divisible_by_1988_l791_791281


namespace total_money_divided_l791_791539

noncomputable def children_share_total (A B E : ℕ) :=
  (12 * A = 8 * B ∧ 8 * B = 6 * E ∧ A = 84) → 
  A + B + E = 378

theorem total_money_divided (A B E : ℕ) : children_share_total A B E :=
by
  intros h
  sorry

end total_money_divided_l791_791539


namespace point_on_transformed_graph_l791_791301

theorem point_on_transformed_graph 
  (f : ℝ → ℝ)
  (h1 : f 12 = 5)
  (x y : ℝ)
  (h2 : 1.5 * y = (f (3 * x) + 3) / 3)
  (point_x : x = 4)
  (point_y : y = 16 / 9) 
  : x + y = 52 / 9 :=
by
  sorry

end point_on_transformed_graph_l791_791301


namespace probability_of_odd_and_gt_10_l791_791737

def possible_values_S : Set ℕ := {1, 2, 3, 4}
def possible_values_T : Set ℕ := {3, 4, 5, 6}
def possible_values_U : Set ℕ := {1, 2, 5, 6}

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_greater_than_10 (n : ℕ) : Prop :=
  n > 10

def count_valid_outcomes : ℕ :=
  (possible_values_S.product (possible_values_T.product possible_values_U)).count (λ s, is_odd (s.1 + s.2.1 + s.2.2) ∧ is_greater_than_10 (s.1 + s.2.1 + s.2.2))

def total_outcomes : ℕ :=
  possible_values_S.card * possible_values_T.card * possible_values_U.card

def probability_valid_sum (cnt_valid : ℕ) (cnt_total : ℕ) : ℚ :=
  cnt_valid / cnt_total

theorem probability_of_odd_and_gt_10 :
  probability_valid_sum count_valid_outcomes total_outcomes = 9 / 64 :=
by sorry

end probability_of_odd_and_gt_10_l791_791737


namespace problem_inequality_l791_791312

def f (x : ℝ) : ℝ := abs (x - 1)

def A := {x : ℝ | -1 < x ∧ x < 1}

theorem problem_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) : 
  f (a * b) > f a - f b := by
  sorry

end problem_inequality_l791_791312


namespace crossing_time_l791_791845

-- Definitions based on conditions
def train_length : ℕ := 120
def time_cross_telegraph_post_train1 : ℕ := 10
def time_cross_telegraph_post_train2 : ℕ := 12

-- Calculate speeds
def speed_train1 : ℝ := train_length / time_cross_telegraph_post_train1
def speed_train2 : ℝ := train_length / time_cross_telegraph_post_train2

-- Calculate relative speed
def relative_speed : ℝ := speed_train1 + speed_train2

-- Total distance to cross each other
def total_distance : ℕ := train_length * 2

-- Calculate time to cross each other
def time_to_cross : ℝ := total_distance / relative_speed

theorem crossing_time : time_to_cross ≈ 10.91 := by
  sorry

end crossing_time_l791_791845


namespace hundredth_digit_zero_l791_791790

theorem hundredth_digit_zero : 
  (∃ I : ℤ, ((√5 + 2)^1997 - (√5 - 2)^1997 = I) ∧ (√5 - 2)^1997 < (0.1)^100) → 
  (∃ D : ℕ, D = 0) :=
by
  intro h
  have h1 := h.1
  have h2 := h.2
  sorry

end hundredth_digit_zero_l791_791790


namespace tony_rope_length_l791_791836

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l791_791836


namespace grains_in_one_tsp_l791_791471

-- Definitions based on conditions
def grains_in_one_cup : Nat := 480
def half_cup_is_8_tbsp : Nat := 8
def one_tbsp_is_3_tsp : Nat := 3

-- Theorem statement
theorem grains_in_one_tsp :
  let grains_in_half_cup := grains_in_one_cup / 2
  let grains_in_one_tbsp := grains_in_half_cup / half_cup_is_8_tbsp
  grains_in_one_tbsp / one_tbsp_is_3_tsp = 10 :=
by
  sorry

end grains_in_one_tsp_l791_791471


namespace correct_interval_for_monotonic_decrease_l791_791321

noncomputable def f (x : ℝ) : ℝ := |Real.tan (1 / 2 * x - Real.pi / 6)|

theorem correct_interval_for_monotonic_decrease :
  ∀ k : ℤ, ∃ I : Set ℝ,
    I = Set.Ioc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3) ∧
    ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x :=
sorry

end correct_interval_for_monotonic_decrease_l791_791321


namespace constant_term_in_expansion_l791_791159

theorem constant_term_in_expansion :
  let p1 := (x^4 + 2*x^2 + 7)
  let p2 := (2*x^5 + 3*x + 20)
  constant_term (p1 * p2) = 140 :=
by
  sorry

end constant_term_in_expansion_l791_791159


namespace triangle_altitudes_parallel_l791_791376

noncomputable section

open EuclideanGeometry Real

variables {A B C E M P : Point}
variables [triangle A B C]
variables [is_altitude A E B C]
variables [is_altitude B M A C]
variables [is_altitude C P A B]
variables [parallel E M (A, B)]
variables [parallel E P (A, C)]

theorem triangle_altitudes_parallel (h1: triangle A B C)
                                     (h2: is_altitude A E B C)
                                     (h3: is_altitude B M A C)
                                     (h4: is_altitude C P A B)
                                     (h5: parallel E M (A, B))
                                     (h6: parallel E P (A, C)) : 
  parallel M P (B, C) :=
sorry

end triangle_altitudes_parallel_l791_791376


namespace cube_side_length_l791_791475

theorem cube_side_length (s : ℝ) (h : 6 * s^2 = 864) : s = 12 := by
  sorry

end cube_side_length_l791_791475


namespace surjective_and_conditions_l791_791745

open Nat

def binom (n k : ℕ) : ℕ := n.choose k

noncomputable def f (a b : ℕ) : ℕ := a.bor b

theorem surjective_and_conditions (f : ℕ → ℕ → ℕ)
  (h1 : ∀ (a b : ℕ), f a b ≤ a + b)
  (h2 : ∀ (a b c : ℕ), f a (f b c) = f (f a b) c)
  (h3 : ∀ (a b : ℕ), (binom (f a b) a) % 2 = 1 ∧ (binom (f a b) b) % 2 = 1)
  (h4 : Function.Surjective f) : f = λ a b => a.bor b := by
  sorry

end surjective_and_conditions_l791_791745


namespace cosine_angle_apothem_half_diagonal_l791_791608

variable (a : ℝ)

def SK := a * Real.sqrt 3 / 2
def KM := a * Real.sqrt 2 / 2

theorem cosine_angle_apothem_half_diagonal (SK := a * Real.sqrt 3 / 2) (KM := a * Real.sqrt 2 / 2):
  Real.cos (Real.arccos (a * Real.sqrt 3 / 2) (a * Real.sqrt 2 / 2)) = Real.sqrt 6 / 6 :=
sorry

end cosine_angle_apothem_half_diagonal_l791_791608


namespace distinct_primes_eq_1980_l791_791529

theorem distinct_primes_eq_1980 (p q r A : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
    (hne1 : p ≠ q) (hne2 : q ≠ r) (hne3 : p ≠ r) 
    (h1 : 2 * p * q * r + 50 * p * q = A)
    (h2 : 7 * p * q * r + 55 * p * r = A)
    (h3 : 8 * p * q * r + 12 * q * r = A) : 
    A = 1980 := by {
  sorry
}

end distinct_primes_eq_1980_l791_791529


namespace initial_investment_l791_791515

theorem initial_investment (P r : ℝ) 
  (h1 : 600 = P * (1 + 0.02 * r)) 
  (h2 : 850 = P * (1 + 0.07 * r)) : 
  P = 500 :=
sorry

end initial_investment_l791_791515


namespace sum_of_valid_nus_l791_791503

theorem sum_of_valid_nus : (∑ ν in {ν : ℕ | (0 < ν) ∧ (Nat.lcm ν 30 = 90)}, ν) = 216 :=
by
  sorry

end sum_of_valid_nus_l791_791503


namespace angle_terminal_side_equivalence_l791_791107

theorem angle_terminal_side_equivalence (k : ℤ) : 
  ∃ m : ℤ, (600° = m * 360° + 240°) :=
by
  sorry

end angle_terminal_side_equivalence_l791_791107


namespace number_of_chairs_in_first_row_l791_791369

-- Define the number of chairs in each row
def chairs_in_second_row := 23
def chairs_in_third_row := 32
def chairs_in_fourth_row := 41
def chairs_in_fifth_row := 50
def chairs_in_sixth_row := 59

-- Define the pattern increment
def increment := 9

-- Define a function to calculate the number of chairs in a given row, given the increment pattern
def chairs_in_row (n : Nat) : Nat :=
if n = 1 then (chairs_in_second_row - increment)
else if n = 2 then chairs_in_second_row
else if n = 3 then chairs_in_third_row
else if n = 4 then chairs_in_fourth_row
else if n = 5 then chairs_in_fifth_row
else if n = 6 then chairs_in_sixth_row
else chairs_in_second_row + (n - 2) * increment

-- The theorem to prove: The number of chairs in the first row is 14
theorem number_of_chairs_in_first_row : chairs_in_row 1 = 14 :=
  by sorry

end number_of_chairs_in_first_row_l791_791369


namespace initial_cats_l791_791764

-- Define the conditions as hypotheses
variables (total_cats now : ℕ) (cats_given : ℕ)

-- State the main theorem
theorem initial_cats:
  total_cats = 31 → cats_given = 14 → (total_cats - cats_given) = 17 :=
by sorry

end initial_cats_l791_791764


namespace complex_division_real_l791_791284

theorem complex_division_real (a : ℝ) (h : ∃ x : ℝ, (1 + a * complex.I) / (2 - complex.I) = x) : a = -1 / 2 :=
sorry

end complex_division_real_l791_791284


namespace largest_five_digit_number_is_99200_l791_791849

def largest_five_digit_number_sum_20 (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧ (n.digits.sum = 20)

theorem largest_five_digit_number_is_99200 : ∃ n : ℕ, largest_five_digit_number_sum_20 n ∧ n = 99200 :=
by
  sorry

end largest_five_digit_number_is_99200_l791_791849


namespace vector_perpendicular_l791_791075

variables {α β : ℝ}

def a : ℝ × ℝ := (Real.cos α, Real.sin α)
def b : ℝ × ℝ := (Real.cos β, Real.sin β)

theorem vector_perpendicular : 
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_b := (a.1 - b.1, a.2 - b.2)
  (a_plus_b.1 * a_minus_b.1 + a_plus_b.2 * a_minus_b.2) = 0 :=
sorry

end vector_perpendicular_l791_791075


namespace area_triangle_PNT_l791_791415

variable (PQ QR : ℝ) (N : ℝ → ℝ → Prop) (T : ℝ → ℝ → Prop)
variables (PT TN : ℝ) 
variables (areaPNT : ℝ)

/-- Define that the length of PQ is 10 and QR is 12. --/
def rectanglePQRS : Prop := PQ = 10 ∧ QR = 12

/-- Define that N is the midpoint of diagonal PS. --/
def midpointN (x y : ℝ) : Prop := N x y = ((PQ / 2), (QR / 2))

/-- Define that point T is on PQ and creates similar triangles PNT and NST where TN = PT. --/
def pointT (x : ℝ) : Prop := 
  T x PQ ∧ (similar_triangles_PNT_NST (PQ / 2, QR / 2) PQ) ∧ (PT = TN)

/-- Define the area of triangle PNT based on the given conditions. --/
def calculateArea (length_diagonal radius_height : ℝ) : ℝ := 
  let half_diag := radius_height / 2 in
  0.5 * half_diag * half_diag

/-- Theorem stating that the area of triangle PNT given all the conditions is 30.5. --/
theorem area_triangle_PNT : 
  rectanglePQRS PQ QR → 
  midpointN PQ QR → 
  pointT PQ → 
  calculateArea √244 (√244 / 2) = 30.5 :=
by 
  simp [rectanglePQRS, midpointN, pointT, calculateArea]
  sorry

end area_triangle_PNT_l791_791415


namespace find_c_min_l791_791229

def y (a b x c d : ℝ) : ℝ := a * Real.cos (b * x + c) + d

theorem find_c_min (c : ℝ) :
  (y 3 5 0 c (-1) = 3 * -1 - 1) →
  c = Real.pi :=
by
  sorry

end find_c_min_l791_791229


namespace total_pages_in_book_l791_791765

theorem total_pages_in_book (x : ℕ) : 
  (x - (x / 6 + 8) - ((5 * x / 6 - 8) / 5 + 10) - ((4 * x / 6 - 18) / 4 + 12) = 72) → 
  x = 195 :=
by
  sorry

end total_pages_in_book_l791_791765


namespace total_problems_l791_791081

-- We define the conditions as provided.
variables (p t : ℕ) -- p and t are positive whole numbers
variables (p_gt_10 : 10 < p) -- p is more than 10

theorem total_problems (p t : ℕ) (p_gt_10 : 10 < p) (h : p * t = (2 * p - 4) * (t - 2)):
  p * t = 60 :=
by
  sorry

end total_problems_l791_791081


namespace probability_of_drawing_different_colors_l791_791716

-- Define the basic setup: Balls in box A and box B.
def boxA := {red := 3, black := 3, white := 3}
def boxB := {yellow := 2, black := 2, white := 2}

-- Define a function to calculate the probability of the events.
noncomputable def probability_different_colors : ℚ :=
  1 - (6 / 54)

theorem probability_of_drawing_different_colors :
  probability_different_colors = 7 / 9 :=
by sorry

end probability_of_drawing_different_colors_l791_791716


namespace alyona_final_balances_l791_791566

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end alyona_final_balances_l791_791566


namespace interval_solution_l791_791257

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l791_791257


namespace max_employees_le_l791_791778

theorem max_employees_le (x : Nat) : ∃ n : Nat, n <= 2 * (Nat.floor (x / (2 * x - 8))) := sorry

end max_employees_le_l791_791778


namespace real_solutions_l791_791604

theorem real_solutions:
  ∀ x: ℝ, 
    (x ≠ 2) ∧ (x ≠ 4) ∧ 
    ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1)) / 
    ((x - 2) * (x - 4) * (x - 2)) = 1 
    → (x = 2 + Real.sqrt 2) ∨ (x = 2 - Real.sqrt 2) :=
by
  sorry

end real_solutions_l791_791604


namespace investment_time_P_l791_791814

-- Variables and conditions
variables {x : ℕ} {time_P : ℕ}

-- Conditions as seen from the mathematical problem
def investment_P (x : ℕ) := 7 * x
def investment_Q (x : ℕ) := 5 * x
def profit_ratio := 1 / 2
def time_Q := 14

-- Statement of the problem
theorem investment_time_P : 
  (profit_ratio = (investment_P x * time_P) / (investment_Q x * time_Q)) → 
  time_P = 5 := 
sorry

end investment_time_P_l791_791814


namespace solve_log_equation_l791_791171

theorem solve_log_equation (a x : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_x_pos : x > 0) (h_x_ne_one : x ≠ 1) :
  (3 * log a x - 2) * (log x a)^2 = log (ℝ.sqrt a) x - 3 ↔ 
  x = 1 / a ∨ x = Real.sqrt a ∨ x = a^2 := 
sorry

end solve_log_equation_l791_791171


namespace ellipse_correct_l791_791647

noncomputable def ellipse_problem := 
  ∃ a b c : ℝ, 
  (a > b) ∧ (b > 0) ∧ 
  (2 * a = 4) ∧ 
  (c / a = real.sqrt 3 / 2) ∧ 
  (a ^ 2 = b ^ 2 + c ^ 2) ∧ 
  (∃ (x y : ℝ), ((x^2 / 4 + y^2 = 1) ∧ 
  (∃ (k m : ℝ), (4 * k ^ 2 - m ^ 2 + 1 > 0) ∧ 
  (5 * m^2 + 16 * k * m + 12 * k^2 = 0) ∧ 
  ((m = -2 * k ∨ m = -6/5 * k) → 
  (m = -6/5 * k) ∧ 
  (2,0) ≠ (6/5,0)))))

theorem ellipse_correct : ellipse_problem := 
  sorry

end ellipse_correct_l791_791647


namespace problem1_problem2_problem3_problem4_l791_791533

-- Problem 1
theorem problem1 (x : ℝ) (hx : -1 ≤ x) : 
  let y := 12 * Real.sqrt (x + 1) / (3 * x + 4)
  in 0 ≤ y ∧ y ≤ 2 * Real.sqrt 3 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : -1 < x) : 
  let y := (x^2 + 3 * x + 3) / (x + 1)
  in ∃ x, y = 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (hx : 0 < x) : 
  let y := (x^2 + 1)^2 / ((3 * x^2 + 2) * (2 * x^2 + 3))
  in ∃ x, y = 4 / 25 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (hx : 0 < x) (hx : x < Real.pi / 2) :
  let y := Real.sin x * Real.cos x ^ 2
  in ∃ x, y = 2 * Real.sqrt 3 / 9 :=
sorry

end problem1_problem2_problem3_problem4_l791_791533


namespace probability_neg1_to_0_l791_791756

noncomputable def standardNormalDistribution (ξ : ℝ) : Prop :=
  ∃ (μ σ : ℝ), μ = 0 ∧ σ = 1 ∧ ξ ∼ gaussian μ σ

theorem probability_neg1_to_0 {ξ : ℝ} (p : ℝ) (h1 : standardNormalDistribution ξ) (h2 : (gaussian cdf 1) = p) : 
  (gaussian cdf 0) - (gaussian cdf (-1)) = 1/2 - p :=
sorry

end probability_neg1_to_0_l791_791756


namespace convex_100gon_distinct_numbers_l791_791362

theorem convex_100gon_distinct_numbers :
  ∀ (vertices : Fin 100 → (ℕ × ℕ)),
  (∀ i, (vertices i).1 ≠ (vertices i).2) →
  ∃ (erase_one_number : ∀ (i : Fin 100), ℕ),
  (∀ i, erase_one_number i = (vertices i).1 ∨ erase_one_number i = (vertices i).2) ∧
  (∀ i j, i ≠ j → (i = j + 1 ∨ (i = 0 ∧ j = 99)) → erase_one_number i ≠ erase_one_number j) :=
by sorry

end convex_100gon_distinct_numbers_l791_791362


namespace value_ranges_of_a_l791_791784

-- Definitions for the given conditions
def quadratic_1 (a : ℝ) (x : ℝ) := 4 * a * x^2 - 16 * x + 4 * a
def quadratic_2 (a : ℝ) (x : ℝ) := 4 * a * x^2 - 36 * x + 9 * a

-- Derivative of a = 4x / (x^2 + 1)
def derivative_1 (x : ℝ) : ℝ := 4 * (1 - x^2) / (x^2 + 1)^2

-- Derivative of a = 36x / (4x^2 + 9)
def derivative_2 (x : ℝ) : ℝ := 36 * (9 - 4 * x^2) / (4 * x^2 + 9)^2

-- Proof statement
theorem value_ranges_of_a (a : ℝ) : (a ∈ set.Icc (8 / 5) 2 ∨ a ∈ set.Icc (36 / 13) 3) :=
sorry

end value_ranges_of_a_l791_791784


namespace sum_of_angles_l791_791730

theorem sum_of_angles : 
    ∀ (angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC : ℝ),
    angle1 + angle3 + angle5 = 180 ∧
    angle2 + angle4 + angle6 = 180 ∧
    angleA + angleB + angleC = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 :=
by
  intro angle1 angle3 angle5 angle2 angle4 angle6 angleA angleB angleC
  intro h
  sorry

end sum_of_angles_l791_791730


namespace inequality_proof_l791_791631

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 
    (3 / (Real.cbrt (a * b * c) * (1 + Real.cbrt (a * b * c)))) := 
sorry

end inequality_proof_l791_791631


namespace ratio_of_area_l791_791841

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def area (triangle : Triangle) : ℝ := sorry

variables {A B C M P Q D E N : Point}

-- Conditions
def M_is_midpoint_AB (A B M : Point) : Prop :=
  midpoint A B = M

def points_on_AB (A B P Q M : Point) : Prop :=
  A < P ∧ P < Q ∧ Q < M

def MD_parallel_PC (M D P C : Point) : Prop :=
  line M D ∥ line P C

def NE_parallel_QC (N E Q C : Point) : Prop :=
  line N E ∥ line Q C

-- Proof problem
theorem ratio_of_area {A B C M P Q D E N : Point}
  (h1 : M_is_midpoint_AB A B M)
  (h2 : points_on_AB A B P Q M)
  (h3 : MD_parallel_PC M D P C)
  (h4 : NE_parallel_QC N E Q C)
  : area (Triangle.mk B N E) / area (Triangle.mk A B C) = 1 / 4 :=
sorry

end ratio_of_area_l791_791841


namespace effective_annual_rate_of_interest_l791_791121

theorem effective_annual_rate_of_interest 
  (i : ℝ) (n : ℕ) (h_i : i = 0.10) (h_n : n = 2) : 
  (1 + i / n)^n - 1 = 0.1025 :=
by
  sorry

end effective_annual_rate_of_interest_l791_791121


namespace trapezoid_isosceles_of_perpendicular_diagonals_l791_791432

theorem trapezoid_isosceles_of_perpendicular_diagonals
(K L M N : Type)
(trapezoid : Type)
[trapezoid K L M N]
(angle_bisectors_intersect_to_form_quadrilateral_with_perpendicular_diagonals : 
(∃ quadrilateral : Type, quadrilateral K L M N ∧ 
  ∀ P Q R S : quadrilateral, 
    (angle_bisectors : ∀ (bisector : Type), bisector intersects ∧ bisector.quadrilateral = 90°) ∧
    (P, Q, R, S form_diagonal P + diagonal Q = P + diagonal S))) :
(trapezoid.is_isosceles) :=
begin
  sorry
end

end trapezoid_isosceles_of_perpendicular_diagonals_l791_791432


namespace min_m_n_l791_791788

theorem min_m_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 108 * m = n^3) : m + n = 8 :=
sorry

end min_m_n_l791_791788


namespace part_i_part_ii_l791_791180

open Real -- Open the Real number space

-- (i) Prove that for any real number x, there exist two points of the same color that are at a distance of x from each other
theorem part_i (color : Real × Real → Bool) :
  ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

-- (ii) Prove that there exists a color such that for every real number x, 
-- we can find two points of that color that are at a distance of x from each other
theorem part_ii (color : Real × Real → Bool) :
  ∃ c : Bool, ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = c ∧ color p2 = c ∧ dist p1 p2 = x :=
by
  sorry

end part_i_part_ii_l791_791180


namespace lemonade_lemon_juice_parts_l791_791149

theorem lemonade_lemon_juice_parts (parts_water parts_lemon water_ratio lemon_ratio total_volume_gallons quarts_per_gallon : ℕ) 
(h_ratio : water_ratio = 3) 
(h_lemon_ratio : lemon_ratio = 2) 
(h_parts_water : parts_water = 3) 
(h_parts_lemon : parts_lemon = 2) 
(h_total_volume_gallons : total_volume_gallons = 1.5) 
(h_quarts_per_gallon : quarts_per_gallon = 4) :
let total_volume_quarts := total_volume_gallons * quarts_per_gallon in 
let total_parts := parts_water + parts_lemon in 
let quarts_per_part := total_volume_quarts / total_parts in 
let lemon_juice_quarts := parts_lemon * quarts_per_part in 
lemon_juice_quarts = 2.4 := 
by { sorry }

end lemonade_lemon_juice_parts_l791_791149


namespace find_principal_amount_l791_791165

noncomputable def principal_amount : ℝ :=
  let R := 10
  let T := 4
  let difference := 64.10
  let SI (P : ℝ) := P * R * T / 100
  let CI (P : ℝ) := P * (1 + R / 100)^T - P
  let proof (P : ℝ) := CI P - SI P = difference
  classical.some (exists P, proof P)

theorem find_principal_amount : principal_amount = 1000 := sorry

end find_principal_amount_l791_791165


namespace verify_inclination_angles_l791_791667

noncomputable def inclination_angle_of_focal_chord (p : ℝ) (θ : ℝ) : Prop :=
  (sqrt(2 * p) * sqrt(2 * p) * sin(θ) * cos(θ) = 8 * p * (sin(θ))/(sin(θ))) → θ = π / 6 ∨ θ = 5 * π / 6

theorem verify_inclination_angles (p : ℝ) (hp : 0 < p):
  inclination_angle_of_focal_chord p (π / 6) ∨ inclination_angle_of_focal_chord p (5 * π / 6) :=
sorry

end verify_inclination_angles_l791_791667


namespace predicted_yield_of_rice_l791_791699

theorem predicted_yield_of_rice (x : ℝ) (h : x = 80) : 5 * x + 250 = 650 :=
by {
  sorry -- proof will be given later
}

end predicted_yield_of_rice_l791_791699


namespace stratified_sampling_35_to_50_years_l791_791019

theorem stratified_sampling_35_to_50_years (total_age_20_35 : ℕ) (total_age_35_50 : ℕ) (total_age_50_60 : ℕ) (survey_total : ℕ) : 
  total_age_20_35 = 180 → total_age_35_50 = 108 → total_age_50_60 = 72 → survey_total = 20 →
  let total_population := total_age_20_35 + total_age_35_50 + total_age_50_60 in
  let expected_age_35_50 := survey_total * total_age_35_50 / total_population in
  expected_age_35_50 = 6 :=
by
  intros
  let total_population := total_age_20_35 + total_age_35_50 + total_age_50_60
  let expected_age_35_50 := survey_total * total_age_35_50 / total_population
  have h : total_population = 360 := by simp [*]
  have h_expected : expected_age_35_50 = 6 := by rw [h]; simp
  exact h_expected

end stratified_sampling_35_to_50_years_l791_791019


namespace residue_of_7_pow_2023_mod_19_l791_791497

theorem residue_of_7_pow_2023_mod_19 : (7^2023) % 19 = 3 :=
by 
  -- The main goal is to construct the proof that matches our explanation.
  sorry

end residue_of_7_pow_2023_mod_19_l791_791497


namespace crystal_running_distance_l791_791585

-- Define conditions
def north_distance : ℝ := 1
def east_distance : ℝ := 2
def south_distance : ℝ := 1
def net_north_south_distance : ℝ := north_distance - south_distance
def net_east_west_distance : ℝ := east_distance 

-- Prove the final distance back to the start
theorem crystal_running_distance : 
  ∀ (d1 d2 d3 d4 : ℝ), 
  d1 = north_distance → 
  d2 = east_distance → 
  d3 = south_distance → 
  d4 = √(net_east_west_distance^2 + net_north_south_distance^2) → 
  d4 = 2 :=
by
  intro d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end crystal_running_distance_l791_791585


namespace num_convex_numbers_without_repeats_l791_791660

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ b > c

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n < 10

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem num_convex_numbers_without_repeats : 
  (∃ (numbers : Finset (ℕ × ℕ × ℕ)), 
    (∀ a b c, (a, b, c) ∈ numbers -> is_convex_number a b c ∧ is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c) ∧
    numbers.card = 204) :=
sorry

end num_convex_numbers_without_repeats_l791_791660


namespace average_marks_correct_l791_791521

-- Definitions used in the Lean 4 statement, reflecting conditions in the problem
def total_students_class1 : ℕ := 25 
def average_marks_class1 : ℕ := 40 
def total_students_class2 : ℕ := 30 
def average_marks_class2 : ℕ := 60 

-- Calculate the total marks for both classes
def total_marks_class1 : ℕ := total_students_class1 * average_marks_class1 
def total_marks_class2 : ℕ := total_students_class2 * average_marks_class2 
def total_marks : ℕ := total_marks_class1 + total_marks_class2 

-- Calculate the total number of students
def total_students : ℕ := total_students_class1 + total_students_class2 

-- Define the average of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_students 

-- The theorem to be proved
theorem average_marks_correct : average_marks_all_students = (2800 : ℚ) / 55 := 
by 
  sorry

end average_marks_correct_l791_791521


namespace solution_set_l791_791545

theorem solution_set (f : ℝ → ℝ) (hf : ∀ x, f(x) + deriv f x < 1) (h0 : f(0) = 3) :
  {x : ℝ | f(x) > 2 / real.exp(x) + 1} = set.Iio 0 :=
by
  sorry

end solution_set_l791_791545


namespace fraction_diff_equals_7_over_12_l791_791848

noncomputable def fraction_diff : ℚ :=
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6)

theorem fraction_diff_equals_7_over_12 : fraction_diff = 7 / 12 := by
  sorry

end fraction_diff_equals_7_over_12_l791_791848


namespace final_dollars_final_euros_final_rubles_l791_791569

-- Defining constants and initial amounts
def initial_euros : ℝ := 3000
def initial_dollars : ℝ := 4000
def initial_rubles : ℝ := 240000

def interest_rate_eur : ℝ := 0.021
def interest_rate_usd : ℝ := 0.021
def interest_rate_rub : ℝ := 0.079

def sell_rate_eur_to_rub : ℝ := 60.10
def buy_rate_rub_to_usd : ℝ := 58.90
def sell_rate_usd_to_rub : ℝ := 58.50
def buy_rate_rub_to_eur : ℝ := 61.20

-- Function to compound interest yearly
def compound_interest (principal rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

-- After year 1
def year1_euros : ℝ := compound_interest initial_euros interest_rate_eur 1
def year1_dollars : ℝ := compound_interest initial_dollars interest_rate_usd 1
def year1_rubles : ℝ := compound_interest initial_rubles interest_rate_rub 1

-- Conversion after year 1
def euros_sold : ℝ := 1000
def rubles_from_euros : ℝ := euros_sold * sell_rate_eur_to_rub
def dollars_purchased : ℝ := rubles_from_euros / buy_rate_rub_to_usd

def year2_euros : ℝ :=
  compound_interest (year1_euros - euros_sold) interest_rate_eur 1
def year2_dollars : ℝ :=
  compound_interest (year1_dollars + dollars_purchased) interest_rate_usd 1
def year2_rubles : ℝ :=
  compound_interest year1_rubles interest_rate_rub 1

-- Conversion after year 2
def dollars_sold : ℝ := 2000
def rubles_from_dollars : ℝ := dollars_sold * sell_rate_usd_to_rub
def euros_purchased : ℝ := rubles_from_dollars / buy_rate_rub_to_eur

def year3_euros : ℝ :=
  compound_interest (year2_euros + euros_purchased) interest_rate_eur 1
def year3_dollars : ℝ :=
  compound_interest (year2_dollars - dollars_sold) interest_rate_usd 1
def year3_rubles : ℝ :=
  compound_interest year2_rubles interest_rate_rub 1

-- Statements to prove
theorem final_dollars : round year3_dollars = 3286 := by sorry
theorem final_euros : round year3_euros = 4040 := by sorry
theorem final_rubles : round year3_rubles = 301504 := by sorry

end final_dollars_final_euros_final_rubles_l791_791569


namespace min_bailing_rate_l791_791100

noncomputable def slowest_bailing_rate (distance : ℝ) (rowing_speed : ℝ) (leak_rate : ℝ) (max_capacity : ℝ) : ℝ :=
  let time_to_shore := distance / rowing_speed
  let time_to_shore_in_minutes := time_to_shore * 60
  let total_water_intake := leak_rate * time_to_shore_in_minutes
  let excess_water := total_water_intake - max_capacity
  excess_water / time_to_shore_in_minutes

theorem min_bailing_rate : slowest_bailing_rate 3 3 14 40 = 13.3 :=
by
  sorry

end min_bailing_rate_l791_791100


namespace cyclic_quadrilateral_B1KLM_l791_791873

theorem cyclic_quadrilateral_B1KLM
  {A B C A₁ B₁ C₁ K M L : Type*}
  (h_acute : ∀ (X Y Z : Type*), is_acute_angle_triangle X Y Z)
  (h_altitudes : is_altitude A A₁ B B₁ C C₁)
  (h_angle_K : ∠ B B₁ K = ∠ B A C)
  (h_angle_M : ∠ B B₁ M = ∠ A C B)
  (h_intersection_L : is_intersection_point_of_altitude_and_segment L B₁ A₁ C₁) :
  is_cyclic_quadrilateral B₁ K L M :=
sorry

end cyclic_quadrilateral_B1KLM_l791_791873


namespace scientific_notation_correct_l791_791115

-- Define the conditions
def circumference_circular_orbit_chinese_space_station := 43000000

-- Define the target scientific notation representation
def scientific_notation_representation := 4.3 * 10 ^ 7

-- State the problem: Prove that the given number in scientific notation is equal to the circumference
theorem scientific_notation_correct : 
  circumference_circular_orbit_chinese_space_station = scientific_notation_representation :=
by
  sorry

end scientific_notation_correct_l791_791115


namespace set_intersection_l791_791074

-- defining universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- defining set A
def A : Set ℕ := {1, 5, 9}

-- defining set B
def B : Set ℕ := {3, 7, 9}

-- complement of A in U
def complU (s : Set ℕ) := {x ∈ U | x ∉ s}

-- defining the intersection of complement of A with B
def intersection := complU A ∩ B

-- statement to be proved
theorem set_intersection : intersection = {3, 7} :=
by
  sorry

end set_intersection_l791_791074


namespace intersecting_lines_k_value_l791_791963

theorem intersecting_lines_k_value :
  ∃ k : ℚ, (∀ x y : ℚ, y = 3 * x + 12 ∧ y = -5 * x - 7 → y = 2 * x + k) → k = 77 / 8 :=
sorry

end intersecting_lines_k_value_l791_791963


namespace problem_solution_l791_791637

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = a 1 * (a 1 / a 0)^(n - 1)

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
∀ n, b n / n = (b 1 / 1) + (n - 1)

def initial_conditions (a b : ℕ → ℝ) : Prop :=
a 1 = 2 * b 1 ∧
a 4 - a 2 = 12 ∧
(∑ i in finset.range 4, a i) + 2 * (∑ i in finset.range 2, a i) = 3 * (∑ i in finset.range 3, a i)

def c_n (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
if n % 2 = 0 then 
  2 / a n 
else 
  n / (b n * (n + 2))

def T_n (c : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, c i

theorem problem_solution (a b : ℕ → ℝ) (n : ℕ) (c : ℕ → ℝ) : 
  is_geometric_sequence a ∧ is_arithmetic_sequence b ∧ initial_conditions a b → 
  (∀ n, a n = 2 ^ n) ∧ 
  (∀ n, b n = n * (n + 1)) ∧ 
  (∀ n, T_n c (2 * n) = (7 / 6) - (1 / (2 * (2 * n + 1))) - (2 / (3 * 4 ^ n))) := 
sorry

end problem_solution_l791_791637


namespace find_bottom_right_corner_number_l791_791812

def is_filled_correctly (grid : ℕ → ℕ → ℕ) :=
  ∀ (i j : ℕ), i < 24 → j < 24 → grid i j = 24 * i + j + 1

def grid_property (grid : ℕ → ℕ → ℕ) (x : ℕ) :=
  let top_left := x - 175
  let top_right := x - 168
  let bottom_left := x - 7
  let bottom_right := x
  (top_left + top_right + bottom_left + bottom_right = 1646)

theorem find_bottom_right_corner_number :
  ∃ (x : ℕ),
    (∀ (grid : ℕ → ℕ → ℕ),
      is_filled_correctly grid →
      grid_property grid x) ∧
    x = 499 :=
begin
  existsi 499,
  sorry
end

end find_bottom_right_corner_number_l791_791812


namespace samson_mother_age_relation_l791_791095

theorem samson_mother_age_relation 
  (S : ℝ) (M : ℝ) (hS : S = 6.25) (hM : M = 30.65) :
  ∃ S' : ℝ, (M + (S' - S) = 4 * S') ∧ S' ≈ 8.1333 :=
by
  sorry

end samson_mother_age_relation_l791_791095


namespace volunteer_assignment_schemes_l791_791964

theorem volunteer_assignment_schemes:
  let students := ['A', 'B', 'C', 'D']
  let halls := ['A', 'B', 'C']
  let student_A := 'A'
  let hall_A := 'A'
  -- Given that student A is not assigned to hall A
  let valid_assignments := {s | ∀ h ∈ halls, h ≠ hall_A → ∃ x ∈ students, s x = h}
  -- The number of valid assignments of students to halls
  set.card valid_assignments = 24 :=
sorry

end volunteer_assignment_schemes_l791_791964


namespace allocate_square_plots_l791_791506

theorem allocate_square_plots (x y : ℤ) (h : x > y) :
  ∃ u v : ℤ, u^2 + v^2 = 2 * (x^2 + y^2) :=
by {
  use [x + y, x - y],
  -- sorry can be used to skip the actual detailed proof which is not required here.
  sorry
}

end allocate_square_plots_l791_791506


namespace original_number_is_80_l791_791208

-- Define the existence of the numbers A and B
variable (A B : ℕ)

-- Define the conditions from the problem
def conditions :=
  A = 35 ∧ A / 7 = B / 9

-- Define the statement to prove
theorem original_number_is_80 (h : conditions A B) : A + B = 80 :=
by
  -- Proof is omitted
  sorry

end original_number_is_80_l791_791208


namespace area_expression_l791_791452

-- Definitions based on the problem statements
def radius := 5
def angle := 90
def side_length := 10
def circle_area := 25 * Real.pi
def square_area := 100
def area_difference := square_area - circle_area
def region_area := area_difference / 4

-- The actual statement to prove
theorem area_expression : ∃ a b c : ℝ, a = 25 ∧ b = 1 ∧ c = -25 / 4 ∧ a + b + c = 19.75 :=
by {
    use [25, 1, (-25 / 4)],
    split,
    { refl, },
    split,
    { refl, },
    split,
    { refl, },
    norm_num,
}

end area_expression_l791_791452


namespace train_speed_is_correct_l791_791563

-- Define the conditions as functions or constants.
def length_of_train : ℝ := 640  -- length in meters
def time_to_cross_post : ℝ := 16  -- time in seconds

-- Conversion factors.
def meters_to_kilometers : ℝ := 1 / 1000
def seconds_to_hours : ℝ := 1 / 3600

-- Define the expected speed in km/h based on the conditions.
def expected_speed_in_kmh (length_of_train : ℝ) (time_in_seconds : ℝ) : ℝ :=
  (length_of_train * meters_to_kilometers) / (time_in_seconds * seconds_to_hours)

-- The statement to be proved.
theorem train_speed_is_correct :
  expected_speed_in_kmh length_of_train time_to_cross_post = 144 :=
by
  sorry

end train_speed_is_correct_l791_791563


namespace votes_distribution_l791_791516

theorem votes_distribution (W : ℕ) 
  (h1 : W + (W - 53) + (W - 79) + (W - 105) = 963) 
  : W = 300 ∧ 247 = W - 53 ∧ 221 = W - 79 ∧ 195 = W - 105 :=
by
  sorry

end votes_distribution_l791_791516


namespace point_lies_on_translated_line_l791_791840

theorem point_lies_on_translated_line :
  let original_function := λ x, x
  let translated_function := λ x, x + 2
  (2, 4) ∈ set_of (λ p : ℝ × ℝ, p.snd = translated_function p.fst) :=
by
  let original_function := λ x, x
  let translated_function := λ x, x + 2
  sorry

end point_lies_on_translated_line_l791_791840


namespace sum_arithmetic_sequence_l791_791390

noncomputable def is_arithmetic (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∃ a1 : ℚ, ∀ n : ℕ, a n = a1 + n * d

noncomputable def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 0 + a (n - 1)) / 2

theorem sum_arithmetic_sequence (a : ℕ → ℚ) (h_arith : is_arithmetic a)
  (h1 : 2 * a 3 = 5) (h2 : a 4 + a 12 = 9) : sum_of_first_n_terms a 10 = 35 :=
by
  -- Proof omitted
  sorry

end sum_arithmetic_sequence_l791_791390


namespace hyperbola_equation_range_of_expression_l791_791992

-- Define the hyperbola and its parameters
def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the foci of the hyperbola
def foci_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

-- Given conditions
def conditions (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ hyperbola_eq a b x y

-- Define dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the vectors from point G to foci F1 and F2
def vectors_from_G_to_foci (c : ℝ) (G : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-c - G.1, -G.2), (c - G.1, -G.2))

-- Problem (1): Prove the equation of the hyperbola
theorem hyperbola_equation (a b c : ℝ) (G : ℝ × ℝ) :
  conditions a b (G.1) (G.2) ∧
  dot_product (vectors_from_G_to_foci c G).1 (vectors_from_G_to_foci c G).2 = 0 →
  a = 1 ∧ b = real.sqrt 2 ∧ hyperbola_eq 1 (real.sqrt 2) (G.1) (G.2) :=
by
  sorry

-- Problem (2): Prove the range of the expression
-- Note: Here, we encapsulate the mathematical statement without proving.
theorem range_of_expression (k : ℝ) :
  0 < k^2 ∧ k^2 < 2 →
  0 < (real.sqrt(3 - k^2) - 1) ∧ (real.sqrt(3 - k^2) - 1) ≤ real.sqrt(3) - 1 :=
by
  sorry

end hyperbola_equation_range_of_expression_l791_791992


namespace knight_even_moves_l791_791200

-- Define the statement of the problem
theorem knight_even_moves
    (KnightStart : ℕ × ℕ := (1, 1))
    (path : List (ℕ × ℕ))
    (condition : (path.head? = some KnightStart) ∧ (path.last? = some KnightStart)) :
    (path.length % 2 = 0) :=
by
  -- Sorry keyword to skip the proof
  sorry

end knight_even_moves_l791_791200


namespace hyperbola_eccentricity_l791_791583

variable (a b : ℝ) (P Q F1 F2 : ℝ × ℝ)

-- The conditions from the problem
def is_hyperbola (y x : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (y^2 / a^2 - x^2 / b^2 = 1)

def is_focus_positioned (F1 F2 : ℝ × ℝ) : Prop :=
  F1.snd > 0 ∧ F2.snd < 0

def line_intersects (F1 P Q : ℝ × ℝ) : Prop :=
  (P - F1).x * (Q - F1).x + (P - F1).y * (Q - F1).y = 0 ∧ (dist Q F1) - (dist P F1) = 2 * a

def orthogonality (PF1 PF2 : ℝ × ℝ) : Prop :=
  PF1.fst * PF2.fst + PF1.snd * PF2.snd = 0

theorem hyperbola_eccentricity :
  is_hyperbola a b Q.fst Q.snd ∧
  is_focus_positioned F1 F2 ∧
  line_intersects F1 P Q ∧
  orthogonality (P - F1) (P - F2) → 
  ∃ e : ℝ, e = (Real.sqrt 10 / 2) :=
by
  sorry

end hyperbola_eccentricity_l791_791583


namespace sum_of_x_intercepts_eq_4_l791_791616

noncomputable theory

open Real

def f (x : ℝ) : ℝ := log (abs (x - 2)) + x^2
def g (x : ℝ) : ℝ := 4 * x

theorem sum_of_x_intercepts_eq_4 :
  let x_intersects := { x : ℝ | f x = g x } in
  ∃ x1 x2 ∈ x_intersects, x1 ≠ x2 ∧ x1 + x2 = 4 :=
by {
  sorry
}

end sum_of_x_intercepts_eq_4_l791_791616


namespace max_volume_of_prism_l791_791712

theorem max_volume_of_prism (a b h : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : h > 0)
  (sum_of_areas : a * h + b * h + a * b = 40) :
  ∃ V_max : ℝ, V_max = (80 * Real.sqrt 30) / 9 :=
begin
  use (80 * Real.sqrt 30) / 9,
  sorry,
end

end max_volume_of_prism_l791_791712


namespace toys_produced_per_week_l791_791891

theorem toys_produced_per_week (daily_production : ℕ) (work_days_per_week : ℕ) (total_production : ℕ) :
  daily_production = 680 ∧ work_days_per_week = 5 → total_production = 3400 := by
  sorry

end toys_produced_per_week_l791_791891


namespace find_triples_l791_791243

theorem find_triples (a b c : ℝ) :
  a^2 + b^2 + c^2 = 1 ∧ a * (2 * b - 2 * a - c) ≥ 1/2 ↔ 
  (a = 1 / Real.sqrt 6 ∧ b = 2 / Real.sqrt 6 ∧ c = -1 / Real.sqrt 6) ∨
  (a = -1 / Real.sqrt 6 ∧ b = -2 / Real.sqrt 6 ∧ c = 1 / Real.sqrt 6) := 
by 
  sorry

end find_triples_l791_791243


namespace circle_eqn_max_value_on_circle_l791_791882

-- Definitions from conditions
def center_of_circle : Point := ⟨-1, 0⟩
def tangent_line (p: Point) : Prop := p.x + p.y + 3 = 0
def circle_center_eq_line : Prop := ∃ p: Point, p.x - p.y + 1 = 0 ∧ p.y = 0

-- Problem 1
theorem circle_eqn (center_eq_tangent : circle_center_eq_line ∧ tangent_line center_of_circle) : 
  ∃ r : ℝ, r^2 = 2 ∧ ∀ p : Point, (p.x + 1)^2 + p.y^2 = r^2 := by
  sorry

-- Problem 2
theorem max_value_on_circle (P_on_circle : ∀ (P: Point), P.x^2 + P.y^2 - 4 * P.y + 3 = 0) : 
  ∃ k : ℝ, k = (√3) / 3 ∧ ∀ x y : ℝ, x ≠ 0 → y = k * x → (√3)/3 <= k := by
  sorry

end circle_eqn_max_value_on_circle_l791_791882


namespace range_of_a_and_m_l791_791289

open Set

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 1 = 0}

-- Conditions as hypotheses
def condition1 : A ∪ B a = A := sorry
def condition2 : A ∩ C m = C m := sorry

-- Theorem to prove the correct range of a and m
theorem range_of_a_and_m : (a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2) :=
by
  -- Proof goes here
  sorry

end range_of_a_and_m_l791_791289


namespace number_of_people_in_group_l791_791172

theorem number_of_people_in_group (P : ℕ) : 
  (∃ (P : ℕ), 0 < P ∧ (364 / P - 1 = 364 / (P + 2))) → P = 26 :=
by
  sorry

end number_of_people_in_group_l791_791172


namespace carol_betty_age_ratio_l791_791937

theorem carol_betty_age_ratio:
  ∀ (C A B : ℕ), 
    C = 5 * A → 
    A = C - 12 → 
    B = 6 → 
    C / B = 5 / 2 :=
by
  intros C A B h1 h2 h3
  sorry

end carol_betty_age_ratio_l791_791937


namespace maxDistanceBetweenVertex_l791_791903

noncomputable def maxDistance (
  (S_rect: ℝ × ℝ)
  (S: ℝ)
) : ℝ :=
  let diagonal_rect := Math.sqrt (S_rect.1 ^ 2 + S_rect.2 ^ 2)
  let diagonal_square := 9 * Math.sqrt 2
  (diagonal_square + diagonal_rect) / 2

theorem maxDistanceBetweenVertex (
  (S_perimeter : ℝ := 36)
  (R_perimeter : ℝ := 26)
  (R_length : ℝ := 5)
  (R_width : ℝ := 8)
  (S_side : ℝ := S_perimeter / 4)
) : 
  maxDistance ((R_length, R_width)) S_side = (9 * Math.sqrt 2 + Math.sqrt 89) / 2 :=
by
  sorry

end maxDistanceBetweenVertex_l791_791903


namespace expand_product_l791_791974

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by sorry

end expand_product_l791_791974


namespace number_of_friends_l791_791215

-- Definitions corresponding to the given conditions
def meters_per_dress := 5
def total_silk := 600
def dresses_made_by_alex := 100
def silk_given_to_friends := total_silk - (dresses_made_by_alex * meters_per_dress)
def silk_per_friend := meters_per_dress

-- The theorem to prove the number of friends Alex gave silk to
theorem number_of_friends (h: silk_given_to_friends = 100): 
  silk_given_to_friends / silk_per_friend = 20 := 
by 
  rw [silk_given_to_friends, silk_per_friend]
  norm_num
  exact h

#check number_of_friends

end number_of_friends_l791_791215


namespace wendy_longest_crafting_pattern_l791_791157

/-- 
  Define the initial lengths purchased by Wendy and the lengths used for the project.
  Calculate the remaining lengths for each ribbon and show that the longest uniform 
  length of the crafting pattern Wendy can create is 24 inches.
--/
def wendy_initial_ribbon_lengths : List ℕ := [84, 96, 112, 54, 120]
def wendy_used_ribbon_lengths : List ℕ := [46, 58, 72, 30, 90]
def wendy_remaining_ribbon_lengths : List ℕ :=
  List.zipWith (λ x y => x - y) wendy_initial_ribbon_lengths wendy_used_ribbon_lengths

theorem wendy_longest_crafting_pattern :
  (wendy_remaining_ribbon_lengths = [38, 38, 40, 24, 30]) →
  List.minimum wendy_remaining_ribbon_lengths = some 24 :=
by
  intros h
  rw h
  apply rfl

end wendy_longest_crafting_pattern_l791_791157


namespace record_loss_of_10_l791_791001

-- Definition of profit and loss recording
def record (x : Int) : Int :=
  if x ≥ 0 then x else -x

-- Condition: A profit of $20 should be recorded as +$20
axiom profit_recording : ∀ (p : Int), p ≥ 0 → record p = p

-- Condition: A loss should be recorded as a negative amount
axiom loss_recording : ∀ (l : Int), l < 0 → record l = l

-- Question: How should a loss of $10 be recorded?
-- Prove that if a small store lost $10, it should be recorded as -$10
theorem record_loss_of_10 : record (-10) = -10 :=
by sorry

end record_loss_of_10_l791_791001


namespace all_solutions_l791_791587

noncomputable def satisfies_functional_eq (f : ℚ → ℤ) : Prop :=
∀ (x : ℚ) (a : ℤ) (b : ℤ) (hb : b > 0), 
  f ((f x + a) / b) = f ((x + a) / b)

theorem all_solutions (f : ℚ → ℤ) :
  satisfies_functional_eq f →
  (∃ C : ℤ, ∀ x : ℚ, f x = C) ∨
  (∀ x : ℚ, f x = Int.floor x) ∨
  (∀ x : ℚ, f x = Int.ceil x) :=
begin
  intros h,
  sorry,  -- Proof to be provided
end

end all_solutions_l791_791587


namespace michael_total_time_l791_791079

def time_for_200_meters (distance speed : ℕ) : ℚ :=
  distance / speed

def total_time_per_lap : ℚ :=
  (time_for_200_meters 200 6) + (time_for_200_meters 200 3)

def total_time_8_laps : ℚ :=
  8 * total_time_per_lap

theorem michael_total_time : total_time_8_laps = 800 :=
by
  -- The proof would go here
  sorry

end michael_total_time_l791_791079


namespace interval_solution_l791_791255

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l791_791255


namespace sqrt280_is_between_16_and_17_l791_791596

theorem sqrt280_is_between_16_and_17 : 16 < Real.sqrt 280 ∧ Real.sqrt 280 < 17 :=
by
  have h1 : 256 = 16^2 := sorry
  have h2 : 289 = 17^2 := sorry
  have h3 : 256 < 280 ∧ 280 < 289 := sorry
  have h4 : Real.sqrt 280 = sqrt 280 := by sorry
  exact ⟨lt_sqrt_of_lt_sq h3.left, sqrt_lt h3.right⟩

end sqrt280_is_between_16_and_17_l791_791596


namespace n_staircase_dissectable_l791_791751

def is_staircase (n : ℕ) : Prop := 
  ∃ (c : ℕ) (p : set (ℕ × ℕ)), (n = (c * (c + 1)) / 2) ∧ 
    (∀ i j, (i, j) ∈ p → i ≤ j ∧ j < c)

theorem n_staircase_dissectable (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ) (p : set (ℕ × ℕ)), (is_staircase k) ∧ (k < n) ∧ 
    (∀ a b, (a, b) ∈ p → (∃ (m : ℕ) (q : set (ℕ × ℕ)), is_staircase m ∧ (m < n) ∧ ((a,b) ∈ q))) :=
sorry

end n_staircase_dissectable_l791_791751


namespace solution_of_system_l791_791786

theorem solution_of_system :
  (∀ x : ℝ,
    (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2)
    → x < 1) :=
by
  sorry

end solution_of_system_l791_791786


namespace true_propositions_l791_791828

theorem true_propositions (x y a b : ℝ) :
  (xy ≠ 1 → ¬(x * y = 1)) ∧
  (∀ (T1 T2 : Triangle), congruent T1 T2 → area T1 = area T2) ∧
  (b < 0 → discriminant a b > 0) ∧
  (¬(x > 2 → x > 3)) :=
sorry

end true_propositions_l791_791828


namespace num_rooms_with_2_windows_l791_791971

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end num_rooms_with_2_windows_l791_791971


namespace inequality_solution_l791_791819

theorem inequality_solution (x : ℝ) :
  (∃ x, 2 < x ∧ x < 3) ↔ ∃ x, (x-2)*(x-3)/(x^2 + 1) < 0 := by
  sorry

end inequality_solution_l791_791819


namespace shipping_boat_earnings_l791_791212

theorem shipping_boat_earnings :
  let hourly_wage_inexperienced := 12
  let hourly_wage_A := hourly_wage_inexperienced + (hourly_wage_inexperienced / 3)
  let hourly_wage_B := hourly_wage_inexperienced + (hourly_wage_inexperienced / 4)
  let monthly_hours_A := 50 * 4
  let monthly_hours_B := 60 * 4
  let monthly_earnings_A := hourly_wage_A * monthly_hours_A
  let monthly_earnings_B := hourly_wage_B * monthly_hours_B
  let total_sailors_A := 12
  let total_sailors_B := 10
  let total_earnings_A := total_sailors_A * monthly_earnings_A
  let total_earnings_B := total_sailors_B * monthly_earnings_B
  total_earnings_A + total_earnings_B = 74400 := 
by
  simp only [hourly_wage_inexperienced, hourly_wage_A, hourly_wage_B,
    monthly_hours_A, monthly_hours_B, monthly_earnings_A, monthly_earnings_B,
    total_sailors_A, total_sailors_B, total_earnings_A, total_earnings_B]
  sorry

end shipping_boat_earnings_l791_791212


namespace interval_proof_l791_791259

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l791_791259


namespace sum_sin3_sol_l791_791615

noncomputable def sum_sin3_conditions (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180) :
  ℝ := if (Real.sin (2 * x))^3 + (Real.sin (6 * x))^3 = 8 * (Real.sin (3 * x))^3 * (Real.sin x)^3 then x else 0

theorem sum_sin3_sol : (sum_sin3_conditions 30 ⟨dec_trivial, dec_trivial⟩) + 
                        (sum_sin3_conditions 90 ⟨dec_trivial, dec_trivial⟩) + 
                        (sum_sin3_conditions 150 ⟨dec_trivial, dec_trivial⟩) = 270 :=
sorry

end sum_sin3_sol_l791_791615


namespace Kindergarten_Students_l791_791124

theorem Kindergarten_Students (X : ℕ) (h1 : 40 * X + 40 * 10 + 40 * 11 = 1200) : X = 9 :=
by
  sorry

end Kindergarten_Students_l791_791124


namespace k_squared_geq_25_div_3_l791_791225

open Real

theorem k_squared_geq_25_div_3 
  (a₁ a₂ a₃ a₄ a₅ k : ℝ)
  (h₁₂ : abs (a₁ - a₂) ≥ 1) (h₁₃ : abs (a₁ - a₃) ≥ 1) (h₁₄ : abs (a₁ - a₄) ≥ 1) (h₁₅ : abs (a₁ - a₅) ≥ 1)
  (h₂₃ : abs (a₂ - a₃) ≥ 1) (h₂₄ : abs (a₂ - a₄) ≥ 1) (h₂₅ : abs (a₂ - a₅) ≥ 1)
  (h₃₄ : abs (a₃ - a₄) ≥ 1) (h₃₅ : abs (a₃ - a₅) ≥ 1)
  (h₄₅ : abs (a₄ - a₅) ≥ 1)
  (eq1 : a₁ + a₂ + a₃ + a₄ + a₅ = 2 * k)
  (eq2 : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 2 * k^2) :
  k^2 ≥ 25 / 3 :=
by
  sorry

end k_squared_geq_25_div_3_l791_791225


namespace relationships_involving_correlation_l791_791439

def relationship_point_curve : Prop := 
  "The relationship between a point on a curve and its coordinates"

def relationship_apple_climate : Prop := 
  "The relationship between apple yield and climate"

def relationship_diameter_height_tree : Prop := 
  "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def relationship_student_id : Prop := 
  "The relationship between a student and his/her student ID"

theorem relationships_involving_correlation :
  list Prop :=
[relationship_apple_climate, relationship_diameter_height_tree] :=
by 
  sorry

end relationships_involving_correlation_l791_791439


namespace exists_positive_epsilon_l791_791069

variable {n : ℕ}
variable {x : Fin n.succ → ℝ} -- Use Fin n.succ to represent sequences of length n+1 (so for positive n it's correct)
variable (x_pos : ∀ i, 0 < x i)

theorem exists_positive_epsilon (n_pos : 0 < n) :
  ∃ ε : ℝ, 0 < ε ∧ ε = 1 / n ∧
    ∀ i : Fin n.succ,
      (∏ i, x i) ^ (1 / (n:ℝ)) ≤ 
      (1 - ε) * (1 / (n:ℝ)) * (∑ i, x i) + ε * (n:ℝ) * (∑ i, (1 / x i))⁻¹ := by
  sorry

end exists_positive_epsilon_l791_791069


namespace solution_set_l791_791071

variable {f : ℝ → ℝ}

theorem solution_set (h_deriv : ∀ x < 0, deriv f x = f' x)
    (h_ineq : ∀ x < 0, 2 * f x + x * f' x > x ^ 2) :
    {x | (x + 2016) ^ 2 * f (x + 2016) - 4 * f (-2) > 0} = set.Iio (-2018) :=
by
  sorry

end solution_set_l791_791071


namespace calculate_expression_l791_791934

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression_l791_791934


namespace price_per_pound_of_tomato_l791_791742

theorem price_per_pound_of_tomato :
  let hamburger_meat_price := 2 * 3.50
  let buns_price := 1.50
  let lettuce_price := 1.00
  let pickles_price := 2.50 - 1.00
  let known_items_total := hamburger_meat_price + buns_price + lettuce_price + pickles_price
  let total_payment := 20
  let change_received := 6
  let total_cost := total_payment - change_received
  let tomato_weight := 1.5
  let tomato_total_cost := total_cost - known_items_total
  let tomato_price_per_pound := tomato_total_cost / tomato_weight
  in tomato_price_per_pound = 2 := sorry

end price_per_pound_of_tomato_l791_791742


namespace distribute_balls_l791_791232

theorem distribute_balls (n : ℕ) (B : ℕ → ℕ) : 
  (∑ i in Finset.range n, B i = n) ∧
  (∀ i : ℕ, 1 ≤ i → i < n → B(i) >= 2 → B(i-1) >= 0 ∧ B(i+1) >= 0) → 
  ∃ B' : ℕ → ℕ, (∀ i : ℕ, i < n → B'(i) = 1) :=
begin
  sorry
end

end distribute_balls_l791_791232


namespace Lisas_mother_age_double_Lisas_in_2028_l791_791406

-- Defining the statement where we need to show that in the year 2028,
-- Lisa's mother will be twice Lisa's age given the initial conditions.

theorem Lisas_mother_age_double_Lisas_in_2028 :
  ∀ (lisa_age_2010 mother_age_2010 : ℕ), 
  lisa_age_2010 = 6 →
  mother_age_2010 = 5 * lisa_age_2010 →
  ∃ y, y = 18 ∧ 2010 + y = 2028 ∧ (mother_age_2010 + y = 2 * (lisa_age_2010 + y)) :=
by
  intros lisa_age_2010 mother_age_2010
  assume h_lisa2010 h_mother2010
  exists 18
  split
  . refl
  split
  . norm_num
  . rw [h_lisa2010, h_mother2010]
    norm_num
    sorry

end Lisas_mother_age_double_Lisas_in_2028_l791_791406


namespace smallest_integer_base17_eq_base10_l791_791500

theorem smallest_integer_base17_eq_base10 :
  ∃ n : ℕ, n > 10 ∧ (let base_10_sum := (n % 10) + ((n / 10) % 10) + (n / 100) in 
                      let base_17_digits := let d := n / 289 in
                                           let e := (n % 289) / 17 in
                                           let f := n % 17 in
                                           (d, e, f) in
                      let base_17_sum := base_17_digits.1 + base_17_digits.2 + base_17_digits.3 in
                      base_10_sum = base_17_sum) ∧ n = 153 :=
begin
  have h : ∃ n : ℕ, n = 153 := ⟨153, rfl⟩,
  cases h with n hn,
  use n,
  rw hn,
  split,
  { norm_num },
  split,
  { simp only [nat.add_zero, nat.mod_eq_of_lt, nat.div_add_mod, nat.mod_self, nat.mod_eq_zero_of_dvd, nat.dvd_refl],
    norm_num },
  { refl }
end

end smallest_integer_base17_eq_base10_l791_791500


namespace B_and_C_finish_in_22_857_days_l791_791197

noncomputable def work_rate_A := 1 / 40
noncomputable def work_rate_B := 1 / 60
noncomputable def work_rate_C := 1 / 80

noncomputable def work_done_by_A : ℚ := 10 * work_rate_A
noncomputable def work_done_by_B : ℚ := 5 * work_rate_B

noncomputable def remaining_work : ℚ := 1 - (work_done_by_A + work_done_by_B)

noncomputable def combined_work_rate_BC : ℚ := work_rate_B + work_rate_C

noncomputable def days_BC_to_finish_remaining_work : ℚ := remaining_work / combined_work_rate_BC

theorem B_and_C_finish_in_22_857_days : days_BC_to_finish_remaining_work = 160 / 7 :=
by
  -- Proof is omitted
  sorry

end B_and_C_finish_in_22_857_days_l791_791197


namespace isosceles_triangle_l791_791350

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) (hAcosB : a * Real.cos B = b * Real.cos A) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l791_791350


namespace oreo_shop_problem_l791_791572

/-- Number of different ways Charlie and Delta can leave the store with 4 products collectively. -/
theorem oreo_shop_problem : 
  let oreo_flavors := 7
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let charlie_products := 4
  ∑ i in Finset.range (charlie_products+1), 
    (Nat.choose total_flavors (charlie_products - i) * 
     (if i = 0 then 1 else (Nat.choose oreo_flavors i * (Multiset.combos i 1 (charlie_products - i)).card)) ) = 4054 :=
by
  let oreo_flavors := 7
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let charlie_products := 4
  let case0 := Nat.choose total_flavors 4
  let case1 := Nat.choose total_flavors 3 * oreo_flavors
  let case2 := Nat.choose total_flavors 2 * (Nat.choose oreo_flavors 2 + oreo_flavors)
  let case3 := Nat.choose total_flavors 1 * (Nat.choose oreo_flavors 3 + oreo_flavors * oreo_flavors + oreo_flavors)
  let case4 := (Nat.choose oreo_flavors 4 + oreo_flavors * (oreo_flavors - 1) + (oreo_flavors * (oreo_flavors - 1)) / 2 + oreo_flavors)
  let total_ways := case0 + case1 + case2 + case3 + case4
  have : total_ways = 4054 := by 
    sorry 
  exact this

end oreo_shop_problem_l791_791572


namespace find_XY_length_l791_791829

noncomputable def length_XY (a b : ℝ) (h : a = b) : ℝ :=
  let xy : ℝ := a * Real.sqrt 2 in
  xy / 2

theorem find_XY_length (a : ℝ) (h : a = 9) : length_XY a a (by rw h) = 9 * Real.sqrt 2 / 2 :=
by {
  rw h,
  sorry
}

end find_XY_length_l791_791829


namespace product_of_sums_contains_kn_terms_l791_791414

theorem product_of_sums_contains_kn_terms {α : Type} [AddCommGroup α] [Mul α] 
  (a b : ℕ → α) (k n : ℕ) :
  let S1 := ∑ i in Finset.range k, a i 
  let S2 := ∑ j in Finset.range n, b j 
  ∃ (terms : Finset ((ℕ × ℕ))), 
    (∀ i j, (i < k ∧ j < n) ↔ (i, j) ∈ terms) 
    ∧ terms.card = k * n :=
by sorry

end product_of_sums_contains_kn_terms_l791_791414


namespace diophantine_solution_exists_if_prime_divisor_l791_791746

theorem diophantine_solution_exists_if_prime_divisor (b : ℕ) (hb : 0 < b) (gcd_b_6 : Nat.gcd b 6 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 3 / (b : ℚ))) ↔ 
  ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 6 * k - 1) ∧ p ∣ b := 
by 
  sorry

end diophantine_solution_exists_if_prime_divisor_l791_791746


namespace lending_rate_is_7_percent_l791_791553

-- Conditions
def principal : ℝ := 5000
def borrowing_rate : ℝ := 0.04  -- 4% p.a. simple interest
def time : ℕ := 2  -- 2 years
def gain_per_year : ℝ := 150

-- Proof of the final statement
theorem lending_rate_is_7_percent :
  let borrowing_interest := principal * borrowing_rate * time / 100
  let interest_per_year := borrowing_interest / 2
  let total_interest_earned_per_year := interest_per_year + gain_per_year
  (total_interest_earned_per_year * 100) / principal = 7 :=
by
  sorry

end lending_rate_is_7_percent_l791_791553


namespace sun_tzu_nests_count_l791_791181

theorem sun_tzu_nests_count :
  let embankments := 9
  let trees_per_embankment := 9
  let branches_per_tree := 9
  let nests_per_branch := 9
  nests_per_branch * branches_per_tree * trees_per_embankment * embankments = 6561 :=
by
  sorry

end sun_tzu_nests_count_l791_791181


namespace vertex_of_parabola_l791_791459

theorem vertex_of_parabola (c d : ℝ) (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  ∃ v : ℝ × ℝ, v = (1, 25) :=
sorry

end vertex_of_parabola_l791_791459


namespace range_of_m_l791_791309

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 1 - abs x else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) :
  (f (f m) ≥ 0) ↔ (m ∈ Set.Icc (-2 : ℝ) (2 + Real.sqrt 2) ∨ m ∈ Set.Ici 4) :=
sorry

end range_of_m_l791_791309


namespace find_y_l791_791031

open Real

variables (A B C D F : Point)
-- Given conditions
def parallel (l1 l2 : Line) : Prop := ∀ (P Q : Point), (P ∈ l1 ∧ Q ∈ l1) → P ∈ l2 ∧ Q ∈ l2 → l1 ≠ l2 → IsParallel(Line join P Q)
def straight_line (P Q R : Point) : Prop := Angle R P Q + Angle Q P R = 180

-- Assumed given angle measures
def angle_BAC := 65
def angle_ACB := 95
def angle_ADC := 130

-- The problem to be solved
theorem find_y
  (h1 : parallel (join A B) (join D C))
  (h2 : straight_line A C F)
  (h3 : angle_BAC = 65)
  (h4 : angle_ACB = 95)
  (h5 : angle_ADC = 130) :
  y = 20 := 
sorry

end find_y_l791_791031


namespace Maggie_apples_l791_791760

-- Definition of our problem conditions
def K : ℕ := 28 -- Kelsey's apples
def L : ℕ := 22 -- Layla's apples
def avg : ℕ := 30 -- The average number of apples picked

-- Main statement to prove Maggie's apples
theorem Maggie_apples : (A : ℕ) → (A + K + L) / 3 = avg → A = 40 := by
  intros A h
  -- sorry is added to skip the proof since it's not required here.
  sorry

end Maggie_apples_l791_791760


namespace find_f_of_1_over_2016_l791_791898

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_property_0 : f 0 = 0 := sorry
lemma f_property_1 (x : ℝ) : f x + f (1 - x) = 1 := sorry
lemma f_property_2 (x : ℝ) : f (x / 3) = (1 / 2) * f x := sorry
lemma f_property_3 {x₁ x₂ : ℝ} (h₀ : 0 ≤ x₁) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1): f x₁ ≤ f x₂ := sorry

theorem find_f_of_1_over_2016 : f (1 / 2016) = 1 / 128 := sorry

end find_f_of_1_over_2016_l791_791898


namespace incorrect_statement_l791_791167

-- Definitions for conditions
structure Parallelogram (A B C D : Type) := 
(right_angle : Prop)
(equal_diagonals : Prop)

structure Quadrilateral (A B C D : Type) := 
(diagonals_bisect : Prop)
(diagonals_perpendicular : Prop)
(equal_sides : Prop)

-- Our proof problem
theorem incorrect_statement :
  ∃ (A B C D : Type),
    ¬ (Parallelogram A B C D).right_angle → 
      (Parallelogram A B C D).equal_diagonals ∧ 
      (Quadrilateral A B C D).diagonals_bisect ∧ 
      (Quadrilateral A B C D).diagonals_perpendicular ∧ 
      (Quadrilateral A B C D).equal_sides → 
      false :=
by
  intro A B C D
  intro h1 h2 h3 h4 h5
  sorry

end incorrect_statement_l791_791167


namespace households_used_both_brands_l791_791535

theorem households_used_both_brands 
  (total_households : ℕ)
  (neither_AB : ℕ)
  (only_A : ℕ)
  (h3 : ∀ (both : ℕ), ∃ (only_B : ℕ), only_B = 3 * both)
  (h_sum : ∀ (both : ℕ), neither_AB + only_A + both + (3 * both) = total_households) :
  ∃ (both : ℕ), both = 10 :=
by 
  sorry

end households_used_both_brands_l791_791535


namespace totalGrossProfit_is_436_l791_791560

def salesPricesA := [44, 47, 50]
def costsA := [20, 22, 25]
def salesPricesB := [60, 63, 65]
def costsB := [30, 33, 35]
def salesPricesC := [80, 83, 85]
def costsC := [40, 42, 45]
def salesPricesD := [100, 103, 105]
def costsD := [50, 52, 55]

def grossProfit (salesPrices : List ℕ) (costs : List ℕ) : ℕ := 
  (List.zipWith (-) salesPrices costs).sum

def totalGrossProfit : ℕ :=
  grossProfit salesPricesA costsA + 
  grossProfit salesPricesB costsB + 
  grossProfit salesPricesC costsC + 
  grossProfit salesPricesD costsD

theorem totalGrossProfit_is_436 : totalGrossProfit = 436 := by
  sorry

end totalGrossProfit_is_436_l791_791560


namespace probability_three_individuals_different_companies_l791_791207

theorem probability_three_individuals_different_companies :
  let total_reps := 5  -- 2 from Company A and 1 from each of the other 3 companies
  let total_ways := nat.choose total_reps 3
  let favorable_ways := (nat.choose 2 1) * (nat.choose 3 2)
  (favorable_ways / total_ways : ℚ) = 0.6 := by
  let total_reps := 5 
  let total_ways := nat.choose total_reps 3
  let favorable_ways := (nat.choose 2 1) * (nat.choose 3 2)
  have : (favorable_ways / total_ways : ℚ) = 0.6 := sorry  -- This needs a formal proof
  exact this

end probability_three_individuals_different_companies_l791_791207


namespace ellipse_and_chord_problem_l791_791670

theorem ellipse_and_chord_problem (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b)
  (hp : b = sqrt 2 ∧ (1 / a^2 + 4 / (3 * b^2) = 1))
  (maximum_S : ∀ Q : ℝ × ℝ, Q ≠ (0, 0) →
    ∀ F2 : ℝ × ℝ, let line_through_F2 := λ x, x - (1, -1) in
    ∀ M N : ℝ × ℝ, M ≠ N → line_through_F2(Q) = M ∧ line_through_F2(Q) = N →
      let S := (area_triangle Q F2 M + area_triangle (0,0) F2 N) in
      S <= sqrt 3 * (quot 4 3)) :
  (a = sqrt 3 ∧ b = sqrt 2 ∧ maximum_S = 2 * sqrt 3 / 3) :=
  by { sorry }

end ellipse_and_chord_problem_l791_791670


namespace factorization1_factorization2_factorization3_l791_791241

-- (1) Prove x^3 - 6x^2 + 9x == x(x-3)^2
theorem factorization1 (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

-- (2) Prove (x-2)^2 - x + 2 == (x-2)(x-3)
theorem factorization2 (x : ℝ) : (x - 2)^2 - x + 2 = (x - 2) * (x - 3) :=
by sorry

-- (3) Prove (x^2 + y^2)^2 - 4x^2*y^2 == (x + y)^2(x - y)^2
theorem factorization3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
by sorry

end factorization1_factorization2_factorization3_l791_791241


namespace rhombus_area_correct_l791_791711

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct (x : ℝ) (h1 : rhombus_area 7 (abs (8 - x)) = 56) 
    (h2 : x ≠ 8) : x = -8 ∨ x = 24 :=
by
  sorry

end rhombus_area_correct_l791_791711


namespace difference_between_x_and_y_is_36_l791_791349

theorem difference_between_x_and_y_is_36 (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := 
by 
  sorry

end difference_between_x_and_y_is_36_l791_791349


namespace sequence_exponential_value_l791_791669

theorem sequence_exponential_value :
  (∀ n : ℕ, S n = ln (1 + 1 / (n: ℝ))) →
  (e ^ (a 7 + a 8 + a 9) = 20 / 21) :=
by
  intro h
  sorry

end sequence_exponential_value_l791_791669


namespace percentage_of_muslim_boys_l791_791713

def total_boys := 400
def hindus_percentage := 0.28
def sikhs_percentage := 0.10
def other_community_boys := 72

theorem percentage_of_muslim_boys : 
  (total_boys - (hindus_percentage * total_boys).toInt - (sikhs_percentage * total_boys).toInt - other_community_boys) * 100 / total_boys = 44 := by
  sorry

end percentage_of_muslim_boys_l791_791713


namespace tank_full_capacity_l791_791890

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity_l791_791890


namespace range_increases_after_ninth_game_l791_791739

def initial_goals : List ℕ := [3, 3, 4, 5, 6, 6, 7, 10]
def ninth_game_goal : ℕ := 2

theorem range_increases_after_ninth_game :
  let updated_goals := initial_goals ++ [ninth_game_goal]
  list.range updated_goals > list.range initial_goals := 
by
  sorry

end range_increases_after_ninth_game_l791_791739


namespace calculate_k_l791_791842

-- Defining given conditions
variables {O P S : Point}
variables (k : ℝ)
variables (OP QR OQ : ℝ)

-- Assume the definitions from the conditions
def point_P := (10 : ℝ, 2 : ℝ)
def point_S := (0 : ℝ, k)
def OP_calculated := Real.sqrt (10^2 + 2^2) -- √104
def QR := 5
def OQ_calculated := OP_calculated - QR -- √104 - 5

-- Statement to prove
theorem calculate_k : k = |OQ_calculated| := by sorry

end calculate_k_l791_791842


namespace part1_conditions_part2_value_l791_791755

variable (x : ℝ)

def A := {3, x, x^2 - 2 * x}

-- Part 1: Prove the conditions for x
theorem part1_conditions : 
  (x ≠ 3 ∧ (x^2 - 2 * x) ≠ x ∧ (x^2 - 2 * x) ≠ 3) ↔ (x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 3) := 
sorry

-- Part 2: Prove the value of x if -2 is in A
theorem part2_value (h : -2 ∈ A) : x = -2 := 
sorry

end part1_conditions_part2_value_l791_791755


namespace largest_five_digit_number_is_99200_l791_791851

def largest_five_digit_number_sum_20 (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n < 100000) ∧ (n.digits.sum = 20)

theorem largest_five_digit_number_is_99200 : ∃ n : ℕ, largest_five_digit_number_sum_20 n ∧ n = 99200 :=
by
  sorry

end largest_five_digit_number_is_99200_l791_791851


namespace joseph_saves_time_l791_791048

theorem joseph_saves_time :
  let route_A_time := 2 * 5 + 2 * 0.5 + 4 * (15 / 60)
  let route_B_time := 2 * 2 + 2 * (45 / 60) + (20 / 60)
  route_A_time - route_B_time ≈ 6.17 :=
by
  sorry

end joseph_saves_time_l791_791048


namespace miraflores_can_be_elected_l791_791728

theorem miraflores_can_be_elected 
  (V : ℕ) (H₁ : V = 20000000)
  (S : ℕ) (H₂ : S = 1 / 100 * V) 
  (vote_division : (Π (V : ℕ), V = V) → (Π (groups : ℕ), groups ≤ V) → (Π (smallest_groups : ℕ), smallest_groups ≤ groups) → (smallest_groups -> bool → bool) → Prop)
  (supporters_strategize : (Π (S : ℕ), S ≤ V) → Prop)
  (supporters_majority : (S -> bool) → bool)
  : (exists election_strategy, vote_division election_strategy supporters_strategize supporters_majority → true) := 
sorry

end miraflores_can_be_elected_l791_791728


namespace number_of_correct_statements_l791_791218

/-- Definitions of the statements to be evaluated. -/
def statement_1 : Prop :=
  ∀ (T : Triangle), is_circumcenter (intersection_point (perpendicular_bisectors T))

def statement_2 : Prop :=
  ∀ (arc1 arc2 : Arc), degrees arc1 = degrees arc2 → congruent_arcs arc1 arc2

def statement_3 : Prop :=
  ∀ (C : Circle), ∀ (chord : Chord C), is_diameter chord → 
    ∀ (other_chord : Chord C), length chord ≥ length other_chord

def statement_4 : Prop :=
  ∀ (C : Circle), ∀ (chord1 chord2 : Chord C), length chord1 = length chord2 → 
    central_angle chord1 = central_angle chord2

/-- The proposition stating that the number of correct statements is 2. -/
theorem number_of_correct_statements :
  (statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ ¬statement_4) →
  2 = 2 :=
by
  sorry

end number_of_correct_statements_l791_791218


namespace abc_def_ratio_l791_791870

theorem abc_def_ratio (a b c d e f : ℝ)
    (h1 : a / b = 1 / 3)
    (h2 : b / c = 2)
    (h3 : c / d = 1 / 2)
    (h4 : d / e = 3)
    (h5 : e / f = 1 / 8) :
    (a * b * c) / (d * e * f) = 1 / 8 :=
by
  sorry

end abc_def_ratio_l791_791870


namespace estimated_probability_is_correct_l791_791156

/-- Definitions of hit condition and number groups from conditions -/
def is_hit (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 9

def count_hits (group : list ℕ) : ℕ :=
  (group.filter is_hit).length

def hits_at_least_3 (group : list ℕ) : Prop :=
  count_hits group ≥ 3

-- List of 20 groups of 4 shots results
def shot_groups : list (list ℕ) := 
  [[7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7], 
   [4, 3, 7, 3], [8, 6, 3, 6], [6, 9, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8], 
   [0, 3, 7, 1], [6, 2, 3, 3], [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1], 
   [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4], [7, 6, 1, 0], [4, 2, 8, 1]]

-- Number of shot groups with at least 3 hits
def count_groups_with_hits_at_least_3 : ℕ :=
  (shot_groups.filter hits_at_least_3).length

-- Total number of shot groups
def total_groups : ℕ := shot_groups.length

-- Estimated probability calculation
def estimated_probability : ℕ → ℕ → ℚ
| favorable, total => (favorable : ℚ) / (total : ℚ)

/-- Statement to prove the probability -/
theorem estimated_probability_is_correct :
  estimated_probability count_groups_with_hits_at_least_3 total_groups = 0.6 := by
  sorry

end estimated_probability_is_correct_l791_791156


namespace unique_solution_cond_l791_791654

open Real

theorem unique_solution_cond (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 :=
by sorry

end unique_solution_cond_l791_791654


namespace sum_f_div_2009_equals_1005_l791_791876

def f (x : ℚ) : ℚ := x^5 / (5*x^4 - 10*x^3 + 10*x^2 - 5*x + 1)

theorem sum_f_div_2009_equals_1005 :
  (∑ i in Finset.range (2009+1).succ, f (i / 2009)) = 1005 :=
sorry

end sum_f_div_2009_equals_1005_l791_791876


namespace radical_matrix_determinant_range_exist_infinite_radical_matrices_with_det_one_exist_infinite_non_radical_matrices_with_det_zero_exist_infinite_non_radical_matrices_with_det_one_l791_791382

section radical_matrix

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℚ) (B : Matrix (Fin n) (Fin n) ℚ)

-- Given Conditions
def is_rational_matrix (A : Matrix (Fin n) (Fin n) ℚ) : Prop := true

def is_radical_matrix (A : Matrix (Fin n) (Fin n) ℚ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ ∃ (X : Matrix (Fin n) (Fin n) ℚ), X ^ k = A

-- Problem Statements
theorem radical_matrix_determinant_range:
  (n ≥ 2) → is_radical_matrix A → (Matrix.det A = -1 ∨ Matrix.det A = 0 ∨ Matrix.det A = 1) :=
sorry

theorem exist_infinite_radical_matrices_with_det_one :
  ∃ (A : Matrix (Fin n) (Fin n) ℚ), is_radical_matrix A ∧ Matrix.det A = 1 :=
sorry

theorem exist_infinite_non_radical_matrices_with_det_zero :
  ∃ (B : Matrix (Fin n) (Fin n) ℚ), ¬is_radical_matrix B ∧ Matrix.det B = 0 :=
sorry

theorem exist_infinite_non_radical_matrices_with_det_one :
  ∃ (B : Matrix (Fin n) (Fin n) ℚ), ¬is_radical_matrix B ∧ Matrix.det B = 1 :=
sorry

end radical_matrix

end radical_matrix_determinant_range_exist_infinite_radical_matrices_with_det_one_exist_infinite_non_radical_matrices_with_det_zero_exist_infinite_non_radical_matrices_with_det_one_l791_791382


namespace find_m_l791_791311

-- Define the function f as given in the problem
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + (m + 2) * x + 3

-- Define the condition that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Formulate the main theorem to prove that m = -2 under the given condition that f is even
theorem find_m (m : ℝ) (h : is_even_function (f x m)) : m = -2 :=
by sorry

end find_m_l791_791311


namespace number_of_lockers_l791_791821

-- Problem Conditions
def locker_numbers_consecutive_from_one := ∀ (n : ℕ), n ≥ 1
def cost_per_digit := 0.02
def total_cost := 137.94

-- Theorem Statement
theorem number_of_lockers (h1 : locker_numbers_consecutive_from_one) (h2 : cost_per_digit = 0.02) (h3 : total_cost = 137.94) : ∃ n : ℕ, n = 2001 :=
sorry

end number_of_lockers_l791_791821


namespace rounded_product_less_than_original_l791_791102

theorem rounded_product_less_than_original
  (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hxy : x > 2 * y) :
  (x + z) * (y - z) < x * y :=
by
  sorry

end rounded_product_less_than_original_l791_791102


namespace exist_x_for_inequalities_l791_791748

theorem exist_x_for_inequalities (θ1 θ2 θ3 θ4 : ℝ)
  (hθ1 : θ1 ∈ Ioo (-π/2) (π/2))
  (hθ2 : θ2 ∈ Ioo (-π/2) (π/2))
  (hθ3 : θ3 ∈ Ioo (-π/2) (π/2))
  (hθ4 : θ4 ∈ Ioo (-π/2) (π/2))
  (h_sum : ∑ i in finset.range 4, (sin [θ1, θ2, θ3, θ4].nth i) ^ 2 ≤ 
    2 * (1 + ((sin θ1) * (sin θ2) * (sin θ3) * (sin θ4)) + ((cos θ1) * (cos θ2) * (cos θ3) * (cos θ4)))) :
  ∃ x : ℝ,
    cos (θ1) ^ 2 * cos (θ2) ^ 2 - (sin (θ1) * sin (θ2) - x) ^ 2 ≥ 0 ∧
    cos (θ3) ^ 2 * cos (θ4) ^ 2 - (sin (θ3) * sin (θ4) - x) ^ 2 ≥ 0 := sorry

end exist_x_for_inequalities_l791_791748


namespace guacamole_servings_l791_791627

theorem guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (additional_avocados : ℕ) (total_avocados : ℕ := initial_avocados + additional_avocados) (servings : ℕ := total_avocados / avocados_per_serving) :
  avocados_per_serving = 3 →
  initial_avocados = 5 →
  additional_avocados = 4 →
  servings = 3 :=
by
  intros h1 h2 h3
  unfold servings total_avocados
  rw [h1, h2, h3]
  norm_num
  rfl
  sorry

end guacamole_servings_l791_791627


namespace min_value_f_l791_791267

def indicator (x : ℝ) : ℝ := if x > 1 then 1 else 0
def f (x : ℝ) : ℝ := x + indicator x

theorem min_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∃ y : ℝ, f y = m) :=
  sorry

end min_value_f_l791_791267


namespace problem_statement_l791_791127

noncomputable def count_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let repeated_digit_choices := 5
  let positions_for_repeated_digits := Nat.choose 5 2
  let cases_for_tens_and_hundreds :=
    2 * 3 + 2 + 1
  let two_remaining_digits_permutations := 2
  repeated_digit_choices * positions_for_repeated_digits * cases_for_tens_and_hundreds * two_remaining_digits_permutations

theorem problem_statement : count_valid_numbers = 800 := by
  sorry

end problem_statement_l791_791127


namespace rooms_with_two_windows_l791_791969

theorem rooms_with_two_windows
  (total_windows : ℕ)
  (rooms_with_four_windows : ℕ)
  (windows_per_four_windows : ℕ)
  (rooms_with_three_windows : ℕ)
  (windows_per_three_windows : ℕ)
  (rooms_with_two_windows : ℕ -> ℕ -> ℕ -> ℕ -> ℕ)
  (total_rooms : 5)
  (windows_four : 4)
  (rooms_three : 8)
  (windows_three : 3)
  (result : 39) :
  rooms_with_two_windows total_windows (rooms_with_four_windows * windows_per_four_windows)
  (rooms_with_three_windows * windows_per_three_windows) 2 = result :=
by
  sorry

end rooms_with_two_windows_l791_791969


namespace triangle_angle_bisector_YE_l791_791037

noncomputable def triangle_segs_YE : ℝ := (36 : ℝ) / 7

theorem triangle_angle_bisector_YE
  (XYZ: Type)
  (XY XZ YZ YE EZ: ℝ)
  (YZ_length : YZ = 12)
  (side_ratios : XY / XZ = 3 / 4 ∧ XY / YZ  = 3 / 5 ∧ XZ / YZ = 4 / 5)
  (angle_bisector : YE / EZ = XY / XZ)
  (seg_sum : YE + EZ = YZ) :
  YE = (36 : ℝ) / 7 :=
by sorry

end triangle_angle_bisector_YE_l791_791037


namespace eight_hash_four_eq_ten_l791_791809

def operation (a b : ℚ) : ℚ := a + a / b

theorem eight_hash_four_eq_ten : operation 8 4 = 10 :=
by
  sorry

end eight_hash_four_eq_ten_l791_791809


namespace prove_inequality_example_l791_791381

noncomputable def inequality_example (n : ℕ) (k : ℝ) (x : Fin n → ℝ) : Prop :=
  2 < n ∧ 1 < k ∧ (∀ i, 0 < x i) ∧ (∑ i, 1 / (x i + 1) = 1) →
  (∑ i, 1 / (x i ^ k + 1)) ≥ n / ((n - 1) ^ k + 1)

theorem prove_inequality_example 
  (n : ℕ) (k : ℝ) (x : Fin n → ℝ) :
  inequality_example n k x :=
sorry

end prove_inequality_example_l791_791381


namespace simplify_poly_l791_791425

-- Define the polynomial expressions
def poly1 (r : ℝ) := 2 * r^3 + 4 * r^2 + 5 * r - 3
def poly2 (r : ℝ) := r^3 + 6 * r^2 + 8 * r - 7

-- Simplification goal
theorem simplify_poly (r : ℝ) : (poly1 r) - (poly2 r) = r^3 - 2 * r^2 - 3 * r + 4 :=
by 
  -- We declare the proof is omitted using sorry
  sorry

end simplify_poly_l791_791425


namespace smaller_circle_circumference_l791_791116

theorem smaller_circle_circumference (R r : ℝ) (pi_approx : ℝ) (R_approx : R = 704 / (2 * pi_approx)) (area_diff : pi_approx * (R ^ 2 - r ^ 2) = 4775.921532301596) : 2 * pi_approx * r = 659.734 :=
by 
  -- Assume the approximation as part of the conditions
  have pi_approx_def : pi_approx = 3.14159 := sorry,
  rw pi_approx_def at *,
  -- Now proceed with the given approximation
  sorry

end smaller_circle_circumference_l791_791116


namespace balls_in_boxes_l791_791332

theorem balls_in_boxes : 
  ∀ (n k : ℕ), n = 6 ∧ k = 3 ∧ ∀ i, i < k → 1 ≤ i → 
             ( ∃ ways : ℕ, ways = Nat.choose ((n - k) + k - 1) (k - 1) ∧ ways = 10 ) :=
by
  sorry

end balls_in_boxes_l791_791332


namespace sum_of_positive_ns_l791_791163

theorem sum_of_positive_ns (n : ℤ) (h : n > 0) (h1 : ∃ d : ℤ, d ∣ 36 ∧ d % 2 = 1 ∧ 2*n - 1 = d) : ∑ n in {n : ℤ | ∃ d : ℤ, d ∣ 36 ∧ d % 2 = 1 ∧ 2*n - 1 = d, n > 0}, n = 8 := sorry

end sum_of_positive_ns_l791_791163


namespace collinear_M_N_P_l791_791744

-- Define the given conditions and relationships in the problem
variables (A B C D O E P M N : Point)
variables (ω : Circle) (r s : Line)
variables (hInscribed : InscribedQuadrilateral ω A B C D O)
variables (hAngleEquality : ∠BAC = ∠ODA)
variables (hIntersectionE : E = intersection (AC) (BD))
variables (hPerpendicularR : r = Line.through E perpendicular BC)
variables (hPerpendicularS : s = Line.through E perpendicular AD)
variables (hIntersectionP : P = Line.intersection r AD)
variables (hIntersectionM : M = Line.intersection s BC)
variables (hMidpointN : N = Point.midpoint E O)

-- The theorem to be proved
theorem collinear_M_N_P : Collinear M N P :=
by
  sorry

end collinear_M_N_P_l791_791744


namespace ratio_of_b_to_a_l791_791536

variable (V A B : ℝ)

def ten_pours_of_a_cup : Prop := 10 * A = V
def five_pours_of_b_cup : Prop := 5 * B = V

theorem ratio_of_b_to_a (h1 : ten_pours_of_a_cup V A) (h2 : five_pours_of_b_cup V B) : B / A = 2 :=
sorry

end ratio_of_b_to_a_l791_791536


namespace average_price_per_book_l791_791776

variable (n1 n2 : ℕ) (p1 p2 : ℕ)
variable (h1 : n1 = 65) (h2 : n2 = 55)
variable (h3 : p1 = 1280) (h4 : p2 = 880)

theorem average_price_per_book : 
  let total_books := n1 + n2
  let total_spent := p1 + p2
  total_books = 120 → total_spent = 2160 → total_spent / total_books = 18 := 
by
  intros h_total_books h_total_spent
  rw [h_total_books, h_total_spent]
  exact Nat.div_eq_of_eq_mul_left (by norm_num) rfl sorry

end average_price_per_book_l791_791776


namespace striped_shorts_difference_l791_791355

variable (students : ℕ)
variable (striped_shirts checkered_shirts shorts : ℕ)

-- Conditions
variable (Hstudents : students = 81)
variable (Hstriped : striped_shirts = 2 * checkered_shirts)
variable (Hcheckered : checkered_shirts = students / 3)
variable (Hshorts : shorts = checkered_shirts + 19)

-- Goal
theorem striped_shorts_difference :
  striped_shirts - shorts = 8 :=
sorry

end striped_shorts_difference_l791_791355


namespace n_is_even_l791_791056

open Nat Int

theorem n_is_even (n : ℕ) (h₁ : n ≠ 0) 
  (x y : Fin n → ℤ)
  (h₂ : (∑ i, x i) = 0)
  (h₃ : (∑ i, y i) = 0)
  (h₄ : ∀ i, x i ^ 2 + y i ^ 2 = x 0 ^ 2 + y 0 ^ 2) :
  Even n := 
sorry

end n_is_even_l791_791056


namespace M_gt_N_l791_791058

-- Define the variables and conditions
variables (a : ℝ)
def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

-- Statement to prove
theorem M_gt_N : M a > N a := by
  -- Placeholder for the actual proof
  sorry

end M_gt_N_l791_791058


namespace probability_woman_lawyer_aged_36_45_l791_791714

def P_woman : ℝ := 0.40
def P_lawyer_given_woman : ℝ := 0.20
def P_aged_36_45_given_woman_lawyer : ℝ := 0.05

theorem probability_woman_lawyer_aged_36_45 :
  P_woman * P_lawyer_given_woman * P_aged_36_45_given_woman_lawyer = 0.004 :=
by
  sorry

end probability_woman_lawyer_aged_36_45_l791_791714


namespace round_table_seating_l791_791027

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem round_table_seating : 
  let n := 6 in 
  (factorial n) / (2 * n) = 60 := 
by
  sorry

end round_table_seating_l791_791027


namespace meatballs_fraction_each_son_eats_l791_791824

theorem meatballs_fraction_each_son_eats
  (f1 f2 f3 : ℝ)
  (h1 : ∃ f1 f2 f3, f1 + f2 + f3 = 2)
  (meatballs_initial : ∀ n, n = 3) :
  f1 = 2/3 ∧ f2 = 2/3 ∧ f3 = 2/3 := by
  sorry

end meatballs_fraction_each_son_eats_l791_791824


namespace diagonals_perpendicular_l791_791021

theorem diagonals_perpendicular 
  (A B C D : ℝ³)
  (M : (A + B) / 2)
  (N : (B + C) / 2)
  (P : (C + D) / 2)
  (Q : (D + A) / 2)
  (h : ∥P - M∥ = ∥Q - N∥) :
  (A - C) • (B - D) = 0 :=
sorry

end diagonals_perpendicular_l791_791021


namespace determine_base_and_digits_l791_791065

theorem determine_base_and_digits :
  ∃ (x y z b : ℕ), 
    (1987 = x * b^2 + y * b + z) ∧ 
    (x + y + z = 25) ∧ 
    (b > 12) ∧ 
    (x = 5) ∧ 
    (y = 9) ∧ 
    (z = 11) ∧ 
    (b = 19) :=
by {
  use 5, 9, 11, 19,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { refl }
}

end determine_base_and_digits_l791_791065


namespace percentage_increase_in_length_l791_791803

theorem percentage_increase_in_length (L B : ℝ) (hB : 0 < B) (hL : 0 < L) :
  (1 + x / 100) * 1.22 = 1.3542 -> x = 11.016393 :=
by
  sorry

end percentage_increase_in_length_l791_791803


namespace second_year_undeclared_fraction_l791_791766

def total_students := 12

def fraction_first_year : ℚ := 1 / 4
def fraction_second_year : ℚ := 1 / 2
def fraction_third_year : ℚ := 1 / 6
def fraction_fourth_year : ℚ := 1 / 12

def fraction_undeclared_first_year : ℚ := 4 / 5
def fraction_undeclared_second_year : ℚ := 3 / 4
def fraction_undeclared_third_year : ℚ := 1 / 3
def fraction_undeclared_fourth_year : ℚ := 1 / 6

def students_first_year : ℚ := total_students * fraction_first_year
def students_second_year : ℚ := total_students * fraction_second_year
def students_third_year : ℚ := total_students * fraction_third_year
def students_fourth_year : ℚ := total_students * fraction_fourth_year

def undeclared_first_year : ℚ := students_first_year * fraction_undeclared_first_year
def undeclared_second_year : ℚ := students_second_year * fraction_undeclared_second_year
def undeclared_third_year : ℚ := students_third_year * fraction_undeclared_third_year
def undeclared_fourth_year : ℚ := students_fourth_year * fraction_undeclared_fourth_year

theorem second_year_undeclared_fraction :
  (undeclared_second_year / total_students) = 1 / 3 :=
by
  sorry  -- Proof to be provided

end second_year_undeclared_fraction_l791_791766


namespace minimum_value_of_y_l791_791384

theorem minimum_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 36 * y) : y ≥ -7 :=
sorry

end minimum_value_of_y_l791_791384


namespace min_possible_value_of_box_l791_791339

theorem min_possible_value_of_box
  (c d : ℤ)
  (distinct : c ≠ d)
  (h_cd : c * d = 29) :
  ∃ (box : ℤ), c^2 + d^2 = box ∧ box = 842 :=
by
  sorry

end min_possible_value_of_box_l791_791339


namespace cover_plane_with_1988_circles_l791_791041

theorem cover_plane_with_1988_circles : ∃ (circles : set (set ℝ²)), 
  (∀ (p : ℝ²), ∃ (C : finset (set ℝ²)), C ⊆ circles ∧ C.card = 1988 ∧ ∀ (c ∈ C), p ∈ c) :=
sorry

end cover_plane_with_1988_circles_l791_791041


namespace inequality_and_equality_condition_l791_791279

theorem inequality_and_equality_condition (a b : ℝ) (h : a < b) :
  a^3 - 3 * a ≤ b^3 - 3 * b + 4 ∧ (a = -1 ∧ b = 1 → a^3 - 3 * a = b^3 - 3 * b + 4) :=
sorry

end inequality_and_equality_condition_l791_791279


namespace find_n_l791_791250

theorem find_n (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n ≤ 11) (h₂ : 10389 % 12 = n) : n = 9 :=
by sorry

end find_n_l791_791250


namespace interval_intersection_l791_791254

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l791_791254


namespace elements_in_set_C_l791_791424

-- Definitions and main theorem
variables (C D : Finset ℕ)  -- Define sets C and D as finite sets of natural numbers
open BigOperators    -- Opens notation for finite sums

-- Given conditions as premises
def condition1 (c d : ℕ) : Prop := c = 3 * d
def condition2 (C D : Finset ℕ) : Prop := (C ∪ D).card = 4500
def condition3 (C D : Finset ℕ) : Prop := (C ∩ D).card = 1200

-- Theorem statement to be proven
theorem elements_in_set_C (c d : ℕ) (h1 : condition1 c d)
  (h2 : ∀ (C D : Finset ℕ), condition2 C D)
  (h3 : ∀ (C D : Finset ℕ), condition3 C D) :
  c = 4275 :=
sorry  -- proof to be completed

end elements_in_set_C_l791_791424


namespace problem1_problem2_l791_791935

theorem problem1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2 * y :=
by 
  sorry

theorem problem2 (m : ℝ) (h : m ≠ 2) : (1 - m / (m + 2)) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 2 / (m - 2) :=
by 
  sorry

end problem1_problem2_l791_791935


namespace area_times_breadth_l791_791433

theorem area_times_breadth (b l A : ℕ) (h1 : b = 11) (h2 : l - b = 10) (h3 : A = l * b) : A / b = 21 := 
by
  sorry

end area_times_breadth_l791_791433


namespace marcus_sees_7_l791_791077

variable (marcus humphrey darrel : ℕ)
variable (humphrey_sees : humphrey = 11)
variable (darrel_sees : darrel = 9)
variable (average_is_9 : (marcus + humphrey + darrel) / 3 = 9)

theorem marcus_sees_7 : marcus = 7 :=
by
  -- Needs proof
  sorry

end marcus_sees_7_l791_791077


namespace ellipse_standard_equation_l791_791820

-- Define the conditions
def F1 := (-2, 0)
def F2 := (2, 0)
def a := 5
def c := 2
def b := Real.sqrt (a ^ 2 - c ^ 2)

-- The proof statement
theorem ellipse_standard_equation :
  ∀ (F1 F2 : ℝ × ℝ) (a c : ℝ), 
    F1 = (-2, 0) ∧ F2 = (2, 0) ∧ a = 5 ∧ c = 2 →
    b = Real.sqrt (a ^ 2 - c ^ 2) →
    (a ^ 2 = 25 ∧ b ^ 2 = 21) →
    (x y : ℝ), (x^2 / 25) + (y^2 / 21) = 1 :=
by
  sorry

end ellipse_standard_equation_l791_791820


namespace sum_first_2016_terms_sequence_l791_791288

noncomputable def sequence (n : ℕ) : ℚ :=
if n % 2 = 0 then 1 else 1 / 2

theorem sum_first_2016_terms_sequence :
  (∑ i in finset.range 2016, sequence i) = 1512 := by
  sorry

end sum_first_2016_terms_sequence_l791_791288


namespace sum_of_digits_of_palindrome_l791_791552

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem sum_of_digits_of_palindrome (x : ℕ) :
  is_palindrome x ∧ (100 ≤ x ∧ x ≤ 999) ∧ is_palindrome (x + 40) ∧ (1000 ≤ x + 40 ∧ x + 40 ≤ 1039) →
  (x.digits 10).sum = 16 := by
  sorry

end sum_of_digits_of_palindrome_l791_791552


namespace longest_rope_tony_can_make_l791_791835

/-- Define ropes with their initial lengths and the length lost per knot -/
def rope_length : Type := ℕ → ℕ → ℝ  -- Initial length and length lost per knot

def rope_constraints : list (rope_length) :=
  [⟨8, 1.2⟩, ⟨20, 1.5⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩,
   ⟨7, 0.8⟩, ⟨5, 1.2⟩, ⟨5, 1.2⟩]

/-- Define relationships between ropes, which ropes cannot be tied to which other ropes -/
def tying_constraints : list (ℕ × ℕ)  -- Pairs of rope indices that cannot be tied
:= [(0, 1), (5, 6), (5, 7)]

noncomputable def max_knots := 3

/-- Main goal: Prove Tony can create a rope of length 25.5 feet under the given constraints and tying rules -/
theorem longest_rope_tony_can_make : ∃ (rope_length : ℝ), rope_length = 25.5 :=
sorry

end longest_rope_tony_can_make_l791_791835


namespace triangle_cotangent_identity_l791_791061

theorem triangle_cotangent_identity (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : a / (Real.sin α) = b / (Real.sin β))
  (h3 : a / (Real.sin α) = c / (Real.sin γ))
  (h4 : b / (Real.sin β) = c / (Real.sin γ)) : 
  (\frac{\Real.cot γ}{\Real.cot α + \Real.cot β} = 1011) :=
by sorry

end triangle_cotangent_identity_l791_791061


namespace coefficient_of_x_squared_in_expansion_l791_791003

theorem coefficient_of_x_squared_in_expansion :
  ∀ (n : ℕ), n = 12 → ∀ (coeff : ℤ), coeff = ∑ k in finset.range (n + 1), ((-1)^k * nat.choose n k * (if n - 2*k = 2 then 1 else 0)) → coeff = -792 :=
by 
  intros n hn coeff hcoeff; sorry

end coefficient_of_x_squared_in_expansion_l791_791003


namespace xiao_ming_final_score_l791_791868

theorem xiao_ming_final_score :
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  (speech_image * weight_speech_image +
   content * weight_content +
   effectiveness * weight_effectiveness) = 8.3 :=
by
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  sorry

end xiao_ming_final_score_l791_791868


namespace no_cube_fourth_power_sum_l791_791097

theorem no_cube_fourth_power_sum (n : ℤ) :
  n ^ 19 ≠ a ^ 3 + b ^ 4 → (n ≡ 7 [MOD 13]) :=
by
  sorry

end no_cube_fourth_power_sum_l791_791097


namespace height_of_remaining_cube_l791_791201

theorem height_of_remaining_cube (s : ℝ) : 
  let H := s * (3 - 2 * Real.sqrt 3) / 3 in 
  H = s * (3 - 2 * Real.sqrt 3) / 3 :=
by
  -- Conditions
  have corner_cut : (cut_through_vertices : ℝ := (s * Real.sqrt 2) / 2)
                    := sorry
  have volume_removed : (volume_remove := (1 / 6) * s^3 := (1 / 6) * s^3
                    := sorry
  -- Conclusion
  sorry

end height_of_remaining_cube_l791_791201


namespace purple_beads_count_l791_791430

theorem purple_beads_count :
  ∀ (P G: ℕ), 
    G = 3 → 
    (let R := 2 * G in 
     let total_pattern_beads := G + P + R in 
     let total_bracelet_beads := 3 * total_pattern_beads in 
     let total_necklace_beads := 10 * 5 * total_pattern_beads in 
     total_bracelet_beads + total_necklace_beads = 742) → 
    P = 5 := 
by 
  sorry

end purple_beads_count_l791_791430


namespace probability_product_multiple_of_three_l791_791477

noncomputable def selected_set : set ℕ := {1, 2, 3, 4, 5, 6}

def num_combinations : ℕ := (finset.card (finset.powerset_len 3 selected_set.to_finset)).nat_cast

def multiples_of_three_combinations : ℕ := (finset.card (finset.powerset_len 3 ({1, 2, 4, 5}.to_finset))).nat_cast

theorem probability_product_multiple_of_three :
  ((num_combinations - multiples_of_three_combinations) / num_combinations : ℚ) = (4 / 5 : ℚ) := by
  sorry

end probability_product_multiple_of_three_l791_791477


namespace sequence_value_a10_l791_791678

theorem sequence_value_a10 (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 2^n) : a 10 = 1023 := by
  sorry

end sequence_value_a10_l791_791678


namespace total_length_all_segments_l791_791227

theorem total_length_all_segments (AB : ℝ) (hAB : AB = 9) : 
  let l := 1 in
  let points := finset.range 10 in
  let all_segments := finset.image2 (λ i j, |(i:ℝ) - j|) points points in
  finset.sum all_segments id = 165 :=
begin
  sorry
end

end total_length_all_segments_l791_791227


namespace purely_imaginary_iff_x_eq_neg3_l791_791396

open Complex

theorem purely_imaginary_iff_x_eq_neg3 (x : ℝ) (h_impure_imag : (x^2 + 2*x - 3 = 0) ∧ (x - 1 ≠ 0)) :
  (∃ z : ℂ, z = ((x^2 + 2*x - 3 : ℂ) + (↑(x - 1) * Complex.i)) ∧ Im z ≠ 0 ∧ Re z = 0) ↔ x = -3 :=
by
  sorry

end purely_imaginary_iff_x_eq_neg3_l791_791396


namespace parabola_increasing_implies_a_lt_zero_l791_791007

theorem parabola_increasing_implies_a_lt_zero (a : ℝ) :
  (∀ x : ℝ, x < 0 → a * (2 * x) > 0) → a < 0 :=
by
  sorry

end parabola_increasing_implies_a_lt_zero_l791_791007


namespace percentage_correct_l791_791084

theorem percentage_correct (x : ℕ) (h : x > 0) : 
  (4 * x / (6 * x) * 100 = 200 / 3) :=
by
  sorry

end percentage_correct_l791_791084


namespace determine_better_robber_l791_791484

def sum_of_odd_series (k : ℕ) : ℕ := k * k
def sum_of_even_series (k : ℕ) : ℕ := k * (k + 1)

def first_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then (k - 1) * (k - 1) + r else k * k

def second_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then k * (k + 1) else k * k - k + r

theorem determine_better_robber (n k r : ℕ) :
  if 2 * k * k - 2 * k < n ∧ n < 2 * k * k then
    first_robber_coins n k r > second_robber_coins n k r
  else if 2 * k * k < n ∧ n < 2 * k * k + 2 * k then
    second_robber_coins n k r > first_robber_coins n k r
  else 
    false :=
sorry

end determine_better_robber_l791_791484


namespace least_x_y_z_value_l791_791013

theorem least_x_y_z_value :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (3 * x = 4 * y) ∧ (4 * y = 7 * z) ∧ (3 * x = 7 * z) ∧ (x - y + z = 19) :=
by
  sorry

end least_x_y_z_value_l791_791013


namespace find_x_l791_791429

noncomputable def f (x : ℝ) := (30 : ℝ) / (x + 5)
noncomputable def h (x : ℝ) := 4 * (f⁻¹ x)

theorem find_x (x : ℝ) (hx : h x = 20) : x = 3 :=
by 
  -- Conditions
  let f_inv := f⁻¹
  have h_def : h x = 4 * f_inv x := rfl
  have f_def : f x = (30 : ℝ) / (x + 5) := rfl
  -- Needed Proof Steps
  sorry

end find_x_l791_791429


namespace simplify_sum1_simplify_sum2_l791_791783

theorem simplify_sum1 : 296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200 := by
  sorry

theorem simplify_sum2 : 457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220 := by
  sorry

end simplify_sum1_simplify_sum2_l791_791783


namespace largest_five_digit_sum_twenty_l791_791853

theorem largest_five_digit_sum_twenty : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 20) ∧ (∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m.digits.sum = 20 → m ≤ n)) ∧ n = 99200 :=
sorry

end largest_five_digit_sum_twenty_l791_791853


namespace molecular_weight_of_compound_l791_791857

theorem molecular_weight_of_compound :
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 15.999
  let atomic_weight_H := 1.008
  let molecular_weight := (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)
  molecular_weight = 78.001 :=
by
  -- Conditions
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 15.999
  let atomic_weight_H := 1.008
  let molecular_weight := (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

  -- Place the end result for the theorem.
  have molecular_weight_value : molecular_weight = 26.98 + 47.997 + 3.024 := by sorry

  have final_value : 26.98 + 47.997 + 3.024 = 78.001 := by sorry

  show molecular_weight = 78.001 from by
    rw [molecular_weight_value, final_value]

end molecular_weight_of_compound_l791_791857


namespace equation_of_line_BC_l791_791015

theorem equation_of_line_BC 
  (A : ℝ × ℝ) (hA : A = (-4, 2)) 
  (median1 : ∀ x y, 3 * x - 2 * y + 2 = 0) 
  (median2 : ∀ x y, 3 * x + 5 * y - 12 = 0) : 
  ∃ B C : ℝ × ℝ, line_eqn_2x_plus_y_minus_8_BC B C :=
begin
  -- Definitions for line_eqn_2x_plus_y_minus_8_BC go here, if necessary
  sorry
end

end equation_of_line_BC_l791_791015


namespace angle_BAE_is_10_l791_791409

theorem angle_BAE_is_10
    (A B C E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E] -- Ensures the points exist
    (E_on_AC : E ∈ line A (C : Set Point))
    (angle_ABE : angle A B E = 20)
    (angle_EBC : angle E B C = 40)
    (angle_ACB : angle A C B = 90) :
    angle B A E = 10 :=
begin
    sorry,
end

end angle_BAE_is_10_l791_791409


namespace triangle_strike_interval_l791_791866

/-- Jacob strikes the cymbals every 7 beats and the triangle every t beats.
    Given both are struck at the same time every 14 beats, this proves t = 2. -/
theorem triangle_strike_interval :
  ∃ t : ℕ, t ≠ 7 ∧ (∀ n : ℕ, (7 * n % t = 0) → ∃ k : ℕ, 7 * n = 14 * k) ∧ t = 2 :=
by
  use 2
  sorry

end triangle_strike_interval_l791_791866


namespace minimum_value_of_M_l791_791673

-- Defining the function f(x)
def f (x a b : ℝ) : ℝ := abs (x + (1 / x) - a * x - b)

-- Define the maximum M(a, b) in the interval [1/2, 2]
def M (a b : ℝ) : ℝ := max (f (1/2) a b) (f 2 a b)

-- The main statement in Lean to prove the minimum of M(a, b)
theorem minimum_value_of_M : 
  (∃ a b : ℝ, M a b = 1/4) :=
sorry

end minimum_value_of_M_l791_791673


namespace first_group_completion_days_l791_791337

-- Defining the daily work ratios and group sizes
def man_daily_work_ratio := 2
def boy_daily_work_ratio := 1

-- Group 1: 12 men and 16 boys working for x days
def group1_men := 12
def group1_boys := 16

-- Group 2: 13 men and 24 boys working for 4 days
def group2_men := 13
def group2_boys := 24
def group2_days := 4

-- Defining the hypothesis based on the conditions
def group2_work := group2_days * (group2_men * man_daily_work_ratio + group2_boys * boy_daily_work_ratio)

theorem first_group_completion_days :
  (group1_men * man_daily_work_ratio + group1_boys * boy_daily_work_ratio) * 5 = group2_work :=
by
  -- Hypothesis: 13 men and 24 boys complete the work in 4 days, with ratio 2:1
  have h1 : group2_work = 200 := by
    unfold group2_work
    have h : 4 * (13 * 2 + 24 * 1) = 200 := by norm_num
    exact h
  rw h1
  -- We need to show this is equal to the first group's work over 5 days
  unfold man_daily_work_ratio boy_daily_work_ratio group1_men group1_boys
  norm_num 
  sorry

end first_group_completion_days_l791_791337


namespace smallest_integer_solution_l791_791502

theorem smallest_integer_solution : 
  ∃ x : ℤ, x^2 = 3 * x + 100 ∧ ∀ y : ℤ, y^2 = 3 * y + 100 → x ≤ y :=
begin
  sorry
end

end smallest_integer_solution_l791_791502


namespace eleven_positive_numbers_equal_sum_of_squares_l791_791965

theorem eleven_positive_numbers_equal_sum_of_squares :
  ∃ a : ℝ, (∀ i : Fin 11, a > 0 ∧ (a = (finset.univ.erase i).sum (λ j, (a : ℝ) ^ 2))) ↔ a = 1 / 10 :=
by
  sorry

end eleven_positive_numbers_equal_sum_of_squares_l791_791965


namespace find_all_valid_n_l791_791977

-- Define the natural number conditions

def is_not_divisible_by_perfect_square(n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

def has_one_prime_divisor_of_form_4k_plus_3(n : ℕ) : Prop :=
  ∃ p : ℕ, (∃ k : ℕ, p = 4 * k + 3) ∧ nat.prime p ∧ p ∣ n ∧ ∀ q : ℕ, q ∣ n → nat.prime q → q = p

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

def num_of_divisors (n : ℕ) : ℕ :=
  (n.divisors).length

def sum_of_digits_plus_two_eq_num_of_divisors(n : ℕ) : Prop :=
  sum_of_digits n + 2 = num_of_divisors n

def n_plus_3_is_perfect_square(n : ℕ) : Prop :=
  ∃ m : ℕ, n + 3 = m * m

def has_no_large_prime_divisor(n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p < 10000

-- Final proof problem
theorem find_all_valid_n (n : ℕ) :
  is_not_divisible_by_perfect_square n ∧
  has_one_prime_divisor_of_form_4k_plus_3 n ∧
  sum_of_digits_plus_two_eq_num_of_divisors n ∧
  n_plus_3_is_perfect_square n ∧
  has_no_large_prime_divisor n →
  n = 222 ∨ n = 2022 :=
by
  sorry

end find_all_valid_n_l791_791977


namespace total_number_of_posters_l791_791213

theorem total_number_of_posters : 
  ∀ (P : ℕ), 
  (2 / 5 : ℚ) * P + (1 / 2 : ℚ) * P + 5 = P → 
  P = 50 :=
by
  intro P
  intro h
  sorry

end total_number_of_posters_l791_791213


namespace sine_angle_OC_plane_AOB_l791_791304

-- Definitions for the given conditions
variables (A B O C : Type)  -- Points A, B, O and C in space
variables [IsPoint A] [IsPoint B] [IsPoint O] [IsPoint C]
variables (angle : Type) [IsAngle angle]

-- Given angles and conditions
axiom ∠AOB_is_90 : IsRightAngle (angle A O B)
axiom ∠AOC_is_60 : IsAngleOfMeasure (angle A O C) 60
axiom ∠BOC_is_60 : IsAngleOfMeasure (angle B O C) 60
axiom C_in_space : InSpace C

-- Target theorem
theorem sine_angle_OC_plane_AOB : ∃ θ, IsSineOfAngle θ (line O C) (plane A O B) (1/2) := by
  sorry

end sine_angle_OC_plane_AOB_l791_791304


namespace probability_between_is_0_33_l791_791584

-- Define the first line p
def line_p (x : ℝ) : ℝ := -2 * x + 8

-- Define the second line q
def line_q (x : ℝ) : ℝ := -3 * x + 8

-- Define the area under line p
def area_p : ℝ := 1 / 2 * 4 * 8

-- Define the area under line q
def area_q : ℝ := 1 / 2 * (8 / 3) * 8

-- Define the area between lines p and q
def area_between : ℝ := area_p - area_q

-- Define the probability of a point falling between the lines
def probability_between : ℝ := area_between / area_p

-- Theorem stating the probability is 0.33
theorem probability_between_is_0_33 : probability_between = 0.33 := 
by 
  -- Proof can be filled here
  sorry

end probability_between_is_0_33_l791_791584


namespace valid_number_count_l791_791328

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime_digit (d : ℕ) : Prop := d = 2

def valid_number (n : ℕ) : Prop :=
  8000 ≤ n ∧ n < 9000 ∧
  is_even n ∧
  is_prime_digit (n % 10) ∧
  (let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
   digits.nodup)

def count_valid_numbers : ℕ :=
  (finset.range 10000).filter valid_number |>.card

theorem valid_number_count : count_valid_numbers = 56 :=
by
  sorry

end valid_number_count_l791_791328


namespace part_a_l791_791757

def P (k : Nat) (n : Nat) : Prop :=
  ∃ m : Nat, n = ∏ i in Ico m (m + k), (i + 1)

theorem part_a : ∃ (k : Nat) (N : Nat), k = 3 ∧ P k N ∧ P (k + 2) N := by
  sorry

end part_a_l791_791757


namespace power_mod_19_l791_791495

theorem power_mod_19 :
  ∀ n : ℕ, 7^n % 19 = 7 ↔ n % 18 = 1 :=
begin
  sorry
end

example : 7^2023 % 19 = 4 :=
begin
  have h := power_mod_19 2023,
  rw nat.mod_eq_of_lt, -- 2023 % 18 = 7
  norm_num,
  exact h,
  sorry
end

end power_mod_19_l791_791495


namespace determine_base_l791_791589

theorem determine_base (x : ℕ) (h : 2 * x^3 + x + 6 = x^3 + 2 * x + 342) : x = 7 := 
sorry

end determine_base_l791_791589


namespace find_minimizing_point_M_l791_791671

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def point_P : ℝ × ℝ := (1, -1)

def left_focus_F : ℝ × ℝ := (-1, 0)

def min_distance_cond (M : ℝ × ℝ) : Prop :=
  let (Mx, My) := M in
  ellipse Mx My ∧ ∀ (N : ℝ × ℝ), ellipse N.1 N.2 →
  let (Nx, Ny) := N in
  dist (Mx, My) point_P + 2 * dist (Mx, My) left_focus_F ≤
  dist (Nx, Ny) point_P + 2 * dist (Nx, Ny) left_focus_F

theorem find_minimizing_point_M : ∃ (M : ℝ × ℝ), min_distance_cond M ∧
  M = (2/3 * (6:ℝ).sqrt, -1) := sorry

end find_minimizing_point_M_l791_791671


namespace maximum_value_f_derivative_F_at_x0_l791_791315

-- Statement for part (1)
theorem maximum_value_f (m : ℝ) (h : m = -1) :
  ∃ x : ℝ, f x = -(1 / (Real.exp 2)) :=
sorry

-- Statement for part (2)
theorem derivative_F_at_x0 (m : ℝ) (x₁ x₂ : ℝ) (hx₀ : 0 < x₁) (hx₁₂ : x₁ < x₂) :
  F (√(x₁ * x₂)) = 0 →
  F x₁ = 0 →
  F x₂ = 0 →
  F' (√(x₁ * x₂)) < 0 :=
sorry

-- Functions definitions
def f (x : ℝ) : ℝ := (m * x^2 - 3 * x - 3) * Real.exp x

def M (x : ℝ) : ℝ := Real.exp (-x) * f x - m * (x^2 - x)

def N (x : ℝ) : ℝ := Real.log x - 4 * x - 3

def F (x : ℝ) : ℝ := M x - N x

-- Definitions of derivative not shown in the problem, but can assume F' is defined
def F' (x : ℝ) : ℝ := (fun x => (d(F x)) / (d(x))) -- Simplified representation, please adjust accordingly

end maximum_value_f_derivative_F_at_x0_l791_791315


namespace find_e_l791_791067

-- Conditions
def f (x : ℝ) (b : ℝ) := 5 * x + b
def g (x : ℝ) (b : ℝ) := b * x + 4
def f_comp_g (x : ℝ) (b : ℝ) (e : ℝ) := 15 * x + e

-- Statement to prove
theorem find_e (b e : ℝ) (x : ℝ): 
  (f (g x b) b = f_comp_g x b e) → 
  (5 * b = 15) → 
  (20 + b = e) → 
  e = 23 :=
by 
  intros h1 h2 h3
  sorry

end find_e_l791_791067


namespace solve_for_x_l791_791428

def h (x : ℝ) : ℝ := 4 * (f⁻¹ x)
def f (x : ℝ) : ℝ := 30 / (x + 2)

theorem solve_for_x (x : ℝ) (h_condition : h(x) = 20) : x = 30 / 7 :=
by
  sorry

end solve_for_x_l791_791428


namespace sum_of_four_consecutive_integers_with_product_5040_eq_34_l791_791449

theorem sum_of_four_consecutive_integers_with_product_5040_eq_34 :
  ∃ a b c d : ℕ, a * b * c * d = 5040 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a + b + c + d) = 34 :=
sorry

end sum_of_four_consecutive_integers_with_product_5040_eq_34_l791_791449


namespace books_borrowed_in_a_week_l791_791050

theorem books_borrowed_in_a_week 
  (daily_avg : ℕ)
  (friday_increase_pct : ℕ)
  (days_open : ℕ)
  (friday_books : ℕ)
  (total_books_week : ℕ)
  (h1 : daily_avg = 40)
  (h2 : friday_increase_pct = 40)
  (h3 : days_open = 5)
  (h4 : friday_books = daily_avg + (daily_avg * friday_increase_pct / 100))
  (h5 : total_books_week = (days_open - 1) * daily_avg + friday_books) :
  total_books_week = 216 :=
by {
  sorry
}

end books_borrowed_in_a_week_l791_791050


namespace possible_values_of_m_l791_791731

theorem possible_values_of_m (m : ℕ) (h1 : 3 * m + 15 > 3 * m + 8) 
  (h2 : 3 * m + 8 > 4 * m - 4) (h3 : m > 11) : m = 11 := 
by
  sorry

end possible_values_of_m_l791_791731


namespace number_of_vertices_in_good_regions_lt_40n_l791_791141

-- Definitions based on the given conditions
def rectangles_on_plane (n : ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → parallel_sides (rectangles i) (rectangles j) ∧ 
  ∀ i j : ℕ, i ≠ j → ¬same_line (sides (rectangles i)) (sides (rectangles j))

def good_region (R: set Point): Prop :=
  ∃ v ∈ vertices, v ∈ boundary_of(R)

-- Theorem statement
theorem number_of_vertices_in_good_regions_lt_40n (n : ℕ)
  (h1 : rectangles_on_plane n)
  (h2 : ∀ R, good_region R → 
    ∃ vertices_boundary, vertices_of_boundary R ≤ 40 * n) : 
  ∑ R, good_region R ∧ vertices_of_boundary R < 40 * n :=
sorry

end number_of_vertices_in_good_regions_lt_40n_l791_791141


namespace least_remaining_matches_is_correct_l791_791906

noncomputable def least_remaining_matches (n : ℕ) : ℕ := 2 * (2 * n - 1)

theorem least_remaining_matches_is_correct (n : ℕ) : 
  ∃ (initial_matches remaining_matches : ℕ),
    (let vertices := (2 * n + 1)^2 in
     let total_matches := 4 * n^2 in
     initial_matches = total_matches ∧
     remaining_matches = least_remaining_matches(n)) :=
sorry

end least_remaining_matches_is_correct_l791_791906


namespace parabola_focus_coordinates_l791_791158

theorem parabola_focus_coordinates (x y p : ℝ) (h : y^2 = 8 * x) : 
  p = 2 → (p, 0) = (2, 0) := 
by 
  sorry

end parabola_focus_coordinates_l791_791158


namespace circle_of_unit_diameter_can_be_placed_without_intersecting_squares_l791_791191

open Real Set

/-- Proving there exists an available space for a circle of unit diameter inside a 20x25 rectangle,
even when 120 unit squares are placed arbitrarily in the rectangle without intersection. -/
theorem circle_of_unit_diameter_can_be_placed_without_intersecting_squares :
  ∃ (circle_center : Real × Real),
  (∀ (sq_center : Real × Real) ((sq_orientation : ℝ), sq_center ∈ unit_squares ∧ within_rectangle) ->
  (dist circle_center sq_center ≥ 1)) ∧
  within_rectangle circle_center :=
sorry

noncomputable def unit_squares : Set (ℝ × ℝ) := 
{ sq | ∃ (sqc : ℝ × ℝ) (sqo : ℝ), within_rectangle sqc ∧ orientation sqo }

def within_rectangle (pt : ℝ × ℝ) : Prop :=
pt.1 ≥ 0 ∧ pt.1 ≤ 20 ∧ pt.2 ≥ 0 ∧ pt.2 ≤ 25

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def orientation (sq : Real × Real) : Prop :=
-- Assuming arbitrary representation of unit square orientation
true -- this is a placeholder

end circle_of_unit_diameter_can_be_placed_without_intersecting_squares_l791_791191


namespace vending_machine_total_snacks_dropped_l791_791139

-- Conditions
def fails_to_drop_snack_prob := 1 / 6
def drops_two_snacks_prob := 1 / 10
def total_uses := 30

-- Theorem to prove
theorem vending_machine_total_snacks_dropped :
  let failed_drops := total_uses * fails_to_drop_snack_prob in
  let double_drops := total_uses * drops_two_snacks_prob in
  let normal_drops := total_uses - failed_drops - double_drops in
  let total_snacks := normal_drops + 2 * double_drops in
  total_snacks = 28 := by
  sorry

end vending_machine_total_snacks_dropped_l791_791139


namespace range_of_m_for_log_function_domain_l791_791798

theorem range_of_m_for_log_function_domain (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + m > 0) → m > 8 :=
by
  sorry

end range_of_m_for_log_function_domain_l791_791798


namespace edge_length_of_small_cube_l791_791635

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end edge_length_of_small_cube_l791_791635


namespace probability_event_1_probability_event_2_l791_791202

-- Definition of the sample space for two rolls of a six-sided die
def sample_space := { (m, n) : ℕ × ℕ | m ∈ {1, 2, 3, 4, 5, 6} ∧ n ∈ {1, 2, 3, 4, 5, 6} }

-- Condition for the quadratic equation having two distinct real roots
def distinct_real_roots (m n : ℕ) : Prop :=
  m^2 - 4 * n^2 > 0

-- Definition of the first probability event
def event_1 := {pair ∈ sample_space | distinct_real_roots pair.1 pair.2 }

-- Probability calculation for event_1
theorem probability_event_1 : (event_1.to_finset.card : ℚ) / sample_space.to_finset.card = 1 / 6 :=
by
  sorry

-- Condition for m/n not being an integer
def not_integer (m n : ℕ) : Prop :=
  ¬ (m % n = 0)

-- Definition of the second probability event
def event_2 := {pair ∈ sample_space | not_integer pair.1 pair.2 }

-- Probability calculation for event_2
theorem probability_event_2 : (event_2.to_finset.card : ℚ) / sample_space.to_finset.card = 11 / 18 :=
by
  sorry

end probability_event_1_probability_event_2_l791_791202


namespace edge_length_of_cut_off_cube_l791_791634

theorem edge_length_of_cut_off_cube (V_large V_remaining : ℕ) 
  (h1 : V_large = 1000) (h2 : V_remaining = 488) : 
  ∃ x : ℕ, x ^ 3 = 64 ∧ 1000 - 8 * x ^ 3 = V_remaining := by
    use 4
    split
    · norm_num
    · exact h2.symm
appointment

end edge_length_of_cut_off_cube_l791_791634


namespace gcd_60_90_l791_791490

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end gcd_60_90_l791_791490


namespace georgie_guacamole_servings_l791_791625

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end georgie_guacamole_servings_l791_791625


namespace sum_of_angles_at_9_20_and_2_50_l791_791231

def angle_at_9_20 : ℝ :=
  let hour_hand_position := 270.0 + 20 * 0.5
  let minute_hand_position := 20 * 6
  let angle_diff := |hour_hand_position - minute_hand_position|
  if angle_diff > 180 then 360 - angle_diff else angle_diff

def angle_at_2_50 : ℝ :=
  let hour_hand_position := 60.0 + 50 * 0.5
  let minute_hand_position := 50 * 6
  let angle_diff := |hour_hand_position - minute_hand_position|
  if angle_diff > 180 then 360 - angle_diff else angle_diff

theorem sum_of_angles_at_9_20_and_2_50 : angle_at_9_20 + angle_at_2_50 = 305 := 
  by
  sorry

end sum_of_angles_at_9_20_and_2_50_l791_791231


namespace circumcircles_tangent_l791_791386

variable (A B C D E F K L : Type*)
variable [EuclideanGeometry : EuclideanGeometry.has_lines A B C D E F K L]

open EuclideanGeometry

def reflections (A B C D E F : Type*) : Prop :=
  ∃ (BC CD : line),
    reflection A BC = E ∧ reflection A CD = F

def quadrilateral (A B C D : Type*) : Prop :=
  is_convex_quadrilateral A B C D ∧
  ∠A B C > 90° ∧ ∠C D A > 90° ∧ ∠D A B = ∠B C D

theorem circumcircles_tangent
  (A B C D E F K L : Type*)
  [EuclideanGeometry : EuclideanGeometry.has_lines A B C D E F K L]
  (h_quad : quadrilateral A B C D)
  (h_reflect : reflections A B C D E F)
  (h_meet_K : meets AE BD K)
  (h_meet_L : meets AF BD L) :
  is_tangent (circumcircle (B E K)) (circumcircle (D F L)) :=
sorry

end circumcircles_tangent_l791_791386


namespace polynomials_dividing_2_pow_p_minus_2_l791_791237

theorem polynomials_dividing_2_pow_p_minus_2 :
  { f : Polynomial ℤ | ∀ p, Prime p → p % 2 = 1 → f.eval p ∣ (2^p - 2) }
  = { Polynomial.C a * X^n | a ∈ {±1, ±2, ±3, ±6} ∧ n ∈ {0, 1} } :=
by
  sorry

end polynomials_dividing_2_pow_p_minus_2_l791_791237


namespace minimum_peaches_eaten_l791_791473

-- Defining the initial number of peaches in each pile
def pile1 : Nat := 93
def pile2 : Nat := 70
def pile3 : Nat := 63

-- Defining the total number of peaches
def total_peaches : Nat := pile1 + pile2 + pile3

-- This theorem states the equivalent math proof problem
theorem minimum_peaches_eaten : ∃ n : Nat, n = 4 ∧ ∃ (moves : (Nat → Nat) → Nat), ∀ f, 
  (∀ k, 0 < k ∧ k ≤ 5 → f (k) = k - 1) ∧
  (pile1 + pile2 + pile3 - (moves f) = 225) :=
begin
    sorry
end

end minimum_peaches_eaten_l791_791473


namespace find_angle_B_and_sin_C_l791_791294

theorem find_angle_B_and_sin_C
  (a b c R : ℝ)
  (h1: 0 < B) (h2: B < Real.pi / 2)
  (equation: 2 * R - b = 2 * b * Real.sin B)
  (h3: a = Real.sqrt 3)
  (h4: c = 3)
  : B = Real.pi / 6 ∧ Real.sin C = Real.sqrt 3 / 2 :=
begin
  sorry
end

end find_angle_B_and_sin_C_l791_791294


namespace correct_sampling_methods_l791_791189

-- Definitions for different sampling methods
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

-- Conditions from the problem
def situation1 (students_selected_per_class : Nat) : Prop :=
  students_selected_per_class = 2

def situation2 (students_above_110 : Nat) (students_between_90_and_100 : Nat) (students_below_90 : Nat) : Prop :=
  students_above_110 = 10 ∧ students_between_90_and_100 = 40 ∧ students_below_90 = 12

def situation3 (tracks_arranged_for_students : Nat) : Prop :=
  tracks_arranged_for_students = 6

-- Theorem
theorem correct_sampling_methods :
  ∀ (students_selected_per_class students_above_110 students_between_90_and_100 students_below_90 tracks_arranged_for_students: Nat),
  situation1 students_selected_per_class →
  situation2 students_above_110 students_between_90_and_100 students_below_90 →
  situation3 tracks_arranged_for_students →
  (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) = (SamplingMethod.Systematic, SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
by
  intros
  rfl

end correct_sampling_methods_l791_791189


namespace equation_of_parallel_line_l791_791248

noncomputable def line_parallel_and_intercept (m : ℝ) : Prop :=
  (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ z : ℝ, 3*z + m = 0)

theorem equation_of_parallel_line {m : ℝ} :
  line_parallel_and_intercept m ↔ (∃ x y : ℝ, x + y + 2 = 0) ∨ (∃ x y : ℝ, x + y - 2 = 0) :=
by
  sorry

end equation_of_parallel_line_l791_791248


namespace ways_to_select_weights_eq_coefficient_l791_791423

open Polynomial

theorem ways_to_select_weights_eq_coefficient : 
  (∃ m : ℕ, (1 + X) * (1 + X^2) * (1 + X^3) * (1 + X^4) * (1 + X^5) * 
  (1 + X^6) * (1 + X^7) * (1 + X^8) * (1 + X^9) * (1 + X^10) * (1 + X^11)) 
  .coeff 9 = m :=
begin
  sorry
end

end ways_to_select_weights_eq_coefficient_l791_791423


namespace condition_an_necessary_but_not_sufficient_l791_791948

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient_l791_791948


namespace solve_sqrt_equation_l791_791785

theorem solve_sqrt_equation (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a ≤ 1) (h_x : x = (a^2 + 1)^2 / (4 * a^2)) :
    (sqrt (x + sqrt x) - sqrt (x - sqrt x)) = ((a + 1) * sqrt (x / (x + sqrt x))) :=
by 
  sorry

end solve_sqrt_equation_l791_791785


namespace number_of_pupils_l791_791210

-- Define the conditions
variable (n : ℕ) -- the number of pupils
variable (incorrect_entry : ℕ := 67)
variable (correct_entry : ℕ := 45)
variable (increase_in_marks : ℕ := incorrect_entry - correct_entry) (increase_in_marks = 22)

-- Define the hypothesis that average increased by 0.5
variable (average_increase : ℚ := 0.5)
variable (h : increase_in_marks / n = average_increase)

-- The statement to prove
theorem number_of_pupils (hn : (22:ℚ) / n = 0.5) : n = 44 := 
sorry

end number_of_pupils_l791_791210


namespace question_I_question_II_l791_791314

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x - 1 + a / Real.exp x

-- Question I proof statement
theorem question_I (a : ℝ) (h : deriv (fun x => f x a) 1 = 0) : a = Real.exp 1 := by
  sorry

-- Define the function g(x) from solution to Question II
def g (x k : ℝ) : ℝ := (1 - k) * x + 1 / Real.exp x

-- Condition that g(x) has no zero crossings (for k <= 1)
def no_zero_crossings (k : ℝ) : Prop :=
  ∀ x : ℝ, g x k ≠ 0

-- Question II proof statement
theorem question_II (k : ℝ) (h : no_zero_crossings k) : k ≤ 1 ∧ (k = 1 → ∀ x : ℝ, g x k > 0) := by
  sorry

end question_I_question_II_l791_791314


namespace exponential_fixed_point_l791_791801

theorem exponential_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (a^(4-4) + 5 = 6) :=
sorry

end exponential_fixed_point_l791_791801


namespace find_b_l791_791675

noncomputable def hyperbola1_asymptotes : Set (ℝ × ℝ) :=
  {p | p.2 = sqrt 2 * p.1 ∨ p.2 = - (sqrt 2) * p.1}

noncomputable def hyperbola2_asymptotes (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = (b / a) * p.1 ∨ p.2 = - (b / a) * p.1}

def hyperbola_focal_length (a b : ℝ) : ℝ :=
  2 * sqrt (a^2 + b^2)

theorem find_b (a b : ℝ) (h_asymptotes : hyperbola1_asymptotes = hyperbola2_asymptotes a b)
  (h_focal_length : hyperbola_focal_length a b = 4 * sqrt 5) : 
  b = 4 := by
  sorry

end find_b_l791_791675


namespace exists_positive_m_l791_791283

noncomputable def f (x : ℕ) : ℕ := 3 * x + 2

def f_iter (x : ℕ) (n : ℕ) : ℕ :=
if h : n = 0 then x else iterate (λ y, f y) n x

def divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem exists_positive_m : ∃ m : ℕ, m > 0 ∧ (divisible_by (f_iter m 100) 1988) :=
sorry

end exists_positive_m_l791_791283


namespace max_annual_profit_l791_791791

-- Define the conditions and functions mentioned in the problem
def revenue (x : ℝ) : ℝ := 
  if x > 20 then 70 * x + 2000 - 9000 / (x + 1) else (180 - 2 * x) * x

def cost (x : ℝ) : ℝ := 500 + 80 * x

def profit (x : ℝ) : ℝ := revenue x - cost x

-- State the theorem to be proved
theorem max_annual_profit :
  let y := profit 29
  in y = 1360 :=
by
  sorry

end max_annual_profit_l791_791791


namespace counterpositive_prop_l791_791996

theorem counterpositive_prop (a b c : ℝ) (h : a^2 + b^2 + c^2 < 3) : a + b + c ≠ 3 := 
sorry

end counterpositive_prop_l791_791996


namespace student_arrangement_l791_791363

theorem student_arrangement :
  ∀ (students : Fin 5 → Nat),
  ∀ (s1 s2 s3 s4 s5 : Nat),
  (students 0) = s1 →
  (students 1) = s2 →
  (students 2) = s3 →
  (students 3) = s4 →
  (students 4) = s5 →
  let refuses_to_stand_next_to : Nat -> Nat -> Prop := 
    λ s1 s2, (s1, s2) = (students 1, students 2) ∨ (s2, s1) = (students 1, students 2) in
  let must_be_at_end : Nat -> Prop := λ s, s = students 0 ∨ s = students 4 in
  ∃ (valid_arrangement_count : Nat), valid_arrangement_count = 144 :=
sorry

end student_arrangement_l791_791363


namespace find_y_l791_791504

theorem find_y (y : ℚ) : (3 / y - (3 / y) * (y / 5) = 1.2) → y = 5 / 3 :=
sorry

end find_y_l791_791504


namespace number_of_common_points_l791_791126

theorem number_of_common_points :
  let line_eq : (ℝ × ℝ) → Prop := λ p, ∃ t, p.1 = 2 + t ∧ p.2 = 4 - t
  let curve_eq : (ℝ × ℝ) → Prop := λ p, ∃ θ, p.1 = 3 + sqrt 2 * cos θ ∧ p.2 = 5 + sqrt 2 * sin θ
  ∃! p : ℝ × ℝ, line_eq p ∧ curve_eq p :=
sorry

end number_of_common_points_l791_791126


namespace distance_difference_PA_PB_l791_791364

noncomputable def parametric_line_eq (t : ℝ) : ℝ × ℝ :=
(1 + 2 * t, 2 + t)

noncomputable def polar_circle_eq : ℝ :=
2

noncomputable def point_P : ℝ × ℝ := (-1, 1)

theorem distance_difference_PA_PB :
  let A := parametric_line_eq
  let B := parametric_line_eq
  (| - / PA - | PB = 2 *  5/ 5 : = sorry

end distance_difference_PA_PB_l791_791364


namespace coupon_calculation_l791_791908

theorem coupon_calculation :
  let initial_stock : ℝ := 40.0
  let sold_books : ℝ := 20.0
  let coupons_per_book : ℝ := 4.0
  let remaining_books := initial_stock - sold_books
  let total_coupons := remaining_books * coupons_per_book
  total_coupons = 80.0 :=
by
  sorry

end coupon_calculation_l791_791908


namespace functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l791_791888

-- Conditions for y1
def cost_price : ℕ := 60
def selling_price_first_10_days : ℕ := 80
def y1 : ℕ → ℕ := fun x => x * x - 8 * x + 56
def items_sold_day4 : ℕ := 40
def items_sold_day6 : ℕ := 44

-- Conditions for y2
def selling_price_post_10_days : ℕ := 100
def y2 : ℕ → ℕ := fun x => 2 * x + 8
def gross_profit_condition : ℕ := 1120

-- 1) Prove functional relationship of y1.
theorem functional_relationship_y1 (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) : 
  y1 x = x * x - 8 * x + 56 := 
by
  sorry

-- 2) Prove value of x for daily gross profit $1120 on any day within first 10 days.
theorem daily_gross_profit_1120_first_10_days (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) (gp : (selling_price_first_10_days - cost_price) * y1 x = gross_profit_condition) : 
  x = 8 := 
by
  sorry

-- 3) Prove total gross profit W and range for 26 < x ≤ 31.
theorem total_gross_profit_W (x : ℕ) (h : 26 < x ∧ x ≤ 31) : 
  (100 - (cost_price - 2 * (y2 x - 60))) * (y2 x) = 8 * x * x - 96 * x - 512 := 
by
  sorry

end functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l791_791888


namespace bake_sale_money_made_l791_791943

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end bake_sale_money_made_l791_791943


namespace pyramid_volume_l791_791112

open Real

variables (α V : ℝ)

def volume_pyramid (α V : ℝ) : ℝ :=
  V * ((cos (α / 2))^2 * sin α) / (3 * π)

theorem pyramid_volume (hα : 0 < α ∧ α < π) (hV : 0 < V) :
  volume_pyramid α V = V * (cos (α / 2))^2 * sin α / (3 * π) := 
sorry

end pyramid_volume_l791_791112


namespace nicky_pace_theorem_l791_791082

variables (head_start : ℕ) (cristina_pace : ℕ) (time : ℕ) (nicky_pace : ℕ)

-- Given conditions
def problem_conditions : Prop :=
  head_start = 54 ∧ cristina_pace = 5 ∧ time = 27

-- We want to prove Nicky's pace
def nicky_pace_proof : Prop :=
  nicky_pace = 3

-- The theorem statement encapsulating the proof problem
theorem nicky_pace_theorem (h : problem_conditions) : nicky_pace_proof :=
  sorry

end nicky_pace_theorem_l791_791082


namespace tony_rope_length_l791_791837

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l791_791837


namespace problem_solution_l791_791034

noncomputable def problem_statement : Prop :=
  ∀ (A B C : Type) [EuclideanGeometry.point A] [EuclideanGeometry.triangle A B C],
  ∠B = 45 ∧ AB = 100 ∧ AC = 100 * Real.sqrt 2 →
  BC = 100

theorem problem_solution: problem_statement :=
by
  sorry

end problem_solution_l791_791034


namespace most_irregular_acute_triangle_l791_791588

theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), α ≤ β ∧ β ≤ γ ∧ γ ≤ (90:ℝ) ∧ 
  ((β - α ≤ 15) ∧ (γ - β ≤ 15) ∧ (90 - γ ≤ 15)) ∧
  (α + β + γ = 180) ∧ 
  (α = 45 ∧ β = 60 ∧ γ = 75) := sorry

end most_irregular_acute_triangle_l791_791588


namespace find_cost_prices_l791_791768

noncomputable theory

-- Definitions of the variables
variables (S M L : ℝ)
-- Conditions based on the problem statement
axiom condition1 : 3.6 * L = 1800
axiom condition2 : 5 * S + 4 * M + 3.6 * L = 3600
axiom condition3 : (5 * S + 4 * M + 3 * L) - 3 * S + 2 * M = 3600

-- Prove the cost prices
theorem find_cost_prices (S M L : ℝ) (h1 : 3.6 * L = 1800) (h2 : 5 * S + 4 * M + 3.6 * L = 3600) 
(h3 : (5 * S + 4 * M + 3 * L) - 3 * S + 2 * M = 3600) : 
  S = 109.09 ∧ M = 313.64 ∧ L = 500 := by
  sorry

end find_cost_prices_l791_791768


namespace tricycles_in_parade_l791_791020

theorem tricycles_in_parade :
  ∃ (b t u : ℕ), b + t + u = 10 ∧ 2 * b + 3 * t + u = 26 ∧ t = 6 :=
by
  existsi 4, 6, 0
  simp [*, add_comm, add_left_comm, two_mul]
  exact ⟨rfl, rfl⟩

end tricycles_in_parade_l791_791020


namespace cosine_periodic_even_is_valid_l791_791169

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def cosine_periodic_even : ℝ → ℝ := λ x, Real.cos (2 * x)

theorem cosine_periodic_even_is_valid :
  is_even cosine_periodic_even ∧ is_periodic cosine_periodic_even Real.pi :=
by
  split
  -- is_even proof omitted
  sorry
  -- is_periodic proof omitted
  sorry

end cosine_periodic_even_is_valid_l791_791169


namespace coeff_x6_in_expansion_l791_791436

theorem coeff_x6_in_expansion : 
  let a := (x : ℚ)^(2 : ℚ)
  let b := -(1/x : ℚ)
  let n := 6
  (binom 6 2 : ℚ) * a^(6-2) * b^2 = (15 : ℚ) :=
by {
  sorry
}

end coeff_x6_in_expansion_l791_791436


namespace painted_cubes_at_least_two_faces_l791_791897

/--
A four-inch wooden cube is painted blue on all six faces and then cut into one-inch cubes.
Prove that the number of one-inch cubes that have blue paint on at least two faces is 32.
-/
theorem painted_cubes_at_least_two_faces : 
  let total_cubes := 4 * 4 * 4 in
  let corner_cubes := 8 in
  let edge_cubes_per_edge := 4 - 2 in
  let edges_count := 12 in
  let two_faces_cubes := edge_cubes_per_edge * edges_count in
  corner_cubes + two_faces_cubes = 32 :=
by 
  -- Define the variables
  let total_cubes := 4 * 4 * 4
  let corner_cubes := 8
  let edge_cubes_per_edge := 4 - 2
  let edges_count := 12
  let two_faces_cubes := edge_cubes_per_edge * edges_count
  -- Prove the statement
  have h1 : total_cubes = 64 := by linarith
  have h2 : corner_cubes + two_faces_cubes = 8 + 24 := by linarith
  have h3 : 8 + 24 = 32 := by linarith
  show corner_cubes + two_faces_cubes = 32 from h3
  sorry

end painted_cubes_at_least_two_faces_l791_791897


namespace max_sheets_one_participant_l791_791357

theorem max_sheets_one_participant
  (n : ℕ) (avg_sheets : ℕ) (h1 : n = 40) (h2 : avg_sheets = 7) 
  (h3 : ∀ i : ℕ, i < n → 1 ≤ 1) : 
  ∃ max_sheets : ℕ, max_sheets = 241 :=
by
  sorry

end max_sheets_one_participant_l791_791357


namespace probability_triangle_PBC_l791_791024

theorem probability_triangle_PBC (
  {ABC : Triangle} (isosceles_right : is_isosceles_right_triangle ABC)
  (side_eq_10 : ABC.side_lengths = (10, 10, 10 * sqrt 2))
  (P : Point) (P_in_ABC : P ∈ ABC)):
  probability (area (triangle P ABC.BC) < (1 / 3) * area ABC) = 2222 / 5000 :=
sorry

end probability_triangle_PBC_l791_791024


namespace all_divisible_by_2013_l791_791877

theorem all_divisible_by_2013 (a : Fin 100 → Fin 100 → ℕ)
    (h1 : ∀ i j, 1 ≤ a i j ∧ a i j ≤ 9) 
    (R C : Fin 100 → ℕ)
    (hR : ∀ i, R i = ∑ j, 10^j * a i j)
    (hC : ∀ j, C j = ∑ i, 10^i * a i j)
    (hdiv : (∑ i, if R i % 2013 = 0 then 1 else 0) + (∑ j, if C j % 2013 = 0 then 1 else 0) ≥ 199) :
    (∀ i, R i % 2013 = 0) ∧ (∀ j, C j % 2013 = 0) :=
by
  sorry

end all_divisible_by_2013_l791_791877


namespace min_distance_sum_l791_791725

open Real

structure Point where
  x : ℝ
  y : ℝ

def dist (A B : Point) : ℝ := sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def PA := dist (Point.mk -3 0)
def PB := dist (Point.mk 1 -1)
def PC := dist (Point.mk 0 3)
def PD := dist (Point.mk -1 3)

theorem min_distance_sum (P : Point) : PA P + PB P + PC P + PD P = 3 * sqrt 2 + 2 * sqrt 5 :=
  sorry

end min_distance_sum_l791_791725


namespace large_block_volume_l791_791199

theorem large_block_volume (W D L : ℝ) (h : W * D * L = 4) :
    (2 * W) * (2 * D) * (2 * L) = 32 :=
by
  sorry

end large_block_volume_l791_791199


namespace sum_digits_94_eights_times_94_sevens_l791_791984

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_digits_94_eights_times_94_sevens :
  sum_of_digits (94 * 8) * sum_of_digits (94 * 7) = 1034 :=
sorry

end sum_digits_94_eights_times_94_sevens_l791_791984


namespace infinite_non_congruent_right_triangles_l791_791128

noncomputable def right_triangle_equal_perimeter_area : Prop :=
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 = c^2) ∧ 
  (a + b + c = (1/2) * a * b)

theorem infinite_non_congruent_right_triangles :
  ∃ (k : ℕ), right_triangle_equal_perimeter_area :=
sorry

end infinite_non_congruent_right_triangles_l791_791128


namespace product_of_roots_cubic_eq_l791_791580

theorem product_of_roots_cubic_eq (x : ℝ) :
  (∀ x, 2 * x ^ 3 - 5 * x ^ 2 + 15 * x - 35 = 0) →
  (∃ a b c : ℝ, x = a * b * c ∧ a * b * c = 35 / 2) :=
begin
  sorry
end

end product_of_roots_cubic_eq_l791_791580


namespace sum_of_inverses_A_B_C_eq_300_l791_791753

theorem sum_of_inverses_A_B_C_eq_300 
  (p q r : ℝ)
  (hroots : ∀ x, (x^3 - 30*x^2 + 105*x - 114 = 0) → (x = p ∨ x = q ∨ x = r))
  (A B C : ℝ)
  (hdecomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    (1 / (s^3 - 30*s^2 + 105*s - 114) = A/(s - p) + B/(s - q) + C/(s - r))) :
  (1 / A) + (1 / B) + (1 / C) = 300 :=
sorry

end sum_of_inverses_A_B_C_eq_300_l791_791753


namespace vector_dot_product_l791_791275

theorem vector_dot_product
  (AB : ℝ × ℝ) (BC : ℝ × ℝ)
  (t : ℝ)
  (hAB : AB = (2, 3))
  (hBC : BC = (3, t))
  (ht : t > 0)
  (hmagnitude : (3^2 + t^2).sqrt = (10:ℝ).sqrt) :
  (AB.1 * (AB.1 + BC.1) + AB.2 * (AB.2 + BC.2) = 22) :=
by
  sorry

end vector_dot_product_l791_791275


namespace total_eggs_found_l791_791004

def eggs_from_club_house : ℕ := 40
def eggs_from_park : ℕ := 25
def eggs_from_town_hall : ℕ := 15

theorem total_eggs_found : eggs_from_club_house + eggs_from_park + eggs_from_town_hall = 80 := by
  -- Proof of this theorem
  sorry

end total_eggs_found_l791_791004


namespace probability_sqrt_less_than_20_of_three_digit_l791_791494

def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

def valid_number (n : ℕ) := n < 400

def valid_three_digit_numbers :=
  {n : ℕ | n ∈ three_digit_numbers ∧ valid_number n}

def count_valid_numbers : ℕ := (valid_three_digit_numbers.to_finset).card

def count_three_digit_numbers : ℕ := (three_digit_numbers.to_finset).card

theorem probability_sqrt_less_than_20_of_three_digit :
  count_valid_numbers.to_real / count_three_digit_numbers.to_real = 1 / 3 :=
sorry

end probability_sqrt_less_than_20_of_three_digit_l791_791494


namespace spotlights_problem_l791_791117

def fully_illuminated (n : ℕ) : Prop :=
∀ (S : finset ℕ), 
(S.card = n ∧ ∀ i ∈ S, convex ((i : convex_shape)) ∧ 
    (∀ i ∈ S, S.erase i → fully_illuminated) ∧
    (∀ i j ∈ S, (S.erase i).erase j → ¬ fully_illuminated))
  
theorem spotlights_problem (n : ℕ) : fully_illuminated n ↔ n ≥ 2 := by
  sorry

end spotlights_problem_l791_791117


namespace total_playtime_l791_791743

noncomputable def lena_playtime_minutes : ℕ := 210
noncomputable def brother_playtime_minutes (lena_playtime: ℕ) : ℕ := lena_playtime + 17
noncomputable def sister_playtime_minutes (brother_playtime: ℕ) : ℕ := 2 * brother_playtime

theorem total_playtime
  (lena_playtime : ℕ)
  (brother_playtime : ℕ)
  (sister_playtime : ℕ)
  (h_lena : lena_playtime = lena_playtime_minutes)
  (h_brother : brother_playtime = brother_playtime_minutes lena_playtime)
  (h_sister : sister_playtime = sister_playtime_minutes brother_playtime) :
  lena_playtime + brother_playtime + sister_playtime = 891 := 
  by sorry

end total_playtime_l791_791743


namespace artifacts_per_wing_is_34_l791_791551

-- Define the basic parameters and conditions
def total_wings : ℕ := 12
def painting_wings : ℕ := 4
def paintings_per_smaller_wing : ℕ := 15
def large_paintings : ℕ := 1
def smaller_painting_wings : ℕ := 3
def artifacts_multiplier : ℕ := 6

-- Calculate the number of paintings
def total_paintings : ℕ := large_paintings + (smaller_painting_wings * paintings_per_smaller_wing)

-- Calculate the number of artifacts
def total_artifacts : ℕ := artifacts_multiplier * total_paintings

-- Calculate the number of artifact wings
def artifact_wings : ℕ := total_wings - painting_wings

-- Calculate the artifacts per artifact wing
def artifacts_per_wing : ℕ := total_artifacts / artifact_wings

-- Prove that each artifact wing has 34 artifacts
theorem artifacts_per_wing_is_34 (h1 : total_wings = 12) 
                                (h2 : painting_wings = 4)
                                (h3 : paintings_per_smaller_wing = 15)
                                (h4 : large_paintings = 1)
                                (h5 : smaller_painting_wings = 3)
                                (h6 : artifacts_multiplier = 6)
                                : artifacts_per_wing = 34 :=
by
  -- Assertions derived from given conditions
  have total_paintings_eq : total_paintings = 46 :=
    by rw [total_paintings, large_paintings, smaller_painting_wings, paintings_per_smaller_wing]; simp,
  have total_artifacts_eq : total_artifacts = 276 :=
    by rw [total_artifacts, artifacts_multiplier, total_paintings_eq]; simp,
  have artifact_wings_eq : artifact_wings = 8 :=
    by rw [artifact_wings, total_wings, painting_wings]; simp,
  have artifacts_per_wing_eq : artifacts_per_wing = (total_artifacts_eq / artifact_wings_eq) :=
    by rw [artifacts_per_wing, total_artifacts_eq, artifact_wings_eq]; simp,
  exact artifacts_per_wing_eq

end artifacts_per_wing_is_34_l791_791551


namespace min_sum_matrix_l791_791052

noncomputable def sum_matrix_elements {n : ℕ} (A : matrix (fin n) (fin n) ℕ) : ℕ :=
  ∑ i j, A i j

theorem min_sum_matrix (n : ℕ) (A : matrix (fin n) (fin n) ℕ) 
  (h : ∀ i j, A i j = 0 → (∑ k, A i k) + (∑ k, A k j) ≥ n) :
  sum_matrix_elements A ≥ n * n / 2 :=
by
  sorry

end min_sum_matrix_l791_791052


namespace ratio_hexagon_octagon_l791_791933

noncomputable def ratio_of_areas (s : ℝ) :=
  let A1 := s / (2 * Real.tan (Real.pi / 6))
  let H1 := s / (2 * Real.sin (Real.pi / 6))
  let area1 := Real.pi * (H1^2 - A1^2)
  let A2 := s / (2 * Real.tan (Real.pi / 8))
  let H2 := s / (2 * Real.sin (Real.pi / 8))
  let area2 := Real.pi * (H2^2 - A2^2)
  area1 / area2

theorem ratio_hexagon_octagon (s : ℝ) (h : s = 3) : ratio_of_areas s = 49 / 25 :=
  sorry

end ratio_hexagon_octagon_l791_791933


namespace problem1_problem2_l791_791880

-- (1) Prove that the expression simplifies correctly
theorem problem1 : 
  1.5 ^ (-1/3) * (-7/6) ^ 0 + 8 ^ 0.25 * 42 + (32 * real.sqrt 3) ^ 6 - real.sqrt ((-2/3) ^ (2/3)) = 
  2 + (32 * real.sqrt 3) ^ 6 := 
by sorry

-- (2) Prove with given condition
theorem problem2 (x : ℝ) (hx : x^0.5 + x^( -0.5) = 3) :
  (x + x^(-1) + 2) / (x^2 + x^(-2) - 2) = 1 / 5 :=
by sorry

end problem1_problem2_l791_791880


namespace trigonometric_identity_l791_791513

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
sorry

end trigonometric_identity_l791_791513


namespace concyclic_points_l791_791054

theorem concyclic_points (A B C D A₁ B₁ A₂ B₂ : Point)
  (h1 : Trapezoid A B C D)
  (h2 : ∃ Ω : Circle, passes_through Ω A ∧ passes_through Ω B ∧ passes_through Ω C ∧ passes_through Ω D)
  (h3 : ∃ ω : Circle, passes_through ω C ∧ passes_through ω D ∧
        intersects ω (line_through C A) A₁ ∧ intersects ω (line_through C B) B₁)
  (h4 : symmetric A₁ (midpoint (line_through C A)) A₂)
  (h5 : symmetric B₁ (midpoint (line_through C B)) B₂) :
  concyclic A B A₂ B₂ :=
sorry

end concyclic_points_l791_791054


namespace percentage_increase_is_60_l791_791538

-- Defining the initial investment
variable (P : ℝ)

-- Definitions based on conditions
def first_year_increase (x : ℝ) := P * (1 + x / 100)
def second_year_value (x : ℝ) := first_year_increase P x * 0.7
def final_value := P * 1.12

-- Theorem statement
theorem percentage_increase_is_60 :
  ∃ x : ℝ, second_year_value P x = final_value → x = 60 := 
sorry

end percentage_increase_is_60_l791_791538


namespace train_speed_proof_l791_791844

theorem train_speed_proof : 
  ∀ (V_A V_B : ℝ) (T_A T_B : ℝ), 
  T_A = 9 ∧ 
  T_B = 4 ∧ 
  V_B = 90 ∧ 
  (V_A / V_B = T_B / T_A) → 
  V_A = 40 := 
by
  intros V_A V_B T_A T_B h
  obtain ⟨hT_A, hT_B, hV_B, hprop⟩ := h
  sorry

end train_speed_proof_l791_791844


namespace Theorem_3_2_14_l791_791780

open Set

variable {E : Type*} [PartialOrder E]

def maximal_chain_cardinality (E : Set E) : ℕ := sorry

def maximal_antichain_cardinality (E : Set E) : ℕ := sorry

def N (E : Set E) (x : E) : ℕ := sorry

def N_inv (E : Set E) (i : ℕ) : Set E := { x | N E x = i } 

theorem Theorem_3_2_14 
  (E : Set E) 
  (h : ℕ) 
  (w : ℕ) 
  (h_max_chain : h = maximal_chain_cardinality E) 
  (remark_3_2_15 : |E| ≤ h * w) 
  (mirsky : ∀ partition, partition_antichains E partition → |partition| ≥ h) 
  (N_def : ∀ x, N E x = maximal_chain_cardinality { e | e ≤ x }) 
  (partition_def : ∀ i, 1 ≤ i → i ≤ h → N_inv E i ⊆ E) :
  (∃ partition, partition_antichains E partition ∧ partition = (N_inv E '' {i | 1 ≤ i ∧ i ≤ h})) :=
sorry

end Theorem_3_2_14_l791_791780


namespace triangle_sufficient_and_necessary_condition_l791_791035

noncomputable def triangle_suff_necc_cosine {A B C : Type} [linear_ordered_field A] [linear_ordered_field B] (a b : A) (angle_A angle_B : B) :=
  a < b ↔ cos angle_A > cos angle_B

-- now give the actual statement
theorem triangle_sufficient_and_necessary_condition {a b : ℝ} {A B : ℝ} (h1 : a < b) (h2: 0 < A) (h3: A < B) (h4: B < real.pi) :
  (a < b ↔ (real.cos A > real.cos B)) :=
sorry

end triangle_sufficient_and_necessary_condition_l791_791035


namespace problem1_problem2_l791_791658

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l791_791658


namespace identifyMinorPremise_l791_791586

variable (A B C D : Type) -- Representing the quadrilateral

-- Definitions of the premises
def isRhombus (x : A) : Prop := sorry
def isParallelogram (x : B) : Prop := sorry

-- Conditions
axiom rhombusImpliesParallelogram :
  ∀ {x : A}, isRhombus x → isParallelogram x

axiom ABCDisRhombus :
  isRhombus A

-- The goal is to prove the minor premise
theorem identifyMinorPremise :
  ABCDisRhombus = isRhombus A :=
by sorry

end identifyMinorPremise_l791_791586


namespace express_as_fraction_l791_791601

theorem express_as_fraction (h : 0.008{repeating} = 1 / 125) : 1.024{repeating} = 128 / 125 :=
by
  sorry

end express_as_fraction_l791_791601


namespace mark_total_collected_l791_791078

def total_households_visited := 60 * 7

def donation (n: ℤ) (pct: ℤ) := n * pct / 100

def usd_from_eur (eur: ℤ) := eur * 11 / 10

def usd_from_gbp (gbp: ℤ) := gbp * 13 / 10

def total_donations (n: ℤ) :=
  donation n 25 * 20 +
  usd_from_eur (donation n 15 * 15) +
  usd_from_gbp (donation n 10 * 10) +
  donation n 5 * (20 + 10 * 11 / 10)

theorem mark_total_collected :
  total_donations total_households_visited = 4336.5 := sorry

end mark_total_collected_l791_791078


namespace altitude_inequality_l791_791872

noncomputable def altitude_length (P Q R : Point) : ℝ :=
if collinear P Q R then 0 else shortest_altitude (triangle P Q R)

theorem altitude_inequality (A B C X : Point) :
  altitude_length A B C ≤ altitude_length A B X + altitude_length A X C + altitude_length X B C :=
sorry

end altitude_inequality_l791_791872


namespace number_of_schools_l791_791030

theorem number_of_schools 
  (unique_scores : ∀ (a b : ℕ), a ≠ b → ∀ (scores : ℕ → ℕ), scores a ≠ scores b)
  (highest_score_of_team : ∀ (team : ℕ → ℕ) (david_index : ℕ) (david : ℕ), 
                            team david_index = david → david = max (team 0) (max (team 1) (max (team 2) (team 3))))
  (david_median : ∀ (participants : ℕ) (scores : ℕ → ℕ), 
                   (∃ (david_index : nat), scores (2 * participants) / (2 * participants + 1) = scores david_index) ∧
                   (∀ (scores' : ℕ → ℕ), scores = scores') → 
                   (2 * participants + 1 = 29 ∧ ∃ (david_rank : ℕ), david_rank < 29))
  (eric_position : ∀ (position : ℕ), position = 29 )
  (fiona_position : ∀ (position : ℕ), position = 82 )
  : ∃ (n : ℕ), n = 21 := sorry

end number_of_schools_l791_791030


namespace new_altitude_l791_791795

-- We first define the initial conditions

def initial_area : ℝ := 800
def initial_base : ℕ := 40

-- Assuming the area formula for a triangle A = 1/2 * b * h
def area_formula (b h : ℝ) : ℝ := 1/2 * b * h

-- New base after extension
def new_base : ℝ := 50

-- The proof statement we need to prove that the new altitude is 32
theorem new_altitude : area_formula new_base 32 = initial_area :=
by
  -- We would include our proof here
  sorry

end new_altitude_l791_791795


namespace symmetric_points_add_l791_791295

theorem symmetric_points_add (a b : ℝ) : 
  (P : ℝ × ℝ) → (Q : ℝ × ℝ) →
  P = (a-1, 5) →
  Q = (2, b-1) →
  (P.fst = Q.fst) →
  P.snd = -Q.snd →
  a + b = -1 :=
by
  sorry

end symmetric_points_add_l791_791295


namespace at_least_one_div_by_5_l791_791093

-- Define natural numbers and divisibility by 5
def is_div_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- Proposition: If a, b are natural numbers and ab is divisible by 5, then at least one of a or b must be divisible by 5.
theorem at_least_one_div_by_5 (a b : ℕ) (h_ab : is_div_by_5 (a * b)) : is_div_by_5 a ∨ is_div_by_5 b :=
  by
    sorry

end at_least_one_div_by_5_l791_791093


namespace find_f2_l791_791800

noncomputable def f : ℝ → ℝ := sorry

-- Given condition
axiom f_condition (x : ℝ) (hx : x ≠ 0) : f(x) - 2 * f(1 / x) = 4^x

-- Prove f(2) = -20 / 3
theorem find_f2 : f(2) = -20 / 3 :=
by
  sorry

end find_f2_l791_791800


namespace combine_radicals_l791_791348

theorem combine_radicals (x : ℝ) (h : sqrt (2 + x) = sqrt (5 - 2x)) : x = 1 :=
sorry

end combine_radicals_l791_791348


namespace median_is_82_l791_791729

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def total_elements : ℕ :=
  2 * sum_of_squares 100

noncomputable def median_value : ℕ :=
  82

theorem median_is_82 :
  let seq := List.bind (List.range 100) (λ n, List.replicate (2 * (n + 1)) (n + 1))
  in (Let total_elements := seq.length) →
      (total_elements == (2 * sum_of_squares 100)) →
      (seq.nth (total_elements / 2)) = some median_value :=
by
  sorry

end median_is_82_l791_791729


namespace identity_proof_l791_791092

theorem identity_proof (a b c x y z : ℝ) : 
  (a * x + b * y + c * z) ^ 2 + (b * x + c * y + a * z) ^ 2 + (c * x + a * y + b * z) ^ 2 = 
  (c * x + b * y + a * z) ^ 2 + (b * x + a * y + c * z) ^ 2 + (a * x + c * y + b * z) ^ 2 := 
by
  sorry

end identity_proof_l791_791092


namespace rearrangement_inequality_l791_791379

open List

theorem rearrangement_inequality (n : ℕ) (a b : List ℕ) (ha : a.Perm (range 1 (n+1))) (hb : b.Perm (range 1 (n+1))) :
  (∑ i in range 1 (n+1), i * ((n+1) - i)) ≤ (∑ i in (range 1 (n+1)), a.nthLe (i - 1) ((i - 1).lt_of_lt_add one_lt_succ_self) * b.nthLe (i - 1) ((i - 1).lt_of_lt_add one_lt_succ_self)) ∧ 
  (∑ i in (range 1 (n+1)), a.nthLe (i - 1) ((i - 1).lt_of_lt_add one_lt_succ_self) * b.nthLe (i - 1) ((i - 1).lt_of_lt_add one_lt_succ_self)) ≤ (∑ i in (range 1 (n+1)), i * i) :=
by
  sorry

end rearrangement_inequality_l791_791379


namespace verify_inclination_angles_l791_791666

noncomputable def inclination_angle_of_focal_chord (p : ℝ) (θ : ℝ) : Prop :=
  (sqrt(2 * p) * sqrt(2 * p) * sin(θ) * cos(θ) = 8 * p * (sin(θ))/(sin(θ))) → θ = π / 6 ∨ θ = 5 * π / 6

theorem verify_inclination_angles (p : ℝ) (hp : 0 < p):
  inclination_angle_of_focal_chord p (π / 6) ∨ inclination_angle_of_focal_chord p (5 * π / 6) :=
sorry

end verify_inclination_angles_l791_791666


namespace sequence_conditions_l791_791130

namespace SequenceProblem

open Real

-- Define the sequence (x_n)
def x_sequence (x1 : ℝ) : ℕ → ℝ 
| 0 => x1
| n + 1 => (λ x, (-x + sqrt (3 - 3 * x^2)) / 2) (x_sequence n)

-- Define periodicity and positivity conditions
noncomputable def periodic (x1 : ℝ) := ∀ n : ℕ, x_sequence x1 (n + 2) = x_sequence x1 n
noncomputable def positive (x1 : ℝ) := ∀ n : ℕ, x_sequence x1 n > 0

-- Final Lean statement for the given mathematical problem
theorem sequence_conditions (x1 : ℝ) (h : x1 ∈ Ioc 0 (sqrt 3 / 2)) : positive x1 ∧ periodic x1 :=
sorry

end SequenceProblem

end sequence_conditions_l791_791130


namespace isosceles_right_triangle_leg_hypotenuse_ratio_l791_791581

theorem isosceles_right_triangle_leg_hypotenuse_ratio (a d k : ℝ) 
  (h_iso : d = a * Real.sqrt 2)
  (h_ratio : k = a / d) : 
  k^2 = 1 / 2 := by sorry

end isosceles_right_triangle_leg_hypotenuse_ratio_l791_791581


namespace determine_angle_FAG_l791_791919

-- Definitions of points and geometric shapes for the proof
variables (A B C F G : Type)

-- Conditions based on the problem statement
constant is_equilateral_triangle : ∀ (A B C : Type), Prop
constant angle_BAC_eq_60 : ∀ (A B C : Type), Prop
constant shares_side_BC_with_square : ∀ (A B C F G : Type), Prop
constant angle_BCF_eq_90 : ∀ (B C F : Type), Prop
constant square_interior_angles_eq_90 : ∀ (B C F G : Type), Prop

-- Define the hypothesis
def hyp := is_equilateral_triangle A B C ∧
            angle_BAC_eq_60 A B C ∧
            shares_side_BC_with_square A B C F G ∧
            angle_BCF_eq_90 B C F ∧
            square_interior_angles_eq_90 B C F G

-- Define the theorem statement based on the problem
theorem determine_angle_FAG (A B C F G : Type) 
  (h : hyp) : 
  ∃ (angle_FAG : ℝ), angle_FAG = 135 :=
by
  sorry

end determine_angle_FAG_l791_791919


namespace sum_ratios_eq_six_l791_791391

noncomputable def omega : ℂ :=
Complex.exp (Complex.I * Real.pi / 2)  -- One possible nonreal root of z^4 = 1

theorem sum_ratios_eq_six (b : ℕ → ℝ) (n : ℕ) (h1 : ∑ k in Finset.range n, (1 / (b k + omega)) = 3 + 4 * Complex.I) :
  ∑ k in Finset.range n, (2 * b k - 1) / (b k ^ 2 - b k + 1) = 6 :=
by
  sorry

end sum_ratios_eq_six_l791_791391


namespace range_of_a_l791_791318

namespace ProofProblem

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
if x >= 0 then x^3 else -x^3

-- State the main theorem to be proved
theorem range_of_a (a : ℝ) : f (3 * a - 1) - 8 * f a ≥ 0 → a ≤ 1/5 ∨ a ≥ 1 := by
  sorry

end ProofProblem

end range_of_a_l791_791318


namespace suzanna_distance_ridden_l791_791789

theorem suzanna_distance_ridden (rate_per_5minutes : ℝ) (time_minutes : ℕ) (total_distance : ℝ) (units_per_interval : ℕ) (interval_distance : ℝ) :
  rate_per_5minutes = 0.75 → time_minutes = 45 → units_per_interval = 5 → interval_distance = 0.75 → total_distance = (time_minutes / units_per_interval) * interval_distance → total_distance = 6.75 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end suzanna_distance_ridden_l791_791789


namespace patanjali_total_distance_l791_791087

-- defining the parameters based on the problem's conditions
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day1_hours : ℕ := day1_distance / day1_speed

def day2_hours : ℕ := day1_hours - 1
def day2_speed : ℕ := day1_speed + 1
def day2_distance : ℕ := day2_hours * day2_speed

def day3_hours : ℕ := day1_hours
def day3_speed : ℕ := day2_speed
def day3_distance : ℕ := day3_hours * day3_speed

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

-- Theorem statement
theorem patanjali_total_distance : total_distance = 62 :=
by
  calc
  total_distance 
    = day1_distance + day2_distance + day3_distance : by rfl
    ... = 18 + 20 + 24 : by rfl
    ... = 62 : by rfl

end patanjali_total_distance_l791_791087


namespace triangle_count_positive_area_and_distinct_l791_791952

-- Triangle problem statement
theorem triangle_count_positive_area_and_distinct (O P Q : ℕ × ℕ)
  (hP : 39 * P.1 + P.2 = 1953)
  (hQ : 39 * Q.1 + Q.2 = 1953)
  (h_distinct : P ≠ Q):
  let valid_x_values := {x : ℕ | 0 ≤ x ∧ x ≤ 50}
  let even_x_values := {x ∈ valid_x_values | x % 2 = 0}
  let odd_x_values := {x ∈ valid_x_values | x % 2 = 1}
  ({x : ℕ | x % 2 = 0}.card.choose 2 + {x : ℕ | x % 2 = 1}.card.choose 2) = 625 := 
begin
  -- Some formalized definitions and operations might be implicit here.
  sorry
end

end triangle_count_positive_area_and_distinct_l791_791952


namespace mod_multiplication_result_l791_791101

theorem mod_multiplication_result :
  ∃ n : ℕ, 507 * 873 ≡ n [MOD 77] ∧ 0 ≤ n ∧ n < 77 ∧ n = 15 := by
  sorry

end mod_multiplication_result_l791_791101


namespace time_increases_with_water_speed_increase_l791_791893

variable (S : ℝ) -- Total distance
variable (V : ℝ) -- Speed of the ferry in still water
variable (V1 V2 : ℝ) -- Speed of the water flow before and after increase

-- Ensure realistic conditions
axiom V_pos : 0 < V
axiom V1_pos : 0 < V1
axiom V2_pos : 0 < V2
axiom V1_less_V : V1 < V
axiom V2_less_V : V2 < V
axiom V1_less_V2 : V1 < V2

theorem time_increases_with_water_speed_increase :
  (S / (V + V1) + S / (V - V1)) < (S / (V + V2) + S / (V - V2)) :=
sorry

end time_increases_with_water_speed_increase_l791_791893


namespace arithmetic_sequence_30th_term_l791_791953

theorem arithmetic_sequence_30th_term :
  let a₁ := 4
  let d₁ := 6
  let n := 30
  (a₁ + (n - 1) * d₁) = 178 :=
by
  sorry

end arithmetic_sequence_30th_term_l791_791953


namespace proof_of_problem_l791_791705

noncomputable def problem_statement (a c : ℝ) (h : a > c) (dot_product : a * c * (1/3) = 2) (b : ℝ) (hb : b = 3) : Prop :=
  a = 3 ∧ c = 2 ∧ cos (real.arccos (1/3) - real.arccos ((2 * real.sqrt 2) / 3)) = 23 / 27

theorem proof_of_problem : problem_statement 3 2 (by linarith) (by linarith) (by linarith) :=
sorry

end proof_of_problem_l791_791705


namespace interval_intersection_l791_791251

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l791_791251


namespace geom_series_ratio_l791_791388

noncomputable def geomSeries (a q : ℝ) (n : ℕ) : ℝ :=
a * ((1 - q ^ n) / (1 - q))

theorem geom_series_ratio (a1 q : ℝ) (h : 8 * a1 * q + a1 * q^4 = 0) :
  (geomSeries a1 q 5) / (geomSeries a1 q 2) = -11 :=
sorry

end geom_series_ratio_l791_791388


namespace quadratic_inequality_solution_range_l791_791878

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end quadratic_inequality_solution_range_l791_791878


namespace prob_of_caps_given_sunglasses_l791_791140

theorem prob_of_caps_given_sunglasses (n_sunglasses n_caps n_both : ℕ) (P_sunglasses_given_caps : ℚ) 
  (h_nsunglasses : n_sunglasses = 80) (h_ncaps : n_caps = 45)
  (h_Psunglasses_given_caps : P_sunglasses_given_caps = 3/8)
  (h_nboth : n_both = P_sunglasses_given_caps * n_sunglasses) :
  (n_both / n_caps) = 2/3 := 
by
  sorry

end prob_of_caps_given_sunglasses_l791_791140


namespace measure_of_angle_F_l791_791373

theorem measure_of_angle_F (D E F : ℝ) (h₁ : D = 85) (h₂ : E = 4 * F + 15) (h₃ : D + E + F = 180) : 
  F = 16 :=
by
  sorry

end measure_of_angle_F_l791_791373


namespace min_val_of_expression_l791_791613

noncomputable def min_val_expr : ℝ → ℝ :=
  λ x, 
  (((Real.sin x + Real.csc x) ^ 2 + (Real.cos x + Real.sec x) ^ 2) * Real.exp (Real.sin x + Real.cos x))

theorem min_val_of_expression : (∀ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.sin x + Real.cos x > 1 → min_val_expr x ≥ 9 * Real.exp (Real.sqrt 2)) ∧
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.sin x + Real.cos x > 1 ∧ min_val_expr x = 9 * Real.exp (Real.sqrt 2)) :=
begin
  sorry
end

end min_val_of_expression_l791_791613


namespace isosceles_triangle_angle_A_l791_791025

theorem isosceles_triangle_angle_A {A B C : Type} [linear_ordered_field C]
  (triangle : Triangle A B C) (isosceles : AB = AC) (angle_70 : ∃ (angle : A), triangle.interior_angles angle = 70) :
  ∃ (angleA : A), angleA = 70 ∨ angleA = 40 :=
by sorry

end isosceles_triangle_angle_A_l791_791025


namespace sum_product_inequality_l791_791474

variable {α : Type*} [LinearOrderedField α]

theorem sum_product_inequality 
  (n : ℕ)
  (a b : Fin n.succ → α)
  (hA : ∀ i j, i < j → a i > a j)
  (hB : ∀ i j, i < j → b i > b j) :
  (∑ i in Finset.range n.succ, a i * b i) >
  (∑ i in Finset.range n.succ, a i * b (n - i)) := 
by sorry

end sum_product_inequality_l791_791474


namespace max_value_of_function_l791_791612

theorem max_value_of_function :
  ∃ x ∈ set.Ioo 0 Real.pi, 
  ∀ y, y = 2 * Real.sin (x / 2) * (1 - Real.sin (x / 2)) * (1 + Real.sin (x / 2))^2 →
  y ≤ (107 + 51 * Real.sqrt 17) / 256 :=
sorry

end max_value_of_function_l791_791612


namespace find_B_find_a_minus_c_l791_791360

-- Definitions of angles and sides
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Vectors m and n
def m := (2 * real.sin (A + C), real.sqrt 3)
def n := (real.cos (2 * B), 1 - 2 * (real.cos (B / 2)) ^ 2)

-- Assumptions
variable (is_acute_triangle : 0 < A ∧ A < real.pi / 2 ∧ 0 < B ∧ B < real.pi / 2 ∧ 0 < C ∧ C < real.pi / 2)
variable (parallel_vectors : m ∥ n)

-- Problems to prove
theorem find_B : B = real.pi / 3 :=
by sorry

variable (sinA_sinC_eq_sin2B : real.sin A * real.sin C = (real.sin B) ^ 2)

theorem find_a_minus_c : a - c = 0 :=
by sorry


end find_B_find_a_minus_c_l791_791360


namespace black_cows_ratio_l791_791420

-- Definitions based on the conditions
def total_cows := 18
def non_black_cows := 4
def black_cows := 5 + total_cows / 2

-- Main statement to prove:
theorem black_cows_ratio (h: black_cows = 14): (7:9) = (black_cows: total_cows) :=
by
  sorry

end black_cows_ratio_l791_791420


namespace max_apartment_size_is_600_l791_791923

-- Define the cost per square foot and Max's budget
def cost_per_square_foot : ℝ := 1.2
def max_budget : ℝ := 720

-- Define the largest apartment size that Max should consider
def largest_apartment_size (s : ℝ) : Prop :=
  cost_per_square_foot * s = max_budget

-- State the theorem that we need to prove
theorem max_apartment_size_is_600 : largest_apartment_size 600 :=
  sorry

end max_apartment_size_is_600_l791_791923


namespace final_solution_percentage_l791_791687

-- Defining the conditions and setting up the proof
variable (V : ℝ) (P : ℝ) (A : ℝ)

-- Conditions
def conditions : Prop :=
  (V = 40) ∧ (0.20 * (V + 40) + 0.70 * 60 = P * ((V + 40) + 60))

-- The statement that we need to prove
theorem final_solution_percentage : conditions V P A → P = 0.4143 :=
by
  -- Assumptions and setting up variables
  intro h
  have h1 := h.1      -- V = 40
  have h2 := h.2      -- 0.20 * (V + 40) + 0.70 * 60 = P * ((V + 40) + 60)
  sorry

end final_solution_percentage_l791_791687


namespace person_age_l791_791554

-- Define the conditions
def current_age : ℕ := 18

-- Define the equation based on the person's statement
def age_equation (A Y : ℕ) : Prop := 3 * (A + 3) - 3 * (A - Y) = A

-- Statement to be proven
theorem person_age (Y : ℕ) : 
  age_equation current_age Y → Y = 3 := 
by 
  sorry

end person_age_l791_791554


namespace average_speed_l791_791565

-- Define the conditions
def initial_reading : ℕ := 2552
def final_reading : ℕ := 2992
def day1_time : ℕ := 6
def day2_time : ℕ := 8

-- Theorem: Proving the average speed is 31 miles per hour.
theorem average_speed :
  final_reading - initial_reading = 440 ∧ day1_time + day2_time = 14 ∧ 
  (final_reading - initial_reading) / (day1_time + day2_time) = 31 :=
by
  sorry

end average_speed_l791_791565


namespace circle_center_coordinates_l791_791113

theorem circle_center_coordinates (h k r : ℝ) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 1 → (x - h)^2 + (y - k)^2 = r^2) →
  (h, k) = (2, -3) :=
by
  intro H
  sorry

end circle_center_coordinates_l791_791113


namespace find_X_d_minus_Y_d_l791_791787

def digits_in_base_d (X Y d : ℕ) : Prop :=
  2 * d * X + X + Y = d^2 + 8 * d + 2 

theorem find_X_d_minus_Y_d (d X Y : ℕ) (h1 : digits_in_base_d X Y d) (h2 : d > 8) : X - Y = d - 8 :=
by 
  sorry

end find_X_d_minus_Y_d_l791_791787


namespace possible_values_of_y_l791_791063

theorem possible_values_of_y (x : ℝ) (h : x^2 + 6 * (x / (x - 3))^2 = 60) :
  (let y := ((x - 3)^2 * (x + 4)) / (2 * x - 5) in y = 0 ∨ y = 10 ∨ y = 192) :=
sorry

end possible_values_of_y_l791_791063


namespace main_theorem_l791_791703

variables (A E F B D C K L M N S T U V : Type*)

-- Conditions
def Condition1 : Prop := ∃ (A E F B D : Type*), 
  B ∈ Fintype(AE) ∧ D ∈ Fintype(AF) 
def Condition2 : Prop := ∃ (C : Type*), 
  C = (ED ∩ FB)
def Condition3 : Prop := ∃ (K L M N : Type*), 
  K ∈ Fintype(AB) ∧ L ∈ Fintype(BC) ∧ M ∈ Fintype(CD) ∧ N ∈ Fintype(DA) 
  ∧ (AK / KB = AD / BC) ∧ (cyclic equivalents)
def Condition4 : Prop := (∃ S T : Type*, isIncircleTouch (triangle A E F) S T)
def Condition5 : Prop := (∃ U V : Type*, isIncircleTouch (triangle C E F) U V)

-- Main statement to prove
theorem main_theorem : 
  (Condition1 A E F B D ∧ Condition2 C ∧ Condition3 K L M N ∧ Condition4 S T ∧ Condition5 U V ∧ areConcyclic K L M N) 
  → areConcyclic S T U V :=
by {
  sorry
}

end main_theorem_l791_791703


namespace counterexample_disproves_proposition_l791_791217

theorem counterexample_disproves_proposition :
  ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 :=
by {
  use [-1, -2],
  simp,
  split,
  { linarith, },
  { norm_num, linarith, }
}

end counterexample_disproves_proposition_l791_791217


namespace not_divisible_by_q_plus_one_l791_791988

theorem not_divisible_by_q_plus_one (q : ℕ) (hq_odd : q % 2 = 1) (hq_gt_two : q > 2) :
  ¬ (q + 1) ∣ ((q + 1) ^ ((q - 1) / 2) + 2) :=
by
  sorry

end not_divisible_by_q_plus_one_l791_791988


namespace problem_f_2011_l791_791661

noncomputable def f : ℝ → ℝ := sorry

axiom H1 : ∀ x : ℝ, f(x + 4) = f(x) + 2 * f(2)
axiom H2 : ∀ x : ℝ, f(-(x - 1) + 1) = f(x - 1)
axiom H3 : f(1) = 2

theorem problem_f_2011 : f(2011) = 2 := 
by
  sorry

end problem_f_2011_l791_791661


namespace at_least_one_prime_of_15_pairwise_relatively_prime_less_than_1998_l791_791629

theorem at_least_one_prime_of_15_pairwise_relatively_prime_less_than_1998 
  {p : Fin 15 → ℕ} (h1 : ∀ i, p i < 1998) 
  (h2 : ∀ i j, i ≠ j → Nat.gcd (p i) (p j) = 1) : 
  ∃ i, Nat.Prime (p i) :=
begin
  sorry
end

end at_least_one_prime_of_15_pairwise_relatively_prime_less_than_1998_l791_791629


namespace range_of_m_l791_791640

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) → m ≤ 9 / 8 :=
by
  intro h
  -- We need to implement the proof here
  sorry

end range_of_m_l791_791640


namespace smallest_positive_period_of_f_l791_791982

-- Conditions
def f (x : ℝ) : ℝ := sqrt 2 * cos (π * x - π / 6)

-- Proof problem
theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 2 := by
  -- Proof goes here
  sorry

end smallest_positive_period_of_f_l791_791982


namespace integral_x_plus_one_integral_sqrt_plus_x_squared_l791_791532

-- Problem 1: Show that the definite integral from 1 to 2 of (x + 1) dx equals 5/2
theorem integral_x_plus_one : ∫ x in 1..2, (x + 1) = 5 / 2 := sorry

-- Problem 2: Show that the definite integral from -2 to 2 of (sqrt(4 - x^2) + x^2) dx equals 2π + 16/3
theorem integral_sqrt_plus_x_squared : ∫ x in -2..2, (Real.sqrt (4 - x^2) + x^2) = 2 * Real.pi + 16 / 3 := sorry

end integral_x_plus_one_integral_sqrt_plus_x_squared_l791_791532


namespace triangle_altitude_l791_791794

theorem triangle_altitude (A b : ℝ) (h : ℝ) 
  (hA : A = 750) 
  (hb : b = 50) 
  (area_formula : A = (1 / 2) * b * h) : 
  h = 30 :=
  sorry

end triangle_altitude_l791_791794


namespace t_50_mod_7_l791_791234

theorem t_50_mod_7 (T : ℕ → ℕ) (h₁ : T 1 = 9) (h₂ : ∀ n > 1, T n = 9 ^ (T (n - 1))) :
  T 50 % 7 = 4 :=
sorry

end t_50_mod_7_l791_791234


namespace cube_volume_l791_791085

theorem cube_volume (a : ℝ) (h : (a - 1) * (a - 1) * (a + 1) = a^3 - 7) : a^3 = 8 :=
  sorry

end cube_volume_l791_791085


namespace determine_length_AE_l791_791717

-- Defining the context as given conditions
variables (A B C D E : Type) 
variables (triangle_AED triangle_BEC : Type)
variables (AB_CD : ℝ) (AC_BD : ℝ) (AE : ℝ)
variables [is_convex_quadrilateral A B C D]
variables [AC_BD_intersect E A C B D]
variables [areas_equal triangle_AED triangle_BEC]

-- Main statement to prove in Lean 4
theorem determine_length_AE (h1: length A B = 10) 
                            (h2: length C D = 15) 
                            (h3: length A C = 15)
                            (h4: areas_equal triangle_AED triangle_BEC)
                            : AE = 6 := 
begin
  sorry -- Proof goes here
end

end determine_length_AE_l791_791717


namespace three_pipes_fill_time_l791_791911

noncomputable def pipe_filling_time : ℕ := 21
noncomputable def rate_A := 1 / pipe_filling_time.toRat
noncomputable def rate_B := 2 * rate_A
noncomputable def rate_C := 2 * rate_B
noncomputable def combined_rate := rate_A + rate_B + rate_C

theorem three_pipes_fill_time :
  3 = (1 / combined_rate) := by
  sorry

end three_pipes_fill_time_l791_791911


namespace area_of_triangle_ABC_l791_791573

open_locale classical

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 4 = 1

-- Define triangle vertices and centroid condition
def centroid_origin (A B C : ℝ × ℝ) : Prop :=
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) in
  G = (0, 0)

-- Define the condition that the vertices A, B, C are on the ellipse
def vertices_on_ellipse (A B C : ℝ × ℝ) : Prop :=
  on_ellipse A.1 A.2 ∧ on_ellipse B.1 B.2 ∧ on_ellipse C.1 C.2

-- Define the problem in Lean
theorem area_of_triangle_ABC (A B C : ℝ × ℝ)
  (h_coord: centroid_origin A B C)
  (h_ellipse: vertices_on_ellipse A B C) :
  ∃ (area : ℝ), area = 9 :=
begin
  -- Placeholder for the proof
  sorry
end

end area_of_triangle_ABC_l791_791573


namespace striped_shirts_more_than_shorts_l791_791354

theorem striped_shirts_more_than_shorts :
  ∀ (total_students striped_students checkered_students short_students : ℕ),
    total_students = 81 →
    striped_students = total_students * 2 / 3 →
    checkered_students = total_students - striped_students →
    short_students = checkered_students + 19 →
    striped_students - short_students = 8 :=
by
  intros total_students striped_students checkered_students short_students
  sorry

end striped_shirts_more_than_shorts_l791_791354


namespace smallest_value_at_x_eq_10_l791_791618

theorem smallest_value_at_x_eq_10 :
  let x := 10 in
  let a := 6 / x in
  let b := 6 / (x + 2) in
  let c := 6 / (x - 2) in
  let d := x^2 / 6 in
  let e := (x + 2)^2 / 6 in
  b < a ∧ b < c ∧ b < d ∧ b < e := by
  sorry

end smallest_value_at_x_eq_10_l791_791618


namespace interval_of_x_l791_791265

theorem interval_of_x (x : ℝ) : 
  (2 < 3 * x ∧ 3 * x < 3) ∧ 
  (2 < 4 * x ∧ 4 * x < 3) ↔ 
  x ∈ set.Ioo (2 / 3) (3 / 4) := 
sorry

end interval_of_x_l791_791265


namespace hyperbola_eq_of_focus_and_asymptote_l791_791663

-- Define the focus point of the hyperbola
def hyperbola_focus : ℝ × ℝ := (3, 0)

-- Define the asymptote of the hyperbola
def hyperbola_asymptote (x y : ℝ) : Prop := y = -real.sqrt 2 * x

-- Define the equation of the hyperbola we need to prove
def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 3) - (y^2 / 6) = 1

-- The theorem stating that the hyperbola has the given equation
theorem hyperbola_eq_of_focus_and_asymptote
  (focus : ℝ × ℝ)
  (asymptote : ∀ x y : ℝ, hyperbola_asymptote x y)
  (equation : ∀ x y : ℝ, hyperbola_equation x y) :
  focus = (3, 0) →
  (∀ x y : ℝ, asymptote x y → hyperbola_asymptote x y) →
  (∀ x y : ℝ, equation x y → hyperbola_equation x y) →
  ∀ x y : ℝ, hyperbola_equation x y :=
begin
  intros h_focus h_asymptote h_equation x y,
  sorry
end

end hyperbola_eq_of_focus_and_asymptote_l791_791663


namespace diagonal_length_15x25x15_l791_791211

def diagonal_length_of_rectangular_prism (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_length_15x25x15 :
  diagonal_length_of_rectangular_prism 15 25 15 = 5 * Real.sqrt 43 :=
by
  -- proof skipped
  sorry

end diagonal_length_15x25x15_l791_791211


namespace number_of_students_l791_791550

theorem number_of_students (n : ℕ) (h1 : n < 40) (h2 : n % 7 = 3) (h3 : n % 6 = 1) : n = 31 := 
by
  sorry

end number_of_students_l791_791550


namespace total_instruments_correct_l791_791939

def numberOfFlutesCharlie : ℕ := 1
def numberOfHornsCharlie : ℕ := 2
def numberOfHarpsCharlie : ℕ := 1
def numberOfDrumsCharlie : ℕ := 5

def numberOfFlutesCarli : ℕ := 3 * numberOfFlutesCharlie
def numberOfHornsCarli : ℕ := numberOfHornsCharlie / 2
def numberOfDrumsCarli : ℕ := 2 * numberOfDrumsCharlie
def numberOfHarpsCarli : ℕ := 0

def numberOfFlutesNick : ℕ := 2 * numberOfFlutesCarli - 1
def numberOfHornsNick : ℕ := numberOfHornsCharlie + numberOfHornsCarli
def numberOfDrumsNick : ℕ := 4 * numberOfDrumsCarli - 2
def numberOfHarpsNick : ℕ := 0

def numberOfFlutesDaisy : ℕ := numberOfFlutesNick * numberOfFlutesNick
def numberOfHornsDaisy : ℕ := (numberOfHornsNick - numberOfHornsCarli) / 2
def numberOfDrumsDaisy : ℕ := (numberOfDrumsCharlie + numberOfDrumsCarli + numberOfDrumsNick) / 3
def numberOfHarpsDaisy : ℕ := numberOfHarpsCharlie

def numberOfInstrumentsCharlie : ℕ := numberOfFlutesCharlie + numberOfHornsCharlie + numberOfHarpsCharlie + numberOfDrumsCharlie
def numberOfInstrumentsCarli : ℕ := numberOfFlutesCarli + numberOfHornsCarli + numberOfDrumsCarli
def numberOfInstrumentsNick : ℕ := numberOfFlutesNick + numberOfHornsNick + numberOfDrumsNick
def numberOfInstrumentsDaisy : ℕ := numberOfFlutesDaisy + numberOfHornsDaisy + numberOfHarpsDaisy + numberOfDrumsDaisy

def totalInstruments : ℕ := numberOfInstrumentsCharlie + numberOfInstrumentsCarli + numberOfInstrumentsNick + numberOfInstrumentsDaisy

theorem total_instruments_correct : totalInstruments = 113 := by
  sorry

end total_instruments_correct_l791_791939


namespace smallest_positive_period_of_f_max_min_of_f_on_interval_l791_791674

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x)^2 + Real.cos (2*x) - 1

theorem smallest_positive_period_of_f : 
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem max_min_of_f_on_interval :
  ∃ (max_val min_val : ℝ), max_val = Real.sqrt 2 ∧ min_val = -Real.sqrt 2 ∧ 
  ∀ x ∈ set.Icc 0 (Real.pi / 2), f x ≤ max_val ∧ f x ≥ min_val :=
sorry

end smallest_positive_period_of_f_max_min_of_f_on_interval_l791_791674


namespace arithmetic_sequence_l791_791822

theorem arithmetic_sequence (S : ℕ → ℕ) (h : ∀ n, S n = 3 * n * n) :
  (∃ a d : ℕ, ∀ n : ℕ, S n - S (n - 1) = a + (n - 1) * d) ∧
  (∀ n, S n - S (n - 1) = 6 * n - 3) :=
by
  sorry

end arithmetic_sequence_l791_791822


namespace arrived_home_simultaneously_l791_791914

noncomputable def hours := 3600  -- seconds in an hour
noncomputable def minutes := 60   -- seconds in a minute

def t1 : ℝ := 2 * hours + 40 * minutes -- t1 = 2 hours 40 minutes in seconds
def t2 : ℝ := 4 * hours  -- t2 = 4 hours in seconds
def r : ℝ := 40 * minutes  -- r = 40 minutes in seconds 
def t3 : ℝ := 14 * hours  -- t3 = 14 hours in seconds
def expected_arrival : ℝ := 16 * hours + 24 * minutes -- 16 hours 24 minutes in seconds

-- Let s be the distance between A and B, v be the truck's speed, and V be the car's speed
variables (s v V x : ℝ)

-- Define the main theorem statement without the proof
theorem arrived_home_simultaneously : 
  ∃ (v V : ℝ), 
  (t1 > 0) ∧ 
  (t2 > t1) ∧ 
  (r > 0) ∧ 
  (t3 > t2) ∧
  ((s = v * t2) ∧ (s = V * (t2 - t1)) ∧ (s / v = s / V + t1 - r)) ∧ 
  (x = t3 + t2 - (V * t1) / (v + V)) ∧ 
  (x = expected_arrival) :=
sorry

end arrived_home_simultaneously_l791_791914


namespace scarlet_savings_l791_791421

noncomputable def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost jewelry_set_discount sales_tax_percentage : ℝ) : ℝ :=
  let total_item_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - jewelry_set_discount / 100)
  let total_cost_before_tax := total_item_cost + discounted_jewelry_set_cost
  let total_sales_tax := total_cost_before_tax * (sales_tax_percentage / 100)
  let final_total_cost := total_cost_before_tax + total_sales_tax
  initial_savings - final_total_cost

theorem scarlet_savings : remaining_savings 200 23 48 35 80 25 5 = 25.70 :=
by
  sorry

end scarlet_savings_l791_791421


namespace monday_polygon_tuesday_segments_wednesday_polygons_l791_791954

-- Definitions for the polygon types and conditions
def polygon := { sides : ℕ // sides = 4 ∨ sides = 5 ∨ sides = 6 }

def sides_to_diagonals (p : polygon) : ℕ :=
  match p with
  | ⟨4, _⟩ => 2
  | ⟨5, _⟩ => 5
  | ⟨6, _⟩ => 9
  | _ => 0

-- Question 1: Proof that the polygon drawn on Monday is a pentagon.
theorem monday_polygon : ∃ p : polygon, p.sides = sides_to_diagonals p :=
by {
  sorry -- Replace this with the actual proof.
}

-- Question 2: Proof that Niky drew 27 line segments on Tuesday.
theorem tuesday_segments : ∀ (n : ℕ), (n * 6 = 18) → ((n * 9) = 27) :=
by {
  sorry -- Replace this with the actual proof.
}

-- Question 3: Proof that 7 polygons were drawn on Wednesday.
theorem wednesday_polygons : ∃ n : ℕ, (n * 10 = 70) :=
by {
  sorry -- Replace this with the actual proof.
}

end monday_polygon_tuesday_segments_wednesday_polygons_l791_791954


namespace sum_2004_impossible_sum_2005_possible_l791_791143

-- Condition Definitions
def is_valid_square (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  s = (1, 2, 3, 4) ∨ s = (1, 2, 4, 3) ∨ s = (1, 3, 2, 4) ∨ s = (1, 3, 4, 2) ∨ 
  s = (1, 4, 2, 3) ∨ s = (1, 4, 3, 2) ∨ s = (2, 1, 3, 4) ∨ s = (2, 1, 4, 3) ∨ 
  s = (2, 3, 1, 4) ∨ s = (2, 3, 4, 1) ∨ s = (2, 4, 1, 3) ∨ s = (2, 4, 3, 1) ∨ 
  s = (3, 1, 2, 4) ∨ s = (3, 1, 4, 2) ∨ s = (3, 2, 1, 4) ∨ s = (3, 2, 4, 1) ∨ 
  s = (3, 4, 1, 2) ∨ s = (3, 4, 2, 1) ∨ s = (4, 1, 2, 3) ∨ s = (4, 1, 3, 2) ∨ 
  s = (4, 2, 1, 3) ∨ s = (4, 2, 3, 1) ∨ s = (4, 3, 1, 2) ∨ s = (4, 3, 2, 1)

-- Proof Problems
theorem sum_2004_impossible (n : ℕ) (corners : ℕ → ℕ × ℕ × ℕ × ℕ) (h : ∀ i, is_valid_square (corners i)) :
  4 * 2004 ≠ n * 10 := 
sorry

theorem sum_2005_possible (h : ∃ n, ∃ corners : ℕ → ℕ × ℕ × ℕ × ℕ, (∀ i, is_valid_square (corners i)) ∧ 4 * 2005 = n * 10 + 2005) :
  true := 
sorry

end sum_2004_impossible_sum_2005_possible_l791_791143


namespace parabola_y_relation_l791_791810

-- Conditions of the problem
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- The proof problem statement
theorem parabola_y_relation (c y1 y2 y3 : ℝ) :
  parabola (-4) c = y1 →
  parabola (-2) c = y2 →
  parabola (1 / 2) c = y3 →
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_y_relation_l791_791810


namespace inscribed_circle_radius_one_third_height_l791_791804

theorem inscribed_circle_radius_one_third_height (a b r h2 : ℝ) 
  (h_ap : ∃ A B C : ℝ, A + C = 2 * B) -- This is the arithmetic progression condition for the side lengths
  (h_triangle : ∃ (S : Type) [triangle S], true)  -- Existence of triangle with some type S
  (h_height : true): -- Assume we have the necessary height

  r = h2 / 3 := 
by sorry

end inscribed_circle_radius_one_third_height_l791_791804


namespace drum_y_final_fullness_l791_791175

variable (C : ℝ) -- Capacity of drum x
variable (drum_x_oil : ℝ) -- Initial oil in drum x
variable (drum_y_oil : ℝ) -- Initial oil in drum y

-- Conditions from the problem
hypothesis drum_x_half_full : drum_x_oil = 1/2 * C
hypothesis drum_y_twice_capacity : 2 * C = 2 * C
hypothesis drum_y_one_fifth_full : drum_y_oil = 1/5 * 2 * C

-- Prove that when oil from drum x is poured into drum y, drum y will be 9/10 full
theorem drum_y_final_fullness : 
  drum_y_oil + drum_x_oil = 9/10 * 2 * C :=
by
  sorry

end drum_y_final_fullness_l791_791175


namespace sled_total_distance_l791_791905

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

theorem sled_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 6 → d = 8 → n = 20 → arithmetic_sequence_sum a₁ d n = 1640 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sled_total_distance_l791_791905


namespace average_of_25_results_l791_791435

-- Defining conditions
def avg_first_12 (results : Fin 25 → ℝ) : Prop :=
  ∑ i in Finset.range 12, results ⟨i, by linarith⟩ = 12 * 14

def avg_last_12 (results : Fin 25 → ℝ) : Prop :=
  ∑ i in Finset.range 12 12, results ⟨i + 13, by linarith⟩ = 12 * 17

def thirteenth_result (results : Fin 25 → ℝ) : Prop :=
  results ⟨13, by linarith⟩ = 128

-- Main statement
theorem average_of_25_results (results : Fin 25 → ℝ)
  (h1 : avg_first_12 results)
  (h2 : avg_last_12 results)
  (h3 : thirteenth_result results) :
  (∑ i in Finset.univ, results i) / 25 = 20 :=
sorry

end average_of_25_results_l791_791435


namespace probability_fewer_heads_than_tails_l791_791859

theorem probability_fewer_heads_than_tails (n : ℕ) (hn : n = 12) (p_heads_tails : ∀ k, k ≤ n → k / 2 = n / 2) :
  (num_fewer : ℚ) × (den_fewer : ℕ) := (793, 2048) := 
by 
  have h_total_outcomes : ℕ := 2^12
  have h_equal_heads_tails : ℕ := nat.choose 12 6
  have p_y : ℚ := (nat.choose 12 6) / h_total_outcomes
  have p_x : ℚ := (1 - p_y) / 2
  exacts [(793, 2048)]

end probability_fewer_heads_than_tails_l791_791859


namespace david_marks_in_mathematics_l791_791955

theorem david_marks_in_mathematics 
  (E : ℝ) (P : ℝ) (C : ℝ) (B : ℝ) (A : ℝ) (N : ℝ)
  (hE : E = 72)
  (hP : P = 35)
  (hC : C = 62)
  (hB : B = 84)
  (hA : A = 62.6)
  (hN : N = 5) :
  let M := A * N - (E + P + C + B) in
  M = 60 := 
by 
  unfold M
  rw [hE, hP, hC, hB, hA, hN]
  norm_num
  sorry

end david_marks_in_mathematics_l791_791955


namespace run_faster_l791_791198

theorem run_faster (v_B k : ℝ) (h1 : ∀ (t : ℝ), 96 / (k * v_B) = t → 24 / v_B = t) : k = 4 :=
by {
  sorry
}

end run_faster_l791_791198


namespace fraction_zero_implies_value_l791_791008

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l791_791008


namespace interval_proof_l791_791260

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l791_791260


namespace symmetry_center_tan_l791_791132

theorem symmetry_center_tan (x : ℝ) : 
  let y := (2 * x - (Real.pi / 4)) in
  y = (2 * x - Real.pi / 4) → 
  (∃ k : ℤ, x = (2 * k + 1) * (Real.pi / 8)) → 
  ∃ k : ℤ, (k = -1) → x = -Real.pi / 8 := 
by
  sorry

end symmetry_center_tan_l791_791132


namespace alyona_final_balances_l791_791567

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end alyona_final_balances_l791_791567


namespace radioactive_decay_l791_791887

def initial_mass : ℝ := 500
def decay_rate : ℝ := 0.1
def mass_after_t_years (t : ℝ) : ℝ := initial_mass * (1 - decay_rate) ^ t

theorem radioactive_decay (t : ℝ) : mass_after_t_years t = 500 * 0.9 ^ t := by
  sorry

end radioactive_decay_l791_791887


namespace gcf_60_90_l791_791488

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l791_791488


namespace sin_addition_example_l791_791466

theorem sin_addition_example :
  sin (150 * (π / 180)) * cos (30 * (π / 180)) + cos (150 * (π / 180)) * sin (30 * (π / 180)) = 0 :=
by
  sorry

end sin_addition_example_l791_791466


namespace inclination_angle_of_focal_chord_l791_791664

theorem inclination_angle_of_focal_chord
  (p : ℝ)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x → True)
  (h_focal_chord_length : ∀ A B : ℝ, |A - B| = 8 * p → True) :
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) :=
by
  sorry

end inclination_angle_of_focal_chord_l791_791664


namespace surface_area_spherical_segment_l791_791091

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end surface_area_spherical_segment_l791_791091


namespace circle_regions_area_l791_791541

noncomputable def area_region_outside_small_circle_inside_large_circles
  (A B : Point) (r_small r_large : ℝ) (dist_AB : ℝ)
  (h_radius_small : r_small = 2)
  (h_radius_large : r_large = 3)
  (h_dist_AB : dist_AB = 4) -- since AB is the diameter, we use 2 * radius of the small circle
  (h_internally_tangent : tangent_at A B)
  (h_circle_centers : centers A B r_small r_large) : ℝ :=
  2.664 * π - 10 / 3

theorem circle_regions_area
  (A B : Point) (r_small r_large : ℝ) (dist_AB : ℝ)
  (h_radius_small : r_small = 2)
  (h_radius_large : r_large = 3)
  (h_dist_AB : dist_AB = 4) -- since AB is the diameter, we use 2 * radius of the small circle
  (h_internally_tangent : tangent_at A B)
  (h_circle_centers : centers A B r_small r_large) :
  area_region_outside_small_circle_inside_large_circles A B r_small r_large dist_AB h_radius_small h_radius_large h_dist_AB h_internally_tangent h_circle_centers = 2.664 * π - 10 / 3 :=
sorry

end circle_regions_area_l791_791541


namespace log_of_inv_sqrt_six_l791_791240

theorem log_of_inv_sqrt_six : log 6 (1 / sqrt 6) = -1 / 2 :=
by
  sorry

end log_of_inv_sqrt_six_l791_791240


namespace number_of_even_factors_of_m_l791_791335

noncomputable def m : ℕ := 2^4 * 3^3 * 5^2 * 7

def is_even (n : ℕ) : Prop :=
  ∃ d e f g : ℕ, 1 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 ∧ 0 ≤ g ∧ g ≤ 1 ∧ n = 2^d * 3^e * 5^f * 7^g

def even_factors_count (n : ℕ) : ℕ :=
  if is_even n then 4 * 4 * 3 * 2 else 0

theorem number_of_even_factors_of_m : even_factors_count m = 96 :=
sorry

end number_of_even_factors_of_m_l791_791335


namespace triangle_construction_possible_l791_791152

theorem triangle_construction_possible
  (ABC A1B1C1 : Triangle)
  (h1 : area ABC = area A1B1C1) :
  ∃ (A2B2C2 : Triangle), congruent A1B1C1 A2B2C2 ∧ (parallel (line_through A A2) (line_through B B2)) ∧ (parallel (line_through B B2) (line_through C C2)) :=
sorry

end triangle_construction_possible_l791_791152


namespace prove_problem_statement_l791_791650

variable (f : ℝ → ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonically_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def problem_statement (f : ℝ → ℝ) : Prop :=
  is_even f ∧ (∀ x, f (x : ℝ) ∈ ⊤) ∧
  is_monotonically_decreasing_on f (Set.Iic 0) ∧
  f 2 = 0 →
  { x : ℝ | x * f x ≥ 0 } = Set.Icc (-2) 0 ∪ Set.Ici 2

theorem prove_problem_statement (f : ℝ → ℝ) :
  problem_statement f :=
  sorry

end prove_problem_statement_l791_791650


namespace slope_multiple_x_intercept_l791_791727

variable (k : ℝ → ℝ)
variable (a m : ℝ)

-- Conditions
def y_intercept_of_k_is_0_25 (x : ℝ) : Prop := k x = m * x + 0.25

noncomputable def x_intercept_of_k (k : ℝ → ℝ) (a : ℝ) : Prop :=
  k a = 0

-- The proof statement
theorem slope_multiple_x_intercept :
  (∀ x, y_intercept_of_k_is_0_25 k x) →
  x_intercept_of_k k a →
  m = -1 / 4 :=
sorry

end slope_multiple_x_intercept_l791_791727


namespace range_m_plus_n_l791_791278

def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_m_plus_n (m n : ℝ) (h1 : ∃ x, f(m, n, x) = 0) (h2 : {x | f(m, n, x) = 0} = {x | f(m, n, f(m, n, x)) = 0}) :
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_m_plus_n_l791_791278


namespace prime_factors_exponent_l791_791985

theorem prime_factors_exponent {x : ℕ} :
  (4 ^ 11).prime_factors.count 2 + (7 ^ 5).prime_factors.count 7 + (11 ^ x).prime_factors.count 11 = 29 → x = 2 :=
by
  -- Define the problem conditions as hypotheses
  have h₁ : (4 ^ 11).prime_factors.count 2 = 22 := by sorry,
  have h₂ : (7 ^ 5).prime_factors.count 7 = 5 := by sorry,
  -- Prove that the total number of prime factors and equations must hold
  intros h_total,
  -- Calculate the remaining prime factors needed to reach 29
  have h₃ : (11 ^ x).prime_factors.count 11 = 2 := by sorry,
  -- Show x must be 2
  show x = 2 from sorry

end prime_factors_exponent_l791_791985


namespace evaluate_expression_l791_791597

theorem evaluate_expression : 
  ⌈real.sqrt(16 / 9)⌉ + ⌈16 / 5⌉ + ⌈(16 / 5)^2⌉ = 17 :=
by
  sorry

end evaluate_expression_l791_791597


namespace trapezoid_or_parallelogram_l791_791454

variables (A B C D P Q M1 M3 : Type)
variables [ConvexQuadrilateral A B C D]
variables [Midpoint M1 A B] [Midpoint M3 C D]
variables [Intersection P M1 M3 (Diagonal A C)]
variables [Intersection Q M1 M3 (Diagonal B D)]
variables [EqualSegments M1 P M3 Q]

theorem trapezoid_or_parallelogram : IsTrapezoid A B C D ∨ IsParallelogram A B C D :=
sorry

end trapezoid_or_parallelogram_l791_791454


namespace solution_set_of_g_inequality_l791_791649

noncomputable def f (x : ℝ) : ℝ := sorry [even function condition]
noncomputable def f' (x : ℝ) : ℝ := sorry [derivative condition]
def g (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_g_inequality : 
  ∀ x : ℝ, g x < g (1 - x) ↔ (x < 0 ∨ (0 < x ∧ x < 1 / 2)) :=
sorry

end solution_set_of_g_inequality_l791_791649


namespace suitable_survey_method_l791_791481

/-- 
To understand the time spent by primary and secondary school students nationwide 
on completing homework under the "double reduction" policy background, the 
most suitable survey method is "sampling survey".
-/
theorem suitable_survey_method :
  (∀ (students : Type) (spends_time_on_homework : students → Prop),
     policy == "double reduction" →
     (survey_method students == "sampling survey" ↔ survey_method students ≠ "comprehensive survey")) :=
sorry

end suitable_survey_method_l791_791481


namespace monotonic_intervals_max_min_values_l791_791599

noncomputable def F (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 8 * x

theorem monotonic_intervals :
  (∀ x, 2 < x → F'(x) > 0) ∧ (∀ x, 0 < x ∧ x < 2 → F'(x) < 0) := 
begin
  sorry
end

theorem max_min_values [x ∈ Icc 1 3]:
  (∃ x, x ∈ Icc 1 3 ∧ F(x) = -6) ∧ (∃ x, x ∈ Icc 1 3 ∧ F(x) = -28/3) := 
begin
  sorry
end

end monotonic_intervals_max_min_values_l791_791599


namespace limit_seq_half_l791_791523

noncomputable def limit_sequence : ℝ :=
  Real.limit (λ n : ℕ, 
    (Real.exp (1 / n) + Real.sin (n / (n^2 + 1)) * Real.cos n) / 
    (1 + Real.cos (1 / n)))

theorem limit_seq_half : limit_sequence = 1 / 2 := 
  sorry

end limit_seq_half_l791_791523


namespace shooting_match_problem_l791_791022

theorem shooting_match_problem : 
  let targets := 9
  let targets_column_A := 4
  let targets_column_B := 3
  let targets_column_C := 2
  let total_arrangements := targets.factorial / (targets_column_A.factorial * targets_column_B.factorial * targets_column_C.factorial)
  in total_arrangements = 1260 := by
  sorry

end shooting_match_problem_l791_791022


namespace exists_convex_ngon_with_side_lengths_l791_791089

theorem exists_convex_ngon_with_side_lengths (n : ℕ) (h : ¬ (∃ p : ℕ, nat.prime p ∧ ∃ k : ℕ, n = p^k)) :
  ∃ polygon : List ℕ, polygon = List.range (n+1) ∧ (∀ i < polygon.length, Polygon.angle i = Polygon.angle 0) :=
sorry

end exists_convex_ngon_with_side_lengths_l791_791089


namespace order_abc_l791_791291

noncomputable def a := Real.exp (0.6 * Real.log 2)
noncomputable def b := Real.log 10 0.6
noncomputable def c := Real.log 10 0.4

theorem order_abc : c < b ∧ b < a := sorry

end order_abc_l791_791291


namespace hyperbola_equation_l791_791950

theorem hyperbola_equation
  (a b c : Real)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (eccentricity_condition : c = a * sqrt 3)
  (directrix_condition : c = a^2) :
  (∀ x y : Real, (y^2 / a^2 - x^2 / b^2 = 1) ↔ (x^2 / 3 - y^2 / 6 = 1)) :=
by
  sorry

end hyperbola_equation_l791_791950


namespace log_floor_difference_probability_l791_791397

open Real

noncomputable def probability_floor_log_diff_zero : Prop :=
  let log_factor := log10 5
  let lower_bound (x : ℝ) := (0 : ℝ) < x ∧ x < 1
  ∃ (prob : ℝ), prob = 8/9 ∧ ∀ (x : ℝ), lower_bound x → (⌊ log10 (5 * x) ⌋ - ⌊ log10 x ⌋ = 0) → prob = 8/9

theorem log_floor_difference_probability : probability_floor_log_diff_zero :=
by
  sorry

end log_floor_difference_probability_l791_791397


namespace square_of_product_of_third_sides_l791_791417

noncomputable def T1 : Triangle := ⟨3, 4, 5⟩ -- sides in 3:4:5 ratio
noncomputable def T2 : Triangle := ⟨3, _, 5⟩ -- side 3 and hypotenuse 5

axiom area_T1 : area T1 = 4
axiom area_T2 : area T2 = 8
axiom sides_T2_congruent : hypotenuse T1 = hypotenuse T2

theorem square_of_product_of_third_sides :
  let third_side_T1 := side_orthogonal_to (3, 4, 5) in
  let third_side_T2 := side_orthogonal_to (3, _, 5) in
  (third_side_T1 * third_side_T2) ^ 2 = 682.67 :=
by
  sorry

end square_of_product_of_third_sides_l791_791417


namespace angle_between_vectors_l791_791652

/--
Given point A(6, 4, -4) and point B(-3, -2, 2), with O being the origin of coordinates,
prove that the angle between vector OA and vector OB is 120 degrees.
-/
theorem angle_between_vectors : 
  let OA := (6, 4, -4:int)
  let OB := (-3, -2, 2:int)
  let dot_product := 6 * (-3) + 4 * (-2) + (-4) * 2
  let magnitude_OA := Real.sqrt (6^2 + 4^2 + (-4)^2)
  let magnitude_OB := Real.sqrt ((-3)^2 + (-2)^2 + 2^2)
  let cos_theta := dot_product / (magnitude_OA * magnitude_OB)
  ∃ θ : ℝ, θ = Real.acos (-1 / 2) ∧ Real.to_degrees θ = 120 :=
by
  sorry

end angle_between_vectors_l791_791652


namespace sector_properties_l791_791642

noncomputable def central_angle (l R : ℝ) : ℝ := l / R

noncomputable def area_of_sector (l R : ℝ) : ℝ := (1 / 2) * l * R

theorem sector_properties (R l : ℝ) (hR : R = 8) (hl : l = 12) :
  central_angle l R = 3 / 2 ∧ area_of_sector l R = 48 :=
by
  sorry

end sector_properties_l791_791642


namespace square_in_S_l791_791750

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

def S (n : ℕ) : Prop :=
  is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1)

theorem square_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end square_in_S_l791_791750


namespace fraction_equation_l791_791689

theorem fraction_equation (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end fraction_equation_l791_791689


namespace sum_of_numbers_is_70_l791_791105

noncomputable def sum_of_numbers_in_ratio (a b : ℕ) (LCM : ℕ) (h : LCM = Nat.lcm a b) (ratio_h : a * 3 = b * 2) : ℕ :=
a + b

theorem sum_of_numbers_is_70 (x : ℕ) (h : Nat.lcm (2 * x) (3 * x) = 42) : sum_of_numbers_in_ratio (2 * x) (3 * x) 42 h (by sorry) = 70 :=
by
  sorry

end sum_of_numbers_is_70_l791_791105


namespace total_resistance_circuit_l791_791617

theorem total_resistance_circuit (r : ℝ) (h : r = 1) : 
  let R_total := 1.4 in R_total = (7/5) * r := by
  sorry

end total_resistance_circuit_l791_791617


namespace amount_after_two_years_l791_791517

noncomputable def amountAfterYears (presentValue : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  presentValue * (1 + rate) ^ n

theorem amount_after_two_years 
  (presentValue : ℝ := 62000) 
  (rate : ℝ := 1 / 8) 
  (n : ℕ := 2) : 
  amountAfterYears presentValue rate n = 78468.75 := 
  sorry

end amount_after_two_years_l791_791517


namespace intersection_points_product_l791_791160

theorem intersection_points_product (x y : ℝ) :
  (x^2 - 2 * x + y^2 - 6 * y + 9 = 0) ∧ (x^2 - 8 * x + y^2 - 6 * y + 28 = 0) → x * y = 6 :=
by
  sorry

end intersection_points_product_l791_791160


namespace sequence_formula_l791_791073

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2 - n * a n + 1

theorem sequence_formula (a : ℕ → ℕ) (h : sequence a) : ∀ n : ℕ, n > 0 → a n = n + 1 := by
  sorry

end sequence_formula_l791_791073


namespace jovana_bucket_shells_l791_791741

theorem jovana_bucket_shells :
  let a0 := 5.2
  let a1 := a0 + 15.7
  let a2 := a1 + 17.5
  let a3 := a2 - 4.3
  let a4 := 3 * a3
  a4 = 102.3 := 
by
  sorry

end jovana_bucket_shells_l791_791741


namespace num_sets_M_l791_791807

theorem num_sets_M :
  ∃ M : Type, (M → Prop) = 
    {M | M ∪ {1} = {1, 2}} ∧
    (∃ M₁ M₂, {M := SetOf (fun x => x = {2} ∨ x = {1, 2}), 
                 M₁ ≠ M₂ ∧ 
                 (M₁ = {2} ∨ M₁ = {1, 2}) ∧ 
                 (M₂ = {2} ∨ M₂ = {1, 2}))} :=
sorry

end num_sets_M_l791_791807


namespace width_of_river_l791_791843

def ferry_problem (v1 v2 W t1 t2 : ℝ) : Prop :=
  v1 * t1 + v2 * t1 = W ∧
  v1 * t1 = 720 ∧
  v2 * t1 = W - 720 ∧
  (v1 * t2 + v2 * t2 = 3 * W) ∧
  v1 * t2 = 2 * W - 400 ∧
  v2 * t2 = W + 400

theorem width_of_river 
  (v1 v2 W t1 t2 : ℝ)
  (h : ferry_problem v1 v2 W t1 t2) :
  W = 1280 :=
by
  sorry

end width_of_river_l791_791843


namespace vertex_of_parabola_l791_791457

theorem vertex_of_parabola (c d: ℝ) (h: ∀ x: ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ 6 ≤ x)) :
  (1, 25) = vertex_of_parabola (-x^2 + c * x + d) :=
by
  sorry

end vertex_of_parabola_l791_791457


namespace fourth_product_is_correct_l791_791723
-- Import Mathlib for broader library support

-- Define the row of random numbers explicitly
def second_row : List ℕ := [7216, 2884, 3660, 1073, 4366, 7575]

-- Function to select the 'n-th' product number from the row following the described method
def product_number (n : ℕ) : ℕ :=
  second_row.drop 2 -- Drop the first two groups
             .init -- Remove the last number to fit the problem's 50 selections
             .nth_le (n-1) sorry -- Select the nth number

-- Define the target product number for the fourth selection
def fourth_product_number := 447

-- The theorem stating that the number of the fourth product selected is 447
theorem fourth_product_is_correct : product_number 4 = fourth_product_number := 
  sorry

end fourth_product_is_correct_l791_791723


namespace square_87_l791_791579

theorem square_87 : 87^2 = 7569 :=
by
  sorry

end square_87_l791_791579


namespace student_arrangement_l791_791719

theorem student_arrangement :
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  valid_arrangements = 336 :=
by
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  exact sorry

end student_arrangement_l791_791719


namespace traffic_minister_can_achieve_goal_l791_791352

def transportation_problem : Prop :=
  ∃ (days : ℕ), days ≤ 214 ∧ ∀ cities : Fin 32 → Fin 32 → Bool, -- cities represented as a function from pairs to road directions
    ∃ f : Fin 32 → Fin 32 → Bool, -- f is the final state of the road network
      (∀ i j : Fin 32, cities i j = f i j ∨ (days ≤ 214 ∧ ∃ k (h₁ : i ≠ k) (h₂ : j ≠ k), f i k ∧ f k j)) -- constraints ensuring no city is re-visitable.

theorem traffic_minister_can_achieve_goal : transportation_problem := 
  sorry -- Proof is omitted

end traffic_minister_can_achieve_goal_l791_791352


namespace limit_seq_half_l791_791524

noncomputable def limit_sequence : ℝ :=
  Real.limit (λ n : ℕ, 
    (Real.exp (1 / n) + Real.sin (n / (n^2 + 1)) * Real.cos n) / 
    (1 + Real.cos (1 / n)))

theorem limit_seq_half : limit_sequence = 1 / 2 := 
  sorry

end limit_seq_half_l791_791524


namespace evaluate_star_l791_791997

-- Define the operation c star d
def star (c d : ℤ) : ℤ := c^2 - 2 * c * d + d^2

-- State the theorem to prove the given problem
theorem evaluate_star : (star 3 5) = 4 := by
  sorry

end evaluate_star_l791_791997


namespace g_at_neg_two_is_fifteen_l791_791441

def g (x : ℤ) : ℤ := 2 * x^2 - 3 * x + 1

theorem g_at_neg_two_is_fifteen : g (-2) = 15 :=
by 
  -- proof is skipped
  sorry

end g_at_neg_two_is_fifteen_l791_791441


namespace power_mod_19_l791_791496

theorem power_mod_19 :
  ∀ n : ℕ, 7^n % 19 = 7 ↔ n % 18 = 1 :=
begin
  sorry
end

example : 7^2023 % 19 = 4 :=
begin
  have h := power_mod_19 2023,
  rw nat.mod_eq_of_lt, -- 2023 % 18 = 7
  norm_num,
  exact h,
  sorry
end

end power_mod_19_l791_791496


namespace combined_perimeter_approx_l791_791558

noncomputable def combined_perimeter : ℕ :=
  let radius : ℝ := 12
  let side_length : ℝ := radius
  let perimeter_hexagon : ℝ := 6 * side_length
  let perimeter_semicircle : ℝ := π * radius + 2 * radius
  let total_perimeter : ℝ := perimeter_hexagon + perimeter_semicircle
  Int.round total_perimeter

theorem combined_perimeter_approx : combined_perimeter = 134 := by
  sorry

end combined_perimeter_approx_l791_791558


namespace interval_of_monotonic_decrease_l791_791346

noncomputable def g (x : ℝ) : ℝ := (1 / 2) ^ x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem interval_of_monotonic_decrease :
  ∀ x : ℝ, 0 < x ∧ x ≤ 3/2 → ∃ I, I = (0, 3/2) ∧ (∀ y ∈ I, ∀ z ∈ I, y < z → f (3 * y - y^2) > f (3 * z - z^2)) :=
by
  sorry

end interval_of_monotonic_decrease_l791_791346


namespace yellow_highlighters_l791_791706

theorem yellow_highlighters (total pink blue : ℕ) (h_total : total = 11) (h_pink : pink = 4) (h_blue : blue = 5) : total - (pink + blue) = 2 :=
by
  rw [h_total, h_pink, h_blue]
  norm_num
  done

end yellow_highlighters_l791_791706


namespace broken_line_length_greater_1248_l791_791537

noncomputable def broken_line_length (n : ℕ) (A : ℕ → ℝ × ℝ) : ℝ :=
  ∑ i in finset.range (n - 1), real.dist (A i) (A (i + 1))

def condition_distance_less_than_1 (n : ℕ) (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ x y : ℝ × ℝ,
  0 ≤ x.1 ∧ x.1 ≤ 50 ∧ 0 ≤ x.2 ∧ x.2 ≤ 50 →
  real.inf_dist x (set.range A) < 1

theorem broken_line_length_greater_1248 :
  ∀ (n : ℕ) (A : ℕ → ℝ × ℝ),
  condition_distance_less_than_1 n A →
  broken_line_length n A > 1248 :=
sorry

end broken_line_length_greater_1248_l791_791537


namespace road_does_not_lead_to_port_l791_791404

-- Definitions
inductive Native
| Truthful : Native
| Liar : Native

-- Native's statements
def native_says (n : Native) (s : Prop) : Prop :=
match n with
| Native.Truthful => s
| Native.Liar => ¬ s
end

-- Problem conditions:
def native1 : Native := Native.Liar -- from the solution conclusion
def native2_says : Prop := native_says native2 (native2 = Native.Liar)

-- Question to be proved:
theorem road_does_not_lead_to_port (native1: Native) (native1_points_road: Prop):
  native_says native1 native2_says → ¬ native1_points_road := 
by
simmer sorry

end road_does_not_lead_to_port_l791_791404


namespace price_per_peach_to_relatives_l791_791758

-- Given conditions translated to Lean definitions
def totalPeaches : ℕ := 15
def peachesToFriends : ℕ := 10
def pricePerPeachToFriends : ℕ := 2
def peachesToRelatives : ℕ := 4
def peachesForHerself : ℕ := 1
def totalEarnings : ℕ := 25

-- The theorem to prove that the price per peach sold to relatives is $1.25
theorem price_per_peach_to_relatives : 
  (totalEarnings - peachesToFriends * pricePerPeachToFriends) / peachesToRelatives = 1.25 := by
  -- TODO: Proof goes here
  sorry

end price_per_peach_to_relatives_l791_791758


namespace range_of_a_l791_791145

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ a ∈ set.Iio (-2) ∪ set.Ioi 2 :=
sorry

end range_of_a_l791_791145


namespace exist_perpendicular_pairs_l791_791146

-- Define non-perpendicular lines passing through O 
variables {O : EuclideanSpace ℝ (Fin 3)} 
variables (l : Fin 1979 → Set (EuclideanSpace ℝ (Fin 3)))
variables (A : Fin 1979 → EuclideanSpace ℝ (Fin 3))

-- Ensure each line passes through point O
def lines_pass_through_O : Prop :=
  ∀ i, O ∈ l i

-- Ensure no lines are mutually perpendicular
def no_mutual_perpendicularity : Prop :=
  ∀ i j, i ≠ j → ∀ (p ∈ l i) (q ∈ l j), (p - O) ⬝ (q - O) ≠ 0

-- Ensure point A1 on line l1 is distinct from O
def distinct_A1 : Prop :=
  A 0 ∈ l 0 ∧ A 0 ≠ O

-- Ensure 1979 pairs of lines are mutually perpendicular according to the problem statement
def perpendicular_pairs : Prop :=
  ∀ i, (A ((i + 1) % 1979) - A ((i - 1) % 1979)) ⬝ (vector_for_line (l i) O A) = 0

-- Main theorem to prove the existence of such points A2 to A1979
theorem exist_perpendicular_pairs
  (h1: lines_pass_through_O l O) 
  (h2: no_mutual_perpendicularity l O) 
  (h3: distinct_A1 l O A) :
  ∃ (A : Fin 1979 → EuclideanSpace ℝ (Fin 3)), perpendicular_pairs l A :=
sorry

end exist_perpendicular_pairs_l791_791146


namespace find_radius_of_inscribed_circle_l791_791747

-- Definitions based on conditions
def length := 8
def width := 3
def radius_A := 4
def radius_B := 4
def radius_C := 1.5
def radius_D := 1.5

-- Definitions based on points
def A := (0, 0)
def B := (length, 0)
def C := (length, width)
def D := (0, width)

-- Definitions based on the distance conditions
def dist (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem find_radius_of_inscribed_circle :
  ∃ r x y, (0 ≤ x ∧ x ≤ length ∧
            0 ≤ y ∧ y ≤ width ∧
            dist A (x, y) = radius_A + r ∧
            dist C (x, y) = radius_C + r ∧
            r = (real.sqrt 73 - 8) / 2) :=
sorry

end find_radius_of_inscribed_circle_l791_791747


namespace slope_of_line_inclination_angle_l791_791131

theorem slope_of_line_inclination_angle 
  (k : ℝ) (θ : ℝ)
  (hθ1 : 30 * (π / 180) < θ)
  (hθ2 : θ < 90 * (π / 180)) :
  k = Real.tan θ → k > Real.tan (30 * (π / 180)) :=
by
  intro h
  sorry

end slope_of_line_inclination_angle_l791_791131


namespace JeansSpeedCorrect_l791_791938

-- Define the parameters and conditions
variables (d : ℝ) (t1 t2 t3 T : ℝ)

-- Define the speeds and segment distances
def ChantalsSpeed1 : ℝ := 5
def ChantalsSpeed2 : ℝ := 2.5
def ChantalsSpeed3 : ℝ := 4

-- Define conditions for time calculation
def time1 := d / ChantalsSpeed1
def time2 := 3 * d / ChantalsSpeed2
def time3 := d / ChantalsSpeed3

-- Total time T taken by Chantal to meet Jean
def totalTime := time1 + time2 + time3

-- Jean's average speed until they meet
def JeansAverageSpeed := d / totalTime

-- Main theorem statement
theorem JeansSpeedCorrect : JeansAverageSpeed d time1 time2 time3 totalTime = 20 / 33 :=
by
  sorry

end JeansSpeedCorrect_l791_791938


namespace cube_mod_35_divisors_l791_791614

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors_l791_791614


namespace solve_quadratic_inequality_l791_791098

theorem solve_quadratic_inequality (x : ℝ) :
  (-3 * x^2 + 8 * x + 5 > 0) ↔ (x < -1 / 3) :=
by
  sorry

end solve_quadratic_inequality_l791_791098


namespace tan_alpha_eq_neg_one_l791_791994

theorem tan_alpha_eq_neg_one (alpha : ℝ) (h1 : Real.tan alpha = -1) (h2 : 0 ≤ alpha ∧ alpha < Real.pi) :
  alpha = (3 * Real.pi) / 4 :=
sorry

end tan_alpha_eq_neg_one_l791_791994


namespace painted_cubes_at_least_two_faces_l791_791896

/--
A four-inch wooden cube is painted blue on all six faces and then cut into one-inch cubes.
Prove that the number of one-inch cubes that have blue paint on at least two faces is 32.
-/
theorem painted_cubes_at_least_two_faces : 
  let total_cubes := 4 * 4 * 4 in
  let corner_cubes := 8 in
  let edge_cubes_per_edge := 4 - 2 in
  let edges_count := 12 in
  let two_faces_cubes := edge_cubes_per_edge * edges_count in
  corner_cubes + two_faces_cubes = 32 :=
by 
  -- Define the variables
  let total_cubes := 4 * 4 * 4
  let corner_cubes := 8
  let edge_cubes_per_edge := 4 - 2
  let edges_count := 12
  let two_faces_cubes := edge_cubes_per_edge * edges_count
  -- Prove the statement
  have h1 : total_cubes = 64 := by linarith
  have h2 : corner_cubes + two_faces_cubes = 8 + 24 := by linarith
  have h3 : 8 + 24 = 32 := by linarith
  show corner_cubes + two_faces_cubes = 32 from h3
  sorry

end painted_cubes_at_least_two_faces_l791_791896


namespace incorrect_options_in_line_l791_791676

-- Define the line l: sqrt(3) x + y - 2 = 0
def line_eq (x y : ℝ) := sqrt 3 * x + y - 2 = 0

-- Define the slope-intercept form of the line
def slope_intercept_form (x : ℝ) := -sqrt 3 * x + 2

-- Define the condition check functions
def slope_is (m : ℝ) := m = -sqrt 3
def angle_of_inclination_is (α : ℝ) := tan α = -sqrt 3
def passes_third_quadrant (x y : ℝ) := line_eq x y ∧ x < 0 ∧ y < 0
def directional_vector_is (v : (ℝ × ℝ)) := v = (-sqrt 3, 3)

-- Now let's state the problem 

theorem incorrect_options_in_line :
  ∃ option, 
    (option = "A" ∧ ¬ slope_is (π / 6))
    ∨ (option = "B" ∧ ¬ slope_is (sqrt 3))
    ∨ (option = "C" ∧ ∃ x y, ¬ passes_third_quadrant x y)
    ∨ (option = "D" ∧ ¬ directional_vector_is (-sqrt 3, 3)) :=
sorry

end incorrect_options_in_line_l791_791676


namespace nat_interval_property_l791_791978

theorem nat_interval_property (n : ℕ) :
  (∀ k : ℕ, k^2 + int.floor (n / k^2) = 1991) ↔ (1024 * 967 ≤ n ∧ n < 1024 * 968) :=
sorry

end nat_interval_property_l791_791978


namespace ratio_of_male_democrats_to_total_male_participants_l791_791827

open Classical

noncomputable def proof_problem : Prop :=
  ∃ (F M D_f D D_m : ℕ),
    F + M = 870 ∧
    D_f = 145 ∧
    D_f = F / 2 ∧
    D = 870 / 3 ∧
    D_m = D - D_f ∧
    (D_m * 4 = M)

theorem ratio_of_male_democrats_to_total_male_participants : proof_problem :=
by {
  use (290, 580, 145, 290, 145), 
  split, { exact rfl }, 
  split, { exact rfl }, 
  split, { exact Nat.div_eq_of_eq_mul_left (by norm_num) rfl }, 
  split, { exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl }, 
  split, { exact nat.sub_self.symm }, 
  exact rfl
}

end ratio_of_male_democrats_to_total_male_participants_l791_791827


namespace find_k_l791_791668

noncomputable def arithmetic_sum (n : ℕ) (a1 d : ℚ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_k 
  (a1 d : ℚ) (k : ℕ)
  (h1 : arithmetic_sum (k - 2) a1 d = -4)
  (h2 : arithmetic_sum k a1 d = 0)
  (h3 : arithmetic_sum (k + 2) a1 d = 8) :
  k = 6 :=
by
  sorry

end find_k_l791_791668


namespace man_speed_in_kmph_l791_791912

theorem man_speed_in_kmph
    (length_train : ℝ)
    (speed_train_kmph : ℝ)
    (time_seconds : ℕ)
    (man_speed_kmph : ℝ) :
    length_train = 330 →
    speed_train_kmph = 60 →
    time_seconds = 18 →
    (man_speed_kmph ≈ 5.99) := by
    sorry

end man_speed_in_kmph_l791_791912


namespace speed_of_mans_train_l791_791548

-- Define the given conditions
def goods_train_speed_kmph : ℝ := 60.2
def goods_train_length_m : ℝ := 320
def passing_time_s : ℝ := 10
def relative_speed_mps := goods_train_length_m / passing_time_s 

-- Convert speed of goods train to m/s
def goods_train_speed_mps := (goods_train_speed_kmph * 1000) / 3600

-- Prove that the speed of the train the man is sitting in is approximately 54.99 km/h
theorem speed_of_mans_train (v : ℝ) : 
    v = relative_speed_mps - goods_train_speed_mps → 
    (v * 3600 / 1000 ≈ 54.99) :=
sorry

end speed_of_mans_train_l791_791548


namespace tournament_games_l791_791080

-- Define the conditions
def number_of_teams : ℕ := 32

-- Define the function that counts the number of games
def count_games (n : ℕ) : ℕ :=
  n - 1

-- Prove the main statement
theorem tournament_games : count_games number_of_teams = 31 :=
  by
    rw [count_games, number_of_teams]
    exact Nat.sub_self 1
    sorry

end tournament_games_l791_791080


namespace sum_of_roots_of_polynomial_l791_791983

noncomputable def polynomial : Polynomial ℝ := 
  ∑ k in Finset.range 1003, (k + 1) * ((Polynomial.C (k + 1) - Polynomial.X) ^ (1003 - k))

theorem sum_of_roots_of_polynomial :
  (rootSum polynomial Polynomial.X) = 1003 + ∑ k in Finset.range (1003 - 1), (k + 2) ^ 2 :=
by
  sorry

end sum_of_roots_of_polynomial_l791_791983


namespace total_pumpkins_l791_791777

theorem total_pumpkins (Sandy Mike Maria Sam : ℕ) (h1 : Sandy = 51) (h2 : Mike = 23) (h3 : Maria = 37) (h4 : Sam = 46) : 
  Sandy + Mike + Maria + Sam = 157 := by 
  rw [h1, h2, h3, h4]
  exact Nat.add_assoc (51 + 23) 37 46
  sorry

end total_pumpkins_l791_791777


namespace units_digit_27_3_sub_17_3_l791_791863

theorem units_digit_27_3_sub_17_3 : 
  (27 ^ 3 - 17 ^ 3) % 10 = 0 :=
sorry

end units_digit_27_3_sub_17_3_l791_791863


namespace parabola_focus_l791_791639

theorem parabola_focus (F : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 4 * x → (x + 1)^2 + y^2 = ((x - F.1)^2 + (y - F.2)^2)) → 
  F = (1, 0) :=
sorry

end parabola_focus_l791_791639


namespace gcd_60_90_l791_791492

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end gcd_60_90_l791_791492


namespace soda_original_price_l791_791272

theorem soda_original_price (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  sorry

end soda_original_price_l791_791272


namespace range_of_x_l791_791014

noncomputable def a (x : ℝ) : ℝ := x
def b : ℝ := 2
def B : ℝ := 60

-- State the problem: Prove the range of x given the conditions
theorem range_of_x (x : ℝ) (A : ℝ) (C : ℝ) (h1 : a x = b / (Real.sin (B * Real.pi / 180)) * (Real.sin (A * Real.pi / 180)))
  (h2 : A + C = 180 - 60) (two_solutions : (60 < A ∧ A < 120)) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
sorry

end range_of_x_l791_791014


namespace sum_of_prime_values_of_h_l791_791273

def h (n : ℤ) : ℤ := n^4 - 3 * n^3 + 360 * n^2 - 3 * n + 401

theorem sum_of_prime_values_of_h :
  (∑ n in (finset.filter (λ n, nat.prime (h n)) (finset.range 1000)), h n) = 401 :=
sorry

end sum_of_prime_values_of_h_l791_791273


namespace probability_three_dice_show_five_l791_791915

theorem probability_three_dice_show_five : 
  let prob_five := (1/6 : ℚ)
  let prob_not_five := (5/6 : ℚ)
  let num_ways := (5.choose 3)
  let prob_three_fives := num_ways * (prob_five ^ 3) * (prob_not_five ^ 2)
  prob_three_fives = (125 / 3888 : ℚ) :=
by
  let prob_five := (1/6 : ℚ)
  let prob_not_five := (5/6 : ℚ)
  let num_ways := (5.choose 3)
  let prob_three_fives := num_ways * (prob_five ^ 3) * (prob_not_five ^ 2)
  have h : prob_three_fives = (125 / 3888 : ℚ) := by sorry
  exact h

end probability_three_dice_show_five_l791_791915


namespace interval_solution_l791_791256

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l791_791256


namespace part1_part2_l791_791651

open Complex

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := 4 + 3 * I

theorem part1 : z1 * z2 = 10 - 5 * I := by
  sorry

noncomputable def z : ℂ := -Real.sqrt 2 - Real.sqrt 2 * I

theorem part2 (h_abs_z : abs z = 2)
              (h_img_eq_real : z.im = (3 * z1 - z2).re)
              (h_quadrant : z.re < 0 ∧ z.im < 0) : z = -Real.sqrt 2 - Real.sqrt 2 * I := by
  sorry

end part1_part2_l791_791651


namespace intersection_A_B_l791_791324

def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}
def inter : Set ℝ := {x | 3 < x ∧ x < 4}

theorem intersection_A_B : A ∩ B = inter := 
by 
  sorry

end intersection_A_B_l791_791324


namespace sequence_limit_l791_791525

open Real

theorem sequence_limit (e_n : ℕ → ℝ) (sin_cos_n : ℕ → ℝ)
  (cos_inv_n : ℕ → ℝ) (hyp_e_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (e_n n - 1) < ε)
  (hyp_sin_cos_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (sin_cos_n n - 0) < ε)
  (hyp_cos_inv_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (cos_inv_n n - 1) < ε)
  (seq_n : ℕ → ℝ) :
  ( ∀ n, e_n n = exp (1 / seq_n n) ) →
  ( ∀ n, sin_cos_n n = sin ( seq_n n / (seq_n n ^ 2 + 1) ) * cos n ) →
  ( ∀ n, cos_inv_n n = cos ( 1 / seq_n n )) →
  tendsto (λ n, (e_n n + sin_cos_n n) / (1 + cos_inv_n n)) at_top (𝓝 (1 / 2)) := by
sorry

end sequence_limit_l791_791525


namespace systematic_sampling_students_l791_791834

theorem systematic_sampling_students :
  ∀ (total_students selected_students : ℕ) (first_student : ℕ),
  total_students = 48 →
  selected_students = 4 →
  first_student = 7 →
  ∃ (student1 student2 student3 : ℕ), 
  student1 = 19 ∧ 
  student2 = 31 ∧
  student3 = 43 :=
begin
  intros total_students selected_students first_student,
  intros h_total h_selected h_first,
  use (first_student + total_students / selected_students),
  use (first_student + 2 * (total_students / selected_students)),
  use (first_student + 3 * (total_students / selected_students)),
  split, 
  { rw [h_first, h_total, h_selected], norm_num },
  split,
  { rw [h_first, h_total, h_selected], norm_num },
  { rw [h_first, h_total, h_selected], norm_num }
end

end systematic_sampling_students_l791_791834


namespace largest_sum_of_palindromes_l791_791813

theorem largest_sum_of_palindromes (a b : ℕ) :
  (a ≥ 100 ∧ a < 1000) ∧ (b ≥ 100 ∧ b < 1000) ∧
  (∀ n, n.reverseDigits = n → ∃ a b, n = a * b) ∧
  a * b = 906609 →
  a + b = 1818 :=
sorry

end largest_sum_of_palindromes_l791_791813


namespace polygon_distance_sum_l791_791542

-- Define the circle and polygon setup
def circle (r : ℝ) : Prop := r > 0

def polygon (n : ℕ) (a : Fin n → ℝ) (P : ℝ) : Prop :=
  ∃ sum_a : ℝ, (∀ i, 0 < a i) ∧ sum_a = P ∧ sum_a = (Finset.univ.sum (λ i, a i))

def touching_points 
  (P : Prop) (points : Fin n → ℝ × ℝ) (a : Fin n → ℝ) (r : ℝ) : Prop := 
  P

-- Define the distances and positions
def distances 
  (center : ℝ × ℝ) (points : Fin n → ℝ × ℝ) (X : ℝ × ℝ) (d : ℝ) : Prop :=
  ∃ ci, (∀ i, dist center (points i) = r) ∧ dist center X = d

def XA_squared 
  (center : ℝ × ℝ) (points : Fin n → ℝ × ℝ) (X : ℝ × ℝ) (a : Fin n → ℝ) : ℝ :=
  Finset.univ.sum (λ i, a i * (dist X (points i))^2)

def perimeter (a : Fin n → ℝ) : ℝ :=
  Finset.univ.sum (λ i, a i)

-- The statement of the problem
theorem polygon_distance_sum 
  (r d : ℝ) (n : ℕ) (a : Fin n → ℝ) (P : ℝ) (center : ℝ × ℝ) (points : Fin n → ℝ × ℝ) 
  (X : ℝ × ℝ) 
  (c_circle : circle r) 
  (c_polygon : polygon n a P) 
  (c_touching_points : touching_points c_polygon points a r) 
  (c_distances : distances center points X d) 
  : XA_squared center points X a = perimeter a * (r^2 + d^2) := 
sorry

end polygon_distance_sum_l791_791542


namespace f_nonnegative_l791_791394

variable {R : Type*} [Real] -- Define the real type

-- Define our function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assume the conditions mentioned
axiom f_has_derivative : ∀ x : ℝ, deriv f x = f' x
axiom positivity_condition : ∀ x : ℝ, 2 * f x + x * f' x > 0

-- The theorem we want to prove
theorem f_nonnegative : ∀ x : ℝ, f x ≥ 0 :=
by sorry

end f_nonnegative_l791_791394


namespace circle_of_unit_diameter_can_be_placed_without_intersecting_squares_l791_791190

open Real Set

/-- Proving there exists an available space for a circle of unit diameter inside a 20x25 rectangle,
even when 120 unit squares are placed arbitrarily in the rectangle without intersection. -/
theorem circle_of_unit_diameter_can_be_placed_without_intersecting_squares :
  ∃ (circle_center : Real × Real),
  (∀ (sq_center : Real × Real) ((sq_orientation : ℝ), sq_center ∈ unit_squares ∧ within_rectangle) ->
  (dist circle_center sq_center ≥ 1)) ∧
  within_rectangle circle_center :=
sorry

noncomputable def unit_squares : Set (ℝ × ℝ) := 
{ sq | ∃ (sqc : ℝ × ℝ) (sqo : ℝ), within_rectangle sqc ∧ orientation sqo }

def within_rectangle (pt : ℝ × ℝ) : Prop :=
pt.1 ≥ 0 ∧ pt.1 ≤ 20 ∧ pt.2 ≥ 0 ∧ pt.2 ≤ 25

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def orientation (sq : Real × Real) : Prop :=
-- Assuming arbitrary representation of unit square orientation
true -- this is a placeholder

end circle_of_unit_diameter_can_be_placed_without_intersecting_squares_l791_791190


namespace fuel_left_in_tank_l791_791900

theorem fuel_left_in_tank (fuel_rate : ℝ) (time_remaining : ℝ) (gallons_left: ℝ) : 
  fuel_rate = 9.5 → time_remaining = 0.6667 → gallons_left = fuel_rate * time_remaining → gallons_left = 6.33365 :=
by
  intros fr_eq tr_eq gl_eq
  rw [fr_eq, tr_eq] at gl_eq
  simp at gl_eq
  exact gl_eq

end fuel_left_in_tank_l791_791900


namespace scientific_notation_of_22_million_l791_791371

theorem scientific_notation_of_22_million : 22_000_000 = 2.2 * 10^7 :=
sorry

end scientific_notation_of_22_million_l791_791371


namespace min_value_of_expression_l791_791591

theorem min_value_of_expression : ∃ x : ℝ, (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 :=
by
  sorry

end min_value_of_expression_l791_791591


namespace exercise_times_mode_and_median_l791_791427

noncomputable def exercise_times : List ℕ := [31, 33, 28, 31, 32, 28, 31]

def mode (l : List ℕ) : ℕ :=
let freq_map := l.frequency_map
freq_map.to_list.max_by (Prod.snd) |>.fst

def median (l : List ℕ) : ℕ :=
let sorted := List.sort (· <= ·) l
sorted.get! (sorted.length / 2)

theorem exercise_times_mode_and_median :
  mode exercise_times = 31 ∧ median exercise_times = 31 :=
by
  -- Proof steps go here
  sorry

end exercise_times_mode_and_median_l791_791427


namespace domain_of_f_l791_791238

noncomputable def f (x : ℝ) : ℝ := sqrt (4 - sqrt (7 - sqrt (x^2)))

theorem domain_of_f :
  {x : ℝ | f x ∈ ℝ} = set.Icc (-7 : ℝ) (7 : ℝ) :=
by
  sorry

end domain_of_f_l791_791238


namespace largest_divisor_of_n_l791_791519

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 7200 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l791_791519


namespace ellipse_semi_major_axis_and_eccentricity_l791_791305

theorem ellipse_semi_major_axis_and_eccentricity (m : ℝ) 
  (h₁ : ∀ x y : ℝ, x^2/m + y^2 = 1) 
  (h₂ : ∀ a e : ℝ, a = 2 * e) : 
  m = 2 ∨ m = 3 / 4 :=
begin
  sorry,
end

end ellipse_semi_major_axis_and_eccentricity_l791_791305


namespace number_of_possible_ticket_prices_l791_791562

-- Define the conditions
def ticket_price_11_total : ℕ := 90
def ticket_price_12_total : ℕ := 150

-- Define the problem statement
theorem number_of_possible_ticket_prices (x : ℕ) : 
    (∀ x : ℕ, (ticket_price_11_total % x = 0) ∧ (ticket_price_12_total % x = 0) → 
    (x ∣ ticket_price_11_total) ∧ (x ∣ ticket_price_12_total)) → 
    (finset.card (finset.filter (λ d, d ∣ ticket_price_11_total ∧ d ∣ ticket_price_12_total) (finset.range (gcd ticket_price_11_total ticket_price_12_total + 1))) = 8) :=
by
  intro h
  sorry

end number_of_possible_ticket_prices_l791_791562


namespace max_additional_payment_expected_value_difference_l791_791595

section EnergySupply

variable (a b c d e f : ℕ) -- readings
variable (t1 t2 t3 t4 t5 t6 : ℕ) -- tariffs
variable (p1 p2 p3 : ℕ → ℕ) -- pieces (consumption differentials)

-- condition for correctness of readings and tariffs
def valid_readings := a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f
def valid_tariffs := t1 < t2 ∧ t2 < t3 ∧ t3 < t4 ∧ t4 < t5 ∧ t5 < t6

-- Tariffs in rubles per kWh
def peak_tariff : ℕ := 403 -- 4.03 rubles/kWh (multiplied by 100)
def night_tariff : ℕ := 101 -- 1.01 rubles/kWh
def semi_peak_tariff : ℕ := 339 -- 3.39 rubles/kWh

-- Definition of total payment
def total_payment (a b c d e f : ℕ) (t1 t2 t3 : ℕ) : ℕ :=
  t1 * (d - c) + t2 * (e - d) + t3 * (f - e)

-- Given correct computation by the client
def client_payment : ℕ := 66072 -- 660.72 rubles (multiplied by 100 to avoid floating point)

-- Proving the maximum possible additional payment for March 2013
theorem max_additional_payment (h1 : valid_readings) (h2 : valid_tariffs) :
  max_total_payment := 105806 - client_payment :=
begin
  sorry
end

-- Proving the expected value of the difference
theorem expected_value_difference (h1 : valid_readings) (h2 : valid_tariffs) :
  expected_value := 1930 :=
begin
  sorry
end

end EnergySupply

end max_additional_payment_expected_value_difference_l791_791595


namespace find_a_l791_791313

def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 else x + 3

theorem find_a (a : ℝ) (h : f a + f 3 = 0) : a = -12 :=
by
  sorry

end find_a_l791_791313


namespace maxOccursAt2_l791_791672

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

theorem maxOccursAt2 {m : ℝ} :
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ f m) ∧ 0 ≤ m ∧ m ≤ 2 → (0 < m ∧ m ≤ 2) :=
sorry

end maxOccursAt2_l791_791672


namespace max_distance_patrol_l791_791235

theorem max_distance_patrol :
  (∀ (fuel_capacity distance : ℕ), (fuel_capacity = 150) → 
  (∀(boats : ℕ), (boats = 3) →
  (∀ (return n distance : ℕ),
  n * distance = fuel_capacity * 2 →
  n = boats →
  return = 75)) → 
  (distance = 50) →
  (max_distance : ℕ) →
  max_distance = distance + 75 →
  max_distance = 125) sorry

end max_distance_patrol_l791_791235


namespace find_diameter_l791_791874

-- Definitions based on conditions
variables {C : Type*} [metric_space C] [normed_group C] [normed_space ℝ C]
variables {A B : C} -- Chord points

def midpoint (x y : C) : C := (x + y) / 2

axiom distance_between_midpoints (AC_midpoint BC_midpoint : C) : dist AC_midpoint BC_midpoint = 10

-- Lean statement for the proof problem
theorem find_diameter (AC BC : C) (h_perpendicular : inner AC BC = 0)
  (M N : C) (h_M : M = midpoint A C) (h_N : N = midpoint B C) :
  dist M N = 10 → 2 * dist A B = 20 :=
by
  intros h_distance
  sorry

end find_diameter_l791_791874


namespace find_y_when_x_is_seven_l791_791137

noncomputable def y_value_when_x_equals_seven : ℝ :=
  let x := 7 in
  let xy_constant := 180 in
  xy_constant / x

-- Conditions
def inversely_proportional (x y : ℝ) : Prop := x * y = 180
def sum_is_28 (x y : ℝ) : Prop := x + y = 28
def difference_is_8 (x y : ℝ) : Prop := x - y = 8

-- Theorem to prove
theorem find_y_when_x_is_seven (x y : ℝ) :
  inversely_proportional x y →
  sum_is_28 x y →
  difference_is_8 x y →
  x = 7 →
  y = y_value_when_x_equals_seven :=
by
  intro h_inv_prop h_sum_is_28 h_diff_is_8 h_x_is_7
  sorry

end find_y_when_x_is_seven_l791_791137


namespace abs_inequality_range_l791_791347

theorem abs_inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := 
sorry

end abs_inequality_range_l791_791347


namespace is_possible_to_fill_grid_l791_791042

theorem is_possible_to_fill_grid :
  ∃ (G : Fin 8 × Fin 8 → ℕ), (∀ i, (∑ j, G (i, j)) = 8) ∧ (∀ j₁ j₂, j₁ ≠ j₂ → (∑ i, G (i, j₁)) ≠ (∑ i, G (i, j₂))) ∧ (∀ i j, G (i, j) = 0 ∨ G (i, j) = 1) :=
sorry

end is_possible_to_fill_grid_l791_791042


namespace length_BC_equals_8_l791_791407

noncomputable def circle_length_BC (r : ℝ) (α : ℝ) (cosα : ℝ) : ℝ :=
  2 * r * cosα

theorem length_BC_equals_8 {α : ℝ} (hα : α ≠ 0 ∧ α ≠ π) :
  let r := 6
      cosα := 2 / 3
  in circle_length_BC r α cosα = 8 := 
by
  sorry

end length_BC_equals_8_l791_791407


namespace min_C_proof_l791_791956

theorem min_C_proof :
  (∃ (C : ℝ), (∀ (a : Fin 1001 → ℝ) (h : (∀ i, a 0 = 0 ∧ ∀ j k, j < k → a j < a k)),
    min (a 0 ^ 2 + (1000:ℝ) ^ 2) ≤ C * (∑ i in Finset.range 1001, a i))) ∧
  (floor (100 * C) = 127) :=
sorry

end min_C_proof_l791_791956


namespace exists_N_divisible_by_k_l791_791380

-- Conditions
variables {k : ℕ} (hk : 0 < k)
variables {P : ℤ[X]} (hP : ∀ n : ℤ, P.eval n ∈ ℤ)

theorem exists_N_divisible_by_k :
  ∃ N : ℕ, 0 < N ∧ (finset.range N).sum (λ n : ℕ, P.eval (n + 1)) % k = 0 :=
sorry

end exists_N_divisible_by_k_l791_791380


namespace not_possible_equal_sum_numbering_l791_791771

theorem not_possible_equal_sum_numbering (n : ℕ) (O : Point) (vertices : fin n → Point)
  (sides_numbered : fin n → ℕ) (segments_numbered : fin n → ℕ) :
  n = 10 →
  (∀ i, 1 ≤ sides_numbered i ∧ sides_numbered i ≤ n) →
  (∀ j, 1 ≤ segments_numbered j ∧ segments_numbered j ≤ n) →
  (∀ i ≠ j, sides_numbered i ≠ sides_numbered j) →
  (∀ i ≠ j, segments_numbered i ≠ segments_numbered j) →
  ¬(∃ S : ℕ, ∀ k : fin n, (sides_numbered k + sides_numbered (next k) + segments_numbered k) = S) := 
by sorry

end not_possible_equal_sum_numbering_l791_791771


namespace sum_solution_eq_neg_half_l791_791051

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x - 3
  else if x < 3 then x / 2 + 1
  else 2 * x - 5

theorem sum_solution_eq_neg_half :
  let s := {x | f x = 0} in
  -3 ∈ s ∧ 5 / 2 ∈ s ∧ (∀ x ∈ s, x = -3 ∨ x = 5 / 2) → 
  ∑ x in s, x = -1 / 2 :=
by
  sorry

end sum_solution_eq_neg_half_l791_791051


namespace sheila_tue_thu_hours_l791_791779

def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def total_hours_mwf : ℕ := hours_mwf * days_mwf

def weekly_earnings : ℕ := 360
def hourly_rate : ℕ := 10
def earnings_mwf : ℕ := total_hours_mwf * hourly_rate

def earnings_tue_thu : ℕ := weekly_earnings - earnings_mwf
def hours_tue_thu : ℕ := earnings_tue_thu / hourly_rate

theorem sheila_tue_thu_hours : hours_tue_thu = 12 := by
  -- proof omitted
  sorry

end sheila_tue_thu_hours_l791_791779


namespace residue_of_7_pow_2023_mod_19_l791_791498

theorem residue_of_7_pow_2023_mod_19 : (7^2023) % 19 = 3 :=
by 
  -- The main goal is to construct the proof that matches our explanation.
  sorry

end residue_of_7_pow_2023_mod_19_l791_791498


namespace problem_statement_l791_791296

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem problem_statement (h_odd : is_odd f) (h_decr : is_decreasing f) (a b : ℝ) (h_ab : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 :=
by
  sorry

end problem_statement_l791_791296


namespace correct_statements_l791_791440

variables (prob_win_game : ℝ) (S_A S_B : ℝ)
-- Define the conditions
def condition_1 := prob_win_game = (1 / 100)
def condition_2 := false  -- A census for mental health (not practical, represented as false)
def condition_3 := false  -- A sampling survey for epidemic control (not practical, represented as false)
def condition_4 := (S_A^2 = 0.2) ∧ (S_B^2 = 0.5)
def is_stable (var_A var_B : ℝ) := var_A < var_B

-- Problem statement
theorem correct_statements :
  (∀ (cond1 cond2 cond3 cond4 : Prop),
    cond1 → cond2 → cond3 → cond4 →
    (condition_1 cond1) → (condition_2 cond2) → (condition_3 cond3) → (condition_4 cond4) →
    is_stable S_A S_B) :=
by
  intros cond1 cond2 cond3 cond4
  intros h_1 h_2 h_3 h_4
  sorry

end correct_statements_l791_791440


namespace saree_original_price_l791_791453

theorem saree_original_price
  (sale_price : ℝ)
  (P : ℝ)
  (h_discount : sale_price = 0.80 * P * 0.95)
  (h_sale_price : sale_price = 266) :
  P = 350 :=
by
  -- Proof to be completed later
  sorry

end saree_original_price_l791_791453


namespace weight_of_b_l791_791111

variable {a b c : ℝ}

theorem weight_of_b (h1 : (a + b + c) / 3 = 45)
                    (h2 : (a + b) / 2 = 40)
                    (h3 : (b + c) / 2 = 43) :
                    b = 31 := by
  sorry

end weight_of_b_l791_791111


namespace angle_of_inclination_vert_line_l791_791793

theorem angle_of_inclination_vert_line (x : ℝ) (h : x = -1) : 
  ∃ θ : ℝ, θ = 90 := 
by
  sorry

end angle_of_inclination_vert_line_l791_791793


namespace number_of_t_in_T_such_that_f_t_mod_8_eq_0_l791_791068

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 4

def T := { n : ℤ | 0 ≤ n ∧ n ≤ 50 }

theorem number_of_t_in_T_such_that_f_t_mod_8_eq_0 : 
  (∃ t ∈ T, f t % 8 = 0) = false := sorry

end number_of_t_in_T_such_that_f_t_mod_8_eq_0_l791_791068


namespace grains_of_rice_in_teaspoon_is_10_l791_791468

noncomputable def grains_of_rice_per_teaspoon : ℕ :=
  let grains_per_cup := 480
  let tablespoons_per_half_cup := 8
  let teaspoons_per_tablespoon := 3
  grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)

theorem grains_of_rice_in_teaspoon_is_10 : grains_of_rice_per_teaspoon = 10 :=
by
  sorry

end grains_of_rice_in_teaspoon_is_10_l791_791468


namespace positive_int_solution_is_perfect_square_l791_791400

variable (t n : ℤ)

theorem positive_int_solution_is_perfect_square (ht : ∃ n : ℕ, n > 0 ∧ n^2 + (4 * t - 1) * n + 4 * t^2 = 0) : ∃ k : ℕ, n = k^2 :=
  sorry

end positive_int_solution_is_perfect_square_l791_791400


namespace equality_of_a_l791_791060

noncomputable def a : ℝ := sorry
def i : ℂ := complex.I
def condition (a : ℝ) (i : ℂ) := ∥(a + i) / i∥ = 2

theorem equality_of_a (h1 : 0 < a) (h2 : a ∈ ℝ) (h3 : condition a i) : a = real.sqrt 3 :=
by
  sorry

end equality_of_a_l791_791060


namespace total_students_in_class_l791_791707

theorem total_students_in_class (R S : ℕ)
  (h1 : 2 + 12 * 1 + 12 * 2 + 3 * R = S * 2)
  (h2 : S = 2 + 12 + 12 + R) :
  S = 42 :=
by
  sorry

end total_students_in_class_l791_791707


namespace object_speed_approx_l791_791695

theorem object_speed_approx :
  ∃ (speed : ℝ), abs (speed - 27.27) < 0.01 ∧
  (∀ (d : ℝ) (t : ℝ)
    (m : ℝ), 
    d = 80 ∧ t = 2 ∧ m = 5280 →
    speed = (d / m) / (t / 3600)) :=
by 
  sorry

end object_speed_approx_l791_791695


namespace length_of_AC_l791_791366

-- Geometry definitions
variables (A B C D : Type) [metric_space A] 

-- Given conditions
def trapezoid_ABCD (A B C D : Point) (AB CD : Line) :=
    parallel AB CD ∧ perpendicular AC CD ∧ CD.length = 10

def tan_C (angle : ℝ) := 2
def tan_B (angle : ℝ) := 2.5

-- Problem statement
theorem length_of_AC
    (A B C D : Point)
    (AB CD : Line)
    (h1 : trapezoid_ABCD A B C D AB CD)
    (h2 : tan_C ∠C = 2)
    (h3 : tan_B ∠B = 2.5) :
    length AC = 20 :=
sorry

end length_of_AC_l791_791366


namespace unique_solution_values_l791_791322

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem unique_solution_values :
  { a : ℝ | let A := { x | a*x^2 - 2*x + 1 = 0 } in A.nonempty ∧ A.card = 1 } = {0, 1} :=
by
  sorry

end unique_solution_values_l791_791322


namespace num_sets_C_l791_791656

open Set

def A : Set ℤ := {-1}
def B : Set ℤ := {-1, 0, 1}

theorem num_sets_C (C : Set ℤ) (hA : A ⊆ C) (hB : C ⊆ B) : finset.card {C : Set ℤ | A ⊆ C ∧ C ⊆ B} = 4 := 
  sorry

end num_sets_C_l791_791656


namespace Fedya_arrival_time_l791_791476

noncomputable def travel_time_to_Palkino (dist_Yolkino_Palkino : ℕ) 
  (dist_Yolkino_oak : ℕ) 
  (dist_oak_Palkino : ℕ)
  (time12 : ℕ) 
  (time1240 : ℕ) 
  (constant_speed : ℕ) : ℕ :=
if h1 : dist_Yolkino_Palkino = 9 ∧ dist_Yolkino_oak = 3 ∧ dist_oak_Palkino = 6 
    ∧ time12 = 720 ∧ time1240 = 760 ∧ constant_speed > 0 then 790 else sorry

theorem Fedya_arrival_time : 
  travel_time_to_Palkino 9 3 6 720 760 1 = 790 :=
begin
  apply if_pos,
  split; try {norm_num},
  split; try {norm_num},
  split; try {norm_num},
  split; try {norm_num},
  split; try {norm_num},
  norm_num,
end

end Fedya_arrival_time_l791_791476


namespace norma_initial_cards_l791_791083

theorem norma_initial_cards (x : ℝ) 
  (H1 : x + 70 = 158) : 
  x = 88 :=
by
  sorry

end norma_initial_cards_l791_791083


namespace vertex_of_parabola_l791_791458

theorem vertex_of_parabola (c d : ℝ) (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  ∃ v : ℝ × ℝ, v = (1, 25) :=
sorry

end vertex_of_parabola_l791_791458


namespace seq_geometric_sum_first_n_terms_abs_seq_l791_791643

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 0 then -2 else 2 * seq n.pred + 4

theorem seq_geometric : ∀ n : ℕ, ∃ r : ℤ, seq n + 4 = r ^ n := by
  sorry

theorem sum_first_n_terms_abs_seq (n : ℕ) : ℤ :=
  let |seq n| :=
  2 ^ (n + 1) - 4 * n + 2

example {n : ℕ} : sum_first_n_terms_abs_seq n = 2 ^ (n + 1) - 4 * n + 2 := by
  sorry

end seq_geometric_sum_first_n_terms_abs_seq_l791_791643


namespace duration_of_time_l791_791511

variable (A B C : String)
variable {a1 : A = "Get up at 6:30"}
variable {b1 : B = "School ends at 3:40"}
variable {c1 : C = "It took 30 minutes to do the homework"}

theorem duration_of_time : C = "It took 30 minutes to do the homework" :=
  sorry

end duration_of_time_l791_791511


namespace length_of_PQ_l791_791493

-- Definitions for the problem conditions
variable (XY UV PQ : ℝ)
variable (hXY_fixed : XY = 120)
variable (hUV_fixed : UV = 90)
variable (hParallel : XY = UV ∧ UV = PQ) -- Ensures XY || UV || PQ

-- The statement to prove
theorem length_of_PQ : PQ = 360 / 7 := by
  -- Definitions for similarity ratios and solving steps can be assumed here
  sorry

end length_of_PQ_l791_791493


namespace unbalance_in_at_most_11_operations_l791_791823

-- Define the context of the problem
def Coin : Type := ℕ  -- represents coins by their weight (grams)

-- Define the initial balanced state
noncomputable def initial_coins (m n : ℕ) (h : m + n = 288) : list Coin :=
  list.replicate 144 m ++ list.replicate 144 n

-- Define the operation of swapping groups of coins
def swap (l r : list Coin) (k : ℕ) (i j : ℕ) : list Coin × list Coin :=
  let l1 := list.take k l in
  let l2 := list.drop k l in
  let r1 := list.take k r in
  let r2 := list.drop k r  in
  (l2 ++ r1, r2 ++ l1)

-- Define the property of the balance being unbalanced
def unbalanced (l r : list Coin) := list.sum l ≠ list.sum r

-- The main theorem
theorem unbalance_in_at_most_11_operations (m n : ℕ) (h : m + n = 288)
   (hl : l₀ = initial_coins m n h)
   (hr : r₀ = initial_coins n m h) 
   : ∃ (k : ℕ) (l r : list Coin), 1 ≤ k ∧ k ≤ 11 ∧ unbalanced l r :=
sorry

end unbalance_in_at_most_11_operations_l791_791823


namespace sin_two_y_plus_x_l791_791657

theorem sin_two_y_plus_x 
  (x y : ℝ) (k : ℤ)
  (hx : sin x = 1 / 3)
  (hxy : sin (x + y) = 1) :
  sin (2 * y + x) = 1 / 3 := sorry

end sin_two_y_plus_x_l791_791657


namespace lattice_triangle_inequality_l791_791136

theorem lattice_triangle_inequality
  (a b c R : ℝ)
  (is_lattice : (a, b, c) ∈ ℚ × ℚ × ℚ)
  (circumradius : ∃ (O : ℝ × ℝ), ∀ (P : (ℝ × ℝ)), P ∈ {(x, y) | (x, y) is vertex of the triangle} → dist O P = R) :
  a * b * c ≥ 2 * R :=
sorry

end lattice_triangle_inequality_l791_791136


namespace min_tangent_length_is_sqrt_31_l791_791910

noncomputable def minimum_tangent_length_to_circle : ℝ :=
  let line : ℝ × ℝ → Prop := λ p, p.snd = p.fst + 2
  let circle_center := (4, -2)
  let circle : ℝ × ℝ → Prop := λ p, (p.fst - 4) ^ 2 + (p.snd + 2) ^ 2 = 1
  let distance_from_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
    (|a * p.fst + b * p.snd + c|) / (Math.sqrt (a ^ 2 + b ^ 2))
  let distance := distance_from_point_to_line circle_center 1 (-1) 2
  let radius := Math.sqrt 1
  let tangent_length := Math.sqrt (distance ^ 2 - radius ^ 2)
  tangent_length

theorem min_tangent_length_is_sqrt_31 :
  minimum_tangent_length_to_circle = Real.sqrt 31 :=
sorry

end min_tangent_length_is_sqrt_31_l791_791910


namespace find_interest_rate_l791_791419

variable (P1 P2 P total_interest : ℝ)
variable (r : ℝ)

-- Given conditions
def condition1 := P1 + P2 = 3600
def condition2 := P1 = 1800
def condition3 := P2 = 3600 - 1800
def condition4 := total_interest = 144
def condition5 := total_interest = (P1 * r * 1) / 100 + (P2 * 5 * 1) / 100

-- Define what we need to prove
theorem find_interest_rate 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) 
  (h4 : condition4) 
  (h5 : condition5) 
  : r = 3 :=
by
  -- The proof would go here
  sorry

end find_interest_rate_l791_791419


namespace problem_statement_l791_791392

-- Define the function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : f 2 = 2
axiom h2 : ∀ x y : ℝ, f (x * y + f x ^ 2) = x * f y + f x

-- The theorem to prove
theorem problem_statement :
  let n := 1 in
  let s := 1 in
  n * s = 1 := 
sorry

end problem_statement_l791_791392


namespace vector_equation_l791_791879

variable {V : Type} [AddCommGroup V]

variables (A B C : V)

theorem vector_equation :
  (B - A) - 2 • (C - A) + (C - B) = (A - C) :=
by
  sorry

end vector_equation_l791_791879


namespace infinitely_many_solutions_l791_791782

-- Define the necessary conditions
variables (x y d a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_gt : a > b) (h_diff : a - b = 1)
noncomputable def proof_problem : Prop :=
  (3 * x^3 + x * y^2) * (x^2 * y + 3 * y^3) = (x - y)^7 ∧
  x = d * a ∧ y = d * b

theorem infinitely_many_solutions : ∃ (x y : ℕ), ∀ d a b, h_coprime → h_gt → h_diff → proof_problem x y d a b :=
sorry

end infinitely_many_solutions_l791_791782


namespace num_rooms_with_2_windows_l791_791972

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end num_rooms_with_2_windows_l791_791972


namespace problem_statement_l791_791395

def largest_odd_factor (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else largest_odd_factor (n / 2)

def f (k : ℕ) : ℕ :=
  (Finset.range (2^k)).sum largest_odd_factor

theorem problem_statement : f 2016 = (4/3) * 4^2015 - 1/3 := by
  sorry

end problem_statement_l791_791395


namespace power_inequality_l791_791290

variable {a b : ℝ}

theorem power_inequality (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := 
by sorry

end power_inequality_l791_791290


namespace part1_part2_l791_791187
open Topology Filter Real

-- Part 1: Prove that the limit of f(x) as x tends to infinity is 0
theorem part1 : tendsto (λ x : ℝ, x^2 / exp x) at_top (𝓝 0) :=
sorry

-- Part 2: Prove that the limit of the second complicated integral expression is e^2
theorem part2 : tendsto (λ k : ℝ, (1 / k) * ∫ x in 0..k, (1 + sin(2 * x))^(1 / x)) (𝓝 0) (𝓝 (exp 2)) :=
sorry

end part1_part2_l791_791187


namespace factorable_iff_k_zero_l791_791962

theorem factorable_iff_k_zero (k : ℤ) :
  (∃ a b c d : ℤ, (x² + 4 * x * y + 2 * x + k * y - k) = (a * x + b*y + c) * (d * x + b*y + d)) ↔ k = 0 :=
sorry

end factorable_iff_k_zero_l791_791962


namespace probability_of_point_within_distance_a_from_vertices_l791_791860

theorem probability_of_point_within_distance_a_from_vertices (a : ℝ) : 
  let area_curved_quadrilateral := a^2 * (π/3 - 2 * sqrt 2 * sin (π / 12));
  let area_square := a^2;
  let probability := area_curved_quadrilateral / area_square;
  probability = (π / 3) + 1 - sqrt 3 :=
begin
  sorry
end

end probability_of_point_within_distance_a_from_vertices_l791_791860


namespace smaller_circle_y_coordinate_l791_791483

theorem smaller_circle_y_coordinate 
  (center : ℝ × ℝ) 
  (P : ℝ × ℝ)
  (S : ℝ × ℝ) 
  (QR : ℝ)
  (r_large : ℝ):
    center = (0, 0) → P = (5, 12) → QR = 2 → S.1 = 0 → S.2 = k → r_large = 13 → k = 11 := 
by
  intros h_center hP hQR hSx hSy hr_large
  sorry

end smaller_circle_y_coordinate_l791_791483


namespace gratuity_percentage_l791_791205

open Real

theorem gratuity_percentage (num_bankers num_clients : ℕ) (total_bill per_person_cost : ℝ) 
    (h1 : num_bankers = 4) (h2 : num_clients = 5) (h3 : total_bill = 756) 
    (h4 : per_person_cost = 70) : 
    ((total_bill - (num_bankers + num_clients) * per_person_cost) / 
     ((num_bankers + num_clients) * per_person_cost)) = 0.2 :=
by 
  sorry

end gratuity_percentage_l791_791205


namespace sum_reciprocal_geq_n_squared_l791_791405

noncomputable theory

open Real

-- Define the necessary variables and conditions
variables {n : ℕ} (a : Fin n → ℝ)
hypothesis h_n_pos : 0 < n
hypothesis h_a_pos {i : Fin n} : 0 < a i
hypothesis h_sum_a : (∑ i, a i) = 1

-- State the theorem
theorem sum_reciprocal_geq_n_squared :
  ∑ i, (1 / a i) ≥ (n : ℝ)^2 :=
sorry

end sum_reciprocal_geq_n_squared_l791_791405


namespace has_unique_zero_in_interval_l791_791186

noncomputable def f (x : ℝ) : ℝ := 2 * x + x^3 - 2

theorem has_unique_zero_in_interval : ∃! x ∈ set.Ioo 0 1, f x = 0 :=
by
  sorry

end has_unique_zero_in_interval_l791_791186


namespace chocolate_milk_probability_l791_791418

theorem chocolate_milk_probability
  (visits_per_week : ℕ := 7)
  (chocolate_milk_probability : ℚ := 3/4)
  (bottled_chocolate_milk_days : ℕ := 5)
  (comb : ℕ → ℕ → ℕ := λ n k, Nat.choose n k)
  (binomial_prob : ℚ := (3/4) ^ 5 * (1/4) ^ 2)
  (total_combinations : ℕ := comb 7 5) :
  (total_combinations * binomial_prob = 5103 / 16384) :=
by
  sorry

end chocolate_milk_probability_l791_791418


namespace percent_increase_approx_58_13_l791_791909

noncomputable def percent_increase_sales 
  (sales_this_year_euro : ℕ) (sales_last_year_dollar : ℕ) 
  (exchange_rate_this_year : ℕ) (exchange_rate_last_year : ℕ)
  : ℚ :=
  let sales_this_year_dollar := sales_this_year_euro * exchange_rate_this_year in
  let increase := sales_this_year_dollar - sales_last_year_dollar in
  (increase.to_rat / sales_last_year_dollar.to_rat) * 100

theorem percent_increase_approx_58_13 :
  percent_increase_sales 460000000 320000000 110 115 ≈ 58.13 :=
sorry

end percent_increase_approx_58_13_l791_791909


namespace miami_logan_ratio_l791_791017

theorem miami_logan_ratio (total_passengers : Real) (kennedy_fraction : Real) 
    (miami_fraction : Real) (logan_passengers : Real) 
    (h1 : total_passengers = 38.3) 
    (h2 : kennedy_fraction = 1 / 3) 
    (h3 : miami_fraction = 1 / 2) 
    (h4 : logan_passengers = 1.5958333333333332) :
    let kennedy_passengers := total_passengers * kennedy_fraction
    let miami_passengers := kennedy_passengers * miami_fraction
    miami_passengers / logan_passengers ≈ 4 :=
by
  sorry

end miami_logan_ratio_l791_791017


namespace second_quadrant_set_l791_791602

-- Define the set P of points in the second quadrant
def P : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 }

-- Statement of the problem: Prove that this definition accurately describes the set of all points in the second quadrant
theorem second_quadrant_set :
  P = { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } :=
by
  sorry

end second_quadrant_set_l791_791602


namespace cakes_bought_l791_791930

theorem cakes_bought (initial_cakes remaining_cakes : ℕ) (h_initial : initial_cakes = 155) (h_remaining : remaining_cakes = 15) : initial_cakes - remaining_cakes = 140 :=
by {
  sorry
}

end cakes_bought_l791_791930


namespace time_for_Dawson_l791_791236

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end time_for_Dawson_l791_791236


namespace equal_distances_l791_791479

-- Define the geometrical elements and conditions of the problem.
variables {A B C M K N O₁ O₂ : Type*}

-- Assume M is the midpoint of AC
def midpoint (A C M : Type*) : Prop := (dist A M) = (dist C M)

-- Assume BM is a median (thus M is the midpoint of AC)
variable (BM : median B M)

-- Define the perpendicular line to BM through B intersects the altitudes from A and C at K and N
def perpendicular_to_median (BM : median B M) (K N : Type*) : Prop := 
  ⊥(BM, line.through B K) ∧ ⊥(BM, line.through B N)

-- Points O₁ and O₂ are centers of circumcircles of triangles ABK and CBN respectively
def center_of_circumcircle (O₁ : Type*) (ABK : triangle A B K) : Prop := 
  O₁ = circumcenter ABK

def center_of_circumcircle (O₂ : Type*) (CBN : triangle C B N) : Prop := 
  O₂ = circumcenter CBN

theorem equal_distances {A B C M K N O₁ O₂ : Type*} 
  (h1 : midpoint A C M) 
  (h2 : perpendicular_to_median BM K N)
  (h3 : center_of_circumcircle O₁ (triangle.mk A B K))
  (h4 : center_of_circumcircle O₂ (triangle.mk C B N)) :
  dist O₁ M = dist O₂ M :=
sorry

end equal_distances_l791_791479


namespace yellowBeanRatio_24_percent_l791_791621

def bagA := 20
def bagB := 25
def bagC := 35
def bagD := 40

def yellowRatioA := 0.40
def yellowRatioB := 0.30
def yellowRatioC := 0.25
def yellowRatioD := 0.10

def yellowBeansA := bagA * yellowRatioA
def yellowBeansB := bagB * yellowRatioB
def yellowBeansC := bagC * yellowRatioC
def yellowBeansD := bagD * yellowRatioD

def totalYellowBeans := yellowBeansA + yellowBeansB + yellowBeansC + yellowBeansD
def totalBeans := bagA + bagB + bagC + bagD

def yellowToTotalRatio := totalYellowBeans / totalBeans

theorem yellowBeanRatio_24_percent : yellowToTotalRatio = 0.24 :=
  sorry

end yellowBeanRatio_24_percent_l791_791621


namespace problem_statement_l791_791693

noncomputable def S : ℝ := ∑ k in Finset.range 99, (λ k, (-1)^(k+1) / (Real.sqrt k * Real.sqrt (k+1) * (Real.sqrt (k+1) - Real.sqrt k))) (k + 1)

theorem problem_statement : 1000 * S = 1100 :=
by
  sorry

end problem_statement_l791_791693


namespace max_area_of_triangle_OPQ_is_24_div_25_l791_791648

-- Define the conditions for the ellipse
def a : ℝ := 2
def b : ℝ := 1
def c : ℝ := sqrt 3
def eccentricity : ℝ := c / a

-- Define the standard form of the ellipse and the given point
def ellipse_standard_form (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def point_on_ellipse : Prop := ellipse_standard_form 1 (sqrt 3 / 2)

-- Define the orthogonality condition
def orthogonal_vectors (x_1 y_1 x_2 y_2 : ℝ) : Prop :=
  let AP := (x_1 - a, y_1) in
  let AQ := (x_2 - a, y_2) in
  Prod.fst AP * Prod.fst AQ + Prod.snd AP * Prod.snd AQ = 0

-- Define the area of the triangle OPQ
def triangle_area (x_1 y_1 x_2 y_2 : ℝ) : ℝ :=
  0.5 * abs (x_1 * y_2 - x_2 * y_1)

-- A noncomputable version may be combined with the maximum calculation if necessary
noncomputable def max_triangle_area (x_1 y_1 x_2 y_2 : ℝ) : ℝ :=
  max (triangle_area x_1 y_1 x_2 y_2) (triangle_area x_1 y_1 x_2 y_2)

-- The main theorem statement to be proved
theorem max_area_of_triangle_OPQ_is_24_div_25 :
  ∀ (x_1 y_1 x_2 y_2 : ℝ),
    ellipse_standard_form x_1 y_1 →
    ellipse_standard_form x_2 y_2 →
    (x_1, y_1) ≠ (x_2, y_2) →
    orthogonal_vectors x_1 y_1 x_2 y_2 →
    a > b ∧ b > 0 →
    point_on_ellipse →
    eccentricity = sqrt 3 / 2 →
    max_triangle_area x_1 y_1 x_2 y_2 = 24 / 25 :=
by sorry

end max_area_of_triangle_OPQ_is_24_div_25_l791_791648


namespace length_of_tunnel_l791_791913

theorem length_of_tunnel
    (length_of_train : ℕ)
    (speed_kmh : ℕ)
    (crossing_time_seconds : ℕ)
    (distance_covered : ℕ)
    (length_of_tunnel : ℕ) :
    length_of_train = 1200 →
    speed_kmh = 96 →
    crossing_time_seconds = 90 →
    distance_covered = (speed_kmh * 1000 / 3600) * crossing_time_seconds →
    length_of_train + length_of_tunnel = distance_covered →
    length_of_tunnel = 6000 :=
by
  sorry

end length_of_tunnel_l791_791913


namespace hyperbola_standard_eq_obtuse_triangle_l791_791638

-- Define the given conditions and prove the equivalency.
def foci_of_ellipse (a b : ℝ) := [(a, 0), (-a, 0)]

def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def passes_through (H : ℝ → ℝ → Prop) (p q : ℝ) := H p q

def condition1 := passes_through (hyperbola 3 (√2)) 3 (-2)

def condition2 := foci_of_ellipse (√5) 2

def MF_condition (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) := 
    |M.1 - F1.1 + M.2 - F1.2| + |M.1 - F2.1 + M.2 - F2.2| = 6 * √3

theorem hyperbola_standard_eq : ∀ a b : ℝ, 
  let H := hyperbola a b
  in condition1 → H 3 (-2) ∧ a = 3 ∧ b = 2 := by
    sorry

theorem obtuse_triangle : ∀ M F1 F2 : ℝ × ℝ, 
  MF_condition M F1 F2 →
  |M.1 - F1.1 + M.2 - F1.2| - |M.1 - F2.1 + M.2 - F2.2| = 2 * √3 →
  ∃ (θ : ℝ), θ > pi/2 := by
    sorry

end hyperbola_standard_eq_obtuse_triangle_l791_791638


namespace draw_ways_l791_791826

open Nat

theorem draw_ways (whiteBalls blackBalls : ℕ) (draws : ℕ) (ways : ℕ) :
  whiteBalls = 6 → blackBalls = 4 → draws = 4 → ways = 
  (choose blackBalls 2) * (choose whiteBalls 2) + 
  (choose blackBalls 3) * (choose whiteBalls 1) + 
  (choose blackBalls 4) → ways = 115 :=
by {
  intros hwhite hblack hdraws hways,
  rw [hwhite, hblack, hdraws, hways],
  sorry,
}

end draw_ways_l791_791826


namespace domain_of_function_l791_791119

def domain_of_f (x: ℝ) : Prop :=
  (x > -2 ∧ x ≠ 1/2)

theorem domain_of_function :
  {x : ℝ | domain_of_f x } = 
  {x : ℝ | x ∈ set.Ioo (-2 : ℝ) (1/2 : ℝ)} ∪ {x : ℝ | x ∈ set.Ioi (1/2 : ℝ)} :=
by
  sorry

end domain_of_function_l791_791119


namespace chord_length_tangent_to_smaller_circle_l791_791434

-- Defining the conditions
def radius_smaller_circle : ℝ := 3
def area_ring : ℝ := 24 * Real.pi
def larger_circle_radius (a : ℝ) : Prop := a^2 - radius_smaller_circle^2 = 24

-- Theorem statement: The length of the chord of the larger circle tangent to the smaller circle
theorem chord_length_tangent_to_smaller_circle
  (a : ℝ) (h : larger_circle_radius a) :
  ∃ (c : ℝ), a^2 = 33 ∧ radius_smaller_circle = 3 ∧ c = 4 * Real.sqrt 6 :=
by
  -- Hypotheses and conditions
  have ha2 : a^2 = 33 := sorry
  have rb : radius_smaller_circle = 3 := sorry

  -- Length of the chord
  have hc : c = 4 * Real.sqrt 6 := sorry
  exact ⟨c, ha2, rb, hc⟩

end chord_length_tangent_to_smaller_circle_l791_791434


namespace concyclicity_equivalence_l791_791057

variables {A B C D H N O' : Point} {a b c R : Real}
variables (triangle_ABC : Triangle A B C) 
variables (orthocenter_H : Orthocenter H triangle_ABC)
variables (circumcenter_O_prime : Circumcenter O' (triangle.triangle_of_orthocircle_triangle orthocenter_H triangle_ABC))
variables (midpoint_N : Midpoint N (segment (line_segment A O')))
variables (reflection_D : Reflection D N (line (segment B C)))
variables (circumradius_R : CircumcircleRadius R triangle_ABC)

theorem concyclicity_equivalence :
  (Cyclic_Quad A B D C) ↔ (b^2 + c^2 - a^2 = 3 * R^2) := 
sorry

end concyclicity_equivalence_l791_791057


namespace flagpole_height_l791_791402

theorem flagpole_height (x : ℝ) (h1 : (x + 2)^2 = x^2 + 6^2) : x = 8 := 
by 
  sorry

end flagpole_height_l791_791402


namespace solve_logarithmic_equation_l791_791426

noncomputable def x_solution (x : ℝ) : Prop :=
  log 3 x + log 9 (x ^ 3) = 9

theorem solve_logarithmic_equation (x : ℝ) (h : x_solution x) : x = 3 ^ (18 / 5) :=
by sorry

end solve_logarithmic_equation_l791_791426


namespace count_one_inch_cubes_with_two_faces_painted_l791_791894

/-- A four-inch wooden cube is painted blue on all six faces. The cube is then cut into one-inch 
cubes. -/
def four_inch_cube_painted_and_cut 
  (cube_size : ℕ) (smaller_cube_size : ℕ) 
  (painted_faces : ℕ) (n : ℕ) : Prop :=
cube_size = 4 ∧ smaller_cube_size = 1 ∧ painted_faces = 6 ∧ n = 4 * 4 * 4

theorem count_one_inch_cubes_with_two_faces_painted
  (cube_size painted_faces smaller_cube_size n : ℕ)
  (h : four_inch_cube_painted_and_cut cube_size smaller_cube_size painted_faces n) :
  ∃ (cubes_with_two_faces_painted : ℕ), cubes_with_two_faces_painted = 40 :=
begin
  use 40,
  sorry
end

end count_one_inch_cubes_with_two_faces_painted_l791_791894


namespace expression_value_eq_16_point_2_l791_791989

def greatestIntegerLessThanOrEqual (x : ℝ) : ℤ := Int.floor x

theorem expression_value_eq_16_point_2 : 
  ∀ x = 6.5, 
  (greatestIntegerLessThanOrEqual x * greatestIntegerLessThanOrEqual (2/3)
  + greatestIntegerLessThanOrEqual 2 * 7.2
  + greatestIntegerLessThanOrEqual 8.4 - 6.2) = 16.2 :=
by
  intro h
  rw [←h]
  sorry

end expression_value_eq_16_point_2_l791_791989


namespace domain_of_fx_l791_791343

theorem domain_of_fx :
  {x : ℝ | x ≥ 1 ∧ x^2 < 2} = {x : ℝ | 1 ≤ x ∧ x < Real.sqrt 2} := by
sorry

end domain_of_fx_l791_791343


namespace proof_problem_l791_791064

open Set

def floor_fun (x : ℝ) := Int.floor x

def prop_1 := {x : ℝ | x^2 - floor_fun x - 1 = 0 ∧ -1 < x ∧ x < 2}
def is_singleton (s : Set α) := ∃ y, ∀ x, x ∈ s ↔ x = y

theorem proof_problem :
  (is_singleton prop_1) ∧ (∀ x : ℝ, floor_fun x + floor_fun (x + 1/2) = floor_fun (2 * x)) := 
by 
  sorry

end proof_problem_l791_791064


namespace count_one_inch_cubes_with_two_faces_painted_l791_791895

/-- A four-inch wooden cube is painted blue on all six faces. The cube is then cut into one-inch 
cubes. -/
def four_inch_cube_painted_and_cut 
  (cube_size : ℕ) (smaller_cube_size : ℕ) 
  (painted_faces : ℕ) (n : ℕ) : Prop :=
cube_size = 4 ∧ smaller_cube_size = 1 ∧ painted_faces = 6 ∧ n = 4 * 4 * 4

theorem count_one_inch_cubes_with_two_faces_painted
  (cube_size painted_faces smaller_cube_size n : ℕ)
  (h : four_inch_cube_painted_and_cut cube_size smaller_cube_size painted_faces n) :
  ∃ (cubes_with_two_faces_painted : ℕ), cubes_with_two_faces_painted = 40 :=
begin
  use 40,
  sorry
end

end count_one_inch_cubes_with_two_faces_painted_l791_791895


namespace possible_values_of_k_l791_791144

theorem possible_values_of_k :
  let k_values_count (k : ℕ) := 7 ≤ k ∧ k ≤ 2013 ∧ ∃ t : ℕ, t ≠ 0 ∧ 2 * t % 1000 = 6 * t % 1000 ∧ 6 * t % 1000 = k * t % 1000 
  in finset.card (finset.filter k_values_count (finset.range 2014)) = 501 :=
by {
  sorry
}

end possible_values_of_k_l791_791144


namespace true_statements_l791_791185

theorem true_statements (α β : Plane) (l1 l2 : Line) (h1 : AreIntersectingLines α l1 l2) (h2 : IsParallelLines l1 β l3) (h3 : IsParallelLines l2 β l4) (h4 : LineNotInPlane l α) (h5 : IsParallelLines l l5) (h6: PlaneIntersectAtLine α β l) (h7 : IsPerpendicularLineToPlan l l6 α) (h8: IsPerpendicularLineToPlan l l7 α) :
  (if ((h1 ∧ h2 ∧ h3) → IsParallelPlanes α β) ∧ ((h4 ∧ h5) → IsParallelLineToPlane l α) then ⟨1, 2⟩ else false) :=
  sorry

end true_statements_l791_791185


namespace find_a_l791_791320

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem find_a (a : ℝ) (h_intersect : ∃ x₀, f a x₀ = g x₀) (h_tangent : ∃ x₀, (f a x₀) = g x₀ ∧ (1/x₀ * a = 1/ (2 * Real.sqrt x₀))):
  a = Real.exp 1 / 2 :=
by
  sorry

end find_a_l791_791320


namespace distance_top_tennis_ball_to_floor_l791_791833

noncomputable def radius_volleyball := 18
noncomputable def radius_tennis_ball := 6

theorem distance_top_tennis_ball_to_floor :
  let r_v := radius_volleyball,
      r_t := radius_tennis_ball,
      height_of_x := 3 * r_v / Math.sqrt(3) + r_t  -- distance from O to A (12√3), plus radius of tennis ball
  in height_of_x = 36 :=
by
  let r_v := radius_volleyball,
      r_t := radius_tennis_ball
  let height_of_o := r_v
  let height_of_d := height_of_o + 12  -- distance OD in the solution steps
  let height_of_x := height_of_d + r_t -- 6 units above D
  show height_of_x = 36 from sorry

end distance_top_tennis_ball_to_floor_l791_791833


namespace obtain_x_squared_obtain_xy_l791_791478

variable (x y : ℝ)

/- 
Definition of steps and end of steps need to be defined in Lean
Given (x, y, 1) and allowed operations, prove that a step result can reach x^2 or xy
-/

-- For part (a): Prove that x^2 can be obtained
theorem obtain_x_squared : ∃ s, (s.steps x y 1 ∧ s.end = x^2) :=
sorry

-- For part (b): Prove that xy can be obtained
theorem obtain_xy : ∃ s, (s.steps x y 1 ∧ s.end = xy) :=
sorry

end obtain_x_squared_obtain_xy_l791_791478


namespace number_of_perpendicular_faces_l791_791032

-- Definitions for the problem setup
constant Point : Type
constant Rectangle : Type
constant Face : Type
constant perpendicular_faces : Face → Face → Prop
constant A B C D P : Point
constant cone : Point → Rectangle → Type
constant base_rectangle : Rectangle
constant vertex_of_cone : Point
constant perpendicular_to_base : Point → Rectangle → Prop

-- Conditions to set up the problem in Lean
axiom base_is_rectangle : Rectangle
axiom P_is_vertex : Point
axiom is_perpendicular : perpendicular_to_base P base_is_rectangle

-- Statement to prove the number of pairs of perpendicular faces
theorem number_of_perpendicular_faces (P : Point) (base : Rectangle) :
  perpendicular_to_base P base →
  (number_of_pairs_of_perpendicular_faces : ℕ) :=
begin
  sorry
end

end number_of_perpendicular_faces_l791_791032


namespace geometric_sequence_limit_l791_791368

open_locale big_operators

noncomputable def a (n : ℕ) : ℝ :=
if n = 0 then 8 else 8 * (1/2)^(n-1)

theorem geometric_sequence_limit
  (a : ℕ → ℝ)
  (h1 : a 1 * a 2 = 32)
  (h2 : a 3 * a 4 = 2)
  : ∃ l : ℝ, l = 16 ∨ l = -16 ∧ ∃ l', l' = ∑ i in finset.range(n), a(i) := 
begin
  sorry,
end

end geometric_sequence_limit_l791_791368


namespace socks_problem_l791_791721

def number_of_same_color_pairs (white black red : ℕ) : ℕ :=
  combinatorics.nat.choose white 2 + combinatorics.nat.choose black 2 + combinatorics.nat.choose red 2

theorem socks_problem : 
  number_of_same_color_pairs 5 4 3 = 19 :=
by
  sorry

end socks_problem_l791_791721


namespace intersection_P_Q_l791_791680

open Set

noncomputable def P : Set ℝ := {x | abs (x - 1) < 4}
noncomputable def Q : Set ℝ := {x | ∃ y, y = Real.log (x + 2) }

theorem intersection_P_Q :
  (P ∩ Q) = {x : ℝ | -2 < x ∧ x < 5} :=
by
  sorry

end intersection_P_Q_l791_791680


namespace correct_option_is_A_l791_791416

-- Define the conditions
def chromosome_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 2
  else if phase = "metaphase" then 2
  else if phase = "anaphase" then if is_meiosis then 2 else 4
  else if phase = "telophase" then if is_meiosis then 1 else 2
  else 0

def dna_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 4
  else if phase = "metaphase" then 4
  else if phase = "anaphase" then 4
  else if phase = "telophase" then 2
  else 0

def chromosome_behavior (phase : String) (is_meiosis : Bool) : String :=
  if is_meiosis && phase = "prophase" then "synapsis"
  else if is_meiosis && phase = "metaphase" then "tetrad formation"
  else if is_meiosis && phase = "anaphase" then "separation"
  else if is_meiosis && phase = "telophase" then "recombination"
  else "no special behavior"

-- Problem statement in terms of a Lean theorem
theorem correct_option_is_A :
  ∀ (phase : String),
  (chromosome_counts phase false = chromosome_counts phase true ∧
   chromosome_behavior phase false ≠ chromosome_behavior phase true ∧
   dna_counts phase false ≠ dna_counts phase true) →
  "A" = "A" :=
by 
  intro phase 
  simp only [imp_self]
  sorry

end correct_option_is_A_l791_791416


namespace exists_good_pair_for_every_m_l791_791076

open Nat

def is_good_pair (m n : ℕ) : Prop :=
  (m * n) = k * k ∧ ((m + 1) * (n + 1)) = l * l

theorem exists_good_pair_for_every_m (m : ℕ) : ∃ n : ℕ, n > m ∧ is_good_pair m n := 
  sorry

end exists_good_pair_for_every_m_l791_791076


namespace segments_length_1_to_8_not_possible_segments_distinct_integers_possible_l791_791710

theorem segments_length_1_to_8_not_possible :
  ∀ (lines : list (set (ℝ × ℝ))),
  lines.length = 4 ∧
  (∀ (l1 l2 l3 : set (ℝ × ℝ)), 
    l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬ (∃ (p : ℝ × ℝ), p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)) ∧
  (∀ (l1 l2 : set (ℝ × ℝ)), 
    l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → (∃ (p : ℝ × ℝ), p ∈ l1 ∧ p ∈ l2)) →
  ¬ (∃ (lengths : list ℝ), lengths = [1, 2, 3, 4, 5, 6, 7, 8]) :=
sorry

theorem segments_distinct_integers_possible :
  ∀ (lines : list (set (ℝ × ℝ))),
  lines.length = 4 ∧
  (∀ (l1 l2 l3 : set (ℝ × ℝ)), 
    l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬ (∃ (p : ℝ × ℝ), p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)) ∧
  (∀ (l1 l2 : set (ℝ × ℝ)), 
    l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → (∃ (p : ℝ × ℝ), p ∈ l1 ∧ p ∈ l2)) →
  ∃ (lengths : list ℝ), (∀ (i j : ℕ), i ≠ j → i < lengths.length → j < lengths.length → (lengths.nth i).get_or_else 0 ≠ (lengths.nth j).get_or_else 0) :=
sorry

end segments_length_1_to_8_not_possible_segments_distinct_integers_possible_l791_791710


namespace parabola_focus_l791_791610

/-- 
  Given the parabola defined by y = -2x^2 - 4x + 1,
  prove that its focus is (-1, 23/8).
--/
theorem parabola_focus : 
  let y := λ x : ℝ, -2 * x ^ 2 - 4 * x + 1 in 
  ∃ (F : ℝ × ℝ), F = (-1, 23 / 8) ∧ 
  (∃ (h k a : ℝ), y = λ x, a * (x + h) ^ 2 + k ∧ 
  a = -2 ∧ h = -1 ∧ k = 3 ∧ 
  F = (h, k - 1 / (4 * |a|)))
:=
by 
  sorry

end parabola_focus_l791_791610


namespace unique_prime_sum_8_l791_791443
-- Import all necessary mathematical libraries

-- Prime number definition
def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Function definition for f(y), number of unique ways to sum primes to form y
def f (y : Nat) : Nat :=
  if y = 8 then 2 else sorry -- We're assuming the correct answer to state the theorem; in a real proof, we would define this correctly.

theorem unique_prime_sum_8 :
  f 8 = 2 :=
by
  -- The proof goes here, but for now, we leave it as a placeholder.
  sorry

end unique_prime_sum_8_l791_791443


namespace perimeter_of_one_rectangle_l791_791460

-- Define the conditions
def is_divided_into_congruent_rectangles (s : ℕ) : Prop :=
  ∃ (height width : ℕ), height = s ∧ width = s / 4

-- Main proof statement
theorem perimeter_of_one_rectangle {s : ℕ} (h₁ : 4 * s = 144)
  (h₂ : is_divided_into_congruent_rectangles s) : 
  ∃ (perimeter : ℕ), perimeter = 90 :=
by 
  sorry

end perimeter_of_one_rectangle_l791_791460


namespace money_lent_to_B_l791_791176

theorem money_lent_to_B (total_money : ℕ) (interest_A_rate : ℚ) (interest_B_rate : ℚ) (interest_difference : ℚ) (years : ℕ) 
  (x y : ℚ) 
  (h1 : total_money = 10000)
  (h2 : interest_A_rate = 0.15)
  (h3 : interest_B_rate = 0.18)
  (h4 : interest_difference = 360)
  (h5 : years = 2)
  (h6 : y = total_money - x)
  (h7 : ((x * interest_A_rate * years) = ((y * interest_B_rate * years) + interest_difference))) : 
  y = 4000 := 
sorry

end money_lent_to_B_l791_791176


namespace concyclic_quadrilateral_ABDC_l791_791632

variables {A B C D P : Type*}
variable [euclidean_geometry A]

-- Define our points and the conditions described
variables {P_inside_triangle : inside_triangle P A B C}
variables {angle_ABP_eq_angle_CBD : ∠ A B P = ∠ C B D}
variables {angle_ACP_eq_angle_BCD : ∠ A C P = ∠ B C D}
variables {AP_eq_DP : segment A P = segment D P}

-- Define the theorem that states A, B, D, C are concyclic.
theorem concyclic_quadrilateral_ABDC :
  is_concyclic A B D C := 
begin
  sorry -- Proof goes here
end

end concyclic_quadrilateral_ABDC_l791_791632


namespace combined_area_correct_l791_791901

-- Define the given dimensions and border width
def length : ℝ := 0.6
def width : ℝ := 0.35
def border_width : ℝ := 0.05

-- Define the area of the rectangle, the new dimensions with the border, 
-- and the combined area of the rectangle and the border
def rectangle_area : ℝ := length * width
def new_length : ℝ := length + 2 * border_width
def new_width : ℝ := width + 2 * border_width
def combined_area : ℝ := new_length * new_width

-- The statement we want to prove
theorem combined_area_correct : combined_area = 0.315 := by
  sorry

end combined_area_correct_l791_791901


namespace sqrt_sum_ineq_l791_791630

theorem sqrt_sum_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (real.sqrt (a / (a + 3 * b)) + real.sqrt (b / (b + 3 * a)) ≥ 1) :=
sorry

end sqrt_sum_ineq_l791_791630


namespace triangle_to_pentagon_ratio_l791_791904

theorem triangle_to_pentagon_ratio (t p : ℕ) 
  (h1 : 3 * t = 15) 
  (h2 : 5 * p = 15) : (t : ℚ) / (p : ℚ) = 5 / 3 :=
by
  sorry

end triangle_to_pentagon_ratio_l791_791904


namespace correct_option_l791_791166

-- Definitions based on the problem's conditions
def option_A (x : ℝ) : Prop := x^2 * x^4 = x^8
def option_B (x : ℝ) : Prop := (x^2)^3 = x^5
def option_C (x : ℝ) : Prop := x^2 + x^2 = 2 * x^2
def option_D (x : ℝ) : Prop := (3 * x)^2 = 3 * x^2

-- Theorem stating that out of the given options, option C is correct
theorem correct_option (x : ℝ) : option_C x :=
by {
  sorry
}

end correct_option_l791_791166


namespace final_answer_l791_791072

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def pointA : ℝ × ℝ := (14, 10)
def pointB : ℝ × ℝ := (4, 6)

def pointC : ℝ × ℝ := midpoint pointA pointB

def calculation (C : ℝ × ℝ) : ℝ :=
  2 * C.1 - 3 * C.2

theorem final_answer (C : ℝ × ℝ) (hC : C = midpoint pointA pointB) : 
  calculation C = -6 :=
by
  rw [hC]
  simp [midpoint, calculation]
  norm_num
  sorry

end final_answer_l791_791072


namespace length_of_steel_cable_l791_791681

-- Conditions
variable {x y z : ℝ}
variable h1 : x + y + z = 8
variable h2 : x * y + y * z + z * x = 14

-- Theorem to prove the length of the steel cable
theorem length_of_steel_cable (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x * y + y * z + z * x = 14) : 
  l = 4 * Real.pi * Real.sqrt (11 / 3) :=
by
  sorry

end length_of_steel_cable_l791_791681


namespace find_p_l791_791334

theorem find_p (h p : Polynomial ℝ) 
  (H1 : h + p = 3 * X^2 - X + 4)
  (H2 : h = X^4 - 5 * X^2 + X + 6) : 
  p = -X^4 + 8 * X^2 - 2 * X - 2 :=
sorry

end find_p_l791_791334


namespace unique_balance_point_exists_l791_791662

def is_balance_point (f : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  f P.1 = P.2

theorem unique_balance_point_exists (f : ℝ → ℝ) (m : ℝ) :
  (f = λ x, (m-1)*x^2 - 3*x + 2*m) →
  (∃! t : ℝ, is_balance_point f (t, t)) ↔ m = 2 ∨ m = -1 ∨ m = 1 :=
by
  intros,
  sorry

end unique_balance_point_exists_l791_791662


namespace even_and_monotonically_increasing_l791_791917

/-- Check if a function is even --/
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- Check if a function is monotonically increasing on (0, +∞) --/
def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, (0 < x ∧ x < y) → f x < f y

/-- Define the candidate functions --/
def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := Real.log (abs x)
def f3 (x : ℝ) : ℝ := -x^2
def f4 (x : ℝ) : ℝ := 2^x

/-- The statement of the proof problem in Lean 4 --/
theorem even_and_monotonically_increasing :
  (isEven f2 ∧ isMonotonicallyIncreasing (λ x, if x > 0 then f2 x else f2 (-x))) ∧
  ¬(isEven f1 ∧ isMonotonicallyIncreasing (λ x, if x > 0 then f1 x else f1 (-x))) ∧
  ¬(isEven f3 ∧ isMonotonicallyIncreasing (λ x, if x > 0 then f3 x else f3 (-x))) ∧
  ¬(isEven f4 ∧ isMonotonicallyIncreasing (λ x, if x > 0 then f4 x else f4 (-x))) :=
sorry

end even_and_monotonically_increasing_l791_791917


namespace find_min_n_and_m_l791_791293

-- Given conditions
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : ∃ p : ℕ, ∀ x : ℝ, x^2 + m * x + (10 + n) = (x + p)^2)

-- Statement: prove that n = 6 and m = 8 or m = -8 given the conditions.
theorem find_min_n_and_m (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ p : ℕ, ∀ x : ℝ, x^2 + m * x + (10 + n) = (x + p)^2) :
  n = 6 ∧ (m = 8 ∨ m = -8) := by
  sorry

end find_min_n_and_m_l791_791293


namespace num_ordered_triples_of_sets_l791_791981

open Set

def ordered_triples_sets_satisfying_conditions : ℕ := 
  let universe := ({1, 2, 3, 4} : Set ℕ)
  let ordered_triples := { Triple | ∃ (T1 T2 T3 : Set ℕ), 
                                     T1 ⊆ universe ∧ 
                                     T2 ⊆ universe ∧ 
                                     T3 ⊆ universe ∧ 
                                     T1 ⊆ T2 ∪ T3 ∧ 
                                     T2 ⊆ T1 ∪ T3 ∧ 
                                     T3 ⊆ T1 ∪ T2 } 
  Triple.univ.finite_to_set.card

theorem num_ordered_triples_of_sets : ordered_triples_sets_satisfying_conditions = 625 := 
sorry

end num_ordered_triples_of_sets_l791_791981


namespace sum_of_reciprocals_implies_odd_l791_791781

def odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_reciprocals (n : ℕ) (s : fin n → ℕ) : Prop :=
  ∑ i, (1 : ℚ) / s i = 1

theorem sum_of_reciprocals_implies_odd (k : ℕ) (s : fin k → ℕ) (h₁ : ∀ i, odd (s i))
  (h₂ : sum_of_reciprocals k s) : odd k :=
sorry

end sum_of_reciprocals_implies_odd_l791_791781


namespace sum_series_l791_791945

noncomputable def f (n : ℕ) : ℝ :=
  (6 * (n : ℝ)^3 - 3 * (n : ℝ)^2 + 2 * (n : ℝ) - 1) / 
  ((n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ)^2 + (n : ℝ) + 1) * ((n : ℝ)^2 - (n : ℝ) + 1))

theorem sum_series:
  (∑' n, if h : 2 ≤ n then f n else 0) = 1 := 
by
  sorry

end sum_series_l791_791945


namespace triangle_area_is_correct_l791_791704

variables (A B C : ℝ) (a b c : ℝ)

def triangle_area := (1/2) * b * c * Real.sin A

theorem triangle_area_is_correct :
  b = 2 → c = 2 * Real.sqrt 2 → C = Real.pi / 4 → 
  triangle_area A B C a b c = Real.sqrt 3 + 1 :=
by
  intros
  sorry

end triangle_area_is_correct_l791_791704


namespace intersection_M_N_l791_791059

def M : set ℝ := {x | -1 ≤ x ∧ x < 5}
def N : set ℕ := {x | |(x : ℤ)| ≤ 2}

theorem intersection_M_N : M ∩ ↑N = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2} := 
by
  -- proof would go here
  sorry

end intersection_M_N_l791_791059


namespace correct_option_d_l791_791297

-- Definitions
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f x = f (-x))
variable (hf_inc : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x ≤ y → f x ≤ f y)

-- Theorem statement
theorem correct_option_d :
  f (Real.sin (Real.pi / 12)) > f (Real.tan (Real.pi / 12)) :=
sorry

end correct_option_d_l791_791297


namespace total_money_raised_l791_791940

variable (clements_cookies jakes_cookies torys_cookies : ℕ)

def baked_clementine : ℕ := 72
def baked_jake : ℕ := baked_clementine * 2
def baked_tory : ℕ := (baked_jake + baked_clementine) / 2
def price_per_cookie : ℕ := 2

theorem total_money_raised :
  let total_cookies := baked_clementine + baked_jake + baked_tory,
      total_money := total_cookies * price_per_cookie
  in
  total_money = 648 := by
  sorry

end total_money_raised_l791_791940


namespace quadruple_perimeter_l791_791557

-- Define the rectangle's original and expanded dimensions and perimeters
def original_perimeter (a b : ℝ) := 2 * (a + b)
def new_perimeter (a b : ℝ) := 2 * ((4 * a) + (4 * b))

-- Statement to be proved
theorem quadruple_perimeter (a b : ℝ) : new_perimeter a b = 4 * original_perimeter a b :=
  sorry

end quadruple_perimeter_l791_791557


namespace calc_fg_expr_fg_expr_gf_l791_791999

def f (x : ℝ) : ℝ := x ^ 2 - 1

def g (x : ℝ) : ℝ :=
  if x > 0 then x - 1 else 2 - x

theorem calc_fg :
  f (g 2) = 0 ∧ g (f 2) = 2 :=
by
  -- Proof will be inserted here
  sorry

theorem expr_fg (x : ℝ) :
  f (g x) =
    if x > 0 then x ^ 2 - 2 * x
    else x ^ 2 - 4 * x + 3 :=
by
  -- Proof will be inserted here
  sorry

theorem expr_gf (x : ℝ) :
  g (f x) =
    if x > 1 ∨ x < -1 then x ^ 2 - 2
    else 3 - x ^ 2 :=
by
  -- Proof will be inserted here
  sorry

end calc_fg_expr_fg_expr_gf_l791_791999


namespace time_XiaoXia_catches_up_with_XiaoLi_l791_791512

noncomputable def distance (speed : ℕ) (time : ℕ) := speed * time

theorem time_XiaoXia_catches_up_with_XiaoLi : 
  let start_time_XH_XL := 6  -- 6 am
  let start_time_XX := 8  -- 8 am
  let end_time := 18  -- 6 pm
  let speed_XH := 5
  let speed_XL := 4
  let travel_time_XH := end_time - start_time_XH_XL
  let distance_AB := distance speed_XH travel_time_XH
  let travel_time_XX := end_time - start_time_XX
  let speed_XX := distance_AB / travel_time_XX
  let distance_XL_8am := distance speed_XL (start_time_XX - start_time_XH_XL)
  let time_to_catch_up := distance_XL_8am / (speed_XX - speed_XL)
  let catch_up_time := start_time_XX + time_to_catch_up
in catch_up_time = 12 := sorry

end time_XiaoXia_catches_up_with_XiaoLi_l791_791512


namespace trapezoidal_park_no_solution_l791_791108

theorem trapezoidal_park_no_solution :
  (∃ b1 b2 : ℕ, 2 * 1800 = 40 * (b1 + b2) ∧ (∃ m : ℕ, b1 = 5 * (2 * m + 1)) ∧ (∃ n : ℕ, b2 = 2 * n)) → false :=
by
  sorry

end trapezoidal_park_no_solution_l791_791108


namespace problem_equivalence_l791_791726

-- Define the parametric equation of line l
def line_l_parametric (t : ℝ) : ℝ × ℝ :=
  (t, t + 2)

-- Standard equation of curve C
def curve_C_standard (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 2 * y = 0

-- Polar coordinates of point P
def point_P_polar : ℝ × ℝ :=
  (2 * Real.sqrt 2, 7 * Real.pi / 4)

-- Define the standard equation of line l
def line_l_standard (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the polar equation of curve C
def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the standard equation of translated line l'
def line_l_prime_standard (x y : ℝ) : Prop :=
  y = x

-- Define the distance function
def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2))

-- Main theorem statement
theorem problem_equivalence :
  (∀ t : ℝ, line_l_parametric t = (t, t + 2)) →
  curve_C_standard 0 0 →
  point_P_polar = (2 * Real.sqrt 2, 7 * Real.pi / 4) →
  (line_l_standard 0 0 → curve_C_polar 4 (Real.pi / 4) →
  ∃ x y : ℝ, line_l_prime_standard x y ∧ curve_C_standard x y ∧
  ∃ A B : ℝ × ℝ, distance point_P_polar A * distance point_P_polar B = 6) :=
sorry

end problem_equivalence_l791_791726


namespace find_angle_x_l791_791367

theorem find_angle_x (A B C D : Type) 
  (angleACB angleBCD : ℝ) 
  (h1 : angleACB = 90)
  (h2 : angleBCD = 40) 
  (h3 : angleACB + angleBCD + x = 180) : 
  x = 50 :=
by
  sorry

end find_angle_x_l791_791367


namespace complement_intersection_l791_791325

noncomputable def M : Set ℝ := {x | |x| > 2}
noncomputable def N : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | 1 < x ∧ x ≤ 2} :=
sorry

end complement_intersection_l791_791325


namespace anisha_final_number_l791_791932

/-- Define the sequence function that performs the specified operations. -/
def sequence : ℕ → ℕ
| 0 := 1000000
| (n + 1) := if n % 2 = 0 then sequence n / 2 else sequence n * 5

/-- State the final result of the sequence calculation after 12 steps. -/
theorem anisha_final_number : sequence 12 = 5 ^ 12 := 
sorry

end anisha_final_number_l791_791932


namespace negative_triangles_parity_l791_791709

-- Definitions based on conditions
variable {M : Type*} [Fintype M] [DecidableEq M]

def is_collinear (a b c : M) : Prop := 
  ∃ l : Line, a ∈ l ∧ b ∈ l ∧ c ∈ l

-- Number of points and edges with their properties
variable (n m : ℕ)
variable (points : Finset M) (E : Finset (M × M))
variable (assign : (M × M) → ℤ)
variable (h_no_collinear : ∀ a b c ∈ points, ¬ is_collinear a b c)
variable (h_edges : ∀ e ∈ E, (assign e = 1) ∨ (assign e = -1))
variable (h_m_edges : (E.filter (λ e, assign e = -1)).card = m)

-- Definition of negative triangle
def negative_triangle (a b c : M) : Prop := 
  assign (a, b) * assign (b, c) * assign (c, a) = -1

-- Final proof statement
theorem negative_triangles_parity :
  let k := (points.triangles.filter (λ t, negative_triangle t.1 t.2 t.3)).card in
  k % 2 = (n * m) % 2 := 
sorry

end negative_triangles_parity_l791_791709


namespace interval_intersection_l791_791253

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l791_791253


namespace count_correct_statements_l791_791220

theorem count_correct_statements
  (S1 : "The circumcenter of a triangle is the intersection point of the perpendicular bisectors of the three sides of the triangle." → True)
  (S2 : "If two arcs have equal degrees, then they are congruent arcs." → False)
  (S3 : "The diameter is the longest chord in a circle." → True)
  (S4 : "If two chords are equal, then the central angles they subtend are equal." → False) :
  ([S1, S2, S3, S4].filter (λ s, s)).length = 2 :=
  sorry

end count_correct_statements_l791_791220


namespace smallest_positive_period_of_f_increasing_interval_of_f_min_max_value_of_f_on_interval_l791_791319

def f (x : ℝ) : ℝ := (Math.sqrt 3 / 2) * (Real.sin (2 * x)) - (Real.cos x) ^ 2 - 1 / 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi := sorry

theorem increasing_interval_of_f (k : ℤ) :
  ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
  ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) →
  x ≤ y → f x ≤ f y := sorry

theorem min_max_value_of_f_on_interval :
  x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12) →
  f x ∈ Set.Icc (-1 - Math.sqrt 3 / 2) 0 := sorry

end smallest_positive_period_of_f_increasing_interval_of_f_min_max_value_of_f_on_interval_l791_791319


namespace exist_projections_l791_791383

-- Define types for lines and points
variable {Point : Type} [MetricSpace Point]

-- Define the projection operator
def projection (t_i t_j : Set Point) (p : Point) : Point := 
  sorry -- projection definition will go here

-- Define t1, t2, ..., tk
variables (t : ℕ → Set Point) (k : ℕ)
  (hk : k > 1)  -- condition: k > 1
  (ht_distinct : ∀ i j, i ≠ j → t i ≠ t j)  -- condition: different lines

-- Define the proposition
theorem exist_projections : 
  ∃ (P : ℕ → Point), 
    (∀ i, 1 ≤ i ∧ i < k → P (i + 1) = projection (t i) (t (i + 1)) (P i)) ∧ 
    P 1 = projection (t k) (t 1) (P k) :=
sorry

end exist_projections_l791_791383


namespace triangle_side_inequality_l791_791378

variable (a b c : ℝ)
def s : ℝ := (a + b + c) / 2

theorem triangle_side_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbac : b + c > a) (hcab : c + a > b) :
    (2 * a * (2 * a - s a b c)) / (b + c) + (2 * b * (2 * b - s a b c)) / (c + a) + (2 * c * (2 * c - s a b c)) / (a + b) ≥ s a b c := 
    by
    sorry

end triangle_side_inequality_l791_791378


namespace count_pairs_a_b_l791_791528

-- Define sequence length and target count
def sequenceLength : ℕ := 2016
def targetCount : ℕ := 508536

-- Define the proposition
theorem count_pairs_a_b :
  ∃ (count : ℕ), count = (sequenceLength - 1) * sequenceLength / 2 ∧ count = targetCount := by 
begin
  have h: (sequenceLength - 1) * sequenceLength / 2 = targetCount, 
  {
    calc (2016 - 1) * 2016 / 2 
      = 2015 * 2016 / 2 : by sorry
      = targetCount : by sorry
  },
  use (sequenceLength - 1) * sequenceLength / 2,
  split,
  { exact rfl, },
  { exact h, },
end

end count_pairs_a_b_l791_791528


namespace Emilio_determines_areas_l791_791530

noncomputable def Emilio_rectangles (n : ℕ) (h : 0 < n) : ℕ :=
2 * n - 1

theorem Emilio_determines_areas (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, (k = Emilio_rectangles n h) ∧
  (∀ areas : fin (n^2) → ℕ, -- areas of the n^2 rectangles
    ∃ determined_areas : fin (n^2) → ℕ, -- Emilio determines these areas
      (∀ i, determined_areas i = areas i)) :=
sorry

end Emilio_determines_areas_l791_791530


namespace smaller_number_is_25_l791_791465

theorem smaller_number_is_25 (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end smaller_number_is_25_l791_791465


namespace total_money_raised_l791_791941

variable (clements_cookies jakes_cookies torys_cookies : ℕ)

def baked_clementine : ℕ := 72
def baked_jake : ℕ := baked_clementine * 2
def baked_tory : ℕ := (baked_jake + baked_clementine) / 2
def price_per_cookie : ℕ := 2

theorem total_money_raised :
  let total_cookies := baked_clementine + baked_jake + baked_tory,
      total_money := total_cookies * price_per_cookie
  in
  total_money = 648 := by
  sorry

end total_money_raised_l791_791941


namespace find_smaller_number_l791_791462

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 := by
  sorry

end find_smaller_number_l791_791462


namespace Seed_Mixture_Y_is_25_percent_ryegrass_l791_791096

variables 
  (X_r : ℝ) (Y_r : ℝ) (M_r : ℝ) (X_weight : ℝ)

-- Conditions given in the problem
def Problem_Conditions : Prop :=
  (X_r = 0.40) ∧                   -- Seed mixture X is 40 percent ryegrass
  (M_r = 0.38) ∧                   -- Final mixture of X and Y is 38 percent ryegrass
  (X_weight = 0.8667) ∧            -- Weight of the mixture is 86.67 percent X
  (13.33 / 100 = 1 - X_weight) ∧   -- Weight of mixture Y is 13.33 percent
  (M_r = X_r * X_weight + Y_r * (13.33 / 100)) -- Mixing equation
  
-- The question we aim to prove:
theorem Seed_Mixture_Y_is_25_percent_ryegrass (h : Problem_Conditions) : Y_r = 0.25 :=
by
  sorry

end Seed_Mixture_Y_is_25_percent_ryegrass_l791_791096


namespace cars_in_garage_l791_791148

/-
Conditions:
1. Total wheels in the garage: 22
2. Riding lawnmower wheels: 4
3. Timmy's bicycle wheels: 2
4. Each of Timmy's parents' bicycles: 2 wheels, and there are 2 bicycles.
5. Joey's tricycle wheels: 3
6. Timmy's dad's unicycle wheels: 1

Question: How many cars are inside the garage?

Correct Answer: The number of cars is 2.
-/
theorem cars_in_garage (total_wheels : ℕ) (lawnmower_wheels : ℕ)
  (timmy_bicycle_wheels : ℕ) (parents_bicycles_wheels : ℕ)
  (joey_tricycle_wheels : ℕ) (dad_unicycle_wheels : ℕ) 
  (cars_wheels : ℕ) (cars : ℕ) :
  total_wheels = 22 →
  lawnmower_wheels = 4 →
  timmy_bicycle_wheels = 2 →
  parents_bicycles_wheels = 2 * 2 →
  joey_tricycle_wheels = 3 →
  dad_unicycle_wheels = 1 →
  cars_wheels = total_wheels - (lawnmower_wheels + timmy_bicycle_wheels + parents_bicycles_wheels + joey_tricycle_wheels + dad_unicycle_wheels) →
  cars = cars_wheels / 4 →
  cars = 2 := by
  sorry

end cars_in_garage_l791_791148


namespace exists_zero_in_interval_l791_791590

noncomputable def f (x : ℝ) := 3^x + x - 3

theorem exists_zero_in_interval :
  (∃ c ∈ set.Ioo 0 1, f c = 0) :=
by sorry

end exists_zero_in_interval_l791_791590


namespace probability_X_eq_Y_l791_791916

-- Define the conditions as functions or predicates.
def is_valid_pair (x y : ℝ) : Prop :=
  -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi ∧ -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi ∧ Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Final statement asserting the required probability.
theorem probability_X_eq_Y :
  ∃ (prob : ℝ), prob = 1 / 11 ∧ ∀ (x y : ℝ), is_valid_pair x y → (x = y ∨ x ≠ y ∧ prob = 1/11) :=
  sorry

end probability_X_eq_Y_l791_791916


namespace flower_bee_difference_proof_l791_791825

variable (flowers bees : ℕ)

def flowers_bees_difference (flowers bees : ℕ) : ℕ :=
  flowers - bees

theorem flower_bee_difference_proof : flowers_bees_difference 5 3 = 2 :=
by
  sorry

end flower_bee_difference_proof_l791_791825


namespace wedding_reception_friends_l791_791544

theorem wedding_reception_friends (total_guests bride_couples groom_couples bride_coworkers groom_coworkers bride_relatives groom_relatives: ℕ)
  (h1: total_guests = 400)
  (h2: bride_couples = 40) 
  (h3: groom_couples = 40)
  (h4: bride_coworkers = 10) 
  (h5: groom_coworkers = 10)
  (h6: bride_relatives = 20)
  (h7: groom_relatives = 20)
  : (total_guests - ((bride_couples + groom_couples) * 2 + (bride_coworkers + groom_coworkers) + (bride_relatives + groom_relatives))) = 180 := 
by 
  sorry

end wedding_reception_friends_l791_791544


namespace sequence_limit_l791_791526

open Real

theorem sequence_limit (e_n : ℕ → ℝ) (sin_cos_n : ℕ → ℝ)
  (cos_inv_n : ℕ → ℝ) (hyp_e_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (e_n n - 1) < ε)
  (hyp_sin_cos_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (sin_cos_n n - 0) < ε)
  (hyp_cos_inv_n : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (cos_inv_n n - 1) < ε)
  (seq_n : ℕ → ℝ) :
  ( ∀ n, e_n n = exp (1 / seq_n n) ) →
  ( ∀ n, sin_cos_n n = sin ( seq_n n / (seq_n n ^ 2 + 1) ) * cos n ) →
  ( ∀ n, cos_inv_n n = cos ( 1 / seq_n n )) →
  tendsto (λ n, (e_n n + sin_cos_n n) / (1 + cos_inv_n n)) at_top (𝓝 (1 / 2)) := by
sorry

end sequence_limit_l791_791526


namespace fiona_probability_correct_l791_791472

def probability_to_reach_pad14 :=
  (1 / 27) + (1 / 3) = 13 / 27 ∧
  (13 / 27) * (1 / 3) = 13 / 81 ∧
  (13 / 81) * (1 / 3) = 13 / 243 ∧
  (13 / 243) * (1 / 3) = 13 / 729 ∧
  (1 / 81) + (1 / 27) + (1 / 27) = 4 / 81 ∧
  (13 / 729) * (4 / 81) = 52 / 59049

theorem fiona_probability_correct :
  (probability_to_reach_pad14 : Prop) := by
  sorry

end fiona_probability_correct_l791_791472


namespace circle_non_intersecting_square_l791_791192

theorem circle_non_intersecting_square {rect_width rect_height : ℝ} (unit_squares : set (set (ℝ × ℝ))) :
  rect_width = 20 ∧ rect_height = 25 → 
  (∀ s, s ∈ unit_squares → (∃ a b : ℝ, s = { p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ a + 1 ∧ b ≤ p.2 ∧ p.2 ≤ b + 1 })) →
  (card unit_squares = 120) →
  (∃ (cx cy : ℝ), (0.5 ≤ cx ∧ cx ≤ rect_width - 0.5) ∧ (0.5 ≤ cy ∧ cy ≤ rect_height - 0.5) ∧
    ∀ s ∈ unit_squares, ∀ (p : ℝ × ℝ), p ∈ s → dist (cx, cy) p ≥ 0.5) :=
by
  sorry

end circle_non_intersecting_square_l791_791192


namespace number_of_single_rooms_l791_791736

theorem number_of_single_rooms (S : ℕ) : 
  (S + 13 * 2 = 40) ∧ (S * 10 + 13 * 2 * 10 = 400) → S = 14 :=
by 
  sorry

end number_of_single_rooms_l791_791736


namespace problem_solution_l791_791884

-- Define the conditions as Lean 4 statements
noncomputable def price_parent_child_tour_for_two := 35
noncomputable def price_family_tour_for_three := 2 * price_parent_child_tour_for_two
noncomputable def presale_amount_parent_child := 21000
noncomputable def presale_amount_family := 10500
noncomputable def sales_volume_difference := 450

-- Define the variables from part 2
noncomputable def initial_plan_parent_child := 1600
noncomputable def initial_plan_family := 400
noncomputable def price_increase_parent_child a := price_parent_child_tour_for_two + (3 * a) / 4
noncomputable def price_increase_family a := price_family_tour_for_three / 2 + a
noncomputable def sales_volume_decrease_parent_child a := initial_plan_parent_child - 32 * a

-- Prove the mathematical problem
theorem problem_solution (a : ℝ) (x : ℝ)
  (h1: 42000 - 10500 = 450 * x)
  (h2: x = 35)
  (h3: (price_increase_parent_child a) * (sales_volume_decrease_parent_child a) + (price_increase_family a) * initial_plan_family 
        = price_parent_child_tour_for_two * initial_plan_parent_child + price_family_tour_for_three * initial_plan_family):
  x = 35 ∧ a = 20 :=
by
  sorry

end problem_solution_l791_791884


namespace general_term_formula_sum_first_n_terms_l791_791302

-- Given condition as a definition in Lean
def Sn (n : ℕ) (an : ℕ → ℝ) : ℝ := (4 / 3) * (an n - 1)

-- Related to the general term formula of the sequence {a_n}
theorem general_term_formula (an : ℕ → ℝ) (n : ℕ) (h : n > 0) : Sn n an = (4 / 3) * (an n - 1) → an n = 4^n := by
  sorry

-- Related to sum of the first n terms of another sequence {T_n}, given bn = log_2 an
theorem sum_first_n_terms (an : ℕ → ℝ) (Tn : ℕ → ℝ) (n : ℕ) (h : n > 0) :
  (∀ n, an n = 4^n)
  → (∀ n, Tn n = 1 / ((2 * log 2 (an n)) - 1) * ((2 * log 2 (an n)) + 1))
  → Tn n = (1 / 2) * (1 - 1 / (2 * n + 1)) := by
  sorry

end general_term_formula_sum_first_n_terms_l791_791302


namespace math_proof_l791_791162

noncomputable def math_problem (x : ℝ) : ℝ :=
  (3 / (2 * x) * (1 / 2) * (2 / 5) * 5020) - ((2 ^ 3) * (1 / (3 * x + 2)) * 250) + Real.sqrt (900 / x)

theorem math_proof :
  math_problem 4 = 60.393 :=
by
  sorry

end math_proof_l791_791162


namespace identify_n_within_10_questions_l791_791043

noncomputable def fibonacci : ℕ → ℕ 
| 0     := 2
| 1     := 3
| (n+2) := fibonacci (n+1) + fibonacci n

theorem identify_n_within_10_questions (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 144) :
  ∃ strategy : (Π k, 1 ≤ k → k ≤ 10 →  ℕ → bool),
  ∀ answers : Π k, 1 ≤ k → k ≤ 10 → bool,
  -- Define Boolean function to check if question k can be answered affirmatively.
  (strategy 10 (by norm_num) (by norm_num) n = false → answers 10 (by norm_num) (by norm_num) = false) ∧
  (strategy 10 (by norm_num) (by norm_num) n = true  → answers 10 (by norm_num) (by norm_num) = true ) ∧
  -- Ensures questions are based on Fibonacci-modified values
  ( ∀ k, 9 ≤ k → k ≤ 10 → fib_sequence (k+1) ≤ 144  ∧ answers k (by norm_num) (by norm_num) = false  → n < fib_sequence k)  ∧
  (answers 10 (by norm_num) (by norm_num) = true  → 1 ≤ n ∧ n ≤ 144) :=
sorry

end identify_n_within_10_questions_l791_791043


namespace prop_c_correct_l791_791925

variable {Line : Type} {Plane : Type} (l m : Line) (α β : Plane)

def parallel (p q : Plane) : Prop := sorry
def perp (p q : Plane) : Prop := sorry
def lays_on (p q : Line) : Prop := sorry
def parallel_line (p q : Line) : Prop := sorry
def perp_line (p q : Line) : Prop := sorry

theorem prop_c_correct :
  parallel α β → 
  perp_line l α → 
  parallel_line m β → 
  perp_line l m := sorry

end prop_c_correct_l791_791925


namespace compare_numbers_l791_791815

theorem compare_numbers : 
  let a := real.exp (- real.sqrt 2)
  let b := real.logb 0.2 3
  let c := real.log real.pi
  in (0 < a ∧ a < 1) ∧ (b < 0) ∧ (1 < c) → 
     b < a ∧ a < c :=
by
  intros
  sorry

end compare_numbers_l791_791815


namespace base_8_subtraction_l791_791609

def subtract_in_base_8 (a b : ℕ) : ℕ := 
  -- Implementing the base 8 subtraction
  sorry

theorem base_8_subtraction : subtract_in_base_8 0o652 0o274 = 0o356 :=
by 
  -- Faking the proof to ensure it can compile.
  sorry

end base_8_subtraction_l791_791609


namespace four_points_no_line_separation_l791_791274

-- Define the four points as P1, P2, P3, P4 on a plane
variable {Point : Type}
variable (P1 P2 P3 P4 : Point)
variable [AffineGeometry Point]

-- The Lean 4 statement representing the mathematically equivalent proof problem
theorem four_points_no_line_separation :
  ∃ (G1 G2 : Finset Point), G1 ∪ G2 = {P1, P2, P3, P4} ∧ G1 ∩ G2 = ∅ ∧
    ∀ (l : AffineSubspace ℝ Point), ¬ ((G1 ⊆ l.1 *ᵥ) ∧ (G2 ⊆ l.1 *ᵥ)) :=
sorry

end four_points_no_line_separation_l791_791274


namespace distinct_primes_12_13_14_15_l791_791592

theorem distinct_primes_12_13_14_15 :
  let n := 12 * 13 * 14 * 15 in
  (∃ P : Finset ℕ, (∀ p ∈ P, Nat.Prime p) ∧ (∀ p ∈ P, p ∣ n) ∧ Multiset.card (Finset.val P) = 5) :=
sorry

end distinct_primes_12_13_14_15_l791_791592


namespace surface_area_of_sphere_l791_791142

noncomputable def sphere_surface_area_problem 
(points_on_sphere: (A B C D : Type)) 
(radius : ℝ) 
(distance: ℝ) 
(volume: ℝ) : Prop :=
(dist A B = 3 ∧ dist B C = 3 ∧ dist C A = 3) →
(volume = (9 * Real.sqrt 3) / 4) →
(radius = 2) →
(4 * Real.pi * radius * radius = 16 * Real.pi)

theorem surface_area_of_sphere (A B C D : Type) (R : ℝ) 
(h : sphere_surface_area_problem A B C D R 3 ((9 * Real.sqrt 3) / 4)) :
surface_area = 16 * Real.pi := sorry

end surface_area_of_sphere_l791_791142


namespace constant_term_expansion_l791_791607

theorem constant_term_expansion :
  (let expr := (x + 3 / x) * (x - 2 / x) ^ 5 in 
   true) → 
  (constant_term expr = 40) :=
by
  let expr := (x + 3 / x) * (x - 2 / x) ^ 5
  sorry

end constant_term_expansion_l791_791607


namespace tangent_at_0_f_greater_ln_div_x_l791_791317

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x - x^3 * Real.exp x

theorem tangent_at_0 : 
  let x := (0 : ℝ) in 
  let fx := f x in 
  let slope := (fderiv ℝ f 0) 1 in 
  ∃ m b, tangent_of_at f x = (λ x, m * x + b) ∧ m = 2 ∧ b = 2 := 
by 
  sorry

theorem f_greater_ln_div_x (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  f x > (Real.log x) / x :=
by 
  sorry

end tangent_at_0_f_greater_ln_div_x_l791_791317


namespace problem1_problem2_l791_791659

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end problem1_problem2_l791_791659


namespace tony_rope_length_l791_791839

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l791_791839


namespace perfect_squares_or_cubes_less_than_500_count_l791_791329

theorem perfect_squares_or_cubes_less_than_500_count : 
  let perfect_squares := {n : ℕ | n < 500 ∧ ∃ k : ℕ, k^2 = n}
  let perfect_cubes := {n : ℕ | n < 500 ∧ ∃ k : ℕ, k^3 = n}
  let perfect_sixths := {n : ℕ | n < 500 ∧ ∃ k : ℕ, k^6 = n}
  let count := (perfect_squares ∪ perfect_cubes).card - perfect_sixths.card
  count = 28 :=
by {
  sorry
}

end perfect_squares_or_cubes_less_than_500_count_l791_791329


namespace inequality_solution_set_l791_791818

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1 / 2} :=
by
  sorry

end inequality_solution_set_l791_791818


namespace units_digit_of_product_is_2_l791_791164

-- Define the first three positive even composite numbers
def first_three_positive_even_composite_numbers : List ℕ := [4, 6, 8]

-- Function to extract the units digit of a given number
def units_digit (n : ℕ) : ℕ := n % 10 

-- Define the product of the first three positive even composite numbers
def product_of_first_three_even_composites : ℕ := 
  List.foldl (*) 1 first_three_positive_even_composite_numbers

-- Theorem stating that the units digit of the product is 2
theorem units_digit_of_product_is_2 : units_digit product_of_first_three_even_composites = 2 :=
by
  sorry

end units_digit_of_product_is_2_l791_791164


namespace connected_if_and_only_if_two_vertex_removals_connected_l791_791398

variables {V : Type} [Fintype V] {E : Type} [DecidableEq V] {G : SimpleGraph V}

/-- The main theorem statement -/
theorem connected_if_and_only_if_two_vertex_removals_connected (hV : Fintype.card V ≥ 3) :
  (G.IsConnected) ↔ ∃ (v1 v2 : V), v1 ≠ v2 ∧ 
    (G.delete v1).IsConnected ∧ (G.delete v2).IsConnected :=
sorry

end connected_if_and_only_if_two_vertex_removals_connected_l791_791398


namespace eq_value_of_2a_plus_b_l791_791336

theorem eq_value_of_2a_plus_b (a b : ℝ) (h : abs (a + 2) + (b - 5)^2 = 0) : 2 * a + b = 1 := by
  sorry

end eq_value_of_2a_plus_b_l791_791336


namespace plot_length_l791_791522

theorem plot_length (b : ℝ) (cost_per_meter cost_total : ℝ)
  (h1 : cost_per_meter = 26.5) 
  (h2 : cost_total = 5300) 
  (h3 : (2 * (b + (b + 20)) * cost_per_meter) = cost_total) : 
  b + 20 = 60 := 
by 
  -- Proof here
  sorry

end plot_length_l791_791522


namespace die_probability_sum_4_l791_791341

theorem die_probability_sum_4 {Ω : Type*} (die : Ω) (throws : Ω → Finset (Fin 6)) :
  let events : Finset (Fin 6 × Fin 6) := 
    (Finset.product (throws die) (throws die))
  let favorable : Finset (Fin 6 × Fin 6) := 
    events.filter (λ x, x.1 + x.2 + 2 = 4)
  (favorable.card : ℚ) / (events.card : ℚ) = 1 / 12 :=
by 
  -- Proof steps go here
  sorry

end die_probability_sum_4_l791_791341


namespace rooms_with_two_windows_l791_791970

theorem rooms_with_two_windows
  (total_windows : ℕ)
  (rooms_with_four_windows : ℕ)
  (windows_per_four_windows : ℕ)
  (rooms_with_three_windows : ℕ)
  (windows_per_three_windows : ℕ)
  (rooms_with_two_windows : ℕ -> ℕ -> ℕ -> ℕ -> ℕ)
  (total_rooms : 5)
  (windows_four : 4)
  (rooms_three : 8)
  (windows_three : 3)
  (result : 39) :
  rooms_with_two_windows total_windows (rooms_with_four_windows * windows_per_four_windows)
  (rooms_with_three_windows * windows_per_three_windows) 2 = result :=
by
  sorry

end rooms_with_two_windows_l791_791970


namespace vector_AM_l791_791375

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (A B C M : V)

-- Conditions
def condition1 : Prop := B - A = a
def condition2 : Prop := C - A = b
def condition3 (k : ℝ) : Prop := B - M = k * (M - C) ∧ k = 3

-- Desired result
def result : V := (1 / 4 : ℝ) • a + (3 / 4 : ℝ) • b

-- Proof statement
theorem vector_AM (h1 : condition1) (h2 : condition2) (h3 : condition3 3) :
  M - A = result :=
sorry

end vector_AM_l791_791375


namespace midpoint_of_segment_l791_791855

theorem midpoint_of_segment :
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (4, 1) :=
by
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint = (4, 1)
  sorry

end midpoint_of_segment_l791_791855


namespace lines_configuration_l791_791772

noncomputable def LineConfiguration (A B : Point) (d : ℝ) : ℕ :=
sorry

theorem lines_configuration (A B : Point) (d : ℝ) (h : dist A B = 8) :
  LineConfiguration A B 8 = 3 :=
sorry

end lines_configuration_l791_791772


namespace calculate_120_percent_l791_791338

theorem calculate_120_percent (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end calculate_120_percent_l791_791338


namespace faster_runner_l791_791885

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- A's speed as a multiple of B's speed
variables (k : ℝ)

-- A's and B's distances in the race
variables (d_A d_B : ℝ)
-- Distance of the race
variables (distance : ℝ)
-- Head start given to B
variables (head_start : ℝ)

-- The theorem to prove that the factor k is 4 given the conditions
theorem faster_runner (k : ℝ) (v_A v_B : ℝ) (d_A d_B distance head_start : ℝ) :
  v_A = k * v_B ∧ d_B = distance - head_start ∧ d_A = distance ∧ (d_A / v_A) = (d_B / v_B) → k = 4 :=
by
  sorry

end faster_runner_l791_791885


namespace number_of_correct_statements_l791_791219

/-- Definitions of the statements to be evaluated. -/
def statement_1 : Prop :=
  ∀ (T : Triangle), is_circumcenter (intersection_point (perpendicular_bisectors T))

def statement_2 : Prop :=
  ∀ (arc1 arc2 : Arc), degrees arc1 = degrees arc2 → congruent_arcs arc1 arc2

def statement_3 : Prop :=
  ∀ (C : Circle), ∀ (chord : Chord C), is_diameter chord → 
    ∀ (other_chord : Chord C), length chord ≥ length other_chord

def statement_4 : Prop :=
  ∀ (C : Circle), ∀ (chord1 chord2 : Chord C), length chord1 = length chord2 → 
    central_angle chord1 = central_angle chord2

/-- The proposition stating that the number of correct statements is 2. -/
theorem number_of_correct_statements :
  (statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ ¬statement_4) →
  2 = 2 :=
by
  sorry

end number_of_correct_statements_l791_791219


namespace variation_of_n_l791_791718

variables {θ : ℝ} (hθ : 0 ≤ θ ∧ θ ≤ 60) -- θ varies from 0° to 60°
noncomputable def n (θ : ℝ) : ℝ := 120 - 2 * θ

theorem variation_of_n :
  ∃ n : ℝ → ℝ, (∀ θ, 0 ≤ θ → θ ≤ 60 → n(θ) = 120 - 2 * θ) ∧
              (n 0 = 120) ∧ (n 60 = 60) := 
begin
  use n,
  split,
  { intros θ hθ1 hθ2,
    simp [n], },
  split,
  { simp [n], },
  { simp [n], },
end

end variation_of_n_l791_791718


namespace range_of_f_l791_791451

-- Define the function f(x) = 2^(1 - x)
def f (x : ℝ) : ℝ := 2^(1 - x)

-- Define the range of x
def domain := { x : ℝ | 1 ≤ x }

-- Prove that the range of the function f(x) = 2^(1 - x) for x in the domain is (0, 1]
theorem range_of_f : set.range f = set.Ioc 0 1 :=
by
  sorry

end range_of_f_l791_791451


namespace pythagoras_schools_count_l791_791365

theorem pythagoras_schools_count
    (high_schools : ℕ)
    (is_team_size : high_schools * 4 = total_students)
    (distinct_scores : ∀ (i j : ℕ), i < total_students → j < total_students → i ≠ j → scores i ≠ scores j)
    (andrea_median : scores ((total_students + 1) / 2) = scores (andrea_rank))
    (andrea_highest_on_team : scores (andrea_rank) > scores 23 ∧ scores 45 ∧ scores 87) :
    high_schools = 22 := sorry

end pythagoras_schools_count_l791_791365


namespace horner_evaluation_l791_791846

noncomputable def f (x : ℝ) : ℝ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_evaluation :
  f 2 = 1397 ∧ 
  f (-1) = -1 ∧ 
  (∃ c ∈ set.Icc (-1:ℝ) (2:ℝ), f c = 0) := by
sorry

end horner_evaluation_l791_791846


namespace find_m_l791_791628

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def f' (x : ℝ) : ℝ := -1 / (x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * x

theorem find_m (m : ℝ) :
  g 2 m = 1 / (f' 2) →
  m = -2 :=
by
  sorry

end find_m_l791_791628


namespace tangent_eqn_at_1_2_l791_791307

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_eqn_at_1_2 : 
  f (1) = 2 → 
  deriv f 1 = -2 → 
  (∀ x y : ℝ, y - 2 = -2 * (x - 1) → 2 * x + y - 4 = 0) :=
by
  intros h₁ h₂ x y h
  rw [h]
  sorry

end tangent_eqn_at_1_2_l791_791307


namespace find_n_mod_11_l791_791249

theorem find_n_mod_11 : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [MOD 11] ∧ n = 5 :=
sorry

end find_n_mod_11_l791_791249


namespace decorative_object_height_l791_791209

def diameter_fountain := 20 -- meters
def radius_fountain := diameter_fountain / 2 -- meters

def max_height := 8 -- meters
def distance_to_max_height := 2 -- meters

-- The initial height of the water jets at the decorative object
def initial_height := 7.5 -- meters

theorem decorative_object_height :
  initial_height = 7.5 :=
  sorry

end decorative_object_height_l791_791209


namespace ivan_ivanovich_increase_l791_791408

variable (p v s i : ℝ)
variable (k : ℝ)

-- Conditions
def initial_shares_sum := p + v + s + i = 1
def petya_doubles := 2 * p + v + s + i = 1.3
def vanya_doubles := p + 2 * v + s + i = 1.4
def sergey_triples := p + v + 3 * s + i = 1.2

-- Target statement to be proved
theorem ivan_ivanovich_increase (hp : p = 0.3) (hv : v = 0.4) (hs : s = 0.1)
  (hi : i = 0.2) (k : ℝ) : k * i > 0.75 → k > 3.75 :=
sorry

end ivan_ivanovich_increase_l791_791408


namespace symmetric_coloring_l791_791761

theorem symmetric_coloring (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) (a_gt_2 : a > 2) (b_gt_2 : b > 2) :
  ∃ S : ℤ, S = (a * b + a + b) / 2 ∧ ∀ c : ℤ,
    (∃ x y : ℕ, a * x + b * y = c) ↔ ¬ (∃ x y : ℕ, a * x + b * y = (a * b + a + b - c)) :=
begin
  sorry
end

end symmetric_coloring_l791_791761


namespace simple_annual_interest_rate_l791_791222

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end simple_annual_interest_rate_l791_791222


namespace reach_point_3n_plus_1_l791_791370

-- Base cases for the number of ways to reach points 1, 2, and 3
def T (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 1
  | 3 => 1
  | k + 4 => T(k + 3) + T(k + 2)  -- recursive definition for intermediate points
  | _ => 0

-- Recursive relation
noncomputable def T_3n_plus_1 : ℕ → ℕ
| 0 => 2  -- T_{3 * 0 + 1} = 2
| n + 1 => 2 * T (3 * n + 1)

-- Proving that T_{3n+1} = 2 * 3^{n-1}
theorem reach_point_3n_plus_1 (n : ℕ) : T_3n_plus_1 n = 2 * 3 ^ (n - 1) := sorry

end reach_point_3n_plus_1_l791_791370


namespace common_property_rhombus_rectangle_diagonals_l791_791450

-- Define a structure for Rhombus and its property
structure Rhombus (R : Type) :=
  (diagonals_perpendicular : Prop)
  (diagonals_bisect : Prop)

-- Define a structure for Rectangle and its property
structure Rectangle (R : Type) :=
  (diagonals_equal_length : Prop)
  (diagonals_bisect : Prop)

-- Define the theorem that states the common property between diagonals of both shapes
theorem common_property_rhombus_rectangle_diagonals (R : Type) 
  (rhombus_properties : Rhombus R) 
  (rectangle_properties : Rectangle R) :
  rhombus_properties.diagonals_bisect ∧ rectangle_properties.diagonals_bisect :=
by {
  -- Since the solution steps are not to be included, we conclude the proof with 'sorry'
  sorry
}

end common_property_rhombus_rectangle_diagonals_l791_791450


namespace guacamole_servings_l791_791626

theorem guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (additional_avocados : ℕ) (total_avocados : ℕ := initial_avocados + additional_avocados) (servings : ℕ := total_avocados / avocados_per_serving) :
  avocados_per_serving = 3 →
  initial_avocados = 5 →
  additional_avocados = 4 →
  servings = 3 :=
by
  intros h1 h2 h3
  unfold servings total_avocados
  rw [h1, h2, h3]
  norm_num
  rfl
  sorry

end guacamole_servings_l791_791626


namespace heroes_total_games_l791_791931

theorem heroes_total_games (y x : ℕ) (h1 : x = 0.40 * y) 
                           (h2 : (x + 8) = 0.52 * (y + 11)) : 
                           y + 11 = 30 := by
sory

end heroes_total_games_l791_791931


namespace transformed_mean_variance_correct_l791_791645

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  List.sum (data.map (λ x => (x - m) ^ 2)) / (data.length - 1)

noncomputable def transformed_data (data : List ℝ) : List ℝ :=
  data.map (λ x => 2 * x - 1)

theorem transformed_mean_variance_correct (x : List ℝ)
  (h_len : x.length = 5)
  (h_mean : mean x = 2)
  (h_variance : variance x = 1 / 3) :
  mean (transformed_data x) = 3 ∧ variance (transformed_data x) = 4 / 3 :=
begin
  sorry
end

end transformed_mean_variance_correct_l791_791645


namespace right_triangle_sum_of_squares_l791_791724

   theorem right_triangle_sum_of_squares {AB AC BC : ℝ} (h_right: AB^2 + AC^2 = BC^2) (h_hypotenuse: BC = 1) :
     AB^2 + AC^2 + BC^2 = 2 :=
   by
     sorry
   
end right_triangle_sum_of_squares_l791_791724


namespace base_of_parallelogram_l791_791246

variable (Area Height Base : ℝ)

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem base_of_parallelogram
  (h_area : Area = 200)
  (h_height : Height = 20)
  (h_area_def : parallelogram_area Base Height = Area) :
  Base = 10 :=
by sorry

end base_of_parallelogram_l791_791246


namespace anatoliy_handshakes_l791_791899

-- Define the total number of handshakes
def total_handshakes := 197

-- Define friends excluding Anatoliy
def handshake_func (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the target problem stating that Anatoliy made 7 handshakes
theorem anatoliy_handshakes (n k : Nat) (h : handshake_func n + k = total_handshakes) : k = 7 :=
by sorry

end anatoliy_handshakes_l791_791899


namespace problem_statement_l791_791393

def f (n : ℕ) : ℝ := ∑ i in finset.range (3 * n), (1 : ℝ) / (i + 1)

theorem problem_statement (n : ℕ) (h : 0 < n) : 
  f (n + 1) - f n = (1 : ℝ) / (3 * n) + (1 : ℝ) / (3 * n + 1) + (1 : ℝ) / (3 * n + 2) :=
sorry

end problem_statement_l791_791393


namespace choose_three_points_in_square_l791_791949

theorem choose_three_points_in_square 
  (n : ℕ) 
  (points : set (ℝ × ℝ)) 
  (h_points : points.card = (n + 1) ^ 2)
  (h_in_square : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ n ∧ 0 ≤ p.2 ∧ p.2 ≤ n) : 
  ∃ p1 p2 p3 ∈ points, area p1 p2 p3 ≤ 1 / 2 := 
sorry

-- Definitions for 'area' and 'card' would be provided as per the formalization libraries in Lean.
-- Assuming 'area' calculates the area of the triangle formed by three points and 'card' gives the cardinality of a finite set.

end choose_three_points_in_square_l791_791949


namespace tony_initial_fish_l791_791150

-- Let's define the problem variables
variable (F : ℕ) -- The number of fish Tony starts with

-- The conditions described in the problem
variable (increase_per_year : ℕ := 1) -- Defined net increase in fish per year
variable (num_years : ℕ := 5) -- Number of years considered
variable (total_fish : ℕ := 7) -- Total number of fish after the specified years

-- The theorem to prove
theorem tony_initial_fish : F = 2 :=
  --c The initial number of fish Tony started with plus the net increase each year times the number of years should equal the total number of fish
  have : total_fish = F + increase_per_year * num_years, by sorry,
  have : 7 = F + 1 * 5, by sorry,
  sorry

end tony_initial_fish_l791_791150


namespace solve_for_x_l791_791691

theorem solve_for_x (x y z : ℕ) 
  (h1 : 3^x * 4^y / 2^z = 59049)
  (h2 : x - y + 2 * z = 10) : 
  x = 10 :=
sorry

end solve_for_x_l791_791691


namespace find_smaller_number_l791_791463

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 := by
  sorry

end find_smaller_number_l791_791463


namespace xy_range_is_e_to_infinity_l791_791655

noncomputable def xy_range (x y : ℝ) : set ℝ :=
{z : ℝ | z = x * y ∧ e^x = x * y * (2 * real.log x + real.log y)}

theorem xy_range_is_e_to_infinity (x y : ℝ) :
  (∃ z, z ∈ xy_range x y) → {z | z ∈ xy_range x y} = set.Ici real.exp 1 :=
sorry

end xy_range_is_e_to_infinity_l791_791655


namespace extra_pieces_of_gum_l791_791094

theorem extra_pieces_of_gum (total_packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  if total_packages = 43 ∧ pieces_per_package = 23 ∧ total_pieces = 997 then
    997 - (43 * 23)
  else
    0  -- This is a dummy value for other cases, as they do not satisfy our conditions.

#print extra_pieces_of_gum

end extra_pieces_of_gum_l791_791094


namespace correct_proposition_l791_791308

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 + x else 1 - x

def prop_A := ∀ x : ℝ, f (Real.sin x) = -f (Real.sin (-x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_B := ∀ x : ℝ, f (Real.sin x) = f (Real.sin (-x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_C := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))
def prop_D := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))

theorem correct_proposition :
  (¬ prop_A ∧ ¬ prop_B ∧ prop_C ∧ ¬ prop_D) :=
sorry

end correct_proposition_l791_791308


namespace arithmetic_sequence_9th_term_l791_791029

variables {a_n : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_9th_term
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 3 = 6)
  (h2 : a 6 = 3)
  (h_seq : arithmetic_sequence a d) :
  a 9 = 0 :=
sorry

end arithmetic_sequence_9th_term_l791_791029


namespace find_ellipse_equation_find_triangle_area_range_l791_791571

-- Define the conditions
def ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b

def foci (a b c : ℝ) :=
  2 * c = 4 ∧ c / a = Real.sqrt 2 / 2 ∧ b^2 = a^2 - c^2

def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 2 / 2

-- Prove the equation of the ellipse
theorem find_ellipse_equation (a b c : ℝ) (h_ellipse : ellipse a b) (h_foci : foci a b c) :
  (a = 2 * Real.sqrt 2) ∧ (c = 2) → (b = 2) → 
  (∀ x y : ℝ, 
    (x^2 / (2 * Real.sqrt 2)^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1)) :=
  sorry

-- Prove the range of the area of triangle AOB
theorem find_triangle_area_range (a b c : ℝ) (M : ℝ × ℝ) (h_ellipse : ellipse a b) (h_foci : foci a b c) :
  M = (0, 1) →
  (∀ (A B : ℝ × ℝ), line_through M A ∧ line_through M B ∧ (is_intersection A B a b) →
    (0 < area_triangle (0, 0) A B) ∧ (area_triangle (0, 0) A B ≤ Real.sqrt 6)) :=
  sorry

-- Define what a line through a point is
def line_through (M P : ℝ × ℝ) : Prop := sorry

-- Define what it means for a point to be an intersection of the ellipse
def is_intersection (A B : ℝ × ℝ) (a b : ℝ) : Prop := sorry

-- Define the area of a triangle given three points
def area_triangle (O A B : ℝ × ℝ) : ℝ := sorry

end find_ellipse_equation_find_triangle_area_range_l791_791571


namespace find_x_l791_791986

theorem find_x (x : ℝ) (h : sqrt (4 - 5 * x) = 9) : x = - (77 / 5) :=
by
  sorry

end find_x_l791_791986


namespace interval_of_x_l791_791263

theorem interval_of_x (x : ℝ) : 
  (2 < 3 * x ∧ 3 * x < 3) ∧ 
  (2 < 4 * x ∧ 4 * x < 3) ↔ 
  x ∈ set.Ioo (2 / 3) (3 / 4) := 
sorry

end interval_of_x_l791_791263


namespace extreme_values_part1_minimum_value_part2_l791_791316

section Part1

def f1 (x : ℝ) := - (1/3) * x^3 - (1/2) * x^2 + 6 * x

theorem extreme_values_part1 :
  ∃ (x_max x_min : ℝ), f1 x_max = 22 / 3 ∧ f1 x_min = -27 / 2 :=
  sorry

end Part1

section Part2

def f2 (m x : ℝ) := (1/3) * x^3 - (1/2) * x^2 + 2 * m * x

theorem minimum_value_part2 (m : ℝ) (hm : -2 < m ∧ m < 0) :
  (∀ x ∈ set.Icc 1 4, f2 m x ≤ (16 / 3)) →
  (∃ x_min : ℝ, x_min ∈ set.Icc 1 4 ∧ f2 m x_min = -10 / 3) :=
  sorry

end Part2

end extreme_values_part1_minimum_value_part2_l791_791316


namespace total_buying_cost_per_unit_correct_l791_791775

def total_cost_unit_A := 500
def units_A := 20
def total_cost_unit_B := 700
def units_B := 15
def total_cost_unit_C := 400
def units_C := 10

def cost_per_unit_A := total_cost_unit_A / units_A
def cost_per_unit_B := total_cost_unit_B / units_B
def cost_per_unit_C := total_cost_unit_C / units_C

def total_cost_all := total_cost_unit_A + total_cost_unit_B + total_cost_unit_C
def total_units_all := units_A + units_B + units_C

def average_cost_per_unit := total_cost_all / total_units_all

theorem total_buying_cost_per_unit_correct : average_cost_per_unit = 35.56 := by
  sorry

end total_buying_cost_per_unit_correct_l791_791775


namespace speed_percentage_proof_l791_791549

-- Definition for usual time and additional time
def usual_time : ℕ := 24
def additional_time : ℕ := 24

-- Definition of the percentage in decimal form
def percentage_speed := (usual_time / (usual_time + additional_time) : ℝ)

theorem speed_percentage_proof : percentage_speed = 0.5 :=
by
  sorry

end speed_percentage_proof_l791_791549


namespace ratio_of_pizzas_l791_791327

-- Define the conditions
def Heather_first_day (H : ℕ) : Prop := 
  ∃ T_Craig, 
    (T_Craig = 40 + 40 + 60) ∧
    (H + (40 - 20) + 40 + (40 + 60) = 380) ∧
    (H + (H + 140) = 380)

-- Define the goal
def ratio_correct (H : ℕ) : Prop :=
  H = 220 → (H : 40) = (11 : 2)

-- Prove the final statement
theorem ratio_of_pizzas (H : ℕ) (h1 : Heather_first_day H) : ratio_correct H :=
by {sorry}

end ratio_of_pizzas_l791_791327


namespace length_of_faster_train_l791_791154

/-- Define the speeds of the trains in kmph -/
def speed_faster_train := 180 -- in kmph
def speed_slower_train := 90  -- in kmph

/-- Convert speeds to m/s -/
def kmph_to_mps (speed : ℕ) : ℕ := speed * 5 / 18

/-- Define the relative speed in m/s -/
def relative_speed := kmph_to_mps speed_faster_train - kmph_to_mps speed_slower_train

/-- Define the time it takes for the faster train to cross the man in seconds -/
def crossing_time := 15 -- in seconds

/-- Define the length of the train calculation in meters -/
noncomputable def length_faster_train := relative_speed * crossing_time

theorem length_of_faster_train :
  length_faster_train = 375 :=
by
  sorry

end length_of_faster_train_l791_791154


namespace polynomial_coefficients_l791_791993

theorem polynomial_coefficients
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : (x-3)^8 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3 + 
                a_4 * (x-2)^4 + a_5 * (x-2)^5 + a_6 * (x-2)^6 + 
                a_7 * (x-2)^7 + a_8 * (x-2)^8) :
  (a_0 = 1) ∧ 
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + a_4 / 2^4 + a_5 / 2^5 + 
   a_6 / 2^6 + a_7 / 2^7 + a_8 / 2^8 = -255 / 256) ∧ 
  (a_0 + a_2 + a_4 + a_6 + a_8 = 128) :=
by sorry

end polynomial_coefficients_l791_791993


namespace trigonometric_identity_l791_791277

open Real

variable (α : ℝ)

theorem trigonometric_identity (h : tan (π - α) = 2) :
  (sin (π / 2 + α) + sin (π - α)) / (cos (3 * π / 2 + α) + 2 * cos (π + α)) = 1 / 4 :=
  sorry

end trigonometric_identity_l791_791277


namespace georgie_guacamole_servings_l791_791624

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end georgie_guacamole_servings_l791_791624


namespace spherical_coordinates_of_M_l791_791486

theorem spherical_coordinates_of_M : 
  let x := -1
  let y := -1
  let z := sqrt 2
  let r := sqrt (x^2 + y^2 + z^2)
  let phi := atan2 y x
  let theta := acos (z / r)
  (r, phi, theta) = (2, π/4, 5*π/4) := 
by 
  sorry

end spherical_coordinates_of_M_l791_791486


namespace calculate_entire_surface_area_l791_791564

-- Define the problem parameters
def cube_edge_length : ℝ := 4
def hole_side_length : ℝ := 2

-- Define the function to compute the total surface area
noncomputable def entire_surface_area : ℝ :=
  let original_surface_area := 6 * (cube_edge_length ^ 2)
  let hole_area := 6 * (hole_side_length ^ 2)
  let exposed_internal_area := 6 * 4 * (hole_side_length ^ 2)
  original_surface_area - hole_area + exposed_internal_area

-- Statement of the problem to prove the given conditions
theorem calculate_entire_surface_area : entire_surface_area = 168 := by
  sorry

end calculate_entire_surface_area_l791_791564


namespace find_f_log2_3_l791_791344

-- Define the conditions
variables (f : ℝ → ℝ) 

axiom monotonic_on_real : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)
axiom key_condition : ∀ x : ℝ, f (f(x) + 2 / (2^x + 1)) = 1 / 3

-- State the theorem
theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = 1 / 2 :=
sorry

end find_f_log2_3_l791_791344


namespace percentage_gain_is_8_05_l791_791514

noncomputable def percentage_gain : ℚ := 
  let total_cost := 118 * 12 in
  let total_revenue := 102 * 15 in
  let profit := total_revenue - total_cost in
  (profit / total_cost) * 100

theorem percentage_gain_is_8_05 :
  percentage_gain = 8.05 := by
  sorry

end percentage_gain_is_8_05_l791_791514


namespace constant_term_binomial_expansion_6_coefficient_middle_term_binomial_expansion_8_of_equal_binomials_l791_791306

theorem constant_term_binomial_expansion_6 :
  (let x := (x^2 + (2 : ℕ)/x)^6 in (some (binomial x 6)) = 240) :=
sorry

theorem coefficient_middle_term_binomial_expansion_8_of_equal_binomials
  (n : ℕ) (h : nat.choose n 2 = nat.choose n 6) :
  n = 8 ∧ (some (binomial (x^2 + (2: ℕ)/x)^n)).middle_term.coefficient = 1120 :=
sorry

end constant_term_binomial_expansion_6_coefficient_middle_term_binomial_expansion_8_of_equal_binomials_l791_791306


namespace square_distance_of_EF_equal_l791_791559

-- Define the problem conditions
variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (dist : A → B → ℝ) (dist : B → C → ℝ) (dist : C → D → ℝ) (dist : A → E → ℝ) (dist : D → F → ℝ)
variables (side_length : ℝ) (exterior_dist_BE : ℝ) (exterior_dist_DF : ℝ) (exterior_dist_AE : ℝ) (exterior_dist_CF : ℝ)

-- Conditions from part a)
def conditions : Prop :=
  dist A B = 15 ∧  -- AB side length is 15
  dist B E = 7 ∧  dist D F = 7 ∧  -- BE and DF are 7
  dist A E = 17 ∧ dist C F = 17    -- AE and CF are 17

-- The statement to prove
theorem square_distance_of_EF_equal :
  conditions dist side_length exterior_dist_BE exterior_dist_DF exterior_dist_AE exterior_dist_CF →
  (dist E F)^2 = 338 :=
by
  intro h
  sorry

end square_distance_of_EF_equal_l791_791559


namespace evaluate_expression_l791_791967

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 9) : 
  2 * x^(y / 2 : ℕ) + 5 * y^(x / 2 : ℕ) = 1429 := by
  sorry

end evaluate_expression_l791_791967


namespace Melanie_gumballs_sale_l791_791763

theorem Melanie_gumballs_sale (gumballs : ℕ) (price_per_gumball : ℕ) (total_price : ℕ) :
  gumballs = 4 →
  price_per_gumball = 8 →
  total_price = gumballs * price_per_gumball →
  total_price = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end Melanie_gumballs_sale_l791_791763


namespace center_polar_of_circle_l791_791448

noncomputable def circle_center_polar_coordinates : Prop :=
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ↔ x = c.fst ∧ y = c.snd) ∧
               (sqrt ((c.fst)^2 + (c.snd)^2) = sqrt 2 ∧ atan2 c.snd c.fst = π / 4)

theorem center_polar_of_circle :
  circle_center_polar_coordinates :=
sorry

end center_polar_of_circle_l791_791448


namespace bingo_first_column_possibilities_l791_791358

theorem bingo_first_column_possibilities :
  let s := finset.range(16) \ {0}
  ∃ choices : finset (finset ℕ), 
  choices.card = 5 ∧
  ∀ c ∈ choices, c ⊆ s ∧
  choices.card.factorial = 15 * 14 * 13 * 12 * 11 :=
by
  let s := finset.range(16) \ {0}
  let choices := s.powerset.filter (λ x, x.card = 5)
  use choices
  split
  · sorry
  · intro c hc
    split
    · exact sorry
    · exact sorry

end bingo_first_column_possibilities_l791_791358
