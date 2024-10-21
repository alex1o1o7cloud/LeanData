import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l632_63261

noncomputable def f (x : ℝ) : ℝ := (7 * x^3 - 4 * x^2 + 6) / (2 * x^3 + 3 * x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 7/2| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l632_63261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_person_longer_shadow_implies_point_light_l632_63240

/-- Represents a person with their height and shadow length -/
structure Person where
  height : ℝ
  shadowLength : ℝ

/-- Represents different types of light sources -/
inductive LightSource
  | Point
  | Parallel

/-- 
Given two people where one is shorter but has a longer shadow,
prove that they must be standing under a point light source
-/
theorem shorter_person_longer_shadow_implies_point_light
  (p1 p2 : Person)
  (h_height : p1.height < p2.height)
  (h_shadow : p1.shadowLength > p2.shadowLength) :
  LightSource.Point = LightSource.Point :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_person_longer_shadow_implies_point_light_l632_63240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_square_prism_condition_l632_63246

/-- A prism type representing a three-dimensional geometric shape. -/
structure Prism where
  base : Set (Fin 3 → ℝ)
  lateral_faces : Set (Set (Fin 3 → ℝ))

/-- Predicate to check if a set of points forms a rhombus. -/
def is_rhombus (s : Set (Fin 3 → ℝ)) : Prop := sorry

/-- Predicate to check if three edges meeting at a vertex are mutually perpendicular. -/
def has_perpendicular_edges_at_vertex (p : Prism) : Prop := sorry

/-- Predicate to check if a prism is a right square prism. -/
def is_right_square_prism (p : Prism) : Prop := sorry

/-- Theorem stating the condition for a prism to be a right square prism. -/
theorem right_square_prism_condition (p : Prism) : 
  is_right_square_prism p ↔ is_rhombus p.base ∧ has_perpendicular_edges_at_vertex p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_square_prism_condition_l632_63246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l632_63216

-- Define the parameters
noncomputable def σ : ℝ := 5.2
def n : ℕ := 100
noncomputable def x_bar : ℝ := 27.56
noncomputable def μ₀ : ℝ := 26
noncomputable def α : ℝ := 0.05

-- Define the test statistic
noncomputable def test_statistic : ℝ := (x_bar - μ₀) * (n.sqrt : ℝ) / σ

-- Define the critical value (for two-tailed test at α = 0.05)
noncomputable def critical_value : ℝ := 1.96

-- Theorem to prove
theorem reject_null_hypothesis : |test_statistic| > critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l632_63216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_f_positive_three_zeros_condition_l632_63268

noncomputable section

variable (a b : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + b / x

axiom f_domain (x : ℝ) : x > 0

axiom f_symmetry : ∀ x > 0, f a b x + f a b (1 / x) = 0

theorem tangent_line_condition : f a a 1 = 0 → HasDerivAt (f a a) (1 - 2 * a) 1 → a = -2 := by
  sorry

theorem f_positive : 0 < a → a < 1 → f a a (a^2 / 2) > 0 := by
  sorry

theorem three_zeros_condition : (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a a x = 0 ∧ f a a y = 0 ∧ f a a z = 0) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_f_positive_three_zeros_condition_l632_63268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l632_63263

-- Define the vectors
noncomputable def a (α : ℝ) : Fin 2 → ℝ := ![Real.cos α, Real.sin α]
noncomputable def b : Fin 2 → ℝ := ![-1/2, Real.sqrt 3/2]

-- Main theorem
theorem vector_properties (α : ℝ) 
  (h1 : 0 ≤ α ∧ α < 2 * Real.pi) : 
  (∀ (i : Fin 2), (a α + b) i * (a α - b) i = 0) ∧
  (‖(Real.sqrt 3 • a α + b)‖ = ‖(a α - Real.sqrt 3 • b)‖ → 
    α = Real.pi/6 ∨ α = 7*Real.pi/6) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l632_63263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l632_63259

noncomputable def a (n : ℕ) : ℝ := (4/9)^(n-1) - (2/3)^(n-1)

theorem sequence_has_max_and_min :
  (∃ M : ℝ, ∀ n : ℕ, n ≥ 1 → a n ≤ M) ∧
  (∃ m : ℝ, ∀ n : ℕ, n ≥ 1 → m ≤ a n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l632_63259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l632_63253

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a = 1 ∧ 
  (Real.sin (2 * t.A + t.B)) / (Real.sin t.A) = 2 * (1 - Real.cos t.C)

/-- Area of the triangle -/
noncomputable def TriangleArea (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.C

/-- Main theorem to prove -/
theorem triangle_theorem (t : Triangle) 
  (h : TriangleConditions t) 
  (area_cond : TriangleArea t = Real.sqrt 3 / 2) : 
  t.b = 2 ∧ (t.c = Real.sqrt 3 ∨ t.c = Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l632_63253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l632_63271

/-- The curve function y = x sin x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.sin x + x * Real.cos x

/-- The tangent line at x = -π/2 -/
noncomputable def tangent_line (x : ℝ) : ℝ := -x

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := Real.pi^2 / 2

theorem area_of_triangle :
  let x₀ : ℝ := -Real.pi/2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ,
  y = tangent_line x →
  x ≥ 0 ∧ x ≤ Real.pi →
  y ≥ 0 →
  triangle_area = (Real.pi * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l632_63271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_expressions_correct_l632_63231

/-- The number of algebraically different expressions obtained by placing parentheses
    in a division of n reciprocals (n ≥ 2) -/
def num_expressions (n : ℕ) : ℕ :=
  2^(n - 2)

/-- The actual number of algebraically different expressions obtained by
    placing parentheses in 1/a₂ ÷ 1/a₃ ÷ ... ÷ 1/aₙ -/
def number_of_different_expressions (n : ℕ) : ℕ :=
  sorry -- This function is not actually defined, but we use it for the theorem statement

/-- Theorem: For n ≥ 2, the number of algebraically different expressions obtained by
    placing parentheses in 1/a₂ ÷ 1/a₃ ÷ ... ÷ 1/aₙ is equal to 2^(n-2) -/
theorem num_expressions_correct (n : ℕ) (h : n ≥ 2) :
  num_expressions n = number_of_different_expressions n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_expressions_correct_l632_63231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l632_63217

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x + y) + y * f y) →
  (∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l632_63217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_H_coordinates_l632_63267

/-- Given points O, A, and B in 3D space, if H is a point on the line OA such that BH is perpendicular to OA, 
    then H has specific coordinates. -/
theorem point_H_coordinates (O A B H : ℝ × ℝ × ℝ) : 
  O = (0, 0, 0) →
  A = (-1, 1, 0) →
  B = (0, 1, 1) →
  (∃ t : ℝ, H = t • (A - O)) →
  ((H - B) • (A - O) = 0) →
  H = (-1/2, 1/2, 0) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_H_coordinates_l632_63267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circles_radii_l632_63294

-- Define Circle as a structure
structure Circle (α : Type*) where
  center : α × α
  radius : ℝ

-- Define a touches relation between circles
def touches (C1 C2 : Circle ℝ) : Prop := sorry

-- Define a touchesExternalTangent relation
def touchesExternalTangent (C1 C2 C3 : Circle ℝ) : Prop := sorry

theorem touching_circles_radii (R r : ℝ) (h : R > r) (h' : r > 0) :
  let x₁ := R * r / (Real.sqrt R + Real.sqrt r)^2
  let x₂ := R * r / (Real.sqrt R - Real.sqrt r)^2
  ∃ (x : ℝ), (x = x₁ ∨ x = x₂) ∧
    (∃ (C : Circle ℝ), C.radius = x ∧
      touches C (Circle.mk (0, 0) R) ∧
      touches C (Circle.mk (R + r, 0) r) ∧
      touchesExternalTangent C (Circle.mk (0, 0) R) (Circle.mk (R + r, 0) r)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circles_radii_l632_63294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_social_network_theorem_l632_63289

/-- Represents a social network as a simple graph -/
structure SocialNetwork where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  symmetric : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges

/-- The operation that can be performed on the social network -/
def perform_operation (G : SocialNetwork) (a b c : ℕ) : SocialNetwork :=
  sorry

/-- The initial condition of the social network -/
def initial_condition (G : SocialNetwork) : Prop :=
  (G.vertices.card = 2019) ∧
  (∃ S₁ S₂ : Finset ℕ, S₁.card = 1010 ∧ S₂.card = 1009 ∧
    S₁ ∪ S₂ = G.vertices ∧ S₁ ∩ S₂ = ∅ ∧
    (∀ v ∈ S₁, (G.edges.filter (λ e ↦ e.1 = v)).card = 1009) ∧
    (∀ v ∈ S₂, (G.edges.filter (λ e ↦ e.1 = v)).card = 1010))

/-- The final condition we want to prove -/
def final_condition (G : SocialNetwork) : Prop :=
  ∀ v ∈ G.vertices, (G.edges.filter (λ e ↦ e.1 = v)).card ≤ 1

/-- The main theorem to be proved -/
theorem social_network_theorem (G : SocialNetwork) :
  initial_condition G →
  ∃ (seq : List (ℕ × ℕ × ℕ)), final_condition (seq.foldl (λ g t ↦ perform_operation g t.1 t.2.1 t.2.2) G) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_social_network_theorem_l632_63289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_discount_percentage_l632_63270

/-- Represents the sales discount percentage -/
def discount_percentage : ℝ → Prop := λ D => D ≥ 0 ∧ D ≤ 100

/-- The number of items sold increases by 15% after applying the discount -/
axiom items_sold_increase : ∀ D : ℝ, (1 - D / 100) * 1.15 = 1.035

theorem sales_discount_percentage :
  ∀ D : ℝ, discount_percentage D → D = 10 :=
by
  intro D h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_discount_percentage_l632_63270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l632_63290

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-3, 4)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_properties :
  vector_AB = (-4, 2) ∧
  magnitude vector_AB = 2 * Real.sqrt 5 ∧
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) / (magnitude (A.1 - O.1, A.2 - O.2) * magnitude (B.1 - O.1, B.2 - O.2)) = Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l632_63290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_function_m_range_l632_63286

-- Define the exponential function as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3*m - 1)^x

-- State the theorem
theorem decreasing_exp_function_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x > f m y) → (1/3 < m ∧ m < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_function_m_range_l632_63286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_theorem_l632_63218

def num_male_athletes : ℕ := 5
def num_female_athletes : ℕ := 4
def num_athletes_to_select : ℕ := 4

def total_selections : ℕ := (Nat.choose (num_male_athletes + num_female_athletes) num_athletes_to_select) -
                             (Nat.choose num_male_athletes num_athletes_to_select) -
                             (Nat.choose num_female_athletes num_athletes_to_select)

def selections_without_A_and_B : ℕ := (Nat.choose (num_male_athletes + num_female_athletes - 2) num_athletes_to_select) - 1

theorem athlete_selection_theorem :
  total_selections - selections_without_A_and_B = 86 := by
  sorry

#eval total_selections - selections_without_A_and_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_theorem_l632_63218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_is_abelian_l632_63214

theorem group_is_abelian (G : Type*) [Group G] [Fintype G]
  (h : ∀ (a b : G), a^2 * b = b * a^2 → a * b = b * a)
  (h_order : ∃ (n : ℕ), Fintype.card G = 2^n) : 
  ∀ (x y : G), x * y = y * x := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_is_abelian_l632_63214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l632_63283

open Real

noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 3)

theorem g_monotone_increasing :
  StrictMonoOn g (Set.Icc (-π/12) (π/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l632_63283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_relation_l632_63222

theorem set_cardinality_relation (a b : ℕ) (A B : Finset ℤ) 
  (h1 : a > 0) (h2 : b > 0)
  (h3 : A ∩ B = ∅)
  (h4 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * Finset.card A = b * Finset.card B := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_relation_l632_63222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_one_to_infinity_l632_63250

/-- The function f(x) defined as (1/2)^(-x^2 + 2x) -/
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x)

/-- The theorem stating that f(x) is monotonically increasing on [1, +∞) -/
theorem f_monotone_increasing_on_one_to_infinity :
  MonotoneOn f (Set.Ici 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_one_to_infinity_l632_63250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9n_equals_9_l632_63239

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < j → i < digits.length → j < digits.length → digits[i]! < digits[j]!

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that for any natural number N with strictly increasing digits,
    the sum of digits of 9N is equal to 9 -/
theorem sum_of_digits_9n_equals_9 (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 := by
  sorry

#eval sum_of_digits 123  -- Example usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9n_equals_9_l632_63239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l632_63288

/-- The distance from a point to a plane in 3D space -/
noncomputable def distance_point_to_plane (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- The theorem stating that the distance from (1, 2, 3) to the plane x + 2y + 2z - 4 = 0 is 7/3 -/
theorem distance_to_specific_plane :
  distance_point_to_plane 1 2 3 1 2 2 (-4) = 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l632_63288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_second_quadrant_l632_63236

theorem sin_2theta_second_quadrant (θ : ℝ) (h1 : π/2 < θ ∧ θ < π) 
  (h2 : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 5/9) : 
  Real.sin (2 * θ) = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_second_quadrant_l632_63236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_class_instructors_l632_63234

/-- The number of instructors in Alice's white water rafting class -/
def num_instructors (total_students : ℕ) (life_vests_on_hand : ℕ) (student_percent_with_vests : ℚ) (additional_vests_needed : ℕ) : ℕ :=
  (additional_vests_needed : ℤ) - ((total_students : ℤ) - (↑(total_students : ℚ) * student_percent_with_vests).floor - (life_vests_on_hand : ℤ)) |>.toNat

theorem alice_class_instructors :
  num_instructors 40 20 (1/5) 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_class_instructors_l632_63234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l632_63272

/-- Circle C in the Cartesian plane -/
def CircleC : Set (ℝ × ℝ) := {p | (p.fst - 1)^2 + p.snd^2 = 5}

/-- Point M -/
def M : ℝ × ℝ := (-4, 0)

/-- Line l passing through M(-4,0) -/
def LineL (k : ℝ) : Set (ℝ × ℝ) := {p | p.snd = k * (p.fst + 4)}

/-- Points A and B are the intersections of LineL and CircleC -/
def IntersectionPoints (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ CircleC ∧ B ∈ CircleC ∧ A ∈ LineL k ∧ B ∈ LineL k ∧ A ≠ B

/-- Point A is the midpoint of MB -/
def AMidpointMB (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.fst = (M.fst + B.fst) / 2 ∧ A.snd = (M.snd + B.snd) / 2

theorem line_equation : 
  ∀ k : ℝ, IntersectionPoints k → 
  (∃ A B : ℝ × ℝ, AMidpointMB k A B) → 
  k = 1/3 ∨ k = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l632_63272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l632_63245

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt x - 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 0 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l632_63245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l632_63244

/-- Represents the time it takes to empty a cistern with a leak -/
noncomputable def time_to_empty (fill_time_no_leak fill_time_with_leak : ℝ) : ℝ :=
  (fill_time_no_leak * fill_time_with_leak) / (fill_time_with_leak - fill_time_no_leak)

/-- Theorem stating that for a cistern that takes 7 hours to fill without a leak
    and 8 hours with a leak, it will take 56 hours for the leak to empty the full cistern -/
theorem cistern_leak_emptying_time :
  time_to_empty 7 8 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l632_63244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l632_63258

theorem unique_function_property (f : ℕ → ℕ) :
  (∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l632_63258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l632_63278

-- Define the radii of the circles
noncomputable def small_radius : ℝ := 3
noncomputable def large_radius : ℝ := 6

-- Define the areas of the rectangles
noncomputable def small_rectangle_area : ℝ := 2 * small_radius * small_radius
noncomputable def large_rectangle_area : ℝ := 2 * large_radius * large_radius

-- Define the areas of the semicircles
noncomputable def small_semicircle_area : ℝ := (1/2) * Real.pi * small_radius^2
noncomputable def large_semicircle_area : ℝ := (1/2) * Real.pi * large_radius^2

-- Theorem statement
theorem shaded_area_calculation :
  (small_rectangle_area - small_semicircle_area) + (large_rectangle_area - large_semicircle_area) = 90 - 22.5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l632_63278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l632_63251

/-- Pyramid with rectangular base -/
structure RectangularBasePyramid where
  -- Base dimensions
  ab : ℝ
  bc : ℝ
  -- Height
  pa : ℝ
  -- Perpendicularity conditions
  pa_perp_ab : True
  pa_perp_ad : True

/-- Volume of a pyramid with rectangular base -/
noncomputable def pyramidVolume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.ab * p.bc * p.pa

/-- The theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∃ (p : RectangularBasePyramid),
    p.ab = 8 ∧ p.bc = 4 ∧ p.pa = 6 ∧ pyramidVolume p = 64 := by
  -- Construct the specific pyramid
  let p : RectangularBasePyramid := {
    ab := 8
    bc := 4
    pa := 6
    pa_perp_ab := True.intro
    pa_perp_ad := True.intro
  }
  -- Prove the existence
  use p
  -- Prove the conditions
  constructor
  · rfl  -- p.ab = 8
  constructor
  · rfl  -- p.bc = 4
  constructor
  · rfl  -- p.pa = 6
  -- Prove the volume
  calc
    pyramidVolume p = (1 / 3) * 8 * 4 * 6 := rfl
    _ = 64 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l632_63251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_monthly_expenses_l632_63291

theorem roberts_monthly_expenses 
  (basic_salary : ℝ) 
  (commission_rate : ℝ) 
  (total_sales : ℝ) 
  (savings_rate : ℝ) 
  (h1 : basic_salary = 1250)
  (h2 : commission_rate = 0.1)
  (h3 : total_sales = 23600)
  (h4 : savings_rate = 0.2) : 
  (basic_salary + commission_rate * total_sales) * (1 - savings_rate) = 2888 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_monthly_expenses_l632_63291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_digit_sum_l632_63223

theorem exists_number_divisible_by_digit_sum (n : ℕ) : 
  ∃ k : ℕ, 
    (Nat.digits 10 k).length = n ∧ 
    (∀ d ∈ Nat.digits 10 k, d ≠ 0) ∧
    k % (Nat.digits 10 k).sum = 0 := by
  sorry

#check exists_number_divisible_by_digit_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_digit_sum_l632_63223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brass_mixtures_zinc_amount_l632_63248

/-- Calculates the amount of zinc in a brass mixture given the total mass and the copper-to-zinc ratio --/
noncomputable def zinc_in_mixture (total_mass : ℝ) (copper : ℕ) (zinc : ℕ) : ℝ :=
  (zinc : ℝ) / ((copper : ℝ) + (zinc : ℝ)) * total_mass

/-- The total amount of zinc in two brass mixtures --/
noncomputable def total_zinc (mass1 mass2 : ℝ) (copper1 zinc1 copper2 zinc2 : ℕ) : ℝ :=
  zinc_in_mixture mass1 copper1 zinc1 + zinc_in_mixture mass2 copper2 zinc2

theorem brass_mixtures_zinc_amount :
  total_zinc 100 50 13 7 5 3 = 53.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brass_mixtures_zinc_amount_l632_63248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_tan_fraction_simplification_l632_63275

-- Problem 1
theorem sin_value_third_quadrant (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : α ∈ Set.Icc π (3*π/2)) :
  Real.sin α = -3/5 := by sorry

-- Problem 2
theorem tan_fraction_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos θ) / (2 * Real.sin θ + Real.cos θ) = 4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_tan_fraction_simplification_l632_63275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_l_l632_63287

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 * Real.sqrt 5 = 0

-- Define the curve C1
def curve_C1 (x y : ℝ) : Prop := (2*x + 1)^2 + y^2 = 4

-- Define the distance function from a point (x, y) to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - y + 2 * Real.sqrt 5) / Real.sqrt 2

-- State the theorem
theorem min_distance_C1_to_l :
  ∃ (d : ℝ), d = Real.sqrt (10 - 5 * Real.sqrt 2) ∧
  (∀ (x y : ℝ), curve_C1 x y → distance_to_line x y ≥ d) ∧
  (∃ (x y : ℝ), curve_C1 x y ∧ distance_to_line x y = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_l_l632_63287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_cannot_form_triangle_l632_63228

-- Define Triangle as a structure
structure Triangle (α : Type*) where
  edge1 : α
  edge2 : α
  edge3 : α

theorem triangle_inequality {α : Type*} [LinearOrderedField α] (a b c : α) : 
  a > 0 → b > 0 → c > 0 → 
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) → 
  ¬(∃ triangle : Triangle α, triangle.edge1 = a ∧ triangle.edge2 = b ∧ triangle.edge3 = c) :=
by
  sorry

theorem cannot_form_triangle : 
  ¬(∃ triangle : Triangle ℝ, triangle.edge1 = 7 ∧ triangle.edge2 = 11 ∧ triangle.edge3 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_cannot_form_triangle_l632_63228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_amount_calculation_l632_63285

-- Define the true discount formula
noncomputable def true_discount (face_value : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (face_value * rate * time) / (100 + (rate * time))

-- Define the theorem
theorem bill_amount_calculation 
  (discount : ℝ) (rate : ℝ) (time_months : ℝ) (face_value : ℝ) :
  discount = 240 ∧ 
  rate = 16 ∧ 
  time_months = 9 ∧
  true_discount face_value rate (time_months / 12) = discount →
  face_value = 2240 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_amount_calculation_l632_63285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_butanoic_acid_l632_63224

/-- Represents an element in a chemical compound -/
structure Element where
  symbol : String
  atomic_mass : Float

/-- Represents a chemical compound -/
structure Compound where
  formula : String
  elements : List (Element × Nat)

/-- Calculates the molar mass of a compound -/
def molar_mass (c : Compound) : Float :=
  c.elements.foldl (fun acc (el, count) => acc + el.atomic_mass * count.toFloat) 0

/-- Calculates the mass percentage of an element in a compound -/
def mass_percentage (el : Element) (c : Compound) : Float :=
  let el_count := (c.elements.filter (fun (e, _) => e.symbol == el.symbol)).head?.map Prod.snd |>.getD 0
  (el.atomic_mass * el_count.toFloat / molar_mass c) * 100

/-- Theorem: The mass percentage of carbon in butanoic acid is approximately 54.51% -/
theorem carbon_percentage_in_butanoic_acid :
  let carbon := Element.mk "C" 12.01
  let hydrogen := Element.mk "H" 1.008
  let oxygen := Element.mk "O" 16.00
  let butanoic_acid := Compound.mk "C4H8O2" [(carbon, 4), (hydrogen, 8), (oxygen, 2)]
  (mass_percentage carbon butanoic_acid - 54.51).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_butanoic_acid_l632_63224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_with_conditions_l632_63241

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_with_conditions :
  ∃ (a : ℕ+ → ℝ) (r : ℝ),
    is_geometric_sequence a ∧
    (∀ n : ℕ+, a (n + 1) > a n) ∧
    (∀ n : ℕ+, (Finset.range n).sum (λ i ↦ a ⟨i + 1, Nat.succ_pos i⟩) >
               (Finset.range (n + 1)).sum (λ i ↦ a ⟨i + 1, Nat.succ_pos i⟩)) ∧
    (0 < r ∧ r < 1) ∧
    (∀ n : ℕ+, a n = -(r^(n : ℕ))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_with_conditions_l632_63241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l632_63279

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 6)

theorem graph_shift (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (d : ℝ), ∀ (n : ℤ), f ω (d + n * (Real.pi / 2)) = 0) :
  ∀ x, g ω x = f ω (x - Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_l632_63279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l632_63208

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (-6 + 5 * Real.cos θ, 5 * Real.sin θ)

-- Define the line l
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_angle (α : ℝ) :
  (∃ θ₁ θ₂ t₁ t₂ : ℝ,
    curve_C θ₁ = line_l α t₁ ∧
    curve_C θ₂ = line_l α t₂ ∧
    distance (curve_C θ₁) (curve_C θ₂) = 2 * Real.sqrt 7) →
  α = π/4 ∨ α = 3*π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l632_63208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_sum_l632_63277

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := (3/4) * x^2 - 3*x + 4

-- State the theorem
theorem solution_set_sum (a b : ℝ) :
  (∀ x, a ≤ f x ∧ f x ≤ b ↔ a ≤ x ∧ x ≤ b) →
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_sum_l632_63277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l632_63237

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) := λ (x y : ℝ) => (m + 2) * x + y + 1 = 0
def l₂ (m : ℝ) := λ (x y : ℝ) => 3 * x + m * y + 4 * m - 3 = 0

-- Define when two lines are parallel
def parallel (f g : ℝ → ℝ → Prop) := ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g x y

-- Define the distance between two parallel lines
noncomputable def distance_parallel_lines (f g : ℝ → ℝ → Prop) := 2 * Real.sqrt 2

-- Define the shortest distance from a point to a line
noncomputable def shortest_distance (f : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := Real.sqrt 17

theorem line_properties (m : ℝ) :
  (¬ (∀ (x y : ℝ), l₁ (-3) x y ↔ l₂ (-3) x y)) ∧
  (parallel (l₁ m) (l₂ m) → m = 1) ∧
  (parallel (l₁ m) (l₂ m) → distance_parallel_lines (l₁ m) (l₂ m) = 2 * Real.sqrt 2) ∧
  (¬ (shortest_distance (l₁ m) 0 0 = Real.sqrt 17)) := by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l632_63237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_sqrt_two_satisfies_equation_l632_63205

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.exp (x * Real.log x)

/-- Theorem: √2 is the unique positive real solution to x^(x^(x^...)) = 4 -/
theorem infinite_power_tower_sqrt_two : 
  ∃! (x : ℝ), x > 0 ∧ infinitePowerTower x = 4 :=
by
  sorry

/-- Corollary: √2 satisfies the equation x^(x^(x^...)) = 4 -/
theorem sqrt_two_satisfies_equation :
  infinitePowerTower (Real.sqrt 2) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_sqrt_two_satisfies_equation_l632_63205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_division_problem_l632_63296

theorem coin_division_problem :
  ∃ n : ℕ, 
    n > 0 ∧
    (∀ m : ℕ, m > 0 → (m % 8 = 6 ∧ m % 7 = 2) → n ≤ m) ∧
    n % 8 = 6 ∧
    n % 7 = 2 ∧
    n % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_division_problem_l632_63296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_quarter_l632_63204

/-- Two equal cones with parallel bases and common height -/
structure EqualCones where
  radius : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def coneVolume (c : EqualCones) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- The volume of the intersection of two equal cones -/
noncomputable def intersectionVolume (c : EqualCones) : ℝ := 
  2 * ((1/3) * Real.pi * (c.radius/2)^2 * (c.height/2))

theorem intersection_volume_is_quarter (c : EqualCones) :
  intersectionVolume c = (1/4) * coneVolume c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_quarter_l632_63204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_and_length_l632_63230

-- Define the equation
def equation (x a : ℝ) : Prop := |x| = a * x - 2

-- Define the interval of a where the equation has no solutions
def no_solution_interval (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

-- Theorem stating the condition for no solutions and the interval length
theorem no_solution_and_length :
  (∀ a : ℝ, (∀ x : ℝ, ¬equation x a) ↔ no_solution_interval a) ∧
  MeasureTheory.volume (Set.Icc (-1 : ℝ) 1) = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_and_length_l632_63230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l632_63206

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_difference_magnitude (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 2) 
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 3) 
  (h3 : angle_between a b = π / 3) : 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l632_63206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_mailboxes_count_our_city_satisfies_eulers_formula_l632_63221

/-- A planar graph representing a city layout --/
structure CityGraph where
  -- Number of faces (excluding the outer face)
  faces : ℕ
  -- Number of edges
  edges : ℕ
  -- Number of vertices
  vertices : ℕ

/-- Euler's formula for planar graphs --/
theorem eulers_formula (g : CityGraph) :
  g.faces + 2 = g.edges - g.vertices + 2 := by
  sorry

/-- Our specific city layout --/
def our_city : CityGraph :=
  { faces := 12,
    edges := 37,
    vertices := 26 }

/-- Theorem: The number of vertices (mailboxes) in our city is 26 --/
theorem mailboxes_count :
  our_city.vertices = 26 := by
  rfl

/-- Theorem: Our city satisfies Euler's formula --/
theorem our_city_satisfies_eulers_formula :
  our_city.faces + 2 = our_city.edges - our_city.vertices + 2 := by
  rw [eulers_formula]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_mailboxes_count_our_city_satisfies_eulers_formula_l632_63221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_monotonicity_imply_m_range_l632_63200

/-- A function f with two distinct extreme value points -/
def has_two_distinct_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (∀ x, f x ≤ f x₁) ∧
    (∀ x, f x ≤ f x₂)

/-- A function f is monotonically decreasing on an interval [a, b] -/
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

/-- The main theorem -/
theorem extreme_points_and_monotonicity_imply_m_range (m : ℝ) :
  let f := λ (x : ℝ) => (1/3) * x^3 + x^2 + m*x + 1
  let g := λ (x : ℝ) => x^2 - m*x + 3
  (has_two_distinct_extreme_points f ∧ 
  ¬(is_monotone_decreasing_on g (-1) 2)) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_monotonicity_imply_m_range_l632_63200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_x_values_l632_63264

def is_valid_x (x : ℕ) : Prop :=
  x > 13 ∧ 203 % x = 13 ∧ 298 % x = 13

theorem valid_x_values : 
  ∀ x : ℕ, is_valid_x x ↔ (x = 19 ∨ x = 95) :=
by sorry

#check valid_x_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_x_values_l632_63264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_is_zero_l632_63213

def factorial_base_repr (n : ℕ) : List ℕ :=
  sorry

theorem a_5_is_zero (h : (factorial_base_repr 801).length ≥ 5) :
  (factorial_base_repr 801)[4] = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_is_zero_l632_63213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l632_63274

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (x : ℝ), f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l632_63274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_length_l632_63220

/-- The length of one side of the pentagons that aligns with one side of the square -/
noncomputable def z : ℝ := 5 * Real.sqrt 3

/-- The area of the original rectangle -/
def rectangle_area : ℝ := 10 * 15

/-- The side length of the square formed by the pentagons -/
noncomputable def square_side : ℝ := Real.sqrt (10 * 15)

theorem pentagon_side_length :
  ∃ (pentagon_area : ℝ),
    pentagon_area > 0 ∧
    2 * pentagon_area = rectangle_area ∧
    z * z = pentagon_area ∧
    2 * z = square_side := by
  sorry

#eval rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_side_length_l632_63220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditionally_constrained_functions_l632_63219

-- Define the concept of a conditionally constrained function
noncomputable def is_conditionally_constrained (f : ℝ → ℝ) : Prop :=
  ∃ ω : ℝ, ω > 0 ∧ ∀ x : ℝ, |f x| ≤ ω * |x|

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := λ x ↦ 4 * x
noncomputable def f2 : ℝ → ℝ := λ x ↦ x^2 + 2
noncomputable def f3 : ℝ → ℝ := λ x ↦ (2 * x) / (x^2 - 2*x + 5)

-- Define the properties of the fourth function
def f4_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x1 x2, f x1 - f x2 ≤ 4 * |x1 - x2|)

-- Theorem stating which functions are conditionally constrained
theorem conditionally_constrained_functions :
  is_conditionally_constrained f1 ∧
  ¬is_conditionally_constrained f2 ∧
  is_conditionally_constrained f3 ∧
  ∀ f, f4_properties f → is_conditionally_constrained f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditionally_constrained_functions_l632_63219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l632_63243

theorem max_value_theorem (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + a)) / x ≤ 4 / (2 * (Real.sqrt 2 + Real.sqrt a)) ∧
  ((x^2 + 2 - Real.sqrt (x^4 + a)) / x = 4 / (2 * (Real.sqrt 2 + Real.sqrt a)) ↔ x = a^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l632_63243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_is_negative_l632_63269

-- Define the second quadrant
def second_quadrant (α : ℝ) : Prop := Real.pi / 2 < α ∧ α < Real.pi

-- Define y as a function of α
noncomputable def y (α : ℝ) : ℝ := Real.sin (Real.cos α) * Real.cos (Real.sin (2 * α))

-- Theorem statement
theorem y_is_negative (α : ℝ) (h : second_quadrant α) : y α < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_is_negative_l632_63269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l632_63292

def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℕ := 120

theorem remaining_payment (total : ℕ) (h1 : deposit_amount = (deposit_percentage * ↑total).floor) : 
  total - deposit_amount = 1080 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l632_63292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_correct_l632_63257

noncomputable section

/-- The probability that a randomly selected point from a rectangle
    with vertices (±3, ±4) is within two units of the origin -/
def probability_within_two_units : ℝ := Real.pi / 12

/-- The width of the rectangle -/
def rectangle_width : ℝ := 8

/-- The length of the rectangle -/
def rectangle_length : ℝ := 6

/-- The area of the rectangle -/
def rectangle_area : ℝ := rectangle_width * rectangle_length

/-- The radius of the circle centered at the origin -/
def circle_radius : ℝ := 2

/-- The area of the circle centered at the origin with radius 2 -/
def circle_area : ℝ := Real.pi * circle_radius^2

theorem probability_correct :
  probability_within_two_units = circle_area / rectangle_area :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_correct_l632_63257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_14_77_l632_63235

noncomputable section

-- Define the markup percentage
def markup : ℝ := 64.28571428571428

-- Define the profit percentage
def profit : ℝ := 40

-- Define the function to calculate the discount percentage
noncomputable def discount_percentage (markup : ℝ) (profit : ℝ) : ℝ :=
  let normal_price := 1 + markup / 100
  let sale_price := 1 + profit / 100
  (normal_price - sale_price) / normal_price * 100

-- Theorem statement
theorem discount_approx_14_77 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |discount_percentage markup profit - 14.77| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_approx_14_77_l632_63235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_condition_characterization_l632_63260

-- Define a polynomial over real numbers
def MyPolynomial := ℝ → ℝ

-- Define the property that p(x) · p(x+1) = p(x+p(x))
def SatisfiesCondition (p : MyPolynomial) : Prop :=
  ∀ x, p x * p (x + 1) = p (x + p x)

-- Define constant polynomials
def ConstantPoly (c : ℝ) : MyPolynomial := λ _ ↦ c

-- Define quadratic polynomials
def QuadraticPoly (a b c : ℝ) : MyPolynomial := λ x ↦ a * x^2 + b * x + c

theorem polynomial_condition_characterization :
  ∀ p : MyPolynomial,
    SatisfiesCondition p ↔
      (p = ConstantPoly 0 ∨
       p = ConstantPoly 1 ∨
       ∃ b c, p = QuadraticPoly 1 b c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_condition_characterization_l632_63260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_g_value_h_range_bounds_l632_63295

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (cos (x + π / 12)) ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / 2) * sin (2 * x)
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem 1
theorem symmetry_axis_g_value (x₀ : ℝ) :
  (∀ x, f (x₀ + x) = f (x₀ - x)) →
  g x₀ = 3/4 ∨ g x₀ = 5/4 := by sorry

-- Theorem 2
theorem h_range_bounds (m : ℝ) :
  (∀ x ∈ Set.Icc (-π/12) (5*π/12), |h x - m| ≤ 1) ↔
  1 ≤ m ∧ m ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_g_value_h_range_bounds_l632_63295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_angles_l632_63280

theorem cos_sum_specific_angles (α β : Real) :
  (∃ r₁ : Real, r₁ > 0 ∧ r₁ * Real.cos α = 1 ∧ r₁ * Real.sin α = 2) →
  (∃ r₂ : Real, r₂ > 0 ∧ r₂ * Real.cos β = -2 ∧ r₂ * Real.sin β = 6) →
  Real.cos (α + β) = -7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_angles_l632_63280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l632_63247

/-- A hyperbola is defined by the equation ax² - by² = k, where a and b are positive real numbers -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  k : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The given hyperbola from the problem -/
noncomputable def given_hyperbola : Hyperbola :=
  { a := 1/2, b := 1, k := 1, a_pos := by norm_num, b_pos := by norm_num }

/-- The point P through which the new hyperbola passes -/
def point_P : ℝ × ℝ := (2, -2)

/-- The new hyperbola we need to prove about -/
noncomputable def new_hyperbola : Hyperbola :=
  { a := 1/2, b := 1, k := -2, a_pos := by norm_num, b_pos := by norm_num }

theorem hyperbola_theorem :
  (new_hyperbola.a * point_P.1^2 - new_hyperbola.b * point_P.2^2 = new_hyperbola.k) ∧
  (new_hyperbola.a = given_hyperbola.a ∧ new_hyperbola.b = given_hyperbola.b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l632_63247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_not_sqrt3_div_2_l632_63215

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Theorem: The eccentricity of the ellipse x^2/2 + y^2 = 1 is not √3/2 -/
theorem ellipse_eccentricity_not_sqrt3_div_2 :
  eccentricity (Real.sqrt 2) 1 ≠ Real.sqrt 3 / 2 := by
  sorry

#check ellipse_eccentricity_not_sqrt3_div_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_not_sqrt3_div_2_l632_63215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l632_63242

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
noncomputable def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

/-- Condition that a point (x, y) is on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  x = p.x_coord y

/-- The vertex of a parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  let y := -p.b / (2 * p.a)
  (p.x_coord y, y)

theorem parabola_c_value (p : Parabola) :
  p.vertex = (-3, 2.5) →
  p.contains_point (-4) 5 →
  p.c = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_c_value_l632_63242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_batch_size_minimizes_cost_l632_63209

/-- The optimal number of items to produce in each batch -/
def optimal_batch_size : ℕ := 80

/-- The preparation cost for each batch in yuan -/
noncomputable def preparation_cost : ℝ := 800

/-- The storage cost per item per day in yuan -/
noncomputable def storage_cost_per_item_per_day : ℝ := 1

/-- The average storage time in days for a batch of size x -/
noncomputable def average_storage_time (x : ℝ) : ℝ := x / 8

/-- The total cost per item for a batch of size x -/
noncomputable def total_cost_per_item (x : ℝ) : ℝ :=
  preparation_cost / x + storage_cost_per_item_per_day * average_storage_time x

/-- Theorem stating that the optimal batch size minimizes the total cost per item -/
theorem optimal_batch_size_minimizes_cost :
  ∀ x : ℝ, x > 0 → total_cost_per_item (optimal_batch_size : ℝ) ≤ total_cost_per_item x := by
  sorry

#check optimal_batch_size_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_batch_size_minimizes_cost_l632_63209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l632_63252

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0
def line (m p : ℝ) (x y : ℝ) : Prop := y = m*x + m*p/2

-- Define the intersection points
def intersection_points (p m x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
  line m p x₁ y₁ ∧ line m p x₂ y₂ ∧
  x₁ < x₂

-- Define the distance between intersection points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Define the directrix and line l
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x+1)

-- Define the dot product for F'M • F'N
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ + 1)*(x₂ + 1) + y₁*y₂

theorem parabola_and_line_theorem (p m x₁ y₁ x₂ y₂ k : ℝ) :
  parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
  line m p x₁ y₁ ∧ line m p x₂ y₂ ∧
  x₁ < x₂ ∧ m = Real.sqrt 2 ∧
  distance x₁ y₁ x₂ y₂ = 6 ∧
  directrix p (-p/2) ∧
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  dot_product x₁ y₁ x₂ y₂ = 12 →
  p = 2 ∧ (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l632_63252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correctAnswerIsMinutes_l632_63262

inductive TimeUnit
  | Hours
  | Minutes
  | Seconds

def correctTimeUnit : TimeUnit := TimeUnit.Minutes

theorem correctAnswerIsMinutes : correctTimeUnit = TimeUnit.Minutes := by
  rfl

#check correctAnswerIsMinutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correctAnswerIsMinutes_l632_63262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_five_eight_sided_die_l632_63225

/-- The probability of rolling a number less than 5 on a fair 8-sided die -/
theorem prob_less_than_five_eight_sided_die : 
  (4 : ℚ) / 8 = 1 / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_less_than_five_eight_sided_die_l632_63225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64m4_l632_63254

def is_valid_m (m : ℕ) : Prop :=
  (Finset.filter (· ∣ 120 * m^3) (Finset.range (120 * m^3 + 1))).card = 120

theorem divisors_of_64m4 (m : ℕ) (h : is_valid_m m) :
  (Finset.filter (· ∣ 64 * m^4) (Finset.range (64 * m^4 + 1))).card = 675 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64m4_l632_63254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l632_63207

theorem solution_value (a b x y : ℝ) : 
  (Real.sqrt (x + y) + Real.sqrt (x - y) = 4) → 
  (x^2 - y^2 = 9) → 
  ((a, b) = (x, y)) → 
  (a * b) / (a + b) = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_l632_63207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_even_digit_numbers_l632_63266

/-- A function that checks if a natural number has all even digits -/
def allEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ Nat.digits 10 n → Even d

/-- A function that checks if a natural number has at least one odd digit -/
def hasOddDigit (n : ℕ) : Prop :=
  ∃ d, d ∈ Nat.digits 10 n ∧ Odd d

/-- The theorem stating the maximum difference between two 6-digit numbers
    with all even digits and any number between them having at least one odd digit -/
theorem max_diff_even_digit_numbers :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    allEvenDigits a ∧
    allEvenDigits b ∧
    (∀ n, a < n ∧ n < b → hasOddDigit n) ∧
    b - a = 111112 ∧
    (∀ a' b', (100000 ≤ a' ∧ a' < 1000000) →
              (100000 ≤ b' ∧ b' < 1000000) →
              allEvenDigits a' →
              allEvenDigits b' →
              (∀ n, a' < n ∧ n < b' → hasOddDigit n) →
              b' - a' ≤ 111112) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diff_even_digit_numbers_l632_63266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_36_l632_63293

/-- The sum of the positive factors of 36 is 91 -/
theorem sum_of_factors_36 : (Finset.filter (λ x => 36 % x = 0) (Finset.range 37)).sum id = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_36_l632_63293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l632_63282

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧ 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l632_63282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_approx_interest_condition_principal_satisfies_condition_l632_63265

/-- The principal amount that satisfies the given compound interest condition --/
noncomputable def principal_amount : ℝ :=
  2500 / (2 - (1 + 0.05)^6)

/-- Theorem stating that the principal amount is approximately 3787.88 --/
theorem principal_amount_approx :
  ∃ ε > 0, |principal_amount - 3787.88| < ε := by
  sorry

/-- The interest after 6 years is 2500 less than the principal --/
theorem interest_condition (P : ℝ) :
  P * (1 + 0.05)^6 - P = P - 2500 := by
  sorry

/-- The principal amount satisfies the interest condition --/
theorem principal_satisfies_condition :
  principal_amount * (1 + 0.05)^6 - principal_amount = principal_amount - 2500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_approx_interest_condition_principal_satisfies_condition_l632_63265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_gross_profit_l632_63298

/-- Calculates the gross profit for a desk sale given the purchase price and markup percentage. -/
theorem desk_gross_profit (purchase_price markup_percentage : ℚ) : 
  purchase_price = 150 →
  markup_percentage = 1/4 →
  let selling_price := purchase_price / (1 - markup_percentage)
  selling_price - purchase_price = 50 := by
  sorry

-- Remove the #eval line as it's not necessary for theorem proving

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_gross_profit_l632_63298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m6n6_in_expansion_l632_63238

theorem coefficient_m6n6_in_expansion : ∀ m n : ℕ,
  (Finset.range 13).sum (λ k ↦ (Nat.choose 12 k) * m^k * n^(12-k)) =
  924 * m^6 * n^6 + (Finset.range 13).sum (λ k ↦ if k ≠ 6 then (Nat.choose 12 k) * m^k * n^(12-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m6n6_in_expansion_l632_63238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l632_63201

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.exp (-x^2)) / x

theorem inequality_proof : 
  Real.log ((Real.sqrt 2009 + Real.sqrt 2010) / (Real.sqrt 2008 + Real.sqrt 2009)) < 
  ∫ x in Set.Icc (Real.sqrt 2008) (Real.sqrt 2009), f x ∧
  ∫ x in Set.Icc (Real.sqrt 2008) (Real.sqrt 2009), f x < 
  Real.sqrt 2009 - Real.sqrt 2008 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l632_63201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l632_63299

noncomputable section

-- Define the equations of the two parallel lines
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + 3 * m * y - m + 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x + 6 * y - 4 = 0

-- Define the distance between two parallel lines
noncomputable def distance_between_lines (A B C D E F : ℝ) : ℝ :=
  abs (C - F) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem parallel_lines_problem :
  ∃ (m : ℝ), m = 2 ∧
  (∀ (x y : ℝ), line1 m x y ↔ 2 * x + 6 * y = 0) ∧
  (∀ (x y : ℝ), line2 m x y ↔ 2 * x + 6 * y - 4 = 0) ∧
  distance_between_lines 2 6 0 2 6 (-4) = Real.sqrt 10 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_problem_l632_63299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_plus_area_sum_l632_63227

/-- A parallelogram in a coordinate plane with integer coordinates for vertices -/
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { v1 := (1, 0)
    v2 := (4, 3)
    v3 := (9, 3)
    v4 := (6, 0) }

/-- Calculate the length of a diagonal in a parallelogram -/
noncomputable def diagonalLength (p : Parallelogram) : ℝ :=
  Real.sqrt (((p.v3.1 - p.v1.1) ^ 2 + (p.v3.2 - p.v1.2) ^ 2) : ℝ)

/-- Calculate the area of a parallelogram -/
noncomputable def parallelogramArea (p : Parallelogram) : ℝ :=
  let base := ((p.v4.1 - p.v1.1).natAbs : ℝ)
  let height := Real.sqrt 2 / 2 * ((p.v2.2 - p.v1.2) * 2 : ℝ)
  base * height

/-- The main theorem to prove -/
theorem diagonal_plus_area_sum (p : Parallelogram) :
  p = specificParallelogram →
  diagonalLength p + parallelogramArea p = Real.sqrt 73 + 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_plus_area_sum_l632_63227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_plus_pi_sixth_l632_63249

theorem sin_double_plus_pi_sixth (α : Real) 
  (h1 : Real.sin α = 3/4) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (2*α + π/6) = -(3*Real.sqrt 21 + 1)/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_plus_pi_sixth_l632_63249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l632_63229

/-- The circumference of the base of a right circular cone formed from a 180-degree sector of a circle --/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (π * r) = 6 * π :=
by
  -- Substitute r = 6
  rw [h]
  -- Simplify
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l632_63229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_power_n_gt_odd_product_l632_63273

/-- The product of the first n odd positive integers -/
def odd_product (n : ℕ) : ℕ := (Finset.range n).prod (fun i => 2 * i + 1)

/-- For any positive integer n, n^n is greater than the product of the first n odd positive integers -/
theorem n_power_n_gt_odd_product (n : ℕ) (hn : n > 0) : n^n > odd_product n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_power_n_gt_odd_product_l632_63273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l632_63226

noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

noncomputable def ceil (x : ℝ) : ℤ := 
  -Int.floor (-x)

noncomputable def g (x : ℝ) : ℝ := 
  (floor x : ℝ) + (ceil x : ℝ) - 2 * x

-- Theorem statement
theorem g_range : 
  ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 ∧ 
  (∃ y : ℝ, g y = -1) ∧ 
  (∃ z : ℝ, g z = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l632_63226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l632_63297

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2) * x

/-- The theorem statement -/
theorem inequality_holds (a : ℝ) (h : a ≤ -1/2) :
  ∀ m n : ℝ, m > 0 → n > 0 → m ≠ n → (f a m - f a n) / (m - n) > a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l632_63297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_collected_640_l632_63212

/-- The amount of money Sara collected from selling cakes in 4 weeks -/
def sara_cake_money (cakes_per_day days_per_week weeks price : ℕ) : ℕ :=
  cakes_per_day * days_per_week * weeks * price

/-- Theorem stating that Sara collected $640 from selling cakes in 4 weeks -/
theorem sara_collected_640 : sara_cake_money 4 5 4 8 = 640 := by
  -- Unfold the definition of sara_cake_money
  unfold sara_cake_money
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_collected_640_l632_63212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tessellating_polygons_count_l632_63281

-- Define the list of regular polygons
inductive RegularPolygon
| Triangle
| Square
| Pentagon
| Hexagon
| Octagon
| Decagon
| Dodecagon

-- Define a function to check if a regular polygon can tessellate a plane
def canTessellatePlane (p : RegularPolygon) : Bool :=
  match p with
  | RegularPolygon.Triangle => true
  | RegularPolygon.Square => true
  | RegularPolygon.Hexagon => true
  | _ => false

-- Define the list of all regular polygons
def allPolygons : List RegularPolygon :=
  [RegularPolygon.Triangle, RegularPolygon.Square, RegularPolygon.Pentagon,
   RegularPolygon.Hexagon, RegularPolygon.Octagon, RegularPolygon.Decagon,
   RegularPolygon.Dodecagon]

-- Theorem statement
theorem tessellating_polygons_count :
  (allPolygons.filter canTessellatePlane).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tessellating_polygons_count_l632_63281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_meter_l632_63276

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure Plot where
  length : ℝ
  breadth : ℝ
  total_fencing_cost : ℝ

/-- Calculates the perimeter of a rectangular plot. -/
noncomputable def perimeter (p : Plot) : ℝ := 2 * (p.length + p.breadth)

/-- Calculates the cost per meter of fencing. -/
noncomputable def cost_per_meter (p : Plot) : ℝ := p.total_fencing_cost / perimeter p

/-- Theorem stating that for a plot with given dimensions and total fencing cost,
    the cost per meter of fencing is 26.5. -/
theorem fencing_cost_per_meter (p : Plot) 
    (h1 : p.length = 63)
    (h2 : p.breadth = 37)
    (h3 : p.total_fencing_cost = 5300) :
  cost_per_meter p = 26.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval cost_per_meter ⟨63, 37, 5300⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_meter_l632_63276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l632_63232

/-- An ellipse with the given properties has a major axis of length 8 -/
theorem ellipse_major_axis_length :
  ∀ (e : Set (ℝ × ℝ)),
    (∃ (x y : ℝ), (x, 0) ∈ e ∧ (0, y) ∈ e) →  -- tangent to x-axis and y-axis
    ((5, -4 + Real.sqrt 8) ∈ e ∧ (5, -4 - Real.sqrt 8) ∈ e) →  -- foci locations
    (∃ (a b : ℝ × ℝ), a ∈ e ∧ b ∈ e ∧ ‖a - b‖ = 8 ∧
      ∀ (c d : ℝ × ℝ), c ∈ e → d ∈ e → ‖c - d‖ ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l632_63232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l632_63256

-- Define the sets M and N
def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = 2^x}

-- State the theorem
theorem complement_intersection_M_N : 
  (Set.univ : Set ℝ) \ (M ∩ N) = Set.Iic (1/2) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_M_N_l632_63256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_six_l632_63202

/-- Ana's current age -/
def A : ℕ := sorry

/-- Bonita's current age -/
def B : ℕ := sorry

/-- The difference in years between Ana and Bonita's ages -/
def n : ℕ := sorry

/-- Ana is n years older than Bonita -/
axiom age_difference : A = B + n

/-- Last year, Ana was 7 times as old as Bonita -/
axiom last_year_ratio : A - 1 = 7 * (B - 1)

/-- This year, Ana's age is the cube of Bonita's age -/
axiom current_year_cube : A = B^3

theorem age_difference_is_six : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_six_l632_63202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l632_63203

theorem tangent_lines_to_circle : 
  ∃ (line1 line2 : ℝ → ℝ),
    (∀ x, line1 x = 7) ∧
    (∀ x, line2 x = -3/4 * x + 7) ∧
    (∀ line : ℝ → ℝ,
      (∃! x y, (x - 15)^2 + (y - 2)^2 = 25 ∧ y = line x) ∧
      line 0 = 7 →
      line = line1 ∨ line = line2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l632_63203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_one_l632_63233

-- Define the function g as a parameter
noncomputable def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else g x

-- Define the property of g
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f g x = g x

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem f_g_minus_one (g : ℝ → ℝ) 
  (h1 : g_property g) 
  (h2 : is_odd (f g)) : 
  f g (g (-1)) = -15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_minus_one_l632_63233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l632_63210

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 1 else 3 - x

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 5 := by
  -- Evaluate f(-1)
  have h1 : f (-1) = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(4)
  have h2 : f 4 = 5 := by
    simp [f]
    norm_num
  
  -- Combine the steps
  calc
    f (f (-1)) = f 4 := by rw [h1]
    _          = 5   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l632_63210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_walk_distance_l632_63255

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points using the Pythagorean theorem -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem chris_walk_distance :
  let a := Point.mk 0 0
  let b := Point.mk (-10) (-20)
  distance a b = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_walk_distance_l632_63255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_result_l632_63284

def fraction_product : ℕ → ℚ
  | 5     => 5 / 8
  | n + 8 => (n + 5 : ℚ) / (n + 8) * fraction_product n
  | _     => 1

theorem fraction_product_result :
  fraction_product 2002 = 1 / 401 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_result_l632_63284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_quarter_dollar_l632_63211

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
  | one
  | other

/-- The probability of rolling a 1 -/
noncomputable def prob_one : ℝ := 1/4

/-- The winnings for rolling a 1 -/
def win_one : ℝ := 8

/-- The loss for rolling any other number -/
def loss_other : ℝ := 3

/-- The probability of rolling any number other than 1 -/
noncomputable def prob_other : ℝ := 1 - prob_one

/-- The expected value of winnings after one roll of the biased die -/
noncomputable def expected_value : ℝ := prob_one * win_one - prob_other * loss_other

/-- Theorem stating that the expected value of winnings is -$0.25 -/
theorem expected_value_is_negative_quarter_dollar : expected_value = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_negative_quarter_dollar_l632_63211
