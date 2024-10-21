import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l933_93388

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 = 1

/-- Definition of the distance from a point to a focus -/
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

/-- Theorem: If a point on the ellipse is at distance 6 from one focus, 
    it is at distance 4 from the other focus -/
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (f1x f1y f2x f2y : ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_focus1 : distance_to_focus x y f1x f1y = 6) :
  distance_to_focus x y f2x f2y = 4 := by
  sorry

#check ellipse_focus_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l933_93388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l933_93318

theorem sin_plus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : 0 < α)
  (h3 : α < Real.pi/2) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l933_93318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l933_93354

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^3 - 1/x + 1

-- Theorem statement
theorem f_composition_value : f (1 / f 2) = 2/17 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l933_93354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_value_l933_93392

theorem max_cos_x_value (x y z : ℝ) 
  (h1 : Real.sin x = Real.tan y⁻¹)
  (h2 : Real.sin y = Real.tan z⁻¹)
  (h3 : Real.sin z = Real.tan x⁻¹) :
  ∃ (max_cos_x : ℝ), 
    (∀ x' y' z' : ℝ, 
      Real.sin x' = Real.tan y'⁻¹ → 
      Real.sin y' = Real.tan z'⁻¹ → 
      Real.sin z' = Real.tan x'⁻¹ → 
      Real.cos x' ≤ max_cos_x) ∧
    max_cos_x = Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_value_l933_93392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_two_points_l933_93347

/-- Two fixed points in a plane -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

/-- Distance between A and B is 4 units -/
axiom dist_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4

/-- Definition of a circle passing through both A and B with radius 3 -/
def CirclePassingAB (C : ℝ × ℝ) : Prop :=
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 ∧
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3

/-- The set of all centers of circles passing through A and B with radius 3 -/
def CenterSet : Set (ℝ × ℝ) :=
  {C | CirclePassingAB C}

/-- Theorem: The locus of centers consists of exactly two points -/
theorem locus_two_points : ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ CenterSet = {P, Q} := by
  sorry

#check locus_two_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_two_points_l933_93347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_inclination_l933_93335

/-- Given a line l with parametric equations x = -3 + t and y = 1 + √3t,
    prove its standard equation and angle of inclination. -/
theorem line_equation_and_inclination :
  ∃ (t : ℝ), 
    let x : ℝ := -3 + t
    let y : ℝ := 1 + Real.sqrt 3 * t
    (Real.sqrt 3 * x - y + 3 * Real.sqrt 3 + 1 = 0) ∧ 
    (Real.arctan (Real.sqrt 3) = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_inclination_l933_93335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_at_negative_two_l933_93326

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem derivative_of_exp_at_negative_two :
  (deriv f) (-2) = Real.exp (-2) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_at_negative_two_l933_93326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_probability_l933_93303

-- Define the cafeterias
inductive Cafeteria : Type
| A : Cafeteria
| B : Cafeteria

-- Define the days
inductive Day : Type
| Monday : Day
| Tuesday : Day

-- Define a function to represent going to a cafeteria on a specific day
def goTo (c : Cafeteria) (d : Day) : Prop := sorry

-- Define the probability function
noncomputable def P (event : Prop) : ℝ := sorry

-- Define the conditional probability function
noncomputable def P_cond (event condition : Prop) : ℝ := sorry

theorem cafeteria_probability :
  -- Xiao Li randomly chooses one cafeteria on Monday
  (P (goTo Cafeteria.A Day.Monday) = 0.5) →
  (P (goTo Cafeteria.B Day.Monday) = 0.5) →
  -- Probability of going to A on Tuesday given A on Monday
  (P_cond (goTo Cafeteria.A Day.Tuesday)
          (goTo Cafeteria.A Day.Monday) = 0.4) →
  -- Probability of going to A on Tuesday given B on Monday
  (P_cond (goTo Cafeteria.A Day.Tuesday)
          (goTo Cafeteria.B Day.Monday) = 0.6) →
  -- Conclusion: Probability of going to A on Tuesday
  P (goTo Cafeteria.A Day.Tuesday) = 0.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_probability_l933_93303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_intersection_l933_93349

/-- Sphere with center (0,0,1) and radius 1 in R^3 -/
def S : Set (Fin 3 → ℝ) :=
  {p | (p 0)^2 + (p 1)^2 + (p 2 - 1)^2 = 1}

/-- Plane tangent to S at point (x₀,y₀,z₀) -/
def P (x₀ y₀ z₀ : ℝ) : Set (Fin 3 → ℝ) :=
  {p | x₀ * (p 0 - x₀) + y₀ * (p 1 - y₀) + (z₀ - 1) * (p 2 - z₀) = 0}

theorem tangent_plane_intersection (x₀ y₀ z₀ : ℝ) 
  (h_positive : x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0)
  (h_tangent : (fun i => [x₀, y₀, z₀].get i) ∈ S)
  (h_intersection : ∀ p ∈ P x₀ y₀ z₀, p 2 = 0 → 2 * (p 0) + p 1 = 10) :
  z₀ = 40 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_plane_intersection_l933_93349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_puppies_and_parrots_l933_93316

theorem students_with_puppies_and_parrots 
  (total_students : ℕ) 
  (puppy_percentage : ℚ) 
  (parrot_percentage : ℚ) 
  (h1 : total_students = 40)
  (h2 : puppy_percentage = 80 / 100)
  (h3 : parrot_percentage = 25 / 100) :
  ⌊(total_students : ℚ) * puppy_percentage * parrot_percentage⌋ = 8 := by
  sorry

#check students_with_puppies_and_parrots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_with_puppies_and_parrots_l933_93316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_shot_theorem_optimal_shot_distance_positive_optimal_shot_angle_range_l933_93362

/-- Represents the geometry of a football field and goal --/
structure FootballField where
  a : ℝ  -- width of the football field
  b : ℝ  -- width of the goal
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_b_lt_a : b < a

/-- Calculates the optimal shooting distance and maximum scoring angle --/
noncomputable def optimal_shot (field : FootballField) :
  ℝ × ℝ :=
  let x := (1/2) * Real.sqrt (field.a^2 - field.b^2)
  let θ := Real.arctan (field.b / Real.sqrt (field.a^2 - field.b^2))
  (x, θ)

/-- Theorem stating the optimal shooting distance and maximum scoring angle --/
theorem optimal_shot_theorem (field : FootballField) :
  let (x, θ) := optimal_shot field
  x = (1/2) * Real.sqrt (field.a^2 - field.b^2) ∧
  θ = Real.arctan (field.b / Real.sqrt (field.a^2 - field.b^2)) := by
  sorry

/-- Theorem stating that the optimal shooting distance is positive --/
theorem optimal_shot_distance_positive (field : FootballField) :
  let (x, _) := optimal_shot field
  0 < x := by
  sorry

/-- Theorem stating that the maximum scoring angle is between 0 and π/2 --/
theorem optimal_shot_angle_range (field : FootballField) :
  let (_, θ) := optimal_shot field
  0 ≤ θ ∧ θ < π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_shot_theorem_optimal_shot_distance_positive_optimal_shot_angle_range_l933_93362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l933_93367

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + Real.sin x * Real.cos x

theorem f_range : Set.Icc (-1 : ℝ) (1/2 + Real.sqrt 2) = Set.range f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l933_93367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_l933_93375

theorem sum_of_coefficients_expansion : 
  ((fun x => (x^3 + 2*x + 1) * (3*x^2 + 4)) 1) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_l933_93375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l933_93361

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Helper function to calculate area (marked as noncomputable)
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c^2 - 2 * t.a^2 = t.b^2) 
  (h2 : t.a = 1) 
  (h3 : Real.tan t.A = 1/3) :
  (t.c * Real.cos t.A - t.a * Real.cos t.C) / t.b = 1/2 ∧ 
  area t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l933_93361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_ratio_l933_93344

theorem classroom_ratio :
  ∀ (num_boys num_girls : ℚ),
  (num_boys > 0) →
  (num_girls > 0) →
  (num_boys / (num_boys + num_girls) = 
   (3/4) * (num_girls / (num_boys + num_girls))) →
  (num_boys / (num_boys + num_girls) = 3/7) :=
by
  intro num_boys num_girls h1 h2 h3
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_ratio_l933_93344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l933_93317

/-- The upper integer part of a real number -/
noncomputable def upperIntegerPart (x : ℝ) : ℤ :=
  ⌈x⌉

/-- Some mathematical operation between a real number and a natural number -/
noncomputable def someOperation (a : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The distance to the nearest perfect square -/
def distanceToNearestSquare (x : ℤ) : ℕ :=
  sorry

/-- Theorem stating the existence of a special number A -/
theorem exists_special_number : ∃ (A : ℝ), ∀ (n : ℕ),
  distanceToNearestSquare (upperIntegerPart (someOperation A n)) = 2 := by
  sorry

#check exists_special_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l933_93317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l933_93365

/-- The quadratic function y = 2x^2 + 3x + r -/
def f (x r : ℝ) : ℝ := 2 * x^2 + 3 * x + r

/-- The maximum value of f on the interval [-2,0] is 4 -/
def max_value_condition (r : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (-2) 0 → f x r ≤ 4 ∧ ∃ y ∈ Set.Icc (-2) 0, f y r = 4

/-- If the maximum value of f on [-2,0] is 4, then r = 2 -/
theorem quadratic_max_value (r : ℝ) :
  max_value_condition r → r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_max_value_l933_93365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangerine_in_third_row_l933_93328

-- Define the types of trees
inductive TreeType
| Apple
| Pear
| Orange
| Lemon
| Tangerine

-- Define a row as a natural number from 0 to 4 (representing 1 to 5)
def Row := Fin 5

-- Define the orchard as a function from Row to TreeType
def Orchard := Row → TreeType

-- Define the property of being adjacent
def adjacent (r1 r2 : Row) : Prop := 
  (r1.val + 1 = r2.val ∧ r1.val < 4) ∨ (r2.val + 1 = r1.val ∧ r2.val < 4)

-- State the theorem
theorem tangerine_in_third_row (o : Orchard) :
  (∃ r1 r2 : Row, adjacent r1 r2 ∧ o r1 = TreeType.Orange ∧ o r2 = TreeType.Lemon) →
  (∀ r1 r2 : Row, o r1 = TreeType.Pear → (o r2 ≠ TreeType.Orange ∧ o r2 ≠ TreeType.Lemon)) →
  (∃ r1 r2 : Row, adjacent r1 r2 ∧ o r1 = TreeType.Apple ∧ o r2 = TreeType.Pear) →
  (∀ r1 r2 : Row, o r1 = TreeType.Apple → (o r2 ≠ TreeType.Orange ∧ o r2 ≠ TreeType.Lemon)) →
  o ⟨2, by norm_num⟩ = TreeType.Tangerine :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangerine_in_third_row_l933_93328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_meeting_probability_l933_93371

/-- The probability of two objects meeting on a coordinate plane --/
theorem object_meeting_probability : ℚ :=
  let total_steps := 9
  let meeting_points := List.range 7
  let paths_c := λ i => Nat.choose total_steps i
  let paths_d := λ i => Nat.choose total_steps i
  let total_paths := 2^total_steps
  let probability := (meeting_points.map (λ i => (paths_c i * paths_d i)) |>.sum) / (total_paths * total_paths)
  have : probability = 42544 / 262144 := by sorry
  42544 / 262144


end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_meeting_probability_l933_93371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stick_length_5_4_3_box_l933_93325

/-- The maximum length of a stick that can fit in a rectangular box -/
noncomputable def max_stick_length (length width height : ℝ) : ℝ :=
  Real.sqrt (length^2 + width^2 + height^2)

/-- Theorem: The maximum length of a stick in a 5x4x3 box is 5√2 -/
theorem max_stick_length_5_4_3_box :
  max_stick_length 5 4 3 = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stick_length_5_4_3_box_l933_93325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_lt_alpha_lt_tan_alpha_l933_93358

theorem sin_alpha_lt_alpha_lt_tan_alpha :
  ∀ α : Real, 0 < α → α < Real.pi / 2 → Real.sin α < α ∧ α < Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_lt_alpha_lt_tan_alpha_l933_93358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_B_eq_one_l933_93324

def A : Finset ℤ := {-1, 0, 1, 2, 3}

def B : Finset ℤ := A.filter (fun x => (1 - x) ∉ A)

theorem card_B_eq_one : Finset.card B = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_B_eq_one_l933_93324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_expr_implies_m_eq_5_l933_93396

noncomputable section

/-- G is a quadratic expression in x with parameter m -/
def G (x m : ℝ) : ℝ := (8*x^2 + 20*x + 5*m) / 8

/-- A linear expression in x -/
def linear_expr (x c d : ℝ) : ℝ := c*x + d

theorem square_of_linear_expr_implies_m_eq_5 :
  ∀ m : ℝ, (∃ c d : ℝ, ∀ x : ℝ, G x m = (linear_expr x c d)^2) → m = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_expr_implies_m_eq_5_l933_93396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_permutations_count_l933_93357

/-- The number of permutations of 5 distinct elements where two specific elements
    are always adjacent and two other specific elements are never adjacent -/
def special_permutations : ℕ := 24

/-- The total number of permutations of 5 distinct elements -/
def total_permutations : ℕ := Nat.factorial 5

/-- The number of permutations where two specific elements are adjacent -/
def adjacent_permutations : ℕ := 2 * Nat.factorial 4

/-- The number of permutations where two specific elements are adjacent,
    considering the other two specific elements as a unit -/
def adjacent_and_unit_permutations : ℕ := 2 * 2 * Nat.factorial 3

theorem special_permutations_count :
  special_permutations = adjacent_permutations - adjacent_and_unit_permutations :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_permutations_count_l933_93357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_area_of_triangle_l933_93329

/-- The area of the incircle of a triangle ABC with given conditions -/
theorem incircle_area_of_triangle (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  Real.cos A = 3/4 →
  Real.sin B = (5 * Real.sqrt 7) / 16 →
  c > 4 →
  let S := (1/2) * a * c * Real.sin B
  let l := a + b + c
  let r := (2 * S) / l
  (π * r^2) = (7/4) * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_area_of_triangle_l933_93329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_chocolates_l933_93322

/-- Represents a distribution of chocolates among children -/
def ChocolateDistribution := Fin 4 → ℕ

/-- The total number of chocolates -/
def totalChocolates : ℕ := 50

/-- A valid distribution satisfies the problem conditions -/
def isValidDistribution (d : ChocolateDistribution) : Prop :=
  (∀ i j, i ≠ j → d i ≠ d j) ∧ (Finset.sum (Finset.univ : Finset (Fin 4)) d = totalChocolates)

/-- The maximum number of chocolates in a distribution -/
def maxChocolates (d : ChocolateDistribution) : ℕ :=
  Finset.sup (Finset.univ : Finset (Fin 4)) d

/-- The theorem to be proved -/
theorem min_max_chocolates :
  ∀ d : ChocolateDistribution, isValidDistribution d →
    14 ≤ maxChocolates d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_chocolates_l933_93322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l933_93360

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

noncomputable def work_done_in_two_days (rate_a rate_b rate_c : ℝ) : ℝ :=
  (rate_a + rate_b) + (rate_a + rate_c)

theorem work_completion_time 
  (rate_a rate_b rate_c : ℝ) 
  (ha : rate_a = work_rate 11)
  (hb : rate_b = work_rate 45)
  (hc : rate_c = work_rate 55) :
  ∃ n : ℕ, n * (work_done_in_two_days rate_a rate_b rate_c) ≥ 1 ∧ 
  n * 2 = 10 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l933_93360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_relationship_l933_93391

-- Define the circles
structure Circle where
  radius : ℝ

-- Define the relationship between the circles
def externally_tangent_and_passes_through_center (c1 c2 : Circle) : Prop :=
  c1.radius * 2 = c2.radius

-- Define the area of a circle
noncomputable def area (c : Circle) : ℝ :=
  Real.pi * c.radius^2

-- Theorem statement
theorem circle_area_relationship (c1 c2 : Circle) 
  (h1 : externally_tangent_and_passes_through_center c1 c2) 
  (h2 : area c1 = 16) : 
  area c2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_relationship_l933_93391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l933_93327

theorem triangle_angle_measure (a b c : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 13) :
  ∃ C : ℝ, 0 < C ∧ C < π ∧ Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) ∧ C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l933_93327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_inequality_l933_93339

-- Define a structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the theorem
theorem line_points_inequality (A B C D E : Point)
  (h1 : A.y = B.y ∧ B.y = C.y ∧ C.y = D.y)  -- A, B, C, D are on the same line
  (h2 : A.x < B.x ∧ B.x < C.x ∧ C.x < D.x)  -- A, B, C, D are in order
  (h3 : E.y ≠ A.y) :  -- E is not on the line
  distance E A + distance E D + |distance A B - distance C D| > distance E B + distance E C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_inequality_l933_93339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_g_condition_l933_93363

noncomputable section

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (Real.exp x)

def g (m : ℝ) (x : ℝ) : ℝ := x * f 1 x + m * (deriv (f 1)) x + 1 / (Real.exp x)

theorem tangent_line_and_g_condition 
  (a b : ℝ)
  (h1 : ∀ x, (deriv (f a)) 0 * x + f a 0 = b)
  (h2 : ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ 2 * g m x₁ < g m x₂)
  (h3 : m > 0) :
  a = 1 ∧ b = 1 ∧ (m ∈ Set.Ioo 0 (1/3) ∪ Set.Ioi (5/2)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_g_condition_l933_93363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pet_time_l933_93308

/-- Represents the time Marcy spends with her pets -/
def pet_time (cat dog parrot : ℝ) : ℝ := cat + dog + parrot

/-- Time spent with the cat -/
noncomputable def cat_time : ℝ := 12 + (1/3 * 12) + (1/4 * (1/3 * 12)) + (1/2 * 12) + 5 + (2/5 * 5)

/-- Time spent with the dog -/
noncomputable def dog_time : ℝ := 18 + (2/3 * 18) + 9 + (1/3 * 9) + (1/4 * 18)

/-- Time spent with the parrot -/
noncomputable def parrot_time : ℝ := 15 + 8 + (1/2 * 8)

theorem total_pet_time : pet_time cat_time dog_time parrot_time = 103.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pet_time_l933_93308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l933_93302

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

-- State the theorem
theorem f_extrema :
  (∀ x, f x ≥ -2) ∧ 
  (∃ x, f x = -2) ∧
  (∀ x, f x ≤ 7) ∧ 
  (∃ x, f x = 7) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l933_93302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l933_93359

-- Define the dimensions of the paper
def paper_width : ℝ := 10
def paper_length : ℝ := 12

-- Define the cylinders
def cylinder_amy : ℝ × ℝ := (paper_width, paper_length)
def cylinder_belinda : ℝ × ℝ := (paper_length, paper_width)

-- Function to calculate cylinder volume
noncomputable def cylinder_volume (circumference height : ℝ) : ℝ :=
  (circumference^2 * height) / (4 * Real.pi)

-- Theorem statement
theorem cylinder_volume_difference : 
  Real.pi * |cylinder_volume cylinder_amy.1 cylinder_amy.2 - 
       cylinder_volume cylinder_belinda.1 cylinder_belinda.2| = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l933_93359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l933_93382

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A set of points is coverable if there exists a point such that
    all points in the set are within distance 1 from it -/
def isCoverable (S : Set Point) : Prop :=
  ∃ c : Point, ∀ p ∈ S, distance c p ≤ 1

theorem circle_coverage (points : Finset Point) :
  (points.card = 25) →
  (∀ p q r : Point, p ∈ points → q ∈ points → r ∈ points →
    distance p q < 1 ∨ distance p r < 1 ∨ distance q r < 1) →
  ∃ S : Finset Point, S ⊆ points ∧ S.card ≥ 13 ∧ isCoverable S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l933_93382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l933_93356

noncomputable def f (x : ℝ) : ℝ := 2 - 4 / (3^x + 1)

theorem f_properties :
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  (∀ x, f (-x) = -f x) ∧
  (Set.Ioi (5/3) = {x | f (2*x - 1) + f (x - 4) > 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l933_93356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suma_work_time_l933_93364

-- Define the work rate as the inverse of the time taken to complete the task
noncomputable def workRate (days : ℝ) : ℝ := 1 / days

-- Theorem statement
theorem suma_work_time (renu_days : ℝ) (combined_days : ℝ) (suma_days : ℝ) :
  renu_days = 6 →
  combined_days = 4 →
  workRate renu_days + workRate suma_days = workRate combined_days →
  suma_days = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suma_work_time_l933_93364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_avoiding_lattice_points_l933_93368

theorem max_slope_avoiding_lattice_points :
  let max_a : ℚ := 101 / 100
  ∀ a : ℚ, (∀ m : ℚ, 1 < m → m < a →
    ∀ x : ℤ, 10 < x → x ≤ 200 →
      ∀ y : ℤ, y ≠ (m * ↑x + 5).floor) →
  a ≤ max_a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_avoiding_lattice_points_l933_93368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l933_93346

/-- The number of solutions to x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2 for 1 ≤ x ≤ n -/
def numSolutions (n : ℕ) : ℕ :=
  n^2 - n + 1

/-- The equation x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2 -/
def satisfiesEquation (x : ℝ) : Prop :=
  x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2

theorem count_solutions (n : ℕ) :
  (∃ s : Finset ℝ, s.card = numSolutions n ∧
    ∀ x ∈ s, satisfiesEquation x ∧ 1 ≤ x ∧ x ≤ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l933_93346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_circ_equation_l933_93393

/-- The operation ∘ defined for real numbers -/
def circ (x y : ℝ) : ℝ := 5*x + 4*y - x*y^2

/-- Theorem stating that there are exactly two real solutions to 2 ∘ y = 9 -/
theorem two_solutions_for_circ_equation :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ y ∈ s, circ 2 y = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_circ_equation_l933_93393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l933_93321

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a) ^ 2)

/-- A point on the ellipse satisfies its equation -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_dimensions (e : Ellipse) 
    (h_contains : e.contains 1 (3/2))
    (h_ecc : e.eccentricity = 1/2) :
    e.a^2 = 4 ∧ e.b^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l933_93321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_0_05_l933_93394

/-- Represents the colors of beads -/
inductive BeadColor
| Red
| White
| Blue
| Green

/-- Represents a sequence of beads -/
def BeadSequence := List BeadColor

/-- The total number of beads -/
def totalBeads : Nat := 10

/-- The number of beads of each color -/
def beadCounts : List Nat := [4, 3, 2, 1]

/-- Checks if no two neighboring beads in a sequence have the same color -/
def noAdjacentSameColor (seq : BeadSequence) : Bool :=
  sorry

/-- Generates all possible bead sequences -/
def allSequences : List BeadSequence :=
  sorry

/-- Counts the number of sequences where no two neighboring beads have the same color -/
def validSequencesCount : Nat :=
  (allSequences.filter noAdjacentSameColor).length

/-- The probability of no two neighboring beads having the same color -/
noncomputable def probability : ℚ :=
  validSequencesCount / allSequences.length

theorem probability_is_0_05 : probability = 1/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_0_05_l933_93394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l933_93387

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_rate_increase : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 18)
  (h2 : regular_hours = 40)
  (h3 : overtime_rate_increase = 0.75)
  (h4 : total_hours = 48.12698412698413) :
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let overtime_hours := max (total_hours - regular_hours) 0
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  ∃ ε > 0, |regular_pay + overtime_pay - 976.00| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l933_93387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_relation_l933_93314

theorem cube_volume_relation (v₁ : ℝ) (s₁ s₂ : ℝ) (a₁ a₂ : ℝ) (v₂ : ℝ) : 
  v₁ = 8 →                   -- volume of first cube
  v₁ = s₁^3 →                -- relation between volume and side length
  a₁ = 6 * s₁^2 →            -- surface area of first cube
  a₂ = 3 * a₁ →              -- surface area of second cube is three times the first
  a₂ = 6 * s₂^2 →            -- relation between surface area and side length
  v₂ = s₂^3 →                -- volume of second cube
  v₂ = 24 * Real.sqrt 3 := by
  intro h1 h2 h3 h4 h5 h6
  -- Proof steps would go here
  sorry

#check cube_volume_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_relation_l933_93314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_l933_93300

theorem chinese_remainder_theorem
  (k : ℕ)
  (m : Fin k → ℕ)
  (a : Fin k → ℤ)
  (coprime : ∀ i j : Fin k, i ≠ j → Nat.Coprime (m i) (m j)) :
  let M := (Finset.univ : Finset (Fin k)).prod m
  ∃ x : ℤ, (∀ i : Fin k, x ≡ a i [ZMOD m i]) ∧
    (∀ x₁ x₂ : ℤ, (∀ i : Fin k, x₁ ≡ a i [ZMOD m i]) →
                  (∀ i : Fin k, x₂ ≡ a i [ZMOD m i]) →
                  x₁ ≡ x₂ [ZMOD M]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_l933_93300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_f_implies_a_range_l933_93397

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1)^x

theorem monotonic_decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (1 < a ∧ a < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_f_implies_a_range_l933_93397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kamal_average_marks_l933_93305

noncomputable def english_marks : ℝ := 76
noncomputable def english_total : ℝ := 120
noncomputable def english_weight : ℝ := 0.20

noncomputable def math_marks : ℝ := 60
noncomputable def math_total : ℝ := 110
noncomputable def math_weight : ℝ := 0.25

noncomputable def physics_marks : ℝ := 82
noncomputable def physics_total : ℝ := 100
noncomputable def physics_weight : ℝ := 0.15

noncomputable def chemistry_marks : ℝ := 67
noncomputable def chemistry_total : ℝ := 90
noncomputable def chemistry_weight : ℝ := 0.20

noncomputable def biology_marks : ℝ := 85
noncomputable def biology_total : ℝ := 100
noncomputable def biology_weight : ℝ := 0.15

noncomputable def history_marks : ℝ := 78
noncomputable def history_total : ℝ := 95
noncomputable def history_weight : ℝ := 0.05

noncomputable def weighted_average : ℝ :=
  (english_marks / english_total * english_weight +
   math_marks / math_total * math_weight +
   physics_marks / physics_total * physics_weight +
   chemistry_marks / chemistry_total * chemistry_weight +
   biology_marks / biology_total * biology_weight +
   history_marks / history_total * history_weight) * 100

theorem kamal_average_marks :
  ∃ ε > 0, |weighted_average - 70.345| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kamal_average_marks_l933_93305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l933_93345

/-- The equation of a hyperbola given its focus and eccentricity -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (c : ℝ), c = Real.sqrt 10 ∧ c^2 = a^2 + b^2) →
  (a / Real.sqrt (a^2 + b^2) = Real.sqrt 10 / 3) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 9 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l933_93345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_four_l933_93323

-- Define the set of five numbers
def five_numbers : Finset ℝ := sorry

-- Define the arithmetic mean function
noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ := (s.sum id) / s.card

-- State the theorem
theorem mean_of_remaining_four (h1 : five_numbers.card = 5) 
  (h2 : arithmetic_mean five_numbers = 92) 
  (h3 : ∃ x ∈ five_numbers, x = 105 ∧ ∀ y ∈ five_numbers, y ≤ x) :
  arithmetic_mean (five_numbers.filter (λ x => x ≠ 105)) = 88.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_four_l933_93323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l933_93386

/-- The ellipse E: x²/5 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

/-- The parabola G: y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

/-- The line l: y = k(x-2) -/
def line (k x y : ℝ) : Prop := y = k * (x - 2)

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Main theorem -/
theorem ellipse_parabola_intersection (k : ℝ) (hk : k ≠ 0) :
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ parabola x3 y3 ∧ parabola x4 y4 ∧
    line k x1 y1 ∧ line k x2 y2 ∧ line k x3 y3 ∧ line k x4 y4 ∧
    (2 / distance x1 y1 x2 y2 - (16 * Real.sqrt 5 / 5) / distance x3 y3 x4 y4 = 1 / (2 * Real.sqrt 5)) := by
  sorry

#check ellipse_parabola_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l933_93386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l933_93366

/-- The speed of a particle given its position function -/
noncomputable def particle_speed (pos : ℝ → ℝ × ℝ) : ℝ :=
  let x := (pos 1).1 - (pos 0).1
  let y := (pos 1).2 - (pos 0).2
  Real.sqrt (x^2 + y^2)

/-- Theorem: The speed of a particle with position (3t + 8, 5t - 15) is √34 -/
theorem particle_speed_is_sqrt34 :
  particle_speed (λ t => (3 * t + 8, 5 * t - 15)) = Real.sqrt 34 := by
  -- Expand the definition of particle_speed
  unfold particle_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt34_l933_93366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_bisectors_l933_93310

-- Define a right triangle with side lengths 6, 8, and 10
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_lengths : a = 6 ∧ b = 8 ∧ c = 10

-- Define the perimeter bisector
def perimeter_bisector (t : RightTriangle) : ℝ → ℝ → Prop :=
  λ _ _ => True

-- Define the angle bisector
def angle_bisector (t : RightTriangle) : ℝ → ℝ → Prop :=
  λ _ _ => True

-- Define the angle between two lines
noncomputable def angle_between_lines (l1 l2 : ℝ → ℝ → Prop) : ℝ :=
  0

-- Theorem statement
theorem angle_between_bisectors (t : RightTriangle) :
  let pb := perimeter_bisector t
  let ab := angle_bisector t
  let φ := angle_between_lines pb ab
  Real.tan φ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_bisectors_l933_93310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l933_93343

/-- The angle of the hour hand at 3:30 PM -/
def hour_hand_angle : ℝ := 105

/-- The angle of the minute hand at 3:30 PM -/
def minute_hand_angle : ℝ := 180

/-- The smaller angle between the hour and minute hands -/
noncomputable def smaller_angle (h m : ℝ) : ℝ :=
  min (abs (m - h)) (360 - abs (m - h))

theorem clock_angle_at_3_30 :
  smaller_angle hour_hand_angle minute_hand_angle = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l933_93343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minus_two_to_three_l933_93398

def distance_on_number_line (x y : ℝ) : ℝ := |y - x|

theorem distance_minus_two_to_three :
  distance_on_number_line (-2) 3 = 5 := by
  -- Unfold the definition of distance_on_number_line
  unfold distance_on_number_line
  -- Simplify the absolute value
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minus_two_to_three_l933_93398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_F_coordinates_l933_93370

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the intersection point F
noncomputable def F : ℝ × ℝ := 
  let m_AE := (E.2 - A.2) / (E.1 - A.1)
  let b_AE := A.2 - m_AE * A.1
  let m_CD := (D.2 - C.2) / (D.1 - C.1)
  let b_CD := C.2 - m_CD * C.1
  let x := (b_CD - b_AE) / (m_AE - m_CD)
  let y := m_AE * x + b_AE
  (x, y)

theorem sum_of_F_coordinates : F.1 + F.2 = 20 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_F_coordinates_l933_93370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l933_93390

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : t.a / t.b = 1 + Real.cos t.C)
  (h2 : Real.cos t.B = 2 * Real.sqrt 7 / 7)
  (h3 : 0 < t.C ∧ t.C < Real.pi/2)  -- C is an acute angle
  (h4 : (1/2) * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2)  -- Area condition
  : Real.sin t.C = Real.tan t.B ∧ t.c = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l933_93390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l933_93383

noncomputable def f (x : ℝ) := Real.tan (2 * x - Real.pi / 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi / 2 + 3 * Real.pi / 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l933_93383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_g_sin_sum_and_tan_sum_l933_93384

noncomputable section

/-- Given that a and b are two distinct real roots of 4x^2 - 4tx - 1 = 0 -/
def are_roots (a b t : ℝ) : Prop :=
  a ≠ b ∧ 4 * a^2 - 4 * t * a - 1 = 0 ∧ 4 * b^2 - 4 * t * b - 1 = 0

/-- The function f(x) = 1 / (x^2 - xt) -/
def f (t : ℝ) (x : ℝ) : ℝ := 1 / (x^2 - x * t)

/-- The domain of f(x) is [a, b] -/
def domain_f (a b t : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a b → f t x ≠ 0

/-- The function g(t) = max f(x) - min f(x) -/
def g (t : ℝ) : ℝ := 4 * t / (1 - t^2)

theorem roots_and_g (a b t : ℝ) (h1 : are_roots a b t) (h2 : domain_f a b t) :
  g t = 4 * t / (1 - t^2) := by
  sorry

theorem sin_sum_and_tan_sum (u₁ u₂ u₃ : ℝ)
  (h1 : u₁ ∈ Set.Ioo 0 (π/2))
  (h2 : u₂ ∈ Set.Ioo 0 (π/2))
  (h3 : u₃ ∈ Set.Ioo 0 (π/2))
  (h4 : Real.sin u₁ + Real.sin u₂ + Real.sin u₃ = 1) :
  Real.tan u₁^2 + Real.tan u₂^2 + Real.tan u₃^2 < 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_g_sin_sum_and_tan_sum_l933_93384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_determination_l933_93379

-- Define the conditions for triangle determination
inductive TriangleCondition
  | SSS : TriangleCondition
  | SAS : TriangleCondition
  | ASA : TriangleCondition
  | TwoSidesOppositeAngle : TriangleCondition
  deriving BEq, Repr

-- Define a function to check if a set of conditions can uniquely determine a triangle
def canUniquelyDetermineTriangle (conditions : List TriangleCondition) : Prop :=
  conditions.length ≥ 3 ∧ 
  conditions.contains TriangleCondition.SSS ∧
  conditions.contains TriangleCondition.SAS ∧
  conditions.contains TriangleCondition.ASA ∧
  ¬conditions.contains TriangleCondition.TwoSidesOppositeAngle

-- Theorem statement
theorem unique_triangle_determination :
  canUniquelyDetermineTriangle [TriangleCondition.SSS, TriangleCondition.SAS, TriangleCondition.ASA] ∧
  ¬canUniquelyDetermineTriangle [TriangleCondition.SSS, TriangleCondition.SAS, TriangleCondition.TwoSidesOppositeAngle] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_determination_l933_93379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_nine_y_plus_two_l933_93376

theorem power_nine_y_plus_two (y : ℝ) (h : (3 : ℝ)^(2*y) = 16) : (9 : ℝ)^(y+2) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_nine_y_plus_two_l933_93376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l933_93381

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.log (x + a)|

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) →
  a ∈ Set.Ici 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l933_93381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_arrangement_probability_l933_93331

/-- The number of green lamps -/
def green_lamps : ℕ := 4

/-- The number of orange lamps -/
def orange_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := green_lamps + orange_lamps

/-- The number of lamps turned on -/
def lamps_on : ℕ := 4

/-- The probability of the leftmost lamp being green and on, 
    and the rightmost lamp being orange and off -/
def target_probability : ℚ := 1 / 7

theorem lamp_arrangement_probability : 
  (Nat.choose (total_lamps - 2) (green_lamps - 1) * Nat.choose (total_lamps - 1) (lamps_on - 1)) / 
  (Nat.choose total_lamps green_lamps * Nat.choose total_lamps lamps_on : ℚ) = target_probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_arrangement_probability_l933_93331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_next_value_l933_93399

/-- A polynomial function satisfying specific conditions -/
def special_polynomial (n : ℕ) (f : ℝ → ℝ) : Prop :=
  (∃ p : Polynomial ℝ, ∀ x, f x = p.eval x) ∧ 
  (∀ k : ℕ, k ≤ n → f k = k / (k + 1))

/-- The value of f(n+1) for a special polynomial -/
theorem special_polynomial_next_value (n : ℕ) (f : ℝ → ℝ) 
  (hf : special_polynomial n f) : 
  f (n + 1) = if n % 2 = 0 then 1 else (n : ℝ) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_next_value_l933_93399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_a_b_perpendicular_vectors_l933_93378

noncomputable section

def A : ℝ × ℝ × ℝ := (-2, 0, 2)
def B : ℝ × ℝ × ℝ := (-1, 1, 2)
def C : ℝ × ℝ × ℝ := (-3, 0, 4)

def a : ℝ × ℝ × ℝ := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
def b : ℝ × ℝ × ℝ := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem cosine_angle_a_b :
  dot_product a b / (magnitude a * magnitude b) = -Real.sqrt 10 / 10 := by sorry

theorem perpendicular_vectors (k : ℝ) :
  dot_product (k • a + b) (k • a - 2 • b) = 0 ↔ k = -5/2 ∨ k = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_a_b_perpendicular_vectors_l933_93378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l933_93353

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a 90-meter train traveling at 124 km/h takes approximately 2.61 seconds to cross an electric pole -/
theorem train_crossing_time_approx :
  ∃ ε > 0, |train_crossing_time 90 124 - 2.61| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l933_93353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l933_93395

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.cos (x - Real.pi / 4)

theorem f_properties :
  let period : ℝ := Real.pi
  let axis_of_symmetry (k : ℤ) : ℝ := Real.pi / 3 + k * Real.pi / 2
  let range_lower : ℝ := -Real.sqrt 3 / 2
  let range_upper : ℝ := 1
  let interval : Set ℝ := Set.Icc (-Real.pi / 12) (Real.pi / 2)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ x : ℝ, ∃ k : ℤ, f (axis_of_symmetry k - x) = f (axis_of_symmetry k + x)) ∧
  (Set.Icc range_lower range_upper = {y : ℝ | ∃ x ∈ interval, f x = y}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l933_93395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rewrite_S_b_l933_93312

noncomputable def S (n : ℕ) : ℝ := sorry -- Sum of first n terms of the original arithmetic sequence

noncomputable def S_b (b : ℝ) (a c : ℝ) (n : ℕ) : ℝ :=
  b * n + (a + b + c) / 3 * (S n - n * S 1)

theorem rewrite_S_b (a c : ℝ) (n : ℕ) :
  S_b 4 a c n = 4 * n + 6 * (S n - n * S 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rewrite_S_b_l933_93312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_point_coordinates_l933_93389

-- Define the circle and parabola
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1
def parabolaE (x y : ℝ) : Prop := y = x^2 - 2

-- Define the tangent line
def tangentLine (k b : ℝ) : Prop := b^2 = k^2 + 1

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Define the slope of QR
def slopeQR (x₁ y₁ x₂ y₂ : ℝ) : Prop := (y₁ - y₂) / (x₁ - x₂) = -Real.sqrt 3

-- Main theorem
theorem tangent_line_and_point_coordinates :
  ∀ (k b x₀ y₀ x₁ y₁ x₂ y₂ : ℝ),
  tangentLine k b →
  parabolaE x₀ y₀ →
  circleO x₁ y₁ →
  circleO x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  slopeQR x₁ y₁ x₂ y₂ →
  ((b = -1 ∧ k = 0) ∧
   ((x₀ = -Real.sqrt 3 / 3 ∧ y₀ = -5/3) ∨ (x₀ = Real.sqrt 3 ∧ y₀ = 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_point_coordinates_l933_93389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l933_93333

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x - 2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < -1 ∨ x > 2

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x y, domain x → domain y → x < y → x < -1 → y < -1 → f x > f y :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l933_93333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l933_93373

/-- Two lines in space are different -/
structure DifferentLines (l m : Line) : Prop where

/-- Two planes are non-coincident -/
structure NonCoincidentPlanes (α β : Plane) : Prop where

/-- A line is perpendicular to a plane -/
def Perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def Parallel (l : Line) (β : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def PlanesPerp (α β : Plane) : Prop := sorry

/-- Line type -/
def Line : Type := sorry

/-- Plane type -/
def Plane : Type := sorry

theorem planes_perpendicular 
  (l m : Line) (α β : Plane)
  (h1 : DifferentLines l m)
  (h2 : NonCoincidentPlanes α β)
  (h3 : Perpendicular l α)
  (h4 : Parallel l β) :
  PlanesPerp α β := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l933_93373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_amount_payment_equalizes_expenses_l933_93342

/-- The amount Liam needs to pay Noah to equalize expenses -/
noncomputable def amount_to_pay (L M N : ℝ) : ℝ := (M + N - 2*L) / 3

/-- Theorem stating the correct amount Liam needs to pay Noah -/
theorem correct_payment_amount (L M N : ℝ) 
  (h1 : L < N) 
  (h2 : L > M) : 
  amount_to_pay L M N = (M + N - 2*L) / 3 := by
  sorry

/-- Theorem proving that the payment equalizes expenses -/
theorem payment_equalizes_expenses (L M N : ℝ) 
  (h1 : L < N) 
  (h2 : L > M) : 
  L + amount_to_pay L M N = N - amount_to_pay L M N ∧
  M + amount_to_pay L M N = N - amount_to_pay L M N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_payment_amount_payment_equalizes_expenses_l933_93342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_append_two_to_two_digit_number_solution_matches_answer_A_l933_93380

/-- 
Theorem: For any two-digit number with tens digit t and units digit u,
the number formed by appending 2 to its right is equal to 100t + 10u + 2.
-/
theorem append_two_to_two_digit_number (t u : ℕ) :
  (10 * t + u) * 10 + 2 = 100 * t + 10 * u + 2 := by
  ring

/-- 
Verifies that the solution matches the given answer choice A.
-/
theorem solution_matches_answer_A (t u : ℕ) :
  (10 * t + u) * 10 + 2 = 100 * t + 10 * u + 2 := by
  rw [append_two_to_two_digit_number]

#check append_two_to_two_digit_number
#check solution_matches_answer_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_append_two_to_two_digit_number_solution_matches_answer_A_l933_93380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_p_equals_two_l933_93340

/-- Parabola type representing y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Function to check if a point lies on a parabola -/
def on_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x ∧ point.x > 0

/-- Function to check if a point lies on a line -/
def on_line (point : Point) (line : Line) : Prop :=
  point.y = line.m * point.x + line.b

/-- Function to check if a point lies inside a parabola -/
def inside_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 < 2 * parabola.p * point.x ∧ point.x > 0

/-- Theorem stating that under given conditions, p must equal 2 -/
theorem parabola_p_equals_two (parabola : Parabola) (P A B C D : Point) (l : Line) :
  inside_parabola P parabola →
  P.x = 2 →
  P.y = 1 →
  l.m = 2 →
  ¬ on_line P l →
  on_parabola A parabola →
  on_parabola B parabola →
  on_parabola C parabola →
  on_parabola D parabola →
  on_line A l →
  on_line B l →
  (∃ lambda : ℝ, lambda > 0 ∧ lambda ≠ 1 ∧
    A.x - P.x = lambda * (P.x - C.x) ∧
    A.y - P.y = lambda * (P.y - C.y) ∧
    B.x - P.x = lambda * (P.x - D.x) ∧
    B.y - P.y = lambda * (P.y - D.y)) →
  parabola.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_p_equals_two_l933_93340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_and_AB_distance_l933_93313

-- Define the parametric equations for C1
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the relation between P and M
def P_relation (M P : ℝ × ℝ) : Prop :=
  P.1 = 2 * M.1 ∧ P.2 = 2 * M.2

-- Define C2 as the set of points P satisfying the relation with some M on C1
def C2 : Set (ℝ × ℝ) :=
  {P | ∃ α, P_relation (C1 α) P}

-- Define the polar coordinates for a point
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define points A and B
noncomputable def A : ℝ × ℝ := polar_to_cartesian (4 * Real.sin (Real.pi/3)) (Real.pi/3)
noncomputable def B : ℝ × ℝ := polar_to_cartesian (8 * Real.sin (Real.pi/3)) (Real.pi/3)

-- State the theorem
theorem C2_and_AB_distance :
  (∀ α, C2 (4 * Real.cos α, 4 + 4 * Real.sin α)) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_and_AB_distance_l933_93313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l933_93350

theorem sin_2alpha_value (α : ℝ) 
  (h1 : α > 0) 
  (h2 : α < π / 2) 
  (h3 : Real.cos (π / 4 - α) = 2 * Real.sqrt 2 * Real.cos (2 * α)) : 
  Real.sin (2 * α) = 15 / 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l933_93350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_T_determination_l933_93351

/-- An arithmetic sequence -/
noncomputable def ArithmeticSequence (a₁ d : ℝ) : ℕ → ℝ := fun n ↦ a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Sum of S₁ to Sn -/
noncomputable def T (a₁ d : ℝ) (n : ℕ) : ℝ := n * (n + 1) * (3 * a₁ + (n - 1) * d) / 6

/-- The theorem stating that Tn can be uniquely determined iff n = 6070 given S₂₀₂₃ -/
theorem unique_T_determination (a₁ d : ℝ) :
  ∃! n : ℕ, (∀ x y : ℝ, S a₁ d 2023 = x ∧ S y d 2023 = x → T a₁ d n = T y d n) ∧ n = 6070 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_T_determination_l933_93351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l933_93301

-- Define π as a real number greater than 3
noncomputable def π : ℝ := Real.pi
axiom π_gt_3 : π > 3

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.rpow (3/2) (1/3) * Real.rpow 12 (1/6) * Real.sqrt ((3 - x)^2)

-- State the theorem
theorem expression_simplification :
  expression π = 8 * (π - 3) :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l933_93301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_seven_l933_93319

/-- Calculates the length of a boat given its breadth, sinking depth, and the mass of a person causing it to sink. -/
noncomputable def boatLength (breadth : ℝ) (sinkDepth : ℝ) (personMass : ℝ) : ℝ :=
  let waterDensity : ℝ := 1000
  let gravity : ℝ := 9.81
  let displacedVolume : ℝ := personMass * gravity / (waterDensity * gravity)
  displacedVolume / (breadth * sinkDepth)

/-- Theorem stating that a boat with given parameters has a specific length. -/
theorem boat_length_is_seven :
  let breadth : ℝ := 2
  let sinkDepth : ℝ := 0.01
  let personMass : ℝ := 140
  boatLength breadth sinkDepth personMass = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_seven_l933_93319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_rate_l933_93355

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the volume of a rectangular tank -/
noncomputable def tankVolume (d : TankDimensions) : ℝ :=
  d.length * d.width * d.depth

/-- Calculates the filling rate of a tank -/
noncomputable def fillingRate (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

/-- Theorem: The filling rate of the given tank is 4 cubic feet per hour -/
theorem tank_filling_rate :
  let tank := TankDimensions.mk 6 4 3
  let volume := tankVolume tank
  let time := 18
  fillingRate volume time = 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_rate_l933_93355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l933_93330

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_property (t : Triangle) 
  (h1 : t.b * Real.sin (2 * t.C) = t.c * Real.sin t.B)
  (h2 : Real.sin (t.B - Real.pi / 3) = 3 / 5) :
  t.C = Real.pi / 3 ∧ 
  Real.sin t.A = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l933_93330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_implies_sin_pi_half_plus_two_alpha_l933_93372

theorem sin_pi_plus_alpha_implies_sin_pi_half_plus_two_alpha
  (α : ℝ)
  (h : Real.sin (Real.pi + α) = 2 * Real.sqrt 5 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_implies_sin_pi_half_plus_two_alpha_l933_93372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_root_l933_93336

theorem existence_of_root (l : ℝ) (h : l ≥ 2 * Real.sqrt 2) :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 1 / Real.sin x + 1 / Real.cos x = l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_root_l933_93336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_600_l933_93348

/-- Represents the length of a train and platform in meters -/
def train_length : ℝ := 600

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 72

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Theorem stating that under the given conditions, the train length is 600 meters -/
theorem train_length_is_600 : train_length = 600 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_600_l933_93348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l933_93320

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define a line through two points
def line_through (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.2 - A.2) * (B.1 - A.1) = (P.1 - A.1) * (B.2 - A.2)

-- Define parallel lines
def parallel_lines (A B C D : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (D.1 - C.1) = (B.1 - A.1) * (D.2 - C.2)

-- Main theorem
theorem hyperbola_property (A B M P Q : ℝ × ℝ) :
  on_hyperbola A →
  on_hyperbola B →
  on_hyperbola P →
  on_hyperbola Q →
  line_through focus A B →
  line_through A B M →
  parallel_lines A B P Q →
  ∃ (t : ℝ), M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧ t = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l933_93320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l933_93341

theorem real_part_of_complex_fraction :
  let z : ℂ := (1 - I) / (2 - I)
  z.re = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l933_93341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pcf_l933_93374

/-- An equilateral triangle in 2D space -/
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- A point is inside a triangle -/
def PointInTriangle (P A B C : ℝ × ℝ) : Prop := sorry

/-- A point is the foot of a perpendicular from another point to a line segment -/
def IsPerpendicularFoot (F P : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- Area of a triangle given its vertices -/
noncomputable def AreaOfTriangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given an equilateral triangle ABC with a point P inside it, and D, E, F being the feet of
    perpendiculars from P to the sides of ABC, if the area of ABC is 2028 square centimeters
    and the areas of PAD and PBE are each 192 square centimeters, then the area of PCF is
    630 square centimeters. -/
theorem area_of_pcf (A B C P D E F : ℝ × ℝ) : 
  EquilateralTriangle A B C →
  PointInTriangle P A B C →
  IsPerpendicularFoot D P (A, B) →
  IsPerpendicularFoot E P (B, C) →
  IsPerpendicularFoot F P (C, A) →
  AreaOfTriangle A B C = 2028 →
  AreaOfTriangle P A D = 192 →
  AreaOfTriangle P B E = 192 →
  AreaOfTriangle P C F = 630 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pcf_l933_93374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_odd_green_even_probability_l933_93352

-- Define the probability function for a ball landing in bin k
noncomputable def p (k : ℕ+) : ℝ := 1 / (k.val * (k.val + 1))

-- Define the sum of probabilities for odd-numbered bins
noncomputable def sum_odd_bins : ℝ := ∑' (k : ℕ+), if k.val % 2 = 1 then p k else 0

-- Define the sum of probabilities for even-numbered bins
noncomputable def sum_even_bins : ℝ := ∑' (k : ℕ+), if k.val % 2 = 0 then p k else 0

-- Theorem statement
theorem red_odd_green_even_probability :
  sum_odd_bins * sum_even_bins = 1/4 := by sorry

-- Additional lemmas that might be useful for the proof
lemma sum_all_bins_eq_one :
  (∑' (k : ℕ+), p k) = 1 := by sorry

lemma sum_odd_bins_eq_half :
  sum_odd_bins = 1/2 := by sorry

lemma sum_even_bins_eq_half :
  sum_even_bins = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_odd_green_even_probability_l933_93352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l933_93304

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions for a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  -- Law of sines
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  -- Ensure A, B, C are angles (between 0 and π)
  angle_range : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi

-- Theorem statement
theorem triangle_inequality (t : Triangle) : Real.sin (t.A / 2) ≤ t.a / (t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l933_93304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_plus_2_l933_93338

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then |Real.log x / Real.log (1/2)|
  else if x > 2 then -1/2 * x + 2
  else 0

theorem f_a_plus_2 (a : ℝ) (h : f a = 2) : f (a + 2) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_plus_2_l933_93338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l933_93385

-- Define the geometric sequence property
def is_geometric_sequence (a b c : ℝ) : Prop := b * b = a * c

-- Define the eccentricity of a conic section
noncomputable def eccentricity (m : ℝ) : ℝ := 
  if m > 0 then Real.sqrt (1 - 1 / m) else Real.sqrt (1 + 1 / (-m))

-- Theorem statement
theorem conic_section_eccentricity :
  ∀ m : ℝ, is_geometric_sequence 3 m 12 →
  (eccentricity m = Real.sqrt 30 / 6 ∨ eccentricity m = Real.sqrt 7) :=
by
  intro m h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l933_93385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_supplies_usage_l933_93315

theorem ship_supplies_usage (initial_supply : ℝ) (remaining_supply : ℝ) :
  initial_supply = 400 →
  remaining_supply = 96 →
  let after_one_day := initial_supply * (1 - 2/5)
  (after_one_day - remaining_supply) / after_one_day = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_supplies_usage_l933_93315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_375_l933_93377

/-- A geometric sequence of positive integers with first term 3 and third term 75 -/
def GeometricSequence := {a : ℕ → ℕ | a 1 = 3 ∧ a 3 = 75 ∧ ∀ n, a (n + 1) = a n * (a 2 / a 1)}

/-- The fourth term of the geometric sequence is 375 -/
theorem fourth_term_is_375 : ∀ a ∈ GeometricSequence, a 4 = 375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_375_l933_93377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l933_93311

theorem trigonometric_problem (x : ℝ) 
  (h : Real.sin (x/2) - 2 * Real.cos (x/2) = 0) : 
  Real.tan x = -4/3 ∧ 
  Real.cos (2*x) / (Real.sqrt 2 * Real.cos (π/4 + x) * Real.sin x) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l933_93311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_equals_one_sixteenth_l933_93307

theorem sin_product_equals_one_sixteenth :
  Real.sin (10 * Real.pi / 180) * Real.sin (30 * Real.pi / 180) * 
  Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_equals_one_sixteenth_l933_93307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_99_l933_93306

def sequenceItem : ℕ → ℕ
  | 1 => 3
  | 2 => 15
  | 3 => 35
  | 4 => 63
  | 5 => 99  -- The fifth term we want to prove
  | 6 => 143
  | _ => 0  -- Default case, not used in the problem

theorem fifth_term_is_99 : sequenceItem 5 = 99 := by
  rfl  -- reflexivity proves this equality immediately

#eval sequenceItem 5  -- This will output 99

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_99_l933_93306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_bounds_min_cuts_for_dodecagons_l933_93337

/-- Represents a polygon in 2D space -/
structure Polygon where
  -- We don't need to define the internal structure of a polygon for this statement

/-- Represents a straight line cut on a polygon -/
structure Cut where
  -- We don't need to define the internal structure of a cut for this statement

/-- The result of applying cuts to a polygon -/
structure CutResult where
  pieces : ℕ
  totalVertices : ℕ

/-- Function to apply cuts to a polygon -/
def applyCuts (p : Polygon) (cuts : List Cut) : CutResult :=
  sorry -- Implementation not needed for the statement

/-- Theorem stating the upper bounds on pieces and vertices after n cuts -/
theorem cut_bounds (p : Polygon) (cuts : List Cut) :
  let n := cuts.length
  let result := applyCuts p cuts
  result.pieces ≤ n + 1 ∧ result.totalVertices ≤ 4 * n + 4 := by
  sorry

/-- Theorem for the minimum number of cuts required for 100 dodecagons -/
theorem min_cuts_for_dodecagons (p : Polygon) (cuts : List Cut) 
  (h : (applyCuts p cuts).pieces ≥ 100) :
  cuts.length ≥ 1707 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_bounds_min_cuts_for_dodecagons_l933_93337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l933_93332

theorem unique_solution : ∃! x : ℤ, x ∈ ({-5, -4, 2, 4} : Set ℤ) ∧ x + 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l933_93332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_remainder_two_l933_93334

theorem smallest_number_with_remainder_two : ∃! n : ℕ, 
  n > 1 ∧ 
  (∀ m : ℕ, m ∈ ({3, 4, 5, 7} : Set ℕ) → n % m = 2) ∧
  (∀ k : ℕ, k > 1 → (∀ m : ℕ, m ∈ ({3, 4, 5, 7} : Set ℕ) → k % m = 2) → k ≥ n) :=
by
  use 422
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_remainder_two_l933_93334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_properties_l933_93309

/-- A random variable with a discrete distribution -/
structure DiscreteRV where
  P : ℕ → ℝ
  sum_to_one : ∑' n, P n = 1

/-- The expected value of a discrete random variable -/
noncomputable def expectation (X : DiscreteRV) : ℝ := ∑' n, n * X.P n

/-- The variance of a discrete random variable -/
noncomputable def variance (X : DiscreteRV) : ℝ := ∑' n, (n - expectation X)^2 * X.P n

/-- Our specific random variable X -/
noncomputable def X : DiscreteRV where
  P := fun
    | 1 => 1/5
    | 2 => 2/5
    | 3 => 2/5
    | _ => 0
  sum_to_one := by sorry

theorem X_properties :
  expectation X = 11/5 ∧
  X.P 2 = 2/5 ∧
  X.P 3 = 2/5 ∧
  (X.P 1 + X.P 2) = 3/5 ∧
  variance X = 14/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_properties_l933_93309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_problems_l933_93369

theorem tan_alpha_problems (α : ℝ) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧ ((6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_problems_l933_93369
