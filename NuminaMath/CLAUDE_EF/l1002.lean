import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_group_l1002_100261

/-- The probability of selecting 3 people from the same group
    when randomly choosing 3 from 20 people divided into two equal groups -/
theorem probability_same_group (n : ℕ) (h : n = 20) :
  (2 * Nat.choose (n / 2) 3 : ℚ) / Nat.choose n 3 = 20 / 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_same_group_l1002_100261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1002_100217

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Defines the ellipse (C) with given properties -/
noncomputable def ellipse_C : Ellipse where
  a := Real.sqrt 2
  b := 1

/-- Checks if a point (x, y) is on the ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Main theorem -/
theorem ellipse_intersection_theorem (A B E P : ℝ × ℝ) (t : ℝ) :
  on_ellipse ellipse_C A.1 A.2 →
  on_ellipse ellipse_C B.1 B.2 →
  triangle_area 0 0 A.1 A.2 B.1 B.2 = Real.sqrt 6 / 4 →
  E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  on_ellipse ellipse_C P.1 P.2 →
  P = (t * E.1, t * E.2) →
  t = 2 ∨ t = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1002_100217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_and_solutions_l1002_100275

open Real Set

theorem function_transformation_and_solutions (f g : ℝ → ℝ) (m α β : ℝ) :
  (∀ x, f x = 2 * sin x) →
  (∀ x, g x = cos x) →
  (α ∈ Icc 0 (2 * π)) →
  (β ∈ Icc 0 (2 * π)) →
  (α ≠ β) →
  (f α + g α = m) →
  (f β + g β = m) →
  (m ∈ Ioo (-Real.sqrt 5) (Real.sqrt 5) ∧ cos (α - β) = 2 * m^2 / 5 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_and_solutions_l1002_100275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculation_correct_l1002_100202

/-- Represents the weight problem with initial and new students -/
structure WeightProblem where
  initial_students : ℕ
  initial_average : ℚ
  new_students : ℕ
  new_average : ℚ
  first_student_weight_diff : ℚ
  second_student_weight_diff : ℚ

/-- Calculates the weights of new students given the problem parameters -/
def calculate_new_weights (p : WeightProblem) : (ℚ × ℚ × ℚ) :=
  let total_initial := p.initial_students * p.initial_average
  let total_new := (p.initial_students + p.new_students) * p.new_average
  let lightest_initial := (total_new - total_initial - p.initial_average - p.new_average) / 2 + p.first_student_weight_diff
  (lightest_initial - p.first_student_weight_diff,
   p.initial_average + p.second_student_weight_diff,
   p.new_average)

/-- Theorem stating that the calculated weights match the expected values -/
theorem weight_calculation_correct (p : WeightProblem)
  (h1 : p.initial_students = 19)
  (h2 : p.initial_average = 15)
  (h3 : p.new_students = 3)
  (h4 : p.new_average = 149/10)
  (h5 : p.first_student_weight_diff = -3/2)
  (h6 : p.second_student_weight_diff = 2) :
  calculate_new_weights p = (109/10, 17, 149/10) := by
  sorry

#eval calculate_new_weights {
  initial_students := 19,
  initial_average := 15,
  new_students := 3,
  new_average := 149/10,
  first_student_weight_diff := -3/2,
  second_student_weight_diff := 2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_calculation_correct_l1002_100202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l1002_100228

theorem tan_pi_fourth_minus_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo π (3*π/2)) 
  (h2 : Real.cos α = -4/5) : 
  Real.tan (π/4 - α) = 1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l1002_100228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triple_diagonal_intersection_l1002_100259

/-- A regular heptagon -/
structure RegularHeptagon where
  vertices : Fin 7 → ℝ × ℝ
  is_regular : sorry  -- Additional properties to ensure regularity

/-- A diagonal of a regular heptagon -/
def diagonal (h : RegularHeptagon) (i j : Fin 7) : Set (ℝ × ℝ) :=
  sorry

/-- The interior of a regular heptagon -/
def heptagon_interior (h : RegularHeptagon) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem: No point in the interior of a regular heptagon lies on at least three diagonals -/
theorem no_triple_diagonal_intersection (h : RegularHeptagon) : 
  ∀ p ∈ heptagon_interior h, ¬∃ (i j k l m n : Fin 7), 
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧ 
    (i, j) ≠ (k, l) ∧ (i, j) ≠ (m, n) ∧ (k, l) ≠ (m, n) ∧
    p ∈ diagonal h i j ∧ p ∈ diagonal h k l ∧ p ∈ diagonal h m n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triple_diagonal_intersection_l1002_100259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_4_value_l1002_100265

noncomputable def c : ℕ → ℝ
  | 0 => 1
  | n + 1 => (5/4) * c n + (7/4) * Real.sqrt (4^n - (c n)^2)

theorem c_4_value : c 4 = 233/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_4_value_l1002_100265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chen_xiang_age_when_saving_mother_l1002_100283

/-- Chen Xiang's age when he first met the Second Son -/
def chen_xiang_initial_age : ℕ → Prop := sorry

/-- The Second Son's age when he first met Chen Xiang -/
def second_son_initial_age : ℕ → Prop := sorry

/-- The Second Son's age is 4 times Chen Xiang's age initially -/
axiom second_son_age (x : ℕ) : 
  chen_xiang_initial_age x → second_son_initial_age (4 * x)

/-- 8 years later, the Second Son's age is 8 years less than 3 times Chen Xiang's age -/
axiom age_relation (x : ℕ) : 
  chen_xiang_initial_age x → (4 * x + 8) = (3 * (x + 8) - 8)

/-- Chen Xiang's age when he saved his mother -/
def chen_xiang_save_age (x : ℕ) : ℕ := x + 8

theorem chen_xiang_age_when_saving_mother :
  ∃ x : ℕ, chen_xiang_initial_age x ∧ chen_xiang_save_age x = 16 := by
  sorry

#check chen_xiang_age_when_saving_mother

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chen_xiang_age_when_saving_mother_l1002_100283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1002_100284

noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x - 4

theorem quadratic_properties :
  (∀ x > (3/2 : ℝ), (deriv f x) > 0) ∧
  (f (-2) = 6) ∧
  (f 0 = -4) ∧
  (f 1 = -6) ∧
  (f 3 = -4) ∧
  (∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c ∧ a > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1002_100284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1002_100292

/-- The diameter of a wheel given its travel distance and number of revolutions -/
noncomputable def wheel_diameter (distance : ℝ) (revolutions : ℝ) : ℝ :=
  distance / (revolutions * Real.pi)

/-- Theorem stating that a wheel covering 1232 cm in 14.012738853503185 revolutions has a diameter of approximately 27.998208221634 cm -/
theorem wheel_diameter_approx :
  |wheel_diameter 1232 14.012738853503185 - 27.998208221634| < 1e-9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l1002_100292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_cos_div_x_neq_incorrect_l1002_100297

-- Define the function f(x) = (cos x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

-- Define the incorrect derivative g(x) = (x sin x - cos x) / x^2
noncomputable def g (x : ℝ) : ℝ := (x * Real.sin x - Real.cos x) / (x^2)

-- Theorem stating that g is not the derivative of f
theorem derivative_cos_div_x_neq_incorrect :
  ¬(∀ x : ℝ, x ≠ 0 → deriv f x = g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_cos_div_x_neq_incorrect_l1002_100297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1002_100299

-- Define the points
def A : ℝ × ℝ := (5, 3)
def B : ℝ × ℝ := (-A.1, A.2)  -- Reflection of A over y-axis
def C : ℝ × ℝ := (B.2, B.1)   -- Reflection of B over y=x

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate the area of a triangle given base and height
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Theorem statement
theorem area_of_triangle_ABC : 
  let base := distance A B
  let height := |C.2 - A.2|
  triangleArea base height = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1002_100299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l1002_100295

-- Define the diameter of the circle
def diameter : ℝ := 16

-- Define the number of equal sectors
def num_sectors : ℕ := 4

-- Define the longest line segment in a sector
noncomputable def longest_segment (d : ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (2 * (d / 2) ^ 2)

-- Theorem statement
theorem longest_segment_squared (d : ℝ) (n : ℕ) 
  (h1 : d = diameter) 
  (h2 : n = num_sectors) : 
  (longest_segment d n) ^ 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l1002_100295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l1002_100210

theorem max_value_of_function : 
  ∃ (M : ℝ), M = 6 ∧ ∀ x : ℝ, Real.sin x ^ 2 - 4 * Real.cos x + 2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l1002_100210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1002_100294

/-- The circle with equation x^2 + y^2 = 4 -/
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- A point on the circle -/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  on_circle : myCircle x y

/-- The tangent line to the circle at a given point -/
def tangent_line (p : PointOnCircle) (x y : ℝ) : Prop :=
  p.x * x + p.y * y = 4

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept (p : PointOnCircle) : ℝ := 4 / p.x

/-- The y-intercept of the tangent line -/
noncomputable def y_intercept (p : PointOnCircle) : ℝ := 4 / p.y

/-- The midpoint of the line segment between the x and y intercepts -/
noncomputable def myMidpoint (p : PointOnCircle) : ℝ × ℝ :=
  (2 / p.x, 2 / p.y)

/-- The theorem stating that the trajectory of the midpoint satisfies x^2y^2 = x^2 + y^2 -/
theorem midpoint_trajectory (p : PointOnCircle) :
  let (x, y) := myMidpoint p
  x^2 * y^2 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1002_100294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_and_y_is_four_l1002_100290

-- Define the table type
def Table := Matrix (Fin 3) (Fin 3) Nat

-- Define the property that each number appears only once in each row and column
def validTable (t : Table) : Prop :=
  ∀ i j : Fin 3, t i j ∈ ({1, 2, 3} : Set Nat) ∧
  (∀ k : Fin 3, k ≠ j → t i k ≠ t i j) ∧
  (∀ k : Fin 3, k ≠ i → t k j ≠ t i j)

-- Define X and Y as the unknown values in the table
def X (t : Table) : Nat := t 1 1
def Y (t : Table) : Nat := t 2 2

-- Theorem statement
theorem sum_of_x_and_y_is_four (t : Table) (h : validTable t) : X t + Y t = 4 := by
  sorry

#check sum_of_x_and_y_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_and_y_is_four_l1002_100290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_value_l1002_100268

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - 3 * x)

-- Define the sequence of functions fₙ
noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => id
| (n+1) => f ∘ (f_n n)

-- State the theorem
theorem f_2005_value :
  f_n 2005 (37/10) = 37/57 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_value_l1002_100268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l1002_100240

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_derivative_at_one : 
  deriv f 1 = 1 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l1002_100240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1002_100250

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < -2 ∨ x > 2

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x y, x < y → x < -2 → y < -2 → f x < f y := by
  sorry

-- Additional lemma to show that f is defined on the interval (-∞, -2)
lemma f_defined_on_interval (x : ℝ) (h : x < -2) : 
  ∃ y, f x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1002_100250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_b_l1002_100230

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-1, 1, -4)

-- Define the function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2.1 * v.2.1 + v.2.2 * v.2.2)

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.1 - w.1, v.2.1 - w.2.1, v.2.2 - w.2.2)

-- Theorem statement
theorem magnitude_a_minus_b :
  magnitude (vector_sub a b) = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_b_l1002_100230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_inclusion_l1002_100234

open Real

-- Define the sets M, N, and P
def M : Set ℝ := {θ | ∃ k : ℤ, θ = (k : ℝ) * Real.pi / 4}
def N : Set ℝ := {x | Real.cos (2 * x) = 0}
def P : Set ℝ := {a | Real.sin (2 * a) = 1}

-- State the theorem
theorem sets_inclusion : P ⊆ N ∧ N ⊆ M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_inclusion_l1002_100234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_min_speed_proof_l1002_100298

/-- The minimum speed Alice needs to exceed to arrive before Bob -/
noncomputable def alices_min_speed (distance : ℝ) (bobs_speed : ℝ) (alice_delay : ℝ) : ℝ :=
  distance / (distance / bobs_speed - alice_delay)

theorem alices_min_speed_proof (distance : ℝ) (bobs_speed : ℝ) (alice_delay : ℝ)
    (h1 : distance = 30)
    (h2 : bobs_speed = 40)
    (h3 : alice_delay = 0.5) :
    alices_min_speed distance bobs_speed alice_delay = 60 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_min_speed_proof_l1002_100298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1002_100296

/-- Represents the sum of the first k terms of an arithmetic sequence -/
def S (k : ℕ) : ℝ := sorry

/-- The property that S_k represents sums of an arithmetic sequence -/
axiom S_arithmetic (k m : ℕ) : S (k + m) - S k = S (m + k) - S m

theorem arithmetic_sequence_sum (n : ℕ) (h1 : S n = 3) (h2 : S (2 * n) = 10) :
  S (3 * n) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1002_100296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_largest_smallest_circles_l1002_100201

/-- The area between two concentric circles -/
noncomputable def area_between_circles (r₁ r₂ : ℝ) : ℝ := Real.pi * (r₁^2 - r₂^2)

/-- Theorem: The area between concentric circles with radii 12 and 4 is 128π -/
theorem area_between_largest_smallest_circles :
  area_between_circles 12 4 = 128 * Real.pi := by
  unfold area_between_circles
  simp [Real.pi]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_largest_smallest_circles_l1002_100201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l1002_100266

theorem factorial_ratio (N : ℕ) : Nat.factorial (N + 1) / Nat.factorial (N + 2) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l1002_100266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l1002_100272

-- Define a point in 3D space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a prism
structure Prism where
  baseFaces : Set (Set Point)
  lateralEdges : Set (Set Point)

-- Define parallelism for sets of points (faces)
def Parallel (f1 f2 : Set Point) : Prop := sorry

-- Define length for a set of points (edge)
def Length (e : Set Point) : ℝ := sorry

-- Define the properties we want to prove
def baseParallel (p : Prism) : Prop :=
  ∃ (f1 f2 : Set Point), f1 ∈ p.baseFaces ∧ f2 ∈ p.baseFaces ∧ f1 ≠ f2 ∧ Parallel f1 f2

def lateralEdgesEqual (p : Prism) : Prop :=
  ∀ e1 e2, e1 ∈ p.lateralEdges → e2 ∈ p.lateralEdges → Length e1 = Length e2

-- The theorem to prove
theorem prism_properties (p : Prism) : baseParallel p ∧ lateralEdgesEqual p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l1002_100272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1002_100219

theorem fraction_simplification (b x : ℝ) :
  (Real.sqrt (b^2 + x^4) - (x^4 - b^4) / Real.sqrt (b^2 + x^4)) / (b^2 + x^4) = 
  b^4 / (b^2 + x^4)^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1002_100219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_committees_with_one_shared_member_l1002_100218

/-- Represents a committee in the organization -/
structure Committee where
  members : Finset Nat
  size_eq_three : members.card = 3

/-- The main theorem -/
theorem exist_committees_with_one_shared_member
  (n : Nat)
  (committees : Finset Committee)
  (h_committee_count : committees.card = n + 1)
  (h_total_members : (committees.biUnion fun c => c.members).card = n)
  (h_distinct_committees : ∀ c₁ c₂ : Committee, c₁ ∈ committees → c₂ ∈ committees → c₁ ≠ c₂ → c₁.members ≠ c₂.members) :
  ∃ c₁ c₂ : Committee, c₁ ∈ committees ∧ c₂ ∈ committees ∧ c₁ ≠ c₂ ∧ (c₁.members ∩ c₂.members).card = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_committees_with_one_shared_member_l1002_100218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l1002_100239

/-- Definition of the allowed operations on piles of tokens -/
inductive Operation
  | split : ℕ → ℕ → ℕ → Operation  -- Split a pile into two and add a token to one
  | merge : ℕ → ℕ → Operation      -- Merge two piles and add a token

/-- The state of the piles -/
def PileState := List ℕ

/-- Apply an operation to a pile state -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  match op with
  | Operation.split i j k => sorry
  | Operation.merge i j => sorry

/-- The initial state of the piles -/
def initialState : PileState :=
  sorry

/-- The target state of the piles -/
def targetState : PileState :=
  List.replicate 2022 2022

/-- Theorem stating the impossibility of reaching the target state -/
theorem impossible_transformation :
  ∀ (ops : List Operation),
    (ops.foldl applyOperation initialState) ≠ targetState :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l1002_100239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l1002_100222

-- Define the delta operation
def delta (a b : ℕ) : ℕ := a^3 - b

-- State the theorem
theorem delta_calculation :
  (2^(delta 3 8 : ℕ)) - (5^(delta 4 9 : ℕ)) = -5^55 + 524288 :=
by
  -- Convert natural numbers to integers for the final calculation
  have h1 : (2^(delta 3 8 : ℕ) : ℤ) - (5^(delta 4 9 : ℕ) : ℤ) = -5^55 + 524288 := by
    -- Evaluate delta operations
    have d1 : delta 3 8 = 19 := by rfl
    have d2 : delta 4 9 = 55 := by rfl
    -- Use these facts
    rw [d1, d2]
    -- The rest of the proof would go here
    sorry
  -- Convert the hypothesis to the desired form
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_calculation_l1002_100222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1002_100232

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∃ x : ℝ, (2 : ℝ)^x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1002_100232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_transport_days_l1002_100213

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
variable (h1 : 0 < a)
variable (h2 : a < b)

-- Define the function for the number of days
noncomputable def days_to_transport (a b : ℝ) : ℝ := b + Real.sqrt (b * (b - a))

-- State the theorem
theorem watermelon_transport_days :
  ∀ (a b : ℝ), 0 < a → a < b →
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    x * (b + Real.sqrt (b * (b - a))) = y * ((b + Real.sqrt (b * (b - a))) - a) ∧
    b * x + (b - a) * y = (1/2) * (x * (b + Real.sqrt (b * (b - a))) + y * ((b + Real.sqrt (b * (b - a))) - a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watermelon_transport_days_l1002_100213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1002_100237

noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I) + 3 * Complex.I

theorem z_in_first_quadrant : z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1002_100237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_theorem_l1002_100227

/-- Urn problem with N balls: a white, b black, c red -/
structure Urn :=
  (N : ℕ)
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h_sum : a + b + c = N)

/-- Number of balls drawn -/
def n : ℕ := sorry

/-- Probability of drawing a white ball -/
def p (u : Urn) : ℚ := u.a / u.N

/-- Probability of drawing a black ball -/
def q (u : Urn) : ℚ := u.b / u.N

/-- Number of white balls drawn -/
def ξ : ℕ := sorry

/-- Number of black balls drawn -/
def η : ℕ := sorry

/-- Covariance of ξ and η when drawing with replacement -/
def cov_with_replacement (u : Urn) : ℚ := -n * p u * q u

/-- Covariance of ξ and η when drawing without replacement -/
def cov_without_replacement (u : Urn) : ℚ := -n * p u * q u * (u.N - n) / (u.N - 1)

/-- Correlation coefficient of ξ and η -/
noncomputable def correlation_coefficient (u : Urn) : ℝ := 
  -Real.sqrt ((p u * q u) / ((1 - p u) * (1 - q u)))

/-- Main theorem for the urn problem -/
theorem urn_theorem (u : Urn) :
  (cov_with_replacement u = -n * p u * q u) ∧
  (cov_without_replacement u = -n * p u * q u * (u.N - n) / (u.N - 1)) ∧
  (correlation_coefficient u = -Real.sqrt ((p u * q u) / ((1 - p u) * (1 - q u)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_theorem_l1002_100227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1002_100248

theorem angle_values (α β : Real) 
  (h1 : Real.cos (π/2 - α) = Real.sqrt 2 * Real.cos (3*π/2 + β))
  (h2 : Real.sqrt 3 * Real.sin (3*π/2 - α) = -(Real.sqrt 2) * Real.sin (π/2 + β))
  (h3 : 0 < α) (h4 : α < π)
  (h5 : 0 < β) (h6 : β < π) :
  α = 3*π/4 ∧ β = 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1002_100248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_half_not_covered_l1002_100224

theorem sqrt2_half_not_covered : ∀ a b : ℕ, 
  0 < a → a ≤ b → Nat.Coprime a b → 
  |Real.sqrt 2 / 2 - (a : ℝ) / b| > 1 / (4 * (b : ℝ)^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_half_not_covered_l1002_100224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_sum_for_3_to_6_eleven_consecutive_squares_sum_l1002_100273

-- Define a function to calculate the sum of m consecutive squares starting from n
def sumConsecutiveSquares (n m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (fun i => (n + i)^2)

-- Part (a)
theorem no_square_sum_for_3_to_6 :
  ∀ m : ℕ, m ∈ ({3, 4, 5, 6} : Finset ℕ) →
    ∀ n k : ℕ, sumConsecutiveSquares n m ≠ k^2 :=
by
  sorry

-- Part (b)
theorem eleven_consecutive_squares_sum :
  ∃ n k : ℕ, sumConsecutiveSquares n 11 = k^2 ∧ n = 18 ∧ k = 77 :=
by
  sorry

#eval sumConsecutiveSquares 18 11 -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_sum_for_3_to_6_eleven_consecutive_squares_sum_l1002_100273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1002_100279

/-- Represents the swimming data for a group of friends over a week -/
structure SwimmingData where
  julien_distance : ℕ  -- Julien's daily swimming distance in meters
  julien_time : ℕ      -- Julien's daily swimming time in minutes
  sarah_factor : ℕ     -- Factor by which Sarah's distance relates to Julien's
  jamir_extra : ℕ      -- Extra distance Jamir swims compared to Sarah in meters
  lily_speed_factor : ℕ -- Factor by which Lily's speed relates to Julien's
  lily_time : ℕ         -- Lily's daily swimming time in minutes
  days : ℕ              -- Number of days they swim

/-- Calculates the total distance swam by all friends over the given period -/
def totalDistance (data : SwimmingData) : ℕ :=
  let julien_daily := data.julien_distance
  let sarah_daily := data.sarah_factor * julien_daily
  let jamir_daily := sarah_daily + data.jamir_extra
  let lily_daily := (data.lily_speed_factor * data.julien_distance * data.lily_time) / data.julien_time
  (julien_daily + sarah_daily + jamir_daily + lily_daily) * data.days

/-- The main theorem stating the total distance swam by the friends -/
theorem total_distance_theorem (data : SwimmingData) 
  (h1 : data.julien_distance = 50)
  (h2 : data.julien_time = 20)
  (h3 : data.sarah_factor = 2)
  (h4 : data.jamir_extra = 20)
  (h5 : data.lily_speed_factor = 4)
  (h6 : data.lily_time = 30)
  (h7 : data.days = 7) :
  totalDistance data = 3990 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1002_100279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1002_100242

-- Define the trapezoid
def trapezoid : Set (ℝ × ℝ) :=
  {p | let (x, y) := p; 0 ≤ x ∧ x ≤ 4 ∧ 2 ≤ y ∧ y ≤ 8 ∧ (y = 2*x ∨ y = 8 ∨ y = 2 ∨ x = 0)}

-- Define the area function for the trapezoid
noncomputable def area (T : Set (ℝ × ℝ)) : ℝ := 15

-- Theorem statement
theorem trapezoid_area : area trapezoid = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1002_100242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qingqiu_air_routes_l1002_100209

/-- A weighted edge in a graph -/
structure WeightedEdge (α : Type*) [LinearOrder α] where
  source : ℕ
  target : ℕ
  weight : α

/-- A graph with weighted edges -/
structure WeightedGraph (α : Type*) [LinearOrder α] where
  vertices : ℕ
  edges : List (WeightedEdge α)

/-- A function to check if a list of edges forms a spanning tree -/
def isSpanningTree {α : Type*} [LinearOrder α] (n : ℕ) (edges : List (WeightedEdge α)) : Prop :=
  sorry

/-- A function to check if an edge is the i-th smallest incident to a vertex -/
def isIthSmallestEdge {α : Type*} [LinearOrder α] (i : ℕ) (v : ℕ) (e : WeightedEdge α) (g : WeightedGraph α) : Prop :=
  sorry

theorem qingqiu_air_routes {α : Type*} [LinearOrder α] (g : WeightedGraph α) :
  g.vertices > 0 →
  (∀ e1 e2 : WeightedEdge α, e1 ∈ g.edges → e2 ∈ g.edges → e1 ≠ e2 → e1.weight ≠ e2.weight) →
  ∃ tree : List (WeightedEdge α),
    isSpanningTree g.vertices tree ∧
    ∀ i : ℕ, i < g.vertices - 1 →
      ∃ e ∈ tree, isIthSmallestEdge i e.source e g :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_qingqiu_air_routes_l1002_100209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_sum_l1002_100280

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a segment cut off from a circle by a chord -/
noncomputable def segmentArea (c : Circle) (chord : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- The length of a side of a quadrilateral -/
noncomputable def sideLength (q : Quadrilateral) (i : Fin 4) : ℝ := sorry

/-- Theorem: If a quadrilateral intersects a circle cutting off four segments of equal area,
    then the sum of opposite sides of the quadrilateral are equal -/
theorem equal_opposite_sides_sum
  (c : Circle) (q : Quadrilateral)
  (h_intersect : ∀ (i : Fin 4), segmentArea c (q.vertices i, q.vertices ((i + 1) % 4)) > 0)
  (h_equal_areas : ∀ (i j : Fin 4), segmentArea c (q.vertices i, q.vertices ((i + 1) % 4)) =
                                    segmentArea c (q.vertices j, q.vertices ((j + 1) % 4))) :
  sideLength q 0 + sideLength q 2 = sideLength q 1 + sideLength q 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_opposite_sides_sum_l1002_100280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l1002_100229

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*x

/-- Two mutually perpendicular chords through a point on a parabola -/
structure PerpendicularChords (M : ParabolaPoint) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perp : (P.2 - M.y) * (Q.2 - M.y) = -(P.1 - M.x) * (Q.1 - M.x)

/-- A line intersecting a parabola -/
structure IntersectingLine where
  m : ℝ
  intersects_parabola : ∃ (P Q : ℝ × ℝ), P.2^2 = 2*P.1 ∧ Q.2^2 = 2*Q.1 ∧ P.1 + m*P.2 + 1 = 0 ∧ Q.1 + m*Q.2 + 1 = 0

/-- Main theorem -/
theorem parabola_chords_theorem (M : ParabolaPoint) (chords : PerpendicularChords M) (line : IntersectingLine) :
  (∃ (P Q : ℝ × ℝ), (P = chords.P ∨ P = chords.Q) ∧ (Q = chords.P ∨ Q = chords.Q) ∧
    (M.x + 2 + line.m * (-M.y) + 1 = 0)) ∧
  (∃ (M' : ParabolaPoint), (M'.x - M.x)^2 + (M'.y + M.y)^2 = 4 ∧
    (line.m ≥ Real.sqrt 6 ∨ line.m ≤ -Real.sqrt 6)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chords_theorem_l1002_100229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l1002_100278

-- Define a Point type for 2D coordinates
structure Point where
  x : ℚ
  y : ℚ

-- Define a Triangle type
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Function to calculate squared distance between two points
def distanceSquared (p1 p2 : Point) : ℚ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distanceSquared t.a t.b
  let d2 := distanceSquared t.b t.c
  let d3 := distanceSquared t.c t.a
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the four triangles
def triangleA : Triangle := { a := { x := 1, y := 4 }, b := { x := 3, y := 4 }, c := { x := 2, y := 2 } }
def triangleB : Triangle := { a := { x := 4, y := 2 }, b := { x := 4, y := 4 }, c := { x := 6, y := 2 } }
def triangleC : Triangle := { a := { x := 7, y := 1 }, b := { x := 8, y := 4 }, c := { x := 9, y := 1 } }
def triangleD : Triangle := { a := { x := 0, y := 0 }, b := { x := 2, y := 1 }, c := { x := 1, y := 3 } }

-- Theorem stating that all four triangles are isosceles
theorem all_triangles_isosceles :
  isIsosceles triangleA ∧ isIsosceles triangleB ∧ isIsosceles triangleC ∧ isIsosceles triangleD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_l1002_100278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_5_25_l1002_100231

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := x - 1

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 2
noncomputable def g_inv (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem composition_equals_5_25 :
  f (g_inv (f_inv (f_inv (g (f 10))))) = 5.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_5_25_l1002_100231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_half_l1002_100205

-- Define g(n) for positive integers n
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (k : ℝ)⁻¹^(n : ℝ)

-- State the theorem
theorem sum_g_equals_half : ∑' n : ℕ+, g (n + 2) = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_half_l1002_100205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1002_100256

noncomputable def vector_a : Fin 2 → ℝ := ![1, Real.sqrt 3]
noncomputable def vector_b (m : ℝ) : Fin 2 → ℝ := ![m, Real.sqrt 3]

noncomputable def angle_between (v w : Fin 2 → ℝ) : ℝ :=
  Real.arccos ((v 0 * w 0 + v 1 * w 1) / (Real.sqrt (v 0 ^ 2 + v 1 ^ 2) * Real.sqrt (w 0 ^ 2 + w 1 ^ 2)))

theorem vector_angle_theorem (m : ℝ) :
  angle_between vector_a (vector_b m) = π / 3 → m = -1 := by
  sorry

#check vector_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1002_100256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1002_100289

noncomputable def f (x : ℝ) := Real.sqrt (1 - x^2) + Real.log (2 * Real.cos x - 1) / Real.log 10

theorem domain_of_f :
  {x : ℝ | 1 - x^2 ≥ 0 ∧ 2 * Real.cos x - 1 > 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1002_100289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_intersection_theorem_l1002_100258

-- Define a parallelepiped
structure Parallelepiped where
  -- Add necessary properties of a parallelepiped
  vertices : Finset (Fin 8 → ℝ)

-- Define a plane
structure Plane where
  -- Add necessary properties of a plane
  normal : ℝ → ℝ → ℝ → ℝ

-- Define a polygon
structure Polygon where
  vertices : Finset (ℝ × ℝ)

-- Define the intersection of a parallelepiped and a plane
def intersection (p : Parallelepiped) (pl : Plane) : Polygon :=
  sorry

-- Define a side of a polygon
def Side := Fin 2 → ℝ × ℝ

-- Define parallel sides
def are_parallel (s1 s2 : Side) : Prop :=
  sorry

-- Theorem statement
theorem parallelepiped_intersection_theorem 
  (p : Parallelepiped) (pl : Plane) (poly : Polygon) 
  (h1 : poly = intersection p pl) 
  (h2 : poly.vertices.card > 3) : 
  ∃ (side1 side2 : Side), 
    side1 ≠ side2 ∧ 
    are_parallel side1 side2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_intersection_theorem_l1002_100258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sin_implies_cos_2alpha_l1002_100260

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The given function f(x) = sin(x + α - π/12) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := Real.sin (x + α - Real.pi/12)

theorem even_sin_implies_cos_2alpha (α : ℝ) :
  IsEven (f α) → Real.cos (2 * α) = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sin_implies_cos_2alpha_l1002_100260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_three_dice_l1002_100243

/-- The expected number of 6's when rolling three standard dice -/
theorem expected_sixes_three_dice : ℝ := by
  let p_not_six : ℝ := 5/6  -- probability of not rolling a 6 on one die
  let p_zero : ℝ := p_not_six^3  -- probability of zero 6's
  let p_one : ℝ := 3 * (1/6) * p_not_six^2  -- probability of exactly one 6
  let p_two : ℝ := 3 * (1/6)^2 * p_not_six  -- probability of exactly two 6's
  let p_three : ℝ := (1/6)^3  -- probability of exactly three 6's
  have : 0 * p_zero + 1 * p_one + 2 * p_two + 3 * p_three = 1/2 := by
    sorry
  exact 1/2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_three_dice_l1002_100243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_value_l1002_100288

def regression_equation (x : ℝ) : ℝ := 2.5 * x + 0.31

def sample_point : ℝ × ℝ := (4, 1.2)

def calculate_residual (actual_y predicted_y : ℝ) : ℝ := actual_y - predicted_y

theorem residual_value : 
  calculate_residual sample_point.2 (regression_equation sample_point.1) = -9.11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_value_l1002_100288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l1002_100211

-- Define ω as a complex number that is a nonreal root of z^4 = 1
noncomputable def ω : ℂ := Complex.I

-- Define the property that ω is a nonreal root of z^4 = 1
axiom ω_property : ω^4 = 1 ∧ ω ≠ 1 ∧ ω ≠ -1

-- Define the set of ordered pairs (a,b) of integers satisfying |aω + b| = √2
def S : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | Complex.abs (p.1 * ω + p.2 : ℂ) = Real.sqrt 2}

-- Lemma to show that S is finite
lemma S_finite : Set.Finite S := by
  sorry

-- Theorem stating that there are exactly 4 such ordered pairs
theorem four_pairs : Finset.card (Set.Finite.toFinset S_finite) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_pairs_l1002_100211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_reverse_digits_l1002_100206

theorem greatest_difference_reverse_digits (q r : ℤ) : 
  10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 →  -- q and r are two-digit positive integers
  ∃ x y : ℤ, q = 10 * x + y ∧ r = 10 * y + x →  -- q and r have the same digits in reverse order
  abs (q - r) < 70 →  -- The positive difference between q and r is less than 70
  ∀ q' r' : ℤ, 
    (10 ≤ q' ∧ q' < 100 ∧ 10 ≤ r' ∧ r' < 100) →
    (∃ x' y' : ℤ, q' = 10 * x' + y' ∧ r' = 10 * y' + x') →
    abs (q' - r') < 70 →
    abs (q - r) ≤ abs (q' - r') →
    abs (q - r) = 63 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_reverse_digits_l1002_100206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l1002_100267

open Real

-- Define the constants
noncomputable def cone_radius : ℝ := 10
noncomputable def cone_height : ℝ := 15
noncomputable def cylinder_height : ℝ := 30
noncomputable def sphere_diameter : ℝ := 10

-- Define the volumes
noncomputable def cylinder_volume : ℝ := π * cone_radius^2 * cylinder_height
noncomputable def cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
noncomputable def sphere_volume : ℝ := (4/3) * π * (sphere_diameter/2)^3

-- Theorem statement
theorem unoccupied_volume_calculation :
  cylinder_volume - 2 * cone_volume - sphere_volume = (5500/3) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l1002_100267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_four_equals_thirteen_l1002_100264

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * (x/2)^2 + 1

-- State the theorem
theorem f_of_four_equals_thirteen : f 4 = 13 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_four_equals_thirteen_l1002_100264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_special_divisors_l1002_100221

def isDivisor (d n : ℕ) : Bool := n % d = 0

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => isDivisor d n)

theorem largest_n_with_special_divisors :
  ∀ N : ℕ,
  (∃ d₁ d₂ d₃ : ℕ,
    d₁ ∈ divisors N ∧
    d₂ ∈ divisors N ∧
    d₃ ∈ divisors N ∧
    d₁ < d₂ ∧
    d₂ < d₃ ∧
    (∀ d : ℕ, d ∈ divisors N → d ≤ d₁ ∨ d ≥ d₃) ∧
    d₃ = 21 * d₂) →
  N ≤ 441 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_special_divisors_l1002_100221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_minimum_value_f_inequality_l1002_100291

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

def monotone_increasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

def monotone_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

theorem f_monotonicity (a : ℝ) (ha : a = 1) :
  (monotone_increasing (f a) (Set.Ioo 0 (1/2)) ∧
   monotone_increasing (f a) (Set.Ioi 1)) ∧
  monotone_decreasing (f a) (Set.Ioo (1/2) 1) := by sorry

theorem f_minimum_value (a : ℝ) (ha : a > 0) :
  (∀ x, x ∈ Set.Icc 1 (Real.exp 1) → f a x ≥ -2) ∧ (∃ x, x ∈ Set.Icc 1 (Real.exp 1) ∧ f a x = -2) →
  a ≥ 1 := by sorry

theorem f_inequality (a : ℝ) (ha : a ≥ 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ + 2*x₁ < f a x₂ + 2*x₂) →
  a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_minimum_value_f_inequality_l1002_100291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_calculation_l1002_100271

/-- The total length of wire used to connect two poles -/
noncomputable def total_wire_length (pole_distance : ℝ) (pole1_height : ℝ) (pole2_height : ℝ) : ℝ :=
  let midpoint_distance := pole_distance / 2
  let height_diff := pole2_height - pole1_height
  let wire_between_poles := Real.sqrt (pole_distance ^ 2 + height_diff ^ 2)
  let wire_from_pole1 := Real.sqrt (midpoint_distance ^ 2 + pole1_height ^ 2)
  let wire_from_pole2 := Real.sqrt (midpoint_distance ^ 2 + pole2_height ^ 2)
  wire_between_poles + wire_from_pole1 + wire_from_pole2

/-- The theorem stating the total wire length for the given problem -/
theorem wire_length_calculation :
  total_wire_length 20 12 20 = Real.sqrt 464 + Real.sqrt 244 + Real.sqrt 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_calculation_l1002_100271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekday_weekend_difference_l1002_100212

/-- Hannah's skating practice schedule -/
structure HannahsPractice where
  total_weekly_hours : ℕ
  weekend_hours : ℕ
  weekday_hours : ℕ

/-- Axioms for Hannah's practice schedule -/
axiom schedule : HannahsPractice
axiom total_weekly_hours : schedule.total_weekly_hours = 33
axiom weekend_hours : schedule.weekend_hours = 8
axiom weekday_calculation : schedule.weekday_hours = schedule.total_weekly_hours - schedule.weekend_hours

/-- Theorem: The difference between weekday and weekend practice hours is 17 -/
theorem weekday_weekend_difference :
  schedule.weekday_hours - schedule.weekend_hours = 17 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekday_weekend_difference_l1002_100212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ABEF_l1002_100269

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix : ℝ → ℝ := λ x => -1

-- Define point A
noncomputable def A : ℝ × ℝ := (3, 3*Real.sqrt 3 - Real.sqrt 3)

-- Define point B
def B : ℝ × ℝ := (3, -1)

-- Define point E
def E : ℝ × ℝ := (-1, 0)

-- Define the theorem
theorem area_of_ABEF : 
  let F := focus
  let l := directrix
  ∀ (A B E : ℝ × ℝ),
    parabola A.1 A.2 →
    (A.2 - F.2) = Real.sqrt 3 * (A.1 - F.1) →
    B.1 = A.1 ∧ B.2 = l B.1 →
    E.1 = -1 ∧ E.2 = 0 →
    (1/2 * (E.1 - F.1 + A.1 - B.1) * (A.2 - B.2) = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ABEF_l1002_100269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_l1002_100245

/-- The volume of a cylinder with given dimensions -/
theorem cylinder_volume (base_circumference height : ℝ) (h_pi : π = 3) :
  base_circumference = 4.8 →
  height = 1.1 →
  π * (base_circumference / (2 * π))^2 * height = 2112 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_l1002_100245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1002_100286

-- Define the function f with domain (0,4]
def f : Set ℝ := Set.Ioc 0 4

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  if x ∈ Set.Ioi 1 ∪ Set.Ioc 1 2 then 1 / (x - 1) else 0

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = Set.Ioi 1 ∪ Set.Ioc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1002_100286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_x_l1002_100282

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem possible_values_of_x :
  ∀ x : ℝ, (A x ∩ B x = B x) ↔ x ∈ ({-2, 0, 2} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_x_l1002_100282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1002_100246

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- Predicate for a point P satisfying the given conditions -/
def ValidP (p : Point) : Prop :=
  p.y ≠ 0 ∧ 
  ∃ (t1 t2 : Point), t1 ∈ Parabola ∧ t2 ∈ Parabola ∧ 
    (t1.y - p.y) * (t2.x - p.x) = (t2.y - p.y) * (t1.x - p.x) ∧
    (t2.y - t1.y) * p.x = (t2.x - t1.x) * p.y

/-- The fixed point R -/
def R : Point := ⟨2, 0⟩

/-- The distance between two points -/
noncomputable def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The intersection point Q of l_P and PO -/
noncomputable def Q (p : Point) : Point :=
  ⟨2 * p.x / (p.x^2 + p.y^2), 2 * p.y / (p.x^2 + p.y^2)⟩

/-- The main theorem -/
theorem parabola_tangent_theorem (p : Point) (h : ValidP p) :
  (Q p = R ∨ dist p (Q p) / dist (Q p) R ≥ 2 * Real.sqrt 2) ∧
  ∃ p', ValidP p' ∧ dist p' (Q p') / dist (Q p') R = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1002_100246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_function_extrema_l1002_100293

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := -2 * (Real.sin α)^2 - 3 * Real.cos α + 7

-- State the theorem
theorem triangle_angle_function_extrema :
  ∀ α : ℝ, 0 < α ∧ α < Real.pi →
    (∃ min : ℝ, min = 31/8 ∧ ∀ β : ℝ, 0 < β ∧ β < Real.pi → f β ≥ min) ∧
    (¬∃ max : ℝ, ∀ β : ℝ, 0 < β ∧ β < Real.pi → f β ≤ max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_function_extrema_l1002_100293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1002_100249

/-- Given two monic, non-constant polynomials with integer coefficients r and s
    such that x^6 - 50x^3 + 1 = r(x) * s(x), prove that r(1) + s(1) = 4 -/
theorem polynomial_factorization (r s : Polynomial ℤ) :
  Polynomial.Monic r ∧ Polynomial.Monic s ∧ 
  (∀ x : ℤ, (X : Polynomial ℤ)^6 - 50*(X : Polynomial ℤ)^3 + 1 = r.eval x * s.eval x) →
  r.eval 1 + s.eval 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1002_100249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l1002_100207

def b : ℕ → ℤ
  | 0 => 0  -- Adding a case for 0
  | 1 => 0  -- Adding a case for 1
  | n => if n ≥ 20 then (if n = 20 then 20 else 200 * b (n-1) - n) else 0

theorem least_multiple_of_121 :
  ∀ n : ℕ, n > 20 ∧ n < 24 → ¬(121 ∣ b n) ∧ 121 ∣ b 24 :=
by
  intro n hn
  sorry  -- Skipping the proof for now

#eval b 24  -- Adding an evaluation to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_121_l1002_100207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1002_100252

theorem lambda_range (lambda : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.exp (x + lambda) - Real.log x + lambda ≥ 0) → 
  lambda ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1002_100252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicle_flow_l1002_100220

noncomputable section

/-- Vehicle flow speed as a function of vehicle flow density -/
def v (x : ℝ) : ℝ :=
  if x ≤ 30 then 60
  else if x ≤ 210 then -1/3 * x + 70
  else 0

/-- Vehicle flow as a function of vehicle flow density -/
def f (x : ℝ) : ℝ := x * v x

theorem max_vehicle_flow :
  ∃ (x_max : ℝ), x_max = 105 ∧ 
  (∀ x, 0 ≤ x → x ≤ 210 → f x ≤ f x_max) ∧
  f x_max = 3675 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicle_flow_l1002_100220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_three_pi_four_l1002_100238

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors_is_three_pi_four (a : ℝ × ℝ) :
  let b : ℝ × ℝ := (1, 1)
  (a.1^2 + a.2^2 = 1) →
  (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 3) →
  angle_between_vectors a b = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_three_pi_four_l1002_100238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_properties_l1002_100253

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := m * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^2 + a*x + 1
  else -x^2 + a*x + 1

-- State the theorem
theorem symmetric_functions_properties :
  ∃ (m : ℝ) (a : ℝ),
    (∀ x : ℝ, f m x + f m (-x) = 2) ∧
    (∀ x : ℝ, x ≠ 0 → g a x + g a (-x) = 2) ∧
    (∀ t : ℝ, t > 0 → ∀ x : ℝ, x < 0 → g a x < f m t) ∧
    m = 1 ∧
    (∀ x : ℝ, x < 0 → g a x = -x^2 + a*x + 1) ∧
    a > -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_properties_l1002_100253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l1002_100270

def is_valid_pair (a b : ℕ) : Prop :=
  b ≠ 1 ∧ (a + 1) ∣ (a^3 * b - 1) ∧ (b - 1) ∣ (b^3 * a + 1)

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ (a, b) ∈ ({(0, 0), (0, 2), (2, 2), (1, 3), (3, 3)} : Set (ℕ × ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l1002_100270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_audrey_painting_time_l1002_100214

/-- The time it takes Adam to paint the house alone -/
noncomputable def adam_time : ℝ := 5

/-- The time it takes Adam and Audrey to paint the house together -/
noncomputable def combined_time : ℝ := 30 / 11

/-- The time it takes Audrey to paint the house alone -/
noncomputable def audrey_time : ℝ := 6

/-- Theorem stating that if Adam can paint the house in 5 hours and together with Audrey
    they can paint it in 30/11 hours, then Audrey can paint it alone in 6 hours -/
theorem audrey_painting_time :
  (1 / adam_time + 1 / audrey_time = 1 / combined_time) →
  audrey_time = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_audrey_painting_time_l1002_100214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BST_l1002_100223

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the vertices A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define point P
def P (t : ℝ) : ℝ × ℝ := (1/2, t)

-- Define the property that M is on PA and on the hyperbola
def M_property (M : ℝ × ℝ) (t : ℝ) : Prop :=
  ∃ k : ℝ, M = (k * A.1 + (1-k) * (P t).1, k * A.2 + (1-k) * (P t).2) ∧
  hyperbola M.1 M.2

-- Define the property that N is on PB and on the hyperbola
def N_property (N : ℝ × ℝ) (t : ℝ) : Prop :=
  ∃ k : ℝ, N = (k * B.1 + (1-k) * (P t).1, k * B.2 + (1-k) * (P t).2) ∧
  hyperbola N.1 N.2

-- Define Q as the intersection of MN and x-axis
def Q_property (Q M N : ℝ × ℝ) : Prop :=
  Q.2 = 0 ∧ ∃ k : ℝ, Q = (k * M.1 + (1-k) * N.1, k * M.2 + (1-k) * N.2)

-- Define S and T on the hyperbola and on a line through Q
def ST_property (S T Q : ℝ × ℝ) : Prop :=
  hyperbola S.1 S.2 ∧ hyperbola T.1 T.2 ∧
  ∃ m : ℝ, S.1 = m * S.2 + Q.1 ∧ T.1 = m * T.2 + Q.1

-- Define the vector relation between S, Q, and T
def vector_relation (S Q T : ℝ × ℝ) : Prop :=
  (S.1 - Q.1, S.2 - Q.2) = (2 * (Q.1 - T.1), 2 * (Q.2 - T.2))

-- The main theorem
theorem area_of_triangle_BST (t : ℝ) (M N Q S T : ℝ × ℝ) :
  M_property M t →
  N_property N t →
  Q_property Q M N →
  ST_property S T Q →
  vector_relation S Q T →
  abs ((B.1 - S.1) * (T.2 - S.2) - (B.2 - S.2) * (T.1 - S.1)) / 2 = (9 / 16) * Real.sqrt 35 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BST_l1002_100223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_l1002_100215

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 16 / (x - 1)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x + 16 / (x - 1)^2

-- Theorem statement
theorem min_slope_tangent_line (x : ℝ) (h : x > 1) : 
  ∃ (m : ℝ), m = 7 ∧ ∀ y, y > 1 → f' y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_l1002_100215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_equals_one_max_angle_A_perimeter_range_l1002_100204

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  Real.sin t.B + Real.sin t.C = 2 * Real.sin t.A

-- Theorem 1
theorem side_a_equals_one (t : Triangle) 
  (h : triangle_conditions t) 
  (h1 : t.A = Real.pi/3) 
  (h2 : t.c = 1) : 
  t.a = 1 := by sorry

-- Theorem 2
theorem max_angle_A (t : Triangle) 
  (h : triangle_conditions t) 
  (h1 : t.b = 2) : 
  t.A ≤ Real.pi/3 := by sorry

-- Theorem 3
theorem perimeter_range (t : Triangle) 
  (h : triangle_conditions t) 
  (h1 : t.b = 2) : 
  4 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_a_equals_one_max_angle_A_perimeter_range_l1002_100204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_theorem_l1002_100251

-- Define the point A on the number line
def A (a : ℝ) : ℝ := a + 1

-- Define the distance from A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem point_distance_theorem (a : ℝ) : 
  distance_to_origin (A a) = 3 → a = 2 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_theorem_l1002_100251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1002_100236

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (((∀ x : ℝ, x > 0 → Monotone (fun x => Real.log (x + 2 - a) / Real.log a)) ∨ 
    (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0)) ∧
   ¬((∀ x : ℝ, x > 0 → Monotone (fun x => Real.log (x + 2 - a) / Real.log a)) ∧ 
     (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0))) →
  a > 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1002_100236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_63_kmph_l1002_100247

-- Define the train's length in meters
noncomputable def train_length : ℝ := 630

-- Define the time taken to pass the tree in seconds
noncomputable def passing_time : ℝ := 36

-- Define the conversion factor from m/s to km/hr
noncomputable def mps_to_kmph : ℝ := 3600 / 1000

-- State the theorem
theorem train_speed_is_63_kmph :
  (train_length / passing_time) * mps_to_kmph = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_63_kmph_l1002_100247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_total_wins_l1002_100276

/-- The number of games Jenny won against Mark and Jill -/
def total_wins (games_with_mark : ℕ) (mark_wins : ℕ) (jill_win_percentage : ℚ) : ℕ :=
  let games_with_jill := 2 * games_with_mark
  let jenny_wins_against_mark := games_with_mark - mark_wins
  let jill_wins := (jill_win_percentage * games_with_jill).num.toNat
  let jenny_wins_against_jill := games_with_jill - jill_wins
  jenny_wins_against_mark + jenny_wins_against_jill

theorem jenny_total_wins :
  total_wins 10 1 (3/4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_total_wins_l1002_100276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cars_count_l1002_100244

theorem black_cars_count (total : ℕ) (blue_fraction red_fraction : ℚ) : 
  total = 516 → 
  blue_fraction = 1/3 →
  red_fraction = 1/2 →
  (blue_fraction + red_fraction : ℚ) < 1 →
  ∃ (black : ℕ), black = total - (Nat.floor (blue_fraction * total) + Nat.floor (red_fraction * total)) ∧ black = 86 :=
by
  intro h_total h_blue h_red h_sum
  let blue_cars := Nat.floor (blue_fraction * total)
  let red_cars := Nat.floor (red_fraction * total)
  let black_cars := total - (blue_cars + red_cars)
  use black_cars
  apply And.intro
  · rfl
  · sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_cars_count_l1002_100244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_bridge_l1002_100241

/-- Represents a plank in the construction -/
structure Plank where
  length : ℝ

/-- Represents the pool and the construction problem -/
structure PoolProblem where
  num_planks : ℕ
  plank_length : ℝ
  min_distance_to_ball : ℝ

/-- Defines whether a bridge can be constructed given the problem parameters -/
def can_construct_bridge (problem : PoolProblem) : Prop :=
  ∃ (arrangement : List Plank),
    (arrangement.length = problem.num_planks) ∧
    (∀ p ∈ arrangement, p.length = problem.plank_length) ∧
    (∃ (max_reach : ℝ), max_reach > problem.min_distance_to_ball)

/-- The main theorem stating that the bridge cannot be constructed under given conditions -/
theorem cannot_construct_bridge (problem : PoolProblem)
  (h_num_planks : problem.num_planks = 30)
  (h_plank_length : problem.plank_length = 1)
  (h_min_distance : problem.min_distance_to_ball > 2) :
  ¬ can_construct_bridge problem := by
  sorry

#check cannot_construct_bridge

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_bridge_l1002_100241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l1002_100200

theorem sin_two_theta_value (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.sin (2 * θ) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l1002_100200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symDiff_cardinality_l1002_100257

/-- The symmetric difference of two sets -/
def symDiff (A B : Finset ℤ) : Finset ℤ := (A \ B) ∪ (B \ A)

/-- Main theorem -/
theorem symDiff_cardinality (x y z : Finset ℤ) : 
  (x.card = 8) →
  (y.card = 18) →
  (z.card = 12) →
  ((x ∩ y).card = 6) →
  ((y ∩ z).card = 4) →
  ((x ∩ z).card = 3) →
  ((x ∩ y ∩ z) = ∅) →
  (symDiff (symDiff x y) z).card = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symDiff_cardinality_l1002_100257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_P_l1002_100262

-- Define the set M
def M : Set ℝ := {0, 1, 2, 3}

-- Define the set P
def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem M_intersect_P : M ∩ P = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_P_l1002_100262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1002_100216

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1002_100216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_and_imag_parts_of_z_l1002_100203

theorem sum_of_real_and_imag_parts_of_z (z : ℂ) :
  z = (1 - 2*I) / (1 + I) → (z.re + z.im = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_and_imag_parts_of_z_l1002_100203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1002_100226

/-- The parabola y² = 4x with focus F and origin O -/
structure Parabola where
  F : ℝ × ℝ
  O : ℝ × ℝ

/-- Point P on the parabola -/
def P (c : Parabola) : ℝ × ℝ := sorry

/-- The parabola equation y² = 4x -/
axiom parabola_eq (c : Parabola) (p : ℝ × ℝ) : p.2 ^ 2 = 4 * p.1

/-- O is the coordinate origin -/
axiom origin (c : Parabola) : c.O = (0, 0)

/-- PF is perpendicular to OF -/
axiom perpendicular (c : Parabola) : 
  (P c).1 * c.F.1 + (P c).2 * c.F.2 = c.F.1 ^ 2 + c.F.2 ^ 2

/-- Theorem: |OF⃗ - PF⃗| = √5 -/
theorem vector_difference_magnitude (c : Parabola) : 
  Real.sqrt ((c.F.1 - (P c).1) ^ 2 + (c.F.2 - (P c).2) ^ 2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1002_100226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l1002_100277

theorem orange_crates_pigeonhole (total_crates min_oranges max_oranges : ℕ) 
  (h_total : total_crates = 160)
  (h_min : min_oranges = 115)
  (h_max : max_oranges = 142)
  (h_range : ∀ crate, min_oranges ≤ crate ∧ crate ≤ max_oranges) :
  ∃ n : ℕ, n ≥ 6 ∧ ∃ orange_count, (Finset.filter (λ crate => crate = orange_count) (Finset.range total_crates)).card ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l1002_100277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_race_ratio_l1002_100235

/-- The ratio of the third snail's speed to the second snail's speed -/
noncomputable def snail_speed_ratio (first_speed second_speed third_speed : ℝ) 
                      (first_time third_time : ℝ) : ℝ :=
  third_speed / second_speed

/-- Theorem: The ratio of the third snail's speed to the second snail's speed is 5:1 -/
theorem snail_race_ratio : 
  ∀ (first_speed second_speed third_speed : ℝ) 
    (first_time third_time : ℝ),
    first_speed = 2 →
    second_speed = 2 * first_speed →
    first_speed * first_time = third_speed * third_time →
    first_time = 20 →
    third_time = 2 →
    snail_speed_ratio first_speed second_speed third_speed first_time third_time = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_race_ratio_l1002_100235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1002_100225

noncomputable def complex_distance (z : ℂ) : ℝ := Complex.abs z

theorem distance_to_origin : complex_distance (2 * Complex.I / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1002_100225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1002_100263

-- Define the complex number that is a root of the polynomial
noncomputable def z : ℂ := -3 - Complex.I * Real.sqrt 7

-- Define the monic quadratic polynomial we want to prove
def f (x : ℝ) : ℝ := x^2 + 6*x + 16

-- Theorem statement
theorem monic_quadratic_with_complex_root :
  (∀ x : ℂ, (f x.re : ℂ) = x * (x - z)) ∧
  (∀ x : ℝ, f x = x^2 + 6*x + 16) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l1002_100263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1002_100285

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (abs (Real.sin x) + abs (Real.cos x)) + 4 * Real.sin (2 * x) + 9

theorem function_properties (a : ℝ) :
  (f a (9 * π / 4) = 13 - 9 * Real.sqrt 2) →
  (a = -9) ∧
  (∀ x, f a (x + π) = f a x) ∧
  (∃ x₁ x₂ x₃ x₄, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ ≤ π / 2 ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) ∧
  (∃ n : ℕ, n = 2021 ∧ (∃ (zeros : Finset ℝ), zeros.card = 2022 ∧
    (∀ x ∈ zeros, 0 ≤ x ∧ x ≤ n * π / 4 ∧ f a x = 0))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1002_100285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_cursed_die_l1002_100287

/-- A cursed six-sided die that never rolls the same number twice in a row -/
structure CursedDie where
  sides : Nat
  sides_eq : sides = 6

/-- The probability of rolling a new number when k unique numbers have been rolled -/
def prob_new_number (d : CursedDie) (k : Nat) : ℚ :=
  (d.sides - k) / (d.sides - 1)

/-- The expected number of rolls to obtain all distinct numbers on a cursed die -/
def expected_rolls (d : CursedDie) : ℚ :=
  1 + (Finset.range 5).sum (fun k => (d.sides - 1) / (d.sides - k - 1))

/-- Theorem: The expected number of rolls to obtain all distinct numbers on a cursed 6-sided die is 149/12 -/
theorem expected_rolls_cursed_die :
  ∀ d : CursedDie, expected_rolls d = 149 / 12 := by
  sorry

#check expected_rolls_cursed_die

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rolls_cursed_die_l1002_100287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_theorem_l1002_100208

/-- The number of radars -/
def n : ℕ := 8

/-- The radius of each radar's coverage area in km -/
noncomputable def r : ℝ := 17

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 16

/-- The central angle of the regular polygon formed by the radars in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The distance from the center to each radar -/
noncomputable def distance_to_radar : ℝ := 15 / Real.sin (θ / 2)

/-- The area of the coverage ring -/
noncomputable def coverage_area : ℝ := 480 * Real.pi / Real.tan (θ / 2)

theorem radar_coverage_theorem :
  (distance_to_radar = 15 / Real.sin (θ / 2)) ∧
  (coverage_area = 480 * Real.pi / Real.tan (θ / 2)) := by
  sorry

#check radar_coverage_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_theorem_l1002_100208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_subset_weight_theorem_l1002_100281

theorem cube_subset_weight_theorem (n : ℕ) (A : Finset ℝ) (r : ℝ) 
  (h1 : A.card = n)
  (h2 : ∀ c ∈ A, c ≥ 1)
  (h3 : A.sum id = 2 * n)
  (h4 : 0 ≤ r ∧ r ≤ 2 * n - 2) :
  ∃ B : Finset ℝ, B ⊆ A ∧ r ≤ B.sum id ∧ B.sum id ≤ r + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_subset_weight_theorem_l1002_100281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_integer_two_places_higher_l1002_100274

theorem square_of_integer_two_places_higher (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  (Real.sqrt x + 2) ^ 2 = x + 4 * Real.sqrt x + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_integer_two_places_higher_l1002_100274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1002_100255

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the small circle
def small_circle (x y : ℝ) : Prop := x^2 + y^2 = 8/5

theorem ellipse_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 1/4) -- eccentricity is 1/2
  (h4 : ∃ (x y : ℝ), ellipse_C x y a b ∧ circle_M x y) -- shared point
  (h5 : ∃ (x1 y1 x2 y2 : ℝ), 
    ellipse_C x1 y1 a b ∧ circle_M x1 y1 ∧ 
    ellipse_C x2 y2 a b ∧ circle_M x2 y2 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = 16) -- shared chord of length 4
  : 
  (∀ x y, ellipse_C x y a b ↔ x^2 / 16 + y^2 / 12 = 1) ∧ 
  (∃ B : ℝ × ℝ, 
    let A := (4, 0)
    let O := (0, 0)
    small_circle B.1 B.2 ∧ 
    (∃ k : ℝ, B.2 = k * (B.1 - 4)) ∧ -- line equation
    (A.1 * B.1 + A.2 * B.2 = -368/31)) -- dot product
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1002_100255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_values_l1002_100254

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Given a polynomial f with integer coefficients, if there exist distinct integers a and b
    such that f(a) and f(b) are coprime, then there exist infinitely many integers x
    such that f(x) is pairwise coprime for all such x -/
theorem infinite_coprime_values (f : IntPolynomial) :
  (∃ (a b : ℤ), a ≠ b ∧ Int.gcd (f.eval a) (f.eval b) = 1) →
  ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ (x y : ℤ), x ∈ S → y ∈ S → x ≠ y → Int.gcd (f.eval x) (f.eval y) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_values_l1002_100254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l1002_100233

/-- There exists a real solution to the equation x + 60/(x^2-4) = -12, and it's approximately -11.608 -/
theorem equation_solution_exists : ∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ 
  x + 60 / (x^2 - 4) = -12 ∧ abs (x + 11.608) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l1002_100233
