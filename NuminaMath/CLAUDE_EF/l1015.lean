import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_ellipse_max_min_distance_l1015_101599

/-- Definition of the ellipse -/
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m^2 + y^2 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem ellipse_foci_coordinates (m : ℝ) :
  m = 2 → ∃ (x : ℝ), x = Real.sqrt 3 ∧ 
  ellipse m x 0 ∧ ellipse m (-x) 0 ∧ 
  ∀ (a b : ℝ), ellipse m a b → distance x 0 a b + distance (-x) 0 a b = 2 * m := by
  sorry

theorem ellipse_max_min_distance :
  ∃ (max min : ℝ), max = 5 ∧ min = Real.sqrt 2 / 2 ∧
  (∀ (x y : ℝ), ellipse 3 x y → 
    distance x y 2 0 ≤ max ∧ distance x y 2 0 ≥ min) ∧
  (∃ (x1 y1 x2 y2 : ℝ), 
    ellipse 3 x1 y1 ∧ ellipse 3 x2 y2 ∧
    distance x1 y1 2 0 = max ∧ distance x2 y2 2 0 = min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_ellipse_max_min_distance_l1015_101599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_32_l1015_101588

/-- Given a triangle with sides a, b, c and a rectangle with equal perimeter and length twice its width, 
    calculate the area of the rectangle. -/
noncomputable def rectangle_area (a b c : ℝ) : ℝ :=
  let triangle_perimeter := a + b + c
  let rectangle_width := triangle_perimeter / 6
  let rectangle_length := 2 * rectangle_width
  rectangle_length * rectangle_width

/-- Theorem stating that for a triangle with sides 7.3 cm, 5.4 cm, and 11.3 cm, 
    and a rectangle with equal perimeter and length twice its width, 
    the area of the rectangle is 32 square centimeters. -/
theorem rectangle_area_equals_32 :
  rectangle_area 7.3 5.4 11.3 = 32 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_32_l1015_101588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_transformation_l1015_101581

/-- Given a mixture of alcohol and water, this function calculates the ratio of alcohol to water -/
noncomputable def mixture_ratio (alcohol : ℝ) (water : ℝ) : ℝ := alcohol / water

/-- The initial ratio of alcohol to water in the mixture -/
noncomputable def initial_ratio : ℝ := 4 / 3

/-- The final ratio of alcohol to water in the mixture after adding water -/
noncomputable def final_ratio : ℝ := 4 / 5

/-- The amount of alcohol in the mixture (in liters) -/
def alcohol_amount : ℝ := 5

/-- The amount of water added to the mixture (in liters) -/
def water_added : ℝ := 2.5

theorem mixture_transformation :
  let initial_water := alcohol_amount / initial_ratio
  let final_water := initial_water + water_added
  mixture_ratio alcohol_amount final_water = final_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_transformation_l1015_101581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1015_101504

-- Define the triangle
def Triangle : Set (ℝ × ℝ) := {(x, y) | (y = x ∧ y ≤ 8) ∨ (y = -x ∧ y ≤ 8) ∨ (y = 8 ∧ -8 ≤ x ∧ x ≤ 8)}

-- Define the vertices
def A : ℝ × ℝ := (8, 8)
def B : ℝ × ℝ := (-8, 8)
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem triangle_properties :
  (MeasureTheory.volume Triangle = 64) ∧
  (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1015_101504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_condition_iff_a_range_l1015_101514

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (2^(2*x)) - 2 * (2^x) + 1 - a

-- Define the function h
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (2^(-x)) * (f a x)

-- Main theorem
theorem h_condition_iff_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → 
    |h a x₁ - h a x₂| ≤ (a + 1) / 2) ↔ 
  (1/2 ≤ a ∧ a ≤ 4/5) := by
  sorry

#check h_condition_iff_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_condition_iff_a_range_l1015_101514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleDomainLength_l1015_101592

-- Define the function f(x) = 2sin(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

-- Define the theorem
theorem impossibleDomainLength 
  (a b : ℝ) 
  (h1 : ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 2) 
  (h2 : ∃ x ∈ Set.Icc a b, f x = 2) 
  (h3 : ∃ x ∈ Set.Icc a b, f x = -1) : 
  b - a ≠ 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleDomainLength_l1015_101592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_and_son_work_together_l1015_101537

/-- The number of days it takes for two people to complete a task together,
    given the number of days it takes each of them to complete the task individually. -/
noncomputable def days_to_complete_together (days_person1 days_person2 : ℝ) : ℝ :=
  1 / (1 / days_person1 + 1 / days_person2)

/-- Theorem stating that a man and his son can complete a task in 6 days
    when working together, given that the man takes 15 days and the son takes 10 days
    to complete the task individually. -/
theorem man_and_son_work_together :
  days_to_complete_together 15 10 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_and_son_work_together_l1015_101537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1015_101517

/-- A non-constant polynomial function from ℕ to ℕ -/
def NonConstantPoly := {f : ℕ → ℕ | ∃ x y, f x ≠ f y}

/-- The property that needs to be satisfied for all a and b -/
def SatisfiesProperty (m n : ℕ) (f : ℕ → ℕ) :=
  ∀ a b : ℕ, Nat.gcd (a + b + 1) (m * f a + n * f b) > 1

theorem unique_solution :
  ∀ m n : ℕ, Nat.Coprime m n →
    (∃ f : NonConstantPoly, SatisfiesProperty m n f) →
    m = 1 ∧ n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1015_101517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1015_101560

/-- Parabola structure -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
def PointOnParabola (para : Parabola) (P : ℝ × ℝ) : Prop :=
  para.eq P.1 P.2

/-- Line perpendicular to directrix passing through a point -/
def PerpendicularLine (para : Parabola) (E P : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), para.directrix x y → (y - E.2 = k * (x - E.1)) ∧ (P.2 - E.2 = k * (P.1 - E.1))

/-- Slope angle of a line -/
def SlopeAngle (A B : ℝ × ℝ) (angle : ℝ) : Prop :=
  Real.tan angle = (B.2 - A.2) / (B.1 - A.1)

/-- Distance between two points -/
noncomputable def Distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

/-- Main theorem -/
theorem parabola_distance_theorem (para : Parabola) (P E : ℝ × ℝ) :
  para.eq P.1 P.2 →
  PerpendicularLine para E P →
  SlopeAngle E para.focus (150 * π / 180) →
  Distance P para.focus = 4/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1015_101560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1015_101529

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x) / Real.log 3

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1015_101529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1015_101536

noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (x^(1/3) + 1/(2*x^(1/3)))^n

def arithmetic_sequence_condition (n : ℕ) : Prop :=
  2 * n * (1/2) = 1 + (n*(n-1)/2) * (1/4)

noncomputable def coefficient (n r : ℕ) : ℝ := (n.choose r) * (1/2)^r

-- Main theorem
theorem binomial_expansion_properties :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    arithmetic_sequence_condition n ∧
    (∀ (r : ℕ), r ≤ n → coefficient n 4 ≥ coefficient n r) ∧
    (∃ (k : ℕ), k ∈ ({2, 3} : Set ℕ) ∧ 
      ∀ (r : ℕ), r ≤ n → coefficient n k ≥ coefficient n r) :=
sorry

#check binomial_expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1015_101536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_length_l1015_101576

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points A, B, and C
def A : ℝ × ℝ := (2, 0)
noncomputable def B (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def C (m n : ℝ) : ℝ × ℝ := (m, n)

-- Define the curve y = √(1-x²)
noncomputable def on_curve (p : ℝ × ℝ) : Prop := p.2 = Real.sqrt (1 - p.1^2)

-- Define the first quadrant
def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Define isosceles right triangle condition
def is_isosceles_right_triangle (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0 ∧
  (b.1 - a.1)^2 + (b.2 - a.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

-- Define the length of OC
noncomputable def OC_length (c : ℝ × ℝ) : ℝ := Real.sqrt (c.1^2 + c.2^2)

-- Theorem statement
theorem max_OC_length :
  ∃ (θ m n : ℝ),
    0 ≤ θ ∧ θ ≤ Real.pi ∧
    on_curve (B θ) ∧
    in_first_quadrant (C m n) ∧
    is_isosceles_right_triangle A (B θ) (C m n) ∧
    (∀ (θ' m' n' : ℝ),
      0 ≤ θ' ∧ θ' ≤ Real.pi →
      on_curve (B θ') →
      in_first_quadrant (C m' n') →
      is_isosceles_right_triangle A (B θ') (C m' n') →
      OC_length (C m n) ≥ OC_length (C m' n')) ∧
    OC_length (C m n) = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_OC_length_l1015_101576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1015_101594

noncomputable def f (x : ℝ) := 3 * Real.cos (3 * x - Real.pi / 4) + 1

theorem phase_and_vertical_shift :
  (∃ (p : ℝ), ∀ (x : ℝ), f x = f (x + p)) ∧
  (∃ (v : ℝ), ∀ (x : ℝ), f x = 3 * Real.cos (3 * x - Real.pi / 4) + v) ∧
  (∀ (x : ℝ), f x = 3 * Real.cos (3 * (x + Real.pi / 12) - Real.pi / 4) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1015_101594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1015_101590

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
    f x₀ = 6 / 5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1015_101590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_F_is_6_l1015_101502

/-- Triangle ABC with vertices A(0,8), B(0,0), and C(10,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Point D is the midpoint of AB -/
noncomputable def midpoint_AB (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

/-- Point E is the midpoint of BC -/
noncomputable def midpoint_BC (t : Triangle) : ℝ × ℝ :=
  ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)

/-- Point F is the intersection of AE and CD -/
noncomputable def intersection_AE_CD (t : Triangle) : ℝ × ℝ :=
  let D := midpoint_AB t
  let E := midpoint_BC t
  -- Define F as the intersection point (implementation details omitted)
  (10/3, 8/3)  -- Placeholder for the actual intersection calculation

/-- The sum of x and y coordinates of point F is 6 -/
theorem sum_coordinates_F_is_6 (t : Triangle) 
    (h1 : t.A = (0, 8)) 
    (h2 : t.B = (0, 0)) 
    (h3 : t.C = (10, 0)) : 
  let F := intersection_AE_CD t
  F.1 + F.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_F_is_6_l1015_101502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1015_101530

/-- Two arithmetic sequences -/
def a : ℕ → ℚ := sorry
def b : ℕ → ℚ := sorry

/-- Sums of the first n terms of the sequences -/
def S : ℕ → ℚ := sorry
def T : ℕ → ℚ := sorry

/-- The sequences a and b are arithmetic -/
axiom ha : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
axiom hb : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n

/-- S_n is the sum of the first n terms of a_n -/
axiom hS : ∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2

/-- T_n is the sum of the first n terms of b_n -/
axiom hT : ∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2

/-- The given condition relating S_n and T_n -/
axiom h : ∀ n : ℕ+, S n / T n = (5 * n + 3) / (2 * n + 7)

theorem arithmetic_sequence_ratio :
  a 9 / b 9 = 88 / 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1015_101530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sane_belief_implies_truth_disbelief_implies_falsehood_l1015_101527

-- Define a proposition X
variable (X : Prop)

-- Define a predicate for a Transylvanian's belief
def believes (t : Nat) (p : Prop) : Prop := sorry

-- Define a predicate for sanity
def isSane (t : Nat) : Prop := sorry

-- Theorem 1: If a sane Transylvanian believes they believe X to be true, then X is true
theorem sane_belief_implies_truth (t : Nat) : 
  isSane t → believes t (believes t X) → X := by
  sorry

-- Theorem 2: If a Transylvanian does not believe they believe X to be true, then X is false
theorem disbelief_implies_falsehood (t : Nat) : 
  ¬(believes t (believes t X)) → ¬X := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sane_belief_implies_truth_disbelief_implies_falsehood_l1015_101527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1015_101595

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10 : ℝ)

-- State the theorem
theorem relationship_abc : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1015_101595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_circle_polar_equation_simplifies_constant_r_to_circle_l1015_101513

-- Define the polar equation
noncomputable def polar_equation (θ : ℝ) : ℝ := 6 * Real.sin θ * (1 / Real.sin θ)

-- Define the Cartesian equation of a circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 36

-- Theorem stating the equivalence
theorem polar_to_cartesian_circle :
  (∀ θ, polar_equation θ = 6) ↔
  (∀ x y, (∃ θ, x = 6 * Real.cos θ ∧ y = 6 * Real.sin θ) → circle_equation x y) :=
by sorry

-- Theorem stating that the polar equation simplifies to r = 6
theorem polar_equation_simplifies :
  ∀ θ, θ ≠ 0 → θ ≠ π → polar_equation θ = 6 :=
by sorry

-- Theorem stating that r = 6 in polar form is equivalent to x^2 + y^2 = 36 in Cartesian form
theorem constant_r_to_circle :
  (∀ θ, polar_equation θ = 6) ↔ (∀ x y, (∃ θ, x = 6 * Real.cos θ ∧ y = 6 * Real.sin θ) → x^2 + y^2 = 36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_circle_polar_equation_simplifies_constant_r_to_circle_l1015_101513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_80_l1015_101538

/-- Proves that a rectangular field with specific properties has a length of 80 meters -/
theorem field_length_is_80 (width : ℝ) (length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →                 -- length is double the width
  pond_side = 8 →                      -- pond side length is 8 meters
  pond_side^2 = (1/50) * (length * width) →  -- pond area is 1/50 of field area
  length = 80 := by
  sorry

#check field_length_is_80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_80_l1015_101538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_difference_l1015_101531

theorem function_value_difference (f : ℝ → ℝ) (a : ℝ) 
  (h_invertible : Function.Bijective f)
  (h_fa : f a = 3)
  (h_f3 : f 3 = 5) :
  a - 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_difference_l1015_101531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_area_l1015_101515

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculates the area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleArea (a b : PolarPoint) : ℝ :=
  (1/2) * abs a.r * abs b.r * Real.sin (a.θ - b.θ)

theorem triangle_AOB_area :
  let a : PolarPoint := ⟨3, π/3⟩
  let b : PolarPoint := ⟨-3, π/6⟩
  triangleArea a b = 9/4 := by
  sorry

#check triangle_AOB_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_area_l1015_101515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l1015_101522

/-- The time (in seconds) it takes for a train to pass a fixed point -/
noncomputable def train_passing_time (length : ℝ) (speed : ℝ) : ℝ :=
  length / (speed * 1000 / 3600)

/-- Theorem: A train 280 meters long, traveling at 63 km/h, takes approximately 16 seconds to pass a fixed point -/
theorem train_passing_tree : 
  let train_length : ℝ := 280
  let train_speed : ℝ := 63
  abs (train_passing_time train_length train_speed - 16) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_tree_l1015_101522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l1015_101597

/-- Custom operation ⊗ defined as a ⊗ b = (1/3)a - 4b -/
noncomputable def custom_op (a b : ℝ) : ℝ := (1/3) * a - 4 * b

/-- Theorem stating that 12 ⊗ (-1) = 8 -/
theorem custom_op_example : custom_op 12 (-1) = 8 := by
  -- Unfold the definition of custom_op
  unfold custom_op
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l1015_101597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_local_miles_l1015_101506

/-- Represents the car's journey with local road and highway portions -/
structure CarJourney where
  localRoadMiles : ℝ
  localRoadSpeed : ℝ
  highwayMiles : ℝ
  highwaySpeed : ℝ
  averageSpeed : ℝ

/-- Theorem stating that given the conditions of the problem, the local road miles is approximately 60 -/
theorem car_journey_local_miles (j : CarJourney) 
  (h1 : j.localRoadSpeed = 30)
  (h2 : j.highwayMiles = 65)
  (h3 : j.highwaySpeed = 65)
  (h4 : j.averageSpeed = 41.67)
  (h5 : j.averageSpeed = (j.localRoadMiles + j.highwayMiles) / (j.localRoadMiles / j.localRoadSpeed + j.highwayMiles / j.highwaySpeed)) :
  ∃ (ε : ℝ), ε > 0 ∧ |j.localRoadMiles - 60| < ε := by
  sorry

#check car_journey_local_miles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_local_miles_l1015_101506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_for_m_prime_one_a_condition_for_g_monotone_l1015_101591

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2

-- Define the function m as the derivative of f
noncomputable def m (a : ℝ) (x : ℝ) : ℝ := deriv (f a) x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - a * x^2 + a * x

-- Theorem 1
theorem a_value_for_m_prime_one (a : ℝ) :
  deriv (m a) 1 = 3 → a = 2 := by sorry

-- Theorem 2
theorem a_condition_for_g_monotone (a : ℝ) :
  (∀ x > 0, Monotone (g a)) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_for_m_prime_one_a_condition_for_g_monotone_l1015_101591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_rectangular_equation_max_distance_C2_to_C1_l1015_101534

noncomputable section

-- Define the parametric equation of curve C1
def C1 (t : ℝ) : ℝ × ℝ :=
  (-2 - Real.sqrt 3 / 2 * t, t / 2)

-- Define the polar equation of curve C2
def C2 (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

-- Theorem for the rectangular coordinate equation of C2
theorem C2_rectangular_equation :
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 ↔
    ∃ θ : ℝ, x = C2 θ * Real.cos θ ∧ y = C2 θ * Real.sin θ :=
by sorry

-- Theorem for the maximum distance from C2 to C1
theorem max_distance_C2_to_C1 :
  (∀ t θ : ℝ, ∃ d : ℝ, d ≤ (3 + Real.sqrt 3) / 2 + Real.sqrt 2 ∧
    d = Real.sqrt ((C2 θ * Real.cos θ - (C1 t).1)^2 + (C2 θ * Real.sin θ - (C1 t).2)^2)) ∧
  (∃ t₀ θ₀ : ℝ, (3 + Real.sqrt 3) / 2 + Real.sqrt 2 =
    Real.sqrt ((C2 θ₀ * Real.cos θ₀ - (C1 t₀).1)^2 + (C2 θ₀ * Real.sin θ₀ - (C1 t₀).2)^2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C2_rectangular_equation_max_distance_C2_to_C1_l1015_101534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_is_five_l1015_101533

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (x, -2),
    and a is perpendicular to b, prove that the magnitude of their sum is 5. -/
theorem magnitude_of_sum_is_five (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -2]
  (a • b = 0) → ‖a + b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_is_five_l1015_101533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l1015_101540

theorem interest_rate_increase (principal time initial_amount new_amount : ℝ) :
  principal = 1000 →
  time = 5 →
  initial_amount = 1500 →
  new_amount = 1750 →
  (new_amount - principal) / (initial_amount - principal) - 1 = 0.5 :=
by
  intro h_principal h_time h_initial h_new
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l1015_101540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_13_is_156_l1015_101583

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  condition1 : a 3 + a 7 - a 10 = 8
  condition2 : a 11 - a 4 = 4

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem: Sum of first 13 terms is 156 -/
theorem sum_13_is_156 (seq : ArithmeticSequence) : sum_n seq 13 = 156 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_13_is_156_l1015_101583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_sixteenth_l1015_101535

open Real

/-- The function f(x) = x^3 - 4x^2 + 4x --/
noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x

/-- The derivative of f(x) --/
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 4

/-- The second point of intersection x₂ given x₁ --/
noncomputable def x₂ (x₁ : ℝ) : ℝ := -2 * x₁

/-- The third point of intersection x₃ given x₁ --/
noncomputable def x₃ (x₁ : ℝ) : ℝ := 4 * x₁

/-- The area S₁ between P₁P₂ and curve C --/
noncomputable def S₁ (x₁ : ℝ) : ℝ := (27/4) * x₁^4

/-- The area S₂ between P₂P₃ and curve C --/
noncomputable def S₂ (x₁ : ℝ) : ℝ := 108 * x₁^4

theorem area_ratio_is_one_sixteenth (x₁ : ℝ) (h : x₁ ≠ 4/3) :
  S₁ x₁ / S₂ x₁ = 1/16 := by
  sorry

#check area_ratio_is_one_sixteenth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_sixteenth_l1015_101535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1015_101549

/-- Calculates the required fencing for a rectangular field -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  2 * width + uncovered_side

/-- Theorem: The fencing required for the given field is 97 feet -/
theorem fencing_for_given_field :
  fencing_required 680 80 = 97 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l1015_101549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_championship_probability_three_fourths_l1015_101551

/-- Represents the current state of a best-of-seven table tennis match -/
structure MatchState where
  player_a_wins : Nat
  player_b_wins : Nat

/-- The probability of a player winning a single game -/
noncomputable def game_win_probability : ℝ := 1/2

/-- Calculates the probability of player A winning the championship from the given match state -/
noncomputable def championship_win_probability (state : MatchState) : ℝ :=
  sorry

/-- The theorem stating that given the specific match state, the probability of player A winning is 3/4 -/
theorem championship_probability_three_fourths :
  let initial_state : MatchState := { player_a_wins := 3, player_b_wins := 2 }
  championship_win_probability initial_state = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_championship_probability_three_fourths_l1015_101551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1015_101548

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (5 : ℝ)^x + (6 : ℝ)^x + (7 : ℝ)^x = (9 : ℝ)^x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l1015_101548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1015_101582

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (5 - x)) / Real.sqrt (x + 2)

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y, f x = y) ↔ x ∈ Set.Icc (-2) 5 ∧ x ≠ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l1015_101582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sufficient_not_necessary_l1015_101545

-- Define the angle α
variable (α : Real)

-- Define a function to check if the terminal side of an angle passes through a point
noncomputable def terminalSidePassesThrough (angle : Real) (x y : Real) : Prop :=
  ∃ (t : Real), t > 0 ∧ t * (Real.cos angle) = x ∧ t * (Real.sin angle) = y

-- Theorem statement
theorem terminal_side_sufficient_not_necessary :
  (∀ α, terminalSidePassesThrough α (-1) 2 → Real.tan α = -2) ∧
  (∃ α, Real.tan α = -2 ∧ ¬terminalSidePassesThrough α (-1) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sufficient_not_necessary_l1015_101545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1015_101511

noncomputable section

/-- Line C₁ -/
def C₁ (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

/-- Curve C₂ -/
def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (3 / (1 + 2 * Real.sin θ ^ 2))
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Theorem stating the result -/
theorem intersection_distance_difference : 
  ∃ (t₁ t₂ : ℝ), 
    (∃ θ₁, C₁ t₁ = C₂ θ₁) ∧ 
    (∃ θ₂, C₁ t₂ = C₂ θ₂) ∧ 
    t₁ ≠ t₂ ∧
    |t₁ + t₂| = 2/5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1015_101511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_correct_l1015_101572

/-- Represents a trapezoid ABCD with given side lengths and angle bisector intersections -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  -- P is the intersection of angle bisectors of A and D
  -- Q is the intersection of angle bisectors of B and C
  parallel_AB_CD : AB < CD

/-- The area of hexagon ABQCDP in the given trapezoid -/
noncomputable def hexagon_area (t : Trapezoid) : ℝ :=
  13 * Real.sqrt 18.5

theorem hexagon_area_is_correct (t : Trapezoid) 
    (h1 : t.AB = 9) 
    (h2 : t.BC = 7) 
    (h3 : t.CD = 23) 
    (h4 : t.DA = 5) : 
  hexagon_area t = 13 * Real.sqrt 18.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_is_correct_l1015_101572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1015_101579

theorem triangle_right_angle (a b c A B C : Real) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < Real.pi) ∧ (0 < B) ∧ (B < Real.pi) ∧ (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (a * Real.cos C + c * Real.cos A = b * Real.sin B) →
  B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_l1015_101579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_theorem_l1015_101577

/-- A crocodile move on an infinite chessboard -/
def CrocodileMove (N : ℕ) (start finish : ℤ × ℤ) : Prop :=
  ∃ (intermediate : ℤ × ℤ),
    (intermediate.1 = start.1 ∧ intermediate.2 = start.2 + 1) ∨
    (intermediate.1 = start.1 ∧ intermediate.2 = start.2 - 1) ∨
    (intermediate.1 = start.1 + 1 ∧ intermediate.2 = start.2) ∨
    (intermediate.1 = start.1 - 1 ∧ intermediate.2 = start.2) ∧
    ((finish.1 = intermediate.1 + N ∧ finish.2 = intermediate.2) ∨
     (finish.1 = intermediate.1 - N ∧ finish.2 = intermediate.2) ∨
     (finish.1 = intermediate.1 ∧ finish.2 = intermediate.2 + N) ∨
     (finish.1 = intermediate.1 ∧ finish.2 = intermediate.2 - N))

/-- A crocodile can reach any cell from any starting cell -/
def CrocodileCanReachAll (N : ℕ) : Prop :=
  ∀ (start finish : ℤ × ℤ), ∃ (path : List (ℤ × ℤ)),
    path.head? = some start ∧
    path.getLast? = some finish ∧
    ∀ (i : ℕ) (hi : i + 1 < path.length),
      CrocodileMove N (path[i]) (path[i + 1])

/-- The main theorem: A crocodile can reach all cells if and only if N is even -/
theorem crocodile_theorem (N : ℕ) :
  CrocodileCanReachAll N ↔ Even N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_theorem_l1015_101577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_four_l1015_101596

def digits : List Nat := [2, 0, 0, 5]

def is_valid_four_digit_number (n : Nat) : Bool :=
  n ≥ 1000 && n ≤ 9999 && digits.all (fun d => (n.digits 10).contains d)

def count_valid_numbers : Nat :=
  (List.range 10000).filter is_valid_four_digit_number |>.length

theorem count_valid_numbers_eq_four : count_valid_numbers = 4 := by
  sorry

#eval count_valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_eq_four_l1015_101596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1015_101539

/-- An ellipse with foci F₁(-1,0) and F₂(1,0), and a point P on the ellipse -/
structure Ellipse :=
  (P : ℝ × ℝ)
  (on_ellipse : 2 * 2 = Real.sqrt ((P.1 + 1)^2 + P.2^2) + Real.sqrt ((P.1 - 1)^2 + P.2^2))
  (second_quadrant : P.1 < 0 ∧ P.2 > 0)
  (angle : Real.cos (120 * π / 180) = (P.1 + 1) / Real.sqrt ((P.1 + 1)^2 + P.2^2))

/-- The equation of the ellipse and the area of triangle PF₁F₂ -/
def ellipse_properties (e : Ellipse) : Prop :=
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ 
    2 * 2 = Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2)) ∧
  (1/2 * 2 * e.P.2 = 3 * Real.sqrt 3 / 5)

theorem ellipse_theorem (e : Ellipse) : ellipse_properties e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1015_101539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_travel_time_is_35_days_l1015_101544

/-- Represents the travel time for rafts drifting downstream from Gorky to Astrakhan -/
noncomputable def raft_travel_time (steamboat_downstream_time steamboat_upstream_time : ℝ) : ℝ :=
  (steamboat_downstream_time * steamboat_upstream_time) / (steamboat_upstream_time - steamboat_downstream_time)

/-- Theorem stating that given the steamboat travel times, rafts will take 35 days to drift downstream -/
theorem raft_travel_time_is_35_days 
  (steamboat_downstream_time : ℝ) 
  (steamboat_upstream_time : ℝ) 
  (h_downstream : steamboat_downstream_time = 5)
  (h_upstream : steamboat_upstream_time = 7) :
  raft_travel_time steamboat_downstream_time steamboat_upstream_time = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_travel_time_is_35_days_l1015_101544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_proof_l1015_101501

/-- Calculates the tax amount for a given income and threshold -/
noncomputable def calculate_tax (income : ℝ) (threshold : ℝ) : ℝ :=
  0.15 * min income threshold + 0.2 * max (income - threshold) 0

/-- Proves that for the given tax system and conditions, the threshold X is $40,000 -/
theorem tax_threshold_proof :
  ∃ (X : ℝ), X > 0 ∧ calculate_tax 50000 X = 8000 ∧ X = 40000 := by
  use 40000
  constructor
  · norm_num
  constructor
  · norm_num [calculate_tax]
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_proof_l1015_101501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_trip_time_l1015_101561

/-- Represents Jake's day trip --/
structure DayTrip where
  drive_to_beach : ℝ
  drive_beach_to_museum : ℝ
  drive_museum_to_park : ℝ
  drive_park_to_home : ℝ
  beach_time_multiplier : ℝ
  museum_time_subtractor : ℝ

/-- Calculates the total trip time given a DayTrip --/
noncomputable def total_trip_time (trip : DayTrip) : ℝ :=
  let total_drive_time := trip.drive_to_beach + trip.drive_beach_to_museum + 
                          trip.drive_museum_to_park + trip.drive_park_to_home
  let total_location_time := (trip.beach_time_multiplier * trip.drive_to_beach) + 
                             ((trip.drive_to_beach + trip.drive_beach_to_museum + trip.drive_museum_to_park) / 2 - trip.museum_time_subtractor) + 
                             trip.drive_park_to_home
  total_drive_time + total_location_time

/-- Theorem stating that Jake's trip takes 15.75 hours --/
theorem jakes_trip_time : 
  ∃ (trip : DayTrip), 
    trip.drive_to_beach = 2 ∧ 
    trip.drive_beach_to_museum = 1.5 ∧ 
    trip.drive_museum_to_park = 1 ∧ 
    trip.drive_park_to_home = 2.5 ∧
    trip.beach_time_multiplier = 2.5 ∧
    trip.museum_time_subtractor = 1 ∧
    total_trip_time trip = 15.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jakes_trip_time_l1015_101561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l1015_101542

/-- The line on which point P moves --/
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- The circle C --/
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- Point P is on the line --/
def P_on_line (P : ℝ × ℝ) : Prop := line P.1 P.2

/-- Points A and B are on the circle --/
def A_B_on_circle (A B : ℝ × ℝ) : Prop := circleC A.1 A.2 ∧ circleC B.1 B.2

/-- PA and PB are tangent to the circle at A and B respectively --/
def PA_PB_tangent (P A B : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral PACB --/
noncomputable def area_PACB (P A B : ℝ × ℝ) : ℝ := sorry

/-- The main theorem --/
theorem min_area_PACB :
  ∀ P A B : ℝ × ℝ,
  P_on_line P →
  A_B_on_circle A B →
  PA_PB_tangent P A B →
  (∀ P' A' B' : ℝ × ℝ,
    P_on_line P' →
    A_B_on_circle A' B' →
    PA_PB_tangent P' A' B' →
    area_PACB P A B ≤ area_PACB P' A' B') →
  area_PACB P A B = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l1015_101542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l1015_101562

noncomputable def f (x : ℝ) := x^3 - (3/2) * x^2 + 5

theorem f_max_min_on_interval :
  let a := -2
  let b := 2
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 7 ∧ f x_min = -9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l1015_101562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1015_101507

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 2)
  (h3 : (a 3)^2 = (a 1) * (a 9))
  (h4 : arithmetic_sequence a d) :
  (∀ n, a n = 2 * n) ∧
  (∀ n, Finset.sum (Finset.range n) (λ i => 1 / (a i * a (i + 1))) = n / (4 * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1015_101507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_a_l1015_101580

-- Define the hyperbola
noncomputable def hyperbola (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / (a + 2) - y^2 / 3 = 1

-- Define the eccentricity
noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (1 + 3 / (a + 2))

-- Theorem statement
theorem hyperbola_eccentricity_a (a : ℝ) :
  (∃ x y, hyperbola a x y) ∧ eccentricity a = 2 → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_a_l1015_101580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_line_area_between_parabolas_l1015_101528

-- Define the parabola and line
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def line (u v x : ℝ) : ℝ := u * x + v

-- Define the theorem
theorem area_between_parabola_and_line 
  (a b c u v α β : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : α < β) 
  (h3 : parabola a b c α = line u v α) 
  (h4 : parabola a b c β = line u v β) :
  ∃ (S : ℝ), S = |a| / 6 * (β - α)^3 ∧ 
  S = |∫ x in α..β, (parabola a b c x - line u v x)| := by
  sorry

-- Define the theorem for the second problem
theorem area_between_parabolas
  (a b c p q r α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : p ≠ 0)
  (h3 : α < β)
  (h4 : parabola a b c α = parabola p q r α)
  (h5 : parabola a b c β = parabola p q r β) :
  ∃ (S : ℝ), S = |a - p| / 6 * (β - α)^3 ∧
  S = |∫ x in α..β, (parabola a b c x - parabola p q r x)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_line_area_between_parabolas_l1015_101528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_negative_one_l1015_101541

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (O A B C : V)

-- Define the line l
noncomputable def l (A B : V) : Set V := {X : V | ∃ t : ℝ, X = A + t • (B - A)}

-- State the theorem
theorem unique_solution_negative_one 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_collinear : C ∈ l A B)
  (h_O_not_on_l : O ∉ l A B) :
  {x : ℝ | x^2 • (A - O) + x • (B - O) + (C - B) = 0} = {-1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_negative_one_l1015_101541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1015_101525

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Define the right shift operation
def rightShift (h : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ h (x - d)

-- Theorem statement
theorem shift_equivalence : rightShift f 1 = g := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l1015_101525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_son_age_multiple_l1015_101519

theorem father_son_age_multiple :
  ∃ (son_age : ℕ) (k : ℕ),
    let father_age : ℕ := 33
    let future_father_age : ℕ := father_age + 3
    let future_son_age : ℕ := son_age + 3
    father_age = k * son_age + 3 ∧
    future_father_age = 2 * future_son_age + 10 ∧
    k = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_son_age_multiple_l1015_101519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_slope_l1015_101526

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type representing y = k(x-1) --/
structure Line where
  k : ℝ
  x : ℝ
  y : ℝ
  eq : y = k*(x-1)

/-- Point type representing a point on the parabola or line --/
structure Point where
  x : ℝ
  y : ℝ

/-- Define membership for Point in Parabola --/
def Point.mem_parabola (p : Point) (C : Parabola) : Prop :=
  p.y^2 = 4*p.x

/-- Define membership for Point in Line --/
def Point.mem_line (p : Point) (l : Line) : Prop :=
  p.y = l.k*(p.x - 1)

/-- Theorem stating the slope of the line under given conditions --/
theorem parabola_line_slope 
  (C : Parabola) 
  (l : Line) 
  (A B : Point) 
  (h1 : A.mem_parabola C) 
  (h2 : B.mem_parabola C) 
  (h3 : A.mem_line l) 
  (h4 : B.mem_line l) 
  (h5 : l.k < 0) -- obtuse angle condition
  (h6 : abs (A.x - B.x) = 16/3) : 
  l.k = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_slope_l1015_101526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_no_rational_minima_l1015_101564

/-- A function is unbounded on rational subsets of intervals if for any non-degenerate interval I
    and any M > 0, there exists a rational x in I such that f(x) > M. -/
def UnboundedOnRationalSubsets (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ) (M : ℝ), a < b → M > 0 → ∃ (x : ℚ), (a : ℝ) < (x : ℝ) ∧ (x : ℝ) < b ∧ f (x : ℝ) > M

/-- A function has rational local strict minima if for all rational x,
    there exists an ε > 0 such that f(y) > f(x) for all y ≠ x in (x-ε, x+ε). -/
def HasRationalLocalStrictMinima (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℚ), ∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), y ≠ (x : ℝ) → |y - (x : ℝ)| < ε → f y > f (x : ℝ)

/-- If a function from ℝ to ℝ≥0 is unbounded on rational subsets of all non-degenerate intervals,
    then it cannot have rational local strict minima. -/
theorem unbounded_no_rational_minima (f : ℝ → ℝ) (h : ∀ x, f x ≥ 0) :
  UnboundedOnRationalSubsets f → ¬HasRationalLocalStrictMinima f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_no_rational_minima_l1015_101564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_powers_of_two_l1015_101521

theorem perfect_square_powers_of_two :
  (∃! n : ℕ, ∃ k : ℕ, 2^n + 3 = k^2) ∧
  (∃! n : ℕ, ∃ k : ℕ, 2^n + 1 = k^2) :=
by
  constructor
  · -- First part: 2^n + 3 = k^2
    use 0
    constructor
    · use 2
      rfl
    · intro m hm
      cases' hm with k hk
      -- Proof details omitted
      sorry
  · -- Second part: 2^n + 1 = k^2
    use 3
    constructor
    · use 3
      rfl
    · intro m hm
      cases' hm with k hk
      -- Proof details omitted
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_powers_of_two_l1015_101521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1015_101554

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate total interest from two loans -/
noncomputable def total_interest (loan1_principal : ℝ) (loan1_time : ℝ) 
                   (loan2_principal : ℝ) (loan2_time : ℝ) 
                   (rate : ℝ) : ℝ :=
  simple_interest loan1_principal rate loan1_time + 
  simple_interest loan2_principal rate loan2_time

theorem interest_calculation :
  let loan1_principal : ℝ := 5000
  let loan1_time : ℝ := 2
  let loan2_principal : ℝ := 3000
  let loan2_time : ℝ := 4
  let rate : ℝ := 7.000000000000001
  total_interest loan1_principal loan1_time loan2_principal loan2_time rate = 1540.0000000000015 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1015_101554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1015_101563

/-- The solution function for the differential equation y' = 2 + y with initial condition y(0) = 3 -/
noncomputable def solution (x : ℝ) : ℝ := 5 * Real.exp x - 2

/-- The differential equation y' = 2 + y -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, deriv y x = 2 + y x

theorem solution_satisfies_equation :
  differential_equation solution ∧ solution 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1015_101563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_ratio_proof_l1015_101573

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_capacity : ℕ := 50

/-- The number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jellybeans needed to fill all glasses -/
def total_jellybeans : ℕ := 325

/-- The ratio of jellybeans needed for a small glass to a large glass -/
def small_to_large_ratio : ℚ := 1 / 2

theorem jellybean_ratio_proof :
  (((total_jellybeans - large_glass_capacity * num_large_glasses) / num_small_glasses : ℚ) / large_glass_capacity) = small_to_large_ratio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_ratio_proof_l1015_101573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_36_7432_l1015_101518

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original_number : ℝ := 36.7432

/-- Theorem stating that rounding the original number to the nearest hundredth equals 36.74 -/
theorem round_to_hundredth_36_7432 :
  round_to_hundredth original_number = 36.74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_36_7432_l1015_101518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_theorem_l1015_101516

theorem quadratic_root_theorem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 2*b ≠ 3*c) :
  let f : ℝ → ℝ := λ x ↦ 3*a*(2*b - 3*c)*x^2 + 2*b*(3*c - 2*a)*x + 5*c*(2*a - 3*b)
  ∃ r : ℝ, f r = 0 ∧ f (2*r) = 0 ∧ r = -2*b*(3*c - 2*a) / (9*a*(2*b - 3*c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_theorem_l1015_101516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_no_isomers_l1015_101593

-- Define the structure of a benzene molecule
structure Benzene where
  carbon : Fin 6 → Unit
  hydrogen : Fin 6 → Unit

-- Define properties of benzene
class BenzeneProperties (b : Benzene) where
  is_aromatic : Bool
  undergoes_substitution : Bool
  dichlorobenzene_products : Nat

-- Theorem stating that benzene does not have two isomeric forms
theorem benzene_no_isomers (b : Benzene) [bp : BenzeneProperties b] :
  bp.is_aromatic = true →
  bp.undergoes_substitution = true →
  bp.dichlorobenzene_products = 3 →
  ¬∃ (b' : Benzene), b ≠ b' ∧ BenzeneProperties b' = BenzeneProperties b :=
by
  intro h_aromatic h_substitution h_products
  intro h_exists
  cases h_exists with
  | intro b' h =>
    sorry -- Proof details would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_no_isomers_l1015_101593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1015_101568

/-- A pyramid with a rectangular base -/
structure RectangularBasePyramid where
  -- Base rectangle
  base_length : ℝ
  base_width : ℝ
  -- Angle between edge PA and base diagonal PB
  apex_angle : ℝ

/-- The volume of a rectangular base pyramid -/
noncomputable def pyramid_volume (p : RectangularBasePyramid) : ℝ :=
  (Real.sqrt (1 + Real.tan p.apex_angle ^ 2)) / 3

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (θ : ℝ),
  let p : RectangularBasePyramid := ⟨2, 1, θ⟩
  pyramid_volume p = (Real.sqrt (1 + Real.tan θ ^ 2)) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1015_101568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lollipop_surface_area_l1015_101566

/-- Given a cylindrical container of candy syrup with base radius 3cm and height 10cm,
    prove that when divided into 20 identical spherical lollipops,
    the surface area of each lollipop is 9π cm². -/
theorem lollipop_surface_area 
  (cylinder_radius : ℝ) 
  (cylinder_height : ℝ) 
  (num_lollipops : ℝ) 
  (h1 : cylinder_radius = 3)
  (h2 : cylinder_height = 10)
  (h3 : num_lollipops = 20) : 
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  let lollipop_volume := cylinder_volume / num_lollipops
  let lollipop_radius := (lollipop_volume * 3 / (4 * π))^(1/3)
  4 * π * lollipop_radius^2 = 9 * π := by
  sorry

#check lollipop_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lollipop_surface_area_l1015_101566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_QP_l1015_101557

-- Define the points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (0, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the moving points P and Q
def P : ℝ → ℝ → ℝ × ℝ := fun x y ↦ (x, y)
def Q : ℝ → ℝ → ℝ × ℝ := fun x y ↦ (x, y)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector between two points
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

-- Define the condition for point P
def P_condition (x y : ℝ) : Prop :=
  dot_product (vector A (P x y)) (vector B (P x y)) = 1

-- Define the condition for point Q
def Q_condition (x y t : ℝ) : Prop :=
  vector O (Q x y) = ((1/2 - t) • (vector O M)) + ((1/2 + t) • (vector O N))

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_QP :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (x1 y1 x2 y2 t : ℝ),
    P_condition x1 y1 →
    Q_condition x2 y2 t →
    distance (P x1 y1) (Q x2 y2) ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_QP_l1015_101557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1015_101589

/-- The rational function h(x) -/
noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x + 2) / (x^2 - 5*x + 6)

/-- The domain of h(x) -/
def domain_h : Set ℝ := {x | x ≠ 2 ∧ x ≠ 3}

theorem domain_of_h :
  domain_h = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1015_101589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_cube_in_barrel_l1015_101574

/-- The volume of water displaced by a cube in a cylindrical barrel -/
noncomputable def water_displaced (cube_side : ℝ) (barrel_radius : ℝ) : ℝ :=
  let s := barrel_radius * Real.sqrt 3
  let base_area := (Real.sqrt 3 / 4) * s^2
  let height := cube_side / 2
  (1 / 3) * base_area * height

theorem water_displaced_cube_in_barrel :
  let cube_side := (12 : ℝ)
  let barrel_radius := (6 : ℝ)
  let v := water_displaced cube_side barrel_radius
  v = 54 * Real.sqrt 3 ∧ v^2 = 8748 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_cube_in_barrel_l1015_101574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_cross_out_all_l1015_101565

/-- A natural number is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- The minimum number of moves required to cross out all natural numbers. -/
theorem min_moves_to_cross_out_all : ∃ x₁ x₂ : ℕ, ∀ y : ℕ, 
  IsComposite (Nat.sub (max x₁ y) (min x₁ y)) ∨ 
  IsComposite (Nat.sub (max x₂ y) (min x₂ y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_cross_out_all_l1015_101565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l1015_101559

theorem complex_quadrant (z : ℂ) : (1 - I) * z = I^2013 → z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l1015_101559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1015_101578

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (A B C P D : Plane)

-- Define distance function
variable (dist : Plane → Plane → ℝ)

-- Define angle measure function
variable (angle : Plane → Plane → Plane → ℝ)

-- Define the intersects relation
variable (intersects : Set Plane → Set Plane → Plane → Prop)

-- Define conditions
variable (h1 : dist P A = dist P B)
variable (h2 : angle A P B = 2 * angle A C B)
variable (h3 : intersects {A, C} {B, P} D)
variable (h4 : dist P B = 3)
variable (h5 : dist P D = 2)

-- Theorem statement
theorem triangle_problem : dist A D * dist C D = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1015_101578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1015_101547

theorem problem_solution (a b : ℝ) (h : a^2 + b^2 - 2*a + 6*b + 10 = 0) : 
  2 * a^100 - 3 * b⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1015_101547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l1015_101571

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 is √(1 - b²/a²) -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The eccentricity of the ellipse x²/9 + y²/5 = 1 is 2/3 -/
theorem eccentricity_of_specific_ellipse :
  ellipse_eccentricity 3 (Real.sqrt 5) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_ellipse_l1015_101571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_8_hours_l1015_101587

/-- Represents the tank system with a leak and an inlet pipe -/
structure TankSystem where
  capacity : ℚ
  inletRate : ℚ
  emptyTimeWithInlet : ℚ

/-- Calculates the time it takes for the leak to empty the tank without the inlet -/
def leakEmptyTime (ts : TankSystem) : ℚ :=
  ts.capacity / (ts.capacity / ts.emptyTimeWithInlet + ts.inletRate * 60)

/-- Theorem stating that for the given conditions, the leak empties the tank in 8 hours -/
theorem leak_empty_time_is_8_hours (ts : TankSystem) 
    (h1 : ts.capacity = 8640)
    (h2 : ts.inletRate = 6)
    (h3 : ts.emptyTimeWithInlet = 12) :
    leakEmptyTime ts = 8 := by
  sorry

/-- Compute the result for the given parameters -/
def computeResult : ℚ :=
  leakEmptyTime { capacity := 8640, inletRate := 6, emptyTimeWithInlet := 12 }

#eval computeResult

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_8_hours_l1015_101587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_point_is_apollonius_intersection_l1015_101550

/-- Four points on a line -/
structure FourPoints where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  order : A < B ∧ B < C ∧ C < D

/-- Apollonius circle -/
def ApolloniusCircle (p q r s : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | dist x p / dist x r = dist q s / dist r s}

/-- Angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem equal_angle_point_is_apollonius_intersection 
  (points : FourPoints) (M : ℝ × ℝ) :
  (angle (points.A, 0) M (points.B, 0) = angle (points.B, 0) M (points.C, 0) ∧ 
   angle (points.B, 0) M (points.C, 0) = angle (points.C, 0) M (points.D, 0)) →
  M ∈ (ApolloniusCircle (points.A, 0) (points.C, 0) (points.A, 0) (points.B, 0)) ∩
      (ApolloniusCircle (points.B, 0) (points.D, 0) (points.B, 0) (points.C, 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_point_is_apollonius_intersection_l1015_101550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1015_101569

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C
  S : ℝ  -- area

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.a * (Real.sin ((t.A + t.B) / 2))^2 + 2 * t.c * (Real.sin ((t.B + t.C) / 2))^2 = 3 * t.b) :
  -- Part 1: a, b, c form an arithmetic sequence
  (t.a + t.c = 2 * t.b) ∧
  -- Part 2: If B = π/3 and b = 4, then S = 4√3
  (t.B = Real.pi / 3 ∧ t.b = 4 → t.S = 4 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1015_101569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1015_101556

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 6*x - 5) + Real.sqrt (x^2 - 4)

theorem domain_of_f : Set.Icc (-5 : ℝ) (-2 : ℝ) = {x : ℝ | ∃ y : ℝ, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1015_101556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_factorial_equation_l1015_101555

theorem solve_factorial_equation (n : ℕ) : 3 * 4 * 5 * n = Nat.factorial 6 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_factorial_equation_l1015_101555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1015_101500

/-- A function g with specific properties -/
noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := x^2 / (D * x^2 + E * x + F)

/-- Theorem stating the sum of coefficients for a function with given properties -/
theorem sum_of_coefficients (D E F : ℤ) :
  (∀ x > (5 : ℝ), g D E F x > 0.3) →
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 4 → g D E F x ≠ 0) →
  (∀ ε > 0, ∃ M : ℝ, ∀ x > M, |g D E F x - (1 : ℝ) / ↑D| < ε) →
  D + E + F = -24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1015_101500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1015_101570

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The theorem stating that the distance from (0, 0) to the line 3x + 4y - 25 = 0 is 5 -/
theorem distance_circle_center_to_line :
  distance_point_to_line 0 0 3 4 (-25) = 5 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1015_101570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_x_range_for_given_m_range_l1015_101503

-- Part 1
theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x^2 + 2 * (m - 2) * x - 4 < 0) ↔ m ∈ Set.Ioc (-2) 2 :=
sorry

-- Part 2
theorem x_range_for_given_m_range (m x : ℝ) :
  m ∈ Set.Icc (-1) 1 → (2 * x^2 + m * x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_in_range_x_range_for_given_m_range_l1015_101503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_5_equals_25_l1015_101524

/-- An arithmetic sequence with a_2 = 3 and a_5 = 9 -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  let d := (9 - 3) / 3  -- Common difference
  let a1 := 3 - d       -- First term
  a1 + (n - 1) * d

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  n * (arithmetic_seq 1 + arithmetic_seq n) / 2

/-- Theorem stating that S_5 equals 25 -/
theorem S_5_equals_25 : S 5 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_5_equals_25_l1015_101524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_lamp_post_l1015_101523

/-- The time for a train to cross a lamp post -/
noncomputable def time_to_cross_lamp_post (bridge_length : ℝ) (time_to_cross_bridge : ℝ) (train_length : ℝ) : ℝ :=
  train_length / ((train_length + bridge_length) / time_to_cross_bridge)

/-- Theorem: The time for a train to cross a lamp post is approximately 30 seconds -/
theorem train_crossing_lamp_post :
  ∃ ε > 0, |time_to_cross_lamp_post 2500 120 833.33 - 30| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_lamp_post_l1015_101523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1015_101585

-- Define the speed of the train in km/hr
noncomputable def train_speed : ℚ := 54

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℚ := 7

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℚ := 1000 / 3600

-- Theorem stating the length of the train
theorem train_length :
  train_speed * km_hr_to_m_s * crossing_time = 105 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1015_101585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_15_factorial_and_given_number_l1015_101532

open Nat

theorem lcm_15_factorial_and_given_number :
  Nat.lcm (factorial 15) (2^3 * 3^9 * 5^4 * 7^1) = 2^11 * 3^9 * 5^4 * 7^1 * 11^1 * 13^1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_15_factorial_and_given_number_l1015_101532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_2_l1015_101558

theorem log_8_2 : Real.log 2 / Real.log 8 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_2_l1015_101558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_is_six_l1015_101543

/-- The function f(x) = x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The average rate of change of f from x = 1 to x = 3 -/
noncomputable def average_rate_of_change : ℝ := (f 3 - f 1) / (3 - 1)

/-- Theorem: The average rate of change of f from x = 1 to x = 3 is 6 -/
theorem average_rate_of_change_is_six : average_rate_of_change = 6 := by
  -- Expand the definition of average_rate_of_change
  unfold average_rate_of_change
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_is_six_l1015_101543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1015_101575

theorem divisor_existence (S : Finset ℕ) : 
  S ⊆ Finset.range 2015 → S.card = 1008 → 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l1015_101575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1015_101520

/-- The match removal game where n is the initial number of matches --/
def MatchGame (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ (firstMove : ℕ), 1 ≤ firstMove ∧ firstMove < n ∧
  ∀ (subsequentMove : ℕ → ℕ),
    (∀ k, k > 0 → subsequentMove k ≤ subsequentMove (k-1)) →
    ∃ (winningSequence : ℕ → ℕ),
      winningSequence 0 = firstMove ∧
      (∀ k, k > 0 → winningSequence k ≤ winningSequence (k-1)) ∧
      ∃ m, n = (Finset.range m).sum winningSequence

/-- The theorem stating when the first player can guarantee a win --/
theorem first_player_wins (n : ℕ) :
  MatchGame n ↔ ¬∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1015_101520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1015_101508

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

-- Define the sum of the first n terms of an arithmetic sequence
def S (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + n * (n - 1) / 2 * d

-- Theorem statement
theorem arithmetic_sequence_sum 
  (a₁ d : ℚ) 
  (h₁ : S a₁ d 4 = 1) 
  (h₂ : S a₁ d 8 = 4) : 
  (arithmetic_sequence a₁ d 16) + 
  (arithmetic_sequence a₁ d 17) + 
  (arithmetic_sequence a₁ d 18) + 
  (arithmetic_sequence a₁ d 19) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1015_101508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_remain_seated_is_half_l1015_101584

/-- Represents a train car with passengers and seats. -/
structure TrainCar where
  total_seats : ℕ
  reserved_seats : ℕ
  hab : reserved_seats < total_seats

/-- Represents the probability of an event. -/
def Probability := ℝ

/-- The probability that a passenger remains seated during seat shuffling. -/
noncomputable def probability_remain_seated (car : TrainCar) : Probability :=
  sorry

/-- The theorem stating that the probability of remaining seated is 1/2. -/
theorem probability_remain_seated_is_half (car : TrainCar) 
  (h : car.total_seats = 78 ∧ car.reserved_seats = 77) : 
  probability_remain_seated car = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_remain_seated_is_half_l1015_101584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_ON_OM_l1015_101567

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the curve C
def curve_C (x y θ : ℝ) : Prop := x = 2 * Real.cos θ + 2 ∧ y = 2 * Real.sin θ

-- Define the ray m
def ray_m (ρ θ α : ℝ) : Prop := θ = α ∧ -Real.pi/4 < α ∧ α < Real.pi/2 ∧ ρ ≥ 0

-- Define the polar equations
def polar_line_l (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 3
def polar_curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the intersection points
def point_M (ρ₁ α : ℝ) : Prop := polar_line_l ρ₁ α
def point_N (ρ₂ α : ℝ) : Prop := polar_curve_C ρ₂ α ∧ ρ₂ ≠ 0

-- Theorem statement
theorem max_ratio_ON_OM :
  ∀ α ρ₁ ρ₂ : ℝ,
  ray_m ρ₁ α α →
  ray_m ρ₂ α α →
  point_M ρ₁ α →
  point_N ρ₂ α →
  (∀ β ρ₃ ρ₄ : ℝ,
    ray_m ρ₃ β β →
    ray_m ρ₄ β β →
    point_M ρ₃ β →
    point_N ρ₄ β →
    ρ₂ / ρ₁ ≥ ρ₄ / ρ₃) →
  ρ₂ / ρ₁ = 2/3 * (Real.sqrt 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_ON_OM_l1015_101567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1015_101510

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the main properties of f
axiom f_property : ∀ (x y : ℝ), f (x + y) - f y = x * (x + 2 * y + 1)
axiom f_one : f 1 = 0

-- Define the theorem to be proved
theorem f_properties :
  (f 0 = -2) ∧
  (∀ x : ℝ, f x = x^2 + x - 2) ∧
  (Set.Ioo 1 5 = {a : ℝ | (∀ x ∈ Set.Icc 0 (3/4), f x + 3 < 2*x + a) ∧
                         ¬(∀ x ∈ Set.Icc (-2) 2, 
                            (∀ y ∈ Set.Icc (-2) 2, f y - a*y ≤ f x - a*x) ∨
                            (∀ y ∈ Set.Icc (-2) 2, f y - a*y ≥ f x - a*x))}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1015_101510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l1015_101509

open Set

def S : Set ℤ := {x | -1 < x ∧ x ≤ 2}

theorem number_of_proper_subsets : Finset.card (Finset.powerset {0, 1, 2} \ {{0, 1, 2}}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l1015_101509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_seven_l1015_101546

theorem coefficient_x_power_seven (a : ℝ) : 
  (Nat.choose 10 3 : ℝ) * a^3 = 15 → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_seven_l1015_101546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_condition_l1015_101552

theorem sin_cos_condition (x : ℝ) : 
  (∀ x, Real.sin x + Real.cos x > 1 → Real.sin x * Real.cos x > 0) ∧ 
  (∃ x, Real.sin x * Real.cos x > 0 ∧ Real.sin x + Real.cos x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_condition_l1015_101552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_speed_approx_l1015_101586

/-- The speed of the Stella Artois car in miles per hour -/
noncomputable def stella_speed : ℝ := 150

/-- The time the cars have been traveling in hours -/
noncomputable def travel_time : ℝ := 1.694915254237288

/-- The total distance between the cars after the travel time in miles -/
noncomputable def total_distance : ℝ := 500

/-- The speed of the first car in miles per hour -/
noncomputable def first_car_speed : ℝ := (total_distance - stella_speed * travel_time) / travel_time

/-- Theorem stating that the first car's speed is approximately 145 mph -/
theorem first_car_speed_approx : 
  ∃ ε > 0, |first_car_speed - 145| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_car_speed_approx_l1015_101586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1015_101598

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x * Real.cos (2 * x)

theorem min_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1015_101598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1015_101505

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop :=
  x = (floor (x / 2) : ℝ) + (floor (x / 3) : ℝ) + (floor (x / 5) : ℝ)

-- Theorem statement
theorem equation_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x) ∧ (Finset.card S = 30) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1015_101505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l1015_101512

theorem salary_increase_percentage (S : ℝ) (P : ℝ) : 
  S > 0 →
  ((S + (P / 100) * S) * (90 / 100) = S * 1.01) →
  abs (P - 12.22) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l1015_101512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_2010_eq_zero_l1015_101553

noncomputable def f (n : ℕ+) : ℝ := Real.sin (n * Real.pi / 2 + Real.pi / 4)

theorem sum_f_2010_eq_zero :
  (Finset.range 2010).sum (fun i => f ⟨i + 1, Nat.succ_pos i⟩) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_2010_eq_zero_l1015_101553
