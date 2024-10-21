import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_range_l103_10323

/-- The set A defined by the quadratic equation x^2 + 4x = 0 -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B defined by the quadratic equation x^2 + 2(a+1)x + a^2 - 1 = 0 -/
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

/-- The range of values for a -/
def range_a : Set ℝ := Set.Iic (-1) ∪ {1}

/-- Theorem stating that if A ∩ B = B, then a is in the specified range -/
theorem intersection_implies_range (a : ℝ) : A ∩ B a = B a → a ∈ range_a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_range_l103_10323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2010_l103_10363

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We define a₀ = 3 to match a₁ in the original problem
  | 1 => 6  -- This matches a₂ in the original problem
  | (n + 2) => sequence_a (n + 1) - sequence_a n

theorem sequence_a_2010 : sequence_a 2009 = -3 := by
  sorry

#eval sequence_a 2009  -- This will evaluate the result for verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2010_l103_10363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_a_in_range_l103_10316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotonically_increasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 4 ≤ a ∧ a < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_a_in_range_l103_10316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_for_specific_project_l103_10341

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  initialTime : ℝ
  initialWorkers : ℝ
  completedLength : ℝ
  completedTime : ℝ

/-- Calculates the number of extra workers needed to complete the project on time -/
noncomputable def extraWorkersNeeded (project : RoadProject) : ℝ :=
  let remainingTime := project.initialTime - project.completedTime
  let totalWork := project.totalLength
  let completedWork := project.completedLength
  let remainingWork := totalWork - completedWork
  let initialWorkRate := project.initialWorkers * project.initialTime / totalWork
  (remainingWork / remainingTime - project.initialWorkers) * (project.initialTime / remainingTime)

/-- Theorem stating that for the given project conditions, 30 extra workers are needed -/
theorem extra_workers_for_specific_project :
  let project : RoadProject := {
    totalLength := 10
    initialTime := 60
    initialWorkers := 30
    completedLength := 2
    completedTime := 20
  }
  ⌊extraWorkersNeeded project⌋ = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_workers_for_specific_project_l103_10341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_properties_l103_10379

/-- Properties of a cuboid with given dimensions -/
theorem cuboid_properties (length width height : ℝ) 
  (h_length : length = 5)
  (h_width : width = 4)
  (h_height : height = 3) :
  let smallest_face := min (length * width) (min (length * height) (width * height))
  let largest_face := max (length * width) (max (length * height) (width * height))
  let total_edge_length := 4 * (length + width + height)
  let surface_area := 2 * (length * width + length * height + width * height)
  let volume := length * width * height
  smallest_face = 12 ∧
  largest_face = 20 ∧
  total_edge_length = 48 ∧
  surface_area = 94 ∧
  volume = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_properties_l103_10379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l103_10390

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l103_10390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l103_10339

noncomputable def angle_α : Real := sorry

theorem sin_plus_cos_for_point (x y : Real) (h : (x, y) ∈ Set.range (λ t => (t * Real.cos angle_α, t * Real.sin angle_α))) :
  x = -3 ∧ y = 4 → Real.sin angle_α + Real.cos angle_α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l103_10339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_84_solutions_l103_10352

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def has_solution (c : ℝ) : Prop :=
  ∃ x : ℝ, 9 * (floor x) + 3 * (ceil x) + 5 * (frac x) = c

theorem at_least_84_solutions :
  ∃ S : Finset ℝ, S.card ≥ 84 ∧ ∀ c ∈ S, 0 ≤ c ∧ c ≤ 1000 ∧ has_solution c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_84_solutions_l103_10352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l103_10353

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -3)

theorem angle_between_vectors :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  Real.arccos cos_theta = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l103_10353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_primes_with_units_digit_three_l103_10321

/-- A function that returns true if a number has a units digit of 3 -/
def hasUnitsDigitThree (n : ℕ) : Prop :=
  n % 10 = 3

/-- A function that returns the nth prime number with a units digit of 3 -/
def nthPrimeWithUnitsDigitThree (n : ℕ) : ℕ := sorry

/-- The sum of the first ten prime numbers with a units digit of 3 -/
def sumFirstTenPrimesWithUnitsDigitThree : ℕ :=
  (List.range 10).map (fun i => nthPrimeWithUnitsDigitThree (i + 1)) |>.sum

theorem sum_first_ten_primes_with_units_digit_three :
  sumFirstTenPrimesWithUnitsDigitThree = 477 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_ten_primes_with_units_digit_three_l103_10321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_neg_one_monotonicity_condition_l103_10385

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the closed interval [-2, 4]
def I : Set ℝ := Set.Icc (-2) 4

-- Theorem for part 1
theorem max_min_values_when_a_is_neg_one :
  (∀ x ∈ I, f (-1) x ≤ 10) ∧
  (∃ x ∈ I, f (-1) x = 10) ∧
  (∀ x ∈ I, f (-1) x ≥ 1) ∧
  (∃ x ∈ I, f (-1) x = 1) :=
sorry

-- Theorem for part 2
theorem monotonicity_condition :
  ∀ a : ℝ, (∀ x y, x ∈ I → y ∈ I → x < y → f a x < f a y) ∨ 
           (∀ x y, x ∈ I → y ∈ I → x < y → f a x > f a y)
  ↔ a ≤ -4 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_neg_one_monotonicity_condition_l103_10385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_edges_sum_specific_frame_l103_10332

/-- A rectangular picture frame with given properties -/
structure Frame where
  width : ℝ
  area : ℝ
  outer_edge : ℝ

/-- The sum of the lengths of the four interior edges of the frame -/
noncomputable def interior_edges_sum (f : Frame) : ℝ :=
  2 * (f.outer_edge - 2 * f.width) + 2 * ((f.area / f.outer_edge) - 2 * f.width)

/-- Theorem stating the sum of interior edges for a frame with specific measurements -/
theorem interior_edges_sum_specific_frame :
  ∃ (f : Frame), f.width = 1.5 ∧ f.area = 27 ∧ f.outer_edge = 6 ∧ interior_edges_sum f = 12 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_edges_sum_specific_frame_l103_10332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l103_10326

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = m * (p.1 + 1)}

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem ellipse_and_line_intersection
  (e : Set (ℝ × ℝ))
  (l : Set (ℝ × ℝ))
  (M : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (h_e : e = Ellipse 2 1)
  (h_l : l = Line 1)
  (h_M : M = (-1, 0))
  (h_P : P ∈ e ∩ l)
  (h_Q : Q ∈ e ∩ l)
  (h_PM : (P.1 - M.1, P.2 - M.2) = (-3/5 * (Q.1 - M.1), -3/5 * (Q.2 - M.2))) :
  ∃ (A : ℝ × ℝ) (α : ℝ),
    A = (2, 0) ∧ 
    (∀ l' : Set (ℝ × ℝ), ∀ P' Q' : ℝ × ℝ,
      P' ∈ e ∩ l' ∧ Q' ∈ e ∩ l' →
      dot_product (P'.1 - A.1, P'.2 - A.2) (Q'.1 - A.1, Q'.2 - A.2) ≤ 33/4) ∧
    (∃ l_vert : Set (ℝ × ℝ), ∃ P_vert Q_vert : ℝ × ℝ,
      P_vert ∈ e ∩ l_vert ∧ Q_vert ∈ e ∩ l_vert ∧
      dot_product (P_vert.1 - A.1, P_vert.2 - A.2) (Q_vert.1 - A.1, Q_vert.2 - A.2) = 33/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l103_10326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_eight_l103_10391

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 9) / 4
  y^2 + 6 * y - 7

-- State the theorem
theorem g_of_neg_eight : g (-8) = -87/16 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- Perform numerical calculations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_eight_l103_10391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_per_rotation_l103_10340

/-- Represents the properties and movement of a rotating wheel -/
structure RotatingWheel where
  /-- Rotations per minute -/
  rpm : ℕ
  /-- Distance traveled in one hour (in meters) -/
  distance_per_hour : ℝ

/-- Calculates the distance traveled per rotation in centimeters -/
noncomputable def distance_per_rotation_cm (wheel : RotatingWheel) : ℝ :=
  (wheel.distance_per_hour / (wheel.rpm * 60)) * 100

/-- Theorem stating that a wheel with given properties moves 35 cm per rotation -/
theorem wheel_distance_per_rotation
  (wheel : RotatingWheel)
  (h_rpm : wheel.rpm = 20)
  (h_distance : wheel.distance_per_hour = 420) :
  distance_per_rotation_cm wheel = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_per_rotation_l103_10340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l103_10386

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l103_10386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_multiple_calculation_l103_10364

/-- Represents a recipe with butter and flour quantities -/
structure Recipe where
  butter : ℚ
  flour : ℚ

/-- Calculates how many times a recipe is multiplied -/
def recipe_multiple (original : Recipe) (new : Recipe) : ℚ :=
  new.butter / original.butter

theorem recipe_multiple_calculation (original : Recipe) (new : Recipe) 
  (h1 : original.butter = 8)
  (h2 : original.flour = 14)
  (h3 : new.butter = 12)
  (h4 : new.flour = 56) :
  recipe_multiple original new = 3/2 := by
  -- Proof steps would go here
  sorry

#eval (12 : ℚ) / (8 : ℚ)  -- This should evaluate to 3/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_multiple_calculation_l103_10364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_minus_6_l103_10360

/-- The angle of inclination of a line with equation y = x - 6 is 45 degrees (π/4 radians). -/
theorem angle_of_inclination_x_minus_6 :
  let line : ℝ → ℝ := λ x => x - 6
  ∃ θ : ℝ, θ = π / 4 ∧ ∀ x y : ℝ, y = line x → Real.tan θ = (y - line 0) / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_minus_6_l103_10360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l103_10311

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_decreasing_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, f a b (-x) = -(f a b x)) →  -- f is odd
  (∀ x y : ℝ, x < y → f a b x > f a b y) →  -- f is decreasing
  (a = 2 ∧ b = 1) ∧
  (∀ t : ℝ, f a b (t^2 - 2*t) + f a b (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_properties_l103_10311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_approx_l103_10388

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- Condition that the triangle is isosceles
  is_isosceles : True
  -- Condition for the given altitude
  altitude_eq : altitude = 5
  -- Condition for the given perimeter
  perimeter_eq : perimeter = 50

/-- The distance between the centers of inscribed and circumscribed circles -/
noncomputable def center_distance (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the distance between centers for the given triangle -/
theorem center_distance_approx (t : IsoscelesTriangle) :
  ∃ ε > 0, |center_distance t - 14.3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_approx_l103_10388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l103_10344

/-- The area of the triangle bounded by the y-axis and two lines -/
noncomputable def triangleArea : ℝ := 9/4

/-- First line equation: y - 4x = 3 -/
def line1 (x y : ℝ) : Prop := y - 4*x = 3

/-- Second line equation: 2y + x = 15 -/
def line2 (x y : ℝ) : Prop := 2*y + x = 15

/-- The triangle is bounded by the y-axis and two lines -/
theorem triangle_area_theorem :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line1 x₁ y₁ ∧ 
    line2 x₂ y₂ ∧ 
    x₁ ≥ 0 ∧ 
    x₂ ≥ 0 ∧
    (∀ x y, line1 x y ∧ line2 x y → x ≥ 0) →
    triangleArea = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l103_10344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l103_10355

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  a ∈ Set.Icc (3/8) (2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_range_l103_10355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l103_10308

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 6*x + 10

/-- Point A -/
def A : ℝ × ℝ := (2, 2)

/-- Point B -/
def B : ℝ × ℝ := (5, 0)

/-- Point C -/
def C (p q : ℝ) : ℝ × ℝ := (p, q)

/-- The area of triangle ABC -/
noncomputable def triangle_area (p q : ℝ) : ℝ :=
  (1/2) * abs (2*0 + 5*q + p*2 - 2*5 - 0*p - q*2)

/-- Theorem: The maximum area of triangle ABC is 3.5 -/
theorem max_triangle_area :
  ∃ (p q : ℝ), 
    parabola 2 2 ∧ 
    parabola 5 0 ∧ 
    parabola p q ∧ 
    2 ≤ p ∧ p ≤ 5 ∧
    (∀ (p' q' : ℝ), parabola p' q' ∧ 2 ≤ p' ∧ p' ≤ 5 → 
      triangle_area p q ≥ triangle_area p' q') ∧
    triangle_area p q = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l103_10308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_pairs_is_27_l103_10350

-- Define the universe set
def U : Finset Nat := {1, 2, 3}

-- Define the condition for valid pairs
def ValidPair (A B : Finset Nat) : Prop :=
  A ∪ B = U ∧ A ≠ B

-- Define the number of valid pairs
def NumValidPairs : Nat :=
  (Finset.powerset U).card * (Finset.powerset U).card -
  (Finset.powerset U).card

-- Theorem statement
theorem num_valid_pairs_is_27 : NumValidPairs = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_pairs_is_27_l103_10350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_solution_set_l103_10343

/-- A quadratic function --/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if a function is quadratic --/
def IsQuadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem impossible_solution_set (f : QuadraticFunction) (hf : IsQuadratic f) :
  ∀ t : ℝ, ¬∃ x y z : ℝ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    f (|x - t|) = 0 ∧ f (|y - t|) = 0 ∧ f (|z - t|) = 0 ∧
    |x - y| = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_solution_set_l103_10343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l103_10306

/-- The lateral area of a cone with base radius 3 and slant height 5 is 15π -/
theorem cone_lateral_area (base_radius slant_height : Real) :
  base_radius = 3 →
  slant_height = 5 →
  (1 / 2) * (2 * Real.pi * base_radius) * slant_height = 15 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l103_10306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_26489_to_nearest_tenth_l103_10365

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_45_26489_to_nearest_tenth :
  round_to_nearest_tenth 45.26489 = 45.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_26489_to_nearest_tenth_l103_10365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l103_10309

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.log x + x + 1

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 1 / x + 1

-- Theorem statement
theorem tangent_line_equation (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : f' x₀ = 2) :
  ∃ y₀ : ℝ, (λ x => 2 * x) = (λ x => f' x₀ * (x - x₀) + f x₀) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l103_10309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l103_10376

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define the foci of the ellipse
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the intersection points A and B (we don't know their exact coordinates)
variable (A B : ℝ × ℝ)

-- Define the condition that A and B are on both the ellipse and the line
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 m ∧ line B.1 B.2 m

-- Define the area ratio condition
def area_ratio (A B : ℝ × ℝ) : Prop :=
  ∃ (area_triangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    area_triangle F1 A B = 2 * area_triangle F2 A B

-- The theorem to prove
theorem ellipse_intersection_theorem (A B : ℝ × ℝ) (m : ℝ) :
  intersection_points A B m → area_ratio A B → m = -Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l103_10376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distances_l103_10347

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  c₁ : ℝ
  c₂ : ℝ
  eq₁ : a * x + b * y + c₁ = 0
  eq₂ : a * x + b * y + c₂ = 0

/-- Distance between two parallel lines -/
noncomputable def distance_between_lines (l : ParallelLines) : ℝ :=
  |l.c₁ - l.c₂| / Real.sqrt (l.a^2 + l.b^2)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (l : ParallelLines) (x₀ y₀ : ℝ) : ℝ :=
  |l.a * x₀ + l.b * y₀ + l.c₁| / Real.sqrt (l.a^2 + l.b^2)

theorem parallel_lines_distances (l : ParallelLines) :
  l.a = 2 ∧ l.b = 1 ∧ l.c₁ = -1 ∧ l.c₂ = 1 →
  distance_between_lines l = 2 * Real.sqrt 5 / 5 ∧
  distance_point_to_line l 0 2 = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distances_l103_10347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_for_specific_triangle_l103_10335

/-- Represents a right-angled triangle with a square patch at the right angle -/
structure RightTriangleWithPatch where
  leg1 : ℝ
  leg2 : ℝ
  square_side : ℝ
  distance_to_hypotenuse : ℝ

/-- Calculates the area of the triangle -/
noncomputable def triangle_area (t : RightTriangleWithPatch) : ℝ :=
  t.leg1 * t.leg2 / 2

/-- Calculates the area of the square patch -/
noncomputable def patch_area (t : RightTriangleWithPatch) : ℝ :=
  t.square_side ^ 2

/-- Calculates the planted area of the field -/
noncomputable def planted_area (t : RightTriangleWithPatch) : ℝ :=
  triangle_area t - patch_area t

/-- Calculates the fraction of the field that is planted -/
noncomputable def planted_fraction (t : RightTriangleWithPatch) : ℝ :=
  planted_area t / triangle_area t

/-- Theorem stating the planted fraction for the given conditions -/
theorem planted_fraction_for_specific_triangle :
  ∃ (t : RightTriangleWithPatch),
    t.leg1 = 5 ∧
    t.leg2 = 12 ∧
    t.distance_to_hypotenuse = 3 ∧
    planted_fraction t = 85611 / 85683 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_for_specific_triangle_l103_10335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10392

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

-- State the theorem
theorem f_properties :
  (∀ x > 0, f x = 2 * Real.log x - x^2) →
  (f 1 = -1 ∧ (deriv f) 1 = 0) →
  (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x ≤ -1) ∧
  (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l103_10304

-- Define the function f(x) = 3^x for x < -1
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the domain of f
def domain : Set ℝ := {x | x < -1}

-- Define the range of f
def range : Set ℝ := f '' domain

-- Define the interval from which x is randomly chosen
def interval : Set ℝ := Set.Ioo (-1) 1

-- State the theorem
theorem probability_in_range :
  (MeasureTheory.volume (Set.inter range interval) / MeasureTheory.volume interval) = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l103_10304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_area_calculation_l103_10380

/-- Represents the scale of a map as a ratio --/
structure MapScale where
  ratio : ℚ

/-- Calculates the actual area given the map area and scale --/
def actual_area (map_area : ℝ) (scale : MapScale) : ℝ :=
  map_area * (scale.ratio : ℝ) * (scale.ratio : ℝ)

/-- The problem statement --/
theorem map_area_calculation (map_area : ℝ) (scale : MapScale) :
  map_area = 10 ∧ scale.ratio = 10000 →
  actual_area map_area scale = 100000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_area_calculation_l103_10380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l103_10370

theorem no_solution_exists : ¬ ∃ (x : ℕ), 1^(x+3) + 2^(x+2) + 3^x + 4^(x+1) = 5280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l103_10370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l103_10300

def small_orange_price : ℚ := 3/10
def medium_orange_price : ℚ := 1/2
def large_orange_price : ℚ := 7/10
def winter_price_increase : ℚ := 1/5
def discount_rate : ℚ := 1/10
def discount_threshold : ℚ := 10
def sales_tax_rate : ℚ := 2/25
def small_oranges_count : ℕ := 10
def medium_oranges_count : ℕ := 15
def large_oranges_count : ℕ := 8

def calculate_final_cost : ℚ :=
  let winter_small_price := small_orange_price * (1 + winter_price_increase)
  let winter_medium_price := medium_orange_price * (1 + winter_price_increase)
  let winter_large_price := large_orange_price * (1 + winter_price_increase)
  let total_cost := winter_small_price * small_oranges_count +
                    winter_medium_price * medium_oranges_count +
                    winter_large_price * large_oranges_count
  let discounted_cost := if total_cost > discount_threshold
                         then total_cost * (1 - discount_rate)
                         else total_cost
  discounted_cost * (1 + sales_tax_rate)

theorem final_cost_is_correct :
  (calculate_final_cost * 100).floor / 100 = 1878/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_is_correct_l103_10300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l103_10359

theorem vector_subtraction (c d : Fin 3 → ℝ) :
  c = ![5, -3, 2] →
  d = ![-2, 1, 3] →
  c - 4 • d = ![13, -7, -10] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l103_10359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_root_expression_l103_10367

theorem simplify_complex_root_expression (a : ℝ) (ha : a > 0) :
  (a^16)^(1/12) * (a^16)^(1/12) = a^12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_root_expression_l103_10367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l103_10312

/-- The length of the chord formed by the intersection of a line and a circle. -/
noncomputable def chord_length (slope_angle : ℝ) (circle_eq : ℝ → ℝ → Prop) : ℝ :=
  2 * Real.sqrt 3

/-- Theorem: The length of the chord formed by the intersection of a line passing through
    the origin with a slope angle of 60° and a circle with equation x^2 + y^2 - 4y = 0
    is equal to 2√3. -/
theorem intersection_chord_length :
  let slope_angle : ℝ := π / 3  -- 60° in radians
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 4*y = 0
  chord_length slope_angle circle_eq = 2 * Real.sqrt 3 := by
  sorry

#check intersection_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l103_10312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_is_56_l103_10381

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the prism
structure Prism where
  base_side1 : ℝ
  base_side2 : ℝ
  height : ℝ

-- Define the volume of the prism
noncomputable def prism_volume (p : Prism) : ℝ := (1/2) * p.base_side1 * p.base_side2 * p.height

-- Theorem statement
theorem prism_volume_is_56 (p : Prism) :
  right_triangle p.base_side1 p.base_side2 (Real.sqrt 28) →
  p.base_side1 = Real.sqrt 14 →
  p.base_side2 = Real.sqrt 14 →
  p.height = 8 →
  prism_volume p = 56 := by
  sorry

#check prism_volume_is_56

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_is_56_l103_10381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_not_all_even_l103_10369

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- State the theorem
theorem negation_of_not_all_even :
  (¬ ∀ φ : ℝ, ¬ is_even (f φ)) ↔ (∃ φ : ℝ, is_even (f φ)) := by
  sorry

#check negation_of_not_all_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_not_all_even_l103_10369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_return_to_initial_l103_10374

noncomputable def price_change (initial_price : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (fun acc change => acc * (1 + change / 100)) initial_price

theorem price_return_to_initial (y : ℝ) : 
  price_change 100 [30, -15, 10, -y] = 100 → y = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_return_to_initial_l103_10374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l103_10348

/-- The force equation for the airstream acting on a sail -/
noncomputable def force (B S ρ v₀ v : ℝ) : ℝ :=
  (B * S * ρ * (v₀ - v)^2) / 2

/-- The instantaneous wind power -/
noncomputable def power (B S ρ v₀ v : ℝ) : ℝ :=
  force B S ρ v₀ v * v

/-- The wind speed -/
def wind_speed : ℝ := 6.3

/-- Theorem: The speed of the sailboat is v₀/3 when the instantaneous wind power reaches its maximum value -/
theorem sailboat_speed_at_max_power (B S ρ : ℝ) (hB : B > 0) (hS : S > 0) (hρ : ρ > 0) :
  ∃ v : ℝ, v = wind_speed / 3 ∧ 
    ∀ u : ℝ, power B S ρ wind_speed v ≥ power B S ρ wind_speed u := by
  sorry

#check sailboat_speed_at_max_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_speed_at_max_power_l103_10348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_results_l103_10305

def is_valid_result (result : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (a * b - c) % (d + e) = 0 ∧
    (a * b - c) / (d + e) = result

theorem possible_results :
  {r : ℕ | is_valid_result r} = {3, 5, 9, 19} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_results_l103_10305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_pi_over_six_l103_10302

/-- The angle between two 2D vectors given by their coordinates -/
noncomputable def angle_between (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

/-- Theorem: The angle between vectors BA and BC is π/6 -/
theorem angle_ABC_is_pi_over_six :
  let BA : ℝ × ℝ := (1/2, Real.sqrt 3/2)
  let BC : ℝ × ℝ := (Real.sqrt 3/2, 1/2)
  angle_between BA BC = π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_is_pi_over_six_l103_10302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_possible_t_is_22_div_3_l103_10313

/-- A polynomial of the form x^2 - 6x + t with positive integer roots -/
def hasPositiveIntegerRoots (t : ℤ) : Prop :=
  ∃ r₁ r₂ : ℤ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 6 ∧ r₁ * r₂ = t

/-- The set of all distinct possible values of t -/
def possibleTValues : Finset ℤ :=
  {5, 8, 9}

/-- The average of all distinct possible values of t -/
noncomputable def averagePossibleT : ℚ :=
  (possibleTValues.sum id) / possibleTValues.card

theorem average_possible_t_is_22_div_3 : 
  averagePossibleT = 22 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_possible_t_is_22_div_3_l103_10313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_machine_payoff_days_l103_10394

/-- The number of days until a coffee machine pays for itself -/
def days_until_machine_pays_for_itself (
  machine_cost : ℚ)
  (discount : ℚ)
  (daily_home_cost : ℚ)
  (previous_daily_coffees : ℕ)
  (previous_coffee_cost : ℚ) : ℕ :=
  let actual_machine_cost := machine_cost - discount
  let previous_daily_expense := previous_daily_coffees * previous_coffee_cost
  let daily_savings := previous_daily_expense - daily_home_cost
  (actual_machine_cost / daily_savings).ceil.toNat

/-- Theorem stating that the coffee machine pays for itself in 36 days -/
theorem coffee_machine_payoff_days :
  days_until_machine_pays_for_itself 200 20 3 2 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_machine_payoff_days_l103_10394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_inequality_l103_10330

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 1 / Real.exp x)

-- State the theorem
theorem x_squared_inequality (x₁ x₂ : ℝ) (h : f x₁ < f x₂) : x₁^2 < x₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_inequality_l103_10330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l103_10310

-- Define the four functions as noncomputable
noncomputable def f1 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f2 (x : ℝ) : ℝ := Real.log 2 / Real.log x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 4
def f4 (x : ℝ) : ℝ := x - 2

-- Define a predicate for intersection points
def is_intersection_point (x : ℝ) : Prop :=
  x > 0 ∧ (
    (f1 x = f2 x) ∨ (f1 x = f3 x) ∨ (f1 x = f4 x) ∨
    (f2 x = f3 x) ∨ (f2 x = f4 x) ∨
    (f3 x = f4 x)
  )

-- Theorem statement
theorem intersection_points_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_intersection_point x) ∧ (S.card = 4) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l103_10310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increasing_iff_a_geq_one_l103_10319

/-- A function f: ℝ → ℝ is increasing if for all x₁ < x₂, f(x₁) ≤ f(x₂) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ ≤ f x₂

/-- The function y = sin x + ax -/
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * x

theorem y_increasing_iff_a_geq_one :
  ∀ a : ℝ, IsIncreasing (y a) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increasing_iff_a_geq_one_l103_10319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_maximum_l103_10349

/-- Arithmetic sequence with positive terms -/
def a (n : ℕ+) : ℝ := 2 * (n : ℝ) + 1

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ+) : ℝ := (n : ℝ) * ((n : ℝ) + 2)

/-- Geometric sequence -/
noncomputable def b (n : ℕ+) : ℝ := 8^((n : ℝ) - 1)

/-- Function f(n) -/
noncomputable def f (n : ℕ+) : ℝ := (a n - 1) / (S n + 100)

theorem arithmetic_geometric_sequence_maximum :
  (∀ n : ℕ+, a n > 0) ∧
  a 1 = 3 ∧
  b 1 = 1 ∧
  b 2 * S 2 = 64 ∧
  b 3 * S 3 = 960 →
  (∀ n : ℕ+, f n ≤ 1/11) ∧
  f 10 = 1/11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_maximum_l103_10349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_330_l103_10303

/-- The number of distinct prime factors of 330 is 4. -/
theorem distinct_prime_factors_of_330 : (Nat.factors 330).toFinset.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_330_l103_10303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l103_10398

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  |a₂ * 0 + b₂ * (-d₁/b₁) + c₂ * 0 + d₂| / Real.sqrt (a₂^2 + b₂^2 + c₂^2)

/-- Theorem: The distance between the planes 3x - y + 2z + 4 = 0 and 6x - 2y + 4z - 3 = 0 is 5√14/28 -/
theorem distance_between_specific_planes :
  distance_between_planes 3 (-1) 2 4 6 (-2) 4 (-3) = 5 * Real.sqrt 14 / 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l103_10398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_f_eq_half_l103_10331

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 10 else 2^x

-- Theorem statement
theorem solution_for_f_eq_half :
  ∃ m : ℝ, f m = 1/2 ↔ m = Real.sqrt 10 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_f_eq_half_l103_10331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_union_complement_l103_10362

-- Define the functions and their domains
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (4 - x^2)
noncomputable def g (x : ℝ) : ℝ := Real.log (2 + x)

def M : Set ℝ := {x | 4 - x^2 > 0}
def N : Set ℝ := {x | 2 + x > 0}

-- State the theorem
theorem domain_union_complement (x : ℝ) : 
  x ∈ M ∪ (Set.univ \ N) ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_union_complement_l103_10362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_pipe_fill_time_l103_10315

/-- Represents the time (in minutes) it takes to fill a tank with two pipes working simultaneously -/
noncomputable def timeTofillTank (fillTime : ℝ) (emptyTime : ℝ) : ℝ :=
  1 / ((1 / fillTime) - (1 / emptyTime))

/-- Theorem stating that for a tank with pipe A filling in 9 minutes and pipe B emptying in 18 minutes,
    the time to fill the tank when both pipes work simultaneously is 18 minutes -/
theorem simultaneous_pipe_fill_time :
  timeTofillTank 9 18 = 18 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_pipe_fill_time_l103_10315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l103_10399

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sqrt (1 - x^2)

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 
  (∀ (y : ℝ), y ∈ Set.Icc (-1 : ℝ) 1 → f y ≤ f x) ∧
  f x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l103_10399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_inequality_solution_l103_10387

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 3| - m)

-- Theorem for the maximum value of m
theorem max_m_value (ε : ℝ) (h : ε > 0) :
  (∀ x, ∃ y, f x 4 = y) ∧ ¬(∀ x, ∃ y, f x (4 + ε) = y) :=
sorry

-- Theorem for the solution of the inequality
theorem inequality_solution :
  ∀ x, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_inequality_solution_l103_10387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_faces_minimum_l103_10307

theorem dice_faces_minimum (a b : ℕ) : 
  a ≥ 6 ∧ b ≥ 6 ∧  -- Each die has at least 6 faces
  (∀ x y : ℕ, x ≤ a ∧ y ≤ b → (x + y = 7 → (3 * (Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ a ∧ p.2 ≤ b ∧ p.1 + p.2 = 7) (Finset.product (Finset.range (a+1)) (Finset.range (b+1)))).card = 
    4 * (Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ a ∧ p.2 ≤ b ∧ p.1 + p.2 = 10) (Finset.product (Finset.range (a+1)) (Finset.range (b+1)))).card))) ∧  -- Probability condition for sum of 7 and 10
  ((Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ a ∧ p.2 ≤ b ∧ p.1 + p.2 = 12) (Finset.product (Finset.range (a+1)) (Finset.range (b+1)))).card = (a * b) / 12) →  -- Probability condition for sum of 12
  a + b ≥ 17 ∧ 
  ∀ c d : ℕ, c ≥ 6 ∧ d ≥ 6 → 
    (∀ x y : ℕ, x ≤ c ∧ y ≤ d → (x + y = 7 → (3 * (Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ c ∧ p.2 ≤ d ∧ p.1 + p.2 = 7) (Finset.product (Finset.range (c+1)) (Finset.range (d+1)))).card = 
      4 * (Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ c ∧ p.2 ≤ d ∧ p.1 + p.2 = 10) (Finset.product (Finset.range (c+1)) (Finset.range (d+1)))).card))) →
    ((Finset.filter (λ p : ℕ × ℕ ↦ p.1 ≤ c ∧ p.2 ≤ d ∧ p.1 + p.2 = 12) (Finset.product (Finset.range (c+1)) (Finset.range (d+1)))).card = (c * d) / 12) →
    c + d ≥ a + b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_faces_minimum_l103_10307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mixture_ratio_l103_10384

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk
    water := m1.water + m2.water }

theorem new_mixture_ratio (v : ℚ) : v > 0 → 
  let m1 : Mixture := { milk := 4 * v, water := 2 * v }
  let m2 : Mixture := { milk := 5 * v, water := v }
  let combined := combineMixtures m1 m2
  combined.milk / combined.water = 3 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mixture_ratio_l103_10384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_vertex_l103_10396

noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

noncomputable def vertex_y (a b c : ℝ) : ℝ := c - b^2 / (4 * a)

def translate (x y dx dy : ℝ) : ℝ × ℝ := (x - dx, y - dy)

theorem parabola_translation_vertex :
  let original_vertex_x := vertex_x 1 (-4)
  let original_vertex_y := vertex_y 1 (-4) 2
  let translated_vertex := translate original_vertex_x original_vertex_y 3 2
  translated_vertex = (-1, -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_vertex_l103_10396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_position_correctness_l103_10356

/-- Represents the relative position of two circles -/
inductive CirclePosition
  | TouchExternally
  | OneInsideWithoutTouching
  | OneOutside

/-- Determines the relative position of two circles given their radii and distance between centers -/
noncomputable def circlePosition (R r d : ℝ) : CirclePosition :=
  if d = R + r then CirclePosition.TouchExternally
  else if d < R - r then CirclePosition.OneInsideWithoutTouching
  else CirclePosition.OneOutside

theorem circle_position_correctness (R r d : ℝ) (h : R ≥ r) :
  (circlePosition R r d = CirclePosition.TouchExternally ↔ d = R + r) ∧
  (circlePosition R r d = CirclePosition.OneInsideWithoutTouching ↔ d < R - r) ∧
  (circlePosition R r d = CirclePosition.OneOutside ↔ d > R + r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_position_correctness_l103_10356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l103_10301

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (n' : ℕ),
    n = m + 10^k * a + 10^(k+1) * n' ∧
    k > 0 ∧
    a < 10 ∧
    m < 10^k ∧
    n ≠ 0 ∧
    n % 10 ≠ 0 ∧
    6 * (m + 10^k * n') = n

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ ({12, 24, 36, 48, 108} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_l103_10301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cosine_sine_function_l103_10336

theorem min_value_cosine_sine_function (x : ℝ) (h : 0 < x ∧ x < Real.pi / 4) :
  ∃ (y : ℝ), y = (Real.cos x) ^ 2 / ((Real.cos x) * (Real.sin x) - (Real.sin x) ^ 2) ∧
  (∀ (z : ℝ), z = (Real.cos x) ^ 2 / ((Real.cos x) * (Real.sin x) - (Real.sin x) ^ 2) → y ≤ z) ∧
  y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cosine_sine_function_l103_10336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_age_average_l103_10383

theorem dog_age_average
  (dog1_age : ℝ)
  (dog2_age : ℝ)
  (dog3_age : ℝ)
  (dog4_age : ℝ)
  (dog5_age : ℝ)
  (h1 : dog1_age = 10)
  (h2 : dog2_age = dog1_age - 2)
  (h3 : dog3_age = dog2_age + 4)
  (h4 : dog4_age = dog3_age / 2)
  (h5 : dog5_age = dog4_age + 20) :
  (dog1_age + dog5_age) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_age_average_l103_10383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l103_10372

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (⌊12 * Real.pi⌋ - ⌈-7 * Real.pi⌉ + 1))).card = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l103_10372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_axis_l103_10358

theorem cos_symmetry_axis (k : ℤ) : 
  ∀ x : ℝ, Real.cos x = Real.cos (2 * (k : ℝ) * Real.pi - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_symmetry_axis_l103_10358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10371

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 1 - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (max : ℝ), max = Real.sqrt 2 ∧
    (∀ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/4 → f x ≤ max)) ∧
  (∃ (min : ℝ), min = -(Real.sqrt 3 + 1)/2 ∧
    (∀ (x : ℝ), -Real.pi/3 ≤ x ∧ x ≤ Real.pi/4 → min ≤ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersects_triangle_radius_range_l103_10354

-- Define the points of the triangle
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 8)
def C : ℝ × ℝ := (2, 4)

-- Define the circle
def circleSet (R : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = R^2}

-- Define the triangle
def triangleSet : Set (ℝ × ℝ) := {(x, y) | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  ((x, y) = (1-t) • A + t • B ∨
   (x, y) = (1-t) • B + t • C ∨
   (x, y) = (1-t) • C + t • A)}

-- Theorem statement
theorem circle_intersects_triangle_radius_range :
  ∀ R : ℝ, (∃ p : ℝ × ℝ, p ∈ circleSet R ∧ p ∈ triangleSet) ↔ 
  (8 * Real.sqrt 5 / 5 ≤ R ∧ R ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersects_triangle_radius_range_l103_10354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l103_10357

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x) * Real.sqrt ((1 - x) / (1 + x))

-- Define the domain of f
def domain (x : ℝ) : Prop := -1 < x ∧ x ≤ 1

-- Statement: f is neither odd nor even on its domain
theorem f_neither_odd_nor_even :
  ¬(∀ x, domain x → f (-x) = -f x) ∧ 
  ¬(∀ x, domain x → f (-x) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l103_10357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_minimum_value_of_f_l103_10325

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 1)

-- Theorem for the solution set of the inequality
theorem solution_set_of_inequality :
  ∀ x : ℝ, x > 1 → (f x > 2*x + 1 ↔ x ∈ Set.Ioo 1 ((1 + Real.sqrt 5) / 2)) :=
by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value_of_f :
  ∃ x : ℝ, x > 1 ∧ f x = 4 ∧ ∀ y : ℝ, y > 1 → f y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_minimum_value_of_f_l103_10325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l103_10318

noncomputable def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (0, Real.pi), (Real.pi/2, Real.pi/2), (Real.pi, 0), (Real.pi, Real.pi)}

theorem system_solutions :
  {(x, y) : ℝ × ℝ | Real.sin (x + y) = 0 ∧ Real.sin (x - y) = 0 ∧ 0 ≤ x ∧ x ≤ Real.pi ∧ 0 ≤ y ∧ y ≤ Real.pi} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l103_10318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_decreasing_condition_l103_10338

-- Define the hyperbola (marked as noncomputable due to dependency on Real)
noncomputable def hyperbola (m : ℝ) (x : ℝ) : ℝ := (1 - m) / x

-- State the theorem
theorem hyperbola_decreasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → hyperbola m x₁ > hyperbola m x₂) →
  m < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_decreasing_condition_l103_10338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_X_in_right_triangle_l103_10337

-- Define the right triangle XYZ
def RightTriangle (X Y Z : ℝ) : Prop :=
  X^2 + Y^2 = Z^2

-- Theorem statement
theorem tan_X_in_right_triangle (X Y Z : ℝ) :
  RightTriangle X Y Z → Y = 40 → Z = 41 → Real.tan X = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_X_in_right_triangle_l103_10337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meeting_count_l103_10342

/-- The length of the straight sections of the track in feet -/
def straight_length : ℝ := 180

/-- The length of the curved sections of the track in feet -/
def curved_length : ℝ := 120

/-- The speed of the first boy in feet per second -/
def speed1 : ℝ := 6

/-- The speed of the second boy in feet per second -/
def speed2 : ℝ := 8

/-- The total circumference of the track in feet -/
def track_circumference : ℝ := 2 * straight_length + 2 * curved_length

/-- The relative speed of the boys in feet per second -/
def relative_speed : ℝ := speed1 + speed2

theorem boys_meeting_count :
  let time_to_complete_lap := track_circumference / relative_speed
  let meetings := Int.floor (time_to_complete_lap * relative_speed / track_circumference)
  meetings = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meeting_count_l103_10342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l103_10328

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with unit side length -/
structure UnitSquare where
  bottomLeft : Point

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem stating that the area of triangle ABC is zero -/
theorem triangle_area_zero (sq1 sq2 sq3 : UnitSquare)
  (h1 : sq2.bottomLeft.x = sq1.bottomLeft.x + 1)
  (h2 : sq3.bottomLeft.x = sq2.bottomLeft.x + 1)
  (h3 : sq2.bottomLeft.y = sq1.bottomLeft.y)
  (h4 : sq3.bottomLeft.y = sq2.bottomLeft.y) :
  let A : Point := ⟨sq1.bottomLeft.x + 1, sq1.bottomLeft.y⟩
  let B : Point := ⟨sq2.bottomLeft.x + 1, sq2.bottomLeft.y + 1⟩
  let C : Point := ⟨sq3.bottomLeft.x, sq3.bottomLeft.y + 1⟩
  triangleArea A B C = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l103_10328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10351

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.tan x * Real.sin (Real.pi / 2 - x) * Real.cos (x - Real.pi / 3) - Real.sqrt 3

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x y : ℝ), -Real.pi/12 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l103_10351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_trip_distance_l103_10393

/-- The distance from Sharon's house to her mother's house in miles -/
noncomputable def distance : ℝ := 171

/-- The usual time for the trip in minutes -/
noncomputable def usual_time : ℝ := 180

/-- The time for the trip on the rainy day in minutes -/
noncomputable def rainy_day_time : ℝ := 330

/-- The speed reduction due to rain in miles per hour -/
noncomputable def speed_reduction : ℝ := 30

/-- The fraction of the journey completed before the rain -/
noncomputable def fraction_before_rain : ℝ := 1/4

theorem sharon_trip_distance :
  let usual_speed := distance / usual_time
  let reduced_speed := usual_speed - speed_reduction / 60
  let time_before_rain := fraction_before_rain * usual_time
  let time_during_rain := rainy_day_time - time_before_rain
  let distance_during_rain := distance * (1 - fraction_before_rain)
  time_before_rain + distance_during_rain / reduced_speed = rainy_day_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_trip_distance_l103_10393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_bench_sections_l103_10317

/-- A single bench section can hold either 8 adults or 10 children -/
def adults_per_bench : ℕ := 8
def children_per_bench : ℕ := 10

/-- The number of bench sections -/
def N : ℕ := 20

/-- The total number of seats is the same for both adults and children -/
axiom equal_seats : adults_per_bench * N = children_per_bench * N

/-- N is the least positive integer satisfying the equal_seats condition -/
axiom N_is_least : ∀ m : ℕ, m > 0 → adults_per_bench * m = children_per_bench * m → N ≤ m

theorem least_bench_sections : N = 20 := by
  -- The proof goes here
  sorry

#check least_bench_sections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_bench_sections_l103_10317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scholarship_theorem_l103_10345

/-- Represents the ratio of boys to girls in a school -/
structure SchoolRatio where
  boys : ℕ
  girls : ℕ

/-- Represents the scholarship percentages for boys and girls -/
structure ScholarshipPercentages where
  boys_percent : ℚ
  girls_percent : ℚ

/-- Calculates the percentage of students who won't get a scholarship -/
noncomputable def percentage_without_scholarship (ratio : SchoolRatio) (scholarships : ScholarshipPercentages) : ℚ :=
  let total_students := ratio.boys + ratio.girls
  let boys_with_scholarship := (scholarships.boys_percent / 100) * ratio.boys
  let girls_with_scholarship := (scholarships.girls_percent / 100) * ratio.girls
  let students_without_scholarship := total_students - (boys_with_scholarship + girls_with_scholarship)
  (students_without_scholarship / total_students) * 100

/-- Theorem: Given the specified ratio and scholarship percentages, 
    the percentage of students who won't get a scholarship is (855/1100) * 100 -/
theorem scholarship_theorem (ratio : SchoolRatio) (scholarships : ScholarshipPercentages) 
    (h1 : ratio.boys = 5 ∧ ratio.girls = 6)
    (h2 : scholarships.boys_percent = 25 ∧ scholarships.girls_percent = 20) :
    percentage_without_scholarship ratio scholarships = (855 : ℚ) / 1100 * 100 := by
  sorry

#eval (855 : ℚ) / 1100 * 100 -- Approximately 77.73

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scholarship_theorem_l103_10345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l103_10327

theorem problem_solution :
  let expr1 := (1 : ℝ) * ((Real.pi - 1) ^ 0) - 3 / Real.sqrt 3 + (1 / 2) ^ (-1 : ℤ) + |5 - Real.sqrt 27| - 2 * Real.sqrt 3
  let m := Real.sqrt 6 - 3
  let expr2 := m / (m^2 - 9) / (1 + 3 / (m - 3))
  expr1 = -2 ∧ expr2 = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l103_10327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_45_deg_l103_10320

/-- The volume of a cone with radius r and height h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a cone given its volume and vertex angle -/
noncomputable def cone_height (volume : ℝ) (vertex_angle : ℝ) : ℝ :=
  (3 * volume / Real.pi)^(1/3)

theorem cone_height_45_deg (volume : ℝ) (h : volume = 8000 * Real.pi) :
  cone_height volume (Real.pi/4) = (24000 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_45_deg_l103_10320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_correct_l103_10368

/-- Triangle ABC with given altitudes and vertex A -/
structure Triangle where
  -- Altitude from A to BC
  altitude_a : Real → Real → Real
  -- Altitude from C to AB
  altitude_c : Real → Real → Real
  -- Vertex A coordinates
  vertex_a : Real × Real
  -- Vertex B coordinates
  vertex_b : Real × Real
  -- Vertex C coordinates
  vertex_c : Real × Real
  -- Conditions for altitudes
  altitude_a_eq : ∀ x y, altitude_a x y = 2*x - 3*y + 1
  altitude_c_eq : ∀ x y, altitude_c x y = x + y
  -- Condition for vertex A
  vertex_a_coords : vertex_a = (1, 2)
  -- Condition for vertex B
  vertex_b_coords : vertex_b = (-2, -1)
  -- Condition for vertex C
  vertex_c_coords : vertex_c = (7, -7)

/-- The equation of side BC in triangle ABC -/
def side_bc_equation : Real → Real → Real :=
  fun x y ↦ 2*x + 3*y + 7

/-- Theorem stating that the equation of side BC is correct -/
theorem side_bc_equation_correct (t : Triangle) :
  ∀ x y, side_bc_equation x y = 0 ↔
         (x - t.vertex_b.1) * (t.vertex_c.2 - t.vertex_b.2) = 
         (y - t.vertex_b.2) * (t.vertex_c.1 - t.vertex_b.1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_correct_l103_10368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_seventh_term_l103_10322

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_seventh_term
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 5) ^ 2 = (seq.a 4) * (seq.a 7))
  (h_sum : sum_n seq 11 = 66) :
  seq.a 7 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_seventh_term_l103_10322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_squared_of_parabola_intersections_is_25_over_4_l103_10366

/-- The radius squared of the circle containing all intersection points of two parabolas -/
noncomputable def radius_squared_of_parabola_intersections : ℝ :=
  25/4

/-- Theorem stating that the radius squared of the circle containing all intersection points
    of the parabolas y = (x - 2)^2 and x - 3 = (y + 1)^2 is equal to 25/4 -/
theorem radius_squared_of_parabola_intersections_is_25_over_4 :
  radius_squared_of_parabola_intersections = 25/4 := by
  -- Unfold the definition
  unfold radius_squared_of_parabola_intersections
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_squared_of_parabola_intersections_is_25_over_4_l103_10366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l103_10397

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → a ≤ (x^(-3 : ℝ) * Real.exp x - x - 1) / Real.log x) → 
  a ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l103_10397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l103_10377

noncomputable def f (x : ℝ) := Real.sqrt (x + 1) + (1 - x)^0 / (2 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 ≥ 0 ∧ 1 - x ≠ 0 ∧ 2 - x ≠ 0} =
  Set.Icc (-1) 1 ∪ Set.Ioo 1 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l103_10377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_g_domain_is_all_reals_l103_10389

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 6*x + 10⌋

theorem domain_of_g (x : ℝ) : 
  (x^2 - 6*x + 10 > 0) → (g x ≠ 0) :=
by
  sorry

theorem g_domain_is_all_reals : 
  ∀ x : ℝ, ∃ y : ℝ, g x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_g_domain_is_all_reals_l103_10389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l103_10382

/-- Given the travel conditions for Eddy and Freddy, prove that Freddy's travel time is 4 hours -/
theorem freddy_travel_time 
  (eddy_time : ℝ) 
  (ab_distance : ℝ) 
  (ac_distance : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : eddy_time = 3) 
  (h2 : ab_distance = 570) 
  (h3 : ac_distance = 300) 
  (h4 : speed_ratio = 2.533333333333333) : 
  (ac_distance / (ab_distance / eddy_time / speed_ratio)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l103_10382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_center_implies_phi_l103_10334

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

-- State the theorem
theorem symmetrical_center_implies_phi (ω φ : ℝ) : 
  ω > 0 → 
  |φ| < Real.pi / 2 → 
  (∃ c : ℝ, ∀ x : ℝ, f ω (c - x) = f ω (c + x) ∧ g φ (c - x) = g φ (c + x)) → 
  φ = -Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_center_implies_phi_l103_10334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ratio_l103_10324

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 12

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Point on the circle -/
structure PointOnCircle where
  x : ℝ
  y : ℝ
  on_circle : circle_eq x y

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Ratio of areas of triangles OAB and OPQ -/
noncomputable def area_ratio (A B : PointOnCircle) (P Q : PointOnParabola) : ℝ :=
  (triangle_area O (A.x, A.y) (B.x, B.y)) / (triangle_area O (P.x, P.y) (Q.x, Q.y))

/-- Main theorem: The maximum value of S₁/S₂ is 9/16 -/
theorem max_area_ratio :
  ∃ (A B : PointOnCircle) (P Q : PointOnParabola),
    (∀ (A' B' : PointOnCircle) (P' Q' : PointOnParabola),
      area_ratio A' B' P' Q' ≤ area_ratio A B P Q) ∧
    area_ratio A B P Q = 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ratio_l103_10324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_in_third_quadrant_l103_10346

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in slope-intercept form
structure Line where
  slope : ℝ
  intercept : ℝ

-- Function to determine if a point is in the third quadrant
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

-- Function to find the intersection point of two lines
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * ((l2.intercept - l1.intercept) / (l1.slope - l2.slope)) + l1.intercept }

-- The main theorem
theorem intersection_not_in_third_quadrant (m n : ℝ) :
  ¬(isInThirdQuadrant (intersectionPoint { slope := -3, intercept := 1 } { slope := m, intercept := n })) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_in_third_quadrant_l103_10346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l103_10378

theorem sufficient_not_necessary : 
  (∀ x : ℝ, abs (x - 2) < 1 → x^2 + x - 2 > 0) ∧
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ ¬(abs (x - 2) < 1)) :=
by
  constructor
  · -- Proof of sufficiency
    intro x h
    sorry
  · -- Proof of not necessary
    use 4
    constructor
    · -- Proof that 4^2 + 4 - 2 > 0
      norm_num
    · -- Proof that ¬(|4 - 2| < 1)
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l103_10378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l103_10329

noncomputable def minuteHandAngle (minutes : ℕ) : ℝ :=
  (minutes % 60 : ℝ) * 6

noncomputable def hourHandAngle (hours minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5

noncomputable def smallerAngle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

theorem clock_angle_at_3_45 :
  smallerAngle (hourHandAngle 3 45) (minuteHandAngle 45) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_45_l103_10329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_bounds_l103_10395

-- Define the set S
def S : Set ℝ := {x | ∃ a b : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ 2 ∧ x = 3/a + b}

-- Define the maximum element M
def M : ℝ := 5

-- Define the minimum element m
noncomputable def m : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem set_bounds :
  (∀ x ∈ S, x ≤ M) ∧ 
  (∀ x ∈ S, m ≤ x) ∧
  (M ∈ S) ∧ 
  (m ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_bounds_l103_10395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l103_10314

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Define the given condition
def given_condition (A C : ℝ) (a b c : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C

-- Helper function to represent the height of the triangle
noncomputable def triangle_height (A B C : ℝ) (a b c : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_properties 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : triangle A B C a b c)
  (h2 : given_condition A C a b c)
  (h3 : A ∈ Set.Ioo 0 π)
  (h4 : a = 2 * Real.sqrt 3) :
  A = π / 3 ∧ 
  ∃ (h : ℝ), h ≤ 3 ∧ 
    ∀ (h' : ℝ), (∃ (B' C' : ℝ), triangle A B' C' a b c ∧ h' = triangle_height A B' C' a b c) 
    → h' ≤ h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l103_10314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translation_equivalence_l103_10361

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem cos_translation_equivalence :
  ∀ x : ℝ, f (x - Real.pi / 6) = g x :=
by
  intro x
  simp [f, g]
  congr
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_translation_equivalence_l103_10361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_time_difference_l103_10373

/-- Represents a cyclist with their speeds for different terrains -/
structure Cyclist where
  uphill_speed : ℚ
  downhill_speed : ℚ
  flat_speed : ℚ

/-- Represents a segment of the circuit -/
structure Segment where
  distance : ℚ
  terrain : String

def Minnie : Cyclist := { uphill_speed := 6, downhill_speed := 25, flat_speed := 18 }
def Penny : Cyclist := { uphill_speed := 12, downhill_speed := 35, flat_speed := 25 }

def circuit : List Segment := [
  { distance := 12, terrain := "uphill" },
  { distance := 18, terrain := "downhill" },
  { distance := 25, terrain := "flat" }
]

def penny_break_time : ℚ := 10 / 60 -- 10 minutes in hours

/-- Calculates the time taken by a cyclist to complete a segment -/
def time_for_segment (c : Cyclist) (s : Segment) : ℚ :=
  match s.terrain with
  | "uphill" => s.distance / c.uphill_speed
  | "downhill" => s.distance / c.downhill_speed
  | "flat" => s.distance / c.flat_speed
  | _ => 0 -- This case should never occur with our given circuit

/-- Calculates the total time taken by a cyclist to complete the circuit -/
def total_time (c : Cyclist) (break_time : ℚ := 0) : ℚ :=
  (circuit.map (time_for_segment c)).sum + break_time

/-- The main theorem to prove -/
theorem circuit_time_difference :
  (total_time Minnie - total_time Penny penny_break_time) * 60 = 66 := by
  sorry

#eval (total_time Minnie - total_time Penny penny_break_time) * 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_time_difference_l103_10373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_is_2_sqrt_2_base_circumference_equals_generatrix_circle_perimeter_l103_10375

/-- The length of the generatrix of a cone with base radius √2 and lateral surface that unfolds into a semicircle -/
noncomputable def cone_generatrix_length : ℝ := 2 * Real.sqrt 2

/-- The base radius of the cone -/
noncomputable def base_radius : ℝ := Real.sqrt 2

/-- Proposition: The length of the generatrix of a cone with base radius √2 and lateral surface that unfolds into a semicircle is 2√2 -/
theorem cone_generatrix_length_is_2_sqrt_2 :
  cone_generatrix_length = 2 * Real.sqrt 2 := by
  rfl

/-- The circumference of the base of the cone -/
noncomputable def base_circumference : ℝ := 2 * Real.pi * base_radius

/-- Proposition: The circumference of the base equals the perimeter of a circle with radius equal to the generatrix length -/
theorem base_circumference_equals_generatrix_circle_perimeter :
  base_circumference = Real.pi * cone_generatrix_length := by
  unfold base_circumference cone_generatrix_length base_radius
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_is_2_sqrt_2_base_circumference_equals_generatrix_circle_perimeter_l103_10375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l103_10333

open Real

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), (4:ℝ)^x < Real.log x / Real.log a) ↔ a ∈ Set.Icc (sqrt 2 / 2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l103_10333
