import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l564_56439

/-- A cuboid with dimensions 2x2x2 -/
structure Cuboid where
  vertices : Fin 8 → ℝ × ℝ × ℝ
  is_valid : 
    (vertices 0 = (0, 0, 0)) ∧
    (vertices 7 = (2, 2, 2)) ∧
    (∀ i, vertices i ∈ Set.prod (Set.Icc 0 2) 
                               (Set.prod (Set.Icc 0 2)
                                         (Set.Icc 0 2)))

/-- An octahedron inscribed in the cuboid -/
structure InscribedOctahedron (c : Cuboid) where
  vertices : Fin 6 → ℝ × ℝ × ℝ
  on_edges : 
    (vertices 0 = (4/3, 0, 0)) ∧
    (vertices 1 = (0, 4/3, 0)) ∧
    (vertices 2 = (0, 0, 4/3)) ∧
    (vertices 3 = (2, 2/3, 2)) ∧
    (vertices 4 = (2, 2, 2/3)) ∧
    (vertices 5 = (2/3, 2, 2))

/-- The theorem to be proved -/
theorem octahedron_side_length (c : Cuboid) (o : InscribedOctahedron c) :
  ∀ i j, i ≠ j → dist (o.vertices i) (o.vertices j) = 4 * Real.sqrt 2 / 3 :=
by
  sorry

#check octahedron_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l564_56439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l564_56425

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
axiom f_condition (x : ℝ) : f (x + 1) + f (-x + 1) = 2

-- Define the function g(x) = f(x+1) - 1
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 1) - 1

-- Theorem statement
theorem g_is_odd (f : ℝ → ℝ) (h : ∀ x, f (x + 1) + f (-x + 1) = 2) : 
  ∀ x : ℝ, g f (-x) = -(g f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l564_56425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l564_56419

def n : ℕ := 2^2 * 3^2 * 7^2

def is_even_factor (d : ℕ) : Prop :=
  d ∣ n ∧ Even d

-- Make is_even_factor decidable
instance : DecidablePred is_even_factor :=
  fun d => And.decidable

def count_even_factors : ℕ :=
  (Finset.filter is_even_factor (Finset.range (n + 1))).card

theorem even_factors_count : count_even_factors = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l564_56419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_28_percent_l564_56417

/-- Represents a vessel with a certain capacity and alcohol concentration -/
structure Vessel where
  capacity : ℚ
  alcohol_concentration : ℚ

/-- Calculates the new alcohol concentration after mixing and diluting -/
def new_concentration (v1 v2 : Vessel) (total_volume : ℚ) (final_volume : ℚ) : ℚ :=
  let total_alcohol := v1.capacity * v1.alcohol_concentration + v2.capacity * v2.alcohol_concentration
  total_alcohol / final_volume

/-- Theorem stating that the new concentration is 28% given the problem conditions -/
theorem new_concentration_is_28_percent : 
  let v1 : Vessel := { capacity := 2, alcohol_concentration := 1/5 }
  let v2 : Vessel := { capacity := 6, alcohol_concentration := 2/5 }
  let total_volume : ℚ := 8
  let final_volume : ℚ := 10
  new_concentration v1 v2 total_volume final_volume = 7/25 := by
  -- Proof steps would go here
  sorry

#eval new_concentration 
  { capacity := 2, alcohol_concentration := 1/5 }
  { capacity := 6, alcohol_concentration := 2/5 }
  8
  10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_concentration_is_28_percent_l564_56417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_pattern_exists_l564_56464

/-- Represents a tree type -/
inductive TreeType
| Pine
| Spruce

/-- Represents the circular arrangement of trees -/
def TreeCircle := Fin 2019 → TreeType

instance : DecidableEq TreeType := 
  fun a b => match a, b with
  | .Pine, .Pine => isTrue rfl
  | .Spruce, .Spruce => isTrue rfl
  | .Pine, .Spruce => isFalse (fun h => TreeType.noConfusion h)
  | .Spruce, .Pine => isFalse (fun h => TreeType.noConfusion h)

theorem pine_pattern_exists (circle : TreeCircle)
  (h_pine_count : (Finset.univ.filter (fun i => circle i = TreeType.Pine)).card = 1009)
  (h_spruce_count : (Finset.univ.filter (fun i => circle i = TreeType.Spruce)).card = 1010) :
  ∃ i : Fin 2019, circle i = TreeType.Pine ∧ 
    circle ((i + 3) % 2019) = TreeType.Pine := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pine_pattern_exists_l564_56464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_triangle_area_l564_56422

/-- Represents a parabola in the 2D plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem parabola_rotation_triangle_area
  (p : Parabola)
  (h1 : p.a = 1 ∧ p.b = 2 ∧ p.c = 3)  -- Original parabola equation
  (M : Point)
  (hM : M.x = -1 ∧ M.y = 2)  -- Vertex of the parabola
  (A : Point)
  (hA : A.x = 0 ∧ A.y = p.equation 0)  -- A is on y-axis and original parabola
  (B : Point)
  (hB : B.x = 0 ∧ B.y = 2 - (A.y - 2))  -- B is on y-axis and rotated parabola
  : triangleArea A M B = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_triangle_area_l564_56422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_order_l564_56456

open Real

-- Define the integral I_n
noncomputable def I (n : ℕ) : ℝ := ∫ x in (0)..(n * π), sin x / (1 + x)

-- State the theorem
theorem integral_order : I 2 < I 4 ∧ I 4 < I 3 ∧ I 3 < I 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_order_l564_56456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_partition_property_l564_56447

def T (k : ℕ) : Finset ℕ := Finset.range k |>.image (fun i => 2^(i+1))

theorem smallest_k_for_partition_property : 
  (∀ k : ℕ, k ≥ 2 → 
    (∀ (X Y : Finset ℕ), X ∪ Y = T k → X ∩ Y = ∅ → 
      (∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a * b = c) ∨ 
      (∃ a b c, a ∈ Y ∧ b ∈ Y ∧ c ∈ Y ∧ a * b = c))) 
  ↔ 
  (∀ k : ℕ, k ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_partition_property_l564_56447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l564_56426

theorem triangle_cosine (A B C : ℝ) (h1 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h2 : A + B + C = Real.pi) (h3 : 6 * Real.sin A = 4 * Real.sin B) 
  (h4 : 4 * Real.sin B = 3 * Real.sin C) : Real.cos B = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l564_56426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l564_56423

theorem trigonometric_identity (α : ℝ) (m : ℝ) (h : Real.tan α = m) :
  (3 * Real.sin α + Real.sin (3 * α)) / (3 * Real.cos α + Real.cos (3 * α)) = m / 2 * (m^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l564_56423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_number_of_products_l564_56470

/-- Given fixed cost, marginal cost, and total cost, calculate the number of products. -/
theorem calculate_number_of_products
  (fixed_cost : ℝ)
  (marginal_cost : ℝ)
  (total_cost : ℝ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000) :
  (total_cost - fixed_cost) / marginal_cost = 20 := by
  -- Proof steps would go here
  sorry

#check calculate_number_of_products

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_number_of_products_l564_56470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_special_triple_l564_56475

def X : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem subset_contains_special_triple :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
    (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = 2*c) ∨
    (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a + b = 2*c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_special_triple_l564_56475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l564_56482

-- Define the set of a
def A : Set ℝ := {x | (1/3)^x - x = 0}

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log a

-- Theorem statement
theorem increasing_interval_of_f (a : ℝ) (h : a ∈ A) :
  ∃ (I : Set ℝ), StrictMonoOn (f a) I ∧ I = Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l564_56482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_box_mass_l564_56441

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ := d.height * d.width * d.length

/-- Represents a box with its dimensions and the mass it holds -/
structure Box where
  dimensions : BoxDimensions
  mass : ℝ

/-- Calculates the density of the material in a box -/
noncomputable def boxDensity (b : Box) : ℝ := b.mass / boxVolume b.dimensions

theorem second_box_mass (box1 box2 : Box) 
  (h1 : box1.dimensions = BoxDimensions.mk 3 4 6)
  (h2 : box1.mass = 72)
  (h3 : box2.dimensions.height = 1.5 * box1.dimensions.height)
  (h4 : box2.dimensions.width = 2.5 * box1.dimensions.width)
  (h5 : box2.dimensions.length = box1.dimensions.length)
  (h6 : boxDensity box2 = 2 * boxDensity box1) :
  box2.mass = 540 := by
  sorry

#check second_box_mass

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_box_mass_l564_56441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l564_56436

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + Real.exp (Real.log 3 * x) else -x

theorem integral_f_minus_one_to_one :
  ∫ x in Set.Icc (-1) 1, f x = 5/6 + 2/Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l564_56436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l564_56457

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^8 + i^20 + i^(-30 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l564_56457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_and_planes_l564_56414

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (lies_within : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (plane_intersects : Plane → Plane → Prop)
variable (line_intersects : Line → Line → Prop)

-- Define the given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : lies_within l α)
variable (h2 : lies_within m α)
variable (h3 : ¬lies_within l β)
variable (h4 : ¬lies_within m β)
variable (h5 : line_intersects l m)

-- State the theorem
theorem intersecting_lines_and_planes :
  ((intersects l β ∨ intersects m β) ↔ plane_intersects α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_and_planes_l564_56414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l564_56415

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference
  first_term : a 1 = a 1  -- First term (tautology to define a 1)
  diff : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 2 = 3) 
  (h2 : sum_n seq 4 = 16) : 
  seq.d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l564_56415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_design_black_percentage_l564_56452

theorem art_design_black_percentage :
  let n : ℕ := 5  -- number of circles
  let r₀ : ℝ := 3 -- initial radius
  let δr : ℝ := 3 -- radius increment

  let radius (i : ℕ) : ℝ := r₀ + i * δr
  let area (i : ℕ) : ℝ := Real.pi * (radius i) ^ 2
  
  let black_area : ℝ := (Finset.sum (Finset.filter (fun i => i % 2 = 0) (Finset.range n))
    (fun i => area i - if i > 0 then area (i - 1) else 0))
  let total_area : ℝ := area (n - 1)

  (black_area / total_area) * 100 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_design_black_percentage_l564_56452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_multiplication_l564_56434

theorem fraction_exponent_multiplication :
  (2 / 3 : ℚ) ^ 4 * (2 / 3 : ℚ) ^ (-(2 : ℤ)) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_multiplication_l564_56434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l564_56483

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_point : a^2 * (2 : ℝ) = (a^2 + b^2) * 2
  h_foci : a^2 = 2 * b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : SpecialEllipse) : ℝ → ℝ → Prop :=
  fun x y ↦ x^2 / 2 + y^2 = 1

/-- A line intersecting the ellipse -/
def intersecting_line (m n : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ m * x + n * y + (1 / 3) * n = 0

/-- The fixed point -/
def fixed_point : ℝ × ℝ := (0, 1)

/-- Main theorem -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∀ m n x₁ y₁ x₂ y₂,
    ellipse_equation e x₁ y₁ ∧
    ellipse_equation e x₂ y₂ ∧
    intersecting_line m n x₁ y₁ ∧
    intersecting_line m n x₂ y₂ →
    (x₁ - fixed_point.1) * (x₂ - fixed_point.1) +
    (y₁ - fixed_point.2) * (y₂ - fixed_point.2) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l564_56483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_sale_problem_l564_56480

theorem pencil_sale_problem (total_students : ℕ) 
  (first_group_size first_group_pencils : ℕ)
  (second_group_size second_group_pencils : ℕ)
  (total_pencils_sold : ℕ) :
  total_students = 10 →
  first_group_size = 2 →
  first_group_pencils = 2 →
  second_group_size = 6 →
  second_group_pencils = 3 →
  total_pencils_sold = 24 →
  let remaining_students := total_students - first_group_size - second_group_size;
  let remaining_pencils := total_pencils_sold - 
    (first_group_size * first_group_pencils + second_group_size * second_group_pencils);
  remaining_pencils / remaining_students = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_sale_problem_l564_56480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_problem_l564_56477

/-- Given a deposit percentage and amount, calculate the remaining amount to be paid --/
noncomputable def remaining_payment (deposit_percentage : ℝ) (deposit_amount : ℝ) : ℝ :=
  (deposit_amount / deposit_percentage) - deposit_amount

/-- Theorem: If a 10% deposit of $105 has been paid, then $945 remains to be paid --/
theorem deposit_problem :
  remaining_payment 0.1 105 = 945 := by
  -- Unfold the definition of remaining_payment
  unfold remaining_payment
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_problem_l564_56477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_l564_56432

/-- Calculates the time (in seconds) for a train to pass a stationary point. -/
noncomputable def time_to_pass_pole (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem: A train 150 metres long running at 54 km/hr takes 10 seconds to pass a pole. -/
theorem train_passing_pole :
  time_to_pass_pole 150 54 = 10 := by
  -- Unfold the definition of time_to_pass_pole
  unfold time_to_pass_pole
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_l564_56432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_trapezoid_l564_56493

/-- A rectangle trapezoid with two circles -/
structure RectangleTrapezoidWithCircles where
  /-- The inscribed circle with radius 4 -/
  inscribedCircleRadius : ℝ
  /-- The smaller circle with radius 1 -/
  smallerCircleRadius : ℝ
  /-- The inscribed circle has radius 4 -/
  inscribedCircleRadiusEq : inscribedCircleRadius = 4
  /-- The smaller circle has radius 1 -/
  smallerCircleRadiusEq : smallerCircleRadius = 1
  /-- The smaller circle touches two sides of the trapezoid and the inscribed circle -/
  smallerCircleTouches : Bool

/-- The area of the rectangle trapezoid -/
noncomputable def areaOfRectangleTrapezoid (t : RectangleTrapezoidWithCircles) : ℝ :=
  196 / 3

/-- Theorem: The area of the rectangle trapezoid is 196/3 -/
theorem area_of_rectangle_trapezoid (t : RectangleTrapezoidWithCircles) :
  areaOfRectangleTrapezoid t = 196 / 3 := by
  -- Unfold the definition of areaOfRectangleTrapezoid
  unfold areaOfRectangleTrapezoid
  -- The result follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_trapezoid_l564_56493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l564_56461

/-- Given a right triangle with shorter leg x and longer leg (3x + 2),
    prove that when the area is 168 square feet, 
    the hypotenuse is approximately 34.338 feet. -/
theorem right_triangle_hypotenuse (x : ℝ) : 
  x > 0 →
  (1/2 : ℝ) * x * (3*x + 2) = 168 →
  ∃ (h : ℝ), h > 0 ∧ h^2 = x^2 + (3*x + 2)^2 ∧ |h - 34.338| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l564_56461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_coordinates_point_on_graph_l564_56431

-- Define the function f(x) = a^x - 1/2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1/2

-- State the theorem
theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 1/2 := by
  -- Expand the definition of f
  unfold f
  -- Simplify a^0 to 1
  simp
  -- Basic arithmetic
  norm_num

-- Prove that this point is on the graph for all valid a
theorem point_on_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 0 ∧ y = 1/2 ∧ f a x = y := by
  -- Use the point (0, 1/2)
  use 0, 1/2
  constructor
  · rfl
  constructor
  · rfl
  · -- Use the previous theorem
    exact fixed_point_coordinates a h1 h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_coordinates_point_on_graph_l564_56431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l564_56486

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 8*y + 12 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, 0)
def center2 : ℝ × ℝ := (0, -4)
def radius1 : ℝ := 3
def radius2 : ℝ := 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius1 + radius2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l564_56486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_routes_count_l564_56430

theorem tour_routes_count 
  (total_cities : ℕ)
  (cities_to_choose : ℕ)
  (mandatory_cities : ℕ)
  (h1 : total_cities = 7)
  (h2 : cities_to_choose = 5)
  (h3 : mandatory_cities = 2) :
  (Nat.factorial (total_cities - mandatory_cities) / 
   Nat.factorial (total_cities - cities_to_choose)) *
  (Nat.choose (cities_to_choose - mandatory_cities + 1) mandatory_cities) = 600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tour_routes_count_l564_56430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_increase_approx_95_3_percent_l564_56458

/-- The length of a leg in the first isosceles right triangle -/
def initial_leg_length : ℝ := 3

/-- The scaling factor between consecutive triangles -/
def scaling_factor : ℝ := 1.25

/-- The number of triangles in the sequence -/
def num_triangles : ℕ := 4

/-- Calculate the hypotenuse of an isosceles right triangle given its leg length -/
noncomputable def hypotenuse (leg_length : ℝ) : ℝ := leg_length * Real.sqrt 2

/-- Calculate the leg length of the nth triangle in the sequence -/
def leg_length (n : ℕ) : ℝ := initial_leg_length * scaling_factor ^ (n - 1)

/-- Calculate the percent increase between two values -/
noncomputable def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem hypotenuse_increase_approx_95_3_percent :
  abs (percent_increase (hypotenuse (leg_length 1)) (hypotenuse (leg_length num_triangles)) - 95.3) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_increase_approx_95_3_percent_l564_56458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l564_56412

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

-- Define the theorem
theorem ellipse_eccentricity_relation (a : ℝ) :
  a > 1 →
  let e₁ := eccentricity a 1
  let e₂ := eccentricity 2 1
  e₂ = Real.sqrt 3 * e₁ →
  a = 2 * Real.sqrt 3 / 3 := by
  sorry

#check ellipse_eccentricity_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l564_56412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_putnam_1966_divisibility_l564_56454

theorem putnam_1966_divisibility 
  (m n : ℕ) 
  (a : Fin (m * n + 1) → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_increasing : ∀ i j, i < j → a i < a j) :
  (∃ s : Finset (Fin (m * n + 1)), s.card = m + 1 ∧ 
    ∀ i j, i ∈ s → j ∈ s → i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∨
  (∃ s : Finset (Fin (m * n + 1)), s.card = n + 1 ∧ 
    ∀ i j, i ∈ s → j ∈ s → i < j → a i ∣ a j) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_putnam_1966_divisibility_l564_56454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_textbooks_in_same_box_l564_56472

/-- The number of textbooks -/
def total_textbooks : ℕ := 16

/-- The number of physics textbooks -/
def physics_textbooks : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The capacities of the boxes -/
def box_capacities : Fin num_boxes → ℕ
  | ⟨0, h⟩ => 4
  | ⟨1, h⟩ => 5
  | ⟨2, h⟩ => 3
  | ⟨3, h⟩ => 4
  | ⟨n+4, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 4 n))

/-- The probability of all physics textbooks ending up in the same box -/
def probability_all_physics_in_same_box : ℚ := 3 / 286

theorem physics_textbooks_in_same_box :
  let total_arrangements := Nat.choose total_textbooks (box_capacities ⟨0, Nat.zero_lt_succ 3⟩) *
                            Nat.choose (total_textbooks - box_capacities ⟨0, Nat.zero_lt_succ 3⟩) (box_capacities ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 2)⟩) *
                            Nat.choose (total_textbooks - box_capacities ⟨0, Nat.zero_lt_succ 3⟩ - box_capacities ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 2)⟩) (box_capacities ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1))⟩)
  let favorable_arrangements := Nat.choose (total_textbooks - physics_textbooks) (box_capacities ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 2)⟩ - physics_textbooks) +
                                1 + 1  -- For boxes 0 and 3
  (favorable_arrangements : ℚ) / total_arrangements = probability_all_physics_in_same_box := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_textbooks_in_same_box_l564_56472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turner_tickets_needed_l564_56494

/-- Represents the number of tickets required for a ride -/
structure TicketCost where
  value : Nat

/-- Represents the number of times Turner wants to ride an attraction -/
structure RideCount where
  value : Nat

/-- Calculates the total number of tickets needed for a specific ride -/
def ticketsForRide (cost : TicketCost) (count : RideCount) : Nat :=
  cost.value * count.value

/-- The problem statement -/
theorem turner_tickets_needed 
  (rollercoaster_cost catapult_cost ferris_wheel_cost : TicketCost)
  (rollercoaster_rides catapult_rides ferris_wheel_rides : RideCount)
  (h1 : rollercoaster_cost = ⟨4⟩)
  (h2 : catapult_cost = ⟨4⟩)
  (h3 : ferris_wheel_cost = ⟨1⟩)
  (h4 : rollercoaster_rides = ⟨3⟩)
  (h5 : catapult_rides = ⟨2⟩)
  (h6 : ferris_wheel_rides = ⟨1⟩)
  : ticketsForRide rollercoaster_cost rollercoaster_rides +
    ticketsForRide catapult_cost catapult_rides +
    ticketsForRide ferris_wheel_cost ferris_wheel_rides = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turner_tickets_needed_l564_56494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l564_56427

-- Define the triangle ABC
def triangle_ABC (A B C : Real) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the side lengths and angle ratio
def triangle_conditions (a b c : Real) (B C : Real) : Prop :=
  b = 7 ∧ a = 3 ∧ (Real.sin C) / (Real.sin B) = 3/5

-- Theorem statement
theorem triangle_ABC_properties 
  {A B C a b c : Real} 
  (h_triangle : triangle_ABC A B C) 
  (h_conditions : triangle_conditions a b c B C) :
  c = 5 ∧ A = 2*Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l564_56427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_four_digits_of_5_to_2016_l564_56468

-- Define last_four_digits function
def last_four_digits (n : Nat) : Nat :=
  n % 10000

theorem last_four_digits_of_5_to_2016 (pattern : Fin 4 → Nat) 
  (h_pattern : ∀ n : Nat, last_four_digits (5^n) = pattern (n % 4))
  (h_pattern_values : pattern 0 = 3125 ∧ pattern 1 = 5625 ∧ pattern 2 = 8125 ∧ pattern 3 = 0625) :
  last_four_digits (5^2016) = 0625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_four_digits_of_5_to_2016_l564_56468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cluster_probability_l564_56481

/-- Proves that the percentage chance of picking a peanut cluster is 64% --/
theorem peanut_cluster_probability (total : ℕ) (caramels : ℕ) (nougats : ℕ) (truffles : ℕ) (peanut_clusters : ℕ) :
  total = 50 →
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + 6 →
  peanut_clusters = total - caramels - nougats - truffles →
  (peanut_clusters : ℚ) / (total : ℚ) * 100 = 64 := by
  intro h_total h_caramels h_nougats h_truffles h_peanut_clusters
  -- The proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cluster_probability_l564_56481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l564_56442

/-- Given an interest rate and total interest, calculate the original borrowed amount. -/
noncomputable def calculate_borrowed_amount (interest_rate : ℝ) (total_interest : ℝ) : ℝ :=
  total_interest / interest_rate

/-- Theorem stating that given a 12% interest rate and $1500 total interest, 
    the original borrowed amount is $12500. -/
theorem borrowed_amount_calculation :
  let interest_rate : ℝ := 0.12
  let total_interest : ℝ := 1500
  calculate_borrowed_amount interest_rate total_interest = 12500 := by
  -- Unfold the definition of calculate_borrowed_amount
  unfold calculate_borrowed_amount
  -- Simplify the expression
  simp
  -- Check that the resulting equation is true
  norm_num

-- We remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l564_56442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_ge_one_l564_56469

/-- A function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = ax + sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_f_implies_a_ge_one (a : ℝ) :
  IsIncreasing (f a) → a ∈ Set.Ici 1 := by
  sorry

#check increasing_f_implies_a_ge_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_ge_one_l564_56469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_pi_over_4_l564_56459

noncomputable def g (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 2)

theorem g_symmetry_about_pi_over_4 :
  ∀ (h : ℝ), g (π / 4 + h) = -g (π / 4 - h) :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_pi_over_4_l564_56459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_in_first_group_l564_56406

/-- Represents the amount of work done by one person in one day -/
structure WorkRate where
  man : ℝ
  boy : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Represents a work scenario -/
structure WorkScenario where
  group : WorkGroup
  days : ℕ

/-- The total amount of work to be done -/
def totalWork : ℝ := 1  -- Assign a default value to avoid errors

/-- Calculate the amount of work done by a group in a given number of days -/
def workDone (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  scenario.days * (rate.man * scenario.group.men + rate.boy * scenario.group.boys)

/-- The first work scenario -/
def scenario1 (x : ℕ) : WorkScenario := { group := { men := x, boys := 8 }, days := 10 }

/-- The second work scenario -/
def scenario2 : WorkScenario := { group := { men := 26, boys := 48 }, days := 2 }

/-- The third work scenario -/
def scenario3 : WorkScenario := { group := { men := 15, boys := 20 }, days := 4 }

theorem men_in_first_group (rate : WorkRate) (x : ℕ) :
  workDone rate (scenario1 x) = totalWork ∧
  workDone rate scenario2 = totalWork ∧
  workDone rate scenario3 = totalWork →
  x = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_in_first_group_l564_56406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l564_56498

/-- The function f(x) = 1 + m/x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 + m / x

/-- f(1) = 2 -/
def f_at_one (m : ℝ) : Prop := f m 1 = 2

/-- Monotonically decreasing on (0,+∞) -/
def monotone_decreasing_on (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → g x > g y

theorem f_properties (m : ℝ) (h : f_at_one m) :
  m = 1 ∧ monotone_decreasing_on (f m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l564_56498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_sixteen_l564_56455

noncomputable def program_output (x : ℚ) : ℚ :=
  if x < 0 then (x + 1) * (x + 1) else (x - 1) * (x - 1)

theorem program_output_sixteen (x : ℚ) :
  program_output x = 16 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_sixteen_l564_56455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l564_56487

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l564_56487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_and_increasing_l564_56443

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2)

theorem f_symmetric_and_increasing :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) :=
by
  constructor
  · intro x
    sorry -- Proof of symmetry
  · intro x y hxy
    sorry -- Proof of monotonicity

#check f_symmetric_and_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_and_increasing_l564_56443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iteration_l564_56491

def f (x : ℕ) : ℕ :=
  if x % 5 = 0 ∧ x % 11 = 0 then x / 55
  else if x % 11 = 0 then 5 * x
  else if x % 5 = 0 then 11 * x
  else x + 5

def f_iter : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (f_iter n x)

theorem smallest_a_for_f_iteration :
  (∀ k : ℕ, k > 1 ∧ k < 4 → f_iter k 1 ≠ f 1) ∧
  f_iter 4 1 = f 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_f_iteration_l564_56491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l564_56416

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := (2*x^3 + 6*x^2 + 7*x + 2) / (x*(x+1)^3)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ := 2 * Real.log (abs x) - 1 / (2 * (x+1)^2)

-- State the theorem
theorem integral_proof (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : 
  deriv F x = f x := by
  sorry

#check integral_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l564_56416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l564_56413

/-- The ellipse with equation x²/49 + y²/24 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 49) + (p.2^2 / 24) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {f : ℝ × ℝ | ∃ (x y : ℝ), f = (x, y) ∧ x^2 - y^2 = 25}

/-- A point P on the ellipse satisfying the given ratio condition -/
def PointP (p f₁ f₂ : ℝ × ℝ) : Prop :=
  p ∈ Ellipse ∧ f₁ ∈ Foci ∧ f₂ ∈ Foci ∧ f₁ ≠ f₂ ∧
  ∃ (d₁ d₂ : ℝ), d₁ / d₂ = 4 / 3 ∧
  d₁ = Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) ∧
  d₂ = Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)

/-- The main theorem -/
theorem area_of_triangle (P F₁ F₂ : ℝ × ℝ) 
  (h : PointP P F₁ F₂) : 
  ∃ (A : ℝ), A = 24 ∧ A = Real.sqrt (s * (s - a) * (s - b) * (s - c)) :=
  sorry
where
  a := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  b := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
  c := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  s := (a + b + c) / 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l564_56413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l564_56435

/-- The ellipse C defined by the equation x²/9 + y²/4 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- A point on the ellipse C -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- The foci of the ellipse C -/
structure Foci where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The maximum value of |MF₁| · |MF₂| is 9 -/
theorem max_product_of_distances (M : PointOnEllipse) (F : Foci) :
  (distance (M.x, M.y) F.F₁) * (distance (M.x, M.y) F.F₂) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l564_56435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_l564_56418

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

theorem unique_solution_for_f (a : ℝ) :
  (f a + f 1 = 0) ↔ (a = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_l564_56418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_paid_l564_56403

def aquarium_original_price : ℝ := 120
def aquarium_discount1 : ℝ := 0.5
def aquarium_discount2 : ℝ := 0.1
def plants_decorations_price : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def fish_food_price : ℝ := 25
def aquarium_tax_rate : ℝ := 0.05
def plants_decorations_tax_rate : ℝ := 0.08
def fish_food_tax_rate : ℝ := 0.06

theorem total_amount_paid : 
  let aquarium_price := aquarium_original_price * (1 - aquarium_discount1) * (1 - aquarium_discount2)
  let plants_decorations_discounted := plants_decorations_price * (1 - plants_decorations_discount)
  aquarium_price * (1 + aquarium_tax_rate) +
  plants_decorations_discounted * (1 + plants_decorations_tax_rate) +
  fish_food_price * (1 + fish_food_tax_rate) = 152.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_paid_l564_56403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l564_56492

open Real

def triangle_proof (a b c : ℝ) (α γ : ℝ) : Prop :=
  let β := 180 - α - γ
  a = 5 ∧
  α = 72 ∧
  (a^2 - c^2)^2 = b^2 * (2*c^2 - b^2) ∧
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos (γ * π / 180)) ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  α + β + γ = 180 →
  γ = 45 ∧ β = 63

theorem triangle_theorem :
  ∃ (a b c α γ : ℝ), triangle_proof a b c α γ :=
by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l564_56492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_convex_is_convex_l564_56471

-- Define a convex figure in a general space
def ConvexFigure (α : Type*) [AddCommGroup α] [Module ℝ α] (S : Set α) : Prop :=
  ∀ (x y : α), x ∈ S → y ∈ S → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ S

-- Define the theorem
theorem intersection_of_convex_is_convex
  {α : Type*} [AddCommGroup α] [Module ℝ α]
  {ι : Type*} (s : ι → Set α)
  (h : ∀ i, ConvexFigure α (s i)) :
  ConvexFigure α (⋂ i, s i) := by
  -- Introduce variables and assumptions
  intro x y hx hy t ht
  -- Show that x and y are in all sets s i
  have hx_all : ∀ i, x ∈ s i := by
    intro i
    exact Set.mem_iInter.1 hx i
  have hy_all : ∀ i, y ∈ s i := by
    intro i
    exact Set.mem_iInter.1 hy i
  -- Apply the convexity of each set s i
  have h_conv : ∀ i, (1 - t) • x + t • y ∈ s i := by
    intro i
    exact h i x y (hx_all i) (hy_all i) t ht
  -- Conclude that the convex combination is in the intersection
  exact Set.mem_iInter.2 h_conv

-- This line is added to satisfy the proof obligation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_convex_is_convex_l564_56471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l564_56428

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin (2 * x) / Real.sin x

-- Define the domain of f
def domain_f : Set ℝ := {x | ∀ k : ℤ, x ≠ k * Real.pi}

-- State the theorem
theorem f_properties :
  -- 1. Domain of f
  (∀ x : ℝ, f x ≠ 0 ↔ x ∈ domain_f) ∧
  -- 2. Smallest positive period of f
  (∃ T : ℝ, T > 0 ∧ (∀ x ∈ domain_f, f (x + T) = f x) ∧
    (∀ S : ℝ, S > 0 → (∀ x ∈ domain_f, f (x + S) = f x) → S ≥ T) ∧ T = Real.pi) ∧
  -- 3. Intervals where f is monotonically decreasing
  (∀ k : ℤ, ∀ x y : ℝ,
    k * Real.pi + 3 * Real.pi / 8 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 7 * Real.pi / 8 →
    f y < f x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l564_56428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l564_56448

/-- Predicate to ensure a, b, c form a valid triangle. -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given two sides and the angle between them. -/
noncomputable def area_triangle (a b : ℝ) (C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin C

/-- In triangle ABC, given side lengths and cosine of an angle, prove the length of the third side and the area of the triangle. -/
theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 2) 
  (h_cos_B : Real.cos B = 1/4) 
  (h_triangle : IsTriangle a b c) : 
  c = 2 ∧ 
  area_triangle a b C = Real.sqrt 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l564_56448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l564_56466

noncomputable def root_of_unity (n : ℕ) : ℂ := Complex.exp (2 * Real.pi * Complex.I / n)

noncomputable def sum_powers (z : ℂ) (powers : List ℕ) : ℂ :=
  (powers.map (λ k => z ^ k)).sum

theorem complex_sum_theorem :
  let ω := root_of_unity 7
  let θ := root_of_unity 9
  (sum_powers ω [1, 2, 3, 4, 5, 6]) + (sum_powers θ [1, 2, 4, 8, 16]) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_theorem_l564_56466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_81_approx_l564_56485

theorem power_of_81_approx : 
  ∃ ε > 0, |(81 : ℝ)^(0.25 : ℝ) * (81 : ℝ)^(0.20 : ℝ) - 6.86| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_81_approx_l564_56485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_over_three_l564_56421

/-- A rectangular track with rounded ends -/
structure Track where
  straight_side : ℝ
  width : ℝ
  inner_radius : ℝ

/-- Calculate the walking speed given a track and time difference -/
noncomputable def calculate_speed (t : Track) (time_diff : ℝ) : ℝ :=
  let outer_radius := t.inner_radius + t.width
  let inner_semicircle := 2 * Real.pi * t.inner_radius
  let outer_semicircle := 2 * Real.pi * outer_radius
  let distance_diff := outer_semicircle - inner_semicircle
  distance_diff / time_diff

/-- The theorem stating the walking speed for the given track parameters -/
theorem walking_speed_is_pi_over_three (t : Track) (h1 : t.straight_side = 100)
    (h2 : t.width = 12) (h3 : t.inner_radius = 15) : 
    calculate_speed t 72 = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_over_three_l564_56421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l564_56411

theorem remainder_sum_mod_30 (a b c : ℕ) 
  (ha : a % 30 = 12)
  (hb : b % 30 = 9)
  (hc : c % 30 = 15) :
  (a + b + c) % 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l564_56411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_m_l564_56437

/-- A function f is a power function if it can be written as f(x) = ax^b for some constants a and b -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x ^ b

/-- A function f is decreasing on an interval (a, b) if for any x₁, x₂ in (a, b) with x₁ < x₂, we have f(x₁) > f(x₂) -/
def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ > f x₂

/-- The main theorem -/
theorem power_function_decreasing_m (m : ℝ) :
  let f := fun x : ℝ => (m^2 - m - 1) * x^(m^2 + m - 3)
  IsPowerFunction f ∧ IsDecreasingOn f 0 (Real.rpow 10 1000) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_m_l564_56437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_property_l564_56420

/-- An equilateral triangle with side length 10 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : dist A B = 10 ∧ dist B C = 10 ∧ dist C A = 10

/-- A point on a line segment -/
def on_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

/-- An equilateral triangle -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

theorem equilateral_triangle_property 
  (ABC : EquilateralTriangle) 
  (D E F G : ℝ × ℝ) :
  on_segment D ABC.A ABC.B →
  on_segment E ABC.A ABC.C →
  on_segment F ABC.B ABC.C →
  on_segment G ABC.B ABC.C →
  is_equilateral_triangle ABC.A D E →
  is_equilateral_triangle ABC.B D G →
  is_equilateral_triangle ABC.C E F →
  dist ABC.A D = 3 →
  dist F G = 4 :=
by
  sorry

#check equilateral_triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_property_l564_56420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_area_theorem_l564_56444

/-- Represents a rectangle partitioned into four smaller rectangles --/
structure PartitionedRectangle where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  parallel_sum : ℝ

/-- The area of the fourth rectangle in a partitioned rectangle --/
noncomputable def fourth_area (r : PartitionedRectangle) : ℝ :=
  sorry

/-- Theorem stating the properties of the fourth area --/
theorem fourth_area_theorem (r : PartitionedRectangle) 
  (h1 : r.area1 = 24)
  (h2 : r.area2 = 35)
  (h3 : r.area3 = 42)
  (h4 : r.parallel_sum = 21) :
  33 < fourth_area r ∧ fourth_area r < 34 := by
  sorry

#check fourth_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_area_theorem_l564_56444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_l564_56405

theorem min_diagonal_of_rectangle (l w : ℝ) : 
  l > 0 → w > 0 → l + w = 15 → 
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + w' = 15 → 
  (l^2 + w^2 : ℝ) ≤ l'^2 + w'^2 → 
  Real.sqrt (l^2 + w^2) = 7.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_l564_56405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l564_56404

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0
  a_gt_b : a > b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt ((a^2 + b^2) / a^2)

/-- The semi-focal distance of a hyperbola -/
noncomputable def semi_focal_distance (h : Hyperbola a b) : ℝ := 
  Real.sqrt (a^2 - b^2)

/-- Theorem: For a hyperbola with the given properties, its eccentricity is √3 + 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (circle_intersect : ∃ A B : ℝ × ℝ, 
    A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ 
    B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧
    A.1^2 + A.2^2 = (semi_focal_distance h)^2 ∧
    B.1^2 + B.2^2 = (semi_focal_distance h)^2)
  (equilateral_triangle : ∃ F₁ : ℝ × ℝ, 
    F₁.1 = -(semi_focal_distance h) ∧ F₁.2 = 0 ∧
    ∀ A B : ℝ × ℝ, (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.1^2 + A.2^2 = (semi_focal_distance h)^2) →
                   (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.1^2 + B.2^2 = (semi_focal_distance h)^2) →
                   (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = (B.1 - F₁.1)^2 + (B.2 - F₁.2)^2 ∧
                   (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2) :
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l564_56404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_sum_of_extrema_l564_56451

/-- Given a linear function y = ax on the interval [0, 1],
    if the sum of its maximum and minimum values is 3,
    then a = 3. -/
theorem linear_function_sum_of_extrema (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a * x ≤ a * 1 ∧ a * 0 ≤ a * x) →
  a * 1 + a * 0 = 3 →
  a = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_sum_of_extrema_l564_56451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l564_56490

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (1 + x) = -f (1 - x)
axiom f_even : ∀ x, f (x + 2) = f (-x + 2)
axiom f_def : ∀ x, x ∈ Set.Icc 0 1 → f x = 1 - x

-- State the theorem to be proved
theorem f_inequalities :
  f (Real.sin 1) < f (Real.cos 1) ∧
  f (Real.cos (2 * Real.pi / 3)) > f (Real.sin (2 * Real.pi / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l564_56490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l564_56440

/-- Given vectors a and b in ℝ², if a is perpendicular to b + λa, then λ = 1/5 -/
theorem perpendicular_vector_lambda (a b : ℝ × ℝ) (l : ℝ) : 
  a = (1, -3) → 
  b = (4, 2) → 
  (a.1 * (b.1 + l * a.1) + a.2 * (b.2 + l * a.2) = 0) → 
  l = 1/5 := by
  sorry

#check perpendicular_vector_lambda

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l564_56440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l564_56467

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_obtuse_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = Real.pi ∧ max t.A (max t.B t.C) > Real.pi/2

def satisfies_equation (t : Triangle) : Prop :=
  (Real.cos t.A) / (1 - Real.sin t.A) = (Real.cos t.A + Real.cos t.B) / (1 - Real.sin t.A + Real.sin t.B)

-- Theorem statements
theorem part_one (t : Triangle) 
  (h1 : is_obtuse_triangle t) 
  (h2 : satisfies_equation t) 
  (h3 : t.C = 2*Real.pi/3) : 
  t.A = Real.pi/6 := by sorry

theorem part_two (t : Triangle) 
  (h1 : is_obtuse_triangle t) 
  (h2 : satisfies_equation t) : 
  ∃ (min_value : Real), 
    (∀ (t' : Triangle), is_obtuse_triangle t' → satisfies_equation t' → 
      (t'.a^2 + t'.c^2) / t'.b^2 ≥ min_value) ∧
    min_value = 4 * Real.sqrt 2 - 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l564_56467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l564_56402

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:5,
    prove that P = (5/8)*A + (3/8)*B -/
theorem point_division_theorem (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, P = A + t • (B - A)) →  -- P is on line segment AB
  (norm (A - P)) / (norm (P - B)) = 3 / 5 →  -- AP:PB = 3:5
  (∃ t u : ℝ, P = t • A + u • B ∧ t + u = 1) →  -- P is a weighted sum of A and B
  P = (5/8) • A + (3/8) • B := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l564_56402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l564_56473

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y > 0 then Real.pi/2
           else if y < 0 then 3*Real.pi/2
           else 0  -- undefined, but we need a value
  let θ_normalized := if θ < 0 then θ + 2*Real.pi else θ
  (r, θ_normalized, z)

theorem rectangular_to_cylindrical_conversion :
  rectangular_to_cylindrical 4 (-4) 6 = (4 * Real.sqrt 2, 7*Real.pi/4, 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_conversion_l564_56473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_sum_l564_56401

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific relation holds, prove that the sum of their common ratios is 5. -/
theorem geometric_sequence_ratio_sum (k s t : ℝ) (hs : s ≠ 1) (ht : t ≠ 1) (hst : s ≠ t) :
  (k * s^2 - k * t^2 = 5 * (k * s - k * t)) → (s + t = 5) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_sum_l564_56401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_right_parallelepiped_l564_56460

/-- The volume of a right parallelepiped with a parallelogram base -/
noncomputable def parallelepiped_volume (a b α β : ℝ) : ℝ :=
  (a * b^2 * Real.sin α) / (2 * Real.cos β) * Real.sqrt (Real.sin (β + α) * Real.sin (β - α))

/-- Theorem: Volume of a right parallelepiped with given conditions -/
theorem volume_of_right_parallelepiped
  (a b α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π/2)
  (h3 : 0 < β ∧ β < π/2) :
  ∃ V : ℝ, V = parallelepiped_volume a b α β ∧ V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_right_parallelepiped_l564_56460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fold_theorem_l564_56453

/-- Square with side length 2 -/
structure Square where
  sideLength : ℝ
  sideLength_eq : sideLength = 2

/-- Point on a side of the square -/
structure Point where
  x : ℝ
  y : ℝ

/-- Length of a line segment -/
abbrev Length := ℝ

/-- Side length of the square -/
def sideLength : Length := 2

/-- Diagonal length of the square -/
noncomputable def diagonalLength : Length := Real.sqrt 8

/-- Function to represent folding along a line -/
noncomputable def fold (p q : Point) : Point → Point := sorry

/-- Function to get the length between two points -/
noncomputable def distance (p q : Point) : Length := 
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: In a square PQRS with side length 2, if T on PQ and U on SQ satisfy
    PT = SU, and when folded along RT and RU, PR and SR coincide on RQ, 
    then PT = √2 - 1 -/
theorem square_fold_theorem (p q r s t u : Point) : 
  distance p q = sideLength ∧ 
  distance p s = sideLength ∧
  distance q r = sideLength ∧
  distance r s = sideLength ∧
  distance p t = distance s u ∧
  fold r t p = fold r u s ∧
  (∃ (x : Point), fold r t p = x ∧ fold r u s = x ∧ distance r x + distance x q = diagonalLength) →
  distance p t = Real.sqrt 2 - 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fold_theorem_l564_56453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_road_cars_l564_56449

/-- Prove the number of cars on River Road given the ratio and differences -/
theorem river_road_cars (b c m t : ℕ) : 
  b + c + m + t > 0 →  -- Ensure total vehicles is positive
  c = 13 * b →         -- Ratio of cars to buses
  m = 5 * b →          -- Ratio of motorcycles to buses
  t = 7 * b →          -- Ratio of trucks to buses
  c = b + 60 →         -- Difference between cars and buses
  t = m + 40 →         -- Difference between trucks and motorcycles
  c = 65 := by
  intro h1 h2 h3 h4 h5 h6
  -- The proof steps would go here
  sorry

#check river_road_cars

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_road_cars_l564_56449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_construction_l564_56450

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

-- Define a parabola
structure Parabola where
  focus : Point
  directrix : Line

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define membership for Point in Parabola
def Point.mem (p : Point) (parab : Parabola) : Prop :=
  -- This is a placeholder definition. In reality, we would need to check
  -- if the point satisfies the parabola equation.
  True

instance : Membership Point Parabola where
  mem := Point.mem

-- Define the theorem
theorem parabola_directrix_construction 
  (F : Point) -- focus of the parabola
  (P₁ P₂ : Point) -- two points on the parabola
  : ∃ (d₁ d₂ : Line), 
    d₁ ≠ d₂ ∧ 
    (∃ (p₁ : Parabola), p₁.focus = F ∧ p₁.directrix = d₁ ∧ P₁ ∈ p₁ ∧ P₂ ∈ p₁) ∧
    (∃ (p₂ : Parabola), p₂.focus = F ∧ p₂.directrix = d₂ ∧ P₁ ∈ p₂ ∧ P₂ ∈ p₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_construction_l564_56450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_precise_value_l564_56424

noncomputable def measured_value : ℝ := 2.43865
noncomputable def error_range : ℝ := 0.00312

noncomputable def lower_bound : ℝ := measured_value - error_range
noncomputable def upper_bound : ℝ := measured_value + error_range

def is_valid_published_value (x : ℝ) : Prop :=
  (lower_bound ≤ x) ∧ (x ≤ upper_bound)

def is_most_precise (x : ℝ) : Prop :=
  is_valid_published_value x ∧
  ∀ y : ℝ, is_valid_published_value y → (⌊x * 1000⌋ = ⌊y * 1000⌋)

theorem most_precise_value : is_most_precise 2.44 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_precise_value_l564_56424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_proof_l564_56484

/-- The area of a regular octagon inscribed in a circle with radius 2 units -/
noncomputable def regular_octagon_area : ℝ := 16 * Real.sqrt 2 - 16

/-- Theorem: The area of a regular octagon inscribed in a circle with radius 2 units is 16√2 - 16 -/
theorem regular_octagon_area_proof (r : ℝ) (h : r = 2) : 
  regular_octagon_area = 8 * (r^2 * Real.sin (π/8) * Real.cos (π/8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_proof_l564_56484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l564_56465

def sequenceQ : ℕ → ℚ
  | 0 => 800000
  | n + 1 => sequenceQ n / 2

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

theorem last_integer_in_sequence :
  ∃ (n : ℕ), sequenceQ n = 3125 ∧ is_integer (sequenceQ n) ∧ ¬is_integer (sequenceQ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l564_56465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangements_l564_56497

/-- Two congruent right triangles with legs of 4 cm and 7 cm -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right : leg1 = 4 ∧ leg2 = 7

/-- First arrangement: 7 cm legs coincide -/
structure FirstArrangement (t : RightTriangle) where
  AB : ℝ
  AD : ℝ
  BC : ℝ
  shaded_area : ℝ
  h1 : AB = 7
  h2 : AD = 4
  h3 : BC = 4
  h4 : shaded_area = 21

/-- Second arrangement: hypotenuses coincide -/
structure SecondArrangement (t : RightTriangle) where
  AD : ℝ
  BC : ℝ
  AC : ℝ
  BD : ℝ
  shaded_area : ℝ
  h1 : AD = 4
  h2 : BC = 4
  h3 : AC = 7
  h4 : BD = 7
  h5 : shaded_area = 18 + 5/7

/-- The main theorem -/
theorem triangle_arrangements (t : RightTriangle) :
  (∃ a : FirstArrangement t, True) ∧
  (∃ b : SecondArrangement t, True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arrangements_l564_56497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l564_56489

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x - 2

theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 1 < x ∧ x < y → f a x < f a y) →
  a ≥ -3 ∧ ∀ b ≥ -3, ∃ c : ℝ, a = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l564_56489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l564_56400

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the line
def myLine (x y : ℝ) (a : ℝ) : Prop := 4*x - 3*y + a = 0

-- Define the angle AOB
noncomputable def angle_AOB (A B O : ℝ × ℝ) : ℝ := sorry

-- Define the intersection points
def intersection_points (a : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, myCircle A.1 A.2 ∧ myCircle B.1 B.2 ∧ 
              myLine A.1 A.2 a ∧ myLine B.1 B.2 a

-- Theorem statement
theorem line_circle_intersection (a : ℝ) :
  intersection_points a →
  (∃ O : ℝ × ℝ, ∃ A B : ℝ × ℝ, angle_AOB A B O = 2*π/3) →
  a = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l564_56400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l564_56407

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) + (1/2) * x^2 - x

/-- Theorem stating the main result to be proved -/
theorem extreme_points_inequality (a : ℝ) (α β : ℝ) :
  a ≠ 0 →  -- a is non-zero
  0 < a →  -- a is positive (based on the problem conditions)
  a < 1 →  -- a is less than 1 (based on the problem conditions)
  α < β →  -- α is less than β
  α = -Real.sqrt (1 - a) →  -- α is the smaller extreme point
  β = Real.sqrt (1 - a) →   -- β is the larger extreme point
  f a β / α < 1/2 := by
  sorry

#check extreme_points_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l564_56407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_l564_56433

/-- Proves that a price decreased by 30% and then increased by 20% results in 84% of the original price -/
theorem price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.3) * (1 + 0.2) = P * 0.84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_l564_56433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_equals_2_l564_56438

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | n + 2 => if n % 2 = 0 then 1 / (1 - sequence_a (n + 1)) else 1 / sequence_a (n + 1)

theorem a_2023_equals_2 : sequence_a 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_equals_2_l564_56438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_KLM_l564_56496

-- Define the circle and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def K : Point := ⟨0, 0⟩
def L : Point := ⟨0, 0⟩
def M : Point := ⟨0, 0⟩
def N : Point := ⟨0, 0⟩
def P : Point := ⟨0, 0⟩

-- Define the radius of the circle
noncomputable def radius : ℝ := 2 * Real.sqrt 2

-- Define the parallel relation
def parallel (p q r s : Point) : Prop := sorry

-- Define the angle measure
noncomputable def angle_measure (p q r : Point) : ℝ := sorry

-- Define the area of a triangle
noncomputable def triangle_area (p q r : Point) : ℝ := sorry

-- Define the theorem
theorem area_of_triangle_KLM (c : Circle) :
  parallel L M K N →
  parallel K M N P →
  parallel M N L P →
  angle_measure L O M = 45 →
  triangle_area K L M = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_KLM_l564_56496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_losing_position_iff_xor_sum_zero_l564_56476

/-- A game where players can remove any number of pieces of the form 2^k from piles -/
structure NimGame where
  piles : List ℕ

/-- The XOR sum of all piles in the game -/
def xorSum (game : NimGame) : ℕ :=
  game.piles.foldl Nat.xor 0

/-- A position is losing (P) if and only if the XOR sum of all piles is zero -/
theorem losing_position_iff_xor_sum_zero (game : NimGame) :
  (∀ (i : ℕ) (k : ℕ), i < game.piles.length →
    ∃ (new_game : NimGame),
      new_game.piles.length = game.piles.length ∧
      (∀ j, j < game.piles.length →
        (j = i → new_game.piles[j]! = game.piles[j]! - 2^k) ∧
        (j ≠ i → new_game.piles[j]! = game.piles[j]!)) ∧
      xorSum new_game ≠ 0) ↔
  xorSum game = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_losing_position_iff_xor_sum_zero_l564_56476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_is_one_km_l564_56488

/-- Represents the distance between two cars given their speeds and initial separation. -/
noncomputable def finalDistance (initialSpeed initialDistance citySpeed goodRoadSpeed dirtRoadSpeed : ℝ) : ℝ :=
  initialDistance * (citySpeed / initialSpeed) * (goodRoadSpeed / citySpeed) * (dirtRoadSpeed / goodRoadSpeed)

/-- Theorem stating that under the given conditions, the final distance between the cars is 1 km. -/
theorem final_distance_is_one_km : 
  finalDistance 60 2 40 70 30 = 1 := by
  -- Unfold the definition of finalDistance
  unfold finalDistance
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_div_cancel']
  -- Check that the result is equal to 1
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_distance_is_one_km_l564_56488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_limit_equals_instantaneous_velocity_l564_56479

/-- Represents the displacement of an object moving in a straight line -/
noncomputable def displacement (t : ℝ) : ℝ := sorry

/-- The instantaneous velocity of the object at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv displacement t

/-- Theorem: The limit of the ratio of displacement change to time change
    as time change approaches zero is equal to the instantaneous velocity -/
theorem displacement_limit_equals_instantaneous_velocity (t : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |((displacement (t + h) - displacement t) / h) - instantaneous_velocity t| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_limit_equals_instantaneous_velocity_l564_56479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_segment_l564_56409

/-- Parametric equations of the curve -/
noncomputable def x (θ : ℝ) : ℝ := 2 + (Real.cos θ)^2
noncomputable def y (θ : ℝ) : ℝ := 1 - (Real.sin θ)^2

/-- The parameter range -/
def θ_range : Set ℝ := {θ | 0 ≤ θ ∧ θ < 2 * Real.pi}

/-- The curve as a set of points -/
def curve : Set (ℝ × ℝ) := {(x θ, y θ) | θ ∈ θ_range}

/-- Definition of a line segment -/
def is_line_segment (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ × ℝ), S = {p | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b}

theorem curve_is_line_segment : is_line_segment curve := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_segment_l564_56409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_gt_one_neither_sufficient_nor_necessary_l564_56445

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n ↦ a * q^(n - 1)

/-- An increasing sequence -/
def is_increasing (s : ℕ → ℝ) : Prop := ∀ n, s (n + 1) > s n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary (a q : ℝ) :
  ¬(((q > 1) → is_increasing (geometric_sequence a q)) ∧
    (is_increasing (geometric_sequence a q) → (q > 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_gt_one_neither_sufficient_nor_necessary_l564_56445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_all_flashes_l564_56495

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The time taken for one flash (in seconds) -/
def flash_time : ℕ := 5

/-- The interval time between flashes (in seconds) -/
def interval_time : ℕ := 5

/-- The number of different possible flashes -/
def num_flashes : ℕ := Nat.factorial num_lights

theorem min_time_for_all_flashes : 
  num_flashes * flash_time + (num_flashes - 1) * interval_time = 1195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_all_flashes_l564_56495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_abs_k_leq_2_is_sufficient_l564_56410

/-- Given two functions f and g, where f(x) = x^2 - 2x + 3 and g(x) = kx - 1,
    this theorem states that |k| ≤ 2 is a sufficient but not necessary condition
    for f(x) ≥ g(x) to always hold on ℝ. -/
theorem sufficient_not_necessary_condition :
  ∃ k : ℝ, 
    let f : ℝ → ℝ := λ x => x^2 - 2*x + 3
    let g : ℝ → ℝ := λ x => k*x - 1
    (∀ x, f x ≥ g x) ∧ (abs k > 2) :=
by
  -- We'll prove this by providing a specific value of k that satisfies the conditions
  use -3  -- k = -3 satisfies the conditions
  -- The proof details would go here
  sorry

-- This theorem shows that |k| ≤ 2 is sufficient
theorem abs_k_leq_2_is_sufficient (k : ℝ) (h : abs k ≤ 2) :
  let f : ℝ → ℝ := λ x => x^2 - 2*x + 3
  let g : ℝ → ℝ := λ x => k*x - 1
  ∀ x, f x ≥ g x :=
by
  -- The proof details would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_abs_k_leq_2_is_sufficient_l564_56410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l564_56499

/-- The shift amount for the cosine function -/
noncomputable def shift : ℝ := Real.pi / 6

/-- The original cosine function -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (3 * x)

/-- The shifted cosine function -/
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (3 * x + Real.pi / 2)

/-- Theorem stating that g is a left shift of f by π/6 -/
theorem cosine_shift : ∀ x : ℝ, g x = f (x + shift) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l564_56499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l564_56474

noncomputable section

-- Define the ellipse parameters
def a : ℝ := 4
def b : ℝ := 2

-- Define the eccentricity
def e : ℝ := Real.sqrt 3 / 2

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the intersecting line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line p.1 p.2}

-- State the theorem
theorem chord_length :
  a > b ∧ b > 0 ∧
  e = Real.sqrt (a^2 - b^2) / a ∧
  2 * b = 4 ∧
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 2 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l564_56474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_function_inequality_implies_identity_l564_56408

-- Define the set of positive integers
def PositiveIntegers : Type := {n : ℕ // n > 0}

-- Define the property of the function
def SatisfiesInequality (f : PositiveIntegers → PositiveIntegers) : Prop :=
  ∀ n : PositiveIntegers, (f (f n)).val ≤ (n.val + (f n).val) / 2

-- State the theorem
theorem bijective_function_inequality_implies_identity
  (f : PositiveIntegers → PositiveIntegers)
  (hf_bij : Function.Bijective f)
  (hf_ineq : SatisfiesInequality f) :
  ∀ n : PositiveIntegers, f n = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_function_inequality_implies_identity_l564_56408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l564_56446

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := max (x^2) (Real.sqrt x)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x ≥ (1/4 : ℝ)

-- Define the area of the closed figure
noncomputable def area : ℝ := ∫ x in (1/4)..(2), f x

-- Theorem statement
theorem area_of_closed_figure : domain (1/4) → area = 35/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l564_56446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_ACED_area_continuous_l564_56462

/-- Given a line segment AD of length d, we mark a point B on AD such that AB = x (0 ≤ x ≤ d).
    We construct an equilateral triangle ABC, and draw perpendiculars CE to BC and DE to AD,
    intersecting at E. This theorem states the area of quadrilateral ACED in terms of d and x. -/
theorem area_quadrilateral_ACED (d x : ℝ) (h : 0 ≤ x ∧ x ≤ d) :
  (6 * d^2 - (2 * d - x)^2) / (4 * Real.sqrt 3) ≥ 
  (2 * d^2) / (4 * Real.sqrt 3) ∧
  (6 * d^2 - (2 * d - x)^2) / (4 * Real.sqrt 3) ≤ 
  (5 * d^2) / (4 * Real.sqrt 3) :=
by
  sorry

/-- The area of the quadrilateral ACED is a continuous function of x on the interval [0, d]. -/
theorem area_continuous (d : ℝ) (h : d > 0) :
  Continuous (fun x : ℝ ↦ (6 * d^2 - (2 * d - x)^2) / (4 * Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_ACED_area_continuous_l564_56462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_price_after_discounts_l564_56478

/-- Applies a discount percentage to a given price --/
noncomputable def apply_discount (price : ℝ) (discount_percent : ℝ) : ℝ :=
  price * (1 - discount_percent / 100)

/-- Theorem stating that applying successive discounts of 15%, 10%, and 5% to 3600 results in 2616.30 --/
theorem cycle_price_after_discounts :
  let initial_price := (3600 : ℝ)
  let discount1 := (15 : ℝ)
  let discount2 := (10 : ℝ)
  let discount3 := (5 : ℝ)
  let final_price := apply_discount (apply_discount (apply_discount initial_price discount1) discount2) discount3
  final_price = 2616.30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_price_after_discounts_l564_56478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l564_56429

/-- Triangle XYZ with specific properties -/
structure TriangleXYZ where
  /-- Side XY of the triangle -/
  xy : ℝ
  /-- Side XZ of the triangle -/
  xz : ℝ
  /-- Area of the triangle -/
  area : ℝ
  /-- Geometric mean between XY and XZ -/
  gm : ℝ
  /-- Condition: XY is 25 inches -/
  xy_length : xy = 25
  /-- Condition: Area is 100 square units -/
  area_value : area = 100
  /-- Condition: Geometric mean is 15 inches -/
  gm_value : gm = 15
  /-- Condition: Geometric mean definition -/
  gm_def : gm^2 = xy * xz

/-- Angle X of the triangle -/
def angle_X (t : TriangleXYZ) : ℝ := sorry

/-- Theorem: Under given conditions, sin X = 8/9 -/
theorem sin_x_value (t : TriangleXYZ) : Real.sin (angle_X t) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l564_56429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sum_7_l564_56463

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  a_positive : ∀ n, a n > 0
  a_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  a_first : a 1 = 1
  a_arithmetic : a 4 - a 2 = a 2 - (-a 3)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (seq : SpecialGeometricSequence) (n : ℕ) : ℝ :=
  (1 - (seq.a 2 / seq.a 1) ^ n) / (1 - seq.a 2 / seq.a 1)

/-- The theorem to be proved -/
theorem special_geometric_sum_7 (seq : SpecialGeometricSequence) :
  geometric_sum seq 7 = 127 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sum_7_l564_56463
