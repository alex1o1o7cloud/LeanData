import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validTripleCount_l29_2977

-- Define the set of numbers
def S : Finset ℕ := Finset.range 10

-- Define a function to check if a sum is divisible by 3
def sumDivisibleBy3 (a b c : ℕ) : Bool := (a + b + c) % 3 = 0

-- Define the set of valid triples
def validTriples : Finset (ℕ × ℕ × ℕ) :=
  Finset.filter (fun (x : ℕ × ℕ × ℕ) => x.1 ∈ S ∧ x.2.1 ∈ S ∧ x.2.2 ∈ S ∧ 
    sumDivisibleBy3 x.1 x.2.1 x.2.2) (Finset.product S (Finset.product S S))

-- Theorem statement
theorem validTripleCount : Finset.card validTriples = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_validTripleCount_l29_2977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l29_2905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a * x^2 + x - 1 else -x + 1

theorem monotonically_decreasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l29_2905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equality_l29_2990

open Real

-- Define the types for points and circles
variable (Point Circle : Type*)

-- Define the cyclic quadrilateral ABCD
variable (A B C D : Point)

-- Define the points on the sides of ABCD
variable (K₁ K₂ L₁ L₂ M₁ M₂ N₁ N₂ : Point)

-- Define the circle ω
variable (ω : Circle)

-- Define the arcs on ω
variable (a b c d : ℝ)

-- Define the necessary functions
variable (CyclicQuadrilateral : Point → Point → Point → Point → Prop)
variable (OnSegment : Point → Point → Point → Prop)
variable (OnCircle : Point → Circle → Prop)
variable (CircularOrder : Circle → Point → Point → Point → Point → Point → Point → Point → Point → Prop)
variable (ArcLength : Circle → Point → Point → ℝ)

-- Hypotheses
variable (h1 : CyclicQuadrilateral A B C D)
variable (h2 : OnSegment K₁ A B)
variable (h3 : OnSegment K₂ A B)
variable (h4 : OnSegment L₁ B C)
variable (h5 : OnSegment L₂ B C)
variable (h6 : OnSegment M₁ C D)
variable (h7 : OnSegment M₂ C D)
variable (h8 : OnSegment N₁ D A)
variable (h9 : OnSegment N₂ D A)
variable (h10 : OnCircle K₁ ω)
variable (h11 : OnCircle K₂ ω)
variable (h12 : OnCircle L₁ ω)
variable (h13 : OnCircle L₂ ω)
variable (h14 : OnCircle M₁ ω)
variable (h15 : OnCircle M₂ ω)
variable (h16 : OnCircle N₁ ω)
variable (h17 : OnCircle N₂ ω)
variable (h18 : CircularOrder ω K₁ K₂ L₁ L₂ M₁ M₂ N₁ N₂)
variable (h19 : a = ArcLength ω N₂ K₁)
variable (h20 : b = ArcLength ω K₂ L₁)
variable (h21 : c = ArcLength ω L₂ M₁)
variable (h22 : d = ArcLength ω M₂ N₁)

-- Theorem
theorem arc_length_equality : a + c = b + d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equality_l29_2990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l29_2951

/-- The area of a triangle given three points in a 2D plane -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

theorem triangle_ABC_area :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 2)
  let C : ℝ × ℝ := (2, 0)
  triangleArea A B C = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l29_2951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l29_2913

noncomputable def f (x : ℝ) : ℝ :=
  if x > 9 then 1 / x else x ^ 2

theorem f_composition_result : f (f (f 2)) = 1 / 16 := by
  -- Evaluate f(2)
  have h1 : f 2 = 4 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(2))
  have h2 : f (f 2) = 16 := by
    rw [h1]
    simp [f]
    norm_num
  
  -- Evaluate f(f(f(2)))
  rw [h2]
  simp [f]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l29_2913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l29_2919

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line passes through the given point -/
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line forms a triangle with the positive x-axis and y-axis -/
def forms_triangle (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (l.c / l.a < 0 ∨ l.c / l.b < 0)

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  abs (l.c / l.a * l.c / l.b) / 2

theorem line_equation : 
  ∃ (l : Line), 
    passes_through l ⟨1, 3⟩ ∧ 
    forms_triangle l ∧ 
    triangle_area l = 6 ∧ 
    l = ⟨3, 1, -6⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l29_2919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequence_composite_l29_2922

/-- The sequence of numbers in the form 1 + 10^4 + 10^8 + ... + 10^(4k) -/
def my_sequence (k : ℕ) : ℕ :=
  (Finset.range (k + 1)).sum (fun i => 10^(4*i))

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem all_sequence_composite :
  ∀ k : ℕ, k > 0 → is_composite (my_sequence k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequence_composite_l29_2922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_ellipse_equation_second_ellipse_equation_l29_2956

-- Define the ellipse structure
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance

-- Define the equation of an ellipse
def ellipse_equation (e : Ellipse) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / e.a^2 + y^2 / e.b^2 = 1

-- First ellipse theorem
theorem first_ellipse_equation (e : Ellipse) 
  (h1 : e.b = 3)  -- Minor axis length is 6 (2b = 6)
  (h2 : e.c = 4)  -- Distance between foci is 8 (2c = 8)
  : ellipse_equation e = λ x y ↦ x^2 / 25 + y^2 / 9 = 1 :=
sorry

-- Second ellipse theorem
theorem second_ellipse_equation (e : Ellipse) 
  (h1 : e.c / e.a = Real.sqrt 3 / 2)  -- Eccentricity is √3/2
  (h2 : ellipse_equation e 4 (2 * Real.sqrt 3))  -- Passes through (4, 2√3)
  : ellipse_equation e = λ x y ↦ x^2 / 8 + y^2 / 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_ellipse_equation_second_ellipse_equation_l29_2956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l29_2978

open Real

-- Define the functions and domain
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (1 / Real.exp 1) ^ (x / 2)
def domain : Set ℝ := Set.Icc (1 / Real.exp 1) (Real.exp 1)

-- Define symmetry condition
def symmetric (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x ∈ domain ∧ y ∈ domain ∧ 
  f k x = y ∧ g y = x

-- Theorem statement
theorem k_range (k : ℝ) : 
  symmetric k → k ∈ Set.Icc (-2 / Real.exp 1) (2 * Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l29_2978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l29_2937

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - Real.sqrt 3

noncomputable def g (x : ℝ) := f (x - Real.pi / 6) + Real.sqrt 3

def is_axis_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem function_properties :
  (∀ k : ℤ, is_axis_of_symmetry f (Real.pi / 12 + k * Real.pi / 2)) ∧
  (∀ k : ℤ, is_increasing_on f (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi)) ∧
  (∀ k : ℤ, is_decreasing_on f (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi)) ∧
  is_odd_function g :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l29_2937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_plane_exists_l29_2900

/-- A color type with exactly five colors -/
inductive Color
| Red
| Blue
| Green
| Yellow
| Purple

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A coloring function that assigns a color to each point in space -/
def Coloring := Point3D → Color

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def Point3D.onPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

theorem four_color_plane_exists (coloring : Coloring) 
  (h1 : ∀ c : Color, ∃ p : Point3D, coloring p = c) :
  ∃ plane : Plane, ∃ s : Finset Color, 
    (∀ p : Point3D, p.onPlane plane → coloring p ∈ s) ∧ s.card ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_color_plane_exists_l29_2900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_repeating_decimals_l29_2947

def is_valid_repeating_decimal (n : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ : ℕ), 
    (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₂ ≠ a₃) ∧
    (5 : ℚ) / n = 0.1 + (a₁ * 10^(-2 : ℤ) + a₂ * 10^(-3 : ℤ) + a₃ * 10^(-4 : ℤ)) / (1 - 10^(-3 : ℤ))

theorem valid_repeating_decimals :
  is_valid_repeating_decimal 27 ∧ is_valid_repeating_decimal 37 := by
  sorry

#check valid_repeating_decimals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_repeating_decimals_l29_2947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_students_count_l29_2953

/-- Represents the type of student: good or troublemaker -/
inductive StudentType
  | Good
  | Troublemaker

/-- Represents a class of students -/
structure ClassOfStudents where
  total_students : Nat
  good_students : Nat
  troublemakers : Nat
  h_total : total_students = good_students + troublemakers

/-- Defines the conditions of the problem -/
def satisfies_conditions (c : ClassOfStudents) : Prop :=
  c.total_students = 25 ∧
  (∀ (s : StudentType), 
    (s = StudentType.Good → c.troublemakers > (c.total_students - 1) / 2) ∨
    (s = StudentType.Troublemaker → c.troublemakers ≤ (c.total_students - 1) / 2)) ∧
  (∀ (s : StudentType),
    (s = StudentType.Good → c.troublemakers = 3 * (c.good_students - 1)) ∨
    (s = StudentType.Troublemaker → c.troublemakers ≠ 3 * (c.good_students - 1)))

/-- The main theorem to prove -/
theorem good_students_count (c : ClassOfStudents) : 
  satisfies_conditions c → c.good_students = 5 ∨ c.good_students = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_students_count_l29_2953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l29_2980

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := (1/2)^2

-- State the theorem
theorem relationship_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l29_2980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l29_2943

theorem problem_solution : 
  (∃ x : ℝ, x = Real.sqrt 27 + Real.sqrt (1/3) - Real.sqrt 12 ∧ x = (4 * Real.sqrt 3) / 3) ∧
  (∃ y : ℝ, y = (Real.sqrt 2 + 1)^2 + 2 * Real.sqrt 2 * (Real.sqrt 2 - 1) ∧ y = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l29_2943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_solution_l29_2909

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that the polynomial satisfies the given equation -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P (y / x) + y * P (x / y) = x + y

/-- The identity function on real numbers -/
def IdentityFunction : RealPolynomial := λ x ↦ x

theorem unique_polynomial_solution :
  ∀ P : RealPolynomial, SatisfiesEquation P → P = IdentityFunction := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_solution_l29_2909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l29_2963

/-- The parabola y = x^2 intersects with a line l passing through P(-1, -1) at points P1 and P2.
    Q is a point on the line segment P1P2 satisfying 1/PP1 + 1/PP2 = 2/PQ. -/
theorem parabola_line_intersection (k : ℝ) (x y : ℝ) :
  let l : ℝ → ℝ := λ t ↦ k * (t + 1) - 1
  let P1 := (Real.sqrt (k + 2) - 1, (Real.sqrt (k + 2) - 1)^2)
  let P2 := (-Real.sqrt (k + 2) - 1, (-Real.sqrt (k + 2) - 1)^2)
  let Q := (x, y)
  (∀ t, l t = t^2 → t = P1.1 ∨ t = P2.1) →
  (y = l x) →
  (1 / Real.sqrt ((P1.1 + 1)^2 + (P1.2 + 1)^2) + 
   1 / Real.sqrt ((P2.1 + 1)^2 + (P2.2 + 1)^2) = 
   2 / Real.sqrt ((x + 1)^2 + (y + 1)^2)) →
  ((k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2) ∧
   2 * x - y + 1 = 0 ∧ 
   -Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l29_2963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_shaped_area_l29_2993

/-- The area of a football-shaped region formed by the overlap of two circular arcs
    inside a 3x4 rectangle. -/
theorem football_shaped_area (π : ℝ) (h : π > 0) : ∃ (area : ℝ),
  let rectangle_width : ℝ := 3
  let rectangle_height : ℝ := 4
  let circle1_radius : ℝ := rectangle_width
  let circle2_radius : ℝ := rectangle_height
  let sector1_area : ℝ := (π * circle1_radius^2) / 4
  let sector2_area : ℝ := (π * circle2_radius^2) / 4
  area = (sector1_area + sector2_area) / 2 ∧ area = (13 * π) / 8 := by
  -- Proof goes here
  sorry

#check football_shaped_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_shaped_area_l29_2993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_one_iff_f_max_value_f_max_value_attained_l29_2987

open Real Set

noncomputable def f (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem f_eq_one_iff (x : ℝ) : f x = 1 ↔ ∃ k : ℤ, x = 2 * k * π + π / 2 := by
  sorry

theorem f_max_value : ∀ x : ℝ, f x ≤ 1 := by
  sorry

theorem f_max_value_attained : ∃ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_one_iff_f_max_value_f_max_value_attained_l29_2987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_layer_length_is_correct_l29_2915

/-- The length of one layer of tape when five 2.7-meter sheets are attached with 0.3-meter overlap and divided into 6 layers -/
noncomputable def tape_layer_length : ℝ :=
  let sheet_length : ℝ := 2.7
  let overlap : ℝ := 0.3
  let num_sheets : ℕ := 5
  let num_layers : ℕ := 6
  let total_length : ℝ := sheet_length + (num_sheets - 1 : ℝ) * (sheet_length - overlap)
  total_length / num_layers

theorem tape_layer_length_is_correct : tape_layer_length = 2.05 := by
  -- Unfold the definition of tape_layer_length
  unfold tape_layer_length
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tape_layer_length_is_correct_l29_2915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_calculation_l29_2992

theorem trip_time_calculation (interstate_distance mountain_pass_distance mountain_pass_time : ℝ)
  (h1 : interstate_distance = 75)
  (h2 : mountain_pass_distance = 15)
  (h3 : mountain_pass_time = 45) : ℝ :=
by
  -- Define the speed ratio between interstate and mountain pass
  let speed_ratio : ℝ := 4

  -- Calculate the speed on mountain pass
  let mountain_pass_speed : ℝ := mountain_pass_distance / mountain_pass_time

  -- Calculate the speed on interstate
  let interstate_speed : ℝ := speed_ratio * mountain_pass_speed

  -- Calculate time spent on interstate
  let interstate_time : ℝ := interstate_distance / interstate_speed

  -- Calculate total trip time
  let total_time : ℝ := mountain_pass_time + interstate_time

  -- Prove that the total trip time is 101.25 minutes
  have : total_time = 101.25 := by sorry

  exact total_time

-- Example usage (not for evaluation)
#check trip_time_calculation 75 15 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_calculation_l29_2992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_catch_ratio_l29_2959

/-- Represents the weight of fish caught by each person -/
structure FishCatch where
  ali : ℝ
  peter : ℝ
  joey : ℝ

/-- Conditions of the fishing problem -/
def fishing_problem (c : FishCatch) : Prop :=
  c.ali = 12 ∧
  c.joey = c.peter + 1 ∧
  c.ali + c.peter + c.joey = 25

/-- The theorem to prove -/
theorem fish_catch_ratio (c : FishCatch) :
  fishing_problem c → c.ali / c.peter = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_catch_ratio_l29_2959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_range_l29_2957

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (x + φ)

-- State the theorem
theorem cos_theta_range (φ θ : ℝ) : 
  (0 < φ) → (φ < π/2) → 
  (∀ x, f x φ = f (π/3 - x) φ) →
  (∀ x, -π/4 ≤ x → x ≤ θ → -Real.sqrt 3 ≤ f x φ → f x φ ≤ 2) →
  ((Real.sqrt 2 - Real.sqrt 6) / 4 ≤ Real.cos θ) ∧ (Real.cos θ ≤ Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_range_l29_2957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l29_2942

/-- A dart board with a central triangle -/
structure DartBoard where
  /-- Side length of the regular hexagon -/
  s : ℝ
  /-- Assumption that s is positive -/
  s_pos : s > 0

/-- The probability of a dart landing in the central triangle -/
noncomputable def landing_probability (board : DartBoard) : ℝ :=
  1 / 8

/-- Theorem stating that the probability of landing in the central triangle is 1/8 -/
theorem dart_landing_probability (board : DartBoard) : 
  landing_probability board = 1 / 8 := by
  -- Unfold the definition of landing_probability
  unfold landing_probability
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l29_2942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_range_l29_2933

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) - 2 * Real.cos x

theorem f_value_and_range :
  ∀ x : ℝ,
  x ∈ Set.Icc (Real.pi / 2) Real.pi →
  (∃ y : ℝ, y ∈ Set.Icc 1 2 ∧ f x = y) ∧
  (Real.sin x = 4 / 5 → f x = (4 * Real.sqrt 3 + 3) / 5) :=
by
  sorry

#check f_value_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_range_l29_2933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l29_2994

theorem cosine_sine_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  Real.cos x ^ 2 * (Real.cos x / Real.sin x) + Real.sin x ^ 2 * (Real.sin x / Real.cos x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l29_2994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_golden_section_point_l29_2950

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Length of the stage -/
def stage_length : ℝ := 20

/-- Golden section point -/
def is_golden_section_point (a b c : ℝ) : Prop :=
  (b - a) / (c - a) = φ ∨ (c - a) / (b - a) = φ

/-- Theorem: Length of AC when C is a golden section point of AB -/
theorem length_of_golden_section_point (a b c : ℝ) :
  (b - a = stage_length) →
  (is_golden_section_point a c b) →
  ((c - a = 10 * Real.sqrt 5 - 10) ∨ (c - a = 30 - 10 * Real.sqrt 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_golden_section_point_l29_2950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recommended_water_intake_l29_2944

theorem recommended_water_intake
  (current_intake : ℕ)
  (increase_percentage : ℚ)
  (h1 : current_intake = 20)
  (h2 : increase_percentage = 60 / 100) :
  let recommended_intake := current_intake + (increase_percentage * ↑current_intake).floor
  recommended_intake = 32 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recommended_water_intake_l29_2944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l29_2924

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2*x + 3) / (2*x - 4)

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∃ (A B : ℝ × ℝ) (L : Set (ℝ × ℝ)),
    -- L is a line passing through P
    P ∈ L ∧
    -- A and B are on the graph of f and on L
    (A.1, f A.1) = A ∧ A ∈ L ∧
    (B.1, f B.1) = B ∧ B ∈ L ∧
    -- The dot product (OA + OB) · OP equals 10
    (A.1 - O.1 + B.1 - O.1) * (P.1 - O.1) +
    (A.2 - O.2 + B.2 - O.2) * (P.2 - O.2) = 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l29_2924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_first_six_probability_l29_2968

/-- The probability of rolling a six on a fair six-sided die -/
noncomputable def prob_six : ℝ := 1 / 6

/-- The probability of not rolling a six on a fair six-sided die -/
noncomputable def prob_not_six : ℝ := 1 - prob_six

/-- The number of players in the sequence -/
def num_players : ℕ := 4

/-- The position of Carol in the sequence (0-based index) -/
def carol_position : ℕ := 2

/-- The probability that Carol is the first to roll a six in the repeated sequence -/
noncomputable def prob_carol_first_six : ℝ :=
  (prob_not_six ^ carol_position * prob_six) / (1 - prob_not_six ^ num_players)

theorem carol_first_six_probability :
  prob_carol_first_six = 125 / 671 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_first_six_probability_l29_2968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l29_2906

noncomputable def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity : 
  eccentricity 5 4 = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l29_2906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_two_and_three_l29_2979

def numbers : List ℕ := [28, 35, 40, 45, 53, 10, 78]

def is_multiple (n m : ℕ) : Bool := n % m = 0

theorem multiples_of_two_and_three :
  (numbers.filter (λ n => is_multiple n 2) = [28, 40, 10, 78]) ∧
  (numbers.filter (λ n => is_multiple n 3) = [45, 78]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_two_and_three_l29_2979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l29_2969

theorem sin_beta_value (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : -π/2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5/13)
  (h4 : Real.sin α = 4/5) :
  Real.sin β = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l29_2969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_coloring_theorem_l29_2967

/-- The number of faces in a regular icosahedron -/
def num_faces : ℕ := 12

/-- The number of rotational symmetries around one face of a regular icosahedron -/
def num_rotational_symmetries : ℕ := 5

/-- Two colored icosahedrons are distinguishable if neither can be rotated to look just like the other -/
axiom distinguishable_definition : True

/-- The number of distinguishable ways to construct a regular icosahedron 
    using congruent equilateral triangles of different colors -/
def num_distinguishable_icosahedrons : ℕ := (num_faces - 1).factorial / num_rotational_symmetries

theorem icosahedron_coloring_theorem : 
  num_distinguishable_icosahedrons = 7983360 := by
  -- Unfold the definition of num_distinguishable_icosahedrons
  unfold num_distinguishable_icosahedrons
  -- Simplify the expression
  simp [num_faces, num_rotational_symmetries]
  -- The rest of the proof
  sorry

#eval num_distinguishable_icosahedrons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_coloring_theorem_l29_2967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l29_2918

-- Define the force function
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 10 else 3 * x + 4

-- Define the work function
noncomputable def work (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation :
  work 0 4 = 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l29_2918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_two_x_plus_pi_fourth_l29_2948

open Real

/-- The minimum positive period of tan(2x + π/4) is π/2 -/
theorem tan_period_two_x_plus_pi_fourth : 
  ∃ (T : ℝ), T > 0 ∧ T = π/2 ∧ 
    (∀ x : ℝ, tan (2*x + π/4) = tan (2*(x + T) + π/4)) ∧
    (∀ S : ℝ, S > 0 → (∀ x : ℝ, tan (2*x + π/4) = tan (2*(x + S) + π/4)) → T ≤ S) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_two_x_plus_pi_fourth_l29_2948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hog_cat_ratio_l29_2952

/-- Proves that the ratio of hogs to cats is 3:1 given the conditions -/
theorem hog_cat_ratio (num_hogs : ℕ) (num_cats : ℕ) : 
  num_hogs = 75 →
  (0.6 * (num_cats : ℝ) - 5 = 10) →
  (num_hogs : ℝ) / (num_cats : ℝ) = 3 := by
  sorry

#check hog_cat_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hog_cat_ratio_l29_2952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l29_2946

/-- A function that checks if a number is between 100 and 999, inclusive. -/
def inRange (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- A function that checks if a number is a multiple of 9. -/
def isMultipleOf9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

/-- A function that represents a permutation of digits of a number. -/
def isPermutation (a b : ℕ) : Prop :=
  sorry  -- Definition of permutation

/-- The main theorem stating that there are 390 numbers satisfying the given property. -/
theorem count_numbers_with_property : 
  ∃ S : Finset ℕ, (∀ n ∈ S, inRange n ∧ 
    ∃ m, inRange m ∧ isMultipleOf9 m ∧ isPermutation n m) ∧ 
  S.card = 390 := by
  sorry

#check count_numbers_with_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l29_2946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_pascal_theorem_l29_2910

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := 
  match n, k with
  | 0, _ => if k = 0 then 1 else 0
  | n+1, k => pascal n (k-1) + pascal n k

/-- Predicate for four-digit numbers -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The third smallest four-digit number in Pascal's triangle -/
def third_smallest_four_digit_pascal : ℕ := 1002

/-- Row where the third smallest four-digit number appears -/
def row_of_third_smallest : ℕ := 1001

/-- Position where the third smallest four-digit number appears -/
def pos_of_third_smallest : ℕ := 3

theorem third_smallest_four_digit_pascal_theorem :
  is_four_digit third_smallest_four_digit_pascal ∧
  pascal row_of_third_smallest (pos_of_third_smallest - 1) = third_smallest_four_digit_pascal ∧
  (∀ n k, pascal n k < third_smallest_four_digit_pascal →
    ¬is_four_digit (pascal n k) ∨
    pascal n k = 1000 ∨
    pascal n k = 1001) := by
  sorry

#eval pascal 1001 2  -- Should output 1002

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_pascal_theorem_l29_2910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l29_2929

/-- Definition of an arithmetic sequence -/
def ArithmeticSequence (x y z : ℚ) : Prop :=
  y - x = z - y

/-- Given two sequences {aₙ} and {bₙ} satisfying certain conditions, 
    we prove several properties about them. -/
theorem sequence_properties (a b : ℕ → ℚ) 
  (h1 : a 1 = 0)
  (h2 : b 1 = 2013)
  (h3 : ∀ n : ℕ, ArithmeticSequence (a n) (a (n + 1)) (b n))
  (h4 : ∀ n : ℕ, ArithmeticSequence (a (n + 1)) (b (n + 1)) (b n)) :
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) - b (n + 1) = q * (a n - b n)) ∧ 
  (∃ r : ℚ, ∀ n : ℕ, a (n + 1) + 2 * b (n + 1) = r * (a n + 2 * b n)) ∧
  (∃! c : ℕ, ∀ n : ℕ, a n < ↑c ∧ ↑c < b n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l29_2929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l29_2961

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem x0_value (x0 : ℝ) (h : (deriv (deriv f)) x0 = 2) : x0 = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l29_2961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l29_2996

/-- 
Given two vectors a and b in ℝ², where a = (1, 2) and b = (lambda, 1),
if a and b are perpendicular, then lambda = -2.
-/
theorem perpendicular_vectors_lambda (lambda : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![lambda, 1]
  (∀ i, i < 2 → a i * b i = 0) → lambda = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l29_2996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polinas_number_l29_2982

theorem polinas_number (n : ℕ) : 
  (((n % 11 = 0 : Bool).toNat + (n % 13 = 0 : Bool).toNat + (n < 15 : Bool).toNat + (n % 143 = 0 : Bool).toNat) = 2) →
  (n = 11 ∨ n = 13) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polinas_number_l29_2982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_coefficients_sum_l29_2916

theorem cos_coefficients_sum :
  ∃ (m n p : ℝ),
  (∀ α : ℝ,
    let cos2α := 2 * (Real.cos α)^2 - 1
    let cos4α := 8 * (Real.cos α)^4 - 8 * (Real.cos α)^2 + 1
    let cos6α := 32 * (Real.cos α)^6 - 48 * (Real.cos α)^4 + 18 * (Real.cos α)^2 - 1
    let cos8α := 128 * (Real.cos α)^8 - 256 * (Real.cos α)^6 + 160 * (Real.cos α)^4 - 32 * (Real.cos α)^2 + 1
    let cos10α := m * (Real.cos α)^10 - 1280 * (Real.cos α)^8 + 1120 * (Real.cos α)^6 + n * (Real.cos α)^4 + p * (Real.cos α)^2 - 1
    cos10α = Real.cos (10 * α)) ∧
  m - n + p = 962
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_coefficients_sum_l29_2916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_rs_ratio_is_one_l29_2958

/-- A decagon formed by unit squares -/
structure Decagon :=
  (total_area : ℚ)
  (bisecting_line : ℚ → ℚ)
  (trapezoid_base : ℚ)
  (h : total_area = 12)
  (h' : trapezoid_base = 3)

/-- The ratio of XR to RS in the decagon -/
noncomputable def xr_rs_ratio (d : Decagon) : ℚ :=
  let below_area := d.total_area / 2
  let trapezoid_height := (below_area - 1) / d.trapezoid_base
  let above_height := below_area / d.trapezoid_base
  d.trapezoid_base / d.trapezoid_base

/-- Theorem stating that the ratio of XR to RS is 1 -/
theorem xr_rs_ratio_is_one (d : Decagon) : xr_rs_ratio d = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_rs_ratio_is_one_l29_2958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_from_room_dimensions_l29_2938

noncomputable def room_volume (floor_area width height : ℝ) : ℝ :=
  floor_area * height

noncomputable def cube_edge (volume : ℝ) : ℝ :=
  Real.rpow volume (1/3)

theorem cube_edge_from_room_dimensions (floor_area longer_wall_area shorter_wall_area : ℝ)
  (h1 : floor_area = 20)
  (h2 : longer_wall_area = 15)
  (h3 : shorter_wall_area = 12) :
  ∃ (width height : ℝ), 
    width * height = shorter_wall_area ∧
    (floor_area / width) * height = longer_wall_area ∧
    abs (cube_edge (room_volume floor_area width height) - 3.9149) < 0.0001 := by
  sorry

#check cube_edge_from_room_dimensions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_from_room_dimensions_l29_2938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_14_l29_2964

-- Define the points
variable (F G H I J : ℝ × ℝ)

-- Define the lengths
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Define the right angles
def right_angle_FGH : (G.1 - F.1) * (G.1 - H.1) + (G.2 - F.2) * (G.2 - H.2) = 0 := sorry
def right_angle_GHI : (H.1 - G.1) * (H.1 - I.1) + (H.2 - G.2) * (H.2 - I.2) = 0 := sorry

-- Define the intersection of GI and HF at J
def intersection_at_J : ∃ t s : ℝ, 
  J.1 = G.1 + t * (I.1 - G.1) ∧ 
  J.2 = G.2 + t * (I.2 - G.2) ∧
  J.1 = H.1 + s * (F.1 - H.1) ∧ 
  J.2 = H.2 + s * (F.2 - H.2) := sorry

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  |((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))| / 2

-- State the theorem
theorem area_difference_is_14 :
  triangle_area F G J - triangle_area H J I = 14 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_14_l29_2964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_divides_triangle_equal_area_l29_2902

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a median
noncomputable def median (t : Triangle) (vertex : Fin 3) : ℝ × ℝ :=
  match vertex with
  | 0 => ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)
  | 1 => ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2)
  | 2 => ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

-- Define the area of a triangle
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Theorem: A median divides a triangle into two triangles of equal area
theorem median_divides_triangle_equal_area (t : Triangle) (v : Fin 3) :
  let m := median t v
  triangleArea t.A t.B t.C / 2 = triangleArea t.A t.B m ∧
  triangleArea t.A t.B t.C / 2 = triangleArea t.A m t.C ∧
  triangleArea t.A t.B t.C / 2 = triangleArea m t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_divides_triangle_equal_area_l29_2902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_series_convergence_l29_2921

noncomputable def functional_series (x : ℝ) (n : ℕ) : ℝ := Real.sin (n * x) / Real.exp (n * x)

theorem functional_series_convergence :
  ∀ x : ℝ, Summable (functional_series x) ↔ x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_series_convergence_l29_2921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_and_q_range_when_p_xor_q_l29_2954

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x, 2*x - 5 > 0 → x > m

def q (m : ℝ) : Prop := ∃ x y, (x^2)/(m-1) + (y^2)/(2-m) = 1

-- Theorem for the range of m when p and q are both true
theorem range_when_p_and_q (m : ℝ) : 
  p m ∧ q m ↔ m ∈ Set.Iio 1 ∪ Set.Ioc 2 (5/2) :=
sorry

-- Theorem for the range of m when exactly one of p or q is true
theorem range_when_p_xor_q (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Icc 1 2 ∪ Set.Ioi (5/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_and_q_range_when_p_xor_q_l29_2954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cylinder_volume_difference_l29_2998

/-- The volume of a cylinder given its circumference and height -/
noncomputable def cylinderVolume (circumference height : Real) : Real :=
  (circumference^2 * height) / (4 * Real.pi)

/-- The problem statement -/
theorem paper_cylinder_volume_difference : 
  let chrisVolume := cylinderVolume 7 10
  let danaVolume := cylinderVolume 9 7
  Real.pi * |danaVolume - chrisVolume| = 19.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cylinder_volume_difference_l29_2998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l29_2926

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (sqrt 3 / 3) * cos (2 * x) + sin (x + π/4) ^ 2

-- State the theorem
theorem f_properties :
  -- 1. Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π) ∧
  -- 2. Monotonically increasing in [kπ - 5π/12, kπ + π/12] for any integer k
  (∀ k : ℤ, ∀ x y, x ∈ Set.Icc (k * π - 5*π/12) (k * π + π/12) →
    y ∈ Set.Icc (k * π - 5*π/12) (k * π + π/12) → x < y → f x < f y) ∧
  -- 3. Range is [0, 3/2] when x ∈ [-π/12, 5π/12]
  (Set.range (fun x => f x) ∩ Set.Icc (-π/12 : ℝ) (5*π/12) = Set.Icc 0 (3/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l29_2926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l29_2970

/-- The area of a triangle with vertices at (4, -3), (4, 7), and (9, 7) is 25 square units. -/
theorem triangle_area : ℝ := by
  let A : ℝ × ℝ := (4, -3)
  let B : ℝ × ℝ := (4, 7)
  let C : ℝ × ℝ := (9, 7)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  have : area = 25 := by sorry
  exact 25


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l29_2970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_l29_2974

theorem exponent_sum (a b : ℝ) (h1 : (3 : ℝ)^a = 2) (h2 : (3 : ℝ)^b = 5) : (3 : ℝ)^(a+b) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sum_l29_2974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l29_2903

theorem integer_solutions_equation (x y : ℤ) :
  1 + (2 : ℤ) ^ x.toNat + (2 : ℤ) ^ (2*x.toNat+1) = y^2 ↔ 
  (x = 0 ∧ (y = 2 ∨ y = -2)) ∨ (x = 4 ∧ (y = 23 ∨ y = -23)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l29_2903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l29_2985

theorem unique_solution_system (x y : ℝ) 
  (h1 : (3 : ℝ)^y * 81 = (9 : ℝ)^(x^2)) 
  (h2 : Real.log y = Real.log x - Real.log 0.5) 
  (h3 : y > 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l29_2985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_7350_l29_2935

theorem least_factorial_divisible_by_7350 :
  (∀ k : ℕ, k < 14 → ¬(7350 ∣ Nat.factorial k)) ∧ (7350 ∣ Nat.factorial 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_7350_l29_2935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_common_divisor_of_sum_l29_2940

/-- An arithmetic sequence with positive integer terms and perfect square common difference -/
structure ArithmeticSequence where
  a : ℕ+  -- First term (positive integer)
  d : ℕ   -- Common difference (non-negative integer)
  is_perfect_square : ∃ k : ℕ, d = k * k

/-- Sum of the first 15 terms of an arithmetic sequence -/
def sum_15_terms (seq : ArithmeticSequence) : ℕ :=
  (seq.a : ℕ) * 15 + seq.d * 105

/-- The theorem to be proved -/
theorem greatest_common_divisor_of_sum :
  (∀ seq : ArithmeticSequence, 15 ∣ sum_15_terms seq) ∧
  (∀ n : ℕ, n > 15 → ∃ seq : ArithmeticSequence, ¬(n ∣ sum_15_terms seq)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_common_divisor_of_sum_l29_2940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_inscribed_sphere_l29_2925

/-- The radius of the inscribed sphere in a regular quadrilateral pyramid -/
noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ :=
  (a * (Real.sqrt 5 - 1)) / 4

/-- Theorem: In a regular quadrilateral pyramid with base side length a and height a,
    the radius of the inscribed sphere is a(√5 - 1)/4. -/
theorem regular_quadrilateral_pyramid_inscribed_sphere
  (a : ℝ)
  (h_positive : a > 0)
  (base_side_length height : ℝ)
  (h_base_side : base_side_length = a)
  (h_height : height = a) :
  inscribed_sphere_radius a = (a * (Real.sqrt 5 - 1)) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_inscribed_sphere_l29_2925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_equality_l29_2984

-- Define the points in a 2D Euclidean space
variable (A B C D E F P O : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of opposite sides
def opposite_sides_intersect (A B C D E F : EuclideanPlane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D P : EuclideanPlane) : Prop := sorry

-- Define the perpendicular line
def is_perpendicular (P O E F : EuclideanPlane) : Prop := sorry

-- Define the angle equality
def angles_equal (A B C D O : EuclideanPlane) : Prop := sorry

-- Theorem statement
theorem quadrilateral_angle_equality 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : opposite_sides_intersect A B C D E F)
  (h3 : diagonals_intersect A B C D P)
  (h4 : is_perpendicular P O E F) :
  angles_equal A B C D O := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_equality_l29_2984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_power_function_l29_2939

-- Define the power function
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem tangent_line_of_power_function (α : ℝ) :
  f α (1/4) = 1/2 →
  ∃ m b, m * (1/4) + b = 1/2 ∧
         m = (deriv (f α)) (1/4) ∧
         ∀ x y, y = m * x + b ↔ 4*x - 4*y + 1 = 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_power_function_l29_2939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_area_l29_2995

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-1, 0)
noncomputable def F2 : ℝ × ℝ := (1, 0)

-- Define the line passing through F2
def line_l (x : ℝ) : Prop := x = 1

-- Define the intersection points P and Q
noncomputable def P : ℝ × ℝ := (1, 3/2)
noncomputable def Q : ℝ × ℝ := (1, -3/2)

-- Define the triangle F1PQ
def triangle_F1PQ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p = F1 ∨ p = P ∨ p = Q}

-- Define the inscribed circle of triangle F1PQ
def inscribed_circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Theorem statement
theorem max_inscribed_circle_area :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    inscribed_circle center radius ⊆ triangle_F1PQ ∧
    ∀ (other_center : ℝ × ℝ) (other_radius : ℝ),
      inscribed_circle other_center other_radius ⊆ triangle_F1PQ →
      π * radius^2 ≥ π * other_radius^2 ∧
      π * radius^2 = 9 * π / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_circle_area_l29_2995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_is_19_l29_2989

/-- Represents a configuration of pieces on a 6x6 checkerboard -/
def Board := Fin 6 → Fin 6 → Bool

/-- Counts the number of pieces in the same row or column as the given position, excluding itself -/
def countPiecesInRowCol (b : Board) (row col : Fin 6) : Nat :=
  (Finset.sum (Finset.univ.filter (fun i => i ≠ col ∧ b row i)) (fun _ => 1)) +
  (Finset.sum (Finset.univ.filter (fun i => i ≠ row ∧ b i col)) (fun _ => 1))

/-- Checks if the board satisfies the condition for a given number n -/
def satisfiesCondition (b : Board) (n : Nat) : Prop :=
  ∃ (row col : Fin 6), b row col ∧ countPiecesInRowCol b row col = n

/-- Counts the total number of pieces on the board -/
def totalPieces (b : Board) : Nat :=
  Finset.sum Finset.univ fun row => Finset.sum Finset.univ fun col => if b row col then 1 else 0

/-- The main theorem stating that the minimum number of pieces is 19 -/
theorem min_pieces_is_19 :
  ∃ (b : Board),
    (∀ n, 2 ≤ n ∧ n ≤ 10 → satisfiesCondition b n) ∧
    totalPieces b = 19 ∧
    (∀ b', (∀ n, 2 ≤ n ∧ n ≤ 10 → satisfiesCondition b' n) → totalPieces b' ≥ 19) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_is_19_l29_2989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l29_2912

-- Define the set of digits
def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define a function to check if a number is odd
def isOdd (n : Nat) : Bool := n % 2 = 1

-- Define a function to count odd digits in a number
def countOddDigits (n : Nat) : Nat :=
  (n.digits 10).filter isOdd |>.length

-- Define the set of valid three-digit numbers
def validNumbers : Finset Nat :=
  Finset.filter (fun n =>
    n ≥ 100 ∧ n < 1000 ∧
    (n.digits 10).toFinset.card = 3 ∧
    (n.digits 10).toFinset ⊆ digits ∧
    countOddDigits n ≤ 1
  ) (Finset.range 1000)

-- Theorem statement
theorem valid_numbers_count : validNumbers.card = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l29_2912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_max_value_l29_2908

/-- Given a function f with symmetry axis, find max value of related function g -/
theorem symmetry_axis_max_value (a : ℝ) :
  (∀ x : ℝ, (Real.sin x + a * Real.cos x) = (Real.sin (10 * Real.pi / 3 - x) + a * Real.cos (10 * Real.pi / 3 - x))) →
  (∃ x : ℝ, ∀ y : ℝ, a * Real.sin y + Real.cos y ≤ a * Real.sin x + Real.cos x) →
  (∃ x : ℝ, a * Real.sin x + Real.cos x = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_max_value_l29_2908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l29_2945

def is_valid (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 10 * k + 2 ∧ 2 * n = 2 * 10^(Nat.log 10 n) + k

theorem smallest_valid_number :
  (∀ m : ℕ, m < 105263157894736842 → ¬ is_valid m) ∧ is_valid 105263157894736842 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l29_2945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_6_l29_2941

/-- Triathlon race parameters -/
structure Triathlon where
  swim_dist : ℝ
  bike_dist : ℝ
  run_dist : ℝ
  swim_speed : ℝ
  bike_speed : ℝ
  run_speed : ℝ

/-- Calculate the average speed for a triathlon race -/
noncomputable def averageSpeed (t : Triathlon) : ℝ :=
  let total_dist := t.swim_dist + t.bike_dist + t.run_dist
  let total_time := t.swim_dist / t.swim_speed + t.bike_dist / t.bike_speed + t.run_dist / t.run_speed
  total_dist / total_time

/-- Theorem stating that the average speed is approximately 6 km/h -/
theorem triathlon_average_speed_approx_6 :
  let t : Triathlon := {
    swim_dist := 1.5,
    bike_dist := 3,
    run_dist := 2,
    swim_speed := 2,
    bike_speed := 25,
    run_speed := 8
  }
  ∃ ε > 0, |averageSpeed t - 6| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_6_l29_2941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_exp_greater_than_square_l29_2962

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x

theorem function_properties :
  (f a 0 = 1) → ((deriv (f a)) 0 = -1) →
  (a = 2 ∧ ∃ x₀, ∀ x, f a x ≥ f a x₀ ∧ f a x₀ = 2 - Real.log 4) :=
sorry

theorem exp_greater_than_square (x : ℝ) :
  x > 0 → x^2 < Real.exp x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_exp_greater_than_square_l29_2962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_properties_l29_2917

/-- Properties of a cube with face diagonal 3 cm -/
theorem cube_properties (d : ℝ) (h : d = 3) : 
  ∃ (a volume space_diagonal : ℝ),
    a = d / Real.sqrt 2 ∧
    volume = a ^ 3 ∧
    space_diagonal = a * Real.sqrt 3 ∧
    volume = 27 * Real.sqrt 2 / 4 ∧ 
    space_diagonal = 3 * Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_properties_l29_2917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l29_2975

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the circle (renamed to avoid conflict with built-in circle)
def custom_circle (x y : ℝ) : Prop := (x-1)^2 + y^2 = 2

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Helper function for area of triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_and_circle_properties :
  -- Given conditions
  (|F₁.1 - F₂.1| = 2) →
  (ellipse 1 (3/2)) →
  -- Conclusions
  (∀ x y, ellipse x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ l : ℝ → ℝ → Prop,
    (∀ x y, l x y → ellipse x y) ∧
    (l F₁.1 F₁.2) ∧
    (∃ A B : ℝ × ℝ,
      l A.1 A.2 ∧ l B.1 B.2 ∧
      area_triangle A F₂ B = 12 * Real.sqrt 2 / 7 ∧
      (∀ x y, custom_circle x y ↔ (x-1)^2 + y^2 = 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l29_2975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l29_2936

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-1, -1)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 2

-- Define a function to count common tangents (placeholder)
def number_of_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  (d < radius1 + radius2) →
  (∃ (n : ℕ), n = 2 ∧ n = (number_of_common_tangents C1 C2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l29_2936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lagrange_mean_value_for_g_l29_2932

-- Define the function g(x) = ln x + x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x

-- Define the derivative of g(x)
noncomputable def g' (x : ℝ) : ℝ := 1 / x + 1

theorem lagrange_mean_value_for_g :
  ∃ c : ℝ, c ∈ Set.Ioo 1 2 ∧ 
  g' c * (2 - 1) = g 2 - g 1 ∧
  c = 1 / Real.log 2 := by
  sorry

#check lagrange_mean_value_for_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lagrange_mean_value_for_g_l29_2932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_nonempty_intersection_count_l29_2955

def A : Finset ℕ := Finset.range 11 \ {0}
def B : Finset ℕ := {1, 2, 3, 4}

theorem subsets_with_nonempty_intersection_count :
  (Finset.filter (fun C => C ⊆ A ∧ (C ∩ B).Nonempty) (Finset.powerset A)).card = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_with_nonempty_intersection_count_l29_2955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_system_l29_2973

theorem integer_solutions_system : 
  {(x, y, z, t) : ℤ × ℤ × ℤ × ℤ | 
    x * z - 2 * y * t = 3 ∧ x * t + y * z = 1} = 
  {(1, 0, 3, 1), (-1, 0, -3, -1), (3, 1, 1, 0), (-3, -1, -1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_system_l29_2973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pooja_crossing_time_l29_2986

/-- Calculates the time taken for Pooja to cross Train B --/
noncomputable def time_to_cross (length_A length_B speed_A speed_B speed_P : ℝ) : ℝ :=
  let speed_A_ms := speed_A * 1000 / 3600
  let speed_B_ms := speed_B * 1000 / 3600
  let speed_P_ms := speed_P * 1000 / 3600
  let relative_speed := speed_A_ms + speed_P_ms
  length_B / relative_speed

/-- The time taken for Pooja to cross Train B is approximately 9.78 seconds --/
theorem pooja_crossing_time :
  let length_A := (225 : ℝ)
  let length_B := (150 : ℝ)
  let speed_A := (54 : ℝ)
  let speed_B := (36 : ℝ)
  let speed_P := (1.2 : ℝ)
  abs (time_to_cross length_A length_B speed_A speed_B speed_P - 9.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pooja_crossing_time_l29_2986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l29_2991

theorem division_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) : 
  (x : ℝ) / (y : ℝ) = 96.12 → 
  x % y = 9 → 
  y = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l29_2991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l29_2904

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

/-- The tangent line function -/
noncomputable def tangent_line (x : ℝ) : ℝ := (1/4) * (x - 1) + 1/2

theorem tangent_line_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  (∀ x : ℝ, ∃ ε > 0, |tangent_line x - f x - ((x - x₀) * (tangent_line x₀ - f x₀))| ≤ ε * |x - x₀|) ∧
  (∀ x : ℝ, x - 4 * tangent_line x + 1 = 0) := by
  sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l29_2904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l29_2907

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
  4 * Real.sin ((t.A + t.B) / 2) ^ 2 - Real.cos (2 * t.C) = 7/2 ∧
  t.c = Real.sqrt 7

-- Helper function to calculate area (not part of the theorem, just for completeness)
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = π/3 ∧ 
  (∀ t' : Triangle, triangle_conditions t' → 
    area t' ≤ 7 * Real.sqrt 3 / 4) ∧
  (∃ t' : Triangle, triangle_conditions t' ∧ 
    area t' = 7 * Real.sqrt 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l29_2907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_C_largest_area_l29_2914

noncomputable section

-- Define the areas of basic shapes
def unit_square_area : ℝ := 1
def right_triangle_area : ℝ := 0.5
def half_unit_square_area : ℝ := 0.5
noncomputable def equilateral_triangle_area : ℝ := Real.sqrt 3 / 4

-- Define the areas of polygons
def polygon_A_area : ℝ := 3 * unit_square_area + 3 * right_triangle_area
def polygon_B_area : ℝ := 2 * unit_square_area + 4 * right_triangle_area + half_unit_square_area
noncomputable def polygon_C_area : ℝ := 4 * unit_square_area + 2 * equilateral_triangle_area
def polygon_D_area : ℝ := 5 * right_triangle_area + half_unit_square_area
noncomputable def polygon_E_area : ℝ := 3 * equilateral_triangle_area + 2 * half_unit_square_area

theorem polygon_C_largest_area :
  polygon_C_area > polygon_A_area ∧
  polygon_C_area > polygon_B_area ∧
  polygon_C_area > polygon_D_area ∧
  polygon_C_area > polygon_E_area :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_C_largest_area_l29_2914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_proof_l29_2999

theorem xy_value_proof (x y : ℝ) : 
  (4 : ℝ) ^ ((x + y) ^ 2) / (4 : ℝ) ^ ((x - y) ^ 2) = 256 → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_proof_l29_2999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_decimal_part_minus_sqrt_17_l29_2997

-- Define the decimal part of a real number
noncomputable def decimal_part (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem sqrt_17_decimal_part_minus_sqrt_17 :
  let a := decimal_part (Real.sqrt 17)
  a - Real.sqrt 17 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_decimal_part_minus_sqrt_17_l29_2997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_equivalence_l29_2934

-- Define the types for planes and lines
variable (Plane Line Point : Type)

-- Define the perpendicular relation for planes and lines
variable (perpPlane : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- Define the "within" relation for lines and planes
variable (within : Line → Plane → Prop)

-- Define the intersection relation
variable (intersect : Plane → Plane → Line → Prop)
variable (intersectAt : Line → Line → Line → Point → Prop)

-- Given conditions
variable (α β : Plane) (a b c : Line) (P : Point)

-- Theorem statement
theorem line_perp_equivalence
  (h1 : perpPlane α β)
  (h2 : intersect α β c)
  (h3 : within a α)
  (h4 : within b β)
  (h5 : ¬ perpLine a c)
  (h6 : intersectAt a b c P) :
  perpLine b c ↔ perpLine b a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_equivalence_l29_2934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_in_anomaly_value_l29_2965

/-- Acceleration due to gravity inside an anomaly --/
def acceleration_in_anomaly (v₀ g S : ℝ) (α : ℝ) : ℝ :=
  250

/-- Theorem stating the acceleration in the anomaly given initial conditions --/
theorem acceleration_in_anomaly_value
  (v₀ : ℝ) (α : ℝ) (g : ℝ) (S : ℝ)
  (h_v₀ : v₀ = 10)
  (h_α : α = Real.pi / 6)  -- 30 degrees in radians
  (h_g : g = 10)
  (h_S : S = 3 * Real.sqrt 3) :
  acceleration_in_anomaly v₀ g S α = 250 := by
  sorry

#eval acceleration_in_anomaly 10 10 (3 * Real.sqrt 3) (Real.pi / 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_in_anomaly_value_l29_2965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_identity_l29_2981

theorem combinatorial_identity 
  (n m k : ℕ) 
  (h1 : 1 ≤ k) 
  (h2 : k < m) 
  (h3 : m ≤ n) : 
  (Finset.range (k + 1)).sum (λ i ↦ Nat.choose k i * Nat.choose n (m - i)) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorial_identity_l29_2981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_pattern_l29_2928

-- Define the property of being an imaginary number
def IsImaginary (z : ℂ) : Prop := ∃ y : ℝ, z = Complex.I * y ∧ y ≠ 0

-- Define the property of two complex numbers being incomparable
def Incomparable (z1 z2 : ℂ) : Prop := ¬(z1.re < z2.re ∨ (z1.re = z2.re ∧ z1.im < z2.im) ∨ z1 = z2 ∨ z2.re < z1.re ∨ (z2.re = z1.re ∧ z2.im < z1.im))

-- State the theorem
theorem syllogism_pattern (z1 z2 : ℂ) :
  (∀ w : ℂ, IsImaginary w → ∀ v : ℂ, IsImaginary v → Incomparable w v) →
  IsImaginary z1 ∧ IsImaginary z2 →
  Incomparable z1 z2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_pattern_l29_2928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_approx_l29_2949

/-- The total number of marbles in a collection with red, blue, green, and yellow marbles -/
noncomputable def total_marbles (r : ℝ) : ℝ :=
  let b := r / 1.3  -- 30% more red than blue
  let g := 1.5 * r  -- 50% more green than red
  let y := 0.8 * g  -- 20% fewer yellow than green
  r + b + g + y

/-- Theorem stating that the total number of marbles is approximately 4.47 times the number of red marbles -/
theorem total_marbles_approx (r : ℝ) (h : r > 0) :
  ∃ ε > 0, |total_marbles r - 4.47 * r| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_approx_l29_2949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l29_2972

/-- Sum of a geometric sequence with n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a r : ℝ) (h_r : r ≠ 1) :
  geometricSum a r 1010 = 300 →
  geometricSum a r 2020 = 540 →
  geometricSum a r 3030 = 732 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l29_2972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_intersection_not_integer_l29_2983

theorem triangle_tangent_intersection_not_integer (a b c : ℕ+) 
  (h_coprime : Nat.Coprime a.val b.val ∧ Nat.Coprime b.val c.val ∧ Nat.Coprime c.val a.val) : 
  ∃ (A B C D : ℝ × ℝ), 
    let d_AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    let d_BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let d_CA := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    let circle := { P : ℝ × ℝ | (P.1 - A.1)^2 + (P.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 }
    let line_BC := { P : ℝ × ℝ | (P.2 - B.2) * (C.1 - B.1) = (P.1 - B.1) * (C.2 - B.2) }
    let tangent_A := { P : ℝ × ℝ | (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 }
    d_AB = a ∧ d_BC = b ∧ d_CA = c ∧
    D ∈ line_BC ∧ D ∈ tangent_A ∧
    ¬ ∃ (n : ℤ), Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_intersection_not_integer_l29_2983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l29_2923

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inequality condition
variable (h : ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≥ |g x₁ - g x₂|)

-- Define evenness for a function
def IsEven (h : ℝ → ℝ) : Prop := ∀ x : ℝ, h (-x) = h x

-- Define the condition for P₂
def P₂Condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → x * (deriv f x) > 0

-- Define monotonicity
def IsMonotonic (h : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → h x < h y ∨ h x > h y

-- State the theorem
theorem problem_statement (f g : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≥ |g x₁ - g x₂|) :
  (IsEven f → IsEven g) ∧ 
  ¬(P₂Condition f → IsMonotonic (λ x ↦ f x + g x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l29_2923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_four_l29_2971

-- Define the expression as noncomputable
noncomputable def logarithmic_expression : ℝ := 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + 2 ^ (Real.log 3 / Real.log 4)

-- State the theorem
theorem logarithmic_expression_equals_four : logarithmic_expression = 4 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_four_l29_2971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_sequence_sum_l29_2966

/-- The sum of an infinite geometric sequence with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric sequence with first term -3/4 and common ratio m,
    if the sum of all terms is equal to m, then m = -1/2 -/
theorem infinite_geometric_sequence_sum (m : ℝ) :
  infiniteGeometricSum (-3/4) m = m → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_sequence_sum_l29_2966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_seconds_l29_2927

/-- The displacement function of a particle moving along a straight line -/
noncomputable def displacement (t : ℝ) : ℝ := (1/3) * t^3 - (3/2) * t^2 + 2 * t

/-- The instantaneous velocity of the particle at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv displacement t

theorem velocity_at_two_seconds :
  instantaneous_velocity 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_seconds_l29_2927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l29_2976

/-- Represents the driving scenario of Shelby --/
structure DrivingScenario where
  speed_sun : ℚ  -- Speed when not raining (miles per hour)
  speed_rain : ℚ  -- Speed when raining (miles per hour)
  total_time : ℚ  -- Total driving time (minutes)
  total_distance : ℚ  -- Total distance driven (miles)

/-- Calculates the time driven in rain given a driving scenario --/
def time_in_rain (scenario : DrivingScenario) : ℚ :=
  let speed_sun_per_minute := scenario.speed_sun / 60
  let speed_rain_per_minute := scenario.speed_rain / 60
  (scenario.total_distance - speed_sun_per_minute * scenario.total_time) /
    (speed_rain_per_minute - speed_sun_per_minute)

/-- Theorem stating that given Shelby's driving scenario, she drove 24 minutes in the rain --/
theorem shelby_rain_time :
  let scenario := DrivingScenario.mk 45 30 60 24
  time_in_rain scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l29_2976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_eq_189_l29_2960

/-- A function that checks if a 3-digit number satisfies the condition that its units digit is at least three times its tens digit -/
def satisfiesCondition (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n % 10) ≥ 3 * ((n / 10) % 10)

/-- The count of 3-digit numbers satisfying the condition -/
def countSatisfyingNumbers : ℕ :=
  (Finset.range 1000).filter (fun n => satisfiesCondition n) |>.card

/-- Theorem stating that the count of numbers satisfying the condition is 189 -/
theorem count_satisfying_numbers_eq_189 : countSatisfyingNumbers = 189 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_eq_189_l29_2960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_light_position_maximizes_illumination_l29_2920

noncomputable def optimal_light_position (r R d : ℝ) : ℝ :=
  if d ≥ r + R * (R / r).sqrt ∧ d ≥ R + r * (r / R).sqrt
  then d / (1 + (R^3 / r^3).sqrt)
  else r

theorem optimal_light_position_maximizes_illumination
  (r R d x : ℝ)
  (h1 : d > r + R)
  (h2 : r ≤ x)
  (h3 : x ≤ d - R)
  (h4 : r > 0)
  (h5 : R > 0) :
  let F := λ y : ℝ ↦ 2 * Real.pi * (R^2 + r^2 - r^3 / y - R^3 / (d - y))
  F (optimal_light_position r R d) ≥ F x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_light_position_maximizes_illumination_l29_2920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_l29_2901

/-- An arithmetic sequence with first term 3 and last term 31 -/
def arithmetic_sequence (a : List ℕ) : Prop :=
  a.head? = some 3 ∧ a.getLast? = some 31 ∧
  ∃ d : ℕ, ∀ i : ℕ, i + 1 < a.length → a[i + 1]! - a[i]! = d

theorem sum_of_xyz (a : List ℕ) (x y z : ℕ) (hseq : arithmetic_sequence a) 
    (hx : x ∈ a) (hy : y ∈ a) (hz : z ∈ a) 
    (hxyz : a.indexOf x < a.indexOf y ∧ a.indexOf y < a.indexOf z) :
  x + y + z = 81 := by
  sorry

#check sum_of_xyz

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_l29_2901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l29_2988

-- Define the hyperbola
def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the line y = x - 2
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

-- Define area_quadrilateral function (not implemented)
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem hyperbola_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = Real.tan (30 * π / 180)) →
  (∃ (c : ℝ), c - a = 2 - Real.sqrt 3) →
  (hyperbola a b = hyperbola (Real.sqrt 3) 1) ∧
  (∃ (A B C D : ℝ × ℝ),
    A ∈ hyperbola (Real.sqrt 3) 1 ∧ A ∈ line ∧
    B ∈ hyperbola (Real.sqrt 3) 1 ∧ B ∈ line ∧
    C ∈ hyperbola (Real.sqrt 3) 1 ∧
    D ∈ hyperbola (Real.sqrt 3) 1 ∧
    (∀ (p : ℝ × ℝ), p ∈ line → (A.1 - p.1) * (p.2 - A.2) = 0) ∧
    (∀ (p : ℝ × ℝ), p ∈ line → (B.1 - p.1) * (p.2 - B.2) = 0) ∧
    area_quadrilateral A B C D = 12 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l29_2988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_four_heads_l29_2911

def biased_coin_prob (p : ℚ) : Prop :=
  (Nat.choose 6 2 : ℚ) * p^2 * (1 - p)^4 = (Nat.choose 6 3 : ℚ) * p^3 * (1 - p)^3

theorem biased_coin_four_heads :
  ∃ p : ℚ, biased_coin_prob p ∧
    (Nat.choose 6 4 : ℚ) * p^4 * (1 - p)^2 = 240 / 1453 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_four_heads_l29_2911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l29_2930

/-- Represents the total worth of the stock in Rupees -/
def total_worth : ℝ := 15000

/-- Percentage of stock sold at profit -/
def profit_stock_percentage : ℝ := 0.20

/-- Profit percentage on the profit_stock_percentage -/
def profit_percentage : ℝ := 0.10

/-- Percentage of stock sold at loss -/
def loss_stock_percentage : ℝ := 0.80

/-- Loss percentage on the loss_stock_percentage -/
def loss_percentage : ℝ := 0.05

/-- Overall loss in Rupees -/
def overall_loss : ℝ := 300

theorem stock_worth_calculation :
  profit_stock_percentage * profit_percentage * total_worth -
  loss_stock_percentage * loss_percentage * total_worth = overall_loss := by
  sorry

#eval total_worth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_calculation_l29_2930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l29_2931

theorem absolute_value_expression (x : ℤ) (h : x = 2017) : 
  (abs (abs (abs x + x) - abs x) + x) = 4034 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l29_2931
