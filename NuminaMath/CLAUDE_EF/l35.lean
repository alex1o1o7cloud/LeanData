import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l35_3572

theorem unique_n_value (n k x : ℕ) (h1 : k ≥ 2) 
  (h2 : 2^(2*n+1) + 2^n + 1 = x^k) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_value_l35_3572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l35_3571

theorem power_equation (m n : ℝ) (h1 : (10 : ℝ)^m = 3) (h2 : (10 : ℝ)^n = 2) : 
  (10 : ℝ)^(2*m + 3*n) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l35_3571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l35_3507

/-- Given two lines p and q that intersect at (1, 5), prove that the slope of q is 4. -/
theorem intersection_slope (j : ℝ) (line_p line_q : Set (ℝ × ℝ)) : 
  (∀ x y, y = 2 * x + 3 → (x, y) ∈ line_p) →  -- Line p equation
  (∀ x y, y = j * x + 1 → (x, y) ∈ line_q) →  -- Line q equation
  (1, 5) ∈ line_p →                           -- Intersection point on p
  (1, 5) ∈ line_q →                           -- Intersection point on q
  j = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l35_3507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_is_90_l35_3536

/-- The length of the first platform in meters. -/
noncomputable def first_platform_length : ℝ := 90

/-- The length of the second platform in meters. -/
noncomputable def second_platform_length : ℝ := 120

/-- The length of the train in meters. -/
noncomputable def train_length : ℝ := 30

/-- The time taken to cross the first platform in seconds. -/
noncomputable def time_first_platform : ℝ := 12

/-- The time taken to cross the second platform in seconds. -/
noncomputable def time_second_platform : ℝ := 15

/-- The speed of the train in meters per second. -/
noncomputable def train_speed : ℝ := (second_platform_length + train_length) / time_second_platform

theorem first_platform_length_is_90 :
  first_platform_length = (train_speed * time_first_platform) - train_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_is_90_l35_3536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_ln3_div_2_l35_3559

open Real MeasureTheory Interval

-- Define the regions A and B
def region_A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 / p.1}

def region_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the probability
noncomputable def probability : ℝ :=
  (volume region_A).toReal / (volume region_B).toReal

-- Theorem statement
theorem probability_is_ln3_div_2 : probability = log 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_ln3_div_2_l35_3559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_abs_cos_l35_3566

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x * abs (Real.cos x)

-- State the theorem
theorem min_period_sin_abs_cos :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = 2 * π :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_abs_cos_l35_3566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l35_3577

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define point A
def point_A : ℝ × ℝ := (-1, 8)

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Statement of the theorem
theorem min_distance_sum :
  ∃ (min : ℝ), min = 9 ∧
  ∀ (P : ℝ × ℝ), point_on_parabola P →
    distance P point_A + distance P focus ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l35_3577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MA_is_2_sqrt_5_l35_3508

noncomputable def M : ℝ × ℝ := (-3, -1)
noncomputable def A : ℝ × ℝ := (1, 1)

noncomputable def tan_function (x : ℝ) : ℝ := Real.tan (Real.pi * x / 4)

theorem distance_MA_is_2_sqrt_5 :
  ∀ (x : ℝ), x ∈ Set.Ioo (-2) 2 →
  tan_function x = 1 →
  A.1 = x →
  A.2 = 1 →
  Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MA_is_2_sqrt_5_l35_3508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l35_3592

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l35_3592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barn_paint_area_l35_3511

/-- Represents the dimensions of a barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn -/
noncomputable def totalPaintArea (dim : BarnDimensions) : ℝ :=
  let frontBackWallArea := 2 * (2 * dim.width * dim.height)
  let sideWallArea := 2 * (2 * dim.length * dim.height)
  let dividingWallArea := 2 * (dim.length / 2 * dim.height)
  let ceilingArea := dim.width * dim.length
  frontBackWallArea + sideWallArea + dividingWallArea + ceilingArea

/-- Theorem stating that the total paint area for the given barn dimensions is 1020 sq yd -/
theorem barn_paint_area :
  let dim : BarnDimensions := ⟨15, 20, 8⟩
  totalPaintArea dim = 1020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barn_paint_area_l35_3511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l35_3546

theorem system_solution (x y z : ℝ) : 
  (x^2 + 7*y + 2 = 2*z + 4*Real.sqrt (7*x - 3)) ∧
  (y^2 + 7*z + 2 = 2*x + 4*Real.sqrt (7*y - 3)) ∧
  (z^2 + 7*x + 2 = 2*y + 4*Real.sqrt (7*z - 3)) →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l35_3546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l35_3505

/-- The circumference of the base of a cone formed from a 270° sector of a circle with radius 4 inches is 6π inches. -/
theorem cone_base_circumference :
  let r : ℝ := 4  -- radius of the original circle in inches
  let full_angle : ℝ := 360  -- full circle angle in degrees
  let sector_angle : ℝ := 270  -- angle of the sector used to form the cone in degrees
  let original_circumference : ℝ := 2 * π * r  -- circumference of the original circle
  let cone_base_fraction : ℝ := sector_angle / full_angle  -- fraction of the original circumference that forms the cone's base
  let cone_base_circumference : ℝ := cone_base_fraction * original_circumference
  cone_base_circumference = 6 * π := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l35_3505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l35_3509

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 64 - y^2 / 36 = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem hyperbola_foci_distance 
  (x y xF₁ yF₁ xF₂ yF₂ : ℝ) 
  (h_hyperbola : is_on_hyperbola x y) 
  (h_distance : distance x y xF₁ yF₁ = 17) : 
  distance x y xF₂ yF₂ = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l35_3509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_1728_l35_3533

/-- The capacity of a tank with a leak and an inlet pipe. -/
noncomputable def tank_capacity (empty_time : ℝ) (fill_rate : ℝ) (combined_empty_time : ℝ) : ℝ :=
  let leak_rate := 1 / empty_time
  let inlet_rate := fill_rate * 60  -- Convert from litres per minute to litres per hour
  let net_empty_rate := 1 / combined_empty_time
  (inlet_rate / (leak_rate - net_empty_rate))

/-- Theorem stating the capacity of the tank under given conditions. -/
theorem tank_capacity_is_1728 :
  tank_capacity 8 6 12 = 1728 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_1728_l35_3533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l35_3548

noncomputable section

-- Define the given numbers
def numbers : List ℝ := [3/5, Real.sqrt 9, Real.pi, 3.14, -(Real.rpow 27 (1/3)), 0, -5.12345, -Real.sqrt 3]

-- Define the sets
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def irrational_numbers : Set ℝ := {x | x ∉ rational_numbers}
def positive_real_numbers : Set ℝ := {x | x > 0}

-- State the theorem
theorem correct_categorization :
  {3/5, Real.sqrt 9, 3.14, -(Real.rpow 27 (1/3)), 0} ⊆ rational_numbers ∧
  {Real.pi, -5.12345, -Real.sqrt 3} ⊆ irrational_numbers ∧
  {3/5, Real.sqrt 9, Real.pi, 3.14} ⊆ positive_real_numbers :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_categorization_l35_3548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cdf_continuity_and_probability_l35_3557

noncomputable section

def F (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 0
  else if x ≤ Real.exp 1 then a * Real.log x
  else 1

theorem cdf_continuity_and_probability (a : ℝ) :
  (∀ x, ContinuousAt (F a) x) →
  F a (Real.exp 1) = 1 →
  a = 1 ∧ 
  F a (Real.exp (1/2)) - F a (Real.exp (-1/3)) = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cdf_continuity_and_probability_l35_3557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l35_3537

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (|x + 1|)

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l35_3537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l35_3532

theorem external_tangent_length (A B : EuclideanSpace ℝ (Fin 2)) (r₁ r₂ : ℝ) (D E : EuclideanSpace ℝ (Fin 2)) :
  (r₁ = 10) →
  (r₂ = 3) →
  (dist A B = r₁ + r₂) →
  (dist A D = r₁) →
  (dist B E = r₂) →
  (dist D E)^2 = 4 * ((r₁ + r₂)^2 - r₁^2 - r₂^2) →
  dist A E = 2 * Real.sqrt 55 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l35_3532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l35_3558

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 7, 0)
noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define the center
def center : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem ellipse_property (x y : ℝ) :
  is_on_ellipse x y →
  x ≠ 4 ∧ x ≠ -4 →
  let p := (x, y)
  distance p left_focus * distance p right_focus + distance p center ^ 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l35_3558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l35_3576

noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_angle (α : ℝ) : 
  (∃ t1 t2 : ℝ, 
    let p1 := line_l t1 α
    let p2 := line_l t2 α
    p1.1^2 + p1.2^2 = 4 * p1.1 ∧ 
    p2.1^2 + p2.2^2 = 4 * p2.1 ∧
    distance p1 p2 = Real.sqrt 14) →
  α = π/4 ∨ α = 3*π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l35_3576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_simplification_l35_3547

theorem arithmetic_simplification : (7 : ℝ) + (-4.5) - (-3) - 5.5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_simplification_l35_3547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l35_3538

-- Define the parameters
variable (α β r : ℝ)

-- Define the conditions
variable (h_α_acute : 0 < α ∧ α < π/2)
variable (h_β_positive : β > 0)
variable (h_r_positive : r > 0)

-- Define the volume function
noncomputable def pyramid_volume (α β r : ℝ) : ℝ :=
  (4 * r^3 * Real.tan β) / (3 * Real.sin α * (Real.tan (β/2))^3)

-- State the theorem
theorem pyramid_volume_formula (α β r : ℝ) 
  (h_α_acute : 0 < α ∧ α < π/2) (h_β_positive : β > 0) (h_r_positive : r > 0) :
  pyramid_volume α β r = (4 * r^3 * Real.tan β) / (3 * Real.sin α * (Real.tan (β/2))^3) :=
by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l35_3538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_cube_of_999_l35_3510

-- Define 999 as 10³ - 1
def n : ℕ := 10^3 - 1

-- Theorem to prove
theorem zeros_in_cube_of_999 : 
  (n^3).repr.count '0' = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_cube_of_999_l35_3510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_integer_in_sequence_l35_3524

theorem least_odd_integer_in_sequence (seq : List Int) : 
  seq.length = 16 ∧ 
  (∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i < seq.get! j) ∧
  (∀ i, 0 < i → i < seq.length → seq.get! i = seq.get! (i-1) + 2) ∧
  ((seq.sum : Int) / seq.length = 414) →
  seq.get! 0 = 399 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_integer_in_sequence_l35_3524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l35_3583

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => a n / (3 * a n + 1)

theorem sequence_properties :
  (a 1 = 2 / 7) ∧
  (∀ n : ℕ, a n = 2 / (6 * n.succ - 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l35_3583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_unique_solution_l35_3588

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
noncomputable def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Determines if a quadratic equation has exactly one solution -/
def hasExactlyOneSolution (eq : QuadraticEquation) : Prop :=
  discriminant eq = 0

/-- Calculates the solution of a quadratic equation with exactly one solution -/
noncomputable def singleSolution (eq : QuadraticEquation) : ℝ :=
  -eq.b / (2 * eq.a)

theorem quadratic_equation_unique_solution :
  ∃ (k : ℝ), 
    let eq : QuadraticEquation := ⟨2 * k, 16, 4⟩
    hasExactlyOneSolution eq ∧ 
    k = 8 ∧ 
    singleSolution eq = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_unique_solution_l35_3588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l35_3526

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Check if a point lies on an ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- The slope of a line passing through two points -/
noncomputable def line_slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

theorem ellipse_properties (e : Ellipse) (p a b : Point) (l : Line) :
  eccentricity e = Real.sqrt 5 / 3 →
  on_ellipse e p →
  p.x = 3 ∧ p.y = 2 →
  (∃ t : ℝ, l.slope = 2/3 ∧ l.intercept = t) →
  on_ellipse e a →
  on_ellipse e b →
  (∃ t : ℝ, 2 * a.x - 3 * a.y + t = 0 ∧ 2 * b.x - 3 * b.y + t = 0) →
  (e.a^2 = 18 ∧ e.b^2 = 8) ∧
  line_slope p a = - line_slope p b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l35_3526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l35_3550

theorem cos_2alpha_value (α β : ℝ) 
  (h1 : π/2 < β) (h2 : β < α) (h3 : α < 3*π/4)
  (h4 : Real.cos (α - β) = 12/13)
  (h5 : Real.sin (α + β) = -3/5) :
  Real.cos (2*α) = -33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_value_l35_3550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_exp_2pi_3_l35_3520

-- Define the complex exponential function as noncomputable
noncomputable def complex_exp (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

-- State the theorem
theorem imaginary_part_exp_2pi_3 :
  Complex.im (complex_exp (2 * Real.pi / 3)) = Real.sqrt 3 / 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_exp_2pi_3_l35_3520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_problem_l35_3589

theorem sphere_diameter_problem (r : ℝ) (V : ℝ → ℝ) (d : ℝ → ℝ) :
  r = 5 →
  (∀ x, V x = (4 / 3) * Real.pi * x^3) →
  (∀ x, d x = 2 * x) →
  ∃ (s : ℝ), V s = 3 * V r ∧ d s = 10 * (3 : ℝ)^(1/3) ∧ 10 + 3 = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_problem_l35_3589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_triangle_area_is_32_l35_3586

/-- The area of the triangle bounded by y = 2x, y = -2x, and y = 8 is 32 square units -/
theorem triangle_area : ℝ → Prop := 
  fun area =>
    let line1 : ℝ → ℝ := fun x => 2 * x
    let line2 : ℝ → ℝ := fun x => -2 * x
    let line3 : ℝ → ℝ := fun _ => 8
    ∃ (A B C : ℝ × ℝ),
      (line1 A.1 = A.2 ∧ line3 A.1 = A.2) ∧
      (line2 B.1 = B.2 ∧ line3 B.1 = B.2) ∧
      (C = (0, 0)) ∧
      area = 32 ∧
      area = (1 / 2 : ℝ) * |A.1 - B.1| * |A.2 - C.2|

theorem triangle_area_is_32 : triangle_area 32 := by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_triangle_area_is_32_l35_3586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l35_3500

/-- Given that the solution set of x^2 - ax + b < 0 is {x | 1 < x < 2},
    prove that the eccentricity of the ellipse (x^2 / a^2) + (y^2 / b^2) = 1 is √5/3 -/
theorem ellipse_eccentricity (a b : ℝ) 
    (h_solution_set : ∀ x : ℝ, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) :
    Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l35_3500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l35_3501

theorem trigonometric_simplification (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : Real.cos x ≠ -1) :
  (Real.sin x / (1 + Real.cos x)) + ((1 + Real.cos x) / Real.sin x) = 2 * (1 / Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l35_3501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_three_digit_sums_l35_3512

/-- The maximum number in our range -/
def max_num : ℕ := 10^23 - 1

/-- Function to calculate the sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + digit_sum (n / 10)

/-- Predicate for numbers with two-digit sum of digits -/
def has_two_digit_sum (n : ℕ) : Prop :=
  10 ≤ digit_sum n ∧ digit_sum n ≤ 99

/-- Predicate for numbers with three-digit sum of digits -/
def has_three_digit_sum (n : ℕ) : Prop :=
  100 ≤ digit_sum n ∧ digit_sum n ≤ 999

/-- The set of numbers from 1 to 10^23 - 1 with two-digit sum of digits -/
def two_digit_sum_set : Set ℕ :=
  {n : ℕ | 1 ≤ n ∧ n ≤ max_num ∧ has_two_digit_sum n}

/-- The set of numbers from 1 to 10^23 - 1 with three-digit sum of digits -/
def three_digit_sum_set : Set ℕ :=
  {n : ℕ | 1 ≤ n ∧ n ≤ max_num ∧ has_three_digit_sum n}

/-- Theorem stating that there are more numbers with three-digit sum of digits -/
theorem more_three_digit_sums :
  ∃ f : two_digit_sum_set → three_digit_sum_set, Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_three_digit_sums_l35_3512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l35_3504

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(x^2 - 2*x)

theorem main_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/9) :
  a = 1/3 ∧
  (∀ b : ℝ, f a 2 ≥ f a (b^2 + 2)) ∧
  Set.range (g a) = Set.Ioo 0 3 := by
  sorry

#check main_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l35_3504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_decreasing_l35_3552

open Real Set

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

-- Theorem for increasing intervals
theorem f_increasing (n : ℤ) :
  ∀ x ∈ Ioo (2 * π * ↑n - π / 3) (2 * π / 3 + 2 * π * ↑n),
  HasDerivAt f (f' x) x ∧ f' x > 0 :=
by sorry

-- Theorem for decreasing intervals
theorem f_decreasing (n : ℤ) :
  ∀ x ∈ Ioo (2 * π / 3 + 2 * π * ↑n) (5 * π / 3 + 2 * π * ↑n),
  HasDerivAt f (f' x) x ∧ f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_decreasing_l35_3552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l35_3531

noncomputable section

-- Define the initial height of candles
def initial_height : ℝ := 1

-- Define the burning rates of candles
def burn_rate_1 : ℝ := 1 / 6
def burn_rate_2 : ℝ := 1 / 5

-- Define the height of each candle as a function of time
def height_1 (t : ℝ) : ℝ := initial_height - burn_rate_1 * t
def height_2 (t : ℝ) : ℝ := initial_height - burn_rate_2 * t

end noncomputable section

-- Theorem statement
theorem candle_height_ratio : 
  ∃ t : ℝ, t = 60 / 13 ∧ height_1 t = 3 * height_2 t := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l35_3531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l35_3597

theorem number_of_subsets_of_three_element_set {α : Type*} (M : Finset α) (h : M.card = 3) :
  (Finset.powerset M).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_three_element_set_l35_3597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_electrode_reaction_correct_negative_electrode_reaction_l35_3502

-- Define the components of the fuel cell
structure FuelCell where
  bacteria : Type
  electrolyte : Type
  negativeElectrode : Type
  positiveElectrode : Type

-- Define the reactions
inductive Reaction
  | overall : Reaction
  | negative : Reaction
  | positive : Reaction

-- Define the chemical species
inductive Species
  | H₂ : Species
  | O₂ : Species
  | H₂O : Species
  | Hplus : Species
  | eminus : Species

-- Define the properties of the fuel cell
class FuelCellProps (fc : FuelCell) where
  usesPhosphoricAcid : fc.electrolyte = PhosphoricAcid
  overallReaction : Prop
  negativeOxidation : fc.negativeElectrode = Oxidation
  positiveReduction : fc.positiveElectrode = Reduction

-- State the theorem
theorem negative_electrode_reaction
  (fc : FuelCell)
  [FuelCellProps fc] :
  Reaction.negative = Reaction.negative := by
  sorry

-- Define helper functions to represent chemical equations
def chemicalEquation (lhs rhs : List Species) : Prop :=
  lhs = rhs

-- Define the actual negative electrode reaction
def actualNegativeReaction : Prop :=
  chemicalEquation [Species.H₂] [Species.Hplus, Species.Hplus]

-- State the main theorem
theorem correct_negative_electrode_reaction
  (fc : FuelCell)
  [FuelCellProps fc] :
  actualNegativeReaction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_electrode_reaction_correct_negative_electrode_reaction_l35_3502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_approx_l35_3517

/-- Calculates the length of the second train given the speeds of two trains,
    the length of the first train, and the time they take to clear each other. -/
noncomputable def second_train_length (speed1 speed2 : ℝ) (length1 time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let total_distance := relative_speed * time
  total_distance - length1

/-- Theorem stating that given the specified conditions, 
    the length of the second train is approximately 164.978 meters. -/
theorem second_train_length_approx :
  let speed1 := (60 : ℝ)
  let speed2 := (90 : ℝ)
  let length1 := (111 : ℝ)
  let time := (6.623470122390208 : ℝ)
  let result := second_train_length speed1 speed2 length1 time
  |result - 164.978| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_approx_l35_3517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_apples_l35_3562

/-- Calculates the number of apples Joyce ends with given the initial conditions --/
theorem joyce_apples (initial larry_gives serena_takes_percent : ℝ) : 
  initial = 75.0 →
  larry_gives = 52.5 →
  serena_takes_percent = 33.75 →
  (initial + larry_gives) - (serena_takes_percent / 100) * (initial + larry_gives) = 84.46875 := by
  sorry

#eval (75.0 + 52.5) - (33.75 / 100) * (75.0 + 52.5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_apples_l35_3562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l35_3599

theorem coefficient_x_squared : 
  let p₁ : Polynomial ℝ := 2 * X^3 + 5 * X^2 - 3 * X + 1
  let p₂ : Polynomial ℝ := 3 * X^2 - 9 * X - 5
  (p₁ * p₂).coeff 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l35_3599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_primes_l35_3585

def primes : List ℕ := [2003, 2011, 2017, 2027, 2029, 2039]

theorem mean_of_remaining_primes :
  ∀ (chosen : List ℕ),
    chosen.length = 3 →
    (∀ x ∈ chosen, x ∈ primes) →
    (chosen.sum : ℚ) / 3 = 2023 →
    ((primes.filter (λ x => x ∉ chosen)).sum : ℚ) / 3 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_primes_l35_3585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_chromium_percent_l35_3534

noncomputable section

-- Define the properties of the alloys
def chromium_percent_alloy1 : ℝ := 10
def chromium_percent_alloy2 : ℝ := 6
def mass_alloy1 : ℝ := 15
def mass_alloy2 : ℝ := 35

-- Define the total mass of the new alloy
def total_mass : ℝ := mass_alloy1 + mass_alloy2

-- Define the function to calculate the chromium mass in an alloy
def chromium_mass (percent : ℝ) (mass : ℝ) : ℝ := (percent / 100) * mass

-- Define the total chromium mass in the new alloy
def total_chromium_mass : ℝ := 
  chromium_mass chromium_percent_alloy1 mass_alloy1 + 
  chromium_mass chromium_percent_alloy2 mass_alloy2

-- Define the chromium percentage in the new alloy
def new_chromium_percent : ℝ := (total_chromium_mass / total_mass) * 100

end noncomputable section

-- Theorem statement
theorem new_alloy_chromium_percent : new_chromium_percent = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_chromium_percent_l35_3534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l35_3503

def our_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n)

theorem sequence_periodic (a : ℕ → ℚ) (h : our_sequence a) (h8 : a 8 = 2) : a 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l35_3503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_speed_l35_3575

noncomputable def meters_to_km (m : ℝ) : ℝ := m / 1000

noncomputable def minutes_to_hours (min : ℝ) : ℝ := min / 60

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem person_speed (distance_m : ℝ) (time_min : ℝ) 
  (h1 : distance_m = 600) 
  (h2 : time_min = 2) : 
  speed (meters_to_km distance_m) (minutes_to_hours time_min) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_speed_l35_3575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_positive_and_not_sufficient_condition_l35_3519

theorem exponential_positive_and_not_sufficient_condition :
  (∀ x : ℝ, (2 : ℝ)^x > 0) ∧
  ¬(∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_positive_and_not_sufficient_condition_l35_3519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l35_3579

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given sequences and conditions -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a)
  (h2 : is_arithmetic_sequence (λ n ↦ (a n)^2 / n))
  (h3 : a 1 = 2) :
  a 10 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l35_3579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_f_equals_four_thirds_l35_3528

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 1 then x^2
  else if x > 1 ∧ x ≤ 2 then 1
  else 0

-- State the theorem
theorem definite_integral_f_equals_four_thirds :
  ∫ x in (0)..(2), f x = 4/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_f_equals_four_thirds_l35_3528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_point_l35_3574

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_specific_point :
  let ρ : Real := -3
  let θ : Real := (7 * Real.pi) / 4
  let φ : Real := Real.pi / 3
  spherical_to_rectangular ρ θ φ = (3 * Real.sqrt 6 / 4, 3 * Real.sqrt 6 / 4, -3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_specific_point_l35_3574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_12_between_15_and_250_l35_3564

theorem multiples_of_12_between_15_and_250 : 
  (Finset.filter (fun n => 12 ∣ n) (Finset.range 251 \ Finset.range 15)).card = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_12_between_15_and_250_l35_3564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_approx_twenty_l35_3539

/-- Represents a rectangular park -/
structure Park where
  length : ℝ
  width : ℝ

/-- Calculates the area of the region visible to a person walking around the park's boundary -/
noncomputable def visibleArea (p : Park) (visibilityRange : ℝ) : ℝ :=
  let innerLength := p.length - 2 * visibilityRange
  let innerWidth := p.width - 2 * visibilityRange
  let innerArea := innerLength * innerWidth
  let parkArea := p.length * p.width
  let visibleInsideArea := parkArea - innerArea
  let visibleOutsideLengthBands := 2 * (p.length * visibilityRange)
  let visibleOutsideWidthBands := 2 * (p.width * visibilityRange)
  let visibleCorners := Real.pi * visibilityRange^2
  visibleInsideArea + visibleOutsideLengthBands + visibleOutsideWidthBands + visibleCorners

theorem visible_area_approx_twenty (p : Park) (v : ℝ) 
    (h1 : p.length = 6) 
    (h2 : p.width = 4) 
    (h3 : v = 0.5) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |visibleArea p v - 20| < ε := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_approx_twenty_l35_3539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l35_3544

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  law_of_sines : a / Real.sin A = b / Real.sin B

-- Theorem statement
theorem angle_B_is_pi_third (t : Triangle) 
  (h : 2 * t.b * Real.cos t.B = t.a * Real.cos t.C + t.c * Real.cos t.A) : 
  t.B = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_l35_3544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_arg_diff_l35_3568

/-- Given complex numbers z and ω satisfying certain conditions, 
    the maximum value of cos(arg z - arg ω) is 1/8 -/
theorem max_cos_arg_diff (z ω : ℂ) : 
  z + ω + 3 = 0 →
  (∃ d : ℝ, Complex.abs z = 2 - d ∧ Complex.abs ω = 2 + d) →
  (∃ θ : ℝ, Real.cos θ = Real.cos (Complex.arg z - Complex.arg ω) ∧ θ ≤ Real.arccos (1/8)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_arg_diff_l35_3568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_coefficients_l35_3554

def binomial_expansion (a b : ℕ) : ℕ → ℕ → ℕ := 
  λ m n => Nat.choose a m * Nat.choose b n

def f (m n : ℕ) : ℕ := binomial_expansion 6 4 m n

theorem sum_of_specific_coefficients : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  -- Proof goes here
  sorry

#eval f 3 0 + f 2 1 + f 1 2 + f 0 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_coefficients_l35_3554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_difference_three_l35_3545

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

def valid_pairs (S : Finset Nat) : Finset (Nat × Nat) :=
  S.product S |>.filter (fun p => p.1 ≠ p.2 ∧ Int.natAbs (p.1 - p.2) = 3)

theorem probability_difference_three :
  (valid_pairs S).card / Nat.choose S.card 2 = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_difference_three_l35_3545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_cutting_theorem_l35_3515

/-- A line segment of unit length -/
structure UnitSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  is_unit : Real.sqrt ((start.1 - endpoint.1)^2 + (start.2 - endpoint.2)^2) = 1

/-- A straight line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two lines are parallel or perpendicular -/
def parallel_or_perpendicular (l1 l2 : Line) : Prop :=
  (l1.a * l2.a + l1.b * l2.b = 0) ∨ (l1.a * l2.b - l1.b * l2.a = 0)

/-- Predicate to check if a line cuts a segment -/
def cuts (l : Line) (s : UnitSegment) : Prop :=
  (l.a * s.start.1 + l.b * s.start.2 + l.c) * (l.a * s.endpoint.1 + l.b * s.endpoint.2 + l.c) ≤ 0

theorem segment_cutting_theorem (n : ℕ) (segments : Fin (4*n) → UnitSegment) (L : Line) :
  ∃ (L' : Line), parallel_or_perpendicular L L' ∧
    ∃ (i j : Fin (4*n)), i ≠ j ∧ cuts L' (segments i) ∧ cuts L' (segments j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_cutting_theorem_l35_3515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l35_3523

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-2)/3^n -/
noncomputable def infiniteSeries : ℝ := ∑' n, (5 * n - 2) / 3^n

/-- Theorem: The sum of the infinite series ∑(n=1 to ∞) (5n-2)/3^n equals 1/4 -/
theorem infiniteSeries_sum : infiniteSeries = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l35_3523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l35_3551

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}

def A : Set Int := {x : Int | x^2 - 1 ≤ 0}

def B : Set Int := {x : Int | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l35_3551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_A_probability_l35_3563

def total_athletes : ℕ := 7
def athletes_A : ℕ := 3
def athletes_B : ℕ := 4
def seeded_A : ℕ := 2
def seeded_B : ℕ := 2
def selected_players : ℕ := 4

def event_A (total : ℕ) (a : ℕ) (b : ℕ) (sa : ℕ) (sb : ℕ) (selected : ℕ) : ℚ :=
  (Nat.choose sa 2 * Nat.choose (a - sa) 2 + Nat.choose sb 2 * Nat.choose (b - sb) 2) / Nat.choose total selected

theorem event_A_probability :
  event_A total_athletes athletes_A athletes_B seeded_A seeded_B selected_players = 6 / 35 := by
  sorry

#eval event_A total_athletes athletes_A athletes_B seeded_A seeded_B selected_players

end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_A_probability_l35_3563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_l35_3580

def scores : List ℤ := [17, -3, 12, -7, -10, -4, -6, 1, 0, 16]

theorem score_analysis : 
  (∀ s ∈ scores, s ≤ 17 ∧ s ≥ -10) ∧
  (scores.filter (λ s => s > 0)).length = 4 ∧
  (scores.sum / scores.length : ℚ) = (16 : ℚ) / 10 := by
  sorry

#check score_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_l35_3580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l35_3573

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line y = x + m -/
structure Line where
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- Placeholder for the area of a quadrilateral function -/
noncomputable def area_quadrilateral (p1 p2 p3 p4 : Point) : ℝ := sorry

/-- Placeholder for the perpendicular relation between a point and a line -/
def perpendicular (p : Point) (l : Line) (q : Point) : Prop := sorry

theorem ellipse_area_theorem (e : Ellipse) (l : Line) :
  eccentricity e = 1/2 →
  (Point.mk (-1) (3/2) : Point) ∈ {p : Point | p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1} →
  (∃ (M N : Point), M ∈ {p : Point | p.y = p.x + l.m} ∧ 
                    N ∈ {p : Point | p.y = p.x + l.m} ∧
                    perpendicular (Point.mk (-Real.sqrt (e.a^2 - e.b^2)) 0) l M ∧
                    perpendicular (Point.mk (Real.sqrt (e.a^2 - e.b^2)) 0) l N) →
  (∃ (A : ℝ), A = Real.sqrt 7 ∧ 
   A = area_quadrilateral 
        (Point.mk (-Real.sqrt (e.a^2 - e.b^2)) 0) 
        M N 
        (Point.mk (Real.sqrt (e.a^2 - e.b^2)) 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_theorem_l35_3573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_16_l35_3553

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculate the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  (r.topRight.x - r.bottomLeft.x) * (r.topRight.y - r.bottomLeft.y)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let det := p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)
  (1/2) * abs det

/-- Theorem statement -/
theorem shaded_area_is_16 (pqrs : Rectangle) 
    (hpqrs : pqrs.bottomLeft = ⟨0, 0⟩ ∧ pqrs.topRight = ⟨4, 5⟩)
    (r : Point) (o : Point) (s : Point)
    (hr : r = ⟨1, 5⟩) (ho : o = ⟨4, 1⟩) (hs : s = ⟨3, 5⟩) :
    rectangleArea pqrs - triangleArea r o s = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_16_l35_3553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l35_3527

/-- The area of a quadrilateral given its diagonal and offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (diagonal * (offset1 + offset2)) / 2

/-- Theorem: The area of a quadrilateral with diagonal 15 cm and offsets 6 cm and 4 cm is 75 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 15 6 4 = 75 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the arithmetic expression
  simp [mul_add, div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l35_3527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l35_3593

-- Problem 1
theorem problem_1 : (-5 : ℤ)^(0 : ℕ) - (1/3 : ℚ)^(-2 : ℤ) + (-2 : ℤ)^(2 : ℕ) = -4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a ≠ 0) : (-3*a^3)^2 * (2*a^3) - 8*a^12 / (2*a^3) = 14*a^9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l35_3593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_salary_taxland_l35_3535

/-- The tax rate function in Taxland -/
noncomputable def taxRate (x : ℝ) : ℝ := x / 1000

/-- The effective salary function in Taxland -/
noncomputable def effectiveSalary (x : ℝ) : ℝ := x - (x^2 / 100000)

/-- The theorem stating the optimal salary in Taxland -/
theorem optimal_salary_taxland :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → effectiveSalary y ≤ effectiveSalary x) ∧
  x = 50000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_salary_taxland_l35_3535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_correct_l35_3529

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := x^2 - 1/x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := 2*x / Real.exp x
noncomputable def f4 (x : ℝ) : ℝ := (2*x + 1)^4

-- State the theorem
theorem derivatives_correct :
  (∀ x, deriv f1 x = 2*x + 1/x^2) ∧
  (∀ x, deriv f2 x = Real.sin x + x * Real.cos x) ∧
  (∀ x, deriv f3 x = (2 - 2*x) / Real.exp x) ∧
  (∀ x, deriv f4 x = 8 * (2*x + 1)^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_correct_l35_3529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_g_contains_all_integers_l35_3542

-- Define the function g as noncomputable
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.floor ((1 : ℝ) / (x + 3))
       else Int.ceil ((1 : ℝ) / (x + 3))

-- Theorem stating that the range of g contains all integers
theorem range_g_contains_all_integers :
  ∀ n : ℤ, ∃ x : ℝ, x ≠ -3 ∧ g x = n :=
by
  sorry

-- Note: The function g is not defined at x = -3, which is implicitly
-- handled by the condition x ≠ -3 in the theorem statement.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_g_contains_all_integers_l35_3542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_nearest_integer_l35_3578

/-- Given positive real numbers a and b where a > b, and the arithmetic mean of a and b 
    is triple their harmonic mean, the nearest integer to the ratio a/b is 4. -/
theorem ratio_nearest_integer (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : (a + b) / 2 = 3 * (2 * a * b) / (a + b)) : 
  ⌊(a / b + 1/2)⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_nearest_integer_l35_3578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l35_3521

theorem trig_inequality : Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l35_3521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l35_3549

noncomputable def f (A ω ϕ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + ϕ)

theorem function_properties (A ω ϕ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |ϕ| < π/2) 
  (h4 : ∀ x, f A ω ϕ (x + π/ω) = f A ω ϕ x)  -- minimum positive period is π
  (h5 : ∀ x, f A ω ϕ x ≤ 2)  -- maximum value is 2
  (h6 : f A ω ϕ 0 = 1)  -- passes through (0, 1)
  : 
  (∀ x, f A ω ϕ x = 2 * Real.sin (2*x + π/6)) ∧ 
  (∀ k : ℤ, ∀ x, x ∈ Set.Icc (k * π + π/6) (k * π + 2*π/3) → 
    ∀ y, y ∈ Set.Icc (k * π + π/6) (k * π + 2*π/3) → x < y → f A ω ϕ x > f A ω ϕ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l35_3549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_name_arrangement_exists_l35_3591

structure Person where
  first : String
  middle : String
  last : String

def shareNameComponent (p1 p2 : Person) : Prop :=
  p1.first = p2.first ∨ p1.middle = p2.middle ∨ p1.last = p2.last

theorem name_arrangement_exists : ∃ (people : Finset Person),
  Finset.card people = 4 ∧
  (∀ p1 p2 p3, p1 ∈ people → p2 ∈ people → p3 ∈ people → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬(p1.first = p2.first ∧ p2.first = p3.first) ∧
    ¬(p1.middle = p2.middle ∧ p2.middle = p3.middle) ∧
    ¬(p1.last = p2.last ∧ p2.last = p3.last)) ∧
  (∀ p1 p2, p1 ∈ people → p2 ∈ people → p1 ≠ p2 → shareNameComponent p1 p2) :=
by
  sorry

#check name_arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_name_arrangement_exists_l35_3591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_half_dollar_l35_3513

/-- Represents the special dice with its probabilities and payoffs -/
structure SpecialDice where
  prob_six : ℚ
  prob_mid : ℚ
  prob_one : ℚ
  payoff_six : ℚ
  payoff_mid : ℚ
  payoff_one : ℚ

/-- Calculate the expected winnings from rolling the special dice -/
def expectedWinnings (dice : SpecialDice) : ℚ :=
  dice.prob_six * dice.payoff_six +
  dice.prob_mid * dice.payoff_mid +
  dice.prob_one * dice.payoff_one

/-- The special dice described in the problem -/
def myDice : SpecialDice := {
  prob_six := 1/4
  prob_mid := 1/2
  prob_one := 1/4
  payoff_six := 4
  payoff_mid := 2
  payoff_one := -6
}

theorem expected_winnings_is_half_dollar :
  expectedWinnings myDice = 1/2 := by
  -- Unfold the definitions
  unfold expectedWinnings myDice
  -- Simplify the arithmetic
  simp [add_mul, mul_add, mul_comm, mul_assoc]
  -- Prove the equality
  ring

#eval expectedWinnings myDice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_half_dollar_l35_3513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_F_zeros_l35_3555

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^2/2 + x^3/3

noncomputable def g (x : ℝ) : ℝ := 1 - x + x^2/2 - x^3/3

noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_interval_for_F_zeros :
  ∃ (a b : ℤ), a < b ∧
  (∀ x : ℝ, F x = 0 → ↑a ≤ x ∧ x ≤ ↑b) ∧
  (∀ a' b' : ℤ, a' < b' →
    (∀ x : ℝ, F x = 0 → ↑a' ≤ x ∧ x ≤ ↑b') →
    b - a ≤ b' - a') ∧
  b - a = 3 := by
  sorry

#check min_interval_for_F_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_F_zeros_l35_3555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_phi_odd_condition_l35_3561

open Real

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := sin (2 * x + φ)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem sin_2x_plus_phi_odd_condition (φ : ℝ) :
  (φ = 0 → is_odd_function (f φ)) ∧
  ¬(is_odd_function (f φ) → φ = 0) := by
  sorry

#check sin_2x_plus_phi_odd_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_plus_phi_odd_condition_l35_3561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_deletion_game_winner_first_player_wins_9x10_second_player_wins_10x12_second_player_wins_9x11_l35_3570

/-- First player wins -/
inductive FirstPlayerWins : Prop

/-- Second player wins -/
inductive SecondPlayerWins : Prop

/-- The grid deletion game -/
def GridDeletionGame (m n : ℕ) : Prop :=
  let sum := m + n
  if sum % 2 = 1 then
    FirstPlayerWins
  else
    SecondPlayerWins

/-- Theorem: The winner of the grid deletion game is determined by the parity of m + n -/
theorem grid_deletion_game_winner (m n : ℕ) :
  GridDeletionGame m n ↔ 
    ((m + n) % 2 = 1 → FirstPlayerWins) ∧
    ((m + n) % 2 = 0 → SecondPlayerWins) :=
by sorry

/-- Corollary: For a 9x10 grid, the first player wins -/
theorem first_player_wins_9x10 : GridDeletionGame 9 10 → FirstPlayerWins :=
by sorry

/-- Corollary: For a 10x12 grid, the second player wins -/
theorem second_player_wins_10x12 : GridDeletionGame 10 12 → SecondPlayerWins :=
by sorry

/-- Corollary: For a 9x11 grid, the second player wins -/
theorem second_player_wins_9x11 : GridDeletionGame 9 11 → SecondPlayerWins :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_deletion_game_winner_first_player_wins_9x10_second_player_wins_10x12_second_player_wins_9x11_l35_3570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_geq_t_l35_3541

theorem s_geq_t (a b : ℝ) : a + b^2 + 1 ≥ a + 2*b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_geq_t_l35_3541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l35_3567

noncomputable section

variable (a b : ℝ × ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

theorem max_dot_product :
  vector_length (vector_sum a b) = 3 →
  ∃ (max_value : ℝ), max_value = 9/4 ∧
    ∀ (c d : ℝ × ℝ), vector_length (vector_sum c d) = 3 →
      dot_product c d ≤ max_value := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l35_3567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hours_is_1_6_l35_3596

/-- Represents the highway construction scenario -/
structure HighwayConstruction where
  initialMen : ℕ
  highwayLength : ℝ
  totalDays : ℕ
  partialDays : ℕ
  partialWork : ℝ
  additionalMen : ℕ
  newHoursPerDay : ℝ

/-- Calculates the initial work hours per day -/
noncomputable def initialHoursPerDay (hc : HighwayConstruction) : ℝ :=
  let totalMen := hc.initialMen + hc.additionalMen
  let remainingWork := 1 - hc.partialWork
  let remainingDays := hc.totalDays - hc.partialDays
  (totalMen * hc.newHoursPerDay * remainingDays * hc.partialWork) /
  (hc.initialMen * hc.totalDays * remainingWork)

/-- Theorem stating that the initial work hours per day is 1.6 -/
theorem initial_hours_is_1_6 (hc : HighwayConstruction) 
  (h1 : hc.initialMen = 100)
  (h2 : hc.highwayLength = 2)
  (h3 : hc.totalDays = 50)
  (h4 : hc.partialDays = 25)
  (h5 : hc.partialWork = 1/3)
  (h6 : hc.additionalMen = 60)
  (h7 : hc.newHoursPerDay = 10) :
  initialHoursPerDay hc = 1.6 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hours_is_1_6_l35_3596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l35_3560

/-- The number of bounces required for a ball dropped from 800 feet,
    bouncing back two-thirds of its previous height each time,
    to first reach a maximum height less than 10 feet. -/
theorem ball_bounce_count : ∃ k : ℕ+, (
  (∀ n : ℕ+, n < k → 800 * (2/3 : ℝ)^(n:ℝ) ≥ 10) ∧
  800 * (2/3 : ℝ)^(k:ℝ) < 10
) ∧ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l35_3560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_geometric_mean_l35_3590

/-- 
For a cone with radius R and slant height l, where the lateral surface area 
is the geometric mean of the base area and the total surface area, 
the angle between the generatrix and the height is arcsin((√5 - 1) / 2).
-/
theorem cone_angle_geometric_mean (R l : ℝ) (h_positive : R > 0 ∧ l > 0) :
  (π * R * l)^2 = (π * R^2) * (π * R^2 + π * R * l) →
  Real.arcsin ((Real.sqrt 5 - 1) / 2) = Real.arccos (R / l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_geometric_mean_l35_3590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_l35_3598

theorem sector_to_cone (sector_angle : Real) (sector_radius : Real) 
  (h1 : sector_angle = 270) (h2 : sector_radius = 12) :
  let arc_length := (sector_angle / 360) * (2 * Real.pi * sector_radius)
  let base_radius := arc_length / (2 * Real.pi)
  let slant_height := sector_radius
  base_radius = 9 ∧ slant_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_to_cone_l35_3598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_angled_step_l35_3582

/-- Represents the angles of a triangle at each step -/
structure TriangleAngles where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Initial triangle angles -/
def initial_angles : TriangleAngles :=
  { x := 45, y := 65, z := 70 }

/-- Computes the next set of angles based on the current set -/
def next_angles (current : TriangleAngles) : TriangleAngles :=
  { x := 90 - current.y,
    y := 90 - current.z,
    z := 90 - current.x }

/-- Checks if a triangle is right-angled -/
noncomputable def is_right_angled (angles : TriangleAngles) : Prop :=
  angles.x = 90 ∨ angles.y = 90 ∨ angles.z = 90

/-- Theorem: The smallest n for which the triangle is right-angled is 4 -/
theorem smallest_right_angled_step :
  ∃ (n : ℕ), n = 4 ∧
    (∀ k < n, ¬is_right_angled ((next_angles^[k]) initial_angles)) ∧
    is_right_angled ((next_angles^[n]) initial_angles) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_angled_step_l35_3582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_overtakes_bonnie_in_three_laps_l35_3581

/-- The number of laps Annie completes when she first overtakes Bonnie -/
noncomputable def laps_when_annie_overtakes_bonnie (track_circumference : ℝ) (annie_speed_ratio : ℝ) : ℝ :=
  annie_speed_ratio * track_circumference / (annie_speed_ratio - 1) / track_circumference

theorem annie_overtakes_bonnie_in_three_laps :
  laps_when_annie_overtakes_bonnie 300 1.5 = 3 := by
  -- Unfold the definition of laps_when_annie_overtakes_bonnie
  unfold laps_when_annie_overtakes_bonnie
  -- Simplify the expression
  simp
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_overtakes_bonnie_in_three_laps_l35_3581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_integral_exists_lebesgue_not_exists_l35_3594

open MeasureTheory

/-- A function for which the Newton integral exists but the Lebesgue integral does not -/
noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.sin (1 / x^2) - 2 / x * Real.cos (1 / x^2)

/-- The antiderivative of f -/
noncomputable def F (x : ℝ) : ℝ := x^2 * Real.sin (1 / x^2)

theorem newton_integral_exists_lebesgue_not_exists :
  ∃ (f : ℝ → ℝ), 
    (∀ x ∈ Set.Ioo 0 1, HasDerivAt F (f x) x) ∧ 
    (F 1 - F 0 = Real.sin 1) ∧
    ¬ IntegrableOn f (Set.Icc 0 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_integral_exists_lebesgue_not_exists_l35_3594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l35_3584

def z (x : ℝ) : ℂ := Complex.mk (x^2 + x - 2) (x - 1)

theorem complex_number_properties (m : ℝ) :
  (∃ x, z x = Complex.mk (z x).re 0 ↔ m = 1) ∧
  (∃ x, z x = Complex.mk 0 (z x).im ∧ (z x).im ≠ 0 ↔ m = -2) ∧
  (∃ x, (z x).re > 0 ∧ (z x).im < 0 ↔ m < -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l35_3584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l35_3518

def P : ℝ × ℝ := (3, -4)

theorem cos_alpha_plus_pi_fourth (α : ℝ) 
  (h : P.1 * Real.cos α = 3 ∧ P.2 * Real.sin α = -4) : 
  Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l35_3518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_twentyfive_thirtysix_l35_3595

theorem fraction_power_product_equals_twentyfive_thirtysix :
  (5 / 6 : ℚ) ^ 4 * (5 / 6 : ℚ) ^ (-2 : ℤ) = 25 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_product_equals_twentyfive_thirtysix_l35_3595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sale_above_mean_l35_3565

noncomputable def sales : List ℚ := [50, 50, 97, 97, 97, 120, 125, 155, 199, 199, 239]

noncomputable def mean (l : List ℚ) : ℚ := (l.sum) / l.length

theorem smallest_sale_above_mean :
  let m := mean sales
  let above_mean := sales.filter (λ x => x > m)
  above_mean.minimum? = some 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sale_above_mean_l35_3565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_find_a_value_l35_3530

-- Define the real number a as a parameter
variable (a : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := |x - a| + 3 * x

-- Assumption that a > 0
axiom a_positive : a > 0

-- Theorem 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1 := by sorry

-- Theorem 2
theorem find_a_value :
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_find_a_value_l35_3530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_theorem_l35_3540

def total_items_after_purchase (marbles : ℚ) : ℚ :=
  let frisbees := marbles / 2
  let cards := frisbees - 20
  let purchase_factor := 7 / 5
  purchase_factor * (marbles + frisbees + cards)

theorem total_items_theorem (marbles : ℚ) 
  (h1 : marbles = 60) : total_items_after_purchase marbles = 140 := by
  sorry

#eval Int.floor (total_items_after_purchase 60)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_theorem_l35_3540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_or_7_l35_3543

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d) / 2

theorem max_sum_at_6_or_7 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 > 0)
  (h2 : sum_n seq 3 = sum_n seq 10) :
  (∃ n : ℕ, ∀ k : ℕ, sum_n seq n ≥ sum_n seq k) →
  (∃ n : ℕ, (n = 6 ∨ n = 7) ∧ ∀ k : ℕ, sum_n seq n ≥ sum_n seq k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_or_7_l35_3543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l35_3514

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if a triangle is equilateral -/
noncomputable def Triangle.isEquilateral (t : Triangle) : Prop :=
  let d1 := ((t.a.x - t.b.x)^2 + (t.a.y - t.b.y)^2).sqrt
  let d2 := ((t.b.x - t.c.x)^2 + (t.b.y - t.c.y)^2).sqrt
  let d3 := ((t.c.x - t.a.x)^2 + (t.c.y - t.a.y)^2).sqrt
  d1 = d2 ∧ d2 = d3

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- Calculate the distance between two points -/
noncomputable def distance (p1 : Point) (p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- The main theorem -/
theorem perimeter_of_figure (a b c d e f g h : Point) : 
  let abc := Triangle.mk a b c
  let ade := Triangle.mk a d e
  let efg := Triangle.mk e f g
  abc.isEquilateral ∧ 
  ade.isEquilateral ∧ 
  efg.isEquilateral ∧ 
  isMidpoint d a c ∧ 
  isMidpoint h a d ∧ 
  distance a b = 6 →
  distance a b + distance b c + distance c d + 
  distance d e + distance e f + distance f g + 
  distance g h + distance h a = 22.5 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l35_3514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_htht_sequence_l35_3556

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of four coin flips -/
def FourFlips := (CoinFlip × CoinFlip × CoinFlip × CoinFlip)

/-- The probability of getting heads on a single flip of a fair coin -/
noncomputable def prob_heads : ℝ := 1 / 2

/-- The probability of getting tails on a single flip of a fair coin -/
noncomputable def prob_tails : ℝ := 1 / 2

/-- The desired sequence of flips: heads, tails, heads, tails -/
def desired_sequence : FourFlips := (CoinFlip.Heads, CoinFlip.Tails, CoinFlip.Heads, CoinFlip.Tails)

/-- 
  Theorem: The probability of getting the sequence heads, tails, heads, tails 
  when flipping a fair coin four times is equal to 1/16.
-/
theorem prob_htht_sequence : 
  prob_heads * prob_tails * prob_heads * prob_tails = 1 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_htht_sequence_l35_3556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_and_degree_l35_3525

noncomputable section

/-- The coefficient of a monomial -/
def coefficient (a : ℝ) : ℝ := a

/-- The degree of a monomial -/
def degree (n m : ℕ) : ℕ := n + m

/-- The monomial -πx³y²/5 -/
def monomial : ℝ := -Real.pi / 5

theorem monomial_coefficient_and_degree :
  coefficient (-Real.pi/5) = -Real.pi/5 ∧ degree 3 2 = 5 :=
by
  constructor
  · rfl
  · rfl

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_and_degree_l35_3525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l35_3587

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

-- State the theorem
theorem t_100_mod_7 : T 100 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l35_3587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_gt_one_iff_x_gt_e_l35_3569

-- Define the natural logarithm properties
axiom ln_monotone : ∀ x y : ℝ, x < y → Real.log x < Real.log y
axiom ln_continuous : Continuous Real.log
axiom ln_e_eq_one : Real.log (Real.exp 1) = 1

-- Statement to prove
theorem ln_gt_one_iff_x_gt_e : ∀ x : ℝ, x > 0 → (Real.log x > 1 ↔ x > Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_gt_one_iff_x_gt_e_l35_3569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gun_fire_probability_player_A_l35_3506

/-- The probability of the gun firing while player A is holding it in a two-player
    game with a six-shot revolver containing one bullet, where A starts and the
    cylinder is randomly spun before each shot. -/
theorem gun_fire_probability_player_A : ℝ := by
  let revolver_chambers : ℕ := 6
  let bullet_count : ℕ := 1
  let p_fire : ℝ := bullet_count / revolver_chambers
  let p_not_fire : ℝ := 1 - p_fire
  
  -- Define p_A as a local variable
  let p_A : ℝ := 6 / 11

  -- State the equation that p_A should satisfy
  have h : p_A = p_fire + p_not_fire * p_not_fire * p_A := by
    sorry -- Skip the proof of this equation
  
  -- Assert the final result
  exact p_A


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gun_fire_probability_player_A_l35_3506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l35_3516

-- Define the circle x^2 + y^2 = 4
def circle_four (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the circle x^2 + y^2 = 1
def circle_one (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the curve x^2/3 + y^2 = 1
def curve (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the chord length
noncomputable def chordLength (l : ℝ → ℝ → Prop) : ℝ := 2*Real.sqrt 3

-- Theorem statement
theorem line_curve_intersection 
  (l : ℝ → ℝ → Prop) 
  (h1 : ∃ a b c, ∀ x y, l x y ↔ line a b c x y) 
  (h2 : ∃ x1 y1 x2 y2, l x1 y1 ∧ l x2 y2 ∧ circle_four x1 y1 ∧ circle_four x2 y2 ∧ 
        ((x2 - x1)^2 + (y2 - y1)^2 = (chordLength l)^2)) :
  (∃ x y, l x y ∧ curve x y) ∧ 
  (∀ x1 y1 x2 y2, l x1 y1 ∧ l x2 y2 ∧ curve x1 y1 ∧ curve x2 y2 → 
    (x1 = x2 ∧ y1 = y2) ∨ (x1 ≠ x2 ∨ y1 ≠ y2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l35_3516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_distance_l35_3522

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points using the Pythagorean theorem -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Represents Biker Bob's journey -/
def bikerBobJourney : List (ℝ × ℝ) :=
  [(20, 0), (0, 6), (-10, 0), (0, 18)]

/-- Calculates the final position after following a list of movements -/
def finalPosition (movements : List (ℝ × ℝ)) : Point :=
  let finalCoords := movements.foldl (fun (acc : ℝ × ℝ) (mov : ℝ × ℝ) => 
    (acc.1 - mov.1, acc.2 + mov.2)) (0, 0)
  { x := finalCoords.1, y := finalCoords.2 }

theorem biker_bob_distance : 
  distance { x := 0, y := 0 } (finalPosition bikerBobJourney) = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_bob_distance_l35_3522
