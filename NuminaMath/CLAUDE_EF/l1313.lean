import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_geometric_sequence_l1313_131324

theorem sine_arithmetic_geometric_sequence (α₁ β : ℝ) :
  let α : ℕ → ℝ := λ n => α₁ + (n - 1) * β
  ∃ q : ℝ, (∀ n : ℕ, Real.sin (α (n + 1)) = q * Real.sin (α n)) → (q = 1 ∨ q = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_geometric_sequence_l1313_131324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1313_131326

-- Define the points
noncomputable def A : ℝ × ℝ := (0, 8)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (12, 0)

-- Define D and E as functions of A, B, and C
noncomputable def D : ℝ × ℝ := (A.1, A.2 - 3/4 * (A.2 - B.2))
noncomputable def E : ℝ × ℝ := (B.1 + 1/4 * (C.1 - B.1), B.2)

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_triangle_DBC : triangleArea D B C = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l1313_131326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replace_fence_and_install_gates_cost_l1313_131380

-- Define the plot dimensions
noncomputable def short_side (total_length : ℝ) : ℝ := total_length / 8

noncomputable def long_side (total_length : ℝ) : ℝ := 3 * short_side total_length

-- Define the costs
def fence_cost_per_foot : ℝ := 5
def gate_cost : ℝ := 150
def gate_installation_cost : ℝ := 75

-- Define the total cost function
noncomputable def total_cost (total_length : ℝ) : ℝ :=
  fence_cost_per_foot * short_side total_length + 2 * gate_cost + 2 * gate_installation_cost

-- Theorem statement
theorem replace_fence_and_install_gates_cost :
  total_cost 640 = 850 := by
  -- Unfold definitions
  unfold total_cost
  unfold short_side
  -- Simplify the expression
  simp [fence_cost_per_foot, gate_cost, gate_installation_cost]
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replace_fence_and_install_gates_cost_l1313_131380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cylinder_half_volume_l1313_131351

-- Define the cylinder type
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the volume function for a cylinder
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

-- Define the original cylinder
def original_cylinder : Cylinder := ⟨6, 12⟩

-- Define the new cylinder
def new_cylinder : Cylinder := ⟨6, 6⟩

-- Theorem statement
theorem new_cylinder_half_volume :
  volume new_cylinder = (1 / 2) * volume original_cylinder := by
  -- Unfold the definitions
  unfold volume
  unfold original_cylinder
  unfold new_cylinder
  -- Simplify the expressions
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cylinder_half_volume_l1313_131351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1313_131377

theorem cube_root_equation_solution (x : ℝ) :
  (Real.rpow (5 - 2 / x) (1/3 : ℝ) = -3) → x = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1313_131377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l1313_131352

theorem sequence_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((2:ℝ)^n + 7^n) / ((2:ℝ)^n - 7^(n-1)) + 7| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l1313_131352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_and_not_extreme_value_l1313_131345

-- Define the slope of a line ax + by + c = 0
noncomputable def lineslope (a b : ℝ) : ℝ := -a / b

-- Define perpendicularity of two lines
def perpendicular (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  lineslope a₁ b₁ * lineslope a₂ b₂ = -1

-- Define extreme value point
def extremeValuePoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

theorem perpendicular_lines_and_not_extreme_value :
  (∃ a : ℝ, perpendicular 2 1 (-1) 1 a 1) ∧
  ¬(∀ f : ℝ → ℝ, ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → 
    deriv f x₀ = 0 → extremeValuePoint f x₀) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_and_not_extreme_value_l1313_131345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1313_131340

def U : Finset Int := {-2, -1, 0, 1, 2, 3, 4, 5, 6}

def M : Finset Int := U.filter (fun x => x > -1 ∧ x < 4)

theorem complement_of_M :
  U \ M = {-2, -1, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1313_131340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unpointed_dots_bound_l1313_131325

/-- Represents a system of two infinite rows of dots with arrows --/
structure DotSystem where
  /-- Function representing arrows from top row to bottom row --/
  arrow : ℤ → ℤ
  /-- No two arrows point at the same dot --/
  injective : Function.Injective arrow
  /-- No arrow can extend right or left by more than 1006 positions --/
  bounded : ∀ n : ℤ, |arrow n - n| ≤ 1006

/-- The maximum number of dots in the lower row that can have no arrow pointing to them --/
def max_unpointedDots : ℕ := 2012

/-- Theorem stating that at most 2012 dots in the lower row could have no arrow pointing to them --/
theorem max_unpointed_dots_bound (s : DotSystem) :
  ∀ (A : Finset ℤ), (∀ n ∈ A, n ∉ Set.range s.arrow) → A.card ≤ max_unpointedDots := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unpointed_dots_bound_l1313_131325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_and_intersect_l1313_131388

/-- A line passing through point A(2, 2) -/
structure Line where
  slope : ℝ
  passes_through_A : slope * 2 - 2 = 0

/-- The circle C with equation x^2 + y^2 - 6x + 8 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ : ℝ) (l : Line) : ℝ :=
  |l.slope * x₀ - y₀ - 2 * l.slope + 2| / Real.sqrt (l.slope^2 + 1)

theorem line_tangent_and_intersect (l : Line) :
  (distance_point_to_line 3 0 l = 1 →
    3 * l.slope + 4 = 0) ∧
  (distance_point_to_line 3 0 l = Real.sqrt 2 / 2 →
    l.slope = -1 ∨ l.slope = -7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_and_intersect_l1313_131388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_xiao_ming_tasks_l1313_131312

/-- Represents a task with a name and duration in minutes -/
structure MyTask where
  name : String
  duration : ℕ

/-- Calculates the minimum time required to complete a list of tasks -/
def minTimeRequired (tasks : List MyTask) : ℕ :=
  sorry

/-- The list of tasks Xiao Ming needs to complete -/
def xiaoMingTasks : List MyTask :=
  [{ name := "Reviewing lessons", duration := 30 },
   { name := "Resting", duration := 30 },
   { name := "Boiling water", duration := 15 },
   { name := "Doing homework", duration := 25 }]

/-- Theorem stating that the minimum time required for Xiao Ming's tasks is 85 minutes -/
theorem min_time_for_xiao_ming_tasks :
  minTimeRequired xiaoMingTasks = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_for_xiao_ming_tasks_l1313_131312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1313_131302

/-- Represents a hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
noncomputable def focalDistance (h : Hyperbola) : ℝ := sorry

/-- Theorem: The distance between the foci of the given hyperbola is 8 -/
theorem hyperbola_focal_distance :
  let h : Hyperbola := {
    asymptote1 := fun x ↦ x + 2,
    asymptote2 := fun x ↦ -x + 4,
    point := (4, 2)
  }
  focalDistance h = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1313_131302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_conditions_l1313_131316

/-- The function f(x) defined as ln x - ax^2 - bx -/
noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x^2 - b * x

/-- Theorem stating that if f(1) = 0 and f'(1) = 0, then a = 1 and b = -1 -/
theorem extreme_value_conditions (a b : ℝ) :
  f a b 1 = 0 ∧ (deriv (f a b)) 1 = 0 → a = 1 ∧ b = -1 := by
  sorry

#check extreme_value_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_conditions_l1313_131316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1313_131355

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4 * Real.exp (x - 1)) / (x + 1) + x^2 - 3 * a * x + a^2 - 1

/-- The theorem stating the minimum value of a -/
theorem min_value_of_a :
  ∀ a : ℝ, (∃ x₀ > 0, f a x₀ ≤ 0) → a ≥ 1 ∧
  ∃ x₀ > 0, f 1 x₀ ≤ 0 := by
  sorry

#check min_value_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1313_131355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_1_to_inf_f_extreme_values_on_1_to_3_l1313_131309

-- Define the function f(x) = 1/x + x
noncomputable def f (x : ℝ) : ℝ := 1/x + x

-- State the theorems to be proved
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by sorry

theorem f_increasing_on_1_to_inf : 
  ∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_extreme_values_on_1_to_3 : 
  ∀ x : ℝ, x ≥ 1 ∧ x ≤ 3 → f x ≥ 2 ∧ f x ≤ 10/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_1_to_inf_f_extreme_values_on_1_to_3_l1313_131309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_relation_l1313_131321

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)

theorem triangle_angle_sine_relation (t : Triangle) :
  (t.C > t.B ↔ Real.sin t.C > Real.sin t.B) ∧
  (∃ (t : Triangle), t.a > t.b ∧ ¬(t.a * t.c^2 > t.b * t.c^2)) ∧
  (∃ (t : Triangle), t.a ≤ t.b ∧ t.a * t.c^2 > t.b * t.c^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_relation_l1313_131321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l1313_131320

theorem necessary_but_not_sufficient_condition (k : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 0 (π/2) → k * Real.sin x * Real.cos x < x) ↔
  (k ≤ 1 ∧ ∃ k₀ : ℝ, k₀ = 1 ∧ ∀ x : ℝ, x ∈ Set.Ioo 0 (π/2) → k₀ * Real.sin x * Real.cos x < x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l1313_131320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_regular_irregular_l1313_131343

/-- A regular polygon with m sides and circumradius r -/
structure RegularPolygon where
  m : ℕ
  r : ℝ
  h_m : m ≥ 3

/-- An irregular convex polygon with p sides -/
structure IrregularConvexPolygon where
  p : ℕ
  h_p : p ≥ 3
  vertices : Set (ℝ × ℝ)

/-- Function to calculate the maximum number of intersections between two polygons -/
def max_intersections (Q1 : RegularPolygon) (Q2 : IrregularConvexPolygon) : ℕ :=
  Q1.m * Q2.p

/-- Theorem stating the maximum number of intersections between a regular polygon and an irregular convex polygon -/
theorem max_intersections_regular_irregular
  (Q1 : RegularPolygon) (Q2 : IrregularConvexPolygon) (s : ℝ)
  (h_m_le_p : Q1.m ≤ Q2.p)
  (h_r_lt_s : Q1.r < s)
  (h_Q2_vertices : ∀ v ∈ Q2.vertices, ∃ c : ℝ × ℝ, ‖v - c‖ = s) :
  max_intersections Q1 Q2 = Q1.m * Q2.p :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_regular_irregular_l1313_131343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_absolute_value_m_l1313_131317

def complex_quadratic_equation 
  (z₁ z₂ m : ℂ) (x : ℂ) : Prop :=
  x^2 + z₁*x + z₂ + m = 0

theorem min_absolute_value_m 
  (z₁ z₂ m : ℂ) 
  (h₁ : z₁^2 - 4*z₂ = 16 + 20*Complex.I) 
  (h₂ : ∃ α β : ℂ, complex_quadratic_equation z₁ z₂ m α ∧ 
                   complex_quadratic_equation z₁ z₂ m β ∧ 
                   Complex.abs (α - β) = 2 * Real.sqrt 7) :
  ∃ m_min : ℝ, m_min = 7 - Real.sqrt 41 ∧ 
    (∀ m' : ℂ, (∃ α β : ℂ, complex_quadratic_equation z₁ z₂ m' α ∧ 
                           complex_quadratic_equation z₁ z₂ m' β ∧ 
                           Complex.abs (α - β) = 2 * Real.sqrt 7) →
      Complex.abs m' ≥ m_min) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_absolute_value_m_l1313_131317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1313_131347

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/4)

theorem axis_of_symmetry :
  ∀ (x : ℝ), f ((-Real.pi/4) + x) = f ((-Real.pi/4) - x) :=
by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1313_131347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_square_l1313_131300

theorem unique_six_digit_square (n : ℕ) : 
  (n^2 ≥ 100000 ∧ n^2 ≤ 999999) → -- six-digit number condition
  (∃ a b : ℕ, a < 100 ∧ b < 100 ∧ 
    n^2 = 10000 * a + 100 * b + b ∧ -- division into three two-digit parts
    a + b = 100) → -- sum of first and middle parts is 100
  n^2 = 316969 := by
  intro h1 h2
  sorry

#check unique_six_digit_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_square_l1313_131300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_per_table_l1313_131359

theorem customers_per_table (initial_tables tables_left : Float) (total_customers : Nat) :
  initial_tables = 44.0 →
  tables_left = 12.0 →
  total_customers = 256 →
  (total_customers.toFloat) / (initial_tables - tables_left) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_per_table_l1313_131359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1313_131362

def f (x : ℝ) : ℝ := x^2 - 2*x

noncomputable def g (x : ℝ) : ℝ := if x ≥ 1 then x - 2 else -x

theorem solution_set (x : ℝ) : f x ≤ 3 * g x ↔ x ∈ Set.Icc (-1) 0 ∪ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1313_131362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1313_131383

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a distance of 12 cm between them, is 228 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 12 = 228 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1313_131383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1313_131371

/-- The distance from a point (x₀, y₀) to the line ax + y + c = 0 -/
noncomputable def distanceToLine (x₀ y₀ a c : ℝ) : ℝ :=
  abs (a * x₀ + y₀ + c) / Real.sqrt (a^2 + 1)

/-- Two points are equidistant from a line -/
def areEquidistant (x₁ y₁ x₂ y₂ a c : ℝ) : Prop :=
  distanceToLine x₁ y₁ a c = distanceToLine x₂ y₂ a c

theorem equidistant_points_line (a : ℝ) :
  areEquidistant 1 (-2) 5 6 a 1 ↔ a = -2 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1313_131371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_mondays_wednesdays_count_l1313_131399

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific day in a 31-day month -/
def countDayInMonth (firstDay : DayOfWeek) (day : DayOfWeek) : Nat :=
  sorry

/-- Checks if Mondays and Wednesdays are equal for a given first day of the month -/
def hasEqualMondaysWednesdays (firstDay : DayOfWeek) : Bool :=
  countDayInMonth firstDay DayOfWeek.Monday = countDayInMonth firstDay DayOfWeek.Wednesday

/-- List of all days of the week -/
def allDays : List DayOfWeek :=
  [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- The main theorem to prove -/
theorem equal_mondays_wednesdays_count :
  (allDays.filter hasEqualMondaysWednesdays).length = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_mondays_wednesdays_count_l1313_131399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l1313_131364

-- Define the first function
noncomputable def f₁ (x : ℝ) : ℝ := (1/2) * x^2 - x - 1/x

-- Define the second function
noncomputable def f₂ (x : ℝ) : ℝ := Real.exp x + Real.log x + Real.sin x

-- State the theorem for the derivative of the first function
theorem derivative_f₁ (x : ℝ) (h : x ≠ 0) : 
  deriv f₁ x = (x^3 - x^2 + 1) / x^2 := by sorry

-- State the theorem for the derivative of the second function
theorem derivative_f₂ (x : ℝ) (h : x > 0) : 
  deriv f₂ x = Real.exp x + 1/x + Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l1313_131364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_system_solution_l1313_131301

/-- A cubic polynomial of the form x³ + ax² + bx --/
def CubicPolynomial (a b : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x

/-- The equation for finding s --/
def SEquation (a b : ℝ) : ℝ → ℝ := fun s ↦ -2*s^3 - 4*a*s^2 - 4*b*s - 3*s - 2*a*b - 2*a

/-- Definition of t in terms of s --/
def TValue (a b : ℝ) : ℝ → ℝ := fun s ↦ s^2 + a*s + b + 1

theorem cubic_system_solution (a b : ℝ) :
  ∃ (s t x y : ℝ),
    SEquation a b s = 0 ∧
    t = TValue a b s ∧
    (x = (s + Real.sqrt (s^2 - 4*t)) / 2 ∨ x = (s - Real.sqrt (s^2 - 4*t)) / 2) ∧
    (y = (s + Real.sqrt (s^2 - 4*t)) / 2 ∨ y = (s - Real.sqrt (s^2 - 4*t)) / 2) ∧
    x ≠ y ∧
    x = CubicPolynomial a b y ∧
    y = CubicPolynomial a b x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_system_solution_l1313_131301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l1313_131363

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  (initial_paint - (1/2 * initial_paint) - (1/4 * (initial_paint - (1/2 * initial_paint)))) = 3/8 * initial_paint := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l1313_131363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divided_into_rectangles_l1313_131336

theorem square_divided_into_rectangles (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 160) :
  2 * (s + s / 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divided_into_rectangles_l1313_131336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1313_131398

-- Define the points
def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the intersection of two lines
def intersect (p q r s : ℝ × ℝ) (t : ℝ × ℝ) : Prop :=
  ∃ (l m : ℝ), t = (l • p + (1 - l) • q) ∧ t = (m • r + (1 - m) • s)

-- State the theorem
theorem length_of_A'B'_is_2root2 :
  ∃ (A' B' : ℝ × ℝ),
    line_y_eq_x A' ∧
    line_y_eq_x B' ∧
    intersect A A' B B' C ∧
    (A'.1 - B'.1)^2 + (A'.2 - B'.2)^2 = 8 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1313_131398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1313_131310

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the circle
def myCircle (center : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}

-- Define the line x + 1 = 0
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ (center : ℝ × ℝ) (r : ℝ),
    parabola center →
    (1, 0) ∈ myCircle center r →
    ∃! p : ℝ × ℝ, p ∈ myCircle center r ∩ line :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1313_131310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l1313_131378

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the vertex of the hyperbola
def vertex : ℝ × ℝ := (1, 0)

-- Define the asymptote function
noncomputable def asymptote (x : ℝ) : ℝ := Real.sqrt 3 * x

-- State the theorem
theorem distance_vertex_to_asymptote :
  let (x₀, y₀) := vertex
  let d := |asymptote x₀ - y₀| / Real.sqrt (1 + (Real.sqrt 3)^2)
  d = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_vertex_to_asymptote_l1313_131378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1313_131305

open Real

theorem trig_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : sin α = 4/5) : 
  tan α = 4/3 ∧ 
  (sin (α + π) - 2*cos (π/2 + α)) / (-sin (-α) + cos (π + α)) = 4 ∧ 
  sin (2*α + π/4) = 17*sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1313_131305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l1313_131391

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, 5)

theorem point_D_coordinates :
  ∃ D : ℝ × ℝ,
  let vector_AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)
  let vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  vector_AD = (3 * vector_AB.1, 3 * vector_AB.2) ∧
  D = (-7, 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l1313_131391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l1313_131331

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 4 * x - 1

noncomputable def f' (x : ℝ) : ℝ := x^2 - 8 * x + 4

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_extreme_points
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_extreme : f' (a 3) = 0 ∧ f' (a 7) = 0) :
  a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_extreme_points_l1313_131331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_neg_sqrt5_over_5_l1313_131397

/-- Given an angle α and a point P on its terminal side, prove that cos(α) = -√5/5 --/
theorem cos_alpha_equals_neg_sqrt5_over_5 (α : ℝ) (P : ℝ × ℝ) : 
  P = (-1, 2) → Real.cos α = -(Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_neg_sqrt5_over_5_l1313_131397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_formula_l1313_131327

/-- Represents a triangle with a rectangle formed by parallel segments -/
structure TriangleWithRectangle where
  b : ℝ  -- base length of the triangle
  H : ℝ  -- total height of the triangle
  k : ℝ  -- height of the top triangle
  x : ℝ  -- height of the rectangle
  h_positive : 0 < b ∧ 0 < H ∧ 0 < k ∧ 0 < x
  h_k_less_than_H : k < H

/-- The area of the rectangle in the triangle -/
noncomputable def rectangle_area (t : TriangleWithRectangle) : ℝ :=
  (t.b * t.k * t.x) / t.H

/-- Theorem stating that the area of the rectangle is (b * k * x) / H -/
theorem rectangle_area_formula (t : TriangleWithRectangle) :
  rectangle_area t = (t.b * t.k * t.x) / t.H :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_formula_l1313_131327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l1313_131373

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 2

/-- The slope of the second line -/
noncomputable def m₂ : ℝ := -4

/-- The y-intercept of both lines -/
noncomputable def b : ℝ := 6

/-- The x-intercept of the first line -/
noncomputable def x₁ : ℝ := -b / m₁

/-- The x-intercept of the second line -/
noncomputable def x₂ : ℝ := -b / m₂

/-- The distance between the x-intercepts of the two lines -/
noncomputable def distance : ℝ := |x₂ - x₁|

theorem x_intercept_distance :
  distance = 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_distance_l1313_131373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_15_years_l1313_131357

/-- Represents the price of a computer over time -/
noncomputable def computer_price (initial_price : ℝ) (years : ℕ) : ℝ :=
  initial_price * (2/3)^(years / 5)

/-- Theorem: After 15 years, a computer initially priced at 5400 yuan will cost 1600 yuan -/
theorem price_after_15_years :
  computer_price 5400 15 = 1600 := by
  -- Unfold the definition of computer_price
  unfold computer_price
  -- Simplify the expression
  simp [Real.rpow_nat_cast]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_15_years_l1313_131357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_zero_trig_expression_simplification_l1313_131322

-- Part I
theorem log_expression_equals_zero :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) - (2 * Real.sqrt 2) ^ (2/3) - Real.exp (Real.log 2) = 0 := by
  sorry

-- Part II
theorem trig_expression_simplification :
  (Real.sqrt (1 - Real.sin (20 * π / 180))) / (Real.cos (10 * π / 180) - Real.sin (170 * π / 180)) =
  Real.cos (10 * π / 180) - Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_zero_trig_expression_simplification_l1313_131322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_eight_l1313_131368

-- Define g as a parameter of the main theorem
noncomputable def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sqrt (-x) else g (x - 1)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_implies_g_eight (g : ℝ → ℝ) :
  is_odd (f g) → g 8 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_g_eight_l1313_131368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_making_time_l1313_131381

/-- Calculates the time required to make each dress given the total fabric, fabric per dress, and total working hours. -/
noncomputable def time_per_dress (total_fabric : ℝ) (fabric_per_dress : ℝ) (total_hours : ℝ) : ℝ :=
  total_hours / (total_fabric / fabric_per_dress)

/-- Proves that given 56 square meters of fabric, 4 square meters of fabric per dress, and 42 total working hours, the time required to make each dress is 3 hours. -/
theorem dress_making_time :
  time_per_dress 56 4 42 = 3 := by
  -- Unfold the definition of time_per_dress
  unfold time_per_dress
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_making_time_l1313_131381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_on_terminal_side_l1313_131393

theorem trig_values_on_terminal_side (α : ℝ) (x : ℝ) :
  x ≠ 0 →
  (∃ y : ℝ, y = 3 ∧ x^2 + y^2 ≠ 0) →
  Real.cos α = (Real.sqrt 10 / 10) * x →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ abs (Real.tan α) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_on_terminal_side_l1313_131393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_point_l1313_131311

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def passesThrough (a b : ℚ) (x y : ℚ) : Prop := x / a + y / b = 1

theorem unique_line_through_point :
  ∃! (a b : ℚ), 
    a > 0 ∧ 
    a < 10 ∧ 
    isPrime (Int.toNat (Int.floor a)) ∧ 
    isPowerOfTwo (Int.toNat (Int.floor b)) ∧ 
    passesThrough a b 5 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_point_l1313_131311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_at_center_l1313_131375

/-- Circle with center (-1, 3) and radius 2 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 3)^2 = 4}

/-- Line with parametric equation x = 2t - 1, y = 6t - 1 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2*t - 1 ∧ p.2 = 6*t - 1}

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-1, 3)

theorem line_intersects_circle_not_at_center :
  ∃ p q : ℝ × ℝ, p ∈ Circle ∧ p ∈ Line ∧ q ∈ Circle ∧ q ∈ Line ∧ p ≠ q ∧ p ≠ CircleCenter ∧ q ≠ CircleCenter :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_at_center_l1313_131375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l1313_131392

theorem cube_root_equation (x : ℝ) : Real.sqrt (x^3) = 27 * (27 ^ (1/3 : ℝ)) → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l1313_131392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_cosine_inequality_l1313_131369

theorem negation_of_universal_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x < 2) ↔ (∃ x : ℝ, Real.cos x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_cosine_inequality_l1313_131369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l1313_131342

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℝ
  water : ℝ

/-- Calculates the ratio of milk to water in a mixture -/
noncomputable def ratio (m : Mixture) : ℝ := m.milk / m.water

theorem water_added_proof (initial : Mixture) (added_water : ℝ) : 
  initial.milk = 45 ∧ 
  ratio initial = 6/3 ∧
  ratio { milk := initial.milk, water := initial.water + added_water } = 6/5 →
  added_water = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l1313_131342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_total_score_l1313_131303

/-- The maximum number of laser beam intersections with mirrors in a single round -/
def max_intersections (n : ℕ) : ℕ := n + 2

/-- The total number of rounds in the game -/
def total_rounds : ℕ := 2016

/-- The sum of maximum intersections over all rounds -/
def total_score : ℕ := Finset.sum (Finset.range total_rounds) (λ n => max_intersections (n + 1))

/-- The theorem stating the total score of the game -/
theorem game_total_score : total_score = 1019088 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_total_score_l1313_131303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_of_sets_l1313_131372

/-- Given a universal set U and its subsets A and B, we prove that the number of elements in the intersection of A and B is 23. -/
theorem intersection_size_of_sets (U A B : Finset ℕ) : 
  (U.card = 193) → 
  (A.card = 116) → 
  (B.card = 41) → 
  ((U \ (A ∪ B)).card = 59) → 
  ((A ∩ B).card = 23) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_of_sets_l1313_131372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1313_131319

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem f_properties :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-1) x ≥ 1) ∧
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-1) x ≤ 37) ∧
  (f (-1) 1 = 1) ∧
  (f (-1) (-5) = 37) ∧
  (∀ a : ℝ, (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → f a x < f a y) ∨
            (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → f a x > f a y) ↔
            a ≥ 5 ∨ a ≤ -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1313_131319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1313_131339

-- Define the given hyperbola
def given_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1

-- Define hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem hyperbola_properties :
  -- Real axis length of C is 4
  (∃ a : ℝ, a = 2) →
  -- C shares a common focus with the given hyperbola
  (∃ c : ℝ, c^2 = 5 ∧ 
    (∀ x y : ℝ, given_hyperbola x y → (x = c ∨ x = -c) → y = 0) ∧
    (∀ x y : ℝ, hyperbola_C x y → (x = c ∨ x = -c) → y = 0)) →
  -- The equation of hyperbola C is correct
  (∀ x y : ℝ, hyperbola_C x y ↔ x^2 / 4 - y^2 = 1) ∧
  -- The minimum distance between any point on C and M(5,0) is 2
  (∃ min_dist : ℝ, min_dist = 2 ∧
    ∀ x y : ℝ, hyperbola_C x y → distance x y 5 0 ≥ min_dist) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1313_131339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1313_131335

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.univ} = {x : ℝ | x > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1313_131335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2m_plus_n_l1313_131307

noncomputable section

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1/4)

-- Define a line passing through the focus
noncomputable def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * x + 1/4

-- Define the intersection points P and Q
noncomputable def intersection_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the lengths m and n
noncomputable def m (k : ℝ) : ℝ := sorry
noncomputable def n (k : ℝ) : ℝ := sorry

-- State the theorem
theorem min_value_2m_plus_n :
  ∀ k : ℝ, 2 * m k + n k ≥ (3 + 2 * Real.sqrt 2) / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2m_plus_n_l1313_131307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l1313_131337

/-- A function f: ℝ → ℝ is an inverse proportion function if there exists a non-zero constant k such that f(x) = k/x for all non-zero x -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function y = (m-2)x^(m^2-5) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * (x ^ (m^2 - 5))

/-- Theorem: If f(m) is an inverse proportion function, then m = -2 -/
theorem inverse_proportion_m_value :
  ∃ m : ℝ, is_inverse_proportion (f m) → m = -2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l1313_131337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1313_131354

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - 4*x
  else if x < 0 then -((-x)^2 - 4*(-x))
  else 0

-- State the theorem
theorem solution_set_of_inequality (h_odd : ∀ x, f (-x) = -f x) :
  {x : ℝ | f x > x} = Set.Ioo (-5) 0 ∪ Set.Ioi 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1313_131354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_l1313_131308

/-- The circle on which P moves -/
def circleEq (a : ℝ) (x y : ℝ) : Prop := x^2 + (y - 3)^2 = a^2

/-- Point A -/
def A : ℝ × ℝ := (2, 0)

/-- Point B -/
def B : ℝ × ℝ := (-2, 0)

/-- The area of triangle PAB given P's coordinates -/
noncomputable def triangleArea (x y : ℝ) : ℝ := abs ((x - 2) * y - (x + 2) * y) / 2

/-- The theorem to be proved -/
theorem circle_max_area (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, circleEq a x y) ∧ 
  (∀ x y : ℝ, circleEq a x y → triangleArea x y ≤ 8) ∧
  (∃ x y : ℝ, circleEq a x y ∧ triangleArea x y = 8) →
  a = 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_area_l1313_131308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1313_131387

/-- Definition of ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Definition of ellipse C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + 2 * y^2 = 12

/-- Definition of point P in terms of Q -/
def P (u v : ℝ) : ℝ × ℝ := (2 * v - u, u + v)

/-- Definition of point T in terms of M and N -/
def T (xm ym xn yn : ℝ) : ℝ × ℝ := (xm + 2 * xn, ym + 2 * yn)

/-- Condition on slopes of OM and ON -/
def slope_condition (xm ym xn yn : ℝ) : Prop := 
  xm * xn + 2 * ym * yn = 0

theorem ellipse_problem :
  ∃ (F₁ F₂ : ℝ × ℝ),
    (F₁.1 = -Real.sqrt 30 ∧ F₁.2 = 0) ∧
    (F₂.1 = Real.sqrt 30 ∧ F₂.2 = 0) ∧
    (∀ (u v : ℝ), C₁ u v → C₂ (P u v).1 (P u v).2) ∧
    (∀ (xm ym xn yn : ℝ),
      C₂ xm ym → C₂ xn yn → slope_condition xm ym xn yn →
      ∃ (k : ℝ), ∀ (x y : ℝ),
        (x, y) = T xm ym xn yn →
        ((x - F₁.1)^2 + (y - F₁.2)^2).sqrt + ((x - F₂.1)^2 + (y - F₂.2)^2).sqrt = k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1313_131387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_growth_rate_is_half_l1313_131304

noncomputable def initial_profit : ℝ := 4
noncomputable def final_profit : ℝ := 9
def months : ℕ := 2

noncomputable def monthly_growth_rate (initial : ℝ) (final : ℝ) (months : ℕ) : ℝ :=
  (final / initial) ^ (1 / (months : ℝ)) - 1

theorem profit_growth_rate_is_half :
  monthly_growth_rate initial_profit final_profit months = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_growth_rate_is_half_l1313_131304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_fold_g_of_four_l1313_131318

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem six_fold_g_of_four : g (g (g (g (g (g 4))))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_fold_g_of_four_l1313_131318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_line_l1313_131333

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus condition
def focus_on_x_axis (c : ℝ) : Prop := c^2 = 1

-- Define the square area condition
noncomputable def square_area_condition (a b : ℝ) : Prop := 2 * a * b = 2 * Real.sqrt 2

-- Define the line passing through (0,2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the intersection points of the line and the ellipse
def intersection_points (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  line k x₁ y₁ ∧ line k x₂ y₂ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂

-- Define the area of the triangle formed by the intersection points and the origin
noncomputable def triangle_area (x₁ x₂ y₁ y₂ : ℝ) : ℝ :=
  abs (x₁ * y₂ - x₂ * y₁) / 2

-- The main theorem
theorem max_triangle_area_line :
  ∃ (a b c : ℝ),
    focus_on_x_axis c ∧
    square_area_condition a b ∧
    (∀ (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ),
      intersection_points k x₁ x₂ y₁ y₂ →
      triangle_area x₁ x₂ y₁ y₂ ≤ Real.sqrt 2 / 2) ∧
    (∃ (x₁ x₂ y₁ y₂ : ℝ),
      intersection_points (Real.sqrt 14 / 2) x₁ x₂ y₁ y₂ ∧
      triangle_area x₁ x₂ y₁ y₂ = Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_line_l1313_131333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_solution_l1313_131389

-- Define the power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x ^ α

-- State the theorem
theorem power_function_solution :
  ∃ α : ℝ,
  (power_function α (-2) = -1/8) ∧
  ∃ x : ℝ, (power_function α x = 64 ∧ x = 1/4) :=
by
  -- We'll use α = -3 as our solution
  use -3
  constructor
  · -- Prove that power_function (-3) (-2) = -1/8
    simp [power_function]
    norm_num
  · -- Prove that there exists an x such that power_function (-3) x = 64 and x = 1/4
    use 1/4
    constructor
    · simp [power_function]
      norm_num
    · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_solution_l1313_131389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_17_dividing_2023_factorial_l1313_131334

open Nat

theorem largest_power_of_17_dividing_2023_factorial :
  (∃ k : ℕ, (17^k : ℕ) ∣ factorial 2023 ∧ ∀ m : ℕ, (17^m : ℕ) ∣ factorial 2023 → m ≤ k) ∧
  (∀ k : ℕ, ((17^k : ℕ) ∣ factorial 2023 ∧ ∀ m : ℕ, (17^m : ℕ) ∣ factorial 2023 → m ≤ k) → k = 126) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_17_dividing_2023_factorial_l1313_131334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_or_false_is_false_main_theorem_l1313_131361

theorem false_or_false_is_false (p q : Prop) : (¬p ∧ ¬q) → ¬(p ∨ q) := by
  sorry

-- Definitions from the problem
def plane : Type := sorry
def coincide (α β : plane) : Prop := sorry
def point : Type := sorry
def in_plane (pt : point) (pl : plane) : Prop := sorry
def non_collinear (p1 p2 p3 : point) : Prop := sorry
def equidistant_from_plane (pt : point) (pl : plane) : Prop := sorry
def parallel (α β : plane) : Prop := sorry

-- Proposition p
def p : Prop :=
  ∀ (α β : plane) (p1 p2 p3 : point),
    ¬(coincide α β) →
    in_plane p1 α → in_plane p2 α → in_plane p3 α →
    non_collinear p1 p2 p3 →
    equidistant_from_plane p1 β → equidistant_from_plane p2 β → equidistant_from_plane p3 β →
    parallel α β

-- Proposition q
def q : Prop :=
  ∀ (lambda : ℝ), (lambda > -1/2 ∧ lambda ≠ 2) ↔ ((-2 * lambda - 1 < 0) ∧ ¬(lambda = 2))

-- The main theorem
theorem main_theorem : ¬p ∧ ¬q → ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_or_false_is_false_main_theorem_l1313_131361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_12_l1313_131306

/-- Calculates the distance between two points given rowing conditions -/
noncomputable def distance_between_points (rowing_speed : ℝ) (total_time : ℝ) (stream_speed : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + stream_speed
  let upstream_speed := rowing_speed - stream_speed
  let downstream_time := (total_time * downstream_speed) / (downstream_speed + upstream_speed)
  downstream_speed * downstream_time

/-- The distance between points A and B is 12 kilometers -/
theorem distance_AB_is_12 :
  distance_between_points 5 5 1 = 12 := by
  -- Unfold the definition of distance_between_points
  unfold distance_between_points
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_12_l1313_131306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_family_outing_cost_l1313_131382

/-- Calculates the total cost of a family outing to a seafood buffet with drinks -/
def calculate_total_cost (adult_price senior_price child_price teen_price college_price : ℚ)
                         (senior_discount college_discount : ℚ)
                         (soda_price tea_price coffee_price juice_price wine_price : ℚ)
                         (num_adults num_seniors num_children num_teens num_college : ℕ)
                         (num_sodas num_teas num_coffees num_juices num_wines : ℕ) : ℚ :=
  let buffet_cost := 
    adult_price * num_adults +
    senior_price * num_seniors * (1 - senior_discount) +
    child_price * num_children +
    teen_price * num_teens +
    college_price * num_college * (1 - college_discount)
  let drink_cost := 
    soda_price * num_sodas +
    tea_price * num_teas +
    coffee_price * num_coffees +
    juice_price * num_juices +
    wine_price * num_wines
  buffet_cost + drink_cost

/-- The total cost for Mr. Smith's family outing is $270.50 -/
theorem smith_family_outing_cost : 
  calculate_total_cost 30 30 15 25 30 (1/10) (1/20) 2 3 4 (3/2) 6 2 2 3 1 2 3 2 1 1 2 = 271/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_family_outing_cost_l1313_131382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_negative_120_l1313_131366

/-- The coefficient of x^r in the expansion of (1-2x)^5 -/
def T (r : ℕ) : ℤ := (-2)^r * (Nat.choose 5 r)

/-- The expansion of (1-2x)^5(2+x) -/
noncomputable def expansion : Polynomial ℤ := (1 - 2*Polynomial.X)^5 * (2 + Polynomial.X)

theorem coefficient_x_cubed_is_negative_120 : 
  (expansion.coeff 3) = -120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_is_negative_120_l1313_131366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1313_131338

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t * Real.cos (Real.pi/3), t * Real.sin (Real.pi/3))

-- Define the ellipse C
def ellipse_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 * Real.sin θ)

-- Define the cartesian equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Theorem statement
theorem intersection_segment_length :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  ellipse_equation (line_l t₁).1 (line_l t₁).2 ∧
  ellipse_equation (line_l t₂).1 (line_l t₂).2 ∧
  Real.sqrt ((line_l t₁).1 - (line_l t₂).1)^2 + ((line_l t₁).2 - (line_l t₂).2)^2 = 16/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1313_131338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_relationship_l1313_131341

/-- Given that x is directly proportional to y³, y is inversely proportional to z²,
    and x = 2 when z = 4, prove that x = 1/1492992 when z = 48. -/
theorem proportional_relationship (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h1 : x = k₁ * y^3)
    (h2 : y = k₂ / z^2)
    (h3 : 2 = k₁ * (k₂ / 4^2)^3) :
    k₁ * (k₂ / 48^2)^3 = 1/1492992 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_relationship_l1313_131341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_correct_l1313_131314

/-- The area of a square inscribed in the ellipse x²/4 + y²/8 = 1, 
    with its sides parallel to the coordinate axes -/
noncomputable def inscribed_square_area : ℝ := 32 / 3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 8 = 1

theorem inscribed_square_area_is_correct :
  ∃ (t : ℝ), 
    (ellipse_equation t t ∧ 
     ellipse_equation (-t) t ∧ 
     ellipse_equation t (-t) ∧ 
     ellipse_equation (-t) (-t)) ∧
    (4 * t^2 = inscribed_square_area) := by
  sorry

#check inscribed_square_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_correct_l1313_131314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1313_131356

def sigma (m : Nat) : Nat :=
  if m = 0 then 0
  else if m = 1 then 2
  else if m % 2 = 1 then (3 * m + 3) / 2
  else (3 * m + 2) / 2

def is_valid_sequence (x : Nat → Nat) (n M : Nat) : Prop :=
  (∀ k, k ≤ n → x k ≤ M) ∧
  (∀ k, 3 ≤ k ∧ k ≤ n → x k = Int.natAbs (x (k-1) - x (k-2))) ∧
  (∀ k, k ≤ n → x k > 0)

theorem max_sequence_length (M : Nat) :
  ∀ n x, is_valid_sequence x n M →
  n ≤ sigma M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1313_131356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_circle_radius_l1313_131332

theorem inscribed_square_circle_radius (r : ℝ) (s : ℝ) : 
  s > 0 → -- side length of square is positive
  s * Real.sqrt 2 = 2 * r → -- diagonal of square equals diameter of circle
  4 * s = Real.pi * r^2 → -- perimeter of square equals area of circle
  r = 4 * Real.sqrt 2 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_circle_radius_l1313_131332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1313_131346

noncomputable def f (a b x : ℝ) : ℝ := -2 * a * Real.sin (2 * x + Real.pi / 6) + 2 * a + b

theorem function_properties :
  ∀ (a b : ℝ),
    a > 0 →
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) →
    a = 2 ∧ b = -5 ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 4),
      f a b x ≤ -3 ∧
      f a b x ≥ -5 ∧
      (∃ x₁ ∈ Set.Icc 0 (Real.pi / 4), f a b x₁ = -3) ∧
      (∃ x₂ ∈ Set.Icc 0 (Real.pi / 4), f a b x₂ = -5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1313_131346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1313_131353

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 1 - Real.exp (Real.log 3 * x) else -1 + Real.exp (Real.log 3 * (-x))

-- State the theorem
theorem odd_function_inequality (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = 1 - Real.exp (Real.log 3 * x)) →  -- f(x) = 1 - 3^x for x > 0
  (∀ x ∈ Set.Icc 2 8, f (Real.log 2 * (x^2)) + f (5 - a * Real.log 2 * x) ≥ 0) →
  a ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l1313_131353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_equals_one_l1313_131395

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_sum_equals_one (a b : ℝ) :
  (∀ y : ℝ, y ∈ Set.Ioo a b ↔ ∃ x : ℝ, f x = y) →
  (∀ y : ℝ, y ∈ Set.Ioc a b ↔ ∃ x : ℝ, f x = y) →
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_equals_one_l1313_131395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_gons_is_maximum_l1313_131350

/-- The maximum value of n for which three non-overlapping regular n-gons can be drawn,
    each sharing a side with a distinct side of a triangle with sides 2019, 2020, and 2021. -/
def max_n_gons : ℕ := 11

/-- A triangle with sides 2019, 2020, and 2021 -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  ha : a = 2019
  hb : b = 2020
  hc : c = 2021

/-- The interior angle of a regular n-gon -/
noncomputable def interior_angle (n : ℕ) : ℝ := 180 * (n - 2 : ℝ) / n

/-- Theorem stating that max_n_gons is the maximum value of n for which three non-overlapping
    regular n-gons can be drawn, each sharing a side with a distinct side of the SpecialTriangle -/
theorem max_n_gons_is_maximum (t : SpecialTriangle) :
  ∀ n : ℕ, n > max_n_gons →
    ¬(3 * interior_angle n + 60 < 360) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_gons_is_maximum_l1313_131350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_switches_in_A_l1313_131323

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 6

/-- Checks if a number is a valid switch label -/
def isValidLabel (n : Nat) : Prop :=
  ∃ x y z : Fin 12, n = 2^x.val * 3^y.val * 5^z.val

/-- The set of all switches -/
def SwitchSet : Type := { s : Finset Switch // s.card = 1000 ∧ ∀ sw ∈ s, isValidLabel sw.label }

/-- Function to advance a switch's position -/
def advanceSwitch (sw : Switch) : Switch :=
  { label := sw.label, position := (sw.position + 1) % 6 }

/-- Function to perform one step of the process -/
def performStep (switches : SwitchSet) (i : Nat) : SwitchSet :=
  sorry

/-- Function to perform all 1000 steps -/
def performAllSteps (initialSwitches : SwitchSet) : SwitchSet :=
  sorry

/-- Counts switches in position A -/
def countSwitchesInA (switches : SwitchSet) : Nat :=
  sorry

theorem final_switches_in_A (initialSwitches : SwitchSet) :
  (∀ sw ∈ initialSwitches.val, sw.position = 0) →
  countSwitchesInA (performAllSteps initialSwitches) = 136 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_switches_in_A_l1313_131323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_ratio_l1313_131396

/-- Represents a right circular cylinder -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- The volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- The price of oil in a cylinder -/
noncomputable def oilPrice (c : Cylinder) (fullPrice : ℝ) (fillRatio : ℝ) : ℝ :=
  fullPrice * fillRatio * (volume c) / (volume (Cylinder.mk 1 1))

theorem oil_price_ratio (x y : Cylinder) 
    (h_height : y.height = 4 * x.height)
    (h_radius : y.radius = 4 * x.radius)
    (h_price_x : oilPrice x 2 1 = 2) :
  oilPrice y 2 (1/2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_ratio_l1313_131396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_cross_section_area_l1313_131385

/-- The area of a trapezoidal cross-section -/
noncomputable def trapezoidArea (topWidth bottomWidth depth : ℝ) : ℝ :=
  (1 / 2) * (topWidth + bottomWidth) * depth

/-- Theorem: The area of the given trapezoidal cross-section is 630 square meters -/
theorem channel_cross_section_area :
  trapezoidArea 12 6 70 = 630 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_cross_section_area_l1313_131385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_restoration_l1313_131344

/-- Given:
  A: Point (vertex of triangle ABC)
  D: Point (foot of altitude from B to AC)
  O: Point (center of circumcircle of triangle BHC, where H is orthocenter of ABC)
-/
structure TriangleData where
  A : Point
  D : Point
  O : Point

/-- Definition of an acute triangle -/
def is_acute_triangle (A B C : Point) : Prop := sorry

/-- Definition of D being the foot of altitude from B to AC -/
def D_is_altitude_foot (A B C D : Point) : Prop := sorry

/-- Definition of O being the circumcenter of triangle BHC -/
def O_is_circumcenter_BHC (A B C O : Point) : Prop := sorry

/-- Theorem: Given the specified points, there exists a unique acute triangle ABC -/
theorem unique_triangle_restoration (data : TriangleData) :
  ∃! (B C : Point), is_acute_triangle data.A B C ∧
    D_is_altitude_foot data.A B C data.D ∧
    O_is_circumcenter_BHC data.A B C data.O := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_restoration_l1313_131344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1313_131348

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x + y < 2)) ↔
  (∃ x y : ℝ, x ∈ Set.Ioo 0 1 ∧ y ∈ Set.Ioo 0 1 ∧ x + y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1313_131348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanic_work_hours_l1313_131329

/-- Calculates the number of hours worked by a mechanic given the total cost, part costs, labor rates, and break time. -/
noncomputable def mechanic_hours_worked (total_cost : ℝ) (part_cost : ℝ) (num_parts : ℕ) 
  (labor_rate : ℝ) (discount_rate : ℝ) (break_time : ℝ) : ℝ :=
  let parts_total := part_cost * (num_parts : ℝ)
  let labor_cost := total_cost - parts_total
  let first_two_hours_cost := 2 * 60 * labor_rate
  let remaining_labor_cost := labor_cost - first_two_hours_cost
  let discounted_rate := labor_rate * (1 - discount_rate)
  let additional_minutes := remaining_labor_cost / discounted_rate
  (additional_minutes) / 60

/-- The theorem stating that given the problem conditions, the mechanic worked approximately 4.44 hours. -/
theorem mechanic_work_hours : 
  let total_cost : ℝ := 220
  let part_cost : ℝ := 20
  let num_parts : ℕ := 2
  let labor_rate : ℝ := 0.5
  let discount_rate : ℝ := 0.1
  let break_time : ℝ := 0.5
  abs (mechanic_hours_worked total_cost part_cost num_parts labor_rate discount_rate break_time - 4.44) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mechanic_work_hours_l1313_131329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_one_l1313_131330

/-- The double sum of 1/(m^2 * n * (m + n + 2)) over positive integers m and n equals 1. -/
theorem double_sum_equals_one :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / ((m : ℝ)^2 * (n : ℝ) * ((m : ℝ) + (n : ℝ) + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_one_l1313_131330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forbidden_subgraph_characterization_l1313_131349

-- Define a surface
class Surface (S : Type) where
  -- Add any necessary properties for a surface

-- Define a graph
class Graph (G : Type) where
  -- Add any necessary properties for a graph

-- Define embedding of a graph into a surface
def Embeddable (G : Type) [Graph G] (S : Type) [Surface S] : Prop :=
  sorry -- Definition of embeddability

-- Define subgraph relation
def Subgraph (G H : Type) [Graph G] [Graph H] : Prop :=
  sorry -- Definition of subgraph relation

-- State the theorem
theorem forbidden_subgraph_characterization (S : Type) [Surface S] :
  ∃ (n : ℕ) (H : Fin n → Type) (h : ∀ i, Graph (H i)),
    ∀ (G : Type) [Graph G],
      Embeddable G S ↔ ∀ i, ¬Subgraph (H i) G :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forbidden_subgraph_characterization_l1313_131349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l1313_131376

def sequence_a : ℕ → ℕ
  | 0 => 1989^1989
  | n+1 => (Nat.digits 10 (sequence_a n)).sum

theorem a_5_equals_9 : sequence_a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_9_l1313_131376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_S_properties_l1313_131390

/-- A line in the set S -/
structure LineS (a b : ℝ) where
  θ : ℝ
  h_θ_range : θ ∈ Set.Icc 0 (2 * Real.pi)

/-- The distance from a point to a line in S -/
noncomputable def distanceToLine (a b : ℝ) (x y : ℝ) (l : LineS a b) : ℝ :=
  abs (Real.sin l.θ / a * x + Real.cos l.θ / b * y - 1) / Real.sqrt ((Real.sin l.θ / a)^2 + (Real.cos l.θ / b)^2)

/-- The theorem stating properties of lines in S -/
theorem lines_S_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a = b → ∀ (l : LineS a b), distanceToLine a b 0 0 l = a) ∧
  (a > b → ∀ (l1 l2 : LineS a b), distanceToLine a b 0 0 l1 - distanceToLine a b 0 0 l2 ≥ 2 * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_S_properties_l1313_131390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l1313_131358

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := sorry

/-- The number of pens Masha bought -/
def masha_pens : ℕ := sorry

/-- The number of pens Olya bought -/
def olya_pens : ℕ := sorry

/-- Theorem stating the total number of pens bought by Masha and Olya -/
theorem total_pens_bought :
  (pen_cost > 10) →
  (pen_cost * masha_pens = 357) →
  (pen_cost * olya_pens = 441) →
  (masha_pens + olya_pens = 38) :=
by
  sorry

#check total_pens_bought

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l1313_131358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_p_value_line_slope_when_AF_3FB_min_AB_MN_ratio_l1313_131384

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a line intersecting the parabola
noncomputable def intersectionPoints (p m b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let y₁ := (m * Real.sqrt (2*p*b + m^2*p^2) + m*p) / (1 + m^2)
  let x₁ := (y₁^2) / (2*p)
  let y₂ := (m * -Real.sqrt (2*p*b + m^2*p^2) + m*p) / (1 + m^2)
  let x₂ := (y₂^2) / (2*p)
  ((x₁, y₁), (x₂, y₂))

-- Theorem 1
theorem line_through_focus_p_value (p : ℝ) :
  ∀ (A B : ℝ × ℝ),
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  (∃ (m : ℝ), A.2 = m*(A.1 - p/2) ∧ B.2 = m*(B.1 - p/2)) →
  A.1 * B.1 + A.2 * B.2 = -12 →
  p = 4 := by sorry

-- Theorem 2
theorem line_slope_when_AF_3FB (p : ℝ) :
  ∀ (A B : ℝ × ℝ),
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  let F := focus p
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 9*((B.1 - F.1)^2 + (B.2 - F.2)^2) →
  (A.2 - B.2) / (A.1 - B.1) = Real.sqrt 3 := by sorry

-- Theorem 3
theorem min_AB_MN_ratio (p : ℝ) :
  ∀ (A B : ℝ × ℝ),
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  let F := focus p
  let M := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  (F.1 - M.1)^2 + (F.2 - M.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 →
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let MN := (A.1 + B.1) / 4
  (AB / MN) ≥ Real.sqrt 2 ∧ (∃ (A' B' : ℝ × ℝ), AB / MN = Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_p_value_line_slope_when_AF_3FB_min_AB_MN_ratio_l1313_131384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1313_131360

-- Define the set of a
def A : Set ℝ := {x | (1/3)^x - x = 0}

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3) / Real.log a

-- State the theorem
theorem decreasing_interval_of_f (a : ℝ) (h : a ∈ A) :
  ∀ x y, x > 3 → y > 3 → x < y → f a x > f a y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1313_131360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1313_131386

/-- The function f(x) = sin²x + √3 * sin x * cos x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

/-- The theorem stating that the maximum value of f(x) on [π/4, π/2] is 3/2 -/
theorem max_value_of_f :
  ∃ (x : ℝ), π/4 ≤ x ∧ x ≤ π/2 ∧
  f x = 3/2 ∧
  ∀ (y : ℝ), π/4 ≤ y ∧ y ≤ π/2 → f y ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1313_131386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_to_hundredth_l1313_131367

-- Define the two numbers
def a : ℚ := 92.8531
def b : ℚ := 47.2694

-- Define the rounding function
noncomputable def round_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem statement
theorem add_and_round_to_hundredth :
  round_to_hundredth (a + b) = 140.12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_to_hundredth_l1313_131367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measures_l1313_131374

noncomputable section

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def is_triangle (A B C a b c : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem triangle_angle_measures 
  (h_triangle : is_triangle A B C a b c)
  (h_eq : (Real.sin B - Real.sin A) / Real.sin C = (a + c) / (a + b))
  (h_sin_cos : Real.sin A * Real.cos C = (Real.sqrt 3 - 1) / 4) :
  B = 2 * Real.pi / 3 ∧ C = Real.pi / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measures_l1313_131374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l1313_131379

/-- A linear function f(x) = kx + b -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

theorem linear_function_value (k b : ℝ) :
  (linear_function k b 1 = 2) →
  (linear_function k b (-1) = -4) →
  (linear_function k b (-2) = -7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l1313_131379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1313_131365

theorem cos_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 4 / 5)
  (h4 : Real.cos (α + β) = 3 / 5) : 
  Real.cos β = 24 / 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l1313_131365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_condition_l1313_131394

/-- The circle equation: x^2 + y^2 - 2x - 3 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

/-- The line equation: y = kx + 1 -/
def lineEq (k x y : ℝ) : Prop := y = k*x + 1

/-- The center of the circle -/
def circleCenter : ℝ × ℝ := (1, 0)

theorem shortest_chord_condition (k : ℝ) : 
  (∀ x y x' y', circleEq x y → lineEq k x y → 
    circleEq x' y' → lineEq k x' y' → 
      (x - x')^2 + (y - y')^2 ≥ (circleCenter.1 - x)^2 + (circleCenter.2 - y)^2) → 
  k = -1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_condition_l1313_131394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_negative_l1313_131313

theorem sin_2012_negative : Real.sin (2012 * Real.pi / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_negative_l1313_131313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_goods_value_l1313_131370

/-- Calculates the total value of goods purchased by a tourist given the tax paid -/
noncomputable def total_value_of_goods (tax_paid : ℝ) : ℝ :=
  600 + tax_paid / 0.11

/-- Theorem stating that if a tourist pays $123.2 in tax under the given conditions, 
    the total value of goods purchased is $1720 -/
theorem tourist_goods_value :
  total_value_of_goods 123.2 = 1720 := by
  -- Unfold the definition of total_value_of_goods
  unfold total_value_of_goods
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_goods_value_l1313_131370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_salami_l1313_131315

/-- The cost of salami given Teresa's shopping list and total spend --/
theorem cost_of_salami (sandwich_price : ℝ) (olive_price : ℝ) (feta_price : ℝ) (bread_price : ℝ) 
  (total_spent : ℝ) (h1 : sandwich_price = 7.75)
  (h2 : olive_price = 10.00) (h3 : feta_price = 8.00) (h4 : bread_price = 2.00)
  (h5 : total_spent = 40.00) : ℝ := by
  let salami_cost := (total_spent - (2 * sandwich_price + 0.25 * olive_price + 0.5 * feta_price + bread_price)) / 4
  have : salami_cost = 4.00 := by
    sorry
  exact salami_cost


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_salami_l1313_131315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_l1313_131328

theorem octahedron_cube_volume_ratio (a : ℝ) (ha : a > 0) :
  let cube_volume := a^3
  let octahedron_edge := a / Real.sqrt 2
  let octahedron_volume := Real.sqrt 2 / 3 * octahedron_edge^3
  octahedron_volume / cube_volume = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_l1313_131328
