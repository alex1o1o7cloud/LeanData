import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l535_53575

-- Define the function f(x) = 2^x + 2^(2-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp ((2 - x) * Real.log 2)

-- Theorem stating that the minimum value of f(x) is 4
theorem min_value_of_f :
  ∃ (x₀ : ℝ), f x₀ = 4 ∧ ∀ (x : ℝ), f x ≥ 4 := by
  -- We'll use x₀ = 1 as our minimum point
  use 1
  constructor
  · -- Show that f(1) = 4
    sorry
  · -- Show that for all x, f(x) ≥ 4
    intro x
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l535_53575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_specific_right_triangle_perimeter_l535_53565

/-- Represents a triangle with its properties -/
structure Triangle where
  circumcircleRadius : ℝ
  incircleRadius : ℝ
  perimeter : ℝ

/-- Predicate to check if a triangle is a right triangle -/
def IsRightTriangle (t : Triangle) : Prop :=
  sorry -- We'll leave this unspecified for now

/-- Given a right triangle with circumcircle radius R and incircle radius r, 
    its perimeter is 2(2R + r) -/
theorem right_triangle_perimeter 
  (t : Triangle) 
  (h_right : IsRightTriangle t) : 
  t.perimeter = 2 * (2 * t.circumcircleRadius + t.incircleRadius) :=
sorry

/-- The perimeter of a right triangle with circumcircle radius 14.5 cm 
    and incircle radius 6 cm is 70 cm -/
theorem specific_right_triangle_perimeter :
  ∃ (t : Triangle), 
    IsRightTriangle t ∧ 
    t.circumcircleRadius = 14.5 ∧ 
    t.incircleRadius = 6 ∧ 
    t.perimeter = 70 :=
by
  -- Construct the specific triangle
  let t : Triangle := {
    circumcircleRadius := 14.5,
    incircleRadius := 6,
    perimeter := 70
  }
  
  -- Prove that this triangle satisfies all conditions
  use t
  constructor
  · sorry -- Prove IsRightTriangle t
  · constructor
    · rfl -- t.circumcircleRadius = 14.5
    · constructor
      · rfl -- t.incircleRadius = 6
      · rfl -- t.perimeter = 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_specific_right_triangle_perimeter_l535_53565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_swappable_l535_53532

-- Define a type for cities
def City : Type := ℕ

-- Define a type for the railway connection list
def ConnectionList : Type := City → City → Prop

-- Define a property that a renumbering keeps the connection list correct
def CorrectRenumbering (original : ConnectionList) (renumbered : ConnectionList) : Prop :=
  ∀ (c1 c2 : City), original c1 c2 ↔ renumbered c1 c2

-- Define the property that any city can be renumbered to any other city
def CanRenumberAnyToAny (connections : ConnectionList) : Prop :=
  ∀ (M N : City), ∃ (renumbered : ConnectionList), 
    CorrectRenumbering connections renumbered ∧ (∀ c, renumbered M c ↔ connections N c)

-- Define the property that any two cities can be swapped
def CanSwapAny (connections : ConnectionList) : Prop :=
  ∀ (M N : City), ∃ (renumbered : ConnectionList),
    CorrectRenumbering connections renumbered ∧ 
    (∀ c, renumbered M c ↔ connections N c) ∧ 
    (∀ c, renumbered N c ↔ connections M c)

-- The main theorem
theorem not_always_swappable :
  ∃ (connections : ConnectionList),
    CanRenumberAnyToAny connections ∧ ¬CanSwapAny connections := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_swappable_l535_53532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_through_point_l535_53535

/-- A parabola passing through a given point with vertex at the origin -/
structure Parabola where
  /-- The x-coordinate of the point the parabola passes through -/
  x : ℝ
  /-- The y-coordinate of the point the parabola passes through -/
  y : ℝ

/-- The standard equations of a parabola -/
inductive ParabolaEquation
  | VerticalAxis (p : ℝ) : ParabolaEquation  -- y² = -2px
  | HorizontalAxis (p : ℝ) : ParabolaEquation  -- x² = 2py

/-- Checks if a given equation is valid for the parabola -/
def isValidEquation (p : Parabola) (eq : ParabolaEquation) : Prop :=
  match eq with
  | ParabolaEquation.VerticalAxis p' => p.y^2 = -2 * p' * p.x
  | ParabolaEquation.HorizontalAxis p' => p.x^2 = 2 * p' * p.y

theorem parabola_equation_through_point (p : Parabola) (h1 : p.x = -2) (h2 : p.y = 4) :
  (isValidEquation p (ParabolaEquation.VerticalAxis 4)) ∨
  (isValidEquation p (ParabolaEquation.HorizontalAxis (1/2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_through_point_l535_53535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l535_53506

-- Define the ∇ operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define the function h
def h (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem nabla_calculation :
  nabla (nabla 3 (h 2)) (h 3) = 31/19 := by
  -- Unfold definitions
  unfold nabla h
  -- Simplify the expression
  simp [pow_two]
  -- Perform numerical calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l535_53506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_correct_l535_53564

/-- The area of a circle with radius r not covered by an inscribed regular star decagon -/
noncomputable def uncovered_area (r : ℝ) : ℝ :=
  r^2 * (Real.pi - 5 * (Real.sqrt 5 + 1) / 2)

/-- Theorem stating that the uncovered area is correct -/
theorem uncovered_area_correct (r : ℝ) (h : r > 0) :
  uncovered_area r = r^2 * (Real.pi - 5 * (Real.sqrt 5 + 1) / 2) := by
  -- Unfold the definition of uncovered_area
  unfold uncovered_area
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_correct_l535_53564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_exists_l535_53502

theorem no_geometric_sequence_exists : ¬∃ (a : ℕ → ℝ) (q : ℝ),
  (∀ n : ℕ, a (n + 1) = q * a n) ∧  -- geometric sequence definition
  (a 1 + a 6 = 11) ∧  -- condition i (part 1)
  (a 3 * a 4 = 32 / 9) ∧  -- condition i (part 2)
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧  -- condition ii
  (∃ m : ℕ, m > 4 ∧  -- condition iii
    (2 / 3 * a (m - 1) - (a m ^ 2) = (a m ^ 2) - (a (m + 1) + 4 / 9))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_geometric_sequence_exists_l535_53502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l535_53577

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) / (x - 2) ≤ 0}
def B : Set ℝ := {x | x - 1 ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l535_53577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2pi_minus_alpha_l535_53509

theorem sin_2pi_minus_alpha (α : ℝ) 
  (h1 : Real.cos (α + Real.pi) = Real.sqrt 3 / 2)
  (h2 : Real.pi < α)
  (h3 : α < 3 * Real.pi / 2) : 
  Real.sin (2 * Real.pi - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2pi_minus_alpha_l535_53509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_equals_315_when_n_is_6_l535_53500

-- Define the sum of natural logarithms of divisors of 6^n
noncomputable def sum_log_divisors (n : ℕ) : ℝ :=
  (n * (n + 1)^2 / 2) * (Real.log 2 + Real.log 3)

-- Theorem statement
theorem sum_log_divisors_equals_315_when_n_is_6 :
  sum_log_divisors 6 = 315 := by
  -- Expand the definition of sum_log_divisors
  unfold sum_log_divisors
  -- Simplify the arithmetic expression
  simp [Real.log]
  -- Assert the approximate equality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_equals_315_when_n_is_6_l535_53500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_multiplicative_function_characterization_l535_53511

def is_periodic (f : ℕ+ → ℝ) : Prop :=
  ∀ x, f (x + 1019) = f x

def is_multiplicative (f : ℕ+ → ℝ) : Prop :=
  ∀ x y, f (x * y) = f x * f y

def is_coprime_to_1019 (x : ℕ+) : Prop :=
  Nat.Coprime x.val 1019

theorem periodic_multiplicative_function_characterization (f : ℕ+ → ℝ) 
  (h_periodic : is_periodic f) (h_multiplicative : is_multiplicative f) :
  (∀ x, is_coprime_to_1019 x → f x = 1) ∧ (∀ x, ¬is_coprime_to_1019 x → f x = 0) ∨
  (∀ x, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_multiplicative_function_characterization_l535_53511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_area_of_triangle_l535_53596

noncomputable section

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

theorem incircle_area_of_triangle
  (a b c : ℝ) (A B C : ℝ)
  (h_triangle : triangle_ABC a b c A B C)
  (h_a : a = 4)
  (h_cos_A : Real.cos A = 3/4)
  (h_sin_B : Real.sin B = 5 * Real.sqrt 7 / 16)
  (h_c : c > 4) :
  b = 5 ∧ Real.pi * (Real.sqrt 7 / 2)^2 = 7 * Real.pi / 4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_area_of_triangle_l535_53596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_b_for_all_real_domain_l535_53581

theorem greatest_integer_b_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, x^2 + b * x + 10 ≠ 0) →
  (∀ c : ℤ, c > b → c^2 ≥ 40) →
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_b_for_all_real_domain_l535_53581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l535_53543

theorem sqrt_calculations :
  (∃ x : ℝ, x = Real.sqrt 18 - Real.sqrt 8 + Real.sqrt 2 ∧ x = 2 * Real.sqrt 2) ∧
  (∃ y : ℝ, y = (Real.sqrt 48 - Real.sqrt 12) / Real.sqrt 3 ∧ y = 2) :=
by
  constructor
  · use 2 * Real.sqrt 2
    constructor
    · sorry -- Proof that Real.sqrt 18 - Real.sqrt 8 + Real.sqrt 2 = 2 * Real.sqrt 2
    · rfl
  · use 2
    constructor
    · sorry -- Proof that (Real.sqrt 48 - Real.sqrt 12) / Real.sqrt 3 = 2
    · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l535_53543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l535_53524

noncomputable section

-- Define the curve in polar coordinates
def ρ (φ : Real) : Real := 4 * φ

-- Define the range of φ
def φ_range : Set Real := { φ | 0 ≤ φ ∧ φ ≤ 3/4 }

-- Define the arc length formula for polar coordinates
noncomputable def arc_length (f : Real → Real) (range : Set Real) : Real :=
  ∫ φ in range, Real.sqrt ((f φ)^2 + (deriv f φ)^2)

-- Theorem statement
theorem arc_length_of_curve :
  arc_length ρ φ_range = 15/8 + Real.log 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l535_53524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_squared_distances_l535_53518

/-- Given a curve C in polar coordinates with equation ρ²(2 + cos 2θ) = 6,
    prove that for any two points A and B on C with angular coordinates α and α + π/2 respectively,
    the sum of the reciprocals of their squared distances from the origin is equal to 2/3 -/
theorem reciprocal_sum_squared_distances (ρ₁ ρ₂ α : ℝ) : 
  ρ₁^2 * (2 + Real.cos (2 * α)) = 6 →
  ρ₂^2 * (2 + Real.cos (2 * (α + π / 2))) = 6 →
  1 / ρ₁^2 + 1 / ρ₂^2 = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_squared_distances_l535_53518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_problem_l535_53563

/-- Two lines are parallel -/
def Parallel (l₁ l₂ : Line) : Prop := sorry

/-- A line intersects another line at a point -/
def LineIntersectsAt (l₁ l₂ : Line) (P : Point) : Prop := sorry

/-- Two angles are vertically opposite -/
def VerticallyOpposite (A B : Point) : Prop := sorry

/-- The measure of an angle in degrees -/
noncomputable def AngleMeasure (P : Point) : ℝ := sorry

/-- Given two parallel lines intersected by a third line, where one angle is 1/4 of another,
    prove that the vertically opposite angle to the smaller angle is 36 degrees. -/
theorem angle_measure_problem (p q r : Line) (A B C : Point) :
  Parallel p q →
  LineIntersectsAt r p A →
  LineIntersectsAt r q B →
  VerticallyOpposite A C →
  AngleMeasure A = (1 / 4 : ℝ) * AngleMeasure B →
  AngleMeasure C = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_problem_l535_53563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l535_53542

-- Define the constants for the tanks
def height_C : ℝ := 10
def circumference_C : ℝ := 8
def height_B : ℝ := 8
def circumference_B : ℝ := 10

-- Define the volume of a cylinder
noncomputable def volume (h : ℝ) (c : ℝ) : ℝ := (h * c^2) / (4 * Real.pi)

-- State the theorem
theorem tank_capacity_ratio :
  (volume height_C circumference_C) / (volume height_B circumference_B) = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l535_53542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_f_not_monotone_increasing_after_pi_over_four_l535_53545

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_monotone_increasing_on_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 4),
  ∀ y ∈ Set.Icc 0 (Real.pi / 4),
  x ≤ y → f x ≤ f y := by
  sorry

theorem f_not_monotone_increasing_after_pi_over_four :
  ∃ x y, x ∈ Set.Ioo (Real.pi / 4) Real.pi ∧
         y ∈ Set.Ioo (Real.pi / 4) Real.pi ∧
         x < y ∧ f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_f_not_monotone_increasing_after_pi_over_four_l535_53545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_implies_a_equals_one_l535_53527

noncomputable def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 + t * Real.sqrt 2 / 2, -4 + t * Real.sqrt 2 / 2)

noncomputable def intersection_points (a : ℝ) : ℝ × ℝ :=
  let t₁ := 4*Real.sqrt 2 + Real.sqrt 2*a - Real.sqrt (32*a + 16)
  let t₂ := 4*Real.sqrt 2 + Real.sqrt 2*a + Real.sqrt (32*a + 16)
  (t₁, t₂)

def geometric_progression (t₁ t₂ : ℝ) : Prop :=
  (t₁ - t₂)^2 = t₁ * t₂

theorem curve_line_intersection_implies_a_equals_one (a : ℝ) :
  (∃ x y : ℝ, curve_C a x y) →
  (∃ t : ℝ, curve_C a (line_l t).1 (line_l t).2) →
  (let (t₁, t₂) := intersection_points a; geometric_progression t₁ t₂) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_implies_a_equals_one_l535_53527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_is_two_hours_l535_53522

/-- Calculates the total round trip time given the speeds and time to work -/
noncomputable def round_trip_time (speed_to_work : ℝ) (speed_from_work : ℝ) (time_to_work_minutes : ℝ) : ℝ :=
  let time_to_work_hours := time_to_work_minutes / 60
  let distance := speed_to_work * time_to_work_hours
  let time_from_work_hours := distance / speed_from_work
  time_to_work_hours + time_from_work_hours

/-- Theorem stating that under the given conditions, the round trip time is 2 hours -/
theorem round_trip_is_two_hours :
  round_trip_time 50 110 82.5 = 2 := by
  -- Unfold the definition of round_trip_time
  unfold round_trip_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_is_two_hours_l535_53522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_schedule_theorem_l535_53555

/-- Represents Ivan Petrovich's daily work schedule and finances --/
structure IvanSchedule where
  workDaysPerMonth : ℕ := 21
  sleepHours : ℕ := 8
  workHours : ℝ
  hobbyHours : ℝ := 2 * workHours
  restHours : ℝ
  lessonRate : ℝ := 3000
  monthlyRent : ℝ := 14000
  monthlyLivingExpenses : ℝ := 70000

/-- Calculates the daily charity donation --/
noncomputable def dailyCharity (s : IvanSchedule) : ℝ :=
  s.restHours / 3 * 1000

/-- Calculates the monthly income --/
noncomputable def monthlyIncome (s : IvanSchedule) : ℝ :=
  s.workDaysPerMonth * s.workHours * s.lessonRate + s.monthlyRent

/-- Calculates the monthly expenditure --/
noncomputable def monthlyExpenditure (s : IvanSchedule) : ℝ :=
  s.monthlyLivingExpenses + s.workDaysPerMonth * dailyCharity s

/-- The main theorem stating Ivan's work hours and monthly charity donation --/
theorem ivan_schedule_theorem (s : IvanSchedule) :
  s.workHours = 2 ∧ 
  s.workDaysPerMonth * dailyCharity s = 70000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_schedule_theorem_l535_53555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_solution_l535_53582

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (3*x - 2) / (x - 6) ≤ 1
def inequality2 (x : ℝ) : Prop := 2*x^2 - x - 1 > 0

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-2) (1/2) ∪ Set.Ioo 1 6

-- Theorem statement
theorem inequalities_solution :
  {x : ℝ | inequality1 x ∧ inequality2 x} = solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_solution_l535_53582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l535_53554

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  d1 : ℝ  -- Length of first diagonal
  d2 : ℝ  -- Length of second diagonal

/-- Calculate the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ := (r.d1 * r.d2) / 2

/-- Calculate the side length of a rhombus given its diagonals -/
noncomputable def side_length (r : Rhombus) : ℝ := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)

/-- Calculate the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ := 4 * side_length r

theorem rhombus_area_and_perimeter (r : Rhombus) 
  (h1 : r.d1 = 18) (h2 : r.d2 = 26) : 
  area r = 234 ∧ perimeter r = 20 * Real.sqrt 10 := by
  sorry

-- Remove #eval statements as they are not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l535_53554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l535_53552

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - t x

-- State the theorem
theorem t_of_f_7 : t (f 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_7_l535_53552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l535_53548

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2)

theorem nested_function_evaluation :
  f (f (f (f (f 8)))) = 1 / 79228162514264337593543950336 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l535_53548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_comparison_l535_53562

/-- Given two points on a line with a negative slope, prove that the y-coordinate of the leftmost point is greater than the y-coordinate of the rightmost point. -/
theorem y_coordinate_comparison (y₁ y₂ : ℝ) : 
  ((-4, y₁) ∈ {p : ℝ × ℝ | p.2 = -p.1 + 3}) → 
  ((2, y₂) ∈ {p : ℝ × ℝ | p.2 = -p.1 + 3}) → 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_comparison_l535_53562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_average_is_19_over_2_l535_53549

-- Define the function f(x) = x^2 + log₂(x)
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x / Real.log 2

-- Define the interval [1, 4]
def I : Set ℝ := Set.Icc 1 4

-- Define the property of having an "average" M
def has_average (f : ℝ → ℝ) (I : Set ℝ) (M : ℝ) : Prop :=
  ∀ x₁, x₁ ∈ I → ∃! x₂, x₂ ∈ I ∧ (f x₁ + f x₂) / 2 = M

-- State the theorem
theorem f_average_is_19_over_2 : has_average f I (19/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_average_is_19_over_2_l535_53549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l535_53529

/-- The function f(x) = cos(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The theorem stating the maximum value of ω given the conditions -/
theorem max_omega_value (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 ≤ φ ∧ φ ≤ π) 
  (h3 : ∀ x, f ω φ x = - f ω φ (-x)) -- f is an odd function
  (h4 : ∀ x y, -π/4 ≤ x ∧ x < y ∧ y ≤ π/3 → f ω φ y < f ω φ x) -- f is monotonically decreasing in [-π/4, π/3]
  : ω ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l535_53529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l535_53541

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = π/3 ∧ 2 * sin (2 * t.A) + sin (2 * t.B + t.C) = sin t.C

/-- The area of the triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * sin t.C

/-- The perimeter of the triangle -/
def trianglePerimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  triangleArea t = 2 * Real.sqrt 3 / 3 ∧
  ∀ (t' : Triangle), TriangleConditions t' → trianglePerimeter t' ≤ 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l535_53541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l535_53558

/-- The largest possible distance between two points on given spheres -/
theorem largest_distance_between_spheres :
  let sphere1_center : ℝ × ℝ × ℝ := (4, -5, 10)
  let sphere1_radius : ℝ := 15
  let sphere2_center : ℝ × ℝ × ℝ := (-6, 20, -10)
  let sphere2_radius : ℝ := 50
  ∃ (p1 p2 : ℝ × ℝ × ℝ),
    (‖p1 - sphere1_center‖ = sphere1_radius) ∧
    (‖p2 - sphere2_center‖ = sphere2_radius) ∧
    (∀ (q1 q2 : ℝ × ℝ × ℝ),
      ‖q1 - sphere1_center‖ = sphere1_radius →
      ‖q2 - sphere2_center‖ = sphere2_radius →
      ‖q1 - q2‖ ≤ ‖p1 - p2‖) ∧
    ‖p1 - p2‖ = 65 + 25 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l535_53558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l535_53598

/-- The time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: The time for a train of length 360 meters, traveling at 54 km/hour, 
    to pass a bridge of length 140 meters is approximately 33.33 seconds -/
theorem train_bridge_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_to_pass_bridge 360 54 140 - 100/3| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l535_53598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l535_53597

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0)
  (h2 : ∀ x, -1 < x ∧ x < 2 ↔ a * x^2 + b * x + c > 0) :
  ∀ x, -2 < x ∧ x < 1 ↔ b * x^2 - a * x - c < 0 := by
  sorry

#check quadratic_inequality_solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l535_53597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_point_l535_53505

/-- Given a circle C with center (0, 1) and radius 1, and a point M at (2, 0),
    the maximum distance between M and any point on C is √5 + 1. -/
theorem max_distance_circle_point :
  ∃ (C : Set (ℝ × ℝ)) (M : ℝ × ℝ),
    (∀ (p : ℝ × ℝ), p ∈ C ↔ ((p.1^2 + (p.2 - 1)^2) = 1)) ∧
    (M = (2, 0)) ∧
    (∀ (N : ℝ × ℝ), N ∈ C → dist M N ≤ Real.sqrt 5 + 1) ∧
    (∃ (N : ℝ × ℝ), N ∈ C ∧ dist M N = Real.sqrt 5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_point_l535_53505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l535_53585

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domain
def domain : Set ℝ := Set.Icc 1 8

-- Define the condition
def satisfies_condition (x : ℝ) : Prop := 1 ≤ f x ∧ f x ≤ 2

-- State the theorem
theorem probability_theorem :
  (MeasureTheory.volume (Set.Icc 2 4 : Set ℝ)) / (MeasureTheory.volume (Set.Icc 1 8 : Set ℝ)) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l535_53585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l535_53593

open Real

noncomputable def Triangle (A B C a b c : ℝ) : Prop :=
  A + B + C = π ∧ 
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def area_triangle (A B C a b c : ℝ) : ℝ := 
  (1 / 2) * a * b * sin C

theorem triangle_properties 
  (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_sin : sin (A + C) = 2 * sin A * cos (A + B))
  (h_C : C = 3 * π / 4) :
  (∃ r : ℝ, a * r = b ∧ b * r = 2 * a) ∧
  (area_triangle A B C a b c = 2 → c = 2 * sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l535_53593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_eq_interval_l535_53517

/-- The proposition p for a given real number a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0

/-- The set of real numbers a for which p is false -/
def S : Set ℝ := {a : ℝ | ¬(p a)}

/-- The theorem stating that S is equal to the interval (-∞, 1] -/
theorem S_eq_interval : S = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_eq_interval_l535_53517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_40_l535_53586

/-- Represents the time it takes for a leak to empty a full cistern. -/
noncomputable def leak_empty_time (normal_fill_time hours_with_leak : ℝ) : ℝ :=
  let fill_rate := 1 / normal_fill_time
  let combined_rate := 1 / hours_with_leak
  let leak_rate := fill_rate - combined_rate
  1 / leak_rate

/-- 
Theorem: Given a cistern that normally fills in 8 hours, but takes 10 hours to fill with a leak,
the time it takes for the leak to empty a full cistern is 40 hours.
-/
theorem leak_empty_time_is_40 :
  leak_empty_time 8 10 = 40 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_is_40_l535_53586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_83_l535_53540

def f : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n+3 => 2 * f (n+2) - f (n+1) + 2*(n+3)

theorem f_10_equals_83 : f 10 = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_83_l535_53540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_minus_one_cube_equation_l535_53573

theorem sqrt_five_minus_one_cube_equation :
  let a : ℝ := Real.sqrt 5 - 1
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_minus_one_cube_equation_l535_53573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l535_53599

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (2 * x + 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-1/2 : ℝ) 1 ∪ {1}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ -1/2 → (f x ≤ 0 ↔ x ∈ solution_set) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l535_53599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equations_have_solutions_l535_53576

theorem at_least_two_equations_have_solutions (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f₁ := λ x : ℝ ↦ (x - a) * (x - b) - (x - c)
  let f₂ := λ x : ℝ ↦ (x - b) * (x - c) - (x - a)
  let f₃ := λ x : ℝ ↦ (x - c) * (x - a) - (x - b)
  ∃ (i j : Fin 3) (x y : ℝ), i ≠ j ∧
    (i.val = 0 → f₁ x = 0) ∧
    (i.val = 1 → f₂ x = 0) ∧
    (i.val = 2 → f₃ x = 0) ∧
    (j.val = 0 → f₁ y = 0) ∧
    (j.val = 1 → f₂ y = 0) ∧
    (j.val = 2 → f₃ y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equations_have_solutions_l535_53576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l535_53550

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 27)
  (sum_abcd : a + b + c + d = 46)
  (sum_abcdef : a + b + c + d + e + f = 65) :
  ∃ (odd_count : ℕ), odd_count ≥ 3 ∧ 
  (∃ (indices : Finset (Fin 6)), indices.card = odd_count ∧ 
    (∀ i ∈ indices, Odd (match i with
                         | 0 => a
                         | 1 => b
                         | 2 => c
                         | 3 => d
                         | 4 => e
                         | _ => f
                         ))) ∧
  (∀ (other_count : ℕ), other_count < odd_count →
    ¬∃ (other_indices : Finset (Fin 6)), other_indices.card = other_count ∧
      (∀ i ∈ other_indices, Odd (match i with
                                 | 0 => a
                                 | 1 => b
                                 | 2 => c
                                 | 3 => d
                                 | 4 => e
                                 | _ => f
                                 ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l535_53550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l535_53592

/-- The focus of a parabola y = ax^2 with directrix y = 1 has coordinates (0, -1) -/
theorem parabola_focus_coordinates (a : ℝ) :
  let parabola := fun x : ℝ => a * x^2
  let directrix := 1
  let focus := (0, -1)
  (∀ x, parabola x ≤ directrix → (x - focus.1)^2 + (parabola x - focus.2)^2 ≤ (parabola x - directrix)^2) ∧
  (∀ x y, y ≤ directrix → (x - focus.1)^2 + (y - focus.2)^2 > (y - directrix)^2 → y < parabola x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l535_53592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_is_19_35_percent_l535_53559

/-- Calculates the overall gain percentage for three items given their selling prices and gains. -/
noncomputable def overall_gain_percentage (sp1 sp2 sp3 g1 g2 g3 : ℝ) : ℝ :=
  let cp1 := sp1 - g1
  let cp2 := sp2 - g2
  let cp3 := sp3 - g3
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  let tg := tsp - tcp
  (tg / tcp) * 100

/-- Theorem stating that the overall gain percentage for the given items is 19.35%. -/
theorem overall_gain_is_19_35_percent :
  ∃ (ε : ℝ), abs (overall_gain_percentage 180 240 320 30 40 50 - 19.35) < ε ∧ ε > 0 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_is_19_35_percent_l535_53559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l535_53513

/-- A sequence defined by a₁ = 1 and aₙ = 2aₙ₋₁ + 1 for n ≥ 2 -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case to handle n = 0
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + 1

/-- Theorem stating that aₙ = 2ⁿ - 1 for all positive integers n -/
theorem a_closed_form (n : ℕ) (h : n > 0) : a n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l535_53513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l535_53583

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x / Real.exp x

-- State the theorem
theorem f_decreasing_on_interval (a b : ℝ) (h1 : a < b) (h2 : b < 1) : f a > f b := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l535_53583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_ordering_l535_53547

noncomputable def eccentricity_ellipse (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))
noncomputable def eccentricity_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))
def eccentricity_parabola : ℝ := 1

theorem eccentricity_ordering (e₁ e₂ e₃ a b c : ℝ) 
  (h₁ : 0 ≤ e₁ ∧ e₁ < 1)
  (h₂ : e₂ > 1)
  (h₃ : e₃ = 1)
  (ha : a = (5 : ℝ) ^ (Real.log e₁ / Real.log 3))
  (hb : b = (1/5 : ℝ) ^ (Real.log e₂ / Real.log (1/2)))
  (hc : c = (5 : ℝ) ^ (Real.log e₃ / Real.log (1/2)))
  : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_ordering_l535_53547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l535_53512

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- Minimum distance from a point on the curve to the line -/
noncomputable def min_distance : ℝ := Real.sqrt 2

theorem min_distance_proof :
  ∀ (P : ℝ × ℝ), (∃ x, P.1 = x ∧ P.2 = f x) →
  ∃ (Q : ℝ × ℝ), line_eq Q.1 Q.2 ∧
  ∀ (R : ℝ × ℝ), line_eq R.1 R.2 →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) ∧
  ∃ (P : ℝ × ℝ), (∃ x, P.1 = x ∧ P.2 = f x) ∧
  ∃ (Q : ℝ × ℝ), line_eq Q.1 Q.2 ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = min_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l535_53512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_l535_53514

/-- Definition of the function f_n(x) -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => (Finset.range k).prod (λ i => (x + i)) / (Nat.factorial k))

/-- Theorem stating that the roots of f_n(x) = 0 are the integers from -1 to -n -/
theorem roots_of_f (n : ℕ) (h : n > 0) :
  ∀ x : ℝ, f n x = 0 ↔ ∃ k : ℕ, k ≥ 1 ∧ k ≤ n ∧ x = -k := by
  sorry

#check roots_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_l535_53514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_a_l535_53531

theorem divisor_of_a (a b : ℕ) (x : ℕ)
  (h1 : a % x = 3)
  (h2 : b % 6 = 5)
  (h3 : (a * b) % 48 = 15) :
  x = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_a_l535_53531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l535_53523

-- Define the given values
def train_speed : ℝ := 63  -- km/hr
def man_speed : ℝ := 3     -- km/hr
def crossing_time : ℝ := 26.997840172786177  -- seconds

-- Define the function to calculate the train length
noncomputable def calculate_train_length (train_speed man_speed crossing_time : ℝ) : ℝ :=
  let relative_speed := (train_speed - man_speed) * 1000 / 3600  -- Convert to m/s
  relative_speed * crossing_time

-- Theorem statement
theorem train_length_calculation :
  ∃ (length : ℝ), 
    |calculate_train_length train_speed man_speed crossing_time - length| < 0.01 ∧ 
    |length - 1349.89| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l535_53523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_l535_53515

theorem set_average (R : Finset ℚ) (h_nonempty : R.Nonempty) : 
  (∃ (m : ℕ) (b₁ bₘ : ℚ), b₁ ∈ R ∧ bₘ ∈ R ∧ 
    (∀ x ∈ R, b₁ ≤ x ∧ x ≤ bₘ) ∧
    (R.sum id - bₘ) / (R.card - 1 : ℚ) = 45 ∧
    (R.sum id - b₁ - bₘ) / (R.card - 2 : ℚ) = 50 ∧
    (R.sum id - b₁) / (R.card - 1 : ℚ) = 55 ∧
    bₘ = b₁ + 85) →
  R.sum id / R.card = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_l535_53515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l535_53508

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Theorem statement
theorem polar_to_cartesian :
  ∀ x y : ℝ, (∃ θ : ℝ, x = C θ * Real.cos θ ∧ y = C θ * Real.sin θ) ↔ x^2 + (y - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_l535_53508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_133_l535_53526

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, (11 : ℤ)^(n + 2) + (12 : ℤ)^(2*n + 1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_133_l535_53526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l535_53528

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/4) - Real.sqrt 2 / 2

theorem f_properties :
  -- Smallest positive period
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  -- Range of cos(B) in triangle ABC
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → b^2 = a*c →
    (1/2 : ℝ) ≤ (a^2 + c^2 - b^2) / (2*a*c) ∧ (a^2 + c^2 - b^2) / (2*a*c) < 1) ∧
  -- Maximum value of f(B)
  (∃ (M : ℝ), M = -Real.sqrt 2 / 2 ∧
    ∀ (B : ℝ), 0 < B → B ≤ Real.pi/2 → f B ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l535_53528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l535_53520

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.ofReal a + 2 * Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_fraction_l535_53520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_equality_l535_53589

open BigOperators

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_product_equality :
  (∏ k in Finset.range 148, (fib (k + 3) / fib (k + 1) - fib (k + 3) / fib (k + 5))) =
  (fib 150 * (fib 151 + fib 149)) / fib 152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_equality_l535_53589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_15_l535_53503

/-- Given a markup percentage and a profit percentage after discount,
    calculate the discount percentage. -/
noncomputable def calculate_discount_percentage (markup_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let marked_price := 1 + markup_percent / 100
  let selling_price := 1 + profit_percent / 100
  (marked_price - selling_price) / marked_price * 100

/-- Theorem stating that with a 40% markup and 19% profit after discount,
    the discount percentage is 15%. -/
theorem discount_percentage_is_15 :
  calculate_discount_percentage 40 19 = 15 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_discount_percentage 40 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_15_l535_53503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_extreme_values_l535_53572

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / Real.exp x + x + 1

-- Theorem 1: Value of a when tangent line is parallel to y = 2x + 3
theorem tangent_line_parallel (a : ℝ) :
  (deriv (f a)) 1 = 2 → a = -Real.exp 1 := by sorry

-- Theorem 2: Extreme values of f
theorem extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, f a x ≠ f a y → ¬(∀ z : ℝ, f a z ≥ f a x ∨ f a z ≥ f a y)) ∧
  (a > 0 → 
    (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) ∧
    (∀ x : ℝ, f a x ≥ f a (Real.log a)) ∧
    (f a (Real.log a) = Real.log a + 2) ∧
    (¬∃ x : ℝ, ∀ y : ℝ, f a x ≥ f a y)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_extreme_values_l535_53572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_ratio_sum_l535_53544

/-- Calculates the number of rectangles on an n x n chessboard -/
def num_rectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Calculates the number of squares on an n x n chessboard -/
def num_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Represents a fraction as a pair of natural numbers -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a fraction to its lowest terms -/
def simplify_fraction (f : Fraction) : Fraction :=
  sorry

/-- Calculates the sum of numerator and denominator of a fraction -/
def sum_num_denom (f : Fraction) : ℕ :=
  f.numerator + f.denominator

/-- Main theorem: For a 7x7 chessboard, the sum of numerator and denominator
    of the simplified ratio of squares to rectangles is 33 -/
theorem chessboard_ratio_sum :
  let squares := num_squares 7
  let rectangles := num_rectangles 7
  let ratio := simplify_fraction { numerator := squares, denominator := rectangles }
  sum_num_denom ratio = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_ratio_sum_l535_53544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l535_53534

noncomputable def q (x : ℝ) : ℝ := (8/5) * (x^3 + 3*x^2 - x - 3)

theorem q_satisfies_conditions :
  q 1 = 0 ∧ q (-1) = 0 ∧ q (-3) = 0 ∧ q 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l535_53534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coordinates_l535_53551

/-- Predicate to check if B is symmetric to A with respect to P -/
def is_symmetric (A B P : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

/-- Given points A and P in a 2D plane, find the coordinates of point B that is symmetric to A with respect to P. -/
theorem symmetric_point_coordinates (A P B : ℝ × ℝ) : 
  A = (1, 2) → P = (3, 4) → is_symmetric A B P → B = (5, 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_coordinates_l535_53551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l535_53530

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.log ((-x + 1) : ℝ) / Real.log (1/2 : ℝ)
  else Real.log ((x + 1) : ℝ) / Real.log (1/2 : ℝ)

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  (∀ a, f (a - 1) - f 1 < 0 → a > 2 ∨ a < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l535_53530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l535_53504

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

-- Define the line l
def line_l (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x y a : ℝ) : ℝ :=
  abs (x + 4 * y - a - 4) / Real.sqrt 17

theorem intersection_points_and_max_distance :
  (∃ θ₁ θ₂ : ℝ, curve_C θ₁ = (3, 0) ∧ curve_C θ₂ = (-21/25, 24/25) ∧
    (∀ t : ℝ, line_l (-1) t ≠ curve_C θ₁ → line_l (-1) t ≠ curve_C θ₂ →
      ∃ θ : ℝ, line_l (-1) t = curve_C θ)) ∧
  (∀ θ : ℝ, distance_point_to_line (3 * Real.cos θ) (Real.sin θ) (-16) ≤ Real.sqrt 17) ∧
  (∃ θ : ℝ, distance_point_to_line (3 * Real.cos θ) (Real.sin θ) (-16) = Real.sqrt 17) ∧
  (∀ θ : ℝ, distance_point_to_line (3 * Real.cos θ) (Real.sin θ) 8 ≤ Real.sqrt 17) ∧
  (∃ θ : ℝ, distance_point_to_line (3 * Real.cos θ) (Real.sin θ) 8 = Real.sqrt 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l535_53504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l535_53501

/-- The side length of both polygons -/
def side_length : ℝ := 3

/-- The area between inscribed and circumscribed circles of a regular hexagon -/
noncomputable def hexagon_area (s : ℝ) : ℝ := Real.pi * (s^2 * (4 - 3))

/-- The area between inscribed and circumscribed circles of a regular octagon -/
noncomputable def octagon_area (s : ℝ) : ℝ := Real.pi * (s^2 * ((2 + Real.sqrt 2)^2 - (Real.sqrt 2 + 1)^2))

/-- Theorem stating that the areas are equal -/
theorem equal_areas : hexagon_area side_length = octagon_area side_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l535_53501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_iff_n_in_set_l535_53588

theorem polynomial_property_iff_n_in_set (n : ℕ+) :
  (∃ P : Polynomial ℤ, 
    (Polynomial.degree P = n) ∧
    (P.eval 0 = 0) ∧
    (∃ (S : Finset ℤ), S.card = n ∧ ∀ x ∈ S, P.eval x = n))
  ↔ n ∈ ({1, 2, 3, 4} : Finset ℕ+) :=
by sorry

#check polynomial_property_iff_n_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_iff_n_in_set_l535_53588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_two_value_l535_53536

/-- Given a cubic polynomial p(x) = x^3 + 2ax^2 + 3bx + 4c with real coefficients a, b, c 
    where a < b < c, and a polynomial h(x) whose roots are the squares of the roots of p(x),
    prove that h(2) = (2 + 2a + 3b + c) / c^2 -/
theorem h_two_value (a b c : ℝ) (h_order : a < b ∧ b < c) :
  let p : ℝ → ℝ := fun x ↦ x^3 + 2*a*x^2 + 3*b*x + 4*c
  let h : ℝ → ℝ := fun x ↦ 
    let s := (2 + 2*a + 3*b + c) / c^2
    x^3 - (s + 1)*x^2 + s*x
  h 2 = (2 + 2*a + 3*b + c) / c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_two_value_l535_53536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multicolor_triangle_exists_l535_53556

/-- A coloring of edges in a complete graph -/
def Coloring (n : ℕ) := Fin (3*n+1) → Fin (3*n+1) → Fin 3

/-- A coloring is valid if each vertex has exactly n edges of each color -/
def valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ v : Fin (3*n+1), ∀ color : Fin 3,
    (Finset.filter (λ w ↦ c v w = color) (Finset.univ : Finset (Fin (3*n+1)))).card = n

/-- There exists a triangle with all three colors -/
def has_multicolor_triangle (n : ℕ) (c : Coloring n) : Prop :=
  ∃ u v w : Fin (3*n+1),
    u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c u v ≠ c v w ∧ c v w ≠ c u w ∧ c u v ≠ c u w

/-- Main theorem: Every valid coloring has a multicolor triangle -/
theorem multicolor_triangle_exists (n : ℕ) (c : Coloring n) 
  (h : valid_coloring n c) : has_multicolor_triangle n c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multicolor_triangle_exists_l535_53556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l535_53569

-- Define the line
def line (t : ℝ) : ℝ × ℝ := (t + 1, t)

-- Define the circle
noncomputable def circle_param (θ : ℝ) : ℝ × ℝ := (2 + Real.cos θ, Real.sin θ)

-- Define the circle's center
def circle_center : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem line_circle_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  (∃ (θ₁ θ₂ : ℝ), line t₁ = circle_param θ₁ ∧ line t₂ = circle_param θ₂) ∧
  (∀ (t : ℝ), line t ≠ circle_center) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l535_53569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_preservation_rectangular_prism_l535_53594

/-- Theorem: Volume preservation in a rectangular prism with length reduction -/
theorem volume_preservation_rectangular_prism
  (L W H : ℝ) -- Original length, width, and height
  (L' W' H' : ℝ) -- New length, width, and height
  (h_positive : L > 0 ∧ W > 0 ∧ H > 0) -- Positive dimensions
  (h_length_reduction : L' = 0.8 * L) -- 20% length reduction
  (h_volume_preserved : L * W * H = L' * W' * H') -- Volume preservation
  : W' = 1.25 * W ∧ H' = H := by
  sorry

#check volume_preservation_rectangular_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_preservation_rectangular_prism_l535_53594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l535_53587

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define the line l passing through P
noncomputable def line_l (t θ : ℝ) : ℝ × ℝ := (-2 + t * Real.cos θ, t * Real.sin θ)

-- Define points A and B on the line l
structure Intersection where
  t : ℝ
  θ : ℝ
  on_circle : circleC (line_l t θ).1 (line_l t θ).2

-- State the theorem
theorem length_AB (A B : Intersection) (h : 8 * A.t = 5 * (B.t - A.t)) :
  Real.sqrt ((line_l B.t B.θ).1 - (line_l A.t A.θ).1)^2 + 
             ((line_l B.t B.θ).2 - (line_l A.t A.θ).2)^2 = 8 * Real.sqrt 5 / 5 := by
  sorry

#check length_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l535_53587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_parallel_side_length_l535_53566

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: In a trapezium with one parallel side of 20 cm, a distance between parallel sides of 13 cm,
    and an area of 247 square centimeters, the length of the other parallel side is 18 cm -/
theorem other_parallel_side_length :
  ∀ x : ℝ, trapezium_area 20 x 13 = 247 → x = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_parallel_side_length_l535_53566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_line_and_perpendicular_l535_53595

/-- The intersection point of a line and its perpendicular line passing through a given point -/
theorem intersection_of_line_and_perpendicular (m : ℝ) (b : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  let line1 := λ x : ℝ => m * x + b
  let perpendicular_slope := -1 / m
  let line2 := λ x : ℝ => perpendicular_slope * (x - x₀) + y₀
  let intersection_x := (y₀ - b + m * x₀) / (m + 1/m)
  let intersection_y := line1 intersection_x
  (line1 intersection_x = line2 intersection_x) ∧ 
  (intersection_x = -9/5 ∧ intersection_y = 7/5) :=
by
  sorry

/-- The specific case for the given problem -/
example : ∃ x y : ℝ, 
  x = -9/5 ∧ y = 7/5 ∧
  y = 2*x + 5 ∧
  y = -1/2*(x - 3) + (-1) :=
by
  use -9/5, 7/5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_line_and_perpendicular_l535_53595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_30_60_90_triangle_with_altitude_5_l535_53525

/-- A right triangle with angles 30°, 60°, and 90° -/
structure RightTriangle30_60_90 where
  /-- The altitude to the hypotenuse -/
  altitude : ℝ
  /-- Assumption that the altitude is positive -/
  altitude_pos : altitude > 0

/-- The area of a right triangle with angles 30°, 60°, and 90° -/
noncomputable def area (t : RightTriangle30_60_90) : ℝ := 25 * Real.sqrt 3 / 3

/-- Theorem stating that the area of a right triangle with angles 30°, 60°, and 90°
    and altitude to the hypotenuse of 5 units is 25√3/3 square units -/
theorem area_of_30_60_90_triangle_with_altitude_5 :
  ∀ t : RightTriangle30_60_90, t.altitude = 5 → area t = 25 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_30_60_90_triangle_with_altitude_5_l535_53525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_roots_integer_part_l535_53580

noncomputable def nested_sqrt (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | n + 1 => Real.sqrt (6 + nested_sqrt n)

noncomputable def nested_cbrt (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | n + 1 => (6 + nested_cbrt n) ^ (1/3 : ℝ)

theorem nested_roots_integer_part :
  ⌊nested_sqrt 100 + nested_cbrt 100⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_roots_integer_part_l535_53580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l535_53557

/-- The eccentricity of a hyperbola with the given properties is √2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (y - (-b^2/a)) / (x - (-c)) = Real.sqrt 2 / 2 ∧ 
    (y - (b^2/a)) / (x - c) = Real.sqrt 2 / 2 → 
    c^2 = a^2 + b^2) →
  c / a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l535_53557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_l535_53521

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def non_collinear (v w : V) : Prop :=
  ∀ (r : ℝ), r • v ≠ w

theorem vector_expression (e₁ e₂ a b c : V)
  (h_non_collinear : non_collinear e₁ e₂)
  (h_a : a = e₁ + e₂)
  (h_b : b = 2 • e₁ - e₂)
  (h_c : c = e₁ + 2 • e₂) :
  c = (5/3) • a - (1/3) • b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_l535_53521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_zero_sum_of_roots_l535_53553

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_zero 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, QuadraticPolynomial a b c (x^3 - x) ≥ QuadraticPolynomial a b c (x^2 - 1)) : 
  b = 0 :=
sorry

-- Corollary for the sum of roots
theorem sum_of_roots 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, QuadraticPolynomial a b c (x^3 - x) ≥ QuadraticPolynomial a b c (x^2 - 1)) : 
  (- b) / (2 * a) = 0 :=
by
  have h_b_zero : b = 0 := sum_of_roots_zero a b c h
  rw [h_b_zero]
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_zero_sum_of_roots_l535_53553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_of_equation_l535_53537

theorem real_roots_of_equation : 
  let f : ℝ → ℝ := λ x => x^10 + 36*x^6 + 13*x^2 - 13*x^8 - x^4 - 36
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_of_equation_l535_53537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_375_l535_53570

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : 0 < side_length

/-- A right pyramid with a regular hexagon base -/
structure RightPyramid where
  base : RegularHexagon
  height : ℝ
  height_pos : 0 < height

/-- Volume of a right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (3 * Real.sqrt 3 * p.base.side_length^2 * p.height) / 2

/-- Main theorem -/
theorem pyramid_volume_is_375 (p : RightPyramid) 
  (h1 : p.base.side_length = 5)
  (h2 : p.height = 10 * Real.sqrt 3 / 3) : 
  volume p = 375 := by
  sorry

#check pyramid_volume_is_375

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_375_l535_53570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_fixed_point_l535_53567

-- Define the curves C₁ and C₂
def C₁ (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0
def C₂ (x y : ℝ) : Prop := x^2 / 5 + y^2 / 3 = 1

-- Define the line l
def l (x : ℝ) : Prop := x = Real.sqrt 2 / 2

-- State that C₁ and C₂ have the same foci
axiom same_foci : ∃ f₁ f₂ : ℝ × ℝ, (∀ x y, C₁ 1 1 x y → (x - f₁.1)^2 + y^2 = (x - f₂.1)^2 + y^2) ∧
                                   (∀ x y, C₂ x y → (x - f₁.1)^2 + y^2 = (x - f₂.1)^2 + y^2)

-- State the eccentricity relationship
axiom eccentricity_relation : ∃ e₁ e₂ : ℝ, e₁ = Real.sqrt 5 * e₂ ∧
  (∀ a b, (∃ x y, C₁ a b x y) → e₁ = Real.sqrt (a^2 + b^2) / a) ∧
  e₂ = Real.sqrt 2 / Real.sqrt 5

-- Main theorem
theorem curve_and_fixed_point :
  -- C₁ equation
  (∀ x y, C₁ 1 1 x y ↔ x^2 - y^2 = 1) ∧
  -- Fixed point property
  ∃ p : ℝ × ℝ, p = (3 * Real.sqrt 2 / 4, 0) ∧
    ∀ A B C : ℝ × ℝ,
      -- A is on the right branch of C₁
      C₁ 1 1 A.1 A.2 ∧ A.1 > 0 →
      -- B is the intersection of AF and C₁
      C₁ 1 1 B.1 B.2 ∧ ∃ t : ℝ, B = A + t • (A - (1, 0)) →
      -- C is on line l
      l C.1 ∧
      -- BC is perpendicular to l
      (B.1 - C.1) * 0 + (B.2 - C.2) * 1 = 0 →
      -- AC passes through p
      ∃ s : ℝ, A + s • (C - A) = p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_fixed_point_l535_53567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_range_of_x_l535_53560

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 1)

-- Define the function to be minimized
noncomputable def f (a b : ℝ) : ℝ := 1/a + 4/b

-- Theorem 1: The minimum value of f(a,b) is 9
theorem min_value_is_nine : 
  ∃ (m : ℝ), (∀ a b, a > 0 → b > 0 → a + b = 1 → f a b ≥ m) ∧ 
             (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ f a b = m) ∧ 
             m = 9 := by
  sorry

-- Theorem 2: The range of x for which 9 ≥ |2x-1| - |x+1| is [-7, 11]
theorem range_of_x : 
  ∀ x, (9 ≥ |2*x - 1| - |x + 1|) ↔ (-7 ≤ x ∧ x ≤ 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_range_of_x_l535_53560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_earnings_proof_l535_53579

/-- Calculates the earnings per widget given the hourly wage, work hours, total earnings, and number of widgets. -/
noncomputable def earnings_per_widget (hourly_wage : ℝ) (work_hours : ℝ) (total_earnings : ℝ) (num_widgets : ℝ) : ℝ :=
  (total_earnings - hourly_wage * work_hours) / num_widgets

/-- Proves that the earnings per widget is $0.16 under the given conditions. -/
theorem widget_earnings_proof :
  let hourly_wage : ℝ := 12.5
  let work_hours : ℝ := 40
  let total_earnings : ℝ := 660
  let num_widgets : ℝ := 1000
  earnings_per_widget hourly_wage work_hours total_earnings num_widgets = 0.16 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_earnings_proof_l535_53579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_puppy_weight_difference_l535_53533

theorem cat_puppy_weight_difference
  (puppy_weight : ℝ)
  (cat_weight : ℝ)
  (num_puppies : ℕ)
  (num_cats : ℕ)
  (h1 : puppy_weight = 7.5)
  (h2 : cat_weight = 2.5)
  (h3 : num_puppies = 4)
  (h4 : num_cats = 14) :
  (cat_weight * (num_cats : ℝ)) - (puppy_weight * (num_puppies : ℝ)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_puppy_weight_difference_l535_53533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l535_53519

-- Define the Sphere structure
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the necessary functions
def is_tangent_to_three_sides (s : Sphere) (width : ℝ) (depth : ℝ) (height : ℝ) : Prop :=
  sorry

def is_tangent_to_sphere (s1 : Sphere) (s2 : Sphere) : Prop :=
  sorry

def is_tangent_to_top_face (s : Sphere) (large_s : Sphere) (height : ℝ) : Prop :=
  sorry

theorem box_height_with_spheres :
  let box_width : ℝ := 6
  let large_sphere_radius : ℝ := 3
  let small_sphere_radius : ℝ := 1.5
  let num_side_small_spheres : ℕ := 8
  let h : ℝ := box_width / 2 + large_sphere_radius + large_sphere_radius + small_sphere_radius
  ∀ (box_height : ℝ),
    (∃ (large_sphere : Sphere) (small_spheres : Finset Sphere) (top_sphere : Sphere),
      large_sphere.radius = large_sphere_radius ∧
      small_spheres.card = num_side_small_spheres ∧
      (∀ s ∈ small_spheres, s.radius = small_sphere_radius) ∧
      top_sphere.radius = small_sphere_radius ∧
      (∀ s ∈ small_spheres, is_tangent_to_three_sides s box_width box_width box_height) ∧
      (∀ s ∈ small_spheres, is_tangent_to_sphere s large_sphere) ∧
      is_tangent_to_top_face top_sphere large_sphere box_height) →
    box_height = h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_with_spheres_l535_53519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_right_angle_l535_53568

/-- Hyperbola C -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- Circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Tangent line l to circle O at point P(x₀, y₀) -/
def tangent_line (x y x₀ y₀ : ℝ) : Prop := x * x₀ + y * y₀ = 2

/-- Theorem: The angle ∠AOB is always 90° -/
theorem angle_AOB_is_right_angle 
  (x₀ y₀ xA yA xB yB : ℝ) 
  (hx₀y₀ : x₀ * y₀ ≠ 0)
  (hP : circle_O x₀ y₀)
  (hA : hyperbola xA yA)
  (hB : hyperbola xB yB)
  (hlA : tangent_line xA yA x₀ y₀)
  (hlB : tangent_line xB yB x₀ y₀)
  (hAB_distinct : (xA, yA) ≠ (xB, yB)) :
  (xA * xB + yA * yB) / (Real.sqrt (xA^2 + yA^2) * Real.sqrt (xB^2 + yB^2)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AOB_is_right_angle_l535_53568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l535_53590

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

/-- The sequence a_n as defined in the problem -/
noncomputable def a : ℕ → ℝ
  | 0 => f 1  -- a_1 = f(1)
  | n + 1 => f (a n)  -- a_{n+1} = f(a_n)

/-- Main theorem: The 2015th term of the sequence equals 1/2016 -/
theorem a_2015_value : a 2014 = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l535_53590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_sixth_l535_53578

open Real BigOperators

noncomputable def series_sum : ℝ := ∑' n, (n^3 + n^2 - n) / (Nat.factorial (n + 3))

theorem series_sum_equals_one_sixth : series_sum = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_sixth_l535_53578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_values_l535_53510

def is_sum_of_even_distinct_integers (n : ℕ) : Prop :=
  ∃ (k : ℕ) (S : Finset ℕ),
    (2 * k = S.card) ∧
    (∀ i ∈ S, i ≤ 2000) ∧
    (n = S.sum id)

theorem excluded_values (n : ℕ) :
  is_sum_of_even_distinct_integers n →
  n ∉ ({1, 2, 200098, 200099} : Set ℕ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_values_l535_53510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_proof_l535_53507

/-- Represents the capacity of a disk in MB -/
def disk_capacity : ℚ := 1.44

/-- Represents the number of 1.0 MB files -/
def files_1mb : ℕ := 5

/-- Represents the number of 0.5 MB files -/
def files_0_5mb : ℕ := 10

/-- Represents the number of 0.3 MB files -/
def files_0_3mb : ℕ := 20

/-- Calculates the minimum number of disks needed to store all files -/
def min_disks : ℕ := 15

/-- Theorem stating that the minimum number of disks needed is 15 -/
theorem min_disks_proof :
  let total_storage : ℚ := (files_1mb : ℚ) * 1 + (files_0_5mb : ℚ) * 0.5 + (files_0_3mb : ℚ) * 0.3
  (total_storage / disk_capacity).ceil = min_disks ∧
  ∀ n : ℕ, n < min_disks → 
    total_storage > (n : ℚ) * disk_capacity :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_disks_proof_l535_53507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_minimizes_sum_of_products_l535_53571

/-- Given a triangle ABC with centroid G, for any point P in the plane,
    AP·AG + BP·BG + CP·CG is minimized when P coincides with G,
    and the minimum value is (a² + b² + c²) / 3 where a, b, c are side lengths of ABC. -/
theorem triangle_centroid_minimizes_sum_of_products (A B C G P : EuclideanSpace ℝ (Fin 2)) 
  (a b c : ℝ) (h_centroid : G = (1/3 : ℝ) • (A + B + C)) (h_sides : a = dist B C ∧ b = dist A C ∧ c = dist A B) :
  dist A P * dist A G + dist B P * dist B G + dist C P * dist C G ≥ (a^2 + b^2 + c^2) / 3 ∧
  (dist A P * dist A G + dist B P * dist B G + dist C P * dist C G = (a^2 + b^2 + c^2) / 3 ↔ P = G) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_minimizes_sum_of_products_l535_53571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l535_53561

/-- Definition of the ellipse E -/
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of point A -/
noncomputable def point_A : ℝ × ℝ := (0, 1)

/-- Definition of point M on the ellipse -/
noncomputable def point_M (k : ℝ) : ℝ × ℝ := (-8*k/(1+4*k^2), (1-4*k^2)/(1+4*k^2))

/-- Definition of point N on the ellipse -/
noncomputable def point_N (k : ℝ) : ℝ × ℝ := (8*k/(k^2+4), (k^2-4)/(k^2+4))

/-- The fixed point Q -/
noncomputable def point_Q : ℝ × ℝ := (0, -5/3)

/-- Main theorem -/
theorem ellipse_fixed_point (k : ℝ) (hk : k > 0) :
  let M := point_M k
  let N := point_N k
  ellipse M.1 M.2 ∧ ellipse N.1 N.2 ∧
  (M.2 - point_A.2) * (N.1 - M.1) = -(N.2 - M.2) * (M.1 - point_A.1) →
  ∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ 
    (1 - t) * M.1 + t * N.1 = point_Q.1 ∧ 
    (1 - t) * M.2 + t * N.2 = point_Q.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l535_53561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_vendor_problem_l535_53538

/-- Represents the percentage of apples remaining after each operation -/
noncomputable def apples_remaining (initial_percent : ℝ) (sell_percent : ℝ) (throw_percent : ℝ) : ℝ :=
  initial_percent * (1 - sell_percent / 100) * (1 - throw_percent / 100)

/-- Represents the total percentage of apples thrown away -/
noncomputable def total_thrown (day1_throw : ℝ) (day2_throw : ℝ) : ℝ :=
  day1_throw + day2_throw

theorem apple_vendor_problem :
  let day1_sell := (30 : ℝ)
  let day1_throw := (20 : ℝ)
  let total_throw := (42 : ℝ)
  let day1_remain := apples_remaining 100 day1_sell day1_throw
  let day2_throw := total_throw - (100 - day1_remain)
  let day2_sell_percent := (day1_remain - day2_throw) / day1_remain * 100
  day2_sell_percent = 50 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_vendor_problem_l535_53538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coin_flip_probability_l535_53584

theorem five_coin_flip_probability : 
  let n : ℕ := 5  -- number of coins
  let p : ℚ := 1/2  -- probability of heads (or tails) for a fair coin
  let favorable_outcomes : ℕ := 2 * (Nat.choose n n + Nat.choose n (n-1))  -- outcomes with at least 4 same face
  let total_outcomes : ℕ := 2^n  -- total possible outcomes
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coin_flip_probability_l535_53584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_empty_time_l535_53591

/-- Given a cistern with the following properties:
  * It normally fills in 8 hours
  * With a leak, it takes 10 hours to fill
  This function calculates the time it takes for the leak to empty a full cistern -/
noncomputable def leak_empty_time (normal_fill_time : ℝ) (leak_fill_time : ℝ) : ℝ :=
  let normal_rate := 1 / normal_fill_time
  let leak_rate := normal_rate - (1 / leak_fill_time)
  1 / leak_rate

/-- Theorem stating that for a cistern with the given properties,
    the time it takes for the leak to empty a full cistern is 40 hours -/
theorem cistern_leak_empty_time :
  leak_empty_time 8 10 = 40 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_empty_time_l535_53591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_sufficient_not_necessary_l535_53516

-- Define the log function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the statement to be proven
theorem log_inequality_sufficient_not_necessary :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ log x > log y ∧ (10 : ℝ)^x > (10 : ℝ)^y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (10 : ℝ)^x > (10 : ℝ)^y ∧ ¬(log x > log y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_sufficient_not_necessary_l535_53516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_with_sine_ratio_l535_53546

theorem triangle_angle_with_sine_ratio (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ Real.sin A = 2 * k ∧ Real.sin B = 3 * k ∧ Real.sin C = 4 * k) :
  B = Real.arccos (11/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_with_sine_ratio_l535_53546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_l535_53574

theorem round_table_seating (n : ℕ) (h : n > 0) : 
  (14 - 31 + n) % n = 17 ∧ (n - 31 + 7) % n = 17 → n = 41 :=
by
  intro h_distances
  -- The proof would go here
  sorry  -- We use 'sorry' as a placeholder for the actual proof

#eval (14 - 31 + 41) % 41  -- Should output 17
#eval (41 - 31 + 7) % 41   -- Should output 17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_l535_53574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_invalid_coloring_l535_53539

/-- A coloring function that assigns one of three colors to each number from 1 to n -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Predicate to check if a coloring is valid (no two numbers of the same color differ by a perfect square) -/
def IsValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ i j : Fin n, i ≠ j → c i = c j → ¬IsPerfectSquare (Int.natAbs (i.val - j.val))

/-- The main theorem: 29 is the smallest number that cannot be validly colored with 3 colors -/
theorem smallest_invalid_coloring :
  (∀ n < 29, ∃ c : Coloring n, IsValidColoring n c) ∧
  (∀ c : Coloring 29, ¬IsValidColoring 29 c) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_invalid_coloring_l535_53539
