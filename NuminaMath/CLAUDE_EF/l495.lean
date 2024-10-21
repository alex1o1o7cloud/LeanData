import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l495_49556

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

-- Define the period
def period (f : ℝ → ℝ) : Set ℝ := 
  {p : ℝ | p > 0 ∧ ∀ x, f (x + p) = f x}

-- Define the range of a function over an interval
def range_over (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

-- Statement for the smallest positive period
theorem smallest_positive_period : ∃ p ∈ period f, p = π := by sorry

-- Statement for the range of f in [0, 2π/3]
theorem range_in_interval : range_over f 0 (2 * π / 3) = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l495_49556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l495_49580

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line structure -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem parabola_theorem (C : Parabola) (F D M N : Point) :
  D.x = C.p ∧ D.y = 0 ∧  -- D(p, 0)
  (M.y - D.y) / (M.x - D.x) = 0 ∧  -- MD perpendicular to x-axis
  distance F M = 3 →
  (∃ (A B : Point),
    -- 1. Equation of C
    (∀ (P : Point), P.y^2 = 4 * P.x ↔ P.y^2 = 2 * C.p * P.x) ∧
    -- 2. Equation of AB when α - β is maximum
    (∃ (L : Line),
      L.m = -Real.sqrt 2 ∧
      L.b = 4 ∧
      (∀ (P : Point), P.y = L.m * P.x + L.b ↔ (P = A ∨ P = B)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l495_49580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_l495_49597

-- Define the structure for a point on the graph
structure Point where
  time : ℝ
  distance : ℝ

-- Define the function representing the graph
noncomputable def graph : ℝ → ℝ := sorry

-- Define average speed
noncomputable def averageSpeed (t1 t2 : ℝ) : ℝ :=
  (graph t2 - graph t1) / (t2 - t1)

-- Define the intervals
def interval1 : Prod ℝ ℝ := (0, 2)
def interval2 : Prod ℝ ℝ := (3, 5)
def interval3 : Prod ℝ ℝ := (4, 6)

-- Theorem statement
theorem highest_average_speed :
  averageSpeed interval2.fst interval2.snd > averageSpeed interval1.fst interval1.snd ∧
  averageSpeed interval2.fst interval2.snd > averageSpeed interval3.fst interval3.snd := by
  sorry

#check highest_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_l495_49597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A2B_l495_49570

theorem det_A2B {n : Type*} [Fintype n] [DecidableEq n] (A B : Matrix n n ℝ) 
  (h1 : Matrix.det A = 3) 
  (h2 : Matrix.det B = 8) : 
  Matrix.det (A ^ 2 * B) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A2B_l495_49570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_sixteenth_power_is_identity_l495_49524

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0, 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0, 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]]

theorem B_sixteenth_power_is_identity :
  B ^ 16 = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_sixteenth_power_is_identity_l495_49524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_area_zero_l495_49569

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  angle : ℝ  -- angle with horizontal in radians

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

theorem circle_tangent_line_area_zero 
  (circleA circleB circleC : Circle)
  (m : Line)
  (h1 : circleA.radius = 2)
  (h2 : circleB.radius = 3)
  (h3 : circleC.radius = 4)
  (h4 : m.angle = π/6)  -- 30 degrees in radians
  (h5 : circleB.center.x = 0 ∧ circleB.center.y = 0)  -- B at origin
  (h6 : circleA.center.x = -5 ∧ circleA.center.y = 0)
  (h7 : circleC.center.x = 7 ∧ circleC.center.y = 0)
  : triangleArea circleA.center circleB.center circleC.center = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_area_zero_l495_49569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l495_49517

-- Define property A
def has_property_A (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f (x + 1) - f x < f (y + 1) - f y

-- Problem 1
theorem problem_1 : has_property_A (λ x : ℝ => x^2 + 2) := by sorry

-- Problem 2
theorem problem_2 : ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x > f y) ∧ has_property_A f := by sorry

-- Problem 3
theorem problem_3 (k : ℝ) :
  has_property_A (λ x : ℝ => k * x^2 + x^3) ↔ k ≥ -3/2 := by sorry

-- Problem 4
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * (Real.sin x)^2 + (Real.sin x)^3 - Real.sin x

theorem problem_4 (k : ℝ) :
  (k = 0 → (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ x₃ ∈ Set.Icc 0 Real.pi ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0)) ∧
  (k < 0 → (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ x₁ ≠ x₂ ∧
    g k x₁ = 0 ∧ g k x₂ = 0 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ g k x = 0 → x = x₁ ∨ x = x₂)) ∧
  (k > 0 → (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ x₃ ∈ Set.Icc 0 Real.pi ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g k x₁ = 0 ∧ g k x₂ = 0 ∧ g k x₃ = 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l495_49517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l495_49506

theorem cylinder_volume_change (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (π * (1.5 * r)^2 * (3 * h)) / (π * r^2 * h) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_change_l495_49506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_imply_omega_range_l495_49539

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

def has_exactly_n_zeros (g : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (zeros : Finset ℝ), zeros.card = n ∧ 
    (∀ x ∈ zeros, a < x ∧ x < b ∧ g x = 0) ∧
    (∀ x, a < x ∧ x < b ∧ g x = 0 → x ∈ zeros)

def has_exactly_n_extreme_points (g : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (extrema : Finset ℝ), extrema.card = n ∧ 
    (∀ x ∈ extrema, a < x ∧ x < b ∧ (DifferentiableAt ℝ g x ∧ (deriv g) x = 0)) ∧
    (∀ x, a < x ∧ x < b ∧ (DifferentiableAt ℝ g x ∧ (deriv g) x = 0) → x ∈ extrema)

theorem f_properties_imply_omega_range :
  ∀ ω : ℝ, (has_exactly_n_zeros (f ω) 0 Real.pi 2 ∧ 
            has_exactly_n_extreme_points (f ω) 0 Real.pi 3) 
           ↔ (13/6 < ω ∧ ω ≤ 8/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_imply_omega_range_l495_49539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l495_49589

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (π - α))^2 * Real.cos (2*π - α) * Real.tan (-π + α) / 
  (Real.sin (-π + α) * Real.tan (-α + 3*π)) = Real.sin α * Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l495_49589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l495_49514

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l495_49514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unique_mixture_without_full_info_l495_49547

/-- Represents a salt solution with a given volume and salt concentration -/
structure SaltSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two salt solutions -/
structure Mixture where
  solution1 : SaltSolution
  solution2 : SaltSolution
  amount1 : ℝ
  amount2 : ℝ

/-- Calculates the resulting concentration of a mixture -/
noncomputable def mixtureConcentration (m : Mixture) : ℝ :=
  (m.solution1.concentration * m.amount1 + m.solution2.concentration * m.amount2) / (m.amount1 + m.amount2)

/-- States that it's impossible to determine a unique mixing ratio without knowing both concentrations -/
theorem no_unique_mixture_without_full_info 
  (s1 : SaltSolution) 
  (s2 : SaltSolution) 
  (target_volume : ℝ) 
  (target_concentration : ℝ) 
  (h1 : s1.concentration = 0.4) 
  (h2 : s1.volume = 30) 
  (h3 : s2.volume = 60) 
  (h4 : target_volume = 50) 
  (h5 : target_concentration = 0.5) :
  ¬∃! (m : Mixture), 
    m.solution1 = s1 ∧ 
    m.solution2 = s2 ∧ 
    m.amount1 + m.amount2 = target_volume ∧ 
    mixtureConcentration m = target_concentration := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unique_mixture_without_full_info_l495_49547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l495_49563

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rational numbers instead of real numbers)
  d : ℚ      -- Common difference
  h : d ≠ 0  -- Non-zero common difference

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (seq.a 1 + seq.a n)

theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.a 6 = 2 * seq.a 3) : 
  S seq 17 / seq.a 3 = 51 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l495_49563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l495_49571

/-- The number of people standing in a row -/
def n : ℕ := 5

/-- The total number of arrangements for n people -/
def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements where A is not at the head and B is not at the end -/
def arrangements_with_restrictions (n : ℕ) : ℕ := 
  Nat.factorial (n - 1) + (n - 2) * (n - 2) * Nat.factorial (n - 2)

/-- The number of arrangements where A and B must stand next to each other -/
def arrangements_adjacent (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of arrangements where A and B cannot stand next to each other -/
def arrangements_not_adjacent (n : ℕ) : ℕ := 
  Nat.factorial (n - 2) * (n - 1) * (n - 2)

theorem arrangement_counts :
  total_arrangements n = 120 ∧
  arrangements_with_restrictions n = 78 ∧
  arrangements_adjacent n = 48 ∧
  arrangements_not_adjacent n = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l495_49571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l495_49519

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^(x+1) + 2)

-- State the theorem
theorem odd_function_inequality (k : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 3, f (k * x^2) + f (2*x - 1) > 0) ↔ k < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l495_49519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l495_49564

theorem cosine_third_angle (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : Real.cos A = 3/5) (h3 : Real.cos B = 5/13) : Real.cos C = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l495_49564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_3_mod_7_l495_49557

theorem two_digit_integers_with_remainder_3_mod_7 : 
  (Finset.filter (fun n => 10 ≤ n ∧ n < 100 ∧ n % 7 = 3) (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_with_remainder_3_mod_7_l495_49557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_projection_not_pentagon_l495_49559

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A unit cube in 3D space -/
def UnitCube : Set Point3D :=
  {p : Point3D | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1}

/-- A 2D plane -/
structure Plane where
  -- We don't need to specify the plane's properties for this proof
  dummy : Unit

/-- Projection of a set of points onto a plane -/
noncomputable def project (s : Set Point3D) (p : Plane) : Set (ℝ × ℝ) :=
  sorry

/-- A simple convex pentagon -/
structure ConvexPentagon where
  -- We don't need to specify the pentagon's properties for this proof
  dummy : Unit

/-- Theorem: The projection of a unit cube onto any 2D plane cannot be a simple convex pentagon -/
theorem cube_projection_not_pentagon (p : Plane) :
  ¬ ∃ (pentagon : Set (ℝ × ℝ)), project UnitCube p = pentagon ∧ ∃ (cp : ConvexPentagon), pentagon = {⟨x, y⟩ | sorry} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_projection_not_pentagon_l495_49559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_lines_l495_49510

/-- Given a line l: 2x + y - 11 = 0, prove that if there exists a line ax + y - b = 0 
    perpendicular to l passing through points A(-2, m) and B(m, 4), 
    and the line through A and B is parallel to l, then a = -0.5 and m = -8. -/
theorem perpendicular_parallel_lines 
  (l : Set (ℝ × ℝ)) 
  (h_l : l = {(x, y) | 2 * x + y - 11 = 0})
  (a b m : ℝ) 
  (h_perp : ∃ (line : Set (ℝ × ℝ)), line = {(x, y) | a * x + y - b = 0} ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ l → q ∈ line → (p.1 - q.1) * (p.2 - q.2) = -1))
  (h_points : ∃ (line : Set (ℝ × ℝ)), line = {(x, y) | a * x + y - b = 0} ∧ 
    (-2, m) ∈ line ∧ (m, 4) ∈ line)
  (h_parallel : ∃ (line : Set (ℝ × ℝ)), line = {(x, y) | y - m = (m - 4) / (-2 - m) * (x + 2)} ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ l → q ∈ line → p.2 - q.2 = 2 * (p.1 - q.1))) :
  a = -0.5 ∧ m = -8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_lines_l495_49510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_salary_salary_calculation_l495_49565

/-- The yearly salary for a servant, excluding a turban -/
noncomputable def yearly_salary : ℝ := 200

/-- The fraction of a year the servant worked -/
noncomputable def work_fraction : ℝ := 3 / 4

/-- The cash amount received by the servant when leaving -/
noncomputable def cash_received : ℝ := 40

/-- The value of the turban -/
noncomputable def turban_value : ℝ := 110

theorem servant_salary :
  yearly_salary = 200 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

theorem salary_calculation :
  work_fraction * yearly_salary + turban_value = cash_received + turban_value :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_servant_salary_salary_calculation_l495_49565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_is_negative_one_l495_49546

noncomputable section

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-13, -23)
def B : ℝ × ℝ := (-33, -43)

-- Define the slope of line segment AB
noncomputable def slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the slope of a line perpendicular to AB
noncomputable def slope_perpendicular : ℝ := -1 / slope_AB

-- Theorem statement
theorem perpendicular_slope_is_negative_one :
  slope_perpendicular = -1 := by
  -- Unfold the definitions
  unfold slope_perpendicular slope_AB
  -- Simplify the expression
  simp [A, B]
  -- The proof is completed
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_is_negative_one_l495_49546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_max_k_value_l495_49587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a * x^2) / 2 + a - x

def g (x : ℝ) : ℝ := 2 - 2*x - x^2

def has_two_distinct_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ (∀ z, deriv f z = 0 → z = x ∨ z = y)

theorem extreme_points_condition (a : ℝ) :
  has_two_distinct_extreme_points (f a) ↔ 0 < a ∧ a < 1 / Real.exp 1 :=
sorry

theorem max_k_value :
  ∃ k : ℕ, k = 4 ∧
    ∀ k' : ℕ, (∀ x : ℝ, x > 2 → k' * (x - 2) + g x < f 2 x) →
    k' ≤ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_max_k_value_l495_49587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l495_49585

/-- The length of a train given its speed and time to cross a pole -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 3600

/-- Theorem: A train running at 90 km/hr that crosses a pole in 9 seconds has a length of 225 meters -/
theorem train_length_calculation :
  train_length 90 9 = 225 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l495_49585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l495_49558

theorem remainder_sum_mod_15 (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_15_l495_49558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_max_value_l495_49545

theorem cos_sin_max_value (α : ℝ) : 
  Real.cos α ^ 2 + Real.sin α + 3 ≤ 17 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_max_value_l495_49545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_probability_l495_49581

/-- Represents a cube with 8 vertices and 12 edges. -/
structure Cube where
  vertices : Fin 8
  edges : Fin 12

/-- Represents a rabbit's path on the cube. -/
def RabbitPath := List (Fin 8)

/-- The probability of choosing any edge at a vertex. -/
def edge_probability : ℚ := 1 / 3

/-- The number of moves the rabbit makes. -/
def num_moves : ℕ := 11

/-- Checks if a path visits all vertices exactly once and doesn't end at the start. -/
def is_valid_path (path : RabbitPath) : Prop :=
  path.length = num_moves + 1 ∧
  path.toFinset.card = 8 ∧
  path.head? ≠ path.getLast?

/-- The probability of a valid path. -/
def valid_path_probability : ℚ := 24 / 177147

theorem rabbit_path_probability (cube : Cube) :
  ∃ (valid_paths : Finset RabbitPath),
    (∀ path ∈ valid_paths, is_valid_path path) ∧
    (valid_paths.card : ℚ) / (3 ^ num_moves) = valid_path_probability := by
  sorry

#check rabbit_path_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_probability_l495_49581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_7_4_l495_49528

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n+2) => sequence_a (n+1) + 1 / ((n+2) * (n+1))

theorem a_4_equals_7_4 : sequence_a 4 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_7_4_l495_49528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l495_49534

/-- A parabola is defined by the equation x^2 = y -/
def is_parabola (f : ℝ → ℝ) : Prop := ∀ x, f x = x^2

/-- The distance from the focus to the directrix of a parabola -/
noncomputable def focus_directrix_distance (f : ℝ → ℝ) : ℝ := 1/2

/-- Theorem: For a parabola defined by x^2 = y, the distance from the focus to the directrix is 1/2 -/
theorem parabola_focus_directrix_distance (f : ℝ → ℝ) (h : is_parabola f) : 
  focus_directrix_distance f = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l495_49534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_equilateral_or_isosceles_right_l495_49576

/-- A triangle with externally constructed squares -/
structure TriangleWithSquares where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ

/-- Predicate to check if a set of points is concyclic -/
def IsConcyclic (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (A B C : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if a triangle is isosceles right -/
def IsIsoscelesRight (A B C : ℝ × ℝ) : Prop :=
  sorry

/-- Main theorem -/
theorem concyclic_iff_equilateral_or_isosceles_right (t : TriangleWithSquares) :
  IsConcyclic [t.D, t.E, t.F, t.G, t.H, t.I] ↔
  IsEquilateral t.A t.B t.C ∨ IsIsoscelesRight t.A t.B t.C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_equilateral_or_isosceles_right_l495_49576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l495_49561

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = 3x + 7 -/
noncomputable def m1 : ℝ := 3

/-- The slope of the second line 4y + kx = 4 in terms of k -/
noncomputable def m2 (k : ℝ) : ℝ := -k / 4

/-- Theorem: If two lines with equations y = 3x + 7 and 4y + kx = 4 are perpendicular, then k = 4/3 -/
theorem perpendicular_lines_k (k : ℝ) : perpendicular m1 (m2 k) → k = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l495_49561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l495_49596

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := 3 * n - 2

noncomputable def S (n : ℕ) : ℝ := (n * (arithmetic_sequence 1 + arithmetic_sequence n)) / 2

noncomputable def b (n : ℕ) : ℝ := 3 / (2 * S n + 4 * n)

noncomputable def T (n : ℕ) : ℝ := n / (n + 1)

theorem arithmetic_sequence_proof :
  (S 7 = 70) ∧
  (arithmetic_sequence 2)^2 = (arithmetic_sequence 1) * (arithmetic_sequence 6) →
  (∀ n : ℕ, arithmetic_sequence n = 3 * n - 2) ∧
  (∀ n : ℕ, T n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l495_49596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l495_49515

open Real
open Set

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

theorem x0_range :
  ∀ x0 : ℝ,
  (∃ P : ℝ × ℝ, 
    on_circle P.1 P.2 ∧ 
    angle (P.1 - x0, P.2 - 1) (2 + x0, 0) = π/3) →
  x0 ∈ Icc (-Real.sqrt 3/3) (Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l495_49515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_sum_l495_49500

def f (y : ℤ) : ℤ := y^5 - 2*y^3 - y^2 - 2

def is_monic_irreducible_factor (q : ℤ → ℤ) : Prop :=
  (∃ n : ℕ, n > 0 ∧ (∀ y : ℤ, ∃ k : ℤ, q y = y^n + k)) ∧
  (∀ g h : ℤ → ℤ, (∀ x : ℤ, q x = g x * h x) → g = (λ _ => 1) ∨ h = (λ _ => 1))

theorem factorization_sum (q₁ q₂ q₃ : ℤ → ℤ) :
  (∀ y : ℤ, f y = q₁ y * q₂ y * q₃ y) →
  is_monic_irreducible_factor q₁ →
  is_monic_irreducible_factor q₂ →
  is_monic_irreducible_factor q₃ →
  q₁ 3 + q₂ 3 + q₃ 3 = 18 := by
  sorry

#check factorization_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_sum_l495_49500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_set_characterization_l495_49548

def is_valid_set (S : Set ℝ) : Prop :=
  (1 ∈ S) ∧
  (∀ x y, x ∈ S → y ∈ S → x > y → Real.sqrt (x^2 - y^2) ∈ S)

def is_infinite_sqrt_set (S : Set ℝ) : Prop :=
  S = {x : ℝ | ∃ k : ℕ+, x = Real.sqrt (k : ℝ)}

def is_finite_sqrt_set (S : Set ℝ) (n : ℕ+) : Prop :=
  S = {x : ℝ | ∃ k : ℕ+, k ≤ n ∧ x = Real.sqrt (k : ℝ)}

theorem sqrt_set_characterization (S : Set ℝ) :
  is_valid_set S →
  (is_infinite_sqrt_set S ∨ ∃ n : ℕ+, is_finite_sqrt_set S n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_set_characterization_l495_49548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_passing_time_l495_49523

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

/-- Calculates the time (in seconds) for an object to travel a given distance at a given speed -/
noncomputable def time_to_pass (length : ℝ) (speed_kmph : ℝ) : ℝ :=
  length / kmph_to_mps speed_kmph

/-- Theorem: A car 10 meters long, moving at 36 km/h, takes 1 second to pass a stationary point -/
theorem car_passing_time :
  time_to_pass 10 36 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_passing_time_l495_49523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_two_l495_49542

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
noncomputable def GeometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a₁ * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio_two
  (a₁ : ℝ) (q : ℝ) (h_q_pos : q > 0)
  (h_S2 : GeometricSum a₁ q 2 = 3 * (GeometricSequence a₁ q 2) + 2)
  (h_S4 : GeometricSum a₁ q 4 = 3 * (GeometricSequence a₁ q 4) + 2) :
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_two_l495_49542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l495_49567

-- Define the expression
def expression : ℚ := (2 / 3) ^ 10 * (5 / 2) ^ (-4 : ℤ)

-- State the theorem
theorem expression_evaluation : expression = 16384 / 36880625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l495_49567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_candies_theorem_l495_49543

/-- Represents the number of students in the class -/
def num_students : ℕ := 25

/-- Represents the minimum number of candies that can be distributed -/
def min_candies : ℕ := 600

/-- 
Given a number of candies, checks if it's possible to distribute them
among the students according to the problem constraints
-/
def is_valid_distribution (n : ℕ) : Prop :=
  ∀ (performance : Fin num_students → ℕ),
    ∃ (distribution : Fin num_students → ℕ),
      (∀ i, distribution i ≤ n) ∧
      (∀ i j, performance i = performance j → distribution i = distribution j) ∧
      (∀ i j, performance i < performance j → distribution i < distribution j) ∧
      (Finset.sum Finset.univ distribution) = n

/--
Theorem stating that min_candies is the smallest number of candies
that satisfies the distribution requirements
-/
theorem min_candies_theorem : 
  is_valid_distribution min_candies ∧ 
  ∀ m : ℕ, m < min_candies → ¬is_valid_distribution m := by
  sorry

#check min_candies_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_candies_theorem_l495_49543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_approx_l495_49505

/-- Represents a triangle with sides a, b, and c, where an altitude h is dropped on side c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ

/-- Calculates the length of the larger segment cut off on side c when an altitude is dropped --/
noncomputable def largerSegment (t : Triangle) : ℝ :=
  let x := (t.c^2 + t.a^2 - t.b^2) / (2 * t.c)
  t.c - x

theorem triangle_larger_segment_approx (t : Triangle) 
  (h1 : t.a = 20)
  (h2 : t.b = 48)
  (h3 : t.c = 52) :
  abs (largerSegment t - 44.31) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_larger_segment_approx_l495_49505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l495_49568

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := 
  (tan (π - α) * cos (2*π - α) * sin (π/2 + α)) / cos (π + α)

-- State the theorem
theorem tan_alpha_value (α : ℝ) 
  (h1 : f (π/2 - α) = -3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  tan α = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l495_49568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equal_angles_not_congruent_l495_49521

/-- Definition: A triangle with three angles and three sides -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Definition: A triangle is right-angled -/
def RightAngled (T : Triangle) : Prop :=
  T.angle3 = Real.pi / 2

/-- Definition: An angle is acute -/
def Acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

/-- Definition: Two triangles are congruent -/
def Congruent (T1 T2 : Triangle) : Prop :=
  T1.side1 = T2.side1 ∧ T1.side2 = T2.side2 ∧ T1.side3 = T2.side3

/-- Two right-angled triangles with equal acute angles are not necessarily congruent -/
theorem right_triangle_equal_angles_not_congruent :
  ∃ (T1 T2 : Triangle) (α β : ℝ),
    RightAngled T1 ∧ RightAngled T2 ∧
    Acute α ∧ Acute β ∧
    T1.angle1 = α ∧ T1.angle2 = β ∧
    T2.angle1 = α ∧ T2.angle2 = β ∧
    ¬ Congruent T1 T2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equal_angles_not_congruent_l495_49521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_cubed_plus_one_l495_49555

-- Define the function to be integrated
def f (x : ℝ) : ℝ := x^3 + 1

-- State the theorem
theorem integral_x_cubed_plus_one : ∫ x in (0 : ℝ)..(2 : ℝ), f x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_cubed_plus_one_l495_49555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_sin_function_l495_49578

theorem max_value_cos_sin_function :
  ∀ x : ℝ, (Real.cos x) ^ 3 + (Real.sin x) ^ 2 - Real.cos x ≤ 32 / 27 ∧
  ∃ x : ℝ, (Real.cos x) ^ 3 + (Real.sin x) ^ 2 - Real.cos x = 32 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_sin_function_l495_49578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l495_49582

theorem order_of_abc : ∀ (a b c : ℝ),
  a = 3/4 →
  b = Real.sqrt (Real.exp 1) - 1 →
  c = Real.log (3/2) →
  c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l495_49582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simpsons_rule_accuracy_l495_49533

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Define Simpson's rule
noncomputable def simpsons_rule (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : ℝ :=
  let h := (b - a) / (2 * n : ℝ)
  let x_i (i : ℕ) := a + i * h
  (h / 3) * (f a + f b + 
    4 * (Finset.sum (Finset.range n) (λ i => f (x_i (2 * i + 1)))) +
    2 * (Finset.sum (Finset.range (n - 1)) (λ i => f (x_i (2 * (i + 1))))))

-- State the theorem
theorem simpsons_rule_accuracy :
  let a := 0
  let b := Real.pi / 2
  let n := 44
  let true_value := 1
  let approximation := simpsons_rule f a b n
  |true_value - approximation| < 0.00001 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simpsons_rule_accuracy_l495_49533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l495_49583

/-- The length of the real axis of a hyperbola -/
noncomputable def realAxisLength (a : ℝ) : ℝ := 2 * a

/-- Slope of the asymptote of the hyperbola -/
noncomputable def asymptoticSlope (a : ℝ) : ℝ := a / 3

theorem hyperbola_real_axis_length 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : asymptoticSlope a * (1/3) = -1) : 
  realAxisLength a = 18 := by
  -- Unfold the definitions
  unfold realAxisLength asymptoticSlope at *
  -- Simplify the equation in h2
  have h3 : a / 3 * (1/3) = -1 := h2
  -- Solve for a
  have h4 : a = 9 := by
    -- Here we would normally solve the equation
    -- For now, we'll use sorry
    sorry
  -- Substitute a = 9 into the goal
  rw [h4]
  -- Evaluate 2 * 9
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l495_49583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l495_49560

theorem zeros_before_first_nonzero_digit (n d : ℕ) (h : d > 0) :
  let f := n / d
  let decimal_representation := toString f
  let zeros_count := (decimal_representation.drop 2).takeWhile (· == '0') |>.length
  7000 ≤ n ∧ n < 8000 ∧ d = 8000 → zeros_count = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l495_49560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_slopes_constant_l495_49586

/-- Parabola C₁: y² = 4x -/
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of C₁ -/
def F : ℝ × ℝ := (1, 0)

/-- Directrix of C₁ -/
def l (x : ℝ) : Prop := x = -1

/-- Point A on the directrix -/
def A : ℝ × ℝ := (-1, 0)

/-- Line n passing through A -/
def n (k y : ℝ) : ℝ := k*y - 1

/-- Slope of a line passing through two points -/
noncomputable def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

/-- Theorem: Sum of slopes of MF and NF is constant -/
theorem sum_slopes_constant (k : ℝ) :
  ∃ (c : ℝ), ∀ (y₁ y₂ : ℝ),
    C₁ (n k y₁) y₁ → C₁ (n k y₂) y₂ → y₁ ≠ y₂ →
    line_slope (n k y₁) y₁ F.1 F.2 + line_slope (n k y₂) y₂ F.1 F.2 = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_slopes_constant_l495_49586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotone_intervals_l495_49579

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

theorem tan_monotone_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo (k * Real.pi / 2 - 3 * Real.pi / 8) (k * Real.pi / 2 + Real.pi / 8)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_monotone_intervals_l495_49579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_number_theorem_l495_49501

-- Define the left-hand side of the equation
noncomputable def pentagonalSum (z : ℂ) : ℂ :=
  ∑' n : ℤ, (-1 : ℂ)^n * z^((n * (3 * n - 1)) / 2)

-- Define the right-hand side of the equation
noncomputable def pentagonalProduct (z : ℂ) : ℂ :=
  ∏' n : ℕ, (1 - z^n)

-- State the pentagonal number theorem
theorem pentagonal_number_theorem (z : ℂ) (h : Complex.abs z < 1) :
  pentagonalSum z = pentagonalProduct z := by
  sorry

#check pentagonal_number_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_number_theorem_l495_49501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l495_49594

/-- The time for two trains to cross each other -/
noncomputable def time_to_cross (train_length : ℝ) (time1 time2 : ℝ) : ℝ :=
  (2 * train_length) / (train_length / time1 + train_length / time2)

/-- Theorem: Two trains of length 120 meters, taking 10 and 15 seconds respectively to cross a 
    telegraph post, will cross each other in 12 seconds when traveling in opposite directions -/
theorem trains_crossing_time :
  time_to_cross 120 10 15 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l495_49594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l495_49522

theorem boys_to_girls_ratio (total_students : ℕ) (boys girls : ℝ) : 
  (boys + girls = total_students) →
  (2 * boys = total_students) →
  (0.7 * boys = boys - girls / 2) →
  (0.4 * girls = girls - boys / 2) →
  boys = 2 * girls :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l495_49522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l495_49504

-- Define the function f with domain (1,3)
def f : Set ℝ := Set.Ioo 1 3

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  if x ∈ Set.Ioo 1 2 then Real.sqrt (x - 1)⁻¹ else 0

-- Theorem statement
theorem domain_of_g :
  Set.Ioo 1 2 = {x : ℝ | ∃ y, g x = y ∧ y ≠ 0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l495_49504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l495_49529

def factors_of_90 : Finset Nat := (Finset.range 91).filter (fun n => n > 0 ∧ 90 % n = 0)

def factors_less_than_10 : Finset Nat := factors_of_90.filter (· < 10)

theorem probability_factor_less_than_10 : 
  (factors_less_than_10.card : ℚ) / (factors_of_90.card : ℚ) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l495_49529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l495_49575

/-- A point in a unit square -/
structure UnitSquarePoint where
  x : Real
  y : Real
  x_bound : 0 ≤ x ∧ x ≤ 1
  y_bound : 0 ≤ y ∧ y ≤ 1

/-- A path in a unit square -/
structure UnitSquarePath where
  points : List UnitSquarePoint
  length : Real

theorem broken_line_theorem (n : Nat) (h : n > 0) (points : Finset UnitSquarePoint) 
    (point_count : points.card = n^2) :
  ∃ (p : UnitSquarePath), p.length ≤ 2*n + 1 ∧ ∀ pt ∈ points, pt ∈ p.points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l495_49575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_investment_l495_49588

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℝ
  profitShare : ℝ

/-- Proves that given the conditions of the problem, c's investment is 4000 -/
theorem c_investment (a b c : Partner) : 
  b.profitShare = 4000 →
  a.profitShare - c.profitShare = 1599.9999999999995 →
  a.investment = 8000 →
  b.investment = 10000 →
  a.profitShare / a.investment = b.profitShare / b.investment →
  a.profitShare / a.investment = c.profitShare / c.investment →
  b.profitShare / b.investment = c.profitShare / c.investment →
  c.investment = 4000 := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

#check c_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_investment_l495_49588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l495_49584

/-- The minimum distance between a point on the circle x² + (y-1)² = 2 and a point on the line x + y = 5 is √2 -/
theorem min_distance_circle_line : ∃ (d : ℝ),
  d = Real.sqrt 2 ∧
  ∀ (P Q : ℝ × ℝ),
    (P.1^2 + (P.2 - 1)^2 = 2) →
    (Q.1 + Q.2 = 5) →
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l495_49584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circles_radius_l495_49566

theorem small_circles_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- Radius of the large circle
  r > 0 → -- Radius of small circles is positive
  4 * r = R * Real.sqrt 2 → -- Relation between radii (derived from the square arrangement)
  r = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circles_radius_l495_49566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_ratio_l495_49507

/-- Represents a pentagon in a plane -/
structure Pentagon where
  points : Fin 5 → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j

/-- Calculates the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Checks if two line segments are parallel -/
def parallel (p1 q1 p2 q2 : ℝ × ℝ) : Prop :=
  (q1.2 - p1.2) * (q2.1 - p2.1) = (q2.2 - p2.2) * (q1.1 - p1.1)

/-- Theorem about the ratio of segment lengths in a specific pentagon configuration -/
theorem pentagon_ratio (P : Pentagon) (a b c d : ℝ) : 
  (∃ i j k l m, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ i ∧
    distance (P.points i) (P.points j) = a ∧
    distance (P.points j) (P.points k) = b ∧
    distance (P.points k) (P.points l) = a ∧
    distance (P.points l) (P.points m) = b ∧
    distance (P.points m) (P.points i) = d) →
  (∃ p q r s, p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ p ∧
    distance (P.points p) (P.points q) = a ∧
    distance (P.points q) (P.points r) = a ∧
    distance (P.points r) (P.points s) = a ∧
    distance (P.points s) (P.points p) = a) →
  (∃ x y, x ≠ y ∧ distance (P.points x) (P.points y) = c) →
  c = 2 * a →
  (∃ u v w z, u ≠ v ∧ v ≠ w ∧ w ≠ z ∧
    distance (P.points u) (P.points v) = b ∧
    distance (P.points w) (P.points z) = b ∧
    parallel (P.points u) (P.points v) (P.points w) (P.points z)) →
  d / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_ratio_l495_49507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l495_49591

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point A
def pointA : ℝ × ℝ := (3, 1)

-- Define a point on the parabola
def pointOnParabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (answer : ℝ), ∀ (P : ℝ × ℝ), pointOnParabola P →
    distance P focus + distance P pointA ≥ answer ∧
    ∃ (P₀ : ℝ × ℝ), pointOnParabola P₀ ∧
      distance P₀ focus + distance P₀ pointA = answer :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l495_49591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_eq_three_fifths_l495_49531

/-- The sequence b defined recursively -/
def b : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 1 + 2 * (b n - 1)^3

/-- The infinite product of the sequence b -/
noncomputable def infinite_product : ℝ := ∏' n, (b n : ℝ)

/-- Theorem stating that the infinite product of the sequence b is equal to 3/5 -/
theorem infinite_product_eq_three_fifths : infinite_product = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_eq_three_fifths_l495_49531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l495_49552

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * sin x * (sin (π / 4 + x / 2))^2 + cos (2 * x)

-- Define the theorem
theorem f_range_theorem (m : ℝ) :
  (∀ x, π / 6 ≤ x ∧ x ≤ 2 * π / 3 → |f x - m| < 2) ↔ 0 < m ∧ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l495_49552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l495_49509

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sin (ω * x - Real.pi / 2)

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, f ω x = f ω (Real.pi / 4 - x)) →
  (∃ x : ℝ, x > -Real.pi / 4 ∧ x < 0 ∧ f ω x = 0) →
  ω ≥ 10 ∧ ∀ ω' : ℝ, ω' ≥ 10 → ω' ≥ ω :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l495_49509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_fixed_points_l495_49536

/-- The function f(x) = 1 + 2/x -/
noncomputable def f (x : ℝ) : ℝ := 1 + 2/x

/-- The nth iteration of f -/
noncomputable def f_n : ℕ → ℝ → ℝ 
  | 0, x => x
  | n+1, x => f (f_n n x)

/-- Theorem: For any positive integer n, the solutions to x = f_n(x) are -1 and 2 -/
theorem f_n_fixed_points (n : ℕ) (h : n > 0) :
  ∀ x : ℝ, x = f_n n x ↔ x = -1 ∨ x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_fixed_points_l495_49536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_real_roots_l495_49577

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - 1

theorem f_has_three_real_roots :
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
    ∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_real_roots_l495_49577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_simultaneous_photo_capture_l495_49595

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℚ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the photographer's setup -/
structure Photographer where
  position : ℚ  -- fraction of track from start line
  coverage : ℚ  -- fraction of track covered by camera

/-- Calculates the position of a runner at a given time -/
noncomputable def runnerPosition (r : Runner) (t : ℚ) : ℚ :=
  if r.direction then
    (t / r.lapTime) % 1
  else
    (1 - (t / r.lapTime) % 1) % 1

/-- Checks if a runner is within the photographer's coverage at a given time -/
noncomputable def inPhoto (r : Runner) (p : Photographer) (t : ℚ) : Prop :=
  let runnerPos := runnerPosition r t
  let lowerBound := (p.position - p.coverage / 2 + 1) % 1
  let upperBound := (p.position + p.coverage / 2) % 1
  if lowerBound < upperBound then
    lowerBound ≤ runnerPos ∧ runnerPos ≤ upperBound
  else
    runnerPos ≤ upperBound ∨ lowerBound ≤ runnerPos

theorem no_simultaneous_photo_capture 
  (andrew : Runner) 
  (bella : Runner) 
  (photographer : Photographer) : 
  andrew.lapTime = 75 ∧ 
  andrew.direction = true ∧ 
  bella.lapTime = 120 ∧ 
  bella.direction = false ∧ 
  photographer.position = 1/3 ∧ 
  photographer.coverage = 1/4 → 
  ∀ t : ℚ, 540 ≤ t ∧ t < 600 → 
  ¬(inPhoto andrew photographer t ∧ inPhoto bella photographer t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_simultaneous_photo_capture_l495_49595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l495_49526

/-- Represents the amount of money in each box -/
def box_values : Finset ℕ := {8, 800, 8000, 40000, 80000}

/-- The number of keys and boxes -/
def num_keys_boxes : ℕ := 5

/-- The threshold for winning -/
def winning_threshold : ℕ := 60000

/-- The probability of winning more than the threshold -/
def win_probability : ℚ := 1 / 4

theorem game_show_probability :
  (Finset.filter (fun x => x > winning_threshold) box_values).card * (num_keys_boxes - 1).factorial /
  (num_keys_boxes.factorial : ℚ) = win_probability :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l495_49526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l495_49599

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / a + a / 2^x - 1

-- Main theorem
theorem main_theorem (a : ℝ) (h_a : a > 0) 
  (h_even : ∀ x, f a x = f a (-x)) :
  (a = 1) ∧ 
  (∀ x, f 1 x < 13/4 ↔ -2 < x ∧ x < 2) ∧
  (∀ m, (∀ x, x > 0 → m * f 1 x ≥ 2^(-x) - m) ↔ m ≥ 1/2) := by
  sorry

#check main_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l495_49599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_up_is_two_days_l495_49549

/-- Represents the hiking scenario with given conditions -/
structure HikingTrip where
  rate_up : ℚ
  rate_down : ℚ
  distance_down : ℚ
  h_rate_up : rate_up = 8
  h_rate_down : rate_down = 3/2 * rate_up
  h_distance_down : distance_down = 24

/-- Calculates the time taken for a part of the trip given distance and rate -/
def time (distance rate : ℚ) : ℚ := distance / rate

/-- Theorem stating that the time to go up the mountain is 2 days -/
theorem time_up_is_two_days (trip : HikingTrip) : 
  time (time trip.distance_down trip.rate_down * trip.rate_up) trip.rate_up = 2 := by
  sorry

#check time_up_is_two_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_up_is_two_days_l495_49549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_scaling_product_l495_49598

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

def scaling_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0],
    ![0, s]]

theorem det_rotation_scaling_product :
  let R := rotation_matrix (π / 4)
  let S := scaling_matrix 3
  Matrix.det (R * S) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_scaling_product_l495_49598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_square_l495_49592

theorem area_ratio_triangle_square :
  let triangle_side : ℝ := 6
  let square_side : ℝ := 6
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let square_area := square_side^2
  (triangle_area / square_area : ℝ) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_square_l495_49592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_count_after_loss_l495_49574

/-- Calculates the number of remaining balloons after a loss --/
def remaining_balloons (initial : ℕ) (loss_percentage : ℚ) : ℕ :=
  (initial : ℚ) - (loss_percentage * initial) |> Int.floor |> Int.toNat

theorem balloon_count_after_loss :
  let orange_initial := 8
  let blue_initial := 10
  let purple_initial := 6
  let orange_loss := 25 / 100
  let blue_loss := 1 / 5
  let purple_loss := 333333 / 1000000  -- Approximation of 33.33%
  (remaining_balloons orange_initial orange_loss = 6) ∧
  (remaining_balloons blue_initial blue_loss = 8) ∧
  (remaining_balloons purple_initial purple_loss = 4) :=
by
  -- Proof goes here
  sorry

#eval remaining_balloons 8 (25 / 100)
#eval remaining_balloons 10 (1 / 5)
#eval remaining_balloons 6 (333333 / 1000000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_count_after_loss_l495_49574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l495_49593

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 2) => sequence_a (n + 1) + 2 * (n + 1)

theorem a_100_value : sequence_a 100 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l495_49593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_travel_time_difference_l495_49512

/-- Represents a ferry with its speed and route length -/
structure Ferry where
  speed : ℝ
  route_length : ℝ

/-- Calculates the travel time of a ferry -/
noncomputable def travel_time (f : Ferry) : ℝ := f.route_length / f.speed

theorem ferry_travel_time_difference (p q : Ferry)
  (h1 : p.speed = 8)
  (h2 : p.route_length = 16)
  (h3 : q.route_length = 3 * p.route_length)
  (h4 : q.speed = p.speed + 4) :
  travel_time q - travel_time p = 2 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_travel_time_difference_l495_49512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l495_49535

theorem ticket_price_possibilities : 
  {x : ℕ | ∃ (a b : ℕ), 90 = a * x ∧ 150 = b * x} = {1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l495_49535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_m_range_l495_49532

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x - m * x

-- State the theorem
theorem f_inequality_implies_m_range (m : ℝ) :
  (m > 0) →
  (∀ x ≥ 1, f m x ≤ (m - 1) / x - 2 * m + 1) →
  m ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_m_range_l495_49532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_and_perimeter_ratio_l495_49553

-- Define the two right triangles
def triangle1 : ℝ × ℝ := (12, 9)
def triangle2 (y : ℝ) : ℝ × ℝ := (y, 6)

-- Define similarity of triangles
def similar (t1 t2 : ℝ × ℝ) : Prop :=
  t1.1 / t2.1 = t1.2 / t2.2

-- Define the perimeter of a right triangle
noncomputable def perimeter (t : ℝ × ℝ) : ℝ :=
  t.1 + t.2 + Real.sqrt (t.1^2 + t.2^2)

-- Theorem statement
theorem triangle_similarity_and_perimeter_ratio :
  ∃ y : ℝ,
    similar triangle1 (triangle2 y) ∧
    y = 8 ∧
    perimeter triangle1 / perimeter (triangle2 y) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_and_perimeter_ratio_l495_49553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imo_1999_shortlist_40_l495_49513

theorem imo_1999_shortlist_40 (N : ℕ) (A : Finset ℕ) :
  A.card = N →
  A ⊆ Finset.range (N^2) →
  ∃ B : Finset ℕ,
    B.card = N ∧
    B ⊆ Finset.range (N^2) ∧
    (Finset.card (Finset.image (λ p : ℕ × ℕ ↦ (p.1 + p.2) % (N^2))
      (A.product B))) ≥ N^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imo_1999_shortlist_40_l495_49513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l495_49508

noncomputable def f (x : ℝ) : ℝ := |Real.sin x| / Real.sin x + Real.cos x / |Real.cos x| + |Real.tan x| / Real.tan x

theorem f_range : Set.range f = {-1, 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l495_49508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l495_49540

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem min_roots_in_interval
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : is_periodic f 5)
  (h_values : f (-1) = -1 ∧ f 2 = -1) :
  ∃ (S : Set ℝ), S.Finite ∧ S ⊆ Set.Icc 1755 2017 ∧ S.ncard < 210 ∧ ∀ x ∈ S, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_in_interval_l495_49540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l495_49541

theorem min_value_abs_sum : 
  ∀ x : ℝ, |x - 4| + |x - 6| ≥ 2 ∧ ∃ x : ℝ, |x - 4| + |x - 6| = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abs_sum_l495_49541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_distinct_sums_l495_49590

/-- The number of distinct pairwise sums in a sequence -/
def L (A : Finset ℚ) : ℕ :=
  (Finset.filter (fun p => p.1 < p.2) (A.product A)).image (fun p => p.1 + p.2) |>.card

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSeq (a d : ℚ) (n : ℕ) : Finset ℚ :=
  Finset.image (fun k => a + d * ↑k) (Finset.range n)

theorem arithmetic_seq_distinct_sums (a d : ℚ) (h : d ≠ 0) :
  L (arithmeticSeq a d 2016) = 4029 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_distinct_sums_l495_49590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_zero_l495_49525

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that af(a) + bf(b) = 0 when ab = 1
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a * b = 1 → a * f a + b * f b = 0

-- State the theorem
theorem integral_zero (f : ℝ → ℝ) (hf : MeasureTheory.Integrable f MeasureTheory.volume) (hp : property f) :
  ∫ x in Set.Ici 0, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_zero_l495_49525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_proof_l495_49554

noncomputable def regular_quadrilateral_pyramid_volume (l : ℝ) : ℝ :=
  (l^3 * Real.sqrt 3) / 12

theorem regular_quadrilateral_pyramid_volume_proof (l : ℝ) (h : l > 0) :
  let lateral_edge := l
  let inclination_angle := 60 * Real.pi / 180
  regular_quadrilateral_pyramid_volume lateral_edge = (l^3 * Real.sqrt 3) / 12 :=
by
  sorry

#check regular_quadrilateral_pyramid_volume_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_proof_l495_49554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_cans_theorem_l495_49527

/-- Calculates the number of soda cans bought given the prices and quantities of movie tickets and popcorn --/
def soda_cans_bought (ticket_price : ℚ) (popcorn_price_ratio : ℚ) (soda_price_ratio : ℚ) 
  (num_tickets : ℕ) (num_popcorn : ℕ) (total_spent : ℚ) : ℕ :=
  let popcorn_price := ticket_price * popcorn_price_ratio
  let soda_price := popcorn_price * soda_price_ratio
  let tickets_cost := ticket_price * num_tickets
  let popcorn_cost := popcorn_price * num_popcorn
  let remaining_for_soda := total_spent - tickets_cost - popcorn_cost
  (remaining_for_soda / soda_price).floor.toNat

/-- Theorem stating that given the specific conditions, the number of soda cans bought is 4 --/
theorem soda_cans_theorem : 
  soda_cans_bought 5 (4/5) (1/2) 4 2 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_cans_theorem_l495_49527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l495_49537

noncomputable def a : ℝ × ℝ := (3, 2)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def c : ℝ × ℝ := (4, 1)

noncomputable def m : ℝ := 5/9
noncomputable def n : ℝ := 8/9

theorem vector_equation_solution :
  a = (m * b.1 + n * c.1, m * b.2 + n * c.2) :=
by
  -- Unfold the definitions
  unfold a b c m n
  -- Perform the calculations
  simp [mul_add, add_mul]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l495_49537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_satisfies_conditions_l495_49550

noncomputable section

def α_set : Set ℚ := {-2, -1, -(1/2), 1/2, 1, 2}

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f y < f x

noncomputable def f (α : ℚ) (x : ℝ) : ℝ := x ^ (α : ℝ)

theorem unique_alpha_satisfies_conditions :
  ∃! α : ℚ, α ∈ α_set ∧ is_odd (f α) ∧ is_decreasing (f α) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_satisfies_conditions_l495_49550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l495_49562

/-- The slope angle of a line with equation ax + by + c = 0 -/
noncomputable def slope_angle (a b : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- Converts radians to degrees -/
noncomputable def rad_to_deg (θ : ℝ) : ℝ :=
  θ * 180 / Real.pi

theorem line_slope_angle :
  rad_to_deg (slope_angle 1 (Real.sqrt 3)) = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l495_49562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_relation_l495_49520

-- Define the function g as noncomputable
noncomputable def g (t : ℝ) : ℝ := 2 * t / (1 - t)

-- State the theorem
theorem inverse_function_relation (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ -2) :
  y = g x → x = -g (-y) := by
  -- Introduce the hypothesis
  intro h
  -- Expand the definition of g
  unfold g at h ⊢
  -- Algebraic manipulation
  field_simp at h ⊢
  -- The rest of the proof is omitted for brevity
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_relation_l495_49520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_vertex_C_y_coordinate_l495_49538

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The area of a square given its side length -/
noncomputable def squareArea (side : ℝ) : ℝ := side * side

/-- Theorem: The y-coordinate of vertex C in the given pentagon is 40/3 -/
theorem pentagon_vertex_C_y_coordinate (p : Pentagon) : 
  p.A = (0, 0) →
  p.B = (0, 5) →
  p.C.1 = 3 →
  p.D = (6, 5) →
  p.E = (6, 0) →
  triangleArea 6 (p.C.2 - 5) + squareArea 5 = 50 →
  p.C.2 = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_vertex_C_y_coordinate_l495_49538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_cost_calculation_l495_49518

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn. -/
theorem gravel_cost_calculation
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sq_m : ℚ)
  (h_lawn_length : lawn_length = 80)
  (h_lawn_width : lawn_width = 60)
  (h_road_width : road_width = 15)
  (h_cost_per_sq_m : cost_per_sq_m = 3)
  : (road_width * lawn_length + road_width * (lawn_width - road_width)) * cost_per_sq_m = 5625 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_cost_calculation_l495_49518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l495_49573

noncomputable def solution_set : Set ℝ :=
  {x | x < -3 ∨ (-2 < x ∧ x < 2) ∨ x > 7}

noncomputable def inequality_function (x : ℝ) : ℝ :=
  (x^2 - 4*x - 21) / (x^2 - 4)

theorem inequality_solution :
  ∀ x : ℝ, x ≠ -2 ∧ x ≠ 2 →
    (inequality_function x > 0 ↔ x ∈ solution_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l495_49573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_half_l495_49551

theorem x_equals_y_iff_t_equals_half (t : ℚ) : 
  (1 - 4 * t = 2 * t - 2) ↔ t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_t_equals_half_l495_49551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_proof_l495_49503

theorem sequence_formula_proof (n : ℕ) :
  let a : ℕ → ℚ := fun n => (-1)^(n+1) * ((2*n + 1) : ℚ) / (2^n : ℚ)
  (a 1 = 3/2) ∧ (a 2 = -5/4) ∧ (a 3 = 7/8) ∧ (a 4 = -9/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_proof_l495_49503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l495_49530

theorem power_equation (a b : ℝ) (h1 : (60 : ℝ)^a = 3) (h2 : (60 : ℝ)^b = 5) :
  (12 : ℝ)^((1 - a - b) / (2 * (1 - b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l495_49530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_in_resistor_l495_49502

-- Define the components of the circuit
variable (C : ℝ) -- Capacitance
variable (ε : ℝ) -- Electromotive force

-- Define the heat released in the resistor
noncomputable def heat_released (C ε : ℝ) : ℝ := (1/2) * C * ε^2

-- Theorem statement
theorem heat_released_in_resistor (C ε : ℝ) (hC : C > 0) (hε : ε > 0) :
  heat_released C ε = (1/2) * C * ε^2 :=
by
  -- Unfold the definition of heat_released
  unfold heat_released
  -- The equation is now trivially true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_in_resistor_l495_49502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l495_49516

theorem sum_of_divisors_143 : (Finset.sum (Nat.divisors 143) id) = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l495_49516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_disproves_fta_l495_49511

-- Define what it means for a number to be expressible as a product of primes
def is_expressible_as_prime_product (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ n = p * q

-- Define the fundamental theorem of arithmetic
def fundamental_theorem_of_arithmetic : Prop :=
  ∀ n : ℕ, n > 1 → is_expressible_as_prime_product n

-- State the theorem
theorem counterexample_disproves_fta :
  (∃ n : ℕ, n > 1 ∧ ¬(is_expressible_as_prime_product n)) →
  ¬fundamental_theorem_of_arithmetic := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_disproves_fta_l495_49511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_simplification_l495_49544

noncomputable def y (x : ℝ) := Real.sqrt (x + 2 * Real.sqrt (x - 1)) + Real.sqrt (x - 2 * Real.sqrt (x - 1))

theorem y_simplification (x : ℝ) (h : x ≥ 1) :
  y x = if x ≤ 2 then 2 else 2 * Real.sqrt (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_simplification_l495_49544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l495_49572

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (2 * x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 3/2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l495_49572
