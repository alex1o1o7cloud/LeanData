import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l458_45886

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of tangency
def point : ℝ × ℝ := (3, 27)

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 27 * x - 54

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := 2

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := -54

-- State the theorem
theorem tangent_triangle_area :
  (1/2 : ℝ) * x_intercept * abs y_intercept = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l458_45886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_result_l458_45892

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ
  speed_positive : speed > 0

/-- The race scenario -/
structure RaceScenario where
  petya : Runner
  vasya : Runner
  race_length : ℝ
  initial_gap : ℝ

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (runner : Runner) (time : ℝ) : ℝ :=
  runner.speed * time

/-- Theorem stating the result of the second race -/
theorem second_race_result (scenario : RaceScenario) 
  (h1 : scenario.race_length = 100)
  (h2 : distance_covered scenario.petya (scenario.race_length / scenario.petya.speed) = 
        distance_covered scenario.vasya (scenario.race_length / scenario.petya.speed) + 10)
  (h3 : scenario.initial_gap = 10) :
  ∃ (finish_time : ℝ),
    distance_covered scenario.petya finish_time = scenario.race_length ∧
    distance_covered scenario.vasya finish_time = scenario.race_length - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_result_l458_45892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_factorial_l458_45835

/-- The sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * a n

/-- Theorem stating that a_n = n! for all natural numbers n -/
theorem a_eq_factorial : ∀ n : ℕ, a n = n.factorial := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_factorial_l458_45835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_prime_circumradius_is_right_angled_l458_45844

-- Define a triangle with integral side lengths and prime circumradius
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  R : ℕ
  h_prime : Nat.Prime R
  h_circumradius : (a * b * c)^2 = 16 * R^2 * (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)

-- Theorem statement
theorem triangle_with_prime_circumradius_is_right_angled (t : Triangle) :
  ∃ A : ℝ, A = 90 ∧ Real.cos A = (t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_prime_circumradius_is_right_angled_l458_45844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l458_45891

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

-- State the theorem
theorem inverse_f_undefined_at_one : 
  ∀ x : ℝ, x ≠ 5 → (∃ y : ℝ, f y = x) → x ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l458_45891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapped_pole_string_length_l458_45832

/-- Represents the properties of a cylindrical pole with a string wrapped around it. -/
structure WrappedPole where
  circumference : ℝ
  height : ℝ
  loops : ℕ

/-- Calculates the length of a string wrapped around a cylindrical pole. -/
noncomputable def stringLength (pole : WrappedPole) : ℝ :=
  let heightPerLoop := pole.height / (pole.loops : ℝ)
  let lengthPerLoop := Real.sqrt (pole.circumference ^ 2 + heightPerLoop ^ 2)
  (pole.loops : ℝ) * lengthPerLoop

/-- Theorem stating that the length of the string wrapped around the specified pole is approximately 24.0966 feet. -/
theorem wrapped_pole_string_length :
  let pole : WrappedPole := { circumference := 3, height := 16, loops := 6 }
  ∃ ε > 0, |stringLength pole - 24.0966| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapped_pole_string_length_l458_45832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_three_shortest_paths_l458_45807

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- We don't need to specify the details of the tetrahedron structure

-- Define a point on the surface of a tetrahedron
structure SurfacePoint (t : RegularTetrahedron) where
  -- We don't need to specify the details of how a point is represented

-- Define a path on the surface of a tetrahedron
structure SurfacePath (t : RegularTetrahedron) where
  start : SurfacePoint t
  finish : SurfacePoint t
  -- We don't need to specify how the path is represented

-- Define a predicate for a shortest path
def isShortestPath (t : RegularTetrahedron) (p : SurfacePath t) : Prop :=
  -- We don't need to specify the details of what makes a path shortest
  True

theorem tetrahedron_three_shortest_paths 
  (t : RegularTetrahedron) (M : SurfacePoint t) :
  ∃ (M' : SurfacePoint t),
    ∃ (p1 p2 p3 : SurfacePath t),
      p1.start = M ∧ p1.finish = M' ∧
      p2.start = M ∧ p2.finish = M' ∧
      p3.start = M ∧ p3.finish = M' ∧
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
      isShortestPath t p1 ∧
      isShortestPath t p2 ∧
      isShortestPath t p3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_three_shortest_paths_l458_45807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_bounds_l458_45830

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the expression
noncomputable def triangle_expression (t : Triangle) : ℝ :=
  (t.a^2 + t.b^2 + t.c^2) / (t.a*t.b + t.b*t.c + t.c*t.a)

-- Theorem statement
theorem triangle_expression_bounds (t : Triangle) :
  1 ≤ triangle_expression t ∧ triangle_expression t < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_bounds_l458_45830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_equals_arc_measure_l458_45843

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points on the circle
variable (P A B : ℝ × ℝ)

-- Define point Q inside the circle
variable (Q : ℝ × ℝ)

-- Define the angle measure function
noncomputable def angle_measure : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define the arc measure function
noncomputable def arc_measure : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define the distance function
noncomputable def distance : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Assumptions
variable (h1 : P ∈ circle)
variable (h2 : A ∈ circle)
variable (h3 : B ∈ circle)
variable (h4 : Q ∉ circle)
variable (h5 : angle_measure P A Q = 90)
variable (h6 : distance P Q = distance B Q)

-- Theorem
theorem angle_difference_equals_arc_measure :
  angle_measure A Q B - angle_measure P Q A = arc_measure A B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_equals_arc_measure_l458_45843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l458_45846

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line
def my_line (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

-- Define the symmetry condition
def symmetric_point_on_circle (a b : ℝ) : Prop :=
  ∀ x y x' y' : ℝ, my_circle x y → my_line a b x y → 
  (x' = x + 2*a*(2*a*x - b*y + 2)/(a^2 + b^2)) →
  (y' = y + 2*b*(2*a*x - b*y + 2)/(a^2 + b^2)) →
  my_circle x' y'

-- Define the theorem
theorem min_value_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_symm : symmetric_point_on_circle a b) : 
  (1/a + 2/b) ≥ 3 + 2*Real.sqrt 2 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l458_45846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l458_45823

noncomputable def original_expression (x : ℝ) : ℝ := 
  (3 * x^2) / (x^2 - 9) / ((9 / (x + 3)) + x - 3)

noncomputable def simplified_expression (x : ℝ) : ℝ := 3 / (x - 3)

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 0 →
    original_expression x = simplified_expression x ∧
    original_expression 6 = 1 :=
by
  intro x h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l458_45823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_primes_less_than_32_mod_32_l458_45836

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

noncomputable def sum_of_odd_primes_less_than_32 : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ p % 2 = 1 ∧ p < 32) (Finset.range 32)).sum id

theorem sum_of_odd_primes_less_than_32_mod_32 :
  sum_of_odd_primes_less_than_32 % 32 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_primes_less_than_32_mod_32_l458_45836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_reconstruction_possible_l458_45857

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define membership for a point on a line -/
instance : Membership Point Line where
  mem p l := l.a * p.x + l.b * p.y + l.c = 0

/-- A configuration of lines in general position -/
structure LineConfiguration where
  n : ℕ
  lines : Fin n → Line
  h_n : n > 2
  h_general_position : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ ∃ (p : Point), p ∈ lines i ∧ p ∈ lines j ∧ p ∈ lines k

/-- The set of intersection points of the lines -/
def intersectionPoints (config : LineConfiguration) : Set Point :=
  { p | ∃ i j, i ≠ j ∧ p ∈ config.lines i ∧ p ∈ config.lines j }

/-- A function that attempts to reconstruct the original lines from intersection points -/
def reconstructLines (config : LineConfiguration) (points : Set Point) : Option (Fin config.n → Line) :=
  sorry

/-- The theorem stating that it's possible to reconstruct the lines -/
theorem line_reconstruction_possible (config : LineConfiguration) :
  ∃ (f : Set Point → Option (Fin config.n → Line)),
    f (intersectionPoints config) = some config.lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_reconstruction_possible_l458_45857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_fill_time_exists_l458_45887

/-- The volume of the rectangular box in cubic feet -/
noncomputable def box_volume : ℝ := 60

/-- The rate at which the box is filled (in cubic feet per hour) -/
noncomputable def fill_rate (t : ℝ) : ℝ := 2 * t^2 + 5 * t

/-- The volume of sand in the box at time t -/
noncomputable def sand_volume (t : ℝ) : ℝ := (2/3) * t^3 + (5/2) * t^2

/-- Theorem: There exists a positive real number t such that the sand volume equals the box volume -/
theorem box_fill_time_exists : ∃ t : ℝ, t > 0 ∧ sand_volume t = box_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_fill_time_exists_l458_45887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_5pi_12_l458_45893

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

-- Theorem statement
theorem f_at_5pi_12 : f (5 * Real.pi / 12) = Real.sqrt 2 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_5pi_12_l458_45893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_sum_l458_45884

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (A B C O A₁ B₁ C₁ : V)

-- Define the midpoint property
def is_midpoint (M X Y : V) : Prop := M = (1/2 : ℝ) • (X + Y)

-- State the theorem
theorem midpoint_vector_sum :
  is_midpoint A₁ B C → is_midpoint B₁ A C → is_midpoint C₁ A B →
  O + A₁ + O + B₁ + O + C₁ = O + A + O + B + O + C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_vector_sum_l458_45884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AP_coordinates_l458_45822

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem
theorem max_AP_coordinates (α : ℝ) (t_m t_n t_p : ℝ) :
  0 ≤ α ∧ α < π →
  (∃ θ_m θ_n, curve_C θ_m = line_l α t_m ∧ curve_C θ_n = line_l α t_n) →
  (2 / distance (line_l α t_p) point_A = 
   1 / distance (line_l α t_m) point_A + 1 / distance (line_l α t_n) point_A) →
  (∀ t, distance (line_l α t) point_A ≤ distance (line_l α t_p) point_A) →
  line_l α t_p = (1, Real.sqrt 3) ∨ line_l α t_p = (1, -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_AP_coordinates_l458_45822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_winning_strategy_x_2020_homer_winning_strategy_x2_plus_1_l458_45859

/-- Represents a polynomial of degree 2020 with real coefficients -/
def Polynomial2020 := Fin 2021 → ℝ

/-- Represents the game state -/
structure GameState where
  coefficients : Polynomial2020
  current_player : Bool  -- true for Albert, false for Homer

/-- Represents a move in the game -/
structure Move where
  index : Fin 2021
  value : ℝ

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Checks if a polynomial is divisible by another polynomial -/
def is_divisible (f m : Polynomial2020) : Prop :=
  ∃ g : Polynomial2020, ∀ x, f x = (m x) * (g x)

/-- The winning condition for Homer -/
def homer_wins (final_state : GameState) (m : Polynomial2020) : Prop :=
  is_divisible final_state.coefficients m

/-- Function to compute the final state of the game -/
def final_state (albert_strategy : Strategy) (homer_strategy : Strategy) : GameState :=
  sorry  -- Implementation of game progression

/-- The theorem statement for part (a) -/
theorem albert_winning_strategy_x_2020 :
  ∃ (strategy : Strategy),
    ∀ (homer_strategy : Strategy),
      ¬(homer_wins (final_state strategy homer_strategy) (fun x => x^2020)) :=
sorry

/-- The theorem statement for part (b) -/
theorem homer_winning_strategy_x2_plus_1 :
  ∃ (strategy : Strategy),
    ∀ (albert_strategy : Strategy),
      homer_wins (final_state albert_strategy strategy) (fun x => x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_winning_strategy_x_2020_homer_winning_strategy_x2_plus_1_l458_45859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l458_45895

def T : Finset Nat := Finset.range 100

def multiples_of_4 : Finset Nat := T.filter (fun n => n % 4 = 0)
def multiples_of_5 : Finset Nat := T.filter (fun n => n % 5 = 0)

def remaining_integers : Finset Nat := T \ (multiples_of_4 ∪ multiples_of_5)

theorem remaining_integers_count : Finset.card remaining_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_integers_count_l458_45895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_existence_l458_45872

noncomputable def f (x : ℝ) : ℝ := x^2 / 8

def l (x y : ℝ) : Prop := x - y = 5

def is_tangent_line (m b a : ℝ) : Prop :=
  ∀ x, m * x + b = f a + (2 * a / 8) * (x - a)

def are_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_tangents_existence :
  ∃ (x₀ y₀ x₁ y₁ x₂ y₂ m₁ b₁ m₂ b₂ : ℝ),
    l x₀ y₀ ∧
    is_tangent_line m₁ b₁ x₁ ∧
    is_tangent_line m₂ b₂ x₂ ∧
    are_perpendicular m₁ m₂ ∧
    m₁ * x₀ + b₁ = y₀ ∧
    m₂ * x₀ + b₂ = y₀ ∧
    x₀ = 3 ∧
    y₀ = -2 ∧
    x₁ = 8 ∧
    y₁ = 8 ∧
    x₂ = -2 ∧
    y₂ = -1/32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_existence_l458_45872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l458_45870

/-- Given a parametric function defined by x = e^t and y = arcsin t, 
    the second derivative y''_xx is equal to (t^2 + t - 1) / (e^(3t) * sqrt((1 - t^2)^3)) -/
theorem second_derivative_parametric_function (t : ℝ) :
  let x := Real.exp t
  let y := Real.arcsin t
  let y_x' := 1 / (Real.exp t * Real.sqrt (1 - t^2))
  let y_xx'' := (t^2 + t - 1) / (Real.exp (3*t) * Real.sqrt ((1 - t^2)^3))
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, h ≠ 0 → |h| < δ → 
    |(((y + h * (y_x' + h * y_xx''/2)) - y) / h - y_x') / h - y_xx''| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l458_45870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l458_45841

/-- A triangle with side lengths not exceeding 1 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a ∧ a ≤ 1
  hb : 0 < b ∧ b ≤ 1
  hc : 0 < c ∧ c ≤ 1

/-- The semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ :=
  sorry

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem triangle_inequality (t : Triangle) :
  let p := semiperimeter t
  let R := circumradius t
  let r := inradius t
  p * (1 - 2 * R * r) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l458_45841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l458_45853

/-- The equation of the tangent line to the curve y = e^x / (x + 1) at the point (1, e/2) -/
theorem tangent_line_equation (x : ℝ) :
  let f (x : ℝ) := Real.exp x / (x + 1)
  let slope (x : ℝ) := (x * Real.exp x) / ((x + 1)^2)
  let point := (1, Real.exp 1 / 2)
  slope 1 * (x - point.1) + point.2 = (Real.exp 1 / 4) * x + (Real.exp 1 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l458_45853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_sum_remainders_l458_45863

/-- The remainder function r(m, n) -/
def r (m : ℕ) (n : ℕ+) : ℕ := m % n

/-- The sum of remainders function -/
def sum_remainders (m : ℕ) : ℕ := 
  (Finset.range 10).sum (λ i => r m ⟨i + 1, Nat.succ_pos i⟩)

/-- Main theorem: 120 is the smallest positive integer satisfying the equation -/
theorem smallest_m_satisfying_sum_remainders : 
  (∀ m : ℕ, 0 < m → m < 120 → sum_remainders m ≠ 4) ∧ sum_remainders 120 = 4 := by
  sorry

#eval sum_remainders 120  -- To check if the sum is indeed 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_sum_remainders_l458_45863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_end_condition_l458_45890

/-- Bob's action: erase three distinct numbers and replace with their sum -/
def bob_action (a b c : ℕ) : Prop :=
  3 ∣ (a + 2*b) ∧ 3 ∣ (a - c)

/-- The set of numbers Bob starts with -/
def initial_set (n : ℕ) : Set ℕ :=
  {k | 1 ≤ k ∧ k ≤ n}

/-- Bob can end up with only one number -/
def can_end_with_one (n : ℕ) : Prop :=
  ∃ (final : ℕ), ∀ (S : Set ℕ),
    S.Finite ∧ 
    (∀ x, x ∈ S → x ∈ initial_set n) ∧
    (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b ∧ b ≠ c ∧ a ≠ c → bob_action a b c → 
      ∃ (S' : Set ℕ), S' = S \ {a, b, c} ∪ {a + b + c}) →
    S = {final}

/-- The main theorem: Bob can end up with one number iff n is divisible by 3 -/
theorem bob_end_condition (n : ℕ) : 
  can_end_with_one n ↔ 3 ∣ n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_end_condition_l458_45890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l458_45860

-- Define the set S
inductive S
  | A
  | A1
  | A2
  | A3
  | A4
  | A5
deriving Repr, DecidableEq, Fintype

-- Define the operation ⊕
def oplus : S → S → S
  | S.A, y => y
  | S.A1, S.A => S.A1
  | S.A1, S.A1 => S.A2
  | S.A1, S.A2 => S.A3
  | S.A1, S.A3 => S.A4
  | S.A1, S.A4 => S.A1
  | S.A1, S.A5 => S.A2
  | S.A2, S.A => S.A2
  | S.A2, S.A1 => S.A3
  | S.A2, S.A2 => S.A4
  | S.A2, S.A3 => S.A1
  | S.A2, S.A4 => S.A2
  | S.A2, S.A5 => S.A3
  | S.A3, S.A => S.A3
  | S.A3, S.A1 => S.A4
  | S.A3, S.A2 => S.A1
  | S.A3, S.A3 => S.A2
  | S.A3, S.A4 => S.A3
  | S.A3, S.A5 => S.A4
  | S.A4, S.A => S.A4
  | S.A4, S.A1 => S.A1
  | S.A4, S.A2 => S.A2
  | S.A4, S.A3 => S.A3
  | S.A4, S.A4 => S.A4
  | S.A4, S.A5 => S.A1
  | S.A5, S.A => S.A1
  | S.A5, S.A1 => S.A2
  | S.A5, S.A2 => S.A3
  | S.A5, S.A3 => S.A4
  | S.A5, S.A4 => S.A1
  | S.A5, S.A5 => S.A2

-- Define the theorem
theorem count_solutions : 
  (Finset.filter (fun x => oplus (oplus x x) S.A2 = S.A) Finset.univ).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l458_45860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l458_45858

theorem complex_fraction_equality : 
  (1 - Complex.I) * (1 - 2*Complex.I) / (1 + Complex.I) = -2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l458_45858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l458_45818

theorem sufficient_not_necessary : 
  (∀ a : ℝ, a = 2 → |a| = 2) ∧ 
  (∃ a : ℝ, |a| = 2 ∧ a ≠ 2) := by
  constructor
  · intro a h
    rw [h]
    simp
  · use -2
    constructor
    · simp
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l458_45818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_cutting_days_l458_45814

/-- The number of days James and his brothers cut trees together -/
def days_cutting_together (james_trees_per_day : ℕ) (james_solo_days : ℕ) 
  (brother_efficiency : ℚ) (num_brothers : ℕ) (total_trees : ℕ) : ℕ :=
  let james_solo_trees := james_trees_per_day * james_solo_days
  let remaining_trees := total_trees - james_solo_trees
  let brothers_trees_per_day := (↑james_trees_per_day * brother_efficiency).floor * num_brothers
  let total_trees_per_day := james_trees_per_day + brothers_trees_per_day
  (remaining_trees / total_trees_per_day).toNat

theorem tree_cutting_days : 
  days_cutting_together 20 2 (4/5) 2 196 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_cutting_days_l458_45814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_optimization_l458_45852

/-- Surface area of a rectangular box with volume 32 and height 2, as a function of base side length -/
noncomputable def surface_area (x : ℝ) : ℝ := 4 * x + 64 / x + 32

/-- The minimum surface area of the rectangular box -/
def min_surface_area : ℝ := 64

/-- The base side length that minimizes the surface area -/
def optimal_side_length : ℝ := 4

/-- Theorem stating the optimization problem for the rectangular box -/
theorem rectangular_box_optimization :
  (∀ x > 0, surface_area x ≥ min_surface_area) ∧
  surface_area optimal_side_length = min_surface_area := by
  sorry

#check rectangular_box_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_optimization_l458_45852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_given_lines_l458_45837

/-- The point of intersection of two lines in 2D space. -/
noncomputable def intersection_point (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  let s := (d.1 - b.1 + (b.2 - d.2) * c.2 / c.1) / (a.1 + a.2 * c.2 / c.1)
  (b.1 + s * a.1, b.2 + s * a.2)

/-- The theorem stating that the intersection point of the given lines is (181/23, 69/23). -/
theorem intersection_of_given_lines :
  let line1_start : ℝ × ℝ := (2, 3)
  let line1_direction : ℝ × ℝ := (3, -4)
  let line2_start : ℝ × ℝ := (7, -5)
  let line2_direction : ℝ × ℝ := (5, 1)
  intersection_point line1_start line1_direction line2_start line2_direction = (181/23, 69/23) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_given_lines_l458_45837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l458_45816

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem function_symmetry (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∀ x, f ω (Real.pi / 3 + x) = f ω (Real.pi / 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l458_45816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_existence_l458_45865

theorem circle_placement_existence 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ)
  (num_squares : ℕ)
  (square_positions : Finset (ℝ × ℝ))
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 25)
  (h3 : num_squares = 120)
  (h4 : square_positions.card = num_squares)
  (h5 : ∀ p ∈ square_positions, 0 ≤ p.1 ∧ p.1 ≤ rectangle_width - 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ rectangle_height - 1) :
  ∃ (x y : ℝ), 
    0.5 ≤ x ∧ x ≤ rectangle_width - 0.5 ∧
    0.5 ≤ y ∧ y ≤ rectangle_height - 0.5 ∧
    ∀ p ∈ square_positions, ((x - p.1)^2 + (y - p.2)^2).sqrt > 0.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_placement_existence_l458_45865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l458_45819

theorem cos_plus_sin_value (α : ℝ) 
  (h : (Real.cos (2 * α)) / (Real.sin (α - π/4)) = -(Real.sqrt 2) / 2) :
  Real.cos α + Real.sin α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l458_45819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l458_45810

/-- An arithmetic sequence with first term a₁, second term a₂, and third term a₃ -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  a₁ : α
  a₂ : α
  a₃ : α

/-- The general term of an arithmetic sequence -/
def generalTerm (n : ℕ) (a : ℝ) (d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_theorem (x : ℝ) :
  let seq := { a₁ := x - 1, a₂ := x + 1, a₃ := 2 * x + 3 : ArithmeticSequence ℝ }
  x = 0 ∧ ∀ n : ℕ, generalTerm n seq.a₁ (seq.a₂ - seq.a₁) = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l458_45810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l458_45889

theorem book_price_change (P : ℝ) (h : P > 0) : 
  let decreased_price := P * (1 - 0.25)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price = P * 0.90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l458_45889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_frequency_count_l458_45805

theorem group_frequency_count 
  (sample_capacity : ℕ) 
  (group_frequency : ℚ) 
  (h1 : sample_capacity = 1000) 
  (h2 : group_frequency = 6/10) : 
  Int.floor (group_frequency * sample_capacity) = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_frequency_count_l458_45805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_program_student_count_l458_45849

/-- Proves that given the conditions of the lunch program, the total number of students is 50 -/
theorem lunch_program_student_count :
  ∀ (total_students : ℕ) 
    (free_lunch_percentage : ℚ)
    (paying_lunch_cost : ℚ)
    (total_cost : ℚ),
  free_lunch_percentage = 2/5 →
  paying_lunch_cost = 7 →
  total_cost = 210 →
  (1 - free_lunch_percentage) * (total_students : ℚ) * paying_lunch_cost = total_cost →
  total_students = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_program_student_count_l458_45849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wire_length_right_triangle_l458_45808

noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1/2) * a * b

noncomputable def wire_length (a b c : ℝ) : ℝ :=
  a + b + c

theorem min_wire_length_right_triangle (a b c : ℝ) :
  is_right_triangle a b c →
  triangle_area a b = 8 →
  wire_length a b c ≥ 8 + 4 * Real.sqrt 2 ∧
  (wire_length a b c = 8 + 4 * Real.sqrt 2 ↔ c = 4 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wire_length_right_triangle_l458_45808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marbles_l458_45803

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def combinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def sum_of_combination (c : Finset ℕ) : ℕ := c.sum id

def total_sum : ℕ := combinations.sum sum_of_combination

def num_combinations : ℕ := combinations.card

theorem expected_value_of_marbles :
  (total_sum : ℚ) / num_combinations = 21/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marbles_l458_45803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_when_g_has_three_zeros_l458_45899

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x^2 + 4*x - 3 else 4

-- Define function g in terms of f
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - 2*x

-- Theorem statement
theorem m_range_when_g_has_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    g m x = 0 ∧ g m y = 0 ∧ g m z = 0 ∧
    (∀ w : ℝ, g m w = 0 → w = x ∨ w = y ∨ w = z)) →
  1 < m ∧ m ≤ 2 := by
  sorry

#check m_range_when_g_has_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_when_g_has_three_zeros_l458_45899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l458_45840

noncomputable def b : ℕ → ℚ
  | 0 => 3
  | 1 => 5
  | (n + 2) => b (n + 1) + 2 * b n

noncomputable def S : ℚ := ∑' n, b n / 9^(n + 1)

theorem sum_of_sequence : S = 13/288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l458_45840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_player_games_l458_45848

theorem tennis_player_games (game_counts : Fin 70 → ℕ) 
  (h1 : ∀ i, game_counts i ≥ 1)
  (h2 : ∀ i, (Finset.range 7).sum (λ j ↦ game_counts ((i + j) % 70)) ≤ 12) :
  ∃ i j, i < j ∧ j ≤ 70 ∧ (Finset.range (j - i)).sum (λ k ↦ game_counts ((i + k) % 70)) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_player_games_l458_45848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l458_45871

theorem median_to_mean (m : ℝ) : 
  let S := ({m, m + 4, m + 7, m + 10, m + 18} : Finset ℝ)
  m + 7 = 12 →
  (Finset.sum S id / S.card : ℝ) = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l458_45871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pillars_optimal_pillars_56_l458_45864

/-- The cost function for the glass partition wall -/
noncomputable def cost_function (a x : ℝ) : ℝ := 6400 * x + 50 * a + 100 * a^2 / (x - 1)

/-- The theorem stating the optimal number of pillars -/
theorem optimal_pillars (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 2 ∧ 
  ∀ (y : ℝ), y > 2 → cost_function a x ≤ cost_function a y ∧
  x = a / 8 + 1 :=
by sorry

/-- The theorem for the specific case when a = 56 -/
theorem optimal_pillars_56 :
  ∃ (x : ℝ), x > 2 ∧ 
  ∀ (y : ℝ), y > 2 → cost_function 56 x ≤ cost_function 56 y ∧
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pillars_optimal_pillars_56_l458_45864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l458_45820

theorem trigonometric_problem (α β : Real) 
  (h1 : Real.tan α = 4/3)
  (h2 : Real.cos (β - α) = Real.sqrt 2 / 10)
  (h3 : 0 < α)
  (h4 : α < π/2)
  (h5 : π/2 < β)
  (h6 : β < π) :
  Real.sin α ^ 2 - Real.sin α * Real.cos α = 4/25 ∧ β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l458_45820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l458_45824

theorem tan_sum_from_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 15) 
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 40) : 
  Real.tan (x + y) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l458_45824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l458_45888

/-- A parabola with focus on a given line -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0
  focus_on_line : ∃ (x y : ℝ), x + 2*y - 4 = 0 ∧ x = p/2 ∧ y = 0

/-- The directrix of a parabola -/
def directrix (par : Parabola) : Set ℝ :=
  {x | x = -par.p/2}

/-- Theorem: If a parabola with equation y^2 = 2px (p > 0) has its focus on the line x + 2y - 4 = 0,
    then its directrix has the equation x = -4 -/
theorem parabola_directrix (par : Parabola) : directrix par = {x | x = -4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l458_45888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l458_45898

/-- The function f(x) = |x| / (√(1+x^2) * √(4+x^2)) --/
noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

/-- Theorem stating that the maximum value of f(x) is 1/3 --/
theorem f_max_value : ∀ x : ℝ, f x ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l458_45898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_rental_maximum_l458_45883

theorem truck_rental_maximum (total_trucks : ℕ) (saturday_trucks : ℕ) 
  (h1 : total_trucks = 24)
  (h2 : saturday_trucks ≥ 12)
  (h3 : saturday_trucks ≤ total_trucks) :
  ∃ (max_rented : ℕ),
    max_rented = total_trucks - saturday_trucks ∧
    max_rented + (max_rented / 2) = total_trucks :=
by
  use total_trucks - saturday_trucks
  constructor
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_rental_maximum_l458_45883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l458_45875

/-- Quadratic function passing through points A(-4, 0), B(1, 0), and C(0, 4) -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := 
  fun x => a * x^2 + b * x + c

/-- Area of the circumscribed circle of a triangle given its vertices -/
noncomputable def area_circumscribed_circle (A B C : ℝ × ℝ) : ℝ := sorry

theorem quadratic_properties 
  (a b c : ℝ) 
  (h1 : quadratic_function a b c (-4) = 0)
  (h2 : quadratic_function a b c 1 = 0)
  (h3 : quadratic_function a b c 0 = 4) :
  (4 * a - 2 * b + c = 6) ∧ 
  (area_circumscribed_circle (-4, 0) (1, 0) (0, 4) = 17 * π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l458_45875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l458_45854

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The problem statement -/
theorem distance_point_to_line_problem :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  let a : ℝ := -1  -- Coefficient of x in the standard form of the line equation
  let b : ℝ := 1   -- Coefficient of y in the standard form of the line equation
  let c : ℝ := -1  -- Constant term in the standard form of the line equation
  distance_point_to_line x₀ y₀ a b c = (3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_problem_l458_45854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l458_45817

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_length / train_speed_mps

/-- Theorem: A train 60 meters long, moving at 72 km/h, takes 3 seconds to pass a stationary point -/
theorem train_passing_telegraph_post :
  train_passing_time 60 72 = 3 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval with noncomputable definitions, so we'll use #check instead
#check train_passing_time 60 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l458_45817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l458_45802

open Real

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  b * cos C = (2*a - c) * cos B →
  b * sin A = a * sin B →
  c * sin A = a * sin C →
  (B = π/3 ∧ 3/2 < sin A + sin C ∧ sin A + sin C ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l458_45802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_selection_probability_l458_45885

/-- The probability of selecting a triangle with at least one vertex in the shaded region -/
def select_triangle_with_shaded_vertex 
  (total_triangles : ℕ) (shaded_triangles : ℕ) : ℚ :=
(shaded_triangles : ℚ) / total_triangles

theorem triangle_selection_probability 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles > 3) 
  (h2 : shaded_triangles ≤ total_triangles) 
  (h3 : shaded_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 
    select_triangle_with_shaded_vertex total_triangles shaded_triangles :=
by
  -- The proof goes here
  sorry

-- Example usage
example : select_triangle_with_shaded_vertex 6 4 = 2/3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_selection_probability_l458_45885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_value_l458_45880

/-- A triangle formed by the line 15x + 10y = 150 and the coordinate axes -/
structure TriangleFromLine where
  /-- The x-intercept of the line -/
  x_intercept : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The equation of the line is 15x + 10y = 150 -/
  line_eq : 15 * x_intercept = 150 ∧ 10 * y_intercept = 150

/-- The sum of the altitudes of the triangle -/
noncomputable def sum_of_altitudes (t : TriangleFromLine) : ℝ :=
  t.x_intercept + t.y_intercept + (30 * Real.sqrt 13) / 13

/-- Theorem stating that the sum of the altitudes of the triangle is (325 + 30√13) / 13 -/
theorem sum_of_altitudes_value (t : TriangleFromLine) :
  sum_of_altitudes t = (325 + 30 * Real.sqrt 13) / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_value_l458_45880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_to_work_time_l458_45833

/-- Represents the time taken for Cole's trip to work in minutes -/
noncomputable def time_to_work (speed_to_work : ℝ) (speed_back_home : ℝ) (total_time : ℝ) : ℝ :=
  let distance := (speed_to_work * speed_back_home * total_time) / (speed_to_work + speed_back_home)
  (distance / speed_to_work) * 60

/-- Theorem stating that Cole's trip to work takes 90 minutes -/
theorem cole_trip_to_work_time :
  time_to_work 30 90 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_trip_to_work_time_l458_45833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_from_min_ratio_l458_45878

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point on the left branch of a hyperbola -/
structure LeftBranchPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  h_left_branch : x < 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ :=
  (-Real.sqrt (h.a^2 + h.b^2), 0)

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

theorem eccentricity_range_from_min_ratio 
    (h : Hyperbola)
    (min_ratio : ∀ p : LeftBranchPoint h, 
      let f1 := left_focus h
      let f2 := right_focus h
      (distance p.x p.y f2.1 f2.2)^2 / distance p.x p.y f1.1 f1.2 ≥ 8 * h.a) :
    1 < eccentricity h ∧ eccentricity h ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_from_min_ratio_l458_45878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l458_45862

/-- Calculates the length of a train given the parameters of two trains passing each other. -/
theorem train_length_calculation 
  (length_A : ℝ) 
  (speed_A speed_B : ℝ) 
  (time : ℝ) 
  (h1 : length_A = 150)
  (h2 : speed_A = 120 * 1000 / 3600)
  (h3 : speed_B = 80 * 1000 / 3600)
  (h4 : time = 9) :
  ∃ length_B : ℝ, abs (length_B - 349.95) < 0.01 ∧ length_A + length_B = (speed_A + speed_B) * time :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l458_45862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l458_45809

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1) - 2^(abs x)

theorem f_inequality (m : ℝ) : f (2*m - 1) > f m ↔ m > 1 ∨ m < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l458_45809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_not_linear_l458_45811

-- Define a linear function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

-- Define the function f(x) = 7/x
noncomputable def f (x : ℝ) : ℝ := 7 / x

-- Theorem statement
theorem f_is_not_linear : ¬(is_linear f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_not_linear_l458_45811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_456_to_hundredth_l458_45842

/-- Round a number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_3_456_to_hundredth :
  roundToHundredth 3.456 = 3.46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_456_to_hundredth_l458_45842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l458_45828

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_slope (m1 m2 : ℝ) : m1 = m2 ↔ m1 • (1 : ℝ × ℝ) = m2 • (1 : ℝ × ℝ)

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
noncomputable def line_slope (a b : ℝ) : ℝ := -a / b

theorem parallel_lines_a_value :
  ∀ a : ℝ, (line_slope 1 (2*a) = line_slope (a+1) (-a)) → (a = -3/2 ∨ a = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l458_45828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l458_45812

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function Q(x)
noncomputable def Q (x : ℝ) : ℂ := 1 + Complex.exp (i * x) - Complex.exp (2 * i * x) + Complex.exp (3 * i * x) - Complex.exp (4 * i * x)

-- Theorem statement
theorem solutions_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x < 4 * Real.pi ∧ Q x = 0) ∧ S.card = 32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l458_45812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_natural_numbers_l458_45801

theorem partition_natural_numbers (r : ℚ) (h : r > 1) :
  ∃ (A B : Set ℕ), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ 
    (∀ a1 a2, a1 ∈ A → a2 ∈ A → (a1 : ℚ) / a2 ≠ r) ∧
    (∀ b1 b2, b1 ∈ B → b2 ∈ B → (b1 : ℚ) / b2 ≠ r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_natural_numbers_l458_45801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l458_45800

open Real

theorem derivative_at_one (f : ℝ → ℝ) (hf : ∀ x > 0, f x = 2 * x * (deriv (deriv f)) 1 + log x) :
  deriv f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_l458_45800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_perimeter_l458_45804

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (4 * x * (x + 2) = 2 * (4 * x + (x + 2))) → x = (1 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_perimeter_l458_45804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l458_45879

/-- The sum of the infinite series ∑(n=1 to ∞) n/3^n is equal to 3/4 -/
theorem series_sum_equals_three_fourths :
  (∑' n : ℕ+, (n : ℝ) / (3 : ℝ) ^ (n : ℕ)) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l458_45879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_B_C_l458_45813

def A : Set ℕ := {1, 2, 3, 6, 9}

def B : Set ℕ := {x | ∃ y ∈ A, x = 3 * y}

def C : Set ℕ := {x | 3 * x ∈ A}

theorem intersection_B_C : B ∩ C = {3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_B_C_l458_45813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_C_to_L_l458_45831

/-- The curve C in polar coordinates -/
noncomputable def C (θ : ℝ) : ℝ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

/-- The line in Cartesian coordinates -/
def L (x y : ℝ) : Prop := x - 2 * y - 4 * Real.sqrt 2 = 0

/-- The minimum distance from points on C to L -/
noncomputable def min_distance : ℝ := 2 * Real.sqrt 10 / 5

theorem min_distance_from_C_to_L :
  ∀ θ : ℝ, ∃ d : ℝ, d ≥ min_distance ∧
  ∃ x y : ℝ, x^2 / 4 + y^2 = 1 ∧ 
  L x y ∧
  d = Real.sqrt ((C θ * Real.cos θ - x)^2 + (C θ * Real.sin θ - y)^2) :=
by
  sorry

#check min_distance_from_C_to_L

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_C_to_L_l458_45831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l458_45874

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k - d| / Real.sqrt (a^2 + b^2 + c^2)

/-- First plane equation: 3x + 6y - 6z + 3 = 0 -/
def plane1 (x y z : ℝ) : Prop :=
  3*x + 6*y - 6*z + 3 = 0

/-- Second plane equation: 6x + 12y - 12z + 15 = 0 -/
def plane2 (x y z : ℝ) : Prop :=
  6*x + 12*y - 12*z + 15 = 0

theorem distance_between_given_planes :
  distance_between_planes 1 2 (-2) 1 2.5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l458_45874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l458_45834

/-- A point on the curve defined by the parametric equations. -/
structure ParametricPoint where
  t : ℝ
  ht : t ≠ 0

/-- The x-coordinate of a parametric point. -/
noncomputable def ParametricPoint.x (p : ParametricPoint) : ℝ := (p.t^2 + 1) / p.t

/-- The y-coordinate of a parametric point. -/
noncomputable def ParametricPoint.y (p : ParametricPoint) : ℝ := (p.t^2 - 1) / p.t

/-- The set of all points on the curve. -/
def CurvePoints : Set (ℝ × ℝ) :=
  {p | ∃ (point : ParametricPoint), (point.x, point.y) = p}

/-- Theorem stating that the curve is a hyperbola. -/
theorem curve_is_hyperbola : ∃ (a b c d e f : ℝ), 
  a ≠ 0 ∧ 
  (∀ (p : ℝ × ℝ), p ∈ CurvePoints → 
    a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 + f = 0) ∧ 
  b^2 - 4*a*c > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l458_45834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l458_45829

open Real

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 1) * exp x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 + m

theorem intersection_range :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = g m x₁ ∧ f x₂ = g m x₂ ∧ f x₃ = g m x₃) ↔
  -3/exp 1 - 1/6 < m ∧ m < -1 := by
  sorry

#check intersection_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l458_45829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_ordinary_equation_l458_45867

/-- A curve C defined by parametric equations -/
noncomputable def C (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt t - 1 / Real.sqrt t, 3 * (t + 1 / t))

/-- The condition that t is positive -/
def t_positive (t : ℝ) : Prop := t > 0

/-- The ordinary equation of curve C -/
def ordinary_equation (x y : ℝ) : Prop :=
  3 * x^2 - y + 6 = 0 ∧ y ≥ 6

/-- Theorem stating that the ordinary equation holds for curve C -/
theorem curve_C_ordinary_equation :
  ∀ t : ℝ, t_positive t →
    let (x, y) := C t
    ordinary_equation x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_ordinary_equation_l458_45867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_max_integer_k_l458_45896

-- Define the function f(x) with parameter k
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - x) * Real.exp x - x - 3

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let f₁ := f 1
  let f₁' := λ x ↦ -x * Real.exp x - 1
  f₁' 0 = -1 ∧ f₁ 0 = -2 →
  ∀ x y, y - (-2) = -1 * (x - 0) ↔ x + y + 2 = 0 :=
by sorry

-- Theorem for the maximum integer value of k
theorem max_integer_k :
  (∀ x > 0, f k x < 0) →
  k ≤ 2 ∧ 
  ∀ m : ℤ, (∀ x > 0, f m x < 0) → m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_max_integer_k_l458_45896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_savings_amount_l458_45850

def monthly_salary : ℚ := 4166.67
def initial_savings_rate : ℚ := 1/5
def expense_increase_rate : ℚ := 1/10

def new_savings : ℚ :=
  let initial_expenses := monthly_salary * (1 - initial_savings_rate)
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  monthly_salary - new_expenses

theorem new_savings_amount : ∃ ε > 0, |new_savings - 499.6704| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_savings_amount_l458_45850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l458_45815

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

def tangent_parallel (a : ℝ) : Prop :=
  (deriv (f a) 1 = 2)

def condition_holds (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ x₀ + 1 / x₀ < m * f (-1) x₀

theorem problem_solution :
  (∀ a : ℝ, tangent_parallel a → a = -1) ∧
  (∀ m : ℝ, condition_holds m → (m > (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ m < -2)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l458_45815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l458_45806

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (a + 2^x)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (∀ x, f a b x ∈ Set.univ) →
  (a = 1 ∧ b = 1) ∧
  (∀ k, (∀ t, f 1 1 (t^2 - 2*t) < f 1 1 (-2*t^2 + k)) ↔ k < -1/3) := by
  sorry

#check odd_function_and_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l458_45806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l458_45877

theorem divisibility_property (n : ℕ+) : ∃ a b : ℤ, (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l458_45877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cos_inequality_l458_45826

theorem negation_of_cos_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cos_inequality_l458_45826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l458_45821

/-- The length of a train given its speed relative to another train and the time it takes to pass a point on the other train. -/
noncomputable def train_length (relative_speed : ℝ) (passing_time : ℝ) : ℝ :=
  relative_speed * passing_time

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_mps : ℝ := 1000 / 3600

theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 15) :
  train_length ((faster_speed - slower_speed) * kmph_to_mps) passing_time = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l458_45821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seashells_after_month_l458_45856

/-- The number of seashells added each week compared to the previous week -/
def weekly_increase : ℕ := 20

/-- The initial number of seashells in the jar -/
def initial_seashells : ℕ := 50

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The number of seashells in the jar after n weeks -/
def seashells_after (n : ℕ) : ℕ :=
  initial_seashells + n * weekly_increase

/-- The total number of seashells accumulated over n weeks -/
def total_seashells (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => seashells_after i)

/-- The theorem stating the total number of seashells after a month -/
theorem seashells_after_month :
  total_seashells weeks_in_month = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seashells_after_month_l458_45856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l458_45839

/-- An ellipse with given eccentricity and focus -/
structure Ellipse where
  e : ℝ  -- eccentricity
  f : ℝ × ℝ  -- focus coordinates
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- distance from center to focus

/-- Standard equation of an ellipse -/
def standard_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.b^2 + y^2 / E.a^2 = 1

/-- Properties of the ellipse -/
def ellipse_properties (E : Ellipse) : Prop :=
  E.e = 1/2 ∧ E.f = (0, -3) ∧ E.c = 3 ∧ E.e = E.c / E.a ∧ E.a^2 = E.b^2 + E.c^2

/-- Theorem: The standard equation of the ellipse is x²/27 + y²/36 = 1 -/
theorem ellipse_equation (E : Ellipse) (x y : ℝ) :
  ellipse_properties E → standard_equation E x y ↔ x^2 / 27 + y^2 / 36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l458_45839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l458_45851

noncomputable def A (S : List ℝ) : List ℝ :=
  List.zipWith (λ a b => (a + b) / 2) S (List.tail S)

noncomputable def A_m : ℕ → List ℝ → List ℝ
  | 0, S => S
  | n + 1, S => A (A_m n S)

def S (x : ℝ) : List ℝ :=
  List.range 101 |>.map (λ n => x^n)

theorem x_value_theorem (x : ℝ) :
  x > 0 →
  A_m 100 (S x) = [1 / 2^50] →
  x = Real.sqrt 2 - 1 := by
  sorry

#check x_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l458_45851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_comparison_l458_45845

def surface_area (cube : Real → Real → Real → Real) : Real :=
  6 * (cube 1 0 0)^2

theorem cube_volume_comparison :
  ∀ (reference_cube first_cube : Real → Real → Real → Real),
  (∀ s, reference_cube s s s = 8) →
  (∀ s, 6 * s^2 = surface_area reference_cube) →
  (∀ s, 6 * s^2 = surface_area first_cube) →
  (surface_area first_cube = 4 * surface_area reference_cube) →
  (∃ s, first_cube s s s = 64) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_comparison_l458_45845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_valid_coloring_l458_45876

/-- A regular polygon with (2n + 1) sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  sides_eq : sides = 2 * n + 1

/-- A coloring of the sides of a polygon -/
structure Coloring (n : ℕ) where
  colors : Fin 3 → Set (Fin (2 * n + 1))
  cover : ∀ i, ∃ c, i ∈ colors c
  disjoint : ∀ c₁ c₂, c₁ ≠ c₂ → colors c₁ ∩ colors c₂ = ∅
  nonempty : ∀ c, colors c ≠ ∅

/-- Visibility of a side from an external point -/
def Visible (n : ℕ) (E : ℝ × ℝ) (i : Fin (2 * n + 1)) : Prop := sorry

/-- The visibility condition: from any external point, at most 2 colors are visible -/
def VisibilityCondition (n : ℕ) (coloring : Coloring n) : Prop :=
  ∀ E : ℝ × ℝ, E ∉ Set.univ → ∃ c₁ c₂, ∀ c, (∃ i ∈ coloring.colors c, Visible n E i) → c = c₁ ∨ c = c₂

/-- The existence of a valid coloring for a given n -/
def ValidColoringExists (n : ℕ) : Prop :=
  ∃ coloring : Coloring n, VisibilityCondition n coloring

/-- The main theorem: the largest n for which a valid coloring exists is 1 -/
theorem largest_n_for_valid_coloring :
  (∀ n > 1, ¬ValidColoringExists n) ∧ ValidColoringExists 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_valid_coloring_l458_45876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l458_45847

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 15*x^2 + 36*x^3) / (9 - x^3)

-- State the theorem
theorem f_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l458_45847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_truck_speed_difference_l458_45855

/-- The speed difference between a bike and a truck -/
noncomputable def speed_difference (bike_distance truck_distance : ℝ) (time : ℝ) : ℝ :=
  (bike_distance / time) - (truck_distance / time)

/-- Theorem stating the speed difference between a bike and a truck -/
theorem bike_truck_speed_difference :
  speed_difference 136 112 8 = 3 := by
  -- Unfold the definition of speed_difference
  unfold speed_difference
  -- Simplify the arithmetic expression
  simp [div_sub_div]
  -- Evaluate the numerical expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_truck_speed_difference_l458_45855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_ratio_theorem_l458_45894

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the sequence aₙ
noncomputable def a (n : ℕ) : ℝ := φ^n * fib n

-- State the theorem
theorem fibonacci_ratio_theorem : 
  ∃ (A B : ℚ), (a 30 + a 29) / (a 26 + a 25) = A + B * Real.sqrt 5 ∧ A + B = 188 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_ratio_theorem_l458_45894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l458_45825

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  ne_zero : a ≠ 0 ∨ b ≠ 0

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Calculate the distance of a line from the origin -/
noncomputable def Line.distanceFromOrigin (l : Line) : ℝ :=
  abs l.c / Real.sqrt (l.a^2 + l.b^2)

/-- The sine of the angle of inclination of a line -/
noncomputable def Line.sineOfInclination (l : Line) : ℝ :=
  abs l.b / Real.sqrt (l.a^2 + l.b^2)

theorem line_equation (l : Line) :
  (l.contains (-4, 0) ∧
   l.sineOfInclination = Real.sqrt 10 / 10 ∧
   l.contains (-2, 1) ∧
   l.distanceFromOrigin = 2) →
  (∃ k, l.a = k ∧ l.b = 0 ∧ l.c = -2 * k) ∨
  (l.a = 3 ∧ l.b = -4 ∧ l.c = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l458_45825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_90_terms_of_specific_ap_l458_45838

/-- Definition of an arithmetic progression --/
noncomputable def arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

/-- Sum of the first n terms of an arithmetic progression --/
noncomputable def sum_arithmetic_progression (a d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1) * d)

/-- Theorem stating the sum of the first 90 terms of the specific arithmetic progression --/
theorem sum_90_terms_of_specific_ap :
  ∃ (a d : ℝ),
    sum_arithmetic_progression a d 15 = 150 ∧
    sum_arithmetic_progression a d 75 = 75 ∧
    sum_arithmetic_progression a d 90 = -112.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_90_terms_of_specific_ap_l458_45838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_l458_45897

-- Define the points
variable (A B C D O H : Fin 3 → ℝ)

-- Define the conditions
axiom north : A 2 = O 2 ∧ A 1 = O 1 ∧ A 0 > O 0
axiom west : B 2 = O 2 ∧ B 0 = O 0 ∧ B 1 < O 1
axiom south : C 2 = O 2 ∧ C 1 = O 1 ∧ C 0 < O 0
axiom east : D 2 = O 2 ∧ D 0 = O 0 ∧ D 1 > O 1
axiom cd_distance : ‖C - D‖ = 160
axiom h_above_o : H 0 = O 0 ∧ H 1 = O 1 ∧ H 2 > O 2
axiom hc_length : ‖H - C‖ = 170
axiom hd_length : ‖H - D‖ = 140
axiom ha_length : ‖H - A‖ = 150

-- Theorem to prove
theorem balloon_height : ‖H - O‖ = 30 * Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_height_l458_45897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_area_with_two_inch_increase_l458_45881

/-- Represents a circular disc with a hole in the center -/
structure DiscWithHole where
  outer_diameter : ℝ
  hole_diameter : ℝ

/-- Calculates the area of a disc with a hole -/
noncomputable def area (d : DiscWithHole) : ℝ :=
  let outer_radius := d.outer_diameter / 2
  let hole_radius := d.hole_diameter / 2
  Real.pi * (outer_radius ^ 2 - hole_radius ^ 2)

/-- The initial disc -/
def initial_disc : DiscWithHole := { outer_diameter := 5, hole_diameter := 1 }

/-- The enlarged disc -/
def enlarged_disc : DiscWithHole := { outer_diameter := 7, hole_diameter := 1 }

theorem double_area_with_two_inch_increase :
  area enlarged_disc = 2 * area initial_disc := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_area_with_two_inch_increase_l458_45881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_edges_l458_45866

-- Define the points
def P₁ : ℝ × ℝ := (1, 2)
def P₂ : ℝ × ℝ := (4, 3)
def P₃ : ℝ × ℝ := (3, -1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the edges
noncomputable def edge₁ : ℝ := distance P₁ P₂
noncomputable def edge₂ : ℝ := distance P₂ P₃
noncomputable def edge₃ : ℝ := distance P₁ P₃

-- State the theorem
theorem triangle_edges :
  (max edge₁ (max edge₂ edge₃) = Real.sqrt 17) ∧
  (min edge₁ (min edge₂ edge₃) = Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_edges_l458_45866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l458_45861

theorem no_such_function : ¬∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = f (n + 1) - f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l458_45861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l458_45882

theorem negation_of_existence :
  (¬∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l458_45882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_range_l458_45827

theorem sin_difference_range (x y : ℝ) (h : Real.cos x + Real.cos y = 1) :
  ∃ t, Real.sin x - Real.sin y = t ∧ -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_range_l458_45827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l458_45869

def geometric_progression (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

theorem fourth_term_of_geometric_progression 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : a₁ = Real.sqrt 3) 
  (h2 : a₂ = 3 * (3 ^ (1/3 : ℝ))) 
  (h3 : a₃ = Real.sqrt 3) 
  (h4 : geometric_progression a₁ a₂ a₃ a₄) :
  a₄ = 3 * (3 ^ (1/3 : ℝ)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l458_45869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_compose_g_exact_range_of_f_compose_g_l458_45868

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := x + 1 / x

-- State the theorem
theorem range_of_f_compose_g :
  ∀ y, y ∈ Set.range (f ∘ g) →
  (y ∈ Set.Ioo 2 5 ∧
  ∀ ε > 0, ∃ x > 0, |f (g x) - 5| < ε) :=
by
  sorry

-- Additional theorem to state that the range is exactly (2, 5]
theorem exact_range_of_f_compose_g :
  Set.range (f ∘ g) = Set.Ioc 2 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_compose_g_exact_range_of_f_compose_g_l458_45868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_throw_properties_l458_45873

/-- Height function for a vertically thrown body -/
noncomputable def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

/-- Velocity function (derivative of height) -/
noncomputable def V (t : ℝ) : ℝ := 200 - 9.8 * t

/-- Time to reach maximum height -/
noncomputable def t_max : ℝ := 200 / 9.8

theorem vertical_throw_properties :
  (V 10 = 102) ∧ 
  (t_max = 200 / 9.8) ∧ 
  (H t_max = (200^2) / (4 * 9.8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_throw_properties_l458_45873
