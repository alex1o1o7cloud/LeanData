import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_six_schemes_l125_12549

/-- Represents a book purchasing scheme -/
structure BookScheme where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a book scheme is valid according to the given conditions -/
def is_valid_scheme (s : BookScheme) : Prop :=
  s.a ≥ 5 ∧ s.a ≤ 6 ∧ s.b > 0 ∧ s.c > 0 ∧
  30 * s.a + 25 * s.b + 20 * s.c = 500

/-- The set of all valid book purchasing schemes -/
def valid_schemes : Set BookScheme :=
  {s : BookScheme | is_valid_scheme s}

/-- Finite type instance for valid_schemes -/
instance : Fintype valid_schemes := by
  sorry

theorem exactly_six_schemes : Fintype.card valid_schemes = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_six_schemes_l125_12549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_two_l125_12510

noncomputable def vector_a : Fin 2 → ℝ := ![1, Real.sqrt 3]
noncomputable def vector_b : Fin 2 → ℝ := ![-2, 0]

theorem magnitude_of_sum_equals_two :
  Real.sqrt ((vector_a 0 + vector_b 0)^2 + (vector_a 1 + vector_b 1)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_equals_two_l125_12510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_determination_quadratic_function_b_range_l125_12512

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part 1
theorem quadratic_function_determination (a b c : ℝ) :
  f a b c 3 = -5 ∧ f a b c (-1) = -5 ∧ (∀ x, f a b c x ≤ 3) →
  ∀ x, f a b c x = f (-2) 4 1 x := by sorry

-- Part 2
theorem quadratic_function_b_range (b c : ℝ) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1 : ℝ) 1 → x₂ ∈ Set.Icc (-1 : ℝ) 1 → 
    |f 1 b c x₁ - f 1 b c x₂| ≤ 4) →
  b ∈ Set.Icc (-2 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_determination_quadratic_function_b_range_l125_12512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_not_at_end_l125_12555

/-- A sequence of numbers from 1 to 9 with the property that any two numbers
    separated by one other number differ by 1. -/
def SpecialSequence : Type := 
  { seq : Fin 9 → Fin 9 // 
    (∀ i : Fin 7, |seq i - seq (i + 2)| = 1) ∧
    (∀ i : Fin 9, (seq i : Nat) ∈ Finset.range 9) ∧
    (∀ i j : Fin 9, i ≠ j → seq i ≠ seq j) }

/-- Theorem stating that in a SpecialSequence, the number 4 cannot be at the end. -/
theorem four_not_at_end (seq : SpecialSequence) : seq.val 8 ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_not_at_end_l125_12555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_reals_l125_12503

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | (n + 1) => λ x ↦ f (f_n n x)

def M : Set ℝ := {x : ℝ | f_n 2036 x = x}

theorem M_equals_reals : M = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_reals_l125_12503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miguel_run_time_l125_12546

/-- Race parameters and runner speeds -/
structure RaceSetup where
  race_length : ℕ
  head_start : ℕ
  ariana_speed : ℕ
  miguel_speed : ℕ

/-- Calculate the time when Ariana catches up to Miguel -/
def catch_up_time (setup : RaceSetup) : ℕ :=
  (setup.miguel_speed * setup.head_start) / (setup.ariana_speed - setup.miguel_speed)

/-- Calculate the total time Miguel has run when Ariana catches up -/
def miguel_total_time (setup : RaceSetup) : ℕ :=
  setup.head_start + catch_up_time setup

/-- Theorem stating that Miguel will have run for 60 seconds when Ariana catches up -/
theorem miguel_run_time (setup : RaceSetup) 
    (h1 : setup.race_length = 500)
    (h2 : setup.head_start = 20)
    (h3 : setup.ariana_speed = 6)
    (h4 : setup.miguel_speed = 4) : 
  miguel_total_time setup = 60 := by
  sorry

#eval miguel_total_time { race_length := 500, head_start := 20, ariana_speed := 6, miguel_speed := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miguel_run_time_l125_12546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_expression_l125_12576

theorem greatest_prime_factor_of_expression :
  (Nat.factors (5^5 + 10^4)).maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_expression_l125_12576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_inequality_for_complement_M_l125_12519

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 1| - 4)

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≤ -2 ∨ x ≥ 2}

-- Theorem 1: The domain of f is M
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = M := by sorry

-- Theorem 2: For all a, b ∈ ℝ \ M, 2|a+b| < |4+ab|
theorem inequality_for_complement_M (a b : ℝ) (ha : -2 < a ∧ a < 2) (hb : -2 < b ∧ b < 2) :
  2 * |a + b| < |4 + a * b| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_inequality_for_complement_M_l125_12519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l125_12597

noncomputable def f (x : ℝ) : ℝ := -8/3 * x^2 + 16 * x - 40/3

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l125_12597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12575

noncomputable def f (x : ℝ) : ℝ := Real.tan (x/4) * (Real.cos (x/4))^2 - 2 * (Real.cos (x/4 + Real.pi/12))^2 + 1

def is_in_domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ 2 * Real.pi + 4 * k * Real.pi

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_properties :
  (∀ x : ℝ, is_in_domain x → f x = Real.sqrt 3 * Real.sin (x/2 - Real.pi/6)) ∧
  has_period f (4 * Real.pi) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 → f x ≥ -Real.sqrt 3) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 → f x ≤ -Real.sqrt 3/2) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 ∧ f x = -Real.sqrt 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 ∧ f x = -Real.sqrt 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_perfect_square_l125_12572

theorem select_perfect_square (nums : Finset ℕ) (primes : Finset ℕ) :
  nums.card = 48 →
  (nums.prod id).factors.toFinset = primes →
  primes.card = 10 →
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (n : ℕ), a * b * c * d = n ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_perfect_square_l125_12572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l125_12507

-- Define the basic structures
structure Line where

structure Plane where

-- Define the relationships
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

def subset_of_plane (l : Line) (p : Plane) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

def skew (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel_to_plane a α) 
  (h2 : subset_of_plane b α) : 
  parallel a b ∨ skew a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l125_12507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kindergarten_lineup_probability_l125_12527

def probability_no_more_than_five_girls_between_first_last_boys (total_children num_girls num_boys : ℕ) : ℚ :=
  sorry

theorem kindergarten_lineup_probability :
  let total_children : ℕ := 20
  let num_girls : ℕ := 11
  let num_boys : ℕ := 9
  let valid_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
  let total_arrangements := Nat.choose total_children num_boys
  (valid_arrangements : ℚ) / total_arrangements =
    probability_no_more_than_five_girls_between_first_last_boys total_children num_girls num_boys :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kindergarten_lineup_probability_l125_12527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_equals_5_l125_12587

/-- A parabola in the cartesian plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The parabola equation -/
def parabola_equation (p : Parabola) : ℝ → ℝ :=
  fun x => p.a * x^2 + p.b * x + p.c

/-- Condition: The parabola has vertex at (2,3) -/
def has_vertex_at_2_3 (p : Parabola) : Prop :=
  ∃ a : ℝ, ∀ x, parabola_equation p x = a * (x - 2)^2 + 3

/-- Condition: The parabola passes through the point (1,5) -/
def passes_through_1_5 (p : Parabola) : Prop :=
  parabola_equation p 1 = 5

/-- Theorem: For a parabola with vertex (2,3) passing through (1,5), 
    when expressed as y = ax^2 + bx + c, the sum a+b+c equals 5 -/
theorem parabola_sum_equals_5 (p : Parabola) 
    (h1 : has_vertex_at_2_3 p) (h2 : passes_through_1_5 p) : 
    p.a + p.b + p.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_equals_5_l125_12587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_quadratic_inequalities_l125_12532

theorem sine_and_quadratic_inequalities :
  (¬ ∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2) ∧
  (∀ x : ℝ, x^2 + x + 1 > 0) ∧
  ((¬ ∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2) ∨ (∀ x : ℝ, x^2 + x + 1 > 0)) ∧
  ¬((∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2) ∧ ¬(∀ x : ℝ, x^2 + x + 1 > 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_and_quadratic_inequalities_l125_12532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_profit_share_l125_12573

/-- Proves that C's share of the profit is 20000 given the investment ratios and total profit -/
theorem c_profit_share (a b c : ℚ) (total_profit : ℚ) : 
  a > 0 → b > 0 → c > 0 → total_profit > 0 →
  a / c = 3 / 2 → 
  a / b = 3 / 1 → 
  total_profit = 60000 → 
  (c * total_profit) / (a + b + c) = 20000 := by
  sorry

#check c_profit_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_profit_share_l125_12573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l125_12586

theorem smallest_angle_theorem (x : Real) : 
  (x > 0 ∧ (4 : Real)^(Real.sin x)^2 * (2 : Real)^(Real.cos x)^2 = 2 * (8 : Real)^(1/4)) → x = π/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l125_12586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_in_W_l125_12569

noncomputable def belongs_to_W (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, (a n + a (n + 2)) / 2 < a (n + 1)) ∧
  (∃ M : ℝ, ∀ n : ℕ, a n ≤ M)

noncomputable def seq1 (n : ℕ) : ℝ := (n : ℝ)^2 + 1
noncomputable def seq2 (n : ℕ) : ℝ := (2 * n + 9 : ℝ) / (2 * n + 11)
noncomputable def seq3 (n : ℕ) : ℝ := 2 + 4 / (n : ℝ)
noncomputable def seq4 (n : ℕ) : ℝ := 1 - 1 / (2 : ℝ)^n

theorem sequences_in_W :
  belongs_to_W seq2 ∧ belongs_to_W seq4 ∧ ¬belongs_to_W seq1 ∧ ¬belongs_to_W seq3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_in_W_l125_12569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_D_closest_l125_12523

noncomputable section

/-- The volume of a sphere -/
def V : ℝ := 1

/-- The actual value of π -/
noncomputable def π : ℝ := Real.pi

/-- The actual diameter of a sphere given its volume -/
noncomputable def actual_diameter (V : ℝ) : ℝ := (6 * V / π) ^ (1/3)

/-- Approximation A for the diameter of a sphere -/
noncomputable def approx_A (V : ℝ) : ℝ := (16 * V / 9) ^ (1/3)

/-- Approximation B for the diameter of a sphere -/
noncomputable def approx_B (V : ℝ) : ℝ := (2 * V) ^ (1/3)

/-- Approximation C for the diameter of a sphere -/
noncomputable def approx_C (V : ℝ) : ℝ := (300 * V / 157) ^ (1/3)

/-- Approximation D for the diameter of a sphere -/
noncomputable def approx_D (V : ℝ) : ℝ := (21 * V / 11) ^ (1/3)

/-- Theorem stating that approximation D is the closest to the actual diameter -/
theorem approx_D_closest :
  let error_A := |actual_diameter V - approx_A V|
  let error_B := |actual_diameter V - approx_B V|
  let error_C := |actual_diameter V - approx_C V|
  let error_D := |actual_diameter V - approx_D V|
  error_D < error_A ∧ error_D < error_B ∧ error_D < error_C :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_D_closest_l125_12523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l125_12533

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem simple_interest_time_period : ∃ t : ℝ, 
  simple_interest 5250 4 t = (1/2) * compound_interest 4000 10 2 ∧ t = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_period_l125_12533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l125_12504

noncomputable section

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point M
def M (x₀ : ℝ) : ℝ × ℝ := (x₀, Real.sqrt 3)

-- Define the angle condition
def angle_condition (O M N : ℝ × ℝ) : Prop :=
  Real.arccos ((O.1 - M.1) * (N.1 - M.1) + (O.2 - M.2) * (N.2 - M.2)) / 
    (Real.sqrt ((O.1 - M.1)^2 + (O.2 - M.2)^2) * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)) ≥ Real.pi/6

-- Define the tangent condition
def is_tangent (M N : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), N = (t * M.1, t * M.2) ∧ circle_eq N.1 N.2

theorem tangent_range (x₀ : ℝ) :
  (∃ (N : ℝ × ℝ), circle_eq N.1 N.2 ∧ 
    is_tangent (M x₀) N ∧ 
    angle_condition (0, 0) (M x₀) N) →
  -1 ≤ x₀ ∧ x₀ ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l125_12504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12541

noncomputable def f (x : ℝ) : ℝ := Real.cos ((Real.pi / 3) * x + Real.pi / 3) - 2 * (Real.cos ((Real.pi / 6) * x))^2

def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def isMonotonicIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (isPeriodic f 6) ∧
  (∀ k : ℤ, isMonotonicIncreasing f (6 * k + 1) (6 * k + 4)) := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_y_eq_x_l125_12550

/-- The set of angles with terminal side on the line y = x -/
def terminal_side_angles : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 180 + 45}

/-- The line y = x in the coordinate plane -/
def line_y_eq_x : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

/-- Predicate to check if a point is on the terminal side of an angle -/
def is_terminal_side (α : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = Real.cos α ∧ p.2 = Real.sin α

theorem angles_on_y_eq_x :
  {α : ℝ | ∃ p : ℝ × ℝ, p ∈ line_y_eq_x ∧ is_terminal_side α p} = terminal_side_angles :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_y_eq_x_l125_12550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l125_12592

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

-- Define a line forming an isosceles right triangle with the coordinate axes
def isosceles_right_triangle_with_axes (a b : ℝ) (x y : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ y - b = k * (x - a) ∧ 
    (|a - (b/k)| = |b - a*k| ∨ |a - (b/k)| = |b + a*k|)

theorem line_equation_proof :
  -- Part 1
  (∀ θ : ℝ, θ ≠ π/2 → 
    (∀ x y : ℝ, line_through_points 3 4 (Real.cos θ) (Real.sin θ) x y ↔ y = (4/3) * x)) ∧
  -- Part 2
  (∀ x y : ℝ, isosceles_right_triangle_with_axes 3 4 x y ↔ 
    (y = x + 1 ∨ y = -x + 7)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l125_12592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l125_12571

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l125_12571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l125_12506

-- Define the function f(x) = |x^2 - 2x - t|
def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_value_of_f (t : ℝ) : 
  (∀ x ∈ interval, f t x ≤ 2) ∧ 
  (∃ x ∈ interval, f t x = 2) ↔ 
  t = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l125_12506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_A_to_line_l125_12595

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 is |ax₀ + by₀ + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- The point A -/
def A : ℝ × ℝ := (2, 1)

/-- The coefficients of the line equation x - y + 1 = 0 -/
def line_coeffs : ℝ × ℝ × ℝ := (1, -1, 1)

theorem distance_from_A_to_line :
  distance_point_to_line A.1 A.2 line_coeffs.1 line_coeffs.2.1 line_coeffs.2.2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_A_to_line_l125_12595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_one_l125_12579

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x) / Real.log a

-- State the theorem
theorem increasing_f_implies_a_greater_than_one :
  ∀ a : ℝ, (∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f a x < f a y) → a > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_greater_than_one_l125_12579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_cosine_theorem_second_cosine_theorem_l125_12539

/-- Represents a trihedral angle --/
structure TrihedralAngle where
  α : Real  -- face angle
  β : Real  -- face angle
  γ : Real  -- face angle
  A : Real  -- dihedral angle
  B : Real  -- dihedral angle
  C : Real  -- dihedral angle

/-- First cosine theorem for a trihedral angle --/
theorem first_cosine_theorem (t : TrihedralAngle) :
  Real.cos t.α = Real.cos t.β * Real.cos t.γ + Real.sin t.β * Real.sin t.γ * Real.cos t.A := by
  sorry

/-- Second cosine theorem for a trihedral angle --/
theorem second_cosine_theorem (t : TrihedralAngle) :
  Real.cos t.A = -Real.cos t.B * Real.cos t.C + Real.sin t.B * Real.sin t.C * Real.cos t.α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_cosine_theorem_second_cosine_theorem_l125_12539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_correct_l125_12500

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + (3 * Real.pi) / 2)

theorem transformations_correct (x : ℝ) : 
  (g x = f (2 * (x + Real.pi / 4))) ∧ 
  (g x = f (2 * (x - Real.pi / 4))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_correct_l125_12500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l125_12599

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²)/a -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity_m (m : ℝ) :
  m > 0 →
  eccentricity 2 m = Real.sqrt 3 →
  m = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l125_12599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_air_hamiltonian_l125_12540

/-- Represents the graph of airports and flights -/
structure AirportGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)

/-- The number of airports -/
def num_airports : Nat := 100

/-- The traffic of an airport is its degree in the graph -/
def traffic (G : AirportGraph) (v : Nat) : Nat :=
  (G.edges.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- Super-Air's graph -/
noncomputable def super_air_graph : AirportGraph :=
  sorry

/-- Concur-Air's graph based on Super-Air's graph -/
def concur_air_graph (G : AirportGraph) : AirportGraph :=
  { vertices := G.vertices,
    edges := G.edges.filter (fun e => traffic G e.1 + traffic G e.2 ≥ 100) }

/-- A path in a graph is a list of vertices where consecutive vertices are connected by edges -/
def is_path (G : AirportGraph) (path : List Nat) : Prop :=
  ∀ i, i + 1 < path.length → ((path.get? i).get! , (path.get? (i+1)).get!) ∈ G.edges

/-- A Hamiltonian cycle is a path that visits all vertices exactly once and returns to the start -/
def is_hamiltonian_cycle (G : AirportGraph) (cycle : List Nat) : Prop :=
  cycle.length = G.vertices.card + 1 ∧
  cycle.head? = cycle.getLast? ∧
  cycle.toFinset = G.vertices ∧
  is_path G cycle

theorem super_air_hamiltonian (G : AirportGraph) :
  (∃ cycle, is_hamiltonian_cycle (concur_air_graph G) cycle) →
  (∃ cycle, is_hamiltonian_cycle G cycle) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_air_hamiltonian_l125_12540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_is_e_l125_12525

-- Define the function f(x) = x²e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- Theorem statement
theorem f_upper_bound_is_e :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x < m) ↔ m > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_is_e_l125_12525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_equivalence_l125_12596

noncomputable section

open Real

def transformation1 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (2 * x)
def transformation2 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x / 2)
def transformation3 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - π / 3)
def transformation4 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x + π / 3)
def transformation5 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - 2 * π / 3)
def transformation6 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x + 2 * π / 3)

def sin_transformed (x : ℝ) : ℝ := sin (x / 2 + π / 3)

theorem sin_transformation_equivalence :
  (∀ x, (transformation2 (transformation4 sin)) x = sin_transformed x) ∧
  (∀ x, (transformation6 (transformation2 sin)) x = sin_transformed x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_equivalence_l125_12596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_dress_price_l125_12516

/-- Calculates the final price of a dress for staff members after discounts -/
theorem staff_dress_price (d : ℝ) : 
  d * (1 - 0.25) * (1 - 0.20) = 0.60 * d := by
  -- Expand the left-hand side
  have h1 : d * (1 - 0.25) * (1 - 0.20) = d * 0.75 * 0.80 := by ring
  -- Simplify the right-hand side
  have h2 : 0.60 * d = d * 0.60 := by ring
  -- Rewrite using h1 and h2
  rw [h1, h2]
  -- Prove equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_staff_dress_price_l125_12516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12558

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∀ x, f x = 2 * Real.cos (2 * x - 2 * Real.pi / 3)) ∧
  (∀ x, f ((-Real.pi / 6) + x) = f ((-Real.pi / 6) - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_equals_open_one_three_l125_12553

open Set Real

def A : Set ℝ := {x | Real.log (x - 1) < 0}
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x - 1}

theorem A_union_B_equals_open_one_three : A ∪ B = Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_union_B_equals_open_one_three_l125_12553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l125_12570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 4)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l125_12570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_marks_l125_12538

theorem passing_marks (total_marks passing_marks : ℝ) 
  (h1 : 0.3 * total_marks = passing_marks - 60)
  (h2 : 0.45 * total_marks = passing_marks + 30) : 
  passing_marks = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_passing_marks_l125_12538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_mean_of_numbers_l125_12578

noncomputable def numbers : List ℝ := [16, 24, 32, 48]

noncomputable def arithmetic_mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def geometric_mean (xs : List ℝ) : ℝ := (xs.prod) ^ (1 / xs.length)

theorem arithmetic_and_geometric_mean_of_numbers :
  arithmetic_mean numbers = 30 ∧ 
  geometric_mean numbers = (589824 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_mean_of_numbers_l125_12578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l125_12563

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Statement 1: f is increasing for all a
theorem f_increasing (a : ℝ) : StrictMono (f a) := by sorry

-- Statement 2: f is odd iff a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x, f a x = -(f a (-x))) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l125_12563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l125_12520

/-- The number of tiles needed to cover a rectangular room completely -/
def tiles_needed (room_length room_width tile_length tile_width : ℚ) : ℕ :=
  (room_length * room_width / (tile_length * tile_width)).ceil.toNat

/-- Theorem stating the number of tiles needed for the given room and tile dimensions -/
theorem tiles_for_room : tiles_needed 10 15 (1/4) (5/12) = 1440 := by
  -- Proof goes here
  sorry

#eval tiles_needed 10 15 (1/4) (5/12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l125_12520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_existence_l125_12561

/-- 
Given four positive real numbers a, b, c, and d that can form a quadrilateral,
prove that they can also form a cyclic quadrilateral.
-/
theorem cyclic_quadrilateral_existence 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (quad : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) :
  ∃ (α : ℝ), 
    0 < α ∧ 
    α < Real.pi ∧
    Real.cos α = (a^2 + d^2 - b^2 - c^2) / (2*a*d + 2*b*c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_existence_l125_12561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l125_12594

/-- An ellipse with equation x²/49 + y²/24 = 1 -/
structure Ellipse where
  equation : ℝ → ℝ → Prop
  eq_def : ∀ x y : ℝ, equation x y ↔ x^2 / 49 + y^2 / 24 = 1

/-- The foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  P : ℝ × ℝ
  on_ellipse : e.equation P.1 P.2

/-- The ratio of distances from P to F₁ and F₂ -/
noncomputable def distance_ratio (e : Ellipse) (f : Foci e) (p : PointOnEllipse e) : ℝ :=
  dist p.P f.F₁ / dist p.P f.F₂

/-- The area of triangle PF₁F₂ -/
noncomputable def triangle_area (e : Ellipse) (f : Foci e) (p : PointOnEllipse e) : ℝ :=
  abs ((p.P.1 - f.F₁.1) * (p.P.2 - f.F₂.2) - (p.P.1 - f.F₂.1) * (p.P.2 - f.F₁.2)) / 2

theorem ellipse_triangle_area 
  (e : Ellipse) (f : Foci e) (p : PointOnEllipse e) 
  (h : distance_ratio e f p = 4 / 3) : 
  triangle_area e f p = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l125_12594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l125_12574

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2^x)
def domain_f_exp (x : ℝ) : Prop := x ∈ Set.Icc (-1 : ℝ) 1

-- Define the domain of f(2x+1)
def domain_f_linear (x : ℝ) : Prop := x ∈ Set.Icc (-1/4 : ℝ) (1/2 : ℝ)

-- Theorem statement
theorem domain_equivalence :
  (∀ x, domain_f_exp x ↔ f (2^x) ∈ Set.range f) →
  (∀ x, domain_f_linear x ↔ f (2*x + 1) ∈ Set.range f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l125_12574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_sequence_l125_12530

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 4
  | 1 => 12 * y^2
  | 2 => 36 * y^5
  | 3 => 108 * y^9
  | n + 4 => geometric_sequence y n * (3 * y^(n + 1))

theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 324 * y^14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_of_sequence_l125_12530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_m_value_l125_12531

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log x + x^2 - 5*x

theorem tangent_slope_implies_m_value (m : ℝ) :
  (deriv (f m)) 1 = Real.tan (3 * π / 4) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_implies_m_value_l125_12531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_three_quarters_plus_half_ln_two_l125_12590

open Real MeasureTheory

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := x^2 / 4 - (log x) / 2

/-- The derivative of the function f -/
noncomputable def f_deriv (x : ℝ) : ℝ := (x^2 - 1) / (2 * x)

/-- The arc length of the curve y = x²/4 - (ln x)/2 from x = 1 to x = 2 -/
noncomputable def arc_length : ℝ := ∫ x in Set.Icc 1 2, sqrt (1 + (f_deriv x)^2)

theorem arc_length_equals_three_quarters_plus_half_ln_two :
  arc_length = 3/4 + (1/2) * log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_three_quarters_plus_half_ln_two_l125_12590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_three_lines_l125_12585

/-- Represents a line in the form y = k(x - 2) + 3 --/
structure Line where
  k : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the x-intercept of the line --/
noncomputable def x_intercept (l : Line) : Point := 
  { x := 2 - 3 / l.k, y := 0 }

/-- Calculates the y-intercept of the line --/
noncomputable def y_intercept (l : Line) : Point := 
  { x := 0, y := 3 - 2 * l.k }

/-- Calculates the area of the triangle formed by the line's intercepts and the origin --/
noncomputable def triangle_area (l : Line) : ℝ :=
  1/2 * |2 - 3 / l.k| * |3 - 2 * l.k|

/-- The main theorem stating that there are exactly 3 lines satisfying the conditions --/
theorem exact_three_lines : 
  ∃ (k₁ k₂ k₃ : ℝ), k₁ ≠ k₂ ∧ k₂ ≠ k₃ ∧ k₁ ≠ k₃ ∧
  (∀ k : ℝ, triangle_area { k := k } = 12 ↔ k = k₁ ∨ k = k₂ ∨ k = k₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_three_lines_l125_12585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exp_sum_magnitude_l125_12514

open Complex

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : exp (I * θ) = cos θ + I * sin θ

/-- The main theorem -/
theorem complex_exp_sum_magnitude :
  abs ((exp (I * (π/3 : ℝ))) + (exp (I * (5*π/6 : ℝ)))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_exp_sum_magnitude_l125_12514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l125_12593

theorem sin_cos_inequality (a b c d : ℝ) :
  a ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  b ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  c ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  d ∈ Set.Icc (-Real.pi/2) (Real.pi/2) →
  Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1 →
  Real.cos (2*a) + Real.cos (2*b) + Real.cos (2*c) + Real.cos (2*d) ≥ 10/3 →
  a ∈ Set.Icc 0 (Real.pi/6) ∧ b ∈ Set.Icc 0 (Real.pi/6) ∧ 
  c ∈ Set.Icc 0 (Real.pi/6) ∧ d ∈ Set.Icc 0 (Real.pi/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_inequality_l125_12593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l125_12528

theorem largest_b_value : 
  ∃ (max_b : ℝ), 
    (∀ b : ℝ, (3 * b + 4) * (b - 2) = 9 * b → b ≤ max_b) ∧ 
    max_b = (11 + Real.sqrt 217) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l125_12528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l125_12584

/-- Represents a decimal expansion of a real number -/
structure DecimalExpansion (α : Type*) [LinearOrder α] where
  digits : ℕ → α
  firstNonzeroIndex : ℕ

/-- Function to convert a rational number to its decimal expansion -/
noncomputable def toDecimalExpansion (q : ℚ) : DecimalExpansion ℕ :=
  { digits := λ n => sorry,  -- Placeholder for actual implementation
    firstNonzeroIndex := sorry }  -- Placeholder for actual implementation

theorem zeros_before_first_nonzero_digit (n d : ℕ) (h : d ≠ 0) :
  let f := n / d
  let decimal_expansion := toDecimalExpansion (↑n / ↑d)
  n = 3 ∧ d = 1250 → decimal_expansion.firstNonzeroIndex = 3 :=
by
  sorry

#check zeros_before_first_nonzero_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l125_12584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_of_cubic_with_constraints_l125_12577

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem max_slope_of_cubic_with_constraints 
  (a b c : ℝ) 
  (p : ℝ) 
  (h_p : π/3 ≤ p ∧ p ≤ 2*π/3) 
  (h_extreme : (∀ x, f a b c x ≤ f a b c p) ∧ (∀ x, f a b c 0 ≤ f a b c x)) 
  (h_curve : f a b c p = p^2 * Real.sin p + p * Real.cos p) :
  (∀ x, (deriv (f a b c)) x ≤ 3*π/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_of_cubic_with_constraints_l125_12577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_60_equals_31_l125_12598

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 50 then 0.5 * x else 25 + 0.6 * (x - 50)

theorem f_60_equals_31 : f 60 = 31 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_60_equals_31_l125_12598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bound_l125_12521

theorem gcd_bound (a b : ℕ+) (h : ((a + 1 : ℚ) / b + (b + 1 : ℚ) / a).isInt) : 
  Nat.gcd a.val b.val ≤ Nat.sqrt (a.val + b.val) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bound_l125_12521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l125_12583

theorem smallest_class_size : ∃ (N : ℕ), N > 0 ∧
  (∀ (k : ℕ), k ∈ ({6, 8, 12, 15} : Set ℕ) → N % k = 0) ∧
  N / 8 ≥ 4 ∧
  N / 12 ≥ 6 ∧
  (∀ (M : ℕ), M > 0 ∧
    (∀ (k : ℕ), k ∈ ({6, 8, 12, 15} : Set ℕ) → M % k = 0) ∧
    M / 8 ≥ 4 ∧
    M / 12 ≥ 6 →
    M ≥ N) ∧
  N = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l125_12583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_eq_60_l125_12554

/-- The number of coins -/
def n : ℕ := 64

/-- The probability of getting heads on a single toss -/
noncomputable def p : ℝ := 1/2

/-- The number of tosses allowed -/
def max_tosses : ℕ := 4

/-- The probability of getting heads after up to four tosses -/
noncomputable def prob_heads : ℝ := 1 - (1 - p)^max_tosses

/-- The expected number of coins showing heads after up to four tosses -/
noncomputable def expected_heads : ℝ := n * prob_heads

theorem expected_heads_eq_60 : expected_heads = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_eq_60_l125_12554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_optimal_quadratic_valid_l125_12513

/-- A quadratic function passing through (2, -1) with two distinct x-intercepts -/
def QuadraticFunction (p q : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧
    (∀ x, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) ∧
    4 + 2*p + q = -1

/-- The area of the triangle formed by the vertex and x-intercepts of a quadratic function -/
noncomputable def TriangleArea (p q : ℝ) : ℝ :=
  (1/8) * Real.sqrt ((p^2 - 4*q)^3)

/-- The theorem stating that x^2 - 4x + 3 minimizes the triangle area -/
theorem min_triangle_area :
  ∀ (p q : ℝ), QuadraticFunction p q →
    TriangleArea p q ≥ TriangleArea (-4) 3 := by
  sorry

/-- The optimal quadratic function -/
def optimal_quadratic (x : ℝ) : ℝ :=
  x^2 - 4*x + 3

/-- Theorem stating that the optimal quadratic function satisfies the conditions -/
theorem optimal_quadratic_valid :
  QuadraticFunction (-4) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_optimal_quadratic_valid_l125_12513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_constraint_l125_12505

-- Define a polynomial of degree exactly 3
def P (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem polynomial_value_constraint :
  ∀ a b c d : ℝ,
  (P a b c d 0 = 1) → (P a b c d 1 = 3) → (P a b c d 3 = 10) →
  (P a b c d 2 ≠ 6) :=
by
  intros a b c d h0 h1 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_constraint_l125_12505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l125_12562

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) + Real.sin (2 * x)

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

theorem min_shift_for_even_function :
  ∃ m : ℝ, m > 0 ∧
    is_even_function (fun x ↦ f (x + m)) ∧
    ∀ m' : ℝ, m' > 0 → is_even_function (fun x ↦ f (x + m')) → m ≤ m' :=
by
  -- The proof goes here
  sorry

#check min_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l125_12562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_points_l125_12526

/-- Parabola struct representing y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Point struct representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating that Q1 and Q2 minimize the distance PQ -/
theorem minimal_distance_points (para : Parabola) :
  let P := Point.mk (3 * para.p) 0
  let Q1 := Point.mk (2 * para.p) (2 * para.p)
  let Q2 := Point.mk (2 * para.p) (-2 * para.p)
  ∀ Q : Point, Q.y^2 = 2 * para.p * Q.x →
    distance P Q ≥ distance P Q1 ∧ distance P Q ≥ distance P Q2 := by
  sorry

#check minimal_distance_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_points_l125_12526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_result_l125_12548

noncomputable def f (α : Real) : Real :=
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α))

theorem f_simplification_and_result (α : Real) (h : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  f α = 2 - (Real.sin α + Real.cos α) ∧
  (f α = 3/5 → (Real.sin α) / (1 + Real.cos α) + (Real.cos α) / (1 + Real.sin α) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_result_l125_12548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_l125_12580

theorem trig_sum (α : Real) (h1 : Real.tan α = -2) (h2 : π/2 < α) (h3 : α < π) : 
  Real.cos α + Real.sin α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_l125_12580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l125_12518

theorem problem_solution : ∀ x y z : ℤ,
  x > 0 → y > 0 → z > 0 →
  x ≥ y → y ≥ z →
  x^2 - y^2 - z^2 + x*y = 3007 →
  x^2 + 4*y^2 + 4*z^2 - 4*x*y - 3*x*z - 3*y*z = -2901 →
  x = 59 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l125_12518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l125_12535

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points -/
def count_paths (start : Point) (finish : Point) : ℕ :=
  sorry

/-- Calculates the number of paths between two points that pass through a given point -/
def count_paths_through (start : Point) (through : Point) (finish : Point) : ℕ :=
  sorry

theorem jack_to_jill_paths : 
  let start := Point.mk 0 0
  let finish := Point.mk 4 3
  let dangerous := Point.mk 2 1
  let total_paths := count_paths start finish
  let dangerous_paths := count_paths_through start dangerous finish
  total_paths - dangerous_paths = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_to_jill_paths_l125_12535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l125_12509

theorem arithmetic_geometric_sequence_ratio (α₁ β : ℝ) :
  let α : ℕ → ℝ := λ n => α₁ + (n - 1) * β
  ∃ q : ℝ, (∀ n : ℕ, Real.sin (α (n + 1)) = q * Real.sin (α n)) →
  (q = 1 ∨ q = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l125_12509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l125_12568

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := if x ≥ 0 ∧ y ≥ 0 then Real.arctan (y / x)
           else if x < 0 ∧ y ≥ 0 then Real.pi - Real.arctan (y / (-x))
           else if x < 0 ∧ y < 0 then Real.pi + Real.arctan ((-y) / (-x))
           else 2 * Real.pi - Real.arctan ((-y) / x)
  let φ := Real.arccos (z / ρ)
  (ρ, θ, φ)

theorem rectangular_to_spherical_conversion :
  let x : ℝ := Real.sqrt 2
  let y : ℝ := -3
  let z : ℝ := 5
  let (ρ, θ, φ) := rectangular_to_spherical x y z
  ρ = 6 ∧ 
  θ = 2 * Real.pi - Real.arctan ((3 * Real.sqrt 11) / 2) ∧
  φ = Real.arccos (5 / 6) ∧
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_conversion_l125_12568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_mass_is_fifteen_l125_12591

/-- Represents the composition of an alloy mixture --/
structure AlloyMixture where
  first_alloy_mass : ℝ
  second_alloy_mass : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_chromium_percent : ℝ
  new_alloy_chromium_percent : ℝ

/-- Checks if the alloy mixture satisfies the given conditions --/
def is_valid_mixture (m : AlloyMixture) : Prop :=
  m.first_alloy_chromium_percent = 12 ∧
  m.second_alloy_chromium_percent = 8 ∧
  m.second_alloy_mass = 40 ∧
  m.new_alloy_chromium_percent = 9.090909090909092

/-- Calculates the total chromium mass in the mixture --/
noncomputable def total_chromium_mass (m : AlloyMixture) : ℝ :=
  (m.first_alloy_mass * m.first_alloy_chromium_percent +
   m.second_alloy_mass * m.second_alloy_chromium_percent) / 100

/-- Calculates the total mass of the mixture --/
def total_mass (m : AlloyMixture) : ℝ :=
  m.first_alloy_mass + m.second_alloy_mass

/-- Theorem stating that the first alloy mass is approximately 15 kg --/
theorem first_alloy_mass_is_fifteen (m : AlloyMixture) 
  (h : is_valid_mixture m) :
  ∃ ε > 0, abs (m.first_alloy_mass - 15) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_mass_is_fifteen_l125_12591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_life_is_six_hours_l125_12522

/-- Represents the battery life of a tablet --/
structure TabletBattery where
  full_idle_hours : ℝ
  full_use_hours : ℝ
  total_on_hours : ℝ
  used_hours : ℝ

/-- Calculates the remaining battery life of a tablet --/
noncomputable def remaining_battery_life (tb : TabletBattery) : ℝ :=
  let idle_rate := 1 / tb.full_idle_hours
  let use_rate := 1 / tb.full_use_hours
  let idle_hours := tb.total_on_hours - tb.used_hours
  let battery_used := idle_hours * idle_rate + tb.used_hours * use_rate
  let battery_remaining := 1 - battery_used
  battery_remaining / idle_rate

/-- Theorem stating that the remaining battery life is 6 hours --/
theorem remaining_life_is_six_hours (tb : TabletBattery) 
  (h1 : tb.full_idle_hours = 20)
  (h2 : tb.full_use_hours = 4)
  (h3 : tb.total_on_hours = 6)
  (h4 : tb.used_hours = 2) :
  remaining_battery_life tb = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_life_is_six_hours_l125_12522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_part1_circle_equation_part2_l125_12566

/- Part 1 -/
theorem circle_equation_part1 (a b : ℝ) (h1 : b = -a) (h2 : (a - 2)^2 + b^2 = a^2 + (b + 4)^2) :
  ∃ (x y : ℝ), (x - 3)^2 + (y + 3)^2 = 10 :=
by sorry

/- Part 2 -/
theorem circle_equation_part2 (a b c : ℝ) 
  (h1 : 5 * a - 3 * b = 8) 
  (h2 : |a| = c) 
  (h3 : |b| = c) :
  ((a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 1 ∧ b = -1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_part1_circle_equation_part2_l125_12566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_l125_12537

def is_meaningful (x : ℝ) : Prop := x ≠ -1

theorem fraction_meaningful_iff (x : ℝ) : is_meaningful x ↔ (x - 1) / (x + 1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_iff_l125_12537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_as_fraction_l125_12589

noncomputable def F : ℚ := 0.48181818  -- Representing the repeating decimal as a rational number

theorem F_as_fraction :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧
    F = a / b ∧
    Nat.Coprime a b ∧
    b - a = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_as_fraction_l125_12589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stretch_stop_frequency_is_two_hours_l125_12552

/-- Represents the road trip scenario -/
structure RoadTrip where
  initial_duration : ℚ  -- Initial trip duration in hours
  food_stops : ℕ        -- Number of food stops
  gas_stops : ℕ         -- Number of gas stops
  stop_duration : ℚ     -- Duration of each stop in hours
  total_duration : ℚ    -- Total trip duration with stops in hours

/-- Calculates the frequency of stops to stretch legs -/
noncomputable def stretch_stop_frequency (trip : RoadTrip) : ℚ :=
  let total_stop_time := trip.total_duration - trip.initial_duration
  let food_gas_stop_time := (trip.food_stops + trip.gas_stops : ℚ) * trip.stop_duration
  let stretch_stop_time := total_stop_time - food_gas_stop_time
  let num_stretch_stops := stretch_stop_time / trip.stop_duration
  trip.initial_duration / num_stretch_stops

/-- Theorem: Given the road trip conditions, the frequency of stops to stretch legs is 2 hours -/
theorem stretch_stop_frequency_is_two_hours (trip : RoadTrip) 
  (h1 : trip.initial_duration = 14)
  (h2 : trip.food_stops = 2)
  (h3 : trip.gas_stops = 3)
  (h4 : trip.stop_duration = 1/3)  -- 20 minutes in hours
  (h5 : trip.total_duration = 18) :
  stretch_stop_frequency trip = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stretch_stop_frequency_is_two_hours_l125_12552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l125_12581

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem common_tangent_implies_a_value (a : ℝ) :
  (∃ t : ℝ, t > 0 ∧ f a t = g t ∧ (deriv (f a)) t = (deriv g) t) →
  a = Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_value_l125_12581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_u_l125_12564

noncomputable section

def a : ℝ × ℝ := (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
def b : ℝ × ℝ := (Real.sin (20 * Real.pi / 180), Real.cos (20 * Real.pi / 180))

def u (t : ℝ) : ℝ × ℝ := (a.1 + t * b.1, a.2 + t * b.2)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem min_magnitude_u :
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 / 2 ∧
  ∀ (t : ℝ), magnitude (u t) ≥ min_val := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_u_l125_12564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l125_12534

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- The theorem to be proved -/
theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : eccentricity a b = 1/2)
  (h4 : ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)), 
    (∃ F₂, F₂ ∈ l) ∧ 
    A ∈ l ∧ B ∈ l ∧
    ellipse A.1 A.2 a b ∧
    ellipse B.1 B.2 a b) :
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)), 
    (∃ F₂, F₂ ∈ l) → 
    A ∈ l → B ∈ l →
    ellipse A.1 A.2 a b →
    ellipse B.1 B.2 a b →
    (A.2 / (A.1 - 4) + B.2 / (B.1 - 4) = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l125_12534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_equations_l125_12545

-- Define a Point type for 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem four_point_equations (A B C D : Point) : 
  (distance A B + distance C D = distance A C + distance B D) ∨
  (distance A B + distance B C = distance A C + distance B D) ∨
  (distance A B - distance C D = distance A C + distance B D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_equations_l125_12545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_property_l125_12582

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
def PointOnParabola (c : Parabola) := { point : ℝ × ℝ // point.1 ^ 2 = 2 * c.p * point.2 }

/-- Focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (0, c.p / 2)

/-- Intersection of tangent line with y-axis -/
def tangentIntersection (c : Parabola) (P : PointOnParabola c) : ℝ × ℝ :=
  (0, -P.val.2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem parabola_tangent_property (c : Parabola) (P : PointOnParabola c) :
  distance P.val (focus c) = 5 →
  distance (tangentIntersection c P) (focus c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_property_l125_12582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12529

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 2*c*x

theorem f_properties (c : ℝ) :
  -- 1. f'(x) is symmetric about x = 2
  (∀ x : ℝ, deriv (f c) x = deriv (f c) (4 - x)) ∧
  -- 2. f(x) has no extreme value when c ≥ 6
  (c ≥ 6 → ∀ x : ℝ, deriv (f c) x ≥ 0) ∧
  -- 3. When f(x) has a minimum value at x = t, t > 2 and the minimum value g(t) < 8
  (∃ t : ℝ, t > 2 ∧ IsLocalMin (f c) t ∧ f c t < 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l125_12529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_no_obtuse_triangle_l125_12544

-- Define a circle
structure Circle where

-- Define a point on a circle
structure Point (c : Circle) where

-- Define a function to choose n points uniformly at random on a circle
def chooseRandomPoints (c : Circle) (n : ℕ) : List (Point c) := sorry

-- Define a function to check if three points form an obtuse triangle with the circle's center
def isObtuseTriangle (c : Circle) (p1 p2 p3 : Point c) : Prop := sorry

-- Define the probability function
noncomputable def probability (event : Prop) : ℝ := sorry

-- The main theorem
theorem four_points_no_obtuse_triangle (c : Circle) :
  let points := chooseRandomPoints c 4
  probability (∀ (p1 p2 p3 : Point c),
    p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ isObtuseTriangle c p1 p2 p3) = 9 / 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_no_obtuse_triangle_l125_12544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l125_12559

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given point in polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (5, 5 * Real.pi / 4)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ := (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l125_12559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l125_12524

theorem count_integers_satisfying_inequality : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ (n + 9) * (n - 4) * (n - 13) < 0) (Finset.range 14)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l125_12524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l125_12501

def is_symmetric_about_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, -y) ∈ T

def is_symmetric_about_x_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (x, -y) ∈ T

def is_symmetric_about_y_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, y) ∈ T

def is_symmetric_about_y_eq_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (y, x) ∈ T

def is_symmetric_about_y_eq_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-y, -x) ∈ T

theorem smallest_symmetric_set_size
  (T : Set (ℝ × ℝ))
  (h1 : is_symmetric_about_origin T)
  (h2 : is_symmetric_about_x_axis T)
  (h3 : is_symmetric_about_y_axis T)
  (h4 : is_symmetric_about_y_eq_x T)
  (h5 : is_symmetric_about_y_eq_neg_x T)
  (h6 : (3, 4) ∈ T) :
  ∃ (S : Finset (ℝ × ℝ)), ↑S ⊆ T ∧ S.card = 8 ∧ (∀ (U : Finset (ℝ × ℝ)), ↑U ⊆ T → U.card < 8 → U ≠ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l125_12501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_formula_l125_12588

noncomputable section

/-- The height of an isosceles trapezoid given its area and the angle between its diagonals. -/
def trapezoidHeight (S : ℝ) (α : ℝ) : ℝ :=
  Real.sqrt (S * Real.tan (α / 2))

theorem trapezoid_height_formula (S α : ℝ) (hS : S > 0) (hα : 0 < α ∧ α < π) :
  trapezoidHeight S α = Real.sqrt (S * Real.tan (α / 2)) := by
  -- Unfold the definition of trapezoidHeight
  unfold trapezoidHeight
  -- The rest of the proof would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_formula_l125_12588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l125_12547

-- Define the coefficients of the line equations
noncomputable def A : ℝ := 2
noncomputable def B : ℝ := -1
noncomputable def C₁ : ℝ := 0
noncomputable def C₂ : ℝ := Real.sqrt 5

-- Define the distance function between two parallel lines
noncomputable def distance_between_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₁ - C₂) / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_between_given_lines :
  distance_between_lines A B C₁ C₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l125_12547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_machine_cartridge_time_l125_12508

-- Define the rates and times
def first_machine_cartridge_rate : ℚ := 800 / 12
def combined_cartridge_rate : ℚ := 800 / 3
def first_machine_envelope_rate : ℚ := 1000 / 20
def combined_envelope_rate : ℚ := 1000 / 5

-- Define the theorem
theorem second_machine_cartridge_time (x : ℚ)
  (h1 : first_machine_cartridge_rate + (800 / x) = combined_cartridge_rate)
  (h2 : first_machine_envelope_rate + (1000 / (800 / (800 / x))) = combined_envelope_rate) :
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_machine_cartridge_time_l125_12508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l125_12515

/-- The equation of a circle with center (0, 4) passing through (3, 0) -/
theorem circle_equation :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), x = 3 * Real.cos t ∧ y = 4 + 3 * Real.sin t) ↔
  x^2 + (y - 4)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l125_12515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l125_12536

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x - 2) / (x - 3) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a^2 - 2) < 0}

theorem problem_solution :
  (∀ x ∈ U, x ∈ (Set.compl (B (1/2)) ∪ A) ↔ x ≤ 1/2 ∨ x > 2) ∧
  {a : ℝ | B a ⊆ A ∧ B a ≠ A} = Set.Icc (-1 : ℝ) 2 \ Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l125_12536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_distribution_l125_12556

/-- A random variable representing the number of sixes in three rolls of a fair six-sided die -/
def X : ℕ → ℝ := sorry

/-- The probability mass function for X -/
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if k = 0 then 125/216
  else if k = 1 then 75/216
  else if k = 2 then 15/216
  else if k = 3 then 1/216
  else 0

/-- The expected value of X -/
noncomputable def E_X : ℝ := 1/2

/-- The variance of X -/
noncomputable def Var_X : ℝ := 5/12

/-- The standard deviation of X -/
noncomputable def StdDev_X : ℝ := Real.sqrt 15 / 6

theorem dice_distribution :
  (∀ k, pmf_X k ≥ 0) ∧
  (∑' k, pmf_X k = 1) ∧
  E_X = ∑' k, k * pmf_X k ∧
  Var_X = ∑' k, (k - E_X)^2 * pmf_X k ∧
  StdDev_X = Real.sqrt Var_X :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_distribution_l125_12556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_given_conditions_l125_12557

-- Define the variables and constants
variable (x y z w k n : ℝ)

-- Define the relationships
noncomputable def x_prop (k : ℝ) : ℝ → ℝ → ℝ → ℝ := λ y z w => k * y^3 / w^2
noncomputable def y_prop (n : ℝ) : ℝ → ℝ := λ z => n / Real.sqrt z

-- State the theorem
theorem x_value_for_given_conditions :
  -- Conditions
  (∃ k, ∀ y z w, x = x_prop k y z w) →
  (∃ n, ∀ z, y = y_prop n z) →
  (x = 5 ∧ z = 16 ∧ w = 2) →
  -- Conclusion
  (z = 64 ∧ w = 4 → x = 5/32) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_given_conditions_l125_12557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l125_12517

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a n

theorem sequence_properties :
  (a 3 = 4) ∧ (∀ n : ℕ, n > 0 → a n = 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l125_12517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l125_12511

theorem trig_identity (α : ℝ) : 
  Real.sin (π + α) ^ 2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l125_12511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_function_1_not_function_2_is_function_3_not_function_4_l125_12560

-- Define the sets and correspondences
def A1 : Set ℕ := {n : ℕ | n > 0}
def B1 : Set ℕ := A1
def f1 (x : ℕ) : ℕ := Int.natAbs (x - 3)

def A2 : Set ℝ := {x : ℝ | x ≥ 0}
def B2 : Set ℝ := Set.univ
def f2 (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

def A3 : Set ℝ := Set.Icc 1 8
def B3 : Set ℝ := Set.Icc 1 3
noncomputable def f3 (x : ℝ) : ℝ := Real.rpow x (1/3)

def A4 : Set (ℝ × ℝ) := Set.univ
def B4 : Set ℝ := Set.univ
def f4 (p : ℝ × ℝ) : ℝ := p.1 + 3 * p.2

-- State the theorems
theorem not_function_1 : ¬(Function.Injective f1) := by sorry

theorem not_function_2 : ¬(∀ x ∈ A2, Set.Subsingleton (f2 x)) := by sorry

theorem is_function_3 : Function.Injective f3 := by sorry

theorem not_function_4 : ¬(Function.Injective f4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_function_1_not_function_2_is_function_3_not_function_4_l125_12560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l125_12565

/-- Represents the state of the sequence at any given time -/
def SequenceState := List ℝ

/-- The swap and double operation performed by the robot -/
def swapAndDouble (s : SequenceState) : Option SequenceState := sorry

/-- Predicate to check if a swap is possible in the current state -/
def canSwap (s : SequenceState) : Prop := sorry

/-- Helper function to apply swapAndDouble n times -/
def applyNTimes (n : ℕ) (s : SequenceState) : Option SequenceState :=
  match n with
  | 0 => some s
  | n+1 => (applyNTimes n s).bind swapAndDouble

/-- The theorem stating that the process will terminate -/
theorem process_terminates (initial : SequenceState) : 
  ∃ (n : ℕ) (final : SequenceState), 
    (∀ k, k ≥ n → applyNTimes k initial = some final) ∧ 
    ¬(canSwap final) := by
  sorry

#check process_terminates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l125_12565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factorable_quadratics_l125_12551

noncomputable def is_factorable (n : ℤ) : Prop :=
  ∃ a b : ℤ, ∀ x : ℝ, x^2 + x - n = (x - a) * (x - b)

theorem count_factorable_quadratics :
  ∃ S : Finset ℤ,
    S.card = 9 ∧
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 100) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 100 → (n ∈ S ↔ is_factorable n)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factorable_quadratics_l125_12551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l125_12542

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3
  else Real.log x - 2

-- Define what it means for x to be a zero of f
def is_zero (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem f_has_two_zeros : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ is_zero x₁ ∧ is_zero x₂ ∧ ∀ (x : ℝ), is_zero x → (x = x₁ ∨ x = x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l125_12542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_range_l125_12502

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem triangle_side_lengths_range (m : ℝ) : 
  (∀ a b c : ℝ, ∃ s1 s2 s3 : ℝ, 
    s1 = f m a ∧ s2 = f m b ∧ s3 = f m c ∧
    s1 + s2 > s3 ∧ s2 + s3 > s1 ∧ s3 + s1 > s2) ↔ 
  7/5 < m ∧ m < 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_range_l125_12502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_distance_l125_12543

/-- The hyperbola equation xy = 1 -/
def hyperbola (x y : ℝ) : Prop := x * y = 1

/-- The circle with diameter AB -/
def circle_diameter (A B X : ℝ × ℝ) : Prop :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (X.1 - midpoint.1)^2 + (X.2 - midpoint.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem hyperbola_circle_intersection_distance :
  let A : ℝ × ℝ := (4, 1/4)
  let B : ℝ × ℝ := (-5, -1/5)
  ∀ X Y : ℝ × ℝ,
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  hyperbola X.1 X.2 →
  hyperbola Y.1 Y.2 →
  circle_diameter A B X →
  circle_diameter A B Y →
  X ≠ A →
  X ≠ B →
  Y ≠ A →
  Y ≠ B →
  distance X Y = Real.sqrt (401/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_distance_l125_12543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_is_2_sqrt_2_l125_12567

/-- The length of the generatrix of a cone with base radius √2 and lateral surface forming a semicircle when unfolded -/
noncomputable def cone_generatrix_length : ℝ := 2 * Real.sqrt 2

/-- The base radius of the cone -/
noncomputable def base_radius : ℝ := Real.sqrt 2

/-- Theorem stating that the length of the generatrix of the cone is 2√2 -/
theorem cone_generatrix_length_is_2_sqrt_2 
  (h1 : base_radius = Real.sqrt 2) 
  (h2 : ∃ (r : ℝ), π * r = 2 * π * base_radius) : 
  cone_generatrix_length = 2 * Real.sqrt 2 := by
  sorry

#check cone_generatrix_length_is_2_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_is_2_sqrt_2_l125_12567
