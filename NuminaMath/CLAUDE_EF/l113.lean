import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_time_completion_time_l113_11326

/-- The number of days it takes for two workers to complete a job when one works part-time -/
noncomputable def days_to_complete (full_time_together full_time_a half_time_b : ℝ) : ℝ :=
  1 / (1 / full_time_a + (1 / full_time_together - 1 / full_time_a) / 2)

/-- 
Given:
- Two workers a and b can complete a work in 12 days together
- Worker a alone can complete the work in 20 days
- Worker b works only half a day daily

Prove: a and b together will complete the work in 15 days
-/
theorem part_time_completion_time :
  days_to_complete 12 20 (1/2) = 15 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_time_completion_time_l113_11326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l113_11352

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 6)

/-- The transformed function g(x) -/
def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3)

/-- Theorem stating that g(x) is the result of transforming f(x) -/
theorem transform_f_to_g :
  ∀ x : ℝ, g x = f (x / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l113_11352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_11_l113_11333

noncomputable section

-- Define the square ABCD
def square_area : ℝ := 25

-- Define the rhombus PQCD
def rhombus_area : ℝ := 20

-- Define the side length of the square
noncomputable def side_length : ℝ := Real.sqrt square_area

-- Define the height of the rhombus
noncomputable def rhombus_height : ℝ := rhombus_area / side_length

-- Define the width of rectangle ABZX
noncomputable def rect_width : ℝ := side_length - rhombus_height

-- Define the area of rectangle ABZX
noncomputable def rect_area : ℝ := side_length * rect_width

-- Define the length of PX in triangle PXD
noncomputable def px_length : ℝ := Real.sqrt (side_length^2 - rhombus_height^2)

-- Define the area of triangle PXD
noncomputable def triangle_area : ℝ := (1/2) * px_length * rhombus_height

-- Theorem to prove
theorem shaded_area_is_11 :
  rect_area + triangle_area = 11 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_11_l113_11333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_bound_l113_11386

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  let binary := n.digits 2
  (binary.map (λ b => if b = 1 then 1 else 0)).foldl (λ acc d => 3 * acc + d) 0

-- State the theorem
theorem a_2023_bound : a 2023 ≤ 100000 := by
  -- Proof goes here
  sorry

#eval a 2023  -- This will evaluate a(2023) for verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_bound_l113_11386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_minimum_integer_a_l113_11318

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Theorem for part (1)
theorem monotonic_decrease_interval (a : ℝ) :
  (f a 1 = 0) →
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x < y → f a y < f a x) :=
by sorry

-- Theorem for part (2)
theorem minimum_integer_a :
  (∃ a : ℤ, ∀ x : ℝ, x > 0 → f (↑a) x ≤ ↑a * x - 1) ∧
  (∀ a : ℤ, a < 2 → ∃ x : ℝ, x > 0 ∧ f (↑a) x > ↑a * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_minimum_integer_a_l113_11318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l113_11369

theorem angle_sum_problem (α β : Real) : 
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 →
  Real.sin α = Real.sqrt 10 / 10 →
  Real.cos β = 2 * Real.sqrt 5 / 5 →
  α + β = π / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l113_11369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l113_11328

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π/2) : 
  Real.tan (2*α) = -8*Real.sqrt 3/47 ∧ β = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l113_11328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_8y_l113_11302

/-- The rectangular region --/
noncomputable def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2030}

/-- The region where x > 8y --/
noncomputable def Region : Set (ℝ × ℝ) :=
  {p ∈ Rectangle | p.1 > 8 * p.2}

/-- The measure of the rectangular region --/
def rectangleArea : ℚ := 2020 * 2030

/-- The measure of the region where x > 8y --/
def regionArea : ℚ := (1 / 2) * 2020 * (2020 / 8)

theorem probability_x_gt_8y :
  (regionArea / rectangleArea : ℚ) = 255025 / 4100600 := by
  sorry

#eval (regionArea / rectangleArea : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_8y_l113_11302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1387_fixed_point_l113_11311

-- Define the function f
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Define the composition of f with itself 1387 times
noncomputable def f_1387 (a b c d : ℝ) : ℝ → ℝ := (f a b c d)^[1387]

-- Main theorem
theorem f_1387_fixed_point
  (a b c d : ℝ)
  (h_cd : c ≠ 0 ∨ d ≠ 0)
  (h_not_fixed : ∀ x, f a b c d x ≠ x)
  (h_exists_fixed : ∃ a₀, f_1387 a b c d a₀ = a₀) :
  ∀ x, f_1387 a b c d x = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1387_fixed_point_l113_11311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_rates_l113_11363

-- Define the clock structure
structure Clock where
  total_degrees : ℚ := 360
  minutes_per_hour : ℕ := 60
  hours_per_rotation : ℕ := 12

-- Define the rotation rates
noncomputable def minute_hand_rotation_rate (c : Clock) : ℚ :=
  c.total_degrees / c.minutes_per_hour

noncomputable def hour_hand_rotation_rate (c : Clock) : ℚ :=
  (c.total_degrees / c.hours_per_rotation) / c.minutes_per_hour

-- Theorem statement
theorem clock_rotation_rates (c : Clock) :
  minute_hand_rotation_rate c = 6 ∧ hour_hand_rotation_rate c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_rates_l113_11363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l113_11385

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a n ≠ 0 ∧ 2 * a (n + 1) - a n = 0

theorem sixth_term_value (a : ℕ → ℚ) (h : geometric_sequence a) : a 6 = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l113_11385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l113_11305

/-- Circle O in the Cartesian coordinate system -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle M in the Cartesian coordinate system -/
def circle_M (x y a : ℝ) : Prop := (x + a + 1)^2 + (y - 2*a)^2 = 1

/-- The existence of points P on circle O and Q on circle M with ∠OQP = 30° -/
def exists_points (a : ℝ) : Prop :=
  ∃ (px py qx qy : ℝ),
    circle_O px py ∧ circle_M qx qy a ∧
    ∃ (angle : ℝ), angle = 30 * Real.pi / 180 ∧
    (qx - px)^2 + (qy - py)^2 = 1 + ((qx + a + 1)^2 + (qy - 2*a)^2) - 2 * Real.cos angle * Real.sqrt (1 * ((qx + a + 1)^2 + (qy - 2*a)^2))

theorem circle_tangency (a : ℝ) : exists_points a → -1 ≤ a ∧ a ≤ 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l113_11305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l113_11314

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

/-- Theorem: The distance between points A(2, π/6) and B(4, 5π/6) in polar coordinates is 2√7 -/
theorem distance_between_polar_points :
  polar_distance 2 (π/6) 4 (5*π/6) = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l113_11314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_domain_of_x_l113_11375

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3)

-- State the theorem
theorem range_of_x : 
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 0 :=
by
  sorry

-- State the corollary about the domain of x
theorem domain_of_x :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≥ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_domain_of_x_l113_11375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l113_11387

noncomputable def expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / (3^2 - 7)|

theorem expression_approximation :
  abs (expression - 634.125009794) < 1e-9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l113_11387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l113_11361

/-- Represents a hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_imaginary_axis : b = 1
  h_focal_length : Real.sqrt (a^2 + b^2) = Real.sqrt 3

/-- The equation of the asymptote of a hyperbola -/
noncomputable def asymptote_equation (h : Hyperbola) : ℝ → ℝ :=
  fun x => (h.b / h.a) * x

/-- Theorem stating that the asymptote equation has the form y = ±(√2/2)x -/
theorem hyperbola_asymptote (h : Hyperbola) :
    ∃ k : ℝ, k = Real.sqrt 2 / 2 ∧ asymptote_equation h = fun x => k * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l113_11361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l113_11389

/-- α(n) represents the number of ways to express n as a sum of 1s and 2s, where order matters -/
def α : ℕ+ → ℕ :=
  sorry

/-- β(n) represents the number of ways to express n as a sum of integers greater than 1, where order matters -/
def β : ℕ+ → ℕ :=
  sorry

/-- For all positive integers n, α(n) equals β(n+2) -/
theorem alpha_beta_equality (n : ℕ+) : α n = β (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l113_11389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l113_11391

/-- The foot of the perpendicular from the origin to a plane -/
def foot : ℝ × ℝ × ℝ := (12, -4, 3)

/-- The equation of the plane -/
def plane_equation (x y z : ℝ) : ℝ := 12 * x - 4 * y + 3 * z - 169

theorem plane_equation_correct :
  let (a, b, c) := foot
  (plane_equation a b c = 0) ∧
  (∃ A B C D : ℤ, ∀ x y z, plane_equation x y z = A * x + B * y + C * z + D) ∧
  (∃ A : ℤ, A > 0 ∧ ∀ x y z, plane_equation x y z = A * x + plane_equation 0 y z) ∧
  (∃ A B C D : ℤ, Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1
    ∧ ∀ x y z, plane_equation x y z = A * x + B * y + C * z + D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l113_11391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_epsilon_existence_l113_11346

open Real

/-- Fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem epsilon_existence (n : ℕ+) :
  ∃ ε : ℝ, 0 < ε ∧ ε < (1 : ℝ) / 2014 ∧
    ∀ (a : Fin n → ℝ), (∀ i, 0 < a i) →
      ∃ u : ℝ, u > 0 ∧ ∀ i, ε < frac (u * a i) ∧ frac (u * a i) < (1 : ℝ) / 2014 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_epsilon_existence_l113_11346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l113_11343

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (-x) - Real.exp x

-- Define what it means for f to be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem solution_set_of_inequality (a : ℝ) :
  is_odd_function (f a) →
  {x : ℝ | f a (x - 1) < Real.exp 1 - 1 / Real.exp 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l113_11343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_survey_l113_11312

theorem household_survey (neither : ℕ) (only_R : ℕ) (both : ℕ)
  (h1 : neither = 80)
  (h2 : only_R = 60)
  (h3 : both = 40)
  (h4 : ∀ x : ℕ, x = both → 3 * x = only_B) :
  neither + only_R + only_B + both = 300 :=
by
  let only_B := 3 * both
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_survey_l113_11312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_definition_g_range_g_domain_l113_11330

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2*x - 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / (x + 2)

-- Theorem for part 1
theorem f_definition (x : ℝ) : f (x + 2) = 2*x + 3 := by sorry

-- Theorems for part 2
theorem g_range : Set.range g = {y : ℝ | y ≠ 2} := by sorry

theorem g_domain : {x : ℝ | x ≠ -2} = {x | g x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_definition_g_range_g_domain_l113_11330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_a_is_neg_two_l113_11388

/-- A parabola with vertex (3/2, -25/4) following the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions for our specific parabola -/
def ParabolaConditions (p : Parabola) : Prop :=
  p.a < 0 ∧ 
  (∃ n : ℤ, p.a * p.b * p.c = n) ∧
  ∀ x y : ℝ, y = p.a * x^2 + p.b * x + p.c ↔ 
    y = p.a * (x - 3/2)^2 - 25/4

/-- The theorem stating that the largest possible value of a is -2 -/
theorem largest_a_is_neg_two (p : Parabola) (h : ParabolaConditions p) :
  ∀ q : Parabola, ParabolaConditions q → q.a ≤ -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_a_is_neg_two_l113_11388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l113_11345

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 + Complex.I) / (4 + 3 * Complex.I)
  Complex.im z = 1 / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l113_11345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factors_product_l113_11373

def n : ℕ := 176543

theorem smallest_prime_factors_product (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r, Nat.Prime r → r ∣ n → r ≥ p) ∧
  (∀ r, Nat.Prime r → r ∣ n → r ≠ p → r ≥ q) →
  p * q = 34 := by
  sorry

#check smallest_prime_factors_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factors_product_l113_11373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_hypotenuse_l113_11381

theorem right_triangle_rotation_hypotenuse :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    (1/3) * Real.pi * b^2 * a = 900 * Real.pi →
    (1/3) * Real.pi * a^2 * b = 1800 * Real.pi →
    Real.sqrt (a^2 + b^2) = Real.sqrt 605 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_hypotenuse_l113_11381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l113_11392

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => sequenceA n / (1 + 2 * sequenceA n)

theorem sequence_formula (n : ℕ) (h : n > 0) : sequenceA n = 1 / (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l113_11392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_a_value_l113_11398

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := (x + 1) / (x - 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := -2 / ((x - 1)^2)

-- Define the slope of the tangent line at x = 3
noncomputable def tangent_slope : ℝ := curve_derivative 3

-- Define the perpendicular line
def perpendicular_line (a : ℝ) (x y : ℝ) : Prop := a * x + y + 1 = 0

-- Theorem statement
theorem perpendicular_line_a_value :
  ∃ (a : ℝ), perpendicular_line a 3 2 ∧ a * tangent_slope = -1 ∧ a = -2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_a_value_l113_11398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_theorem_l113_11360

/-- Ferris wheel model -/
structure FerrisWheel where
  radius : ℝ
  centerHeight : ℝ
  period : ℝ

/-- Height function for a point on the Ferris wheel -/
noncomputable def heightFunction (fw : FerrisWheel) (t : ℝ) : ℝ :=
  -fw.radius * Real.cos (2 * Real.pi / fw.period * t) + fw.centerHeight

/-- Time when the height reaches a specific value -/
noncomputable def timeAtHeight (fw : FerrisWheel) (h : ℝ) : ℝ :=
  fw.period - fw.period / Real.pi * Real.arccos ((fw.centerHeight - h) / fw.radius)

/-- Theorem about the Ferris wheel model -/
theorem ferris_wheel_theorem (fw : FerrisWheel) 
    (h_radius : fw.radius = 50)
    (h_center : fw.centerHeight = 50)
    (h_period : fw.period = 3) :
  (∀ t, heightFunction fw t = -50 * Real.cos (2 * Real.pi / 3 * t) + 50) ∧
  timeAtHeight fw 85 = 3 - 3 / Real.pi * Real.arccos (-7/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_theorem_l113_11360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alicia_average_speed_l113_11337

/-- Calculates the average speed given two segments of a journey -/
noncomputable def average_speed (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

/-- Theorem: Alicia's average speed for the entire journey -/
theorem alicia_average_speed :
  let speed := average_speed 320 6 420 7
  ∀ ε > 0, |speed - 56.92| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alicia_average_speed_l113_11337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_path_tiles_l113_11376

/-- Represents a rectangular floor -/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Represents a tile on the floor -/
structure Tile where
  x : ℕ
  y : ℕ

/-- Represents the path of the bug -/
def bugPath (floor : RectangularFloor) : Set Tile :=
  {tile : Tile | tile.x ≤ floor.length ∧ tile.y = floor.width / 2}

/-- The number of tiles visited by the bug -/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.length

/-- The main theorem -/
theorem bug_path_tiles (floor : RectangularFloor) 
  (h1 : floor.width = 15) 
  (h2 : floor.length = 20) : 
  tilesVisited floor = 20 := by
  rw [tilesVisited]
  exact h2

#eval tilesVisited ⟨15, 20⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_path_tiles_l113_11376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l113_11367

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k + 4) / x

theorem inverse_proportion_k_range 
  (k : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : k ≠ -4)
  (h2 : x₁ < 0)
  (h3 : 0 < x₂)
  (h4 : y₁ > y₂)
  (h5 : inverse_proportion k x₁ = y₁)
  (h6 : inverse_proportion k x₂ = y₂) :
  k < -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l113_11367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_power_l113_11340

theorem consecutive_product_power (n m k : ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) :
  (n - 1) * n * (n + 1) = m ^ k → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_power_l113_11340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d₁_d₂_relation_l113_11393

/-- Definition of d₁ for a three-term sequence (a, b, c) -/
noncomputable def d₁ (a b c : ℝ) : ℝ := |b - (a + c) / 2|

/-- Definition of d₂ for a three-term sequence (a, b, c) -/
noncomputable def d₂ (a b c : ℝ) : ℝ := |c - (2 * b - a)|

/-- Theorem stating that 2d₁ = d₂ for any three-term sequence -/
theorem d₁_d₂_relation (a b c : ℝ) : 2 * d₁ a b c = d₂ a b c := by
  sorry

#check d₁_d₂_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d₁_d₂_relation_l113_11393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_division_theorem_l113_11313

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a division of the chessboard -/
structure ChessboardDivision where
  rectangles : List Rectangle
  white_squares : List Nat

/-- Checks if a rectangle has an equal number of white and black squares -/
def has_equal_colors (r : Rectangle) : Bool :=
  (r.width * r.height) % 2 = 0

/-- Checks if the sequence of white squares is strictly increasing -/
def is_strictly_increasing (l : List Nat) : Bool :=
  l.zip l.tail |>.all (fun (a, b) => a < b)

/-- Checks if the division is valid according to the problem conditions -/
def is_valid_division (d : ChessboardDivision) : Bool :=
  d.rectangles.all has_equal_colors &&
  is_strictly_increasing d.white_squares &&
  d.rectangles.foldl (fun sum r => sum + r.width * r.height) 0 = 64

/-- The theorem to be proved -/
theorem chessboard_division_theorem :
  ∃ (d : ChessboardDivision),
    is_valid_division d = true ∧
    d.rectangles.length = 6 ∧
    d.white_squares = [2, 4, 8, 12, 18, 16] ∧
    (∀ (d' : ChessboardDivision), is_valid_division d' = true → d'.rectangles.length ≤ 6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_division_theorem_l113_11313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l113_11364

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 0 + a (n - 1)) / 2

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : sum_of_arithmetic_sequence a 5 = 15) 
  (h_a2 : a 1 = 5) : 
  a 1 - a 0 = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l113_11364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l113_11324

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal time interest : ℝ) 
  (h_principal : principal = 500)
  (h_time : time = 4)
  (h_interest : interest = 100) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 5 := by
  use 5
  constructor
  · rw [simple_interest, h_principal, h_time, h_interest]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l113_11324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l113_11357

def initial_numbers : Set ℚ := {1, 1/2, 1/3, 1/4, 1/5, 1/6}

def can_obtain (S : Set ℚ) (x : ℚ) : Prop :=
  x ∈ S ∨ ∃ a b, a ∈ S ∧ b ∈ S ∧ (x = a + b ∨ x = a * b)

def closed_set (S : Set ℚ) : Set ℚ :=
  {x | ∃ n : ℕ, (λ T => {y | can_obtain T y})^[n] S x}

theorem board_numbers (x : ℚ) :
  x ∈ closed_set initial_numbers ↔
    (x = 1/60 ∨ x = 2011/375) ∧ x ≠ 1/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l113_11357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_turning_points_l113_11380

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then (1/a) * x
  else if a < x ∧ x ≤ 1 then (1/(1-a)) * (1-x)
  else 0

/-- Definition of a turning point -/
def is_turning_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (f x) = x ∧ f x ≠ x

/-- Main theorem statement -/
theorem f_has_two_turning_points (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ (x₁ x₂ : ℝ), 
    x₁ = a / (1 - a^2 + a) ∧
    x₂ = 1 / (1 - a^2 + a) ∧
    is_turning_point (f a) x₁ ∧
    is_turning_point (f a) x₂ ∧
    ∀ (x : ℝ), is_turning_point (f a) x → (x = x₁ ∨ x = x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_turning_points_l113_11380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_sum_l113_11356

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the parabola
structure Parabola where
  a : ℝ
  b : ℝ

-- Define the theorem
theorem parabola_roots_sum (ABCD : Square) (p : Parabola) : 
  -- Square properties
  (ABCD.A.1 = 0 ∧ ABCD.B.1 = 0) →  -- AB on y-axis
  (ABCD.A.2 = ABCD.D.2 ∧ ABCD.B.2 = ABCD.C.2) →  -- Horizontal sides parallel
  (ABCD.A.1 = ABCD.D.1 ∧ ABCD.B.1 = ABCD.C.1) →  -- Vertical sides parallel
  -- Parabola properties
  ((λ x ↦ (1/5) * x^2 + p.a * x + p.b) ABCD.B.1 = ABCD.B.2) →  -- B on parabola
  ((λ x ↦ (1/5) * x^2 + p.a * x + p.b) ABCD.C.1 = ABCD.C.2) →  -- C on parabola
  (∃ E : ℝ × ℝ, E.1 = (ABCD.A.1 + ABCD.D.1) / 2 ∧ 
   E.2 = (λ x ↦ (1/5) * x^2 + p.a * x + p.b) E.1) →  -- Vertex E on AD
  -- Conclusion
  (-(p.a * 5)) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_sum_l113_11356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l113_11394

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the point F
def F : ℝ × ℝ := (2, 0)

-- Define the line x = 4
def line_C (x : ℝ) : Prop := x = 4

-- Define an equilateral triangle
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  let d := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = d B C ∧ d B C = d C A

-- Main theorem
theorem equilateral_triangle_area 
  (A B C : ℝ × ℝ) 
  (hA : ellipse A.1 A.2) 
  (hB : ellipse B.1 B.2) 
  (hC : line_C C.1) 
  (hF : ∃ (m : ℝ), A.1 = m * A.2 + F.1 ∧ B.1 = m * B.2 + F.1) 
  (hEq : is_equilateral A B C) : 
  Real.sqrt 3 / 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 72 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l113_11394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l113_11335

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 3

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (3*x - 2) / (2*x - 3)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc 1 (3/2) ∪ Set.Ioc (3/2) (5/3)

-- Theorem statement
theorem domain_of_g :
  ∀ x : ℝ, x ∈ domain_g ↔ (x ∈ Set.Icc 1 (5/3) ∧ x ≠ 3/2 ∧ (3*x - 2) ∈ domain_f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l113_11335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_value_l113_11334

/-- The function f(x) as defined in the problem -/
noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

/-- The theorem statement -/
theorem exists_x0_implies_a_value (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ a = 3) → a = 1 - Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_value_l113_11334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiao_ri_method_pi_approximation_l113_11315

def tiao_ri (a b c d : ℕ) : ℚ := (b + d) / (a + c)

def is_simpler (q1 q2 : ℚ) : Prop := 
  (q1.num.natAbs + q1.den < q2.num.natAbs + q2.den) ∨
  (q1.num.natAbs + q1.den = q2.num.natAbs + q2.den ∧ q1.num.natAbs < q2.num.natAbs)

theorem tiao_ri_method_pi_approximation :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let step1 := tiao_ri 10 31 15 49
  let step2 := tiao_ri 10 31 5 16
  let step3 := tiao_ri 15 47 5 16
  let step4 := tiao_ri 15 47 20 63
  (initial_lower < Real.pi) ∧ 
  (Real.pi < initial_upper) ∧
  (∀ q, initial_lower < q ∧ q < step1 → is_simpler step1 q) ∧
  (∀ q, step1 < q ∧ q < step2 → is_simpler step2 q) ∧
  (∀ q, step2 < q ∧ q < step3 → is_simpler step3 q) ∧
  (∀ q, step3 < q ∧ q < step4 → is_simpler step4 q) →
  step4 = 22 / 7 ∧ Real.pi < 22 / 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiao_ri_method_pi_approximation_l113_11315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_four_boxes_problem_solution_l113_11304

def box_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

theorem total_volume_of_four_boxes (edge_length : ℝ) (h : edge_length = 5) : 
  4 * box_volume edge_length = 500 := by
  -- Unfold the definition of box_volume
  unfold box_volume
  -- Substitute the edge length
  rw [h]
  -- Simplify the expression
  norm_num

-- The main theorem
theorem problem_solution : 
  ∃ (v : ℝ), v = 500 ∧ v = 4 * box_volume 5 := by
  -- Use 500 as the witness
  use 500
  -- Split the goal into two parts
  constructor
  · -- First part: v = 500
    rfl
  · -- Second part: 500 = 4 * box_volume 5
    exact Eq.symm (total_volume_of_four_boxes 5 rfl)

-- Proof completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_four_boxes_problem_solution_l113_11304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_good_students_probability_l113_11325

def total_group_size : ℕ := 12
def three_good_students : ℕ := 5
def selected_people : ℕ := 6

def probability_three_good_selected (n k m : ℕ) : ℚ :=
  (Nat.choose m 3 * Nat.choose (n - m) (k - 3)) / Nat.choose n k

theorem three_good_students_probability :
  probability_three_good_selected total_group_size selected_people three_good_students =
  (Nat.choose three_good_students 3 * Nat.choose (total_group_size - three_good_students) (selected_people - 3)) /
  Nat.choose total_group_size selected_people :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_good_students_probability_l113_11325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_two_sqrt_three_thirds_l113_11374

/-- A regular octahedron with an inscribed cube. -/
structure OctahedronWithCube where
  /-- The side length of the square base of the octahedron. -/
  a : ℝ
  /-- The half side length of the inscribed cube. -/
  b : ℝ
  /-- Assumption that the cube is inscribed in the octahedron such that its vertices lie on the edges. -/
  cube_inscribed : b = a * Real.sqrt 2 / 4

/-- The ratio of the surface area of the octahedron to the surface area of the inscribed cube. -/
noncomputable def surface_area_ratio (oc : OctahedronWithCube) : ℝ :=
  (2 * oc.a^2 * Real.sqrt 3) / (3 * oc.a^2)

/-- Theorem stating that the surface area ratio is 2√3/3. -/
theorem surface_area_ratio_is_two_sqrt_three_thirds (oc : OctahedronWithCube) :
  surface_area_ratio oc = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_is_two_sqrt_three_thirds_l113_11374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_correct_l113_11341

/-- The number of triples of natural numbers whose sum equals 6n -/
def count_triples (n : ℕ) : ℕ :=
  3 * n^2

/-- Theorem stating that count_triples correctly counts the number of triples -/
theorem count_triples_correct (n : ℕ) :
  count_triples n = Finset.card (Finset.filter (fun p : ℕ × ℕ × ℕ => p.1 + p.2.1 + p.2.2 = 6 * n) (Finset.product (Finset.range (6*n+1)) (Finset.product (Finset.range (6*n+1)) (Finset.range (6*n+1))))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_correct_l113_11341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_proof_l113_11342

/-- Given a square divided into n rectangles with no common interior points,
    the maximal number of distinct lines determined by the sides of these rectangles is n + 3 -/
def max_lines_in_divided_square (n : ℕ) : ℕ :=
  n + 3

/-- The maximal number of distinct lines for 2022 rectangles -/
def max_lines_2022 : ℕ :=
  max_lines_in_divided_square 2022

#eval max_lines_2022  -- Should output 2025

/-- Proof of the theorem -/
theorem max_lines_proof (n : ℕ) :
  max_lines_in_divided_square n = n + 3 := by
  -- Unfold the definition of max_lines_in_divided_square
  unfold max_lines_in_divided_square
  -- The equality now holds by reflexivity
  rfl

#check max_lines_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lines_proof_l113_11342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l113_11309

-- Define the function f(x) = x^3 - 12x + 8
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Define the interval [-3, 3]
def I : Set ℝ := Set.Icc (-3) 3

-- Theorem statement
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∀ x ∈ I, m ≤ f x) ∧ 
               (∃ x₁ ∈ I, f x₁ = M) ∧ 
               (∃ x₂ ∈ I, f x₂ = m) ∧ 
               M - m = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l113_11309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l113_11397

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- A point on the parabola -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem parabola_triangle_area :
  ∀ P : ℝ × ℝ,
  point_on_parabola P →
  distance P focus = 4 →
  triangle_area (0, 0) P focus = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l113_11397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_power_four_l113_11332

theorem eleventh_power_four : 11^4 = 14641 := by
  -- Define 11 as 10 + 1
  have h1 : 11 = 10 + 1 := by norm_num
  
  -- State the binomial expansion for (a + b)^4
  have binomial_expansion : ∀ a b : ℕ, (a + b)^4 = 
    a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4 := by
    intros a b
    ring

  -- Apply the binomial expansion and simplify
  calc 11^4 = (10 + 1)^4 := by rw [h1]
       _ = 10^4 + 4*10^3*1 + 6*10^2*1^2 + 4*10*1^3 + 1^4 := by rw [binomial_expansion]
       _ = 10000 + 4000 + 600 + 40 + 1 := by ring
       _ = 14641 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_power_four_l113_11332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l113_11368

/-- Given sequences {a_n} and {b_n}, prove that a > 2 -/
theorem sequence_range (a : ℝ) (a_n b_n : ℕ → ℝ) : 
  (∀ n : ℕ, n > 0 → b_n n = (a_n n - 2) / (a_n n - 1)) → 
  (∀ n : ℕ, n > 0 → b_n n = b_n 1 * (2/3)^(n-1)) → 
  (∀ n : ℕ, n > 0 → a_n n > a_n (n+1)) → 
  a_n 1 = a → 
  a > 2 ∧ ∀ x > 2, ∃ y : ℕ, y > 0 ∧ a_n y = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l113_11368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_travel_l113_11382

/-- Represents a traveler on a road -/
structure Traveler where
  position : ℝ
  speed : ℝ

/-- The scenario of two travelers on perpendicular roads -/
structure TravelScenario where
  traveler1 : Traveler
  traveler2 : Traveler
  initialDistance1 : ℝ  -- Distance of traveler1 when traveler2 crosses intersection
  initialDistance2 : ℝ  -- Distance of traveler2 when traveler1 crosses intersection

/-- Calculate the distance between two points on perpendicular roads -/
noncomputable def distanceBetween (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- The main theorem -/
theorem distance_after_travel (scenario : TravelScenario) :
  scenario.initialDistance1 = 4200 →
  scenario.initialDistance2 = 3500 →
  distanceBetween 8400 3500 = 9100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_travel_l113_11382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Ω_eq_two_pi_l113_11301

/-- The surface defining the lower bound of the body Ω -/
noncomputable def lower_surface (x y : ℝ) : ℝ := (9/2) * Real.sqrt (x^2 + y^2)

/-- The surface defining the upper bound of the body Ω -/
noncomputable def upper_surface (x y : ℝ) : ℝ := (11/2) - x^2 - y^2

/-- The body Ω bounded by the two surfaces -/
def Ω : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ (x y z : ℝ), p = (x, y, z) ∧ 
       lower_surface x y ≤ z ∧ z ≤ upper_surface x y}

/-- The volume of the body Ω -/
noncomputable def volume_Ω : ℝ := (MeasureTheory.volume Ω).toReal

theorem volume_Ω_eq_two_pi : volume_Ω = 2 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Ω_eq_two_pi_l113_11301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_true_propositions_l113_11396

/-- A proposition about geometric solids -/
inductive GeometricProposition
| PrismDefinition
| PyramidDefinition
| FrustumDefinition

/-- Defines what a prism is -/
def is_prism (solid : Type) : Prop := sorry

/-- Defines what a pyramid is -/
def is_pyramid (solid : Type) : Prop := sorry

/-- Defines what a frustum is -/
def is_frustum (solid : Type) : Prop := sorry

/-- Determines if a given geometric proposition is true -/
def is_true_proposition (prop : GeometricProposition) : Prop :=
  match prop with
  | GeometricProposition.PrismDefinition => 
      ∃ (solid : Type), ∃ (has_parallel_faces : solid → Prop) (other_faces_parallelograms : solid → Prop),
        ∀ s : solid, has_parallel_faces s ∧ other_faces_parallelograms s → is_prism solid
  | GeometricProposition.PyramidDefinition => 
      ∃ (solid : Type), ∃ (has_polygonal_face : solid → Prop) (other_faces_triangles : solid → Prop),
        ∀ s : solid, has_polygonal_face s ∧ other_faces_triangles s → is_pyramid solid
  | GeometricProposition.FrustumDefinition => 
      ∃ (solid : Type), ∃ (is_cut_pyramid : solid → Prop) (cut_parallel_to_base : solid → Prop),
        ∀ s : solid, is_cut_pyramid s ∧ cut_parallel_to_base s → is_frustum solid

/-- The main theorem stating that none of the given propositions are true -/
theorem no_true_propositions : 
  ∀ prop : GeometricProposition, ¬(is_true_proposition prop) :=
by
  intro prop
  cases prop
  all_goals {
    intro h
    sorry  -- The actual proof would go here
  }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_true_propositions_l113_11396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_ten_l113_11378

/-- Definition of series A -/
noncomputable def A : ℝ := ∑' n, if (n % 2 = 1 ∧ n % 3 ≠ 0) then ((-1)^((n-1)/2) / n^2) else 0

/-- Definition of series B -/
noncomputable def B : ℝ := ∑' n, if (n % 2 = 1 ∧ n % 3 = 0) then ((-1)^((n-3)/6) / n^2) else 0

/-- Theorem stating that A/B = 10 -/
theorem A_div_B_eq_ten : A / B = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_ten_l113_11378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_numbers_unique_l113_11362

def is_valid_distribution (x y z : ℕ) : Prop :=
  x ∈ Finset.range 11 ∧ 
  y ∈ Finset.range 11 ∧ 
  z ∈ Finset.range 11 ∧ 
  x < y ∧ y < z ∧
  x + y + z = 17 ∧
  ∃ (a b c d e f g h i : ℕ), 
    Finset.toSet {a, b, c} = Finset.toSet {x, y, z} ∧
    Finset.toSet {d, e, f} = Finset.toSet {x, y, z} ∧
    Finset.toSet {g, h, i} = Finset.toSet {x, y, z} ∧
    a + d + g = 13 ∧
    b + e + h = 15 ∧
    c + f + i = 23

theorem card_numbers_unique :
  ∀ x y z : ℕ, is_valid_distribution x y z → x = 3 ∧ y = 5 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_numbers_unique_l113_11362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_bounds_l113_11322

/-- The power function for three electric generators -/
noncomputable def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  (x₁^2 + x₂*x₃).sqrt + (x₂^2 + x₁*x₃).sqrt + (x₃^2 + x₁*x₂).sqrt

/-- Theorem stating the bounds of the power function -/
theorem power_function_bounds (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h₄ : x₁ + x₂ + x₃ ≤ 2) : 
  0 ≤ f x₁ x₂ x₃ ∧ f x₁ x₂ x₃ ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_bounds_l113_11322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PD_l113_11370

-- Define the square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)

-- Define distances
noncomputable def u (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
noncomputable def v (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
noncomputable def w (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
noncomputable def distPD (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)

-- State the theorem
theorem max_distance_PD :
  ∀ P : ℝ × ℝ, (u P)^2 + (v P)^2 = 2 * (w P)^2 → distPD P ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PD_l113_11370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l113_11307

theorem smallest_positive_z (x z : ℝ) : 
  Real.sin x = 0 → 
  Real.cos (x + z) = -1/2 → 
  z > 0 → 
  (∀ w, w > 0 ∧ Real.sin x = 0 ∧ Real.cos (x + w) = -1/2 → z ≤ w) → 
  z = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l113_11307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_odd_function_l113_11348

open Real

theorem shifted_sine_odd_function (φ : ℝ) (h1 : |φ| < π/2) : 
  (∀ x, sin (2*(x + π/3) + φ) = -sin (2*(-x + π/3) + φ)) → φ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_odd_function_l113_11348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l113_11347

theorem inequality_proof (b : ℝ) (k : Nat) (h : b ≥ 0) (hk : k ∈ ({2, 3, 4} : Set Nat)) : 
  let n := 2^k - 1
  (Finset.range (n + 1)).sum (λ i => b^(i * k)) ≥ (1 + b^n)^k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l113_11347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_over_two_plus_cos_l113_11399

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (2 + Real.cos x)

-- State the theorem
theorem integral_cos_over_two_plus_cos :
  ∫ x in (0)..(Real.pi / 2), f x = (9 - 4 * Real.sqrt 3) * Real.pi / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_over_two_plus_cos_l113_11399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_l113_11306

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem increasing_f_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (3/2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_range_l113_11306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_per_tankful_is_560_l113_11344

/-- Represents the characteristics of a car's fuel consumption -/
structure CarFuelConsumption where
  cityMilesPerTankful : ℝ
  cityMilesPerGallon : ℝ
  highwayMilesPerGallonDifference : ℝ

/-- Calculates the miles per tankful of gasoline on the highway -/
noncomputable def highwayMilesPerTankful (car : CarFuelConsumption) : ℝ :=
  let tankSize := car.cityMilesPerTankful / car.cityMilesPerGallon
  let highwayMilesPerGallon := car.cityMilesPerGallon + car.highwayMilesPerGallonDifference
  highwayMilesPerGallon * tankSize

/-- Theorem stating that under given conditions, the car travels 560 miles per tankful on the highway -/
theorem highway_miles_per_tankful_is_560 (car : CarFuelConsumption)
    (h1 : car.cityMilesPerTankful = 336)
    (h2 : car.highwayMilesPerGallonDifference = 6)
    (h3 : car.cityMilesPerGallon = 9) :
    highwayMilesPerTankful car = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_per_tankful_is_560_l113_11344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_12_value_l113_11317

def a : ℕ → ℚ
  | 0 => 1  -- Define a value for 0 to cover all natural numbers
  | 1 => 1
  | (n+2) => a (n+1) / (2 * a (n+1) + 1)

theorem a_12_value : a 12 = 1 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_12_value_l113_11317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_of_A_l113_11365

def A : Set ℕ := {1, 2}

theorem power_set_of_A :
  𝒫 A = {∅, {1}, {2}, {1, 2}} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_of_A_l113_11365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_minus_two_greater_than_half_l113_11320

theorem sqrt_seven_minus_two_greater_than_half : Real.sqrt 7 - 2 > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_minus_two_greater_than_half_l113_11320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_increases_production_by_80_percent_l113_11371

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears_per_week : ℝ
  hours_per_week : ℝ

/-- Calculates the percentage increase in bear production when working with an assistant --/
noncomputable def assistant_production_increase (prod : BearProduction) : ℝ :=
  let assistant_rate := 2 * (prod.bears_per_week / prod.hours_per_week)
  let assistant_hours := 0.9 * prod.hours_per_week
  let assistant_bears := assistant_rate * assistant_hours
  (assistant_bears - prod.bears_per_week) / prod.bears_per_week * 100

/-- Theorem stating that the percentage increase in bear production with an assistant is 80% --/
theorem assistant_increases_production_by_80_percent (prod : BearProduction) :
  assistant_production_increase prod = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_increases_production_by_80_percent_l113_11371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_12_equals_1705_l113_11327

def T : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 4 => T (n + 3) + T (n + 2) + T (n + 1)

theorem T_12_equals_1705 : T 12 = 1705 := by
  -- Compute T for values up to 12
  have h4 : T 4 = 13 := by rfl
  have h5 : T 5 = 24 := by rfl
  have h6 : T 6 = 44 := by rfl
  have h7 : T 7 = 81 := by rfl
  have h8 : T 8 = 149 := by rfl
  have h9 : T 9 = 274 := by rfl
  have h10 : T 10 = 504 := by rfl
  have h11 : T 11 = 927 := by rfl
  have h12 : T 12 = 1705 := by rfl
  
  -- Use the final computation
  exact h12

#eval T 12  -- This will evaluate T 12 and show the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_12_equals_1705_l113_11327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_domain_or_range_condition_l113_11308

noncomputable def f (a x : ℝ) : ℝ := Real.log ((a^2 - 1) * x^2 + (a - 1) * x + 1)

def domain_is_R (a : ℝ) : Prop := ∀ x, (a^2 - 1) * x^2 + (a - 1) * x + 1 > 0

def range_is_R (a : ℝ) : Prop := ∀ y, ∃ x, f a x = y

theorem domain_condition (a : ℝ) : domain_is_R a → a ∈ Set.Ioi (-Real.pi) ∪ Set.Icc 1 Real.pi := by
  sorry

theorem domain_or_range_condition (a : ℝ) : 
  (domain_is_R a ∨ range_is_R a) ∧ ¬(domain_is_R a ∧ range_is_R a) → 
  a ∈ Set.Iic (-1) ∪ Set.Icc 1 Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_domain_or_range_condition_l113_11308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_inequality_l113_11303

theorem max_k_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  ∃ (k : ℝ), k = Real.sqrt (9 + 6 * Real.sqrt 3) ∧
  (∀ (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0),
    a^3 + b^3 + c^3 - 3*a*b*c ≥ k * abs ((a-b)*(b-c)*(c-a))) ∧
  (∀ (k' : ℝ), (∀ (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0),
    a^3 + b^3 + c^3 - 3*a*b*c ≥ k' * abs ((a-b)*(b-c)*(c-a))) → k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_inequality_l113_11303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l113_11366

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / (x + 1)

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x - a) / ((x + 1)^2)

theorem extreme_value_at_one (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l113_11366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_cos_fourth_l113_11379

theorem cos_squared_plus_cos_fourth (θ : ℝ) :
  (Real.sin θ) ^ 2 + Real.sin θ = 1 → (Real.cos θ) ^ 2 + (Real.cos θ) ^ 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_plus_cos_fourth_l113_11379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l113_11319

noncomputable def fuel_consumption (x k : ℝ) : ℝ := (1/5) * (x - k + 4500/x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ := (100/x) * fuel_consumption x k

theorem car_fuel_consumption 
  (x k : ℝ) 
  (h1 : 60 ≤ x ∧ x ≤ 120) 
  (h2 : 60 ≤ k ∧ k ≤ 100) 
  (h3 : fuel_consumption 120 k = 11.5) :
  (∀ x', 60 ≤ x' ∧ x' ≤ 100 ↔ fuel_consumption x' k ≤ 9) ∧
  (75 ≤ k ∧ k < 100 → 
    (∃ x_min, fuel_consumption_100km x_min k = 20 - k^2/900 ∧ 
    ∀ y, 60 ≤ y ∧ y ≤ 120 → fuel_consumption_100km y k ≥ fuel_consumption_100km x_min k)) ∧
  (60 ≤ k ∧ k < 75 → 
    fuel_consumption_100km 120 k = 105/4 - k/6 ∧
    ∀ y, 60 ≤ y ∧ y ≤ 120 → fuel_consumption_100km y k ≥ fuel_consumption_100km 120 k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l113_11319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polygon_diagonals_l113_11390

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3
  vertices : Finset (ℕ × ℕ)
  vertex_count : vertices.card = n

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 6

/-- A convex 30-sided polygon containing an inner regular hexagon -/
structure SpecialPolygon where
  outer : ConvexPolygon 30
  inner : RegularHexagon
  shared_vertices : inner.vertices ⊆ outer.vertices

/-- The number of diagonals in a polygon -/
def num_diagonals (p : SpecialPolygon) : ℕ := sorry

/-- Theorem: The number of diagonals in the special polygon is 396 -/
theorem special_polygon_diagonals (p : SpecialPolygon) : num_diagonals p = 396 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polygon_diagonals_l113_11390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_lost_theorem_l113_11331

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the number of games played by each girl --/
def games_played (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- The total number of games played --/
def total_games : ℕ := 19

/-- Represents the games Anya lost --/
def anya_lost_games : List ℕ := [4, 8, 12, 16]

/-- Main theorem: Given the conditions, Anya lost games 4, 8, 12, and 16 --/
theorem anya_lost_theorem :
  (∀ g : Girl, games_played g ≤ total_games) ∧
  (games_played Girl.Anya = 4) ∧
  (∀ i j : ℕ, i ∈ anya_lost_games → j ∈ anya_lost_games → i ≠ j → (i : Int) - (j : Int) ≥ 3 ∨ (j : Int) - (i : Int) ≥ 3) ∧
  (∀ i : ℕ, i ∈ anya_lost_games → i ≤ total_games) ∧
  (anya_lost_games.length = games_played Girl.Anya) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anya_lost_theorem_l113_11331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_equals_initial_l113_11350

/-- The initial position of the particle -/
def initial_position : ℂ := 6

/-- The rotation factor for each move -/
noncomputable def rotation_factor : ℂ := Complex.exp (Complex.I * (Real.pi / 6))

/-- The translation distance for each move -/
def translation_distance : ℝ := 12

/-- The number of moves -/
def num_moves : ℕ := 72

/-- The position after a single move -/
noncomputable def move (z : ℂ) : ℂ :=
  rotation_factor * z + translation_distance

/-- The final position after all moves -/
noncomputable def final_position : ℂ :=
  (move^[num_moves]) initial_position

/-- Theorem stating that the final position is equal to the initial position -/
theorem final_position_equals_initial :
  final_position = initial_position :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_equals_initial_l113_11350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l113_11353

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) - x

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := Real.log (x + 1)

theorem problem_solution :
  -- Part I: Prove that a = 1
  (∃! x, f 1 x = 0) ∧
  -- Part II: Prove that the maximum value of k is -1/2
  (∀ k, (∀ x > 0, f 1 x ≥ k * x^2) ↔ k ≤ -1/2) ∧
  -- Part III: Prove the inequality
  (∀ x₁ x₂, x₁ > -1 → x₂ > -1 → x₁ ≠ x₂ →
    (x₁ - x₂) / (h x₁ - h x₂) > Real.sqrt (x₁*x₂ + x₁ + x₂ + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l113_11353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l113_11349

theorem sample_size_calculation (freshmen sophomores juniors : ℕ) 
  (prob : ℝ) (n : ℕ) 
  (h1 : freshmen = 280) 
  (h2 : sophomores = 320) 
  (h3 : juniors = 400) 
  (h4 : prob = 0.2) 
  (h5 : n = Nat.floor (prob * (freshmen + sophomores + juniors : ℝ))) : 
  n = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l113_11349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AOB_l113_11359

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

/-- Theorem: Maximum area of triangle AOB -/
theorem max_area_triangle_AOB
  (a b c : ℝ)
  (h_ellipse : ∀ x y, ellipse_C x y a b)
  (h_eccentricity : eccentricity a c = Real.sqrt 3 / 3) :
  ∃ A B : ℝ × ℝ,
    A ∈ {p | ellipse_C p.1 p.2 a b} ∧
    B ∈ {p | ellipse_C p.1 p.2 a b} ∧
    (∃ l : ℝ → ℝ → Prop, 
      (l A.1 A.2 ∧ l B.1 B.2) ∧
      l (-c) 0 ∧
      (∀ X Y : ℝ × ℝ, l X.1 X.2 ∧ l Y.1 Y.2 → X.1 + Y.1 = -2*c)) ∧
    (∀ A' B' : ℝ × ℝ,
      A' ∈ {p | ellipse_C p.1 p.2 a b} →
      B' ∈ {p | ellipse_C p.1 p.2 a b} →
      (∃ l : ℝ → ℝ → Prop, 
        (l A'.1 A'.2 ∧ l B'.1 B'.2) ∧
        l (-c) 0 ∧
        (∀ X Y : ℝ × ℝ, l X.1 X.2 ∧ l Y.1 Y.2 → X.1 + Y.1 = -2*c)) →
      abs (A.1 * B.2 - A.2 * B.1) / 2 ≥ abs (A'.1 * B'.2 - A'.2 * B'.1) / 2) ∧
    abs (A.1 * B.2 - A.2 * B.1) / 2 = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AOB_l113_11359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l113_11372

theorem tan_double_angle_second_quadrant (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : π/2 < θ ∧ θ < π) :
  Real.tan (2 * θ) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l113_11372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenicek_equals_total_row_flow_l113_11383

/-- Represents the flow rate at a specific point in the water network -/
structure FlowRate where
  value : ℝ

/-- The initial flow rate at the input of the water network -/
def initialFlow : FlowRate :=
  ⟨1⟩

/-- Calculates the flow rate at a specific point given its row and position -/
def flowAtPoint (row : ℕ) (position : ℕ) : FlowRate :=
  sorry

/-- Calculates the sum of flow rates through Jeníček's highlighted points -/
def jenicekFlow : FlowRate :=
  let f1 := flowAtPoint 3 1
  let f2 := flowAtPoint 3 3
  let f3 := flowAtPoint 5 1
  let f4 := flowAtPoint 5 2
  ⟨f1.value + f2.value + f3.value + f4.value⟩

/-- Represents the total flow rate in any complete row of the network -/
def totalRowFlow : FlowRate :=
  initialFlow

/-- Theorem stating that Jeníček's flow equals the total flow in any row -/
theorem jenicek_equals_total_row_flow : jenicekFlow = totalRowFlow := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenicek_equals_total_row_flow_l113_11383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSmallDenominators_l113_11336

-- Define a fraction type
structure Fraction where
  numerator : Int
  denominator : Int
  denominator_positive : denominator > 0
  irreducible : Int.gcd numerator denominator = 1

-- Define the circle of fractions
def CircleFractions := Fin 5 → Fraction

-- Define a property for fractions with odd denominators > 10^10
def LargeDenominator (f : Fraction) : Prop :=
  f.denominator % 2 = 1 ∧ f.denominator > 10^10

-- Define the sum of two fractions
noncomputable def FractionSum (f1 f2 : Fraction) : Fraction :=
  sorry

-- Define a property for fractions with denominators < 100
def SmallDenominator (f : Fraction) : Prop :=
  f.denominator < 100

-- The main theorem
theorem impossibleSmallDenominators (circle : CircleFractions) 
  (h1 : ∀ i : Fin 5, LargeDenominator (circle i)) :
  ¬(∀ i : Fin 5, SmallDenominator (FractionSum (circle i) (circle ((i + 1) % 5)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSmallDenominators_l113_11336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_extended_data_l113_11358

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem variance_of_extended_data
  (a : List ℝ)
  (h_len : a.length = 10)
  (h_mean : a.sum / a.length = a.sum / 10)
  (h_var : variance a = 1.1)
  : variance (a ++ [a.sum / 10]) = 1.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_extended_data_l113_11358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_transformed_cosine_l113_11355

noncomputable def original_function (x : ℝ) : ℝ := Real.cos (2 * x)

noncomputable def shifted_function (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 10)

noncomputable def final_function (x : ℝ) : ℝ := Real.cos (4 * x - Real.pi / 10)

theorem symmetric_axis_of_transformed_cosine :
  ∃ (k : ℤ), (∀ x : ℝ, final_function (π / 40 + x) = final_function (π / 40 - x)) ∧ 
  (∀ y : ℝ, y ≠ π / 40 → ∃ x : ℝ, final_function (y + x) ≠ final_function (y - x)) :=
by
  sorry

#check symmetric_axis_of_transformed_cosine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_transformed_cosine_l113_11355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l113_11339

-- Define the complex number
noncomputable def z : ℂ := 5 / (2 + Complex.I)

-- Theorem statement
theorem point_in_fourth_quadrant :
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  -- Simplify the complex number
  have h1 : z = 2 - Complex.I := by
    sorry
  
  -- Check the real part
  have h_re : z.re = 2 := by
    sorry
  
  -- Check the imaginary part
  have h_im : z.im = -1 := by
    sorry
  
  -- Conclude that the point is in the fourth quadrant
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l113_11339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_long_distance_fare_l113_11323

/-- Taxi fare function for Xiaoshan taxis after December 25, 2011 -/
noncomputable def taxi_fare (x : ℝ) : ℝ :=
  if x ≤ 2 then 6
  else if x ≤ 6 then 6 + (x - 2) * 2.4
  else 6 + 4 * 2.4 + (x - 6) * 3.6

/-- Theorem: For distances greater than 6 km, the fare is 3.6x - 6 yuan -/
theorem long_distance_fare (x : ℝ) (h : x > 6) : taxi_fare x = 3.6 * x - 6 := by
  sorry

#check long_distance_fare

end NUMINAMATH_CALUDE_ERRORFEEDBACK_long_distance_fare_l113_11323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_line_x_equals_one_slope_angle_l113_11338

/-- Slope angle of a line -/
noncomputable def slope_angle (L : Set (ℝ × ℝ)) : ℝ := sorry

/-- The slope angle of a vertical line is 90 degrees -/
theorem slope_angle_vertical_line :
  ∀ (x : ℝ), (∀ y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 = x}) → slope_angle ({p : ℝ × ℝ | p.1 = x}) = 90 := by
  sorry

/-- The line x = 1 has a slope angle of 90 degrees -/
theorem line_x_equals_one_slope_angle :
  slope_angle ({p : ℝ × ℝ | p.1 = 1}) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_line_x_equals_one_slope_angle_l113_11338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_to_cover_expenses_l113_11354

-- Define constants
def babysitting_rate : ℚ := 20
def online_teaching_rate : ℚ := 100
def monthly_expenses : ℚ := 1200
def weekday_babysitting_hours : ℚ := 3
def tax_rate : ℚ := 15 / 100
def reduced_babysitting_rate : ℚ := 15
def reduced_rate_threshold : ℚ := 15

-- Define functions
noncomputable def saturday_babysitting_hours : ℚ := (2 + 5) / 2

noncomputable def weekly_earnings : ℚ :=
  weekday_babysitting_hours * 5 * babysitting_rate +
  saturday_babysitting_hours * reduced_babysitting_rate +
  online_teaching_rate

noncomputable def monthly_earnings : ℚ := weekly_earnings * 4

noncomputable def after_tax_monthly_earnings : ℚ := monthly_earnings * (1 - tax_rate)

noncomputable def after_tax_weekly_earnings : ℚ := after_tax_monthly_earnings / 4

noncomputable def weeks_to_reach_expenses : ℚ := monthly_expenses / after_tax_weekly_earnings

-- Theorem statement
theorem weeks_to_cover_expenses :
  3 < weeks_to_reach_expenses ∧ weeks_to_reach_expenses < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_to_cover_expenses_l113_11354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_radii_l113_11300

theorem intersecting_circles_radii 
  (O₁ O₂ : ℝ × ℝ) 
  (r R : ℝ) 
  (common_chord : ℝ × ℝ → ℝ × ℝ → ℝ) 
  (angle_at : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ) :
  let d := Real.sqrt 3 + 1
  (∃ A B : ℝ × ℝ, 
    common_chord A B = common_chord B A ∧ 
    angle_at O₁ A B = π / 2 ∧ 
    angle_at O₂ A B = π / 3 ∧
    Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) = d) →
  r = Real.sqrt 2 ∧ R = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_radii_l113_11300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l113_11329

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part1 (a m : ℝ) : 
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 := by sorry

-- Part 2
theorem part2 (t : ℝ) (h : t ≥ 0) :
  (∀ x, f 2 x + t ≥ f 2 (x + 2*t)) ↔ 
    (t = 0 ∧ ∀ x : ℝ, True) ∨ 
    (t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l113_11329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l113_11310

-- Define the points A, B, C, and D
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (0, 2)
def D : ℝ × ℝ := (3, 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the sum of distances function
noncomputable def sum_distances (P : ℝ × ℝ) : ℝ :=
  distance P A + distance P B + distance P C + distance P D

-- State the theorem
theorem min_sum_distances :
  ∀ P : ℝ × ℝ, sum_distances P ≥ sum_distances (4/3, 1) ∧
  sum_distances (4/3, 1) = 2 * Real.sqrt 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l113_11310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l113_11351

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * x^3 + 12 * y^3 + 50 * z^3 + 1 / (5 * x * y * z) ≥ 4 * Real.sqrt 3 ∧
  (8 * x^3 + 12 * y^3 + 50 * z^3 + 1 / (5 * x * y * z) = 4 * Real.sqrt 3 ↔
    x = 1 / (8 : ℝ)^(1/3) ∧ y = 1 / (12 : ℝ)^(1/3) ∧ z = 1 / (50 : ℝ)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l113_11351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_region_l113_11384

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
noncomputable def areaRegion (circleA circleB : Circle) : ℝ :=
  24 - 17.5 * Real.pi

/-- Theorem stating the area of the specific region -/
theorem area_specific_region :
  let circleA : Circle := ⟨(4, 4), 4⟩
  let circleB : Circle := ⟨(10, 4), 6⟩
  areaRegion circleA circleB = 24 - 17.5 * Real.pi := by
  sorry

#check area_specific_region

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_specific_region_l113_11384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_min_and_monotone_l113_11377

/-- The function f(x) defined as (x-1)^2 + b*ln(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + b * Real.log x

theorem f_local_min_and_monotone (b : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ IsLocalMin (f b) x) →
  (∀ (x : ℝ), x > 0 → StrictMono (f b)) →
  b = -4 ∧ b ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_local_min_and_monotone_l113_11377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_relationship_puzzle_l113_11395

/-- Represents a person in the family --/
inductive Person : Type
  | G1 : Person  -- Grandmother 1
  | G2 : Person  -- Grandmother 2
  | H1 : Person  -- Husband 1
  | H2 : Person  -- Husband 2
  | W1 : Person  -- Wife 1
  | W2 : Person  -- Wife 2

/-- Represents the relationships between people --/
inductive Relationship : Type
  | grandmother : Person → Person → Relationship
  | husband : Person → Person → Relationship
  | wife : Person → Person → Relationship
  | father : Person → Person → Relationship
  | daughter : Person → Person → Relationship
  | brother : Person → Person → Relationship
  | sister : Person → Person → Relationship
  | mother : Person → Person → Relationship
  | son : Person → Person → Relationship

/-- The family structure satisfies all conditions --/
def satisfies_conditions (family : List Person) (relations : List (Person × Person × Relationship)) : Prop :=
  (family.length = 6) ∧
  (∃ g1 g2 gd1 gd2, (g1, gd1, Relationship.grandmother g1 gd1) ∈ relations ∧ (g2, gd2, Relationship.grandmother g2 gd2) ∈ relations) ∧
  (∃ h1 h2 w1 w2, (h1, w1, Relationship.husband h1 w1) ∈ relations ∧ (h2, w2, Relationship.husband h2 w2) ∈ relations) ∧
  (∃ f1 f2 d1 d2, (f1, d1, Relationship.father f1 d1) ∈ relations ∧ (f2, d2, Relationship.father f2 d2) ∈ relations) ∧
  (∃ b1 b2 s1 s2, (b1, b2, Relationship.brother b1 b2) ∈ relations ∧ (s1, s2, Relationship.sister s1 s2) ∈ relations) ∧
  (∃ m1 m2 s1 s2, (m1, s1, Relationship.mother m1 s1) ∈ relations ∧ (m2, s2, Relationship.mother m2 s2) ∈ relations) ∧
  (∃ w1 w2 m1 m2, (w1, m1, Relationship.daughter w1 m1) ∈ relations ∧ (w2, m2, Relationship.daughter w2 m2) ∈ relations)

/-- Theorem stating that it's possible to have a family of 6 people satisfying all conditions --/
theorem family_relationship_puzzle :
  ∃ (family : List Person) (relations : List (Person × Person × Relationship)),
    satisfies_conditions family relations := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_relationship_puzzle_l113_11395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l113_11321

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2*x - 3*y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l113_11321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_E_votes_l113_11316

def total_registered_voters : ℕ := 20000
def urban_potential_voters : ℕ := 10000
def rural_potential_voters : ℕ := 10000
def urban_turnout : ℚ := 4/5
def rural_turnout : ℚ := 3/5
def invalid_vote_percentage : ℚ := 3/20
def candidate_A_percentage : ℚ := 2/5
def candidate_B_percentage : ℚ := 3/10
def candidate_C_percentage : ℚ := 3/20
def candidate_D_percentage : ℚ := 1/10

theorem candidate_E_votes :
  let total_votes : ℚ := urban_potential_voters * urban_turnout + rural_potential_voters * rural_turnout
  let valid_votes : ℚ := total_votes * (1 - invalid_vote_percentage)
  let candidate_E_votes : ℚ := valid_votes - (valid_votes * (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage + candidate_D_percentage))
  candidate_E_votes = 595 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_E_votes_l113_11316
