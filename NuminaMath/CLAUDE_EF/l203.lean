import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_intersection_l203_20331

/-- Represents a parabola with equation x^2 = 2py -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line with slope 1 passing through the focus of a parabola -/
def focus_line (par : Parabola) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | point.2 = point.1 + par.p / 2}

/-- Represents the area of the quadrilateral formed by the intersection points 
    and their x-axis projections -/
noncomputable def quadrilateral_area (par : Parabola) : ℝ :=
  3 * Real.sqrt 2 * par.p^2

theorem parabola_focus_line_intersection (par : Parabola) 
  (h_area : quadrilateral_area par = 12 * Real.sqrt 2) : 
  par.p = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_intersection_l203_20331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_angle_bisector_l203_20374

theorem right_triangle_hypotenuse_angle_bisector 
  (a l : ℝ) 
  (ha : a > 0) 
  (hl : l > 0) 
  (hl2 : l < a) :
  ∃ (b c : ℝ),
    b > 0 ∧ c > 0 ∧
    b^2 + c^2 = a^2 ∧
    l = (2 * b * c) / (b + c) := by
  sorry

#check right_triangle_hypotenuse_angle_bisector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_angle_bisector_l203_20374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l203_20373

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ m*x - y + b = 0) ∧
    (let (x₀, y₀) := point;
     y₀ = f x₀ ∧
     m = (deriv f) x₀ ∧
     y₀ = m*x₀ + b) ∧
    (m = 5 ∧ b = -2) :=
by sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l203_20373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l203_20381

/-- The eccentricity of an ellipse with equation x^2/(2m^2) + y^2/(m+1) = 1 
    and one focus on the line √2x - y + 2 = 0 is 2/3 -/
theorem ellipse_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), x^2 / (2*m^2) + y^2 / (m+1) = 1 ∧ Real.sqrt 2*x - y + 2 = 0) →
  (∃ (a b c : ℝ), a^2 = 2*m^2 ∧ b^2 = m+1 ∧ c = Real.sqrt 2 ∧ a^2 - b^2 = c^2) →
  (∃ (e : ℝ), e = c/a ∧ e = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l203_20381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_quartic_quadratic_is_quartic_l203_20395

/-- Represents a polynomial with real coefficients in one variable -/
abbrev RealPoly := MvPolynomial (Fin 1) ℝ

/-- A polynomial is quartic if its degree is at most 4 -/
def is_quartic (p : RealPoly) : Prop := p.totalDegree ≤ 4

/-- A polynomial is quadratic if its degree is at most 2 -/
def is_quadratic (p : RealPoly) : Prop := p.totalDegree ≤ 2

/-- Theorem: The sum of a quartic polynomial and a quadratic polynomial is a quartic expression -/
theorem sum_quartic_quadratic_is_quartic (p q : RealPoly) 
  (hp : is_quartic p) (hq : is_quadratic q) : 
  is_quartic (p + q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_quartic_quadratic_is_quartic_l203_20395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_ge_11_l203_20320

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem prime_factor_ge_11 (N : Nat) (h1 : N > 10) (h2 : all_digits_valid N) :
  ∃ p : Nat, Nat.Prime p ∧ p ≥ 11 ∧ p ∣ N :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_ge_11_l203_20320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_sum_and_divisibility_l203_20310

/-- Given a list of positive integers that sum to 65 and have a product divisible by 100,
    the maximum possible product is 12914016300. -/
theorem max_product_with_sum_and_divisibility :
  ∀ (l : List ℕ),
    (l.sum = 65) →
    (l.prod % 100 = 0) →
    (∀ p : ℕ, p ∈ l → p > 0) →
    l.prod ≤ 12914016300 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_with_sum_and_divisibility_l203_20310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_not_parallel_to_x_axis_l203_20347

/-- Given a function g(x) = x^2 - 2ln(x) + bx, where b is a real number,
    if x₁ and x₂ are two distinct positive zeros of g(x) with x₁ < x₂,
    then the derivative of g at the midpoint of x₁ and x₂ is not zero. -/
theorem tangent_not_parallel_to_x_axis (b : ℝ) (x₁ x₂ : ℝ) 
    (h₁ : 0 < x₁) (h₂ : x₁ < x₂)
    (h₃ : x₁^2 - 2*Real.log x₁ + b*x₁ = 0)
    (h₄ : x₂^2 - 2*Real.log x₂ + b*x₂ = 0) :
    let g := fun x : ℝ ↦ x^2 - 2*Real.log x + b*x
    let x₀ := (x₁ + x₂) / 2
    (deriv g) x₀ ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_not_parallel_to_x_axis_l203_20347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_points_perpendicular_to_plane_l203_20363

/-- A plane passing through two points and perpendicular to another plane -/
theorem plane_through_points_perpendicular_to_plane
  (p₁ p₂ n : ℝ × ℝ × ℝ)
  (d : ℝ)
  (h₁ : p₁ = (2, 0, -1))
  (h₂ : p₂ = (-2, 4, -1))
  (h₃ : n = (2, -1, 2))
  (h₄ : d = 4) :
  ∃ (A B C D : ℤ),
    (A : ℝ) * p₁.1 + (B : ℝ) * p₁.2.1 + (C : ℝ) * p₁.2.2 + (D : ℝ) = 0 ∧
    (A : ℝ) * p₂.1 + (B : ℝ) * p₂.2.1 + (C : ℝ) * p₂.2.2 + (D : ℝ) = 0 ∧
    ((A : ℝ) * n.1 + (B : ℝ) * n.2.1 + (C : ℝ) * n.2.2 = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 2 ∧ B = 2 ∧ C = 3 ∧ D = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_points_perpendicular_to_plane_l203_20363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l203_20369

/-- The distance between points A and B -/
def S : ℝ := sorry

/-- The speed of the first car -/
def v1 : ℝ := sorry

/-- The speed of the second car -/
def v2 : ℝ := sorry

/-- Time taken by the first car to travel half the distance -/
def t1 : ℝ := sorry

/-- Time taken by the second car to travel 26.25 km -/
def t2 : ℝ := sorry

/-- Time taken by the second car to travel half the distance -/
def t3 : ℝ := sorry

/-- Time taken by the first car to travel 31.2 km -/
def t4 : ℝ := sorry

theorem distance_between_points (h1 : S / 2 = v1 * t1)
                                 (h2 : 26.25 = v2 * t2)
                                 (h3 : S / 2 = v2 * t3)
                                 (h4 : 31.2 = v1 * t4)
                                 (h5 : (S / 2 + 2) / (S - S / 2 - 2) = (S - 2 - 26.25) / (S + 2 - 26.25))
                                 (h6 : (S - 2 - 26.25) / (S + 2 - 26.25) = (S - 2 - 31.2) / (S + 2 - 31.2)) :
  S = 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l203_20369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l203_20391

def a : ℕ → ℚ
  | 0 => 1/4  -- We define a(0) as 1/4 to match a₁ in the original problem
  | n + 1 => 1/2 * a n + 1/(2^(n+2))  -- Adjusted the exponent to match the original problem

theorem sequence_formula (n : ℕ) : a n = (2*n + 1) / 2^(n+2) := by
  induction n with
  | zero =>
    simp [a]
    -- The base case is trivial
    rfl
  | succ n ih =>
    simp [a]
    rw [ih]
    -- The inductive step requires algebraic manipulation
    sorry  -- We use sorry to skip the detailed proof

#eval a 0  -- Should output 1/4
#eval a 1  -- Should output 3/8
#eval a 2  -- Should output 5/16
#eval a 3  -- Should output 7/32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l203_20391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l203_20335

theorem problem_solution (y : ℝ) 
  (h : y + Real.sqrt (y^2 - 4) + 4 / (y - Real.sqrt (y^2 - 4)) = 18) :
  y^2 + Real.sqrt (y^4 - 4) + 16 / (y^2 + Real.sqrt (y^4 - 4)) = 
    3 * (256 / 2025 - Real.sqrt ((65536 - 16402500) / 4100625)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l203_20335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l203_20352

-- Define the parabola
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 4*x + 3

-- Define the vertex
def vertex : ℝ × ℝ := (4, -5)

-- Theorem statement
theorem parabola_vertex :
  (∀ x : ℝ, f x ≥ f (vertex.fst)) ∧ f (vertex.fst) = vertex.snd := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l203_20352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_assistant_usage_time_l203_20336

/-- Represents a barrel with two taps -/
structure Barrel where
  capacity : ℝ
  midway_tap_rate : ℝ
  bottom_tap_rate : ℝ

/-- Represents the scenario of the new assistant using the barrel -/
structure NewAssistantScenario (barrel : Barrel) where
  early_usage : ℝ
  assistant_usage : ℝ

/-- The theorem to prove -/
theorem new_assistant_usage_time 
  (barrel : Barrel)
  (scenario : NewAssistantScenario barrel)
  (h1 : barrel.capacity = 36)
  (h2 : barrel.midway_tap_rate = 1 / 6)
  (h3 : barrel.bottom_tap_rate = 1 / 4)
  (h4 : scenario.early_usage = 24)
  : scenario.assistant_usage = 48 := by
  sorry

#check new_assistant_usage_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_assistant_usage_time_l203_20336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_8_l203_20341

/-- Represents a car with specific driving and cooling characteristics -/
structure Car where
  speed : ℝ  -- Speed of the car in miles per hour
  drive_time : ℝ  -- Time the car can drive before needing to cool down (in hours)
  cool_time : ℝ  -- Time needed for the car to cool down (in hours)

/-- Calculates the total distance traveled by the car in a given time -/
noncomputable def total_distance (car : Car) (total_time : ℝ) : ℝ :=
  let cycle_time := car.drive_time + car.cool_time
  let full_cycles := ⌊total_time / cycle_time⌋
  let remaining_time := total_time - full_cycles * cycle_time
  let driving_time := full_cycles * car.drive_time + min remaining_time car.drive_time
  car.speed * driving_time

/-- Theorem stating that a car with the given characteristics drives at 8 miles per hour -/
theorem car_speed_is_8 : 
  ∃ (car : Car), car.drive_time = 5 ∧ car.cool_time = 1 ∧ 
  total_distance car 13 = 88 ∧ car.speed = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_8_l203_20341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_eq_ratio_l203_20348

open BigOperators

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

def fibonacci_product (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 2), 
    (((fibonacci (k + 3) + fibonacci (k + 5)) / fibonacci (k + 2) : ℚ) - 
     ((fibonacci (k + 3) + fibonacci (k + 1)) / fibonacci (k + 4) : ℚ))

theorem fibonacci_product_eq_ratio : 
  fibonacci_product 50 = (fibonacci 50 / fibonacci 51 : ℚ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_eq_ratio_l203_20348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l203_20371

theorem sine_function_value (f : ℝ → ℝ) (ω φ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : |φ| < π / 2)
  (h4 : ∀ x ∈ Set.Icc (π / 6) (2 * π / 3), 
    ∀ y ∈ Set.Icc (π / 6) (2 * π / 3), 
    x < y → f x > f y)
  (h5 : f (π / 6) = 1)
  (h6 : f (2 * π / 3) = -1) :
  f (π / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_value_l203_20371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_four_consecutive_even_numbers_l203_20307

theorem smallest_of_four_consecutive_even_numbers (x : ℤ) : 
  (∃ y z w : ℤ, 
    x % 2 = 0 ∧ 
    y = x + 2 ∧ 
    z = x + 4 ∧ 
    w = x + 6 ∧ 
    x + y + z + w = 140) → 
  x = 32 :=
by
  intro h
  cases' h with y h
  cases' h with z h
  cases' h with w h
  rcases h with ⟨heven, hy, hz, hw, hsum⟩
  have : 4 * x + 12 = 140 := by
    calc
      4 * x + 12 = x + (x + 2) + (x + 4) + (x + 6) := by ring
      _          = x + y + z + w := by rw [hy, hz, hw]
      _          = 140 := hsum
  have : 4 * x = 128 := by
    linarith
  have : x = 32 := by
    linarith
  exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_four_consecutive_even_numbers_l203_20307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_implies_a_range_l203_20385

-- Define the logarithmic function
noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log (a + 2)

-- State the theorem
theorem log_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → log_function a x₁ > log_function a x₂) →
  -2 < a ∧ a < -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_implies_a_range_l203_20385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_2_3_sin_2x_solution_cos_2x_solution_l203_20345

-- Define the general solution
noncomputable def general_solution (C₁ C₂ x : ℝ) : ℝ := C₁ * Real.sin (2 * x) + C₂ * Real.cos (2 * x)

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 4 * y x = 0

-- Theorem 1
theorem particular_solution_2_3 :
  differential_equation (λ x => general_solution 2 3 x) := by sorry

-- Theorem 2
theorem sin_2x_solution :
  ∀ x, general_solution 1 0 x = Real.sin (2 * x) := by sorry

-- Theorem 3
theorem cos_2x_solution :
  ∀ x, general_solution 0 1 x = Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_2_3_sin_2x_solution_cos_2x_solution_l203_20345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_extremum_condition_inequality_condition_l203_20302

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.log (x + 1)

-- Define the domain of f
def dom (x : ℝ) : Prop := x > -1

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a / (x + 1)

-- Theorem for part (1)
theorem tangent_line_at_zero (a : ℝ) :
  a = -2 →
  ∃ (m b : ℝ), ∀ (x y : ℝ), dom x →
    (y = m * x + b ↔ y = f a x + (f a 0 - f_deriv a 0 * 0) + f_deriv a 0 * (x - 0)) →
    m = -1 ∧ b = 1 :=
sorry

-- Theorem for part (2)
theorem extremum_condition (a : ℝ) :
  (∃ (x : ℝ), dom x ∧ (∀ (y : ℝ), dom y → f a x ≤ f a y ∨ f a x ≥ f a y)) ↔
  a < 0 :=
sorry

-- Theorem for part (3)
theorem inequality_condition (a : ℝ) :
  (∀ (x : ℝ), dom x → f a x ≥ 1 - Real.sin x) ↔
  a = -2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_extremum_condition_inequality_condition_l203_20302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l203_20354

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Define the line l passing through A and intersecting the circle
def line_intersects_circle (M N : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_eq M.1 M.2 ∧ circle_eq N.1 N.2 ∧
  (∃ (s : ℝ), M = (A.1 + t * (N.1 - A.1), A.2 + t * (N.2 - A.2)))

-- Define the chord length
noncomputable def chord_length (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Theorem statement
theorem min_chord_length :
  ∃ (M N : ℝ × ℝ), line_intersects_circle M N ∧
  (∀ (P Q : ℝ × ℝ), line_intersects_circle P Q →
    chord_length M N ≤ chord_length P Q) ∧
  chord_length M N = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l203_20354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_on_ellipse_l203_20346

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  a : ℝ
  b : ℝ
  ha : 0 < a
  hb : 0 < b

/-- A point on a 2D plane -/
structure Point (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  x : ℝ
  y : ℝ

/-- A line on a 2D plane defined by a point and an angle -/
structure Line (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  p : Point α
  angle : ℝ

/-- Theorem: Concyclic points on an ellipse -/
theorem concyclic_points_on_ellipse 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (E : Ellipse α) (P : Point α) (l₁ l₂ : Line α) 
  (A B C D : Point α) :
  (P.x^2 / E.a^2 + P.y^2 / E.b^2 ≠ 1) →  -- P is not on the ellipse
  (l₁.p = P ∧ l₂.p = P) →  -- l₁ and l₂ pass through P
  (l₁.angle + l₂.angle = π) →  -- Sum of angles is π
  (∃ t₁ t₂, A = Point.mk (P.x + t₁ * Real.cos l₁.angle) (P.y + t₁ * Real.sin l₁.angle) ∧
            B = Point.mk (P.x + t₂ * Real.cos l₁.angle) (P.y + t₂ * Real.sin l₁.angle) ∧
            A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1 ∧
            B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) →  -- A and B are on the ellipse and l₁
  (∃ p₁ p₂, C = Point.mk (P.x + p₁ * Real.cos l₂.angle) (P.y + p₁ * Real.sin l₂.angle) ∧
            D = Point.mk (P.x + p₂ * Real.cos l₂.angle) (P.y + p₂ * Real.sin l₂.angle) ∧
            C.x^2 / E.a^2 + C.y^2 / E.b^2 = 1 ∧
            D.x^2 / E.a^2 + D.y^2 / E.b^2 = 1) →  -- C and D are on the ellipse and l₂
  ∃ r : ℝ, r > 0 ∧ 
    (A.x - P.x)^2 + (A.y - P.y)^2 = r^2 ∧
    (B.x - P.x)^2 + (B.y - P.y)^2 = r^2 ∧
    (C.x - P.x)^2 + (C.y - P.y)^2 = r^2 ∧
    (D.x - P.x)^2 + (D.y - P.y)^2 = r^2  -- A, B, C, D are equidistant from P
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_on_ellipse_l203_20346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l203_20344

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b x_N x_M : ℝ) (A B M N : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  a > 0 →
  b > 0 →
  x_N ≠ 0 →
  x_N = 4 * x_M →
  l ≠ {(x, y) | y = 0} →
  N = (x_N, 0) →
  N ∈ l →
  (∀ (x y : ℝ), (x, y) ∈ {(x, y) | x^2 / a^2 - y^2 / b^2 = 1} ↔ (x, y) = A ∨ (x, y) = B) →
  (∀ (P : ℝ × ℝ), P ∈ l ↔ dist P A = dist P B) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  M.1 = x_M →
  (b^2 / a^2 + 1) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l203_20344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_intersection_k_l203_20359

/-- The line equation y = -x + 4 -/
noncomputable def line (x : ℝ) : ℝ := -x + 4

/-- The reciprocal function y = k/x -/
noncomputable def reciprocal (k : ℝ) (x : ℝ) : ℝ := k / x

/-- Point A where the line intersects the x-axis -/
def A : ℝ × ℝ := (4, 0)

/-- Point B where the line intersects the y-axis -/
def B : ℝ × ℝ := (0, 4)

/-- The length of AB -/
noncomputable def AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The intersection points C and D of the line and reciprocal function -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = line x ∧ p.2 = reciprocal k x ∧ x ≠ 0}

/-- The condition that k is positive -/
def k_positive (k : ℝ) : Prop := k > 0

/-- The condition that AB = √2 * CD -/
def length_condition (k : ℝ) : Prop :=
  ∀ C D, C ∈ intersection_points k → D ∈ intersection_points k → 
    AB = Real.sqrt 2 * Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)

theorem reciprocal_intersection_k (k : ℝ) :
  k_positive k → length_condition k → k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_intersection_k_l203_20359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_quiz_performance_l203_20305

/-- Represents the quiz performance and goal -/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  completed_as : ℕ

/-- Calculates the maximum number of remaining quizzes that can be scored lower than A -/
def max_lower_grade_quizzes (qp : QuizPerformance) : ℕ :=
  let remaining_quizzes := qp.total_quizzes - qp.completed_quizzes
  let required_as := Int.ceil (qp.goal_percentage * qp.total_quizzes) - qp.completed_as
  (remaining_quizzes - required_as).toNat.max 0

/-- Theorem stating that for the given conditions, the maximum number of remaining quizzes
    that can be scored lower than A is 0 -/
theorem tim_quiz_performance :
  let qp : QuizPerformance := {
    total_quizzes := 60,
    goal_percentage := 9/10,
    completed_quizzes := 25,
    completed_as := 18
  }
  max_lower_grade_quizzes qp = 0 := by
  sorry

#eval max_lower_grade_quizzes {
  total_quizzes := 60,
  goal_percentage := 9/10,
  completed_quizzes := 25,
  completed_as := 18
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_quiz_performance_l203_20305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l203_20323

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x

-- State the theorem about the derivative of f
theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 0 → deriv f x = 3*x^2 + 1/x^2 := by
  intro x hx
  have h1 : Differentiable ℝ f := sorry
  have h2 : deriv f x = deriv (fun x => x^3) x - deriv (fun x => 1/x) x := sorry
  have h3 : deriv (fun x => x^3) x = 3*x^2 := sorry
  have h4 : deriv (fun x => 1/x) x = -1/x^2 := sorry
  rw [h2, h3, h4]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l203_20323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_with_exponents_l203_20382

theorem simplify_fraction_with_exponents (x : ℝ) (h : x ≠ 0) :
  5 / (4 * x^4) * (4 * x^3) / 3 = 5 * x^7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_with_exponents_l203_20382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_hyperbola_eccentricity_l203_20394

/-- Given a hyperbola and a parabola with specific properties, prove the value of p -/
theorem parabola_focus_hyperbola_eccentricity (e : ℝ) :
  let hyperbola := λ x y : ℝ ↦ x^2 / 4 - y^2 / 12 = 1
  let parabola := λ x y p : ℝ ↦ x = 2 * p * y^2
  let hyperbola_eccentricity := e
  let parabola_focus := (e, 0)
  ∃ p : ℝ, p = 1/8 ∧ 
    (∀ x y : ℝ, hyperbola x y → ∃ p : ℝ, parabola x y p) ∧
    (parabola_focus.1 = hyperbola_eccentricity) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_hyperbola_eccentricity_l203_20394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_b_value_l203_20365

/-- The circle equation -/
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

/-- The line equation -/
def myLine (x y b : ℝ) : Prop := y = 2*x + b

/-- The center of the circle -/
def myCenter (x y : ℝ) : Prop := myCircle x y ∧ ∀ x' y', myCircle x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The line bisects the circle if it passes through the center -/
def myBisects (b : ℝ) : Prop := ∃ x y, myCenter x y ∧ myLine x y b

theorem bisecting_line_b_value :
  ∀ b : ℝ, myBisects b → b = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_b_value_l203_20365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_foci_form_square_l203_20361

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Condition that perpendicular lines through foci form a square -/
def foci_form_square (e : Ellipse) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ c^2 / e.a^2 + c^2 / e.b^2 = 1

/-- Theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity_when_foci_form_square (e : Ellipse) 
  (h : foci_form_square e) : eccentricity e = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_foci_form_square_l203_20361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mole_fraction_and_partial_pressure_l203_20398

-- Define the gas mixture
structure GasMixture where
  n_N2O5 : ℝ
  n_O2 : ℝ
  n_N2 : ℝ
  P : ℝ
  T : ℝ

-- Define mole fraction and partial pressure
noncomputable def mole_fraction (mixture : GasMixture) : ℝ :=
  mixture.n_N2O5 / (mixture.n_N2O5 + mixture.n_O2 + mixture.n_N2)

noncomputable def partial_pressure (mixture : GasMixture) : ℝ :=
  mole_fraction mixture * mixture.P

-- Theorem stating the relationship between mole fraction, partial pressure, and total pressure
theorem mole_fraction_and_partial_pressure (mixture : GasMixture) :
  (mole_fraction mixture = mixture.n_N2O5 / (mixture.n_N2O5 + mixture.n_O2 + mixture.n_N2)) ∧
  (partial_pressure mixture = mole_fraction mixture * mixture.P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mole_fraction_and_partial_pressure_l203_20398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_one_is_odd_l203_20362

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem f_plus_one_is_odd (ω φ : ℝ) (h_ω : ω > 0) (h_f_1 : f ω φ 1 = 0) :
  ∀ x, f ω φ (x + 1) = -f ω φ (-x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_one_is_odd_l203_20362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l203_20342

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem: The dot product of F₂P and F₂F₁ is 2 for a specific hyperbola -/
theorem hyperbola_dot_product (h : Hyperbola) (f1 f2 p : Point) (e : ℝ) :
  h.a = 1 →
  h.b = Real.sqrt 3 →
  onHyperbola h p →
  distance f1 f2 = 2 →
  (Real.sin (Real.arccos ((distance p f2)^2 + (distance f2 f1)^2 - (distance p f1)^2) / (2 * distance p f2 * distance f2 f1))) /
  (Real.sin (Real.arccos ((distance p f1)^2 + (distance f2 f1)^2 - (distance p f2)^2) / (2 * distance p f1 * distance f2 f1))) = e →
  dotProduct (Point.mk (p.x - f2.x) (p.y - f2.y)) (Point.mk (f1.x - f2.x) (f1.y - f2.y)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l203_20342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_approx_l203_20384

-- Define the parameters
variable (p q : ℝ)
variable (V : ℝ)

-- Define the ratios for each solution
def ratio1 (p : ℝ) : ℝ × ℝ := (2*p, 3)
def ratio2 (q : ℝ) : ℝ × ℝ := (q, 2)

-- Define the volumes
def vol1 (V : ℝ) : ℝ := V
def vol2 (V : ℝ) : ℝ := 2*V

-- Define the function to calculate the ratio of alcohol to water in the mixture
noncomputable def mixRatio (r1 r2 : ℝ × ℝ) (v1 v2 : ℝ) : ℝ :=
  (r1.1 * v1 / (r1.1 + r1.2) + r2.1 * v2 / (r2.1 + r2.2)) /
  (r1.2 * v1 / (r1.1 + r1.2) + r2.2 * v2 / (r2.1 + r2.2))

-- Theorem statement
theorem mixture_ratio_approx (p q : ℝ) (V : ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (mixRatio (ratio1 p) (ratio2 q) (vol1 V) (vol2 V) - (2*p + 2*q) / (3*p + q + 10)) < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_approx_l203_20384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_arithmetic_l203_20329

/-- In a country with unusual arithmetic, prove that if 1/6 of 20 equals 15, then 1/3 of 4 equals 10. -/
theorem unusual_arithmetic {country : Type} 
  (arithmetic : country → ℕ → ℕ → ℕ) 
  (c : country)
  (h : arithmetic c 6 20 = 15) : 
  arithmetic c 3 4 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unusual_arithmetic_l203_20329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_five_power_l203_20317

theorem factorial_five_power (n : ℕ) : 
  (∃ k : ℕ, Nat.factorial 20 = k * 5^n) ∧ 
  (∀ m : ℕ, Nat.factorial 20 ≠ m * 5^(n+1)) → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_five_power_l203_20317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_values_l203_20308

theorem fourth_quadrant_trig_values (α : Real) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- α is in the fourth quadrant
  (h2 : Real.tan α = -3) : 
  Real.sin α = -3 * Real.sqrt 10 / 10 ∧ Real.cos α = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_values_l203_20308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_par_value_per_hole_l203_20379

-- Define the given values
def rounds : ℕ := 9
def holes_per_round : ℕ := 18
def avg_strokes_per_hole : ℕ := 4
def strokes_over_par : ℕ := 9

-- Define the theorem
theorem par_value_per_hole :
  let total_holes : ℕ := rounds * holes_per_round
  let total_strokes : ℕ := total_holes * avg_strokes_per_hole
  let total_par : ℕ := total_strokes - strokes_over_par
  (total_par : ℚ) / total_holes = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_par_value_per_hole_l203_20379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptotes_of_hyperbola_with_sqrt2_eccentricity_l203_20389

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The distance from a point to a line in 2D space -/
noncomputable def distanceToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The main theorem: distance from (4,0) to asymptotes of hyperbola with eccentricity √2 is 2√2 -/
theorem distance_to_asymptotes_of_hyperbola_with_sqrt2_eccentricity
  (a b : ℝ) (h : Hyperbola a b) (h_ecc : eccentricity h = Real.sqrt 2) :
  ∃ d : ℝ, d = distanceToLine 4 0 1 (-1) 0 ∧ 
             d = distanceToLine 4 0 (-1) (-1) 0 ∧ 
             d = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptotes_of_hyperbola_with_sqrt2_eccentricity_l203_20389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l203_20312

-- Define the envelope structure
structure Envelope where
  length : ℚ
  height : ℚ

-- Define the function to check if an envelope needs extra postage
def needs_extra_postage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 3/2 || ratio > 3

-- Define the list of envelopes
def envelopes : List Envelope := [
  ⟨8, 5⟩,
  ⟨10, 2⟩,
  ⟨8, 8⟩,
  ⟨14, 5⟩
]

-- Theorem statement
theorem extra_postage_count :
  (envelopes.filter needs_extra_postage).length = 2 := by
  -- Proof goes here
  sorry

#eval (envelopes.filter needs_extra_postage).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l203_20312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_terms_l203_20375

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  a1_pos : a 1 > 0  -- First term is positive
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

theorem largest_sum_terms (seq : ArithmeticSequence) 
  (h : sum seq 9 = sum seq 12) :
  ∃ n : ℕ, (n = 10 ∨ n = 11) ∧ 
  (∀ m : ℕ, sum seq m ≤ sum seq n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_terms_l203_20375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l203_20399

-- Define the circle equation
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define point P
def P : ℝ × ℝ := (-1, 4)

-- Define the length of the tangent line
noncomputable def tangentLength (P : ℝ × ℝ) (circleEq : ℝ → ℝ → Prop) : ℝ :=
  let (px, py) := P
  let cx : ℝ := 2  -- x-coordinate of circle center
  let cy : ℝ := 3  -- y-coordinate of circle center
  let d := Real.sqrt ((cx - px)^2 + (cy - py)^2)
  let r : ℝ := 1  -- radius of the circle
  Real.sqrt (d^2 - r^2)

-- Theorem statement
theorem tangent_length_is_three :
  tangentLength P circleEq = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_three_l203_20399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l203_20327

/-- Given an arithmetic sequence with 2n+1 terms, where n is a natural number,
    if the sum of odd-numbered terms is 44 and the sum of even-numbered terms is 33,
    then the total number of terms is 7. -/
theorem arithmetic_sequence_terms (n : ℕ) : 
  (∃ (a : ℕ → ℚ), 
    -- The sequence has 2n+1 terms
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) ∧ 
    -- Sum of odd-numbered terms is 44
    ((Finset.range (n + 1)).sum (λ i ↦ a (2 * i + 1)) = 44) ∧
    -- Sum of even-numbered terms is 33
    ((Finset.range n).sum (λ i ↦ a (2 * i + 2)) = 33)) →
  -- The total number of terms is 7
  2 * n + 1 = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l203_20327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_neg_45_deg_l203_20311

theorem cot_neg_45_deg : Real.tan (-(π / 4)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_neg_45_deg_l203_20311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l203_20322

theorem min_abs_difference (a b : ℤ) (h : a * b - 8 * a + 7 * b = 600) :
  ∃ (c d : ℤ), c * d - 8 * c + 7 * d = 600 ∧
  ∀ (x y : ℤ), x * y - 8 * x + 7 * y = 600 →
  |c - d| ≤ |x - y| ∧ |c - d| = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l203_20322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l203_20376

/-- Given an angle θ in the fourth quadrant such that cos(2θ - 3π) = 7/25,
    prove that cos θ = 3/5, sin θ = -4/5, and the given trigonometric expression equals 8/9. -/
theorem trig_problem (θ : Real) 
    (h1 : Real.cos (2 * θ - 3 * Real.pi) = 7/25)
    (h2 : 3 * Real.pi / 2 < θ) (h3 : θ < 2 * Real.pi) : 
    Real.cos θ = 3/5 ∧ 
    Real.sin θ = -4/5 ∧ 
    (Real.cos (Real.pi/2 - θ) / (Real.tan θ * (Real.cos (Real.pi + θ) - 1)) + 
     Real.sin (θ - 3 * Real.pi/2) / (Real.tan (Real.pi - θ) * Real.cos (-θ))) = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l203_20376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bishop_king_probability_l203_20313

theorem bishop_king_probability : ∃ m n : ℕ, 
  m = 1333 ∧ 
  n = 2001000 ∧ 
  (m : ℚ) / n = (2 * (2000 * 2001 * 3998 / 6 + 1999 * 2000 * 3997 / 6)) / (2000^2 * (2000^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bishop_king_probability_l203_20313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_sum_is_14_l203_20303

/-- A rhombus with side length 5 and diagonals subject to certain constraints. -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  side_is_5 : side_length = 5
  diag1_not_longer_than_6 : diagonal1 ≤ 6
  diag2_not_shorter_than_6 : diagonal2 ≥ 6

/-- The maximum sum of the lengths of the two diagonals in the given rhombus. -/
noncomputable def max_diagonal_sum (r : Rhombus) : ℝ :=
  Real.sqrt ((2 * r.side_length) ^ 2 + (2 * r.side_length) ^ 2)

/-- Theorem stating that the maximum sum of diagonal lengths is 14. -/
theorem max_diagonal_sum_is_14 (r : Rhombus) :
  max_diagonal_sum r = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_sum_is_14_l203_20303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_property_l203_20390

theorem determinant_property (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 5 →
  Matrix.det !![p, 4*p + 2*q; r, 4*r + 2*s] = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_property_l203_20390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l203_20368

def a : Fin 2 → ℝ := ![3, -4]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_properties :
  (Real.sqrt (a 0 ^ 2 + a 1 ^ 2) = 5) ∧
  (Real.sqrt (b 0 ^ 2 + b 1 ^ 2) = Real.sqrt 5) ∧
  ((a 0 * b 0 + a 1 * b 1) / (Real.sqrt (a 0 ^ 2 + a 1 ^ 2) * Real.sqrt (b 0 ^ 2 + b 1 ^ 2)) = 2 * Real.sqrt 5 / 5) := by
  sorry

#eval a
#eval b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l203_20368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_percentage_l203_20325

/-- Calculates the gain percentage for a dishonest shopkeeper using a false weight --/
noncomputable def calculate_gain_percentage (true_weight : ℝ) (false_weight : ℝ) : ℝ :=
  (true_weight - false_weight) / false_weight * 100

/-- The gain percentage of a shopkeeper using a false weight of 970 grams instead of 1 kilogram --/
theorem shopkeeper_gain_percentage :
  let true_weight : ℝ := 1000
  let false_weight : ℝ := 970
  abs (calculate_gain_percentage true_weight false_weight - 3.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_percentage_l203_20325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l203_20349

-- Definition of a critical point
def CriticalPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  DifferentiableAt ℝ f x ∧ (deriv f x = 0 ∨ ¬DifferentiableAt ℝ f x)

-- Definition of monotonicity on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

-- Definition of extreme value
def HasExtremeValueOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ (∀ y, a < y ∧ y < b → f y ≤ f x ∨ f x ≤ f y)

-- Definition of a cubic function
def CubicFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

theorem function_properties :
  (∀ f : ℝ → ℝ, ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → deriv f x₀ = 0 → CriticalPoint f x₀) ∧
  (∀ f : ℝ → ℝ, ∀ a b : ℝ, MonotonicOn f a b → ¬HasExtremeValueOn f a b) ∧
  (∃ f : ℝ → ℝ, CubicFunction f ∧ ¬∃ x : ℝ, CriticalPoint f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l203_20349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_function_property_l203_20397

/-- Hook function -/
noncomputable def hook_function (a : ℝ) (x : ℝ) : ℝ := x + a / x

/-- The maximum value of f on [2, 4] -/
noncomputable def max_value (a : ℝ) : ℝ := max (hook_function a 2) (hook_function a 4)

/-- The minimum value of f on [2, 4] -/
noncomputable def min_value (a : ℝ) : ℝ := min (hook_function a 2) (hook_function a 4)

theorem hook_function_property (a : ℝ) (h1 : a > 0) :
  max_value a - min_value a = 1 → a = 4 ∨ a = 6 + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_function_property_l203_20397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_squares_l203_20334

theorem max_perfect_squares (a b : ℕ) : 
  let products := {a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), a * (a + 2), b * (b + 2)}
  (∃ (S : Finset ℕ), S ⊆ products ∧ (∀ x ∈ S, ∃ y : ℕ, x = y^2) ∧ S.card = 2) ∧
  (∀ (S : Finset ℕ), S ⊆ products → (∀ x ∈ S, ∃ y : ℕ, x = y^2) → S.card ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perfect_squares_l203_20334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l203_20367

/-- An arithmetic progression with an even number of terms -/
structure ArithmeticProgression where
  n : ℕ
  a : ℚ
  d : ℚ
  even_terms : Even n

/-- The sum of odd-numbered terms in the arithmetic progression -/
noncomputable def sum_odd_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2) * (ap.a + (ap.a + (ap.n - 2) * ap.d))

/-- The sum of even-numbered terms in the arithmetic progression -/
noncomputable def sum_even_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2) * ((ap.a + ap.d) + (ap.a + (ap.n - 1) * ap.d))

/-- The difference between the last and first terms -/
def last_first_diff (ap : ArithmeticProgression) : ℚ :=
  (ap.n - 1) * ap.d

/-- Theorem stating the conditions and conclusion of the problem -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_odd_terms ap = 30 ∧
  sum_even_terms ap = 40 ∧
  last_first_diff ap = 16 →
  ap.n = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l203_20367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_surface_area_l203_20353

/-- The total surface area of a solid composed of a hemisphere and a cylinder -/
noncomputable def total_surface_area (base_area : ℝ) : ℝ :=
  let r := (base_area / Real.pi).sqrt
  2 * Real.pi * r^2 + 2 * Real.pi * r^2

/-- Theorem stating the total surface area of the combined solid -/
theorem combined_solid_surface_area :
  total_surface_area (144 * Real.pi) = 576 * Real.pi :=
by
  -- Unfold the definition of total_surface_area
  unfold total_surface_area
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_surface_area_l203_20353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_monotonicity_l203_20324

noncomputable def m (x : ℝ) : ℝ × ℝ := (-2 * Real.sin (Real.pi - x), Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.sin (Real.pi / 2 - x))

noncomputable def f (x : ℝ) : ℝ := 1 - (m x).1 * (n x).1 - (m x).2 * (n x).2

theorem f_expression_and_monotonicity :
  ∀ x ∈ Set.Icc 0 Real.pi,
    f x = 2 * Real.sin (2 * x - Real.pi / 6) ∧
    (∀ y ∈ Set.Icc 0 (Real.pi / 3), x ≤ y → f x ≤ f y) ∧
    (∀ y ∈ Set.Icc (5 * Real.pi / 6) Real.pi, x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_and_monotonicity_l203_20324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l203_20396

/-- Given points A, B, and C in 3D space, this theorem states that A is equidistant from B and C. -/
theorem point_equidistant_from_two_points :
  let A : Fin 3 → ℝ := ![(-5), 0, 0]
  let B : Fin 3 → ℝ := ![(-2), 0, 6]
  let C : Fin 3 → ℝ := ![0, (-2), (-4)]
  (A 0 - B 0)^2 + (A 1 - B 1)^2 + (A 2 - B 2)^2 = 
  (A 0 - C 0)^2 + (A 1 - C 1)^2 + (A 2 - C 2)^2 := by
  -- Expand the squares
  simp
  -- Perform arithmetic
  ring

#check point_equidistant_from_two_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l203_20396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_power_f_max_value_l203_20314

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x^2 + 7*x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5*x + 10) / (x^2 + 5*x + 20)

-- State the theorems to be proved
theorem f_max_value : ∀ x : ℝ, f x ≤ 2 := by sorry

theorem g_power_f_max_value : ∀ x : ℝ, (g x) ^ (f x) ≤ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_power_f_max_value_l203_20314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_fraction_is_one_third_l203_20328

noncomputable def salary : ℝ := 8123.08
noncomputable def rent_fraction : ℝ := 1/4
noncomputable def clothes_fraction : ℝ := 1/5
noncomputable def amount_left : ℝ := 1760

noncomputable def food_fraction : ℝ := 1 - rent_fraction - clothes_fraction - amount_left / salary

theorem food_fraction_is_one_third :
  ∃ ε > 0, |food_fraction - 1/3| < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_fraction_is_one_third_l203_20328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_circle_radius_is_correct_l203_20392

/-- The radius of the circle to which seven congruent parabolas are tangent -/
noncomputable def circle_radius : ℝ := 1/4

/-- The equation of one of the parabolas -/
noncomputable def parabola_equation (x : ℝ) : ℝ := x^2 + circle_radius

/-- The line to which the parabola is tangent -/
noncomputable def tangent_line (x : ℝ) : ℝ := x

theorem parabola_circle_tangency :
  ∃! x : ℝ, parabola_equation x = tangent_line x :=
by sorry

theorem circle_radius_is_correct :
  circle_radius = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_circle_radius_is_correct_l203_20392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_difference_l203_20332

-- Define the operation using a different name to avoid conflict with 'at' keyword
def atOp (x y : ℤ) : ℤ := x * y + 3 * x - 4 * y

-- State the theorem
theorem at_difference : (atOp 7 5) - (atOp 5 7) = 14 := by
  -- Unfold the definition of atOp
  unfold atOp
  -- Perform the arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_difference_l203_20332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_adjustment_angle_l203_20321

/-- The number of minutes in a full rotation of a clock's minute hand. -/
def minutesPerRotation : ℕ := 60

/-- The number of degrees in a full rotation. -/
def degreesPerRotation : ℕ := 360

/-- The number of minutes the clock is adjusted backwards. -/
def adjustmentMinutes : ℕ := 10

/-- Converts an angle from degrees to radians. -/
noncomputable def degreesToRadians (degrees : ℝ) : ℝ :=
  degrees * (Real.pi / 180)

/-- Theorem: Adjusting a clock's minute hand backwards by 10 minutes results in an angle of π/3 radians. -/
theorem clock_adjustment_angle :
  degreesToRadians (adjustmentMinutes / minutesPerRotation * degreesPerRotation : ℝ) = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_adjustment_angle_l203_20321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_growth_l203_20300

/-- Given natural numbers a₁, a₂, ..., aₖ, S(n) represents the number of solutions
    in nonnegative integers to a₁x₁ + a₂x₂ + ... + aₖxₖ = n -/
def S (k : ℕ) (a : Fin k → ℕ) (n : ℕ) : ℕ :=
  sorry -- Placeholder definition

/-- For any k and a, S(n) is eventually nonzero -/
axiom S_eventually_nonzero (k : ℕ) (a : Fin k → ℕ) :
  ∃ N, ∀ n ≥ N, S k a n ≠ 0

theorem S_growth (k : ℕ) (a : Fin k → ℕ) :
  ∃ N, ∀ n ≥ N, S k a (n + 1) < 2 * S k a n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_growth_l203_20300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l203_20356

noncomputable def f (x y : ℝ) : ℝ := (x + y) / (⌊x⌋ * ⌊y⌋ + ⌊x⌋ + ⌊y⌋ + 1)

theorem range_of_f :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
  ∃ z : ℝ, f x y = z ∧ (z = 1/2 ∨ (5/6 ≤ z ∧ z < 5/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l203_20356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l203_20326

theorem midpoint_coordinate_sum :
  let p₁ : ℝ × ℝ := (5, 4)
  let p₂ : ℝ × ℝ := (11, 22)
  let m := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  m.1 + m.2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l203_20326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l203_20337

noncomputable def f (x : ℝ) : ℝ := Real.tan x ^ 2 + Real.tan x + 1

theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y ∧ ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2) ↔ y ≥ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l203_20337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lecture_arrangements_l203_20355

-- Define the number of lecturers
def n : ℕ := 7

-- Define the function to calculate the number of valid arrangements
def validArrangements (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * Nat.factorial (n - 3)

-- Theorem statement
theorem lecture_arrangements :
  validArrangements n = 2520 := by
  -- Unfold the definition of validArrangements
  unfold validArrangements
  -- Unfold the definition of n
  unfold n
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lecture_arrangements_l203_20355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_z_axis_l203_20387

-- Define points in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Theorem statement
theorem point_on_z_axis (z : ℝ) : 
  let A : Point3D := (1, 0, 2)
  let B : Point3D := (1, -3, 1)
  let P : Point3D := (0, 0, z)
  distance P A = distance P B → z = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_z_axis_l203_20387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_negative_two_b_l203_20377

-- Define the vectors
noncomputable def a : ℝ × ℝ := (4, 0)
noncomputable def b : ℝ × ℝ := (Real.cos (2 * Real.pi / 3), Real.sin (2 * Real.pi / 3))

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the projection vector
noncomputable def proj_vector (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product v w) / (magnitude w)^2
  (scalar * w.1, scalar * w.2)

-- Theorem statement
theorem projection_equals_negative_two_b :
  proj_vector a b = (-2 * b.1, -2 * b.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_negative_two_b_l203_20377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l203_20316

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r) ^ t - 1)

theorem principal_approximation (P : ℝ) (h_positive : P > 0) :
  let r : ℝ := 0.10
  let t : ℕ := 2
  let I : ℝ := 147.0000000000001
  compound_interest P r t = I →
  abs (P - 700) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l203_20316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_when_a_is_2_a_value_when_f_min_is_zero_l203_20370

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4^x - a * 2^(x+1) + 1

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 1 / (f a x)

-- Theorem for the range of g when a = 2
theorem range_of_g_when_a_is_2 :
  ∀ x ∈ Set.Icc 1 2, g 2 x ∈ Set.Icc (1/7) (1/3) := by sorry

-- Theorem for the value of a when minimum of f is 0
theorem a_value_when_f_min_is_zero :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = 0) ∧
  a = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_when_a_is_2_a_value_when_f_min_is_zero_l203_20370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l203_20306

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x - b) / (9 - x^2)

-- State the theorem
theorem odd_function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-3 : ℝ) 3 → f a b x = -f a b (-x)) →  -- f is odd on (-3, 3)
  f a b 1 = 1/8 →                                         -- f(1) = 1/8
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-3 : ℝ) 3 → g x = x / (9 - x^2)) ∧  -- g(x) = x / (9 - x^2)
    (∀ x, x ∈ Set.Ioo (-3 : ℝ) 3 → f a b x = g x) ∧        -- f(x) = g(x) on (-3, 3)
    (∀ x y, x ∈ Set.Ioo (-3 : ℝ) 3 → y ∈ Set.Ioo (-3 : ℝ) 3 → x < y → g x < g y) ∧  -- g is monotonically increasing
    (Set.Ioo (-2 : ℝ) (1/2) = {t | g (t-1) + g t < 0})) -- Solution set for g(t-1) + g(t) < 0
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l203_20306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l203_20304

/-- Definition of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a) ^ 2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_perpendicular_bisector : 
    ∃ (B : ℝ × ℝ), 
      B.1 > 0 ∧ 
      B.2 = 0 ∧ 
      (∃ (M : ℝ × ℝ), M.1 = B.1 / 2 ∧ M.2 = Real.sqrt 2 * h.b / 2)) :
  eccentricity h = Real.sqrt 10 / 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l203_20304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_sqrt11_l203_20383

-- Define the operation ¤
noncomputable def diamond (x y : ℝ) : ℝ := (2 * Real.sqrt x)^2 - (x - y)^2

-- Theorem statement
theorem diamond_sqrt11 : diamond (Real.sqrt 11) (Real.sqrt 11) = 44 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_sqrt11_l203_20383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l203_20343

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem symmetry_implies_phi (φ : ℝ) :
  (∀ x : ℝ, f x φ = f (π/4 - x) φ) →
  ∃ k : ℤ, φ = π/4 + k * π := by
  sorry

#check symmetry_implies_phi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l203_20343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_birds_theorem_l203_20301

/-- Represents the bird population in the Goshawk-Eurasian Nature Reserve -/
structure BirdPopulation where
  total : ℝ
  hawk_percent : ℝ
  paddyfield_warbler_percent_of_nonhawks : ℝ
  kingfisher_to_paddyfield_warbler_ratio : ℝ

/-- Calculates the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
noncomputable def other_birds_percent (pop : BirdPopulation) : ℝ :=
  let nonhawks := pop.total * (1 - pop.hawk_percent)
  let paddyfield_warblers := nonhawks * pop.paddyfield_warbler_percent_of_nonhawks
  let kingfishers := paddyfield_warblers * pop.kingfisher_to_paddyfield_warbler_ratio
  let other_birds := pop.total - (pop.total * pop.hawk_percent + paddyfield_warblers + kingfishers)
  (other_birds / pop.total) * 100

/-- Theorem stating that the percentage of other birds is 35% given the specified conditions -/
theorem other_birds_theorem (pop : BirdPopulation) 
  (h1 : pop.hawk_percent = 0.3)
  (h2 : pop.paddyfield_warbler_percent_of_nonhawks = 0.4)
  (h3 : pop.kingfisher_to_paddyfield_warbler_ratio = 0.25)
  (h4 : pop.total > 0) :
  other_birds_percent pop = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_birds_theorem_l203_20301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_is_1500_total_profit_is_12000_l203_20330

-- Define the given constants
def original_price : ℚ := 1800
def discount_rate : ℚ := 1/10
def profit_margin : ℚ := 2/25
def items_sold : ℕ := 100

-- Define the discounted price
noncomputable def discounted_price : ℚ := original_price * (1 - discount_rate)

-- Define the cost price
noncomputable def cost_price : ℚ := discounted_price / (1 + profit_margin)

-- Define the profit per item
noncomputable def profit_per_item : ℚ := discounted_price - cost_price

-- Define the total profit
noncomputable def total_profit : ℚ := profit_per_item * items_sold

-- Theorem statements
theorem cost_price_is_1500 : cost_price = 1500 := by sorry

theorem total_profit_is_12000 : total_profit = 12000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_is_1500_total_profit_is_12000_l203_20330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_is_26_l203_20309

theorem one_integer_is_26 
  (x y z w : ℤ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hxw : x ≠ w) (hyz : y ≠ z) (hyw : y ≠ w) (hzw : z ≠ w)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h1 : (x + y + z) / 3 + w = 34)
  (h2 : (x + y + w) / 3 + z = 22)
  (h3 : (x + z + w) / 3 + y = 26)
  (h4 : (y + z + w) / 3 + x = 18) :
  x = 26 ∨ y = 26 ∨ z = 26 ∨ w = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_is_26_l203_20309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersections_and_m_position_parabola_specific_case_l203_20380

-- Define the parabola
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the point M on the parabola
structure PointM (p q : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  on_parabola : y₀ = parabola p q x₀
  below_x_axis : y₀ < 0

-- Theorem statement
theorem parabola_intersections_and_m_position 
  (p q : ℝ) (M : PointM p q) :
  ∃ (x₁ x₂ : ℝ), 
    x₁ < x₂ ∧ 
    parabola p q x₁ = 0 ∧ 
    parabola p q x₂ = 0 ∧
    x₁ < M.x₀ ∧ 
    M.x₀ < x₂ := by sorry

-- Special case when M is (1, -1997)
theorem parabola_specific_case :
  ∃ (p q : ℝ),
    let M : PointM p q := {
      x₀ := 1,
      y₀ := -1997,
      on_parabola := by sorry,
      below_x_axis := by norm_num
    }
    ∃ (x₁ x₂ : ℤ),
      x₁ < x₂ ∧ 
      parabola p q ↑x₁ = 0 ∧ 
      parabola p q ↑x₂ = 0 ∧
      x₁ = 0 ∧ 
      x₂ = 1998 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersections_and_m_position_parabola_specific_case_l203_20380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_l203_20319

/-- The trajectory of point M given fixed points A and B -/
def trajectory (x y : ℝ) : Prop :=
  (x - 6)^2 + (y - 3)^2 = 4

/-- The line l -/
def line (x y : ℝ) : Prop :=
  y = x - 5

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem trajectory_and_intersection :
  ∀ (x y : ℝ),
    distance x y 2 3 = 2 * distance x y 5 3 →
    trajectory x y ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      trajectory x1 y1 ∧ trajectory x2 y2 ∧
      line x1 y1 ∧ line x2 y2 ∧
      distance x1 y1 x2 y2 = 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_intersection_l203_20319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l203_20339

/-- The area of a semicircle with diameter d -/
noncomputable def semicircle_area (d : ℝ) : ℝ := (Real.pi * d^2) / 8

/-- The diameter of the largest semicircle -/
def large_diameter : ℝ := 30

/-- The diameter of each smaller semicircle -/
def small_diameter : ℝ := 5

/-- The number of smaller semicircles -/
def num_small_semicircles : ℕ := 6

theorem shaded_area_calculation :
  semicircle_area large_diameter - num_small_semicircles * semicircle_area small_diameter = (375 / 4) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l203_20339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_theorem_l203_20333

theorem systematic_sampling_theorem (population : ℕ) (sample_size : ℕ) (first_draw : ℕ) 
  (range_start : ℕ) (range_end : ℕ) : 
  population = 960 →
  sample_size = 32 →
  first_draw = 9 →
  range_start = 450 →
  range_end = 750 →
  (∃ (selected : ℕ), 
    selected = (Finset.filter (λ n : ℕ ↦ 
      range_start ≤ (first_draw + (population / sample_size) * (n - 1)) ∧ 
      (first_draw + (population / sample_size) * (n - 1)) ≤ range_end) 
      (Finset.range (sample_size + 1))).card ∧
    selected = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_theorem_l203_20333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_factorial_difference_l203_20357

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastDigit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_factorial_difference : lastDigit (factorial 2014 - factorial 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_factorial_difference_l203_20357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glassious_ray_angle_change_l203_20350

noncomputable section

-- Define the angle θ as a function of time t
def θ (t : ℝ) : ℝ := t^2

-- Define the index of refraction for glass
def n : ℝ := 1.50

-- Define Snell's law
def snells_law (θ α : ℝ) : Prop := Real.sin (θ - α) = (2 / 3) * Real.sin θ

-- Theorem statement
theorem glassious_ray_angle_change 
  (t : ℝ) 
  (h : θ t = π / 6) :
  ∃ (α : ℝ → ℝ), 
    snells_law (θ t) (α t) ∧ 
    (deriv α t = (Real.sqrt π * (4 - Real.sqrt 6)) / (2 * Real.sqrt 6)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_glassious_ray_angle_change_l203_20350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_mult_eq_zero_l203_20358

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem scalar_mult_eq_zero (a : V) (l : ℝ) : l • a = 0 → l = 0 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_mult_eq_zero_l203_20358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_results_l203_20338

/-- Represents the survey data and problem conditions -/
structure SurveyData where
  total_sample : ℕ := 100
  models_in_sample : ℕ := 8
  total_school : ℕ := 1600
  boys_in_models : ℕ := 3
  girls_in_models : ℕ := 5

/-- Represents the random variable X (number of boys among selected models) -/
inductive X
  | zero : X
  | one : X
  | two : X
  | three : X

/-- The probability mass function of X -/
def pmf : X → ℚ
  | X.zero => 1/14
  | X.one => 3/7
  | X.two => 3/7
  | X.three => 1/14

/-- Theorem stating the main results of the problem -/
theorem survey_results (data : SurveyData) :
  -- 1. Estimated number of models in school
  (data.total_school * data.models_in_sample / data.total_sample : ℚ) = 128 ∧
  -- 2. Probability of event A
  (1 - (Nat.choose data.girls_in_models 4 : ℚ) / (Nat.choose (data.boys_in_models + data.girls_in_models) 4)) = 13/14 ∧
  -- 3. Probability distribution of X is correct
  (∀ x : X, pmf x ≥ 0) ∧ (pmf X.zero + pmf X.one + pmf X.two + pmf X.three = 1) ∧
  -- 4. Expected value of X
  (0 * pmf X.zero + 1 * pmf X.one + 2 * pmf X.two + 3 * pmf X.three : ℚ) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_results_l203_20338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_score_percentage_l203_20372

-- Define the cricket score components
def total_runs : ℕ := 138
def boundaries : ℕ := 12
def sixes : ℕ := 2

-- Define the runs per boundary and six
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

-- Calculate runs made by running
def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

-- Define the percentage calculation
noncomputable def percentage_runs_by_running : ℝ := (runs_by_running : ℝ) / (total_runs : ℝ) * 100

-- Theorem statement
theorem cricket_score_percentage :
  abs (percentage_runs_by_running - 56.52) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_score_percentage_l203_20372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l203_20351

/-- Given two non-collinear vectors a and b in a real vector space, 
    if λa + b is parallel to 2a + λb, then λ = √2 or λ = -√2 -/
theorem parallel_vectors_lambda (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] 
  (a b : V) (lambda : ℝ) (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_parallel : ∃ (k : ℝ), lambda • a + b = k • (2 • a + lambda • b)) :
  lambda = Real.sqrt 2 ∨ lambda = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l203_20351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triangle_side_length_is_80_l203_20388

/-- The side length of the first triangle in an infinite series of equilateral triangles -/
noncomputable def first_triangle_side_length (sum_of_perimeters : ℝ) : ℝ :=
  sum_of_perimeters / 6

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infinite_geometric_series_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

/-- Theorem: If the sum of perimeters of an infinite series of equilateral triangles
    (where each triangle is formed by joining the midpoints of the previous one)
    is 480 cm, then the side length of the first triangle is 80 cm. -/
theorem first_triangle_side_length_is_80 :
  first_triangle_side_length 480 = 80 := by
  sorry

/-- Lemma: The sum of perimeters of the infinite series of triangles
    is equal to the infinite geometric series sum with a = 3s and r = 1/2,
    where s is the side length of the first triangle. -/
lemma sum_of_perimeters_is_geometric_series (s : ℝ) :
  infinite_geometric_series_sum (3 * s) (1 / 2) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_triangle_side_length_is_80_l203_20388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_by_integer_factor_iff_only_first_two_nonzero_l203_20360

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Checks if a list of digits represents a valid natural number -/
def isValidNumber (d : Digits) : Prop :=
  d.length > 0 ∧ d.head! ≠ 0 ∧ d.all (· < 10)

/-- Converts a list of digits to a natural number -/
def digitsToNat (d : Digits) : Nat :=
  d.foldl (fun acc digit => acc * 10 + digit) 0

/-- Removes the third digit from a list of digits -/
def removeThirdDigit (d : Digits) : Digits :=
  if d.length < 3 then d else d.take 2 ++ d.drop 3

/-- Checks if a number decreases by an integer factor when its third digit is removed -/
def decreasesByIntegerFactor (d : Digits) : Prop :=
  ∃ m : Nat, m > 1 ∧ digitsToNat (removeThirdDigit d) * m = digitsToNat d

/-- Checks if a number has only its first two digits non-zero and the rest are zeros -/
def onlyFirstTwoNonZero (d : Digits) : Prop :=
  d.length ≥ 2 ∧ d.drop 2 = List.replicate (d.length - 2) 0

/-- The main theorem: A number decreases by an integer factor when its third digit is removed
    if and only if it has only its first two digits non-zero and the rest are zeros -/
theorem decrease_by_integer_factor_iff_only_first_two_nonzero (d : Digits) 
    (h : isValidNumber d) :
    decreasesByIntegerFactor d ↔ onlyFirstTwoNonZero d :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_by_integer_factor_iff_only_first_two_nonzero_l203_20360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_sin_l203_20378

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => fun x => Real.cos x
  | (n + 1) => fun x => deriv (f n) x

theorem f_2016_is_sin : f 2016 = fun x => Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_sin_l203_20378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sequence_not_standard_type_l203_20340

/-- Given a, b, d in harmonic progression with 1 < a < b < d, and n > 1 an integer,
    the sequence log_a n, log_b n, log_d n is not a standard sequence type. -/
theorem log_sequence_not_standard_type (a b d n : ℝ) (hn : n > 1) (ha : 1 < a) (hb : a < b) (hd : b < d)
  (h_harmonic : (1 / b) = (1 / 2) * ((1 / a) + (1 / d))) :
  let seq := [Real.log n / Real.log a, Real.log n / Real.log b, Real.log n / Real.log d]
  ¬ (∃ r : ℝ, seq.zipWith (· / ·) (seq.tail) = List.replicate 2 r) ∧  -- Not G.P.
  ¬ (∃ d : ℝ, seq.zipWith (· - ·) (seq.tail) = List.replicate 2 d) ∧  -- Not A.P.
  ¬ (∃ r : ℝ, (seq.map (1 / ·)).zipWith (· / ·) ((seq.map (1 / ·)).tail) = List.replicate 2 r) ∧  -- Reciprocals not G.P.
  ¬ (∃ d : ℝ, (seq.map (1 / ·)).zipWith (· - ·) ((seq.map (1 / ·)).tail) = List.replicate 2 d)  -- Reciprocals not A.P.
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sequence_not_standard_type_l203_20340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l203_20318

/-- Predicate to determine if a point is the focus of a parabola -/
def is_focus (p : ℝ × ℝ) (f : ℝ → ℝ → Prop) : Prop := sorry

/-- Given a parabola with equation y = (1/8)x^2, its focus has coordinates (0, 2) -/
theorem parabola_focus_coordinates :
  ∃ p : ℝ × ℝ, p = (0, 2) ∧ is_focus p (fun x y ↦ y - (1/8) * x^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l203_20318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_area_represents_frequency_l203_20386

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  groups : ℕ  -- number of groups
  total_sample_size : ℕ  -- total sample size
  group_frequencies : Fin groups → ℕ  -- frequency of each group
  group_areas : Fin groups → ℝ  -- area of each rectangle

/-- The area of each rectangle in a frequency histogram represents the frequency of the corresponding group --/
theorem frequency_histogram_area_represents_frequency 
  (h : FrequencyHistogram) 
  (i : Fin h.groups) : 
  h.group_areas i = (h.group_frequencies i : ℝ) / h.total_sample_size :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_histogram_area_represents_frequency_l203_20386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_zero_l203_20393

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := x^2 + 1
def h (x : ℝ) : ℝ := |x|
noncomputable def k (x : ℝ) : ℝ := 2^x

-- Define what it means for a function to attain an extreme value at x=0
def attains_extreme_at_zero (f : ℝ → ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f 0 ≤ f x ∨ f 0 ≥ f x

-- State the theorem
theorem extreme_value_at_zero :
  ¬(attains_extreme_at_zero f) ∧
  (attains_extreme_at_zero g) ∧
  (attains_extreme_at_zero h) ∧
  ¬(attains_extreme_at_zero k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_zero_l203_20393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_over_sum_equals_1120_l203_20366

theorem product_over_sum_equals_1120 : (Nat.factorial 8) / (List.sum (List.range 9)) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_over_sum_equals_1120_l203_20366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l203_20364

/-- Given a function f where the tangent line at x = 5 has the equation y = -x + 8,
    prove that f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (fun x => f 5 + deriv f 5 * (x - 5)) = fun x => -x + 8) : 
    f 5 + deriv f 5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l203_20364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reconstruct_triangle_l203_20315

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Triangle type -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point is on a circle -/
def is_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point is on the angle bisector -/
def is_on_angle_bisector (p : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is on a line defined by two points -/
def is_on_line (p : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  sorry

/-- Calculate the center of the circumcircle of a triangle -/
noncomputable def center_of_circumcircle (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Given a circle Ω and points W and O, we can uniquely determine triangle ABC -/
theorem reconstruct_triangle (Ω : Circle) (W O : ℝ × ℝ) : 
  ∃! (ABC : Triangle),
    -- Ω is the circumscribed circle of triangle ABC
    (is_on_circle ABC.A Ω) ∧ (is_on_circle ABC.B Ω) ∧ (is_on_circle ABC.C Ω) ∧
    -- W is on the angle bisector of ∠A and on circle Ω
    (is_on_circle W Ω) ∧ (is_on_angle_bisector W ABC.A ABC.B ABC.C) ∧
    -- O is the center of the circumscribed circle of triangle ACL
    (∃ L : ℝ × ℝ, 
      is_on_angle_bisector L ABC.A ABC.B ABC.C ∧
      is_on_line L ABC.B ABC.C ∧
      O = center_of_circumcircle ABC.A ABC.C L) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reconstruct_triangle_l203_20315
